/* Entry-method and idle hooks for the +nvtx and +gputrace timeline tools; see cknvtx.h for the flags.
 * Every entry-method delivery (CkDeliverMessageFree / CkDeliverMessageReadonly) is bracketed. With +nvtx
 * that is an NVTX range named "Chare::entry" (a string registered once per entry index, per PE), coloured
 * by chare type, with the array element's index as the integer payload: x for 1D, x*1000+y for 2D,
 * x*1000000+y*1000+z for 3D. With +gputrace <dir> the same bracket is a text record in this PE's file:
 *   E start end ep idx      entry method (CUPTI clock, ns)      P ep Chare::entry   (name table)
 *   I start end             scheduler idle                      O objId idx         (element id -> index)
 * The scheduler's BEGIN_IDLE / END_IDLE conditions bracket the idle ranges, so a PE row reads as busy
 * (entry ranges), idle, or runtime (neither: converse handlers, HAPI polling, message progress).
 * Under Nsight Systems the NVTX ranges sit on each PE thread's row and are projected onto the GPU rows
 * under the kernels launched inside them. */
#include "ck.h"
#include "cknvtx.h"
#if CMK_NVTX
#include "hapi.h"
#include "nvtx3/nvToolsExt.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>
#include <unistd.h>
#include <sys/stat.h>

extern bool hapiNvtxOn __attribute__((weak));   // hybridAPI/hapi_nvtx.h
std::atomic<bool> _nvtxOn{false};
static bool _nvtxRanges = false;                // +nvtx
static bool _nvtxIdleOn = true;                 // not +nvtxnoidle
static const char* _gpuTraceDir = nullptr;      // +gputrace <dir>
static nvtxDomainHandle_t _nvtxDomain = nullptr;
static nvtxStringHandle_t _nvtxIdleStr = nullptr;

// Per PE (one thread per PE).
static thread_local std::vector<nvtxStringHandle_t> t_epStr;   // registered NVTX string per entry index
static thread_local bool t_inIdle = false;
static thread_local FILE* t_tr = nullptr;                       // this PE's +gputrace file
static thread_local bool t_trFailed = false;
static thread_local std::vector<char> t_epNamed;                // P line written for this entry index
static thread_local std::unordered_set<CmiUInt8> t_objNamed;    // O line written for this element
struct TraceFrame { uint64_t start; int ep; long long idx; };
static thread_local std::vector<TraceFrame> t_stack;            // open entry brackets (deliveries can nest)
static thread_local uint64_t t_idleStart = 0;

static const uint32_t kPalette[] = {
  0xff1abc9c, 0xff3498db, 0xff9b59b6, 0xffe67e22, 0xffe74c3c, 0xff2ecc71,
  0xfff1c40f, 0xff2980b9, 0xff8e44ad, 0xffd35400, 0xff16a085, 0xffc0392b };
static const uint32_t kIdleColor = 0xff95a5a6;

// "Chare::entry", without the parameter list charmxi appends to the entry name.
static std::string epName(int ep) {
  const EntryInfo *e = _entryTable[ep];
  const char *chare = (e->chareIdx >= 0 && (size_t)e->chareIdx < _chareTable.size()) ? _chareTable[e->chareIdx]->name : "?";
  std::string s = std::string(chare) + "::" + (e->name ? e->name : "?");
  const size_t paren = s.find('(');
  if (paren != std::string::npos) s.resize(paren);
  return s;
}

static nvtxStringHandle_t epString(int ep) {
  if ((size_t)ep >= t_epStr.size()) t_epStr.resize(std::max(_entryTable.size(), (size_t)ep + 1), nullptr);
  nvtxStringHandle_t &h = t_epStr[ep];
  if (h == nullptr) h = nvtxDomainRegisterStringA(_nvtxDomain, epName(ep).c_str());
  return h;
}

static long long packIndex(const CkArrayIndex &idx) {
  const int nd = idx.getDimension();
  long long v = 0;
  if (nd <= 3) { const int *d = idx.data(); for (int i = 0; i < nd; i++) v = v * 1000 + d[i]; }
  else { const short *d = idx.shortData(); for (int i = 0; i < nd && i < 4; i++) v = v * 1000 + d[i]; }
  return v;
}

static void attrInit(nvtxEventAttributes_t &a, nvtxStringHandle_t str, uint32_t color) {
  memset(&a, 0, sizeof a);
  a.version = NVTX_VERSION;
  a.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  a.colorType = NVTX_COLOR_ARGB;
  a.color = color;
  a.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
  a.message.registered = str;
}

static FILE* traceFile() {
  if (t_tr != nullptr || t_trFailed) return t_tr;
  char host[256] = "?";
  gethostname(host, sizeof host - 1); host[sizeof host - 1] = 0;
  if (char* dot = strchr(host, '.')) *dot = 0;
  char path[1024];
  snprintf(path, sizeof path, "%s/gputrace.%s.%d.pe%d", _gpuTraceDir, host, (int)getpid(), CkMyPe());
  t_tr = fopen(path, "w");
  if (t_tr == nullptr) {
    CmiPrintf("Charm++[pe=%d]: +gputrace: cannot open %s: %s\n", CkMyPe(), path, strerror(errno));
    t_trFailed = true;
    return nullptr;
  }
  setvbuf(t_tr, nullptr, _IOFBF, 1 << 20);
  fprintf(t_tr, "# gputrace v1 pe=%d pid=%d host=%s\n", CkMyPe(), (int)getpid(), host);
  return t_tr;
}

static void idleEnd() {
  if (!t_inIdle) return;
  t_inIdle = false;
  if (_nvtxRanges) nvtxDomainRangePop(_nvtxDomain);
  if (_gpuTraceDir != nullptr && t_idleStart != 0) {
    const uint64_t end = hapiTraceTimestamp();
    if (FILE* f = traceFile()) fprintf(f, "I %llu %llu\n", (unsigned long long)t_idleStart, (unsigned long long)end);
    t_idleStart = 0;
  }
}

void _nvtxBeginEntry(int ep, const void *msg, void *obj) {
  idleEnd();   // END_IDLE normally did this
  const bool arrayElt = (msg != nullptr && obj != nullptr && UsrToEnv(msg)->getMsgtype() == ForArrayEltMsg);
  const long long idx = arrayElt ? packIndex(((CkMigratable *)obj)->thisIndexMax) : -1;
  if (_nvtxRanges) {
    nvtxEventAttributes_t a;
    attrInit(a, epString(ep), kPalette[(unsigned)_entryTable[ep]->chareIdx % (sizeof kPalette / sizeof kPalette[0])]);
    if (arrayElt) { a.payloadType = NVTX_PAYLOAD_TYPE_INT64; a.payload.llValue = idx; }
    nvtxDomainRangePushEx(_nvtxDomain, &a);
  }
  if (_gpuTraceDir != nullptr) {
    if (FILE* f = traceFile()) {
      if ((size_t)ep >= t_epNamed.size()) t_epNamed.resize(std::max(_entryTable.size(), (size_t)ep + 1), 0);
      if (!t_epNamed[ep]) { t_epNamed[ep] = 1; fprintf(f, "P %d %s\n", ep, epName(ep).c_str()); }
      if (arrayElt) {
        const CmiUInt8 id = ((CkMigratable *)obj)->ckGetID();
        if (t_objNamed.insert(id).second) fprintf(f, "O %llu %lld\n", (unsigned long long)id, idx);
      }
    }
    t_stack.push_back({hapiTraceTimestamp(), ep, idx});
  }
}

void _nvtxEndEntry() {
  if (_nvtxRanges) nvtxDomainRangePop(_nvtxDomain);
  if (_gpuTraceDir != nullptr && !t_stack.empty()) {
    const TraceFrame fr = t_stack.back();
    t_stack.pop_back();
    const uint64_t end = hapiTraceTimestamp();
    if (FILE* f = traceFile())
      fprintf(f, "E %llu %llu %d %lld\n", (unsigned long long)fr.start, (unsigned long long)end, fr.ep, fr.idx);
  }
}

static void nvtxIdleBegin(void *) {
  if (t_inIdle) return;
  t_inIdle = true;
  if (_nvtxRanges) {
    nvtxEventAttributes_t a;
    attrInit(a, _nvtxIdleStr, kIdleColor);
    nvtxDomainRangePushEx(_nvtxDomain, &a);
  }
  if (_gpuTraceDir != nullptr) t_idleStart = hapiTraceTimestamp();
}
static void nvtxIdleEnd(void *) { idleEnd(); }

void _nvtxInit(char **argv) {
  const bool nvtx = CmiGetArgFlagDesc(argv, "+nvtx", "NVTX ranges around entry methods and PE idle time (profile with nsys -t cuda,nvtx)");
  const bool noIdle = CmiGetArgFlagDesc(argv, "+nvtxnoidle", "+nvtx / +gputrace without the idle ranges");
  char *dir = nullptr;
  const bool trace = CmiGetArgStringDesc(argv, "+gputrace", &dir, "write the runtime's GPU timeline (CUPTI kernels, entry methods, idle) to this directory; gputrace_merge.py builds the Perfetto trace");
  if (!nvtx && !trace) return;
  _nvtxIdleOn = !noIdle;
  if (nvtx) {
    _nvtxRanges = true;
    _nvtxDomain = nvtxDomainCreateA("charm");
    _nvtxIdleStr = nvtxDomainRegisterStringA(_nvtxDomain, "idle");
    if (&hapiNvtxOn != nullptr) hapiNvtxOn = true;   // the HAPI layer's own ranges (weak: absent in non-HAPI links)
  }
  if (trace) {
    _gpuTraceDir = strdup(dir);
    mkdir(_gpuTraceDir, 0775);
  }
  _nvtxOn.store(true);
  if (CkMyPe() == 0) {
    if (nvtx) CmiPrintf("Charm++> NVTX ranges on: entry methods%s, domain \"charm\"\n", noIdle ? "" : " and PE idle time");
    if (trace) CmiPrintf("Charm++> GPU timeline trace on: entry methods%s and CUPTI kernels to %s\n", noIdle ? "" : ", PE idle time", _gpuTraceDir);
  }
}

void _nvtxInitPE() {
  if (!_nvtxOn.load() || !_nvtxIdleOn) return;
  CcdCallOnConditionKeep(CcdPROCESSOR_BEGIN_IDLE, nvtxIdleBegin, nullptr);
  CcdCallOnConditionKeep(CcdPROCESSOR_END_IDLE, nvtxIdleEnd, nullptr);
}

void _nvtxStartTrace() {
  if (_gpuTraceDir != nullptr && CmiMyRank() == 0) hapiTraceStart(_gpuTraceDir);
}

void _nvtxExit() {
  if (t_tr != nullptr) { fclose(t_tr); t_tr = nullptr; }
}
#endif
