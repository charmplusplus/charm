// nokeepaudit: count how many distinct message buffers a broadcast delivers,
// system-wide, for each entry-method kind (marshalled, fixed message, varsize
// message, custom pack/unpack message), with and without [nokeep], to a group
// and to a chare array with perPe elements per PE.
//
// How a buffer instance is identified. Every payload carries a 64-bit marker
// word, zero when the sender builds the message. The first receiver to see a
// zero marker in a given buffer stamps it (atomic CAS) with a fresh id that is
// unique across the job; later receivers of the same buffer read that id. A
// receiver reports (process, buffer address, id). Two deliveries are the same
// buffer instance iff all three agree:
//   - the id separates two copies that the allocator placed at the same
//     address one after the other (free, then malloc returns the same block);
//   - the address separates a copy made AFTER the source was stamped (the copy
//     inherits the id but lives elsewhere).
// Receivers that own their message (non-nokeep message entries) also hold it
// until the next test, so its block cannot be recycled meanwhile.
//
// The marker write violates the nokeep "do not modify" rule on purpose; it is
// the same probe tests/charm++/within_node_bcast uses (an atomic in the
// message), and it is idempotent across receivers.
//
// Usage: ./nokeepaudit [perPe=10] [payloadWords=1024] [verbose=0] [withUnsafe=0]
//                      [stressBcasts=200] [stressVarOnly=0]
//
// After the table, a stress phase sends stressBcasts keep broadcasts back to
// back (custom-packed and varsize alternately) to the array while every element
// migrates to the next PE after each odd one, so some deliveries come from the
// broadcaster's stored copy of an earlier broadcast. Each element checks that it
// sees every broadcast once, in order, with an intact payload. 0 skips it.
// stressVarOnly=1 sends only the varsize message (no custom pack/unpack).
//
// withUnsafe (bitmask) also runs two cases that corrupt the heap (found by this
// audit under ASan), so they are OFF by default:
//  - bit 1 ([threaded, nokeep] marshalled) and bit 4 (the variants that
//    suspend before reading their data, marshalled and message):
//    before the charmxi fix, the threaded wrapper deletes the message when the
//    thread ends (as for any threaded marshalled entry) while the nokeep
//    registration makes the runtime share the same buffer with every local
//    receiver and drop its own reference at the thread's first suspend:
//    heap-use-after-free in _callthr_marshTNK_*. Safe with the fix (the thread
//    takes a reference of its own, CkReferenceMsg, released at thread end).
//  - bit 2, array broadcast of a custom-packed message to a keep entry with >1 element
//    per PE: CkDeliverMessageReadonly -> CkCopyMsg calls pack() on the source,
//    which (manual idiom) deletes it; CkCopyMsg returns the re-unpacked source
//    through *pMsg, but the caller discards it (only an error-checking build
//    aborts, "pack/unpack changed message pointer"), and the broadcaster keeps
//    delivering the freed pointer: heap-use-after-free in PackMsg::pack.

#include <atomic>
#include <cinttypes>
#include <cstring>
#include <map>
#include <set>
#include <tuple>
#include <vector>

#include "charm++.h"
#include "register.h"  // _entryTable[ep]->noKeep: what the runtime registered

struct FixedMsg;
struct VarMsg;
struct PackMsg;
#include "nokeepaudit.decl.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int payloadWords;
/*readonly*/ int stressBcasts;

static const int kFixedWords = 1024;  // FixedMsg payload, independent of argv

struct FixedMsg : public CMessage_FixedMsg {
  uint64_t data[kFixedWords];  // data[0] is the marker
};

struct VarMsg : public CMessage_VarMsg {
  uint64_t* data;  // data[0] is the marker
  int n;
  int seq = -1;    // stress phase only
};

// Custom-packed message in the manual's idiom: the payload lives in a
// separately allocated array; pack serializes it and deletes the message,
// unpack allocates a fresh message with CkAllocBuffer and frees the buffer.
struct PackMsg : public CMessage_PackMsg {
  uint64_t marker = 0;
  int n = 0;
  int seq = -1;    // stress phase only
  uint64_t* data = nullptr;
  PackMsg() {}
  explicit PackMsg(int n_) : n(n_), data(new uint64_t[n_]) {}
  ~PackMsg() { delete[] data; }

  static void* pack(PackMsg* m) {
    size_t sz = 3 * sizeof(uint64_t) + (size_t)m->n * sizeof(uint64_t);
    char* buf = (char*)CkAllocBuffer(m, (int)sz);
    uint64_t* w = (uint64_t*)buf;
    w[0] = m->marker;
    w[1] = (uint64_t)m->n;
    w[2] = (uint64_t)(int64_t)m->seq;
    memcpy(w + 3, m->data, (size_t)m->n * sizeof(uint64_t));
    delete m;
    return buf;
  }
  static PackMsg* unpack(void* buf) {
    uint64_t* w = (uint64_t*)buf;
    PackMsg* m = (PackMsg*)CkAllocBuffer(buf, sizeof(PackMsg));
    m = new ((void*)m) PackMsg();
    m->marker = w[0];
    m->n = (int)w[1];
    m->seq = (int)(int64_t)w[2];
    m->data = new uint64_t[m->n];
    memcpy(m->data, w + 3, (size_t)m->n * sizeof(uint64_t));
    CkFreeMsg(buf);
    return m;
  }
};

// One record per delivery. Fixed layout, contributed with CkReduction::concat.
struct Rec {
  int32_t pe, node, physnode, elem;
  int32_t envNokeep;  // CMI_MSG_NOKEEP of the delivered envelope, -1 = n/a
  int32_t payloadOk;
  uint64_t addr;      // message pointer (message kinds) / data pointer (marshalled)
  uint64_t dataAddr;  // payload pointer
  uint64_t id;        // marker id after stamping
};

static std::atomic<uint64_t> g_nextId{0};  // one per process (shared by PEs)

static uint64_t stamp(uint64_t* slot) {
  uint64_t cur = __atomic_load_n(slot, __ATOMIC_ACQUIRE);
  if (cur == 0) {
    uint64_t fresh = ((uint64_t)(CkMyNode() + 1) << 40) | (++g_nextId);
    if (__atomic_compare_exchange_n(slot, &cur, fresh, false, __ATOMIC_ACQ_REL,
                                    __ATOMIC_ACQUIRE))
      cur = fresh;
  }
  return cur;
}

// Return to the scheduler a few times, so the wrapper that started this thread
// has returned (and the runtime has dropped its reference to a nokeep message)
// before the caller touches its data; also give other PEs time to finish.
static void suspendAWhile() {
  for (int i = 0; i < 4; i++) CthYield();
}

static void fillPayload(uint64_t* d, int n) {
  d[0] = 0;
  for (int i = 1; i < n; i++) d[i] = 0x5a5a000000000000ull + (uint64_t)i;
}
static int checkPayload(const uint64_t* d, int n, int expectN) {
  if (n != expectN) return 0;
  for (int i = 1; i < n; i++)
    if (d[i] != 0x5a5a000000000000ull + (uint64_t)i) return 0;
  return 1;
}

enum Kind {
  K_MARSH, K_MARSH_NK, K_MARSH_T, K_MARSH_TNK, K_MARSH_TNKS, K_FIXED_TNKS,
  K_FIXED, K_FIXED_NK, K_VAR, K_VAR_NK, K_PACK, K_PACK_NK, K_COUNT
};
static const char* kindName[K_COUNT] = {
  "marshalled", "marshalled [nokeep]", "marshalled [threaded]",
  "marshalled [threaded,nokeep]", "marsh [thr,nokeep] suspends",
  "fixed [thr,nokeep] suspends", "fixed msg", "fixed msg [nokeep]",
  "varsize msg", "varsize msg [nokeep]", "packed msg", "packed msg [nokeep]"};

// Shared receiver logic for G and A. Owned messages are held until the next
// delivery to this receiver, so their blocks are not recycled mid-test.
struct Receiver {
  void* held = nullptr;
  void (*heldDelete)(void*) = nullptr;
  void releaseHeld() {
    if (held) { heldDelete(held); held = nullptr; }
  }

  Rec base(int elem) {
    Rec r;
    memset(&r, 0, sizeof(r));
    r.pe = CkMyPe();
    r.node = CkMyNode();
    r.physnode = CmiPhysicalNodeID(CkMyPe());
    r.elem = elem;
    r.envNokeep = -1;
    return r;
  }
  Rec onMarsh(int elem, int n, const uint64_t* data) {
    releaseHeld();
    Rec r = base(elem);
    uint64_t* d = const_cast<uint64_t*>(data);
    if (((uintptr_t)d & 7) != 0) CkAbort("marshalled array not 8-byte aligned");
    r.addr = r.dataAddr = (uint64_t)(uintptr_t)d;
    r.id = stamp(&d[0]);
    r.payloadOk = checkPayload(d, n, payloadWords);
    return r;
  }
  template <class M>
  Rec onMsg(int elem, M* m, uint64_t* d, uint64_t* markerSlot, int n, int expectN,
            bool owned) {
    releaseHeld();
    Rec r = base(elem);
    r.addr = (uint64_t)(uintptr_t)m;
    r.dataAddr = (uint64_t)(uintptr_t)d;
    r.envNokeep = CMI_MSG_NOKEEP(UsrToEnv(m)) ? 1 : 0;
    r.id = stamp(markerSlot);
    r.payloadOk = checkPayload(d, n, expectN);
    if (owned) { held = m; heldDelete = [](void* p) { delete (M*)p; }; }
    return r;
  }
};

#define RECEIVER_METHODS(ELEM)                                                 \
  void send(const Rec& r) {                                                    \
    contribute(sizeof(Rec), &r, CkReduction::concat,                           \
               CkCallback(CkIndex_Main::report(NULL), mainProxy));             \
  }                                                                            \
  void marsh(int n, const uint64_t* d) { send(rx.onMarsh(ELEM, n, d)); }       \
  void marshNK(int n, const uint64_t* d) { send(rx.onMarsh(ELEM, n, d)); }     \
  void marshT(int n, const uint64_t* d) { send(rx.onMarsh(ELEM, n, d)); }      \
  void marshTNK(int n, const uint64_t* d) { send(rx.onMarsh(ELEM, n, d)); }    \
  void marshTNKS(int n, const uint64_t* d) {                                   \
    suspendAWhile();                                                           \
    send(rx.onMarsh(ELEM, n, d));                                              \
  }                                                                            \
  void fixedTNKS(FixedMsg* m) {                                                \
    suspendAWhile();                                                           \
    send(rx.onMsg(ELEM, m, m->data, &m->data[0], kFixedWords, kFixedWords,     \
                  false));                                                     \
  }                                                                            \
  void fixed(FixedMsg* m) {                                                    \
    send(rx.onMsg(ELEM, m, m->data, &m->data[0], kFixedWords, kFixedWords,     \
                  true));                                                      \
  }                                                                            \
  void fixedNK(FixedMsg* m) {                                                  \
    send(rx.onMsg(ELEM, m, m->data, &m->data[0], kFixedWords, kFixedWords,     \
                  false));                                                     \
  }                                                                            \
  void var(VarMsg* m) {                                                        \
    send(rx.onMsg(ELEM, m, m->data, &m->data[0], m->n, payloadWords, true));   \
  }                                                                            \
  void varNK(VarMsg* m) {                                                      \
    send(rx.onMsg(ELEM, m, m->data, &m->data[0], m->n, payloadWords, false));  \
  }                                                                            \
  void pack(PackMsg* m) {                                                      \
    send(rx.onMsg(ELEM, m, m->data, &m->marker, m->n, payloadWords, true));    \
  }                                                                            \
  void packNK(PackMsg* m) {                                                    \
    send(rx.onMsg(ELEM, m, m->data, &m->marker, m->n, payloadWords, false));   \
  }

class G : public CBase_G {
  Receiver rx;
 public:
  G() {}
  RECEIVER_METHODS(-1)
};

class A : public CBase_A {
  Receiver rx;
  int stressNext = 0, stressBad = 0, stressMigs = 0;
 public:
  A() {}
  A(CkMigrateMessage*) {}
  void pup(PUP::er& p) {
    p | stressNext; p | stressBad; p | stressMigs;
  }
  RECEIVER_METHODS(thisIndex)

  template <class M>
  void stressRecv(M* m) {
    rx.releaseHeld();  // nothing is held across a migration
    if (m->seq != stressNext || !checkPayload(m->data, m->n, payloadWords)) stressBad++;
    stressNext++;
    delete m;
    if (stressNext == stressBcasts) {
      int v[2] = {stressBad, stressMigs};
      contribute(sizeof(v), v, CkReduction::sum_int,
                 CkCallback(CkReductionTarget(Main, stressDone), mainProxy));
    } else if (stressNext % 2 == 0) {
      stressMigs++;
      migrateMe((CkMyPe() + 1) % CkNumPes());
    }
  }
  void stressP(PackMsg* m) { stressRecv(m); }
  void stressV(VarMsg* m) { stressRecv(m); }
};

class Main : public CBase_Main {
  CProxy_G gp;
  CProxy_A ap;
  int perPe, verbose, withUnsafe, stressVarOnly;
  int test = 0;          // index into the (target, kind) sequence
  struct Row { std::string target, kind; int regNokeep; long recv, distinct,
               procs, minPP, maxPP, recvProcs, badPayload;
               int envNK; long reused = 0; };
  std::vector<Row> rows;
  int failures = 0;  // rows whose buffer count or payload check is wrong

 public:
  Main(CkArgMsg* m) {
    perPe = m->argc > 1 ? atoi(m->argv[1]) : 10;
    payloadWords = m->argc > 2 ? atoi(m->argv[2]) : 1024;
    verbose = m->argc > 3 ? atoi(m->argv[3]) : 0;
    withUnsafe = m->argc > 4 ? atoi(m->argv[4]) : 0;
    stressBcasts = m->argc > 5 ? atoi(m->argv[5]) : 200;
    stressVarOnly = m->argc > 6 ? atoi(m->argv[6]) : 0;
    delete m;
    if (payloadWords < 2) payloadWords = 2;
    mainProxy = thisProxy;
    CkPrintf("nokeepaudit: %d PEs, %d processes, %d physical nodes; "
             "array has %d elements (%d per PE); payload %d words "
             "(fixed msg %d words)\n",
             CkNumPes(), CkNumNodes(), CmiNumPhysicalNodes(), perPe * CkNumPes(),
             perPe, payloadWords, kFixedWords);
    gp = CProxy_G::ckNew();
    CkArrayOptions opts(perPe * CkNumPes());
    opts.setMap(CProxy_BlockMap::ckNew());
    opts.setInitCallback(CkCallback(CkIndex_Main::arrayReady(), thisProxy));
    ap = CProxy_A::ckNew(opts);
  }

  static int epFor(bool isGroup, int k) {
#define EP(T, name) CkIndex_##T::idx_##name(&T::name)
#define PICK(name) (isGroup ? EP(G, name) : EP(A, name))
    switch (k) {
      case K_MARSH: return PICK(marsh);
      case K_MARSH_NK: return PICK(marshNK);
      case K_MARSH_T: return PICK(marshT);
      case K_MARSH_TNK: return PICK(marshTNK);
      case K_MARSH_TNKS: return PICK(marshTNKS);
      case K_FIXED_TNKS: return PICK(fixedTNKS);
      case K_FIXED: return PICK(fixed);
      case K_FIXED_NK: return PICK(fixedNK);
      case K_VAR: return PICK(var);
      case K_VAR_NK: return PICK(varNK);
      case K_PACK: return PICK(pack);
      default: return PICK(packNK);
    }
#undef PICK
#undef EP
  }

  void arrayReady() {
    if (verbose) CkPrintf("array ready\n");
    next();
  }

  template <class P>
  void fire(P& p, int k) {
    switch (k) {
      case K_MARSH: case K_MARSH_NK: case K_MARSH_T: case K_MARSH_TNK:
      case K_MARSH_TNKS: {
        std::vector<uint64_t> buf(payloadWords);
        fillPayload(buf.data(), payloadWords);
        if (k == K_MARSH) p.marsh(payloadWords, buf.data());
        else if (k == K_MARSH_NK) p.marshNK(payloadWords, buf.data());
        else if (k == K_MARSH_T) p.marshT(payloadWords, buf.data());
        else if (k == K_MARSH_TNKS) p.marshTNKS(payloadWords, buf.data());
        else p.marshTNK(payloadWords, buf.data());
        break;
      }
      case K_FIXED: case K_FIXED_NK: case K_FIXED_TNKS: {
        FixedMsg* m = new FixedMsg;
        fillPayload(m->data, kFixedWords);
        if (k == K_FIXED) p.fixed(m);
        else if (k == K_FIXED_NK) p.fixedNK(m);
        else p.fixedTNKS(m);
        break;
      }
      case K_VAR: case K_VAR_NK: {
        VarMsg* m = new (payloadWords) VarMsg;
        m->n = payloadWords;
        fillPayload(m->data, payloadWords);
        if (k == K_VAR) p.var(m); else p.varNK(m);
        break;
      }
      default: {
        PackMsg* m = new PackMsg(payloadWords);
        fillPayload(m->data, payloadWords);
        if (k == K_PACK) p.pack(m); else p.packNK(m);
        break;
      }
    }
  }

  void next() {
    if (test == 2 * K_COUNT) {
      summary();
      if (failures) {
        CkPrintf("FAIL: %d rows wrong\n", failures);
        CkExit(1);
      }
      if (stressBcasts > 0) stress(); else { CkPrintf("PASS\n"); CkExit(); }
      return;
    }
    bool isGroup = test < K_COUNT;
    int k = test % K_COUNT;
    bool tnks = k == K_MARSH_TNKS || k == K_FIXED_TNKS;
    if ((k == K_MARSH_TNK && !(withUnsafe & 1)) || (tnks && !(withUnsafe & 4)) ||
        (!isGroup && k == K_PACK && perPe > 1 && !(withUnsafe & 2))) {
      rows.push_back(Row{isGroup ? "group" : "array", kindName[k], -1, 0, 0, 0, 0, 0,
                         0, 0, -3});
      test++; next(); return;
    }
    if (verbose) CkPrintf("test %d: %s %s\n", test, isGroup ? "group" : "array", kindName[k]);
    if (isGroup) fire(gp, k); else fire(ap, k);
  }

  void report(CkReductionMsg* msg) {
    bool isGroup = test < K_COUNT;
    int k = test % K_COUNT;
    int n = msg->getSize() / (int)sizeof(Rec);
    const Rec* r = (const Rec*)msg->getData();

    std::set<std::tuple<int, uint64_t, uint64_t>> inst;       // (proc, addr, id)
    std::map<int, std::set<std::pair<uint64_t, uint64_t>>> perProc;
    std::map<int, int> recvPerProc;
    std::set<std::pair<int, uint64_t>> addrs;  // (proc, address)
    long bad = 0;
    int envNK = -2;
    for (int i = 0; i < n; i++) {
      inst.insert(std::make_tuple(r[i].node, r[i].addr, r[i].id));
      perProc[r[i].node].insert(std::make_pair(r[i].addr, r[i].id));
      recvPerProc[r[i].node]++;
      addrs.insert(std::make_pair(r[i].node, r[i].addr));
      if (!r[i].payloadOk) bad++;
      if (envNK == -2) envNK = r[i].envNokeep;
      else if (envNK != r[i].envNokeep) envNK = 9;  // mixed
    }
    long mn = 1L << 60, mx = 0;
    for (auto& e : perProc) {
      mn = std::min(mn, (long)e.second.size());
      mx = std::max(mx, (long)e.second.size());
    }
    Row row{isGroup ? "group" : "array", kindName[k],
            (int)_entryTable[epFor(isGroup, k)]->noKeep, n, (long)inst.size(),
            (long)perProc.size(), mn, mx, (long)recvPerProc.size(), bad,
            envNK};
    row.reused = (long)inst.size() - (long)addrs.size();
    // What the runtime promises: nokeep shares one buffer per process, keep
    // gives every receiver its own; and every payload arrives intact.
    long want = row.regNokeep ? (long)CkNumNodes() : (long)n;
    if (row.distinct != want || bad != 0) {
      failures++;
      CkPrintf("FAIL: %s %s: %ld buffers (expected %ld), %ld bad payloads\n",
               row.target.c_str(), row.kind.c_str(), row.distinct, want, bad);
    }
    rows.push_back(row);
    long expect = isGroup ? CkNumPes() : (long)perPe * CkNumPes();
    if (n != expect)
      CkPrintf("WARNING: %s %s: %d deliveries, expected %ld\n", row.target.c_str(),
               row.kind.c_str(), n, expect);
    if (verbose) {
      for (auto& e : perProc) {
        CkPrintf("  %s %-30s proc %3d: %4d deliveries, %4zu buffers:", row.target.c_str(),
                 row.kind.c_str(), e.first, recvPerProc[e.first], e.second.size());
        int shown = 0;
        for (auto& b : e.second) {
          if (shown++ == 4) { CkPrintf(" ..."); break; }
          CkPrintf(" %#" PRIx64 "/%" PRIx64, b.first, b.second);
        }
        CkPrintf("\n");
      }
    }
    delete msg;
    test++;
    next();
  }

  // All stress broadcasts go out at once, so elements migrate while later ones
  // are still arriving; odd-numbered sends use the custom-packed message.
  void stress() {
    CkPrintf("\nstress: %d keep broadcasts (%s) to %d migrating elements\n",
             stressBcasts, stressVarOnly ? "varsize only" : "packed/varsize alternating",
             perPe * CkNumPes());
    for (int i = 0; i < stressBcasts; i++) {
      if (i % 2 && !stressVarOnly) {
        PackMsg* m = new PackMsg(payloadWords);
        fillPayload(m->data, payloadWords);
        m->seq = i;
        ap.stressP(m);
      } else {
        VarMsg* m = new (payloadWords) VarMsg;
        m->n = payloadWords;
        fillPayload(m->data, payloadWords);
        m->seq = i;
        ap.stressV(m);
      }
    }
  }
  void stressDone(int bad, int migrations) {
    CkPrintf("stress: %d missing, repeated, out-of-order or corrupt deliveries; "
             "%d migrations -- %s\n", bad, migrations,
             bad == 0 ? "PASS" : "FAIL");
    CkExit(bad == 0 ? 0 : 1);
  }

  void summary() {
    CkPrintf("\n%-6s %-29s %5s %5s %6s %8s %8s %7s %7s %4s\n", "target", "entry kind",
             "regNK", "envNK", "deliv", "buffers", "reused", "min/pr", "max/pr", "bad");
    for (auto& r : rows) {
      if (r.envNK == -3) {
        CkPrintf("%-6s %-29s  skipped (unsafe; see withUnsafe)\n",
                 r.target.c_str(), r.kind.c_str());
        continue;
      }
      const char* env = r.envNK == -1 ? "-" : r.envNK == 9 ? "mixed"
                        : r.envNK ? "1" : "0";
      CkPrintf("%-6s %-29s %5d %5s %6ld %8ld %8ld %7ld %7ld %4ld\n",
               r.target.c_str(), r.kind.c_str(), r.regNokeep, env, r.recv, r.distinct,
               r.reused, r.minPP, r.maxPP, r.badPayload);
    }
    CkPrintf("\nregNK  = _entryTable[ep]->noKeep as registered by charmxi\n"
             "envNK  = CMI_MSG_NOKEEP on the delivered envelope (- = marshalled, "
             "not visible)\n"
             "buffers = distinct (process, address, marker-id) instances delivered, "
             "system-wide\n"
             "reused = buffers found at an address an earlier buffer of the same test "
             "occupied (allocator reuse; separated by marker id)\n"
             "bad    = deliveries whose payload failed the pattern check\n"
             "min/pr, max/pr = distinct buffers in the least/most loaded process\n"
             "Lower bound for a broadcast: 1 buffer per process = %d\n",
             CkNumNodes());
  }
};

#include "nokeepaudit.def.h"
