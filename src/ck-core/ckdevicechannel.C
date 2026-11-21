/*
 * Device channels: see CkDeviceChannel in ckdevicechannel.h and the manual,
 * "GPU Channels".
 *
 * Sections: the endpoint transport (IPC mappings, the per-process
 * registration table, the receiver's read); the protocol messages and per-PE
 * tables; same-host signalling through shared memory; the watchdog; the
 * Converse handlers; batched device copies; CkDeviceChannel itself; the
 * migration hooks called from CkLocMgr::emigrate.
 */
#include "charm++.h"
#include "ck.h"
#include "envelope.h"
#include "ckrdmadevice.h"

#if (CMK_CUDA || CMK_HIP)

#include <atomic>
#include <climits>
#include <cstring>
#include <ctime>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <map>
#include <mutex>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include "hapi.h"

// CHARM_DEBUG_CHANNEL: one line per protocol event (resolve, credit, closing,
// ack, staged put, read), and the stall scan every 10 s.
static bool deviceChannelDebug() {
  static const bool on = (getenv("CHARM_DEBUG_CHANNEL") != NULL);
  return on;
}

// The element whose entry method is running, or NULL (a group, the main
// chare, a Converse handler).
static CkLocRec* chanActiveRec() {
  Chare* obj = CkActiveObj();
  return (obj && obj->ckInitialized) ? obj->getCkLocRec() : NULL;
}

/****************************** Endpoints ******************************/

static std::atomic<bool> channel_exiting{false};   // set once a PE's exit sequence reaches ConverseExit

// Called on each PE when its exit sequence reaches ConverseExit (init.C):
// after it the network layer is gone, and an endpoint closed at process exit
// drops its entry without deregistering.
void CkRdmaDeviceExit() { channel_exiting = true; }

// The base of the allocation holding ptr: what an IPC handle maps in the other
// process (pool allocators, cupy and torch hand out pieces of allocations).
static bool chanAllocBase(const void* ptr, uintptr_t* base) {
#if CMK_HIP
  void* b = nullptr; size_t sz = 0;
  if (hipMemGetAddressRange(&b, &sz, (void*)ptr) != hipSuccess) { (void)hipGetLastError(); return false; }
  *base = (uintptr_t)b;
  return true;
#else
  typedef int (*RangeFn)(unsigned long long*, size_t*, unsigned long long);
  static RangeFn fn = nullptr;
  static bool looked = false;
  if (!looked) {
    void* s = dlsym(RTLD_DEFAULT, "cuMemGetAddressRange_v2");
    if (!s) {
      void* lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
      if (!lib) lib = dlopen("libcuda.so.1", RTLD_NOW);
      if (lib) s = dlsym(lib, "cuMemGetAddressRange_v2");
    }
    fn = (RangeFn)s;
    looked = true;
  }
  if (!fn) return false;
  unsigned long long b = 0; size_t sz = 0;
  if (fn(&b, &sz, (unsigned long long)(uintptr_t)ptr) != 0) return false;
  *base = (uintptr_t)b;
  return true;
#endif
}

// IPC mappings of other processes' allocations, one per process, shared by
// every holder on every PE of the process and reference counted. A handle
// that did not open is kept with base null and no references, so that the
// failure is reported once. The tables of this section are never destroyed,
// so that an endpoint released at exit can still use them.
struct ChanIpcMap { void* base; int refs; };
static std::mutex& chan_ipc_lock = *new std::mutex;
static std::map<std::string, ChanIpcMap>& chan_ipc = *new std::map<std::string, ChanIpcMap>;

CkDeviceChannelEndpoint::CkDeviceChannelEndpoint(const void* ptr_, size_t cnt_)
  : ptr(ptr_), cnt(cnt_) {
  pe = CkMyPe();
  node = CkMyNode();
  phys_node = CmiPhysicalNodeID(CkMyPe());
}

void* CkDeviceChannelEndpoint::view() const {
  if (ipc_open) return ipc_ptr;
  if (!ipc_valid) return nullptr;
  std::string key((const char*)&ipc_handle, sizeof(hapiIpcMemHandle_t));
  std::lock_guard<std::mutex> g(chan_ipc_lock);
  auto it = chan_ipc.find(key);
  void* base = nullptr;
  if (it != chan_ipc.end()) {
    base = it->second.base;
    if (base) it->second.refs++;
  } else {
    if (hapiIpcOpenMemHandle(&base, ipc_handle, hapiIpcMemLazyEnablePeerAccess) != hapiSuccess) {
      (void)hapiGetLastError();
      CmiPrintf("[%d] CkDeviceChannel> IPC open of PE %d's buffer failed; RDMA is used for it\n", CkMyPe(), pe);
      base = nullptr;
    }
    chan_ipc[key] = ChanIpcMap{ base, base ? 1 : 0 };
  }
  ipc_open = true;
  ipc_ptr = base ? (char*)base + ipc_offset : nullptr;
  return ipc_ptr;
}

// A channel releases only when none of its copies through the mapping are in
// flight, so the last release can close the mapping at once. A holder that
// cannot tell (CkDyn) asks for a device synchronize before the close.
void CkDeviceChannelEndpoint::releaseView(bool synchronize) const {
  if (!ipc_open) return;
  ipc_open = false;
  void* p = ipc_ptr;
  ipc_ptr = nullptr;
  if (!p) return;
  std::string key((const char*)&ipc_handle, sizeof(hapiIpcMemHandle_t));
  std::lock_guard<std::mutex> g(chan_ipc_lock);
  auto it = chan_ipc.find(key);
  if (it == chan_ipc.end() || --it->second.refs > 0) return;
  if (!channel_exiting) {
    if (synchronize) hapiCheck(hapiDeviceSynchronize());
    hapiCheck(hapiIpcCloseMemHandle(it->second.base));
  }
  chan_ipc.erase(it);
}

void CkDeviceChannelEndpoint::takeView(const CkDeviceChannelEndpoint& o) const {
  CmiAssert(!ipc_open);
  ipc_ptr = o.ipc_ptr; ipc_open = o.ipc_open;
  o.ipc_ptr = nullptr; o.ipc_open = false;
}

int CkDeviceChannelEndpoint::mode() const {
  if (node == CkMyNode()) return 0;
  if (phys_node == CmiPhysicalNodeID(CkMyPe()) && ipc_valid && view() != nullptr) return 1;
  return 2;
}

// Network registrations of opened buffers, one table per process. open() adds
// an entry and close() removes it; the memory is registered the first time a
// transfer needs RDMA, so a run that never needs it (one process, or IPC on
// one host) never registers device memory with the network.
struct ChanReg {
  const void* ptr;
  size_t cnt;              // the length registered (cnt at open())
  int state = 0;           // 0 not yet registered, 1 registered, -1 cannot be registered
  const char* why = "";    // the reason for state -1
  CmiNcpyBuffer ncpy;
};
static std::mutex& chan_reg_lock = *new std::mutex;
static std::map<uint64_t, ChanReg>& chan_reg = *new std::map<uint64_t, ChanReg>;
static std::atomic<uint64_t> chan_next_id{1};

// An id is node (20 bits) | run token (16 bits) | counter (28 bits). The token
// differs between a run and its restart from a checkpoint, so an object
// restored from the checkpoint does not name a buffer opened in the new run.
static uint64_t chanRunToken() {
  static const uint64_t token = ((uint64_t)time(nullptr) ^ ((uint64_t)getpid() << 5)) & 0xffff;
  return token;
}

// Registers r once; called with chan_reg_lock held.
static void chanRegisterLocked(ChanReg& r) {
  if (r.state != 0) return;
  if (r.cnt > (size_t)INT_MAX) {
    r.state = -1;
    r.why = "the buffer is 2 GiB or larger, and the network registration takes an int length";
    return;
  }
  r.ncpy.deviceRdmaOpInfo = nullptr;
  memset(r.ncpy.layerInfo, 0, sizeof(r.ncpy.layerInfo));   // stays zero if nothing is registered
  r.ncpy.init(r.ptr, r.cnt, CMK_BUFFER_REG, CMK_BUFFER_NODEREG);   // released by close()
  // The memory region handle is the first field of the layer information
  // (CmiSetRdmaBufferInfo); the backend leaves it null when it registered nothing.
  void* mr = nullptr;
  memcpy(&mr, r.ncpy.layerInfo + CmiGetRdmaCommonInfoSize(), sizeof(mr));
  if (!r.ncpy.isRegistered || mr == nullptr) {
    r.state = -1;
    r.why = "the network layer did not register the device memory";
    return;
  }
  r.state = 1;
}

void CkDeviceChannelEndpoint::open() {
  if (pe != CkMyPe()) CkAbort("CkDeviceChannel: a buffer of PE %d opened on PE %d", pe, CkMyPe());
  if (id != 0) close();
  uintptr_t base = 0;
  if (chanAllocBase(ptr, &base) && hapiIpcGetMemHandle(&ipc_handle, (void*)ptr) == hapiSuccess) {
    ipc_offset = (size_t)((uintptr_t)ptr - base);
    ipc_valid = true;
  } else {
    (void)hapiGetLastError();
    ipc_valid = false;
  }
  id = ((uint64_t)CkMyNode() << 44) | (chanRunToken() << 28) | (chan_next_id++ & 0xfffffff);
  rdma_valid = false;
  ChanReg r;
  r.ptr = ptr;
  r.cnt = cnt;
  std::lock_guard<std::mutex> g(chan_reg_lock);
  chan_reg.emplace(id, r);
}

void CkDeviceChannelEndpoint::close() {
  if (id == 0) return;
  if (node == CkMyNode()) {
    std::lock_guard<std::mutex> g(chan_reg_lock);
    auto it = chan_reg.find(id);
    if (it != chan_reg.end()) {
      if (it->second.state == 1 && !channel_exiting) it->second.ncpy.deregisterMem();
      chan_reg.erase(it);
    }
  }
  id = 0;
  rdma_valid = false;
}

bool CkDeviceChannelEndpoint::registration(CmiNcpyBuffer* out, const char** why) const {
  const char* reason = "";
  bool ok = false;
  if (id == 0) reason = "the buffer was not opened";
  else if (node != CkMyNode()) reason = "the buffer belongs to another process";
  else {
    std::lock_guard<std::mutex> g(chan_reg_lock);
    auto it = chan_reg.find(id);
    if (it == chan_reg.end()) reason = "the buffer was closed";
    else {
      chanRegisterLocked(it->second);
      if (it->second.state != 1) reason = it->second.why;
      else if (cnt > it->second.cnt) reason = "the transfer is larger than the buffer at open()";
      else {
        *out = it->second.ncpy;
        out->cnt = cnt;
        out->deviceRdmaOpInfo = nullptr;
        ok = true;
      }
    }
  }
  if (why) *why = reason;
  return ok;
}

void CkDeviceChannelEndpoint::pup(PUP::er& p) {
  p((char*)&ptr, sizeof(ptr));
  p|cnt;
  p|pe;
  p|node;
  p|phys_node;
  p((char*)&ipc_handle, sizeof(ipc_handle));
  p|ipc_offset;
  p|ipc_valid;
  p|id;
  p|rdma_valid;
  if (rdma_valid) p|rdma;
  if (p.isUnpacking()) {
    ipc_ptr = nullptr;
    ipc_open = false;
  }
}

static void chanCheckRdmaUsable() {
  if (CmiUseCopyBasedRDMA)
    CkAbort("CkDeviceChannel: a put between PEs on different hosts, or whose IPC handle did not open, is "
            "read with RDMA, but this run moves RDMA data through host memory (the LCW backend, or "
            "+nordma), which cannot read or write device memory");
}

static void chanIssueGet(CmiNcpyBuffer d, CmiNcpyBuffer s, const CkCallback& done) {
  d.deviceRdmaOpInfo = nullptr;
  s.deviceRdmaOpInfo = nullptr;
  CkCallback srcAck(CkCallback::ignore), dstAck(done);
  QdCreate(1);   // matched in CkRdmaDirectAckHandler
  d.rdmaGet(s, sizeof(CkCallback), (char*)&srcAck, (char*)&dstAck);
}

// The receiver's read of an announced put into its buffer: a copy within the
// process or through the IPC mapping, or an RDMA get of the registration the
// source's process put in the announce. done runs on this PE once the bytes
// are in place. Nothing blocks the PE.
static void chanRead(const CkDeviceChannelEndpoint& dst, const CkDeviceChannelEndpoint& src, size_t bytes,
                     const CkCallback& done) {
  const int m = src.mode();
  if (m == 2) {
    chanCheckRdmaUsable();
    if (!src.rdma_valid)
      CkAbort("CkDeviceChannel: PE %d reads a put from PE %d with RDMA (another host, or an IPC handle that "
              "did not open), and the source could not be registered with the network\n", CkMyPe(), src.pe);
    CmiNcpyBuffer d;
    const char* why = "";
    CkDeviceChannelEndpoint dd = dst;
    dd.cnt = bytes;
    if (!dd.registration(&d, &why))
      CkAbort("CkDeviceChannel: the receive buffer on PE %d cannot be registered with the network: %s\n",
              CkMyPe(), why);
    CmiNcpyBuffer s = src.rdma;
    s.cnt = bytes;
    chanIssueGet(d, s, done);
    return;
  }
  const void* from = (m == 0) ? src.ptr : src.view();
  hapiCheck(hapiMemcpyAsync((void*)dst.ptr, from, bytes, hapiMemcpyDefault, hapiStreamPerThread));
  hapiAddCallback(hapiStreamPerThread, done);
}

/****************************** Endpoint transfers (CkDyn) ******************************/
//
// CkDyn's persistent device buffers (ckdyn.C) are endpoints read or written by
// any peer holding the handle, with no channel around them. A get copies a
// peer's buffer into this process's buffer, a put the other way. Across hosts
// (or where the IPC handle does not open) both are RDMA gets: by this PE for a
// get; for a put, by the destination's PE, which then acknowledges, because an
// RDMA put completes at its source before the data is visible at the
// destination. The source's registration is asked of its process the first
// time and kept by id. Before an RDMA transfer the local buffer's stream is
// synchronized.

int device_endpoint_reg_req_handler;
int device_endpoint_reg_rep_handler;
int device_endpoint_put_handler;
int device_endpoint_ack_handler;
static int endpoint_verbose = 0;   // +gpupersistverbose

// Other processes' registrations, by endpoint id (under chan_reg_lock).
static std::map<uint64_t, CmiNcpyBuffer>& chan_remote = *new std::map<uint64_t, CmiNcpyBuffer>;

// With +gpupersistverbose: one line per process, operation and mode the first
// time it is used, so that a run states which path it took.
static void endpointAnnounce(const char* what, int mode) {
  if (!endpoint_verbose) return;
  static std::mutex lock;
  static std::map<std::string, int> said;
  {
    std::lock_guard<std::mutex> g(lock);
    std::string key = std::string(what) + (char)('0' + mode);
    if (said.count(key)) return;
    said[key] = 1;
  }
  static const char* names[3] = {
    "copy on the device (one process)",
    "CUDA/HIP IPC between processes on one host",
    "RDMA on registered device memory" };
  CmiPrintf("[%d] CkDyn> persistent %s: %s\n", CkMyPe(), what, names[mode]);
}

// The registration of b, a buffer of this process, for a transfer of bytes.
static CmiNcpyBuffer endpointLocalNcpy(const CkDeviceChannelEndpoint& b, size_t bytes) {
  CkDeviceChannelEndpoint t = b;
  t.cnt = bytes;
  CmiNcpyBuffer n;
  const char* why = "";
  if (!t.registration(&n, &why))
    CkAbort("CkDyn: an RDMA transfer of %zu bytes on PE %d's persistent device buffer cannot use it: %s",
            bytes, b.pe, why);
  return n;
}

struct EndpointPending { CmiNcpyBuffer d; size_t bytes; int src_pe; CkCallback done; };
struct EndpointRegReqMsg {
  char header[CmiMsgHeaderSizeBytes];
  int issuer;
  uint64_t id;
  uint64_t pending;
};
struct EndpointRegRepMsg {
  char header[CmiMsgHeaderSizeBytes];
  uint64_t id;
  uint64_t pending;
  int ok;
  size_t cnt;
  char why[128];
  CmiNcpyBuffer ncpy;
};

static void endpointRdmaGet(const CkDeviceChannelEndpoint& dst, const CkDeviceChannelEndpoint& src, size_t bytes,
                            const CkCallback& done) {
  chanCheckRdmaUsable();
  if (src.id == 0)
    CkAbort("CkDyn: an RDMA transfer from PE %d's persistent device buffer, which was not opened", src.pe);
  CmiNcpyBuffer d = endpointLocalNcpy(dst, bytes);
  CmiNcpyBuffer s;
  bool have = false;
  {
    std::lock_guard<std::mutex> g(chan_reg_lock);
    auto it = chan_remote.find(src.id);
    if (it != chan_remote.end()) { s = it->second; have = true; }
  }
  if (have) {
    if (bytes > s.cnt)
      CkAbort("CkDyn: a transfer of %zu bytes from PE %d's persistent device buffer exceeds the %zu bytes it "
              "had at open()", bytes, src.pe, s.cnt);
    s.cnt = bytes;
    chanIssueGet(d, s, done);
    return;
  }
  EndpointRegReqMsg* m = (EndpointRegReqMsg*)CmiAlloc(sizeof(EndpointRegReqMsg));
  m->issuer = CkMyPe();
  m->id = src.id;
  m->pending = (uint64_t)(uintptr_t)new EndpointPending{ d, bytes, src.pe, done };
  CmiSetHandler(m, device_endpoint_reg_req_handler);
  QdCreate(1);
  CmiSyncSendAndFree(src.pe, sizeof(EndpointRegReqMsg), (char*)m);
}

// On the source's process: register the buffer if needed and reply.
extern "C" void* device_endpoint_reg_req_bridge(void* arg) {
  EndpointRegReqMsg* m = (EndpointRegReqMsg*)arg;
  QdProcess(1);
  EndpointRegRepMsg* r = (EndpointRegRepMsg*)CmiAlloc(sizeof(EndpointRegRepMsg));
  r->id = m->id;
  r->pending = m->pending;
  r->ok = 0;
  r->cnt = 0;
  r->why[0] = '\0';
  {
    std::lock_guard<std::mutex> g(chan_reg_lock);
    auto it = chan_reg.find(m->id);
    if (it == chan_reg.end()) {
      snprintf(r->why, sizeof(r->why), "the buffer's owner closed it");
    } else {
      chanRegisterLocked(it->second);
      r->cnt = it->second.cnt;
      if (it->second.state == 1) { r->ok = 1; r->ncpy = it->second.ncpy; }
      else snprintf(r->why, sizeof(r->why), "%s", it->second.why);
    }
  }
  CmiSetHandler(r, device_endpoint_reg_rep_handler);
  QdCreate(1);
  CmiSyncSendAndFree(m->issuer, sizeof(EndpointRegRepMsg), (char*)r);
  CmiFree(m);
  return NULL;
}

// On the issuing PE: keep the source's registration and issue the get.
extern "C" void* device_endpoint_reg_rep_bridge(void* arg) {
  EndpointRegRepMsg* r = (EndpointRegRepMsg*)arg;
  QdProcess(1);
  EndpointPending* p = (EndpointPending*)(uintptr_t)r->pending;
  if (!r->ok)
    CkAbort("CkDyn: a transfer needs RDMA on PE %d's persistent device buffer of %zu bytes, but it cannot be "
            "registered with the network: %s", p->src_pe, r->cnt, r->why);
  CmiNcpyBuffer s = r->ncpy;
  s.deviceRdmaOpInfo = nullptr;
  {
    std::lock_guard<std::mutex> g(chan_reg_lock);
    chan_remote[r->id] = s;
  }
  if (p->bytes > s.cnt)
    CkAbort("CkDyn: a transfer of %zu bytes from PE %d's persistent device buffer exceeds the %zu bytes it "
            "had at open()", p->bytes, p->src_pe, s.cnt);
  s.cnt = p->bytes;
  chanIssueGet(p->d, s, p->done);
  delete p;
  CmiFree(r);
  return NULL;
}

struct EndpointPutMsg {
  char header[CmiMsgHeaderSizeBytes];
  int issuer;
  int len;
  uint64_t rec;     // the issuer's done callback
  size_t bytes;
  CmiNcpyBuffer src;
  // then the pupped destination endpoint
};
struct EndpointAckMsg {
  char header[CmiMsgHeaderSizeBytes];
  uint64_t rec;
};
struct EndpointPutAck { int issuer; uint64_t rec; };

static void endpointPutGot(void* param, void* msg) {
  if (msg) CkFreeMsg(msg);
  EndpointPutAck* a = (EndpointPutAck*)param;
  EndpointAckMsg* m = (EndpointAckMsg*)CmiAlloc(sizeof(EndpointAckMsg));
  m->rec = a->rec;
  CmiSetHandler(m, device_endpoint_ack_handler);
  QdCreate(1);
  CmiSyncSendAndFree(a->issuer, sizeof(EndpointAckMsg), (char*)m);
  delete a;
}

// On the destination's PE: get the source into the destination.
extern "C" void* device_endpoint_put_bridge(void* arg) {
  EndpointPutMsg* m = (EndpointPutMsg*)arg;
  QdProcess(1);
  CkDeviceChannelEndpoint dst;
  PUP::fromMem up((char*)m + sizeof(EndpointPutMsg));
  up | dst;
  CmiNcpyBuffer s = m->src;
  s.cnt = m->bytes;
  chanIssueGet(endpointLocalNcpy(dst, m->bytes), s, CkCallback(endpointPutGot, new EndpointPutAck{ m->issuer, m->rec }));
  CmiFree(m);
  return NULL;
}

// On the issuing PE: the destination has the bytes.
extern "C" void* device_endpoint_ack_bridge(void* arg) {
  EndpointAckMsg* m = (EndpointAckMsg*)arg;
  QdProcess(1);
  CkCallback* done = (CkCallback*)(uintptr_t)m->rec;
  done->send(NULL);
  delete done;
  CmiFree(m);
  return NULL;
}

void CkDeviceEndpointGet(const CkDeviceChannelEndpoint& dst, const CkDeviceChannelEndpoint& src, size_t bytes,
                         hapiStream_t stream, const CkCallback& done) {
  const int m = src.mode();
  endpointAnnounce("get", m);
  if (m == 2) {
    hapiCheck(hapiStreamSynchronize(stream));   // earlier work on the destination is done
    endpointRdmaGet(dst, src, bytes, done);
    return;
  }
  const void* from = (m == 0) ? src.ptr : src.view();
  hapiCheck(hapiMemcpyAsync((void*)dst.ptr, from, bytes, hapiMemcpyDeviceToDevice, stream));
  hapiAddCallback(stream, done);
}

void CkDeviceEndpointPut(const CkDeviceChannelEndpoint& dst, const CkDeviceChannelEndpoint& src, size_t bytes,
                         hapiStream_t stream, const CkCallback& done) {
  const int m = dst.mode();
  endpointAnnounce("put", m);
  if (m == 2) {
    chanCheckRdmaUsable();
    if (dst.id == 0)
      CkAbort("CkDyn: an RDMA transfer into PE %d's persistent device buffer, which was not opened", dst.pe);
    CmiNcpyBuffer s = endpointLocalNcpy(src, bytes);
    hapiCheck(hapiStreamSynchronize(stream));   // the source's data is complete
    CkDeviceChannelEndpoint dcopy = dst;
    PUP::sizer ps;
    ps | dcopy;
    const int len = (int)ps.size();
    EndpointPutMsg* msg = (EndpointPutMsg*)CmiAlloc(sizeof(EndpointPutMsg) + len);
    msg->issuer = CkMyPe();
    msg->len = len;
    msg->rec = (uint64_t)(uintptr_t)new CkCallback(done);
    msg->bytes = bytes;
    msg->src = s;
    PUP::toMem pm((char*)msg + sizeof(EndpointPutMsg));
    pm | dcopy;
    CmiSetHandler(msg, device_endpoint_put_handler);
    QdCreate(1);
    CmiSyncSendAndFree(dst.pe, sizeof(EndpointPutMsg) + len, (char*)msg);
    return;
  }
  void* to = (m == 0) ? (void*)dst.ptr : dst.view();
  hapiCheck(hapiMemcpyAsync(to, src.ptr, bytes, hapiMemcpyDeviceToDevice, stream));
  hapiAddCallback(stream, done);
}

void CkDeviceEndpointForget(const CkDeviceChannelEndpoint& peer) {
  if (peer.node == CkMyNode()) return;
  {
    std::lock_guard<std::mutex> g(chan_reg_lock);
    chan_remote.erase(peer.id);
  }
  peer.releaseView(/*synchronize=*/true);
}

/****************************** Protocol messages and per-PE tables ******************************/

int device_channel_credit_handler;
int device_channel_closing_handler;
int device_channel_closeack_handler;
int device_channel_resolved_handler;
int device_channel_done_handler;
int device_channel_probe_handler;

struct DeviceChannelCredit {
  char core[CmiMsgHeaderSizeBytes];
  uint64_t token;
  uint64_t channel;
  uint64_t seq;
  uint64_t epoch;   // the resolution the credit belongs to
  int dst_pe;       // where it was addressed (a misdelivery is a transport fault, printed at once)
  int src_pe;
};
// Watchdog probe: a sender whose put waits for a credit asks its owner to
// print its own view of the channel.
struct DeviceChannelProbe {
  char core[CmiMsgHeaderSizeBytes];
  uint64_t channel;
  uint64_t token;
  uint64_t epoch;
  uint64_t seq;
  int pe;
  int owner_known;      // the owner element as the copy knows it
  uint64_t owner_id;
  CkGroupID owner_mgr;
};
struct DeviceChannelClosing {
  char core[CmiMsgHeaderSizeBytes];
  uint64_t token;
  uint64_t channel;
  uint64_t generation;
  uint64_t epoch;   // the resolution being closed
  int owner_pe;
};
struct DeviceChannelCloseAck {
  char core[CmiMsgHeaderSizeBytes];
  uint64_t channel;
  uint64_t token;
  int pe;
  uint64_t last_seq;    // the sender's last put written into the buffer (UINT64_MAX: unknown, the copy had left)
  uint64_t last_bytes;  // its size
};
struct DeviceChannelDone {   // the receiver has read put `seq`: the sender's source is free
  char core[CmiMsgHeaderSizeBytes];
  uint64_t token;
  uint64_t channel;
  uint64_t seq;
};
struct DeviceChannelResolved {
  char core[CmiMsgHeaderSizeBytes];
  uint64_t token;
  uint64_t channel;
  uint64_t generation;
  uint64_t epoch;   // this resolution's number; its credits and closing carry it
  int owner_pe;
  int free;
  uint64_t taken_seq;   // the last put of this sender's in the buffer at the reply (free says whether it is still held)
  int desc_bytes;   // the pup of the receiver's endpoint follows the struct
  int dst_pe;       // addressed to; a misdelivery is printed at once
  int src_pe;
  int credit_pe;    // same-host sender: the owner's shared-memory area and slot for its credits
  int credit_slot;  // (-1: credits are messages)
};

// Each PE's last watchdog tick, by rank: a sibling that stops ticking is
// spinning or blocked, and a stuck PE never reports itself.
static double device_channel_last_tick[2048];
// Watchdog counters (per PE): what arrived for nobody, or was refused.
static thread_local long device_channel_credits_dropped = 0;
static thread_local DeviceChannelCredit device_channel_last_dropped_credit;
static thread_local long device_channel_credits_refused = 0;
static thread_local char device_channel_last_refused[200];
static thread_local long device_channel_resolved_older = 0;   // resolutions ignored: older than (or equal to) the one held
static thread_local char device_channel_last_resolved_older[200];
static thread_local long device_channel_credits_early = 0;
static thread_local char device_channel_last_early[200];
static thread_local long device_channel_resolved_dropped = 0;
static thread_local char device_channel_last_resolved_dropped[200];
static thread_local long device_channel_closing_acked_blind = 0;
static thread_local char device_channel_last_closing_blind[200];

typedef std::unordered_map<uint64_t, CkDeviceChannel*> DeviceChannelTable;
CkpvStaticDeclare(DeviceChannelTable*, device_channel_senders);
CkpvStaticDeclare(DeviceChannelTable*, device_channel_owners);
CkpvStaticDeclare(uint64_t, device_channel_counter);
// Credits that arrived ahead of the resolution they belong to, by sender
// token: the owner replies to a resolve and releases the buffer afterwards,
// and the two messages can reach the sender in either order. A credit for a
// later epoch than the copy's current one can only be the owner's answer to
// this copy's outstanding resolve, so it is kept and applied when that
// resolution arrives.
struct DeviceChannelEarlyCredit { uint64_t epoch, seq; };
typedef std::unordered_map<uint64_t, std::vector<DeviceChannelEarlyCredit>> DeviceChannelEarlyCredits;
CkpvStaticDeclare(DeviceChannelEarlyCredits*, device_channel_early_credits);
// Staging buffers of puts made while the receiver's channel was closed
// (issueStaged), by (sender token, seq): freed by the receiver's DONE, which
// finds them here even if the sending element's channel copy is gone.
struct DeviceChannelStaged {
  void* buf;
  CkDeviceChannelEndpoint ep;   // the staging buffer as announced (its registration is closed with it)
  CkGroupID holder_mgr;
  CmiUInt8 holder_elem;
  bool holder_known;
};
typedef std::map<std::pair<uint64_t, uint64_t>, DeviceChannelStaged> DeviceChannelStagedTable;
CkpvStaticDeclare(DeviceChannelStagedTable*, device_channel_staged);
// The puts a PE issues on one producing stream within one entry method become
// one batched copy with one completion callback (deviceChannelBatchFlush).
// Flushed when the stream changes, when full, when the entry method returns,
// at every scheduler iteration and at idle: a batch never outlives the entry
// method that opened it.
struct DeviceChannelBatchEntry { uint64_t token; uint64_t seq; CkCallback user; CkCallback note; void* note_msg; };
struct DeviceChannelBatch {
  hapiStream_t stream;
  bool in_entry;   // opened inside an element's entry method: CkCallstackPop flushes it; otherwise the scheduler loop does
  std::vector<void*> dsts;
  std::vector<const void*> srcs;
  std::vector<size_t> sizes;
  std::vector<DeviceChannelBatchEntry> entries;
};
CkpvStaticDeclare(DeviceChannelBatch*, device_channel_batch);
static const size_t kChanBatchMax = 256;
static thread_local bool device_channel_batch_ready = false;   // this PE's tables exist (CkDeviceChannelInit ran)
static void deviceChannelBatchFlush();
// Channel events handled on this PE (puts, releases, resolves, credits,
// completions): the watchdog reads it every 10 s.
CkpvStaticDeclare(uint64_t, device_channel_progress);
CkpvStaticDeclare(uint64_t, device_channel_progress_seen);
CkpvStaticDeclare(int, device_channel_quiet_ticks);
CkpvStaticDeclare(bool, device_channel_late_init_done);
static inline void deviceChannelProgress() { ++CkpvAccess(device_channel_progress); }
static void deviceChannelLateInit();

// The protocol's control messages: the closing, its ack, the resolution, the
// credit and the DONE. Ordinary sends; where the network has an urgent queue
// that the destination serves ahead of its scheduler queue, this is the place
// to use it (each hop otherwise waits for the queue's depth).
static inline void chanSendControl(int pe, int size, void* m) {
  CmiSyncSendAndFree(pe, size, (char*)m);
}

/****************************** Same host: shared memory ******************************/
//
// A credit or an arrival note between two processes on one host would be a
// message through the network's loopback. Instead every PE of the host owns an
// area in one segment: credit slots the PE grants to same-host senders (a word
// the owner increments at release(), the sender reads), and a ring of arrival
// notes other PEs' completions write and this PE's scheduler-loop poll turns
// into the local event. Cross-host pairs and pairs inside one process keep
// their messages (the latter are a local queue push already).
//
// The segment's pages are zero when created, and zero is the empty state of
// everything in it, so no rendezvous is needed: each process opens-or-creates
// the file, sizes it (idempotent) and maps it. The name carries the host and
// the job step (a step killed mid-way leaves the segment behind with live
// state; the next step must not open it), and is unlinked at exit.
namespace {
constexpr int CHAN_SHM_CREDITS = 8192;    // credit slots per PE area
constexpr int CHAN_SHM_NOTES = 8192;      // note ring entries per PE area
constexpr int CHAN_SHM_NOTE_MARGIN = 512; // tickets producers may hold between the check and the write

struct ChanShmCredit {
  std::atomic<uint64_t> channel;   // 0: free. Owner sets channel+epoch, sender checks both around its read.
  std::atomic<uint64_t> epoch;
  std::atomic<uint64_t> count;     // credits granted under that resolution
};
struct ChanShmNote {
  std::atomic<uint64_t> ready;     // ticket + 1 once the entry is complete
  CkDeviceChannelEvent ev;
  alignas(8) char cb[sizeof(CkCallback)];   // the owner's arrival callback, as the sender's copy holds it
};
struct ChanShmArea {
  ChanShmCredit credit[CHAN_SHM_CREDITS];
  std::atomic<uint64_t> note_head;  // producers' ticket counter
  std::atomic<uint64_t> note_tail;  // consumer's progress, published for the producers' overflow check
  char pad[64];
  ChanShmNote note[CHAN_SHM_NOTES];
};
// The words live in shared memory, so their atomics must be lock-free; 64-bit
// atomics are on x86-64 and aarch64.

ChanShmArea* chan_shm_base = nullptr;   // per process
int chan_shm_areas = 0;                 // PEs on this host
std::string chan_shm_name;
std::once_flag chan_shm_once;

void chanShmUnlink() { if (!chan_shm_name.empty()) shm_unlink(chan_shm_name.c_str()); }

void chanShmMap() {
  const int host = CmiPhysicalNodeID(CkMyPe());
  chan_shm_areas = CmiNumPesOnPhysicalNode(host);
  const char* job = getenv("SLURM_JOB_ID");
  const char* step = getenv("SLURM_STEP_ID");
  chan_shm_name = "charm-chan-" + std::to_string(host) + "-" + (job ? std::string(job) : std::to_string((long)getppid())) +
                  (step ? "-" + std::string(step) : std::string());
  const size_t bytes = sizeof(ChanShmArea) * (size_t)chan_shm_areas;
  int fd = shm_open(chan_shm_name.c_str(), O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
  if (fd < 0) CkAbort("CkDeviceChannel: shm_open(%s) failed: %s\n", chan_shm_name.c_str(), strerror(errno));
  if (ftruncate(fd, (off_t)bytes) != 0) CkAbort("CkDeviceChannel: ftruncate(%s, %zu) failed: %s\n", chan_shm_name.c_str(), bytes, strerror(errno));
  void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  ::close(fd);
  if (p == MAP_FAILED) CkAbort("CkDeviceChannel: mmap(%s, %zu) failed: %s\n", chan_shm_name.c_str(), bytes, strerror(errno));
  chan_shm_base = (ChanShmArea*)p;
  atexit(chanShmUnlink);
}

inline ChanShmArea& chanShmArea(int node_local_pe) { return chan_shm_base[node_local_pe]; }
inline int chanShmIndexOf(int pe) { return CmiPhysicalRank(pe); }
// A pair that signals through shared memory: same host, different process.
inline bool chanShmPair(int pe) {
  return chan_shm_base != nullptr && CmiPhysicalNodeID(pe) == CmiPhysicalNodeID(CkMyPe()) &&
         CmiNodeOf(pe) != CmiMyNode();
}

// Owner: a free slot in this PE's area, tagged for (channel, epoch), count 0.
int chanShmAllocCredit(uint64_t channel, uint64_t epoch) {
  static thread_local int next = 0;
  ChanShmArea& a = chanShmArea(chanShmIndexOf(CkMyPe()));
  for (int tries = 0; tries < CHAN_SHM_CREDITS; tries++) {
    const int i = next; next = (next + 1) % CHAN_SHM_CREDITS;
    uint64_t zero = 0;
    if (a.credit[i].channel.compare_exchange_strong(zero, channel, std::memory_order_acq_rel)) {
      a.credit[i].count.store(0, std::memory_order_relaxed);
      a.credit[i].epoch.store(epoch, std::memory_order_release);
      return i;
    }
  }
  return -1;   // none free: this resolution's credits go by message
}

// Sender: the credits granted so far under (channel, epoch), or -1 when the
// slot no longer carries that resolution (freed, or reused by another).
long chanShmReadCredit(int area_pe, int slot, uint64_t channel, uint64_t epoch) {
  ChanShmCredit& c = chanShmArea(area_pe).credit[slot];
  const uint64_t ch1 = c.channel.load(std::memory_order_acquire);
  const uint64_t ep1 = c.epoch.load(std::memory_order_acquire);
  const uint64_t n = c.count.load(std::memory_order_acquire);
  const uint64_t ep2 = c.epoch.load(std::memory_order_acquire);
  const uint64_t ch2 = c.channel.load(std::memory_order_acquire);
  if (ch1 != channel || ch2 != channel || ep1 != epoch || ep2 != epoch) return -1;
  return (long)n;
}

// Senders of this PE with a put queued behind a shared-memory credit.
CkpvStaticDeclare(std::vector<CkDeviceChannel*>*, device_channel_shm_waiters);

// The note a put's completion writes into the owner PE's ring, or sends as a
// message when the ring is full.
struct DeviceChannelShmNote {
  int area_pe;
  CkDeviceChannelEvent ev;
  CkCallback arrived;
};
void deviceChannelShmNoteFn(void* param, void*) {
  auto* d = (DeviceChannelShmNote*)param;
  ChanShmArea& a = chanShmArea(d->area_pe);
  const uint64_t head = a.note_head.load(std::memory_order_relaxed);
  const uint64_t tail = a.note_tail.load(std::memory_order_acquire);
  if (head - tail < (uint64_t)(CHAN_SHM_NOTES - CHAN_SHM_NOTE_MARGIN)) {
    const uint64_t ticket = a.note_head.fetch_add(1, std::memory_order_acq_rel);
    ChanShmNote& e = a.note[ticket % CHAN_SHM_NOTES];
    e.ev = d->ev;
    memcpy(e.cb, &d->arrived, sizeof(CkCallback));
    e.ready.store(ticket + 1, std::memory_order_release);
  } else {
    CkDataMsg* note = CkDataMsg::buildNew((int)sizeof(d->ev), &d->ev);
    CkSetRefNum(note, (CMK_REFNUM_TYPE)d->ev.tag);
    d->arrived.send(note);
  }
  delete d;
}

// This PE's scheduler-loop poll: notes other PEs wrote for it, and credits
// its queued puts wait for.
void deviceChannelShmPoll(void*) {
  if (chan_shm_base == nullptr) return;
  ChanShmArea& a = chanShmArea(chanShmIndexOf(CkMyPe()));
  uint64_t tail = a.note_tail.load(std::memory_order_relaxed);
  if (tail != a.note_head.load(std::memory_order_acquire)) {
    int taken = 0;
    while (taken < 64) {
      ChanShmNote& e = a.note[tail % CHAN_SHM_NOTES];
      if (e.ready.load(std::memory_order_acquire) != tail + 1) break;   // not written yet
      CkDeviceChannelEvent ev = e.ev;
      CkCallback cb;
      memcpy(&cb, e.cb, sizeof(CkCallback));
      tail++;
      a.note_tail.store(tail, std::memory_order_release);
      CkDataMsg* note = CkDataMsg::buildNew((int)sizeof(ev), &ev);
      CkSetRefNum(note, (CMK_REFNUM_TYPE)ev.tag);
      cb.send(note);
      taken++;
    }
  }
  std::vector<CkDeviceChannel*>& w = *CkpvAccess(device_channel_shm_waiters);
  if (w.empty()) return;
  // pollShmCredit may issue the queued put, and issue() unlists the copy
  // (swap with the back, pop): walk a copy, then drop what no longer waits.
  std::vector<CkDeviceChannel*> walk(w);
  for (CkDeviceChannel* c : walk) c->pollShmCredit();
  for (size_t i = 0; i < w.size();) {
    if (!w[i]->putQueued()) { w[i] = w.back(); w.pop_back(); } else i++;
  }
}
void deviceChannelShmUnlist(CkDeviceChannel* c) {
  std::vector<CkDeviceChannel*>& w = *CkpvAccess(device_channel_shm_waiters);
  for (size_t i = 0; i < w.size(); i++) if (w[i] == c) { w[i] = w.back(); w.pop_back(); return; }
}
}  // namespace

/****************************** Watchdog ******************************/

// While the watchdog scans, describeStall tallies into this instead of
// printing every stuck copy (a PE can hold thousands).
struct DeviceChannelWatchTally {
  int s_inflight = 0, s_pending = 0, s_resolving = 0, s_closeack = 0;
  int o_waiting = 0, o_closing = 0, o_pulls = 0, shown = 0;
};
static thread_local DeviceChannelWatchTally* device_channel_watch_tally = nullptr;
static CkDeviceChannel* deviceChannelOwner(uint64_t channel);
struct DeviceChannelProbeCtx { int pe; uint64_t token, epoch, seq; };
static thread_local DeviceChannelProbeCtx* device_channel_probe_ctx = nullptr;

extern "C" void* device_channel_probe_bridge(void* arg) {
  auto* m = (DeviceChannelProbe*)arg;
  DeviceChannelProbeCtx ctx{m->pe, m->token, m->epoch, m->seq};
  device_channel_probe_ctx = &ctx;
  CkDeviceChannel* c = deviceChannelOwner(m->channel);
  if (c) c->describeStall("owner");
  int copies = 0;
  for (auto& kv : *CkpvAccess(device_channel_senders))
    if (kv.second->id() == m->channel) { copies++; kv.second->describeStall("sender-copy"); }
  device_channel_probe_ctx = nullptr;
  if (!c) {
    CmiPrintf("[CHAN-PROBE %d] channel %llx asked by pe %d token %llx: no owner registered here (%d sender copies of it here)\n",
              CkMyPe(), (unsigned long long)m->channel, m->pe, (unsigned long long)m->token, copies);
    if (m->owner_known) {
      CkLocMgr* mgr = (CkLocMgr*)CkLocalBranch(m->owner_mgr);
      CkLocRec* rec = mgr ? mgr->localRecord(m->owner_id) : NULL;
      CmiPrintf("[CHAN-PROBE %d] the owner element %llu is %s\n", CkMyPe(), (unsigned long long)m->owner_id,
                rec ? "here, but its channel is not registered: arrived without reopen, or packed and not yet gone"
                    : "not on this PE either");
    }
  }
  CmiFree(m);
  return NULL;
}

// Always armed, silent while channels move: after two 10 s ticks without a
// channel event on this PE, the copies that are waiting on something are
// described -- a summary and a few examples -- and again once a minute. A
// healthy run never prints.
static void deviceChannelWatch(void*) {
  const double wall = CkWallTimer();
  if (CkMyRank() < 2048) device_channel_last_tick[CkMyRank()] = wall;
  const uint64_t now = CkpvAccess(device_channel_progress);
  if (now != CkpvAccess(device_channel_progress_seen)) {
    CkpvAccess(device_channel_progress_seen) = now;
    CkpvAccess(device_channel_quiet_ticks) = 0;
    return;
  }
  const int q = ++CkpvAccess(device_channel_quiet_ticks);
  if (q < 2 || (q - 2) % 6 != 0) return;
  for (int r = 0; r < CkMyNodeSize() && r < 2048; r++) {
    const double t = device_channel_last_tick[r];
    if (r != CkMyRank() && t > 0.0 && wall - t > 25.0)
      CmiPrintf("[CHAN-WATCH %d] PE %d (rank %d of this process) has not run its watchdog for %.0f s: spinning or blocked\n",
                CkMyPe(), CkNodeFirst(CkMyNode()) + r, r, wall - t);
  }
  DeviceChannelWatchTally t;
  device_channel_watch_tally = &t;
  for (auto& kv : *CkpvAccess(device_channel_senders)) kv.second->describeStall("sender");
  for (auto& kv : *CkpvAccess(device_channel_owners)) kv.second->describeStall("owner");
  device_channel_watch_tally = nullptr;
  if (device_channel_credits_refused > 0)
    CmiPrintf("[CHAN-WATCH %d] credits refused by a copy here: %ld (last: %s)\n", CkMyPe(),
              device_channel_credits_refused, device_channel_last_refused);
  if (device_channel_resolved_older > 0)
    CmiPrintf("[CHAN-WATCH %d] resolutions ignored here as older than the one held: %ld (last: %s)\n", CkMyPe(),
              device_channel_resolved_older, device_channel_last_resolved_older);
  if (device_channel_credits_early > 0)
    CmiPrintf("[CHAN-WATCH %d] credits kept ahead of their resolution here: %ld, %zu still held (last: %s)\n", CkMyPe(),
              device_channel_credits_early, CkpvAccess(device_channel_early_credits)->size(), device_channel_last_early);
  if (device_channel_resolved_dropped > 0)
    CmiPrintf("[CHAN-WATCH %d] resolutions that found no sender copy here: %ld (last: %s)\n", CkMyPe(),
              device_channel_resolved_dropped, device_channel_last_resolved_dropped);
  if (device_channel_closing_acked_blind > 0)
    CmiPrintf("[CHAN-WATCH %d] closings acked for a copy no longer here: %ld (last: %s)\n", CkMyPe(),
              device_channel_closing_acked_blind, device_channel_last_closing_blind);
  if (device_channel_credits_dropped > 0) {
    const DeviceChannelCredit& d = device_channel_last_dropped_credit;
    CmiPrintf("[CHAN-WATCH %d] credits that found no sender copy here: %ld (last: channel %llx token %llx seq %llu epoch %llu from pe %d)\n",
              CkMyPe(), device_channel_credits_dropped, (unsigned long long)d.channel, (unsigned long long)d.token,
              (unsigned long long)d.seq, (unsigned long long)d.epoch, d.src_pe);
  }
  if (t.s_inflight + t.s_pending + t.s_resolving + t.s_closeack + t.o_waiting + t.o_closing + t.o_pulls == 0) return;
  CmiPrintf("[CHAN-WATCH %d] no channel event for %d s (%llu handled): senders in_flight %d, pending %d, "
            "resolving %d, close_ack_due %d; owners holding resolves %d, closing %d, announces queued %d "
            "(%zu senders, %zu owners here)\n",
            CkMyPe(), q * 10, (unsigned long long)now, t.s_inflight, t.s_pending, t.s_resolving, t.s_closeack,
            t.o_waiting, t.o_closing, t.o_pulls, CkpvAccess(device_channel_senders)->size(),
            CkpvAccess(device_channel_owners)->size());
}

static void deviceChannelStallScan(void*) {
  for (auto& kv : *CkpvAccess(device_channel_senders)) kv.second->describeStall("sender");
  for (auto& kv : *CkpvAccess(device_channel_owners)) kv.second->describeStall("owner");
}

static void deviceChannelBatchPoll(void*) { deviceChannelBatchFlush(); }

void CkDeviceChannelInit(char** argv) {
  if (CmiGetArgFlagDesc(argv, "+gpupersistverbose",
                        "print the mode of CkDyn's persistent device transfers, once per process and mode"))
    endpoint_verbose = 1;
  CkpvInitialize(DeviceChannelTable*, device_channel_senders);
  CkpvAccess(device_channel_senders) = new DeviceChannelTable();
  CkpvInitialize(DeviceChannelTable*, device_channel_owners);
  CkpvAccess(device_channel_owners) = new DeviceChannelTable();
  CkpvInitialize(uint64_t, device_channel_counter);
  CkpvAccess(device_channel_counter) = 0;
  CkpvInitialize(DeviceChannelEarlyCredits*, device_channel_early_credits);
  CkpvAccess(device_channel_early_credits) = new DeviceChannelEarlyCredits();
  CkpvInitialize(DeviceChannelStagedTable*, device_channel_staged);
  CkpvAccess(device_channel_staged) = new DeviceChannelStagedTable();
  CkpvInitialize(DeviceChannelBatch*, device_channel_batch);
  CkpvAccess(device_channel_batch) = nullptr;
  device_channel_batch_ready = true;
  CcdCallOnConditionKeep(CcdSCHEDLOOP, deviceChannelBatchPoll, NULL);
  CcdCallOnConditionKeep(CcdPROCESSOR_BEGIN_IDLE, deviceChannelBatchPoll, NULL);
  CcdCallOnConditionKeep(CcdPROCESSOR_STILL_IDLE, deviceChannelBatchPoll, NULL);
  CkpvInitialize(uint64_t, device_channel_progress);
  CkpvAccess(device_channel_progress) = 0;
  CkpvInitialize(uint64_t, device_channel_progress_seen);
  CkpvAccess(device_channel_progress_seen) = 0;
  CkpvInitialize(int, device_channel_quiet_ticks);
  CkpvAccess(device_channel_quiet_ticks) = 0;
  if (deviceChannelDebug()) CcdCallOnConditionKeep(CcdPERIODIC_10seconds, deviceChannelStallScan, NULL);
  CcdCallOnConditionKeep(CcdPERIODIC_10seconds, deviceChannelWatch, NULL);
  CkpvInitialize(std::vector<CkDeviceChannel*>*, device_channel_shm_waiters);
  CkpvAccess(device_channel_shm_waiters) = new std::vector<CkDeviceChannel*>();
  CkpvInitialize(bool, device_channel_late_init_done);
  CkpvAccess(device_channel_late_init_done) = false;
  // The host topology (which PEs share a host) is computed after this runs
  // (init.C: CmiInitCPUTopology comes later), so the shared-memory segment and
  // its poll wait for the first use: deviceChannelLateInit. Run before the
  // topology, "same host" would silently mean "same process".
}

// Per PE, at the first channel use, once the host topology is known: the
// shared-memory segment and its poll when several processes share this host.
static void deviceChannelLateInit() {
  if (CkpvAccess(device_channel_late_init_done)) return;
  CkpvAccess(device_channel_late_init_done) = true;
  const int host = CmiPhysicalNodeID(CkMyPe());
  if (CmiNumPesOnPhysicalNode(host) > CkMyNodeSize()) {
    std::call_once(chan_shm_once, chanShmMap);
    CcdCallOnConditionKeep(CcdSCHEDLOOP, deviceChannelShmPoll, NULL);
  }
}

// Unique on this PE and distinct from every other PE's: channel ids and sender
// tokens both come from it.
static uint64_t deviceChannelToken() {
  return ((uint64_t)CkMyPe() << 40) | (++CkpvAccess(device_channel_counter));
}

static CkDeviceChannel* deviceChannelSender(uint64_t token) {
  DeviceChannelTable& tab = *CkpvAccess(device_channel_senders);
  auto it = tab.find(token);
  return it == tab.end() ? NULL : it->second;
}

static CkDeviceChannel* deviceChannelOwner(uint64_t channel) {
  DeviceChannelTable& tab = *CkpvAccess(device_channel_owners);
  auto it = tab.find(channel);
  return it == tab.end() ? NULL : it->second;
}

// A protocol message: allocated, addressed to its handler, counted for
// quiescence (the handler releases the count).
template <class M>
static M* deviceChannelMsg(int handler, size_t extra = 0) {
  M* m = (M*)CmiAlloc(sizeof(M) + extra);
  CmiEnforce(m);
  CmiSetHandler(m, handler);
  QdCreate(1);
  return m;
}

/****************************** Converse handlers ******************************/

extern "C" void* device_channel_credit_bridge(void* arg) {
  QdProcess(1);
  auto* m = (DeviceChannelCredit*)arg;
  if (m->dst_pe != CkMyPe())
    CmiPrintf("[CHAN-MISDELIVERED %d] credit for channel %llx token %llx epoch %llu from pe %d was addressed to pe %d\n",
              CkMyPe(), (unsigned long long)m->channel, (unsigned long long)m->token, (unsigned long long)m->epoch,
              m->src_pe, m->dst_pe);
  CkDeviceChannel* c = deviceChannelSender(m->token);
  // A copy that left this PE re-resolves before its next put and learns the
  // buffer's state then, so a credit that finds nobody is simply dropped.
  if (c) c->onCredit(m->seq, m->epoch);
  else { device_channel_credits_dropped++; device_channel_last_dropped_credit = *m; }
  CmiFree(m);
  return NULL;
}

extern "C" void* device_channel_closing_bridge(void* arg) {
  QdProcess(1);
  auto* m = (DeviceChannelClosing*)arg;
  CkDeviceChannel* c = deviceChannelSender(m->token);
  if (c) {
    c->onClosing(m->generation, m->owner_pe, m->epoch);
  } else {
    // The holder left this PE (its chare migrated or died), so nothing of it is
    // in flight from here: acknowledge for it, or the receiver never moves.
    auto* a = deviceChannelMsg<DeviceChannelCloseAck>(device_channel_closeack_handler);
    a->channel = m->channel;
    a->token = m->token;
    a->pe = CkMyPe();
    a->last_seq = UINT64_MAX;   // nobody here knows what that copy last put
    a->last_bytes = 0;
    chanSendControl(m->owner_pe, sizeof(DeviceChannelCloseAck), a);
    device_channel_closing_acked_blind++;
    snprintf(device_channel_last_closing_blind, sizeof(device_channel_last_closing_blind),
             "channel %llx token %llx epoch %llu gen %llu from pe %d",
             (unsigned long long)m->channel, (unsigned long long)m->token, (unsigned long long)m->epoch,
             (unsigned long long)m->generation, m->owner_pe);
  }
  CmiFree(m);
  return NULL;
}

extern "C" void* device_channel_closeack_bridge(void* arg) {
  QdProcess(1);
  auto* m = (DeviceChannelCloseAck*)arg;
  CkDeviceChannel* c = deviceChannelOwner(m->channel);
  if (c) c->onCloseAck(m->pe, m->token, m->last_seq, m->last_bytes);
  CmiFree(m);
  return NULL;
}

extern "C" void* device_channel_done_bridge(void* arg) {
  QdProcess(1);
  auto* m = (DeviceChannelDone*)arg;
  DeviceChannelStagedTable& st = *CkpvAccess(device_channel_staged);
  auto it = st.find(std::make_pair(m->token, m->seq));
  if (it != st.end()) {
    // A staged put was read: its buffer goes back, the sending element is no
    // longer held by it, and its channel copy (if still here) rebalances.
    DeviceChannelStaged d = it->second;
    st.erase(it);
    d.ep.close();
    if (d.buf) CkDeviceFree(d.buf);
    if (d.holder_known) {
      CkLocMgr* mgr = (CkLocMgr*)CkLocalBranch(d.holder_mgr);
      CkLocRec* rec = mgr ? mgr->localRecord(d.holder_elem) : NULL;
      if (rec) rec->noteChannelWorkDone();
    }
    CkDeviceChannel* c = deviceChannelSender(m->token);
    if (c) c->onStagedDone(m->seq);
  } else {
    CkDeviceChannel* c = deviceChannelSender(m->token);
    if (c) c->onDone(m->seq);
  }
  CmiFree(m);
  return NULL;
}

// The announce of a put, sent once the source is complete on the sender's stream.
struct DeviceChannelAnnounce { CkCallback to; CkDataMsg* msg; };
static void deviceChannelAnnounceReadyFn(void* param, void*) {
  auto* a = (DeviceChannelAnnounce*)param;
  a->to.send(a->msg);
  delete a;
}
// The receiver's read of an announced put completed (runs on the receiver PE).
struct DeviceChannelPulled { uint64_t channel; };
static void deviceChannelPulledFn(void* param, void* msg) {
  if (msg) CkFreeMsg(msg);
  auto* d = (DeviceChannelPulled*)param;
  CkDeviceChannel* c = deviceChannelOwner(d->channel);
  if (c) c->onPulled();
  else CmiPrintf("[CHAN %d] a read completed for channel %llx, which is no longer open here\n", CkMyPe(), (unsigned long long)d->channel);
  delete d;
}

extern "C" void* device_channel_resolved_bridge(void* arg) {
  QdProcess(1);
  auto* m = (DeviceChannelResolved*)arg;
  if (m->dst_pe != CkMyPe())
    CmiPrintf("[CHAN-MISDELIVERED %d] resolution %llu of channel %llx token %llx from pe %d was addressed to pe %d\n",
              CkMyPe(), (unsigned long long)m->epoch, (unsigned long long)m->channel, (unsigned long long)m->token,
              m->src_pe, m->dst_pe);
  CkDeviceChannel* c = deviceChannelSender(m->token);
  if (c) {
    CkDeviceChannelEndpoint ep;
    PUP::fromMem p((char*)m + sizeof(DeviceChannelResolved));
    p | ep;
    c->onResolved(ep, m->generation, m->owner_pe, m->free != 0, m->epoch, m->credit_pe, m->credit_slot,
                  m->taken_seq);
  } else {
    device_channel_resolved_dropped++;
    snprintf(device_channel_last_resolved_dropped, sizeof(device_channel_last_resolved_dropped),
             "channel %llx token %llx epoch %llu gen %llu free %d from pe %d",
             (unsigned long long)m->channel, (unsigned long long)m->token, (unsigned long long)m->epoch,
             (unsigned long long)m->generation, (int)m->free, m->src_pe);
    if (deviceChannelDebug())
      CmiPrintf("[CHAN-DROP %d] resolution with no sender copy here: %s\n", CkMyPe(), device_channel_last_resolved_dropped);
  }
  CmiFree(m);
  return NULL;
}

// A put's bytes on the sending element's edge to the receiver, for the
// balancers' communication graph (what CkArray::recordSend does for a
// message). The destination key is the OWNER's location manager and element:
// keyed with the sender's manager, a put names an unrelated element of the
// sender's array that happens to share the id.
static void deviceChannelRecordSend(CkLocRec* rec, size_t bytes, int dest_pe, bool owner_known,
                                    CkGroupID owner_mgr, CmiUInt8 dest_elem) {
#if CMK_LBDB_ON
  if (rec == NULL || !owner_known) return;
  CkLocMgr* ownerMgr = (CkLocMgr*)CkLocalBranch(owner_mgr);
  if (ownerMgr == NULL || ownerMgr->getLBMgr() == NULL || !ownerMgr->getLBMgr()->CollectingCommStats()) return;
#  if CMK_GLOBAL_LOCATION_UPDATE
  const CmiUInt8 lbObjId = ownerMgr->makeID(dest_elem).getID();
#  else
  const CmiUInt8 lbObjId = dest_elem;
#  endif
  ownerMgr->getLBMgr()->Send(ownerMgr->getOMHandle(), lbObjId, (unsigned int)bytes, dest_pe, 1);
#endif
}

// Sender-side completion of a put: the source may be reused. Runs on the
// sender's PE. The copy is found by token, never by pointer: it may have been
// assigned elsewhere since.
struct DeviceChannelSrcDone {
  uint64_t token;
  uint64_t seq;
  CkCallback user;
};

static void deviceChannelSrcDoneFn(void* param, void* msg) {
  auto* d = (DeviceChannelSrcDone*)param;
  CkDeviceChannel* c = deviceChannelSender(d->token);
  if (c) c->onSrcDone(d->seq);
  if (d->user.type != CkCallback::ignore) d->user.send(msg);
  delete d;
}

/****************************** Batched copies ******************************/
//
// One copy and one completion flag per put cost the copy engine more than the
// zero-copy path's batched landings (leanmd: 41 ms per 2.5 s step on the GPU
// that set the step). The puts a PE issues on one producing stream within one
// entry method are one batched copy on that stream with one completion
// callback, which delivers every put's note and source-done.
static void deviceChannelBatchDoneFn(void* param, void* msg) {
  auto* b = (DeviceChannelBatch*)param;
  for (const DeviceChannelBatchEntry& e : b->entries) {
    CkDeviceChannel* c = deviceChannelSender(e.token);
    if (c) c->onSrcDone(e.seq);
    if (e.user.type != CkCallback::ignore) e.user.send(msg);
    e.note.send(e.note_msg);
  }
  delete b;
}

static void deviceChannelBatchFlush() {
  DeviceChannelBatch*& cur = CkpvAccess(device_channel_batch);
  if (cur == nullptr) return;
  DeviceChannelBatch* b = cur;
  cur = nullptr;
  const size_t n = b->entries.size();
  hapiStream_t st = b->stream;
  hapiError_t err = hapiErrorNotSupported;
#if CMK_CUDA && defined(CUDART_VERSION) && CUDART_VERSION >= 12080
  if (n > 1) {
    cudaMemcpyAttributes attr{};
    attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    size_t attr_idx = 0;
#if CUDART_VERSION >= 13000
    err = cudaMemcpyBatchAsync(b->dsts.data(), b->srcs.data(), b->sizes.data(), n, &attr, &attr_idx, 1, st);
#else
    size_t fail_idx = 0;
    err = cudaMemcpyBatchAsync(b->dsts.data(), b->srcs.data(), b->sizes.data(), n, &attr, &attr_idx, 1, &fail_idx, st);
#endif
  }
#endif
  if (err != hapiSuccess) {
    if (n > 1) (void)hapiGetLastError();
    // hapiMemcpyDefault: a destination may sit on another device (a peer's
    // buffer through its IPC mapping, or a receiver on another GPU).
    for (size_t i = 0; i < n; i++)
      hapiCheck(hapiMemcpyAsync(b->dsts[i], b->srcs[i], b->sizes[i], hapiMemcpyDefault, st));
  }
  hapiAddCallback(st, CkCallback(deviceChannelBatchDoneFn, (void*)b));
}

static void deviceChannelBatchAdd(hapiStream_t stream, void* dst, const void* src, size_t bytes,
                                  const DeviceChannelBatchEntry& e) {
  DeviceChannelBatch*& cur = CkpvAccess(device_channel_batch);
  if (cur != nullptr && cur->stream != stream) deviceChannelBatchFlush();   // one producing stream per batch
  if (cur == nullptr) {
    cur = new DeviceChannelBatch;
    cur->stream = stream;
    cur->in_entry = (chanActiveRec() != NULL);
    cur->dsts.reserve(16); cur->srcs.reserve(16); cur->sizes.reserve(16); cur->entries.reserve(16);
  }
  cur->dsts.push_back(dst);
  cur->srcs.push_back(src);
  cur->sizes.push_back(bytes);
  cur->entries.push_back(e);
  if (cur->entries.size() >= kChanBatchMax) deviceChannelBatchFlush();
}

// From CkCallstackPop: the batch's copies are launched while the element that
// made the puts is still the running object, so per-object device load
// accounting keeps them. Only a batch opened inside an element's entry method:
// one opened from a Converse handler (a credit or a resolution issuing a
// queued put) goes out from the scheduler loop.
void CkDeviceChannelBatchFlush() {
  if (!device_channel_batch_ready) return;
  DeviceChannelBatch* cur = CkpvAccess(device_channel_batch);
  if (cur != nullptr && cur->in_entry) deviceChannelBatchFlush();
}

/****************************** CkDeviceChannel ******************************/

CkDeviceChannel::CkDeviceChannel()
  : arrived(CkCallback(CkCallback::ignore)), control(CkCallback(CkCallback::ignore)),
    channel_id(0), generation(0), cap(0),
    is_owner(false), owner_id(0), owner_known(false), busy(false), taken_seq(0),
    credit_slot_pe(-1), credit_slot(-1), credit_shm_pe(-1), credit_shm_slot(-1), credits_seen(0), shm_waiting(false),
    landed_unseen(0), landed_seq(0), last_put_bytes(0), last_direct_seq(0), sender_pe(-1), sender_token(0),
    sender_known(false), closing_gen(0), closing_sent(false), sender_closed(false), pulling(false),
    resolved_during_closing(false), epoch(0), closing_rec(NULL), have_last(false),
    resolved(false), resolving(false), send_seq(0), credits(0), uncredited(0), pull_mode(false), tier(0),
    closed_epoch(0), in_flight(0), staged_outstanding(0), acct_seq(0), occupant_credit_due(false),
    close_ack_due(false), close_ack_pe(-1), my_token(0), sender_rec(NULL), holder_elem(0), holder_known(false),
    src_ep_last(0) {
  pending.valid = false;
  memset(&last, 0, sizeof(last));
  memset(&pull_ev, 0, sizeof(pull_ev));
}

CkDeviceChannel::CkDeviceChannel(void* dst_, size_t capacity, const CkCallback& arrived_)
  : CkDeviceChannel() {
  dst = CkDeviceChannelEndpoint(dst_, capacity);
  arrived = arrived_;
  cap = capacity;
}

// A copy takes over the original's table entries, its ownership of the
// receive buffer, and the runtime resources it holds (IPC views, registered
// sources): the usual shape is `peer = param;` in an entry method, after which
// the parameter dies.
void CkDeviceChannel::takeOver(const CkDeviceChannel& c) {
  CkDeviceChannel& o = const_cast<CkDeviceChannel&>(c);
  memcpy(hist, o.hist, sizeof(hist)); hist_n = o.hist_n;
  if (o.my_token != 0) (*CkpvAccess(device_channel_senders))[o.my_token] = this;
  if (o.is_owner && o.channel_id != 0) (*CkpvAccess(device_channel_owners))[o.channel_id] = this;
  o.is_owner = false;
  dst.takeView(o.dst);
  pull_src.takeView(o.pull_src);
  src_ep = o.src_ep; o.src_ep = CkDeviceChannelEndpoint();
  retired_srcs.swap(o.retired_srcs);
  for (CkDeviceChannelEndpoint& v : o.stale_views) {
    stale_views.push_back(v);
    stale_views.back().takeView(v);
  }
  o.stale_views.clear();
}

CkDeviceChannel::CkDeviceChannel(const CkDeviceChannel& o)
  : dst(o.dst), arrived(o.arrived), control(o.control), channel_id(o.channel_id), generation(o.generation),
    cap(o.cap), is_owner(o.is_owner), owner_id(o.owner_id), owner_known(o.owner_known), owner_mgr(o.owner_mgr),
    busy(o.busy), taken_seq(o.taken_seq),
    credit_slot_pe(o.credit_slot_pe), credit_slot(o.credit_slot), credit_shm_pe(o.credit_shm_pe),
    credit_shm_slot(o.credit_shm_slot), credits_seen(o.credits_seen), shm_waiting(false),
    landed_unseen(o.landed_unseen), landed_seq(o.landed_seq), last_put_bytes(o.last_put_bytes),
    last_direct_seq(o.last_direct_seq), sender_pe(o.sender_pe), sender_token(o.sender_token),
    sender_known(o.sender_known), closing_gen(o.closing_gen), closing_sent(o.closing_sent),
    sender_closed(o.sender_closed), pulls(o.pulls), pulling(o.pulling), pull_ev(o.pull_ev), pull_src(o.pull_src),
    resolved_during_closing(o.resolved_during_closing), epoch(o.epoch), closing_rec(o.closing_rec),
    waiting(o.waiting), last(o.last), have_last(o.have_last), resolved(o.resolved), resolving(o.resolving),
    send_seq(o.send_seq), credits(o.credits), uncredited(o.uncredited), pull_mode(o.pull_mode), tier(o.tier),
    done_cbs(o.done_cbs), closed_epoch(o.closed_epoch), in_flight(o.in_flight),
    staged_outstanding(o.staged_outstanding), acct_seq(o.acct_seq), occupant_credit_due(o.occupant_credit_due),
    direct_issued(o.direct_issued), dones_seen(o.dones_seen), close_ack_due(o.close_ack_due),
    close_ack_pe(o.close_ack_pe), my_token(o.my_token), sender_rec(o.sender_rec), holder_elem(o.holder_elem),
    holder_mgr(o.holder_mgr), holder_known(o.holder_known), pending(o.pending), src_ep_last(o.src_ep_last) {
  takeOver(o);
}

CkDeviceChannel& CkDeviceChannel::operator=(const CkDeviceChannel& o) {
  if (this == &o) return *this;
  retire();   // what this object held goes first, as its destructor would release it
  dst = o.dst; arrived = o.arrived; control = o.control; channel_id = o.channel_id; generation = o.generation;
  cap = o.cap; is_owner = o.is_owner; owner_id = o.owner_id; owner_known = o.owner_known; owner_mgr = o.owner_mgr;
  busy = o.busy; taken_seq = o.taken_seq;
  credit_slot_pe = o.credit_slot_pe; credit_slot = o.credit_slot; credit_shm_pe = o.credit_shm_pe;
  credit_shm_slot = o.credit_shm_slot; credits_seen = o.credits_seen;
  landed_unseen = o.landed_unseen; landed_seq = o.landed_seq; last_put_bytes = o.last_put_bytes;
  last_direct_seq = o.last_direct_seq;
  sender_pe = o.sender_pe; sender_token = o.sender_token; sender_known = o.sender_known;
  closing_gen = o.closing_gen; closing_sent = o.closing_sent; sender_closed = o.sender_closed;
  pulls = o.pulls; pulling = o.pulling; pull_ev = o.pull_ev; pull_src = o.pull_src;
  resolved_during_closing = o.resolved_during_closing; epoch = o.epoch; closing_rec = o.closing_rec;
  waiting = o.waiting; last = o.last; have_last = o.have_last;
  resolved = o.resolved; resolving = o.resolving; send_seq = o.send_seq;
  credits = o.credits; uncredited = o.uncredited; pull_mode = o.pull_mode; tier = o.tier; done_cbs = o.done_cbs;
  closed_epoch = o.closed_epoch; in_flight = o.in_flight; staged_outstanding = o.staged_outstanding;
  acct_seq = o.acct_seq; occupant_credit_due = o.occupant_credit_due; direct_issued = o.direct_issued;
  dones_seen = o.dones_seen; close_ack_due = o.close_ack_due; close_ack_pe = o.close_ack_pe;
  my_token = o.my_token; sender_rec = o.sender_rec;
  holder_elem = o.holder_elem; holder_mgr = o.holder_mgr; holder_known = o.holder_known;
  pending = o.pending; src_ep_last = o.src_ep_last;
  takeOver(o);
  return *this;
}

CkDeviceChannel::~CkDeviceChannel() { retire(); }

// What this object holds of the runtime: its table entries, the owner's
// export and registration, its IPC views and registered sources.
void CkDeviceChannel::retire() {
  if (shm_waiting) { deviceChannelShmUnlist(this); shm_waiting = false; }
  if (is_owner) retireCreditSlot();
  if (my_token != 0) {
    DeviceChannelTable* tab = CkpvAccess(device_channel_senders);
    auto it = tab->find(my_token);
    if (it != tab->end() && it->second == this) tab->erase(it);
    CkpvAccess(device_channel_early_credits)->erase(my_token);
    my_token = 0;
  }
  if (is_owner && channel_id != 0) {
    unregisterOwner();
    dst.close();   // the owner's export and registration go with it
  }
  is_owner = false;
  // A view or a source still in use by a copy in flight is left as it is:
  // closing a mapping or deregistering memory under a transfer is worse.
  if (in_flight == 0) {
    dst.releaseView();
    for (const CkDeviceChannelEndpoint& v : stale_views) v.releaseView();
  }
  stale_views.clear();
  if (!pulling) pull_src.releaseView();
  closeSources();
}

void CkDeviceChannel::registerOwner() {
  (*CkpvAccess(device_channel_owners))[channel_id] = this;
}

void CkDeviceChannel::unregisterOwner() {
  DeviceChannelTable* tab = CkpvAccess(device_channel_owners);
  auto it = tab->find(channel_id);
  if (it != tab->end() && it->second == this) tab->erase(it);
}

void CkDeviceChannel::open() {
  if (dst.ptr == nullptr || dst.cnt == 0) {
    CkAbort("CkDeviceChannel::open: no destination buffer; construct the channel with one\n");
  }
  if (arrived.type == CkCallback::ignore) {
    CkAbort("CkDeviceChannel::open: no arrival callback; a channel needs an entry that learns of "
            "each put\n");
  }
  if (channel_id == 0) channel_id = deviceChannelToken();
  generation = 1;
  cap = dst.cnt;
  // The receiving element, for the migration hook. NULL outside an element's
  // entry method (a group, or a call from the main chare): the channel then has
  // no owner to stand down, which is right for a chare that never migrates.
  CkLocRec* rec = chanActiveRec();
  owner_id = rec ? rec->getID() : 0;
  owner_known = (rec != NULL);
  if (rec) owner_mgr = rec->getLocMgr()->ckGetGroupID();
  // Runtime control events (a sender's resolve, an announce) go to the element
  // itself, not through the application's entry: an SDAG `when arrived[step]`
  // would hold them forever. For an array element that is its
  // ckDeviceChannelControl entry; anything else (a group) gets them on
  // `arrived`, where event() hands the application NULL.
  if (rec && arrived.type == CkCallback::sendArray) {
    control = CkCallback(CkIndex_ArrayElement::ckDeviceChannelControl(NULL), rec->getIndex(),
                         CkArrayID(arrived.d.array.id));
  } else {
    control = arrived;
  }
  if (deviceChannelDebug())
    CmiPrintf("[CHAN-OPEN %d] id=%llx owner_id=%llu obj=%p\n", CkMyPe(), (unsigned long long)channel_id,
              (unsigned long long)owner_id, (void*)this);
  dst.open();
  is_owner = true;
  registerOwner();
}

void CkDeviceChannel::reopen(void* new_dst) {
  if (channel_id == 0 || cap == 0) {
    CkAbort("CkDeviceChannel::reopen: the channel was never open()ed\n");
  }
  dst = CkDeviceChannelEndpoint(new_dst, cap);
  generation++;
  dst.open();
  is_owner = true;
  // Every sender of the previous home was closed; each re-resolves before its next put.
  sender_known = false;
  closing_gen = 0;
  closing_sent = false;
  resolved_during_closing = false;
  sender_closed = false;
  closing_rec = NULL;
  registerOwner();
  // Senders that asked while the element was on its way.
  std::vector<Waiting> w;
  w.swap(waiting);
  for (const Waiting& x : w) replyResolve(x.pe, x.token);
  // Announces that travelled with the element: read into the new buffer.
  if (!busy && !pulling) startPull();
}

void CkDeviceChannel::close() {
  if (!is_owner) return;
  retireCreditSlot();
  dst.close();
  unregisterOwner();
  is_owner = false;
}

const CkDeviceChannelEvent* CkDeviceChannel::event(CkDataMsg* m) {
  if (m == NULL || m->getLength() < (int)sizeof(CkDeviceChannelEvent)) {
    CkAbort("CkDeviceChannel::event: the message is not a channel event (%d bytes, %zu expected)\n",
            m ? m->getLength() : -1, sizeof(CkDeviceChannelEvent));
  }
  CkDeviceChannelEvent ev;
  memcpy(&ev, m->getData(), sizeof(ev));
  if (ev.kind == CkDeviceChannelEvent::ANNOUNCE) {
    // The sender's source follows the event; this side reads it when the
    // buffer is free. The announce followed the element here if it moved, and
    // the source stays valid until this side says DONE.
    CkDeviceChannelEndpoint src;
    PUP::fromMem p((const char*)m->getData() + sizeof(ev));
    p | src;
    onAnnounce(ev, src);
    return NULL;
  }
  if (ev.channel != channel_id) {
    CkAbort("CkDeviceChannel::event: event for channel %llx taken on channel %llx\n",
            (unsigned long long)ev.channel, (unsigned long long)channel_id);
  }
  if (ev.kind == CkDeviceChannelEvent::RESOLVE) {
    if (!is_owner) {
      // Packed and on the way to another PE: answered after reopen there.
      Waiting w; w.pe = ev.sender_pe; w.token = ev.sender_token;
      waiting.push_back(w);
      return NULL;
    }
    // Answered even while a migration is pending: the buffer is valid until the
    // pack, and holding the sender would hold the element's own step, which is
    // what it must finish before it can move. The sender is then closed at
    // once, so the stand-down covers its puts too; it re-resolves after the move.
    replyResolve(ev.sender_pe, ev.sender_token);
    if (closing_gen == generation && closing_rec != NULL) closingForMigration(closing_rec);
    return NULL;
  }
  if (ev.generation != generation && deviceChannelDebug())
    CmiPrintf("[CHAN-STALE %d] channel %llx: put %llu into generation %llu taken at generation %llu (its bytes came with the pup)\n",
              CkMyPe(), (unsigned long long)channel_id, (unsigned long long)ev.seq,
              (unsigned long long)ev.generation, (unsigned long long)generation);
  last = ev;
  have_last = true;
  busy = true;
  taken_seq = ev.seq;
  note(10, ev.seq, epoch, ev.sender_pe, -1, (int)sender_known);
  landed_unseen = 0;
  landed_seq = 0;
  return &last;
}

// The current endpoint to a sender; that sender is the one credits and the
// closing go to from now on.
void CkDeviceChannel::replyResolve(int pe, uint64_t token) {
  deviceChannelProgress();
  deviceChannelLateInit();
  sender_pe = pe;
  sender_token = token;
  sender_known = true;
  sender_closed = false;
  // A closing on its way to the previous resolution may be acked after this
  // reply is out (the sender's request crossed its ack): the ack then closes
  // this resolution too, before the element may move.
  if (closing_sent) resolved_during_closing = true;
  epoch++;
  // The previous resolution's credit slot: a sender reads it only while that
  // resolution is current for it, and checks channel and epoch around the
  // read, so it can go back now.
  retireCreditSlot();
  if (chanShmPair(pe)) {
    credit_slot = chanShmAllocCredit(channel_id, epoch);
    credit_slot_pe = (credit_slot >= 0) ? chanShmIndexOf(CkMyPe()) : -1;
  }
  note(12, taken_seq, epoch, pe, credit_slot, (int)busy);
  if (deviceChannelDebug())
    CmiPrintf("[CHAN-RESOLVE %d] channel %llx gen %llu: resolution %llu to pe %d token %llx (free=%d, closing in flight=%d, pulling=%d taken_seq=%llu pulls=%zu)\n",
              CkMyPe(), (unsigned long long)channel_id, (unsigned long long)generation, (unsigned long long)epoch,
              pe, (unsigned long long)token, (int)!busy, (int)closing_sent, (int)pulling, (unsigned long long)taken_seq, pulls.size());
  PUP::sizer sz;
  sz | dst;
  const size_t n = sz.size();
  auto* m = deviceChannelMsg<DeviceChannelResolved>(device_channel_resolved_handler, n);
  m->token = token;
  m->channel = channel_id;
  m->generation = generation;
  m->owner_pe = CkMyPe();
  m->epoch = epoch;
  m->free = busy ? 0 : 1;
  // The last of the sender's puts to have entered the buffer: taken_seq once
  // event() took it; before that, an announced put read (pull_ev; onPulled set
  // busy and sent its DONE) or a direct put that landed before the pack and
  // came with the pup (landed_seq, from the close ack). The sender counts only
  // puts after this one as still on their way.
  m->taken_seq = taken_seq;
  if (busy && !have_last) {
    if (landed_unseen > 0 && landed_unseen != SIZE_MAX) { if (landed_seq > m->taken_seq) m->taken_seq = landed_seq; }
    else if (pull_ev.seq > m->taken_seq) m->taken_seq = pull_ev.seq;
  }
  m->desc_bytes = (int)n;
  m->dst_pe = pe;
  m->src_pe = CkMyPe();
  m->credit_pe = credit_slot_pe;
  m->credit_slot = credit_slot;
  PUP::toMem tm((char*)m + sizeof(DeviceChannelResolved));
  tm | dst;
  chanSendControl(pe, (int)(sizeof(DeviceChannelResolved) + n), m);
}

void CkDeviceChannel::retireCreditSlot() {
  if (credit_slot < 0) return;
  ChanShmCredit& c = chanShmArea(credit_slot_pe).credit[credit_slot];
  c.epoch.store(0, std::memory_order_release);
  c.channel.store(0, std::memory_order_release);
  credit_slot = -1;
  credit_slot_pe = -1;
}

// Sender: credits granted through shared memory since the last look. A put
// queued behind one is issued here.
void CkDeviceChannel::pollShmCredit() {
  if (credit_shm_slot < 0 || !resolved) return;
  const long n = chanShmReadCredit(credit_shm_pe, credit_shm_slot, channel_id, epoch);
  if (n < 0 || (uint32_t)n <= credits_seen) return;
  const uint32_t fresh = (uint32_t)n - credits_seen;
  if (deviceChannelDebug())
    CmiPrintf("[CHAN-CREDIT-RX %d] channel %llx token %llx: %u shm credit(s) from slot %d of area %d under epoch %llu: credits %d pending %d\n",
              CkMyPe(), (unsigned long long)channel_id, (unsigned long long)my_token, fresh, credit_shm_slot, credit_shm_pe,
              (unsigned long long)epoch, credits, (int)pending.valid);
  credits_seen = (uint32_t)n;
  note(6, 0, epoch, credit_shm_pe, credit_shm_slot, (int)fresh);
  deviceChannelProgress();
  for (uint32_t i = 0; i < fresh; i++) {
    credits++;
    if (uncredited > 0) uncredited--;
    creditedPut(0);   // the slot carries no sequence: credits come in put order
  }
  if (pending.valid && credits > 0) {
    Pending p = pending;
    pending.valid = false;
    credits--;
    uncredited++;
    issue(p.src, p.bytes, p.stream, p.done, p.tag);
  }
}

void CkDeviceChannel::release() {
  deviceChannelProgress();
  if (!have_last) {
    printHist(is_owner ? "owner" : "copy");
    CkAbort("CkDeviceChannel::release: nothing to release; event() records the put first\n");
  }
  have_last = false;
  busy = false;
  // The next announced put, if one waits and is next in order.
  if (!pulls.empty()) startPull();
  // Every release credits the current sender, after a read put as after a
  // direct one: the sender debits a read staged put at its DONE
  // (onStagedDone), so the two pair off and the balance stays one buffer. No
  // resolved sender (it re-resolves after our migration and learns the buffer
  // is free then): nothing to credit.
  if (!sender_known) return;
  note(11, last.seq, epoch, sender_pe, credit_slot, 0);
  if (deviceChannelDebug())
    CmiPrintf("[CHAN-CREDIT %d] channel %llx: release of put %llu credits pe %d token %llx under epoch %llu via %s\n",
              CkMyPe(), (unsigned long long)channel_id, (unsigned long long)last.seq, sender_pe, (unsigned long long)sender_token,
              (unsigned long long)epoch, credit_slot >= 0 ? "shm" : "msg");
  if (credit_slot >= 0) {
    // Same host, other process: one shared-memory increment, no message.
    chanShmArea(credit_slot_pe).credit[credit_slot].count.fetch_add(1, std::memory_order_acq_rel);
    return;
  }
  auto* m = deviceChannelMsg<DeviceChannelCredit>(device_channel_credit_handler);
  m->token = sender_token;
  m->channel = channel_id;
  m->seq = last.seq;
  m->epoch = epoch;
  m->dst_pe = sender_pe;
  m->src_pe = CkMyPe();
  chanSendControl(sender_pe, sizeof(DeviceChannelCredit), m);
}

// Migration of the receiving element (from CkLocMgr::emigrate): the resolved
// sender is told the channel is closing and the element is held by one unit
// of channel work until the ack. Once per generation.
void CkDeviceChannel::closingForMigration(CkLocRec* rec) {
  if (!is_owner) return;
  note(13, taken_seq, epoch, sender_pe, -1, (int)sender_known);
  // The pending migration is noted on every owned channel, resolved sender or
  // not: a sender that resolves while the element stands down (for another
  // channel's ack) is closed at once from event(), or its put could land after
  // the pack.
  closing_gen = generation;
  closing_rec = rec;
  if (!sender_known || sender_closed || closing_sent) return;
  closing_sent = true;
  resolved_during_closing = false;
  rec->noteChannelWorkPosted();
  if (deviceChannelDebug())
    CmiPrintf("[CHAN-CLOSE %d] channel %llx gen %llu: closing resolution %llu sent to pe %d token %llx; element %llu work now %d\n",
              CkMyPe(), (unsigned long long)channel_id, (unsigned long long)generation, (unsigned long long)epoch, sender_pe,
              (unsigned long long)sender_token, (unsigned long long)rec->getID(), rec->outstandingChannelWork);
  auto* m = deviceChannelMsg<DeviceChannelClosing>(device_channel_closing_handler);
  m->token = sender_token;
  m->channel = channel_id;
  m->generation = generation;
  m->epoch = epoch;
  m->owner_pe = CkMyPe();
  chanSendControl(sender_pe, sizeof(DeviceChannelClosing), m);
}

void CkDeviceChannel::onCloseAck(int pe, uint64_t token, uint64_t last_seq, uint64_t last_bytes) {
  deviceChannelProgress();
  note(14, last_seq, epoch, pe, -1, (int)last_bytes);
  // What the sender put last against what this side has taken: a put that
  // landed while the element was closing has its event still queued, and the
  // element's pup must carry those bytes (unseenBytes()). Not free until the
  // application takes and releases it.
  if (last_seq == UINT64_MAX) landed_unseen = SIZE_MAX;
  else if (last_seq > taken_seq) { landed_unseen = (size_t)last_bytes; landed_seq = last_seq; busy = true; }
  if (deviceChannelDebug())
    CmiPrintf("[CHAN-CLOSE %d] channel %llx: ack from pe %d token %llx (owner=%d closing_gen %llu gen %llu sent=%d closed=%d)\n",
              CkMyPe(), (unsigned long long)channel_id, pe, (unsigned long long)token, (int)is_owner,
              (unsigned long long)closing_gen, (unsigned long long)generation, (int)closing_sent, (int)sender_closed);
  if (!is_owner || !closing_sent) return;
  closing_sent = false;
  retireCreditSlot();
  CkLocMgr* mgr = (CkLocMgr*)CkLocalBranch(owner_mgr);
  CkLocRec* rec = mgr ? mgr->localRecord(owner_id) : NULL;
  const bool reclose = resolved_during_closing;
  resolved_during_closing = false;
  // The sender resolved again while that closing was out (its request crossed
  // the ack): that resolution is closed too, posted before this ack's count-down
  // so the element's count never touches zero in between.
  if (reclose && rec) closingForMigration(rec);
  else sender_closed = true;
  if (rec) rec->noteChannelWorkDone();
  else if (deviceChannelDebug())
    CmiPrintf("[CHAN-CLOSE %d] channel %llx: ack but element %llu not found here\n", CkMyPe(),
              (unsigned long long)channel_id, (unsigned long long)owner_id);
}

static const char* const chan_hist_names[] = {
  "put", "pend", "staged", "resolve-req", "resolved", "credit-msg", "credit-shm", "closing-rx", "closeack-tx",
  "srcdone", "taken", "release", "reply", "closing-tx", "closeack-rx", "-", "-", "pack",
  "unpack", "done-rx", "early-credit", "stale-credit", "pulled", "announce-rx"};
void CkDeviceChannel::note(uint8_t kind, uint64_t seq, uint64_t epoch_, int pe, int slot, int n) {
  Hist& h = hist[hist_n++ % HIST_N];
  h.kind = kind; h.pe = pe; h.slot = slot; h.n = n; h.seq = seq; h.epoch = epoch_; h.t = CkWallTimer();
}
void CkDeviceChannel::printHist(const char* role) const {
  char buf[2600]; int off = 0;
  buf[0] = '\0';
  const uint32_t n = hist_n < (uint32_t)HIST_N ? hist_n : (uint32_t)HIST_N;
  const unsigned nnames = sizeof(chan_hist_names) / sizeof(*chan_hist_names);
  for (uint32_t i = 0; i < n; i++) {
    const Hist& h = hist[(hist_n - n + i) % HIST_N];
    off += snprintf(buf + off, sizeof(buf) - off, " [%s s%llu e%llu pe%d sl%d n%d @%.3f]",
                    h.kind < nnames ? chan_hist_names[h.kind] : "?", (unsigned long long)h.seq,
                    (unsigned long long)h.epoch, h.pe, h.slot, h.n, h.t);
    if (off >= (int)sizeof(buf) - 80) break;
  }
  CmiPrintf("[CHAN-HIST %d] %s channel %llx token %llx: credit slot (area %d slot %d) seen %u, credits %d; last %u events:%s\n",
            CkMyPe(), role, (unsigned long long)channel_id, (unsigned long long)my_token,
            is_owner ? credit_slot_pe : credit_shm_pe, is_owner ? credit_slot : credit_shm_slot,
            credits_seen, credits, n, buf);
}

void CkDeviceChannel::describeStall(const char* role) const {
  if (const DeviceChannelProbeCtx* pc = device_channel_probe_ctx) {
    CmiPrintf("[CHAN-PROBE %d] %s channel %llx gen %llu, asked by pe %d token %llx (its epoch %llu, seq %llu): "
              "is_owner=%d busy=%d have_last=%d last(seq %llu tag %llu kind %d from pe %d token %llx) epoch=%llu "
              "sender_known=%d sender=%d/%llx sender_closed=%d closing_sent=%d closing_gen=%llu waiting=%zu pulls=%zu taken_seq=%llu unseen=%zu "
              "| as sender: resolved=%d credits=%d in_flight=%d pending=%d token=%llx\n",
              CkMyPe(), role, (unsigned long long)channel_id, (unsigned long long)generation, pc->pe,
              (unsigned long long)pc->token, (unsigned long long)pc->epoch, (unsigned long long)pc->seq,
              (int)is_owner, (int)busy, (int)have_last, (unsigned long long)last.seq, (unsigned long long)last.tag,
              (int)last.kind, last.sender_pe, (unsigned long long)last.sender_token, (unsigned long long)epoch,
              (int)sender_known, sender_pe, (unsigned long long)sender_token, (int)sender_closed, (int)closing_sent,
              (unsigned long long)closing_gen, waiting.size(), pulls.size(), (unsigned long long)taken_seq, landed_unseen,
              (int)resolved, credits, in_flight, (int)pending.valid, (unsigned long long)my_token);
    printHist(role);
    return;
  }
  const bool sender_stuck = pending.valid || resolving || in_flight > 0 || close_ack_due || staged_outstanding > 0;
  const bool owner_stuck = is_owner && (!waiting.empty() || closing_sent || !pulls.empty());
  if (!sender_stuck && !owner_stuck) return;
  if (device_channel_watch_tally && !is_owner && pending.valid && resolved && credits == 0 && in_flight == 0) {
    // Waiting for a credit that is not coming: ask the owner what it thinks.
    auto* m = deviceChannelMsg<DeviceChannelProbe>(device_channel_probe_handler);
    m->channel = channel_id; m->token = my_token; m->epoch = epoch; m->seq = send_seq; m->pe = CkMyPe();
    m->owner_known = owner_known ? 1 : 0; m->owner_id = owner_id; m->owner_mgr = owner_mgr;
    CmiSyncSendAndFree(dst.pe, sizeof(DeviceChannelProbe), (char*)m);
  }
  if (DeviceChannelWatchTally* t = device_channel_watch_tally) {
    t->s_inflight += in_flight > 0; t->s_pending += pending.valid; t->s_resolving += resolving;
    t->s_closeack += close_ack_due;
    if (is_owner) { t->o_waiting += !waiting.empty(); t->o_closing += closing_sent; t->o_pulls += !pulls.empty(); }
    if (t->shown >= 12) return;
    t->shown++;
  }
  // A put in flight: is its stream still working, or did its completion
  // callback get lost?
  const char* stream_state = "-";
  if (in_flight > 0) {
    const hapiError_t q = hapiStreamQuery(last_stream);
    stream_state = (q == hapiSuccess) ? "idle" : (q == hapiErrorNotReady) ? "busy" : hapiGetErrorName(q);
  }
  static const char* const tier_names[3] = {"copy", "ipc", "read"};
  CmiPrintf("[CHAN-STALL %d] %s channel %llx gen %llu owner_pe %d | sender: resolved=%d resolving=%d credits=%d "
            "in_flight=%d(stream %s, tier %s) pending=%d(tag %llu) close_ack_due=%d token=%llx seq=%llu epoch=%llu closed=%llu | owner: is_owner=%d busy=%d pulls=%zu pulling=%d "
            "sender_known=%d sender=%d/%llx closing_gen=%llu sent=%d closed=%d waiting=%zu have_last=%d\n",
            CkMyPe(), role, (unsigned long long)channel_id, (unsigned long long)generation, dst.pe,
            (int)resolved, (int)resolving, credits, in_flight, stream_state,
            !resolved ? "?" : tier_names[tier < 0 || tier > 2 ? 0 : tier],
            (int)pending.valid, (unsigned long long)(pending.valid ? pending.tag : 0), (int)close_ack_due,
            (unsigned long long)my_token, (unsigned long long)send_seq, (unsigned long long)epoch,
            (unsigned long long)closed_epoch, (int)is_owner, (int)busy, pulls.size(), (int)pulling,
            (int)sender_known, sender_pe, (unsigned long long)sender_token, (unsigned long long)closing_gen,
            (int)closing_sent, (int)sender_closed, waiting.size(), (int)have_last);
  if (staged_outstanding > 0 || !pulls.empty())
    CmiPrintf("[CHAN-STALL %d]   staged puts outstanding %d; queued announces %zu, head seq %llu, taken_seq %llu\n",
              CkMyPe(), staged_outstanding, pulls.size(),
              pulls.empty() ? 0ULL : (unsigned long long)pulls.front().ev.seq, (unsigned long long)taken_seq);
  printHist(role);
}

// Past the stand-down, before the pack: the buffer is about to go.
void CkDeviceChannel::packForMigration() {
  if (!is_owner) return;
  note(17, taken_seq, epoch, sender_pe, credit_slot, (int)busy);
  retireCreditSlot();
  dst.close();
  unregisterOwner();
  is_owner = false;
  sender_known = false;
  closing_sent = false;
  resolved_during_closing = false;
  sender_closed = false;
  closing_rec = NULL;
}

CkDeviceStatus CkDeviceChannel::put(const void* src, size_t bytes, hapiStream_t stream,
                                    const CkCallback& src_done, uint64_t tag) {
  deviceChannelProgress();
  deviceChannelLateInit();
  if (channel_id == 0) {
    CkAbort("CkDeviceChannel::put: the channel was never open()ed by its receiver\n");
  }
  if (bytes > cap) {
    CkAbort("CkDeviceChannel::put: %zu bytes into a channel of %zu\n", bytes, cap);
  }
  if (sender_rec == NULL) {
    sender_rec = chanActiveRec();
    if (sender_rec) {
      holder_elem = sender_rec->getID();
      holder_mgr = sender_rec->getLocMgr()->ckGetGroupID();
      holder_known = true;
    }
  }
  // The put holds the sending element in place from now until its source is
  // done, queued or not: the element does not migrate with a put of its
  // buffers in flight or waiting.
  if (sender_rec) sender_rec->noteChannelWorkPosted();
  // The balancers' communication edge, recorded here while the sending element
  // is the running object: a put issued later from a credit or resolve handler
  // would be charged to whatever object runs then.
  deviceChannelRecordSend(sender_rec, bytes, dst.pe, owner_known, owner_mgr, owner_id);
  if (deviceChannelDebug())
    CmiPrintf("[CHAN-PUT %d] channel %llx token %llx: put tag %llu: resolved %d resolving %d closed_epoch %llu credits %d staged %d pending %d\n",
              CkMyPe(), (unsigned long long)channel_id, (unsigned long long)my_token, (unsigned long long)tag, (int)resolved,
              (int)resolving, (unsigned long long)closed_epoch, credits, staged_outstanding, (int)pending.valid);
  // Closed by the receiver's migration, or still behind puts staged while it
  // was: the put goes as a staged announce, holding nothing here.
  if ((!resolved && closed_epoch > 0) || (resolved && staged_outstanding > 0)) {
    issueStaged(src, bytes, stream, src_done, tag);
    return CkDeviceStatus::incomplete;
  }
  if (resolved && !pull_mode && credits == 0 && credit_shm_slot >= 0) pollShmCredit();   // a credit may be waiting in shared memory
  if (resolved && (pull_mode || credits > 0)) {
    if (!pull_mode) { credits--; uncredited++; }
    issue(src, bytes, stream, src_done, tag);
  } else if (!pending.valid) {
    pending.src = src;
    pending.bytes = bytes;
    pending.stream = stream;
    pending.done = src_done;
    pending.tag = tag;
    pending.valid = true;
    note(1, send_seq, epoch, dst.pe, credit_shm_slot, credits);
    if (resolved && credit_shm_slot >= 0 && !shm_waiting) {
      shm_waiting = true;
      CkpvAccess(device_channel_shm_waiters)->push_back(this);
    }
    if (!resolved && !resolving) requestResolve();
  } else {
    CkAbort("CkDeviceChannel::put: a put is already queued behind the credit for put %llu on "
            "channel %llx; the receiver's release() frees the buffer, a second channel overlaps\n",
            (unsigned long long)send_seq, (unsigned long long)channel_id);
  }
  return CkDeviceStatus::incomplete;
}

// From the owning element's pup on unpack: this channel arrived with it and is
// reopened later (on the final buffer). In the table from now so that a
// resolve or an announce reaching the element first is queued here, not
// refused. A copy unpacked as an entry-method parameter must not call it (the
// parameter is a temporary; a table entry for it would dangle).
void CkDeviceChannel::arriving() {
  if (channel_id != 0 && !is_owner) registerOwner();
}

void CkDeviceChannel::resolve() {
  if (channel_id == 0 || resolved || resolving) return;
  requestResolve();
}

void CkDeviceChannel::requestResolve() {
  if (my_token == 0) {
    my_token = deviceChannelToken();
    (*CkpvAccess(device_channel_senders))[my_token] = this;
  }
  resolving = true;
  note(3, send_seq, epoch, dst.pe, -1, credits);
  CkDeviceChannelEvent ev;
  memset(&ev, 0, sizeof(ev));
  ev.kind = CkDeviceChannelEvent::RESOLVE;
  ev.channel = channel_id;
  ev.sender_pe = CkMyPe();
  ev.sender_token = my_token;
  control.send(CkDataMsg::buildNew((int)sizeof(ev), &ev));
}

// The view of an endpoint this copy no longer uses: released now if nothing of
// this copy is in flight, else when the last put completes (onSrcDone).
void CkDeviceChannel::releaseDstView() {
  if (!dst.ipc_open) return;
  if (in_flight == 0) { dst.releaseView(); return; }
  stale_views.push_back(dst);
  stale_views.back().takeView(dst);
}

void CkDeviceChannel::onResolved(const CkDeviceChannelEndpoint& ep, uint64_t gen_, int owner_pe,
                                 bool free, uint64_t epoch_, int credit_pe, int credit_slot_,
                                 uint64_t taken_seq_) {
  deviceChannelProgress();
  // Messages between two PEs are not ordered: the closing of this resolution
  // may have come (and been acked) before the resolution itself. An endpoint
  // already closed is never used; the request made at the closing (or the next
  // put) brings the current one.
  if (epoch_ <= closed_epoch) {
    if (deviceChannelDebug())
      CmiPrintf("[CHAN-STALE %d] channel %llx: resolution %llu arrived after its closing was acked (closed %llu); dropped\n",
                CkMyPe(), (unsigned long long)channel_id, (unsigned long long)epoch_, (unsigned long long)closed_epoch);
    return;
  }
  // Two resolutions can be out to one copy at once, and an older one arriving
  // after a newer one must not replace it: the owner's credits are under the
  // newest, and the copy would wait for them under the old one forever. The
  // same one twice would double its credit.
  if (epoch != 0 && epoch_ <= epoch) {
    device_channel_resolved_older++;
    snprintf(device_channel_last_resolved_older, sizeof(device_channel_last_resolved_older),
             "channel %llx token %llx: resolution %llu after %llu (resolved %d, closed %llu)",
             (unsigned long long)channel_id, (unsigned long long)my_token, (unsigned long long)epoch_,
             (unsigned long long)epoch, (int)resolved, (unsigned long long)closed_epoch);
    if (deviceChannelDebug()) CmiPrintf("[CHAN-STALE %d] %s; ignored\n", CkMyPe(), device_channel_last_resolved_older);
    return;
  }
  // The buffer's one credit, reconciled by sequence number rather than reset.
  // The reply's free flag describes the buffer as of the receiver's last take
  // (taken_seq_), and the receiver's DONEs, credits and the reply itself reach
  // this copy in any order. So: a direct put the receiver had not taken at the
  // reply will occupy the buffer (-1; its credit comes under this resolution);
  // a staged put it read after the reply does the same when its DONE is seen,
  // here if the DONE overtook the reply (dones_seen) or in onStagedDone
  // otherwise; a release under this resolution that overtook it (an early
  // credit) is +1 when it frees a put taken after the reply, or the put held
  // at it. Everything up to taken_seq_ is the free flag's.
  acct_seq = taken_seq_;
  int c = free ? 1 : 0;
  occupant_credit_due = !free;
  {
    std::vector<uint64_t> keep;
    for (uint64_t sq : direct_issued) if (sq > acct_seq) { keep.push_back(sq); c--; }
    direct_issued.swap(keep);
  }
  for (uint64_t sq : dones_seen) if (sq > acct_seq) c--;
  dones_seen.clear();
  int early_credit = 0;
  if (my_token != 0) {
    DeviceChannelEarlyCredits& early = *CkpvAccess(device_channel_early_credits);
    auto it = early.find(my_token);
    if (it != early.end()) {
      std::vector<DeviceChannelEarlyCredit> keep;
      for (const DeviceChannelEarlyCredit& ec : it->second) {
        if (ec.epoch == epoch_ && (ec.seq > acct_seq || (ec.seq == acct_seq && !free))) {
          c++; early_credit++;
          if (ec.seq == acct_seq) occupant_credit_due = false;
        } else if (ec.epoch > epoch_) keep.push_back(ec);   // of a resolution still to come
      }
      if (keep.empty()) early.erase(it); else it->second.swap(keep);
    }
  }
  note(4, taken_seq_, epoch_, owner_pe, credit_slot_, c);
  if (deviceChannelDebug())
    CmiPrintf("[CHAN-RESOLVED %d] channel %llx token %llx: resolution %llu applied (held %llu resolved %d closed %llu, free %d taken_seq %llu, early %d, direct puts in flight %zu, credits %d)\n",
              CkMyPe(), (unsigned long long)channel_id, (unsigned long long)my_token, (unsigned long long)epoch_,
              (unsigned long long)epoch, (int)resolved, (unsigned long long)closed_epoch, (int)free, (unsigned long long)acct_seq,
              early_credit, direct_issued.size(), c);
  epoch = epoch_;
  if (dst.id != ep.id || dst.pe != ep.pe) {
    releaseDstView();
    dst = ep;
  } else {
    // The same buffer: this copy keeps its view of it.
    CkDeviceChannelEndpoint held;
    held.takeView(dst);
    dst = ep;
    dst.takeView(held);
  }
  generation = gen_;
  resolved = true;
  resolving = false;
  credits = c;
  credit_shm_pe = credit_pe;
  credit_shm_slot = credit_slot_;
  credits_seen = 0;
  // How this copy reaches the buffer. A copy (same process) or a copy through
  // the IPC mapping (same host) is issued here and waits for credits; anything
  // else is announced and read by the receiver, which needs no credit: its
  // read completes locally, nothing follows the data on its path, and on
  // Slingshot a read into device memory runs at the link's rate where a write
  // into it does not.
  tier = dst.mode();
  pull_mode = (tier == 2);
  if (pending.valid && staged_outstanding == 0 && (pull_mode || credits > 0)) {
    Pending p = pending;
    pending.valid = false;
    if (!pull_mode) { credits--; uncredited++; }
    issue(p.src, p.bytes, p.stream, p.done, p.tag);
  } else if (pending.valid) {
    // Queued before this resolution (put() lists a copy for the per-PE
    // shared-memory credit poll only when it queues resolved): the credit comes
    // through the word, and the owner's release can precede this reply's
    // application. List it, and read the word now.
    awaitShmCredit();
  }
}

// A put is queued behind a credit that comes through shared memory: list this
// copy for the per-PE poll and read the word once now.
void CkDeviceChannel::awaitShmCredit() {
  if (!pending.valid || !resolved || pull_mode || credit_shm_slot < 0) return;
  if (!shm_waiting) {
    shm_waiting = true;
    CkpvAccess(device_channel_shm_waiters)->push_back(this);
  }
  pollShmCredit();
}

void CkDeviceChannel::onClosing(uint64_t gen_, int owner_pe, uint64_t epoch_) {
  deviceChannelProgress();
  (void)gen_;
  if (deviceChannelDebug())
    CmiPrintf("[CHAN-CLOSE %d] channel %llx: closing of resolution %llu received (current %llu resolved %d in_flight %d pending %d)\n",
              CkMyPe(), (unsigned long long)channel_id, (unsigned long long)epoch_, (unsigned long long)epoch,
              (int)resolved, in_flight, (int)pending.valid);
  if (epoch_ > closed_epoch) closed_epoch = epoch_;
  close_ack_pe = owner_pe;
  note(7, send_seq, epoch_, owner_pe, -1, in_flight);
  if (resolved && epoch_ < epoch) {
    // A closing of a resolution this copy already left behind: nothing of it is
    // in flight, the owner only needs the ack (it closes the current one next).
    sendCloseAck();
    return;
  }
  resolved = false;
  resolving = false;
  uncredited = 0;   // the next resolution's free flag accounts for the buffer
  credit_shm_slot = -1; credit_shm_pe = -1;
  if (shm_waiting) { deviceChannelShmUnlist(this); shm_waiting = false; }
  // The buffer is about to be freed: this copy's view of it goes with the ack,
  // once nothing of this copy is in flight.
  if (in_flight == 0) { releaseDstView(); sendCloseAck(); }
  else close_ack_due = true;
  // A put queued here for a credit goes to the receiver's new home as a staged
  // announce now, instead of waiting for the next resolution.
  if (pending.valid) {
    Pending p = pending;
    pending.valid = false;
    issueStaged(p.src, p.bytes, p.stream, p.done, p.tag);
  }
}

void CkDeviceChannel::sendCloseAck() {
  close_ack_due = false;
  note(8, last_direct_seq, epoch, close_ack_pe, -1, in_flight);
  auto* m = deviceChannelMsg<DeviceChannelCloseAck>(device_channel_closeack_handler);
  m->channel = channel_id;
  m->token = my_token;
  m->pe = CkMyPe();
  // The last put WRITTEN into the buffer: the owner carries its bytes in the
  // pup if their event is still queued. An announced put was read, not
  // written; reporting it would mark the buffer busy for nothing.
  m->last_seq = last_direct_seq;
  m->last_bytes = last_put_bytes;
  chanSendControl(close_ack_pe, sizeof(DeviceChannelCloseAck), m);
}

void CkDeviceChannel::onSrcDone(uint64_t seq) {
  deviceChannelProgress();
  if (in_flight > 0) in_flight--;
  note(9, seq, epoch, -1, -1, in_flight);
  if (sender_rec) sender_rec->noteChannelWorkDone();
  if (in_flight == 0 && !stale_views.empty()) {
    for (const CkDeviceChannelEndpoint& v : stale_views) v.releaseView();
    stale_views.clear();
  }
  if (close_ack_due && in_flight == 0) { releaseDstView(); sendCloseAck(); }
}

// The receiver has read announced put `seq`: the source may be reused.
void CkDeviceChannel::onDone(uint64_t seq) {
  CkCallback cb(CkCallback::ignore);
  auto it = done_cbs.find(seq);
  if (it != done_cbs.end()) { cb = it->second; done_cbs.erase(it); }
  // Replaced sources whose every announced put has now been read.
  for (size_t i = 0; i < retired_srcs.size();) {
    if (done_cbs.empty() || done_cbs.begin()->first > retired_srcs[i].last_seq) {
      retired_srcs[i].ep.close();
      retired_srcs[i] = retired_srcs.back();
      retired_srcs.pop_back();
    } else i++;
  }
  onSrcDone(seq);
  if (cb.type != CkCallback::ignore) cb.send(NULL);
}

// The registered source of pull mode is replaced: closed now if every put
// announced from it has been read, else once the last one has (onDone).
void CkDeviceChannel::retireSource(uint64_t through_seq) {
  if (src_ep.id == 0) return;
  const bool outstanding = !done_cbs.empty() && done_cbs.begin()->first <= through_seq;
  if (outstanding) retired_srcs.push_back(Retired{src_ep, through_seq});
  else src_ep.close();
  src_ep = CkDeviceChannelEndpoint();
}

// This copy's registered sources, at its end: closed unless a read of one may
// still be in flight (deregistering memory under a transfer is worse than
// keeping a registration).
void CkDeviceChannel::closeSources() {
  if (!done_cbs.empty()) return;
  src_ep.close();
  src_ep = CkDeviceChannelEndpoint();
  for (Retired& r : retired_srcs) r.ep.close();
  retired_srcs.clear();
}

// An announced put: read it now if the buffer is free, else after release().
void CkDeviceChannel::onAnnounce(const CkDeviceChannelEvent& ev, const CkDeviceChannelEndpoint& src) {
  deviceChannelProgress();
  Queued q; q.ev = ev; q.src = src;
  pulls.push_back(q);
  note(23, ev.seq, epoch, ev.sender_pe, -1, (int)pulls.size());
  if (is_owner && !busy && !pulling) startPull();
}

void CkDeviceChannel::startPull() {
  if (!is_owner || busy || pulling || pulls.empty()) return;
  // Puts are read in the order they were made: a direct put whose note is
  // still on its way is taken and released before an announced put that
  // followed it is read over the buffer.
  if (pulls.front().ev.seq != taken_seq + 1) return;
  Queued q = pulls.front();
  pulls.erase(pulls.begin());
  pulling = true;
  pull_ev = q.ev;
  pull_src = q.src;
  pull_src.cnt = q.ev.bytes;
  if (pull_src.cnt > cap) {
    CkAbort("CkDeviceChannel: announced put of %zu bytes into a channel of %zu\n", pull_src.cnt, cap);
  }
  // The read writes this buffer outside every stream the element uses: the
  // element must not leave with it in flight.
  if (owner_known) {
    CkLocMgr* mgr = (CkLocMgr*)CkLocalBranch(owner_mgr);
    CkLocRec* rec = mgr ? mgr->localRecord(owner_id) : NULL;
    if (rec) rec->noteChannelWorkPosted();
  }
  if (deviceChannelDebug())
    CmiPrintf("[CHAN-READ %d] channel %llx: put %llu (%zu bytes) from pe %d, %s\n", CkMyPe(),
              (unsigned long long)channel_id, (unsigned long long)pull_ev.seq, pull_src.cnt, pull_ev.sender_pe,
              pull_src.cnt == 0 ? "nothing to read" : pull_src.mode() == 2 ? "RDMA" : "copy");
  if (pull_src.cnt == 0) { onPulled(); return; }   // a zero-byte put: the note alone
  chanRead(dst, pull_src, pull_src.cnt, CkCallback(deviceChannelPulledFn, (void*)new DeviceChannelPulled{channel_id}));
}

// The read completed: the application's DATA event, and DONE to the sender.
void CkDeviceChannel::onPulled() {
  deviceChannelProgress();
  pulling = false;
  pull_src.releaseView();
  // The buffer holds unconsumed data from now until release(): the application
  // may take the DATA event much later (an SDAG waits for its tag), and the
  // next announce must not be read over it.
  busy = true;
  // Until event() takes it, these bytes are in the buffer and nowhere else: a
  // pack in between must carry them, and unseenBytes() is how the owner learns
  // that.
  landed_unseen = pull_ev.bytes;
  landed_seq = pull_ev.seq;
  note(22, pull_ev.seq, epoch, pull_ev.sender_pe, -1, (int)pull_ev.bytes);
  if (owner_known) {
    CkLocMgr* mgr = (CkLocMgr*)CkLocalBranch(owner_mgr);
    CkLocRec* rec = mgr ? mgr->localRecord(owner_id) : NULL;
    if (rec) rec->noteChannelWorkDone();
  }
  auto* d = deviceChannelMsg<DeviceChannelDone>(device_channel_done_handler);
  d->token = pull_ev.sender_token;
  d->channel = channel_id;
  d->seq = pull_ev.seq;
  chanSendControl(pull_ev.sender_pe, sizeof(DeviceChannelDone), d);
  CkDeviceChannelEvent ev = pull_ev;
  ev.kind = CkDeviceChannelEvent::DATA;
  CkDataMsg* note = CkDataMsg::buildNew((int)sizeof(ev), &ev);
  CkSetRefNum(note, (CMK_REFNUM_TYPE)ev.tag);
  arrived.send(note);
}

void CkDeviceChannel::issue(const void* src, size_t bytes, hapiStream_t stream,
                            const CkCallback& src_done, uint64_t tag) {
  const uint64_t seq = ++send_seq;
  note(0, seq, epoch, dst.pe, credit_shm_slot, credits);
  last_stream = stream;
  // What the closing ack reports as written into the owner's buffer and what
  // the credits account for: a direct put (and a zero-byte one, whose note
  // alone goes). An announced put is read by the owner when it is ready, so
  // it is in neither.
  if (!(pull_mode && bytes > 0)) {
    direct_issued.push_back(seq);
    last_put_bytes = bytes;
    last_direct_seq = seq;
  }
  if (shm_waiting && !pending.valid) { deviceChannelShmUnlist(this); shm_waiting = false; }
  if (my_token == 0) {
    my_token = deviceChannelToken();
    (*CkpvAccess(device_channel_senders))[my_token] = this;
  }
  CkDeviceChannelEvent ev;
  memset(&ev, 0, sizeof(ev));
  ev.kind = CkDeviceChannelEvent::DATA;
  ev.channel = channel_id;
  ev.generation = generation;
  ev.seq = seq;
  ev.bytes = bytes;
  ev.sender_pe = CkMyPe();
  ev.sender_token = my_token;
  ev.tag = tag;
  // Nothing to move: the notification goes now and the source is done. The
  // receiver still releases, as for any put. Not when the receiver reads: a
  // put there is issued without a credit, so this note would overtake the
  // previous put's release at the receiver; the announce queue keeps the
  // order, and the receiver takes a zero-byte announce as the note alone.
  if (bytes == 0 && !pull_mode) {
    in_flight++;
    CkDataMsg* note = CkDataMsg::buildNew((int)sizeof(ev), &ev);
    CkSetRefNum(note, (CMK_REFNUM_TYPE)tag);
    arrived.send(note);
    deviceChannelSrcDoneFn(new DeviceChannelSrcDone{my_token, seq, src_done}, NULL);
    return;
  }
  if (pull_mode) {
    // The receiver reads: the source is opened once per buffer (registered
    // for a reader on another host), the user's source-done waits for the
    // receiver's DONE, and the announce goes when the producing stream is
    // complete. Several puts may be outstanding; the receiver queues them and
    // reads in order as its buffer frees.
    if (bytes > 0) {
      if (src_ep.id == 0 || src_ep.ptr != src || bytes > src_ep.cnt) {
        retireSource(src_ep_last);
        src_ep = CkDeviceChannelEndpoint(src, bytes);
        src_ep.open();
        src_ep.rdma_valid = src_ep.registration(&src_ep.rdma);   // without one the reader must map it
      }
      src_ep_last = seq;
    }
    done_cbs[seq] = src_done;
    in_flight++;
    ev.kind = CkDeviceChannelEvent::ANNOUNCE;
    CkDeviceChannelEndpoint a = src_ep;
    a.cnt = bytes;
    if (a.rdma_valid) a.rdma.cnt = bytes;
    PUP::sizer sz;
    sz | a;
    std::vector<char> buf(sizeof(ev) + sz.size());
    memcpy(buf.data(), &ev, sizeof(ev));
    PUP::toMem tm(buf.data() + sizeof(ev));
    tm | a;
    CkDataMsg* ann = CkDataMsg::buildNew((int)buf.size(), buf.data());
    hapiAddCallback(stream, CkCallback(deviceChannelAnnounceReadyFn, (void*)new DeviceChannelAnnounce{control, ann}));
    return;
  }
  // A copy within the process or through the IPC mapping: into this PE's open
  // batch on the producing stream, whose completion delivers the receiver's
  // note (through the owner's shared-memory ring for a same-host pair) and the
  // source-done.
  CkCallback note_cb = arrived;
  void* note_msg = nullptr;
  if (chanShmPair(dst.pe)) {
    note_cb = CkCallback(deviceChannelShmNoteFn, (void*)new DeviceChannelShmNote{chanShmIndexOf(dst.pe), ev, arrived});
  } else {
    CkDataMsg* note = CkDataMsg::buildNew((int)sizeof(ev), &ev);
    CkSetRefNum(note, (CMK_REFNUM_TYPE)tag);
    note_msg = note;
  }
  void* dptr = (tier == 0) ? (void*)dst.ptr : dst.view();
  if (dptr == nullptr)
    CkAbort("CkDeviceChannel::put: channel %llx has no mapping of PE %d's buffer\n",
            (unsigned long long)channel_id, dst.pe);
  in_flight++;
  deviceChannelBatchAdd(stream, dptr, src, bytes, DeviceChannelBatchEntry{my_token, seq, src_done, note_cb, note_msg});
}

// A put while the receiver's channel is closed (its element is migrating), or
// while earlier such puts are still unread: the bytes are copied to a staging
// buffer on the producing stream, the application's source is released as soon
// as that copy has read it (src_done), and the staging buffer is announced to
// the receiving element -- an element message, which the location manager
// forwards, so it is waiting at the new home when the element reopens and
// reads it in put order (startPull). Nothing holds the sender; the sending
// element is held by one unit of channel work until the receiver's DONE frees
// the staging buffer (device_channel_done_bridge), so it does not move with
// one unread. Balance: the receiver's read takes the buffer as a direct put
// would and its release credits, so this side debits one credit at the DONE
// (onStagedDone) and the two pair off.
struct DeviceChannelStagedCopied { CkCallback user; };
static void deviceChannelStagedCopiedFn(void* param, void* msg) {
  auto* d = (DeviceChannelStagedCopied*)param;
  if (d->user.type != CkCallback::ignore) d->user.send(msg);
  delete d;
}

void CkDeviceChannel::issueStaged(const void* src, size_t bytes, hapiStream_t stream,
                                  const CkCallback& src_done, uint64_t tag) {
  note(2, send_seq, epoch, dst.pe, -1, staged_outstanding);
  if (my_token == 0) {
    my_token = deviceChannelToken();
    (*CkpvAccess(device_channel_senders))[my_token] = this;
  }
  void* stage = nullptr;
  if (bytes > 0) {
    stage = CkDeviceMalloc(bytes);
    if (stage == nullptr) {
      // No device memory for the copy: wait for the new home. The queued put
      // is issued directly once resolved and credited, and only behind the
      // staged ones (onStagedDone).
      if (pending.valid) {
        CkAbort("CkDeviceChannel::put: a put is already queued on channel %llx while its receiver migrates, "
                "and there is no device memory (%zu bytes) to stage another\n", (unsigned long long)channel_id, bytes);
      }
      pending.src = src; pending.bytes = bytes; pending.stream = stream; pending.done = src_done; pending.tag = tag;
      pending.valid = true;
      if (!resolved && !resolving) requestResolve();
      return;
    }
  }
  const uint64_t seq = ++send_seq;   // in the one sequence with direct puts: the receiver reads in order
  CkDeviceChannelEndpoint sep;
  if (bytes > 0) {
    hapiCheck(hapiMemcpyAsync(stage, src, bytes, hapiMemcpyDeviceToDevice, stream));
    sep = CkDeviceChannelEndpoint(stage, bytes);
    sep.open();
    // The new home is not known yet: a reader on another host needs the
    // registration (on one host the reader maps the staging buffer).
    if (CmiNumPhysicalNodes() > 1) sep.rdma_valid = sep.registration(&sep.rdma);
  }
  // The application's source is free once the copy has read it.
  hapiAddCallback(stream, CkCallback(deviceChannelStagedCopiedFn, (void*)new DeviceChannelStagedCopied{src_done}));
  CkDeviceChannelEvent ev;
  memset(&ev, 0, sizeof(ev));
  ev.kind = CkDeviceChannelEvent::ANNOUNCE;
  ev.channel = channel_id;
  ev.generation = generation;
  ev.seq = seq;
  ev.bytes = bytes;
  ev.sender_pe = CkMyPe();
  ev.sender_token = my_token;
  ev.tag = tag;
  PUP::sizer sz;
  sz | sep;
  std::vector<char> buf(sizeof(ev) + sz.size());
  memcpy(buf.data(), &ev, sizeof(ev));
  PUP::toMem tm(buf.data() + sizeof(ev));
  tm | sep;
  CkDataMsg* ann = CkDataMsg::buildNew((int)buf.size(), buf.data());
  (*CkpvAccess(device_channel_staged))[std::make_pair(my_token, seq)] =
      DeviceChannelStaged{stage, sep, holder_mgr, holder_elem, holder_known};
  staged_outstanding++;
  if (deviceChannelDebug())
    CmiPrintf("[CHAN-STAGED %d] channel %llx token %llx: put %llu (%zu bytes, tag %llu) staged at %p; %d outstanding\n",
              CkMyPe(), (unsigned long long)channel_id, (unsigned long long)my_token, (unsigned long long)seq, bytes,
              (unsigned long long)tag, stage, staged_outstanding);
  // Announced once the copy is complete on the stream.
  hapiAddCallback(stream, CkCallback(deviceChannelAnnounceReadyFn, (void*)new DeviceChannelAnnounce{control, ann}));
  // The new home answers after reopen.
  if (!resolved && !resolving) requestResolve();
}

// The receiver read a staged put (its DONE; the staging buffer is already
// freed by the handler): the read took the buffer as a direct put would, and
// the receiver's release of it will credit, so one credit is debited here.
void CkDeviceChannel::onStagedDone(uint64_t seq) {
  deviceChannelProgress();
  note(19, seq, epoch, -1, -1, staged_outstanding);
  if (deviceChannelDebug())
    CmiPrintf("[CHAN-DONE-RX %d] channel %llx token %llx: staged put %llu read by the owner: credits %d staged %d pending %d resolved %d epoch %llu\n",
              CkMyPe(), (unsigned long long)channel_id, (unsigned long long)my_token, (unsigned long long)seq, credits,
              staged_outstanding, (int)pending.valid, (int)resolved, (unsigned long long)epoch);
  if (staged_outstanding > 0) staged_outstanding--;
  // Unresolved: the next reply's free flag speaks for every put up to its
  // taken_seq, so the DONE is kept and reconciled there. Resolved: a put read
  // after the reply is debited here; one read before it was counted by its
  // free flag.
  if (!resolved) dones_seen.push_back(seq);
  else if (seq > acct_seq) credits--;
  if (staged_outstanding == 0 && pending.valid && resolved && credits > 0) {
    Pending p = pending;
    pending.valid = false;
    credits--;
    uncredited++;
    issue(p.src, p.bytes, p.stream, p.done, p.tag);
  } else if (staged_outstanding == 0 && pending.valid) {
    awaitShmCredit();   // the queued put now waits for a credit that may come through shared memory
  }
}

// A credit under the current resolution frees one put: the one held at the
// reply first (not in direct_issued, which the reply's reconciliation pruned),
// then the oldest direct put. seq names it when the credit came as a message;
// 0 (shared memory) takes the put order.
void CkDeviceChannel::creditedPut(uint64_t seq) {
  if (occupant_credit_due && (seq == 0 || seq <= acct_seq)) { occupant_credit_due = false; return; }
  occupant_credit_due = false;
  if (direct_issued.empty()) return;
  if (seq != 0) {
    for (size_t i = 0; i < direct_issued.size(); i++)
      if (direct_issued[i] == seq) { direct_issued.erase(direct_issued.begin() + i); return; }
    if (seq < direct_issued.front()) return;   // a staged put's release: no direct put freed
  }
  direct_issued.erase(direct_issued.begin());
}

void CkDeviceChannel::onCredit(uint64_t seq, uint64_t epoch_) {
  deviceChannelProgress();
  note(5, seq, epoch_, -1, -1, credits);
  if (deviceChannelDebug())
    CmiPrintf("[CHAN-CREDIT-RX %d] channel %llx token %llx: credit for put %llu epoch %llu arrives: resolved %d resolving %d epoch %llu credits %d staged %d pending %d uncredited %d\n",
              CkMyPe(), (unsigned long long)channel_id, (unsigned long long)my_token, (unsigned long long)seq, (unsigned long long)epoch_,
              (int)resolved, (int)resolving, (unsigned long long)epoch, credits, staged_outstanding, (int)pending.valid, uncredited);
  if (pull_mode && resolved && epoch_ == epoch) return;   // the receiver reads when it is ready
  // A credit of another resolution (the owner released after it closed this
  // copy, or before it answered the current request) is already in the free
  // flag the next Resolved carries, or will be; counting it would allow two
  // puts into one buffer.
  if (epoch_ > epoch && my_token != 0) {
    // Ahead of its resolution (see device_channel_early_credits).
    (*CkpvAccess(device_channel_early_credits))[my_token].push_back(DeviceChannelEarlyCredit{epoch_, seq});
    device_channel_credits_early++;
    note(20, seq, epoch_, -1, -1, credits);
    snprintf(device_channel_last_early, sizeof(device_channel_last_early),
             "channel %llx put %llu of resolution %llu while current %llu resolved %d resolving %d pending %d",
             (unsigned long long)channel_id, (unsigned long long)seq, (unsigned long long)epoch_,
             (unsigned long long)epoch, (int)resolved, (int)resolving, (int)pending.valid);
    return;
  }
  if (!resolved || epoch_ != epoch) {
    device_channel_credits_refused++;
    note(21, seq, epoch_, -1, -1, credits);
    snprintf(device_channel_last_refused, sizeof(device_channel_last_refused),
             "channel %llx put %llu of resolution %llu while current %llu resolved %d resolving %d pending %d credits %d",
             (unsigned long long)channel_id, (unsigned long long)seq, (unsigned long long)epoch_,
             (unsigned long long)epoch, (int)resolved, (int)resolving, (int)pending.valid, credits);
    if (deviceChannelDebug())
      CmiPrintf("[CHAN-STALE %d] channel %llx: credit for put %llu of resolution %llu dropped (current %llu, resolved %d)\n",
                CkMyPe(), (unsigned long long)channel_id, (unsigned long long)seq, (unsigned long long)epoch_,
                (unsigned long long)epoch, (int)resolved);
    if (pending.valid && !resolved && !resolving) requestResolve();
    return;
  }
  credits++;
  if (uncredited > 0) uncredited--;
  creditedPut(seq);
  if (staged_outstanding > 0) return;   // the queued put follows the staged ones (onStagedDone)
  if (!pending.valid) return;
  Pending p = pending;
  pending.valid = false;
  credits--;
  uncredited++;
  issue(p.src, p.bytes, p.stream, p.done, p.tag);
}

void CkDeviceChannel::pup(PUP::er& p) {
  p|dst;
  p|arrived;
  p|control;
  p|channel_id;
  p|generation;
  p|cap;
  p|owner_id;
  p|owner_known;
  p|owner_mgr;
  p|busy;
  p|epoch;
  p|sender_known;
  p|sender_pe;
  p|sender_token;
  // A receiver may move between event() and release(): the taken event goes
  // with it, so the release at the new home finds it (the credit then goes
  // nowhere, and the sender learns the buffer is free when it re-resolves).
  p|have_last;
  p((char*)&last, sizeof(last));
  p|send_seq;
  p|taken_seq;
  p|landed_unseen;
  p|landed_seq;
  p|last_put_bytes;
  p|last_direct_seq;
  // Announced puts not yet read travel with the element; the sources stay
  // valid until DONE, which the new home sends after reading them.
  int np = (int)pulls.size();
  p|np;
  if (p.isUnpacking()) pulls.resize(np);
  for (int i = 0; i < np; i++) { p((char*)&pulls[i].ev, sizeof(pulls[i].ev)); p|pulls[i].src; }
  int nw = (int)waiting.size();
  p|nw;
  if (p.isUnpacking()) waiting.resize(nw);
  for (int i = 0; i < nw; i++) { p|waiting[i].pe; p|waiting[i].token; }
  if (p.isUnpacking()) {
    // A copy, whichever side it lands on, starts unresolved and unregistered:
    // the receiver re-opens it (reopen) and senders re-resolve.
    is_owner = false;
    sender_closed = sender_known;
    closing_gen = 0;
    closing_sent = false;
    resolved_during_closing = false;
    closing_rec = NULL;
    resolved = false;
    resolving = false;
    credits = 0;
    uncredited = 0;
    pull_mode = false;
    tier = 0;
    done_cbs.clear();
    pulling = false;
    memset(&pull_ev, 0, sizeof(pull_ev));
    pull_src = CkDeviceChannelEndpoint();
    closed_epoch = 0;   // a new token: nothing stale can reach this copy
    in_flight = 0;
    staged_outstanding = 0;   // a holder does not move with a staged put unread (it holds the element)
    acct_seq = 0; occupant_credit_due = false; direct_issued.clear(); dones_seen.clear();
    close_ack_due = false;
    close_ack_pe = -1;
    my_token = 0;
    credit_shm_slot = -1; credit_shm_pe = -1; credits_seen = 0; shm_waiting = false;
    credit_slot = -1; credit_slot_pe = -1;
    hist_n = 0; note(18, send_seq, epoch, CkMyPe());
    sender_rec = NULL;
    holder_elem = 0;
    holder_known = false;
    pending.valid = false;
    src_ep = CkDeviceChannelEndpoint();
    src_ep_last = 0;
    retired_srcs.clear();
    stale_views.clear();
  }
}

/****************************** Control events and migration hooks ******************************/

// ArrayElement::ckDeviceChannelControl: a sender's resolve or announce for one
// of this element's channels, delivered where the element lives.
void CkDeviceChannelControl(void* msg) {
  CkDataMsg* m = (CkDataMsg*)msg;
  if (m->getLength() < (int)sizeof(CkDeviceChannelEvent)) {
    CkAbort("CkDeviceChannelControl: not a channel event (%d bytes)\n", m->getLength());
  }
  CkDeviceChannelEvent ev;
  memcpy(&ev, m->getData(), sizeof(ev));
  CkDeviceChannel* c = deviceChannelOwner(ev.channel);
  if (c == NULL) {
    CkAbort("CkDeviceChannel: a control event for channel %llx reached PE %d, where its element "
            "holds no channel of that id; an element's pup must carry its channels, call arriving() on "
            "unpack and reopen() them on the new buffers\n", (unsigned long long)ev.channel, CkMyPe());
  }
  c->event(m);
  delete m;
}

void CkDeviceChannelsBeforeMigrate(CkLocRec* rec) {
  if (!device_channel_batch_ready) return;
  DeviceChannelTable& tab = *CkpvAccess(device_channel_owners);
  if (tab.empty()) return;
  const CmiUInt8 id = rec->getID();
  const int mgr = rec->getLocMgr()->ckGetGroupID().idx;
  std::vector<CkDeviceChannel*> owned;
  for (auto& kv : tab)
    if (kv.second->ownedBy(mgr, id)) owned.push_back(kv.second);
  for (CkDeviceChannel* c : owned) c->closingForMigration(rec);
}

void CkDeviceChannelsPack(CkLocRec* rec) {
  if (!device_channel_batch_ready) return;
  DeviceChannelTable& tab = *CkpvAccess(device_channel_owners);
  if (tab.empty()) return;
  const CmiUInt8 id = rec->getID();
  const int mgr = rec->getLocMgr()->ckGetGroupID().idx;
  std::vector<CkDeviceChannel*> owned;
  for (auto& kv : tab)
    if (kv.second->ownedBy(mgr, id)) owned.push_back(kv.second);
  for (CkDeviceChannel* c : owned) c->packForMigration();
}

#endif // (CMK_CUDA || CMK_HIP)
