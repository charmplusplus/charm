#include <map>
#include <vector>
#ifndef _CKRDMADEVICE_H_
#define _CKRDMADEVICE_H_

#include "ckcallback.h"
#include "conv-rdmadevice.h"

#if CMK_GPU
#include "hapi_portable.h"
#include <functional>

#define CkNcpyModeDevice CmiNcpyModeDevice
#define CkDeviceStatus CmiDeviceStatus

// A persistent device endpoint (the "Direct API"): a device buffer exported once
// by its owner with open(), then read or written repeatedly by a peer holding a
// copy of the endpoint, with get()/put() carrying no metadata beyond the two
// endpoints. Copies reach the peers by pup. Three transports, chosen by where
// the two PEs are: a stream memcpy inside one process; a device-to-device copy
// through the process-wide IPC import cache on the same host, so pool and vmm
// memory export like any other allocation; and a one-sided RDMA put or get
// between hosts, with the registration taken once at open(). Completion is
// delivered through the two endpoints' callbacks, fired by the initiator once
// the bytes are in place (a CkCallback to a chare is a message, so the passive
// side's reaches it wherever it lives).
//
// The caller orders the data: a put ships what hapi_stream has produced when
// the copy, or the RDMA write, runs; a get reads what the peer's buffer holds at
// that moment, so the peer must have finished writing it. The owner must not
// free or overwrite a buffer a peer may still be reading; the callbacks are
// what say when a transfer is over.
struct CkDevicePersistent {
  const void* ptr;
  size_t cnt;
  CkCallback cb;
  void* cb_msg;
  hapiStream_t hapi_stream = hapiStreamPerThread;

  int pe;
  // Same-host, other-process transport: the export of the allocation holding
  // ptr as hapiIpcExportBuffer hands it out. The handle names the allocation
  // (or the owner's vmm heap); ptr sits ipc_offset past its base, which is
  // ipc_base in the owner's address space. A peer resolves it through the
  // import cache, never with its own cudaIpcOpenMemHandle: one open maps a
  // whole region, and a second mapping of a region the cache already holds is
  // refused (cudaErrorAlreadyMapped).
  hapiIpcMemHandle_t hapi_ipc_handle;
  size_t ipc_offset;
  void* ipc_base;
  bool ipc_exported;
  void* ipc_ptr;   // the peer's view of ptr once imported
  bool ipc_open;   // used only by the remote chare
  // Cross-host transport: the RDMA registration of [ptr, ptr+cnt), taken at
  // open() when the job spans more than one host (the arena's one registration
  // under +gpupool). Travels in the pup so a peer can name this buffer in a
  // one-sided put or get.
  CmiNcpyBuffer rdma;
  bool rdma_registered;
  bool rdma_owned;   // this endpoint's own registration, retired at close()

  CkDevicePersistent() : ptr(nullptr), cnt(0), cb(CkCallback(CkCallback::ignore)),
                         cb_msg(nullptr), pe(-1), ipc_offset(0), ipc_base(nullptr),
                         ipc_exported(false), ipc_ptr(nullptr), ipc_open(false),
                         rdma_registered(false), rdma_owned(false) {}

  explicit CkDevicePersistent(const void* ptr_, size_t cnt_)
    : ptr(ptr_), cnt(cnt_), cb(CkCallback(CkCallback::ignore)) {
    init();
  }

  explicit CkDevicePersistent(const void* ptr_, size_t cnt_, const CkCallback& cb_)
    : ptr(ptr_), cnt(cnt_), cb(cb_) {
    init();
  }

  explicit CkDevicePersistent(const void* ptr_, size_t cnt_, hapiStream_t hapi_stream_)
    : ptr(ptr_), cnt(cnt_), cb(CkCallback(CkCallback::ignore)),
      hapi_stream(hapi_stream_) {
    init();
  }

  explicit CkDevicePersistent(const void* ptr_, size_t cnt_, const CkCallback& cb_,
      hapiStream_t hapi_stream_)
    : ptr(ptr_), cnt(cnt_), cb(cb_), hapi_stream(hapi_stream_) {
    init();
  }

  void init();
  // Export (and, across hosts, register) the buffer; the owner calls it once
  // before sending copies of the endpoint to its peers.
  void open();
  // Owner: retire the export and an owned registration, before the buffer is
  // freed. Peer copy: forget the imported view (the mapping stays with the
  // import cache, which other transfers into the same region keep using).
  void close();
  void set_msg(void* msg);

  // Should only be used for exchanging between chares, not for migration.
  // After the owner chare migrates, CkDevicePersistent needs to be recreated
  // and exchanged again.
  void pup(PUP::er& p);

  CkDeviceStatus get(CkDevicePersistent& src);
  CkDeviceStatus put(CkDevicePersistent& dst);
};

// What the receiver's `arrived` entry gets as the payload of a CkDataMsg. A DATA
// event is a put: which channel, which put, how many bytes. A RESOLVE event is a
// sender asking for the channel's current endpoint; CkDeviceChannel::event()
// answers it and hands the application nothing.
struct CkDeviceChannelEvent {
  // ANNOUNCE: a put on the cross-host tier; the receiver reads the bytes itself
  // (an RDMA read beat the sender's write at every size on Slingshot, 2026-10-18)
  // and the application sees a DATA event once they are in place.
  enum Kind : int { DATA = 1, RESOLVE = 2, ANNOUNCE = 3 };
  int kind;
  uint64_t channel;
  uint64_t generation;
  uint64_t seq;
  size_t bytes;
  int sender_pe;
  uint64_t sender_token;
  uint64_t tag;   // the sender's put() tag; also the message's reference number
};

class CkDataMsg;
class CkLocRec;

// A persistent channel: one fixed destination buffer on the receiver, exported
// and registered once by open(). The receiver keeps its object; copies reach
// the senders by pup, as an entry-method argument. A sender writes with put(),
// which carries nothing but the bytes and a sequence number, in whichever of
// the three transports the two PEs call for (CkDevicePersistent above). The
// receiver's `arrived` callback fires once the bytes are in place -- a
// CkCallback to a chare, so it follows the receiver wherever it lives -- with a
// CkDataMsg holding a CkDeviceChannelEvent.
//
// The sender must not overwrite the buffer while the receiver may still read
// it. The receiver calls release() after consuming a put (once the kernels that
// read it are ordered behind it), which credits the sender; a put issued before
// the credit for the previous one arrived is queued, one deep, and issued when
// it does, never blocking the PE. Overlapping two puts means two channels on
// two buffers. One sender per channel.
//
// Migration, without application re-exchange. A sender resolves the channel
// before its first put: a RESOLVE event to the receiver's entry (so it reaches
// the receiver wherever it lives), answered with the current endpoint, its
// generation and whether the buffer is free. When the receiving element is
// chosen to migrate, the runtime tells every resolved sender the channel is
// closing and the element stands down until each has acknowledged -- a sender
// acknowledges once it has no put in flight -- so no put ever lands in a buffer
// that is being freed (CkRdmaDeviceChannelsBeforeMigrate, from emigrate). The
// receiver re-opens at its new home with reopen(new buffer) in its pup, which
// bumps the generation; the senders' next put re-resolves. A put's bytes that
// landed before the pack are in the buffer the application pups, as its other
// device state; its notification is forwarded to the new home. A sender element
// itself cannot migrate while one of its puts is in flight (the put counts as
// an outstanding device send), and a migrated sender re-resolves before its
// next put.
//
// A sender never waits for the move (2026-10-20). A put made while its channel
// is closed is copied to a staging buffer on the producing stream, the source
// is released to the application once that copy has read it, and the staging
// buffer is announced to the receiving element: an element message, which the
// location manager parks and forwards, so it waits at the new home and is read
// into the reopened buffer in put order (startPull). Every put behind it goes
// the same way until the receiver's DONE has freed the last staging buffer,
// so a direct put never races a read. The closing, its ack, the resolution,
// the credit and the DONE travel as urgent messages: each hop costs the entry
// method running at the destination, not the depth of its queue (which was
// 50-400 ms at 448 objects per PE, and the whole loss of channels to zero-copy
// under the asynchronous balancer: holds of 300-1100 ms per moved element).
class CkDeviceChannel {
public:
  CkDeviceChannel();
  // Receiver side: the destination buffer, its capacity, and the entry that
  // learns of each put. The entry takes a CkDataMsg*, hands it to event() and
  // deletes it. Open the channel inside an entry method or constructor of the
  // receiving element.
  CkDeviceChannel(void* dst, size_t capacity, const CkCallback& arrived);
  CkDeviceChannel(const CkDeviceChannel& o);
  CkDeviceChannel& operator=(const CkDeviceChannel& o);
  ~CkDeviceChannel();

  // Receiver side.
  void open();                 // export and register the buffer; assign the channel id
  void reopen(void* new_dst);  // after migration: the same channel on the new buffer
  void close();                // retire the export and registration, before the buffer goes
  // The put described by the message `arrived` delivered (the message stays the
  // caller's), or NULL for an event the runtime consumed (a resolve).
  const CkDeviceChannelEvent* event(CkDataMsg* m);
  // The buffer may be overwritten: credit the sender.
  void release();
  // A holder copy: ask for the receiver's current endpoint now (e.g. right after
  // the holding element migrated), instead of at the next put.
  void resolve();
  // The owning element's pup, on unpack: the channel arrived and reopens later.
  void arriving();

  // Sender side: write bytes of src, produced on stream, into the buffer.
  // src_done fires when src may be reused. tag travels in the event and is set
  // as the reference number of the message `arrived` receives, so an SDAG
  // `when arrived[step](CkDataMsg* m)` matches puts by step.
  CkDeviceStatus put(const void* src, size_t bytes, hapiStream_t stream,
                     const CkCallback& src_done = CkCallback(CkCallback::ignore),
                     uint64_t tag = 0);
  bool putQueued() const { return pending.valid; }

  uint64_t id() const { return channel_id; }
  uint64_t gen() const { return generation; }
  size_t capacity() const { return cap; }
  // Owner, at pack time: bytes of a put that landed after the last one taken
  // (its event is still queued), learned from the sender's closing ack; 0 when
  // nothing is outstanding; SIZE_MAX when unknown (the sender's copy had left
  // its PE): the pup must then carry the whole buffer.
  size_t unseenBytes() const { return landed_unseen; }
  int ownerPe() const { return dst.pe; }
  void pup(PUP::er& p);

  // Runtime side (the converse handlers below).
  void onCredit(uint64_t seq, uint64_t epoch_);
  void creditedPut(uint64_t seq);
  void onClosing(uint64_t gen_, int owner_pe, uint64_t epoch_);
  void onResolved(const CkDevicePersistent& ep, uint64_t gen_, int owner_pe, bool free, uint64_t epoch_,
                  int credit_pe = -1, int credit_slot = -1, int signal_slot_ = -1,
                  const CmiNcpyBuffer* signal_hdr_ = NULL, uint64_t taken_seq_ = 0);
  void retireSignalSlot();     // owner: give the slot of the current resolution back
  void stashSignalSlot();      // owner: keep it for the closing ack to release (pack, or a reply before the ack)
  void pollShmCredit();        // sender: take credits the owner granted through shared memory
  void retireCreditSlot();     // owner: give the slot of the current resolution back
  void onCloseAck(int pe, uint64_t token, uint64_t last_seq, uint64_t last_bytes);
  void onSrcDone(uint64_t seq);
  void onDone(uint64_t seq);   // the receiver finished reading put `seq`
  void onStagedDone(uint64_t seq);   // the receiver read a staged put (its DONE): the balance moves
  void onAnnounce(const CkDeviceChannelEvent& ev, const CkDevicePersistent& src);
  void startPull();
  void onPulled();
  void sendCloseAck();
  // Element ids are unique only within one location manager (one array, or
  // bound arrays): an element is (manager, id).
  bool ownedBy(int mgr_idx, CmiUInt8 element) const {
    return is_owner && owner_known && owner_id == element && owner_mgr.idx == mgr_idx;
  }
  CmiUInt8 ownerElement() const { return owner_id; }
  void closingForMigration(CkLocRec* rec);
  void packForMigration();
  // One line of state when something is stuck (CHARM_DEBUG_MIGRATE scan).
  void describeStall(const char* role) const;
  friend void CkRdmaDeviceChannelsStandDownWhy(CkLocRec* rec, int* own_inflight, int* own_pending, int* acks);
  // The last events on this object (both roles), kept always and printed by
  // describeStall: what a stall needs explained, without the full trace
  // (CHARM_DEBUG_SIGNAL costs gigabytes per run and slows the race).
  struct Hist { uint8_t kind; int32_t pe; int32_t slot; int32_t n; uint64_t seq; uint64_t epoch; double t; };
  enum { HIST_N = 24 };
  Hist hist[HIST_N];
  uint32_t hist_n = 0;
  void note(uint8_t kind, uint64_t seq, uint64_t epoch_, int pe = -1, int slot = -1, int n = 0);
  void printHist(const char* role) const;
  // The element moved to another PE of this process without a pup (the
  // intra-process handoff): the object is the same, the buffer and its exports
  // stay valid, only the per-PE registrations move.
  void leaveForHandoff();
  void arriveFromHandoff();

private:
  void issue(const void* src, size_t bytes, hapiStream_t stream, const CkCallback& src_done,
             uint64_t tag);
  // A put while the receiver is closed for migration, or behind such puts:
  // copied to a staging buffer on the producing stream and announced to the
  // receiving element wherever it lands (see issueStaged).
  void issueStaged(const void* src, size_t bytes, hapiStream_t stream, const CkCallback& src_done,
                   uint64_t tag);
  void requestResolve();
  void awaitShmCredit();
  void replyResolve(int pe, uint64_t token);
  void takeOver(const CkDeviceChannel& o);
  void registerOwner();
  void unregisterOwner();

  CkDevicePersistent dst;   // the receiver's buffer, exported/registered by open()
  CkCallback arrived;
  CkCallback control;       // where runtime events (a resolve) go: the element's own entry, or `arrived`
  uint64_t channel_id;
  uint64_t generation;      // bumped by every (re)open of the receiver's buffer
  size_t cap;
  // Receiver (owner) side.
  bool is_owner;
  CmiUInt8 owner_id;        // the receiving element, for the migration hook
  bool owner_known;         // opened inside an array element's entry (ids may be 0)
  CkGroupID owner_mgr;      // its location manager
  bool busy;                // a put taken and not yet released
  uint64_t taken_seq;       // the last put taken by event()
  // Same-host, other-process pairs signal through shared memory instead of
  // transport messages (see device_channel_shm in ckrdmadevice.C). Owner: the
  // credit slot it granted to the current sender; sender: where to read its
  // credits, and how many it has consumed from that word.
  int credit_slot_pe;       // owner: node-local PE whose area holds the slot (-1: message credits)
  int credit_slot;          // owner: the slot
  int credit_shm_pe;        // sender: the owner's area (-1: message credits)
  int credit_shm_slot;      // sender: the slot
  uint32_t credits_seen;    // sender: credits already taken from the word
  bool shm_waiting;         // sender: listed for the per-PE credit poll (a put is queued)
  // Cross-host pairs: the note rides on the data write as immediate data
  // (ckrdmadevice.C, chan_signal_*). Owner: the slot it granted; sender: the
  // slot and the owner process's header region for the per-put header write.
  int owner_signal_slot;
  int signal_slot;
  int departed_signal_slot; // owner: the slot it held when it packed (or re-resolved before the ack); freed where it
  int departed_signal_pe;   //   was granted once the sender's acked last put has been taken there
  CmiNcpyBuffer signal_hdr;
  size_t landed_unseen;     // see unseenBytes()
  uint64_t landed_seq;      // owner: the put those bytes are (from the close ack), 0 once taken
  size_t last_put_bytes;    // sender: bytes of the last put written into the buffer (carried by the closing ack)
  uint64_t last_direct_seq; // sender: that put's seq. Staged puts (announced, never written) are not the ack's business.
  int sender_pe;            // the resolved sender: where credits and closings go
  uint64_t sender_token;
  bool sender_known;
  uint64_t closing_gen;     // generation under which a migration is pending (closings sent), 0 if none
  bool closing_sent;        // a closing to the current sender is awaiting its ack
  bool sender_closed;       // the current sender acked a closing and has not resolved since
  bool sender_pulls;        // the resolved sender announces and this side reads (no credits needed)
  // Announces waiting for the buffer, the read in progress, and its source.
  struct Queued { CkDeviceChannelEvent ev; CkDevicePersistent src; };
  std::vector<Queued> pulls;
  bool pulling;
  CkDeviceChannelEvent pull_ev;
  CkDevicePersistent pull_src;
  bool resolved_during_closing;   // a resolve was answered while that closing was in flight: close the new resolution on the ack
  uint64_t epoch;           // owner: resolves answered so far; each Resolved, Closing and Credit names the resolution it belongs to.
                            // sender: the resolution the current endpoint came from
  CkLocRec* closing_rec;    // the element standing down for the pending migration
  struct Waiting { int pe; uint64_t token; };
  std::vector<Waiting> waiting;   // resolves received while migrating, answered after reopen
  CkDeviceChannelEvent last;
  bool have_last;
  // Sender (holder) side.
  bool resolved, resolving;
  uint64_t send_seq;
  int credits;
  int uncredited;           // sender, put mode: puts issued under this resolution not yet credited back
  bool pull_mode;           // sender: the resolved home is on another host, puts are announced and read
  std::map<uint64_t, CkCallback> done_cbs;   // sender: source-done callbacks awaiting the receiver's DONE
  uint64_t closed_epoch;    // sender: the newest resolution a closing was acked for; a Resolved or Credit of it (or older) is stale
  int in_flight;
  int staged_outstanding;   // sender: puts staged and announced, not yet read by the receiver (its DONE)
  uint64_t acct_seq;        // sender: the last put the receiver had taken at the current resolution's reply; its free flag covers puts up to it
  bool occupant_credit_due; // sender: the put held at that reply is still to be credited (the first credit of this resolution is its)
  std::vector<uint64_t> direct_issued;   // sender: direct puts issued and not yet credited back, in order
  std::vector<uint64_t> dones_seen;      // sender: staged puts the receiver read while this copy was unresolved, reconciled at the next resolution
  bool close_ack_due;
  int close_ack_pe;
  uint64_t my_token;        // this copy's entry in its PE's token table, 0 if none
  double closing_at;        // sender: when the closing of its resolution arrived (0: none held); owner: when it sent one
  CkLocRec* sender_rec;     // the sending element, charged with the puts in flight
  CmiUInt8 holder_elem;     // that element's id, for the handoff hooks
  CkGroupID holder_mgr;     // and its location manager
  bool holder_known;
  struct Pending {
    const void* src;
    size_t bytes;
    hapiStream_t stream;
    CkCallback done;
    uint64_t tag;
    bool valid;
  } pending;
  CkDevicePersistent src_ep;   // the source endpoint of the last put, reused while (src, bytes) hold
};

// Converse handlers of the channel protocol (credit, closing, close-ack,
// resolve reply); registered in init.C.
extern "C" void* device_channel_credit_bridge(void* arg);
extern "C" void* device_channel_closing_bridge(void* arg);
extern "C" void* device_channel_closeack_bridge(void* arg);
extern "C" void* device_channel_resolved_bridge(void* arg);
extern int device_channel_credit_handler;
extern int device_channel_closing_handler;
extern int device_channel_closeack_handler;
extern int device_channel_resolved_handler;
extern int device_channel_done_handler;
extern "C" void* device_channel_done_bridge(void* arg);
void CkRdmaDeviceChannelInit();
// ArrayElement::ckDeviceChannelControl: a control event for one of this element's channels.
void CkDeviceChannelControl(void* msg);
// Migration hooks, from CkLocMgr::emigrate: tell the senders of the channels
// this element owns to stop and stand the element down until they have (raises
// outstandingDeviceSends per awaited ack); then, once past the stand-down,
// retire the channels' exports before the pack.
void CkRdmaDeviceChannelsBeforeMigrate(CkLocRec* rec);
void CkRdmaDeviceChannelsStandDownWhy(CkLocRec* rec, int* own_inflight, int* own_pending, int* acks);
// From CkCallstackPop: launch the entry method's batched channel puts while it is still the attributed object.
void CkRdmaDeviceChannelBatchFlush();
void CkRdmaDeviceChannelsPack(CkLocRec* rec);
// The intra-process handoff (CkLocMgr::emigrateIntraProcess / immigrateIntraProcess):
// the element's channel objects move with it unpacked, so their per-PE table
// entries leave the old PE and are made on the new one.
void CkRdmaDeviceChannelsLeaving(CkLocRec* rec);
void CkRdmaDeviceChannelsArrived(CkLocRec* rec);

struct CkDeviceBufferPost {
  // CUDA stream for device transfers
  hapiStream_t hapi_stream;

  // The posted buffer is free: nothing still running on hapi_stream reads or
  // writes it, so the runtime may land the transfer as soon as the sender's
  // work is done. Left false, the transfer is also held until every
  // operation issued on hapi_stream before this post has completed -- the
  // ordering a copy on that stream used to give. Neither case puts a wait on
  // hapi_stream or on any other stream: the copy runs on the PE's receive
  // stream and is issued only once it can run (see DeviceRecvPending).
  bool buffer_free = false;

  // Use per-thread stream by default
  CkDeviceBufferPost() : hapi_stream(hapiStreamPerThread) {}
};

class CkDeviceBuffer : public CmiDeviceBuffer {
public:
  // Callback to be invoked on the sender/receiver
  CkCallback cb;

  CkDeviceBuffer() : CmiDeviceBuffer() {
    cb = CkCallback(CkCallback::ignore);
  }

  explicit CkDeviceBuffer(const void* ptr_) : CmiDeviceBuffer(ptr_, 0) {
    cb = CkCallback(CkCallback::ignore);
  }

  explicit CkDeviceBuffer(const void* ptr_, const CkCallback& cb_) : CmiDeviceBuffer(ptr_, 0) {
    cb = cb_;
  }

  // With a stream: the buffer is produced by work on that stream and the send
  // orders behind its tail. Without one: the buffer is complete now (see
  // CmiDeviceBuffer::source_ready).
  explicit CkDeviceBuffer(const void* ptr_, hapiStream_t hapi_stream_) : CmiDeviceBuffer(ptr_, 0) {
    cb = CkCallback(CkCallback::ignore);
    hapi_stream = hapi_stream_;
    source_ready = false;
  }

  explicit CkDeviceBuffer(const void* ptr_, const CkCallback& cb_, hapiStream_t hapi_stream_) : CmiDeviceBuffer(ptr_, 0) {
    cb = cb_;
    hapi_stream = hapi_stream_;
    source_ready = false;
  }

  explicit CkDeviceBuffer(const void* ptr_, size_t cnt_) : CmiDeviceBuffer(ptr_, cnt_) {
    cb = CkCallback(CkCallback::ignore);
  }

  explicit CkDeviceBuffer(const void* ptr_, size_t cnt_, const CkCallback& cb_) : CmiDeviceBuffer(ptr_, cnt_) {
    cb = cb_;
  }

  explicit CkDeviceBuffer(const void* ptr_, size_t cnt_, hapiStream_t hapi_stream_) : CmiDeviceBuffer(ptr_, cnt_) {
    cb = CkCallback(CkCallback::ignore);
    hapi_stream = hapi_stream_;
    source_ready = false;
  }

  explicit CkDeviceBuffer(const void* ptr_, size_t cnt_, const CkCallback& cb_, hapiStream_t hapi_stream_) : CmiDeviceBuffer(ptr_, cnt_) {
    cb = cb_;
    hapi_stream = hapi_stream_;
    source_ready = false;
  }

  void pup(PUP::er &p) {
    CmiDeviceBuffer::pup(p);
    p|cb;
  }

  friend void CkRdmaDeviceIssueRgets(envelope *env, int numops, void **arrPtrs, int *arrSizes, CkDeviceBufferPost *postStructs);
};

void CkRdmaDeviceRecvHandler(void* data);
void CkRdmaDeviceRecvHandler(void* data, void* msg);
void CkRdmaDeviceIssueRgets(envelope *env, int numops, void **arrPtrs, int *arrSizes, CkDeviceBufferPost *postStructs);
// A device message parked at the PE its element is migrating to: pull its
// payload into landing buffers now and complete the sender, so the sender is
// not pinned for the whole landing (see the definition). Returns false when
// the message was left as it was.
bool CkRdmaDeviceStageParked(envelope* env);
void CkRdmaDeviceOnSender(int dest_pe, int numops, CkDeviceBuffer** buffers);
// Generated asynchronous proxies use the four-argument prepare followed by
// SendWhenReady. The latter owns the marshalled message and continuation until
// all producer streams complete. The three-argument API remains synchronous.
struct CkDeviceDeferredSend;
void CkRdmaDeviceOnSender(int dest_pe, int numops, CkDeviceBuffer** buffers,
                          CkDeviceDeferredSend** pending);
void CkRdmaDeviceSendWhenReady(CkDeviceDeferredSend* pending, void* msg,
                              const std::function<void()>& send);
// A device-send message that is leaving this process for another one on the
// same physical node: re-prepare every memcpy-prepared payload in it as a
// direct IPC send, in place, so the new home reads it without a correction.
// Three outcomes, and they must not be collapsed into one bool: the redirect
// bridge treats a SECOND redirect as fatal (the source process failed to repair
// its own payload) but has to let a park through, since a park is delivered by
// the callback that resumes the repair. Merging the two aborted the async arm
// at the first parked redirect.
enum class CkDeviceRepairResult {
  CallerDelivers,   // repaired in place, or nothing needed repairing
  Redirected,       // shipped to the source process, which repairs and delivers
  Parked,           // waiting on a producer; deviceForwardRepairReady delivers
};
// A parked message is delivered by the callback that resumes the repair, which
// is why the caller's delivery options have to travel with it.
CkDeviceRepairResult CkRdmaDeviceRepairForward(envelope* env, int newPe,
                                               int opts = 0);
extern "C" void* device_forward_redirect_bridge(void* arg);
extern "C" int device_forward_redirect_handler;

class CkLocRec;
CkpvExtern(CkLocRec*, _currentLocRec);

// Device registration cache, off unless CHARM_DEVICE_MR_CACHE is set; see the
// design note in ckrdmadevice.C. DropRegistrations must be called wherever an
// element's device buffers are about to be freed, and only where no transfer
// against them is in flight.
void CkRdmaDeviceRegistrationCacheInit();
// Stalled-receive watchdog, off unless CHARM_ZC_STALL_SECS is set.
void CkRdmaDeviceStallWatchInit();

// A correction defers a receive across a round trip; these count it against the
// receiving element so migration stands down until it lands. See the note at
// the definitions in cklocation.C.
void CkNoteDeviceRecvDeferred(CkGroupID aid, CmiUInt8 id);
void CkNoteDeviceRecvComplete(CkGroupID aid, CmiUInt8 id);
// The element a device zerocopy receive is addressed to, or NULL. Lets the
// registration cache attribute a receive-side registration to the element
// whose migration must retire it (see acquireDeviceRegistration).
CkLocRec* CkFindDeviceRecvElement(CkGroupID aid, CmiUInt8 id);
// Deferred-migration kick, enqueued by CkLocRec::noteDeviceSendDone so a
// migration never starts inside the handler that released it.
extern int _deferredMigrateHandlerIdx;
extern "C" void _deferredMigrateHandler(void* arg);
void CkRdmaDeviceDropRegistrations(CkLocRec* owner);

extern "C" {
  void* loopback_bridge(void* arg);
  extern int loopback_handler;
  // Migration-mismatch payload correction; see the protocol notes in
  // ckrdmadevice.C.
  void* device_restage_req_bridge(void* arg);
  void* device_restage_meta_bridge(void* arg);
  void* device_restage_put_done_bridge(void* arg);
  extern int device_restage_req_handler;
  extern int device_restage_meta_handler;
  extern int device_restage_put_done_handler;
}

#endif // CMK_GPU

// Returns and clears the device payload size recorded by the most recent
// CkRdmaDeviceOnSender on this PE, so the LB communication graph can weight a
// device zerocopy edge by what actually crosses the wire rather than by the
// descriptor envelope. Returns 0 for a non-device send. Declared outside the
// CUDA guard so the send path can call it without an #ifdef.
#if CMK_GPU
size_t CkRdmaDeviceTakePendingSendBytes();
// Claimed IPC event slots for this PE, or -1 when shm IPC is not in use.
int CkRdmaDeviceBusyIpcSlots();
// Allocate `size` bytes from the device load-balance region of `dm`'s buffer,
// reclaiming retired IPC slots and retrying once if the first attempt does not
// fit. Returns nullptr if it still does not. Handles dm->lock itself, so the
// caller must not hold it.
//
// The migration path allocates from this region but never sends through
// acquireIpcSendSlot, so it has no other way to reach the scan that hands these
// blocks back. `dm` is a DeviceManager*, passed opaquely so this declaration
// does not drag the HAPI headers into every includer.
void* CkRdmaDeviceAllocLbBuffer(void* dm, size_t size);

// Device pool: CkDeviceMalloc / CkDeviceFree hand out device buffers from
// arenas -- one large device allocation each, managed by the buddy allocator
// that already backs the communication buffer, grown on demand (CK_GPU_ARENA_MB,
// default 256), one arena set per device. Both calls are host-side bookkeeping:
// no cudaMalloc and no cudaFree per buffer, so neither synchronizes the device.
//
// Ported from device-rdma-dereg-3960 (8580c18ba, a58bf3ce4) without that
// branch's lazy whole-arena registration, which hooks into an interval map of
// registered regions this branch does not have.
//
// The pool answers the allocation-cost question only. CkDeviceFree returns the
// block at once with no stream ordering, so a buffer with device work still in
// flight against it must not be freed -- the caller owns that, exactly as with
// cudaFreeAsync. A buffer that was rebound into a migration arena by
// pup_buffer_device is NOT from this pool and must go back through
// hapiFreeMigratable; callers that mix the two have to know which is which.
void* CkDeviceMalloc(size_t size);
// The same, but never grows the pool: nullptr when no arena has a block that
// fits. The admission gate reserves with it (cklocation.C).
void* CkDeviceMallocNoGrow(size_t size);
void CkDeviceFree(void* ptr);
// +gpupool: the pool is the runtime's allocation policy (migration arenas and
// payloads from it, direct CUDA IPC only, no comm or load-balancing buffer).
// An application that follows the runtime's choice allocates through
// CkDeviceMalloc when this is true and through hapiMalloc otherwise.
bool CkDevicePoolOn();
// Set by the location manager around the send of a migration payload: the
// runtime's own buffer, released by the receiver's ack rather than by a
// completion callback, so the "direct send with no callback" warning does not
// apply to it.
void CkRdmaDeviceMarkMigrationPayload(bool sending);

// Load balance pool block recycling, without a host barrier.
//
// A block handed out by CkRdmaDeviceAllocLbBuffer had a previous life, and the
// device work of that life -- the landing write into it, and the unpack copies
// out of it -- can still be in flight when it is handed out again. Until now
// that was covered by synchronizing the host, which waits for far more than
// this block and stops the scheduler to do it.
//
// Instead: when a block goes back to the pool, record an event on the stream
// whose work touched it (Freed). When one comes out, make the stream that is
// about to write into it wait on those events (Gate). The ordering is then
// exact, expressed on the device, and costs no host time. Events are reused
// per (device, stream), so this allocates nothing per migration.
void CkRdmaDeviceNoteLbBufferFreed(void* dm, hapiStream_t usedBy);
void CkRdmaDeviceGateLbBuffer(void* dm, hapiStream_t consumer);
#else
inline size_t CkRdmaDeviceTakePendingSendBytes() { return 0; }
inline int CkRdmaDeviceBusyIpcSlots() { return -1; }
inline void* CkRdmaDeviceAllocLbBuffer(void* dm, size_t size) { return nullptr; }
inline void* CkDeviceMalloc(size_t size) { return nullptr; }
inline void* CkDeviceMallocNoGrow(size_t size) { return nullptr; }
inline void CkDeviceFree(void* ptr) {}
inline bool CkRdmaDeviceStageParked(envelope* env) { return false; }
inline bool CkDevicePoolOn() { return false; }
inline void CkRdmaDeviceMarkMigrationPayload(bool sending) {}
inline void CkRdmaDeviceNoteLbBufferFreed(void* dm, hapiStream_t usedBy) {}
inline void CkRdmaDeviceGateLbBuffer(void* dm, hapiStream_t consumer) {}
#endif

#endif // _CKRDMADEVICE_H_
