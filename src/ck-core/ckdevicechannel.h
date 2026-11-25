#ifndef _CKDEVICECHANNEL_H_
#define _CKDEVICECHANNEL_H_

/* Device channels (manual: "GPU Channels"). Included by ckrdmadevice.h inside
 * its (CMK_CUDA || CMK_HIP) block. */

#include <map>
#include <vector>

class CkDataMsg;
class CkLocRec;

// One device buffer as the channel runtime names it: where it lives (PE,
// process, host), the CUDA/HIP IPC handle of its allocation, and, once its
// owner has opened it, its entry in the owning process's registration table.
// Not an application interface: CkChannel builds and moves these.
//
// A holder in another process on the same host reaches the buffer through an
// IPC mapping shared by the whole process and reference counted; the holder's
// view (ipc_ptr) is neither copied nor pupped, and a holder releases it only
// when none of its copies through the mapping are in flight, so the mapping
// can be closed with the last reference without a device synchronize.
struct CkDeviceChannelEndpoint {
  const void* ptr = nullptr;
  size_t cnt = 0;
  int pe = -1;
  int node = -1;             // the process of pe
  int phys_node = -1;        // the host of pe
  hapiIpcMemHandle_t ipc_handle;
  size_t ipc_offset = 0;     // ptr minus the base of its allocation (what the handle maps)
  bool ipc_valid = false;    // ipc_handle names ptr's allocation
  uint64_t id = 0;           // nonzero once opened: the entry in its process's registration table
  // The buffer's network registration, filled in by its owner process before
  // the endpoint is sent to a reader on another host (a source that is read).
  CmiNcpyBuffer rdma;
  bool rdma_valid = false;
  // This holder's view of the IPC mapping (see above).
  mutable void* ipc_ptr = nullptr;
  mutable bool ipc_open = false;

  CkDeviceChannelEndpoint() {}
  // A buffer of this PE.
  CkDeviceChannelEndpoint(const void* ptr_, size_t cnt_);
  CkDeviceChannelEndpoint(const CkDeviceChannelEndpoint& o) { copyFrom(o); }
  CkDeviceChannelEndpoint& operator=(const CkDeviceChannelEndpoint& o) {
    if (this != &o) copyFrom(o);   // the caller releases this holder's view first, if it holds one
    return *this;
  }

  // Owner: take the IPC handle of the allocation and add the buffer to this
  // process's registration table (registered with the network on first use).
  void open();
  // Owner: deregister if registered and remove the entry.
  void close();
  // How this PE reaches the buffer: 0 a copy within the process, 1 a copy
  // through the IPC mapping (taking this holder's reference on it), 2 RDMA.
  int mode() const;
  // The buffer as mapped in this process (mode 1), or null.
  void* view() const;
  // Drop this holder's reference on the IPC mapping; synchronize: synchronize
  // the device before the last reference closes it (a holder that cannot tell
  // whether its copies through the mapping have finished).
  void releaseView(bool synchronize = false) const;
  // Move another holder's reference to this one.
  void takeView(const CkDeviceChannelEndpoint& o) const;
  // Owner process: the registration, registering now if needed, with cnt set
  // to this endpoint's cnt; false (and why) when the memory cannot be registered.
  bool registration(CmiNcpyBuffer* out, const char** why = nullptr) const;
  void pup(PUP::er& p);

private:
  void copyFrom(const CkDeviceChannelEndpoint& o) {
    ptr = o.ptr; cnt = o.cnt; pe = o.pe; node = o.node; phys_node = o.phys_node;
    ipc_handle = o.ipc_handle; ipc_offset = o.ipc_offset; ipc_valid = o.ipc_valid; id = o.id;
    rdma = o.rdma; rdma_valid = o.rdma_valid;
    ipc_ptr = nullptr; ipc_open = false;
  }
};

// What the receiver's `arrived` entry gets as the payload of a CkDataMsg. A DATA
// event is a put: which channel, which put, how many bytes. A RESOLVE event is a
// sender asking for the channel's current endpoint; CkChannel::event()
// answers it and hands the application nothing. An ANNOUNCE event is a put the
// receiver reads itself (a sender on another host, or one whose receiver was
// migrating); event() queues it and the application sees a DATA event once the
// bytes are in the buffer.
struct CkChannelEvent {
  enum Kind : int { DATA = 1, RESOLVE = 2, ANNOUNCE = 3 };
  int kind;
  uint64_t channel;
  uint64_t generation;
  uint64_t seq;
  size_t bytes;
  int sender_pe;
  int sender_mgr;          // RESOLVE: the sending element's location manager (0: not an array element)
  uint64_t sender_token;
  uint64_t tag;   // the sender's put() tag; also the message's reference number, which keeps
                  // its low CMK_REFNUM_TYPE bits (16 by default)
  uint64_t sender_elem;    // RESOLVE: the sending element's id
};

// The kind of memory a channel's buffer is. Only device memory is implemented.
enum class CkMem : int { Host = 0, Device = 1 };

// What CkPut takes beyond the channel, the source and the size: the stream that
// produces the source (device data), the callback invoked when the source may
// be reused, and the tag the receiver's event carries.
class CkPutOpts {
public:
  CkPutOpts() : stream_(0), done_(CkCallback::ignore), tag_(0) {}
  CkPutOpts& stream(hapiStream_t s) { stream_ = s; return *this; }
  CkPutOpts& done(const CkCallback& cb) { done_ = cb; return *this; }
  CkPutOpts& tag(uint64_t t) { tag_ = t; return *this; }
  hapiStream_t stream_;
  CkCallback done_;
  uint64_t tag_;
};

// A channel: one fixed destination buffer on the receiver (its owner),
// exported once by the constructor. The receiver keeps its object; copies
// reach the senders by pup, as an entry-method argument. A sender writes with
// CkPut, which carries nothing but the bytes, a sequence number and a tag, in
// whichever transport the two PEs call for: a batched device copy within a
// process, a device copy through the IPC mapping between processes on one
// host, and between hosts (or where the IPC handle does not open) an announce
// the receiver answers by reading the bytes with RDMA. The receiver's
// `arrived` callback fires once the bytes are in place -- a CkCallback to a
// chare, so it follows the receiver wherever it lives -- with a CkDataMsg
// holding a CkChannelEvent.
//
// The sender must not overwrite the buffer while the receiver may still read
// it. The receiver calls release() after consuming a put, which credits the
// sender; a put issued before the credit for the previous one arrived is
// queued, one deep, and issued when it does, never blocking the PE.
// Overlapping two puts means two channels on two buffers. One sender per
// channel.
//
// Migration, without application re-exchange. A sender resolves the channel
// before its first put: a RESOLVE event to the receiver's element (so it
// reaches the receiver wherever it lives), answered with the current endpoint,
// its generation and whether the buffer is free. When the receiving element is
// chosen to migrate, the runtime tells the resolved sender the channel is
// closing and defers the element's move until the sender has acknowledged -- a
// sender acknowledges once it has no put in flight -- so no put lands in a
// buffer that is being freed. The receiver re-opens at its new home with
// reopen(new buffer) in its pup, which bumps the generation; the sender's next
// put re-resolves. A put's bytes that landed before the pack are in the buffer
// the application pups, as its other device state (unseenBytes()); its
// notification is forwarded to the new home. A sending element does not
// migrate while one of its puts is in flight, and a migrated sender
// re-resolves before its next put.
//
// A sender never waits for the move. A put made while its channel is closed
// is copied to a staging buffer on the producing stream, the source is
// released to the application once that copy has read it, and the staging
// buffer is announced to the receiving element: an element message, which the
// location manager forwards, so it waits at the new home and is read into the
// reopened buffer in put order. Every put behind it goes the same way until
// the receiver has read the last staged one, so a direct put never races a
// read. No step of the protocol relies on two messages arriving in order.
//
// Teardown. destroy(cb) returns at once and asks the resolved sender to
// detach: the sender waits for its puts into the buffer to complete, releases
// its IPC view of it and acknowledges (the closing and ack of a migration,
// marked as a destroy). An announced put not yet read is dropped and its
// source handed back. Then the owner deregisters the buffer and invokes cb,
// after which the buffer may be freed. An owner object destroyed without the
// call starts the same teardown, with no callback. A sender learns of the
// destroy from that closing, or from the answer to its next attach; a put
// after it aborts.
class CkChannel;
// Sender side: write bytes of src into the channel's buffer.
CkDeviceStatus CkPut(CkChannel& ch, const void* src, size_t bytes, const CkPutOpts& opts = CkPutOpts());

class CkChannel {
public:
  CkChannel();
  // Receiver side: the destination buffer, its capacity, the kind of memory
  // and its device, and the entry that learns of each put. The entry takes a
  // CkDataMsg*, hands it to event() and deletes it. Construct the channel
  // inside an entry method or constructor of the receiving element: it exports
  // the buffer then.
  CkChannel(void* dst, size_t capacity, CkMem kind, int device, const CkCallback& arrived);
  // A copy takes over the original's registrations with the runtime: the
  // usual shape is `peer = param;` in an entry method, after which the
  // parameter dies. The original is left holding nothing, for an owner and a
  // sender's copy alike: every call on it but the queries below aborts.
  CkChannel(const CkChannel& o);
  CkChannel& operator=(const CkChannel& o);
  ~CkChannel();

  // Receiver side.
  void reopen(void* new_dst);  // after migration: the same channel on the new buffer
  // Teardown (see above): cb is invoked once the buffer may be freed.
  void destroy(const CkCallback& cb = CkCallback(CkCallback::ignore));
  // The put described by the message `arrived` delivered (the message stays the
  // caller's), or NULL for an event the runtime consumed (an attach, an
  // announce, or any event once the channel is destroyed).
  const CkChannelEvent* event(CkDataMsg* m);
  // The buffer may be overwritten: credit the sender.
  void release();
  // A sender's copy: attach now (look up the receiver's current endpoint, and
  // check the one-sender rule), e.g. at setup or right after the holding
  // element migrated, instead of at the next put.
  void attach();
  // The owning element's pup, on unpack: the channel arrived and reopens later.
  void arriving();

  bool putQueued() const { return pending.valid; }
  bool isDestroyed() const { return destroyed; }
  // A sender's copy: it holds a resolution (the buffer's endpoint, and through
  // it possibly an IPC view of the buffer). False for a copy that has not
  // attached since it was made or moved, and once told of a destroy.
  bool isAttached() const { return resolved; }

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

  // Runtime side (the Converse handlers and migration hooks in ckdevicechannel.C).
  void onCredit(uint64_t seq, uint64_t epoch_);
  void onClosing(uint64_t gen_, int owner_pe, uint64_t epoch_, int flags);
  void onResolved(const CkDeviceChannelEndpoint& ep, uint64_t gen_, int owner_pe, bool free,
                  uint64_t epoch_, int credit_pe, int credit_slot_, uint64_t taken_seq_);
  void onCloseAck(int pe, uint64_t token, uint64_t last_seq, uint64_t last_bytes);
  void onSrcDone(uint64_t seq);
  void onDone(uint64_t seq);         // the receiver finished reading announced put `seq`
  void onStagedDone(uint64_t seq);   // the receiver read a staged put
  void onPulled();
  void pollShmCredit();
  // Element ids are unique only within one location manager (one array, or
  // bound arrays): an element is (manager, id).
  bool ownedBy(int mgr_idx, CmiUInt8 element) const {
    return is_owner && owner_known && owner_id == element && owner_mgr.idx == mgr_idx;
  }
  CmiUInt8 ownerElement() const { return owner_id; }
  void closingForMigration(CkLocRec* rec);
  void packForMigration();
  // One line of state when something is stuck (the watchdog, CHARM_DEBUG_CHANNEL).
  void describeStall(const char* role) const;

  // The last events on this object (both roles), kept always and printed by
  // describeStall: what a stall needs explained without a full trace.
  struct Hist { uint8_t kind; int32_t pe; int32_t slot; int32_t n; uint64_t seq; uint64_t epoch; double t; };
  enum { HIST_N = 24 };

private:
  friend CkDeviceStatus CkPut(CkChannel& ch, const void* src, size_t bytes, const CkPutOpts& opts);
  // Sender side: write bytes of src, produced on stream, into the buffer.
  // src_done fires when src may be reused. tag travels in the event and is set
  // as the reference number of the message `arrived` receives, so an SDAG
  // `when arrived[step](CkDataMsg* m)` matches puts by step.
  CkDeviceStatus put(const void* src, size_t bytes, hapiStream_t stream, const CkCallback& src_done,
                     uint64_t tag);
  void open();                 // export the buffer; assign the channel id
  void noteHolder();           // the sending element, at the first put or attach
  void startDestroy(const CkCallback& cb, bool hold);
  void destroyCheck();
  void sendDestroyed(int pe, uint64_t token);   // an attach after the destroy: the sender learns of it
  void dropAnnounce(const CkChannelEvent& ev); // an announced put the destroyed owner will not read
  void sendClosing(CkLocRec* hold_rec);
  void issue(const void* src, size_t bytes, hapiStream_t stream, const CkCallback& src_done,
             uint64_t tag);
  void issueStaged(const void* src, size_t bytes, hapiStream_t stream, const CkCallback& src_done,
                   uint64_t tag);
  void creditedPut(uint64_t seq);
  void requestResolve();
  void awaitShmCredit();
  void replyResolve(int pe, uint64_t token, int mgr, CmiUInt8 elem);
  void checkOneSender(int pe, int mgr, CmiUInt8 elem) const;
  void retireCreditSlot();
  void startPull();
  void sendCloseAck();
  void onAnnounce(const CkChannelEvent& ev, const CkDeviceChannelEndpoint& src);
  void takeOver(const CkChannel& o);
  void checkLive(const char* call) const;   // aborts on an original left by a copy
  void retire();
  void registerOwner();
  void unregisterOwner();
  void releaseDstView();
  void retireSource(uint64_t through_seq);
  void closeSources();
  void note(uint8_t kind, uint64_t seq, uint64_t epoch_, int pe = -1, int slot = -1, int n = 0);
  void printHist(const char* role) const;

  CkDeviceChannelEndpoint dst;   // the receiver's buffer, exported by open()
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
  // messages. Owner: the credit slot it granted to the current sender; sender:
  // where to read its credits, and how many it has consumed from that word.
  int credit_slot_pe;       // owner: node-local PE whose area holds the slot (-1: message credits)
  int credit_slot;          // owner: the slot
  int credit_shm_pe;        // sender: the owner's area (-1: message credits)
  int credit_shm_slot;      // sender: the slot
  uint32_t credits_seen;    // sender: credits already taken from the word
  bool shm_waiting;         // sender: listed for the per-PE credit poll (a put is queued)
  size_t landed_unseen;     // see unseenBytes()
  uint64_t landed_seq;      // owner: the put those bytes are (from the close ack), 0 once taken
  size_t last_put_bytes;    // sender: bytes of the last put written into the buffer (carried by the closing ack)
  uint64_t last_direct_seq; // sender: that put's seq. Announced puts are read, never written.
  int sender_pe;            // the resolved sender: where credits and closings go
  uint64_t sender_token;
  bool sender_known;
  int sender_mgr;           // the resolved sender's element (location manager, id); mgr 0: not known
  CmiUInt8 sender_elem;
  uint64_t closing_gen;     // generation under which a migration is pending (closings sent), 0 if none
  bool closing_sent;        // a closing to the current sender is awaiting its ack
  bool sender_closed;       // the current sender acked a closing and has not resolved since
  // Announces waiting for the buffer, the read in progress, and its source.
  struct Queued { CkChannelEvent ev; CkDeviceChannelEndpoint src; };
  std::vector<Queued> pulls;
  bool pulling;
  CkChannelEvent pull_ev;
  CkDeviceChannelEndpoint pull_src;
  bool resolved_during_closing;   // a resolve was answered while that closing was in flight: close the new resolution on the ack
  uint64_t epoch;           // owner: resolves answered so far; each Resolved, Closing and Credit names the resolution it belongs to.
                            // sender: the resolution the current endpoint came from
  CkLocRec* closing_rec;    // the element standing down for the pending migration
  struct Waiting { int pe; uint64_t token; int mgr; CmiUInt8 elem; };
  std::vector<Waiting> waiting;   // resolves received while migrating, answered after reopen
  CkChannelEvent last;
  bool have_last;
  // Sender (holder) side.
  bool resolved, resolving;
  uint64_t send_seq;
  int credits;
  int uncredited;           // puts issued under this resolution not yet credited back
  bool pull_mode;           // the receiver reads (another host, or an IPC handle that does not open here)
  int tier;                 // dst.mode() at the resolution: 0 copy, 1 IPC, 2 read by the receiver
  hapiStream_t last_stream = 0;   // the stream of the last put, for the stall report
  std::map<uint64_t, CkCallback> done_cbs;   // source-done callbacks awaiting the receiver's DONE
  uint64_t closed_epoch;    // the newest resolution a closing was acked for; a Resolved or Credit of it (or older) is stale
  int in_flight;
  int staged_outstanding;   // puts staged and announced, not yet read by the receiver (its DONE)
  uint64_t acct_seq;        // the last put the receiver had taken at the current resolution's reply; its free flag covers puts up to it
  bool occupant_credit_due; // the put held at that reply is still to be credited (the first credit of this resolution is its)
  std::vector<uint64_t> direct_issued;   // direct puts issued and not yet credited back, in order
  std::vector<uint64_t> dones_seen;      // staged puts the receiver read while this copy was unresolved
  bool close_ack_due;
  int close_ack_pe;
  uint64_t my_token;        // this copy's entry in its PE's token table, 0 if none
  CkLocRec* sender_rec;     // the sending element, charged with the puts in flight
  CmiUInt8 holder_elem;     // that element's id and location manager: a staged put's DONE
  CkGroupID holder_mgr;     //   releases the element's hold even if this copy is gone
  bool holder_known;
  // Teardown (destroy()).
  bool destroying;          // owner: destroy() called; the sender's detach (its closing ack) or a read awaited
  bool destroyed;           // owner: torn down; a sender's copy: told the owner destroyed the channel
  bool destroy_hold;        // owner: the element is held in place until the teardown completes
  bool self_owned;          // a heap copy that finishes the teardown of an owner destroyed without the call
  bool closing_held;        // owner: the closing out holds the element in place (a migration's)
  CkCallback destroy_cb;
  // CHARM_CHANNEL_STATS: when the pending closing went out (owner), when the attach was asked (sender).
  double closing_t0;
  double resolve_t0;
  struct Pending {
    const void* src;
    size_t bytes;
    hapiStream_t stream;
    CkCallback done;
    uint64_t tag;
    bool valid;
  } pending;
  // The source the receiver reads in pull mode, kept and registered across puts
  // of the same buffer; a replaced one is closed once the receiver has read
  // every put announced from it (its DONE).
  CkDeviceChannelEndpoint src_ep;
  uint64_t src_ep_last;     // the last put announced from src_ep
  struct Retired { CkDeviceChannelEndpoint ep; uint64_t last_seq; };
  std::vector<Retired> retired_srcs;
  std::vector<CkDeviceChannelEndpoint> stale_views;   // views of replaced endpoints, released once nothing is in flight
  Hist hist[HIST_N];
  uint32_t hist_n = 0;
  uint64_t moved_from = 0;  // nonzero: this object was copied and the copy holds this channel id
};

// Transfers between two endpoints with no channel around them, for CkDyn's
// persistent device buffers (ckdyn.C). done runs on this PE once the bytes are
// in the destination. Get: dst is this process's buffer; put: src is. A holder
// done with a peer's endpoint forgets it (its IPC reference and the kept
// registration go).
void CkDeviceEndpointGet(const CkDeviceChannelEndpoint& dst, const CkDeviceChannelEndpoint& src, size_t bytes,
                         hapiStream_t stream, const CkCallback& done);
void CkDeviceEndpointPut(const CkDeviceChannelEndpoint& dst, const CkDeviceChannelEndpoint& src, size_t bytes,
                         hapiStream_t stream, const CkCallback& done);
void CkDeviceEndpointForget(const CkDeviceChannelEndpoint& peer);

// Converse handlers of the channel protocol and the endpoint transfers; registered in init.C.
extern "C" {
  void* device_channel_credit_bridge(void* arg);
  void* device_channel_closing_bridge(void* arg);
  void* device_channel_closeack_bridge(void* arg);
  void* device_channel_resolved_bridge(void* arg);
  void* device_channel_done_bridge(void* arg);
  void* device_channel_probe_bridge(void* arg);
  void* device_endpoint_reg_req_bridge(void* arg);
  void* device_endpoint_reg_rep_bridge(void* arg);
  void* device_endpoint_put_bridge(void* arg);
  void* device_endpoint_ack_bridge(void* arg);
}
extern int device_channel_credit_handler;
extern int device_channel_closing_handler;
extern int device_channel_closeack_handler;
extern int device_channel_resolved_handler;
extern int device_channel_done_handler;
extern int device_channel_probe_handler;
extern int device_endpoint_reg_req_handler;
extern int device_endpoint_reg_rep_handler;
extern int device_endpoint_put_handler;
extern int device_endpoint_ack_handler;
// Per PE, from init.C: the tables, the polls, and +gpupersistverbose.
void CkDeviceChannelInit(char** argv);
// ArrayElement::ckDeviceChannelControl: a control event for one of this element's channels.
void CkDeviceChannelControl(void* msg);
// From CkLocMgr::emigrate: tell the senders of the channels this element owns
// that the channel is closing (each awaited ack is outstanding channel work of
// the element, which defers the move); then, once the move goes ahead, retire
// the channels' exports before the pack.
void CkDeviceChannelsBeforeMigrate(CkLocRec* rec);
void CkDeviceChannelsPack(CkLocRec* rec);
// From CkCallstackPop: launch the entry method's batched puts while its object is still the running one.
void CkDeviceChannelBatchFlush();

#endif
