/*
 * ckcoro: Charm++ entry methods as C++20 coroutines. Requires C++20.
 *
 * A coroutine entry method is declared [coroutine] in the .ci file (void;
 * a message or marshalled parameters, taken by value in C++ and supplied
 * again on a restart) and defined in C++ with the return type
 * ck::coroutine. Its body is a CK_MIGRATABLE { ... } block, which suspends
 * at CK_WHEN sites until a delivery to an [inbox] entry matches. The chare puts the generated X_CORO_CODE macro in its
 * class: it defines the ck::coro::state member (the active coroutines and
 * the inbox, packed with the element), the bodies of the inbox entries, and
 * inbox::name() / inbox::name(ref), which name the entry at a site.
 *
 *   // .ci: entry [coroutine] void doStep();
 *   //      entry [inbox] void recvGhost(GhostMsg *m);
 *   //      entry [inbox] void recvCount(int iter, int count);
 *   class Jacobi : public CBase_Jacobi {
 *     Jacobi_CORO_CODE
 *     int imsg, iter, i, count;      // state across sites: members, pupped
 *     ck::coroutine doStep() {
 *       CK_MIGRATABLE {
 *         sendGhosts();
 *         for (imsg = 0; imsg < 6; imsg++) {
 *           CK_WHEN_BLOCK(inbox::recvGhost(iter), gmsg) { // when recvGhost[iter](GhostMsg *gmsg) {
 *             process(gmsg);
 *           }
 *         }
 *         CK_WHEN(inbox::recvCount(iter), i, count); // when recvCount[iter](int i, int count)
 *         compute();
 *       }
 *     }
 *   };
 *
 * A wait has two forms. CK_WHEN_BLOCK(want, name...) { ... } declares the
 * entry's parameters, in order, as variables of the block: the message
 * pointer, or every marshalled parameter (an array as a pointer into the
 * message); the block may not contain a site (the compiler rejects the
 * restart jump past the declarations). CK_WHEN(want, out...) stores the
 * delivery into members of the chare that follow the want, in the same
 * order, a parameter used only for matching included (ck::coro::skip
 * discards one); an array parameter into a pointer member (valid until
 * the next wait) or a container member (copied). A received message is
 * owned by the coroutine: delete it before the next wait or pup it. A
 * value needed after a later wait is a member, because a member is pupped
 * with the element and a local does not survive a restart (see Migration
 * below). inbox::name(ref) matches the reference number:
 * the envelope's for a message entry, the first arithmetic parameter for a
 * marshalled entry; inbox::name() matches any. Deliveries that no coroutine
 * is waiting for are buffered as messages (a marshalled entry's as it
 * arrived) and move with the element. The message of a [coroutine] entry
 * belongs to the state, as the message holding its marshalled parameters
 * does: the body reads it until the coroutine ends and does not delete
 * it. CK_WHEN_ALL(CK_ALT(...), ...) waits for all of several (an SDAG
 * when a(), b()); CK_WHEN_ANY(idx, CK_ALT(...), CK_ALL(...), ...) waits for
 * the first of several alternatives, each a when or a conjunction (an SDAG
 * case). Through a reduction wrapper the reference number is the
 * reduction message's.
 *
 * Without the attributes, the same works by hand: a ck::coro::state member
 * pupped by the chare, inbox entries whose bodies call co.deliver(ep, msg)
 * or co.deliver(ep, args...), CK_COROUTINE_STATE(co, ep) as the first
 * statement of the body (the promise registers the coroutine only through
 * X_CORO_CODE), then CK_MIGRATABLE { ... } with CK_WHEN(ep, ref, ...) sites,
 * ep = CkIndex_X::name(...).
 *
 * Migration and checkpoint. A coroutine frame is never packed. The state
 * object packs, for each active coroutine, the entry index, a callback to
 * the owner and the chain of restart sites, and on unpacking sends the
 * entry's message to the owner again: the body runs again from the top
 * and the switch opened by CK_MIGRATABLE jumps to the recorded site, where
 * CK_WHEN finds the buffered message or waits for it. CK_MIGRATABLE expands
 * to a frame guard and `switch (site) case 0:` followed by the block, so a
 * site is a case of that switch and a restart that reaches none aborts. Rules: (1) a value
 * that is live across a site is a member of the chare (C++ rejects the jump
 * past a local with an initializer; a scalar declared without one is not
 * caught, and +ck_fakemigrate is the check); (2) code before a site is
 * skipped on a restart, so its results must be members; (3) a chare may
 * migrate or be checkpointed only while its coroutines are suspended at
 * sites (CK_WHEN, CK_WHEN_ANY, CK_CALL); a plain co_await is allowed between
 * sites, but packing a chare suspended there aborts. A site is one
 * statement and may be the body of an if or a loop; its case label must not
 * be inside a switch of the user's own; two sites must not share a line. A
 * function awaited from a coroutine that contains a site returns
 * ck::coro::task<T>, has its own CK_MIGRATABLE block (co_return inside it)
 * and is awaited with CK_CALL or CK_CALL_TO. Every coroutine body is a
 * CK_MIGRATABLE block, since each site is a case of the switch it opens.
 *
 * The owner is a chare of any kind: an array element, a group or nodegroup
 * branch (its inbox and coroutine entries are exclusive, added by charmxi),
 * a singleton chare or the mainchare. At most one coroutine per entry
 * method is active in a chare at a time.
 * Inbox entries take a message or marshalled parameters (not conditional
 * or nocopy ones). CK_OVERLAP(task...) and CK_FORALL(i, lo, hi, expr) run
 * ck::coro::task<void> helpers as child coroutines, each with its own
 * sites, and wait for all of them (SDAG overlap, forall); a child that
 * finished before a migration is not run again, so its results live in
 * pupped members.
 *
 * +ck_fakemigrate on the command line makes every site rebuild the chare's
 * coroutines in place (state packed and unpacked, frames destroyed, entries
 * invoked again to the recorded sites) the first time a coroutine suspends
 * there, so programs can check the rules without a load balancer.
 */
#ifndef CKCORO_H
#define CKCORO_H

#if !defined(__cplusplus) || __cplusplus < 202002L
#error "ckcoro.h requires C++20: compile with -std=c++20"
#endif

#include <coroutine>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <list>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include "charm++.h"
#include "pup_stl.h"
#include "ckcoro_core.h"

namespace ck {
namespace coro {

class state;
struct record;

// The record of the coroutine running on this PE thread, if any.
inline thread_local record *current_ = nullptr;

// The callback that invokes an entry of the owner again (the restart after a
// migration or a checkpoint, +ck_fakemigrate), one overload per chare kind.
// It is rebuilt from the owner whenever the state is unpacked: an array
// element keeps its index, a chare restarted from a checkpoint has a new
// handle, a group branch the PE it was unpacked on.
inline CkCallback self_target(int ep, ArrayElement *e) {
  return CkCallback(ep, e->ckGetArrayIndex(), e->ckGetArrayID());
}
inline CkCallback self_target(int ep, NodeGroup *g) {
  return CkCallback(ep, CkMyNode(), g->ckGetGroupID(), false, true);
}
inline CkCallback self_target(int ep, IrrGroup *g) {
  return CkCallback(ep, CkMyPe(), g->ckGetGroupID());
}
inline CkCallback self_target(int ep, Chare *c) { return CkCallback(ep, c->ckGetChareID()); }

// The callback that broadcasts to every element of the owner's collection
// (CK_FUTURE_ALL): the array, the group or nodegroup; a chare is its own.
inline CkCallback all_target(int ep, ArrayElement *e) { return CkCallback(ep, e->ckGetArrayID()); }
inline CkCallback all_target(int ep, NodeGroup *g) { return CkCallback(ep, g->ckGetGroupID(), true); }
inline CkCallback all_target(int ep, IrrGroup *g) { return CkCallback(ep, g->ckGetGroupID()); }
inline CkCallback all_target(int ep, Chare *c) { return CkCallback(ep, c->ckGetChareID()); }
// The frame handle of a coroutine entry just created, picked up by state::begin.
inline thread_local std::coroutine_handle<> pending_root_ = {};

template <class T>
inline constexpr bool is_message_ptr_v =
    std::is_pointer_v<T> && std::is_base_of_v<CkMessage, std::remove_pointer_t<T>>;

// The message parameter of a coroutine entry's frame: where the frame holds
// it, so that it can be pointed at the message that survives a pack (a
// custom pack replaces the message), and how to delete it (with its
// destructor, which frees what a custom unpack allocated).
struct entry_msg {
  void **slot = nullptr;
  void (*del)(void *) = nullptr;
};
template <class Msg> void delete_msg(void *m) { delete static_cast<Msg *>(m); }
template <class A> inline void note_entry_msg(entry_msg &em, A &a) noexcept {
  using P = std::remove_cvref_t<A>;
  if constexpr (is_message_ptr_v<P>) {
    static_assert(!std::is_const_v<std::remove_reference_t<A>>,
                  "the message parameter of a coroutine entry method is not const");
    em.slot = reinterpret_cast<void **>(const_cast<std::remove_cv_t<std::remove_pointer_t<P>> **>(&a));
    em.del = &delete_msg<std::remove_cv_t<std::remove_pointer_t<P>>>;
  }
}

/* ---------------- coroutine: the entry method type ---------------- */

// The return type of a coroutine entry method. The body starts at once (the
// generated entry wrapper discards the returned object); the object's
// destructor runs when the body first suspends or finishes and hands control
// to the state object.
class coroutine {
 public:
  struct promise_type {
    // A member coroutine of a chare with X_CORO_CODE registers itself: the
    // promise is constructed with the chare the member function runs on
    // (C++20 passes *this and the arguments to a promise constructor that
    // accepts them) and the record is created or adopted when the return
    // object is made, before the body starts. Without X_CORO_CODE (the
    // hand-written form) the body registers with CK_COROUTINE_STATE.
    // GCC deduces Owner as a reference type (Jacobi&) from *this, clang as
    // the class; owner_t strips it, or GCC's constraint fails and the
    // promise is default-constructed without registering (CI, 2026-11-11).
    // The arguments are the frame's copies of the parameters, so the
    // message parameter's slot is the one the body reads.
    template <class T> using owner_t = std::remove_cv_t<std::remove_reference_t<T>>;
    template <class Owner, class... Args>
      requires requires(owner_t<Owner> &o) { o.__coro; owner_t<Owner>::__coro_is_entry(0); }
    promise_type(Owner &self, Args &...args) noexcept
        : owner_(&self), begin_(&begin_for<owner_t<Owner>>) {
      (note_entry_msg(msg_, args), ...);
    }
    promise_type() noexcept = default;
    coroutine get_return_object() {
      auto h = std::coroutine_handle<promise_type>::from_promise(*this);
      pending_root_ = h;
      if (begin_) begin_(owner_, msg_);
      return coroutine(h);
    }
    std::suspend_never initial_suspend() const noexcept { return {}; }
    std::suspend_always final_suspend() const noexcept { return {}; }
    void return_void() const noexcept {}
    void unhandled_exception() const noexcept { abort_unhandled_exception(); }

   private:
    template <class Owner> static void begin_for(void *o, entry_msg em) {
      Owner *self = static_cast<Owner *>(o);
      self->__coro.begin(self, Owner::__coro_is_entry(::CkCurrentEntry()), em);
    }
    void *owner_ = nullptr;
    void (*begin_)(void *, entry_msg) = nullptr;
    entry_msg msg_;
  };
  explicit coroutine(std::coroutine_handle<> h) noexcept : h_(h) {}
  coroutine(const coroutine &) = delete;
  coroutine &operator=(const coroutine &) = delete;
  ~coroutine();

 private:
  std::coroutine_handle<> h_;
};

} // namespace coro
} // namespace ck

// A coroutine entry method takes its parameters by value: a reference would
// bind to the generated wrapper's local, which is gone after the first wait
// (the usual `const std::vector<T> &` of a marshalled entry is rejected at
// compile time; review of PR #4231).
template <class Self, class... Params>
struct std::coroutine_traits<ck::coro::coroutine, Self, Params...> {
  static_assert((!std::is_reference_v<Params> && ...),
                "a coroutine entry method takes its parameters by value: a reference "
                "parameter dangles after the first wait");
  using promise_type = ck::coro::coroutine::promise_type;
};

namespace ck {
namespace coro {

/* ---------------- the inbox ---------------- */

// True for the message that carries an entry's marshalled parameters.
inline bool is_marshall_msg(const void *m) noexcept {
  return m != nullptr && UsrToEnv(m)->getMsgIdx() == CMessage_CkMarshallMsg::__idx;
}

// One buffered delivery: the entry's message, or a CkMarshallMsg holding the
// marshalled parameters, or no message (a void entry). The state owns it.
struct buffer {
  int ep = -1;
  CMK_REFNUM_TYPE ref = 0;
  void *msg = nullptr;

  buffer() = default;
  buffer(buffer &&o) noexcept : ep(o.ep), ref(o.ref), msg(std::exchange(o.msg, nullptr)) {}
  buffer &operator=(buffer &&o) noexcept {
    if (this != &o) {
      release();
      ep = o.ep; ref = o.ref; msg = std::exchange(o.msg, nullptr);
    }
    return *this;
  }
  buffer(const buffer &) = delete;
  buffer &operator=(const buffer &) = delete;
  ~buffer() { release(); }
  void release() { if (msg) { CkFreeMsg(msg); msg = nullptr; } }

  void pup(PUP::er &p) {
    p | ep; p | ref;
    bool hasMsg = msg != nullptr;
    p | hasMsg;
    if (hasMsg) CkPupMessage(p, &msg);
  }
};

// What a when site waits for: a delivery to entry ep, with reference number
// ref or with any (noref). The generated X_CORO_CODE macro's inbox::name()
// and inbox::name(ref) functions make these.
struct want {
  int ep;
  bool hasRef;
  CMK_REFNUM_TYPE ref;
  int arm = 0;   // which alternative of a when_any this want belongs to
  explicit want(int ep_) noexcept : ep(ep_), hasRef(false), ref(0) {}
  want(int ep_, CMK_REFNUM_TYPE ref_) noexcept : ep(ep_), hasRef(true), ref(ref_) {}
  bool matches(const buffer &b) const noexcept {
    return b.ep == ep && (!hasRef || b.ref == ref);
  }
};

struct noref_t {};
inline constexpr noref_t noref{};

// The parameter list of an inbox entry, as the generated inbox::name_params
// type: each marshalled parameter's value type, an array parameter as
// array_param<T>, a message entry as the message pointer, a void entry as
// params<>. A want_t<P> is a want that carries it, so that a when can check
// its outs against the entry and unpack arrays.
template <class... Ps> struct params {};
template <class T> struct array_param {};
template <class P> struct want_t : want {
  using want::want;
  using params_type = P;
};
template <class T> using value_t = std::remove_cv_t<std::remove_reference_t<T>>;
template <class T> inline constexpr bool is_array_param_v = false;
template <class T> inline constexpr bool is_array_param_v<array_param<T>> = true;
template <class P> struct out_of { using type = P; };
template <class T> struct out_of<array_param<T>> { using type = T *; };
template <class P> using out_t = typename out_of<P>::type;

// A future receives the deliveries of a callback to the chare's
// _coro_callback entry (an [inbox] entry charmxi adds to a chare with a
// [coroutine] entry), whose reference number is the future's id; CK_WHEN(f,
// msg) waits for one delivery. CK_FUTURE(f) makes a callback to this element
// only. CK_FUTURE_ALL(f) makes one that broadcasts to every element of the
// collection, for a reduction or anything every element awaits: its id
// counts the element's CK_FUTURE_ALL calls, so the elements agree on it when
// each makes the same calls, as with contribute. The top bit of the id tells
// the two kinds apart. A future waited for across a migration is a pupped
// member, as any value live across a wait.
struct future {
  int ep = -1;            // the _coro_callback entry; -1 until CK_FUTURE
  CMK_REFNUM_TYPE id = 0;
  void pup(PUP::er &p) { p | ep; p | id; }
};
// The schema of a future's delivery: a message, stored into one message
// pointer out, or discarded when the when has no out.
struct future_msg {};

// An array argument handed to deliver() by a generated inbox body reached
// otherwise than through its own message (a reduction target, a direct
// call): the pointer and the byte count, packed in charmxi's layout.
template <class T> struct array_arg {
  const T *data;
  size_t bytes;
};
template <class T> inline constexpr bool is_array_arg_v = false;
template <class T> inline constexpr bool is_array_arg_v<array_arg<T>> = true;

// An out that discards its parameter: CK_WHEN(inbox::recv(iter), ck::coro::skip, values).
struct skip_t {};
inline skip_t skip;

/* ---------------- records of active coroutines ---------------- */

struct record {
  state *owner = nullptr;
  int ep = -1;                   // the coroutine entry; pupped
  CkCallback target;             // the owner (self_target); pupped, rebuilt by state::pup(p, owner)
  void *msg = nullptr;           // the entry's message, for a restart; pupped (null: a void entry).
                                 // A message entry's is a packed copy, a marshalled entry's
                                 // the message its parameters point into
  chain ch;                      // restart sites; its sites pupped
  std::coroutine_handle<> root;  // the entry's frame
  void *frameMsg = nullptr;      // a message entry's: the message the frame received, deleted
  void (*frameMsgDel)(void *) = nullptr; // with the frame
  std::coroutine_handle<> h;     // the innermost suspended frame
  record *prev = nullptr;        // current_ before this record ran
  chain *prevChain = nullptr;    // current_chain_ before this record ran
  bool waiting = false;          // suspended in a when awaitable
  bool atSite = false;           // ... reached through CK_WHEN
  std::vector<want> wants;       // what it waits for, grouped by arm (when_any alternatives)
  bool ready = false;            // delivered: which arm and its buffers
  int readyIndex = -1;
  buffer ready0;                 // the first (usually only) delivery of the arm:
  std::vector<buffer> readyMore; // kept out of the vector, which costs on the hot path
  buffer &ready_buf(size_t k) { return k == 0 ? ready0 : readyMore[k - 1]; }
  void set_ready(std::vector<buffer> &&bufs) {
    ready0 = std::move(bufs[0]);
    readyMore.clear();
    for (size_t k = 1; k < bufs.size(); k++) readyMore.push_back(std::move(bufs[k]));
  }
  void clear_ready() { ready0.release(); readyMore.clear(); }
  bool fakeRestart = false;      // rebuild once it has suspended (+ck_fakemigrate)
  bool rebuilt = false;          // restarted and not yet past the recorded site: no rebuild there
  bool pending = false;          // a restart message is on its way to the element; pupped
  std::vector<buffer> held;      // the deliveries of the last wait, kept while array outs point into them
  // An overlap (CK_OVERLAP, CK_FORALL) runs its tasks as child records of
  // the record at the site: each has its own chain, pupped with the parent,
  // and is created again by the parent's restart. A finished child stays
  // (done, pupped) so that a restart does not run it again.
  std::list<record> kids;
  record *parent = nullptr;
  bool done = false;
  int kidsRunning = 0;

  void pup(PUP::er &p) {
    if (parent == nullptr) {
      p | ep;
      p | target;
      p | pending;
      // A restart message in flight when a checkpoint was taken does not
      // survive the restart from it: send it again on unpacking.
      if (p.isUnpacking() && p.isCheckpoint()) pending = false;
      bool hasMsg = msg != nullptr;
      p | hasMsg;
      if (hasMsg) CkPupMessage(p, &msg);
    }
    p | done;
    ch.pup(p);
    size_t nkids = kids.size();
    p | nkids;
    if (p.isUnpacking()) kids.resize(nkids);
    for (record &k : kids) {
      k.parent = this;
      k.owner = owner;
      k.pup(p);
    }
  }
};

class state;
inline state &current_state();

/* ---------------- when awaitable ---------------- */

// The record of the running coroutine, entering a when or when_any: a
// restarted body has reached its site; remember whether the await was
// reached through a site macro.
inline record *enter_await();

// Schema: the entry's params<...> when the want came from the generated
// inbox struct (the outs are checked against it, arrays unpacked), else void.
template <class Schema, class... Outs> class when_awaiter {
 public:
  when_awaiter(state &s, want w, Outs &...outs) noexcept : s_(s), w_(w), outs_(outs...) {}
  const want &spec() const noexcept { return w_; }
  void unpack(buffer &b, record *rec);
  bool await_ready();
  void await_suspend(std::coroutine_handle<> h);
  void await_resume();
  // As an alternative of a when_any, or a member of a when_all.
  void add_wants(std::vector<want> &ws, int arm) const { ws.push_back(w_); ws.back().arm = arm; }
  bool try_ready(record *rec);                  // take from the inbox and unpack
  void unpack_ready(record *rec) { unpack(rec->ready0, rec); }

 private:
  template <class... Ps, size_t... I>
  void unpack_marshalled(params<Ps...>, std::index_sequence<I...>, buffer &b, record *rec);
  state &s_;
  want w_;
  std::tuple<Outs &...> outs_;
  record *rec_ = nullptr;
};

// The block form: the outs are the awaiter's own, returned as a tuple for a
// structured binding (CK_WHEN_BLOCK).
template <class... Ps> struct tuple_vals { std::tuple<out_t<Ps>...> vals_{}; };
template <class... Ps>
class tuple_awaiter : tuple_vals<Ps...>, public when_awaiter<params<Ps...>, out_t<Ps>...> {
  using base = when_awaiter<params<Ps...>, out_t<Ps>...>;
 public:
  tuple_awaiter(state &s, want w) noexcept
      : tuple_vals<Ps...>(),
        base(std::apply([&](auto &...v) { return base(s, w, v...); }, this->vals_)) {}
  std::tuple<out_t<Ps>...> await_resume() {
    base::await_resume();
    return std::move(this->vals_);
  }
};

// Await several deliveries together (SDAG when a(), b()): ready only when
// every member has a match, and then all are taken at once.
template <class... W> class all_awaiter {
 public:
  explicit all_awaiter(state &s, W &&...w) noexcept : s_(s), parts_(std::move(w)...) {}
  bool await_ready();
  void await_suspend(std::coroutine_handle<> h);
  void await_resume();
  void add_wants(std::vector<want> &ws, int arm) const {
    std::apply([&](const auto &...a) { (a.add_wants(ws, arm), ...); }, parts_);
  }
  bool try_ready(record *rec);
  void unpack_ready(record *rec) {
    std::apply([&](auto &...a) { size_t k = 0; (a.unpack(rec->ready_buf(k++), rec), ...); }, parts_);
  }

 private:
  state &s_;
  std::tuple<W...> parts_;
  record *rec_ = nullptr;
};

// Await the first of several alternatives (SDAG case), each a when or a
// when_all: the awaited value is the index of the alternative that matched,
// whose outputs are stored.
template <class... W> class any_awaiter {
 public:
  explicit any_awaiter(state &s, W &&...w) noexcept : s_(s), alts_(std::move(w)...) {}
  bool await_ready();
  void await_suspend(std::coroutine_handle<> h);
  int await_resume();

 private:
  state &s_;
  std::tuple<W...> alts_;
  record *rec_ = nullptr;
  int ready_ = -1;
};

// Await the completion of child tasks (SDAG overlap / forall). await_ready
// starts each task (or adopts its record on a restart) and is true when all
// have finished; a finishing child resumes the parent when it was the last.
class overlap_awaiter {
 public:
  overlap_awaiter(state &s, std::vector<task<void>> &&t) noexcept : s_(s), tasks_(std::move(t)) {}
  bool await_ready();
  void await_suspend(std::coroutine_handle<> h);
  void await_resume();

 private:
  state &s_;
  std::vector<task<void>> tasks_;
  record *rec_ = nullptr;
};

template <class... T> inline std::vector<task<void>> tasks(T &&...t) {
  std::vector<task<void>> v;
  v.reserve(sizeof...(T));
  (v.push_back(std::move(t)), ...);
  return v;
}

/* ---------------- state: per-element coroutines and inbox ---------------- */

class state {
 public:
  state() = default;
  state(const state &) = delete;
  state &operator=(const state &) = delete;
  ~state();

  // Inbox entry bodies (generated by [inbox], or written by hand): hand the
  // delivery to the state. A message entry hands over its message. A void
  // entry, or a marshalled entry reached other than through its proxy (a
  // reduction target, a direct call), passes its arguments, packed into a
  // new message; the reference number is the first arithmetic argument, else
  // that of the entry's own message being delivered (a reduction target's is
  // the reduction message's). Through its proxy a marshalled entry's message
  // goes to deliver_message as it arrived.
  template <class Msg, std::enable_if_t<is_message_ptr_v<Msg *>, int> = 0>
  void deliver(int ep, Msg *m);
  template <class... Args, std::enable_if_t<!(sizeof...(Args) == 1 && (is_message_ptr_v<std::decay_t<Args>> && ...)), int> = 0>
  void deliver(int ep, const Args &...args);

  // Await a delivery to entry ep with reference number ref (or any, with
  // noref), storing the message or arguments in outs. CK_WHEN wraps this.
  template <class... Outs>
  when_awaiter<void, Outs...> when(int ep, CMK_REFNUM_TYPE ref, Outs &...outs) noexcept {
    return when_awaiter<void, Outs...>(*this, want(ep, ref), outs...);
  }
  template <class... Outs>
  when_awaiter<void, Outs...> when(int ep, noref_t, Outs &...outs) noexcept {
    return when_awaiter<void, Outs...>(*this, want(ep), outs...);
  }
  template <class... Outs>
  when_awaiter<void, Outs...> when(want w, Outs &...outs) noexcept {
    return when_awaiter<void, Outs...>(*this, w, outs...);
  }
  template <class P, class... Outs>
  when_awaiter<P, Outs...> when(want_t<P> w, Outs &...outs) noexcept {
    return when_awaiter<P, Outs...>(*this, w, outs...);
  }
  // A future's delivery (CK_FUTURE): into one message pointer, or discarded.
  template <class... Outs>
  when_awaiter<future_msg, Outs...> when(const future &f, Outs &...outs) {
    return when_awaiter<future_msg, Outs...>(*this, future_want(f), outs...);
  }
  // The block form: the awaited value is a tuple of the entry's parameters
  // (for a future, its message as CkMessage *).
  template <class... Ps>
  tuple_awaiter<Ps...> when_block(want_t<params<Ps...>> w) noexcept {
    return tuple_awaiter<Ps...>(*this, w);
  }
  tuple_awaiter<CkMessage *> when_block(const future &f) {
    return tuple_awaiter<CkMessage *>(*this, future_want(f));
  }
  // A new future id of this element (CK_FUTURE), or of its collection
  // (CK_FUTURE_ALL, counted alike on every element).
  static constexpr CMK_REFNUM_TYPE kOwnFuture = (CMK_REFNUM_TYPE)1 << (8 * sizeof(CMK_REFNUM_TYPE) - 1);
  CMK_REFNUM_TYPE new_future_id(bool all) noexcept {
    if (all) return (CMK_REFNUM_TYPE)(nextAllFuture_++ & (kOwnFuture - 1));
    return (CMK_REFNUM_TYPE)(kOwnFuture | (nextFuture_++ & (kOwnFuture - 1)));
  }
  // The generated wrapper of a marshalled inbox entry: the message is handed
  // over unmarshalled; the reference number is its first parameter if that
  // is arithmetic, else the envelope's.
  template <class P> void deliver_message(int ep, CkMarshallMsg *m);
  // The first of several alternatives to match: co_await co.any(co.when(...), co.when(...)).
  template <class... W> any_awaiter<W...> any(W &&...w) noexcept {
    return any_awaiter<W...>(*this, std::move(w)...);
  }
  // All of several whens together: co_await co.all(co.when(...), co.when(...)).
  template <class... W> all_awaiter<W...> all(W &&...w) noexcept {
    return all_awaiter<W...>(*this, std::move(w)...);
  }
  // Run the tasks as child coroutines of the current one and wait for all
  // of them (SDAG overlap, forall). CK_OVERLAP and CK_FORALL wrap this.
  overlap_awaiter overlap(std::vector<task<void>> &&tasks) noexcept;

  // Pack or unpack the state. The form with the owner (what X_CORO_CODE's
  // _coro_pup calls) rebuilds each restart target from the owner on unpack;
  // the form without keeps the packed targets (the library form, p | state,
  // where the owner is an array element whose index does not change).
  void pup(PUP::er &p) { pup_impl(p, nullptr, nullptr); }
  template <class Owner> void pup(PUP::er &p, Owner *owner) {
    note_owner(owner);
    pup_impl(p, [](int ep, void *o) { return self_target(ep, static_cast<Owner *>(o)); }, owner);
  }

  // Internal.
  template <class Owner>
  record &begin(Owner *owner, int ep, void *msg, bool own = false, entry_msg em = {}) {
    note_owner(owner);
    return begin_record(ep, msg, own, em, [](int ep_, void *o) {
      return self_target(ep_, static_cast<Owner *>(o));
    }, owner);
  }
  template <class Owner> record &begin(Owner *owner, bool isEntry, entry_msg em);
  void finish_run(record *rec);
  void run(record *rec);
  bool take(const want &w, buffer &out);
  bool try_arm(const want *ws, size_t n, std::vector<buffer> &out);
  void rebuild();
  size_t num_buffered() const noexcept { return inbox_.size(); }
  size_t num_active() const noexcept { return recs_.size(); }

 private:
  using target_fn = CkCallback (*)(int ep, void *owner);
  record &begin_record(int ep, void *msg, bool own, entry_msg em, target_fn mk, void *owner);
  void pup_impl(PUP::er &p, target_fn mk, void *owner);
  void destroy_frames(record &rec);
  void send_restart(record &rec);
  void push(buffer &&b);
  want future_want(const future &f) const;
  void trace_resume(const record *rec);
  void fake_migrate();
  // The owner, for the trace event of a resumed segment and for
  // +ck_fakemigrate_pup; set whenever a coroutine begins or the state is
  // pupped with its owner.
  // An array element that uses AtSync is not moved: a migration it starts
  // while waiting at the load balancer's barrier would leave the barrier
  // waiting for it; the hand-written form (no X_CORO_CODE) is not moved.
  template <class Owner> void note_owner(Owner *o) noexcept {
    owner_ = o;
    if constexpr (std::is_base_of_v<ArrayElement, Owner>) {
      ownerIdx_ = o->ckGetArrayIndex().getProjectionID();
      ownerHasIdx_ = true;
      if constexpr (requires(const Owner &x) { x._coro_uses_at_sync(); })
        migrate_ = [](void *p, int pe) {
          Owner *e = static_cast<Owner *>(p);
          if (e->_coro_uses_at_sync()) return false;
          e->migrateMe(pe);
          return true;
        };
    }
  }

  template <class F> static bool visit(std::list<record> &recs, F &&f) {
    for (record &rec : recs) {
      if (f(rec)) return true;
      if (visit(rec.kids, f)) return true;
    }
    return false;
  }
  std::list<record> recs_;
  std::vector<buffer> inbox_;
  bool pendingRebuild_ = false;
  CMK_REFNUM_TYPE nextFuture_ = 0;     // pupped
  CMK_REFNUM_TYPE nextAllFuture_ = 0;  // pupped
  void *owner_ = nullptr;
  CmiObjId ownerIdx_;
  bool ownerHasIdx_ = false;
  bool (*migrate_)(void *owner, int pe) = nullptr;  // an array element's migrateMe, if it may move
  bool migrating_ = false;          // +ck_fakemigrate_pup: a migration is on its way
};

// The registration of a hand-written coroutine entry (CK_COROUTINE_STATE):
// creates or adopts the record.
class root_guard {
 public:
  template <class Owner> root_guard(state &s, Owner *owner, int ep, std::nullptr_t) {
    s.begin(owner, ep, nullptr);
  }
  // msg is the body's message parameter (the frame's slot).
  template <class Owner, class Msg> root_guard(state &s, Owner *owner, int ep, Msg *&msg) {
    entry_msg em;
    note_entry_msg(em, msg);
    s.begin(owner, ep, msg, false, em);
  }
  ~root_guard() = default;
  root_guard(const root_guard &) = delete;
  root_guard &operator=(const root_guard &) = delete;
};

/* ======================= inline implementation ======================= */

// +ck_fakemigrate_pup: an array element migrates to the next PE instead of
// rebuilding in place, so that its own pup runs (other owners never migrate
// and rebuild in place; one PE has nowhere to go).
inline bool fake_migrate_pup_enabled() noexcept {
  return _ckCoroFakeMigratePup && CkNumPes() > 1;
}

inline bool fake_migrate_enabled() noexcept {
  static thread_local int v = -1;
  if (v < 0) {
    v = (_ckCoroFakeMigrate || _ckCoroFakeMigratePup) ? 1 : 0; // the flags are parsed in init.C
    if (v && CkMyPe() == 0) {
      if (fake_migrate_pup_enabled())
        CkPrintf("ck::coro> +ck_fakemigrate_pup: at every when site an array element migrates "
                 "to the next PE, other chares rebuild their coroutines in place\n");
      else if (_ckCoroFakeMigratePup)
        CkPrintf("ck::coro> +ck_fakemigrate_pup needs two or more PEs: every when site "
                 "rebuilds the chare's coroutines in place\n");
      else
        CkPrintf("ck::coro> +ck_fakemigrate: every when site rebuilds the chare's coroutines in place\n");
    }
  }
  return v == 1;
}

// CKCORO_DEBUG=1 in the environment prints rebuilds and restarts.
inline bool debug_enabled() noexcept {
  static thread_local int dbg = -1;
  if (dbg < 0) dbg = getenv("CKCORO_DEBUG") != nullptr;
  return dbg == 1;
}

inline state &current_state() {
  if (current_ == nullptr)
    CkAbort("ck::coro: CK_WHEN used outside a coroutine entry method");
  return *current_->owner;
}

inline coroutine::~coroutine() {
  record *rec = current_;
  if (rec == nullptr || rec->root != h_) {
    // The coroutine was never registered (no X_CORO_CODE and no
    // CK_COROUTINE_STATE).
    if (h_.done()) { h_.destroy(); return; }
    CkAbort("ck::coro: a coroutine entry method suspended without being registered (a chare "
            "without X_CORO_CODE registers with CK_COROUTINE_STATE at the start of the body)");
  }
  rec->owner->finish_run(rec);
}

inline state::~state() {
  for (record &rec : recs_) destroy_frames(rec);
  // ~buffer frees buffered messages; ~record frees nothing: msg is freed here
  for (record &rec : recs_) if (rec.msg) CkFreeMsg(rec.msg);
}

inline void state::destroy_frames(record &rec) {
  for (record &k : rec.kids) destroy_frames(k);
  if (rec.root) {
    rec.ch.destroying = true;
    rec.root.destroy();
    rec.ch.destroying = false;
  }
  rec.root = {};
  rec.h = {};
  if (rec.frameMsg) {
    if (rec.frameMsgDel) rec.frameMsgDel(rec.frameMsg);
    else CkFreeMsg(rec.frameMsg);
  }
  rec.frameMsg = nullptr;
  rec.frameMsgDel = nullptr;
  rec.ch.depth = 0;
  rec.ch.linked.clear();
  rec.ch.noted = false;
  rec.waiting = false;
  rec.atSite = false;
  rec.wants.clear();
  rec.ready = false;
  rec.clear_ready();
}

// A packed copy of the message at *pm. Packing it in place may replace it
// (a custom pack frees its input), so *pm is set to the unpacked message.
inline void *packed_copy(void **pm) {
  envelope *env = UsrToEnv(*pm);
  const bool wasPacked = env->isPacked();
  if (!wasPacked) CkPackMessage(&env);
  const int size = env->getTotalsize();
  envelope *c = (envelope *)CmiAlloc(size);
  std::memcpy(c, env, size);
  setMemoryTypeMessage(c);
  if (!wasPacked) CkUnpackMessage(&env);
  *pm = EnvToUsr(env);
  return EnvToUsr(c);
}

// An unpacked copy of a message the state holds at *pm. A packed one (a
// message entry's restart copy) is copied as bytes and only the copy is
// unpacked: CkCopyMsg would unpack the source too. An unpacked one (a
// marshalled entry's, varsize: its pointer must become an offset before a
// byte copy) goes through CkCopyMsg, which packs and unpacks it in place.
inline void *restart_copy(void **pm) {
  envelope *env = UsrToEnv(*pm);
  if (!env->isPacked()) return CkCopyMsg(pm);
  const int size = env->getTotalsize();
  envelope *c = (envelope *)CmiAlloc(size);
  std::memcpy(c, env, size);
  setMemoryTypeMessage(c);
  CkUnpackMessage(&c);
  return EnvToUsr(c);
}

inline record &state::begin_record(int ep, void *msg, bool own, entry_msg em, target_fn mk,
                                   void *owner) {
  std::coroutine_handle<> root = std::exchange(pending_root_, {});
  if (!root) CkAbort("ck::coro: CK_COROUTINE_STATE in a function that does not return ck::coroutine");
  record *rec = nullptr;
  for (record &r : recs_) {
    if (r.ep != ep) continue;
    if (r.root)
      CkAbort("ck::coro: entry %d invoked while a coroutine of that entry is active in this "
              "element (one active coroutine per entry)", ep);
    rec = &r; // restarting: adopt the record
  }
  if (rec == nullptr) {
    recs_.emplace_back();
    rec = &recs_.back();
    rec->owner = this;
    rec->ep = ep;
    rec->target = mk(ep, owner);
    if (msg && own) rec->msg = msg;
    else if (msg) {
      // A message entry: the restart copy is made by packing the frame's
      // message, which may replace it; the frame's parameter and the
      // current message follow.
      void *m = msg;
      rec->msg = packed_copy(&m);
      if (em.slot) *em.slot = m;
      if (CkpvAccess(_ckCurrentMsg) == msg) CkpvAccess(_ckCurrentMsg) = m;
      msg = m;
    }
  } else {
    rec->ch.restarting = !rec->ch.sites.empty();
    rec->pending = false;
    if (debug_enabled())
      CkPrintf("[%d] ck::coro: restarting entry %s toward its recorded site\n", CkMyPe(),
               _entryTable[ep]->name);
    if (own) {
      // The restarted body's marshalled parameters point into this copy.
      if (rec->msg) CkFreeMsg(rec->msg);
      rec->msg = msg;
    }
  }
  // A message entry's message belongs to the state while the frame lives
  // (a restart gives the new frame a copy of its own).
  if (msg && !own) {
    rec->frameMsg = msg;
    rec->frameMsgDel = em.del;
  }
  rec->root = root;
  rec->h = root;
  rec->prev = current_;
  rec->prevChain = current_chain_;
  current_ = rec;
  current_chain_ = &rec->ch;
  return *rec;
}

// Registration by the promise of ck::coroutine: the entry and its message
// are the ones being delivered (the generated __coro_is_entry says whether
// the running entry is one of the chare's [coroutine] entries). A void
// entry's system message is not kept.
template <class Owner> inline record &state::begin(Owner *owner, bool isEntry, entry_msg em) {
  if (!isEntry) {
    // CkCurrentEntry is set only in [coroutine] and [inbox] entries.
    int ep = CkCurrentEntry();
    if (ep < 0)
      CkAbort("ck::coro: a ck::coroutine member function was called directly; a [coroutine] "
              "entry method is invoked through its proxy");
    CkAbort("ck::coro: a ck::coroutine member function was called directly from entry %s; a "
            "[coroutine] entry method is invoked through its proxy", _entryTable[ep]->name);
  }
  void *m = CkCurrentMessage();
  if (m && UsrToEnv(m)->getMsgIdx() == 0) m = nullptr;
  // A marshalled [coroutine] entry is not nokeep: the state owns its
  // message, so array parameters stay valid while the coroutine lives.
  const bool marshalled = m != nullptr && is_marshall_msg(m);
  if (m && !marshalled && (em.slot == nullptr || *em.slot != m))
    CkAbort("ck::coro: the message of entry %s is not the coroutine's message parameter",
            _entryTable[CkCurrentEntry()]->name);
  return begin(owner, CkCurrentEntry(), m, marshalled, marshalled ? entry_msg{} : em);
}

// After the body ran (first run or a resume) and suspended or finished.
inline void state::finish_run(record *rec) {
  current_ = rec->prev;
  current_chain_ = rec->prevChain;
  rec->prev = nullptr;
  rec->prevChain = nullptr;
  if (rec->root.done()) {
    destroy_frames(*rec);
    if (rec->parent) {
      // A child: the parent waits for its last child.
      record *par = rec->parent;
      rec->done = true;
      if (--par->kidsRunning == 0 && par->waiting) {
        par->waiting = false;
        run(par);
      }
    } else {
      if (rec->msg) CkFreeMsg(rec->msg);
      rec->msg = nullptr;
      for (auto it = recs_.begin(); it != recs_.end(); ++it)
        if (&*it == rec) { recs_.erase(it); break; }
    }
  } else {
    if (!rec->h) CkAbort("ck::coro: coroutine suspended at an awaitable that is not ck::coro's");
    if (rec->fakeRestart) {
      rec->fakeRestart = false;
      pendingRebuild_ = true;
    }
  }
  // A rebuild destroys every frame, so it waits until no coroutine of this
  // element is on the stack (a child suspending inside its parent's start).
  if (current_ == nullptr && pendingRebuild_) {
    pendingRebuild_ = false;
    fake_migrate();
  }
}

// What +ck_fakemigrate does once no coroutine of the chare is on the stack:
// rebuild in place, or under +ck_fakemigrate_pup move an array element to
// the next PE (migrateMe sends the element a message; it moves when that is
// handled, suspended at whichever site it has reached by then).
inline void state::fake_migrate() {
  if (migrating_) return;
  if (migrate_ && fake_migrate_pup_enabled() && migrate_(owner_, (CkMyPe() + 1) % CkNumPes())) {
    migrating_ = true;
    if (debug_enabled())
      CkPrintf("[%d] ck::coro: migrating to PE %d (+ck_fakemigrate_pup)\n", CkMyPe(),
               (CkMyPe() + 1) % CkNumPes());
    return;
  }
  rebuild();
}

// Resume rec (its innermost suspended frame) as the running coroutine.
inline void state::run(record *rec) {
  rec->prev = current_;
  rec->prevChain = current_chain_;
  current_ = rec;
  current_chain_ = &rec->ch;
  std::coroutine_handle<> h = std::exchange(rec->h, {});
  trace_resume(rec);
  h.resume();
  _TRACE_END_EXECUTE();
  finish_run(rec);
}

// A resumed segment is traced as an execution of the coroutine entry,
// nested in the delivery that resumed it, as SDAG traces a serial block.
inline void state::trace_resume(const record *rec) {
#if CMK_TRACE_ENABLED
  while (rec->parent) rec = rec->parent;
  _TRACE_BEGIN_EXECUTE_DETAILED(-1, -1, rec->ep, CkMyPe(), 0, ownerHasIdx_ ? &ownerIdx_ : nullptr,
                                owner_);
#else
  (void)rec;
#endif
}

inline want state::future_want(const future &f) const {
  if (f.ep < 0) CkAbort("ck::coro: a when on a future that CK_FUTURE has not given a callback");
  return want(f.ep, f.id);
}

// CK_FUTURE, CK_FUTURE_ALL: a new id for f and the callback that delivers to it.
template <class Owner>
inline CkCallback make_future(state &s, Owner *owner, future &f, int ep, bool all) {
  f.ep = ep;
  f.id = s.new_future_id(all);
  CkCallback cb = all ? all_target(ep, owner) : self_target(ep, owner);
  cb.setRefnum(f.id);
  return cb;
}

// Remove and return the first buffered delivery matching w, keeping the order
// of the rest.
inline bool state::take(const want &w, buffer &out) {
  for (size_t i = 0; i < inbox_.size(); i++)
    if (w.matches(inbox_[i])) {
      out = std::move(inbox_[i]);
      inbox_.erase(inbox_.begin() + i);
      return true;
    }
  return false;
}

inline void state::push(buffer &&b) {
  // A suspended coroutine waiting for this delivery resumes at once, inside
  // the delivering entry method (as an SDAG when body runs); otherwise the
  // delivery is buffered.
  // The delivery joins the inbox first; an alternative of a waiting
  // coroutine is satisfied when every want of it has a buffered match.
  // Fast path, the common case: a waiting record with a one-want arm that
  // matches the delivery takes it directly (no inbox traffic, no
  // allocation).
  bool delivered = visit(recs_, [&](record &rec) {
    if (!rec.waiting) return false;
    const size_t nw = rec.wants.size();
    for (size_t i = 0; i < nw; i++) {
      if (!rec.wants[i].matches(b)) continue;
      const int arm = rec.wants[i].arm;
      if (nw > 1) {
        size_t armSize = 0;
        for (const want &w : rec.wants) if (w.arm == arm) armSize++;
        if (armSize != 1) return false; // an earlier conjunction arm may be satisfied: general path
      }
      rec.ready0 = std::move(b);
      rec.waiting = false;
      rec.wants.clear();
      rec.ready = true;
      rec.readyIndex = arm;
      run(&rec);
      return true;
    }
    return false;
  });
  if (delivered) return;
  // Otherwise it joins the inbox; an arm with several wants (a conjunction)
  // is satisfied when every want has a buffered match.
  inbox_.push_back(std::move(b));
  const buffer &nb = inbox_.back();
  visit(recs_, [&](record &rec) {
    if (!rec.waiting) return false;
    for (size_t i = 0; i < rec.wants.size(); i++) {
      if (!rec.wants[i].matches(nb)) continue;
      const int arm = rec.wants[i].arm;
      std::vector<want> armWants;
      for (const want &w : rec.wants) if (w.arm == arm) armWants.push_back(w);
      std::vector<buffer> bufs;
      if (!try_arm(armWants.data(), armWants.size(), bufs)) continue;
      rec.set_ready(std::move(bufs));
      rec.waiting = false;
      rec.wants.clear();
      rec.ready = true;
      rec.readyIndex = arm;
      run(&rec);
      return true;
    }
    return false;
  });
}

// Take the buffered deliveries matching all of ws (distinct buffers, in
// the order of ws) if every one has a match; else take nothing.
inline bool state::try_arm(const want *ws, size_t n, std::vector<buffer> &out) {
  std::vector<size_t> idx;
  idx.reserve(n);
  for (size_t k = 0; k < n; k++) {
    bool found = false;
    for (size_t i = 0; i < inbox_.size() && !found; i++) {
      if (!ws[k].matches(inbox_[i])) continue;
      bool used = false;
      for (size_t j : idx) if (j == i) { used = true; break; }
      if (!used) { idx.push_back(i); found = true; }
    }
    if (!found) return false;
  }
  out.clear();
  out.reserve(n);
  for (size_t i : idx) out.push_back(std::move(inbox_[i]));
  std::vector<size_t> sorted(idx);
  std::sort(sorted.begin(), sorted.end());
  for (size_t k = sorted.size(); k-- > 0;) inbox_.erase(inbox_.begin() + sorted[k]);
  return true;
}

template <class Msg, std::enable_if_t<is_message_ptr_v<Msg *>, int>>
inline void state::deliver(int ep, Msg *m) {
  buffer b;
  b.ep = ep;
  b.msg = m;
  b.ref = CkGetRefNum(m);
  push(std::move(b));
}

template <class... Args, std::enable_if_t<!(sizeof...(Args) == 1 && (is_message_ptr_v<std::decay_t<Args>> && ...)), int>>
inline void state::deliver(int ep, const Args &...args) {
  buffer b;
  b.ep = ep;
  // The entry's own message, when this is the delivery its wrapper set up
  // (a void entry's system message, a reduction target's reduction
  // message); not when a coroutine resumed by that delivery calls an inbox
  // entry directly, so the delivery is taken out of the scope before it
  // can resume one.
  void *own = CkCurrentEntry() == ep ? CkCurrentMessage() : nullptr;
  if (own) {
    CkpvAccess(_ckCurrentMsg) = nullptr;
    CkpvAccess(_ckCurrentEp) = -1;
  }
  bool hasRef = false;
  if constexpr (sizeof...(Args) > 0) {
    using First = std::tuple_element_t<0, std::tuple<Args...>>;
    if constexpr (std::is_arithmetic_v<First>) {
      hasRef = true;
      b.ref = (CMK_REFNUM_TYPE)std::get<0>(std::tie(args...));
    }
  }
  // Through a reduction wrapper the reference number is the reduction
  // message's (set on the callback), as SDAG does.
  if (own && UsrToEnv(own)->getMsgIdx() == CMessage_CkReductionMsg::__idx) hasRef = false;
  if (!hasRef && own) b.ref = CkGetRefNum(own);
  if constexpr (sizeof...(Args) > 0) {
    // Pack the arguments in the layout the marshalled message would have:
    // the scalars in order, an array as its offset and byte count, then the
    // arrays from a 16-byte aligned start.
    constexpr size_t N = sizeof...(Args);
    int offs[N] = {}, cnts[N] = {};
    int off = 0, k = 0;
    auto place = [&](const auto &a) {
      using A = std::decay_t<decltype(a)>;
      if constexpr (is_array_arg_v<A>) {
        using T = std::remove_cv_t<std::remove_pointer_t<decltype(a.data)>>;
        offs[k] = off = CK_ALIGN(off, (int)sizeof(T));
        cnts[k] = (int)a.bytes;
        off += cnts[k];
      }
      k++;
    };
    (place(args), ...);
    auto pup_one = [&](PUP::er &p, const auto &a, int i) {
      using A = std::decay_t<decltype(a)>;
      if constexpr (is_array_arg_v<A>) { p | offs[i]; p | cnts[i]; }
      else p | const_cast<A &>(a);
    };
    PUP::sizer ps;
    k = 0;
    (pup_one(ps, args, k++), ...);
    int arrstart = CK_ALIGN((int)ps.size(), 16);
    CkMarshallMsg *m = CkAllocateMarshallMsg(arrstart + off);
    PUP::toMem pm(m->msgBuf);
    k = 0;
    (pup_one(pm, args, k++), ...);
    auto copy_one = [&](const auto &a, int i) {
      using A = std::decay_t<decltype(a)>;
      if constexpr (is_array_arg_v<A>)
        if (cnts[i] > 0) std::memcpy(m->msgBuf + arrstart + offs[i], a.data, cnts[i]);
    };
    k = 0;
    (copy_one(args, k++), ...);
    b.msg = m;
  }
  push(std::move(b));
}

template <class P> inline void state::deliver_message(int ep, CkMarshallMsg *m) {
  buffer b;
  b.ep = ep;
  b.msg = m;
  b.ref = CkGetRefNum(m);
  [&]<class... Ps>(params<Ps...>) {
    if constexpr (sizeof...(Ps) > 0) {
      using First = std::tuple_element_t<0, std::tuple<Ps...>>;
      if constexpr (std::is_arithmetic_v<First>) {
        PUP::fromMem pf(m->msgBuf);
        First v;
        pf | v;
        b.ref = (CMK_REFNUM_TYPE)v;
      }
    }
  }(P{});
  push(std::move(b));
}

// Invoke the entry again. A record whose restart message is already on its
// way (sent before a migration; the location manager forwards it) gets none.
inline void state::send_restart(record &rec) {
  if (rec.pending) return;
  rec.pending = true;
  rec.target.send(rec.msg ? restart_copy(&rec.msg) : CkAllocSysMsg());
}

// +ck_fakemigrate: what a migration does to the element's coroutines, without
// moving the element: pack and unpack the state into itself, destroy the
// frames and invoke every active entry again toward its recorded site.
inline void state::rebuild() {
  if (debug_enabled())
    CkPrintf("[%d] ck::coro: rebuilding %zu coroutine(s), %zu buffered deliveries\n", CkMyPe(),
             recs_.size(), inbox_.size());
  PUP::sizer ps;
  pup(ps);
  std::vector<char> buf(ps.size());
  PUP::toMem pt(buf.data());
  pup(pt);
  for (record &rec : recs_) {
    destroy_frames(rec);
    if (rec.msg) CkFreeMsg(rec.msg);
    rec.msg = nullptr;
  }
  recs_.clear();
  inbox_.clear();
  PUP::fromMem pf(buf.data());
  pup(pf); // unpacking sends the restarts
}

inline void state::pup_impl(PUP::er &p, target_fn mk, void *owner) {
  size_t nbuf = inbox_.size(), nrec = recs_.size();
  p | nbuf;
  p | nrec;
  p | nextFuture_;
  p | nextAllFuture_;
  if (p.isUnpacking()) {
    inbox_.resize(nbuf);
    recs_.resize(nrec);
  }
  for (buffer &b : inbox_) b.pup(p);
  if (!p.isUnpacking())
    visit(recs_, [&](record &rec) {
      int ep = rec.parent ? rec.parent->ep : rec.ep;
      if (current_ == &rec)
        CkAbort("ck::coro: a chare is packed from inside its own coroutine entry %d", ep);
      if (rec.root && !(rec.waiting && rec.atSite))
        CkAbort("ck::coro: chare packed while a coroutine of entry %d is suspended at a "
                "co_await that is not a CK_WHEN, CK_OVERLAP or CK_CALL site", ep);
      return false;
    });
  for (record &rec : recs_) {
    rec.owner = this;
    rec.pup(p);
    if (p.isUnpacking()) {
      if (rec.ch.sites.empty())
        CkAbort("ck::coro: unpacked a coroutine of entry %d with no recorded site", rec.ep);
      if (mk) rec.target = mk(rec.ep, owner);
      send_restart(rec);
    }
  }
}

/* ---- when_awaiter, any_awaiter ---- */

inline record *enter_await() {
  record *rec = current_;
  if (rec == nullptr) CkAbort("ck::coro: co_await when outside a coroutine entry method");
  rec->held.clear(); // an array received at the previous wait is no longer valid
  rec->rebuilt = rec->ch.restarting; // a restarted body has reached its recorded site
  rec->ch.restarting = false;
  rec->atSite = rec->ch.noted && !rec->ch.notedCall;
  if (rec->atSite) {
    // A site inside a helper that was awaited with a plain co_await instead
    // of CK_CALL is not on any restart path: the restart would run the
    // caller from the top (review of PR #4231).
    for (unsigned d = 0; d < rec->ch.depth; d++)
      if (!rec->ch.linked[d])
        CkAbort("ck::coro: the wait at line %d is inside a ck::coro::task that was awaited "
                "with co_await instead of CK_CALL (depth %u of %u): a restart could not reach it",
                rec->ch.sites.empty() ? 0 : rec->ch.sites.back(), d, rec->ch.depth);
  }
  rec->ch.noted = false;
  return rec;
}

// +ck_fakemigrate rebuilds at a site once per arrival: a body that was just
// restarted to this site waits here without rebuilding again, so a program
// progresses (and reaches quiescence) as without the flag.
inline void arm_wait(record *rec, std::coroutine_handle<> h) {
  rec->h = h;
  rec->waiting = true;
  if (rec->atSite && !rec->rebuilt && fake_migrate_enabled()) rec->fakeRestart = true;
}

template <class Schema, class... Outs>
inline void when_awaiter<Schema, Outs...>::unpack(buffer &b, record *rec) {
  if constexpr (sizeof...(Outs) == 1 && (is_message_ptr_v<Outs> && ...)) {
    if (b.msg == nullptr || is_marshall_msg(b.msg))
      CkAbort("ck::coro: entry %d delivered marshalled arguments but the when expects a message",
              w_.ep);
    std::get<0>(outs_) = static_cast<std::tuple_element_t<0, std::tuple<Outs...>>>(b.msg);
    b.msg = nullptr; // the coroutine owns the message now
  } else if constexpr (std::is_same_v<Schema, future_msg>) {
    static_assert(sizeof...(Outs) == 0,
                  "a when on a future stores its message into one message pointer, or has no out");
    if (b.msg) CkFreeMsg(b.msg);
    b.msg = nullptr;
  } else if constexpr (std::is_void_v<Schema>) {
    if constexpr (sizeof...(Outs) > 0) {
      static_assert(!(std::is_same_v<Outs, skip_t> || ...),
                    "ck::coro::skip needs a want from the generated inbox struct");
      if (!is_marshall_msg(b.msg))
        CkAbort("ck::coro: entry %d delivered %s but the when expects marshalled arguments", w_.ep,
                b.msg ? "a message" : "no arguments");
      PUP::fromMem pf(static_cast<CkMarshallMsg *>(b.msg)->msgBuf);
      std::apply([&](Outs &...o) { ((pf | o), ...); }, outs_);
    }
  } else {
    unpack_marshalled(Schema{}, std::index_sequence_for<Outs...>{}, b, rec);
  }
}

// Read every parameter in order, as charmxi's unmarshalling does: a scalar
// into its out (or into a temporary for ck::coro::skip), an array's offset
// and count into a slot; then point each array out into the message, or
// copy the elements into a container out. A message with a pointer out is
// kept in the record until the next wait.
template <class Schema, class... Outs>
template <class... Ps, size_t... I>
inline void when_awaiter<Schema, Outs...>::unpack_marshalled(params<Ps...>, std::index_sequence<I...>,
                                                             buffer &b, record *rec) {
  static_assert(sizeof...(Ps) == sizeof...(Outs),
                "a when lists every parameter of the inbox entry, in order (ck::coro::skip for one it does not use)");
  if constexpr (sizeof...(Ps) > 0) {
    if (!is_marshall_msg(b.msg))
      CkAbort("ck::coro: entry %d delivered %s but the when expects marshalled arguments", w_.ep,
              b.msg ? "a message" : "no arguments");
    char *buf = static_cast<CkMarshallMsg *>(b.msg)->msgBuf;
    PUP::fromMem pf(buf);
    int offs[sizeof...(Ps)] = {}, cnts[sizeof...(Ps)] = {};
    auto read = [&](auto tag, auto &o, size_t i) {
      using P = typename decltype(tag)::type;
      using O = std::remove_reference_t<decltype(o)>;
      if constexpr (is_array_param_v<P>) { pf | offs[i]; pf | cnts[i]; }
      else if constexpr (std::is_same_v<O, skip_t>) { P tmp; pf | tmp; }
      else pf | o;
    };
    (read(std::type_identity<Ps>{}, std::get<I>(outs_), I), ...);
    char *arrs = buf + CK_ALIGN(pf.size(), 16);
    bool borrowed = false;
    auto fill = [&](auto tag, auto &o, size_t i) {
      using P = typename decltype(tag)::type;
      using O = std::remove_reference_t<decltype(o)>;
      if constexpr (is_array_param_v<P>) {
        if constexpr (std::is_same_v<O, skip_t>) {
        } else if constexpr (std::is_pointer_v<O>) {
          o = reinterpret_cast<O>(arrs + offs[i]); // borrowed until the next wait
          borrowed = true;
        } else {
          using T = std::remove_pointer_t<out_t<P>>;
          const T *p = reinterpret_cast<const T *>(arrs + offs[i]);
          o.assign(p, p + cnts[i] / sizeof(T));
        }
      }
    };
    (fill(std::type_identity<Ps>{}, std::get<I>(outs_), I), ...);
    if (borrowed) rec->held.push_back(std::move(b));
  }
}

template <class Schema, class... Outs> inline bool when_awaiter<Schema, Outs...>::try_ready(record *rec) {
  buffer taken;
  if (!s_.take(w_, taken)) return false;
  unpack(taken, rec);
  return true;
}

template <class Schema, class... Outs> inline bool when_awaiter<Schema, Outs...>::await_ready() {
  record *rec = enter_await();
  if (try_ready(rec)) return true;
  rec_ = rec;
  return false;
}

template <class Schema, class... Outs>
inline void when_awaiter<Schema, Outs...>::await_suspend(std::coroutine_handle<> h) {
  rec_->wants.clear();
  add_wants(rec_->wants, 0);
  arm_wait(rec_, h);
}

template <class Schema, class... Outs> inline void when_awaiter<Schema, Outs...>::await_resume() {
  if (rec_ == nullptr) return; // was ready
  record *rec = rec_;
  if (!rec->ready) CkAbort("ck::coro: coroutine resumed without a delivery");
  rec->ready = false;
  rec->atSite = false;
  unpack_ready(rec);
  rec->clear_ready();
}

template <class... W> inline bool all_awaiter<W...>::try_ready(record *rec) {
  std::vector<want> ws;
  add_wants(ws, 0);
  std::vector<buffer> bufs;
  if (!s_.try_arm(ws.data(), ws.size(), bufs)) return false;
  std::apply([&](auto &...a) { size_t k = 0; (a.unpack(bufs[k++], rec), ...); }, parts_);
  return true;
}

template <class... W> inline bool all_awaiter<W...>::await_ready() {
  record *rec = enter_await();
  if (try_ready(rec)) return true;
  rec_ = rec;
  return false;
}

template <class... W> inline void all_awaiter<W...>::await_suspend(std::coroutine_handle<> h) {
  rec_->wants.clear();
  add_wants(rec_->wants, 0);
  arm_wait(rec_, h);
}

template <class... W> inline void all_awaiter<W...>::await_resume() {
  if (rec_ == nullptr) return; // was ready
  record *rec = rec_;
  if (!rec->ready) CkAbort("ck::coro: coroutine resumed without a delivery");
  rec->ready = false;
  rec->atSite = false;
  unpack_ready(rec);
  rec->clear_ready();
}

template <class... W> inline bool any_awaiter<W...>::await_ready() {
  record *rec = enter_await();
  int i = 0;
  bool found = std::apply([&](auto &...a) {
    return ((a.try_ready(rec) ? (ready_ = i, true) : (i++, false)) || ...);
  }, alts_);
  if (found) return true;
  rec_ = rec;
  return false;
}

template <class... W> inline void any_awaiter<W...>::await_suspend(std::coroutine_handle<> h) {
  rec_->wants.clear();
  std::apply([&](auto &...a) { int k = 0; (a.add_wants(rec_->wants, k++), ...); }, alts_);
  arm_wait(rec_, h);
}

template <class... W> inline int any_awaiter<W...>::await_resume() {
  if (rec_ == nullptr) return ready_; // was ready
  record *rec = rec_;
  if (!rec->ready) CkAbort("ck::coro: coroutine resumed without a delivery");
  rec->ready = false;
  rec->atSite = false;
  int i = rec->readyIndex;
  std::apply([&](auto &...a) { int k = 0; ((k++ == i ? (a.unpack_ready(rec), 0) : 0), ...); }, alts_);
  rec->clear_ready();
  return i;
}

inline overlap_awaiter state::overlap(std::vector<task<void>> &&tasks) noexcept {
  return overlap_awaiter(*this, std::move(tasks));
}

inline bool overlap_awaiter::await_ready() {
  record *rec = enter_await();
  rec_ = rec;
  size_t n = tasks_.size();
  if (rec->kids.empty()) {
    for (size_t k = 0; k < n; k++) {
      rec->kids.emplace_back();
      rec->kids.back().owner = rec->owner;
      rec->kids.back().parent = rec;
    }
  } else if (rec->kids.size() != n) {
    CkAbort("ck::coro: an overlap restarted with %zu tasks where %zu were recorded", n,
            rec->kids.size());
  }
  // Count the unfinished children first: a child that finishes while being
  // started decrements the count in finish_run, and the parent is not
  // waiting yet, so no resume happens before this returns.
  rec->kidsRunning = 0;
  for (record &kid : rec->kids) if (!kid.done) rec->kidsRunning++;
  size_t k = 0;
  for (record &kid : rec->kids) {
    task<void> &t = tasks_[k++];
    if (kid.done || kid.root) continue; // finished before a restart: not run again
    kid.root = kid.h = t.release();
    kid.ch.restarting = !kid.ch.sites.empty();
    s_.run(&kid);                       // to its first site, or to its end
  }
  return rec->kidsRunning == 0;
}

inline void overlap_awaiter::await_suspend(std::coroutine_handle<> h) {
  rec_->wants.clear();
  arm_wait(rec_, h);
}

inline void overlap_awaiter::await_resume() {
  record *rec = rec_;
  rec->atSite = false;
  rec->waiting = false;
  rec->kids.clear(); // all done; a later overlap at another site starts afresh
}

} // namespace coro

// The return type of a coroutine entry method.
using coroutine = coro::coroutine;

} // namespace ck

/* ---------------- macros ---------------- */

// The block of a body that can be rebuilt after a migration: CK_MIGRATABLE
// { ... } opens the restart switch, whose body is the block that follows
// (the else branch of an if (false)); each site in the block is a case of
// it. Site 0 enters the block from the top; a restart jumps to the
// recorded site; a recorded site that is not a case of this function lands
// on the default label and aborts before any statement of the block runs
// (the shape is the CAMPI session's, 2026-11-12: with default: next to
// case 0 the code before the first site ran a second time before the
// abort). abort_no_site is [[noreturn]], so the end of a value-returning
// task stays unreachable for the compiler and co_return inside the block
// is enough. The same block serves a coroutine entry and a
// ck::coro::task<T> helper.
#define CK_MIGRATABLE                                                         \
  ::ck::coro::frame_guard ck_frame_;                                          \
  switch (ck_frame_.site())                                                   \
  case 0:                                                                     \
    if (false)                                                                \
    default:                                                                  \
      ::ck::coro::abort_no_site(ck_frame_);                                   \
    else

// The hand-written form (no X_CORO_CODE): the first statement of a coroutine
// entry's body names the chare's ck::coro::state member and the entry's
// index (CkIndex_X::name(...)); the _MSG form names the entry's message
// parameter, which then belongs to the state (as with [coroutine]: not
// deleted by the body). With X_CORO_CODE the promise registers the
// coroutine and the body starts with CK_MIGRATABLE.
#define CK_COROUTINE_STATE_MSG(co_, ep_, msg_)                                \
  ::ck::coro::root_guard ck_root_((co_), this, (ep_), (msg_))
#define CK_COROUTINE_STATE(co_, ep_) CK_COROUTINE_STATE_MSG(co_, ep_, nullptr)

// A when site: CK_WHEN(ep, refnum-or-ck::coro::noref, out...) stores the
// delivered message pointer or arguments into out.
#define CK_WHEN(...)                                                          \
  case __LINE__:                                                              \
    do {                                                                      \
      ::ck::coro::note_site(ck_frame_, __LINE__);                             \
      co_await ::ck::coro::current_state().when(__VA_ARGS__);                 \
    } while (0)

// The block form of a when: CK_WHEN_BLOCK(want, name...) { ... } declares the
// entry's parameters, in order, as variables of the block that follows (an
// array as a pointer into the message, valid in the block). The block
// cannot contain a site: the restart jump would bypass the declarations,
// which the compiler rejects. The want must come from the generated inbox
// struct, or be a future (its message as CkMessage *). Being an if
// statement, it cannot be followed by an else.
#define CK_WHEN_BLOCK(want_, ...)                                             \
  case __LINE__:                                                              \
    if (auto [__VA_ARGS__] = (::ck::coro::note_site(ck_frame_, __LINE__),     \
                              co_await ::ck::coro::current_state().when_block(want_)); \
        true)

// A when-all site (SDAG when a(), b()): CK_WHEN_ALL(CK_ALT(...), CK_ALT(...))
// waits until every alternative has a delivery and takes them together.
// A when-any site (SDAG case): CK_WHEN_ANY(idx, CK_ALT(...), CK_ALL(CK_ALT(...), ...), ...)
// stores the index of the alternative (a when or a when-all) that matched in idx.
#define CK_ALT(...) ::ck::coro::current_state().when(__VA_ARGS__)

// CK_FUTURE(f): a new id for the ck::coro::future member f and a callback to
// this element that delivers to it, to pass to any API taking a CkCallback;
// CK_FUTURE_ALL(f): the same with a callback to every element of the
// collection, for contribute and other results every element awaits.
// CK_WHEN(f, msg), CK_WHEN(f), CK_ALT(f, ...) and CK_WHEN_BLOCK(f, m) wait
// for a delivery. Needs X_CORO_CODE (the chare's _coro_callback entry).
#define CK_FUTURE(f_) \
  ::ck::coro::make_future(__coro, this, (f_), inbox::_coro_callback().ep, false)
#define CK_FUTURE_ALL(f_) \
  ::ck::coro::make_future(__coro, this, (f_), inbox::_coro_callback().ep, true)
#define CK_ALL(...) ::ck::coro::current_state().all(__VA_ARGS__)
#define CK_WHEN_ALL(...)                                                      \
  case __LINE__:                                                              \
    do {                                                                      \
      ::ck::coro::note_site(ck_frame_, __LINE__);                             \
      co_await ::ck::coro::current_state().all(__VA_ARGS__);                  \
    } while (0)
#define CK_WHEN_ANY(idx_, ...)                                                \
  case __LINE__:                                                              \
    do {                                                                      \
      ::ck::coro::note_site(ck_frame_, __LINE__);                             \
      (idx_) = co_await ::ck::coro::current_state().any(__VA_ARGS__);         \
    } while (0)

// An overlap site (SDAG overlap): runs the ck::coro::task<void> helpers as
// child coroutines of this one, each with its own sites, and waits for all
// of them. CK_FORALL(i, lo, hi, expr) makes one task per i in [lo, hi).
#define CK_OVERLAP(...)                                                       \
  case __LINE__:                                                              \
    do {                                                                      \
      ::ck::coro::note_site(ck_frame_, __LINE__);                             \
      co_await ::ck::coro::current_state().overlap(::ck::coro::tasks(__VA_ARGS__)); \
    } while (0)
#define CK_FORALL(var_, lo_, hi_, expr_)                                      \
  case __LINE__:                                                              \
    do {                                                                      \
      ::ck::coro::note_site(ck_frame_, __LINE__);                             \
      co_await ::ck::coro::current_state().overlap([&]() {                    \
        std::vector<::ck::coro::task<void>> ck_tasks_;                        \
        for (int var_ = (lo_); var_ < (hi_); var_++) ck_tasks_.push_back(expr_); \
        return ck_tasks_;                                                     \
      }());                                                                   \
    } while (0)
// SDAG's forall [i] (start:end, stride): end included, stride of either sign.
#define CK_FORALL_STRIDE(var_, start_, end_, stride_, expr_)                  \
  case __LINE__:                                                              \
    do {                                                                      \
      ::ck::coro::note_site(ck_frame_, __LINE__);                             \
      co_await ::ck::coro::current_state().overlap([&]() {                    \
        std::vector<::ck::coro::task<void>> ck_tasks_;                        \
        const int ck_end_ = (end_), ck_stride_ = (stride_);                   \
        for (int var_ = (start_); ck_stride_ > 0 ? var_ <= ck_end_ : var_ >= ck_end_; \
             var_ += ck_stride_)                                              \
          ck_tasks_.push_back(expr_);                                         \
        return ck_tasks_;                                                     \
      }());                                                                   \
    } while (0)

// A call site: awaits a resumable helper that contains a site.
#define CK_CALL(expr_)                                                        \
  case __LINE__:                                                              \
    do {                                                                      \
      ::ck::coro::note_site(ck_frame_, __LINE__, true);                       \
      co_await (expr_);                                                       \
    } while (0)
#define CK_CALL_TO(lhs_, expr_)                                               \
  case __LINE__:                                                              \
    do {                                                                      \
      ::ck::coro::note_site(ck_frame_, __LINE__, true);                       \
      (lhs_) = co_await (expr_);                                              \
    } while (0)

#endif // CKCORO_H
