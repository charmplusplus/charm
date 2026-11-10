/*
 * ckcoro: Charm++ entry methods as C++20 coroutines. Requires C++20.
 *
 * A coroutine entry method is declared [coroutine] in the .ci file (void,
 * with no parameters or one message) and defined in C++ with the return
 * type ck::coroutine. Its body begins with CK_COROUTINE() and ends with
 * CK_END(), and suspends at CK_WHEN sites until a delivery to an [inbox]
 * entry matches. The chare puts the generated X_CORO_CODE macro in its
 * class: it defines the ck::coro::state member (the active coroutines and
 * the inbox, packed with the element), the bodies of the inbox entries, and
 * inbox::name() / inbox::name(ref), which name the entry at a site.
 *
 *   // .ci: entry [coroutine] void doStep();
 *   //      entry [inbox] void recvGhost(GhostMsg *m);
 *   //      entry [inbox] void recvCount(int iter, int count);
 *   class Jacobi : public CBase_Jacobi {
 *     Jacobi_CORO_CODE
 *     GhostMsg *gmsg;                // scratch; assigned at a site, used after it
 *     int imsg, iter, i, count;      // state across sites: members, pupped
 *     ck::coroutine doStep() {
 *       CK_COROUTINE();
 *       sendGhosts();
 *       for (imsg = 0; imsg < 6; imsg++) {
 *         CK_WHEN(inbox::recvGhost(iter), gmsg);   // when recvGhost[iter](GhostMsg *gmsg)
 *         process(gmsg);
 *       }
 *       CK_WHEN(inbox::recvCount(iter), i, count); // when recvCount[iter](int i, int count)
 *       compute();
 *       CK_END();
 *     }
 *   };
 *
 * A when stores the delivery into the lvalues that follow the want: the
 * message pointer (the coroutine owns it), or every marshalled parameter of
 * the entry in order. inbox::name(ref) matches the reference number:
 * the envelope's for a message entry, the first arithmetic parameter for a
 * marshalled entry; inbox::name() matches any. Deliveries that no coroutine
 * is waiting for are buffered as messages (the marshalled parameters stay
 * in the message being delivered, see CkCurrentMessage) and move with the
 * element. CK_WHEN_ANY(idx, CK_ALT(...), ...) waits for the first of
 * several (an SDAG case).
 *
 * Without the attributes, the same works by hand: a ck::coro::state member
 * pupped by the chare, inbox entries whose bodies call co.deliver(ep, msg)
 * or co.deliver(ep, args...), CK_COROUTINE(co, ep) and CK_WHEN(ep, ref, ...)
 * with ep = CkIndex_X::name(...).
 *
 * Migration and checkpoint. A coroutine frame is never packed. The state
 * object packs, for each active coroutine, the entry index, a callback to
 * the element and the chain of restart sites, and on unpacking sends the
 * entry's message to the element again: the body runs again from the top
 * and the switch opened by CK_COROUTINE jumps to the recorded site, where
 * CK_WHEN finds the buffered message or waits for it. Rules: (1) a value
 * that is live across a site is a member of the chare (C++ rejects the jump
 * past a local with an initializer; a scalar declared without one is not
 * caught, and +ck_fakemigrate is the check); (2) code before a site is
 * skipped on a restart, so its results must be members; (3) an element may
 * migrate or be checkpointed only while its coroutines are suspended at
 * sites (CK_WHEN, CK_WHEN_ANY, CK_CALL); a plain co_await is allowed between
 * sites, but packing an element suspended there aborts. A site is one
 * statement and may be the body of an if or a loop; its case label must not
 * be inside a switch of the user's own; two sites must not share a line. A
 * function awaited from a coroutine that contains a site returns
 * ck::coro::task<T>, has its own CK_RESUMABLE/CK_END and is awaited with
 * CK_CALL or CK_CALL_TO.
 *
 * At most one coroutine per entry method is active in an element at a time.
 * Inbox entries take a message or marshalled parameters without arrays.
 *
 * +ck_fakemigrate on the command line makes every site rebuild the element's
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
// The frame handle of a coroutine entry just created, picked up by CK_COROUTINE.
inline thread_local std::coroutine_handle<> pending_root_ = {};

/* ---------------- coroutine: the entry method type ---------------- */

// The return type of a coroutine entry method. The body starts at once (the
// generated entry wrapper discards the returned object); the object's
// destructor runs when the body first suspends or finishes and hands control
// to the state object.
class coroutine {
 public:
  struct promise_type {
    coroutine get_return_object() noexcept {
      auto h = std::coroutine_handle<promise_type>::from_promise(*this);
      pending_root_ = h;
      return coroutine(h);
    }
    std::suspend_never initial_suspend() const noexcept { return {}; }
    std::suspend_always final_suspend() const noexcept { return {}; }
    void return_void() const noexcept {}
    void unhandled_exception() const noexcept { abort_unhandled_exception(); }
  };
  explicit coroutine(std::coroutine_handle<> h) noexcept : h_(h) {}
  coroutine(const coroutine &) = delete;
  coroutine &operator=(const coroutine &) = delete;
  ~coroutine();

 private:
  std::coroutine_handle<> h_;
};

/* ---------------- the inbox ---------------- */

template <class T>
inline constexpr bool is_message_ptr_v =
    std::is_pointer_v<T> && std::is_base_of_v<CkMessage, std::remove_pointer_t<T>>;

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
  explicit want(int ep_) noexcept : ep(ep_), hasRef(false), ref(0) {}
  want(int ep_, CMK_REFNUM_TYPE ref_) noexcept : ep(ep_), hasRef(true), ref(ref_) {}
  bool matches(const buffer &b) const noexcept {
    return b.ep == ep && (!hasRef || b.ref == ref);
  }
};

struct noref_t {};
inline constexpr noref_t noref{};

/* ---------------- records of active coroutines ---------------- */

struct record {
  state *owner = nullptr;
  int ep = -1;                   // the coroutine entry; pupped
  CkCallback target;             // the element; pupped
  void *msg = nullptr;           // the entry's message, for a restart; pupped (null: a void entry)
  chain ch;                      // restart sites; its sites pupped
  std::coroutine_handle<> root;  // the entry's frame
  std::coroutine_handle<> h;     // the innermost suspended frame
  record *prev = nullptr;        // current_ before this record ran
  chain *prevChain = nullptr;    // current_chain_ before this record ran
  bool waiting = false;          // suspended in a when awaitable
  bool atSite = false;           // ... reached through CK_WHEN
  std::vector<want> wants;       // what it waits for (one entry; several for when_any)
  bool ready = false;            // delivered: which and the buffer
  int readyIndex = -1;
  buffer readyBuf;
  bool fakeRestart = false;      // rebuild once it has suspended (+ck_fakemigrate)
  bool rebuilt = false;          // restarted and not yet past the recorded site: no rebuild there
  bool pending = false;          // a restart message is on its way to the element; pupped

  void pup(PUP::er &p) {
    p | ep;
    p | target;
    p | pending;
    bool hasMsg = msg != nullptr;
    p | hasMsg;
    if (hasMsg) CkPupMessage(p, &msg);
    ch.pup(p);
  }
};

class state;
inline state &current_state();

/* ---------------- when awaitable ---------------- */

// The record of the running coroutine, entering a when or when_any: a
// restarted body has reached its site; remember whether the await was
// reached through a site macro.
inline record *enter_await();

template <class... Outs> class when_awaiter {
 public:
  when_awaiter(state &s, want w, Outs &...outs) noexcept : s_(s), w_(w), outs_(outs...) {}
  const want &spec() const noexcept { return w_; }
  void unpack(buffer &b);
  bool await_ready();
  void await_suspend(std::coroutine_handle<> h);
  void await_resume();

 private:
  state &s_;
  want w_;
  std::tuple<Outs &...> outs_;
  record *rec_ = nullptr;
};

// Await the first of several deliveries (SDAG case): the awaited value is
// the index of the alternative that matched, whose outputs are stored.
template <class... W> class any_awaiter {
 public:
  explicit any_awaiter(state &s, W &&...w) noexcept : s_(s), alts_(std::move(w)...) {}
  bool await_ready();
  void await_suspend(std::coroutine_handle<> h);
  int await_resume();

 private:
  void unpack_at(int i, buffer &b) {
    std::apply([&](auto &...a) { int k = 0; ((k++ == i ? (a.unpack(b), 0) : 0), ...); }, alts_);
  }
  state &s_;
  std::tuple<W...> alts_;
  record *rec_ = nullptr;
  int ready_ = -1;
};

/* ---------------- state: per-element coroutines and inbox ---------------- */

class state {
 public:
  state() = default;
  state(const state &) = delete;
  state &operator=(const state &) = delete;
  ~state();

  // Inbox entry bodies (generated by [inbox], or written by hand): hand the
  // delivery to the state. A message entry hands over its message. A
  // marshalled or void entry passes its arguments: the reference number is
  // the first arithmetic argument, else the envelope's; the arguments
  // themselves stay in the message being delivered (CkCurrentMessage), which
  // the state keeps, unless the entry was reached some other way (a reduction
  // target, a direct call), in which case they are packed into a new message.
  template <class Msg, std::enable_if_t<is_message_ptr_v<Msg *>, int> = 0>
  void deliver(int ep, Msg *m);
  template <class... Args, std::enable_if_t<!(sizeof...(Args) == 1 && (is_message_ptr_v<std::decay_t<Args>> && ...)), int> = 0>
  void deliver(int ep, const Args &...args);

  // Await a delivery to entry ep with reference number ref (or any, with
  // noref), storing the message or arguments in outs. CK_WHEN wraps this.
  template <class... Outs>
  when_awaiter<Outs...> when(int ep, CMK_REFNUM_TYPE ref, Outs &...outs) noexcept {
    return when_awaiter<Outs...>(*this, want(ep, ref), outs...);
  }
  template <class... Outs>
  when_awaiter<Outs...> when(int ep, noref_t, Outs &...outs) noexcept {
    return when_awaiter<Outs...>(*this, want(ep), outs...);
  }
  template <class... Outs>
  when_awaiter<Outs...> when(want w, Outs &...outs) noexcept {
    return when_awaiter<Outs...>(*this, w, outs...);
  }
  // The first of several whens to match: co_await co.any(co.when(...), co.when(...)).
  template <class... W> any_awaiter<W...> any(W &&...w) noexcept {
    return any_awaiter<W...>(*this, std::move(w)...);
  }

  void pup(PUP::er &p);

  // Internal.
  record &begin(ArrayElement *elt, int ep, void *msg);
  record &begin(ArrayElement *elt, bool isEntry);
  void finish_run(record *rec);
  bool take(const want &w, buffer &out);
  void rebuild();
  size_t num_buffered() const noexcept { return inbox_.size(); }
  size_t num_active() const noexcept { return recs_.size(); }

 private:
  void destroy_frames(record &rec);
  void send_restart(record &rec);
  void push(buffer &&b);

  std::list<record> recs_;
  std::vector<buffer> inbox_;
};

// The frame guard of a coroutine entry: creates or adopts the record.
class root_guard {
 public:
  root_guard(state &s, ArrayElement *elt, int ep, void *msg);
  root_guard(state &s, ArrayElement *elt, bool isEntry);
  ~root_guard() = default;
  root_guard(const root_guard &) = delete;
  root_guard &operator=(const root_guard &) = delete;
};

/* ======================= inline implementation ======================= */

inline bool fake_migrate_enabled() noexcept {
  static thread_local int v = -1;
  if (v < 0) {
    v = 0;
    char **argv = CkGetArgv();
    for (int i = 0; argv && argv[i]; i++)
      if (std::strcmp(argv[i], "+ck_fakemigrate") == 0) v = 1;
    if (v && CkMyPe() == 0)
      CkPrintf("ck::coro> +ck_fakemigrate: every when site rebuilds the element's coroutines in place\n");
  }
  return v == 1;
}

inline state &current_state() {
  if (current_ == nullptr)
    CkAbort("ck::coro: CK_WHEN used outside a coroutine entry method");
  return *current_->owner;
}

inline coroutine::~coroutine() {
  record *rec = current_;
  if (rec == nullptr || rec->root != h_) {
    // The body never reached CK_COROUTINE.
    if (h_.done()) { h_.destroy(); return; }
    CkAbort("ck::coro: a coroutine entry method suspended without CK_COROUTINE at its start");
  }
  rec->owner->finish_run(rec);
}

inline root_guard::root_guard(state &s, ArrayElement *elt, int ep, void *msg) {
  s.begin(elt, ep, msg);
}
inline root_guard::root_guard(state &s, ArrayElement *elt, bool isEntry) {
  s.begin(elt, isEntry);
}

inline state::~state() {
  for (record &rec : recs_) destroy_frames(rec);
  // ~buffer frees buffered messages; ~record frees nothing: msg is freed here
  for (record &rec : recs_) if (rec.msg) CkFreeMsg(rec.msg);
}

inline void state::destroy_frames(record &rec) {
  if (rec.root) {
    rec.ch.destroying = true;
    rec.root.destroy();
    rec.ch.destroying = false;
  }
  rec.root = {};
  rec.h = {};
  rec.ch.depth = 0;
  rec.ch.linked.clear();
  rec.ch.noted = false;
  rec.waiting = false;
  rec.atSite = false;
  rec.wants.clear();
  rec.ready = false;
  rec.readyBuf.release();
}

inline record &state::begin(ArrayElement *elt, int ep, void *msg) {
  std::coroutine_handle<> root = std::exchange(pending_root_, {});
  if (!root) CkAbort("ck::coro: CK_COROUTINE in a function that does not return ck::coroutine");
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
    rec->target = CkCallback(ep, elt->ckGetArrayIndex(), elt->ckGetArrayID());
    if (msg) { void *m = msg; rec->msg = CkCopyMsg(&m); }
  } else {
    rec->ch.restarting = !rec->ch.sites.empty();
    rec->pending = false;
  }
  rec->root = root;
  rec->h = root;
  rec->prev = current_;
  rec->prevChain = current_chain_;
  current_ = rec;
  current_chain_ = &rec->ch;
  return *rec;
}

// CK_COROUTINE() with no arguments: the entry and its message are the ones
// being delivered (the generated __coro_is_entry says whether the running
// entry is one of the chare's [coroutine] entries). A void entry's system
// message is not kept.
inline record &state::begin(ArrayElement *elt, bool isEntry) {
  if (!isEntry)
    CkAbort("ck::coro: CK_COROUTINE() in a function that is not running as a [coroutine] entry "
            "method (entry %d is being delivered)", CkCurrentEntry());
  void *m = CkCurrentMessage();
  if (m && (is_marshall_msg(m) || UsrToEnv(m)->getMsgIdx() == 0)) m = nullptr;
  return begin(elt, CkCurrentEntry(), m);
}

// After the body ran (first run or a resume) and suspended or finished.
inline void state::finish_run(record *rec) {
  current_ = rec->prev;
  current_chain_ = rec->prevChain;
  rec->prev = nullptr;
  rec->prevChain = nullptr;
  if (rec->root.done()) {
    destroy_frames(*rec);
    if (rec->msg) CkFreeMsg(rec->msg);
    rec->msg = nullptr;
    for (auto it = recs_.begin(); it != recs_.end(); ++it)
      if (&*it == rec) { recs_.erase(it); break; }
    return;
  }
  if (!rec->h) CkAbort("ck::coro: coroutine suspended at an awaitable that is not ck::coro's");
  if (rec->fakeRestart) {
    rec->fakeRestart = false;
    rebuild();
  }
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
  for (record &rec : recs_) {
    if (!rec.waiting) continue;
    for (size_t i = 0; i < rec.wants.size(); i++) {
      if (!rec.wants[i].matches(b)) continue;
      rec.waiting = false;
      rec.wants.clear();
      rec.ready = true;
      rec.readyIndex = (int)i;
      rec.readyBuf = std::move(b);
      rec.prev = current_;
      rec.prevChain = current_chain_;
      current_ = &rec;
      current_chain_ = &rec.ch;
      std::coroutine_handle<> h = std::exchange(rec.h, {});
      h.resume();
      finish_run(&rec);
      return;
    }
  }
  inbox_.push_back(std::move(b));
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
  void *cur = CkCurrentMessage();
  bool hasRef = false;
  if constexpr (sizeof...(Args) > 0) {
    using First = std::tuple_element_t<0, std::tuple<Args...>>;
    if constexpr (std::is_arithmetic_v<First>) {
      hasRef = true;
      b.ref = (CMK_REFNUM_TYPE)std::get<0>(std::tie(args...));
    }
  }
  if (!hasRef && cur) b.ref = CkGetRefNum(cur);
  if (CkCurrentEntry() == ep && is_marshall_msg(cur)) {
    // The entry's own marshalled parameters: keep the message.
    CkReferenceMsg(cur);
    b.msg = cur;
  } else if constexpr (sizeof...(Args) > 0) {
    // Reached otherwise (a reduction target, a direct call): pack the
    // arguments in the layout the marshalled message would have.
    PUP::sizer ps;
    ((ps | const_cast<Args &>(args)), ...);
    CkMarshallMsg *m = CkAllocateMarshallMsg(ps.size());
    PUP::toMem pm(m->msgBuf);
    ((pm | const_cast<Args &>(args)), ...);
    b.msg = m;
  }
  push(std::move(b));
}

// Invoke the entry again. A record whose restart message is already on its
// way (sent before a migration; the location manager forwards it) gets none.
inline void state::send_restart(record &rec) {
  if (rec.pending) return;
  rec.pending = true;
  void *m;
  if (rec.msg) { void *c = rec.msg; m = CkCopyMsg(&c); }
  else m = CkAllocSysMsg();
  rec.target.send(m);
}

// +ck_fakemigrate: what a migration does to the element's coroutines, without
// moving the element: pack and unpack the state into itself, destroy the
// frames and invoke every active entry again toward its recorded site.
inline void state::rebuild() {
  static thread_local int dbg = -1;
  if (dbg < 0) dbg = getenv("CKCORO_DEBUG") != nullptr;
  if (dbg)
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

inline void state::pup(PUP::er &p) {
  size_t nbuf = inbox_.size(), nrec = recs_.size();
  p | nbuf;
  p | nrec;
  if (p.isUnpacking()) {
    inbox_.resize(nbuf);
    recs_.resize(nrec);
  }
  for (buffer &b : inbox_) b.pup(p);
  for (record &rec : recs_) {
    if (!p.isUnpacking()) {
      if (current_ == &rec)
        CkAbort("ck::coro: an element is packed from inside its own coroutine entry %d", rec.ep);
      if (rec.root && !(rec.waiting && rec.atSite))
        CkAbort("ck::coro: element packed while a coroutine of entry %d is suspended at a "
                "co_await that is not a CK_WHEN or CK_CALL site", rec.ep);
    }
    rec.pup(p);
    if (p.isUnpacking()) {
      rec.owner = this;
      if (rec.ch.sites.empty())
        CkAbort("ck::coro: unpacked a coroutine of entry %d with no recorded site", rec.ep);
      send_restart(rec);
    }
  }
}

/* ---- when_awaiter, any_awaiter ---- */

inline record *enter_await() {
  record *rec = current_;
  if (rec == nullptr) CkAbort("ck::coro: co_await when outside a coroutine entry method");
  rec->rebuilt = rec->ch.restarting; // a restarted body has reached its recorded site
  rec->ch.restarting = false;
  rec->atSite = rec->ch.noted && !rec->ch.notedCall;
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

template <class... Outs> inline void when_awaiter<Outs...>::unpack(buffer &b) {
  if constexpr (sizeof...(Outs) == 1 && (is_message_ptr_v<Outs> && ...)) {
    if (b.msg == nullptr || is_marshall_msg(b.msg))
      CkAbort("ck::coro: entry %d delivered marshalled arguments but the when expects a message",
              w_.ep);
    std::get<0>(outs_) = static_cast<std::tuple_element_t<0, std::tuple<Outs...>>>(b.msg);
    b.msg = nullptr; // the coroutine owns the message now
  } else if constexpr (sizeof...(Outs) > 0) {
    if (!is_marshall_msg(b.msg))
      CkAbort("ck::coro: entry %d delivered %s but the when expects marshalled arguments", w_.ep,
              b.msg ? "a message" : "no arguments");
    PUP::fromMem pf(static_cast<CkMarshallMsg *>(b.msg)->msgBuf);
    std::apply([&](Outs &...o) { ((pf | o), ...); }, outs_);
  }
}

template <class... Outs> inline bool when_awaiter<Outs...>::await_ready() {
  record *rec = enter_await();
  buffer taken;
  if (!s_.take(w_, taken)) {
    rec_ = rec;
    return false;
  }
  unpack(taken);
  return true;
}

template <class... Outs> inline void when_awaiter<Outs...>::await_suspend(std::coroutine_handle<> h) {
  rec_->wants.assign(1, w_);
  arm_wait(rec_, h);
}

template <class... Outs> inline void when_awaiter<Outs...>::await_resume() {
  if (rec_ == nullptr) return; // was ready
  record *rec = rec_;
  if (!rec->ready) CkAbort("ck::coro: coroutine resumed without a delivery");
  rec->ready = false;
  rec->atSite = false;
  unpack(rec->readyBuf);
  rec->readyBuf.release();
}

template <class... W> inline bool any_awaiter<W...>::await_ready() {
  record *rec = enter_await();
  buffer taken;
  int i = 0;
  bool found = std::apply([&](auto &...a) {
    return ((s_.take(a.spec(), taken) ? (a.unpack(taken), ready_ = i, true) : (i++, false)) || ...);
  }, alts_);
  if (found) return true;
  rec_ = rec;
  return false;
}

template <class... W> inline void any_awaiter<W...>::await_suspend(std::coroutine_handle<> h) {
  rec_->wants.clear();
  std::apply([&](auto &...a) { (rec_->wants.push_back(a.spec()), ...); }, alts_);
  arm_wait(rec_, h);
}

template <class... W> inline int any_awaiter<W...>::await_resume() {
  if (rec_ == nullptr) return ready_; // was ready
  record *rec = rec_;
  if (!rec->ready) CkAbort("ck::coro: coroutine resumed without a delivery");
  rec->ready = false;
  rec->atSite = false;
  int i = rec->readyIndex;
  unpack_at(i, rec->readyBuf);
  rec->readyBuf.release();
  return i;
}

} // namespace coro

// The return type of a coroutine entry method.
using coroutine = coro::coroutine;

} // namespace ck

/* ---------------- macros ---------------- */

// The start of a coroutine entry method: co_ is the element's ck::coro::state
// member, ep_ the entry's index (CkIndex_X::name(...)); the _MSG form passes
// the entry's message, which is copied for restarts.
#define CK_COROUTINE_MSG(co_, ep_, msg_)                                      \
  ::ck::coro::root_guard ck_root_((co_), this, (ep_), (msg_));                \
  ::ck::coro::frame_guard ck_frame_;                                          \
  switch (ck_frame_.site()) {                                                 \
  case 0:
#define CK_COROUTINE_2_(co_, ep_) CK_COROUTINE_MSG(co_, ep_, nullptr)
// With the [coroutine] attribute and X_CORO_CODE in the class, CK_COROUTINE()
// takes no arguments: the state is the generated member and the entry is
// the one being delivered.
#define CK_COROUTINE_0_()                                                     \
  ::ck::coro::root_guard ck_root_(__coro, this, __coro_is_entry(::CkCurrentEntry())); \
  ::ck::coro::frame_guard ck_frame_;                                          \
  switch (ck_frame_.site()) {                                                 \
  case 0:
#define CK_CORO_SELECT_(a_, b_, name_, ...) name_
#define CK_COROUTINE(...)                                                     \
  CK_CORO_SELECT_(__VA_ARGS__ __VA_OPT__(, ) CK_COROUTINE_2_, CK_COROUTINE_0_, CK_COROUTINE_0_)(__VA_ARGS__)

// The start of the restart switch of an awaited helper (ck::coro::task<T>).
#define CK_RESUMABLE()                                                        \
  ::ck::coro::frame_guard ck_frame_;                                          \
  switch (ck_frame_.site()) {                                                 \
  case 0:

// The end of a coroutine entry method or resumable helper.
#define CK_END()                                                              \
  break;                                                                      \
  default:                                                                    \
    ::ck::coro::abort_no_site(ck_frame_);                                     \
  }

// A when site: CK_WHEN(ep, refnum-or-ck::coro::noref, out...) stores the
// delivered message pointer or arguments into out.
#define CK_WHEN(...)                                                          \
  case __LINE__:                                                              \
    do {                                                                      \
      ::ck::coro::note_site(ck_frame_, __LINE__);                             \
      co_await ::ck::coro::current_state().when(__VA_ARGS__);                 \
    } while (0)

// A when-any site (SDAG case): CK_WHEN_ANY(idx, CK_ALT(ep, ref, out...), ...)
// stores the index of the alternative that matched in idx.
#define CK_ALT(...) ::ck::coro::current_state().when(__VA_ARGS__)
#define CK_WHEN_ANY(idx_, ...)                                                \
  case __LINE__:                                                              \
    do {                                                                      \
      ::ck::coro::note_site(ck_frame_, __LINE__);                             \
      (idx_) = co_await ::ck::coro::current_state().any(__VA_ARGS__);         \
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
