/*
 * ckcoro: Charm++ entry methods as C++20 coroutines. Requires C++20.
 *
 * A coroutine entry method is declared in the .ci file as an ordinary void
 * entry method (no parameters, or one message) and defined in C++ with the
 * return type ck::coroutine. Its body begins with CK_COROUTINE and ends with
 * CK_END, and suspends at CK_WHEN sites until a message for an inbox entry
 * arrives. An inbox entry is an ordinary entry method whose body hands its
 * arguments to the element's ck::coro::state object:
 *
 *   // .ci: entry void doStep(); entry void recvGhost(GhostMsg *m);
 *   class Jacobi : public CBase_Jacobi {
 *     ck::coro::state co;            // active coroutines and the inbox
 *     GhostMsg *gmsg;                // scratch; assigned at a site, used after it
 *     int imsg, iter;                // state: members, pupped
 *     void pup(PUP::er &p) { ...; p | co; }
 *     void recvGhost(GhostMsg *m) { co.deliver(CkIndex_Jacobi::recvGhost(0), m); }
 *     ck::coroutine doStep() {
 *       CK_COROUTINE(co, CkIndex_Jacobi::doStep());
 *       sendGhosts();
 *       for (imsg = 0; imsg < 6; imsg++) {
 *         CK_WHEN(CkIndex_Jacobi::recvGhost(0), iter, gmsg);   // like when recvGhost[iter](GhostMsg *gmsg)
 *         process(gmsg);
 *       }
 *       compute();
 *       CK_END();
 *     }
 *   };
 *
 * Messages for an inbox entry that no coroutine is waiting for are buffered
 * in the state object, matched later by entry and reference number (the
 * envelope's for a message, the first arithmetic argument for marshalled
 * parameters), and move with the element.
 *
 * Migration and checkpoint. A coroutine frame is never packed. The state
 * object packs, for each active coroutine, the entry index, a callback to the
 * element and the chain of restart sites, and on unpacking sends the entry's
 * message to the element again: the body runs again from the top and the
 * switch opened by CK_COROUTINE jumps to the recorded site, where CK_WHEN
 * finds the buffered message or waits for it. Rules: (1) a value that is
 * live across a site is a member of the chare (C++ rejects the jump past a
 * local with an initializer; a scalar declared without one is not caught,
 * and +ck_fakemigrate is the check); (2) code before a site is skipped on
 * a restart, so its results must be members; (3) an element may migrate or
 * be checkpointed only while its coroutines are suspended at sites (CK_WHEN,
 * CK_CALL); a plain co_await is allowed between sites, but packing an
 * element suspended there aborts. A site is one statement and may be the
 * body of an if or a loop; its case label must not be inside a switch of
 * the user's own; two sites must not share a line. A function awaited from
 * a coroutine that contains a site returns ck::coro::task<T>, has its own
 * CK_RESUMABLE/CK_END and is awaited with CK_CALL or CK_CALL_TO.
 *
 * At most one coroutine per entry method is active in an element at a time.
 *
 * +ck_fakemigrate on the command line makes every site rebuild the element's
 * coroutines in place (state packed and unpacked, frames destroyed, entries
 * invoked again to the recorded sites), so programs can check the rules
 * without a load balancer.
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

namespace ck {
namespace coro {

class state;
struct record;

// The record of the coroutine running on this PE thread, if any.
inline thread_local record *current_ = nullptr;
// The frame handle of a coroutine entry just created, picked up by CK_COROUTINE.
inline thread_local std::coroutine_handle<> pending_root_ = {};

[[noreturn]] inline void abort_unhandled_exception() noexcept {
  CkAbort("ck::coro: unhandled C++ exception in a coroutine entry method");
  std::abort();
}

/* ---------------- task<T>: awaited helper coroutines ---------------- */

template <class T> class task;

namespace detail {

struct promise_base {
  std::coroutine_handle<> continuation;
  std::suspend_always initial_suspend() const noexcept { return {}; }
  struct final_awaiter {
    bool await_ready() const noexcept { return false; }
    template <class P>
    std::coroutine_handle<> await_suspend(std::coroutine_handle<P> h) const noexcept {
      std::coroutine_handle<> c = h.promise().continuation;
      return c ? c : std::noop_coroutine();
    }
    void await_resume() const noexcept {}
  };
  final_awaiter final_suspend() const noexcept { return {}; }
  void unhandled_exception() const noexcept { abort_unhandled_exception(); }
};

template <class T> struct promise : promise_base {
  T value{};
  task<T> get_return_object() noexcept;
  template <class U> void return_value(U &&v) { value = std::forward<U>(v); }
};

template <> struct promise<void> : promise_base {
  task<void> get_return_object() noexcept;
  void return_void() const noexcept {}
};

} // namespace detail

// A lazily started coroutine; awaiting it starts it, and it resumes the
// awaiter when it finishes (as campi::task).
template <class T = void> class [[nodiscard]] task {
 public:
  using promise_type = detail::promise<T>;
  using handle_type = std::coroutine_handle<promise_type>;

  task() noexcept = default;
  explicit task(handle_type h) noexcept : h_(h) {}
  task(task &&o) noexcept : h_(std::exchange(o.h_, {})) {}
  task &operator=(task &&o) noexcept {
    if (this != &o) { if (h_) h_.destroy(); h_ = std::exchange(o.h_, {}); }
    return *this;
  }
  task(const task &) = delete;
  task &operator=(const task &) = delete;
  ~task() { if (h_) h_.destroy(); }

  bool await_ready() const noexcept { return !h_ || h_.done(); }
  std::coroutine_handle<> await_suspend(std::coroutine_handle<> awaiting) noexcept {
    h_.promise().continuation = awaiting;
    return h_;
  }
  T await_resume() {
    if constexpr (!std::is_void_v<T>) return std::move(h_.promise().value);
  }

 private:
  handle_type h_{};
};

namespace detail {
template <class T> task<T> promise<T>::get_return_object() noexcept {
  return task<T>(std::coroutine_handle<promise<T>>::from_promise(*this));
}
inline task<void> promise<void>::get_return_object() noexcept {
  return task<void>(std::coroutine_handle<promise<void>>::from_promise(*this));
}
} // namespace detail

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

/* ---------------- restart sites ---------------- */

// The restart chain of one coroutine: entry d is the last site (CK_WHEN or
// CK_CALL, identified by __LINE__) noted by the resumable function at depth
// d. On a restarted coroutine, each function on the recorded path switches
// to its recorded site.
struct chain {
  std::vector<int> sites;      // pupped
  std::vector<char> linked;    // per depth: entered through a noted call site
  unsigned depth = 0;          // active frame_guards
  bool restarting = false;     // the body runs again toward the recorded site
  bool noted = false;          // a site was noted and its await is next
  bool notedCall = false;      // ... a call site (else a when site)
  unsigned notedDepth = 0;     // ... by the function at this depth
  bool destroying = false;     // frame_guard destructors leave the sites alone
  void pup(PUP::er &p) { p | sites; }
};

inline chain &current_chain();

class frame_guard;
void note_site(frame_guard &g, int line, bool call = false) noexcept;
[[noreturn]] void abort_no_site(const frame_guard &g) noexcept;

class frame_guard {
 public:
  frame_guard() noexcept;
  ~frame_guard();
  frame_guard(const frame_guard &) = delete;
  frame_guard &operator=(const frame_guard &) = delete;
  int site() const noexcept { return site_; }

 private:
  friend void note_site(frame_guard &g, int line, bool call) noexcept;
  friend void abort_no_site(const frame_guard &g) noexcept;
  chain *c_;
  unsigned depth_;
  int site_;
};

/* ---------------- the inbox ---------------- */

template <class T>
inline constexpr bool is_message_ptr_v =
    std::is_pointer_v<T> && std::is_base_of_v<CkMessage, std::remove_pointer_t<T>>;

// One buffered delivery: a message, or marshalled arguments as pupped bytes.
struct buffer {
  int ep = -1;
  bool hasRef = false;
  CMK_REFNUM_TYPE ref = 0;
  bool isMsg = false;
  void *msg = nullptr;
  std::vector<char> bytes;

  buffer() = default;
  buffer(buffer &&o) noexcept
      : ep(o.ep), hasRef(o.hasRef), ref(o.ref), isMsg(o.isMsg),
        msg(std::exchange(o.msg, nullptr)), bytes(std::move(o.bytes)) {}
  buffer &operator=(buffer &&o) noexcept {
    if (this != &o) {
      release();
      ep = o.ep; hasRef = o.hasRef; ref = o.ref; isMsg = o.isMsg;
      msg = std::exchange(o.msg, nullptr); bytes = std::move(o.bytes);
    }
    return *this;
  }
  buffer(const buffer &) = delete;
  buffer &operator=(const buffer &) = delete;
  ~buffer() { release(); }
  void release() { if (msg) { CkFreeMsg(msg); msg = nullptr; } }

  void pup(PUP::er &p) {
    p | ep; p | hasRef; p | ref; p | isMsg;
    if (isMsg) CkPupMessage(p, &msg);
    else p | bytes;
  }
};

// What a suspended coroutine waits for.
struct want {
  int ep;
  bool hasRef;
  CMK_REFNUM_TYPE ref;
  bool matches(const buffer &b) const noexcept {
    return b.ep == ep && (!hasRef || (b.hasRef && b.ref == ref));
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
  bool waiting = false;          // suspended in a when awaitable
  bool atSite = false;           // ... reached through CK_WHEN
  std::vector<want> wants;       // what it waits for (one entry; several for when_any)
  bool ready = false;            // delivered: which and the buffer
  int readyIndex = -1;
  buffer readyBuf;
  bool fakeRestart = false;      // rebuild once it has suspended (+ck_fakemigrate)

  void pup(PUP::er &p) {
    p | ep;
    p | target;
    bool hasMsg = msg != nullptr;
    p | hasMsg;
    if (hasMsg) CkPupMessage(p, &msg);
    ch.pup(p);
  }
};

class state;
inline state &current_state();

/* ---------------- when awaitable ---------------- */

template <class... Outs> class when_awaiter {
 public:
  when_awaiter(state &s, want w, Outs &...outs) noexcept : s_(s), w_(w), outs_(outs...) {}
  bool await_ready();
  void await_suspend(std::coroutine_handle<> h);
  void await_resume();

 private:
  void unpack(buffer &b);
  state &s_;
  want w_;
  std::tuple<Outs &...> outs_;
  record *rec_ = nullptr;
};

/* ---------------- state: per-element coroutines and inbox ---------------- */

class state {
 public:
  state() = default;
  state(const state &) = delete;
  state &operator=(const state &) = delete;
  ~state();

  // Inbox entry bodies: hand the entry's message or arguments to the state.
  template <class... Args> void deliver(int ep, Args &&...args);

  // Await a delivery to entry ep with reference number ref (or any, with
  // noref), storing the message or arguments in outs. CK_WHEN wraps this.
  template <class... Outs>
  when_awaiter<Outs...> when(int ep, CMK_REFNUM_TYPE ref, Outs &...outs) noexcept {
    return when_awaiter<Outs...>(*this, want{ep, true, ref}, outs...);
  }
  template <class... Outs>
  when_awaiter<Outs...> when(int ep, noref_t, Outs &...outs) noexcept {
    return when_awaiter<Outs...>(*this, want{ep, false, 0}, outs...);
  }

  void pup(PUP::er &p);

  // Internal.
  record &begin(ArrayElement *elt, int ep, void *msg);
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
  }
  return v == 1;
}

inline chain &current_chain() {
  if (current_ == nullptr)
    CkAbort("ck::coro: CK_RESUMABLE/CK_WHEN/CK_CALL used outside a coroutine entry method");
  return current_->ch;
}

inline state &current_state() {
  if (current_ == nullptr)
    CkAbort("ck::coro: CK_WHEN used outside a coroutine entry method");
  return *current_->owner;
}

inline frame_guard::frame_guard() noexcept : c_(&current_chain()), depth_(0), site_(0) {
  chain &c = *c_;
  depth_ = c.depth++;
  if (c.linked.size() < c.depth) c.linked.resize(c.depth);
  c.linked[depth_] = depth_ == 0 || (c.noted && c.notedCall && c.notedDepth == depth_ - 1);
  c.noted = false;
  if (c.restarting) {
    if (depth_ >= c.sites.size())
      CkAbort("ck::coro: restart reached a function that was not on the recorded path "
              "(depth %u, %zu sites recorded)", depth_, c.sites.size());
    site_ = c.sites[depth_];
  }
}

inline frame_guard::~frame_guard() {
  chain &c = *c_;
  if (c.destroying) return;
  c.depth--;
  if (!c.restarting && c.sites.size() > depth_) c.sites.resize(depth_);
}

inline void note_site(frame_guard &g, int line, bool call) noexcept {
  chain &c = *g.c_;
  const unsigned d = g.depth_;
  if (c.restarting) {
    if (d >= c.sites.size() || c.sites[d] != line)
      CkAbort("ck::coro: restart reached site %d, recorded site %d (depth %u)", line,
              d < c.sites.size() ? c.sites[d] : -1, d);
    const bool innermost = (d + 1 == c.sites.size());
    if (innermost == call)
      CkAbort("ck::coro: recorded site %d (depth %u) is a %s site but the recorded path %s",
              line, d, call ? "call" : "when", call ? "ends there" : "continues into a call");
  } else {
    if (c.sites.size() < d + 1) c.sites.resize(d + 1);
    c.sites[d] = line;
    c.sites.resize(d + 1);
  }
  c.noted = true;
  c.notedCall = call;
  c.notedDepth = d;
}

inline void abort_no_site(const frame_guard &g) noexcept {
  CkAbort("ck::coro: recorded restart site %d is not a site of the resumable function at "
          "depth %u", g.site_, g.depth_);
  std::abort();
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
  }
  rec->root = root;
  rec->h = root;
  rec->prev = current_;
  current_ = rec;
  return *rec;
}

// After the body ran (first run or a resume) and suspended or finished.
inline void state::finish_run(record *rec) {
  current_ = rec->prev;
  rec->prev = nullptr;
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
      current_ = &rec;
      std::coroutine_handle<> h = std::exchange(rec.h, {});
      h.resume();
      finish_run(&rec);
      return;
    }
  }
  inbox_.push_back(std::move(b));
}

template <class... Args> inline void state::deliver(int ep, Args &&...args) {
  buffer b;
  b.ep = ep;
  if constexpr (sizeof...(Args) == 1 && (is_message_ptr_v<std::decay_t<Args>> && ...)) {
    void *m = (std::forward<Args>(args), ...);
    b.isMsg = true;
    b.msg = m;
    b.hasRef = true;
    b.ref = CkGetRefNum(m);
  } else {
    std::tuple<std::decay_t<Args>...> t(std::forward<Args>(args)...);
    if constexpr (sizeof...(Args) > 0) {
      using First = std::tuple_element_t<0, decltype(t)>;
      if constexpr (std::is_arithmetic_v<First>) {
        b.hasRef = true;
        b.ref = (CMK_REFNUM_TYPE)std::get<0>(t);
      }
    }
    PUP::sizer ps;
    ps | t;
    b.bytes.resize(ps.size());
    PUP::toMem pm(b.bytes.data());
    pm | t;
  }
  push(std::move(b));
}

inline void state::send_restart(record &rec) {
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
      rec.ch.restarting = true;
      send_restart(rec);
    }
  }
}

/* ---- when_awaiter ---- */

template <class... Outs> inline void when_awaiter<Outs...>::unpack(buffer &b) {
  if constexpr (sizeof...(Outs) == 1 && (is_message_ptr_v<Outs> && ...)) {
    if (!b.isMsg) CkAbort("ck::coro: entry %d delivered marshalled arguments but the when "
                          "expects a message", w_.ep);
    std::get<0>(outs_) = static_cast<std::tuple_element_t<0, std::tuple<Outs...>>>(b.msg);
    b.msg = nullptr; // the coroutine owns the message now
  } else {
    if (b.isMsg) CkAbort("ck::coro: entry %d delivered a message but the when expects "
                         "marshalled arguments", w_.ep);
    PUP::fromMem pf(b.bytes.data());
    std::apply([&](Outs &...o) { ((pf | o), ...); }, outs_);
  }
}

template <class... Outs> inline bool when_awaiter<Outs...>::await_ready() {
  record *rec = current_;
  if (rec == nullptr) CkAbort("ck::coro: co_await when outside a coroutine entry method");
  rec->ch.restarting = false; // a restarted body has reached its site
  const bool atSite = rec->ch.noted && !rec->ch.notedCall;
  rec->ch.noted = false;
  buffer taken;
  if (!s_.take(w_, taken)) {
    rec_ = rec;
    rec->atSite = atSite;
    return false;
  }
  unpack(taken);
  return true;
}

template <class... Outs> inline void when_awaiter<Outs...>::await_suspend(std::coroutine_handle<> h) {
  record *rec = rec_;
  rec->h = h;
  rec->waiting = true;
  rec->wants.assign(1, w_);
  if (rec->atSite && fake_migrate_enabled()) rec->fakeRestart = true;
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
#define CK_COROUTINE(co_, ep_) CK_COROUTINE_MSG(co_, ep_, nullptr)

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
