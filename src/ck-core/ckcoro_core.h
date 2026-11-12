/*
 * ckcoro_core: the C++20 coroutine machinery shared by Charm++ coroutine
 * entry methods (ckcoro.h) and coroutine AMPI (campi_core.h): the awaited
 * task<T> and the chain of restart sites that lets a rebuilt coroutine run
 * again from the top to its recorded site. Requires C++20.
 *
 * The owner of a coroutine (a ck::coro::state record, a campi rank) keeps
 * one chain per coroutine and sets current_chain_ to it around every run of
 * the coroutine; frame_guard, created by CK_CORO_TASK/CAMPI_RESUMABLE (and
 * CK_COROUTINE), reads it there.
 */
#ifndef CKCORO_CORE_H
#define CKCORO_CORE_H

#if !defined(__cplusplus) || __cplusplus < 202002L
#error "ckcoro_core.h requires C++20: compile with -std=c++20"
#endif

#include <coroutine>
#include <cstdlib>
#include <type_traits>
#include <utility>
#include <vector>
#include "charm.h"
#include "pup.h"
#include "pup_stl.h"

namespace ck {
namespace coro {

[[noreturn]] inline void abort_unhandled_exception() noexcept {
  CkAbort("ck::coro: unhandled C++ exception in a coroutine");
  std::abort(); // CkAbort is not declared noreturn
}

/* ---------------- task<T>: awaited helper coroutines ---------------- */

template <class T> class task;

namespace detail {

struct promise_base {
  std::coroutine_handle<> continuation; // the awaiting coroutine, if nested
  std::suspend_always initial_suspend() const noexcept { return {}; }
  struct final_awaiter {
    bool await_ready() const noexcept { return false; }
    template <class P>
    std::coroutine_handle<> await_suspend(std::coroutine_handle<P> h) const noexcept {
      // A nested task resumes the task that awaited it (symmetric transfer);
      // the outermost task returns control to its owner's resume loop.
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
// awaiter when it finishes.
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

  // For an owner that runs the task as a root: take ownership of the frame.
  handle_type release() noexcept { return std::exchange(h_, {}); }

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

/* ---------------- restart sites ---------------- */

// The restart chain of one coroutine: entry d is the last site (identified
// by __LINE__) noted by the resumable function at depth d. On a restarted
// coroutine, each function on the recorded path switches to its recorded
// site: frame_guard::site() returns it while the chain is restarting, and 0
// (the function's start) otherwise.
struct chain {
  std::vector<int> sites;      // pupped
  std::vector<char> linked;    // per depth: entered through a noted call site
  unsigned depth = 0;          // active frame_guards
  bool restarting = false;     // the body runs again toward the recorded site
  bool noted = false;          // a site was noted and its await is next
  bool notedCall = false;      // ... a call site (else a when or migrate site)
  unsigned notedDepth = 0;     // ... by the function at this depth
  bool destroying = false;     // frame_guard destructors leave the sites alone
  void pup(PUP::er &p) { p | sites; }
};

// The chain of the coroutine running on this PE thread, if any; set by the
// coroutine's owner around each run.
inline thread_local chain *current_chain_ = nullptr;

class frame_guard;
// Record line as the site of g's function: a suspension site (call == false)
// or a call site (call == true, followed by co_await of a resumable
// function). On a restarting chain it checks that line is the recorded site.
void note_site(frame_guard &g, int line, bool call = false) noexcept;
// The recorded site of a restarting chain is not a case in g's function.
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

inline frame_guard::frame_guard() noexcept : c_(current_chain_), depth_(0), site_(0) {
  if (c_ == nullptr) {
    CkAbort("ck::coro: CK_CORO_TASK/CAMPI_RESUMABLE outside a coroutine");
    std::abort();
  }
  chain &c = *c_;
  depth_ = c.depth++;
  if (c.linked.size() < c.depth) c.linked.resize(c.depth);
  // Entered through a call site noted by the caller (the root needs none).
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
  if (c.destroying) return; // the frames are destroyed for a rebuild
  c.depth--;
  // A finished function's sites (and its callees') are not on any later path.
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

} // namespace coro
} // namespace ck

#endif // CKCORO_CORE_H
