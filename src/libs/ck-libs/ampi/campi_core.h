/*
 * campi (coroutine AMPI), core header: the rank task type and the awaitable
 * replacements for AMPI's blocking calls. Requires C++20 (<coroutine>).
 *
 * An AMPI rank in campi mode is a C++20 coroutine instead of a user-level
 * thread. Every call that would block a thread-based rank is a coroutine
 * here, awaited with co_await: it posts AMPI's nonblocking variant and
 * suspends until AMPI's delivery code wakes the rank (TCharm::resume, which
 * campi redirects from CthAwaken to a coroutine resume).
 *
 * Applications normally include campi.h (this header plus macros that
 * redirect the blocking MPI names to co_await expressions) and define their
 * main with CAMPI_MAIN. campi.C includes only this core header.
 */
#ifndef CAMPI_CORE_H
#define CAMPI_CORE_H

#if !defined(__cplusplus) || __cplusplus < 202002L
#error "campi requires C++20: compile with -std=c++20"
#endif

#include <coroutine>
#include <type_traits>
#include <utility>
#include "mpi.h"

namespace campi {

struct Rank; // per-rank record, defined in campi.C

// Called by campi's coroutine machinery; defined in campi.C.
[[noreturn]] void abort_unhandled_exception() noexcept;
// Record h as the frame to resume when the running rank is next woken.
void set_resume_point(std::coroutine_handle<> h) noexcept;

/* ---------------- task<T>: the rank's coroutine type ---------------- */

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
      // the outermost task returns control to campi's resume loop.
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

  // Awaiting a task starts it; it resumes the awaiter when it finishes.
  bool await_ready() const noexcept { return !h_ || h_.done(); }
  std::coroutine_handle<> await_suspend(std::coroutine_handle<> awaiting) noexcept {
    h_.promise().continuation = awaiting;
    return h_;
  }
  T await_resume() {
    if constexpr (!std::is_void_v<T>) return std::move(h_.promise().value);
  }

  // For campi's rank start: take ownership of the coroutine frame.
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

/* ---------------- rank main registration ---------------- */

using main_fn = task<int> (*)(int argc, char **argv);
extern main_fn registered_main; // non-null selects campi mode
void register_main(main_fn f) noexcept;
struct main_registrar {
  explicit main_registrar(main_fn f) noexcept { register_main(f); }
};

/* ---------------- suspension primitive ---------------- */

// Return to the scheduler until something wakes this rank. Wake-ups may be
// spurious, so always use it in a loop that re-checks the condition:
//   while (!pred()) co_await campi::suspend();
struct suspend_awaiter {
  bool await_ready() const noexcept { return false; }
  void await_suspend(std::coroutine_handle<> h) const noexcept { set_resume_point(h); }
  void await_resume() const noexcept {}
};
inline suspend_awaiter suspend() noexcept { return {}; }

// Like suspend(), but also schedules a wake-up of this rank, so the rank runs
// again after the messages already queued on this PE (cf. AMPI_Yield).
task<void> yield();

// An awaitable that is already complete with an int result.
struct ready_int {
  int v;
  bool await_ready() const noexcept { return true; }
  void await_suspend(std::coroutine_handle<>) const noexcept {}
  int await_resume() const noexcept { return v; }
};

/* ---------------- request completion ---------------- */

task<int> Wait(MPI_Request *req, MPI_Status *sts);
task<int> Waitall(int n, MPI_Request *reqs, MPI_Status *sts);
inline task<int> wait(MPI_Request *req, MPI_Status *sts) { return Wait(req, sts); }
inline task<int> waitall(int n, MPI_Request *reqs, MPI_Status *sts) { return Waitall(n, reqs, sts); }

/* ---------------- blocking point-to-point ---------------- */

task<int> Send(const void *buf, int count, MPI_Datatype type, int dest, int tag, MPI_Comm comm);
task<int> Ssend(const void *buf, int count, MPI_Datatype type, int dest, int tag, MPI_Comm comm);
task<int> Recv(void *buf, int count, MPI_Datatype type, int src, int tag, MPI_Comm comm,
               MPI_Status *sts);
task<int> Sendrecv(const void *sbuf, int scount, MPI_Datatype stype, int dest, int stag,
                   void *rbuf, int rcount, MPI_Datatype rtype, int src, int rtag,
                   MPI_Comm comm, MPI_Status *sts);

/* ---------------- blocking collectives ---------------- */

task<int> Barrier(MPI_Comm comm);
task<int> Bcast(void *buf, int count, MPI_Datatype type, int root, MPI_Comm comm);
task<int> Reduce(const void *sbuf, void *rbuf, int count, MPI_Datatype type, MPI_Op op,
                 int root, MPI_Comm comm);
task<int> Allreduce(const void *sbuf, void *rbuf, int count, MPI_Datatype type, MPI_Op op,
                    MPI_Comm comm);

/* ---------------- communicator creation ---------------- */

task<int> Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm);
task<int> Comm_create(MPI_Comm comm, MPI_Group group, MPI_Comm *newcomm);

/* ---------------- startup, shutdown, migration ---------------- */

task<int> Init(int *argc, char ***argv);
task<int> Init_thread(int *argc, char ***argv, int required, int *provided);
task<int> Finalize();
// Increment 1: migration is not supported; this keeps converted programs
// (AMPI_Migrate calls) compiling and returns MPI_SUCCESS at once.
inline ready_int Migrate(MPI_Info) noexcept { return ready_int{MPI_SUCCESS}; }

} // namespace campi

#endif // CAMPI_CORE_H
