#ifndef LB_OFF_THREAD_H
#define LB_OFF_THREAD_H
#include <sched.h>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

// CPU set for an LB helper thread, from a spec string:
//   near[:<domain>:<first>-<last>]  the idle cores of the calling PE's NUMA domain (default 16-core
//                                   domains with the PEs on the first half: the gpuA40x4 wrappers)
//   <a>-<b>[,<c>-<d>...]            an explicit core list
//   anything else                   unpinned
// Returns false for an unset, empty or "0" spec (no helper). Shared by the CUPTI load build
// (CHARM_LB_CUPTI_OFFTHREAD, DistBaseLB.C) and the strategy (CHARM_LB_STRATEGY_THREAD, CentralLB.C).
// One switch for everything the helper runs: CHARM_LB_HELPER=near|<cpulist>|1 turns on both the
// CUPTI load build and the strategy; the older per-job names (CHARM_LB_CUPTI_OFFTHREAD,
// CHARM_LB_STRATEGY_THREAD) still work alone, for bisecting.
inline const char* lbHelperSpec(const char* perJobName) {
  const char* all = getenv("CHARM_LB_HELPER");
  return all != nullptr ? all : getenv(perJobName);
}

inline bool lbOffThreadCpus(const char* spec, cpu_set_t* set) {
  if (spec == nullptr || *spec == '\0' || strcmp(spec, "0") == 0) return false;
  CPU_ZERO(set);
  bool any = false;
  if (strncmp(spec, "near", 4) == 0) {
    int dom = 16, lo = 8, hi = 15;
    if (spec[4] == ':') sscanf(spec + 5, "%d:%d-%d", &dom, &lo, &hi);
    const int cpu = sched_getcpu();
    if (cpu >= 0 && dom > 0) {
      const int base = (cpu / dom) * dom;
      for (int c = base + lo; c <= base + hi && c < CPU_SETSIZE; c++) { CPU_SET(c, set); any = true; }
    }
  } else if (strchr(spec, '-') != nullptr || strchr(spec, ',') != nullptr) {
    const char* p = spec;
    while (*p) {
      char* end = nullptr;
      long lo = strtol(p, &end, 10), hi = lo;
      if (end == p) break;
      if (*end == '-') { p = end + 1; hi = strtol(p, &end, 10); }
      for (long c = lo; c <= hi && c < CPU_SETSIZE; c++) if (c >= 0) { CPU_SET((int)c, set); any = true; }
      p = (*end == ',') ? end + 1 : end;
      if (*end != ',') break;
    }
  }
  if (!any) for (int c = 0; c < CPU_SETSIZE; c++) CPU_SET(c, set);
  return true;
}

// ONE persistent helper thread per process for LB work that must not hold a PE thread: the CUPTI load
// build at the window close and the strategy on the decision PE. Jobs run in FIFO order; the
// completion is whatever the job sets (a flag the PE polls from CcdSCHEDLOOP). Persistent rather
// than one thread per job: a thread that has called into CUPTI leaves per-thread tracing state
// behind (DistBaseLB.C). The state is heap-allocated and never destroyed, on purpose: a static
// condition_variable is destroyed by exit() while the detached helper waits on it, and
// pthread_cond_destroy then blocks every process in __run_exit_handlers (job 22284602).
// One helper per LANE: a process driving several devices has one device group per device, each
// with its own CUPTI build and strategy per round, and one thread serialised them (lane = the
// device index; a one-device process uses lane 0 only, as before).
struct LBHelper {
  std::mutex m;
  std::condition_variable cv;
  std::deque<std::function<void()>> jobs;
  bool started = false;
  static LBHelper& lane(int i) {
    static std::mutex* lanes_m = new std::mutex;
    static std::vector<LBHelper*>* lanes = new std::vector<LBHelper*>;
    std::lock_guard<std::mutex> lk(*lanes_m);
    if (i < 0) i = 0;
    while ((int)lanes->size() <= i) lanes->push_back(new LBHelper);
    return *(*lanes)[i];
  }
  static LBHelper& get() { return lane(0); }
};

inline void lbHelperSubmit(const cpu_set_t& cpus, std::function<void()> job, int laneIdx = 0) {
  LBHelper& h = LBHelper::lane(laneIdx);
  std::unique_lock<std::mutex> lk(h.m);
  if (!h.started) {
    h.started = true;
    std::thread([cpus, laneIdx]() {
      sched_setaffinity(0, sizeof(cpus), &cpus);
      LBHelper& h = LBHelper::lane(laneIdx);
      for (;;) {
        std::function<void()> job;
        {
          std::unique_lock<std::mutex> lk(h.m);
          h.cv.wait(lk, [&h] { return !h.jobs.empty(); });
          job = std::move(h.jobs.front());
          h.jobs.pop_front();
        }
        job();
      }
    }).detach();
  }
  h.jobs.push_back(std::move(job));
  h.cv.notify_one();
}
#endif
