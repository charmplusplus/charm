/*
 * campi (coroutine AMPI): AMPI ranks as C++20 coroutines instead of threads.
 * Compiled with -std=c++20 into libmoduleampi. See campi_core.h.
 *
 * Structure:
 *  - Rank record and context switching (campi_start, campi_resume): every
 *    rank on a PE shares the scheduler thread's Ctv slots, so the rank's
 *    AMPI context (_curTCharm, ampiPtr, ampiInitDone, ampiFinalized) is
 *    installed at each resume and saved back at each suspension.
 *  - Wake-ups: TCharm::start()/resume() call campi_wake (via a tcharm hook),
 *    which enqueues a small Converse message whose handler resumes the rank,
 *    as CthAwaken enqueues a thread. Resuming inline instead is unsafe: AMPI
 *    wakes ranks from inside constructors (registerAmpi's semaPut runs in the
 *    ampi element constructor before myComm is set) and from the middle of
 *    delivery code, which a thread-based rank never observes.
 *  - Awaitables: blocking MPI calls as "post the nonblocking variant, then
 *    suspend until the request completes", mirroring what the request's
 *    wait() does before it blocks a thread.
 *  - Init/Finalize and communicator creation.
 */
#include "ampiimpl.h"
#include "tcharm_impl.h"

#if !(__cplusplus >= 202002L && defined(__cpp_impl_coroutine))
// The compiler cannot build coroutines (e.g. GCC 10 without -fcoroutines):
// libampi is built without campi, and no program can select campi mode.
void campi_parent_attached(TCharm *, ampiParent *) noexcept {}
#else

#include "campi_core.h"

#include <cstdlib>
#include <cstring>

CtvExtern(ampiParent*, ampiPtr);
CtvExtern(bool, ampiInitDone);
CtvExtern(bool, ampiFinalized);
CtvExtern(void*, stackBottom);
CkpvExtern(int, ampiThreadLevel);

// Set CAMPI_DEBUG=1 in the environment for a trace of rank scheduling.
static int campiDebug = -1;
#define CAMPI_DBG(...)                                                     \
  do {                                                                     \
    if (campiDebug > 0) { CmiPrintf("[campi pe %d] ", CkMyPe());          \
                          CmiPrintf(__VA_ARGS__); fflush(stdout); }        \
  } while (0)

namespace campi {

/* ====================== rank record ====================== */

struct Rank {
  TCharm *tc = nullptr;
  ampiParent *parent = nullptr;
  std::coroutine_handle<> h;                          // frame to resume next
  std::coroutine_handle<detail::promise<int>> root;   // the rank's main
  bool running = false;        // inside campi_resume for this rank
  bool resumePending = false;  // woken while running
  bool wakeQueued = false;     // a wake message is in the scheduler queue
  bool initDone = false;       // saved Ctv ampiInitDone
  bool finalized = false;      // saved Ctv ampiFinalized
  bool finished = false;       // main returned and AMPI_Exit was called
  int exitCode = 0;
  char **argv = nullptr;
  // Init on element 0: completion of an array creation (CkCallback target)
  bool arrayCreated = false;
  CkArrayID createdAid;
};

main_fn registered_main = nullptr;

static void campi_start(TCharm *tc);
static void campi_wake(void *rank);

void register_main(main_fn f) noexcept {
  registered_main = f;
  // Selects campi mode in TCharm::procInit (tcharm cannot refer to campi).
  tcharm_campi_start_fn = campi_start;
  tcharm_campi_resume_fn = campi_wake;
}

[[noreturn]] void abort_unhandled_exception() noexcept {
  CkAbort("campi: unhandled C++ exception in an AMPI rank coroutine");
  std::abort(); // CkAbort is not declared noreturn
}

static inline Rank *current() noexcept {
  TCharm *tc = TCharm::getNULL();
  return tc ? (Rank *)tc->getCampiRank() : nullptr;
}

void set_resume_point(std::coroutine_handle<> h) noexcept {
  Rank *r = current();
  if (r == nullptr)
    CkAbort("campi: co_await on a campi awaitable outside a campi rank");
  r->h = h;
}

/* ====================== resume / wake ====================== */

static void campi_resume(Rank *r);

struct WakeMsg {
  char header[CmiMsgHeaderSizeBytes];
  Rank *rank;
};

// Per PE (the handler table is per PE; wake messages never leave the PE).
static thread_local int wakeHandlerIdx = -1;

static void wakeHandler(void *msg) {
  Rank *r = ((WakeMsg *)msg)->rank;
  CmiFree(msg);
  r->wakeQueued = false;
  campi_resume(r);
}

static void enqueue_wake(Rank *r) {
  if (r->wakeQueued || r->finished) return;
  r->wakeQueued = true;
  WakeMsg *m = (WakeMsg *)CmiAlloc(sizeof(WakeMsg));
  m->rank = r;
  CmiSetHandler(m, wakeHandlerIdx);
  CsdEnqueue(m);
}

// tcharm hook: TCharm::start() (and so resume(), semaPut, setInitDoneFlag,
// AMPI's delivery code) lands here instead of CthAwaken.
static void campi_wake(void *rp) {
  Rank *r = (Rank *)rp;
  CAMPI_DBG("wake rank %d (running %d, queued %d)\n", r ? r->tc->getElement() : -1,
            r ? (int)r->running : -1, r ? (int)r->wakeQueued : -1);
  if (r == nullptr || r->finished) return;
  enqueue_wake(r);
}

static void finish_rank(Rank *r) {
  // The rank's main returned: exit on its behalf, with its context installed.
  r->exitCode = r->root.promise().value;
  r->root.destroy();
  r->root = {};
  r->h = {};
  r->finished = true;
  AMPI_Exit(r->exitCode); // contributes to TCharm's exit reduction; returns in campi mode
}

static void campi_resume(Rank *r) {
  if (r->finished) return;
  if (r->running) { // re-entrancy guard (cannot happen with queued wake-ups)
    r->resumePending = true;
    return;
  }
  do {
    r->resumePending = false;
    if (!r->h) CkAbort("campi: rank woken with no resume point");

    // Save the current context and install the rank's.
    TCharm *oldTc = CtvAccess(_curTCharm);
    ampiParent *oldParent = CtvAccess(ampiPtr);
    bool oldInitDone = CtvAccess(ampiInitDone);
    bool oldFinalized = CtvAccess(ampiFinalized);
    CtvAccess(_curTCharm) = r->tc;
    CtvAccess(ampiPtr) = r->parent;
    CtvAccess(ampiInitDone) = r->initDone;
    CtvAccess(ampiFinalized) = r->finalized;
    r->running = true;
    r->tc->setStopped(false);
#if CMK_LBDB_ON
    // as TCharm::stop_static does around a thread's run
    r->tc->getCkLocRec()->startTiming();
#endif

    std::coroutine_handle<> h = std::exchange(r->h, {});
    CAMPI_DBG("resume rank %d\n", r->tc->getElement());
    h.resume(); // runs until the rank suspends at an awaitable or main returns
    CAMPI_DBG("rank %d suspended (done %d)\n", r->tc->getElement(), (int)r->root.done());

    r->initDone = CtvAccess(ampiInitDone);
    r->finalized = CtvAccess(ampiFinalized);
    if (CtvAccess(ampiPtr) != nullptr) r->parent = CtvAccess(ampiPtr);
    if (r->root.done())
      finish_rank(r);
    else if (!r->h)
      CkAbort("campi: rank suspended at an awaitable that is not campi's");

#if CMK_LBDB_ON
    r->tc->getCkLocRec()->stopTiming();
#endif
    r->tc->setStopped(true);
    r->running = false;
    CtvAccess(_curTCharm) = oldTc;
    CtvAccess(ampiPtr) = oldParent;
    CtvAccess(ampiInitDone) = oldInitDone;
    CtvAccess(ampiFinalized) = oldFinalized;
  } while (r->resumePending && !r->finished);
}

// tcharm hook, called from TCharm::run() (the element constructor) in place
// of creating and awakening a thread: what AMPI_threadstart does, as a coroutine.
static void campi_start(TCharm *tc) {
  if (campiDebug < 0) {
    const char *dbg = getenv("CAMPI_DEBUG");
    campiDebug = (dbg != nullptr && atoi(dbg) != 0) ? 1 : 0;
  }
  if (wakeHandlerIdx < 0) {
    wakeHandlerIdx = CmiRegisterHandler((CmiHandler)wakeHandler);
  }
  Rank *r = new Rank;
  r->tc = tc;
  tc->setCampiRank(r);

  r->argv = CmiCopyArgs(CkGetArgv());
  int argc = CkGetArgc();
  // A pointer near the "bottom of the stack", for ampiCurrentStackUsage:
  // meaningless without a thread, but keep it non-null.
  CtvAccess(stackBottom) = &r->argv;

  // initial_suspend is suspend_always: no user code runs here.
  r->root = registered_main(r->argv ? argc : 0, r->argv).release();
  r->h = r->root;
  campi_wake(r);
}

} // namespace campi

// ampi.C hook: ampiParent::prepareCtv in campi mode.
void campi_parent_attached(TCharm *tc, ampiParent *parent) noexcept {
  campi::Rank *r = (campi::Rank *)tc->getCampiRank();
  if (r == nullptr) return;
  r->parent = parent;
  if (campi::current() == r) CtvAccess(ampiPtr) = parent;
}

namespace campi {

task<void> yield() {
  Rank *r = current();
  enqueue_wake(r);
  co_await suspend();
}

/* ====================== request completion ====================== */

static inline AmpiRequest *lookup(ampiParent *p, MPI_Request idx) {
  return p->getReqs()[idx];
}

// Completion check without MPI_Test's side effects (status, free).
static inline bool req_done(ampiParent *p, MPI_Request idx) {
  if (idx == MPI_REQUEST_NULL) return true;
  return lookup(p, idx)->test(MPI_STATUS_IGNORE);
}

static inline bool is_coll_req(AmpiReqType t) {
  return t == AMPI_REDN_REQ || t == AMPI_GATHER_REQ || t == AMPI_GATHERV_REQ;
}

// Mark an incomplete request as blocked on, as its wait() does before it
// suspends a thread; returns the number of requests counted in numBlockedReqs.
static int mark_blocked(ampiParent *p, MPI_Request idx, bool blocked) {
  AmpiRequest *q = lookup(p, idx);
  int n = 0;
  switch (q->getType()) {
    case AMPI_ATA_REQ: { // ATAReq::wait waits on its internal requests
      ATAReq *a = (ATAReq *)q;
      for (MPI_Request ri : a->reqs) {
        if (ri == MPI_REQUEST_NULL) continue;
        AmpiRequest *x = lookup(p, ri);
        if (!blocked) { x->setBlocked(false); continue; }
        if (!x->test(MPI_STATUS_IGNORE)) { x->setBlocked(true); n++; }
      }
      break;
    }
    case AMPI_G_REQ: // generalized requests complete by polling
      if (blocked) enqueue_wake(current());
      break;
    default:
      q->setBlocked(blocked);
      n = blocked ? 1 : 0;
      break;
  }
  return n;
}

task<int> Wait(MPI_Request *req, MPI_Status *sts) {
  if (*req != MPI_REQUEST_NULL) {
    ampiParent *p = getAmpiParent();
    while (!req_done(p, *req)) {
      CkAssert(p->numBlockedReqs == 0);
      AmpiReqType t = lookup(p, *req)->getType();
      p->numBlockedReqs = mark_blocked(p, *req, true);
      CAMPI_DBG("rank %d waits on request %d (type %d)\n", TCHARM_Element(), *req, (int)t);
      // IReq/SendReq/SsendReq::wait block via blockOnRecv,
      // RednReq/GatherReq/GathervReq::wait via blockOnColl.
      if (is_coll_req(t)) p->resumeOnColl = true;
      else p->resumeOnRecv = true;
      co_await suspend(); // woken by resumeThreadIfReady / irednResult
      p = getAmpiParent();
      mark_blocked(p, *req, false);
      p->resumeOnRecv = false;
      p->resumeOnColl = false;
      p->numBlockedReqs = 0;
    }
  }
  co_return MPI_Wait(req, sts); // the request is complete: does not block
}

task<int> Waitall(int n, MPI_Request *reqs, MPI_Status *sts) {
  ampiParent *p = getAmpiParent();
  for (;;) {
    bool all = true;
    for (int i = 0; i < n; i++)
      if (!req_done(p, reqs[i])) { all = false; break; }
    if (all) break;
    // as ampiParent::waitall: count the incomplete requests, block once
    CkAssert(p->numBlockedReqs == 0);
    int blocked = 0;
    for (int i = 0; i < n; i++)
      if (!req_done(p, reqs[i])) blocked += mark_blocked(p, reqs[i], true);
    p->numBlockedReqs = blocked;
    CAMPI_DBG("rank %d waits on %d of %d requests\n", TCHARM_Element(), blocked, n);
    p->resumeOnRecv = true;
    p->resumeOnColl = true; // a collective request's completion (irednResult) checks this flag
    co_await suspend();
    p = getAmpiParent();
    for (int i = 0; i < n; i++)
      if (reqs[i] != MPI_REQUEST_NULL) mark_blocked(p, reqs[i], false);
    p->resumeOnRecv = false;
    p->resumeOnColl = false;
    p->numBlockedReqs = 0;
  }
  co_return MPI_Waitall(n, reqs, sts); // all complete: does not block
}

/* ====================== point-to-point ====================== */

task<int> Send(const void *buf, int count, MPI_Datatype type, int dest, int tag, MPI_Comm comm) {
  MPI_Request req;
  int e = MPI_Isend(buf, count, type, dest, tag, comm, &req);
  if (e != MPI_SUCCESS) co_return e;
  co_return co_await Wait(&req, MPI_STATUS_IGNORE);
}

task<int> Ssend(const void *buf, int count, MPI_Datatype type, int dest, int tag, MPI_Comm comm) {
  MPI_Request req;
  int e = MPI_Issend(buf, count, type, dest, tag, comm, &req);
  if (e != MPI_SUCCESS) co_return e;
  co_return co_await Wait(&req, MPI_STATUS_IGNORE);
}

task<int> Recv(void *buf, int count, MPI_Datatype type, int src, int tag, MPI_Comm comm,
               MPI_Status *sts) {
  MPI_Request req;
  int e = MPI_Irecv(buf, count, type, src, tag, comm, &req);
  if (e != MPI_SUCCESS) co_return e;
  co_return co_await Wait(&req, sts);
}

task<int> Sendrecv(const void *sbuf, int scount, MPI_Datatype stype, int dest, int stag,
                   void *rbuf, int rcount, MPI_Datatype rtype, int src, int rtag,
                   MPI_Comm comm, MPI_Status *sts) {
  MPI_Request reqs[2];
  int e = MPI_Irecv(rbuf, rcount, rtype, src, rtag, comm, &reqs[0]);
  if (e != MPI_SUCCESS) co_return e;
  e = MPI_Isend(sbuf, scount, stype, dest, stag, comm, &reqs[1]);
  if (e != MPI_SUCCESS) co_return e;
  MPI_Status st[2];
  e = co_await Waitall(2, reqs, st);
  if (sts != MPI_STATUS_IGNORE) *sts = st[0];
  co_return e;
}

/* ====================== collectives ====================== */

#define CAMPI_POST_AND_WAIT(call)               \
  MPI_Request req;                              \
  int e = (call);                               \
  if (e != MPI_SUCCESS) co_return e;            \
  co_return co_await Wait(&req, MPI_STATUS_IGNORE);

task<int> Barrier(MPI_Comm comm) {
  CAMPI_POST_AND_WAIT(MPI_Ibarrier(comm, &req))
}
task<int> Bcast(void *buf, int count, MPI_Datatype type, int root, MPI_Comm comm) {
  CAMPI_POST_AND_WAIT(MPI_Ibcast(buf, count, type, root, comm, &req))
}
task<int> Reduce(const void *sbuf, void *rbuf, int count, MPI_Datatype type, MPI_Op op,
                 int root, MPI_Comm comm) {
  CAMPI_POST_AND_WAIT(MPI_Ireduce(sbuf, rbuf, count, type, op, root, comm, &req))
}
task<int> Allreduce(const void *sbuf, void *rbuf, int count, MPI_Datatype type, MPI_Op op,
                    MPI_Comm comm) {
  CAMPI_POST_AND_WAIT(MPI_Iallreduce(sbuf, rbuf, count, type, op, comm, &req))
}
#undef CAMPI_POST_AND_WAIT

/* ====================== communicator creation ====================== */

// ampi::split, with the suspension replaced by an await on this comm's
// unblock() (sent by ampi::registrationFinish when the new arrays exist).
task<int> Comm_split(MPI_Comm src, int color, int key, MPI_Comm *dest) {
  ampi *ptr = getAmpiParent()->comm2ampi(src);
  const int before = ptr->getNumUnblocks();
  ptr->splitPost(color, key, ptr->getCommStruct().getType());
  while (ptr->getNumUnblocks() == before) co_await suspend();
  ptr->splitFinish(dest);
  if (color == MPI_UNDEFINED) *dest = MPI_COMM_NULL;
  co_return MPI_SUCCESS;
}

// MPI_Comm_create for intracommunicators (a split, as in ampi.C)
task<int> Comm_create(MPI_Comm comm, MPI_Group group, MPI_Comm *newcomm) {
  std::vector<int> vec = getAmpiParent()->group2vec(group);
  if (vec.size() == 0) {
    *newcomm = MPI_COMM_NULL;
    co_return MPI_SUCCESS;
  }
  if (getAmpiParent()->isInter(comm))
    CkAbort("campi: MPI_Comm_create on an intercommunicator is not supported yet");
  int rank_in_group, color, key;
  MPI_Group_rank(group, &rank_in_group);
  if (rank_in_group == MPI_UNDEFINED) {
    color = MPI_UNDEFINED;
    key = 0;
  } else {
    MPI_Group group_of_comm;
    MPI_Comm_group(comm, &group_of_comm);
    int zero = 0;
    MPI_Group_translate_ranks(group, 1, &zero, group_of_comm, &color);
    key = rank_in_group;
  }
  co_return co_await Comm_split(comm, color, key, newcomm);
}

/* ====================== Init / Finalize ====================== */

static void array_created(void *param, void *msg) {
  Rank *r = (Rank *)param;
  r->createdAid = ((CkArrayCreatedMsg *)msg)->aid;
  r->arrayCreated = true;
  delete (CkArrayCreatedMsg *)msg;
  r->tc->resume();
}

// TCharm::semaGet, awaiting instead of suspending the thread
static task<void *> sema_get(int id) {
  TCharm *tc = TCharm::get();
  while (tc->semaPeek(id) == nullptr) {
    tc->findSema(id)->thread = CthSelf(); // non-null: semaPut then calls resume()
    co_await suspend();
    tc = TCharm::get();
  }
  co_return tc->semaGet(id);
}

// createCommSelf (ampi.C) with the awaitable Comm_create
static task<void> create_comm_self() {
  MPI_Comm selfComm;
  MPI_Group worldGroup, selfGroup;
  int ranks[1] = { getAmpiInstance(MPI_COMM_WORLD)->getRank() };
  MPI_Comm_group(MPI_COMM_WORLD, &worldGroup);
  MPI_Group_incl(worldGroup, 1, ranks, &selfGroup);
  co_await Comm_create(MPI_COMM_WORLD, selfGroup, &selfComm);
  MPI_Comm_set_name(selfComm, "MPI_COMM_SELF");
  CkAssert(selfComm == MPI_COMM_SELF);
}

// ampiInit (ampi.C) with awaits in place of its blocking waits
static task<void> ampi_init() {
  if (CtvAccess(ampiInitDone)) co_return; /* Already called ampiInit */
  Rank *r = current();

  int nchunks = TCHARM_Num_elements();
  CkArrayOptions opts;
  CkArrayID parentAid;
  if (TCHARM_Element() == 0) { /* I'm responsible for building the arrays: */
    CkArrayID threads;
    opts = ampiInitRootArrayOptions(&threads, &nchunks);
    r->arrayCreated = false;
    CProxy_ampiParent::ckNew(threads, nchunks, opts, CkCallback(array_created, r));
    while (!r->arrayCreated) co_await suspend();
    parentAid = r->createdAid;
    CAMPI_DBG("init: ampiParent array created\n");
  }
  (void)co_await sema_get(AMPI_BARRIER_SEMAID);
  r = current();
  CAMPI_DBG("init: rank %d has its ampiParent %p\n", TCHARM_Element(), (void *)r->parent);
  CtvAccess(ampiPtr) = r->parent; // set by ampiParent::prepareCtv (campi_parent_attached)

  if (TCHARM_Element() == 0) {
    // Make a new ampi array
    CkArrayID empty;
    ampiCommStruct worldComm(MPI_COMM_WORLD, empty, nchunks);
    r->arrayCreated = false;
    CProxy_ampi::ckNew(parentAid, worldComm, opts, CkCallback(array_created, r));
    while (!r->arrayCreated) co_await suspend();
  }

  // Find our ampi object:
  ampi *ptr = (ampi *)co_await sema_get(AMPI_TCHARM_SEMAID);
  CAMPI_DBG("init: rank %d has its MPI_COMM_WORLD ampi\n", TCHARM_Element());
  ampiParent *pptr = ampiInitWorldLocal(ptr, nchunks);
  current()->initDone = true;
  while (pptr->ampiInitCallDone != 1) {
    co_await suspend(); // woken by ampi::setInitDoneFlag
    pptr = getAmpiParent();
  }

  CAMPI_DBG("init: rank %d past allInitDone\n", TCHARM_Element());
  co_await create_comm_self();
  CAMPI_DBG("init: rank %d has MPI_COMM_SELF\n", TCHARM_Element());
  removeUnimportantArrayObjsfromPeCache();
}

task<int> Init(int *p_argc, char ***p_argv) {
  // MPI_Init around ampiInit
  char **argv;
  if (p_argv) argv = *p_argv;
  else argv = CkGetArgv();
  co_await ampi_init();
  if (p_argc) *p_argc = CmiGetArgc(argv);
  co_return MPI_SUCCESS;
}

task<int> Init_thread(int *p_argc, char ***p_argv, int required, int *provided) {
  // MPI_Init_thread: record the thread level, then MPI_Init
  if (required < MPI_THREAD_SINGLE || required > MPI_THREAD_MULTIPLE)
    co_return MPI_ERR_ARG;
  // AMPI does not support MPI_THREAD_SERIALIZED or MPI_THREAD_MULTIPLE
  CkpvAccess(ampiThreadLevel) =
      (required == MPI_THREAD_SINGLE) ? MPI_THREAD_SINGLE : MPI_THREAD_FUNNELED;
  *provided = CkpvAccess(ampiThreadLevel);
  co_return co_await Init(p_argc, p_argv);
}

task<int> Finalize() {
  // MPI_Finalize except AMPI_Exit: the rank exits when its main returns.
  int ret = ampiFinalizeBody();
  current()->finalized = true;
  co_return ret;
}

} // namespace campi

#endif // coroutine support
