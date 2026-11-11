/*
 * CkDyn acceptance test, driven by the C++ test binding (cbinding.C).
 * TestMain is an ordinary main chare; every phase ends in a binding future
 * whose callback checks the result and calls mainProxy.nextPhase().
 * Run: ./ckdyntest +pe 4 +balancer RotateLB -requiremoves [-skip <phase>]... [-ckptdir <dir>]
 * (-requiremoves: the sections/migration phase checks that elements moved;
 * -stalesend: the sections/delete phase sends to the deleted section, which must abort;
 * -badlatereg: CkDynRegisterClass after start-up, which must abort)
 * The checkpoint phase writes <dir> (default ckdyntest.ckpt); then
 *      ./ckdyntest +pe 4 +restart <dir>
 * checks that the checkpoint callback reaches the restored main chare DMain.
 */
#include "ckdyntest.decl.h"
#include "cbinding.h"
#if CMK_CUDA || CMK_HIP
#include "hapi.h"
#endif
#include <atomic>
#include <cmath>
#include <cstring>
#include <mutex>
#include <deque>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

/* readonly */ CProxy_TestMain mainProxy;
/* readonly */ int aidPing;
/* readonly */ int gidGrp;
/* readonly */ int gidNGrp;
/* readonly */ int numElems;
/* readonly */ int aidSecA = -1;   /* section test arrays (-1 when the sections phases are skipped) */
/* readonly */ int aidSecB = -1;
/* readonly */ int nSecA;
/* readonly */ int nSecB;

#define CHECK(cond, ...) do { if (!(cond)) cbFail(__VA_ARGS__); } while (0)

static const int PINGPONG_ROUNDS = 10000;
static const int NG_MSGS = 50;
static const int VEC_LEN = 64;
static const int CK_QUEUEING_IFIFO_VALUE = 4;   /* CQS_QUEUEING_IFIFO in converse.h */

static int redType(const char *name) {
  const CkDynReducerInfo *t;
  int n = CkDynReducers(&t);
  for (int i = 0; i < n; i++) if (strcmp(t[i].name, name) == 0) return t[i].type;
  cbFail("no reducer named %s", name);
  return -1;
}

static double expectedSum() { double m = numElems; return m * (m + 1) / 2; }
static CkDynMsg *emptyMsg() { return MsgB().finish(0); }

static void contributeInts(Obj *o, const int *v, int n, const CkDynCallback &cb) {
  CkDynRedData d = { v, (uint32_t)(n * sizeof(int)), redType("sum_int") };
  CkDynContribute(&o->t, &d, &cb);
}
static void contributeDoubles(Obj *o, const double *v, int n, const CkDynCallback &cb) {
  CkDynRedData d = { v, (uint32_t)(n * sizeof(double)), redType("sum_double") };
  CkDynContribute(&o->t, &d, &cb);
}
static double redDouble(const CkDynMsgView *v, int i) {
  uint32_t len; const double *p = (const double *)cbRedData(v, &len);
  CHECK(len >= (i + 1) * sizeof(double), "reduction result has %u bytes", len);
  return p[i];
}
static int redInt(const CkDynMsgView *v, int i) {
  uint32_t len; const int *p = (const int *)cbRedData(v, &len);
  CHECK(len >= (i + 1) * sizeof(int), "reduction result has %u bytes", len);
  return p[i];
}
static CkDynCallback viewCb(const CkDynMsgView *v, int slot) { CkDynCallback c; CkDynViewCallback(v, slot, &c); return c; }

/* ====================================================================== */
/* Classes                                                                 */
/* ====================================================================== */

extern CbClass Ping, Sec, Grp, NGrp, Single, DMain, Late, Tmp, Mig;

enum { P_CTOR, P_RECV, P_RECVI, P_STARTPING, P_COUNT, P_DORED, P_FOREIGN, P_GOTRED, P_GOTREDB, P_COMPUTE, P_CORO, P_SYNCSTART, P_RESUMED,
       P_ZSTART, P_ZRECV, P_ZPOST, P_ZSPIN, P_ZACK, P_DSTART, P_DRECV, P_DCHECK,
       P_PSTART, P_PRECV, P_PDONE, P_PCHECK };
enum { SC_CTOR, SC_RECV, SC_GOTELEM, SC_GOTBCAST, SC_GOTSEC, SC_SYNCSTART, SC_RESUMED, SC_REDTOSEC, SC_SINK, SC_REDNO };
enum { G_CTOR, G_GOTRED, G_GOTREDB, G_NGSEND, G_NGREPORT, G_PRIOSTART, G_PRIORECV, G_EXPRECV,
       G_SECRECV, G_GOTELEM, G_GOTSEC, G_SECCREATE, G_LATEREG, G_CSREPORT,
       G_CBRECV, G_CBCOUNT, G_REDTOSEC, G_SENDDELETE };
enum { N_CTOR, N_INCR };
enum { S_CTOR, S_GOTRED, S_INL, S_PLAIN };
enum { D_CTOR, D_RESUMED };
enum { L_CTOR, L_ADD, L_TWICE };
enum { T_CTOR, T_DESTROY };

static inline int myIndex(Obj *o) { return o->t.u.array.index[0]; }
static void sendPing(int i, int m, CkDynMsg *msg) { CkDynSendArray(aidPing, 1, &i, Ping.ep(m), msg, nullptr); }

/* ---- Ping (1-D array) ---------------------------------------------------- */

/* The target CkDyn passes to construct (and unpack) is zero-filled, so it
   compares byte-equal with the one a binding builds (cbArrayTarget1 and
   friends memset theirs). */
static void checkTargetBytes(const Obj *o, const CkDynTarget &expect, const char *who) {
  CHECK(memcmp(&o->t, &expect, sizeof expect) == 0, "%s: the target CkDyn built differs in its bytes from the "
        "binding-built one (padding or unused index entries not zero)", who);
}

static int64_t pingCtor(Obj *o, const CkDynMsgView *v) {
  CHECK(v->hdr && (v->hdr->flags & CKDYN_MSG_CREATION), "Ping constructor message lost its flags");
  checkTargetBytes(o, cbArrayTarget1(o->t.u.array.aid, Ping.chareIdx, myIndex(o)), "Ping constructor");
  return 0;
}

static void pingFinish(Obj *o, int64_t pe1, int64_t syncCount) {
  double el = CkWallTimer() - o->s.t0;
  CkDynFutureSet(0, (uint32_t)o->s.fid,
                 MsgB().i64(o->s.replies).f64(el).i64(CkMyPe()).i64(pe1).i64(syncCount).finish(0));
}

/* recv(i, senderPe): plain ping-pong between elements 0 and 1 */
static int64_t pingRecv(Obj *o, const CkDynMsgView *v) {
  int64_t i = cbI64(v, 0);
  if (myIndex(o) == 1) { sendPing(0, P_RECV, MsgB().i64(i).i64(CkMyPe()).finish(0)); return 0; }
  o->s.replies = i;
  if (i == o->s.n) pingFinish(o, cbI64(v, 1), 0);
  else sendPing(1, P_RECV, MsgB().i64(i + 1).i64(CkMyPe()).finish(0));
  return 0;
}

/* recvI(i, senderPe): the INLINE entry; element 0 drives a loop, so the
 * inline recursion is two levels deep, not 2*rounds */
static int64_t pingRecvI(Obj *o, const CkDynMsgView *v) {
  int64_t i = cbI64(v, 0);
  if (myIndex(o) == 1) { sendPing(0, P_RECVI, MsgB().i64(i).i64(CkMyPe()).finish(0)); return 0; }
  o->s.replies++;
  o->s.counter = cbI64(v, 1);   /* PE of element 1 */
  return 0;
}

static int64_t pingStart(Obj *o, const CkDynMsgView *v) {
  CHECK(myIndex(o) == 0, "startPing on element %d", myIndex(o));
  o->s.n = cbI64(v, 0);
  o->s.mode = cbI64(v, 1);
  int pe; uint32_t id;
  CHECK(cbFindFuture(v, &pe, &id), "startPing without a future");
  o->s.fid = id;
  o->s.replies = 0;
  o->s.t0 = CkWallTimer();
  if (o->s.mode == 0) {
    sendPing(1, P_RECV, MsgB().i64(1).i64(CkMyPe()).finish(0));
    return 0;
  }
  int64_t sync = 0;
  for (int64_t i = 0; i < o->s.n; i++) {
    sendPing(1, P_RECVI, MsgB().i64(i).i64(CkMyPe()).finish(0));
    if (o->s.replies == i + 1) sync++;
  }
  if (sync == o->s.n) pingFinish(o, o->s.counter, sync);
  else CkDynFutureSet(0, (uint32_t)o->s.fid, MsgB().i64(-1).f64(0).i64(CkMyPe()).i64(-1).i64(sync).finish(0));
  return 0;
}

/* count(value, cb): array broadcast through deliver_many */
static int64_t pingCount(Obj *o, const CkDynMsgView *v) {
  int64_t val = cbI64(v, 0);
  o->s.counter += val;
  int calls = (int)cbDeliverManyCalls(Ping.ep(P_COUNT));
  int r[4] = { (int)val, cbInDeliverMany() ? 1 : 0, calls == 1 ? 1 : 0, (int)o->s.counter };
  contributeInts(o, r, 4, viewCb(v, 1));
  return 0;
}

/* doRed(mode, cb): mode 0 sum_double of index+1; mode 1 the foreign reducer */
static int64_t pingDoRed(Obj *o, const CkDynMsgView *v) {
  int64_t mode = cbI64(v, 0);
  CkDynCallback cb = viewCb(v, 1);
  if (mode == 0) {
    double x = myIndex(o) + 1.0;
    contributeDoubles(o, &x, 1, cb);
  } else {
    int i = myIndex(o);
    sendPing(i, P_FOREIGN, MsgB().user(CB_REDUCER_SUM_I64).i64(i + 1).cb(cb).finish(0));
  }
  return 0;
}

/* foreignContrib: contribute this message's own payload (header to tail) */
static int64_t pingForeign(Obj *o, const CkDynMsgView *v) {
  CHECK(v->hdr->user_id == CB_REDUCER_SUM_I64, "foreign contribution lost its user_id");
  CkDynRedData d = { v->hdr, cbPayloadLen(v), CkDynForeignReducerType() };
  CkDynCallback cb = viewCb(v, 1);
  CkDynContribute(&o->t, &d, &cb);
  return 0;
}

static void checkSumRed(const char *who, const CkDynMsgView *v) {
  CHECK(v->kind == CKDYN_MSGKIND_REDRESULT, "%s: view kind %d, expected REDRESULT", who, (int)v->kind);
  CHECK(v->slots[0].elem == CKDYN_ELEM_F64, "%s: BUFFER elem %d, expected F64", who, v->slots[0].elem);
  double got = redDouble(v, 0);
  CHECK(got == expectedSum(), "%s: sum %g, expected %g", who, got, expectedSum());
  CHECK(v->refnum != 0, "%s: no reference number", who);
  int pe; uint32_t id;
  CHECK(cbFindFuture(v, &pe, &id) && id == v->refnum && pe == CkMyPe(), "%s: FUTURE slot missing or wrong", who);
}

static int64_t gotRedTo0(Obj *, const CkDynMsgView *v, const char *who) {
  checkSumRed(who, v);
  CkDynFutureSet(0, v->refnum, emptyMsg());
  return 0;
}
static int64_t gotRedBContribute(Obj *o, const CkDynMsgView *v, const char *who) {
  checkSumRed(who, v);
  o->s.counter++;
  int one = 1;
  contributeInts(o, &one, 1, cbFutureCb(0, v->refnum));
  return 0;
}

static int64_t pingGotRed(Obj *o, const CkDynMsgView *v) { return gotRedTo0(o, v, "Ping.gotRed"); }
static int64_t pingGotRedB(Obj *o, const CkDynMsgView *v) { return gotRedBContribute(o, v, "Ping.gotRedB"); }
static int64_t pingCompute(Obj *, const CkDynMsgView *v) { return 2 * cbI64(v, 0); }

/* coro(cb): a fake coroutine. The element sets a future on its own PE; the
 * future's function runs inside CkDynMgr.deposit, where a binding resumes a
 * coroutine, and brackets its body with CkDynCoroResume/Suspend. During the
 * body the element must be the running object (so the load balancer charges
 * the time to it); after Suspend the deposit context must be back. The same
 * is checked for a group branch. */
static int64_t pingCoro(Obj *o, const CkDynMsgView *v) {
  CkDynCallback cb = viewCb(v, 0);
  CkGroupID ag; ag.idx = aidPing;
  Chare *self = CProxyElement_ArrayBase(CkArrayID(ag), CkArrayIndex1D(myIndex(o))).ckLocal();
  CHECK(self != nullptr, "coro: element %d is not local", myIndex(o));
  uint32_t f = cbNewFuture([o, self, cb](const CkDynMsgView *) {
    Chare *outer = CkActiveObj();
    CHECK(outer != self, "coro: the element is the running object inside CkDynMgr.deposit");
    CkDynCoroResume(&o->t, Ping.ep(P_CORO), 1000 + myIndex(o));
    CHECK(CkActiveObj() == self, "coro: element %d is not the running object after CkDynCoroResume", myIndex(o));
    double t0 = CkWallTimer();
    while (CkWallTimer() - t0 < 1e-4) {}
    CkDynCoroSuspend(&o->t);
    CHECK(CkActiveObj() == outer, "coro: the running object was not restored by CkDynCoroSuspend");
    CkDynTarget g = cbGroupTarget(gidGrp, Grp.chareIdx, CkMyPe());
    CkGroupID gid; gid.idx = gidGrp;
    CkDynCoroResume(&g, Grp.ep(G_GOTRED), 7);
    CHECK(CkActiveObj() == (Chare *)CkLocalBranch(gid), "coro: the Grp branch is not the running object");
    CkDynCoroSuspend(&g);
    CHECK(CkActiveObj() == outer, "coro: the running object was not restored after the group bracket");
    int one = 1;
    contributeInts(o, &one, 1, cb);
  });
  CkDynFutureSet(CkMyPe(), f, emptyMsg());
  return 0;
}

static int64_t pingSyncStart(Obj *o, const CkDynMsgView *v) {
  int pe; uint32_t id;
  CHECK(cbFindFuture(v, &pe, &id), "syncStart without a future");
  o->s.fid = id;
  o->s.origPe = CkMyPe();
  o->vec.resize(VEC_LEN);
  for (int k = 0; k < VEC_LEN; k++) o->vec[k] = myIndex(o) * 100.0 + k;
  CkDynAtSync(&o->t);
  return 0;
}

/* resumed: sent by the binding's resume_from_sync hook */
static int64_t pingResumed(Obj *o, const CkDynMsgView *) {
  double s = 0;
  for (double x : o->vec) s += x;
  double r[3] = { s, o->s.origPe != CkMyPe() ? 1.0 : 0.0, 1.0 };
  contributeDoubles(o, r, 3, cbFutureCb(0, (uint32_t)o->s.fid));
  return 0;
}

/* ---- Ping: zero-copy (the ncpy phases) ------------------------------------
 * zStart(mode, target, FUTURE) on element 0 sends a 16 MB float64 buffer as
 * an NCPY slot to element `target`: zRecv(mode, replyTo, NCPY data, seed).
 * The target checks the data and echoes it back with replyTo -1. Element 0
 * reports to the future after three events (zAck): the echo, its own done
 * function, and the target's done function. Modes: 0 non-post (the target
 * retains the view and echoes from the retained pointer; its done function
 * releases the message), 1 post before the send (zPost on the target posts,
 * then starts element 0), 2 post after the send has been issued. */
static const int64_t ZN = 2 * 1024 * 1024;   /* doubles: 16 MB */
static const int ZSLOT = 2;                 /* the NCPY slot's index in a zRecv message */

static std::mutex g_zLock;
static std::map<int, std::vector<double> > g_zPost;   /* posted buffer per element index */
static std::atomic<int> g_zSpinB(0);                   /* post_after within one process */

static double zVal(int64_t seed, int64_t i) { return (double)seed + 0.5 * (double)i; }

static double *zPostBuf(int idx) {
  std::lock_guard<std::mutex> g(g_zLock);
  std::vector<double> &b = g_zPost[idx];
  b.assign(ZN, -1.0);
  return b.data();
}

/* done records: called exactly once, on the PE that sent */
struct ZDone { int pe; int calls; void *relMsg; double *freeBuf; int ackTo; };
static void zDoneFn(void *arg) {
  ZDone *d = (ZDone *)arg;
  CHECK(d->pe == CkMyPe(), "ncpy: done ran on PE %d, the buffer was sent from PE %d", CkMyPe(), d->pe);
  CHECK(++d->calls == 1, "ncpy: done called %d times for one send", d->calls);
  if (d->relMsg) CkDynMsgRelease(d->relMsg);   /* the retained message the echo was sent from */
  delete[] d->freeBuf;
  d->relMsg = nullptr; d->freeBuf = nullptr;
  if (d->ackTo >= 0) sendPing(d->ackTo, P_ZACK, MsgB().i64(0).finish(0));
  /* d is kept so that a second call is detected */
}

static void zSend(int to, int64_t mode, int64_t replyTo, const double *data, int64_t seed,
                  void *relMsg, double *freeBuf, int ackTo) {
  static const int64_t shape[1] = { ZN };
  CkDynBufDesc d = { data, (uint64_t)(ZN * sizeof(double)), CKDYN_ELEM_F64, 1, 0, 0, shape, nullptr };
  ZDone *rec = new ZDone{ CkMyPe(), 0, relMsg, freeBuf, ackTo };
  MsgB b;
  b.i64(mode).i64(replyTo);
  CkDynMsgAddNcpy(b.b, &d, zDoneFn, rec);
  sendPing(to, P_ZRECV, b.i64(seed).finish(0));
}

static void zMaybeReport(Obj *o) {
  if (o->s.counter < 3) return;
  CkDynFutureSet(0, (uint32_t)o->s.fid, MsgB().i64(o->s.replies).f64(CkWallTimer() - o->s.t0).finish(0));
}

static void zPostFor(int idx) {
  CkDynTarget t = cbArrayTarget1(aidPing, Ping.chareIdx, idx);
  CkDynBufDesc d = { zPostBuf(idx), (uint64_t)(ZN * sizeof(double)), CKDYN_ELEM_F64, 0, 0, 0, nullptr, nullptr };
  CkDynPostBuffer(&t, Ping.ep(P_ZRECV), ZSLOT, &d);
}

/* zStart(mode, target, FUTURE) */
static int64_t pingZStart(Obj *o, const CkDynMsgView *v) {
  int64_t mode = cbI64(v, 0);
  int target = (int)cbI64(v, 1);
  int pe; uint32_t id;
  CHECK(cbFindFuture(v, &pe, &id), "zStart without a future");
  o->s.fid = id; o->s.counter = 0; o->s.mode = mode; o->s.replies = 0;
  int64_t seed = 1000 * mode + 7;
  double *src = new double[ZN];   /* freed by the done function */
  for (int64_t i = 0; i < ZN; i++) src[i] = zVal(seed, i);
  o->s.t0 = CkWallTimer();
  zSend(target, mode, myIndex(o), src, seed, nullptr, src, myIndex(o));
  if (mode == 2) {
    if (CkNumPes() == 1) {
      zPostFor(target);   /* the target is on this PE: post for it right after the send */
    } else if (CkNodeOf(CkNumPes() - 1) == CkMyNode()) {
      /* the target's PE spins in zSpin until the send is issued, then posts */
      g_zSpinB.store(1);
    } else {
      /* another process: the post request follows the data, and the data
         is delivered only after its transfer */
      sendPing(target, P_ZPOST, MsgB().i64(mode).i64(-1).finish(0));
    }
  }
  return 0;
}

/* zRecv(mode, replyTo, NCPY data, seed) */
static int64_t pingZRecv(Obj *o, const CkDynMsgView *v) {
  int64_t mode = cbI64(v, 0), replyTo = cbI64(v, 1), seed = cbI64(v, 3);
  CHECK(v->hdr->nbuf == 1 && v->slots[ZSLOT].type == CKDYN_SLOT_NCPY, "ncpy: slot %d is type %d, nbuf %u",
        ZSLOT, (int)v->slots[ZSLOT].type, v->hdr->nbuf);
  const int64_t *shape;
  const double *p = (const double *)CkDynViewBufferPtr(v, ZSLOT, &shape);
  CHECK(shape && shape[0] == ZN && v->slots[ZSLOT].len == ZN * sizeof(double) && v->slots[ZSLOT].elem == CKDYN_ELEM_F64,
        "ncpy: slot shape or length lost");
  for (int64_t i = 0; i < ZN; i++)
    if (p[i] != zVal(seed, i)) cbFail("ncpy: element %d received %g at %lld, expected %g", myIndex(o), p[i], (long long)i, zVal(seed, i));
  bool posted = false;
  {
    std::lock_guard<std::mutex> g(g_zLock);
    auto it = g_zPost.find(myIndex(o));
    posted = it != g_zPost.end() && !it->second.empty() && p == it->second.data();
  }
  if (replyTo < 0) {   /* the echo, back at element 0 */
    o->s.replies = posted ? 2 : 1;
    o->s.counter++;
    zMaybeReport(o);
    return 0;
  }
  CHECK(posted == (mode != 0), "ncpy: mode %lld delivered %s the posted buffer", (long long)mode, posted ? "into" : "outside");
  if (mode == 0) {
    /* echo from the retained view after this hook returns; the done function releases it */
    CkDynMsgRetain(v);
    zSend((int)replyTo, mode, -1, p, seed, v->msg, nullptr, (int)replyTo);
  } else {
    zSend((int)replyTo, mode, -1, p, seed, nullptr, nullptr, (int)replyTo);
  }
  return 0;
}

/* zPost(mode, startFuture): post a buffer for zRecv's NCPY slot; with a
   future id >= 0, then start element 0 (post before the send) */
static int64_t pingZPost(Obj *o, const CkDynMsgView *v) {
  int64_t mode = cbI64(v, 0), fut = cbI64(v, 1);
  zPostFor(myIndex(o));
  if (fut >= 0) sendPing(0, P_ZSTART, MsgB().i64(mode).i64(myIndex(o)).fut(0, (uint32_t)fut).finish(0));
  return 0;
}

/* zSpin(fut): post after the send within one process. Start element 0, spin
   until it has issued the send (its message cannot be processed on this PE
   meanwhile), post, return. */
static int64_t pingZSpin(Obj *o, const CkDynMsgView *v) {
  int64_t fut = cbI64(v, 0);
  CHECK(CkMyPe() == CkNumPes() - 1 && CkMyPe() != 0, "zSpin on PE %d", CkMyPe());
  g_zSpinB.store(0);
  sendPing(0, P_ZSTART, MsgB().i64(2).i64(myIndex(o)).fut(0, (uint32_t)fut).finish(0));
  while (g_zSpinB.load() == 0) {}
  zPostFor(myIndex(o));
  return 0;
}

static int64_t pingZAck(Obj *o, const CkDynMsgView *) { o->s.counter++; zMaybeReport(o); return 0; }

/* ---- Ping: device buffers (the device phases; CUDA or HIP builds) --------
 * dStart(target, FUTURE) on element 0 copies a 16 MB pattern to device memory
 * and sends it as a DEVICE slot: dRecv(replyTo, DEVICE data, seed). The
 * target copies it back to the host and checks it, then sends itself
 * dCheck(ptr, len), which runs after the delivery's memory was released and
 * checks that the pool hands out the same block again. Element 0 reports
 * after its done function, the data check and the pool check (zAck). */
#if CMK_CUDA || CMK_HIP
static const int DSLOT = 1;

static void dDoneFn(void *arg) {
  ZDone *d = (ZDone *)arg;
  CHECK(d->pe == CkMyPe(), "device: done ran on PE %d, the buffer was sent from PE %d", CkMyPe(), d->pe);
  CHECK(++d->calls == 1, "device: done called %d times for one send", d->calls);
  hapiCheck(hapiFree(d->relMsg));
  d->relMsg = nullptr;
  sendPing(d->ackTo, P_ZACK, MsgB().i64(0).finish(0));
}

static int64_t pingDStart(Obj *o, const CkDynMsgView *v) {
  int target = (int)cbI64(v, 0);
  int pe; uint32_t id;
  CHECK(cbFindFuture(v, &pe, &id), "dStart without a future");
  o->s.fid = id; o->s.counter = 0; o->s.replies = 1;
  int64_t seed = 77 + target;
  std::vector<double> h(ZN);
  for (int64_t i = 0; i < ZN; i++) h[i] = zVal(seed, i);
  void *dsrc = nullptr;
  hapiCheck(hapiMalloc(&dsrc, ZN * sizeof(double)));
  hapiCheck(hapiMemcpy(dsrc, h.data(), ZN * sizeof(double), hapiMemcpyHostToDevice));
  int dev = 0;
  hapiCheck(hapiGetDevice(&dev));
  static const int64_t shape[1] = { ZN };
  CkDynDevDesc d = { dsrc, (uint64_t)(ZN * sizeof(double)), CKDYN_ELEM_F64, 1, 0, dev, 0, shape };
  ZDone *rec = new ZDone{ CkMyPe(), 0, dsrc, nullptr, myIndex(o) };
  o->s.t0 = CkWallTimer();
  MsgB b;
  b.i64(myIndex(o));
  CkDynMsgAddDevice(b.b, &d, dDoneFn, rec);
  sendPing(target, P_DRECV, b.i64(seed).finish(0));
  return 0;
}

static int64_t pingDRecv(Obj *o, const CkDynMsgView *v) {
  int replyTo = (int)cbI64(v, 0);
  int64_t seed = cbI64(v, 2);
  const CkDynSlot &sl = v->slots[DSLOT];
  CHECK(v->hdr->nbuf == 1 && sl.type == CKDYN_SLOT_DEVICE && sl.len == ZN * sizeof(double), "device: slot lost its type or length");
  int dev = 0;
  hapiCheck(hapiGetDevice(&dev));
  CHECK(sl.reserved == dev, "device: slot names device %d, this PE uses device %d", (int)sl.reserved, dev);
  const int64_t *shape;
  const void *dp = CkDynViewBufferPtr(v, DSLOT, &shape);
  CHECK(shape && shape[0] == ZN, "device: shape lost");
  std::vector<double> h(ZN, -1.0);
  hapiCheck(hapiMemcpy(h.data(), dp, ZN * sizeof(double), hapiMemcpyDeviceToHost));
  for (int64_t i = 0; i < ZN; i++)
    if (h[i] != zVal(seed, i)) cbFail("device: element %d received %g at %lld, expected %g", myIndex(o), h[i], (long long)i, zVal(seed, i));
  sendPing(replyTo, P_ZACK, MsgB().i64(0).finish(0));
  sendPing(myIndex(o), P_DCHECK, MsgB().i64((int64_t)(uintptr_t)dp).i64(replyTo).finish(0));
  return 0;
}

/* dCheck(ptr, replyTo): the delivery's pool block was freed when its message was */
static int64_t pingDCheck(Obj *, const CkDynMsgView *v) {
  void *was = (void *)(uintptr_t)cbI64(v, 0);
  void *again = CkDeviceMalloc(ZN * sizeof(double));
  CHECK(again == was, "device: the received buffer's pool block %p was not free again after the delivery (got %p)", was, again);
  CkDeviceFree(again);
  sendPing((int)cbI64(v, 1), P_ZACK, MsgB().i64(0).finish(0));
  return 0;
}

/* ---- Ping: persistent device buffers (the devpersist phases) -------------
 * pStart(target, FUTURE) on element 0 fills a 16 MB device buffer A with
 * pattern seed1, opens it and sends its handle: pRecv(handle, round 1,
 * seed1). The target opens its own buffer B, gets A into B (pDone refnum 1),
 * checks seed1, writes seed2 into B and puts B back into A (pDone refnum 2),
 * then tells element 0: pCheck(round 1, seed2, mode). Element 0 checks A,
 * closes A's buffer and opens it again, and sends the new handle: pRecv(handle,
 * round 2, seed2); the target gets again, checks seed2, closes and frees B,
 * and sends pCheck(round 2); element 0 closes, frees A and reports. Both
 * done callbacks are TARGET callbacks to pDone with the call's refnum. */
struct DpState { void *buf = nullptr; CkDynDevPersist *p = nullptr; std::vector<char> remote; int64_t seed = 0, round = 0; int replyTo = 0; uint32_t fid = 0; double t0 = 0, tGet = 0; };
static std::mutex g_dpLock;
static std::map<int, DpState> g_dp;   /* per element index */
static DpState &dpState(int idx) { std::lock_guard<std::mutex> g(g_dpLock); return g_dp[idx]; }

static void dpFill(void *dbuf, int64_t seed) {
  std::vector<double> h(ZN);
  for (int64_t i = 0; i < ZN; i++) h[i] = zVal(seed, i);
  hapiCheck(hapiMemcpy(dbuf, h.data(), ZN * sizeof(double), hapiMemcpyHostToDevice));
}
static void dpCheck(Obj *o, const void *dbuf, int64_t seed, const char *what) {
  std::vector<double> h(ZN, -1.0);
  hapiCheck(hapiMemcpy(h.data(), dbuf, ZN * sizeof(double), hapiMemcpyDeviceToHost));
  for (int64_t i = 0; i < ZN; i++)
    if (h[i] != zVal(seed, i)) cbFail("devpersist: %s on element %d: %g at %lld, expected %g", what, myIndex(o), h[i], (long long)i, zVal(seed, i));
}
static CkDynDevPersist *dpOpen(Obj *o, void *dbuf) {
  int dev = 0;
  hapiCheck(hapiGetDevice(&dev));
  static const int64_t shape[1] = { ZN };
  CkDynDevDesc d = { dbuf, (uint64_t)(ZN * sizeof(double)), CKDYN_ELEM_F64, 1, 0, dev, 0, shape };
  CkDynCallback done = cbTargetCb(o->t, Ping.ep(P_PDONE), 0, 0);
  return CkDynDevPersistOpen(&d, &done);
}
static CkDynMsg *dpHandleMsg(CkDynDevPersist *p, int64_t round, int64_t seed) {
  std::vector<char> h(CkDynDevPersistHandleSize());
  CkDynDevPersistHandle(p, h.data());
  MsgB b;
  CkDynMsgAddBytes(b.b, h.data(), (uint32_t)h.size(), 0);
  return b.i64(round).i64(seed).finish(0);
}

static int64_t pingPStart(Obj *o, const CkDynMsgView *v) {
  int target = (int)cbI64(v, 0);
  int pe; uint32_t id;
  CHECK(cbFindFuture(v, &pe, &id), "pStart without a future");
  DpState &st = dpState(myIndex(o));
  st.fid = id; st.replyTo = target; st.seed = 501 + target;
  hapiCheck(hapiMalloc(&st.buf, ZN * sizeof(double)));
  dpFill(st.buf, st.seed);
  st.p = dpOpen(o, st.buf);
  st.t0 = CkWallTimer();
  sendPing(target, P_PRECV, dpHandleMsg(st.p, 1, st.seed));
  return 0;
}

/* pRecv(handle, round, seed) on the target */
static int64_t pingPRecv(Obj *o, const CkDynMsgView *v) {
  CHECK(v->slots[0].type == CKDYN_SLOT_BYTES && v->slots[0].len == CkDynDevPersistHandleSize(),
        "devpersist: handle slot of %u bytes, expected %zu", v->slots[0].len, CkDynDevPersistHandleSize());
  DpState &st = dpState(myIndex(o));
  const char *h = (const char *)CkDynViewBytesPtr(v, 0);
  st.remote.assign(h, h + v->slots[0].len);
  st.round = cbI64(v, 1); st.seed = cbI64(v, 2);
  if (!st.p) {
    hapiCheck(hapiMalloc(&st.buf, ZN * sizeof(double)));
    hapiCheck(hapiMemset(st.buf, 0, ZN * sizeof(double)));
    st.p = dpOpen(o, st.buf);
  }
  st.t0 = CkWallTimer();
  CkDynDevPersistGet(st.p, st.remote.data(), 1);
  return 0;
}

/* pDone: refnum 1 a get completed, 2 a put completed (on the issuing PE) */
static int64_t pingPDone(Obj *o, const CkDynMsgView *v) {
  DpState &st = dpState(myIndex(o));
  int mode = CkDynDevPersistMode(st.p, st.remote.data());
  if (v->refnum == 1) {
    st.tGet = CkWallTimer() - st.t0;
    dpCheck(o, st.buf, st.seed, "get");
    if (st.round == 1) {
      dpFill(st.buf, st.seed + 1);
      CkDynDevPersistPut(st.remote.data(), st.p, 2);
    } else {
      CkDynDevPersistClose(st.p);
      hapiCheck(hapiFree(st.buf));
      st.p = nullptr; st.buf = nullptr;
      sendPing(0, P_PCHECK, MsgB().i64(2).i64(st.seed).i64(mode).f64(st.tGet).finish(0));
    }
  } else if (v->refnum == 2) {
    sendPing(0, P_PCHECK, MsgB().i64(1).i64(st.seed + 1).i64(mode).f64(st.tGet).finish(0));
  } else {
    cbFail("devpersist: done with refnum %u", v->refnum);
  }
  return 0;
}

/* pCheck(round, seed, mode, getTime) on element 0 */
static int64_t pingPCheck(Obj *o, const CkDynMsgView *v) {
  DpState &st = dpState(myIndex(o));
  int64_t round = cbI64(v, 0), seed = cbI64(v, 1), mode = cbI64(v, 2);
  dpCheck(o, st.buf, seed, round == 1 ? "put" : "after the round-2 get");
  CkDynDevPersistClose(st.p);
  st.p = nullptr;
  if (round == 1) {
    st.p = dpOpen(o, st.buf);   /* close and reopen: a new handle for the same memory */
    sendPing(st.replyTo, P_PRECV, dpHandleMsg(st.p, 2, seed));
    return 0;
  }
  hapiCheck(hapiFree(st.buf));
  st.buf = nullptr;
  CkDynFutureSet(0, st.fid, MsgB().i64(mode).f64(cbF64(v, 3)).f64(CkWallTimer() - st.t0).finish(0));
  return 0;
}
#else
static int64_t pingPStart(Obj *, const CkDynMsgView *) { cbFail("devpersist phase in a build without CUDA or HIP"); return 0; }
static int64_t pingPRecv(Obj *, const CkDynMsgView *) { return 0; }
static int64_t pingPDone(Obj *, const CkDynMsgView *) { return 0; }
static int64_t pingPCheck(Obj *, const CkDynMsgView *) { return 0; }
static int64_t pingDStart(Obj *, const CkDynMsgView *) { cbFail("device phase in a build without CUDA or HIP"); return 0; }
static int64_t pingDRecv(Obj *, const CkDynMsgView *) { return 0; }
static int64_t pingDCheck(Obj *, const CkDynMsgView *) { return 0; }
#endif

CbClass Ping = { "Ping", CKDYN_ARRAY, 1, {
  { "Ping.Ping", 0, pingCtor },
  { "Ping.recv", 0, pingRecv },
  { "Ping.recvI", CKDYN_EP_INLINE, pingRecvI },
  { "Ping.startPing", 0, pingStart },
  { "Ping.count", 0, pingCount },
  { "Ping.doRed", 0, pingDoRed },
  { "Ping.foreignContrib", 0, pingForeign },
  { "Ping.gotRed", CKDYN_EP_REDNTARGET, pingGotRed },
  { "Ping.gotRedB", CKDYN_EP_REDNTARGET, pingGotRedB },
  { "Ping.compute", 0, pingCompute },
  { "Ping.coro", 0, pingCoro },
  { "Ping.syncStart", 0, pingSyncStart },
  { "Ping.resumed", 0, pingResumed },
  { "Ping.zStart", 0, pingZStart },
  { "Ping.zRecv", 0, pingZRecv },
  { "Ping.zPost", 0, pingZPost },
  { "Ping.zSpin", 0, pingZSpin },
  { "Ping.zAck", 0, pingZAck },
  { "Ping.dStart", 0, pingDStart },
  { "Ping.dRecv", 0, pingDRecv },
  { "Ping.dCheck", 0, pingDCheck },
  { "Ping.pStart", 0, pingPStart },
  { "Ping.pRecv", 0, pingPRecv },
  { "Ping.pDone", 0, pingPDone },
  { "Ping.pCheck", 0, pingPCheck },
} };

/* ---- section members (Sec array elements and Grp branches) --------------- */

/* The value a member contributes: Sec element i of array A gives i+1, of
   array B 1000*(i+1); a Grp branch gives its PE + 1. */
static double memberValue(Obj *o) {
  if (o->t.kind == CKDYN_GROUP) return o->t.u.group.pe_or_node + 1;
  return (o->t.u.array.aid == aidSecA ? 1.0 : 1000.0) * (o->t.u.array.index[0] + 1);
}

/* secRecv(mode, expectRedNo, cb): a section multicast. Checks the view and
 * the member's section reduction number, then contributes to the section:
 * mode 0 sum_double of [value, 1]; mode 1 the foreign reducer over value. */
static int64_t secRecv(Obj *o, const CkDynMsgView *v) {
  CHECK(v->kind == CKDYN_MSGKIND_MCAST && v->section != nullptr, "secRecv: view kind %d, section %p", (int)v->kind, (const void *)v->section);
  CHECK(o->s.joined > 0, "secRecv: the member never got section_joined");
  CkDynSectionID sid = *v->section;
  int64_t mode = cbI64(v, 0), expect = cbI64(v, 1);
  CkDynCallback cb = viewCb(v, 2);
  int rn = CkDynSectionRedNo(&o->t, &sid);
  CHECK(expect < 0 || rn == expect, "secRecv: section (root %d, idx %d) reduction number %d, expected %lld",
        sid.root_pe, sid.idx, rn, (long long)expect);
  o->s.counter++;
  double val = memberValue(o);
  if (mode == 0) {
    double r[2] = { val, 1.0 };
    CkDynRedData d = { r, sizeof r, redType("sum_double") };
    CkDynContributeSection(&o->t, &sid, &d, &cb);
  } else {
    CkDynMsg *m = MsgB().user(CB_REDUCER_SUM_I64).i64((int64_t)val).finish(0);
    uint32_t len;
    const void *p = CkDynMsgPayload(m, &len);
    CkDynRedData d = { p, len, CkDynForeignReducerType() };
    CkDynContributeSection(&o->t, &sid, &d, &cb);
    CkDynMsgFree(m);
  }
  return 0;
}

/* The (value, count) of a section reduction result in any of its forms: a
   REDRESULT or a section callback's BUFFER slot of doubles (count is the
   second double), or a foreign result's I64 slot (count -1). */
static void resultOf(const CkDynMsgView *v, double *value, double *count) {
  CHECK(v->nslots >= 1, "result view without slots (kind %d)", (int)v->kind);
  if (v->slots[0].type == CKDYN_SLOT_I64) {
    cbCheckReduceBuffer(v);
    *value = (double)v->slots[0].v.i64; *count = -1;
    return;
  }
  CHECK(v->slots[0].type == CKDYN_SLOT_BUFFER, "result slot 0 has type %d", v->slots[0].type);
  if (v->slots[0].elem == CKDYN_ELEM_BYTE) {
    /* a set reduction of (value, count) pairs: sum them */
    *value = 0; *count = 0;
    const CkReduction::setElement *e = (const CkReduction::setElement *)CkDynViewBufferPtr(v, 0, nullptr);
    for (; e != nullptr; e = const_cast<CkReduction::setElement *>(e)->next()) {
      CHECK(e->dataSize == 2 * sizeof(double), "set result element of %d bytes", e->dataSize);
      const double *d = (const double *)e->data;
      *value += d[0]; *count += d[1];
    }
    return;
  }
  CHECK(v->slots[0].elem == CKDYN_ELEM_F64, "result buffer elem %d, expected F64", v->slots[0].elem);
  const double *p = (const double *)CkDynViewBufferPtr(v, 0, nullptr);
  *value = p[0];
  *count = v->slots[0].len >= 16 ? p[1] : -1;
}

/* gotElem: a TARGET callback to one member; reports (value, count) to the future */
static int64_t secGotElem(Obj *, const CkDynMsgView *v) {
  CHECK(v->refnum != 0, "gotElem: no reference number");
  double val, cnt;
  resultOf(v, &val, &cnt);
  CkDynFutureSet(0, v->refnum, MsgB().f64(val).f64(cnt).finish(0));
  return 0;
}
/* gotBcast: a TARGET broadcast; every element of the array reduces [value, count, 1] */
static int64_t secGotBcast(Obj *o, const CkDynMsgView *v) {
  CHECK(v->kind == CKDYN_MSGKIND_REDRESULT || v->kind == CKDYN_MSGKIND_DYN, "gotBcast: view kind %d", (int)v->kind);
  double r[3];
  resultOf(v, &r[0], &r[1]);
  r[2] = 1;
  CkDynRedData d = { r, sizeof r, redType("sum_double") };
  CkDynCallback cb = cbFutureCb(0, v->refnum);
  CkDynContribute(&o->t, &d, &cb);
  return 0;
}
/* gotSec: a SECTION callback; every member reduces [value, count, 1, refnum]
 * over the section, so the root can check the refnum each member received */
static int64_t secGotSec(Obj *o, const CkDynMsgView *v) {
  CHECK(v->kind == CKDYN_MSGKIND_MCAST && v->section, "gotSec: view kind %d", (int)v->kind);
  CHECK(v->refnum != 0, "gotSec: the section callback lost its reference number");
  double r[4];
  resultOf(v, &r[0], &r[1]);
  r[2] = 1;
  r[3] = v->refnum;
  CkDynRedData d = { r, sizeof r, redType("sum_double") };
  CkDynCallback cb = cbFutureCb(0, v->refnum);
  CkDynSectionID sid = *v->section;
  CkDynContributeSection(&o->t, &sid, &d, &cb);
  return 0;
}

/* redToSec(mode, cb): a reduction over the whole array or group (not a
 * section) whose callback is the SECTION callback cb: mode 0 sum_double of
 * [value, 1], 1 the foreign reducer over value, 2 the set reducer over
 * [value, 1]. Contributors on different PEs must carry the same Charm++
 * callback (CkDynContribute's section-tagged reducer). */
static int64_t memberRedToSec(Obj *o, const CkDynMsgView *v) {
  int64_t mode = cbI64(v, 0);
  CkDynCallback cb = viewCb(v, 1);
  double r[2] = { memberValue(o), 1.0 };
  if (mode == 1) {
    CkDynMsg *m = MsgB().user(CB_REDUCER_SUM_I64).i64((int64_t)r[0]).finish(0);
    uint32_t len;
    const void *p = CkDynMsgPayload(m, &len);
    CkDynRedData d = { p, len, CkDynForeignReducerType() };
    CkDynContribute(&o->t, &d, &cb);
    CkDynMsgFree(m);
  } else {
    CkDynRedData d = { r, sizeof r, redType(mode == 0 ? "sum_double" : "set") };
    CkDynContribute(&o->t, &d, &cb);
  }
  return 0;
}

/* ---- Sec (1-D array, section and migration tests) ------------------------ */

static int64_t secCtor(Obj *, const CkDynMsgView *) { return 0; }
static int64_t secSyncStart(Obj *o, const CkDynMsgView *v) {
  int pe; uint32_t id;
  CHECK(cbFindFuture(v, &pe, &id), "Sec.syncStart without a future");
  o->s.fid = id;
  o->s.origPe = CkMyPe();
  CkDynAtSync(&o->t);
  return 0;
}
/* sink(n, bytes): a large section message (sections/order); checks its length */
static const uint32_t ORDER_BYTES = 256 * 1024;
static int64_t secSink(Obj *, const CkDynMsgView *v) {
  CHECK(v->kind == CKDYN_MSGKIND_MCAST && v->nslots == 2 && v->slots[1].type == CKDYN_SLOT_BYTES &&
        v->slots[1].len == ORDER_BYTES, "Sec.sink: a malformed section message");
  return 0;
}
/* redNo(SECTION sid, FUTURE f): CkDynSectionRedNo of this member, to f */
static int64_t secRedNo(Obj *o, const CkDynMsgView *v) {
  CkDynSectionID sid;
  CkDynViewSection(v, 0, &sid);
  int pe; uint32_t id;
  CHECK(cbFindFuture(v, &pe, &id), "Sec.redNo without a future");
  int rn = CkDynSectionRedNo(&o->t, &sid);
  CkDynFutureSet(pe, id, MsgB().i64(rn).finish(0));
  return 0;
}
static int64_t secResumed(Obj *o, const CkDynMsgView *) {
  double r[2] = { 1.0, o->s.origPe != CkMyPe() ? 1.0 : 0.0 };
  CkDynRedData d = { r, sizeof r, redType("sum_double") };
  CkDynCallback cb = cbFutureCb(0, (uint32_t)o->s.fid);
  CkDynContribute(&o->t, &d, &cb);
  return 0;
}

CbClass Sec = { "Sec", CKDYN_ARRAY, 1, {
  { "Sec.Sec", 0, secCtor },
  { "Sec.secRecv", 0, secRecv },
  { "Sec.gotElem", CKDYN_EP_REDNTARGET, secGotElem },
  { "Sec.gotBcast", CKDYN_EP_REDNTARGET, secGotBcast },
  { "Sec.gotSec", CKDYN_EP_REDNTARGET, secGotSec },
  { "Sec.syncStart", 0, secSyncStart },
  { "Sec.resumed", 0, secResumed },
  { "Sec.redToSec", 0, memberRedToSec },
  { "Sec.sink", 0, secSink },
  { "Sec.redNo", 0, secRedNo },
} };

/* ---- Grp (group) ---------------------------------------------------------- */

static Obj *volatile g_ngObj = nullptr;   /* this process's NGrp branch (one per process) */

static int64_t grpCtor(Obj *o, const CkDynMsgView *) {
  CHECK(o->t.kind == CKDYN_GROUP && o->t.u.group.pe_or_node == CkMyPe(), "Grp constructed with a wrong target");
  checkTargetBytes(o, cbGroupTarget(o->t.u.group.gid, Grp.chareIdx, CkMyPe()), "Grp constructor");
  return 0;
}
static int64_t grpGotRed(Obj *o, const CkDynMsgView *v) { return gotRedTo0(o, v, "Grp.gotRed"); }
static int64_t grpGotRedB(Obj *o, const CkDynMsgView *v) { return gotRedBContribute(o, v, "Grp.gotRedB"); }

static int64_t grpNgSend(Obj *, const CkDynMsgView *v) {
  int64_t n = cbI64(v, 0);
  for (int64_t i = 0; i < n; i++)
    CkDynSendNodeGroup(gidNGrp, 0, NGrp.ep(N_INCR), MsgB().i64(CkMyPe()).finish(0), nullptr);
  return 0;
}

static int64_t grpNgReport(Obj *o, const CkDynMsgView *v) {
  int c = 0;
  if (CkMyRank() == 0) {
    CHECK(g_ngObj != nullptr, "no NGrp branch in this process");
    c = (int)g_ngObj->s.counter;
  }
  contributeInts(o, &c, 1, viewCb(v, 0));
  return 0;
}

static void grpMaybeReport(Obj *o) {
  if (o->order.size() == 3 && o->s.expGot)
    CkDynFutureSet(0, (uint32_t)o->s.fid, MsgB().i64(o->order[0]).i64(o->order[1]).i64(o->order[2]).finish(0));
}

static int64_t grpPrioStart(Obj *o, const CkDynMsgView *v) {
  int pe; uint32_t id;
  CHECK(cbFindFuture(v, &pe, &id), "prioStart without a future");
  o->s.fid = id;
  o->order.clear();
  o->s.expGot = 0;
  const int prios[3] = { 5, 1, 3 };
  for (int p : prios) {
    CkDynMsg *m = MsgB(8 * sizeof(int)).i64(p).finish(0);
    *(int *)CkDynMsgPriorityPtr(m) = p;
    CkDynMsgSetQueueing(m, CK_QUEUEING_IFIFO_VALUE);
    CkDynSendGroup(gidGrp, CkMyPe(), Grp.ep(G_PRIORECV), m, nullptr);
  }
  CkDynSendOpts ex = { CKDYN_SEND_EXPEDITED };
  CkDynSendGroup(gidGrp, CkMyPe(), Grp.ep(G_EXPRECV), emptyMsg(), &ex);
  return 0;
}
static int64_t grpPrioRecv(Obj *o, const CkDynMsgView *v) { o->order.push_back(cbI64(v, 0)); grpMaybeReport(o); return 0; }
static int64_t grpExpRecv(Obj *o, const CkDynMsgView *) { o->s.expGot = 1; grpMaybeReport(o); return 0; }

/* secCreate(n, FUTURE sidFuture, CALLBACK done): create the section of the
 * first n elements of array A on this PE (its root) and report its id. */
static int64_t grpSecCreate(Obj *, const CkDynMsgView *v) {
  int n = (int)cbI64(v, 0);
  int pe; uint32_t id;
  CHECK(cbFindFuture(v, &pe, &id), "secCreate without a future");
  CkDynCallback done = viewCb(v, 2);
  std::vector<int> idx(n);
  for (int i = 0; i < n; i++) idx[i] = i;
  CkDynSectionID sid;
  CkDynSectionCreateArray(aidSecA, n, 1, idx.data(), 0, &done, &sid);
  CHECK(sid.root_pe == CkMyPe() && sid.kind == CKDYN_ARRAY && sid.coll_id == aidSecA, "secCreate: section id fields");
  CkDynFutureSet(pe, id, MsgB().sec(sid).finish(0));
  return 0;
}

/* sendDelete(n, FUTURE f, SECTION sid x n): from a PE that is not the root of
 * the sections, send a large message to each section and delete it at once,
 * then set f. The root must apply the send before the delete. */
static int64_t grpSendDelete(Obj *, const CkDynMsgView *v) {
  int n = (int)cbI64(v, 0);
  int pe; uint32_t id;
  CHECK(cbFindFuture(v, &pe, &id), "sendDelete without a future");
  std::vector<char> big(ORDER_BYTES, 'x');
  int ep = Sec.ep(SC_SINK);
  for (int i = 0; i < n; i++) {
    CkDynSectionID sid;
    CkDynViewSection(v, 2 + i, &sid);
    CHECK(sid.root_pe != CkMyPe(), "sendDelete on the root PE %d", sid.root_pe);
    MsgB b;
    b.i64(i);
    CkDynMsgAddBytes(b.b, big.data(), ORDER_BYTES, 0);
    CkDynSendSection(&sid, ep, b.finishMcast(ep), nullptr);
    CkDynSectionDelete(&sid);
  }
  CkDynFutureSet(pe, id, emptyMsg());
  return 0;
}

/* lateReg(CALLBACK done): register Late after start-up. The main chare runs
 * this phase only with one PE per process, so each PE registers for its
 * process; the reduction tells it that every process has. */
static int64_t grpLateReg(Obj *o, const CkDynMsgView *v) {
  cbRegisterLate(&Late);
  int r[3] = { 1, Late.chareIdx, Late.firstEp };
  contributeInts(o, r, 3, viewCb(v, 0));
  return 0;
}

/* csReport(cb): contribute the number of Tmp elements that destroyed
   themselves inside a CkDynCallstackPush/Pop bracket on this PE */
static std::mutex g_csLock;
static std::map<int, int> g_csDone;   /* PE -> count */
static int64_t grpCsReport(Obj *o, const CkDynMsgView *v) {
  int c;
  { std::lock_guard<std::mutex> g(g_csLock); c = g_csDone[CkMyPe()]; }
  contributeInts(o, &c, 1, viewCb(v, 0));
  return 0;
}

/* cbRecv: on a member of the constrained group, a broadcast TARGET callback
   (a CkDynMsg, slot 0 the value) or a foreign reduction result (slot 0 the
   sum); the reference number is the future to report to. Reports
   cbCount(future, value) to branch 0 of the full group Grp. */
static int64_t grpCbRecv(Obj *o, const CkDynMsgView *v) {
  CHECK(v->kind == CKDYN_MSGKIND_DYN && v->refnum != 0, "constrained: cbRecv view kind %d, refnum %u", (int)v->kind, v->refnum);
  cbCheckReduceBuffer(v);
  CHECK(o->t.u.group.pe_or_node % 2 == 0, "constrained: cbRecv ran on PE %d, which is not a member", CkMyPe());
  CkDynSendGroup(gidGrp, 0, Grp.ep(G_CBCOUNT), MsgB().i64(v->refnum).i64(cbI64(v, 0)).finish(0), nullptr);
  return 0;
}
/* cbCount(future, value) on Grp branch 0: count the members that reported
   for the future; after the last one, set it to (count, value) */
static std::map<int64_t, std::pair<int, int64_t> > g_cbCount;   /* PE 0 only */
static int64_t g_cbExpect = 0;
static int64_t grpCbCount(Obj *, const CkDynMsgView *v) {
  int64_t f = cbI64(v, 0), val = cbI64(v, 1);
  auto &e = g_cbCount[f];
  CHECK(e.first == 0 || e.second == val, "constrained: members received %lld and %lld", (long long)e.second, (long long)val);
  e.second = val;
  if (++e.first == g_cbExpect) CkDynFutureSet(0, (uint32_t)f, MsgB().i64(e.first).i64(val).finish(0));
  return 0;
}

CbClass Grp = { "Grp", CKDYN_GROUP, 0, {
  { "Grp.Grp", 0, grpCtor },
  { "Grp.gotRed", CKDYN_EP_REDNTARGET, grpGotRed },
  { "Grp.gotRedB", CKDYN_EP_REDNTARGET, grpGotRedB },
  { "Grp.ngSend", 0, grpNgSend },
  { "Grp.ngReport", 0, grpNgReport },
  { "Grp.prioStart", 0, grpPrioStart },
  { "Grp.prioRecv", 0, grpPrioRecv },
  { "Grp.expRecv", 0, grpExpRecv },
  { "Grp.secRecv", 0, secRecv },
  { "Grp.gotElem", CKDYN_EP_REDNTARGET, secGotElem },
  { "Grp.gotSec", CKDYN_EP_REDNTARGET, secGotSec },
  { "Grp.secCreate", 0, grpSecCreate },
  { "Grp.lateReg", 0, grpLateReg },
  { "Grp.csReport", 0, grpCsReport },
  { "Grp.cbRecv", 0, grpCbRecv },
  { "Grp.cbCount", 0, grpCbCount },
  { "Grp.redToSec", 0, memberRedToSec },
  { "Grp.sendDelete", 0, grpSendDelete },
} };

/* ---- NGrp (nodegroup) ---------------------------------------------------- */

static int64_t ngCtor(Obj *o, const CkDynMsgView *) {
  CHECK(o->t.kind == CKDYN_NODEGROUP && o->t.u.group.pe_or_node == CkMyNode(), "NGrp constructed with a wrong target");
  CHECK(CkDynGetReductionNumber(&o->t) == 0, "NGrp: reduction number of a new branch is not 0");
  g_ngObj = o;
  return 0;
}
/* incr: EXCLUSIVE, so a plain read-modify-write must not lose updates */
static int64_t ngIncr(Obj *o, const CkDynMsgView *) {
  int64_t c = o->s.counter;
  for (volatile int k = 0; k < 200; k++) {}
  o->s.counter = c + 1;
  return 0;
}

CbClass NGrp = { "NGrp", CKDYN_NODEGROUP, 0, {
  { "NGrp.NGrp", 0, ngCtor },
  { "NGrp.incr", CKDYN_EP_EXCLUSIVE, ngIncr },
} };

/* ---- Single (singleton chare) -------------------------------------------- */

static CkDynTarget g_singleHere;      /* a Single constructed on PE 0, as its constructor saw it */
static bool g_singleHereSet = false;
static int64_t singleCtor(Obj *o, const CkDynMsgView *) {
  CHECK(o->t.kind == CKDYN_CHARE && o->t.u.chare.pe == CkMyPe(), "Single constructed with a wrong target");
  CkDynTarget t;
  memset(&t, 0, sizeof t);
  t.kind = CKDYN_CHARE; t.chare_idx = Single.chareIdx; t.u.chare.pe = CkMyPe(); t.u.chare.ptr = o->t.u.chare.ptr;
  checkTargetBytes(o, t, "Single constructor");
  if (CkMyPe() == 0 && !g_singleHereSet) { g_singleHere = o->t; g_singleHereSet = true; }
  return 0;
}
static int64_t singleGotRed(Obj *o, const CkDynMsgView *v) { return gotRedTo0(o, v, "Single.gotRed"); }
/* inl (an INLINE entry) and plain: record the value and reference number, so
   the inline_chare phase can check the delivery ran inside the send */
static int64_t g_singleSeen = 0;
static uint32_t g_singleRef = 0;
static int64_t singleNote(Obj *, const CkDynMsgView *v) {
  CHECK(v->kind == CKDYN_MSGKIND_DYN, "Single.inl/plain: view kind %d", (int)v->kind);
  if (v->nslots > 1) {   /* the NCPY round of inline_chare: slot 1 holds 0.5 * i */
    CHECK(v->slots[1].type == CKDYN_SLOT_NCPY, "Single.inl: slot 1 has type %d", (int)v->slots[1].type);
    const double *p = (const double *)CkDynViewBufferPtr(v, 1, nullptr);
    for (int i = 0; i < 8; i++) CHECK(p[i] == 0.5 * i, "Single.inl: NCPY element %d is %g", i, p[i]);
  }
  g_singleSeen = cbI64(v, 0);
  g_singleRef = v->refnum;
  return 0;
}

CbClass Single = { "Single", CKDYN_CHARE, 0, {
  { "Single.Single", 0, singleCtor },
  { "Single.gotRed", CKDYN_EP_REDNTARGET, singleGotRed },
  { "Single.inl", CKDYN_EP_INLINE, singleNote },
  { "Single.plain", 0, singleNote },
} };

/* ---- DMain (dynamic main chare: the checkpoint callback target) --------- */

static Obj *g_dmain = nullptr;          /* PE 0 */
static bool g_dmainBuiltHere = false;   /* constructed in this process, so not restored from a checkpoint */
static const uint32_t CKPT_MARKER = 4242;
static const int64_t CKPT_STATE = 777;

static int64_t dmainCtor(Obj *o, const CkDynMsgView *v) {
  CHECK(v == nullptr && o->t.kind == CKDYN_MAINCHARE && CkMyPe() == 0, "DMain constructed with a wrong target");
  g_dmain = o;
  g_dmainBuiltHere = true;
  return 0;
}

/* resumed: the checkpoint callback, with the marker as its reference number.
 * In the checkpointing run it reports to TestMain; after +restart it is the
 * whole test. */
static int64_t dmainResumed(Obj *o, const CkDynMsgView *v) {
  CHECK(CkMyPe() == 0, "DMain.resumed on PE %d", CkMyPe());
  CHECK(v->refnum == CKPT_MARKER, "DMain.resumed: reference number %u, expected the marker %u", v->refnum, CKPT_MARKER);
  CHECK(o->s.counter == CKPT_STATE, "DMain.resumed: state %lld, expected %lld", (long long)o->s.counter, (long long)CKPT_STATE);
  if (g_dmainBuiltHere) { CkDynFutureSet(0, (uint32_t)o->s.fid, emptyMsg()); return 0; }
  bool staleRestart = false;
  for (char **a = CkGetArgv(); *a; a++) if (strcmp(*a, "-stalerestart") == 0) staleRestart = true;
  if (staleRestart) {
    /* the first section made after the restart reuses the held id's root,
       index and generation; only the run counter tells them apart */
    const CkDynSectionID held = o->s.heldSid;
    CHECK(held.gen != 0 && aidSecA >= 0, "-stalerestart: the checkpoint holds no section id");
    int zero = 0;
    CkDynSectionID fresh;
    CkDynSectionCreateArray(aidSecA, 1, 1, &zero, 0, nullptr, &fresh);
    CHECK(fresh.root_pe == held.root_pe && fresh.idx == held.idx && fresh.gen == held.gen &&
          fresh.reserved == (uint8_t)(held.reserved + 1),
          "-stalerestart: new id (%d, %d, %u, run %u), held id (%d, %d, %u, run %u)", fresh.root_pe, fresh.idx,
          fresh.gen, fresh.reserved, held.root_pe, held.idx, held.gen, held.reserved);
    CkPrintf("  sending to a section id from before the restart; CkDyn must abort\n");
    int ep = Sec.ep(SC_RECV);
    CkDynSendSection(&held, ep, MsgB().i64(0).i64(-1).finishMcast(ep), nullptr);
    CHECK(false, "a send with a section id from before the restart did not abort");
  }
  CkPrintf("PASS restart: DMain.resumed ran on the restored main chare (marker %u, state %lld, view kind %d)\n",
           v->refnum, (long long)o->s.counter, (int)v->kind);
  CHECK(cbUnpacks() > 0, "restart: no unpack call on PE 0");
  CkPrintf("PASS restart: unpack saw the registered chare index in the target (%lld objects on PE 0)\n", (long long)cbUnpacks());
  CkPrintf("ALL DONE (restart)\n");
  CkDynExit(0);
  return 0;
}

CbClass DMain = { "DMain", CKDYN_MAINCHARE, 0, {
  { "DMain.DMain", 0, dmainCtor },
  { "DMain.resumed", 0, dmainResumed },
} };

/* ---- Mig (1-D array without deliver_many, the bcast_migrate phase) ------ */

/* this PE's Mig elements constructed here and not migrated away, by index */
typedef std::map<int, Obj *> MigLocalMap;
CkpvStaticDeclare(MigLocalMap, migLocal);

static int64_t migCtor(Obj *o, const CkDynMsgView *) {
  o->s.counter = 0;
  CkpvAccess(migLocal)[myIndex(o)] = o;
  return 0;
}
/* bcast(CALLBACK cb): count the broadcast; if the partner element (index ^ 1)
 * is local and has not received it yet, migrate it to the next PE, so that it
 * leaves in the middle of this PE's delivery of the broadcast. Contribute
 * [1, moved]. */
static int64_t migBcast(Obj *o, const CkDynMsgView *v) {
  o->s.counter++;
  int moved = 0;
  auto &local = CkpvAccess(migLocal);
  auto it = local.find(myIndex(o) ^ 1);
  if (CkNumPes() > 1 && it != local.end() && it->second->s.counter < o->s.counter) {
    Obj *p = it->second;
    local.erase(it);
    CkDynMigrateMe(&p->t, (CkMyPe() + 1) % CkNumPes());
    moved = 1;
  }
  local.erase(myIndex(o));   /* it may move later (load balancing); stop naming it */
  int r[2] = { 1, moved };
  contributeInts(o, r, 2, viewCb(v, 0));
  return 0;
}

CbClass Mig = { "Mig", CKDYN_ARRAY, 1, {
  { "Mig.Mig", 0, migCtor },
  { "Mig.bcast", 0, migBcast },
} };

/* ---- Late (1-D array registered after start-up, the late phase) -------- */

static int64_t lateCtor(Obj *o, const CkDynMsgView *v) {
  CHECK(v->hdr && (v->hdr->flags & CKDYN_MSG_CREATION), "Late constructor message lost its flags");
  CHECK(o->t.kind == CKDYN_ARRAY && o->t.chare_idx == Late.chareIdx, "Late constructed with a wrong target");
  o->s.counter = 0;
  return 0;
}
/* add(value, cb): counter += value * (index + 1); contribute (counter, 1) */
static int64_t lateAdd(Obj *o, const CkDynMsgView *v) {
  o->s.counter += cbI64(v, 0) * (myIndex(o) + 1);
  int r[2] = { (int)o->s.counter, 1 };
  contributeInts(o, r, 2, viewCb(v, 1));
  return 0;
}
/* twice(x) with CKDYN_MSG_BLOCK: a point-to-point send whose future gets 2x */
static int64_t lateTwice(Obj *, const CkDynMsgView *v) { return 2 * cbI64(v, 0); }

CbClass Late = { "Late", CKDYN_ARRAY, 1, {
  { "Late.Late", 0, lateCtor },
  { "Late.add", CKDYN_EP_REDNTARGET, lateAdd },
  { "Late.twice", 0, lateTwice },
} };

/* ---- Tmp (1-D array whose elements destroy themselves, the callstack phase) */

static int64_t tmpCtor(Obj *, const CkDynMsgView *) { return 0; }
/* destroy: inside a CkDynCallstackPush/Pop bracket the element destroys
   itself; Pop must still pop its entry, leaving the running object as it was */
static int64_t tmpDestroy(Obj *o, const CkDynMsgView *) {
  CkDynTarget t = o->t;   /* o is freed by the destruct hook when the element goes */
  /* an outer bracket on the local Grp branch, so that the object expected
     after Pop is not NULL (a nulled entry left on the stack would read as NULL) */
  CkDynTarget g = cbGroupTarget(gidGrp, Grp.chareIdx, CkMyPe());
  CkDynCallstackPush(&g);
  Chare *before = CkActiveObj();
  CHECK(before != nullptr, "callstack: the Grp branch is not the running object after CkDynCallstackPush");
  CkDynCallstackPush(&t);
  Chare *self = CkActiveObj();
  CHECK(self != nullptr && self != before, "callstack: CkDynCallstackPush did not make the element the running object");
  CkDynArrayDestroyElement(t.u.array.aid, 1, t.u.array.index);
  CkDynCallstackPop(&t);
  CHECK(CkActiveObj() == before, "callstack: after the element destroyed itself, CkDynCallstackPop left %p "
        "as the running object; expected %p", (void *)CkActiveObj(), (void *)before);
  CkDynCallstackPop(&g);
  std::lock_guard<std::mutex> lk(g_csLock);
  g_csDone[CkMyPe()]++;
  return 0;
}

CbClass Tmp = { "Tmp", CKDYN_ARRAY, 1, {
  { "Tmp.Tmp", 0, tmpCtor },
  { "Tmp.destroy", 0, tmpDestroy },
} };

/* ====================================================================== */
/* Registration                                                            */
/* ====================================================================== */

void ckdynRegister(void) {
  CkDynInit(cbHooks());
  cbRegister(&Ping);
  cbRegister(&Sec);
  cbRegister(&Grp);
  cbRegister(&NGrp);
  cbRegister(&Single);
  cbRegister(&DMain);
  cbRegister(&Tmp);
  Mig.noMany = true;
  cbRegister(&Mig);
  CkDynAllowLateRegistration();   /* Late is registered by the late phase */
}

void ckdynTestInitProc(void) {
  cbInitPe();
  CkpvInitialize(MigLocalMap, migLocal);
}

/* ====================================================================== */
/* The main chare                                                          */
/* ====================================================================== */

static void bcastPing(int m, CkDynMsg *msg) { CkDynSendArray(aidPing, 1, nullptr, Ping.ep(m), msg, nullptr); }

class TestMain : public CBase_TestMain {
  std::set<std::string> skip;
  std::vector<std::string> steps = {
    "pingpong", "pingpong_inline", "bcast", "bcast_migrate",
    "reductions/table", "reductions/single", "reductions/single_any", "reductions/group_elem", "reductions/group_bcast",
    "reductions/array_elem", "reductions/array_bcast", "reductions/future", "reductions/foreign",
    "future_return", "nodegroup", "qd", "priority", "coroutine", "builder", "callstack", "inline_chare",
    "constrained",
    "ncpy/nonpost", "ncpy/post_before", "ncpy/post_after", "ncpy/samepe",
    "device/remote", "device/samepe", "hapi", "devpersist/remote", "devpersist/samepe",
    "sections/array", "sections/group", "sections/contribute", "sections/remote", "sections/combine_overlap",
    "sections/combine_arrays", "sections/delete", "sections/order", "sections/migration",
    "migration", "checkpoint", "late", "exit" };
  std::string ckptDir = "ckdyntest.ckpt";
  int step = -1;
  int pending = 0;
  double tStart = 0;
  CkDynTarget single0, singleAny;
  bool staleSend = false;   /* -stalesend: the delete phase sends to the deleted section, which must abort */
  bool requireMoves = false; /* -requiremoves: a load balancer that moves elements is loaded (RotateLB) */
  bool zcBcast = false;      /* -zcbcast: the ncpy/nonpost phase broadcasts an NCPY slot, which must abort */
  bool xiSection = false;    /* -xisection: a section of a charmxi array, which must abort */
  bool ckptSection = false;  /* -ckptsection: the checkpoint phase passes a SECTION callback, which must abort */
  bool deleteWaiting = false; /* -deletewaiting: the delete phase deletes a part a combine waits on, which must abort */
  bool staleRedNo = false;   /* -staleredno: a member asks for the reduction number of a deleted section, which must abort */

  /* sections: the expected membership of each section the phases create */
  struct Sect {
    CkDynSectionID sid;
    std::set<std::pair<int, int> > mem;   /* (aid, index), or (-1, pe) for a group section */
    bool group = false;
    int size = 0;
    double sum = 0;
    int reds = 1;                         /* section reductions so far: the creation reduction is the first */
  };
  std::map<std::string, Sect> sects;
  std::deque<std::function<void()> > q;   /* the steps of the current sections phase */
  std::string cur;

  bool skipped(const std::string &s) {
    if (skip.count(s)) return true;
    size_t slash = s.find('/');
    return slash != std::string::npos && skip.count(s.substr(0, slash));
  }
  void pass(const std::string &s) { CkPrintf("PASS %s (%.3f s)\n", s.c_str(), CkWallTimer() - tStart); next(); }
  void next() { thisProxy.nextPhase(); }

public:
  TestMain(CkArgMsg *m) {
    for (int i = 1; i < m->argc; i++)
      if (strcmp(m->argv[i], "-skip") == 0 && i + 1 < m->argc) skip.insert(m->argv[++i]);
      else if (strcmp(m->argv[i], "-ckptdir") == 0 && i + 1 < m->argc) ckptDir = m->argv[++i];
      else if (strcmp(m->argv[i], "-stalesend") == 0) staleSend = true;
      else if (strcmp(m->argv[i], "-requiremoves") == 0) requireMoves = true;
      else if (strcmp(m->argv[i], "-zcbcast") == 0) zcBcast = true;
      else if (strcmp(m->argv[i], "-badlatereg") == 0) {
        /* CkDynRegisterClass after start-up must abort and name CkDynRegisterClassLate */
        CkPrintf("  registering Late with CkDynRegisterClass after start-up; CkDyn must abort\n");
        cbRegister(&Late);
        CHECK(false, "CkDynRegisterClass after start-up did not abort");
      }
      else if (strcmp(m->argv[i], "-xisection") == 0) xiSection = true;
      else if (strcmp(m->argv[i], "-ckptsection") == 0) ckptSection = true;
      else if (strcmp(m->argv[i], "-deletewaiting") == 0) deleteWaiting = true;
      else if (strcmp(m->argv[i], "-staleredno") == 0) staleRedNo = true;
    delete m;
    mainProxy = thisProxy;
    numElems = 2 * CkNumPes();
    CkPrintf("ckdyntest: %d PEs, %d nodes, %d Ping elements, refnum %d bytes\n",
             CkNumPes(), CkNumNodes(), numElems, CkDynRefnumBytes());
    tStart = CkWallTimer();
    CkDynCreateManagers();
    if (xiSection) {
      CkPrintf("  making a section of a charmxi array; CkDyn must abort\n");
      CProxy_Plain plain = CProxy_Plain::ckNew(2);
      int idx[2] = { 0, 1 };
      CkDynSectionID sid;
      CkDynSectionCreateArray(((CkGroupID)plain.ckGetArrayID()).idx, 2, 1, idx, 0, nullptr, &sid);
      CkStartQD(CkCallback(CkIndex_TestMain::nextPhase(), thisProxy));
      return;
    }

    /* phase creation: all five creation futures */
    /* -skip nodegroup also skips creating NGrp (its constructor is where a
       nodegroup defect would show) */
    bool withNg = skip.count("nodegroup") == 0;
    bool withSec = skip.count("sections") == 0;
    pending = (withNg ? 5 : 4) + (withSec ? 2 : 0);
    auto created = [this](const char *what) {
      return cbNewFuture([this, what](const CkDynMsgView *v) {
        CkPrintf("  created %s (view kind %d)\n", what, (int)v->kind);
        if (--pending == 0) pass("creation");
      });
    };
    int dims[1] = { numElems };
    CkDynArrayOpts ao = { 1, dims, -1, 0, 0, -1 };
    aidPing = CkDynCreateArray(Ping.chareIdx,
        MsgB().flags(CKDYN_MSG_CREATION | CKDYN_MSG_USESATSYNC).fut(0, created("Ping")).finish(Ping.ep(0)), &ao);
    gidGrp = CkDynCreateGroup(Grp.chareIdx, MsgB().flags(CKDYN_MSG_CREATION).fut(0, created("Grp")).finish(Grp.ep(0)));
    gidNGrp = !withNg ? -1 : CkDynCreateNodeGroup(NGrp.chareIdx, MsgB().flags(CKDYN_MSG_CREATION).fut(0, created("NGrp")).finish(NGrp.ep(0)));
    if (withSec) {
      nSecA = 2 * CkNumPes() + 1;
      nSecB = CkNumPes() + 2;
      int da[1] = { nSecA }, db[1] = { nSecB };
      CkDynArrayOpts oa = { 1, da, -1, 0, 0, -1 }, ob = { 1, db, -1, 0, 0, -1 };
      aidSecA = CkDynCreateArray(Sec.chareIdx,
          MsgB().flags(CKDYN_MSG_CREATION | CKDYN_MSG_USESATSYNC).fut(0, created("Sec A")).finish(Sec.ep(0)), &oa);
      aidSecB = CkDynCreateArray(Sec.chareIdx,
          MsgB().flags(CKDYN_MSG_CREATION | CKDYN_MSG_USESATSYNC).fut(0, created("Sec B")).finish(Sec.ep(0)), &ob);
    }
    CkDynCreateChare(Single.chareIdx, 0, MsgB().flags(CKDYN_MSG_CREATION).fut(0, created("Single(pe 0)")).finish(Single.ep(0)), &single0);
    CkDynCreateChare(Single.chareIdx, CKDYN_PE_ANY, MsgB().flags(CKDYN_MSG_CREATION).fut(0, created("Single(any)")).finish(Single.ep(0)), &singleAny);
  }

  void nextPhase() {
    CHECK(!xiSection, "a section of a charmxi array did not abort");
    step++;
    CkEnforce(step < (int)steps.size());
    const std::string s = steps[step];
    if (skipped(s) && s != "exit") { CkPrintf("SKIP %s\n", s.c_str()); next(); return; }
    if (s == "pingpong" || s == "pingpong_inline") runPingpong(s, s == "pingpong" ? 0 : 1);
    else if (s == "bcast") runBcast();
    else if (s == "bcast_migrate") runBcastMigrate();
    else if (s.compare(0, 11, "reductions/") == 0) runReduction(s);
    else if (s == "future_return") runFutureReturn();
    else if (s == "nodegroup") runNodegroup();
    else if (s == "qd") runQD();
    else if (s == "priority") runPriority();
    else if (s == "coroutine") runCoroutine();
    else if (s == "builder") runBuilder();
    else if (s == "callstack") runCallstack();
    else if (s == "inline_chare") runInlineChare();
    else if (s == "constrained") runConstrained();
    else if (s.compare(0, 5, "ncpy/") == 0) runNcpy(s);
    else if (s.compare(0, 7, "device/") == 0) runDevice(s);
    else if (s == "hapi") runHapi();
    else if (s.compare(0, 11, "devpersist/") == 0) runDevPersist(s);
    else if (s.compare(0, 9, "sections/") == 0) runSections(s);
    else if (s == "migration") runMigration();
    else if (s == "checkpoint") runCheckpoint();
    else if (s == "late") runLate();
    else if (s == "exit") { CkPrintf("ALL DONE (%.3f s)\n", CkWallTimer() - tStart); CkDynExit(0); }
  }

  /* bcast_migrate: an array without deliver_many; during one broadcast the
     handler of one element of each co-located pair migrates the other away
     before it has received the broadcast. Every element must still receive
     it exactly once (each contributes 1); quiescence ends the phase, so a
     lost delivery fails instead of hanging. */
  void runBcastMigrate() {
    const int n = 2 * CkNumPes();
    uint32_t fc = cbNewFuture([this, n](const CkDynMsgView *) {
      int *got = new int[2]{ -1, -1 };
      uint32_t fr = cbNewFuture([got](const CkDynMsgView *v) { got[0] = redInt(v, 0); got[1] = redInt(v, 1); });
      uint32_t fq = cbNewFuture([this, n, got](const CkDynMsgView *) {
        int cnt = got[0], moved = got[1];
        delete[] got;
        CHECK(cnt == n, "bcast_migrate: %d of %d elements received the broadcast", cnt, n);
        CHECK(CkNumPes() == 1 || moved > 0, "bcast_migrate: no element was migrated during the broadcast");
        CkPrintf("  bcast_migrate: %d elements received the broadcast once; %d were migrated during it\n", cnt, moved);
        pass("bcast_migrate");
      });
      CkDynSendArray(migAid, 1, nullptr, Mig.ep(1), MsgB().cb(cbFutureCb(0, fr)).finish(0), nullptr);
      CkDynCallback qd = cbFutureCb(0, fq);
      CkDynStartQD(&qd);
    });
    int dims[1] = { n };
    CkDynArrayOpts ao = { 1, dims, -1, 0, 0, -1 };
    migAid = CkDynCreateArray(Mig.chareIdx, MsgB().flags(CKDYN_MSG_CREATION).fut(0, fc).finish(Mig.ep(0)), &ao);
  }
  int migAid = -1;

  void runPingpong(std::string s, int mode) {
    uint32_t f = cbNewFuture([this, s, mode](const CkDynMsgView *v) {
      int64_t rounds = cbI64(v, 0);
      double el = cbF64(v, 1);
      int64_t pe0 = cbI64(v, 2), pe1 = cbI64(v, 3), sync = cbI64(v, 4);
      if (mode == 1 && rounds < 0) {
        CkPrintf("  %s: element 0 on PE %lld; inline ran synchronously for %lld of %d sends "
                 "(elements on different PEs?)\n", s.c_str(), (long long)pe0, (long long)sync, PINGPONG_ROUNDS);
        CHECK(false, "inline sends to a local element did not run synchronously");
      }
      CHECK(rounds == PINGPONG_ROUNDS, "%s: %lld rounds, expected %d", s.c_str(), (long long)rounds, PINGPONG_ROUNDS);
      double rt = el / rounds * 1e6;
      CkPrintf("  %s: element 0 on PE %lld, element 1 on PE %lld%s\n", s.c_str(), (long long)pe0, (long long)pe1,
               mode ? " (inline ran synchronously on every send)" : "");
      CkPrintf("  %s round trip (us) = %.3f\n", s.c_str(), rt);
      CkPrintf("  %s per message (us) = %.3f\n", s.c_str(), rt / 2);
      pass(s);
    });
    int zero = 0;
    CkDynSendArray(aidPing, 1, &zero, Ping.ep(P_STARTPING),
                   MsgB().i64(PINGPONG_ROUNDS).i64(mode).fut(0, f).finish(0), nullptr);
  }

  void runBcast() {
    const int value = 3;
    uint32_t f = cbNewFuture([this, value](const CkDynMsgView *v) {
      int sum = redInt(v, 0), many = redInt(v, 1), once = redInt(v, 2), counters = redInt(v, 3);
      CHECK(sum == numElems * value, "bcast: sum %d, expected %d", sum, numElems * value);
      CHECK(many == numElems, "bcast: %d of %d elements were delivered through deliver_many", many, numElems);
      CHECK(once == numElems, "bcast: deliver_many was not called exactly once per PE (%d of %d elements saw one call)", once, numElems);
      CHECK(counters == numElems * value, "bcast: per-element counters sum to %d, expected %d", counters, numElems * value);
      pass("bcast");
    });
    bcastPing(P_COUNT, MsgB().i64(value).cb(cbFutureCb(0, f)).finish(0));
  }

  void runReduction(const std::string &s) {
    std::string kind = s.substr(11);
    CkDynCallback cb;
    int mode = 0;
    if (kind == "table") {
      /* element kinds of the reducers whose data type depends on the platform */
      const CkDynReducerInfo *t;
      int n = CkDynReducers(&t);
      std::map<std::string, int> elem;
      for (int i = 0; i < n; i++) elem[t[i].name] = t[i].elem;
      int charElem = std::is_signed<char>::value ? CKDYN_ELEM_I8 : CKDYN_ELEM_U8;
      for (const char *r : { "sum_char", "product_char", "max_char", "min_char" })
        CHECK(elem.count(r) && elem[r] == charElem, "%s: element kind %d, expected %d (char is %s here)",
              r, elem.count(r) ? elem[r] : -1, charElem, std::is_signed<char>::value ? "signed" : "unsigned");
      CHECK(elem["sum_uchar"] == CKDYN_ELEM_U8, "sum_uchar: element kind %d", elem["sum_uchar"]);
      CHECK(elem["random"] == CKDYN_ELEM_BYTE, "random: element kind %d, expected BYTE", elem["random"]);
      pass(s);
      return;
    }
    if (kind == "group_bcast" || kind == "array_bcast") {
      int expect = kind == "group_bcast" ? CkNumPes() : numElems;
      uint32_t f = cbNewFuture([this, s, expect](const CkDynMsgView *v) {
        int got = redInt(v, 0);
        CHECK(got == expect, "%s: %d receivers confirmed, expected %d", s.c_str(), got, expect);
        pass(s);
      });
      if (kind == "group_bcast") cb = cbTargetCb(cbGroupTarget(gidGrp, Grp.chareIdx, -1), Grp.ep(G_GOTREDB), 1, f);
      else cb = cbTargetCb(cbArrayTarget1(aidPing, Ping.chareIdx, 0), Ping.ep(P_GOTREDB), 1, f);
    } else if (kind == "future") {
      cb = cbFutureCb(0, cbNewFuture([this, s](const CkDynMsgView *v) {
        CHECK(v->kind == CKDYN_MSGKIND_REDRESULT, "%s: view kind %d", s.c_str(), (int)v->kind);
        double got = redDouble(v, 0);
        CHECK(got == expectedSum(), "%s: sum %g, expected %g", s.c_str(), got, expectedSum());
        pass(s);
      }));
    } else if (kind == "foreign") {
      mode = 1;
      cb = cbFutureCb(0, cbNewFuture([this, s](const CkDynMsgView *v) {
        CHECK(v->kind == CKDYN_MSGKIND_DYN && v->hdr, "%s: view kind %d, expected a CkDyn payload", s.c_str(), (int)v->kind);
        CHECK(v->hdr->user_id == CB_REDUCER_SUM_I64, "%s: result user_id %u", s.c_str(), v->hdr->user_id);
        int64_t got = cbI64(v, 0);
        CHECK(got == (int64_t)expectedSum(), "%s: sum %lld, expected %g", s.c_str(), (long long)got, expectedSum());
        CHECK(cbCheckReduceBuffer(v) == 1, "%s: the result did not go through the reduce hook", s.c_str());
        pass(s);
      }));
    } else {
      uint32_t f = cbNewFuture([this, s](const CkDynMsgView *) { pass(s); });
      if (kind == "single") cb = cbTargetCb(single0, Single.ep(S_GOTRED), 0, f);
      else if (kind == "single_any") cb = cbTargetCb(singleAny, Single.ep(S_GOTRED), 0, f);
      else if (kind == "group_elem") cb = cbTargetCb(cbGroupTarget(gidGrp, Grp.chareIdx, 0), Grp.ep(G_GOTRED), 0, f);
      else if (kind == "array_elem") cb = cbTargetCb(cbArrayTarget1(aidPing, Ping.chareIdx, 1), Ping.ep(P_GOTRED), 0, f);
      else CkAbort("unknown reduction step %s", s.c_str());
    }
    bcastPing(P_DORED, MsgB().i64(mode).cb(cb).finish(0));
  }

  void runFutureReturn() {
    uint32_t f = cbNewFuture([this](const CkDynMsgView *v) {
      int64_t got = cbI64(v, 0);
      CHECK(got == 42, "future_return: got %lld, expected 42", (long long)got);
      pass("future_return");
    });
    int one = 1;
    CkDynSendArray(aidPing, 1, &one, Ping.ep(P_COMPUTE),
                   MsgB().flags(CKDYN_MSG_BLOCK).i64(21).fut(0, f).finish(0), nullptr);
  }

  void runNodegroup() {
    CkDynSendGroup(gidGrp, -1, Grp.ep(G_NGSEND), MsgB().i64(NG_MSGS).finish(0), nullptr);
    uint32_t fq = cbNewFuture([this](const CkDynMsgView *) {
      uint32_t fr = cbNewFuture([this](const CkDynMsgView *v) {
        int got = redInt(v, 0);
        CHECK(got == NG_MSGS * CkNumPes(), "nodegroup: EXCLUSIVE counter %d, expected %d", got, NG_MSGS * CkNumPes());
        pass("nodegroup");
      });
      CkDynSendGroup(gidGrp, -1, Grp.ep(G_NGREPORT), MsgB().cb(cbFutureCb(0, fr)).finish(0), nullptr);
    });
    CkDynCallback cb = cbFutureCb(0, fq);
    CkDynStartQD(&cb);
  }

  void runQD() {
    CkDynCallback cb = cbFutureCb(0, cbNewFuture([this](const CkDynMsgView *v) {
      CkPrintf("  qd: callback view kind %d, %d slots\n", (int)v->kind, v->nslots);
      pass("qd");
    }));
    CkDynStartQD(&cb);
  }

  void runPriority() {
    uint32_t f = cbNewFuture([this](const CkDynMsgView *v) {
      int64_t a = cbI64(v, 0), b = cbI64(v, 1), c = cbI64(v, 2);
      CHECK(a == 1 && b == 3 && c == 5, "priority: delivery order %lld %lld %lld, expected 1 3 5",
            (long long)a, (long long)b, (long long)c);
      CkPrintf("  priority: order 1 3 5, expedited message delivered\n");
      pass("priority");
    });
    CkDynSendGroup(gidGrp, 0, Grp.ep(G_PRIOSTART), MsgB().fut(0, f).finish(0), nullptr);
  }

  void runCoroutine() {
    uint32_t f = cbNewFuture([this](const CkDynMsgView *v) {
      int got = redInt(v, 0);
      CHECK(got == numElems, "coroutine: %d elements confirmed, expected %d", got, numElems);
      pass("coroutine");
    });
    bcastPing(P_CORO, MsgB().cb(cbFutureCb(0, f)).finish(0));
  }

  /* callstack: every element of a new Tmp array destroys itself between
     CkDynCallstackPush and Pop (tmpDestroy checks the running object); then
     QD, and the Grp branches report how many did. Also Push/Pop of a target
     that is not local keeps the stack balanced. */
  /* builder: edge cases of the message builder that must not abort: a
     negative slot hint, and an empty buffer with real strides (numpy 2's
     np.zeros((5, 3))[:, :0] has shape (5, 0) and strides (24, 8)) */
  void runBuilder() {
    CkDynMsgBuilder *b = CkDynMsgBegin(-1, 0, 0, 0, -1);
    static const int64_t shape[2] = { 5, 0 }, strides[2] = { 24, 8 };
    static double dummy;
    CkDynBufDesc d = { &dummy, 0, CKDYN_ELEM_F64, 2, 0, 0, shape, strides };
    CkDynMsgAddBuffer(b, &d);
    CkDynMsg *m = CkDynMsgFinish(b, 0);
    uint32_t len;
    const CkDynHdr *h = (const CkDynHdr *)CkDynMsgPayload(m, &len);
    const CkDynSlot *sl = (const CkDynSlot *)(h + 1);
    CHECK(h->nslots == 1 && sl[0].type == CKDYN_SLOT_BUFFER && sl[0].len == 0 && sl[0].ndim == 2,
          "builder: the empty strided buffer was not kept as a 2-D BUFFER slot of 0 bytes");
    CkDynMsgFree(m);
    pass("builder");
  }

  void runCallstack() {
    int n = numElems;
    uint32_t fc = cbNewFuture([this, n](const CkDynMsgView *) {
      CkDynSendArray(tmpAid, 1, nullptr, Tmp.ep(T_DESTROY), emptyMsg(), nullptr);
      CkDynCallback qd = cbFutureCb(0, cbNewFuture([this, n](const CkDynMsgView *) {
        Chare *before = CkActiveObj();
        CkDynTarget gone = cbArrayTarget1(tmpAid, Tmp.chareIdx, 0);   /* destroyed: not local anywhere */
        CkDynCallstackPush(&gone);
        CkDynCallstackPop(&gone);
        CHECK(CkActiveObj() == before, "callstack: Push/Pop of a non-local target changed the running object");
        uint32_t fr = cbNewFuture([this, n](const CkDynMsgView *v) {
          int got = redInt(v, 0);
          CHECK(got == n, "callstack: %d elements destroyed themselves inside the bracket, expected %d", got, n);
          CkPrintf("  callstack: %d elements destroyed themselves between Push and Pop; running object restored\n", n);
          pass("callstack");
        });
        CkDynSendGroup(gidGrp, -1, Grp.ep(G_CSREPORT), MsgB().cb(cbFutureCb(0, fr)).finish(0), nullptr);
      }));
      CkDynStartQD(&qd);
    });
    int dims[1] = { n };
    CkDynArrayOpts ao = { 1, dims, -1, 0, 0, -1 };
    tmpAid = CkDynCreateArray(Tmp.chareIdx, MsgB().flags(CKDYN_MSG_CREATION).fut(0, fc).finish(Tmp.ep(T_CTOR)), &ao);
  }
  int tmpAid = -1;

  /* inline_chare: inline delivery to the singleton chare on this PE (PE 0):
     an INLINE entry, CKDYN_SEND_INLINE to a plain entry, and a TARGET callback
     to the INLINE entry. Each must run inside the call. */
  void runInlineChare() {
    /* the constructor's own target: CkDynCreateChare's is a virtual chare id,
       which CkSendMsgInline sends through the scheduler */
    CHECK(g_singleHereSet && g_singleHere.u.chare.pe == CkMyPe(), "inline_chare: no Single on PE 0");
    const CkDynTarget &single = g_singleHere;
    g_singleSeen = 0;
    CkDynSendChare(&single, Single.ep(S_INL), MsgB().i64(11).finish(0), nullptr);
    CHECK(g_singleSeen == 11, "inline_chare: the INLINE entry did not run inside CkDynSendChare (saw %lld)", (long long)g_singleSeen);
    CkDynSendOpts io = { CKDYN_SEND_INLINE };
    CkDynSendChare(&single, Single.ep(S_PLAIN), MsgB().i64(12).finish(0), &io);
    CHECK(g_singleSeen == 12, "inline_chare: CKDYN_SEND_INLINE did not run inside CkDynSendChare (saw %lld)", (long long)g_singleSeen);
    CkDynCallback cb = cbTargetCb(single, Single.ep(S_INL), 0, 77);
    CkDynCallbackSend(&cb, MsgB().i64(13).finish(0));
    CHECK(g_singleSeen == 13 && g_singleRef == 77, "inline_chare: the TARGET callback to the INLINE entry saw %lld, "
          "refnum %u (expected 13, 77 inside CkDynCallbackSend)", (long long)g_singleSeen, g_singleRef);
    /* with an NCPY slot the same TARGET callback must not deliver inline: the
       transfer runs in the scheduler's handler */
    static double zbuf[8];
    for (int i = 0; i < 8; i++) zbuf[i] = 0.5 * i;
    CkDynBufDesc d = { zbuf, sizeof zbuf, CKDYN_ELEM_F64, 0, 0, 0, nullptr, nullptr };
    MsgB b;
    b.i64(14);
    CkDynMsgAddNcpy(b.b, &d, nullptr, nullptr);
    g_singleSeen = 0;
    CkDynCallback cbz = cbTargetCb(single, Single.ep(S_INL), 0, 78);
    CkDynCallbackSend(&cbz, b.finish(0));
    CHECK(g_singleSeen == 0, "inline_chare: a message with an NCPY slot was delivered inline through a TARGET callback");
    CkDynCallback qd = cbFutureCb(0, cbNewFuture([this](const CkDynMsgView *) {
      CHECK(g_singleSeen == 14 && g_singleRef == 78, "inline_chare: the NCPY message through the TARGET callback "
            "saw %lld, refnum %u (expected 14, 78)", (long long)g_singleSeen, g_singleRef);
      CkPrintf("  inline_chare: INLINE entry, CKDYN_SEND_INLINE and an inline TARGET callback ran inside the call; "
               "with an NCPY slot the callback went through the scheduler\n");
      pass("inline_chare");
    }));
    CkDynStartQD(&qd);
  }

  /* constrained: a Grp group with branches on the even PEs only. A
     broadcast TARGET callback (CkDynCallbackSend) and a foreign reduction
     result over Ping to a group broadcast callback both reach every branch;
     the members report (cbRecv), the other branches must drop them. */
  void runConstrained() {
    std::vector<int32_t> pes;
    for (int p = 0; p < CkNumPes(); p += 2) pes.push_back(p);
    g_cbExpect = (int64_t)pes.size();
    uint32_t fc = cbNewFuture([this](const CkDynMsgView *) {
      CkDynTarget all = cbGroupTarget(cgid, Grp.chareIdx, -1);
      uint32_t f2 = cbNewFuture([this](const CkDynMsgView *v) {
        CHECK(cbI64(v, 0) == g_cbExpect && cbI64(v, 1) == (int64_t)expectedSum(),
              "constrained: foreign result reached %lld members with %lld; expected %lld with %g",
              (long long)cbI64(v, 0), (long long)cbI64(v, 1), (long long)g_cbExpect, expectedSum());
        CkPrintf("  constrained: %lld of %d branches are members; a broadcast callback and a foreign reduction "
                 "result reached each member once and were dropped elsewhere\n", (long long)g_cbExpect, CkNumPes());
        pass("constrained");
      });
      uint32_t f1 = cbNewFuture([this, all, f2](const CkDynMsgView *v) {
        CHECK(cbI64(v, 0) == g_cbExpect && cbI64(v, 1) == 5, "constrained: broadcast callback reached %lld members "
              "with %lld; expected %lld with 5", (long long)cbI64(v, 0), (long long)cbI64(v, 1), (long long)g_cbExpect);
        bcastPing(P_DORED, MsgB().i64(1).cb(cbTargetCb(all, Grp.ep(G_CBRECV), 1, f2)).finish(0));
      });
      CkDynCallback cb = cbTargetCb(all, Grp.ep(G_CBRECV), 1, f1);
      CkDynCallbackSend(&cb, MsgB().i64(5).finish(0));
    });
    MsgB b;
    b.flags(CKDYN_MSG_CREATION | CKDYN_MSG_CONSTRAINED).fut(0, fc);
    CkDynMsgAddBytes(b.b, pes.data(), (uint32_t)(pes.size() * sizeof(int32_t)), 0);
    cgid = CkDynCreateGroup(Grp.chareIdx, b.finish(Grp.ep(G_CTOR)));
  }
  int cgid = -1;

  /* ncpy: 16 MB as an NCPY slot from element 0 to the last element (on the
     last PE) and back, then the post path with the post made before and
     after the send, then to element 1 (element 0's PE). See zStart. */
  void runNcpy(const std::string &s) {
    int last = numElems - 1;
    std::string kind = s.substr(5);
    uint32_t f = cbNewFuture([this, s](const CkDynMsgView *v) {
      CHECK(cbI64(v, 0) == 1, "%s: the echo to element 0 landed in a posted buffer", s.c_str());
      CkPrintf("  %s: 16 MB out and back, each done function called once (%.3f ms)\n", s.c_str(), cbF64(v, 1) * 1e3);
      pass(s);
    });
    if (zcBcast) {
      static double one[4] = { 1, 2, 3, 4 };
      CkDynBufDesc d = { one, sizeof one, CKDYN_ELEM_F64, 0, 0, 0, nullptr, nullptr };
      MsgB b;
      b.i64(0).i64(-1);
      CkDynMsgAddNcpy(b.b, &d, nullptr, nullptr);
      CkPrintf("  broadcasting an NCPY slot; CkDyn must abort\n");
      bcastPing(P_ZRECV, b.i64(0).finish(0));
      CHECK(false, "a broadcast with an NCPY slot did not abort");
    }
    if (kind == "nonpost") sendPing(0, P_ZSTART, MsgB().i64(0).i64(last).fut(0, f).finish(0));
    else if (kind == "post_before") sendPing(last, P_ZPOST, MsgB().i64(1).i64(f).finish(0));
    else if (kind == "post_after") {
      /* one PE, or the last PE in another process: zStart orders the post itself */
      if (CkNumPes() == 1 || CkNodeOf(CkNumPes() - 1) != CkMyNode())
        sendPing(0, P_ZSTART, MsgB().i64(2).i64(last).fut(0, f).finish(0));
      else
        sendPing(last, P_ZSPIN, MsgB().i64(f).finish(0));
    } else sendPing(0, P_ZSTART, MsgB().i64(0).i64(1).fut(0, f).finish(0));
  }

  /* device: 16 MB of device memory as a DEVICE slot from element 0 to the
     last element (another PE when there is more than one) and to element 1
     (element 0's PE). See dStart. */
  void runDevice(const std::string &s) {
#if CMK_CUDA || CMK_HIP
    uint32_t f = cbNewFuture([this, s](const CkDynMsgView *v) {
      CkPrintf("  %s: 16 MB device to device, checked on the host, pool block freed (%.3f ms)\n", s.c_str(), cbF64(v, 1) * 1e3);
      pass(s);
    });
    int target = s == "device/remote" ? numElems - 1 : 1;
    sendPing(0, P_DSTART, MsgB().i64(target).fut(0, f).finish(0));
#else
    CkPrintf("SKIP %s (no CUDA or HIP in this build)\n", s.c_str());
    next();
#endif
  }

  /* devpersist: persistent device buffers between element 0 and the last
     element (another PE when there is more than one) and element 1 (element
     0's PE). See pStart. */
  void runDevPersist(const std::string &s) {
#if CMK_CUDA || CMK_HIP
    uint32_t f = cbNewFuture([this, s](const CkDynMsgView *v) {
      static const char *modes[3] = { "copy in one process", "IPC", "RDMA" };
      int64_t mode = cbI64(v, 0);
      CkPrintf("  %s: 16 MB get, put, close, reopen and get, checked on the host; mode %s; "
               "16 MB get %.3f ms (including the done callback); phase %.3f ms\n",
               s.c_str(), mode >= 0 && mode < 3 ? modes[mode] : "?", cbF64(v, 1) * 1e3, cbF64(v, 2) * 1e3);
      pass(s);
    });
    int target = s == "devpersist/remote" ? numElems - 1 : 1;
    sendPing(0, P_PSTART, MsgB().i64(target).fut(0, f).finish(0));
#else
    CkPrintf("SKIP %s (no CUDA or HIP in this build)\n", s.c_str());
    next();
#endif
  }

  /* hapi: a host-device-host copy on a stream; CkDynHapiAddCallback sends to
     a future when the stream reaches the call, and the copy must be complete. */
  void runHapi() {
#if CMK_CUDA || CMK_HIP
    const size_t n = 1 << 20;
    double *h = new double[n], *back = new double[n];
    for (size_t i = 0; i < n; i++) { h[i] = 0.5 * (double)i; back[i] = -1; }
    void *d = nullptr;
    hapiStream_t st;
    hapiCheck(hapiMalloc(&d, n * sizeof(double)));
    hapiCheck(hapiStreamCreate(&st));
    hapiCheck(hapiMemcpyAsync(d, h, n * sizeof(double), hapiMemcpyHostToDevice, st));
    hapiCheck(hapiMemcpyAsync(back, d, n * sizeof(double), hapiMemcpyDeviceToHost, st));
    uint32_t f = cbNewFuture([this, h, back, d, st, n](const CkDynMsgView *v) {
      for (size_t i = 0; i < n; i++)
        CHECK(back[i] == h[i], "hapi: element %zu is %g after the stream callback, expected %g", i, back[i], h[i]);
      delete[] h; delete[] back;
      hapiCheck(hapiFree(d));
      hapiCheck(hapiStreamDestroy(st));
      CkPrintf("  hapi: %zu doubles copied host-device-host, complete when the callback ran\n", n);
      pass("hapi");
    });
    CkDynCallback cb = cbFutureCb(0, f);
    CkDynHapiAddCallback((uint64_t)(uintptr_t)st, &cb);
#else
    CkPrintf("SKIP hapi (no CUDA or HIP in this build)\n");
    next();
#endif
  }

  void runCheckpoint() {
    CHECK(g_dmain != nullptr, "checkpoint: DMain was not constructed on PE 0");
    g_dmain->s.counter = CKPT_STATE;
    /* the first section of the run (root PE 0, index 0, generation 1), for
       the -stalerestart restart run */
    if (sects.count("Aeven")) g_dmain->s.heldSid = sects["Aeven"].sid;
    g_dmain->s.fid = cbNewFuture([this](const CkDynMsgView *) {
      CkPrintf("  checkpoint: written to %s; DMain.resumed ran; restart with +restart %s\n", ckptDir.c_str(), ckptDir.c_str());
      pass("checkpoint");
    });
    CkDynCallback cb = cbTargetCb(g_dmain->t, DMain.ep(D_RESUMED), 0, CKPT_MARKER);
    if (ckptSection) {
      CkPrintf("  checkpoint with a SECTION done callback; CkDyn must abort\n");
      CkDynSectionID sid;
      memset(&sid, 0, sizeof sid);
      cb = cbSectionCb(sid, Sec.ep(SC_GOTSEC), CKPT_MARKER);
      CkDynCheckpoint((ckptDir + ".section").c_str(), &cb);
      CHECK(false, "a checkpoint with a SECTION callback did not abort");
    }
    CkDynCheckpoint(ckptDir.c_str(), &cb);
  }

  /* late: register Late on every process after start-up (after the checkpoint
     phase, so that the checkpoint holds no Late object: a restart would need
     Late registered at start-up), then create an array of it, send to one
     element, broadcast to all and reduce. Needs one PE per process. */
  void runLate() {
    if (CkMyNodeSize() != 1) {
      CkPrintf("SKIP late (%d PEs per process; late registration needs one)\n", CkMyNodeSize());
      next();
      return;
    }
    uint32_t freg = cbNewFuture([this](const CkDynMsgView *v) {
      int n = redInt(v, 0), idx = redInt(v, 1), ep = redInt(v, 2);
      CHECK(n == CkNumPes(), "late: %d PEs registered Late, expected %d", n, CkNumPes());
      CHECK(Late.chareIdx > 0 && idx == n * Late.chareIdx && ep == n * Late.firstEp,
            "late: processes disagree on Late's indices (chare %d, ep %d on PE 0)", Late.chareIdx, Late.firstEp);
      CHECK(CkDynClassOf(Late.chareIdx) != nullptr, "late: CkDynClassOf(Late) is NULL");
      CkPrintf("  late: Late registered on %d processes: chare index %d, entries %d..%d\n",
               n, Late.chareIdx, Late.firstEp, Late.ep(L_TWICE));
      lateArray();
    });
    CkDynSendGroup(gidGrp, -1, Grp.ep(G_LATEREG), MsgB().cb(cbFutureCb(0, freg)).finish(0), nullptr);
  }
  void lateArray() {
    int n = 2 * CkNumPes() + 1;
    uint32_t fc = cbNewFuture([this, n](const CkDynMsgView *) {
      uint32_t ft = cbNewFuture([this, n](const CkDynMsgView *v) {
        CHECK(cbI64(v, 0) == 42, "late: twice(21) returned %lld", (long long)cbI64(v, 0));
        const int value = 3;
        uint32_t fr = cbNewFuture([this, n, value](const CkDynMsgView *v) {
          int sum = redInt(v, 0), cnt = redInt(v, 1);
          CHECK(cnt == n, "late: %d contributions, expected %d", cnt, n);
          CHECK(sum == value * n * (n + 1) / 2, "late: sum %d, expected %d", sum, value * n * (n + 1) / 2);
          CkPrintf("  late: %d elements, point-to-point send, broadcast and reduction\n", n);
          pass("late");
        });
        CkDynSendArray(lateAid, 1, nullptr, Late.ep(L_ADD), MsgB().i64(value).cb(cbFutureCb(0, fr)).finish(0), nullptr);
      });
      int last = n - 1;
      CkDynSendArray(lateAid, 1, &last, Late.ep(L_TWICE), MsgB().flags(CKDYN_MSG_BLOCK).i64(21).fut(0, ft).finish(0), nullptr);
    });
    int dims[1] = { n };
    CkDynArrayOpts ao = { 1, dims, -1, 0, 0, -1 };
    lateAid = CkDynCreateArray(Late.chareIdx, MsgB().flags(CKDYN_MSG_CREATION).fut(0, fc).finish(Late.ep(L_CTOR)), &ao);
  }
  int lateAid = -1;
  /* AtSync on every element of the arrays created with usesAtSync (the load
     balancer starts on a PE only when all of its such elements have called
     it); `ping` checks Ping's result, `done` gets the Sec elements that moved */
  void syncAll(std::function<void(const CkDynMsgView *)> ping, std::function<void(double)> done) {
    int *left = new int(aidSecA >= 0 ? 3 : 1);
    double *moved = new double(0);
    auto fin = [left, moved, done]() { if (--*left == 0) { double m = *moved; delete left; delete moved; done(m); } };
    uint32_t fp = cbNewFuture([ping, fin](const CkDynMsgView *v) { ping(v); fin(); });
    bcastPing(P_SYNCSTART, MsgB().fut(0, fp).finish(0));
    if (aidSecA < 0) return;
    for (int k = 0; k < 2; k++) {
      int aid = k == 0 ? aidSecA : aidSecB, n = k == 0 ? nSecA : nSecB;
      uint32_t f = cbNewFuture([moved, fin, n](const CkDynMsgView *v) {
        CHECK(redDouble(v, 0) == n, "sync: %g Sec elements resumed, expected %d", redDouble(v, 0), n);
        *moved += redDouble(v, 1);
        fin();
      });
      CkDynSendArray(aid, 1, nullptr, Sec.ep(SC_SYNCSTART), MsgB().fut(0, f).finish(0), nullptr);
    }
  }

  void runMigration() {
    if (CkNumPes() == 1) { CkPrintf("SKIP migration (one PE)\n"); next(); return; }
    auto f = [this](const CkDynMsgView *v) {
      double m = numElems;
      double expect = 100.0 * VEC_LEN * m * (m - 1) / 2 + m * (VEC_LEN * (VEC_LEN - 1) / 2);
      double sum = redDouble(v, 0), moved = redDouble(v, 1), count = redDouble(v, 2);
      CHECK(count == numElems, "migration: %g elements resumed, expected %d", count, numElems);
      CHECK(sum == expect, "migration: state sum %g, expected %g", sum, expect);
      CkPrintf("  migration: %d elements resumed, %d changed PE%s\n", numElems, (int)moved,
               CkDynHasLB() ? "" : " (no LB in this build)");
    };
    syncAll(f, [this](double) { pass("migration"); });
  }

  /* ---- sections ---------------------------------------------------------- */

  static double valOf(int aid, int i) { return aid < 0 ? i + 1.0 : (aid == aidSecA ? 1.0 : 1000.0) * (i + 1); }
  void finalize(Sect &S) {
    S.size = (int)S.mem.size();
    S.sum = 0;
    for (auto &m : S.mem) S.sum += valOf(m.first, m.second);
  }
  void go() {
    if (q.empty()) { pass(cur); return; }
    std::function<void()> f = q.front();
    q.pop_front();
    f();
  }
  uint32_t doneFuture(const std::string &name) {
    return cbNewFuture([this, name](const CkDynMsgView *v) {
      CHECK(v->kind == CKDYN_MSGKIND_REDRESULT, "%s: section %s done: view kind %d", cur.c_str(), name.c_str(), (int)v->kind);
      Sect &S = sects[name];
      CkPrintf("  section %s ready: root PE %d, index %d, generation %u, %d members\n", name.c_str(),
               S.sid.root_pe, S.sid.idx, S.sid.gen, S.size);
      go();
    });
  }
  void qCreateArray(const std::string &name, int aid, std::vector<int> idx) {
    q.push_back([this, name, aid, idx] {
      Sect &S = sects[name];
      S = Sect();
      for (int i : idx) S.mem.insert({ aid, i });
      finalize(S);
      CkDynCallback d = cbFutureCb(0, doneFuture(name));
      CkDynSectionCreateArray(aid, (int)idx.size(), 1, idx.data(), 0, &d, &S.sid);
      CHECK(S.sid.root_pe == CkMyPe() && S.sid.kind == CKDYN_ARRAY && S.sid.coll_id == aid && S.sid.gen != 0,
            "%s: section id fields", name.c_str());
    });
  }
  void qCreateGroup(const std::string &name, std::vector<int> pes) {
    q.push_back([this, name, pes] {
      Sect &S = sects[name];
      S = Sect();
      S.group = true;
      for (int p : pes) S.mem.insert({ -1, p });
      finalize(S);
      CkDynCallback d = cbFutureCb(0, doneFuture(name));
      CkDynSectionCreateGroup(gidGrp, (int)pes.size(), pes.data(), 0, &d, &S.sid);
      CHECK(S.sid.kind == CKDYN_GROUP && S.sid.coll_id == gidGrp, "%s: section id fields", name.c_str());
    });
  }
  void qCombine(const std::string &name, std::vector<std::string> parts) {
    q.push_back([this, name, parts] {
      Sect &S = sects[name];
      S = Sect();
      std::vector<CkDynSectionID> ids;
      for (const std::string &p : parts) { S.mem.insert(sects[p].mem.begin(), sects[p].mem.end()); ids.push_back(sects[p].sid); }
      finalize(S);
      CkDynCallback d = cbFutureCb(0, doneFuture(name));
      CkDynSectionCombine((int)ids.size(), ids.data(), &d, &S.sid);
      CHECK(S.sid.kind == 0 && S.sid.coll_id == -1 && S.sid.root_pe == CkMyPe(), "%s: combined section id fields", name.c_str());
    });
  }
  /* One multicast of secRecv to a section and one section reduction, mode 0
     (sum_double) or 1 (foreign), to a callback of kind 0 FUTURE, 1 TARGET
     element, 2 TARGET array broadcast (array A), 3 SECTION (the section). */
  void qRound(const std::string &name, int mode, int kind) {
    q.push_back([this, name, mode, kind] {
      static const char *kinds[] = { "future", "element", "bcast", "section" };
      Sect &S = sects[name];
      std::string what = cur + ": " + name + (mode ? " foreign" : " sum_double") + " to " + kinds[kind];
      uint32_t *fp = new uint32_t(0);
      uint32_t f = cbNewFuture([this, &S, mode, kind, what, fp](const CkDynMsgView *v) {
        uint32_t f = *fp;
        delete fp;
        double val, cnt, n = 1;
        if (kind == 0) {
          CHECK(v->kind == (mode == 0 ? CKDYN_MSGKIND_REDRESULT : CKDYN_MSGKIND_DYN), "%s: view kind %d", what.c_str(), (int)v->kind);
          resultOf(v, &val, &cnt);
          CHECK(v->refnum != 0, "%s: no reference number", what.c_str());
        } else if (kind == 1) {
          val = cbF64(v, 0); cnt = cbF64(v, 1);
        } else {
          val = redDouble(v, 0); cnt = redDouble(v, 1); n = redDouble(v, 2);
        }
        double expectN = kind == 2 ? nSecA : (kind == 3 ? S.size : 1);
        CHECK(n == expectN, "%s: %g receivers of the result, expected %g", what.c_str(), n, expectN);
        if (kind == 3)
          CHECK(redDouble(v, 3) == (double)f * expectN, "%s: members received refnum sum %g, expected %u each",
                what.c_str(), redDouble(v, 3), f);
        CHECK(val == S.sum * expectN, "%s: sum %g, expected %g", what.c_str(), val, S.sum * expectN);
        if (mode == 0) CHECK(cnt == S.size * expectN, "%s: %g contributions, expected %g", what.c_str(), cnt, S.size * expectN);
        go();
      });
      *fp = f;
      CkDynCallback cb;
      if (kind == 0) cb = cbFutureCb(0, f);
      else if (kind == 1) cb = S.group ? cbTargetCb(cbGroupTarget(gidGrp, Grp.chareIdx, 0), Grp.ep(G_GOTELEM), 0, f)
                                       : cbTargetCb(cbArrayTarget1(aidSecA, Sec.chareIdx, 0), Sec.ep(SC_GOTELEM), 0, f);
      else if (kind == 2) cb = cbTargetCb(cbArrayTarget1(aidSecA, Sec.chareIdx, 0), Sec.ep(SC_GOTBCAST), 1, f);
      else cb = cbSectionCb(S.sid, S.group ? Grp.ep(G_GOTSEC) : Sec.ep(SC_GOTSEC), f);
      int expect = S.reds;
      S.reds += kind == 3 ? 2 : 1;
      int ep = S.group ? Grp.ep(G_SECRECV) : Sec.ep(SC_RECV);
      CkDynSendSection(&S.sid, ep, MsgB().i64(mode).i64(expect).cb(cb).finishMcast(ep), nullptr);
    });
  }
  /* Every element of array A (or every Grp branch) contributes with a
     SECTION callback to the section `name` (redToSec); each member of the
     section gets the result with the callback's refnum and reduces
     [value, count, 1, refnum] over the section to the future (gotSec). */
  void qRoundAll(const std::string &name, int mode) {
    q.push_back([this, name, mode] {
      static const char *modes[] = { "sum_double", "foreign", "set" };
      Sect &S = sects[name];
      std::string what = cur + ": all contribute (" + modes[mode] + ") to section " + name;
      int n = S.group ? CkNumPes() : nSecA;
      double total = 0;
      for (int i = 0; i < n; i++) total += S.group ? i + 1.0 : valOf(aidSecA, i);
      uint32_t *fp = new uint32_t(0);
      uint32_t f = cbNewFuture([this, &S, mode, what, fp, n, total](const CkDynMsgView *v) {
        uint32_t f = *fp;
        delete fp;
        double val = redDouble(v, 0), cnt = redDouble(v, 1), recv = redDouble(v, 2), refs = redDouble(v, 3);
        CHECK(recv == S.size, "%s: %g members received the result, expected %d", what.c_str(), recv, S.size);
        CHECK(refs == (double)f * S.size, "%s: members received refnum sum %g, expected %u each", what.c_str(), refs, f);
        CHECK(val == total * S.size, "%s: sum %g, expected %g on each of %d members", what.c_str(), val / S.size, total, S.size);
        if (mode != 1) CHECK(cnt == (double)n * S.size, "%s: %g contributions, expected %d", what.c_str(), cnt / S.size, n);
        CkPrintf("  %s: %d contributions reduced to %g, delivered to %d members with refnum %u\n",
                 what.c_str(), n, total, S.size, f);
        go();
      });
      *fp = f;
      S.reds += 1;   /* the members' gotSec reduction */
      CkDynCallback cb = cbSectionCb(S.sid, S.group ? Grp.ep(G_GOTSEC) : Sec.ep(SC_GOTSEC), f);
      MsgB m;
      m.i64(mode).cb(cb);
      if (S.group) CkDynSendGroup(gidGrp, -1, Grp.ep(G_REDTOSEC), m.finish(0), nullptr);
      else CkDynSendArray(aidSecA, 1, nullptr, Sec.ep(SC_REDTOSEC), m.finish(0), nullptr);
    });
  }
  void qSync() {
    q.push_back([this] {
      if (CkNumPes() == 1) { go(); return; }
      syncAll([](const CkDynMsgView *) {}, [this](double moved) {
        CHECK(moved > 0 || !requireMoves, "%s: no Sec element changed PE at the load balancing step", cur.c_str());
        CkPrintf("  %s: %d Sec elements changed PE\n", cur.c_str(), (int)moved);
        go();
      });
    });
  }

  void runSections(const std::string &s) {
    cur = s;
    q.clear();
    std::vector<int> evens, odds, allB;
    for (int i = 0; i < nSecA; i++) (i % 2 ? odds : evens).push_back(i);
    for (int i = 0; i < nSecB; i++) allB.push_back(i);
    if (s == "sections/array") {
      qCreateArray("Aeven", aidSecA, evens);
      qRound("Aeven", 0, 0); qRound("Aeven", 1, 0); qRound("Aeven", 0, 1);
      qRound("Aeven", 1, 2); qRound("Aeven", 0, 3); qRound("Aeven", 1, 3);
      /* the same SECTION callback (sid, ep) again with another refnum: one relay
         slot per PE serves both, and each round checks its own refnum arrived */
      qRound("Aeven", 0, 3);
    } else if (s == "sections/group") {
      std::vector<int> pes;
      for (int p = 0; p < CkNumPes(); p += 2) pes.push_back(p);
      qCreateGroup("G", pes);
      qRound("G", 0, 0); qRound("G", 1, 0); qRound("G", 0, 1); qRound("G", 0, 3);
    } else if (s == "sections/contribute") {
      /* reductions over a whole collection to a SECTION callback */
      for (int mode = 0; mode < 3; mode++) { qRoundAll("Aeven", mode); qRoundAll("G", mode); }
      qRound("Aeven", 0, 0);   /* the section's own reduction numbers are unaffected */
    } else if (s == "sections/remote") {
      /* created on the last PE (its root); the main chare sends from PE 0 */
      q.push_back([this] {
        int n = (nSecA + 1) / 2;
        Sect &S = sects["R"];
        S = Sect();
        for (int i = 0; i < n; i++) S.mem.insert({ aidSecA, i });
        finalize(S);
        int *left = new int(2);
        uint32_t fs = cbNewFuture([this, left](const CkDynMsgView *v) {
          CkDynViewSection(v, 0, &sects["R"].sid);
          CHECK(sects["R"].sid.root_pe == CkNumPes() - 1, "remote: root PE %d", sects["R"].sid.root_pe);
          if (--*left == 0) { delete left; go(); }
        });
        uint32_t fd = cbNewFuture([this, left](const CkDynMsgView *v) {
          CHECK(v->kind == CKDYN_MSGKIND_REDRESULT, "remote: done view kind %d", (int)v->kind);
          if (--*left == 0) { delete left; go(); }
        });
        CkDynSendGroup(gidGrp, CkNumPes() - 1, Grp.ep(G_SECCREATE),
                       MsgB().i64(n).fut(0, fs).cb(cbFutureCb(0, fd)).finish(0), nullptr);
      });
      qRound("R", 0, 0); qRound("R", 0, 3); qRound("R", 1, 0); qRound("R", 0, 1);
    } else if (s == "sections/combine_overlap") {
      /* Aeven and R overlap in the even indices of the first half */
      qCombine("C1", { "Aeven", "R" });
      qRound("C1", 0, 0); qRound("C1", 1, 0); qRound("C1", 0, 3); qRound("C1", 0, 1);
      qRound("Aeven", 0, 0); qRound("R", 0, 0);   /* the parts keep their own cookies */
    } else if (s == "sections/combine_arrays") {
      qCreateArray("Ball", aidSecB, allB);
      qCombine("C2", { "Aeven", "Ball" });
      qRound("C2", 0, 0); qRound("C2", 1, 0); qRound("C2", 0, 3); qRound("C2", 1, 3); qRound("C2", 0, 2);
    } else if (s == "sections/delete") {
      if (deleteWaiting) q.push_back([this] {
        /* W waits for R's member list from R's root (the last PE); W2 names W,
           so W's root queues W2's query until W is ready */
        CHECK(sects["R"].sid.root_pe != CkMyPe(), "-deletewaiting needs two PEs");
        CkDynSectionID ids[2] = { sects["Aeven"].sid, sects["R"].sid }, w, w2;
        CkDynSectionCombine(2, ids, nullptr, &w);
        CkDynSectionCombine(1, &w, nullptr, &w2);
        CkPrintf("  deleting a combined section that a combine waits on; CkDyn must abort\n");
        CkDynSectionDelete(&w);
        CHECK(false, "deleting a part a combine waits on did not abort");
      });
      qCreateArray("D", aidSecA, { 0, 1 });
      q.push_back([this] {
        CkDynSectionDelete(&sects["D"].sid);
        if (staleRedNo) {
          /* after quiescence the deletion has reached the members */
          uint32_t fq = cbNewFuture([this](const CkDynMsgView *) {
            CkPrintf("  a member of the deleted section asks for its reduction number; CkDyn must abort\n");
            uint32_t f = cbNewFuture([](const CkDynMsgView *) {
              CHECK(false, "CkDynSectionRedNo on a deleted section did not abort");
            });
            int one = 1;
            CkDynSendArray(aidSecA, 1, &one, Sec.ep(SC_REDNO), MsgB().sec(sects["D"].sid).fut(0, f).finish(0), nullptr);
          });
          CkDynCallback qd = cbFutureCb(0, fq);
          CkDynStartQD(&qd);
          return;
        }
        if (staleSend) {
          CkPrintf("  sending to the deleted section; CkDyn must abort\n");
          int ep = Sec.ep(SC_RECV);
          CkDynSendSection(&sects["D"].sid, ep, MsgB().i64(0).i64(-1).cb(cbFutureCb(0, 1)).finishMcast(ep), nullptr);
          CHECK(false, "a send to a deleted section did not abort");
        }
        go();
      });
      qCreateArray("D2", aidSecA, { 2 });
      q.push_back([this] {
        const CkDynSectionID &a = sects["D"].sid, &b = sects["D2"].sid;
        CHECK(a.idx == b.idx && a.gen != b.gen, "delete: the new section has index %d generation %u; "
              "expected the deleted slot %d with a new generation (was %u)", b.idx, b.gen, a.idx, a.gen);
        go();
      });
      qRound("D2", 0, 0);
    } else if (s == "sections/order") {
      /* send then delete from a PE that is not the root, many times: the
         root must apply them in that order, whatever order they arrive in */
      q.push_back([this] {
        if (CkNumPes() == 1) { CkPrintf("  %s: one PE, nothing to order\n", cur.c_str()); go(); return; }
        const int n = 32;
        MsgB b;
        b.i64(n);
        uint32_t f = cbNewFuture([this, n](const CkDynMsgView *) {
          uint32_t fq = cbNewFuture([this, n](const CkDynMsgView *) {
            CkPrintf("  %s: %d sends of %u bytes, each followed by a delete, from PE %d to root PE 0\n",
                     cur.c_str(), n, ORDER_BYTES, CkNumPes() - 1);
            go();
          });
          CkDynCallback qd = cbFutureCb(0, fq);
          CkDynStartQD(&qd);
        });
        b.fut(0, f);
        for (int i = 0; i < n; i++) {
          int idx[2] = { i % nSecA, (i + 1) % nSecA };
          CkDynSectionID sid;
          CkDynSectionCreateArray(aidSecA, 2, 1, idx, 0, nullptr, &sid);
          b.sec(sid);
        }
        CkDynSendGroup(gidGrp, CkNumPes() - 1, Grp.ep(G_SENDDELETE), b.finish(0), nullptr);
      });
    } else if (s == "sections/migration") {
      /* section reductions with RotateLB migrations between them */
      qCreateArray("Aodd", aidSecA, odds);
      qCombine("C3", { "Aodd", "Ball" });
      for (int r = 0; r < 4; r++) {
        qRound("Aodd", 0, 0); qRound("C3", 0, 0); qRound("C3", 1, 3); qRound("C2", 0, 3); qRound("G", 0, 0);
        qSync();
      }
      qRound("Aodd", 0, 0); qRound("C3", 0, 3); qRound("C1", 0, 0); qRound("R", 1, 0);
    }
    go();
  }
};

/* a charmxi array element (-xisection) */
class Plain : public CBase_Plain {
public:
  Plain() {}
  Plain(CkMigrateMessage *m) : CBase_Plain(m) {}
};

#include "ckdyntest.def.h"
