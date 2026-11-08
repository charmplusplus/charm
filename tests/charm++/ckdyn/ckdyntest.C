/*
 * CkDyn acceptance test, driven by the C++ test binding (cbinding.C).
 * TestMain is an ordinary main chare; every phase ends in a binding future
 * whose callback checks the result and calls mainProxy.nextPhase().
 * Run: ./ckdyntest +pe 4 +balancer RotateLB -requiremoves [-skip <phase>]... [-ckptdir <dir>]
 * (-requiremoves: the sections/migration phase checks that elements moved;
 * -stalesend: the sections/delete phase sends to the deleted section, which must abort)
 * The checkpoint phase writes <dir> (default ckdyntest.ckpt); then
 *      ./ckdyntest +pe 4 +restart <dir>
 * checks that the checkpoint callback reaches the restored main chare DMain.
 */
#include "ckdyntest.decl.h"
#include "cbinding.h"
#if CMK_CUDA || CMK_HIP
#include "hapi.h"
#endif
#include <cmath>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <set>
#include <string>
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

extern CbClass Ping, Sec, Grp, NGrp, Single, DMain, Late;

enum { P_CTOR, P_RECV, P_RECVI, P_STARTPING, P_COUNT, P_DORED, P_FOREIGN, P_GOTRED, P_GOTREDB, P_COMPUTE, P_CORO, P_SYNCSTART, P_RESUMED };
enum { SC_CTOR, SC_RECV, SC_GOTELEM, SC_GOTBCAST, SC_GOTSEC, SC_SYNCSTART, SC_RESUMED };
enum { G_CTOR, G_GOTRED, G_GOTREDB, G_NGSEND, G_NGREPORT, G_PRIOSTART, G_PRIORECV, G_EXPRECV,
       G_SECRECV, G_GOTELEM, G_GOTSEC, G_SECCREATE, G_LATEREG };
enum { N_CTOR, N_INCR };
enum { S_CTOR, S_GOTRED };
enum { D_CTOR, D_RESUMED };
enum { L_CTOR, L_ADD, L_TWICE };

static inline int myIndex(Obj *o) { return o->t.u.array.index[0]; }
static void sendPing(int i, int m, CkDynMsg *msg) { CkDynSendArray(aidPing, 1, &i, Ping.ep(m), msg, nullptr); }

/* ---- Ping (1-D array) ---------------------------------------------------- */

static int64_t pingCtor(Obj *, const CkDynMsgView *v) {
  CHECK(v->hdr && (v->hdr->flags & CKDYN_MSG_CREATION), "Ping constructor message lost its flags");
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
  if (v->slots[0].type == CKDYN_SLOT_I64) { *value = (double)v->slots[0].v.i64; *count = -1; return; }
  CHECK(v->slots[0].type == CKDYN_SLOT_BUFFER, "result slot 0 has type %d", v->slots[0].type);
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
} };

/* ---- Grp (group) ---------------------------------------------------------- */

static Obj *volatile g_ngObj = nullptr;   /* this process's NGrp branch (one per process) */

static int64_t grpCtor(Obj *o, const CkDynMsgView *) {
  CHECK(o->t.kind == CKDYN_GROUP && o->t.u.group.pe_or_node == CkMyPe(), "Grp constructed with a wrong target");
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

/* lateReg(CALLBACK done): register Late after start-up. The main chare runs
 * this phase only with one PE per process, so each PE registers for its
 * process; the reduction tells it that every process has. */
static int64_t grpLateReg(Obj *o, const CkDynMsgView *v) {
  cbRegisterLate(&Late);
  int r[3] = { 1, Late.chareIdx, Late.firstEp };
  contributeInts(o, r, 3, viewCb(v, 0));
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
} };

/* ---- NGrp (nodegroup) ---------------------------------------------------- */

static int64_t ngCtor(Obj *o, const CkDynMsgView *) {
  CHECK(o->t.kind == CKDYN_NODEGROUP && o->t.u.group.pe_or_node == CkMyNode(), "NGrp constructed with a wrong target");
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

static int64_t singleCtor(Obj *o, const CkDynMsgView *) {
  CHECK(o->t.kind == CKDYN_CHARE && o->t.u.chare.pe == CkMyPe(), "Single constructed with a wrong target");
  return 0;
}
static int64_t singleGotRed(Obj *o, const CkDynMsgView *v) { return gotRedTo0(o, v, "Single.gotRed"); }

CbClass Single = { "Single", CKDYN_CHARE, 0, {
  { "Single.Single", 0, singleCtor },
  { "Single.gotRed", CKDYN_EP_REDNTARGET, singleGotRed },
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
  CkDynAllowLateRegistration();   /* Late is registered by the late phase */
}

void ckdynTestInitProc(void) { cbInitPe(); }

/* ====================================================================== */
/* The main chare                                                          */
/* ====================================================================== */

static void bcastPing(int m, CkDynMsg *msg) { CkDynSendArray(aidPing, 1, nullptr, Ping.ep(m), msg, nullptr); }

class TestMain : public CBase_TestMain {
  std::set<std::string> skip;
  std::vector<std::string> steps = {
    "pingpong", "pingpong_inline", "bcast",
    "reductions/single", "reductions/single_any", "reductions/group_elem", "reductions/group_bcast",
    "reductions/array_elem", "reductions/array_bcast", "reductions/future", "reductions/foreign",
    "future_return", "nodegroup", "qd", "priority", "coroutine", "hapi",
    "sections/array", "sections/group", "sections/remote", "sections/combine_overlap",
    "sections/combine_arrays", "sections/delete", "sections/migration",
    "migration", "checkpoint", "late", "exit" };
  std::string ckptDir = "ckdyntest.ckpt";
  int step = -1;
  int pending = 0;
  double tStart = 0;
  CkDynTarget single0, singleAny;
  bool staleSend = false;   /* -stalesend: the delete phase sends to the deleted section, which must abort */
  bool requireMoves = false; /* -requiremoves: a load balancer that moves elements is loaded (RotateLB) */

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
    delete m;
    mainProxy = thisProxy;
    numElems = 2 * CkNumPes();
    CkPrintf("ckdyntest: %d PEs, %d nodes, %d Ping elements, refnum %d bytes\n",
             CkNumPes(), CkNumNodes(), numElems, CkDynRefnumBytes());
    tStart = CkWallTimer();
    CkDynCreateManagers();

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
    step++;
    CkEnforce(step < (int)steps.size());
    const std::string s = steps[step];
    if (skipped(s) && s != "exit") { CkPrintf("SKIP %s\n", s.c_str()); next(); return; }
    if (s == "pingpong" || s == "pingpong_inline") runPingpong(s, s == "pingpong" ? 0 : 1);
    else if (s == "bcast") runBcast();
    else if (s.compare(0, 11, "reductions/") == 0) runReduction(s);
    else if (s == "future_return") runFutureReturn();
    else if (s == "nodegroup") runNodegroup();
    else if (s == "qd") runQD();
    else if (s == "priority") runPriority();
    else if (s == "coroutine") runCoroutine();
    else if (s == "hapi") runHapi();
    else if (s.compare(0, 9, "sections/") == 0) runSections(s);
    else if (s == "migration") runMigration();
    else if (s == "checkpoint") runCheckpoint();
    else if (s == "late") runLate();
    else if (s == "exit") { CkPrintf("ALL DONE (%.3f s)\n", CkWallTimer() - tStart); CkDynExit(0); }
  }

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
    g_dmain->s.fid = cbNewFuture([this](const CkDynMsgView *) {
      CkPrintf("  checkpoint: written to %s; DMain.resumed ran; restart with +restart %s\n", ckptDir.c_str(), ckptDir.c_str());
      pass("checkpoint");
    });
    CkDynCallback cb = cbTargetCb(g_dmain->t, DMain.ep(D_RESUMED), 0, CKPT_MARKER);
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
      qCreateArray("D", aidSecA, { 0, 1 });
      q.push_back([this] {
        CkDynSectionDelete(&sects["D"].sid);
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

#include "ckdyntest.def.h"
