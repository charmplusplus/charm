/*
 * futures: the futures and semaphore API of ckfutures.C, and [sync]/[iget]
 * entry methods.
 *
 * Every PE runs one test thread (futures_worker::run). Its partners are the
 * next two PEs (itself on one PE). Each thread:
 *   - testIds: CkFuture / CkFutureID calls. Some futures are filled (by a
 *     partner) before the wait starts, checked with CkProbeFutureID; others
 *     are requested just before the wait, so their value arrives while the
 *     thread is suspended (the fill runs on this PE only after the thread
 *     suspends). CkWaitAllIDs over a mixed set, CkWaitFuture,
 *     CkWaitReleaseFuture, CkWaitVoidFuture, release and id reuse.
 *   - testWaitAny: CkWaitAnyID with one future ready, and with none ready.
 *   - testTyped: ck::future<int>: get, wait_all, wait_any, release,
 *     operator==, pup (passed as an entry method parameter); testWaitSome:
 *     wait_some (with none and with some futures ready).
 *   - testCollectiveValues (PE 0): futures filled through a CkCallback with a
 *     CkDataMsg, a CkReductionMsg and a CkArrayCreatedMsg (the other cases
 *     of CkGetMsgBuffer).
 *   - testSync: [sync] entry methods of a chare, a group and a nodegroup
 *     (CkRemoteCall, CkRemoteBranchCall, CkRemoteNodeBranchCall), and the
 *     Async forms called directly, waited on together with CkWaitAllIDs.
 *   - testArray: [sync] and [iget] array entry methods, and
 *     CkCreateAttachedFuture / CkCreateAttachedFutureSend called directly.
 *   - testSema: CkSemaCreate, signals from several PEs, CkSemaWait,
 *     CkSemaWaitN, CkSemaDestroy, and a second semaphore that reuses the slot.
 * Every value is checked; a mismatch aborts naming the step.
 */
#include <algorithm>
#include <vector>
#include "futures.h"

#define FUTURES_POLL_LIMIT 30.0  // seconds before a poll loop gives up

static void futures_check(int got, int want, const char* what)
{
  if (got != want)
    CkAbort("futures: PE %d %s: got %d, expected %d\n", CkMyPe(), what, got, want);
}

// Yield until ready() holds; used to make a value arrive before a wait starts.
template <class F>
static void futures_pollUntil(F ready, const char* what)
{
  double start = CkWallTimer();
  while (!ready())
  {
    if (CkWallTimer() - start > FUTURES_POLL_LIMIT)
      CkAbort("futures: PE %d: %s never became ready\n", CkMyPe(), what);
    CthYield();
  }
}

static int futures_msgVal(void* m, const char* what)
{
  if (m == NULL) CkAbort("futures: PE %d %s: NULL message\n", CkMyPe(), what);
  return ((futures_msg*)m)->val;
}

/******************** driver ********************/

futures_driver::futures_driver(void)
{
  CProxy_futures_helper helper = CProxy_futures_helper::ckNew();
  CProxy_futures_nodehelper nodehelper = CProxy_futures_nodehelper::ckNew();
  int nArr = 2 * CkNumPes() + 1;
  CProxy_futures_arr arr = CProxy_futures_arr::ckNew(nArr);
  CProxy_futures_worker workers =
      CProxy_futures_worker::ckNew(thisProxy, helper, nodehelper, arr, nArr);
  workers.run();
}

void futures_driver::done(int nPes)
{
  futures_check(nPes, CkNumPes(), "threads finished");
  megatest_finish();
  delete this;
}

/******************** worker ********************/

void futures_worker::run(void)
{
  me = CkMyPe();
  n = CkNumPes();
  q1 = (me + 1) % n;
  q2 = (me + 2) % n;
  base = 1000 * (me + 1);
  testIds();
  testWaitAny();
  testTyped();
  testWaitSome();
  if (me == 0) testCollectiveValues();
  testSync();
  testArray();
  testSema();
  int one = 1;
  contribute(sizeof(int), &one, CkReduction::sum_int,
             CkCallback(CkReductionTarget(futures_driver, done), driver));
}

void futures_worker::testIds(void)
{
  CkFuture f[6];
  for (int i = 0; i < 6; i++)
  {
    f[i] = CkCreateFuture();
    futures_check(f[i].pe, me, "CkCreateFuture pe");
    futures_check(CkProbeFutureID(f[i].id), 0, "probe before fill");
    futures_check(CkProbeFuture(f[i]), 0, "CkProbeFuture before fill");
  }

  // f0, f1 filled before the wait; f2, f3 while the thread waits.
  helper[q1].fill(f[0], base + 0);
  helper[q2].fillId(f[1].id, me, base + 1);
  futures_pollUntil([&] { return CkProbeFutureID(f[0].id) && CkProbeFuture(f[1]); },
                    "pre-filled futures");
  helper[q2].fill(f[2], base + 2);
  helper[q1].fillId(f[3].id, me, base + 3);
  std::vector<CkFutureID> ids = {f[2].id, f[0].id, f[3].id, f[1].id};
  std::vector<void*> vals = CkWaitAllIDs(ids);
  futures_check((int)vals.size(), 4, "CkWaitAllIDs size");
  const int want[] = {2, 0, 3, 1};
  for (int i = 0; i < 4; i++)
  {
    futures_check(futures_msgVal(vals[i], "CkWaitAllIDs"), base + want[i], "CkWaitAllIDs value");
    futures_check(CkProbeFutureID(ids[i]), 1, "probe after CkWaitAllIDs");
  }
  // CkWaitFuture on a future that already holds a value returns it at once.
  futures_check(futures_msgVal(CkWaitFuture(f[0]), "CkWaitFuture"), base, "CkWaitFuture ready");
  for (int i = 0; i < 4; i++) CkFreeMsg(vals[i]);
  CkReleaseFuture(f[0]);
  for (int i = 1; i < 4; i++) CkReleaseFutureID(f[i].id);
  futures_check(CkProbeFutureID(f[1].id), 0, "probe after release");

  // CkWaitAllIDs with every value present: no suspension.
  helper[q1].fill(f[4], base + 4);
  helper[q2].fill(f[5], base + 5);
  futures_pollUntil([&] { return CkProbeFuture(f[4]) && CkProbeFuture(f[5]); }, "f4, f5");
  std::vector<CkFutureID> ready = {f[4].id, f[5].id};
  vals = CkWaitAllIDs(ready);
  futures_check(futures_msgVal(vals[0], "all ready"), base + 4, "CkWaitAllIDs all ready 0");
  futures_check(futures_msgVal(vals[1], "all ready"), base + 5, "CkWaitAllIDs all ready 1");
  CkFreeMsg(vals[0]);
  CkFreeMsg(vals[1]);
  CkReleaseFutureID(f[4].id);
  CkReleaseFutureID(f[5].id);
  // The next CkWaitAllIDs that has to wait returns when its value is there.
  CkFuture w = CkCreateFuture();
  helper[q1].fill(w, base + 8);
  std::vector<CkFutureID> wid = {w.id};
  vals = CkWaitAllIDs(wid);
  futures_check(futures_msgVal(vals[0], "CkWaitAllIDs after an all-ready call"), base + 8,
                "CkWaitAllIDs after an all-ready call");
  CkFreeMsg(vals[0]);
  CkReleaseFuture(w);

  // Released ids are reused. CkWaitFuture suspends until the value arrives;
  // CkWaitReleaseFuture waits and releases.
  CkFuture g = CkCreateFuture();
  helper[q1].fill(g, base + 6);
  void* m = CkWaitFuture(g);
  futures_check(futures_msgVal(m, "CkWaitFuture"), base + 6, "CkWaitFuture value");
  CkFreeMsg(m);
  CkReleaseFuture(g);

  CkFuture h = CkCreateFuture();
  helper[q2].fillId(h.id, me, base + 7);
  m = CkWaitReleaseFuture(h.id);
  futures_check(futures_msgVal(m, "CkWaitReleaseFuture"), base + 7, "CkWaitReleaseFuture value");
  CkFreeMsg(m);
  futures_check(CkProbeFutureID(h.id), 0, "probe after CkWaitReleaseFuture");

  // A future filled with an empty system message, as a void [sync] reply is.
  CkFuture v = CkCreateFuture();
  helper[q1].fillVoid(v);
  CkWaitVoidFuture(v.id);
  CkReleaseFutureID(v.id);

  // Released without ever being filled.
  CkFuture u = CkCreateFuture();
  CkReleaseFuture(u);
}

void futures_worker::testWaitAny(void)
{
  // One future ready: returned without suspending.
  CkFuture a = CkCreateFuture(), b = CkCreateFuture();
  helper[q1].fill(b, base + 10);
  futures_pollUntil([&] { return CkProbeFuture(b); }, "wait-any ready future");
  std::vector<CkFutureID> ids = {a.id, b.id};
  std::pair<void*, CkFutureID> r = CkWaitAnyID(ids);
  futures_check(r.second, b.id, "CkWaitAnyID ready id");
  futures_check(futures_msgVal(r.first, "CkWaitAnyID"), base + 10, "CkWaitAnyID ready value");
  CkFreeMsg(r.first);
  CkReleaseFutureID(b.id);
  helper[q2].fill(a, base + 11);
  std::vector<CkFutureID> one = {a.id};
  r = CkWaitAnyID(one);
  futures_check(r.second, a.id, "CkWaitAnyID single id");
  futures_check(futures_msgVal(r.first, "CkWaitAnyID"), base + 11, "CkWaitAnyID single value");
  CkFreeMsg(r.first);
  CkReleaseFutureID(a.id);

  // None ready: each call returns one, until all three are collected.
  CkFuture c[3];
  std::vector<CkFutureID> rest;
  for (int i = 0; i < 3; i++)
  {
    c[i] = CkCreateFuture();
    rest.push_back(c[i].id);
  }
  for (int i = 0; i < 3; i++) helper[(me + i) % n].fill(c[i], base + 20 + i);
  int seen = 0;
  while (!rest.empty())
  {
    r = CkWaitAnyID(rest);
    int i = -1;
    for (int k = 0; k < 3; k++)
      if (c[k].id == r.second) i = k;
    if (i < 0) CkAbort("futures: PE %d CkWaitAnyID returned unknown id %d\n", me, (int)r.second);
    futures_check(futures_msgVal(r.first, "CkWaitAnyID"), base + 20 + i, "CkWaitAnyID value");
    if (seen & (1 << i)) CkAbort("futures: PE %d CkWaitAnyID returned id %d twice\n", me, i);
    seen |= 1 << i;
    rest.erase(std::find(rest.begin(), rest.end(), r.second));
    CkFreeMsg(r.first);
    CkReleaseFutureID(r.second);
  }
  futures_check(seen, 7, "CkWaitAnyID coverage");
}

void futures_worker::testTyped(void)
{
  typedef ck::future<int> fut;

  // get() on a value filled from another PE; copy, operator== and pup.
  fut a;
  fut a2(a);
  if (!(a == a2) || !a.is_local() || a.is_ready())
    CkAbort("futures: PE %d ck::future copy/is_local/is_ready wrong\n", me);
  fut b(a.handle());
  if (!(b == a)) CkAbort("futures: PE %d ck::future from handle differs\n", me);
  helper[q1].fillTyped(a, base + 30);
  futures_check(a.get(), base + 30, "ck::future get");
  if (!a.is_ready()) CkAbort("futures: PE %d ck::future not ready after get\n", me);
  a.release();

  // wait_all: two filled before, two during the wait.
  std::vector<fut> fs(4);
  helper[q1].fillTyped(fs[0], base + 40);
  helper[q2].fillTyped(fs[1], base + 41);
  futures_pollUntil([&] { return fs[0].is_ready() && fs[1].is_ready(); }, "wait_all pre-fill");
  helper[q2].fillTyped(fs[2], base + 42);
  helper[q1].fillTyped(fs[3], base + 43);
  std::vector<int> all = ck::wait_all(fs.begin(), fs.end());
  futures_check((int)all.size(), 4, "wait_all size");
  for (int i = 0; i < 4; i++) futures_check(all[i], base + 40 + i, "wait_all value");
  for (auto& f : fs) f.release();

  // wait_any until empty.
  std::vector<fut> anyv(3);
  std::vector<int> wantAny;
  for (int i = 0; i < 3; i++)
  {
    helper[(me + i) % n].fillTyped(anyv[i], base + 50 + i);
    wantAny.push_back(base + 50 + i);
  }
  std::vector<std::pair<fut, int>> expect;
  for (int i = 0; i < 3; i++) expect.push_back(std::make_pair(anyv[i], base + 50 + i));
  while (!anyv.empty())
  {
    auto r = ck::wait_any(anyv.begin(), anyv.end());
    if (r.second == anyv.end()) CkAbort("futures: PE %d wait_any returned end\n", me);
    int want = -1;
    for (auto& e : expect)
      if (e.first == *r.second) want = e.second;
    futures_check(r.first, want, "wait_any value");
    r.second->release();
    anyv.erase(r.second);
  }

}

void futures_worker::testWaitSome(void)
{
  typedef ck::future<int> fut;
  std::vector<fut> some(4);
  // None ready: nothing returned, every future left pending.
  auto none = ck::wait_some(some.begin(), some.end());
  futures_check((int)(none.second - some.begin()), 0, "wait_some with none ready");
  futures_check((int)none.first.size(), 0, "wait_some values with none ready");
  // Two of four ready: wait_some returns those and leaves the others (it calls
  // CkWaitAllIDs on the ready ones).
  helper[q1].fillTyped(some[1], base + 61);
  helper[q2].fillTyped(some[3], base + 63);
  futures_pollUntil([&] { return some[1].is_ready() && some[3].is_ready(); }, "wait_some pre-fill");
  auto part = ck::wait_some(some.begin(), some.end());
  futures_check((int)(part.second - some.begin()), 2, "wait_some ready count");
  futures_check((int)part.first.size(), 2, "wait_some values");
  futures_check(part.first[0] + part.first[1], 2 * base + 124, "wait_some sum");
  for (auto it = part.second; it != some.end(); ++it)
  {
    if (it->is_ready()) CkAbort("futures: PE %d wait_some left a ready future pending\n", me);
    helper[q1].fillTyped(*it, base + 70);
  }
  std::vector<int> late = ck::wait_all(part.second, some.end());
  for (int x : late) futures_check(x, base + 70, "wait_some remainder");
  for (auto& f : some) f.release();
}

void futures_worker::testCollectiveValues(void)
{
  // CkDataMsg: a CkCallback to a future, sent with (length, data).
  ck::future<int> d;
  helper[q1].fillData(CkCallback(d.handle()), base + 80);
  futures_check(d.get(), base + 80, "future from CkDataMsg");
  d.release();

  // CkReductionMsg: a sum reduction whose callback is a future.
  ck::future<int> r;
  helper.reduceTo(r, 7);
  futures_check(r.get(), 7 * n + n * (n - 1) / 2, "future from reduction");
  r.release();

  // CkArrayCreatedMsg: asynchronous array creation.
  ck::future<CkArrayID> created;
  CProxy_futures_arr::ckNew(3, CkCallback(created.handle()));
  CProxy_futures_arr fresh(created.get());
  created.release();
  futures_msg* m = fresh[2].twice(new futures_msg(5));
  futures_check(m->val, 11, "[sync] on asynchronously created array");
  futures_check(m->where, 2, "[sync] on asynchronously created array index");
  delete m;
  fresh.ckDestroy();
}

void futures_worker::testSync(void)
{
  int node = (CkMyNode() + 1) % CkNumNodes();
  CProxy_futures_chare c = CProxy_futures_chare::ckNew(q1);

  // Generated [sync] proxies: CkRemoteCall, CkRemoteBranchCall, CkRemoteNodeBranchCall.
  futures_msg* m = c.twice(new futures_msg(base + 1));
  futures_check(m->val, 2 * (base + 1) + 1, "chare [sync] value");
  futures_check(m->where, q1, "chare [sync] PE");
  delete m;
  futures_check(c.add(base, 3), base + 3, "chare [sync] marshalled");
  m = helper[q2].twice(new futures_msg(base + 2));
  futures_check(m->val, 2 * (base + 2) + 1, "group [sync] value");
  futures_check(m->where, q2, "group [sync] PE");
  delete m;
  futures_check(helper[q1].add(base, 4), base + 4, "group [sync] marshalled");
  m = nodehelper[node].twice(new futures_msg(base + 3));
  futures_check(m->val, 2 * (base + 3) + 1, "nodegroup [sync] value");
  futures_check(m->where, node, "nodegroup [sync] node");
  delete m;
  futures_check(nodehelper[node].add(base, 5), base + 5, "nodegroup [sync] marshalled");

  // The Async forms called directly, all outstanding at once.
  std::vector<CkFutureID> ids;
  ids.push_back(CkRemoteCallAsync(CkIndex_futures_chare::twice((futures_msg*)NULL),
                                  new futures_msg(base + 11), &c.ckGetChareID()));
  ids.push_back(CkRemoteBranchCallAsync(CkIndex_futures_helper::twice((futures_msg*)NULL),
                                        new futures_msg(base + 12), helper.ckGetGroupID(), q2));
  ids.push_back(CkRemoteNodeBranchCallAsync(CkIndex_futures_nodehelper::twice((futures_msg*)NULL),
                                            new futures_msg(base + 13),
                                            nodehelper.ckGetGroupID(), node));
  std::vector<void*> vals = CkWaitAllIDs(ids);
  const int where[] = {q1, q2, node};
  for (int i = 0; i < 3; i++)
  {
    futures_msg* r = (futures_msg*)vals[i];
    futures_check(r->val, 2 * (base + 11 + i) + 1, "Async call value");
    futures_check(r->where, where[i], "Async call target");
    delete r;
    CkReleaseFutureID(ids[i]);
  }

  // The blocking forms called directly.
  m = (futures_msg*)CkRemoteCall(CkIndex_futures_chare::twice((futures_msg*)NULL),
                                 new futures_msg(base + 21), &c.ckGetChareID());
  futures_check(m->val, 2 * (base + 21) + 1, "CkRemoteCall value");
  delete m;
  m = (futures_msg*)CkRemoteBranchCall(CkIndex_futures_helper::twice((futures_msg*)NULL),
                                       new futures_msg(base + 22), helper.ckGetGroupID(), q1);
  futures_check(m->val, 2 * (base + 22) + 1, "CkRemoteBranchCall value");
  futures_check(m->where, q1, "CkRemoteBranchCall PE");
  delete m;
  m = (futures_msg*)CkRemoteNodeBranchCall(
      CkIndex_futures_nodehelper::twice((futures_msg*)NULL), new futures_msg(base + 23),
      nodehelper.ckGetGroupID(), node);
  futures_check(m->val, 2 * (base + 23) + 1, "CkRemoteNodeBranchCall value");
  futures_check(m->where, node, "CkRemoteNodeBranchCall node");
  delete m;
}

static void futures_sendToElement(CkArrayID aid, CkArrayIndex idx, void* msg, int ep, int opts)
{
  CkSendMsgArray(ep, msg, aid, idx, opts);
}

void futures_worker::testArray(void)
{
  int i = (2 * me + 1) % nArr;
  int j = (2 * me + 2) % nArr;

  // [sync]: ckSendSync, CkCreateAttachedFuture.
  futures_msg* m = arr[i].twice(new futures_msg(base + 31));
  futures_check(m->val, 2 * (base + 31) + 1, "array [sync] value");
  futures_check(m->where, i, "array [sync] index");
  delete m;

  // [iget]: the proxy returns the future (CkCreateAttachedFutureSend).
  CkFutureID f1 = arr[j].igetTwice(new futures_msg(base + 32));
  CkFutureID f2 = arr[i].igetVoid(base);
  m = (futures_msg*)CkWaitReleaseFuture(f1);
  futures_check(m->val, 2 * (base + 32) + 1, "array [iget] value");
  futures_check(m->where, j, "array [iget] index");
  delete m;
  CkWaitVoidFuture(f2);
  CkReleaseFutureID(f2);

  // Called directly.
  int ep = CkIndex_futures_arr::twice((futures_msg*)NULL);
  futures_msg* in = new futures_msg(base + 33);
  CkFutureID f3 = CkCreateAttachedFuture(in);
  futures_check(CkGetRefNum(in), f3, "attached future refnum");
  CkSendMsgArray(ep, in, arr.ckGetArrayID(), CkArrayIndex1D(j));
  in = new futures_msg(base + 34);
  CkFutureID f4 = CkCreateAttachedFutureSend(in, ep, arr.ckGetArrayID(), CkArrayIndex1D(i),
                                             futures_sendToElement, 0);
  m = (futures_msg*)CkWaitReleaseFuture(f4);
  futures_check(m->val, 2 * (base + 34) + 1, "CkCreateAttachedFutureSend value");
  futures_check(m->where, i, "CkCreateAttachedFutureSend index");
  delete m;
  m = (futures_msg*)CkWaitReleaseFuture(f3);
  futures_check(m->val, 2 * (base + 33) + 1, "CkCreateAttachedFuture value");
  futures_check(m->where, j, "CkCreateAttachedFuture index");
  delete m;
}

void futures_worker::testSema(void)
{
  // Four signals from four PEs (fewer distinct ones on small runs).
  CkSemaID s = CkSemaCreate();
  futures_check(s.pe, me, "CkSemaCreate pe");
  for (int k = 0; k < 4; k++) helper[(me + k) % n].signal(s, base + 90 + k);
  void* first = CkSemaWait(s);
  void* rest[3];
  CkSemaWaitN(s, 3, rest);
  int seen = 0;
  for (int k = 0; k < 4; k++)
  {
    int v = futures_msgVal(k == 0 ? first : rest[k - 1], "semaphore") - (base + 90);
    if (v < 0 || v > 3 || (seen & (1 << v)))
      CkAbort("futures: PE %d semaphore delivered unexpected value %d\n", me, v + base + 90);
    seen |= 1 << v;
    CkFreeMsg(k == 0 ? first : rest[k - 1]);
  }
  futures_check(seen, 15, "semaphore values");
  CkSemaDestroy(s);

  // A second semaphore takes the freed slot; its messages queue before the waits.
  CkSemaID t = CkSemaCreate();
  futures_check(t.idx, s.idx, "semaphore slot reuse");
  helper[q1].signal(t, base + 95);
  helper[q2].signal(t, base + 96);
  CkSemaWaitN(t, 2, rest);
  int sum = futures_msgVal(rest[0], "semaphore 2") + futures_msgVal(rest[1], "semaphore 2");
  futures_check(sum, 2 * base + 191, "semaphore 2 values");
  CkFreeMsg(rest[0]);
  CkFreeMsg(rest[1]);
  helper[q1].signal(t, base + 97);
  // Give the signal time to arrive so this wait finds it queued.
  double start = CkWallTimer();
  while (CkWallTimer() - start < 0.002) CthYield();
  void* last = CkSemaWait(t);
  futures_check(futures_msgVal(last, "semaphore 2"), base + 97, "semaphore 2 last");
  CkFreeMsg(last);
  CkSemaDestroy(t);
}

/******************** helpers ********************/

void futures_helper::fill(CkFuture f, int val) { CkSendToFuture(f, new futures_msg(val, CkMyPe())); }

void futures_helper::fillId(CkFutureID id, int pe, int val)
{
  CkSendToFutureID(id, new futures_msg(val, CkMyPe()), pe);
}

void futures_helper::fillVoid(CkFuture f) { CkSendToFuture(f, CkAllocSysMsg()); }

void futures_helper::fillTyped(ck::future<int> f, int val)
{
  if (f.is_local() != (f.handle().pe == CkMyPe()))
    CkAbort("futures: PE %d ck::future::is_local wrong\n", CkMyPe());
  f.set(val);
}

void futures_helper::fillData(CkCallback cb, int val) { cb.send(sizeof(int), &val); }

void futures_helper::signal(CkSemaID s, int val) { CkSemaSignal(s, new futures_msg(val, CkMyPe())); }

void futures_helper::reduceTo(ck::future<int> f, int val)
{
  int mine = val + CkMyPe();
  contribute(sizeof(int), &mine, CkReduction::sum_int, CkCallback(f.handle()));
}

futures_msg* futures_helper::twice(futures_msg* m)
{
  futures_msg* r = new futures_msg(2 * m->val + 1, CkMyPe());
  delete m;
  return r;
}

int futures_helper::add(int a, int b) { return a + b; }

futures_msg* futures_nodehelper::twice(futures_msg* m)
{
  futures_msg* r = new futures_msg(2 * m->val + 1, CkMyNode());
  delete m;
  return r;
}

int futures_nodehelper::add(int a, int b) { return a + b; }

futures_msg* futures_chare::twice(futures_msg* m)
{
  futures_msg* r = new futures_msg(2 * m->val + 1, CkMyPe());
  delete m;
  return r;
}

int futures_chare::add(int a, int b) { return a + b; }

futures_msg* futures_arr::twice(futures_msg* m)
{
  futures_msg* r = new futures_msg(2 * m->val + 1, thisIndex);
  delete m;
  return r;
}

futures_msg* futures_arr::igetTwice(futures_msg* m) { return twice(m); }

void futures_arr::igetVoid(int x) {}

/******************** megatest hooks ********************/

void futures_init(void) { CProxy_futures_driver::ckNew(0); }

void futures_moduleinit(void) {}

MEGATEST_REGISTER_TEST(futures, "kale", 0)
#include "futures.def.h"
