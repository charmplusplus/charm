/*
 * callbackkinds: every kind of CkCallback that can be constructed, sent from
 * another PE after travelling as a marshalled parameter (so CkCallback::pup
 * runs on both sides).
 *
 * The driver (one thread on PE 0) builds a callback, passes it to the
 * reflector group on some PE, and the reflector sends it in one of three ways:
 * with a message, with no message (send()), or with data (send(length, data),
 * which builds a CkDataMsg). Each target entry method reports the delivery;
 * the driver checks the kind, the value, the refnum when one was set, and that
 * every expected PE/node/element received exactly one delivery (CkNumPes for a
 * group broadcast, CkNumNodes for a nodegroup broadcast, every element for an
 * array broadcast). A fourth mode sends the callback back unchanged, and the
 * driver compares the copy with operator==.
 *
 * Kinds: chare, group element, group broadcast, nodegroup element, nodegroup
 * broadcast, array element, array broadcast, array section (sent from PE 0;
 * see the CALLBACKKINDS_* switches in callbackkinds.h), each through every
 * constructor that makes it (including the inline forms and the Chare*,
 * Group*, NodeGroup* and ArrayElement* constructors, built inside the
 * target), C function (callCFn, forwarded back to PE 0), call1Fn (runs where
 * it is sent; sent only within the process of PE 0, since it carries a
 * function pointer), resumeThread (plain, CkCallbackResumeThread with and
 * without a result pointer), future, and ignore. invalid, ckExit and replyCCS are built and checked but never sent.
 */
#include <string.h>
#include "callbackkinds.h"

#define CALLBACKKINDS_WAIT_LIMIT 30.0  // seconds

// call1Fn callbacks report to the driver through this proxy. It is written on
// PE 0 before any such callback exists, and they are sent only from PEs of
// PE 0's process.
static CProxy_callbackkinds_driver callbackkinds_c1Driver;

static void callbackkinds_decode(void* msg, int& val, int& refnum)
{
  if (msg == NULL)
  {
    val = -1;
    refnum = -1;
    return;
  }
  refnum = CkGetRefNum(msg);
  if (UsrToEnv(msg)->getMsgIdx() == CMessage_CkDataMsg::__idx)
  {
    CkDataMsg* d = (CkDataMsg*)msg;
    d->check();
    if (d->getLength() != (int)sizeof(int))
      CkAbort("callbackkinds: CkDataMsg of %d bytes, expected %d\n", d->getLength(),
              (int)sizeof(int));
    memcpy(&val, d->getData(), sizeof(int));
  }
  else
    val = ((callbackkinds_msg*)msg)->val;
}

void callbackkinds_report(CProxy_callbackkinds_driver d, int kind, int where, void* msg)
{
  int val, refnum;
  callbackkinds_decode(msg, val, refnum);
  if (msg) CkFreeMsg(msg);
  d.got(kind, val, where, refnum);
}

static void callbackkinds_cfn(void* param, void* msg)
{
  if (CkMyPe() != 0) CkAbort("callbackkinds: callCFn ran on PE %d, not 0\n", CkMyPe());
  int val, refnum;
  callbackkinds_decode(msg, val, refnum);
  if (msg) CkFreeMsg(msg);
  ((callbackkinds_driver*)param)->got(CBK_CFN, val, 0, refnum);
}

static void callbackkinds_c1fn(void* msg)
{
  int val, refnum;
  callbackkinds_decode(msg, val, refnum);
  if (msg) CkFreeMsg(msg);
  callbackkinds_c1Driver.got(CBK_C1FN, val, CkMyPe(), refnum);
}

/******************** reflector ********************/

void callbackkinds_reflector::reflect(CkCallback cb, int mode, int val)
{
  switch (mode)
  {
    case CBK_MSG: cb.send(new callbackkinds_msg(val)); break;
    case CBK_NOMSG: cb.send(); break;
    case CBK_DATA: cb.send(sizeof(int), &val); break;
    case CBK_ECHO: driver.echo(cb); break;
  }
  driver.reflected();
}

/******************** driver ********************/

callbackkinds_driver::callbackkinds_driver(void)
    : step(1), what(""), expKind(0), expVal(0), expRefnum(-1), expTotal(0), nGot(0),
      nReflected(0), nEchoed(0)
{
  callbackkinds_c1Driver = thisProxy;
  nArr = CkNumPes() + 2;
  charePe = CkNumPes() - 1;
  reflector = CProxy_callbackkinds_reflector::ckNew(thisProxy);
  chare = CProxy_callbackkinds_chare::ckNew(thisProxy, charePe);
  group = CProxy_callbackkinds_group::ckNew(thisProxy);
  nodegroup = CProxy_callbackkinds_nodegroup::ckNew(thisProxy);
  array = CProxy_callbackkinds_array::ckNew(thisProxy, nArr);
  thisProxy.run();
}

void callbackkinds_driver::got(int kind, int val, int where, int refnum)
{
  if (kind != expKind || val != expVal)
    CkAbort("callbackkinds: %s: delivery of kind %d value %d, expected kind %d value %d\n", what,
            kind, val, expKind, expVal);
  if (expRefnum >= 0 && refnum != -1 && refnum != expRefnum)
    CkAbort("callbackkinds: %s: refnum %d, expected %d\n", what, refnum, expRefnum);
  if (where < 0 || where >= (int)whereCount.size())
    CkAbort("callbackkinds: %s: delivery at %d, outside 0..%d\n", what, where,
            (int)whereCount.size() - 1);
  whereCount[where]++;
  nGot++;
}

void callbackkinds_driver::echo(CkCallback cb)
{
  echoed = cb;
  nEchoed++;
}

void callbackkinds_driver::expect(int kind, int val, int refnum, const std::vector<int>& where)
{
  expKind = kind;
  expVal = val;
  expRefnum = refnum;
  expWhere = where;
  whereCount.assign(where.size(), 0);
  expTotal = 0;
  for (int c : where) expTotal += c;
  nGot = 0;
  nReflected = 0;
}

// Yields until every expected delivery and nReflect reflector reports came in.
void callbackkinds_driver::waitFor(int nReflect)
{
  double start = CkWallTimer();
  while (nGot < expTotal || nReflected < nReflect)
  {
    if (CkWallTimer() - start > CALLBACKKINDS_WAIT_LIMIT)
      CkAbort("callbackkinds: %s: %d of %d deliveries, %d of %d reflections after %.0f s\n",
              what, nGot, expTotal, nReflected, nReflect, CALLBACKKINDS_WAIT_LIMIT);
    CthYield();
  }
  for (size_t i = 0; i < expWhere.size(); i++)
    if (whereCount[i] != expWhere[i])
      CkAbort("callbackkinds: %s: %d deliveries at %d, expected %d\n", what, whereCount[i],
              (int)i, expWhere[i]);
}

// Sends cb to a reflector and back; the copy must compare equal.
void callbackkinds_driver::echoCheck(CkCallback& cb, int pe)
{
  int before = nEchoed;
  nReflected = 0;
  expTotal = 0;
  nGot = 0;
  reflector[pe].reflect(cb, CBK_ECHO, 0);
  double start = CkWallTimer();
  while (nEchoed == before || nReflected < 1)
  {
    if (CkWallTimer() - start > CALLBACKKINDS_WAIT_LIMIT)
      CkAbort("callbackkinds: %s: echo never came back\n", what);
    CthYield();
  }
  if (!(echoed == cb) || echoed.type != cb.type)
    CkAbort("callbackkinds: %s: callback changed in a pup round trip (type %d -> %d)\n", what,
            (int)cb.type, (int)echoed.type);
  echoed = CkCallback();
}

void callbackkinds_driver::viaReflector(const char* name, CkCallback cb, int kind, int mode,
                                        int val, int refnum, const std::vector<int>& where,
                                        int pe)
{
  what = name;
  expect(kind, mode == CBK_NOMSG ? -1 : val, refnum, where);
  reflector[pe].reflect(cb, mode, val);
  waitFor(1);
}

// The same target through each send mode; make() builds the callback for an ep.
void callbackkinds_driver::threeModes(const char* name, int kind, int caseNo, int refnum,
                                      const std::vector<int>& where,
                                      CkCallback (*make)(callbackkinds_driver*, int ep),
                                      const int* eps)
{
  for (int mode = CBK_MSG; mode <= CBK_DATA; mode++)
  {
    CkCallback cb = make(this, eps[mode]);
    if (refnum >= 0) cb.setRefNum(refnum);
    if (cb.epIndex() != eps[mode])
      CkAbort("callbackkinds: %s: epIndex %d, expected %d\n", name, cb.epIndex(), eps[mode]);
    if (!cb.requiresMsgConstruction() || cb.isInvalid())
      CkAbort("callbackkinds: %s: entry-method callback reports no message needed\n", name);
    what = name;
    echoCheck(cb, nextPe());
    viaReflector(name, cb, kind, mode, 100 * caseNo + mode, refnum, where, nextPe());
  }
}

void callbackkinds_driver::checkAccessors(const CkCallback& cb, CkCallback::callbackType type,
                                          bool pointer, int ep)
{
  if (cb.type != type)
    CkAbort("callbackkinds: %s: type %d, expected %d\n", what, (int)cb.type, (int)type);
  if (cb.containsPointer() != pointer)
    CkAbort("callbackkinds: %s: containsPointer %d, expected %d\n", what,
            (int)cb.containsPointer(), (int)pointer);
  if (cb.epIndex() != ep)
    CkAbort("callbackkinds: %s: epIndex %d, expected %d\n", what, cb.epIndex(), ep);
}

static std::vector<int> callbackkinds_one(int n, int at)
{
  std::vector<int> v(n, 0);
  v[at] = 1;
  return v;
}

#define EPS(cls)                                                               \
  const int cls##_eps[3] = {callbackkinds_ep<CkIndex_##cls>(CBK_MSG),          \
                            callbackkinds_ep<CkIndex_##cls>(CBK_NOMSG),        \
                            callbackkinds_ep<CkIndex_##cls>(CBK_DATA)}

void callbackkinds_driver::entryCases(void)
{
  const int P = CkNumPes(), N = CkNumNodes();
  EPS(callbackkinds_chare);
  EPS(callbackkinds_group);
  EPS(callbackkinds_nodegroup);
  EPS(callbackkinds_array);
#if CMK_CHARE_USE_PTR
  const bool charePtr = true;
#else
  const bool charePtr = false;
#endif

  // Chare.
  std::vector<int> atChare = callbackkinds_one(P, charePe);
  threeModes("chare", CBK_CHARE, 1, -1, atChare,
             [](callbackkinds_driver* d, int ep) {
               return CkCallback(ep, d->chareProxy().ckGetChareID());
             },
             callbackkinds_chare_eps);
  threeModes("chare inline with refnum", CBK_CHARE, 2, 21, atChare,
             [](callbackkinds_driver* d, int ep) {
               return CkCallback(ep, d->chareProxy().ckGetChareID(), true);
             },
             callbackkinds_chare_eps);
  {
    CkCallback c(callbackkinds_chare_eps[0], chare.ckGetChareID());
    what = "chare accessors";
    checkAccessors(c, CkCallback::sendChare, charePtr, callbackkinds_chare_eps[0]);
    CkCallback ci(callbackkinds_chare_eps[0], chare.ckGetChareID(), true);
    checkAccessors(ci, CkCallback::isendChare, charePtr, callbackkinds_chare_eps[0]);
    if (c == ci) CkAbort("callbackkinds: sendChare equals isendChare\n");
  }
  for (int mode = CBK_MSG; mode <= CBK_DATA; mode++)
  {
    what = "chare self (Chare*)";
    expect(CBK_CHARE, mode == CBK_NOMSG ? -1 : 300 + mode, -1, atChare);
    chare.bounce(reflector, nextPe(), mode, 300 + mode);
    waitFor(1);
  }

  // Group element, on each PE in turn.
  for (int pe = 0; pe < P; pe++)
  {
    targetPe = pe;
    std::vector<int> at = callbackkinds_one(P, pe);
    threeModes("group element (pe, id)", CBK_GROUP, 4, -1, at,
               [](callbackkinds_driver* d, int ep) {
                 return CkCallback(ep, d->targetPe, d->groupProxy().ckGetGroupID());
               },
               callbackkinds_group_eps);
    threeModes("group element (proxy element) with refnum", CBK_GROUP, 5, 51, at,
               [](callbackkinds_driver* d, int ep) {
                 return CkCallback(ep, d->groupProxy()[d->targetPe]);
               },
               callbackkinds_group_eps);
    threeModes("group element inline", CBK_GROUP, 6, -1, at,
               [](callbackkinds_driver* d, int ep) {
                 return d->targetPe % 2 ? CkCallback(ep, d->groupProxy()[d->targetPe], true)
                                        : CkCallback(ep, d->targetPe,
                                                     d->groupProxy().ckGetGroupID(), true);
               },
               callbackkinds_group_eps);
    for (int mode = CBK_MSG; mode <= CBK_DATA; mode++)
    {
      what = "group self (Group*)";
      expect(CBK_GROUP, mode == CBK_NOMSG ? -1 : 700 + mode, -1, at);
      group[pe].bounce(reflector, nextPe(), mode, 700 + mode);
      waitFor(1);
    }
  }
  std::vector<int> allPes(P, 1);
  threeModes("group broadcast with refnum", CBK_GROUP, 8, 81, allPes,
             [](callbackkinds_driver* d, int ep) {
               return CkCallback(ep, d->groupProxy().ckGetGroupID());
             },
             callbackkinds_group_eps);
  {
    what = "group accessors";
    int ep = callbackkinds_group_eps[0];
    CkCallback b(ep, group.ckGetGroupID());
    checkAccessors(b, CkCallback::bcastGroup, false, ep);
    CkCallback e(ep, P - 1, group.ckGetGroupID());
    checkAccessors(e, CkCallback::sendGroup, false, ep);
    CkCallback ei(ep, group[P - 1], true);
    checkAccessors(ei, CkCallback::isendGroup, false, ep);
    if (b == e) CkAbort("callbackkinds: bcastGroup equals sendGroup\n");
    b.transformBcastToLocalElem(P - 1);
    if (!(b == e)) CkAbort("callbackkinds: transformBcastToLocalElem(pe) differs from sendGroup\n");
    CkCallback here(ep, group.ckGetGroupID());
    here.transformBcastToLocalElem();
    CkCallback e0(ep, CkMyPe(), group.ckGetGroupID());
    if (!(here == e0)) CkAbort("callbackkinds: transformBcastToLocalElem() is not this PE\n");
    CkCallback other(ep, 0, group.ckGetGroupID());
    if (P > 1 && other == e) CkAbort("callbackkinds: group elements on different PEs equal\n");
  }

  // Nodegroup element, on each node in turn.
  for (int node = 0; node < N; node++)
  {
    targetNode = node;
    std::vector<int> at = callbackkinds_one(N, node);
    threeModes("nodegroup element (node, proxy)", CBK_NODEGROUP, 10, -1, at,
               [](callbackkinds_driver* d, int ep) {
                 return CkCallback(ep, d->targetNode, d->nodegroupProxy());
               },
               callbackkinds_nodegroup_eps);
    threeModes("nodegroup element (proxy element) with refnum", CBK_NODEGROUP, 11, 111, at,
               [](callbackkinds_driver* d, int ep) {
                 return CkCallback(ep, d->nodegroupProxy()[d->targetNode]);
               },
               callbackkinds_nodegroup_eps);
    threeModes("nodegroup element inline", CBK_NODEGROUP, 12, -1, at,
               [](callbackkinds_driver* d, int ep) {
                 return d->targetNode % 2
                            ? CkCallback(ep, d->targetNode, d->nodegroupProxy(), true)
                            : CkCallback(ep, d->targetNode, d->nodegroupProxy().ckGetGroupID(),
                                         true, true);
               },
               callbackkinds_nodegroup_eps);
    for (int mode = CBK_MSG; mode <= CBK_DATA; mode++)
    {
      what = "nodegroup self (NodeGroup*)";
      expect(CBK_NODEGROUP, mode == CBK_NOMSG ? -1 : 1300 + mode, -1, at);
      nodegroup[node].bounce(reflector, nextPe(), mode, 1300 + mode);
      waitFor(1);
    }
  }
  std::vector<int> allNodes(N, 1);
  threeModes("nodegroup broadcast (proxy)", CBK_NODEGROUP, 14, -1, allNodes,
             [](callbackkinds_driver* d, int ep) { return CkCallback(ep, d->nodegroupProxy()); },
             callbackkinds_nodegroup_eps);
  threeModes("nodegroup broadcast (id) with refnum", CBK_NODEGROUP, 15, 151, allNodes,
             [](callbackkinds_driver* d, int ep) {
               return CkCallback(ep, d->nodegroupProxy().ckGetGroupID(), true);
             },
             callbackkinds_nodegroup_eps);
  {
    what = "nodegroup accessors";
    int ep = callbackkinds_nodegroup_eps[0];
    CkCallback b(ep, nodegroup);
    checkAccessors(b, CkCallback::bcastNodeGroup, false, ep);
    CkCallback b2(ep, nodegroup.ckGetGroupID(), true);
    if (!(b == b2)) CkAbort("callbackkinds: the two nodegroup broadcast constructors differ\n");
    CkCallback e(ep, CkMyNode(), nodegroup);
    checkAccessors(e, CkCallback::sendNodeGroup, false, ep);
    b.transformBcastToLocalElem();
    if (!(b == e)) CkAbort("callbackkinds: nodegroup transformBcastToLocalElem() differs\n");
    b2.transformBcastToLocalElem(N - 1);
    CkCallback last(ep, nodegroup[N - 1]);
    if (!(b2 == last)) CkAbort("callbackkinds: nodegroup transformBcastToLocalElem(n) differs\n");
    CkCallback ei(ep, N - 1, nodegroup, true);
    checkAccessors(ei, CkCallback::isendNodeGroup, false, ep);
  }

  // Array element: first, last and a middle element.
  const int idxs[3] = {0, nArr / 2, nArr - 1};
  for (int k = 0; k < 3; k++)
  {
    targetIdx = idxs[k];
    std::vector<int> at = callbackkinds_one(nArr, idxs[k]);
    threeModes("array element (index, id)", CBK_ARRAY, 16, -1, at,
               [](callbackkinds_driver* d, int ep) {
                 return CkCallback(ep, CkArrayIndex1D(d->targetIdx),
                                   d->arrayProxy().ckGetArrayID());
               },
               callbackkinds_array_eps);
    threeModes("array element (proxy element) with refnum", CBK_ARRAY, 17, 171, at,
               [](callbackkinds_driver* d, int ep) {
                 return CkCallback(ep, d->arrayProxy()[d->targetIdx]);
               },
               callbackkinds_array_eps);
    threeModes("array element inline", CBK_ARRAY, 18, -1, at,
               [](callbackkinds_driver* d, int ep) {
                 return d->targetIdx % 2
                            ? CkCallback(ep, d->arrayProxy()[d->targetIdx], true)
                            : CkCallback(ep, CkArrayIndex1D(d->targetIdx),
                                         d->arrayProxy().ckGetArrayID(), true);
               },
               callbackkinds_array_eps);
    for (int mode = CBK_MSG; mode <= CBK_DATA; mode++)
    {
      what = "array self (ArrayElement*)";
      expect(CBK_ARRAY, mode == CBK_NOMSG ? -1 : 1900 + mode, -1, at);
      array[idxs[k]].bounce(reflector, nextPe(), mode, 1900 + mode);
      waitFor(1);
    }
  }
  std::vector<int> allElts(nArr, 1);
  threeModes("array broadcast with refnum", CBK_ARRAY, 20, 201, allElts,
             [](callbackkinds_driver* d, int ep) {
               return CkCallback(ep, d->arrayProxy().ckGetArrayID());
             },
             callbackkinds_array_eps);
  {
    what = "array accessors";
    int ep = callbackkinds_array_eps[0];
    CkCallback b(ep, array.ckGetArrayID());
    checkAccessors(b, CkCallback::bcastArray, false, ep);
    CkCallback e(ep, array[1]);
    checkAccessors(e, CkCallback::sendArray, false, ep);
    CkCallback e2(ep, CkArrayIndex1D(1), array.ckGetArrayID());
    if (!(e == e2)) CkAbort("callbackkinds: the two array element constructors differ\n");
    CkCallback e3(ep, CkArrayIndex1D(0), array.ckGetArrayID());
    if (e == e3) CkAbort("callbackkinds: callbacks to different elements equal\n");
    CkCallback ei(ep, array[1], true);
    checkAccessors(ei, CkCallback::isendArray, false, ep);
    if (b == e) CkAbort("callbackkinds: bcastArray equals sendArray\n");
  }
}

void callbackkinds_driver::sectionCases(void)
{
  EPS(callbackkinds_array);
  // Even-numbered elements.
  CProxySection_callbackkinds_array sec =
      CProxySection_callbackkinds_array::ckNew(array.ckGetArrayID(), 0, nArr - 1, 2);
  std::vector<int> evens(nArr, 0);
  for (int i = 0; i < nArr; i += 2) evens[i] = 1;
  {
    // Built from the proxy, checked, never sent (see CALLBACKKINDS_SECTION_PROXY).
    CkCallback p(callbackkinds_array_eps[0], sec);
    what = "section (proxy) accessors";
    checkAccessors(p, CkCallback::bcastSection, true, -1);
  }
  for (int form = CALLBACKKINDS_SECTION_PROXY ? 0 : 1; form < 2; form++)
    for (int mode = CBK_MSG; mode <= CBK_DATA; mode++)
    {
      int ep = callbackkinds_array_eps[mode];
      CkCallback cb = form == 0 ? CkCallback(ep, sec) : CkCallback(ep, sec.ckGetSectionID(0));
      what = form == 0 ? "section (proxy)" : "section (section id)";
      checkAccessors(cb, CkCallback::bcastSection, true, -1);
      if (!cb.requiresMsgConstruction()) CkAbort("callbackkinds: %s needs no message\n", what);
      int val = 2200 + 10 * form + mode;
      cb.setRefnum(221);
#if CALLBACKKINDS_SECTION_EQ
      CkCallback same = cb;
      if (!(same == cb)) CkAbort("callbackkinds: %s copy differs\n", what);
#endif
      expect(CBK_ARRAY, mode == CBK_NOMSG ? -1 : val, 221, evens);
#if CALLBACKKINDS_PUP_SECTION
      reflector[nextPe()].reflect(cb, mode, val);
      waitFor(1);
#else
      if (mode == CBK_MSG) cb.send(new callbackkinds_msg(val));
      else if (mode == CBK_NOMSG) cb.send();
      else cb.send(sizeof(int), &val);
      waitFor(0);
#endif
    }
}

void callbackkinds_driver::functionCases(void)
{
  const int P = CkNumPes();
  // callCFn: runs on the PE that built it (0), forwarded there by
  // ckcallback_group when sent elsewhere.
  for (int mode = CBK_MSG; mode <= CBK_DATA; mode++)
  {
    CkCallback cb(callbackkinds_cfn, this);
    what = "callCFn";
    checkAccessors(cb, CkCallback::callCFn, true, -1);
    echoCheck(cb, nextPe());
    for (int pe = 0; pe < P; pe++)
      viaReflector("callCFn", cb, CBK_CFN, mode, 2300 + mode, -1, callbackkinds_one(P, 0), pe);
  }
  {
    CkCallback a(callbackkinds_cfn, this), b(callbackkinds_cfn, (void*)&nArr);
    if (a == b) CkAbort("callbackkinds: callCFn with different parameters equal\n");
  }
  // call1Fn: runs on the PE that sends it; only PEs of this process.
  int first = CkNodeFirst(CkMyNode()), size = CkNodeSize(CkMyNode());
  for (int mode = CBK_MSG; mode <= CBK_DATA; mode++)
    for (int r = 0; r < size; r++)
    {
      CkCallback cb(callbackkinds_c1fn);
      what = "call1Fn";
      checkAccessors(cb, CkCallback::call1Fn, true, -1);
      if (r == 0) echoCheck(cb, first + (mode % size));
      viaReflector("call1Fn", cb, CBK_C1FN, mode, 2400 + mode, -1,
                   callbackkinds_one(P, first + r), first + r);
    }
}

static void callbackkinds_checkThread(void* m, int mode, int val, const char* what)
{
  int got, refnum;
  callbackkinds_decode(m, got, refnum);
  int want = mode == CBK_NOMSG ? -1 : val;
  if (got != want)
    CkAbort("callbackkinds: %s mode %d: value %d, expected %d\n", what, mode, got, want);
  if (m) CkFreeMsg(m);
}

void callbackkinds_driver::threadCases(void)
{
  std::vector<int> none(1, 0);
  for (int mode = CBK_MSG; mode <= CBK_DATA; mode++)
  {
    int val = 2500 + mode;
    // Plain resumeThread: sent from a reflector, then thread_delay suspends.
    {
      CkCallback cb(CkCallback::resumeThread);
      what = "resumeThread";
      checkAccessors(cb, CkCallback::resumeThread, true, -1);
      echoCheck(cb, nextPe());
      expect(0, 0, -1, none);
      reflector[nextPe()].reflect(cb, mode, val);
      callbackkinds_checkThread(cb.thread_delay(), mode, val, what);
      waitFor(1);
    }
    // Sent before thread_delay: the value is already there.
    {
      CkCallback cb(CkCallback::resumeThread);
      what = "resumeThread sent locally";
      if (mode == CBK_MSG) cb.send(new callbackkinds_msg(val));
      else if (mode == CBK_NOMSG) cb.send();
      else cb.send(sizeof(int), &val);
      callbackkinds_checkThread(cb.thread_delay(), mode, val, what);
    }
    // CkCallbackResumeThread with a result pointer: its destructor waits.
    {
      void* result = (void*)-1;
      what = "CkCallbackResumeThread(result)";
      expect(0, 0, -1, none);
      reflector[nextPe()].reflect(CkCallbackResumeThread(result), mode, val);
      callbackkinds_checkThread(result, mode, val, what);
      waitFor(1);
    }
    // CkCallbackResumeThread without one: the result is freed.
    {
      what = "CkCallbackResumeThread()";
      expect(0, 0, -1, none);
      reflector[nextPe()].reflect(CkCallbackResumeThread(), mode, val);
      waitFor(1);
    }
  }
}

void callbackkinds_driver::futureCases(void)
{
  std::vector<int> none(1, 0);
  for (int mode = CBK_MSG; mode <= CBK_DATA; mode++)
  {
#if !CALLBACKKINDS_FUTURE_NOMSG
    if (mode == CBK_NOMSG) continue;
#endif
    int val = 2600 + mode;
    CkFuture f = CkCreateFuture();
    CkCallback cb(f);
    what = "sendFuture";
#if CALLBACKKINDS_FUTURE_CONTAINSPTR
    checkAccessors(cb, CkCallback::sendFuture, false, -1);
#else
    if (cb.type != CkCallback::sendFuture || cb.epIndex() != -1 || !cb.requiresMsgConstruction())
      CkAbort("callbackkinds: sendFuture accessors wrong\n");
#endif
    echoCheck(cb, nextPe());
    CkCallback other(CkCreateFuture());
    if (cb == other) CkAbort("callbackkinds: callbacks to different futures equal\n");
    expect(0, 0, -1, none);
    reflector[nextPe()].reflect(cb, mode, val);
    callbackkinds_checkThread(CkWaitFuture(f), mode, val, what);
    waitFor(1);
    CkReleaseFuture(f);
    CkReleaseFutureID(other.d.future.fut.id);
  }
}

void callbackkinds_driver::otherCases(void)
{
  std::vector<int> none(1, 0);
  // ignore: delivers nothing, in every mode.
  for (int mode = CBK_MSG; mode <= CBK_DATA; mode++)
  {
    CkCallback cb(CkCallback::ignore);
    what = "ignore";
    checkAccessors(cb, CkCallback::ignore, false, -1);
    if (cb.requiresMsgConstruction()) CkAbort("callbackkinds: ignore wants a message\n");
    echoCheck(cb, nextPe());
    viaReflector("ignore", cb, 0, mode, 2700 + mode, -1, none, nextPe());
  }
  // Built and checked, never sent.
  {
    CkCallback inv;
    what = "invalid";
    if (!inv.isInvalid() || inv.requiresMsgConstruction())
      CkAbort("callbackkinds: default callback is not invalid\n");
    checkAccessors(inv, CkCallback::invalid, false, -1);
    echoCheck(inv, nextPe());
    CkCallback ex(CkCallback::ckExit);
    what = "ckExit";
    if (ex.isInvalid() || ex.requiresMsgConstruction())
      CkAbort("callbackkinds: ckExit callback accessors wrong\n");
    checkAccessors(ex, CkCallback::ckExit, false, -1);
    echoCheck(ex, nextPe());
    if (ex == inv) CkAbort("callbackkinds: ckExit equals invalid\n");
    CcsDelayedReply reply;
    memset(&reply, 0, sizeof(reply));
    CkCallback ccs(reply);
    what = "replyCCS";
    checkAccessors(ccs, CkCallback::replyCCS, true, -1);
    echoCheck(ccs, nextPe());
  }
}

void callbackkinds_driver::run(void)
{
  entryCases();
  sectionCases();
  functionCases();
  threadCases();
  futureCases();
  otherCases();
  megatest_finish();
}

/******************** megatest hooks ********************/

void callbackkinds_init(void) { CProxy_callbackkinds_driver::ckNew(0); }

void callbackkinds_moduleinit(void) {}

MEGATEST_REGISTER_TEST(callbackkinds, "kale", 0)
#include "callbackkinds.def.h"
