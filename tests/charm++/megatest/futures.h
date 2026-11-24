#ifndef _FUTURES_H
#define _FUTURES_H

#include "megatest.h"
#include "ckfutures.h"
#include "futures.decl.h"

class futures_msg : public CMessage_futures_msg
{
 public:
  int val;
  int where;  // PE (node for the nodegroup) that produced the value
  futures_msg(int v, int w = -1) : val(v), where(w) {}
};

class futures_driver : public CBase_futures_driver
{
 public:
  futures_driver(void);
  futures_driver(CkMigrateMessage* m) {}
  void done(int nPes);
};

class futures_worker : public CBase_futures_worker
{
  CProxy_futures_driver driver;
  CProxy_futures_helper helper;
  CProxy_futures_nodehelper nodehelper;
  CProxy_futures_arr arr;
  int nArr;
  int me, n, q1, q2, base;

  void testIds(void);
  void testWaitAny(void);
  void testTyped(void);
  void testWaitSome(void);
  void testCollectiveValues(void);
  void testSync(void);
  void testArray(void);
  void testSema(void);

 public:
  futures_worker(CProxy_futures_driver d, CProxy_futures_helper h,
                 CProxy_futures_nodehelper nh, CProxy_futures_arr a, int na)
      : driver(d), helper(h), nodehelper(nh), arr(a), nArr(na) {}
  futures_worker(CkMigrateMessage* m) {}
  void run(void);
};

class futures_helper : public CBase_futures_helper
{
 public:
  futures_helper(void) {}
  futures_helper(CkMigrateMessage* m) {}
  void fill(CkFuture f, int val);
  void fillId(CkFutureID id, int pe, int val);
  void fillVoid(CkFuture f);
  void fillTyped(ck::future<int> f, int val);
  void fillData(CkCallback cb, int val);
  void signal(CkSemaID s, int val);
  void reduceTo(ck::future<int> f, int val);
  futures_msg* twice(futures_msg* m);
  int add(int a, int b);
};

class futures_nodehelper : public CBase_futures_nodehelper
{
 public:
  futures_nodehelper(void) {}
  futures_nodehelper(CkMigrateMessage* m) {}
  futures_msg* twice(futures_msg* m);
  int add(int a, int b);
};

class futures_chare : public CBase_futures_chare
{
 public:
  futures_chare(void) {}
  futures_chare(CkMigrateMessage* m) {}
  futures_msg* twice(futures_msg* m);
  int add(int a, int b);
};

class futures_arr : public CBase_futures_arr
{
 public:
  futures_arr(void) {}
  futures_arr(CkMigrateMessage* m) {}
  futures_msg* twice(futures_msg* m);
  futures_msg* igetTwice(futures_msg* m);
  void igetVoid(int x);
};

#endif
