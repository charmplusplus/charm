#ifndef _CALLBACKKINDS_H
#define _CALLBACKKINDS_H

#include <vector>
#include "megatest.h"

#include "callbackkinds.decl.h"

// How the reflector sends the callback it receives.
enum { CBK_MSG = 0, CBK_NOMSG = 1, CBK_DATA = 2, CBK_ECHO = 3 };

class callbackkinds_msg : public CMessage_callbackkinds_msg
{
 public:
  int val;
  callbackkinds_msg(int v) : val(v) {}
};

class callbackkinds_driver : public CBase_callbackkinds_driver
{
 public:
  callbackkinds_driver(void);
  callbackkinds_driver(CkMigrateMessage* m) {}
  void run(void);
  void got(int kind, int val, int where, int refnum);
  void reflected(void) { nReflected++; }
  void echo(CkCallback cb);

 private:
  CProxy_callbackkinds_reflector reflector;
  CProxy_callbackkinds_chare chare;
  CProxy_callbackkinds_group group;
  CProxy_callbackkinds_nodegroup nodegroup;
  CProxy_callbackkinds_array array;
  int nArr, charePe, step;
  const char* what;

  // Expectations for the current case.
  int expKind, expVal, expRefnum, expTotal, nGot, nReflected, nEchoed;
  std::vector<int> expWhere, whereCount;
  CkCallback echoed;

  int nextPe(void) { return (step++) % CkNumPes(); }
  void expect(int kind, int val, int refnum, const std::vector<int>& where);
  void waitFor(int nReflect);
  void echoCheck(CkCallback& cb, int pe);
  void viaReflector(const char* name, CkCallback cb, int kind, int mode, int val, int refnum,
                    const std::vector<int>& where, int pe);
  void threeModes(const char* name, int kind, int caseNo, int refnum,
                  const std::vector<int>& where, CkCallback (*make)(callbackkinds_driver*, int ep),
                  const int* eps);
  void checkAccessors(const CkCallback& cb, CkCallback::callbackType type, bool pointer,
                      int ep);

  void entryCases(void);
  void sectionCases(void);
  void functionCases(void);
  void threadCases(void);
  void futureCases(void);
  void otherCases(void);

 public:
  // Used by the callback builders in callbackkinds.C.
  int targetPe, targetNode, targetIdx;
  const CProxy_callbackkinds_chare& chareProxy(void) const { return chare; }
  const CProxy_callbackkinds_group& groupProxy(void) const { return group; }
  const CProxy_callbackkinds_nodegroup& nodegroupProxy(void) const { return nodegroup; }
  const CProxy_callbackkinds_array& arrayProxy(void) const { return array; }
};

class callbackkinds_reflector : public CBase_callbackkinds_reflector
{
  CProxy_callbackkinds_driver driver;

 public:
  callbackkinds_reflector(CProxy_callbackkinds_driver d) : driver(d) {}
  callbackkinds_reflector(CkMigrateMessage* m) {}
  void reflect(CkCallback cb, int mode, int val);
};

// Delivery reports shared by the four target kinds.
void callbackkinds_report(CProxy_callbackkinds_driver d, int kind, int where, void* msg);

template <class Index>
int callbackkinds_ep(int mode)
{
  if (mode == CBK_MSG) return Index::accept((callbackkinds_msg*)NULL);
  if (mode == CBK_NOMSG) return Index::ping();
  return Index::data((CkDataMsg*)NULL);
}

#define CALLBACKKINDS_TARGET(cls, kind, whereExpr)                                   \
  class cls : public CBase_##cls                                                     \
  {                                                                                  \
    CProxy_callbackkinds_driver driver;                                              \
                                                                                     \
   public:                                                                           \
    cls(CProxy_callbackkinds_driver d) : driver(d) {}                                \
    cls(CkMigrateMessage* m) {}                                                      \
    void accept(callbackkinds_msg* m) { callbackkinds_report(driver, kind, whereExpr, m); } \
    void ping(void) { callbackkinds_report(driver, kind, whereExpr, NULL); }          \
    void data(CkDataMsg* m) { callbackkinds_report(driver, kind, whereExpr, m); }    \
    /* Builds the callback from inside the object (the Chare*, Group*,       */      \
    /* NodeGroup* and ArrayElement* constructors) and has it reflected.      */      \
    void bounce(CProxy_callbackkinds_reflector r, int reflectPe, int mode, int val)  \
    {                                                                                \
      bool inl = mode == CBK_NOMSG;                                                  \
      CkCallback cb(this, callbackkinds_ep<CkIndex_##cls>(mode), inl);               \
      r[reflectPe].reflect(cb, mode, val);                                           \
    }                                                                                \
  };

enum { CBK_CHARE = 1, CBK_GROUP, CBK_NODEGROUP, CBK_ARRAY, CBK_CFN, CBK_C1FN };

CALLBACKKINDS_TARGET(callbackkinds_chare, CBK_CHARE, CkMyPe())
CALLBACKKINDS_TARGET(callbackkinds_group, CBK_GROUP, CkMyPe())
CALLBACKKINDS_TARGET(callbackkinds_nodegroup, CBK_NODEGROUP, CkMyNode())
CALLBACKKINDS_TARGET(callbackkinds_array, CBK_ARRAY, thisIndex)

#endif
