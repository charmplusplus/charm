/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/**
 \defgroup CkLdb  Charm++ Load Balancing Framework 
*/
/*@{*/

#ifndef BASELB_H
#define BASELB_H

#include "LBDatabase.h"

/// Base class for all LB strategies.
/**
  BaseLB is the base class for all LB strategy class.
  it does some tracking about how many lb strategies are created.
  it also defines some common functions.
*/
class BaseLB: public IrrGroup
{
protected:
  int  seqno;
  char *lbname;
  LBDatabase *theLbdb;
  LDBarrierReceiver receiver;
  int  notifier;
  int  startLbFnHdl;
private:
  void initLB(const CkLBOptions &);
public:
  BaseLB(const CkLBOptions &opt)  { initLB(opt); }
  BaseLB(CkMigrateMessage *m):IrrGroup(m) {}
  virtual ~BaseLB();

  void unregister(); 
  inline char *lbName() { return lbname; }
  virtual void turnOff() { CmiAbort("turnOff not implemented"); }
  virtual void turnOn()  { CmiAbort("turnOn not implemented"); }
  void pup(PUP::er &p);
};

/// migration decision for an obj.
struct MigrateInfo {  
    int index;   // object index in objData array
    LDObjHandle obj;
    int from_pe;
    int to_pe;
};

/**
  message contains the migration decision from LB strategies.
*/
class LBMigrateMsg : public CMessage_LBMigrateMsg {
public:
  int n_moves;
  MigrateInfo* moves;

  char * avail_vector;
  int next_lb;

  double * expectedLoad;
};

#if CMK_LBDB_ON
#define CreateLBFunc_Def(x)		\
void Create##x(void) { 	\
  int seqno = LBDatabaseObj()->getLoadbalancerTicket();	\
  CProxy_##x::ckNew(CkLBOptions(seqno)); 	\
}	\
BaseLB *Allocate##x(void) { \
  return new x((CkMigrateMessage*)NULL);	\
}
#else		/* CMK_LBDB_ON */
#define CreateLBFunc_Def(x)	\
void Create##x(void) {} 	\
BaseLB *Allocate##x(void) { return NULL; }
#endif

#endif

/*@}*/
