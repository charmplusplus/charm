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

#define PER_MESSAGE_SEND_OVERHEAD_DEFAULT   3.5e-5
#define PER_BYTE_SEND_OVERHEAD_DEFAULT      8.5e-9
#define PER_MESSAGE_RECV_OVERHEAD  	    0.0
#define PER_BYTE_RECV_OVERHEAD      	    0.0

/// Base class for all LB strategies.
/**
  BaseLB is the base class for all LB strategy class.
  it does some tracking about how many lb strategies are created.
  it also defines some common functions.
*/
class BaseLB: public CBase_BaseLB
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
  struct ProcStats {  // per processor data
    double total_walltime;
    double total_cputime;
    double idletime;
    double bg_walltime;
    double bg_cputime;
    //double utilization;
    int   n_objs;
    int   pe_speed;
    CmiBool available;
    ProcStats(): total_walltime(0.0), total_cputime(0.0), idletime(0.0),
	   	 bg_walltime(0.0), bg_cputime(0.0), n_objs(0), 
		 pe_speed(1), available(CmiTrue)  {}
    inline void pup(PUP::er &p) {
      p|total_walltime;  p|total_cputime; p|idletime;
      p|bg_walltime; p|bg_cputime; p|pe_speed;
      p|available; p|n_objs;
    }
  };

  struct LDStats {  // Passed to Strategy
    ProcStats  *procs;
    int count; 
    
    int   n_objs;
    int   n_migrateobjs;
    CkVec<LDObjData> objData;
    int   n_comm;
    CkVec<LDCommData> commData;
    CkVec<int>  from_proc, to_proc;

    int *objHash; 
    int  hashSize;

    LDStats();
    void assign(int oid, int pe) { CmiAssert(procs[pe].available); to_proc[oid] = pe; }
      // build hash table
    void makeCommHash();
    void deleteCommHash();
    int getHash(const LDObjKey &);
    int getHash(const LDObjid &oid, const LDOMid &mid);
    int getSendHash(LDCommData &cData);
    int getRecvHash(LDCommData &cData);
    void clear() {
      n_objs = n_comm = 0;
      objData.free();
      commData.free();
      from_proc.free();
      to_proc.free();
      deleteCommHash();
    }
    void print();
    double computeAverageLoad();
    void pup(PUP::er &p);
    int useMem();
  };

  BaseLB(const CkLBOptions &opt)  { initLB(opt); }
  BaseLB(CkMigrateMessage *m):CBase_BaseLB(m) {}
  virtual ~BaseLB();

  void unregister(); 
  inline char *lbName() { return lbname; }
  virtual void turnOff() { CmiAbort("turnOff not implemented"); }
  virtual void turnOn()  { CmiAbort("turnOn not implemented"); }
  void pup(PUP::er &p);
  virtual void flushStates();
};

/// migration decision for an obj.
struct MigrateInfo {  
    int index;   // object index in objData array
    LDObjHandle obj;
    int from_pe;
    int to_pe;
    int async_arrival;	    // if an object is available for immediate migrate
    MigrateInfo():  async_arrival(0) {}
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

// for a FooLB, the following macro defines these functions for each LB:
// CreateFooLB(): which register with LBDatabase with sequence ticket
// , 
// AllocateFooLB(): which only locally allocate the class
// static void lbinit(): which is an init call
#if CMK_LBDB_ON
#define CreateLBFunc_Def(x, str)		\
void Create##x(void) { 	\
  int seqno = LBDatabaseObj()->getLoadbalancerTicket();	\
  CProxy_##x::ckNew(CkLBOptions(seqno)); 	\
}	\
BaseLB *Allocate##x(void) { \
  return new x((CkMigrateMessage*)NULL);	\
}	\
static void lbinit(void) {	\
  LBRegisterBalancer(#x,	\
                     Create##x,	\
                     Allocate##x,	\
                     str);	\
}
#else		/* CMK_LBDB_ON */
#define CreateLBFunc_Def(x, str)	\
void Create##x(void) {} 	\
BaseLB *Allocate##x(void) { return NULL; }	\
static void lbinit(void) {}
#endif

#endif

/*@}*/
