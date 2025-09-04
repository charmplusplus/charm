/**
 \defgroup CkLdb  Charm++ Load Balancing Framework 
*/
/*@{*/

#ifndef BASELB_H
#define BASELB_H

#include "LBManager.h"

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
  const char *lbname;
  LBManager *lbmgr;
  int  startLbFnHdl;
private:
  void initLB(const CkLBOptions &);
public:
  struct ProcStats {		// per processor data
    int n_objs;			// number of objects on the processor
    double pe_speed;		// processor frequency
#if defined(TEMP_LDB)
	float pe_temp;
#endif

    /// total time (total_walltime) = idletime + overhead (bg_walltime)
    ///                             + object load (obj_walltime)
    /// walltime and cputime may be different on shared compute nodes
    /// it is advisable to use walltime in most cases
    double total_walltime;
    /// time for which the processor is sitting idle
    double idletime;
    /// bg_walltime called background load (overhead in ckgraph.h) is a
    /// derived quantity: total_walltime - idletime - object load (obj_walltime)
    double bg_walltime;
#if CMK_LB_CPUTIMER
    double total_cputime;
    double bg_cputime;
#endif
    // double utilization;
    int pe;			// processor id
    bool available;
    ProcStats(): n_objs(0), pe_speed(1), total_walltime(0.0), idletime(0.0),
#if CMK_LB_CPUTIMER
		 total_cputime(0.0), bg_cputime(0.0),
#endif
	   	 bg_walltime(0.0), pe(-1), available(true) {}
    inline void clearBgLoad() {
      idletime = bg_walltime = 
#if CMK_LB_CPUTIMER
      bg_cputime = 
#endif
      0.0;
    }
    inline void pup(PUP::er &p) {
      p|total_walltime;
      p|idletime;
      p|bg_walltime;
#if CMK_LB_CPUTIMER
      p|total_cputime;
      p|bg_cputime;
#endif
      p|pe_speed;
      if (_lb_args.lbversion() < 1 && p.isUnpacking()) {
         double dummy;  p|dummy;    // for old format with utilization
      }
      p|available; p|n_objs;
      if (_lb_args.lbversion()>=2) p|pe; 
    }
  };

  /** Passed to the virtual functions Strategy(...) and work(...) */
  struct LDStats {
    std::vector<ProcStats> procs;		// processor statistics

    int n_migrateobjs;		// total number of migratable objects
    std::vector<LDObjData> objData;	// LDObjData and LDCommData defined in lbdb.h
    std::vector<int> from_proc;	// current pe an object is on
    std::vector<int> to_proc;		// new pe you want the object to be on

    std::vector<LDCommData> commData;	// communication data - edge list representation
				// of the communication between objects

    std::vector<int> objHash;		// this a map from the hash for the 4 integer
				// LDObjId to the index in the vector "objData"
    int  hashSize;

    int complete_flag;		// if this ocg is complete, eg in HybridLB,
    // this LDStats may not be complete

    int is_prev_lb_refine;
    double after_lb_max;
    double after_lb_avg;

    size_t nprocs() const { return procs.size(); }
    void setNprocs(size_t count) { procs.resize(count); }
    LDStats(int npes = CkNumPes(), int complete_flag = 1);

    void assign(int oid, int pe) { CmiAssert(procs[pe].available); to_proc[oid] = pe; }
    /// build hash table
    void makeCommHash();
    void deleteCommHash();
    /// given an LDObjKey, returns the index in the objData vector
    /// this index changes every time one does load balancing even within a run
    int getHash(const LDObjKey &);
    int getHash(const CmiUInt8 &oid, const LDOMid &mid);
    int getSendHash(LDCommData &cData);
    int getRecvHash(LDCommData &cData);
    void clearCommHash();
    void clear() {
      n_migrateobjs = 0;
      objData.clear();
      commData.clear();
      from_proc.clear();
      to_proc.clear();
      deleteCommHash();
    }
    void clearBgLoad() {
      for (auto& proc : procs) proc.clearBgLoad();
    }
    void computeNonlocalComm(int &nmsgs, int &nbytes);
    double computeAverageLoad();
    void normalize_speed();
    void print();
    // edit functions
    void removeObject(int obj);
    void pup(PUP::er &p);
    int useMem();
  };

  BaseLB(const CkLBOptions &opt)  { initLB(opt); }
  BaseLB(CkMigrateMessage *m):CBase_BaseLB(m) {
    lbmgr = CProxy_LBManager(_lbmgr).ckLocalBranch();
  }
  virtual ~BaseLB() {}

  inline const char *lbName() { return lbname; }
  inline int step() { return lbmgr->step(); }
  virtual void turnOff();
  virtual void turnOn();
  virtual int  useMem()  { return 0; }
  virtual void pup(PUP::er &p);
  virtual void flushStates();
//  virtual void AtSync(void) {  CmiAbort("AtSync not implemented"); } // Everything is at the PE barrier
  virtual void InvokeLB(void) { CmiAbort("InvokeLB not implemented"); }
  virtual void Migrated(int waitBarrier=1) { CmiAbort("Migrated not implemented"); }

  CkGroupID getGroupID() {return thisgroup;}
};

/// migration decision for an obj.
struct MigrateInfo {  
    int index;   // object index in objData array
    LDObjHandle obj;
    int from_pe;
    int to_pe;
    int async_arrival;	    // if an object is available for immediate migrate
    MigrateInfo():  async_arrival(0) {
    }

    void pup(PUP::er &p) {
      p | index;
      p | obj;
      p | from_pe;
      p | to_pe;
      p | async_arrival;
    }
};

struct MigrateDecision {
  LDObjIndex dbIndex;
  int fromPe;
  int toPe;

  MigrateDecision &operator=(const MigrateInfo &mInfo) {
    dbIndex = mInfo.obj.handle;
    fromPe = mInfo.from_pe;
    toPe = mInfo.to_pe;

    return *this;
  }

};

class LBScatterMsg : public CMessage_LBScatterMsg {
public:
  int numMigrates;
  int firstPeInSpan;
  int lastPeInSpan;
  int *numMigratesPerPe;
  MigrateDecision *moves;

  LBScatterMsg(int firstPe, int lastPe) {
    numMigrates = 0;
    firstPeInSpan = firstPe;
    lastPeInSpan = lastPe;
  }
};



/**
  message contains the migration decision from LB strategies.
*/
class LBMigrateMsg : public CMessage_LBMigrateMsg {
public:
  int level;			// which level in hierarchy, used in hybridLB

  int n_moves;			// number of moves
  MigrateInfo* moves;

  char * avail_vector;		// processor bit vector
  int next_lb;			// next load balancer

  double * expectedLoad;	// expected load for future

public:
  LBMigrateMsg(): level(0), n_moves(0), next_lb(0) {}
  void pup(PUP::er &p) {
    int i;
    p | level;
    p | n_moves;
    // Warning: it relies on the fact that the message has been allocated already
    // with the correct number of moves!
    p | next_lb;
    int numPes = CkNumPes();
    p | numPes;
    CkAssert(numPes == CkNumPes());
    for (i=0; i<n_moves; ++i) p | moves[i];
    p(avail_vector, numPes);
    for (i=0; i<numPes; ++i) p | expectedLoad[i];
  }
};

struct VectorMigrateInfo {  
    int from_pe;
    int to_pe;
    double load;
    int async_arrival;	    // if an object is available for immediate migrate
    VectorMigrateInfo():  async_arrival(0) {}
};

class LBVectorMigrateMsg : public CMessage_LBVectorMigrateMsg {
public:
  int level;			// which level in hierarchy, used in hybridLB

  int n_moves;			// number of moves
  VectorMigrateInfo* moves;

public:
  LBVectorMigrateMsg(): level(0), n_moves(0) {}
};

// Deprecated, left around to support external custom load balancers
// Internal load balancers should define lbinit directly and use LBRegisterBalancer<T>
//
// For a FooLB, the following macro defines these functions for each LB:
// CreateFooLB():        create BOC and register with LBManager with a
//                       sequence ticket,
// AllocateFooLB():      allocate the class instead of a BOC
// static void lbinit(): an init call for charm module registration
#if CMK_LBDB_ON

#  define CreateLBFunc_Def(x, str)                                                       \
    CMK_DEPRECATED_MSG("Use LBRegisterBalancer() instead of CreateLBFunc_Def()")         \
    void Create##x(const CkLBOptions& opts) { CProxy_##x::ckNew(opts); }                 \
    CMK_DEPRECATED_MSG("Use LBRegisterBalancer() instead of CreateLBFunc_Def()")         \
    BaseLB* Allocate##x(void) { return new x(static_cast<CkMigrateMessage*>(nullptr)); } \
    CMK_DEPRECATED_MSG("Use LBRegisterBalancer() instead of CreateLBFunc_Def()")         \
    static void lbinit(void) { LBRegisterBalancer(#x, Create##x, Allocate##x, str); }

#else /* CMK_LBDB_ON */

#  define CreateLBFunc_Def(x, str)                \
    void Create##x(void) {}                       \
    BaseLB* Allocate##x(void) { return nullptr; } \
    static void lbinit(void) {}

#endif /* CMK_LBDB_ON */

#endif

/*@}*/
