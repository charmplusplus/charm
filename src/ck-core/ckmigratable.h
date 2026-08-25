#ifndef CKMIGRATABLE_H
#define CKMIGRATABLE_H

#include "rescalepoint.h"

class CkMigratable : public Chare {
protected:
private:
  int thisChareType;//My chare type
  int atsync_iteration;
  double prev_load;
  enum state : uint8_t {
    OFF,
    ON,
    PAUSE,
    DECIDED,
    LOAD_BALANCE
  } local_state;
  bool can_reset;
  // Last iteration this element offered to checkRescale(); -1 before the
  // first offer. Consumed by AtSync() to tag the barrier arrival, which is
  // what lets a rescale commit rewind exactly the elements parked at a
  // tentative ceiling and no others. Transient, like the barrier handle.
  int lastRescaleCheckIter = -1;
protected:
  bool usesAtSync;//You must set this in the constructor to use AtSync().
  bool usesAutoMeasure; //You must set this to use auto lb instrumentation.
  bool barrierRegistered;//True iff barrier handle below is set

private: //Load balancer state:
  LDBarrierClient ldBarrierHandle;//Transient (not migrated)
  LDBarrierReceiver ldBarrierRecvHandle;//Transient (not migrated)
public:
  CkArrayIndex thisIndexMax;

private:
  void commonInit(void);
public:
  CkMigratable(void);
  CkMigratable(CkMigrateMessage *m);
  virtual ~CkMigratable();
  virtual void pup(PUP::er &p);
  virtual void CkAddThreadListeners(CthThread tid, void *msg);

  virtual int ckGetChareType(void) const;// {return thisChareType;}
  const CkArrayIndex &ckGetArrayIndex(void) const {return myRec->getIndex();}
  CmiUInt8 ckGetID(void) const { return myRec->getID(); }

#if CMK_LBDB_ON  //For load balancing:
  inline LBManager *getLBMgr(void) const {return myRec->getLBMgr();}
  inline MetaBalancer *getMetaBalancer(void) const {return myRec->getMetaBalancer();}
#endif

  //Initiate a migration to the given processor
  inline void ckMigrate(int toPe) {myRec->migrateMe(toPe);}
  
  /// Called by the system just before and after migration to another processor:  
  virtual void ckAboutToMigrate(void); /*default is empty*/
  virtual void ckJustMigrated(void); /*default is empty*/

  void recvLBPeriod(void *data);
  void metaLBCallLB();
  void clearMetaLBData(void);

  //used for out-of-core emulation
  virtual void ckJustRestored(void); /*default is empty*/

  /// Delete this object
  virtual void ckDestroy(void);

  /// Execute the given entry method.  Returns false if the element 
  /// deleted itself or migrated away during execution.
  // TODO: Why does this have a different signature than other invoke calls?
  inline bool ckInvokeEntry(int epIdx,void *msg,bool doFree) 
	  {return myRec->invokeEntry(this,msg,epIdx,doFree);}

protected:
  /// A more verbose form of abort
  CMK_NORETURN
#if defined __GNUC__ || defined __clang__
  __attribute__ ((format (printf, 2, 3)))
#endif
  virtual void CkAbort(const char *format, ...) const;

public:
  virtual void ResumeFromSync(void);
  virtual void UserSetLBLoad(void);  /// user define this when setLBLoad is true
  void setObjTime(double cputime);
  double getObjTime();
  void setObjGPUTime(double cputime);
  double getObjGPUTime();
#if CMK_LB_USER_DATA
  void *getObjUserData(int idx);
#endif

#if CMK_LBDB_ON  //For load balancing:
  void AtSync(int waitForMigration=1);

  /** Offer this iteration boundary as a place the job could change width.
   *
   * As checkRescale below, for a job with no globally meaningful iteration
   * counter; see ../ck-ldb/rescalepoint.h for what it gives up. */
  bool rescalePending();

  /** Should this iteration be the one the job changes width at?
   *
   *      void iterate() {
   *        if (checkRescale(iteration)) { quiesce(); AtSync(); return; }
   *        ... send, compute, advance ...
   *      }
   *
   * That is the whole application-side contract: one predicate, and AtSync()
   * where you would have called it anyway. False at every boundary but one.
   * Requires usesAtSync = true. See ../ck-ldb/rescalepoint.h for why it is
   * safe for this to answer true before the exact iteration is settled. */
  bool checkRescale(int iter);

  int MigrateToPe()  { return myRec->MigrateToPe(); }

private:
  void ResumeFromSyncHelper();
public:

  void ReadyMigrate(bool ready);
  void ckFinishConstruction(int epoch = -1);
  void setMigratable(int migratable);
  void setPupSize(size_t obj_pup_size);
  void setGPUPupSize(size_t obj_gpu_pup_size);
#else
  void AtSync(int waitForMigration=1) { ResumeFromSync();}
  bool checkRescale(int iter) { return false; }
  bool rescalePending() { return false; }
  void setMigratable(int migratable)  { }
  void setPupSize(size_t obj_pup_size) { }
public:
  void ckFinishConstruction(int epoch) { }
#endif

#if CMK_OUT_OF_CORE
private:
  friend class CkLocMgr;
  friend int CkArrayPrefetch_msg2ObjId(void *msg);
  friend void CkArrayPrefetch_writeToSwap(FILE *swapfile,void *objptr);
  friend void CkArrayPrefetch_readFromSwap(FILE *swapfile,void *objptr);
  int prefetchObjID; //From CooRegisterObject
  bool isInCore; //If true, the object is present in memory
#endif
};

#endif // CKMIGRATABLE_H
