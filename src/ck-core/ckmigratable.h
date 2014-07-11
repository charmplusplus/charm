#ifndef CKMIGRATABLE_H
#define CKMIGRATABLE_H

class CkMigratable : public Chare {
protected:
  CkLocRec *myRec;
private:
  int thisChareType;//My chare type
  void commonInit(void);
  bool asyncEvacuate;
  int atsync_iteration;

  enum state {
    OFF,
    ON,
    PAUSE,
    DECIDED,
    LOAD_BALANCE
  } local_state;
  double  prev_load;
  bool can_reset;

public:
  CkArrayIndex thisIndexMax;

  CkMigratable(void);
  CkMigratable(CkMigrateMessage *m);
  virtual ~CkMigratable();
  virtual void pup(PUP::er &p);
  virtual void CkAddThreadListeners(CthThread tid, void *msg);

  virtual int ckGetChareType(void) const;// {return thisChareType;}
  const CkArrayIndex &ckGetArrayIndex(void) const {return myRec->getIndex();}
  CmiUInt8 ckGetID(void) const { return myRec->getID(); }

#if CMK_LBDB_ON  //For load balancing:
  //Suspend load balancer measurements (e.g., before CthSuspend)
  inline void ckStopTiming(void) {myRec->stopTiming();}
  //Begin load balancer measurements again (e.g., after CthSuspend)
  inline void ckStartTiming(void) {myRec->startTiming();}
  inline LBDatabase *getLBDB(void) const {return myRec->getLBDB();}
  inline MetaBalancer *getMetaBalancer(void) const {return myRec->getMetaBalancer();}
#else
  inline void ckStopTiming(void) { }
  inline void ckStartTiming(void) { }
#endif

  /// for inline call
  LDObjHandle timingBeforeCall(int *objstopped);
  void timingAfterCall(LDObjHandle objHandle,int *objstopped);

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
  inline bool ckInvokeEntry(int epIdx,void *msg,bool doFree) 
	  {return myRec->invokeEntry(this,msg,epIdx,doFree);}

protected:
  /// A more verbose form of abort
  virtual void CkAbort(const char *str) const;

  bool usesAtSync;//You must set this in the constructor to use AtSync().
  bool usesAutoMeasure; //You must set this to use auto lb instrumentation.
  bool barrierRegistered;//True iff barrier handle below is set

public:
  virtual void ResumeFromSync(void);
  virtual void UserSetLBLoad(void);  /// user define this when setLBLoad is true
  void setObjTime(double cputime);
  double getObjTime();
#if CMK_LB_USER_DATA
  void *getObjUserData(int idx);
#endif

#if CMK_LBDB_ON  //For load balancing:
  void AtSync(int waitForMigration=1);
  int MigrateToPe()  { return myRec->MigrateToPe(); }

private: //Load balancer state:
  LDBarrierClient ldBarrierHandle;//Transient (not migrated)  
  LDBarrierReceiver ldBarrierRecvHandle;//Transient (not migrated)  
  static void staticResumeFromSync(void* data);
public:
  void ReadyMigrate(bool ready);
  void ckFinishConstruction(void);
  void setMigratable(int migratable);
  void setPupSize(size_t obj_pup_size);
#else
  void AtSync(int waitForMigration=1) { ResumeFromSync();}
  void setMigratable(int migratable)  { }
  void setPupSize(size_t obj_pup_size) { }
public:
  void ckFinishConstruction(void) { }
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

  // FAULT_EVAC
  void AsyncEvacuate(bool set){myRec->AsyncEvacuate(set);asyncEvacuate = set;};
public:
  bool isAsyncEvacuate(){return asyncEvacuate;};
};

#endif // CKMIGRATABLE_H
