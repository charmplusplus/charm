#ifndef CKMIGRATABLE_H
#define CKMIGRATABLE_H

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
  // Sequence number of the last load balancing step this chare joined. It rises
  // to CkpvAccess(_lbStepRequested) in AtSyncStart; comparing the two is the
  // whole cost of an AtSyncStart that has no step to start.
  int lbStepSeen;
  // +LBAsync state. lbStepPending is set when AtSyncStart joins a step and
  // cleared when that step resumes clients here; waitParked records that
  // AtSyncWait found the step still running and stopped the element. lbWaitEpoch
  // is the LBManager resume epoch the element is waiting past, which lets an
  // element migrated by the very step it waits on notice, on arrival, that its
  // destination has already resumed. All three are pupped across migration.
  bool lbStepPending;
  bool waitParked;
  int lbWaitEpoch;
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

  // MetaBalancer's AtSync split in two, so that collecting load and starting a
  // step can run at different rates. Call AtSyncSample() every few iterations
  // (it flushes GPU counters and feeds a background reduction) and
  // AtSyncStart() every iteration (it costs one integer compare unless a step
  // has actually been requested, so a step begins one iteration after the
  // imbalance that caused it was seen).
  void AtSyncSample();
  void AtSyncStart(int waitForMigration=1);

  // The other half of the split. Under +LBAsync this blocks the element until
  // every migration the in-flight step planned, on every PE, has completed, and
  // then calls ResumeFromSync() exactly once; with no step in flight it calls
  // ResumeFromSync() inline. Without +LBAsync it does nothing, because the
  // resume already arrives from the AtSync barrier -- so an application written
  // to the async pattern runs correctly either way.
  void AtSyncWait();

  // True when the next AtSyncStart() will actually start a step, so an
  // application can do its pre-migration work -- draining device streams, say
  // -- only on the iterations where that work is needed.
  bool AtSyncPending() const;

  // Release an element that parked in AtSyncWait() and was then migrated by the
  // step it was waiting on, landing after its destination had already resumed.
  // Called from the migration arrival paths in CkLocMgr.
  void lbCheckWaitRelease();

  int MigrateToPe()  { return myRec->MigrateToPe(); }

private:
  void ResumeFromSyncHelper();
  void recordLBSizes(bool forStep);
  void sampleMetaLBLoad();
public:

  void ReadyMigrate(bool ready);
  void ckFinishConstruction(int epoch = -1);
  void setMigratable(int migratable);
  void setPupSize(size_t obj_pup_size);
  void setGPUPupSize(size_t obj_gpu_pup_size);
#else
  void AtSync(int waitForMigration=1) { ResumeFromSync();}
  void AtSyncSample() { }
  void AtSyncStart(int waitForMigration=1) { ResumeFromSync();}
  void AtSyncWait() { }
  bool AtSyncPending() const { return false; }
  void lbCheckWaitRelease() { }
  void setMigratable(int migratable)  { }
  void setPupSize(size_t obj_pup_size) { }
public:
  void ckFinishConstruction(int epoch) { }
#endif

public:
  // Intra-process (same CmiNode / SMP mode) migration fast path.
  // Called on the source PE just before ownership is transferred: returns
  // the current AtSync barrier epoch and detaches from the source PE's
  // sync barrier. Returns -1 if no epoch tracking is needed.
#if CMK_LBDB_ON
  /// Detach a chare from the AtSync barrier while it is held alive on the
  /// source PE waiting for a migration ack. Without this, the held chare would
  /// keep iterating in parallel with the migrated copy on the destination.
  /// ~CkMigratable sees barrierRegistered already false and skips the
  /// duplicate removeClient call.
  void ckSuspendBarrierForDeferredDestroy();
#endif

  virtual int ckPrepareIntraProcessMigrate();
  // Called on the destination PE after ownership is transferred: rebinds
  // the chare to its new CkLocRec and re-registers with the destination
  // PE's sync barrier at the given epoch.
  virtual void ckFinalizeIntraProcessMigrate(CkLocRec* newRec, int epoch);

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
