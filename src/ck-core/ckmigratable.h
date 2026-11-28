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
  // +LBAsync state. lbStepPending is set when AtSyncStart joins a step and
  // cleared when that step resumes clients here; waitParked records that
  // AtSyncWait found the step still running and stopped the element. lbWaitEpoch
  // is the LBManager resume epoch the element is waiting past, which lets an
  // element migrated by the very step it waits on notice, on arrival, that its
  // destination has already resumed. All three are pupped across migration.
  bool lbStepPending;
  bool waitParked;
  int lbWaitEpoch;
  // This element's measurement window is closed: from its join until the
  // step is handed back to it, its work is not billed to it (LBObj::
  // setJoinedStep, hapiCuptiObjectJoinedStep). Pupped across migration, since
  // the destination's database starts the element with its window open.
  bool lbMeasurementClosed;
  void lbSetMeasurementClosed(bool closed);
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
  // AtSyncStart() followed by AtSyncWait(): the element stops until the step
  // is over. Unlike AtSyncStart, AtSync may be called again before the step's
  // resume -- the element joins the next step too and is resumed once per step.
  void AtSync(int waitForMigration=1);

  // The first half of AtSync. It joins the load balancing step when one is due
  // and reports what it did:
  //   Continue -- no step at this call; carry on iterating.
  //   Started  -- this call joined the step.
  //   Blocked  -- MetaBalancer has not yet settled when this element joins.
  //               It is stopped: do not iterate, do not call AtSyncStart
  //               again. It is resumed through ResumeFromSync, or joined for
  //               you, once that is decided.
  // Without +LBAsync a joined element waits for ResumeFromSync, as under
  // AtSync. With +LBAsync it returns to its caller and keeps running while the
  // strategy and the migrations proceed; AtSyncWait() is where it finds out
  // the step is over. A step must be waited out before the next AtSyncStart:
  // calling it with the previous step still in flight aborts.
  enum class AtSyncStatus { Continue, Started, Blocked };
  AtSyncStatus AtSyncStart(int waitForMigration=1);

  // The second half. If the step this element joined is still in flight, it
  // stops the element until the step is over -- under +LBAsync, until every
  // migration the step planned, on every PE, has completed -- and the element
  // is then handed back through ResumeFromSync(). If no step was joined, or the
  // step is already over, it calls ResumeFromSync() inline. Either way the
  // element is resumed exactly once per AtSyncStart(), so an application
  // written to the split pattern runs correctly with or without +LBAsync.
  void AtSyncWait();

  // True when the next AtSyncStart() will actually join a step, so an
  // application can do its pre-migration work -- draining device streams, say
  // -- only on the iterations where that work is needed.
  bool AtSyncPending() const;

  // True while this element owes AtSyncWait() to a step it has already joined,
  // including a step it was joined into by the runtime after AtSyncStart
  // returned Blocked: the application cannot see that join itself.
  bool AtSyncStepInFlight() const { return lbStepPending; }

  // Release an element that stopped in AtSyncWait() and was then migrated by
  // the step it was waiting on, arriving after its destination had already
  // resumed. Called from the migration arrival paths in CkLocMgr.
  void lbCheckWaitRelease();

  int MigrateToPe()  { return myRec->MigrateToPe(); }

private:
  void ResumeFromSyncHelper();
  AtSyncStatus lbStart(int waitForMigration, bool split);
  void lbJoinStep(int waitForMigration);
  void recordLBSizes();
  void sampleMetaLBLoad();
public:

  void ReadyMigrate(bool ready);
  void ckFinishConstruction(int epoch = -1);
  void setMigratable(int migratable);
  void setPupSize(size_t obj_pup_size);
  void setGPUPupSize(size_t obj_gpu_pup_size);
#else
  void AtSync(int waitForMigration=1) { ResumeFromSync();}
  enum class AtSyncStatus { Continue, Started, Blocked };
  AtSyncStatus AtSyncStart(int waitForMigration=1) { return AtSyncStatus::Continue; }
  void AtSyncWait() { ResumeFromSync(); }
  bool AtSyncPending() const { return false; }
  bool AtSyncStepInFlight() const { return false; }
  void lbCheckWaitRelease() { }
  void setMigratable(int migratable)  { }
  void setPupSize(size_t obj_pup_size) { }
  void setGPUPupSize(size_t obj_gpu_pup_size) { }
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
