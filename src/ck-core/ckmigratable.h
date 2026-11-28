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

  /// The balancer's replication decisions, for an element that declared itself
  /// replicable (setLBReplica): make a copy of this element on toPe with copy
  /// index `rank`; retire this copy. Delivered as soon as the decision reaches
  /// this PE, which may be any time in the element's step: the element records
  /// it and acts at a point of its own choosing, and it alone moves the state
  /// (the runtime does not pack a copy). The defaults abort for a replicable
  /// element that did not override them and do nothing for any other element.
  virtual void ckClone(int toPe, int rank);
  virtual void ckRetire(void);
  /// The balancer makes a copy of (or retires) another copy of this element's group: `rank` is the
  /// copy made on `toPe` (or the copy retired). Delivered before the round releases the element,
  /// so every copy can act in the same step. Default: nothing.
  virtual void ckCloneNotice(int toPe, int rank) { (void)toPe; (void)rank; }
  virtual void ckRetireNotice(int rank) { (void)rank; }
  // The location manager's path to ckClone (CkLocRec::staticClone); not for
  // applications.
  void ckCloneFromDecision(void* args);
  void ckCloneNoticeFromDecision(void* args);
  void ckRetireNoticeFromDecision(void* args);

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
  // Register this object's position for geometric load balancers (DiffusionLB's
  // centroid metric). Optional: left uncalled, the position stays empty and
  // centroid-based balancing has nothing to work with.
  void setObjPosition(const std::vector<LBRealType>& pos);
  double getObjTime();
  void setObjGPUTime(double cputime);
  double getObjGPUTime();
#if CMK_LB_USER_DATA
  void *getObjUserData(int idx);
#endif

#if CMK_LBDB_ON  //For load balancing:
  void AtSync(int waitForMigration=1);
  int MigrateToPe()  { return myRec->MigrateToPe(); }

private:
  void ResumeFromSyncHelper();
  // The replication declaration (setLBReplica), kept on the element so it
  // travels with it and is declared again to the database of every PE it
  // reaches. lbReplicaGroup < 0 means not replicable, the default.
  int lbReplicaGroup = -1;
  int lbReplicaRank = 0;
  int lbReplicaCount = 1;
  int lbReplicaMax = 1;
  size_t lbReplicaExtraBytes = 0;
  double lbReplicaCostNode = 0.0;
  double lbReplicaCostRemote = 0.0;
  int lbReplicaParts = 0;
  size_t lbReplicaExchangeBytes = 0;
  double lbReplicaExchangeMsgs = 0.0;
  size_t lbReplicaExtraBytesPerPeer = 0;
public:

  void ReadyMigrate(bool ready);
  void ckFinishConstruction(int epoch = -1);
  void setMigratable(int migratable);
  void setPupSize(size_t obj_pup_size);
  void setGPUPupSize(size_t obj_gpu_pup_size);
  // Declare this element one of `count` copies of logical object `group`
  // (its copy index `rank`), of which the application can keep up to
  // `maxCount` consistent; `extraBytes` is what one copy holds on its device
  // beyond the element's own footprint. A balancer that understands groups
  // (ShedLB) may then ask for a copy with ckClone or take one back with
  // ckRetire. Call it before every AtSync, as with setObjGPUTime; an element
  // that never calls it is never replicated. `costNode` / `costRemote`: the
  // measured load one more copy adds to each copy per balancing interval for
  // its exchange with that peer, on the same node or across nodes (0 while
  // unmeasured); `parts`, `exchangeBytes`, `exchangeMsgs` and `extraPerPeer`
  // as LDObjData::replicaParts and the fields after it describe them.
  void setLBReplica(int group, int rank, int count, int maxCount, size_t extraBytes,
                    double costNode = 0.0, double costRemote = 0.0, int parts = 0,
                    size_t exchangeBytes = 0, double exchangeMsgs = 0.0,
                    size_t extraPerPeer = 0);
  bool lbReplicable() const { return lbReplicaGroup >= 0 && lbReplicaMax > 1; }
#else
  void AtSync(int waitForMigration=1) { ResumeFromSync();}
  void setMigratable(int migratable)  { }
  void setPupSize(size_t obj_pup_size) { }
  void setGPUPupSize(size_t obj_gpu_pup_size) { }
  void setLBReplica(int, int, int, int, size_t, double = 0.0, double = 0.0, int = 0, size_t = 0,
                    double = 0.0, size_t = 0) {}
  bool lbReplicable() const { return false; }
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
