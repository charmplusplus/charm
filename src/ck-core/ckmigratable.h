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
#if CMK_LB_USER_DATA
  void *getObjUserData(int idx);
#endif

#if CMK_LBDB_ON  //For load balancing:
  void AtSync(int waitForMigration=1);
  int MigrateToPe()  { return myRec->MigrateToPe(); }

private:
  void ResumeFromSyncHelper();
public:

  void ReadyMigrate(bool ready);
  void ckFinishConstruction(int epoch = -1);
  void setMigratable(int migratable);
  void setPupSize(size_t obj_pup_size);
#else
  void AtSync(int waitForMigration=1) { ResumeFromSync();}
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
