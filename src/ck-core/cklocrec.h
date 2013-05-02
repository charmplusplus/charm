#ifndef CK_LOC_REC_H
#define CK_LOC_REC_H

class CkArray;//Array manager
class CkLocMgr;//Location manager
class CkMigratable;//Migratable object

/**
 * A CkLocRec is our local representation of an array element.
 * The location manager's main hashtable maps array indices to
 * CkLocRec *'s.
 */
class CkLocRec {
protected:
  CkLocMgr *myLocMgr;
public:
  CkLocRec(CkLocMgr *mgr) :myLocMgr(mgr) { }
  virtual ~CkLocRec();

  /// Return the type of this ArrayRec:
  typedef enum {
    base=0,//Base class (invalid type)
    local,//Array element that lives on this Pe
    remote,//Array element that lives on some other Pe
    buffering,//Array element that was just created
  } RecType;
  virtual RecType type(void)=0;
};

/**
 * Represents a local array element.
 */
class CkLocRec_local : public CkLocRec {
  CkArrayIndex idx;/// Element's array index
  bool running; /// True when inside a startTiming/stopTiming pair
  bool *deletedMarker; /// Set this if we're deleted during processing
public:
  CkQ<CkArrayMessage *> halfCreated; /// Stores messages for nonexistent siblings of existing elements
  //Creation and Destruction:
  CkLocRec_local(CkLocMgr *mgr,bool fromMigration,bool ignoreArrival, const CkArrayIndex &idx_);
  void migrateMe(int toPe); //Leave this processor
  void informIdealLBPeriod(int lb_ideal_period);
  void metaLBCallLB();
  void destroy(void); //User called destructor
  virtual ~CkLocRec_local();

  /// A new element has been added to this index
  void addedElement(void);

  /** Invoke the given entry method on this element.
   *   Returns false if the element died during the receive.
   *   If doFree is true, the message is freed after send;
   *    if false, the message can be reused.
   */
  bool invokeEntry(CkMigratable *obj,void *msg,int idx,bool doFree);

  virtual RecType type(void);

#if CMK_LBDB_ON  //For load balancing:
  /// Control the load balancer:
  void startTiming(int ignore_running=0);
  void stopTiming(int ignore_running=0);
  void setObjTime(double cputime);
  double getObjTime();
  void *getObjUserData(int idx);
#else
  inline void startTiming(int ignore_running=0) {  }
  inline void stopTiming(int ignore_running=0) { }
#endif
  inline const CkArrayIndex &getIndex(void) const {return idx;}

#if CMK_LBDB_ON
public:
  inline LBDatabase *getLBDB(void) const {return the_lbdb;}
  inline MetaBalancer *getMetaBalancer(void) const {return the_metalb;}
  inline LDObjHandle getLdHandle() const{return ldHandle;}
  static void staticMigrate(LDObjHandle h, int dest);
  static void staticMetaLBResumeWaitingChares(LDObjHandle h, int lb_ideal_period);
  static void staticMetaLBCallLBOnChares(LDObjHandle h);
  void metaLBResumeWaitingChares(int lb_ideal_period);
  void metaLBCallLBOnChares();
  void recvMigrate(int dest);
  void setMigratable(int migratable);	/// set migratable
  void setPupSize(size_t obj_pup_size);
  void AsyncMigrate(bool use);
  bool isAsyncMigrate()   { return asyncMigrate; }
  void ReadyMigrate(bool ready) { readyMigrate = ready; } ///called from user
  int  isReadyMigrate()	{ return readyMigrate; }
  bool checkBufferedMigration();	// check and execute pending migration
  int   MigrateToPe();
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
        void Migrated();
#endif
  inline void setMeasure(bool status) { enable_measure = status; }
private:
  LBDatabase *the_lbdb;
  MetaBalancer *the_metalb;
  LDObjHandle ldHandle;
  bool  asyncMigrate;  /// if readyMove is inited
  bool  readyMigrate;    /// status whether it is ready to migrate
  bool  enable_measure;
  int  nextPe;              /// next migration dest processor
#else
  void AsyncMigrate(bool use){};
#endif
/**FAULT_EVAC*/
private:
	bool asyncEvacuate; //can the element be evacuated anytime, false for tcharm
	bool bounced; //did this element try to immigrate into a processor which was evacuating
											// and was bounced away to some other processor. This is assumed to happen
											//only if this object was migrated by a load balancer, but the processor
											// started crashing soon after
public:	
	bool isAsyncEvacuate(){return asyncEvacuate;}
	void AsyncEvacuate(bool set){asyncEvacuate = set;}
	bool isBounced(){return bounced;}
	void Bounced(bool set){bounced = set;}
};
class CkLocRec_remote;

#endif // CK_LOC_REC_H

