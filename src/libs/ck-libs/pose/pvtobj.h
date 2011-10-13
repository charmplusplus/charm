/// pvtObjects: a list to hold records of posers registered with a PVT branch.
/** Implements a simple list of object records stored one per PVT branch.
    Provides a means to perform common operations (gather safe times, fossil
    colleciton, etc) on all registered objectssy on a processor. */
#ifndef PVTOBJ_H
#define PVTOBJ_H

/// A pvtObjects entry for storing poser data
/** This class is used in pvtObjects to store poser data local to a
    processor. */
class pvtObjectNode {
  /// Last reported safe time of poser
  POSE_TimeType ovt, ovt2;
  /// Index of poser in POSE_Objects array
  int index;  
  /// Flag to indicate if object data is stored at this index
  /** present==true indicates that this node contains a valid object, present==false
      indicates the node can be recycled */
  bool present;
  /// The synchronization strategy of the poser (OPTIMISTIC or CONSERVATIVE)
  short int sync; 
  /// Time spent executing events on this object within a DOP_QUANTA
  double qdo;
 public:
  /// A pointer to the actual poser
  sim *localObjPtr;
  /// Basic Constructor
  pvtObjectNode() : ovt(POSE_UnsetTS), ovt2(POSE_UnsetTS), index(-1), present(false), sync(0), qdo(0.0) {  }
  /// Sets all data fields
  inline void set(POSE_TimeType ts, int idx, bool on, short int s, sim *p) {
    ovt = ts; index = idx; present = on; sync = s; localObjPtr = p; qdo = 0.0;
    ovt2 = POSE_UnsetTS;
  }
  /// Sets ovt to -1 to indicate idle
  inline void setIdle() { ovt = ovt2 = POSE_UnsetTS; }
  /// Test present flag
  inline bool isPresent() { return present; }
  /// Test if synchronization strategy is optimistic
  inline int isOptimistic() { return (sync == OPTIMISTIC); }
  /// Test if synchronization strategy is conservative
  inline int isConservative() { return (sync == CONSERVATIVE); }
  /// Return ovt
  inline POSE_TimeType getOVT() { return ovt; }
  /// Return ovt2
  inline POSE_TimeType getOVT2() { return ovt2; }
  /// Set ovt to st
  inline void setOVT(POSE_TimeType st) { ovt = st; }
  /// Set ovt2 to st
  inline void setOVT2(POSE_TimeType st) { ovt2 = st; }
  /// Add time to qdo
  inline void addQdoTime(double t) { qdo += t; }
  /// Return qdo
  inline double getQdo() { return qdo; }
  /// Reset qdo at start of quanta
  inline void resetQdo() { qdo = 0.0; }
  /// Dump data fields
  void dump() {
    if (localObjPtr == NULL)
      CkPrintf("ovt=%d index=%d present=%s sync=%s ptr=NULL",
	       ovt, index, present?"true":"false", (sync==0)?"OPT":"CON");
    else 
      CkPrintf("ovt=%d index=%d present=%s sync=%s ptr!=NULL",
	       ovt, index, present?"true":"false", (sync==0)?"OPT":"CON");
  }
  /// Check validity of data fields
  void sanitize();
};

/// List to hold records of posers registered with a PVT branch.
class pvtObjects {
  /// Number of posers present in the list
  int numObjs;
  /// number of consecutive spaces in list that are or have been occupied
  int numSpaces;
  /// number of spaces allocated in objs
  int size;
  /// lowest index of an empty slot in objs
  int firstEmpty; 
  /// counter for strat calculations
  int stratIterCount;
 public:
  /// the list of posers
  pvtObjectNode *objs;
  /// Basic Constructor: preallocates space for 100 objects
  pvtObjects();    
  /// Get number of objects in the list
  inline int getNumObjs() { return numObjs; }
  /// Get number of spaces in use in list
  inline int getNumSpaces() { return numSpaces; }
  /// Set posers to idle (ovt==-1)
  inline void SetIdle() { 
    register int i; 
    for (i=0; i<numSpaces; i++) objs[i].setIdle();
  }                           
  /// Wake up all posers in list
  void Wake();
  void callAtSync();
  /// Call Commit on all posers
  void Commit();
  /// Call CheckpointCommit on all posers
  void CheckpointCommit();
  /// Perform synchronization strategy calculations
  void StratCalcs();
  /// Insert poser in list
  /** Inserts an object in the list in the firstEmpty slot, expanding the list
      size if necessary */
  int Insert(int index, POSE_TimeType ovt, int sync, sim *myPtr); 
  /// Delete a poser from the list
  inline void Delete(int idx) {
    objs[idx].set(POSE_UnsetTS, POSE_UnsetTS, false, 0, NULL);
    numObjs--;
    if (idx < firstEmpty) firstEmpty = idx; // recalculate firstEmpty
  }                       
  /// Dump data fields
  void dump();
  /// Check validity of data fields
  void sanitize();
};
  
#endif
