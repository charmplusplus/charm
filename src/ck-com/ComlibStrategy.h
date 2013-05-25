/**
   @addtogroup CharmComlib
   @{
   @file

   @brief Defines CharmStrategy and CharmMessageHolder.
*/

#ifndef COMLIBSTRATEGY_H
#define COMLIBSTRATEGY_H

#include "charm++.h"
#include "ckhashtable.h"
#include "convcomlibstrategy.h"
#include "ComlibLearner.h"
#include "envelope.h"

CkpvExtern(int, migrationDoneHandlerID);


/// Defines labels for distinguishing between types of sends in a CharmMessageHolder
enum CmhMessageType { CMH_ARRAYSEND, 
		      CMH_GROUPSEND, 
		      CMH_ARRAYBROADCAST, 
		      CMH_ARRAYSECTIONSEND, 
		      CMH_GROUPBROADCAST     };



/// Class managing Charm++ messages in the communication library.
/// It is aware of envelopes, arrays, etc
class CharmMessageHolder : public MessageHolder{
 public:
    /// An unused, and probably unnecessary array that was used to avoid a memory corruption that clobbers members in this class. The bug has likely been fixed.

  //#define BUGTRAPSIZE 5000

   //    int bug_trap[BUGTRAPSIZE];

    /// The section information for an enqueued multicast message
    CkSectionID *sec_id;

    /// Saves a copy of the CkSectionID sec_id when enqueing a multicast message
    CkSectionID *copy_of_sec_id;  

    CkArrayID array_id; 

    /// Type of message we are buffering
    CmhMessageType type; 
    
 CharmMessageHolder(char * msg, int proc, CmhMessageType t) : 
    MessageHolder((char *)UsrToEnv(msg), UsrToEnv(msg)->getTotalsize(), proc){
      type = t;
      sec_id = NULL;
      copy_of_sec_id = NULL;
      /*
      for(int i=0;i<BUGTRAPSIZE;i++){
	bug_trap[i] = 0;
      }
      
      checkme();
      */
    }
    
    /// Verfiy that the bug_trap array has not been corrupted. Noone should ever write to that array.
    /*
    void checkme() {
	for(int i=0;i<BUGTRAPSIZE;i++){
	  if(bug_trap[i] != 0){
	    CkPrintf("bug_trap[%d]=%d (should be 0) bug_trap[%d] is at %p\n", i, bug_trap[i], i, &bug_trap[i]);
	    CkAbort("Corruption of CharmMessageHolder has occured\n");
	  }
	}
    }
    */

    CharmMessageHolder(CkMigrateMessage *m) : MessageHolder(m) {
      /*
      for(int i=0;i<BUGTRAPSIZE;i++){
	bug_trap[i] = 0;
      }
      checkme();
      */
    }
    


    ~CharmMessageHolder(){
      //      checkme();
    }

    inline char * getCharmMessage() {
        return (char *)EnvToUsr((envelope *) data);
    }
    
    /// Store a local copy of the sec_id, so I can use it later.
    inline void saveCopyOf_sec_id(){
      //      ComlibPrintf("[%d] saveCopyOf_sec_id sec_id=%p NULL=%d\n", CkMyPe(), sec_id, NULL);

      //      checkme();

      if(sec_id!=NULL){

	//ComlibPrintf("Original has values: _nElems=%d, npes=%d\n", sec_id->_nElems, sec_id->npes );
	CkAssert(sec_id->_nElems>=0);
	CkAssert(sec_id->npes>=0);


	// Create a new CkSectionID, allocating its members
	copy_of_sec_id = new CkSectionID();
	copy_of_sec_id->_elems = new CkArrayIndex[sec_id->_nElems];
	copy_of_sec_id->pelist = new int[sec_id->npes];
	
	// Copy in the values
	copy_of_sec_id->_cookie = sec_id->_cookie;
	copy_of_sec_id->_nElems = sec_id->_nElems;
	copy_of_sec_id->npes = sec_id->npes;
	//	ComlibPrintf("Copy has values: _nElems=%d, npes=%d\n", copy_of_sec_id->_nElems, copy_of_sec_id->npes );
	for(int i=0; i<sec_id->_nElems; i++){
	  copy_of_sec_id->_elems[i] = sec_id->_elems[i];
	}
	for(int i=0; i<sec_id->npes; i++){
	  copy_of_sec_id->pelist[i] = sec_id->pelist[i];
	}

	// change local pointer to the new copy of the CkSectionID
	//	ComlibPrintf("saving copy of sec_id into %p\n", copy_of_sec_id);
      }

      //      checkme();

    }

    inline void freeCopyOf_sec_id(){
    /*   if(copy_of_sec_id != NULL){ */
/* 	CkPrintf("delete %p\n", sec_id); */
/* 	delete[] copy_of_sec_id->_elems; */
/* 	delete[] copy_of_sec_id->pelist; */
/* 	delete copy_of_sec_id; */
/* 	copy_of_sec_id = NULL; */
/*       } */

//      checkme();

    }


    virtual void pup(PUP::er &p);
    PUPable_decl(CharmMessageHolder);
};



// Info classes that help bracketed streategies manage objects.
// Each info class points to a list of source (or destination) objects.
// ArrayInfo also access the array listener interface

class ComlibNodeGroupInfo {
 protected:
    CkNodeGroupID ngid;
    int isNodeGroup;

 public:
    ComlibNodeGroupInfo();

    void setSourceNodeGroup(CkNodeGroupID gid) {
        ngid = gid;
        isNodeGroup = 1;
    }

    int isSourceNodeGroup(){return isNodeGroup;}
    CkNodeGroupID getSourceNodeGroup();

    void pup(PUP::er &p);
};

class ComlibGroupInfo {
 protected:
    CkGroupID sgid, dgid;
    int *srcpelist, nsrcpes; //src processors for the elements
    int *destpelist, ndestpes;
    int isSrcGroup;   
    int isDestGroup;

 public:
    ComlibGroupInfo();
    ~ComlibGroupInfo();

    int isSourceGroup(){return isSrcGroup;}
    int isDestinationGroup(){return isDestGroup;}

    void setSourceGroup(CkGroupID gid, int *srcpelist=0, int nsrcpes=0);    
    void getSourceGroup(CkGroupID &gid);
    void getSourceGroup(CkGroupID &gid, int *&pelist, int &npes);

    void setDestinationGroup(CkGroupID sgid,int *destpelist=0,int ndestpes=0);
    void getDestinationGroup(CkGroupID &gid);
    void getDestinationGroup(CkGroupID &dgid,int *&destpelist, int &ndestpes);

    /// This routine returnes an array of size CkNumPes() where each element
    /// follow the convention for bracketed strategies counts.
    int *getCombinedCountList();
    //void getCombinedPeList(int *&pelist, int &npes);
    void pup(PUP::er &p);
};

class ComlibMulticastMsg;

/** Array strategy helper class.
   Stores the source and destination arrays.
   Computes most recent processor maps of source and destinaton arrays.
   
   Array section helper functions, make use of sections easier for the
   communication library.
*/
class ComlibArrayInfo {
 private:

    CkArrayID src_aid; 	///< Source Array ID
    CkArrayID dest_aid; ///< Destination Array ID
    
    /**  Destination indices that are local to this PE 
	 (as determined by the bracketed counting scheme from a previous iteration)  
    */
    CkVec<CkArrayIndex> src_elements; 

    /**  Destination indices that are currently being updated. 
	 At the beginning of the next iteration these will be 
	 moved into dest_elements by useNewDestinationList()
    */
    CkVec<CkArrayIndex> new_src_elements;

    /**  Destination indices that are local to this PE 
	 (as determined by the bracketed counting scheme from a previous iteration)  
    */
    CkVec<CkArrayIndex> dest_elements; 

    /**  Destination indices that are currently being updated. 
	 At the beginning of the next iteration these will be 
	 moved into dest_elements by useNewDestinationList()
    */
    CkVec<CkArrayIndex> new_dest_elements;

    int isSrcArray;
    int isDestArray;

    bool isAllSrc; ///< if true then all the array is involved in the operation
    int totalSrc; ///< The total number of src elements involved in the strategy
    bool isAllDest; ///< if true then all the array is involved in the operation
    int totalDest; ///< The total number of array elements involved in the strategy
    
 public:
    ComlibArrayInfo();
    //~ComlibArrayInfo();

    /** Set the  source array used for this strategy. 
	The list of array indices should be the whole portion of the array involved in the strategy.
	The non-local array elements will be cleaned up inside purge() at migration of the strategy
    */
    void setSourceArray(CkArrayID aid, CkArrayIndex *e=0, int nind=0);
    int isSourceArray(){return isSrcArray;}
    CkArrayID getSourceArrayID() {return src_aid;}
    const CkVec<CkArrayIndex> & getSourceElements() {return src_elements;}

    /** Set the destination array used for this strategy. 
	The list of array indices should be the whole portion of the array involved in the strategy.
	The non-local array elements will be cleaned up inside purge() at migration of the strategy
    */
    void setDestinationArray(CkArrayID aid, CkArrayIndex *e=0, int nind=0);
    int isDestinationArray(){return isDestArray;}
    CkArrayID getDestinationArrayID() {return dest_aid;}
    const CkVec<CkArrayIndex> & getDestinationElements() {return dest_elements;}

    /// Get the number of source array elements
    int getTotalSrc() {return totalSrc;}
    int getLocalSrc() {return src_elements.size();}

    /** Add a destination object that is local to this PE to list used in future iterations */
    void addNewLocalDestination(const CkArrayIndex &e) {
        CkAssert(e.nInts > 0);
	new_dest_elements.push_back(e);
    }

    /** Add a source object that is local to this PE to list used in future iterations */
    void addNewLocalSource(const CkArrayIndex &e) {
        CkAssert(e.nInts > 0);
	new_src_elements.push_back(e);
    }
  
    /// Switch to using the new destination and source lists if the previous iteration found an error and constructed the new list
    void useNewSourceAndDestinations() {
      ComlibPrintf("[%d] useNewSourceAndDestinations previously had %d elements, now have %d elements\n", CkMyPe(), dest_elements.size(), new_src_elements.size() );
      dest_elements.removeAll();
      dest_elements = new_dest_elements;
      CkAssert(dest_elements.size() == new_dest_elements.size());
      new_dest_elements.removeAll();
      src_elements.removeAll();
      src_elements = new_src_elements;
      CkAssert(src_elements.size() == new_src_elements.size());
      new_src_elements.removeAll();
      if(com_debug)
	printDestElementList();
    }

    int newDestinationListSize(){ return new_dest_elements.size(); }
    int newSourceListSize(){ return new_src_elements.size(); }

    int getTotalDest() {return totalDest;}
    int getLocalDest() {return dest_elements.size();}

    void newElement(CkArrayID &id, const CkArrayIndex &idx);

    void localBroadcast(envelope *env);
    static int localMulticast(CkVec<CkArrayIndex> *idx_vec,envelope *env);
    static void deliver(envelope *env);

    /// This routine is called only once at the beginning and will take the list
    /// of source and destination elements and leave only those that have homePe
    /// here.
    void purge();

    /// This routine returnes an array of size CkNumPes() where each element
    /// follow the convention for bracketed strategies counts.
    int *getCombinedCountList();

    void printDestElementList();


    void pup(PUP::er &p);
};

/** Implementation of CkLocIterator to get all the local elements from a
    specified processor. Currently used by ComlibArrayInfo */
class ComlibElementIterator : public CkLocIterator {
 public:
  CkVec<CkArrayIndex> *list;

  ComlibElementIterator(CkVec<CkArrayIndex> *l) : CkLocIterator() {
    list = l;
  }

  virtual void addLocation (CkLocation &loc) {
    list->push_back(loc.getIndex());
  }
};



// forward declaration of wrapper for ComlibManager::registerArrayListener
void ComlibManagerRegisterArrayListener(CkArrayID, ComlibArrayInfo*);

/** All Charm++ communication library strategies should inherit from 
    CharmStrategy, rather than inheriting from class Strategy (or one of its
    subclasses). They should specify their object domain by setting
    Strategy::type. They have three helpers predefined for them for node groups,
    groups and arrays. */
class CharmStrategy {
    
 protected:
  //int forwardOnMigration;
    ComlibLearner *learner;
    bool mflag;    //Does this strategy handle point-to-point or 
    CkCallback onFinish;


    /** Deliver a message to a set of indices using the array manager. Indices can be local or remote. */
    int deliverToIndices(void *msg, int numDestIdxs, const CkArrayIndex* indices );
    
    /** Deliver a message to a set of indices using the array manager. Indices can be local or remote. */
    inline void deliverToIndices(void *msg, const CkVec< CkArrayIndex > &indices ){
      deliverToIndices(msg, indices.size(), indices.getVec() );
    }
    
    
 public:
    //int iterationNumber;
    ComlibGroupInfo ginfo;
    ComlibNodeGroupInfo nginfo;

    /// The communication library array listener watches and monitors
    /// the array elements belonging to ainfo.src_aid
    ComlibArrayInfo ainfo;
    
    CharmStrategy() { //: Strategy() {
      //setType(GROUP_STRATEGY); 
      //forwardOnMigration = 0;
      //iterationNumber = 0;
      learner = NULL;
      mflag = false;
    }

    CharmStrategy(CkMigrateMessage *m) { //: Strategy(m){
      learner = NULL;
    }

    //Set flag to optimize strategy for 
    inline void setMulticast(){
        mflag = true;
    }

    //get the multicast flag
    bool getMulticast () {
        return mflag;
    }

    inline void setOnFinish (CkCallback of) {
      onFinish = of;
    }

    inline CkCallback getOnFinish () {
      return onFinish;
    }

    
    ComlibLearner *getLearner() {return learner;}
    void setLearner(ComlibLearner *l) {learner = l;}
    
    virtual void pup(PUP::er &p);
 
};

//API calls which will be valid when communication library is not linked
void ComlibNotifyMigrationDone();
//int ComlibGetLastKnown(CkArrayID aid, CkArrayIndex idx);


#endif

/*@}*/
