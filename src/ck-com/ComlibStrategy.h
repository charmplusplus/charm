#ifndef COMMLIBSTRATEGY_H
#define COMMLIBSTRATEGY_H

#include "charm++.h"
#include "ckhashtable.h"
#include "convcomlibstrategy.h"
#include "ComlibLearner.h"
#include "envelope.h"

CkpvExtern(int, migrationDoneHandlerID);

//Class managing Charm++ messages in the communication library.
//It is aware of envelopes, arrays, etc
class CharmMessageHolder : public MessageHolder{
 public:
    CkSectionID *sec_id;

    CharmMessageHolder() : MessageHolder() {sec_id = NULL;}
    CharmMessageHolder(CkMigrateMessage *m) : MessageHolder(m) {}
    
    CharmMessageHolder(char * msg, int dest_proc);
    ~CharmMessageHolder();

    inline char * getCharmMessage() {
        return (char *)EnvToUsr((envelope *) data);
    }
    
    virtual void pup(PUP::er &p);
    PUPable_decl(CharmMessageHolder);
};


//Struct to store the comlib location table info
struct ClibGlobalArrayIndex {
    CkArrayID aid;
    CkArrayIndexMax idx;

    //These routines allow ClibGlobalArrayIndex to be used in
    //  a CkHashtableT
    CkHashCode hash(void) const;
    static CkHashCode staticHash(const void *a,size_t);
    int compare(const ClibGlobalArrayIndex &ind) const;
    static int staticCompare(const void *a,const void *b,size_t);
};
PUPbytes(ClibGlobalArrayIndex);

/*********** CkHashTable functions ******************/
inline CkHashCode ClibGlobalArrayIndex::hash(void) const
{
    register CkHashCode ret = idx.hash() | (CkGroupID(aid).idx << 16);
    return ret;
}

inline int ClibGlobalArrayIndex::compare(const ClibGlobalArrayIndex &k2) const
{
    if(idx == k2.idx && aid == k2.aid)
        return 1;
    
    return 0;
}

//ClibGlobalArrayIndex CODE
inline int ClibGlobalArrayIndex::staticCompare(const void *k1, const void *k2, 
                                                size_t ){
    return ((const ClibGlobalArrayIndex *)k1)->
        compare(*(const ClibGlobalArrayIndex *)k2);
}

inline CkHashCode ClibGlobalArrayIndex::staticHash(const void *v,size_t){
    return ((const ClibGlobalArrayIndex *)v)->hash();
}


typedef CkHashtableT<ClibGlobalArrayIndex,int> ClibLocationTableType;
    
//Stores the location of many array elements used by the
//strategies.  Since hash table returns a reference to the object
//and for an int that will be 0, the actual value stored is pe +
//CkNumPes so 0 would mean processor -CkNumPes which is invalid.
CkpvExtern(ClibLocationTableType *, locationTable);

CkpvExtern(CkArrayIndexMax, cache_index);
CkpvExtern(int, cache_pe);
CkpvExtern(CkArrayID, cache_aid);

//Info classes that help bracketed streategies manage objects
//Each info class points to a list of source (or destination) objects
//ArrayInfo also access the array listener interface

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

    void getCombinedPeList(int *&pelist, int &npes);
    void pup(PUP::er &p);
};

class ComlibMulticastMsg;

/* Array strategy helper class.
   Stores the source and destination arrays.
   Computes most recent processor maps of source and destinaton arrays.
   
   Array section helper functions, make use of sections easier for the
   communication library.
*/

class ComlibArrayInfo {
 protected:
    CkArrayID src_aid;
    CkArrayIndexMax *src_elements; //src array elements
    int nSrcIndices;              //number of source indices   
    int isSrcArray;

    CkArrayID dest_aid;
    CkArrayIndexMax *dest_elements; //dest array elements
    int nDestIndices;              //number of destintation indices   
    int isDestArray;

    CkVec<CkArrayIndexMax> localDestIndexVec;
    
 public:
    ComlibArrayInfo();
    ~ComlibArrayInfo();

    void setSourceArray(CkArrayID aid, CkArrayIndexMax *e=0, int nind=0);
    int isSourceArray(){return isSrcArray;}
    void getSourceArray(CkArrayID &aid, CkArrayIndexMax *&e, int &nind);
    
    void setDestinationArray(CkArrayID aid, CkArrayIndexMax *e=0, int nind=0);
    int isDestinationArray(){return isDestArray;}
    void getDestinationArray(CkArrayID &aid, CkArrayIndexMax *&e, int &nind);

    void localBroadcast(envelope *env);
    static void localMulticast(CkVec<CkArrayIndexMax> *idx_vec,envelope *env);
    static void deliver(envelope *env);

    void getSourcePeList(int *&pelist, int &npes);
    void getDestinationPeList(int *&pelist, int &npes);
    void getCombinedPeList(int *&pelist, int &npes);
    
    void pup(PUP::er &p);
};


/* All Charm++ communication library strategies should inherit from
   this strategy. They should specify their object domain by setting
   Strategy::type. They have three helpers predefined for them for
   node groups, groups and arrays */

class CharmStrategy : public Strategy {
    
 protected:
    int forwardOnMigration;
    ComlibLearner *learner;
    CmiBool mflag;    //Does this strategy handle point-to-point or 

 public:
    ComlibGroupInfo ginfo;
    ComlibNodeGroupInfo nginfo;

    //The communication library array listener watches and monitors
    //the array elements belonging to ainfo.src_aid
    ComlibArrayInfo ainfo;
    
    CharmStrategy() : Strategy() {
        setType(GROUP_STRATEGY); 
        forwardOnMigration = 0;
        learner = NULL;
        mflag = CmiFalse;
    }

    CharmStrategy(CkMigrateMessage *m) : Strategy(m){
        learner = NULL;
    }

    //Set flag to optimize strategy for 
    inline void setMulticast(){
        mflag = CmiTrue;
    }

    //get the multicast flag
    CmiBool getMulticast () {
        return mflag;
    }

    //Called for each message
    //Function inserts a Charm++ message
    virtual void insertMessage(CharmMessageHolder *msg) {
        CkAbort("Bummer Should Not come here:CharmStrategy is abstract\n");
    }

    //Removed the virtual!
    //Charm strategies should not use Message Holder
    void insertMessage(MessageHolder *msg);
    
    //Added a new call that is called after the strategy had be
    //created on every processor.
    //DOES NOT exist in Converse Strategies
    virtual void beginProcessing(int nelements){}

    //Added a new call that is called after the strategy had be
    //created on every processor.  DOES NOT exist in Converse
    //Strategies. Called when the strategy is deactivated, possibly as
    //a result of a learning decision
    virtual void finalizeProcessing(){}

    //Called when a message is received in the strategy handler
    virtual void handleMessage(void *msg) {}
    
    ComlibLearner *getLearner() {return learner;}
    void setLearner(ComlibLearner *l) {learner = l;}
    
    virtual void pup(PUP::er &p);
    PUPable_decl(CharmStrategy);

    void setForwardOnMigration(int f) {
        forwardOnMigration = f;
    }
    
    int getForwardOnMigration() {
        return forwardOnMigration;
    }
};

//API calls which will be valid when communication library is not linked
void ComlibNotifyMigrationDone();
int ComlibGetLastKnown(CkArrayID aid, CkArrayIndexMax idx);

#endif
