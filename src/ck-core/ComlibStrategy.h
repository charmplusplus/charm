#ifndef COMMLIBSTRATEGY_H
#define COMMLIBSTRATEGY_H

#include "charm++.h"
#include "convcomlibstrategy.h"

//Class managing Charm++ messages in the communication library.
//It is aware of envelopes, arrays, etc
class CharmMessageHolder : public MessageHolder{
 public:
    CkSectionID *sec_id;

    CharmMessageHolder() : MessageHolder() {sec_id = NULL;}
    CharmMessageHolder(CkMigrateMessage *m) : MessageHolder(m) {}
    
    CharmMessageHolder(char * msg, int dest_proc);
    ~CharmMessageHolder();

    char * getCharmMessage();
    
    virtual void pup(PUP::er &p);
    PUPable_decl(CharmMessageHolder);
};

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
    CkGroupID gid;
    int *srcpelist, nsrcpes; //src processors for the elements
    int isGroup;   

 public:
    ComlibGroupInfo();

    void setSourceGroup(CkGroupID gid, int *srcpelist=0, int nsrcpes=0);
    int isSourceGroup(){return isGroup;}
    void getSourceGroup(CkGroupID &gid, int *&pelist, int &npes);
    
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

    void setSourceArray(CkArrayID aid, CkArrayIndexMax *e=0, int nind=0);
    int isSourceArray(){return isSrcArray;}
    void getSourceArray(CkArrayID &aid, CkArrayIndexMax *&e, int &nind);
    
    void setDestinationArray(CkArrayID aid, CkArrayIndexMax *e=0, int nind=0);
    int isDestinationArray(){return isDestArray;}
    void getDestinationArray(CkArrayID &aid, CkArrayIndexMax *&e, int &nind);

    void localMulticast(envelope *env);
    void localMulticast(CkVec<CkArrayIndexMax> *idx_vec,envelope *env);

    void getSourcePeList(int *&pelist, int &npes);
    void getDestinationPeList(int *&pelist, int &npes);
    void getCombinedPeList(int *&pelist, int &npes);
    
    void initSectionID(CkSectionID *sid);
    ComlibMulticastMsg * getNewMulticastMessage(CharmMessageHolder *cmsg);

    void pup(PUP::er &p);
};


/* All Charm++ communication library strategies should inherit from
   this strategy. They should specify their object domain by setting
   Strategy::type. They have three helpers predefined for them for
   node groups, groups and arrays */

class CharmStrategy : public Strategy {
    int forwardOnMigration;

 public:
    ComlibGroupInfo ginfo;
    ComlibNodeGroupInfo nginfo;
    ComlibArrayInfo ainfo;    

    CharmStrategy() : Strategy() {setType(GROUP_STRATEGY); forwardOnMigration = 0;}
    CharmStrategy(CkMigrateMessage *m) : Strategy(m){}

    //Called for each message
    //Function inserts a Charm++ message
    virtual void insertMessage(CharmMessageHolder *msg) {
        CkAbort("Bummer Should Not come here:CharmStrategy is abstract\n");
    }

    //Removed the virtual!
    //Charm strategies should not use Message Holder
    void insertMessage(MessageHolder *msg);
    
    //Called after all chares and groups have finished depositing their 
    //messages on that processor.
    virtual void doneInserting() {}

    //Added a new call that is called after the strategy had be
    //created on every processor.
    //DOES NOT exist in Converse Strategies
    virtual void beginProcessing(int nelements){}

    virtual void pup(PUP::er &p);
    PUPable_decl(CharmStrategy);

    void setForwardOnMigration(int f) {
        forwardOnMigration = f;
    }
    
    int getForwardOnMigration() {
        return forwardOnMigration;
    }
};

#endif
