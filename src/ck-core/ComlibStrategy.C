
#include "charm++.h"
#include "envelope.h"

//Class that defines the entry methods that a Charm level strategy
//must define.  To write a new strategy inherit from this class and
//define the virtual methods.  Every strategy can also define its own
//constructor and have any number of arguments. Also call the parent
//class methods in those methods.

void CharmStrategy::insertMessage(MessageHolder *mh){
    insertMessage((CharmMessageHolder *)mh);
}

void CharmStrategy::pup(PUP::er &p) {
    Strategy::pup(p);
    p | nginfo;
    p | ginfo;
    p | ainfo;
    p | forwardOnMigration;
}

CharmMessageHolder::CharmMessageHolder(char * msg, int proc) 
    : MessageHolder((char *)UsrToEnv(msg), proc, 
                    UsrToEnv(msg)->getTotalsize()){
    
    sec_id = NULL;    
}

CharmMessageHolder::~CharmMessageHolder() { 
}

char * CharmMessageHolder::getCharmMessage(){
    return (char *)EnvToUsr((envelope *) data);
}

void CharmMessageHolder::pup(PUP::er &p) {

    //    CkPrintf("In CharmMessageHolder::pup \n"); 

    MessageHolder::pup(p);

    //Sec ID depends on the message
    //Currently this pup is only being used for remote messages
    sec_id = NULL;
}

PUPable_def(CharmStrategy);
PUPable_def(CharmMessageHolder);

ComlibNodeGroupInfo::ComlibNodeGroupInfo() {
    isNodeGroup = 0;
    ngid.setZero();
};

void ComlibNodeGroupInfo::pup(PUP::er &p) {
    p | isNodeGroup;
    p | ngid;
}

ComlibGroupInfo::ComlibGroupInfo() {
    
    isGroup = 0;
    nsrcpes = 0;
    srcpelist = NULL;
    gid.setZero();
};

void ComlibGroupInfo::pup(PUP::er &p){

    p | gid;
    p | nsrcpes;

    if(p.isUnpacking()) {
        if(nsrcpes >0) 
            srcpelist = new int[nsrcpes];
    }

    if(nsrcpes > 0) 
        p(srcpelist, nsrcpes);
}

void ComlibGroupInfo::setSourceGroup(CkGroupID gid, int *pelist, 
                                         int npes) {
    this->gid = gid;
    srcpelist = pelist;
    nsrcpes = npes;
    isGroup = 1;
}

void ComlibGroupInfo::getSourceGroup(CkGroupID &gid, int *&pelist, 
                                         int &npes){
    gid = this->gid;

    pelist = srcpelist;
    npes = nsrcpes;
}


ComlibArrayInfo::ComlibArrayInfo() {

    src_aid.setZero();
    nSrcIndices = -1;
    src_elements = NULL;
    isSrcArray = 0;

    dest_aid.setZero();
    nDestIndices = -1;
    dest_elements = NULL;
    isDestArray = 0;
};

void ComlibArrayInfo::setSourceArray(CkArrayID aid, 
                                         CkArrayIndexMax *e, int nind){
    src_aid = aid;
    isSrcArray = 1;    
    nSrcIndices = nind;
    if(nind > 0) {
        src_elements = new CkArrayIndexMax[nind];
        memcpy(src_elements, e, sizeof(CkArrayIndexMax) * nind);
    }
}


void ComlibArrayInfo::getSourceArray(CkArrayID &aid, 
                                         CkArrayIndexMax *&e, int &nind){
    aid = src_aid;
    nind = nSrcIndices;
    e = src_elements;
}


void ComlibArrayInfo::setDestinationArray(CkArrayID aid, 
                                          CkArrayIndexMax *e, int nind){
    dest_aid = aid;
    isDestArray = 1;    
    nDestIndices = nind;
    if(nind > 0) {
        dest_elements = new CkArrayIndexMax[nind];
        memcpy(dest_elements, e, sizeof(CkArrayIndexMax) * nind);
    }
}


void ComlibArrayInfo::getDestinationArray(CkArrayID &aid, 
                                          CkArrayIndexMax *&e, int &nind){
    aid = dest_aid;
    nind = nDestIndices;
    e = dest_elements;
}


//Each strategy must define his own Pup interface.
void ComlibArrayInfo::pup(PUP::er &p){ 
    p | src_aid;
    p | nSrcIndices;
    p | isSrcArray;
    
    p | dest_aid;
    p | nDestIndices;
    p | isDestArray;
    
    if(p.isUnpacking() && nSrcIndices > 0) 
        src_elements = new CkArrayIndexMax[nSrcIndices];
    
    if(p.isUnpacking() && nDestIndices > 0) 
        dest_elements = new CkArrayIndexMax[nDestIndices];        
    
    if(nSrcIndices > 0)
        p((char *)src_elements, nSrcIndices * sizeof(CkArrayIndexMax));    
    else
        src_elements = NULL;

    if(nDestIndices > 0)
        p((char *)dest_elements, nDestIndices * sizeof(CkArrayIndexMax));    
    else
        dest_elements = NULL;

    //Insert all local elements into a vector
    if(p.isUnpacking() && !dest_aid.isZero()) {
        CkArray *dest_array = CkArrayID::CkLocalBranch(dest_aid);
        
        if(nDestIndices == 0){            
            dest_array->getComlibArrayListener()->getLocalIndices
                (localDestIndexVec);
        }
        else {
            for(int count = 0; count < nDestIndices; count++) 
                if(dest_array->lastKnown(dest_elements[count]) == CkMyPe())
                    localDestIndexVec.insertAtEnd(dest_elements[count]);
        }
    }
}

//Get the list of destination processors
void ComlibArrayInfo::getDestinationPeList(int *&destpelist, int &ndestpes) {
    
    int count = 0, acount =0;
    
    //Destination has not been set
    if(nDestIndices < 0) {
        destpelist = 0;
        ndestpes = 0;
        return;
    }

    //Create an array of size CkNumPes()
    //Inefficient in space
    ndestpes = CkNumPes();
    destpelist = new int[ndestpes];

    memset(destpelist, 0, ndestpes * sizeof(int));    

    if(nDestIndices == 0){
        for(count =0; count < CkNumPes(); count ++) 
            destpelist[count] = count;             
        return;
    }

    ndestpes = 0;

    //Find the last known processors of the array elements
    for(acount = 0; acount < nDestIndices; acount++) {
        int p = CkArrayID::CkLocalBranch(dest_aid)->
            lastKnown(dest_elements[acount]);        
        
        for(count = 0; count < ndestpes; count ++)
            if(destpelist[count] == p)
                break;
        
        if(count == ndestpes) {
            destpelist[ndestpes ++] = p; 
        }       
    }                            
}

void ComlibArrayInfo::getSourcePeList(int *&srcpelist, int &nsrcpes) {
    
    int count = 0, acount =0;

    if(nSrcIndices < 0) {
        srcpelist = 0;
        nsrcpes = 0;
        return;
    }

    nsrcpes = CkNumPes();
    srcpelist = new int[nsrcpes];

    memset(srcpelist, 0, nsrcpes * sizeof(int));    

    if(nSrcIndices == 0){
        for(count =0; count < CkNumPes(); count ++) 
            srcpelist[count] = count;             
        return;
    }

    nsrcpes = 0;
    for(acount = 0; acount < nSrcIndices; acount++) {
        int p = CkArrayID::CkLocalBranch(src_aid)->
            lastKnown(src_elements[acount]);        
        
        for(count = 0; count < nsrcpes; count ++)
            if(srcpelist[count] == p)
                break;
        
        if(count == nsrcpes) {
            srcpelist[nsrcpes ++] = p; 
        }       
    }                            
}

void ComlibArrayInfo::getCombinedPeList(int *&pelist, int &npes) {

    int count = 0;        
    pelist = 0;
    npes = 0;
    
    //Both arrays empty;
    //Sanity check, this should really not happen
    if(nSrcIndices < 0 && nDestIndices < 0) {
        CkAbort("Arrays have not been set\n");
        return;
    }
    
    //One of them is the entire array Hence set the number of
    //processors to all Currently does not work for the case where
    //number of array elements less than number of processors
    //Will fix it later!
    if(nSrcIndices == 0 || nDestIndices == 0) {
        npes = CkNumPes();        
        pelist = new int[npes];
        for(count = 0; count < CkNumPes(); count ++) 
            pelist[count] = count;                         
    }
    else {
        getDestinationPeList(pelist, npes);
        
        //Destination has not been set
        //Strategy does not care about destination
        //Is an error case
        if(npes == 0)
            pelist = new int[CkNumPes()];
        
        //Add source processors to the destination processors
        //already obtained
        for(int acount = 0; acount < nSrcIndices; acount++) {
            int p = CkArrayID::CkLocalBranch(src_aid)->
                lastKnown(src_elements[acount]);
            
            for(count = 0; count < npes; count ++)
                if(pelist[count] == p)
                    break;
            if(count == npes)
                pelist[npes ++] = p;
        }                        
    }
}

void ComlibArrayInfo::localBroadcast(envelope *env) {
    ComlibArrayInfo::localMulticast(&localDestIndexVec, env);
}

/*
  This method multicasts the message to all the indices in vec.  It
  also takes care to check if the entry method is readonly or not? If
  readonly (nokeep) the message is not copied.

  It also makes sure that the entry methods are logged in projections
  and that the array manager is notified about array element
  migrations.  Hence this function should be used extensively in the
  communication library strategies */

#include "register.h"
void ComlibArrayInfo::localMulticast(CkVec<CkArrayIndexMax>*vec,
                                     envelope *env){

    //Multicast the messages to all elements in vec
    int nelements = vec->size();
    if(nelements == 0) {
        CmiFree(env);
        return;
    }

    void *msg = EnvToUsr(env);
    int ep = env->getsetArrayEp();
    CkUnpackMessage(&env);

    CkArrayID dest_aid = env->getsetArrayMgr();
    env->setPacked(0);
    env->getsetArrayHops()=1;
    env->setUsed(0);

    for(int count = 0; count < nelements; count ++){
        CkArrayIndexMax idx = (*vec)[count];
        
        //if(comm_debug) idx.print();

        env->getsetArrayIndex() = idx;
        
        CkArray *a=(CkArray *)_localBranch(dest_aid);
        if(_entryTable[ep]->noKeep)
            a->deliver((CkArrayMessage *)msg, CkDeliver_inline, CmiFalse);
        else {
            void *newmsg = CkCopyMsg(&msg);
            a->deliver((CkArrayMessage *)newmsg, CkDeliver_queue, CmiTrue);
        }

    }

    CmiFree(env);
}

/* Delivers a message to an array element, making sure that
   projections is notified */
void ComlibArrayInfo::deliver(envelope *env){
    
    env->setUsed(0);
    env->getsetArrayHops()=1;
    CkUnpackMessage(&env);
    
    CkArray *a=(CkArray *)_localBranch(env->getsetArrayMgr());
    a->deliver((CkArrayMessage *)EnvToUsr(env), CkDeliver_queue, CmiTrue);    
}
