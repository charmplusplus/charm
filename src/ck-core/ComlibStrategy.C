#include "charm++.h"
#include "envelope.h"

//Class that defines the entry methods that a strategy must define.
//To write a new strategy inherit from this class and define the
//virtual methods.  Every strategy can also define its own constructor
//and have any number of arguments. Also call the parent class methods
//in those methods.

Strategy::Strategy() {
    isGroup = 0;
    isArray = 0;
    nIndices = -1;
    elements = NULL;
};

void Strategy::setSourceArray(CkArrayID aid, CkArrayIndexMax *e, int nind){
    this->aid = aid;
    elements = new CkArrayIndexMax[nind];
    nIndices = nind;
    memcpy(elements, e, sizeof(CkArrayIndexMax) * nind);
    isArray = 1;
}

void Strategy::setSourceGroup(CkGroupID gid, int *pelist, int npes) {
    this->gid = gid;
    isGroup = 1;

    srcpelist = pelist;
    nsrcpes = npes;
}

void Strategy::getSourceArray(CkArrayID &aid, CkArrayIndexMax *&e, int &nind){
    aid = this->aid;
    nind = nIndices;
    e = elements;
}

void Strategy::getSourceGroup(CkGroupID &gid, int *&pelist, int &npes){
    gid = this->gid;

    pelist = srcpelist;
    npes = nsrcpes;
}

/*
void Strategy::removeLocalIndex(CkArrayIndexMax idx) {
    
    for(int count = 0; count < localSrcIndices.length(); count++)
        if(memcmp(&localSrcIndices[count], &idx, sizeof(CkArrayIndexMax)) 
           == 0)
            localSrcIndices.remove(count);
}
*/

//Each strategy must define his own Pup interface.
void Strategy::pup(PUP::er &p){ 
    p | isArray;
    p | isGroup;
    p | aid;
    p | gid;
    p | nIndices;
    p | nsrcpes;
    
    if(p.isUnpacking()) {
        if(nIndices > 0) 
            elements = new CkArrayIndexMax[nIndices];
        if(nsrcpes >0) 
            srcpelist = new int[nsrcpes];
    }
    
    //Unable to get this going, 
    //cannot pack a list of indices!!! Check with Orion
    //Packing them as bytes
    if(nIndices > 0)
        p((char *)elements, nIndices * sizeof(CkArrayIndexMax));    

    if(nsrcpes >0) 
        p(srcpelist, nsrcpes);
}

CharmMessageHolder::CharmMessageHolder(char * msg, int proc) {
    data = (char *)UsrToEnv(msg);
    dest_proc = proc;
    isDummy = 0;

    nIndices =0;
    npes = 0;
    
    indexlist = 0;
    pelist = 0;
}

CharmMessageHolder::~CharmMessageHolder() {
    
    if(pelist != NULL && npes > 0)
        delete[] pelist;

    if(indexlist != NULL && nIndices > 0)
        delete [] indexlist;
}

char * CharmMessageHolder::getCharmMessage(){
    return (char *)EnvToUsr((envelope *) data);
}


void StrategyWrapper::pup (PUP::er &p) {

    //CkPrintf("In PUP of StrategyWrapper\n");

    p | nstrats;
    if(p.isUnpacking())
	s_table = new Strategy * [nstrats];
    
    for(int count = 0; count < nstrats; count ++)
        p | s_table[count];
}
