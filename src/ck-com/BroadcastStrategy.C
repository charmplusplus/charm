/**
   @addtogroup ComlibCharmStrategy
   @{
   @file
*/

#include "BroadcastStrategy.h"
//#include "ComlibManager.h"

CkpvExtern(CkGroupID, cmgrID);
extern int sfactor;

/*
static void recv_bcast_handler(void *msg) {
    CmiMsgHeaderExt *conv_header = (CmiMsgHeaderExt *) msg;
    int instid = conv_header->stratid;

    BroadcastStrategy *bstrat = (BroadcastStrategy *)
        CProxy_ComlibManager(CkpvAccess(cmgrID)).ckLocalBranch()->getStrategy(instid);
    
    bstrat->handleMessage((char *)msg);    
}
*/

//Initialize the hypercube variables
void BroadcastStrategy::initHypercube() {
    logp = log((double) CkNumPes())/log(2.0);
    logp = ceil(logp);
}


//Constructor, 
//Can read spanning factor from command line
BroadcastStrategy::BroadcastStrategy(int topology) : 
    Strategy(), CharmStrategy(), _topology(topology) {
    spanning_factor = DEFAULT_BROADCAST_SPANNING_FACTOR;
    if(sfactor > 0)
        spanning_factor = sfactor;
    
    setType(GROUP_STRATEGY);
    initHypercube();
}

//Array Constructor
//Can read spanning factor from command line
BroadcastStrategy::BroadcastStrategy(CkArrayID aid, int topology) : 
    Strategy(), CharmStrategy(), _topology(topology) {
        
	CkAbort("BroadcastStrategy currently works only for groups");
    setType(ARRAY_STRATEGY);
    ainfo.setDestinationArray(aid);
    
    spanning_factor = DEFAULT_BROADCAST_SPANNING_FACTOR;
    if(sfactor > 0)
        spanning_factor = sfactor;    

    initHypercube();
    //if(topology == USE_HYPERCUBE)
    //  CkPrintf("Warning: hypercube only works on powers of two PES\n");
}


//Receives the message and sends it along the spanning tree.
void BroadcastStrategy::insertMessage(CharmMessageHolder *cmsg){
    //CkPrintf("[%d] BROADCASTING\n", CkMyPe());

    char *msg = cmsg->getCharmMessage();

    envelope *env = UsrToEnv(msg);
    CmiMsgHeaderExt *conv_header = (CmiMsgHeaderExt *) env;

    conv_header->root = 0;        //Use root later
    if(_topology == USE_HYPERCUBE) 
        conv_header->xhdl = 0;
    else
        //conv_header->root = CkMyPe();
        conv_header->xhdl = CkMyPe();
    
    CkPackMessage(&env);
    handleMessage((char *) env);
    
    delete cmsg;
}

//not implemented here because no bracketing is required for this strategy
//void BroadcastStrategy::doneInserting(){
//}

/*
//register the converse handler to recieve the broadcast message
void BroadcastStrategy::beginProcessing(int nelements) {
    handlerId = CkRegisterHandler((CmiHandler)recv_bcast_handler);
}
*/

void BroadcastStrategy::handleMessage(void *msg) {
    if(_topology == USE_TREE)
        handleTree(msg);
    else if(_topology == USE_HYPERCUBE) 
        handleHypercube(msg);
    else CkAbort("Unknown Topology");
}

void BroadcastStrategy::handleTree(void *msg){
    
    envelope *env = (envelope *)msg;
    CmiMsgHeaderExt *conv_header = (CmiMsgHeaderExt *) msg;

    int startpe = conv_header->xhdl;
    int size = env->getTotalsize();
    
    CkAssert(startpe>=0 && startpe < CkNumPes());
    
    CmiSetHandler(msg, CkpvAccess(comlib_handler));
    
    conv_header->stratid = getInstance();
    
    //Sending along the spanning tree
    //Gengbins tree building code stolen from the MPI machine layer    
    int i;
    for (i=1; i<=spanning_factor; i++) {
        
        int p = CkMyPe() - startpe;
        if (p<0) 
            p += CkNumPes();

        p = spanning_factor*p + i;

        if (p > CkNumPes() - 1) break;

        p += startpe;
        p = p % CkNumPes();

        CkAssert(p>=0 && p < CkNumPes() && p != CkMyPe());

        CmiSyncSend(p, size, (char*)msg);
    }

    if(getType() == GROUP_STRATEGY) {
      ComlibPrintf("BroadcastStrategy: delivering message\n");
        CkSendMsgBranch(env->getEpIdx(), EnvToUsr(env), CkMyPe(), 
                        env->getGroupNum());
    }
    else if(getType() == ARRAY_STRATEGY)
        ainfo.localBroadcast(env);        
}


void BroadcastStrategy::handleHypercube(void *msg){
    envelope *env = (envelope *)msg;

    CmiMsgHeaderExt *conv_header = (CmiMsgHeaderExt *) msg;
    //int curcycle = conv_header->root;
    int curcycle = conv_header->xhdl;

    int i;
    int size = env->getTotalsize();
        
    //CkPrintf("In hypercube %d, %d\n", (int)logp, curcycle); 
    
    /* assert(startpe>=0 && startpe<_Cmi_numpes); */
    CmiSetHandler(msg, CkpvAccess(comlib_handler));

    conv_header->stratid = getInstance();

    //Copied from system hypercube message passing

    for (i = logp - curcycle - 1; i >= 0; i--) {
        int p = CkMyPe() ^ (1 << i);

        int newcycle = ++curcycle;
        //CkPrintf("%d --> %d, %d\n", CkMyPe(), p, newcycle); 
        
        //conv_header->root = newcycle;
        conv_header->xhdl = newcycle;

        if(p >= CkNumPes()) {
            p &= (-1) << i;
            
            //loadbalancing
            if (p < CkNumPes())
                p += (CkMyPe() - 
                      (CkMyPe() & ((-1) << i))) % (CkNumPes() - p);
        }     
        
        if(p < CkNumPes())
            CmiSyncSendFn(p, size, (char*)msg);                    
    }
    
    if(getType() == GROUP_STRATEGY) {
      ComlibPrintf("BroadcastStrategy: delivering message\n");
        CkSendMsgBranch(env->getEpIdx(), EnvToUsr(env), CkMyPe(), 
                        env->getGroupNum());
    }
    else if(getType() == ARRAY_STRATEGY)
        ainfo.localBroadcast(env);        
}


//Pack the group id and the entry point of the user message
void BroadcastStrategy::pup(PUP::er &p){
    Strategy::pup(p);
    CharmStrategy::pup(p);    

    p | spanning_factor;
    p | _topology;
    p | logp;
}

/*@}*/
