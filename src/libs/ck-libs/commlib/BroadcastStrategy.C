//Broadcast strategy for charm++ programs using the net version
//This stategy will wonly work for groups.
//This strategy implements a tree based broadcast
//I will extent it for arrays later.
//Developed by Sameer Kumar 04/10/04

#include "BroadcastStrategy.h"

CkpvExtern(CkGroupID, cmgrID);
extern int sfactor;

static void recv_bcast_handler(void *msg) {
    int instid = CmiGetXHandler(msg);
    BroadcastStrategy *bstrat = (BroadcastStrategy *)
        CProxy_ComlibManager(CkpvAccess(cmgrID)).ckLocalBranch()->getStrategy(instid);
    
    bstrat->handleMessage((char *)msg);    
}

//Constructor, 
//Can read spanning factor from command line
BroadcastStrategy::BroadcastStrategy() : Strategy() {
    spanning_factor = DEFAULT_BROADCAST_SPANNING_FACTOR;
    if(sfactor > 0)
        spanning_factor = sfactor;
    
}


//Receives the message and sends it along the spanning tree.
void BroadcastStrategy::insertMessage(CharmMessageHolder *cmsg){
    CkPrintf("[%d] BRAODCASTING\n", CkMyPe());

    char *msg = cmsg->getCharmMessage();
    handleMessage((char *)UsrToEnv(msg));
    
    delete cmsg;
}

//not implemented here because no bracketing is required for this strategy
void BroadcastStrategy::doneInserting(){
}


//register the converse handler to recieve the broadcast message
void BroadcastStrategy::beginProcessing(int nelements) {
    handlerId = CkRegisterHandler((CmiHandler)recv_bcast_handler);
}

void BroadcastStrategy::handleMessage(char *msg)
{
    envelope *env = (envelope *)msg;
    int startpe = env->getSrcPe();
    int size = env->getTotalsize();
    
    CkAssert(startpe>=0 && startpe < CkNumPes());
    
    CmiSetHandler(msg, handlerId);
    CmiSetXHandler(msg, getInstance());    
    
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

        CmiSyncSend(p, size, msg);
    }

    CkSendMsgBranch(env->getEpIdx(), EnvToUsr(env), CkMyPe(), 
                    env->getGroupNum());
}

//Pack the group id and the entry point of the user message
void BroadcastStrategy::pup(PUP::er &p){
    Strategy::pup(p);    
    p | spanning_factor;
}
