/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include "graphrouter.h"
#include "hypercubetopology.h"

#define gmap(pe) {if (gpes) pe=gpes[pe];}

GraphRouter::GraphRouter(int n, int me){
    init(n, me, new HypercubeTopology(n, me));
}

void GraphRouter::init(int n, int me, TopologyDescriptor *tp)
{  
    NumPes=n;
    MyPe=me;
    gpes=NULL;
    this->tp = tp;
    
    PeGraph = new PeTable(NumPes);
    pesToSend = new int[NumPes];
    nstages = tp->getNumStages() + 1;
    currentIteration = 0;
    
    stageComplete = new int[nstages];
    recvExpected = new int[nstages];
    recvCount = new int[nstages];

    bzero(stageComplete, nstages * sizeof(int));
    bzero(recvCount, nstages *sizeof(int));
    for(int count = 1; count < nstages; count++)
        recvExpected[count] = tp->getNumMessagesExpected(count);

    curStage = 0;
    ComlibPrintf("me=%d NUMPES=%d nstages=%d\n", MyPe, n, nstages);
}

GraphRouter::~GraphRouter()
{
    delete PeGraph;
    delete pesToSend;
    delete tp;
    delete [] stageComplete;
    delete [] recvExpected;
    delete [] recvCount;
    delete [] neighborPeList;
}

void GraphRouter::NumDeposits(comID, int num)
{
}

void GraphRouter::EachToAllMulticast(comID id, int size, void *msg, int more)
{
    int npe=NumPes;
    int * destpes=(int *)CmiAlloc(sizeof(int)*npe);
    for (int i=0;i<npe;i++) destpes[i]=i;
    EachToManyMulticast(id, size, msg, npe, destpes, more);
}

void GraphRouter::sendMessages(int cur_stage){
    int nsteps = tp->getNumSteps(cur_stage);
    int nextpe = 0, npestosend = 0;
    
    for(int stepcount = 0; stepcount < nsteps; stepcount ++){
        tp->getPesToSend(stepcount, cur_stage, npestosend, pesToSend, nextpe);
        
        gmap(nextpe);
        ComlibPrintf("%d:sending to %d for %d pes in stage %d\n", MyPe, nextpe, npestosend, cur_stage);

        int len;
	char *newmsg;
        newmsg=PeGraph->ExtractAndPack(MyID, cur_stage + 1, npestosend, 
                                       pesToSend, &len);
        
#if CMK_PERSISTENT_COMM
        if(len < PERSISTENT_BUFSIZE)
            if(currentIteration % 2)
                CmiUsePersistentHandle(&handlerArrayOdd[cur_stage], 1);
            else
                CmiUsePersistentHandle(&handlerArrayEven[cur_stage], 1);
#endif          
        
	if (newmsg) {
            if(cur_stage < nstages - 2)
                CmiSetHandler(newmsg, CpvAccess(RecvHandle));
            else
                CmiSetHandler(newmsg, CpvAccess(ProcHandle));

            CmiSyncSendAndFree(nextpe, len, newmsg);
        }
	else {
            KSendDummyMsg(MyID, nextpe, cur_stage + 1);
	}
        
#if CMK_PERSISTENT_COMM
        if(len < PERSISTENT_BUFSIZE)
            CmiUsePersistentHandle(NULL, 0);
#endif          
    }
}

void GraphRouter::EachToManyMulticast(comID id, int size, void *msg, 
                                      int numpes, int *destpes, int more)
{
    PeGraph->InsertMsgs(numpes, destpes, size, msg);
    if (more) return;

    ComlibPrintf("All messages received %d\n", MyPe);
    sendMessages(0);

    curStage = 1;

    int stage_itr;
    for(stage_itr = curStage; stage_itr < nstages - 1; stage_itr ++){
        if(stageComplete[stage_itr]){
            sendMessages(stage_itr);
            stageComplete[stage_itr] = 0;
        }
        else break;
    }
    curStage = stage_itr;
    if(curStage == nstages - 1)
        ProcManyMsg(id, NULL);
    else 
        PeGraph->ExtractAndDeliverLocalMsgs(MyPe);
}

void GraphRouter::RecvManyMsg(comID id, char *msg)
{
    int stage = 0;
    stage = PeGraph->UnpackAndInsert(msg);
    
    recvCount[stage] ++;
    if (recvCount[stage] == recvExpected[stage]) {
        ComlibPrintf("%d recvcount=%d recvexpected = %d stage=%d\n", MyPe, recvCount[stage], recvExpected[stage], stage);
        
        recvCount[stage] = 0;
        stageComplete[stage] = 1;
    }
    
    int stage_itr;
    for(stage_itr = curStage; stage_itr < nstages - 1; stage_itr ++){
        if(stageComplete[stage_itr]){
            sendMessages(stage_itr);
            stageComplete[stage_itr] = 0;
        }
        else break;
    }
    curStage = stage_itr;
    if(curStage == nstages - 1)
        ProcManyMsg(id, NULL);
    else 
        PeGraph->ExtractAndDeliverLocalMsgs(MyPe);
}

void GraphRouter::DummyEP(comID id, int stage)
{
    if(stage < nstages - 1) {
        recvCount[stage] ++;
        if (recvCount[stage] == recvExpected[stage]) {
            ComlibPrintf("%d DUMMY recvcount=%d recvexpected = %d refno=%d\n", MyPe, recvCount[stage], recvExpected[stage], KMyActiveRefno(MyID));
            recvCount[stage] = 0;
            stageComplete[stage] = 1;
        }

        int stage_itr;
        for(stage_itr = curStage; stage_itr < nstages - 1; stage_itr ++){
            if(stageComplete[stage_itr]){
                sendMessages(stage_itr);
                stageComplete[stage] = 0;
            }
            else break;
        }
        curStage = stage_itr;
        if(curStage == nstages - 1)
            ProcManyMsg(id, NULL);
        else 
            PeGraph->ExtractAndDeliverLocalMsgs(MyPe);
    }
    else 
        ProcManyMsg(id, NULL);
}

void GraphRouter:: ProcManyMsg(comID id, char *m)
{
    int stage = nstages - 1;
    if(m) {
        PeGraph->UnpackAndInsert(m);
        recvCount[stage] ++;
    }

    if(recvCount[stage] == recvExpected[stage]) {
        ComlibPrintf("%d proc many msg %d\n", MyPe, stage);
        stageComplete[stage] = 1;
    }
    else 
        return;
    
    if(curStage != nstages -1)
        return;

    currentIteration ++;
    recvCount[stage] = 0;
    PeGraph->ExtractAndDeliverLocalMsgs(MyPe);
    
    PeGraph->Purge();
    curStage = 0;
    KDone(MyID);
}

Router * newgraphobject(int n, int me)
{
    ComlibPrintf("In create graph router \n");
    Router *obj = new GraphRouter(n, me);
    return(obj);
}

void GraphRouter::SetID(comID id)
{
    MyID=id;
    if(id.NumMembers == NumPes)
        SetMap(NULL);
}

void GraphRouter :: SetMap(int *pes)
{
    gpes=pes;

#if CMK_PERSISTENT_COMM
    numNeighbors=0;
    neighborPeList = new int[NumPes];

    tp->getNeighbors(numNeighbors, neighborPeList);
    handlerArrayOdd = new PersistentHandle[numNeighbors];
    handlerArrayEven = new PersistentHandle[numNeighbors];

    //Persistent handlers for all the neighbors
    int pcount = 0;
    for (pcount = 0; pcount < numNeighbors; pcount++) {
        int dest = neighborPeList[pcount];
        gmap(dest);
        ComlibPrintf("%d:Creating Persistent Buffer of size %d at %d\n", MyPe,
                     PERSISTENT_BUFSIZE, dest);
        handlerArrayOdd[pcount] = CmiCreatePersistent(dest, 
                                                      PERSISTENT_BUFSIZE);
        ComlibPrintf("%d:Creating Even Persistent Buffer of size %d at %d\n",
                     MyPe, PERSISTENT_BUFSIZE, dest);
        handlerArrayEven[pcount] = CmiCreatePersistent(dest, 
                                                       PERSISTENT_BUFSIZE);
    }
#endif
}
