#include "immediatering.h"

static CProxy_immRing_nodegroup immring_nodegrp;
static CProxy_immRing_group     immring_grp;

// testing Converse immeidate message
int immediatering_startHandlerIdx=0;
int immediatering_finishHandlerIdx=0;

static int numTests = 0;

// testing Converse message
static void sendImmediate(int destNode,int iter) {
  immediateMsg *msg=(immediateMsg *)CmiAlloc(sizeof(immediateMsg));
  msg->iter=iter;
  sprintf(msg->data, "Array!");
  CmiSetHandler(msg, immediatering_startHandlerIdx);
#if 1 /* Use immediate converse message */
  CmiBecomeImmediate(msg);
#endif
  CmiSyncNodeSendAndFree(destNode,sizeof(immediateMsg),(char *)msg);
}

// Converse immediate handler
void immediatering_startHandler(void *vmsg)
{
  const int maxRings = 1000;
  immediateMsg *msg=(immediateMsg *)vmsg;

  if(0!=strcmp(msg->data,"Array!")) {
    CkAbort("Message corrupted");
  }
  if(CkMyNode()==0)
    msg->iter++;
  if (msg->iter < maxRings) { /* Keep passing message around the ring */
    /* This is questionable: */
    sendImmediate((CkMyNode()+1) % CkNumNodes(),msg->iter);
  } else /* msg->iter>=maxRings, so stop: */ { 
    /* megatest_finish isn't safe from an immediate, so send a 
       regular non-immediate message out: */
    int size=CmiMsgHeaderSizeBytes;
    void *msg=CmiAlloc(size);
    CmiSetHandler(msg,immediatering_finishHandlerIdx);
    CmiSyncSendAndFree(0,size,(char *)msg);
  }
  CmiFree(msg);
}

// on node 0
static int waitFor = 0;

extern "C" void immediatering_finishHandler(void *msg) {
  CmiFree(msg);
  waitFor ++;
  // only send one megatest_finish when all tests finish
  if (waitFor%numTests == 0) {
    megatest_finish(); // Not safe from inside immediate
  }
}

// testing Charm immediate handler

void immRing_nodegroup::start(immMessage *msg)
{
  const int maxRings = 50;

//CkPrintf("[%d] start %d\n", thisIndex, msg->iter);

  if(!msg->check()) {
    CkError("Message corrupted!\n");
    megatest_finish();
    return;
  }
  if(CkMyNode()==0)
    msg->iter++;
  if (msg->iter < maxRings) {
    thisProxy[(CkMyNode()+1) % CkNumNodes()].start(msg);
  } else {
    delete msg;
    //megatest_finish();
    int size=CmiMsgHeaderSizeBytes;
    void *msg=CmiAlloc(size);
    CmiSetHandler(msg,immediatering_finishHandlerIdx);
    CmiSyncSendAndFree(0,size,(char *)msg);
  }
}

void immRing_group::start(immMessage *msg)
{
  const int maxRings = 50;

//CkPrintf("[%d] start %d\n", thisIndex, msg->iter);

  if(!msg->check()) {
    CkError("Message corrupted!\n");
    megatest_finish();
    return;
  }
  if(CkMyPe()==0)
    msg->iter++;
  if (msg->iter < maxRings) {
    thisProxy[(CkMyPe()+1) % CkNumPes()].start(msg);
  } else {
    delete msg;
    //megatest_finish();
    int size=CmiMsgHeaderSizeBytes;
    void *msg=CmiAlloc(size);
    CmiSetHandler(msg,immediatering_finishHandlerIdx);
    CmiSyncSendAndFree(0,size,(char *)msg);
  }
}

void immediatering_init(void)
{ 
  int setNum = 0;
  if (CkMyRank()==0 && numTests==0) setNum = 1;
#if 1
  // test Charm immediate messages
  if (setNum) numTests +=2;
  immring_nodegrp[0].start(new immMessage);
  immring_grp[0].start(new immMessage);
#endif
#if 1
  if (setNum) numTests ++;
  sendImmediate(0,0);
#endif
}

void immediatering_moduleinit(void)
{
  immring_nodegrp = CProxy_immRing_nodegroup::ckNew();
  immring_grp = CProxy_immRing_group::ckNew();
}

void immediatering_initcall(void) {
  // Register converse handlers
  immediatering_startHandlerIdx=CmiRegisterHandler(immediatering_startHandler);
  immediatering_finishHandlerIdx=CmiRegisterHandler(immediatering_finishHandler);
}

MEGATEST_REGISTER_TEST(immediatering,"gengbin",1)
#include "immediatering.def.h"
