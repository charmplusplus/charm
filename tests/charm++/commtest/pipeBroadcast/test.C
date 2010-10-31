/*
  Test to check whether PipeBroadcastStrategy works in charm++

  Written by Filippo Gioachin, Jul 2004
*/

#include "test.h"
#include <PipeBroadcastStrategy.h>

#define LENGTH 10000

CProxy_TheMain mainProxy;
CProxy_Test arr;
ComlibInstanceHandle cinst;
int numEl;

TheMain::TheMain(CkArgMsg *msg) {
  if (msg->argc>1) numEl = atoi(msg->argv[1]);
  else numEl = 100;
  delete msg;

  CkPrintf("Main: Started\n");
  called = 0;
  arr = CProxy_Test::ckNew(numEl);

//  cinst = CkGetComlibInstance();
  mainProxy = thishandle;

  PipeBroadcastStrategy *strategy = new PipeBroadcastStrategy(USE_HYPERCUBE, arr);
  cinst = ComlibRegister(strategy);
  CkPrintf("Main: calling send on %d elements\n",numEl);
  arr.send();
  //arr[2].send();
  //arr[0].send();
}

void TheMain::exit() {
  //CkPrintf("called = %d\n",called);
  if (++called >= numEl*numEl) {
    CkPrintf("All done\n");
    CkExit();
  }
}

Test::Test() { }

Test::Test(CkMigrateMessage *msg) {}

void Test::send() {
  MyMess *mess = new (LENGTH,0) MyMess;
  mess->data[0] = CkMyPe();
  for (int i=1; i<LENGTH; ++i) mess->data[i] = i+1000;
 
  CProxy_Test copy = thisProxy;
  // CProxy_Test copy = thisArrayID;
  ComlibAssociateProxy(cinst, copy);
  
 // ComlibDelegateProxy(&copy);
  CkPrintf("[%d-%d] sending broadcast\n",CkMyPe(),thisIndex);
  copy.receive(mess);
}

void Test::receive(MyMess *msg) {
  int correct = true;
  for (int i=1; i<LENGTH; ++i) if (msg->data[i] != i+1000) {
    CkPrintf("[%d] wrong: i=%d, data=%d\n",CkMyPe(),i,msg->data[i]);
    correct=false;
  }
  if (correct);// CkPrintf("[%d] received message from %d, all correct\n",CkMyPe(),(int)msg->data[0]);
  else CkPrintf("[%d] received message from %d, WRONG!\n",CkMyPe(),(int)msg->data[0]);
  mainProxy.exit();
  //CkPrintf("[%d] FINISHED\n", CkMyPe());
  //printf("sizeof(MessageBase)=%d\n",sizeof(class CkMcastBaseMsg));
  //printf("sizeof(envelope)=%d, CmiMsgHeader=%d, sizeof(s_attrib)=%d\n",sizeof(envelope),CmiMsgHeaderSizeBytes,sizeof(envelope::s_attribs));
}


#include "test.def.h"
