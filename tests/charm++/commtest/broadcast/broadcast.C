#include "broadcast.decl.h"

#include <comlib.h>
#include <cassert>
#include <unistd.h>
/*
 * Test of Broadcast Strategy
 */

CProxy_Main mainProxy;
CProxy_Broadcast broadcastProxy;
int numEl;

ComlibInstanceHandle stratBroadcast;

class broadcastMessage : public CMessage_broadcastMessage {
public:
  int length;
  char* msg;
};

// mainchare

class Main : public CBase_Main
{
private:
  int nDone;

public:

  Main(CkArgMsg *m) {    
    nDone = 0;
	if (m->argc>1) numEl = atoi(m->argv[1]);
  	else numEl = 100;


    delete m;

    mainProxy = thishandle;
	
    broadcastProxy = CProxy_Broadcast::ckNew();
    
   // create broadcast strategy using the hypercube topology
    BroadcastStrategy *strategy = new BroadcastStrategy(USE_HYPERCUBE);
  // PipeBroadcastStrategy *strategy = new PipeBroadcastStrategy(broadcastProxy,USE_TREE);
    stratBroadcast = ComlibRegister(strategy);

//	CkPrintf("Main: Sleeping %d \n", CkNumPes());
//	usleep(50*1000*1000);
	broadcastProxy.TestBroadcast();
//	broadcastProxy.IntermediateCall();
//	thisProxy.Intermediate1();
	CkPrintf("Main: Started %d \n", CkNumPes());
  }

  void Intermediate(){
    nDone++;
    if (nDone == CkNumPes()){ nDone = 0;
      
	broadcastProxy.TestBroadcast();
	}
  }
   void Intermediate1(){
    
	CProxy_Broadcast commBroadcastProxy = broadcastProxy;

	sleep(300);
   	ComlibAssociateProxy(stratBroadcast, commBroadcastProxy);
    CkPrintf("Associated proxy to commlib  \n");

	char msg[] = "|This is a short broadcast message|";
    broadcastMessage* b = new(strlen(msg)+1,0) broadcastMessage;
	
    memcpy(b->msg, msg, strlen(msg)+1);
    b->length = strlen(msg);

	commBroadcastProxy.receive(b);
  }
 

  void exit() {
    nDone++;
    if (nDone == CkNumPes())
      CkExit();
  }

};

class Broadcast : public CBase_Broadcast {
private:
  CProxy_Broadcast localProxy;

public:

  Broadcast() {

    CkPrintf("element %d \n", CkMyPe());
    localProxy = thisProxy;
  }

  Broadcast(CkMigrateMessage *m) {}
    
  void TestBroadcast() {

//	usleep(50*1000*1000);
	  if (CkMyPe() == 0) {
   	  ComlibAssociateProxy(stratBroadcast, localProxy);
      char msg[] = "|This is a short broadcast message|";
      broadcastMessage* b = new(strlen(msg)+1,0) broadcastMessage;
	
      memcpy(b->msg, msg, strlen(msg)+1);
      b->length = strlen(msg);
      localProxy.receive(b);
    }
	
  }

  void receive(broadcastMessage* m) {

    CkPrintf("Message: %s arrived at element %d\n", m->msg, CkMyPe());
    assert(strcmp(m->msg,"|This is a short broadcast message|") == 0);
    mainProxy.exit();
  
  }

  void IntermediateCall() {

    CkPrintf("Intermediate %d of %d \n", CkMyPe(), CkNumPes());
    //CkPrintf("Message: %s arrived at element %d\n", m->msg, CkMyPe());
   // assert(strcmp(m->msg,"|This is a short broadcast message|") == 0);
    mainProxy.Intermediate();
  }

};

#include "broadcast.def.h"
