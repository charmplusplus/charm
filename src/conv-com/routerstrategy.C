/**
   @addtogroup ConvComlibRouter
   @{
   @file
*/


#include "routerstrategy.h"

#include "gridrouter.h"
#include "graphrouter.h"
#include "hypercuberouter.h"
#include "treerouter.h"
#include "3dgridrouter.h"
#include "prefixrouter.h"
#include "converse.h"
#include "charm++.h"

//Handlers that call the entry funtions of routers 
//Refer to router.h for details on these entry functions

CkpvDeclare(int, RouterProcHandle);
///Correspods to Router::ProcManyMsg
void routerProcManyCombinedMsg(char *msg) {
    //comID id;
    int instance_id;

    ComlibPrintf("In Proc combined message at %d\n", CkMyPe());
    //memcpy(&id,(msg+CmiReservedHeaderSize+sizeof(int)), sizeof(comID));

    //Comid specific -- to change to a better reading!
    memcpy(&instance_id, (char*) msg + CmiReservedHeaderSize + 2*sizeof(int)
           , sizeof(int));

    RouterStrategy *s = (RouterStrategy*)ConvComlibGetStrategy(instance_id);
    s->getRouter()->ProcManyMsg(s->getComID(), msg);
}

CkpvDeclare(int, RouterDummyHandle);
///Correspods to Router::DummyEP
void routerDummyMsg(DummyMsg *m) {
    RouterStrategy *s = (RouterStrategy*)ConvComlibGetStrategy(m->id.instanceID);
    s->getRouter()->DummyEP(m->id, m->magic);
}

CkpvDeclare(int, RouterRecvHandle);
///Correspods to Router::RecvManyMsg
void routerRecvManyCombinedMsg(char *msg) {
    //comID id;
    int instance_id;
    ComlibPrintf("In Recv combined message at %d\n", CkMyPe());
    //memcpy(&id,(msg+CmiReservedHeaderSize+sizeof(int)), sizeof(comID));
    
    //Comid specific -- to change to a better reading!
    memcpy(&instance_id, (char*) msg + CmiReservedHeaderSize + 2*sizeof(int)
           , sizeof(int));

    RouterStrategy *s = (RouterStrategy*)ConvComlibGetStrategy(instance_id);
    s->getRouter()->RecvManyMsg(s->getComID(), msg);
}

/* DEPRECATED: method notifyDone substitutes it
void doneHandler(DummyMsg *m){
    Strategy *s = ConvComlibGetStrategy(m->id.instanceID);
    
    ((RouterStrategy *)s)->Done(m);
}
*/

void RouterStrategy::setReverseMap(){
    int pcount;
    for(pcount = 0; pcount < CkNumPes(); pcount++)
        procMap[pcount] = -1;

    //All processors not in the domain will point to -1
    for(pcount = 0; pcount < npes; pcount++) {
        if (pelist[pcount] == CkMyPe())
            myPe = pcount;

        procMap[pelist[pcount]] = pcount;
    }
}

RouterStrategy::RouterStrategy(int stratid) {
  ComlibPrintf("[%d] RouterStrategy protected constructor\n",CkMyPe());
  setType(CONVERSE_STRATEGY);
  id.instanceID = 0;
  id.isAllToAll = 0;
  id.refno = 0;
  doneHandle = 0;
  routerIDsaved = stratid;
  router = NULL;
  pelist = NULL;
  bcast_pemap = NULL;
  procMap = new int[CkNumPes()];
  doneFlag = 1;
  bufferedDoneInserting = 0;
}

RouterStrategy::RouterStrategy(int stratid, int handle, int _nsrc, int *_srclist,
			       int _ndest, int *_destlist) : Strategy() {

  ComlibPrintf("[%d] RouterStrategy constructor\n",CkMyPe());

    setType(CONVERSE_STRATEGY);

    //CkpvInitialize(int, RecvHandle);
    //CkpvInitialize(int, ProcHandle);
    //CkpvInitialize(int, DummyHandle);

    id.instanceID = 0; //Set later in doneInserting
    
    id.isAllToAll = 0;
    id.refno = 0;

    /*
    CkpvAccess(RecvHandle) =
        CkRegisterHandler((CmiHandler)recvManyCombinedMsg);
    CkpvAccess(ProcHandle) =
        CkRegisterHandler((CmiHandler)procManyCombinedMsg);
    CkpvAccess(DummyHandle) = 
        CkRegisterHandler((CmiHandler)dummyEP);    
    */

    //myDoneHandle = CkRegisterHandler((CmiHandler)doneHandler);    

    // Iteration done handle
    doneHandle = handle;

    routerIDsaved = stratid;

    router = NULL;
    pelist = NULL;
    bcast_pemap = NULL;
    procMap = new int[CkNumPes()];    

    //Start with all iterations done
    doneFlag = 1;
    
    //No Buffered doneInserting at the begining
    bufferedDoneInserting = 0;

    newKnowledgeSrc = _srclist;
    newKnowledgeSrcSize = _nsrc;
    if (_ndest == 0) {
      newKnowledgeDest = new int[_nsrc];
      newKnowledge = new int[_nsrc];
      newKnowledgeDestSize = _nsrc;
      newKnowledgeSize = _nsrc;
      memcpy(newKnowledgeDest, _srclist, _nsrc);
      memcpy(newKnowledge, _srclist, _nsrc);
    } else {
      newKnowledgeDest = _destlist;
      newKnowledgeDestSize = _ndest;
      int *tmplist = new int[CkNumPes()];
      for (int i=0; i<CkNumPes(); ++i) tmplist[i]=0;
      for (int i=0; i<_nsrc; ++i) tmplist[newKnowledgeSrc[i]]++;
      for (int i=0; i<_ndest; ++i) tmplist[newKnowledgeDest[i]]++;
      newKnowledgeSize = 0;
      for (int i=0; i<CkNumPes(); ++i) if (tmplist[i]!=0) newKnowledgeSize++;
      newKnowledge = new int[newKnowledgeSize];
      for (int i=0, count=0; i<CkNumPes(); ++i) if (tmplist[i]!=0) newKnowledge[count++]=i;
      delete [] tmplist;
    }

    setupRouter();
    /*
    npes = _npes;
    //pelist = new int[npes];
    pelist = _pelist;
    //memcpy(pelist, _pelist, sizeof(int) * npes);    

    if(npes <= 1)
        routerID = USE_DIRECT;

    myPe = -1;
    setReverseMap();

    ComlibPrintf("Router Strategy : %d, MYPE = %d, NUMPES = %d \n", stratid, 
                 myPe, npes);

    if(myPe < 0) {
        //I am not part of this strategy
        router = NULL;
	routerID = USE_DIRECT;
        return;        
    }

    switch(stratid) {
    case USE_TREE: 
        router = new TreeRouter(npes, myPe);
        break;
        
    case USE_MESH:
        router = new GridRouter(npes, myPe);
        break;
        
    case USE_HYPERCUBE:
        router = new HypercubeRouter(npes, myPe);
        break;
        
    case USE_GRID:
        router = new D3GridRouter(npes, myPe);
        break;
	
    case USE_PREFIX:
       	router = new PrefixRouter(npes, myPe);
	break;

    case USE_DIRECT: router = NULL;
        break;
        
    default: CmiAbort("Unknown Strategy\n");
        break;
    }

    if(router) {
        router->SetMap(pelist);
        //router->setDoneHandle(myDoneHandle);
        //router->SetID(id);
    }
    */
}

void RouterStrategy::setupRouter() {
    if (bcast_pemap != NULL && ndestPes != newKnowledgeDestSize) {
      delete[] bcast_pemap;
      bcast_pemap = NULL;
    }

    npes = newKnowledgeSize;
    nsrcPes = newKnowledgeSrcSize;
    ndestPes = newKnowledgeDestSize;
    if (pelist != NULL) {
      delete[] pelist;
      delete[] srcPelist;
      delete[] destPelist;
    }
    pelist = newKnowledge;
    srcPelist = newKnowledgeSrc;
    destPelist = newKnowledgeDest;

    newKnowledge = NULL;

    if (npes <= 1) routerID = USE_DIRECT;
    else routerID = routerIDsaved;

    myPe = -1;
    setReverseMap();

    ComlibPrintf("[%d] Router Strategy : %d, MYPE = %d, NUMPES = %d \n", CkMyPe(),  routerID, myPe, npes);

    ComlibPrintf("[%d] router=%p\n", CkMyPe(),  router);

   	delete router;
    router = NULL;

    if (myPe < 0) {
        //I am not part of this strategy
        router = NULL;
        routerID = USE_DIRECT;
        return;        
    }

    switch(routerID) {
    case USE_TREE: 
        router = new TreeRouter(npes, myPe, this);
        break;
        
    case USE_MESH:
        router = new GridRouter(npes, myPe, this);
        break;
        
    case USE_HYPERCUBE:
        router = new HypercubeRouter(npes, myPe, this);
        break;
        
    case USE_GRID:
        router = new D3GridRouter(npes, myPe, this);
        break;
	
    case USE_PREFIX:
       	router = new PrefixRouter(npes, myPe, this);
	break;

    case USE_DIRECT: router = NULL;
        break;
        
    default: CmiAbort("Unknown Strategy\n");
        break;
    }

    if(router) {
        router->SetMap(pelist);
        //router->setDoneHandle(myDoneHandle);
        //router->SetID(id);
    }
}

RouterStrategy::~RouterStrategy() {
  ComlibPrintf("[%d] RouterStrategy destructor\n",CkMyPe());

    delete [] pelist;
    delete [] srcPelist;
    delete [] destPelist;

    delete [] bcast_pemap;
    
    delete [] procMap;

    delete router;
    router = NULL;
}

/// Receive a message from the upper layer and buffer it in the msgQ until
/// doneInserting is called. If the strategy is USE_DIRECT then just send it to the handleMessage method for the Strategy.
void RouterStrategy::insertMessage(MessageHolder *cmsg){
	ComlibPrintf("[%d] RouterStrategy::insertMessage\n", CkMyPe());

       
  //if(myPe < 0)
  //    CmiAbort("insertMessage: mype < 0\n");

    int count = 0;
    if(routerID == USE_DIRECT) {
#if 0
    	// THE OLD VERSION. THIS IS BAD, as it can cause messages to be lost before errors are detected.
    	if(cmsg->dest_proc == IS_BROADCAST) {
	  // ndestPes = npes;
           	for(count = 0; count < ndestPes-1; count ++){
        		CmiSyncSend(destPelist[count], cmsg->size, cmsg->getMessage());
        		int destPe = destPelist[count];
        		ComlibPrintf("[%d] RouterStrategy::insertMessage Broadcasting to PE %d\n", CkMyPe(), destPe );
        	}
        	if(ndestPes > 0){
        		CmiSyncSendAndFree(destPelist[ndestPes-1], cmsg->size, cmsg->getMessage());
        		int destPe = destPelist[ndestPes-1];
        		ComlibPrintf("[%d] RouterStrategy::insertMessage Broadcasting to PE %d\n", CkMyPe(), destPe);
        	}
        }
        else
            CmiSyncSendAndFree(cmsg->dest_proc, cmsg->size, 
                               cmsg->getMessage());
    	delete cmsg;
#else
    	if(cmsg->dest_proc == IS_BROADCAST) {
    		ComlibPrintf("[%d] RouterStrategy::insertMessage Broadcasting to all PEs\n", CkMyPe());
	
#if 0
		CmiSyncBroadcastAndFree(cmsg->size, cmsg->getMessage() ); // This ought to be the same as the following alternative
#else
            	for(int destPe = 0; destPe < CkNumPes()-1; destPe++){
            		ComlibPrintf("[%d] RouterStrategy::insertMessage Broadcasting to all, PE %d\n", CkMyPe(), destPe );
            		CmiSyncSend(destPe, cmsg->size, cmsg->getMessage());
         	}
            	if(CkNumPes()>0){
            		CmiSyncSendAndFree(CkNumPes()-1, cmsg->size, cmsg->getMessage());
         		ComlibPrintf("[%d] RouterStrategy::insertMessage Broadcasting to all, PE %d\n", CkMyPe(), CkNumPes()-1 );
            	}
#endif

    	}	
        else {
	  CmiSyncSendAndFree(cmsg->dest_proc, cmsg->size, cmsg->getMessage());
	}
    	delete cmsg;
    
#endif
    }
    else {
        if(cmsg->dest_proc >= 0) {
            cmsg->pelist = &procMap[cmsg->dest_proc];
            cmsg->npes = 1;
        }
        else if (cmsg->dest_proc == IS_BROADCAST){

	  // if we are calling a broadcast then we set AllToAll flag
	  id.isAllToAll = 1;

            if(bcast_pemap == NULL) {
                bcast_pemap = new int[ndestPes];
                for(count = 0; count < ndestPes; count ++) {
                    bcast_pemap[count] = count;
                }
            }

            cmsg->pelist = bcast_pemap;
            cmsg->npes = npes;
        }
        
        msgQ.push(cmsg);
    }
}

void RouterStrategy::doneInserting(){
  ComlibPrintf("[%d] RouterStrategy::doneInserting msgQ.length()=%d \n", CkMyPe(), msgQ.length());
  
  
  if(myPe < 0) return; // nothing to do if I have not objects in my processor
      //CmiAbort("insertMessage: mype < 0\n");

    id.instanceID = getInstance();

    //ComlibPrintf("Instance ID = %d\n", getInstance());
    
    if(doneFlag == 0) {
        ComlibPrintf("[%d] Waiting for previous iteration to Finish\n", 
                     CkMyPe());
        bufferedDoneInserting = 1;
        ComlibPrintf("[%d] RouterStrategy::doneInserting returning\n", CkMyPe());
        return;
    }
    
    if(routerID == USE_DIRECT) {
      CkAssert(msgQ.length() == 0);
      //DummyMsg *m = (DummyMsg *)CmiAlloc(sizeof(DummyMsg));
      //memset((char *)m, 0, sizeof(DummyMsg)); 
      //m->id.instanceID = getInstance();
      ComlibPrintf("[%d] RouterStrategy::doneInserting calling notifyDone()\n", CkMyPe());
      notifyDone();
      return;
    }

    doneFlag = 0;
    bufferedDoneInserting = 0;

    id.refno ++;

    if(msgQ.length() == 0) {
        DummyMsg * dummymsg = (DummyMsg *)CmiAlloc(sizeof(DummyMsg));
        ComlibPrintf("[%d] Creating a dummy message\n", CkMyPe());
        CmiSetHandler(dummymsg, CkpvAccess(RecvdummyHandle));
        
        MessageHolder *cmsg = new MessageHolder((char *)dummymsg, 
                                                     myPe, 
                                                     sizeof(DummyMsg));
        cmsg->isDummy = 1;
        cmsg->pelist = &myPe;
        cmsg->npes = 1;
        msgQ.push(cmsg);
    }

    ComlibPrintf("Calling router->EachToManyMulticastQ??????????????????????????\n");
    router->EachToManyMulticastQ(id, msgQ);

}

void RouterStrategy::deliver(char *msg, int size) {
  CmiSyncSendAndFree(CkMyPe(), size, msg);
}

/// Update the router accordingly to the new information. If the router is
/// currently active (doneFlag==0), then wait for it to finish and store the
/// knowledge in "newKnowledge"
void RouterStrategy::bracketedUpdatePeKnowledge(int *count) {
  ComlibPrintf("[%d] RouterStrategy: Updating knowledge\n", CkMyPe());

  newKnowledgeSize = 0;
  newKnowledgeSrcSize = 0;
  newKnowledgeDestSize = 0;
  for (int i=0; i<CkNumPes(); ++i) {
    if (count[i] != 0) newKnowledgeSize++;
    if ((count[i]&2) == 2) newKnowledgeDestSize++;
    if ((count[i]&1) == 1) newKnowledgeSrcSize++;
  }
  newKnowledge = new int[newKnowledgeSize];
  newKnowledgeSrc = new int[newKnowledgeSrcSize];
  newKnowledgeDest = new int[newKnowledgeDestSize];
  
  for(int i=0;i<newKnowledgeDestSize;i++){
	  newKnowledgeDest[i] = -1;
  }

  for(int i=0;i<newKnowledgeSrcSize;i++){
	  newKnowledgeSrc[i] = -1;
  }

  for(int i=0;i<newKnowledgeSize;i++){
	  newKnowledge[i] = -1;
  }

  int c=0, cS=0, cD=0;
  for (int i=0; i<CkNumPes(); ++i) {
    if (count[i] != 0) newKnowledge[c++]=i;
    if ((count[i]&2) == 2) newKnowledgeDest[cD++]=i;
    if ((count[i]&1) == 1) newKnowledgeSrc[cS++]=i;
  }
  
  ComlibPrintf("[%d] RouterStrategy::bracketedUpdatePeKnowledge c=%d cS=%d cD=%d\n", CkMyPe(), c, cS, cD);

  for (int i=0; i<newKnowledgeDestSize; ++i) {
    ComlibPrintf("[%d] RouterStrategy::bracketedUpdatePeKnowledge newKnowledgeDest[%d]=%d\n", CkMyPe(), i, newKnowledgeDest[i]);
  }

  
  
  if (doneFlag == 0) return;

  // here we can update the knowledge
  setupRouter();
}

void RouterStrategy::notifyDone(){

    ComlibPrintf("[%d] RouterStrategy: Finished iteration\n", CkMyPe());

    if(doneHandle > 0) {
      DummyMsg *m = (DummyMsg *)CmiAlloc(sizeof(DummyMsg));
      memset((char *)m, 0, sizeof(DummyMsg)); 
      m->id.instanceID = getInstance();
      CmiSetHandler(m, doneHandle);
      CmiSyncSendAndFree(CkMyPe(), sizeof(DummyMsg), (char*)m);
    }

    doneFlag = 1;
    // at this point, if we have some knowledge update, we apply it
    if (newKnowledge != NULL) {
      //bracketedUpdatePeKnowledge(newKnowledge);
      setupRouter();
    }

    if(bufferedDoneInserting) doneInserting();
}

void RouterStrategy::pup(PUP::er &p) {
  ComlibPrintf("[%d] RouterStrategy::pup called for %s\n",CkMyPe(),
	       p.isPacking()?"packing":(p.isUnpacking()?"unpacking":"sizing"));
  Strategy::pup(p);

  p | id;
  if (p.isUnpacking()) {
    pelist = NULL;
    bcast_pemap = NULL;
    procMap = new int[CkNumPes()];
    doneFlag = 1;
    bufferedDoneInserting = 0;
  }

  p | npes;
  p | nsrcPes;
  p | ndestPes;
  newKnowledgeSize = npes;
  newKnowledgeSrcSize = nsrcPes;
  newKnowledgeDestSize = ndestPes;

  if (p.isUnpacking()) {
    newKnowledge = new int[npes];
    newKnowledgeSrc = new int[nsrcPes];
    newKnowledgeDest = new int[ndestPes];
  } else {
    newKnowledge = pelist;
    newKnowledgeSrc = srcPelist;
    newKnowledgeDest = destPelist;
  }
  p(newKnowledge, npes);
  p(newKnowledgeSrc, nsrcPes);
  p(newKnowledgeDest, ndestPes);

  p | routerIDsaved;
  p | doneHandle;

  if (p.isUnpacking()) {
	  // Because we are unpacking, the router strategy should be initialized to NULL, so that setupRouter will correctly instantiate it
	  router = NULL;
	  setupRouter();
  }
  else newKnowledge = NULL;
}

PUPable_def(RouterStrategy)

/*@}*/
