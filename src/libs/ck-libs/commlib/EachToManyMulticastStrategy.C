#include "EachToManyMulticastStrategy.h"

CpvExtern(int, RecvdummyHandle);
EachToManyMulticastStrategy *nm_mgr;

void *E2MHandler(void *msg){
  ComlibPrintf("[%d]:In Node MulticastCallbackHandler\n", CkMyPe());
  register envelope *env = (envelope *)msg;
  CkUnpackMessage(&env);
  //nm_mgr->getCallback().send(EnvToUsr(env));
  
  nm_mgr->getHandler()(env);
  return NULL;
}

EachToManyMulticastStrategy::EachToManyMulticastStrategy
(int substrategy,ComlibMulticastHandler h){

    ComlibPrintf("In constructor, %d\n", substrategy);
    routerID = substrategy;
    messageBuf = 0;

    ComlibPrintf("Before instance\n");
    comid = ComlibInstance(routerID, CkNumPes());
    this->npes = CkNumPes();
    ComlibPrintf("After instance\n");

    npes = CkNumPes();
    this->pelist = new int[npes];
    for(int count =0; count < npes; count ++)
      this->pelist[count] = count;
    
    handler = (long) h;

    //procMap = new int[CkNumPes()];
    //for(int count = 0; count < CkNumPes(); count ++){
    //  procMap[count] = count;
    //}
    ComlibPrintf("After Constructor\n");
}

void EachToManyMulticastStrategy::checkPeList(){
    int flag = 0, count , pos;
    for(count = 0; count < npes; count++){
	for(pos = 0; pos < npes; pos ++)
	    if(pelist[count] == pelist[pos] && count!=pos){
		flag = 1;
		break;
	    }
	if( flag )
	    break;
    }
    
    int *newpelist = new int[npes], newpos = 0;
    
    for(count = 0; count < npes; count++)
	newpelist[count] = -1;
    
    if( flag ) {
	for(count = 0; count < npes; count++){
	    int flag1 = 0;
	    for(pos = 0; pos < newpos; pos ++)
		if(newpelist[pos] == pelist[count])
		    flag1 = 1;
	    
	    if(!flag1)
		newpelist[newpos++] = pelist[count];
	    
	    flag1 = 0;
	}
    }
    
    npes = newpos;
    pelist = newpelist;
}

EachToManyMulticastStrategy::EachToManyMulticastStrategy
(int substrategy, int npes,int *pelist, ComlibMulticastHandler h){
  
    this->npes = npes;
    //checkPeList();

    routerID = substrategy;
    messageBuf = NULL;

    comid = ComlibInstance(routerID, CkNumPes());
    if(npes < CkNumPes())
	comid = ComlibEstablishGroup(comid, this->npes, pelist);

    this->pelist = new int[npes];
    for(int count =0; count < npes; count ++)
	this->pelist[count] = count;
}

void EachToManyMulticastStrategy::insertMessage(CharmMessageHolder *cmsg){
    
    if(messageBuf == NULL) {
	ComlibPrintf("ERROR MESSAGE BUF IS NULL\n");
	return;
    }
    ComlibPrintf("EachToMany: insertMessage\n");

    messageBuf->enq(cmsg);
}

void EachToManyMulticastStrategy::doneInserting(){
    ComlibPrintf("%d: DoneInserting \n", CkMyPe());
    
    if((messageBuf->length() == 0) && (CkNumPes() > 0)) {
        DummyMsg * dummymsg = new DummyMsg;
        
        ComlibPrintf("Creating a dummy message\n");
        
        CmiSetHandler(UsrToEnv(dummymsg), 
                      CpvAccess(RecvdummyHandle));
        
        messageBuf->enq(new CharmMessageHolder((char *)dummymsg, CkMyPe()));
    }

    NumDeposits(comid, messageBuf->length());
    
    while(!messageBuf->isEmpty()) {
	CharmMessageHolder *cmsg = messageBuf->deq();
        char *msg = cmsg->getCharmMessage();
	
	CmiSetHandler(UsrToEnv(msg), handlerId);

        ComlibPrintf("Calling EachToMany %d %d %d\n", 
                     UsrToEnv(msg)->getTotalsize(), CkMyPe(), 
                     cmsg->dest_proc);
        EachToManyMulticast(comid, UsrToEnv(msg)->getTotalsize(), 
                            UsrToEnv(msg), npes, pelist);
	delete cmsg; 
    }
}

void EachToManyMulticastStrategy::pup(PUP::er &p){

    ComlibPrintf("Each To many :: pup\n");

    Strategy::pup(p);
    
    p | routerID;
    p | comid;
    p | npes;
    p | handler;

    if(p.isUnpacking()) 
      pelist = new int[npes];
    p(pelist, npes);
    
    if(p.isUnpacking()){
	messageBuf = new CkQ<CharmMessageHolder *>;
	handlerId = CmiRegisterHandler((CmiHandler)E2MHandler);
	nm_mgr = this;
    }
}


