#include "EachToManyStrategy.h"

CpvExtern(int, RecvmsgHandle);
CpvExtern(int, RecvdummyHandle);

void setReverseMap(int *procMap, int *pelist, int npes){
    
    for(int pcount = 0; pcount < CkNumPes(); pcount++)
        procMap[pcount] = -1;
    
    for(int pcount = 0; pcount < npes; pcount++) 
        procMap[pelist[pcount]] = pcount;
}

EachToManyStrategy::EachToManyStrategy(int substrategy){
    ComlibPrintf("In constructor, %d\n", substrategy);
    routerID = substrategy;
    messageBuf = 0;
    messageCount = 0;

    ComlibPrintf("Before instance\n");
    comid = ComlibInstance(routerID, CkNumPes());
    this->npes = CkNumPes();
    ComlibPrintf("After instance\n");

    procMap = new int[CkNumPes()];
    for(int count = 0; count < CkNumPes(); count ++){
        procMap[count] = count;
    }
    ComlibPrintf("After Constructor\n");
}

void EachToManyStrategy::checkPeList(){
  int flag = 0, count , pos;
  for(count = 0; count < npes; count++){
    for(pos = 0; pos < npes; pos ++)
      if(pelist[count] == pelist[pos]){
	flag = 1;
	break;
      }
    if( flag )
      break;
  }

  int *newpelist = new int[npes], newpos = 0;
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

EachToManyStrategy::EachToManyStrategy(int substrategy, int npes, int *pelist){

    this->npes = npes;
    this->pelist = pelist;
    checkPeList();

    routerID = substrategy;
    messageBuf = NULL;
    messageCount = 0;

    procMap = new int[CkNumPes()];
    setReverseMap(procMap, this->pelist, this->npes);
    
    comid = ComlibInstance(routerID, CkNumPes());
    if(npes < CkNumPes())
      comid = ComlibEstablishGroup(comid, this->npes, this->pelist);
}

void EachToManyStrategy::insertMessage(CharmMessageHolder *cmsg){

    if(messageBuf == NULL) {
	ComlibPrintf("ERROR MESSAGE BUF IS NULL\n");
	return;
    }
    ComlibPrintf("EachToMany: insertMessage\n");
    
    messageBuf->enq(cmsg);
    messageCount ++;
}

void EachToManyStrategy::doneInserting(){
    ComlibPrintf("%d: DoneInserting \n", CkMyPe());
    //ComlibPrintf("%d:Setting Num Deposit to %d\n", CkMyPe(), messageCount);

    if((messageBuf->length() == 0) && (CkNumPes() > 0)) {
        DummyMsg * dummymsg = new DummyMsg;
        
        ComlibPrintf("Creating a dummy message\n");
        
        CmiSetHandler(UsrToEnv(dummymsg), 
                      CpvAccess(RecvdummyHandle));
        
        messageBuf->enq(new CharmMessageHolder((char *)dummymsg, CkMyPe()));
        messageCount ++;
    }

    NumDeposits(comid, messageBuf->length());
    
    while(!messageBuf->isEmpty()) {
	CharmMessageHolder *cmsg = messageBuf->deq();
        char * msg = cmsg->getCharmMessage();
        ComlibPrintf("Calling EachToMany %d %d %d procMap=%d\n", 
                     UsrToEnv(msg)->getTotalsize(), CkMyPe(), 
                     cmsg->dest_proc, procMap[cmsg->dest_proc]);
        EachToManyMulticast(comid, UsrToEnv(msg)->getTotalsize(), 
                            UsrToEnv(msg), 1, 
                            &procMap[cmsg->dest_proc]);
	delete cmsg; 
    }
    messageCount = 0;
}

void EachToManyStrategy::pup(PUP::er &p){

  ComlibPrintf("Each To many :: pup\n");

    Strategy::pup(p);
    
    p | messageCount;
    p | routerID;
    p | comid;
    p | npes;
    
    if(p.isUnpacking()) 
      procMap = new int[CkNumPes()];
        
    p(procMap, CkNumPes());

    if(p.isUnpacking()){
      messageBuf = new CkQ<CharmMessageHolder *>;
    }
}

PUPable_def(EachToManyStrategy); 
