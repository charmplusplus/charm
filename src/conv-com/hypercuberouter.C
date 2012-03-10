/**
   @addtogroup ConvComlibRouter
   @{
  
   @file 
    
    @brief Dimensional Exchange (Hypercube) Router 

    Modified to send last k stages directly for all to all multicast by
    Sameer Kumar 9/07/03.

    Adapted to the new communication library 05/14/04.
 
*/

#include "hypercuberouter.h"

#define gmap(pe) {if (gpes) pe=gpes[pe];}

//#define gmap(pe) (gpes ? gpes[pe] : pe)

/**The only communication op used. Modify this to use
 ** vector send */

#if CMK_COMLIB_USE_VECTORIZE
#define HCUBESENDFN(kid, u1, u2, knpe, kpelist, khndl, knextpe, pehcube)  \
  	{int len;\
	PTvectorlist newmsg;\
 	newmsg=pehcube->ExtractAndVectorize(kid, u1, knpe, kpelist);\
	if (newmsg) {\
	  CmiSetHandler(newmsg->msgs[0], khndl);\
  	  CmiSyncVectorSendAndFree(knextpe, -newmsg->count, newmsg->sizes, newmsg->msgs);\
	}\
	else {\
	  SendDummyMsg(kid, knextpe, u2);\
	}\
}
#else
#define HCUBESENDFN(kid, u1, u2, knpe, kpelist, khndl, knextpe, pehcube)  \
  	{int len;\
	char *newmsg;\
 	newmsg=pehcube->ExtractAndPack(kid, u1, knpe, kpelist, &len);\
	if (newmsg) {\
	  CmiSetHandler(newmsg, khndl);\
  	  CmiSyncSendAndFree(knextpe, len, newmsg);\
	}\
	else {\
	  SendDummyMsg(kid, knextpe, u2);\
	}\
}
#endif

inline int maxdim(int n)
{
  int maxpes=1, dim=0;

  while (maxpes< n) {
  	maxpes *=2;
	dim++;
  }
  if (maxpes==n) return(dim);
  else return(dim-1);
}

inline int neighbor(int pe, int dim)
{
  return(pe ^ (1<<dim));
}

inline int adjust(int dim, int pe)
{
  int mymax=1<<dim;
  if (pe >= mymax) return(neighbor(pe, dim));
  else return(pe);
}

inline int setIC(int dim, int pe, int N)
{
  int mymax= 1<< dim;
  int initCounter=1, myneighb;
  if (mymax < N) {
	myneighb= neighbor(pe, dim);
	if (myneighb < N && myneighb >= mymax) {
              initCounter=0;
	}
  }
  if (pe >= mymax) initCounter = -1;
  return(initCounter);
}

/*********************************************************************
 * Total preallocated memory=(P+Dim+Dim*P)ints + MAXNUMMSGS msgstruct
 **********************************************************************/
HypercubeRouter::HypercubeRouter(int n, int me, Strategy *parent, int ndirect) : Router(parent)
{
  int i;
 
  //last ndirect steps will be sent directly
  numDirectSteps = ndirect;
  //2 raised to the power of ndirect
  two_pow_ndirect = 1;
  for(int count = 0; count < ndirect; count ++)
      two_pow_ndirect *= 2;

  //Initialize the no: of pes and my Pe number
  NumPes=n;
  MyPe=me;
  gpes=NULL;

  //Initialize Dimension and no: of stages
  Dim=maxdim(NumPes);

  PeHcube=new PeTable(NumPes);
  PeHcube1 = new PeTable(NumPes);

  InitVars();

  //Create the message array, buffer and the next stage table
  buffer=new int[Dim+1];
  next= new int* [Dim];
  for (i=0;i<Dim;i++) {
	next[i]=new int[NumPes];
	buffer[i]=0;
	for (int j=0;j<NumPes;j++) next[i][j]=-1;
  }
  buffer[Dim]=0;

  //Create and initialize the indexes to the above table
  penum=new int[NumPes];
  int *dp=new int[NumPes];
  for (i=0;i<NumPes;i++) {
	penum[i]=0;
	dp[i]=i;
  }

  CreateStageTable(NumPes, dp);
  delete [] dp;

  //CmiPrintf("%d DE constructor done dim=%d, mymax=%d IC=%d\n", MyPe, Dim, 1<<Dim, InitCounter);

  if(numDirectSteps > Dim - 1)
      numDirectSteps = Dim - 1;
}
 
HypercubeRouter :: ~HypercubeRouter()
{
  int i;
  delete PeHcube;
  delete PeHcube1;
  delete buffer;
  for (i=0;i<Dim;i++) {
	delete next[i];
  }
  delete next;
  delete penum;
}

void HypercubeRouter :: SetMap(int *pes)
{
  gpes=pes;
}

void HypercubeRouter :: InitVars()
{
  stage=Dim-1;
  InitCounter=setIC(Dim, MyPe, NumPes);
  procMsgCount = 0;
}

void HypercubeRouter::EachToAllMulticast(comID id, int size, void *msg, int more)
{
  int npe=NumPes;
  int * destpes=(int *)CmiAlloc(sizeof(int)*npe);
  for (int i=0;i<npe;i++) destpes[i]=i;
  EachToManyMulticast(id, size, msg, npe, destpes, more);
  CmiFree(destpes);
}

void HypercubeRouter::NumDeposits(comID, int num)
{
  //CmiPrintf("Deposit=%d\n", num);
}

void HypercubeRouter::EachToManyMulticast(comID id, int size, void *msg, int numpes, int *destpes, int more)
{

    SetID(id);

    //Create the message
    if (msg && size) {
  	PeHcube->InsertMsgs(numpes, destpes, size, msg);
    }
    
    if (more) return;
    start_hcube(id);
}

void HypercubeRouter::EachToManyMulticastQ(comID id, CkQ<MessageHolder *> &msgq) {
    SetID(id);

    int count = 0;
    int length = msgq.length();

    for(count = 0; count < length; count ++) {
        MessageHolder *mhdl = msgq.deq();
        PeHcube->InsertMsgs(mhdl->npes, mhdl->pelist, mhdl->size, 
                            mhdl->getMessage());
        delete mhdl;
    }
    
    start_hcube(id);
}

void HypercubeRouter::start_hcube(comID id) {

    if (InitCounter <0) {
        ComlibPrintf("%d Sending to the lower hypercube\n", MyPe);
  	int nextpe=neighbor(MyPe, Dim);
	int * pelist=(int *)CmiAlloc(NumPes*sizeof(int));
	for (int i=0;i<NumPes;i++) {
            pelist[i]=i;
	}
        
        ComlibPrintf("Before Gmap %d\n", nextpe);
	gmap(nextpe);
        ComlibPrintf("%d: EachToMany Sending to %d\n", MyPe, nextpe);
	HCUBESENDFN(id, Dim, Dim, NumPes, pelist, CkpvAccess(RouterRecvHandle), nextpe, PeHcube);
 	CmiFree(pelist);
	return;
    }
    
    //Done: no more stages.
    if (stage <0) {
	//CmiPrintf("calling lp in multicast call %d\n", stage);
	LocalProcMsg(id);
	return;
    }
    
    InitCounter++;
    RecvManyMsg(id,NULL);
}

//Send the messages for the next stage to the next dimension neighbor
//If only numDirectStage's are left send messages directly using prefix send
void HypercubeRouter::RecvManyMsg(comID id, char *msg)
{
    ComlibPrintf("%d recvmanymsg called\n", MyPe);
    int msgstage;
    if (msg) {
        msgstage=PeHcube->UnpackAndInsert(msg);
        //CmiPrintf("%d recvd msg for stage=%d\n", MyPe, msgstage);
        if (msgstage == Dim) InitCounter++;
        else buffer[msgstage]=1;
    }
  
    //Check the buffers 
    while ((InitCounter==2) || (stage >=numDirectSteps && buffer[stage+1])) {
	InitCounter=setIC(Dim, MyPe, NumPes);
  	if (InitCounter != 2) { 
            buffer[stage+1]=0;
 	}

  	//Send the data to the neighbor in this stage
  	int nextpe=neighbor(MyPe, stage);
        
        ComlibPrintf("Before Gmap %d\n", nextpe);
        gmap(nextpe);
        ComlibPrintf("%d RecvManyMsg Sending to %d\n", MyPe, nextpe);
	HCUBESENDFN(id, stage, stage, penum[stage], next[stage], CkpvAccess(RouterRecvHandle), nextpe, PeHcube);

  	//Go to the next stage
  	stage--; 
    }        

    if (stage < numDirectSteps && buffer[numDirectSteps]) {
                
        InitCounter=setIC(Dim, MyPe, NumPes);
        
        //I am a processor in the smaller hypercube and there are some
        //processors to send directly
        if(InitCounter >= 0 && numDirectSteps > 0) {
            //Sending through prefix send to save on copying overhead   
            //of the hypercube algorithm            
            
#if CMK_COMLIB_USE_VECTORIZE
            PTvectorlist newmsg;
            newmsg=PeHcube->ExtractAndVectorizeAll(id, stage);
            if (newmsg) {
                CmiSetHandler(newmsg->msgs[0], CkpvAccess(RouterProcHandle));
		for (int count=0; count<two_pow_ndirect; ++count) {
		  int nextpe = count ^ MyPe;
		  gmap(nextpe);
		  ComlibPrintf("%d Sending to %d\n", MyPe, nextpe);
		  CmiSyncVectorSend(nextpe, -newmsg->count, newmsg->sizes, newmsg->msgs);
		}
		for(int i=0;i<newmsg->count;i++) CmiFree(newmsg->msgs[i]);
		CmiFree(newmsg->sizes);
		CmiFree(newmsg->msgs);
            }
#else
            int *pelist = (int *)CmiAlloc(two_pow_ndirect * sizeof(int));
            for(int count = 0; count < two_pow_ndirect; count ++){
                int nextpe = count ^ MyPe;
                gmap(nextpe);

                ComlibPrintf("%d Sending to %d\n", MyPe, nextpe);
                pelist[count] = nextpe;
            }
            
            int len;
            char *newmsg;
            newmsg=PeHcube->ExtractAndPackAll(id, stage, &len);
            if (newmsg) {
                CmiSetHandler(newmsg, CkpvAccess(RouterProcHandle));
                CmiSyncListSendAndFree(two_pow_ndirect, pelist, len, newmsg);
            }
	    CmiFree(pelist);
#endif

            stage -= numDirectSteps;

            //if(procMsgCount == two_pow_ndirect)
            //  LocalProcMsg();
        }
        else if(numDirectSteps == 0) {
            LocalProcMsg(id);
            ComlibPrintf("Calling local proc msg %d\n", 
                         buffer[numDirectSteps]);
        }
        
	buffer[numDirectSteps]=0;
    }
}

void HypercubeRouter :: ProcManyMsg(comID id, char *m)
{
    ComlibPrintf("%d: In procmanymsg\n", MyPe);
    InitCounter=setIC(Dim, MyPe, NumPes);
    if(id.isAllToAll) {
        int pe_list[2];
        int npes = 2;

        if(InitCounter > 0)
            npes = 1;
        
        pe_list[0] = MyPe;
        pe_list[1] = neighbor(MyPe, Dim);
        
        PeHcube1->UnpackAndInsertAll(m, npes, pe_list);
    }
    else
        PeHcube->UnpackAndInsert(m);
    
    procMsgCount ++;

    if(InitCounter >= 0){
        if((procMsgCount == two_pow_ndirect) && stage < 0) {
            ComlibPrintf("%d Calling lp %d %d\n", MyPe, 
                         procMsgCount, stage);
            LocalProcMsg(id);
        }
    }
    else
        //CmiPrintf("calling lp in procmsg call\n");
        LocalProcMsg(id);
}

void HypercubeRouter:: LocalProcMsg(comID id)
{
    //CmiPrintf("%d local procmsg called\n", MyPe);

    int mynext=neighbor(MyPe, Dim);
    int mymax=1<<Dim;
    
    if (mynext >=mymax && mynext < NumPes) {
        ComlibPrintf("Before Gmap %d\n", mynext);
        int pelist[1];
        pelist[0] = mynext;
        ComlibPrintf("%d Sending to upper hypercube  %d\n", MyPe, mynext);
        
        if(id.isAllToAll){
            gmap(mynext);
            HCUBESENDFN(id, Dim, -1, 1, pelist, CkpvAccess(RouterProcHandle), mynext, PeHcube1);
        }
        else {
            gmap(mynext);
            HCUBESENDFN(id, Dim, -1, 1, pelist, CkpvAccess(RouterProcHandle), mynext, PeHcube);
        }
    }
  
    if(id.isAllToAll)
        PeHcube1->ExtractAndDeliverLocalMsgs(MyPe, container);
    else
        PeHcube->ExtractAndDeliverLocalMsgs(MyPe, container);

    PeHcube->Purge();
    PeHcube1->Purge();
    InitVars();
    Done(id);
}

void HypercubeRouter::DummyEP(comID id, int msgstage)
{
  if (msgstage >= 0) {
  	buffer[msgstage]=1;
  	RecvManyMsg(id, NULL);
  }
  else {
  	//CmiPrintf("%d Dummy calling lp\n", MyPe);
	LocalProcMsg(id);
  }
}

void HypercubeRouter::CreateStageTable(int numpes, int *destpes)
{
  int *dir=new int[numpes];
  int nextdim, j, i;
  for (i=0;i<numpes;i++) {
	dir[i]=MyPe ^ adjust(Dim, destpes[i]);
  }
  for (nextdim=Dim-1; nextdim>=0; nextdim--) {
    int mask=1<<nextdim;
    for (i=0;i<numpes;i++) {
	if (dir[i] & mask) {
		dir[i]=0;
		for (j=0;(j<penum[nextdim]) && (destpes[i]!=next[nextdim][j]);j++);
		if (destpes[i]==next[nextdim][j]) { 
			//CmiPrintf("EQUAL %d\n", destpes[i]);
			continue;
  		}
		next[nextdim][penum[nextdim]]=destpes[i];
		penum[nextdim]+=1;
		//CmiPrintf("%d next[%d][%d]=%d\n",MyPe, nextdim, penum[nextdim],destpes[i]);
        }
    }
  }
  delete [] dir;
  return;
}

Router * newhcubeobject(int n, int me, Strategy *strat)
{
    Router *obj=new HypercubeRouter(n, me, strat);
    return(obj);
}

//MOVE this CODE else where, this method has been depricated!!!
void HypercubeRouter::SetID(comID id) { 

    if(id.isAllToAll) {
        numDirectSteps = 2;
        two_pow_ndirect = 1;
          for(int count = 0; count < numDirectSteps; count ++)
              two_pow_ndirect *= 2;
    }
}

/*@}*/
