/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/*************************************************
 * File : de.C
 *
 * Author : Krishnan V.
 *
 * Dimensional Exchange (Hypercube) Router 
 *  
 * Modified to send last k stages directly by Sameer Kumar 9/07/03
 *
 ************************************************/
#include "de.h"

#define gmap(pe) (gpes ? gpes[pe] : pe)

/**The only communication op used. Modify this to use
 ** vector send */

#define HCUBESENDFN(kid, u1, u2, knpe, kpelist, khndl, knextpe, pehcube)  \
  	{int len;\
	char *newmsg;\
 	newmsg=pehcube->ExtractAndPack(kid, u1, knpe, kpelist, &len);\
	if (newmsg) {\
	  CmiSetHandler(newmsg, khndl);\
  	  CmiSyncSendAndFree(knextpe, len, newmsg);\
	}\
	else {\
	  KSendDummyMsg(kid, knextpe, u2);\
	}\
}

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
DimexRouter::DimexRouter(int n, int me, int ndirect)
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
  next=(int **)CmiAlloc(sizeof(int *)*Dim);
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
  delete(dp);

  //CmiPrintf("%d DE constructor done dim=%d, mymax=%d IC=%d\n", CkMyPe(), Dim, 1<<Dim, InitCounter);

  if(numDirectSteps > Dim - 1)
      numDirectSteps = Dim - 1;
}
 
DimexRouter :: ~DimexRouter()
{
  int i;
  delete PeHcube;
  delete PeHcube1;
  delete buffer;
  for (i=0;i<Dim;i++) {
	delete next[i];
  }
  delete next;
}

void DimexRouter :: SetMap(int *pes)
{
  gpes=pes;
}

void DimexRouter :: InitVars()
{
  stage=Dim-1;
  InitCounter=setIC(Dim, MyPe, NumPes);
  procMsgCount = 0;
}

void DimexRouter::EachToAllMulticast(comID id, int size, void *msg, int more)
{
  int npe=NumPes;
  int * destpes=(int *)CmiAlloc(sizeof(int)*npe);
  for (int i=0;i<npe;i++) destpes[i]=i;
  EachToManyMulticast(id, size, msg, npe, destpes, more);
  CmiFree(destpes);
}

void DimexRouter::NumDeposits(comID, int num)
{
  //CmiPrintf("Deposit=%d\n", num);
}

void DimexRouter::EachToManyMulticast(comID id, int size, void *msg, int numpes, int *destpes, int more)
{

    SetID(id);

    //Create the message
    if (msg && size) {
  	PeHcube->InsertMsgs(numpes, destpes, size, msg);
    }
    
    if (more >0) return;
    
    if (InitCounter <0) {
        ComlibPrintf("%d Sending to the lower hypercube\n", MyPe);
  	int nextpe=neighbor(MyPe, Dim);
	int * pelist=(int *)CmiAlloc(NumPes*sizeof(int));
	for (int i=0;i<NumPes;i++) {
            pelist[i]=i;
	}
        ComlibPrintf("Before Gmap %d\n", nextpe);
	nextpe=gmap(nextpe);
	HCUBESENDFN(MyID, Dim, Dim, NumPes, pelist, CkpvAccess(RecvHandle), nextpe, PeHcube);
 	CmiFree(pelist);
	return;
    }
    
    //Done: no more stages.
    if (stage <0) {
	//CmiPrintf("calling lp in multicast call %d\n", stage);
	LocalProcMsg();
	return;
    }
    
    InitCounter++;
    RecvManyMsg(id,NULL);
}

//Send the messages for the next stage to the next dimension neighbor
//If only numDirectStage's are left send messages directly using prefix send
void DimexRouter::RecvManyMsg(comID id, char *msg)
{
  //CmiPrintf("%d recv called\n", MyPe);
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
        nextpe=gmap(nextpe);
        ComlibPrintf("%d Sending to %d\n", MyPe, nextpe);
	HCUBESENDFN(MyID, stage, stage, penum[stage], next[stage], CkpvAccess(RecvHandle), nextpe, PeHcube);

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
            
            int *pelist = (int *)CmiAlloc(two_pow_ndirect * sizeof(int));
            for(int count = 0; count < two_pow_ndirect; count ++){
                int nextpe = count ^ MyPe;
                gmap(nextpe);
                
                ComlibPrintf("%d Sending to %d\n", MyPe, nextpe);
                pelist[count] = nextpe;
            }
            
            int len;
            char *newmsg;
            newmsg=PeHcube->ExtractAndPackAll(MyID, stage, &len);
            if (newmsg) {
                CmiSetHandler(newmsg, CkpvAccess(ProcHandle));
                CmiSyncListSendAndFree(two_pow_ndirect, pelist, len, newmsg);
            }
            
            stage -= numDirectSteps;

            //if(procMsgCount == two_pow_ndirect)
            //  LocalProcMsg();
        }
        else if(numDirectSteps == 0) {
            LocalProcMsg();
            ComlibPrintf("Calling local proc msg %d\n", 
                         buffer[numDirectSteps]);
        }
        
	buffer[numDirectSteps]=0;
    }
}

void DimexRouter :: ProcManyMsg(comID, char *m)
{

    InitCounter=setIC(Dim, MyPe, NumPes);
    if(MyID.isAllToAll) {
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
            ComlibPrintf("%d Calling lp %d %d\n", CkMyPe(), 
                         procMsgCount, stage);
            LocalProcMsg();
        }
    }
    else
        //CmiPrintf("calling lp in procmsg call\n");
        LocalProcMsg();
}

void DimexRouter:: LocalProcMsg()
{
    //CmiPrintf("%d local procmsg called\n", CkMyPe());

    int mynext=neighbor(MyPe, Dim);
    int mymax=1<<Dim;
    
    if (mynext >=mymax && mynext < NumPes) {
        ComlibPrintf("Before Gmap %d\n", mynext);
        mynext=gmap(mynext);
        int *pelist=&mynext;
        ComlibPrintf("%d Sending to %d\n", MyPe, mynext);        
        
        if(MyID.isAllToAll){
            HCUBESENDFN(MyID, Dim, -1, 1, pelist, CkpvAccess(ProcHandle), mynext, PeHcube1);
        }
        else {
            HCUBESENDFN(MyID, Dim, -1, 1, pelist, CkpvAccess(ProcHandle), mynext, PeHcube);
        }
    }
  
    if(MyID.isAllToAll)
        PeHcube1->ExtractAndDeliverLocalMsgs(MyPe);
    else
        PeHcube->ExtractAndDeliverLocalMsgs(MyPe);

    PeHcube->Purge();
    PeHcube1->Purge();
    InitVars();
    KDone(MyID);
}

void DimexRouter::DummyEP(comID id, int msgstage)
{
  if (msgstage >= 0) {
  	buffer[msgstage]=1;
  	RecvManyMsg(id, NULL);
  }
  else {
  	//CmiPrintf("%d Dummy calling lp\n", MyPe);
	LocalProcMsg();
  }
}

void DimexRouter::CreateStageTable(int numpes, int *destpes)
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
  delete dir;
  return;
}

Router * newhcubeobject(int n, int me)
{
    Router *obj=new DimexRouter(n, me);
    return(obj);
}


void DimexRouter::SetID(comID id) { 
    MyID=id;

    if(MyID.isAllToAll) {
        numDirectSteps = 2;
        two_pow_ndirect = 1;
          for(int count = 0; count < numDirectSteps; count ++)
              two_pow_ndirect *= 2;
    }
}
