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
 ************************************************/
#include "de.h"

#define gmap(pe) (gpes ? gpes[pe] : pe)

/**The only communication op used. Modify this to use
 ** vector send */
//CmiPrintf("[%d]DE sending message of size %d to %d\n", CmiMyPe(), len, knextpe); 

#define HCUBESENDFN(kid, u1, u2, knpe, kpelist, khndl, knextpe)  \
  	{int len;\
	char *newmsg;\
 	newmsg=PeHcube->ExtractAndPack(kid, u1, knpe, kpelist, &len);\
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
DimexRouter::DimexRouter(int n, int me)
{
  int i;

  //Initialize the no: of pes and my Pe number
  NumPes=n;
  MyPe=me;
  gpes=NULL;

  //Initialize Dimension and no: of stages
  Dim=maxdim(NumPes);

  PeHcube=new PeTable(NumPes);

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

  //CmiPrintf("%d DE constructor done dim=%d, mymax=%d IC=%d\n", CmiMyPe(), Dim, 1<<Dim, InitCounter);
}
 
DimexRouter :: ~DimexRouter()
{
  int i;
  delete PeHcube;
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
 //Create the message
  if (size) {
  	PeHcube->InsertMsgs(numpes, destpes, size, msg);
  }

  if (more >0) return;

  if (InitCounter <0) {
	//CmiPrintf("%d Sending to the lower hypercube\n", MyPe);
  	int nextpe=neighbor(MyPe, Dim);
	int * pelist=(int *)CmiAlloc(NumPes*sizeof(int));
	for (int i=0;i<NumPes;i++) {
		pelist[i]=i;
	}
	nextpe=gmap(nextpe);
	HCUBESENDFN(MyID, Dim, Dim, NumPes, pelist, CpvAccess(RecvHandle), nextpe);
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
  while ((InitCounter==2) || (stage >=0 && buffer[stage+1])) {
	InitCounter=setIC(Dim, MyPe, NumPes);
  	if (InitCounter != 2) { 
		buffer[stage+1]=0;
 	}
  	//Send the data to the neighbor in this stage
  	int nextpe=neighbor(MyPe, stage);
        nextpe=gmap(nextpe);
	HCUBESENDFN(MyID, stage, stage, penum[stage], next[stage], CpvAccess(RecvHandle), nextpe);

  	//Go to the next stage
  	stage--; 
  }
  if (stage <0 && buffer[0]) {
	//CmiPrintf("Calling local proc msg %d\n", buffer[0]);
	buffer[0]=0;
	LocalProcMsg();
  }
}

void DimexRouter :: ProcManyMsg(comID, char *m)
{
  int msgstage=PeHcube->UnpackAndInsert(m);

  //CmiPrintf("calling lp in procmsg call\n");
  LocalProcMsg();
}

void DimexRouter:: LocalProcMsg()
{
  //CmiPrintf("%d local procmsg called\n", CmiMyPe());

  int mynext=neighbor(MyPe, Dim);
  int mymax=1<<Dim;

  if (mynext >=mymax && mynext < NumPes) {
	mynext=gmap(mynext);
	int *pelist=&mynext;
	HCUBESENDFN(MyID, Dim, -1, 1, pelist, CpvAccess(ProcHandle), mynext);
  }
  
  PeHcube->ExtractAndDeliverLocalMsgs(MyPe);
  PeHcube->Purge();
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
