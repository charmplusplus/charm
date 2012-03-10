/**
   @addtogroup ConvComlibRouter
   @{
   @file 
   @brief  Tree based converse level routing strategy 
*/

#include "treerouter.h"
#define DEGREE 4
#define gmap(pe) (gpes ? gpes[pe] : pe)


/**The only communication op used. Modify this to use vector send. */
#if CMK_COMLIB_USE_VECTORIZE
#define TREESENDFN(kid, u, knewmsg, khndl, knextpe)  \
	{if (knewmsg) {\
	  CmiSetHandler(knewmsg->msgs[0], khndl);\
  	  CmiSyncVectorSendAndFree(knextpe, -knewmsg->count, knewmsg->sizes, knewmsg->msgs);\
	}\
	else {\
	  SendDummyMsg(kid, knextpe, u);\
	}\
}
#else
#define TREESENDFN(kid, u, knewmsg, klen, khndl, knextpe)  \
	{if (knewmsg) {\
	  CmiSetHandler(knewmsg, khndl);\
  	  CmiSyncSendAndFree(knextpe, klen, knewmsg);\
	}\
	else {\
	  SendDummyMsg(kid, knextpe, u);\
	}\
}
#endif




TreeRouter :: TreeRouter(int n, int me, Strategy *parent) : Router(parent)
{
  int i;
  MyPe=me;
  NumPes=n;
  gpes=NULL;

  numChildren=0;
  for (i=0;i<DEGREE;i++) {
  	if ((MyPe*DEGREE+i+1) < NumPes) numChildren++;
  }

  PeTree = new PeTable(NumPes);

  recvExpected=1+numChildren;
  InitVars();
  //CmiPrintf("Tree with n=%d, me=%d\n", n, me);
}

TreeRouter :: ~TreeRouter()
{
  delete PeTree;
}

void TreeRouter :: InitVars()
{
  recvCount=0;
}


void TreeRouter::NumDeposits(comID , int num)
{
}

void TreeRouter::EachToAllMulticast(comID id, int size, void *msg, int more)
{
  int npe=NumPes;
  int * destpes=(int *)CmiAlloc(sizeof(int)*npe);
  for (int i=0;i<npe;i++) destpes[i]=i;
  EachToManyMulticast(id, size, msg, npe, destpes, more);
}

void TreeRouter::EachToManyMulticast(comID id, int size, void *msg, int numpes, int *destpes, int more)
{
 //Create the message
  if (size) {
  	PeTree->InsertMsgs(numpes, destpes, size, msg);
  }

  if (more >0) return;

  //Hand it over to Recv. 
  RecvManyMsg(id, NULL);
}

void TreeRouter :: RecvManyMsg(comID id, char *msg)
{
  if (msg)
	PeTree->UnpackAndInsert(msg);
  recvCount++;
  //CmiPrintf("%d Tree: recvCount=%d, recvExpected=%d\n",MyPe, recvCount, recvExpected);
  if (recvCount == recvExpected) {
	if (MyPe) {
		int len;
		int parent=(MyPe-1)/DEGREE;
		parent=gmap(parent);
#if CMK_COMLIB_USE_VECTORIZE
		PTvectorlist newmsg=SortBufferUp(id, 0);
		TREESENDFN(id, 0, newmsg, CkpvAccess(RouterRecvHandle), parent);
#else
		char *newmsg=SortBufferUp(id, 0, &len);
		TREESENDFN(id, 0, newmsg, len, CkpvAccess(RouterRecvHandle), parent);
#endif
	}
	else {
		DownStreamMsg(id);
	}
  }
  if (recvCount > recvExpected) DownStreamMsg(id);
}

#if CMK_COMLIB_USE_VECTORIZE
PTvectorlist TreeRouter :: SortBufferUp(comID id, int ufield)
#else
char * TreeRouter :: SortBufferUp(comID id, int ufield, int *len)
#endif
{
  int np=0, i;
  int * pelst=(int *)CmiAlloc(sizeof(int)*NumPes);

  for (i=0;i<NumPes;i++) {

	//if (i==MyPe) continue;
	int pe=i;
	while (pe!=MyPe && pe>0) pe =(pe-1)/DEGREE;
	if (pe == MyPe) continue;

	pelst[np++]=i;
  }
#if CMK_COMLIB_USE_VECTORIZE
  PTvectorlist newmsg=PeTree->ExtractAndVectorize(id, ufield, np, pelst); 
#else
  char *newmsg=PeTree->ExtractAndPack(id, ufield, np, pelst, len); 
#endif
  CmiFree(pelst);
  return(newmsg);
}
  
#if CMK_COMLIB_USE_VECTORIZE
PTvectorlist TreeRouter :: SortBufferDown(comID id, int ufield, int s)
#else
char * TreeRouter :: SortBufferDown(comID id, int ufield, int *len, int s)
#endif
{
  int np=0, i;
  int * plist=(int *)CmiAlloc(sizeof(int)*NumPes);

  for (i=0;i<NumPes;i++) {
	if (i==MyPe) continue;
	int pe=i;
	int rep=MyPe*DEGREE+s;
	while (pe!=rep && pe>0) pe =(pe-1)/DEGREE;
	if (pe == rep) plist[np++]=i;
  }

#if CMK_COMLIB_USE_VECTORIZE
  PTvectorlist newmsg=PeTree->ExtractAndVectorize(id, ufield, np, plist); 
#else
  char * newmsg=PeTree->ExtractAndPack(id, ufield, np, plist, len); 
#endif
  CmiFree(plist);
  return(newmsg);
}

void TreeRouter :: DownStreamMsg(comID id)
{
  int deg=DEGREE;
  if (NumPes < deg) deg=NumPes;

  for (int i=0;i<deg;i++) {
    int len;
#if CMK_COMLIB_USE_VECTORIZE
    PTvectorlist newmsg=SortBufferDown(id, 0, i+1);
#else
    char *newmsg=SortBufferDown(id, 0, &len, i+1);
#endif
    int child=MyPe*DEGREE+i+1;
    if (child >=NumPes || child==MyPe) break;
    child=gmap(child);
#if CMK_COMLIB_USE_VECTORIZE
    TREESENDFN(id, 0, newmsg, CkpvAccess(RouterRecvHandle), child);
#else
    TREESENDFN(id, 0, newmsg, len, CkpvAccess(RouterRecvHandle), child);
#endif
  }

  LocalProcMsg(id);
}

void TreeRouter :: ProcManyMsg(comID id, char *m)
{
  PeTree->UnpackAndInsert(m);
  LocalProcMsg(id);
}

void TreeRouter:: LocalProcMsg(comID id)
{
  PeTree->ExtractAndDeliverLocalMsgs(MyPe, container);
  PeTree->Purge();
  InitVars();
  Done(id);
}

void TreeRouter :: DummyEP(comID id, int)
{
  RecvManyMsg(id, NULL);
}

Router * newtreeobject(int n, int me, Strategy *strat)
{
  Router * obj=new TreeRouter(n, me, strat);
  return(obj);
}

void TreeRouter :: SetMap(int *pes)
{
  gpes=pes;
}


/*@}*/
