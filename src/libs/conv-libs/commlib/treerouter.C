/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/**************************************
 * File: treerouter.C
 *
 * Author: Krishnan V
 *
 * Tree based router
 *********************************************/
#include "treerouter.h"
#define DEGREE 4
#define gmap(pe) (gpes ? gpes[pe] : pe)


/**The only communication op used. Modify this to use
 ** vector send */
#define TREESENDFN(kid, u, knewmsg, klen, khndl, knextpe)  \
	{if (knewmsg) {\
	  CmiSetHandler(knewmsg, khndl);\
  	  CmiSyncSendAndFree(knextpe, klen, knewmsg);\
	}\
	else {\
	  KSendDummyMsg(kid, knextpe, u);\
	}\
}

/************************************************
 ************************************************/
TreeRouter :: TreeRouter(int n, int me)
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
		char *newmsg=SortBufferUp(id, 0, &len);
		TREESENDFN(MyID, 0, newmsg, len, CpvAccess(RecvHandle), parent);

	}
	else {
		DownStreamMsg(id);
	}
  }
  if (recvCount > recvExpected) DownStreamMsg(id);
}

char * TreeRouter :: SortBufferUp(comID, int ufield, int *len)
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
  char *newmsg=PeTree->ExtractAndPack(MyID, ufield, np, pelst, len); 
  CmiFree(pelst);
  return(newmsg);
}
  
char * TreeRouter :: SortBufferDown(comID , int ufield, int *len, int s)
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

  char * newmsg=PeTree->ExtractAndPack(MyID, ufield, np, plist, len); 
  CmiFree(plist);
  return(newmsg);
}

void TreeRouter :: DownStreamMsg(comID id)
{
  int deg=DEGREE;
  if (NumPes < deg) deg=NumPes;

  for (int i=0;i<deg;i++) {
    int len;
    char *newmsg=SortBufferDown(id, 0, &len, i+1);
    int child=MyPe*DEGREE+i+1;
    if (child >=NumPes || child==MyPe) break;
    child=gmap(child);
    TREESENDFN(MyID, 0, newmsg, len, CpvAccess(RecvHandle), child);
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
  PeTree->ExtractAndDeliverLocalMsgs(MyPe);
  PeTree->Purge();
  InitVars();
  KDone(id);
}

void TreeRouter :: DummyEP(comID id, int)
{
  RecvManyMsg(id, NULL);
}

Router * newtreeobject(int n, int me)
{
  Router * obj=new TreeRouter(n, me);
  return(obj);
}

void TreeRouter :: SetMap(int *pes)
{
  gpes=pes;
}
