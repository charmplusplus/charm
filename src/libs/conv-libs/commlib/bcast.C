/*********************************************
* File : bcast.C
* Author : Krishnan V
*
* Broadcast based multicast. Not fully functional
* Doesnot work with multicast groups
**************************************************/
#include <strings.h>
#include "bcast.h"

#define NULL 0
#define gmap(pe) {if (gpes) pe=gpes[pe];}

BcastRouter::BcastRouter(int n, int me)
{
  //Initialize the no: of pes and my Pe number
  NumPes=n;
  MyPe=me;
  gpes=NULL;
  //CmiPrintf("Bcast with n=%d, me=%d\n", n, me);
  InitVars();
}
 
BcastRouter :: ~BcastRouter()
{
}

void BcastRouter :: InitVars()
{
  PSendCounter=0;
  PSendExpected=0;
  recvCount=0;
  MsgBuffer=NULL;
}

void BcastRouter::EachToAllMulticast(comID id, int size, void *msg, int more)
{
  PSendCounter++;

  int npe=NumPes;
  int * destpes=(int *)CmiAlloc(sizeof(int)*npe);
  for (int i=0;i<npe;i++) destpes[i]=i;
  EachToManyMulticast(id, size, msg, npe, destpes, more);
}

void BcastRouter::NumDeposits(comID, int num)
{
  PSendExpected=num;
}

void BcastRouter::EachToManyMulticast(comID id, int size, void *msg, int numpes, int *destpes, int more)
{
  PSendCounter++;
  Buffer *node;

  if (size) {

  int mask=~7;
  int offset=(CmiMsgHeaderSizeBytes+sizeof(comID)+3*sizeof(int)+7)&mask;
  int totsize=offset+numpes*sizeof(int)+size;
  char *m=(char *)CmiAlloc(totsize);
  char *p=m+CmiMsgHeaderSizeBytes;

  CmiSetHandler(m, CpvAccess(RecvHandle));

  int refno=KMyActiveRefno(id);
  memcpy(p, (char *)&refno, sizeof(int)); 
  p +=sizeof(int);

  memcpy(p, (char *)&MyID, sizeof(comID));
  p +=sizeof(comID);

  memcpy(p, (char *)&size, sizeof(int));
  p +=sizeof(int);

  memcpy(p, (char *)&numpes, sizeof(int));
  p +=sizeof(int);

  p=m+offset;

  for (int i=0;i<numpes;i++) {
  	*(int *)p= destpes[i];
	p += sizeof(int);
  }

  memcpy(p, msg, size);

  //CmiPrintf("%d broadcasting refno=%d\n", MyPe,refno);
  CmiSyncBroadcastAndFree(totsize, m);

  node=(Buffer *)CmiAlloc(sizeof(Buffer));
  node->size=size;
  node->msg=msg;
  node->next=MsgBuffer;
  MsgBuffer=node;
  }

  //if (PSendCounter == PSendExpected) recvCount++;
  if (!more) recvCount++;
  //CmiPrintf("%d more=%d recvcount=%d\n", MyPe, more, recvCount);
  if (recvCount==NumPes) {
  	while (MsgBuffer) {
  		CmiSyncSendAndFree(MyPe, MsgBuffer->size, MsgBuffer->msg);
		node=MsgBuffer;
		MsgBuffer=MsgBuffer->next;
		CmiFree(node);
	}
	InitVars();
  	//if (PSendCounter >= PSendExpected) KDone(MyID);
  	KDone(MyID);
  }
}


void BcastRouter::RecvManyMsg(comID, char *m)
{
  //CmiPrintf("%d recvd msg\n", MyPe);
  if (m==NULL) {
  	CmiPrintf("%d null msg for Bcast recvmsg\n", MyPe);
	return;
  }
  Buffer *node;
  int mask=~7;
  int offset=(CmiMsgHeaderSizeBytes+sizeof(comID)+3*sizeof(int)+7)&mask;
  char *p=m+CmiMsgHeaderSizeBytes+sizeof(int)+sizeof(comID);

  int size;
  memcpy(&size, p, sizeof(int));
  p+=sizeof(int);

  int npe;
  memcpy(&npe, p, sizeof(int));
  p+=sizeof(int);

  p = m+offset;
  
  int flag=0;
  for (int i=0;i<npe;i++) {
  	int pe=*(int *)p;
	//CmiPrintf("pe[%d]=%d\n", i, pe);
	if (pe == MyPe) flag=1;
	p+=sizeof(int);
  }
  //p -=sizeof(int);

  if (flag) {
  	node=(Buffer *)CmiAlloc(sizeof(Buffer));
  	node->size=size;
  	node->msg=CmiAlloc(size);
	memcpy(node->msg, p, size);
  	node->next=MsgBuffer;
  	MsgBuffer=node;
  }

  CmiFree(m);

  recvCount++;
  if (recvCount==NumPes) {
  	while (MsgBuffer) {
		//CmiPrintf("%d Calling CmiSyncSend with size=%d\n", MyPe, MsgBuffer->size);
  		CmiSyncSendAndFree(MyPe, MsgBuffer->size, MsgBuffer->msg);
		node=MsgBuffer;
		MsgBuffer=MsgBuffer->next;
		CmiFree(node);
	}
	InitVars();
  	//if (PSendCounter == PSendExpected) KDone(MyID);
  	KDone(MyID);
  }
}

void BcastRouter :: ProcManyMsg(comID, char *)
{
}

void BcastRouter :: SetID(comID id)
{
  MyID=id;
}

Router * newbcastobject(int n, int me)
{
  Router *obj=new BcastRouter(n, me);
  return(obj);
}

void BcastRouter :: SetMap(int *pes)
{
  gpes=pes;
}
