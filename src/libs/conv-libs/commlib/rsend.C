/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/*************************************
 * File: rsend.C
 *
 * Author : Krishnan V.
 *
 * Multicast based on repeated sends
 *****************************************/
#include "rsend.h"

#ifndef NULL
#define NULL 0
#endif

#define gmap(pe) {if (gpes) pe=gpes[pe];}

RsendRouter::RsendRouter(int n, int me)
{
  //Initialize the no: of pes and my Pe number
  NumPes=n;
  MyPe=me;
  gpes=NULL;
  PSendCounter=0;
  //CmiPrintf("%d Rsend constructor done\n", MyPe);
}
 
RsendRouter :: ~RsendRouter()
{
}

void RsendRouter::EachToAllMulticast(comID id, int size, void *msg, int more)
{
  int npe=NumPes;
  int * destpes=(int *)CmiAlloc(sizeof(int)*npe);
  for (int i=0;i<npe;i++) destpes[i]=i;
  EachToManyMulticast(id, size, msg, npe, destpes, more);
}

void RsendRouter::NumDeposits(comID, int num)
{
  PSendExpected=num;
}

void RsendRouter::EachToManyMulticast(comID id, int size, void *msg, int numpes, int *destpes, int more)
{
  int i;

  if (!size) return;
  for (i=0;i<numpes-1;i++) {
	//CmiPrintf("%d sending to %d\n", CmiMyPe(), destpes[i]);
	CmiSyncSend(destpes[i], size, msg);
  }
  CmiSyncSendAndFree(destpes[i], size, msg);
  //CmiPrintf("%d rsend calling kdone\n", MyPe);
  KDone(id);
}


void RsendRouter::RecvManyMsg(comID , char *)
{
}

void RsendRouter :: ProcManyMsg(comID, char *)
{
}

void RsendRouter :: DummyEP(comID , int)
{
}

void RsendRouter :: SetID(comID id)
{
  MyID=id;
}

Router * newrsendobject(int n, int me)
{
  Router *obj=new RsendRouter(n, me);
  return(obj);
}

void RsendRouter :: SetMap(int * pes)
{
  gpes=pes;
}
