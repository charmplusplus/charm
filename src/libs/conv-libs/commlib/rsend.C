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

#define gmap(pe) {if (gpes) pe=gpes[pe];}

RsendRouter::RsendRouter(int n, int me)
{
  //Initialize the no: of pes and my Pe number
  NumPes=n;
  MyPe=me;
  gpes=NULL;
  PSendCounter=0;
  //CmiPrintf("%d Rsend constructor done\n", MyPe);

#if CMK_PERSISTENT_COMM
  handlerArray = new PersistentHandle[NumPes];
  handlerArrayEven = new PersistentHandle[NumPes];
  
  for(int pcount = 0; pcount < NumPes; pcount++){
      handlerArray[pcount] = CmiCreatePersistent(pcount, PERSISTENT_BUFSIZE);
      handlerArrayEven[pcount] = CmiCreatePersistent(pcount, 
                                                     PERSISTENT_BUFSIZE);
  }
#endif
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
    static int step = 0;
    int i;

    if(!more)
        step ++;

    if (!size) return;
    for (i=0;i<numpes-1;i++) {
	//CmiPrintf("%d sending to %d\n", CmiMyPe(), destpes[i]);
#if CMK_PERSISTENT_COMM
        //if(step % 2 == 1)
        //  CmiUsePersistentHandle(&handlerArray[destpes[i]], 1);
        //else
        //  CmiUsePersistentHandle(&handlerArrayEven[destpes[i]], 1);
#endif        
	CmiSyncSend(destpes[i], size, msg);
#if CMK_PERSISTENT_COMM
        CmiUsePersistentHandle(NULL, 0);
#endif        
    }
    
#if CMK_PERSISTENT_COMM
    //if(step % 2 == 1)
    //  CmiUsePersistentHandle(&handlerArray[destpes[i]], 1);
    //else
    //  CmiUsePersistentHandle(&handlerArrayEven[destpes[i]], 1);
#endif        
    CmiSyncSendAndFree(destpes[i], size, msg);

#if CMK_PERSISTENT_COMM
    CmiUsePersistentHandle(NULL, 0);
#endif        

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
