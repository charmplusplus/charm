/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/********************************************************
 * File : commlib.C
 *
 * Author : Krishnan V
 *
 * All the entry points and the user interface functions 
 *******************************************************/
#include <stdlib.h>

int comm_debug = 0;

#include "commlib.h"
#include "commlib_pvt.h"
#include "globals.h"
#include "overlapper.h"

extern "C" void CqsEnqueueLifo(void *, void *);

#define MAXINSTANCE 100

Router * newgridobject(int, int);
Router * newd3gridobject(int, int);
Router * newtreeobject(int, int);
Router * newhcubeobject(int, int);
Router * newrsendobject(int, int);
Router * newbcastobject(int, int);
Router * newgraphobject(int, int);

/********************************************
 * Internal utility functions
 ********************************************/
void UpdateImplTable(comID id)
{
  if (CkpvAccess(ImplTable)[id.srcpe] == NULL) {
        CkpvAccess(ImplTable)[id.srcpe]=(Overlapper **) CmiAlloc(sizeof(Overlapper *)*MAXINSTANCE);
        for (int j=1;j<MAXINSTANCE;j++) CkpvAccess(ImplTable)[id.srcpe][j]=0;
        CkpvAccess(ImplTable)[id.srcpe][0]=0;
  }

  if (CkpvAccess(ImplTable)[id.srcpe][id.ImplIndex]) {
  	if (id.SwitchVal >0) { 
  	  CkpvAccess(ImplTable)[id.srcpe][id.ImplIndex]->SetID(id);
  	}
	return;
  }

  if(CkpvAccess(ImplTable)[id.srcpe][id.ImplIndex]== NULL)
      CkpvAccess(ImplTable)[id.srcpe][id.ImplIndex]=new Overlapper(id);

  if (id.SwitchVal >0) { 
      CkpvAccess(ImplTable)[id.srcpe][id.ImplIndex]->SetID(id);
  }
}

int ComlibRegisterStrategy(NEWFN newfun)
{
  CkpvAccess(StrategyTable)[CkpvAccess(StrategyTableIndex)++]=newfun;
  return(CkpvAccess(StrategyTableIndex)-1);
}

/**************************************************
 * User Interface functions
 *************************************************/
void ComlibInit()
{
  CkpvInitialize(int, ImplIndex);
  CkpvInitialize(int, RecvHandle);
  CkpvInitialize(int, ProcHandle);
  CkpvInitialize(int, DummyHandle);
  CkpvInitialize(int, SwitchHandle);
  CkpvInitialize(int, KDoneHandle);
  CkpvInitialize(int, KGMsgHandle);
  CkpvInitialize(Overlapperppp, ImplTable);
  CkpvInitialize(NEWFN *, StrategyTable);
  CkpvInitialize(int, StrategyTableIndex);

  CkpvAccess(StrategyTable)=(NEWFN *)CmiAlloc(MAXNUMSTRATEGY*sizeof(NEWFN *));
  CkpvAccess(StrategyTableIndex)=0;

  CkpvAccess(RecvHandle)=CkRegisterHandler((CmiHandler)KRecvManyCombinedMsg);
  CkpvAccess(ProcHandle)=CkRegisterHandler((CmiHandler)KProcManyCombinedMsg);
  CkpvAccess(DummyHandle)=CkRegisterHandler((CmiHandler)KDummyEP);
  CkpvAccess(SwitchHandle)=CkRegisterHandler((CmiHandler)KSwitchEP);
  CkpvAccess(KDoneHandle)=CkRegisterHandler((CmiHandler)KDoneEP);
  CkpvAccess(KGMsgHandle)=CkRegisterHandler((CmiHandler)KGMsgHandler);

  CkpvAccess(ImplIndex)=1;
  CkpvAccess(ImplTable)=(Overlapper ***) CmiAlloc(sizeof(Overlapper *)*CkNumPes());
  
  for (int i=0;i<CkNumPes();i++) {
      CkpvAccess(ImplTable)[i]=NULL;
  }
  
  ComlibRegisterStrategy(newbcastobject);
  ComlibRegisterStrategy(newtreeobject);
  ComlibRegisterStrategy(newgridobject);
  ComlibRegisterStrategy(newhcubeobject);
  ComlibRegisterStrategy(newrsendobject);
  ComlibRegisterStrategy(newd3gridobject);
  ComlibRegisterStrategy(newgraphobject);
}

Router * GetStrategyObject(int n, int me, int indx)
{
    return((CkpvAccess(StrategyTable)[indx])(n, me));
}

//comID CreateInstance(int ImplType, int numpart)
comID ComlibInstance(int ImplType, int numpart)
{
  comID id;
  id.ImplType=ImplType;
  id.ImplIndex=CkpvAccess(ImplIndex);
  id.srcpe=CkMyPe();
  id.SwitchVal=-1;
  id.callbackHandler = 0;
  id.isAllToAll = 0;

  id.NumMembers=numpart;
  UpdateImplTable(id);

  CkpvAccess(ImplIndex) +=1 % MAXINSTANCE;
  return(id);
}

comID ComlibEstablishGroup(comID id, int npes, int *pes) 
{
  id.NumMembers=-1;
  id.grp=CmiEstablishGroup(npes, pes);
  UpdateImplTable(id);
  
  CkpvAccess(ImplTable)[id.srcpe][id.ImplIndex]->GroupMap(npes, pes);
  CkpvAccess(ImplTable)[id.srcpe][id.ImplIndex]->SetID(id);
  return(id);
}

Overlapper * GetComlibObject(comID id)
{
  UpdateImplTable(id);
  return(CkpvAccess(ImplTable)[id.srcpe][id.ImplIndex]);
}

void DeleteInstance(comID id)
{
  Overlapper *o=CkpvAccess(ImplTable)[id.srcpe][id.ImplIndex];
  o->DeleteInstance();
  //CkpvAccess(ImplTable)[id.srcpe][id.ImplIndex]=NULL;
}

void EachToAllMulticast(comID id, int size, void * msg)
{
  UpdateImplTable(id);
  (CkpvAccess(ImplTable)[id.srcpe][id.ImplIndex])->EachToAllMulticast(id, size, msg);
}

void EachToManyMulticast(comID id, int size, void * msg, int pesize, int * pelist)
{
  UpdateImplTable(id);
  (CkpvAccess(ImplTable)[id.srcpe][id.ImplIndex])->EachToManyMulticast(id, size, msg, pesize, pelist);
}

void NumDeposits(comID id, int num)
{
  UpdateImplTable(id);
  (CkpvAccess(ImplTable)[id.srcpe][id.ImplIndex])->NumDeposits(id, num);
}

/*****************************************************
 * Entry Points
 ****************************************************/
void KRecvManyCombinedMsg(char *msg)
{
    //  ComlibPrintf("In Recv combined message at %d\n", CkMyPe());

  comID id;
  memcpy(&id,(msg+CmiReservedHeaderSize+sizeof(int)), sizeof(comID));

  UpdateImplTable(id);
  (CkpvAccess(ImplTable)[id.srcpe][id.ImplIndex])->RecvManyMsg(id, msg);
}

void KProcManyCombinedMsg(char *msg)
{
  comID id;
  //  ComlibPrintf("In Recv combined message at %d\n", CkMyPe());
  memcpy(&id,(msg+CmiReservedHeaderSize+sizeof(int)), sizeof(comID));

  UpdateImplTable(id);
  (CkpvAccess(ImplTable)[id.srcpe][id.ImplIndex])->ProcManyMsg(id, msg);
}

void KDummyEP(DummyMsg *m)
{

    //  ComlibPrintf("In Recv dummy message at %d\n", CkMyPe());
  comID id=m->id;
  UpdateImplTable(id);
  CkpvAccess(ImplTable)[id.srcpe][id.ImplIndex]->DummyEP(m->id, m->magic, m->refno);
  CmiFree(m);
}

/****************************************************
 * Routines used by the router objects
 ***************************************************/
void KBcastSwitchMsg(comID id)
{
  SwitchMsg *m=(SwitchMsg *)CmiAlloc(sizeof(SwitchMsg));
  CmiSetHandler(m, CkpvAccess(SwitchHandle));
  m->id=id;
  CmiSyncBroadcastAndFree(sizeof(SwitchMsg), (char *)m);
}

void KSwitchEP(SwitchMsg *m)
{
  //ComlibPrintf("%d switchep called\n", CkMyPe());
  comID id=m->id;
  UpdateImplTable(id);

  CmiFree(m);
} 

void KSendDummyMsg(comID id, int pe, int magic)
{
  DummyMsg *m=(DummyMsg *)CmiAlloc(sizeof(DummyMsg));
  CmiSetHandler(m, CkpvAccess(DummyHandle));
  m->id=id;
  m->magic=magic;
  m->refno=KMyActiveRefno(id);
  CmiSyncSendAndFree(pe, sizeof(DummyMsg),(char*) m);
}

void KCsdEnqueue(void *m)
{
  CsdEnqueueLifo(m);
}

int KMyActiveRefno(comID id)
{
  //ComlibPrintf("KMyActive calling update\n");
  return(CkpvAccess(ImplTable)[id.srcpe][id.ImplIndex]->MyActiveIndex());
}

void KDoneEP(DummyMsg *m)
{
  comID id=m->id;
  CkpvAccess(ImplTable)[id.srcpe][id.ImplIndex]->Done();
  
  if(id.callbackHandler != 0) {
      CmiSetHandler(m, id.callbackHandler);
      CmiSyncSendAndFree(CkMyPe(), sizeof(DummyMsg), (char*)m);
  }
  else 
      CmiFree(m);
}

void KDone(comID id)
{
  DummyMsg *m=(DummyMsg *)CmiAlloc(sizeof(DummyMsg));
  m->id=id;
  CmiSetHandler(m, CkpvAccess(KDoneHandle));
  CmiSyncSendAndFree(CkMyPe(), sizeof(DummyMsg), (char*)m);
}

/****************************************************
 * Multicast Groups
 ***************************************************/
void KsendGmsg(comID id)
{
  GMsg *gmsg=(GMsg *)CmiAlloc(sizeof(GMsg));
  gmsg->id=id;
  CmiSetHandler(gmsg, CkpvAccess(KGMsgHandle));
  CmiSyncSendAndFree(CkMyPe(),sizeof(GMsg), (char*)gmsg);
  //KCsdEnqueue(gmsg);
}

void KGMsgHandler(GMsg *msg)
{
  //ComlibPrintf("KGhandler called\n");
  comID id=msg->id;
  int npes, *pes;
  //CmiGroup grp=id.grp;
  //ComlibPrintf("grppe=%d, grpindex=%d\n", grp.pe, grp.id);
  CmiLookupGroup((msg->id).grp, &npes, &pes);
  if (pes==0) {
  	//CmiSyncSendAndFree(CkMyPe(),sizeof(GMsg), (char*)msg);
	KCsdEnqueue(msg);
  }
  else {
  	(CkpvAccess(ImplTable)[id.srcpe][id.ImplIndex])->GroupMap(npes, pes);
  }
}

