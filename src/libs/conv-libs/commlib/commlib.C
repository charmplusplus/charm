/********************************************************
 * File : commlib.C
 *
 * Author : Krishnan V
 *
 * All the entry points and the user interface functions 
 *******************************************************/
#include <stdlib.h>
#include "commlib.h"
#include "commlib_pvt.h"
#include "globals.h"
#include "overlapper.h"

extern "C" void CqsEnqueueLifo(void *, void *);

#define MAXINSTANCE 100

Router * newgridobject(int, int);
Router * newtreeobject(int, int);
Router * newhcubeobject(int, int);
Router * newrsendobject(int, int);
Router * newbcastobject(int, int);

/********************************************
 * Internal utility functions
 ********************************************/
void UpdateImplTable(comID id)
{
  if (CpvAccess(ImplTable)[id.srcpe] == NULL) {
	CpvAccess(ImplTable)[id.srcpe]=(Overlapper **) CmiAlloc(sizeof(Overlapper *)*MAXINSTANCE);
        for (int j=1;j<MAXINSTANCE;j++) CpvAccess(ImplTable)[id.srcpe][j]=0;
        CpvAccess(ImplTable)[id.srcpe][0]=0;
  }

  if (CpvAccess(ImplTable)[id.srcpe][id.ImplIndex]) {
  	if (id.SwitchVal >0) { 
  	  CpvAccess(ImplTable)[id.srcpe][id.ImplIndex]->SetID(id);
  	}
	return;
  }

  CpvAccess(ImplTable)[id.srcpe][id.ImplIndex]=new Overlapper(id);

  if (id.SwitchVal >0) { 
  	CpvAccess(ImplTable)[id.srcpe][id.ImplIndex]->SetID(id);
  }
}

int ComlibRegisterStrategy(NEWFN newfun)
{
  CpvAccess(StrategyTable)[CpvAccess(StrategyTableIndex)++]=newfun;
  return(CpvAccess(StrategyTableIndex)-1);
}

/**************************************************
 * User Interface functions
 *************************************************/
void ComlibInit()
{
  CpvInitialize(int, ImplIndex);
  CpvInitialize(int, RecvHandle);
  CpvInitialize(int, ProcHandle);
  CpvInitialize(int, DummyHandle);
  CpvInitialize(int, SwitchHandle);
  CpvInitialize(int, KDoneHandle);
  CpvInitialize(int, KGMsgHandle);
  CpvInitialize(Overlapperppp, ImplTable);
  CpvInitialize(NEWFN *, StrategyTable);
  CpvInitialize(int, StrategyTableIndex);

  CpvAccess(StrategyTable)=(NEWFN *)CmiAlloc(MAXNUMSTRATEGY*sizeof(NEWFN *));
  CpvAccess(StrategyTableIndex)=0;

  CpvAccess(RecvHandle)=CmiRegisterHandler((CmiHandler)KRecvManyCombinedMsg);
  CpvAccess(ProcHandle)=CmiRegisterHandler((CmiHandler)KProcManyCombinedMsg);
  CpvAccess(DummyHandle)=CmiRegisterHandler((CmiHandler)KDummyEP);
  CpvAccess(SwitchHandle)=CmiRegisterHandler((CmiHandler)KSwitchEP);
  CpvAccess(KDoneHandle)=CmiRegisterHandler((CmiHandler)KDoneEP);
  CpvAccess(KGMsgHandle)=CmiRegisterHandler((CmiHandler)KGMsgHandler);

  CpvAccess(ImplIndex)=1;
  CpvAccess(ImplTable)=(Overlapper ***) CmiAlloc(sizeof(Overlapper *)*CmiNumPes());
  
  for (int i=0;i<CmiNumPes();i++) {
	CpvAccess(ImplTable)[i]=NULL;
	//CpvAccess(ImplTable)[i]=(Overlapper **) CmiAlloc(sizeof(Overlapper *)*MAXINSTANCE);
        //for (int j=0;j<MAXINSTANCE;j++) CpvAccess(ImplTable)[i][j]=0;
  }
  ComlibRegisterStrategy(newbcastobject);
  ComlibRegisterStrategy(newtreeobject);
  ComlibRegisterStrategy(newgridobject);
  ComlibRegisterStrategy(newhcubeobject);
  ComlibRegisterStrategy(newrsendobject);
}


Router * GetStrategyObject(int n, int me, int indx)
{
  return((CpvAccess(StrategyTable)[indx])(n, me));
}

//comID CreateInstance(int ImplType, int numpart)
comID ComlibInstance(int ImplType, int numpart)
{
  comID id;
  id.ImplType=ImplType;
  id.ImplIndex=CpvAccess(ImplIndex);
  id.srcpe=CmiMyPe();
  id.SwitchVal=-1;
  id.NumMembers=numpart;
  UpdateImplTable(id);

  CpvAccess(ImplIndex) +=1 % MAXINSTANCE;
  return(id);
}

comID ComlibEstablishGroup(comID id, int npes, int *pes) 
{
  id.NumMembers=-1;
  id.grp=CmiEstablishGroup(npes, pes);
  UpdateImplTable(id);
  CpvAccess(ImplTable)[id.srcpe][id.ImplIndex]->SetID(id);
  return(id);
}

Overlapper * GetComlibObject(comID id)
{
  UpdateImplTable(id);
  return(CpvAccess(ImplTable)[id.srcpe][id.ImplIndex]);
}

void DeleteInstance(comID id)
{
  Overlapper *o=CpvAccess(ImplTable)[id.srcpe][id.ImplIndex];
  o->DeleteInstance();
  //CpvAccess(ImplTable)[id.srcpe][id.ImplIndex]=NULL;
}

void EachToAllMulticast(comID  id, int size, void * msg)
{
  UpdateImplTable(id);
  (CpvAccess(ImplTable)[id.srcpe][id.ImplIndex])->EachToAllMulticast(id, size, msg);
}

void EachToManyMulticast(comID  id, int size, void * msg, int pesize, int * pelist)
{
  UpdateImplTable(id);
  (CpvAccess(ImplTable)[id.srcpe][id.ImplIndex])->EachToManyMulticast(id, size, msg, pesize, pelist);
}

void NumDeposits(comID id, int num)
{
  UpdateImplTable(id);
  (CpvAccess(ImplTable)[id.srcpe][id.ImplIndex])->NumDeposits(id, num);
}

/*****************************************************
 * Entry Points
 ****************************************************/
void KRecvManyCombinedMsg(char *msg)
{
  comID id;
  CmiGrabBuffer((void **)&msg);
  memcpy(&id,(msg+CmiMsgHeaderSizeBytes+sizeof(int)), sizeof(comID));

  UpdateImplTable(id);
  (CpvAccess(ImplTable)[id.srcpe][id.ImplIndex])->RecvManyMsg(id, msg);
}

void KProcManyCombinedMsg(char *msg)
{
  comID id;
  CmiGrabBuffer((void **)&msg);
  memcpy(&id,(msg+CmiMsgHeaderSizeBytes+sizeof(int)), sizeof(comID));

  UpdateImplTable(id);
  (CpvAccess(ImplTable)[id.srcpe][id.ImplIndex])->ProcManyMsg(id, msg);
}

void KDummyEP(DummyMsg *m)
{
  comID id=m->id;
  UpdateImplTable(id);
  CpvAccess(ImplTable)[id.srcpe][id.ImplIndex]->DummyEP(m->id, m->magic, m->refno);
}

/****************************************************
 * Routines used by the router objects
 ***************************************************/
void KBcastSwitchMsg(comID id)
{
  SwitchMsg *m=(SwitchMsg *)CmiAlloc(sizeof(SwitchMsg));
  CmiSetHandler(m, CpvAccess(SwitchHandle));
  m->id=id;
  CmiSyncBroadcastAndFree(sizeof(SwitchMsg), m);
}

void KSwitchEP(SwitchMsg *m)
{
  //CmiPrintf("%d switchep called\n", CmiMyPe());
  comID id=m->id;
  UpdateImplTable(id);
} 

void KSendDummyMsg(comID id, int pe, int magic)
{
  DummyMsg *m=(DummyMsg *)CmiAlloc(sizeof(DummyMsg));
  CmiSetHandler(m, CpvAccess(DummyHandle));
  m->id=id;
  m->magic=magic;
  m->refno=KMyActiveRefno(id);
  CmiSyncSendAndFree(pe, sizeof(DummyMsg), m);
}

void KCsdEnqueue(void *m)
{
  CsdEnqueueLifo(m);
}

int KMyActiveRefno(comID id)
{
  //CmiPrintf("KMyActive calling update\n");
  return(CpvAccess(ImplTable)[id.srcpe][id.ImplIndex]->MyActiveIndex());
}

void KDoneEP(DummyMsg *m)
{
  comID id=m->id;
  CpvAccess(ImplTable)[id.srcpe][id.ImplIndex]->Done();
}

void KDone(comID id)
{
  DummyMsg *m=(DummyMsg *)CmiAlloc(sizeof(DummyMsg));
  m->id=id;
  CmiSetHandler(m, CpvAccess(KDoneHandle));
  CmiSyncSendAndFree(CmiMyPe(), sizeof(DummyMsg), m);
}

/****************************************************
 * Multicast Groups
 ***************************************************/
void KsendGmsg(comID id)
{
  GMsg *gmsg=(GMsg *)CmiAlloc(sizeof(GMsg));
  gmsg->id=id;
  CmiSetHandler(gmsg, CpvAccess(KGMsgHandle));
  //CmiSyncSendAndFree(CmiMyPe(),sizeof(GMsg), gmsg);
	KCsdEnqueue(gmsg);
}

void KGMsgHandler(GMsg *msg)
{
  //CmiPrintf("KGhandler called\n");
  comID id=msg->id;
  int npes, *pes;
  //CmiGroup grp=id.grp;
  //CmiPrintf("grppe=%d, grpindex=%d\n", grp.pe, grp.id);
  CmiLookupGroup((msg->id).grp, &npes, &pes);
  if (pes==0) {
  	CmiGrabBuffer((void **)&msg);
  	//CmiSyncSendAndFree(CmiMyPe(),sizeof(GMsg), msg);
	KCsdEnqueue(msg);
  }
  else {
  	(CpvAccess(ImplTable)[id.srcpe][id.ImplIndex])->GroupMap(npes, pes);
  }
}

