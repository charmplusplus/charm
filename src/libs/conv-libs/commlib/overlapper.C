/***************************************************
 * File : overlapper.C
 *
 * Author: Krishnan V
 *
 * The overlap/switch manager
 *************************************************/
#include <string.h>
#include <converse.h>
#include "commlib.h"
#include "commlib_pvt.h"
#include "overlapper.h"
#include "ovmacros.h"

#define SWITCHCYCLE 100

CpvExtern(int, SwitchHandle);
extern Router * GetStrategyObject(int, int, int );

Overlapper:: Overlapper(comID id)
{
  //CmiPrintf("%d ovl called with numMem=%d\n", CmiMyPe(), id.NumMembers);
  MyID=id;
  NoSwitch=1;
  gpes=NULL;
  gnpes=0;
  if (id.NumMembers > 0) {
  	NumPes=id.NumMembers;
  	MyPe=CmiMyPe();
  	routerObj=GetStrategyObject(NumPes, MyPe, id.ImplType);
  	routerObj->SetID(MyID);
  	Active=RecvActive=0;
  }
  else {
        gnpes=0;
	gpes=0;
	CmiLookupGroup(id.grp, &gnpes, &gpes);
	if (gpes==0) { 
		KsendGmsg(id);
		Active=RecvActive=1;
	}
	else {
		NumPes=gnpes;
		int gg;
		for (gg=0;gg<gnpes;gg++) { 
			if (gpes[gg]==CmiMyPe()) {
				MyPe=gg;
				break;
			}
		}
		//CmiPrintf("gg=%d, MyPe=%d, gnpes=%d\n",gg, MyPe, gnpes);
  		if (MyPe <0) CmiPrintf("%d Errorin mapping Pes\n", CmiMyPe());
  		routerObj=GetStrategyObject(NumPes, MyPe, id.ImplType);
  		routerObj->SetID(MyID);
		routerObj->SetMap(gpes);
  		Active=RecvActive=0;
	}
  }
  ActiveRefno=0;
  OBFirst=NULL;OBLast=NULL;
  ORFirst=NULL;ODFirst=NULL;
  OPFirst=NULL;
  SwitchDecision=0;
  MoreDeposits=1;
  DeleteFlag=0;
  ORFreeList=NULL;
  OBFreeList=NULL;
  OPFreeList=NULL;
  ODFreeList=NULL;
}

Overlapper:: ~Overlapper()
{
  delete routerObj;
  OverlapBuffer *otmp;
  OverlapDummyBuffer *dtmp;
  OverlapRecvBuffer *rtmp;
  OverlapProcBuffer *ptmp;
  while (OBFirst) {
	otmp=OBFirst;
	OBFirst=OBFirst->next;
	CmiFree(otmp);
  }
  while (ODFirst) {
	dtmp=ODFirst;
	ODFirst=ODFirst->next;
	CmiFree(dtmp);
  }
  while (ORFirst) {
	rtmp=ORFirst;
	ORFirst=ORFirst->next;
	CmiFree(rtmp);
  }
  while (OPFirst) {
	ptmp=OPFirst;
	OPFirst=OPFirst->next;
	CmiFree(ptmp);
  }
  while (OPFreeList) {
	ptmp=OPFreeList;
	OPFreeList=OPFreeList->next;
	CmiFree(ptmp);
  }
  while (ORFreeList) {
	rtmp=ORFreeList;
	ORFreeList=ORFreeList->next;
	CmiFree(rtmp);
  }
  while (OBFreeList) {
	otmp=OBFreeList;
	OBFreeList=OBFreeList->next;
	CmiFree(otmp);
  }
  while (ODFreeList) {
	dtmp=ODFreeList;
	ODFreeList=ODFreeList->next;
	CmiFree(dtmp);
  }
}

void Overlapper::GroupMap(int npes, int *pes)
{
  gnpes=npes;
  gpes=pes;
  NumPes=gnpes;
  MyPe=-1;
  for (int gg=0;gg<gnpes;gg++) { 
	if (gpes[gg]==CmiMyPe()) {
		//CmiPrintf("Groupmap %d to %d\n", CmiMyPe(), gg);
		MyPe=gg;
		break;
	}
  }
  if (MyPe <0) CmiPrintf("%d Errorin mapping Pes\n", CmiMyPe());
  routerObj=GetStrategyObject(NumPes, MyPe, MyID.ImplType);
  routerObj->SetID(MyID);
  routerObj->SetMap(gpes);
  Active=RecvActive=0;
  Done();
}
  	
  
void Overlapper::NumDeposits(comID id, int n)
{
  MoreDeposits=n;
  if (Active) {
	return;
  }
}

void Overlapper::EachToAllMulticast(comID id, int size, void *msg)
{
  int moredeps=1;
  if (!NoSwitch && (id.srcpe==CmiMyPe())) {
	CommAnalyzer(id);
  }

  if (Active) {
	//CmiPrintf("%d buffering multicast\n", CmiMyPe());
	//OverlapBuffer *ob=(OverlapBuffer*)CmiAlloc(sizeof(OverlapBuffer));
	OverlapBuffer *ob;
	OBAlloc(ob, sizeof(OverlapBuffer));
	ob->id=id;
	ob->npe=-1;
	ob->pelist=NULL;
	ob->msgsize=size;
	ob->msg=msg;
 	ob->more=MoreDeposits;
	ob->next=NULL;
	if (OBLast!=NULL) { 
		OBLast->next=ob;
	}
	OBLast=ob;
	if (OBFirst==NULL) OBFirst=ob;
	MoreDeposits--;
	if (!MoreDeposits) MoreDeposits=1;
  }
  else {
      if (!RecvActive) {
	ActiveRefno=(ActiveRefno+1)%1000;
        if (MyID.SwitchVal==ActiveRefno) SwitchStrategy();
      }
      RecvActive=1;

      MoreDeposits--;
      if (!MoreDeposits) {
	Active=1;
  	moredeps=0;
	MoreDeposits=1;
      }

      //CmiPrintf("%d Active Multicast refno %di MoreDeps=%d\n", CmiMyPe(), ActiveRefno, MoreDeposits);
      routerObj->EachToAllMulticast(id, size, msg, moredeps);
  }
}

void Overlapper::EachToManyMulticast(comID id, int size, void *msg, int npe, int *pelist)
{
  int moredeps=1;
  if (!NoSwitch && (id.srcpe==CmiMyPe())) {
	CommAnalyzer(id);
  }

  if (Active) {
	//CmiPrintf("%d buffering multicast\n", CmiMyPe());
	//OverlapBuffer *ob=(OverlapBuffer*)CmiAlloc(sizeof(OverlapBuffer));
	OverlapBuffer *ob;
	OBAlloc(ob, sizeof(OverlapBuffer));
	ob->id=id;
	ob->npe=npe;
	ob->pelist=(int *)CmiAlloc(npe*sizeof(int));
	memcpy((char *)(ob->pelist), (char *)pelist, npe*sizeof(int));
	ob->msgsize=size;
	ob->msg=msg;
	ob->more=MoreDeposits;
	ob->next=NULL;
	if (OBLast!=NULL) { 
		OBLast->next=ob;
	}
	OBLast=ob;
	if (OBFirst==NULL) OBFirst=ob;
	MoreDeposits--;
	if (!MoreDeposits) MoreDeposits=1;
  }
  else {
      if (!RecvActive) {
	ActiveRefno=(ActiveRefno+1)%1000;
        if (MyID.SwitchVal==ActiveRefno) SwitchStrategy();
      }
      RecvActive=1;

      MoreDeposits--;
      if (!MoreDeposits) { 
	Active=1;
  	moredeps=0;
	MoreDeposits=1;
	if (SwitchDecision)
		InsertSwitchMsgs(npe, pelist);
      }

      //CmiPrintf("%d Active Multicast refno %d More=%d\n", CmiMyPe(), ActiveRefno, MoreDeposits);
      routerObj->EachToManyMulticast(id, size, msg, npe, pelist, moredeps);
  }
}

void Overlapper :: RecvManyMsg(comID id, char *msg)
{
  int refno=*(int *)(msg+CmiMsgHeaderSizeBytes);
  if (RecvActive) {
    if (ActiveRefno != refno) {
	//CmiPrintf("%d buffering recv refno %d, Activerefno=%d\n", CmiMyPe(), refno, ActiveRefno);
	//OverlapRecvBuffer *ovr=(OverlapRecvBuffer *)CmiAlloc(sizeof(OverlapRecvBuffer));
	OverlapRecvBuffer *ovr;
	ORAlloc(ovr, sizeof(OverlapRecvBuffer));
	ovr->refno=refno;
	ovr->id=id;
	ovr->msg=msg;
	ovr->next=ORFirst;
	ORFirst=ovr;
    }
    else {
	routerObj->RecvManyMsg(id, msg);
    }
  }
  else {
	//ActiveRefno=(ActiveRefno+1)%1000;
	ActiveRefno=refno;
	RecvActive=1;
        if (MyID.SwitchVal==ActiveRefno) SwitchStrategy();
	routerObj->RecvManyMsg(id, msg);
  }
}
  
  
void Overlapper :: DummyEP(comID id, int magic, int refno)
{
  if (RecvActive) {
    if (ActiveRefno != refno) {
	//CmiPrintf("%d buffering dummy refno %d\n", CmiMyPe(), refno);
	//OverlapDummyBuffer *ovr=(OverlapDummyBuffer *)CmiAlloc(sizeof(OverlapDummyBuffer));
	OverlapDummyBuffer *ovr;
	ODAlloc(ovr, sizeof(OverlapDummyBuffer));
	ovr->refno=refno;
	ovr->id=id;
	ovr->magic=magic;
	ovr->next=ODFirst;
	ODFirst=ovr;
    }
    else {
	routerObj->DummyEP(id, magic);
    }
  }
  else {
	//ActiveRefno=(ActiveRefno+1)%1000;
	ActiveRefno=refno;
        if (MyID.SwitchVal==ActiveRefno) SwitchStrategy();
	
	RecvActive=1;
	routerObj->DummyEP(id, magic);
  }
}

void Overlapper :: ProcManyMsg(comID id, char *m)
{
  //CmiPrintf("%d Procmanymsg in overlapper called\n", CmiMyPe());
  int refno=*(int *)(m+CmiMsgHeaderSizeBytes);
  if (RecvActive) {
    if (ActiveRefno != refno) {
	//CmiPrintf("%d buffering proc msg refno %d\n", CmiMyPe(), refno);
	//OverlapProcBuffer *op=(OverlapProcBuffer *)CmiAlloc(sizeof(OverlapProcBuffer));
	OverlapProcBuffer *op;
	OPAlloc(op, sizeof(OverlapProcBuffer));
	op->refno=refno;
	op->id=id;
	op->msg=m;
	op->next=OPFirst;
	OPFirst=op;
    }
    else {
	routerObj->ProcManyMsg(id, m);
    }
  }
  else {
	//ActiveRefno=(ActiveRefno+1)%1000;
	ActiveRefno=refno;
        if (MyID.SwitchVal==ActiveRefno) SwitchStrategy();
	
	RecvActive=1;
	routerObj->ProcManyMsg(id, m);
  }
}

void Overlapper :: StartNext()
{
  if (Active || RecvActive) {
	CmiPrintf("ERROR: Operation Active\n");
	return;
  }
  if (OBFirst || ORFirst || ODFirst || OPFirst) {
	Active=1;
	RecvActive=1;
	ActiveRefno = (ActiveRefno +1)%1000;
        if (MyID.SwitchVal==ActiveRefno) SwitchStrategy();
  }
  else {
	if (DeleteFlag) delete this;
  }

  if (OBFirst) {
	OverlapBuffer *ob=OBFirst, *prev;
	//ActiveRefno=ob->refno;
	int moredeps=(ob->more);
	//routerObj->NumDeposits(ob->id, moredeps);
	if (SwitchDecision) 
		InsertSwitchMsgs(ob->npe, ob->pelist);
	while (moredeps && ob) {
		moredeps--;
		//CmiPrintf("%d Start multicast refno %d more=%d\n", CmiMyPe(), ActiveRefno, moredeps);
      		if (ob->npe == -1) {
			routerObj->EachToAllMulticast(ob->id, ob->msgsize, ob->msg, moredeps);
      		}
      		else {
			routerObj->EachToManyMulticast(ob->id, ob->msgsize, ob->msg, ob->npe, ob->pelist, moredeps); 
      		}
		prev=ob;
		ob=ob->next;
  		CmiFree(prev->pelist);
		//CmiFree(prev);
		OBFree(prev);
	}
	OBFirst=ob;
	if (OBFirst == NULL) OBLast=NULL;
  	if (moredeps) {
		//CmiPrintf("%d more deps=%d expected:%d deactivating...\n",CmiMyPe(), moredeps, ActiveRefno);
		Active=0;
	}
  }	
  if (ORFirst) {
	//CmiPrintf("%d Start recv refno %d\n", CmiMyPe(), ActiveRefno);
	OverlapRecvBuffer *ovr=ORFirst, *prev=NULL;
	while (ORFirst && (ORFirst->refno == ActiveRefno)) { 
		ovr=ORFirst;
		ORFirst=ovr->next;
		routerObj->RecvManyMsg(ovr->id, ovr->msg);
		//CmiFree(ovr);
		ORFree(ovr);
	}
 	prev=ORFirst;
	if (prev) {
	  ovr=ORFirst->next;
	  while (ovr) {
		if (ovr->refno == ActiveRefno) {
			prev->next=ovr->next;
			routerObj->RecvManyMsg(ovr->id, ovr->msg);
			//CmiFree(ovr);
			ORFree(ovr);
			ovr=prev->next;
		}
		else {
			prev=ovr;
			ovr=ovr->next;
		}
	  }
	}
  }

  if (ODFirst) {
	//CmiPrintf("%d Start dummy refno %d\n", CmiMyPe(), ActiveRefno);
	OverlapDummyBuffer *od, *prevd=NULL;
	while (ODFirst && (ODFirst->refno == ActiveRefno)) { 
		od=ODFirst;
		routerObj->DummyEP(od->id, od->magic);
		ODFirst=od->next;
		//CmiFree(od);
		ODFree(od);
	}
 	prevd=ODFirst;
	if (prevd) {
	  od=ODFirst->next;
	  while (od) {
		if (od->refno == ActiveRefno) {
			prevd->next=od->next;
			routerObj->DummyEP(od->id, od->magic);
			//CmiFree(od);
			ODFree(od);
			od=prevd->next;
		}
		else {
			prevd=od;
			od=od->next;
		}
	  }
	}//End of if prevd
  }

  if (OPFirst) {
	//CmiPrintf("%d Start Proc refno %d\n", CmiMyPe(), ActiveRefno);
	OverlapProcBuffer *op=OPFirst, *prev=NULL;
	while (OPFirst && (OPFirst->refno == ActiveRefno)) { 
		op=OPFirst;
		OPFirst=op->next;
		routerObj->ProcManyMsg(op->id, op->msg);
		//CmiFree(op);
		OPFree(op);
	}
 	prev=OPFirst;
	if (prev) {
	  op=OPFirst->next;
	  while (op) {
		if (op->refno == ActiveRefno) {
			prev->next=op->next;
			routerObj->ProcManyMsg(op->id, op->msg);
			//CmiFree(op);
			OPFree(op);
			op=prev->next;
		}
		else {
			prev=op;
			op=op->next;
		}
	  }
	}
  }
  //CmiPrintf("StartNext done\n");
}


void Overlapper::Done()
{
  Active=0;
  RecvActive=0;
  //CmiPrintf("%d DONE refno %d.......................\n", CmiMyPe(), ActiveRefno);
  StartNext();
}
  
void Overlapper::SetID(comID id)
{
  MyID=id;
  routerObj->SetID(MyID);
}

void Overlapper::SwitchStrategy()
{
  delete routerObj;
  routerObj=GetStrategyObject(MyID.NumMembers, CmiMyPe(), MyID.ImplType);
  MyID.SwitchVal=-1;
  routerObj->SetID(MyID);
  //CmiPrintf("%d Switching Strategy to %d refno=%d\n", CmiMyPe(), MyID.ImplType, ActiveRefno);
  SwitchDecision=0;
}

/* This needs to be written. This imparts intelligence to the switch
 * manager */
void Overlapper::CommAnalyzer(comID)
{
  static int i=0;
  i++;
  //CmiPrintf("i=%d\n", i);
  if ((i % SWITCHCYCLE)==1) {
  	//CmiPrintf("%d com to set switchval\n", CmiMyPe());
  	MyID.SwitchVal=3;
  	MyID.ImplType=3;
  	routerObj->SetID(MyID);
  	SwitchDecision=1;
  }
}

void Overlapper :: InsertSwitchMsgs(int npe, int *pelist)
{
  SwitchMsg *sm=(SwitchMsg *)CmiAlloc(sizeof(SwitchMsg));
  int *templist=(int *)CmiAlloc(CmiNumPes()*sizeof(int));
  int *newlist=(int *)CmiAlloc((CmiNumPes()-npe)*sizeof(int));
  int i, j=0;

  for (i=0;i<CmiNumPes();i++) templist[i]=0;
  for (i=0;i<npe;i++) templist[pelist[i]]=1;
  for (i=0;i<CmiNumPes();i++) 
	if (templist[i]==0) newlist[j++]=i;

  sm->id=MyID;
  CmiSetHandler(sm, CpvAccess(SwitchHandle));
  if (j)
  	routerObj->EachToManyMulticast(MyID, sizeof(SwitchMsg), sm, j, newlist, 1);
  else
  	routerObj->EachToManyMulticast(MyID, 0, NULL, j, newlist, 1);

  CmiFree(templist);
  CmiFree(newlist);

  SwitchDecision=0;
}

void Overlapper::DeleteInstance()
{
  if (!Active && !RecvActive) {
	delete this;
	return;
  }
  else DeleteFlag=1;
}
  
