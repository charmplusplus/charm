/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/************************************************************
 * File : gridrouter.C
 *
 * Author : Krishnan V.
 *
 * Grid (mesh) based router
 ***********************************************************/
#include "gridrouter.h"
#define NULL 0

#define gmap(pe) {if (gpes) pe=gpes[pe];}

/**The only communication op used. Modify this to use
 ** vector send */
#define GRIDSENDFN(kid, u1, u2, knpe, kpelist, khndl, knextpe)  \
  	{int len;\
	char *newmsg;\
 	newmsg=PeMesh->ExtractAndPack(kid, u1, knpe, kpelist, &len);\
	if (newmsg) {\
	  CmiSetHandler(newmsg, khndl);\
  	  CmiSyncSendAndFree(knextpe, len, newmsg);\
	}\
	else {\
	  KSendDummyMsg(kid, knextpe, u2);\
	}\
}

/****************************************************
 * Preallocated memory=P ints + MAXNUMMSGS msgstructs
 *****************************************************/
GridRouter::GridRouter(int n, int me)
{
  NumPes=n;
  MyPe=me;
  gpes=NULL;
  COLLEN=ColLen(NumPes);
  recvExpected=0;
  int myrow=MyPe/COLLEN;
  int myrep=myrow*COLLEN;
  int numunmappedpes=myrep+ROWLEN-NumPes;
  int nummappedpes=ROWLEN;
  if (numunmappedpes >0) {
	nummappedpes=NumPes-myrep;
	int i=NumPes+MyPe-myrep;
	while (i<myrep+ROWLEN) {
		recvExpected+=Expect(i, NumPes);
		i+=nummappedpes;
	}
  }

  recvExpected += Expect(MyPe, NumPes);
  LPMsgExpected=nummappedpes;
  //CmiPrintf("%d LPMsgExpected=%d\n", MyPe, LPMsgExpected);

  PeMesh = new PeTable(NumPes);

  onerow=(int *)CmiAlloc(ROWLEN*sizeof(int));
  
  InitVars();
 // CmiPrintf("%d:COLLEN=%d, recvexpected=%d\n", MyPe, COLLEN, recvExpected);
}

GridRouter::~GridRouter()
{
  delete PeMesh;
  CmiFree(onerow);
}

void GridRouter :: InitVars()
{
  recvCount=0;
  LPMsgCount=0;
}
void GridRouter::NumDeposits(comID, int num)
{
}

void GridRouter::EachToAllMulticast(comID id, int size, void *msg, int more)
{
  int npe=NumPes;
  int * destpes=(int *)CmiAlloc(sizeof(int)*npe);
  for (int i=0;i<npe;i++) destpes[i]=i;
  EachToManyMulticast(id, size, msg, npe, destpes, more);
}

void GridRouter::EachToManyMulticast(comID id, int size, void *msg, int numpes, int *destpes, int more)
{
  int i;

  //Buffer the message
  if (size) {
  	PeMesh->InsertMsgs(numpes, destpes, size, msg);
  }

  if (more) return;

  //Send the messages
  for (i=0;i<COLLEN;i++) {
    int MYROW=MyPe/ROWLEN;
    int nextrowrep=i*ROWLEN;
    int myrep=MYROW*ROWLEN;
    int nextpe=MyPe-myrep+nextrowrep;
    int nummappedpes=NumPes-nextrowrep;
    if (nummappedpes <= 0) continue;
    if (nextpe >= NumPes) {
    int mm=(nextpe-NumPes) % nummappedpes;
    nextpe=nextrowrep+mm;
    }

    int rowlength=ROWLEN;
    if (ROWLEN > nummappedpes) rowlength=nummappedpes;
    for (int j=0;j<rowlength;j++) {
	onerow[j]=nextrowrep+j;
    }
    if (nextpe == MyPe) {
	//CmiPrintf("%d calling recv directly refno=%d\n", MyPe, KMyActiveRefno(MyID));
  	RecvManyMsg(id, NULL);
	continue;
    }
    gmap(nextpe);
    GRIDSENDFN(MyID, 0, 0, rowlength, onerow, CpvAccess(RecvHandle), nextpe); 
  }
}

void GridRouter::RecvManyMsg(comID id, char *msg)
{
  //CmiPrintf("%d recvcount=%d refno=%d\n", MyPe,recvCount, KMyActiveRefno(MyID));
  if (msg)
  	PeMesh->UnpackAndInsert(msg);

  recvCount++;
  if (recvCount == recvExpected) {
    for (int i=0;i<ROWLEN;i++) {
      int myrow=MyPe/COLLEN;
      int myrep=myrow*ROWLEN;
      int nextpe=myrep+i;
      if (nextpe >= NumPes || nextpe==MyPe) continue;
      gmap(nextpe);
      int *pelist=&nextpe;
      GRIDSENDFN(MyID, 0, 1, 1, pelist, CpvAccess(ProcHandle), nextpe);
    }
    LocalProcMsg();
  }
}

void GridRouter::DummyEP(comID id, int magic)
{
  if (magic == 1) {
	//CmiPrintf("%d dummy calling lp\n", MyPe);
	LocalProcMsg();
  }
  else {
	//CmiPrintf("%d dummy calling recv\n", MyPe);
  	RecvManyMsg(id, NULL);
  }
}

void GridRouter:: ProcManyMsg(comID, char *m)
{
  PeMesh->UnpackAndInsert(m);
  //CmiPrintf("%d proc calling lp\n");
  LocalProcMsg();
}

void GridRouter:: LocalProcMsg()
{
  //CmiPrintf("%d local procmsg called\n", MyPe);

  LPMsgCount++;
  PeMesh->ExtractAndDeliverLocalMsgs(MyPe);

  if (LPMsgCount==LPMsgExpected) {
	PeMesh->Purge();
	InitVars();
	KDone(MyID);
  }

}

Router * newgridobject(int n, int me)
{
  Router *obj=new GridRouter(n, me);
  return(obj);
}

void GridRouter::SetID(comID id)
{
  MyID=id;
}

void GridRouter :: SetMap(int *pes)
{
  gpes=pes;
}
