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
  //ComlibPrintf("PE=%d me=%d NUMPES=%d\n", CmiMyPe(), me, n);
  
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
  //ComlibPrintf("%d LPMsgExpected=%d\n", MyPe, LPMsgExpected);

  PeMesh = new PeTable(/*CmiNumPes()*/NumPes);

  onerow=(int *)CmiAlloc(ROWLEN*sizeof(int));
  
  InitVars();
  //  ComlibPrintf("%d:%d:COLLEN=%d, ROWLEN=%d, recvexpected=%d\n", CmiMyPe(), MyPe, COLLEN, ROWLEN, recvExpected);
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

  ComlibPrintf("All messages received %d %d\n", CmiMyPe(), COLLEN);

  //Send the messages
  for (i=0;i<COLLEN;i++) {

      //    ComlibPrintf("ROWLEN = %d, COLLEN =%d\n", ROWLEN, COLLEN);

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
        //      ComlibPrintf("%d calling recv directly refno=%d\n", MyPe, KMyActiveRefno(MyID));
      RecvManyMsg(id, NULL);
      continue;
    }
    
    //    ComlibPrintf("nummappedpes = %d, NumPes = %d, nextrowrep = %d, nextpe = %d, mype = %d\n", nummappedpes, NumPes, nextrowrep,  nextpe, MyPe);

    gmap(nextpe);
    ComlibPrintf("sending to column %d in %d\n", i, CmiMyPe());
    GRIDSENDFN(MyID, 0, 0, rowlength, onerow, CpvAccess(RecvHandle), nextpe); 
  }
}

void GridRouter::RecvManyMsg(comID id, char *msg)
{
    //  ComlibPrintf("%d recvcount=%d recvexpected = %d refno=%d\n", MyPe, recvCount, recvExpected, KMyActiveRefno(MyID));
  if (msg)
    PeMesh->UnpackAndInsert(msg);

  recvCount++;
  if (recvCount == recvExpected) {

    for (int i=0;i<ROWLEN;i++) {
      int myrow=MyPe/COLLEN;
      int myrep=myrow*ROWLEN;
      int nextpe=myrep+i;
      
      ComlibPrintf("sending message %d %d %d\n", nextpe, NumPes, MyPe);

      if (nextpe >= NumPes || nextpe==MyPe) continue;

      int gnextpe = nextpe;
      int *pelist=&gnextpe;

      ComlibPrintf("Before gmap %d\n", nextpe);

      gmap(nextpe);

      ComlibPrintf("After gmap %d\n", nextpe);

      GRIDSENDFN(MyID, 0, 1, 1, pelist, CpvAccess(ProcHandle), nextpe);
    }
    LocalProcMsg();
  }
}

void GridRouter::DummyEP(comID id, int magic)
{
  if (magic == 1) {
	//ComlibPrintf("%d dummy calling lp\n", MyPe);
	LocalProcMsg();
  }
  else {
	//ComlibPrintf("%d dummy calling recv\n", MyPe);
  	RecvManyMsg(id, NULL);
  }
}

void GridRouter:: ProcManyMsg(comID, char *m)
{
  PeMesh->UnpackAndInsert(m);
  //ComlibPrintf("%d proc calling lp\n");
  LocalProcMsg();
}

void GridRouter:: LocalProcMsg()
{
    //  ComlibPrintf("%d local procmsg called\n", MyPe);

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

