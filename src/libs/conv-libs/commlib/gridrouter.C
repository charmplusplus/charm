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
#include "globals.h"

//#define NULL 0

#define PERSISTENT_BUFSIZE 65536

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
  //CmiPrintf("PE=%d me=%d NUMPES=%d\n", MyPe, me, n);
  
  NumPes=n;
  MyPe=me;
  gpes=NULL;
  COLLEN=ColLen(NumPes);
  LPMsgExpected = Expect(MyPe, NumPes);
  recvExpected = 0;

  int myrow=MyPe/COLLEN;
  int mycol=MyPe%COLLEN;  
  int lastrow = (NumPes - 1)/COLLEN;
  
  if(myrow < lastrow) 
      recvExpected = ROWLEN;
  else
      recvExpected = (NumPes - 1)%ROWLEN + 1;

  if(lastrow * COLLEN + mycol > NumPes - 1) {
      //We have a hole in the lastrow
      if(lastrow * COLLEN + myrow <= NumPes - 1) 
          //We have a processor which wants to send data to that hole
          recvExpected ++;
      
      if((myrow == 0) && (NumPes == ROWLEN*(COLLEN-1) - 1))
          //Special case with one hole only
	  recvExpected ++;      
  }
  
  ComlibPrintf("%d LPMsgExpected=%d\n", MyPe, LPMsgExpected);

  PeMesh = new PeTable(NumPes);

  onerow=(int *)CmiAlloc(ROWLEN*sizeof(int));
  
  InitVars();
  ComlibPrintf("%d:COLLEN=%d, ROWLEN=%d, recvexpected=%d\n", MyPe, COLLEN, ROWLEN, recvExpected);

#if CMK_PERSISTENT_COMM
  rowHandleArray = new PersistentHandle[COLLEN];
  rowHandleArrayEven = new PersistentHandle[COLLEN];
  columnHandleArray = new PersistentHandle[COLLEN];
  columnHandleArrayEven = new PersistentHandle[COLLEN];

  //handles for all the same column elements
  int pcount = 0;
  for (pcount = 0; pcount < COLLEN; pcount ++) {
    int dest = pcount *COLLEN + mycol;

    if(dest < NumPes) {
        ComlibPrintf("%d:Creating Persistent Buffer of size %d at %d\n", MyPe,
                     PERSISTENT_BUFSIZE, dest);
        gmap(dest);
        columnHandleArray[pcount] = CmiCreatePersistent(dest, 
                                                        PERSISTENT_BUFSIZE);
        ComlibPrintf("%d:Creating Even Persistent Buffer of size %d at %d\n",
                     MyPe, PERSISTENT_BUFSIZE, dest);
        columnHandleArrayEven[pcount] = CmiCreatePersistent
            (dest, PERSISTENT_BUFSIZE);
    }
    else
        columnHandleArray[pcount] = NULL;
  }

  //handles for all same row elements
  for (pcount = 0; pcount < COLLEN; pcount++) {
    int dest = myrow *COLLEN + pcount;

    if(dest >= NumPes){
        dest = COLLEN * (mycol % myrow) + pcount;
    }
    
    if(dest < NumPes && dest != MyPe){
      ComlibPrintf("[%d] Creating Persistent Buffer of size %d at %d\n", MyPe,
		   PERSISTENT_BUFSIZE, dest);
      gmap(dest);
      rowHandleArray[pcount] = CmiCreatePersistent(dest, PERSISTENT_BUFSIZE);
      ComlibPrintf("[%d] Creating Even Persistent Buffer of size %d at %d\n",
                   MyPe, PERSISTENT_BUFSIZE, dest);
      rowHandleArrayEven[pcount] = CmiCreatePersistent(dest, 
                                                       PERSISTENT_BUFSIZE);
    }
    else 
        rowHandleArray[pcount] = NULL;
  }

  ComlibPrintf("After Initializing Persistent Buffers\n");
#endif
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
  static int step = 0;

  //Buffer the message
  if (size) {
  	PeMesh->InsertMsgs(numpes, destpes, size, msg);
  }

  if (more) return;

  ComlibPrintf("All messages received %d %d\n", MyPe, COLLEN);

  step ++;

  //Send the messages
  int MYROW=MyPe/COLLEN;
  int MYCOL = MyPe%COLLEN;
  int myrep=MYROW*COLLEN;
  
  for (int colcount= MYCOL; colcount < ROWLEN + MYCOL; colcount++) {
      i = colcount % ROWLEN;
      int nextpe= myrep + i;

      if (nextpe >= NumPes) {
          //Previously hole assigned to elements in the same row as nextpe
          // Now they are spread across the grid in the same column as nextpe
          nextpe = COLLEN * (MYCOL % MYROW) + i;
      }

      int length = (NumPes - 1)/COLLEN + 1;
      if((length - 1)* COLLEN + i >= NumPes)
          length --;
      
      for (int j=0;j<length;j++) {
          onerow[j]=j * COLLEN + i;
      }
      
      if (nextpe == MyPe) {
          RecvManyMsg(id, NULL);
          continue;
      }
    
      gmap(nextpe);
      ComlibPrintf("%d:sending to %d of column %d\n", MyPe, nextpe, i);

#if CMK_PERSISTENT_COMM
      if(step % 2 == 1)
          CmiUsePersistentHandle(&rowHandleArray[i], 1);
      else
          CmiUsePersistentHandle(&rowHandleArrayEven[i], 1);
#endif          

      GRIDSENDFN(MyID, 0, 0, length, onerow, CpvAccess(RecvHandle), nextpe); 
      
#if CMK_PERSISTENT_COMM
      CmiUsePersistentHandle(NULL, 0);
#endif          
  }
}

void GridRouter::RecvManyMsg(comID id, char *msg)
{
  static int step = 0;
  if (msg)
    PeMesh->UnpackAndInsert(msg);

  recvCount++;
  if (recvCount == recvExpected) {
    step ++;
    ComlibPrintf("%d recvcount=%d recvexpected = %d refno=%d\n", MyPe, recvCount, recvExpected, KMyActiveRefno(MyID));

    int myrow=MyPe/COLLEN;
    int mycol=MyPe%COLLEN;
    
    for (int rowcount= myrow; rowcount < COLLEN + myrow; rowcount++) {
      int i = rowcount % COLLEN;
      int nextrowrep=i*COLLEN;
      int nextpe=nextrowrep+mycol;
      
      ComlibPrintf("sending message %d %d %d\n", nextpe, NumPes, MyPe);

      if (nextpe >= NumPes || nextpe==MyPe) continue;

      int gnextpe = nextpe;
      int *pelist=&gnextpe;

      ComlibPrintf("Before gmap %d\n", nextpe);

      gmap(nextpe);

      ComlibPrintf("After gmap %d\n", nextpe);

#if CMK_PERSISTENT_COMM
      if(step % 2 == 1)
          CmiUsePersistentHandle(&columnHandleArray[i], 1);
      else
          CmiUsePersistentHandle(&columnHandleArrayEven[i], 1);
#endif          
      
      GRIDSENDFN(MyID, 0, 1, 1, pelist, CpvAccess(ProcHandle), nextpe);

#if CMK_PERSISTENT_COMM
      CmiUsePersistentHandle(NULL, 0);
#endif          
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

  LPMsgCount++;
  PeMesh->ExtractAndDeliverLocalMsgs(MyPe);

  if (LPMsgCount==LPMsgExpected) {
    //    CkPrintf("%d local procmsg called\n", MyPe);
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

