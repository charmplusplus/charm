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
  PeMesh1 = new PeTable(NumPes);
  PeMesh2 = new PeTable(NumPes);

  onerow=(int *)CmiAlloc(ROWLEN*sizeof(int));

  rowVector = (int *)CmiAlloc(ROWLEN*sizeof(int));
  colVector = (int *)CmiAlloc(COLLEN*sizeof(int));

  int myrep=myrow*COLLEN;
  int count = 0;
  int pos = 0;

  for(count = myrow; count < ROWLEN+myrow; count ++){
      int nextpe= myrep + count%ROWLEN;
      
      if (nextpe >= NumPes) {
          int new_row = mycol % (myrow+1);
          
          if(new_row >= myrow)
              new_row = 0;
          
          nextpe = COLLEN * new_row + count;
      }
      
      if(nextpe == MyPe)
          continue;

      rowVector[pos ++] = nextpe;
  }
  rvecSize = pos;

  pos = 0;
  for(count = mycol; count < COLLEN+mycol; count ++){
      int nextrowrep = (count % COLLEN) *COLLEN;
      int nextpe = nextrowrep+mycol;
      
      if(nextpe < NumPes && nextpe != MyPe)
          colVector[pos ++] = nextpe;
  }
  
  cvecSize = pos;

  growVector = new int[rvecSize];
  gcolVector = new int[cvecSize];

  for(count = 0; count < rvecSize; count ++)
      growVector[count] = rowVector[count];
  
  for(count = 0; count < cvecSize; count ++)
      gcolVector[count] = colVector[count];
  

  InitVars();
  ComlibPrintf("%d:COLLEN=%d, ROWLEN=%d, recvexpected=%d\n", MyPe, COLLEN, ROWLEN, recvExpected);
}

GridRouter::~GridRouter()
{
  delete PeMesh;
  delete PeMesh1;
  delete PeMesh2;
    
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

extern void CmiReference(void *blk);

void GridRouter::EachToManyMulticast(comID id, int size, void *msg, int numpes, int *destpes, int more)
{
  int i=0;
  
  if(id.isAllToAll)
      PeMesh->InsertMsgs(1, &MyPe, size, msg);
  else
      PeMesh->InsertMsgs(numpes, destpes, size, msg);
  
  if (more) return;

  ComlibPrintf("All messages received %d %d %d\n", MyPe, COLLEN,id.isAllToAll);

  char *a2amsg = NULL;
  int a2a_len;
  if(id.isAllToAll) {
      a2amsg = PeMesh->ExtractAndPackAll(id, 0, &a2a_len);
      CmiSetHandler(a2amsg, CpvAccess(RecvHandle));
      CmiReference(a2amsg);
      CmiSyncListSendAndFree(rvecSize, growVector, a2a_len, a2amsg);      
      RecvManyMsg(id, a2amsg);
      return;
  }

  //Send the messages
  int MYROW=MyPe/COLLEN;
  int MYCOL = MyPe%COLLEN;
  int myrep=MYROW*COLLEN;
  
  for (int colcount = 0; colcount < rvecSize; ++colcount) {
      int nextpe = rowVector[colcount];
      i = nextpe % COLLEN;
      
      int length = (NumPes - 1)/COLLEN + 1;
      if((length - 1)* COLLEN + i >= NumPes)
          length --;
      
      for (int j = 0; j < length; j++) {
          onerow[j]=j * COLLEN + i;
      }
      
      ComlibPrintf("%d: before gmap sending to %d of column %d\n",
                   MyPe, nextpe, i);
      gmap(nextpe);
      ComlibPrintf("%d:sending to %d of column %d\n", MyPe, nextpe, i);
      
      GRIDSENDFN(MyID, 0, 0, length, onerow, CpvAccess(RecvHandle), nextpe); 
  }
  RecvManyMsg(id, NULL);
}

void GridRouter::RecvManyMsg(comID id, char *msg)
{
  if (msg) {  
      if(id.isAllToAll)
          PeMesh1->UnpackAndInsertAll(msg, 1, &MyPe);
      else
          PeMesh->UnpackAndInsert(msg);
  }

  recvCount++;
  if (recvCount == recvExpected) {
      ComlibPrintf("%d recvcount=%d recvexpected = %d refno=%d\n", MyPe, recvCount, recvExpected, KMyActiveRefno(MyID));
      
      int myrow=MyPe/COLLEN;
      int mycol=MyPe%COLLEN;
      
      char *a2amsg;
      int a2a_len;
      if(id.isAllToAll) {
          a2amsg = PeMesh1->ExtractAndPackAll(id, 1, &a2a_len);
          CmiSetHandler(a2amsg, CpvAccess(ProcHandle));
          CmiReference(a2amsg);
          CmiSyncListSendAndFree(cvecSize, gcolVector, a2a_len, a2amsg);   
          ProcManyMsg(id, a2amsg);
          return;
      }

      for (int rowcount=0; rowcount < cvecSize; rowcount++) {
          int nextpe = colVector[rowcount];
                    
          int gnextpe = nextpe;
          int *pelist=&gnextpe;
          
          ComlibPrintf("Before gmap %d\n", nextpe);
          
          gmap(nextpe);

          ComlibPrintf("After gmap %d\n", nextpe);
          
          ComlibPrintf("%d:sending message to %d of row %d\n", MyPe, nextpe, 
                       rowcount);
          
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

void GridRouter:: ProcManyMsg(comID id, char *m)
{
    if(id.isAllToAll)
        PeMesh2->UnpackAndInsertAll(m, 1, &MyPe);
    else
        PeMesh->UnpackAndInsert(m);
    //ComlibPrintf("%d proc calling lp\n");
    
    LocalProcMsg();
}

void GridRouter:: LocalProcMsg()
{
    LPMsgCount++;
    PeMesh->ExtractAndDeliverLocalMsgs(MyPe);
    PeMesh2->ExtractAndDeliverLocalMsgs(MyPe);
    
    ComlibPrintf("%d local procmsg called\n", MyPe);
    if (LPMsgCount==LPMsgExpected) {
        PeMesh->Purge();
        PeMesh2->Purge();
        
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
  
  if(!gpes)
      return;

  int count = 0;

  for(count = 0; count < rvecSize; count ++)
      growVector[count] = gpes[rowVector[count]];
  
  for(count = 0; count < cvecSize; count ++)
      gcolVector[count] = gpes[colVector[count]];
}

