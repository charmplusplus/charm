/**
   @addtogroup ConvComlibRouter
   @{
   @file
   @brief Grid (mesh) based router 
*/


#include "gridrouter.h"

#define gmap(pe) {if (gpes) pe=gpes[pe];}

/**The only communication op used. Modify this to use
 ** vector send */
#if CMK_COMLIB_USE_VECTORIZE
#define GRIDSENDFN(kid, u1, u2, knpe, kpelist, khndl, knextpe)  \
  	{int len;\
	PTvectorlist newmsg;\
        ComlibPrintf("Entering Gridsendfn\n");\
        newmsg=PeMesh->ExtractAndVectorize(kid, u1, knpe, kpelist);\
	if (newmsg) {\
	  CmiSetHandler(newmsg->msgs[0], khndl);\
          ComlibPrintf("Sending in Gridsendfn to %d\n",knextpe);\
          CmiSyncVectorSendAndFree(knextpe, -newmsg->count, newmsg->sizes, newmsg->msgs);\
        }\
	else {\
	  SendDummyMsg(kid, knextpe, u2);\
	}\
}
#else
#define GRIDSENDFN(kid, u1, u2, knpe, kpelist, khndl, knextpe)  \
  	{int len;\
	char *newmsg;\
        newmsg=PeMesh->ExtractAndPack(kid, u1, knpe, kpelist, &len);\
	if (newmsg) {\
	  CmiSetHandler(newmsg, khndl);\
          CmiSyncSendAndFree(knextpe, len, newmsg);\
        }\
	else {\
	  SendDummyMsg(kid, knextpe, u2);\
	}\
}
#endif

/****************************************************
 * Preallocated memory=P ints + MAXNUMMSGS msgstructs
 *****************************************************/
GridRouter::GridRouter(int n, int me, Strategy *parent) : Router(parent)
{
  //CmiPrintf("PE=%d me=%d NUMPES=%d\n", MyPe, me, n);
  
  NumPes=n;
  MyPe=me;
  gpes=NULL;
  COLLEN=ColLen(NumPes);
  LPMsgExpected = Expect(MyPe, NumPes);
  recvExpected = 0;

  myrow=MyPe/COLLEN;
  mycol=MyPe%COLLEN;  
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

  for(count = mycol; count < ROWLEN+mycol; count ++){
      int nextpe= myrep + count%ROWLEN;
      
      if (nextpe >= NumPes) {
          int new_row = mycol % (myrow+1);
          
          if(new_row >= myrow)
              new_row = 0;
          
          nextpe = COLLEN * new_row + count%ROWLEN;
      }
      
      if(nextpe == MyPe)
          continue;

      ComlibPrintf("Row[%d] = %d\n", MyPe, nextpe);

      rowVector[pos ++] = nextpe;
  }
  rvecSize = pos;

  pos = 0;
  for(count = myrow; count < COLLEN+myrow; count ++){
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
    
  delete [] growVector;
  delete [] gcolVector;

  CmiFree(onerow);
  CmiFree(rowVector);
  CmiFree(colVector);
}

void GridRouter :: InitVars()
{
  recvCount=0;
  LPMsgCount=0;
}

/*
void GridRouter::NumDeposits(comID, int num)
{
}
*/

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
  if(id.isAllToAll)
      PeMesh->InsertMsgs(1, &MyPe, size, msg);
  else
      PeMesh->InsertMsgs(numpes, destpes, size, msg);
  
  if (more) return;

  ComlibPrintf("All messages received %d %d %d\n", MyPe, COLLEN,id.isAllToAll);
  sendRow(id);
}

void GridRouter::EachToManyMulticastQ(comID id, CkQ<MessageHolder *> &msgq) {
   
    ComlibPrintf("Grid Router :: EachToManyQ\n");
 
    int count = 0;
    int length = msgq.length();
    if(id.isAllToAll) {
        for(count = 0; count < length; count ++) {
            MessageHolder *mhdl = msgq.deq();
            PeMesh->InsertMsgs(1, &MyPe, mhdl->size, mhdl->getMessage());
            delete mhdl;
        }
    }
    else {
        for(count = 0; count < length; count ++) {
            MessageHolder *mhdl = msgq.deq();
            PeMesh->InsertMsgs(mhdl->npes, mhdl->pelist, mhdl->size, 
                               mhdl->getMessage());
            delete mhdl;
        }
    }
    
    sendRow(id);
}


void GridRouter::sendRow(comID id) {

    int i=0;
    char *a2amsg = NULL;
    int a2a_len;
    if(id.isAllToAll) {
        ComlibPrintf("ALL to ALL flag set\n");
        
        a2amsg = PeMesh->ExtractAndPackAll(id, 0, &a2a_len);
        CmiSetHandler(a2amsg, CkpvAccess(RouterRecvHandle));
        CmiReference(a2amsg);
        CmiSyncListSendAndFree(rvecSize, growVector, a2a_len, a2amsg);      
        RecvManyMsg(id, a2amsg);
        return;
    }
    
    //Send the messages
    //int MYROW  =MyPe/COLLEN;
    //int MYCOL = MyPe%COLLEN;
    int myrep= myrow*COLLEN; 
    int maxlength = (NumPes - 1)/COLLEN + 1;
    
    for (int colcount = 0; colcount < rvecSize; ++colcount) {
        int nextpe = rowVector[colcount];
        i = nextpe % COLLEN;
        
        int length = maxlength;
        if((length - 1)* COLLEN + i >= NumPes)
            length --;
        
        for (int j = 0; j < length; j++) {
            onerow[j]=j * COLLEN + i;
        }
        
        ComlibPrintf("%d: before gmap sending to %d of column %d for %d procs,  %d,%d,%d,%d\n",
                     MyPe, nextpe, i, length, onerow[0], onerow[1], onerow[2], onerow[3]);
        
        gmap(nextpe);
        
        GRIDSENDFN(id, 0, 0,length, onerow, CkpvAccess(RouterRecvHandle), nextpe); 
        ComlibPrintf("%d:sending to %d of column %d\n", MyPe, nextpe, i);
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

  ComlibPrintf("%d recvcount=%d recvexpected = %d\n", MyPe, recvCount, recvExpected);

  if (recvCount == recvExpected) {
      
      char *a2amsg;
      int a2a_len;
      if(id.isAllToAll) {
          a2amsg = PeMesh1->ExtractAndPackAll(id, 1, &a2a_len);
          CmiSetHandler(a2amsg, CkpvAccess(RouterProcHandle));
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
          
          GRIDSENDFN(id, 0, 1, 1, pelist, CkpvAccess(RouterProcHandle), nextpe);
      }
      
      LocalProcMsg(id);
  }
}

void GridRouter::DummyEP(comID id, int magic)
{
  if (magic == 1) {
	//ComlibPrintf("%d dummy calling lp\n", MyPe);
	LocalProcMsg(id);
  }
  else {
	//ComlibPrintf("%d dummy calling recv\n", MyPe);
  	RecvManyMsg(id, NULL);
  }
}

void GridRouter:: ProcManyMsg(comID id, char *m)
{
    ComlibPrintf("%d: ProcManyMsg\n", MyPe);
    
    if(id.isAllToAll)
        PeMesh2->UnpackAndInsertAll(m, 1, &MyPe);
    else
        PeMesh->UnpackAndInsert(m);
    //ComlibPrintf("%d proc calling lp\n");
    
    LocalProcMsg(id);
}

void GridRouter:: LocalProcMsg(comID id)
{
    LPMsgCount++;
    PeMesh->ExtractAndDeliverLocalMsgs(MyPe, container);
    if(id.isAllToAll)
        PeMesh2->ExtractAndDeliverLocalMsgs(MyPe, container);
    
    ComlibPrintf("%d local procmsg called, %d, %d\n", MyPe, 
                 LPMsgCount, LPMsgExpected);
    if (LPMsgCount==LPMsgExpected) {
        PeMesh->Purge();
        if(id.isAllToAll)
            PeMesh2->Purge();
        
        InitVars();
        Done(id);
    }
}

Router * newgridobject(int n, int me, Strategy *strat)
{
  Router *obj=new GridRouter(n, me, strat);
  return(obj);
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
/*@}*/
