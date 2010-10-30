#include "queens.h"

readonly<CkGroupID> queens_counterGroup;

void queens_init(void)
{
  queens_DMSG *dmsg = new queens_DMSG;
  dmsg->counterGroup = queens_counterGroup; 
  CProxy_queens_main::ckNew(dmsg, 0);
}

void queens_moduleinit(void)
{
  queens_counterGroup = CProxy_queens_counter::ckNew(); 
}

queens_main::queens_main(queens_DMSG * dmsg)
{ 
  // CkPrintf("[%d] queens_main created\n", CkMyPe());
  counterGroup = dmsg->counterGroup;
  queens_PartialBoard *pMsg = new queens_PartialBoard ;

  pMsg->nextRow = 0;
  pMsg->counterGroup = counterGroup;
  CProxy_queens_queens::ckNew(pMsg, NULL, CK_PE_ANY);

  CkStartQD(CkIndex_queens_main::Quiescence1((CkQdMsg *)0), 
            &thishandle);
}

void 
queens_main::Quiescence1(CkQdMsg *msg) 
{ 
  // CkPrintf("QD reached\n");
  int ns = CProxy_queens_counter(counterGroup).ckLocalBranch()->getTotalCount();

  if (ns != EXPECTNUM){
    CkError("queens: expected solutions = %d, got=%d\n", EXPECTNUM, ns);
    CkAbort("queens: unexpected number of solutions\n");
  }
  delete msg;
  megatest_finish();
}

queens_queens::queens_queens(queens_PartialBoard *m)
{ 
  int col,i, row;
  queens_PartialBoard *newMsg;
  
  row = m->nextRow;  
  counterGroup = m->counterGroup;
  if (row == NUMQUEENS) 
    solutionFound();
  else if ( (NUMQUEENS - row) < Q_GRAIN) 
    seqQueens(m->Queens, row); 
  else
  {
    for (col = 0; col<NUMQUEENS; col++) 
      if (consistent(m->Queens, row, col))
      {
	newMsg = new  queens_PartialBoard;
	newMsg->nextRow = row + 1;
	newMsg->counterGroup = counterGroup;
	for (i=0; i< NUMQUEENS; i++)
	  newMsg->Queens[i] = m->Queens[i];
	newMsg->Queens[row] = col;
    	CProxy_queens_queens::ckNew(newMsg, NULL, CK_PE_ANY);
      }
  }
  delete m;
  delete this;
}


void 
queens_queens::solutionFound(void)
{ 
  CProxy_queens_counter(counterGroup).ckLocalBranch()->increment();
}

void 
queens_queens::seqQueens(int kqueens[], int nextRow)
{ 
  int col;

  if (nextRow == NUMQUEENS) { 
    solutionFound();
    return; 
  }
  for (col = 0; col < NUMQUEENS; col++) 
    if (consistent(kqueens, nextRow, col)) {
      kqueens[nextRow] = col;
      seqQueens(kqueens, nextRow+1);
    }
}

int 
queens_queens::consistent(int kqueens[], int lastRow, int col)
{ 
  for (int x=0; x<lastRow; x++)
  { 
    int y = kqueens[x];
    if ((y==col)  || ((lastRow-x) == (col-y)) || ((lastRow-x) == (y-col)) ) {
      return(0);
    }
  }
  return(1);
}

queens_counter::queens_counter(void)
{ 
  myCount = 0;
  waitFor = CkNumPes(); // wait for all processors to report
  threadId = 0;
}

void 
queens_counter::increment(void)
{
  myCount++;
}

void 
queens_counter::sendCounts(void)
{
  // CkPrintf("[%d] sending counts\n", CkMyPe());
  CProxy_queens_counter grp(thisgroup);
  grp[0].childCount(new queens_countMsg(myCount));
  myCount = 0;
}

void 
queens_counter::childCount(queens_countMsg *m)
{
  totalCount += m->count;
  delete m;
  waitFor--;
  // CkPrintf("[%d] received count. Remaining = %d\n", CkMyPe(), waitFor);
  if (waitFor == 0) {
    CthAwaken(threadId);
    waitFor = CkNumPes();
  }
}

int 
queens_counter::getTotalCount(void)
{
  if(CkMyPe() != 0) {
    CkAbort("queens: queens_main not created on processor 0 ?\n");
    megatest_finish();
  }
  totalCount = 0;
  CProxy_queens_counter grp(thisgroup);
  grp.sendCounts();
  threadId = CthSelf(); 
  // CkPrintf("[%d] suspending\n", CkMyPe());
  CthSuspend();
  // CkPrintf("[%d] awakened\n", CkMyPe());
  threadId = 0;
  return totalCount;
}

MEGATEST_REGISTER_TEST(queens,"jackie",0)
#include "queens.def.h"
