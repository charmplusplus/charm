#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "charm++.h"

#include "pgm.h"
#include "Pgm.def.h"

#include "counter.h"

extern "C" void CApplicationInit(void);
extern "C" void CApplicationDepositData(char *data);

#define TIME_INTERVAL 500
#ifndef WIN32
#include <sys/time.h>
#include <unistd.h>
#endif

main::main(CkArgMsg *m)
{ 
  int arg_index = 1;
  mainhandle=thishandle;
  PartialBoard *pMsg = new  PartialBoard ;

  if(m->argc<3)
    CmiAbort("Usage: pgm N grain\n");
  while ( m->argv[arg_index][0] == '+' )
    arg_index++;
  N = atoi(m->argv[arg_index]);
  arg_index++;
  grain = atoi(m->argv[arg_index]);
  pMsg->nextRow = 0;
  CProxy_queens::ckNew(pMsg);
  t0 = CkTimer();
  counterGroup = counterInit();
  CkStartQD(CkIndex_main::Quiescence1((DUMMYMSG *)0), &mainhandle);
}

void main::Quiescence1(DUMMYMSG *msg) 
{ 
  int numSolutions = CProxy_counter(counterGroup).ckLocalBranch()->getTotalCount();
  CkPrintf("There are %d Solutions to %d queens. Finish time=%f\n",
	  numSolutions, N, CkTimer()-t0);
  CkExit();
}

queens::queens(PartialBoard *m)
{ 
  int col,i, row;
  PartialBoard *newMsg;

  row = m->nextRow;  
  if (row == N) 
    solutionFound(m->Queens);
  else if ( (N-row) < grain) 
    seqQueens(m->Queens, row); 
  else
  {
    for (col = 0; col<N; col++) 
      if (consistent(m->Queens, row, col))
      {
	newMsg = new  PartialBoard;
	newMsg->nextRow = row + 1;
	for (i=0; i<N; i++)
	  newMsg->Queens[i] = m->Queens[i];
	newMsg->Queens[row] = col;
	CProxy_queens::ckNew(newMsg);  
      }
  }
  delete m;
  delete this;
}


void queens::solutionFound(int kqueens[])
{ 
  int row;
  char buf[800], tmp[800];
  CProxy_counter(counterGroup).ckLocalBranch()->increment();
  buf[0] = '\0';
  sprintf(tmp,"%d:", CkMyPe());
  strcat(buf,tmp);
  for (row=0; row<N; row++)
  { 
    sprintf(tmp, "(%d,%d) ", row, kqueens[row]);
    strcat(buf, tmp);
  }
  //  CkPrintf("%s\n", buf);
}

void queens::seqQueens(int kqueens[], int nextRow)
{ 
  int col;

  if (nextRow == N) 
  { solutionFound(kqueens);
    return; 
  }
  for (col = 0; col<N; col++) 
    if (consistent(kqueens, nextRow, col))
    {
      kqueens[nextRow] = col;
      seqQueens(kqueens, nextRow+1);
    }
}

int queens::consistent(int kqueens[], int lastRow, int col)
{ 
  /* check if the new queen is safe from each of the */
  /* previously placed queens */
  int x,y;

  for (x=0; x<lastRow; x++)
  { 
    y = kqueens[x];
    if ((y==col)  || ((lastRow-x) == (col-y)) || ((lastRow-x) == (y-col)) ) {
      return(0);
    }
  }
  return(1);
}
