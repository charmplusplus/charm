#include "Pgm.decl.h"
// must include this to allow our message types (e.g. PartialBoard) to 
// inherit from system-generated ones.


#define Nmax 20

int grain, N;
CkGroupID counterGroup;
CkChareID mainhandle;

class PartialBoard : public CMessage_PartialBoard { 
public:
  int nextRow;
  int Queens[Nmax];  
};

class DUMMYMSG : public CMessage_DUMMYMSG {
public:
};

class main : public Chare {
private:
  double t0; /* starting time */
  
public:
  main(CkMigrateMessage *m) {}
  main(CkArgMsg *m);
  void Quiescence1(DUMMYMSG *msg);
};

class queens : public Chare {
private:
  void seqQueens(int queens[], int nextRow);
  int consistent(int queens[], int lastRow, int col);
  void solutionFound(int queens[]);

public:
  queens(CkMigrateMessage *m) {}
  queens(PartialBoard *m);
};
