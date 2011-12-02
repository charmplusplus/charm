#ifndef _QUEENS_H
#define _QUEENS_H

#include "queens.decl.h"
#include "megatest.h"

#define NUMQUEENS  9
#define Q_GRAIN 6
#define EXPECTNUM  352


extern readonly<CkGroupID> queens_counterGroup;

class queens_PartialBoard : public CMessage_queens_PartialBoard { 
 public:
  int nextRow;
  int Queens[NUMQUEENS];
  CkGroupID counterGroup;
};

class queens_DMSG: public CMessage_queens_DMSG {
 public:
  CkGroupID counterGroup;
};

class queens_countMsg : public CMessage_queens_countMsg {
 public:
  int count;
  queens_countMsg(int c) : count(c) {};
};

class queens_main : public CBase_queens_main {
 public:
  CkGroupID counterGroup;
  queens_main(queens_DMSG * dmsg);
  queens_main(CkMigrateMessage *m) {}
  void Quiescence1(CkQdMsg *);
};

class queens_queens : public CBase_queens_queens {
 private:
  void seqQueens(int queens[], int nextRow);
  int consistent(int queens[], int lastRow, int col);
  void solutionFound(void);

 public:
  CkGroupID counterGroup;
  queens_queens(queens_PartialBoard *m);
  queens_queens(CkMigrateMessage *m) {}
};

class queens_counter: public Group {
 private:
  int myCount;
  int totalCount;
  int waitFor;
  CthThread threadId;
 public:
  queens_counter(void);
  queens_counter(CkMigrateMessage *m) {}
  void childCount(queens_countMsg *);
  void increment(void);
  void sendCounts(void);
  int  getTotalCount(void);
};

#endif
