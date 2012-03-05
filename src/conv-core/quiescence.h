#ifndef _QUIESCENCE_H_
#define _QUIESCENCE_H_

struct ConvQdMsg 
{  
  char core[CmiMsgHeaderSizeBytes];
  int phase; /* 0..2*/
  union 
  {
    struct { int created; int processed; } p1;
    struct { int dirty; } p2;
  } u;
};


struct ConvQdState 
{
  int stage; /* 0..2*/
  int oProcessed;
  int mCreated, mProcessed;
  int cCreated, cProcessed;
  int cDirty;
  int nReported;
  int nChildren;
  int parent;
  int *children;
};


/* Declarations for CQdMsg related operations */
int  CQdMsgGetPhase(CQdMsg); 
void CQdMsgSetPhase(CQdMsg, int); 
int  CQdMsgGetCreated(CQdMsg); 
void CQdMsgSetCreated(CQdMsg, int); 
int  CQdMsgGetProcessed(CQdMsg); 
void CQdMsgSetProcessed(CQdMsg, int); 
int  CQdMsgGetDirty(CQdMsg); 
void CQdMsgSetDirty(CQdMsg, int); 

/* Declarations for CQdState related operations */
void CQdInit(void);
int  CQdGetCreated(CQdState);
void CQdCreate(CQdState, int);
int  CQdGetProcessed(CQdState);
void CQdProcess(CQdState, int);
void CQdPropagate(CQdState, CQdMsg); 
int  CQdGetParent(CQdState); 
int  CQdGetCCreated(CQdState); 
int  CQdGetCProcessed(CQdState); 
void CQdSubtreeCreate(CQdState, int); 
void CQdSubtreeProcess(CQdState, int); 
int  CQdGetStage(CQdState); 
void CQdSetStage(CQdState, int); 
void CQdReported(CQdState); 
int  CQdAllReported(CQdState); 
void CQdReset(CQdState); 
void CQdMarkProcessed(CQdState); 
int  CQdIsDirty(CQdState); 
void CQdSubtreeSetDirty(CQdState, int); 

CQdState CQdStateCreate(void);
void CQdHandler(CQdMsg);

#endif
