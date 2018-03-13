#ifndef _QUIESCENCE_H_
#define _QUIESCENCE_H_

struct ConvQdMsg 
{  
  char core[CmiMsgHeaderSizeBytes];
  int phase; /* 0..2*/
  union 
  {
    struct { CmiInt8 created; CmiInt8 processed; } p1;
    struct { char dirty; } p2;
  } u;
};


struct ConvQdState 
{
  int stage; /* 0..2*/
  char cDirty;
  CmiInt8 oProcessed;
  CmiInt8 mCreated, mProcessed;
  CmiInt8 cCreated, cProcessed;
  int nReported;
  int nChildren;
  int parent;
  int *children;
};


/* Declarations for CQdMsg related operations */
int  CQdMsgGetPhase(CQdMsg); 
void CQdMsgSetPhase(CQdMsg, int); 
CmiInt8 CQdMsgGetCreated(CQdMsg);
void CQdMsgSetCreated(CQdMsg, CmiInt8);
CmiInt8 CQdMsgGetProcessed(CQdMsg);
void CQdMsgSetProcessed(CQdMsg, CmiInt8);
char CQdMsgGetDirty(CQdMsg); 
void CQdMsgSetDirty(CQdMsg, char); 

/* Declarations for CQdState related operations */
void CQdInit(void);
CmiInt8 CQdGetCreated(CQdState);
void CQdCreate(CQdState, CmiInt8);
CmiInt8 CQdGetProcessed(CQdState);
void CQdProcess(CQdState, CmiInt8);
void CQdPropagate(CQdState, CQdMsg); 
int  CQdGetParent(CQdState); 
CmiInt8 CQdGetCCreated(CQdState);
CmiInt8 CQdGetCProcessed(CQdState);
void CQdSubtreeCreate(CQdState, CmiInt8);
void CQdSubtreeProcess(CQdState, CmiInt8);
int  CQdGetStage(CQdState); 
void CQdSetStage(CQdState, int); 
void CQdReported(CQdState); 
int  CQdAllReported(CQdState); 
void CQdReset(CQdState); 
void CQdMarkProcessed(CQdState); 
char CQdIsDirty(CQdState); 
void CQdSubtreeSetDirty(CQdState, char); 

CQdState CQdStateCreate(void);
void CQdHandler(CQdMsg);

#endif
