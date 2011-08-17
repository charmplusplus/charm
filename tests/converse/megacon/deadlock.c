#include <stdio.h>
#include <converse.h>

typedef struct incmsg
{
  char head[CmiMsgHeaderSizeBytes];
  int n;
}
*incmsg;

CpvDeclare(int, deadlock_inc_idx);
CpvDeclare(int, deadlock_cram_idx);
CpvDeclare(int, deadlock_count);

void Cpm_megacon_ack();

void deadlock_inc(incmsg m)
{
  CpvAccess(deadlock_count) += m->n;
  if (CpvAccess(deadlock_count)==0) {
    Cpm_megacon_ack(CpmSend(0));
  }
  CmiFree(m);
}

void deadlock_cram(char *msg)
{
  struct incmsg m={{0},1};
  int count = 0;
  CmiSetHandler(&m, CpvAccess(deadlock_inc_idx));
  while (count<5000) {
    CmiSyncSend(1-CmiMyPe(), sizeof(m), &m);
    count++;
  } 
  m.n = -count;
  CmiSyncSend(1-CmiMyPe(), sizeof(m), &m);
  CmiFree(msg);
}

void deadlock_init()
{
  char msg[CmiMsgHeaderSizeBytes]={0};
  if (CmiNumPes()<2) {
    CmiPrintf("warning: need 2 processors for deadlock-test, skipping.\n");
    Cpm_megacon_ack(CpmSend(0));
    Cpm_megacon_ack(CpmSend(0));
  } else {
    CmiSetHandler(msg, CpvAccess(deadlock_cram_idx));
    CmiSyncSend(0, sizeof(msg), msg);
    CmiSyncSend(1, sizeof(msg), msg);
  }
}

#if CMK_DEBUG_MODE

static const char* _fCramHeaderStr = "DeadLock Cram Header";
static const char* _fCramContentStr = "DeadLock Cram Message";
static const char* _fIncHeaderStr = "DeadLock Inc Header";
static const char* _fIncContentStr = "DeadLock Inc Message";

static char* fCramHeader(char *msg){
  char *temp;

  temp = (char *)malloc(strlen(_fCramHeaderStr) + 1);
  strcpy(temp, _fCramHeaderStr);
  return(temp);
}

static char* fCramContent(char *msg){
  char *temp;
  
  temp = (char *)malloc(strlen(_fCramContentStr) + 1);
  strcpy(temp, _fCramContentStr);
  return(temp);
}

static char* fIncHeader(char *msg){
  char *temp;

  temp = (char *)malloc(strlen(_fIncHeaderStr) + 1);
  strcpy(temp, _fIncHeaderStr);
  return(temp);
}

static char* fIncContent(char *msg){
  char *temp;
  
  temp = (char *)malloc(strlen(_fIncContentStr) + 1 + 5);
  sprintf(temp, "%s:%d", _fIncContentStr, ((incmsg)msg)->n);
  return(temp);
}

char* makeIncSymbolTableInfo()
{
  int i, chareIndex;
  int size;
  char *returnInfo;
  char temp[10];
  
  size = 200;
  returnInfo = (char *)malloc(size * sizeof(char));
  strcpy(returnInfo, "");
  strcat(returnInfo, "Converse Handler : deadlock_inc");
  strcat(returnInfo, "#");
  
  return(returnInfo);
}

int getInd(char *msg)
{
  return 0;
}

char* makeCramSymbolTableInfo()
{
  int i, chareIndex;
  int size;
  char *returnInfo;
  char temp[10];
  
  size = 200;
  returnInfo = (char *)malloc(size * sizeof(char));
  strcpy(returnInfo, "");
  strcat(returnInfo, "Converse Handler : deadlock_cram");
  strcat(returnInfo, "#");
  
  return(returnInfo);
}

#endif

void deadlock_moduleinit()
{
  CpvInitialize(int, deadlock_inc_idx);
  CpvInitialize(int, deadlock_cram_idx);
  CpvInitialize(int, deadlock_count);
  CpvAccess(deadlock_inc_idx) = CmiRegisterHandler((CmiHandler)deadlock_inc);
  CpvAccess(deadlock_cram_idx) = CmiRegisterHandler((CmiHandler)deadlock_cram);
#if CMK_DEBUG_MODE
  handlerArrayRegister(CpvAccess(deadlock_inc_idx), fIncHeader, fIncContent);
  handlerArrayRegister(CpvAccess(deadlock_cram_idx),fCramHeader, fCramContent);
  
  symbolTableFnArrayRegister(CpvAccess(deadlock_inc_idx), 1,
			     makeIncSymbolTableInfo,
			     getInd);
  symbolTableFnArrayRegister(CpvAccess(deadlock_cram_idx), 1,
			     makeCramSymbolTableInfo,
			     getInd);
  
#endif
  CpvAccess(deadlock_count) = 0;
}
