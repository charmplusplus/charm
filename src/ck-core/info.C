#include "ck.h"
#include "stdio.h"

#if CMK_DEBUG_MODE

int getCharmMsgHandlers(int *handleArray)
{
  *(handleArray) = _charmHandlerIdx;
  *(handleArray+1) = _initHandlerIdx;
  return(2);
}

CsvDeclare(int*,  BreakPoints);
CsvDeclare(char*, SymbolTableInfo);

CpvExtern(int*, handlerArray);
CpvExtern(int,  noOfHandlers);

char* getEnvInfo(envelope *env)
{
  char *returnInfo;
  int size;
  int chareIndex;
  int epIndex = env->getEpIdx();
  size = strlen(_entryTable[epIndex]->name)+1;
  chareIndex = _entryTable[epIndex]->chareIdx;
  size += strlen(_chareTable[chareIndex]->name)+1;
  
  returnInfo = (char *)malloc((size + 2) * sizeof(char));
  strcpy(returnInfo, _entryTable[epIndex]->name);
  strcat(returnInfo, "#");
  strcat(returnInfo, _chareTable[chareIndex]->name);
  strcat(returnInfo, "#");
  return(returnInfo);
}

char* makeSymbolTableInfo(void)
{
  int i, chareIndex;
  int size;
  char *returnInfo;
  char temp[10];
  
  size = _numEntries * 100;
  returnInfo = (char *)malloc(size * sizeof(char));
  strcpy(returnInfo, "");
  for(i = 0; i < _numEntries; i++){
    strcat(returnInfo, _entryTable[i]->name);
    strcat(returnInfo, "#");
    chareIndex = _entryTable[i]->chareIdx;
    strcat(returnInfo, _chareTable[chareIndex]->name);
    strcat(returnInfo, "#");
    sprintf(temp, "%d", i);
    strcat(returnInfo, temp);
    strcat(returnInfo, "#");
  }

  return(returnInfo);
}

extern "C"
void CpdInitializeBreakPoints(void)
{
  int i;

  CsvInitialize(int *, BreakPoints);
  CsvInitialize(char *, SymbolTableInfo);
  CsvAccess(BreakPoints) = 0;
  CsvAccess(SymbolTableInfo) = 0;
}

char *getSymbolTableInfo(void)
{
  if(CsvAccess(SymbolTableInfo) == 0) 
    CsvAccess(SymbolTableInfo) = makeSymbolTableInfo();
  return(CsvAccess(SymbolTableInfo));
}

void setBreakPoints(char *newBreakPoints)
{
  int i;
  char *temp;

  for(i = 0; i < _numEntries; i++)
    CsvAccess(BreakPoints)[i] = (newBreakPoints[i] - '0');
}

char *getBreakPoints(void)
{
  char *temp;
  int i;

  if(CsvAccess(BreakPoints) == 0){
    CsvAccess(BreakPoints) = (int *)malloc(_numEntries * sizeof(int));
    for(i = 0; i < _numEntries; i++)
      CsvAccess(BreakPoints)[i] = 0;
  }

  temp = (char *)malloc(_numEntries*2*sizeof(char)+6);
  strcpy(temp, "");
  sprintf(temp, "%d#", _numEntries);
  for(i = 0; i < _numEntries; i++){
    char t[3];
    sprintf(t, "%d#", CsvAccess(BreakPoints)[i]);
    strcat(temp, t);
  }
  return(temp);
}

int isBreakPoint(char *msg)
{
  envelope *env;
  int epIndex;
  int hndlrID;
  int i;

  if(CsvAccess(BreakPoints) == 0){
    CsvAccess(BreakPoints) = (int *)malloc(_numEntries*sizeof(int));
    for(i = 0; i < _numEntries; i++)
      CsvAccess(BreakPoints)[i] = 0;
  }
  hndlrID = CmiGetHandler(msg);
  if((hndlrID == CpvAccess(handlerArray)[0]) || 
     (hndlrID == CpvAccess(handlerArray)[1])){
    env = (envelope *)msg;
    return(CsvAccess(BreakPoints)[env->getEpIdx()]);
  } else {
    return 0;
  }
}

int isEntryPoint(char *msg)
{
  int hndlrID;

  hndlrID = CmiGetHandler(msg);
  if((hndlrID == CpvAccess(handlerArray)[0]) || 
     (hndlrID == CpvAccess(handlerArray)[1])){
    return 1;
  } else {
    return 0;
  }
} 

#endif
