#include <charm.h>

#if CMK_DEBUG_MODE

int getCharmMsgHandlers(int *handleArray){
  *(handleArray) = CsvAccess(BUFFER_INCOMING_MSG_Index);
  *(handleArray + 1) = CsvAccess(MAIN_HANDLE_INCOMING_MSG_Index);
  return(2);
}

CsvDeclare(int *, BreakPoints);
CsvDeclare(char *, SymbolTableInfo);

CpvExtern(int *, handlerArray);
CpvExtern(int, noOfHandlers);

char* getEnvInfo(ENVELOPE *env){
  char *returnInfo;
  int size;
  int chareIndex;
  int epIndex = env -> EP;
  size = strlen((CsvAccess(EpInfoTable) + epIndex) -> name);
  chareIndex = (CsvAccess(EpInfoTable) + epIndex) -> chareindex;
  size += strlen((CsvAccess(ChareNamesTable))[chareIndex]);
  
  returnInfo = (char *)malloc((size + 2) * sizeof(char));
  strcpy(returnInfo, (CsvAccess(EpInfoTable) + epIndex) -> name);
  strcat(returnInfo, "#");
  strcat(returnInfo, (CsvAccess(ChareNamesTable))[chareIndex]);
  strcat(returnInfo, "#");
  return(returnInfo);
}

char* makeSymbolTableInfo(){
  int i, chareIndex;
  int size;
  char *returnInfo;
  char temp[10];
  
  size = CsvAccess(TotalEps) * 100;
  returnInfo = (char *)malloc(size * sizeof(char));
  strcpy(returnInfo, "");
  for(i = 0; i < CsvAccess(TotalEps); i++){
    strcat(returnInfo, CsvAccess(EpInfoTable)[i].name);
    strcat(returnInfo, "#");
    chareIndex = CsvAccess(EpInfoTable)[i].chareindex;
    strcat(returnInfo, CsvAccess(ChareNamesTable)[chareIndex]);
    strcat(returnInfo, "#");
    sprintf(temp, "%d", i);
    strcat(returnInfo, temp);
    strcat(returnInfo, "#");
  }

  return(returnInfo);
}

void CpdInitializeBreakPoints()
{
  int i;

  CsvInitialize(int *, BreakPoints);
  CsvInitialize(char *, SymbolTableInfo);
  CsvAccess(BreakPoints) = 0;
  CsvAccess(SymbolTableInfo) = 0;
}

char *getSymbolTableInfo()
{
  if(CsvAccess(SymbolTableInfo) == 0) 
    CsvAccess(SymbolTableInfo) = makeSymbolTableInfo();
  return(CsvAccess(SymbolTableInfo));
}

void setBreakPoints(char *newBreakPoints)
{
  int i;
  char *temp;

  for(i = 0; i < CsvAccess(TotalEps); i++)
    CsvAccess(BreakPoints)[i] = (newBreakPoints[i] - '0');
}

char *getBreakPoints()
{
  char *temp;
  int i;

  if(CsvAccess(BreakPoints) == 0){
    CsvAccess(BreakPoints) = (int *)malloc(CsvAccess(TotalEps) * sizeof(int));
    for(i = 0; i < CsvAccess(TotalEps); i++)
      CsvAccess(BreakPoints)[i] = 0;
  }

  temp = (char *)malloc((CsvAccess(TotalEps) + 1) * 2 * sizeof(char));
  strcpy(temp, "");
  sprintf(temp, "%s%d", temp, CsvAccess(TotalEps));
  strcat(temp, "#");
  for(i = 0; i < CsvAccess(TotalEps); i++){
    sprintf(temp, "%s%d#", temp, CsvAccess(BreakPoints)[i]);
  }

  return(temp);
}

int isBreakPoint(char *msg)
{
  ENVELOPE *env;
  int epIndex;
  int hndlrID;
  int i;

  if(CsvAccess(BreakPoints) == 0){
    CsvAccess(BreakPoints) = (int *)malloc(CsvAccess(TotalEps) * sizeof(int));
    for(i = 0; i < CsvAccess(TotalEps); i++)
      CsvAccess(BreakPoints)[i] = 0;
  }
  
  hndlrID = CmiGetHandler(msg);

  if((hndlrID == CpvAccess(handlerArray)[0]) || (hndlrID == CpvAccess(handlerArray)[1])){
    env = (ENVELOPE *)msg;
    return(CsvAccess(BreakPoints)[env->EP]);
  }
  else{
    return 0;
  }
}

int isEntryPoint(char *msg)
{
  int hndlrID;

  hndlrID = CmiGetHandler(msg);

  if((hndlrID == CpvAccess(handlerArray)[0]) || (hndlrID == CpvAccess(handlerArray)[1])){
    return 1;
  }
  else{
    return 0;
  }
} 

#endif
