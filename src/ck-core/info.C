/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include <converse.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if CMK_DEBUG_MODE

CsvDeclare(int*,  BreakPoints);
CsvDeclare(char*, SymbolTableInfo);
CpvDeclare(symbolTableType, SymbolTableFnArray);
typedef int offsetType[MAX_NUM_HANDLERS];
CpvDeclare(offsetType, offsetArray);
CpvDeclare(int, numBreakPoints);
CpvDeclare(indirectionType, indirectionFnArray);

CpvExtern(handlerType, handlerArray);

char *makeSymbolTableInfo(void)
{
  symbolTableFunction f;
  char *returnInfo, *newReturnInfo;
  int i;

  returnInfo = (char *)malloc(1);
  _MEMCHECK(returnInfo);
  returnInfo[0] = '\0';
  for(i = 0; i < MAX_NUM_HANDLERS; i++){
    if((f = CpvAccess(SymbolTableFnArray)[i]) != 0){
      char *p;

      p = (*f)();
      newReturnInfo = (char *)malloc(strlen(p) + strlen(returnInfo) + 1);
      _MEMCHECK(newReturnInfo);
      strcpy(newReturnInfo, returnInfo);
      strcat(newReturnInfo, p);
      free(returnInfo);
      free(p);
      returnInfo = newReturnInfo;
    }
  }
  
  return(returnInfo);
}

extern "C"
void CpdInitializeBreakPoints(void)
{
  int i;

  CsvInitialize(int *, BreakPoints);
  CsvInitialize(char *, SymbolTableInfo);
  CpvInitialize(symbolTableType, SymbolTableFnArray);
  CpvInitialize(offsetType, offsetArray);
  CpvInitialize(int, numBreakPoints);
  CpvInitialize(indirectionType, indirectionFnArray);

  CsvAccess(BreakPoints) = 0;
  CsvAccess(SymbolTableInfo) = 0;
  for(i = 0; i < MAX_NUM_HANDLERS; i++){
    CpvAccess(SymbolTableFnArray)[i] = 0;
    CpvAccess(offsetArray)[i] = 0;
    CpvAccess(indirectionFnArray)[i] = 0;
  }
  CpvAccess(numBreakPoints) = 0;
}

void symbolTableFnArrayRegister(int hndlrID, int noOfBreakPoints,
				symbolTableFunction f, indirectionFunction g)
{
  CpvAccess(SymbolTableFnArray)[hndlrID] = f;
  CpvAccess(indirectionFnArray)[hndlrID] = g;
  CpvAccess(offsetArray)[hndlrID] = CpvAccess(numBreakPoints);
  CpvAccess(numBreakPoints) += noOfBreakPoints;
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

  for(i = 0; i < CpvAccess(numBreakPoints); i++)
    CsvAccess(BreakPoints)[i] = (newBreakPoints[i] - '0');
}

char *getBreakPoints(void)
{
  char *temp;
  int i;

  if(CsvAccess(BreakPoints) == 0){
    CsvAccess(BreakPoints) = (int *)malloc(CpvAccess(numBreakPoints) 
					   *sizeof(int));
    _MEMCHECK(CsvAccess(BreakPoints));
    for(i = 0; i < CpvAccess(numBreakPoints); i++)
      CsvAccess(BreakPoints)[i] = 0;
  }

  temp = (char *)malloc(CpvAccess(numBreakPoints)*2*sizeof(char)+6);
  _MEMCHECK(temp);
  strcpy(temp, "");
  sprintf(temp, "%d#", CpvAccess(numBreakPoints));
  for(i = 0; i < CpvAccess(numBreakPoints); i++){
    char t[3];
    sprintf(t, "%d#", CsvAccess(BreakPoints)[i]);
    strcat(temp, t);
  }
  return(temp);
}

int isBreakPoint(char *msg)
{
  int hndlrID;
  int i;

  if(CsvAccess(BreakPoints) == 0){
    CsvAccess(BreakPoints) = (int *)malloc(CpvAccess(numBreakPoints)
					   *sizeof(int));
    _MEMCHECK(CsvAccess(BreakPoints));
    for(i = 0; i < CpvAccess(numBreakPoints); i++)
      CsvAccess(BreakPoints)[i] = 0;
  }
  hndlrID = CmiGetHandler(msg);
  if(CpvAccess(handlerArray)[hndlrID][0] != 0){
    int offset;
    indirectionFunction f;

    f = CpvAccess(indirectionFnArray)[hndlrID];
    offset = CpvAccess(offsetArray)[hndlrID] + (*f)(msg);

    return(CsvAccess(BreakPoints)[offset]);
  } else {
    return 0;
  }
}

int isEntryPoint(char *msg)
{
  int hndlrID;

  hndlrID = CmiGetHandler(msg);
  if(CpvAccess(handlerArray)[hndlrID][0] != 0){
    return 1;
  } else {
    return 0;
  }
} 

#endif
