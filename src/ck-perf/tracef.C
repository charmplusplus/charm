/****************************************************************************
             Fortran API for common Trace functions
****************************************************************************/

#include<charm++.h>
#include<charm-api.h>
                                                                                
CpvDeclare(int, a);
                                                                                
static int isInitialized=0;
                                                                                
void checkInit(void) {
        if (isInitialized) return;
        isInitialized=1;
        CpvInitialize(int,a);
        CpvAccess(a)=0;
}

FDECL {
void FTN_NAME(FTRACEBEGIN, ftracebegin)()
{
          checkInit();
          if ( CpvAccess(a) ==0)
                {
                        traceBegin();
                        CpvAccess(a)++;
                }
          else
                { CpvAccess(a)++;}
}

void FTN_NAME(FTRACEEND, ftraceend)()
{
          checkInit();
          if ( CpvAccess(a) == 1)
                {
                        traceEnd();
                        CpvAccess(a)--;
                }
          else
                { CpvAccess(a)--;}
                                                                                
}

void FTN_NAME(FTRACEREGISTERUSEREVENT, ftraceregisteruserevent)(char *x, int *ein, int *eout, int len)
{
  char *newstr = new char[len + 1];
  _MEMCHECK(newstr);
  strncpy(newstr, x, len);
  newstr[len] = 0;
  int newe = traceRegisterUserEvent(newstr, *ein);
  *eout = newe;
}

void FTN_NAME(FTRACEUSERBRACKETEVENT, ftraceuserbracketevent)(int *e, double *begint, double *endt)
{
  traceUserBracketEvent(*e, *begint, *endt);
}

void FTN_NAME(FCMIWALLTIMER, fcmiwalltimer)(double *t)
{
  *t = CmiWallTimer();
}

}  // FDECL

