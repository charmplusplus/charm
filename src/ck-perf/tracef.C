/****************************************************************************
             Fortran API for common Trace functions
****************************************************************************/

#include<charm++.h>
#include<charm-api.h>
                                                                                
CpvStaticDeclare(int, a);
                                                                                
static int isInitialized=0;
                                                                                
static void checkInit(void) {
        if (isInitialized) return;
        isInitialized=1;
        CpvInitialize(int,a);
        CpvAccess(a)=0;
}

static char * FortrantoCString(char *x,int len){
	char *newstr = new char[len + 1];
  _MEMCHECK(newstr);
  strncpy(newstr, x, len);
  newstr[len] = 0;
	return newstr;
}


FDECL {

#define ftracebegin              FTN_NAME(FTRACEBEGIN, ftracebegin)
#define ftraceend		 FTN_NAME(FTRACEEND, ftraceend)
#define ftraceregisteruserevent  FTN_NAME(FTRACEREGISTERUSEREVENT, ftraceregisteruserevent)
#define ftraceuserbracketevent   FTN_NAME(FTRACEUSERBRACKETEVENT, ftraceuserbracketevent)
#define ftraceUserEvent   	 FTN_NAME(FTRACEUSEREVENT, ftraceuserevent)
#define ftraceFlushLog   	 FTN_NAME(FTRACEFLUSHLOG, ftraceflushlog)
#define ftraceRegisterFunc	 FTN_NAME(FTRACEREGISTERFUNC,ftraceregisterfunc) 
#define ftraceBeginFunc		 FTN_NAME(FTRACEBEGINFUNC,ftracebeginfunc)
#define ftraceEndFunc		 FTN_NAME(FTRACEENDFUNC,ftraceendfunc)

#define fbgprintf		 FTN_NAME(FBGPRINTF, fbgprintf)

void ftracebegin()
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

void ftraceend()
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

void ftraceregisteruserevent(char *x, int *ein, int *eout, int len)
{
  char *newstr = new char[len + 1];
  _MEMCHECK(newstr);
  strncpy(newstr, x, len);
  newstr[len] = 0;
  int newe = traceRegisterUserEvent(newstr, *ein);
  *eout = newe;
}

void ftraceuserbracketevent(int *e, double *begint, double *endt)
{
  traceUserBracketEvent(*e, *begint, *endt);
}

void ftraceUserEvent(int *e)
{
  traceUserEvent(*e);
}

void ftraceFlushLog()
{
  traceFlushLog();
}

void ftraceRegisterFunc(char *name,int *outIdx,int lenName){
	char *newstr = FortrantoCString(name,lenName);
	*outIdx = traceRegisterFunction(newstr);
	delete [] newstr;
}

void ftraceBeginFunc(int *idx){
	traceBeginFuncIndexProj(*idx,(char*)"FORTRAN",0);
}

void ftraceEndFunc(int *idx){
	traceEndFuncIndexProj(*idx);
}

#if CMK_BIGSIM_CHARM
void fbgprintf(char *str, int len)
{
  char *newstr = new char[len + 1];
  _MEMCHECK(newstr);
  strncpy(newstr, str, len);
  newstr[len] = 0;
  BgPrintf(newstr);
  delete [] newstr;
}
#endif

}  // FDECL

