/*   define some FORTRAN interface for charm++ kernel functions
     Gengbin Zheng    12/15/2000
*/

#include "charm++.h"
#include <stdarg.h>
#include "charmf.h"
#include "cktiming.h"

extern "C" int typesize(int type, int count)
{
  switch(type) {
    case CMPI_DOUBLE_PRECISION : return count*sizeof(double);
    case CMPI_INTEGER : return count*sizeof(int);
    case CMPI_REAL : return count*sizeof(float);
    case CMPI_COMPLEX: return 2*count*sizeof(double);
    case CMPI_LOGICAL: return 2*count*sizeof(int);
    case CMPI_CHAR:
    case CMPI_BYTE:
    case CMPI_PACKED:
    default:
      return 2*count;
  }
}         

extern "C" void FTN_NAME(CKEXIT, ckexit)()
{
  CkExit();
}

extern "C" void FTN_NAME(CKMYPE, ckmype)(int *pe)
{
  *pe = CkMyPe();
}

extern "C" void FTN_NAME(CKNUMPES, cknumpes)(int *npe)
{
  *npe = CkNumPes();
}

extern "C" void FTN_NAME(CKPRINTF, ckprintf)(const char *format, ...)
{
  int ifmt, str_len=0, temp_len, flag;
  int *i; float *f; double *d;
  char str[100], temp_fmt[10];
  int j;

  va_list args;
  va_start(args,format);
  for (ifmt=0;;) {
    if (format[ifmt]=='$') break; // $ is end of input
    if (format[ifmt]=='%') {
      temp_fmt[0]='%'; temp_len=1;
      ifmt++;
      for (j=0; ; j++) {
	flag=0;
	switch (format[ifmt]) {
	case 'i': 
	case 'd':
	  i = va_arg(args, int *);
	  temp_fmt[temp_len] = 'i'; temp_fmt[++temp_len]='\0';
	  str_len += sprintf(str+str_len,temp_fmt,*i); 
	  ifmt++;
	  flag=1; break; 
	case 'e':
	case 'f':
	  f = va_arg(args, float *);
	  temp_fmt[temp_len] = format[ifmt]; temp_fmt[++temp_len]='\0';
	  str_len += sprintf(str+str_len,temp_fmt,*f); 
	  ifmt++;
	  flag=1; break;
	case 'E':
	case 'F':
	  d = va_arg(args, double *);
	  temp_fmt[temp_len] = format[ifmt]+32; temp_fmt[++temp_len]='\0';
	  str_len += sprintf(str+str_len,temp_fmt,*d); 
	  ifmt++;
	  flag=1; break;
	default:
	  if ((format[ifmt]=='.')&&(format[ifmt]<='9')||(format[ifmt]>='0')) {
	    temp_fmt[temp_len] = format[ifmt];
	    temp_len++; ifmt++;
	  }
	  else {
	    printf("Print format error!\n"); return;
	  }
	} // end of switch
	if (flag) break; // break for(j=0;;j++)
      }
    }
    else if (format[ifmt]=='\\') {
      ifmt++;
      if (format[ifmt]=='n') { 
	str[str_len] = '\n'; 
	str_len++;
	ifmt++;
      }
    }
    else {
      str[str_len] = format[ifmt]; 
      str_len++;
      ifmt++;
    }
  } // for (ifmt=0;;)
  str[str_len]='\0';
  CkPrintf("%s",str);
  //vprintf(format, args);
  //fflush(stdout);
  va_end(args);
}

FDECL int FTN_NAME(CHARM_IARGC,charm_iargc)(void) {
  return CkGetArgc()-1;
}

FDECL void FTN_NAME(CHARM_GETARG,charm_getarg)
        (int *i_p,char *dest,int destLen)
{
  int i=*i_p;
  if (i<0) CkAbort("charm_getarg called with negative argument!");
  if (i>=CkGetArgc()) CkAbort("charm_getarg called with argument > iargc!");
  const char *src=CkGetArgv()[i];
  strcpy(dest,src);
  for (i=strlen(dest);i<destLen;i++) dest[i]=' ';
}

// memory functions

FDECL CmiInt8 FTN_NAME(CMIMEMORYUSAGE, cmimemoryusage) ()
{
  CmiInt8 mem = CmiMemoryUsage();
  return mem;
}

FDECL CmiInt8 FTN_NAME(CMIMAXMEMORYUSAGE, cmimaxmemoryusage) ()
{
  return CmiMaxMemoryUsage();
}

FDECL CmiFloat8 FTN_NAME(CMIWALLTIMER, cmiwalltimer) ()
{
  return CmiWallTimer();
}

FDECL CmiFloat8 FTN_NAME(CKWALLTIMER, ckwalltimer) ()
{
  return CkWallTimer();
}

FDECL CmiFloat8 FTN_NAME(CMICPUTIMER, cmicputimer) ()
{
  return CmiCpuTimer();
}

FDECL CmiFloat8 FTN_NAME(CKCPUTIMER, ckcputimer) ()
{
  return CkCpuTimer();
}

FDECL void FTN_NAME(CMIDISABLEISOMALLOC, cmidisableisomalloc) ()
{
  CmiDisableIsomalloc();
}

FDECL void FTN_NAME(CMIENABLEISOMALLOC, cmienableisomalloc) ()
{
  CmiEnableIsomalloc();
}

FDECL void FTN_NAME(CMIDISABLETLS, cmidisabletls) ()
{
  CmiDisableTLS();
}

FDECL void FTN_NAME(CMIENABLETLS, cmienabletls) ()
{
  CmiEnableTLS();
}

FDECL void FTN_NAME(CMIMEMORYCHECK, cmimemorycheck) ()
{
  CmiMemoryCheck();
}

// cktiming utility

FDECL void FTN_NAME(INITBIGSIMTRACE, initbigsimtrace)(int *outputParams, int *outputtiming)
{
  initBigSimTrace(*outputParams, *outputtiming);
}

FDECL void FTN_NAME(FINALIZEBIGSIMTRACE, finalizebigsimtrace)()
{
  finalizeBigSimTrace();
}

FDECL void FTN_NAME(STARTTRACEBIGSIM, starttracebigsim)()
{
  startTraceBigSim();
}

FDECL void FTN_NAME(ENDTRACEBIGSIM1, endtracebigsim1)(char *eventName, int *stepno, double *p1, int len)
{
  char str[128];
  strncpy(str,eventName, len);
  str[len] = 0;
  endTraceBigSim(str, *stepno, *p1);
}


FDECL void FTN_NAME(ENDTRACEBIGSIM2, endtracebigsim2)(char *eventName, int *stepno, double *p1, double *p2, int len)
{
  char str[128];
  strncpy(str,eventName, len);
  str[len] = 0;
  endTraceBigSim(str, *stepno, *p1, *p2);
}

FDECL void FTN_NAME(ENDTRACEBIGSIM3, endtracebigsim3)(char *eventName, int *stepno, double *p1, double *p2, double *p3, int len)
{
  // printf("%d %f %f %f\n", *stepno, *p1, *p2, *p3);
  char str[128];
  strncpy(str,eventName, len);
  str[len] = 0;
  endTraceBigSim(str, *stepno, *p1, *p2, *p3);
}

FDECL void FTN_NAME(ENDTRACEBIGSIM4, endtracebigsim4)(char *eventName, int *stepno, double *p1, double *p2, double *p3, double *p4, int len)
{
  // printf("%d %f %f %f\n", *stepno, *p1, *p2, *p3, *p4);
  char str[128];
  strncpy(str,eventName, len);
  str[len] = 0;
  endTraceBigSim(str, *stepno, *p1, *p2, *p3, *p4);
}

FDECL void FTN_NAME(ENDTRACEBIGSIM5, endtracebigsim5)(char *eventName, int *stepno, double *p1, double *p2, double *p3, double *p4, double *p5, int len)
{
  // printf("%d %f %f %f\n", *stepno, *p1, *p2, *p3, *p4, *p5);
  char str[128];
  strncpy(str,eventName, len);
  str[len] = 0;
  endTraceBigSim(str, *stepno, *p1, *p2, *p3, *p4, *p5);
}

FDECL void FTN_NAME(ENDTRACEBIGSIM6, endtracebigsim6)(char *eventName, int *stepno, double *p1, double *p2, double *p3, double *p4, double *p5,  double *p6, int len)
{
  // printf("%d %f %f %f\n", *stepno, *p1, *p2, *p3, *p4, *p5, *p6);
  char str[128];
  strncpy(str,eventName, len);
  str[len] = 0;
  endTraceBigSim(str, *stepno, *p1, *p2, *p3, *p4, *p5, *p6);
}

FDECL void FTN_NAME(ENDTRACEBIGSIM7, endtracebigsim7)(char *eventName, int *stepno, double *p1, double *p2, double *p3, double *p4, double *p5,  double *p6, double *p7, int len)
{
  // printf("%d %f %f %f\n", *stepno, *p1, *p2, *p3, *p4, *p5, *p6, *p7);
  char str[128];
  strncpy(str,eventName, len);
  str[len] = 0;
  endTraceBigSim(str, *stepno, *p1, *p2, *p3, *p4, *p5, *p6, *p7);
}

FDECL void FTN_NAME(ENDTRACEBIGSIM8, endtracebigsim8)(char *eventName, int *stepno, double *p1, double *p2, double *p3, double *p4, double *p5,  double *p6, double *p7, double *p8, int len)
{
  // printf("%d %f %f %f\n", *stepno, *p1, *p2, *p3, *p4, *p5, *p6, *p7, *p8);
  char str[128];
  strncpy(str,eventName, len);
  str[len] = 0;
  endTraceBigSim(str, *stepno, *p1, *p2, *p3, *p4, *p5, *p6, *p7, *p8);
}

FDECL void FTN_NAME(ENDTRACEBIGSIM9, endtracebigsim9)(char *eventName, int *stepno, double *p1, double *p2, double *p3, double *p4, double *p5,  double *p6, double *p7, double *p8, double *p9, int len)
{
  // printf("%d %f %f %f\n", *stepno, *p1, *p2, *p3, *p4, *p5, *p6, *p7, *p8, *p9);
  char str[128];
  strncpy(str,eventName, len);
  str[len] = 0;
  endTraceBigSim(str, *stepno, *p1, *p2, *p3, *p4, *p5, *p6, *p7, *p8, *p9);
}

FDECL void FTN_NAME(ENDTRACEBIGSIM10, endtracebigsim10)(char *eventName, int *stepno, double *p1, double *p2, double *p3, double *p4, double *p5,  double *p6, double *p7, double *p8, double *p9, double *p10, int len)
{
  // printf("%d %f %f %f\n", *stepno, *p1, *p2, *p3, *p4, *p5, *p6, *p7, *p8, *p9, *p10);
  char str[128];
  strncpy(str,eventName, len);
  str[len] = 0;
  endTraceBigSim(str, *stepno, *p1, *p2, *p3, *p4, *p5, *p6, *p7, *p8, *p9, *p10);
}

FDECL void FTN_NAME(ENDTRACEBIGSIM11, endtracebigsim11)(char *eventName, int *stepno, double *p1, double *p2, double *p3, double *p4, double *p5,  double *p6, double *p7, double *p8, double *p9, double *p10, double *p11, int len)
{
  // printf("%d %f %f %f\n", *stepno, *p1, *p2, *p3, *p4, *p5, *p6, *p7);
  char str[128];
  strncpy(str,eventName, len);
  str[len] = 0;
  endTraceBigSim(str, *stepno, *p1, *p2, *p3, *p4, *p5, *p6, *p7, *p8, *p9, *p10, *p11);
}




