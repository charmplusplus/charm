/*   define some FORTRAN interface for charm++ kernel functions
     Gengbin Zheng    12/15/2000

TODO:
   add other fortran name styles like all captial.
*/
#include "charm++.h"
#include <stdarg.h>
#include "charmf.h"

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

extern "C" void ckexit_()
{
  CkExit();
}

extern "C" void ckmype_(int *pe)
{
  *pe = CkMyPe();
}

extern "C" void cknumpes_(int *npe)
{
  *npe = CkNumPes();
}

extern "C" void ckprintf_(const char *format, ...)
{
  int ifmt, format_len, str_len=0, temp_len, flag;
  int *i; float *f; double *d;
  char str[100], temp_fmt[10];
  int j;

  va_list args;
  va_start(args,format);
  //format_len = strlen(format);
  //for (ifmt=0; ifmt<format_len;) {
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
