#include <stdio.h>
#include "xl-lex.h"
#include "xl-sym.h"
#include <string.h>

extern int OUTPUTOFF;
extern char *calloc();
extern char *Map();
extern int IMPORTFLAG,ImportStruct,ImportLevel;

extern FILE *outh0,*outh,*outh1,*outh2;
OUTPTR OUT0,OUT1,OUT2,OUT,CurrentOut;

OUTPTR GetOutStruct(fptr)
FILE *fptr;
{ OUTPTR dummy;

  dummy=(OUTPTR)calloc(1,sizeof(struct outstruct));
  dummy->lineno=dummy->linelength=0;
  dummy->file=fptr;
  return(dummy);
}

dummycall(){} 

void error(message,exitflag)
int exitflag;
char *message;
{ if (!exitflag&&InPass1) return;
  if (exitflag) dummycall();

  fprintf(stderr,"\"%s\", line %d: ",CurrentFileName,CurrentInputLineNo);
  PutOnScreen(message);
  if (exitflag) { fprintf(stderr,"** Stop.\n");
		  exit(1);
		}
}

void warning(message)
char *message;
{ if (InPass1) return;
  fprintf(stderr,"\"%s\", line %d: warning:",CurrentFileName,
          CurrentInputLineNo);
  PutOnScreen(message);
}

char *Prefix(to,by,prefix)
char *to,*by,*prefix;
{ char *temp;
  int bylength;
  
  temp=calloc(strlen(to)+(bylength=strlen(by))+2+strlen(prefix),
		sizeof (char));
  if (temp==NULL) error("Out of Memory in Prefix()",EXIT);
  strcpy(temp,prefix);strcat(temp,by);strcat(temp,".");strcat(temp,to);
  return(temp);
}

char *ModuleCharePrefix(module,chare,name)
char *module,*chare,*name;
{ char *temp,*dummy;

  temp=CharePrefix(chare,name);
  dummy=ModulePrefix(module,temp);
  dontfree(temp);
  return(dummy);
}

char *MyModuleCharePrefix(module,chare,name)
char *module,*chare,*name;
{ char *temp,*dummy;

  temp=CharePrefix(chare,name);
  dummy=MyModulePrefix(module,temp);
  dontfree(temp);
  return(dummy);
}

void writeinbuffer(string,flag)
int flag;
char *string;
{ int size;

  size = strlen(string);
  if (buffer.count+size >= 1000)
	{ error("Buffer Overflow",EXIT); exit(1); }
  strcpy(&(buffer.a[buffer.count]),string);
  buffer.count += size;
  if (flag) dontfree(string);
}

void WriteReturn()
{ if ((OUTPUTOFF)||(InPass1)) return;
  if (CurrentOut==NULL) return;
  if (BUFFEROUTPUT) { writeinbuffer("\n",0); return; }
  fprintf(CurrentOut->file,"\n"); CurrentOut->lineno++; CurrentOut->linelength=0; 
}

void writeoutput(string,freeflag)
char *string;
int freeflag;
{ int length;

  if ((CurrentOut==NULL)||(OUTPUTOFF)||(InPass1))
	{ if (freeflag) dontfree(string);
 	  return;
	}

  if (BUFFEROUTPUT) { writeinbuffer(string,freeflag); return; }

/* Removed on Nov. 12, 1991 - discussion with Sanjay. 

  if (((ImportLevel==0)&&(IMPORTFLAG))||(ImportStruct))
	string=Map(CurrentModule->name,(ImportStruct)?"_CKTYPE":"0",string);
*/
  length = strlen(string);
  if (CurrentOut->linelength+length >= MAXLINELENGTH)
	WriteReturn();
  CurrentOut->linelength += length;
  if (CurrentInputLineNo != CurrentOut->lineno)
	{ CurrentOut->lineno = CurrentInputLineNo;
	/* Assuming that when I & O fall out, \n has been inserted! */
	  fprintf(CurrentOut->file,"\n# line %d \"%s\"\n",CurrentInputLineNo,CurrentFileName);
	}
  WriteString(string);
  if (freeflag) dontfree(string);
}

