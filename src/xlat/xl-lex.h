/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile$
 *	$Author$	$Locker$		$State$
 *	$Revision$	$Date$
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 ***************************************************************************
 * REVISION HISTORY:
 *
 * $Log$
 * Revision 2.3  1997-03-25 15:04:58  milind
 * Made changes suggested by Ed Kornkven to fix bugs in Dagger.
 *
 * Revision 2.2  1995/06/15 20:57:00  jyelon
 * *** empty log message ***
 *
 * Revision 2.1  1995/06/15  20:27:11  jyelon
 * got rid of dontfree.
 *
 * Revision 2.0  1995/06/05  18:52:05  brunner
 * Reorganized file structure
 *
 * Revision 1.2  1994/11/11  05:32:33  brunner
 * Removed ident added by accident with RCS header
 *
 * Revision 1.1  1994/11/07  15:40:59  brunner
 * Initial revision
 *
 ***************************************************************************/
#include <stdio.h>

#define dontfree(x) 0

extern char *CkPrefix;	/* defined in outh.c */
extern char *CkPrefix_;	/* defined in outh.c */
extern char *CKMAINDATAFUNCTION;	/* defined in outh.c */
extern char *CKMAINCHAREFUNCTION;	/* defined in outh.c */
extern char *CKMAINQUIESCENCEFUNCTION;	/* defined in outh.c */
extern char *REFSUFFIX;			/* defined in outh.c */

extern char *CkMyData;	/* defined in yaccspec */

#define FILENAMELENGTH 1000

#define UNDEFINED 1000
		/* UNDEFINED also defined in symtab.h */
#define NOEXIT 0
#define EXIT 1

#define NOFREE 0
#define FREE   1

#define PutOnScreen(message)	fprintf(stderr,"%s",message)
#define memory_error error

extern char *MakeString();

typedef struct token
{ char *name;
  int  tokenvalue;
} USELESS_TOKEN;

extern struct token *TokenArray;
extern struct token *KeyArray;
extern int TotalTokens;

extern int CurrentInputLineNo,CurrentOutputLineNo,OutputLineLength;
extern char CurrentFileName[];
extern FILE *outfile;

extern void error(/*char *message,int exitflag*/);
extern void warning(/*char *message*/);

extern int IsKey(/* char *tokenstring */);	/* in readtokens.c */

/* *** Following are required for yacc *** */

typedef struct ysn
{ char *string;
  char *modstring;
  struct listnode *listptr;
  int idtype;
  struct typenode *type;
  struct ysn *ysn;
  struct symtabnode *table;
  int count;
} YaccStackNode,*YSNPTR;

typedef struct listnode
{ YSNPTR ysn;
  struct listnode *next;
  struct listnode *prev;
} LISTNODE,*LISTPTR;

extern int TypeID(/* char *name */);
extern LISTPTR GetListNode(/* YSNPTR eleptr */);
extern void InsertNode(/*  LISTPTR listptr, YSNPTR eleptr */);
extern YSNPTR GetYSN();

/* *** Following for writeoutput *** */
extern int IMPORTFLAG;
#define MAXLINELENGTH 1024
#define WriteString(string)	fprintf(CurrentOut->file,"%s",string)

#define MyModulePrefix(module,name)	Prefix(name,module,CkPrefix_)
#define ModulePrefix(module,name)	Prefix(name,module,CkPrefix)
#define CharePrefix(chare,name)		Prefix(name,chare,CkPrefix)

extern void WriteReturn();
extern void writeoutput(/* char *string, int freeflag */);
extern char *ModuleCharePrefix(/* char *module, char *chare, char *name */);
extern char *MyModuleCharePrefix(/* char *module, char *chare, char *name */);
extern char *Prefix(/* char *to, char *by , char *prefix */);

extern int InPass1;
extern int MESSAGEON;

extern char *GetMem();

typedef struct outstruct
{ FILE *file;
  int lineno;
  int linelength;
} *OUTPTR;

extern OUTPTR GetOutStruct(/* fptr */);
extern OUTPTR OUT0,OUT1,OUT2,OUT,CurrentOut;

typedef struct
{ char a[1000];
  int  count;
} BUFFERTYPE;

extern BUFFERTYPE buffer;

extern int BUFFEROUTPUT;
