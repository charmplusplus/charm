#include <stdio.h>

#define dontfree(x) 0

extern char *MakeString();		/* in "string.c" */

#define TRUE 1
#define FALSE 0

#define ON 1
#define OFF 0

#define UNDEFINED 1000

#define VARNAME 1
#define FNNAME 2
#define ENTRYNAME 3
#define CHARENAME 4
#define MODULENAME 5
#define TYPENAME 6
#define FIELDNAME 7
#define ARRAYNAME 8
#define STRUCTNAME 9
#define UNIONNAME 10
#define MESSAGENAME 11
#define PRIVATEFNNAME 12
#define BOCNAME 13
#define SYSCALLNAME 14
#define PUBLICFNNAME 15
#define VARSIZENAME 16
#define SYSTEMTYPENAME 17
#define READONLYVAR 18
#define READONLYARRAY 19
#define READONLYPTR 20
#define READONLYMSG 21
#define ACCNAME 22
#define MONONAME 23
#define TABLENAME 24
#define OTHERNAME 25
#define ENUMNAME  26
#define ENTRYNAMEIMP 27

#define CHARTYPE 1
#define SHORTTYPE 2
#define INTTYPE 3
#define LONGTYPE 4
#define UNSIGNEDTYPE 5
#define FLOATTYPE 6
#define DOUBLETYPE 7
#define POINTERTYPE 8
#define STRUCTTYPE 9
#define UNIONTYPE 10
#define MESSAGETYPE 11
#define FUNCTIONTYPE 12
#define ARRAYTYPE 13
#define ENUMTYPE 14

#define DECLARED 1
#define IMPORTED 2
#define NOTDECLARED 3

#define AUTO_SC 1
#define STATIC_SC 2
#define EXTERN_SC 3
#define REGISTER_SC 4
#define TYPEDEF_SC 5

#define CHARSIZE 1
#define SHORTSIZE 2
#define INTSIZE 4
#define LONGSIZE 4
#define UNSIGNEDSIZE 4
#define FLOATSIZE 4
#define DOUBLESIZE 8

typedef struct typenode
{ int basictype;			/* CHARTYPE .. MESSAGETYPE */
  int count;				/* how many times referenced */
  int size;
  int declflag;
  struct typenode *type;		/* array or some such basic type */
  struct symtabnode *table;		/* NULL unless message/struct/union */  
} *TYPEPTR;

typedef struct symtabnode
{ char *name;				/* name identifier */
  int idtype;				/* its type - VARNAME .. OTHERNAME */
  struct symtabnode *modname;		/* module in which declared */
  struct symtabnode *charename;		/* chare in which declared */
  int level;				/* level where declared */
  int declflag;				/* DECLARED or IMPORTED */
  int storageclass;			/* AUTO_SC .. TYPEDEF_SC */
  struct typenode *type;		/* pointer to its type */
  struct symtabnode *left,*right;	/* to support binary tree */
  struct symtabnode *prev,*next;	/* to support doubly linked list */
  struct ysn *ysn;
  int localid;				/* used with MESSAGES */
  int msgno,userpack;
  int implicit_entry;                  /* used with entrynames, */
} *SYMTABPTR;

typedef struct stacknode
{ struct stacknode *prev,*next;
  struct symtabnode *tableptr;
  int level;
} *STACKPTR;

extern SYMTABPTR CurrentTable;
extern STACKPTR StackTop;
extern STACKPTR StackBase;
extern int CurrentLevel;
extern SYMTABPTR CurrentModule;
extern SYMTABPTR CurrentChare;
extern TYPEPTR CHARPTR,INTPTR,SHORTPTR,LONGPTR,UNSIGNEDPTR,FLOATPTR,DOUBLEPTR,
	DUMMYPTR,VOIDPTR;
extern TYPEPTR CHAREIDPTR,ENTRYPOINTPTR,PENUMPTR,PACKIDPTR,WRITEONCEIDPTR,
		PVECTORPTR,CHARENAMEPTR,CHARENUMPTR,ENTRYNUMPTR,BOOLEANPTR;

extern TYPEPTR FUNCTIONPTR, FUNCTIONREFPTR; /* Jan 17 1992, Added by Attila */

extern int FUNCTIONCOUNT;

extern SYMTABPTR GetSymTabNode(/*char *name*/);
extern STACKPTR GetStackNode();
extern TYPEPTR GetTypeNode(/*int count, int size */);

extern void InitSymTable();
extern void PushStack();
extern void PopStack(/* int freeflag */);
extern void FreeTree(/*SYMTABPTR root*/);

extern SYMTABPTR FindInTable(/* SYMTABPTR root,char *name,int *i*/);

extern SYMTABPTR LocalFind(/*char *name*/);
extern SYMTABPTR GlobalFind(/*char *name*/);
 
extern SYMTABPTR Insert(/*char *name*/);

extern int TypeID(/* char *name */);	/* returns 0 iff name is not a typeid */

extern void FillSymTabNode( /* 	SYMTABPTR node,
				int idtype, int declflag,
				int storageclass, int basictype,
				int count, int tableflag,
				TYPEPTR typeptr */
			  );

extern SYMTABPTR ImportModule,ModuleDefined,Pass1Module;
