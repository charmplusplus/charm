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
 * Revision 2.2  1998-06-16 17:02:34  milind
 * Fixed typedefs for net-rs6k and sp3.
 * Also fixed a longstanding charm translator bug to deal with quirks of
 * ld on AIX.
 *
 * Revision 2.1  1995/06/15 20:57:00  jyelon
 * *** empty log message ***
 *
 * Revision 2.0  1995/06/05  18:52:05  brunner
 * Reorganized file structure
 *
 * Revision 1.1  1994/11/03  17:41:51  brunner
 * Initial revision
 *
 ***************************************************************************/
static char ident[] = "@(#)$Header$";

#include "xl-lex.h"
#include "xl-sym.h"

extern char *calloc();
SYMTABPTR CheckInsert();
extern TYPEPTR SetType();

char *GetMem(n)
int n;
{ char *dummy;

  dummy=calloc(n,sizeof(char));
  if (dummy==NULL) error("Out of Memory in GetMem()",EXIT);

  return(dummy);
}

LISTPTR GetListNode(eleptr)
YSNPTR eleptr;
{ LISTPTR dummy;

  dummy=(LISTPTR)calloc(1,sizeof(struct listnode));
  if (dummy==NULL) error("Out of Memory in GetListNode()",EXIT);
  
  dummy->next=dummy->prev=dummy;
  dummy->ysn=eleptr;
  return(dummy);
}

void InsertNode(listptr,eleptr)
LISTPTR listptr;
YSNPTR eleptr;
{ LISTPTR dummy;

  if (listptr==NULL) error("InsertNode() : Following NULL",EXIT);
 
  listptr->prev->next=dummy=GetListNode(eleptr);
  dummy->next=listptr;
  dummy->prev=listptr->prev;
  listptr->prev=dummy;
}

YSNPTR GetYSN()
{ YSNPTR dummy;

  dummy=(YSNPTR)calloc(1,sizeof(struct ysn));
  dummy->idtype=UNDEFINED;
  dummy->listptr=NULL;
  dummy->string=NULL;
  dummy->ysn=NULL;
  dummy->type=NULL;
  dummy->table=NULL;
  dummy->count=0;
  dummy->modstring=NULL;
  return(dummy);
}

typedef struct mapnode
{ char *module;
  char *chare;
  char *name;
  char *mappedname;
  struct mapnode *next;
  int  mapid;
} *MAPPTR;

MAPPTR MapHead;

MAPPTR GetMapNode(module,chare,name)
char *module,*chare,*name;
{ MAPPTR dummy;

  dummy=(MAPPTR)calloc(1,sizeof(struct mapnode));
  if (dummy==NULL) error("Out of Memory in GetMapNode()",EXIT);

  if (module!=NULL) dummy->module=MakeString(module); else dummy->module=NULL;
  if (chare!=NULL) dummy->chare=MakeString(chare); else dummy->chare=NULL;
  if (name!=NULL) dummy->name=MakeString(name); else dummy->name=NULL;
  dummy->mappedname=NULL;
  dummy->mapid = -1;dummy->next=NULL;

  return(dummy);
}

void InitMapHead()
{ MapHead=GetMapNode(NULL,NULL,NULL); }

MAPPTR SearchMap(module,chare,name)
{ MAPPTR dummy;

  dummy=MapHead;
  while (dummy!=NULL)
  { if (samename(module,dummy->module) &&
	(samename(chare,dummy->chare)) &&
	(samename(name,dummy->name)))
    	return(dummy);
    else dummy=dummy->next;
  }
  return(NULL);
}

int samename(s,t)
char *s,*t;
{ if (s==t) return(TRUE);
  if ((s==NULL)||(t==NULL)) return(FALSE);
  return(!strcmp(s,t));
}

MAPPTR NewMap(module,chare,name)
char *module,*chare,*name;
{ MAPPTR dummy;
  char   string[2000];
 
  dummy=GetMapNode(module,chare,name);
  dummy->mapid=FUNCTIONCOUNT++;
  if(strcmp(name,"CharmInit")==0) {
    sprintf(string,"%s7%s",CkPrefix_,name);
  } else {
    sprintf(string,"%s%d%s",CkPrefix_,dummy->mapid,name);
  }
  dummy->mappedname=MakeString(string);
  dummy->next=MapHead->next;MapHead->next=dummy;
  return(dummy);
}
  
char *Map(module,chare,name)
char *module,*name,*chare;
{ MAPPTR dummy;

  dummy=SearchMap(module,chare,name);
  if (dummy==NULL) dummy=NewMap(module,chare,name);
  return(dummy->mappedname);
}

/* From end of yaccspec */


void RestoreCurrentTable()
{ if (StackTop!=StackBase) { CurrentTable=StackTop->tableptr; return; }
  if (CurrentChare!=NULL)  { CurrentTable=CurrentChare->type->table; return; }
  if (CurrentModule!=NULL) { CurrentTable=CurrentModule->type->table; return; } 
  CurrentTable=NULL;
}

void SetIdList(type,listptr,name,localid)
TYPEPTR type;
LISTPTR listptr;
int name,localid;
{ LISTPTR ptr;
  SYMTABPTR worksymtab;
  int i=0;

  ptr=listptr;
  if ((type==NULL)||(ptr==NULL)) return;

  do
  { if (ptr->ysn->string!=NULL)
    	{ worksymtab=CheckInsert(ptr->ysn->string,CurrentTable);
	  worksymtab->idtype=name;
	  worksymtab->localid=localid;
	  worksymtab->ysn=ptr->ysn;
	  if ((ptr->ysn->ysn!=NULL) &&(ptr->ysn->ysn->idtype==FUNCTIONTYPE))
		worksymtab->idtype=FNNAME;
	  worksymtab->type=SetType(type);
	  if (type->declflag==NOTDECLARED)
		{ worksymtab->type=type->type;
		  type->type=(TYPEPTR)worksymtab;
		}
 	}
    ptr=ptr->next;
    if (i) dontfree(ptr->prev); else i++;
  } while (ptr!=listptr);
  dontfree(listptr);
}

int CheckDeclaration(newtable,oldtable)
SYMTABPTR newtable,oldtable;
{ int i;

  if (oldtable==NULL) return(0);
  if (CheckDeclaration(newtable,oldtable->left))
	return(1);
  if (CheckDeclaration(newtable,oldtable->right))
	return(1);
  
  FindInTable(newtable,oldtable->name,&i);
  return(i);
}

