#include "xl-sym.h"
#include "xl-lex.h"

extern SYMTABPTR ImportModule;
SYMTABPTR SearchImportModule();

SYMTABPTR CurrentTable=NULL;
STACKPTR StackTop=NULL;
STACKPTR StackBase=NULL;
int CurrentLevel=0;
SYMTABPTR CurrentModule=NULL;
SYMTABPTR CurrentChare=NULL;
TYPEPTR CHARPTR,INTPTR,SHORTPTR,LONGPTR,UNSIGNEDPTR,FLOATPTR,DOUBLEPTR,DUMMYPTR;
TYPEPTR VOIDPTR,CHAREIDPTR,ENTRYPOINTPTR,PENUMPTR,PACKIDPTR,WRITEONCEIDPTR;
TYPEPTR PVECTORPTR,CHARENAMEPTR,CHARENUMPTR,ENTRYNUMPTR,BOOLEANPTR;
TYPEPTR FUNCTIONPTR,FUNCTIONREFPTR; /* Jan 17,1992 Added by Attila */

SYMTABPTR GetSymTabNode(name)
char *name;
{ SYMTABPTR dummy;

  dummy=(SYMTABPTR)calloc(1,sizeof(struct symtabnode));
  if (dummy==NULL) memory_error("Out of Memory in GetSymTabNode()",EXIT);
  dummy->left=dummy->right=NULL;dummy->prev=dummy->next=dummy;
  dummy->modname=dummy->charename=NULL;
  dummy->type=NULL;dummy->ysn=NULL;
  dummy->localid=0;dummy->declflag=DECLARED;dummy->msgno=dummy->userpack=0;
  if (strcmp(name,"")) dummy->name=MakeString(name);
  return(dummy);
}

TYPEPTR GetTypeNode(count,size)
int count,size;
{ TYPEPTR dummy;

  dummy=(TYPEPTR)calloc(1,sizeof(struct typenode));
  if (dummy==NULL) memory_error("Out o Memory in GetTypeNode()",EXIT);
  dummy->table=NULL;
  dummy->count=count;
  dummy->size=size;
  dummy->type=NULL;
  dummy->declflag=DECLARED;
  return(dummy);
}

STACKPTR GetStackNode()
{ STACKPTR dummy;

  dummy = (STACKPTR)calloc(1,sizeof(struct stacknode));
  if (dummy==NULL) memory_error("Out of Memory in GetStackNode()",EXIT);
  dummy->prev=dummy->next=NULL;
  dummy->tableptr=NULL;
  return(dummy);
}

TYPEPTR SysGetTypeNode(count,size,name)
int count,size;
char *name;
{ TYPEPTR dummy;

  dummy=GetTypeNode(count,size);
  dummy->table=GetSymTabNode(name);
  dummy->table->idtype=SYSTEMTYPENAME;
  return(dummy);
}

void InitBasicTypes()
{ CHARPTR=SysGetTypeNode(1,CHARSIZE,"char");
  SHORTPTR=SysGetTypeNode(1,SHORTSIZE,"short");
  INTPTR=SysGetTypeNode(1,INTSIZE,"int");
  FLOATPTR=SysGetTypeNode(1,FLOATSIZE,"float");
  LONGPTR=SysGetTypeNode(1,LONGSIZE,"long");
  DOUBLEPTR=SysGetTypeNode(1,DOUBLESIZE,"double");
  UNSIGNEDPTR=SysGetTypeNode(1,UNSIGNEDSIZE,"unsigned");
  DUMMYPTR=SysGetTypeNode(0,0,"dummy");
  VOIDPTR=SysGetTypeNode(0,0,"void");
  CHAREIDPTR=SysGetTypeNode(0,0,"ChareIDType");
  ENTRYPOINTPTR=SysGetTypeNode(0,0,"EntryPointType");
  PENUMPTR=SysGetTypeNode(0,0,"PeNumType");
  PACKIDPTR=SysGetTypeNode(0,0,"PackIDType");
  WRITEONCEIDPTR=SysGetTypeNode(0,0,"WriteOnceID");
  PVECTORPTR=SysGetTypeNode(0,0,"PVECTOR");
  CHARENAMEPTR=SysGetTypeNode(0,0,"ChareNameType");
  CHARENUMPTR=SysGetTypeNode(0,0,"ChareNumType");
  ENTRYNUMPTR=SysGetTypeNode(0,0,"EntryNumType");
  BOOLEANPTR=SysGetTypeNode(0,0,"BOOLEAN");
  /* Jan 17 1992, Added by Attila */
  FUNCTIONPTR=SysGetTypeNode(0,0,"FUNCTION_PTR");
  FUNCTIONREFPTR=SysGetTypeNode(0,0,"FunctionRefType");
}

void InitSymTable()
{ StackBase=StackTop=GetStackNode();  
  StackTop->level= ++CurrentLevel;
  CurrentTable = StackTop->tableptr = GetSymTabNode(" ");
  CurrentTable->next=CurrentTable->prev=CurrentTable;
  InitBasicTypes();
}

void PushStack()
{ StackTop->next=GetStackNode();
  StackTop->next->prev=StackTop;
  StackTop=StackTop->next;
  StackTop->level = ++CurrentLevel;
  CurrentTable = StackTop->tableptr = GetSymTabNode(" ");
  CurrentTable->next=CurrentTable->prev=CurrentTable;
}

void PopStack(freeflag)
int freeflag;
{ if (freeflag) FreeTree(StackTop->tableptr);
  StackTop=StackTop->prev;
  dontfree(StackTop->next);
  StackTop->next=NULL;
  CurrentLevel--;
  CurrentTable=StackTop->tableptr;
}

void FreeTypeNode(type)
TYPEPTR type;
{ /*
  if (type==NULL) return;
  (type->count)--;
  if (type->count) return;
  if (type->basictype > POINTERTYPE)
	FreeTree(type->table);
  dontfree(type);
  */
}
  
void FreeSymTabStruct(root)
SYMTABPTR root;
{ /*
  if (root==NULL) return;
  dontfree(root->name); 
  FreeTypeNode(root->type);
  dontfree(root);
  */
}

void FreeTree(root)
SYMTABPTR root;
{ /*
  if (root==NULL) return;
  FreeTree(root->left);
  FreeTree(root->right);
  FreeSymTabStruct(root);
  */
}

void FillSymTabNode(node,idtype,declflag,storageclass,basictype,
			count,typeflag,typeptr)
SYMTABPTR node;
TYPEPTR typeptr;
int idtype,declflag,storageclass,basictype,count,typeflag;
{ node->type=GetTypeNode(count,0);
  node->type->basictype=basictype;
  if (typeflag) { node->type->table=GetSymTabNode(" ");
		  node->type->table->next=node->type->table->prev=
			node->type->table;
		}
  node->type->type=typeptr;
  node->idtype=idtype;
  node->declflag=declflag;
  node->storageclass=storageclass;
}

/* test function */

PrintTable(table)
SYMTABPTR table;
{ if (table==NULL) return;
  PrintTable(table->left);
  printf("%s\n",table->name);
  PrintTable(table->right);
}

