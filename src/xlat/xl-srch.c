#include "xl-lex.h"
#include "xl-sym.h"

extern int InPass1;
extern int IMPORTFLAG,ImportLevel;
char *CkGlobalFunctionTable="CsvAccess(_CK_9_GlobalFunctionTable)[";
extern char *Map();
extern char *REFSUFFIX;
extern int FNNAMETOREFFLAG;
extern int READMSGINITFLAG;

char *AppendMap();
char *AppendMapIndex();
char *AppendedString();

IsModule(node)
SYMTABPTR node;
{ return((node!=NULL)&&(node->idtype==MODULENAME)); }

IsChare(node)
SYMTABPTR node;
{ return((node!=NULL)&&((node->idtype==CHARENAME)||(node->idtype==BOCNAME))); }

IsEntry(node)
SYMTABPTR node;
{ return((node!=NULL)&&(node->idtype==ENTRYNAME)); }

IsPublic(node)
SYMTABPTR node;
{ return((node!=NULL)&&(node->idtype==PUBLICFNNAME)); }

IsPrivate(node)
SYMTABPTR node;
{ return((node!=NULL)&&(node->idtype==PRIVATEFNNAME)); }

IsMessage(node)
SYMTABPTR node;
{ return((node!=NULL)&&(node->idtype==MESSAGENAME)); }

IsFunction(node)
SYMTABPTR node;
{ return((node!=NULL)&&(node->idtype==FNNAME)); }

IsReadOnly(node)
SYMTABPTR node;
{ return((node!=NULL)&&((node->idtype==READONLYVAR)||(node->idtype==READONLYMSG)
	 || (node->idtype==READONLYARRAY))); }

IsAccumulator(node)
SYMTABPTR node;
{ return((node!=NULL)&&(node->idtype==ACCNAME)); }

IsMonotonic(node)
SYMTABPTR node;
{ return((node!=NULL)&&(node->idtype==MONONAME)); }

IsTable(node)
SYMTABPTR node;
{ return((node!=NULL)&&(node->idtype==TABLENAME)); }

writefunction(node)
SYMTABPTR node;
{ char *temp,*dummy;

  if (InPass1) return;
  if (READMSGINITFLAG)
	error("Bad ReadInitMsg",EXIT);
  if (!strcmp(node->modname->name,CurrentModule->name))
	{ if (FNNAMETOREFFLAG)
		temp=MyModulePrefix(node->modname->name,node->name);
	  else	temp=MakeString(node->name);
	}
  else  temp=ModulePrefix(node->modname->name,node->name);
  if (FNNAMETOREFFLAG)
	{ dummy=GetMem(strlen(temp)+strlen(REFSUFFIX)+1);
	  strcpy(dummy,temp);strcat(dummy,REFSUFFIX);
	  writeoutput(dummy,FREE);
	}
  else  /*  put the else, Jan 16,1991 Attila */
  writeoutput(temp,NOFREE);
  dontfree(temp);
}

writeentry(node)
SYMTABPTR node;
{ if (FNNAMETOREFFLAG)
	error("Bad FnNameToRef",EXIT);
  if (READMSGINITFLAG)
	error("Bad ReadInitMsg",EXIT);
  if (InPass1) return;
  if (!strcmp(node->modname->name,CurrentModule->name))
	writeoutput(MyModuleCharePrefix(node->modname->name,node->charename->name,
		node->name),FREE); 
  else writeoutput(ModuleCharePrefix(node->modname->name,node->charename->name,
		node->name),FREE); 
}

writepublic(node)
SYMTABPTR node;
{ if (FNNAMETOREFFLAG)
	{ FNNAMETOREFFLAG=FALSE; writeentry(node); return; }
  if (READMSGINITFLAG)
	error("Bad ReadInitMsg",EXIT);
  if (InPass1) return;
  writeoutput(CkGlobalFunctionTable,NOFREE);
  writeentry(node);
  writeoutput("]",NOFREE);
}

writeprivate(node)
SYMTABPTR node;
{ if (FNNAMETOREFFLAG)
	error("Bad FnNameToRef",EXIT);
  if (READMSGINITFLAG)
	error("Bad ReadInitMsg",EXIT);
  if (InPass1) return;
  writeoutput(Map(node->modname->name,node->charename->name,node->name),NOFREE);
}

writechare(node)
SYMTABPTR node;
{ if (FNNAMETOREFFLAG)
	error("Bad FnNameToRef",EXIT);
  if (READMSGINITFLAG)
	error("Bad ReadInitMsg",EXIT);
  if (InPass1) return;
  if (!strcmp(node->modname->name,CurrentModule->name))
	writeoutput(MyModulePrefix(node->modname->name,node->name),FREE); 
  else writeoutput(ModulePrefix(node->modname->name,node->name),FREE); 
}

writereadonly(node)
SYMTABPTR node;
{ if (InPass1) return;
  if (FNNAMETOREFFLAG)
	error("Bad FnNameToRef",EXIT);
  writeoutput(AppendMap(node->modname->name,node->name),NOFREE); 
  if (READMSGINITFLAG)
	{ writeoutput(",",NOFREE);
	  writeoutput(AppendMapIndex(node->modname->name,node->name),NOFREE);
	}
}

writeaccname(node)
SYMTABPTR node;
{ char *temp;

  if (InPass1) return;
  if (FNNAMETOREFFLAG)
	error("Bad FnNameToRef",EXIT);
  if (READMSGINITFLAG)
	error("Bad Msg Init",EXIT);
 
  if (!strcmp(node->modname->name,CurrentModule->name))
	temp=MyModulePrefix(node->modname->name,node->name);
  else 	temp=ModulePrefix(node->modname->name,node->name);
  writeoutput(temp,FREE);
}

writemononame(node)
SYMTABPTR node;
{ char *temp;

  if (InPass1) return;
  if (FNNAMETOREFFLAG)
	error("Bad FnNameToRef",EXIT);
  if (READMSGINITFLAG)
	error("Bad Msg Init",EXIT);
 
  if (!strcmp(node->modname->name,CurrentModule->name))
	temp=MyModulePrefix(node->modname->name,node->name);
  else 	temp=ModulePrefix(node->modname->name,node->name);
  writeoutput(temp,FREE);
}

writetable(node)
SYMTABPTR node;
{ if (InPass1) return;
  if (FNNAMETOREFFLAG)
	error("Bad FnNameToRef",EXIT);
  if (READMSGINITFLAG)
	error("Bad Msg Init",EXIT);
 
  writeoutput(AppendMap(node->modname->name,node->name),NOFREE);
}

SYMTABPTR GlobalModuleSearch(name,modname)
char *name,*modname;
{ SYMTABPTR node,dummy;
  int i;

  node=GlobalFind(modname);
  if (node==NULL) return(NULL);
  dummy=FindInTable(node->type->table,name,&i);
  if (i==0) return(dummy);
  if (InPass1) return(NULL);
  if (!strcmp(modname,Pass1Module->name))
	{ dummy=FindInTable(Pass1Module->type->table,name,&i);
	  if (i==0) return(dummy);
	}
  return(NULL);
}

SYMTABPTR GlobalEntryFind(entryname,charename,modname)
char *charename,*entryname,*modname;
{ SYMTABPTR node,dummy;
  int i;

  node=GlobalModuleSearch(charename,modname);
  if (node==NULL) return(NULL);
  dummy=FindInTable(node->type->table,entryname,&i);
  if (i==0) return(dummy);
  if (InPass1) return(NULL);
  if (!strcmp(modname,Pass1Module->name))
	{ node=FindInTable(Pass1Module->type->table,charename,&i);
	  if (i!=0) return(NULL);
	  dummy=FindInTable(node->type->table,entryname,&i);
	  if (i==0) return(dummy);
	}
  return(NULL);
}

SYMTABPTR FindInTable(root,name,i)
SYMTABPTR root;
char *name;
int *i;
{ int temp;
 
  if (root==NULL) error("Searching for identifier in emptiness",EXIT); 
  temp=strcmp(name,root->name);
  switch ((temp==0)?0:(temp<0)?-1:1)
  { case  0 : *i=0;return(root);
    case -1 : if (root->left==NULL)
		   { *i = -1; return(root); }
              else return(FindInTable(root->left,name,i));
    case  1 : if (root->right==NULL)
		   { *i = 1; return(root); }
              else return(FindInTable(root->right,name,i));
    default : error("What is this?",EXIT);
  }
}

SYMTABPTR LocalFind(name)
char *name;
{ SYMTABPTR dummy;
  STACKPTR current;
  int i;

  current=StackTop;
  while (current!=StackBase)
  { dummy=FindInTable(current->tableptr,name,&i);
    if (i==0) return(dummy);
    current=current->prev;
  }
  if (CurrentChare!=NULL)
	{ dummy=FindInTable(CurrentChare->type->table,name,&i);
	  if (i==0) return(dummy);
	}
  if (CurrentModule!=NULL)
	{ dummy=FindInTable(CurrentModule->type->table,name,&i);
	  if (i==0) return(dummy);
	}
  dummy=FindInTable(StackBase->tableptr,name,&i);
  if (i==0) return(dummy);
  return(NULL);
}

SYMTABPTR GlobalFind(name)
char *name;
{ SYMTABPTR dummy;
  int i;

  dummy=LocalFind(name);
  if ((dummy==NULL)&&(Pass1Module!=NULL))
	{ if (CurrentChare!=NULL)
		{ dummy=FindInTable(Pass1Module->type->table,CurrentChare->name,
					&i);
		  if (i==0) dummy=FindInTable(dummy->type->table,name,&i);
		  if (i==0) return(dummy);
		}
	  dummy=FindInTable(Pass1Module->type->table,name,&i);
	  if (i==0) return(dummy); else return(NULL);
	}
  else return(dummy);
}

SYMTABPTR Insert(name,table)
char *name;
SYMTABPTR table;
{ SYMTABPTR dummy,temp;
  int i;

  if (table==NULL) error("Trying to insert into NULL",EXIT);
  dummy=FindInTable(table,name,&i);
  if (i==0) { 
	      error("Duplicate Entry: ",NOEXIT); 
              PutOnScreen(name);
              PutOnScreen("\n");
              return(dummy);
             }
  temp=GetSymTabNode(name);
  if (i==1) dummy->right=temp; else dummy->left=temp;
  temp->prev=table->prev;
  table->prev=temp;
  temp->next=table;
  temp->prev->next=temp;
  temp->modname=CurrentModule;
  temp->charename=CurrentChare;
  if (IMPORTFLAG) temp->level=ImportLevel;
  return(temp);
}

int IDType(name)
char *name;
{ SYMTABPTR dummy;

  dummy=LocalFind(name); 
  if (dummy==NULL) return(0);
  return(dummy->idtype);
}

InChareEnv(name,ptr)
char *name;
SYMTABPTR ptr;
{ int i;
  SYMTABPTR dummy;

  if (CurrentChare==NULL) return(0);
  dummy=FindInTable(CurrentChare->type->table,name,&i);
  if ((!i)&&(dummy==ptr)) return(1); else return(0);
}


InModuleEnv(name,ptr)
char *name;
SYMTABPTR ptr;
{ int i;
  SYMTABPTR dummy;

  if (CurrentModule==NULL) return(0);
  dummy=FindInTable(CurrentModule->type->table,name,&i);
  if ((!i)&&(dummy==ptr)) return(1); else return(0);
}

SYMTABPTR CheckInsert(name,table)
{ SYMTABPTR dummy;
  int i;

  dummy=FindInTable(table,name,&i);
  if (i==0) return(dummy);
  return(Insert(name,table));
}

char *AppendMap(modname,name)
char *modname,*name;
{ return(AppendedString(modname,name,"20")); }

char *AppendMapIndex(modname,name)
char *modname,*name;
{ return(AppendedString(modname,name,"21")); }

char *AppendedString(modname,name,padding)
char *modname,*name,*padding;
{ static char *temp=NULL;
 
  if (temp!=NULL) dontfree(temp);

  temp=GetMem(strlen(modname)+strlen(name)+2*strlen(CkPrefix)+1+strlen(padding));
  strcpy(temp,CkPrefix);
  strcat(temp,padding);
  strcat(temp,modname);
  strcat(temp,CkPrefix);
  strcat(temp,name);
  return(temp);
}
