%{
#include "xl-lex.h"
#include "xl-sym.h"

extern int CurrentInputLineNo,SavedLineNo;
extern char CurrentFileName[],SavedFileName[];
extern char ModuleName[];

#define USERDEFINED 1
#define SYSTEMDEFINED 2

#define YYSTYPE YSNPTR
#define YYMAXDEPTH 300

#define dump(string)	writeoutput(string,NOFREE)

extern char *AppendMap(),*AppendMapIndex();  
char token[256];

BUFFERTYPE buffer;

extern void RestoreCurrentTable();
extern void SetIdList();
extern void GenerateStruct();		/* outh.c */
extern void InsertSysCalls();		/* readtokens.c */
extern TYPEPTR ProcessTypeIdentifier();

char *dummy_ptr=NULL;
SYMTABPTR worksymtab,SavedFnNode;
SYMTABPTR ImportModule=NULL;
SYMTABPTR ModuleDefined=NULL;
SYMTABPTR Pass1Module=NULL;
SYMTABPTR sym1,sym2;
TYPEPTR SavedTypeNode;
char *SavedName,*itoa(),*Map(),*AccMonoName,*AccMonoTypeName;
char *ForwardedMsgName;
int SaveMsgCount;

SYMTABPTR GlobalModuleSearch(),GlobalEntryFind(),ProcessIdDcolonId();
void SwapFile();

char *CkMyData="_CK_4mydata->";
char *CkLocalPtr="_CK_4localptr";
char *CkDummyPtr="_CK_4dummyptr";
char *CkMsgQualifier="MSG";
char *DataSuffix="_Data";
char *AssignMyDataPtr=" *_CK_4mydata = (";
char *AssignMyID="\nChareIDType ThisChareID=((CHARE_BLOCK *)_CK_4mydata)[-1].selfID;";
char *AssignMyBOC="\nint ThisBOC=((CHARE_BLOCK *)_CK_4mydata)[-1].x.boc_num;";
char *DummyField="  char _CK;";
extern char *REFSUFFIX;		/* outh.c */
extern char *CkGenericAlloc,*CkVarSizeAlloc;


int INSIDE_PRIVFUNC_PROTO=0; /* Jun 14, 1995, added by Attila */
int PRIVATEFLAG=0;
int StoredEPFlag;
int IMPORTFLAG=0;
int IMPORTMSGFLAG=0; /* Dec 30, 1991 Added by Attila */
int SELFIMPMODULE=0; /* Jan 2, 1992  Added by Attila */
int MULTI_STMT_TR=0; /* Feb 18, 1992  Added by Attila */
int export_to_c_flag=0; /* March, 1994  Added by Attila */
int ImportStruct=0;
int ImportLevel=0;
int OUTPUTOFF=0;
int FUNCTIONCOUNT=0;
int FNNAMETOREFFLAG=0;
int FNREFTONAMEFLAG=0; /* Jan 16, 1992 Added by Attila */
int READMSGINITFLAG=0;
int BUFFEROUTPUT=0;
int FORWARDEDMESSAGE=0;
int MsgCount=10;	/* ARBITRARY - Set to > 1 */
int MsgToStructFlag=0;
FILE *outfile;

TYPEPTR SetType(typeptr)
TYPEPTR typeptr;
{ (typeptr->count)++;
  return(typeptr);
}

void RecursiveFree(typeptr)
TYPEPTR typeptr;
{ if (typeptr->type != NULL) RecursiveFree(typeptr->type);
  if (typeptr->table != NULL) FreeTree(typeptr->table); 
  dontfree(typeptr);
}

void FreeType(typeptr)
TYPEPTR typeptr;
{ if (typeptr==NULL) return;
  if (--(typeptr->count) <= 0) RecursiveFree(typeptr); }

%}
%token CHARE
%token MAIN
%token MESSAGE
%token ENTRY
%token PRIVATE
%token DATAINIT
%token CHAREINIT
%token QUIESCENCE
%token MODULE
%token BRANCHOFFICE
%token READONLY
%token MONOTONIC
%token TABLE
%token ACCUMULATOR
%token STATIC
%token PACK
%token UNPACK
%token VARSIZE
%token FUNCTION
%token BRANCH
%token SIZEOF
%token AUTO
%token BREAK
%token CASE
%token CHAR
%token CONST
%token CONTINUE
%token DEFAULT
%token DO
%token DOUBLE
%token ELSE
%token ENUM
%token EXTERN
%token FLOAT
%token FOR
%token GOTO
%token IF
%token INT
%token LONG
%token REGISTER
%token RETURN
%token SHORT
%token SIGNED
%token STRUCT
%token SWITCH
%token TYPEDEF
%token UNION
%token UNSIGNED
%token VOID
%token WHILE
%token INTERFACE
%token ChareIDType 
%token EntryPointType 
%token PeNumType     
%token PackIDType   
%token WriteOnceID
%token PVECTOR  
%token ChareNumType
%token EntryNumType
%token BOOLEAN

%token R_PAREN
%token R_SQUARE
%token L_BRACE
%token R_BRACE
%token SEMICOLON
%token IDENTIFIER
%token TYPE_IDENTIFIER
%token EPIDENTIFIER
%token MCIDENTIFIER
%token STRING
%token NUMBER
%token CHAR_CONST
%token COMMA_R_BRACE
%token COMMA_L_BRACE
%token AT

%left COMMA
%right ASGNOP EQUAL
%left QUESTION COLON
%left OROR
%left ANDAND
%left OR
%left HAT
%left AND
%left EQUALEQUAL NOTEQUAL
%left COMPARE
%left SHIFT
%left PLUS MINUS
%left MULT DIV MOD
%left UNARY INCDEC TILDE EXCLAIM
%left L_PAREN L_SQUARE DOT POINTERREF

%token PRIVATECALL
%token BRANCHCALL
%token ID_DCOLON_ID
%token PUBLIC
%token FNNAMETOREF
%token CKALLOCMSG
%token CKALLOCPRIOMSG
%token READMSGINIT
%token ACCUMULATE
%token NEWVALUE
%token ACCIDTYPE
%token MONOIDTYPE
%token DUMMYMSG
%token FunctionRefType
%token FUNCTION_PTR
%token FNREFTONAME
%token IMPLICIT
%token BLOCKEDRECV
%token DAG
%token MATCH
%token AUTOFREE
%token WHEN
%token CHARECALL
%token ChareNameType
%token EXPORT_TO_C
%start program

%%
program		: file_text
			{ if (yylex()) 
			     warning("Ignoring Extraneous Characters at End"); 
			  return(0); 
			}
		;

file_text	: module_text
			{ CurrentTable=StackBase->tableptr;
			  if (InPass1)
				{Pass1Module=GetSymTabNode(CurrentModule->name);
				 Pass1Module->type=CurrentModule->type;
				} 
			}
		| import_text module_text
			{ CurrentTable=StackBase->tableptr;
			  if (InPass1)
				{Pass1Module=GetSymTabNode(CurrentModule->name);
				 Pass1Module->type=CurrentModule->type;
				} 
			}
		;

import_text	: import_decl	
			{ CurrentTable=StackBase->tableptr;
			  CurrentModule=NULL;
			}
		| import_text import_decl
			{ CurrentTable=StackBase->tableptr;
			  CurrentModule=NULL;
			}
		;

import_decl	: INTERFACE	{ IMPORTFLAG=TRUE; ImportLevel=0;}
		  MODULE IDENTIFIER 
			{ CurrentModule=Insert($4->string,StackBase->tableptr); 
			  FillSymTabNode(CurrentModule,MODULENAME,IMPORTED,
				UNDEFINED,UNDEFINED,1,TRUE,NULL);
			  CurrentTable=CurrentModule->type->table;
/* Adding On Oct. 27, 1991.*/
			  SwapFile(OUT1);
			  if (strcmp($4->string,ModuleName)) {
                                SELFIMPMODULE = 0;
			  	writeoutput("\nextern struct { ",NOFREE);
                              } 
			  else{
                                SELFIMPMODULE = 1;
                                writeoutput("\nstruct { ",NOFREE);
                          }
			  SwapFile(NULL);
			  DestroyString($4->string);dontfree($4);
			}
		  L_BRACE import_constructs R_BRACE
			{ SwapFile(OUT1);writeoutput("} ",NOFREE);
			  writeoutput(CkPrefix,NOFREE);
			  writeoutput(CurrentModule->name,NOFREE);
			  writeoutput(" ;",NOFREE);WriteReturn();
			  SwapFile(NULL);
			  IMPORTFLAG=FALSE;
			}
		;
 
import_constructs : imp_construct
		| import_constructs imp_construct
		;

ITypeDef	: TYPEDEF { SwapFile(OUT0);writeoutput("typedef ",NOFREE); }
		;

imp_construct	: ITypeDef
		  type_spec init_decl_list semicolon
			  { SetIdList($2->type,$3->listptr,TYPENAME,0);
			    dontfree($2);dontfree($3);
	                    WriteReturn();	
			    SwapFile(NULL);
			  }
		| ITypeDef
		  init_decl_list semicolon
			  { SetIdList(DUMMYPTR,$2->listptr,TYPENAME,0);
			    dontfree($2);
	                    WriteReturn();	
			    SwapFile(NULL);
			  }
		| imp_fn_decl 
			{ SwapFile(NULL); }
		| imp_message_defn
			{ SwapFile(NULL); }
		| imp_acc
			{ SwapFile(NULL); }
		| imp_mono
			{ SwapFile(NULL); }
		| imp_chare_defn
			{ SwapFile(NULL); }
		| imp_main_chare_defn
			{ SwapFile(NULL); }
		| imp_boc_defn
			{ SwapFile(NULL); }
		| { SwapFile(OUT0);  /* Jan 2 1992, Attila */
                    if (SELFIMPMODULE) writeoutput("/* ",NOFREE);} 
		  imp_readonly 
		  { if (SELFIMPMODULE) {
                       writeoutput("*/",NOFREE);
              /*       WriteReturn(); Dec 15, 1993*/
                    }
                    WriteReturn(); /* Dec 15,1993 Attila */
                    SwapFile(NULL); }
		| { SwapFile(OUT0); /* Jan 6 1992, Attila */
                    if (SELFIMPMODULE) writeoutput("/* ",NOFREE);}
		  table_defn
                  { if (SELFIMPMODULE) {
                       writeoutput("*/",NOFREE);
              /*       WriteReturn(); Dec 15 1993 */
                    } 
                    WriteReturn(); /* Dec 15, 1993 */
		    SwapFile(NULL); }
		;

imp_acc		: ACCUMULATOR { SwapFile(OUT1); }
		  aid_list SEMICOLON
			{ writeoutput(";",NOFREE);WriteReturn(); }
		;

imp_mono	: MONOTONIC { SwapFile(OUT1); }
		  mid_list SEMICOLON
			{ writeoutput(";",NOFREE);WriteReturn(); }
		;

aid_list	: IDENTIFIER 
			{ worksymtab=Insert($1->string,CurrentTable);
			  worksymtab->idtype=ACCNAME;
			  writeoutput("int ",NOFREE);
			  writeoutput($1->string,NOFREE);
			  DestroyString($1->string);dontfree($1);
			}
		| aid_list COMMA IDENTIFIER
			{ worksymtab=Insert($3->string,CurrentTable);
			  worksymtab->idtype=ACCNAME;
			  writeoutput(",",NOFREE);
			  writeoutput($3->string,NOFREE);
			  DestroyString($3->string);dontfree($3);
			}
		;

mid_list	: IDENTIFIER
			{ worksymtab=Insert($1->string,CurrentTable);
			  worksymtab->idtype=MONONAME;
			  writeoutput("int ",NOFREE);
			  writeoutput($1->string,NOFREE);
			  DestroyString($1->string);dontfree($1);
			}
		| mid_list COMMA IDENTIFIER
			{ worksymtab=Insert($3->string,CurrentTable);
			  worksymtab->idtype=MONONAME;
			  writeoutput(",",NOFREE);
			  writeoutput($3->string,NOFREE);
			  DestroyString($3->string);dontfree($3);
			}
		;

imp_readonly	: readonly
		;

ID_L_R_S	: IDENTIFIER L_PAREN R_PAREN SEMICOLON
			{ worksymtab=Insert($1->string,CurrentTable);
			  worksymtab->idtype=FNNAME;
			  writeoutput(" (* ",NOFREE);
			  writeoutput($1->string,NOFREE);
			  writeoutput(")();\n  int ",NOFREE);
			  writeoutput($1->string,NOFREE);
			  writeoutput(REFSUFFIX,NOFREE);
			  writeoutput(";",NOFREE);WriteReturn();
			  DestroyString($1->string);dontfree($1);
			}
		;

imp_fn_decl	: { SwapFile(OUT1); writeoutput("  int",NOFREE); } ID_L_R_S
		| { SwapFile(OUT1); writeoutput(" ",NOFREE); }
		  system_types {writeoutput($2->string);} ID_L_R_S
		| TYPE_IDENTIFIER
			{
                          SwapFile(OUT1);
                          writeoutput("  ",NOFREE); 
			  writeoutput(Map(CurrentModule->name,"0",
						$1->string),NOFREE);
			  DestroyString($1->string);dontfree($1);
			}
		  ID_L_R_S
		| {SwapFile(OUT1); writeoutput("  ",NOFREE);} su_spec ID_L_R_S
		;
	
optID		: 	
		| IDENTIFIER	
			{ writeoutput(Map(CurrentModule->name,"0",$1->string),NOFREE);
			  worksymtab=Insert($1->string,CurrentTable);
			  worksymtab->idtype=STRUCTNAME;
			  DestroyString($1->string);dontfree($1);
	/* NOT BOTHERING TO SET ITS TYPE */
			}
		;

imp_message_defn: MESSAGE 	{ SwapFile(OUT0);
				  writeoutput("typedef struct ",NOFREE);
				  ImportLevel=0;
                                  /* Dec 30 ,1991 added by Attila */
                                  IMPORTMSGFLAG = 1;
                                  /* End of Addition */
				}
		  optID L_BRACE { PushStack(); writeoutput(" {",NOFREE); }
		  msg_opt_su_rest R_BRACE IDENTIFIER SEMICOLON
			{ TYPEPTR typeptr;
                          /* Added by Attila, Dec 23 1991 */
                                  SwapFile(OUT1);
                                  writeoutput("    int ",NOFREE);
                                  writeoutput($8->string,NOFREE);
                                  writeoutput(";",NOFREE);
                                  WriteReturn();
                                  SwapFile(OUT0);
                          /* End Of Addition */
			  writeoutput(" } ",NOFREE);
			  typeptr=GetTypeNode(1,0);
			  typeptr->basictype=STRUCTTYPE;
			  typeptr->table=CurrentTable;
			  PopStack(NOFREE);RestoreCurrentTable();
			  worksymtab=Insert($8->string,CurrentTable);
			  worksymtab->idtype=MESSAGENAME;
			  worksymtab->type=typeptr;
 /*		  writeoutput(Map(CurrentModule,"0",$8->string),NOFREE); */
 /* The statement above has been modified as follows, Attila Dec 23 1991 */
                          writeoutput(Map(CurrentModule->name,
                                          "0",$8->string),NOFREE);
			  writeoutput(";",NOFREE);WriteReturn();
			  DestroyString($8->string);dontfree($8);
                          IMPORTMSGFLAG = 0; /* Dec 30,1991 added by Attila */
			}
		;

imp_chare_defn	: CHARE IDENTIFIER
			{ SwapFile(OUT1);
			  writeoutput("  struct {",NOFREE);
			  SwapFile(NULL);
			  worksymtab=Insert($2->string,CurrentTable);
			  FillSymTabNode(worksymtab,CHARENAME,IMPORTED,
				UNDEFINED,UNDEFINED,1,TRUE,NULL);
			  CurrentTable=worksymtab->type->table;
			  CurrentChare=worksymtab;
			}
		  L_BRACE entry_pub_fn_list R_BRACE
			{ SwapFile(OUT1);
			  writeoutput("  } ",NOFREE);
			  writeoutput(CkPrefix,NOFREE);
			  writeoutput($2->string,NOFREE);
			  writeoutput(";\n  int ",NOFREE);
			  writeoutput($2->string,NOFREE);
			  writeoutput(";",NOFREE);WriteReturn();
			  SwapFile(NULL);
			  DestroyString($2->string);dontfree($2);
			  CurrentTable=CurrentModule->type->table;
			  CurrentChare=NULL;
			}
		;

imp_boc_defn	: BRANCHOFFICE IDENTIFIER
			{ SwapFile(OUT1);
			  writeoutput("  struct {",NOFREE);
			  worksymtab=Insert($2->string,CurrentTable);
			  FillSymTabNode(worksymtab,BOCNAME,IMPORTED,
				UNDEFINED,UNDEFINED,1,TRUE,NULL);
			  CurrentTable=worksymtab->type->table;
			  CurrentChare=worksymtab;
			}
		  L_BRACE entry_pub_fn_list R_BRACE
			{ SwapFile(OUT1);
			  writeoutput("  } ",NOFREE);
			  writeoutput(CkPrefix,NOFREE);
			  writeoutput($2->string,NOFREE);
			  writeoutput(";\n  int ",NOFREE);
			  writeoutput($2->string,NOFREE);
			  writeoutput(";",NOFREE);WriteReturn();
			  DestroyString($2->string);dontfree($2);
			  CurrentTable=CurrentModule->type->table;
			  CurrentChare=NULL;
			}
		;

imp_main_chare_defn : CHARE MAIN 
			{ SwapFile(OUT1);
			  writeoutput("  struct {",NOFREE);
                          /* Jan 6, 1992, added by Attila */
                          worksymtab=Insert("main",CurrentTable);
                          FillSymTabNode(worksymtab,CHARENAME,IMPORTED,
                                UNDEFINED,UNDEFINED,1,TRUE,NULL);
                          CurrentTable=worksymtab->type->table;
                          CurrentChare=worksymtab;
                          /* end of addition */
			}
		  L_BRACE entry_list R_BRACE
			{ /* Jan 6 1992 , Attila */ SwapFile(OUT1);
                          writeoutput("  } ",NOFREE);
			  writeoutput(CkPrefix,NOFREE);
			  writeoutput("main",NOFREE);
			  writeoutput(";\n  int ",NOFREE);
			  writeoutput("main",NOFREE);
			  writeoutput(";",NOFREE);WriteReturn();
                          /* Jan 6 1992, added by Attila */
			  CurrentTable=CurrentModule->type->table;
			  CurrentChare=NULL;
                          /* end of addition */
			}
		;

EIS		: ENTRY IDENTIFIER SEMICOLON
			{ SwapFile(OUT1);
			  writeoutput("    int ",NOFREE);
			  writeoutput($2->string,NOFREE);
		 	  writeoutput(";",NOFREE);WriteReturn();
			  SwapFile(NULL);
			  worksymtab=Insert($2->string,CurrentTable);
			  worksymtab->idtype=ENTRYNAME;
			  worksymtab->declflag=IMPORTED;
			  DestroyString($2->string);dontfree($2);
			}
		;

entry_list	: EIS
		| entry_list EIS
		;

ILRS		: IDENTIFIER L_PAREN R_PAREN SEMICOLON
			{ OUTPUTOFF=FALSE;SwapFile(OUT1);
			  writeoutput("    int ",NOFREE);
			  writeoutput($1->string,NOFREE);
			  writeoutput(";",NOFREE);WriteReturn();
			  worksymtab=Insert($1->string,CurrentTable);
			  worksymtab->idtype=PUBLICFNNAME;
			  worksymtab->declflag=IMPORTED;
			  DestroyString($1->string);dontfree($1);
			}

PUBFN		: PUBLIC 	{ OUTPUTOFF=TRUE; }
		  type_spec ILRS
		| PUBLIC ILRS
		;
entry_pub_fn_list : EIS
		| PUBFN
		| entry_pub_fn_list EIS
		| entry_pub_fn_list PUBFN
		;

module_text	: MODULE IDENTIFIER
			{ worksymtab=LocalFind($2->string);
			  InPass1 = !InPass1;
			  if (!InPass1) SwapFile(OUT); else SwapFile(NULL);
			  SavedLineNo=CurrentInputLineNo;
			  strcpy(SavedFileName,CurrentFileName);	
			  if (worksymtab==NULL)
				CurrentModule=Insert($2->string,StackBase->tableptr); 
			  else  { if (InPass1)
				  { CurrentModule=worksymtab;
				    GenerateStruct($2->string,
				     worksymtab->type->table,TRUE,CkPrefix,0);
				    ImportModule=worksymtab->type->table;
				  }
				}
			  FillSymTabNode(CurrentModule,MODULENAME,DECLARED,
				UNDEFINED,UNDEFINED,1,TRUE,NULL);
			  CurrentTable=CurrentModule->type->table;
		 	  ModuleDefined=CurrentModule;
			  DestroyString($2->string);dontfree($2);
			  InsertSysCalls();
			  _dag_init(CurrentModule->name);
			}
		  L_BRACE mod_constructs R_BRACE
		;

mod_constructs	: construct
		| mod_constructs construct
		;

ext_const	: IDENTIFIER
			{ writeoutput($1->string,NOFREE);
			  worksymtab = Insert($1->string,
						CurrentModule->type->table);
			  worksymtab->idtype=VARNAME;
			  worksymtab->type=INTPTR;
			  DestroyString($1->string);dontfree($1); }
		| IDENTIFIER {writeoutput($1->string,NOFREE);}
                  L_SQUARE {writeoutput("[",NOFREE);}
                  ext_constant 
                  R_SQUARE 
                          {writeoutput("]",NOFREE);
			  worksymtab = Insert($1->string,
						CurrentModule->type->table);
			  worksymtab->idtype=ARRAYNAME;
			  worksymtab->type=INTPTR;
			  DestroyString($1->string);dontfree($1); }
		| IDENTIFIER L_PAREN R_PAREN
			{ writeoutput($1->string,NOFREE);
			  writeoutput("()",NOFREE);
			  worksymtab = Insert($1->string,
						CurrentModule->type->table);
			  DestroyString($1->string);dontfree($1);
			  worksymtab->idtype=SYSCALLNAME;
			  worksymtab->type=INTPTR;
			}
        | IDENTIFIER L_PAREN
            { writeoutput($1->string, NOFREE);
              writeoutput("(", NOFREE);
            }
            ext_parameter_list R_PAREN
            { worksymtab = Insert($1->string,
                    CurrentModule->type->table);
              DestroyString($1->string);dontfree($1);
              worksymtab->idtype=SYSCALLNAME;
              worksymtab->type=INTPTR;
              writeoutput(")", NOFREE);
            }
		;

ext_parameter_list : typedef_name {
              ;
            }
        | typedef_name IDENTIFIER {
              writeoutput($1->string, NOFREE);
            }
        | ext_parameter_list comma typedef_name {
             /* writeoutput($3->string, NOFREE);*/
            }
        | ext_parameter_list comma typedef_name IDENTIFIER {
              writeoutput($4->string, NOFREE);
            }
        ;


ext_constant	: constant
		|
                ;
ext_list	: mopt {if($1!=NULL) {writeoutput("*",NOFREE);dontfree($1);} }
                  ext_const
                      
		| ext_list comma mopt 
                       {if($3!=NULL) {writeoutput("*",NOFREE);dontfree($3);}}
                  ext_const
		;

construct	: TypeDef type_spec init_decl_list semicolon
			{ SetIdList($2->type,$3->listptr,TYPENAME,0);
			  dontfree($2);dontfree($3);
			}
		| TypeDef init_decl_list semicolon
			  { SetIdList(DUMMYPTR,$2->listptr,TYPENAME,0);
			    dontfree($2);
			  }
		| module_su_spec module_init_decl semicolon 
		    {  SetIdList($1->type,$2->listptr,VARNAME,0);
		       dontfree($2); dontfree($1);
                    }
		| EXTERN type_spec ext_list semicolon
		| EXTERN ext_list semicolon
		| function_decl semicolon
			    { worksymtab=$1->table;
			      if (worksymtab->localid==0)
					worksymtab->idtype=SYSCALLNAME;
				PopStack(FREE);PopStack(FREE);
				RestoreCurrentTable(); }
		| function_decl function_body
		  { worksymtab=$1->table;
 		    worksymtab->localid=1;
		    dontfree($1); RestoreCurrentTable();} 
		| EXPORT_TO_C
			{ export_to_c_flag = 1;}
		  function_decl
			{ export_to_c_flag = 0;}
		  function_body
			{ worksymtab=$3->table;
			  worksymtab->localid=1;
			  dontfree($3); RestoreCurrentTable();
			}
		| message_defn
		| acc_defn
		| mono_defn
		| chare_defn
		| dag_chare_defn
		| boc_defn
		| dag_boc_defn
		| main_chare_defn
		| readonly
		| table_defn
		| const_declaration
		;


const_declaration : CONST typename IDENTIFIER
            {
              writeoutput($3->string, NOFREE);
                      writeoutput("=", NOFREE);
            }
            EQUAL expression SEMICOLON
            { writeoutput(";", NOFREE); }




/* Feb 18, 1992 added by Attila */
module_su_spec	: Struct IDENTIFIER { if (IMPORTFLAG) ImportStruct=TRUE;
			  	      if (ImportStruct)
						writeoutput(Map(CurrentModule->name,"_CKTYPE",$2->string),NOFREE); 
				      else	writeoutput($2->string,NOFREE); 
				      ImportStruct=FALSE;
				    }
	  	  su_rest
			{ SYMTABPTR dummy,temp;int i;TYPEPTR type;

			  dummy=FindInTable(CurrentTable,$2->string,&i);
			  if (i!=0) 
				{ worksymtab=Insert($2->string,CurrentTable);
				  worksymtab->type=GetTypeNode(1,0);
				}
			  else 	{ if ((dummy->declflag!=NOTDECLARED) ||
					(dummy->idtype!=STRUCTNAME))
					error("Bad Struct Name",EXIT);
				  worksymtab=dummy;
				  type=GetTypeNode(1+worksymtab->type->count,0);
				  dummy=(SYMTABPTR)worksymtab->type->type;
				  while (dummy!=NULL)
					{ temp=(SYMTABPTR)dummy;
					  dummy->type=type;dummy=temp;
                                          /* this is an infinite loop */
                                          /* set dummy to NULL */
                                          dummy = NULL;
					}
				  dontfree(worksymtab->type);
				  worksymtab->type=type;
				}
	
			  worksymtab->type->basictype=STRUCTTYPE;
 			  worksymtab->idtype=STRUCTNAME;
			  worksymtab->declflag=DECLARED;
			  worksymtab->type->table=$4->table;
			  $$=GetYSN();$$->type=worksymtab->type;
		 	  DestroyString($2->string);dontfree($2);dontfree($4);
			}
		| Union IDENTIFIER 	{ if (IMPORTFLAG) ImportStruct=TRUE;
					  if (ImportStruct)
			  			writeoutput(Map(CurrentModule->name,"_CKTYPE",$2->string),NOFREE); 
					  else 	writeoutput($2->string,NOFREE); 
					  ImportStruct=FALSE;
					}
		  su_rest
			{ SYMTABPTR dummy,temp;int i;TYPEPTR type;

			  
			  {
			  dummy=FindInTable(CurrentTable,$2->string,&i);
			  if (i!=0) 
				{ worksymtab=Insert($2->string,CurrentTable);
				  worksymtab->type=GetTypeNode(1,0);
				}
			  else 	{ if ((dummy->declflag!=NOTDECLARED) ||
					(dummy->idtype!=UNIONNAME))
					error("Bad Struct Name",EXIT);
				  worksymtab=dummy;
				  type=GetTypeNode(1+worksymtab->type->count,0);
				  dummy=(SYMTABPTR)worksymtab->type->type;
				  while (dummy!=NULL)
					{ temp=(SYMTABPTR)dummy;
					  dummy->type=type;dummy=temp;
                                          /* this is an infinite loop */
                                          /* set dummy to NULL */
                                          dummy = NULL;
					}
				  dontfree(worksymtab->type);
				  worksymtab->type=type;
				}
	
			  worksymtab->type->basictype=UNIONTYPE;
 			  worksymtab->idtype=UNIONNAME;
			  worksymtab->declflag=DECLARED;
			  worksymtab->type->table=$4->table;
			  $$=GetYSN();$$->type=worksymtab->type;
		 	  DestroyString($2->string);dontfree($2);dontfree($4); 
			  }
			}
		| Enum IDENTIFIER 
			{ SYMTABPTR dummy; int i;

			  if (IMPORTFLAG) ImportStruct=TRUE;
			  if (!ImportStruct)
				writeoutput($2->string,NOFREE);  
			  else 	writeoutput(Map(CurrentModule->name,"_CKTYPE",$2->string),NOFREE); 
                          writeoutput(" ",NOFREE);/*Feb18,1992 added by Attila*/
			  ImportStruct=FALSE;
			  $$=GetYSN();
			  dummy=FindInTable(CurrentTable,$2->string,&i);
			  if (i==0) 
				worksymtab=dummy;
			  else 	{ worksymtab=LocalFind($2->string);
			  	  if ((worksymtab==NULL)||
                                      (worksymtab->idtype!=ENUMNAME))
				  	{ worksymtab =
					  	Insert($2->string,CurrentTable);
					  worksymtab->type=GetTypeNode(0,1);
					  worksymtab->idtype=ENUMNAME;
					  worksymtab->declflag=NOTDECLARED;
					  worksymtab->type->declflag=NOTDECLARED;
					  worksymtab->type->type=NULL;
					}
				}
				  
			  if (worksymtab->idtype!=ENUMNAME)
				error("Bad Struct Name",EXIT);
			  $$->type=worksymtab->type; 
			  DestroyString($2->string);dontfree($2);
			}
		| Enum IDENTIFIER { if (IMPORTFLAG) ImportStruct=TRUE;
			  	      if (ImportStruct)
						writeoutput(Map(CurrentModule->name,"_CKTYPE",$2->string),NOFREE); 
				      else	writeoutput($2->string,NOFREE); 
                                      writeoutput(" ",NOFREE); /* Feb 18,1992*/
				      ImportStruct=FALSE;
				    }
	  	  enum_rest 
			{ SYMTABPTR dummy,temp;int i;TYPEPTR type;

			  dummy=FindInTable(CurrentTable,$2->string,&i);
			  if (i!=0) 
				{ worksymtab=Insert($2->string,CurrentTable);
				  worksymtab->type=GetTypeNode(1,0);
				}
			  else 	{ if ((dummy->declflag!=NOTDECLARED) ||
					(dummy->idtype!=ENUMNAME))
					error("Bad Struct Name",EXIT);
				  worksymtab=dummy;
				  type=GetTypeNode(1+worksymtab->type->count,0);
				  dummy=(SYMTABPTR)worksymtab->type->type;
				  while (dummy!=NULL)
					{ temp=(SYMTABPTR)dummy;
					  dummy->type=type;dummy=temp;
                                          /* this is an infinite loop */
                                          /* set dummy to NULL */
                                          dummy = NULL;
					}
				  dontfree(worksymtab->type);
				  worksymtab->type=type;
				}
	
			  worksymtab->type->basictype=ENUMTYPE;
 			  worksymtab->idtype=ENUMNAME;
			  worksymtab->declflag=DECLARED;
			  worksymtab->type->table=$4->table;
			  $$=GetYSN();$$->type=worksymtab->type;
		 	  DestroyString($2->string);dontfree($2);dontfree($4);
			}
		;

module_init_decl	: {$$=GetYSN();}
                        ;

/*                                      */




decl_specs	: type_spec		{ $$=$1; }
		| sc_spec		{$$=GetYSN();$$->type=DUMMYPTR;}
		| type_spec sc_spec	{ $$=$1; }
		| sc_spec type_spec	{ $$=$2; }
		;

sc_spec		: AUTO 
		| STATIC
		| REGISTER
		| CONST
		;

mod_declaration	: decl_specs init_decl_list semicolon
			  { SetIdList($1->type,$2->listptr,VARNAME,0);
			    dontfree($2);dontfree($1);
			  }
		| TypeDef init_decl_list semicolon
			  { SetIdList(DUMMYPTR,$2->listptr,TYPENAME,0);
			    dontfree($2);
			  }
		| TypeDef type_spec init_decl_list semicolon
			  { SetIdList($2->type,$3->listptr,TYPENAME,0);
			    dontfree($2);dontfree($3);
			  }
		| EXTERN type_spec mopt ext_list semicolon
		| EXTERN ext_list semicolon
		;

declaration	: mod_declaration
		| privpub 	{ StoredEPFlag=PRIVATEFLAG;
				  PRIVATEFLAG=FALSE; 
			          INSIDE_PRIVFUNC_PROTO=1;}
		  function_decl semicolon
			  { $3->table->idtype=$1->idtype; dontfree($3); 
			    PopStack(FREE);PopStack(FREE);dontfree($1);
			    PRIVATEFLAG=StoredEPFlag;
			    INSIDE_PRIVFUNC_PROTO=0;
			  }
		;

system_types	: CHAR		{ $$=$1; $$->type=CHARPTR; }
		| UNSIGNED CHAR { $$=$1; $$->type=CHARPTR;
                                  CreateUnsignedTerm(&($$->string),$2->string);
                                }
		| SIGNED CHAR { $$=$1; $$->type=CHARPTR;
                                  CreateUnsignedTerm(&($$->string),$2->string);
                                }
		| SHORT		{ $$=$1; $$->type=SHORTPTR; }
		| UNSIGNED SHORT{ $$=$1; $$->type=SHORTPTR; 
				  CreateUnsignedTerm(&($$->string), $2->string);				}
		| SIGNED SHORT{ $$=$1; $$->type=SHORTPTR; 
				  CreateUnsignedTerm(&($$->string), $2->string);				}
		| INT		{ $$=$1; $$->type=INTPTR; }
		| LONG		{ $$=$1; $$->type=LONGPTR; }
		| UNSIGNED LONG { $$=$1; $$->type=LONGPTR; 
				  CreateUnsignedTerm(&($$->string), $2->string);				}
		| SIGNED LONG { $$=$1; $$->type=LONGPTR; 
				  CreateUnsignedTerm(&($$->string), $2->string);				}
		| SIGNED INT { $$=$1; $$->type=INTPTR; 
				  CreateUnsignedTerm(&($$->string), $2->string);				}
		| DOUBLE	{ $$=$1; $$->type=DOUBLEPTR; }
		| FLOAT		{ $$=$1; $$->type=FLOATPTR; }
		| UNSIGNED	{ $$=$1; $$->type=UNSIGNEDPTR; }
		| SHORT	INT	{ $$=$1; $$->type=SHORTPTR; 
				  CreateUnsignedTerm(&($$->string), $2->string);				}
		| LONG INT	{ $$=$1; $$->type=LONGPTR; 
				  CreateUnsignedTerm(&($$->string), $2->string);				}
		| LONG LONG	{ $$=$1; $$->type=LONGPTR; 
				  CreateUnsignedTerm(&($$->string), $2->string);				}
		| LONG FLOAT	{ $$=$1; $$->type=DOUBLEPTR; 
				  CreateUnsignedTerm(&($$->string), $2->string);				}
		| LONG DOUBLE	{ $$=$1; $$->type=DOUBLEPTR; 
				  CreateUnsignedTerm(&($$->string), $2->string);				}
		| UNSIGNED INT	{ $$=$1; $$->type=UNSIGNEDPTR; 
				  CreateUnsignedTerm(&($$->string), $2->string);				}
		| UNSIGNED INT INT { $$=$1;  $$->type=UNSIGNEDPTR;
				  CreateUnsignedTerm2(&($$->string), $2->string,$3->string);
				}
		| UNSIGNED LONG LONG { $$=$1;  $$->type=UNSIGNEDPTR;
				  CreateUnsignedTerm2(&($$->string), $2->string,$3->string);
				}
		| UNSIGNED LONG INT { $$=$1;  $$->type=UNSIGNEDPTR;
				  CreateUnsignedTerm2(&($$->string), $2->string,$3->string);
				}
		| VOID		{ $$=$1; $$->type=VOIDPTR; }
		| ChareIDType	{ $$=$1; $$->type=CHAREIDPTR; }
		| EntryPointType { $$=$1; $$->type=ENTRYPOINTPTR; }
		| PeNumType	{ $$=$1; $$->type=PENUMPTR; }
		| PackIDType 	{ $$=$1; $$->type=PACKIDPTR; }
		| WriteOnceID	{ $$=$1; $$->type=WRITEONCEIDPTR; }
		| PVECTOR	{ $$=$1; $$->type=PVECTORPTR; }
		| ChareNameType	{ $$=$1; $$->type=CHARENAMEPTR; }
		| ChareNumType	{ $$=$1; $$->type=CHARENUMPTR; }
		| EntryNumType	{ $$=$1; $$->type=ENTRYNUMPTR; }
		| BOOLEAN 	{ $$=$1; $$->type=BOOLEANPTR; }
		| ACCIDTYPE 	{ $$=$1; $$->type=INTPTR; /* TO SIMPLIFY */}
		| MONOIDTYPE	{ $$=$1; $$->type=INTPTR; }
		| DUMMYMSG	{ $$=$1; $$->type=INTPTR; }
                | FunctionRefType { $$=$1; $$->type=FUNCTIONREFPTR; }
                | FUNCTION_PTR  { $$=$1; $$->type=FUNCTIONPTR; } 
		;

type_spec	: su_spec	{ $$=$1; }
		| typedef_name	{ $$=$1; }
		;

init_decl_list	: init_decl	{ $$=GetYSN();$$->listptr=GetListNode($1); }
		| init_decl_list comma init_decl
			  { $$ = $1;
			    if ($$->listptr==NULL)
				error("Unexpected ','",EXIT);
			    InsertNode($$->listptr,$3); 
			  }
		|	{  $$=GetYSN(); }
		;

init_decl	: declarator initializer_opt	{ $$=$1; }
		;

declarator	: IDENTIFIER	{ /*writeoutput($1->string,NOFREE);*/
			/* When in import, modified if level=0 : see lex.c */
				/*  if ((IMPORTFLAG)&&(ImportLevel==0)) */
                                /*  Dec 30, 1991, modified by Attila */
				  if ((IMPORTFLAG)&&(ImportLevel==0)
                                      && (!IMPORTMSGFLAG) )
				  writeoutput(Map(CurrentModule->name,"0",
							$1->string),NOFREE);
				  else writeoutput($1->string,NOFREE);
				  $$=$1;
				}
		| lparen declarator rparen	%prec L_PAREN
			{ $$=$2; }
		| MULT {dump("*");} declarator	
			{ YSNPTR dummy;
			  
			  $$=$3;dummy=$$;
			  while (dummy->ysn!=NULL) dummy=dummy->ysn;
			  dummy->ysn=GetYSN();dummy->ysn->idtype=POINTERTYPE;
			}
		| declarator lparen rparen	%prec L_PAREN
			{ YSNPTR dummy;
			  
			  $$=$1;dummy=$$;
			  while (dummy->ysn!=NULL) dummy=dummy->ysn;
			  dummy->ysn=GetYSN();dummy->ysn->idtype=FUNCTIONTYPE;
			}  
		| declarator lsquare rsquare	%prec L_SQUARE
			{ YSNPTR dummy;
			  
			  $$=$1;dummy=$$;
			  while (dummy->ysn!=NULL) dummy=dummy->ysn;
			  dummy->ysn=GetYSN();dummy->ysn->idtype=ARRAYTYPE;
			}
		| declarator lsquare expression rsquare %prec L_SQUARE
			{ YSNPTR dummy;
			  
			  $$=$1;dummy=$$;
			  while (dummy->ysn!=NULL) dummy=dummy->ysn;
			  dummy->ysn=GetYSN();dummy->ysn->idtype=ARRAYTYPE;
			}
		;

su_spec		: Struct su_rest
			{ $$=GetYSN();
			  $$->type=GetTypeNode(0,0);
			  $$->type->basictype=STRUCTTYPE;
			  $$->type->table=$2->table;
			  dontfree($2);
			} 	
		| Struct TYPE_OR_ID 
			{ SYMTABPTR dummy; int i;

			  if (IMPORTFLAG) ImportStruct=TRUE;
			  if (!ImportStruct)
				writeoutput($2->string,NOFREE);  
			  else 	writeoutput(Map(CurrentModule->name,"_CKTYPE",$2->string),NOFREE); 
                          writeoutput(" ",NOFREE);/*Feb18,1992 added by Attila*/
			  ImportStruct=FALSE;
			  $$=GetYSN();
			  dummy=FindInTable(CurrentTable,$2->string,&i);
			  if (i==0) 
				worksymtab=dummy;
			  else 	{ worksymtab=LocalFind($2->string);
			  	  if ((worksymtab==NULL)||(worksymtab->idtype!=STRUCTNAME))
				  	{ worksymtab =
					  	Insert($2->string,CurrentTable);
					  worksymtab->type=GetTypeNode(0,1);
					  worksymtab->idtype=STRUCTNAME;
					  worksymtab->declflag=NOTDECLARED;
					  worksymtab->type->declflag=NOTDECLARED;
					  worksymtab->type->type=NULL;
					}
				}
				  
			  if (worksymtab->idtype!=STRUCTNAME)
				error("Bad Struct Name",EXIT);
			  $$->type=worksymtab->type; 
			  DestroyString($2->string);dontfree($2);
			}
		| Struct IDENTIFIER { if (IMPORTFLAG) ImportStruct=TRUE;
			  	      if (ImportStruct)
						writeoutput(Map(CurrentModule->name,"_CKTYPE",$2->string),NOFREE); 
				      else	writeoutput($2->string,NOFREE); 
                                      writeoutput(" ",NOFREE); /* Feb 18,1992*/
				      ImportStruct=FALSE;
				    }
	  	  su_rest
			{ SYMTABPTR dummy,temp;int i;TYPEPTR type;

			  dummy=FindInTable(CurrentTable,$2->string,&i);
			  if (i!=0) 
				{ worksymtab=Insert($2->string,CurrentTable);
				  worksymtab->type=GetTypeNode(1,0);
				}
			  else 	{ if ((dummy->declflag!=NOTDECLARED) ||
					(dummy->idtype!=STRUCTNAME))
					error("Bad Struct Name",EXIT);
				  worksymtab=dummy;
				  type=GetTypeNode(1+worksymtab->type->count,0);
				  dummy=(SYMTABPTR)worksymtab->type->type;
				  while (dummy!=NULL)
					{ temp=(SYMTABPTR)dummy;
					  dummy->type=type;dummy=temp;
                                          /* this is an infinite loop */
                                          /* set dummy to NULL */
                                          dummy = NULL;
					}
				  dontfree(worksymtab->type);
				  worksymtab->type=type;
				}
	
			  worksymtab->type->basictype=STRUCTTYPE;
 			  worksymtab->idtype=STRUCTNAME;
			  worksymtab->declflag=DECLARED;
			  worksymtab->type->table=$4->table;
			  $$=GetYSN();$$->type=worksymtab->type;
		 	  DestroyString($2->string);dontfree($2);dontfree($4);
			}
		| Union su_rest
			{ 
			  { $$=GetYSN();
			  $$->type=GetTypeNode(0,0);
			  $$->type->basictype=UNIONTYPE;
			  $$->type->table=$2->table;
			  dontfree($2);
			  }
			} 	
		| Union IDENTIFIER
			{ SYMTABPTR dummy; int i;

			  if (IMPORTFLAG) ImportStruct=TRUE;
			  if (ImportStruct)
			  	writeoutput(Map(CurrentModule->name,"_CKTYPE",$2->string),NOFREE); 
			  else	writeoutput($2->string,NOFREE);  
                          writeoutput(" ",NOFREE); /* Feb 18 1992 */
			  ImportStruct=FALSE;
			  $$=GetYSN();
			  dummy=FindInTable(CurrentTable,$2->string,&i);
			  if (i==0) 
				worksymtab=dummy;
			  else 	{ worksymtab=LocalFind($2->string);
			  	  if ((worksymtab==NULL)||(worksymtab->idtype!=UNIONNAME))
				  	{ worksymtab =
					  	Insert($2->string,CurrentTable);
					  worksymtab->type=GetTypeNode(0,1);
					  worksymtab->idtype=UNIONNAME;
					  worksymtab->declflag=NOTDECLARED;
					  worksymtab->type->declflag=NOTDECLARED;
					  worksymtab->type->type=NULL;
					}
				}
				  
			  if (worksymtab->idtype!=UNIONNAME)
				error("Bad Struct Name",EXIT);
			  $$->type=worksymtab->type; 
			  DestroyString($2->string);dontfree($2);
			}
		| Union IDENTIFIER 	{ if (IMPORTFLAG) ImportStruct=TRUE;
					  if (ImportStruct)
			  			writeoutput(Map(CurrentModule->name,"_CKTYPE",$2->string),NOFREE); 
					  else 	writeoutput($2->string,NOFREE); 
                                          writeoutput(" ",NOFREE);/*Feb18 1992*/
					  ImportStruct=FALSE;
					}
		  su_rest
			{ SYMTABPTR dummy,temp;int i;TYPEPTR type;

			  
			  {
			  dummy=FindInTable(CurrentTable,$2->string,&i);
			  if (i!=0) 
				{ worksymtab=Insert($2->string,CurrentTable);
				  worksymtab->type=GetTypeNode(1,0);
				}
			  else 	{ if ((dummy->declflag!=NOTDECLARED) ||
					(dummy->idtype!=UNIONNAME))
					error("Bad Struct Name",EXIT);
				  worksymtab=dummy;
				  type=GetTypeNode(1+worksymtab->type->count,0);
				  dummy=(SYMTABPTR)worksymtab->type->type;
				  while (dummy!=NULL)
					{ temp=(SYMTABPTR)dummy;
					  dummy->type=type;dummy=temp;
                                          /* this is an infinite loop */
                                          /* set dummy to NULL */
                                          dummy = NULL;
					}
				  dontfree(worksymtab->type);
				  worksymtab->type=type;
				}
	
			  worksymtab->type->basictype=UNIONTYPE;
 			  worksymtab->idtype=UNIONNAME;
			  worksymtab->declflag=DECLARED;
			  worksymtab->type->table=$4->table;
			  $$=GetYSN();$$->type=worksymtab->type;
		 	  DestroyString($2->string);dontfree($2);dontfree($4); 
			  }
			}

		| Enum enum_rest
			{ $$=GetYSN();
			  $$->type=GetTypeNode(0,0);
			  $$->type->basictype=STRUCTTYPE;
			  $$->type->table=$2->table;
			  dontfree($2);
			} 	
		| Enum IDENTIFIER 
			{ SYMTABPTR dummy; int i;

			  if (IMPORTFLAG) ImportStruct=TRUE;
			  if (!ImportStruct)
				writeoutput($2->string,NOFREE);  
			  else 	writeoutput(Map(CurrentModule->name,"_CKTYPE",$2->string),NOFREE); 
                          writeoutput(" ",NOFREE);/*Feb18,1992 added by Attila*/
			  ImportStruct=FALSE;
			  $$=GetYSN();
			  dummy=FindInTable(CurrentTable,$2->string,&i);
			  if (i==0) 
				worksymtab=dummy;
			  else 	{ worksymtab=LocalFind($2->string);
			  	  if ((worksymtab==NULL)||
                                      (worksymtab->idtype!=ENUMNAME))
				  	{ worksymtab =
					  	Insert($2->string,CurrentTable);
					  worksymtab->type=GetTypeNode(0,1);
					  worksymtab->idtype=ENUMNAME;
					  worksymtab->declflag=NOTDECLARED;
					  worksymtab->type->declflag=NOTDECLARED;
					  worksymtab->type->type=NULL;
					}
				}
				  
			  if (worksymtab->idtype!=ENUMNAME)
				error("Bad Struct Name",EXIT);
			  $$->type=worksymtab->type; 
			  DestroyString($2->string);dontfree($2);
			}
		| Enum IDENTIFIER { if (IMPORTFLAG) ImportStruct=TRUE;
			  	      if (ImportStruct)
						writeoutput(Map(CurrentModule->name,"_CKTYPE",$2->string),NOFREE); 
				      else	writeoutput($2->string,NOFREE); 
                                      writeoutput(" ",NOFREE); /* Feb 18,1992*/
				      ImportStruct=FALSE;
				    }
	  	  enum_rest 
			{ SYMTABPTR dummy,temp;int i;TYPEPTR type;

			  dummy=FindInTable(CurrentTable,$2->string,&i);
			  if (i!=0) 
				{ worksymtab=Insert($2->string,CurrentTable);
				  worksymtab->type=GetTypeNode(1,0);
				}
			  else 	{ if ((dummy->declflag!=NOTDECLARED) ||
					(dummy->idtype!=ENUMNAME))
					error("Bad Struct Name",EXIT);
				  worksymtab=dummy;
				  type=GetTypeNode(1+worksymtab->type->count,0);
				  dummy=(SYMTABPTR)worksymtab->type->type;
				  while (dummy!=NULL)
					{ temp=(SYMTABPTR)dummy;
					  dummy->type=type;dummy=temp;
                                          /* this is an infinite loop */
                                          /* set dummy to NULL */
                                          dummy = NULL;
					}
				  dontfree(worksymtab->type);
				  worksymtab->type=type;
				}
	
			  worksymtab->type->basictype=ENUMTYPE;
 			  worksymtab->idtype=ENUMNAME;
			  worksymtab->declflag=DECLARED;
			  worksymtab->type->table=$4->table;
			  $$=GetYSN();$$->type=worksymtab->type;
		 	  DestroyString($2->string);dontfree($2);dontfree($4);
			}
		;


TYPE_OR_ID	: IDENTIFIER
		| TYPE_IDENTIFIER
		;

su_rest		: L_BRACE 	{ if (IMPORTFLAG) ImportLevel++; 
				  writeoutput("{",NOFREE); PushStack(); 
				}
		  s_decl_list R_BRACE 	
			{ writeoutput("}",NOFREE); 
			  $$=GetYSN(); $$->table=CurrentTable;
			  PopStack(NOFREE);RestoreCurrentTable();
			  if (IMPORTFLAG) ImportLevel--;
			}
		;

s_decl_list	: struct_decl
		| s_decl_list struct_decl
		;

struct_decl	: type_spec s_declarator_list semicolon
			{ SetIdList($1->type,$2->listptr,FIELDNAME,0);
			  dontfree($2);dontfree($1);
			}
		;

s_declarator_list	: s_declarator	{ $$=GetYSN();
					  $$->listptr=GetListNode($1);
					}
		| s_declarator_list comma s_declarator
			{ $$=$1;InsertNode($$->listptr,$3); }
		;

s_declarator	: declarator			{ $$=$1; }
		| declarator colon const_exp	{ $$=$1; }
		| colon const_exp		{  $$=GetYSN(); } 
		;




enum_rest	: L_BRACE 	{ if (IMPORTFLAG) ImportLevel++; 
				  writeoutput("{",NOFREE);
				}
		  enum_opt enum_list R_BRACE 	
			{ 
                          SetIdList($3->type,$4->listptr,VARNAME,0);
                          writeoutput("}",NOFREE); 
			  $$=GetYSN(); $$->table=CurrentTable;
			  if (IMPORTFLAG) ImportLevel--;
			}
		;

enum_opt	: {$$=GetYSN(); $$->type = INTPTR;}

enum_list	: enum_member { $$ = GetYSN(); $$->listptr=GetListNode($1);}
		| enum_list comma enum_member
                      { $$ = $1;
                        if ($$->listptr==NULL)
                           error("error in enum definition ",EXIT);
                        InsertNode($$->listptr,$3);
                      } 

enum_member	: IDENTIFIER
                    { writeoutput($1->string,NOFREE);
                      writeoutput(" ",NOFREE);
		      $$ = $1;
                    } 
		| IDENTIFIER 
                    {  writeoutput($1->string,NOFREE);
		       writeoutput(" ",NOFREE);	 
                       $$ = $1;
		    }
                  equal base_expression
                ;

initializer_opt	:
		| initializer
		;

initializer	: equal expression
		| equal L_BRACE 	{  
						writeoutput("{",NOFREE); }
		  init_rest
		;

init_rest	: initializer_list R_BRACE	{ 
						      writeoutput("}",NOFREE); }
		| initializer_list comma_R_BRACE
		;

initializer_list 	: expression
			| initializer_list comma_L_BRACE initializer_list 
			  R_BRACE		{ 
						     writeoutput("}",NOFREE); }
			| L_BRACE 		{ 
						     writeoutput("{",NOFREE); }
			  initializer_list R_BRACE
				{  writeoutput("}",NOFREE); }
			;


typename	: type_spec abstract_decl { 
					    { FreeType($1->type);dontfree($1); }
					  }
		;

abstract_decl	:
		| lparen abstract_decl rparen	%prec L_PAREN
		| MULT {dump("*");} abstract_decl
		| abstract_decl lparen rparen 	%prec L_PAREN
		| abstract_decl lsquare rsquare	%prec L_SQUARE
		| abstract_decl lsquare expression rsquare %prec L_SQUARE
		;

typedef_name	: TYPE_IDENTIFIER	
			{ TYPEPTR typeptr;

			  worksymtab=LocalFind($1->string);typeptr=NULL;
			  if (IMPORTFLAG)
			 /* it is a type id : MUST be at level 0 */
				{ writeoutput(Map(CurrentModule->name,"0",
					$1->string),NOFREE);
				  if ((worksymtab==NULL)||
		/*			(worksymtab->idtype!=TYPENAME)) */
                /*  Dec 31, Attila */
					((worksymtab->idtype!=TYPENAME) && 
					(worksymtab->idtype!=MESSAGENAME)))
					error("Bad Type Reference",EXIT);
				  else 	typeptr=worksymtab->type;
				}
			  else	/* writeoutput($1->string,NOFREE); */
				typeptr=ProcessTypeIdentifier($1);
			  $$=GetYSN();$$->type=typeptr;
			  $$->table=($1->modstring==NULL)?GlobalFind($1->string)
				:GlobalModuleSearch($1->string,$1->modstring);
			  DestroyString($1->modstring);
			  DestroyString($1->string);dontfree($1);
			}
		| system_types	{ $$=$1;writeoutput($1->string,NOFREE);
				  writeoutput(" ",NOFREE);
				  $$->table=$$->type->table; }
		;

base_expression	: primary
		| unary  base_expression %prec UNARY
		| SIZEOF lparen typedef_name rparen	%prec UNARY 
		| SIZEOF base_expression 	%prec UNARY
		| lparen typename rparen base_expression %prec UNARY
		| base_expression MULTDIV base_expression %prec MULT
		| base_expression ADDSUB base_expression %prec PLUS
		| base_expression shift base_expression	%prec SHIFT
		| base_expression compare base_expression	%prec COMPARE
		| base_expression equalequal base_expression	%prec EQUALEQUAL
		| base_expression notequal base_expression	%prec NOTEQUAL
		| base_expression AND {dump("&");} base_expression
		| base_expression HAT {dump("^");} base_expression
		| base_expression OR { dump("|"); } base_expression
		| base_expression ANDAND { dump("&&"); } base_expression
		| base_expression OROR {dump("||");} base_expression
		| base_expression QUESTION {dump("?");} base_expression colon base_expression
		| primary asgnop base_expression	%prec ASGNOP
		| primary equal base_expression	%prec EQUAL
		;
CKPRIO_L	: CKALLOCPRIOMSG L_PAREN
		;

CK_L		: CKALLOCMSG L_PAREN
		;

expression	: primary
		| unary  expression %prec UNARY
		| SIZEOF lparen typedef_name rparen	%prec UNARY 
		| SIZEOF expression 	%prec UNARY
		| lparen typename rparen expression %prec UNARY
		| expression MULTDIV expression %prec MULT
		| expression ADDSUB expression %prec PLUS
		| expression shift expression	%prec SHIFT
		| expression compare expression	%prec COMPARE
		| expression equalequal expression	%prec EQUALEQUAL
		| expression notequal expression	%prec NOTEQUAL
		| expression AND {dump("&");} expression
		| expression HAT {dump("^");} expression
		| expression OR { dump("|"); } expression
		| expression ANDAND { dump("&&"); } expression
		| expression OROR {dump("||");} expression
		| expression QUESTION {dump("?");} expression colon expression
		| primary asgnop expression	%prec ASGNOP
		| primary equal expression	%prec EQUAL
		| expression COMMA { dump(","); } expression
		| PRIVATECALL lparen IDENTIFIER 
			{ if (!InPass1)
			  { worksymtab=GlobalFind($3->string);
			  DestroyString($3->string);dontfree($3);
			  if ((worksymtab==NULL)||(worksymtab->idtype!=
					PRIVATEFNNAME))
				if (!InPass1)
					error("Bad Private Call",EXIT);
			  if (!InPass1)
			  	writeoutput(Map(CurrentModule->name,
				CurrentChare->name,worksymtab->name),NOFREE);
			  }
			} 
		  lparen
			{  
				writeoutput(CkLocalPtr,NOFREE);
			}
		  expression_opt rparen rparen
		| BRANCHCALL lparen { BUFFEROUTPUT=TRUE; }
		  base_expression   { BUFFEROUTPUT=FALSE; }
		  COMMA ck_primary lparen
			{ writeoutput("GetBocDataPtr(",NOFREE);
			  writeoutput(buffer.a,NOFREE);buffer.count=0;
			  writeoutput(")",NOFREE);
			}
		  expression_opt rparen rparen
		| CHARECALL lparen {  BUFFEROUTPUT=TRUE; }
		  base_expression {  BUFFEROUTPUT=FALSE; }
		  COMMA ck_primary lparen
			{ writeoutput("GetChareDataPtr(",NOFREE);
			  writeoutput(buffer.a,NOFREE);buffer.count=0;
			  writeoutput(")",NOFREE); 
			}
		  expression_opt rparen rparen
		| FNNAMETOREF L_PAREN 	{ FNNAMETOREFFLAG=TRUE; }
		  ck_primary R_PAREN    { FNNAMETOREFFLAG=FALSE; } 
		  /*semicolon */ /* Jan 16, 1992 Attila */
                /* following rule added, Jan 16 1992, by Attila */
                | FNREFTONAME L_PAREN   {
                    FNREFTONAMEFLAG = TRUE;
                    writeoutput("CsvAccess(_CK_9_GlobalFunctionTable)[",NOFREE);
                    }
                  expression R_PAREN    { 
                    writeoutput("]",NOFREE);
                    FNREFTONAMEFLAG=FALSE; 
                    } 
		| CK_L IDENTIFIER ckallocrest
			{ if (!InPass1)
				{ error("Bad Message Type",EXIT); }
			}
		| CK_L TYPE_IDENTIFIER 
			{ if (!InPass1)

/** I changed LocalFind to GlobalFind on Oct. 27, 1991. The same change
    has been made in Prio.. I hope it works!
**/

			  {worksymtab=($2->modstring==NULL)?GlobalFind($2->string):GlobalModuleSearch($2->string,$2->modstring);
			  if ((worksymtab==NULL)||(worksymtab->idtype!=MESSAGENAME))
				error("Bad Message Type",NOEXIT);
			  else	{

				  if (strcmp(CurrentModule->name,worksymtab->modname->name))
				  	SavedName=ModulePrefix(worksymtab->modname->name,worksymtab->name);
				  else	SavedName=MyModulePrefix(worksymtab->modname->name,worksymtab->name);
				  if (worksymtab->localid<=0) {
					writeoutput(CkGenericAlloc,NOFREE); 
				  }
				  else	{ 
					writeoutput("(",NOFREE);
					writeoutput(CkVarSizeAlloc,NOFREE); 
					writeoutput("[",NOFREE);
					writeoutput(SavedName,NOFREE);
					writeoutput("].alloc)",NOFREE);
				  }
				  writeoutput("(",NOFREE);
				  writeoutput(SavedName,FREE);
			 	  writeoutput(",",NOFREE);
				  writeoutput("sizeof(",NOFREE);
				  if (strcmp(CurrentModule->name,worksymtab->modname->name))
					SavedName=Map(worksymtab->modname->name,"0",worksymtab->name);
				  else 	SavedName=worksymtab->name;
				  writeoutput(SavedName,NOFREE);
				  writeoutput("),0",NOFREE);
				}
			 }
			}
		  ckallocrest
		| CKPRIO_L TYPE_IDENTIFIER COMMA
			{ if (!InPass1)
			  {worksymtab=($2->modstring==NULL)?GlobalFind($2->string):GlobalModuleSearch($2->string,$2->modstring);
			  if ((worksymtab==NULL)||(worksymtab->idtype!=MESSAGENAME))
				error("Bad Message Type",NOEXIT);
			  else	{
				  if (strcmp(CurrentModule->name,worksymtab->modname->name))
				  	SavedName=ModulePrefix(worksymtab->modname->name,worksymtab->name);
				  else	SavedName=MyModulePrefix(worksymtab->modname->name,worksymtab->name);
				   if (worksymtab->localid<=0) {
						writeoutput(CkGenericAlloc,NOFREE); 
				  }
				  else	{
					writeoutput("(",NOFREE);
					writeoutput(CkVarSizeAlloc,NOFREE); 
					writeoutput("[",NOFREE);
					writeoutput(SavedName,NOFREE);
					writeoutput("].alloc)",NOFREE);
				  }
				  writeoutput("(",NOFREE);
				  writeoutput(SavedName,FREE);
			 	  writeoutput(",",NOFREE);
				  writeoutput("sizeof(",NOFREE);
				  if (strcmp(CurrentModule->name,worksymtab->modname->name))
					SavedName=Map(worksymtab->modname->name,"0",worksymtab->name);
				  else 	SavedName=worksymtab->name;
				  writeoutput(SavedName,NOFREE);
				  writeoutput("),",NOFREE);
				}
			  }
			}
		  base_expression ckallocrest
		| CKPRIO_L IDENTIFIER COMMA
		  base_expression ckallocrest
			{ if (!InPass1)
				{ error("Bad Message Type",EXIT); }
			}
		| READMSGINIT { READMSGINITFLAG=TRUE; } 
		  lparen ck_primary rparen 
			{ READMSGINITFLAG=FALSE; }
		| ACCUMULATE L_PAREN	{ BUFFEROUTPUT = TRUE; }
		  base_expression COMMA
			{ BUFFEROUTPUT = FALSE;
			  writeoutput("\n{ _CK_4AccDataAreaType *_CK_4dptr;\n",NOFREE);
			  writeoutput("  _CK_4dptr=_CK_9GetAccDataArea(",NOFREE);
			  writeoutput(buffer.a,NOFREE);buffer.count=0;
			  writeoutput(");\n");
			  writeoutput("  _CK_9LockAccDataArea(_CK_4dptr);\n",NOFREE);
			  writeoutput("  _CK_9GetAccumulateFn(_CK_4dptr)(_CK_9GetAccDataPtr(_CK_4dptr)",NOFREE);
			}
		  IDENTIFIER L_PAREN expression_R_PAREN rparen
			{ if (strcmp($7->string,"addfn"))
				error("addfn expected",EXIT);
			  writeoutput(";\n  _CK_9UnlockAccDataArea(_CK_4dptr);\n}\n",NOFREE);
                          MULTI_STMT_TR = 1;
			}
		| NEWVALUE L_PAREN	{ BUFFEROUTPUT = TRUE; }
		  base_expression COMMA
			{ BUFFEROUTPUT = FALSE;
			  writeoutput("\n{ _CK_4MonoDataAreaType *_CK_4dptr;\n",NOFREE);
			  writeoutput("  _CK_4dptr=_CK_9GetMonoDataArea(",NOFREE);
			  writeoutput(buffer.a,NOFREE);buffer.count=0;
			  writeoutput(");\n");
			  writeoutput("  _CK_9LockMonoDataArea(_CK_4dptr);\n",NOFREE);
			  /*writeoutput("  _CK_9Mono(_CK_4dptr)(_CK_9GetMonoDataPtr(_CK_4dptr)",NOFREE);*/
			  writeoutput("  _CK_9MONO_BranchNewValue(_CK_4dptr",NOFREE);
			}
		  IDENTIFIER L_PAREN expression_R_PAREN rparen
			{ if (strcmp($7->string,"updatefn"))
				error("updatefn expected",EXIT);
			  writeoutput(";\n  _CK_9UnlockMonoDataArea(_CK_4dptr);\n}\n",NOFREE);
                          MULTI_STMT_TR = 1;
			}
		| BLOCKEDRECV 
                     {writeoutput("CkBlockedRecv",NOFREE);} 
                     lparen
                     {
                          if ( CurrentChare == NULL )
                             error("CkBlockedRecv not allowed here",EXIT);
                          else if (CurrentChare->idtype == CHARENAME)
                                 writeoutput("NULL,",NOFREE);
                          else
                                 writeoutput("_CK_4mydata,",NOFREE);
                     }
                  ck_primary rparen
		;

expression_R_PAREN : R_PAREN 
		|	{ writeoutput(",",NOFREE); }
		  expression R_PAREN 
		;

ckallocrest	: R_PAREN 
			{ writeoutput(")",NOFREE); }
		| COMMA { writeoutput(",",NOFREE); }
		  base_expression R_PAREN
			{ writeoutput(")",NOFREE); }
		;

expression_opt	:		
		| { writeoutput(",",NOFREE); } expression   	
		;

unary		: MULT 		{dump("*");}	
		| AND 		{dump("&");}	
		| MINUS		{dump("-");}	
		| TILDE		{dump("~");}	
		| EXCLAIM	{dump("!");}	
		;
MULTDIV		: MULT 		{dump("*");}
		| DIV 		{dump("/");}
		| MOD		{dump("%");}
		;

ADDSUB 		: PLUS		{dump("+");}
		| MINUS		{dump("-");}
		;

primary		: constant 
		| string
		| lparen expression rparen
		| primary lparen rparen			%prec L_PAREN
		| primary lparen expression rparen	%prec L_PAREN
		| primary lsquare expression rsquare  %prec L_SQUARE
		| primary DOT {dump(".");} IDENTIFIER
			{ 
			  { writeoutput($4->string,NOFREE);
			    DestroyString($4->string);dontfree($4);
			  }
			}	
		| primary POINTERREF {dump("->");} IDENTIFIER
			{ 
			  { writeoutput($4->string,NOFREE);
			    DestroyString($4->string);dontfree($4);
			  }
			}	
		| incdec primary	%prec INCDEC
		| primary incdec	%prec INCDEC
		| ck_primary
		;

ck_primary	: IDENTIFIER	{ SYMTABPTR dummy;
				  int i;
				  
				  if ((!InPass1)&&(!SpecialVar($1)))
				  { 
				  worksymtab=GlobalFind($1->string);
				  if (worksymtab==NULL)
					{
                                        /*
					error("Undeclared Identifier: ",NOEXIT);
					PutOnScreen($1->string);
                                        PutOnScreen("\n");
                                        */
					}
				  if (IsFunction(worksymtab))
					writefunction(worksymtab);
				  else if (IsEntry(worksymtab))
						writeentry(worksymtab);
				       else if (IsPublic(worksymtab))
						writepublic(worksymtab);
				       else if (IsReadOnly(worksymtab))
						writereadonly(worksymtab);
				       else if (IsAccumulator(worksymtab))
						writeaccname(worksymtab);
				       else if (IsMonotonic(worksymtab))
						writemononame(worksymtab);
				       else if (IsTable(worksymtab))
						writetable(worksymtab);
				       else if (IsChare(worksymtab))
						writechare(worksymtab);
				       else { if ((worksymtab==NULL) ||
						  (worksymtab !=
						  LocalFind(worksymtab->name)))
					     {if (!InPass1)
						 /*  PutOnScreen($1->string); */
					      {  /*
                                               error("Bad Identifier: ",NOEXIT);
					       PutOnScreen($1->string);
                                               PutOnScreen("\n"); */  }
					     }
					      if (InChareEnv($1->string,worksymtab))
						writeoutput(CkMyData,NOFREE);
					      writeoutput($1->string,NOFREE); 
				       	    }
				  DestroyString($1->string);dontfree($1);
				  }
				}
		| ID_DCOLON_ID
			{ if (!InPass1)
			  { worksymtab=ProcessIdDcolonId($1);
			    if (IsReadOnly(worksymtab))
				writereadonly(worksymtab);
			    else if (IsChare(worksymtab))
				 	writechare(worksymtab);
				 else if (IsFunction(worksymtab))
				 	writefunction(worksymtab);
				      else if (IsAccumulator(worksymtab))
						writeaccname(worksymtab);
					   else if (IsMonotonic(worksymtab))
						    writemononame(worksymtab);
						else if (IsTable(worksymtab))
							writetable(worksymtab);
				      		else error("Bad Component Ref.",EXIT);
			    DestroyString($1->modstring);DestroyString($1->string);dontfree($1);
			  }
			}
		| ID_DCOLON_ID AT IDENTIFIER
			{ if (!InPass1)
			  { worksymtab=ProcessIdDcolonId($1);
			    if (!IsChare(worksymtab))
				error("Bad EP Reference",EXIT);
			  sym2=GlobalEntryFind($3->string,$1->string,$1->modstring);
			  if (IsPublic(sym2))
			       writepublic(sym2);
			  else if (IsEntry(sym2))
				    writeentry(sym2);
			       else if (IsPrivate(sym2))
					writeprivate(sym2);
				    else error("Bad Component Ref.",EXIT);
			  DestroyString($1->modstring);
			  DestroyString($1->string);dontfree($1);
			  DestroyString($3->string);dontfree($3);
			  }
			}
		| IDENTIFIER AT IDENTIFIER
			{ sym1=GlobalFind($1->string);
			  sym2=GlobalEntryFind($3->string,$1->string,CurrentModule->name);
 			  if (!(IsChare(sym1)&&(IsPublic(sym2)||IsEntry(sym2))||IsPrivate(sym2)))
				error("Bad Reference",NOEXIT);
			  else 	if (IsPublic(sym2))
					writepublic(sym2);
				else if (IsEntry(sym2))
					writeentry(sym2);
				     else writeprivate(sym2);
			  DestroyString($1->string);dontfree($1);
			  DestroyString($3->string);dontfree($3);
			}
		;

compound_statement	: L_BRACE	{ 
				 	  { writeoutput("{",NOFREE);
					    PushStack();
					  }
					}
			  decl_list_opt stmt_list_opt R_BRACE
					{ 
					  { writeoutput("}",NOFREE);
					    PopStack(FREE);
					    RestoreCurrentTable();
					  }
					}
			;

decl_list_opt	: 
		| decl_list
		;

decl_list	: declaration
		| decl_list declaration
		;

stmt_list_opt	:
		| stmt_list
		;

stmt_list	: statement
		| stmt_list statement
		;

statement	: compound_statement
                | expression SEMICOLON 
                    {                      /* Feb 18,1991 added by Attila */
                      if (MULTI_STMT_TR)   /* Accumulate, Newvalue statements*/ 
                         MULTI_STMT_TR = 0;/* are translated into a block.   */
                      else                 /* If expression is one of them   */
                         writeoutput(";",NOFREE); /* don't print semicolon   */
                    }
		| IF lparen expression rparen statement	
		| IF lparen expression rparen statement ELSE statement
		| WHILE lparen expression rparen statement
		| DO statement 
		  WHILE lparen expression rparen 		
		| FOR lparen exp_opt semicolon exp_opt semicolon
			exp_opt rparen statement
		| SWITCH lparen expression rparen statement
		| CASE expression colon statement
		| DEFAULT colon statement
		| BREAK semicolon
		| CONTINUE semicolon
		| RETURN semicolon
		| RETURN expression semicolon
		| GOTO IDENTIFIER { 
				    { writeoutput($2->string,NOFREE);
				      DestroyString($2->string);dontfree($2);
				    }
				  }
		  semicolon 
		| IDENTIFIER      { 
                                    writeoutput($1->string,NOFREE);
				    { if (GlobalFind($1->string)!=NULL)
					error("Bad Label",EXIT);
				      DestroyString($1->string);dontfree($1);
				    }
				  }
		  colon statement
		| semicolon
		;


function_decl	: IDENTIFIER 	
			{ int i;
			  if (export_to_c_flag == 0) {
                             if (INSIDE_PRIVFUNC_PROTO ==1)
			        writeoutput(" PROTO_PUB_PRIV ",NOFREE);
                             else
			        writeoutput(" static ",NOFREE);
                          }
		 	  if (CurrentChare!=NULL)
				writeoutput(Map(CurrentModule->name,CurrentChare
						->name,$1->string),NOFREE);
			  else writeoutput($1->string,NOFREE);
			  worksymtab=FindInTable(CurrentTable,$1->string,&i);
			  if (i!=0)
			  	worksymtab=Insert($1->string,CurrentTable);
			  DestroyString($1->string);dontfree($1);
			  worksymtab->type=SetType(INTPTR);
			  worksymtab->idtype=FNNAME;
			  worksymtab->declflag=DECLARED;
			  PushStack();
		 	  SavedFnNode=worksymtab;
			}
		  lparen 
			{ 
			     if (PRIVATEFLAG)
				writeoutput(CkLocalPtr,NOFREE);
			}
		  parameter_list_opt rparen
			{ 
			  { PushStack(); $$=GetYSN();$$->table=SavedFnNode; 
		 	  if (PRIVATEFLAG) {
			  WriteReturn();
			  writeoutput("void *",NOFREE);
			  writeoutput(CkLocalPtr,NOFREE);
			  writeoutput(";",NOFREE); }
			  }
			}
		| type_spec mopt IDENTIFIER 
			{ int i;
			  if (export_to_c_flag == 0) {
                             if (INSIDE_PRIVFUNC_PROTO==1)
			        writeoutput(" PROTO_PUB_PRIV ",NOFREE);
                             else
			        writeoutput(" static ",NOFREE);
                          }
			  if ($2!=NULL) { writeoutput("*",NOFREE);dontfree($2); }
		 	  if (CurrentChare!=NULL)
				writeoutput(Map(CurrentModule->name,CurrentChare
						->name,$3->string),NOFREE);
			  else writeoutput($3->string,NOFREE);
		/*Attila  worksymtab=FindInTable(CurrentTable,$1->string,&i);*/
			  worksymtab=FindInTable(CurrentTable,$3->string,&i);
			  if (i!=0)
        	/* Attila       worksymtab=Insert($1->string,CurrentTable); */
			  	worksymtab=Insert($3->string,CurrentTable);
			  DestroyString($3->string);dontfree($3);
			  worksymtab->type=SetType($1->type);dontfree($1);
			  worksymtab->idtype=FNNAME;
			  worksymtab->declflag=DECLARED;
			  PushStack();
		 	  SavedFnNode=worksymtab;
			}
		  lparen 
			{ 
			    if (PRIVATEFLAG)
				writeoutput(CkLocalPtr,NOFREE);
			}
		  parameter_list_opt rparen
			{ 
			  { PushStack(); $$=GetYSN();$$->table=SavedFnNode; 
			  if (PRIVATEFLAG) {
			  WriteReturn();
			  writeoutput("void *",NOFREE);
			  writeoutput(CkLocalPtr,NOFREE);
			  writeoutput(";",NOFREE); }
			  }
			}
		;

mopt		:  MULT { $$=GetYSN(); }
		| { $$=NULL; }
		;

parameter_list_opt	: 
			| parameter_list
			;

parameter_list	: IDENTIFIER	{ 
				  { if (PRIVATEFLAG)
					writeoutput(",",NOFREE);
				  writeoutput($1->string,NOFREE);
				  Insert($1->string,CurrentTable);
				  DestroyString($1->string);dontfree($1); 
				  }
				}
		| parameter_list comma IDENTIFIER
			{ 
			  { writeoutput($3->string,NOFREE);
			  Insert($3->string,CurrentTable);
			  DestroyString($3->string);dontfree($3);
			  }
			}
          /* added by Robert Allan Zeh, 9/19/1993 */
        | typedef_name IDENTIFIER {
            { if (PRIVATEFLAG)
                writeoutput(",", NOFREE);
              writeoutput($2->string, NOFREE);
              Insert($2->string, CurrentTable);
              DestroyString($2->string); dontfree($2); } }
		;

function_body	: decl_list_opt
	 	  { 
		    { worksymtab=CurrentTable;
		    PopStack(NOFREE);
			/* Removed by Robert Allan Zeh, 9/20/1993, so that
               ANSI C type declarations aren't flagged as errors. */
            /* if (CheckDeclaration(worksymtab,CurrentTable))
			error("Bad Parameter List",EXIT);
		    if (CheckDeclaration(CurrentTable,worksymtab))
			error("Bad Parameter List",EXIT); */
		   FreeTree(CurrentTable);
		   CurrentTable=StackTop->tableptr=worksymtab;
		   PushStack();
		  }
		 }
		  function_stmt
		  { 
		    {
		    PopStack(FREE);
		    PopStack(FREE);
		    RestoreCurrentTable();
		    }
		  }
		;

function_stmt	: L_BRACE 
			{ 
			  { writeoutput("{ ",NOFREE); 
			  if (PRIVATEFLAG)		  
				{ writeoutput(CurrentChare->name,NOFREE);
				  writeoutput(DataSuffix,NOFREE);
				  writeoutput(AssignMyDataPtr,NOFREE);
				  writeoutput(CurrentChare->name,NOFREE);
				  writeoutput(DataSuffix,NOFREE);
				  writeoutput(" *)",NOFREE);
				  writeoutput(CkLocalPtr,NOFREE);
				  writeoutput(";",NOFREE);
                                  writeoutput(AssignMyID,NOFREE);
                                  if (CurrentChare->idtype==BOCNAME)
                                    writeoutput(AssignMyBOC,NOFREE);
                                  WriteReturn();
				}
			  }
			}
		  decl_list_opt stmt_list_opt R_BRACE
			{  writeoutput("}",NOFREE); }
		;

exp_opt		:
		| expression
		;

constant	: number
		| char_const
		;

const_exp	: constant    /* const_exp:expression*/
		;

varsize_decl	: VARSIZE typedef_name IDENTIFIER L_SQUARE R_SQUARE SEMICOLON
			{ worksymtab=Insert($3->string,CurrentTable);
			  worksymtab->idtype=VARSIZENAME;
			  /*
			  worksymtab->type=ProcessTypeIdentifier($2);
			  worksymtab->type=(TYPEPTR)($2->modstring==NULL)?
				GlobalFind($2->string):
				GlobalModuleSearch($2->string,$2->modstring);
			  */
			  worksymtab->type = (TYPEPTR) $2->table;
			  writeoutput(" *",NOFREE);
			  writeoutput($3->string,NOFREE);
			  writeoutput(";",NOFREE);WriteReturn();
			  DestroyString($2->string);DestroyString($2->modstring);
			  DestroyString($3->string);dontfree($2);dontfree($3);
			}
		;

msg_opt_su_rest	: {$$=GetYSN(); writeoutput("int _CK_dummy_var;",NOFREE);}
		| msg_su_rest {$$ = $1;}
		;

msg_su_rest	: struct_decl { $$=GetYSN(); }
		| varsize_decl { $$=GetYSN();$$->count++; }
		| msg_su_rest struct_decl { $$=$1; }
		| msg_su_rest varsize_decl { $$=$1;$$->count++;} 
		;

msg_opt_id	:
		| IDENTIFIER	{ writeoutput($1->string,NOFREE);
				  worksymtab=Insert($1->string,CurrentTable);
				  worksymtab->idtype=STRUCTNAME;
				  DestroyString($1->string);dontfree($1);
				}
		;
	
optname_insquare: L_SQUARE IDENTIFIER R_SQUARE
			{ ForwardedMsgName=MakeString($2->string);
			  worksymtab=Insert($2->string,CurrentTable);
			  worksymtab->idtype=MESSAGENAME;
			  DestroyString($2->string);dontfree($2);
			  FORWARDEDMESSAGE=TRUE; 
			}
		|
		;	
message_defn	: MESSAGE 	{ writeoutput("typedef struct ",NOFREE); }
		  optname_insquare
		  msg_opt_id 
		  L_BRACE	{PushStack();writeoutput(" { ",NOFREE);}
		  msg_opt_su_rest
			{ SavedTypeNode=GetTypeNode(1,0);
			  SavedTypeNode->table=CurrentTable;
			  PopStack(NOFREE);RestoreCurrentTable();
			  writeoutput(" } ",NOFREE);
			  if (!InPass1)
			  	{ MsgCount++;
			  	  writeoutput(SavedName=Map(CurrentModule->name,"0",
						itoa(MsgCount)),NOFREE);
				}
			  writeoutput(";",NOFREE);WriteReturn();
			  if (FORWARDEDMESSAGE)
				{ int i;

				  writeoutput("typedef ",NOFREE);
				  writeoutput(SavedName,NOFREE);
				  writeoutput(" ",NOFREE);
				  writeoutput(ForwardedMsgName,NOFREE);
			  	  writeoutput(";",NOFREE);WriteReturn();
				  worksymtab=FindInTable(CurrentTable,ForwardedMsgName,&i);
			  	  worksymtab->type=SavedTypeNode;
				}
			  MsgToStructFlag=TRUE;
			}
		  msg_pu_rest TypeOrId SEMICOLON
			{ if (FORWARDEDMESSAGE)
				{ int i;

			 	  if (strcmp($10->string,ForwardedMsgName))
					error("Message Tag Doesn't Match Name",EXIT);
				  worksymtab=FindInTable(CurrentTable,$10->string,&i);
				}
			  else { writeoutput("typedef ",NOFREE);
			  	 writeoutput(SavedName,NOFREE);
			  	 writeoutput(" ",NOFREE);
			  	 writeoutput($10->string,NOFREE);
			  	 writeoutput(";",NOFREE);WriteReturn();
			  	 worksymtab=Insert($10->string,CurrentTable);
			       }
			  worksymtab->localid=$7->count;
			  /* Added Sept. 16, 1991 */
			  worksymtab->msgno=MsgCount;
			  /* Added Sept. 16, 1991 */
			  if (worksymtab->localid==0)
				worksymtab->localid=$9->count;
			  worksymtab->userpack=$9->count;
			  worksymtab->idtype=MESSAGENAME;
			  worksymtab->type=SavedTypeNode;
			  DestroyString($10->string);dontfree($10);
			  MsgToStructFlag=FALSE;
			  if (FORWARDEDMESSAGE)
				DestroyString(ForwardedMsgName);
			  FORWARDEDMESSAGE=FALSE;
			} 	
		;

TypeOrId	: IDENTIFIER {$$=$1;}
		| TYPE_IDENTIFIER { $$=$1; /* THIS IS CORRECT!! Nov. 12,'91 */}
		;

msg_pu_rest	: R_BRACE  	{ $$=GetYSN(); }
		| PACK pfn
			{ RestoreCurrentTable(); }
		  UNPACK ufn R_BRACE	
			{ $$=GetYSN(); $$->count = -1; RestoreCurrentTable(); }
		;

pfn		: IDENTIFIER 
			{ writeoutput(" static ",NOFREE);
			  writeoutput(Map(CurrentModule->name,itoa(MsgCount),
					 	"PACK"),NOFREE);
			  PushStack();
			}
		  lparen  parameter_list_opt rparen
			{ PushStack(); }
		  function_body
		;

ufn		: IDENTIFIER
			{ writeoutput(" static ",NOFREE);
			  writeoutput(Map(CurrentModule->name,itoa(MsgCount),
					 	"UNPACK"),NOFREE);
			  PushStack();
			}
		  lparen  parameter_list_opt rparen
			{ PushStack(); }
		  function_body
		;

table_defn 	: TABLE L_BRACE hashfn R_BRACE
				{
					if (IMPORTFLAG) writeoutput("extern ",NOFREE);
                  	writeoutput("int ",NOFREE);
				}
				tid_list semicolon
			| TABLE 
				{ if (IMPORTFLAG) writeoutput("extern ",NOFREE);
                  writeoutput("int ",NOFREE);
				  if (!InPass1) {
					SaveMsgCount = MsgCount; 
					MsgCount = 0;
				  }
                }
				tid_list semicolon { if (!InPass1) MsgCount = SaveMsgCount; }
			;

tid_list    : IDENTIFIER
            { worksymtab=Insert($1->string,CurrentTable);
              worksymtab->idtype=TABLENAME;
			  worksymtab->msgno=MsgCount;
              writeoutput(AppendMap(CurrentModule->name,$1->string),NOFREE);
             DestroyString($1->string);dontfree($1);
            }
        | tid_list comma IDENTIFIER
            { worksymtab=Insert($3->string,CurrentTable);
              worksymtab->idtype=TABLENAME;
			  worksymtab->msgno=MsgCount;
              writeoutput(AppendMap(CurrentModule->name,$3->string),NOFREE);
             DestroyString($3->string);dontfree($3);
            }
        ;



mono_defn	: MONOTONIC L_BRACE TYPE_IDENTIFIER MULT IDENTIFIER SEMICOLON
			{ if (!InPass1) MsgCount++; AccMonoName=$5->string; 
			  AccMonoTypeName=$3->string;
			}
		  initfn updatefn R_BRACE IDENTIFIER SEMICOLON
			{ worksymtab=Insert($11->string,CurrentTable);
			  worksymtab->idtype=MONONAME;
			  worksymtab->msgno=MsgCount;
			}
		;

acc_defn	: ACCUMULATOR L_BRACE TYPE_IDENTIFIER MULT IDENTIFIER SEMICOLON
			{ if (!InPass1) MsgCount++; AccMonoName=$5->string; 
			  AccMonoTypeName=$3->string;
			}
		  initfn addfn combinefn R_BRACE IDENTIFIER SEMICOLON
			{ worksymtab=Insert($12->string,CurrentTable);
			  worksymtab->idtype=ACCNAME;
			  worksymtab->msgno=MsgCount;
			}
		;

initfn		: type_spec mopt IDENTIFIER 
			{ if (strcmp($3->string,"initfn"))
				error("initfn expected",EXIT);
			  writeoutput(" static ",NOFREE);
			  if ($2!=NULL) { dontfree($2); writeoutput("*",NOFREE); }
			  writeoutput(Map(CurrentModule->name,itoa(MsgCount),
					"INIT"),NOFREE);
			  PushStack();
			  worksymtab=Insert(AccMonoName,CurrentTable);
			  worksymtab->idtype==VARNAME;
			  writeoutput("(",NOFREE);
			  writeoutput(AccMonoName,NOFREE);
			}
		  L_PAREN paralist_opt_R_PAREN
			{ PushStack();
			  worksymtab=Insert(AccMonoName,CurrentTable);
			  worksymtab->idtype==VARNAME;
			  writeoutput(AccMonoTypeName,NOFREE);
			  writeoutput(" *",NOFREE);
			  writeoutput(AccMonoName,NOFREE);
			  writeoutput(";",NOFREE);WriteReturn();
			}
		  function_body
			{ RestoreCurrentTable(); }
		;

hashfn		: IDENTIFIER 
			 {
			  if (!InPass1) MsgCount++; 
			  if (strcmp($1->string,"hashfn"))
				  error("hashfn expected",EXIT);
			  writeoutput(" static ",NOFREE);
			  writeoutput(Map(CurrentModule->name,itoa(MsgCount),
					"HASH"),NOFREE);
			  PushStack();
			}
			 L_PAREN IDENTIFIER
			 {
			  	writeoutput("( ",NOFREE);
                writeoutput($4->string,NOFREE);
                Insert($4->string,CurrentTable);
                DestroyString($4->string);dontfree($4);
			 }
			 rparen { PushStack(); }
			 function_body { RestoreCurrentTable(); }
			;

paralist_opt_R_PAREN : rparen
		| { writeoutput(",",NOFREE); } parameter_list rparen
		;

addfn		: IDENTIFIER
			{ if (strcmp($1->string,"addfn"))
				  error("addfn expected",EXIT);
			  writeoutput(" static ",NOFREE);
			  writeoutput(Map(CurrentModule->name,itoa(MsgCount),
					"INCREMENT"),NOFREE);
			  PushStack();
			  worksymtab=Insert(AccMonoName,CurrentTable);
			  worksymtab->idtype==VARNAME;
			  writeoutput("(",NOFREE);
			  writeoutput(AccMonoName,NOFREE);
			}
		  L_PAREN paralist_opt_R_PAREN
			{ PushStack();
			  worksymtab=Insert(AccMonoName,CurrentTable);
			  worksymtab->idtype==VARNAME;
			  writeoutput(AccMonoTypeName,NOFREE);
			  writeoutput(" *",NOFREE);
			  writeoutput(AccMonoName,NOFREE);
			  writeoutput(";",NOFREE);WriteReturn();
			}
		  function_body
			{ RestoreCurrentTable(); }
		;

combinefn	: IDENTIFIER
			{ if (strcmp($1->string,"combinefn"))
				error("combinefn expected",EXIT);
			  writeoutput(" static ",NOFREE);
			  writeoutput(Map(CurrentModule->name,itoa(MsgCount),
					"COMBINE"),NOFREE);
			  PushStack();
			  worksymtab=Insert(AccMonoName,CurrentTable);
			  worksymtab->idtype==VARNAME;
			  writeoutput("(",NOFREE);
			  writeoutput(AccMonoName,NOFREE);
			}
		  L_PAREN paralist_opt_R_PAREN
			{ PushStack();
			  worksymtab=Insert(AccMonoName,CurrentTable);
			  worksymtab->idtype==VARNAME;
			  writeoutput(AccMonoTypeName,NOFREE);
			  writeoutput(" *",NOFREE);
			  writeoutput(AccMonoName,NOFREE);
			  writeoutput(";",NOFREE);WriteReturn();
			}
		  function_body
			{ RestoreCurrentTable(); }
		;

updatefn	: IDENTIFIER
			{ if (strcmp($1->string,"updatefn"))
				error("updatefn expected",EXIT);
			  writeoutput(" static ",NOFREE);
			  writeoutput(Map(CurrentModule->name,itoa(MsgCount),
					"UPDATE"),NOFREE);
			  PushStack();
			  worksymtab=Insert(AccMonoName,CurrentTable);
			  worksymtab->idtype==VARNAME;
			  writeoutput("(",NOFREE);
			  writeoutput(AccMonoName,NOFREE);
			}
		  L_PAREN paralist_opt_R_PAREN
			{ PushStack();
			  worksymtab=Insert(AccMonoName,CurrentTable);
			  worksymtab->idtype==VARNAME;
			  writeoutput(AccMonoTypeName,NOFREE);
			  writeoutput(" *",NOFREE);
			  writeoutput(AccMonoName,NOFREE);
			  writeoutput(";",NOFREE);WriteReturn();
			}
		  function_body
			{ RestoreCurrentTable(); }
		;
	
chare_defn	: CHARE IDENTIFIER 
			{ 
			  { CurrentChare=Insert($2->string,CurrentTable);
			  FillSymTabNode(CurrentChare,CHARENAME,DECLARED,
				UNDEFINED,UNDEFINED,1,TRUE,NULL);
			  CurrentTable=CurrentChare->type->table;
			  DestroyString($2->string);dontfree($2);
			  writeoutput("typedef struct {",NOFREE);
			  WriteReturn();
                          }
			}
		  L_BRACE c_decl_list_opt entry_or_fns R_BRACE
			{ 
                          { CurrentChare=NULL;RestoreCurrentTable(); }
			}
		;

boc_defn	: BRANCHOFFICE IDENTIFIER 
			{ 
			  { CurrentChare=Insert($2->string,CurrentTable);
			  FillSymTabNode(CurrentChare,BOCNAME,DECLARED,
				UNDEFINED,UNDEFINED,1,TRUE,NULL);
			  CurrentTable=CurrentChare->type->table;
			  DestroyString($2->string);dontfree($2);
			  writeoutput("typedef struct {",NOFREE);
			  WriteReturn();
			  }
			}
		  L_BRACE c_decl_list_opt entry_or_fns R_BRACE
			{ 
				{ CurrentChare=NULL;RestoreCurrentTable(); }
			}
		;

entry_or_fns	: entry_or_fn
		| entry_or_fns entry_or_fn
		;

entry_head	: ENTRY IDENTIFIER 
			{ 
			  { writeoutput("void static ",NOFREE);
			  writeoutput(Map(CurrentModule->name,CurrentChare->name
						,$2->string),NOFREE);
			  worksymtab=Insert($2->string,CurrentTable);
			  worksymtab->idtype=ENTRYNAME;
			  worksymtab->declflag=DECLARED;
			  worksymtab->implicit_entry=ENTRYNAME;
			  SavedFnNode=worksymtab;
			  }
			}
		  COLON lparen {  OUTPUTOFF=TRUE; } 
/* 22 May 1992, Attila */
		| ENTRY IMPLICIT IDENTIFIER
                        {
                          { writeoutput("void static ",NOFREE);
                          writeoutput(Map(CurrentModule->name,CurrentChare->name
                                                ,$3->string),NOFREE);
                          worksymtab=Insert($3->string,CurrentTable);
                          worksymtab->idtype=ENTRYNAME;
                          worksymtab->declflag=DECLARED;
                          worksymtab->implicit_entry = ENTRYNAMEIMP;
                          SavedFnNode=worksymtab;
                          }
                        }
		COLON lparen {  OUTPUTOFF=TRUE; }
		;

msg_type	: TYPE_IDENTIFIER
			{ $$=GetYSN();
			  if (($1->modstring==NULL) || 
				(!strcmp($1->modstring,CurrentModule->name)))
			  	{ $$->table=LocalFind($1->string);
				  $$->string=$1->string;
				  dontfree($1);
				}
			  else	{ $$->table=GlobalModuleSearch($1->string,
							$1->modstring);
				  $$->string=MakeString(Map($1->modstring,"0",
							$1->string));
				  DestroyString($1->modstring);
				  DestroyString($1->string);dontfree($1);
				}
			}
		;

entry_or_fn	: entry_head
		  MESSAGE msg_type MULT IDENTIFIER 
			{ 
			  { OUTPUTOFF=FALSE; 
			  writeoutput($5->string,NOFREE); 
			  writeoutput(",",NOFREE);
			  writeoutput(CkLocalPtr,NOFREE);
			  sym1=SavedFnNode;
			  sym2=$3->table;
			  if ((sym1==NULL)||(sym2==NULL)||(sym2->idtype!=MESSAGENAME))
				{error("Bad Identifier Near: ",NOEXIT); 
				 PutOnScreen($5->string);
                                 PutOnScreen("\n");
				}
			  else { sym1->type = (TYPEPTR) sym2; }
			  }
			}
		  rparen 
			{ 
			  { WriteReturn();
			  writeoutput("void *",NOFREE);
			  writeoutput(CkLocalPtr,NOFREE);
			  writeoutput(";",NOFREE);
			  WriteReturn();
			  writeoutput($3->string,NOFREE);
			  writeoutput(" *",NOFREE);
			  writeoutput($5->string,NOFREE);
			  writeoutput(";",NOFREE);
			  PushStack();
			  worksymtab=Insert($5->string,CurrentTable);
			  worksymtab->idtype=VARNAME;
			  sym1=$3->table;
			  if (sym1==NULL) error("Bad Type",EXIT);
			  else worksymtab->type=sym1->type;
			  }
			}
		  entry_stmt
			{ 
				{ PopStack(FREE);RestoreCurrentTable(); }
			}
		| entry_head R_PAREN
			{ 
			  { writeoutput(CkDummyPtr,NOFREE);
			    writeoutput(",",NOFREE);
			    writeoutput(CkLocalPtr,NOFREE);
			    writeoutput(")",NOFREE);
			    WriteReturn();
			    writeoutput("void *",NOFREE);
			    writeoutput(CkLocalPtr,NOFREE);
			    writeoutput(",*",NOFREE);writeoutput(CkDummyPtr,NOFREE);
			    writeoutput(";\n{\n",NOFREE);
			    PushStack();
			  }
			}
		  entry_stmt
			{ 
				{ PopStack(FREE);RestoreCurrentTable();
				  writeoutput("CkFree(",NOFREE);
				  writeoutput(CkDummyPtr,NOFREE);
				  writeoutput(");\n}\n",NOFREE);
				}
			}
		| privpub {  
			    	{ PRIVATEFLAG = TRUE;
			    	 /* Now we allow public functions in Chares 
                                        if (($1->idtype==PUBLICFNNAME) && 
					(CurrentChare->idtype!=BOCNAME))
					error("Cannot Declare Publics",EXIT);
                                 */
				} 
			  }
		  function_decl	
			{ 
			  { $3->table->idtype=$1->idtype;
			    dontfree($1);
			  dontfree($3);
			  }
			}
		  function_body
			{  PRIVATEFLAG=FALSE; }
		; 

privpub		: PRIVATE	{ $$=GetYSN(); $$->idtype=PRIVATEFNNAME; }
		| PUBLIC	{ $$=GetYSN(); $$->idtype=PUBLICFNNAME; }
		;

entry_stmt	: L_BRACE 
			{ 
			  { writeoutput("{ ",NOFREE); 
			  writeoutput(CurrentChare->name,NOFREE);
			  writeoutput(DataSuffix,NOFREE);
			  writeoutput(AssignMyDataPtr,NOFREE);
			  writeoutput(CurrentChare->name,NOFREE);
			  writeoutput(DataSuffix,NOFREE);
			  writeoutput(" *)",NOFREE);
			  writeoutput(CkLocalPtr,NOFREE);
			  writeoutput(";",NOFREE);
                          writeoutput(AssignMyID,NOFREE); 
                          if (CurrentChare->idtype==BOCNAME)
                              writeoutput(AssignMyBOC,NOFREE);
                          WriteReturn();
			  }
			}
		  decl_list_opt stmt_list_opt R_BRACE
			{ 
			  { writeoutput("}",NOFREE); RestoreCurrentTable();}
			}
		;

main_chare_defn	: CHARE MAIN 
			{ 
			  { CurrentChare=Insert("main",CurrentTable);
			  FillSymTabNode(CurrentChare,CHARENAME,DECLARED,
				UNDEFINED,UNDEFINED,1,TRUE,NULL);
			  CurrentTable=CurrentChare->type->table;
			  writeoutput("typedef struct {",NOFREE);
			  WriteReturn();
			  }
			}
		  L_BRACE c_decl_list_opt m_entry_or_fns R_BRACE
			{ 
				{ CurrentChare=NULL;RestoreCurrentTable(); }
			}
		;

m_entry_or_fns	: m_entry_or_fn
		| m_entry_or_fns m_entry_or_fn
		;

m_entry_or_fn	: entry_or_fn
                /* 
		| ENTRY DATAINIT 
			{ HandleSpecialEntries(CKMAINDATAFUNCTION,"DataInit");}
		  COLON entry_stmt
			{ if (!InPass1) */
/* { PopStack(FREE);RestoreCurrentTable(); } Jan 24 1992, changed by Attila*/
/*				{ PopStack(FREE);PopStack(FREE);
                                  RestoreCurrentTable();
                                } 
			} */
		| ENTRY QUIESCENCE 
			{ HandleSpecialEntries(CKMAINQUIESCENCEFUNCTION,"QUIESCENCE"); }
		  COLON entry_stmt
			{ if (!InPass1)
				{ PopStack(FREE);RestoreCurrentTable(); }
			}
		| ENTRY CHAREINIT 
			{ HandleSpecialEntries(CKMAINCHAREFUNCTION,"CharmInit");}
		  COLON entry_stmt
			{ if (!InPass1)
    /* {PopStack(FREE);RestoreCurrentTable();} Jan 24 1992, changed by Attila */
				{ PopStack(FREE);PopStack(FREE);
                                  RestoreCurrentTable();
                                }
			}
		; 

c_decl_list_opt	:	{ writeoutput(DummyField,NOFREE);WriteReturn();
			  writeoutput("} ",NOFREE);
			  writeoutput(CurrentChare->name,NOFREE);
			  writeoutput(DataSuffix,NOFREE);
			  writeoutput(";",NOFREE);WriteReturn();
			}
		| c_decl_list	
			{ WriteReturn();
			  writeoutput("} ",NOFREE);
			  writeoutput(CurrentChare->name,NOFREE);
			  writeoutput(DataSuffix,NOFREE);
			  writeoutput(";",NOFREE);WriteReturn();
			}
		;

c_decl_list	: mod_declaration
		| c_decl_list mod_declaration
		;

lsquare		: L_SQUARE	{ writeoutput("[",NOFREE); }
		;
rsquare		: R_SQUARE	{ writeoutput("]",NOFREE); }
		;
lparen		: L_PAREN	{ writeoutput("(",NOFREE); }
		;
rparen		: R_PAREN	{ writeoutput(")",NOFREE); }
		;
incdec		: INCDEC	{ writeoutput($1->string,NOFREE);
				  DestroyString($1->string);dontfree($1);
				}
		;
asgnop		: ASGNOP	{ writeoutput($1->string,NOFREE);
				  DestroyString($1->string);dontfree($1);
				}
		;
equal		: EQUAL		{ writeoutput("=",NOFREE); }
		;
notequal	: NOTEQUAL	{ writeoutput("!=",NOFREE); }
		;
equalequal	: EQUALEQUAL	{ writeoutput("==",NOFREE); }
		;
shift		: SHIFT		{ writeoutput($1->string,NOFREE);
				  DestroyString($1->string);dontfree($1);
				}
		;
compare		: COMPARE	{ writeoutput($1->string,NOFREE);
				  DestroyString($1->string);dontfree($1);
				}
		;
number		: NUMBER	{ writeoutput($1->string,NOFREE);
				  DestroyString($1->string);dontfree($1);
				}
		;
char_const	: CHAR_CONST	{ writeoutput($1->string,NOFREE);
				  DestroyString($1->string);dontfree($1);
				}
		;
string		: STRING	{ writeoutput($1->string,NOFREE);
				  DestroyString($1->string);dontfree($1);
				}
		;
semicolon	: SEMICOLON	{ writeoutput(";",NOFREE); }
		;
colon		: COLON		{ writeoutput(":",NOFREE); }
		;
comma		: COMMA		{ writeoutput(",",NOFREE); }
		;
comma_L_BRACE	: COMMA_L_BRACE	{ writeoutput(",{",NOFREE); }
		;
comma_R_BRACE	: COMMA_R_BRACE { writeoutput(",}",NOFREE); }
		;

TypeDef		: TYPEDEF	{ writeoutput("typedef ",NOFREE); }
		;

Struct		: STRUCT	{ writeoutput("struct ",NOFREE); }
		;

Enum		: ENUM		{ writeoutput("enum ",NOFREE); }
		;

Union		: UNION		{ writeoutput("union ",NOFREE); }
		;

readonly	: READONLY	{ if (IMPORTFLAG)
					writeoutput("extern ",NOFREE);
				  writeoutput("SHARED_DECL ",NOFREE); 
				}
		  typedef_name	{ SavedFnNode=$3->table; } 
		  read_id_list semicolon
		;

read_id_list	: read_id
		| read_id_list comma read_id
		;

read_id		: IDENTIFIER
			{ worksymtab=Insert($1->string,CurrentTable);
			  worksymtab->idtype=READONLYVAR;
			  worksymtab->type=SavedFnNode->type;
			  if (SavedFnNode->idtype==MESSAGENAME)
				error("Bad Read Only",EXIT);
			  writeoutput(AppendMap(CurrentModule->name,$1->string),NOFREE);
			  DestroyString($1->string);dontfree($1);
			}
		| MULT IDENTIFIER
			{ worksymtab=Insert($2->string,CurrentTable);
			  worksymtab->idtype=READONLYMSG;
			  worksymtab->type=SavedFnNode->type;
			  if (SavedFnNode->idtype!=MESSAGENAME)
				error("Bad ReadOnly Msg",EXIT);
			  writeoutput(" *",NOFREE);
			  writeoutput(AppendMap(CurrentModule->name,$2->string),NOFREE);
			  DestroyString($2->string);dontfree($2);
			}
		| read_array
		;

read_array	: IDENTIFIER 
			{ worksymtab=Insert($1->string,CurrentTable);
			  worksymtab->idtype=READONLYARRAY;
			  worksymtab->type=SavedFnNode->type;
			  if (SavedFnNode->idtype==MESSAGENAME)
				error("Bad Array",EXIT);
			  writeoutput(AppendMap(CurrentModule->name,$1->string),NOFREE);
			  DestroyString($1->string);dontfree($1);
			}
		  lsquare expression rsquare
		| read_array lsquare expression rsquare
		;


/* dagger */

dag_boc_defn	: DAG BRANCHOFFICE IDENTIFIER
		   {
		     CurrentChare=Insert($3->string,CurrentTable);
		     FillSymTabNode(CurrentChare,BOCNAME,DECLARED,
		        UNDEFINED,UNDEFINED,1,TRUE,NULL);
		     CurrentTable=CurrentChare->type->table;
		     _dag_begindag(CurrentChare->name,0); /* 1: chare */
		     _dag2_begindag();
		     DestroyString($3->string); dontfree($3);
		     writeoutput("typedef struct {",NOFREE);
		     WriteReturn();
		   }
		  dag_rest
		;


dag_chare_defn	: DAG CHARE IDENTIFIER
		   {
		     CurrentChare=Insert($3->string,CurrentTable);
		     FillSymTabNode(CurrentChare,CHARENAME,DECLARED,
			UNDEFINED,UNDEFINED,1,TRUE,NULL);
		     CurrentTable=CurrentChare->type->table;
		     _dag_begindag(CurrentChare->name,1); /* 1: chare */
		     _dag2_begindag();
		     DestroyString($3->string); dontfree($3);
		     writeoutput("typedef struct {",NOFREE);
		     WriteReturn();
		   }
		  dag_rest
		;

dag_rest	: L_BRACE 
		   { _dag2_decl(); } 
		  c_decl_list_opt 
		   { char *sname;
                     /* declare a dummy entry */
		     worksymtab=Insert("_dag9",CurrentTable);
                     worksymtab->idtype = ENTRYNAME;
                     worksymtab->declflag = DECLARED;
		     _dag2_activator(CurrentModule->name,CurrentChare->name);
		     _dag2_send_act(0);
                     sname = MyModulePrefix(CurrentModule->name,"_dag3_MSG");
		     writeoutput(sname,FREE);
		     _dag2_send_act(1);
                     writeentry(worksymtab);
		     _dag2_send_act(2);
		     _dag2_conv0();
                   }
		  dag_entry_list
		   {
		     _dag2_conv1();
		     _dag2_entry_code(CurrentModule->name,CurrentChare->name);
		     _dag2_efunction_code();
		   }
		  dag_when_fn_list R_BRACE
		   {
		       _dag2_whenswitch();
		       _dag2_cond_code();	
                       _dag_enddag();
		       CurrentChare=NULL; RestoreCurrentTable(); 
		   }
		;


dag_entry_list	: dag_entry
		| dag_entry_list dag_entry 
		;

dag_entry	: ENTRY IDENTIFIER
			{
			  _dag_newentry($2->string);
			  worksymtab=Insert($2->string,CurrentTable);
			  worksymtab->idtype = ENTRYNAME;
			  worksymtab->declflag = DECLARED;
			  SavedFnNode=worksymtab;

                          _dag2_conv2();
			  writeentry(worksymtab);
			  _dag2_conv3(); 

			}
		  dag_multentry dag_matching dag_autofree COLON L_PAREN 
                        { OUTPUTOFF = TRUE;}
		  MESSAGE msg_type MULT IDENTIFIER 
			{
			  OUTPUTOFF = FALSE;
			  _dag_entrymsg($11->string,$13->string);
			  sym1=SavedFnNode;
			  sym2=$11->table;
			  if( (sym1==NULL)||(sym2==NULL)||(sym2->idtype!=MESSAGENAME))
				{error("Bad Identifier Near: ",NOEXIT);
				 PutOnScreen($13->string);
				 PutOnScreen("\n");
				}
			  else {sym1->type = (TYPEPTR) sym2;}
			}
		  R_PAREN SEMICOLON
		;
		

dag_matching	: {_dag_entry_match(0);}
		| MATCH {_dag_entry_match(1); }
		;

dag_autofree	: {_dag_entry_free(0);}
		| AUTOFREE {_dag_entry_free(1);}
		;

dag_multentry	: L_SQUARE IDENTIFIER R_SQUARE
		  { _dag_entrytype($2->string); }
		| { _dag_entrytype(NULL);}
		;

dag_when_fn_list	: dag_when_fn
			| dag_when_fn_list dag_when_fn
			;

dag_when_fn	: dag_when
		| dag_fn
		;

dag_when	: WHEN 
			{ _dag_newwhen(); _dag2_set_current_when();}	
		  dag_when_conds COLON 
                        { 
			  PushStack();
                          _dag2_when_header(CurrentTable);
			}
		  dag_when_stmt
			{
			  PopStack(FREE); RestoreCurrentTable();
			}
		;

dag_when_conds  : dag_when_cond
		| dag_when_conds COMMA dag_when_cond
		;

dag_when_cond	: IDENTIFIER dag_mult_cond 
			{ _dag_when_cond($1->string); }
		;

dag_mult_cond	: L_SQUARE IDENTIFIER R_SQUARE
			{ if (strcmp($2->string,"ANY") == 0)
				_dag_when_any(TRUE); 
                          else
                             error("ANY expected",NOEXIT);
			}
		| {_dag_when_any(FALSE); }
		;

dag_when_stmt	: entry_stmt
		;

dag_fn		: privpub {
				PRIVATEFLAG = TRUE;
				if( ($1->idtype==PUBLICFNNAME) &&
				    (CurrentChare->idtype!=BOCNAME))
					error("Cannot Declare Publics",EXIT);
			  }
		  function_decl
			  {
				$3->table->idtype=$1->idtype;
				dontfree($1);
				dontfree($3);
			  }
		  function_body
			  { PRIVATEFLAG = FALSE;}
		;	
/* dagger */

%%
yyerror(string)
char *string;
{   char temp[256];
    sprintf(temp,"%s: %s\n",string,token); 
    error(temp,EXIT);
}
/* {fprintf(stderr,"Token : %s - ",token); error(string,EXIT); } */

HandleSpecialEntries(fnname,string)
char *fnname,*string;
{ char *temp,*dummy;
  if (InPass1) return;
  writeoutput("void ",NOFREE);
  dummy=Map(CurrentModule->name,"main",string);
  temp=GetMem(strlen(fnname)+strlen(dummy)+1);
  strcpy(temp,dummy);strcat(temp,fnname);
  writeoutput(temp,FREE);
  WriteReturn();
  writeoutput("void *",NOFREE);
  writeoutput(CkLocalPtr,NOFREE);
  writeoutput(",*_CK_4NULL;",NOFREE);
  if (strcmp(string,"QUIESCENCE")) 
	 writeoutput("\nint argc;\nchar *argv[];",NOFREE); 
  WriteReturn();
  worksymtab=Insert(string,CurrentTable);
  worksymtab->idtype=ENTRYNAME;
  PushStack();
/* Jan 24 1992, added by Attila */
  if (strcmp(string,"QUIESCENCE")) {
     Insert("argc",CurrentTable);
     Insert("argv",CurrentTable);
     PushStack();
  }
}

void SwapFile(filestruct)
OUTPTR filestruct;
{ static OUTPTR  savedfile;

  savedfile=CurrentOut;
  CurrentOut=filestruct;
}

TYPEPTR ProcessTypeIdentifier(node)
YSNPTR node;
{ SYMTABPTR sym;

  if (node->modstring == NULL)
	writeoutput(node->string,NOFREE);
  else 	{ if ((IMPORTFLAG)||(strcmp(node->modstring,CurrentModule->name)))
		writeoutput(Map(node->modstring,"0",node->string),NOFREE);
	  else 	writeoutput(node->string,NOFREE);
	}
  if (node->modstring==NULL)
	sym=GlobalFind(node->string);
  else	sym=GlobalModuleSearch(node->string,node->modstring);
  if (sym==NULL) return(NULL); else return(sym->type);
}

char *itoa(v)
int v;
{ int i;

  if (dummy_ptr!=NULL) dontfree(dummy_ptr);
  dummy_ptr=GetMem(15);
  dummy_ptr[9]='\0';
  i=9;
  while (v>0)
  { dummy_ptr[--i]='0'+v%10;
    v=v/10;
  }
  i--;dummy_ptr[i]='_';
  return(dummy_ptr+i);
}
  
SYMTABPTR ProcessIdDcolonId(node)
YSNPTR node;
{ SYMTABPTR dummy;

  dummy=GlobalModuleSearch(node->string,node->modstring);
  if (dummy==NULL)
	error("Bad Module::Id Reference",EXIT);
  return(dummy);
/* Following code replaced by the above statement, Jan 6 1992, Attila */
/*  if (IsReadOnly(dummy)||IsFunction(dummy)||IsChare(dummy))
  	return(dummy);
  else 	error("Bad Module::Id Reference",EXIT); 
*/
}

SpecialVar(node)
YSNPTR node;
{ if (!strcmp(node->string,"NULL_PE")) return(writespecial(node));
  if (!strcmp(node->string,"NULL_VID")) return(writespecial(node));
  if (!strcmp(node->string,"NULL")) return(writespecial(node));
  if (!strcmp(node->string,"_CK_4mydata")) return(writespecial(node));
  return(0);
}

writespecial(node)
YSNPTR node;
{ writeoutput(node->string,NOFREE);DestroyString(node->string);
  dontfree(node); return(1);
}

CreateUnsignedTerm(s1,s2)
char **s1,*s2;
{
     char *buffer;

     buffer = (char *) malloc(strlen(*s1) + strlen(s2) + 2);
     if (buffer == NULL) printf("Out of memory\n"); 
     sprintf(buffer,"%s %s",*s1,s2);
     dontfree(*s1);
     *s1 = buffer;
}



CreateUnsignedTerm2(s1,s2,s3)
char **s1, *s2, *s3;
{
     char *buffer;

     buffer = (char *) malloc(strlen(*s1) + strlen(s2) + strlen(s3) + 3);
     if (buffer == NULL) printf("Out of memory\n");
     sprintf(buffer,"%s %s %s",*s1,s2,s3);
     dontfree(*s1);
     *s1 = buffer;
}
