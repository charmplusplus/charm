%{
#include <iostream.h>
#include "xi-symbol.h"

extern int yylex (void) ;
void yyerror(const char *);
extern unsigned int lineno;
ModuleList *modlist;

%}

%union {
  ModuleList *modlist;
  Module *module;
  ConstructList *conslist;
  Construct *construct;
  TParam *tparam;
  TParamList *tparlist;
  Type *type;
  EnType *rtype;
  PtrType *ptype;
  NamedType *ntype;
  FuncType *ftype;
  Readonly *readonly;
  Message *message;
  Chare *chare;
  Entry *entry;
  Template *templat;
  TypeList *typelist;
  MemberList *mbrlist;
  Member *member;
  TVar *tvar;
  TVarList *tvarlist;
  Value *val;
  ValueList *vallist;
  MsgVar *mv;
  MsgVarList *mvlist;
  char *strval;
  int intval;
}

%token MODULE
%token MAINMODULE
%token EXTERN
%token READONLY
%token <intval> CHARE GROUP NODEGROUP ARRAY
%token MESSAGE
%token CLASS
%token STACKSIZE
%token THREADED
%token MIGRATABLE
%token TEMPLATE
%token SYNC EXCLUSIVE VIRTUAL
%token VOID
%token PACKED
%token VARSIZE
%token ENTRY
%token <intval> MAINCHARE
%token <strval> IDENT NUMBER LITERAL
%token <intval> INT LONG SHORT CHAR FLOAT DOUBLE UNSIGNED

%type <modlist>		ModuleEList File
%type <module>		Module
%type <conslist>	ConstructEList ConstructList
%type <construct>	Construct
%type <strval>		Name QualName OptNameInit
%type <val>		OptStackSize
%type <intval>		OptExtern OptSemiColon MAttribs MAttribList MAttrib
%type <intval>		EAttribs EAttribList EAttrib OptPure
%type <intval>		CAttribs CAttribList CAttrib
%type <tparam>		TParam
%type <tparlist>	TParamList TParamEList OptTParams
%type <type>		Type SimpleType OptTypeInit 
%type <rtype>		OptType EParam
%type <type>		BuiltinType ArrayType
%type <ftype>		FuncType
%type <ntype>		NamedType ArrayIndexType
%type <ptype>		PtrType OnePtrType
%type <readonly>	Readonly ReadonlyMsg
%type <message>		Message TMessage
%type <chare>		Chare Group NodeGroup Array TChare TGroup TNodeGroup TArray
%type <entry>		Entry
%type <templat>		Template
%type <typelist>	BaseList OptBaseList TypeList
%type <mbrlist>		MemberEList MemberList
%type <member>		Member
%type <tvar>		TVar
%type <tvarlist>	TVarList TemplateSpec
%type <val>		ArrayDim Dim
%type <vallist>		DimList
%type <mv>		Var
%type <mvlist>		VarList

%%

File		: ModuleEList
		{ $$ = $1; modlist = $1; }
		;

ModuleEList	: /* Empty */
		{ $$ = 0; }
		| Module ModuleEList
		{ $$ = new ModuleList(lineno, $1, $2); }
		;

OptExtern	: /* Empty */
		{ $$ = 0; }
		| EXTERN
		{ $$ = 1; }
		;

OptSemiColon	: /* Empty */
		{ $$ = 0; }
		| ';'
		{ $$ = 1; }
		;

Name		: IDENT
		{ $$ = $1; }
		;

QualName	: IDENT
		{ $$ = $1; }
		| QualName ':'':' IDENT
		{
		  char *tmp = new char[strlen($1)+strlen($4)+3];
		  sprintf(tmp,"%s::%s", $1, $4);
		  $$ = tmp;
		}
		;

Module		: MODULE Name ConstructEList
		{ $$ = new Module(lineno, $2, $3); }
		| MAINMODULE Name ConstructEList
		{ $$ = new Module(lineno, $2, $3); $$->setMain(); }
		;

ConstructEList	: ';'
		{ $$ = 0; }
		| '{' ConstructList '}' OptSemiColon
		{ $$ = $2; }
		;

ConstructList	: /* Empty */
		{ $$ = 0; }
		| Construct ConstructList
		{ $$ = new ConstructList(lineno, $1, $2); }
		;

Construct	: OptExtern '{' ConstructList '}' OptSemiColon
		{ if($3) $3->setExtern($1); $$ = $3; }
		| OptExtern Module
		{ $2->setExtern($1); $$ = $2; }
		| OptExtern Readonly ';'
		{ $2->setExtern($1); $$ = $2; }
		| OptExtern ReadonlyMsg ';'
		{ $2->setExtern($1); $$ = $2; }
		| OptExtern Message ';'
		{ $2->setExtern($1); $$ = $2; }
		| OptExtern Chare
		{ $2->setExtern($1); $$ = $2; }
		| OptExtern Group
		{ $2->setExtern($1); $$ = $2; }
		| OptExtern NodeGroup
		{ $2->setExtern($1); $$ = $2; }
		| OptExtern Array
		{ $2->setExtern($1); $$ = $2; }
		| OptExtern Template
		{ $2->setExtern($1); $$ = $2; }
		;

TParam		: Type
		{ $$ = new TParamType($1); }
		| NUMBER
		{ $$ = new TParamVal($1); }
		| LITERAL
		{ $$ = new TParamVal($1); }
		;

TParamList	: TParam
		{ $$ = new TParamList($1); }
		| TParam ',' TParamList
		{ $$ = new TParamList($1, $3); }
		;

TParamEList	: /* Empty */
		{ $$ = 0; }
		| TParamList
		{ $$ = $1; }
		;

OptTParams	: /* Empty */
		{ $$ = 0; }
		| '<' TParamEList '>'
		{ $$ = $2; }
		;

BuiltinType	: INT
		{ $$ = new BuiltinType("int"); }
		| LONG
		{ $$ = new BuiltinType("long"); }
		| SHORT
		{ $$ = new BuiltinType("short"); }
		| CHAR
		{ $$ = new BuiltinType("char"); }
		| UNSIGNED INT
		{ $$ = new BuiltinType("unsigned int"); }
		| UNSIGNED LONG
		{ $$ = new BuiltinType("long"); }
		| UNSIGNED SHORT
		{ $$ = new BuiltinType("short"); }
		| UNSIGNED CHAR
		{ $$ = new BuiltinType("char"); }
		| LONG LONG
		{ $$ = new BuiltinType("long long"); }
		| FLOAT
		{ $$ = new BuiltinType("float"); }
		| DOUBLE
		{ $$ = new BuiltinType("double"); }
		| LONG DOUBLE
		{ $$ = new BuiltinType("long double"); }
		| VOID
		{ $$ = new BuiltinType("void"); }
		;

NamedType	: Name OptTParams
		{ $$ = new NamedType($1, $2); }
		;

SimpleType	: BuiltinType
		{ $$ = $1; }
		| NamedType
		{ $$ = $1; }
		;

OnePtrType	: SimpleType '*'
		{ $$ = new PtrType($1); }
		;

PtrType		: OnePtrType '*'
		{ $1->indirect(); $$ = $1; }
		| PtrType '*'
		{ $1->indirect(); $$ = $1; }
		;

ArrayDim	: NUMBER
		{ $$ = new Value($1); }
		| Name
		{ $$ = new Value($1); }
		;

ArrayType	: Type '[' ArrayDim ']'
		{ $$ = new ArrayType($1, $3); }
		;

FuncType	: Type '(' '*' Name ')' '(' TypeList ')'
		{ $$ = new FuncType($1, $4, $7); }
		;

Type		: SimpleType
		{ $$ = $1; }
		| OnePtrType
		{ $$ = (Type*) $1; }
		| PtrType
		{ $$ = (Type*) $1; }
		| ArrayType
		{ $$ = $1; }
		| FuncType
		{ $$ = $1; }
		;

TypeList	: /* Empty */
		{ $$ = 0; }
		| Type
		{ $$ = new TypeList($1); }
		| Type ',' TypeList
		{ $$ = new TypeList($1, $3); }
		;

Dim		: '[' ArrayDim ']'
		{ $$ = $2; }
		;

DimList		: /* Empty */
		{ $$ = 0; }
		| Dim DimList
		{ $$ = new ValueList($1, $2); }
		;

Readonly	: READONLY Type QualName DimList
		{ $$ = new Readonly(lineno, $2, $3, $4); }
		;

ReadonlyMsg	: READONLY MESSAGE SimpleType '*'  Name
		{ $$ = new Readonly(lineno, $3, $5, 0, 1); }
		;

MAttribs	: /* Empty */
		{ $$ = 0; }
		| '[' MAttribList ']'
		{ $$ = $2; }
		;

MAttribList	: MAttrib
		{ $$ = $1; }
		| MAttrib ',' MAttribList
		{ $$ = $1 | $3; }
		;

MAttrib		: PACKED
		{ $$ = SPACKED; }
		| VARSIZE
		{ $$ = SVARSIZE; }
		;

CAttribs	: /* Empty */
		{ $$ = 0; }
		| '[' CAttribList ']'
		{ $$ = $2; }
		;

CAttribList	: CAttrib
		{ $$ = $1; }
		| CAttrib ',' CAttribList
		{ $$ = $1 | $3; }
		;

CAttrib		: MIGRATABLE
		{ $$ = 0x01; }
		;

Var		: Type Name '[' ']' ';'
		{ $$ = new MsgVar($1, $2); }
		;

VarList		: Var
		{ $$ = new MsgVarList($1); }
		| Var VarList
		{ $$ = new MsgVarList($1, $2); }
		;

Message		: MESSAGE MAttribs NamedType
		{ $$ = new Message(lineno, $3, $2); }
		| MESSAGE MAttribs NamedType '{' TypeList '}'
		{ $$ = new Message(lineno, $3, $2, $5); }
		| MESSAGE MAttribs NamedType '{' VarList '}'
		{ $$ = new Message(lineno, $3, $2, 0, $5); }
		;

OptBaseList	: /* Empty */
		{ $$ = 0; }
		| ':' BaseList
		{ $$ = $2; }
		;

BaseList	: NamedType
		{ $$ = new TypeList($1); }
		| NamedType ',' BaseList
		{ $$ = new TypeList($1, $3); }
		;

Chare		: CHARE CAttribs NamedType OptBaseList MemberEList
		{ $$ = new Chare(lineno, $3, $4, $5, $2); }
		| MAINCHARE CAttribs NamedType OptBaseList MemberEList
		{ $$ = new MainChare(lineno, $3, $4, $5, $2); }
		;

Group		: GROUP CAttribs NamedType OptBaseList MemberEList
		{ $$ = new Group(lineno, $3, $4, $5, $2); }
		;

NodeGroup	: NODEGROUP CAttribs NamedType OptBaseList MemberEList
		{ $$ = new NodeGroup(lineno, $3, $4, $5, $2); }
		;

ArrayIndexType	: '[' NUMBER Name ']'
		{/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",$2);
			$$ = new NamedType(buf); 
		}
		| '[' Name ']'
		{ $$ = new NamedType($2); }
		;

Array		: ARRAY ArrayIndexType NamedType OptBaseList MemberEList
		{ $$ = new Array(lineno, $2, $3, $4, $5); }
		;

TChare		: CHARE CAttribs Name OptBaseList MemberEList
		{ $$ = new Chare(lineno, new NamedType($3), $4, $5, $2);}
		| MAINCHARE CAttribs Name OptBaseList MemberEList
		{ $$ = new MainChare(lineno, new NamedType($3), $4, $5, $2); }
		;

TGroup		: GROUP CAttribs Name OptBaseList MemberEList
		{ $$ = new Group(lineno, new NamedType($3), $4, $5, $2); }
		;

TNodeGroup	: NODEGROUP CAttribs Name OptBaseList MemberEList
		{ $$ = new NodeGroup( lineno, new NamedType($3), $4, $5, $2); }
		;

TArray		: ARRAY ArrayIndexType Name OptBaseList MemberEList
		{ $$ = new Array( lineno, $2, new NamedType($3), $4, $5); }
		;

TMessage	: MESSAGE MAttribs Name ';'
		{ $$ = new Message(lineno, new NamedType($3), $2); }
		;

OptTypeInit	: /* Empty */
		{ $$ = 0; }
		| '=' Type
		{ $$ = $2; }
		;

OptNameInit	: /* Empty */
		{ $$ = 0; }
		| '=' NUMBER
		{ $$ = $2; }
		| '=' LITERAL
		{ $$ = $2; }
		;

TVar		: CLASS Name OptTypeInit
		{ $$ = new TType(new NamedType($2), $3); }
		| FuncType OptNameInit
		{ $$ = new TFunc($1, $2); }
		| Type Name OptNameInit
		{ $$ = new TName($1, $2, $3); }
		;

TVarList	: TVar
		{ $$ = new TVarList($1); }
		| TVar ',' TVarList
		{ $$ = new TVarList($1, $3); }
		;

TemplateSpec	: TEMPLATE '<' TVarList '>'
		{ $$ = $3; }
		;

Template	: TemplateSpec TChare
		{ $$ = new Template($1, $2); $2->setTemplate($$); }
		| TemplateSpec TGroup
		{ $$ = new Template($1, $2); $2->setTemplate($$); }
		| TemplateSpec TNodeGroup
		{ $$ = new Template($1, $2); $2->setTemplate($$); }
		| TemplateSpec TArray
		{ $$ = new Template($1, $2); $2->setTemplate($$); }
		| TemplateSpec TMessage
		{ $$ = new Template($1, $2); $2->setTemplate($$); }
		;

MemberEList	: ';'
		{ $$ = 0; }
		| '{' MemberList '}' OptSemiColon
		{ $$ = $2; }
		;

MemberList	: /* Empty */
		{ $$ = 0; }
		| Member MemberList
		{ $$ = new MemberList($1, $2); }
		;

Member		: Entry ';'
		{ $$ = $1; }
		| Readonly ';'
		{ $$ = $1; }
		| ReadonlyMsg ';'
		{ $$ = $1; }
		;

Entry		: ENTRY EAttribs VOID Name EParam OptPure OptStackSize
		{ $$ = new Entry(lineno, $2|$6, new BuiltinType("void"), $4, $5, $7); }
		| ENTRY EAttribs OnePtrType Name EParam OptPure OptStackSize
		{ $$ = new Entry(lineno, $2|$6, $3, $4, $5, $7); }
		| ENTRY EAttribs Name EParam
		{ $$ = new Entry(lineno, $2, 0, $3, $4, 0); }
		;

EAttribs	: /* Empty */
		{ $$ = 0; }
		| '[' EAttribList ']'
		{ $$ = $2; }
		;

EAttribList	: EAttrib
		{ $$ = $1; }
		| EAttrib ',' EAttribList
		{ $$ = $1 | $3; }
		;

EAttrib		: THREADED
		{ $$ = STHREADED; }
		| SYNC
		{ $$ = SSYNC; }
		| EXCLUSIVE
		{ $$ = SLOCKED; }
		| VIRTUAL
		{ $$ = SVIRTUAL; }
		;

OptType		: /* Empty */
		{ $$ = 0; }
		| VOID
		{ $$ = new BuiltinType("void"); }
		| OnePtrType
		{ $$ = $1; }
		;

EParam		: '(' OptType ')'
		{ $$ = $2; }
		;

OptStackSize	: /* Empty */
		{ $$ = 0; }
		| STACKSIZE '=' NUMBER
		{ $$ = new Value($3); }
		;

OptPure		: /* Empty */
		{ $$ = 0; }
		| '=' NUMBER
		{ if(strcmp($2, "0")) { yyerror("expected 0"); exit(1); }
		  $$ = SPURE; 
		}
		;
%%
void yyerror(const char *mesg)
{
  cout << "Syntax error at line " << lineno << ": " << mesg << endl;
  // return 0;
}
