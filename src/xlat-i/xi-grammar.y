%expect 7
%{
#include <iostream>
#include <string>
#include <string.h>
#include "xi-symbol.h"
#include "sdag/constructs/Constructs.h"
#include "EToken.h"
#include "xi-Chare.h"

// Has to be a macro since YYABORT can only be used within rule actions.
#define ERROR(...) \
  if (xi::num_errors++ == xi::MAX_NUM_ERRORS) { \
    YYABORT;                                    \
  } else {                                      \
    xi::pretty_msg("error", __VA_ARGS__);       \
  }

#define WARNING(...) \
  if (enable_warnings) {                    \
    xi::pretty_msg("warning", __VA_ARGS__); \
  }

using namespace xi;
extern int yylex (void) ;
extern unsigned char in_comment;
extern unsigned int lineno;
extern int in_bracket,in_braces,in_int_expr;
extern std::list<Entry *> connectEntries;
AstChildren<Module> *modlist;

void yyerror(const char *);

namespace xi {

const int MAX_NUM_ERRORS = 10;
int num_errors = 0;

bool enable_warnings = true;

extern int macroDefined(const char *str, int istrue);
extern const char *python_doc;
extern char *fname;
void splitScopedName(const char* name, const char** scope, const char** basename);
void ReservedWord(int token, int fCol, int lCol);
}
%}

%locations

%union {
  AstChildren<Module> *modlist;
  Module *module;
  ConstructList *conslist;
  Construct *construct;
  TParam *tparam;
  TParamList *tparlist;
  Type *type;
  PtrType *ptype;
  NamedType *ntype;
  FuncType *ftype;
  Readonly *readonly;
  Message *message;
  Chare *chare;
  Entry *entry;
  EntryList *entrylist;
  Parameter *pname;
  ParamList *plist;
  Template *templat;
  TypeList *typelist;
  AstChildren<Member> *mbrlist;
  Member *member;
  TVar *tvar;
  TVarList *tvarlist;
  Value *val;
  ValueList *vallist;
  MsgVar *mv;
  MsgVarList *mvlist;
  PUPableClass *pupable;
  IncludeFile *includeFile;
  const char *strval;
  int intval;
  unsigned int cattr; // actually Chare::attrib_t, but referring to that creates nasty #include issues
  SdagConstruct *sc;
  IntExprConstruct *intexpr;
  WhenConstruct *when;
  SListConstruct *slist;
  CaseListConstruct *clist;
  OListConstruct *olist;
  SdagEntryConstruct *sentry;
  XStr* xstrptr;
  AccelBlock* accelBlock;
}

%token MODULE
%token MAINMODULE
%token EXTERN
%token READONLY
%token INITCALL
%token INITNODE
%token INITPROC
%token PUPABLE
%token <intval> CHARE MAINCHARE GROUP NODEGROUP ARRAY
%token MESSAGE
%token CONDITIONAL
%token CLASS
%token INCLUDE
%token STACKSIZE
%token THREADED
%token TEMPLATE
%token SYNC IGET EXCLUSIVE IMMEDIATE SKIPSCHED INLINE VIRTUAL MIGRATABLE AGGREGATE
%token CREATEHERE CREATEHOME NOKEEP NOTRACE APPWORK
%token VOID
%token CONST
%token PACKED
%token VARSIZE
%token ENTRY
%token FOR
%token FORALL
%token WHILE
%token WHEN
%token OVERLAP
%token ATOMIC
%token IF
%token ELSE
%token PYTHON LOCAL
%token NAMESPACE
%token USING
%token <strval> IDENT NUMBER LITERAL CPROGRAM HASHIF HASHIFDEF
%token <intval> INT LONG SHORT CHAR FLOAT DOUBLE UNSIGNED
%token ACCEL
%token READWRITE
%token WRITEONLY
%token ACCELBLOCK
%token MEMCRITICAL
%token REDUCTIONTARGET
%token CASE

%type <modlist>		ModuleEList File
%type <module>		Module
%type <conslist>	ConstructEList ConstructList
%type <construct>	Construct ConstructSemi
%type <strval>		Name QualName CCode CPROGRAM_List OptNameInit
%type <strval>		OptTraceName
%type <val>		OptStackSize
%type <intval>		OptExtern OptSemiColon MAttribs MAttribList MAttrib
%type <intval>		OptConditional MsgArray
%type <intval>		EAttribs EAttribList EAttrib OptVoid
%type <cattr>		CAttribs CAttribList CAttrib
%type <cattr>		ArrayAttribs ArrayAttribList ArrayAttrib
%type <tparam>		TParam
%type <tparlist>	TParamList TParamEList OptTParams
%type <type>		BaseDataType BaseType RestrictedType Type SimpleType OptTypeInit EReturn
%type <type>		BuiltinType
%type <ftype>		FuncType
%type <ntype>		NamedType QualNamedType ArrayIndexType
%type <ptype>		PtrType OnePtrType
%type <readonly>	Readonly ReadonlyMsg
%type <message>		Message TMessage
%type <chare>		Chare Group NodeGroup Array TChare TGroup TNodeGroup TArray
%type <entry>		Entry SEntry
%type <entrylist>	SEntryList
%type <templat>		Template
%type <pname>           Parameter ParamBracketStart AccelParameter AccelArrayParam
%type <plist>           ParamList EParameters AccelParamList AccelEParameters
%type <intval>          AccelBufferType
%type <xstrptr>         AccelInstName
%type <accelBlock>      AccelBlock
%type <typelist>	BaseList OptBaseList
%type <mbrlist>		MemberEList MemberList
%type <member>		Member MemberBody NonEntryMember InitNode InitProc UnexpectedToken
%type <pupable>		PUPableClass
%type <includeFile>	IncludeFile
%type <tvar>		TVar
%type <tvarlist>	TVarList TemplateSpec
%type <val>		ArrayDim Dim DefaultParameter
%type <vallist>		DimList
%type <mv>		Var
%type <mvlist>		VarList
%type <intval>		ParamBraceStart ParamBraceEnd SParamBracketStart SParamBracketEnd StartIntExpr EndIntExpr
%type <sc>		SingleConstruct HasElse
%type <intexpr>		IntExpr
%type <slist>		Slist
%type <clist>		CaseList
%type <olist>		Olist
%type <sentry>		OptSdagCode
%type <when>            WhenConstruct NonWhenConstruct
%type <intval>		PythonOptions

%%

File		: ModuleEList
		{ $$ = $1; modlist = $1; }
		;

ModuleEList	: /* Empty */
		{ 
		  $$ = 0; 
		}
		| Module ModuleEList
		{ $$ = new AstChildren<Module>(lineno, $1, $2); }
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

// Commented reserved words introduce parsing conflicts, so they're currently not handled
Name		: IDENT
		{ $$ = $1; }
		| MODULE { ReservedWord(MODULE, @$.first_column, @$.last_column); YYABORT; }
		| MAINMODULE { ReservedWord(MAINMODULE, @$.first_column, @$.last_column); YYABORT; }
		| EXTERN { ReservedWord(EXTERN, @$.first_column, @$.last_column); YYABORT; }
		/* | READONLY { ReservedWord(READONLY, @$.first_column, @$.last_column); YYABORT; } */
		| INITCALL { ReservedWord(INITCALL, @$.first_column, @$.last_column); YYABORT; }
		| INITNODE { ReservedWord(INITNODE, @$.first_column, @$.last_column); YYABORT; }
		| INITPROC { ReservedWord(INITPROC, @$.first_column, @$.last_column); YYABORT; }
		/* | PUPABLE { ReservedWord(PUPABLE, @$.first_column, @$.last_column); YYABORT; } */
		| CHARE { ReservedWord(CHARE, @$.first_column, @$.last_column); YYABORT; }
		| MAINCHARE { ReservedWord(MAINCHARE, @$.first_column, @$.last_column); YYABORT; }
		| GROUP { ReservedWord(GROUP, @$.first_column, @$.last_column); YYABORT; }
		| NODEGROUP { ReservedWord(NODEGROUP, @$.first_column, @$.last_column); YYABORT; }
		| ARRAY { ReservedWord(ARRAY, @$.first_column, @$.last_column); YYABORT; }
		/* | MESSAGE { ReservedWord(MESSAGE, @$.first_column, @$.last_column); YYABORT; } */
		/* | CONDITIONAL { ReservedWord(CONDITIONAL, @$.first_column, @$.last_column); YYABORT; } */
		/* | CLASS { ReservedWord(CLASS, @$.first_column, @$.last_column); YYABORT; } */
		| INCLUDE { ReservedWord(INCLUDE, @$.first_column, @$.last_column); YYABORT; }
		| STACKSIZE { ReservedWord(STACKSIZE, @$.first_column, @$.last_column); YYABORT; }
		| THREADED { ReservedWord(THREADED, @$.first_column, @$.last_column); YYABORT; }
		| TEMPLATE { ReservedWord(TEMPLATE, @$.first_column, @$.last_column); YYABORT; }
		| SYNC { ReservedWord(SYNC, @$.first_column, @$.last_column); YYABORT; }
		| IGET { ReservedWord(IGET, @$.first_column, @$.last_column); YYABORT; }
		| EXCLUSIVE { ReservedWord(EXCLUSIVE, @$.first_column, @$.last_column); YYABORT; }
		| IMMEDIATE { ReservedWord(IMMEDIATE, @$.first_column, @$.last_column); YYABORT; }
		| SKIPSCHED { ReservedWord(SKIPSCHED, @$.first_column, @$.last_column); YYABORT; }
		| INLINE { ReservedWord(INLINE, @$.first_column, @$.last_column); YYABORT; }
		| VIRTUAL { ReservedWord(VIRTUAL, @$.first_column, @$.last_column); YYABORT; }
		| MIGRATABLE { ReservedWord(MIGRATABLE, @$.first_column, @$.last_column); YYABORT; }
		| CREATEHERE { ReservedWord(CREATEHERE, @$.first_column, @$.last_column); YYABORT; }
		| CREATEHOME { ReservedWord(CREATEHOME, @$.first_column, @$.last_column); YYABORT; }
		| NOKEEP { ReservedWord(NOKEEP, @$.first_column, @$.last_column); YYABORT; }
		| NOTRACE { ReservedWord(NOTRACE, @$.first_column, @$.last_column); YYABORT; }
		| APPWORK { ReservedWord(APPWORK, @$.first_column, @$.last_column); YYABORT; }
		/* | VOID { ReservedWord(VOID, @$.first_column, @$.last_column); YYABORT; } */
		/* | CONST { ReservedWord(CONST, @$.first_column, @$.last_column); YYABORT; } */
		| PACKED { ReservedWord(PACKED, @$.first_column, @$.last_column); YYABORT; }
		| VARSIZE { ReservedWord(VARSIZE, @$.first_column, @$.last_column); YYABORT; }
		| ENTRY { ReservedWord(ENTRY, @$.first_column, @$.last_column); YYABORT; }
		| FOR { ReservedWord(FOR, @$.first_column, @$.last_column); YYABORT; }
		| FORALL { ReservedWord(FORALL, @$.first_column, @$.last_column); YYABORT; }
		| WHILE { ReservedWord(WHILE, @$.first_column, @$.last_column); YYABORT; }
		| WHEN { ReservedWord(WHEN, @$.first_column, @$.last_column); YYABORT; }
		| OVERLAP { ReservedWord(OVERLAP, @$.first_column, @$.last_column); YYABORT; }
		| ATOMIC { ReservedWord(ATOMIC, @$.first_column, @$.last_column); YYABORT; }
		| IF { ReservedWord(IF, @$.first_column, @$.last_column); YYABORT; }
		| ELSE { ReservedWord(ELSE, @$.first_column, @$.last_column); YYABORT; }
		/* | PYTHON { ReservedWord(PYTHON, @$.first_column, @$.last_column); YYABORT; } */
		| LOCAL { ReservedWord(LOCAL, @$.first_column, @$.last_column); YYABORT; }
		/* | NAMESPACE { ReservedWord(NAMESPACE, @$.first_column, @$.last_column); YYABORT; } */
		| USING { ReservedWord(USING, @$.first_column, @$.last_column); YYABORT; }
		| ACCEL { ReservedWord(ACCEL, @$.first_column, @$.last_column); YYABORT; }
		/* | READWRITE { ReservedWord(READWRITE, @$.first_column, @$.last_column); YYABORT; } */
		/* | WRITEONLY { ReservedWord(WRITEONLY, @$.first_column, @$.last_column); YYABORT; } */
		| ACCELBLOCK { ReservedWord(ACCELBLOCK, @$.first_column, @$.last_column); YYABORT; }
		| MEMCRITICAL { ReservedWord(MEMCRITICAL, @$.first_column, @$.last_column); YYABORT; }
		| REDUCTIONTARGET { ReservedWord(REDUCTIONTARGET, @$.first_column, @$.last_column); YYABORT; }
		| CASE { ReservedWord(CASE, @$.first_column, @$.last_column); YYABORT; }
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
		{ 
		    $$ = new Module(lineno, $2, $3); 
		}
		| MAINMODULE Name ConstructEList
		{  
		    $$ = new Module(lineno, $2, $3); 
		    $$->setMain();
		}
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

ConstructSemi   : USING NAMESPACE QualName
                { $$ = new UsingScope($3, false); }
                | USING QualName
                { $$ = new UsingScope($2, true); }
                | OptExtern NonEntryMember
                { $2->setExtern($1); $$ = $2; }
                | OptExtern Message
                { $2->setExtern($1); $$ = $2; }
                | EXTERN ENTRY EReturn QualNamedType Name OptTParams EParameters
                {
                  Entry *e = new Entry(lineno, 0, $3, $5, $7, 0, 0, 0, @1.first_line, @$.last_line);
                  int isExtern = 1;
                  e->setExtern(isExtern);
                  e->targs = $6;
                  e->label = new XStr;
                  $4->print(*e->label);
                  $$ = e;
                }
                ;

Construct	: OptExtern '{' ConstructList '}' OptSemiColon
        { if($3) $3->recurse<int&>($1, &Construct::setExtern); $$ = $3; }
        | NAMESPACE Name '{' ConstructList '}'
        { $$ = new Scope($2, $4); }
        | ConstructSemi ';'
        { $$ = $1; }
        | ConstructSemi UnexpectedToken
        {
          ERROR("preceding construct must be semicolon terminated",
                @$.first_column, @$.last_column);
          YYABORT;
        }
        | OptExtern Module
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
        | HashIFComment
        { $$ = NULL; }
        | HashIFDefComment
        { $$ = NULL; }
        | AccelBlock
        { $$ = $1; }
        | error
        {
          ERROR("invalid construct",
                @$.first_column, @$.last_column);
          YYABORT;
        }
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

OptTParams	:  /* Empty */
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
		{ $$ = new BuiltinType("unsigned long"); }
		| UNSIGNED LONG INT
		{ $$ = new BuiltinType("unsigned long"); }
		| UNSIGNED LONG LONG
		{ $$ = new BuiltinType("unsigned long long"); }
		| UNSIGNED SHORT
		{ $$ = new BuiltinType("unsigned short"); }
		| UNSIGNED CHAR
		{ $$ = new BuiltinType("unsigned char"); }
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

NamedType	: Name OptTParams { $$ = new NamedType($1,$2); };
QualNamedType	: QualName OptTParams { 
                    const char* basename, *scope;
                    splitScopedName($1, &scope, &basename);
                    $$ = new NamedType(basename, $2, scope);
                }
                ;

SimpleType	: BuiltinType
		{ $$ = $1; }
		| QualNamedType
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

FuncType	: BaseType '(' '*' Name ')' '(' ParamList ')'
		{ $$ = new FuncType($1, $4, $7); }
		;

BaseType	: SimpleType
		{ $$ = $1; }
		| OnePtrType
		{ $$ = $1; }
		| PtrType
		{ $$ = $1; }
		| FuncType
		{ $$ = $1; }
		| CONST BaseType 
		{ $$ = new ConstType($2); }
		| BaseType CONST
		{ $$ = new ConstType($1); }
		;

BaseDataType	: SimpleType
		{ $$ = $1; }
		| OnePtrType
		{ $$ = $1; }
		| PtrType
		{ $$ = $1; }
		| CONST BaseDataType
		{ $$ = new ConstType($2); }
		| BaseDataType CONST
		{ $$ = new ConstType($1); }
		;

RestrictedType : BaseDataType '&'
		{ $$ = new ReferenceType($1); }
		| BaseDataType
		{ $$ = $1; }
		;

Type		: BaseType '&'
                { $$ = new ReferenceType($1); }
		| BaseType
		{ $$ = $1; }
		;

ArrayDim	: CCode
		{ $$ = new Value($1); }
		;

Dim		: SParamBracketStart ArrayDim SParamBracketEnd
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

ReadonlyMsg	: READONLY MESSAGE SimpleType '*'  QualName DimList
		{ $$ = new Readonly(lineno, $3, $5, $6, 1); }
		;

OptVoid		: /*Empty*/
		{ $$ = 0;}
		| VOID
		{ $$ = 0;}
		;

MAttribs	: /* Empty */
		{ $$ = 0; }
		| '[' MAttribList ']'
		{ 
		  /*
		  printf("Warning: Message attributes are being phased out.\n");
		  printf("Warning: Please remove them from interface files.\n");
		  */
		  $$ = $2; 
		}
		;

MAttribList	: MAttrib
		{ $$ = $1; }
		| MAttrib ',' MAttribList
		{ $$ = $1 | $3; }
		;

MAttrib		: PACKED
		{ $$ = 0; }
		| VARSIZE
		{ $$ = 0; }
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

PythonOptions	: /* Empty */
		{ python_doc = NULL; $$ = 0; }
		| LITERAL
		{ python_doc = $1; $$ = 0; }
		;

ArrayAttrib	: PYTHON
		{ $$ = Chare::CPYTHON; }
		;

ArrayAttribs	: /* Empty */
		{ $$ = 0; }
		| '[' ArrayAttribList ']'
		{ $$ = $2; }
		;

ArrayAttribList	: ArrayAttrib
		{ $$ = $1; }
		| ArrayAttrib ',' ArrayAttribList
		{ $$ = $1 | $3; }
		;

CAttrib		: MIGRATABLE
		{ $$ = Chare::CMIGRATABLE; }
		| PYTHON
		{ $$ = Chare::CPYTHON; }
		;

OptConditional	: /* Empty */
		{ $$ = 0; }
		| CONDITIONAL
		{ $$ = 1; }

MsgArray	: /* Empty */
		{ $$ = 0; }
		| '[' ']'
		{ $$ = 1; }

Var		: OptConditional Type Name MsgArray ';'
		{ $$ = new MsgVar($2, $3, $1, $4); }
		;

VarList		: Var
		{ $$ = new MsgVarList($1); }
		| Var VarList
		{ $$ = new MsgVarList($1, $2); }
		;

Message		: MESSAGE MAttribs NamedType
		{ $$ = new Message(lineno, $3); }
		| MESSAGE MAttribs NamedType '{' VarList '}'
		{ $$ = new Message(lineno, $3, $5); }
		;

OptBaseList	: /* Empty */
		{ $$ = 0; }
		| ':' BaseList
		{ $$ = $2; }
		;

BaseList	: QualNamedType
		{ $$ = new TypeList($1); }
		| QualNamedType ',' BaseList
		{ $$ = new TypeList($1, $3); }
		;

Chare		: CHARE CAttribs NamedType OptBaseList MemberEList
		{ $$ = new Chare(lineno, $2|Chare::CCHARE, $3, $4, $5); }
		| MAINCHARE CAttribs NamedType OptBaseList MemberEList
		{ $$ = new MainChare(lineno, $2, $3, $4, $5); }
		;

Group		: GROUP CAttribs NamedType OptBaseList MemberEList
		{ $$ = new Group(lineno, $2, $3, $4, $5); }
		;

NodeGroup	: NODEGROUP CAttribs NamedType OptBaseList MemberEList
		{ $$ = new NodeGroup(lineno, $2, $3, $4, $5); }
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

Array		: ARRAY ArrayAttribs ArrayIndexType NamedType OptBaseList MemberEList
		{  $$ = new Array(lineno, $2, $3, $4, $5, $6); }
		| ARRAY ArrayIndexType ArrayAttribs NamedType OptBaseList MemberEList
		{  $$ = new Array(lineno, $3, $2, $4, $5, $6); }
		;

TChare		: CHARE CAttribs Name OptBaseList MemberEList
		{ $$ = new Chare(lineno, $2|Chare::CCHARE, new NamedType($3), $4, $5);}
		| MAINCHARE CAttribs Name OptBaseList MemberEList
		{ $$ = new MainChare(lineno, $2, new NamedType($3), $4, $5); }
		;

TGroup		: GROUP CAttribs Name OptBaseList MemberEList
		{ $$ = new Group(lineno, $2, new NamedType($3), $4, $5); }
		;

TNodeGroup	: NODEGROUP CAttribs Name OptBaseList MemberEList
		{ $$ = new NodeGroup( lineno, $2, new NamedType($3), $4, $5); }
		;

TArray		: ARRAY ArrayIndexType Name OptBaseList MemberEList
		{ $$ = new Array( lineno, 0, $2, new NamedType($3), $4, $5); }
		;

TMessage	: MESSAGE MAttribs Name ';'
		{ $$ = new Message(lineno, new NamedType($3)); }
		| MESSAGE MAttribs Name '{' VarList '}' ';'
		{ $$ = new Message(lineno, new NamedType($3), $5); }
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
		| '=' QualNamedType
		{
		  XStr typeStr;
		  $2->print(typeStr);
		  char *tmp = strdup(typeStr.get_string());
		  $$ = tmp;
		}
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
		{ 
                  if (!connectEntries.empty()) {
                    $$ = new AstChildren<Member>(connectEntries);
		  } else {
		    $$ = 0; 
                  }
		}
		| Member MemberList
		{ $$ = new AstChildren<Member>(-1, $1, $2); }
		;

NonEntryMember  : Readonly
		{ $$ = $1; }
		| ReadonlyMsg
		{ $$ = $1; }
		| InitProc
		| InitNode
		{ $$ = $1; }
		| PUPABLE PUPableClass
		{ $$ = $2; }
		| INCLUDE IncludeFile
		{ $$ = $2; }
		| CLASS Name
		{ $$ = new ClassDeclaration(lineno,$2); } 
		;

InitNode	: INITNODE OptVoid QualName
		{ $$ = new InitCall(lineno, $3, 1); }
		| INITNODE OptVoid QualName '(' OptVoid ')'
		{ $$ = new InitCall(lineno, $3, 1); }
                | INITNODE OptVoid QualName '<' TParamList '>' '(' OptVoid ')'
                { $$ = new InitCall(lineno,
				    strdup((std::string($3) + '<' +
					    ($5)->to_string() + '>').c_str()),
				    1);
		}
		| INITCALL OptVoid QualName
		{
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          @1.first_column, @1.last_column, @1.first_line);
		  $$ = new InitCall(lineno, $3, 1);
		}
		| INITCALL OptVoid QualName '(' OptVoid ')'
		{
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          @1.first_column, @1.last_column, @1.first_line);
		  $$ = new InitCall(lineno, $3, 1);
		}
		;


InitProc	: INITPROC OptVoid QualName
		{ $$ = new InitCall(lineno, $3, 0); }
		| INITPROC OptVoid QualName '(' OptVoid ')'
		{ $$ = new InitCall(lineno, $3, 0); }
                | INITPROC OptVoid QualName '<' TParamList '>' '(' OptVoid ')'
                { $$ = new InitCall(lineno,
				    strdup((std::string($3) + '<' +
					    ($5)->to_string() + '>').c_str()),
				    0);
		}
                | INITPROC '[' ACCEL ']' OptVoid QualName '(' OptVoid ')'
                {
                  InitCall* rtn = new InitCall(lineno, $6, 0);
                  rtn->setAccel();
                  $$ = rtn;
		}
		;

PUPableClass    : QualNamedType
		{ $$ = new PUPableClass(lineno,$1,0); } 
		| QualNamedType ',' PUPableClass
		{ $$ = new PUPableClass(lineno,$1,$3); }
		;
IncludeFile    : LITERAL
		{ $$ = new IncludeFile(lineno,$1); } 
		;

Member : MemberBody
		{ $$ = $1; }
		;

MemberBody	: Entry
		{ $$ = $1; }
                | TemplateSpec Entry
                {
                  $2->tspec = $1;
                  $$ = $2;
                }
		| NonEntryMember ';'
		{ $$ = $1; }
        | error
        {
          ERROR("invalid SDAG member",
                @$.first_column, @$.last_column);
          YYABORT;
        }
		;

UnexpectedToken : ENTRY
                { $$ = 0; }
                | '}'
                { $$ = 0; }
                | INITCALL
                { $$ = 0; }
                | INITNODE
                { $$ = 0; }
                | INITPROC
                { $$ = 0; }
                | CHARE
                { $$ = 0; }
                | MAINCHARE
                { $$ = 0; }
                | ARRAY
                { $$ = 0; }
                | GROUP
                { $$ = 0; }
                | NODEGROUP
                { $$ = 0; }
                | READONLY
                { $$ = 0; }

Entry		: ENTRY EAttribs EReturn Name EParameters OptStackSize OptSdagCode
		{ 
                  $$ = new Entry(lineno, $2, $3, $4, $5, $6, $7, (const char *) NULL, @1.first_line, @$.last_line);
		  if ($7 != 0) { 
		    $7->con1 = new SdagConstruct(SIDENT, $4);
                    $7->entry = $$;
                    $7->con1->entry = $$;
                    $7->param = new ParamList($5);
                  }
		}
		| ENTRY EAttribs Name EParameters OptSdagCode /*Constructor*/
		{ 
                  Entry *e = new Entry(lineno, $2, 0, $3, $4,  0, $5, (const char *) NULL, @1.first_line, @$.last_line);
                  if ($5 != 0) {
		    $5->con1 = new SdagConstruct(SIDENT, $3);
                    $5->entry = e;
                    $5->con1->entry = e;
                    $5->param = new ParamList($4);
                  }
		  if (e->param && e->param->isCkMigMsgPtr()) {
		    WARNING("CkMigrateMsg chare constructor is taken for granted",
		            @$.first_column, @$.last_column);
		    $$ = NULL;
		  } else {
		    $$ = e;
		  }
		}
		| ENTRY '[' ACCEL ']' VOID Name EParameters AccelEParameters ParamBraceStart CCode ParamBraceEnd Name ';' /* DMK : Accelerated Entry Method */
                {
                  int attribs = SACCEL;
                  const char* name = $6;
                  ParamList* paramList = $7;
                  ParamList* accelParamList = $8;
		  XStr* codeBody = new XStr($10);
                  const char* callbackName = $12;

                  $$ = new Entry(lineno, attribs, new BuiltinType("void"), name, paramList, 0, 0, 0 );
                  $$->setAccelParam(accelParamList);
                  $$->setAccelCodeBody(codeBody);
                  $$->setAccelCallbackName(new XStr(callbackName));
                }
		;

AccelBlock      : ACCELBLOCK ParamBraceStart CCode ParamBraceEnd ';'
                { $$ = new AccelBlock(lineno, new XStr($3)); }
                | ACCELBLOCK ';'
                { $$ = new AccelBlock(lineno, NULL); }
                ;

EReturn	: RestrictedType
		{ $$ = $1; }
		;

EAttribs	: /* Empty */
		{ $$ = 0; }
		| '[' EAttribList ']'
		{ $$ = $2; }
		| error
		{ ERROR("invalid entry method attribute list",
		        @$.first_column, @$.last_column);
		  YYABORT;
		}
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
                | IGET
                { $$ = SIGET; }
		| EXCLUSIVE
		{ $$ = SLOCKED; }
		| CREATEHERE
		{ $$ = SCREATEHERE; }
		| CREATEHOME
		{ $$ = SCREATEHOME; }
		| NOKEEP
		{ $$ = SNOKEEP; }
		| NOTRACE
		{ $$ = SNOTRACE; }
		| APPWORK 
		{ $$ = SAPPWORK; }
		| IMMEDIATE
                { $$ = SIMMEDIATE; }
		| SKIPSCHED
                { $$ = SSKIPSCHED; }
		| INLINE
                { $$ = SINLINE; }
		| LOCAL
                { $$ = SLOCAL; }
		| PYTHON PythonOptions
                { $$ = SPYTHON; }
		| MEMCRITICAL
		{ $$ = SMEM; }
                | REDUCTIONTARGET
                { $$ = SREDUCE; }
                | AGGREGATE
		{
#ifdef CMK_USING_XLC
        WARNING("a known bug in xl compilers (PMR 18366,122,000) currently breaks "
                "aggregate entry methods.\n"
                "Until a fix is released, this tag will be ignored on those compilers.",
                @1.first_column, @1.last_column, @1.first_line);
        $$ = 0;
#else
        $$ = SAGGREGATE;
#endif
    }
		| error
		{
		  ERROR("invalid entry method attribute",
		        @1.first_column, @1.last_column);
		  yyclearin;
		  yyerrok;
		}
		;

DefaultParameter: LITERAL
		{ $$ = new Value($1); }
		| NUMBER
		{ $$ = new Value($1); }
		| QualName
		{ $$ = new Value($1); }
		;

CPROGRAM_List   :  /* Empty */
		{ $$ = ""; }
		| CPROGRAM
		{ $$ = $1; }
		| CPROGRAM ',' CPROGRAM_List
		{  /*Returned only when in_bracket*/
			char *tmp = new char[strlen($1)+strlen($3)+3];
			sprintf(tmp,"%s, %s", $1, $3);
			$$ = tmp;
		}
		;

CCode		: /* Empty */
		{ $$ = ""; }
		| CPROGRAM
		{ $$ = $1; }
		| CPROGRAM '[' CCode ']' CCode
		{  /*Returned only when in_bracket*/
			char *tmp = new char[strlen($1)+strlen($3)+strlen($5)+3];
			sprintf(tmp,"%s[%s]%s", $1, $3, $5);
			$$ = tmp;
		}
		| CPROGRAM '{' CCode '}' CCode
		{ /*Returned only when in_braces*/
			char *tmp = new char[strlen($1)+strlen($3)+strlen($5)+3];
			sprintf(tmp,"%s{%s}%s", $1, $3, $5);
			$$ = tmp;
		}
		| CPROGRAM '(' CPROGRAM_List ')' CCode
		{ /*Returned only when in_braces*/
			char *tmp = new char[strlen($1)+strlen($3)+strlen($5)+3];
			sprintf(tmp,"%s(%s)%s", $1, $3, $5);
			$$ = tmp;
		}
		|'(' CCode ')' CCode
		{ /*Returned only when in_braces*/
			char *tmp = new char[strlen($2)+strlen($4)+3];
			sprintf(tmp,"(%s)%s", $2, $4);
			$$ = tmp;
		}
		;

ParamBracketStart : Type Name '['
		{  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			$$ = new Parameter(lineno, $1,$2);
		}
		;

ParamBraceStart : '{'
		{ 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			$$ = 0;
		}
		;

ParamBraceEnd 	: '}'
		{ 
			in_braces=0;
			$$ = 0;
		}
		;

Parameter	: Type
		{ $$ = new Parameter(lineno, $1);}
		| Type Name OptConditional
		{ $$ = new Parameter(lineno, $1,$2); $$->setConditional($3); }
		| Type Name '=' DefaultParameter
		{ $$ = new Parameter(lineno, $1,$2,0,$4);} 
		| ParamBracketStart CCode ']'
		{ /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			$$ = new Parameter(lineno, $1->getType(), $1->getName() ,$2);
		} 
		;

AccelBufferType : READONLY  { $$ = Parameter::ACCEL_BUFFER_TYPE_READONLY; }
                | READWRITE { $$ = Parameter::ACCEL_BUFFER_TYPE_READWRITE; }
                | WRITEONLY { $$ = Parameter::ACCEL_BUFFER_TYPE_WRITEONLY; }
                ;

AccelInstName   : Name { $$ = new XStr($1); }
                | AccelInstName '-' '>' Name { $$ = new XStr(""); *($$) << *($1) << "->" << $4; }
                | AccelInstName '.' Name { $$ = new XStr(""); *($$) << *($1) << "." << $3; }
                | AccelInstName '[' AccelInstName ']'
                {
                  $$ = new XStr("");
                  *($$) << *($1) << "[" << *($3) << "]";
                  delete $1;
                  delete $3;
                }
                | AccelInstName '[' NUMBER ']'
                {
                  $$ = new XStr("");
                  *($$) << *($1) << "[" << $3 << "]";
                  delete $1;
                }
                | AccelInstName '(' AccelInstName ')'
                {
                  $$ = new XStr("");
                  *($$) << *($1) << "(" << *($3) << ")";
                  delete $1;
                  delete $3;
                }
                ;

AccelArrayParam : ParamBracketStart CCode ']'
                {
                  in_bracket = 0;
                  $$ = new Parameter(lineno, $1->getType(), $1->getName(), $2);
                }
                ;

AccelParameter	: AccelBufferType ':' Type Name '<' AccelInstName '>'
                {
                  $$ = new Parameter(lineno, $3, $4);
                  $$->setAccelInstName($6);
                  $$->setAccelBufferType($1);
                }
                | Type Name '<' AccelInstName '>'
                {
		  $$ = new Parameter(lineno, $1, $2);
                  $$->setAccelInstName($4);
                  $$->setAccelBufferType(Parameter::ACCEL_BUFFER_TYPE_READWRITE);
		}
                | AccelBufferType ':' AccelArrayParam '<' AccelInstName '>'
                {
                  $$ = $3;
                  $$->setAccelInstName($5);
                  $$->setAccelBufferType($1);
		}
		;

ParamList	: Parameter
		{ $$ = new ParamList($1); }
		| Parameter ',' ParamList
		{ $$ = new ParamList($1,$3); }
		;

AccelParamList	: AccelParameter
		{ $$ = new ParamList($1); }
		| AccelParameter ',' AccelParamList
		{ $$ = new ParamList($1,$3); }
		;

EParameters	: '(' ParamList ')'
		{ $$ = $2; }
		| '(' ')'
		{ $$ = new ParamList(new Parameter(0, new BuiltinType("void"))); }
		;

AccelEParameters  : '[' AccelParamList ']'
                  { $$ = $2; }
		  | '[' ']'
		  { $$ = 0; }
		  ;

OptStackSize	: /* Empty */
		{ $$ = 0; }
		| STACKSIZE '=' NUMBER
		{ $$ = new Value($3); }
		;

OptSdagCode	: ';' /* Empty */
		{ $$ = 0; }
		| SingleConstruct
		{ $$ = new SdagEntryConstruct($1); }
		| '{' Slist '}' OptSemiColon
		{ $$ = new SdagEntryConstruct($2); }
		;

Slist		: SingleConstruct
		{ $$ = new SListConstruct($1); }
		| SingleConstruct Slist
		{ $$ = new SListConstruct($1, $2);  }
		;

Olist		: SingleConstruct
		{ $$ = new OListConstruct($1); }
		| SingleConstruct Slist
		{ $$ = new OListConstruct($1, $2); }
		;

CaseList	: WhenConstruct
		{ $$ = new CaseListConstruct($1); }
		| WhenConstruct CaseList
		{ $$ = new CaseListConstruct($1, $2); }
		| NonWhenConstruct
		{
		  ERROR("case blocks can only contain when clauses",
		        @1.first_column, @1.last_column);
		  $$ = 0;
		}
		;

OptTraceName	: LITERAL
		 { $$ = $1; }
		|
		 { $$ = 0; }
		;

WhenConstruct   : WHEN SEntryList '{' '}'
		{ $$ = new WhenConstruct($2, 0); }
		| WHEN SEntryList SingleConstruct
		{ $$ = new WhenConstruct($2, $3); }
		| WHEN SEntryList '{' Slist '}'
		{ $$ = new WhenConstruct($2, $4); }
		;

NonWhenConstruct : ATOMIC OptTraceName ParamBraceStart CCode ParamBraceEnd
		{ $$ = 0; }
		| OVERLAP '{' Olist '}'
		{ $$ = 0; }
		| CASE '{' CaseList '}'
		{ $$ = 0; }
		| FOR StartIntExpr CCode ';' CCode ';' CCode  EndIntExpr '{' Slist '}'
		{ $$ = 0; }
		| FOR StartIntExpr CCode ';' CCode ';' CCode  EndIntExpr SingleConstruct
		{ $$ = 0; }
		| FORALL '[' IDENT ']' StartIntExpr CCode ':' CCode ',' CCode  EndIntExpr SingleConstruct
		{ $$ = 0; }
		| FORALL '[' IDENT ']' StartIntExpr CCode ':' CCode ',' CCode  EndIntExpr '{' Slist '}' 
		{ $$ = 0; }
		| IF StartIntExpr CCode EndIntExpr SingleConstruct HasElse
		{ $$ = 0; }
		| IF StartIntExpr CCode EndIntExpr '{' Slist '}' HasElse
		{ $$ = 0; }
		| WHILE StartIntExpr CCode EndIntExpr SingleConstruct 
		{ $$ = 0; }
		| WHILE StartIntExpr CCode EndIntExpr '{' Slist '}' 
		{ $$ = 0; }
		| ParamBraceStart CCode ParamBraceEnd
		{ $$ = 0; }
		;

SingleConstruct : ATOMIC OptTraceName ParamBraceStart CCode ParamBraceEnd
		{ $$ = new AtomicConstruct($4, $2, @3.first_line); }
		| OVERLAP '{' Olist '}'
		{ $$ = new OverlapConstruct($3); }	
		| WhenConstruct
		{ $$ = $1; }
		| CASE '{' CaseList '}'
		{ $$ = new CaseConstruct($3); }
		| FOR StartIntExpr IntExpr ';' IntExpr ';' IntExpr  EndIntExpr '{' Slist '}'
		{ $$ = new ForConstruct($3, $5, $7, $10); }
		| FOR StartIntExpr IntExpr ';' IntExpr ';' IntExpr  EndIntExpr SingleConstruct
		{ $$ = new ForConstruct($3, $5, $7, $9); }
		| FORALL '[' IDENT ']' StartIntExpr IntExpr ':' IntExpr ',' IntExpr  EndIntExpr SingleConstruct
		{ $$ = new ForallConstruct(new SdagConstruct(SIDENT, $3), $6,
		             $8, $10, $12); }
		| FORALL '[' IDENT ']' StartIntExpr IntExpr ':' IntExpr ',' IntExpr  EndIntExpr '{' Slist '}'
		{ $$ = new ForallConstruct(new SdagConstruct(SIDENT, $3), $6,
		             $8, $10, $13); }
		| IF StartIntExpr IntExpr EndIntExpr SingleConstruct HasElse
		{ $$ = new IfConstruct($3, $5, $6); }
		| IF StartIntExpr IntExpr EndIntExpr '{' Slist '}' HasElse
		{ $$ = new IfConstruct($3, $6, $8); }
		| WHILE StartIntExpr IntExpr EndIntExpr SingleConstruct
		{ $$ = new WhileConstruct($3, $5); }
		| WHILE StartIntExpr IntExpr EndIntExpr '{' Slist '}'
		{ $$ = new WhileConstruct($3, $6); }
		| ParamBraceStart CCode ParamBraceEnd
		{ $$ = new AtomicConstruct($2, NULL, @$.first_line); }
		| error
		{
		  ERROR("unknown SDAG construct or malformed entry method declaration.\n"
		        "You may have forgotten to terminate a previous entry method declaration with a"
		        " semicolon or forgotten to mark a block of sequential SDAG code as 'serial'",
		        @$.first_column, @$.last_column);
		  YYABORT;
		}
		;

HasElse		: /* Empty */
		{ $$ = 0; }
		| ELSE SingleConstruct
		{ $$ = new ElseConstruct($2); }
		| ELSE '{' Slist '}'
		{ $$ = new ElseConstruct($3); }
		;

IntExpr	: CCode
		{ $$ = new IntExprConstruct($1); }
		;

EndIntExpr	: ')'
		{ in_int_expr = 0; $$ = 0; }
		;

StartIntExpr	: '('
		{ in_int_expr = 1; $$ = 0; }
		;

SEntry		: IDENT EParameters
		{
		  $$ = new Entry(lineno, 0, 0, $1, $2, 0, 0, 0, @$.first_line, @$.last_line);
		}
		| IDENT SParamBracketStart CCode SParamBracketEnd EParameters 
		{
		  $$ = new Entry(lineno, 0, 0, $1, $5, 0, 0, $3, @$.first_line, @$.last_line);
		}
		;

SEntryList	: SEntry 
		{ $$ = new EntryList($1); }
		| SEntry ',' SEntryList
		{ $$ = new EntryList($1,$3); }
		;

SParamBracketStart : '['
		   { in_bracket=1; } 
		   ;
SParamBracketEnd   : ']'
		   { in_bracket=0; } 
		   ;

HashIFComment	: HASHIF Name
		{ if (!macroDefined($2, 1)) in_comment = 1; }
		;

HashIFDefComment: HASHIFDEF Name
		{ if (!macroDefined($2, 0)) in_comment = 1; }
		;

%%

void yyerror(const char *msg) { }
