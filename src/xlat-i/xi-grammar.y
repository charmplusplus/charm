%expect 6
%{
#include <iostream>
#include <string>
#include <string.h>
#include "xi-symbol.h"
#include "EToken.h"
using namespace xi;
extern int yylex (void) ;
extern unsigned char in_comment;
void yyerror(const char *);
extern unsigned int lineno;
extern int in_bracket,in_braces,in_int_expr;
extern TList<Entry *> *connectEntries;
ModuleList *modlist;
namespace xi {
extern int macroDefined(const char *str, int istrue);
extern const char *python_doc;
void splitScopedName(const char* name, const char** scope, const char** basename);
}
%}

%union {
  ModuleList *modlist;
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
  MemberList *mbrlist;
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
  Chare::attrib_t cattr;
  SdagConstruct *sc;
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
%token SYNC IGET EXCLUSIVE IMMEDIATE SKIPSCHED INLINE VIRTUAL MIGRATABLE 
%token CREATEHERE CREATEHOME NOKEEP NOTRACE
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
%token FORWARD
%token IF
%token ELSE
%token CONNECT
%token PUBLISHES 
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

%type <modlist>		ModuleEList File
%type <module>		Module
%type <conslist>	ConstructEList ConstructList
%type <construct>	Construct
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
%type <type>		BaseType Type SimpleType OptTypeInit EReturn
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
%type <member>		Member NonEntryMember InitNode InitProc
%type <pupable>		PUPableClass
%type <includeFile>	IncludeFile
%type <tvar>		TVar
%type <tvarlist>	TVarList TemplateSpec
%type <val>		ArrayDim Dim DefaultParameter
%type <vallist>		DimList
%type <mv>		Var
%type <mvlist>		VarList
%type <intval>		ParamBraceStart ParamBraceEnd SParamBracketStart SParamBracketEnd StartIntExpr EndIntExpr
%type <sc>		Slist SingleConstruct Olist OptSdagCode HasElse ForwardList PublishesList OptPubList
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

Construct	: OptExtern '{' ConstructList '}' OptSemiColon
        { if($3) $3->setExtern($1); $$ = $3; }
        | NAMESPACE Name '{' ConstructList '}'
        { $$ = new Scope($2, $4); }
        | USING NAMESPACE QualName ';'
        { $$ = new UsingScope($3, false); }
        | USING QualName ';'
        { $$ = new UsingScope($2, true); }
        | OptExtern Module
        { $2->setExtern($1); $$ = $2; }
        | OptExtern NonEntryMember
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
        | EXTERN ENTRY EReturn QualNamedType Name OptTParams EParameters ';'
        {
          Entry *e = new Entry(lineno, 0, $3, $5, $7, 0, 0, 0, 0, 0);
          int isExtern = 1;
          e->setExtern(isExtern);
          e->targs = $6;
          e->label = new XStr;
          $4->print(*e->label);
          $$ = e;
        }
        | HashIFComment
        { $$ = NULL; }
        | HashIFDefComment
        { $$ = NULL; }
        | AccelBlock
        { $$ = $1; }
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
		//{ $$ = $1; }
		| CONST BaseType 
		{ $$ = $2; }
		| BaseType CONST
		{ $$ = $1; }
		;

Type		: BaseType '&'
                { $$ = new ReferenceType($1); }
		| BaseType
		{ $$ = $1; }
		;

ArrayDim	: NUMBER
		{ $$ = new Value($1); }
		| QualName
		{ $$ = new Value($1); }
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
		  Entry *tempEntry;
		  if (!connectEntries->empty()) {
		    tempEntry = connectEntries->begin();
		    MemberList *ml;
		    ml = new MemberList(tempEntry, 0);
		    tempEntry = connectEntries->next();
		    for(; !connectEntries->end(); tempEntry = connectEntries->next()) {
                      ml->appendMember(tempEntry); 
		    }
		    while (!connectEntries->empty())
		      connectEntries->pop();
                    $$ = ml; 
		  }
		  else {
		    $$ = 0; 
                  }
		}
		| Member MemberList
		{ $$ = new MemberList($1, $2); }
		;

NonEntryMember  : Readonly ';'
		{ $$ = $1; }
		| ReadonlyMsg ';'
		{ $$ = $1; }
		| InitProc ';'
		| InitNode ';'
		{ $$ = $1; }
		| PUPABLE PUPableClass ';'
		{ $$ = $2; }
		| INCLUDE IncludeFile ';'
		{ $$ = $2; }
		| CLASS Name ';'
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
		{ printf("Warning: deprecated use of initcall. Use initnode or initproc instead.\n"); 
		  $$ = new InitCall(lineno, $3, 1); }
		| INITCALL OptVoid QualName '(' OptVoid ')'
		{ printf("Warning: deprecated use of initcall. Use initnode or initproc instead.\n");
		  $$ = new InitCall(lineno, $3, 1); }
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

Member		: Entry ';'
		{ $$ = $1; }
                | TemplateSpec Entry ';'
                {
                  $2->tspec = $1;
                  $$ = $2;
                }
		| NonEntryMember
		{ $$ = $1; }
		;

Entry		: ENTRY EAttribs EReturn Name EParameters OptStackSize OptSdagCode
		{ 
		  if ($7 != 0) { 
		    $7->con1 = new SdagConstruct(SIDENT, $4);
                    $7->param = new ParamList($5);
                  }
		  $$ = new Entry(lineno, $2, $3, $4, $5, $6, $7, 0, 0); 
		}
		| ENTRY EAttribs Name EParameters OptSdagCode /*Constructor*/
		{ 
		  if ($5 != 0) {
		    $5->con1 = new SdagConstruct(SIDENT, $3);
                    $5->param = new ParamList($4);
                  }
		  Entry *e = new Entry(lineno, $2,     0, $3, $4,  0, $5, 0, 0);
		  if (e->param && e->param->isCkMigMsgPtr()) {
		    yyerror("Charm++ takes a CkMigrateMsg chare constructor for granted, but continuing anyway");
		    $$ = NULL;
		  } else
		    $$ = e;
		}
		| ENTRY '[' ACCEL ']' VOID Name EParameters AccelEParameters ParamBraceStart CCode ParamBraceEnd Name /* DMK : Accelerated Entry Method */
                {
                  int attribs = SACCEL;
                  const char* name = $6;
                  ParamList* paramList = $7;
                  ParamList* accelParamList = $8;
		  XStr* codeBody = new XStr($10);
                  const char* callbackName = $12;

                  $$ = new Entry(lineno, attribs, new BuiltinType("void"), name, paramList,
                                 0, 0, 0, 0, 0
                                );
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

EReturn		: VOID
		{ $$ = new BuiltinType("void"); }
		| OnePtrType
		{ $$ = $1; }
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

OptSdagCode	: /* Empty */
		{ $$ = 0; }
		| SingleConstruct
		{ $$ = new SdagConstruct(SSDAGENTRY, $1); }
		| '{' Slist '}'
		{ $$ = new SdagConstruct(SSDAGENTRY, $2); }
		;

Slist		: SingleConstruct
		{ $$ = new SdagConstruct(SSLIST, $1); }
		| SingleConstruct Slist
		{ $$ = new SdagConstruct(SSLIST, $1, $2);  }
		;

Olist		: SingleConstruct
		{ $$ = new SdagConstruct(SOLIST, $1); }
		| SingleConstruct Slist
		{ $$ = new SdagConstruct(SOLIST, $1, $2); } 
		;

OptPubList	: /* Empty */
		{ $$ = 0; }
		| PUBLISHES '(' PublishesList ')'
		{ $$ = $3; }
		;

PublishesList	: IDENT	
		{ $$ = new SdagConstruct(SPUBLISHES, new SdagConstruct(SIDENT, $1)); }
		| IDENT ',' PublishesList 
		{ $$ = new SdagConstruct(SPUBLISHES, new SdagConstruct(SIDENT, $1), $3);  }
		;

OptTraceName	: LITERAL
		 { $$ = $1; }
		|
		 { $$ = 0; }
		;

SingleConstruct : ATOMIC OptTraceName ParamBraceStart CCode ParamBraceEnd OptPubList 
                 {
		   $$ = buildAtomic($4, $6, $2);
		 }
		| CONNECT '(' IDENT EParameters ')' ParamBraceStart CCode '}'
		{  
		   in_braces = 0;
		   if (($4->isVoid() == 0) && ($4->isMessage() == 0))
                   {
		      connectEntries->append(new Entry(0, 0, new BuiltinType("void"), $3, 
	 	 			new ParamList(new Parameter(lineno, new PtrType( 
                                        new NamedType("CkMarshallMsg")), "_msg")), 0, 0, 0, 1, $4));
		   }
		   else  {
		      connectEntries->append(new Entry(0, 0, new BuiltinType("void"), $3, $4, 0, 0, 0, 1, $4));
                   }
                   $$ = new SdagConstruct(SCONNECT, $3, $7, $4);
		}
		| WHEN SEntryList '{' '}'
		{ $$ = new SdagConstruct(SWHEN, 0, 0, 0,0,0, 0,  $2); }
		| WHEN SEntryList SingleConstruct
		{ $$ = new SdagConstruct(SWHEN, 0, 0, 0,0,0, $3, $2); }
		| WHEN SEntryList '{' Slist '}'
		{ $$ = new SdagConstruct(SWHEN, 0, 0, 0,0,0, $4, $2); }
		| OVERLAP '{' Olist '}'
		{ $$ = new SdagConstruct(SOVERLAP,0, 0,0,0,0,$3, 0); }	
		| FOR StartIntExpr CCode ';' CCode ';' CCode  EndIntExpr '{' Slist '}'
		{ $$ = new SdagConstruct(SFOR, 0, new SdagConstruct(SINT_EXPR, $3), new SdagConstruct(SINT_EXPR, $5),
		             new SdagConstruct(SINT_EXPR, $7), 0, $10, 0); }
		| FOR StartIntExpr CCode ';' CCode ';' CCode  EndIntExpr SingleConstruct
		{ $$ = new SdagConstruct(SFOR, 0, new SdagConstruct(SINT_EXPR, $3), new SdagConstruct(SINT_EXPR, $5), 
		         new SdagConstruct(SINT_EXPR, $7), 0, $9, 0); }
		| FORALL '[' IDENT ']' StartIntExpr CCode ':' CCode ',' CCode  EndIntExpr SingleConstruct
		{ $$ = new SdagConstruct(SFORALL, 0, new SdagConstruct(SIDENT, $3), new SdagConstruct(SINT_EXPR, $6), 
		             new SdagConstruct(SINT_EXPR, $8), new SdagConstruct(SINT_EXPR, $10), $12, 0); }
		| FORALL '[' IDENT ']' StartIntExpr CCode ':' CCode ',' CCode  EndIntExpr '{' Slist '}' 
		{ $$ = new SdagConstruct(SFORALL, 0, new SdagConstruct(SIDENT, $3), new SdagConstruct(SINT_EXPR, $6), 
		                 new SdagConstruct(SINT_EXPR, $8), new SdagConstruct(SINT_EXPR, $10), $13, 0); }
		| IF StartIntExpr CCode EndIntExpr SingleConstruct HasElse
		{ $$ = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, $3), $6,0,0,$5,0); }
		| IF StartIntExpr CCode EndIntExpr '{' Slist '}' HasElse
		{ $$ = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, $3), $8,0,0,$6,0); }
		| WHILE StartIntExpr CCode EndIntExpr SingleConstruct 
		{ $$ = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, $3), 0,0,0,$5,0); }
		| WHILE StartIntExpr CCode EndIntExpr '{' Slist '}' 
		{ $$ = new SdagConstruct(SWHILE, 0, new SdagConstruct(SINT_EXPR, $3), 0,0,0,$6,0); }
		| FORWARD ForwardList ';'
		{ $$ = $2; }
		| ParamBraceStart CCode ParamBraceEnd
		{ $$ = buildAtomic($2, NULL, NULL); }

HasElse		: /* Empty */
		{ $$ = 0; }
		| ELSE SingleConstruct
		{ $$ = new SdagConstruct(SELSE, 0,0,0,0,0, $2,0); }
		| ELSE '{' Slist '}'
		{ $$ = new SdagConstruct(SELSE, 0,0,0,0,0, $3,0); }
		;
ForwardList	: IDENT 
		{ $$ = new SdagConstruct(SFORWARD, new SdagConstruct(SIDENT, $1)); }
		| IDENT ',' ForwardList 
		{ $$ = new SdagConstruct(SFORWARD, new SdagConstruct(SIDENT, $1), $3);  }
		;

EndIntExpr	: ')'
		{ in_int_expr = 0; $$ = 0; }
		;

StartIntExpr	: '('
		{ in_int_expr = 1; $$ = 0; }
		;

SEntry		: IDENT EParameters
		{ $$ = new Entry(lineno, 0, 0, $1, $2, 0, 0, 0, 0); }
		| IDENT SParamBracketStart CCode SParamBracketEnd EParameters 
		{ $$ = new Entry(lineno, 0, 0, $1, $5, 0, 0, $3, 0); }
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
void yyerror(const char *mesg)
{
    std::cerr << cur_file<<":"<<lineno<<": Charmxi syntax error> "
	      << mesg << std::endl;
}
