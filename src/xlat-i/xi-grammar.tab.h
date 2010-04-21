#ifndef YY_parse_h_included
#define YY_parse_h_included
/*#define YY_USE_CLASS 
*/
#line 1 "/usr/share/bison++/bison.h"
/* before anything */
#ifdef c_plusplus
 #ifndef __cplusplus
  #define __cplusplus
 #endif
#endif


 #line 8 "/usr/share/bison++/bison.h"

#line 21 "xi-grammar.y"
typedef union {
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
} yy_parse_stype;
#define YY_parse_STYPE yy_parse_stype
#ifndef YY_USE_CLASS
#define YYSTYPE yy_parse_stype
#endif

#line 21 "/usr/share/bison++/bison.h"
 /* %{ and %header{ and %union, during decl */
#ifndef YY_parse_COMPATIBILITY
 #ifndef YY_USE_CLASS
  #define  YY_parse_COMPATIBILITY 1
 #else
  #define  YY_parse_COMPATIBILITY 0
 #endif
#endif

#if YY_parse_COMPATIBILITY != 0
/* backward compatibility */
 #ifdef YYLTYPE
  #ifndef YY_parse_LTYPE
   #define YY_parse_LTYPE YYLTYPE
/* WARNING obsolete !!! user defined YYLTYPE not reported into generated header */
/* use %define LTYPE */
  #endif
 #endif
/*#ifdef YYSTYPE*/
  #ifndef YY_parse_STYPE
   #define YY_parse_STYPE YYSTYPE
  /* WARNING obsolete !!! user defined YYSTYPE not reported into generated header */
   /* use %define STYPE */
  #endif
/*#endif*/
 #ifdef YYDEBUG
  #ifndef YY_parse_DEBUG
   #define  YY_parse_DEBUG YYDEBUG
   /* WARNING obsolete !!! user defined YYDEBUG not reported into generated header */
   /* use %define DEBUG */
  #endif
 #endif 
 /* use goto to be compatible */
 #ifndef YY_parse_USE_GOTO
  #define YY_parse_USE_GOTO 1
 #endif
#endif

/* use no goto to be clean in C++ */
#ifndef YY_parse_USE_GOTO
 #define YY_parse_USE_GOTO 0
#endif

#ifndef YY_parse_PURE

 #line 65 "/usr/share/bison++/bison.h"

#line 65 "/usr/share/bison++/bison.h"
/* YY_parse_PURE */
#endif


 #line 68 "/usr/share/bison++/bison.h"

#line 68 "/usr/share/bison++/bison.h"
/* prefix */

#ifndef YY_parse_DEBUG

 #line 71 "/usr/share/bison++/bison.h"

#line 71 "/usr/share/bison++/bison.h"
/* YY_parse_DEBUG */
#endif

#ifndef YY_parse_LSP_NEEDED

 #line 75 "/usr/share/bison++/bison.h"

#line 75 "/usr/share/bison++/bison.h"
 /* YY_parse_LSP_NEEDED*/
#endif

/* DEFAULT LTYPE*/
#ifdef YY_parse_LSP_NEEDED
 #ifndef YY_parse_LTYPE
  #ifndef BISON_YYLTYPE_ISDECLARED
   #define BISON_YYLTYPE_ISDECLARED
typedef
  struct yyltype
    {
      int timestamp;
      int first_line;
      int first_column;
      int last_line;
      int last_column;
      char *text;
   }
  yyltype;
  #endif

  #define YY_parse_LTYPE yyltype
 #endif
#endif

/* DEFAULT STYPE*/
#ifndef YY_parse_STYPE
 #define YY_parse_STYPE int
#endif

/* DEFAULT MISCELANEOUS */
#ifndef YY_parse_PARSE
 #define YY_parse_PARSE yyparse
#endif

#ifndef YY_parse_LEX
 #define YY_parse_LEX yylex
#endif

#ifndef YY_parse_LVAL
 #define YY_parse_LVAL yylval
#endif

#ifndef YY_parse_LLOC
 #define YY_parse_LLOC yylloc
#endif

#ifndef YY_parse_CHAR
 #define YY_parse_CHAR yychar
#endif

#ifndef YY_parse_NERRS
 #define YY_parse_NERRS yynerrs
#endif

#ifndef YY_parse_DEBUG_FLAG
 #define YY_parse_DEBUG_FLAG yydebug
#endif

#ifndef YY_parse_ERROR
 #define YY_parse_ERROR yyerror
#endif

#ifndef YY_parse_PARSE_PARAM
 #ifndef __STDC__
  #ifndef __cplusplus
   #ifndef YY_USE_CLASS
    #define YY_parse_PARSE_PARAM
    #ifndef YY_parse_PARSE_PARAM_DEF
     #define YY_parse_PARSE_PARAM_DEF
    #endif
   #endif
  #endif
 #endif
 #ifndef YY_parse_PARSE_PARAM
  #define YY_parse_PARSE_PARAM void
 #endif
#endif

/* TOKEN C */
#ifndef YY_USE_CLASS

 #ifndef YY_parse_PURE
  #ifndef yylval
   extern YY_parse_STYPE YY_parse_LVAL;
  #else
   #if yylval != YY_parse_LVAL
    extern YY_parse_STYPE YY_parse_LVAL;
   #else
    #warning "Namespace conflict, disabling some functionality (bison++ only)"
   #endif
  #endif
 #endif


 #line 169 "/usr/share/bison++/bison.h"
#define	MODULE	258
#define	MAINMODULE	259
#define	EXTERN	260
#define	READONLY	261
#define	INITCALL	262
#define	INITNODE	263
#define	INITPROC	264
#define	PUPABLE	265
#define	CHARE	266
#define	MAINCHARE	267
#define	GROUP	268
#define	NODEGROUP	269
#define	ARRAY	270
#define	MESSAGE	271
#define	CONDITIONAL	272
#define	CLASS	273
#define	INCLUDE	274
#define	STACKSIZE	275
#define	THREADED	276
#define	TEMPLATE	277
#define	SYNC	278
#define	IGET	279
#define	EXCLUSIVE	280
#define	IMMEDIATE	281
#define	SKIPSCHED	282
#define	INLINE	283
#define	VIRTUAL	284
#define	MIGRATABLE	285
#define	CREATEHERE	286
#define	CREATEHOME	287
#define	NOKEEP	288
#define	NOTRACE	289
#define	VOID	290
#define	CONST	291
#define	PACKED	292
#define	VARSIZE	293
#define	ENTRY	294
#define	FOR	295
#define	FORALL	296
#define	WHILE	297
#define	WHEN	298
#define	OVERLAP	299
#define	ATOMIC	300
#define	FORWARD	301
#define	IF	302
#define	ELSE	303
#define	CONNECT	304
#define	PUBLISHES	305
#define	PYTHON	306
#define	LOCAL	307
#define	NAMESPACE	308
#define	USING	309
#define	IDENT	310
#define	NUMBER	311
#define	LITERAL	312
#define	CPROGRAM	313
#define	HASHIF	314
#define	HASHIFDEF	315
#define	INT	316
#define	LONG	317
#define	SHORT	318
#define	CHAR	319
#define	FLOAT	320
#define	DOUBLE	321
#define	UNSIGNED	322
#define	ACCEL	323
#define	READWRITE	324
#define	WRITEONLY	325
#define	ACCELBLOCK	326
#define	MEMCRITICAL	327


#line 169 "/usr/share/bison++/bison.h"
 /* #defines token */
/* after #define tokens, before const tokens S5*/
#else
 #ifndef YY_parse_CLASS
  #define YY_parse_CLASS parse
 #endif

 #ifndef YY_parse_INHERIT
  #define YY_parse_INHERIT
 #endif

 #ifndef YY_parse_MEMBERS
  #define YY_parse_MEMBERS 
 #endif

 #ifndef YY_parse_LEX_BODY
  #define YY_parse_LEX_BODY  
 #endif

 #ifndef YY_parse_ERROR_BODY
  #define YY_parse_ERROR_BODY  
 #endif

 #ifndef YY_parse_CONSTRUCTOR_PARAM
  #define YY_parse_CONSTRUCTOR_PARAM
 #endif
 /* choose between enum and const */
 #ifndef YY_parse_USE_CONST_TOKEN
  #define YY_parse_USE_CONST_TOKEN 0
  /* yes enum is more compatible with flex,  */
  /* so by default we use it */ 
 #endif
 #if YY_parse_USE_CONST_TOKEN != 0
  #ifndef YY_parse_ENUM_TOKEN
   #define YY_parse_ENUM_TOKEN yy_parse_enum_token
  #endif
 #endif

class YY_parse_CLASS YY_parse_INHERIT
{
public: 
 #if YY_parse_USE_CONST_TOKEN != 0
  /* static const int token ... */
  
 #line 212 "/usr/share/bison++/bison.h"
static const int MODULE;
static const int MAINMODULE;
static const int EXTERN;
static const int READONLY;
static const int INITCALL;
static const int INITNODE;
static const int INITPROC;
static const int PUPABLE;
static const int CHARE;
static const int MAINCHARE;
static const int GROUP;
static const int NODEGROUP;
static const int ARRAY;
static const int MESSAGE;
static const int CONDITIONAL;
static const int CLASS;
static const int INCLUDE;
static const int STACKSIZE;
static const int THREADED;
static const int TEMPLATE;
static const int SYNC;
static const int IGET;
static const int EXCLUSIVE;
static const int IMMEDIATE;
static const int SKIPSCHED;
static const int INLINE;
static const int VIRTUAL;
static const int MIGRATABLE;
static const int CREATEHERE;
static const int CREATEHOME;
static const int NOKEEP;
static const int NOTRACE;
static const int VOID;
static const int CONST;
static const int PACKED;
static const int VARSIZE;
static const int ENTRY;
static const int FOR;
static const int FORALL;
static const int WHILE;
static const int WHEN;
static const int OVERLAP;
static const int ATOMIC;
static const int FORWARD;
static const int IF;
static const int ELSE;
static const int CONNECT;
static const int PUBLISHES;
static const int PYTHON;
static const int LOCAL;
static const int NAMESPACE;
static const int USING;
static const int IDENT;
static const int NUMBER;
static const int LITERAL;
static const int CPROGRAM;
static const int HASHIF;
static const int HASHIFDEF;
static const int INT;
static const int LONG;
static const int SHORT;
static const int CHAR;
static const int FLOAT;
static const int DOUBLE;
static const int UNSIGNED;
static const int ACCEL;
static const int READWRITE;
static const int WRITEONLY;
static const int ACCELBLOCK;
static const int MEMCRITICAL;


#line 212 "/usr/share/bison++/bison.h"
 /* decl const */
 #else
  enum YY_parse_ENUM_TOKEN { YY_parse_NULL_TOKEN=0
  
 #line 215 "/usr/share/bison++/bison.h"
	,MODULE=258
	,MAINMODULE=259
	,EXTERN=260
	,READONLY=261
	,INITCALL=262
	,INITNODE=263
	,INITPROC=264
	,PUPABLE=265
	,CHARE=266
	,MAINCHARE=267
	,GROUP=268
	,NODEGROUP=269
	,ARRAY=270
	,MESSAGE=271
	,CONDITIONAL=272
	,CLASS=273
	,INCLUDE=274
	,STACKSIZE=275
	,THREADED=276
	,TEMPLATE=277
	,SYNC=278
	,IGET=279
	,EXCLUSIVE=280
	,IMMEDIATE=281
	,SKIPSCHED=282
	,INLINE=283
	,VIRTUAL=284
	,MIGRATABLE=285
	,CREATEHERE=286
	,CREATEHOME=287
	,NOKEEP=288
	,NOTRACE=289
	,VOID=290
	,CONST=291
	,PACKED=292
	,VARSIZE=293
	,ENTRY=294
	,FOR=295
	,FORALL=296
	,WHILE=297
	,WHEN=298
	,OVERLAP=299
	,ATOMIC=300
	,FORWARD=301
	,IF=302
	,ELSE=303
	,CONNECT=304
	,PUBLISHES=305
	,PYTHON=306
	,LOCAL=307
	,NAMESPACE=308
	,USING=309
	,IDENT=310
	,NUMBER=311
	,LITERAL=312
	,CPROGRAM=313
	,HASHIF=314
	,HASHIFDEF=315
	,INT=316
	,LONG=317
	,SHORT=318
	,CHAR=319
	,FLOAT=320
	,DOUBLE=321
	,UNSIGNED=322
	,ACCEL=323
	,READWRITE=324
	,WRITEONLY=325
	,ACCELBLOCK=326
	,MEMCRITICAL=327


#line 215 "/usr/share/bison++/bison.h"
 /* enum token */
     }; /* end of enum declaration */
 #endif
public:
 int YY_parse_PARSE(YY_parse_PARSE_PARAM);
 virtual void YY_parse_ERROR(char *msg) YY_parse_ERROR_BODY;
 #ifdef YY_parse_PURE
  #ifdef YY_parse_LSP_NEEDED
   virtual int  YY_parse_LEX(YY_parse_STYPE *YY_parse_LVAL,YY_parse_LTYPE *YY_parse_LLOC) YY_parse_LEX_BODY;
  #else
   virtual int  YY_parse_LEX(YY_parse_STYPE *YY_parse_LVAL) YY_parse_LEX_BODY;
  #endif
 #else
  virtual int YY_parse_LEX() YY_parse_LEX_BODY;
  YY_parse_STYPE YY_parse_LVAL;
  #ifdef YY_parse_LSP_NEEDED
   YY_parse_LTYPE YY_parse_LLOC;
  #endif
  int YY_parse_NERRS;
  int YY_parse_CHAR;
 #endif
 #if YY_parse_DEBUG != 0
  public:
   int YY_parse_DEBUG_FLAG;	/*  nonzero means print parse trace	*/
 #endif
public:
 YY_parse_CLASS(YY_parse_CONSTRUCTOR_PARAM);
public:
 YY_parse_MEMBERS 
};
/* other declare folow */
#endif


#if YY_parse_COMPATIBILITY != 0
 /* backward compatibility */
 /* Removed due to bison problems
 /#ifndef YYSTYPE
 / #define YYSTYPE YY_parse_STYPE
 /#endif*/

 #ifndef YYLTYPE
  #define YYLTYPE YY_parse_LTYPE
 #endif
 #ifndef YYDEBUG
  #ifdef YY_parse_DEBUG 
   #define YYDEBUG YY_parse_DEBUG
  #endif
 #endif

#endif
/* END */

 #line 267 "/usr/share/bison++/bison.h"
#endif
