
# line 2 "xi-grammar.y"
#include <iostream.h>
#include "xi-symbol.h"

extern int yylex (void) ;
int yyerror(char *);
extern int lineno;
ModuleList *modlist;


# line 12 "xi-grammar.y"
typedef union
#ifdef __cplusplus
	YYSTYPE
#endif
 {
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
  char *strval;
  int intval;
} YYSTYPE;
# define MODULE 257
# define MAINMODULE 258
# define EXTERN 259
# define READONLY 260
# define CHARE 261
# define GROUP 262
# define ARRAY 263
# define MESSAGE 264
# define CLASS 265
# define STACKSIZE 266
# define THREADED 267
# define TEMPLATE 268
# define SYNC 269
# define VOID 270
# define PACKED 271
# define VARSIZE 272
# define ENTRY 273
# define MAINCHARE 274
# define IDENT 275
# define NUMBER 276
# define LITERAL 277
# define INT 278
# define LONG 279
# define SHORT 280
# define CHAR 281
# define FLOAT 282
# define DOUBLE 283
# define UNSIGNED 284

#ifdef __STDC__
#include <stdlib.h>
#include <string.h>
#else
#include <malloc.h>
#include <memory.h>
#endif

#include <values.h>

#ifdef __cplusplus

#ifndef yyerror
	int yyerror(char *);
#endif

#ifndef yylex
#ifdef __EXTERN_C__
	extern "C" { int yylex(void); }
#else
	int yylex(void);
#endif
#endif
	int yyparse(void);

#endif
#define yyclearin yychar = -1
#define yyerrok yyerrflag = 0
extern int yychar;
extern int yyerrflag;
YYSTYPE yylval;
YYSTYPE yyval;
typedef int yytabelem;
#ifndef YYMAXDEPTH
#define YYMAXDEPTH 150
#endif
#if YYMAXDEPTH > 0
int yy_yys[YYMAXDEPTH], *yys = yy_yys;
YYSTYPE yy_yyv[YYMAXDEPTH], *yyv = yy_yyv;
#else	/* user does initial allocation */
int *yys;
YYSTYPE *yyv;
#endif
static int yymaxdepth = YYMAXDEPTH;
# define YYERRCODE 256

# line 456 "xi-grammar.y"

int yyerror(char *mesg)
{
  cout << "Syntax error at line " << lineno << ": " << mesg << endl;
  return 0;
}
static const yytabelem yyexca[] ={
-1, 1,
	0, -1,
	-2, 0,
-1, 12,
	125, 13,
	-2, 4,
-1, 15,
	125, 13,
	-2, 4,
-1, 20,
	125, 13,
	-2, 4,
-1, 111,
	44, 91,
	62, 91,
	-2, 60,
-1, 210,
	41, 121,
	-2, 45,
	};
# define YYNPROD 126
# define YYLAST 297
static const yytabelem yyact[]={

    59,    87,    88,    89,    90,     8,   126,   127,    52,    53,
    54,    55,    57,    58,    56,   110,    20,    85,   172,   173,
    59,    86,     8,   115,    79,     8,    29,   219,    52,    53,
    54,    55,    57,    58,    56,    44,     8,    17,   197,   162,
   195,    59,   196,    96,    97,   215,     8,     4,     5,    52,
    53,    54,    55,    57,    58,    56,    59,   193,   214,   112,
    46,     8,    45,    37,    52,    53,    54,    55,    57,    58,
    56,   210,   201,   133,   157,    78,     8,   108,   123,    52,
    53,    54,    55,    57,    58,    56,   189,   146,   178,    43,
    94,     8,    77,    49,    52,    53,    54,    55,    57,    58,
    56,    71,    73,    74,    75,    60,    18,    81,   131,    11,
     7,     9,    51,    79,    72,   118,   203,   206,   176,   149,
   128,   184,   114,   120,    79,    62,    61,   154,   143,   218,
   147,   171,    92,    99,    76,    38,   182,   181,   130,   202,
   180,   113,   168,   217,    63,    64,    65,    66,    42,    80,
     4,     5,   125,    29,    31,    33,    34,    30,   204,    92,
    41,    36,    40,   163,    78,   155,   144,    32,   129,    82,
   111,   121,   132,    12,    93,    78,   117,   103,   104,   105,
   106,    84,    83,   205,   116,   175,    98,   202,   187,    14,
   161,   160,     3,    10,     2,   119,    35,   109,     6,   158,
    28,   159,   107,    13,    69,    19,    68,    23,    22,    21,
    39,    67,   134,   142,    27,   125,   145,    26,   148,    25,
   156,    70,   169,   150,    24,    47,   116,   153,    48,    50,
   209,   186,   170,   179,   177,   151,   174,   185,   111,   135,
   136,   137,   188,   152,   190,   213,   192,   198,    91,   122,
   124,   100,   101,   102,   194,   183,    95,    16,    15,     8,
     1,     0,   212,   211,     0,   192,   198,   216,     0,     0,
     0,     0,   207,   208,     0,     0,   134,   164,   165,   166,
   167,     0,     0,     0,     0,     0,     0,     0,     0,   191,
   138,   139,   140,   141,     0,   199,   200 };
static const yytabelem yypact[]={

  -210,-10000000,-10000000,  -210,  -239,  -239,-10000000,    50,-10000000,    50,
-10000000,-10000000,  -222,-10000000,   -19,  -222,  -107,-10000000,    76,-10000000,
  -222,-10000000,   103,   101,    89,-10000000,-10000000,-10000000,-10000000,  -229,
    34,  -239,  -239,  -239,  -239,  -160,    74,-10000000,-10000000,   -33,
-10000000,-10000000,-10000000,   -16,  -214,   127,   140,   139,-10000000,-10000000,
-10000000,-10000000,-10000000,  -262,-10000000,-10000000,  -277,-10000000,-10000000,-10000000,
    72,  -239,  -228,    75,    75,    75,    75,-10000000,-10000000,-10000000,
-10000000,  -239,  -239,  -239,  -239,    34,  -250,    76,  -253,   134,
    32,   129,-10000000,-10000000,-10000000,-10000000,-10000000,-10000000,-10000000,-10000000,
-10000000,-10000000,  -270,-10000000,    27,   124,-10000000,-10000000,    49,  -239,
    49,    49,    49,    75,    75,    75,    75,  -239,    66,   122,
  -239,    69,   -16,-10000000,    26,-10000000,-10000000,  -239,-10000000,    32,
  -253,  -239,    65,-10000000,   121,    84,-10000000,-10000000,-10000000,  -228,
-10000000,-10000000,  -234,-10000000,   119,-10000000,-10000000,-10000000,    49,    49,
    49,    49,    83,-10000000,  -250,    70,-10000000,  -258,    69,-10000000,
   144,-10000000,    25,-10000000,-10000000,  -270,-10000000,   -37,  -234,    81,
    78,    77,    30,  -239,-10000000,-10000000,-10000000,-10000000,-10000000,-10000000,
-10000000,  -214,-10000000,-10000000,-10000000,   148,-10000000,-10000000,    76,-10000000,
-10000000,-10000000,-10000000,  -184,  -227,-10000000,    84,  -214,-10000000,  -239,
  -239,    99,   127,    23,   114,-10000000,-10000000,   142,    73,   147,
   147,-10000000,  -199,-10000000,  -227,-10000000,  -214,  -221,  -221,   102,
-10000000,-10000000,-10000000,-10000000,-10000000,    68,-10000000,-10000000,  -249,-10000000 };
static const yytabelem yypgo[]={

     0,   194,   260,   192,   193,   189,   258,   105,    87,    58,
   257,    63,   126,    90,   256,   255,    57,   254,   250,    78,
   249,   248,    59,    62,   232,   230,    72,   229,   228,    93,
   112,   225,    60,   191,   190,   224,   221,   219,   217,   214,
   211,   206,   204,   201,   200,    73,   186,    38,   138,    74,
   199,   197,    77,   196,   122,   195,   115 };
static const yytabelem yyr1[]={

     0,     2,     1,     1,    10,    10,    11,    11,     7,     3,
     3,     4,     4,     5,     5,     6,     6,     6,     6,     6,
     6,     6,     6,     6,    18,    18,    18,    19,    19,    20,
    20,    21,    21,    27,    27,    27,    27,    27,    27,    27,
    27,    27,    27,    27,    27,    27,    30,    23,    23,    32,
    31,    31,    54,    54,    28,    29,    22,    22,    22,    22,
    22,    47,    47,    47,    55,    56,    56,    33,    34,    12,
    12,    13,    13,    14,    14,    35,    46,    46,    45,    45,
    37,    37,    38,    39,    40,    40,    41,    42,    36,    24,
    24,     8,     8,     8,    51,    51,    51,    52,    52,    53,
    44,    44,    44,    44,    48,    48,    49,    49,    50,    50,
    50,    43,    43,    43,    15,    15,    16,    16,    17,    17,
    25,    25,    25,    26,     9,     9 };
static const yytabelem yyr2[]={

     0,     3,     1,     5,     1,     3,     1,     3,     3,     7,
     7,     3,     9,     1,     5,    11,     5,     7,     7,     7,
     5,     5,     5,     5,     3,     3,     3,     3,     7,     1,
     3,     1,     7,     3,     3,     3,     3,     5,     5,     5,
     5,     5,     3,     3,     5,     3,     5,     3,     3,     5,
     5,     5,     3,     3,     9,    17,     3,     3,     3,     3,
     3,     1,     3,     7,     7,     1,     5,     9,    11,     1,
     7,     3,     7,     3,     3,     7,     1,     5,     3,     7,
     9,     9,     9,     9,     9,     9,     9,     9,     9,     1,
     5,     1,     5,     5,     7,     5,     7,     3,     7,     9,
     5,     5,     5,     5,     3,     9,     1,     5,     5,     5,
     5,    13,    13,     9,     1,     7,     3,     7,     3,     3,
     1,     3,     3,     7,     1,     7 };
static const yytabelem yychk[]={

-10000000,    -2,    -1,    -3,   257,   258,    -1,    -7,   275,    -7,
    -4,    59,   123,    -4,    -5,    -6,   -10,   259,   125,    -5,
   123,    -3,   -33,   -34,   -35,   -37,   -38,   -39,   -44,   260,
   264,   261,   274,   262,   263,   -53,   268,   -11,    59,    -5,
    59,    59,    59,   -22,   264,   -23,   -32,   -31,   -28,   -29,
   -27,   -30,   278,   279,   280,   281,   284,   282,   283,   270,
    -7,   -12,    91,   -30,   -30,   -30,   -30,   -40,   -41,   -42,
   -36,   261,   274,   262,   263,   264,    60,   125,    91,    40,
    -7,   -23,    42,    42,    42,   279,   283,   278,   279,   280,
   281,   -21,    60,   -30,   -13,   -14,   271,   272,   -46,    58,
   -46,   -46,   -46,    -7,    -7,    -7,    -7,   -12,   -52,   -51,
   265,   -29,   -22,   -11,   -54,   276,    -7,    42,   -56,   -55,
    91,    42,   -20,   -19,   -18,   -22,   276,   277,    93,    44,
   -48,    59,   123,   -45,   -30,   -48,   -48,   -48,   -46,   -46,
   -46,   -46,    -7,    62,    44,    -7,    -8,    61,    -7,    93,
    -7,   -56,   -54,    -7,    62,    44,   -13,   -49,   -50,   -43,
   -33,   -34,   273,    44,   -48,   -48,   -48,   -48,    59,   -52,
   -24,    61,   276,   277,    -8,    41,    93,   -19,   125,   -49,
    59,    59,    59,   -15,    91,   -45,   -22,    40,   -11,   270,
   -32,    -7,   -23,   -16,   -17,   267,   269,   -47,   -22,    -7,
    -7,   -26,    40,    93,    44,    41,    44,   -26,   -26,   -25,
   270,   -32,   -16,   -47,    -9,   266,    -9,    41,    61,   276 };
static const yytabelem yydef[]={

     2,    -2,     1,     2,     0,     0,     3,     0,     8,     0,
     9,    11,    -2,    10,     0,    -2,     0,     5,     6,    14,
    -2,    16,     0,     0,     0,    20,    21,    22,    23,     0,
    69,     0,     0,     0,     0,     0,     0,    12,     7,     0,
    17,    18,    19,     0,     0,    56,    57,    58,    59,    60,
    47,    48,    33,    34,    35,    36,     0,    42,    43,    45,
    31,     0,     0,    76,    76,    76,    76,   100,   101,   102,
   103,     0,     0,     0,     0,    69,     0,     6,     0,     0,
    65,     0,    49,    50,    51,    41,    44,    37,    38,    39,
    40,    46,    29,    75,     0,    71,    73,    74,     0,     0,
     0,     0,     0,    76,    76,    76,    76,     0,     0,    97,
     0,    -2,     0,    15,     0,    52,    53,     0,    67,    65,
     0,     0,     0,    30,    27,    24,    25,    26,    70,     0,
    80,   104,   106,    77,    78,    81,    82,    83,     0,     0,
     0,     0,     0,    99,     0,    89,    95,     0,    91,    54,
     0,    66,     0,    68,    32,     0,    72,     0,   106,     0,
     0,     0,   114,     0,    84,    85,    86,    87,    88,    98,
    94,     0,    92,    93,    96,     0,    64,    28,     6,   107,
   108,   109,   110,     0,     0,    79,    90,    61,   105,    45,
     0,    31,     0,     0,   116,   118,   119,     0,    62,     0,
     0,   113,   120,   115,     0,    55,    61,   124,   124,     0,
    -2,   122,   117,    63,   111,     0,   112,   123,     0,   125 };
typedef struct
#ifdef __cplusplus
	yytoktype
#endif
{ char *t_name; int t_val; } yytoktype;
#ifndef YYDEBUG
#	define YYDEBUG	0	/* don't allow debugging */
#endif

#if YYDEBUG

yytoktype yytoks[] =
{
	"MODULE",	257,
	"MAINMODULE",	258,
	"EXTERN",	259,
	"READONLY",	260,
	"CHARE",	261,
	"GROUP",	262,
	"ARRAY",	263,
	"MESSAGE",	264,
	"CLASS",	265,
	"STACKSIZE",	266,
	"THREADED",	267,
	"TEMPLATE",	268,
	"SYNC",	269,
	"VOID",	270,
	"PACKED",	271,
	"VARSIZE",	272,
	"ENTRY",	273,
	"MAINCHARE",	274,
	"IDENT",	275,
	"NUMBER",	276,
	"LITERAL",	277,
	"INT",	278,
	"LONG",	279,
	"SHORT",	280,
	"CHAR",	281,
	"FLOAT",	282,
	"DOUBLE",	283,
	"UNSIGNED",	284,
	"-unknown-",	-1	/* ends search */
};

char * yyreds[] =
{
	"-no such reduction-",
	"File : ModuleEList",
	"ModuleEList : /* empty */",
	"ModuleEList : Module ModuleEList",
	"OptExtern : /* empty */",
	"OptExtern : EXTERN",
	"OptSemiColon : /* empty */",
	"OptSemiColon : ';'",
	"Name : IDENT",
	"Module : MODULE Name ConstructEList",
	"Module : MAINMODULE Name ConstructEList",
	"ConstructEList : ';'",
	"ConstructEList : '{' ConstructList '}' OptSemiColon",
	"ConstructList : /* empty */",
	"ConstructList : Construct ConstructList",
	"Construct : OptExtern '{' ConstructList '}' OptSemiColon",
	"Construct : OptExtern Module",
	"Construct : OptExtern Readonly ';'",
	"Construct : OptExtern ReadonlyMsg ';'",
	"Construct : OptExtern Message ';'",
	"Construct : OptExtern Chare",
	"Construct : OptExtern Group",
	"Construct : OptExtern Array",
	"Construct : OptExtern Template",
	"TParam : Type",
	"TParam : NUMBER",
	"TParam : LITERAL",
	"TParamList : TParam",
	"TParamList : TParam ',' TParamList",
	"TParamEList : /* empty */",
	"TParamEList : TParamList",
	"OptTParams : /* empty */",
	"OptTParams : '<' TParamEList '>'",
	"BuiltinType : INT",
	"BuiltinType : LONG",
	"BuiltinType : SHORT",
	"BuiltinType : CHAR",
	"BuiltinType : UNSIGNED INT",
	"BuiltinType : UNSIGNED LONG",
	"BuiltinType : UNSIGNED SHORT",
	"BuiltinType : UNSIGNED CHAR",
	"BuiltinType : LONG LONG",
	"BuiltinType : FLOAT",
	"BuiltinType : DOUBLE",
	"BuiltinType : LONG DOUBLE",
	"BuiltinType : VOID",
	"NamedType : Name OptTParams",
	"SimpleType : BuiltinType",
	"SimpleType : NamedType",
	"OnePtrType : SimpleType '*'",
	"PtrType : OnePtrType '*'",
	"PtrType : PtrType '*'",
	"ArrayDim : NUMBER",
	"ArrayDim : Name",
	"ArrayType : Type '[' ArrayDim ']'",
	"FuncType : Type '(' '*' Name ')' '(' TypeList ')'",
	"Type : SimpleType",
	"Type : OnePtrType",
	"Type : PtrType",
	"Type : ArrayType",
	"Type : FuncType",
	"TypeList : /* empty */",
	"TypeList : Type",
	"TypeList : Type ',' TypeList",
	"Dim : '[' ArrayDim ']'",
	"DimList : /* empty */",
	"DimList : Dim DimList",
	"Readonly : READONLY Type Name DimList",
	"ReadonlyMsg : READONLY MESSAGE SimpleType '*' Name",
	"MAttribs : /* empty */",
	"MAttribs : '[' MAttribList ']'",
	"MAttribList : MAttrib",
	"MAttribList : MAttrib ',' MAttribList",
	"MAttrib : PACKED",
	"MAttrib : VARSIZE",
	"Message : MESSAGE MAttribs NamedType",
	"OptBaseList : /* empty */",
	"OptBaseList : ':' BaseList",
	"BaseList : NamedType",
	"BaseList : NamedType ',' BaseList",
	"Chare : CHARE NamedType OptBaseList MemberEList",
	"Chare : MAINCHARE NamedType OptBaseList MemberEList",
	"Group : GROUP NamedType OptBaseList MemberEList",
	"Array : ARRAY NamedType OptBaseList MemberEList",
	"TChare : CHARE Name OptBaseList MemberEList",
	"TChare : MAINCHARE Name OptBaseList MemberEList",
	"TGroup : GROUP Name OptBaseList MemberEList",
	"TArray : ARRAY Name OptBaseList MemberEList",
	"TMessage : MESSAGE MAttribs Name ';'",
	"OptTypeInit : /* empty */",
	"OptTypeInit : '=' Type",
	"OptNameInit : /* empty */",
	"OptNameInit : '=' NUMBER",
	"OptNameInit : '=' LITERAL",
	"TVar : CLASS Name OptTypeInit",
	"TVar : FuncType OptNameInit",
	"TVar : Type Name OptNameInit",
	"TVarList : TVar",
	"TVarList : TVar ',' TVarList",
	"TemplateSpec : TEMPLATE '<' TVarList '>'",
	"Template : TemplateSpec TChare",
	"Template : TemplateSpec TGroup",
	"Template : TemplateSpec TArray",
	"Template : TemplateSpec TMessage",
	"MemberEList : ';'",
	"MemberEList : '{' MemberList '}' OptSemiColon",
	"MemberList : /* empty */",
	"MemberList : Member MemberList",
	"Member : Entry ';'",
	"Member : Readonly ';'",
	"Member : ReadonlyMsg ';'",
	"Entry : ENTRY EAttribs VOID Name EParam OptStackSize",
	"Entry : ENTRY EAttribs OnePtrType Name EParam OptStackSize",
	"Entry : ENTRY EAttribs Name EParam",
	"EAttribs : /* empty */",
	"EAttribs : '[' EAttribList ']'",
	"EAttribList : EAttrib",
	"EAttribList : EAttrib ',' EAttribList",
	"EAttrib : THREADED",
	"EAttrib : SYNC",
	"OptType : /* empty */",
	"OptType : VOID",
	"OptType : OnePtrType",
	"EParam : '(' OptType ')'",
	"OptStackSize : /* empty */",
	"OptStackSize : STACKSIZE '=' NUMBER",
};
#endif /* YYDEBUG */
# line	1 "/usr/ccs/bin/yaccpar"
/*
 * Copyright (c) 1993 by Sun Microsystems, Inc.
 */

#pragma ident	"@(#)yaccpar	6.14	97/01/16 SMI"

/*
** Skeleton parser driver for yacc output
*/

/*
** yacc user known macros and defines
*/
#define YYERROR		goto yyerrlab
#define YYACCEPT	return(0)
#define YYABORT		return(1)
#define YYBACKUP( newtoken, newvalue )\
{\
	if ( yychar >= 0 || ( yyr2[ yytmp ] >> 1 ) != 1 )\
	{\
		yyerror( "syntax error - cannot backup" );\
		goto yyerrlab;\
	}\
	yychar = newtoken;\
	yystate = *yyps;\
	yylval = newvalue;\
	goto yynewstate;\
}
#define YYRECOVERING()	(!!yyerrflag)
#define YYNEW(type)	malloc(sizeof(type) * yynewmax)
#define YYCOPY(to, from, type) \
	(type *) memcpy(to, (char *) from, yymaxdepth * sizeof (type))
#define YYENLARGE( from, type) \
	(type *) realloc((char *) from, yynewmax * sizeof(type))
#ifndef YYDEBUG
#	define YYDEBUG	1	/* make debugging available */
#endif

/*
** user known globals
*/
int yydebug;			/* set to 1 to get debugging */

/*
** driver internal defines
*/
#define YYFLAG		(-10000000)

/*
** global variables used by the parser
*/
YYSTYPE *yypv;			/* top of value stack */
int *yyps;			/* top of state stack */

int yystate;			/* current state */
int yytmp;			/* extra var (lasts between blocks) */

int yynerrs;			/* number of errors */
int yyerrflag;			/* error recovery flag */
int yychar;			/* current input token number */



#ifdef YYNMBCHARS
#define YYLEX()		yycvtok(yylex())
/*
** yycvtok - return a token if i is a wchar_t value that exceeds 255.
**	If i<255, i itself is the token.  If i>255 but the neither 
**	of the 30th or 31st bit is on, i is already a token.
*/
#if defined(__STDC__) || defined(__cplusplus)
int yycvtok(int i)
#else
int yycvtok(i) int i;
#endif
{
	int first = 0;
	int last = YYNMBCHARS - 1;
	int mid;
	wchar_t j;

	if(i&0x60000000){/*Must convert to a token. */
		if( yymbchars[last].character < i ){
			return i;/*Giving up*/
		}
		while ((last>=first)&&(first>=0)) {/*Binary search loop*/
			mid = (first+last)/2;
			j = yymbchars[mid].character;
			if( j==i ){/*Found*/ 
				return yymbchars[mid].tvalue;
			}else if( j<i ){
				first = mid + 1;
			}else{
				last = mid -1;
			}
		}
		/*No entry in the table.*/
		return i;/* Giving up.*/
	}else{/* i is already a token. */
		return i;
	}
}
#else/*!YYNMBCHARS*/
#define YYLEX()		yylex()
#endif/*!YYNMBCHARS*/

/*
** yyparse - return 0 if worked, 1 if syntax error not recovered from
*/
#if defined(__STDC__) || defined(__cplusplus)
int yyparse(void)
#else
int yyparse()
#endif
{
	register YYSTYPE *yypvt = 0;	/* top of value stack for $vars */

#if defined(__cplusplus) || defined(lint)
/*
	hacks to please C++ and lint - goto's inside
	switch should never be executed
*/
	static int __yaccpar_lint_hack__ = 0;
	switch (__yaccpar_lint_hack__)
	{
		case 1: goto yyerrlab;
		case 2: goto yynewstate;
	}
#endif

	/*
	** Initialize externals - yyparse may be called more than once
	*/
	yypv = &yyv[-1];
	yyps = &yys[-1];
	yystate = 0;
	yytmp = 0;
	yynerrs = 0;
	yyerrflag = 0;
	yychar = -1;

#if YYMAXDEPTH <= 0
	if (yymaxdepth <= 0)
	{
		if ((yymaxdepth = YYEXPAND(0)) <= 0)
		{
			yyerror("yacc initialization error");
			YYABORT;
		}
	}
#endif

	{
		register YYSTYPE *yy_pv;	/* top of value stack */
		register int *yy_ps;		/* top of state stack */
		register int yy_state;		/* current state */
		register int  yy_n;		/* internal state number info */
	goto yystack;	/* moved from 6 lines above to here to please C++ */

		/*
		** get globals into registers.
		** branch to here only if YYBACKUP was called.
		*/
	yynewstate:
		yy_pv = yypv;
		yy_ps = yyps;
		yy_state = yystate;
		goto yy_newstate;

		/*
		** get globals into registers.
		** either we just started, or we just finished a reduction
		*/
	yystack:
		yy_pv = yypv;
		yy_ps = yyps;
		yy_state = yystate;

		/*
		** top of for (;;) loop while no reductions done
		*/
	yy_stack:
		/*
		** put a state and value onto the stacks
		*/
#if YYDEBUG
		/*
		** if debugging, look up token value in list of value vs.
		** name pairs.  0 and negative (-1) are special values.
		** Note: linear search is used since time is not a real
		** consideration while debugging.
		*/
		if ( yydebug )
		{
			register int yy_i;

			printf( "State %d, token ", yy_state );
			if ( yychar == 0 )
				printf( "end-of-file\n" );
			else if ( yychar < 0 )
				printf( "-none-\n" );
			else
			{
				for ( yy_i = 0; yytoks[yy_i].t_val >= 0;
					yy_i++ )
				{
					if ( yytoks[yy_i].t_val == yychar )
						break;
				}
				printf( "%s\n", yytoks[yy_i].t_name );
			}
		}
#endif /* YYDEBUG */
		if ( ++yy_ps >= &yys[ yymaxdepth ] )	/* room on stack? */
		{
			/*
			** reallocate and recover.  Note that pointers
			** have to be reset, or bad things will happen
			*/
			int yyps_index = (yy_ps - yys);
			int yypv_index = (yy_pv - yyv);
			int yypvt_index = (yypvt - yyv);
			int yynewmax;
#ifdef YYEXPAND
			yynewmax = YYEXPAND(yymaxdepth);
#else
			yynewmax = 2 * yymaxdepth;	/* double table size */
			if (yymaxdepth == YYMAXDEPTH)	/* first time growth */
			{
				char *newyys = (char *)YYNEW(int);
				char *newyyv = (char *)YYNEW(YYSTYPE);
				if (newyys != 0 && newyyv != 0)
				{
					yys = YYCOPY(newyys, yys, int);
					yyv = YYCOPY(newyyv, yyv, YYSTYPE);
				}
				else
					yynewmax = 0;	/* failed */
			}
			else				/* not first time */
			{
				yys = YYENLARGE(yys, int);
				yyv = YYENLARGE(yyv, YYSTYPE);
				if (yys == 0 || yyv == 0)
					yynewmax = 0;	/* failed */
			}
#endif
			if (yynewmax <= yymaxdepth)	/* tables not expanded */
			{
				yyerror( "yacc stack overflow" );
				YYABORT;
			}
			yymaxdepth = yynewmax;

			yy_ps = yys + yyps_index;
			yy_pv = yyv + yypv_index;
			yypvt = yyv + yypvt_index;
		}
		*yy_ps = yy_state;
		*++yy_pv = yyval;

		/*
		** we have a new state - find out what to do
		*/
	yy_newstate:
		if ( ( yy_n = yypact[ yy_state ] ) <= YYFLAG )
			goto yydefault;		/* simple state */
#if YYDEBUG
		/*
		** if debugging, need to mark whether new token grabbed
		*/
		yytmp = yychar < 0;
#endif
		if ( ( yychar < 0 ) && ( ( yychar = YYLEX() ) < 0 ) )
			yychar = 0;		/* reached EOF */
#if YYDEBUG
		if ( yydebug && yytmp )
		{
			register int yy_i;

			printf( "Received token " );
			if ( yychar == 0 )
				printf( "end-of-file\n" );
			else if ( yychar < 0 )
				printf( "-none-\n" );
			else
			{
				for ( yy_i = 0; yytoks[yy_i].t_val >= 0;
					yy_i++ )
				{
					if ( yytoks[yy_i].t_val == yychar )
						break;
				}
				printf( "%s\n", yytoks[yy_i].t_name );
			}
		}
#endif /* YYDEBUG */
		if ( ( ( yy_n += yychar ) < 0 ) || ( yy_n >= YYLAST ) )
			goto yydefault;
		if ( yychk[ yy_n = yyact[ yy_n ] ] == yychar )	/*valid shift*/
		{
			yychar = -1;
			yyval = yylval;
			yy_state = yy_n;
			if ( yyerrflag > 0 )
				yyerrflag--;
			goto yy_stack;
		}

	yydefault:
		if ( ( yy_n = yydef[ yy_state ] ) == -2 )
		{
#if YYDEBUG
			yytmp = yychar < 0;
#endif
			if ( ( yychar < 0 ) && ( ( yychar = YYLEX() ) < 0 ) )
				yychar = 0;		/* reached EOF */
#if YYDEBUG
			if ( yydebug && yytmp )
			{
				register int yy_i;

				printf( "Received token " );
				if ( yychar == 0 )
					printf( "end-of-file\n" );
				else if ( yychar < 0 )
					printf( "-none-\n" );
				else
				{
					for ( yy_i = 0;
						yytoks[yy_i].t_val >= 0;
						yy_i++ )
					{
						if ( yytoks[yy_i].t_val
							== yychar )
						{
							break;
						}
					}
					printf( "%s\n", yytoks[yy_i].t_name );
				}
			}
#endif /* YYDEBUG */
			/*
			** look through exception table
			*/
			{
				register const int *yyxi = yyexca;

				while ( ( *yyxi != -1 ) ||
					( yyxi[1] != yy_state ) )
				{
					yyxi += 2;
				}
				while ( ( *(yyxi += 2) >= 0 ) &&
					( *yyxi != yychar ) )
					;
				if ( ( yy_n = yyxi[1] ) < 0 )
					YYACCEPT;
			}
		}

		/*
		** check for syntax error
		*/
		if ( yy_n == 0 )	/* have an error */
		{
			/* no worry about speed here! */
			switch ( yyerrflag )
			{
			case 0:		/* new error */
				yyerror( "syntax error" );
				goto skip_init;
			yyerrlab:
				/*
				** get globals into registers.
				** we have a user generated syntax type error
				*/
				yy_pv = yypv;
				yy_ps = yyps;
				yy_state = yystate;
			skip_init:
				yynerrs++;
				/* FALLTHRU */
			case 1:
			case 2:		/* incompletely recovered error */
					/* try again... */
				yyerrflag = 3;
				/*
				** find state where "error" is a legal
				** shift action
				*/
				while ( yy_ps >= yys )
				{
					yy_n = yypact[ *yy_ps ] + YYERRCODE;
					if ( yy_n >= 0 && yy_n < YYLAST &&
						yychk[yyact[yy_n]] == YYERRCODE)					{
						/*
						** simulate shift of "error"
						*/
						yy_state = yyact[ yy_n ];
						goto yy_stack;
					}
					/*
					** current state has no shift on
					** "error", pop stack
					*/
#if YYDEBUG
#	define _POP_ "Error recovery pops state %d, uncovers state %d\n"
					if ( yydebug )
						printf( _POP_, *yy_ps,
							yy_ps[-1] );
#	undef _POP_
#endif
					yy_ps--;
					yy_pv--;
				}
				/*
				** there is no state on stack with "error" as
				** a valid shift.  give up.
				*/
				YYABORT;
			case 3:		/* no shift yet; eat a token */
#if YYDEBUG
				/*
				** if debugging, look up token in list of
				** pairs.  0 and negative shouldn't occur,
				** but since timing doesn't matter when
				** debugging, it doesn't hurt to leave the
				** tests here.
				*/
				if ( yydebug )
				{
					register int yy_i;

					printf( "Error recovery discards " );
					if ( yychar == 0 )
						printf( "token end-of-file\n" );
					else if ( yychar < 0 )
						printf( "token -none-\n" );
					else
					{
						for ( yy_i = 0;
							yytoks[yy_i].t_val >= 0;
							yy_i++ )
						{
							if ( yytoks[yy_i].t_val
								== yychar )
							{
								break;
							}
						}
						printf( "token %s\n",
							yytoks[yy_i].t_name );
					}
				}
#endif /* YYDEBUG */
				if ( yychar == 0 )	/* reached EOF. quit */
					YYABORT;
				yychar = -1;
				goto yy_newstate;
			}
		}/* end if ( yy_n == 0 ) */
		/*
		** reduction by production yy_n
		** put stack tops, etc. so things right after switch
		*/
#if YYDEBUG
		/*
		** if debugging, print the string that is the user's
		** specification of the reduction which is just about
		** to be done.
		*/
		if ( yydebug )
			printf( "Reduce by (%d) \"%s\"\n",
				yy_n, yyreds[ yy_n ] );
#endif
		yytmp = yy_n;			/* value to switch over */
		yypvt = yy_pv;			/* $vars top of value stack */
		/*
		** Look in goto table for next state
		** Sorry about using yy_state here as temporary
		** register variable, but why not, if it works...
		** If yyr2[ yy_n ] doesn't have the low order bit
		** set, then there is no action to be done for
		** this reduction.  So, no saving & unsaving of
		** registers done.  The only difference between the
		** code just after the if and the body of the if is
		** the goto yy_stack in the body.  This way the test
		** can be made before the choice of what to do is needed.
		*/
		{
			/* length of production doubled with extra bit */
			register int yy_len = yyr2[ yy_n ];

			if ( !( yy_len & 01 ) )
			{
				yy_len >>= 1;
				yyval = ( yy_pv -= yy_len )[1];	/* $$ = $1 */
				yy_state = yypgo[ yy_n = yyr1[ yy_n ] ] +
					*( yy_ps -= yy_len ) + 1;
				if ( yy_state >= YYLAST ||
					yychk[ yy_state =
					yyact[ yy_state ] ] != -yy_n )
				{
					yy_state = yyact[ yypgo[ yy_n ] ];
				}
				goto yy_stack;
			}
			yy_len >>= 1;
			yyval = ( yy_pv -= yy_len )[1];	/* $$ = $1 */
			yy_state = yypgo[ yy_n = yyr1[ yy_n ] ] +
				*( yy_ps -= yy_len ) + 1;
			if ( yy_state >= YYLAST ||
				yychk[ yy_state = yyact[ yy_state ] ] != -yy_n )
			{
				yy_state = yyact[ yypgo[ yy_n ] ];
			}
		}
					/* save until reenter driver code */
		yystate = yy_state;
		yyps = yy_ps;
		yypv = yy_pv;
	}
	/*
	** code supplied by user is placed in this switch
	*/
	switch( yytmp )
	{
		
case 1:
# line 91 "xi-grammar.y"
{ yyval.modlist = yypvt[-0].modlist; modlist = yypvt[-0].modlist; } break;
case 2:
# line 95 "xi-grammar.y"
{ yyval.modlist = 0; } break;
case 3:
# line 97 "xi-grammar.y"
{ yyval.modlist = new ModuleList(yypvt[-1].module, yypvt[-0].modlist); } break;
case 4:
# line 101 "xi-grammar.y"
{ yyval.intval = 0; } break;
case 5:
# line 103 "xi-grammar.y"
{ yyval.intval = 1; } break;
case 6:
# line 107 "xi-grammar.y"
{ yyval.intval = 0; } break;
case 7:
# line 109 "xi-grammar.y"
{ yyval.intval = 1; } break;
case 8:
# line 113 "xi-grammar.y"
{ yyval.strval = yypvt[-0].strval; } break;
case 9:
# line 117 "xi-grammar.y"
{ yyval.module = new Module(yypvt[-1].strval, yypvt[-0].conslist); } break;
case 10:
# line 119 "xi-grammar.y"
{ yyval.module = new Module(yypvt[-1].strval, yypvt[-0].conslist); yyval.module->setMain(); } break;
case 11:
# line 123 "xi-grammar.y"
{ yyval.conslist = 0; } break;
case 12:
# line 125 "xi-grammar.y"
{ yyval.conslist = yypvt[-2].conslist; } break;
case 13:
# line 129 "xi-grammar.y"
{ yyval.conslist = 0; } break;
case 14:
# line 131 "xi-grammar.y"
{ yyval.conslist = new ConstructList(yypvt[-1].construct, yypvt[-0].conslist); } break;
case 15:
# line 135 "xi-grammar.y"
{ if(yypvt[-2].conslist) yypvt[-2].conslist->setExtern(yypvt[-4].intval); yyval.construct = yypvt[-2].conslist; } break;
case 16:
# line 137 "xi-grammar.y"
{ yypvt[-0].module->setExtern(yypvt[-1].intval); yyval.construct = yypvt[-0].module; } break;
case 17:
# line 139 "xi-grammar.y"
{ yypvt[-1].readonly->setExtern(yypvt[-2].intval); yyval.construct = yypvt[-1].readonly; } break;
case 18:
# line 141 "xi-grammar.y"
{ yypvt[-1].readonly->setExtern(yypvt[-2].intval); yyval.construct = yypvt[-1].readonly; } break;
case 19:
# line 143 "xi-grammar.y"
{ yypvt[-1].message->setExtern(yypvt[-2].intval); yyval.construct = yypvt[-1].message; } break;
case 20:
# line 145 "xi-grammar.y"
{ yypvt[-0].chare->setExtern(yypvt[-1].intval); yyval.construct = yypvt[-0].chare; } break;
case 21:
# line 147 "xi-grammar.y"
{ yypvt[-0].chare->setExtern(yypvt[-1].intval); yyval.construct = yypvt[-0].chare; } break;
case 22:
# line 149 "xi-grammar.y"
{ yypvt[-0].chare->setExtern(yypvt[-1].intval); yyval.construct = yypvt[-0].chare; } break;
case 23:
# line 151 "xi-grammar.y"
{ yypvt[-0].templat->setExtern(yypvt[-1].intval); yyval.construct = yypvt[-0].templat; } break;
case 24:
# line 155 "xi-grammar.y"
{ yyval.tparam = new TParamType(yypvt[-0].type); } break;
case 25:
# line 157 "xi-grammar.y"
{ yyval.tparam = new TParamVal(yypvt[-0].strval); } break;
case 26:
# line 159 "xi-grammar.y"
{ yyval.tparam = new TParamVal(yypvt[-0].strval); } break;
case 27:
# line 163 "xi-grammar.y"
{ yyval.tparlist = new TParamList(yypvt[-0].tparam); } break;
case 28:
# line 165 "xi-grammar.y"
{ yyval.tparlist = new TParamList(yypvt[-2].tparam, yypvt[-0].tparlist); } break;
case 29:
# line 169 "xi-grammar.y"
{ yyval.tparlist = 0; } break;
case 30:
# line 171 "xi-grammar.y"
{ yyval.tparlist = yypvt[-0].tparlist; } break;
case 31:
# line 175 "xi-grammar.y"
{ yyval.tparlist = 0; } break;
case 32:
# line 177 "xi-grammar.y"
{ yyval.tparlist = yypvt[-1].tparlist; } break;
case 33:
# line 181 "xi-grammar.y"
{ yyval.type = new BuiltinType("int"); } break;
case 34:
# line 183 "xi-grammar.y"
{ yyval.type = new BuiltinType("long"); } break;
case 35:
# line 185 "xi-grammar.y"
{ yyval.type = new BuiltinType("short"); } break;
case 36:
# line 187 "xi-grammar.y"
{ yyval.type = new BuiltinType("char"); } break;
case 37:
# line 189 "xi-grammar.y"
{ yyval.type = new BuiltinType("unsigned int"); } break;
case 38:
# line 191 "xi-grammar.y"
{ yyval.type = new BuiltinType("long"); } break;
case 39:
# line 193 "xi-grammar.y"
{ yyval.type = new BuiltinType("short"); } break;
case 40:
# line 195 "xi-grammar.y"
{ yyval.type = new BuiltinType("char"); } break;
case 41:
# line 197 "xi-grammar.y"
{ yyval.type = new BuiltinType("long long"); } break;
case 42:
# line 199 "xi-grammar.y"
{ yyval.type = new BuiltinType("float"); } break;
case 43:
# line 201 "xi-grammar.y"
{ yyval.type = new BuiltinType("double"); } break;
case 44:
# line 203 "xi-grammar.y"
{ yyval.type = new BuiltinType("long double"); } break;
case 45:
# line 205 "xi-grammar.y"
{ yyval.type = new BuiltinType("void"); } break;
case 46:
# line 209 "xi-grammar.y"
{ yyval.ntype = new NamedType(yypvt[-1].strval, yypvt[-0].tparlist); } break;
case 47:
# line 213 "xi-grammar.y"
{ yyval.type = yypvt[-0].type; } break;
case 48:
# line 215 "xi-grammar.y"
{ yyval.type = yypvt[-0].ntype; } break;
case 49:
# line 219 "xi-grammar.y"
{ yyval.ptype = new PtrType(yypvt[-1].type); } break;
case 50:
# line 223 "xi-grammar.y"
{ yypvt[-1].ptype->indirect(); yyval.ptype = yypvt[-1].ptype; } break;
case 51:
# line 225 "xi-grammar.y"
{ yypvt[-1].ptype->indirect(); yyval.ptype = yypvt[-1].ptype; } break;
case 52:
# line 229 "xi-grammar.y"
{ yyval.val = new Value(yypvt[-0].strval); } break;
case 53:
# line 231 "xi-grammar.y"
{ yyval.val = new Value(yypvt[-0].strval); } break;
case 54:
# line 235 "xi-grammar.y"
{ yyval.type = new ArrayType(yypvt[-3].type, yypvt[-1].val); } break;
case 55:
# line 239 "xi-grammar.y"
{ yyval.ftype = new FuncType(yypvt[-7].type, yypvt[-4].strval, yypvt[-1].typelist); } break;
case 56:
# line 243 "xi-grammar.y"
{ yyval.type = yypvt[-0].type; } break;
case 57:
# line 245 "xi-grammar.y"
{ yyval.type = (Type*) yypvt[-0].ptype; } break;
case 58:
# line 247 "xi-grammar.y"
{ yyval.type = (Type*) yypvt[-0].ptype; } break;
case 59:
# line 249 "xi-grammar.y"
{ yyval.type = yypvt[-0].type; } break;
case 60:
# line 251 "xi-grammar.y"
{ yyval.type = yypvt[-0].ftype; } break;
case 61:
# line 255 "xi-grammar.y"
{ yyval.typelist = 0; } break;
case 62:
# line 257 "xi-grammar.y"
{ yyval.typelist = new TypeList(yypvt[-0].type); } break;
case 63:
# line 259 "xi-grammar.y"
{ yyval.typelist = new TypeList(yypvt[-2].type, yypvt[-0].typelist); } break;
case 64:
# line 263 "xi-grammar.y"
{ yyval.val = yypvt[-1].val; } break;
case 65:
# line 267 "xi-grammar.y"
{ yyval.vallist = 0; } break;
case 66:
# line 269 "xi-grammar.y"
{ yyval.vallist = new ValueList(yypvt[-1].val, yypvt[-0].vallist); } break;
case 67:
# line 273 "xi-grammar.y"
{ yyval.readonly = new Readonly(yypvt[-2].type, yypvt[-1].strval, yypvt[-0].vallist); } break;
case 68:
# line 277 "xi-grammar.y"
{ yyval.readonly = new Readonly(yypvt[-2].type, yypvt[-0].strval, 0, 1); } break;
case 69:
# line 281 "xi-grammar.y"
{ yyval.intval = 0; } break;
case 70:
# line 283 "xi-grammar.y"
{ yyval.intval = yypvt[-1].intval; } break;
case 71:
# line 287 "xi-grammar.y"
{ yyval.intval = yypvt[-0].intval; } break;
case 72:
# line 289 "xi-grammar.y"
{ yyval.intval = yypvt[-2].intval | yypvt[-0].intval; } break;
case 73:
# line 293 "xi-grammar.y"
{ yyval.intval = SPACKED; } break;
case 74:
# line 295 "xi-grammar.y"
{ yyval.intval = SVARSIZE; } break;
case 75:
# line 299 "xi-grammar.y"
{ yyval.message = new Message(yypvt[-0].ntype, yypvt[-1].intval); } break;
case 76:
# line 303 "xi-grammar.y"
{ yyval.typelist = 0; } break;
case 77:
# line 305 "xi-grammar.y"
{ yyval.typelist = yypvt[-0].typelist; } break;
case 78:
# line 309 "xi-grammar.y"
{ yyval.typelist = new TypeList(yypvt[-0].ntype); } break;
case 79:
# line 311 "xi-grammar.y"
{ yyval.typelist = new TypeList(yypvt[-2].ntype, yypvt[-0].typelist); } break;
case 80:
# line 315 "xi-grammar.y"
{ yyval.chare = new Chare(SCHARE, yypvt[-2].ntype, yypvt[-1].typelist, yypvt[-0].mbrlist); if(yypvt[-0].mbrlist) yypvt[-0].mbrlist->setChare(yyval.chare);} break;
case 81:
# line 317 "xi-grammar.y"
{ yyval.chare = new Chare(SMAINCHARE, yypvt[-2].ntype, yypvt[-1].typelist, yypvt[-0].mbrlist); 
                  if(yypvt[-0].mbrlist) yypvt[-0].mbrlist->setChare(yyval.chare);} break;
case 82:
# line 322 "xi-grammar.y"
{ yyval.chare = new Chare(SGROUP, yypvt[-2].ntype, yypvt[-1].typelist, yypvt[-0].mbrlist); if(yypvt[-0].mbrlist) yypvt[-0].mbrlist->setChare(yyval.chare);} break;
case 83:
# line 326 "xi-grammar.y"
{ yyval.chare = new Chare(SARRAY, yypvt[-2].ntype, yypvt[-1].typelist, yypvt[-0].mbrlist); if(yypvt[-0].mbrlist) yypvt[-0].mbrlist->setChare(yyval.chare);} break;
case 84:
# line 330 "xi-grammar.y"
{ yyval.chare = new Chare(SCHARE, new NamedType(yypvt[-2].strval), yypvt[-1].typelist, yypvt[-0].mbrlist); 
                  if(yypvt[-0].mbrlist) yypvt[-0].mbrlist->setChare(yyval.chare);} break;
case 85:
# line 333 "xi-grammar.y"
{ yyval.chare = new Chare(SMAINCHARE, new NamedType(yypvt[-2].strval), yypvt[-1].typelist, yypvt[-0].mbrlist); 
                  if(yypvt[-0].mbrlist) yypvt[-0].mbrlist->setChare(yyval.chare);} break;
case 86:
# line 338 "xi-grammar.y"
{ yyval.chare = new Chare(SGROUP, new NamedType(yypvt[-2].strval), yypvt[-1].typelist, yypvt[-0].mbrlist); 
                  if(yypvt[-0].mbrlist) yypvt[-0].mbrlist->setChare(yyval.chare);} break;
case 87:
# line 343 "xi-grammar.y"
{ yyval.chare = new Chare(SARRAY, new NamedType(yypvt[-2].strval), yypvt[-1].typelist, yypvt[-0].mbrlist); 
                  if(yypvt[-0].mbrlist) yypvt[-0].mbrlist->setChare(yyval.chare);} break;
case 88:
# line 348 "xi-grammar.y"
{ yyval.message = new Message(new NamedType(yypvt[-1].strval), yypvt[-2].intval); } break;
case 89:
# line 352 "xi-grammar.y"
{ yyval.type = 0; } break;
case 90:
# line 354 "xi-grammar.y"
{ yyval.type = yypvt[-0].type; } break;
case 91:
# line 358 "xi-grammar.y"
{ yyval.strval = 0; } break;
case 92:
# line 360 "xi-grammar.y"
{ yyval.strval = yypvt[-0].strval; } break;
case 93:
# line 362 "xi-grammar.y"
{ yyval.strval = yypvt[-0].strval; } break;
case 94:
# line 366 "xi-grammar.y"
{ yyval.tvar = new TType(new NamedType(yypvt[-1].strval), yypvt[-0].type); } break;
case 95:
# line 368 "xi-grammar.y"
{ yyval.tvar = new TFunc(yypvt[-1].ftype, yypvt[-0].strval); } break;
case 96:
# line 370 "xi-grammar.y"
{ yyval.tvar = new TName(yypvt[-2].type, yypvt[-1].strval, yypvt[-0].strval); } break;
case 97:
# line 374 "xi-grammar.y"
{ yyval.tvarlist = new TVarList(yypvt[-0].tvar); } break;
case 98:
# line 376 "xi-grammar.y"
{ yyval.tvarlist = new TVarList(yypvt[-2].tvar, yypvt[-0].tvarlist); } break;
case 99:
# line 380 "xi-grammar.y"
{ yyval.tvarlist = yypvt[-1].tvarlist; } break;
case 100:
# line 384 "xi-grammar.y"
{ yyval.templat = new Template(yypvt[-1].tvarlist, yypvt[-0].chare); yypvt[-0].chare->setTemplate(yyval.templat); } break;
case 101:
# line 386 "xi-grammar.y"
{ yyval.templat = new Template(yypvt[-1].tvarlist, yypvt[-0].chare); yypvt[-0].chare->setTemplate(yyval.templat); } break;
case 102:
# line 388 "xi-grammar.y"
{ yyval.templat = new Template(yypvt[-1].tvarlist, yypvt[-0].chare); yypvt[-0].chare->setTemplate(yyval.templat); } break;
case 103:
# line 390 "xi-grammar.y"
{ yyval.templat = new Template(yypvt[-1].tvarlist, yypvt[-0].message); yypvt[-0].message->setTemplate(yyval.templat); } break;
case 104:
# line 394 "xi-grammar.y"
{ yyval.mbrlist = 0; } break;
case 105:
# line 396 "xi-grammar.y"
{ yyval.mbrlist = yypvt[-2].mbrlist; } break;
case 106:
# line 400 "xi-grammar.y"
{ yyval.mbrlist = 0; } break;
case 107:
# line 402 "xi-grammar.y"
{ yyval.mbrlist = new MemberList(yypvt[-1].member, yypvt[-0].mbrlist); } break;
case 108:
# line 406 "xi-grammar.y"
{ yyval.member = yypvt[-1].entry; } break;
case 109:
# line 408 "xi-grammar.y"
{ yyval.member = yypvt[-1].readonly; } break;
case 110:
# line 410 "xi-grammar.y"
{ yyval.member = yypvt[-1].readonly; } break;
case 111:
# line 414 "xi-grammar.y"
{ yyval.entry = new Entry(yypvt[-4].intval, new BuiltinType("void"), yypvt[-2].strval, yypvt[-1].rtype, yypvt[-0].val); } break;
case 112:
# line 416 "xi-grammar.y"
{ yyval.entry = new Entry(yypvt[-4].intval, yypvt[-3].ptype, yypvt[-2].strval, yypvt[-1].rtype, yypvt[-0].val); } break;
case 113:
# line 418 "xi-grammar.y"
{ yyval.entry = new Entry(yypvt[-2].intval, 0, yypvt[-1].strval, yypvt[-0].rtype, 0); } break;
case 114:
# line 422 "xi-grammar.y"
{ yyval.intval = 0; } break;
case 115:
# line 424 "xi-grammar.y"
{ yyval.intval = yypvt[-1].intval; } break;
case 116:
# line 428 "xi-grammar.y"
{ yyval.intval = yypvt[-0].intval; } break;
case 117:
# line 430 "xi-grammar.y"
{ yyval.intval = yypvt[-2].intval | yypvt[-0].intval; } break;
case 118:
# line 434 "xi-grammar.y"
{ yyval.intval = STHREADED; } break;
case 119:
# line 436 "xi-grammar.y"
{ yyval.intval = SSYNC; } break;
case 120:
# line 440 "xi-grammar.y"
{ yyval.rtype = 0; } break;
case 121:
# line 442 "xi-grammar.y"
{ yyval.rtype = new BuiltinType("void"); } break;
case 122:
# line 444 "xi-grammar.y"
{ yyval.rtype = yypvt[-0].ptype; } break;
case 123:
# line 448 "xi-grammar.y"
{ yyval.rtype = yypvt[-1].rtype; } break;
case 124:
# line 452 "xi-grammar.y"
{ yyval.val = 0; } break;
case 125:
# line 454 "xi-grammar.y"
{ yyval.val = new Value(yypvt[-0].strval); } break;
# line	531 "/usr/ccs/bin/yaccpar"
	}
	goto yystack;		/* reset registers in driver code */
}

