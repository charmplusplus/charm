
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
# define NODEGROUP 263
# define ARRAY 264
# define MESSAGE 265
# define CLASS 266
# define STACKSIZE 267
# define THREADED 268
# define TEMPLATE 269
# define SYNC 270
# define VOID 271
# define PACKED 272
# define VARSIZE 273
# define ENTRY 274
# define MAINCHARE 275
# define IDENT 276
# define NUMBER 277
# define LITERAL 278
# define INT 279
# define LONG 280
# define SHORT 281
# define CHAR 282
# define FLOAT 283
# define DOUBLE 284
# define UNSIGNED 285

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
	void yyerror(const char *);
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

# line 469 "xi-grammar.y"

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
-1, 118,
	44, 94,
	62, 94,
	-2, 61,
-1, 220,
	41, 125,
	-2, 46,
	};
# define YYNPROD 130
# define YYLAST 313
static const yytabelem yyact[]={

    61,    92,    93,    94,    95,     8,   133,   134,    54,    55,
    56,    57,    59,    60,    58,   117,    20,    90,   182,   183,
    61,    91,     8,   122,    84,     8,    30,   229,    54,    55,
    56,    57,    59,    60,    58,    46,     8,   101,   102,   225,
   171,    61,   205,    17,   206,   224,     8,     4,     5,    54,
    55,    56,    57,    59,    60,    58,    61,   207,   203,    47,
   211,     8,    39,    48,    54,    55,    56,    57,    59,    60,
    58,   220,   140,   166,   130,    83,     8,   119,   137,    54,
    55,    56,    57,    59,    60,    58,   199,    62,    53,   155,
    51,     8,     7,     9,    54,    55,    56,    57,    59,    60,
    58,    75,    77,    78,    79,    80,    86,   121,    45,   115,
   125,    99,   188,    82,    18,    76,   138,    11,   213,   186,
   158,    65,    66,    67,    68,    69,    84,   135,    84,   194,
   216,   127,    64,    85,   163,   152,    63,   228,   156,   212,
   181,    97,    81,    40,   192,   120,   191,   214,   190,   178,
     4,     5,    98,    30,    32,    34,    35,    36,    31,    97,
    44,    43,    38,   109,   110,   111,   112,   113,    33,    42,
   104,   123,   118,   172,   164,   132,   153,    83,   136,    83,
   139,    12,    87,   128,   142,   143,   144,   145,   124,    89,
    88,   227,   215,   141,   185,   212,   197,    14,   103,    10,
   126,    37,   151,   170,   169,   154,     3,   157,   116,    13,
     2,   167,   159,    19,     6,   123,   162,   114,    41,    29,
    23,    22,   168,    21,    73,   173,   174,   175,   176,   177,
    72,    71,    70,    28,    27,   161,    26,   160,    25,   187,
    74,   189,   132,    24,   118,   195,    49,   184,   165,    50,
    52,   198,   219,   202,   180,    96,   129,   200,   131,   196,
     8,   141,   204,   179,   226,   105,   106,   107,   108,   193,
   217,   218,   202,   222,   223,   208,   221,   100,    16,    15,
     1,   201,     0,     0,     0,     0,     0,   209,   210,     0,
     0,     0,     0,     0,   208,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,   146,   147,
   148,   149,   150 };
static const yytabelem yypact[]={

  -210,-10000000,-10000000,  -210,  -240,  -240,-10000000,    58,-10000000,    58,
-10000000,-10000000,  -216,-10000000,   -11,  -216,  -107,-10000000,    84,-10000000,
  -216,-10000000,   110,   102,   101,-10000000,-10000000,-10000000,-10000000,-10000000,
  -230,    41,  -240,  -240,  -240,  -240,  -240,  -160,    82,-10000000,
-10000000,   -12,-10000000,-10000000,-10000000,   -16,  -215,   140,   148,   147,
-10000000,-10000000,-10000000,-10000000,-10000000,  -263,-10000000,-10000000,  -278,-10000000,
-10000000,-10000000,    81,  -240,  -235,   112,   112,   112,   112,   112,
-10000000,-10000000,-10000000,-10000000,-10000000,  -240,  -240,  -240,  -240,  -240,
    41,  -251,    84,  -254,   146,    40,   141,-10000000,-10000000,-10000000,
-10000000,-10000000,-10000000,-10000000,-10000000,-10000000,-10000000,  -271,-10000000,    34,
   134,-10000000,-10000000,    57,  -240,    57,    57,    57,    57,   112,
   112,   112,   112,   112,  -240,    73,   132,  -240,    77,   -16,
-10000000,    27,-10000000,-10000000,  -240,-10000000,    40,  -254,  -240,    72,
-10000000,   130,    88,-10000000,-10000000,-10000000,  -235,-10000000,-10000000,  -234,
-10000000,   129,-10000000,-10000000,-10000000,-10000000,    57,    57,    57,    57,
    57,    90,-10000000,  -251,    79,-10000000,  -259,    77,-10000000,   153,
-10000000,    26,-10000000,-10000000,  -271,-10000000,   -13,  -234,    89,    87,
    85,    38,  -240,-10000000,-10000000,-10000000,-10000000,-10000000,-10000000,-10000000,
-10000000,  -215,-10000000,-10000000,-10000000,   156,-10000000,-10000000,    84,-10000000,
-10000000,-10000000,-10000000,  -185,  -226,-10000000,    88,  -215,-10000000,  -240,
  -240,    99,   140,    25,   103,-10000000,-10000000,   151,    86,   155,
   155,-10000000,  -200,-10000000,  -226,-10000000,  -215,  -228,  -228,   150,
-10000000,-10000000,-10000000,-10000000,-10000000,    76,-10000000,-10000000,  -250,-10000000 };
static const yytabelem yypgo[]={

     0,   210,   280,   206,   199,   197,   279,    87,    89,    45,
   278,    62,   136,   111,   277,   269,    58,   262,   258,    74,
   256,   255,    77,    59,   254,   252,    60,   250,   249,    90,
    88,   246,    63,   204,   203,   243,   240,   238,   236,   234,
   233,   232,   231,   230,   224,   222,   219,    72,   198,    57,
    78,    73,   211,   208,   109,   201,   107,   200,   110 };
static const yytabelem yyr1[]={

     0,     2,     1,     1,    10,    10,    11,    11,     7,     3,
     3,     4,     4,     5,     5,     6,     6,     6,     6,     6,
     6,     6,     6,     6,     6,    18,    18,    18,    19,    19,
    20,    20,    21,    21,    27,    27,    27,    27,    27,    27,
    27,    27,    27,    27,    27,    27,    27,    30,    23,    23,
    32,    31,    31,    56,    56,    28,    29,    22,    22,    22,
    22,    22,    49,    49,    49,    57,    58,    58,    33,    34,
    12,    12,    13,    13,    14,    14,    35,    48,    48,    47,
    47,    37,    37,    38,    39,    40,    41,    41,    42,    43,
    44,    36,    24,    24,     8,     8,     8,    53,    53,    53,
    54,    54,    55,    46,    46,    46,    46,    46,    50,    50,
    51,    51,    52,    52,    52,    45,    45,    45,    15,    15,
    16,    16,    17,    17,    25,    25,    25,    26,     9,     9 };
static const yytabelem yyr2[]={

     0,     3,     1,     5,     1,     3,     1,     3,     3,     7,
     7,     3,     9,     1,     5,    11,     5,     7,     7,     7,
     5,     5,     5,     5,     5,     3,     3,     3,     3,     7,
     1,     3,     1,     7,     3,     3,     3,     3,     5,     5,
     5,     5,     5,     3,     3,     5,     3,     5,     3,     3,
     5,     5,     5,     3,     3,     9,    17,     3,     3,     3,
     3,     3,     1,     3,     7,     7,     1,     5,     9,    11,
     1,     7,     3,     7,     3,     3,     7,     1,     5,     3,
     7,     9,     9,     9,     9,     9,     9,     9,     9,     9,
     9,     9,     1,     5,     1,     5,     5,     7,     5,     7,
     3,     7,     9,     5,     5,     5,     5,     5,     3,     9,
     1,     5,     5,     5,     5,    13,    13,     9,     1,     7,
     3,     7,     3,     3,     1,     3,     3,     7,     1,     7 };
static const yytabelem yychk[]={

-10000000,    -2,    -1,    -3,   257,   258,    -1,    -7,   276,    -7,
    -4,    59,   123,    -4,    -5,    -6,   -10,   259,   125,    -5,
   123,    -3,   -33,   -34,   -35,   -37,   -38,   -39,   -40,   -46,
   260,   265,   261,   275,   262,   263,   264,   -55,   269,   -11,
    59,    -5,    59,    59,    59,   -22,   265,   -23,   -32,   -31,
   -28,   -29,   -27,   -30,   279,   280,   281,   282,   285,   283,
   284,   271,    -7,   -12,    91,   -30,   -30,   -30,   -30,   -30,
   -41,   -42,   -43,   -44,   -36,   261,   275,   262,   263,   264,
   265,    60,   125,    91,    40,    -7,   -23,    42,    42,    42,
   280,   284,   279,   280,   281,   282,   -21,    60,   -30,   -13,
   -14,   272,   273,   -48,    58,   -48,   -48,   -48,   -48,    -7,
    -7,    -7,    -7,    -7,   -12,   -54,   -53,   266,   -29,   -22,
   -11,   -56,   277,    -7,    42,   -58,   -57,    91,    42,   -20,
   -19,   -18,   -22,   277,   278,    93,    44,   -50,    59,   123,
   -47,   -30,   -50,   -50,   -50,   -50,   -48,   -48,   -48,   -48,
   -48,    -7,    62,    44,    -7,    -8,    61,    -7,    93,    -7,
   -58,   -56,    -7,    62,    44,   -13,   -51,   -52,   -45,   -33,
   -34,   274,    44,   -50,   -50,   -50,   -50,   -50,    59,   -54,
   -24,    61,   277,   278,    -8,    41,    93,   -19,   125,   -51,
    59,    59,    59,   -15,    91,   -47,   -22,    40,   -11,   271,
   -32,    -7,   -23,   -16,   -17,   268,   270,   -49,   -22,    -7,
    -7,   -26,    40,    93,    44,    41,    44,   -26,   -26,   -25,
   271,   -32,   -16,   -49,    -9,   267,    -9,    41,    61,   277 };
static const yytabelem yydef[]={

     2,    -2,     1,     2,     0,     0,     3,     0,     8,     0,
     9,    11,    -2,    10,     0,    -2,     0,     5,     6,    14,
    -2,    16,     0,     0,     0,    20,    21,    22,    23,    24,
     0,    70,     0,     0,     0,     0,     0,     0,     0,    12,
     7,     0,    17,    18,    19,     0,     0,    57,    58,    59,
    60,    61,    48,    49,    34,    35,    36,    37,     0,    43,
    44,    46,    32,     0,     0,    77,    77,    77,    77,    77,
   103,   104,   105,   106,   107,     0,     0,     0,     0,     0,
    70,     0,     6,     0,     0,    66,     0,    50,    51,    52,
    42,    45,    38,    39,    40,    41,    47,    30,    76,     0,
    72,    74,    75,     0,     0,     0,     0,     0,     0,    77,
    77,    77,    77,    77,     0,     0,   100,     0,    -2,     0,
    15,     0,    53,    54,     0,    68,    66,     0,     0,     0,
    31,    28,    25,    26,    27,    71,     0,    81,   108,   110,
    78,    79,    82,    83,    84,    85,     0,     0,     0,     0,
     0,     0,   102,     0,    92,    98,     0,    94,    55,     0,
    67,     0,    69,    33,     0,    73,     0,   110,     0,     0,
     0,   118,     0,    86,    87,    88,    89,    90,    91,   101,
    97,     0,    95,    96,    99,     0,    65,    29,     6,   111,
   112,   113,   114,     0,     0,    80,    93,    62,   109,    46,
     0,    32,     0,     0,   120,   122,   123,     0,    63,     0,
     0,   117,   124,   119,     0,    56,    62,   128,   128,     0,
    -2,   126,   121,    64,   115,     0,   116,   127,     0,   129 };
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
	"NODEGROUP",	263,
	"ARRAY",	264,
	"MESSAGE",	265,
	"CLASS",	266,
	"STACKSIZE",	267,
	"THREADED",	268,
	"TEMPLATE",	269,
	"SYNC",	270,
	"VOID",	271,
	"PACKED",	272,
	"VARSIZE",	273,
	"ENTRY",	274,
	"MAINCHARE",	275,
	"IDENT",	276,
	"NUMBER",	277,
	"LITERAL",	278,
	"INT",	279,
	"LONG",	280,
	"SHORT",	281,
	"CHAR",	282,
	"FLOAT",	283,
	"DOUBLE",	284,
	"UNSIGNED",	285,
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
	"Construct : OptExtern NodeGroup",
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
	"NodeGroup : NODEGROUP NamedType OptBaseList MemberEList",
	"Array : ARRAY NamedType OptBaseList MemberEList",
	"TChare : CHARE Name OptBaseList MemberEList",
	"TChare : MAINCHARE Name OptBaseList MemberEList",
	"TGroup : GROUP Name OptBaseList MemberEList",
	"TNodeGroup : NODEGROUP Name OptBaseList MemberEList",
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
	"Template : TemplateSpec TNodeGroup",
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
{ yypvt[-0].chare->setExtern(yypvt[-1].intval); yyval.construct = yypvt[-0].chare; } break;
case 24:
# line 153 "xi-grammar.y"
{ yypvt[-0].templat->setExtern(yypvt[-1].intval); yyval.construct = yypvt[-0].templat; } break;
case 25:
# line 157 "xi-grammar.y"
{ yyval.tparam = new TParamType(yypvt[-0].type); } break;
case 26:
# line 159 "xi-grammar.y"
{ yyval.tparam = new TParamVal(yypvt[-0].strval); } break;
case 27:
# line 161 "xi-grammar.y"
{ yyval.tparam = new TParamVal(yypvt[-0].strval); } break;
case 28:
# line 165 "xi-grammar.y"
{ yyval.tparlist = new TParamList(yypvt[-0].tparam); } break;
case 29:
# line 167 "xi-grammar.y"
{ yyval.tparlist = new TParamList(yypvt[-2].tparam, yypvt[-0].tparlist); } break;
case 30:
# line 171 "xi-grammar.y"
{ yyval.tparlist = 0; } break;
case 31:
# line 173 "xi-grammar.y"
{ yyval.tparlist = yypvt[-0].tparlist; } break;
case 32:
# line 177 "xi-grammar.y"
{ yyval.tparlist = 0; } break;
case 33:
# line 179 "xi-grammar.y"
{ yyval.tparlist = yypvt[-1].tparlist; } break;
case 34:
# line 183 "xi-grammar.y"
{ yyval.type = new BuiltinType("int"); } break;
case 35:
# line 185 "xi-grammar.y"
{ yyval.type = new BuiltinType("long"); } break;
case 36:
# line 187 "xi-grammar.y"
{ yyval.type = new BuiltinType("short"); } break;
case 37:
# line 189 "xi-grammar.y"
{ yyval.type = new BuiltinType("char"); } break;
case 38:
# line 191 "xi-grammar.y"
{ yyval.type = new BuiltinType("unsigned int"); } break;
case 39:
# line 193 "xi-grammar.y"
{ yyval.type = new BuiltinType("long"); } break;
case 40:
# line 195 "xi-grammar.y"
{ yyval.type = new BuiltinType("short"); } break;
case 41:
# line 197 "xi-grammar.y"
{ yyval.type = new BuiltinType("char"); } break;
case 42:
# line 199 "xi-grammar.y"
{ yyval.type = new BuiltinType("long long"); } break;
case 43:
# line 201 "xi-grammar.y"
{ yyval.type = new BuiltinType("float"); } break;
case 44:
# line 203 "xi-grammar.y"
{ yyval.type = new BuiltinType("double"); } break;
case 45:
# line 205 "xi-grammar.y"
{ yyval.type = new BuiltinType("long double"); } break;
case 46:
# line 207 "xi-grammar.y"
{ yyval.type = new BuiltinType("void"); } break;
case 47:
# line 211 "xi-grammar.y"
{ yyval.ntype = new NamedType(yypvt[-1].strval, yypvt[-0].tparlist); } break;
case 48:
# line 215 "xi-grammar.y"
{ yyval.type = yypvt[-0].type; } break;
case 49:
# line 217 "xi-grammar.y"
{ yyval.type = yypvt[-0].ntype; } break;
case 50:
# line 221 "xi-grammar.y"
{ yyval.ptype = new PtrType(yypvt[-1].type); } break;
case 51:
# line 225 "xi-grammar.y"
{ yypvt[-1].ptype->indirect(); yyval.ptype = yypvt[-1].ptype; } break;
case 52:
# line 227 "xi-grammar.y"
{ yypvt[-1].ptype->indirect(); yyval.ptype = yypvt[-1].ptype; } break;
case 53:
# line 231 "xi-grammar.y"
{ yyval.val = new Value(yypvt[-0].strval); } break;
case 54:
# line 233 "xi-grammar.y"
{ yyval.val = new Value(yypvt[-0].strval); } break;
case 55:
# line 237 "xi-grammar.y"
{ yyval.type = new ArrayType(yypvt[-3].type, yypvt[-1].val); } break;
case 56:
# line 241 "xi-grammar.y"
{ yyval.ftype = new FuncType(yypvt[-7].type, yypvt[-4].strval, yypvt[-1].typelist); } break;
case 57:
# line 245 "xi-grammar.y"
{ yyval.type = yypvt[-0].type; } break;
case 58:
# line 247 "xi-grammar.y"
{ yyval.type = (Type*) yypvt[-0].ptype; } break;
case 59:
# line 249 "xi-grammar.y"
{ yyval.type = (Type*) yypvt[-0].ptype; } break;
case 60:
# line 251 "xi-grammar.y"
{ yyval.type = yypvt[-0].type; } break;
case 61:
# line 253 "xi-grammar.y"
{ yyval.type = yypvt[-0].ftype; } break;
case 62:
# line 257 "xi-grammar.y"
{ yyval.typelist = 0; } break;
case 63:
# line 259 "xi-grammar.y"
{ yyval.typelist = new TypeList(yypvt[-0].type); } break;
case 64:
# line 261 "xi-grammar.y"
{ yyval.typelist = new TypeList(yypvt[-2].type, yypvt[-0].typelist); } break;
case 65:
# line 265 "xi-grammar.y"
{ yyval.val = yypvt[-1].val; } break;
case 66:
# line 269 "xi-grammar.y"
{ yyval.vallist = 0; } break;
case 67:
# line 271 "xi-grammar.y"
{ yyval.vallist = new ValueList(yypvt[-1].val, yypvt[-0].vallist); } break;
case 68:
# line 275 "xi-grammar.y"
{ yyval.readonly = new Readonly(yypvt[-2].type, yypvt[-1].strval, yypvt[-0].vallist); } break;
case 69:
# line 279 "xi-grammar.y"
{ yyval.readonly = new Readonly(yypvt[-2].type, yypvt[-0].strval, 0, 1); } break;
case 70:
# line 283 "xi-grammar.y"
{ yyval.intval = 0; } break;
case 71:
# line 285 "xi-grammar.y"
{ yyval.intval = yypvt[-1].intval; } break;
case 72:
# line 289 "xi-grammar.y"
{ yyval.intval = yypvt[-0].intval; } break;
case 73:
# line 291 "xi-grammar.y"
{ yyval.intval = yypvt[-2].intval | yypvt[-0].intval; } break;
case 74:
# line 295 "xi-grammar.y"
{ yyval.intval = SPACKED; } break;
case 75:
# line 297 "xi-grammar.y"
{ yyval.intval = SVARSIZE; } break;
case 76:
# line 301 "xi-grammar.y"
{ yyval.message = new Message(yypvt[-0].ntype, yypvt[-1].intval); } break;
case 77:
# line 305 "xi-grammar.y"
{ yyval.typelist = 0; } break;
case 78:
# line 307 "xi-grammar.y"
{ yyval.typelist = yypvt[-0].typelist; } break;
case 79:
# line 311 "xi-grammar.y"
{ yyval.typelist = new TypeList(yypvt[-0].ntype); } break;
case 80:
# line 313 "xi-grammar.y"
{ yyval.typelist = new TypeList(yypvt[-2].ntype, yypvt[-0].typelist); } break;
case 81:
# line 317 "xi-grammar.y"
{ yyval.chare = new Chare(SCHARE, yypvt[-2].ntype, yypvt[-1].typelist, yypvt[-0].mbrlist); if(yypvt[-0].mbrlist) yypvt[-0].mbrlist->setChare(yyval.chare);} break;
case 82:
# line 319 "xi-grammar.y"
{ yyval.chare = new Chare(SMAINCHARE, yypvt[-2].ntype, yypvt[-1].typelist, yypvt[-0].mbrlist); 
                  if(yypvt[-0].mbrlist) yypvt[-0].mbrlist->setChare(yyval.chare);} break;
case 83:
# line 324 "xi-grammar.y"
{ yyval.chare = new Chare(SGROUP, yypvt[-2].ntype, yypvt[-1].typelist, yypvt[-0].mbrlist); if(yypvt[-0].mbrlist) yypvt[-0].mbrlist->setChare(yyval.chare);} break;
case 84:
# line 328 "xi-grammar.y"
{ yyval.chare = new Chare(SNODEGROUP, yypvt[-2].ntype, yypvt[-1].typelist, yypvt[-0].mbrlist); if(yypvt[-0].mbrlist) yypvt[-0].mbrlist->setChare(yyval.chare);} break;
case 85:
# line 332 "xi-grammar.y"
{ yyval.chare = new Chare(SARRAY, yypvt[-2].ntype, yypvt[-1].typelist, yypvt[-0].mbrlist); if(yypvt[-0].mbrlist) yypvt[-0].mbrlist->setChare(yyval.chare);} break;
case 86:
# line 336 "xi-grammar.y"
{ yyval.chare = new Chare(SCHARE, new NamedType(yypvt[-2].strval), yypvt[-1].typelist, yypvt[-0].mbrlist); 
                  if(yypvt[-0].mbrlist) yypvt[-0].mbrlist->setChare(yyval.chare);} break;
case 87:
# line 339 "xi-grammar.y"
{ yyval.chare = new Chare(SMAINCHARE, new NamedType(yypvt[-2].strval), yypvt[-1].typelist, yypvt[-0].mbrlist); 
                  if(yypvt[-0].mbrlist) yypvt[-0].mbrlist->setChare(yyval.chare);} break;
case 88:
# line 344 "xi-grammar.y"
{ yyval.chare = new Chare(SGROUP, new NamedType(yypvt[-2].strval), yypvt[-1].typelist, yypvt[-0].mbrlist); 
                  if(yypvt[-0].mbrlist) yypvt[-0].mbrlist->setChare(yyval.chare);} break;
case 89:
# line 349 "xi-grammar.y"
{ yyval.chare = new Chare(SNODEGROUP, new NamedType(yypvt[-2].strval), yypvt[-1].typelist, yypvt[-0].mbrlist); 
                  if(yypvt[-0].mbrlist) yypvt[-0].mbrlist->setChare(yyval.chare);} break;
case 90:
# line 354 "xi-grammar.y"
{ yyval.chare = new Chare(SARRAY, new NamedType(yypvt[-2].strval), yypvt[-1].typelist, yypvt[-0].mbrlist); 
                  if(yypvt[-0].mbrlist) yypvt[-0].mbrlist->setChare(yyval.chare);} break;
case 91:
# line 359 "xi-grammar.y"
{ yyval.message = new Message(new NamedType(yypvt[-1].strval), yypvt[-2].intval); } break;
case 92:
# line 363 "xi-grammar.y"
{ yyval.type = 0; } break;
case 93:
# line 365 "xi-grammar.y"
{ yyval.type = yypvt[-0].type; } break;
case 94:
# line 369 "xi-grammar.y"
{ yyval.strval = 0; } break;
case 95:
# line 371 "xi-grammar.y"
{ yyval.strval = yypvt[-0].strval; } break;
case 96:
# line 373 "xi-grammar.y"
{ yyval.strval = yypvt[-0].strval; } break;
case 97:
# line 377 "xi-grammar.y"
{ yyval.tvar = new TType(new NamedType(yypvt[-1].strval), yypvt[-0].type); } break;
case 98:
# line 379 "xi-grammar.y"
{ yyval.tvar = new TFunc(yypvt[-1].ftype, yypvt[-0].strval); } break;
case 99:
# line 381 "xi-grammar.y"
{ yyval.tvar = new TName(yypvt[-2].type, yypvt[-1].strval, yypvt[-0].strval); } break;
case 100:
# line 385 "xi-grammar.y"
{ yyval.tvarlist = new TVarList(yypvt[-0].tvar); } break;
case 101:
# line 387 "xi-grammar.y"
{ yyval.tvarlist = new TVarList(yypvt[-2].tvar, yypvt[-0].tvarlist); } break;
case 102:
# line 391 "xi-grammar.y"
{ yyval.tvarlist = yypvt[-1].tvarlist; } break;
case 103:
# line 395 "xi-grammar.y"
{ yyval.templat = new Template(yypvt[-1].tvarlist, yypvt[-0].chare); yypvt[-0].chare->setTemplate(yyval.templat); } break;
case 104:
# line 397 "xi-grammar.y"
{ yyval.templat = new Template(yypvt[-1].tvarlist, yypvt[-0].chare); yypvt[-0].chare->setTemplate(yyval.templat); } break;
case 105:
# line 399 "xi-grammar.y"
{ yyval.templat = new Template(yypvt[-1].tvarlist, yypvt[-0].chare); yypvt[-0].chare->setTemplate(yyval.templat); } break;
case 106:
# line 401 "xi-grammar.y"
{ yyval.templat = new Template(yypvt[-1].tvarlist, yypvt[-0].chare); yypvt[-0].chare->setTemplate(yyval.templat); } break;
case 107:
# line 403 "xi-grammar.y"
{ yyval.templat = new Template(yypvt[-1].tvarlist, yypvt[-0].message); yypvt[-0].message->setTemplate(yyval.templat); } break;
case 108:
# line 407 "xi-grammar.y"
{ yyval.mbrlist = 0; } break;
case 109:
# line 409 "xi-grammar.y"
{ yyval.mbrlist = yypvt[-2].mbrlist; } break;
case 110:
# line 413 "xi-grammar.y"
{ yyval.mbrlist = 0; } break;
case 111:
# line 415 "xi-grammar.y"
{ yyval.mbrlist = new MemberList(yypvt[-1].member, yypvt[-0].mbrlist); } break;
case 112:
# line 419 "xi-grammar.y"
{ yyval.member = yypvt[-1].entry; } break;
case 113:
# line 421 "xi-grammar.y"
{ yyval.member = yypvt[-1].readonly; } break;
case 114:
# line 423 "xi-grammar.y"
{ yyval.member = yypvt[-1].readonly; } break;
case 115:
# line 427 "xi-grammar.y"
{ yyval.entry = new Entry(yypvt[-4].intval, new BuiltinType("void"), yypvt[-2].strval, yypvt[-1].rtype, yypvt[-0].val); } break;
case 116:
# line 429 "xi-grammar.y"
{ yyval.entry = new Entry(yypvt[-4].intval, yypvt[-3].ptype, yypvt[-2].strval, yypvt[-1].rtype, yypvt[-0].val); } break;
case 117:
# line 431 "xi-grammar.y"
{ yyval.entry = new Entry(yypvt[-2].intval, 0, yypvt[-1].strval, yypvt[-0].rtype, 0); } break;
case 118:
# line 435 "xi-grammar.y"
{ yyval.intval = 0; } break;
case 119:
# line 437 "xi-grammar.y"
{ yyval.intval = yypvt[-1].intval; } break;
case 120:
# line 441 "xi-grammar.y"
{ yyval.intval = yypvt[-0].intval; } break;
case 121:
# line 443 "xi-grammar.y"
{ yyval.intval = yypvt[-2].intval | yypvt[-0].intval; } break;
case 122:
# line 447 "xi-grammar.y"
{ yyval.intval = STHREADED; } break;
case 123:
# line 449 "xi-grammar.y"
{ yyval.intval = SSYNC; } break;
case 124:
# line 453 "xi-grammar.y"
{ yyval.rtype = 0; } break;
case 125:
# line 455 "xi-grammar.y"
{ yyval.rtype = new BuiltinType("void"); } break;
case 126:
# line 457 "xi-grammar.y"
{ yyval.rtype = yypvt[-0].ptype; } break;
case 127:
# line 461 "xi-grammar.y"
{ yyval.rtype = yypvt[-1].rtype; } break;
case 128:
# line 465 "xi-grammar.y"
{ yyval.val = 0; } break;
case 129:
# line 467 "xi-grammar.y"
{ yyval.val = new Value(yypvt[-0].strval); } break;
# line	531 "/usr/ccs/bin/yaccpar"
	}
	goto yystack;		/* reset registers in driver code */
}

