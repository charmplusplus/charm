
/*  A Bison parser, made from xi-grammar.y
    by GNU Bison version 1.28  */

#define YYBISON 1  /* Identify Bison output.  */

#define	MODULE	257
#define	MAINMODULE	258
#define	EXTERN	259
#define	READONLY	260
#define	CHARE	261
#define	GROUP	262
#define	NODEGROUP	263
#define	ARRAY	264
#define	MESSAGE	265
#define	CLASS	266
#define	STACKSIZE	267
#define	THREADED	268
#define	MIGRATABLE	269
#define	TEMPLATE	270
#define	SYNC	271
#define	EXCLUSIVE	272
#define	VIRTUAL	273
#define	VOID	274
#define	PACKED	275
#define	VARSIZE	276
#define	VARRAYS	277
#define	ENTRY	278
#define	MAINCHARE	279
#define	IDENT	280
#define	NUMBER	281
#define	LITERAL	282
#define	INT	283
#define	LONG	284
#define	SHORT	285
#define	CHAR	286
#define	FLOAT	287
#define	DOUBLE	288
#define	UNSIGNED	289

#line 1 "xi-grammar.y"

#include <iostream.h>
#include "xi-symbol.h"

extern int yylex (void) ;
void yyerror(const char *);
extern unsigned int lineno;
ModuleList *modlist;


#line 12 "xi-grammar.y"
typedef union {
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
} YYSTYPE;
#include <stdio.h>

#ifndef __cplusplus
#ifndef __STDC__
#define const
#endif
#endif



#define	YYFINAL		278
#define	YYFLAG		-32768
#define	YYNTBASE	49

#define YYTRANSLATE(x) ((unsigned)(x) <= 289 ? yytranslate[x] : 115)

static const char yytranslate[] = {     0,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,    46,
    47,    43,     2,    40,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,    37,    36,    41,
    48,    42,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
    44,     2,    45,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,    38,     2,    39,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     1,     3,     4,     5,     6,
     7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
    17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
    27,    28,    29,    30,    31,    32,    33,    34,    35
};

#if YYDEBUG != 0
static const short yyprhs[] = {     0,
     0,     2,     3,     6,     7,     9,    10,    12,    14,    16,
    21,    25,    29,    31,    36,    37,    40,    46,    49,    53,
    57,    61,    64,    67,    70,    73,    76,    78,    80,    82,
    84,    88,    89,    91,    92,    96,    98,   100,   102,   104,
   107,   110,   113,   116,   119,   121,   123,   126,   128,   131,
   133,   135,   138,   141,   144,   146,   148,   153,   162,   164,
   166,   168,   170,   172,   173,   175,   179,   183,   184,   187,
   192,   198,   199,   203,   205,   209,   211,   213,   215,   216,
   220,   222,   226,   228,   234,   236,   239,   243,   250,   257,
   258,   261,   263,   267,   273,   279,   285,   291,   296,   300,
   306,   312,   318,   324,   330,   336,   341,   342,   345,   346,
   349,   352,   356,   359,   363,   365,   369,   374,   377,   380,
   383,   386,   389,   391,   396,   397,   400,   403,   406,   409,
   417,   425,   430,   431,   435,   437,   441,   443,   445,   447,
   449,   450,   452,   454,   458,   459,   463,   464
};

static const short yyrhs[] = {    50,
     0,     0,    55,    50,     0,     0,     5,     0,     0,    36,
     0,    26,     0,    26,     0,    54,    37,    37,    26,     0,
     3,    53,    56,     0,     4,    53,    56,     0,    36,     0,
    38,    57,    39,    52,     0,     0,    58,    57,     0,    51,
    38,    57,    39,    52,     0,    51,    55,     0,    51,    75,
    36,     0,    51,    76,    36,     0,    51,    85,    36,     0,
    51,    88,     0,    51,    89,     0,    51,    90,     0,    51,
    92,     0,    51,   103,     0,    71,     0,    27,     0,    28,
     0,    59,     0,    59,    40,    60,     0,     0,    60,     0,
     0,    41,    61,    42,     0,    29,     0,    30,     0,    31,
     0,    32,     0,    35,    29,     0,    35,    30,     0,    35,
    31,     0,    35,    32,     0,    30,    30,     0,    33,     0,
    34,     0,    30,    34,     0,    20,     0,    53,    62,     0,
    63,     0,    64,     0,    65,    43,     0,    66,    43,     0,
    67,    43,     0,    27,     0,    53,     0,    71,    44,    68,
    45,     0,    71,    46,    43,    53,    47,    46,    72,    47,
     0,    65,     0,    66,     0,    67,     0,    69,     0,    70,
     0,     0,    71,     0,    71,    40,    72,     0,    44,    68,
    45,     0,     0,    73,    74,     0,     6,    71,    54,    74,
     0,     6,    11,    65,    43,    53,     0,     0,    44,    78,
    45,     0,    79,     0,    79,    40,    78,     0,    21,     0,
    22,     0,    23,     0,     0,    44,    81,    45,     0,    82,
     0,    82,    40,    81,     0,    15,     0,    71,    53,    44,
    45,    36,     0,    83,     0,    83,    84,     0,    11,    77,
    64,     0,    11,    77,    64,    38,    72,    39,     0,    11,
    77,    64,    38,    84,    39,     0,     0,    37,    87,     0,
    64,     0,    64,    40,    87,     0,     7,    80,    64,    86,
   104,     0,    25,    80,    64,    86,   104,     0,     8,    80,
    64,    86,   104,     0,     9,    80,    64,    86,   104,     0,
    44,    27,    53,    45,     0,    44,    53,    45,     0,    10,
    91,    64,    86,   104,     0,     7,    80,    53,    86,   104,
     0,    25,    80,    53,    86,   104,     0,     8,    80,    53,
    86,   104,     0,     9,    80,    53,    86,   104,     0,    10,
    91,    53,    86,   104,     0,    11,    77,    53,    36,     0,
     0,    48,    71,     0,     0,    48,    27,     0,    48,    28,
     0,    12,    53,    98,     0,    70,    99,     0,    71,    53,
    99,     0,   100,     0,   100,    40,   101,     0,    16,    41,
   101,    42,     0,   102,    93,     0,   102,    94,     0,   102,
    95,     0,   102,    96,     0,   102,    97,     0,    36,     0,
    38,   105,    39,    52,     0,     0,   106,   105,     0,   107,
    36,     0,    75,    36,     0,    76,    36,     0,    24,   108,
    20,    53,   112,   114,   113,     0,    24,   108,    66,    53,
   112,   114,   113,     0,    24,   108,    53,   112,     0,     0,
    44,   109,    45,     0,   110,     0,   110,    40,   109,     0,
    14,     0,    17,     0,    18,     0,    19,     0,     0,    20,
     0,    66,     0,    46,   111,    47,     0,     0,    13,    48,
    27,     0,     0,    48,    27,     0
};

#endif

#if YYDEBUG != 0
static const short yyrline[] = { 0,
    96,   100,   102,   106,   108,   112,   114,   118,   122,   124,
   132,   134,   138,   140,   144,   146,   150,   152,   154,   156,
   158,   160,   162,   164,   166,   168,   172,   174,   176,   180,
   182,   186,   188,   192,   194,   198,   200,   202,   204,   206,
   208,   210,   212,   214,   216,   218,   220,   222,   226,   230,
   232,   236,   240,   242,   246,   248,   252,   256,   260,   262,
   264,   266,   268,   272,   274,   276,   280,   284,   286,   290,
   294,   298,   300,   304,   306,   310,   312,   314,   318,   320,
   324,   326,   330,   334,   338,   340,   344,   346,   348,   352,
   354,   358,   360,   364,   366,   370,   374,   378,   384,   388,
   392,   394,   398,   402,   406,   410,   414,   416,   420,   422,
   424,   428,   430,   432,   436,   438,   442,   446,   448,   450,
   452,   454,   458,   460,   464,   466,   470,   472,   474,   478,
   480,   482,   486,   488,   492,   494,   498,   500,   502,   504,
   508,   510,   512,   516,   520,   522,   526,   528
};
#endif


#if YYDEBUG != 0 || defined (YYERROR_VERBOSE)

static const char * const yytname[] = {   "$","error","$undefined.","MODULE",
"MAINMODULE","EXTERN","READONLY","CHARE","GROUP","NODEGROUP","ARRAY","MESSAGE",
"CLASS","STACKSIZE","THREADED","MIGRATABLE","TEMPLATE","SYNC","EXCLUSIVE","VIRTUAL",
"VOID","PACKED","VARSIZE","VARRAYS","ENTRY","MAINCHARE","IDENT","NUMBER","LITERAL",
"INT","LONG","SHORT","CHAR","FLOAT","DOUBLE","UNSIGNED","';'","':'","'{'","'}'",
"','","'<'","'>'","'*'","'['","']'","'('","')'","'='","File","ModuleEList","OptExtern",
"OptSemiColon","Name","QualName","Module","ConstructEList","ConstructList","Construct",
"TParam","TParamList","TParamEList","OptTParams","BuiltinType","NamedType","SimpleType",
"OnePtrType","PtrType","ArrayDim","ArrayType","FuncType","Type","TypeList","Dim",
"DimList","Readonly","ReadonlyMsg","MAttribs","MAttribList","MAttrib","CAttribs",
"CAttribList","CAttrib","Var","VarList","Message","OptBaseList","BaseList","Chare",
"Group","NodeGroup","ArrayIndexType","Array","TChare","TGroup","TNodeGroup",
"TArray","TMessage","OptTypeInit","OptNameInit","TVar","TVarList","TemplateSpec",
"Template","MemberEList","MemberList","Member","Entry","EAttribs","EAttribList",
"EAttrib","OptType","EParam","OptStackSize","OptPure", NULL
};
#endif

static const short yyr1[] = {     0,
    49,    50,    50,    51,    51,    52,    52,    53,    54,    54,
    55,    55,    56,    56,    57,    57,    58,    58,    58,    58,
    58,    58,    58,    58,    58,    58,    59,    59,    59,    60,
    60,    61,    61,    62,    62,    63,    63,    63,    63,    63,
    63,    63,    63,    63,    63,    63,    63,    63,    64,    65,
    65,    66,    67,    67,    68,    68,    69,    70,    71,    71,
    71,    71,    71,    72,    72,    72,    73,    74,    74,    75,
    76,    77,    77,    78,    78,    79,    79,    79,    80,    80,
    81,    81,    82,    83,    84,    84,    85,    85,    85,    86,
    86,    87,    87,    88,    88,    89,    90,    91,    91,    92,
    93,    93,    94,    95,    96,    97,    98,    98,    99,    99,
    99,   100,   100,   100,   101,   101,   102,   103,   103,   103,
   103,   103,   104,   104,   105,   105,   106,   106,   106,   107,
   107,   107,   108,   108,   109,   109,   110,   110,   110,   110,
   111,   111,   111,   112,   113,   113,   114,   114
};

static const short yyr2[] = {     0,
     1,     0,     2,     0,     1,     0,     1,     1,     1,     4,
     3,     3,     1,     4,     0,     2,     5,     2,     3,     3,
     3,     2,     2,     2,     2,     2,     1,     1,     1,     1,
     3,     0,     1,     0,     3,     1,     1,     1,     1,     2,
     2,     2,     2,     2,     1,     1,     2,     1,     2,     1,
     1,     2,     2,     2,     1,     1,     4,     8,     1,     1,
     1,     1,     1,     0,     1,     3,     3,     0,     2,     4,
     5,     0,     3,     1,     3,     1,     1,     1,     0,     3,
     1,     3,     1,     5,     1,     2,     3,     6,     6,     0,
     2,     1,     3,     5,     5,     5,     5,     4,     3,     5,
     5,     5,     5,     5,     5,     4,     0,     2,     0,     2,
     2,     3,     2,     3,     1,     3,     4,     2,     2,     2,
     2,     2,     1,     4,     0,     2,     2,     2,     2,     7,
     7,     4,     0,     3,     1,     3,     1,     1,     1,     1,
     0,     1,     1,     3,     0,     3,     0,     2
};

static const short yydefact[] = {     2,
     0,     0,     1,     2,     8,     0,     0,     3,    13,     4,
    11,    12,     5,     0,     0,     4,     0,    79,    79,    79,
     0,    72,     0,    79,     4,    18,     0,     0,     0,    22,
    23,    24,    25,     0,    26,     6,    16,     0,    48,    36,
    37,    38,    39,    45,    46,     0,    34,    50,    51,    59,
    60,    61,    62,    63,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,    19,    20,    21,    79,
    79,    79,     0,    72,    79,   118,   119,   120,   121,   122,
     7,    14,     0,    44,    47,    40,    41,    42,    43,    32,
    49,    52,    53,    54,     9,     0,     0,    68,    83,     0,
    81,    90,    90,    90,     0,     0,    90,    76,    77,    78,
     0,    74,    87,     0,    63,     0,   115,     0,    90,     6,
     0,     0,     0,     0,     0,     0,     0,    28,    29,    30,
    33,     0,    27,    55,    56,     0,     0,     0,     0,    68,
    70,    80,     0,     0,     0,     0,     0,     0,    99,     0,
    73,     0,    64,   107,     0,   113,   109,     0,   117,     0,
    17,    90,    90,    90,    90,     0,    90,    71,     0,    35,
    57,     0,     0,     0,    69,    82,    92,    91,   123,   125,
    94,    96,    97,    98,   100,    75,    65,     0,    85,     0,
     0,   112,   110,   111,   114,   116,    95,     0,     0,     0,
     0,   106,     0,    31,     0,    10,    67,     0,   133,     0,
     0,     0,   125,     0,    64,     0,    88,     0,    86,    89,
   108,   101,   103,   104,   105,   102,    64,    93,     0,     0,
   128,   129,     6,   126,   127,    65,    66,     0,     0,   137,
   138,   139,   140,     0,   135,    48,    34,     0,     0,   124,
     0,    58,   134,     0,     0,   141,   132,     0,    84,   136,
   147,    48,   143,     0,   147,     0,   145,   144,   145,   148,
     0,   130,   131,     0,   146,     0,     0,     0
};

static const short yydefgoto[] = {   276,
     3,    14,    82,    47,    98,     4,    11,    15,    16,   130,
   131,   132,    91,    48,    49,    50,    51,    52,   136,    53,
    54,   116,   188,   140,   141,   210,   211,    63,   111,   112,
    57,   100,   101,   189,   190,    29,   145,   178,    30,    31,
    32,    61,    33,    76,    77,    78,    79,    80,   192,   156,
   117,   118,    34,    35,   181,   212,   213,   214,   230,   244,
   245,   264,   257,   272,   267
};

static const short yypact[] = {    25,
   -11,   -11,-32768,    25,-32768,    78,    78,-32768,-32768,     9,
-32768,-32768,-32768,    36,    59,     9,   117,    10,    10,    10,
    61,    83,    93,    10,     9,-32768,   105,   109,   123,-32768,
-32768,-32768,-32768,    16,-32768,   140,-32768,   158,-32768,-32768,
    60,-32768,-32768,-32768,-32768,   100,   112,-32768,-32768,   136,
   138,   139,-32768,-32768,     5,   165,   -11,   -11,   -11,    92,
   -11,    43,   -11,    77,   -11,   144,-32768,-32768,-32768,    10,
    10,    10,    61,    83,    10,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,   142,-32768,-32768,-32768,-32768,-32768,-32768,   137,
-32768,-32768,-32768,-32768,-32768,   135,   152,    23,-32768,   153,
   157,   162,   162,   162,   -11,   155,   162,-32768,-32768,-32768,
   159,   163,   177,   -11,    15,    12,   174,   186,   162,   140,
   -11,   -11,   -11,   -11,   -11,   -11,   -11,-32768,-32768,   179,
-32768,   188,    89,-32768,-32768,   171,   -11,   194,   135,   189,
-32768,-32768,   165,   -11,   106,   106,   106,   187,-32768,   106,
-32768,    43,   158,   190,   146,-32768,   191,    77,-32768,   106,
-32768,   162,   162,   162,   162,   198,   162,-32768,   137,-32768,
-32768,   193,   209,   196,-32768,-32768,   197,-32768,-32768,    29,
-32768,-32768,-32768,-32768,-32768,-32768,   114,   203,   158,   204,
   158,-32768,-32768,-32768,-32768,-32768,-32768,   106,   106,   106,
   106,-32768,   106,-32768,   200,-32768,-32768,   -11,   205,   208,
   211,   212,    29,   214,   158,   210,-32768,    12,-32768,-32768,
    89,-32768,-32768,-32768,-32768,-32768,   158,-32768,    82,   176,
-32768,-32768,   140,-32768,-32768,    47,-32768,   207,   206,-32768,
-32768,-32768,-32768,   213,   215,   -11,    40,   136,   -11,-32768,
   220,-32768,-32768,    82,   216,   192,-32768,   216,-32768,-32768,
   217,   219,-32768,   221,   217,   230,   223,-32768,   223,-32768,
   222,-32768,-32768,   232,-32768,   260,   261,-32768
};

static const short yypgoto[] = {-32768,
   259,-32768,  -116,    -1,-32768,   250,   262,    -4,-32768,-32768,
    98,-32768,-32768,-32768,   -52,   -36,  -220,-32768,   132,-32768,
   -56,   -14,  -165,-32768,   133,   258,   263,   201,   122,-32768,
    13,   141,-32768,-32768,    87,-32768,   -85,    70,-32768,-32768,
-32768,   218,-32768,-32768,-32768,-32768,-32768,-32768,-32768,   124,
-32768,   121,-32768,-32768,  -130,    67,-32768,-32768,-32768,    28,
-32768,-32768,  -183,    14,    20
};


#define	YYLAST		291


static const short yytable[] = {     6,
     7,    83,    55,   161,   102,   103,   104,   115,   107,   249,
   113,    37,   119,    13,     5,   182,   183,   146,   147,   185,
    66,   150,    70,    71,    72,    73,    74,     1,     2,   197,
    95,    58,    59,   160,    17,   263,    65,     5,     1,     2,
    75,    17,    18,    19,    20,    21,    22,   -15,    96,   237,
    97,    23,   209,    56,  -109,    96,  -109,    97,   106,   138,
    24,   239,   155,   108,   109,   110,   139,   222,   223,   224,
   225,   261,   226,    25,   265,   133,   198,   199,   200,   201,
    90,   203,   121,   122,   123,   256,   215,   126,   114,    84,
    96,   177,    97,    85,   135,   240,    39,    36,   241,   242,
   243,   115,     5,   148,    60,    40,    41,    42,    43,    44,
    45,    46,   154,     9,   157,    10,   250,     5,   105,   162,
   163,   164,   165,   166,   167,   168,    62,    38,    86,    87,
    88,    89,    96,    64,    97,   172,    39,   135,   187,     5,
    67,   179,     5,   180,    68,    40,    41,    42,    43,    44,
    45,    46,    90,   215,   133,   177,    39,    96,    69,    97,
     5,   134,     5,   128,   129,    40,    41,    42,    43,    44,
    45,    46,   193,   194,   218,    81,   221,    39,    92,    99,
    93,    94,   120,     5,   127,   216,    40,    41,    42,    43,
    44,    45,    46,   248,   137,   246,   143,   142,   144,   149,
   236,     5,   152,   151,    40,    41,    42,    43,    44,    45,
    46,   262,   236,   158,   153,   171,   216,     5,   169,   248,
    40,    41,    42,    43,    44,    45,    46,   159,   247,   170,
   173,   184,   139,   202,   206,   271,   208,   191,   155,   205,
   207,   217,   220,   231,   255,   227,   232,   258,   229,   235,
   233,   251,   252,   238,   254,   259,   270,   253,   275,   277,
   278,   256,     8,    26,   266,  -142,   204,   268,    12,   274,
   174,    27,   175,   186,   125,   219,    28,   228,   196,   234,
   195,   260,   273,   176,   269,     0,     0,     0,     0,     0,
   124
};

static const short yycheck[] = {     1,
     2,    38,    17,   120,    57,    58,    59,    64,    61,   230,
    63,    16,    65,     5,    26,   146,   147,   103,   104,   150,
    25,   107,     7,     8,     9,    10,    11,     3,     4,   160,
    26,    19,    20,   119,     6,   256,    24,    26,     3,     4,
    25,     6,     7,     8,     9,    10,    11,    39,    44,   215,
    46,    16,    24,    44,    40,    44,    42,    46,    60,    37,
    25,   227,    48,    21,    22,    23,    44,   198,   199,   200,
   201,   255,   203,    38,   258,    90,   162,   163,   164,   165,
    41,   167,    70,    71,    72,    46,    40,    75,    12,    30,
    44,   144,    46,    34,    96,    14,    20,    39,    17,    18,
    19,   158,    26,   105,    44,    29,    30,    31,    32,    33,
    34,    35,   114,    36,   116,    38,   233,    26,    27,   121,
   122,   123,   124,   125,   126,   127,    44,    11,    29,    30,
    31,    32,    44,    41,    46,   137,    20,   139,   153,    26,
    36,    36,    26,    38,    36,    29,    30,    31,    32,    33,
    34,    35,    41,    40,   169,   208,    20,    44,    36,    46,
    26,    27,    26,    27,    28,    29,    30,    31,    32,    33,
    34,    35,    27,    28,   189,    36,   191,    20,    43,    15,
    43,    43,    39,    26,    43,   187,    29,    30,    31,    32,
    33,    34,    35,   230,    43,    20,    40,    45,    37,    45,
   215,    26,    40,    45,    29,    30,    31,    32,    33,    34,
    35,    20,   227,    40,    38,    45,   218,    26,    40,   256,
    29,    30,    31,    32,    33,    34,    35,    42,   230,    42,
    37,    45,    44,    36,    26,    13,    40,    48,    48,    47,
    45,    39,    39,    36,   246,    46,    36,   249,    44,    36,
    39,    45,    47,    44,    40,    36,    27,    45,    27,     0,
     0,    46,     4,    14,    48,    47,   169,    47,     7,    48,
   139,    14,   140,   152,    74,   189,    14,   208,   158,   213,
   157,   254,   269,   143,   265,    -1,    -1,    -1,    -1,    -1,
    73
};
/* -*-C-*-  Note some compilers choke on comments on `#line' lines.  */
#line 3 "/usr/lib/bison.simple"
/* This file comes from bison-1.28.  */

/* Skeleton output parser for bison,
   Copyright (C) 1984, 1989, 1990 Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place - Suite 330,
   Boston, MA 02111-1307, USA.  */

/* As a special exception, when this file is copied by Bison into a
   Bison output file, you may use that output file without restriction.
   This special exception was added by the Free Software Foundation
   in version 1.24 of Bison.  */

/* This is the parser code that is written into each bison parser
  when the %semantic_parser declaration is not specified in the grammar.
  It was written by Richard Stallman by simplifying the hairy parser
  used when %semantic_parser is specified.  */

#ifndef YYSTACK_USE_ALLOCA
#ifdef alloca
#define YYSTACK_USE_ALLOCA
#else /* alloca not defined */
#ifdef __GNUC__
#define YYSTACK_USE_ALLOCA
#define alloca __builtin_alloca
#else /* not GNU C.  */
#if (!defined (__STDC__) && defined (sparc)) || defined (__sparc__) || defined (__sparc) || defined (__sgi) || (defined (__sun) && defined (__i386))
#define YYSTACK_USE_ALLOCA
#include <alloca.h>
#else /* not sparc */
/* We think this test detects Watcom and Microsoft C.  */
/* This used to test MSDOS, but that is a bad idea
   since that symbol is in the user namespace.  */
#if (defined (_MSDOS) || defined (_MSDOS_)) && !defined (__TURBOC__)
#if 0 /* No need for malloc.h, which pollutes the namespace;
	 instead, just don't use alloca.  */
#include <malloc.h>
#endif
#else /* not MSDOS, or __TURBOC__ */
#if defined(_AIX)
/* I don't know what this was needed for, but it pollutes the namespace.
   So I turned it off.   rms, 2 May 1997.  */
/* #include <malloc.h>  */
 #pragma alloca
#define YYSTACK_USE_ALLOCA
#else /* not MSDOS, or __TURBOC__, or _AIX */
#if 0
#ifdef __hpux /* haible@ilog.fr says this works for HPUX 9.05 and up,
		 and on HPUX 10.  Eventually we can turn this on.  */
#define YYSTACK_USE_ALLOCA
#define alloca __builtin_alloca
#endif /* __hpux */
#endif
#endif /* not _AIX */
#endif /* not MSDOS, or __TURBOC__ */
#endif /* not sparc */
#endif /* not GNU C */
#endif /* alloca not defined */
#endif /* YYSTACK_USE_ALLOCA not defined */

#ifdef YYSTACK_USE_ALLOCA
#define YYSTACK_ALLOC alloca
#else
#define YYSTACK_ALLOC malloc
#endif

/* Note: there must be only one dollar sign in this file.
   It is replaced by the list of actions, each action
   as one case of the switch.  */

#define yyerrok		(yyerrstatus = 0)
#define yyclearin	(yychar = YYEMPTY)
#define YYEMPTY		-2
#define YYEOF		0
#define YYACCEPT	goto yyacceptlab
#define YYABORT 	goto yyabortlab
#define YYERROR		goto yyerrlab1
/* Like YYERROR except do call yyerror.
   This remains here temporarily to ease the
   transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  */
#define YYFAIL		goto yyerrlab
#define YYRECOVERING()  (!!yyerrstatus)
#define YYBACKUP(token, value) \
do								\
  if (yychar == YYEMPTY && yylen == 1)				\
    { yychar = (token), yylval = (value);			\
      yychar1 = YYTRANSLATE (yychar);				\
      YYPOPSTACK;						\
      goto yybackup;						\
    }								\
  else								\
    { yyerror ("syntax error: cannot back up"); YYERROR; }	\
while (0)

#define YYTERROR	1
#define YYERRCODE	256

#ifndef YYPURE
#define YYLEX		yylex()
#endif

#ifdef YYPURE
#ifdef YYLSP_NEEDED
#ifdef YYLEX_PARAM
#define YYLEX		yylex(&yylval, &yylloc, YYLEX_PARAM)
#else
#define YYLEX		yylex(&yylval, &yylloc)
#endif
#else /* not YYLSP_NEEDED */
#ifdef YYLEX_PARAM
#define YYLEX		yylex(&yylval, YYLEX_PARAM)
#else
#define YYLEX		yylex(&yylval)
#endif
#endif /* not YYLSP_NEEDED */
#endif

/* If nonreentrant, generate the variables here */

#ifndef YYPURE

int	yychar;			/*  the lookahead symbol		*/
YYSTYPE	yylval;			/*  the semantic value of the		*/
				/*  lookahead symbol			*/

#ifdef YYLSP_NEEDED
YYLTYPE yylloc;			/*  location data for the lookahead	*/
				/*  symbol				*/
#endif

int yynerrs;			/*  number of parse errors so far       */
#endif  /* not YYPURE */

#if YYDEBUG != 0
int yydebug;			/*  nonzero means print parse trace	*/
/* Since this is uninitialized, it does not stop multiple parsers
   from coexisting.  */
#endif

/*  YYINITDEPTH indicates the initial size of the parser's stacks	*/

#ifndef	YYINITDEPTH
#define YYINITDEPTH 200
#endif

/*  YYMAXDEPTH is the maximum size the stacks can grow to
    (effective only if the built-in stack extension method is used).  */

#if YYMAXDEPTH == 0
#undef YYMAXDEPTH
#endif

#ifndef YYMAXDEPTH
#define YYMAXDEPTH 10000
#endif

/* Define __yy_memcpy.  Note that the size argument
   should be passed with type unsigned int, because that is what the non-GCC
   definitions require.  With GCC, __builtin_memcpy takes an arg
   of type size_t, but it can handle unsigned int.  */

#if __GNUC__ > 1		/* GNU C and GNU C++ define this.  */
#define __yy_memcpy(TO,FROM,COUNT)	__builtin_memcpy(TO,FROM,COUNT)
#else				/* not GNU C or C++ */
#ifndef __cplusplus

/* This is the most reliable way to avoid incompatibilities
   in available built-in functions on various systems.  */
static void
__yy_memcpy (to, from, count)
     char *to;
     char *from;
     unsigned int count;
{
  register char *f = from;
  register char *t = to;
  register int i = count;

  while (i-- > 0)
    *t++ = *f++;
}

#else /* __cplusplus */

/* This is the most reliable way to avoid incompatibilities
   in available built-in functions on various systems.  */
static void
__yy_memcpy (char *to, char *from, unsigned int count)
{
  register char *t = to;
  register char *f = from;
  register int i = count;

  while (i-- > 0)
    *t++ = *f++;
}

#endif
#endif

#line 217 "/usr/lib/bison.simple"

/* The user can define YYPARSE_PARAM as the name of an argument to be passed
   into yyparse.  The argument should have type void *.
   It should actually point to an object.
   Grammar actions can access the variable by casting it
   to the proper pointer type.  */

#ifdef YYPARSE_PARAM
#ifdef __cplusplus
#define YYPARSE_PARAM_ARG void *YYPARSE_PARAM
#define YYPARSE_PARAM_DECL
#else /* not __cplusplus */
#define YYPARSE_PARAM_ARG YYPARSE_PARAM
#define YYPARSE_PARAM_DECL void *YYPARSE_PARAM;
#endif /* not __cplusplus */
#else /* not YYPARSE_PARAM */
#define YYPARSE_PARAM_ARG
#define YYPARSE_PARAM_DECL
#endif /* not YYPARSE_PARAM */

/* Prevent warning if -Wstrict-prototypes.  */
#ifdef __GNUC__
#ifdef YYPARSE_PARAM
int yyparse (void *);
#else
int yyparse (void);
#endif
#endif

int
yyparse(YYPARSE_PARAM_ARG)
     YYPARSE_PARAM_DECL
{
  register int yystate;
  register int yyn;
  register short *yyssp;
  register YYSTYPE *yyvsp;
  int yyerrstatus;	/*  number of tokens to shift before error messages enabled */
  int yychar1 = 0;		/*  lookahead token as an internal (translated) token number */

  short	yyssa[YYINITDEPTH];	/*  the state stack			*/
  YYSTYPE yyvsa[YYINITDEPTH];	/*  the semantic value stack		*/

  short *yyss = yyssa;		/*  refer to the stacks thru separate pointers */
  YYSTYPE *yyvs = yyvsa;	/*  to allow yyoverflow to reallocate them elsewhere */

#ifdef YYLSP_NEEDED
  YYLTYPE yylsa[YYINITDEPTH];	/*  the location stack			*/
  YYLTYPE *yyls = yylsa;
  YYLTYPE *yylsp;

#define YYPOPSTACK   (yyvsp--, yyssp--, yylsp--)
#else
#define YYPOPSTACK   (yyvsp--, yyssp--)
#endif

  int yystacksize = YYINITDEPTH;
  int yyfree_stacks = 0;

#ifdef YYPURE
  int yychar;
  YYSTYPE yylval;
  int yynerrs;
#ifdef YYLSP_NEEDED
  YYLTYPE yylloc;
#endif
#endif

  YYSTYPE yyval;		/*  the variable used to return		*/
				/*  semantic values from the action	*/
				/*  routines				*/

  int yylen;

#if YYDEBUG != 0
  if (yydebug)
    fprintf(stderr, "Starting parse\n");
#endif

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY;		/* Cause a token to be read.  */

  /* Initialize stack pointers.
     Waste one element of value and location stack
     so that they stay on the same level as the state stack.
     The wasted elements are never initialized.  */

  yyssp = yyss - 1;
  yyvsp = yyvs;
#ifdef YYLSP_NEEDED
  yylsp = yyls;
#endif

/* Push a new state, which is found in  yystate  .  */
/* In all cases, when you get here, the value and location stacks
   have just been pushed. so pushing a state here evens the stacks.  */
yynewstate:

  *++yyssp = yystate;

  if (yyssp >= yyss + yystacksize - 1)
    {
      /* Give user a chance to reallocate the stack */
      /* Use copies of these so that the &'s don't force the real ones into memory. */
      YYSTYPE *yyvs1 = yyvs;
      short *yyss1 = yyss;
#ifdef YYLSP_NEEDED
      YYLTYPE *yyls1 = yyls;
#endif

      /* Get the current used size of the three stacks, in elements.  */
      int size = yyssp - yyss + 1;

#ifdef yyoverflow
      /* Each stack pointer address is followed by the size of
	 the data in use in that stack, in bytes.  */
#ifdef YYLSP_NEEDED
      /* This used to be a conditional around just the two extra args,
	 but that might be undefined if yyoverflow is a macro.  */
      yyoverflow("parser stack overflow",
		 &yyss1, size * sizeof (*yyssp),
		 &yyvs1, size * sizeof (*yyvsp),
		 &yyls1, size * sizeof (*yylsp),
		 &yystacksize);
#else
      yyoverflow("parser stack overflow",
		 &yyss1, size * sizeof (*yyssp),
		 &yyvs1, size * sizeof (*yyvsp),
		 &yystacksize);
#endif

      yyss = yyss1; yyvs = yyvs1;
#ifdef YYLSP_NEEDED
      yyls = yyls1;
#endif
#else /* no yyoverflow */
      /* Extend the stack our own way.  */
      if (yystacksize >= YYMAXDEPTH)
	{
	  yyerror("parser stack overflow");
	  if (yyfree_stacks)
	    {
	      free (yyss);
	      free (yyvs);
#ifdef YYLSP_NEEDED
	      free (yyls);
#endif
	    }
	  return 2;
	}
      yystacksize *= 2;
      if (yystacksize > YYMAXDEPTH)
	yystacksize = YYMAXDEPTH;
#ifndef YYSTACK_USE_ALLOCA
      yyfree_stacks = 1;
#endif
      yyss = (short *) YYSTACK_ALLOC (yystacksize * sizeof (*yyssp));
      __yy_memcpy ((char *)yyss, (char *)yyss1,
		   size * (unsigned int) sizeof (*yyssp));
      yyvs = (YYSTYPE *) YYSTACK_ALLOC (yystacksize * sizeof (*yyvsp));
      __yy_memcpy ((char *)yyvs, (char *)yyvs1,
		   size * (unsigned int) sizeof (*yyvsp));
#ifdef YYLSP_NEEDED
      yyls = (YYLTYPE *) YYSTACK_ALLOC (yystacksize * sizeof (*yylsp));
      __yy_memcpy ((char *)yyls, (char *)yyls1,
		   size * (unsigned int) sizeof (*yylsp));
#endif
#endif /* no yyoverflow */

      yyssp = yyss + size - 1;
      yyvsp = yyvs + size - 1;
#ifdef YYLSP_NEEDED
      yylsp = yyls + size - 1;
#endif

#if YYDEBUG != 0
      if (yydebug)
	fprintf(stderr, "Stack size increased to %d\n", yystacksize);
#endif

      if (yyssp >= yyss + yystacksize - 1)
	YYABORT;
    }

#if YYDEBUG != 0
  if (yydebug)
    fprintf(stderr, "Entering state %d\n", yystate);
#endif

  goto yybackup;
 yybackup:

/* Do appropriate processing given the current state.  */
/* Read a lookahead token if we need one and don't already have one.  */
/* yyresume: */

  /* First try to decide what to do without reference to lookahead token.  */

  yyn = yypact[yystate];
  if (yyn == YYFLAG)
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* yychar is either YYEMPTY or YYEOF
     or a valid token in external form.  */

  if (yychar == YYEMPTY)
    {
#if YYDEBUG != 0
      if (yydebug)
	fprintf(stderr, "Reading a token: ");
#endif
      yychar = YYLEX;
    }

  /* Convert token to internal form (in yychar1) for indexing tables with */

  if (yychar <= 0)		/* This means end of input. */
    {
      yychar1 = 0;
      yychar = YYEOF;		/* Don't call YYLEX any more */

#if YYDEBUG != 0
      if (yydebug)
	fprintf(stderr, "Now at end of input.\n");
#endif
    }
  else
    {
      yychar1 = YYTRANSLATE(yychar);

#if YYDEBUG != 0
      if (yydebug)
	{
	  fprintf (stderr, "Next token is %d (%s", yychar, yytname[yychar1]);
	  /* Give the individual parser a way to print the precise meaning
	     of a token, for further debugging info.  */
#ifdef YYPRINT
	  YYPRINT (stderr, yychar, yylval);
#endif
	  fprintf (stderr, ")\n");
	}
#endif
    }

  yyn += yychar1;
  if (yyn < 0 || yyn > YYLAST || yycheck[yyn] != yychar1)
    goto yydefault;

  yyn = yytable[yyn];

  /* yyn is what to do for this token type in this state.
     Negative => reduce, -yyn is rule number.
     Positive => shift, yyn is new state.
       New state is final state => don't bother to shift,
       just return success.
     0, or most negative number => error.  */

  if (yyn < 0)
    {
      if (yyn == YYFLAG)
	goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }
  else if (yyn == 0)
    goto yyerrlab;

  if (yyn == YYFINAL)
    YYACCEPT;

  /* Shift the lookahead token.  */

#if YYDEBUG != 0
  if (yydebug)
    fprintf(stderr, "Shifting token %d (%s), ", yychar, yytname[yychar1]);
#endif

  /* Discard the token being shifted unless it is eof.  */
  if (yychar != YYEOF)
    yychar = YYEMPTY;

  *++yyvsp = yylval;
#ifdef YYLSP_NEEDED
  *++yylsp = yylloc;
#endif

  /* count tokens shifted since error; after three, turn off error status.  */
  if (yyerrstatus) yyerrstatus--;

  yystate = yyn;
  goto yynewstate;

/* Do the default action for the current state.  */
yydefault:

  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;

/* Do a reduction.  yyn is the number of a rule to reduce with.  */
yyreduce:
  yylen = yyr2[yyn];
  if (yylen > 0)
    yyval = yyvsp[1-yylen]; /* implement default value of the action */

#if YYDEBUG != 0
  if (yydebug)
    {
      int i;

      fprintf (stderr, "Reducing via rule %d (line %d), ",
	       yyn, yyrline[yyn]);

      /* Print the symbols being reduced, and their result.  */
      for (i = yyprhs[yyn]; yyrhs[i] > 0; i++)
	fprintf (stderr, "%s ", yytname[yyrhs[i]]);
      fprintf (stderr, " -> %s\n", yytname[yyr1[yyn]]);
    }
#endif


  switch (yyn) {

case 1:
#line 97 "xi-grammar.y"
{ yyval.modlist = yyvsp[0].modlist; modlist = yyvsp[0].modlist; ;
    break;}
case 2:
#line 101 "xi-grammar.y"
{ yyval.modlist = 0; ;
    break;}
case 3:
#line 103 "xi-grammar.y"
{ yyval.modlist = new ModuleList(lineno, yyvsp[-1].module, yyvsp[0].modlist); ;
    break;}
case 4:
#line 107 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 5:
#line 109 "xi-grammar.y"
{ yyval.intval = 1; ;
    break;}
case 6:
#line 113 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 7:
#line 115 "xi-grammar.y"
{ yyval.intval = 1; ;
    break;}
case 8:
#line 119 "xi-grammar.y"
{ yyval.strval = yyvsp[0].strval; ;
    break;}
case 9:
#line 123 "xi-grammar.y"
{ yyval.strval = yyvsp[0].strval; ;
    break;}
case 10:
#line 125 "xi-grammar.y"
{
		  char *tmp = new char[strlen(yyvsp[-3].strval)+strlen(yyvsp[0].strval)+3];
		  sprintf(tmp,"%s::%s", yyvsp[-3].strval, yyvsp[0].strval);
		  yyval.strval = tmp;
		;
    break;}
case 11:
#line 133 "xi-grammar.y"
{ yyval.module = new Module(lineno, yyvsp[-1].strval, yyvsp[0].conslist); ;
    break;}
case 12:
#line 135 "xi-grammar.y"
{ yyval.module = new Module(lineno, yyvsp[-1].strval, yyvsp[0].conslist); yyval.module->setMain(); ;
    break;}
case 13:
#line 139 "xi-grammar.y"
{ yyval.conslist = 0; ;
    break;}
case 14:
#line 141 "xi-grammar.y"
{ yyval.conslist = yyvsp[-2].conslist; ;
    break;}
case 15:
#line 145 "xi-grammar.y"
{ yyval.conslist = 0; ;
    break;}
case 16:
#line 147 "xi-grammar.y"
{ yyval.conslist = new ConstructList(lineno, yyvsp[-1].construct, yyvsp[0].conslist); ;
    break;}
case 17:
#line 151 "xi-grammar.y"
{ if(yyvsp[-2].conslist) yyvsp[-2].conslist->setExtern(yyvsp[-4].intval); yyval.construct = yyvsp[-2].conslist; ;
    break;}
case 18:
#line 153 "xi-grammar.y"
{ yyvsp[0].module->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].module; ;
    break;}
case 19:
#line 155 "xi-grammar.y"
{ yyvsp[-1].readonly->setExtern(yyvsp[-2].intval); yyval.construct = yyvsp[-1].readonly; ;
    break;}
case 20:
#line 157 "xi-grammar.y"
{ yyvsp[-1].readonly->setExtern(yyvsp[-2].intval); yyval.construct = yyvsp[-1].readonly; ;
    break;}
case 21:
#line 159 "xi-grammar.y"
{ yyvsp[-1].message->setExtern(yyvsp[-2].intval); yyval.construct = yyvsp[-1].message; ;
    break;}
case 22:
#line 161 "xi-grammar.y"
{ yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; ;
    break;}
case 23:
#line 163 "xi-grammar.y"
{ yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; ;
    break;}
case 24:
#line 165 "xi-grammar.y"
{ yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; ;
    break;}
case 25:
#line 167 "xi-grammar.y"
{ yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; ;
    break;}
case 26:
#line 169 "xi-grammar.y"
{ yyvsp[0].templat->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].templat; ;
    break;}
case 27:
#line 173 "xi-grammar.y"
{ yyval.tparam = new TParamType(yyvsp[0].type); ;
    break;}
case 28:
#line 175 "xi-grammar.y"
{ yyval.tparam = new TParamVal(yyvsp[0].strval); ;
    break;}
case 29:
#line 177 "xi-grammar.y"
{ yyval.tparam = new TParamVal(yyvsp[0].strval); ;
    break;}
case 30:
#line 181 "xi-grammar.y"
{ yyval.tparlist = new TParamList(yyvsp[0].tparam); ;
    break;}
case 31:
#line 183 "xi-grammar.y"
{ yyval.tparlist = new TParamList(yyvsp[-2].tparam, yyvsp[0].tparlist); ;
    break;}
case 32:
#line 187 "xi-grammar.y"
{ yyval.tparlist = 0; ;
    break;}
case 33:
#line 189 "xi-grammar.y"
{ yyval.tparlist = yyvsp[0].tparlist; ;
    break;}
case 34:
#line 193 "xi-grammar.y"
{ yyval.tparlist = 0; ;
    break;}
case 35:
#line 195 "xi-grammar.y"
{ yyval.tparlist = yyvsp[-1].tparlist; ;
    break;}
case 36:
#line 199 "xi-grammar.y"
{ yyval.type = new BuiltinType("int"); ;
    break;}
case 37:
#line 201 "xi-grammar.y"
{ yyval.type = new BuiltinType("long"); ;
    break;}
case 38:
#line 203 "xi-grammar.y"
{ yyval.type = new BuiltinType("short"); ;
    break;}
case 39:
#line 205 "xi-grammar.y"
{ yyval.type = new BuiltinType("char"); ;
    break;}
case 40:
#line 207 "xi-grammar.y"
{ yyval.type = new BuiltinType("unsigned int"); ;
    break;}
case 41:
#line 209 "xi-grammar.y"
{ yyval.type = new BuiltinType("long"); ;
    break;}
case 42:
#line 211 "xi-grammar.y"
{ yyval.type = new BuiltinType("short"); ;
    break;}
case 43:
#line 213 "xi-grammar.y"
{ yyval.type = new BuiltinType("char"); ;
    break;}
case 44:
#line 215 "xi-grammar.y"
{ yyval.type = new BuiltinType("long long"); ;
    break;}
case 45:
#line 217 "xi-grammar.y"
{ yyval.type = new BuiltinType("float"); ;
    break;}
case 46:
#line 219 "xi-grammar.y"
{ yyval.type = new BuiltinType("double"); ;
    break;}
case 47:
#line 221 "xi-grammar.y"
{ yyval.type = new BuiltinType("long double"); ;
    break;}
case 48:
#line 223 "xi-grammar.y"
{ yyval.type = new BuiltinType("void"); ;
    break;}
case 49:
#line 227 "xi-grammar.y"
{ yyval.ntype = new NamedType(yyvsp[-1].strval, yyvsp[0].tparlist); ;
    break;}
case 50:
#line 231 "xi-grammar.y"
{ yyval.type = yyvsp[0].type; ;
    break;}
case 51:
#line 233 "xi-grammar.y"
{ yyval.type = yyvsp[0].ntype; ;
    break;}
case 52:
#line 237 "xi-grammar.y"
{ yyval.ptype = new PtrType(yyvsp[-1].type); ;
    break;}
case 53:
#line 241 "xi-grammar.y"
{ yyvsp[-1].ptype->indirect(); yyval.ptype = yyvsp[-1].ptype; ;
    break;}
case 54:
#line 243 "xi-grammar.y"
{ yyvsp[-1].ptype->indirect(); yyval.ptype = yyvsp[-1].ptype; ;
    break;}
case 55:
#line 247 "xi-grammar.y"
{ yyval.val = new Value(yyvsp[0].strval); ;
    break;}
case 56:
#line 249 "xi-grammar.y"
{ yyval.val = new Value(yyvsp[0].strval); ;
    break;}
case 57:
#line 253 "xi-grammar.y"
{ yyval.type = new ArrayType(yyvsp[-3].type, yyvsp[-1].val); ;
    break;}
case 58:
#line 257 "xi-grammar.y"
{ yyval.ftype = new FuncType(yyvsp[-7].type, yyvsp[-4].strval, yyvsp[-1].typelist); ;
    break;}
case 59:
#line 261 "xi-grammar.y"
{ yyval.type = yyvsp[0].type; ;
    break;}
case 60:
#line 263 "xi-grammar.y"
{ yyval.type = (Type*) yyvsp[0].ptype; ;
    break;}
case 61:
#line 265 "xi-grammar.y"
{ yyval.type = (Type*) yyvsp[0].ptype; ;
    break;}
case 62:
#line 267 "xi-grammar.y"
{ yyval.type = yyvsp[0].type; ;
    break;}
case 63:
#line 269 "xi-grammar.y"
{ yyval.type = yyvsp[0].ftype; ;
    break;}
case 64:
#line 273 "xi-grammar.y"
{ yyval.typelist = 0; ;
    break;}
case 65:
#line 275 "xi-grammar.y"
{ yyval.typelist = new TypeList(yyvsp[0].type); ;
    break;}
case 66:
#line 277 "xi-grammar.y"
{ yyval.typelist = new TypeList(yyvsp[-2].type, yyvsp[0].typelist); ;
    break;}
case 67:
#line 281 "xi-grammar.y"
{ yyval.val = yyvsp[-1].val; ;
    break;}
case 68:
#line 285 "xi-grammar.y"
{ yyval.vallist = 0; ;
    break;}
case 69:
#line 287 "xi-grammar.y"
{ yyval.vallist = new ValueList(yyvsp[-1].val, yyvsp[0].vallist); ;
    break;}
case 70:
#line 291 "xi-grammar.y"
{ yyval.readonly = new Readonly(lineno, yyvsp[-2].type, yyvsp[-1].strval, yyvsp[0].vallist); ;
    break;}
case 71:
#line 295 "xi-grammar.y"
{ yyval.readonly = new Readonly(lineno, yyvsp[-2].type, yyvsp[0].strval, 0, 1); ;
    break;}
case 72:
#line 299 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 73:
#line 301 "xi-grammar.y"
{ yyval.intval = yyvsp[-1].intval; ;
    break;}
case 74:
#line 305 "xi-grammar.y"
{ yyval.intval = yyvsp[0].intval; ;
    break;}
case 75:
#line 307 "xi-grammar.y"
{ yyval.intval = yyvsp[-2].intval | yyvsp[0].intval; ;
    break;}
case 76:
#line 311 "xi-grammar.y"
{ yyval.intval = SPACKED; ;
    break;}
case 77:
#line 313 "xi-grammar.y"
{ yyval.intval = SVARSIZE; ;
    break;}
case 78:
#line 315 "xi-grammar.y"
{ yyval.intval = SVARRAYS; ;
    break;}
case 79:
#line 319 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 80:
#line 321 "xi-grammar.y"
{ yyval.intval = yyvsp[-1].intval; ;
    break;}
case 81:
#line 325 "xi-grammar.y"
{ yyval.intval = yyvsp[0].intval; ;
    break;}
case 82:
#line 327 "xi-grammar.y"
{ yyval.intval = yyvsp[-2].intval | yyvsp[0].intval; ;
    break;}
case 83:
#line 331 "xi-grammar.y"
{ yyval.intval = 0x01; ;
    break;}
case 84:
#line 335 "xi-grammar.y"
{ yyval.mv = new MsgVar(yyvsp[-4].type, yyvsp[-3].strval); ;
    break;}
case 85:
#line 339 "xi-grammar.y"
{ yyval.mvlist = new MsgVarList(yyvsp[0].mv); ;
    break;}
case 86:
#line 341 "xi-grammar.y"
{ yyval.mvlist = new MsgVarList(yyvsp[-1].mv, yyvsp[0].mvlist); ;
    break;}
case 87:
#line 345 "xi-grammar.y"
{ yyval.message = new Message(lineno, yyvsp[0].ntype, yyvsp[-1].intval); ;
    break;}
case 88:
#line 347 "xi-grammar.y"
{ yyval.message = new Message(lineno, yyvsp[-3].ntype, yyvsp[-4].intval, yyvsp[-1].typelist); ;
    break;}
case 89:
#line 349 "xi-grammar.y"
{ yyval.message = new Message(lineno, yyvsp[-3].ntype, yyvsp[-4].intval, 0, yyvsp[-1].mvlist); ;
    break;}
case 90:
#line 353 "xi-grammar.y"
{ yyval.typelist = 0; ;
    break;}
case 91:
#line 355 "xi-grammar.y"
{ yyval.typelist = yyvsp[0].typelist; ;
    break;}
case 92:
#line 359 "xi-grammar.y"
{ yyval.typelist = new TypeList(yyvsp[0].ntype); ;
    break;}
case 93:
#line 361 "xi-grammar.y"
{ yyval.typelist = new TypeList(yyvsp[-2].ntype, yyvsp[0].typelist); ;
    break;}
case 94:
#line 365 "xi-grammar.y"
{ yyval.chare = new Chare(lineno, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist, yyvsp[-3].intval); ;
    break;}
case 95:
#line 367 "xi-grammar.y"
{ yyval.chare = new MainChare(lineno, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist, yyvsp[-3].intval); ;
    break;}
case 96:
#line 371 "xi-grammar.y"
{ yyval.chare = new Group(lineno, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist, yyvsp[-3].intval); ;
    break;}
case 97:
#line 375 "xi-grammar.y"
{ yyval.chare = new NodeGroup(lineno, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist, yyvsp[-3].intval); ;
    break;}
case 98:
#line 379 "xi-grammar.y"
{/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",yyvsp[-2].strval);
			yyval.ntype = new NamedType(buf); 
		;
    break;}
case 99:
#line 385 "xi-grammar.y"
{ yyval.ntype = new NamedType(yyvsp[-1].strval); ;
    break;}
case 100:
#line 389 "xi-grammar.y"
{ yyval.chare = new Array(lineno, yyvsp[-3].ntype, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 101:
#line 393 "xi-grammar.y"
{ yyval.chare = new Chare(lineno, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist, yyvsp[-3].intval);;
    break;}
case 102:
#line 395 "xi-grammar.y"
{ yyval.chare = new MainChare(lineno, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist, yyvsp[-3].intval); ;
    break;}
case 103:
#line 399 "xi-grammar.y"
{ yyval.chare = new Group(lineno, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist, yyvsp[-3].intval); ;
    break;}
case 104:
#line 403 "xi-grammar.y"
{ yyval.chare = new NodeGroup( lineno, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist, yyvsp[-3].intval); ;
    break;}
case 105:
#line 407 "xi-grammar.y"
{ yyval.chare = new Array( lineno, yyvsp[-3].ntype, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 106:
#line 411 "xi-grammar.y"
{ yyval.message = new Message(lineno, new NamedType(yyvsp[-1].strval), yyvsp[-2].intval); ;
    break;}
case 107:
#line 415 "xi-grammar.y"
{ yyval.type = 0; ;
    break;}
case 108:
#line 417 "xi-grammar.y"
{ yyval.type = yyvsp[0].type; ;
    break;}
case 109:
#line 421 "xi-grammar.y"
{ yyval.strval = 0; ;
    break;}
case 110:
#line 423 "xi-grammar.y"
{ yyval.strval = yyvsp[0].strval; ;
    break;}
case 111:
#line 425 "xi-grammar.y"
{ yyval.strval = yyvsp[0].strval; ;
    break;}
case 112:
#line 429 "xi-grammar.y"
{ yyval.tvar = new TType(new NamedType(yyvsp[-1].strval), yyvsp[0].type); ;
    break;}
case 113:
#line 431 "xi-grammar.y"
{ yyval.tvar = new TFunc(yyvsp[-1].ftype, yyvsp[0].strval); ;
    break;}
case 114:
#line 433 "xi-grammar.y"
{ yyval.tvar = new TName(yyvsp[-2].type, yyvsp[-1].strval, yyvsp[0].strval); ;
    break;}
case 115:
#line 437 "xi-grammar.y"
{ yyval.tvarlist = new TVarList(yyvsp[0].tvar); ;
    break;}
case 116:
#line 439 "xi-grammar.y"
{ yyval.tvarlist = new TVarList(yyvsp[-2].tvar, yyvsp[0].tvarlist); ;
    break;}
case 117:
#line 443 "xi-grammar.y"
{ yyval.tvarlist = yyvsp[-1].tvarlist; ;
    break;}
case 118:
#line 447 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); ;
    break;}
case 119:
#line 449 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); ;
    break;}
case 120:
#line 451 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); ;
    break;}
case 121:
#line 453 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); ;
    break;}
case 122:
#line 455 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].message); yyvsp[0].message->setTemplate(yyval.templat); ;
    break;}
case 123:
#line 459 "xi-grammar.y"
{ yyval.mbrlist = 0; ;
    break;}
case 124:
#line 461 "xi-grammar.y"
{ yyval.mbrlist = yyvsp[-2].mbrlist; ;
    break;}
case 125:
#line 465 "xi-grammar.y"
{ yyval.mbrlist = 0; ;
    break;}
case 126:
#line 467 "xi-grammar.y"
{ yyval.mbrlist = new MemberList(yyvsp[-1].member, yyvsp[0].mbrlist); ;
    break;}
case 127:
#line 471 "xi-grammar.y"
{ yyval.member = yyvsp[-1].entry; ;
    break;}
case 128:
#line 473 "xi-grammar.y"
{ yyval.member = yyvsp[-1].readonly; ;
    break;}
case 129:
#line 475 "xi-grammar.y"
{ yyval.member = yyvsp[-1].readonly; ;
    break;}
case 130:
#line 479 "xi-grammar.y"
{ yyval.entry = new Entry(lineno, yyvsp[-5].intval|yyvsp[-1].intval, new BuiltinType("void"), yyvsp[-3].strval, yyvsp[-2].rtype, yyvsp[0].val); ;
    break;}
case 131:
#line 481 "xi-grammar.y"
{ yyval.entry = new Entry(lineno, yyvsp[-5].intval|yyvsp[-1].intval, yyvsp[-4].ptype, yyvsp[-3].strval, yyvsp[-2].rtype, yyvsp[0].val); ;
    break;}
case 132:
#line 483 "xi-grammar.y"
{ yyval.entry = new Entry(lineno, yyvsp[-2].intval, 0, yyvsp[-1].strval, yyvsp[0].rtype, 0); ;
    break;}
case 133:
#line 487 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 134:
#line 489 "xi-grammar.y"
{ yyval.intval = yyvsp[-1].intval; ;
    break;}
case 135:
#line 493 "xi-grammar.y"
{ yyval.intval = yyvsp[0].intval; ;
    break;}
case 136:
#line 495 "xi-grammar.y"
{ yyval.intval = yyvsp[-2].intval | yyvsp[0].intval; ;
    break;}
case 137:
#line 499 "xi-grammar.y"
{ yyval.intval = STHREADED; ;
    break;}
case 138:
#line 501 "xi-grammar.y"
{ yyval.intval = SSYNC; ;
    break;}
case 139:
#line 503 "xi-grammar.y"
{ yyval.intval = SLOCKED; ;
    break;}
case 140:
#line 505 "xi-grammar.y"
{ yyval.intval = SVIRTUAL; ;
    break;}
case 141:
#line 509 "xi-grammar.y"
{ yyval.rtype = 0; ;
    break;}
case 142:
#line 511 "xi-grammar.y"
{ yyval.rtype = new BuiltinType("void"); ;
    break;}
case 143:
#line 513 "xi-grammar.y"
{ yyval.rtype = yyvsp[0].ptype; ;
    break;}
case 144:
#line 517 "xi-grammar.y"
{ yyval.rtype = yyvsp[-1].rtype; ;
    break;}
case 145:
#line 521 "xi-grammar.y"
{ yyval.val = 0; ;
    break;}
case 146:
#line 523 "xi-grammar.y"
{ yyval.val = new Value(yyvsp[0].strval); ;
    break;}
case 147:
#line 527 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 148:
#line 529 "xi-grammar.y"
{ if(strcmp(yyvsp[0].strval, "0")) { yyerror("expected 0"); exit(1); }
		  yyval.intval = SPURE; 
		;
    break;}
}
   /* the action file gets copied in in place of this dollarsign */
#line 543 "/usr/lib/bison.simple"

  yyvsp -= yylen;
  yyssp -= yylen;
#ifdef YYLSP_NEEDED
  yylsp -= yylen;
#endif

#if YYDEBUG != 0
  if (yydebug)
    {
      short *ssp1 = yyss - 1;
      fprintf (stderr, "state stack now");
      while (ssp1 != yyssp)
	fprintf (stderr, " %d", *++ssp1);
      fprintf (stderr, "\n");
    }
#endif

  *++yyvsp = yyval;

#ifdef YYLSP_NEEDED
  yylsp++;
  if (yylen == 0)
    {
      yylsp->first_line = yylloc.first_line;
      yylsp->first_column = yylloc.first_column;
      yylsp->last_line = (yylsp-1)->last_line;
      yylsp->last_column = (yylsp-1)->last_column;
      yylsp->text = 0;
    }
  else
    {
      yylsp->last_line = (yylsp+yylen-1)->last_line;
      yylsp->last_column = (yylsp+yylen-1)->last_column;
    }
#endif

  /* Now "shift" the result of the reduction.
     Determine what state that goes to,
     based on the state we popped back to
     and the rule number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTBASE] + *yyssp;
  if (yystate >= 0 && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTBASE];

  goto yynewstate;

yyerrlab:   /* here on detecting error */

  if (! yyerrstatus)
    /* If not already recovering from an error, report this error.  */
    {
      ++yynerrs;

#ifdef YYERROR_VERBOSE
      yyn = yypact[yystate];

      if (yyn > YYFLAG && yyn < YYLAST)
	{
	  int size = 0;
	  char *msg;
	  int x, count;

	  count = 0;
	  /* Start X at -yyn if nec to avoid negative indexes in yycheck.  */
	  for (x = (yyn < 0 ? -yyn : 0);
	       x < (sizeof(yytname) / sizeof(char *)); x++)
	    if (yycheck[x + yyn] == x)
	      size += strlen(yytname[x]) + 15, count++;
	  msg = (char *) malloc(size + 15);
	  if (msg != 0)
	    {
	      strcpy(msg, "parse error");

	      if (count < 5)
		{
		  count = 0;
		  for (x = (yyn < 0 ? -yyn : 0);
		       x < (sizeof(yytname) / sizeof(char *)); x++)
		    if (yycheck[x + yyn] == x)
		      {
			strcat(msg, count == 0 ? ", expecting `" : " or `");
			strcat(msg, yytname[x]);
			strcat(msg, "'");
			count++;
		      }
		}
	      yyerror(msg);
	      free(msg);
	    }
	  else
	    yyerror ("parse error; also virtual memory exceeded");
	}
      else
#endif /* YYERROR_VERBOSE */
	yyerror("parse error");
    }

  goto yyerrlab1;
yyerrlab1:   /* here on error raised explicitly by an action */

  if (yyerrstatus == 3)
    {
      /* if just tried and failed to reuse lookahead token after an error, discard it.  */

      /* return failure if at end of input */
      if (yychar == YYEOF)
	YYABORT;

#if YYDEBUG != 0
      if (yydebug)
	fprintf(stderr, "Discarding token %d (%s).\n", yychar, yytname[yychar1]);
#endif

      yychar = YYEMPTY;
    }

  /* Else will try to reuse lookahead token
     after shifting the error token.  */

  yyerrstatus = 3;		/* Each real token shifted decrements this */

  goto yyerrhandle;

yyerrdefault:  /* current state does not do anything special for the error token. */

#if 0
  /* This is wrong; only states that explicitly want error tokens
     should shift them.  */
  yyn = yydefact[yystate];  /* If its default is to accept any token, ok.  Otherwise pop it.*/
  if (yyn) goto yydefault;
#endif

yyerrpop:   /* pop the current state because it cannot handle the error token */

  if (yyssp == yyss) YYABORT;
  yyvsp--;
  yystate = *--yyssp;
#ifdef YYLSP_NEEDED
  yylsp--;
#endif

#if YYDEBUG != 0
  if (yydebug)
    {
      short *ssp1 = yyss - 1;
      fprintf (stderr, "Error: state stack now");
      while (ssp1 != yyssp)
	fprintf (stderr, " %d", *++ssp1);
      fprintf (stderr, "\n");
    }
#endif

yyerrhandle:

  yyn = yypact[yystate];
  if (yyn == YYFLAG)
    goto yyerrdefault;

  yyn += YYTERROR;
  if (yyn < 0 || yyn > YYLAST || yycheck[yyn] != YYTERROR)
    goto yyerrdefault;

  yyn = yytable[yyn];
  if (yyn < 0)
    {
      if (yyn == YYFLAG)
	goto yyerrpop;
      yyn = -yyn;
      goto yyreduce;
    }
  else if (yyn == 0)
    goto yyerrpop;

  if (yyn == YYFINAL)
    YYACCEPT;

#if YYDEBUG != 0
  if (yydebug)
    fprintf(stderr, "Shifting error token, ");
#endif

  *++yyvsp = yylval;
#ifdef YYLSP_NEEDED
  *++yylsp = yylloc;
#endif

  yystate = yyn;
  goto yynewstate;

 yyacceptlab:
  /* YYACCEPT comes here.  */
  if (yyfree_stacks)
    {
      free (yyss);
      free (yyvs);
#ifdef YYLSP_NEEDED
      free (yyls);
#endif
    }
  return 0;

 yyabortlab:
  /* YYABORT comes here.  */
  if (yyfree_stacks)
    {
      free (yyss);
      free (yyvs);
#ifdef YYLSP_NEEDED
      free (yyls);
#endif
    }
  return 1;
}
#line 533 "xi-grammar.y"

void yyerror(const char *mesg)
{
  cout << "Syntax error at line " << lineno << ": " << mesg << endl;
  // return 0;
}
