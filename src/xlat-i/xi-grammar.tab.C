
/*  A Bison parser, made from xi-grammar.y
 by  GNU Bison version 1.27
  */

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
#define	TEMPLATE	269
#define	SYNC	270
#define	EXCLUSIVE	271
#define	VOID	272
#define	PACKED	273
#define	VARSIZE	274
#define	ENTRY	275
#define	MAINCHARE	276
#define	IDENT	277
#define	NUMBER	278
#define	LITERAL	279
#define	INT	280
#define	LONG	281
#define	SHORT	282
#define	CHAR	283
#define	FLOAT	284
#define	DOUBLE	285
#define	UNSIGNED	286

#line 1 "xi-grammar.y"

#include <iostream.h>
#include "xi-symbol.h"

extern int yylex (void) ;
void yyerror(char *);
extern int lineno;
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
  char *strval;
  int intval;
} YYSTYPE;
#include <stdio.h>

#ifndef __cplusplus
#ifndef __STDC__
#define const
#endif
#endif



#define	YYFINAL		232
#define	YYFLAG		-32768
#define	YYNTBASE	46

#define YYTRANSLATE(x) ((unsigned)(x) <= 286 ? yytranslate[x] : 104)

static const char yytranslate[] = {     0,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,    42,
    43,    39,     2,    36,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,    44,    33,    37,
    45,    38,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
    40,     2,    41,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,    34,     2,    35,     2,     2,     2,     2,     2,
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
    27,    28,    29,    30,    31,    32
};

#if YYDEBUG != 0
static const short yyprhs[] = {     0,
     0,     2,     3,     6,     7,     9,    10,    12,    14,    18,
    22,    24,    29,    30,    33,    39,    42,    46,    50,    54,
    57,    60,    63,    66,    69,    71,    73,    75,    77,    81,
    82,    84,    85,    89,    91,    93,    95,    97,   100,   103,
   106,   109,   112,   114,   116,   119,   121,   124,   126,   128,
   131,   134,   137,   139,   141,   146,   155,   157,   159,   161,
   163,   165,   166,   168,   172,   176,   177,   180,   185,   191,
   192,   196,   198,   202,   204,   206,   210,   211,   214,   216,
   220,   225,   230,   235,   240,   245,   250,   255,   260,   265,
   270,   275,   276,   279,   280,   283,   286,   290,   293,   297,
   299,   303,   308,   311,   314,   317,   320,   323,   325,   330,
   331,   334,   337,   340,   343,   350,   357,   362,   363,   367,
   369,   373,   375,   377,   379,   380,   382,   384,   388,   389
};

static const short yyrhs[] = {    47,
     0,     0,    51,    47,     0,     0,     5,     0,     0,    33,
     0,    23,     0,     3,    50,    52,     0,     4,    50,    52,
     0,    33,     0,    34,    53,    35,    49,     0,     0,    54,
    53,     0,    48,    34,    53,    35,    49,     0,    48,    51,
     0,    48,    71,    33,     0,    48,    72,    33,     0,    48,
    76,    33,     0,    48,    79,     0,    48,    80,     0,    48,
    81,     0,    48,    82,     0,    48,    93,     0,    67,     0,
    24,     0,    25,     0,    55,     0,    55,    36,    56,     0,
     0,    56,     0,     0,    37,    57,    38,     0,    26,     0,
    27,     0,    28,     0,    29,     0,    32,    26,     0,    32,
    27,     0,    32,    28,     0,    32,    29,     0,    27,    27,
     0,    30,     0,    31,     0,    27,    31,     0,    18,     0,
    50,    58,     0,    59,     0,    60,     0,    61,    39,     0,
    62,    39,     0,    63,    39,     0,    24,     0,    50,     0,
    67,    40,    64,    41,     0,    67,    42,    39,    50,    43,
    42,    68,    43,     0,    61,     0,    62,     0,    63,     0,
    65,     0,    66,     0,     0,    67,     0,    67,    36,    68,
     0,    40,    64,    41,     0,     0,    69,    70,     0,     6,
    67,    50,    70,     0,     6,    11,    61,    39,    50,     0,
     0,    40,    74,    41,     0,    75,     0,    75,    36,    74,
     0,    19,     0,    20,     0,    11,    73,    60,     0,     0,
    44,    78,     0,    60,     0,    60,    36,    78,     0,     7,
    60,    77,    94,     0,    22,    60,    77,    94,     0,     8,
    60,    77,    94,     0,     9,    60,    77,    94,     0,    10,
    60,    77,    94,     0,     7,    50,    77,    94,     0,    22,
    50,    77,    94,     0,     8,    50,    77,    94,     0,     9,
    50,    77,    94,     0,    10,    50,    77,    94,     0,    11,
    73,    50,    33,     0,     0,    45,    67,     0,     0,    45,
    24,     0,    45,    25,     0,    12,    50,    88,     0,    66,
    89,     0,    67,    50,    89,     0,    90,     0,    90,    36,
    91,     0,    15,    37,    91,    38,     0,    92,    83,     0,
    92,    84,     0,    92,    85,     0,    92,    86,     0,    92,
    87,     0,    33,     0,    34,    95,    35,    49,     0,     0,
    96,    95,     0,    97,    33,     0,    71,    33,     0,    72,
    33,     0,    21,    98,    18,    50,   102,   103,     0,    21,
    98,    62,    50,   102,   103,     0,    21,    98,    50,   102,
     0,     0,    40,    99,    41,     0,   100,     0,   100,    36,
    99,     0,    14,     0,    16,     0,    17,     0,     0,    18,
     0,    62,     0,    42,   101,    43,     0,     0,    13,    45,
    24,     0
};

#endif

#if YYDEBUG != 0
static const short yyrline[] = { 0,
    90,    94,    96,   100,   102,   106,   108,   112,   116,   118,
   122,   124,   128,   130,   134,   136,   138,   140,   142,   144,
   146,   148,   150,   152,   156,   158,   160,   164,   166,   170,
   172,   176,   178,   182,   184,   186,   188,   190,   192,   194,
   196,   198,   200,   202,   204,   206,   210,   214,   216,   220,
   224,   226,   230,   232,   236,   240,   244,   246,   248,   250,
   252,   256,   258,   260,   264,   268,   270,   274,   278,   282,
   284,   288,   290,   294,   296,   300,   304,   306,   310,   312,
   316,   318,   323,   327,   331,   335,   338,   343,   348,   353,
   358,   362,   364,   368,   370,   372,   376,   378,   380,   384,
   386,   390,   394,   396,   398,   400,   402,   406,   408,   412,
   414,   418,   420,   422,   426,   428,   430,   434,   436,   440,
   442,   446,   448,   450,   454,   456,   458,   462,   466,   468
};
#endif


#if YYDEBUG != 0 || defined (YYERROR_VERBOSE)

static const char * const yytname[] = {   "$","error","$undefined.","MODULE",
"MAINMODULE","EXTERN","READONLY","CHARE","GROUP","NODEGROUP","ARRAY","MESSAGE",
"CLASS","STACKSIZE","THREADED","TEMPLATE","SYNC","EXCLUSIVE","VOID","PACKED",
"VARSIZE","ENTRY","MAINCHARE","IDENT","NUMBER","LITERAL","INT","LONG","SHORT",
"CHAR","FLOAT","DOUBLE","UNSIGNED","';'","'{'","'}'","','","'<'","'>'","'*'",
"'['","']'","'('","')'","':'","'='","File","ModuleEList","OptExtern","OptSemiColon",
"Name","Module","ConstructEList","ConstructList","Construct","TParam","TParamList",
"TParamEList","OptTParams","BuiltinType","NamedType","SimpleType","OnePtrType",
"PtrType","ArrayDim","ArrayType","FuncType","Type","TypeList","Dim","DimList",
"Readonly","ReadonlyMsg","MAttribs","MAttribList","MAttrib","Message","OptBaseList",
"BaseList","Chare","Group","NodeGroup","Array","TChare","TGroup","TNodeGroup",
"TArray","TMessage","OptTypeInit","OptNameInit","TVar","TVarList","TemplateSpec",
"Template","MemberEList","MemberList","Member","Entry","EAttribs","EAttribList",
"EAttrib","OptType","EParam","OptStackSize", NULL
};
#endif

static const short yyr1[] = {     0,
    46,    47,    47,    48,    48,    49,    49,    50,    51,    51,
    52,    52,    53,    53,    54,    54,    54,    54,    54,    54,
    54,    54,    54,    54,    55,    55,    55,    56,    56,    57,
    57,    58,    58,    59,    59,    59,    59,    59,    59,    59,
    59,    59,    59,    59,    59,    59,    60,    61,    61,    62,
    63,    63,    64,    64,    65,    66,    67,    67,    67,    67,
    67,    68,    68,    68,    69,    70,    70,    71,    72,    73,
    73,    74,    74,    75,    75,    76,    77,    77,    78,    78,
    79,    79,    80,    81,    82,    83,    83,    84,    85,    86,
    87,    88,    88,    89,    89,    89,    90,    90,    90,    91,
    91,    92,    93,    93,    93,    93,    93,    94,    94,    95,
    95,    96,    96,    96,    97,    97,    97,    98,    98,    99,
    99,   100,   100,   100,   101,   101,   101,   102,   103,   103
};

static const short yyr2[] = {     0,
     1,     0,     2,     0,     1,     0,     1,     1,     3,     3,
     1,     4,     0,     2,     5,     2,     3,     3,     3,     2,
     2,     2,     2,     2,     1,     1,     1,     1,     3,     0,
     1,     0,     3,     1,     1,     1,     1,     2,     2,     2,
     2,     2,     1,     1,     2,     1,     2,     1,     1,     2,
     2,     2,     1,     1,     4,     8,     1,     1,     1,     1,
     1,     0,     1,     3,     3,     0,     2,     4,     5,     0,
     3,     1,     3,     1,     1,     3,     0,     2,     1,     3,
     4,     4,     4,     4,     4,     4,     4,     4,     4,     4,
     4,     0,     2,     0,     2,     2,     3,     2,     3,     1,
     3,     4,     2,     2,     2,     2,     2,     1,     4,     0,
     2,     2,     2,     2,     6,     6,     4,     0,     3,     1,
     3,     1,     1,     1,     0,     1,     1,     3,     0,     3
};

static const short yydefact[] = {     2,
     0,     0,     1,     2,     8,     0,     0,     3,    11,     4,
     9,    10,     5,     0,     0,     4,     0,     0,     0,     0,
     0,    70,     0,     0,     4,    16,     0,     0,     0,    20,
    21,    22,    23,     0,    24,     6,    14,     0,    46,    34,
    35,    36,    37,    43,    44,     0,    32,    48,    49,    57,
    58,    59,    60,    61,     0,    77,    77,    77,    77,     0,
     0,     0,    77,     0,    17,    18,    19,     0,     0,     0,
     0,    70,     0,   103,   104,   105,   106,   107,     7,    12,
     0,    42,    45,    38,    39,    40,    41,    30,    47,    50,
    51,    52,     0,     0,    66,     0,     0,     0,     0,     0,
    74,    75,     0,    72,    76,     0,    61,     0,   100,     0,
     0,     6,    77,    77,    77,    77,     0,    77,     0,    26,
    27,    28,    31,     0,    25,    53,    54,     0,     0,     0,
    66,    68,    79,    78,   108,   110,    81,    83,    84,    85,
    71,     0,    92,     0,    98,    94,     0,   102,    82,    15,
     0,     0,     0,     0,     0,     0,    69,     0,    33,    55,
     0,     0,    67,     0,   118,     0,     0,     0,   110,     0,
    73,     0,    97,    95,    96,    99,   101,    86,    88,    89,
    90,    91,    87,    29,     0,    65,    80,     0,     0,   113,
   114,     6,   111,   112,    93,    62,   122,   123,   124,     0,
   120,    46,    32,     0,     0,   109,    63,     0,   119,     0,
     0,   125,   117,     0,    62,    56,   121,   129,    46,   127,
     0,   129,    64,     0,   115,   128,   116,     0,   130,     0,
     0,     0
};

static const short yydefgoto[] = {   230,
     3,    14,    80,    47,     4,    11,    15,    16,   122,   123,
   124,    89,    48,    49,    50,    51,    52,   128,    53,    54,
   108,   208,   131,   132,   166,   167,    61,   103,   104,    29,
    97,   134,    30,    31,    32,    33,    74,    75,    76,    77,
    78,   173,   145,   109,   110,    34,    35,   137,   168,   169,
   170,   189,   200,   201,   221,   213,   225
};

static const short yypact[] = {    47,
    12,    12,-32768,    47,-32768,   -17,   -17,-32768,-32768,     7,
-32768,-32768,-32768,    30,    28,     7,    83,    12,    12,    12,
    12,    41,    53,    12,     7,-32768,    71,    75,    84,-32768,
-32768,-32768,-32768,    49,-32768,    91,-32768,   164,-32768,-32768,
    60,-32768,-32768,-32768,-32768,    69,    89,-32768,-32768,    88,
    92,    93,-32768,-32768,     6,    96,    96,    96,    96,    80,
    12,   107,    96,   106,-32768,-32768,-32768,    12,    12,    12,
    12,    41,    12,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
   103,-32768,-32768,-32768,-32768,-32768,-32768,   135,-32768,-32768,
-32768,-32768,    79,   105,   108,    12,    87,    87,    87,    87,
-32768,-32768,   104,   110,-32768,    12,   -23,     6,   111,   113,
    87,    91,    96,    96,    96,    96,    12,    96,    12,-32768,
-32768,   116,-32768,   117,    38,-32768,-32768,   115,    12,    79,
   108,-32768,   132,-32768,-32768,     5,-32768,-32768,-32768,-32768,
-32768,    80,   124,    98,-32768,   125,   107,-32768,-32768,-32768,
    87,    87,    87,    87,   138,    87,-32768,   135,-32768,-32768,
   129,   133,-32768,    12,   109,   140,   142,   141,     5,   145,
-32768,   164,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,   137,-32768,-32768,    11,   179,-32768,
-32768,    91,-32768,-32768,    38,   164,-32768,-32768,-32768,   139,
   147,    12,    24,    88,    12,-32768,    13,   143,-32768,    11,
   156,   194,-32768,   156,   164,-32768,-32768,   171,   146,-32768,
   160,   171,-32768,   154,-32768,-32768,-32768,   161,-32768,   213,
   214,-32768
};

static const short yypgoto[] = {-32768,
   211,-32768,  -103,    -1,   202,   212,    -2,-32768,-32768,    70,
-32768,-32768,-32768,   -14,   -35,  -168,-32768,    97,-32768,   -54,
   -15,     3,-32768,    99,   215,   217,   162,    90,-32768,-32768,
   -39,    72,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,    94,-32768,    86,-32768,-32768,   -68,    66,-32768,
-32768,-32768,    27,-32768,-32768,  -149,    16
};


#define	YYLAST		240


static const short yytable[] = {     6,
     7,    55,    81,    56,    57,    58,    59,   107,   150,    63,
    17,    13,   -94,    37,   -94,     9,    10,    98,    99,   100,
   205,   144,    64,   111,   197,   165,   198,   199,     5,   138,
   139,   140,     1,     2,     5,    17,    18,    19,    20,    21,
    22,   -13,   149,   220,    23,    93,   105,    94,   215,     1,
     2,    24,    93,    95,    94,    68,    69,    70,    71,    72,
    88,   218,    36,    25,   222,   212,   113,   114,   115,   116,
    73,   118,   125,   151,   152,   153,   154,    93,   156,    94,
    60,   133,   178,   179,   180,   181,    82,   183,   206,    62,
    83,   127,   107,    38,    84,    85,    86,    87,   101,   102,
    39,     5,   126,    65,   143,     5,   146,    66,    40,    41,
    42,    43,    44,    45,    46,   155,    67,   157,   106,   135,
   136,   174,   175,    79,    39,    88,    90,   161,   127,     5,
    91,    92,    40,    41,    42,    43,    44,    45,    46,    96,
   112,   119,   125,   129,   141,   142,   147,   130,   188,   133,
   148,   158,    39,   204,   159,   160,   195,     5,   120,   121,
    40,    41,    42,    43,    44,    45,    46,   164,   172,   144,
   182,   185,   190,   186,   191,   192,   204,   194,   196,   209,
   207,    39,   210,   224,   229,   216,     5,   203,  -126,    40,
    41,    42,    43,    44,    45,    46,   202,   212,   228,   207,
   211,     5,   226,   214,    40,    41,    42,    43,    44,    45,
    46,   219,   231,   232,     8,    26,     5,   223,    12,    40,
    41,    42,    43,    44,    45,    46,   162,   184,    27,   163,
    28,   171,   177,   117,   193,   187,   217,   227,     0,   176
};

static const short yycheck[] = {     1,
     2,    17,    38,    18,    19,    20,    21,    62,   112,    24,
     6,     5,    36,    16,    38,    33,    34,    57,    58,    59,
   189,    45,    25,    63,    14,    21,    16,    17,    23,    98,
    99,   100,     3,     4,    23,     6,     7,     8,     9,    10,
    11,    35,   111,   212,    15,    40,    61,    42,    36,     3,
     4,    22,    40,    55,    42,     7,     8,     9,    10,    11,
    37,   211,    35,    34,   214,    42,    68,    69,    70,    71,
    22,    73,    88,   113,   114,   115,   116,    40,   118,    42,
    40,    96,   151,   152,   153,   154,    27,   156,   192,    37,
    31,    93,   147,    11,    26,    27,    28,    29,    19,    20,
    18,    23,    24,    33,   106,    23,   108,    33,    26,    27,
    28,    29,    30,    31,    32,   117,    33,   119,    12,    33,
    34,    24,    25,    33,    18,    37,    39,   129,   130,    23,
    39,    39,    26,    27,    28,    29,    30,    31,    32,    44,
    35,    39,   158,    39,    41,    36,    36,    40,    40,   164,
    38,    36,    18,   189,    38,    41,   172,    23,    24,    25,
    26,    27,    28,    29,    30,    31,    32,    36,    45,    45,
    33,    43,    33,    41,    33,    35,   212,    33,    42,    41,
   196,    18,    36,    13,    24,    43,    23,   189,    43,    26,
    27,    28,    29,    30,    31,    32,    18,    42,    45,   215,
   202,    23,    43,   205,    26,    27,    28,    29,    30,    31,
    32,    18,     0,     0,     4,    14,    23,   215,     7,    26,
    27,    28,    29,    30,    31,    32,   130,   158,    14,   131,
    14,   142,   147,    72,   169,   164,   210,   222,    -1,   146
};
/* -*-C-*-  Note some compilers choke on comments on `#line' lines.  */
#line 3 "/usr/dcs/software/supported/encap/bison-1.27/share/bison.simple"
/* This file comes from bison-1.27.  */

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

#line 216 "/usr/dcs/software/supported/encap/bison-1.27/share/bison.simple"

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
#line 91 "xi-grammar.y"
{ yyval.modlist = yyvsp[0].modlist; modlist = yyvsp[0].modlist; ;
    break;}
case 2:
#line 95 "xi-grammar.y"
{ yyval.modlist = 0; ;
    break;}
case 3:
#line 97 "xi-grammar.y"
{ yyval.modlist = new ModuleList(yyvsp[-1].module, yyvsp[0].modlist); ;
    break;}
case 4:
#line 101 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 5:
#line 103 "xi-grammar.y"
{ yyval.intval = 1; ;
    break;}
case 6:
#line 107 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 7:
#line 109 "xi-grammar.y"
{ yyval.intval = 1; ;
    break;}
case 8:
#line 113 "xi-grammar.y"
{ yyval.strval = yyvsp[0].strval; ;
    break;}
case 9:
#line 117 "xi-grammar.y"
{ yyval.module = new Module(yyvsp[-1].strval, yyvsp[0].conslist); ;
    break;}
case 10:
#line 119 "xi-grammar.y"
{ yyval.module = new Module(yyvsp[-1].strval, yyvsp[0].conslist); yyval.module->setMain(); ;
    break;}
case 11:
#line 123 "xi-grammar.y"
{ yyval.conslist = 0; ;
    break;}
case 12:
#line 125 "xi-grammar.y"
{ yyval.conslist = yyvsp[-2].conslist; ;
    break;}
case 13:
#line 129 "xi-grammar.y"
{ yyval.conslist = 0; ;
    break;}
case 14:
#line 131 "xi-grammar.y"
{ yyval.conslist = new ConstructList(yyvsp[-1].construct, yyvsp[0].conslist); ;
    break;}
case 15:
#line 135 "xi-grammar.y"
{ if(yyvsp[-2].conslist) yyvsp[-2].conslist->setExtern(yyvsp[-4].intval); yyval.construct = yyvsp[-2].conslist; ;
    break;}
case 16:
#line 137 "xi-grammar.y"
{ yyvsp[0].module->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].module; ;
    break;}
case 17:
#line 139 "xi-grammar.y"
{ yyvsp[-1].readonly->setExtern(yyvsp[-2].intval); yyval.construct = yyvsp[-1].readonly; ;
    break;}
case 18:
#line 141 "xi-grammar.y"
{ yyvsp[-1].readonly->setExtern(yyvsp[-2].intval); yyval.construct = yyvsp[-1].readonly; ;
    break;}
case 19:
#line 143 "xi-grammar.y"
{ yyvsp[-1].message->setExtern(yyvsp[-2].intval); yyval.construct = yyvsp[-1].message; ;
    break;}
case 20:
#line 145 "xi-grammar.y"
{ yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; ;
    break;}
case 21:
#line 147 "xi-grammar.y"
{ yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; ;
    break;}
case 22:
#line 149 "xi-grammar.y"
{ yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; ;
    break;}
case 23:
#line 151 "xi-grammar.y"
{ yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; ;
    break;}
case 24:
#line 153 "xi-grammar.y"
{ yyvsp[0].templat->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].templat; ;
    break;}
case 25:
#line 157 "xi-grammar.y"
{ yyval.tparam = new TParamType(yyvsp[0].type); ;
    break;}
case 26:
#line 159 "xi-grammar.y"
{ yyval.tparam = new TParamVal(yyvsp[0].strval); ;
    break;}
case 27:
#line 161 "xi-grammar.y"
{ yyval.tparam = new TParamVal(yyvsp[0].strval); ;
    break;}
case 28:
#line 165 "xi-grammar.y"
{ yyval.tparlist = new TParamList(yyvsp[0].tparam); ;
    break;}
case 29:
#line 167 "xi-grammar.y"
{ yyval.tparlist = new TParamList(yyvsp[-2].tparam, yyvsp[0].tparlist); ;
    break;}
case 30:
#line 171 "xi-grammar.y"
{ yyval.tparlist = 0; ;
    break;}
case 31:
#line 173 "xi-grammar.y"
{ yyval.tparlist = yyvsp[0].tparlist; ;
    break;}
case 32:
#line 177 "xi-grammar.y"
{ yyval.tparlist = 0; ;
    break;}
case 33:
#line 179 "xi-grammar.y"
{ yyval.tparlist = yyvsp[-1].tparlist; ;
    break;}
case 34:
#line 183 "xi-grammar.y"
{ yyval.type = new BuiltinType("int"); ;
    break;}
case 35:
#line 185 "xi-grammar.y"
{ yyval.type = new BuiltinType("long"); ;
    break;}
case 36:
#line 187 "xi-grammar.y"
{ yyval.type = new BuiltinType("short"); ;
    break;}
case 37:
#line 189 "xi-grammar.y"
{ yyval.type = new BuiltinType("char"); ;
    break;}
case 38:
#line 191 "xi-grammar.y"
{ yyval.type = new BuiltinType("unsigned int"); ;
    break;}
case 39:
#line 193 "xi-grammar.y"
{ yyval.type = new BuiltinType("long"); ;
    break;}
case 40:
#line 195 "xi-grammar.y"
{ yyval.type = new BuiltinType("short"); ;
    break;}
case 41:
#line 197 "xi-grammar.y"
{ yyval.type = new BuiltinType("char"); ;
    break;}
case 42:
#line 199 "xi-grammar.y"
{ yyval.type = new BuiltinType("long long"); ;
    break;}
case 43:
#line 201 "xi-grammar.y"
{ yyval.type = new BuiltinType("float"); ;
    break;}
case 44:
#line 203 "xi-grammar.y"
{ yyval.type = new BuiltinType("double"); ;
    break;}
case 45:
#line 205 "xi-grammar.y"
{ yyval.type = new BuiltinType("long double"); ;
    break;}
case 46:
#line 207 "xi-grammar.y"
{ yyval.type = new BuiltinType("void"); ;
    break;}
case 47:
#line 211 "xi-grammar.y"
{ yyval.ntype = new NamedType(yyvsp[-1].strval, yyvsp[0].tparlist); ;
    break;}
case 48:
#line 215 "xi-grammar.y"
{ yyval.type = yyvsp[0].type; ;
    break;}
case 49:
#line 217 "xi-grammar.y"
{ yyval.type = yyvsp[0].ntype; ;
    break;}
case 50:
#line 221 "xi-grammar.y"
{ yyval.ptype = new PtrType(yyvsp[-1].type); ;
    break;}
case 51:
#line 225 "xi-grammar.y"
{ yyvsp[-1].ptype->indirect(); yyval.ptype = yyvsp[-1].ptype; ;
    break;}
case 52:
#line 227 "xi-grammar.y"
{ yyvsp[-1].ptype->indirect(); yyval.ptype = yyvsp[-1].ptype; ;
    break;}
case 53:
#line 231 "xi-grammar.y"
{ yyval.val = new Value(yyvsp[0].strval); ;
    break;}
case 54:
#line 233 "xi-grammar.y"
{ yyval.val = new Value(yyvsp[0].strval); ;
    break;}
case 55:
#line 237 "xi-grammar.y"
{ yyval.type = new ArrayType(yyvsp[-3].type, yyvsp[-1].val); ;
    break;}
case 56:
#line 241 "xi-grammar.y"
{ yyval.ftype = new FuncType(yyvsp[-7].type, yyvsp[-4].strval, yyvsp[-1].typelist); ;
    break;}
case 57:
#line 245 "xi-grammar.y"
{ yyval.type = yyvsp[0].type; ;
    break;}
case 58:
#line 247 "xi-grammar.y"
{ yyval.type = (Type*) yyvsp[0].ptype; ;
    break;}
case 59:
#line 249 "xi-grammar.y"
{ yyval.type = (Type*) yyvsp[0].ptype; ;
    break;}
case 60:
#line 251 "xi-grammar.y"
{ yyval.type = yyvsp[0].type; ;
    break;}
case 61:
#line 253 "xi-grammar.y"
{ yyval.type = yyvsp[0].ftype; ;
    break;}
case 62:
#line 257 "xi-grammar.y"
{ yyval.typelist = 0; ;
    break;}
case 63:
#line 259 "xi-grammar.y"
{ yyval.typelist = new TypeList(yyvsp[0].type); ;
    break;}
case 64:
#line 261 "xi-grammar.y"
{ yyval.typelist = new TypeList(yyvsp[-2].type, yyvsp[0].typelist); ;
    break;}
case 65:
#line 265 "xi-grammar.y"
{ yyval.val = yyvsp[-1].val; ;
    break;}
case 66:
#line 269 "xi-grammar.y"
{ yyval.vallist = 0; ;
    break;}
case 67:
#line 271 "xi-grammar.y"
{ yyval.vallist = new ValueList(yyvsp[-1].val, yyvsp[0].vallist); ;
    break;}
case 68:
#line 275 "xi-grammar.y"
{ yyval.readonly = new Readonly(yyvsp[-2].type, yyvsp[-1].strval, yyvsp[0].vallist); ;
    break;}
case 69:
#line 279 "xi-grammar.y"
{ yyval.readonly = new Readonly(yyvsp[-2].type, yyvsp[0].strval, 0, 1); ;
    break;}
case 70:
#line 283 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 71:
#line 285 "xi-grammar.y"
{ yyval.intval = yyvsp[-1].intval; ;
    break;}
case 72:
#line 289 "xi-grammar.y"
{ yyval.intval = yyvsp[0].intval; ;
    break;}
case 73:
#line 291 "xi-grammar.y"
{ yyval.intval = yyvsp[-2].intval | yyvsp[0].intval; ;
    break;}
case 74:
#line 295 "xi-grammar.y"
{ yyval.intval = SPACKED; ;
    break;}
case 75:
#line 297 "xi-grammar.y"
{ yyval.intval = SVARSIZE; ;
    break;}
case 76:
#line 301 "xi-grammar.y"
{ yyval.message = new Message(yyvsp[0].ntype, yyvsp[-1].intval); ;
    break;}
case 77:
#line 305 "xi-grammar.y"
{ yyval.typelist = 0; ;
    break;}
case 78:
#line 307 "xi-grammar.y"
{ yyval.typelist = yyvsp[0].typelist; ;
    break;}
case 79:
#line 311 "xi-grammar.y"
{ yyval.typelist = new TypeList(yyvsp[0].ntype); ;
    break;}
case 80:
#line 313 "xi-grammar.y"
{ yyval.typelist = new TypeList(yyvsp[-2].ntype, yyvsp[0].typelist); ;
    break;}
case 81:
#line 317 "xi-grammar.y"
{ yyval.chare = new Chare(SCHARE, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); if(yyvsp[0].mbrlist) yyvsp[0].mbrlist->setChare(yyval.chare);;
    break;}
case 82:
#line 319 "xi-grammar.y"
{ yyval.chare = new Chare(SMAINCHARE, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); 
                  if(yyvsp[0].mbrlist) yyvsp[0].mbrlist->setChare(yyval.chare);;
    break;}
case 83:
#line 324 "xi-grammar.y"
{ yyval.chare = new Chare(SGROUP, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); if(yyvsp[0].mbrlist) yyvsp[0].mbrlist->setChare(yyval.chare);;
    break;}
case 84:
#line 328 "xi-grammar.y"
{ yyval.chare = new Chare(SNODEGROUP, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); if(yyvsp[0].mbrlist) yyvsp[0].mbrlist->setChare(yyval.chare);;
    break;}
case 85:
#line 332 "xi-grammar.y"
{ yyval.chare = new Chare(SARRAY, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); if(yyvsp[0].mbrlist) yyvsp[0].mbrlist->setChare(yyval.chare);;
    break;}
case 86:
#line 336 "xi-grammar.y"
{ yyval.chare = new Chare(SCHARE, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); 
                  if(yyvsp[0].mbrlist) yyvsp[0].mbrlist->setChare(yyval.chare);;
    break;}
case 87:
#line 339 "xi-grammar.y"
{ yyval.chare = new Chare(SMAINCHARE, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); 
                  if(yyvsp[0].mbrlist) yyvsp[0].mbrlist->setChare(yyval.chare);;
    break;}
case 88:
#line 344 "xi-grammar.y"
{ yyval.chare = new Chare(SGROUP, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); 
                  if(yyvsp[0].mbrlist) yyvsp[0].mbrlist->setChare(yyval.chare);;
    break;}
case 89:
#line 349 "xi-grammar.y"
{ yyval.chare = new Chare(SNODEGROUP, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); 
                  if(yyvsp[0].mbrlist) yyvsp[0].mbrlist->setChare(yyval.chare);;
    break;}
case 90:
#line 354 "xi-grammar.y"
{ yyval.chare = new Chare(SARRAY, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); 
                  if(yyvsp[0].mbrlist) yyvsp[0].mbrlist->setChare(yyval.chare);;
    break;}
case 91:
#line 359 "xi-grammar.y"
{ yyval.message = new Message(new NamedType(yyvsp[-1].strval), yyvsp[-2].intval); ;
    break;}
case 92:
#line 363 "xi-grammar.y"
{ yyval.type = 0; ;
    break;}
case 93:
#line 365 "xi-grammar.y"
{ yyval.type = yyvsp[0].type; ;
    break;}
case 94:
#line 369 "xi-grammar.y"
{ yyval.strval = 0; ;
    break;}
case 95:
#line 371 "xi-grammar.y"
{ yyval.strval = yyvsp[0].strval; ;
    break;}
case 96:
#line 373 "xi-grammar.y"
{ yyval.strval = yyvsp[0].strval; ;
    break;}
case 97:
#line 377 "xi-grammar.y"
{ yyval.tvar = new TType(new NamedType(yyvsp[-1].strval), yyvsp[0].type); ;
    break;}
case 98:
#line 379 "xi-grammar.y"
{ yyval.tvar = new TFunc(yyvsp[-1].ftype, yyvsp[0].strval); ;
    break;}
case 99:
#line 381 "xi-grammar.y"
{ yyval.tvar = new TName(yyvsp[-2].type, yyvsp[-1].strval, yyvsp[0].strval); ;
    break;}
case 100:
#line 385 "xi-grammar.y"
{ yyval.tvarlist = new TVarList(yyvsp[0].tvar); ;
    break;}
case 101:
#line 387 "xi-grammar.y"
{ yyval.tvarlist = new TVarList(yyvsp[-2].tvar, yyvsp[0].tvarlist); ;
    break;}
case 102:
#line 391 "xi-grammar.y"
{ yyval.tvarlist = yyvsp[-1].tvarlist; ;
    break;}
case 103:
#line 395 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); ;
    break;}
case 104:
#line 397 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); ;
    break;}
case 105:
#line 399 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); ;
    break;}
case 106:
#line 401 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); ;
    break;}
case 107:
#line 403 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].message); yyvsp[0].message->setTemplate(yyval.templat); ;
    break;}
case 108:
#line 407 "xi-grammar.y"
{ yyval.mbrlist = 0; ;
    break;}
case 109:
#line 409 "xi-grammar.y"
{ yyval.mbrlist = yyvsp[-2].mbrlist; ;
    break;}
case 110:
#line 413 "xi-grammar.y"
{ yyval.mbrlist = 0; ;
    break;}
case 111:
#line 415 "xi-grammar.y"
{ yyval.mbrlist = new MemberList(yyvsp[-1].member, yyvsp[0].mbrlist); ;
    break;}
case 112:
#line 419 "xi-grammar.y"
{ yyval.member = yyvsp[-1].entry; ;
    break;}
case 113:
#line 421 "xi-grammar.y"
{ yyval.member = yyvsp[-1].readonly; ;
    break;}
case 114:
#line 423 "xi-grammar.y"
{ yyval.member = yyvsp[-1].readonly; ;
    break;}
case 115:
#line 427 "xi-grammar.y"
{ yyval.entry = new Entry(yyvsp[-4].intval, new BuiltinType("void"), yyvsp[-2].strval, yyvsp[-1].rtype, yyvsp[0].val); ;
    break;}
case 116:
#line 429 "xi-grammar.y"
{ yyval.entry = new Entry(yyvsp[-4].intval, yyvsp[-3].ptype, yyvsp[-2].strval, yyvsp[-1].rtype, yyvsp[0].val); ;
    break;}
case 117:
#line 431 "xi-grammar.y"
{ yyval.entry = new Entry(yyvsp[-2].intval, 0, yyvsp[-1].strval, yyvsp[0].rtype, 0); ;
    break;}
case 118:
#line 435 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 119:
#line 437 "xi-grammar.y"
{ yyval.intval = yyvsp[-1].intval; ;
    break;}
case 120:
#line 441 "xi-grammar.y"
{ yyval.intval = yyvsp[0].intval; ;
    break;}
case 121:
#line 443 "xi-grammar.y"
{ yyval.intval = yyvsp[-2].intval | yyvsp[0].intval; ;
    break;}
case 122:
#line 447 "xi-grammar.y"
{ yyval.intval = STHREADED; ;
    break;}
case 123:
#line 449 "xi-grammar.y"
{ yyval.intval = SSYNC; ;
    break;}
case 124:
#line 451 "xi-grammar.y"
{ yyval.intval = SLOCKED; ;
    break;}
case 125:
#line 455 "xi-grammar.y"
{ yyval.rtype = 0; ;
    break;}
case 126:
#line 457 "xi-grammar.y"
{ yyval.rtype = new BuiltinType("void"); ;
    break;}
case 127:
#line 459 "xi-grammar.y"
{ yyval.rtype = yyvsp[0].ptype; ;
    break;}
case 128:
#line 463 "xi-grammar.y"
{ yyval.rtype = yyvsp[-1].rtype; ;
    break;}
case 129:
#line 467 "xi-grammar.y"
{ yyval.val = 0; ;
    break;}
case 130:
#line 469 "xi-grammar.y"
{ yyval.val = new Value(yyvsp[0].strval); ;
    break;}
}
   /* the action file gets copied in in place of this dollarsign */
#line 542 "/usr/dcs/software/supported/encap/bison-1.27/share/bison.simple"

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
#line 471 "xi-grammar.y"

void yyerror(char *mesg)
{
  cout << "Syntax error at line " << lineno << ": " << mesg << endl;
  // return 0;
}
