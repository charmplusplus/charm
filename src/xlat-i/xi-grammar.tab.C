
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
#define	ENTRY	277
#define	MAINCHARE	278
#define	IDENT	279
#define	NUMBER	280
#define	LITERAL	281
#define	INT	282
#define	LONG	283
#define	SHORT	284
#define	CHAR	285
#define	FLOAT	286
#define	DOUBLE	287
#define	UNSIGNED	288

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
  char *strval;
  int intval;
} YYSTYPE;
#include <stdio.h>

#ifndef __cplusplus
#ifndef __STDC__
#define const
#endif
#endif



#define	YYFINAL		267
#define	YYFLAG		-32768
#define	YYNTBASE	48

#define YYTRANSLATE(x) ((unsigned)(x) <= 288 ? yytranslate[x] : 112)

static const char yytranslate[] = {     0,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,    45,
    46,    42,     2,    39,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,    36,    35,    40,
    47,    41,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
    43,     2,    44,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,    37,     2,    38,     2,     2,     2,     2,     2,
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
    27,    28,    29,    30,    31,    32,    33,    34
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
   192,   198,   199,   203,   205,   209,   211,   213,   214,   218,
   220,   224,   226,   230,   237,   238,   241,   243,   247,   253,
   259,   265,   271,   276,   280,   286,   292,   298,   304,   310,
   316,   321,   322,   325,   326,   329,   332,   336,   339,   343,
   345,   349,   354,   357,   360,   363,   366,   369,   371,   376,
   377,   380,   383,   386,   389,   397,   405,   410,   411,   415,
   417,   421,   423,   425,   427,   429,   430,   432,   434,   438,
   439,   443,   444
};

static const short yyrhs[] = {    49,
     0,     0,    54,    49,     0,     0,     5,     0,     0,    35,
     0,    25,     0,    25,     0,    53,    36,    36,    25,     0,
     3,    52,    55,     0,     4,    52,    55,     0,    35,     0,
    37,    56,    38,    51,     0,     0,    57,    56,     0,    50,
    37,    56,    38,    51,     0,    50,    54,     0,    50,    74,
    35,     0,    50,    75,    35,     0,    50,    82,    35,     0,
    50,    85,     0,    50,    86,     0,    50,    87,     0,    50,
    89,     0,    50,   100,     0,    70,     0,    26,     0,    27,
     0,    58,     0,    58,    39,    59,     0,     0,    59,     0,
     0,    40,    60,    41,     0,    28,     0,    29,     0,    30,
     0,    31,     0,    34,    28,     0,    34,    29,     0,    34,
    30,     0,    34,    31,     0,    29,    29,     0,    32,     0,
    33,     0,    29,    33,     0,    20,     0,    52,    61,     0,
    62,     0,    63,     0,    64,    42,     0,    65,    42,     0,
    66,    42,     0,    26,     0,    52,     0,    70,    43,    67,
    44,     0,    70,    45,    42,    52,    46,    45,    71,    46,
     0,    64,     0,    65,     0,    66,     0,    68,     0,    69,
     0,     0,    70,     0,    70,    39,    71,     0,    43,    67,
    44,     0,     0,    72,    73,     0,     6,    70,    53,    73,
     0,     6,    11,    64,    42,    52,     0,     0,    43,    77,
    44,     0,    78,     0,    78,    39,    77,     0,    21,     0,
    22,     0,     0,    43,    80,    44,     0,    81,     0,    81,
    39,    80,     0,    15,     0,    11,    76,    63,     0,    11,
    76,    63,    37,    71,    38,     0,     0,    36,    84,     0,
    63,     0,    63,    39,    84,     0,     7,    79,    63,    83,
   101,     0,    24,    79,    63,    83,   101,     0,     8,    79,
    63,    83,   101,     0,     9,    79,    63,    83,   101,     0,
    43,    26,    52,    44,     0,    43,    52,    44,     0,    10,
    88,    63,    83,   101,     0,     7,    79,    52,    83,   101,
     0,    24,    79,    52,    83,   101,     0,     8,    79,    52,
    83,   101,     0,     9,    79,    52,    83,   101,     0,    10,
    88,    52,    83,   101,     0,    11,    76,    52,    35,     0,
     0,    47,    70,     0,     0,    47,    26,     0,    47,    27,
     0,    12,    52,    95,     0,    69,    96,     0,    70,    52,
    96,     0,    97,     0,    97,    39,    98,     0,    16,    40,
    98,    41,     0,    99,    90,     0,    99,    91,     0,    99,
    92,     0,    99,    93,     0,    99,    94,     0,    35,     0,
    37,   102,    38,    51,     0,     0,   103,   102,     0,   104,
    35,     0,    74,    35,     0,    75,    35,     0,    23,   105,
    20,    52,   109,   111,   110,     0,    23,   105,    65,    52,
   109,   111,   110,     0,    23,   105,    52,   109,     0,     0,
    43,   106,    44,     0,   107,     0,   107,    39,   106,     0,
    14,     0,    17,     0,    18,     0,    19,     0,     0,    20,
     0,    65,     0,    45,   108,    46,     0,     0,    13,    47,
    26,     0,     0,    47,    26,     0
};

#endif

#if YYDEBUG != 0
static const short yyrline[] = { 0,
    92,    96,    98,   102,   104,   108,   110,   114,   118,   120,
   128,   130,   134,   136,   140,   142,   146,   148,   150,   152,
   154,   156,   158,   160,   162,   164,   168,   170,   172,   176,
   178,   182,   184,   188,   190,   194,   196,   198,   200,   202,
   204,   206,   208,   210,   212,   214,   216,   218,   222,   226,
   228,   232,   236,   238,   242,   244,   248,   252,   256,   258,
   260,   262,   264,   268,   270,   272,   276,   280,   282,   286,
   290,   294,   296,   300,   302,   306,   308,   312,   314,   318,
   320,   324,   328,   330,   334,   336,   340,   342,   346,   348,
   352,   356,   360,   366,   370,   374,   376,   380,   384,   388,
   392,   396,   398,   402,   404,   406,   410,   412,   414,   418,
   420,   424,   428,   430,   432,   434,   436,   440,   442,   446,
   448,   452,   454,   456,   460,   462,   464,   468,   470,   474,
   476,   480,   482,   484,   486,   490,   492,   494,   498,   502,
   504,   508,   510
};
#endif


#if YYDEBUG != 0 || defined (YYERROR_VERBOSE)

static const char * const yytname[] = {   "$","error","$undefined.","MODULE",
"MAINMODULE","EXTERN","READONLY","CHARE","GROUP","NODEGROUP","ARRAY","MESSAGE",
"CLASS","STACKSIZE","THREADED","MIGRATABLE","TEMPLATE","SYNC","EXCLUSIVE","VIRTUAL",
"VOID","PACKED","VARSIZE","ENTRY","MAINCHARE","IDENT","NUMBER","LITERAL","INT",
"LONG","SHORT","CHAR","FLOAT","DOUBLE","UNSIGNED","';'","':'","'{'","'}'","','",
"'<'","'>'","'*'","'['","']'","'('","')'","'='","File","ModuleEList","OptExtern",
"OptSemiColon","Name","QualName","Module","ConstructEList","ConstructList","Construct",
"TParam","TParamList","TParamEList","OptTParams","BuiltinType","NamedType","SimpleType",
"OnePtrType","PtrType","ArrayDim","ArrayType","FuncType","Type","TypeList","Dim",
"DimList","Readonly","ReadonlyMsg","MAttribs","MAttribList","MAttrib","CAttribs",
"CAttribList","CAttrib","Message","OptBaseList","BaseList","Chare","Group","NodeGroup",
"ArrayIndexType","Array","TChare","TGroup","TNodeGroup","TArray","TMessage",
"OptTypeInit","OptNameInit","TVar","TVarList","TemplateSpec","Template","MemberEList",
"MemberList","Member","Entry","EAttribs","EAttribList","EAttrib","OptType","EParam",
"OptStackSize","OptPure", NULL
};
#endif

static const short yyr1[] = {     0,
    48,    49,    49,    50,    50,    51,    51,    52,    53,    53,
    54,    54,    55,    55,    56,    56,    57,    57,    57,    57,
    57,    57,    57,    57,    57,    57,    58,    58,    58,    59,
    59,    60,    60,    61,    61,    62,    62,    62,    62,    62,
    62,    62,    62,    62,    62,    62,    62,    62,    63,    64,
    64,    65,    66,    66,    67,    67,    68,    69,    70,    70,
    70,    70,    70,    71,    71,    71,    72,    73,    73,    74,
    75,    76,    76,    77,    77,    78,    78,    79,    79,    80,
    80,    81,    82,    82,    83,    83,    84,    84,    85,    85,
    86,    87,    88,    88,    89,    90,    90,    91,    92,    93,
    94,    95,    95,    96,    96,    96,    97,    97,    97,    98,
    98,    99,   100,   100,   100,   100,   100,   101,   101,   102,
   102,   103,   103,   103,   104,   104,   104,   105,   105,   106,
   106,   107,   107,   107,   107,   108,   108,   108,   109,   110,
   110,   111,   111
};

static const short yyr2[] = {     0,
     1,     0,     2,     0,     1,     0,     1,     1,     1,     4,
     3,     3,     1,     4,     0,     2,     5,     2,     3,     3,
     3,     2,     2,     2,     2,     2,     1,     1,     1,     1,
     3,     0,     1,     0,     3,     1,     1,     1,     1,     2,
     2,     2,     2,     2,     1,     1,     2,     1,     2,     1,
     1,     2,     2,     2,     1,     1,     4,     8,     1,     1,
     1,     1,     1,     0,     1,     3,     3,     0,     2,     4,
     5,     0,     3,     1,     3,     1,     1,     0,     3,     1,
     3,     1,     3,     6,     0,     2,     1,     3,     5,     5,
     5,     5,     4,     3,     5,     5,     5,     5,     5,     5,
     4,     0,     2,     0,     2,     2,     3,     2,     3,     1,
     3,     4,     2,     2,     2,     2,     2,     1,     4,     0,
     2,     2,     2,     2,     7,     7,     4,     0,     3,     1,
     3,     1,     1,     1,     1,     0,     1,     1,     3,     0,
     3,     0,     2
};

static const short yydefact[] = {     2,
     0,     0,     1,     2,     8,     0,     0,     3,    13,     4,
    11,    12,     5,     0,     0,     4,     0,    78,    78,    78,
     0,    72,     0,    78,     4,    18,     0,     0,     0,    22,
    23,    24,    25,     0,    26,     6,    16,     0,    48,    36,
    37,    38,    39,    45,    46,     0,    34,    50,    51,    59,
    60,    61,    62,    63,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,    19,    20,    21,    78,
    78,    78,     0,    72,    78,   113,   114,   115,   116,   117,
     7,    14,     0,    44,    47,    40,    41,    42,    43,    32,
    49,    52,    53,    54,     9,     0,     0,    68,    82,     0,
    80,    85,    85,    85,     0,     0,    85,    76,    77,     0,
    74,    83,     0,    63,     0,   110,     0,    85,     6,     0,
     0,     0,     0,     0,     0,     0,    28,    29,    30,    33,
     0,    27,    55,    56,     0,     0,     0,     0,    68,    70,
    79,     0,     0,     0,     0,     0,     0,    94,     0,    73,
     0,    64,   102,     0,   108,   104,     0,   112,     0,    17,
    85,    85,    85,    85,     0,    85,    71,     0,    35,    57,
     0,     0,     0,    69,    81,    87,    86,   118,   120,    89,
    91,    92,    93,    95,    75,    65,     0,     0,   107,   105,
   106,   109,   111,    90,     0,     0,     0,     0,   101,     0,
    31,     0,    10,    67,     0,   128,     0,     0,     0,   120,
     0,    64,    84,   103,    96,    98,    99,   100,    97,    64,
    88,     0,     0,   123,   124,     6,   121,   122,    66,     0,
   132,   133,   134,   135,     0,   130,    48,    34,     0,     0,
   119,    58,   129,     0,     0,   136,   127,     0,   131,   142,
    48,   138,     0,   142,     0,   140,   139,   140,   143,     0,
   125,   126,     0,   141,     0,     0,     0
};

static const short yydefgoto[] = {   265,
     3,    14,    82,    47,    98,     4,    11,    15,    16,   129,
   130,   131,    91,    48,    49,    50,    51,    52,   135,    53,
    54,   186,   187,   139,   140,   207,   208,    63,   110,   111,
    57,   100,   101,    29,   144,   177,    30,    31,    32,    61,
    33,    76,    77,    78,    79,    80,   189,   155,   116,   117,
    34,    35,   180,   209,   210,   211,   223,   235,   236,   253,
   247,   261,   256
};

static const short yypact[] = {    44,
    40,    40,-32768,    44,-32768,    64,    64,-32768,-32768,    14,
-32768,-32768,-32768,    33,    65,    14,    77,   -19,   -19,   -19,
    70,    74,    86,   -19,    14,-32768,    99,   118,   125,-32768,
-32768,-32768,-32768,   148,-32768,   126,-32768,   165,-32768,-32768,
    -7,-32768,-32768,-32768,-32768,    32,   100,-32768,-32768,   109,
   120,   121,-32768,-32768,   -12,   127,    40,    40,    40,   105,
    40,    72,    40,   116,    40,   128,-32768,-32768,-32768,   -19,
   -19,   -19,    70,    74,   -19,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,   122,-32768,-32768,-32768,-32768,-32768,-32768,   150,
-32768,-32768,-32768,-32768,-32768,   107,   123,    48,-32768,   124,
   130,   131,   131,   131,    40,   129,   131,-32768,-32768,   142,
   132,   151,    40,    17,    10,   152,   159,   131,   126,    40,
    40,    40,    40,    40,    40,    40,-32768,-32768,   153,-32768,
   160,   -20,-32768,-32768,   145,    40,   166,   107,   161,-32768,
-32768,   127,    40,    92,    92,    92,   162,-32768,    92,-32768,
    72,   165,   158,   112,-32768,   172,   116,-32768,    92,-32768,
   131,   131,   131,   131,   174,   131,-32768,   150,-32768,-32768,
   175,   182,   176,-32768,-32768,   185,-32768,-32768,    15,-32768,
-32768,-32768,-32768,-32768,-32768,    53,   187,   165,-32768,-32768,
-32768,-32768,-32768,-32768,    92,    92,    92,    92,-32768,    92,
-32768,   188,-32768,-32768,    40,   191,   200,   202,   203,    15,
   205,   165,-32768,   -20,-32768,-32768,-32768,-32768,-32768,   165,
-32768,    68,   183,-32768,-32768,   126,-32768,-32768,-32768,   192,
-32768,-32768,-32768,-32768,   199,   206,    40,     6,   109,    40,
-32768,-32768,-32768,    68,   197,   198,-32768,   197,-32768,   201,
   204,-32768,   207,   201,   218,   233,-32768,   233,-32768,   208,
-32768,-32768,   221,-32768,   249,   251,-32768
};

static const short yypgoto[] = {-32768,
   248,-32768,  -110,    -1,-32768,   240,   250,    29,-32768,-32768,
    88,-32768,-32768,-32768,   -53,   -36,  -212,-32768,   133,-32768,
   -57,   -14,  -131,-32768,   119,   245,   246,   189,   110,-32768,
     8,   134,-32768,-32768,   -89,    57,-32768,-32768,-32768,   193,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,   108,-32768,   111,
-32768,-32768,  -129,    55,-32768,-32768,-32768,    23,-32768,-32768,
  -130,    11,    16
};


#define	YYLAST		276


static const short yytable[] = {     6,
     7,    83,    55,   102,   103,   104,   114,   107,   160,   112,
   240,   118,    95,   145,   146,   181,   182,   149,    13,   184,
    17,    84,    96,    56,    97,    85,    58,    59,   159,   194,
    96,    65,    97,   252,     5,     1,     2,   206,    17,    18,
    19,    20,    21,    22,    37,    90,     1,     2,    23,   115,
   246,   -15,    96,    66,    97,  -104,    24,  -104,   106,    86,
    87,    88,    89,   154,     5,   215,   216,   217,   218,    25,
   219,   195,   196,   197,   198,   132,   200,   120,   121,   122,
   229,   231,   125,   137,   232,   233,   234,    38,   230,   176,
   138,   212,   108,   109,   134,    96,    39,    97,     9,   114,
    10,     5,    36,   147,    40,    41,    42,    43,    44,    45,
    46,   153,    60,   156,   250,   241,    62,   254,   161,   162,
   163,   164,   165,   166,   167,    64,   178,   113,   179,     5,
   105,     5,   133,    67,   171,    39,   134,   190,   191,    90,
     5,    99,   115,    40,    41,    42,    43,    44,    45,    46,
    92,   176,    68,   132,    70,    71,    72,    73,    74,    69,
    81,    93,    94,   126,   136,   119,   143,   141,   142,    39,
   151,    75,   148,   214,     5,   127,   128,    40,    41,    42,
    43,    44,    45,    46,    39,   150,   239,   152,   170,     5,
   157,   168,    40,    41,    42,    43,    44,    45,    46,   158,
   169,   172,   237,   138,   188,   183,   203,     5,   199,   239,
    40,    41,    42,    43,    44,    45,    46,   251,   154,   204,
   202,   238,     5,   205,   213,    40,    41,    42,    43,    44,
    45,    46,   220,   222,   224,   245,   225,   242,   248,   228,
   226,   246,   243,   259,   244,   260,   264,   255,   266,  -137,
   267,     8,   257,    26,   263,   201,    12,   174,    27,    28,
   185,   221,   124,   192,   227,   123,   249,   193,   262,   258,
   173,     0,     0,     0,     0,   175
};

static const short yycheck[] = {     1,
     2,    38,    17,    57,    58,    59,    64,    61,   119,    63,
   223,    65,    25,   103,   104,   145,   146,   107,     5,   149,
     6,    29,    43,    43,    45,    33,    19,    20,   118,   159,
    43,    24,    45,   246,    25,     3,     4,    23,     6,     7,
     8,     9,    10,    11,    16,    40,     3,     4,    16,    64,
    45,    38,    43,    25,    45,    39,    24,    41,    60,    28,
    29,    30,    31,    47,    25,   195,   196,   197,   198,    37,
   200,   161,   162,   163,   164,    90,   166,    70,    71,    72,
   212,    14,    75,    36,    17,    18,    19,    11,   220,   143,
    43,    39,    21,    22,    96,    43,    20,    45,    35,   157,
    37,    25,    38,   105,    28,    29,    30,    31,    32,    33,
    34,   113,    43,   115,   245,   226,    43,   248,   120,   121,
   122,   123,   124,   125,   126,    40,    35,    12,    37,    25,
    26,    25,    26,    35,   136,    20,   138,    26,    27,    40,
    25,    15,   157,    28,    29,    30,    31,    32,    33,    34,
    42,   205,    35,   168,     7,     8,     9,    10,    11,    35,
    35,    42,    42,    42,    42,    38,    36,    44,    39,    20,
    39,    24,    44,   188,    25,    26,    27,    28,    29,    30,
    31,    32,    33,    34,    20,    44,   223,    37,    44,    25,
    39,    39,    28,    29,    30,    31,    32,    33,    34,    41,
    41,    36,    20,    43,    47,    44,    25,    25,    35,   246,
    28,    29,    30,    31,    32,    33,    34,    20,    47,    44,
    46,   223,    25,    39,    38,    28,    29,    30,    31,    32,
    33,    34,    45,    43,    35,   237,    35,    46,   240,    35,
    38,    45,    44,    26,    39,    13,    26,    47,     0,    46,
     0,     4,    46,    14,    47,   168,     7,   139,    14,    14,
   151,   205,    74,   156,   210,    73,   244,   157,   258,   254,
   138,    -1,    -1,    -1,    -1,   142
};
/* -*-C-*-  Note some compilers choke on comments on `#line' lines.  */
#line 3 "/home/csar2/bhandark/share/bison.simple"
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

#line 217 "/home/csar2/bhandark/share/bison.simple"

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
#line 93 "xi-grammar.y"
{ yyval.modlist = yyvsp[0].modlist; modlist = yyvsp[0].modlist; ;
    break;}
case 2:
#line 97 "xi-grammar.y"
{ yyval.modlist = 0; ;
    break;}
case 3:
#line 99 "xi-grammar.y"
{ yyval.modlist = new ModuleList(lineno, yyvsp[-1].module, yyvsp[0].modlist); ;
    break;}
case 4:
#line 103 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 5:
#line 105 "xi-grammar.y"
{ yyval.intval = 1; ;
    break;}
case 6:
#line 109 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 7:
#line 111 "xi-grammar.y"
{ yyval.intval = 1; ;
    break;}
case 8:
#line 115 "xi-grammar.y"
{ yyval.strval = yyvsp[0].strval; ;
    break;}
case 9:
#line 119 "xi-grammar.y"
{ yyval.strval = yyvsp[0].strval; ;
    break;}
case 10:
#line 121 "xi-grammar.y"
{
		  char *tmp = new char[strlen(yyvsp[-3].strval)+strlen(yyvsp[0].strval)+3];
		  sprintf(tmp,"%s::%s", yyvsp[-3].strval, yyvsp[0].strval);
		  yyval.strval = tmp;
		;
    break;}
case 11:
#line 129 "xi-grammar.y"
{ yyval.module = new Module(lineno, yyvsp[-1].strval, yyvsp[0].conslist); ;
    break;}
case 12:
#line 131 "xi-grammar.y"
{ yyval.module = new Module(lineno, yyvsp[-1].strval, yyvsp[0].conslist); yyval.module->setMain(); ;
    break;}
case 13:
#line 135 "xi-grammar.y"
{ yyval.conslist = 0; ;
    break;}
case 14:
#line 137 "xi-grammar.y"
{ yyval.conslist = yyvsp[-2].conslist; ;
    break;}
case 15:
#line 141 "xi-grammar.y"
{ yyval.conslist = 0; ;
    break;}
case 16:
#line 143 "xi-grammar.y"
{ yyval.conslist = new ConstructList(lineno, yyvsp[-1].construct, yyvsp[0].conslist); ;
    break;}
case 17:
#line 147 "xi-grammar.y"
{ if(yyvsp[-2].conslist) yyvsp[-2].conslist->setExtern(yyvsp[-4].intval); yyval.construct = yyvsp[-2].conslist; ;
    break;}
case 18:
#line 149 "xi-grammar.y"
{ yyvsp[0].module->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].module; ;
    break;}
case 19:
#line 151 "xi-grammar.y"
{ yyvsp[-1].readonly->setExtern(yyvsp[-2].intval); yyval.construct = yyvsp[-1].readonly; ;
    break;}
case 20:
#line 153 "xi-grammar.y"
{ yyvsp[-1].readonly->setExtern(yyvsp[-2].intval); yyval.construct = yyvsp[-1].readonly; ;
    break;}
case 21:
#line 155 "xi-grammar.y"
{ yyvsp[-1].message->setExtern(yyvsp[-2].intval); yyval.construct = yyvsp[-1].message; ;
    break;}
case 22:
#line 157 "xi-grammar.y"
{ yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; ;
    break;}
case 23:
#line 159 "xi-grammar.y"
{ yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; ;
    break;}
case 24:
#line 161 "xi-grammar.y"
{ yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; ;
    break;}
case 25:
#line 163 "xi-grammar.y"
{ yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; ;
    break;}
case 26:
#line 165 "xi-grammar.y"
{ yyvsp[0].templat->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].templat; ;
    break;}
case 27:
#line 169 "xi-grammar.y"
{ yyval.tparam = new TParamType(yyvsp[0].type); ;
    break;}
case 28:
#line 171 "xi-grammar.y"
{ yyval.tparam = new TParamVal(yyvsp[0].strval); ;
    break;}
case 29:
#line 173 "xi-grammar.y"
{ yyval.tparam = new TParamVal(yyvsp[0].strval); ;
    break;}
case 30:
#line 177 "xi-grammar.y"
{ yyval.tparlist = new TParamList(yyvsp[0].tparam); ;
    break;}
case 31:
#line 179 "xi-grammar.y"
{ yyval.tparlist = new TParamList(yyvsp[-2].tparam, yyvsp[0].tparlist); ;
    break;}
case 32:
#line 183 "xi-grammar.y"
{ yyval.tparlist = 0; ;
    break;}
case 33:
#line 185 "xi-grammar.y"
{ yyval.tparlist = yyvsp[0].tparlist; ;
    break;}
case 34:
#line 189 "xi-grammar.y"
{ yyval.tparlist = 0; ;
    break;}
case 35:
#line 191 "xi-grammar.y"
{ yyval.tparlist = yyvsp[-1].tparlist; ;
    break;}
case 36:
#line 195 "xi-grammar.y"
{ yyval.type = new BuiltinType("int"); ;
    break;}
case 37:
#line 197 "xi-grammar.y"
{ yyval.type = new BuiltinType("long"); ;
    break;}
case 38:
#line 199 "xi-grammar.y"
{ yyval.type = new BuiltinType("short"); ;
    break;}
case 39:
#line 201 "xi-grammar.y"
{ yyval.type = new BuiltinType("char"); ;
    break;}
case 40:
#line 203 "xi-grammar.y"
{ yyval.type = new BuiltinType("unsigned int"); ;
    break;}
case 41:
#line 205 "xi-grammar.y"
{ yyval.type = new BuiltinType("long"); ;
    break;}
case 42:
#line 207 "xi-grammar.y"
{ yyval.type = new BuiltinType("short"); ;
    break;}
case 43:
#line 209 "xi-grammar.y"
{ yyval.type = new BuiltinType("char"); ;
    break;}
case 44:
#line 211 "xi-grammar.y"
{ yyval.type = new BuiltinType("long long"); ;
    break;}
case 45:
#line 213 "xi-grammar.y"
{ yyval.type = new BuiltinType("float"); ;
    break;}
case 46:
#line 215 "xi-grammar.y"
{ yyval.type = new BuiltinType("double"); ;
    break;}
case 47:
#line 217 "xi-grammar.y"
{ yyval.type = new BuiltinType("long double"); ;
    break;}
case 48:
#line 219 "xi-grammar.y"
{ yyval.type = new BuiltinType("void"); ;
    break;}
case 49:
#line 223 "xi-grammar.y"
{ yyval.ntype = new NamedType(yyvsp[-1].strval, yyvsp[0].tparlist); ;
    break;}
case 50:
#line 227 "xi-grammar.y"
{ yyval.type = yyvsp[0].type; ;
    break;}
case 51:
#line 229 "xi-grammar.y"
{ yyval.type = yyvsp[0].ntype; ;
    break;}
case 52:
#line 233 "xi-grammar.y"
{ yyval.ptype = new PtrType(yyvsp[-1].type); ;
    break;}
case 53:
#line 237 "xi-grammar.y"
{ yyvsp[-1].ptype->indirect(); yyval.ptype = yyvsp[-1].ptype; ;
    break;}
case 54:
#line 239 "xi-grammar.y"
{ yyvsp[-1].ptype->indirect(); yyval.ptype = yyvsp[-1].ptype; ;
    break;}
case 55:
#line 243 "xi-grammar.y"
{ yyval.val = new Value(yyvsp[0].strval); ;
    break;}
case 56:
#line 245 "xi-grammar.y"
{ yyval.val = new Value(yyvsp[0].strval); ;
    break;}
case 57:
#line 249 "xi-grammar.y"
{ yyval.type = new ArrayType(yyvsp[-3].type, yyvsp[-1].val); ;
    break;}
case 58:
#line 253 "xi-grammar.y"
{ yyval.ftype = new FuncType(yyvsp[-7].type, yyvsp[-4].strval, yyvsp[-1].typelist); ;
    break;}
case 59:
#line 257 "xi-grammar.y"
{ yyval.type = yyvsp[0].type; ;
    break;}
case 60:
#line 259 "xi-grammar.y"
{ yyval.type = (Type*) yyvsp[0].ptype; ;
    break;}
case 61:
#line 261 "xi-grammar.y"
{ yyval.type = (Type*) yyvsp[0].ptype; ;
    break;}
case 62:
#line 263 "xi-grammar.y"
{ yyval.type = yyvsp[0].type; ;
    break;}
case 63:
#line 265 "xi-grammar.y"
{ yyval.type = yyvsp[0].ftype; ;
    break;}
case 64:
#line 269 "xi-grammar.y"
{ yyval.typelist = 0; ;
    break;}
case 65:
#line 271 "xi-grammar.y"
{ yyval.typelist = new TypeList(yyvsp[0].type); ;
    break;}
case 66:
#line 273 "xi-grammar.y"
{ yyval.typelist = new TypeList(yyvsp[-2].type, yyvsp[0].typelist); ;
    break;}
case 67:
#line 277 "xi-grammar.y"
{ yyval.val = yyvsp[-1].val; ;
    break;}
case 68:
#line 281 "xi-grammar.y"
{ yyval.vallist = 0; ;
    break;}
case 69:
#line 283 "xi-grammar.y"
{ yyval.vallist = new ValueList(yyvsp[-1].val, yyvsp[0].vallist); ;
    break;}
case 70:
#line 287 "xi-grammar.y"
{ yyval.readonly = new Readonly(lineno, yyvsp[-2].type, yyvsp[-1].strval, yyvsp[0].vallist); ;
    break;}
case 71:
#line 291 "xi-grammar.y"
{ yyval.readonly = new Readonly(lineno, yyvsp[-2].type, yyvsp[0].strval, 0, 1); ;
    break;}
case 72:
#line 295 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 73:
#line 297 "xi-grammar.y"
{ yyval.intval = yyvsp[-1].intval; ;
    break;}
case 74:
#line 301 "xi-grammar.y"
{ yyval.intval = yyvsp[0].intval; ;
    break;}
case 75:
#line 303 "xi-grammar.y"
{ yyval.intval = yyvsp[-2].intval | yyvsp[0].intval; ;
    break;}
case 76:
#line 307 "xi-grammar.y"
{ yyval.intval = SPACKED; ;
    break;}
case 77:
#line 309 "xi-grammar.y"
{ yyval.intval = SVARSIZE; ;
    break;}
case 78:
#line 313 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 79:
#line 315 "xi-grammar.y"
{ yyval.intval = yyvsp[-1].intval; ;
    break;}
case 80:
#line 319 "xi-grammar.y"
{ yyval.intval = yyvsp[0].intval; ;
    break;}
case 81:
#line 321 "xi-grammar.y"
{ yyval.intval = yyvsp[-2].intval | yyvsp[0].intval; ;
    break;}
case 82:
#line 325 "xi-grammar.y"
{ yyval.intval = 0x01; ;
    break;}
case 83:
#line 329 "xi-grammar.y"
{ yyval.message = new Message(lineno, yyvsp[0].ntype, yyvsp[-1].intval); ;
    break;}
case 84:
#line 331 "xi-grammar.y"
{ yyval.message = new Message(lineno, yyvsp[-3].ntype, yyvsp[-4].intval, yyvsp[-1].typelist); ;
    break;}
case 85:
#line 335 "xi-grammar.y"
{ yyval.typelist = 0; ;
    break;}
case 86:
#line 337 "xi-grammar.y"
{ yyval.typelist = yyvsp[0].typelist; ;
    break;}
case 87:
#line 341 "xi-grammar.y"
{ yyval.typelist = new TypeList(yyvsp[0].ntype); ;
    break;}
case 88:
#line 343 "xi-grammar.y"
{ yyval.typelist = new TypeList(yyvsp[-2].ntype, yyvsp[0].typelist); ;
    break;}
case 89:
#line 347 "xi-grammar.y"
{ yyval.chare = new Chare(lineno, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist, yyvsp[-3].intval); ;
    break;}
case 90:
#line 349 "xi-grammar.y"
{ yyval.chare = new MainChare(lineno, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist, yyvsp[-3].intval); ;
    break;}
case 91:
#line 353 "xi-grammar.y"
{ yyval.chare = new Group(lineno, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist, yyvsp[-3].intval); ;
    break;}
case 92:
#line 357 "xi-grammar.y"
{ yyval.chare = new NodeGroup(lineno, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist, yyvsp[-3].intval); ;
    break;}
case 93:
#line 361 "xi-grammar.y"
{/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",yyvsp[-2].strval);
			yyval.ntype = new NamedType(buf); 
		;
    break;}
case 94:
#line 367 "xi-grammar.y"
{ yyval.ntype = new NamedType(yyvsp[-1].strval); ;
    break;}
case 95:
#line 371 "xi-grammar.y"
{ yyval.chare = new Array(lineno, yyvsp[-3].ntype, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 96:
#line 375 "xi-grammar.y"
{ yyval.chare = new Chare(lineno, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist, yyvsp[-3].intval);;
    break;}
case 97:
#line 377 "xi-grammar.y"
{ yyval.chare = new MainChare(lineno, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist, yyvsp[-3].intval); ;
    break;}
case 98:
#line 381 "xi-grammar.y"
{ yyval.chare = new Group(lineno, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist, yyvsp[-3].intval); ;
    break;}
case 99:
#line 385 "xi-grammar.y"
{ yyval.chare = new NodeGroup( lineno, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist, yyvsp[-3].intval); ;
    break;}
case 100:
#line 389 "xi-grammar.y"
{ yyval.chare = new Array( lineno, yyvsp[-3].ntype, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 101:
#line 393 "xi-grammar.y"
{ yyval.message = new Message(lineno, new NamedType(yyvsp[-1].strval), yyvsp[-2].intval); ;
    break;}
case 102:
#line 397 "xi-grammar.y"
{ yyval.type = 0; ;
    break;}
case 103:
#line 399 "xi-grammar.y"
{ yyval.type = yyvsp[0].type; ;
    break;}
case 104:
#line 403 "xi-grammar.y"
{ yyval.strval = 0; ;
    break;}
case 105:
#line 405 "xi-grammar.y"
{ yyval.strval = yyvsp[0].strval; ;
    break;}
case 106:
#line 407 "xi-grammar.y"
{ yyval.strval = yyvsp[0].strval; ;
    break;}
case 107:
#line 411 "xi-grammar.y"
{ yyval.tvar = new TType(new NamedType(yyvsp[-1].strval), yyvsp[0].type); ;
    break;}
case 108:
#line 413 "xi-grammar.y"
{ yyval.tvar = new TFunc(yyvsp[-1].ftype, yyvsp[0].strval); ;
    break;}
case 109:
#line 415 "xi-grammar.y"
{ yyval.tvar = new TName(yyvsp[-2].type, yyvsp[-1].strval, yyvsp[0].strval); ;
    break;}
case 110:
#line 419 "xi-grammar.y"
{ yyval.tvarlist = new TVarList(yyvsp[0].tvar); ;
    break;}
case 111:
#line 421 "xi-grammar.y"
{ yyval.tvarlist = new TVarList(yyvsp[-2].tvar, yyvsp[0].tvarlist); ;
    break;}
case 112:
#line 425 "xi-grammar.y"
{ yyval.tvarlist = yyvsp[-1].tvarlist; ;
    break;}
case 113:
#line 429 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); ;
    break;}
case 114:
#line 431 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); ;
    break;}
case 115:
#line 433 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); ;
    break;}
case 116:
#line 435 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); ;
    break;}
case 117:
#line 437 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].message); yyvsp[0].message->setTemplate(yyval.templat); ;
    break;}
case 118:
#line 441 "xi-grammar.y"
{ yyval.mbrlist = 0; ;
    break;}
case 119:
#line 443 "xi-grammar.y"
{ yyval.mbrlist = yyvsp[-2].mbrlist; ;
    break;}
case 120:
#line 447 "xi-grammar.y"
{ yyval.mbrlist = 0; ;
    break;}
case 121:
#line 449 "xi-grammar.y"
{ yyval.mbrlist = new MemberList(yyvsp[-1].member, yyvsp[0].mbrlist); ;
    break;}
case 122:
#line 453 "xi-grammar.y"
{ yyval.member = yyvsp[-1].entry; ;
    break;}
case 123:
#line 455 "xi-grammar.y"
{ yyval.member = yyvsp[-1].readonly; ;
    break;}
case 124:
#line 457 "xi-grammar.y"
{ yyval.member = yyvsp[-1].readonly; ;
    break;}
case 125:
#line 461 "xi-grammar.y"
{ yyval.entry = new Entry(lineno, yyvsp[-5].intval|yyvsp[-1].intval, new BuiltinType("void"), yyvsp[-3].strval, yyvsp[-2].rtype, yyvsp[0].val); ;
    break;}
case 126:
#line 463 "xi-grammar.y"
{ yyval.entry = new Entry(lineno, yyvsp[-5].intval|yyvsp[-1].intval, yyvsp[-4].ptype, yyvsp[-3].strval, yyvsp[-2].rtype, yyvsp[0].val); ;
    break;}
case 127:
#line 465 "xi-grammar.y"
{ yyval.entry = new Entry(lineno, yyvsp[-2].intval, 0, yyvsp[-1].strval, yyvsp[0].rtype, 0); ;
    break;}
case 128:
#line 469 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 129:
#line 471 "xi-grammar.y"
{ yyval.intval = yyvsp[-1].intval; ;
    break;}
case 130:
#line 475 "xi-grammar.y"
{ yyval.intval = yyvsp[0].intval; ;
    break;}
case 131:
#line 477 "xi-grammar.y"
{ yyval.intval = yyvsp[-2].intval | yyvsp[0].intval; ;
    break;}
case 132:
#line 481 "xi-grammar.y"
{ yyval.intval = STHREADED; ;
    break;}
case 133:
#line 483 "xi-grammar.y"
{ yyval.intval = SSYNC; ;
    break;}
case 134:
#line 485 "xi-grammar.y"
{ yyval.intval = SLOCKED; ;
    break;}
case 135:
#line 487 "xi-grammar.y"
{ yyval.intval = SVIRTUAL; ;
    break;}
case 136:
#line 491 "xi-grammar.y"
{ yyval.rtype = 0; ;
    break;}
case 137:
#line 493 "xi-grammar.y"
{ yyval.rtype = new BuiltinType("void"); ;
    break;}
case 138:
#line 495 "xi-grammar.y"
{ yyval.rtype = yyvsp[0].ptype; ;
    break;}
case 139:
#line 499 "xi-grammar.y"
{ yyval.rtype = yyvsp[-1].rtype; ;
    break;}
case 140:
#line 503 "xi-grammar.y"
{ yyval.val = 0; ;
    break;}
case 141:
#line 505 "xi-grammar.y"
{ yyval.val = new Value(yyvsp[0].strval); ;
    break;}
case 142:
#line 509 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 143:
#line 511 "xi-grammar.y"
{ if(strcmp(yyvsp[0].strval, "0")) { yyerror("expected 0"); exit(1); }
		  yyval.intval = SPURE; 
		;
    break;}
}
   /* the action file gets copied in in place of this dollarsign */
#line 543 "/home/csar2/bhandark/share/bison.simple"

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
#line 515 "xi-grammar.y"

void yyerror(const char *mesg)
{
  cout << "Syntax error at line " << lineno << ": " << mesg << endl;
  // return 0;
}
