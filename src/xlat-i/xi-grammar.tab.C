
/*  A Bison parser, made from xi-grammar.y
 by  GNU Bison version 1.27
  */

#define YYBISON 1  /* Identify Bison output.  */

#define	MODULE	257
#define	MAINMODULE	258
#define	EXTERN	259
#define	READONLY	260
#define	CHARE	261
#define	MAINCHARE	262
#define	GROUP	263
#define	NODEGROUP	264
#define	ARRAY	265
#define	MESSAGE	266
#define	CLASS	267
#define	STACKSIZE	268
#define	THREADED	269
#define	TEMPLATE	270
#define	SYNC	271
#define	EXCLUSIVE	272
#define	VIRTUAL	273
#define	MIGRATABLE	274
#define	CREATEHERE	275
#define	CREATEHOME	276
#define	VOID	277
#define	CONST	278
#define	PACKED	279
#define	VARSIZE	280
#define	ENTRY	281
#define	IDENT	282
#define	NUMBER	283
#define	LITERAL	284
#define	CPROGRAM	285
#define	INT	286
#define	LONG	287
#define	SHORT	288
#define	CHAR	289
#define	FLOAT	290
#define	DOUBLE	291
#define	UNSIGNED	292

#line 1 "xi-grammar.y"

#include <iostream.h>
#include "xi-symbol.h"

extern int yylex (void) ;
void yyerror(const char *);
extern unsigned int lineno;
extern int in_bracket,in_braces;
ModuleList *modlist;


#line 13 "xi-grammar.y"
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
  char *strval;
  int intval;
  Chare::attrib_t cattr;
} YYSTYPE;
#include <stdio.h>

#ifndef __cplusplus
#ifndef __STDC__
#define const
#endif
#endif



#define	YYFINAL		295
#define	YYFLAG		-32768
#define	YYNTBASE	53

#define YYTRANSLATE(x) ((unsigned)(x) <= 292 ? yytranslate[x] : 123)

static const char yytranslate[] = {     0,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,    51,     2,    47,
    48,    46,     2,    43,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,    40,    39,    44,
    52,    45,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
    49,     2,    50,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,    41,     2,    42,     2,     2,     2,     2,     2,
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
    27,    28,    29,    30,    31,    32,    33,    34,    35,    36,
    37,    38
};

#if YYDEBUG != 0
static const short yyprhs[] = {     0,
     0,     2,     3,     6,     7,     9,    10,    12,    14,    16,
    21,    25,    29,    31,    36,    37,    40,    46,    49,    53,
    57,    61,    64,    67,    70,    73,    76,    78,    80,    82,
    84,    88,    89,    91,    92,    96,    98,   100,   102,   104,
   107,   110,   113,   116,   119,   121,   123,   126,   128,   131,
   134,   136,   138,   141,   144,   147,   156,   158,   160,   162,
   164,   169,   172,   174,   176,   180,   181,   184,   189,   195,
   196,   200,   202,   206,   208,   210,   211,   215,   217,   221,
   223,   229,   231,   234,   238,   245,   246,   249,   251,   255,
   261,   267,   273,   279,   284,   288,   294,   300,   306,   312,
   318,   324,   329,   330,   333,   334,   337,   340,   344,   347,
   351,   353,   357,   362,   365,   368,   371,   374,   377,   379,
   384,   385,   388,   391,   394,   397,   405,   410,   412,   414,
   415,   419,   421,   425,   427,   429,   431,   433,   435,   437,
   439,   441,   443,   445,   451,   457,   461,   463,   466,   471,
   475,   477,   481,   485,   488,   489,   493,   494
};

static const short yyrhs[] = {    54,
     0,     0,    59,    54,     0,     0,     5,     0,     0,    39,
     0,    28,     0,    28,     0,    58,    40,    40,    28,     0,
     3,    57,    60,     0,     4,    57,    60,     0,    39,     0,
    41,    61,    42,    56,     0,     0,    62,    61,     0,    55,
    41,    61,    42,    56,     0,    55,    59,     0,    55,    78,
    39,     0,    55,    79,    39,     0,    55,    88,    39,     0,
    55,    91,     0,    55,    92,     0,    55,    93,     0,    55,
    95,     0,    55,   106,     0,    74,     0,    29,     0,    30,
     0,    63,     0,    63,    43,    64,     0,     0,    64,     0,
     0,    44,    65,    45,     0,    32,     0,    33,     0,    34,
     0,    35,     0,    38,    32,     0,    38,    33,     0,    38,
    34,     0,    38,    35,     0,    33,    33,     0,    36,     0,
    37,     0,    33,    37,     0,    23,     0,    57,    66,     0,
    58,    66,     0,    67,     0,    69,     0,    70,    46,     0,
    71,    46,     0,    72,    46,     0,    74,    47,    46,    57,
    48,    47,   119,    48,     0,    70,     0,    71,     0,    72,
     0,    73,     0,    74,    49,    75,    50,     0,    74,    51,
     0,    29,     0,    58,     0,    49,    75,    50,     0,     0,
    76,    77,     0,     6,    74,    58,    77,     0,     6,    12,
    70,    46,    57,     0,     0,    49,    81,    50,     0,    82,
     0,    82,    43,    81,     0,    25,     0,    26,     0,     0,
    49,    84,    50,     0,    85,     0,    85,    43,    84,     0,
    20,     0,    74,    57,    49,    50,    39,     0,    86,     0,
    86,    87,     0,    12,    80,    68,     0,    12,    80,    68,
    41,    87,    42,     0,     0,    40,    90,     0,    68,     0,
    68,    43,    90,     0,     7,    83,    68,    89,   107,     0,
     8,    83,    68,    89,   107,     0,     9,    83,    68,    89,
   107,     0,    10,    83,    68,    89,   107,     0,    49,    29,
    57,    50,     0,    49,    57,    50,     0,    11,    94,    68,
    89,   107,     0,     7,    83,    57,    89,   107,     0,     8,
    83,    57,    89,   107,     0,     9,    83,    57,    89,   107,
     0,    10,    83,    57,    89,   107,     0,    11,    94,    57,
    89,   107,     0,    12,    80,    57,    39,     0,     0,    52,
    74,     0,     0,    52,    29,     0,    52,    30,     0,    13,
    57,   101,     0,    73,   102,     0,    74,    57,   102,     0,
   103,     0,   103,    43,   104,     0,    16,    44,   104,    45,
     0,   105,    96,     0,   105,    97,     0,   105,    98,     0,
   105,    99,     0,   105,   100,     0,    39,     0,    41,   108,
    42,    56,     0,     0,   109,   108,     0,   110,    39,     0,
    78,    39,     0,    79,    39,     0,    27,   112,   111,    57,
   120,   122,   121,     0,    27,   112,    57,   120,     0,    23,
     0,    71,     0,     0,    49,   113,    50,     0,   114,     0,
   114,    43,   113,     0,    15,     0,    17,     0,    18,     0,
    19,     0,    21,     0,    22,     0,    30,     0,    29,     0,
    58,     0,    31,     0,    31,    49,    31,    50,    31,     0,
    31,    41,    31,    42,    31,     0,    74,    57,    49,     0,
    74,     0,    74,    57,     0,    74,    57,    52,   115,     0,
   117,   116,    50,     0,   118,     0,   118,    43,   119,     0,
    47,   119,    48,     0,    47,    48,     0,     0,    14,    52,
    29,     0,     0,    52,    29,     0
};

#endif

#if YYDEBUG != 0
static const short yyrline[] = { 0,
    99,   103,   105,   109,   111,   115,   117,   121,   125,   127,
   135,   137,   141,   143,   147,   149,   153,   155,   157,   159,
   161,   163,   165,   167,   169,   171,   175,   177,   179,   183,
   185,   189,   191,   195,   197,   201,   203,   205,   207,   209,
   211,   213,   215,   217,   219,   221,   223,   225,   229,   230,
   232,   234,   238,   242,   244,   248,   252,   254,   256,   258,
   260,   262,   268,   270,   274,   278,   280,   284,   288,   292,
   294,   304,   306,   310,   312,   316,   318,   322,   324,   328,
   332,   336,   338,   342,   344,   348,   350,   354,   356,   360,
   362,   366,   370,   374,   380,   384,   388,   390,   394,   398,
   402,   406,   410,   412,   416,   418,   420,   424,   426,   428,
   432,   434,   438,   442,   444,   446,   448,   450,   454,   456,
   460,   462,   466,   468,   470,   474,   476,   480,   482,   486,
   488,   492,   494,   498,   500,   502,   504,   506,   508,   512,
   514,   516,   520,   522,   528,   536,   543,   545,   547,   549,
   556,   558,   562,   564,   568,   570,   574,   576
};
#endif


#if YYDEBUG != 0 || defined (YYERROR_VERBOSE)

static const char * const yytname[] = {   "$","error","$undefined.","MODULE",
"MAINMODULE","EXTERN","READONLY","CHARE","MAINCHARE","GROUP","NODEGROUP","ARRAY",
"MESSAGE","CLASS","STACKSIZE","THREADED","TEMPLATE","SYNC","EXCLUSIVE","VIRTUAL",
"MIGRATABLE","CREATEHERE","CREATEHOME","VOID","CONST","PACKED","VARSIZE","ENTRY",
"IDENT","NUMBER","LITERAL","CPROGRAM","INT","LONG","SHORT","CHAR","FLOAT","DOUBLE",
"UNSIGNED","';'","':'","'{'","'}'","','","'<'","'>'","'*'","'('","')'","'['",
"']'","'&'","'='","File","ModuleEList","OptExtern","OptSemiColon","Name","QualName",
"Module","ConstructEList","ConstructList","Construct","TParam","TParamList",
"TParamEList","OptTParams","BuiltinType","NamedType","QualNamedType","SimpleType",
"OnePtrType","PtrType","FuncType","Type","ArrayDim","Dim","DimList","Readonly",
"ReadonlyMsg","MAttribs","MAttribList","MAttrib","CAttribs","CAttribList","CAttrib",
"Var","VarList","Message","OptBaseList","BaseList","Chare","Group","NodeGroup",
"ArrayIndexType","Array","TChare","TGroup","TNodeGroup","TArray","TMessage",
"OptTypeInit","OptNameInit","TVar","TVarList","TemplateSpec","Template","MemberEList",
"MemberList","Member","Entry","EReturn","EAttribs","EAttribList","EAttrib","DefaultParameter",
"CCode","ParamBracketStart","Parameter","ParamList","EParameters","OptStackSize",
"OptPure", NULL
};
#endif

static const short yyr1[] = {     0,
    53,    54,    54,    55,    55,    56,    56,    57,    58,    58,
    59,    59,    60,    60,    61,    61,    62,    62,    62,    62,
    62,    62,    62,    62,    62,    62,    63,    63,    63,    64,
    64,    65,    65,    66,    66,    67,    67,    67,    67,    67,
    67,    67,    67,    67,    67,    67,    67,    67,    68,    69,
    70,    70,    71,    72,    72,    73,    74,    74,    74,    74,
    74,    74,    75,    75,    76,    77,    77,    78,    79,    80,
    80,    81,    81,    82,    82,    83,    83,    84,    84,    85,
    86,    87,    87,    88,    88,    89,    89,    90,    90,    91,
    91,    92,    93,    94,    94,    95,    96,    96,    97,    98,
    99,   100,   101,   101,   102,   102,   102,   103,   103,   103,
   104,   104,   105,   106,   106,   106,   106,   106,   107,   107,
   108,   108,   109,   109,   109,   110,   110,   111,   111,   112,
   112,   113,   113,   114,   114,   114,   114,   114,   114,   115,
   115,   115,   116,   116,   116,   117,   118,   118,   118,   118,
   119,   119,   120,   120,   121,   121,   122,   122
};

static const short yyr2[] = {     0,
     1,     0,     2,     0,     1,     0,     1,     1,     1,     4,
     3,     3,     1,     4,     0,     2,     5,     2,     3,     3,
     3,     2,     2,     2,     2,     2,     1,     1,     1,     1,
     3,     0,     1,     0,     3,     1,     1,     1,     1,     2,
     2,     2,     2,     2,     1,     1,     2,     1,     2,     2,
     1,     1,     2,     2,     2,     8,     1,     1,     1,     1,
     4,     2,     1,     1,     3,     0,     2,     4,     5,     0,
     3,     1,     3,     1,     1,     0,     3,     1,     3,     1,
     5,     1,     2,     3,     6,     0,     2,     1,     3,     5,
     5,     5,     5,     4,     3,     5,     5,     5,     5,     5,
     5,     4,     0,     2,     0,     2,     2,     3,     2,     3,
     1,     3,     4,     2,     2,     2,     2,     2,     1,     4,
     0,     2,     2,     2,     2,     7,     4,     1,     1,     0,
     3,     1,     3,     1,     1,     1,     1,     1,     1,     1,
     1,     1,     1,     5,     5,     3,     1,     2,     4,     3,
     1,     3,     3,     2,     0,     3,     0,     2
};

static const short yydefact[] = {     2,
     0,     0,     1,     2,     8,     0,     0,     3,    13,     4,
    11,    12,     5,     0,     0,     4,     0,    76,    76,    76,
    76,     0,    70,     0,     4,    18,     0,     0,     0,    22,
    23,    24,    25,     0,    26,     6,    16,     0,    48,     9,
    36,    37,    38,    39,    45,    46,     0,    34,    51,    52,
    57,    58,    59,    60,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,    19,    20,    21,    76,
    76,    76,    76,     0,    70,   114,   115,   116,   117,   118,
     7,    14,     0,    44,    47,    40,    41,    42,    43,     0,
    32,    50,    53,    54,    55,     0,     0,    62,    66,    80,
     0,    78,    34,    86,    86,    86,    86,     0,     0,    86,
    74,    75,     0,    72,    84,     0,    60,     0,   111,     0,
     6,     0,     0,     0,     0,     0,     0,     0,     0,    28,
    29,    30,    33,     0,    27,     0,    63,    64,     0,     0,
    66,    68,    77,     0,    49,     0,     0,     0,     0,     0,
     0,    95,     0,    71,     0,     0,   103,     0,   109,   105,
     0,   113,    17,    86,    86,    86,    86,    86,     0,    69,
    10,     0,    35,     0,    61,     0,    67,    79,    88,    87,
   119,   121,    90,    91,    92,    93,    94,    96,    73,     0,
    82,     0,     0,   108,   106,   107,   110,   112,     0,     0,
     0,     0,     0,   102,    31,     0,    65,     0,   130,     0,
     0,     0,   121,     0,     0,    83,    85,   104,    97,    98,
    99,   100,   101,     0,    89,     0,     0,   124,   125,     6,
   122,   123,     0,   147,     0,   151,     0,   134,   135,   136,
   137,   138,   139,     0,   132,    48,     9,     0,     0,   129,
     0,   120,     0,   148,   143,     0,     0,    56,   131,     0,
     0,   127,     0,    81,   146,     0,     0,     0,   150,   152,
   133,   154,     0,   157,   141,   140,   142,   149,     0,     0,
   153,     0,   155,     0,     0,   158,     0,   126,   145,   144,
     0,   156,     0,     0,     0
};

static const short yydefgoto[] = {   293,
     3,    14,    82,   103,    48,     4,    11,    15,    16,   132,
   133,   134,    92,    49,   179,    50,    51,    52,    53,    54,
   234,   139,   141,   142,   210,   211,    64,   113,   114,    57,
   101,   102,   191,   192,    29,   147,   180,    30,    31,    32,
    62,    33,    76,    77,    78,    79,    80,   194,   159,   119,
   120,    34,    35,   183,   212,   213,   214,   251,   227,   244,
   245,   278,   256,   235,   236,   237,   262,   288,   283
};

static const short yypact[] = {   143,
    15,    15,-32768,   143,-32768,    71,    71,-32768,-32768,    19,
-32768,-32768,-32768,     7,   -15,    19,   117,   -17,   -17,   -17,
   -17,    -4,    28,    45,    19,-32768,    55,    66,    70,-32768,
-32768,-32768,-32768,   207,-32768,    74,-32768,   149,-32768,-32768,
-32768,    25,-32768,-32768,-32768,-32768,   141,     2,-32768,-32768,
    73,    82,    85,-32768,     6,   118,    15,    15,    15,    15,
   129,    15,   169,    15,    63,   102,-32768,-32768,-32768,   -17,
   -17,   -17,   -17,    -4,    28,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,    96,-32768,-32768,-32768,-32768,-32768,-32768,   120,
   133,-32768,-32768,-32768,-32768,   150,   162,-32768,    10,-32768,
   114,   136,   164,   170,   170,   170,   170,    15,   172,   170,
-32768,-32768,   173,   181,   171,    15,     4,    57,   182,   183,
    74,    15,    15,    15,    15,    15,    15,    15,   199,-32768,
-32768,   186,-32768,   185,    69,    15,-32768,   192,   184,   162,
   187,-32768,-32768,   118,-32768,    15,   100,   100,   100,   100,
   188,-32768,   100,-32768,   169,   149,   179,   191,-32768,   189,
    63,-32768,-32768,   170,   170,   170,   170,   170,   196,-32768,
-32768,   133,-32768,   194,-32768,   190,-32768,-32768,   200,-32768,
-32768,    27,-32768,-32768,-32768,-32768,-32768,-32768,-32768,    57,
   149,   195,   149,-32768,-32768,-32768,-32768,-32768,   100,   100,
   100,   100,   100,-32768,-32768,   197,-32768,    15,   198,   206,
   209,   204,    27,   210,   202,-32768,-32768,    69,-32768,-32768,
-32768,-32768,-32768,   149,-32768,   115,   165,-32768,-32768,    74,
-32768,-32768,   203,    57,   208,   211,   212,-32768,-32768,-32768,
-32768,-32768,-32768,   205,   213,   224,   214,   215,    73,-32768,
    15,-32768,   218,    41,    62,   216,   149,-32768,-32768,   115,
     3,-32768,   215,-32768,-32768,    35,   227,   228,-32768,-32768,
-32768,-32768,   217,   219,-32768,-32768,   192,-32768,   221,   220,
-32768,   235,   253,   237,   238,-32768,   222,-32768,-32768,-32768,
   243,-32768,   273,   275,-32768
};

static const short yypgoto[] = {-32768,
   272,-32768,  -116,    -1,   -53,   263,   271,    67,-32768,-32768,
   107,-32768,   177,-32768,   147,-32768,   -35,    54,-32768,   -59,
   -13,   142,-32768,   144,   269,   270,   223,   131,-32768,     9,
   145,-32768,-32768,    97,-32768,   -98,    79,-32768,-32768,-32768,
   225,-32768,-32768,-32768,-32768,-32768,-32768,-32768,   130,-32768,
   132,-32768,-32768,  -128,    78,-32768,-32768,-32768,-32768,    32,
-32768,-32768,-32768,-32768,-32768,  -173,    31,-32768,-32768
};


#define	YYLAST		299


static const short yytable[] = {     6,
     7,    99,    83,    55,   163,   117,   148,   149,   150,     1,
     2,   153,    17,    18,    19,    20,    21,    22,    23,   184,
   185,   186,    24,    13,   188,    39,    36,    58,    59,    60,
    40,    56,    17,    40,    41,    42,    43,    44,    45,    46,
    47,    90,     5,   138,    61,    91,  -105,    25,  -105,    90,
   272,   118,    96,   209,    97,   158,    98,    84,   140,   109,
   -15,    85,    40,   275,   276,   199,   200,   201,   202,   203,
   219,   220,   221,   222,   223,   116,    63,   135,   122,   123,
   124,   125,    37,   270,     5,    39,   138,   273,    65,   265,
    40,    66,   266,    67,    41,    42,    43,    44,    45,    46,
    47,   117,   267,    96,    68,    97,   151,    98,    69,     9,
   268,    10,    81,   252,   157,    96,   160,    97,    93,    98,
   164,   165,   166,   167,   168,   169,   170,    94,    38,   238,
    95,   239,   240,   241,   174,   242,   243,   100,   181,    39,
   182,   128,   190,   121,    40,     1,     2,   118,    41,    42,
    43,    44,    45,    46,    47,    39,     5,   108,   135,   129,
    40,   130,   131,   143,    41,    42,    43,    44,    45,    46,
    47,    39,    86,    87,    88,    89,    40,   190,   144,   218,
    41,    42,    43,    44,    45,    46,    47,   246,   215,    40,
   137,   249,   247,   111,   112,   136,    41,    42,    43,    44,
    45,    46,    47,   104,   105,   106,   107,    91,   110,   146,
   115,   156,   277,    70,    71,    72,    73,    74,    75,   195,
   196,   152,   154,   155,   161,   248,   171,   162,   172,   173,
   193,    90,   254,   175,   204,   140,   217,   187,   255,   207,
   158,   206,   208,   224,   228,   230,   226,   229,   232,   263,
   233,  -128,   253,   257,   259,   260,   264,   279,   280,   258,
    -8,   261,   284,   286,   281,   269,   287,   289,   290,   285,
   282,   292,   294,   291,   295,     8,    26,    12,   205,   145,
   250,   176,    27,    28,   177,   189,   225,   216,   178,   197,
   231,   271,   198,   274,     0,     0,     0,   127,   126
};

static const short yycheck[] = {     1,
     2,    55,    38,    17,   121,    65,   105,   106,   107,     3,
     4,   110,     6,     7,     8,     9,    10,    11,    12,   148,
   149,   150,    16,     5,   153,    23,    42,    19,    20,    21,
    28,    49,     6,    28,    32,    33,    34,    35,    36,    37,
    38,    40,    28,    97,    49,    44,    43,    41,    45,    40,
    48,    65,    47,    27,    49,    52,    51,    33,    49,    61,
    42,    37,    28,    29,    30,   164,   165,   166,   167,   168,
   199,   200,   201,   202,   203,    13,    49,    91,    70,    71,
    72,    73,    16,   257,    28,    23,   140,   261,    44,    49,
    28,    25,    52,    39,    32,    33,    34,    35,    36,    37,
    38,   161,    41,    47,    39,    49,   108,    51,    39,    39,
    49,    41,    39,   230,   116,    47,   118,    49,    46,    51,
   122,   123,   124,   125,   126,   127,   128,    46,    12,    15,
    46,    17,    18,    19,   136,    21,    22,    20,    39,    23,
    41,    46,   156,    42,    28,     3,     4,   161,    32,    33,
    34,    35,    36,    37,    38,    23,    28,    29,   172,    40,
    28,    29,    30,    50,    32,    33,    34,    35,    36,    37,
    38,    23,    32,    33,    34,    35,    28,   191,    43,   193,
    32,    33,    34,    35,    36,    37,    38,    23,   190,    28,
    29,   227,    28,    25,    26,    46,    32,    33,    34,    35,
    36,    37,    38,    57,    58,    59,    60,    44,    62,    40,
    64,    41,   266,     7,     8,     9,    10,    11,    12,    29,
    30,    50,    50,    43,    43,   227,    28,    45,    43,    45,
    52,    40,   234,    50,    39,    49,    42,    50,    31,    50,
    52,    48,    43,    47,    39,    42,    49,    39,    39,   251,
    49,    28,    50,    43,    50,    43,    39,    31,    31,    48,
    47,    47,    42,    29,    48,    50,    14,    31,    31,    50,
    52,    29,     0,    52,     0,     4,    14,     7,   172,   103,
   227,   140,    14,    14,   141,   155,   208,   191,   144,   160,
   213,   260,   161,   263,    -1,    -1,    -1,    75,    74
};
/* -*-C-*-  Note some compilers choke on comments on `#line' lines.  */
#line 3 "/usr/share/bison.simple"
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

#line 216 "/usr/share/bison.simple"

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
#line 100 "xi-grammar.y"
{ yyval.modlist = yyvsp[0].modlist; modlist = yyvsp[0].modlist; ;
    break;}
case 2:
#line 104 "xi-grammar.y"
{ yyval.modlist = 0; ;
    break;}
case 3:
#line 106 "xi-grammar.y"
{ yyval.modlist = new ModuleList(lineno, yyvsp[-1].module, yyvsp[0].modlist); ;
    break;}
case 4:
#line 110 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 5:
#line 112 "xi-grammar.y"
{ yyval.intval = 1; ;
    break;}
case 6:
#line 116 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 7:
#line 118 "xi-grammar.y"
{ yyval.intval = 1; ;
    break;}
case 8:
#line 122 "xi-grammar.y"
{ yyval.strval = yyvsp[0].strval; ;
    break;}
case 9:
#line 126 "xi-grammar.y"
{ yyval.strval = yyvsp[0].strval; ;
    break;}
case 10:
#line 128 "xi-grammar.y"
{
		  char *tmp = new char[strlen(yyvsp[-3].strval)+strlen(yyvsp[0].strval)+3];
		  sprintf(tmp,"%s::%s", yyvsp[-3].strval, yyvsp[0].strval);
		  yyval.strval = tmp;
		;
    break;}
case 11:
#line 136 "xi-grammar.y"
{ yyval.module = new Module(lineno, yyvsp[-1].strval, yyvsp[0].conslist); ;
    break;}
case 12:
#line 138 "xi-grammar.y"
{ yyval.module = new Module(lineno, yyvsp[-1].strval, yyvsp[0].conslist); yyval.module->setMain(); ;
    break;}
case 13:
#line 142 "xi-grammar.y"
{ yyval.conslist = 0; ;
    break;}
case 14:
#line 144 "xi-grammar.y"
{ yyval.conslist = yyvsp[-2].conslist; ;
    break;}
case 15:
#line 148 "xi-grammar.y"
{ yyval.conslist = 0; ;
    break;}
case 16:
#line 150 "xi-grammar.y"
{ yyval.conslist = new ConstructList(lineno, yyvsp[-1].construct, yyvsp[0].conslist); ;
    break;}
case 17:
#line 154 "xi-grammar.y"
{ if(yyvsp[-2].conslist) yyvsp[-2].conslist->setExtern(yyvsp[-4].intval); yyval.construct = yyvsp[-2].conslist; ;
    break;}
case 18:
#line 156 "xi-grammar.y"
{ yyvsp[0].module->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].module; ;
    break;}
case 19:
#line 158 "xi-grammar.y"
{ yyvsp[-1].readonly->setExtern(yyvsp[-2].intval); yyval.construct = yyvsp[-1].readonly; ;
    break;}
case 20:
#line 160 "xi-grammar.y"
{ yyvsp[-1].readonly->setExtern(yyvsp[-2].intval); yyval.construct = yyvsp[-1].readonly; ;
    break;}
case 21:
#line 162 "xi-grammar.y"
{ yyvsp[-1].message->setExtern(yyvsp[-2].intval); yyval.construct = yyvsp[-1].message; ;
    break;}
case 22:
#line 164 "xi-grammar.y"
{ yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; ;
    break;}
case 23:
#line 166 "xi-grammar.y"
{ yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; ;
    break;}
case 24:
#line 168 "xi-grammar.y"
{ yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; ;
    break;}
case 25:
#line 170 "xi-grammar.y"
{ yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; ;
    break;}
case 26:
#line 172 "xi-grammar.y"
{ yyvsp[0].templat->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].templat; ;
    break;}
case 27:
#line 176 "xi-grammar.y"
{ yyval.tparam = new TParamType(yyvsp[0].type); ;
    break;}
case 28:
#line 178 "xi-grammar.y"
{ yyval.tparam = new TParamVal(yyvsp[0].strval); ;
    break;}
case 29:
#line 180 "xi-grammar.y"
{ yyval.tparam = new TParamVal(yyvsp[0].strval); ;
    break;}
case 30:
#line 184 "xi-grammar.y"
{ yyval.tparlist = new TParamList(yyvsp[0].tparam); ;
    break;}
case 31:
#line 186 "xi-grammar.y"
{ yyval.tparlist = new TParamList(yyvsp[-2].tparam, yyvsp[0].tparlist); ;
    break;}
case 32:
#line 190 "xi-grammar.y"
{ yyval.tparlist = 0; ;
    break;}
case 33:
#line 192 "xi-grammar.y"
{ yyval.tparlist = yyvsp[0].tparlist; ;
    break;}
case 34:
#line 196 "xi-grammar.y"
{ yyval.tparlist = 0; ;
    break;}
case 35:
#line 198 "xi-grammar.y"
{ yyval.tparlist = yyvsp[-1].tparlist; ;
    break;}
case 36:
#line 202 "xi-grammar.y"
{ yyval.type = new BuiltinType("int"); ;
    break;}
case 37:
#line 204 "xi-grammar.y"
{ yyval.type = new BuiltinType("long"); ;
    break;}
case 38:
#line 206 "xi-grammar.y"
{ yyval.type = new BuiltinType("short"); ;
    break;}
case 39:
#line 208 "xi-grammar.y"
{ yyval.type = new BuiltinType("char"); ;
    break;}
case 40:
#line 210 "xi-grammar.y"
{ yyval.type = new BuiltinType("unsigned int"); ;
    break;}
case 41:
#line 212 "xi-grammar.y"
{ yyval.type = new BuiltinType("unsigned long"); ;
    break;}
case 42:
#line 214 "xi-grammar.y"
{ yyval.type = new BuiltinType("unsigned short"); ;
    break;}
case 43:
#line 216 "xi-grammar.y"
{ yyval.type = new BuiltinType("unsigned char"); ;
    break;}
case 44:
#line 218 "xi-grammar.y"
{ yyval.type = new BuiltinType("long long"); ;
    break;}
case 45:
#line 220 "xi-grammar.y"
{ yyval.type = new BuiltinType("float"); ;
    break;}
case 46:
#line 222 "xi-grammar.y"
{ yyval.type = new BuiltinType("double"); ;
    break;}
case 47:
#line 224 "xi-grammar.y"
{ yyval.type = new BuiltinType("long double"); ;
    break;}
case 48:
#line 226 "xi-grammar.y"
{ yyval.type = new BuiltinType("void"); ;
    break;}
case 49:
#line 229 "xi-grammar.y"
{ yyval.ntype = new NamedType(yyvsp[-1].strval,yyvsp[0].tparlist); ;
    break;}
case 50:
#line 230 "xi-grammar.y"
{ yyval.ntype = new NamedType(yyvsp[-1].strval,yyvsp[0].tparlist); ;
    break;}
case 51:
#line 233 "xi-grammar.y"
{ yyval.type = yyvsp[0].type; ;
    break;}
case 52:
#line 235 "xi-grammar.y"
{ yyval.type = yyvsp[0].ntype; ;
    break;}
case 53:
#line 239 "xi-grammar.y"
{ yyval.ptype = new PtrType(yyvsp[-1].type); ;
    break;}
case 54:
#line 243 "xi-grammar.y"
{ yyvsp[-1].ptype->indirect(); yyval.ptype = yyvsp[-1].ptype; ;
    break;}
case 55:
#line 245 "xi-grammar.y"
{ yyvsp[-1].ptype->indirect(); yyval.ptype = yyvsp[-1].ptype; ;
    break;}
case 56:
#line 249 "xi-grammar.y"
{ yyval.ftype = new FuncType(yyvsp[-7].type, yyvsp[-4].strval, yyvsp[-1].plist); ;
    break;}
case 57:
#line 253 "xi-grammar.y"
{ yyval.type = yyvsp[0].type; ;
    break;}
case 58:
#line 255 "xi-grammar.y"
{ yyval.type = yyvsp[0].ptype; ;
    break;}
case 59:
#line 257 "xi-grammar.y"
{ yyval.type = yyvsp[0].ptype; ;
    break;}
case 60:
#line 259 "xi-grammar.y"
{ yyval.type = yyvsp[0].ftype; ;
    break;}
case 61:
#line 261 "xi-grammar.y"
{ yyval.type = new ArrayType(yyvsp[-3].type,yyvsp[-1].val); ;
    break;}
case 62:
#line 263 "xi-grammar.y"
{ yyval.type = new ReferenceType(yyvsp[-1].type); ;
    break;}
case 63:
#line 269 "xi-grammar.y"
{ yyval.val = new Value(yyvsp[0].strval); ;
    break;}
case 64:
#line 271 "xi-grammar.y"
{ yyval.val = new Value(yyvsp[0].strval); ;
    break;}
case 65:
#line 275 "xi-grammar.y"
{ yyval.val = yyvsp[-1].val; ;
    break;}
case 66:
#line 279 "xi-grammar.y"
{ yyval.vallist = 0; ;
    break;}
case 67:
#line 281 "xi-grammar.y"
{ yyval.vallist = new ValueList(yyvsp[-1].val, yyvsp[0].vallist); ;
    break;}
case 68:
#line 285 "xi-grammar.y"
{ yyval.readonly = new Readonly(lineno, yyvsp[-2].type, yyvsp[-1].strval, yyvsp[0].vallist); ;
    break;}
case 69:
#line 289 "xi-grammar.y"
{ yyval.readonly = new Readonly(lineno, yyvsp[-2].type, yyvsp[0].strval, 0, 1); ;
    break;}
case 70:
#line 293 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 71:
#line 295 "xi-grammar.y"
{ 
		  /*
		  printf("Warning: Message attributes are being phased out.\n");
		  printf("Warning: Please remove them from interface files.\n");
		  */
		  yyval.intval = yyvsp[-1].intval; 
		;
    break;}
case 72:
#line 305 "xi-grammar.y"
{ yyval.intval = yyvsp[0].intval; ;
    break;}
case 73:
#line 307 "xi-grammar.y"
{ yyval.intval = yyvsp[-2].intval | yyvsp[0].intval; ;
    break;}
case 74:
#line 311 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 75:
#line 313 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 76:
#line 317 "xi-grammar.y"
{ yyval.cattr = 0; ;
    break;}
case 77:
#line 319 "xi-grammar.y"
{ yyval.cattr = yyvsp[-1].cattr; ;
    break;}
case 78:
#line 323 "xi-grammar.y"
{ yyval.cattr = yyvsp[0].cattr; ;
    break;}
case 79:
#line 325 "xi-grammar.y"
{ yyval.cattr = yyvsp[-2].cattr | yyvsp[0].cattr; ;
    break;}
case 80:
#line 329 "xi-grammar.y"
{ yyval.cattr = Chare::CMIGRATABLE; ;
    break;}
case 81:
#line 333 "xi-grammar.y"
{ yyval.mv = new MsgVar(yyvsp[-4].type, yyvsp[-3].strval); ;
    break;}
case 82:
#line 337 "xi-grammar.y"
{ yyval.mvlist = new MsgVarList(yyvsp[0].mv); ;
    break;}
case 83:
#line 339 "xi-grammar.y"
{ yyval.mvlist = new MsgVarList(yyvsp[-1].mv, yyvsp[0].mvlist); ;
    break;}
case 84:
#line 343 "xi-grammar.y"
{ yyval.message = new Message(lineno, yyvsp[0].ntype); ;
    break;}
case 85:
#line 345 "xi-grammar.y"
{ yyval.message = new Message(lineno, yyvsp[-3].ntype, yyvsp[-1].mvlist); ;
    break;}
case 86:
#line 349 "xi-grammar.y"
{ yyval.typelist = 0; ;
    break;}
case 87:
#line 351 "xi-grammar.y"
{ yyval.typelist = yyvsp[0].typelist; ;
    break;}
case 88:
#line 355 "xi-grammar.y"
{ yyval.typelist = new TypeList(yyvsp[0].ntype); ;
    break;}
case 89:
#line 357 "xi-grammar.y"
{ yyval.typelist = new TypeList(yyvsp[-2].ntype, yyvsp[0].typelist); ;
    break;}
case 90:
#line 361 "xi-grammar.y"
{ yyval.chare = new Chare(lineno, yyvsp[-3].cattr, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 91:
#line 363 "xi-grammar.y"
{ yyval.chare = new MainChare(lineno, yyvsp[-3].cattr, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 92:
#line 367 "xi-grammar.y"
{ yyval.chare = new Group(lineno, yyvsp[-3].cattr, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 93:
#line 371 "xi-grammar.y"
{ yyval.chare = new NodeGroup(lineno, yyvsp[-3].cattr, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 94:
#line 375 "xi-grammar.y"
{/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",yyvsp[-2].strval);
			yyval.ntype = new NamedType(buf); 
		;
    break;}
case 95:
#line 381 "xi-grammar.y"
{ yyval.ntype = new NamedType(yyvsp[-1].strval); ;
    break;}
case 96:
#line 385 "xi-grammar.y"
{ yyval.chare = new Array(lineno, 0, yyvsp[-3].ntype, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 97:
#line 389 "xi-grammar.y"
{ yyval.chare = new Chare(lineno, yyvsp[-3].cattr, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist);;
    break;}
case 98:
#line 391 "xi-grammar.y"
{ yyval.chare = new MainChare(lineno, yyvsp[-3].cattr, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 99:
#line 395 "xi-grammar.y"
{ yyval.chare = new Group(lineno, yyvsp[-3].cattr, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 100:
#line 399 "xi-grammar.y"
{ yyval.chare = new NodeGroup( lineno, yyvsp[-3].cattr, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 101:
#line 403 "xi-grammar.y"
{ yyval.chare = new Array( lineno, 0, yyvsp[-3].ntype, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 102:
#line 407 "xi-grammar.y"
{ yyval.message = new Message(lineno, new NamedType(yyvsp[-1].strval)); ;
    break;}
case 103:
#line 411 "xi-grammar.y"
{ yyval.type = 0; ;
    break;}
case 104:
#line 413 "xi-grammar.y"
{ yyval.type = yyvsp[0].type; ;
    break;}
case 105:
#line 417 "xi-grammar.y"
{ yyval.strval = 0; ;
    break;}
case 106:
#line 419 "xi-grammar.y"
{ yyval.strval = yyvsp[0].strval; ;
    break;}
case 107:
#line 421 "xi-grammar.y"
{ yyval.strval = yyvsp[0].strval; ;
    break;}
case 108:
#line 425 "xi-grammar.y"
{ yyval.tvar = new TType(new NamedType(yyvsp[-1].strval), yyvsp[0].type); ;
    break;}
case 109:
#line 427 "xi-grammar.y"
{ yyval.tvar = new TFunc(yyvsp[-1].ftype, yyvsp[0].strval); ;
    break;}
case 110:
#line 429 "xi-grammar.y"
{ yyval.tvar = new TName(yyvsp[-2].type, yyvsp[-1].strval, yyvsp[0].strval); ;
    break;}
case 111:
#line 433 "xi-grammar.y"
{ yyval.tvarlist = new TVarList(yyvsp[0].tvar); ;
    break;}
case 112:
#line 435 "xi-grammar.y"
{ yyval.tvarlist = new TVarList(yyvsp[-2].tvar, yyvsp[0].tvarlist); ;
    break;}
case 113:
#line 439 "xi-grammar.y"
{ yyval.tvarlist = yyvsp[-1].tvarlist; ;
    break;}
case 114:
#line 443 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); ;
    break;}
case 115:
#line 445 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); ;
    break;}
case 116:
#line 447 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); ;
    break;}
case 117:
#line 449 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); ;
    break;}
case 118:
#line 451 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].message); yyvsp[0].message->setTemplate(yyval.templat); ;
    break;}
case 119:
#line 455 "xi-grammar.y"
{ yyval.mbrlist = 0; ;
    break;}
case 120:
#line 457 "xi-grammar.y"
{ yyval.mbrlist = yyvsp[-2].mbrlist; ;
    break;}
case 121:
#line 461 "xi-grammar.y"
{ yyval.mbrlist = 0; ;
    break;}
case 122:
#line 463 "xi-grammar.y"
{ yyval.mbrlist = new MemberList(yyvsp[-1].member, yyvsp[0].mbrlist); ;
    break;}
case 123:
#line 467 "xi-grammar.y"
{ yyval.member = yyvsp[-1].entry; ;
    break;}
case 124:
#line 469 "xi-grammar.y"
{ yyval.member = yyvsp[-1].readonly; ;
    break;}
case 125:
#line 471 "xi-grammar.y"
{ yyval.member = yyvsp[-1].readonly; ;
    break;}
case 126:
#line 475 "xi-grammar.y"
{ yyval.entry = new Entry(lineno, yyvsp[-5].intval|yyvsp[-1].intval, yyvsp[-4].type, yyvsp[-3].strval, yyvsp[-2].plist, yyvsp[0].val); ;
    break;}
case 127:
#line 477 "xi-grammar.y"
{ yyval.entry = new Entry(lineno, yyvsp[-2].intval,     0, yyvsp[-1].strval, yyvsp[0].plist,  0); ;
    break;}
case 128:
#line 481 "xi-grammar.y"
{ yyval.type = new BuiltinType("void"); ;
    break;}
case 129:
#line 483 "xi-grammar.y"
{ yyval.type = yyvsp[0].ptype; ;
    break;}
case 130:
#line 487 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 131:
#line 489 "xi-grammar.y"
{ yyval.intval = yyvsp[-1].intval; ;
    break;}
case 132:
#line 493 "xi-grammar.y"
{ yyval.intval = yyvsp[0].intval; ;
    break;}
case 133:
#line 495 "xi-grammar.y"
{ yyval.intval = yyvsp[-2].intval | yyvsp[0].intval; ;
    break;}
case 134:
#line 499 "xi-grammar.y"
{ yyval.intval = STHREADED; ;
    break;}
case 135:
#line 501 "xi-grammar.y"
{ yyval.intval = SSYNC; ;
    break;}
case 136:
#line 503 "xi-grammar.y"
{ yyval.intval = SLOCKED; ;
    break;}
case 137:
#line 505 "xi-grammar.y"
{ yyval.intval = SVIRTUAL; ;
    break;}
case 138:
#line 507 "xi-grammar.y"
{ yyval.intval = SCREATEHERE; ;
    break;}
case 139:
#line 509 "xi-grammar.y"
{ yyval.intval = SCREATEHOME; ;
    break;}
case 140:
#line 513 "xi-grammar.y"
{ yyval.val = new Value(yyvsp[0].strval); ;
    break;}
case 141:
#line 515 "xi-grammar.y"
{ yyval.val = new Value(yyvsp[0].strval); ;
    break;}
case 142:
#line 517 "xi-grammar.y"
{ yyval.val = new Value(yyvsp[0].strval); ;
    break;}
case 143:
#line 521 "xi-grammar.y"
{ yyval.strval = yyvsp[0].strval; ;
    break;}
case 144:
#line 523 "xi-grammar.y"
{  /*Returned only when in_bracket*/
			char *tmp = new char[strlen(yyvsp[-4].strval)+strlen(yyvsp[-2].strval)+strlen(yyvsp[0].strval)+3];
			sprintf(tmp,"%s[%s]%s", yyvsp[-4].strval, yyvsp[-2].strval, yyvsp[0].strval);
			yyval.strval = tmp;
		;
    break;}
case 145:
#line 529 "xi-grammar.y"
{ /*Returned only when in_braces*/
			char *tmp = new char[strlen(yyvsp[-4].strval)+strlen(yyvsp[-2].strval)+strlen(yyvsp[0].strval)+3];
			sprintf(tmp,"%s{%s}%s", yyvsp[-4].strval, yyvsp[-2].strval, yyvsp[0].strval);
			yyval.strval = tmp;
		;
    break;}
case 146:
#line 537 "xi-grammar.y"
{  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			yyval.pname = new Parameter(lineno, yyvsp[-2].type,yyvsp[-1].strval);
		;
    break;}
case 147:
#line 544 "xi-grammar.y"
{ yyval.pname = new Parameter(lineno, yyvsp[0].type);;
    break;}
case 148:
#line 546 "xi-grammar.y"
{ yyval.pname = new Parameter(lineno, yyvsp[-1].type,yyvsp[0].strval);;
    break;}
case 149:
#line 548 "xi-grammar.y"
{ yyval.pname = new Parameter(lineno, yyvsp[-3].type,yyvsp[-2].strval,0,yyvsp[0].val);;
    break;}
case 150:
#line 550 "xi-grammar.y"
{ /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			yyval.pname = new Parameter(lineno, yyvsp[-2].pname->getType(), yyvsp[-2].pname->getName() ,yyvsp[-1].strval);
		;
    break;}
case 151:
#line 557 "xi-grammar.y"
{ yyval.plist = new ParamList(yyvsp[0].pname); ;
    break;}
case 152:
#line 559 "xi-grammar.y"
{ yyval.plist = new ParamList(yyvsp[-2].pname,yyvsp[0].plist); ;
    break;}
case 153:
#line 563 "xi-grammar.y"
{ yyval.plist = yyvsp[-1].plist; ;
    break;}
case 154:
#line 565 "xi-grammar.y"
{ yyval.plist = 0; ;
    break;}
case 155:
#line 569 "xi-grammar.y"
{ yyval.val = 0; ;
    break;}
case 156:
#line 571 "xi-grammar.y"
{ yyval.val = new Value(yyvsp[0].strval); ;
    break;}
case 157:
#line 575 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 158:
#line 577 "xi-grammar.y"
{ if(strcmp(yyvsp[0].strval, "0")) { yyerror("pure virtual must '=0'"); exit(1); }
		  yyval.intval = SPURE; 
		;
    break;}
}
   /* the action file gets copied in in place of this dollarsign */
#line 542 "/usr/share/bison.simple"

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
#line 581 "xi-grammar.y"

void yyerror(const char *mesg)
{
  cout << cur_file<<":"<<lineno<<": Charmxi syntax error> " << mesg << endl;
  // return 0;
}
