
/*  A Bison parser, made from xi-grammar.y
    by GNU Bison version 1.28  */

#define YYBISON 1  /* Identify Bison output.  */

#define	MODULE	257
#define	MAINMODULE	258
#define	EXTERN	259
#define	READONLY	260
#define	INITCALL	261
#define	PUPABLE	262
#define	CHARE	263
#define	MAINCHARE	264
#define	GROUP	265
#define	NODEGROUP	266
#define	ARRAY	267
#define	MESSAGE	268
#define	CLASS	269
#define	STACKSIZE	270
#define	THREADED	271
#define	TEMPLATE	272
#define	SYNC	273
#define	EXCLUSIVE	274
#define	IMMEDIATE	275
#define	VIRTUAL	276
#define	MIGRATABLE	277
#define	CREATEHERE	278
#define	CREATEHOME	279
#define	NOKEEP	280
#define	VOID	281
#define	CONST	282
#define	PACKED	283
#define	VARSIZE	284
#define	ENTRY	285
#define	FOR	286
#define	FORALL	287
#define	WHILE	288
#define	WHEN	289
#define	OVERLAP	290
#define	ATOMIC	291
#define	FORWARD	292
#define	IF	293
#define	ELSE	294
#define	CONNECT	295
#define	PUBLISHES	296
#define	IDENT	297
#define	NUMBER	298
#define	LITERAL	299
#define	CPROGRAM	300
#define	INT	301
#define	LONG	302
#define	SHORT	303
#define	CHAR	304
#define	FLOAT	305
#define	DOUBLE	306
#define	UNSIGNED	307

#line 2 "xi-grammar.y"

#include <iostream.h>
#include "xi-symbol.h"
#include "EToken.h"
extern int yylex (void) ;
void yyerror(const char *);
extern unsigned int lineno;
extern int in_bracket,in_braces,in_int_expr;
extern TList<Entry *> *connectEntries;
ModuleList *modlist;


#line 15 "xi-grammar.y"
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
  char *strval;
  int intval;
  Chare::attrib_t cattr;
  SdagConstruct *sc;
} YYSTYPE;
#include <stdio.h>

#ifndef __cplusplus
#ifndef __STDC__
#define const
#endif
#endif



#define	YYFINAL		429
#define	YYFLAG		-32768
#define	YYNTBASE	68

#define YYTRANSLATE(x) ((unsigned)(x) <= 307 ? yytranslate[x] : 157)

static const char yytranslate[] = {     0,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,    64,     2,    62,
    63,    61,     2,    58,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,    55,    54,    59,
    67,    60,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
    65,     2,    66,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,    56,     2,    57,     2,     2,     2,     2,     2,
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
    37,    38,    39,    40,    41,    42,    43,    44,    45,    46,
    47,    48,    49,    50,    51,    52,    53
};

#if YYDEBUG != 0
static const short yyprhs[] = {     0,
     0,     2,     3,     6,     7,     9,    10,    12,    14,    16,
    21,    25,    29,    31,    36,    37,    40,    46,    49,    52,
    56,    59,    62,    65,    68,    71,    73,    75,    77,    79,
    83,    84,    86,    87,    91,    93,    95,    97,    99,   102,
   105,   108,   111,   114,   116,   118,   121,   123,   126,   129,
   131,   133,   136,   139,   142,   151,   153,   155,   157,   159,
   162,   165,   167,   169,   173,   174,   177,   182,   188,   189,
   191,   192,   196,   198,   202,   204,   206,   207,   211,   213,
   217,   219,   225,   227,   230,   234,   241,   242,   245,   247,
   251,   257,   263,   269,   275,   280,   284,   290,   296,   302,
   308,   314,   320,   325,   333,   334,   337,   338,   341,   344,
   348,   351,   355,   357,   361,   366,   369,   372,   375,   378,
   381,   383,   388,   389,   392,   395,   398,   401,   405,   409,
   416,   418,   422,   425,   427,   435,   441,   443,   445,   446,
   450,   452,   456,   458,   460,   462,   464,   466,   468,   470,
   472,   474,   476,   477,   479,   485,   491,   497,   502,   506,
   508,   510,   512,   515,   520,   524,   526,   530,   534,   537,
   538,   542,   543,   545,   549,   551,   554,   556,   559,   560,
   565,   567,   571,   577,   586,   591,   595,   601,   606,   618,
   628,   641,   656,   663,   672,   678,   686,   690,   691,   694,
   699,   701,   705,   707,   709,   712,   718,   720,   724,   726
};

static const short yyrhs[] = {    69,
     0,     0,    74,    69,     0,     0,     5,     0,     0,    54,
     0,    43,     0,    43,     0,    73,    55,    55,    43,     0,
     3,    72,    75,     0,     4,    72,    75,     0,    54,     0,
    56,    76,    57,    71,     0,     0,    77,    76,     0,    70,
    56,    76,    57,    71,     0,    70,    74,     0,    70,   125,
     0,    70,   104,    54,     0,    70,   107,     0,    70,   108,
     0,    70,   109,     0,    70,   111,     0,    70,   122,     0,
    89,     0,    44,     0,    45,     0,    78,     0,    78,    58,
    79,     0,     0,    79,     0,     0,    59,    80,    60,     0,
    47,     0,    48,     0,    49,     0,    50,     0,    53,    47,
     0,    53,    48,     0,    53,    49,     0,    53,    50,     0,
    48,    48,     0,    51,     0,    52,     0,    48,    52,     0,
    27,     0,    72,    81,     0,    73,    81,     0,    82,     0,
    84,     0,    85,    61,     0,    86,    61,     0,    87,    61,
     0,    89,    62,    61,    72,    63,    62,   140,    63,     0,
    85,     0,    86,     0,    87,     0,    88,     0,    89,    64,
     0,    28,    89,     0,    44,     0,    73,     0,    65,    90,
    66,     0,     0,    91,    92,     0,     6,    89,    73,    92,
     0,     6,    14,    85,    61,    72,     0,     0,    27,     0,
     0,    65,    97,    66,     0,    98,     0,    98,    58,    97,
     0,    29,     0,    30,     0,     0,    65,   100,    66,     0,
   101,     0,   101,    58,   100,     0,    23,     0,    89,    72,
    65,    66,    54,     0,   102,     0,   102,   103,     0,    14,
    96,    83,     0,    14,    96,    83,    56,   103,    57,     0,
     0,    55,   106,     0,    83,     0,    83,    58,   106,     0,
     9,    99,    83,   105,   123,     0,    10,    99,    83,   105,
   123,     0,    11,    99,    83,   105,   123,     0,    12,    99,
    83,   105,   123,     0,    65,    44,    72,    66,     0,    65,
    72,    66,     0,    13,   110,    83,   105,   123,     0,     9,
    99,    72,   105,   123,     0,    10,    99,    72,   105,   123,
     0,    11,    99,    72,   105,   123,     0,    12,    99,    72,
   105,   123,     0,    13,   110,    72,   105,   123,     0,    14,
    96,    72,    54,     0,    14,    96,    72,    56,   103,    57,
    54,     0,     0,    67,    89,     0,     0,    67,    44,     0,
    67,    45,     0,    15,    72,   117,     0,    88,   118,     0,
    89,    72,   118,     0,   119,     0,   119,    58,   120,     0,
    18,    59,   120,    60,     0,   121,   112,     0,   121,   113,
     0,   121,   114,     0,   121,   115,     0,   121,   116,     0,
    54,     0,    56,   124,    57,    71,     0,     0,   128,   124,
     0,    93,    54,     0,    94,    54,     0,   126,    54,     0,
     8,   127,    54,     0,     7,    95,    73,     0,     7,    95,
    73,    62,    95,    63,     0,    73,     0,    73,    58,   127,
     0,   129,    54,     0,   125,     0,    31,   131,   130,    72,
   141,   142,   143,     0,    31,   131,    72,   141,   143,     0,
    27,     0,    86,     0,     0,    65,   132,    66,     0,   133,
     0,   133,    58,   132,     0,    17,     0,    19,     0,    20,
     0,    24,     0,    25,     0,    26,     0,    21,     0,    45,
     0,    44,     0,    73,     0,     0,    46,     0,    46,    65,
   135,    66,   135,     0,    46,    56,   135,    57,   135,     0,
    46,    62,   135,    63,   135,     0,    62,   135,    63,   135,
     0,    89,    72,    65,     0,    56,     0,    57,     0,    89,
     0,    89,    72,     0,    89,    72,    67,   134,     0,   136,
   135,    66,     0,   139,     0,   139,    58,   140,     0,    62,
   140,    63,     0,    62,    63,     0,     0,    16,    67,    44,
     0,     0,   148,     0,    56,   144,    57,     0,   148,     0,
   148,   144,     0,   148,     0,   148,   144,     0,     0,    42,
    62,   147,    63,     0,    43,     0,    43,    58,   147,     0,
    37,   137,   135,   138,   146,     0,    41,    62,    43,   141,
    63,   137,   135,    57,     0,    35,   154,    56,    57,     0,
    35,   154,   148,     0,    35,   154,    56,   144,    57,     0,
    36,    56,   145,    57,     0,    32,   152,   135,    54,   135,
    54,   135,   151,    56,   144,    57,     0,    32,   152,   135,
    54,   135,    54,   135,   151,   148,     0,    33,    65,    43,
    66,   152,   135,    55,   135,    58,   135,   151,   148,     0,
    33,    65,    43,    66,   152,   135,    55,   135,    58,   135,
   151,    56,   144,    57,     0,    39,   152,   135,   151,   148,
   149,     0,    39,   152,   135,   151,    56,   144,    57,   149,
     0,    34,   152,   135,   151,   148,     0,    34,   152,   135,
   151,    56,   144,    57,     0,    38,   150,    54,     0,     0,
    40,   148,     0,    40,    56,   144,    57,     0,    43,     0,
    43,    58,   150,     0,    63,     0,    62,     0,    43,   141,
     0,    43,   155,   135,   156,   141,     0,   153,     0,   153,
    58,   154,     0,    65,     0,    66,     0
};

#endif

#if YYDEBUG != 0
static const short yyrline[] = { 0,
   122,   126,   130,   134,   136,   140,   142,   146,   150,   152,
   160,   164,   171,   173,   177,   179,   183,   185,   187,   189,
   191,   193,   195,   197,   199,   203,   205,   207,   211,   213,
   217,   219,   223,   225,   229,   231,   233,   235,   237,   239,
   241,   243,   245,   247,   249,   251,   253,   257,   258,   260,
   262,   266,   270,   272,   276,   280,   282,   284,   286,   288,
   291,   295,   297,   301,   305,   307,   311,   315,   319,   321,
   325,   327,   337,   339,   343,   345,   349,   351,   355,   357,
   361,   365,   369,   371,   375,   377,   381,   383,   387,   389,
   393,   395,   399,   403,   407,   413,   417,   421,   423,   427,
   431,   435,   439,   441,   445,   447,   451,   453,   455,   459,
   461,   463,   467,   469,   473,   477,   479,   481,   483,   485,
   489,   491,   495,   514,   518,   520,   522,   524,   528,   530,
   534,   536,   540,   542,   546,   557,   570,   572,   576,   578,
   582,   584,   588,   590,   592,   594,   596,   598,   600,   604,
   606,   608,   612,   614,   616,   622,   628,   634,   642,   649,
   657,   664,   666,   668,   670,   677,   679,   683,   685,   689,
   691,   695,   697,   699,   703,   705,   709,   711,   715,   717,
   721,   723,   727,   729,   743,   745,   747,   749,   751,   754,
   757,   760,   763,   765,   767,   769,   771,   775,   777,   779,
   782,   784,   788,   792,   796,   804,   812,   814,   818,   821
};
#endif


#if YYDEBUG != 0 || defined (YYERROR_VERBOSE)

static const char * const yytname[] = {   "$","error","$undefined.","MODULE",
"MAINMODULE","EXTERN","READONLY","INITCALL","PUPABLE","CHARE","MAINCHARE","GROUP",
"NODEGROUP","ARRAY","MESSAGE","CLASS","STACKSIZE","THREADED","TEMPLATE","SYNC",
"EXCLUSIVE","IMMEDIATE","VIRTUAL","MIGRATABLE","CREATEHERE","CREATEHOME","NOKEEP",
"VOID","CONST","PACKED","VARSIZE","ENTRY","FOR","FORALL","WHILE","WHEN","OVERLAP",
"ATOMIC","FORWARD","IF","ELSE","CONNECT","PUBLISHES","IDENT","NUMBER","LITERAL",
"CPROGRAM","INT","LONG","SHORT","CHAR","FLOAT","DOUBLE","UNSIGNED","';'","':'",
"'{'","'}'","','","'<'","'>'","'*'","'('","')'","'&'","'['","']'","'='","File",
"ModuleEList","OptExtern","OptSemiColon","Name","QualName","Module","ConstructEList",
"ConstructList","Construct","TParam","TParamList","TParamEList","OptTParams",
"BuiltinType","NamedType","QualNamedType","SimpleType","OnePtrType","PtrType",
"FuncType","Type","ArrayDim","Dim","DimList","Readonly","ReadonlyMsg","OptVoid",
"MAttribs","MAttribList","MAttrib","CAttribs","CAttribList","CAttrib","Var",
"VarList","Message","OptBaseList","BaseList","Chare","Group","NodeGroup","ArrayIndexType",
"Array","TChare","TGroup","TNodeGroup","TArray","TMessage","OptTypeInit","OptNameInit",
"TVar","TVarList","TemplateSpec","Template","MemberEList","MemberList","NonEntryMember",
"InitCall","PUPableClass","Member","Entry","EReturn","EAttribs","EAttribList",
"EAttrib","DefaultParameter","CCode","ParamBracketStart","ParamBraceStart","ParamBraceEnd",
"Parameter","ParamList","EParameters","OptStackSize","OptSdagCode","Slist","Olist",
"OptPubList","PublishesList","SingleConstruct","HasElse","ForwardList","EndIntExpr",
"StartIntExpr","SEntry","SEntryList","SParamBracketStart","SParamBracketEnd", NULL
};
#endif

static const short yyr1[] = {     0,
    68,    69,    69,    70,    70,    71,    71,    72,    73,    73,
    74,    74,    75,    75,    76,    76,    77,    77,    77,    77,
    77,    77,    77,    77,    77,    78,    78,    78,    79,    79,
    80,    80,    81,    81,    82,    82,    82,    82,    82,    82,
    82,    82,    82,    82,    82,    82,    82,    83,    84,    85,
    85,    86,    87,    87,    88,    89,    89,    89,    89,    89,
    89,    90,    90,    91,    92,    92,    93,    94,    95,    95,
    96,    96,    97,    97,    98,    98,    99,    99,   100,   100,
   101,   102,   103,   103,   104,   104,   105,   105,   106,   106,
   107,   107,   108,   109,   110,   110,   111,   112,   112,   113,
   114,   115,   116,   116,   117,   117,   118,   118,   118,   119,
   119,   119,   120,   120,   121,   122,   122,   122,   122,   122,
   123,   123,   124,   124,   125,   125,   125,   125,   126,   126,
   127,   127,   128,   128,   129,   129,   130,   130,   131,   131,
   132,   132,   133,   133,   133,   133,   133,   133,   133,   134,
   134,   134,   135,   135,   135,   135,   135,   135,   136,   137,
   138,   139,   139,   139,   139,   140,   140,   141,   141,   142,
   142,   143,   143,   143,   144,   144,   145,   145,   146,   146,
   147,   147,   148,   148,   148,   148,   148,   148,   148,   148,
   148,   148,   148,   148,   148,   148,   148,   149,   149,   149,
   150,   150,   151,   152,   153,   153,   154,   154,   155,   156
};

static const short yyr2[] = {     0,
     1,     0,     2,     0,     1,     0,     1,     1,     1,     4,
     3,     3,     1,     4,     0,     2,     5,     2,     2,     3,
     2,     2,     2,     2,     2,     1,     1,     1,     1,     3,
     0,     1,     0,     3,     1,     1,     1,     1,     2,     2,
     2,     2,     2,     1,     1,     2,     1,     2,     2,     1,
     1,     2,     2,     2,     8,     1,     1,     1,     1,     2,
     2,     1,     1,     3,     0,     2,     4,     5,     0,     1,
     0,     3,     1,     3,     1,     1,     0,     3,     1,     3,
     1,     5,     1,     2,     3,     6,     0,     2,     1,     3,
     5,     5,     5,     5,     4,     3,     5,     5,     5,     5,
     5,     5,     4,     7,     0,     2,     0,     2,     2,     3,
     2,     3,     1,     3,     4,     2,     2,     2,     2,     2,
     1,     4,     0,     2,     2,     2,     2,     3,     3,     6,
     1,     3,     2,     1,     7,     5,     1,     1,     0,     3,
     1,     3,     1,     1,     1,     1,     1,     1,     1,     1,
     1,     1,     0,     1,     5,     5,     5,     4,     3,     1,
     1,     1,     2,     4,     3,     1,     3,     3,     2,     0,
     3,     0,     1,     3,     1,     2,     1,     2,     0,     4,
     1,     3,     5,     8,     4,     3,     5,     4,    11,     9,
    12,    14,     6,     8,     5,     7,     3,     0,     2,     4,
     1,     3,     1,     1,     2,     5,     1,     3,     1,     1
};

static const short yydefact[] = {     2,
     0,     0,     1,     2,     8,     0,     0,     3,    13,     4,
    11,    12,     5,     0,     0,     4,     0,    69,     0,    77,
    77,    77,    77,     0,    71,     0,     4,    18,     0,     0,
     0,    21,    22,    23,    24,     0,    25,    19,     0,     6,
    16,     0,    47,     0,     9,    35,    36,    37,    38,    44,
    45,     0,    33,    50,    51,    56,    57,    58,    59,     0,
    70,     0,   131,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,   125,   126,    20,    77,    77,
    77,    77,     0,    71,   116,   117,   118,   119,   120,   127,
     7,    14,     0,    61,    43,    46,    39,    40,    41,    42,
     0,    31,    49,    52,    53,    54,     0,    60,    65,   129,
     0,   128,    81,     0,    79,    33,    87,    87,    87,    87,
     0,     0,    87,    75,    76,     0,    73,    85,     0,    59,
     0,   113,     0,     6,     0,     0,     0,     0,     0,     0,
     0,     0,    27,    28,    29,    32,     0,    26,     0,     0,
    65,    67,    69,   132,    78,     0,    48,     0,     0,     0,
     0,     0,     0,    96,     0,    72,     0,     0,   105,     0,
   111,   107,     0,   115,    17,    87,    87,    87,    87,    87,
     0,    68,    10,     0,    34,     0,    62,    63,     0,    66,
     0,    80,    89,    88,   121,   123,    91,    92,    93,    94,
    95,    97,    74,     0,    83,     0,     0,   110,   108,   109,
   112,   114,     0,     0,     0,     0,     0,   103,     0,    30,
     0,    64,   130,     0,   139,     0,   134,   123,     0,     0,
    84,    86,   106,    98,    99,   100,   101,   102,     0,     0,
    90,     0,     0,     6,   124,   133,     0,     0,   162,   153,
   166,     0,   143,   144,   145,   149,   146,   147,   148,     0,
   141,    47,     9,     0,     0,   138,     0,   122,     0,   104,
   163,   154,   153,     0,     0,    55,   140,     0,     0,   172,
     0,    82,   159,     0,   153,   153,   153,     0,   165,   167,
   142,   169,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,   136,   173,   170,   151,   150,   152,   164,
     0,     0,     0,   153,   168,   204,   153,     0,   153,     0,
   207,     0,     0,   160,   153,   201,     0,   153,     0,     0,
   175,     0,   172,   153,   153,   153,   158,     0,     0,     0,
   209,   205,   153,     0,     0,   186,     0,   177,     0,     0,
   197,     0,     0,   174,   176,     0,   135,   156,   157,   155,
   153,     0,   203,     0,     0,   208,   185,     0,   188,   178,
   161,   179,   202,     0,     0,   171,     0,   153,     0,   195,
   210,     0,   187,     0,   183,     0,   198,     0,   153,     0,
     0,   206,     0,     0,     0,   193,   153,     0,   153,   196,
   181,     0,   198,     0,   199,     0,     0,     0,     0,   180,
   194,     0,   184,     0,   190,   153,   182,   200,     0,     0,
   189,     0,     0,   191,     0,   192,     0,     0,     0
};

static const short yydefgoto[] = {   427,
     3,    14,    92,   116,    53,     4,    11,    15,    16,   145,
   146,   147,   103,    54,   193,    55,    56,    57,    58,    59,
   204,   189,   151,   152,    29,    30,    62,    73,   126,   127,
    66,   114,   115,   205,   206,    31,   159,   194,    32,    33,
    34,    71,    35,    85,    86,    87,    88,    89,   208,   171,
   132,   133,    36,    37,   197,   226,   227,    39,    64,   228,
   229,   267,   243,   260,   261,   310,   274,   250,   325,   372,
   251,   252,   280,   333,   304,   330,   347,   385,   402,   331,
   396,   327,   364,   317,   321,   322,   343,   382
};

static const short yypact[] = {   128,
   -30,   -30,-32768,   128,-32768,    48,    48,-32768,-32768,     6,
-32768,-32768,-32768,   178,   -21,     6,   180,    -8,     4,    25,
    25,    25,    25,    28,    38,    21,     6,-32768,    20,    65,
    72,-32768,-32768,-32768,-32768,   155,-32768,-32768,    87,    89,
-32768,   168,-32768,   264,-32768,-32768,   -27,-32768,-32768,-32768,
-32768,   210,   -11,-32768,-32768,    45,    88,    99,-32768,   -31,
-32768,     4,     1,   118,   139,   -30,   -30,   -30,   -30,   108,
   -30,   125,   -30,   225,   116,-32768,-32768,-32768,    25,    25,
    25,    25,    28,    38,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,   122,    47,-32768,-32768,-32768,-32768,-32768,-32768,
   124,   253,-32768,-32768,-32768,-32768,   136,-32768,   -13,    15,
     4,-32768,-32768,   138,   148,   153,   158,   158,   158,   158,
   -30,   156,   158,-32768,-32768,   159,   166,   179,   -30,    -5,
   -23,   186,   154,    89,   -30,   -30,   -30,   -30,   -30,   -30,
   -30,   202,-32768,-32768,   188,-32768,   209,    47,   -30,   114,
   197,-32768,    -8,-32768,-32768,   139,-32768,   -30,    58,    58,
    58,    58,   204,-32768,    58,-32768,   125,   264,   215,   165,
-32768,   223,   225,-32768,-32768,   158,   158,   158,   158,   158,
    68,-32768,-32768,   253,-32768,   208,-32768,   238,   228,-32768,
   232,-32768,   251,-32768,-32768,     9,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,   -23,   264,   261,   264,-32768,-32768,-32768,
-32768,-32768,    58,    58,    58,    58,    58,-32768,   264,-32768,
   248,-32768,-32768,   -30,   254,   263,-32768,     9,   267,   257,
-32768,-32768,    47,-32768,-32768,-32768,-32768,-32768,   266,   264,
-32768,   230,   281,    89,-32768,-32768,   259,   272,   -23,   -24,
   269,   280,-32768,-32768,-32768,-32768,-32768,-32768,-32768,   287,
   297,   314,   294,   296,    45,-32768,   -30,-32768,   305,-32768,
    62,     2,   -24,   315,   264,-32768,-32768,   230,   236,   313,
   296,-32768,-32768,    52,   -24,   -24,   -24,   316,-32768,-32768,
-32768,-32768,   319,   321,   340,   321,   341,   330,   351,   366,
   321,   346,   407,-32768,-32768,   394,-32768,-32768,   238,-32768,
   374,   349,   367,   -24,-32768,-32768,   -24,   392,   -24,     3,
   376,   329,   407,-32768,   -24,   378,   384,   -24,   406,   393,
   407,   385,   313,   -24,   -24,   -24,-32768,   397,   387,   395,
-32768,-32768,   -24,   341,   303,-32768,   398,   407,   399,   366,
-32768,   395,   296,-32768,-32768,   410,-32768,-32768,-32768,-32768,
   -24,   321,-32768,   339,   396,-32768,-32768,   400,-32768,-32768,
-32768,   417,-32768,   355,   401,-32768,   409,   -24,   407,-32768,
-32768,   296,-32768,   403,-32768,   407,   420,   351,   -24,   411,
   404,-32768,   424,   412,   365,-32768,   -24,   395,   -24,-32768,
   413,   405,   420,   407,-32768,   415,   381,   416,   424,-32768,
-32768,   418,-32768,   407,-32768,   -24,-32768,-32768,   419,   395,
-32768,   391,   407,-32768,   421,-32768,   470,   473,-32768
};

static const short yypgoto[] = {-32768,
   475,-32768,  -129,    -1,   -17,   463,   474,     7,-32768,-32768,
   298,-32768,   364,-32768,   170,-32768,   -38,   240,-32768,   -68,
   -14,-32768,-32768,   333,-32768,-32768,   332,   402,   320,-32768,
     5,   334,-32768,-32768,  -195,-32768,    -2,   265,-32768,-32768,
-32768,   408,-32768,-32768,-32768,-32768,-32768,-32768,-32768,   322,
-32768,   323,-32768,-32768,   -15,   260,   478,-32768,   382,-32768,
-32768,-32768,-32768,   217,-32768,-32768,  -236,-32768,   109,-32768,
-32768,  -218,  -274,-32768,   167,  -313,-32768,-32768,    90,  -251,
    95,   151,  -344,  -287,-32768,   160,-32768,-32768
};


#define	YYLAST		504


static const short yytable[] = {     6,
     7,    63,    60,    93,   175,   130,   306,   374,   319,   231,
    13,    45,     5,   328,    17,    18,    19,   355,    61,     5,
    95,   272,    41,   239,    96,    67,    68,    69,   305,    94,
   107,   368,   108,    75,   370,    40,   288,   273,   107,   225,
   108,   101,   109,   101,   110,   342,    45,   102,   311,   312,
   313,   150,  -107,   407,  -107,   101,   290,   285,   111,   131,
   293,   170,   -15,   286,   279,   391,   287,   341,   122,   101,
   346,   348,   394,    76,   378,   422,   153,   337,   375,    74,
   338,   305,   340,   135,   136,   137,   138,   148,   349,    65,
   412,   352,    70,    63,    45,   307,   308,   358,   359,   360,
   419,     9,    72,    10,   130,   104,   365,   392,   107,   425,
   108,   195,   380,   196,   268,   160,   161,   162,    77,   163,
   165,   218,   387,   219,   377,    78,   283,   169,   284,   172,
     1,     2,   188,   176,   177,   178,   179,   180,   181,   182,
    90,   390,    91,   405,   198,   199,   200,   186,   105,   202,
     5,   121,   398,   124,   125,   415,    45,   187,   131,   106,
   406,   113,   408,    79,    80,    81,    82,    83,    84,   148,
   424,   112,   134,   213,   214,   215,   216,   217,   142,   420,
     1,     2,   141,    17,    18,    19,    20,    21,    22,    23,
    24,    25,   233,    42,    43,    26,   149,   234,   235,   236,
   237,   238,   230,   155,   265,   156,    43,    44,   209,   210,
    45,   102,   158,   174,    46,    47,    48,    49,    50,    51,
    52,   164,    45,   167,   166,   249,    46,    47,    48,    49,
    50,    51,    52,    27,   168,   117,   118,   119,   120,   129,
   123,   264,   128,   173,   183,   184,   253,   271,   254,   255,
   256,    43,    44,   257,   258,   259,    97,    98,    99,   100,
   249,   150,    43,    44,   249,   281,   309,    45,   185,   201,
   221,    46,    47,    48,    49,    50,    51,    52,    45,    43,
    44,   207,    46,    47,    48,    49,    50,    51,    52,   170,
    43,    44,   101,   222,   223,    45,   143,   144,   292,    46,
    47,    48,    49,    50,    51,    52,    45,   262,   224,   240,
    46,    47,    48,    49,    50,    51,    52,   232,   242,   244,
   246,   247,   248,   263,   269,   270,   275,    46,    47,    48,
    49,    50,    51,    52,   294,   295,   296,   297,   298,   299,
   300,   301,   276,   302,   294,   295,   296,   297,   298,   299,
   300,   301,   277,   302,   278,    -8,  -137,   279,   282,   367,
   294,   295,   296,   297,   298,   299,   300,   301,   303,   302,
   294,   295,   296,   297,   298,   299,   300,   301,   314,   302,
   289,   315,   316,   320,   345,   323,   294,   295,   296,   297,
   298,   299,   300,   301,   379,   302,   294,   295,   296,   297,
   298,   299,   300,   301,   318,   302,   324,   329,   326,   332,
   386,   335,   294,   295,   296,   297,   298,   299,   300,   301,
   404,   302,   294,   295,   296,   297,   298,   299,   300,   301,
   334,   302,   336,   344,   339,   350,   414,   351,   294,   295,
   296,   297,   298,   299,   300,   301,   423,   302,   353,   354,
   361,   356,   362,   376,   369,   371,   383,   363,   384,   395,
   400,   381,   389,   388,   393,   399,   401,   410,   403,   428,
   409,   413,   429,   416,   418,   421,    28,   426,     8,   157,
    12,   220,   266,   190,   191,   140,   203,   245,   241,   192,
   139,    38,   154,   211,   291,   212,   397,   411,   417,   357,
   373,     0,     0,   366
};

static const short yycheck[] = {     1,
     2,    19,    17,    42,   134,    74,   281,   352,   296,   205,
     5,    43,    43,   301,     6,     7,     8,   331,    27,    43,
    48,    46,    16,   219,    52,    21,    22,    23,   280,    44,
    62,   345,    64,    27,   348,    57,   273,    62,    62,    31,
    64,    55,    60,    55,    62,   320,    43,    59,   285,   286,
   287,    65,    58,   398,    60,    55,   275,    56,    58,    74,
   279,    67,    57,    62,    62,   379,    65,    65,    70,    55,
   322,   323,   386,    54,   362,   420,    62,   314,   353,    59,
   317,   333,   319,    79,    80,    81,    82,   102,   325,    65,
   404,   328,    65,   111,    43,    44,    45,   334,   335,   336,
   414,    54,    65,    56,   173,    61,   343,   382,    62,   423,
    64,    54,   364,    56,   244,   118,   119,   120,    54,   121,
   123,    54,   374,    56,   361,    54,    65,   129,    67,   131,
     3,     4,   150,   135,   136,   137,   138,   139,   140,   141,
    54,   378,    54,   395,   160,   161,   162,   149,    61,   165,
    43,    44,   389,    29,    30,   407,    43,    44,   173,    61,
   397,    23,   399,     9,    10,    11,    12,    13,    14,   184,
   422,    54,    57,   176,   177,   178,   179,   180,    55,   416,
     3,     4,    61,     6,     7,     8,     9,    10,    11,    12,
    13,    14,   207,    14,    27,    18,    61,   213,   214,   215,
   216,   217,   204,    66,   243,    58,    27,    28,    44,    45,
    43,    59,    55,    60,    47,    48,    49,    50,    51,    52,
    53,    66,    43,    58,    66,   240,    47,    48,    49,    50,
    51,    52,    53,    56,    56,    66,    67,    68,    69,    15,
    71,   243,    73,    58,    43,    58,    17,   249,    19,    20,
    21,    27,    28,    24,    25,    26,    47,    48,    49,    50,
   275,    65,    27,    28,   279,   267,   284,    43,    60,    66,
    63,    47,    48,    49,    50,    51,    52,    53,    43,    27,
    28,    67,    47,    48,    49,    50,    51,    52,    53,    67,
    27,    28,    55,    66,    63,    43,    44,    45,    63,    47,
    48,    49,    50,    51,    52,    53,    43,    27,    58,    62,
    47,    48,    49,    50,    51,    52,    53,    57,    65,    57,
    54,    65,    57,    43,    66,    54,    58,    47,    48,    49,
    50,    51,    52,    53,    32,    33,    34,    35,    36,    37,
    38,    39,    63,    41,    32,    33,    34,    35,    36,    37,
    38,    39,    66,    41,    58,    62,    43,    62,    54,    57,
    32,    33,    34,    35,    36,    37,    38,    39,    56,    41,
    32,    33,    34,    35,    36,    37,    38,    39,    63,    41,
    66,    63,    62,    43,    56,    56,    32,    33,    34,    35,
    36,    37,    38,    39,    56,    41,    32,    33,    34,    35,
    36,    37,    38,    39,    65,    41,    56,    62,    43,    16,
    56,    63,    32,    33,    34,    35,    36,    37,    38,    39,
    56,    41,    32,    33,    34,    35,    36,    37,    38,    39,
    57,    41,    66,    58,    43,    58,    56,    54,    32,    33,
    34,    35,    36,    37,    38,    39,    56,    41,    43,    57,
    54,    67,    66,    44,    57,    57,    57,    63,    42,    40,
    57,    66,    54,    63,    62,    55,    43,    63,    57,     0,
    58,    57,     0,    58,    57,    57,    14,    57,     4,   116,
     7,   184,   243,   151,   153,    84,   167,   228,   224,   156,
    83,    14,   111,   172,   278,   173,   388,   403,   409,   333,
   350,    -1,    -1,   344
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
#line 123 "xi-grammar.y"
{ yyval.modlist = yyvsp[0].modlist; modlist = yyvsp[0].modlist; ;
    break;}
case 2:
#line 127 "xi-grammar.y"
{ 
		  yyval.modlist = 0; 
		;
    break;}
case 3:
#line 131 "xi-grammar.y"
{ yyval.modlist = new ModuleList(lineno, yyvsp[-1].module, yyvsp[0].modlist); ;
    break;}
case 4:
#line 135 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 5:
#line 137 "xi-grammar.y"
{ yyval.intval = 1; ;
    break;}
case 6:
#line 141 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 7:
#line 143 "xi-grammar.y"
{ yyval.intval = 1; ;
    break;}
case 8:
#line 147 "xi-grammar.y"
{ yyval.strval = yyvsp[0].strval; ;
    break;}
case 9:
#line 151 "xi-grammar.y"
{ yyval.strval = yyvsp[0].strval; ;
    break;}
case 10:
#line 153 "xi-grammar.y"
{
		  char *tmp = new char[strlen(yyvsp[-3].strval)+strlen(yyvsp[0].strval)+3];
		  sprintf(tmp,"%s::%s", yyvsp[-3].strval, yyvsp[0].strval);
		  yyval.strval = tmp;
		;
    break;}
case 11:
#line 161 "xi-grammar.y"
{ 
		    yyval.module = new Module(lineno, yyvsp[-1].strval, yyvsp[0].conslist); 
		;
    break;}
case 12:
#line 165 "xi-grammar.y"
{  
		    yyval.module = new Module(lineno, yyvsp[-1].strval, yyvsp[0].conslist); 
		    yyval.module->setMain();
		;
    break;}
case 13:
#line 172 "xi-grammar.y"
{ yyval.conslist = 0; ;
    break;}
case 14:
#line 174 "xi-grammar.y"
{ yyval.conslist = yyvsp[-2].conslist; ;
    break;}
case 15:
#line 178 "xi-grammar.y"
{ yyval.conslist = 0; ;
    break;}
case 16:
#line 180 "xi-grammar.y"
{ yyval.conslist = new ConstructList(lineno, yyvsp[-1].construct, yyvsp[0].conslist); ;
    break;}
case 17:
#line 184 "xi-grammar.y"
{ if(yyvsp[-2].conslist) yyvsp[-2].conslist->setExtern(yyvsp[-4].intval); yyval.construct = yyvsp[-2].conslist; ;
    break;}
case 18:
#line 186 "xi-grammar.y"
{ yyvsp[0].module->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].module; ;
    break;}
case 19:
#line 188 "xi-grammar.y"
{ yyvsp[0].member->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].member; ;
    break;}
case 20:
#line 190 "xi-grammar.y"
{ yyvsp[-1].message->setExtern(yyvsp[-2].intval); yyval.construct = yyvsp[-1].message; ;
    break;}
case 21:
#line 192 "xi-grammar.y"
{ yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; ;
    break;}
case 22:
#line 194 "xi-grammar.y"
{ yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; ;
    break;}
case 23:
#line 196 "xi-grammar.y"
{ yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; ;
    break;}
case 24:
#line 198 "xi-grammar.y"
{ yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; ;
    break;}
case 25:
#line 200 "xi-grammar.y"
{ yyvsp[0].templat->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].templat; ;
    break;}
case 26:
#line 204 "xi-grammar.y"
{ yyval.tparam = new TParamType(yyvsp[0].type); ;
    break;}
case 27:
#line 206 "xi-grammar.y"
{ yyval.tparam = new TParamVal(yyvsp[0].strval); ;
    break;}
case 28:
#line 208 "xi-grammar.y"
{ yyval.tparam = new TParamVal(yyvsp[0].strval); ;
    break;}
case 29:
#line 212 "xi-grammar.y"
{ yyval.tparlist = new TParamList(yyvsp[0].tparam); ;
    break;}
case 30:
#line 214 "xi-grammar.y"
{ yyval.tparlist = new TParamList(yyvsp[-2].tparam, yyvsp[0].tparlist); ;
    break;}
case 31:
#line 218 "xi-grammar.y"
{ yyval.tparlist = 0; ;
    break;}
case 32:
#line 220 "xi-grammar.y"
{ yyval.tparlist = yyvsp[0].tparlist; ;
    break;}
case 33:
#line 224 "xi-grammar.y"
{ yyval.tparlist = 0; ;
    break;}
case 34:
#line 226 "xi-grammar.y"
{ yyval.tparlist = yyvsp[-1].tparlist; ;
    break;}
case 35:
#line 230 "xi-grammar.y"
{ yyval.type = new BuiltinType("int"); ;
    break;}
case 36:
#line 232 "xi-grammar.y"
{ yyval.type = new BuiltinType("long"); ;
    break;}
case 37:
#line 234 "xi-grammar.y"
{ yyval.type = new BuiltinType("short"); ;
    break;}
case 38:
#line 236 "xi-grammar.y"
{ yyval.type = new BuiltinType("char"); ;
    break;}
case 39:
#line 238 "xi-grammar.y"
{ yyval.type = new BuiltinType("unsigned int"); ;
    break;}
case 40:
#line 240 "xi-grammar.y"
{ yyval.type = new BuiltinType("unsigned long"); ;
    break;}
case 41:
#line 242 "xi-grammar.y"
{ yyval.type = new BuiltinType("unsigned short"); ;
    break;}
case 42:
#line 244 "xi-grammar.y"
{ yyval.type = new BuiltinType("unsigned char"); ;
    break;}
case 43:
#line 246 "xi-grammar.y"
{ yyval.type = new BuiltinType("long long"); ;
    break;}
case 44:
#line 248 "xi-grammar.y"
{ yyval.type = new BuiltinType("float"); ;
    break;}
case 45:
#line 250 "xi-grammar.y"
{ yyval.type = new BuiltinType("double"); ;
    break;}
case 46:
#line 252 "xi-grammar.y"
{ yyval.type = new BuiltinType("long double"); ;
    break;}
case 47:
#line 254 "xi-grammar.y"
{ yyval.type = new BuiltinType("void"); ;
    break;}
case 48:
#line 257 "xi-grammar.y"
{ yyval.ntype = new NamedType(yyvsp[-1].strval,yyvsp[0].tparlist); ;
    break;}
case 49:
#line 258 "xi-grammar.y"
{ yyval.ntype = new NamedType(yyvsp[-1].strval,yyvsp[0].tparlist); ;
    break;}
case 50:
#line 261 "xi-grammar.y"
{ yyval.type = yyvsp[0].type; ;
    break;}
case 51:
#line 263 "xi-grammar.y"
{ yyval.type = yyvsp[0].ntype; ;
    break;}
case 52:
#line 267 "xi-grammar.y"
{ yyval.ptype = new PtrType(yyvsp[-1].type); ;
    break;}
case 53:
#line 271 "xi-grammar.y"
{ yyvsp[-1].ptype->indirect(); yyval.ptype = yyvsp[-1].ptype; ;
    break;}
case 54:
#line 273 "xi-grammar.y"
{ yyvsp[-1].ptype->indirect(); yyval.ptype = yyvsp[-1].ptype; ;
    break;}
case 55:
#line 277 "xi-grammar.y"
{ yyval.ftype = new FuncType(yyvsp[-7].type, yyvsp[-4].strval, yyvsp[-1].plist); ;
    break;}
case 56:
#line 281 "xi-grammar.y"
{ yyval.type = yyvsp[0].type; ;
    break;}
case 57:
#line 283 "xi-grammar.y"
{ yyval.type = yyvsp[0].ptype; ;
    break;}
case 58:
#line 285 "xi-grammar.y"
{ yyval.type = yyvsp[0].ptype; ;
    break;}
case 59:
#line 287 "xi-grammar.y"
{ yyval.type = yyvsp[0].ftype; ;
    break;}
case 60:
#line 289 "xi-grammar.y"
{ yyval.type = new ReferenceType(yyvsp[-1].type); ;
    break;}
case 61:
#line 292 "xi-grammar.y"
{ yyval.type = yyvsp[0].type; ;
    break;}
case 62:
#line 296 "xi-grammar.y"
{ yyval.val = new Value(yyvsp[0].strval); ;
    break;}
case 63:
#line 298 "xi-grammar.y"
{ yyval.val = new Value(yyvsp[0].strval); ;
    break;}
case 64:
#line 302 "xi-grammar.y"
{ yyval.val = yyvsp[-1].val; ;
    break;}
case 65:
#line 306 "xi-grammar.y"
{ yyval.vallist = 0; ;
    break;}
case 66:
#line 308 "xi-grammar.y"
{ yyval.vallist = new ValueList(yyvsp[-1].val, yyvsp[0].vallist); ;
    break;}
case 67:
#line 312 "xi-grammar.y"
{ yyval.readonly = new Readonly(lineno, yyvsp[-2].type, yyvsp[-1].strval, yyvsp[0].vallist); ;
    break;}
case 68:
#line 316 "xi-grammar.y"
{ yyval.readonly = new Readonly(lineno, yyvsp[-2].type, yyvsp[0].strval, 0, 1); ;
    break;}
case 69:
#line 320 "xi-grammar.y"
{ yyval.intval = 0;;
    break;}
case 70:
#line 322 "xi-grammar.y"
{ yyval.intval = 0;;
    break;}
case 71:
#line 326 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 72:
#line 328 "xi-grammar.y"
{ 
		  /*
		  printf("Warning: Message attributes are being phased out.\n");
		  printf("Warning: Please remove them from interface files.\n");
		  */
		  yyval.intval = yyvsp[-1].intval; 
		;
    break;}
case 73:
#line 338 "xi-grammar.y"
{ yyval.intval = yyvsp[0].intval; ;
    break;}
case 74:
#line 340 "xi-grammar.y"
{ yyval.intval = yyvsp[-2].intval | yyvsp[0].intval; ;
    break;}
case 75:
#line 344 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 76:
#line 346 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 77:
#line 350 "xi-grammar.y"
{ yyval.cattr = 0; ;
    break;}
case 78:
#line 352 "xi-grammar.y"
{ yyval.cattr = yyvsp[-1].cattr; ;
    break;}
case 79:
#line 356 "xi-grammar.y"
{ yyval.cattr = yyvsp[0].cattr; ;
    break;}
case 80:
#line 358 "xi-grammar.y"
{ yyval.cattr = yyvsp[-2].cattr | yyvsp[0].cattr; ;
    break;}
case 81:
#line 362 "xi-grammar.y"
{ yyval.cattr = Chare::CMIGRATABLE; ;
    break;}
case 82:
#line 366 "xi-grammar.y"
{ yyval.mv = new MsgVar(yyvsp[-4].type, yyvsp[-3].strval); ;
    break;}
case 83:
#line 370 "xi-grammar.y"
{ yyval.mvlist = new MsgVarList(yyvsp[0].mv); ;
    break;}
case 84:
#line 372 "xi-grammar.y"
{ yyval.mvlist = new MsgVarList(yyvsp[-1].mv, yyvsp[0].mvlist); ;
    break;}
case 85:
#line 376 "xi-grammar.y"
{ yyval.message = new Message(lineno, yyvsp[0].ntype); ;
    break;}
case 86:
#line 378 "xi-grammar.y"
{ yyval.message = new Message(lineno, yyvsp[-3].ntype, yyvsp[-1].mvlist); ;
    break;}
case 87:
#line 382 "xi-grammar.y"
{ yyval.typelist = 0; ;
    break;}
case 88:
#line 384 "xi-grammar.y"
{ yyval.typelist = yyvsp[0].typelist; ;
    break;}
case 89:
#line 388 "xi-grammar.y"
{ yyval.typelist = new TypeList(yyvsp[0].ntype); ;
    break;}
case 90:
#line 390 "xi-grammar.y"
{ yyval.typelist = new TypeList(yyvsp[-2].ntype, yyvsp[0].typelist); ;
    break;}
case 91:
#line 394 "xi-grammar.y"
{ yyval.chare = new Chare(lineno, yyvsp[-3].cattr, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 92:
#line 396 "xi-grammar.y"
{ yyval.chare = new MainChare(lineno, yyvsp[-3].cattr, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 93:
#line 400 "xi-grammar.y"
{ yyval.chare = new Group(lineno, yyvsp[-3].cattr, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 94:
#line 404 "xi-grammar.y"
{ yyval.chare = new NodeGroup(lineno, yyvsp[-3].cattr, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 95:
#line 408 "xi-grammar.y"
{/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",yyvsp[-2].strval);
			yyval.ntype = new NamedType(buf); 
		;
    break;}
case 96:
#line 414 "xi-grammar.y"
{ yyval.ntype = new NamedType(yyvsp[-1].strval); ;
    break;}
case 97:
#line 418 "xi-grammar.y"
{  yyval.chare = new Array(lineno, 0, yyvsp[-3].ntype, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 98:
#line 422 "xi-grammar.y"
{ yyval.chare = new Chare(lineno, yyvsp[-3].cattr, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist);;
    break;}
case 99:
#line 424 "xi-grammar.y"
{ yyval.chare = new MainChare(lineno, yyvsp[-3].cattr, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 100:
#line 428 "xi-grammar.y"
{ yyval.chare = new Group(lineno, yyvsp[-3].cattr, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 101:
#line 432 "xi-grammar.y"
{ yyval.chare = new NodeGroup( lineno, yyvsp[-3].cattr, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 102:
#line 436 "xi-grammar.y"
{ yyval.chare = new Array( lineno, 0, yyvsp[-3].ntype, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 103:
#line 440 "xi-grammar.y"
{ yyval.message = new Message(lineno, new NamedType(yyvsp[-1].strval)); ;
    break;}
case 104:
#line 442 "xi-grammar.y"
{ yyval.message = new Message(lineno, new NamedType(yyvsp[-4].strval), yyvsp[-2].mvlist); ;
    break;}
case 105:
#line 446 "xi-grammar.y"
{ yyval.type = 0; ;
    break;}
case 106:
#line 448 "xi-grammar.y"
{ yyval.type = yyvsp[0].type; ;
    break;}
case 107:
#line 452 "xi-grammar.y"
{ yyval.strval = 0; ;
    break;}
case 108:
#line 454 "xi-grammar.y"
{ yyval.strval = yyvsp[0].strval; ;
    break;}
case 109:
#line 456 "xi-grammar.y"
{ yyval.strval = yyvsp[0].strval; ;
    break;}
case 110:
#line 460 "xi-grammar.y"
{ yyval.tvar = new TType(new NamedType(yyvsp[-1].strval), yyvsp[0].type); ;
    break;}
case 111:
#line 462 "xi-grammar.y"
{ yyval.tvar = new TFunc(yyvsp[-1].ftype, yyvsp[0].strval); ;
    break;}
case 112:
#line 464 "xi-grammar.y"
{ yyval.tvar = new TName(yyvsp[-2].type, yyvsp[-1].strval, yyvsp[0].strval); ;
    break;}
case 113:
#line 468 "xi-grammar.y"
{ yyval.tvarlist = new TVarList(yyvsp[0].tvar); ;
    break;}
case 114:
#line 470 "xi-grammar.y"
{ yyval.tvarlist = new TVarList(yyvsp[-2].tvar, yyvsp[0].tvarlist); ;
    break;}
case 115:
#line 474 "xi-grammar.y"
{ yyval.tvarlist = yyvsp[-1].tvarlist; ;
    break;}
case 116:
#line 478 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); ;
    break;}
case 117:
#line 480 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); ;
    break;}
case 118:
#line 482 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); ;
    break;}
case 119:
#line 484 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); ;
    break;}
case 120:
#line 486 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].message); yyvsp[0].message->setTemplate(yyval.templat); ;
    break;}
case 121:
#line 490 "xi-grammar.y"
{ yyval.mbrlist = 0; ;
    break;}
case 122:
#line 492 "xi-grammar.y"
{ yyval.mbrlist = yyvsp[-2].mbrlist; ;
    break;}
case 123:
#line 496 "xi-grammar.y"
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
                    yyval.mbrlist = ml; 
		  }
		  else {
		    yyval.mbrlist = 0; 
                  }
		;
    break;}
case 124:
#line 515 "xi-grammar.y"
{ yyval.mbrlist = new MemberList(yyvsp[-1].member, yyvsp[0].mbrlist); ;
    break;}
case 125:
#line 519 "xi-grammar.y"
{ yyval.member = yyvsp[-1].readonly; ;
    break;}
case 126:
#line 521 "xi-grammar.y"
{ yyval.member = yyvsp[-1].readonly; ;
    break;}
case 127:
#line 523 "xi-grammar.y"
{ yyval.member = yyvsp[-1].member; ;
    break;}
case 128:
#line 525 "xi-grammar.y"
{ yyval.member = yyvsp[-1].pupable; ;
    break;}
case 129:
#line 529 "xi-grammar.y"
{ yyval.member = new InitCall(lineno, yyvsp[0].strval); ;
    break;}
case 130:
#line 531 "xi-grammar.y"
{ yyval.member = new InitCall(lineno, yyvsp[-3].strval); ;
    break;}
case 131:
#line 535 "xi-grammar.y"
{ yyval.pupable = new PUPableClass(lineno,yyvsp[0].strval,0); ;
    break;}
case 132:
#line 537 "xi-grammar.y"
{ yyval.pupable = new PUPableClass(lineno,yyvsp[-2].strval,yyvsp[0].pupable); ;
    break;}
case 133:
#line 541 "xi-grammar.y"
{ yyval.member = yyvsp[-1].entry; ;
    break;}
case 134:
#line 543 "xi-grammar.y"
{ yyval.member = yyvsp[0].member; ;
    break;}
case 135:
#line 547 "xi-grammar.y"
{ 
		  if (yyvsp[0].sc != 0) { 
		    yyvsp[0].sc->con1 = new SdagConstruct(SIDENT, yyvsp[-3].strval);
  		    if (yyvsp[-2].plist != 0)
                      yyvsp[0].sc->param = new ParamList(yyvsp[-2].plist);
 		    else 
 	 	      yyvsp[0].sc->param = new ParamList(new Parameter(0, new BuiltinType("void")));
                  }
		  yyval.entry = new Entry(lineno, yyvsp[-5].intval, yyvsp[-4].type, yyvsp[-3].strval, yyvsp[-2].plist, yyvsp[-1].val, yyvsp[0].sc, 0, 0); 
		;
    break;}
case 136:
#line 558 "xi-grammar.y"
{ 
		  if (yyvsp[0].sc != 0) {
		    yyvsp[0].sc->con1 = new SdagConstruct(SIDENT, yyvsp[-2].strval);
		    if (yyvsp[-1].plist != 0)
                      yyvsp[0].sc->param = new ParamList(yyvsp[-1].plist);
		    else
                      yyvsp[0].sc->param = new ParamList(new Parameter(0, new BuiltinType("void")));
                  }
		  yyval.entry = new Entry(lineno, yyvsp[-3].intval,     0, yyvsp[-2].strval, yyvsp[-1].plist,  0, yyvsp[0].sc, 0, 0); 
		;
    break;}
case 137:
#line 571 "xi-grammar.y"
{ yyval.type = new BuiltinType("void"); ;
    break;}
case 138:
#line 573 "xi-grammar.y"
{ yyval.type = yyvsp[0].ptype; ;
    break;}
case 139:
#line 577 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 140:
#line 579 "xi-grammar.y"
{ yyval.intval = yyvsp[-1].intval; ;
    break;}
case 141:
#line 583 "xi-grammar.y"
{ yyval.intval = yyvsp[0].intval; ;
    break;}
case 142:
#line 585 "xi-grammar.y"
{ yyval.intval = yyvsp[-2].intval | yyvsp[0].intval; ;
    break;}
case 143:
#line 589 "xi-grammar.y"
{ yyval.intval = STHREADED; ;
    break;}
case 144:
#line 591 "xi-grammar.y"
{ yyval.intval = SSYNC; ;
    break;}
case 145:
#line 593 "xi-grammar.y"
{ yyval.intval = SLOCKED; ;
    break;}
case 146:
#line 595 "xi-grammar.y"
{ yyval.intval = SCREATEHERE; ;
    break;}
case 147:
#line 597 "xi-grammar.y"
{ yyval.intval = SCREATEHOME; ;
    break;}
case 148:
#line 599 "xi-grammar.y"
{ yyval.intval = SNOKEEP; ;
    break;}
case 149:
#line 601 "xi-grammar.y"
{ yyval.intval = SIMMEDIATE; ;
    break;}
case 150:
#line 605 "xi-grammar.y"
{ yyval.val = new Value(yyvsp[0].strval); ;
    break;}
case 151:
#line 607 "xi-grammar.y"
{ yyval.val = new Value(yyvsp[0].strval); ;
    break;}
case 152:
#line 609 "xi-grammar.y"
{ yyval.val = new Value(yyvsp[0].strval); ;
    break;}
case 153:
#line 613 "xi-grammar.y"
{ yyval.strval = ""; ;
    break;}
case 154:
#line 615 "xi-grammar.y"
{ yyval.strval = yyvsp[0].strval; ;
    break;}
case 155:
#line 617 "xi-grammar.y"
{  /*Returned only when in_bracket*/
			char *tmp = new char[strlen(yyvsp[-4].strval)+strlen(yyvsp[-2].strval)+strlen(yyvsp[0].strval)+3];
			sprintf(tmp,"%s[%s]%s", yyvsp[-4].strval, yyvsp[-2].strval, yyvsp[0].strval);
			yyval.strval = tmp;
		;
    break;}
case 156:
#line 623 "xi-grammar.y"
{ /*Returned only when in_braces*/
			char *tmp = new char[strlen(yyvsp[-4].strval)+strlen(yyvsp[-2].strval)+strlen(yyvsp[0].strval)+3];
			sprintf(tmp,"%s{%s}%s", yyvsp[-4].strval, yyvsp[-2].strval, yyvsp[0].strval);
			yyval.strval = tmp;
		;
    break;}
case 157:
#line 629 "xi-grammar.y"
{ /*Returned only when in_braces*/
			char *tmp = new char[strlen(yyvsp[-4].strval)+strlen(yyvsp[-2].strval)+strlen(yyvsp[0].strval)+3];
			sprintf(tmp,"%s(%s)%s", yyvsp[-4].strval, yyvsp[-2].strval, yyvsp[0].strval);
			yyval.strval = tmp;
		;
    break;}
case 158:
#line 635 "xi-grammar.y"
{ /*Returned only when in_braces*/
			char *tmp = new char[strlen(yyvsp[-2].strval)+strlen(yyvsp[0].strval)+3];
			sprintf(tmp,"(%s)%s", yyvsp[-2].strval, yyvsp[0].strval);
			yyval.strval = tmp;
		;
    break;}
case 159:
#line 643 "xi-grammar.y"
{  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			yyval.pname = new Parameter(lineno, yyvsp[-2].type,yyvsp[-1].strval);
		;
    break;}
case 160:
#line 650 "xi-grammar.y"
{ 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			yyval.intval = 0;
		;
    break;}
case 161:
#line 658 "xi-grammar.y"
{ 
			in_braces=0;
			yyval.intval = 0;
		;
    break;}
case 162:
#line 665 "xi-grammar.y"
{ yyval.pname = new Parameter(lineno, yyvsp[0].type);;
    break;}
case 163:
#line 667 "xi-grammar.y"
{ yyval.pname = new Parameter(lineno, yyvsp[-1].type,yyvsp[0].strval);;
    break;}
case 164:
#line 669 "xi-grammar.y"
{ yyval.pname = new Parameter(lineno, yyvsp[-3].type,yyvsp[-2].strval,0,yyvsp[0].val);;
    break;}
case 165:
#line 671 "xi-grammar.y"
{ /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			yyval.pname = new Parameter(lineno, yyvsp[-2].pname->getType(), yyvsp[-2].pname->getName() ,yyvsp[-1].strval);
		;
    break;}
case 166:
#line 678 "xi-grammar.y"
{ yyval.plist = new ParamList(yyvsp[0].pname); ;
    break;}
case 167:
#line 680 "xi-grammar.y"
{ yyval.plist = new ParamList(yyvsp[-2].pname,yyvsp[0].plist); ;
    break;}
case 168:
#line 684 "xi-grammar.y"
{ yyval.plist = yyvsp[-1].plist; ;
    break;}
case 169:
#line 686 "xi-grammar.y"
{ yyval.plist = 0; ;
    break;}
case 170:
#line 690 "xi-grammar.y"
{ yyval.val = 0; ;
    break;}
case 171:
#line 692 "xi-grammar.y"
{ yyval.val = new Value(yyvsp[0].strval); ;
    break;}
case 172:
#line 696 "xi-grammar.y"
{ yyval.sc = 0; ;
    break;}
case 173:
#line 698 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SSDAGENTRY, yyvsp[0].sc); ;
    break;}
case 174:
#line 700 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SSDAGENTRY, yyvsp[-1].sc); ;
    break;}
case 175:
#line 704 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SSLIST, yyvsp[0].sc); ;
    break;}
case 176:
#line 706 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SSLIST, yyvsp[-1].sc, yyvsp[0].sc);  ;
    break;}
case 177:
#line 710 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SOLIST, yyvsp[0].sc); ;
    break;}
case 178:
#line 712 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SOLIST, yyvsp[-1].sc, yyvsp[0].sc); ;
    break;}
case 179:
#line 716 "xi-grammar.y"
{ yyval.sc = 0; ;
    break;}
case 180:
#line 718 "xi-grammar.y"
{ yyval.sc = yyvsp[-1].sc; ;
    break;}
case 181:
#line 722 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SPUBLISHES, new SdagConstruct(SIDENT, yyvsp[0].strval)); ;
    break;}
case 182:
#line 724 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SPUBLISHES, new SdagConstruct(SIDENT, yyvsp[-2].strval), yyvsp[0].sc);  ;
    break;}
case 183:
#line 728 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SATOMIC, new XStr(yyvsp[-2].strval), yyvsp[0].sc, 0,0,0,0, 0 ); ;
    break;}
case 184:
#line 730 "xi-grammar.y"
{  
		   in_braces = 0;
		   if ((yyvsp[-4].plist->isVoid() == 0) && (yyvsp[-4].plist->isMessage() == 0))
                   {
		      connectEntries->append(new Entry(0, 0, new BuiltinType("void"), yyvsp[-5].strval, 
	 	 			new ParamList(new Parameter(lineno, new PtrType( 
                                        new NamedType("CkMarshallMsg")), "_msg")), 0, 0, 0, 1, yyvsp[-4].plist));
		   }
		   else  {
		      connectEntries->append(new Entry(0, 0, new BuiltinType("void"), yyvsp[-5].strval, yyvsp[-4].plist, 0, 0, 0, 1, yyvsp[-4].plist));
                   }
                   yyval.sc = new SdagConstruct(SCONNECT, yyvsp[-5].strval, yyvsp[-1].strval, yyvsp[-4].plist);
		;
    break;}
case 185:
#line 744 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SWHEN, 0, 0,0,0,0,0,yyvsp[-2].entrylist); ;
    break;}
case 186:
#line 746 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SWHEN, 0, 0, 0,0,0, yyvsp[0].sc, yyvsp[-1].entrylist); ;
    break;}
case 187:
#line 748 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SWHEN, 0, 0, 0,0,0, yyvsp[-1].sc, yyvsp[-3].entrylist); ;
    break;}
case 188:
#line 750 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SOVERLAP,0, 0,0,0,0,yyvsp[-1].sc, 0); ;
    break;}
case 189:
#line 752 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SFOR, 0, new SdagConstruct(SINT_EXPR, yyvsp[-8].strval), new SdagConstruct(SINT_EXPR, yyvsp[-6].strval),
		             new SdagConstruct(SINT_EXPR, yyvsp[-4].strval), 0, yyvsp[-1].sc, 0); ;
    break;}
case 190:
#line 755 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SFOR, 0, new SdagConstruct(SINT_EXPR, yyvsp[-6].strval), new SdagConstruct(SINT_EXPR, yyvsp[-4].strval), 
		         new SdagConstruct(SINT_EXPR, yyvsp[-2].strval), 0, yyvsp[0].sc, 0); ;
    break;}
case 191:
#line 758 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SFORALL, 0, new SdagConstruct(SIDENT, yyvsp[-9].strval), new SdagConstruct(SINT_EXPR, yyvsp[-6].strval), 
		             new SdagConstruct(SINT_EXPR, yyvsp[-4].strval), new SdagConstruct(SINT_EXPR, yyvsp[-2].strval), yyvsp[0].sc, 0); ;
    break;}
case 192:
#line 761 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SFORALL, 0, new SdagConstruct(SIDENT, yyvsp[-11].strval), new SdagConstruct(SINT_EXPR, yyvsp[-8].strval), 
		                 new SdagConstruct(SINT_EXPR, yyvsp[-6].strval), new SdagConstruct(SINT_EXPR, yyvsp[-4].strval), yyvsp[-1].sc, 0); ;
    break;}
case 193:
#line 764 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, yyvsp[-3].strval), yyvsp[0].sc,0,0,yyvsp[-1].sc,0); ;
    break;}
case 194:
#line 766 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, yyvsp[-5].strval), yyvsp[0].sc,0,0,yyvsp[-2].sc,0); ;
    break;}
case 195:
#line 768 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, yyvsp[-2].strval), 0,0,0,yyvsp[0].sc,0); ;
    break;}
case 196:
#line 770 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SWHILE, 0, new SdagConstruct(SINT_EXPR, yyvsp[-4].strval), 0,0,0,yyvsp[-1].sc,0); ;
    break;}
case 197:
#line 772 "xi-grammar.y"
{ yyval.sc = yyvsp[-1].sc; ;
    break;}
case 198:
#line 776 "xi-grammar.y"
{ yyval.sc = 0; ;
    break;}
case 199:
#line 778 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SELSE, 0,0,0,0,0, yyvsp[0].sc,0); ;
    break;}
case 200:
#line 780 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SELSE, 0,0,0,0,0, yyvsp[-1].sc,0); ;
    break;}
case 201:
#line 783 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SFORWARD, new SdagConstruct(SIDENT, yyvsp[0].strval)); ;
    break;}
case 202:
#line 785 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SFORWARD, new SdagConstruct(SIDENT, yyvsp[-2].strval), yyvsp[0].sc);  ;
    break;}
case 203:
#line 789 "xi-grammar.y"
{ in_int_expr = 0; yyval.intval = 0; ;
    break;}
case 204:
#line 793 "xi-grammar.y"
{ in_int_expr = 1; yyval.intval = 0; ;
    break;}
case 205:
#line 797 "xi-grammar.y"
{ 
		  if (yyvsp[0].plist != 0)
		     yyval.entry = new Entry(lineno, 0, 0, yyvsp[-1].strval, yyvsp[0].plist, 0, 0, 0, 0); 
		  else
		     yyval.entry = new Entry(lineno, 0, 0, yyvsp[-1].strval, 
				new ParamList(new Parameter(0, new BuiltinType("void"))), 0, 0, 0, 0); 
		;
    break;}
case 206:
#line 805 "xi-grammar.y"
{ if (yyvsp[0].plist != 0)
		    yyval.entry = new Entry(lineno, 0, 0, yyvsp[-4].strval, yyvsp[0].plist, 0, 0, yyvsp[-2].strval, 0); 
		  else
		    yyval.entry = new Entry(lineno, 0, 0, yyvsp[-4].strval, new ParamList(new Parameter(0, new BuiltinType("void"))), 0, 0, yyvsp[-2].strval, 0); 
		;
    break;}
case 207:
#line 813 "xi-grammar.y"
{ yyval.entrylist = new EntryList(yyvsp[0].entry); ;
    break;}
case 208:
#line 815 "xi-grammar.y"
{ yyval.entrylist = new EntryList(yyvsp[-2].entry,yyvsp[0].entrylist); ;
    break;}
case 209:
#line 819 "xi-grammar.y"
{ in_bracket=1; ;
    break;}
case 210:
#line 822 "xi-grammar.y"
{ in_bracket=0; ;
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
#line 826 "xi-grammar.y"

void yyerror(const char *mesg)
{
  cout << cur_file<<":"<<lineno<<": Charmxi syntax error> " << mesg << endl;
  // return 0;
}
