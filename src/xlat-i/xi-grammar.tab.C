
/*  A Bison parser, made from xi-grammar.y
    by GNU Bison version 1.28  */

#define YYBISON 1  /* Identify Bison output.  */

#define	MODULE	257
#define	MAINMODULE	258
#define	EXTERN	259
#define	READONLY	260
#define	INITCALL	261
#define	INITNODE	262
#define	INITPROC	263
#define	PUPABLE	264
#define	CHARE	265
#define	MAINCHARE	266
#define	GROUP	267
#define	NODEGROUP	268
#define	ARRAY	269
#define	MESSAGE	270
#define	CLASS	271
#define	STACKSIZE	272
#define	THREADED	273
#define	TEMPLATE	274
#define	SYNC	275
#define	EXCLUSIVE	276
#define	IMMEDIATE	277
#define	VIRTUAL	278
#define	MIGRATABLE	279
#define	CREATEHERE	280
#define	CREATEHOME	281
#define	NOKEEP	282
#define	VOID	283
#define	CONST	284
#define	PACKED	285
#define	VARSIZE	286
#define	ENTRY	287
#define	FOR	288
#define	FORALL	289
#define	WHILE	290
#define	WHEN	291
#define	OVERLAP	292
#define	ATOMIC	293
#define	FORWARD	294
#define	IF	295
#define	ELSE	296
#define	CONNECT	297
#define	PUBLISHES	298
#define	IDENT	299
#define	NUMBER	300
#define	LITERAL	301
#define	CPROGRAM	302
#define	INT	303
#define	LONG	304
#define	SHORT	305
#define	CHAR	306
#define	FLOAT	307
#define	DOUBLE	308
#define	UNSIGNED	309

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



#define	YYFINAL		443
#define	YYFLAG		-32768
#define	YYNTBASE	70

#define YYTRANSLATE(x) ((unsigned)(x) <= 309 ? yytranslate[x] : 160)

static const char yytranslate[] = {     0,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,    66,     2,    64,
    65,    63,     2,    60,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,    57,    56,    61,
    69,    62,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
    67,     2,    68,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,    58,     2,    59,     2,     2,     2,     2,     2,
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
    47,    48,    49,    50,    51,    52,    53,    54,    55
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
   381,   383,   388,   389,   392,   395,   398,   401,   404,   408,
   412,   419,   423,   430,   434,   441,   443,   447,   450,   452,
   460,   466,   468,   470,   471,   475,   477,   481,   483,   485,
   487,   489,   491,   493,   495,   497,   499,   501,   502,   504,
   510,   516,   522,   527,   531,   533,   535,   537,   540,   545,
   549,   551,   555,   559,   562,   563,   567,   568,   570,   574,
   576,   579,   581,   584,   585,   590,   592,   596,   602,   611,
   616,   620,   626,   631,   643,   653,   666,   681,   688,   697,
   703,   711,   715,   716,   719,   724,   726,   730,   732,   734,
   737,   743,   745,   749,   751
};

static const short yyrhs[] = {    71,
     0,     0,    76,    71,     0,     0,     5,     0,     0,    56,
     0,    45,     0,    45,     0,    75,    57,    57,    45,     0,
     3,    74,    77,     0,     4,    74,    77,     0,    56,     0,
    58,    78,    59,    73,     0,     0,    79,    78,     0,    72,
    58,    78,    59,    73,     0,    72,    76,     0,    72,   127,
     0,    72,   106,    56,     0,    72,   109,     0,    72,   110,
     0,    72,   111,     0,    72,   113,     0,    72,   124,     0,
    91,     0,    46,     0,    47,     0,    80,     0,    80,    60,
    81,     0,     0,    81,     0,     0,    61,    82,    62,     0,
    49,     0,    50,     0,    51,     0,    52,     0,    55,    49,
     0,    55,    50,     0,    55,    51,     0,    55,    52,     0,
    50,    50,     0,    53,     0,    54,     0,    50,    54,     0,
    29,     0,    74,    83,     0,    75,    83,     0,    84,     0,
    86,     0,    87,    63,     0,    88,    63,     0,    89,    63,
     0,    91,    64,    63,    74,    65,    64,   143,    65,     0,
    87,     0,    88,     0,    89,     0,    90,     0,    91,    66,
     0,    30,    91,     0,    46,     0,    75,     0,    67,    92,
    68,     0,     0,    93,    94,     0,     6,    91,    75,    94,
     0,     6,    16,    87,    63,    74,     0,     0,    29,     0,
     0,    67,    99,    68,     0,   100,     0,   100,    60,    99,
     0,    31,     0,    32,     0,     0,    67,   102,    68,     0,
   103,     0,   103,    60,   102,     0,    25,     0,    91,    74,
    67,    68,    56,     0,   104,     0,   104,   105,     0,    16,
    98,    85,     0,    16,    98,    85,    58,   105,    59,     0,
     0,    57,   108,     0,    85,     0,    85,    60,   108,     0,
    11,   101,    85,   107,   125,     0,    12,   101,    85,   107,
   125,     0,    13,   101,    85,   107,   125,     0,    14,   101,
    85,   107,   125,     0,    67,    46,    74,    68,     0,    67,
    74,    68,     0,    15,   112,    85,   107,   125,     0,    11,
   101,    74,   107,   125,     0,    12,   101,    74,   107,   125,
     0,    13,   101,    74,   107,   125,     0,    14,   101,    74,
   107,   125,     0,    15,   112,    74,   107,   125,     0,    16,
    98,    74,    56,     0,    16,    98,    74,    58,   105,    59,
    56,     0,     0,    69,    91,     0,     0,    69,    46,     0,
    69,    47,     0,    17,    74,   119,     0,    90,   120,     0,
    91,    74,   120,     0,   121,     0,   121,    60,   122,     0,
    20,    61,   122,    62,     0,   123,   114,     0,   123,   115,
     0,   123,   116,     0,   123,   117,     0,   123,   118,     0,
    56,     0,    58,   126,    59,    73,     0,     0,   131,   126,
     0,    95,    56,     0,    96,    56,     0,   129,    56,     0,
   128,    56,     0,    10,   130,    56,     0,     8,    97,    75,
     0,     8,    97,    75,    64,    97,    65,     0,     7,    97,
    75,     0,     7,    97,    75,    64,    97,    65,     0,     9,
    97,    75,     0,     9,    97,    75,    64,    97,    65,     0,
    75,     0,    75,    60,   130,     0,   132,    56,     0,   127,
     0,    33,   134,   133,    74,   144,   145,   146,     0,    33,
   134,    74,   144,   146,     0,    29,     0,    88,     0,     0,
    67,   135,    68,     0,   136,     0,   136,    60,   135,     0,
    19,     0,    21,     0,    22,     0,    26,     0,    27,     0,
    28,     0,    23,     0,    47,     0,    46,     0,    75,     0,
     0,    48,     0,    48,    67,   138,    68,   138,     0,    48,
    58,   138,    59,   138,     0,    48,    64,   138,    65,   138,
     0,    64,   138,    65,   138,     0,    91,    74,    67,     0,
    58,     0,    59,     0,    91,     0,    91,    74,     0,    91,
    74,    69,   137,     0,   139,   138,    68,     0,   142,     0,
   142,    60,   143,     0,    64,   143,    65,     0,    64,    65,
     0,     0,    18,    69,    46,     0,     0,   151,     0,    58,
   147,    59,     0,   151,     0,   151,   147,     0,   151,     0,
   151,   147,     0,     0,    44,    64,   150,    65,     0,    45,
     0,    45,    60,   150,     0,    39,   140,   138,   141,   149,
     0,    43,    64,    45,   144,    65,   140,   138,    59,     0,
    37,   157,    58,    59,     0,    37,   157,   151,     0,    37,
   157,    58,   147,    59,     0,    38,    58,   148,    59,     0,
    34,   155,   138,    56,   138,    56,   138,   154,    58,   147,
    59,     0,    34,   155,   138,    56,   138,    56,   138,   154,
   151,     0,    35,    67,    45,    68,   155,   138,    57,   138,
    60,   138,   154,   151,     0,    35,    67,    45,    68,   155,
   138,    57,   138,    60,   138,   154,    58,   147,    59,     0,
    41,   155,   138,   154,   151,   152,     0,    41,   155,   138,
   154,    58,   147,    59,   152,     0,    36,   155,   138,   154,
   151,     0,    36,   155,   138,   154,    58,   147,    59,     0,
    40,   153,    56,     0,     0,    42,   151,     0,    42,    58,
   147,    59,     0,    45,     0,    45,    60,   153,     0,    65,
     0,    64,     0,    45,   144,     0,    45,   158,   138,   159,
   144,     0,   156,     0,   156,    60,   157,     0,    67,     0,
    68,     0
};

#endif

#if YYDEBUG != 0
static const short yyrline[] = { 0,
   124,   128,   132,   136,   138,   142,   144,   148,   152,   154,
   162,   166,   173,   175,   179,   181,   185,   187,   189,   191,
   193,   195,   197,   199,   201,   205,   207,   209,   213,   215,
   219,   221,   225,   227,   231,   233,   235,   237,   239,   241,
   243,   245,   247,   249,   251,   253,   255,   259,   260,   262,
   264,   268,   272,   274,   278,   282,   284,   286,   288,   290,
   293,   297,   299,   303,   307,   309,   313,   317,   321,   323,
   327,   329,   339,   341,   345,   347,   351,   353,   357,   359,
   363,   367,   371,   373,   377,   379,   383,   385,   389,   391,
   395,   397,   401,   405,   409,   415,   419,   423,   425,   429,
   433,   437,   441,   443,   447,   449,   453,   455,   457,   461,
   463,   465,   469,   471,   475,   479,   481,   483,   485,   487,
   491,   493,   497,   516,   520,   522,   524,   525,   527,   531,
   533,   535,   538,   543,   545,   549,   551,   555,   557,   561,
   572,   585,   587,   591,   593,   597,   599,   603,   605,   607,
   609,   611,   613,   615,   619,   621,   623,   627,   629,   631,
   637,   643,   649,   657,   664,   672,   679,   681,   683,   685,
   692,   694,   698,   700,   704,   706,   710,   712,   714,   718,
   720,   724,   726,   730,   732,   736,   738,   742,   745,   759,
   761,   763,   765,   767,   770,   773,   776,   779,   781,   783,
   785,   787,   791,   793,   795,   798,   800,   804,   808,   812,
   820,   828,   830,   834,   837
};
#endif


#if YYDEBUG != 0 || defined (YYERROR_VERBOSE)

static const char * const yytname[] = {   "$","error","$undefined.","MODULE",
"MAINMODULE","EXTERN","READONLY","INITCALL","INITNODE","INITPROC","PUPABLE",
"CHARE","MAINCHARE","GROUP","NODEGROUP","ARRAY","MESSAGE","CLASS","STACKSIZE",
"THREADED","TEMPLATE","SYNC","EXCLUSIVE","IMMEDIATE","VIRTUAL","MIGRATABLE",
"CREATEHERE","CREATEHOME","NOKEEP","VOID","CONST","PACKED","VARSIZE","ENTRY",
"FOR","FORALL","WHILE","WHEN","OVERLAP","ATOMIC","FORWARD","IF","ELSE","CONNECT",
"PUBLISHES","IDENT","NUMBER","LITERAL","CPROGRAM","INT","LONG","SHORT","CHAR",
"FLOAT","DOUBLE","UNSIGNED","';'","':'","'{'","'}'","','","'<'","'>'","'*'",
"'('","')'","'&'","'['","']'","'='","File","ModuleEList","OptExtern","OptSemiColon",
"Name","QualName","Module","ConstructEList","ConstructList","Construct","TParam",
"TParamList","TParamEList","OptTParams","BuiltinType","NamedType","QualNamedType",
"SimpleType","OnePtrType","PtrType","FuncType","Type","ArrayDim","Dim","DimList",
"Readonly","ReadonlyMsg","OptVoid","MAttribs","MAttribList","MAttrib","CAttribs",
"CAttribList","CAttrib","Var","VarList","Message","OptBaseList","BaseList","Chare",
"Group","NodeGroup","ArrayIndexType","Array","TChare","TGroup","TNodeGroup",
"TArray","TMessage","OptTypeInit","OptNameInit","TVar","TVarList","TemplateSpec",
"Template","MemberEList","MemberList","NonEntryMember","InitNode","InitProc",
"PUPableClass","Member","Entry","EReturn","EAttribs","EAttribList","EAttrib",
"DefaultParameter","CCode","ParamBracketStart","ParamBraceStart","ParamBraceEnd",
"Parameter","ParamList","EParameters","OptStackSize","OptSdagCode","Slist","Olist",
"OptPubList","PublishesList","SingleConstruct","HasElse","ForwardList","EndIntExpr",
"StartIntExpr","SEntry","SEntryList","SParamBracketStart","SParamBracketEnd", NULL
};
#endif

static const short yyr1[] = {     0,
    70,    71,    71,    72,    72,    73,    73,    74,    75,    75,
    76,    76,    77,    77,    78,    78,    79,    79,    79,    79,
    79,    79,    79,    79,    79,    80,    80,    80,    81,    81,
    82,    82,    83,    83,    84,    84,    84,    84,    84,    84,
    84,    84,    84,    84,    84,    84,    84,    85,    86,    87,
    87,    88,    89,    89,    90,    91,    91,    91,    91,    91,
    91,    92,    92,    93,    94,    94,    95,    96,    97,    97,
    98,    98,    99,    99,   100,   100,   101,   101,   102,   102,
   103,   104,   105,   105,   106,   106,   107,   107,   108,   108,
   109,   109,   110,   111,   112,   112,   113,   114,   114,   115,
   116,   117,   118,   118,   119,   119,   120,   120,   120,   121,
   121,   121,   122,   122,   123,   124,   124,   124,   124,   124,
   125,   125,   126,   126,   127,   127,   127,   127,   127,   128,
   128,   128,   128,   129,   129,   130,   130,   131,   131,   132,
   132,   133,   133,   134,   134,   135,   135,   136,   136,   136,
   136,   136,   136,   136,   137,   137,   137,   138,   138,   138,
   138,   138,   138,   139,   140,   141,   142,   142,   142,   142,
   143,   143,   144,   144,   145,   145,   146,   146,   146,   147,
   147,   148,   148,   149,   149,   150,   150,   151,   151,   151,
   151,   151,   151,   151,   151,   151,   151,   151,   151,   151,
   151,   151,   152,   152,   152,   153,   153,   154,   155,   156,
   156,   157,   157,   158,   159
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
     1,     4,     0,     2,     2,     2,     2,     2,     3,     3,
     6,     3,     6,     3,     6,     1,     3,     2,     1,     7,
     5,     1,     1,     0,     3,     1,     3,     1,     1,     1,
     1,     1,     1,     1,     1,     1,     1,     0,     1,     5,
     5,     5,     4,     3,     1,     1,     1,     2,     4,     3,
     1,     3,     3,     2,     0,     3,     0,     1,     3,     1,
     2,     1,     2,     0,     4,     1,     3,     5,     8,     4,
     3,     5,     4,    11,     9,    12,    14,     6,     8,     5,
     7,     3,     0,     2,     4,     1,     3,     1,     1,     2,
     5,     1,     3,     1,     1
};

static const short yydefact[] = {     2,
     0,     0,     1,     2,     8,     0,     0,     3,    13,     4,
    11,    12,     5,     0,     0,     4,     0,    69,    69,    69,
     0,    77,    77,    77,    77,     0,    71,     0,     4,    18,
     0,     0,     0,    21,    22,    23,    24,     0,    25,    19,
     0,     0,     6,    16,     0,    47,     0,     9,    35,    36,
    37,    38,    44,    45,     0,    33,    50,    51,    56,    57,
    58,    59,     0,    70,     0,     0,     0,   136,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
   125,   126,    20,    77,    77,    77,    77,     0,    71,   116,
   117,   118,   119,   120,   128,   127,     7,    14,     0,    61,
    43,    46,    39,    40,    41,    42,     0,    31,    49,    52,
    53,    54,     0,    60,    65,   132,   130,   134,     0,   129,
    81,     0,    79,    33,    87,    87,    87,    87,     0,     0,
    87,    75,    76,     0,    73,    85,     0,    59,     0,   113,
     0,     6,     0,     0,     0,     0,     0,     0,     0,     0,
    27,    28,    29,    32,     0,    26,     0,     0,    65,    67,
    69,    69,    69,   137,    78,     0,    48,     0,     0,     0,
     0,     0,     0,    96,     0,    72,     0,     0,   105,     0,
   111,   107,     0,   115,    17,    87,    87,    87,    87,    87,
     0,    68,    10,     0,    34,     0,    62,    63,     0,    66,
     0,     0,     0,    80,    89,    88,   121,   123,    91,    92,
    93,    94,    95,    97,    74,     0,    83,     0,     0,   110,
   108,   109,   112,   114,     0,     0,     0,     0,     0,   103,
     0,    30,     0,    64,   133,   131,   135,     0,   144,     0,
   139,   123,     0,     0,    84,    86,   106,    98,    99,   100,
   101,   102,     0,     0,    90,     0,     0,     6,   124,   138,
     0,     0,   167,   158,   171,     0,   148,   149,   150,   154,
   151,   152,   153,     0,   146,    47,     9,     0,     0,   143,
     0,   122,     0,   104,   168,   159,   158,     0,     0,    55,
   145,     0,     0,   177,     0,    82,   164,     0,   158,   158,
   158,     0,   170,   172,   147,   174,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,   141,   178,   175,
   156,   155,   157,   169,     0,     0,     0,   158,   173,   209,
   158,     0,   158,     0,   212,     0,     0,   165,   158,   206,
     0,   158,     0,     0,   180,     0,   177,   158,   158,   158,
   163,     0,     0,     0,   214,   210,   158,     0,     0,   191,
     0,   182,     0,     0,   202,     0,     0,   179,   181,     0,
   140,   161,   162,   160,   158,     0,   208,     0,     0,   213,
   190,     0,   193,   183,   166,   184,   207,     0,     0,   176,
     0,   158,     0,   200,   215,     0,   192,     0,   188,     0,
   203,     0,   158,     0,     0,   211,     0,     0,     0,   198,
   158,     0,   158,   201,   186,     0,   203,     0,   204,     0,
     0,     0,     0,   185,   199,     0,   189,     0,   195,   158,
   187,   205,     0,     0,   194,     0,     0,   196,     0,   197,
     0,     0,     0
};

static const short yydefgoto[] = {   441,
     3,    14,    98,   124,    56,     4,    11,    15,    16,   153,
   154,   155,   109,    57,   205,    58,    59,    60,    61,    62,
   216,   199,   159,   160,    31,    32,    65,    78,   134,   135,
    71,   122,   123,   217,   218,    33,   169,   206,    34,    35,
    36,    76,    37,    90,    91,    92,    93,    94,   220,   181,
   140,   141,    38,    39,   209,   240,   241,    41,    42,    69,
   242,   243,   281,   257,   274,   275,   324,   288,   264,   339,
   386,   265,   266,   294,   347,   318,   344,   361,   399,   416,
   345,   410,   341,   378,   331,   335,   336,   357,   396
};

static const short yypact[] = {    55,
    32,    32,-32768,    55,-32768,   105,   105,-32768,-32768,     6,
-32768,-32768,-32768,   215,    27,     6,   216,    65,    65,    65,
    69,    68,    68,    68,    68,    88,   124,   116,     6,-32768,
   140,   184,   191,-32768,-32768,-32768,-32768,   197,-32768,-32768,
   226,   227,   228,-32768,   135,-32768,   273,-32768,-32768,   -36,
-32768,-32768,-32768,-32768,   108,    92,-32768,-32768,   143,   181,
   222,-32768,   -32,-32768,    69,    69,    69,   110,   230,   263,
    32,    32,    32,    32,   -23,    32,   137,    32,   149,   234,
-32768,-32768,-32768,    68,    68,    68,    68,    88,   124,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,   224,   117,
-32768,-32768,-32768,-32768,-32768,-32768,   232,   262,-32768,-32768,
-32768,-32768,   238,-32768,    73,    -7,    34,    45,    69,-32768,
-32768,   236,   246,   244,   264,   264,   264,   264,    32,   252,
   264,-32768,-32768,   266,   270,   296,    32,    72,    21,   304,
   305,   228,    32,    32,    32,    32,    32,    32,    32,   321,
-32768,-32768,   308,-32768,   307,   117,    32,   188,   303,-32768,
    65,    65,    65,-32768,-32768,   263,-32768,    32,   118,   118,
   118,   118,   324,-32768,   118,-32768,   137,   273,   325,   217,
-32768,   326,   149,-32768,-32768,   264,   264,   264,   264,   264,
   139,-32768,-32768,   262,-32768,   328,-32768,   333,   329,-32768,
   351,   353,   354,-32768,   360,-32768,-32768,    19,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,    21,   273,   362,   273,-32768,
-32768,-32768,-32768,-32768,   118,   118,   118,   118,   118,-32768,
   273,-32768,   359,-32768,-32768,-32768,-32768,    32,   367,   376,
-32768,    19,   380,   370,-32768,-32768,   117,-32768,-32768,-32768,
-32768,-32768,   379,   273,-32768,   310,   290,   228,-32768,-32768,
   371,   384,    21,    24,   381,   377,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,   375,   385,   399,   382,   383,   143,-32768,
    32,-32768,   392,-32768,   209,    39,    24,   386,   273,-32768,
-32768,   310,   245,     2,   383,-32768,-32768,   213,    24,    24,
    24,   387,-32768,-32768,-32768,-32768,   388,   391,   389,   391,
   404,   393,   400,   405,   391,   395,   390,-32768,-32768,   439,
-32768,-32768,   333,-32768,   401,   396,   394,    24,-32768,-32768,
    24,   418,    24,   150,   406,   214,   390,-32768,    24,   407,
   408,    24,   420,   409,   390,   402,     2,    24,    24,    24,
-32768,   413,   410,   411,-32768,-32768,    24,   404,   312,-32768,
   414,   390,   415,   405,-32768,   411,   383,-32768,-32768,   424,
-32768,-32768,-32768,-32768,    24,   391,-32768,   322,   412,-32768,
-32768,   416,-32768,-32768,-32768,   428,-32768,   338,   417,-32768,
   421,    24,   390,-32768,-32768,   383,-32768,   419,-32768,   390,
   437,   400,    24,   427,   422,-32768,   440,   429,   348,-32768,
    24,   411,    24,-32768,   426,   425,   437,   390,-32768,   430,
   364,   431,   440,-32768,-32768,   433,-32768,   390,-32768,    24,
-32768,-32768,   434,   411,-32768,   374,   390,-32768,   435,-32768,
   487,   495,-32768
};

static const short yypgoto[] = {-32768,
   492,-32768,  -136,    -1,   -19,   483,   491,    76,-32768,-32768,
   306,-32768,   378,-32768,   165,-32768,   -41,   242,-32768,   -76,
   -12,-32768,-32768,   342,-32768,-32768,   -11,   423,   327,-32768,
    31,   337,-32768,-32768,  -200,-32768,  -107,   267,-32768,-32768,
-32768,   432,-32768,-32768,-32768,-32768,-32768,-32768,-32768,   331,
-32768,   323,-32768,-32768,  -102,   265,   494,-32768,-32768,   397,
-32768,-32768,-32768,-32768,   218,-32768,-32768,  -238,-32768,   107,
-32768,-32768,  -160,  -283,-32768,   164,  -329,-32768,-32768,    91,
  -216,    98,   153,  -359,  -300,-32768,   160,-32768,-32768
};


#define	YYLAST		520


static const short yytable[] = {     6,
     7,    68,   138,    99,    63,   185,   388,    66,    67,   333,
    13,   320,    48,   101,   342,   369,   245,   102,   170,   171,
   172,     5,   129,   175,    17,    18,    19,    20,    21,   382,
   253,   113,   384,   114,   100,   308,   309,   310,   311,   312,
   313,   314,   315,   115,   316,   116,   117,   118,   302,   107,
   356,   239,   421,    72,    73,    74,   161,     1,     2,   317,
   325,   326,   327,   405,   -15,     5,   139,   210,   211,   212,
   408,   286,   214,   130,   436,   392,     5,   319,   225,   226,
   227,   228,   229,   389,   113,    43,   114,   287,   426,   351,
   107,    44,   352,    64,   354,   156,   299,   162,   433,    68,
   363,   107,   300,   366,    80,   301,   138,   439,   163,   372,
   373,   374,   406,    48,   143,   144,   145,   146,   379,   360,
   362,   282,   248,   249,   250,   251,   252,   173,   304,   107,
   319,  -107,   307,  -107,    70,   179,   391,   182,   198,   158,
   180,   186,   187,   188,   189,   190,   191,   192,   107,   201,
   202,   203,   108,   404,    75,   196,   103,   104,   105,   106,
     9,   394,    10,    46,   412,   137,   107,   132,   133,   119,
   139,   401,   420,   207,   422,   208,    79,    46,    47,    48,
   113,   156,   114,    49,    50,    51,    52,    53,    54,    55,
    77,   434,   419,    48,   230,    81,   231,    49,    50,    51,
    52,    53,    54,    55,   429,   110,   247,    84,    85,    86,
    87,    88,    89,   293,   244,   279,   355,     1,     2,   438,
    17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
    27,    45,    48,   197,    28,   125,   126,   127,   128,    82,
   131,   263,   136,   111,    46,    47,    83,   308,   309,   310,
   311,   312,   313,   314,   315,   278,   316,    48,   321,   322,
    48,   285,   221,   222,    49,    50,    51,    52,    53,    54,
    55,   359,    29,    46,    47,   297,   263,   298,   323,   295,
   263,    95,    96,    97,   112,   120,   149,   121,   150,    48,
    46,    47,   142,    49,    50,    51,    52,    53,    54,    55,
   157,    46,    47,   165,   108,   166,    48,   151,   152,   306,
    49,    50,    51,    52,    53,    54,    55,    48,   276,   174,
   168,    49,    50,    51,    52,    53,    54,    55,   267,   177,
   268,   269,   270,   176,   277,   271,   272,   273,    49,    50,
    51,    52,    53,    54,    55,   308,   309,   310,   311,   312,
   313,   314,   315,   178,   316,   308,   309,   310,   311,   312,
   313,   314,   315,   183,   316,   193,   184,   194,   195,   158,
   381,   308,   309,   310,   311,   312,   313,   314,   315,   393,
   316,   308,   309,   310,   311,   312,   313,   314,   315,   107,
   316,   213,   233,   219,   180,   400,   234,   308,   309,   310,
   311,   312,   313,   314,   315,   418,   316,   308,   309,   310,
   311,   312,   313,   314,   315,   235,   316,   236,   237,   238,
   246,   428,   254,   308,   309,   310,   311,   312,   313,   314,
   315,   437,   316,   256,   258,   260,   261,   262,   283,   284,
   289,   290,   291,  -142,   292,    -8,   293,   296,   334,   340,
   337,   328,   329,   303,   330,   332,   346,   338,   343,   348,
   349,   350,   353,   365,   367,   358,   364,   368,   375,   390,
   370,   398,   383,   385,   397,   377,   403,   376,   409,   395,
   414,   402,   407,   413,   415,   423,   442,   417,   427,   424,
   430,   432,   435,   440,   443,     8,    30,    12,   280,   232,
   200,   167,   204,   215,   255,   224,   259,    40,   411,   305,
   371,   148,   223,   431,   425,   164,   387,   380,     0,   147
};

static const short yycheck[] = {     1,
     2,    21,    79,    45,    17,   142,   366,    19,    20,   310,
     5,   295,    45,    50,   315,   345,   217,    54,   126,   127,
   128,    45,    46,   131,     6,     7,     8,     9,    10,   359,
   231,    64,   362,    66,    47,    34,    35,    36,    37,    38,
    39,    40,    41,    63,    43,    65,    66,    67,   287,    57,
   334,    33,   412,    23,    24,    25,    64,     3,     4,    58,
   299,   300,   301,   393,    59,    45,    79,   170,   171,   172,
   400,    48,   175,    75,   434,   376,    45,   294,   186,   187,
   188,   189,   190,   367,    64,    59,    66,    64,   418,   328,
    57,    16,   331,    29,   333,   108,    58,    64,   428,   119,
   339,    57,    64,   342,    29,    67,   183,   437,    64,   348,
   349,   350,   396,    45,    84,    85,    86,    87,   357,   336,
   337,   258,   225,   226,   227,   228,   229,   129,   289,    57,
   347,    60,   293,    62,    67,   137,   375,   139,   158,    67,
    69,   143,   144,   145,   146,   147,   148,   149,    57,   161,
   162,   163,    61,   392,    67,   157,    49,    50,    51,    52,
    56,   378,    58,    29,   403,    17,    57,    31,    32,    60,
   183,   388,   411,    56,   413,    58,    61,    29,    30,    45,
    64,   194,    66,    49,    50,    51,    52,    53,    54,    55,
    67,   430,   409,    45,    56,    56,    58,    49,    50,    51,
    52,    53,    54,    55,   421,    63,   219,    11,    12,    13,
    14,    15,    16,    64,   216,   257,    67,     3,     4,   436,
     6,     7,     8,     9,    10,    11,    12,    13,    14,    15,
    16,    16,    45,    46,    20,    71,    72,    73,    74,    56,
    76,   254,    78,    63,    29,    30,    56,    34,    35,    36,
    37,    38,    39,    40,    41,   257,    43,    45,    46,    47,
    45,   263,    46,    47,    49,    50,    51,    52,    53,    54,
    55,    58,    58,    29,    30,    67,   289,    69,   298,   281,
   293,    56,    56,    56,    63,    56,    63,    25,    57,    45,
    29,    30,    59,    49,    50,    51,    52,    53,    54,    55,
    63,    29,    30,    68,    61,    60,    45,    46,    47,    65,
    49,    50,    51,    52,    53,    54,    55,    45,    29,    68,
    57,    49,    50,    51,    52,    53,    54,    55,    19,    60,
    21,    22,    23,    68,    45,    26,    27,    28,    49,    50,
    51,    52,    53,    54,    55,    34,    35,    36,    37,    38,
    39,    40,    41,    58,    43,    34,    35,    36,    37,    38,
    39,    40,    41,    60,    43,    45,    62,    60,    62,    67,
    59,    34,    35,    36,    37,    38,    39,    40,    41,    58,
    43,    34,    35,    36,    37,    38,    39,    40,    41,    57,
    43,    68,    65,    69,    69,    58,    68,    34,    35,    36,
    37,    38,    39,    40,    41,    58,    43,    34,    35,    36,
    37,    38,    39,    40,    41,    65,    43,    65,    65,    60,
    59,    58,    64,    34,    35,    36,    37,    38,    39,    40,
    41,    58,    43,    67,    59,    56,    67,    59,    68,    56,
    60,    65,    68,    45,    60,    64,    64,    56,    45,    45,
    58,    65,    65,    68,    64,    67,    18,    58,    64,    59,
    65,    68,    45,    56,    45,    60,    60,    59,    56,    46,
    69,    44,    59,    59,    59,    65,    56,    68,    42,    68,
    59,    65,    64,    57,    45,    60,     0,    59,    59,    65,
    60,    59,    59,    59,     0,     4,    14,     7,   257,   194,
   159,   124,   166,   177,   238,   183,   242,    14,   402,   292,
   347,    89,   182,   423,   417,   119,   364,   358,    -1,    88
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
#line 125 "xi-grammar.y"
{ yyval.modlist = yyvsp[0].modlist; modlist = yyvsp[0].modlist; ;
    break;}
case 2:
#line 129 "xi-grammar.y"
{ 
		  yyval.modlist = 0; 
		;
    break;}
case 3:
#line 133 "xi-grammar.y"
{ yyval.modlist = new ModuleList(lineno, yyvsp[-1].module, yyvsp[0].modlist); ;
    break;}
case 4:
#line 137 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 5:
#line 139 "xi-grammar.y"
{ yyval.intval = 1; ;
    break;}
case 6:
#line 143 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 7:
#line 145 "xi-grammar.y"
{ yyval.intval = 1; ;
    break;}
case 8:
#line 149 "xi-grammar.y"
{ yyval.strval = yyvsp[0].strval; ;
    break;}
case 9:
#line 153 "xi-grammar.y"
{ yyval.strval = yyvsp[0].strval; ;
    break;}
case 10:
#line 155 "xi-grammar.y"
{
		  char *tmp = new char[strlen(yyvsp[-3].strval)+strlen(yyvsp[0].strval)+3];
		  sprintf(tmp,"%s::%s", yyvsp[-3].strval, yyvsp[0].strval);
		  yyval.strval = tmp;
		;
    break;}
case 11:
#line 163 "xi-grammar.y"
{ 
		    yyval.module = new Module(lineno, yyvsp[-1].strval, yyvsp[0].conslist); 
		;
    break;}
case 12:
#line 167 "xi-grammar.y"
{  
		    yyval.module = new Module(lineno, yyvsp[-1].strval, yyvsp[0].conslist); 
		    yyval.module->setMain();
		;
    break;}
case 13:
#line 174 "xi-grammar.y"
{ yyval.conslist = 0; ;
    break;}
case 14:
#line 176 "xi-grammar.y"
{ yyval.conslist = yyvsp[-2].conslist; ;
    break;}
case 15:
#line 180 "xi-grammar.y"
{ yyval.conslist = 0; ;
    break;}
case 16:
#line 182 "xi-grammar.y"
{ yyval.conslist = new ConstructList(lineno, yyvsp[-1].construct, yyvsp[0].conslist); ;
    break;}
case 17:
#line 186 "xi-grammar.y"
{ if(yyvsp[-2].conslist) yyvsp[-2].conslist->setExtern(yyvsp[-4].intval); yyval.construct = yyvsp[-2].conslist; ;
    break;}
case 18:
#line 188 "xi-grammar.y"
{ yyvsp[0].module->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].module; ;
    break;}
case 19:
#line 190 "xi-grammar.y"
{ yyvsp[0].member->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].member; ;
    break;}
case 20:
#line 192 "xi-grammar.y"
{ yyvsp[-1].message->setExtern(yyvsp[-2].intval); yyval.construct = yyvsp[-1].message; ;
    break;}
case 21:
#line 194 "xi-grammar.y"
{ yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; ;
    break;}
case 22:
#line 196 "xi-grammar.y"
{ yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; ;
    break;}
case 23:
#line 198 "xi-grammar.y"
{ yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; ;
    break;}
case 24:
#line 200 "xi-grammar.y"
{ yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; ;
    break;}
case 25:
#line 202 "xi-grammar.y"
{ yyvsp[0].templat->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].templat; ;
    break;}
case 26:
#line 206 "xi-grammar.y"
{ yyval.tparam = new TParamType(yyvsp[0].type); ;
    break;}
case 27:
#line 208 "xi-grammar.y"
{ yyval.tparam = new TParamVal(yyvsp[0].strval); ;
    break;}
case 28:
#line 210 "xi-grammar.y"
{ yyval.tparam = new TParamVal(yyvsp[0].strval); ;
    break;}
case 29:
#line 214 "xi-grammar.y"
{ yyval.tparlist = new TParamList(yyvsp[0].tparam); ;
    break;}
case 30:
#line 216 "xi-grammar.y"
{ yyval.tparlist = new TParamList(yyvsp[-2].tparam, yyvsp[0].tparlist); ;
    break;}
case 31:
#line 220 "xi-grammar.y"
{ yyval.tparlist = 0; ;
    break;}
case 32:
#line 222 "xi-grammar.y"
{ yyval.tparlist = yyvsp[0].tparlist; ;
    break;}
case 33:
#line 226 "xi-grammar.y"
{ yyval.tparlist = 0; ;
    break;}
case 34:
#line 228 "xi-grammar.y"
{ yyval.tparlist = yyvsp[-1].tparlist; ;
    break;}
case 35:
#line 232 "xi-grammar.y"
{ yyval.type = new BuiltinType("int"); ;
    break;}
case 36:
#line 234 "xi-grammar.y"
{ yyval.type = new BuiltinType("long"); ;
    break;}
case 37:
#line 236 "xi-grammar.y"
{ yyval.type = new BuiltinType("short"); ;
    break;}
case 38:
#line 238 "xi-grammar.y"
{ yyval.type = new BuiltinType("char"); ;
    break;}
case 39:
#line 240 "xi-grammar.y"
{ yyval.type = new BuiltinType("unsigned int"); ;
    break;}
case 40:
#line 242 "xi-grammar.y"
{ yyval.type = new BuiltinType("unsigned long"); ;
    break;}
case 41:
#line 244 "xi-grammar.y"
{ yyval.type = new BuiltinType("unsigned short"); ;
    break;}
case 42:
#line 246 "xi-grammar.y"
{ yyval.type = new BuiltinType("unsigned char"); ;
    break;}
case 43:
#line 248 "xi-grammar.y"
{ yyval.type = new BuiltinType("long long"); ;
    break;}
case 44:
#line 250 "xi-grammar.y"
{ yyval.type = new BuiltinType("float"); ;
    break;}
case 45:
#line 252 "xi-grammar.y"
{ yyval.type = new BuiltinType("double"); ;
    break;}
case 46:
#line 254 "xi-grammar.y"
{ yyval.type = new BuiltinType("long double"); ;
    break;}
case 47:
#line 256 "xi-grammar.y"
{ yyval.type = new BuiltinType("void"); ;
    break;}
case 48:
#line 259 "xi-grammar.y"
{ yyval.ntype = new NamedType(yyvsp[-1].strval,yyvsp[0].tparlist); ;
    break;}
case 49:
#line 260 "xi-grammar.y"
{ yyval.ntype = new NamedType(yyvsp[-1].strval,yyvsp[0].tparlist); ;
    break;}
case 50:
#line 263 "xi-grammar.y"
{ yyval.type = yyvsp[0].type; ;
    break;}
case 51:
#line 265 "xi-grammar.y"
{ yyval.type = yyvsp[0].ntype; ;
    break;}
case 52:
#line 269 "xi-grammar.y"
{ yyval.ptype = new PtrType(yyvsp[-1].type); ;
    break;}
case 53:
#line 273 "xi-grammar.y"
{ yyvsp[-1].ptype->indirect(); yyval.ptype = yyvsp[-1].ptype; ;
    break;}
case 54:
#line 275 "xi-grammar.y"
{ yyvsp[-1].ptype->indirect(); yyval.ptype = yyvsp[-1].ptype; ;
    break;}
case 55:
#line 279 "xi-grammar.y"
{ yyval.ftype = new FuncType(yyvsp[-7].type, yyvsp[-4].strval, yyvsp[-1].plist); ;
    break;}
case 56:
#line 283 "xi-grammar.y"
{ yyval.type = yyvsp[0].type; ;
    break;}
case 57:
#line 285 "xi-grammar.y"
{ yyval.type = yyvsp[0].ptype; ;
    break;}
case 58:
#line 287 "xi-grammar.y"
{ yyval.type = yyvsp[0].ptype; ;
    break;}
case 59:
#line 289 "xi-grammar.y"
{ yyval.type = yyvsp[0].ftype; ;
    break;}
case 60:
#line 291 "xi-grammar.y"
{ yyval.type = new ReferenceType(yyvsp[-1].type); ;
    break;}
case 61:
#line 294 "xi-grammar.y"
{ yyval.type = yyvsp[0].type; ;
    break;}
case 62:
#line 298 "xi-grammar.y"
{ yyval.val = new Value(yyvsp[0].strval); ;
    break;}
case 63:
#line 300 "xi-grammar.y"
{ yyval.val = new Value(yyvsp[0].strval); ;
    break;}
case 64:
#line 304 "xi-grammar.y"
{ yyval.val = yyvsp[-1].val; ;
    break;}
case 65:
#line 308 "xi-grammar.y"
{ yyval.vallist = 0; ;
    break;}
case 66:
#line 310 "xi-grammar.y"
{ yyval.vallist = new ValueList(yyvsp[-1].val, yyvsp[0].vallist); ;
    break;}
case 67:
#line 314 "xi-grammar.y"
{ yyval.readonly = new Readonly(lineno, yyvsp[-2].type, yyvsp[-1].strval, yyvsp[0].vallist); ;
    break;}
case 68:
#line 318 "xi-grammar.y"
{ yyval.readonly = new Readonly(lineno, yyvsp[-2].type, yyvsp[0].strval, 0, 1); ;
    break;}
case 69:
#line 322 "xi-grammar.y"
{ yyval.intval = 0;;
    break;}
case 70:
#line 324 "xi-grammar.y"
{ yyval.intval = 0;;
    break;}
case 71:
#line 328 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 72:
#line 330 "xi-grammar.y"
{ 
		  /*
		  printf("Warning: Message attributes are being phased out.\n");
		  printf("Warning: Please remove them from interface files.\n");
		  */
		  yyval.intval = yyvsp[-1].intval; 
		;
    break;}
case 73:
#line 340 "xi-grammar.y"
{ yyval.intval = yyvsp[0].intval; ;
    break;}
case 74:
#line 342 "xi-grammar.y"
{ yyval.intval = yyvsp[-2].intval | yyvsp[0].intval; ;
    break;}
case 75:
#line 346 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 76:
#line 348 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 77:
#line 352 "xi-grammar.y"
{ yyval.cattr = 0; ;
    break;}
case 78:
#line 354 "xi-grammar.y"
{ yyval.cattr = yyvsp[-1].cattr; ;
    break;}
case 79:
#line 358 "xi-grammar.y"
{ yyval.cattr = yyvsp[0].cattr; ;
    break;}
case 80:
#line 360 "xi-grammar.y"
{ yyval.cattr = yyvsp[-2].cattr | yyvsp[0].cattr; ;
    break;}
case 81:
#line 364 "xi-grammar.y"
{ yyval.cattr = Chare::CMIGRATABLE; ;
    break;}
case 82:
#line 368 "xi-grammar.y"
{ yyval.mv = new MsgVar(yyvsp[-4].type, yyvsp[-3].strval); ;
    break;}
case 83:
#line 372 "xi-grammar.y"
{ yyval.mvlist = new MsgVarList(yyvsp[0].mv); ;
    break;}
case 84:
#line 374 "xi-grammar.y"
{ yyval.mvlist = new MsgVarList(yyvsp[-1].mv, yyvsp[0].mvlist); ;
    break;}
case 85:
#line 378 "xi-grammar.y"
{ yyval.message = new Message(lineno, yyvsp[0].ntype); ;
    break;}
case 86:
#line 380 "xi-grammar.y"
{ yyval.message = new Message(lineno, yyvsp[-3].ntype, yyvsp[-1].mvlist); ;
    break;}
case 87:
#line 384 "xi-grammar.y"
{ yyval.typelist = 0; ;
    break;}
case 88:
#line 386 "xi-grammar.y"
{ yyval.typelist = yyvsp[0].typelist; ;
    break;}
case 89:
#line 390 "xi-grammar.y"
{ yyval.typelist = new TypeList(yyvsp[0].ntype); ;
    break;}
case 90:
#line 392 "xi-grammar.y"
{ yyval.typelist = new TypeList(yyvsp[-2].ntype, yyvsp[0].typelist); ;
    break;}
case 91:
#line 396 "xi-grammar.y"
{ yyval.chare = new Chare(lineno, yyvsp[-3].cattr, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 92:
#line 398 "xi-grammar.y"
{ yyval.chare = new MainChare(lineno, yyvsp[-3].cattr, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 93:
#line 402 "xi-grammar.y"
{ yyval.chare = new Group(lineno, yyvsp[-3].cattr, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 94:
#line 406 "xi-grammar.y"
{ yyval.chare = new NodeGroup(lineno, yyvsp[-3].cattr, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 95:
#line 410 "xi-grammar.y"
{/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",yyvsp[-2].strval);
			yyval.ntype = new NamedType(buf); 
		;
    break;}
case 96:
#line 416 "xi-grammar.y"
{ yyval.ntype = new NamedType(yyvsp[-1].strval); ;
    break;}
case 97:
#line 420 "xi-grammar.y"
{  yyval.chare = new Array(lineno, 0, yyvsp[-3].ntype, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 98:
#line 424 "xi-grammar.y"
{ yyval.chare = new Chare(lineno, yyvsp[-3].cattr, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist);;
    break;}
case 99:
#line 426 "xi-grammar.y"
{ yyval.chare = new MainChare(lineno, yyvsp[-3].cattr, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 100:
#line 430 "xi-grammar.y"
{ yyval.chare = new Group(lineno, yyvsp[-3].cattr, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 101:
#line 434 "xi-grammar.y"
{ yyval.chare = new NodeGroup( lineno, yyvsp[-3].cattr, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 102:
#line 438 "xi-grammar.y"
{ yyval.chare = new Array( lineno, 0, yyvsp[-3].ntype, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 103:
#line 442 "xi-grammar.y"
{ yyval.message = new Message(lineno, new NamedType(yyvsp[-1].strval)); ;
    break;}
case 104:
#line 444 "xi-grammar.y"
{ yyval.message = new Message(lineno, new NamedType(yyvsp[-4].strval), yyvsp[-2].mvlist); ;
    break;}
case 105:
#line 448 "xi-grammar.y"
{ yyval.type = 0; ;
    break;}
case 106:
#line 450 "xi-grammar.y"
{ yyval.type = yyvsp[0].type; ;
    break;}
case 107:
#line 454 "xi-grammar.y"
{ yyval.strval = 0; ;
    break;}
case 108:
#line 456 "xi-grammar.y"
{ yyval.strval = yyvsp[0].strval; ;
    break;}
case 109:
#line 458 "xi-grammar.y"
{ yyval.strval = yyvsp[0].strval; ;
    break;}
case 110:
#line 462 "xi-grammar.y"
{ yyval.tvar = new TType(new NamedType(yyvsp[-1].strval), yyvsp[0].type); ;
    break;}
case 111:
#line 464 "xi-grammar.y"
{ yyval.tvar = new TFunc(yyvsp[-1].ftype, yyvsp[0].strval); ;
    break;}
case 112:
#line 466 "xi-grammar.y"
{ yyval.tvar = new TName(yyvsp[-2].type, yyvsp[-1].strval, yyvsp[0].strval); ;
    break;}
case 113:
#line 470 "xi-grammar.y"
{ yyval.tvarlist = new TVarList(yyvsp[0].tvar); ;
    break;}
case 114:
#line 472 "xi-grammar.y"
{ yyval.tvarlist = new TVarList(yyvsp[-2].tvar, yyvsp[0].tvarlist); ;
    break;}
case 115:
#line 476 "xi-grammar.y"
{ yyval.tvarlist = yyvsp[-1].tvarlist; ;
    break;}
case 116:
#line 480 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); ;
    break;}
case 117:
#line 482 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); ;
    break;}
case 118:
#line 484 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); ;
    break;}
case 119:
#line 486 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); ;
    break;}
case 120:
#line 488 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].message); yyvsp[0].message->setTemplate(yyval.templat); ;
    break;}
case 121:
#line 492 "xi-grammar.y"
{ yyval.mbrlist = 0; ;
    break;}
case 122:
#line 494 "xi-grammar.y"
{ yyval.mbrlist = yyvsp[-2].mbrlist; ;
    break;}
case 123:
#line 498 "xi-grammar.y"
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
#line 517 "xi-grammar.y"
{ yyval.mbrlist = new MemberList(yyvsp[-1].member, yyvsp[0].mbrlist); ;
    break;}
case 125:
#line 521 "xi-grammar.y"
{ yyval.member = yyvsp[-1].readonly; ;
    break;}
case 126:
#line 523 "xi-grammar.y"
{ yyval.member = yyvsp[-1].readonly; ;
    break;}
case 128:
#line 526 "xi-grammar.y"
{ yyval.member = yyvsp[-1].member; ;
    break;}
case 129:
#line 528 "xi-grammar.y"
{ yyval.member = yyvsp[-1].pupable; ;
    break;}
case 130:
#line 532 "xi-grammar.y"
{ yyval.member = new InitCall(lineno, yyvsp[0].strval, 1); ;
    break;}
case 131:
#line 534 "xi-grammar.y"
{ yyval.member = new InitCall(lineno, yyvsp[-3].strval, 1); ;
    break;}
case 132:
#line 536 "xi-grammar.y"
{ printf("Warning: deprecated use of initcall. Use initnode or initproc instead.\n"); 
		  yyval.member = new InitCall(lineno, yyvsp[0].strval, 1); ;
    break;}
case 133:
#line 539 "xi-grammar.y"
{ printf("Warning: deprecated use of initcall. Use initnode or initproc instead.\n");
		  yyval.member = new InitCall(lineno, yyvsp[-3].strval, 1); ;
    break;}
case 134:
#line 544 "xi-grammar.y"
{ yyval.member = new InitCall(lineno, yyvsp[0].strval, 0); ;
    break;}
case 135:
#line 546 "xi-grammar.y"
{ yyval.member = new InitCall(lineno, yyvsp[-3].strval, 0); ;
    break;}
case 136:
#line 550 "xi-grammar.y"
{ yyval.pupable = new PUPableClass(lineno,yyvsp[0].strval,0); ;
    break;}
case 137:
#line 552 "xi-grammar.y"
{ yyval.pupable = new PUPableClass(lineno,yyvsp[-2].strval,yyvsp[0].pupable); ;
    break;}
case 138:
#line 556 "xi-grammar.y"
{ yyval.member = yyvsp[-1].entry; ;
    break;}
case 139:
#line 558 "xi-grammar.y"
{ yyval.member = yyvsp[0].member; ;
    break;}
case 140:
#line 562 "xi-grammar.y"
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
case 141:
#line 573 "xi-grammar.y"
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
case 142:
#line 586 "xi-grammar.y"
{ yyval.type = new BuiltinType("void"); ;
    break;}
case 143:
#line 588 "xi-grammar.y"
{ yyval.type = yyvsp[0].ptype; ;
    break;}
case 144:
#line 592 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 145:
#line 594 "xi-grammar.y"
{ yyval.intval = yyvsp[-1].intval; ;
    break;}
case 146:
#line 598 "xi-grammar.y"
{ yyval.intval = yyvsp[0].intval; ;
    break;}
case 147:
#line 600 "xi-grammar.y"
{ yyval.intval = yyvsp[-2].intval | yyvsp[0].intval; ;
    break;}
case 148:
#line 604 "xi-grammar.y"
{ yyval.intval = STHREADED; ;
    break;}
case 149:
#line 606 "xi-grammar.y"
{ yyval.intval = SSYNC; ;
    break;}
case 150:
#line 608 "xi-grammar.y"
{ yyval.intval = SLOCKED; ;
    break;}
case 151:
#line 610 "xi-grammar.y"
{ yyval.intval = SCREATEHERE; ;
    break;}
case 152:
#line 612 "xi-grammar.y"
{ yyval.intval = SCREATEHOME; ;
    break;}
case 153:
#line 614 "xi-grammar.y"
{ yyval.intval = SNOKEEP; ;
    break;}
case 154:
#line 616 "xi-grammar.y"
{ yyval.intval = SIMMEDIATE; ;
    break;}
case 155:
#line 620 "xi-grammar.y"
{ yyval.val = new Value(yyvsp[0].strval); ;
    break;}
case 156:
#line 622 "xi-grammar.y"
{ yyval.val = new Value(yyvsp[0].strval); ;
    break;}
case 157:
#line 624 "xi-grammar.y"
{ yyval.val = new Value(yyvsp[0].strval); ;
    break;}
case 158:
#line 628 "xi-grammar.y"
{ yyval.strval = ""; ;
    break;}
case 159:
#line 630 "xi-grammar.y"
{ yyval.strval = yyvsp[0].strval; ;
    break;}
case 160:
#line 632 "xi-grammar.y"
{  /*Returned only when in_bracket*/
			char *tmp = new char[strlen(yyvsp[-4].strval)+strlen(yyvsp[-2].strval)+strlen(yyvsp[0].strval)+3];
			sprintf(tmp,"%s[%s]%s", yyvsp[-4].strval, yyvsp[-2].strval, yyvsp[0].strval);
			yyval.strval = tmp;
		;
    break;}
case 161:
#line 638 "xi-grammar.y"
{ /*Returned only when in_braces*/
			char *tmp = new char[strlen(yyvsp[-4].strval)+strlen(yyvsp[-2].strval)+strlen(yyvsp[0].strval)+3];
			sprintf(tmp,"%s{%s}%s", yyvsp[-4].strval, yyvsp[-2].strval, yyvsp[0].strval);
			yyval.strval = tmp;
		;
    break;}
case 162:
#line 644 "xi-grammar.y"
{ /*Returned only when in_braces*/
			char *tmp = new char[strlen(yyvsp[-4].strval)+strlen(yyvsp[-2].strval)+strlen(yyvsp[0].strval)+3];
			sprintf(tmp,"%s(%s)%s", yyvsp[-4].strval, yyvsp[-2].strval, yyvsp[0].strval);
			yyval.strval = tmp;
		;
    break;}
case 163:
#line 650 "xi-grammar.y"
{ /*Returned only when in_braces*/
			char *tmp = new char[strlen(yyvsp[-2].strval)+strlen(yyvsp[0].strval)+3];
			sprintf(tmp,"(%s)%s", yyvsp[-2].strval, yyvsp[0].strval);
			yyval.strval = tmp;
		;
    break;}
case 164:
#line 658 "xi-grammar.y"
{  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			yyval.pname = new Parameter(lineno, yyvsp[-2].type,yyvsp[-1].strval);
		;
    break;}
case 165:
#line 665 "xi-grammar.y"
{ 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			yyval.intval = 0;
		;
    break;}
case 166:
#line 673 "xi-grammar.y"
{ 
			in_braces=0;
			yyval.intval = 0;
		;
    break;}
case 167:
#line 680 "xi-grammar.y"
{ yyval.pname = new Parameter(lineno, yyvsp[0].type);;
    break;}
case 168:
#line 682 "xi-grammar.y"
{ yyval.pname = new Parameter(lineno, yyvsp[-1].type,yyvsp[0].strval);;
    break;}
case 169:
#line 684 "xi-grammar.y"
{ yyval.pname = new Parameter(lineno, yyvsp[-3].type,yyvsp[-2].strval,0,yyvsp[0].val);;
    break;}
case 170:
#line 686 "xi-grammar.y"
{ /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			yyval.pname = new Parameter(lineno, yyvsp[-2].pname->getType(), yyvsp[-2].pname->getName() ,yyvsp[-1].strval);
		;
    break;}
case 171:
#line 693 "xi-grammar.y"
{ yyval.plist = new ParamList(yyvsp[0].pname); ;
    break;}
case 172:
#line 695 "xi-grammar.y"
{ yyval.plist = new ParamList(yyvsp[-2].pname,yyvsp[0].plist); ;
    break;}
case 173:
#line 699 "xi-grammar.y"
{ yyval.plist = yyvsp[-1].plist; ;
    break;}
case 174:
#line 701 "xi-grammar.y"
{ yyval.plist = 0; ;
    break;}
case 175:
#line 705 "xi-grammar.y"
{ yyval.val = 0; ;
    break;}
case 176:
#line 707 "xi-grammar.y"
{ yyval.val = new Value(yyvsp[0].strval); ;
    break;}
case 177:
#line 711 "xi-grammar.y"
{ yyval.sc = 0; ;
    break;}
case 178:
#line 713 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SSDAGENTRY, yyvsp[0].sc); ;
    break;}
case 179:
#line 715 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SSDAGENTRY, yyvsp[-1].sc); ;
    break;}
case 180:
#line 719 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SSLIST, yyvsp[0].sc); ;
    break;}
case 181:
#line 721 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SSLIST, yyvsp[-1].sc, yyvsp[0].sc);  ;
    break;}
case 182:
#line 725 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SOLIST, yyvsp[0].sc); ;
    break;}
case 183:
#line 727 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SOLIST, yyvsp[-1].sc, yyvsp[0].sc); ;
    break;}
case 184:
#line 731 "xi-grammar.y"
{ yyval.sc = 0; ;
    break;}
case 185:
#line 733 "xi-grammar.y"
{ yyval.sc = yyvsp[-1].sc; ;
    break;}
case 186:
#line 737 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SPUBLISHES, new SdagConstruct(SIDENT, yyvsp[0].strval)); ;
    break;}
case 187:
#line 739 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SPUBLISHES, new SdagConstruct(SIDENT, yyvsp[-2].strval), yyvsp[0].sc);  ;
    break;}
case 188:
#line 743 "xi-grammar.y"
{ RemoveSdagComments(yyvsp[-2].strval);
		   yyval.sc = new SdagConstruct(SATOMIC, new XStr(yyvsp[-2].strval), yyvsp[0].sc, 0,0,0,0, 0 ); ;
    break;}
case 189:
#line 746 "xi-grammar.y"
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
case 190:
#line 760 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SWHEN, 0, 0,0,0,0,0,yyvsp[-2].entrylist); ;
    break;}
case 191:
#line 762 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SWHEN, 0, 0, 0,0,0, yyvsp[0].sc, yyvsp[-1].entrylist); ;
    break;}
case 192:
#line 764 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SWHEN, 0, 0, 0,0,0, yyvsp[-1].sc, yyvsp[-3].entrylist); ;
    break;}
case 193:
#line 766 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SOVERLAP,0, 0,0,0,0,yyvsp[-1].sc, 0); ;
    break;}
case 194:
#line 768 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SFOR, 0, new SdagConstruct(SINT_EXPR, yyvsp[-8].strval), new SdagConstruct(SINT_EXPR, yyvsp[-6].strval),
		             new SdagConstruct(SINT_EXPR, yyvsp[-4].strval), 0, yyvsp[-1].sc, 0); ;
    break;}
case 195:
#line 771 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SFOR, 0, new SdagConstruct(SINT_EXPR, yyvsp[-6].strval), new SdagConstruct(SINT_EXPR, yyvsp[-4].strval), 
		         new SdagConstruct(SINT_EXPR, yyvsp[-2].strval), 0, yyvsp[0].sc, 0); ;
    break;}
case 196:
#line 774 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SFORALL, 0, new SdagConstruct(SIDENT, yyvsp[-9].strval), new SdagConstruct(SINT_EXPR, yyvsp[-6].strval), 
		             new SdagConstruct(SINT_EXPR, yyvsp[-4].strval), new SdagConstruct(SINT_EXPR, yyvsp[-2].strval), yyvsp[0].sc, 0); ;
    break;}
case 197:
#line 777 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SFORALL, 0, new SdagConstruct(SIDENT, yyvsp[-11].strval), new SdagConstruct(SINT_EXPR, yyvsp[-8].strval), 
		                 new SdagConstruct(SINT_EXPR, yyvsp[-6].strval), new SdagConstruct(SINT_EXPR, yyvsp[-4].strval), yyvsp[-1].sc, 0); ;
    break;}
case 198:
#line 780 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, yyvsp[-3].strval), yyvsp[0].sc,0,0,yyvsp[-1].sc,0); ;
    break;}
case 199:
#line 782 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, yyvsp[-5].strval), yyvsp[0].sc,0,0,yyvsp[-2].sc,0); ;
    break;}
case 200:
#line 784 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, yyvsp[-2].strval), 0,0,0,yyvsp[0].sc,0); ;
    break;}
case 201:
#line 786 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SWHILE, 0, new SdagConstruct(SINT_EXPR, yyvsp[-4].strval), 0,0,0,yyvsp[-1].sc,0); ;
    break;}
case 202:
#line 788 "xi-grammar.y"
{ yyval.sc = yyvsp[-1].sc; ;
    break;}
case 203:
#line 792 "xi-grammar.y"
{ yyval.sc = 0; ;
    break;}
case 204:
#line 794 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SELSE, 0,0,0,0,0, yyvsp[0].sc,0); ;
    break;}
case 205:
#line 796 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SELSE, 0,0,0,0,0, yyvsp[-1].sc,0); ;
    break;}
case 206:
#line 799 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SFORWARD, new SdagConstruct(SIDENT, yyvsp[0].strval)); ;
    break;}
case 207:
#line 801 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SFORWARD, new SdagConstruct(SIDENT, yyvsp[-2].strval), yyvsp[0].sc);  ;
    break;}
case 208:
#line 805 "xi-grammar.y"
{ in_int_expr = 0; yyval.intval = 0; ;
    break;}
case 209:
#line 809 "xi-grammar.y"
{ in_int_expr = 1; yyval.intval = 0; ;
    break;}
case 210:
#line 813 "xi-grammar.y"
{ 
		  if (yyvsp[0].plist != 0)
		     yyval.entry = new Entry(lineno, 0, 0, yyvsp[-1].strval, yyvsp[0].plist, 0, 0, 0, 0); 
		  else
		     yyval.entry = new Entry(lineno, 0, 0, yyvsp[-1].strval, 
				new ParamList(new Parameter(0, new BuiltinType("void"))), 0, 0, 0, 0); 
		;
    break;}
case 211:
#line 821 "xi-grammar.y"
{ if (yyvsp[0].plist != 0)
		    yyval.entry = new Entry(lineno, 0, 0, yyvsp[-4].strval, yyvsp[0].plist, 0, 0, yyvsp[-2].strval, 0); 
		  else
		    yyval.entry = new Entry(lineno, 0, 0, yyvsp[-4].strval, new ParamList(new Parameter(0, new BuiltinType("void"))), 0, 0, yyvsp[-2].strval, 0); 
		;
    break;}
case 212:
#line 829 "xi-grammar.y"
{ yyval.entrylist = new EntryList(yyvsp[0].entry); ;
    break;}
case 213:
#line 831 "xi-grammar.y"
{ yyval.entrylist = new EntryList(yyvsp[-2].entry,yyvsp[0].entrylist); ;
    break;}
case 214:
#line 835 "xi-grammar.y"
{ in_bracket=1; ;
    break;}
case 215:
#line 838 "xi-grammar.y"
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
#line 842 "xi-grammar.y"

void yyerror(const char *mesg)
{
  cout << cur_file<<":"<<lineno<<": Charmxi syntax error> " << mesg << endl;
  // return 0;
}
