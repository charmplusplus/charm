
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
#define	IDENT	295
#define	NUMBER	296
#define	LITERAL	297
#define	CPROGRAM	298
#define	INT	299
#define	LONG	300
#define	SHORT	301
#define	CHAR	302
#define	FLOAT	303
#define	DOUBLE	304
#define	UNSIGNED	305

#line 2 "xi-grammar.y"

#include <iostream.h>
#include "xi-symbol.h"
#include "EToken.h"
extern int yylex (void) ;
void yyerror(const char *);
extern unsigned int lineno;
extern int in_bracket,in_braces,in_int_expr;
ModuleList *modlist;


#line 14 "xi-grammar.y"
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



#define	YYFINAL		408
#define	YYFLAG		-32768
#define	YYNTBASE	66

#define YYTRANSLATE(x) ((unsigned)(x) <= 305 ? yytranslate[x] : 152)

static const char yytranslate[] = {     0,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,    62,     2,    60,
    61,    59,     2,    56,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,    53,    52,    57,
    65,    58,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
    63,     2,    64,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,    54,     2,    55,     2,     2,     2,     2,     2,
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
    47,    48,    49,    50,    51
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
   472,   474,   476,   477,   479,   485,   491,   497,   501,   503,
   505,   508,   513,   517,   519,   523,   527,   530,   531,   535,
   536,   538,   542,   544,   547,   549,   552,   557,   562,   566,
   572,   577,   589,   599,   612,   627,   634,   643,   649,   657,
   661,   662,   665,   670,   672,   676,   678,   680,   683,   689,
   691,   695,   697
};

static const short yyrhs[] = {    67,
     0,     0,    72,    67,     0,     0,     5,     0,     0,    52,
     0,    41,     0,    41,     0,    71,    53,    53,    41,     0,
     3,    70,    73,     0,     4,    70,    73,     0,    52,     0,
    54,    74,    55,    69,     0,     0,    75,    74,     0,    68,
    54,    74,    55,    69,     0,    68,    72,     0,    68,   123,
     0,    68,   102,    52,     0,    68,   105,     0,    68,   106,
     0,    68,   107,     0,    68,   109,     0,    68,   120,     0,
    87,     0,    42,     0,    43,     0,    76,     0,    76,    56,
    77,     0,     0,    77,     0,     0,    57,    78,    58,     0,
    45,     0,    46,     0,    47,     0,    48,     0,    51,    45,
     0,    51,    46,     0,    51,    47,     0,    51,    48,     0,
    46,    46,     0,    49,     0,    50,     0,    46,    50,     0,
    27,     0,    70,    79,     0,    71,    79,     0,    80,     0,
    82,     0,    83,    59,     0,    84,    59,     0,    85,    59,
     0,    87,    60,    59,    70,    61,    60,   137,    61,     0,
    83,     0,    84,     0,    85,     0,    86,     0,    87,    62,
     0,    28,    87,     0,    42,     0,    71,     0,    63,    88,
    64,     0,     0,    89,    90,     0,     6,    87,    71,    90,
     0,     6,    14,    83,    59,    70,     0,     0,    27,     0,
     0,    63,    95,    64,     0,    96,     0,    96,    56,    95,
     0,    29,     0,    30,     0,     0,    63,    98,    64,     0,
    99,     0,    99,    56,    98,     0,    23,     0,    87,    70,
    63,    64,    52,     0,   100,     0,   100,   101,     0,    14,
    94,    81,     0,    14,    94,    81,    54,   101,    55,     0,
     0,    53,   104,     0,    81,     0,    81,    56,   104,     0,
     9,    97,    81,   103,   121,     0,    10,    97,    81,   103,
   121,     0,    11,    97,    81,   103,   121,     0,    12,    97,
    81,   103,   121,     0,    63,    42,    70,    64,     0,    63,
    70,    64,     0,    13,   108,    81,   103,   121,     0,     9,
    97,    70,   103,   121,     0,    10,    97,    70,   103,   121,
     0,    11,    97,    70,   103,   121,     0,    12,    97,    70,
   103,   121,     0,    13,   108,    70,   103,   121,     0,    14,
    94,    70,    52,     0,    14,    94,    70,    54,   101,    55,
    52,     0,     0,    65,    87,     0,     0,    65,    42,     0,
    65,    43,     0,    15,    70,   115,     0,    86,   116,     0,
    87,    70,   116,     0,   117,     0,   117,    56,   118,     0,
    18,    57,   118,    58,     0,   119,   110,     0,   119,   111,
     0,   119,   112,     0,   119,   113,     0,   119,   114,     0,
    52,     0,    54,   122,    55,    69,     0,     0,   126,   122,
     0,    91,    52,     0,    92,    52,     0,   124,    52,     0,
     8,   125,    52,     0,     7,    93,    71,     0,     7,    93,
    71,    60,    93,    61,     0,    71,     0,    71,    56,   125,
     0,   127,    52,     0,   123,     0,    31,   129,   128,    70,
   138,   139,   140,     0,    31,   129,    70,   138,   140,     0,
    27,     0,    84,     0,     0,    63,   130,    64,     0,   131,
     0,   131,    56,   130,     0,    17,     0,    19,     0,    20,
     0,    24,     0,    25,     0,    26,     0,    21,     0,    43,
     0,    42,     0,    71,     0,     0,    44,     0,    44,    63,
   133,    64,   133,     0,    44,    54,   133,    55,   133,     0,
    44,    60,   133,    61,   133,     0,    87,    70,    63,     0,
    54,     0,    87,     0,    87,    70,     0,    87,    70,    65,
   132,     0,   134,   133,    64,     0,   136,     0,   136,    56,
   137,     0,    60,   137,    61,     0,    60,    61,     0,     0,
    16,    65,    42,     0,     0,   143,     0,    54,   141,    55,
     0,   143,     0,   143,   141,     0,   143,     0,   143,   141,
     0,    37,   135,   133,    55,     0,    35,   149,    54,    55,
     0,    35,   149,   143,     0,    35,   149,    54,   141,    55,
     0,    36,    54,   142,    55,     0,    32,   147,   133,    52,
   133,    52,   133,   146,    54,   141,    55,     0,    32,   147,
   133,    52,   133,    52,   133,   146,   143,     0,    33,    63,
    41,    64,   147,   133,    53,   133,    56,   133,   146,   143,
     0,    33,    63,    41,    64,   147,   133,    53,   133,    56,
   133,   146,    54,   141,    55,     0,    39,   147,   133,   146,
   143,   144,     0,    39,   147,   133,   146,    54,   141,    55,
   144,     0,    34,   147,   133,   146,   143,     0,    34,   147,
   133,   146,    54,   141,    55,     0,    38,   145,    52,     0,
     0,    40,   143,     0,    40,    54,   141,    55,     0,    41,
     0,    41,    56,   145,     0,    61,     0,    60,     0,    41,
   138,     0,    41,   150,   133,   151,   138,     0,   148,     0,
   148,    56,   149,     0,    63,     0,    64,     0
};

#endif

#if YYDEBUG != 0
static const short yyrline[] = { 0,
   119,   123,   125,   129,   131,   135,   137,   141,   145,   147,
   155,   157,   161,   163,   167,   169,   173,   175,   177,   179,
   181,   183,   185,   187,   189,   193,   195,   197,   201,   203,
   207,   209,   213,   215,   219,   221,   223,   225,   227,   229,
   231,   233,   235,   237,   239,   241,   243,   247,   248,   250,
   252,   256,   260,   262,   266,   270,   272,   274,   276,   278,
   281,   285,   287,   291,   295,   297,   301,   305,   309,   311,
   315,   317,   327,   329,   333,   335,   339,   341,   345,   347,
   351,   355,   359,   361,   365,   367,   371,   373,   377,   379,
   383,   385,   389,   393,   397,   403,   407,   411,   413,   417,
   421,   425,   429,   431,   435,   437,   441,   443,   445,   449,
   451,   453,   457,   459,   463,   467,   469,   471,   473,   475,
   479,   481,   485,   487,   491,   493,   495,   497,   501,   503,
   507,   509,   513,   515,   519,   530,   543,   545,   549,   551,
   555,   557,   561,   563,   565,   567,   569,   571,   573,   577,
   579,   581,   585,   587,   589,   595,   601,   609,   616,   624,
   626,   628,   630,   637,   639,   643,   645,   649,   651,   655,
   657,   659,   663,   665,   669,   671,   675,   680,   682,   684,
   686,   688,   691,   694,   697,   700,   702,   704,   706,   708,
   712,   714,   716,   719,   721,   725,   729,   733,   741,   749,
   751,   755,   758
};
#endif


#if YYDEBUG != 0 || defined (YYERROR_VERBOSE)

static const char * const yytname[] = {   "$","error","$undefined.","MODULE",
"MAINMODULE","EXTERN","READONLY","INITCALL","PUPABLE","CHARE","MAINCHARE","GROUP",
"NODEGROUP","ARRAY","MESSAGE","CLASS","STACKSIZE","THREADED","TEMPLATE","SYNC",
"EXCLUSIVE","IMMEDIATE","VIRTUAL","MIGRATABLE","CREATEHERE","CREATEHOME","NOKEEP",
"VOID","CONST","PACKED","VARSIZE","ENTRY","FOR","FORALL","WHILE","WHEN","OVERLAP",
"ATOMIC","FORWARD","IF","ELSE","IDENT","NUMBER","LITERAL","CPROGRAM","INT","LONG",
"SHORT","CHAR","FLOAT","DOUBLE","UNSIGNED","';'","':'","'{'","'}'","','","'<'",
"'>'","'*'","'('","')'","'&'","'['","']'","'='","File","ModuleEList","OptExtern",
"OptSemiColon","Name","QualName","Module","ConstructEList","ConstructList","Construct",
"TParam","TParamList","TParamEList","OptTParams","BuiltinType","NamedType","QualNamedType",
"SimpleType","OnePtrType","PtrType","FuncType","Type","ArrayDim","Dim","DimList",
"Readonly","ReadonlyMsg","OptVoid","MAttribs","MAttribList","MAttrib","CAttribs",
"CAttribList","CAttrib","Var","VarList","Message","OptBaseList","BaseList","Chare",
"Group","NodeGroup","ArrayIndexType","Array","TChare","TGroup","TNodeGroup",
"TArray","TMessage","OptTypeInit","OptNameInit","TVar","TVarList","TemplateSpec",
"Template","MemberEList","MemberList","NonEntryMember","InitCall","PUPableClass",
"Member","Entry","EReturn","EAttribs","EAttribList","EAttrib","DefaultParameter",
"CCode","ParamBracketStart","ParamBraceStart","Parameter","ParamList","EParameters",
"OptStackSize","OptSdagCode","Slist","Olist","SingleConstruct","HasElse","ForwardList",
"EndIntExpr","StartIntExpr","SEntry","SEntryList","SParamBracketStart","SParamBracketEnd", NULL
};
#endif

static const short yyr1[] = {     0,
    66,    67,    67,    68,    68,    69,    69,    70,    71,    71,
    72,    72,    73,    73,    74,    74,    75,    75,    75,    75,
    75,    75,    75,    75,    75,    76,    76,    76,    77,    77,
    78,    78,    79,    79,    80,    80,    80,    80,    80,    80,
    80,    80,    80,    80,    80,    80,    80,    81,    82,    83,
    83,    84,    85,    85,    86,    87,    87,    87,    87,    87,
    87,    88,    88,    89,    90,    90,    91,    92,    93,    93,
    94,    94,    95,    95,    96,    96,    97,    97,    98,    98,
    99,   100,   101,   101,   102,   102,   103,   103,   104,   104,
   105,   105,   106,   107,   108,   108,   109,   110,   110,   111,
   112,   113,   114,   114,   115,   115,   116,   116,   116,   117,
   117,   117,   118,   118,   119,   120,   120,   120,   120,   120,
   121,   121,   122,   122,   123,   123,   123,   123,   124,   124,
   125,   125,   126,   126,   127,   127,   128,   128,   129,   129,
   130,   130,   131,   131,   131,   131,   131,   131,   131,   132,
   132,   132,   133,   133,   133,   133,   133,   134,   135,   136,
   136,   136,   136,   137,   137,   138,   138,   139,   139,   140,
   140,   140,   141,   141,   142,   142,   143,   143,   143,   143,
   143,   143,   143,   143,   143,   143,   143,   143,   143,   143,
   144,   144,   144,   145,   145,   146,   147,   148,   148,   149,
   149,   150,   151
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
     1,     1,     0,     1,     5,     5,     5,     3,     1,     1,
     2,     4,     3,     1,     3,     3,     2,     0,     3,     0,
     1,     3,     1,     2,     1,     2,     4,     4,     3,     5,
     4,    11,     9,    12,    14,     6,     8,     5,     7,     3,
     0,     2,     4,     1,     3,     1,     1,     2,     5,     1,
     3,     1,     1
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
    90,     0,     0,     6,   124,   133,     0,     0,   160,   153,
   164,     0,   143,   144,   145,   149,   146,   147,   148,     0,
   141,    47,     9,     0,     0,   138,     0,   122,     0,   104,
   161,   154,     0,     0,    55,   140,     0,     0,   170,     0,
    82,   158,     0,   153,   153,   153,   163,   165,   142,   167,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
   136,   171,   168,   151,   150,   152,   162,     0,     0,     0,
   166,   197,   153,     0,   153,     0,   200,     0,     0,   159,
   153,   194,     0,   153,     0,   173,     0,   170,   153,   153,
   153,     0,     0,     0,   202,   198,   153,     0,     0,   179,
     0,   175,     0,     0,   190,     0,   172,   174,     0,   135,
   156,   157,   155,   153,     0,   196,     0,     0,   201,   178,
     0,   181,   176,   177,   195,     0,   169,     0,   153,     0,
   188,   203,     0,   180,     0,   191,   153,     0,     0,   199,
     0,     0,   186,     0,   153,   189,   191,     0,   192,     0,
     0,   187,     0,     0,   183,   153,   193,     0,     0,   182,
     0,     0,   184,     0,   185,     0,     0,     0
};

static const short yydefgoto[] = {   406,
     3,    14,    92,   116,    53,     4,    11,    15,    16,   145,
   146,   147,   103,    54,   193,    55,    56,    57,    58,    59,
   204,   189,   151,   152,    29,    30,    62,    73,   126,   127,
    66,   114,   115,   205,   206,    31,   159,   194,    32,    33,
    34,    71,    35,    85,    86,    87,    88,    89,   208,   171,
   132,   133,    36,    37,   197,   226,   227,    39,    64,   228,
   229,   267,   243,   260,   261,   307,   273,   250,   321,   251,
   252,   279,   328,   301,   325,   341,   326,   383,   323,   357,
   313,   317,   318,   337,   373
};

static const short yypact[] = {   168,
   -24,   -24,-32768,   168,-32768,    11,    11,-32768,-32768,     7,
-32768,-32768,-32768,   143,   -31,     7,   160,     8,    16,    10,
    10,    10,    10,    17,    26,    37,     7,-32768,    -4,    61,
    66,-32768,-32768,-32768,-32768,   353,-32768,-32768,    79,   110,
-32768,   241,-32768,   230,-32768,-32768,    14,-32768,-32768,-32768,
-32768,   248,   -16,-32768,-32768,   131,   137,   141,-32768,   -35,
-32768,    16,    22,   150,   192,   -24,   -24,   -24,   -24,   134,
   -24,   155,   -24,   171,   175,-32768,-32768,-32768,    10,    10,
    10,    10,    17,    26,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,   167,    45,-32768,-32768,-32768,-32768,-32768,-32768,
   191,   186,-32768,-32768,-32768,-32768,   187,-32768,   -37,   -40,
    16,-32768,-32768,   183,   204,   205,   208,   208,   208,   208,
   -24,   200,   208,-32768,-32768,   201,   211,   215,   -24,    23,
   -26,   214,   226,   110,   -24,   -24,   -24,   -24,   -24,   -24,
   -24,   232,-32768,-32768,   229,-32768,   240,    45,   -24,   153,
   236,-32768,     8,-32768,-32768,   192,-32768,   -24,    67,    67,
    67,    67,   260,-32768,    67,-32768,   155,   230,   235,   181,
-32768,   261,   171,-32768,-32768,   208,   208,   208,   208,   208,
   116,-32768,-32768,   186,-32768,   270,-32768,   282,   272,-32768,
   294,-32768,   281,-32768,-32768,    24,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,   -26,   230,   301,   230,-32768,-32768,-32768,
-32768,-32768,    67,    67,    67,    67,    67,-32768,   230,-32768,
   297,-32768,-32768,   -24,   295,   304,-32768,    24,   316,   323,
-32768,-32768,    45,-32768,-32768,-32768,-32768,-32768,   305,   230,
-32768,   308,   256,   110,-32768,-32768,   324,   335,   -26,   345,
   334,   330,-32768,-32768,-32768,-32768,-32768,-32768,-32768,   329,
   339,   355,   337,   357,   131,-32768,   -24,-32768,   342,-32768,
   126,    69,   354,   230,-32768,-32768,   308,   132,   276,   357,
-32768,-32768,    83,   345,   345,   345,-32768,-32768,-32768,-32768,
   358,   360,   336,   360,   380,   344,   368,   383,   360,   377,
-32768,-32768,   409,-32768,-32768,   282,-32768,   371,   366,   364,
-32768,-32768,   345,   388,   345,    30,   374,   284,   377,-32768,
   345,   375,   381,   345,   379,   377,   367,   276,   345,   345,
   345,   384,   373,   378,-32768,-32768,   345,   380,   217,-32768,
   385,   377,   386,   383,-32768,   378,-32768,-32768,   393,-32768,
-32768,-32768,-32768,   345,   360,-32768,   307,   382,-32768,-32768,
   387,-32768,-32768,-32768,-32768,   315,-32768,   391,   345,   377,
-32768,-32768,   357,-32768,   377,   398,   345,   392,   389,-32768,
   394,   338,-32768,   378,   345,-32768,   398,   377,-32768,   346,
   395,-32768,   397,   377,-32768,   345,-32768,   399,   378,-32768,
   369,   377,-32768,   400,-32768,   447,   448,-32768
};

static const short yypgoto[] = {-32768,
   446,-32768,  -127,    -1,    -9,   439,   449,    -8,-32768,-32768,
   273,-32768,   343,-32768,   172,-32768,   -39,   218,-32768,   -70,
   -15,-32768,-32768,   309,-32768,-32768,   310,   390,   291,-32768,
    63,   306,-32768,-32768,  -177,-32768,   -80,   242,-32768,-32768,
-32768,   396,-32768,-32768,-32768,-32768,-32768,-32768,-32768,   292,
-32768,   296,-32768,-32768,   -50,   237,   453,-32768,   359,-32768,
-32768,-32768,-32768,   194,-32768,-32768,  -263,-32768,-32768,-32768,
  -202,  -269,-32768,   140,  -293,-32768,  -274,    85,   129,  -328,
  -285,-32768,   138,-32768,-32768
};


#define	YYLAST		479


static const short yytable[] = {     6,
     7,    60,    93,   130,   302,    45,   175,    41,   315,    63,
   303,    13,   101,   324,     5,   101,     5,   366,    75,   153,
   308,   309,   310,    40,   107,   150,   108,   231,    94,    17,
    18,    19,   348,   107,    61,   108,   101,   160,   161,   162,
   102,   239,   165,   340,   342,   361,   336,    76,   363,   332,
   109,   334,   110,   302,   225,   390,    45,   343,   131,    95,
   346,   -15,     9,    96,    10,   351,   352,   353,   122,   369,
   401,   288,    65,   358,   101,   291,   379,   111,  -107,    70,
  -107,   381,   371,    67,    68,    69,   148,   170,    72,   278,
   368,   376,   335,    74,   393,   213,   214,   215,   216,   217,
   398,    63,   130,   380,   107,   378,   108,   389,   404,   198,
   199,   200,    77,   384,   202,   395,   268,    78,   195,   163,
   196,   391,   284,    45,   304,   305,   403,   169,   285,   172,
    90,   286,   399,   176,   177,   178,   179,   180,   181,   182,
   188,   135,   136,   137,   138,     1,     2,   186,    17,    18,
    19,    20,    21,    22,    23,    24,    25,   131,    43,    44,
    26,    91,   234,   235,   236,   237,   238,   218,   148,   219,
     1,     2,    45,    42,     5,   121,    46,    47,    48,    49,
    50,    51,    52,   124,   125,   129,    43,    44,   282,   104,
   283,   233,   290,    45,   187,   105,    27,    43,    44,   106,
    45,   112,   230,   265,    46,    47,    48,    49,    50,    51,
    52,    45,    43,    44,   113,    46,    47,    48,    49,    50,
    51,    52,   209,   210,   249,   141,    45,   143,   144,   134,
    46,    47,    48,    49,    50,    51,    52,   117,   118,   119,
   120,   264,   123,   142,   128,   149,   155,   271,   292,   293,
   294,   295,   296,   297,   298,   299,    43,    44,   249,   156,
   158,   102,   249,   164,   166,   280,   167,    43,   168,   173,
    45,   360,   183,   306,    46,    47,    48,    49,    50,    51,
    52,    45,   262,   174,   184,    46,    47,    48,    49,    50,
    51,    52,    97,    98,    99,   100,   263,   185,   150,   207,
    46,    47,    48,    49,    50,    51,    52,   292,   293,   294,
   295,   296,   297,   298,   299,   292,   293,   294,   295,   296,
   297,   298,   299,   201,   253,   170,   254,   255,   256,   300,
   221,   257,   258,   259,   101,   222,   224,   339,   292,   293,
   294,   295,   296,   297,   298,   299,   292,   293,   294,   295,
   296,   297,   298,   299,   223,   232,   240,   242,   244,   248,
   370,    79,    80,    81,    82,    83,    84,   246,   375,   292,
   293,   294,   295,   296,   297,   298,   299,   292,   293,   294,
   295,   296,   297,   298,   299,   247,   270,   269,   272,   274,
   275,   388,   276,   281,   277,  -137,    -8,   319,   314,   394,
   292,   293,   294,   295,   296,   297,   298,   299,   292,   293,
   294,   295,   296,   297,   298,   299,   278,   287,   311,   312,
   316,   320,   402,   322,   327,   329,   330,   331,   333,   338,
   344,   349,   345,   347,   367,   354,   355,   382,   356,   362,
   364,   374,   377,   386,   385,   372,   407,   408,   387,     8,
   396,   397,    28,   400,   405,    12,   220,   203,   157,   190,
   266,   192,   191,   211,   245,   241,    38,   350,   212,   154,
   289,   392,   365,   140,     0,   359,     0,     0,   139
};

static const short yycheck[] = {     1,
     2,    17,    42,    74,   279,    41,   134,    16,   294,    19,
   280,     5,    53,   299,    41,    53,    41,   346,    27,    60,
   284,   285,   286,    55,    60,    63,    62,   205,    44,     6,
     7,     8,   326,    60,    27,    62,    53,   118,   119,   120,
    57,   219,   123,   318,   319,   339,   316,    52,   342,   313,
    60,   315,    62,   328,    31,   384,    41,   321,    74,    46,
   324,    55,    52,    50,    54,   329,   330,   331,    70,   355,
   399,   274,    63,   337,    53,   278,   370,    56,    56,    63,
    58,   375,   357,    21,    22,    23,   102,    65,    63,    60,
   354,   366,    63,    57,   388,   176,   177,   178,   179,   180,
   394,   111,   173,   373,    60,   369,    62,   382,   402,   160,
   161,   162,    52,   377,   165,   390,   244,    52,    52,   121,
    54,   385,    54,    41,    42,    43,   401,   129,    60,   131,
    52,    63,   396,   135,   136,   137,   138,   139,   140,   141,
   150,    79,    80,    81,    82,     3,     4,   149,     6,     7,
     8,     9,    10,    11,    12,    13,    14,   173,    27,    28,
    18,    52,   213,   214,   215,   216,   217,    52,   184,    54,
     3,     4,    41,    14,    41,    42,    45,    46,    47,    48,
    49,    50,    51,    29,    30,    15,    27,    28,    63,    59,
    65,   207,    61,    41,    42,    59,    54,    27,    28,    59,
    41,    52,   204,   243,    45,    46,    47,    48,    49,    50,
    51,    41,    27,    28,    23,    45,    46,    47,    48,    49,
    50,    51,    42,    43,   240,    59,    41,    42,    43,    55,
    45,    46,    47,    48,    49,    50,    51,    66,    67,    68,
    69,   243,    71,    53,    73,    59,    64,   249,    32,    33,
    34,    35,    36,    37,    38,    39,    27,    28,   274,    56,
    53,    57,   278,    64,    64,   267,    56,    27,    54,    56,
    41,    55,    41,   283,    45,    46,    47,    48,    49,    50,
    51,    41,    27,    58,    56,    45,    46,    47,    48,    49,
    50,    51,    45,    46,    47,    48,    41,    58,    63,    65,
    45,    46,    47,    48,    49,    50,    51,    32,    33,    34,
    35,    36,    37,    38,    39,    32,    33,    34,    35,    36,
    37,    38,    39,    64,    17,    65,    19,    20,    21,    54,
    61,    24,    25,    26,    53,    64,    56,    54,    32,    33,
    34,    35,    36,    37,    38,    39,    32,    33,    34,    35,
    36,    37,    38,    39,    61,    55,    60,    63,    55,    55,
    54,     9,    10,    11,    12,    13,    14,    52,    54,    32,
    33,    34,    35,    36,    37,    38,    39,    32,    33,    34,
    35,    36,    37,    38,    39,    63,    52,    64,    44,    56,
    61,    54,    64,    52,    56,    41,    60,    54,    63,    54,
    32,    33,    34,    35,    36,    37,    38,    39,    32,    33,
    34,    35,    36,    37,    38,    39,    60,    64,    61,    60,
    41,    54,    54,    41,    16,    55,    61,    64,    41,    56,
    56,    65,    52,    55,    42,    52,    64,    40,    61,    55,
    55,    55,    52,    55,    53,    64,     0,     0,    55,     4,
    56,    55,    14,    55,    55,     7,   184,   167,   116,   151,
   243,   156,   153,   172,   228,   224,    14,   328,   173,   111,
   277,   387,   344,    84,    -1,   338,    -1,    -1,    83
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
#line 120 "xi-grammar.y"
{ yyval.modlist = yyvsp[0].modlist; modlist = yyvsp[0].modlist; ;
    break;}
case 2:
#line 124 "xi-grammar.y"
{ yyval.modlist = 0; ;
    break;}
case 3:
#line 126 "xi-grammar.y"
{ yyval.modlist = new ModuleList(lineno, yyvsp[-1].module, yyvsp[0].modlist); ;
    break;}
case 4:
#line 130 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 5:
#line 132 "xi-grammar.y"
{ yyval.intval = 1; ;
    break;}
case 6:
#line 136 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 7:
#line 138 "xi-grammar.y"
{ yyval.intval = 1; ;
    break;}
case 8:
#line 142 "xi-grammar.y"
{ yyval.strval = yyvsp[0].strval; ;
    break;}
case 9:
#line 146 "xi-grammar.y"
{ yyval.strval = yyvsp[0].strval; ;
    break;}
case 10:
#line 148 "xi-grammar.y"
{
		  char *tmp = new char[strlen(yyvsp[-3].strval)+strlen(yyvsp[0].strval)+3];
		  sprintf(tmp,"%s::%s", yyvsp[-3].strval, yyvsp[0].strval);
		  yyval.strval = tmp;
		;
    break;}
case 11:
#line 156 "xi-grammar.y"
{ yyval.module = new Module(lineno, yyvsp[-1].strval, yyvsp[0].conslist); ;
    break;}
case 12:
#line 158 "xi-grammar.y"
{ yyval.module = new Module(lineno, yyvsp[-1].strval, yyvsp[0].conslist); yyval.module->setMain(); ;
    break;}
case 13:
#line 162 "xi-grammar.y"
{ yyval.conslist = 0; ;
    break;}
case 14:
#line 164 "xi-grammar.y"
{ yyval.conslist = yyvsp[-2].conslist; ;
    break;}
case 15:
#line 168 "xi-grammar.y"
{ yyval.conslist = 0; ;
    break;}
case 16:
#line 170 "xi-grammar.y"
{ yyval.conslist = new ConstructList(lineno, yyvsp[-1].construct, yyvsp[0].conslist); ;
    break;}
case 17:
#line 174 "xi-grammar.y"
{ if(yyvsp[-2].conslist) yyvsp[-2].conslist->setExtern(yyvsp[-4].intval); yyval.construct = yyvsp[-2].conslist; ;
    break;}
case 18:
#line 176 "xi-grammar.y"
{ yyvsp[0].module->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].module; ;
    break;}
case 19:
#line 178 "xi-grammar.y"
{ yyvsp[0].member->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].member; ;
    break;}
case 20:
#line 180 "xi-grammar.y"
{ yyvsp[-1].message->setExtern(yyvsp[-2].intval); yyval.construct = yyvsp[-1].message; ;
    break;}
case 21:
#line 182 "xi-grammar.y"
{ yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; ;
    break;}
case 22:
#line 184 "xi-grammar.y"
{ yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; ;
    break;}
case 23:
#line 186 "xi-grammar.y"
{ yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; ;
    break;}
case 24:
#line 188 "xi-grammar.y"
{ yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; ;
    break;}
case 25:
#line 190 "xi-grammar.y"
{ yyvsp[0].templat->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].templat; ;
    break;}
case 26:
#line 194 "xi-grammar.y"
{ yyval.tparam = new TParamType(yyvsp[0].type); ;
    break;}
case 27:
#line 196 "xi-grammar.y"
{ yyval.tparam = new TParamVal(yyvsp[0].strval); ;
    break;}
case 28:
#line 198 "xi-grammar.y"
{ yyval.tparam = new TParamVal(yyvsp[0].strval); ;
    break;}
case 29:
#line 202 "xi-grammar.y"
{ yyval.tparlist = new TParamList(yyvsp[0].tparam); ;
    break;}
case 30:
#line 204 "xi-grammar.y"
{ yyval.tparlist = new TParamList(yyvsp[-2].tparam, yyvsp[0].tparlist); ;
    break;}
case 31:
#line 208 "xi-grammar.y"
{ yyval.tparlist = 0; ;
    break;}
case 32:
#line 210 "xi-grammar.y"
{ yyval.tparlist = yyvsp[0].tparlist; ;
    break;}
case 33:
#line 214 "xi-grammar.y"
{ yyval.tparlist = 0; ;
    break;}
case 34:
#line 216 "xi-grammar.y"
{ yyval.tparlist = yyvsp[-1].tparlist; ;
    break;}
case 35:
#line 220 "xi-grammar.y"
{ yyval.type = new BuiltinType("int"); ;
    break;}
case 36:
#line 222 "xi-grammar.y"
{ yyval.type = new BuiltinType("long"); ;
    break;}
case 37:
#line 224 "xi-grammar.y"
{ yyval.type = new BuiltinType("short"); ;
    break;}
case 38:
#line 226 "xi-grammar.y"
{ yyval.type = new BuiltinType("char"); ;
    break;}
case 39:
#line 228 "xi-grammar.y"
{ yyval.type = new BuiltinType("unsigned int"); ;
    break;}
case 40:
#line 230 "xi-grammar.y"
{ yyval.type = new BuiltinType("unsigned long"); ;
    break;}
case 41:
#line 232 "xi-grammar.y"
{ yyval.type = new BuiltinType("unsigned short"); ;
    break;}
case 42:
#line 234 "xi-grammar.y"
{ yyval.type = new BuiltinType("unsigned char"); ;
    break;}
case 43:
#line 236 "xi-grammar.y"
{ yyval.type = new BuiltinType("long long"); ;
    break;}
case 44:
#line 238 "xi-grammar.y"
{ yyval.type = new BuiltinType("float"); ;
    break;}
case 45:
#line 240 "xi-grammar.y"
{ yyval.type = new BuiltinType("double"); ;
    break;}
case 46:
#line 242 "xi-grammar.y"
{ yyval.type = new BuiltinType("long double"); ;
    break;}
case 47:
#line 244 "xi-grammar.y"
{ yyval.type = new BuiltinType("void"); ;
    break;}
case 48:
#line 247 "xi-grammar.y"
{ yyval.ntype = new NamedType(yyvsp[-1].strval,yyvsp[0].tparlist); ;
    break;}
case 49:
#line 248 "xi-grammar.y"
{ yyval.ntype = new NamedType(yyvsp[-1].strval,yyvsp[0].tparlist); ;
    break;}
case 50:
#line 251 "xi-grammar.y"
{ yyval.type = yyvsp[0].type; ;
    break;}
case 51:
#line 253 "xi-grammar.y"
{ yyval.type = yyvsp[0].ntype; ;
    break;}
case 52:
#line 257 "xi-grammar.y"
{ yyval.ptype = new PtrType(yyvsp[-1].type); ;
    break;}
case 53:
#line 261 "xi-grammar.y"
{ yyvsp[-1].ptype->indirect(); yyval.ptype = yyvsp[-1].ptype; ;
    break;}
case 54:
#line 263 "xi-grammar.y"
{ yyvsp[-1].ptype->indirect(); yyval.ptype = yyvsp[-1].ptype; ;
    break;}
case 55:
#line 267 "xi-grammar.y"
{ yyval.ftype = new FuncType(yyvsp[-7].type, yyvsp[-4].strval, yyvsp[-1].plist); ;
    break;}
case 56:
#line 271 "xi-grammar.y"
{ yyval.type = yyvsp[0].type; ;
    break;}
case 57:
#line 273 "xi-grammar.y"
{ yyval.type = yyvsp[0].ptype; ;
    break;}
case 58:
#line 275 "xi-grammar.y"
{ yyval.type = yyvsp[0].ptype; ;
    break;}
case 59:
#line 277 "xi-grammar.y"
{ yyval.type = yyvsp[0].ftype; ;
    break;}
case 60:
#line 279 "xi-grammar.y"
{ yyval.type = new ReferenceType(yyvsp[-1].type); ;
    break;}
case 61:
#line 282 "xi-grammar.y"
{ yyval.type = yyvsp[0].type; ;
    break;}
case 62:
#line 286 "xi-grammar.y"
{ yyval.val = new Value(yyvsp[0].strval); ;
    break;}
case 63:
#line 288 "xi-grammar.y"
{ yyval.val = new Value(yyvsp[0].strval); ;
    break;}
case 64:
#line 292 "xi-grammar.y"
{ yyval.val = yyvsp[-1].val; ;
    break;}
case 65:
#line 296 "xi-grammar.y"
{ yyval.vallist = 0; ;
    break;}
case 66:
#line 298 "xi-grammar.y"
{ yyval.vallist = new ValueList(yyvsp[-1].val, yyvsp[0].vallist); ;
    break;}
case 67:
#line 302 "xi-grammar.y"
{ yyval.readonly = new Readonly(lineno, yyvsp[-2].type, yyvsp[-1].strval, yyvsp[0].vallist); ;
    break;}
case 68:
#line 306 "xi-grammar.y"
{ yyval.readonly = new Readonly(lineno, yyvsp[-2].type, yyvsp[0].strval, 0, 1); ;
    break;}
case 69:
#line 310 "xi-grammar.y"
{ yyval.intval = 0;;
    break;}
case 70:
#line 312 "xi-grammar.y"
{ yyval.intval = 0;;
    break;}
case 71:
#line 316 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 72:
#line 318 "xi-grammar.y"
{ 
		  /*
		  printf("Warning: Message attributes are being phased out.\n");
		  printf("Warning: Please remove them from interface files.\n");
		  */
		  yyval.intval = yyvsp[-1].intval; 
		;
    break;}
case 73:
#line 328 "xi-grammar.y"
{ yyval.intval = yyvsp[0].intval; ;
    break;}
case 74:
#line 330 "xi-grammar.y"
{ yyval.intval = yyvsp[-2].intval | yyvsp[0].intval; ;
    break;}
case 75:
#line 334 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 76:
#line 336 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 77:
#line 340 "xi-grammar.y"
{ yyval.cattr = 0; ;
    break;}
case 78:
#line 342 "xi-grammar.y"
{ yyval.cattr = yyvsp[-1].cattr; ;
    break;}
case 79:
#line 346 "xi-grammar.y"
{ yyval.cattr = yyvsp[0].cattr; ;
    break;}
case 80:
#line 348 "xi-grammar.y"
{ yyval.cattr = yyvsp[-2].cattr | yyvsp[0].cattr; ;
    break;}
case 81:
#line 352 "xi-grammar.y"
{ yyval.cattr = Chare::CMIGRATABLE; ;
    break;}
case 82:
#line 356 "xi-grammar.y"
{ yyval.mv = new MsgVar(yyvsp[-4].type, yyvsp[-3].strval); ;
    break;}
case 83:
#line 360 "xi-grammar.y"
{ yyval.mvlist = new MsgVarList(yyvsp[0].mv); ;
    break;}
case 84:
#line 362 "xi-grammar.y"
{ yyval.mvlist = new MsgVarList(yyvsp[-1].mv, yyvsp[0].mvlist); ;
    break;}
case 85:
#line 366 "xi-grammar.y"
{ yyval.message = new Message(lineno, yyvsp[0].ntype); ;
    break;}
case 86:
#line 368 "xi-grammar.y"
{ yyval.message = new Message(lineno, yyvsp[-3].ntype, yyvsp[-1].mvlist); ;
    break;}
case 87:
#line 372 "xi-grammar.y"
{ yyval.typelist = 0; ;
    break;}
case 88:
#line 374 "xi-grammar.y"
{ yyval.typelist = yyvsp[0].typelist; ;
    break;}
case 89:
#line 378 "xi-grammar.y"
{ yyval.typelist = new TypeList(yyvsp[0].ntype); ;
    break;}
case 90:
#line 380 "xi-grammar.y"
{ yyval.typelist = new TypeList(yyvsp[-2].ntype, yyvsp[0].typelist); ;
    break;}
case 91:
#line 384 "xi-grammar.y"
{ yyval.chare = new Chare(lineno, yyvsp[-3].cattr, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 92:
#line 386 "xi-grammar.y"
{ yyval.chare = new MainChare(lineno, yyvsp[-3].cattr, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 93:
#line 390 "xi-grammar.y"
{ yyval.chare = new Group(lineno, yyvsp[-3].cattr, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 94:
#line 394 "xi-grammar.y"
{ yyval.chare = new NodeGroup(lineno, yyvsp[-3].cattr, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 95:
#line 398 "xi-grammar.y"
{/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",yyvsp[-2].strval);
			yyval.ntype = new NamedType(buf); 
		;
    break;}
case 96:
#line 404 "xi-grammar.y"
{ yyval.ntype = new NamedType(yyvsp[-1].strval); ;
    break;}
case 97:
#line 408 "xi-grammar.y"
{ yyval.chare = new Array(lineno, 0, yyvsp[-3].ntype, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 98:
#line 412 "xi-grammar.y"
{ yyval.chare = new Chare(lineno, yyvsp[-3].cattr, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist);;
    break;}
case 99:
#line 414 "xi-grammar.y"
{ yyval.chare = new MainChare(lineno, yyvsp[-3].cattr, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 100:
#line 418 "xi-grammar.y"
{ yyval.chare = new Group(lineno, yyvsp[-3].cattr, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 101:
#line 422 "xi-grammar.y"
{ yyval.chare = new NodeGroup( lineno, yyvsp[-3].cattr, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 102:
#line 426 "xi-grammar.y"
{ yyval.chare = new Array( lineno, 0, yyvsp[-3].ntype, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 103:
#line 430 "xi-grammar.y"
{ yyval.message = new Message(lineno, new NamedType(yyvsp[-1].strval)); ;
    break;}
case 104:
#line 432 "xi-grammar.y"
{ yyval.message = new Message(lineno, new NamedType(yyvsp[-4].strval), yyvsp[-2].mvlist); ;
    break;}
case 105:
#line 436 "xi-grammar.y"
{ yyval.type = 0; ;
    break;}
case 106:
#line 438 "xi-grammar.y"
{ yyval.type = yyvsp[0].type; ;
    break;}
case 107:
#line 442 "xi-grammar.y"
{ yyval.strval = 0; ;
    break;}
case 108:
#line 444 "xi-grammar.y"
{ yyval.strval = yyvsp[0].strval; ;
    break;}
case 109:
#line 446 "xi-grammar.y"
{ yyval.strval = yyvsp[0].strval; ;
    break;}
case 110:
#line 450 "xi-grammar.y"
{ yyval.tvar = new TType(new NamedType(yyvsp[-1].strval), yyvsp[0].type); ;
    break;}
case 111:
#line 452 "xi-grammar.y"
{ yyval.tvar = new TFunc(yyvsp[-1].ftype, yyvsp[0].strval); ;
    break;}
case 112:
#line 454 "xi-grammar.y"
{ yyval.tvar = new TName(yyvsp[-2].type, yyvsp[-1].strval, yyvsp[0].strval); ;
    break;}
case 113:
#line 458 "xi-grammar.y"
{ yyval.tvarlist = new TVarList(yyvsp[0].tvar); ;
    break;}
case 114:
#line 460 "xi-grammar.y"
{ yyval.tvarlist = new TVarList(yyvsp[-2].tvar, yyvsp[0].tvarlist); ;
    break;}
case 115:
#line 464 "xi-grammar.y"
{ yyval.tvarlist = yyvsp[-1].tvarlist; ;
    break;}
case 116:
#line 468 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); ;
    break;}
case 117:
#line 470 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); ;
    break;}
case 118:
#line 472 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); ;
    break;}
case 119:
#line 474 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); ;
    break;}
case 120:
#line 476 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].message); yyvsp[0].message->setTemplate(yyval.templat); ;
    break;}
case 121:
#line 480 "xi-grammar.y"
{ yyval.mbrlist = 0; ;
    break;}
case 122:
#line 482 "xi-grammar.y"
{ yyval.mbrlist = yyvsp[-2].mbrlist; ;
    break;}
case 123:
#line 486 "xi-grammar.y"
{ yyval.mbrlist = 0; ;
    break;}
case 124:
#line 488 "xi-grammar.y"
{ yyval.mbrlist = new MemberList(yyvsp[-1].member, yyvsp[0].mbrlist); ;
    break;}
case 125:
#line 492 "xi-grammar.y"
{ yyval.member = yyvsp[-1].readonly; ;
    break;}
case 126:
#line 494 "xi-grammar.y"
{ yyval.member = yyvsp[-1].readonly; ;
    break;}
case 127:
#line 496 "xi-grammar.y"
{ yyval.member = yyvsp[-1].member; ;
    break;}
case 128:
#line 498 "xi-grammar.y"
{ yyval.member = yyvsp[-1].pupable; ;
    break;}
case 129:
#line 502 "xi-grammar.y"
{ yyval.member = new InitCall(lineno, yyvsp[0].strval); ;
    break;}
case 130:
#line 504 "xi-grammar.y"
{ yyval.member = new InitCall(lineno, yyvsp[-3].strval); ;
    break;}
case 131:
#line 508 "xi-grammar.y"
{ yyval.pupable = new PUPableClass(lineno,yyvsp[0].strval,0); ;
    break;}
case 132:
#line 510 "xi-grammar.y"
{ yyval.pupable = new PUPableClass(lineno,yyvsp[-2].strval,yyvsp[0].pupable); ;
    break;}
case 133:
#line 514 "xi-grammar.y"
{ yyval.member = yyvsp[-1].entry; ;
    break;}
case 134:
#line 516 "xi-grammar.y"
{ yyval.member = yyvsp[0].member; ;
    break;}
case 135:
#line 520 "xi-grammar.y"
{ 
		  if (yyvsp[0].sc != 0) { 
		    yyvsp[0].sc->con1 = new SdagConstruct(SIDENT, yyvsp[-3].strval);
  		    if (yyvsp[-2].plist != 0)
                      yyvsp[0].sc->param = new ParamList(yyvsp[-2].plist);
 		    else 
 	 	      yyvsp[0].sc->param = new ParamList(new Parameter(0, new BuiltinType("void")));
                  }
		  yyval.entry = new Entry(lineno, yyvsp[-5].intval, yyvsp[-4].type, yyvsp[-3].strval, yyvsp[-2].plist, yyvsp[-1].val, yyvsp[0].sc, 0); 
		;
    break;}
case 136:
#line 531 "xi-grammar.y"
{ 
		  if (yyvsp[0].sc != 0) {
		    yyvsp[0].sc->con1 = new SdagConstruct(SIDENT, yyvsp[-2].strval);
		    if (yyvsp[-1].plist != 0)
                      yyvsp[0].sc->param = new ParamList(yyvsp[-1].plist);
		    else
                      yyvsp[0].sc->param = new ParamList(new Parameter(0, new BuiltinType("void")));
                  }
		  yyval.entry = new Entry(lineno, yyvsp[-3].intval,     0, yyvsp[-2].strval, yyvsp[-1].plist,  0, yyvsp[0].sc, 0); 
		;
    break;}
case 137:
#line 544 "xi-grammar.y"
{ yyval.type = new BuiltinType("void"); ;
    break;}
case 138:
#line 546 "xi-grammar.y"
{ yyval.type = yyvsp[0].ptype; ;
    break;}
case 139:
#line 550 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 140:
#line 552 "xi-grammar.y"
{ yyval.intval = yyvsp[-1].intval; ;
    break;}
case 141:
#line 556 "xi-grammar.y"
{ yyval.intval = yyvsp[0].intval; ;
    break;}
case 142:
#line 558 "xi-grammar.y"
{ yyval.intval = yyvsp[-2].intval | yyvsp[0].intval; ;
    break;}
case 143:
#line 562 "xi-grammar.y"
{ yyval.intval = STHREADED; ;
    break;}
case 144:
#line 564 "xi-grammar.y"
{ yyval.intval = SSYNC; ;
    break;}
case 145:
#line 566 "xi-grammar.y"
{ yyval.intval = SLOCKED; ;
    break;}
case 146:
#line 568 "xi-grammar.y"
{ yyval.intval = SCREATEHERE; ;
    break;}
case 147:
#line 570 "xi-grammar.y"
{ yyval.intval = SCREATEHOME; ;
    break;}
case 148:
#line 572 "xi-grammar.y"
{ yyval.intval = SNOKEEP; ;
    break;}
case 149:
#line 574 "xi-grammar.y"
{ yyval.intval = SIMMEDIATE; ;
    break;}
case 150:
#line 578 "xi-grammar.y"
{ yyval.val = new Value(yyvsp[0].strval); ;
    break;}
case 151:
#line 580 "xi-grammar.y"
{ yyval.val = new Value(yyvsp[0].strval); ;
    break;}
case 152:
#line 582 "xi-grammar.y"
{ yyval.val = new Value(yyvsp[0].strval); ;
    break;}
case 153:
#line 586 "xi-grammar.y"
{ yyval.strval = ""; ;
    break;}
case 154:
#line 588 "xi-grammar.y"
{ yyval.strval = yyvsp[0].strval; ;
    break;}
case 155:
#line 590 "xi-grammar.y"
{  /*Returned only when in_bracket*/
			char *tmp = new char[strlen(yyvsp[-4].strval)+strlen(yyvsp[-2].strval)+strlen(yyvsp[0].strval)+3];
			sprintf(tmp,"%s[%s]%s", yyvsp[-4].strval, yyvsp[-2].strval, yyvsp[0].strval);
			yyval.strval = tmp;
		;
    break;}
case 156:
#line 596 "xi-grammar.y"
{ /*Returned only when in_braces*/
			char *tmp = new char[strlen(yyvsp[-4].strval)+strlen(yyvsp[-2].strval)+strlen(yyvsp[0].strval)+3];
			sprintf(tmp,"%s{%s}%s", yyvsp[-4].strval, yyvsp[-2].strval, yyvsp[0].strval);
			yyval.strval = tmp;
		;
    break;}
case 157:
#line 602 "xi-grammar.y"
{ /*Returned only when in_braces*/
			char *tmp = new char[strlen(yyvsp[-4].strval)+strlen(yyvsp[-2].strval)+strlen(yyvsp[0].strval)+3];
			sprintf(tmp,"%s(%s)%s", yyvsp[-4].strval, yyvsp[-2].strval, yyvsp[0].strval);
			yyval.strval = tmp;
		;
    break;}
case 158:
#line 610 "xi-grammar.y"
{  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			yyval.pname = new Parameter(lineno, yyvsp[-2].type,yyvsp[-1].strval);
		;
    break;}
case 159:
#line 617 "xi-grammar.y"
{ 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			yyval.intval = 0;
		;
    break;}
case 160:
#line 625 "xi-grammar.y"
{ yyval.pname = new Parameter(lineno, yyvsp[0].type);;
    break;}
case 161:
#line 627 "xi-grammar.y"
{ yyval.pname = new Parameter(lineno, yyvsp[-1].type,yyvsp[0].strval);;
    break;}
case 162:
#line 629 "xi-grammar.y"
{ yyval.pname = new Parameter(lineno, yyvsp[-3].type,yyvsp[-2].strval,0,yyvsp[0].val);;
    break;}
case 163:
#line 631 "xi-grammar.y"
{ /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			yyval.pname = new Parameter(lineno, yyvsp[-2].pname->getType(), yyvsp[-2].pname->getName() ,yyvsp[-1].strval);
		;
    break;}
case 164:
#line 638 "xi-grammar.y"
{ yyval.plist = new ParamList(yyvsp[0].pname); ;
    break;}
case 165:
#line 640 "xi-grammar.y"
{ yyval.plist = new ParamList(yyvsp[-2].pname,yyvsp[0].plist); ;
    break;}
case 166:
#line 644 "xi-grammar.y"
{ yyval.plist = yyvsp[-1].plist; ;
    break;}
case 167:
#line 646 "xi-grammar.y"
{ yyval.plist = 0; ;
    break;}
case 168:
#line 650 "xi-grammar.y"
{ yyval.val = 0; ;
    break;}
case 169:
#line 652 "xi-grammar.y"
{ yyval.val = new Value(yyvsp[0].strval); ;
    break;}
case 170:
#line 656 "xi-grammar.y"
{ yyval.sc = 0; ;
    break;}
case 171:
#line 658 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SSDAGENTRY, yyvsp[0].sc); ;
    break;}
case 172:
#line 660 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SSDAGENTRY, yyvsp[-1].sc); ;
    break;}
case 173:
#line 664 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SSLIST, yyvsp[0].sc); ;
    break;}
case 174:
#line 666 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SSLIST, yyvsp[-1].sc, yyvsp[0].sc);  ;
    break;}
case 175:
#line 670 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SOLIST, yyvsp[0].sc); ;
    break;}
case 176:
#line 672 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SOLIST, yyvsp[-1].sc, yyvsp[0].sc); ;
    break;}
case 177:
#line 676 "xi-grammar.y"
{
		     in_braces =0;
		     yyval.sc = new SdagConstruct(SATOMIC, yyvsp[-1].strval);  
		  ;
    break;}
case 178:
#line 681 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SWHEN, 0,0,0,0,0,yyvsp[-2].entrylist); ;
    break;}
case 179:
#line 683 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SWHEN, 0, 0,0,0, yyvsp[0].sc, yyvsp[-1].entrylist); ;
    break;}
case 180:
#line 685 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SWHEN, 0, 0,0,0, yyvsp[-1].sc, yyvsp[-3].entrylist); ;
    break;}
case 181:
#line 687 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SOVERLAP, 0,0,0,0,yyvsp[-1].sc, 0); ;
    break;}
case 182:
#line 689 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SFOR, new SdagConstruct(SINT_EXPR, yyvsp[-8].strval), new SdagConstruct(SINT_EXPR, yyvsp[-6].strval),
		             new SdagConstruct(SINT_EXPR, yyvsp[-4].strval), 0, yyvsp[-1].sc, 0); ;
    break;}
case 183:
#line 692 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SFOR, new SdagConstruct(SINT_EXPR, yyvsp[-6].strval), new SdagConstruct(SINT_EXPR, yyvsp[-4].strval), 
		         new SdagConstruct(SINT_EXPR, yyvsp[-2].strval), 0, yyvsp[0].sc, 0); ;
    break;}
case 184:
#line 695 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SFORALL, new SdagConstruct(SIDENT, yyvsp[-9].strval), new SdagConstruct(SINT_EXPR, yyvsp[-6].strval), 
		             new SdagConstruct(SINT_EXPR, yyvsp[-4].strval), new SdagConstruct(SINT_EXPR, yyvsp[-2].strval), yyvsp[0].sc, 0); ;
    break;}
case 185:
#line 698 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SFORALL, new SdagConstruct(SIDENT, yyvsp[-11].strval), new SdagConstruct(SINT_EXPR, yyvsp[-8].strval), 
		                 new SdagConstruct(SINT_EXPR, yyvsp[-6].strval), new SdagConstruct(SINT_EXPR, yyvsp[-4].strval), yyvsp[-1].sc, 0); ;
    break;}
case 186:
#line 701 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SIF, new SdagConstruct(SINT_EXPR, yyvsp[-3].strval), yyvsp[0].sc,0,0,yyvsp[-1].sc,0); ;
    break;}
case 187:
#line 703 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SIF, new SdagConstruct(SINT_EXPR, yyvsp[-5].strval), yyvsp[0].sc,0,0,yyvsp[-2].sc,0); ;
    break;}
case 188:
#line 705 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SIF, new SdagConstruct(SINT_EXPR, yyvsp[-2].strval), 0,0,0,yyvsp[0].sc,0); ;
    break;}
case 189:
#line 707 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SWHILE, new SdagConstruct(SINT_EXPR, yyvsp[-4].strval), 0,0,0,yyvsp[-1].sc,0); ;
    break;}
case 190:
#line 709 "xi-grammar.y"
{ yyval.sc = yyvsp[-1].sc; ;
    break;}
case 191:
#line 713 "xi-grammar.y"
{ yyval.sc = 0; ;
    break;}
case 192:
#line 715 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SELSE, 0,0,0,0, yyvsp[0].sc,0); ;
    break;}
case 193:
#line 717 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SELSE, 0,0,0,0, yyvsp[-1].sc,0); ;
    break;}
case 194:
#line 720 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SFORWARD, new SdagConstruct(SIDENT, yyvsp[0].strval)); ;
    break;}
case 195:
#line 722 "xi-grammar.y"
{ yyval.sc = new SdagConstruct(SFORWARD, new SdagConstruct(SIDENT, yyvsp[-2].strval), yyvsp[0].sc);  ;
    break;}
case 196:
#line 726 "xi-grammar.y"
{ in_int_expr = 0; yyval.intval = 0; ;
    break;}
case 197:
#line 730 "xi-grammar.y"
{ in_int_expr = 1; yyval.intval = 0; ;
    break;}
case 198:
#line 734 "xi-grammar.y"
{ 
		  if (yyvsp[0].plist != 0)
		     yyval.entry = new Entry(lineno, 0, 0, yyvsp[-1].strval, yyvsp[0].plist, 0, 0, 0); 
		  else
		     yyval.entry = new Entry(lineno, 0, 0, yyvsp[-1].strval, 
				new ParamList(new Parameter(0, new BuiltinType("void"))), 0, 0, 0); 
		;
    break;}
case 199:
#line 742 "xi-grammar.y"
{ if (yyvsp[0].plist != 0)
		    yyval.entry = new Entry(lineno, 0, 0, yyvsp[-4].strval, yyvsp[0].plist, 0, 0, yyvsp[-2].strval); 
		  else
		    yyval.entry = new Entry(lineno, 0, 0, yyvsp[-4].strval, new ParamList(new Parameter(0, new BuiltinType("void"))), 0, 0, yyvsp[-2].strval); 
		;
    break;}
case 200:
#line 750 "xi-grammar.y"
{ yyval.entrylist = new EntryList(yyvsp[0].entry); ;
    break;}
case 201:
#line 752 "xi-grammar.y"
{ yyval.entrylist = new EntryList(yyvsp[-2].entry,yyvsp[0].entrylist); ;
    break;}
case 202:
#line 756 "xi-grammar.y"
{ in_bracket=1; ;
    break;}
case 203:
#line 759 "xi-grammar.y"
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
#line 763 "xi-grammar.y"

void yyerror(const char *mesg)
{
  cout << cur_file<<":"<<lineno<<": Charmxi syntax error> " << mesg << endl;
  // return 0;
}
