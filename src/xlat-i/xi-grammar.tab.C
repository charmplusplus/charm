
/*  A Bison parser, made from xi-grammar.y
    by GNU Bison version 1.28  */

#define YYBISON 1  /* Identify Bison output.  */

#define	MODULE	257
#define	MAINMODULE	258
#define	EXTERN	259
#define	READONLY	260
#define	INITCALL	261
#define	CHARE	262
#define	MAINCHARE	263
#define	GROUP	264
#define	NODEGROUP	265
#define	ARRAY	266
#define	MESSAGE	267
#define	CLASS	268
#define	STACKSIZE	269
#define	THREADED	270
#define	TEMPLATE	271
#define	SYNC	272
#define	EXCLUSIVE	273
#define	VIRTUAL	274
#define	MIGRATABLE	275
#define	CREATEHERE	276
#define	CREATEHOME	277
#define	VOID	278
#define	CONST	279
#define	PACKED	280
#define	VARSIZE	281
#define	ENTRY	282
#define	IDENT	283
#define	NUMBER	284
#define	LITERAL	285
#define	CPROGRAM	286
#define	INT	287
#define	LONG	288
#define	SHORT	289
#define	CHAR	290
#define	FLOAT	291
#define	DOUBLE	292
#define	UNSIGNED	293

#line 2 "xi-grammar.y"

#include <iostream.h>
#include "xi-symbol.h"

extern int yylex (void) ;
void yyerror(const char *);
extern unsigned int lineno;
extern int in_bracket,in_braces;
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



#define	YYFINAL		303
#define	YYFLAG		-32768
#define	YYNTBASE	54

#define YYTRANSLATE(x) ((unsigned)(x) <= 293 ? yytranslate[x] : 128)

static const char yytranslate[] = {     0,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,    50,     2,    48,
    49,    47,     2,    44,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,    41,    40,    45,
    53,    46,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
    51,     2,    52,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,    42,     2,    43,     2,     2,     2,     2,     2,
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
    37,    38,    39
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
   191,   195,   202,   203,   207,   209,   213,   215,   217,   218,
   222,   224,   228,   230,   236,   238,   241,   245,   252,   253,
   256,   258,   262,   268,   274,   280,   286,   291,   295,   301,
   307,   313,   319,   325,   331,   336,   337,   340,   341,   344,
   347,   351,   354,   358,   360,   364,   369,   372,   375,   378,
   381,   384,   386,   391,   392,   395,   398,   401,   404,   407,
   409,   417,   423,   425,   427,   428,   432,   434,   438,   440,
   442,   444,   446,   448,   450,   452,   454,   455,   457,   463,
   469,   473,   475,   477,   480,   485,   489,   491,   495,   499,
   502,   503,   507,   508
};

static const short yyrhs[] = {    55,
     0,     0,    60,    55,     0,     0,     5,     0,     0,    40,
     0,    29,     0,    29,     0,    59,    41,    41,    29,     0,
     3,    58,    61,     0,     4,    58,    61,     0,    40,     0,
    42,    62,    43,    57,     0,     0,    63,    62,     0,    56,
    42,    62,    43,    57,     0,    56,    60,     0,    56,   112,
     0,    56,    91,    40,     0,    56,    94,     0,    56,    95,
     0,    56,    96,     0,    56,    98,     0,    56,   109,     0,
    75,     0,    30,     0,    31,     0,    64,     0,    64,    44,
    65,     0,     0,    65,     0,     0,    45,    66,    46,     0,
    33,     0,    34,     0,    35,     0,    36,     0,    39,    33,
     0,    39,    34,     0,    39,    35,     0,    39,    36,     0,
    34,    34,     0,    37,     0,    38,     0,    34,    38,     0,
    24,     0,    58,    67,     0,    59,    67,     0,    68,     0,
    70,     0,    71,    47,     0,    72,    47,     0,    73,    47,
     0,    75,    48,    47,    58,    49,    48,   124,    49,     0,
    71,     0,    72,     0,    73,     0,    74,     0,    75,    50,
     0,    25,    75,     0,    30,     0,    59,     0,    51,    76,
    52,     0,     0,    77,    78,     0,     6,    75,    59,    78,
     0,     6,    13,    71,    47,    58,     0,     0,    24,     0,
     7,    81,    59,     0,     7,    81,    59,    48,    81,    49,
     0,     0,    51,    84,    52,     0,    85,     0,    85,    44,
    84,     0,    26,     0,    27,     0,     0,    51,    87,    52,
     0,    88,     0,    88,    44,    87,     0,    21,     0,    75,
    58,    51,    52,    40,     0,    89,     0,    89,    90,     0,
    13,    83,    69,     0,    13,    83,    69,    42,    90,    43,
     0,     0,    41,    93,     0,    69,     0,    69,    44,    93,
     0,     8,    86,    69,    92,   110,     0,     9,    86,    69,
    92,   110,     0,    10,    86,    69,    92,   110,     0,    11,
    86,    69,    92,   110,     0,    51,    30,    58,    52,     0,
    51,    58,    52,     0,    12,    97,    69,    92,   110,     0,
     8,    86,    58,    92,   110,     0,     9,    86,    58,    92,
   110,     0,    10,    86,    58,    92,   110,     0,    11,    86,
    58,    92,   110,     0,    12,    97,    58,    92,   110,     0,
    13,    83,    58,    40,     0,     0,    53,    75,     0,     0,
    53,    30,     0,    53,    31,     0,    14,    58,   104,     0,
    74,   105,     0,    75,    58,   105,     0,   106,     0,   106,
    44,   107,     0,    17,    45,   107,    46,     0,   108,    99,
     0,   108,   100,     0,   108,   101,     0,   108,   102,     0,
   108,   103,     0,    40,     0,    42,   111,    43,    57,     0,
     0,   113,   111,     0,    79,    40,     0,    80,    40,     0,
    82,    40,     0,   114,    40,     0,   112,     0,    28,   116,
   115,    58,   125,   126,   127,     0,    28,   116,    58,   125,
   127,     0,    24,     0,    72,     0,     0,    51,   117,    52,
     0,   118,     0,   118,    44,   117,     0,    16,     0,    18,
     0,    19,     0,    22,     0,    23,     0,    31,     0,    30,
     0,    59,     0,     0,    32,     0,    32,    51,   120,    52,
   120,     0,    32,    42,   120,    43,   120,     0,    75,    58,
    51,     0,    42,     0,    75,     0,    75,    58,     0,    75,
    58,    53,   119,     0,   121,   120,    52,     0,   123,     0,
   123,    44,   124,     0,    48,   124,    49,     0,    48,    49,
     0,     0,    15,    53,    30,     0,     0,   122,   120,    43,
     0
};

#endif

#if YYDEBUG != 0
static const short yyrline[] = { 0,
   102,   106,   108,   112,   114,   118,   120,   124,   128,   130,
   138,   140,   144,   146,   150,   152,   156,   158,   160,   162,
   164,   166,   168,   170,   172,   176,   178,   180,   184,   186,
   190,   192,   196,   198,   202,   204,   206,   208,   210,   212,
   214,   216,   218,   220,   222,   224,   226,   230,   231,   233,
   235,   239,   243,   245,   249,   253,   255,   257,   259,   261,
   263,   267,   269,   273,   277,   279,   283,   287,   291,   293,
   297,   299,   303,   305,   315,   317,   321,   323,   327,   329,
   333,   335,   339,   343,   347,   349,   353,   355,   359,   361,
   365,   367,   371,   373,   377,   381,   385,   391,   395,   399,
   401,   405,   409,   413,   417,   421,   423,   427,   429,   431,
   435,   437,   439,   443,   445,   449,   453,   455,   457,   459,
   461,   465,   467,   471,   473,   477,   479,   481,   485,   487,
   491,   495,   501,   503,   507,   509,   513,   515,   519,   521,
   523,   525,   527,   531,   533,   535,   539,   541,   543,   549,
   557,   564,   571,   573,   575,   577,   584,   586,   590,   592,
   596,   598,   602,   604
};
#endif


#if YYDEBUG != 0 || defined (YYERROR_VERBOSE)

static const char * const yytname[] = {   "$","error","$undefined.","MODULE",
"MAINMODULE","EXTERN","READONLY","INITCALL","CHARE","MAINCHARE","GROUP","NODEGROUP",
"ARRAY","MESSAGE","CLASS","STACKSIZE","THREADED","TEMPLATE","SYNC","EXCLUSIVE",
"VIRTUAL","MIGRATABLE","CREATEHERE","CREATEHOME","VOID","CONST","PACKED","VARSIZE",
"ENTRY","IDENT","NUMBER","LITERAL","CPROGRAM","INT","LONG","SHORT","CHAR","FLOAT",
"DOUBLE","UNSIGNED","';'","':'","'{'","'}'","','","'<'","'>'","'*'","'('","')'",
"'&'","'['","']'","'='","File","ModuleEList","OptExtern","OptSemiColon","Name",
"QualName","Module","ConstructEList","ConstructList","Construct","TParam","TParamList",
"TParamEList","OptTParams","BuiltinType","NamedType","QualNamedType","SimpleType",
"OnePtrType","PtrType","FuncType","Type","ArrayDim","Dim","DimList","Readonly",
"ReadonlyMsg","OptVoid","InitCall","MAttribs","MAttribList","MAttrib","CAttribs",
"CAttribList","CAttrib","Var","VarList","Message","OptBaseList","BaseList","Chare",
"Group","NodeGroup","ArrayIndexType","Array","TChare","TGroup","TNodeGroup",
"TArray","TMessage","OptTypeInit","OptNameInit","TVar","TVarList","TemplateSpec",
"Template","MemberEList","MemberList","NonEntryMember","Member","Entry","EReturn",
"EAttribs","EAttribList","EAttrib","DefaultParameter","CCode","ParamBracketStart",
"ParamBraceStart","Parameter","ParamList","EParameters","OptStackSize","OptSdagCode", NULL
};
#endif

static const short yyr1[] = {     0,
    54,    55,    55,    56,    56,    57,    57,    58,    59,    59,
    60,    60,    61,    61,    62,    62,    63,    63,    63,    63,
    63,    63,    63,    63,    63,    64,    64,    64,    65,    65,
    66,    66,    67,    67,    68,    68,    68,    68,    68,    68,
    68,    68,    68,    68,    68,    68,    68,    69,    70,    71,
    71,    72,    73,    73,    74,    75,    75,    75,    75,    75,
    75,    76,    76,    77,    78,    78,    79,    80,    81,    81,
    82,    82,    83,    83,    84,    84,    85,    85,    86,    86,
    87,    87,    88,    89,    90,    90,    91,    91,    92,    92,
    93,    93,    94,    94,    95,    96,    97,    97,    98,    99,
    99,   100,   101,   102,   103,   104,   104,   105,   105,   105,
   106,   106,   106,   107,   107,   108,   109,   109,   109,   109,
   109,   110,   110,   111,   111,   112,   112,   112,   113,   113,
   114,   114,   115,   115,   116,   116,   117,   117,   118,   118,
   118,   118,   118,   119,   119,   119,   120,   120,   120,   120,
   121,   122,   123,   123,   123,   123,   124,   124,   125,   125,
   126,   126,   127,   127
};

static const short yyr2[] = {     0,
     1,     0,     2,     0,     1,     0,     1,     1,     1,     4,
     3,     3,     1,     4,     0,     2,     5,     2,     2,     3,
     2,     2,     2,     2,     2,     1,     1,     1,     1,     3,
     0,     1,     0,     3,     1,     1,     1,     1,     2,     2,
     2,     2,     2,     1,     1,     2,     1,     2,     2,     1,
     1,     2,     2,     2,     8,     1,     1,     1,     1,     2,
     2,     1,     1,     3,     0,     2,     4,     5,     0,     1,
     3,     6,     0,     3,     1,     3,     1,     1,     0,     3,
     1,     3,     1,     5,     1,     2,     3,     6,     0,     2,
     1,     3,     5,     5,     5,     5,     4,     3,     5,     5,
     5,     5,     5,     5,     4,     0,     2,     0,     2,     2,
     3,     2,     3,     1,     3,     4,     2,     2,     2,     2,
     2,     1,     4,     0,     2,     2,     2,     2,     2,     1,
     7,     5,     1,     1,     0,     3,     1,     3,     1,     1,
     1,     1,     1,     1,     1,     1,     0,     1,     5,     5,
     3,     1,     1,     2,     4,     3,     1,     3,     3,     2,
     0,     3,     0,     3
};

static const short yydefact[] = {     2,
     0,     0,     1,     2,     8,     0,     0,     3,    13,     4,
    11,    12,     5,     0,     0,     4,     0,    69,    79,    79,
    79,    79,     0,    73,     0,     4,    18,     0,     0,     0,
     0,    21,    22,    23,    24,     0,    25,    19,     6,    16,
     0,    47,     0,     9,    35,    36,    37,    38,    44,    45,
     0,    33,    50,    51,    56,    57,    58,    59,     0,    70,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,   126,   127,   128,    20,    79,    79,    79,    79,
     0,    73,   117,   118,   119,   120,   121,     7,    14,     0,
    61,    43,    46,    39,    40,    41,    42,     0,    31,    49,
    52,    53,    54,     0,    60,    65,    71,    83,     0,    81,
    33,    89,    89,    89,    89,     0,     0,    89,    77,    78,
     0,    75,    87,     0,    59,     0,   114,     0,     6,     0,
     0,     0,     0,     0,     0,     0,     0,    27,    28,    29,
    32,     0,    26,     0,     0,    65,    67,    69,    80,     0,
    48,     0,     0,     0,     0,     0,     0,    98,     0,    74,
     0,     0,   106,     0,   112,   108,     0,   116,    17,    89,
    89,    89,    89,    89,     0,    68,    10,     0,    34,     0,
    62,    63,     0,    66,     0,    82,    91,    90,   122,   124,
    93,    94,    95,    96,    97,    99,    76,     0,    85,     0,
     0,   111,   109,   110,   113,   115,     0,     0,     0,     0,
     0,   105,    30,     0,    64,    72,     0,   135,     0,   130,
   124,     0,     0,    86,    88,   107,   100,   101,   102,   103,
   104,     0,    92,     0,     0,     6,   125,   129,     0,   153,
   147,   157,     0,   139,   140,   141,   142,   143,     0,   137,
    47,     9,     0,     0,   134,     0,   123,     0,   154,   148,
     0,     0,    55,   136,     0,     0,   163,     0,    84,   151,
     0,   147,   147,   156,   158,   138,   160,     0,   152,   147,
   132,   161,   145,   144,   146,   155,     0,     0,   159,     0,
     0,   163,   147,   147,   164,     0,   131,   150,   149,   162,
     0,     0,     0
};

static const short yydefgoto[] = {   301,
     3,    14,    89,   111,    52,     4,    11,    15,    16,   140,
   141,   142,   100,    53,   187,    54,    55,    56,    57,    58,
   240,   183,   146,   147,    28,    29,    61,    30,    70,   121,
   122,    63,   109,   110,   199,   200,    31,   153,   188,    32,
    33,    34,    68,    35,    83,    84,    85,    86,    87,   202,
   165,   127,   128,    36,    37,   191,   219,   220,   221,   222,
   256,   235,   249,   250,   286,   261,   241,   280,   242,   243,
   267,   292,   281
};

static const short yypact[] = {   124,
    -9,    -9,-32768,   124,-32768,    11,    11,-32768,-32768,     3,
-32768,-32768,-32768,    30,    22,     3,    71,    78,    41,    41,
    41,    41,    60,    66,   104,     3,-32768,   114,   132,   146,
   155,-32768,-32768,-32768,-32768,   129,-32768,-32768,   158,-32768,
   183,-32768,   167,-32768,-32768,    14,-32768,-32768,-32768,-32768,
   148,    42,-32768,-32768,   161,   162,   164,-32768,    64,-32768,
   181,   192,    -9,    -9,    -9,    -9,   115,    -9,   136,    -9,
   122,   172,-32768,-32768,-32768,-32768,    41,    41,    41,    41,
    60,    66,-32768,-32768,-32768,-32768,-32768,-32768,-32768,   176,
    51,-32768,-32768,-32768,-32768,-32768,-32768,   184,    25,-32768,
-32768,-32768,-32768,   177,-32768,   -32,   -20,-32768,   175,   186,
   190,   196,   196,   196,   196,    -9,   195,   196,-32768,-32768,
   197,   194,   206,    -9,    45,    68,   208,   204,   158,    -9,
    -9,    -9,    -9,    -9,    -9,    -9,   222,-32768,-32768,   209,
-32768,   211,    51,    -9,   159,   203,-32768,    78,-32768,   192,
-32768,    -9,   110,   110,   110,   110,   212,-32768,   110,-32768,
   136,   167,   205,   163,-32768,   213,   122,-32768,-32768,   196,
   196,   196,   196,   196,   216,-32768,-32768,    25,-32768,   214,
-32768,   218,   215,-32768,   219,-32768,   217,-32768,-32768,     7,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,    68,   167,   226,
   167,-32768,-32768,-32768,-32768,-32768,   110,   110,   110,   110,
   110,-32768,-32768,   223,-32768,-32768,    -9,   221,   227,-32768,
     7,   220,   224,-32768,-32768,    51,-32768,-32768,-32768,-32768,
-32768,   167,-32768,   210,   207,   158,-32768,-32768,   225,    68,
   230,   229,   231,-32768,-32768,-32768,-32768,-32768,   232,   234,
   236,   228,   233,   161,-32768,    -9,-32768,   239,   120,    35,
   235,   167,-32768,-32768,   210,   141,   240,   233,-32768,-32768,
   138,   230,   230,-32768,-32768,-32768,-32768,   237,-32768,   230,
-32768,   259,-32768,-32768,   218,-32768,   242,   238,-32768,   245,
   241,   240,   230,   230,-32768,   253,-32768,-32768,-32768,-32768,
   289,   291,-32768
};

static const short yypgoto[] = {-32768,
   288,-32768,  -123,    -1,   -57,   279,   290,     6,-32768,-32768,
   117,-32768,   185,-32768,    56,-32768,   -36,    63,-32768,   -64,
   -14,-32768,-32768,   153,-32768,-32768,   152,-32768,   243,   140,
-32768,    -4,   154,-32768,-32768,   103,-32768,  -103,    86,-32768,
-32768,-32768,   246,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
   139,-32768,   142,-32768,-32768,  -129,    85,   293,-32768,-32768,
-32768,-32768,    43,-32768,-32768,  -249,-32768,-32768,-32768,  -172,
    44,-32768,    18
};


#define	YYLAST		327


static const short yytable[] = {     6,
     7,   106,    59,   107,    90,   169,   125,    13,    98,   154,
   155,   156,    17,    18,   159,    64,    65,    66,   145,     5,
    98,    40,   287,   288,   192,   193,   194,   148,    91,   196,
   290,    72,     1,     2,   218,    17,    18,    19,    20,    21,
    22,    23,    24,   298,   299,   -15,    25,    92,    42,    43,
     9,    93,    10,    44,   138,   139,   126,    45,    46,    47,
    48,    49,    50,    51,    39,   117,   207,   208,   209,   210,
   211,    26,   130,   131,   132,   133,   272,   227,   228,   229,
   230,   231,    98,    41,   143,   273,    99,   182,  -108,   275,
  -108,    62,    44,   278,    42,    43,     5,   164,   104,    44,
   105,    60,   125,    45,    46,    47,    48,    49,    50,    51,
    67,   104,   257,   105,   157,   104,    69,   105,   112,   113,
   114,   115,   163,   118,   166,   123,     1,     2,   170,   171,
   172,   173,   174,   175,   176,   124,    77,    78,    79,    80,
    81,    82,   180,     5,   116,    42,    43,   198,    71,   189,
    44,   190,   126,    73,    45,    46,    47,    48,    49,    50,
    51,   119,   120,   143,    42,    43,    44,   283,   284,    44,
   270,    74,   271,    45,    46,    47,    48,    49,    50,    51,
    94,    95,    96,    97,   198,    75,   226,    44,   181,   277,
    42,    43,   203,   204,    76,    44,   223,    88,   254,    45,
    46,    47,    48,    49,    50,    51,    42,   101,   102,    44,
   103,    44,   108,   285,   129,    45,    46,    47,    48,    49,
    50,    51,   136,   144,   137,   244,   149,   245,   246,   150,
   251,   247,   248,   253,    99,   252,   152,   161,   259,    45,
    46,    47,    48,    49,    50,    51,   158,   162,   160,   168,
   177,   167,   178,   145,   268,   212,   179,   201,    98,   238,
   217,   260,   214,   195,  -133,   164,   215,   216,   225,   236,
   232,   234,   262,   291,   239,    -8,   258,   265,   269,   263,
   266,   279,   300,   264,   293,   289,   274,   295,   302,   294,
   303,     8,    27,   296,   213,   151,    12,   255,   184,   185,
   197,   224,   233,   186,   205,   237,    38,   276,   206,   297,
     0,   282,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,   135,     0,   134
};

static const short yycheck[] = {     1,
     2,    59,    17,    61,    41,   129,    71,     5,    41,   113,
   114,   115,     6,     7,   118,    20,    21,    22,    51,    29,
    41,    16,   272,   273,   154,   155,   156,    48,    43,   159,
   280,    26,     3,     4,    28,     6,     7,     8,     9,    10,
    11,    12,    13,   293,   294,    43,    17,    34,    24,    25,
    40,    38,    42,    29,    30,    31,    71,    33,    34,    35,
    36,    37,    38,    39,    43,    67,   170,   171,   172,   173,
   174,    42,    77,    78,    79,    80,    42,   207,   208,   209,
   210,   211,    41,    13,    99,    51,    45,   145,    44,   262,
    46,    51,    29,   266,    24,    25,    29,    53,    48,    29,
    50,    24,   167,    33,    34,    35,    36,    37,    38,    39,
    51,    48,   236,    50,   116,    48,    51,    50,    63,    64,
    65,    66,   124,    68,   126,    70,     3,     4,   130,   131,
   132,   133,   134,   135,   136,    14,     8,     9,    10,    11,
    12,    13,   144,    29,    30,    24,    25,   162,    45,    40,
    29,    42,   167,    40,    33,    34,    35,    36,    37,    38,
    39,    26,    27,   178,    24,    25,    29,    30,    31,    29,
    51,    40,    53,    33,    34,    35,    36,    37,    38,    39,
    33,    34,    35,    36,   199,    40,   201,    29,    30,    49,
    24,    25,    30,    31,    40,    29,   198,    40,   235,    33,
    34,    35,    36,    37,    38,    39,    24,    47,    47,    29,
    47,    29,    21,   271,    43,    33,    34,    35,    36,    37,
    38,    39,    47,    47,    41,    16,    52,    18,    19,    44,
    24,    22,    23,   235,    45,    29,    41,    44,   240,    33,
    34,    35,    36,    37,    38,    39,    52,    42,    52,    46,
    29,    44,    44,    51,   256,    40,    46,    53,    41,    40,
    44,    32,    49,    52,    29,    53,    52,    49,    43,    43,
    48,    51,    44,    15,    51,    48,    52,    44,    40,    49,
    48,    42,    30,    52,    43,    49,    52,    43,     0,    52,
     0,     4,    14,    53,   178,   111,     7,   235,   146,   148,
   161,   199,   217,   150,   166,   221,    14,   265,   167,   292,
    -1,   268,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    82,    -1,    81
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
#line 103 "xi-grammar.y"
{ yyval.modlist = yyvsp[0].modlist; modlist = yyvsp[0].modlist; ;
    break;}
case 2:
#line 107 "xi-grammar.y"
{ yyval.modlist = 0; ;
    break;}
case 3:
#line 109 "xi-grammar.y"
{ yyval.modlist = new ModuleList(lineno, yyvsp[-1].module, yyvsp[0].modlist); ;
    break;}
case 4:
#line 113 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 5:
#line 115 "xi-grammar.y"
{ yyval.intval = 1; ;
    break;}
case 6:
#line 119 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 7:
#line 121 "xi-grammar.y"
{ yyval.intval = 1; ;
    break;}
case 8:
#line 125 "xi-grammar.y"
{ yyval.strval = yyvsp[0].strval; ;
    break;}
case 9:
#line 129 "xi-grammar.y"
{ yyval.strval = yyvsp[0].strval; ;
    break;}
case 10:
#line 131 "xi-grammar.y"
{
		  char *tmp = new char[strlen(yyvsp[-3].strval)+strlen(yyvsp[0].strval)+3];
		  sprintf(tmp,"%s::%s", yyvsp[-3].strval, yyvsp[0].strval);
		  yyval.strval = tmp;
		;
    break;}
case 11:
#line 139 "xi-grammar.y"
{ yyval.module = new Module(lineno, yyvsp[-1].strval, yyvsp[0].conslist); ;
    break;}
case 12:
#line 141 "xi-grammar.y"
{ yyval.module = new Module(lineno, yyvsp[-1].strval, yyvsp[0].conslist); yyval.module->setMain(); ;
    break;}
case 13:
#line 145 "xi-grammar.y"
{ yyval.conslist = 0; ;
    break;}
case 14:
#line 147 "xi-grammar.y"
{ yyval.conslist = yyvsp[-2].conslist; ;
    break;}
case 15:
#line 151 "xi-grammar.y"
{ yyval.conslist = 0; ;
    break;}
case 16:
#line 153 "xi-grammar.y"
{ yyval.conslist = new ConstructList(lineno, yyvsp[-1].construct, yyvsp[0].conslist); ;
    break;}
case 17:
#line 157 "xi-grammar.y"
{ if(yyvsp[-2].conslist) yyvsp[-2].conslist->setExtern(yyvsp[-4].intval); yyval.construct = yyvsp[-2].conslist; ;
    break;}
case 18:
#line 159 "xi-grammar.y"
{ yyvsp[0].module->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].module; ;
    break;}
case 19:
#line 161 "xi-grammar.y"
{ yyvsp[0].member->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].member; ;
    break;}
case 20:
#line 163 "xi-grammar.y"
{ yyvsp[-1].message->setExtern(yyvsp[-2].intval); yyval.construct = yyvsp[-1].message; ;
    break;}
case 21:
#line 165 "xi-grammar.y"
{ yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; ;
    break;}
case 22:
#line 167 "xi-grammar.y"
{ yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; ;
    break;}
case 23:
#line 169 "xi-grammar.y"
{ yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; ;
    break;}
case 24:
#line 171 "xi-grammar.y"
{ yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; ;
    break;}
case 25:
#line 173 "xi-grammar.y"
{ yyvsp[0].templat->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].templat; ;
    break;}
case 26:
#line 177 "xi-grammar.y"
{ yyval.tparam = new TParamType(yyvsp[0].type); ;
    break;}
case 27:
#line 179 "xi-grammar.y"
{ yyval.tparam = new TParamVal(yyvsp[0].strval); ;
    break;}
case 28:
#line 181 "xi-grammar.y"
{ yyval.tparam = new TParamVal(yyvsp[0].strval); ;
    break;}
case 29:
#line 185 "xi-grammar.y"
{ yyval.tparlist = new TParamList(yyvsp[0].tparam); ;
    break;}
case 30:
#line 187 "xi-grammar.y"
{ yyval.tparlist = new TParamList(yyvsp[-2].tparam, yyvsp[0].tparlist); ;
    break;}
case 31:
#line 191 "xi-grammar.y"
{ yyval.tparlist = 0; ;
    break;}
case 32:
#line 193 "xi-grammar.y"
{ yyval.tparlist = yyvsp[0].tparlist; ;
    break;}
case 33:
#line 197 "xi-grammar.y"
{ yyval.tparlist = 0; ;
    break;}
case 34:
#line 199 "xi-grammar.y"
{ yyval.tparlist = yyvsp[-1].tparlist; ;
    break;}
case 35:
#line 203 "xi-grammar.y"
{ yyval.type = new BuiltinType("int"); ;
    break;}
case 36:
#line 205 "xi-grammar.y"
{ yyval.type = new BuiltinType("long"); ;
    break;}
case 37:
#line 207 "xi-grammar.y"
{ yyval.type = new BuiltinType("short"); ;
    break;}
case 38:
#line 209 "xi-grammar.y"
{ yyval.type = new BuiltinType("char"); ;
    break;}
case 39:
#line 211 "xi-grammar.y"
{ yyval.type = new BuiltinType("unsigned int"); ;
    break;}
case 40:
#line 213 "xi-grammar.y"
{ yyval.type = new BuiltinType("unsigned long"); ;
    break;}
case 41:
#line 215 "xi-grammar.y"
{ yyval.type = new BuiltinType("unsigned short"); ;
    break;}
case 42:
#line 217 "xi-grammar.y"
{ yyval.type = new BuiltinType("unsigned char"); ;
    break;}
case 43:
#line 219 "xi-grammar.y"
{ yyval.type = new BuiltinType("long long"); ;
    break;}
case 44:
#line 221 "xi-grammar.y"
{ yyval.type = new BuiltinType("float"); ;
    break;}
case 45:
#line 223 "xi-grammar.y"
{ yyval.type = new BuiltinType("double"); ;
    break;}
case 46:
#line 225 "xi-grammar.y"
{ yyval.type = new BuiltinType("long double"); ;
    break;}
case 47:
#line 227 "xi-grammar.y"
{ yyval.type = new BuiltinType("void"); ;
    break;}
case 48:
#line 230 "xi-grammar.y"
{ yyval.ntype = new NamedType(yyvsp[-1].strval,yyvsp[0].tparlist); ;
    break;}
case 49:
#line 231 "xi-grammar.y"
{ yyval.ntype = new NamedType(yyvsp[-1].strval,yyvsp[0].tparlist); ;
    break;}
case 50:
#line 234 "xi-grammar.y"
{ yyval.type = yyvsp[0].type; ;
    break;}
case 51:
#line 236 "xi-grammar.y"
{ yyval.type = yyvsp[0].ntype; ;
    break;}
case 52:
#line 240 "xi-grammar.y"
{ yyval.ptype = new PtrType(yyvsp[-1].type); ;
    break;}
case 53:
#line 244 "xi-grammar.y"
{ yyvsp[-1].ptype->indirect(); yyval.ptype = yyvsp[-1].ptype; ;
    break;}
case 54:
#line 246 "xi-grammar.y"
{ yyvsp[-1].ptype->indirect(); yyval.ptype = yyvsp[-1].ptype; ;
    break;}
case 55:
#line 250 "xi-grammar.y"
{ yyval.ftype = new FuncType(yyvsp[-7].type, yyvsp[-4].strval, yyvsp[-1].plist); ;
    break;}
case 56:
#line 254 "xi-grammar.y"
{ yyval.type = yyvsp[0].type; ;
    break;}
case 57:
#line 256 "xi-grammar.y"
{ yyval.type = yyvsp[0].ptype; ;
    break;}
case 58:
#line 258 "xi-grammar.y"
{ yyval.type = yyvsp[0].ptype; ;
    break;}
case 59:
#line 260 "xi-grammar.y"
{ yyval.type = yyvsp[0].ftype; ;
    break;}
case 60:
#line 262 "xi-grammar.y"
{ yyval.type = yyvsp[-1].type; ;
    break;}
case 61:
#line 264 "xi-grammar.y"
{ yyval.type = yyvsp[0].type; ;
    break;}
case 62:
#line 268 "xi-grammar.y"
{ yyval.val = new Value(yyvsp[0].strval); ;
    break;}
case 63:
#line 270 "xi-grammar.y"
{ yyval.val = new Value(yyvsp[0].strval); ;
    break;}
case 64:
#line 274 "xi-grammar.y"
{ yyval.val = yyvsp[-1].val; ;
    break;}
case 65:
#line 278 "xi-grammar.y"
{ yyval.vallist = 0; ;
    break;}
case 66:
#line 280 "xi-grammar.y"
{ yyval.vallist = new ValueList(yyvsp[-1].val, yyvsp[0].vallist); ;
    break;}
case 67:
#line 284 "xi-grammar.y"
{ yyval.readonly = new Readonly(lineno, yyvsp[-2].type, yyvsp[-1].strval, yyvsp[0].vallist); ;
    break;}
case 68:
#line 288 "xi-grammar.y"
{ yyval.readonly = new Readonly(lineno, yyvsp[-2].type, yyvsp[0].strval, 0, 1); ;
    break;}
case 69:
#line 292 "xi-grammar.y"
{ yyval.intval = 0;;
    break;}
case 70:
#line 294 "xi-grammar.y"
{ yyval.intval = 0;;
    break;}
case 71:
#line 298 "xi-grammar.y"
{ yyval.member = new InitCall(lineno, yyvsp[0].strval); ;
    break;}
case 72:
#line 300 "xi-grammar.y"
{ yyval.member = new InitCall(lineno, yyvsp[-3].strval); ;
    break;}
case 73:
#line 304 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 74:
#line 306 "xi-grammar.y"
{ 
		  /*
		  printf("Warning: Message attributes are being phased out.\n");
		  printf("Warning: Please remove them from interface files.\n");
		  */
		  yyval.intval = yyvsp[-1].intval; 
		;
    break;}
case 75:
#line 316 "xi-grammar.y"
{ yyval.intval = yyvsp[0].intval; ;
    break;}
case 76:
#line 318 "xi-grammar.y"
{ yyval.intval = yyvsp[-2].intval | yyvsp[0].intval; ;
    break;}
case 77:
#line 322 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 78:
#line 324 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 79:
#line 328 "xi-grammar.y"
{ yyval.cattr = 0; ;
    break;}
case 80:
#line 330 "xi-grammar.y"
{ yyval.cattr = yyvsp[-1].cattr; ;
    break;}
case 81:
#line 334 "xi-grammar.y"
{ yyval.cattr = yyvsp[0].cattr; ;
    break;}
case 82:
#line 336 "xi-grammar.y"
{ yyval.cattr = yyvsp[-2].cattr | yyvsp[0].cattr; ;
    break;}
case 83:
#line 340 "xi-grammar.y"
{ yyval.cattr = Chare::CMIGRATABLE; ;
    break;}
case 84:
#line 344 "xi-grammar.y"
{ yyval.mv = new MsgVar(yyvsp[-4].type, yyvsp[-3].strval); ;
    break;}
case 85:
#line 348 "xi-grammar.y"
{ yyval.mvlist = new MsgVarList(yyvsp[0].mv); ;
    break;}
case 86:
#line 350 "xi-grammar.y"
{ yyval.mvlist = new MsgVarList(yyvsp[-1].mv, yyvsp[0].mvlist); ;
    break;}
case 87:
#line 354 "xi-grammar.y"
{ yyval.message = new Message(lineno, yyvsp[0].ntype); ;
    break;}
case 88:
#line 356 "xi-grammar.y"
{ yyval.message = new Message(lineno, yyvsp[-3].ntype, yyvsp[-1].mvlist); ;
    break;}
case 89:
#line 360 "xi-grammar.y"
{ yyval.typelist = 0; ;
    break;}
case 90:
#line 362 "xi-grammar.y"
{ yyval.typelist = yyvsp[0].typelist; ;
    break;}
case 91:
#line 366 "xi-grammar.y"
{ yyval.typelist = new TypeList(yyvsp[0].ntype); ;
    break;}
case 92:
#line 368 "xi-grammar.y"
{ yyval.typelist = new TypeList(yyvsp[-2].ntype, yyvsp[0].typelist); ;
    break;}
case 93:
#line 372 "xi-grammar.y"
{ yyval.chare = new Chare(lineno, yyvsp[-3].cattr, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 94:
#line 374 "xi-grammar.y"
{ yyval.chare = new MainChare(lineno, yyvsp[-3].cattr, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 95:
#line 378 "xi-grammar.y"
{ yyval.chare = new Group(lineno, yyvsp[-3].cattr, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 96:
#line 382 "xi-grammar.y"
{ yyval.chare = new NodeGroup(lineno, yyvsp[-3].cattr, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 97:
#line 386 "xi-grammar.y"
{/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",yyvsp[-2].strval);
			yyval.ntype = new NamedType(buf); 
		;
    break;}
case 98:
#line 392 "xi-grammar.y"
{ yyval.ntype = new NamedType(yyvsp[-1].strval); ;
    break;}
case 99:
#line 396 "xi-grammar.y"
{ yyval.chare = new Array(lineno, 0, yyvsp[-3].ntype, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 100:
#line 400 "xi-grammar.y"
{ yyval.chare = new Chare(lineno, yyvsp[-3].cattr, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist);;
    break;}
case 101:
#line 402 "xi-grammar.y"
{ yyval.chare = new MainChare(lineno, yyvsp[-3].cattr, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 102:
#line 406 "xi-grammar.y"
{ yyval.chare = new Group(lineno, yyvsp[-3].cattr, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 103:
#line 410 "xi-grammar.y"
{ yyval.chare = new NodeGroup( lineno, yyvsp[-3].cattr, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 104:
#line 414 "xi-grammar.y"
{ yyval.chare = new Array( lineno, 0, yyvsp[-3].ntype, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 105:
#line 418 "xi-grammar.y"
{ yyval.message = new Message(lineno, new NamedType(yyvsp[-1].strval)); ;
    break;}
case 106:
#line 422 "xi-grammar.y"
{ yyval.type = 0; ;
    break;}
case 107:
#line 424 "xi-grammar.y"
{ yyval.type = yyvsp[0].type; ;
    break;}
case 108:
#line 428 "xi-grammar.y"
{ yyval.strval = 0; ;
    break;}
case 109:
#line 430 "xi-grammar.y"
{ yyval.strval = yyvsp[0].strval; ;
    break;}
case 110:
#line 432 "xi-grammar.y"
{ yyval.strval = yyvsp[0].strval; ;
    break;}
case 111:
#line 436 "xi-grammar.y"
{ yyval.tvar = new TType(new NamedType(yyvsp[-1].strval), yyvsp[0].type); ;
    break;}
case 112:
#line 438 "xi-grammar.y"
{ yyval.tvar = new TFunc(yyvsp[-1].ftype, yyvsp[0].strval); ;
    break;}
case 113:
#line 440 "xi-grammar.y"
{ yyval.tvar = new TName(yyvsp[-2].type, yyvsp[-1].strval, yyvsp[0].strval); ;
    break;}
case 114:
#line 444 "xi-grammar.y"
{ yyval.tvarlist = new TVarList(yyvsp[0].tvar); ;
    break;}
case 115:
#line 446 "xi-grammar.y"
{ yyval.tvarlist = new TVarList(yyvsp[-2].tvar, yyvsp[0].tvarlist); ;
    break;}
case 116:
#line 450 "xi-grammar.y"
{ yyval.tvarlist = yyvsp[-1].tvarlist; ;
    break;}
case 117:
#line 454 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); ;
    break;}
case 118:
#line 456 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); ;
    break;}
case 119:
#line 458 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); ;
    break;}
case 120:
#line 460 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); ;
    break;}
case 121:
#line 462 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].message); yyvsp[0].message->setTemplate(yyval.templat); ;
    break;}
case 122:
#line 466 "xi-grammar.y"
{ yyval.mbrlist = 0; ;
    break;}
case 123:
#line 468 "xi-grammar.y"
{ yyval.mbrlist = yyvsp[-2].mbrlist; ;
    break;}
case 124:
#line 472 "xi-grammar.y"
{ yyval.mbrlist = 0; ;
    break;}
case 125:
#line 474 "xi-grammar.y"
{ yyval.mbrlist = new MemberList(yyvsp[-1].member, yyvsp[0].mbrlist); ;
    break;}
case 126:
#line 478 "xi-grammar.y"
{ yyval.member = yyvsp[-1].readonly; ;
    break;}
case 127:
#line 480 "xi-grammar.y"
{ yyval.member = yyvsp[-1].readonly; ;
    break;}
case 128:
#line 482 "xi-grammar.y"
{ yyval.member = yyvsp[-1].member; ;
    break;}
case 129:
#line 486 "xi-grammar.y"
{ yyval.member = yyvsp[-1].entry; ;
    break;}
case 130:
#line 488 "xi-grammar.y"
{ yyval.member = yyvsp[0].member; ;
    break;}
case 131:
#line 492 "xi-grammar.y"
{ yyval.entry = new Entry(lineno, yyvsp[-5].intval, yyvsp[-4].type, yyvsp[-3].strval, yyvsp[-2].plist, yyvsp[-1].val); 
		  yyval.entry->setSdagCode(yyvsp[0].strval);
		;
    break;}
case 132:
#line 496 "xi-grammar.y"
{ yyval.entry = new Entry(lineno, yyvsp[-3].intval,     0, yyvsp[-2].strval, yyvsp[-1].plist,  0); 
		  yyval.entry->setSdagCode(yyvsp[0].strval);
		;
    break;}
case 133:
#line 502 "xi-grammar.y"
{ yyval.type = new BuiltinType("void"); ;
    break;}
case 134:
#line 504 "xi-grammar.y"
{ yyval.type = yyvsp[0].ptype; ;
    break;}
case 135:
#line 508 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 136:
#line 510 "xi-grammar.y"
{ yyval.intval = yyvsp[-1].intval; ;
    break;}
case 137:
#line 514 "xi-grammar.y"
{ yyval.intval = yyvsp[0].intval; ;
    break;}
case 138:
#line 516 "xi-grammar.y"
{ yyval.intval = yyvsp[-2].intval | yyvsp[0].intval; ;
    break;}
case 139:
#line 520 "xi-grammar.y"
{ yyval.intval = STHREADED; ;
    break;}
case 140:
#line 522 "xi-grammar.y"
{ yyval.intval = SSYNC; ;
    break;}
case 141:
#line 524 "xi-grammar.y"
{ yyval.intval = SLOCKED; ;
    break;}
case 142:
#line 526 "xi-grammar.y"
{ yyval.intval = SCREATEHERE; ;
    break;}
case 143:
#line 528 "xi-grammar.y"
{ yyval.intval = SCREATEHOME; ;
    break;}
case 144:
#line 532 "xi-grammar.y"
{ yyval.val = new Value(yyvsp[0].strval); ;
    break;}
case 145:
#line 534 "xi-grammar.y"
{ yyval.val = new Value(yyvsp[0].strval); ;
    break;}
case 146:
#line 536 "xi-grammar.y"
{ yyval.val = new Value(yyvsp[0].strval); ;
    break;}
case 147:
#line 540 "xi-grammar.y"
{ yyval.strval = ""; ;
    break;}
case 148:
#line 542 "xi-grammar.y"
{ yyval.strval = yyvsp[0].strval; ;
    break;}
case 149:
#line 544 "xi-grammar.y"
{  /*Returned only when in_bracket*/
			char *tmp = new char[strlen(yyvsp[-4].strval)+strlen(yyvsp[-2].strval)+strlen(yyvsp[0].strval)+3];
			sprintf(tmp,"%s[%s]%s", yyvsp[-4].strval, yyvsp[-2].strval, yyvsp[0].strval);
			yyval.strval = tmp;
		;
    break;}
case 150:
#line 550 "xi-grammar.y"
{ /*Returned only when in_braces*/
			char *tmp = new char[strlen(yyvsp[-4].strval)+strlen(yyvsp[-2].strval)+strlen(yyvsp[0].strval)+3];
			sprintf(tmp,"%s{%s}%s", yyvsp[-4].strval, yyvsp[-2].strval, yyvsp[0].strval);
			yyval.strval = tmp;
		;
    break;}
case 151:
#line 558 "xi-grammar.y"
{  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			yyval.pname = new Parameter(lineno, yyvsp[-2].type,yyvsp[-1].strval);
		;
    break;}
case 152:
#line 565 "xi-grammar.y"
{  /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			yyval.intval = 0;
		;
    break;}
case 153:
#line 572 "xi-grammar.y"
{ yyval.pname = new Parameter(lineno, yyvsp[0].type);;
    break;}
case 154:
#line 574 "xi-grammar.y"
{ yyval.pname = new Parameter(lineno, yyvsp[-1].type,yyvsp[0].strval);;
    break;}
case 155:
#line 576 "xi-grammar.y"
{ yyval.pname = new Parameter(lineno, yyvsp[-3].type,yyvsp[-2].strval,0,yyvsp[0].val);;
    break;}
case 156:
#line 578 "xi-grammar.y"
{ /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			yyval.pname = new Parameter(lineno, yyvsp[-2].pname->getType(), yyvsp[-2].pname->getName() ,yyvsp[-1].strval);
		;
    break;}
case 157:
#line 585 "xi-grammar.y"
{ yyval.plist = new ParamList(yyvsp[0].pname); ;
    break;}
case 158:
#line 587 "xi-grammar.y"
{ yyval.plist = new ParamList(yyvsp[-2].pname,yyvsp[0].plist); ;
    break;}
case 159:
#line 591 "xi-grammar.y"
{ yyval.plist = yyvsp[-1].plist; ;
    break;}
case 160:
#line 593 "xi-grammar.y"
{ yyval.plist = 0; ;
    break;}
case 161:
#line 597 "xi-grammar.y"
{ yyval.val = 0; ;
    break;}
case 162:
#line 599 "xi-grammar.y"
{ yyval.val = new Value(yyvsp[0].strval); ;
    break;}
case 163:
#line 603 "xi-grammar.y"
{ yyval.strval = 0; ;
    break;}
case 164:
#line 605 "xi-grammar.y"
{ in_braces = 0; yyval.strval = yyvsp[-1].strval; ;
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
#line 608 "xi-grammar.y"

void yyerror(const char *mesg)
{
  cout << cur_file<<":"<<lineno<<": Charmxi syntax error> " << mesg << endl;
  // return 0;
}
