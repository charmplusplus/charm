
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



#define	YYFINAL		263
#define	YYFLAG		-32768
#define	YYNTBASE	48

#define YYTRANSLATE(x) ((unsigned)(x) <= 288 ? yytranslate[x] : 111)

static const char yytranslate[] = {     0,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,    44,
    45,    41,     2,    38,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,    46,    35,    39,
    47,    40,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
    42,     2,    43,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,    36,     2,    37,     2,     2,     2,     2,     2,
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
     0,     2,     3,     6,     7,     9,    10,    12,    14,    18,
    22,    24,    29,    30,    33,    39,    42,    46,    50,    54,
    57,    60,    63,    66,    69,    71,    73,    75,    77,    81,
    82,    84,    85,    89,    91,    93,    95,    97,   100,   103,
   106,   109,   112,   114,   116,   119,   121,   124,   126,   128,
   131,   134,   137,   139,   141,   146,   155,   157,   159,   161,
   163,   165,   166,   168,   172,   176,   177,   180,   185,   191,
   192,   196,   198,   202,   204,   206,   207,   211,   213,   217,
   219,   223,   230,   231,   234,   236,   240,   246,   252,   258,
   264,   269,   273,   279,   285,   291,   297,   303,   309,   314,
   315,   318,   319,   322,   325,   329,   332,   336,   338,   342,
   347,   350,   353,   356,   359,   362,   364,   369,   370,   373,
   376,   379,   382,   390,   398,   403,   404,   408,   410,   414,
   416,   418,   420,   422,   423,   425,   427,   431,   432,   436,
   437
};

static const short yyrhs[] = {    49,
     0,     0,    53,    49,     0,     0,     5,     0,     0,    35,
     0,    25,     0,     3,    52,    54,     0,     4,    52,    54,
     0,    35,     0,    36,    55,    37,    51,     0,     0,    56,
    55,     0,    50,    36,    55,    37,    51,     0,    50,    53,
     0,    50,    73,    35,     0,    50,    74,    35,     0,    50,
    81,    35,     0,    50,    84,     0,    50,    85,     0,    50,
    86,     0,    50,    88,     0,    50,    99,     0,    69,     0,
    26,     0,    27,     0,    57,     0,    57,    38,    58,     0,
     0,    58,     0,     0,    39,    59,    40,     0,    28,     0,
    29,     0,    30,     0,    31,     0,    34,    28,     0,    34,
    29,     0,    34,    30,     0,    34,    31,     0,    29,    29,
     0,    32,     0,    33,     0,    29,    33,     0,    20,     0,
    52,    60,     0,    61,     0,    62,     0,    63,    41,     0,
    64,    41,     0,    65,    41,     0,    26,     0,    52,     0,
    69,    42,    66,    43,     0,    69,    44,    41,    52,    45,
    44,    70,    45,     0,    63,     0,    64,     0,    65,     0,
    67,     0,    68,     0,     0,    69,     0,    69,    38,    70,
     0,    42,    66,    43,     0,     0,    71,    72,     0,     6,
    69,    52,    72,     0,     6,    11,    63,    41,    52,     0,
     0,    42,    76,    43,     0,    77,     0,    77,    38,    76,
     0,    21,     0,    22,     0,     0,    42,    79,    43,     0,
    80,     0,    80,    38,    79,     0,    15,     0,    11,    75,
    62,     0,    11,    75,    62,    36,    70,    37,     0,     0,
    46,    83,     0,    62,     0,    62,    38,    83,     0,     7,
    78,    62,    82,   100,     0,    24,    78,    62,    82,   100,
     0,     8,    78,    62,    82,   100,     0,     9,    78,    62,
    82,   100,     0,    42,    26,    52,    43,     0,    42,    52,
    43,     0,    10,    87,    62,    82,   100,     0,     7,    78,
    52,    82,   100,     0,    24,    78,    52,    82,   100,     0,
     8,    78,    52,    82,   100,     0,     9,    78,    52,    82,
   100,     0,    10,    87,    52,    82,   100,     0,    11,    75,
    52,    35,     0,     0,    47,    69,     0,     0,    47,    26,
     0,    47,    27,     0,    12,    52,    94,     0,    68,    95,
     0,    69,    52,    95,     0,    96,     0,    96,    38,    97,
     0,    16,    39,    97,    40,     0,    98,    89,     0,    98,
    90,     0,    98,    91,     0,    98,    92,     0,    98,    93,
     0,    35,     0,    36,   101,    37,    51,     0,     0,   102,
   101,     0,   103,    35,     0,    73,    35,     0,    74,    35,
     0,    23,   104,    20,    52,   108,   110,   109,     0,    23,
   104,    64,    52,   108,   110,   109,     0,    23,   104,    52,
   108,     0,     0,    42,   105,    43,     0,   106,     0,   106,
    38,   105,     0,    14,     0,    17,     0,    18,     0,    19,
     0,     0,    20,     0,    64,     0,    44,   107,    45,     0,
     0,    13,    47,    26,     0,     0,    47,    26,     0
};

#endif

#if YYDEBUG != 0
static const short yyrline[] = { 0,
    92,    96,    98,   102,   104,   108,   110,   114,   118,   120,
   124,   126,   130,   132,   136,   138,   140,   142,   144,   146,
   148,   150,   152,   154,   158,   160,   162,   166,   168,   172,
   174,   178,   180,   184,   186,   188,   190,   192,   194,   196,
   198,   200,   202,   204,   206,   208,   212,   216,   218,   222,
   226,   228,   232,   234,   238,   242,   246,   248,   250,   252,
   254,   258,   260,   262,   266,   270,   272,   276,   280,   284,
   286,   290,   292,   296,   298,   302,   304,   308,   310,   314,
   318,   320,   324,   326,   330,   332,   336,   338,   342,   346,
   350,   356,   360,   364,   366,   370,   374,   378,   382,   386,
   388,   392,   394,   396,   400,   402,   404,   408,   410,   414,
   418,   420,   422,   424,   426,   430,   432,   436,   438,   442,
   444,   446,   450,   452,   454,   458,   460,   464,   466,   470,
   472,   474,   476,   480,   482,   484,   488,   492,   494,   498,
   500
};
#endif


#if YYDEBUG != 0 || defined (YYERROR_VERBOSE)

static const char * const yytname[] = {   "$","error","$undefined.","MODULE",
"MAINMODULE","EXTERN","READONLY","CHARE","GROUP","NODEGROUP","ARRAY","MESSAGE",
"CLASS","STACKSIZE","THREADED","MIGRATABLE","TEMPLATE","SYNC","EXCLUSIVE","VIRTUAL",
"VOID","PACKED","VARSIZE","ENTRY","MAINCHARE","IDENT","NUMBER","LITERAL","INT",
"LONG","SHORT","CHAR","FLOAT","DOUBLE","UNSIGNED","';'","'{'","'}'","','","'<'",
"'>'","'*'","'['","']'","'('","')'","':'","'='","File","ModuleEList","OptExtern",
"OptSemiColon","Name","Module","ConstructEList","ConstructList","Construct",
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
    54,    54,    55,    55,    56,    56,    56,    56,    56,    56,
    56,    56,    56,    56,    57,    57,    57,    58,    58,    59,
    59,    60,    60,    61,    61,    61,    61,    61,    61,    61,
    61,    61,    61,    61,    61,    61,    62,    63,    63,    64,
    65,    65,    66,    66,    67,    68,    69,    69,    69,    69,
    69,    70,    70,    70,    71,    72,    72,    73,    74,    75,
    75,    76,    76,    77,    77,    78,    78,    79,    79,    80,
    81,    81,    82,    82,    83,    83,    84,    84,    85,    86,
    87,    87,    88,    89,    89,    90,    91,    92,    93,    94,
    94,    95,    95,    95,    96,    96,    96,    97,    97,    98,
    99,    99,    99,    99,    99,   100,   100,   101,   101,   102,
   102,   102,   103,   103,   103,   104,   104,   105,   105,   106,
   106,   106,   106,   107,   107,   107,   108,   109,   109,   110,
   110
};

static const short yyr2[] = {     0,
     1,     0,     2,     0,     1,     0,     1,     1,     3,     3,
     1,     4,     0,     2,     5,     2,     3,     3,     3,     2,
     2,     2,     2,     2,     1,     1,     1,     1,     3,     0,
     1,     0,     3,     1,     1,     1,     1,     2,     2,     2,
     2,     2,     1,     1,     2,     1,     2,     1,     1,     2,
     2,     2,     1,     1,     4,     8,     1,     1,     1,     1,
     1,     0,     1,     3,     3,     0,     2,     4,     5,     0,
     3,     1,     3,     1,     1,     0,     3,     1,     3,     1,
     3,     6,     0,     2,     1,     3,     5,     5,     5,     5,
     4,     3,     5,     5,     5,     5,     5,     5,     4,     0,
     2,     0,     2,     2,     3,     2,     3,     1,     3,     4,
     2,     2,     2,     2,     2,     1,     4,     0,     2,     2,
     2,     2,     7,     7,     4,     0,     3,     1,     3,     1,
     1,     1,     1,     0,     1,     1,     3,     0,     3,     0,
     2
};

static const short yydefact[] = {     2,
     0,     0,     1,     2,     8,     0,     0,     3,    11,     4,
     9,    10,     5,     0,     0,     4,     0,    76,    76,    76,
     0,    70,     0,    76,     4,    16,     0,     0,     0,    20,
    21,    22,    23,     0,    24,     6,    14,     0,    46,    34,
    35,    36,    37,    43,    44,     0,    32,    48,    49,    57,
    58,    59,    60,    61,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,    17,    18,    19,    76,
    76,    76,     0,    70,    76,   111,   112,   113,   114,   115,
     7,    12,     0,    42,    45,    38,    39,    40,    41,    30,
    47,    50,    51,    52,     0,     0,    66,    80,     0,    78,
    83,    83,    83,     0,     0,    83,    74,    75,     0,    72,
    81,     0,    61,     0,   108,     0,    83,     6,     0,     0,
     0,     0,     0,     0,     0,    26,    27,    28,    31,     0,
    25,    53,    54,     0,     0,     0,    66,    68,    77,     0,
     0,     0,     0,     0,     0,    92,     0,    71,     0,    62,
   100,     0,   106,   102,     0,   110,     0,    15,    83,    83,
    83,    83,     0,    83,    69,     0,    33,    55,     0,     0,
    67,    79,    85,    84,   116,   118,    87,    89,    90,    91,
    93,    73,    63,     0,     0,   105,   103,   104,   107,   109,
    88,     0,     0,     0,     0,    99,     0,    29,     0,    65,
     0,   126,     0,     0,     0,   118,     0,    62,    82,   101,
    94,    96,    97,    98,    95,    62,    86,     0,     0,   121,
   122,     6,   119,   120,    64,     0,   130,   131,   132,   133,
     0,   128,    46,    32,     0,     0,   117,    56,   127,     0,
     0,   134,   125,     0,   129,   140,    46,   136,     0,   140,
     0,   138,   137,   138,   141,     0,   123,   124,     0,   139,
     0,     0,     0
};

static const short yydefgoto[] = {   261,
     3,    14,    82,    47,     4,    11,    15,    16,   128,   129,
   130,    91,    48,    49,    50,    51,    52,   134,    53,    54,
   183,   184,   137,   138,   203,   204,    63,   109,   110,    57,
    99,   100,    29,   142,   174,    30,    31,    32,    61,    33,
    76,    77,    78,    79,    80,   186,   153,   115,   116,    34,
    35,   177,   205,   206,   207,   219,   231,   232,   249,   243,
   257,   252
};

static const short yypact[] = {    77,
     0,     0,-32768,    77,-32768,    64,    64,-32768,-32768,    15,
-32768,-32768,-32768,    32,    37,    15,    76,    48,    48,    48,
    53,    60,    58,    48,    15,-32768,    79,   101,   105,-32768,
-32768,-32768,-32768,   118,-32768,   111,-32768,   162,-32768,-32768,
    29,-32768,-32768,-32768,-32768,    55,    73,-32768,-32768,   106,
   108,   110,-32768,-32768,     5,   145,     0,     0,     0,    91,
     0,   109,     0,   125,     0,   124,-32768,-32768,-32768,    48,
    48,    48,    53,    60,    48,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,   121,-32768,-32768,-32768,-32768,-32768,-32768,   147,
-32768,-32768,-32768,-32768,   107,   122,   123,-32768,   126,   128,
   138,   138,   138,     0,   127,   138,-32768,-32768,   142,   130,
   150,     0,     6,     5,   151,   148,   138,   111,     0,     0,
     0,     0,     0,     0,     0,-32768,-32768,   159,-32768,   158,
    49,-32768,-32768,   157,     0,   107,   123,-32768,-32768,   145,
     0,   103,   103,   103,   160,-32768,   103,-32768,   109,   162,
   154,   117,-32768,   155,   125,-32768,   103,-32768,   138,   138,
   138,   138,   129,   138,-32768,   147,-32768,-32768,   170,   173,
-32768,-32768,   167,-32768,-32768,    28,-32768,-32768,-32768,-32768,
-32768,-32768,    19,   180,   162,-32768,-32768,-32768,-32768,-32768,
-32768,   103,   103,   103,   103,-32768,   103,-32768,   176,-32768,
     0,   187,   186,   195,   196,    28,   199,   162,-32768,    49,
-32768,-32768,-32768,-32768,-32768,   162,-32768,     4,   179,-32768,
-32768,   111,-32768,-32768,-32768,   191,-32768,-32768,-32768,-32768,
   188,   200,     0,    16,   106,     0,-32768,-32768,-32768,     4,
   193,   194,-32768,   193,-32768,   192,   197,-32768,   198,   192,
   214,   228,-32768,   228,-32768,   201,-32768,-32768,   218,-32768,
   245,   246,-32768
};

static const short yypgoto[] = {-32768,
   243,-32768,  -107,    -1,   235,   244,     8,-32768,-32768,    84,
-32768,-32768,-32768,   -53,   -36,  -210,-32768,   116,-32768,   -57,
   -14,  -171,-32768,   119,   239,   240,   181,   112,-32768,     7,
   120,-32768,-32768,   -89,    56,-32768,-32768,-32768,   185,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,   113,-32768,   104,-32768,
-32768,  -128,    57,-32768,-32768,-32768,    22,-32768,-32768,  -152,
    10,    18
};


#define	YYLAST		268


static const short yytable[] = {     6,
     7,    83,    55,   101,   102,   103,   113,   106,   236,   111,
   158,   117,   143,   144,   178,   179,   147,   227,   181,    13,
   228,   229,   230,    37,     5,    58,    59,   157,   191,     5,
    65,   248,    66,    17,     1,     2,   225,    17,    18,    19,
    20,    21,    22,  -102,   226,  -102,    95,    23,    96,   114,
   202,   -13,   152,    97,    90,    24,   208,    84,   105,   242,
    95,    85,    96,   211,   212,   213,   214,    25,   215,   192,
   193,   194,   195,    36,   197,   131,   119,   120,   121,     1,
     2,   124,    86,    87,    88,    89,    38,   173,   246,    56,
    95,   250,    96,   133,    60,    39,    64,   113,     9,    10,
     5,    62,   145,    40,    41,    42,    43,    44,    45,    46,
   151,    90,   154,    67,   237,     5,   104,   159,   160,   161,
   162,   163,   164,   165,    70,    71,    72,    73,    74,   107,
   108,     5,   132,   169,   133,    68,   112,   175,   176,    69,
   114,    75,   187,   188,    39,    81,    92,   173,    93,     5,
    94,   131,    40,    41,    42,    43,    44,    45,    46,    98,
   118,   125,   135,   196,   136,   140,    39,   149,   139,   146,
   210,     5,   126,   127,    40,    41,    42,    43,    44,    45,
    46,    39,   235,   141,   148,   150,     5,   156,   155,    40,
    41,    42,    43,    44,    45,    46,   166,   167,   233,   168,
   185,   152,   180,     5,   201,   235,    40,    41,    42,    43,
    44,    45,    46,   247,   199,   200,   209,   234,     5,   216,
   220,    40,    41,    42,    43,    44,    45,    46,   218,   221,
   239,   241,   222,   224,   244,   238,   242,   240,   251,   255,
   256,  -135,   253,   260,   262,   263,     8,   259,    26,   198,
    12,   170,    27,    28,   123,   171,   217,   122,   190,   172,
   182,   245,   223,   258,     0,     0,   189,   254
};

static const short yycheck[] = {     1,
     2,    38,    17,    57,    58,    59,    64,    61,   219,    63,
   118,    65,   102,   103,   143,   144,   106,    14,   147,     5,
    17,    18,    19,    16,    25,    19,    20,   117,   157,    25,
    24,   242,    25,     6,     3,     4,   208,     6,     7,     8,
     9,    10,    11,    38,   216,    40,    42,    16,    44,    64,
    23,    37,    47,    55,    39,    24,    38,    29,    60,    44,
    42,    33,    44,   192,   193,   194,   195,    36,   197,   159,
   160,   161,   162,    37,   164,    90,    70,    71,    72,     3,
     4,    75,    28,    29,    30,    31,    11,   141,   241,    42,
    42,   244,    44,    95,    42,    20,    39,   155,    35,    36,
    25,    42,   104,    28,    29,    30,    31,    32,    33,    34,
   112,    39,   114,    35,   222,    25,    26,   119,   120,   121,
   122,   123,   124,   125,     7,     8,     9,    10,    11,    21,
    22,    25,    26,   135,   136,    35,    12,    35,    36,    35,
   155,    24,    26,    27,    20,    35,    41,   201,    41,    25,
    41,   166,    28,    29,    30,    31,    32,    33,    34,    15,
    37,    41,    41,    35,    42,    38,    20,    38,    43,    43,
   185,    25,    26,    27,    28,    29,    30,    31,    32,    33,
    34,    20,   219,    46,    43,    36,    25,    40,    38,    28,
    29,    30,    31,    32,    33,    34,    38,    40,    20,    43,
    47,    47,    43,    25,    38,   242,    28,    29,    30,    31,
    32,    33,    34,    20,    45,    43,    37,   219,    25,    44,
    35,    28,    29,    30,    31,    32,    33,    34,    42,    35,
    43,   233,    37,    35,   236,    45,    44,    38,    47,    26,
    13,    45,    45,    26,     0,     0,     4,    47,    14,   166,
     7,   136,    14,    14,    74,   137,   201,    73,   155,   140,
   149,   240,   206,   254,    -1,    -1,   154,   250
};
/* -*-C-*-  Note some compilers choke on comments on `#line' lines.  */
#line 3 "/usr/dcs/software/supported/encap/bison-1.28/share/bison.simple"
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

#line 217 "/usr/dcs/software/supported/encap/bison-1.28/share/bison.simple"

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
{ yyval.module = new Module(lineno, yyvsp[-1].strval, yyvsp[0].conslist); ;
    break;}
case 10:
#line 121 "xi-grammar.y"
{ yyval.module = new Module(lineno, yyvsp[-1].strval, yyvsp[0].conslist); yyval.module->setMain(); ;
    break;}
case 11:
#line 125 "xi-grammar.y"
{ yyval.conslist = 0; ;
    break;}
case 12:
#line 127 "xi-grammar.y"
{ yyval.conslist = yyvsp[-2].conslist; ;
    break;}
case 13:
#line 131 "xi-grammar.y"
{ yyval.conslist = 0; ;
    break;}
case 14:
#line 133 "xi-grammar.y"
{ yyval.conslist = new ConstructList(lineno, yyvsp[-1].construct, yyvsp[0].conslist); ;
    break;}
case 15:
#line 137 "xi-grammar.y"
{ if(yyvsp[-2].conslist) yyvsp[-2].conslist->setExtern(yyvsp[-4].intval); yyval.construct = yyvsp[-2].conslist; ;
    break;}
case 16:
#line 139 "xi-grammar.y"
{ yyvsp[0].module->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].module; ;
    break;}
case 17:
#line 141 "xi-grammar.y"
{ yyvsp[-1].readonly->setExtern(yyvsp[-2].intval); yyval.construct = yyvsp[-1].readonly; ;
    break;}
case 18:
#line 143 "xi-grammar.y"
{ yyvsp[-1].readonly->setExtern(yyvsp[-2].intval); yyval.construct = yyvsp[-1].readonly; ;
    break;}
case 19:
#line 145 "xi-grammar.y"
{ yyvsp[-1].message->setExtern(yyvsp[-2].intval); yyval.construct = yyvsp[-1].message; ;
    break;}
case 20:
#line 147 "xi-grammar.y"
{ yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; ;
    break;}
case 21:
#line 149 "xi-grammar.y"
{ yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; ;
    break;}
case 22:
#line 151 "xi-grammar.y"
{ yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; ;
    break;}
case 23:
#line 153 "xi-grammar.y"
{ yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; ;
    break;}
case 24:
#line 155 "xi-grammar.y"
{ yyvsp[0].templat->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].templat; ;
    break;}
case 25:
#line 159 "xi-grammar.y"
{ yyval.tparam = new TParamType(yyvsp[0].type); ;
    break;}
case 26:
#line 161 "xi-grammar.y"
{ yyval.tparam = new TParamVal(yyvsp[0].strval); ;
    break;}
case 27:
#line 163 "xi-grammar.y"
{ yyval.tparam = new TParamVal(yyvsp[0].strval); ;
    break;}
case 28:
#line 167 "xi-grammar.y"
{ yyval.tparlist = new TParamList(yyvsp[0].tparam); ;
    break;}
case 29:
#line 169 "xi-grammar.y"
{ yyval.tparlist = new TParamList(yyvsp[-2].tparam, yyvsp[0].tparlist); ;
    break;}
case 30:
#line 173 "xi-grammar.y"
{ yyval.tparlist = 0; ;
    break;}
case 31:
#line 175 "xi-grammar.y"
{ yyval.tparlist = yyvsp[0].tparlist; ;
    break;}
case 32:
#line 179 "xi-grammar.y"
{ yyval.tparlist = 0; ;
    break;}
case 33:
#line 181 "xi-grammar.y"
{ yyval.tparlist = yyvsp[-1].tparlist; ;
    break;}
case 34:
#line 185 "xi-grammar.y"
{ yyval.type = new BuiltinType("int"); ;
    break;}
case 35:
#line 187 "xi-grammar.y"
{ yyval.type = new BuiltinType("long"); ;
    break;}
case 36:
#line 189 "xi-grammar.y"
{ yyval.type = new BuiltinType("short"); ;
    break;}
case 37:
#line 191 "xi-grammar.y"
{ yyval.type = new BuiltinType("char"); ;
    break;}
case 38:
#line 193 "xi-grammar.y"
{ yyval.type = new BuiltinType("unsigned int"); ;
    break;}
case 39:
#line 195 "xi-grammar.y"
{ yyval.type = new BuiltinType("long"); ;
    break;}
case 40:
#line 197 "xi-grammar.y"
{ yyval.type = new BuiltinType("short"); ;
    break;}
case 41:
#line 199 "xi-grammar.y"
{ yyval.type = new BuiltinType("char"); ;
    break;}
case 42:
#line 201 "xi-grammar.y"
{ yyval.type = new BuiltinType("long long"); ;
    break;}
case 43:
#line 203 "xi-grammar.y"
{ yyval.type = new BuiltinType("float"); ;
    break;}
case 44:
#line 205 "xi-grammar.y"
{ yyval.type = new BuiltinType("double"); ;
    break;}
case 45:
#line 207 "xi-grammar.y"
{ yyval.type = new BuiltinType("long double"); ;
    break;}
case 46:
#line 209 "xi-grammar.y"
{ yyval.type = new BuiltinType("void"); ;
    break;}
case 47:
#line 213 "xi-grammar.y"
{ yyval.ntype = new NamedType(yyvsp[-1].strval, yyvsp[0].tparlist); ;
    break;}
case 48:
#line 217 "xi-grammar.y"
{ yyval.type = yyvsp[0].type; ;
    break;}
case 49:
#line 219 "xi-grammar.y"
{ yyval.type = yyvsp[0].ntype; ;
    break;}
case 50:
#line 223 "xi-grammar.y"
{ yyval.ptype = new PtrType(yyvsp[-1].type); ;
    break;}
case 51:
#line 227 "xi-grammar.y"
{ yyvsp[-1].ptype->indirect(); yyval.ptype = yyvsp[-1].ptype; ;
    break;}
case 52:
#line 229 "xi-grammar.y"
{ yyvsp[-1].ptype->indirect(); yyval.ptype = yyvsp[-1].ptype; ;
    break;}
case 53:
#line 233 "xi-grammar.y"
{ yyval.val = new Value(yyvsp[0].strval); ;
    break;}
case 54:
#line 235 "xi-grammar.y"
{ yyval.val = new Value(yyvsp[0].strval); ;
    break;}
case 55:
#line 239 "xi-grammar.y"
{ yyval.type = new ArrayType(yyvsp[-3].type, yyvsp[-1].val); ;
    break;}
case 56:
#line 243 "xi-grammar.y"
{ yyval.ftype = new FuncType(yyvsp[-7].type, yyvsp[-4].strval, yyvsp[-1].typelist); ;
    break;}
case 57:
#line 247 "xi-grammar.y"
{ yyval.type = yyvsp[0].type; ;
    break;}
case 58:
#line 249 "xi-grammar.y"
{ yyval.type = (Type*) yyvsp[0].ptype; ;
    break;}
case 59:
#line 251 "xi-grammar.y"
{ yyval.type = (Type*) yyvsp[0].ptype; ;
    break;}
case 60:
#line 253 "xi-grammar.y"
{ yyval.type = yyvsp[0].type; ;
    break;}
case 61:
#line 255 "xi-grammar.y"
{ yyval.type = yyvsp[0].ftype; ;
    break;}
case 62:
#line 259 "xi-grammar.y"
{ yyval.typelist = 0; ;
    break;}
case 63:
#line 261 "xi-grammar.y"
{ yyval.typelist = new TypeList(yyvsp[0].type); ;
    break;}
case 64:
#line 263 "xi-grammar.y"
{ yyval.typelist = new TypeList(yyvsp[-2].type, yyvsp[0].typelist); ;
    break;}
case 65:
#line 267 "xi-grammar.y"
{ yyval.val = yyvsp[-1].val; ;
    break;}
case 66:
#line 271 "xi-grammar.y"
{ yyval.vallist = 0; ;
    break;}
case 67:
#line 273 "xi-grammar.y"
{ yyval.vallist = new ValueList(yyvsp[-1].val, yyvsp[0].vallist); ;
    break;}
case 68:
#line 277 "xi-grammar.y"
{ yyval.readonly = new Readonly(lineno, yyvsp[-2].type, yyvsp[-1].strval, yyvsp[0].vallist); ;
    break;}
case 69:
#line 281 "xi-grammar.y"
{ yyval.readonly = new Readonly(lineno, yyvsp[-2].type, yyvsp[0].strval, 0, 1); ;
    break;}
case 70:
#line 285 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 71:
#line 287 "xi-grammar.y"
{ yyval.intval = yyvsp[-1].intval; ;
    break;}
case 72:
#line 291 "xi-grammar.y"
{ yyval.intval = yyvsp[0].intval; ;
    break;}
case 73:
#line 293 "xi-grammar.y"
{ yyval.intval = yyvsp[-2].intval | yyvsp[0].intval; ;
    break;}
case 74:
#line 297 "xi-grammar.y"
{ yyval.intval = SPACKED; ;
    break;}
case 75:
#line 299 "xi-grammar.y"
{ yyval.intval = SVARSIZE; ;
    break;}
case 76:
#line 303 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 77:
#line 305 "xi-grammar.y"
{ yyval.intval = yyvsp[-1].intval; ;
    break;}
case 78:
#line 309 "xi-grammar.y"
{ yyval.intval = yyvsp[0].intval; ;
    break;}
case 79:
#line 311 "xi-grammar.y"
{ yyval.intval = yyvsp[-2].intval | yyvsp[0].intval; ;
    break;}
case 80:
#line 315 "xi-grammar.y"
{ yyval.intval = 0x01; ;
    break;}
case 81:
#line 319 "xi-grammar.y"
{ yyval.message = new Message(lineno, yyvsp[0].ntype, yyvsp[-1].intval); ;
    break;}
case 82:
#line 321 "xi-grammar.y"
{ yyval.message = new Message(lineno, yyvsp[-3].ntype, yyvsp[-4].intval, yyvsp[-1].typelist); ;
    break;}
case 83:
#line 325 "xi-grammar.y"
{ yyval.typelist = 0; ;
    break;}
case 84:
#line 327 "xi-grammar.y"
{ yyval.typelist = yyvsp[0].typelist; ;
    break;}
case 85:
#line 331 "xi-grammar.y"
{ yyval.typelist = new TypeList(yyvsp[0].ntype); ;
    break;}
case 86:
#line 333 "xi-grammar.y"
{ yyval.typelist = new TypeList(yyvsp[-2].ntype, yyvsp[0].typelist); ;
    break;}
case 87:
#line 337 "xi-grammar.y"
{ yyval.chare = new Chare(lineno, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist, yyvsp[-3].intval); ;
    break;}
case 88:
#line 339 "xi-grammar.y"
{ yyval.chare = new MainChare(lineno, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist, yyvsp[-3].intval); ;
    break;}
case 89:
#line 343 "xi-grammar.y"
{ yyval.chare = new Group(lineno, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist, yyvsp[-3].intval); ;
    break;}
case 90:
#line 347 "xi-grammar.y"
{ yyval.chare = new NodeGroup(lineno, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist, yyvsp[-3].intval); ;
    break;}
case 91:
#line 351 "xi-grammar.y"
{/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",yyvsp[-2].strval);
			yyval.ntype = new NamedType(buf); 
		;
    break;}
case 92:
#line 357 "xi-grammar.y"
{ yyval.ntype = new NamedType(yyvsp[-1].strval); ;
    break;}
case 93:
#line 361 "xi-grammar.y"
{ yyval.chare = new Array(lineno, yyvsp[-3].ntype, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 94:
#line 365 "xi-grammar.y"
{ yyval.chare = new Chare(lineno, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist, yyvsp[-3].intval);;
    break;}
case 95:
#line 367 "xi-grammar.y"
{ yyval.chare = new MainChare(lineno, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist, yyvsp[-3].intval); ;
    break;}
case 96:
#line 371 "xi-grammar.y"
{ yyval.chare = new Group(lineno, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist, yyvsp[-3].intval); ;
    break;}
case 97:
#line 375 "xi-grammar.y"
{ yyval.chare = new NodeGroup( lineno, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist, yyvsp[-3].intval); ;
    break;}
case 98:
#line 379 "xi-grammar.y"
{ yyval.chare = new Array( lineno, yyvsp[-3].ntype, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 99:
#line 383 "xi-grammar.y"
{ yyval.message = new Message(lineno, new NamedType(yyvsp[-1].strval), yyvsp[-2].intval); ;
    break;}
case 100:
#line 387 "xi-grammar.y"
{ yyval.type = 0; ;
    break;}
case 101:
#line 389 "xi-grammar.y"
{ yyval.type = yyvsp[0].type; ;
    break;}
case 102:
#line 393 "xi-grammar.y"
{ yyval.strval = 0; ;
    break;}
case 103:
#line 395 "xi-grammar.y"
{ yyval.strval = yyvsp[0].strval; ;
    break;}
case 104:
#line 397 "xi-grammar.y"
{ yyval.strval = yyvsp[0].strval; ;
    break;}
case 105:
#line 401 "xi-grammar.y"
{ yyval.tvar = new TType(new NamedType(yyvsp[-1].strval), yyvsp[0].type); ;
    break;}
case 106:
#line 403 "xi-grammar.y"
{ yyval.tvar = new TFunc(yyvsp[-1].ftype, yyvsp[0].strval); ;
    break;}
case 107:
#line 405 "xi-grammar.y"
{ yyval.tvar = new TName(yyvsp[-2].type, yyvsp[-1].strval, yyvsp[0].strval); ;
    break;}
case 108:
#line 409 "xi-grammar.y"
{ yyval.tvarlist = new TVarList(yyvsp[0].tvar); ;
    break;}
case 109:
#line 411 "xi-grammar.y"
{ yyval.tvarlist = new TVarList(yyvsp[-2].tvar, yyvsp[0].tvarlist); ;
    break;}
case 110:
#line 415 "xi-grammar.y"
{ yyval.tvarlist = yyvsp[-1].tvarlist; ;
    break;}
case 111:
#line 419 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); ;
    break;}
case 112:
#line 421 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); ;
    break;}
case 113:
#line 423 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); ;
    break;}
case 114:
#line 425 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); ;
    break;}
case 115:
#line 427 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].message); yyvsp[0].message->setTemplate(yyval.templat); ;
    break;}
case 116:
#line 431 "xi-grammar.y"
{ yyval.mbrlist = 0; ;
    break;}
case 117:
#line 433 "xi-grammar.y"
{ yyval.mbrlist = yyvsp[-2].mbrlist; ;
    break;}
case 118:
#line 437 "xi-grammar.y"
{ yyval.mbrlist = 0; ;
    break;}
case 119:
#line 439 "xi-grammar.y"
{ yyval.mbrlist = new MemberList(yyvsp[-1].member, yyvsp[0].mbrlist); ;
    break;}
case 120:
#line 443 "xi-grammar.y"
{ yyval.member = yyvsp[-1].entry; ;
    break;}
case 121:
#line 445 "xi-grammar.y"
{ yyval.member = yyvsp[-1].readonly; ;
    break;}
case 122:
#line 447 "xi-grammar.y"
{ yyval.member = yyvsp[-1].readonly; ;
    break;}
case 123:
#line 451 "xi-grammar.y"
{ yyval.entry = new Entry(lineno, yyvsp[-5].intval|yyvsp[-1].intval, new BuiltinType("void"), yyvsp[-3].strval, yyvsp[-2].rtype, yyvsp[0].val); ;
    break;}
case 124:
#line 453 "xi-grammar.y"
{ yyval.entry = new Entry(lineno, yyvsp[-5].intval|yyvsp[-1].intval, yyvsp[-4].ptype, yyvsp[-3].strval, yyvsp[-2].rtype, yyvsp[0].val); ;
    break;}
case 125:
#line 455 "xi-grammar.y"
{ yyval.entry = new Entry(lineno, yyvsp[-2].intval, 0, yyvsp[-1].strval, yyvsp[0].rtype, 0); ;
    break;}
case 126:
#line 459 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 127:
#line 461 "xi-grammar.y"
{ yyval.intval = yyvsp[-1].intval; ;
    break;}
case 128:
#line 465 "xi-grammar.y"
{ yyval.intval = yyvsp[0].intval; ;
    break;}
case 129:
#line 467 "xi-grammar.y"
{ yyval.intval = yyvsp[-2].intval | yyvsp[0].intval; ;
    break;}
case 130:
#line 471 "xi-grammar.y"
{ yyval.intval = STHREADED; ;
    break;}
case 131:
#line 473 "xi-grammar.y"
{ yyval.intval = SSYNC; ;
    break;}
case 132:
#line 475 "xi-grammar.y"
{ yyval.intval = SLOCKED; ;
    break;}
case 133:
#line 477 "xi-grammar.y"
{ yyval.intval = SVIRTUAL; ;
    break;}
case 134:
#line 481 "xi-grammar.y"
{ yyval.rtype = 0; ;
    break;}
case 135:
#line 483 "xi-grammar.y"
{ yyval.rtype = new BuiltinType("void"); ;
    break;}
case 136:
#line 485 "xi-grammar.y"
{ yyval.rtype = yyvsp[0].ptype; ;
    break;}
case 137:
#line 489 "xi-grammar.y"
{ yyval.rtype = yyvsp[-1].rtype; ;
    break;}
case 138:
#line 493 "xi-grammar.y"
{ yyval.val = 0; ;
    break;}
case 139:
#line 495 "xi-grammar.y"
{ yyval.val = new Value(yyvsp[0].strval); ;
    break;}
case 140:
#line 499 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 141:
#line 501 "xi-grammar.y"
{ if(strcmp(yyvsp[0].strval, "0")) { yyerror("expected 0"); exit(1); }
		  yyval.intval = SPURE; 
		;
    break;}
}
   /* the action file gets copied in in place of this dollarsign */
#line 543 "/usr/dcs/software/supported/encap/bison-1.28/share/bison.simple"

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
#line 505 "xi-grammar.y"

void yyerror(const char *mesg)
{
  cout << "Syntax error at line " << lineno << ": " << mesg << endl;
  // return 0;
}
