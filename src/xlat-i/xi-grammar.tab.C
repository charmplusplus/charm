
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
#define	VIRTUAL	272
#define	VOID	273
#define	PACKED	274
#define	VARSIZE	275
#define	ENTRY	276
#define	MAINCHARE	277
#define	IDENT	278
#define	NUMBER	279
#define	LITERAL	280
#define	INT	281
#define	LONG	282
#define	SHORT	283
#define	CHAR	284
#define	FLOAT	285
#define	DOUBLE	286
#define	UNSIGNED	287

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



#define	YYFINAL		248
#define	YYFLAG		-32768
#define	YYNTBASE	47

#define YYTRANSLATE(x) ((unsigned)(x) <= 287 ? yytranslate[x] : 107)

static const char yytranslate[] = {     0,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,    43,
    44,    40,     2,    37,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,    45,    34,    38,
    46,    39,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
    41,     2,    42,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,    35,     2,    36,     2,     2,     2,     2,     2,
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
    27,    28,    29,    30,    31,    32,    33
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
   192,   196,   198,   202,   204,   206,   210,   217,   218,   221,
   223,   227,   232,   237,   242,   247,   252,   256,   262,   267,
   272,   277,   282,   288,   293,   294,   297,   298,   301,   304,
   308,   311,   315,   317,   321,   326,   329,   332,   335,   338,
   341,   343,   348,   349,   352,   355,   358,   361,   369,   377,
   382,   383,   387,   389,   393,   395,   397,   399,   401,   402,
   404,   406,   410,   411,   415,   416
};

static const short yyrhs[] = {    48,
     0,     0,    52,    48,     0,     0,     5,     0,     0,    34,
     0,    24,     0,     3,    51,    53,     0,     4,    51,    53,
     0,    34,     0,    35,    54,    36,    50,     0,     0,    55,
    54,     0,    49,    35,    54,    36,    50,     0,    49,    52,
     0,    49,    72,    34,     0,    49,    73,    34,     0,    49,
    77,    34,     0,    49,    80,     0,    49,    81,     0,    49,
    82,     0,    49,    84,     0,    49,    95,     0,    68,     0,
    25,     0,    26,     0,    56,     0,    56,    37,    57,     0,
     0,    57,     0,     0,    38,    58,    39,     0,    27,     0,
    28,     0,    29,     0,    30,     0,    33,    27,     0,    33,
    28,     0,    33,    29,     0,    33,    30,     0,    28,    28,
     0,    31,     0,    32,     0,    28,    32,     0,    19,     0,
    51,    59,     0,    60,     0,    61,     0,    62,    40,     0,
    63,    40,     0,    64,    40,     0,    25,     0,    51,     0,
    68,    41,    65,    42,     0,    68,    43,    40,    51,    44,
    43,    69,    44,     0,    62,     0,    63,     0,    64,     0,
    66,     0,    67,     0,     0,    68,     0,    68,    37,    69,
     0,    41,    65,    42,     0,     0,    70,    71,     0,     6,
    68,    51,    71,     0,     6,    11,    62,    40,    51,     0,
     0,    41,    75,    42,     0,    76,     0,    76,    37,    75,
     0,    20,     0,    21,     0,    11,    74,    61,     0,    11,
    74,    61,    35,    69,    36,     0,     0,    45,    79,     0,
    61,     0,    61,    37,    79,     0,     7,    61,    78,    96,
     0,    23,    61,    78,    96,     0,     8,    61,    78,    96,
     0,     9,    61,    78,    96,     0,    41,    25,    51,    42,
     0,    41,    51,    42,     0,    10,    83,    61,    78,    96,
     0,     7,    51,    78,    96,     0,    23,    51,    78,    96,
     0,     8,    51,    78,    96,     0,     9,    51,    78,    96,
     0,    10,    83,    51,    78,    96,     0,    11,    74,    51,
    34,     0,     0,    46,    68,     0,     0,    46,    25,     0,
    46,    26,     0,    12,    51,    90,     0,    67,    91,     0,
    68,    51,    91,     0,    92,     0,    92,    37,    93,     0,
    15,    38,    93,    39,     0,    94,    85,     0,    94,    86,
     0,    94,    87,     0,    94,    88,     0,    94,    89,     0,
    34,     0,    35,    97,    36,    50,     0,     0,    98,    97,
     0,    99,    34,     0,    72,    34,     0,    73,    34,     0,
    22,   100,    19,    51,   104,   106,   105,     0,    22,   100,
    63,    51,   104,   106,   105,     0,    22,   100,    51,   104,
     0,     0,    41,   101,    42,     0,   102,     0,   102,    37,
   101,     0,    14,     0,    16,     0,    17,     0,    18,     0,
     0,    19,     0,    63,     0,    43,   103,    44,     0,     0,
    13,    46,    25,     0,     0,    46,    25,     0
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
   284,   288,   290,   294,   296,   300,   302,   306,   308,   312,
   314,   318,   320,   324,   328,   332,   338,   342,   346,   348,
   352,   356,   360,   364,   368,   370,   374,   376,   378,   382,
   384,   386,   390,   392,   396,   400,   402,   404,   406,   408,
   412,   414,   418,   420,   424,   426,   428,   432,   434,   436,
   440,   442,   446,   448,   452,   454,   456,   458,   462,   464,
   466,   470,   474,   476,   480,   482
};
#endif


#if YYDEBUG != 0 || defined (YYERROR_VERBOSE)

static const char * const yytname[] = {   "$","error","$undefined.","MODULE",
"MAINMODULE","EXTERN","READONLY","CHARE","GROUP","NODEGROUP","ARRAY","MESSAGE",
"CLASS","STACKSIZE","THREADED","TEMPLATE","SYNC","EXCLUSIVE","VIRTUAL","VOID",
"PACKED","VARSIZE","ENTRY","MAINCHARE","IDENT","NUMBER","LITERAL","INT","LONG",
"SHORT","CHAR","FLOAT","DOUBLE","UNSIGNED","';'","'{'","'}'","','","'<'","'>'",
"'*'","'['","']'","'('","')'","':'","'='","File","ModuleEList","OptExtern","OptSemiColon",
"Name","Module","ConstructEList","ConstructList","Construct","TParam","TParamList",
"TParamEList","OptTParams","BuiltinType","NamedType","SimpleType","OnePtrType",
"PtrType","ArrayDim","ArrayType","FuncType","Type","TypeList","Dim","DimList",
"Readonly","ReadonlyMsg","MAttribs","MAttribList","MAttrib","Message","OptBaseList",
"BaseList","Chare","Group","NodeGroup","ArrayIndexType","Array","TChare","TGroup",
"TNodeGroup","TArray","TMessage","OptTypeInit","OptNameInit","TVar","TVarList",
"TemplateSpec","Template","MemberEList","MemberList","Member","Entry","EAttribs",
"EAttribList","EAttrib","OptType","EParam","OptStackSize","OptPure", NULL
};
#endif

static const short yyr1[] = {     0,
    47,    48,    48,    49,    49,    50,    50,    51,    52,    52,
    53,    53,    54,    54,    55,    55,    55,    55,    55,    55,
    55,    55,    55,    55,    56,    56,    56,    57,    57,    58,
    58,    59,    59,    60,    60,    60,    60,    60,    60,    60,
    60,    60,    60,    60,    60,    60,    61,    62,    62,    63,
    64,    64,    65,    65,    66,    67,    68,    68,    68,    68,
    68,    69,    69,    69,    70,    71,    71,    72,    73,    74,
    74,    75,    75,    76,    76,    77,    77,    78,    78,    79,
    79,    80,    80,    81,    82,    83,    83,    84,    85,    85,
    86,    87,    88,    89,    90,    90,    91,    91,    91,    92,
    92,    92,    93,    93,    94,    95,    95,    95,    95,    95,
    96,    96,    97,    97,    98,    98,    98,    99,    99,    99,
   100,   100,   101,   101,   102,   102,   102,   102,   103,   103,
   103,   104,   105,   105,   106,   106
};

static const short yyr2[] = {     0,
     1,     0,     2,     0,     1,     0,     1,     1,     3,     3,
     1,     4,     0,     2,     5,     2,     3,     3,     3,     2,
     2,     2,     2,     2,     1,     1,     1,     1,     3,     0,
     1,     0,     3,     1,     1,     1,     1,     2,     2,     2,
     2,     2,     1,     1,     2,     1,     2,     1,     1,     2,
     2,     2,     1,     1,     4,     8,     1,     1,     1,     1,
     1,     0,     1,     3,     3,     0,     2,     4,     5,     0,
     3,     1,     3,     1,     1,     3,     6,     0,     2,     1,
     3,     4,     4,     4,     4,     4,     3,     5,     4,     4,
     4,     4,     5,     4,     0,     2,     0,     2,     2,     3,
     2,     3,     1,     3,     4,     2,     2,     2,     2,     2,
     1,     4,     0,     2,     2,     2,     2,     7,     7,     4,
     0,     3,     1,     3,     1,     1,     1,     1,     0,     1,
     1,     3,     0,     3,     0,     2
};

static const short yydefact[] = {     2,
     0,     0,     1,     2,     8,     0,     0,     3,    11,     4,
     9,    10,     5,     0,     0,     4,     0,     0,     0,     0,
     0,    70,     0,     0,     4,    16,     0,     0,     0,    20,
    21,    22,    23,     0,    24,     6,    14,     0,    46,    34,
    35,    36,    37,    43,    44,     0,    32,    48,    49,    57,
    58,    59,    60,    61,     0,    78,    78,    78,     0,     0,
     0,     0,     0,    78,     0,    17,    18,    19,     0,     0,
     0,     0,    70,     0,   106,   107,   108,   109,   110,     7,
    12,     0,    42,    45,    38,    39,    40,    41,    30,    47,
    50,    51,    52,     0,     0,    66,     0,     0,     0,     0,
     0,     0,    78,    74,    75,     0,    72,    76,     0,    61,
     0,   103,     0,     0,     6,    78,    78,    78,     0,     0,
    78,     0,    26,    27,    28,    31,     0,    25,    53,    54,
     0,     0,     0,    66,    68,    80,    79,   111,   113,    82,
    84,    85,     0,    87,     0,    71,     0,    62,    95,     0,
   101,    97,     0,   105,    83,    15,     0,     0,     0,    78,
     0,     0,    69,     0,    33,    55,     0,     0,    67,     0,
   121,     0,     0,     0,   113,     0,    86,    88,    73,    63,
     0,     0,   100,    98,    99,   102,   104,    89,    91,    92,
     0,    94,    90,    29,     0,    65,    81,     0,     0,   116,
   117,     6,   114,   115,    62,    77,    96,    93,    62,   125,
   126,   127,   128,     0,   123,    46,    32,     0,     0,   112,
    64,     0,   122,     0,     0,   129,   120,     0,    56,   124,
   135,    46,   131,     0,   135,     0,   133,   132,   133,   136,
     0,   118,   119,     0,   134,     0,     0,     0
};

static const short yydefgoto[] = {   246,
     3,    14,    81,    47,     4,    11,    15,    16,   125,   126,
   127,    90,    48,    49,    50,    51,    52,   131,    53,    54,
   180,   181,   134,   135,   172,   173,    62,   106,   107,    29,
    98,   137,    30,    31,    32,    60,    33,    75,    76,    77,
    78,    79,   183,   151,   112,   113,    34,    35,   140,   174,
   175,   176,   199,   214,   215,   234,   227,   242,   237
};

static const short yypact[] = {    63,
   -19,   -19,-32768,    63,-32768,    45,    45,-32768,-32768,     4,
-32768,-32768,-32768,    24,   -13,     4,   114,   -19,   -19,   -19,
   -26,    -3,     8,   -19,     4,-32768,    28,    38,    58,-32768,
-32768,-32768,-32768,    79,-32768,    88,-32768,   192,-32768,-32768,
    32,-32768,-32768,-32768,-32768,    76,    75,-32768,-32768,    83,
    84,    86,-32768,-32768,     1,    82,    82,    82,    59,   -19,
    77,   -19,   142,    82,    93,-32768,-32768,-32768,   -19,   -19,
   -19,   -26,    -3,   -19,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,    90,-32768,-32768,-32768,-32768,-32768,-32768,   175,-32768,
-32768,-32768,-32768,    87,    94,    95,   -19,    80,    80,    80,
   -19,    98,    82,-32768,-32768,   106,   100,   116,   -19,     6,
     1,   112,    96,    80,    88,    82,    82,    82,   -19,   -19,
    82,   -19,-32768,-32768,   115,-32768,   117,    66,-32768,-32768,
   111,   -19,    87,    95,-32768,   118,-32768,-32768,     7,-32768,
-32768,-32768,   120,-32768,    80,-32768,    77,   192,   113,    91,
-32768,   119,   142,-32768,-32768,-32768,    80,    80,    80,    82,
   123,    80,-32768,   175,-32768,-32768,   132,   122,-32768,   -19,
   126,   143,   144,   124,     7,   145,-32768,-32768,-32768,   -21,
   146,   192,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
    80,-32768,-32768,-32768,   137,-32768,-32768,    39,   207,-32768,
-32768,    88,-32768,-32768,   192,-32768,    66,-32768,   192,-32768,
-32768,-32768,-32768,   139,   147,   -19,   -17,    83,   -19,-32768,
-32768,   141,-32768,    39,   140,   222,-32768,   140,-32768,-32768,
   149,   148,-32768,   152,   149,   161,   174,-32768,   174,-32768,
   151,-32768,-32768,   163,-32768,   189,   191,-32768
};

static const short yypgoto[] = {-32768,
   205,-32768,  -111,    -1,   179,   203,    -6,-32768,-32768,    48,
-32768,-32768,-32768,   -12,   -36,  -185,-32768,    81,-32768,   -52,
   -14,  -144,-32768,    99,   199,   213,   155,    70,-32768,-32768,
   -40,    60,-32768,-32768,-32768,   157,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,    92,-32768,    89,-32768,-32768,   -63,    57,
-32768,-32768,-32768,    19,-32768,-32768,  -154,     9,    10
};


#define	YYLAST		255


static const short yytable[] = {     6,
     7,    82,    55,   156,     5,    56,    57,    58,    13,    37,
   110,    64,    17,   219,    59,   205,    99,   100,    65,    94,
    89,    95,    36,   114,     5,   226,     1,     2,   171,    17,
    18,    19,    20,    21,    22,   141,   142,    61,    23,   -13,
   233,    94,   -97,    95,   -97,    63,    24,   103,   111,   108,
   155,   150,   210,    96,   211,   212,   213,   102,    25,    83,
   221,    66,   145,    84,   222,     1,     2,   116,   117,   118,
   231,    67,   121,   235,   128,   157,   158,   159,     9,    10,
   162,   178,     5,   101,   136,    69,    70,    71,    72,    73,
   220,    68,   130,   188,   189,   190,   104,   105,   193,   143,
   110,    74,    85,    86,    87,    88,    94,   149,    95,   152,
     5,   129,    89,   138,   139,   184,   185,   160,   161,   191,
   163,    80,    91,    92,    38,    93,    97,   208,   115,   122,
   167,   130,    39,   132,   154,   133,   147,     5,   111,   144,
    40,    41,    42,    43,    44,    45,    46,   146,   153,   128,
   148,   164,   166,   109,   170,   165,   192,   136,   182,   202,
    39,   177,   218,   196,   150,     5,   198,   207,    40,    41,
    42,    43,    44,    45,    46,   195,   200,   201,   204,   209,
   223,   206,   226,   224,   229,   240,   241,   245,   247,   218,
   248,  -130,    26,    39,   236,   238,   244,   217,     5,   123,
   124,    40,    41,    42,    43,    44,    45,    46,     8,    12,
    39,   194,    27,   168,   225,     5,   179,   228,    40,    41,
    42,    43,    44,    45,    46,   216,    28,   120,   119,   197,
     5,   203,   169,    40,    41,    42,    43,    44,    45,    46,
   232,   187,   230,   186,   239,     5,     0,   243,    40,    41,
    42,    43,    44,    45,    46
};

static const short yycheck[] = {     1,
     2,    38,    17,   115,    24,    18,    19,    20,     5,    16,
    63,    24,     6,   199,    41,    37,    57,    58,    25,    41,
    38,    43,    36,    64,    24,    43,     3,     4,    22,     6,
     7,     8,     9,    10,    11,    99,   100,    41,    15,    36,
   226,    41,    37,    43,    39,    38,    23,    60,    63,    62,
   114,    46,    14,    55,    16,    17,    18,    59,    35,    28,
   205,    34,   103,    32,   209,     3,     4,    69,    70,    71,
   225,    34,    74,   228,    89,   116,   117,   118,    34,    35,
   121,   145,    24,    25,    97,     7,     8,     9,    10,    11,
   202,    34,    94,   157,   158,   159,    20,    21,   162,   101,
   153,    23,    27,    28,    29,    30,    41,   109,    43,   111,
    24,    25,    38,    34,    35,    25,    26,   119,   120,   160,
   122,    34,    40,    40,    11,    40,    45,   191,    36,    40,
   132,   133,    19,    40,    39,    41,    37,    24,   153,    42,
    27,    28,    29,    30,    31,    32,    33,    42,    37,   164,
    35,    37,    42,    12,    37,    39,    34,   170,    46,    36,
    19,    42,   199,    42,    46,    24,    41,   182,    27,    28,
    29,    30,    31,    32,    33,    44,    34,    34,    34,    43,
    42,    36,    43,    37,    44,    25,    13,    25,     0,   226,
     0,    44,    14,    19,    46,    44,    46,   199,    24,    25,
    26,    27,    28,    29,    30,    31,    32,    33,     4,     7,
    19,   164,    14,   133,   216,    24,   147,   219,    27,    28,
    29,    30,    31,    32,    33,    19,    14,    73,    72,   170,
    24,   175,   134,    27,    28,    29,    30,    31,    32,    33,
    19,   153,   224,   152,   235,    24,    -1,   239,    27,    28,
    29,    30,    31,    32,    33
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
#line 91 "xi-grammar.y"
{ yyval.modlist = yyvsp[0].modlist; modlist = yyvsp[0].modlist; ;
    break;}
case 2:
#line 95 "xi-grammar.y"
{ yyval.modlist = 0; ;
    break;}
case 3:
#line 97 "xi-grammar.y"
{ yyval.modlist = new ModuleList(lineno, yyvsp[-1].module, yyvsp[0].modlist); ;
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
{ yyval.module = new Module(lineno, yyvsp[-1].strval, yyvsp[0].conslist); ;
    break;}
case 10:
#line 119 "xi-grammar.y"
{ yyval.module = new Module(lineno, yyvsp[-1].strval, yyvsp[0].conslist); yyval.module->setMain(); ;
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
{ yyval.conslist = new ConstructList(lineno, yyvsp[-1].construct, yyvsp[0].conslist); ;
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
{ yyval.readonly = new Readonly(lineno, yyvsp[-2].type, yyvsp[-1].strval, yyvsp[0].vallist); ;
    break;}
case 69:
#line 279 "xi-grammar.y"
{ yyval.readonly = new Readonly(lineno, yyvsp[-2].type, yyvsp[0].strval, 0, 1); ;
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
{ yyval.message = new Message(lineno, yyvsp[0].ntype, yyvsp[-1].intval); ;
    break;}
case 77:
#line 303 "xi-grammar.y"
{ yyval.message = new Message(lineno, yyvsp[-3].ntype, yyvsp[-4].intval, yyvsp[-1].typelist); ;
    break;}
case 78:
#line 307 "xi-grammar.y"
{ yyval.typelist = 0; ;
    break;}
case 79:
#line 309 "xi-grammar.y"
{ yyval.typelist = yyvsp[0].typelist; ;
    break;}
case 80:
#line 313 "xi-grammar.y"
{ yyval.typelist = new TypeList(yyvsp[0].ntype); ;
    break;}
case 81:
#line 315 "xi-grammar.y"
{ yyval.typelist = new TypeList(yyvsp[-2].ntype, yyvsp[0].typelist); ;
    break;}
case 82:
#line 319 "xi-grammar.y"
{ yyval.chare = new Chare(lineno, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 83:
#line 321 "xi-grammar.y"
{ yyval.chare = new MainChare(lineno, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 84:
#line 325 "xi-grammar.y"
{ yyval.chare = new Group(lineno, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 85:
#line 329 "xi-grammar.y"
{ yyval.chare = new NodeGroup(lineno, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 86:
#line 333 "xi-grammar.y"
{/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",yyvsp[-2].strval);
			yyval.ntype = new NamedType(buf); 
		;
    break;}
case 87:
#line 339 "xi-grammar.y"
{ yyval.ntype = new NamedType(yyvsp[-1].strval); ;
    break;}
case 88:
#line 343 "xi-grammar.y"
{ yyval.chare = new Array(lineno, yyvsp[-3].ntype, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 89:
#line 347 "xi-grammar.y"
{ yyval.chare = new Chare(lineno, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist);;
    break;}
case 90:
#line 349 "xi-grammar.y"
{ yyval.chare = new MainChare(lineno, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 91:
#line 353 "xi-grammar.y"
{ yyval.chare = new Group(lineno, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 92:
#line 357 "xi-grammar.y"
{ yyval.chare = new NodeGroup( lineno, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 93:
#line 361 "xi-grammar.y"
{ yyval.chare = new Array( lineno, yyvsp[-3].ntype, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); ;
    break;}
case 94:
#line 365 "xi-grammar.y"
{ yyval.message = new Message(lineno, new NamedType(yyvsp[-1].strval), yyvsp[-2].intval); ;
    break;}
case 95:
#line 369 "xi-grammar.y"
{ yyval.type = 0; ;
    break;}
case 96:
#line 371 "xi-grammar.y"
{ yyval.type = yyvsp[0].type; ;
    break;}
case 97:
#line 375 "xi-grammar.y"
{ yyval.strval = 0; ;
    break;}
case 98:
#line 377 "xi-grammar.y"
{ yyval.strval = yyvsp[0].strval; ;
    break;}
case 99:
#line 379 "xi-grammar.y"
{ yyval.strval = yyvsp[0].strval; ;
    break;}
case 100:
#line 383 "xi-grammar.y"
{ yyval.tvar = new TType(new NamedType(yyvsp[-1].strval), yyvsp[0].type); ;
    break;}
case 101:
#line 385 "xi-grammar.y"
{ yyval.tvar = new TFunc(yyvsp[-1].ftype, yyvsp[0].strval); ;
    break;}
case 102:
#line 387 "xi-grammar.y"
{ yyval.tvar = new TName(yyvsp[-2].type, yyvsp[-1].strval, yyvsp[0].strval); ;
    break;}
case 103:
#line 391 "xi-grammar.y"
{ yyval.tvarlist = new TVarList(yyvsp[0].tvar); ;
    break;}
case 104:
#line 393 "xi-grammar.y"
{ yyval.tvarlist = new TVarList(yyvsp[-2].tvar, yyvsp[0].tvarlist); ;
    break;}
case 105:
#line 397 "xi-grammar.y"
{ yyval.tvarlist = yyvsp[-1].tvarlist; ;
    break;}
case 106:
#line 401 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); ;
    break;}
case 107:
#line 403 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); ;
    break;}
case 108:
#line 405 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); ;
    break;}
case 109:
#line 407 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); ;
    break;}
case 110:
#line 409 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].message); yyvsp[0].message->setTemplate(yyval.templat); ;
    break;}
case 111:
#line 413 "xi-grammar.y"
{ yyval.mbrlist = 0; ;
    break;}
case 112:
#line 415 "xi-grammar.y"
{ yyval.mbrlist = yyvsp[-2].mbrlist; ;
    break;}
case 113:
#line 419 "xi-grammar.y"
{ yyval.mbrlist = 0; ;
    break;}
case 114:
#line 421 "xi-grammar.y"
{ yyval.mbrlist = new MemberList(yyvsp[-1].member, yyvsp[0].mbrlist); ;
    break;}
case 115:
#line 425 "xi-grammar.y"
{ yyval.member = yyvsp[-1].entry; ;
    break;}
case 116:
#line 427 "xi-grammar.y"
{ yyval.member = yyvsp[-1].readonly; ;
    break;}
case 117:
#line 429 "xi-grammar.y"
{ yyval.member = yyvsp[-1].readonly; ;
    break;}
case 118:
#line 433 "xi-grammar.y"
{ yyval.entry = new Entry(lineno, yyvsp[-5].intval|yyvsp[-1].intval, new BuiltinType("void"), yyvsp[-3].strval, yyvsp[-2].rtype, yyvsp[0].val); ;
    break;}
case 119:
#line 435 "xi-grammar.y"
{ yyval.entry = new Entry(lineno, yyvsp[-5].intval|yyvsp[-1].intval, yyvsp[-4].ptype, yyvsp[-3].strval, yyvsp[-2].rtype, yyvsp[0].val); ;
    break;}
case 120:
#line 437 "xi-grammar.y"
{ yyval.entry = new Entry(lineno, yyvsp[-2].intval, 0, yyvsp[-1].strval, yyvsp[0].rtype, 0); ;
    break;}
case 121:
#line 441 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 122:
#line 443 "xi-grammar.y"
{ yyval.intval = yyvsp[-1].intval; ;
    break;}
case 123:
#line 447 "xi-grammar.y"
{ yyval.intval = yyvsp[0].intval; ;
    break;}
case 124:
#line 449 "xi-grammar.y"
{ yyval.intval = yyvsp[-2].intval | yyvsp[0].intval; ;
    break;}
case 125:
#line 453 "xi-grammar.y"
{ yyval.intval = STHREADED; ;
    break;}
case 126:
#line 455 "xi-grammar.y"
{ yyval.intval = SSYNC; ;
    break;}
case 127:
#line 457 "xi-grammar.y"
{ yyval.intval = SLOCKED; ;
    break;}
case 128:
#line 459 "xi-grammar.y"
{ yyval.intval = SVIRTUAL; ;
    break;}
case 129:
#line 463 "xi-grammar.y"
{ yyval.rtype = 0; ;
    break;}
case 130:
#line 465 "xi-grammar.y"
{ yyval.rtype = new BuiltinType("void"); ;
    break;}
case 131:
#line 467 "xi-grammar.y"
{ yyval.rtype = yyvsp[0].ptype; ;
    break;}
case 132:
#line 471 "xi-grammar.y"
{ yyval.rtype = yyvsp[-1].rtype; ;
    break;}
case 133:
#line 475 "xi-grammar.y"
{ yyval.val = 0; ;
    break;}
case 134:
#line 477 "xi-grammar.y"
{ yyval.val = new Value(yyvsp[0].strval); ;
    break;}
case 135:
#line 481 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 136:
#line 483 "xi-grammar.y"
{ if(strcmp(yyvsp[0].strval, "0")) { yyerror("expected 0"); exit(1); }
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
#line 487 "xi-grammar.y"

void yyerror(const char *mesg)
{
  cout << "Syntax error at line " << lineno << ": " << mesg << endl;
  // return 0;
}
