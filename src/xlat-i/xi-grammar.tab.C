
/*  A Bison parser, made from xi-grammar.y
 by  GNU Bison version 1.25
  */

#define YYBISON 1  /* Identify Bison output.  */

#define	MODULE	258
#define	MAINMODULE	259
#define	EXTERN	260
#define	READONLY	261
#define	CHARE	262
#define	GROUP	263
#define	MESSAGE	264
#define	CLASS	265
#define	STACKSIZE	266
#define	THREADED	267
#define	TEMPLATE	268
#define	SYNC	269
#define	VOID	270
#define	PACKED	271
#define	VARSIZE	272
#define	ENTRY	273
#define	MAINCHARE	274
#define	IDENT	275
#define	NUMBER	276
#define	LITERAL	277
#define	INT	278
#define	LONG	279
#define	SHORT	280
#define	CHAR	281
#define	FLOAT	282
#define	DOUBLE	283
#define	UNSIGNED	284

#line 1 "xi-grammar.y"

#include <iostream.h>
#include "xi-symbol.h"

extern int yylex (void) ;
int yyerror(char *);
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
  char *strval;
  int intval;
} YYSTYPE;
#include <stdio.h>

#ifndef __cplusplus
#ifndef __STDC__
#define const
#endif
#endif



#define	YYFINAL		205
#define	YYFLAG		-32768
#define	YYNTBASE	43

#define YYTRANSLATE(x) ((unsigned)(x) <= 284 ? yytranslate[x] : 95)

static const char yytranslate[] = {     0,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,    39,
    40,    36,     2,    33,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,    41,    30,    34,
    42,    35,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
    37,     2,    38,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,    31,     2,    32,     2,     2,     2,     2,     2,
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
     2,     2,     2,     2,     2,     1,     2,     3,     4,     5,
     6,     7,     8,     9,    10,    11,    12,    13,    14,    15,
    16,    17,    18,    19,    20,    21,    22,    23,    24,    25,
    26,    27,    28,    29
};

#if YYDEBUG != 0
static const short yyprhs[] = {     0,
     0,     2,     3,     6,     7,     9,    10,    12,    14,    18,
    22,    24,    29,    30,    33,    39,    42,    46,    50,    54,
    57,    60,    63,    65,    67,    69,    71,    75,    76,    78,
    79,    83,    85,    87,    89,    91,    94,    97,   100,   103,
   106,   108,   110,   113,   115,   118,   120,   122,   125,   128,
   131,   133,   135,   140,   149,   151,   153,   155,   157,   159,
   160,   162,   166,   170,   176,   177,   181,   183,   187,   189,
   191,   195,   196,   199,   201,   205,   210,   215,   220,   225,
   230,   235,   240,   241,   244,   245,   248,   251,   255,   258,
   262,   264,   268,   273,   276,   279,   282,   284,   289,   290,
   293,   296,   299,   302,   309,   316,   321,   322,   326,   328,
   332,   334,   336,   337,   339,   341,   345,   346
};

static const short yyrhs[] = {    44,
     0,     0,    48,    44,     0,     0,     5,     0,     0,    30,
     0,    20,     0,     3,    47,    49,     0,     4,    47,    49,
     0,    30,     0,    31,    50,    32,    46,     0,     0,    51,
    50,     0,    45,    31,    50,    32,    46,     0,    45,    48,
     0,    45,    66,    30,     0,    45,    67,    30,     0,    45,
    71,    30,     0,    45,    74,     0,    45,    75,     0,    45,
    84,     0,    64,     0,    21,     0,    22,     0,    52,     0,
    52,    33,    53,     0,     0,    53,     0,     0,    34,    54,
    35,     0,    23,     0,    24,     0,    25,     0,    26,     0,
    29,    23,     0,    29,    24,     0,    29,    25,     0,    29,
    26,     0,    24,    24,     0,    27,     0,    28,     0,    24,
    28,     0,    15,     0,    47,    55,     0,    56,     0,    57,
     0,    58,    36,     0,    59,    36,     0,    60,    36,     0,
    21,     0,    47,     0,    64,    37,    61,    38,     0,    64,
    39,    36,    47,    40,    39,    65,    40,     0,    58,     0,
    59,     0,    60,     0,    62,     0,    63,     0,     0,    64,
     0,    64,    33,    65,     0,     6,    64,    47,     0,     6,
     9,    58,    36,    47,     0,     0,    37,    69,    38,     0,
    70,     0,    70,    33,    69,     0,    16,     0,    17,     0,
     9,    68,    57,     0,     0,    41,    73,     0,    57,     0,
    57,    33,    73,     0,     7,    57,    72,    85,     0,    19,
    57,    72,    85,     0,     8,    57,    72,    85,     0,     7,
    47,    72,    85,     0,    19,    47,    72,    85,     0,     8,
    47,    72,    85,     0,     9,    68,    47,    30,     0,     0,
    42,    64,     0,     0,    42,    21,     0,    42,    22,     0,
    10,    47,    79,     0,    63,    80,     0,    64,    47,    80,
     0,    81,     0,    81,    33,    82,     0,    13,    34,    82,
    35,     0,    83,    76,     0,    83,    77,     0,    83,    78,
     0,    30,     0,    31,    86,    32,    46,     0,     0,    87,
    86,     0,    88,    30,     0,    66,    30,     0,    67,    30,
     0,    18,    89,    15,    47,    93,    94,     0,    18,    89,
    59,    47,    93,    94,     0,    18,    89,    47,    93,     0,
     0,    37,    90,    38,     0,    91,     0,    91,    33,    90,
     0,    12,     0,    14,     0,     0,    15,     0,    59,     0,
    39,    92,    40,     0,     0,    11,    42,    21,     0
};

#endif

#if YYDEBUG != 0
static const short yyrline[] = { 0,
    88,    92,    94,    98,   100,   104,   106,   110,   114,   116,
   120,   122,   126,   128,   132,   134,   136,   138,   140,   142,
   144,   146,   150,   152,   154,   158,   160,   164,   166,   170,
   172,   176,   178,   180,   182,   184,   186,   188,   190,   192,
   194,   196,   198,   200,   204,   208,   210,   214,   218,   220,
   224,   226,   230,   234,   238,   240,   242,   244,   246,   250,
   252,   254,   258,   262,   266,   268,   272,   274,   278,   280,
   284,   288,   290,   294,   296,   300,   302,   307,   311,   314,
   319,   324,   328,   330,   334,   336,   338,   342,   344,   346,
   350,   352,   356,   360,   362,   364,   368,   370,   374,   376,
   380,   382,   384,   388,   390,   392,   396,   398,   402,   404,
   408,   410,   414,   416,   418,   422,   426,   428
};
#endif


#if YYDEBUG != 0 || defined (YYERROR_VERBOSE)

static const char * const yytname[] = {   "$","error","$undefined.","MODULE",
"MAINMODULE","EXTERN","READONLY","CHARE","GROUP","MESSAGE","CLASS","STACKSIZE",
"THREADED","TEMPLATE","SYNC","VOID","PACKED","VARSIZE","ENTRY","MAINCHARE","IDENT",
"NUMBER","LITERAL","INT","LONG","SHORT","CHAR","FLOAT","DOUBLE","UNSIGNED","';'",
"'{'","'}'","','","'<'","'>'","'*'","'['","']'","'('","')'","':'","'='","File",
"ModuleEList","OptExtern","OptSemiColon","Name","Module","ConstructEList","ConstructList",
"Construct","TParam","TParamList","TParamEList","OptTParams","BuiltinType","NamedType",
"SimpleType","OnePtrType","PtrType","ArrayDim","ArrayType","FuncType","Type",
"TypeList","Readonly","ReadonlyMsg","MAttribs","MAttribList","MAttrib","Message",
"OptBaseList","BaseList","Chare","Group","TChare","TGroup","TMessage","OptTypeInit",
"OptNameInit","TVar","TVarList","TemplateSpec","Template","MemberEList","MemberList",
"Member","Entry","EAttribs","EAttribList","EAttrib","OptType","EParam","OptStackSize", NULL
};
#endif

static const short yyr1[] = {     0,
    43,    44,    44,    45,    45,    46,    46,    47,    48,    48,
    49,    49,    50,    50,    51,    51,    51,    51,    51,    51,
    51,    51,    52,    52,    52,    53,    53,    54,    54,    55,
    55,    56,    56,    56,    56,    56,    56,    56,    56,    56,
    56,    56,    56,    56,    57,    58,    58,    59,    60,    60,
    61,    61,    62,    63,    64,    64,    64,    64,    64,    65,
    65,    65,    66,    67,    68,    68,    69,    69,    70,    70,
    71,    72,    72,    73,    73,    74,    74,    75,    76,    76,
    77,    78,    79,    79,    80,    80,    80,    81,    81,    81,
    82,    82,    83,    84,    84,    84,    85,    85,    86,    86,
    87,    87,    87,    88,    88,    88,    89,    89,    90,    90,
    91,    91,    92,    92,    92,    93,    94,    94
};

static const short yyr2[] = {     0,
     1,     0,     2,     0,     1,     0,     1,     1,     3,     3,
     1,     4,     0,     2,     5,     2,     3,     3,     3,     2,
     2,     2,     1,     1,     1,     1,     3,     0,     1,     0,
     3,     1,     1,     1,     1,     2,     2,     2,     2,     2,
     1,     1,     2,     1,     2,     1,     1,     2,     2,     2,
     1,     1,     4,     8,     1,     1,     1,     1,     1,     0,
     1,     3,     3,     5,     0,     3,     1,     3,     1,     1,
     3,     0,     2,     1,     3,     4,     4,     4,     4,     4,
     4,     4,     0,     2,     0,     2,     2,     3,     2,     3,
     1,     3,     4,     2,     2,     2,     1,     4,     0,     2,
     2,     2,     2,     6,     6,     4,     0,     3,     1,     3,
     1,     1,     0,     1,     1,     3,     0,     3
};

static const short yydefact[] = {     2,
     0,     0,     1,     2,     8,     0,     0,     3,    11,     4,
     9,    10,     5,     0,     0,     4,     0,     0,     0,    65,
     0,     0,     4,    16,     0,     0,     0,    20,    21,     0,
    22,     6,    14,     0,    44,    32,    33,    34,    35,    41,
    42,     0,    30,    46,    47,    55,    56,    57,    58,    59,
     0,    72,    72,     0,     0,     0,    72,     0,    17,    18,
    19,     0,     0,    65,     0,    94,    95,    96,     7,    12,
     0,    40,    43,    36,    37,    38,    39,    28,    45,    48,
    49,    50,     0,     0,    63,     0,     0,     0,    69,    70,
     0,    67,    71,     0,    59,     0,    91,     0,     0,     6,
    72,    72,     0,    72,     0,    24,    25,    26,    29,     0,
    23,    51,    52,     0,     0,    74,    73,    97,    99,    76,
    78,    66,     0,    83,     0,    89,    85,     0,    93,    77,
    15,     0,     0,     0,     0,    64,     0,    31,    53,     0,
     0,   107,     0,     0,     0,    99,     0,    68,     0,    88,
    86,    87,    90,    92,    79,    81,    82,    80,    27,     0,
    75,     0,     0,   102,   103,     6,   100,   101,    84,    60,
   111,   112,     0,   109,    44,    30,     0,     0,    98,    61,
     0,   108,     0,     0,   113,   106,     0,    60,    54,   110,
   117,    44,   115,     0,   117,    62,     0,   104,   116,   105,
     0,   118,     0,     0,     0
};

static const short yydefgoto[] = {   203,
     3,    14,    70,    43,     4,    11,    15,    16,   108,   109,
   110,    79,    44,    45,    46,    47,    48,   114,    49,    50,
    96,   181,   143,   144,    55,    91,    92,    27,    87,   117,
    28,    29,    66,    67,    68,   150,   126,    97,    98,    30,
    31,   120,   145,   146,   147,   163,   173,   174,   194,   186,
   198
};

static const short yypact[] = {    93,
    -8,    -8,-32768,    93,-32768,    70,    70,-32768,-32768,     5,
-32768,-32768,-32768,    11,     7,     5,    20,    -8,    -8,    17,
    25,    -8,     5,-32768,    47,    51,    61,-32768,-32768,    19,
-32768,    64,-32768,    83,-32768,-32768,    28,-32768,-32768,-32768,
-32768,   100,    45,-32768,-32768,    69,    82,    95,-32768,-32768,
    14,    92,    92,    99,    -8,    60,    92,   116,-32768,-32768,
-32768,    -8,    -8,    17,    -8,-32768,-32768,-32768,-32768,-32768,
   111,-32768,-32768,-32768,-32768,-32768,-32768,   115,-32768,-32768,
-32768,-32768,   108,   113,-32768,    -8,    90,    90,-32768,-32768,
   112,   118,-32768,    -8,    57,    14,   119,   122,    90,    64,
    92,    92,    -8,    92,    -8,-32768,-32768,   120,-32768,   123,
    32,-32768,-32768,   121,    -8,   127,-32768,-32768,    49,-32768,
-32768,-32768,    99,   130,   124,-32768,   133,    60,-32768,-32768,
-32768,    90,    90,   148,    90,-32768,   115,-32768,-32768,   146,
    -8,   126,   157,   158,   159,    49,   160,-32768,    83,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,   150,
-32768,    62,   141,-32768,-32768,    64,-32768,-32768,    32,    83,
-32768,-32768,   154,   161,    -8,    -3,    69,    -8,-32768,    80,
   153,-32768,    62,   162,   156,-32768,   162,    83,-32768,-32768,
   184,   163,-32768,   164,   184,-32768,   155,-32768,-32768,-32768,
   175,-32768,   198,   199,-32768
};

static const short yypgoto[] = {-32768,
   196,-32768,   -93,    -1,   188,   200,     0,-32768,-32768,    68,
-32768,-32768,-32768,   -14,   -31,  -152,-32768,-32768,-32768,   -50,
   -15,    18,   194,   195,   147,    87,-32768,-32768,   -44,    71,
-32768,-32768,-32768,-32768,-32768,-32768,    86,-32768,    88,-32768,
-32768,   -67,    72,-32768,-32768,-32768,    31,-32768,-32768,  -162,
    22
};


#define	YYLAST		218


static const short yytable[] = {     6,
     7,    51,    71,    52,    53,    95,   131,    57,    88,    13,
   178,     5,    99,     1,     2,    33,    17,    18,    19,    20,
   121,   191,    58,    21,   195,    62,    63,    64,    34,    22,
    78,   130,   193,     5,    35,   185,   -13,    65,    32,     5,
    93,    23,    36,    37,    38,    39,    40,    41,    42,    85,
    83,    72,    84,    54,    17,    73,   132,   133,    56,   135,
   101,   102,   111,   104,   155,   156,   142,   158,    83,    94,
    84,   116,   179,   171,    35,   172,    59,    95,    78,     5,
    60,   113,    36,    37,    38,    39,    40,    41,    42,   -85,
    61,   -85,   124,    69,   127,     1,     2,    35,   125,     9,
    10,   134,     5,   136,    80,    36,    37,    38,    39,    40,
    41,    42,   188,   140,    89,    90,    83,    81,    84,   118,
   119,   111,    74,    75,    76,    77,   116,     5,   112,    35,
    82,   177,    86,   169,     5,   106,   107,    36,    37,    38,
    39,    40,    41,    42,   151,   152,   105,   100,   115,   122,
   123,   128,   137,   177,   180,   175,   129,   138,   139,   141,
     5,   176,   162,    36,    37,    38,    39,    40,    41,    42,
   192,   149,   180,   184,   125,     5,   187,   157,    36,    37,
    38,    39,    40,    41,    42,   160,   164,   165,   170,   168,
   166,   182,   189,   183,   197,   202,   201,   204,   205,     8,
   185,    24,  -114,   199,   159,   196,    12,    25,    26,   148,
   103,   161,   153,   190,     0,   154,   200,   167
};

static const short yycheck[] = {     1,
     2,    17,    34,    18,    19,    56,   100,    22,    53,     5,
   163,    20,    57,     3,     4,    16,     6,     7,     8,     9,
    88,   184,    23,    13,   187,     7,     8,     9,     9,    19,
    34,    99,   185,    20,    15,    39,    32,    19,    32,    20,
    55,    31,    23,    24,    25,    26,    27,    28,    29,    51,
    37,    24,    39,    37,     6,    28,   101,   102,    34,   104,
    62,    63,    78,    65,   132,   133,    18,   135,    37,    10,
    39,    86,   166,    12,    15,    14,    30,   128,    34,    20,
    30,    83,    23,    24,    25,    26,    27,    28,    29,    33,
    30,    35,    94,    30,    96,     3,     4,    15,    42,    30,
    31,   103,    20,   105,    36,    23,    24,    25,    26,    27,
    28,    29,    33,   115,    16,    17,    37,    36,    39,    30,
    31,   137,    23,    24,    25,    26,   141,    20,    21,    15,
    36,   163,    41,   149,    20,    21,    22,    23,    24,    25,
    26,    27,    28,    29,    21,    22,    36,    32,    36,    38,
    33,    33,    33,   185,   170,    15,    35,    35,    38,    33,
    20,   163,    37,    23,    24,    25,    26,    27,    28,    29,
    15,    42,   188,   175,    42,    20,   178,    30,    23,    24,
    25,    26,    27,    28,    29,    40,    30,    30,    39,    30,
    32,    38,    40,    33,    11,    21,    42,     0,     0,     4,
    39,    14,    40,    40,   137,   188,     7,    14,    14,   123,
    64,   141,   127,   183,    -1,   128,   195,   146
};
/* -*-C-*-  Note some compilers choke on comments on `#line' lines.  */
#line 3 "/usr/local/encap/bison-1.25/share/bison.simple"

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
   Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.  */

/* As a special exception, when this file is copied by Bison into a
   Bison output file, you may use that output file without restriction.
   This special exception was added by the Free Software Foundation
   in version 1.24 of Bison.  */

#ifndef alloca
#ifdef __GNUC__
#define alloca __builtin_alloca
#else /* not GNU C.  */
#if (!defined (__STDC__) && defined (sparc)) || defined (__sparc__) || defined (__sparc) || defined (__sgi)
#include <alloca.h>
#else /* not sparc */
#if defined (MSDOS) && !defined (__TURBOC__)
#include <malloc.h>
#else /* not MSDOS, or __TURBOC__ */
#if defined(_AIX)
#include <malloc.h>
 #pragma alloca
#else /* not MSDOS, __TURBOC__, or _AIX */
#ifdef __hpux
#ifdef __cplusplus
extern "C" {
void *alloca (unsigned int);
};
#else /* not __cplusplus */
void *alloca ();
#endif /* not __cplusplus */
#endif /* __hpux */
#endif /* not _AIX */
#endif /* not MSDOS, or __TURBOC__ */
#endif /* not sparc.  */
#endif /* not GNU C.  */
#endif /* alloca not defined.  */

/* This is the parser code that is written into each bison parser
  when the %semantic_parser declaration is not specified in the grammar.
  It was written by Richard Stallman by simplifying the hairy parser
  used when %semantic_parser is specified.  */

/* Note: there must be only one dollar sign in this file.
   It is replaced by the list of actions, each action
   as one case of the switch.  */

#define yyerrok		(yyerrstatus = 0)
#define yyclearin	(yychar = YYEMPTY)
#define YYEMPTY		-2
#define YYEOF		0
#define YYACCEPT	return(0)
#define YYABORT 	return(1)
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

/* Prevent warning if -Wstrict-prototypes.  */
#ifdef __GNUC__
int yyparse (void);
#endif

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
     int count;
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
__yy_memcpy (char *to, char *from, int count)
{
  register char *f = from;
  register char *t = to;
  register int i = count;

  while (i-- > 0)
    *t++ = *f++;
}

#endif
#endif

#line 196 "/usr/local/encap/bison-1.25/share/bison.simple"

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
	  return 2;
	}
      yystacksize *= 2;
      if (yystacksize > YYMAXDEPTH)
	yystacksize = YYMAXDEPTH;
      yyss = (short *) alloca (yystacksize * sizeof (*yyssp));
      __yy_memcpy ((char *)yyss, (char *)yyss1, size * sizeof (*yyssp));
      yyvs = (YYSTYPE *) alloca (yystacksize * sizeof (*yyvsp));
      __yy_memcpy ((char *)yyvs, (char *)yyvs1, size * sizeof (*yyvsp));
#ifdef YYLSP_NEEDED
      yyls = (YYLTYPE *) alloca (yystacksize * sizeof (*yylsp));
      __yy_memcpy ((char *)yyls, (char *)yyls1, size * sizeof (*yylsp));
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
#line 89 "xi-grammar.y"
{ yyval.modlist = yyvsp[0].modlist; modlist = yyvsp[0].modlist; ;
    break;}
case 2:
#line 93 "xi-grammar.y"
{ yyval.modlist = 0; ;
    break;}
case 3:
#line 95 "xi-grammar.y"
{ yyval.modlist = new ModuleList(yyvsp[-1].module, yyvsp[0].modlist); ;
    break;}
case 4:
#line 99 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 5:
#line 101 "xi-grammar.y"
{ yyval.intval = 1; ;
    break;}
case 6:
#line 105 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 7:
#line 107 "xi-grammar.y"
{ yyval.intval = 1; ;
    break;}
case 8:
#line 111 "xi-grammar.y"
{ yyval.strval = yyvsp[0].strval; ;
    break;}
case 9:
#line 115 "xi-grammar.y"
{ yyval.module = new Module(yyvsp[-1].strval, yyvsp[0].conslist); ;
    break;}
case 10:
#line 117 "xi-grammar.y"
{ yyval.module = new Module(yyvsp[-1].strval, yyvsp[0].conslist); yyval.module->setMain(); ;
    break;}
case 11:
#line 121 "xi-grammar.y"
{ yyval.conslist = 0; ;
    break;}
case 12:
#line 123 "xi-grammar.y"
{ yyval.conslist = yyvsp[-2].conslist; ;
    break;}
case 13:
#line 127 "xi-grammar.y"
{ yyval.conslist = 0; ;
    break;}
case 14:
#line 129 "xi-grammar.y"
{ yyval.conslist = new ConstructList(yyvsp[-1].construct, yyvsp[0].conslist); ;
    break;}
case 15:
#line 133 "xi-grammar.y"
{ if(yyvsp[-2].conslist) yyvsp[-2].conslist->setExtern(yyvsp[-4].intval); yyval.construct = yyvsp[-2].conslist; ;
    break;}
case 16:
#line 135 "xi-grammar.y"
{ yyvsp[0].module->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].module; ;
    break;}
case 17:
#line 137 "xi-grammar.y"
{ yyvsp[-1].readonly->setExtern(yyvsp[-2].intval); yyval.construct = yyvsp[-1].readonly; ;
    break;}
case 18:
#line 139 "xi-grammar.y"
{ yyvsp[-1].readonly->setExtern(yyvsp[-2].intval); yyval.construct = yyvsp[-1].readonly; ;
    break;}
case 19:
#line 141 "xi-grammar.y"
{ yyvsp[-1].message->setExtern(yyvsp[-2].intval); yyval.construct = yyvsp[-1].message; ;
    break;}
case 20:
#line 143 "xi-grammar.y"
{ yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; ;
    break;}
case 21:
#line 145 "xi-grammar.y"
{ yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; ;
    break;}
case 22:
#line 147 "xi-grammar.y"
{ yyvsp[0].templat->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].templat; ;
    break;}
case 23:
#line 151 "xi-grammar.y"
{ yyval.tparam = new TParamType(yyvsp[0].type); ;
    break;}
case 24:
#line 153 "xi-grammar.y"
{ yyval.tparam = new TParamVal(yyvsp[0].strval); ;
    break;}
case 25:
#line 155 "xi-grammar.y"
{ yyval.tparam = new TParamVal(yyvsp[0].strval); ;
    break;}
case 26:
#line 159 "xi-grammar.y"
{ yyval.tparlist = new TParamList(yyvsp[0].tparam); ;
    break;}
case 27:
#line 161 "xi-grammar.y"
{ yyval.tparlist = new TParamList(yyvsp[-2].tparam, yyvsp[0].tparlist); ;
    break;}
case 28:
#line 165 "xi-grammar.y"
{ yyval.tparlist = 0; ;
    break;}
case 29:
#line 167 "xi-grammar.y"
{ yyval.tparlist = yyvsp[0].tparlist; ;
    break;}
case 30:
#line 171 "xi-grammar.y"
{ yyval.tparlist = 0; ;
    break;}
case 31:
#line 173 "xi-grammar.y"
{ yyval.tparlist = yyvsp[-1].tparlist; ;
    break;}
case 32:
#line 177 "xi-grammar.y"
{ yyval.type = new BuiltinType("int"); ;
    break;}
case 33:
#line 179 "xi-grammar.y"
{ yyval.type = new BuiltinType("long"); ;
    break;}
case 34:
#line 181 "xi-grammar.y"
{ yyval.type = new BuiltinType("short"); ;
    break;}
case 35:
#line 183 "xi-grammar.y"
{ yyval.type = new BuiltinType("char"); ;
    break;}
case 36:
#line 185 "xi-grammar.y"
{ yyval.type = new BuiltinType("unsigned int"); ;
    break;}
case 37:
#line 187 "xi-grammar.y"
{ yyval.type = new BuiltinType("long"); ;
    break;}
case 38:
#line 189 "xi-grammar.y"
{ yyval.type = new BuiltinType("short"); ;
    break;}
case 39:
#line 191 "xi-grammar.y"
{ yyval.type = new BuiltinType("char"); ;
    break;}
case 40:
#line 193 "xi-grammar.y"
{ yyval.type = new BuiltinType("long long"); ;
    break;}
case 41:
#line 195 "xi-grammar.y"
{ yyval.type = new BuiltinType("float"); ;
    break;}
case 42:
#line 197 "xi-grammar.y"
{ yyval.type = new BuiltinType("double"); ;
    break;}
case 43:
#line 199 "xi-grammar.y"
{ yyval.type = new BuiltinType("long double"); ;
    break;}
case 44:
#line 201 "xi-grammar.y"
{ yyval.type = new BuiltinType("void"); ;
    break;}
case 45:
#line 205 "xi-grammar.y"
{ yyval.ntype = new NamedType(yyvsp[-1].strval, yyvsp[0].tparlist); ;
    break;}
case 46:
#line 209 "xi-grammar.y"
{ yyval.type = yyvsp[0].type; ;
    break;}
case 47:
#line 211 "xi-grammar.y"
{ yyval.type = yyvsp[0].ntype; ;
    break;}
case 48:
#line 215 "xi-grammar.y"
{ yyval.ptype = new PtrType(yyvsp[-1].type); ;
    break;}
case 49:
#line 219 "xi-grammar.y"
{ yyvsp[-1].ptype->indirect(); yyval.ptype = yyvsp[-1].ptype; ;
    break;}
case 50:
#line 221 "xi-grammar.y"
{ yyvsp[-1].ptype->indirect(); yyval.ptype = yyvsp[-1].ptype; ;
    break;}
case 51:
#line 225 "xi-grammar.y"
{ yyval.val = new Value(yyvsp[0].strval); ;
    break;}
case 52:
#line 227 "xi-grammar.y"
{ yyval.val = new Value(yyvsp[0].strval); ;
    break;}
case 53:
#line 231 "xi-grammar.y"
{ yyval.type = new ArrayType(yyvsp[-3].type, yyvsp[-1].val); ;
    break;}
case 54:
#line 235 "xi-grammar.y"
{ yyval.ftype = new FuncType(yyvsp[-7].type, yyvsp[-4].strval, yyvsp[-1].typelist); ;
    break;}
case 55:
#line 239 "xi-grammar.y"
{ yyval.type = yyvsp[0].type; ;
    break;}
case 56:
#line 241 "xi-grammar.y"
{ yyval.type = (Type*) yyvsp[0].ptype; ;
    break;}
case 57:
#line 243 "xi-grammar.y"
{ yyval.type = (Type*) yyvsp[0].ptype; ;
    break;}
case 58:
#line 245 "xi-grammar.y"
{ yyval.type = yyvsp[0].type; ;
    break;}
case 59:
#line 247 "xi-grammar.y"
{ yyval.type = yyvsp[0].ftype; ;
    break;}
case 60:
#line 251 "xi-grammar.y"
{ yyval.typelist = 0; ;
    break;}
case 61:
#line 253 "xi-grammar.y"
{ yyval.typelist = new TypeList(yyvsp[0].type); ;
    break;}
case 62:
#line 255 "xi-grammar.y"
{ yyval.typelist = new TypeList(yyvsp[-2].type, yyvsp[0].typelist); ;
    break;}
case 63:
#line 259 "xi-grammar.y"
{ yyval.readonly = new Readonly(yyvsp[-1].type, yyvsp[0].strval); ;
    break;}
case 64:
#line 263 "xi-grammar.y"
{ yyval.readonly = new Readonly(yyvsp[-2].type, yyvsp[0].strval, 1); ;
    break;}
case 65:
#line 267 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 66:
#line 269 "xi-grammar.y"
{ yyval.intval = yyvsp[-1].intval; ;
    break;}
case 67:
#line 273 "xi-grammar.y"
{ yyval.intval = yyvsp[0].intval; ;
    break;}
case 68:
#line 275 "xi-grammar.y"
{ yyval.intval = yyvsp[-2].intval | yyvsp[0].intval; ;
    break;}
case 69:
#line 279 "xi-grammar.y"
{ yyval.intval = SPACKED; ;
    break;}
case 70:
#line 281 "xi-grammar.y"
{ yyval.intval = SVARSIZE; ;
    break;}
case 71:
#line 285 "xi-grammar.y"
{ yyval.message = new Message(yyvsp[0].ntype, yyvsp[-1].intval); ;
    break;}
case 72:
#line 289 "xi-grammar.y"
{ yyval.typelist = 0; ;
    break;}
case 73:
#line 291 "xi-grammar.y"
{ yyval.typelist = yyvsp[0].typelist; ;
    break;}
case 74:
#line 295 "xi-grammar.y"
{ yyval.typelist = new TypeList(yyvsp[0].ntype); ;
    break;}
case 75:
#line 297 "xi-grammar.y"
{ yyval.typelist = new TypeList(yyvsp[-2].ntype, yyvsp[0].typelist); ;
    break;}
case 76:
#line 301 "xi-grammar.y"
{ yyval.chare = new Chare(SCHARE, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); if(yyvsp[0].mbrlist) yyvsp[0].mbrlist->setChare(yyval.chare);;
    break;}
case 77:
#line 303 "xi-grammar.y"
{ yyval.chare = new Chare(SMAINCHARE, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); 
                  if(yyvsp[0].mbrlist) yyvsp[0].mbrlist->setChare(yyval.chare);;
    break;}
case 78:
#line 308 "xi-grammar.y"
{ yyval.chare = new Chare(SGROUP, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); if(yyvsp[0].mbrlist) yyvsp[0].mbrlist->setChare(yyval.chare);;
    break;}
case 79:
#line 312 "xi-grammar.y"
{ yyval.chare = new Chare(SCHARE, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); 
                  if(yyvsp[0].mbrlist) yyvsp[0].mbrlist->setChare(yyval.chare);;
    break;}
case 80:
#line 315 "xi-grammar.y"
{ yyval.chare = new Chare(SMAINCHARE, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); 
                  if(yyvsp[0].mbrlist) yyvsp[0].mbrlist->setChare(yyval.chare);;
    break;}
case 81:
#line 320 "xi-grammar.y"
{ yyval.chare = new Chare(SGROUP, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); 
                  if(yyvsp[0].mbrlist) yyvsp[0].mbrlist->setChare(yyval.chare);;
    break;}
case 82:
#line 325 "xi-grammar.y"
{ yyval.message = new Message(new NamedType(yyvsp[-1].strval), yyvsp[-2].intval); ;
    break;}
case 83:
#line 329 "xi-grammar.y"
{ yyval.type = 0; ;
    break;}
case 84:
#line 331 "xi-grammar.y"
{ yyval.type = yyvsp[0].type; ;
    break;}
case 85:
#line 335 "xi-grammar.y"
{ yyval.strval = 0; ;
    break;}
case 86:
#line 337 "xi-grammar.y"
{ yyval.strval = yyvsp[0].strval; ;
    break;}
case 87:
#line 339 "xi-grammar.y"
{ yyval.strval = yyvsp[0].strval; ;
    break;}
case 88:
#line 343 "xi-grammar.y"
{ yyval.tvar = new TType(new NamedType(yyvsp[-1].strval), yyvsp[0].type); ;
    break;}
case 89:
#line 345 "xi-grammar.y"
{ yyval.tvar = new TFunc(yyvsp[-1].ftype, yyvsp[0].strval); ;
    break;}
case 90:
#line 347 "xi-grammar.y"
{ yyval.tvar = new TName(yyvsp[-2].type, yyvsp[-1].strval, yyvsp[0].strval); ;
    break;}
case 91:
#line 351 "xi-grammar.y"
{ yyval.tvarlist = new TVarList(yyvsp[0].tvar); ;
    break;}
case 92:
#line 353 "xi-grammar.y"
{ yyval.tvarlist = new TVarList(yyvsp[-2].tvar, yyvsp[0].tvarlist); ;
    break;}
case 93:
#line 357 "xi-grammar.y"
{ yyval.tvarlist = yyvsp[-1].tvarlist; ;
    break;}
case 94:
#line 361 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); ;
    break;}
case 95:
#line 363 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); ;
    break;}
case 96:
#line 365 "xi-grammar.y"
{ yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].message); yyvsp[0].message->setTemplate(yyval.templat); ;
    break;}
case 97:
#line 369 "xi-grammar.y"
{ yyval.mbrlist = 0; ;
    break;}
case 98:
#line 371 "xi-grammar.y"
{ yyval.mbrlist = yyvsp[-2].mbrlist; ;
    break;}
case 99:
#line 375 "xi-grammar.y"
{ yyval.mbrlist = 0; ;
    break;}
case 100:
#line 377 "xi-grammar.y"
{ yyval.mbrlist = new MemberList(yyvsp[-1].member, yyvsp[0].mbrlist); ;
    break;}
case 101:
#line 381 "xi-grammar.y"
{ yyval.member = yyvsp[-1].entry; ;
    break;}
case 102:
#line 383 "xi-grammar.y"
{ yyval.member = yyvsp[-1].readonly; ;
    break;}
case 103:
#line 385 "xi-grammar.y"
{ yyval.member = yyvsp[-1].readonly; ;
    break;}
case 104:
#line 389 "xi-grammar.y"
{ yyval.entry = new Entry(yyvsp[-4].intval, new BuiltinType("void"), yyvsp[-2].strval, yyvsp[-1].rtype, yyvsp[0].val); ;
    break;}
case 105:
#line 391 "xi-grammar.y"
{ yyval.entry = new Entry(yyvsp[-4].intval, yyvsp[-3].ptype, yyvsp[-2].strval, yyvsp[-1].rtype, yyvsp[0].val); ;
    break;}
case 106:
#line 393 "xi-grammar.y"
{ yyval.entry = new Entry(yyvsp[-2].intval, 0, yyvsp[-1].strval, yyvsp[0].rtype, 0); ;
    break;}
case 107:
#line 397 "xi-grammar.y"
{ yyval.intval = 0; ;
    break;}
case 108:
#line 399 "xi-grammar.y"
{ yyval.intval = yyvsp[-1].intval; ;
    break;}
case 109:
#line 403 "xi-grammar.y"
{ yyval.intval = yyvsp[0].intval; ;
    break;}
case 110:
#line 405 "xi-grammar.y"
{ yyval.intval = yyvsp[-2].intval | yyvsp[0].intval; ;
    break;}
case 111:
#line 409 "xi-grammar.y"
{ yyval.intval = STHREADED; ;
    break;}
case 112:
#line 411 "xi-grammar.y"
{ yyval.intval = SSYNC; ;
    break;}
case 113:
#line 415 "xi-grammar.y"
{ yyval.rtype = 0; ;
    break;}
case 114:
#line 417 "xi-grammar.y"
{ yyval.rtype = new BuiltinType("void"); ;
    break;}
case 115:
#line 419 "xi-grammar.y"
{ yyval.rtype = yyvsp[0].ptype; ;
    break;}
case 116:
#line 423 "xi-grammar.y"
{ yyval.rtype = yyvsp[-1].rtype; ;
    break;}
case 117:
#line 427 "xi-grammar.y"
{ yyval.val = 0; ;
    break;}
case 118:
#line 429 "xi-grammar.y"
{ yyval.val = new Value(yyvsp[0].strval); ;
    break;}
}
   /* the action file gets copied in in place of this dollarsign */
#line 498 "/usr/local/encap/bison-1.25/share/bison.simple"

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
}
#line 431 "xi-grammar.y"

int yyerror(char *mesg)
{
  cout << "Syntax error at line " << lineno << ": " << mesg << endl;
  return 0;
}
