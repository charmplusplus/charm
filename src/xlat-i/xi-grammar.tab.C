/* A Bison parser, made by GNU Bison 1.875.  */

/* Skeleton parser for Yacc-like parsing with Bison,
   Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002 Free Software Foundation, Inc.

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

/* Written by Richard Stallman by simplifying the original so called
   ``semantic'' parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Using locations.  */
#define YYLSP_NEEDED 0



/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum yytokentype {
     MODULE = 258,
     MAINMODULE = 259,
     EXTERN = 260,
     READONLY = 261,
     INITCALL = 262,
     INITNODE = 263,
     INITPROC = 264,
     PUPABLE = 265,
     CHARE = 266,
     MAINCHARE = 267,
     GROUP = 268,
     NODEGROUP = 269,
     ARRAY = 270,
     MESSAGE = 271,
     CLASS = 272,
     INCLUDE = 273,
     STACKSIZE = 274,
     THREADED = 275,
     TEMPLATE = 276,
     SYNC = 277,
     EXCLUSIVE = 278,
     IMMEDIATE = 279,
     SKIPSCHED = 280,
     VIRTUAL = 281,
     MIGRATABLE = 282,
     CREATEHERE = 283,
     CREATEHOME = 284,
     NOKEEP = 285,
     NOTRACE = 286,
     VOID = 287,
     CONST = 288,
     PACKED = 289,
     VARSIZE = 290,
     ENTRY = 291,
     FOR = 292,
     FORALL = 293,
     WHILE = 294,
     WHEN = 295,
     OVERLAP = 296,
     ATOMIC = 297,
     FORWARD = 298,
     IF = 299,
     ELSE = 300,
     CONNECT = 301,
     PUBLISHES = 302,
     PYTHON = 303,
     IDENT = 304,
     NUMBER = 305,
     LITERAL = 306,
     CPROGRAM = 307,
     HASHIF = 308,
     HASHIFDEF = 309,
     INT = 310,
     LONG = 311,
     SHORT = 312,
     CHAR = 313,
     FLOAT = 314,
     DOUBLE = 315,
     UNSIGNED = 316
   };
#endif
#define MODULE 258
#define MAINMODULE 259
#define EXTERN 260
#define READONLY 261
#define INITCALL 262
#define INITNODE 263
#define INITPROC 264
#define PUPABLE 265
#define CHARE 266
#define MAINCHARE 267
#define GROUP 268
#define NODEGROUP 269
#define ARRAY 270
#define MESSAGE 271
#define CLASS 272
#define INCLUDE 273
#define STACKSIZE 274
#define THREADED 275
#define TEMPLATE 276
#define SYNC 277
#define EXCLUSIVE 278
#define IMMEDIATE 279
#define SKIPSCHED 280
#define VIRTUAL 281
#define MIGRATABLE 282
#define CREATEHERE 283
#define CREATEHOME 284
#define NOKEEP 285
#define NOTRACE 286
#define VOID 287
#define CONST 288
#define PACKED 289
#define VARSIZE 290
#define ENTRY 291
#define FOR 292
#define FORALL 293
#define WHILE 294
#define WHEN 295
#define OVERLAP 296
#define ATOMIC 297
#define FORWARD 298
#define IF 299
#define ELSE 300
#define CONNECT 301
#define PUBLISHES 302
#define PYTHON 303
#define IDENT 304
#define NUMBER 305
#define LITERAL 306
#define CPROGRAM 307
#define HASHIF 308
#define HASHIFDEF 309
#define INT 310
#define LONG 311
#define SHORT 312
#define CHAR 313
#define FLOAT 314
#define DOUBLE 315
#define UNSIGNED 316




/* Copy the first part of user declarations.  */
#line 2 "xi-grammar.y"

#include "xi-symbol.h"
#include "EToken.h"
extern int yylex (void) ;
extern unsigned char in_comment;
void yyerror(const char *);
extern unsigned int lineno;
extern int in_bracket,in_braces,in_int_expr;
extern TList<Entry *> *connectEntries;
ModuleList *modlist;
extern int macroDefined(char *str, int istrue);



/* Enabling traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

#if ! defined (YYSTYPE) && ! defined (YYSTYPE_IS_DECLARED)
#line 16 "xi-grammar.y"
typedef union YYSTYPE {
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
  IncludeFile *includeFile;
  char *strval;
  int intval;
  Chare::attrib_t cattr;
  SdagConstruct *sc;
} YYSTYPE;
/* Line 191 of yacc.c.  */
#line 247 "y.tab.c"
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 214 of yacc.c.  */
#line 259 "y.tab.c"

#if ! defined (yyoverflow) || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# if YYSTACK_USE_ALLOCA
#  define YYSTACK_ALLOC alloca
# else
#  ifndef YYSTACK_USE_ALLOCA
#   if defined (alloca) || defined (_ALLOCA_H)
#    define YYSTACK_ALLOC alloca
#   else
#    ifdef __GNUC__
#     define YYSTACK_ALLOC __builtin_alloca
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's `empty if-body' warning. */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
# else
#  if defined (__STDC__) || defined (__cplusplus)
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   define YYSIZE_T size_t
#  endif
#  define YYSTACK_ALLOC malloc
#  define YYSTACK_FREE free
# endif
#endif /* ! defined (yyoverflow) || YYERROR_VERBOSE */


#if (! defined (yyoverflow) \
     && (! defined (__cplusplus) \
	 || (YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  short yyss;
  YYSTYPE yyvs;
  };

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (short) + sizeof (YYSTYPE))				\
      + YYSTACK_GAP_MAXIMUM)

/* Copy COUNT objects from FROM to TO.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if 1 < __GNUC__
#   define YYCOPY(To, From, Count) \
      __builtin_memcpy (To, From, (Count) * sizeof (*(From)))
#  else
#   define YYCOPY(To, From, Count)		\
      do					\
	{					\
	  register YYSIZE_T yyi;		\
	  for (yyi = 0; yyi < (Count); yyi++)	\
	    (To)[yyi] = (From)[yyi];		\
	}					\
      while (0)
#  endif
# endif

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack)					\
    do									\
      {									\
	YYSIZE_T yynewbytes;						\
	YYCOPY (&yyptr->Stack, Stack, yysize);				\
	Stack = &yyptr->Stack;						\
	yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
	yyptr += yynewbytes / sizeof (*yyptr);				\
      }									\
    while (0)

#endif

#if defined (__STDC__) || defined (__cplusplus)
   typedef signed char yysigned_char;
#else
   typedef short yysigned_char;
#endif

/* YYFINAL -- State number of the termination state. */
#define YYFINAL  9
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   546

/* YYNTOKENS -- Number of terminals. */
#define YYNTOKENS  76
/* YYNNTS -- Number of nonterminals. */
#define YYNNTS  100
/* YYNRULES -- Number of rules. */
#define YYNRULES  239
/* YYNRULES -- Number of states. */
#define YYNSTATES  478

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   316

#define YYTRANSLATE(YYX) 						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const unsigned char yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    72,     2,
      70,    71,    69,     2,    66,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    63,    62,
      67,    75,    68,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    73,     2,    74,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    64,     2,    65,     2,     2,     2,     2,
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
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const unsigned short yyprhs[] =
{
       0,     0,     3,     5,     6,     9,    10,    12,    13,    15,
      17,    19,    24,    28,    32,    34,    39,    40,    43,    49,
      52,    55,    59,    62,    65,    68,    71,    74,    76,    78,
      80,    82,    84,    86,    90,    91,    93,    94,    98,   100,
     102,   104,   106,   109,   112,   115,   118,   121,   123,   125,
     128,   130,   133,   136,   138,   140,   143,   146,   149,   158,
     160,   162,   164,   166,   169,   172,   175,   177,   179,   181,
     185,   186,   189,   194,   200,   201,   203,   204,   208,   210,
     214,   216,   218,   219,   223,   225,   229,   231,   232,   236,
     238,   242,   244,   246,   252,   254,   257,   261,   268,   269,
     272,   274,   278,   284,   290,   296,   302,   307,   311,   318,
     325,   331,   337,   343,   349,   355,   360,   368,   369,   372,
     373,   376,   379,   383,   386,   390,   392,   396,   401,   404,
     407,   410,   413,   416,   418,   423,   424,   427,   430,   433,
     436,   439,   443,   447,   451,   458,   462,   469,   473,   480,
     482,   486,   488,   491,   493,   501,   507,   509,   511,   512,
     516,   518,   522,   524,   526,   528,   530,   532,   534,   536,
     538,   540,   542,   544,   546,   548,   549,   551,   555,   556,
     558,   564,   570,   576,   581,   585,   587,   589,   591,   594,
     599,   603,   605,   609,   613,   616,   617,   621,   622,   624,
     628,   630,   633,   635,   638,   639,   644,   646,   650,   652,
     653,   660,   669,   674,   678,   684,   689,   701,   711,   724,
     739,   746,   755,   761,   769,   773,   774,   777,   782,   784,
     788,   790,   792,   795,   801,   803,   807,   809,   811,   814
};

/* YYRHS -- A `-1'-separated list of the rules' RHS. */
static const short yyrhs[] =
{
      77,     0,    -1,    78,    -1,    -1,    83,    78,    -1,    -1,
       5,    -1,    -1,    62,    -1,    49,    -1,    49,    -1,    82,
      63,    63,    49,    -1,     3,    81,    84,    -1,     4,    81,
      84,    -1,    62,    -1,    64,    85,    65,    80,    -1,    -1,
      86,    85,    -1,    79,    64,    85,    65,    80,    -1,    79,
      83,    -1,    79,   138,    -1,    79,   117,    62,    -1,    79,
     120,    -1,    79,   121,    -1,    79,   122,    -1,    79,   124,
      -1,    79,   135,    -1,   174,    -1,   175,    -1,    99,    -1,
      50,    -1,    51,    -1,    87,    -1,    87,    66,    88,    -1,
      -1,    88,    -1,    -1,    67,    89,    68,    -1,    55,    -1,
      56,    -1,    57,    -1,    58,    -1,    61,    55,    -1,    61,
      56,    -1,    61,    57,    -1,    61,    58,    -1,    56,    56,
      -1,    59,    -1,    60,    -1,    56,    60,    -1,    32,    -1,
      81,    90,    -1,    82,    90,    -1,    91,    -1,    93,    -1,
      94,    69,    -1,    95,    69,    -1,    96,    69,    -1,    98,
      70,    69,    81,    71,    70,   156,    71,    -1,    94,    -1,
      95,    -1,    96,    -1,    97,    -1,    33,    98,    -1,    98,
      33,    -1,    98,    72,    -1,    98,    -1,    50,    -1,    82,
      -1,    73,   100,    74,    -1,    -1,   101,   102,    -1,     6,
      99,    82,   102,    -1,     6,    16,    94,    69,    81,    -1,
      -1,    32,    -1,    -1,    73,   107,    74,    -1,   108,    -1,
     108,    66,   107,    -1,    34,    -1,    35,    -1,    -1,    73,
     110,    74,    -1,   114,    -1,   114,    66,   110,    -1,    48,
      -1,    -1,    73,   113,    74,    -1,   111,    -1,   111,    66,
     113,    -1,    27,    -1,    48,    -1,    99,    81,    73,    74,
      62,    -1,   115,    -1,   115,   116,    -1,    16,   106,    92,
      -1,    16,   106,    92,    64,   116,    65,    -1,    -1,    63,
     119,    -1,    92,    -1,    92,    66,   119,    -1,    11,   109,
      92,   118,   136,    -1,    12,   109,    92,   118,   136,    -1,
      13,   109,    92,   118,   136,    -1,    14,   109,    92,   118,
     136,    -1,    73,    50,    81,    74,    -1,    73,    81,    74,
      -1,    15,   112,   123,    92,   118,   136,    -1,    15,   123,
     112,    92,   118,   136,    -1,    11,   109,    81,   118,   136,
      -1,    12,   109,    81,   118,   136,    -1,    13,   109,    81,
     118,   136,    -1,    14,   109,    81,   118,   136,    -1,    15,
     123,    81,   118,   136,    -1,    16,   106,    81,    62,    -1,
      16,   106,    81,    64,   116,    65,    62,    -1,    -1,    75,
      99,    -1,    -1,    75,    50,    -1,    75,    51,    -1,    17,
      81,   130,    -1,    97,   131,    -1,    99,    81,   131,    -1,
     132,    -1,   132,    66,   133,    -1,    21,    67,   133,    68,
      -1,   134,   125,    -1,   134,   126,    -1,   134,   127,    -1,
     134,   128,    -1,   134,   129,    -1,    62,    -1,    64,   137,
      65,    80,    -1,    -1,   143,   137,    -1,   103,    62,    -1,
     104,    62,    -1,   140,    62,    -1,   139,    62,    -1,    10,
     141,    62,    -1,    18,   142,    62,    -1,     8,   105,    82,
      -1,     8,   105,    82,    70,   105,    71,    -1,     7,   105,
      82,    -1,     7,   105,    82,    70,   105,    71,    -1,     9,
     105,    82,    -1,     9,   105,    82,    70,   105,    71,    -1,
      82,    -1,    82,    66,   141,    -1,    51,    -1,   144,    62,
      -1,   138,    -1,    36,   146,   145,    81,   157,   158,   159,
      -1,    36,   146,    81,   157,   159,    -1,    32,    -1,    95,
      -1,    -1,    73,   147,    74,    -1,   148,    -1,   148,    66,
     147,    -1,    20,    -1,    22,    -1,    23,    -1,    28,    -1,
      29,    -1,    30,    -1,    31,    -1,    24,    -1,    25,    -1,
      48,    -1,    51,    -1,    50,    -1,    82,    -1,    -1,    52,
      -1,    52,    66,   150,    -1,    -1,    52,    -1,    52,    73,
     151,    74,   151,    -1,    52,    64,   151,    65,   151,    -1,
      52,    70,   150,    71,   151,    -1,    70,   151,    71,   151,
      -1,    99,    81,    73,    -1,    64,    -1,    65,    -1,    99,
      -1,    99,    81,    -1,    99,    81,    75,   149,    -1,   152,
     151,    74,    -1,   155,    -1,   155,    66,   156,    -1,    70,
     156,    71,    -1,    70,    71,    -1,    -1,    19,    75,    50,
      -1,    -1,   165,    -1,    64,   160,    65,    -1,   165,    -1,
     165,   160,    -1,   165,    -1,   165,   160,    -1,    -1,    47,
      70,   163,    71,    -1,    49,    -1,    49,    66,   163,    -1,
      51,    -1,    -1,    42,   164,   153,   151,   154,   162,    -1,
      46,    70,    49,   157,    71,   153,   151,    65,    -1,    40,
     171,    64,    65,    -1,    40,   171,   165,    -1,    40,   171,
      64,   160,    65,    -1,    41,    64,   161,    65,    -1,    37,
     169,   151,    62,   151,    62,   151,   168,    64,   160,    65,
      -1,    37,   169,   151,    62,   151,    62,   151,   168,   165,
      -1,    38,    73,    49,    74,   169,   151,    63,   151,    66,
     151,   168,   165,    -1,    38,    73,    49,    74,   169,   151,
      63,   151,    66,   151,   168,    64,   160,    65,    -1,    44,
     169,   151,   168,   165,   166,    -1,    44,   169,   151,   168,
      64,   160,    65,   166,    -1,    39,   169,   151,   168,   165,
      -1,    39,   169,   151,   168,    64,   160,    65,    -1,    43,
     167,    62,    -1,    -1,    45,   165,    -1,    45,    64,   160,
      65,    -1,    49,    -1,    49,    66,   167,    -1,    71,    -1,
      70,    -1,    49,   157,    -1,    49,   172,   151,   173,   157,
      -1,   170,    -1,   170,    66,   171,    -1,    73,    -1,    74,
      -1,    53,    81,    -1,    54,    81,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const unsigned short yyrline[] =
{
       0,   132,   132,   137,   140,   145,   146,   151,   152,   156,
     160,   162,   170,   174,   181,   183,   188,   189,   193,   195,
     197,   199,   201,   203,   205,   207,   209,   211,   213,   217,
     219,   221,   225,   227,   232,   233,   238,   239,   243,   245,
     247,   249,   251,   253,   255,   257,   259,   261,   263,   265,
     267,   271,   272,   274,   276,   280,   284,   286,   290,   294,
     296,   298,   300,   303,   305,   309,   311,   315,   317,   321,
     326,   327,   331,   335,   340,   341,   346,   347,   357,   359,
     363,   365,   370,   371,   375,   377,   381,   386,   387,   391,
     393,   397,   399,   403,   407,   409,   413,   415,   420,   421,
     425,   427,   431,   433,   437,   441,   445,   451,   455,   457,
     461,   463,   467,   471,   475,   479,   481,   486,   487,   492,
     493,   495,   499,   501,   503,   507,   509,   513,   517,   519,
     521,   523,   525,   529,   531,   536,   554,   558,   560,   562,
     563,   565,   567,   571,   573,   575,   578,   583,   585,   589,
     591,   595,   599,   601,   605,   616,   629,   631,   636,   637,
     641,   643,   647,   649,   651,   653,   655,   657,   659,   661,
     663,   665,   669,   671,   673,   678,   679,   681,   690,   691,
     693,   699,   705,   711,   719,   726,   734,   741,   743,   745,
     747,   754,   756,   760,   762,   767,   768,   773,   774,   776,
     780,   782,   786,   788,   793,   794,   798,   800,   804,   807,
     810,   815,   829,   831,   833,   835,   837,   840,   843,   846,
     849,   851,   853,   855,   857,   862,   863,   865,   868,   870,
     874,   878,   882,   890,   898,   900,   904,   907,   911,   915
};
#endif

#if YYDEBUG || YYERROR_VERBOSE
/* YYTNME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals. */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "MODULE", "MAINMODULE", "EXTERN", 
  "READONLY", "INITCALL", "INITNODE", "INITPROC", "PUPABLE", "CHARE", 
  "MAINCHARE", "GROUP", "NODEGROUP", "ARRAY", "MESSAGE", "CLASS", 
  "INCLUDE", "STACKSIZE", "THREADED", "TEMPLATE", "SYNC", "EXCLUSIVE", 
  "IMMEDIATE", "SKIPSCHED", "VIRTUAL", "MIGRATABLE", "CREATEHERE", 
  "CREATEHOME", "NOKEEP", "NOTRACE", "VOID", "CONST", "PACKED", "VARSIZE", 
  "ENTRY", "FOR", "FORALL", "WHILE", "WHEN", "OVERLAP", "ATOMIC", 
  "FORWARD", "IF", "ELSE", "CONNECT", "PUBLISHES", "PYTHON", "IDENT", 
  "NUMBER", "LITERAL", "CPROGRAM", "HASHIF", "HASHIFDEF", "INT", "LONG", 
  "SHORT", "CHAR", "FLOAT", "DOUBLE", "UNSIGNED", "';'", "':'", "'{'", 
  "'}'", "','", "'<'", "'>'", "'*'", "'('", "')'", "'&'", "'['", "']'", 
  "'='", "$accept", "File", "ModuleEList", "OptExtern", "OptSemiColon", 
  "Name", "QualName", "Module", "ConstructEList", "ConstructList", 
  "Construct", "TParam", "TParamList", "TParamEList", "OptTParams", 
  "BuiltinType", "NamedType", "QualNamedType", "SimpleType", "OnePtrType", 
  "PtrType", "FuncType", "BaseType", "Type", "ArrayDim", "Dim", "DimList", 
  "Readonly", "ReadonlyMsg", "OptVoid", "MAttribs", "MAttribList", 
  "MAttrib", "CAttribs", "CAttribList", "ArrayAttrib", "ArrayAttribs", 
  "ArrayAttribList", "CAttrib", "Var", "VarList", "Message", 
  "OptBaseList", "BaseList", "Chare", "Group", "NodeGroup", 
  "ArrayIndexType", "Array", "TChare", "TGroup", "TNodeGroup", "TArray", 
  "TMessage", "OptTypeInit", "OptNameInit", "TVar", "TVarList", 
  "TemplateSpec", "Template", "MemberEList", "MemberList", 
  "NonEntryMember", "InitNode", "InitProc", "PUPableClass", "IncludeFile", 
  "Member", "Entry", "EReturn", "EAttribs", "EAttribList", "EAttrib", 
  "DefaultParameter", "CPROGRAM_List", "CCode", "ParamBracketStart", 
  "ParamBraceStart", "ParamBraceEnd", "Parameter", "ParamList", 
  "EParameters", "OptStackSize", "OptSdagCode", "Slist", "Olist", 
  "OptPubList", "PublishesList", "OptTraceName", "SingleConstruct", 
  "HasElse", "ForwardList", "EndIntExpr", "StartIntExpr", "SEntry", 
  "SEntryList", "SParamBracketStart", "SParamBracketEnd", "HashIFComment", 
  "HashIFDefComment", 0
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
static const unsigned short yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
     315,   316,    59,    58,   123,   125,    44,    60,    62,    42,
      40,    41,    38,    91,    93,    61
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const unsigned char yyr1[] =
{
       0,    76,    77,    78,    78,    79,    79,    80,    80,    81,
      82,    82,    83,    83,    84,    84,    85,    85,    86,    86,
      86,    86,    86,    86,    86,    86,    86,    86,    86,    87,
      87,    87,    88,    88,    89,    89,    90,    90,    91,    91,
      91,    91,    91,    91,    91,    91,    91,    91,    91,    91,
      91,    92,    93,    94,    94,    95,    96,    96,    97,    98,
      98,    98,    98,    98,    98,    99,    99,   100,   100,   101,
     102,   102,   103,   104,   105,   105,   106,   106,   107,   107,
     108,   108,   109,   109,   110,   110,   111,   112,   112,   113,
     113,   114,   114,   115,   116,   116,   117,   117,   118,   118,
     119,   119,   120,   120,   121,   122,   123,   123,   124,   124,
     125,   125,   126,   127,   128,   129,   129,   130,   130,   131,
     131,   131,   132,   132,   132,   133,   133,   134,   135,   135,
     135,   135,   135,   136,   136,   137,   137,   138,   138,   138,
     138,   138,   138,   139,   139,   139,   139,   140,   140,   141,
     141,   142,   143,   143,   144,   144,   145,   145,   146,   146,
     147,   147,   148,   148,   148,   148,   148,   148,   148,   148,
     148,   148,   149,   149,   149,   150,   150,   150,   151,   151,
     151,   151,   151,   151,   152,   153,   154,   155,   155,   155,
     155,   156,   156,   157,   157,   158,   158,   159,   159,   159,
     160,   160,   161,   161,   162,   162,   163,   163,   164,   164,
     165,   165,   165,   165,   165,   165,   165,   165,   165,   165,
     165,   165,   165,   165,   165,   166,   166,   166,   167,   167,
     168,   169,   170,   170,   171,   171,   172,   173,   174,   175
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const unsigned char yyr2[] =
{
       0,     2,     1,     0,     2,     0,     1,     0,     1,     1,
       1,     4,     3,     3,     1,     4,     0,     2,     5,     2,
       2,     3,     2,     2,     2,     2,     2,     1,     1,     1,
       1,     1,     1,     3,     0,     1,     0,     3,     1,     1,
       1,     1,     2,     2,     2,     2,     2,     1,     1,     2,
       1,     2,     2,     1,     1,     2,     2,     2,     8,     1,
       1,     1,     1,     2,     2,     2,     1,     1,     1,     3,
       0,     2,     4,     5,     0,     1,     0,     3,     1,     3,
       1,     1,     0,     3,     1,     3,     1,     0,     3,     1,
       3,     1,     1,     5,     1,     2,     3,     6,     0,     2,
       1,     3,     5,     5,     5,     5,     4,     3,     6,     6,
       5,     5,     5,     5,     5,     4,     7,     0,     2,     0,
       2,     2,     3,     2,     3,     1,     3,     4,     2,     2,
       2,     2,     2,     1,     4,     0,     2,     2,     2,     2,
       2,     3,     3,     3,     6,     3,     6,     3,     6,     1,
       3,     1,     2,     1,     7,     5,     1,     1,     0,     3,
       1,     3,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     0,     1,     3,     0,     1,
       5,     5,     5,     4,     3,     1,     1,     1,     2,     4,
       3,     1,     3,     3,     2,     0,     3,     0,     1,     3,
       1,     2,     1,     2,     0,     4,     1,     3,     1,     0,
       6,     8,     4,     3,     5,     4,    11,     9,    12,    14,
       6,     8,     5,     7,     3,     0,     2,     4,     1,     3,
       1,     1,     2,     5,     1,     3,     1,     1,     2,     2
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const unsigned char yydefact[] =
{
       3,     0,     0,     0,     2,     3,     9,     0,     0,     1,
       4,    14,     5,    12,    13,     6,     0,     0,     0,     0,
       5,    27,    28,   238,   239,     0,    74,    74,    74,     0,
      82,    82,    82,    82,     0,    76,     0,     0,     5,    19,
       0,     0,     0,    22,    23,    24,    25,     0,    26,    20,
       0,     0,     7,    17,     0,    50,     0,    10,    38,    39,
      40,    41,    47,    48,     0,    36,    53,    54,    59,    60,
      61,    62,    66,     0,    75,     0,     0,     0,   149,     0,
       0,     0,     0,     0,     0,     0,     0,    87,     0,     0,
     151,     0,     0,     0,   137,   138,    21,    82,    82,    82,
      82,     0,    76,   128,   129,   130,   131,   132,   140,   139,
       8,    15,     0,    63,    46,    49,    42,    43,    44,    45,
       0,    34,    52,    55,    56,    57,    64,     0,    65,    70,
     145,   143,   147,     0,   141,    91,    92,     0,    84,    36,
      98,    98,    98,    98,    86,     0,     0,    89,     0,     0,
       0,     0,     0,    80,    81,     0,    78,    96,   142,     0,
      62,     0,   125,     0,     7,     0,     0,     0,     0,     0,
       0,     0,     0,    30,    31,    32,    35,     0,    29,     0,
       0,    70,    72,    74,    74,    74,   150,    83,     0,    51,
       0,     0,     0,     0,     0,     0,   107,     0,    88,    98,
      98,    77,     0,     0,   117,     0,   123,   119,     0,   127,
      18,    98,    98,    98,    98,    98,     0,    73,    11,     0,
      37,     0,    67,    68,     0,    71,     0,     0,     0,    85,
     100,    99,   133,   135,   102,   103,   104,   105,   106,    90,
       0,     0,    79,     0,    94,     0,     0,   122,   120,   121,
     124,   126,     0,     0,     0,     0,     0,   115,     0,    33,
       0,    69,   146,   144,   148,     0,   158,     0,   153,   135,
       0,   108,   109,     0,    95,    97,   118,   110,   111,   112,
     113,   114,     0,     0,   101,     0,     0,     7,   136,   152,
       0,     0,   187,   178,   191,     0,   162,   163,   164,   169,
     170,   165,   166,   167,   168,   171,     0,   160,    50,    10,
       0,     0,   157,     0,   134,     0,   116,   188,   179,   178,
       0,     0,    58,   159,     0,     0,   197,     0,    93,   184,
       0,   178,   175,   178,     0,   190,   192,   161,   194,     0,
       0,     0,     0,     0,     0,   209,     0,     0,     0,     0,
     155,   198,   195,   173,   172,   174,   189,     0,   176,     0,
       0,   178,   193,   231,   178,     0,   178,     0,   234,     0,
       0,   208,     0,   228,     0,   178,     0,     0,   200,     0,
     197,   178,   175,   178,   178,   183,     0,     0,     0,   236,
     232,   178,     0,     0,   213,     0,   202,   185,   178,     0,
     224,     0,     0,   199,   201,     0,   154,   181,   177,   182,
     180,   178,     0,   230,     0,     0,   235,   212,     0,   215,
     203,     0,   229,     0,     0,   196,     0,   178,     0,   222,
     237,     0,   214,   186,   204,     0,   225,     0,   178,     0,
       0,   233,     0,   210,     0,     0,   220,   178,     0,   178,
     223,     0,   225,     0,   226,     0,     0,     0,   206,     0,
     221,     0,   211,     0,   217,   178,     0,   205,   227,     0,
       0,   207,   216,     0,     0,   218,     0,   219
};

/* YYDEFGOTO[NTERM-NUM]. */
static const short yydefgoto[] =
{
      -1,     3,     4,    18,   111,   139,    65,     5,    13,    19,
      20,   175,   176,   177,   122,    66,   230,    67,    68,    69,
      70,    71,    72,   243,   224,   181,   182,    40,    41,    75,
      89,   155,   156,    81,   137,   147,    86,   148,   138,   244,
     245,    42,   191,   231,    43,    44,    45,    87,    46,   103,
     104,   105,   106,   107,   247,   206,   162,   163,    47,    48,
     234,   267,   268,    50,    51,    79,    91,   269,   270,   313,
     286,   306,   307,   356,   359,   320,   293,   398,   434,   294,
     295,   326,   380,   350,   377,   395,   443,   459,   372,   378,
     446,   374,   414,   364,   368,   369,   391,   431,    21,    22
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -392
static const short yypact[] =
{
     176,   -19,   -19,    64,  -392,   176,  -392,    78,    78,  -392,
    -392,  -392,    15,  -392,  -392,  -392,   -19,   -19,   193,     1,
      15,  -392,  -392,  -392,  -392,   217,    46,    46,    46,    36,
      26,    26,    26,    26,    39,    70,   108,   110,    15,  -392,
     124,   148,   153,  -392,  -392,  -392,  -392,   160,  -392,  -392,
     154,   155,   161,  -392,   300,  -392,   282,  -392,  -392,    51,
    -392,  -392,  -392,  -392,    79,   -34,  -392,  -392,   157,   158,
     159,  -392,   -11,    36,  -392,    36,    36,    36,    28,   167,
      33,   -19,   -19,   -19,   -19,    72,   151,   163,   147,   -19,
    -392,   172,   237,   173,  -392,  -392,  -392,    26,    26,    26,
      26,   151,    70,  -392,  -392,  -392,  -392,  -392,  -392,  -392,
    -392,  -392,   168,   -15,  -392,  -392,  -392,  -392,  -392,  -392,
     177,   269,  -392,  -392,  -392,  -392,  -392,   170,  -392,    14,
     -36,     2,    23,    36,  -392,  -392,  -392,   178,   175,   184,
     190,   190,   190,   190,  -392,   -19,   181,   192,   182,   139,
     -19,   211,   -19,  -392,  -392,   186,   195,   199,  -392,   -19,
      29,   -19,   198,   197,   161,   -19,   -19,   -19,   -19,   -19,
     -19,   -19,   218,  -392,  -392,   202,  -392,   203,  -392,   -19,
     145,   206,  -392,    46,    46,    46,  -392,  -392,    33,  -392,
     -19,    87,    87,    87,    87,   207,  -392,   211,  -392,   190,
     190,  -392,   147,   282,   205,   162,  -392,   209,   237,  -392,
    -392,   190,   190,   190,   190,   190,    88,  -392,  -392,   269,
    -392,   216,  -392,   225,   215,  -392,   219,   242,   245,  -392,
     251,  -392,  -392,   212,  -392,  -392,  -392,  -392,  -392,  -392,
      87,    87,  -392,   -19,   282,   257,   282,  -392,  -392,  -392,
    -392,  -392,    87,    87,    87,    87,    87,  -392,   282,  -392,
     253,  -392,  -392,  -392,  -392,   -19,   260,   270,  -392,   212,
     272,  -392,  -392,   263,  -392,  -392,  -392,  -392,  -392,  -392,
    -392,  -392,   279,   282,  -392,   355,   313,   161,  -392,  -392,
     273,   284,   -19,   -33,   285,   277,  -392,  -392,  -392,  -392,
    -392,  -392,  -392,  -392,  -392,  -392,   276,   286,   304,   293,
     294,   157,  -392,   -19,  -392,   292,  -392,    81,   -38,   -33,
     291,   282,  -392,  -392,   355,   250,   367,   294,  -392,  -392,
      96,   -33,   314,   -33,   296,  -392,  -392,  -392,  -392,   305,
     311,   309,   311,   346,   333,   347,   351,   311,   329,   463,
    -392,  -392,   382,  -392,  -392,   225,  -392,   337,   348,   341,
     352,   -33,  -392,  -392,   -33,   375,   -33,    58,   361,   379,
     463,  -392,   364,   363,   368,   -33,   391,   377,   463,   380,
     367,   -33,   314,   -33,   -33,  -392,   390,   383,   385,  -392,
    -392,   -33,   346,   350,  -392,   389,   463,  -392,   -33,   351,
    -392,   385,   294,  -392,  -392,   408,  -392,  -392,  -392,  -392,
    -392,   -33,   311,  -392,   395,   394,  -392,  -392,   405,  -392,
    -392,   415,  -392,   407,   411,  -392,   421,   -33,   463,  -392,
    -392,   294,  -392,  -392,   437,   463,   440,   364,   -33,   433,
     443,  -392,   416,  -392,   445,   423,  -392,   -33,   385,   -33,
    -392,   449,   440,   463,  -392,   446,   435,   447,   448,   441,
    -392,   452,  -392,   463,  -392,   -33,   449,  -392,  -392,   453,
     385,  -392,  -392,   451,   463,  -392,   454,  -392
};

/* YYPGOTO[NTERM-NUM].  */
static const short yypgoto[] =
{
    -392,  -392,   511,  -392,  -155,    -1,   -27,   502,   513,    54,
    -392,  -392,   303,  -392,   384,  -392,   -42,  -392,   -51,   238,
    -392,   -85,   469,   -21,  -392,  -392,   345,  -392,  -392,   -22,
     425,   326,  -392,    93,   342,  -392,   442,   334,  -392,  -392,
    -220,  -392,   -98,   267,  -392,  -392,  -392,   -65,  -392,  -392,
    -392,  -392,  -392,  -392,  -392,   327,  -392,   325,  -392,  -392,
      -9,   266,   518,  -392,  -392,   404,  -392,  -392,  -392,  -392,
    -392,   214,  -392,  -392,   164,  -308,  -392,   102,  -392,  -392,
    -192,  -313,  -392,   165,  -365,  -392,  -392,    74,  -392,  -318,
      89,   143,  -391,  -330,  -392,   152,  -392,  -392,  -392,  -392
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -157
static const short yytable[] =
{
       7,     8,    78,   112,    73,    76,    77,   160,   351,   210,
     423,   334,   366,   404,   352,    23,    24,   375,   126,   318,
      15,   150,   126,   357,   274,   360,   331,   120,   418,   120,
       6,   420,   332,   121,   183,   333,   169,   319,   282,   140,
     141,   142,   143,   192,   193,   194,   129,   157,   130,   131,
     132,   394,   396,   385,   390,   127,   386,   456,   388,   127,
     135,   128,   351,   440,     9,   120,    52,   401,    16,    17,
     444,   161,   184,   407,    53,   409,   410,   120,    74,   473,
     -16,   136,   427,   415,   146,    57,   120,   180,   461,   424,
     421,   120,    93,   185,   133,  -119,   429,  -119,   469,    80,
     178,   240,   241,   426,   205,   436,    78,   114,   199,   476,
     200,   115,    85,   252,   253,   254,   255,   256,   441,   439,
     144,     6,   145,   160,    82,    83,    84,   454,   325,   336,
     448,   389,   314,   339,   116,   117,   118,   119,   464,   455,
      11,   457,    12,    88,   195,    57,   353,   354,   146,   232,
     257,   233,   258,   223,   329,   475,   330,   470,   204,    90,
     207,   226,   227,   228,   211,   212,   213,   214,   215,   216,
     217,    97,    98,    99,   100,   101,   102,    92,   221,     1,
       2,   153,   154,   235,   236,   237,    94,   161,     6,   145,
     165,   166,   167,   168,    57,   222,     1,     2,   178,    25,
      26,    27,    28,    29,    30,    31,    32,    33,    34,    35,
      95,    36,   248,   249,    37,    96,   108,   109,    25,    26,
      27,    28,    29,   110,   149,   276,   123,   124,   125,   134,
      36,   271,   272,    54,   158,   311,   151,   171,   164,   179,
     172,   188,   273,   277,   278,   279,   280,   281,   266,    55,
      56,   121,   187,   190,   159,   196,   198,    38,   197,   144,
     201,   202,   292,   203,   208,   209,    57,   218,   219,    55,
      56,   220,    58,    59,    60,    61,    62,    63,    64,   180,
     246,   238,    55,    56,   205,   310,    57,   260,   120,   261,
     262,   317,    58,    59,    60,    61,    62,    63,    64,    57,
     292,    55,    56,   355,   292,    58,    59,    60,    61,    62,
      63,    64,   327,   263,    55,    56,   264,   265,    57,   173,
     174,   338,   275,   283,    58,    59,    60,    61,    62,    63,
      64,    57,    55,   285,   289,   287,   290,    58,    59,    60,
      61,    62,    63,    64,   291,   308,   316,   315,   322,    57,
     323,   321,   324,  -156,   328,    58,    59,    60,    61,    62,
      63,    64,   309,    -9,   325,   335,   358,   361,    58,    59,
      60,    61,    62,    63,    64,   296,   362,   297,   298,   299,
     300,   363,   365,   301,   302,   303,   304,   340,   341,   342,
     343,   344,   345,   346,   347,   367,   348,   370,   371,   376,
     373,   379,   381,   305,   340,   341,   342,   343,   344,   345,
     346,   347,   383,   348,   382,   417,   340,   341,   342,   343,
     344,   345,   346,   347,   387,   348,   384,   392,   397,   399,
     400,   349,   340,   341,   342,   343,   344,   345,   346,   347,
     402,   348,   403,   393,   340,   341,   342,   343,   344,   345,
     346,   347,   411,   348,   419,   405,   413,   412,   425,   428,
     340,   341,   342,   343,   344,   345,   346,   347,   430,   348,
     432,   435,   340,   341,   342,   343,   344,   345,   346,   347,
     433,   348,   437,   438,   442,   445,   451,   453,   340,   341,
     342,   343,   344,   345,   346,   347,   449,   348,   458,   463,
     340,   341,   342,   343,   344,   345,   346,   347,   450,   348,
     452,   462,   467,   465,   466,   474,    10,   468,   472,   477,
      39,    14,   259,   189,   312,   113,   225,   170,   242,   152,
     229,   239,   284,   251,   250,   288,    49,   186,   337,   447,
     471,   460,   422,     0,   416,   406,   408
};

static const short yycheck[] =
{
       1,     2,    29,    54,    25,    27,    28,    92,   326,   164,
     401,   319,   342,   378,   327,    16,    17,   347,    33,    52,
       5,    86,    33,   331,   244,   333,    64,    63,   393,    63,
      49,   396,    70,    67,    70,    73,   101,    70,   258,    81,
      82,    83,    84,   141,   142,   143,    73,    89,    75,    76,
      77,   369,   370,   361,   367,    70,   364,   448,   366,    70,
      27,    72,   380,   428,     0,    63,    65,   375,    53,    54,
     435,    92,    70,   381,    20,   383,   384,    63,    32,   470,
      65,    48,   412,   391,    85,    49,    63,    73,   453,   402,
     398,    63,    38,    70,    66,    66,   414,    68,   463,    73,
     121,   199,   200,   411,    75,   423,   133,    56,   150,   474,
     152,    60,    73,   211,   212,   213,   214,   215,   431,   427,
      48,    49,    50,   208,    31,    32,    33,   445,    70,   321,
     438,    73,   287,   325,    55,    56,    57,    58,   456,   447,
      62,   449,    64,    73,   145,    49,    50,    51,   149,    62,
      62,    64,    64,   180,    73,   473,    75,   465,   159,    51,
     161,   183,   184,   185,   165,   166,   167,   168,   169,   170,
     171,    11,    12,    13,    14,    15,    16,    67,   179,     3,
       4,    34,    35,   192,   193,   194,    62,   208,    49,    50,
      97,    98,    99,   100,    49,    50,     3,     4,   219,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      62,    18,    50,    51,    21,    62,    62,    62,     6,     7,
       8,     9,    10,    62,    73,   246,    69,    69,    69,    62,
      18,   240,   241,    16,    62,   286,    73,    69,    65,    69,
      63,    66,   243,   252,   253,   254,   255,   256,    36,    32,
      33,    67,    74,    63,    17,    74,    74,    64,    66,    48,
      74,    66,   283,    64,    66,    68,    49,    49,    66,    32,
      33,    68,    55,    56,    57,    58,    59,    60,    61,    73,
      75,    74,    32,    33,    75,   286,    49,    71,    63,    74,
      71,   292,    55,    56,    57,    58,    59,    60,    61,    49,
     321,    32,    33,   330,   325,    55,    56,    57,    58,    59,
      60,    61,   313,    71,    32,    33,    71,    66,    49,    50,
      51,    71,    65,    70,    55,    56,    57,    58,    59,    60,
      61,    49,    32,    73,    62,    65,    73,    55,    56,    57,
      58,    59,    60,    61,    65,    32,    62,    74,    71,    49,
      74,    66,    66,    49,    62,    55,    56,    57,    58,    59,
      60,    61,    49,    70,    70,    74,    52,    71,    55,    56,
      57,    58,    59,    60,    61,    20,    71,    22,    23,    24,
      25,    70,    73,    28,    29,    30,    31,    37,    38,    39,
      40,    41,    42,    43,    44,    49,    46,    64,    51,    70,
      49,    19,    65,    48,    37,    38,    39,    40,    41,    42,
      43,    44,    71,    46,    66,    65,    37,    38,    39,    40,
      41,    42,    43,    44,    49,    46,    74,    66,    64,    66,
      62,    64,    37,    38,    39,    40,    41,    42,    43,    44,
      49,    46,    65,    64,    37,    38,    39,    40,    41,    42,
      43,    44,    62,    46,    65,    75,    71,    74,    50,    64,
      37,    38,    39,    40,    41,    42,    43,    44,    74,    46,
      65,    64,    37,    38,    39,    40,    41,    42,    43,    44,
      65,    46,    71,    62,    47,    45,    70,    64,    37,    38,
      39,    40,    41,    42,    43,    44,    63,    46,    49,    64,
      37,    38,    39,    40,    41,    42,    43,    44,    65,    46,
      65,    65,    71,    66,    66,    64,     5,    65,    65,    65,
      18,     8,   219,   139,   286,    56,   181,   102,   202,    87,
     188,   197,   265,   208,   207,   269,    18,   133,   324,   437,
     466,   452,   399,    -1,   392,   380,   382
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const unsigned char yystos[] =
{
       0,     3,     4,    77,    78,    83,    49,    81,    81,     0,
      78,    62,    64,    84,    84,     5,    53,    54,    79,    85,
      86,   174,   175,    81,    81,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    18,    21,    64,    83,
     103,   104,   117,   120,   121,   122,   124,   134,   135,   138,
     139,   140,    65,    85,    16,    32,    33,    49,    55,    56,
      57,    58,    59,    60,    61,    82,    91,    93,    94,    95,
      96,    97,    98,    99,    32,   105,   105,   105,    82,   141,
      73,   109,   109,   109,   109,    73,   112,   123,    73,   106,
      51,   142,    67,    85,    62,    62,    62,    11,    12,    13,
      14,    15,    16,   125,   126,   127,   128,   129,    62,    62,
      62,    80,    94,    98,    56,    60,    55,    56,    57,    58,
      63,    67,    90,    69,    69,    69,    33,    70,    72,    82,
      82,    82,    82,    66,    62,    27,    48,   110,   114,    81,
      92,    92,    92,    92,    48,    50,    81,   111,   113,    73,
     123,    73,   112,    34,    35,   107,   108,    92,    62,    17,
      97,    99,   132,   133,    65,   109,   109,   109,   109,   123,
     106,    69,    63,    50,    51,    87,    88,    89,    99,    69,
      73,   101,   102,    70,    70,    70,   141,    74,    66,    90,
      63,   118,   118,   118,   118,    81,    74,    66,    74,    92,
      92,    74,    66,    64,    81,    75,   131,    81,    66,    68,
      80,    81,    81,    81,    81,    81,    81,    81,    49,    66,
      68,    81,    50,    82,   100,   102,   105,   105,   105,   110,
      92,   119,    62,    64,   136,   136,   136,   136,    74,   113,
     118,   118,   107,    99,   115,   116,    75,   130,    50,    51,
     131,   133,   118,   118,   118,   118,   118,    62,    64,    88,
      71,    74,    71,    71,    71,    66,    36,   137,   138,   143,
     144,   136,   136,    81,   116,    65,    99,   136,   136,   136,
     136,   136,   116,    70,   119,    73,   146,    65,   137,    62,
      73,    65,    99,   152,   155,   156,    20,    22,    23,    24,
      25,    28,    29,    30,    31,    48,   147,   148,    32,    49,
      81,    94,    95,   145,    80,    74,    62,    81,    52,    70,
     151,    66,    71,    74,    66,    70,   157,    81,    62,    73,
      75,    64,    70,    73,   151,    74,   156,   147,    71,   156,
      37,    38,    39,    40,    41,    42,    43,    44,    46,    64,
     159,   165,   157,    50,    51,    82,   149,   151,    52,   150,
     151,    71,    71,    70,   169,    73,   169,    49,   170,   171,
      64,    51,   164,    49,   167,   169,    70,   160,   165,    19,
     158,    65,    66,    71,    74,   151,   151,    49,   151,    73,
     157,   172,    66,    64,   165,   161,   165,    64,   153,    66,
      62,   151,    49,    65,   160,    75,   159,   151,   150,   151,
     151,    62,    74,    71,   168,   151,   171,    65,   160,    65,
     160,   151,   167,   168,   157,    50,   151,   169,    64,   165,
      74,   173,    65,    65,   154,    64,   165,    71,    62,   151,
     160,   157,    47,   162,   160,    45,   166,   153,   151,    63,
      65,    70,    65,    64,   165,   151,   168,   151,    49,   163,
     166,   160,    65,    64,   165,    66,    66,    71,    65,   160,
     151,   163,    65,   168,    64,   165,   160,    65
};

#if ! defined (YYSIZE_T) && defined (__SIZE_TYPE__)
# define YYSIZE_T __SIZE_TYPE__
#endif
#if ! defined (YYSIZE_T) && defined (size_t)
# define YYSIZE_T size_t
#endif
#if ! defined (YYSIZE_T)
# if defined (__STDC__) || defined (__cplusplus)
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# endif
#endif
#if ! defined (YYSIZE_T)
# define YYSIZE_T unsigned int
#endif

#define yyerrok		(yyerrstatus = 0)
#define yyclearin	(yychar = YYEMPTY)
#define YYEMPTY		(-2)
#define YYEOF		0

#define YYACCEPT	goto yyacceptlab
#define YYABORT		goto yyabortlab
#define YYERROR		goto yyerrlab1

/* Like YYERROR except do call yyerror.  This remains here temporarily
   to ease the transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  */

#define YYFAIL		goto yyerrlab

#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)					\
do								\
  if (yychar == YYEMPTY && yylen == 1)				\
    {								\
      yychar = (Token);						\
      yylval = (Value);						\
      yytoken = YYTRANSLATE (yychar);				\
      YYPOPSTACK;						\
      goto yybackup;						\
    }								\
  else								\
    { 								\
      yyerror ("syntax error: cannot back up");\
      YYERROR;							\
    }								\
while (0)

#define YYTERROR	1
#define YYERRCODE	256

/* YYLLOC_DEFAULT -- Compute the default location (before the actions
   are run).  */

#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)         \
  Current.first_line   = Rhs[1].first_line;      \
  Current.first_column = Rhs[1].first_column;    \
  Current.last_line    = Rhs[N].last_line;       \
  Current.last_column  = Rhs[N].last_column;
#endif

/* YYLEX -- calling `yylex' with the right arguments.  */

#ifdef YYLEX_PARAM
# define YYLEX yylex (YYLEX_PARAM)
#else
# define YYLEX yylex ()
#endif

/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)			\
do {						\
  if (yydebug)					\
    YYFPRINTF Args;				\
} while (0)

# define YYDSYMPRINT(Args)			\
do {						\
  if (yydebug)					\
    yysymprint Args;				\
} while (0)

# define YYDSYMPRINTF(Title, Token, Value, Location)		\
do {								\
  if (yydebug)							\
    {								\
      YYFPRINTF (stderr, "%s ", Title);				\
      yysymprint (stderr, 					\
                  Token, Value);	\
      YYFPRINTF (stderr, "\n");					\
    }								\
} while (0)

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (cinluded).                                                   |
`------------------------------------------------------------------*/

#if defined (__STDC__) || defined (__cplusplus)
static void
yy_stack_print (short *bottom, short *top)
#else
static void
yy_stack_print (bottom, top)
    short *bottom;
    short *top;
#endif
{
  YYFPRINTF (stderr, "Stack now");
  for (/* Nothing. */; bottom <= top; ++bottom)
    YYFPRINTF (stderr, " %d", *bottom);
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)				\
do {								\
  if (yydebug)							\
    yy_stack_print ((Bottom), (Top));				\
} while (0)


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

#if defined (__STDC__) || defined (__cplusplus)
static void
yy_reduce_print (int yyrule)
#else
static void
yy_reduce_print (yyrule)
    int yyrule;
#endif
{
  int yyi;
  unsigned int yylineno = yyrline[yyrule];
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %u), ",
             yyrule - 1, yylineno);
  /* Print the symbols being reduced, and their result.  */
  for (yyi = yyprhs[yyrule]; 0 <= yyrhs[yyi]; yyi++)
    YYFPRINTF (stderr, "%s ", yytname [yyrhs[yyi]]);
  YYFPRINTF (stderr, "-> %s\n", yytname [yyr1[yyrule]]);
}

# define YY_REDUCE_PRINT(Rule)		\
do {					\
  if (yydebug)				\
    yy_reduce_print (Rule);		\
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YYDSYMPRINT(Args)
# define YYDSYMPRINTF(Title, Token, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef	YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   SIZE_MAX < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#if YYMAXDEPTH == 0
# undef YYMAXDEPTH
#endif

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif



#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined (__GLIBC__) && defined (_STRING_H)
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
static YYSIZE_T
#   if defined (__STDC__) || defined (__cplusplus)
yystrlen (const char *yystr)
#   else
yystrlen (yystr)
     const char *yystr;
#   endif
{
  register const char *yys = yystr;

  while (*yys++ != '\0')
    continue;

  return yys - yystr - 1;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined (__GLIBC__) && defined (_STRING_H) && defined (_GNU_SOURCE)
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
static char *
#   if defined (__STDC__) || defined (__cplusplus)
yystpcpy (char *yydest, const char *yysrc)
#   else
yystpcpy (yydest, yysrc)
     char *yydest;
     const char *yysrc;
#   endif
{
  register char *yyd = yydest;
  register const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

#endif /* !YYERROR_VERBOSE */



#if YYDEBUG
/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

#if defined (__STDC__) || defined (__cplusplus)
static void
yysymprint (FILE *yyoutput, int yytype, YYSTYPE *yyvaluep)
#else
static void
yysymprint (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE *yyvaluep;
#endif
{
  /* Pacify ``unused variable'' warnings.  */
  (void) yyvaluep;

  if (yytype < YYNTOKENS)
    {
      YYFPRINTF (yyoutput, "token %s (", yytname[yytype]);
# ifdef YYPRINT
      YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# endif
    }
  else
    YYFPRINTF (yyoutput, "nterm %s (", yytname[yytype]);

  switch (yytype)
    {
      default:
        break;
    }
  YYFPRINTF (yyoutput, ")");
}

#endif /* ! YYDEBUG */
/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

#if defined (__STDC__) || defined (__cplusplus)
static void
yydestruct (int yytype, YYSTYPE *yyvaluep)
#else
static void
yydestruct (yytype, yyvaluep)
    int yytype;
    YYSTYPE *yyvaluep;
#endif
{
  /* Pacify ``unused variable'' warnings.  */
  (void) yyvaluep;

  switch (yytype)
    {

      default:
        break;
    }
}


/* Prevent warnings from -Wmissing-prototypes.  */

#ifdef YYPARSE_PARAM
# if defined (__STDC__) || defined (__cplusplus)
int yyparse (void *YYPARSE_PARAM);
# else
int yyparse ();
# endif
#else /* ! YYPARSE_PARAM */
#if defined (__STDC__) || defined (__cplusplus)
int yyparse (void);
#else
int yyparse ();
#endif
#endif /* ! YYPARSE_PARAM */



/* The lookahead symbol.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;

/* Number of syntax errors so far.  */
int yynerrs;



/*----------.
| yyparse.  |
`----------*/

#ifdef YYPARSE_PARAM
# if defined (__STDC__) || defined (__cplusplus)
int yyparse (void *YYPARSE_PARAM)
# else
int yyparse (YYPARSE_PARAM)
  void *YYPARSE_PARAM;
# endif
#else /* ! YYPARSE_PARAM */
#if defined (__STDC__) || defined (__cplusplus)
int
yyparse (void)
#else
int
yyparse ()

#endif
#endif
{
  
  register int yystate;
  register int yyn;
  int yyresult;
  /* Number of tokens to shift before error messages enabled.  */
  int yyerrstatus;
  /* Lookahead token as an internal (translated) token number.  */
  int yytoken = 0;

  /* Three stacks and their tools:
     `yyss': related to states,
     `yyvs': related to semantic values,
     `yyls': related to locations.

     Refer to the stacks thru separate pointers, to allow yyoverflow
     to reallocate them elsewhere.  */

  /* The state stack.  */
  short	yyssa[YYINITDEPTH];
  short *yyss = yyssa;
  register short *yyssp;

  /* The semantic value stack.  */
  YYSTYPE yyvsa[YYINITDEPTH];
  YYSTYPE *yyvs = yyvsa;
  register YYSTYPE *yyvsp;



#define YYPOPSTACK   (yyvsp--, yyssp--)

  YYSIZE_T yystacksize = YYINITDEPTH;

  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;


  /* When reducing, the number of symbols on the RHS of the reduced
     rule.  */
  int yylen;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY;		/* Cause a token to be read.  */

  /* Initialize stack pointers.
     Waste one element of value and location stack
     so that they stay on the same level as the state stack.
     The wasted elements are never initialized.  */

  yyssp = yyss;
  yyvsp = yyvs;

  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed. so pushing a state here evens the stacks.
     */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
	/* Give user a chance to reallocate the stack. Use copies of
	   these so that the &'s don't force the real ones into
	   memory.  */
	YYSTYPE *yyvs1 = yyvs;
	short *yyss1 = yyss;


	/* Each stack pointer address is followed by the size of the
	   data in use in that stack, in bytes.  This used to be a
	   conditional around just the two extra args, but that might
	   be undefined if yyoverflow is a macro.  */
	yyoverflow ("parser stack overflow",
		    &yyss1, yysize * sizeof (*yyssp),
		    &yyvs1, yysize * sizeof (*yyvsp),

		    &yystacksize);

	yyss = yyss1;
	yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyoverflowlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
	goto yyoverflowlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
	yystacksize = YYMAXDEPTH;

      {
	short *yyss1 = yyss;
	union yyalloc *yyptr =
	  (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
	if (! yyptr)
	  goto yyoverflowlab;
	YYSTACK_RELOCATE (yyss);
	YYSTACK_RELOCATE (yyvs);

#  undef YYSTACK_RELOCATE
	if (yyss1 != yyssa)
	  YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;


      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
		  (unsigned long int) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
	YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

/* Do appropriate processing given the current state.  */
/* Read a lookahead token if we need one and don't already have one.  */
/* yyresume: */

  /* First try to decide what to do without reference to lookahead token.  */

  yyn = yypact[yystate];
  if (yyn == YYPACT_NINF)
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid lookahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = YYLEX;
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YYDSYMPRINTF ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yyn == 0 || yyn == YYTABLE_NINF)
	goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  if (yyn == YYFINAL)
    YYACCEPT;

  /* Shift the lookahead token.  */
  YYDPRINTF ((stderr, "Shifting token %s, ", yytname[yytoken]));

  /* Discard the token being shifted unless it is eof.  */
  if (yychar != YYEOF)
    yychar = YYEMPTY;

  *++yyvsp = yylval;


  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  yystate = yyn;
  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     `$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 2:
#line 133 "xi-grammar.y"
    { yyval.modlist = yyvsp[0].modlist; modlist = yyvsp[0].modlist; }
    break;

  case 3:
#line 137 "xi-grammar.y"
    { 
		  yyval.modlist = 0; 
		}
    break;

  case 4:
#line 141 "xi-grammar.y"
    { yyval.modlist = new ModuleList(lineno, yyvsp[-1].module, yyvsp[0].modlist); }
    break;

  case 5:
#line 145 "xi-grammar.y"
    { yyval.intval = 0; }
    break;

  case 6:
#line 147 "xi-grammar.y"
    { yyval.intval = 1; }
    break;

  case 7:
#line 151 "xi-grammar.y"
    { yyval.intval = 0; }
    break;

  case 8:
#line 153 "xi-grammar.y"
    { yyval.intval = 1; }
    break;

  case 9:
#line 157 "xi-grammar.y"
    { yyval.strval = yyvsp[0].strval; }
    break;

  case 10:
#line 161 "xi-grammar.y"
    { yyval.strval = yyvsp[0].strval; }
    break;

  case 11:
#line 163 "xi-grammar.y"
    {
		  char *tmp = new char[strlen(yyvsp[-3].strval)+strlen(yyvsp[0].strval)+3];
		  sprintf(tmp,"%s::%s", yyvsp[-3].strval, yyvsp[0].strval);
		  yyval.strval = tmp;
		}
    break;

  case 12:
#line 171 "xi-grammar.y"
    { 
		    yyval.module = new Module(lineno, yyvsp[-1].strval, yyvsp[0].conslist); 
		}
    break;

  case 13:
#line 175 "xi-grammar.y"
    {  
		    yyval.module = new Module(lineno, yyvsp[-1].strval, yyvsp[0].conslist); 
		    yyval.module->setMain();
		}
    break;

  case 14:
#line 182 "xi-grammar.y"
    { yyval.conslist = 0; }
    break;

  case 15:
#line 184 "xi-grammar.y"
    { yyval.conslist = yyvsp[-2].conslist; }
    break;

  case 16:
#line 188 "xi-grammar.y"
    { yyval.conslist = 0; }
    break;

  case 17:
#line 190 "xi-grammar.y"
    { yyval.conslist = new ConstructList(lineno, yyvsp[-1].construct, yyvsp[0].conslist); }
    break;

  case 18:
#line 194 "xi-grammar.y"
    { if(yyvsp[-2].conslist) yyvsp[-2].conslist->setExtern(yyvsp[-4].intval); yyval.construct = yyvsp[-2].conslist; }
    break;

  case 19:
#line 196 "xi-grammar.y"
    { yyvsp[0].module->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].module; }
    break;

  case 20:
#line 198 "xi-grammar.y"
    { yyvsp[0].member->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].member; }
    break;

  case 21:
#line 200 "xi-grammar.y"
    { yyvsp[-1].message->setExtern(yyvsp[-2].intval); yyval.construct = yyvsp[-1].message; }
    break;

  case 22:
#line 202 "xi-grammar.y"
    { yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; }
    break;

  case 23:
#line 204 "xi-grammar.y"
    { yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; }
    break;

  case 24:
#line 206 "xi-grammar.y"
    { yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; }
    break;

  case 25:
#line 208 "xi-grammar.y"
    { yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; }
    break;

  case 26:
#line 210 "xi-grammar.y"
    { yyvsp[0].templat->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].templat; }
    break;

  case 27:
#line 212 "xi-grammar.y"
    { yyval.construct = NULL; }
    break;

  case 28:
#line 214 "xi-grammar.y"
    { yyval.construct = NULL; }
    break;

  case 29:
#line 218 "xi-grammar.y"
    { yyval.tparam = new TParamType(yyvsp[0].type); }
    break;

  case 30:
#line 220 "xi-grammar.y"
    { yyval.tparam = new TParamVal(yyvsp[0].strval); }
    break;

  case 31:
#line 222 "xi-grammar.y"
    { yyval.tparam = new TParamVal(yyvsp[0].strval); }
    break;

  case 32:
#line 226 "xi-grammar.y"
    { yyval.tparlist = new TParamList(yyvsp[0].tparam); }
    break;

  case 33:
#line 228 "xi-grammar.y"
    { yyval.tparlist = new TParamList(yyvsp[-2].tparam, yyvsp[0].tparlist); }
    break;

  case 34:
#line 232 "xi-grammar.y"
    { yyval.tparlist = 0; }
    break;

  case 35:
#line 234 "xi-grammar.y"
    { yyval.tparlist = yyvsp[0].tparlist; }
    break;

  case 36:
#line 238 "xi-grammar.y"
    { yyval.tparlist = 0; }
    break;

  case 37:
#line 240 "xi-grammar.y"
    { yyval.tparlist = yyvsp[-1].tparlist; }
    break;

  case 38:
#line 244 "xi-grammar.y"
    { yyval.type = new BuiltinType("int"); }
    break;

  case 39:
#line 246 "xi-grammar.y"
    { yyval.type = new BuiltinType("long"); }
    break;

  case 40:
#line 248 "xi-grammar.y"
    { yyval.type = new BuiltinType("short"); }
    break;

  case 41:
#line 250 "xi-grammar.y"
    { yyval.type = new BuiltinType("char"); }
    break;

  case 42:
#line 252 "xi-grammar.y"
    { yyval.type = new BuiltinType("unsigned int"); }
    break;

  case 43:
#line 254 "xi-grammar.y"
    { yyval.type = new BuiltinType("unsigned long"); }
    break;

  case 44:
#line 256 "xi-grammar.y"
    { yyval.type = new BuiltinType("unsigned short"); }
    break;

  case 45:
#line 258 "xi-grammar.y"
    { yyval.type = new BuiltinType("unsigned char"); }
    break;

  case 46:
#line 260 "xi-grammar.y"
    { yyval.type = new BuiltinType("long long"); }
    break;

  case 47:
#line 262 "xi-grammar.y"
    { yyval.type = new BuiltinType("float"); }
    break;

  case 48:
#line 264 "xi-grammar.y"
    { yyval.type = new BuiltinType("double"); }
    break;

  case 49:
#line 266 "xi-grammar.y"
    { yyval.type = new BuiltinType("long double"); }
    break;

  case 50:
#line 268 "xi-grammar.y"
    { yyval.type = new BuiltinType("void"); }
    break;

  case 51:
#line 271 "xi-grammar.y"
    { yyval.ntype = new NamedType(yyvsp[-1].strval,yyvsp[0].tparlist); }
    break;

  case 52:
#line 272 "xi-grammar.y"
    { yyval.ntype = new NamedType(yyvsp[-1].strval,yyvsp[0].tparlist); }
    break;

  case 53:
#line 275 "xi-grammar.y"
    { yyval.type = yyvsp[0].type; }
    break;

  case 54:
#line 277 "xi-grammar.y"
    { yyval.type = yyvsp[0].ntype; }
    break;

  case 55:
#line 281 "xi-grammar.y"
    { yyval.ptype = new PtrType(yyvsp[-1].type); }
    break;

  case 56:
#line 285 "xi-grammar.y"
    { yyvsp[-1].ptype->indirect(); yyval.ptype = yyvsp[-1].ptype; }
    break;

  case 57:
#line 287 "xi-grammar.y"
    { yyvsp[-1].ptype->indirect(); yyval.ptype = yyvsp[-1].ptype; }
    break;

  case 58:
#line 291 "xi-grammar.y"
    { yyval.ftype = new FuncType(yyvsp[-7].type, yyvsp[-4].strval, yyvsp[-1].plist); }
    break;

  case 59:
#line 295 "xi-grammar.y"
    { yyval.type = yyvsp[0].type; }
    break;

  case 60:
#line 297 "xi-grammar.y"
    { yyval.type = yyvsp[0].ptype; }
    break;

  case 61:
#line 299 "xi-grammar.y"
    { yyval.type = yyvsp[0].ptype; }
    break;

  case 62:
#line 301 "xi-grammar.y"
    { yyval.type = yyvsp[0].ftype; }
    break;

  case 63:
#line 304 "xi-grammar.y"
    { yyval.type = yyvsp[0].type; }
    break;

  case 64:
#line 306 "xi-grammar.y"
    { yyval.type = yyvsp[-1].type; }
    break;

  case 65:
#line 310 "xi-grammar.y"
    { yyval.type = new ReferenceType(yyvsp[-1].type); }
    break;

  case 66:
#line 312 "xi-grammar.y"
    { yyval.type = yyvsp[0].type; }
    break;

  case 67:
#line 316 "xi-grammar.y"
    { yyval.val = new Value(yyvsp[0].strval); }
    break;

  case 68:
#line 318 "xi-grammar.y"
    { yyval.val = new Value(yyvsp[0].strval); }
    break;

  case 69:
#line 322 "xi-grammar.y"
    { yyval.val = yyvsp[-1].val; }
    break;

  case 70:
#line 326 "xi-grammar.y"
    { yyval.vallist = 0; }
    break;

  case 71:
#line 328 "xi-grammar.y"
    { yyval.vallist = new ValueList(yyvsp[-1].val, yyvsp[0].vallist); }
    break;

  case 72:
#line 332 "xi-grammar.y"
    { yyval.readonly = new Readonly(lineno, yyvsp[-2].type, yyvsp[-1].strval, yyvsp[0].vallist); }
    break;

  case 73:
#line 336 "xi-grammar.y"
    { yyval.readonly = new Readonly(lineno, yyvsp[-2].type, yyvsp[0].strval, 0, 1); }
    break;

  case 74:
#line 340 "xi-grammar.y"
    { yyval.intval = 0;}
    break;

  case 75:
#line 342 "xi-grammar.y"
    { yyval.intval = 0;}
    break;

  case 76:
#line 346 "xi-grammar.y"
    { yyval.intval = 0; }
    break;

  case 77:
#line 348 "xi-grammar.y"
    { 
		  /*
		  printf("Warning: Message attributes are being phased out.\n");
		  printf("Warning: Please remove them from interface files.\n");
		  */
		  yyval.intval = yyvsp[-1].intval; 
		}
    break;

  case 78:
#line 358 "xi-grammar.y"
    { yyval.intval = yyvsp[0].intval; }
    break;

  case 79:
#line 360 "xi-grammar.y"
    { yyval.intval = yyvsp[-2].intval | yyvsp[0].intval; }
    break;

  case 80:
#line 364 "xi-grammar.y"
    { yyval.intval = 0; }
    break;

  case 81:
#line 366 "xi-grammar.y"
    { yyval.intval = 0; }
    break;

  case 82:
#line 370 "xi-grammar.y"
    { yyval.cattr = 0; }
    break;

  case 83:
#line 372 "xi-grammar.y"
    { yyval.cattr = yyvsp[-1].cattr; }
    break;

  case 84:
#line 376 "xi-grammar.y"
    { yyval.cattr = yyvsp[0].cattr; }
    break;

  case 85:
#line 378 "xi-grammar.y"
    { yyval.cattr = yyvsp[-2].cattr | yyvsp[0].cattr; }
    break;

  case 86:
#line 382 "xi-grammar.y"
    { yyval.cattr = Chare::CPYTHON; }
    break;

  case 87:
#line 386 "xi-grammar.y"
    { yyval.cattr = 0; }
    break;

  case 88:
#line 388 "xi-grammar.y"
    { yyval.cattr = yyvsp[-1].cattr; }
    break;

  case 89:
#line 392 "xi-grammar.y"
    { yyval.cattr = yyvsp[0].cattr; }
    break;

  case 90:
#line 394 "xi-grammar.y"
    { yyval.cattr = yyvsp[-2].cattr | yyvsp[0].cattr; }
    break;

  case 91:
#line 398 "xi-grammar.y"
    { yyval.cattr = Chare::CMIGRATABLE; }
    break;

  case 92:
#line 400 "xi-grammar.y"
    { yyval.cattr = Chare::CPYTHON; }
    break;

  case 93:
#line 404 "xi-grammar.y"
    { yyval.mv = new MsgVar(yyvsp[-4].type, yyvsp[-3].strval); }
    break;

  case 94:
#line 408 "xi-grammar.y"
    { yyval.mvlist = new MsgVarList(yyvsp[0].mv); }
    break;

  case 95:
#line 410 "xi-grammar.y"
    { yyval.mvlist = new MsgVarList(yyvsp[-1].mv, yyvsp[0].mvlist); }
    break;

  case 96:
#line 414 "xi-grammar.y"
    { yyval.message = new Message(lineno, yyvsp[0].ntype); }
    break;

  case 97:
#line 416 "xi-grammar.y"
    { yyval.message = new Message(lineno, yyvsp[-3].ntype, yyvsp[-1].mvlist); }
    break;

  case 98:
#line 420 "xi-grammar.y"
    { yyval.typelist = 0; }
    break;

  case 99:
#line 422 "xi-grammar.y"
    { yyval.typelist = yyvsp[0].typelist; }
    break;

  case 100:
#line 426 "xi-grammar.y"
    { yyval.typelist = new TypeList(yyvsp[0].ntype); }
    break;

  case 101:
#line 428 "xi-grammar.y"
    { yyval.typelist = new TypeList(yyvsp[-2].ntype, yyvsp[0].typelist); }
    break;

  case 102:
#line 432 "xi-grammar.y"
    { yyval.chare = new Chare(lineno, yyvsp[-3].cattr, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 103:
#line 434 "xi-grammar.y"
    { yyval.chare = new MainChare(lineno, yyvsp[-3].cattr, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 104:
#line 438 "xi-grammar.y"
    { yyval.chare = new Group(lineno, yyvsp[-3].cattr, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 105:
#line 442 "xi-grammar.y"
    { yyval.chare = new NodeGroup(lineno, yyvsp[-3].cattr, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 106:
#line 446 "xi-grammar.y"
    {/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",yyvsp[-2].strval);
			yyval.ntype = new NamedType(buf); 
		}
    break;

  case 107:
#line 452 "xi-grammar.y"
    { yyval.ntype = new NamedType(yyvsp[-1].strval); }
    break;

  case 108:
#line 456 "xi-grammar.y"
    {  yyval.chare = new Array(lineno, yyvsp[-4].cattr, yyvsp[-3].ntype, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 109:
#line 458 "xi-grammar.y"
    {  yyval.chare = new Array(lineno, yyvsp[-3].cattr, yyvsp[-4].ntype, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 110:
#line 462 "xi-grammar.y"
    { yyval.chare = new Chare(lineno, yyvsp[-3].cattr, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist);}
    break;

  case 111:
#line 464 "xi-grammar.y"
    { yyval.chare = new MainChare(lineno, yyvsp[-3].cattr, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 112:
#line 468 "xi-grammar.y"
    { yyval.chare = new Group(lineno, yyvsp[-3].cattr, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 113:
#line 472 "xi-grammar.y"
    { yyval.chare = new NodeGroup( lineno, yyvsp[-3].cattr, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 114:
#line 476 "xi-grammar.y"
    { yyval.chare = new Array( lineno, 0, yyvsp[-3].ntype, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 115:
#line 480 "xi-grammar.y"
    { yyval.message = new Message(lineno, new NamedType(yyvsp[-1].strval)); }
    break;

  case 116:
#line 482 "xi-grammar.y"
    { yyval.message = new Message(lineno, new NamedType(yyvsp[-4].strval), yyvsp[-2].mvlist); }
    break;

  case 117:
#line 486 "xi-grammar.y"
    { yyval.type = 0; }
    break;

  case 118:
#line 488 "xi-grammar.y"
    { yyval.type = yyvsp[0].type; }
    break;

  case 119:
#line 492 "xi-grammar.y"
    { yyval.strval = 0; }
    break;

  case 120:
#line 494 "xi-grammar.y"
    { yyval.strval = yyvsp[0].strval; }
    break;

  case 121:
#line 496 "xi-grammar.y"
    { yyval.strval = yyvsp[0].strval; }
    break;

  case 122:
#line 500 "xi-grammar.y"
    { yyval.tvar = new TType(new NamedType(yyvsp[-1].strval), yyvsp[0].type); }
    break;

  case 123:
#line 502 "xi-grammar.y"
    { yyval.tvar = new TFunc(yyvsp[-1].ftype, yyvsp[0].strval); }
    break;

  case 124:
#line 504 "xi-grammar.y"
    { yyval.tvar = new TName(yyvsp[-2].type, yyvsp[-1].strval, yyvsp[0].strval); }
    break;

  case 125:
#line 508 "xi-grammar.y"
    { yyval.tvarlist = new TVarList(yyvsp[0].tvar); }
    break;

  case 126:
#line 510 "xi-grammar.y"
    { yyval.tvarlist = new TVarList(yyvsp[-2].tvar, yyvsp[0].tvarlist); }
    break;

  case 127:
#line 514 "xi-grammar.y"
    { yyval.tvarlist = yyvsp[-1].tvarlist; }
    break;

  case 128:
#line 518 "xi-grammar.y"
    { yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); }
    break;

  case 129:
#line 520 "xi-grammar.y"
    { yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); }
    break;

  case 130:
#line 522 "xi-grammar.y"
    { yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); }
    break;

  case 131:
#line 524 "xi-grammar.y"
    { yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); }
    break;

  case 132:
#line 526 "xi-grammar.y"
    { yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].message); yyvsp[0].message->setTemplate(yyval.templat); }
    break;

  case 133:
#line 530 "xi-grammar.y"
    { yyval.mbrlist = 0; }
    break;

  case 134:
#line 532 "xi-grammar.y"
    { yyval.mbrlist = yyvsp[-2].mbrlist; }
    break;

  case 135:
#line 536 "xi-grammar.y"
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
		}
    break;

  case 136:
#line 555 "xi-grammar.y"
    { yyval.mbrlist = new MemberList(yyvsp[-1].member, yyvsp[0].mbrlist); }
    break;

  case 137:
#line 559 "xi-grammar.y"
    { yyval.member = yyvsp[-1].readonly; }
    break;

  case 138:
#line 561 "xi-grammar.y"
    { yyval.member = yyvsp[-1].readonly; }
    break;

  case 140:
#line 564 "xi-grammar.y"
    { yyval.member = yyvsp[-1].member; }
    break;

  case 141:
#line 566 "xi-grammar.y"
    { yyval.member = yyvsp[-1].pupable; }
    break;

  case 142:
#line 568 "xi-grammar.y"
    { yyval.member = yyvsp[-1].includeFile; }
    break;

  case 143:
#line 572 "xi-grammar.y"
    { yyval.member = new InitCall(lineno, yyvsp[0].strval, 1); }
    break;

  case 144:
#line 574 "xi-grammar.y"
    { yyval.member = new InitCall(lineno, yyvsp[-3].strval, 1); }
    break;

  case 145:
#line 576 "xi-grammar.y"
    { printf("Warning: deprecated use of initcall. Use initnode or initproc instead.\n"); 
		  yyval.member = new InitCall(lineno, yyvsp[0].strval, 1); }
    break;

  case 146:
#line 579 "xi-grammar.y"
    { printf("Warning: deprecated use of initcall. Use initnode or initproc instead.\n");
		  yyval.member = new InitCall(lineno, yyvsp[-3].strval, 1); }
    break;

  case 147:
#line 584 "xi-grammar.y"
    { yyval.member = new InitCall(lineno, yyvsp[0].strval, 0); }
    break;

  case 148:
#line 586 "xi-grammar.y"
    { yyval.member = new InitCall(lineno, yyvsp[-3].strval, 0); }
    break;

  case 149:
#line 590 "xi-grammar.y"
    { yyval.pupable = new PUPableClass(lineno,yyvsp[0].strval,0); }
    break;

  case 150:
#line 592 "xi-grammar.y"
    { yyval.pupable = new PUPableClass(lineno,yyvsp[-2].strval,yyvsp[0].pupable); }
    break;

  case 151:
#line 596 "xi-grammar.y"
    { yyval.includeFile = new IncludeFile(lineno,yyvsp[0].strval,0); }
    break;

  case 152:
#line 600 "xi-grammar.y"
    { yyval.member = yyvsp[-1].entry; }
    break;

  case 153:
#line 602 "xi-grammar.y"
    { yyval.member = yyvsp[0].member; }
    break;

  case 154:
#line 606 "xi-grammar.y"
    { 
		  if (yyvsp[0].sc != 0) { 
		    yyvsp[0].sc->con1 = new SdagConstruct(SIDENT, yyvsp[-3].strval);
  		    if (yyvsp[-2].plist != 0)
                      yyvsp[0].sc->param = new ParamList(yyvsp[-2].plist);
 		    else 
 	 	      yyvsp[0].sc->param = new ParamList(new Parameter(0, new BuiltinType("void")));
                  }
		  yyval.entry = new Entry(lineno, yyvsp[-5].intval, yyvsp[-4].type, yyvsp[-3].strval, yyvsp[-2].plist, yyvsp[-1].val, yyvsp[0].sc, 0, 0); 
		}
    break;

  case 155:
#line 617 "xi-grammar.y"
    { 
		  if (yyvsp[0].sc != 0) {
		    yyvsp[0].sc->con1 = new SdagConstruct(SIDENT, yyvsp[-2].strval);
		    if (yyvsp[-1].plist != 0)
                      yyvsp[0].sc->param = new ParamList(yyvsp[-1].plist);
		    else
                      yyvsp[0].sc->param = new ParamList(new Parameter(0, new BuiltinType("void")));
                  }
		  yyval.entry = new Entry(lineno, yyvsp[-3].intval,     0, yyvsp[-2].strval, yyvsp[-1].plist,  0, yyvsp[0].sc, 0, 0); 
		}
    break;

  case 156:
#line 630 "xi-grammar.y"
    { yyval.type = new BuiltinType("void"); }
    break;

  case 157:
#line 632 "xi-grammar.y"
    { yyval.type = yyvsp[0].ptype; }
    break;

  case 158:
#line 636 "xi-grammar.y"
    { yyval.intval = 0; }
    break;

  case 159:
#line 638 "xi-grammar.y"
    { yyval.intval = yyvsp[-1].intval; }
    break;

  case 160:
#line 642 "xi-grammar.y"
    { yyval.intval = yyvsp[0].intval; }
    break;

  case 161:
#line 644 "xi-grammar.y"
    { yyval.intval = yyvsp[-2].intval | yyvsp[0].intval; }
    break;

  case 162:
#line 648 "xi-grammar.y"
    { yyval.intval = STHREADED; }
    break;

  case 163:
#line 650 "xi-grammar.y"
    { yyval.intval = SSYNC; }
    break;

  case 164:
#line 652 "xi-grammar.y"
    { yyval.intval = SLOCKED; }
    break;

  case 165:
#line 654 "xi-grammar.y"
    { yyval.intval = SCREATEHERE; }
    break;

  case 166:
#line 656 "xi-grammar.y"
    { yyval.intval = SCREATEHOME; }
    break;

  case 167:
#line 658 "xi-grammar.y"
    { yyval.intval = SNOKEEP; }
    break;

  case 168:
#line 660 "xi-grammar.y"
    { yyval.intval = SNOTRACE; }
    break;

  case 169:
#line 662 "xi-grammar.y"
    { yyval.intval = SIMMEDIATE; }
    break;

  case 170:
#line 664 "xi-grammar.y"
    { yyval.intval = SSKIPSCHED; }
    break;

  case 171:
#line 666 "xi-grammar.y"
    { yyval.intval = SPYTHON; }
    break;

  case 172:
#line 670 "xi-grammar.y"
    { yyval.val = new Value(yyvsp[0].strval); }
    break;

  case 173:
#line 672 "xi-grammar.y"
    { yyval.val = new Value(yyvsp[0].strval); }
    break;

  case 174:
#line 674 "xi-grammar.y"
    { yyval.val = new Value(yyvsp[0].strval); }
    break;

  case 175:
#line 678 "xi-grammar.y"
    { yyval.strval = ""; }
    break;

  case 176:
#line 680 "xi-grammar.y"
    { yyval.strval = yyvsp[0].strval; }
    break;

  case 177:
#line 682 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen(yyvsp[-2].strval)+strlen(yyvsp[0].strval)+3];
			sprintf(tmp,"%s, %s", yyvsp[-2].strval, yyvsp[0].strval);
			yyval.strval = tmp;
		}
    break;

  case 178:
#line 690 "xi-grammar.y"
    { yyval.strval = ""; }
    break;

  case 179:
#line 692 "xi-grammar.y"
    { yyval.strval = yyvsp[0].strval; }
    break;

  case 180:
#line 694 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen(yyvsp[-4].strval)+strlen(yyvsp[-2].strval)+strlen(yyvsp[0].strval)+3];
			sprintf(tmp,"%s[%s]%s", yyvsp[-4].strval, yyvsp[-2].strval, yyvsp[0].strval);
			yyval.strval = tmp;
		}
    break;

  case 181:
#line 700 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen(yyvsp[-4].strval)+strlen(yyvsp[-2].strval)+strlen(yyvsp[0].strval)+3];
			sprintf(tmp,"%s{%s}%s", yyvsp[-4].strval, yyvsp[-2].strval, yyvsp[0].strval);
			yyval.strval = tmp;
		}
    break;

  case 182:
#line 706 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen(yyvsp[-4].strval)+strlen(yyvsp[-2].strval)+strlen(yyvsp[0].strval)+3];
			sprintf(tmp,"%s(%s)%s", yyvsp[-4].strval, yyvsp[-2].strval, yyvsp[0].strval);
			yyval.strval = tmp;
		}
    break;

  case 183:
#line 712 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen(yyvsp[-2].strval)+strlen(yyvsp[0].strval)+3];
			sprintf(tmp,"(%s)%s", yyvsp[-2].strval, yyvsp[0].strval);
			yyval.strval = tmp;
		}
    break;

  case 184:
#line 720 "xi-grammar.y"
    {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			yyval.pname = new Parameter(lineno, yyvsp[-2].type,yyvsp[-1].strval);
		}
    break;

  case 185:
#line 727 "xi-grammar.y"
    { 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			yyval.intval = 0;
		}
    break;

  case 186:
#line 735 "xi-grammar.y"
    { 
			in_braces=0;
			yyval.intval = 0;
		}
    break;

  case 187:
#line 742 "xi-grammar.y"
    { yyval.pname = new Parameter(lineno, yyvsp[0].type);}
    break;

  case 188:
#line 744 "xi-grammar.y"
    { yyval.pname = new Parameter(lineno, yyvsp[-1].type,yyvsp[0].strval);}
    break;

  case 189:
#line 746 "xi-grammar.y"
    { yyval.pname = new Parameter(lineno, yyvsp[-3].type,yyvsp[-2].strval,0,yyvsp[0].val);}
    break;

  case 190:
#line 748 "xi-grammar.y"
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			yyval.pname = new Parameter(lineno, yyvsp[-2].pname->getType(), yyvsp[-2].pname->getName() ,yyvsp[-1].strval);
		}
    break;

  case 191:
#line 755 "xi-grammar.y"
    { yyval.plist = new ParamList(yyvsp[0].pname); }
    break;

  case 192:
#line 757 "xi-grammar.y"
    { yyval.plist = new ParamList(yyvsp[-2].pname,yyvsp[0].plist); }
    break;

  case 193:
#line 761 "xi-grammar.y"
    { yyval.plist = yyvsp[-1].plist; }
    break;

  case 194:
#line 763 "xi-grammar.y"
    { yyval.plist = 0; }
    break;

  case 195:
#line 767 "xi-grammar.y"
    { yyval.val = 0; }
    break;

  case 196:
#line 769 "xi-grammar.y"
    { yyval.val = new Value(yyvsp[0].strval); }
    break;

  case 197:
#line 773 "xi-grammar.y"
    { yyval.sc = 0; }
    break;

  case 198:
#line 775 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SSDAGENTRY, yyvsp[0].sc); }
    break;

  case 199:
#line 777 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SSDAGENTRY, yyvsp[-1].sc); }
    break;

  case 200:
#line 781 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SSLIST, yyvsp[0].sc); }
    break;

  case 201:
#line 783 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SSLIST, yyvsp[-1].sc, yyvsp[0].sc);  }
    break;

  case 202:
#line 787 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SOLIST, yyvsp[0].sc); }
    break;

  case 203:
#line 789 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SOLIST, yyvsp[-1].sc, yyvsp[0].sc); }
    break;

  case 204:
#line 793 "xi-grammar.y"
    { yyval.sc = 0; }
    break;

  case 205:
#line 795 "xi-grammar.y"
    { yyval.sc = yyvsp[-1].sc; }
    break;

  case 206:
#line 799 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SPUBLISHES, new SdagConstruct(SIDENT, yyvsp[0].strval)); }
    break;

  case 207:
#line 801 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SPUBLISHES, new SdagConstruct(SIDENT, yyvsp[-2].strval), yyvsp[0].sc);  }
    break;

  case 208:
#line 805 "xi-grammar.y"
    { yyval.strval = yyvsp[0].strval; }
    break;

  case 209:
#line 807 "xi-grammar.y"
    { yyval.strval = 0; }
    break;

  case 210:
#line 811 "xi-grammar.y"
    { RemoveSdagComments(yyvsp[-2].strval);
		   yyval.sc = new SdagConstruct(SATOMIC, new XStr(yyvsp[-2].strval), yyvsp[0].sc, 0,0,0,0, 0 ); 
		   if (yyvsp[-4].strval) { yyvsp[-4].strval[strlen(yyvsp[-4].strval)-1]=0; yyval.sc->traceName = new XStr(yyvsp[-4].strval+1); }
		 }
    break;

  case 211:
#line 816 "xi-grammar.y"
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
		}
    break;

  case 212:
#line 830 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SWHEN, 0, 0,0,0,0,0,yyvsp[-2].entrylist); }
    break;

  case 213:
#line 832 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SWHEN, 0, 0, 0,0,0, yyvsp[0].sc, yyvsp[-1].entrylist); }
    break;

  case 214:
#line 834 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SWHEN, 0, 0, 0,0,0, yyvsp[-1].sc, yyvsp[-3].entrylist); }
    break;

  case 215:
#line 836 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SOVERLAP,0, 0,0,0,0,yyvsp[-1].sc, 0); }
    break;

  case 216:
#line 838 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SFOR, 0, new SdagConstruct(SINT_EXPR, yyvsp[-8].strval), new SdagConstruct(SINT_EXPR, yyvsp[-6].strval),
		             new SdagConstruct(SINT_EXPR, yyvsp[-4].strval), 0, yyvsp[-1].sc, 0); }
    break;

  case 217:
#line 841 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SFOR, 0, new SdagConstruct(SINT_EXPR, yyvsp[-6].strval), new SdagConstruct(SINT_EXPR, yyvsp[-4].strval), 
		         new SdagConstruct(SINT_EXPR, yyvsp[-2].strval), 0, yyvsp[0].sc, 0); }
    break;

  case 218:
#line 844 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SFORALL, 0, new SdagConstruct(SIDENT, yyvsp[-9].strval), new SdagConstruct(SINT_EXPR, yyvsp[-6].strval), 
		             new SdagConstruct(SINT_EXPR, yyvsp[-4].strval), new SdagConstruct(SINT_EXPR, yyvsp[-2].strval), yyvsp[0].sc, 0); }
    break;

  case 219:
#line 847 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SFORALL, 0, new SdagConstruct(SIDENT, yyvsp[-11].strval), new SdagConstruct(SINT_EXPR, yyvsp[-8].strval), 
		                 new SdagConstruct(SINT_EXPR, yyvsp[-6].strval), new SdagConstruct(SINT_EXPR, yyvsp[-4].strval), yyvsp[-1].sc, 0); }
    break;

  case 220:
#line 850 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, yyvsp[-3].strval), yyvsp[0].sc,0,0,yyvsp[-1].sc,0); }
    break;

  case 221:
#line 852 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, yyvsp[-5].strval), yyvsp[0].sc,0,0,yyvsp[-2].sc,0); }
    break;

  case 222:
#line 854 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, yyvsp[-2].strval), 0,0,0,yyvsp[0].sc,0); }
    break;

  case 223:
#line 856 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SWHILE, 0, new SdagConstruct(SINT_EXPR, yyvsp[-4].strval), 0,0,0,yyvsp[-1].sc,0); }
    break;

  case 224:
#line 858 "xi-grammar.y"
    { yyval.sc = yyvsp[-1].sc; }
    break;

  case 225:
#line 862 "xi-grammar.y"
    { yyval.sc = 0; }
    break;

  case 226:
#line 864 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SELSE, 0,0,0,0,0, yyvsp[0].sc,0); }
    break;

  case 227:
#line 866 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SELSE, 0,0,0,0,0, yyvsp[-1].sc,0); }
    break;

  case 228:
#line 869 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SFORWARD, new SdagConstruct(SIDENT, yyvsp[0].strval)); }
    break;

  case 229:
#line 871 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SFORWARD, new SdagConstruct(SIDENT, yyvsp[-2].strval), yyvsp[0].sc);  }
    break;

  case 230:
#line 875 "xi-grammar.y"
    { in_int_expr = 0; yyval.intval = 0; }
    break;

  case 231:
#line 879 "xi-grammar.y"
    { in_int_expr = 1; yyval.intval = 0; }
    break;

  case 232:
#line 883 "xi-grammar.y"
    { 
		  if (yyvsp[0].plist != 0)
		     yyval.entry = new Entry(lineno, 0, 0, yyvsp[-1].strval, yyvsp[0].plist, 0, 0, 0, 0); 
		  else
		     yyval.entry = new Entry(lineno, 0, 0, yyvsp[-1].strval, 
				new ParamList(new Parameter(0, new BuiltinType("void"))), 0, 0, 0, 0); 
		}
    break;

  case 233:
#line 891 "xi-grammar.y"
    { if (yyvsp[0].plist != 0)
		    yyval.entry = new Entry(lineno, 0, 0, yyvsp[-4].strval, yyvsp[0].plist, 0, 0, yyvsp[-2].strval, 0); 
		  else
		    yyval.entry = new Entry(lineno, 0, 0, yyvsp[-4].strval, new ParamList(new Parameter(0, new BuiltinType("void"))), 0, 0, yyvsp[-2].strval, 0); 
		}
    break;

  case 234:
#line 899 "xi-grammar.y"
    { yyval.entrylist = new EntryList(yyvsp[0].entry); }
    break;

  case 235:
#line 901 "xi-grammar.y"
    { yyval.entrylist = new EntryList(yyvsp[-2].entry,yyvsp[0].entrylist); }
    break;

  case 236:
#line 905 "xi-grammar.y"
    { in_bracket=1; }
    break;

  case 237:
#line 908 "xi-grammar.y"
    { in_bracket=0; }
    break;

  case 238:
#line 912 "xi-grammar.y"
    { if (!macroDefined(yyvsp[0].strval, 1)) in_comment = 1; }
    break;

  case 239:
#line 916 "xi-grammar.y"
    { if (!macroDefined(yyvsp[0].strval, 0)) in_comment = 1; }
    break;


    }

/* Line 991 of yacc.c.  */
#line 2907 "y.tab.c"

  yyvsp -= yylen;
  yyssp -= yylen;


  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;


  /* Now `shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*------------------------------------.
| yyerrlab -- here on detecting error |
`------------------------------------*/
yyerrlab:
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if YYERROR_VERBOSE
      yyn = yypact[yystate];

      if (YYPACT_NINF < yyn && yyn < YYLAST)
	{
	  YYSIZE_T yysize = 0;
	  int yytype = YYTRANSLATE (yychar);
	  char *yymsg;
	  int yyx, yycount;

	  yycount = 0;
	  /* Start YYX at -YYN if negative to avoid negative indexes in
	     YYCHECK.  */
	  for (yyx = yyn < 0 ? -yyn : 0;
	       yyx < (int) (sizeof (yytname) / sizeof (char *)); yyx++)
	    if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR)
	      yysize += yystrlen (yytname[yyx]) + 15, yycount++;
	  yysize += yystrlen ("syntax error, unexpected ") + 1;
	  yysize += yystrlen (yytname[yytype]);
	  yymsg = (char *) YYSTACK_ALLOC (yysize);
	  if (yymsg != 0)
	    {
	      char *yyp = yystpcpy (yymsg, "syntax error, unexpected ");
	      yyp = yystpcpy (yyp, yytname[yytype]);

	      if (yycount < 5)
		{
		  yycount = 0;
		  for (yyx = yyn < 0 ? -yyn : 0;
		       yyx < (int) (sizeof (yytname) / sizeof (char *));
		       yyx++)
		    if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR)
		      {
			const char *yyq = ! yycount ? ", expecting " : " or ";
			yyp = yystpcpy (yyp, yyq);
			yyp = yystpcpy (yyp, yytname[yyx]);
			yycount++;
		      }
		}
	      yyerror (yymsg);
	      YYSTACK_FREE (yymsg);
	    }
	  else
	    yyerror ("syntax error; also virtual memory exhausted");
	}
      else
#endif /* YYERROR_VERBOSE */
	yyerror ("syntax error");
    }



  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
	 error, discard it.  */

      /* Return failure if at end of input.  */
      if (yychar == YYEOF)
        {
	  /* Pop the error token.  */
          YYPOPSTACK;
	  /* Pop the rest of the stack.  */
	  while (yyss < yyssp)
	    {
	      YYDSYMPRINTF ("Error: popping", yystos[*yyssp], yyvsp, yylsp);
	      yydestruct (yystos[*yyssp], yyvsp);
	      YYPOPSTACK;
	    }
	  YYABORT;
        }

      YYDSYMPRINTF ("Error: discarding", yytoken, &yylval, &yylloc);
      yydestruct (yytoken, &yylval);
      yychar = YYEMPTY;

    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab2;


/*----------------------------------------------------.
| yyerrlab1 -- error raised explicitly by an action.  |
`----------------------------------------------------*/
yyerrlab1:

  /* Suppress GCC warning that yyerrlab1 is unused when no action
     invokes YYERROR.  */
#if defined (__GNUC_MINOR__) && 2093 <= (__GNUC__ * 1000 + __GNUC_MINOR__) \
    && !defined __cplusplus
  __attribute__ ((__unused__))
#endif


  goto yyerrlab2;


/*---------------------------------------------------------------.
| yyerrlab2 -- pop states until the error token can be shifted.  |
`---------------------------------------------------------------*/
yyerrlab2:
  yyerrstatus = 3;	/* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (yyn != YYPACT_NINF)
	{
	  yyn += YYTERROR;
	  if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
	    {
	      yyn = yytable[yyn];
	      if (0 < yyn)
		break;
	    }
	}

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
	YYABORT;

      YYDSYMPRINTF ("Error: popping", yystos[*yyssp], yyvsp, yylsp);
      yydestruct (yystos[yystate], yyvsp);
      yyvsp--;
      yystate = *--yyssp;

      YY_STACK_PRINT (yyss, yyssp);
    }

  if (yyn == YYFINAL)
    YYACCEPT;

  YYDPRINTF ((stderr, "Shifting error token, "));

  *++yyvsp = yylval;


  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#ifndef yyoverflow
/*----------------------------------------------.
| yyoverflowlab -- parser overflow comes here.  |
`----------------------------------------------*/
yyoverflowlab:
  yyerror ("parser stack overflow");
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
  return yyresult;
}


#line 919 "xi-grammar.y"

void yyerror(const char *mesg)
{
  cout << cur_file<<":"<<lineno<<": Charmxi syntax error> " << mesg << endl;
  // return 0;
}

