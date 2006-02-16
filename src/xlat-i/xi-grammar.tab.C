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
     IGET = 278,
     EXCLUSIVE = 279,
     IMMEDIATE = 280,
     SKIPSCHED = 281,
     INLINE = 282,
     VIRTUAL = 283,
     MIGRATABLE = 284,
     CREATEHERE = 285,
     CREATEHOME = 286,
     NOKEEP = 287,
     NOTRACE = 288,
     VOID = 289,
     CONST = 290,
     PACKED = 291,
     VARSIZE = 292,
     ENTRY = 293,
     FOR = 294,
     FORALL = 295,
     WHILE = 296,
     WHEN = 297,
     OVERLAP = 298,
     ATOMIC = 299,
     FORWARD = 300,
     IF = 301,
     ELSE = 302,
     CONNECT = 303,
     PUBLISHES = 304,
     PYTHON = 305,
     IDENT = 306,
     NUMBER = 307,
     LITERAL = 308,
     CPROGRAM = 309,
     HASHIF = 310,
     HASHIFDEF = 311,
     INT = 312,
     LONG = 313,
     SHORT = 314,
     CHAR = 315,
     FLOAT = 316,
     DOUBLE = 317,
     UNSIGNED = 318
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
#define IGET 278
#define EXCLUSIVE 279
#define IMMEDIATE 280
#define SKIPSCHED 281
#define INLINE 282
#define VIRTUAL 283
#define MIGRATABLE 284
#define CREATEHERE 285
#define CREATEHOME 286
#define NOKEEP 287
#define NOTRACE 288
#define VOID 289
#define CONST 290
#define PACKED 291
#define VARSIZE 292
#define ENTRY 293
#define FOR 294
#define FORALL 295
#define WHILE 296
#define WHEN 297
#define OVERLAP 298
#define ATOMIC 299
#define FORWARD 300
#define IF 301
#define ELSE 302
#define CONNECT 303
#define PUBLISHES 304
#define PYTHON 305
#define IDENT 306
#define NUMBER 307
#define LITERAL 308
#define CPROGRAM 309
#define HASHIF 310
#define HASHIFDEF 311
#define INT 312
#define LONG 313
#define SHORT 314
#define CHAR 315
#define FLOAT 316
#define DOUBLE 317
#define UNSIGNED 318




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
extern char *python_doc;



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
#line 17 "xi-grammar.y"
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
#line 252 "y.tab.c"
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 214 of yacc.c.  */
#line 264 "y.tab.c"

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
#define YYLAST   554

/* YYNTOKENS -- Number of terminals. */
#define YYNTOKENS  78
/* YYNNTS -- Number of nonterminals. */
#define YYNNTS  101
/* YYNRULES -- Number of rules. */
#define YYNRULES  246
/* YYNRULES -- Number of states. */
#define YYNSTATES  487

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   318

#define YYTRANSLATE(YYX) 						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const unsigned char yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    74,     2,
      72,    73,    71,     2,    68,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    65,    64,
      69,    77,    70,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    75,     2,    76,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    66,     2,    67,     2,     2,     2,     2,
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
      55,    56,    57,    58,    59,    60,    61,    62,    63
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
     102,   104,   106,   109,   112,   116,   120,   123,   126,   129,
     131,   133,   136,   138,   141,   144,   146,   148,   151,   154,
     157,   166,   168,   170,   172,   174,   177,   180,   183,   185,
     187,   189,   193,   194,   197,   202,   208,   209,   211,   212,
     216,   218,   222,   224,   226,   227,   231,   233,   237,   238,
     240,   242,   243,   247,   249,   253,   255,   257,   263,   265,
     268,   272,   279,   280,   283,   285,   289,   295,   301,   307,
     313,   318,   322,   329,   336,   342,   348,   354,   360,   366,
     371,   379,   380,   383,   384,   387,   390,   394,   397,   401,
     403,   407,   412,   415,   418,   421,   424,   427,   429,   434,
     435,   438,   441,   444,   447,   450,   454,   458,   462,   466,
     473,   477,   484,   488,   495,   497,   501,   503,   506,   508,
     516,   522,   524,   526,   527,   531,   533,   537,   539,   541,
     543,   545,   547,   549,   551,   553,   555,   557,   559,   562,
     564,   566,   568,   569,   571,   575,   576,   578,   584,   590,
     596,   601,   605,   607,   609,   611,   614,   619,   623,   625,
     629,   633,   636,   637,   641,   642,   644,   648,   650,   653,
     655,   658,   659,   664,   666,   670,   672,   673,   680,   689,
     694,   698,   704,   709,   721,   731,   744,   759,   766,   775,
     781,   789,   793,   794,   797,   802,   804,   808,   810,   812,
     815,   821,   823,   827,   829,   831,   834
};

/* YYRHS -- A `-1'-separated list of the rules' RHS. */
static const short yyrhs[] =
{
      79,     0,    -1,    80,    -1,    -1,    85,    80,    -1,    -1,
       5,    -1,    -1,    64,    -1,    51,    -1,    51,    -1,    84,
      65,    65,    51,    -1,     3,    83,    86,    -1,     4,    83,
      86,    -1,    64,    -1,    66,    87,    67,    82,    -1,    -1,
      88,    87,    -1,    81,    66,    87,    67,    82,    -1,    81,
      85,    -1,    81,   141,    -1,    81,   120,    64,    -1,    81,
     123,    -1,    81,   124,    -1,    81,   125,    -1,    81,   127,
      -1,    81,   138,    -1,   177,    -1,   178,    -1,   101,    -1,
      52,    -1,    53,    -1,    89,    -1,    89,    68,    90,    -1,
      -1,    90,    -1,    -1,    69,    91,    70,    -1,    57,    -1,
      58,    -1,    59,    -1,    60,    -1,    63,    57,    -1,    63,
      58,    -1,    63,    58,    57,    -1,    63,    58,    58,    -1,
      63,    59,    -1,    63,    60,    -1,    58,    58,    -1,    61,
      -1,    62,    -1,    58,    62,    -1,    34,    -1,    83,    92,
      -1,    84,    92,    -1,    93,    -1,    95,    -1,    96,    71,
      -1,    97,    71,    -1,    98,    71,    -1,   100,    72,    71,
      83,    73,    72,   159,    73,    -1,    96,    -1,    97,    -1,
      98,    -1,    99,    -1,    35,   100,    -1,   100,    35,    -1,
     100,    74,    -1,   100,    -1,    52,    -1,    84,    -1,    75,
     102,    76,    -1,    -1,   103,   104,    -1,     6,   101,    84,
     104,    -1,     6,    16,    96,    71,    83,    -1,    -1,    34,
      -1,    -1,    75,   109,    76,    -1,   110,    -1,   110,    68,
     109,    -1,    36,    -1,    37,    -1,    -1,    75,   112,    76,
      -1,   117,    -1,   117,    68,   112,    -1,    -1,    53,    -1,
      50,    -1,    -1,    75,   116,    76,    -1,   114,    -1,   114,
      68,   116,    -1,    29,    -1,    50,    -1,   101,    83,    75,
      76,    64,    -1,   118,    -1,   118,   119,    -1,    16,   108,
      94,    -1,    16,   108,    94,    66,   119,    67,    -1,    -1,
      65,   122,    -1,    94,    -1,    94,    68,   122,    -1,    11,
     111,    94,   121,   139,    -1,    12,   111,    94,   121,   139,
      -1,    13,   111,    94,   121,   139,    -1,    14,   111,    94,
     121,   139,    -1,    75,    52,    83,    76,    -1,    75,    83,
      76,    -1,    15,   115,   126,    94,   121,   139,    -1,    15,
     126,   115,    94,   121,   139,    -1,    11,   111,    83,   121,
     139,    -1,    12,   111,    83,   121,   139,    -1,    13,   111,
      83,   121,   139,    -1,    14,   111,    83,   121,   139,    -1,
      15,   126,    83,   121,   139,    -1,    16,   108,    83,    64,
      -1,    16,   108,    83,    66,   119,    67,    64,    -1,    -1,
      77,   101,    -1,    -1,    77,    52,    -1,    77,    53,    -1,
      17,    83,   133,    -1,    99,   134,    -1,   101,    83,   134,
      -1,   135,    -1,   135,    68,   136,    -1,    21,    69,   136,
      70,    -1,   137,   128,    -1,   137,   129,    -1,   137,   130,
      -1,   137,   131,    -1,   137,   132,    -1,    64,    -1,    66,
     140,    67,    82,    -1,    -1,   146,   140,    -1,   105,    64,
      -1,   106,    64,    -1,   143,    64,    -1,   142,    64,    -1,
      10,   144,    64,    -1,    18,   145,    64,    -1,    17,    83,
      64,    -1,     8,   107,    84,    -1,     8,   107,    84,    72,
     107,    73,    -1,     7,   107,    84,    -1,     7,   107,    84,
      72,   107,    73,    -1,     9,   107,    84,    -1,     9,   107,
      84,    72,   107,    73,    -1,    84,    -1,    84,    68,   144,
      -1,    53,    -1,   147,    64,    -1,   141,    -1,    38,   149,
     148,    83,   160,   161,   162,    -1,    38,   149,    83,   160,
     162,    -1,    34,    -1,    97,    -1,    -1,    75,   150,    76,
      -1,   151,    -1,   151,    68,   150,    -1,    20,    -1,    22,
      -1,    23,    -1,    24,    -1,    30,    -1,    31,    -1,    32,
      -1,    33,    -1,    25,    -1,    26,    -1,    27,    -1,    50,
     113,    -1,    53,    -1,    52,    -1,    84,    -1,    -1,    54,
      -1,    54,    68,   153,    -1,    -1,    54,    -1,    54,    75,
     154,    76,   154,    -1,    54,    66,   154,    67,   154,    -1,
      54,    72,   153,    73,   154,    -1,    72,   154,    73,   154,
      -1,   101,    83,    75,    -1,    66,    -1,    67,    -1,   101,
      -1,   101,    83,    -1,   101,    83,    77,   152,    -1,   155,
     154,    76,    -1,   158,    -1,   158,    68,   159,    -1,    72,
     159,    73,    -1,    72,    73,    -1,    -1,    19,    77,    52,
      -1,    -1,   168,    -1,    66,   163,    67,    -1,   168,    -1,
     168,   163,    -1,   168,    -1,   168,   163,    -1,    -1,    49,
      72,   166,    73,    -1,    51,    -1,    51,    68,   166,    -1,
      53,    -1,    -1,    44,   167,   156,   154,   157,   165,    -1,
      48,    72,    51,   160,    73,   156,   154,    67,    -1,    42,
     174,    66,    67,    -1,    42,   174,   168,    -1,    42,   174,
      66,   163,    67,    -1,    43,    66,   164,    67,    -1,    39,
     172,   154,    64,   154,    64,   154,   171,    66,   163,    67,
      -1,    39,   172,   154,    64,   154,    64,   154,   171,   168,
      -1,    40,    75,    51,    76,   172,   154,    65,   154,    68,
     154,   171,   168,    -1,    40,    75,    51,    76,   172,   154,
      65,   154,    68,   154,   171,    66,   163,    67,    -1,    46,
     172,   154,   171,   168,   169,    -1,    46,   172,   154,   171,
      66,   163,    67,   169,    -1,    41,   172,   154,   171,   168,
      -1,    41,   172,   154,   171,    66,   163,    67,    -1,    45,
     170,    64,    -1,    -1,    47,   168,    -1,    47,    66,   163,
      67,    -1,    51,    -1,    51,    68,   170,    -1,    73,    -1,
      72,    -1,    51,   160,    -1,    51,   175,   154,   176,   160,
      -1,   173,    -1,   173,    68,   174,    -1,    75,    -1,    76,
      -1,    55,    83,    -1,    56,    83,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const unsigned short yyrline[] =
{
       0,   133,   133,   138,   141,   146,   147,   152,   153,   157,
     161,   163,   171,   175,   182,   184,   189,   190,   194,   196,
     198,   200,   202,   204,   206,   208,   210,   212,   214,   218,
     220,   222,   226,   228,   233,   234,   239,   240,   244,   246,
     248,   250,   252,   254,   256,   258,   260,   262,   264,   266,
     268,   270,   272,   276,   277,   279,   281,   285,   289,   291,
     295,   299,   301,   303,   305,   308,   310,   314,   316,   320,
     322,   326,   331,   332,   336,   340,   345,   346,   351,   352,
     362,   364,   368,   370,   375,   376,   380,   382,   387,   388,
     392,   397,   398,   402,   404,   408,   410,   414,   418,   420,
     424,   426,   431,   432,   436,   438,   442,   444,   448,   452,
     456,   462,   466,   468,   472,   474,   478,   482,   486,   490,
     492,   497,   498,   503,   504,   506,   510,   512,   514,   518,
     520,   524,   528,   530,   532,   534,   536,   540,   542,   547,
     565,   569,   571,   573,   574,   576,   578,   580,   584,   586,
     588,   591,   596,   598,   602,   604,   608,   612,   614,   618,
     629,   642,   644,   649,   650,   654,   656,   660,   662,   664,
     666,   668,   670,   672,   674,   676,   678,   680,   682,   686,
     688,   690,   695,   696,   698,   707,   708,   710,   716,   722,
     728,   736,   743,   751,   758,   760,   762,   764,   771,   773,
     777,   779,   784,   785,   790,   791,   793,   797,   799,   803,
     805,   810,   811,   815,   817,   821,   824,   827,   832,   846,
     848,   850,   852,   854,   857,   860,   863,   866,   868,   870,
     872,   874,   879,   880,   882,   885,   887,   891,   895,   899,
     907,   915,   917,   921,   924,   928,   932
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
  "INCLUDE", "STACKSIZE", "THREADED", "TEMPLATE", "SYNC", "IGET", 
  "EXCLUSIVE", "IMMEDIATE", "SKIPSCHED", "INLINE", "VIRTUAL", 
  "MIGRATABLE", "CREATEHERE", "CREATEHOME", "NOKEEP", "NOTRACE", "VOID", 
  "CONST", "PACKED", "VARSIZE", "ENTRY", "FOR", "FORALL", "WHILE", "WHEN", 
  "OVERLAP", "ATOMIC", "FORWARD", "IF", "ELSE", "CONNECT", "PUBLISHES", 
  "PYTHON", "IDENT", "NUMBER", "LITERAL", "CPROGRAM", "HASHIF", 
  "HASHIFDEF", "INT", "LONG", "SHORT", "CHAR", "FLOAT", "DOUBLE", 
  "UNSIGNED", "';'", "':'", "'{'", "'}'", "','", "'<'", "'>'", "'*'", 
  "'('", "')'", "'&'", "'['", "']'", "'='", "$accept", "File", 
  "ModuleEList", "OptExtern", "OptSemiColon", "Name", "QualName", 
  "Module", "ConstructEList", "ConstructList", "Construct", "TParam", 
  "TParamList", "TParamEList", "OptTParams", "BuiltinType", "NamedType", 
  "QualNamedType", "SimpleType", "OnePtrType", "PtrType", "FuncType", 
  "BaseType", "Type", "ArrayDim", "Dim", "DimList", "Readonly", 
  "ReadonlyMsg", "OptVoid", "MAttribs", "MAttribList", "MAttrib", 
  "CAttribs", "CAttribList", "PythonOptions", "ArrayAttrib", 
  "ArrayAttribs", "ArrayAttribList", "CAttrib", "Var", "VarList", 
  "Message", "OptBaseList", "BaseList", "Chare", "Group", "NodeGroup", 
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
     315,   316,   317,   318,    59,    58,   123,   125,    44,    60,
      62,    42,    40,    41,    38,    91,    93,    61
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const unsigned char yyr1[] =
{
       0,    78,    79,    80,    80,    81,    81,    82,    82,    83,
      84,    84,    85,    85,    86,    86,    87,    87,    88,    88,
      88,    88,    88,    88,    88,    88,    88,    88,    88,    89,
      89,    89,    90,    90,    91,    91,    92,    92,    93,    93,
      93,    93,    93,    93,    93,    93,    93,    93,    93,    93,
      93,    93,    93,    94,    95,    96,    96,    97,    98,    98,
      99,   100,   100,   100,   100,   100,   100,   101,   101,   102,
     102,   103,   104,   104,   105,   106,   107,   107,   108,   108,
     109,   109,   110,   110,   111,   111,   112,   112,   113,   113,
     114,   115,   115,   116,   116,   117,   117,   118,   119,   119,
     120,   120,   121,   121,   122,   122,   123,   123,   124,   125,
     126,   126,   127,   127,   128,   128,   129,   130,   131,   132,
     132,   133,   133,   134,   134,   134,   135,   135,   135,   136,
     136,   137,   138,   138,   138,   138,   138,   139,   139,   140,
     140,   141,   141,   141,   141,   141,   141,   141,   142,   142,
     142,   142,   143,   143,   144,   144,   145,   146,   146,   147,
     147,   148,   148,   149,   149,   150,   150,   151,   151,   151,
     151,   151,   151,   151,   151,   151,   151,   151,   151,   152,
     152,   152,   153,   153,   153,   154,   154,   154,   154,   154,
     154,   155,   156,   157,   158,   158,   158,   158,   159,   159,
     160,   160,   161,   161,   162,   162,   162,   163,   163,   164,
     164,   165,   165,   166,   166,   167,   167,   168,   168,   168,
     168,   168,   168,   168,   168,   168,   168,   168,   168,   168,
     168,   168,   169,   169,   169,   170,   170,   171,   172,   173,
     173,   174,   174,   175,   176,   177,   178
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const unsigned char yyr2[] =
{
       0,     2,     1,     0,     2,     0,     1,     0,     1,     1,
       1,     4,     3,     3,     1,     4,     0,     2,     5,     2,
       2,     3,     2,     2,     2,     2,     2,     1,     1,     1,
       1,     1,     1,     3,     0,     1,     0,     3,     1,     1,
       1,     1,     2,     2,     3,     3,     2,     2,     2,     1,
       1,     2,     1,     2,     2,     1,     1,     2,     2,     2,
       8,     1,     1,     1,     1,     2,     2,     2,     1,     1,
       1,     3,     0,     2,     4,     5,     0,     1,     0,     3,
       1,     3,     1,     1,     0,     3,     1,     3,     0,     1,
       1,     0,     3,     1,     3,     1,     1,     5,     1,     2,
       3,     6,     0,     2,     1,     3,     5,     5,     5,     5,
       4,     3,     6,     6,     5,     5,     5,     5,     5,     4,
       7,     0,     2,     0,     2,     2,     3,     2,     3,     1,
       3,     4,     2,     2,     2,     2,     2,     1,     4,     0,
       2,     2,     2,     2,     2,     3,     3,     3,     3,     6,
       3,     6,     3,     6,     1,     3,     1,     2,     1,     7,
       5,     1,     1,     0,     3,     1,     3,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     2,     1,
       1,     1,     0,     1,     3,     0,     1,     5,     5,     5,
       4,     3,     1,     1,     1,     2,     4,     3,     1,     3,
       3,     2,     0,     3,     0,     1,     3,     1,     2,     1,
       2,     0,     4,     1,     3,     1,     0,     6,     8,     4,
       3,     5,     4,    11,     9,    12,    14,     6,     8,     5,
       7,     3,     0,     2,     4,     1,     3,     1,     1,     2,
       5,     1,     3,     1,     1,     2,     2
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const unsigned char yydefact[] =
{
       3,     0,     0,     0,     2,     3,     9,     0,     0,     1,
       4,    14,     5,    12,    13,     6,     0,     0,     0,     0,
       5,    27,    28,   245,   246,     0,    76,    76,    76,     0,
      84,    84,    84,    84,     0,    78,     0,     0,     0,     5,
      19,     0,     0,     0,    22,    23,    24,    25,     0,    26,
      20,     0,     0,     7,    17,     0,    52,     0,    10,    38,
      39,    40,    41,    49,    50,     0,    36,    55,    56,    61,
      62,    63,    64,    68,     0,    77,     0,     0,     0,   154,
       0,     0,     0,     0,     0,     0,     0,     0,    91,     0,
       0,     0,   156,     0,     0,     0,   141,   142,    21,    84,
      84,    84,    84,     0,    78,   132,   133,   134,   135,   136,
     144,   143,     8,    15,     0,    65,    48,    51,    42,    43,
      46,    47,     0,    34,    54,    57,    58,    59,    66,     0,
      67,    72,   150,   148,   152,     0,   145,    95,    96,     0,
      86,    36,   102,   102,   102,   102,    90,     0,     0,    93,
       0,     0,     0,     0,     0,    82,    83,     0,    80,   100,
     147,   146,     0,    64,     0,   129,     0,     7,     0,     0,
       0,     0,     0,     0,     0,    44,    45,     0,    30,    31,
      32,    35,     0,    29,     0,     0,    72,    74,    76,    76,
      76,   155,    85,     0,    53,     0,     0,     0,     0,     0,
       0,   111,     0,    92,   102,   102,    79,     0,     0,   121,
       0,   127,   123,     0,   131,    18,   102,   102,   102,   102,
     102,     0,    75,    11,     0,    37,     0,    69,    70,     0,
      73,     0,     0,     0,    87,   104,   103,   137,   139,   106,
     107,   108,   109,   110,    94,     0,     0,    81,     0,    98,
       0,     0,   126,   124,   125,   128,   130,     0,     0,     0,
       0,     0,   119,     0,    33,     0,    71,   151,   149,   153,
       0,   163,     0,   158,   139,     0,   112,   113,     0,    99,
     101,   122,   114,   115,   116,   117,   118,     0,     0,   105,
       0,     0,     7,   140,   157,     0,     0,   194,   185,   198,
       0,   167,   168,   169,   170,   175,   176,   177,   171,   172,
     173,   174,    88,     0,   165,    52,    10,     0,     0,   162,
       0,   138,     0,   120,   195,   186,   185,     0,     0,    60,
      89,   178,   164,     0,     0,   204,     0,    97,   191,     0,
     185,   182,   185,     0,   197,   199,   166,   201,     0,     0,
       0,     0,     0,     0,   216,     0,     0,     0,     0,   160,
     205,   202,   180,   179,   181,   196,     0,   183,     0,     0,
     185,   200,   238,   185,     0,   185,     0,   241,     0,     0,
     215,     0,   235,     0,   185,     0,     0,   207,     0,   204,
     185,   182,   185,   185,   190,     0,     0,     0,   243,   239,
     185,     0,     0,   220,     0,   209,   192,   185,     0,   231,
       0,     0,   206,   208,     0,   159,   188,   184,   189,   187,
     185,     0,   237,     0,     0,   242,   219,     0,   222,   210,
       0,   236,     0,     0,   203,     0,   185,     0,   229,   244,
       0,   221,   193,   211,     0,   232,     0,   185,     0,     0,
     240,     0,   217,     0,     0,   227,   185,     0,   185,   230,
       0,   232,     0,   233,     0,     0,     0,   213,     0,   228,
       0,   218,     0,   224,   185,     0,   212,   234,     0,     0,
     214,   223,     0,     0,   225,     0,   226
};

/* YYDEFGOTO[NTERM-NUM]. */
static const short yydefgoto[] =
{
      -1,     3,     4,    18,   113,   141,    66,     5,    13,    19,
      20,   180,   181,   182,   124,    67,   235,    68,    69,    70,
      71,    72,    73,   248,   229,   186,   187,    41,    42,    76,
      90,   157,   158,    82,   139,   331,   149,    87,   150,   140,
     249,   250,    43,   196,   236,    44,    45,    46,    88,    47,
     105,   106,   107,   108,   109,   252,   211,   165,   166,    48,
      49,   239,   272,   273,    51,    52,    80,    93,   274,   275,
     320,   291,   313,   314,   365,   368,   327,   298,   407,   443,
     299,   300,   335,   389,   359,   386,   404,   452,   468,   381,
     387,   455,   383,   423,   373,   377,   378,   400,   440,    21,
      22
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -400
static const short yypact[] =
{
     111,     6,     6,    67,  -400,   111,  -400,    55,    55,  -400,
    -400,  -400,    72,  -400,  -400,  -400,     6,     6,   242,    23,
      72,  -400,  -400,  -400,  -400,   263,    61,    61,    61,   100,
      36,    36,    36,    36,    48,    84,     6,   109,   124,    72,
    -400,   102,   130,   133,  -400,  -400,  -400,  -400,   166,  -400,
    -400,   134,   135,   138,  -400,   345,  -400,   327,  -400,  -400,
      38,  -400,  -400,  -400,  -400,   131,    32,  -400,  -400,   154,
     156,   157,  -400,   -28,   100,  -400,   100,   100,   100,    45,
     140,   -20,     6,     6,     6,     6,   -18,   162,   163,    95,
       6,   141,  -400,   142,   276,   164,  -400,  -400,  -400,    36,
      36,    36,    36,   162,    84,  -400,  -400,  -400,  -400,  -400,
    -400,  -400,  -400,  -400,   158,   -12,  -400,  -400,  -400,    80,
    -400,  -400,   175,   314,  -400,  -400,  -400,  -400,  -400,   170,
    -400,   -34,   -29,    10,    19,   100,  -400,  -400,  -400,   185,
     174,   193,   200,   200,   200,   200,  -400,     6,   190,   201,
     192,    97,     6,   220,     6,  -400,  -400,   195,   204,   207,
    -400,  -400,     6,    -5,     6,   206,   205,   138,     6,     6,
       6,     6,     6,     6,     6,  -400,  -400,   225,  -400,  -400,
     212,  -400,   211,  -400,     6,   104,   208,  -400,    61,    61,
      61,  -400,  -400,   -20,  -400,     6,    60,    60,    60,    60,
     209,  -400,   220,  -400,   200,   200,  -400,    95,   327,   210,
     112,  -400,   214,   276,  -400,  -400,   200,   200,   200,   200,
     200,    77,  -400,  -400,   314,  -400,   213,  -400,   217,   216,
    -400,   215,   222,   236,  -400,   221,  -400,  -400,   226,  -400,
    -400,  -400,  -400,  -400,  -400,    60,    60,  -400,     6,   327,
     232,   327,  -400,  -400,  -400,  -400,  -400,    60,    60,    60,
      60,    60,  -400,   327,  -400,   245,  -400,  -400,  -400,  -400,
       6,   240,   251,  -400,   226,   264,  -400,  -400,   254,  -400,
    -400,  -400,  -400,  -400,  -400,  -400,  -400,   265,   327,  -400,
     320,   358,   138,  -400,  -400,   255,   266,     6,   -17,   273,
     281,  -400,  -400,  -400,  -400,  -400,  -400,  -400,  -400,  -400,
    -400,  -400,   231,   279,   288,   306,   286,   287,   154,  -400,
       6,  -400,   296,  -400,    70,    -4,   -17,   292,   327,  -400,
    -400,  -400,  -400,   320,   243,   393,   287,  -400,  -400,    52,
     -17,   309,   -17,   291,  -400,  -400,  -400,  -400,   307,   297,
     308,   297,   330,   316,   338,   342,   297,   322,   178,  -400,
    -400,   376,  -400,  -400,   217,  -400,   331,   329,   326,   324,
     -17,  -400,  -400,   -17,   350,   -17,    -6,   343,   403,   178,
    -400,   344,   346,   348,   -17,   362,   363,   178,   375,   393,
     -17,   309,   -17,   -17,  -400,   389,   364,   381,  -400,  -400,
     -17,   330,   383,  -400,   388,   178,  -400,   -17,   342,  -400,
     381,   287,  -400,  -400,   404,  -400,  -400,  -400,  -400,  -400,
     -17,   297,  -400,   422,   382,  -400,  -400,   390,  -400,  -400,
     412,  -400,   432,   387,  -400,   417,   -17,   178,  -400,  -400,
     287,  -400,  -400,   433,   178,   436,   344,   -17,   419,   418,
    -400,   414,  -400,   420,   451,  -400,   -17,   381,   -17,  -400,
     438,   436,   178,  -400,   441,   461,   442,   443,   439,  -400,
     446,  -400,   178,  -400,   -17,   438,  -400,  -400,   447,   381,
    -400,  -400,   480,   178,  -400,   448,  -400
};

/* YYPGOTO[NTERM-NUM].  */
static const short yypgoto[] =
{
    -400,  -400,   511,  -400,  -162,    -1,   -27,   500,   521,     1,
    -400,  -400,   310,  -400,   391,  -400,   -65,  -400,   -52,   239,
    -400,   -88,   474,   -21,  -400,  -400,   347,  -400,  -400,   -14,
     431,   332,  -400,    85,   349,  -400,  -400,   449,   334,  -400,
    -400,  -211,  -400,    -9,   268,  -400,  -400,  -400,   -58,  -400,
    -400,  -400,  -400,  -400,  -400,  -400,   328,  -400,   335,  -400,
    -400,   -45,   267,   525,  -400,  -400,   409,  -400,  -400,  -400,
    -400,  -400,   218,  -400,  -400,   159,  -314,  -400,    99,  -400,
    -400,  -240,  -328,  -400,   160,  -363,  -400,  -400,    78,  -400,
    -325,    86,   144,  -399,  -329,  -400,   153,  -400,  -400,  -400,
    -400
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -162
static const short yytable[] =
{
       7,     8,    79,   114,    74,   215,   163,   128,   361,   137,
     360,   432,   343,    77,    78,    23,    24,   142,   143,   144,
     145,    54,   375,   128,   413,   159,   366,   384,   369,   152,
     138,   122,   146,     6,   147,    91,   122,   325,   279,   427,
      95,   185,   429,   188,   129,   172,   130,   131,   399,   132,
     133,   134,   287,   403,   405,   326,   394,     6,   465,   395,
     129,   397,   340,  -123,   360,  -123,   334,     9,   341,   398,
     410,   342,   210,   164,   449,   122,   416,    15,   418,   419,
     482,   453,   189,   433,   122,   148,   424,   204,   345,   205,
      53,   190,   436,   430,   348,    75,   116,   122,   438,   470,
     117,   123,   183,    58,   362,   363,   435,   445,    79,   478,
     122,    81,   450,   135,     1,     2,    83,    84,    85,    11,
     485,    12,   448,    86,   237,   163,   238,    16,    17,   463,
     321,   155,   156,   457,   197,   198,   199,   175,   176,   -16,
     473,   262,   464,   263,   466,   338,   200,   339,     6,   147,
     148,    58,   240,   241,   242,    58,   227,   484,   228,    89,
     479,   209,    92,   212,   253,   254,    96,   216,   217,   218,
     219,   220,   221,   222,   231,   232,   233,    99,   100,   101,
     102,   103,   104,   226,   168,   169,   170,   171,   118,   119,
     120,   121,   164,    94,    97,   245,   246,    98,   110,   111,
     276,   277,   112,   183,   136,   160,   161,   257,   258,   259,
     260,   261,   282,   283,   284,   285,   286,   349,   350,   351,
     352,   353,   354,   355,   356,   125,   357,   126,   127,   174,
     281,   167,    25,    26,    27,    28,    29,   151,   153,   318,
     177,   184,   193,    36,    37,     1,     2,   278,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    36,
      37,   192,   123,    38,   271,   195,   201,   297,   203,   202,
     146,   206,   207,   208,   213,   214,   223,    56,    57,    55,
     224,   225,   122,   185,   330,   243,   265,   251,   267,   270,
     317,   210,   266,   162,    58,   268,   324,    56,    57,   280,
      59,    60,    61,    62,    63,    64,    65,   297,    39,   269,
      56,    57,   364,   297,    58,   290,   347,   288,   292,   336,
      59,    60,    61,    62,    63,    64,    65,    58,   294,   295,
     323,   322,   296,    59,    60,    61,    62,    63,    64,    65,
     301,   328,   302,   303,   304,   305,   306,   307,    56,    57,
     308,   309,   310,   311,   329,   332,   333,  -161,    -9,   334,
     337,    56,    57,   367,   370,    58,   178,   179,   344,   372,
     312,    59,    60,    61,    62,    63,    64,    65,    58,    56,
     371,   376,   379,   374,    59,    60,    61,    62,    63,    64,
      65,   380,   315,   382,   385,   388,    58,   391,   390,   392,
     393,   396,    59,    60,    61,    62,    63,    64,    65,   316,
     406,   401,   409,   411,   408,    59,    60,    61,    62,    63,
      64,    65,   349,   350,   351,   352,   353,   354,   355,   356,
     412,   357,   349,   350,   351,   352,   353,   354,   355,   356,
     421,   357,   349,   350,   351,   352,   353,   354,   355,   356,
     426,   357,   414,   420,   422,   428,   434,   441,   439,   358,
     446,   349,   350,   351,   352,   353,   354,   355,   356,   402,
     357,   349,   350,   351,   352,   353,   354,   355,   356,   442,
     357,   447,   451,   454,   458,   459,   460,   461,   437,   467,
     349,   350,   351,   352,   353,   354,   355,   356,   444,   357,
     349,   350,   351,   352,   353,   354,   355,   356,   471,   357,
     474,   475,   476,   477,   481,   486,    10,   462,    40,   349,
     350,   351,   352,   353,   354,   355,   356,   472,   357,    14,
     319,   115,   194,   230,   264,   173,   244,   154,   289,   247,
     255,   293,   234,    50,   191,   456,   483,   469,   256,   415,
     417,   346,   431,   480,   425
};

static const unsigned short yycheck[] =
{
       1,     2,    29,    55,    25,   167,    94,    35,   336,    29,
     335,   410,   326,    27,    28,    16,    17,    82,    83,    84,
      85,    20,   351,    35,   387,    90,   340,   356,   342,    87,
      50,    65,    50,    51,    52,    36,    65,    54,   249,   402,
      39,    75,   405,    72,    72,   103,    74,    74,   376,    76,
      77,    78,   263,   378,   379,    72,   370,    51,   457,   373,
      72,   375,    66,    68,   389,    70,    72,     0,    72,    75,
     384,    75,    77,    94,   437,    65,   390,     5,   392,   393,
     479,   444,    72,   411,    65,    86,   400,   152,   328,   154,
      67,    72,   421,   407,   334,    34,    58,    65,   423,   462,
      62,    69,   123,    51,    52,    53,   420,   432,   135,   472,
      65,    75,   440,    68,     3,     4,    31,    32,    33,    64,
     483,    66,   436,    75,    64,   213,    66,    55,    56,   454,
     292,    36,    37,   447,   143,   144,   145,    57,    58,    67,
     465,    64,   456,    66,   458,    75,   147,    77,    51,    52,
     151,    51,   197,   198,   199,    51,    52,   482,   185,    75,
     474,   162,    53,   164,    52,    53,    64,   168,   169,   170,
     171,   172,   173,   174,   188,   189,   190,    11,    12,    13,
      14,    15,    16,   184,    99,   100,   101,   102,    57,    58,
      59,    60,   213,    69,    64,   204,   205,    64,    64,    64,
     245,   246,    64,   224,    64,    64,    64,   216,   217,   218,
     219,   220,   257,   258,   259,   260,   261,    39,    40,    41,
      42,    43,    44,    45,    46,    71,    48,    71,    71,    71,
     251,    67,     6,     7,     8,     9,    10,    75,    75,   291,
      65,    71,    68,    17,    18,     3,     4,   248,     6,     7,
       8,     9,    10,    11,    12,    13,    14,    15,    16,    17,
      18,    76,    69,    21,    38,    65,    76,   288,    76,    68,
      50,    76,    68,    66,    68,    70,    51,    34,    35,    16,
      68,    70,    65,    75,    53,    76,    73,    77,    73,    68,
     291,    77,    76,    17,    51,    73,   297,    34,    35,    67,
      57,    58,    59,    60,    61,    62,    63,   328,    66,    73,
      34,    35,   339,   334,    51,    75,    73,    72,    67,   320,
      57,    58,    59,    60,    61,    62,    63,    51,    64,    75,
      64,    76,    67,    57,    58,    59,    60,    61,    62,    63,
      20,    68,    22,    23,    24,    25,    26,    27,    34,    35,
      30,    31,    32,    33,    73,    76,    68,    51,    72,    72,
      64,    34,    35,    54,    73,    51,    52,    53,    76,    72,
      50,    57,    58,    59,    60,    61,    62,    63,    51,    34,
      73,    51,    66,    75,    57,    58,    59,    60,    61,    62,
      63,    53,    34,    51,    72,    19,    51,    68,    67,    73,
      76,    51,    57,    58,    59,    60,    61,    62,    63,    51,
      66,    68,    64,    51,    68,    57,    58,    59,    60,    61,
      62,    63,    39,    40,    41,    42,    43,    44,    45,    46,
      67,    48,    39,    40,    41,    42,    43,    44,    45,    46,
      76,    48,    39,    40,    41,    42,    43,    44,    45,    46,
      67,    48,    77,    64,    73,    67,    52,    67,    76,    66,
      73,    39,    40,    41,    42,    43,    44,    45,    46,    66,
      48,    39,    40,    41,    42,    43,    44,    45,    46,    67,
      48,    64,    49,    47,    65,    67,    72,    67,    66,    51,
      39,    40,    41,    42,    43,    44,    45,    46,    66,    48,
      39,    40,    41,    42,    43,    44,    45,    46,    67,    48,
      68,    68,    73,    67,    67,    67,     5,    66,    18,    39,
      40,    41,    42,    43,    44,    45,    46,    66,    48,     8,
     291,    57,   141,   186,   224,   104,   202,    88,   270,   207,
     212,   274,   193,    18,   135,   446,    66,   461,   213,   389,
     391,   333,   408,   475,   401
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const unsigned char yystos[] =
{
       0,     3,     4,    79,    80,    85,    51,    83,    83,     0,
      80,    64,    66,    86,    86,     5,    55,    56,    81,    87,
      88,   177,   178,    83,    83,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    17,    18,    21,    66,
      85,   105,   106,   120,   123,   124,   125,   127,   137,   138,
     141,   142,   143,    67,    87,    16,    34,    35,    51,    57,
      58,    59,    60,    61,    62,    63,    84,    93,    95,    96,
      97,    98,    99,   100,   101,    34,   107,   107,   107,    84,
     144,    75,   111,   111,   111,   111,    75,   115,   126,    75,
     108,    83,    53,   145,    69,    87,    64,    64,    64,    11,
      12,    13,    14,    15,    16,   128,   129,   130,   131,   132,
      64,    64,    64,    82,    96,   100,    58,    62,    57,    58,
      59,    60,    65,    69,    92,    71,    71,    71,    35,    72,
      74,    84,    84,    84,    84,    68,    64,    29,    50,   112,
     117,    83,    94,    94,    94,    94,    50,    52,    83,   114,
     116,    75,   126,    75,   115,    36,    37,   109,   110,    94,
      64,    64,    17,    99,   101,   135,   136,    67,   111,   111,
     111,   111,   126,   108,    71,    57,    58,    65,    52,    53,
      89,    90,    91,   101,    71,    75,   103,   104,    72,    72,
      72,   144,    76,    68,    92,    65,   121,   121,   121,   121,
      83,    76,    68,    76,    94,    94,    76,    68,    66,    83,
      77,   134,    83,    68,    70,    82,    83,    83,    83,    83,
      83,    83,    83,    51,    68,    70,    83,    52,    84,   102,
     104,   107,   107,   107,   112,    94,   122,    64,    66,   139,
     139,   139,   139,    76,   116,   121,   121,   109,   101,   118,
     119,    77,   133,    52,    53,   134,   136,   121,   121,   121,
     121,   121,    64,    66,    90,    73,    76,    73,    73,    73,
      68,    38,   140,   141,   146,   147,   139,   139,    83,   119,
      67,   101,   139,   139,   139,   139,   139,   119,    72,   122,
      75,   149,    67,   140,    64,    75,    67,   101,   155,   158,
     159,    20,    22,    23,    24,    25,    26,    27,    30,    31,
      32,    33,    50,   150,   151,    34,    51,    83,    96,    97,
     148,    82,    76,    64,    83,    54,    72,   154,    68,    73,
      53,   113,    76,    68,    72,   160,    83,    64,    75,    77,
      66,    72,    75,   154,    76,   159,   150,    73,   159,    39,
      40,    41,    42,    43,    44,    45,    46,    48,    66,   162,
     168,   160,    52,    53,    84,   152,   154,    54,   153,   154,
      73,    73,    72,   172,    75,   172,    51,   173,   174,    66,
      53,   167,    51,   170,   172,    72,   163,   168,    19,   161,
      67,    68,    73,    76,   154,   154,    51,   154,    75,   160,
     175,    68,    66,   168,   164,   168,    66,   156,    68,    64,
     154,    51,    67,   163,    77,   162,   154,   153,   154,   154,
      64,    76,    73,   171,   154,   174,    67,   163,    67,   163,
     154,   170,   171,   160,    52,   154,   172,    66,   168,    76,
     176,    67,    67,   157,    66,   168,    73,    64,   154,   163,
     160,    49,   165,   163,    47,   169,   156,   154,    65,    67,
      72,    67,    66,   168,   154,   171,   154,    51,   166,   169,
     163,    67,    66,   168,    68,    68,    73,    67,   163,   154,
     166,    67,   171,    66,   168,   163,    67
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
#line 134 "xi-grammar.y"
    { yyval.modlist = yyvsp[0].modlist; modlist = yyvsp[0].modlist; }
    break;

  case 3:
#line 138 "xi-grammar.y"
    { 
		  yyval.modlist = 0; 
		}
    break;

  case 4:
#line 142 "xi-grammar.y"
    { yyval.modlist = new ModuleList(lineno, yyvsp[-1].module, yyvsp[0].modlist); }
    break;

  case 5:
#line 146 "xi-grammar.y"
    { yyval.intval = 0; }
    break;

  case 6:
#line 148 "xi-grammar.y"
    { yyval.intval = 1; }
    break;

  case 7:
#line 152 "xi-grammar.y"
    { yyval.intval = 0; }
    break;

  case 8:
#line 154 "xi-grammar.y"
    { yyval.intval = 1; }
    break;

  case 9:
#line 158 "xi-grammar.y"
    { yyval.strval = yyvsp[0].strval; }
    break;

  case 10:
#line 162 "xi-grammar.y"
    { yyval.strval = yyvsp[0].strval; }
    break;

  case 11:
#line 164 "xi-grammar.y"
    {
		  char *tmp = new char[strlen(yyvsp[-3].strval)+strlen(yyvsp[0].strval)+3];
		  sprintf(tmp,"%s::%s", yyvsp[-3].strval, yyvsp[0].strval);
		  yyval.strval = tmp;
		}
    break;

  case 12:
#line 172 "xi-grammar.y"
    { 
		    yyval.module = new Module(lineno, yyvsp[-1].strval, yyvsp[0].conslist); 
		}
    break;

  case 13:
#line 176 "xi-grammar.y"
    {  
		    yyval.module = new Module(lineno, yyvsp[-1].strval, yyvsp[0].conslist); 
		    yyval.module->setMain();
		}
    break;

  case 14:
#line 183 "xi-grammar.y"
    { yyval.conslist = 0; }
    break;

  case 15:
#line 185 "xi-grammar.y"
    { yyval.conslist = yyvsp[-2].conslist; }
    break;

  case 16:
#line 189 "xi-grammar.y"
    { yyval.conslist = 0; }
    break;

  case 17:
#line 191 "xi-grammar.y"
    { yyval.conslist = new ConstructList(lineno, yyvsp[-1].construct, yyvsp[0].conslist); }
    break;

  case 18:
#line 195 "xi-grammar.y"
    { if(yyvsp[-2].conslist) yyvsp[-2].conslist->setExtern(yyvsp[-4].intval); yyval.construct = yyvsp[-2].conslist; }
    break;

  case 19:
#line 197 "xi-grammar.y"
    { yyvsp[0].module->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].module; }
    break;

  case 20:
#line 199 "xi-grammar.y"
    { yyvsp[0].member->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].member; }
    break;

  case 21:
#line 201 "xi-grammar.y"
    { yyvsp[-1].message->setExtern(yyvsp[-2].intval); yyval.construct = yyvsp[-1].message; }
    break;

  case 22:
#line 203 "xi-grammar.y"
    { yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; }
    break;

  case 23:
#line 205 "xi-grammar.y"
    { yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; }
    break;

  case 24:
#line 207 "xi-grammar.y"
    { yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; }
    break;

  case 25:
#line 209 "xi-grammar.y"
    { yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; }
    break;

  case 26:
#line 211 "xi-grammar.y"
    { yyvsp[0].templat->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].templat; }
    break;

  case 27:
#line 213 "xi-grammar.y"
    { yyval.construct = NULL; }
    break;

  case 28:
#line 215 "xi-grammar.y"
    { yyval.construct = NULL; }
    break;

  case 29:
#line 219 "xi-grammar.y"
    { yyval.tparam = new TParamType(yyvsp[0].type); }
    break;

  case 30:
#line 221 "xi-grammar.y"
    { yyval.tparam = new TParamVal(yyvsp[0].strval); }
    break;

  case 31:
#line 223 "xi-grammar.y"
    { yyval.tparam = new TParamVal(yyvsp[0].strval); }
    break;

  case 32:
#line 227 "xi-grammar.y"
    { yyval.tparlist = new TParamList(yyvsp[0].tparam); }
    break;

  case 33:
#line 229 "xi-grammar.y"
    { yyval.tparlist = new TParamList(yyvsp[-2].tparam, yyvsp[0].tparlist); }
    break;

  case 34:
#line 233 "xi-grammar.y"
    { yyval.tparlist = 0; }
    break;

  case 35:
#line 235 "xi-grammar.y"
    { yyval.tparlist = yyvsp[0].tparlist; }
    break;

  case 36:
#line 239 "xi-grammar.y"
    { yyval.tparlist = 0; }
    break;

  case 37:
#line 241 "xi-grammar.y"
    { yyval.tparlist = yyvsp[-1].tparlist; }
    break;

  case 38:
#line 245 "xi-grammar.y"
    { yyval.type = new BuiltinType("int"); }
    break;

  case 39:
#line 247 "xi-grammar.y"
    { yyval.type = new BuiltinType("long"); }
    break;

  case 40:
#line 249 "xi-grammar.y"
    { yyval.type = new BuiltinType("short"); }
    break;

  case 41:
#line 251 "xi-grammar.y"
    { yyval.type = new BuiltinType("char"); }
    break;

  case 42:
#line 253 "xi-grammar.y"
    { yyval.type = new BuiltinType("unsigned int"); }
    break;

  case 43:
#line 255 "xi-grammar.y"
    { yyval.type = new BuiltinType("unsigned long"); }
    break;

  case 44:
#line 257 "xi-grammar.y"
    { yyval.type = new BuiltinType("unsigned long"); }
    break;

  case 45:
#line 259 "xi-grammar.y"
    { yyval.type = new BuiltinType("unsigned long long"); }
    break;

  case 46:
#line 261 "xi-grammar.y"
    { yyval.type = new BuiltinType("unsigned short"); }
    break;

  case 47:
#line 263 "xi-grammar.y"
    { yyval.type = new BuiltinType("unsigned char"); }
    break;

  case 48:
#line 265 "xi-grammar.y"
    { yyval.type = new BuiltinType("long long"); }
    break;

  case 49:
#line 267 "xi-grammar.y"
    { yyval.type = new BuiltinType("float"); }
    break;

  case 50:
#line 269 "xi-grammar.y"
    { yyval.type = new BuiltinType("double"); }
    break;

  case 51:
#line 271 "xi-grammar.y"
    { yyval.type = new BuiltinType("long double"); }
    break;

  case 52:
#line 273 "xi-grammar.y"
    { yyval.type = new BuiltinType("void"); }
    break;

  case 53:
#line 276 "xi-grammar.y"
    { yyval.ntype = new NamedType(yyvsp[-1].strval,yyvsp[0].tparlist); }
    break;

  case 54:
#line 277 "xi-grammar.y"
    { yyval.ntype = new NamedType(yyvsp[-1].strval,yyvsp[0].tparlist); }
    break;

  case 55:
#line 280 "xi-grammar.y"
    { yyval.type = yyvsp[0].type; }
    break;

  case 56:
#line 282 "xi-grammar.y"
    { yyval.type = yyvsp[0].ntype; }
    break;

  case 57:
#line 286 "xi-grammar.y"
    { yyval.ptype = new PtrType(yyvsp[-1].type); }
    break;

  case 58:
#line 290 "xi-grammar.y"
    { yyvsp[-1].ptype->indirect(); yyval.ptype = yyvsp[-1].ptype; }
    break;

  case 59:
#line 292 "xi-grammar.y"
    { yyvsp[-1].ptype->indirect(); yyval.ptype = yyvsp[-1].ptype; }
    break;

  case 60:
#line 296 "xi-grammar.y"
    { yyval.ftype = new FuncType(yyvsp[-7].type, yyvsp[-4].strval, yyvsp[-1].plist); }
    break;

  case 61:
#line 300 "xi-grammar.y"
    { yyval.type = yyvsp[0].type; }
    break;

  case 62:
#line 302 "xi-grammar.y"
    { yyval.type = yyvsp[0].ptype; }
    break;

  case 63:
#line 304 "xi-grammar.y"
    { yyval.type = yyvsp[0].ptype; }
    break;

  case 64:
#line 306 "xi-grammar.y"
    { yyval.type = yyvsp[0].ftype; }
    break;

  case 65:
#line 309 "xi-grammar.y"
    { yyval.type = yyvsp[0].type; }
    break;

  case 66:
#line 311 "xi-grammar.y"
    { yyval.type = yyvsp[-1].type; }
    break;

  case 67:
#line 315 "xi-grammar.y"
    { yyval.type = new ReferenceType(yyvsp[-1].type); }
    break;

  case 68:
#line 317 "xi-grammar.y"
    { yyval.type = yyvsp[0].type; }
    break;

  case 69:
#line 321 "xi-grammar.y"
    { yyval.val = new Value(yyvsp[0].strval); }
    break;

  case 70:
#line 323 "xi-grammar.y"
    { yyval.val = new Value(yyvsp[0].strval); }
    break;

  case 71:
#line 327 "xi-grammar.y"
    { yyval.val = yyvsp[-1].val; }
    break;

  case 72:
#line 331 "xi-grammar.y"
    { yyval.vallist = 0; }
    break;

  case 73:
#line 333 "xi-grammar.y"
    { yyval.vallist = new ValueList(yyvsp[-1].val, yyvsp[0].vallist); }
    break;

  case 74:
#line 337 "xi-grammar.y"
    { yyval.readonly = new Readonly(lineno, yyvsp[-2].type, yyvsp[-1].strval, yyvsp[0].vallist); }
    break;

  case 75:
#line 341 "xi-grammar.y"
    { yyval.readonly = new Readonly(lineno, yyvsp[-2].type, yyvsp[0].strval, 0, 1); }
    break;

  case 76:
#line 345 "xi-grammar.y"
    { yyval.intval = 0;}
    break;

  case 77:
#line 347 "xi-grammar.y"
    { yyval.intval = 0;}
    break;

  case 78:
#line 351 "xi-grammar.y"
    { yyval.intval = 0; }
    break;

  case 79:
#line 353 "xi-grammar.y"
    { 
		  /*
		  printf("Warning: Message attributes are being phased out.\n");
		  printf("Warning: Please remove them from interface files.\n");
		  */
		  yyval.intval = yyvsp[-1].intval; 
		}
    break;

  case 80:
#line 363 "xi-grammar.y"
    { yyval.intval = yyvsp[0].intval; }
    break;

  case 81:
#line 365 "xi-grammar.y"
    { yyval.intval = yyvsp[-2].intval | yyvsp[0].intval; }
    break;

  case 82:
#line 369 "xi-grammar.y"
    { yyval.intval = 0; }
    break;

  case 83:
#line 371 "xi-grammar.y"
    { yyval.intval = 0; }
    break;

  case 84:
#line 375 "xi-grammar.y"
    { yyval.cattr = 0; }
    break;

  case 85:
#line 377 "xi-grammar.y"
    { yyval.cattr = yyvsp[-1].cattr; }
    break;

  case 86:
#line 381 "xi-grammar.y"
    { yyval.cattr = yyvsp[0].cattr; }
    break;

  case 87:
#line 383 "xi-grammar.y"
    { yyval.cattr = yyvsp[-2].cattr | yyvsp[0].cattr; }
    break;

  case 88:
#line 387 "xi-grammar.y"
    { python_doc = NULL; yyval.intval = 0; }
    break;

  case 89:
#line 389 "xi-grammar.y"
    { python_doc = yyvsp[0].strval; yyval.intval = 0; }
    break;

  case 90:
#line 393 "xi-grammar.y"
    { yyval.cattr = Chare::CPYTHON; }
    break;

  case 91:
#line 397 "xi-grammar.y"
    { yyval.cattr = 0; }
    break;

  case 92:
#line 399 "xi-grammar.y"
    { yyval.cattr = yyvsp[-1].cattr; }
    break;

  case 93:
#line 403 "xi-grammar.y"
    { yyval.cattr = yyvsp[0].cattr; }
    break;

  case 94:
#line 405 "xi-grammar.y"
    { yyval.cattr = yyvsp[-2].cattr | yyvsp[0].cattr; }
    break;

  case 95:
#line 409 "xi-grammar.y"
    { yyval.cattr = Chare::CMIGRATABLE; }
    break;

  case 96:
#line 411 "xi-grammar.y"
    { yyval.cattr = Chare::CPYTHON; }
    break;

  case 97:
#line 415 "xi-grammar.y"
    { yyval.mv = new MsgVar(yyvsp[-4].type, yyvsp[-3].strval); }
    break;

  case 98:
#line 419 "xi-grammar.y"
    { yyval.mvlist = new MsgVarList(yyvsp[0].mv); }
    break;

  case 99:
#line 421 "xi-grammar.y"
    { yyval.mvlist = new MsgVarList(yyvsp[-1].mv, yyvsp[0].mvlist); }
    break;

  case 100:
#line 425 "xi-grammar.y"
    { yyval.message = new Message(lineno, yyvsp[0].ntype); }
    break;

  case 101:
#line 427 "xi-grammar.y"
    { yyval.message = new Message(lineno, yyvsp[-3].ntype, yyvsp[-1].mvlist); }
    break;

  case 102:
#line 431 "xi-grammar.y"
    { yyval.typelist = 0; }
    break;

  case 103:
#line 433 "xi-grammar.y"
    { yyval.typelist = yyvsp[0].typelist; }
    break;

  case 104:
#line 437 "xi-grammar.y"
    { yyval.typelist = new TypeList(yyvsp[0].ntype); }
    break;

  case 105:
#line 439 "xi-grammar.y"
    { yyval.typelist = new TypeList(yyvsp[-2].ntype, yyvsp[0].typelist); }
    break;

  case 106:
#line 443 "xi-grammar.y"
    { yyval.chare = new Chare(lineno, yyvsp[-3].cattr, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 107:
#line 445 "xi-grammar.y"
    { yyval.chare = new MainChare(lineno, yyvsp[-3].cattr, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 108:
#line 449 "xi-grammar.y"
    { yyval.chare = new Group(lineno, yyvsp[-3].cattr, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 109:
#line 453 "xi-grammar.y"
    { yyval.chare = new NodeGroup(lineno, yyvsp[-3].cattr, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 110:
#line 457 "xi-grammar.y"
    {/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",yyvsp[-2].strval);
			yyval.ntype = new NamedType(buf); 
		}
    break;

  case 111:
#line 463 "xi-grammar.y"
    { yyval.ntype = new NamedType(yyvsp[-1].strval); }
    break;

  case 112:
#line 467 "xi-grammar.y"
    {  yyval.chare = new Array(lineno, yyvsp[-4].cattr, yyvsp[-3].ntype, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 113:
#line 469 "xi-grammar.y"
    {  yyval.chare = new Array(lineno, yyvsp[-3].cattr, yyvsp[-4].ntype, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 114:
#line 473 "xi-grammar.y"
    { yyval.chare = new Chare(lineno, yyvsp[-3].cattr, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist);}
    break;

  case 115:
#line 475 "xi-grammar.y"
    { yyval.chare = new MainChare(lineno, yyvsp[-3].cattr, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 116:
#line 479 "xi-grammar.y"
    { yyval.chare = new Group(lineno, yyvsp[-3].cattr, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 117:
#line 483 "xi-grammar.y"
    { yyval.chare = new NodeGroup( lineno, yyvsp[-3].cattr, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 118:
#line 487 "xi-grammar.y"
    { yyval.chare = new Array( lineno, 0, yyvsp[-3].ntype, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 119:
#line 491 "xi-grammar.y"
    { yyval.message = new Message(lineno, new NamedType(yyvsp[-1].strval)); }
    break;

  case 120:
#line 493 "xi-grammar.y"
    { yyval.message = new Message(lineno, new NamedType(yyvsp[-4].strval), yyvsp[-2].mvlist); }
    break;

  case 121:
#line 497 "xi-grammar.y"
    { yyval.type = 0; }
    break;

  case 122:
#line 499 "xi-grammar.y"
    { yyval.type = yyvsp[0].type; }
    break;

  case 123:
#line 503 "xi-grammar.y"
    { yyval.strval = 0; }
    break;

  case 124:
#line 505 "xi-grammar.y"
    { yyval.strval = yyvsp[0].strval; }
    break;

  case 125:
#line 507 "xi-grammar.y"
    { yyval.strval = yyvsp[0].strval; }
    break;

  case 126:
#line 511 "xi-grammar.y"
    { yyval.tvar = new TType(new NamedType(yyvsp[-1].strval), yyvsp[0].type); }
    break;

  case 127:
#line 513 "xi-grammar.y"
    { yyval.tvar = new TFunc(yyvsp[-1].ftype, yyvsp[0].strval); }
    break;

  case 128:
#line 515 "xi-grammar.y"
    { yyval.tvar = new TName(yyvsp[-2].type, yyvsp[-1].strval, yyvsp[0].strval); }
    break;

  case 129:
#line 519 "xi-grammar.y"
    { yyval.tvarlist = new TVarList(yyvsp[0].tvar); }
    break;

  case 130:
#line 521 "xi-grammar.y"
    { yyval.tvarlist = new TVarList(yyvsp[-2].tvar, yyvsp[0].tvarlist); }
    break;

  case 131:
#line 525 "xi-grammar.y"
    { yyval.tvarlist = yyvsp[-1].tvarlist; }
    break;

  case 132:
#line 529 "xi-grammar.y"
    { yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); }
    break;

  case 133:
#line 531 "xi-grammar.y"
    { yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); }
    break;

  case 134:
#line 533 "xi-grammar.y"
    { yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); }
    break;

  case 135:
#line 535 "xi-grammar.y"
    { yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); }
    break;

  case 136:
#line 537 "xi-grammar.y"
    { yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].message); yyvsp[0].message->setTemplate(yyval.templat); }
    break;

  case 137:
#line 541 "xi-grammar.y"
    { yyval.mbrlist = 0; }
    break;

  case 138:
#line 543 "xi-grammar.y"
    { yyval.mbrlist = yyvsp[-2].mbrlist; }
    break;

  case 139:
#line 547 "xi-grammar.y"
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

  case 140:
#line 566 "xi-grammar.y"
    { yyval.mbrlist = new MemberList(yyvsp[-1].member, yyvsp[0].mbrlist); }
    break;

  case 141:
#line 570 "xi-grammar.y"
    { yyval.member = yyvsp[-1].readonly; }
    break;

  case 142:
#line 572 "xi-grammar.y"
    { yyval.member = yyvsp[-1].readonly; }
    break;

  case 144:
#line 575 "xi-grammar.y"
    { yyval.member = yyvsp[-1].member; }
    break;

  case 145:
#line 577 "xi-grammar.y"
    { yyval.member = yyvsp[-1].pupable; }
    break;

  case 146:
#line 579 "xi-grammar.y"
    { yyval.member = yyvsp[-1].includeFile; }
    break;

  case 147:
#line 581 "xi-grammar.y"
    { yyval.member = new ClassDeclaration(lineno,yyvsp[-1].strval); }
    break;

  case 148:
#line 585 "xi-grammar.y"
    { yyval.member = new InitCall(lineno, yyvsp[0].strval, 1); }
    break;

  case 149:
#line 587 "xi-grammar.y"
    { yyval.member = new InitCall(lineno, yyvsp[-3].strval, 1); }
    break;

  case 150:
#line 589 "xi-grammar.y"
    { printf("Warning: deprecated use of initcall. Use initnode or initproc instead.\n"); 
		  yyval.member = new InitCall(lineno, yyvsp[0].strval, 1); }
    break;

  case 151:
#line 592 "xi-grammar.y"
    { printf("Warning: deprecated use of initcall. Use initnode or initproc instead.\n");
		  yyval.member = new InitCall(lineno, yyvsp[-3].strval, 1); }
    break;

  case 152:
#line 597 "xi-grammar.y"
    { yyval.member = new InitCall(lineno, yyvsp[0].strval, 0); }
    break;

  case 153:
#line 599 "xi-grammar.y"
    { yyval.member = new InitCall(lineno, yyvsp[-3].strval, 0); }
    break;

  case 154:
#line 603 "xi-grammar.y"
    { yyval.pupable = new PUPableClass(lineno,yyvsp[0].strval,0); }
    break;

  case 155:
#line 605 "xi-grammar.y"
    { yyval.pupable = new PUPableClass(lineno,yyvsp[-2].strval,yyvsp[0].pupable); }
    break;

  case 156:
#line 609 "xi-grammar.y"
    { yyval.includeFile = new IncludeFile(lineno,yyvsp[0].strval); }
    break;

  case 157:
#line 613 "xi-grammar.y"
    { yyval.member = yyvsp[-1].entry; }
    break;

  case 158:
#line 615 "xi-grammar.y"
    { yyval.member = yyvsp[0].member; }
    break;

  case 159:
#line 619 "xi-grammar.y"
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

  case 160:
#line 630 "xi-grammar.y"
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

  case 161:
#line 643 "xi-grammar.y"
    { yyval.type = new BuiltinType("void"); }
    break;

  case 162:
#line 645 "xi-grammar.y"
    { yyval.type = yyvsp[0].ptype; }
    break;

  case 163:
#line 649 "xi-grammar.y"
    { yyval.intval = 0; }
    break;

  case 164:
#line 651 "xi-grammar.y"
    { yyval.intval = yyvsp[-1].intval; }
    break;

  case 165:
#line 655 "xi-grammar.y"
    { yyval.intval = yyvsp[0].intval; }
    break;

  case 166:
#line 657 "xi-grammar.y"
    { yyval.intval = yyvsp[-2].intval | yyvsp[0].intval; }
    break;

  case 167:
#line 661 "xi-grammar.y"
    { yyval.intval = STHREADED; }
    break;

  case 168:
#line 663 "xi-grammar.y"
    { yyval.intval = SSYNC; }
    break;

  case 169:
#line 665 "xi-grammar.y"
    { yyval.intval = SIGET; }
    break;

  case 170:
#line 667 "xi-grammar.y"
    { yyval.intval = SLOCKED; }
    break;

  case 171:
#line 669 "xi-grammar.y"
    { yyval.intval = SCREATEHERE; }
    break;

  case 172:
#line 671 "xi-grammar.y"
    { yyval.intval = SCREATEHOME; }
    break;

  case 173:
#line 673 "xi-grammar.y"
    { yyval.intval = SNOKEEP; }
    break;

  case 174:
#line 675 "xi-grammar.y"
    { yyval.intval = SNOTRACE; }
    break;

  case 175:
#line 677 "xi-grammar.y"
    { yyval.intval = SIMMEDIATE; }
    break;

  case 176:
#line 679 "xi-grammar.y"
    { yyval.intval = SSKIPSCHED; }
    break;

  case 177:
#line 681 "xi-grammar.y"
    { yyval.intval = SINLINE; }
    break;

  case 178:
#line 683 "xi-grammar.y"
    { yyval.intval = SPYTHON; }
    break;

  case 179:
#line 687 "xi-grammar.y"
    { yyval.val = new Value(yyvsp[0].strval); }
    break;

  case 180:
#line 689 "xi-grammar.y"
    { yyval.val = new Value(yyvsp[0].strval); }
    break;

  case 181:
#line 691 "xi-grammar.y"
    { yyval.val = new Value(yyvsp[0].strval); }
    break;

  case 182:
#line 695 "xi-grammar.y"
    { yyval.strval = ""; }
    break;

  case 183:
#line 697 "xi-grammar.y"
    { yyval.strval = yyvsp[0].strval; }
    break;

  case 184:
#line 699 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen(yyvsp[-2].strval)+strlen(yyvsp[0].strval)+3];
			sprintf(tmp,"%s, %s", yyvsp[-2].strval, yyvsp[0].strval);
			yyval.strval = tmp;
		}
    break;

  case 185:
#line 707 "xi-grammar.y"
    { yyval.strval = ""; }
    break;

  case 186:
#line 709 "xi-grammar.y"
    { yyval.strval = yyvsp[0].strval; }
    break;

  case 187:
#line 711 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen(yyvsp[-4].strval)+strlen(yyvsp[-2].strval)+strlen(yyvsp[0].strval)+3];
			sprintf(tmp,"%s[%s]%s", yyvsp[-4].strval, yyvsp[-2].strval, yyvsp[0].strval);
			yyval.strval = tmp;
		}
    break;

  case 188:
#line 717 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen(yyvsp[-4].strval)+strlen(yyvsp[-2].strval)+strlen(yyvsp[0].strval)+3];
			sprintf(tmp,"%s{%s}%s", yyvsp[-4].strval, yyvsp[-2].strval, yyvsp[0].strval);
			yyval.strval = tmp;
		}
    break;

  case 189:
#line 723 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen(yyvsp[-4].strval)+strlen(yyvsp[-2].strval)+strlen(yyvsp[0].strval)+3];
			sprintf(tmp,"%s(%s)%s", yyvsp[-4].strval, yyvsp[-2].strval, yyvsp[0].strval);
			yyval.strval = tmp;
		}
    break;

  case 190:
#line 729 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen(yyvsp[-2].strval)+strlen(yyvsp[0].strval)+3];
			sprintf(tmp,"(%s)%s", yyvsp[-2].strval, yyvsp[0].strval);
			yyval.strval = tmp;
		}
    break;

  case 191:
#line 737 "xi-grammar.y"
    {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			yyval.pname = new Parameter(lineno, yyvsp[-2].type,yyvsp[-1].strval);
		}
    break;

  case 192:
#line 744 "xi-grammar.y"
    { 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			yyval.intval = 0;
		}
    break;

  case 193:
#line 752 "xi-grammar.y"
    { 
			in_braces=0;
			yyval.intval = 0;
		}
    break;

  case 194:
#line 759 "xi-grammar.y"
    { yyval.pname = new Parameter(lineno, yyvsp[0].type);}
    break;

  case 195:
#line 761 "xi-grammar.y"
    { yyval.pname = new Parameter(lineno, yyvsp[-1].type,yyvsp[0].strval);}
    break;

  case 196:
#line 763 "xi-grammar.y"
    { yyval.pname = new Parameter(lineno, yyvsp[-3].type,yyvsp[-2].strval,0,yyvsp[0].val);}
    break;

  case 197:
#line 765 "xi-grammar.y"
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			yyval.pname = new Parameter(lineno, yyvsp[-2].pname->getType(), yyvsp[-2].pname->getName() ,yyvsp[-1].strval);
		}
    break;

  case 198:
#line 772 "xi-grammar.y"
    { yyval.plist = new ParamList(yyvsp[0].pname); }
    break;

  case 199:
#line 774 "xi-grammar.y"
    { yyval.plist = new ParamList(yyvsp[-2].pname,yyvsp[0].plist); }
    break;

  case 200:
#line 778 "xi-grammar.y"
    { yyval.plist = yyvsp[-1].plist; }
    break;

  case 201:
#line 780 "xi-grammar.y"
    { yyval.plist = 0; }
    break;

  case 202:
#line 784 "xi-grammar.y"
    { yyval.val = 0; }
    break;

  case 203:
#line 786 "xi-grammar.y"
    { yyval.val = new Value(yyvsp[0].strval); }
    break;

  case 204:
#line 790 "xi-grammar.y"
    { yyval.sc = 0; }
    break;

  case 205:
#line 792 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SSDAGENTRY, yyvsp[0].sc); }
    break;

  case 206:
#line 794 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SSDAGENTRY, yyvsp[-1].sc); }
    break;

  case 207:
#line 798 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SSLIST, yyvsp[0].sc); }
    break;

  case 208:
#line 800 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SSLIST, yyvsp[-1].sc, yyvsp[0].sc);  }
    break;

  case 209:
#line 804 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SOLIST, yyvsp[0].sc); }
    break;

  case 210:
#line 806 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SOLIST, yyvsp[-1].sc, yyvsp[0].sc); }
    break;

  case 211:
#line 810 "xi-grammar.y"
    { yyval.sc = 0; }
    break;

  case 212:
#line 812 "xi-grammar.y"
    { yyval.sc = yyvsp[-1].sc; }
    break;

  case 213:
#line 816 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SPUBLISHES, new SdagConstruct(SIDENT, yyvsp[0].strval)); }
    break;

  case 214:
#line 818 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SPUBLISHES, new SdagConstruct(SIDENT, yyvsp[-2].strval), yyvsp[0].sc);  }
    break;

  case 215:
#line 822 "xi-grammar.y"
    { yyval.strval = yyvsp[0].strval; }
    break;

  case 216:
#line 824 "xi-grammar.y"
    { yyval.strval = 0; }
    break;

  case 217:
#line 828 "xi-grammar.y"
    { RemoveSdagComments(yyvsp[-2].strval);
		   yyval.sc = new SdagConstruct(SATOMIC, new XStr(yyvsp[-2].strval), yyvsp[0].sc, 0,0,0,0, 0 ); 
		   if (yyvsp[-4].strval) { yyvsp[-4].strval[strlen(yyvsp[-4].strval)-1]=0; yyval.sc->traceName = new XStr(yyvsp[-4].strval+1); }
		 }
    break;

  case 218:
#line 833 "xi-grammar.y"
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

  case 219:
#line 847 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SWHEN, 0, 0,0,0,0,0,yyvsp[-2].entrylist); }
    break;

  case 220:
#line 849 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SWHEN, 0, 0, 0,0,0, yyvsp[0].sc, yyvsp[-1].entrylist); }
    break;

  case 221:
#line 851 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SWHEN, 0, 0, 0,0,0, yyvsp[-1].sc, yyvsp[-3].entrylist); }
    break;

  case 222:
#line 853 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SOVERLAP,0, 0,0,0,0,yyvsp[-1].sc, 0); }
    break;

  case 223:
#line 855 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SFOR, 0, new SdagConstruct(SINT_EXPR, yyvsp[-8].strval), new SdagConstruct(SINT_EXPR, yyvsp[-6].strval),
		             new SdagConstruct(SINT_EXPR, yyvsp[-4].strval), 0, yyvsp[-1].sc, 0); }
    break;

  case 224:
#line 858 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SFOR, 0, new SdagConstruct(SINT_EXPR, yyvsp[-6].strval), new SdagConstruct(SINT_EXPR, yyvsp[-4].strval), 
		         new SdagConstruct(SINT_EXPR, yyvsp[-2].strval), 0, yyvsp[0].sc, 0); }
    break;

  case 225:
#line 861 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SFORALL, 0, new SdagConstruct(SIDENT, yyvsp[-9].strval), new SdagConstruct(SINT_EXPR, yyvsp[-6].strval), 
		             new SdagConstruct(SINT_EXPR, yyvsp[-4].strval), new SdagConstruct(SINT_EXPR, yyvsp[-2].strval), yyvsp[0].sc, 0); }
    break;

  case 226:
#line 864 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SFORALL, 0, new SdagConstruct(SIDENT, yyvsp[-11].strval), new SdagConstruct(SINT_EXPR, yyvsp[-8].strval), 
		                 new SdagConstruct(SINT_EXPR, yyvsp[-6].strval), new SdagConstruct(SINT_EXPR, yyvsp[-4].strval), yyvsp[-1].sc, 0); }
    break;

  case 227:
#line 867 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, yyvsp[-3].strval), yyvsp[0].sc,0,0,yyvsp[-1].sc,0); }
    break;

  case 228:
#line 869 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, yyvsp[-5].strval), yyvsp[0].sc,0,0,yyvsp[-2].sc,0); }
    break;

  case 229:
#line 871 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, yyvsp[-2].strval), 0,0,0,yyvsp[0].sc,0); }
    break;

  case 230:
#line 873 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SWHILE, 0, new SdagConstruct(SINT_EXPR, yyvsp[-4].strval), 0,0,0,yyvsp[-1].sc,0); }
    break;

  case 231:
#line 875 "xi-grammar.y"
    { yyval.sc = yyvsp[-1].sc; }
    break;

  case 232:
#line 879 "xi-grammar.y"
    { yyval.sc = 0; }
    break;

  case 233:
#line 881 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SELSE, 0,0,0,0,0, yyvsp[0].sc,0); }
    break;

  case 234:
#line 883 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SELSE, 0,0,0,0,0, yyvsp[-1].sc,0); }
    break;

  case 235:
#line 886 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SFORWARD, new SdagConstruct(SIDENT, yyvsp[0].strval)); }
    break;

  case 236:
#line 888 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SFORWARD, new SdagConstruct(SIDENT, yyvsp[-2].strval), yyvsp[0].sc);  }
    break;

  case 237:
#line 892 "xi-grammar.y"
    { in_int_expr = 0; yyval.intval = 0; }
    break;

  case 238:
#line 896 "xi-grammar.y"
    { in_int_expr = 1; yyval.intval = 0; }
    break;

  case 239:
#line 900 "xi-grammar.y"
    { 
		  if (yyvsp[0].plist != 0)
		     yyval.entry = new Entry(lineno, 0, 0, yyvsp[-1].strval, yyvsp[0].plist, 0, 0, 0, 0); 
		  else
		     yyval.entry = new Entry(lineno, 0, 0, yyvsp[-1].strval, 
				new ParamList(new Parameter(0, new BuiltinType("void"))), 0, 0, 0, 0); 
		}
    break;

  case 240:
#line 908 "xi-grammar.y"
    { if (yyvsp[0].plist != 0)
		    yyval.entry = new Entry(lineno, 0, 0, yyvsp[-4].strval, yyvsp[0].plist, 0, 0, yyvsp[-2].strval, 0); 
		  else
		    yyval.entry = new Entry(lineno, 0, 0, yyvsp[-4].strval, new ParamList(new Parameter(0, new BuiltinType("void"))), 0, 0, yyvsp[-2].strval, 0); 
		}
    break;

  case 241:
#line 916 "xi-grammar.y"
    { yyval.entrylist = new EntryList(yyvsp[0].entry); }
    break;

  case 242:
#line 918 "xi-grammar.y"
    { yyval.entrylist = new EntryList(yyvsp[-2].entry,yyvsp[0].entrylist); }
    break;

  case 243:
#line 922 "xi-grammar.y"
    { in_bracket=1; }
    break;

  case 244:
#line 925 "xi-grammar.y"
    { in_bracket=0; }
    break;

  case 245:
#line 929 "xi-grammar.y"
    { if (!macroDefined(yyvsp[0].strval, 1)) in_comment = 1; }
    break;

  case 246:
#line 933 "xi-grammar.y"
    { if (!macroDefined(yyvsp[0].strval, 0)) in_comment = 1; }
    break;


    }

/* Line 991 of yacc.c.  */
#line 2961 "y.tab.c"

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


#line 936 "xi-grammar.y"

void yyerror(const char *mesg)
{
  cout << cur_file<<":"<<lineno<<": Charmxi syntax error> " << mesg << endl;
  // return 0;
}

