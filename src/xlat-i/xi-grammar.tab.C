/* A Bison parser, made by GNU Bison 2.3.  */

/* Skeleton implementation for Bison's Yacc-like parsers in C

   Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003, 2004, 2005, 2006
   Free Software Foundation, Inc.

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
   Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301, USA.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "2.3"

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
     CONDITIONAL = 272,
     CLASS = 273,
     INCLUDE = 274,
     STACKSIZE = 275,
     THREADED = 276,
     TEMPLATE = 277,
     SYNC = 278,
     IGET = 279,
     EXCLUSIVE = 280,
     IMMEDIATE = 281,
     SKIPSCHED = 282,
     INLINE = 283,
     VIRTUAL = 284,
     MIGRATABLE = 285,
     CREATEHERE = 286,
     CREATEHOME = 287,
     NOKEEP = 288,
     NOTRACE = 289,
     VOID = 290,
     CONST = 291,
     PACKED = 292,
     VARSIZE = 293,
     ENTRY = 294,
     FOR = 295,
     FORALL = 296,
     WHILE = 297,
     WHEN = 298,
     OVERLAP = 299,
     ATOMIC = 300,
     FORWARD = 301,
     IF = 302,
     ELSE = 303,
     CONNECT = 304,
     PUBLISHES = 305,
     PYTHON = 306,
     LOCAL = 307,
     NAMESPACE = 308,
     USING = 309,
     IDENT = 310,
     NUMBER = 311,
     LITERAL = 312,
     CPROGRAM = 313,
     HASHIF = 314,
     HASHIFDEF = 315,
     INT = 316,
     LONG = 317,
     SHORT = 318,
     CHAR = 319,
     FLOAT = 320,
     DOUBLE = 321,
     UNSIGNED = 322,
     ACCEL = 323,
     READWRITE = 324,
     WRITEONLY = 325,
     ACCELBLOCK = 326,
     MEMCRITICAL = 327,
     REDUCTIONTARGET = 328
   };
#endif
/* Tokens.  */
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
#define CONDITIONAL 272
#define CLASS 273
#define INCLUDE 274
#define STACKSIZE 275
#define THREADED 276
#define TEMPLATE 277
#define SYNC 278
#define IGET 279
#define EXCLUSIVE 280
#define IMMEDIATE 281
#define SKIPSCHED 282
#define INLINE 283
#define VIRTUAL 284
#define MIGRATABLE 285
#define CREATEHERE 286
#define CREATEHOME 287
#define NOKEEP 288
#define NOTRACE 289
#define VOID 290
#define CONST 291
#define PACKED 292
#define VARSIZE 293
#define ENTRY 294
#define FOR 295
#define FORALL 296
#define WHILE 297
#define WHEN 298
#define OVERLAP 299
#define ATOMIC 300
#define FORWARD 301
#define IF 302
#define ELSE 303
#define CONNECT 304
#define PUBLISHES 305
#define PYTHON 306
#define LOCAL 307
#define NAMESPACE 308
#define USING 309
#define IDENT 310
#define NUMBER 311
#define LITERAL 312
#define CPROGRAM 313
#define HASHIF 314
#define HASHIFDEF 315
#define INT 316
#define LONG 317
#define SHORT 318
#define CHAR 319
#define FLOAT 320
#define DOUBLE 321
#define UNSIGNED 322
#define ACCEL 323
#define READWRITE 324
#define WRITEONLY 325
#define ACCELBLOCK 326
#define MEMCRITICAL 327
#define REDUCTIONTARGET 328




/* Copy the first part of user declarations.  */
#line 2 "xi-grammar.y"

#include <iostream>
#include <string>
#include <string.h>
#include "xi-symbol.h"
#include "EToken.h"
using namespace xi;
extern int yylex (void) ;
extern unsigned char in_comment;
void yyerror(const char *);
extern unsigned int lineno;
extern int in_bracket,in_braces,in_int_expr;
extern TList<Entry *> *connectEntries;
ModuleList *modlist;
namespace xi {
extern int macroDefined(const char *str, int istrue);
extern const char *python_doc;
void splitScopedName(const char* name, const char** scope, const char** basename);
}

// Error handling
bool hasSeenConstructor = false;
char *lastConstructor;


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

/* Enabling the token table.  */
#ifndef YYTOKEN_TABLE
# define YYTOKEN_TABLE 0
#endif

#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE
#line 27 "xi-grammar.y"
{
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
  const char *strval;
  int intval;
  Chare::attrib_t cattr;
  SdagConstruct *sc;
  XStr* xstrptr;
  AccelBlock* accelBlock;
}
/* Line 193 of yacc.c.  */
#line 305 "y.tab.c"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 318 "y.tab.c"

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#elif (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
typedef signed char yytype_int8;
#else
typedef short int yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short int yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short int yytype_int16;
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif ! defined YYSIZE_T && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned int
# endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T) -1)

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(msgid) dgettext ("bison-runtime", msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(msgid) msgid
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(e) ((void) (e))
#else
# define YYUSE(e) /* empty */
#endif

/* Identity function, used to suppress warnings about constant conditions.  */
#ifndef lint
# define YYID(n) (n)
#else
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static int
YYID (int i)
#else
static int
YYID (i)
    int i;
#endif
{
  return i;
}
#endif

#if ! defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#     ifndef _STDLIB_H
#      define _STDLIB_H 1
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's `empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (YYID (0))
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined _STDLIB_H \
       && ! ((defined YYMALLOC || defined malloc) \
	     && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef _STDLIB_H
#    define _STDLIB_H 1
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */


#if (! defined yyoverflow \
     && (! defined __cplusplus \
	 || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yytype_int16 yyss;
  YYSTYPE yyvs;
  };

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

/* Copy COUNT objects from FROM to TO.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(To, From, Count) \
      __builtin_memcpy (To, From, (Count) * sizeof (*(From)))
#  else
#   define YYCOPY(To, From, Count)		\
      do					\
	{					\
	  YYSIZE_T yyi;				\
	  for (yyi = 0; yyi < (Count); yyi++)	\
	    (To)[yyi] = (From)[yyi];		\
	}					\
      while (YYID (0))
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
    while (YYID (0))

#endif

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  9
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   899

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  90
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  110
/* YYNRULES -- Number of rules.  */
#define YYNRULES  287
/* YYNRULES -- Number of states.  */
#define YYNSTATES  602

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   328

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    84,     2,
      82,    83,    81,     2,    78,    88,    89,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    75,    74,
      79,    87,    80,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    85,     2,    86,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    76,     2,    77,     2,     2,     2,     2,
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
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     5,     6,     9,    10,    12,    13,    15,
      17,    19,    24,    28,    32,    34,    39,    40,    43,    49,
      55,    60,    64,    67,    70,    74,    77,    80,    83,    86,
      89,    98,   100,   102,   104,   106,   108,   110,   112,   114,
     118,   119,   121,   122,   126,   128,   130,   132,   134,   137,
     140,   144,   148,   151,   154,   157,   159,   161,   164,   166,
     169,   172,   174,   176,   179,   182,   185,   194,   196,   198,
     200,   202,   205,   208,   211,   213,   215,   217,   221,   222,
     225,   230,   236,   237,   239,   240,   244,   246,   250,   252,
     254,   255,   259,   261,   265,   266,   268,   270,   271,   275,
     277,   281,   283,   285,   286,   288,   289,   292,   298,   300,
     303,   307,   314,   315,   318,   320,   324,   330,   336,   342,
     348,   353,   357,   364,   371,   377,   383,   389,   395,   401,
     406,   414,   415,   418,   419,   422,   425,   429,   432,   436,
     438,   442,   447,   450,   453,   456,   459,   462,   464,   469,
     470,   473,   476,   479,   482,   485,   489,   493,   497,   501,
     508,   518,   522,   529,   533,   540,   550,   560,   562,   566,
     568,   571,   575,   577,   585,   591,   604,   610,   613,   615,
     617,   618,   622,   624,   626,   630,   632,   634,   636,   638,
     640,   642,   644,   646,   648,   650,   652,   654,   657,   659,
     661,   663,   665,   667,   669,   670,   672,   676,   677,   679,
     685,   691,   697,   702,   706,   708,   710,   712,   716,   721,
     725,   727,   729,   731,   733,   738,   742,   747,   752,   757,
     761,   769,   775,   782,   784,   788,   790,   794,   798,   801,
     805,   808,   809,   813,   814,   816,   820,   822,   825,   827,
     830,   831,   836,   838,   842,   844,   845,   852,   861,   866,
     870,   876,   881,   893,   903,   916,   931,   938,   947,   953,
     961,   965,   969,   971,   972,   975,   980,   982,   986,   988,
     990,   993,   999,  1001,  1005,  1007,  1009,  1012
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
      91,     0,    -1,    92,    -1,    -1,    97,    92,    -1,    -1,
       5,    -1,    -1,    74,    -1,    55,    -1,    55,    -1,    96,
      75,    75,    55,    -1,     3,    95,    98,    -1,     4,    95,
      98,    -1,    74,    -1,    76,    99,    77,    94,    -1,    -1,
     100,    99,    -1,    93,    76,    99,    77,    94,    -1,    53,
      95,    76,    99,    77,    -1,    54,    53,    96,    74,    -1,
      54,    96,    74,    -1,    93,    97,    -1,    93,   155,    -1,
      93,   134,    74,    -1,    93,   137,    -1,    93,   138,    -1,
      93,   139,    -1,    93,   141,    -1,    93,   152,    -1,     5,
      39,   163,   107,    95,   104,   180,    74,    -1,   198,    -1,
     199,    -1,   162,    -1,     1,    -1,   113,    -1,    56,    -1,
      57,    -1,   101,    -1,   101,    78,   102,    -1,    -1,   102,
      -1,    -1,    79,   103,    80,    -1,    61,    -1,    62,    -1,
      63,    -1,    64,    -1,    67,    61,    -1,    67,    62,    -1,
      67,    62,    61,    -1,    67,    62,    62,    -1,    67,    63,
      -1,    67,    64,    -1,    62,    62,    -1,    65,    -1,    66,
      -1,    62,    66,    -1,    35,    -1,    95,   104,    -1,    96,
     104,    -1,   105,    -1,   107,    -1,   108,    81,    -1,   109,
      81,    -1,   110,    81,    -1,   112,    82,    81,    95,    83,
      82,   178,    83,    -1,   108,    -1,   109,    -1,   110,    -1,
     111,    -1,    36,   112,    -1,   112,    36,    -1,   112,    84,
      -1,   112,    -1,    56,    -1,    96,    -1,    85,   114,    86,
      -1,    -1,   115,   116,    -1,     6,   113,    96,   116,    -1,
       6,    16,   108,    81,    95,    -1,    -1,    35,    -1,    -1,
      85,   121,    86,    -1,   122,    -1,   122,    78,   121,    -1,
      37,    -1,    38,    -1,    -1,    85,   124,    86,    -1,   129,
      -1,   129,    78,   124,    -1,    -1,    57,    -1,    51,    -1,
      -1,    85,   128,    86,    -1,   126,    -1,   126,    78,   128,
      -1,    30,    -1,    51,    -1,    -1,    17,    -1,    -1,    85,
      86,    -1,   130,   113,    95,   131,    74,    -1,   132,    -1,
     132,   133,    -1,    16,   120,   106,    -1,    16,   120,   106,
      76,   133,    77,    -1,    -1,    75,   136,    -1,   107,    -1,
     107,    78,   136,    -1,    11,   123,   106,   135,   153,    -1,
      12,   123,   106,   135,   153,    -1,    13,   123,   106,   135,
     153,    -1,    14,   123,   106,   135,   153,    -1,    85,    56,
      95,    86,    -1,    85,    95,    86,    -1,    15,   127,   140,
     106,   135,   153,    -1,    15,   140,   127,   106,   135,   153,
      -1,    11,   123,    95,   135,   153,    -1,    12,   123,    95,
     135,   153,    -1,    13,   123,    95,   135,   153,    -1,    14,
     123,    95,   135,   153,    -1,    15,   140,    95,   135,   153,
      -1,    16,   120,    95,    74,    -1,    16,   120,    95,    76,
     133,    77,    74,    -1,    -1,    87,   113,    -1,    -1,    87,
      56,    -1,    87,    57,    -1,    18,    95,   147,    -1,   111,
     148,    -1,   113,    95,   148,    -1,   149,    -1,   149,    78,
     150,    -1,    22,    79,   150,    80,    -1,   151,   142,    -1,
     151,   143,    -1,   151,   144,    -1,   151,   145,    -1,   151,
     146,    -1,    74,    -1,    76,   154,    77,    94,    -1,    -1,
     160,   154,    -1,   117,    74,    -1,   118,    74,    -1,   157,
      74,    -1,   156,    74,    -1,    10,   158,    74,    -1,    19,
     159,    74,    -1,    18,    95,    74,    -1,     8,   119,    96,
      -1,     8,   119,    96,    82,   119,    83,    -1,     8,   119,
      96,    79,   102,    80,    82,   119,    83,    -1,     7,   119,
      96,    -1,     7,   119,    96,    82,   119,    83,    -1,     9,
     119,    96,    -1,     9,   119,    96,    82,   119,    83,    -1,
       9,   119,    96,    79,   102,    80,    82,   119,    83,    -1,
       9,    85,    68,    86,   119,    96,    82,   119,    83,    -1,
     107,    -1,   107,    78,   158,    -1,    57,    -1,   161,    74,
      -1,   151,   161,    74,    -1,   155,    -1,    39,   164,   163,
      95,   180,   182,   183,    -1,    39,   164,    95,   180,   183,
      -1,    39,    85,    68,    86,    35,    95,   180,   181,   171,
     169,   172,    95,    -1,    71,   171,   169,   172,    74,    -1,
      71,    74,    -1,    35,    -1,   109,    -1,    -1,    85,   165,
      86,    -1,     1,    -1,   166,    -1,   166,    78,   165,    -1,
      21,    -1,    23,    -1,    24,    -1,    25,    -1,    31,    -1,
      32,    -1,    33,    -1,    34,    -1,    26,    -1,    27,    -1,
      28,    -1,    52,    -1,    51,   125,    -1,    72,    -1,    73,
      -1,     1,    -1,    57,    -1,    56,    -1,    96,    -1,    -1,
      58,    -1,    58,    78,   168,    -1,    -1,    58,    -1,    58,
      85,   169,    86,   169,    -1,    58,    76,   169,    77,   169,
      -1,    58,    82,   168,    83,   169,    -1,    82,   169,    83,
     169,    -1,   113,    95,    85,    -1,    76,    -1,    77,    -1,
     113,    -1,   113,    95,   130,    -1,   113,    95,    87,   167,
      -1,   170,   169,    86,    -1,     6,    -1,    69,    -1,    70,
      -1,    95,    -1,   175,    88,    80,    95,    -1,   175,    89,
      95,    -1,   175,    85,   175,    86,    -1,   175,    85,    56,
      86,    -1,   175,    82,   175,    83,    -1,   170,   169,    86,
      -1,   174,    75,   113,    95,    79,   175,    80,    -1,   113,
      95,    79,   175,    80,    -1,   174,    75,   176,    79,   175,
      80,    -1,   173,    -1,   173,    78,   178,    -1,   177,    -1,
     177,    78,   179,    -1,    82,   178,    83,    -1,    82,    83,
      -1,    85,   179,    86,    -1,    85,    86,    -1,    -1,    20,
      87,    56,    -1,    -1,   189,    -1,    76,   184,    77,    -1,
     189,    -1,   189,   184,    -1,   189,    -1,   189,   184,    -1,
      -1,    50,    82,   187,    83,    -1,    55,    -1,    55,    78,
     187,    -1,    57,    -1,    -1,    45,   188,   171,   169,   172,
     186,    -1,    49,    82,    55,   180,    83,   171,   169,    77,
      -1,    43,   195,    76,    77,    -1,    43,   195,   189,    -1,
      43,   195,    76,   184,    77,    -1,    44,    76,   185,    77,
      -1,    40,   193,   169,    74,   169,    74,   169,   192,    76,
     184,    77,    -1,    40,   193,   169,    74,   169,    74,   169,
     192,   189,    -1,    41,    85,    55,    86,   193,   169,    75,
     169,    78,   169,   192,   189,    -1,    41,    85,    55,    86,
     193,   169,    75,   169,    78,   169,   192,    76,   184,    77,
      -1,    47,   193,   169,   192,   189,   190,    -1,    47,   193,
     169,   192,    76,   184,    77,   190,    -1,    42,   193,   169,
     192,   189,    -1,    42,   193,   169,   192,    76,   184,    77,
      -1,    46,   191,    74,    -1,   171,   169,   172,    -1,     1,
      -1,    -1,    48,   189,    -1,    48,    76,   184,    77,    -1,
      55,    -1,    55,    78,   191,    -1,    83,    -1,    82,    -1,
      55,   180,    -1,    55,   196,   169,   197,   180,    -1,   194,
      -1,   194,    78,   195,    -1,    85,    -1,    86,    -1,    59,
      95,    -1,    60,    95,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   158,   158,   163,   166,   171,   172,   177,   178,   182,
     186,   188,   196,   200,   207,   209,   214,   215,   219,   221,
     223,   225,   227,   229,   231,   233,   235,   237,   239,   241,
     243,   253,   255,   257,   259,   263,   265,   267,   271,   273,
     278,   279,   284,   285,   289,   291,   293,   295,   297,   299,
     301,   303,   305,   307,   309,   311,   313,   315,   317,   321,
     322,   329,   331,   335,   339,   341,   345,   349,   351,   353,
     355,   358,   360,   364,   366,   370,   372,   376,   381,   382,
     386,   390,   395,   396,   401,   402,   412,   414,   418,   420,
     425,   426,   430,   432,   437,   438,   442,   447,   448,   452,
     454,   458,   460,   465,   466,   470,   471,   474,   478,   480,
     484,   486,   491,   492,   496,   498,   502,   506,   512,   518,
     524,   530,   534,   538,   544,   548,   554,   560,   566,   572,
     574,   579,   580,   585,   586,   588,   592,   594,   596,   600,
     602,   606,   610,   612,   614,   616,   618,   622,   624,   629,
     647,   651,   653,   655,   656,   658,   660,   662,   666,   668,
     670,   676,   679,   684,   686,   688,   694,   702,   704,   707,
     711,   713,   718,   722,   730,   754,   772,   774,   778,   780,
     785,   786,   788,   792,   794,   798,   800,   802,   804,   806,
     808,   810,   812,   814,   816,   818,   820,   822,   824,   826,
     828,   832,   834,   836,   841,   842,   844,   853,   854,   856,
     862,   868,   874,   882,   889,   897,   904,   906,   908,   910,
     917,   918,   919,   922,   923,   924,   925,   932,   938,   947,
     954,   960,   966,   974,   976,   980,   982,   986,   988,   992,
     994,   999,  1000,  1005,  1006,  1008,  1012,  1014,  1018,  1020,
    1025,  1026,  1030,  1032,  1036,  1039,  1042,  1046,  1060,  1062,
    1064,  1066,  1068,  1071,  1074,  1077,  1080,  1082,  1084,  1086,
    1088,  1090,  1092,  1099,  1100,  1102,  1105,  1107,  1111,  1115,
    1119,  1121,  1125,  1127,  1131,  1134,  1138,  1142
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "MODULE", "MAINMODULE", "EXTERN",
  "READONLY", "INITCALL", "INITNODE", "INITPROC", "PUPABLE", "CHARE",
  "MAINCHARE", "GROUP", "NODEGROUP", "ARRAY", "MESSAGE", "CONDITIONAL",
  "CLASS", "INCLUDE", "STACKSIZE", "THREADED", "TEMPLATE", "SYNC", "IGET",
  "EXCLUSIVE", "IMMEDIATE", "SKIPSCHED", "INLINE", "VIRTUAL", "MIGRATABLE",
  "CREATEHERE", "CREATEHOME", "NOKEEP", "NOTRACE", "VOID", "CONST",
  "PACKED", "VARSIZE", "ENTRY", "FOR", "FORALL", "WHILE", "WHEN",
  "OVERLAP", "ATOMIC", "FORWARD", "IF", "ELSE", "CONNECT", "PUBLISHES",
  "PYTHON", "LOCAL", "NAMESPACE", "USING", "IDENT", "NUMBER", "LITERAL",
  "CPROGRAM", "HASHIF", "HASHIFDEF", "INT", "LONG", "SHORT", "CHAR",
  "FLOAT", "DOUBLE", "UNSIGNED", "ACCEL", "READWRITE", "WRITEONLY",
  "ACCELBLOCK", "MEMCRITICAL", "REDUCTIONTARGET", "';'", "':'", "'{'",
  "'}'", "','", "'<'", "'>'", "'*'", "'('", "')'", "'&'", "'['", "']'",
  "'='", "'-'", "'.'", "$accept", "File", "ModuleEList", "OptExtern",
  "OptSemiColon", "Name", "QualName", "Module", "ConstructEList",
  "ConstructList", "Construct", "TParam", "TParamList", "TParamEList",
  "OptTParams", "BuiltinType", "NamedType", "QualNamedType", "SimpleType",
  "OnePtrType", "PtrType", "FuncType", "BaseType", "Type", "ArrayDim",
  "Dim", "DimList", "Readonly", "ReadonlyMsg", "OptVoid", "MAttribs",
  "MAttribList", "MAttrib", "CAttribs", "CAttribList", "PythonOptions",
  "ArrayAttrib", "ArrayAttribs", "ArrayAttribList", "CAttrib",
  "OptConditional", "MsgArray", "Var", "VarList", "Message", "OptBaseList",
  "BaseList", "Chare", "Group", "NodeGroup", "ArrayIndexType", "Array",
  "TChare", "TGroup", "TNodeGroup", "TArray", "TMessage", "OptTypeInit",
  "OptNameInit", "TVar", "TVarList", "TemplateSpec", "Template",
  "MemberEList", "MemberList", "NonEntryMember", "InitNode", "InitProc",
  "PUPableClass", "IncludeFile", "Member", "Entry", "AccelBlock",
  "EReturn", "EAttribs", "EAttribList", "EAttrib", "DefaultParameter",
  "CPROGRAM_List", "CCode", "ParamBracketStart", "ParamBraceStart",
  "ParamBraceEnd", "Parameter", "AccelBufferType", "AccelInstName",
  "AccelArrayParam", "AccelParameter", "ParamList", "AccelParamList",
  "EParameters", "AccelEParameters", "OptStackSize", "OptSdagCode",
  "Slist", "Olist", "OptPubList", "PublishesList", "OptTraceName",
  "SingleConstruct", "HasElse", "ForwardList", "EndIntExpr",
  "StartIntExpr", "SEntry", "SEntryList", "SParamBracketStart",
  "SParamBracketEnd", "HashIFComment", "HashIFDefComment", 0
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
     315,   316,   317,   318,   319,   320,   321,   322,   323,   324,
     325,   326,   327,   328,    59,    58,   123,   125,    44,    60,
      62,    42,    40,    41,    38,    91,    93,    61,    45,    46
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    90,    91,    92,    92,    93,    93,    94,    94,    95,
      96,    96,    97,    97,    98,    98,    99,    99,   100,   100,
     100,   100,   100,   100,   100,   100,   100,   100,   100,   100,
     100,   100,   100,   100,   100,   101,   101,   101,   102,   102,
     103,   103,   104,   104,   105,   105,   105,   105,   105,   105,
     105,   105,   105,   105,   105,   105,   105,   105,   105,   106,
     107,   108,   108,   109,   110,   110,   111,   112,   112,   112,
     112,   112,   112,   113,   113,   114,   114,   115,   116,   116,
     117,   118,   119,   119,   120,   120,   121,   121,   122,   122,
     123,   123,   124,   124,   125,   125,   126,   127,   127,   128,
     128,   129,   129,   130,   130,   131,   131,   132,   133,   133,
     134,   134,   135,   135,   136,   136,   137,   137,   138,   139,
     140,   140,   141,   141,   142,   142,   143,   144,   145,   146,
     146,   147,   147,   148,   148,   148,   149,   149,   149,   150,
     150,   151,   152,   152,   152,   152,   152,   153,   153,   154,
     154,   155,   155,   155,   155,   155,   155,   155,   156,   156,
     156,   156,   156,   157,   157,   157,   157,   158,   158,   159,
     160,   160,   160,   161,   161,   161,   162,   162,   163,   163,
     164,   164,   164,   165,   165,   166,   166,   166,   166,   166,
     166,   166,   166,   166,   166,   166,   166,   166,   166,   166,
     166,   167,   167,   167,   168,   168,   168,   169,   169,   169,
     169,   169,   169,   170,   171,   172,   173,   173,   173,   173,
     174,   174,   174,   175,   175,   175,   175,   175,   175,   176,
     177,   177,   177,   178,   178,   179,   179,   180,   180,   181,
     181,   182,   182,   183,   183,   183,   184,   184,   185,   185,
     186,   186,   187,   187,   188,   188,   189,   189,   189,   189,
     189,   189,   189,   189,   189,   189,   189,   189,   189,   189,
     189,   189,   189,   190,   190,   190,   191,   191,   192,   193,
     194,   194,   195,   195,   196,   197,   198,   199
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     0,     2,     0,     1,     0,     1,     1,
       1,     4,     3,     3,     1,     4,     0,     2,     5,     5,
       4,     3,     2,     2,     3,     2,     2,     2,     2,     2,
       8,     1,     1,     1,     1,     1,     1,     1,     1,     3,
       0,     1,     0,     3,     1,     1,     1,     1,     2,     2,
       3,     3,     2,     2,     2,     1,     1,     2,     1,     2,
       2,     1,     1,     2,     2,     2,     8,     1,     1,     1,
       1,     2,     2,     2,     1,     1,     1,     3,     0,     2,
       4,     5,     0,     1,     0,     3,     1,     3,     1,     1,
       0,     3,     1,     3,     0,     1,     1,     0,     3,     1,
       3,     1,     1,     0,     1,     0,     2,     5,     1,     2,
       3,     6,     0,     2,     1,     3,     5,     5,     5,     5,
       4,     3,     6,     6,     5,     5,     5,     5,     5,     4,
       7,     0,     2,     0,     2,     2,     3,     2,     3,     1,
       3,     4,     2,     2,     2,     2,     2,     1,     4,     0,
       2,     2,     2,     2,     2,     3,     3,     3,     3,     6,
       9,     3,     6,     3,     6,     9,     9,     1,     3,     1,
       2,     3,     1,     7,     5,    12,     5,     2,     1,     1,
       0,     3,     1,     1,     3,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     2,     1,     1,
       1,     1,     1,     1,     0,     1,     3,     0,     1,     5,
       5,     5,     4,     3,     1,     1,     1,     3,     4,     3,
       1,     1,     1,     1,     4,     3,     4,     4,     4,     3,
       7,     5,     6,     1,     3,     1,     3,     3,     2,     3,
       2,     0,     3,     0,     1,     3,     1,     2,     1,     2,
       0,     4,     1,     3,     1,     0,     6,     8,     4,     3,
       5,     4,    11,     9,    12,    14,     6,     8,     5,     7,
       3,     3,     1,     0,     2,     4,     1,     3,     1,     1,
       2,     5,     1,     3,     1,     1,     2,     2
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       3,     0,     0,     0,     2,     3,     9,     0,     0,     1,
       4,    14,     0,    12,    13,    34,     6,     0,     0,     0,
       0,     0,     0,     0,     0,    33,    31,    32,     0,     0,
       0,    10,     0,   286,   287,   177,   214,   207,     0,    82,
      82,    82,     0,    90,    90,    90,    90,     0,    84,     0,
       0,     0,     0,    22,     0,     0,     0,    25,    26,    27,
      28,     0,    29,    23,     0,     0,     7,    17,    58,    44,
      45,    46,    47,    55,    56,     0,    42,    61,    62,     0,
     179,     0,     0,     0,    21,     0,   208,   207,     0,     0,
      58,     0,    67,    68,    69,    70,    74,     0,    83,     0,
       0,     0,     0,   167,     0,     0,     0,     0,     0,     0,
       0,     0,    97,     0,     0,     0,   169,     0,     0,     0,
     151,   152,    24,    90,    90,    90,    90,     0,    84,   142,
     143,   144,   145,   146,   154,   153,     8,    15,    54,    57,
      48,    49,    52,    53,    40,    60,    63,     0,     0,    20,
       0,   207,   204,   207,     0,   215,     0,     0,    71,    64,
      65,    72,     0,    73,    78,   161,   158,     0,   163,     0,
     155,   101,   102,     0,    92,    42,   112,   112,   112,   112,
      96,     0,     0,    99,     0,     0,     0,     0,     0,    88,
      89,     0,    86,   110,   157,   156,     0,    70,     0,   139,
       0,     7,     0,     0,     0,     0,     0,     0,    50,    51,
      36,    37,    38,    41,     0,    35,    42,    19,    11,     0,
     205,     0,     0,   207,   176,     0,     0,     0,    78,    80,
      82,     0,    82,    82,     0,    82,   168,    91,     0,    59,
       0,     0,     0,     0,     0,     0,   121,     0,    98,   112,
     112,    85,     0,   103,   131,     0,   137,   133,     0,   141,
      18,   112,   112,   112,   112,   112,     0,     0,    43,     0,
     207,   204,   207,   207,   212,    81,     0,    75,    76,     0,
      79,     0,     0,     0,     0,     0,     0,    93,   114,   113,
     147,   149,   116,   117,   118,   119,   120,   100,     0,     0,
      87,   104,     0,   103,     0,     0,   136,   134,   135,   138,
     140,     0,     0,     0,     0,     0,   129,   103,    39,     0,
       0,   210,   206,   211,   209,     0,    77,   162,     0,   159,
       0,     0,   164,     0,     0,     0,     0,   172,   149,     0,
     122,   123,     0,   109,   111,   132,   124,   125,   126,   127,
     128,     0,   238,   216,   207,   233,     0,    30,     0,    82,
      82,    82,   115,   182,     0,     0,     0,     7,   150,   170,
     105,     0,   103,     0,     0,   237,     0,     0,     0,     0,
     200,   185,   186,   187,   188,   193,   194,   195,   189,   190,
     191,   192,    94,   196,     0,   198,   199,     0,   183,    10,
       0,     0,   171,   148,     0,     0,   130,   213,     0,   217,
     219,   234,    66,   160,   166,   165,    95,   197,     0,   181,
       0,     0,     0,   106,   107,   202,   201,   203,   218,     0,
     184,   272,     0,     0,     0,     0,     0,   255,     0,     0,
       0,     0,   207,   174,   244,   241,     0,   279,   207,     0,
     207,     0,   282,     0,     0,   254,     0,   276,     0,   207,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     284,   280,   207,     0,     0,   259,     0,     0,   207,     0,
     270,     0,     0,   245,   247,   271,     0,   173,     0,     0,
     207,     0,   278,     0,     0,   283,   258,     0,   261,   249,
       0,   277,     0,     0,   242,   220,   221,   222,   240,     0,
       0,   235,     0,   207,     0,   207,     0,   268,   285,     0,
     260,   250,     0,   273,     0,     0,     0,     0,   239,     0,
     207,     0,     0,   281,     0,   256,     0,     0,   266,   207,
       0,     0,   207,     0,   236,     0,     0,   207,   269,     0,
     273,     0,   274,     0,   223,     0,     0,     0,     0,   175,
       0,     0,   252,     0,   267,     0,   257,   231,     0,     0,
       0,     0,     0,   229,     0,     0,   263,   207,     0,   251,
     275,     0,     0,     0,     0,   225,     0,   232,     0,     0,
     253,   228,   227,   226,   224,   230,   262,     0,     0,   264,
       0,   265
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     3,     4,    22,   137,   175,    76,     5,    13,    23,
      24,   212,   213,   214,   145,    77,   176,    78,    92,    93,
      94,    95,    96,   215,   279,   228,   229,    54,    55,    99,
     114,   191,   192,   106,   173,   417,   183,   111,   184,   174,
     302,   405,   303,   304,    56,   241,   289,    57,    58,    59,
     112,    60,   129,   130,   131,   132,   133,   306,   256,   199,
     200,   335,    62,   292,   336,   337,    64,    65,   104,   117,
     338,   339,    25,    81,   365,   397,   398,   428,   221,    88,
     354,   442,   156,   355,   510,   555,   543,   511,   356,   512,
     320,   489,   465,   443,   461,   476,   535,   563,   456,   462,
     538,   458,   493,   448,   452,   453,   472,   519,    26,    27
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -471
static const yytype_int16 yypact[] =
{
      57,   -18,   -18,    82,  -471,    57,  -471,    42,    42,  -471,
    -471,  -471,   407,  -471,  -471,  -471,    60,   -18,   138,   -18,
     -18,    71,   680,    36,   407,  -471,  -471,  -471,   612,    84,
     115,  -471,   191,  -471,  -471,  -471,  -471,   -28,   723,   140,
     140,   -13,   115,    93,    93,    93,    93,    96,   101,   -18,
     137,   131,   407,  -471,   148,   156,   173,  -471,  -471,  -471,
    -471,   342,  -471,  -471,   176,   180,   205,  -471,   165,  -471,
      72,  -471,  -471,  -471,  -471,   287,   117,  -471,  -471,   206,
    -471,   115,   407,   194,  -471,   221,    51,   -28,   223,   811,
    -471,   798,   206,   222,   224,  -471,   -15,   115,  -471,   115,
     115,   236,   115,   229,   246,    12,   -18,   -18,   -18,   -18,
     200,   255,   259,   235,   -18,   254,  -471,   286,   761,   284,
    -471,  -471,  -471,    93,    93,    93,    93,   255,   101,  -471,
    -471,  -471,  -471,  -471,  -471,  -471,  -471,  -471,  -471,  -471,
    -471,   216,  -471,  -471,   775,  -471,  -471,   -18,   285,  -471,
     308,   -28,   320,   -28,   283,  -471,   305,   299,    -2,  -471,
    -471,  -471,   300,  -471,     0,     4,    75,   296,    77,   115,
    -471,  -471,  -471,   297,   306,   307,   310,   310,   310,   310,
    -471,   -18,   301,   312,   302,   230,   -18,   340,   -18,  -471,
    -471,   311,   314,   317,  -471,  -471,   -18,    61,   -18,   316,
     319,   205,   -18,   -18,   -18,   -18,   -18,   -18,  -471,  -471,
    -471,  -471,   318,  -471,   321,  -471,   307,  -471,  -471,   325,
     331,   323,   338,   -28,  -471,   -18,   -18,   243,   346,  -471,
     140,   775,   140,   140,   775,   140,  -471,  -471,    12,  -471,
     115,   133,   133,   133,   133,   341,  -471,   340,  -471,   310,
     310,  -471,   235,   386,   347,   270,  -471,   348,   761,  -471,
    -471,   310,   310,   310,   310,   310,   153,   775,  -471,   351,
     -28,   320,   -28,   -28,  -471,  -471,   353,  -471,   368,   358,
    -471,   362,   366,   365,   115,   369,   372,  -471,   374,  -471,
    -471,   432,  -471,  -471,  -471,  -471,  -471,  -471,   133,   133,
    -471,  -471,   798,    -3,   380,   798,  -471,  -471,  -471,  -471,
    -471,   133,   133,   133,   133,   133,  -471,   386,  -471,   197,
     388,  -471,  -471,  -471,  -471,   381,  -471,  -471,   383,  -471,
      33,   387,  -471,   115,   100,   429,   396,  -471,   432,   400,
    -471,  -471,   -18,  -471,  -471,  -471,  -471,  -471,  -471,  -471,
    -471,   398,  -471,   -18,   -28,   399,   393,  -471,   798,   140,
     140,   140,  -471,  -471,   696,   832,   405,   205,  -471,  -471,
     395,   408,    11,   401,   798,  -471,   402,   403,   406,   410,
    -471,  -471,  -471,  -471,  -471,  -471,  -471,  -471,  -471,  -471,
    -471,  -471,   424,  -471,   404,  -471,  -471,   409,   418,   425,
     351,   -18,  -471,  -471,   423,   414,  -471,  -471,   181,  -471,
    -471,  -471,  -471,  -471,  -471,  -471,  -471,  -471,   484,  -471,
     710,   471,   351,  -471,  -471,  -471,  -471,   368,  -471,   -18,
    -471,  -471,   439,   437,   439,   468,   448,   469,   470,   439,
     445,   199,   -28,  -471,  -471,   508,   351,  -471,   -28,   474,
     -28,    15,   452,   509,   535,  -471,   455,   454,   461,   -28,
     482,   464,   328,   223,   451,   471,   458,   472,   456,   477,
    -471,  -471,   -28,   468,   248,  -471,   485,   457,   -28,   470,
    -471,   477,   351,  -471,  -471,  -471,   505,  -471,   247,   455,
     -28,   439,  -471,   547,   478,  -471,  -471,   486,  -471,  -471,
     223,  -471,   558,   483,  -471,  -471,  -471,  -471,  -471,   -18,
     490,   491,   488,   -28,   498,   -28,   199,  -471,  -471,   351,
    -471,   536,   199,   560,   455,   516,   798,   739,  -471,   223,
     -28,   531,   532,  -471,   538,  -471,   545,   572,  -471,   -28,
     -18,   -18,   -28,   546,  -471,   -18,   477,   -28,  -471,   569,
     560,   199,  -471,   549,  -471,    32,    92,   541,   -18,  -471,
     596,   550,   551,   548,  -471,   553,  -471,  -471,   -18,   280,
     552,   -18,   -18,  -471,    94,   199,  -471,   -28,   569,  -471,
    -471,   249,   571,   233,   -18,  -471,   146,  -471,   556,   477,
    -471,  -471,  -471,  -471,  -471,  -471,  -471,   609,   199,  -471,
     567,  -471
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -471,  -471,   630,  -471,  -194,    -1,   -10,   624,   651,    -9,
    -471,  -471,  -176,  -471,  -122,  -471,   -57,   -32,   -23,   -22,
    -471,  -109,   570,   -35,  -471,  -471,   434,  -471,  -471,   -14,
     537,   411,  -471,   -21,   422,  -471,  -471,   554,   417,  -471,
     298,  -471,  -471,  -270,  -471,  -139,   335,  -471,  -471,  -471,
     -65,  -471,  -471,  -471,  -471,  -471,  -471,  -471,   412,  -471,
     413,   658,  -471,  -100,   343,   660,  -471,  -471,   534,  -471,
    -471,   370,  -471,   336,  -471,   288,  -471,  -471,   433,   -83,
     174,   -19,  -422,  -471,  -471,  -400,  -471,  -471,  -293,   179,
    -387,  -471,  -471,   242,  -445,  -471,  -471,   132,  -471,  -409,
     159,   234,  -470,  -403,  -471,   239,  -471,  -471,  -471,  -471
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -249
static const yytype_int16 yytable[] =
{
       7,     8,    37,    97,   154,    79,    80,   260,    32,   197,
     103,   502,   444,   421,   301,    67,    29,   484,    33,    34,
      83,   161,    98,   107,   108,   109,   100,   102,   301,   497,
      86,   450,   499,   343,   161,   445,   459,     6,   242,   243,
     244,   485,   171,   119,   475,   477,   186,   351,   115,   147,
     177,   178,   179,   239,    87,   282,   444,   193,   285,   466,
       1,     2,   206,   172,   471,   376,   157,   162,   219,   163,
     222,   532,   101,   148,  -108,    85,   560,   536,   521,    85,
     162,   411,     9,   198,   517,   227,   230,   164,   515,   165,
     166,   318,   168,   523,   269,   503,   407,   319,   408,    28,
     470,   363,   202,   203,   204,   205,   565,   545,    85,   182,
     298,   299,   567,    66,   568,   360,    11,   569,    12,   597,
     570,   571,   311,   312,   313,   314,   315,   151,   552,   249,
     588,   250,   533,   152,   138,  -180,   153,   103,   139,  -133,
     274,  -133,   293,   294,   295,    35,   216,    36,   255,   197,
      85,   576,    85,   600,   231,  -180,   234,   232,   574,   235,
      82,  -180,  -180,  -180,  -180,  -180,  -180,  -180,   581,   583,
      31,   572,   586,   403,   587,    98,   568,   407,   105,   569,
     245,   110,   570,   571,   182,   364,   113,   321,   599,   323,
     324,    30,    85,    31,   116,   254,   144,   257,   340,   341,
     431,   261,   262,   263,   264,   265,   266,   290,   288,   291,
     118,   346,   347,   348,   349,   350,   281,   278,   283,   284,
    -178,   286,   120,   198,   275,   276,   595,   316,   568,   317,
     121,   569,    90,    91,   570,   571,    31,   425,   426,   432,
     433,   434,   435,   436,   437,   438,   439,   122,   440,   431,
     134,   180,    31,   505,   135,     6,   181,  -214,    69,    70,
      71,    72,    73,    74,    75,    84,    85,   342,   149,    85,
     345,   373,   189,   190,   330,    36,  -214,   208,   209,   136,
     352,  -214,    90,    91,   353,     6,   181,   146,   432,   433,
     434,   435,   436,   437,   438,   439,   150,   440,    31,   277,
     155,   288,    31,   159,   167,   160,  -214,   169,    69,    70,
      71,    72,    73,    74,    75,   568,   506,   507,   569,   593,
     170,   570,   571,   353,    36,   496,   307,   308,   194,   431,
    -214,   568,   591,   508,   569,     6,   582,   570,   571,   353,
     185,   370,    79,    80,   187,   377,   378,   379,   140,   141,
     142,   143,   372,   123,   124,   125,   126,   127,   128,   463,
     195,   201,   217,   218,   400,   467,   223,   469,   432,   433,
     434,   435,   436,   437,   438,   439,   481,   440,   220,   224,
     225,   226,   233,   237,   238,   240,   144,   246,   248,   494,
     247,   180,   252,   253,   258,   500,   267,   251,   427,   259,
     422,   268,   270,   301,    36,  -246,   272,   514,    15,   271,
      -5,    -5,    16,    -5,    -5,    -5,    -5,    -5,    -5,    -5,
      -5,    -5,    -5,    -5,   273,    -5,    -5,   296,   446,    -5,
     529,   227,   531,   319,   305,   255,   325,   478,    38,    39,
      40,    41,    42,    85,   326,   327,   328,   546,   329,   331,
      49,    50,   333,   509,    51,   332,   553,   344,   431,   557,
      17,    18,   357,   358,   561,   359,    19,    20,   334,   361,
     513,   334,   431,   367,   369,   371,   375,   374,    21,   402,
     404,   416,   406,    -5,   -16,   412,   413,   410,   424,   414,
     418,   541,   509,   415,   589,   419,   420,   432,   433,   434,
     435,   436,   437,   438,   439,   539,   440,    -9,   525,   423,
     431,   432,   433,   434,   435,   436,   437,   438,   439,   429,
     440,   447,   449,   451,   454,   457,   455,   460,   464,   468,
     473,    36,   479,    36,  -248,   480,   431,   482,   486,   554,
     556,   483,   491,   488,   559,  -243,   490,   441,   431,   432,
     433,   434,   435,   436,   437,   438,   439,   554,   440,   431,
     492,   504,   498,   520,   518,   526,   524,   554,   554,   527,
     585,   554,   530,   431,   528,   432,   433,   434,   435,   436,
     437,   438,   439,   594,   440,   474,   534,   432,   433,   434,
     435,   436,   437,   438,   439,   540,   440,   431,   432,   433,
     434,   435,   436,   437,   438,   439,   547,   440,   537,   548,
     431,    36,   432,   433,   434,   435,   436,   437,   438,   439,
     549,   440,   550,   516,   562,   558,   566,   573,   577,   578,
     580,   579,   584,   596,   522,    10,   432,   433,   434,   435,
     436,   437,   438,   439,   601,   440,    53,    68,   551,   432,
     433,   434,   435,   436,   437,   438,   439,   592,   440,    14,
     287,   158,   280,   300,   297,   207,   188,    31,   362,   309,
     409,   310,   575,    69,    70,    71,    72,    73,    74,    75,
      61,   368,    63,     1,     2,   598,    38,    39,    40,    41,
      42,    43,    44,    45,    46,    47,    48,   380,    49,    50,
     542,   401,    51,   236,   322,   366,   544,   487,   430,   564,
     590,   380,   495,   501,     0,     0,     0,   381,     0,   382,
     383,   384,   385,   386,   387,     0,     0,   388,   389,   390,
     391,   381,     0,   382,   383,   384,   385,   386,   387,    89,
       0,   388,   389,   390,   391,   505,     0,   392,   393,     0,
       0,     0,     0,     0,     0,     0,    52,     0,    90,    91,
       0,   392,   393,     0,   394,     0,     0,     0,   395,   396,
       0,     0,     0,     0,    90,    91,     0,     0,    31,   196,
       0,     0,   395,   396,    69,    70,    71,    72,    73,    74,
      75,     0,     0,     0,    31,     0,    90,    91,     0,     0,
      69,    70,    71,    72,    73,    74,    75,     0,   506,   507,
      90,    91,     0,     0,     0,     0,    31,     0,     0,     0,
       0,     0,    69,    70,    71,    72,    73,    74,    75,     0,
      31,   210,   211,    90,    91,     0,    69,    70,    71,    72,
      73,    74,    75,     0,     0,     0,    90,     0,     0,     0,
       0,     0,     0,    31,     0,     0,     0,     0,     0,    69,
      70,    71,    72,    73,    74,    75,    31,    68,     0,     0,
       0,     0,    69,    70,    71,    72,    73,    74,    75,     0,
       0,     0,     0,     0,     0,     0,     0,   399,     0,     0,
       0,     0,     0,    69,    70,    71,    72,    73,    74,    75
};

static const yytype_int16 yycheck[] =
{
       1,     2,    21,    38,    87,    28,    28,   201,    18,   118,
      42,   481,   421,   400,    17,    24,    17,   462,    19,    20,
      30,    36,    35,    44,    45,    46,    40,    41,    17,   474,
      58,   434,   477,   303,    36,   422,   439,    55,   177,   178,
     179,   463,    30,    52,   453,   454,   111,   317,    49,    81,
     107,   108,   109,   175,    82,   231,   465,   114,   234,   446,
       3,     4,   127,    51,   451,   358,    89,    82,   151,    84,
     153,   516,    85,    82,    77,    75,   546,   522,   500,    75,
      82,   374,     0,   118,   493,    85,    82,    97,   491,    99,
     100,   267,   102,   502,   216,   482,    85,    82,    87,    39,
      85,     1,   123,   124,   125,   126,   551,   529,    75,   110,
     249,   250,    80,    77,    82,    82,    74,    85,    76,   589,
      88,    89,   261,   262,   263,   264,   265,    76,   537,   186,
     575,   188,   519,    82,    62,    35,    85,   169,    66,    78,
     223,    80,   242,   243,   244,    74,   147,    76,    87,   258,
      75,   560,    75,   598,    79,    55,    79,    82,   558,    82,
      76,    61,    62,    63,    64,    65,    66,    67,   568,   569,
      55,    79,   572,   367,    80,    35,    82,    85,    85,    85,
     181,    85,    88,    89,   185,    85,    85,   270,   597,   272,
     273,    53,    75,    55,    57,   196,    79,   198,   298,   299,
       1,   202,   203,   204,   205,   206,   207,    74,   240,    76,
      79,   311,   312,   313,   314,   315,   230,   227,   232,   233,
      55,   235,    74,   258,   225,   226,    80,    74,    82,    76,
      74,    85,    35,    36,    88,    89,    55,    56,    57,    40,
      41,    42,    43,    44,    45,    46,    47,    74,    49,     1,
      74,    51,    55,     6,    74,    55,    56,    58,    61,    62,
      63,    64,    65,    66,    67,    74,    75,   302,    74,    75,
     305,   354,    37,    38,   284,    76,    77,    61,    62,    74,
      83,    82,    35,    36,   319,    55,    56,    81,    40,    41,
      42,    43,    44,    45,    46,    47,    75,    49,    55,    56,
      77,   333,    55,    81,    68,    81,    58,    78,    61,    62,
      63,    64,    65,    66,    67,    82,    69,    70,    85,    86,
      74,    88,    89,   358,    76,    77,    56,    57,    74,     1,
      82,    82,    83,    86,    85,    55,    56,    88,    89,   374,
      85,   342,   365,   365,    85,   359,   360,   361,    61,    62,
      63,    64,   353,    11,    12,    13,    14,    15,    16,   442,
      74,    77,    77,    55,   365,   448,    83,   450,    40,    41,
      42,    43,    44,    45,    46,    47,   459,    49,    58,    74,
      81,    81,    86,    86,    78,    75,    79,    86,    86,   472,
      78,    51,    78,    76,    78,   478,    78,    86,   408,    80,
     401,    80,    77,    17,    76,    77,    83,   490,     1,    78,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    14,    15,    16,    86,    18,    19,    86,   429,    22,
     513,    85,   515,    82,    87,    87,    83,   456,     6,     7,
       8,     9,    10,    75,    86,    83,    80,   530,    83,    80,
      18,    19,    78,   488,    22,    83,   539,    77,     1,   542,
      53,    54,    74,    82,   547,    82,    59,    60,    39,    82,
     489,    39,     1,    77,    74,    77,    83,    78,    71,    74,
      85,    57,    74,    76,    77,    83,    83,    86,    74,    83,
      86,   526,   527,    83,   577,    86,    78,    40,    41,    42,
      43,    44,    45,    46,    47,   524,    49,    82,   509,    86,
       1,    40,    41,    42,    43,    44,    45,    46,    47,    35,
      49,    82,    85,    55,    76,    55,    57,    82,    20,    55,
      78,    76,    78,    76,    77,    74,     1,    55,    87,   540,
     541,    77,    86,    85,   545,    74,    74,    76,     1,    40,
      41,    42,    43,    44,    45,    46,    47,   558,    49,     1,
      83,    56,    77,    77,    86,    75,    83,   568,   569,    78,
     571,   572,    74,     1,    86,    40,    41,    42,    43,    44,
      45,    46,    47,   584,    49,    76,    50,    40,    41,    42,
      43,    44,    45,    46,    47,    79,    49,     1,    40,    41,
      42,    43,    44,    45,    46,    47,    75,    49,    48,    77,
       1,    76,    40,    41,    42,    43,    44,    45,    46,    47,
      82,    49,    77,    76,    55,    79,    77,    86,    78,    78,
      77,    83,    80,    77,    76,     5,    40,    41,    42,    43,
      44,    45,    46,    47,    77,    49,    22,    35,    76,    40,
      41,    42,    43,    44,    45,    46,    47,    86,    49,     8,
     238,    91,   228,   252,   247,   128,   112,    55,   333,   257,
     372,   258,    76,    61,    62,    63,    64,    65,    66,    67,
      22,   338,    22,     3,     4,    76,     6,     7,     8,     9,
      10,    11,    12,    13,    14,    15,    16,     1,    18,    19,
     526,   365,    22,   169,   271,   335,   527,   465,   420,   550,
     578,     1,   473,   479,    -1,    -1,    -1,    21,    -1,    23,
      24,    25,    26,    27,    28,    -1,    -1,    31,    32,    33,
      34,    21,    -1,    23,    24,    25,    26,    27,    28,    16,
      -1,    31,    32,    33,    34,     6,    -1,    51,    52,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    76,    -1,    35,    36,
      -1,    51,    52,    -1,    68,    -1,    -1,    -1,    72,    73,
      -1,    -1,    -1,    -1,    35,    36,    -1,    -1,    55,    18,
      -1,    -1,    72,    73,    61,    62,    63,    64,    65,    66,
      67,    -1,    -1,    -1,    55,    -1,    35,    36,    -1,    -1,
      61,    62,    63,    64,    65,    66,    67,    -1,    69,    70,
      35,    36,    -1,    -1,    -1,    -1,    55,    -1,    -1,    -1,
      -1,    -1,    61,    62,    63,    64,    65,    66,    67,    -1,
      55,    56,    57,    35,    36,    -1,    61,    62,    63,    64,
      65,    66,    67,    -1,    -1,    -1,    35,    -1,    -1,    -1,
      -1,    -1,    -1,    55,    -1,    -1,    -1,    -1,    -1,    61,
      62,    63,    64,    65,    66,    67,    55,    35,    -1,    -1,
      -1,    -1,    61,    62,    63,    64,    65,    66,    67,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    55,    -1,    -1,
      -1,    -1,    -1,    61,    62,    63,    64,    65,    66,    67
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     3,     4,    91,    92,    97,    55,    95,    95,     0,
      92,    74,    76,    98,    98,     1,     5,    53,    54,    59,
      60,    71,    93,    99,   100,   162,   198,   199,    39,    95,
      53,    55,    96,    95,    95,    74,    76,   171,     6,     7,
       8,     9,    10,    11,    12,    13,    14,    15,    16,    18,
      19,    22,    76,    97,   117,   118,   134,   137,   138,   139,
     141,   151,   152,   155,   156,   157,    77,    99,    35,    61,
      62,    63,    64,    65,    66,    67,    96,   105,   107,   108,
     109,   163,    76,    96,    74,    75,    58,    82,   169,    16,
      35,    36,   108,   109,   110,   111,   112,   113,    35,   119,
     119,    85,   119,   107,   158,    85,   123,   123,   123,   123,
      85,   127,   140,    85,   120,    95,    57,   159,    79,    99,
      74,    74,    74,    11,    12,    13,    14,    15,    16,   142,
     143,   144,   145,   146,    74,    74,    74,    94,    62,    66,
      61,    62,    63,    64,    79,   104,    81,   107,    99,    74,
      75,    76,    82,    85,   169,    77,   172,   108,   112,    81,
      81,    36,    82,    84,    96,    96,    96,    68,    96,    78,
      74,    30,    51,   124,   129,    95,   106,   106,   106,   106,
      51,    56,    95,   126,   128,    85,   140,    85,   127,    37,
      38,   121,   122,   106,    74,    74,    18,   111,   113,   149,
     150,    77,   123,   123,   123,   123,   140,   120,    61,    62,
      56,    57,   101,   102,   103,   113,    95,    77,    55,   169,
      58,   168,   169,    83,    74,    81,    81,    85,   115,   116,
      82,    79,    82,    86,    79,    82,   158,    86,    78,   104,
      75,   135,   135,   135,   135,    95,    86,    78,    86,   106,
     106,    86,    78,    76,    95,    87,   148,    95,    78,    80,
      94,    95,    95,    95,    95,    95,    95,    78,    80,   104,
      77,    78,    83,    86,   169,    95,    95,    56,    96,   114,
     116,   119,   102,   119,   119,   102,   119,   124,   107,   136,
      74,    76,   153,   153,   153,   153,    86,   128,   135,   135,
     121,    17,   130,   132,   133,    87,   147,    56,    57,   148,
     150,   135,   135,   135,   135,   135,    74,    76,   102,    82,
     180,   169,   168,   169,   169,    83,    86,    83,    80,    83,
      96,    80,    83,    78,    39,   151,   154,   155,   160,   161,
     153,   153,   113,   133,    77,   113,   153,   153,   153,   153,
     153,   133,    83,   113,   170,   173,   178,    74,    82,    82,
      82,    82,   136,     1,    85,   164,   161,    77,   154,    74,
      95,    77,    95,   169,    78,    83,   178,   119,   119,   119,
       1,    21,    23,    24,    25,    26,    27,    28,    31,    32,
      33,    34,    51,    52,    68,    72,    73,   165,   166,    55,
      95,   163,    74,    94,    85,   131,    74,    85,    87,   130,
      86,   178,    83,    83,    83,    83,    57,   125,    86,    86,
      78,   180,    95,    86,    74,    56,    57,    96,   167,    35,
     165,     1,    40,    41,    42,    43,    44,    45,    46,    47,
      49,    76,   171,   183,   189,   180,    95,    82,   193,    85,
     193,    55,   194,   195,    76,    57,   188,    55,   191,   193,
      82,   184,   189,   169,    20,   182,   180,   169,    55,   169,
      85,   180,   196,    78,    76,   189,   185,   189,   171,    78,
      74,   169,    55,    77,   184,   172,    87,   183,    85,   181,
      74,    86,    83,   192,   169,   195,    77,   184,    77,   184,
     169,   191,   192,   180,    56,     6,    69,    70,    86,   113,
     174,   177,   179,   171,   169,   193,    76,   189,    86,   197,
      77,   172,    76,   189,    83,    95,    75,    78,    86,   169,
      74,   169,   184,   180,    50,   186,   184,    48,   190,   171,
      79,   113,   170,   176,   179,   172,   169,    75,    77,    82,
      77,    76,   189,   169,    95,   175,    95,   169,    79,    95,
     192,   169,    55,   187,   190,   184,    77,    80,    82,    85,
      88,    89,    79,    86,   175,    76,   189,    78,    78,    83,
      77,   175,    56,   175,    80,    95,   175,    80,   184,   169,
     187,    83,    86,    86,    95,    80,    77,   192,    76,   189,
     184,    77
};

#define yyerrok		(yyerrstatus = 0)
#define yyclearin	(yychar = YYEMPTY)
#define YYEMPTY		(-2)
#define YYEOF		0

#define YYACCEPT	goto yyacceptlab
#define YYABORT		goto yyabortlab
#define YYERROR		goto yyerrorlab


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
      YYPOPSTACK (1);						\
      goto yybackup;						\
    }								\
  else								\
    {								\
      yyerror (YY_("syntax error: cannot back up")); \
      YYERROR;							\
    }								\
while (YYID (0))


#define YYTERROR	1
#define YYERRCODE	256


/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#define YYRHSLOC(Rhs, K) ((Rhs)[K])
#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)				\
    do									\
      if (YYID (N))                                                    \
	{								\
	  (Current).first_line   = YYRHSLOC (Rhs, 1).first_line;	\
	  (Current).first_column = YYRHSLOC (Rhs, 1).first_column;	\
	  (Current).last_line    = YYRHSLOC (Rhs, N).last_line;		\
	  (Current).last_column  = YYRHSLOC (Rhs, N).last_column;	\
	}								\
      else								\
	{								\
	  (Current).first_line   = (Current).last_line   =		\
	    YYRHSLOC (Rhs, 0).last_line;				\
	  (Current).first_column = (Current).last_column =		\
	    YYRHSLOC (Rhs, 0).last_column;				\
	}								\
    while (YYID (0))
#endif


/* YY_LOCATION_PRINT -- Print the location on the stream.
   This macro was not mandated originally: define only if we know
   we won't break user code: when these are the locations we know.  */

#ifndef YY_LOCATION_PRINT
# if defined YYLTYPE_IS_TRIVIAL && YYLTYPE_IS_TRIVIAL
#  define YY_LOCATION_PRINT(File, Loc)			\
     fprintf (File, "%d.%d-%d.%d",			\
	      (Loc).first_line, (Loc).first_column,	\
	      (Loc).last_line,  (Loc).last_column)
# else
#  define YY_LOCATION_PRINT(File, Loc) ((void) 0)
# endif
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
} while (YYID (0))

# define YY_SYMBOL_PRINT(Title, Type, Value, Location)			  \
do {									  \
  if (yydebug)								  \
    {									  \
      YYFPRINTF (stderr, "%s ", Title);					  \
      yy_symbol_print (stderr,						  \
		  Type, Value); \
      YYFPRINTF (stderr, "\n");						  \
    }									  \
} while (YYID (0))


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_value_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# else
  YYUSE (yyoutput);
# endif
  switch (yytype)
    {
      default:
	break;
    }
}


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (yytype < YYNTOKENS)
    YYFPRINTF (yyoutput, "token %s (", yytname[yytype]);
  else
    YYFPRINTF (yyoutput, "nterm %s (", yytname[yytype]);

  yy_symbol_value_print (yyoutput, yytype, yyvaluep);
  YYFPRINTF (yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_stack_print (yytype_int16 *bottom, yytype_int16 *top)
#else
static void
yy_stack_print (bottom, top)
    yytype_int16 *bottom;
    yytype_int16 *top;
#endif
{
  YYFPRINTF (stderr, "Stack now");
  for (; bottom <= top; ++bottom)
    YYFPRINTF (stderr, " %d", *bottom);
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)				\
do {								\
  if (yydebug)							\
    yy_stack_print ((Bottom), (Top));				\
} while (YYID (0))


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_reduce_print (YYSTYPE *yyvsp, int yyrule)
#else
static void
yy_reduce_print (yyvsp, yyrule)
    YYSTYPE *yyvsp;
    int yyrule;
#endif
{
  int yynrhs = yyr2[yyrule];
  int yyi;
  unsigned long int yylno = yyrline[yyrule];
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
	     yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      fprintf (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr, yyrhs[yyprhs[yyrule] + yyi],
		       &(yyvsp[(yyi + 1) - (yynrhs)])
		       		       );
      fprintf (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)		\
do {					\
  if (yydebug)				\
    yy_reduce_print (yyvsp, Rule); \
} while (YYID (0))

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
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
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif



#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static YYSIZE_T
yystrlen (const char *yystr)
#else
static YYSIZE_T
yystrlen (yystr)
    const char *yystr;
#endif
{
  YYSIZE_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static char *
yystpcpy (char *yydest, const char *yysrc)
#else
static char *
yystpcpy (yydest, yysrc)
    char *yydest;
    const char *yysrc;
#endif
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYSIZE_T yyn = 0;
      char const *yyp = yystr;

      for (;;)
	switch (*++yyp)
	  {
	  case '\'':
	  case ',':
	    goto do_not_strip_quotes;

	  case '\\':
	    if (*++yyp != '\\')
	      goto do_not_strip_quotes;
	    /* Fall through.  */
	  default:
	    if (yyres)
	      yyres[yyn] = *yyp;
	    yyn++;
	    break;

	  case '"':
	    if (yyres)
	      yyres[yyn] = '\0';
	    return yyn;
	  }
    do_not_strip_quotes: ;
    }

  if (! yyres)
    return yystrlen (yystr);

  return yystpcpy (yyres, yystr) - yyres;
}
# endif

/* Copy into YYRESULT an error message about the unexpected token
   YYCHAR while in state YYSTATE.  Return the number of bytes copied,
   including the terminating null byte.  If YYRESULT is null, do not
   copy anything; just return the number of bytes that would be
   copied.  As a special case, return 0 if an ordinary "syntax error"
   message will do.  Return YYSIZE_MAXIMUM if overflow occurs during
   size calculation.  */
static YYSIZE_T
yysyntax_error (char *yyresult, int yystate, int yychar)
{
  int yyn = yypact[yystate];

  if (! (YYPACT_NINF < yyn && yyn <= YYLAST))
    return 0;
  else
    {
      int yytype = YYTRANSLATE (yychar);
      YYSIZE_T yysize0 = yytnamerr (0, yytname[yytype]);
      YYSIZE_T yysize = yysize0;
      YYSIZE_T yysize1;
      int yysize_overflow = 0;
      enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
      char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
      int yyx;

# if 0
      /* This is so xgettext sees the translatable formats that are
	 constructed on the fly.  */
      YY_("syntax error, unexpected %s");
      YY_("syntax error, unexpected %s, expecting %s");
      YY_("syntax error, unexpected %s, expecting %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s");
# endif
      char *yyfmt;
      char const *yyf;
      static char const yyunexpected[] = "syntax error, unexpected %s";
      static char const yyexpecting[] = ", expecting %s";
      static char const yyor[] = " or %s";
      char yyformat[sizeof yyunexpected
		    + sizeof yyexpecting - 1
		    + ((YYERROR_VERBOSE_ARGS_MAXIMUM - 2)
		       * (sizeof yyor - 1))];
      char const *yyprefix = yyexpecting;

      /* Start YYX at -YYN if negative to avoid negative indexes in
	 YYCHECK.  */
      int yyxbegin = yyn < 0 ? -yyn : 0;

      /* Stay within bounds of both yycheck and yytname.  */
      int yychecklim = YYLAST - yyn + 1;
      int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
      int yycount = 1;

      yyarg[0] = yytname[yytype];
      yyfmt = yystpcpy (yyformat, yyunexpected);

      for (yyx = yyxbegin; yyx < yyxend; ++yyx)
	if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR)
	  {
	    if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
	      {
		yycount = 1;
		yysize = yysize0;
		yyformat[sizeof yyunexpected - 1] = '\0';
		break;
	      }
	    yyarg[yycount++] = yytname[yyx];
	    yysize1 = yysize + yytnamerr (0, yytname[yyx]);
	    yysize_overflow |= (yysize1 < yysize);
	    yysize = yysize1;
	    yyfmt = yystpcpy (yyfmt, yyprefix);
	    yyprefix = yyor;
	  }

      yyf = YY_(yyformat);
      yysize1 = yysize + yystrlen (yyf);
      yysize_overflow |= (yysize1 < yysize);
      yysize = yysize1;

      if (yysize_overflow)
	return YYSIZE_MAXIMUM;

      if (yyresult)
	{
	  /* Avoid sprintf, as that infringes on the user's name space.
	     Don't have undefined behavior even if the translation
	     produced a string with the wrong number of "%s"s.  */
	  char *yyp = yyresult;
	  int yyi = 0;
	  while ((*yyp = *yyf) != '\0')
	    {
	      if (*yyp == '%' && yyf[1] == 's' && yyi < yycount)
		{
		  yyp += yytnamerr (yyp, yyarg[yyi++]);
		  yyf += 2;
		}
	      else
		{
		  yyp++;
		  yyf++;
		}
	    }
	}
      return yysize;
    }
}
#endif /* YYERROR_VERBOSE */


/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep)
#else
static void
yydestruct (yymsg, yytype, yyvaluep)
    const char *yymsg;
    int yytype;
    YYSTYPE *yyvaluep;
#endif
{
  YYUSE (yyvaluep);

  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  switch (yytype)
    {

      default:
	break;
    }
}


/* Prevent warnings from -Wmissing-prototypes.  */

#ifdef YYPARSE_PARAM
#if defined __STDC__ || defined __cplusplus
int yyparse (void *YYPARSE_PARAM);
#else
int yyparse ();
#endif
#else /* ! YYPARSE_PARAM */
#if defined __STDC__ || defined __cplusplus
int yyparse (void);
#else
int yyparse ();
#endif
#endif /* ! YYPARSE_PARAM */



/* The look-ahead symbol.  */
int yychar;

/* The semantic value of the look-ahead symbol.  */
YYSTYPE yylval;

/* Number of syntax errors so far.  */
int yynerrs;



/*----------.
| yyparse.  |
`----------*/

#ifdef YYPARSE_PARAM
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void *YYPARSE_PARAM)
#else
int
yyparse (YYPARSE_PARAM)
    void *YYPARSE_PARAM;
#endif
#else /* ! YYPARSE_PARAM */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void)
#else
int
yyparse ()

#endif
#endif
{
  
  int yystate;
  int yyn;
  int yyresult;
  /* Number of tokens to shift before error messages enabled.  */
  int yyerrstatus;
  /* Look-ahead token as an internal (translated) token number.  */
  int yytoken = 0;
#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

  /* Three stacks and their tools:
     `yyss': related to states,
     `yyvs': related to semantic values,
     `yyls': related to locations.

     Refer to the stacks thru separate pointers, to allow yyoverflow
     to reallocate them elsewhere.  */

  /* The state stack.  */
  yytype_int16 yyssa[YYINITDEPTH];
  yytype_int16 *yyss = yyssa;
  yytype_int16 *yyssp;

  /* The semantic value stack.  */
  YYSTYPE yyvsa[YYINITDEPTH];
  YYSTYPE *yyvs = yyvsa;
  YYSTYPE *yyvsp;



#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  YYSIZE_T yystacksize = YYINITDEPTH;

  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;


  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

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
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
	/* Give user a chance to reallocate the stack.  Use copies of
	   these so that the &'s don't force the real ones into
	   memory.  */
	YYSTYPE *yyvs1 = yyvs;
	yytype_int16 *yyss1 = yyss;


	/* Each stack pointer address is followed by the size of the
	   data in use in that stack, in bytes.  This used to be a
	   conditional around just the two extra args, but that might
	   be undefined if yyoverflow is a macro.  */
	yyoverflow (YY_("memory exhausted"),
		    &yyss1, yysize * sizeof (*yyssp),
		    &yyvs1, yysize * sizeof (*yyvsp),

		    &yystacksize);

	yyss = yyss1;
	yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
	goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
	yystacksize = YYMAXDEPTH;

      {
	yytype_int16 *yyss1 = yyss;
	union yyalloc *yyptr =
	  (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
	if (! yyptr)
	  goto yyexhaustedlab;
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

  /* Do appropriate processing given the current state.  Read a
     look-ahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to look-ahead token.  */
  yyn = yypact[yystate];
  if (yyn == YYPACT_NINF)
    goto yydefault;

  /* Not known => get a look-ahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid look-ahead symbol.  */
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
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
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

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the look-ahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token unless it is eof.  */
  if (yychar != YYEOF)
    yychar = YYEMPTY;

  yystate = yyn;
  *++yyvsp = yylval;

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
#line 159 "xi-grammar.y"
    { (yyval.modlist) = (yyvsp[(1) - (1)].modlist); modlist = (yyvsp[(1) - (1)].modlist); }
    break;

  case 3:
#line 163 "xi-grammar.y"
    { 
		  (yyval.modlist) = 0; 
		}
    break;

  case 4:
#line 167 "xi-grammar.y"
    { (yyval.modlist) = new ModuleList(lineno, (yyvsp[(1) - (2)].module), (yyvsp[(2) - (2)].modlist)); }
    break;

  case 5:
#line 171 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 6:
#line 173 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 7:
#line 177 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 8:
#line 179 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 9:
#line 183 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 10:
#line 187 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 11:
#line 189 "xi-grammar.y"
    {
		  char *tmp = new char[strlen((yyvsp[(1) - (4)].strval))+strlen((yyvsp[(4) - (4)].strval))+3];
		  sprintf(tmp,"%s::%s", (yyvsp[(1) - (4)].strval), (yyvsp[(4) - (4)].strval));
		  (yyval.strval) = tmp;
		}
    break;

  case 12:
#line 197 "xi-grammar.y"
    { 
		    (yyval.module) = new Module(lineno, (yyvsp[(2) - (3)].strval), (yyvsp[(3) - (3)].conslist)); 
		}
    break;

  case 13:
#line 201 "xi-grammar.y"
    {  
		    (yyval.module) = new Module(lineno, (yyvsp[(2) - (3)].strval), (yyvsp[(3) - (3)].conslist)); 
		    (yyval.module)->setMain();
		}
    break;

  case 14:
#line 208 "xi-grammar.y"
    { (yyval.conslist) = 0; }
    break;

  case 15:
#line 210 "xi-grammar.y"
    { (yyval.conslist) = (yyvsp[(2) - (4)].conslist); }
    break;

  case 16:
#line 214 "xi-grammar.y"
    { (yyval.conslist) = 0; }
    break;

  case 17:
#line 216 "xi-grammar.y"
    { (yyval.conslist) = new ConstructList(lineno, (yyvsp[(1) - (2)].construct), (yyvsp[(2) - (2)].conslist)); }
    break;

  case 18:
#line 220 "xi-grammar.y"
    { if((yyvsp[(3) - (5)].conslist)) (yyvsp[(3) - (5)].conslist)->setExtern((yyvsp[(1) - (5)].intval)); (yyval.construct) = (yyvsp[(3) - (5)].conslist); }
    break;

  case 19:
#line 222 "xi-grammar.y"
    { (yyval.construct) = new Scope((yyvsp[(2) - (5)].strval), (yyvsp[(4) - (5)].conslist)); }
    break;

  case 20:
#line 224 "xi-grammar.y"
    { (yyval.construct) = new UsingScope((yyvsp[(3) - (4)].strval), false); }
    break;

  case 21:
#line 226 "xi-grammar.y"
    { (yyval.construct) = new UsingScope((yyvsp[(2) - (3)].strval), true); }
    break;

  case 22:
#line 228 "xi-grammar.y"
    { (yyvsp[(2) - (2)].module)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].module); }
    break;

  case 23:
#line 230 "xi-grammar.y"
    { (yyvsp[(2) - (2)].member)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].member); }
    break;

  case 24:
#line 232 "xi-grammar.y"
    { (yyvsp[(2) - (3)].message)->setExtern((yyvsp[(1) - (3)].intval)); (yyval.construct) = (yyvsp[(2) - (3)].message); }
    break;

  case 25:
#line 234 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 26:
#line 236 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 27:
#line 238 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 28:
#line 240 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 29:
#line 242 "xi-grammar.y"
    { (yyvsp[(2) - (2)].templat)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].templat); }
    break;

  case 30:
#line 244 "xi-grammar.y"
    {
          Entry *e = new Entry(lineno, 0, (yyvsp[(3) - (8)].type), (yyvsp[(5) - (8)].strval), (yyvsp[(7) - (8)].plist), 0, 0, 0, 0, 0);
          int isExtern = 1;
          e->setExtern(isExtern);
          e->targs = (yyvsp[(6) - (8)].tparlist);
          e->label = new XStr;
          (yyvsp[(4) - (8)].ntype)->print(*e->label);
          (yyval.construct) = e;
        }
    break;

  case 31:
#line 254 "xi-grammar.y"
    { (yyval.construct) = NULL; }
    break;

  case 32:
#line 256 "xi-grammar.y"
    { (yyval.construct) = NULL; }
    break;

  case 33:
#line 258 "xi-grammar.y"
    { (yyval.construct) = (yyvsp[(1) - (1)].accelBlock); }
    break;

  case 34:
#line 260 "xi-grammar.y"
    { printf("Invalid construct\n"); YYABORT; }
    break;

  case 35:
#line 264 "xi-grammar.y"
    { (yyval.tparam) = new TParamType((yyvsp[(1) - (1)].type)); }
    break;

  case 36:
#line 266 "xi-grammar.y"
    { (yyval.tparam) = new TParamVal((yyvsp[(1) - (1)].strval)); }
    break;

  case 37:
#line 268 "xi-grammar.y"
    { (yyval.tparam) = new TParamVal((yyvsp[(1) - (1)].strval)); }
    break;

  case 38:
#line 272 "xi-grammar.y"
    { (yyval.tparlist) = new TParamList((yyvsp[(1) - (1)].tparam)); }
    break;

  case 39:
#line 274 "xi-grammar.y"
    { (yyval.tparlist) = new TParamList((yyvsp[(1) - (3)].tparam), (yyvsp[(3) - (3)].tparlist)); }
    break;

  case 40:
#line 278 "xi-grammar.y"
    { (yyval.tparlist) = 0; }
    break;

  case 41:
#line 280 "xi-grammar.y"
    { (yyval.tparlist) = (yyvsp[(1) - (1)].tparlist); }
    break;

  case 42:
#line 284 "xi-grammar.y"
    { (yyval.tparlist) = 0; }
    break;

  case 43:
#line 286 "xi-grammar.y"
    { (yyval.tparlist) = (yyvsp[(2) - (3)].tparlist); }
    break;

  case 44:
#line 290 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("int"); }
    break;

  case 45:
#line 292 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long"); }
    break;

  case 46:
#line 294 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("short"); }
    break;

  case 47:
#line 296 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("char"); }
    break;

  case 48:
#line 298 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned int"); }
    break;

  case 49:
#line 300 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long"); }
    break;

  case 50:
#line 302 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long"); }
    break;

  case 51:
#line 304 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long long"); }
    break;

  case 52:
#line 306 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned short"); }
    break;

  case 53:
#line 308 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned char"); }
    break;

  case 54:
#line 310 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long long"); }
    break;

  case 55:
#line 312 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("float"); }
    break;

  case 56:
#line 314 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("double"); }
    break;

  case 57:
#line 316 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long double"); }
    break;

  case 58:
#line 318 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("void"); }
    break;

  case 59:
#line 321 "xi-grammar.y"
    { (yyval.ntype) = new NamedType((yyvsp[(1) - (2)].strval),(yyvsp[(2) - (2)].tparlist)); }
    break;

  case 60:
#line 322 "xi-grammar.y"
    { 
                    const char* basename, *scope;
                    splitScopedName((yyvsp[(1) - (2)].strval), &scope, &basename);
                    (yyval.ntype) = new NamedType(basename, (yyvsp[(2) - (2)].tparlist), scope);
                }
    break;

  case 61:
#line 330 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 62:
#line 332 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ntype); }
    break;

  case 63:
#line 336 "xi-grammar.y"
    { (yyval.ptype) = new PtrType((yyvsp[(1) - (2)].type)); }
    break;

  case 64:
#line 340 "xi-grammar.y"
    { (yyvsp[(1) - (2)].ptype)->indirect(); (yyval.ptype) = (yyvsp[(1) - (2)].ptype); }
    break;

  case 65:
#line 342 "xi-grammar.y"
    { (yyvsp[(1) - (2)].ptype)->indirect(); (yyval.ptype) = (yyvsp[(1) - (2)].ptype); }
    break;

  case 66:
#line 346 "xi-grammar.y"
    { (yyval.ftype) = new FuncType((yyvsp[(1) - (8)].type), (yyvsp[(4) - (8)].strval), (yyvsp[(7) - (8)].plist)); }
    break;

  case 67:
#line 350 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 68:
#line 352 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 69:
#line 354 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 70:
#line 356 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ftype); }
    break;

  case 71:
#line 359 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(2) - (2)].type); }
    break;

  case 72:
#line 361 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (2)].type); }
    break;

  case 73:
#line 365 "xi-grammar.y"
    { (yyval.type) = new ReferenceType((yyvsp[(1) - (2)].type)); }
    break;

  case 74:
#line 367 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 75:
#line 371 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 76:
#line 373 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 77:
#line 377 "xi-grammar.y"
    { (yyval.val) = (yyvsp[(2) - (3)].val); }
    break;

  case 78:
#line 381 "xi-grammar.y"
    { (yyval.vallist) = 0; }
    break;

  case 79:
#line 383 "xi-grammar.y"
    { (yyval.vallist) = new ValueList((yyvsp[(1) - (2)].val), (yyvsp[(2) - (2)].vallist)); }
    break;

  case 80:
#line 387 "xi-grammar.y"
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[(2) - (4)].type), (yyvsp[(3) - (4)].strval), (yyvsp[(4) - (4)].vallist)); }
    break;

  case 81:
#line 391 "xi-grammar.y"
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[(3) - (5)].type), (yyvsp[(5) - (5)].strval), 0, 1); }
    break;

  case 82:
#line 395 "xi-grammar.y"
    { (yyval.intval) = 0;}
    break;

  case 83:
#line 397 "xi-grammar.y"
    { (yyval.intval) = 0;}
    break;

  case 84:
#line 401 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 85:
#line 403 "xi-grammar.y"
    { 
		  /*
		  printf("Warning: Message attributes are being phased out.\n");
		  printf("Warning: Please remove them from interface files.\n");
		  */
		  (yyval.intval) = (yyvsp[(2) - (3)].intval); 
		}
    break;

  case 86:
#line 413 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (1)].intval); }
    break;

  case 87:
#line 415 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (3)].intval) | (yyvsp[(3) - (3)].intval); }
    break;

  case 88:
#line 419 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 89:
#line 421 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 90:
#line 425 "xi-grammar.y"
    { (yyval.cattr) = 0; }
    break;

  case 91:
#line 427 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(2) - (3)].cattr); }
    break;

  case 92:
#line 431 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (1)].cattr); }
    break;

  case 93:
#line 433 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (3)].cattr) | (yyvsp[(3) - (3)].cattr); }
    break;

  case 94:
#line 437 "xi-grammar.y"
    { python_doc = NULL; (yyval.intval) = 0; }
    break;

  case 95:
#line 439 "xi-grammar.y"
    { python_doc = (yyvsp[(1) - (1)].strval); (yyval.intval) = 0; }
    break;

  case 96:
#line 443 "xi-grammar.y"
    { (yyval.cattr) = Chare::CPYTHON; }
    break;

  case 97:
#line 447 "xi-grammar.y"
    { (yyval.cattr) = 0; }
    break;

  case 98:
#line 449 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(2) - (3)].cattr); }
    break;

  case 99:
#line 453 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (1)].cattr); }
    break;

  case 100:
#line 455 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (3)].cattr) | (yyvsp[(3) - (3)].cattr); }
    break;

  case 101:
#line 459 "xi-grammar.y"
    { (yyval.cattr) = Chare::CMIGRATABLE; }
    break;

  case 102:
#line 461 "xi-grammar.y"
    { (yyval.cattr) = Chare::CPYTHON; }
    break;

  case 103:
#line 465 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 104:
#line 467 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 105:
#line 470 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 106:
#line 472 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 107:
#line 475 "xi-grammar.y"
    { (yyval.mv) = new MsgVar((yyvsp[(2) - (5)].type), (yyvsp[(3) - (5)].strval), (yyvsp[(1) - (5)].intval), (yyvsp[(4) - (5)].intval)); }
    break;

  case 108:
#line 479 "xi-grammar.y"
    { (yyval.mvlist) = new MsgVarList((yyvsp[(1) - (1)].mv)); }
    break;

  case 109:
#line 481 "xi-grammar.y"
    { (yyval.mvlist) = new MsgVarList((yyvsp[(1) - (2)].mv), (yyvsp[(2) - (2)].mvlist)); }
    break;

  case 110:
#line 485 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, (yyvsp[(3) - (3)].ntype)); }
    break;

  case 111:
#line 487 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, (yyvsp[(3) - (6)].ntype), (yyvsp[(5) - (6)].mvlist)); }
    break;

  case 112:
#line 491 "xi-grammar.y"
    { (yyval.typelist) = 0; }
    break;

  case 113:
#line 493 "xi-grammar.y"
    { (yyval.typelist) = (yyvsp[(2) - (2)].typelist); }
    break;

  case 114:
#line 497 "xi-grammar.y"
    { (yyval.typelist) = new TypeList((yyvsp[(1) - (1)].ntype)); }
    break;

  case 115:
#line 499 "xi-grammar.y"
    { (yyval.typelist) = new TypeList((yyvsp[(1) - (3)].ntype), (yyvsp[(3) - (3)].typelist)); }
    break;

  case 116:
#line 503 "xi-grammar.y"
    { (yyval.chare) = new Chare(lineno, (yyvsp[(2) - (5)].cattr)|Chare::CCHARE, (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist));
          hasSeenConstructor = false;
        }
    break;

  case 117:
#line 507 "xi-grammar.y"
    { (yyval.chare) = new MainChare(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist));
          hasSeenConstructor = false;
        }
    break;

  case 118:
#line 513 "xi-grammar.y"
    { (yyval.chare) = new Group(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist));
          hasSeenConstructor = false;
        }
    break;

  case 119:
#line 519 "xi-grammar.y"
    { (yyval.chare) = new NodeGroup(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist));
          hasSeenConstructor = false;
        }
    break;

  case 120:
#line 525 "xi-grammar.y"
    {/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",(yyvsp[(2) - (4)].strval));
			(yyval.ntype) = new NamedType(buf); 
		}
    break;

  case 121:
#line 531 "xi-grammar.y"
    { (yyval.ntype) = new NamedType((yyvsp[(2) - (3)].strval)); }
    break;

  case 122:
#line 535 "xi-grammar.y"
    {  (yyval.chare) = new Array(lineno, (yyvsp[(2) - (6)].cattr), (yyvsp[(3) - (6)].ntype), (yyvsp[(4) - (6)].ntype), (yyvsp[(5) - (6)].typelist), (yyvsp[(6) - (6)].mbrlist));
          hasSeenConstructor = false;
        }
    break;

  case 123:
#line 539 "xi-grammar.y"
    {  (yyval.chare) = new Array(lineno, (yyvsp[(3) - (6)].cattr), (yyvsp[(2) - (6)].ntype), (yyvsp[(4) - (6)].ntype), (yyvsp[(5) - (6)].typelist), (yyvsp[(6) - (6)].mbrlist));
          hasSeenConstructor = false;
        }
    break;

  case 124:
#line 545 "xi-grammar.y"
    { (yyval.chare) = new Chare(lineno, (yyvsp[(2) - (5)].cattr)|Chare::CCHARE, new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist));
          hasSeenConstructor = false;
        }
    break;

  case 125:
#line 549 "xi-grammar.y"
    { (yyval.chare) = new MainChare(lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist));
          hasSeenConstructor = false;
        }
    break;

  case 126:
#line 555 "xi-grammar.y"
    { (yyval.chare) = new Group(lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist));
          hasSeenConstructor = false;
        }
    break;

  case 127:
#line 561 "xi-grammar.y"
    { (yyval.chare) = new NodeGroup( lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist));
          hasSeenConstructor = false;
        }
    break;

  case 128:
#line 567 "xi-grammar.y"
    { (yyval.chare) = new Array( lineno, 0, (yyvsp[(2) - (5)].ntype), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist));
          hasSeenConstructor = false;
        }
    break;

  case 129:
#line 573 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[(3) - (4)].strval))); }
    break;

  case 130:
#line 575 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[(3) - (7)].strval)), (yyvsp[(5) - (7)].mvlist)); }
    break;

  case 131:
#line 579 "xi-grammar.y"
    { (yyval.type) = 0; }
    break;

  case 132:
#line 581 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(2) - (2)].type); }
    break;

  case 133:
#line 585 "xi-grammar.y"
    { (yyval.strval) = 0; }
    break;

  case 134:
#line 587 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(2) - (2)].strval); }
    break;

  case 135:
#line 589 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(2) - (2)].strval); }
    break;

  case 136:
#line 593 "xi-grammar.y"
    { (yyval.tvar) = new TType(new NamedType((yyvsp[(2) - (3)].strval)), (yyvsp[(3) - (3)].type)); }
    break;

  case 137:
#line 595 "xi-grammar.y"
    { (yyval.tvar) = new TFunc((yyvsp[(1) - (2)].ftype), (yyvsp[(2) - (2)].strval)); }
    break;

  case 138:
#line 597 "xi-grammar.y"
    { (yyval.tvar) = new TName((yyvsp[(1) - (3)].type), (yyvsp[(2) - (3)].strval), (yyvsp[(3) - (3)].strval)); }
    break;

  case 139:
#line 601 "xi-grammar.y"
    { (yyval.tvarlist) = new TVarList((yyvsp[(1) - (1)].tvar)); }
    break;

  case 140:
#line 603 "xi-grammar.y"
    { (yyval.tvarlist) = new TVarList((yyvsp[(1) - (3)].tvar), (yyvsp[(3) - (3)].tvarlist)); }
    break;

  case 141:
#line 607 "xi-grammar.y"
    { (yyval.tvarlist) = (yyvsp[(3) - (4)].tvarlist); }
    break;

  case 142:
#line 611 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 143:
#line 613 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 144:
#line 615 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 145:
#line 617 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 146:
#line 619 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].message)); (yyvsp[(2) - (2)].message)->setTemplate((yyval.templat)); }
    break;

  case 147:
#line 623 "xi-grammar.y"
    { (yyval.mbrlist) = 0; }
    break;

  case 148:
#line 625 "xi-grammar.y"
    { (yyval.mbrlist) = (yyvsp[(2) - (4)].mbrlist); }
    break;

  case 149:
#line 629 "xi-grammar.y"
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
                    (yyval.mbrlist) = ml; 
		  }
		  else {
		    (yyval.mbrlist) = 0; 
                  }
		}
    break;

  case 150:
#line 648 "xi-grammar.y"
    { (yyval.mbrlist) = new MemberList((yyvsp[(1) - (2)].member), (yyvsp[(2) - (2)].mbrlist)); }
    break;

  case 151:
#line 652 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (2)].readonly); }
    break;

  case 152:
#line 654 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (2)].readonly); }
    break;

  case 154:
#line 657 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (2)].member); }
    break;

  case 155:
#line 659 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(2) - (3)].pupable); }
    break;

  case 156:
#line 661 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(2) - (3)].includeFile); }
    break;

  case 157:
#line 663 "xi-grammar.y"
    { (yyval.member) = new ClassDeclaration(lineno,(yyvsp[(2) - (3)].strval)); }
    break;

  case 158:
#line 667 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 1); }
    break;

  case 159:
#line 669 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 1); }
    break;

  case 160:
#line 671 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[(3) - (9)].strval)) + '<' +
					    ((yyvsp[(5) - (9)].tparlist))->to_string() + '>').c_str()),
				    1);
		}
    break;

  case 161:
#line 677 "xi-grammar.y"
    { printf("Warning: deprecated use of initcall. Use initnode or initproc instead.\n"); 
		  (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 1); }
    break;

  case 162:
#line 680 "xi-grammar.y"
    { printf("Warning: deprecated use of initcall. Use initnode or initproc instead.\n");
		  (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 1); }
    break;

  case 163:
#line 685 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 0); }
    break;

  case 164:
#line 687 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 0); }
    break;

  case 165:
#line 689 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[(3) - (9)].strval)) + '<' +
					    ((yyvsp[(5) - (9)].tparlist))->to_string() + '>').c_str()),
				    0);
		}
    break;

  case 166:
#line 695 "xi-grammar.y"
    {
                  InitCall* rtn = new InitCall(lineno, (yyvsp[(6) - (9)].strval), 0);
                  rtn->setAccel();
                  (yyval.member) = rtn;
		}
    break;

  case 167:
#line 703 "xi-grammar.y"
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[(1) - (1)].ntype),0); }
    break;

  case 168:
#line 705 "xi-grammar.y"
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[(1) - (3)].ntype),(yyvsp[(3) - (3)].pupable)); }
    break;

  case 169:
#line 708 "xi-grammar.y"
    { (yyval.includeFile) = new IncludeFile(lineno,(yyvsp[(1) - (1)].strval)); }
    break;

  case 170:
#line 712 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (2)].entry); }
    break;

  case 171:
#line 714 "xi-grammar.y"
    {
                  (yyvsp[(2) - (3)].entry)->tspec = (yyvsp[(1) - (3)].tvarlist);
                  (yyval.member) = (yyvsp[(2) - (3)].entry);
                }
    break;

  case 172:
#line 719 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].member); }
    break;

  case 173:
#line 723 "xi-grammar.y"
    { 
		  if ((yyvsp[(7) - (7)].sc) != 0) { 
		    (yyvsp[(7) - (7)].sc)->con1 = new SdagConstruct(SIDENT, (yyvsp[(4) - (7)].strval));
                    (yyvsp[(7) - (7)].sc)->param = new ParamList((yyvsp[(5) - (7)].plist));
                  }
		  (yyval.entry) = new Entry(lineno, (yyvsp[(2) - (7)].intval), (yyvsp[(3) - (7)].type), (yyvsp[(4) - (7)].strval), (yyvsp[(5) - (7)].plist), (yyvsp[(6) - (7)].val), (yyvsp[(7) - (7)].sc), 0, 0); 
		}
    break;

  case 174:
#line 731 "xi-grammar.y"
    { 
          if (hasSeenConstructor && strcmp(lastConstructor, (yyvsp[(3) - (5)].strval)) != 0) {
            yyerror("Entry method has no return type and we've seen a constructor already\n");
          } else {
            // If we see another method without a return type, it should better be an overloaded constructor!
            if (!lastConstructor) free(lastConstructor);
            lastConstructor = (char *) malloc(strlen((yyvsp[(3) - (5)].strval)) + 1);
            strcpy(lastConstructor, (yyvsp[(3) - (5)].strval));
            hasSeenConstructor = true;

            if ((yyvsp[(5) - (5)].sc) != 0) {
              (yyvsp[(5) - (5)].sc)->con1 = new SdagConstruct(SIDENT, (yyvsp[(3) - (5)].strval));
              (yyvsp[(5) - (5)].sc)->param = new ParamList((yyvsp[(4) - (5)].plist));
            }
            Entry *e = new Entry(lineno, (yyvsp[(2) - (5)].intval), 0, (yyvsp[(3) - (5)].strval), (yyvsp[(4) - (5)].plist), 0, (yyvsp[(5) - (5)].sc), 0, 0);
            if (e->param && e->param->isCkMigMsgPtr()) {
              yyerror("Charm++ generates a CkMigrateMsg chare constructor implicitly, but continuing anyway");
              (yyval.entry) = NULL;
            } else {
              (yyval.entry) = e;
            }
          }
		}
    break;

  case 175:
#line 755 "xi-grammar.y"
    {
          int attribs = SACCEL;
          const char* name = (yyvsp[(6) - (12)].strval);
          ParamList* paramList = (yyvsp[(7) - (12)].plist);
          ParamList* accelParamList = (yyvsp[(8) - (12)].plist);
          XStr* codeBody = new XStr((yyvsp[(10) - (12)].strval));
          const char* callbackName = (yyvsp[(12) - (12)].strval);

          (yyval.entry) = new Entry(lineno, attribs, new BuiltinType("void"), name, paramList,
                         0, 0, 0, 0, 0
                        );
          (yyval.entry)->setAccelParam(accelParamList);
          (yyval.entry)->setAccelCodeBody(codeBody);
          (yyval.entry)->setAccelCallbackName(new XStr(callbackName));
        }
    break;

  case 176:
#line 773 "xi-grammar.y"
    { (yyval.accelBlock) = new AccelBlock(lineno, new XStr((yyvsp[(3) - (5)].strval))); }
    break;

  case 177:
#line 775 "xi-grammar.y"
    { (yyval.accelBlock) = new AccelBlock(lineno, NULL); }
    break;

  case 178:
#line 779 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("void"); }
    break;

  case 179:
#line 781 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 180:
#line 785 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 181:
#line 787 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(2) - (3)].intval); }
    break;

  case 182:
#line 789 "xi-grammar.y"
    { printf("Invalid entry method attribute list\n"); YYABORT; }
    break;

  case 183:
#line 793 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (1)].intval); }
    break;

  case 184:
#line 795 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (3)].intval) | (yyvsp[(3) - (3)].intval); }
    break;

  case 185:
#line 799 "xi-grammar.y"
    { (yyval.intval) = STHREADED; }
    break;

  case 186:
#line 801 "xi-grammar.y"
    { (yyval.intval) = SSYNC; }
    break;

  case 187:
#line 803 "xi-grammar.y"
    { (yyval.intval) = SIGET; }
    break;

  case 188:
#line 805 "xi-grammar.y"
    { (yyval.intval) = SLOCKED; }
    break;

  case 189:
#line 807 "xi-grammar.y"
    { (yyval.intval) = SCREATEHERE; }
    break;

  case 190:
#line 809 "xi-grammar.y"
    { (yyval.intval) = SCREATEHOME; }
    break;

  case 191:
#line 811 "xi-grammar.y"
    { (yyval.intval) = SNOKEEP; }
    break;

  case 192:
#line 813 "xi-grammar.y"
    { (yyval.intval) = SNOTRACE; }
    break;

  case 193:
#line 815 "xi-grammar.y"
    { (yyval.intval) = SIMMEDIATE; }
    break;

  case 194:
#line 817 "xi-grammar.y"
    { (yyval.intval) = SSKIPSCHED; }
    break;

  case 195:
#line 819 "xi-grammar.y"
    { (yyval.intval) = SINLINE; }
    break;

  case 196:
#line 821 "xi-grammar.y"
    { (yyval.intval) = SLOCAL; }
    break;

  case 197:
#line 823 "xi-grammar.y"
    { (yyval.intval) = SPYTHON; }
    break;

  case 198:
#line 825 "xi-grammar.y"
    { (yyval.intval) = SMEM; }
    break;

  case 199:
#line 827 "xi-grammar.y"
    { (yyval.intval) = SREDUCE; }
    break;

  case 200:
#line 829 "xi-grammar.y"
    { printf("Invalid entry method attribute: %s\n", yylval); YYABORT; }
    break;

  case 201:
#line 833 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 202:
#line 835 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 203:
#line 837 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 204:
#line 841 "xi-grammar.y"
    { (yyval.strval) = ""; }
    break;

  case 205:
#line 843 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 206:
#line 845 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[(1) - (3)].strval))+strlen((yyvsp[(3) - (3)].strval))+3];
			sprintf(tmp,"%s, %s", (yyvsp[(1) - (3)].strval), (yyvsp[(3) - (3)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 207:
#line 853 "xi-grammar.y"
    { (yyval.strval) = ""; }
    break;

  case 208:
#line 855 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 209:
#line 857 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s[%s]%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 210:
#line 863 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s{%s}%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 211:
#line 869 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s(%s)%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 212:
#line 875 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(2) - (4)].strval))+strlen((yyvsp[(4) - (4)].strval))+3];
			sprintf(tmp,"(%s)%s", (yyvsp[(2) - (4)].strval), (yyvsp[(4) - (4)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 213:
#line 883 "xi-grammar.y"
    {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			(yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].type),(yyvsp[(2) - (3)].strval));
		}
    break;

  case 214:
#line 890 "xi-grammar.y"
    { 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			(yyval.intval) = 0;
		}
    break;

  case 215:
#line 898 "xi-grammar.y"
    { 
			in_braces=0;
			(yyval.intval) = 0;
		}
    break;

  case 216:
#line 905 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (1)].type));}
    break;

  case 217:
#line 907 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].type),(yyvsp[(2) - (3)].strval)); (yyval.pname)->setConditional((yyvsp[(3) - (3)].intval)); }
    break;

  case 218:
#line 909 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (4)].type),(yyvsp[(2) - (4)].strval),0,(yyvsp[(4) - (4)].val));}
    break;

  case 219:
#line 911 "xi-grammar.y"
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].pname)->getType(), (yyvsp[(1) - (3)].pname)->getName() ,(yyvsp[(2) - (3)].strval));
		}
    break;

  case 220:
#line 917 "xi-grammar.y"
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READONLY; }
    break;

  case 221:
#line 918 "xi-grammar.y"
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READWRITE; }
    break;

  case 222:
#line 919 "xi-grammar.y"
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_WRITEONLY; }
    break;

  case 223:
#line 922 "xi-grammar.y"
    { (yyval.xstrptr) = new XStr((yyvsp[(1) - (1)].strval)); }
    break;

  case 224:
#line 923 "xi-grammar.y"
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "->" << (yyvsp[(4) - (4)].strval); }
    break;

  case 225:
#line 924 "xi-grammar.y"
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[(1) - (3)].xstrptr)) << "." << (yyvsp[(3) - (3)].strval); }
    break;

  case 226:
#line 926 "xi-grammar.y"
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "[" << *((yyvsp[(3) - (4)].xstrptr)) << "]";
                  delete (yyvsp[(1) - (4)].xstrptr);
                  delete (yyvsp[(3) - (4)].xstrptr);
                }
    break;

  case 227:
#line 933 "xi-grammar.y"
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "[" << (yyvsp[(3) - (4)].strval) << "]";
                  delete (yyvsp[(1) - (4)].xstrptr);
                }
    break;

  case 228:
#line 939 "xi-grammar.y"
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "(" << *((yyvsp[(3) - (4)].xstrptr)) << ")";
                  delete (yyvsp[(1) - (4)].xstrptr);
                  delete (yyvsp[(3) - (4)].xstrptr);
                }
    break;

  case 229:
#line 948 "xi-grammar.y"
    {
                  in_bracket = 0;
                  (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].pname)->getType(), (yyvsp[(1) - (3)].pname)->getName(), (yyvsp[(2) - (3)].strval));
                }
    break;

  case 230:
#line 955 "xi-grammar.y"
    {
                  (yyval.pname) = new Parameter(lineno, (yyvsp[(3) - (7)].type), (yyvsp[(4) - (7)].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[(6) - (7)].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[(1) - (7)].intval));
                }
    break;

  case 231:
#line 961 "xi-grammar.y"
    {
		  (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (5)].type), (yyvsp[(2) - (5)].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[(4) - (5)].xstrptr));
                  (yyval.pname)->setAccelBufferType(Parameter::ACCEL_BUFFER_TYPE_READWRITE);
		}
    break;

  case 232:
#line 967 "xi-grammar.y"
    {
                  (yyval.pname) = (yyvsp[(3) - (6)].pname);
                  (yyval.pname)->setAccelInstName((yyvsp[(5) - (6)].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[(1) - (6)].intval));
		}
    break;

  case 233:
#line 975 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (1)].pname)); }
    break;

  case 234:
#line 977 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (3)].pname),(yyvsp[(3) - (3)].plist)); }
    break;

  case 235:
#line 981 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (1)].pname)); }
    break;

  case 236:
#line 983 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (3)].pname),(yyvsp[(3) - (3)].plist)); }
    break;

  case 237:
#line 987 "xi-grammar.y"
    { (yyval.plist) = (yyvsp[(2) - (3)].plist); }
    break;

  case 238:
#line 989 "xi-grammar.y"
    { (yyval.plist) = new ParamList(new Parameter(0, new BuiltinType("void"))); }
    break;

  case 239:
#line 993 "xi-grammar.y"
    { (yyval.plist) = (yyvsp[(2) - (3)].plist); }
    break;

  case 240:
#line 995 "xi-grammar.y"
    { (yyval.plist) = 0; }
    break;

  case 241:
#line 999 "xi-grammar.y"
    { (yyval.val) = 0; }
    break;

  case 242:
#line 1001 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(3) - (3)].strval)); }
    break;

  case 243:
#line 1005 "xi-grammar.y"
    { (yyval.sc) = 0; }
    break;

  case 244:
#line 1007 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSDAGENTRY, (yyvsp[(1) - (1)].sc)); }
    break;

  case 245:
#line 1009 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSDAGENTRY, (yyvsp[(2) - (3)].sc)); }
    break;

  case 246:
#line 1013 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSLIST, (yyvsp[(1) - (1)].sc)); }
    break;

  case 247:
#line 1015 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSLIST, (yyvsp[(1) - (2)].sc), (yyvsp[(2) - (2)].sc));  }
    break;

  case 248:
#line 1019 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SOLIST, (yyvsp[(1) - (1)].sc)); }
    break;

  case 249:
#line 1021 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SOLIST, (yyvsp[(1) - (2)].sc), (yyvsp[(2) - (2)].sc)); }
    break;

  case 250:
#line 1025 "xi-grammar.y"
    { (yyval.sc) = 0; }
    break;

  case 251:
#line 1027 "xi-grammar.y"
    { (yyval.sc) = (yyvsp[(3) - (4)].sc); }
    break;

  case 252:
#line 1031 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SPUBLISHES, new SdagConstruct(SIDENT, (yyvsp[(1) - (1)].strval))); }
    break;

  case 253:
#line 1033 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SPUBLISHES, new SdagConstruct(SIDENT, (yyvsp[(1) - (3)].strval)), (yyvsp[(3) - (3)].sc));  }
    break;

  case 254:
#line 1037 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 255:
#line 1039 "xi-grammar.y"
    { (yyval.strval) = 0; }
    break;

  case 256:
#line 1043 "xi-grammar.y"
    {
		   (yyval.sc) = buildAtomic((yyvsp[(4) - (6)].strval), (yyvsp[(6) - (6)].sc), (yyvsp[(2) - (6)].strval));
		 }
    break;

  case 257:
#line 1047 "xi-grammar.y"
    {  
		   in_braces = 0;
		   if (((yyvsp[(4) - (8)].plist)->isVoid() == 0) && ((yyvsp[(4) - (8)].plist)->isMessage() == 0))
                   {
		      connectEntries->append(new Entry(0, 0, new BuiltinType("void"), (yyvsp[(3) - (8)].strval), 
	 	 			new ParamList(new Parameter(lineno, new PtrType( 
                                        new NamedType("CkMarshallMsg")), "_msg")), 0, 0, 0, 1, (yyvsp[(4) - (8)].plist)));
		   }
		   else  {
		      connectEntries->append(new Entry(0, 0, new BuiltinType("void"), (yyvsp[(3) - (8)].strval), (yyvsp[(4) - (8)].plist), 0, 0, 0, 1, (yyvsp[(4) - (8)].plist)));
                   }
                   (yyval.sc) = new SdagConstruct(SCONNECT, (yyvsp[(3) - (8)].strval), (yyvsp[(7) - (8)].strval), (yyvsp[(4) - (8)].plist));
		}
    break;

  case 258:
#line 1061 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SWHEN, 0, 0, 0,0,0, 0,  (yyvsp[(2) - (4)].entrylist)); }
    break;

  case 259:
#line 1063 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SWHEN, 0, 0, 0,0,0, (yyvsp[(3) - (3)].sc), (yyvsp[(2) - (3)].entrylist)); }
    break;

  case 260:
#line 1065 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SWHEN, 0, 0, 0,0,0, (yyvsp[(4) - (5)].sc), (yyvsp[(2) - (5)].entrylist)); }
    break;

  case 261:
#line 1067 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SOVERLAP,0, 0,0,0,0,(yyvsp[(3) - (4)].sc), 0); }
    break;

  case 262:
#line 1069 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFOR, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (11)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(5) - (11)].strval)),
		             new SdagConstruct(SINT_EXPR, (yyvsp[(7) - (11)].strval)), 0, (yyvsp[(10) - (11)].sc), 0); }
    break;

  case 263:
#line 1072 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFOR, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (9)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(5) - (9)].strval)), 
		         new SdagConstruct(SINT_EXPR, (yyvsp[(7) - (9)].strval)), 0, (yyvsp[(9) - (9)].sc), 0); }
    break;

  case 264:
#line 1075 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFORALL, 0, new SdagConstruct(SIDENT, (yyvsp[(3) - (12)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(6) - (12)].strval)), 
		             new SdagConstruct(SINT_EXPR, (yyvsp[(8) - (12)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(10) - (12)].strval)), (yyvsp[(12) - (12)].sc), 0); }
    break;

  case 265:
#line 1078 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFORALL, 0, new SdagConstruct(SIDENT, (yyvsp[(3) - (14)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(6) - (14)].strval)), 
		                 new SdagConstruct(SINT_EXPR, (yyvsp[(8) - (14)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(10) - (14)].strval)), (yyvsp[(13) - (14)].sc), 0); }
    break;

  case 266:
#line 1081 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (6)].strval)), (yyvsp[(6) - (6)].sc),0,0,(yyvsp[(5) - (6)].sc),0); }
    break;

  case 267:
#line 1083 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (8)].strval)), (yyvsp[(8) - (8)].sc),0,0,(yyvsp[(6) - (8)].sc),0); }
    break;

  case 268:
#line 1085 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (5)].strval)), 0,0,0,(yyvsp[(5) - (5)].sc),0); }
    break;

  case 269:
#line 1087 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SWHILE, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (7)].strval)), 0,0,0,(yyvsp[(6) - (7)].sc),0); }
    break;

  case 270:
#line 1089 "xi-grammar.y"
    { (yyval.sc) = (yyvsp[(2) - (3)].sc); }
    break;

  case 271:
#line 1091 "xi-grammar.y"
    { (yyval.sc) = buildAtomic((yyvsp[(2) - (3)].strval), NULL, NULL); }
    break;

  case 272:
#line 1093 "xi-grammar.y"
    { printf("Unknown SDAG construct or malformed entry method definition.\n"
                 "You may have forgotten to terminate an entry method definition with a"
                 " semicolon or forgotten to mark a block of sequential SDAG code as 'atomic'\n"); YYABORT; }
    break;

  case 273:
#line 1099 "xi-grammar.y"
    { (yyval.sc) = 0; }
    break;

  case 274:
#line 1101 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SELSE, 0,0,0,0,0, (yyvsp[(2) - (2)].sc),0); }
    break;

  case 275:
#line 1103 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SELSE, 0,0,0,0,0, (yyvsp[(3) - (4)].sc),0); }
    break;

  case 276:
#line 1106 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFORWARD, new SdagConstruct(SIDENT, (yyvsp[(1) - (1)].strval))); }
    break;

  case 277:
#line 1108 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFORWARD, new SdagConstruct(SIDENT, (yyvsp[(1) - (3)].strval)), (yyvsp[(3) - (3)].sc));  }
    break;

  case 278:
#line 1112 "xi-grammar.y"
    { in_int_expr = 0; (yyval.intval) = 0; }
    break;

  case 279:
#line 1116 "xi-grammar.y"
    { in_int_expr = 1; (yyval.intval) = 0; }
    break;

  case 280:
#line 1120 "xi-grammar.y"
    { (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[(1) - (2)].strval), (yyvsp[(2) - (2)].plist), 0, 0, 0, 0); }
    break;

  case 281:
#line 1122 "xi-grammar.y"
    { (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[(1) - (5)].strval), (yyvsp[(5) - (5)].plist), 0, 0, (yyvsp[(3) - (5)].strval), 0); }
    break;

  case 282:
#line 1126 "xi-grammar.y"
    { (yyval.entrylist) = new EntryList((yyvsp[(1) - (1)].entry)); }
    break;

  case 283:
#line 1128 "xi-grammar.y"
    { (yyval.entrylist) = new EntryList((yyvsp[(1) - (3)].entry),(yyvsp[(3) - (3)].entrylist)); }
    break;

  case 284:
#line 1132 "xi-grammar.y"
    { in_bracket=1; }
    break;

  case 285:
#line 1135 "xi-grammar.y"
    { in_bracket=0; }
    break;

  case 286:
#line 1139 "xi-grammar.y"
    { if (!macroDefined((yyvsp[(2) - (2)].strval), 1)) in_comment = 1; }
    break;

  case 287:
#line 1143 "xi-grammar.y"
    { if (!macroDefined((yyvsp[(2) - (2)].strval), 0)) in_comment = 1; }
    break;


/* Line 1267 of yacc.c.  */
#line 3775 "y.tab.c"
      default: break;
    }
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
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
#if ! YYERROR_VERBOSE
      yyerror (YY_("syntax error"));
#else
      {
	YYSIZE_T yysize = yysyntax_error (0, yystate, yychar);
	if (yymsg_alloc < yysize && yymsg_alloc < YYSTACK_ALLOC_MAXIMUM)
	  {
	    YYSIZE_T yyalloc = 2 * yysize;
	    if (! (yysize <= yyalloc && yyalloc <= YYSTACK_ALLOC_MAXIMUM))
	      yyalloc = YYSTACK_ALLOC_MAXIMUM;
	    if (yymsg != yymsgbuf)
	      YYSTACK_FREE (yymsg);
	    yymsg = (char *) YYSTACK_ALLOC (yyalloc);
	    if (yymsg)
	      yymsg_alloc = yyalloc;
	    else
	      {
		yymsg = yymsgbuf;
		yymsg_alloc = sizeof yymsgbuf;
	      }
	  }

	if (0 < yysize && yysize <= yymsg_alloc)
	  {
	    (void) yysyntax_error (yymsg, yystate, yychar);
	    yyerror (yymsg);
	  }
	else
	  {
	    yyerror (YY_("syntax error"));
	    if (yysize != 0)
	      goto yyexhaustedlab;
	  }
      }
#endif
    }



  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse look-ahead token after an
	 error, discard it.  */

      if (yychar <= YYEOF)
	{
	  /* Return failure if at end of input.  */
	  if (yychar == YYEOF)
	    YYABORT;
	}
      else
	{
	  yydestruct ("Error: discarding",
		      yytoken, &yylval);
	  yychar = YYEMPTY;
	}
    }

  /* Else will try to reuse look-ahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto yyerrorlab;

  /* Do not reclaim the symbols of the rule which action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
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


      yydestruct ("Error: popping",
		  yystos[yystate], yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  if (yyn == YYFINAL)
    YYACCEPT;

  *++yyvsp = yylval;


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

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
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEOF && yychar != YYEMPTY)
     yydestruct ("Cleanup: discarding lookahead",
		 yytoken, &yylval);
  /* Do not reclaim the symbols of the rule which action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
		  yystos[*yyssp], yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
#endif
  /* Make sure YYID is used.  */
  return YYID (yyresult);
}


#line 1146 "xi-grammar.y"

void yyerror(const char *mesg)
{
    std::cerr << cur_file<<":"<<lineno<<": Charmxi syntax error> "
	      << mesg << std::endl;
}

