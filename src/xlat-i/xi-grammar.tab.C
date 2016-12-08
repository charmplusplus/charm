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
#define YYLSP_NEEDED 1



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
     AGGREGATE = 286,
     CREATEHERE = 287,
     CREATEHOME = 288,
     NOKEEP = 289,
     NOTRACE = 290,
     APPWORK = 291,
     VOID = 292,
     CONST = 293,
     SCATTER = 294,
     PACKED = 295,
     VARSIZE = 296,
     ENTRY = 297,
     FOR = 298,
     FORALL = 299,
     WHILE = 300,
     WHEN = 301,
     OVERLAP = 302,
     ATOMIC = 303,
     IF = 304,
     ELSE = 305,
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
     REDUCTIONTARGET = 328,
     SCATTERV = 329,
     CASE = 330
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
#define AGGREGATE 286
#define CREATEHERE 287
#define CREATEHOME 288
#define NOKEEP 289
#define NOTRACE 290
#define APPWORK 291
#define VOID 292
#define CONST 293
#define SCATTER 294
#define PACKED 295
#define VARSIZE 296
#define ENTRY 297
#define FOR 298
#define FORALL 299
#define WHILE 300
#define WHEN 301
#define OVERLAP 302
#define ATOMIC 303
#define IF 304
#define ELSE 305
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
#define SCATTERV 329
#define CASE 330




/* Copy the first part of user declarations.  */
#line 2 "xi-grammar.y"

#include <iostream>
#include <string>
#include <string.h>
#include "xi-symbol.h"
#include "sdag/constructs/Constructs.h"
#include "EToken.h"
#include "xi-Chare.h"

// Has to be a macro since YYABORT can only be used within rule actions.
#define ERROR(...) \
  if (xi::num_errors++ == xi::MAX_NUM_ERRORS) { \
    YYABORT;                                    \
  } else {                                      \
    xi::pretty_msg("error", __VA_ARGS__);       \
  }

#define WARNING(...) \
  if (enable_warnings) {                    \
    xi::pretty_msg("warning", __VA_ARGS__); \
  }

using namespace xi;
extern int yylex (void) ;
extern unsigned char in_comment;
extern unsigned int lineno;
extern int in_bracket,in_braces,in_int_expr;
extern std::list<Entry *> connectEntries;
AstChildren<Module> *modlist;

void yyerror(const char *);

namespace xi {

const int MAX_NUM_ERRORS = 10;
int num_errors = 0;

bool enable_warnings = true;

extern int macroDefined(const char *str, int istrue);
extern const char *python_doc;
extern char *fname;
void splitScopedName(const char* name, const char** scope, const char** basename);
void ReservedWord(int token, int fCol, int lCol);
}


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
#line 51 "xi-grammar.y"
{
  AstChildren<Module> *modlist;
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
  AstChildren<Member> *mbrlist;
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
  unsigned int cattr; // actually Chare::attrib_t, but referring to that creates nasty #include issues
  SdagConstruct *sc;
  IntExprConstruct *intexpr;
  WhenConstruct *when;
  SListConstruct *slist;
  CaseListConstruct *clist;
  OListConstruct *olist;
  SdagEntryConstruct *sentry;
  XStr* xstrptr;
  AccelBlock* accelBlock;
}
/* Line 193 of yacc.c.  */
#line 337 "y.tab.c"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif

#if ! defined YYLTYPE && ! defined YYLTYPE_IS_DECLARED
typedef struct YYLTYPE
{
  int first_line;
  int first_column;
  int last_line;
  int last_column;
} YYLTYPE;
# define yyltype YYLTYPE /* obsolescent; will be withdrawn */
# define YYLTYPE_IS_DECLARED 1
# define YYLTYPE_IS_TRIVIAL 1
#endif


/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 362 "y.tab.c"

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
	 || (defined YYLTYPE_IS_TRIVIAL && YYLTYPE_IS_TRIVIAL \
	     && defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yytype_int16 yyss;
  YYSTYPE yyvs;
    YYLTYPE yyls;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE) + sizeof (YYLTYPE)) \
      + 2 * YYSTACK_GAP_MAXIMUM)

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
#define YYFINAL  57
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   1524

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  92
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  117
/* YYNRULES -- Number of rules.  */
#define YYNRULES  372
/* YYNRULES -- Number of states.  */
#define YYNSTATES  723

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   330

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    86,     2,
      84,    85,    83,     2,    80,    90,    91,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    77,    76,
      81,    89,    82,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    87,     2,    88,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    78,     2,    79,     2,     2,     2,     2,
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
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     5,     6,     9,    10,    12,    14,    17,
      18,    20,    22,    24,    26,    28,    30,    32,    34,    36,
      38,    40,    42,    44,    46,    48,    50,    52,    54,    56,
      58,    60,    62,    64,    66,    68,    70,    72,    74,    76,
      78,    80,    82,    84,    86,    88,    90,    92,    94,    96,
      98,   100,   102,   104,   106,   108,   110,   112,   114,   116,
     118,   120,   125,   129,   133,   135,   140,   141,   144,   148,
     151,   154,   157,   165,   171,   177,   180,   183,   186,   189,
     192,   195,   198,   201,   203,   205,   207,   209,   211,   213,
     215,   217,   221,   222,   224,   225,   229,   231,   233,   235,
     237,   240,   243,   247,   251,   254,   257,   260,   262,   264,
     267,   269,   272,   275,   277,   279,   282,   285,   288,   297,
     299,   301,   303,   305,   308,   311,   314,   316,   318,   320,
     323,   326,   329,   331,   334,   336,   338,   342,   343,   346,
     351,   358,   359,   361,   362,   366,   368,   372,   374,   376,
     377,   381,   383,   387,   388,   390,   392,   393,   397,   399,
     403,   405,   407,   408,   410,   411,   414,   420,   422,   425,
     429,   436,   437,   440,   442,   446,   452,   458,   464,   470,
     475,   479,   486,   493,   499,   505,   511,   517,   523,   528,
     536,   537,   540,   541,   544,   547,   550,   554,   557,   561,
     563,   567,   572,   575,   578,   581,   584,   587,   589,   594,
     595,   598,   600,   602,   604,   606,   609,   612,   615,   619,
     626,   636,   640,   647,   651,   658,   668,   678,   680,   684,
     686,   688,   690,   693,   696,   698,   700,   702,   704,   706,
     708,   710,   712,   714,   716,   718,   720,   728,   734,   748,
     754,   757,   759,   760,   764,   766,   768,   772,   774,   776,
     778,   780,   782,   784,   786,   788,   790,   792,   794,   796,
     798,   801,   803,   805,   807,   809,   811,   813,   815,   817,
     818,   820,   824,   825,   827,   833,   839,   845,   850,   854,
     856,   858,   860,   864,   869,   873,   875,   877,   879,   881,
     886,   890,   895,   900,   905,   909,   917,   923,   930,   932,
     936,   938,   942,   946,   949,   953,   956,   957,   961,   963,
     965,   970,   972,   975,   977,   980,   982,   985,   987,   989,
     990,   995,   999,  1005,  1012,  1017,  1022,  1034,  1044,  1057,
    1072,  1079,  1088,  1094,  1102,  1107,  1114,  1119,  1121,  1126,
    1138,  1148,  1161,  1176,  1183,  1192,  1198,  1206,  1211,  1213,
    1214,  1217,  1222,  1224,  1226,  1228,  1231,  1237,  1239,  1243,
    1245,  1247,  1250
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
      93,     0,    -1,    94,    -1,    -1,   100,    94,    -1,    -1,
       5,    -1,    76,    -1,    96,    76,    -1,    -1,    96,    -1,
      55,    -1,     3,    -1,     4,    -1,     5,    -1,     7,    -1,
       8,    -1,     9,    -1,    11,    -1,    12,    -1,    13,    -1,
      14,    -1,    15,    -1,    19,    -1,    20,    -1,    21,    -1,
      22,    -1,    23,    -1,    24,    -1,    25,    -1,    26,    -1,
      27,    -1,    39,    -1,    28,    -1,    29,    -1,    30,    -1,
      32,    -1,    33,    -1,    34,    -1,    35,    -1,    36,    -1,
      40,    -1,    41,    -1,    42,    -1,    43,    -1,    44,    -1,
      45,    -1,    46,    -1,    47,    -1,    48,    -1,    49,    -1,
      50,    -1,    52,    -1,    54,    -1,    68,    -1,    71,    -1,
      72,    -1,    73,    -1,    74,    -1,    75,    -1,    55,    -1,
      99,    77,    77,    55,    -1,     3,    98,   101,    -1,     4,
      98,   101,    -1,    96,    -1,    78,   102,    79,    97,    -1,
      -1,   104,   102,    -1,    54,    53,    99,    -1,    54,    99,
      -1,    95,   161,    -1,    95,   140,    -1,     5,    42,   171,
     111,    98,   108,   188,    -1,    95,    78,   102,    79,    97,
      -1,    53,    98,    78,   102,    79,    -1,   103,    96,    -1,
     103,   168,    -1,    95,   100,    -1,    95,   143,    -1,    95,
     144,    -1,    95,   145,    -1,    95,   147,    -1,    95,   158,
      -1,   207,    -1,   208,    -1,   170,    -1,     1,    -1,   119,
      -1,    56,    -1,    57,    -1,   105,    -1,   105,    80,   106,
      -1,    -1,   106,    -1,    -1,    81,   107,    82,    -1,    61,
      -1,    62,    -1,    63,    -1,    64,    -1,    67,    61,    -1,
      67,    62,    -1,    67,    62,    61,    -1,    67,    62,    62,
      -1,    67,    63,    -1,    67,    64,    -1,    62,    62,    -1,
      65,    -1,    66,    -1,    62,    66,    -1,    37,    -1,    98,
     108,    -1,    99,   108,    -1,   109,    -1,   111,    -1,   112,
      83,    -1,   113,    83,    -1,   114,    83,    -1,   116,    84,
      83,    98,    85,    84,   186,    85,    -1,   112,    -1,   113,
      -1,   114,    -1,   115,    -1,    39,   112,    -1,    38,   116,
      -1,   116,    38,    -1,   112,    -1,   113,    -1,   114,    -1,
      38,   117,    -1,   117,    38,    -1,   117,    86,    -1,   117,
      -1,   116,    86,    -1,   116,    -1,   177,    -1,   205,   120,
     206,    -1,    -1,   121,   122,    -1,     6,   119,    99,   122,
      -1,     6,    16,   112,    83,    99,   122,    -1,    -1,    37,
      -1,    -1,    87,   127,    88,    -1,   128,    -1,   128,    80,
     127,    -1,    40,    -1,    41,    -1,    -1,    87,   130,    88,
      -1,   135,    -1,   135,    80,   130,    -1,    -1,    57,    -1,
      51,    -1,    -1,    87,   134,    88,    -1,   132,    -1,   132,
      80,   134,    -1,    30,    -1,    51,    -1,    -1,    17,    -1,
      -1,    87,    88,    -1,   136,   119,    98,   137,    96,    -1,
     138,    -1,   138,   139,    -1,    16,   126,   110,    -1,    16,
     126,   110,    78,   139,    79,    -1,    -1,    77,   142,    -1,
     111,    -1,   111,    80,   142,    -1,    11,   129,   110,   141,
     159,    -1,    12,   129,   110,   141,   159,    -1,    13,   129,
     110,   141,   159,    -1,    14,   129,   110,   141,   159,    -1,
      87,    56,    98,    88,    -1,    87,    98,    88,    -1,    15,
     133,   146,   110,   141,   159,    -1,    15,   146,   133,   110,
     141,   159,    -1,    11,   129,    98,   141,   159,    -1,    12,
     129,    98,   141,   159,    -1,    13,   129,    98,   141,   159,
      -1,    14,   129,    98,   141,   159,    -1,    15,   146,    98,
     141,   159,    -1,    16,   126,    98,    96,    -1,    16,   126,
      98,    78,   139,    79,    96,    -1,    -1,    89,   119,    -1,
      -1,    89,    56,    -1,    89,    57,    -1,    89,   111,    -1,
      18,    98,   153,    -1,   115,   154,    -1,   119,    98,   154,
      -1,   155,    -1,   155,    80,   156,    -1,    22,    81,   156,
      82,    -1,   157,   148,    -1,   157,   149,    -1,   157,   150,
      -1,   157,   151,    -1,   157,   152,    -1,    96,    -1,    78,
     160,    79,    97,    -1,    -1,   166,   160,    -1,   123,    -1,
     124,    -1,   163,    -1,   162,    -1,    10,   164,    -1,    19,
     165,    -1,    18,    98,    -1,     8,   125,    99,    -1,     8,
     125,    99,    84,   125,    85,    -1,     8,   125,    99,    81,
     106,    82,    84,   125,    85,    -1,     7,   125,    99,    -1,
       7,   125,    99,    84,   125,    85,    -1,     9,   125,    99,
      -1,     9,   125,    99,    84,   125,    85,    -1,     9,   125,
      99,    81,   106,    82,    84,   125,    85,    -1,     9,    87,
      68,    88,   125,    99,    84,   125,    85,    -1,   111,    -1,
     111,    80,   164,    -1,    57,    -1,   167,    -1,   169,    -1,
     157,   169,    -1,   161,    96,    -1,     1,    -1,    42,    -1,
      79,    -1,     7,    -1,     8,    -1,     9,    -1,    11,    -1,
      12,    -1,    15,    -1,    13,    -1,    14,    -1,     6,    -1,
      42,   172,   171,    98,   188,   190,   191,    -1,    42,   172,
      98,   188,   191,    -1,    42,    87,    68,    88,    37,    98,
     188,   189,   179,   177,   180,    98,    96,    -1,    71,   179,
     177,   180,    96,    -1,    71,    96,    -1,   118,    -1,    -1,
      87,   173,    88,    -1,     1,    -1,   174,    -1,   174,    80,
     173,    -1,    21,    -1,    23,    -1,    24,    -1,    25,    -1,
      32,    -1,    33,    -1,    34,    -1,    35,    -1,    36,    -1,
      26,    -1,    27,    -1,    28,    -1,    52,    -1,    51,   131,
      -1,    72,    -1,    73,    -1,    31,    -1,    74,    -1,     1,
      -1,    57,    -1,    56,    -1,    99,    -1,    -1,    58,    -1,
      58,    80,   176,    -1,    -1,    58,    -1,    58,    87,   177,
      88,   177,    -1,    58,    78,   177,    79,   177,    -1,    58,
      84,   176,    85,   177,    -1,    84,   177,    85,   177,    -1,
     119,    98,    87,    -1,    78,    -1,    79,    -1,   119,    -1,
     119,    98,   136,    -1,   119,    98,    89,   175,    -1,   178,
     177,    88,    -1,     6,    -1,    69,    -1,    70,    -1,    98,
      -1,   183,    90,    82,    98,    -1,   183,    91,    98,    -1,
     183,    87,   183,    88,    -1,   183,    87,    56,    88,    -1,
     183,    84,   183,    85,    -1,   178,   177,    88,    -1,   182,
      77,   119,    98,    81,   183,    82,    -1,   119,    98,    81,
     183,    82,    -1,   182,    77,   184,    81,   183,    82,    -1,
     181,    -1,   181,    80,   186,    -1,   185,    -1,   185,    80,
     187,    -1,    84,   186,    85,    -1,    84,    85,    -1,    87,
     187,    88,    -1,    87,    88,    -1,    -1,    20,    89,    56,
      -1,    96,    -1,   198,    -1,    78,   192,    79,    97,    -1,
     198,    -1,   198,   192,    -1,   198,    -1,   198,   192,    -1,
     196,    -1,   196,   194,    -1,   197,    -1,    57,    -1,    -1,
      46,   204,    78,    79,    -1,    46,   204,   198,    -1,    46,
     204,    78,   192,    79,    -1,    48,   195,   179,   177,   180,
      97,    -1,    47,    78,   193,    79,    -1,    75,    78,   194,
      79,    -1,    43,   202,   177,    76,   177,    76,   177,   201,
      78,   192,    79,    -1,    43,   202,   177,    76,   177,    76,
     177,   201,   198,    -1,    44,    87,    55,    88,   202,   177,
      77,   177,    80,   177,   201,   198,    -1,    44,    87,    55,
      88,   202,   177,    77,   177,    80,   177,   201,    78,   192,
      79,    -1,    49,   202,   177,   201,   198,   199,    -1,    49,
     202,   177,   201,    78,   192,    79,   199,    -1,    45,   202,
     177,   201,   198,    -1,    45,   202,   177,   201,    78,   192,
      79,    -1,   179,   177,   180,    97,    -1,    48,   195,   179,
     177,   180,    97,    -1,    47,    78,   193,    79,    -1,   196,
      -1,    75,    78,   194,    79,    -1,    43,   202,   200,    76,
     200,    76,   200,   201,    78,   192,    79,    -1,    43,   202,
     200,    76,   200,    76,   200,   201,   198,    -1,    44,    87,
      55,    88,   202,   200,    77,   200,    80,   200,   201,   198,
      -1,    44,    87,    55,    88,   202,   200,    77,   200,    80,
     200,   201,    78,   192,    79,    -1,    49,   202,   200,   201,
     198,   199,    -1,    49,   202,   200,   201,    78,   192,    79,
     199,    -1,    45,   202,   200,   201,   198,    -1,    45,   202,
     200,   201,    78,   192,    79,    -1,   179,   177,   180,    97,
      -1,     1,    -1,    -1,    50,   198,    -1,    50,    78,   192,
      79,    -1,   177,    -1,    85,    -1,    84,    -1,    55,   188,
      -1,    55,   205,   177,   206,   188,    -1,   203,    -1,   203,
      80,   204,    -1,    87,    -1,    88,    -1,    59,    98,    -1,
      60,    98,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   194,   194,   199,   202,   207,   208,   212,   214,   219,
     220,   225,   227,   228,   229,   231,   232,   233,   235,   236,
     237,   238,   239,   243,   244,   245,   246,   247,   248,   249,
     250,   251,   252,   253,   254,   255,   256,   257,   258,   259,
     260,   263,   264,   265,   266,   267,   268,   269,   270,   271,
     272,   273,   275,   277,   278,   281,   282,   283,   284,   285,
     288,   290,   297,   301,   308,   310,   315,   316,   320,   322,
     324,   326,   328,   340,   342,   344,   346,   352,   354,   356,
     358,   360,   362,   364,   366,   368,   370,   378,   380,   382,
     386,   388,   393,   394,   399,   400,   404,   406,   408,   410,
     412,   414,   416,   418,   420,   422,   424,   426,   428,   430,
     432,   436,   437,   444,   446,   450,   454,   456,   460,   464,
     466,   468,   470,   472,   474,   476,   480,   482,   484,   486,
     488,   492,   494,   498,   500,   504,   508,   513,   514,   518,
     522,   527,   528,   533,   534,   544,   546,   550,   552,   557,
     558,   562,   564,   569,   570,   574,   579,   580,   584,   586,
     590,   592,   597,   598,   602,   603,   606,   610,   612,   616,
     618,   623,   624,   628,   630,   634,   636,   640,   644,   648,
     654,   658,   660,   664,   666,   670,   674,   678,   682,   684,
     689,   690,   695,   696,   698,   700,   709,   711,   713,   717,
     719,   723,   727,   729,   731,   733,   735,   739,   741,   746,
     753,   757,   759,   761,   762,   764,   766,   768,   772,   774,
     776,   782,   788,   797,   799,   801,   807,   815,   817,   820,
     824,   828,   830,   835,   837,   845,   847,   849,   851,   853,
     855,   857,   859,   861,   863,   865,   868,   877,   893,   909,
     911,   915,   920,   921,   923,   930,   932,   936,   938,   940,
     942,   944,   946,   948,   950,   952,   954,   956,   958,   960,
     962,   964,   966,   968,   980,   982,   991,   993,   995,  1000,
    1001,  1003,  1012,  1013,  1015,  1021,  1027,  1033,  1041,  1048,
    1056,  1063,  1065,  1067,  1069,  1076,  1077,  1078,  1081,  1082,
    1083,  1084,  1091,  1097,  1106,  1113,  1119,  1125,  1133,  1135,
    1139,  1141,  1145,  1147,  1151,  1153,  1158,  1159,  1163,  1165,
    1167,  1171,  1173,  1177,  1179,  1183,  1185,  1187,  1195,  1198,
    1201,  1203,  1205,  1209,  1211,  1213,  1215,  1217,  1219,  1221,
    1223,  1225,  1227,  1229,  1231,  1235,  1237,  1239,  1241,  1243,
    1245,  1247,  1250,  1253,  1255,  1257,  1259,  1261,  1263,  1274,
    1275,  1277,  1281,  1285,  1289,  1293,  1297,  1303,  1305,  1309,
    1312,  1316,  1320
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
  "AGGREGATE", "CREATEHERE", "CREATEHOME", "NOKEEP", "NOTRACE", "APPWORK",
  "VOID", "CONST", "SCATTER", "PACKED", "VARSIZE", "ENTRY", "FOR",
  "FORALL", "WHILE", "WHEN", "OVERLAP", "ATOMIC", "IF", "ELSE", "PYTHON",
  "LOCAL", "NAMESPACE", "USING", "IDENT", "NUMBER", "LITERAL", "CPROGRAM",
  "HASHIF", "HASHIFDEF", "INT", "LONG", "SHORT", "CHAR", "FLOAT", "DOUBLE",
  "UNSIGNED", "ACCEL", "READWRITE", "WRITEONLY", "ACCELBLOCK",
  "MEMCRITICAL", "REDUCTIONTARGET", "SCATTERV", "CASE", "';'", "':'",
  "'{'", "'}'", "','", "'<'", "'>'", "'*'", "'('", "')'", "'&'", "'['",
  "']'", "'='", "'-'", "'.'", "$accept", "File", "ModuleEList",
  "OptExtern", "OneOrMoreSemiColon", "OptSemiColon", "Name", "QualName",
  "Module", "ConstructEList", "ConstructList", "ConstructSemi",
  "Construct", "TParam", "TParamList", "TParamEList", "OptTParams",
  "BuiltinType", "NamedType", "QualNamedType", "SimpleType", "OnePtrType",
  "PtrType", "FuncType", "BaseType", "BaseDataType", "RestrictedType",
  "Type", "ArrayDim", "Dim", "DimList", "Readonly", "ReadonlyMsg",
  "OptVoid", "MAttribs", "MAttribList", "MAttrib", "CAttribs",
  "CAttribList", "PythonOptions", "ArrayAttrib", "ArrayAttribs",
  "ArrayAttribList", "CAttrib", "OptConditional", "MsgArray", "Var",
  "VarList", "Message", "OptBaseList", "BaseList", "Chare", "Group",
  "NodeGroup", "ArrayIndexType", "Array", "TChare", "TGroup", "TNodeGroup",
  "TArray", "TMessage", "OptTypeInit", "OptNameInit", "TVar", "TVarList",
  "TemplateSpec", "Template", "MemberEList", "MemberList",
  "NonEntryMember", "InitNode", "InitProc", "PUPableClass", "IncludeFile",
  "Member", "MemberBody", "UnexpectedToken", "Entry", "AccelBlock",
  "EReturn", "EAttribs", "EAttribList", "EAttrib", "DefaultParameter",
  "CPROGRAM_List", "CCode", "ParamBracketStart", "ParamBraceStart",
  "ParamBraceEnd", "Parameter", "AccelBufferType", "AccelInstName",
  "AccelArrayParam", "AccelParameter", "ParamList", "AccelParamList",
  "EParameters", "AccelEParameters", "OptStackSize", "OptSdagCode",
  "Slist", "Olist", "CaseList", "OptTraceName", "WhenConstruct",
  "NonWhenConstruct", "SingleConstruct", "HasElse", "IntExpr",
  "EndIntExpr", "StartIntExpr", "SEntry", "SEntryList",
  "SParamBracketStart", "SParamBracketEnd", "HashIFComment",
  "HashIFDefComment", 0
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
     325,   326,   327,   328,   329,   330,    59,    58,   123,   125,
      44,    60,    62,    42,    40,    41,    38,    91,    93,    61,
      45,    46
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    92,    93,    94,    94,    95,    95,    96,    96,    97,
      97,    98,    98,    98,    98,    98,    98,    98,    98,    98,
      98,    98,    98,    98,    98,    98,    98,    98,    98,    98,
      98,    98,    98,    98,    98,    98,    98,    98,    98,    98,
      98,    98,    98,    98,    98,    98,    98,    98,    98,    98,
      98,    98,    98,    98,    98,    98,    98,    98,    98,    98,
      99,    99,   100,   100,   101,   101,   102,   102,   103,   103,
     103,   103,   103,   104,   104,   104,   104,   104,   104,   104,
     104,   104,   104,   104,   104,   104,   104,   105,   105,   105,
     106,   106,   107,   107,   108,   108,   109,   109,   109,   109,
     109,   109,   109,   109,   109,   109,   109,   109,   109,   109,
     109,   110,   111,   112,   112,   113,   114,   114,   115,   116,
     116,   116,   116,   116,   116,   116,   117,   117,   117,   117,
     117,   118,   118,   119,   119,   120,   121,   122,   122,   123,
     124,   125,   125,   126,   126,   127,   127,   128,   128,   129,
     129,   130,   130,   131,   131,   132,   133,   133,   134,   134,
     135,   135,   136,   136,   137,   137,   138,   139,   139,   140,
     140,   141,   141,   142,   142,   143,   143,   144,   145,   146,
     146,   147,   147,   148,   148,   149,   150,   151,   152,   152,
     153,   153,   154,   154,   154,   154,   155,   155,   155,   156,
     156,   157,   158,   158,   158,   158,   158,   159,   159,   160,
     160,   161,   161,   161,   161,   161,   161,   161,   162,   162,
     162,   162,   162,   163,   163,   163,   163,   164,   164,   165,
     166,   167,   167,   167,   167,   168,   168,   168,   168,   168,
     168,   168,   168,   168,   168,   168,   169,   169,   169,   170,
     170,   171,   172,   172,   172,   173,   173,   174,   174,   174,
     174,   174,   174,   174,   174,   174,   174,   174,   174,   174,
     174,   174,   174,   174,   174,   174,   175,   175,   175,   176,
     176,   176,   177,   177,   177,   177,   177,   177,   178,   179,
     180,   181,   181,   181,   181,   182,   182,   182,   183,   183,
     183,   183,   183,   183,   184,   185,   185,   185,   186,   186,
     187,   187,   188,   188,   189,   189,   190,   190,   191,   191,
     191,   192,   192,   193,   193,   194,   194,   194,   195,   195,
     196,   196,   196,   197,   197,   197,   197,   197,   197,   197,
     197,   197,   197,   197,   197,   198,   198,   198,   198,   198,
     198,   198,   198,   198,   198,   198,   198,   198,   198,   199,
     199,   199,   200,   201,   202,   203,   203,   204,   204,   205,
     206,   207,   208
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     0,     2,     0,     1,     1,     2,     0,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     4,     3,     3,     1,     4,     0,     2,     3,     2,
       2,     2,     7,     5,     5,     2,     2,     2,     2,     2,
       2,     2,     2,     1,     1,     1,     1,     1,     1,     1,
       1,     3,     0,     1,     0,     3,     1,     1,     1,     1,
       2,     2,     3,     3,     2,     2,     2,     1,     1,     2,
       1,     2,     2,     1,     1,     2,     2,     2,     8,     1,
       1,     1,     1,     2,     2,     2,     1,     1,     1,     2,
       2,     2,     1,     2,     1,     1,     3,     0,     2,     4,
       6,     0,     1,     0,     3,     1,     3,     1,     1,     0,
       3,     1,     3,     0,     1,     1,     0,     3,     1,     3,
       1,     1,     0,     1,     0,     2,     5,     1,     2,     3,
       6,     0,     2,     1,     3,     5,     5,     5,     5,     4,
       3,     6,     6,     5,     5,     5,     5,     5,     4,     7,
       0,     2,     0,     2,     2,     2,     3,     2,     3,     1,
       3,     4,     2,     2,     2,     2,     2,     1,     4,     0,
       2,     1,     1,     1,     1,     2,     2,     2,     3,     6,
       9,     3,     6,     3,     6,     9,     9,     1,     3,     1,
       1,     1,     2,     2,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     7,     5,    13,     5,
       2,     1,     0,     3,     1,     1,     3,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       2,     1,     1,     1,     1,     1,     1,     1,     1,     0,
       1,     3,     0,     1,     5,     5,     5,     4,     3,     1,
       1,     1,     3,     4,     3,     1,     1,     1,     1,     4,
       3,     4,     4,     4,     3,     7,     5,     6,     1,     3,
       1,     3,     3,     2,     3,     2,     0,     3,     1,     1,
       4,     1,     2,     1,     2,     1,     2,     1,     1,     0,
       4,     3,     5,     6,     4,     4,    11,     9,    12,    14,
       6,     8,     5,     7,     4,     6,     4,     1,     4,    11,
       9,    12,    14,     6,     8,     5,     7,     4,     1,     0,
       2,     4,     1,     1,     1,     2,     5,     1,     3,     1,
       1,     2,     2
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       3,     0,     0,     0,     2,     3,    12,    13,    14,    15,
      16,    17,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    31,    33,    34,    35,    36,
      37,    38,    39,    40,    32,    41,    42,    43,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    11,    54,
      55,    56,    57,    58,    59,     0,     0,     1,     4,     7,
       0,    64,    62,    63,    86,     6,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    85,    83,    84,     8,     0,
       0,     0,    60,    69,   371,   372,   289,   250,   282,     0,
     141,   141,   141,     0,   149,   149,   149,   149,     0,   143,
       0,     0,     0,     0,    77,   211,   212,    71,    78,    79,
      80,    81,     0,    82,    70,   214,   213,     9,   245,   237,
     238,   239,   240,   241,   243,   244,   242,   235,   236,    75,
      76,    67,   110,     0,    96,    97,    98,    99,   107,   108,
       0,    94,   113,   114,   126,   127,   128,   132,   251,     0,
       0,    68,     0,   283,   282,     0,     0,     0,     0,   119,
     120,   121,   122,   134,     0,   142,     0,     0,     0,     0,
     227,   215,     0,     0,     0,     0,     0,     0,     0,   156,
       0,     0,   217,   229,   216,     0,     0,   149,   149,   149,
     149,     0,   143,   202,   203,   204,   205,   206,    10,    65,
     129,   106,   109,   100,   101,   104,   105,    92,   112,   115,
     116,   117,   130,   131,     0,     0,     0,   282,   279,   282,
       0,   290,     0,     0,   124,   123,   125,     0,   133,   137,
     221,   218,     0,   223,     0,   160,   161,     0,   151,    94,
     171,   171,   171,   171,   155,     0,     0,   158,     0,     0,
       0,     0,     0,   147,   148,     0,   145,   169,     0,   122,
       0,   199,     0,     9,     0,     0,     0,     0,     0,     0,
     102,   103,    88,    89,    90,    93,     0,    87,    94,    74,
      61,     0,   280,     0,     0,   282,   249,     0,     0,   369,
     137,   139,   282,   141,     0,   141,   141,     0,   141,   228,
     150,     0,   111,     0,     0,     0,     0,     0,     0,   180,
       0,   157,   171,   171,   144,     0,   162,   190,     0,   197,
     192,     0,   201,    73,   171,   171,   171,   171,   171,     0,
       0,    95,     0,   282,   279,   282,   282,   287,   137,     0,
     138,     0,   135,     0,     0,     0,     0,     0,     0,   152,
     173,   172,     0,   207,   175,   176,   177,   178,   179,   159,
       0,     0,   146,   163,     0,   162,     0,     0,   196,   193,
     194,   195,   198,   200,     0,     0,     0,     0,     0,   162,
     188,    91,     0,    72,   285,   281,   286,   284,   140,     0,
     370,   136,   222,     0,   219,     0,     0,   224,     0,   234,
       0,     0,     0,     0,     0,   230,   231,   181,   182,     0,
     168,   170,   191,   183,   184,   185,   186,   187,     0,   313,
     291,   282,   308,     0,     0,   141,   141,   141,   174,   254,
       0,     0,   232,     9,   233,   210,   164,     0,   162,     0,
       0,   312,     0,     0,     0,     0,   275,   257,   258,   259,
     260,   266,   267,   268,   273,   261,   262,   263,   264,   265,
     153,   269,     0,   271,   272,   274,     0,   255,    60,     0,
       0,   208,     0,     0,   189,   288,     0,   292,   294,   309,
     118,   220,   226,   225,   154,   270,     0,   253,     0,     0,
       0,   165,   166,   277,   276,   278,   293,     0,   256,   358,
       0,     0,     0,     0,     0,   329,     0,     0,     0,   318,
     282,   247,   347,   319,   316,     0,   364,   282,     0,   282,
       0,   367,     0,     0,   328,     0,   282,     0,     0,     0,
       0,     0,     0,     0,   362,     0,     0,     0,   365,   282,
       0,     0,   331,     0,     0,   282,     0,     0,     0,     0,
       0,   329,     0,     0,   282,     0,   325,   327,     9,   322,
       9,     0,   246,     0,     0,   282,     0,   363,     0,     0,
     368,   330,     0,   346,   324,     0,     0,   282,     0,   282,
       0,     0,   282,     0,     0,   348,   326,   320,   357,   317,
     295,   296,   297,   315,     0,     0,   310,     0,   282,     0,
     282,     0,   355,     0,   332,     9,     0,   359,     0,     0,
       0,     0,   282,     0,     0,     9,     0,     0,     0,   314,
       0,   282,     0,     0,   366,   345,     0,     0,   353,   282,
       0,     0,   334,     0,     0,   335,   344,     0,     0,   282,
       0,   311,     0,     0,   282,   356,   359,     0,   360,     0,
     282,     0,   342,     9,     0,   359,   298,     0,     0,     0,
       0,     0,     0,     0,   354,     0,   282,     0,     0,   333,
       0,   340,   306,     0,     0,     0,     0,     0,   304,     0,
     248,     0,   350,   282,   361,     0,   282,   343,   359,     0,
       0,     0,     0,   300,     0,   307,     0,     0,     0,     0,
     341,   303,   302,   301,   299,   305,   349,     0,     0,   337,
     282,     0,   351,     0,     0,     0,   336,     0,   352,     0,
     338,     0,   339
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     3,     4,    71,   353,   199,   239,   141,     5,    62,
      72,    73,    74,   274,   275,   276,   208,   142,   240,   143,
     159,   160,   161,   162,   163,   147,   148,   277,   341,   290,
     291,   105,   106,   166,   181,   255,   256,   173,   237,   485,
     247,   178,   248,   238,   364,   473,   365,   366,   107,   304,
     351,   108,   109,   110,   179,   111,   193,   194,   195,   196,
     197,   368,   319,   261,   262,   401,   113,   354,   402,   403,
     115,   116,   171,   184,   404,   405,   130,   406,    75,   149,
     431,   466,   467,   496,   283,   534,   421,   510,   222,   422,
     595,   657,   640,   596,   423,   597,   383,   564,   532,   511,
     528,   543,   555,   525,   512,   557,   529,   628,   535,   568,
     517,   521,   522,   292,   391,    76,    77
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -526
static const yytype_int16 yypact[] =
{
      56,  1348,  1348,    24,  -526,    56,  -526,  -526,  -526,  -526,
    -526,  -526,  -526,  -526,  -526,  -526,  -526,  -526,  -526,  -526,
    -526,  -526,  -526,  -526,  -526,  -526,  -526,  -526,  -526,  -526,
    -526,  -526,  -526,  -526,  -526,  -526,  -526,  -526,  -526,  -526,
    -526,  -526,  -526,  -526,  -526,  -526,  -526,  -526,  -526,  -526,
    -526,  -526,  -526,  -526,  -526,   107,   107,  -526,  -526,  -526,
     766,   -27,  -526,  -526,  -526,    92,  1348,   201,  1348,  1348,
     207,   930,    81,   908,   766,  -526,  -526,  -526,  -526,   785,
      88,   136,  -526,   129,  -526,  -526,  -526,   -27,    34,  1369,
     183,   183,   -14,   136,   142,   142,   142,   142,   147,   150,
    1348,   186,   179,   766,  -526,  -526,  -526,  -526,  -526,  -526,
    -526,  -526,   436,  -526,  -526,  -526,  -526,   197,  -526,  -526,
    -526,  -526,  -526,  -526,  -526,  -526,  -526,  -526,  -526,   -27,
    -526,  -526,  -526,   785,  -526,   141,  -526,  -526,  -526,  -526,
     227,   128,  -526,  -526,   220,   222,   231,    -3,  -526,   136,
     766,   129,   203,    66,    34,   236,   133,  1457,   133,   220,
     222,   231,  -526,    -7,   136,  -526,   136,   136,   254,   136,
     243,  -526,    31,  1348,  1348,  1348,  1348,  1129,   251,   256,
     229,  1348,  -526,  -526,  -526,  1400,   261,   142,   142,   142,
     142,   251,   150,  -526,  -526,  -526,  -526,  -526,   -27,  -526,
     310,  -526,  -526,  -526,   113,  -526,  -526,  1444,  -526,  -526,
    -526,  -526,  -526,  -526,  1348,   275,   303,    34,   301,    34,
     277,  -526,   197,   280,   -10,  -526,  -526,   282,  -526,   -34,
      71,   105,   278,   198,   136,  -526,  -526,   287,   296,   297,
     300,   300,   300,   300,  -526,  1348,   291,   305,   293,  1202,
    1348,   335,  1348,  -526,  -526,   299,   308,   318,  1348,   122,
    1348,   317,   316,   197,  1348,  1348,  1348,  1348,  1348,  1348,
    -526,  -526,  -526,  -526,   321,  -526,   320,  -526,   297,  -526,
    -526,   326,   327,   332,   322,    34,   -27,   136,  1348,  -526,
     319,  -526,    34,   183,  1444,   183,   183,  1444,   183,  -526,
    -526,    31,  -526,   136,   257,   257,   257,   257,   323,  -526,
     335,  -526,   300,   300,  -526,   229,   395,   329,   245,  -526,
     333,  1400,  -526,  -526,   300,   300,   300,   300,   300,   258,
    1444,  -526,   339,    34,   301,    34,    34,  -526,   -34,   342,
    -526,   340,  -526,   344,   350,   348,   136,   352,   359,  -526,
     355,  -526,   467,   -27,  -526,  -526,  -526,  -526,  -526,  -526,
     257,   257,  -526,  -526,  1457,    29,   374,  1457,  -526,  -526,
    -526,  -526,  -526,  -526,   257,   257,   257,   257,   257,   395,
     -27,  -526,  1413,  -526,  -526,  -526,  -526,  -526,  -526,   370,
    -526,  -526,  -526,   372,  -526,   151,   373,  -526,   136,  -526,
     681,   416,   380,   197,   467,  -526,  -526,  -526,  -526,  1348,
    -526,  -526,  -526,  -526,  -526,  -526,  -526,  -526,   384,  -526,
    1348,    34,   385,   381,  1457,   183,   183,   183,  -526,  -526,
     946,  1056,  -526,   197,   -27,  -526,   391,   197,     1,   379,
    1457,  -526,   386,   387,   396,   397,  -526,  -526,  -526,  -526,
    -526,  -526,  -526,  -526,  -526,  -526,  -526,  -526,  -526,  -526,
     426,  -526,   399,  -526,  -526,  -526,   400,   404,   406,   339,
    1348,  -526,   409,   197,   -27,  -526,   263,  -526,  -526,  -526,
    -526,  -526,  -526,  -526,  -526,  -526,   455,  -526,  1000,   249,
     339,  -526,   -27,  -526,  -526,   129,  -526,  1348,  -526,  -526,
     417,   413,   417,   447,   425,   449,   417,   430,    93,   -27,
      34,  -526,  -526,  -526,   484,   339,  -526,    34,   456,    34,
      59,   432,   531,   542,  -526,   435,    34,   264,   431,   346,
     236,   428,   249,   433,  -526,   439,   434,   438,  -526,    34,
     447,   325,  -526,   440,   493,    34,   438,   417,   437,   417,
     443,   449,   417,   448,    34,   446,   264,  -526,   197,  -526,
     197,   471,  -526,   376,   435,    34,   417,  -526,   602,   340,
    -526,  -526,   450,  -526,  -526,   236,   749,    34,   475,    34,
     542,   435,    34,   264,   236,  -526,  -526,  -526,  -526,  -526,
    -526,  -526,  -526,  -526,  1348,   454,   453,   459,    34,   468,
      34,    93,  -526,   339,  -526,   197,    93,   498,   473,   462,
     438,   472,    34,   438,   474,   197,   476,  1457,   601,  -526,
     236,    34,   477,   482,  -526,  -526,   485,   756,  -526,    34,
     417,   763,  -526,   236,   812,  -526,  -526,  1348,  1348,    34,
     486,  -526,  1348,   438,    34,  -526,   498,    93,  -526,   480,
      34,    93,  -526,   197,    93,   498,  -526,   134,    41,   481,
    1348,   197,   820,   483,  -526,   491,    34,   488,   494,  -526,
     502,  -526,  -526,  1348,  1275,   501,  1348,  1348,  -526,   149,
     -27,    93,  -526,    34,  -526,   438,    34,  -526,   498,   187,
     496,   265,  1348,  -526,   171,  -526,   513,   438,   827,   514,
    -526,  -526,  -526,  -526,  -526,  -526,  -526,   834,    93,  -526,
      34,    93,  -526,   516,   438,   517,  -526,   883,  -526,    93,
    -526,   518,  -526
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -526,  -526,   593,  -526,   -53,  -254,    -1,   -60,   528,   545,
     -47,  -526,  -526,  -526,  -252,  -526,  -220,  -526,  -137,   -77,
     -71,   -67,   -64,  -172,   458,   478,  -526,   -83,  -526,  -526,
    -227,  -526,  -526,   -81,   412,   290,  -526,   -63,   309,  -526,
    -526,   442,   302,  -526,   178,  -526,  -526,  -281,  -526,     4,
     221,  -526,  -526,  -526,   -46,  -526,  -526,  -526,  -526,  -526,
    -526,  -526,   298,  -526,   304,   551,  -526,   253,   219,   553,
    -526,  -526,   398,  -526,  -526,  -526,  -526,   232,  -526,   204,
    -526,   146,  -526,  -526,   324,   -84,    25,   -65,  -504,  -526,
    -526,  -493,  -526,  -526,  -399,    26,  -439,  -526,  -526,   111,
    -489,    72,  -460,   102,  -436,  -526,  -475,  -525,  -490,  -524,
    -452,  -526,   114,   135,    91,  -526,  -526
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -324
static const yytype_int16 yytable[] =
{
      55,    56,    61,    61,   155,    88,   164,    83,   144,   323,
     167,   169,   145,   259,   513,   146,   170,    87,   363,   302,
     129,   151,   576,   165,    57,   442,   560,   131,   226,   537,
     489,   226,   174,   175,   176,   212,   546,   241,   242,   243,
     559,   479,   344,   152,   257,   347,   363,   542,   544,    78,
     519,   514,   572,   289,   526,   574,   186,   513,   332,     1,
       2,   235,   144,   340,   198,    80,   145,    84,    85,   146,
     220,   605,   214,   168,   227,   599,   533,   227,   381,   228,
     615,   538,   236,   213,   410,   223,   631,   225,   475,   634,
     476,   556,   153,   602,   499,   577,   586,   579,   418,   182,
     582,   607,   260,   215,   229,   544,   230,   231,  -167,   233,
     622,   388,   623,   312,   600,   313,   642,   626,   154,   662,
     556,   664,   677,   614,   264,   265,   266,   267,   475,   653,
     671,   643,   250,   281,    79,   284,   500,   501,   502,   503,
     504,   505,   506,   382,   217,   268,   289,   556,   152,   259,
     218,  -289,   648,   219,   663,   293,   652,   170,   665,   655,
     117,   698,   668,   700,   624,   670,   150,   679,   507,   286,
     132,    86,  -289,   707,   270,   271,   246,  -289,   650,   471,
     689,   691,   152,    59,   694,    60,   294,   682,    82,   295,
     717,    82,   696,   697,   134,   135,   136,   137,   138,   139,
     140,   337,  -192,   201,  -192,   152,   152,   202,   342,   207,
     198,   318,   343,   278,   345,   346,   672,   348,   673,   713,
     165,   674,   715,   709,   675,   676,   350,   338,   152,   172,
     721,   695,   712,   673,   177,   426,   674,   180,   260,   675,
     676,   371,   720,   183,   308,   305,   306,   307,   246,   384,
     499,   386,   387,   705,    81,   673,    82,   317,   674,   320,
     185,   675,   676,   324,   325,   326,   327,   328,   329,   253,
     254,   673,   701,    59,   674,   152,   380,   675,   676,   297,
     216,   409,   298,    59,   412,    86,   395,   339,   203,   204,
     205,   206,   500,   501,   502,   503,   504,   505,   506,   420,
      82,   369,   370,   209,   587,   210,   588,   547,   548,   549,
     503,   550,   551,   552,   211,   221,   360,   361,    82,   493,
     494,   350,   232,   234,   507,    59,   499,   508,   374,   375,
     376,   377,   378,    59,    59,   352,   379,   439,   249,   553,
     263,   420,    86,   251,   443,   444,   445,   499,   212,   673,
     434,   625,   674,   703,   279,   675,   676,   420,   280,   282,
     144,   636,   285,   287,   145,   288,   296,   146,   500,   501,
     502,   503,   504,   505,   506,   300,   301,   303,   207,   309,
     198,   311,   590,  -289,   474,   310,   244,   314,   315,   500,
     501,   502,   503,   504,   505,   506,   316,   321,   322,   669,
     507,   330,   331,    86,   571,   333,   289,   334,   436,  -289,
     336,   358,   363,   132,   157,   158,   495,   335,   367,   438,
     492,   507,   318,   382,    86,  -321,   530,   389,   390,   392,
     469,    82,   393,   394,   396,   398,   509,   134,   135,   136,
     137,   138,   139,   140,   397,   591,   592,   187,   188,   189,
     190,   191,   192,   411,   424,   569,   425,   427,   400,   433,
     545,   575,   554,   437,   593,   440,   441,   478,   399,   490,
     584,   480,   481,    89,    90,    91,    92,    93,   472,   509,
     594,   482,   483,   484,   488,   100,   101,   486,   487,   102,
     -11,   554,   497,   608,   499,   610,   515,   491,   613,   598,
     518,   516,   520,   523,   531,   198,   524,   198,   527,   400,
     558,   536,   540,    86,   620,   565,   612,   561,   554,   573,
     563,   580,   566,   567,   578,   585,   583,   589,   633,   604,
     609,   617,   499,   618,   638,   594,   500,   501,   502,   503,
     504,   505,   506,   499,   621,   649,  -209,   619,   627,   629,
     630,   632,   198,   635,   644,   659,   666,   637,   355,   356,
     357,   645,   198,   683,   646,   686,   667,   660,   507,   678,
     684,    86,  -323,   687,   500,   501,   502,   503,   504,   505,
     506,   688,   685,   692,   702,   500,   501,   502,   503,   504,
     505,   506,   706,   616,   710,   716,   718,   722,    58,   104,
     198,    63,   699,   499,   269,   362,   507,   590,   680,   541,
     349,   200,   359,   407,   408,   224,   477,   507,   372,   428,
      86,   252,   112,   435,   114,   373,   714,   413,   414,   415,
     416,   417,   299,   432,   498,   470,   656,   658,   132,   157,
     158,   661,   639,   562,   641,   500,   501,   502,   503,   504,
     505,   506,   611,   581,   570,   539,    82,     0,   385,   656,
     603,     0,   134,   135,   136,   137,   138,   139,   140,     0,
     591,   592,   656,   656,     0,   693,   656,   507,     0,     0,
     601,     0,   429,     0,  -252,  -252,  -252,     0,  -252,  -252,
    -252,   704,  -252,  -252,  -252,  -252,  -252,     0,     0,     0,
    -252,  -252,  -252,  -252,  -252,  -252,  -252,  -252,  -252,  -252,
    -252,  -252,     0,  -252,  -252,  -252,  -252,  -252,  -252,  -252,
    -252,  -252,  -252,  -252,  -252,  -252,  -252,  -252,  -252,  -252,
    -252,  -252,     0,  -252,     0,  -252,  -252,     0,     0,     0,
       0,     0,  -252,  -252,  -252,  -252,  -252,  -252,  -252,  -252,
     499,     0,  -252,  -252,  -252,  -252,  -252,   499,     0,     0,
       0,     0,     0,     0,   499,     0,     0,    64,   430,    -5,
      -5,    65,    -5,    -5,    -5,    -5,    -5,    -5,    -5,    -5,
      -5,    -5,    -5,     0,    -5,    -5,     0,     0,    -5,     0,
       0,     0,   500,   501,   502,   503,   504,   505,   506,   500,
     501,   502,   503,   504,   505,   506,   500,   501,   502,   503,
     504,   505,   506,   499,     0,     0,     0,     0,     0,    66,
      67,   499,   132,   133,   507,    68,    69,   606,   499,     0,
       0,   507,     0,     0,   647,   499,     0,    70,   507,     0,
      82,   651,     0,     0,    -5,   -66,   134,   135,   136,   137,
     138,   139,   140,     0,     0,   500,   501,   502,   503,   504,
     505,   506,     0,   500,   501,   502,   503,   504,   505,   506,
     500,   501,   502,   503,   504,   505,   506,   500,   501,   502,
     503,   504,   505,   506,   499,     0,     0,   507,     0,     0,
     654,     0,     0,     0,     0,   507,     0,     0,   681,     0,
       0,     0,   507,     0,     0,   708,     0,     0,     0,   507,
       0,     0,   711,     0,   118,   119,   120,   121,     0,   122,
     123,   124,   125,   126,     0,     0,   500,   501,   502,   503,
     504,   505,   506,     1,     2,     0,    89,    90,    91,    92,
      93,    94,    95,    96,    97,    98,    99,   446,   100,   101,
     127,     0,   102,     0,     0,     0,     0,     0,   507,     0,
       0,   719,     0,     0,     0,     0,     0,   447,     0,   448,
     449,   450,   451,   452,   453,     0,     0,   454,   455,   456,
     457,   458,   459,     0,    59,     0,     0,   128,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   460,   461,     0,
       0,   446,     0,     0,     0,     0,     0,     0,   103,     0,
       0,     0,     0,     0,   462,     0,     0,     0,   463,   464,
     465,   447,     0,   448,   449,   450,   451,   452,   453,     0,
       0,   454,   455,   456,   457,   458,   459,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   460,   461,     0,     0,     0,     0,     0,     0,     6,
       7,     8,     0,     9,    10,    11,     0,    12,    13,    14,
      15,    16,   463,   464,   465,    17,    18,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,     0,    29,    30,
      31,    32,    33,   132,   133,    34,    35,    36,    37,    38,
      39,    40,    41,    42,    43,    44,    45,     0,    46,     0,
      47,   468,     0,     0,     0,     0,     0,   134,   135,   136,
     137,   138,   139,   140,    49,     0,     0,    50,    51,    52,
      53,    54,     6,     7,     8,     0,     9,    10,    11,     0,
      12,    13,    14,    15,    16,     0,     0,     0,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
       0,    29,    30,    31,    32,    33,     0,     0,    34,    35,
      36,    37,    38,    39,    40,    41,    42,    43,    44,    45,
     244,    46,     0,    47,    48,   245,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    49,     0,     0,
      50,    51,    52,    53,    54,     6,     7,     8,     0,     9,
      10,    11,     0,    12,    13,    14,    15,    16,     0,     0,
       0,    17,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,     0,    29,    30,    31,    32,    33,     0,
       0,    34,    35,    36,    37,    38,    39,    40,    41,    42,
      43,    44,    45,     0,    46,     0,    47,    48,   245,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      49,     0,     0,    50,    51,    52,    53,    54,     6,     7,
       8,     0,     9,    10,    11,     0,    12,    13,    14,    15,
      16,     0,     0,     0,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,     0,    29,    30,    31,
      32,    33,     0,     0,    34,    35,    36,    37,    38,    39,
      40,    41,    42,    43,    44,    45,     0,    46,     0,    47,
      48,   690,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    49,     0,     0,    50,    51,    52,    53,
      54,     6,     7,     8,     0,     9,    10,    11,     0,    12,
      13,    14,    15,    16,     0,     0,     0,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,     0,
      29,    30,    31,    32,    33,   156,     0,    34,    35,    36,
      37,    38,    39,    40,    41,    42,    43,    44,    45,     0,
      46,     0,    47,    48,     0,     0,   132,   157,   158,     0,
       0,     0,     0,     0,     0,     0,    49,     0,   258,    50,
      51,    52,    53,    54,    82,     0,     0,     0,     0,     0,
     134,   135,   136,   137,   138,   139,   140,   132,   157,   158,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     132,   157,   158,     0,     0,    82,     0,     0,     0,     0,
       0,   134,   135,   136,   137,   138,   139,   140,    82,     0,
       0,     0,     0,     0,   134,   135,   136,   137,   138,   139,
     140,   132,   157,   158,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   132,   157,   158,     0,   419,    82,
     272,   273,     0,     0,     0,   134,   135,   136,   137,   138,
     139,   140,    82,     0,     0,     0,     0,     0,   134,   135,
     136,   137,   138,   139,   140
};

static const yytype_int16 yycheck[] =
{
       1,     2,    55,    56,    88,    70,    89,    67,    79,   263,
      91,    92,    79,   185,   489,    79,    93,    70,    17,   239,
      73,    81,   546,    37,     0,   424,   530,    74,    38,   519,
     469,    38,    95,    96,    97,    38,   526,   174,   175,   176,
     529,   440,   294,    77,   181,   297,    17,   522,   523,    76,
     502,   490,   541,    87,   506,   544,   103,   532,   278,     3,
       4,    30,   133,   290,   117,    66,   133,    68,    69,   133,
     154,   575,   149,    87,    84,   565,   515,    84,   330,    86,
     584,   520,    51,    86,   365,   156,   610,   158,    87,   613,
      89,   527,    58,   568,     1,   547,   556,   549,   379,   100,
     552,   576,   185,   150,   164,   580,   166,   167,    79,   169,
     600,   338,   601,   250,   566,   252,   620,   606,    84,   643,
     556,   646,    81,   583,   187,   188,   189,   190,    87,   633,
     655,   621,   178,   217,    42,   219,    43,    44,    45,    46,
      47,    48,    49,    84,    78,   191,    87,   583,    77,   321,
      84,    58,   627,    87,   644,    84,   631,   234,   647,   634,
      79,   685,   651,   688,   603,   654,    78,   660,    75,   222,
      37,    78,    79,   697,    61,    62,   177,    84,   630,   433,
     673,   674,    77,    76,   677,    78,    81,   662,    55,    84,
     714,    55,   681,   683,    61,    62,    63,    64,    65,    66,
      67,   285,    80,    62,    82,    77,    77,    66,   292,    81,
     263,    89,   293,   214,   295,   296,    82,   298,    84,   708,
      37,    87,   711,   698,    90,    91,   303,   287,    77,    87,
     719,    82,   707,    84,    87,    84,    87,    87,   321,    90,
      91,   318,   717,    57,   245,   241,   242,   243,   249,   333,
       1,   335,   336,    82,    53,    84,    55,   258,    87,   260,
      81,    90,    91,   264,   265,   266,   267,   268,   269,    40,
      41,    84,    85,    76,    87,    77,   329,    90,    91,    81,
      77,   364,    84,    76,   367,    78,   346,   288,    61,    62,
      63,    64,    43,    44,    45,    46,    47,    48,    49,   382,
      55,    56,    57,    83,   558,    83,   560,    43,    44,    45,
      46,    47,    48,    49,    83,    79,   312,   313,    55,    56,
      57,   398,    68,    80,    75,    76,     1,    78,   324,   325,
     326,   327,   328,    76,    76,    78,    78,   421,    87,    75,
      79,   424,    78,    87,   425,   426,   427,     1,    38,    84,
     403,   605,    87,    88,    79,    90,    91,   440,    55,    58,
     431,   615,    85,    83,   431,    83,    88,   431,    43,    44,
      45,    46,    47,    48,    49,    88,    80,    77,    81,    88,
     433,    88,     6,    58,   437,    80,    51,    88,    80,    43,
      44,    45,    46,    47,    48,    49,    78,    80,    82,   653,
      75,    80,    82,    78,    79,    79,    87,    80,   409,    84,
      88,    88,    17,    37,    38,    39,   476,    85,    89,   420,
     473,    75,    89,    84,    78,    79,   510,    85,    88,    85,
     431,    55,    82,    85,    82,    80,   489,    61,    62,    63,
      64,    65,    66,    67,    85,    69,    70,    11,    12,    13,
      14,    15,    16,    79,    84,   539,    84,    84,    42,    79,
     525,   545,   527,    79,    88,    80,    85,    88,     1,   470,
     554,    85,    85,     6,     7,     8,     9,    10,    87,   532,
     563,    85,    85,    57,    80,    18,    19,    88,    88,    22,
      84,   556,    37,   577,     1,   579,   497,    88,   582,   564,
      87,    84,    55,    78,    20,   558,    57,   560,    78,    42,
      79,    55,    80,    78,   598,    76,   581,    89,   583,    79,
      87,    78,    88,    85,    87,    79,    78,    56,   612,    79,
      55,    77,     1,    80,   617,   618,    43,    44,    45,    46,
      47,    48,    49,     1,    76,   629,    79,    88,    50,    76,
      88,    79,   605,    79,    77,   639,    76,    81,   305,   306,
     307,    79,   615,    80,    79,    77,   650,    81,    75,    88,
      79,    78,    79,    79,    43,    44,    45,    46,    47,    48,
      49,    79,   666,    82,    88,    43,    44,    45,    46,    47,
      48,    49,    79,   594,    80,    79,    79,    79,     5,    71,
     653,    56,   686,     1,   192,   315,    75,     6,   661,    78,
     301,   133,   310,   360,   361,   157,   438,    75,   320,   398,
      78,   179,    71,   404,    71,   321,   710,   374,   375,   376,
     377,   378,   234,   401,   488,   431,   637,   638,    37,    38,
      39,   642,   617,   532,   618,    43,    44,    45,    46,    47,
      48,    49,   580,   551,   540,   520,    55,    -1,   334,   660,
     569,    -1,    61,    62,    63,    64,    65,    66,    67,    -1,
      69,    70,   673,   674,    -1,   676,   677,    75,    -1,    -1,
      78,    -1,     1,    -1,     3,     4,     5,    -1,     7,     8,
       9,   692,    11,    12,    13,    14,    15,    -1,    -1,    -1,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    -1,    32,    33,    34,    35,    36,    37,    38,
      39,    40,    41,    42,    43,    44,    45,    46,    47,    48,
      49,    50,    -1,    52,    -1,    54,    55,    -1,    -1,    -1,
      -1,    -1,    61,    62,    63,    64,    65,    66,    67,    68,
       1,    -1,    71,    72,    73,    74,    75,     1,    -1,    -1,
      -1,    -1,    -1,    -1,     1,    -1,    -1,     1,    87,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      14,    15,    16,    -1,    18,    19,    -1,    -1,    22,    -1,
      -1,    -1,    43,    44,    45,    46,    47,    48,    49,    43,
      44,    45,    46,    47,    48,    49,    43,    44,    45,    46,
      47,    48,    49,     1,    -1,    -1,    -1,    -1,    -1,    53,
      54,     1,    37,    38,    75,    59,    60,    78,     1,    -1,
      -1,    75,    -1,    -1,    78,     1,    -1,    71,    75,    -1,
      55,    78,    -1,    -1,    78,    79,    61,    62,    63,    64,
      65,    66,    67,    -1,    -1,    43,    44,    45,    46,    47,
      48,    49,    -1,    43,    44,    45,    46,    47,    48,    49,
      43,    44,    45,    46,    47,    48,    49,    43,    44,    45,
      46,    47,    48,    49,     1,    -1,    -1,    75,    -1,    -1,
      78,    -1,    -1,    -1,    -1,    75,    -1,    -1,    78,    -1,
      -1,    -1,    75,    -1,    -1,    78,    -1,    -1,    -1,    75,
      -1,    -1,    78,    -1,     6,     7,     8,     9,    -1,    11,
      12,    13,    14,    15,    -1,    -1,    43,    44,    45,    46,
      47,    48,    49,     3,     4,    -1,     6,     7,     8,     9,
      10,    11,    12,    13,    14,    15,    16,     1,    18,    19,
      42,    -1,    22,    -1,    -1,    -1,    -1,    -1,    75,    -1,
      -1,    78,    -1,    -1,    -1,    -1,    -1,    21,    -1,    23,
      24,    25,    26,    27,    28,    -1,    -1,    31,    32,    33,
      34,    35,    36,    -1,    76,    -1,    -1,    79,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    51,    52,    -1,
      -1,     1,    -1,    -1,    -1,    -1,    -1,    -1,    78,    -1,
      -1,    -1,    -1,    -1,    68,    -1,    -1,    -1,    72,    73,
      74,    21,    -1,    23,    24,    25,    26,    27,    28,    -1,
      -1,    31,    32,    33,    34,    35,    36,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    51,    52,    -1,    -1,    -1,    -1,    -1,    -1,     3,
       4,     5,    -1,     7,     8,     9,    -1,    11,    12,    13,
      14,    15,    72,    73,    74,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,    29,    30,    -1,    32,    33,
      34,    35,    36,    37,    38,    39,    40,    41,    42,    43,
      44,    45,    46,    47,    48,    49,    50,    -1,    52,    -1,
      54,    55,    -1,    -1,    -1,    -1,    -1,    61,    62,    63,
      64,    65,    66,    67,    68,    -1,    -1,    71,    72,    73,
      74,    75,     3,     4,     5,    -1,     7,     8,     9,    -1,
      11,    12,    13,    14,    15,    -1,    -1,    -1,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      -1,    32,    33,    34,    35,    36,    -1,    -1,    39,    40,
      41,    42,    43,    44,    45,    46,    47,    48,    49,    50,
      51,    52,    -1,    54,    55,    56,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    68,    -1,    -1,
      71,    72,    73,    74,    75,     3,     4,     5,    -1,     7,
       8,     9,    -1,    11,    12,    13,    14,    15,    -1,    -1,
      -1,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      28,    29,    30,    -1,    32,    33,    34,    35,    36,    -1,
      -1,    39,    40,    41,    42,    43,    44,    45,    46,    47,
      48,    49,    50,    -1,    52,    -1,    54,    55,    56,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      68,    -1,    -1,    71,    72,    73,    74,    75,     3,     4,
       5,    -1,     7,     8,     9,    -1,    11,    12,    13,    14,
      15,    -1,    -1,    -1,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    -1,    32,    33,    34,
      35,    36,    -1,    -1,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    -1,    52,    -1,    54,
      55,    56,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    68,    -1,    -1,    71,    72,    73,    74,
      75,     3,     4,     5,    -1,     7,     8,     9,    -1,    11,
      12,    13,    14,    15,    -1,    -1,    -1,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    -1,
      32,    33,    34,    35,    36,    16,    -1,    39,    40,    41,
      42,    43,    44,    45,    46,    47,    48,    49,    50,    -1,
      52,    -1,    54,    55,    -1,    -1,    37,    38,    39,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    68,    -1,    18,    71,
      72,    73,    74,    75,    55,    -1,    -1,    -1,    -1,    -1,
      61,    62,    63,    64,    65,    66,    67,    37,    38,    39,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      37,    38,    39,    -1,    -1,    55,    -1,    -1,    -1,    -1,
      -1,    61,    62,    63,    64,    65,    66,    67,    55,    -1,
      -1,    -1,    -1,    -1,    61,    62,    63,    64,    65,    66,
      67,    37,    38,    39,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    37,    38,    39,    -1,    85,    55,
      56,    57,    -1,    -1,    -1,    61,    62,    63,    64,    65,
      66,    67,    55,    -1,    -1,    -1,    -1,    -1,    61,    62,
      63,    64,    65,    66,    67
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     3,     4,    93,    94,   100,     3,     4,     5,     7,
       8,     9,    11,    12,    13,    14,    15,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    32,
      33,    34,    35,    36,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    52,    54,    55,    68,
      71,    72,    73,    74,    75,    98,    98,     0,    94,    76,
      78,    96,   101,   101,     1,     5,    53,    54,    59,    60,
      71,    95,   102,   103,   104,   170,   207,   208,    76,    42,
      98,    53,    55,    99,    98,    98,    78,    96,   179,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      18,    19,    22,    78,   100,   123,   124,   140,   143,   144,
     145,   147,   157,   158,   161,   162,   163,    79,     6,     7,
       8,     9,    11,    12,    13,    14,    15,    42,    79,    96,
     168,   102,    37,    38,    61,    62,    63,    64,    65,    66,
      67,    99,   109,   111,   112,   113,   114,   117,   118,   171,
      78,    99,    77,    58,    84,   177,    16,    38,    39,   112,
     113,   114,   115,   116,   119,    37,   125,   125,    87,   125,
     111,   164,    87,   129,   129,   129,   129,    87,   133,   146,
      87,   126,    98,    57,   165,    81,   102,    11,    12,    13,
      14,    15,    16,   148,   149,   150,   151,   152,    96,    97,
     117,    62,    66,    61,    62,    63,    64,    81,   108,    83,
      83,    83,    38,    86,   111,   102,    77,    78,    84,    87,
     177,    79,   180,   112,   116,   112,    38,    84,    86,    99,
      99,    99,    68,    99,    80,    30,    51,   130,   135,    98,
     110,   110,   110,   110,    51,    56,    98,   132,   134,    87,
     146,    87,   133,    40,    41,   127,   128,   110,    18,   115,
     119,   155,   156,    79,   129,   129,   129,   129,   146,   126,
      61,    62,    56,    57,   105,   106,   107,   119,    98,    79,
      55,   177,    58,   176,   177,    85,    96,    83,    83,    87,
     121,   122,   205,    84,    81,    84,    88,    81,    84,   164,
      88,    80,   108,    77,   141,   141,   141,   141,    98,    88,
      80,    88,   110,   110,    88,    80,    78,    98,    89,   154,
      98,    80,    82,    97,    98,    98,    98,    98,    98,    98,
      80,    82,   108,    79,    80,    85,    88,   177,    99,    98,
     122,   120,   177,   125,   106,   125,   125,   106,   125,   130,
     111,   142,    78,    96,   159,   159,   159,   159,    88,   134,
     141,   141,   127,    17,   136,   138,   139,    89,   153,    56,
      57,   111,   154,   156,   141,   141,   141,   141,   141,    78,
      96,   106,    84,   188,   177,   176,   177,   177,   122,    85,
      88,   206,    85,    82,    85,    99,    82,    85,    80,     1,
      42,   157,   160,   161,   166,   167,   169,   159,   159,   119,
     139,    79,   119,   159,   159,   159,   159,   159,   139,    85,
     119,   178,   181,   186,    84,    84,    84,    84,   142,     1,
      87,   172,   169,    79,    96,   160,    98,    79,    98,   177,
      80,    85,   186,   125,   125,   125,     1,    21,    23,    24,
      25,    26,    27,    28,    31,    32,    33,    34,    35,    36,
      51,    52,    68,    72,    73,    74,   173,   174,    55,    98,
     171,    97,    87,   137,    96,    87,    89,   136,    88,   186,
      85,    85,    85,    85,    57,   131,    88,    88,    80,   188,
      98,    88,    96,    56,    57,    99,   175,    37,   173,     1,
      43,    44,    45,    46,    47,    48,    49,    75,    78,    96,
     179,   191,   196,   198,   188,    98,    84,   202,    87,   202,
      55,   203,   204,    78,    57,   195,   202,    78,   192,   198,
     177,    20,   190,   188,   177,   200,    55,   200,   188,   205,
      80,    78,   198,   193,   198,   179,   200,    43,    44,    45,
      47,    48,    49,    75,   179,   194,   196,   197,    79,   192,
     180,    89,   191,    87,   189,    76,    88,    85,   201,   177,
     204,    79,   192,    79,   192,   177,   201,   202,    87,   202,
      78,   195,   202,    78,   177,    79,   194,    97,    97,    56,
       6,    69,    70,    88,   119,   182,   185,   187,   179,   200,
     202,    78,   198,   206,    79,   180,    78,   198,   177,    55,
     177,   193,   179,   177,   194,   180,    98,    77,    80,    88,
     177,    76,   200,   192,   188,    97,   192,    50,   199,    76,
      88,   201,    79,   177,   201,    79,    97,    81,   119,   178,
     184,   187,   180,   200,    77,    79,    79,    78,   198,   177,
     202,    78,   198,   180,    78,   198,    98,   183,    98,   177,
      81,    98,   201,   200,   199,   192,    76,   177,   192,    97,
     192,   199,    82,    84,    87,    90,    91,    81,    88,   183,
      96,    78,   198,    80,    79,   177,    77,    79,    79,   183,
      56,   183,    82,    98,   183,    82,   192,   200,   201,   177,
     199,    85,    88,    88,    98,    82,    79,   201,    78,   198,
      80,    78,   198,   192,   177,   192,    79,   201,    79,    78,
     198,   192,    79
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
		  Type, Value, Location); \
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
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep, YYLTYPE const * const yylocationp)
#else
static void
yy_symbol_value_print (yyoutput, yytype, yyvaluep, yylocationp)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
    YYLTYPE const * const yylocationp;
#endif
{
  if (!yyvaluep)
    return;
  YYUSE (yylocationp);
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
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep, YYLTYPE const * const yylocationp)
#else
static void
yy_symbol_print (yyoutput, yytype, yyvaluep, yylocationp)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
    YYLTYPE const * const yylocationp;
#endif
{
  if (yytype < YYNTOKENS)
    YYFPRINTF (yyoutput, "token %s (", yytname[yytype]);
  else
    YYFPRINTF (yyoutput, "nterm %s (", yytname[yytype]);

  YY_LOCATION_PRINT (yyoutput, *yylocationp);
  YYFPRINTF (yyoutput, ": ");
  yy_symbol_value_print (yyoutput, yytype, yyvaluep, yylocationp);
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
yy_reduce_print (YYSTYPE *yyvsp, YYLTYPE *yylsp, int yyrule)
#else
static void
yy_reduce_print (yyvsp, yylsp, yyrule)
    YYSTYPE *yyvsp;
    YYLTYPE *yylsp;
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
		       , &(yylsp[(yyi + 1) - (yynrhs)])		       );
      fprintf (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)		\
do {					\
  if (yydebug)				\
    yy_reduce_print (yyvsp, yylsp, Rule); \
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
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep, YYLTYPE *yylocationp)
#else
static void
yydestruct (yymsg, yytype, yyvaluep, yylocationp)
    const char *yymsg;
    int yytype;
    YYSTYPE *yyvaluep;
    YYLTYPE *yylocationp;
#endif
{
  YYUSE (yyvaluep);
  YYUSE (yylocationp);

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
/* Location data for the look-ahead symbol.  */
YYLTYPE yylloc;



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

  /* The location stack.  */
  YYLTYPE yylsa[YYINITDEPTH];
  YYLTYPE *yyls = yylsa;
  YYLTYPE *yylsp;
  /* The locations where the error started and ended.  */
  YYLTYPE yyerror_range[2];

#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N), yylsp -= (N))

  YYSIZE_T yystacksize = YYINITDEPTH;

  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;
  YYLTYPE yyloc;

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
  yylsp = yyls;
#if defined YYLTYPE_IS_TRIVIAL && YYLTYPE_IS_TRIVIAL
  /* Initialize the default location before parsing starts.  */
  yylloc.first_line   = yylloc.last_line   = 1;
  yylloc.first_column = yylloc.last_column = 0;
#endif

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
	YYLTYPE *yyls1 = yyls;

	/* Each stack pointer address is followed by the size of the
	   data in use in that stack, in bytes.  This used to be a
	   conditional around just the two extra args, but that might
	   be undefined if yyoverflow is a macro.  */
	yyoverflow (YY_("memory exhausted"),
		    &yyss1, yysize * sizeof (*yyssp),
		    &yyvs1, yysize * sizeof (*yyvsp),
		    &yyls1, yysize * sizeof (*yylsp),
		    &yystacksize);
	yyls = yyls1;
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
	YYSTACK_RELOCATE (yyls);
#  undef YYSTACK_RELOCATE
	if (yyss1 != yyssa)
	  YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;
      yylsp = yyls + yysize - 1;

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
  *++yylsp = yylloc;
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

  /* Default location.  */
  YYLLOC_DEFAULT (yyloc, (yylsp - yylen), yylen);
  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 2:
#line 195 "xi-grammar.y"
    { (yyval.modlist) = (yyvsp[(1) - (1)].modlist); modlist = (yyvsp[(1) - (1)].modlist); }
    break;

  case 3:
#line 199 "xi-grammar.y"
    { 
		  (yyval.modlist) = 0; 
		}
    break;

  case 4:
#line 203 "xi-grammar.y"
    { (yyval.modlist) = new AstChildren<Module>(lineno, (yyvsp[(1) - (2)].module), (yyvsp[(2) - (2)].modlist)); }
    break;

  case 5:
#line 207 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 6:
#line 209 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 7:
#line 213 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 8:
#line 215 "xi-grammar.y"
    { (yyval.intval) = 2; }
    break;

  case 9:
#line 219 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 10:
#line 221 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 11:
#line 226 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 12:
#line 227 "xi-grammar.y"
    { ReservedWord(MODULE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 13:
#line 228 "xi-grammar.y"
    { ReservedWord(MAINMODULE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 14:
#line 229 "xi-grammar.y"
    { ReservedWord(EXTERN, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 15:
#line 231 "xi-grammar.y"
    { ReservedWord(INITCALL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 16:
#line 232 "xi-grammar.y"
    { ReservedWord(INITNODE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 17:
#line 233 "xi-grammar.y"
    { ReservedWord(INITPROC, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 18:
#line 235 "xi-grammar.y"
    { ReservedWord(CHARE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 19:
#line 236 "xi-grammar.y"
    { ReservedWord(MAINCHARE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 20:
#line 237 "xi-grammar.y"
    { ReservedWord(GROUP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 21:
#line 238 "xi-grammar.y"
    { ReservedWord(NODEGROUP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 22:
#line 239 "xi-grammar.y"
    { ReservedWord(ARRAY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 23:
#line 243 "xi-grammar.y"
    { ReservedWord(INCLUDE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 24:
#line 244 "xi-grammar.y"
    { ReservedWord(STACKSIZE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 25:
#line 245 "xi-grammar.y"
    { ReservedWord(THREADED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 26:
#line 246 "xi-grammar.y"
    { ReservedWord(TEMPLATE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 27:
#line 247 "xi-grammar.y"
    { ReservedWord(SYNC, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 28:
#line 248 "xi-grammar.y"
    { ReservedWord(IGET, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 29:
#line 249 "xi-grammar.y"
    { ReservedWord(EXCLUSIVE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 30:
#line 250 "xi-grammar.y"
    { ReservedWord(IMMEDIATE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 31:
#line 251 "xi-grammar.y"
    { ReservedWord(SKIPSCHED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 32:
#line 252 "xi-grammar.y"
    { ReservedWord(SCATTER, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 33:
#line 253 "xi-grammar.y"
    { ReservedWord(INLINE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 34:
#line 254 "xi-grammar.y"
    { ReservedWord(VIRTUAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 35:
#line 255 "xi-grammar.y"
    { ReservedWord(MIGRATABLE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 36:
#line 256 "xi-grammar.y"
    { ReservedWord(CREATEHERE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 37:
#line 257 "xi-grammar.y"
    { ReservedWord(CREATEHOME, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 38:
#line 258 "xi-grammar.y"
    { ReservedWord(NOKEEP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 39:
#line 259 "xi-grammar.y"
    { ReservedWord(NOTRACE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 40:
#line 260 "xi-grammar.y"
    { ReservedWord(APPWORK, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 41:
#line 263 "xi-grammar.y"
    { ReservedWord(PACKED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 42:
#line 264 "xi-grammar.y"
    { ReservedWord(VARSIZE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 43:
#line 265 "xi-grammar.y"
    { ReservedWord(ENTRY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 44:
#line 266 "xi-grammar.y"
    { ReservedWord(FOR, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 45:
#line 267 "xi-grammar.y"
    { ReservedWord(FORALL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 46:
#line 268 "xi-grammar.y"
    { ReservedWord(WHILE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 47:
#line 269 "xi-grammar.y"
    { ReservedWord(WHEN, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 48:
#line 270 "xi-grammar.y"
    { ReservedWord(OVERLAP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 49:
#line 271 "xi-grammar.y"
    { ReservedWord(ATOMIC, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 50:
#line 272 "xi-grammar.y"
    { ReservedWord(IF, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 51:
#line 273 "xi-grammar.y"
    { ReservedWord(ELSE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 52:
#line 275 "xi-grammar.y"
    { ReservedWord(LOCAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 53:
#line 277 "xi-grammar.y"
    { ReservedWord(USING, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 54:
#line 278 "xi-grammar.y"
    { ReservedWord(ACCEL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 55:
#line 281 "xi-grammar.y"
    { ReservedWord(ACCELBLOCK, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 56:
#line 282 "xi-grammar.y"
    { ReservedWord(MEMCRITICAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 57:
#line 283 "xi-grammar.y"
    { ReservedWord(REDUCTIONTARGET, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 58:
#line 284 "xi-grammar.y"
    { ReservedWord(SCATTERV, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 59:
#line 285 "xi-grammar.y"
    { ReservedWord(CASE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 60:
#line 289 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 61:
#line 291 "xi-grammar.y"
    {
		  char *tmp = new char[strlen((yyvsp[(1) - (4)].strval))+strlen((yyvsp[(4) - (4)].strval))+3];
		  sprintf(tmp,"%s::%s", (yyvsp[(1) - (4)].strval), (yyvsp[(4) - (4)].strval));
		  (yyval.strval) = tmp;
		}
    break;

  case 62:
#line 298 "xi-grammar.y"
    { 
		    (yyval.module) = new Module(lineno, (yyvsp[(2) - (3)].strval), (yyvsp[(3) - (3)].conslist)); 
		}
    break;

  case 63:
#line 302 "xi-grammar.y"
    {  
		    (yyval.module) = new Module(lineno, (yyvsp[(2) - (3)].strval), (yyvsp[(3) - (3)].conslist)); 
		    (yyval.module)->setMain();
		}
    break;

  case 64:
#line 309 "xi-grammar.y"
    { (yyval.conslist) = 0; }
    break;

  case 65:
#line 311 "xi-grammar.y"
    { (yyval.conslist) = (yyvsp[(2) - (4)].conslist); }
    break;

  case 66:
#line 315 "xi-grammar.y"
    { (yyval.conslist) = 0; }
    break;

  case 67:
#line 317 "xi-grammar.y"
    { (yyval.conslist) = new ConstructList(lineno, (yyvsp[(1) - (2)].construct), (yyvsp[(2) - (2)].conslist)); }
    break;

  case 68:
#line 321 "xi-grammar.y"
    { (yyval.construct) = new UsingScope((yyvsp[(3) - (3)].strval), false); }
    break;

  case 69:
#line 323 "xi-grammar.y"
    { (yyval.construct) = new UsingScope((yyvsp[(2) - (2)].strval), true); }
    break;

  case 70:
#line 325 "xi-grammar.y"
    { (yyvsp[(2) - (2)].member)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].member); }
    break;

  case 71:
#line 327 "xi-grammar.y"
    { (yyvsp[(2) - (2)].message)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].message); }
    break;

  case 72:
#line 329 "xi-grammar.y"
    {
                  Entry *e = new Entry(lineno, 0, (yyvsp[(3) - (7)].type), (yyvsp[(5) - (7)].strval), (yyvsp[(7) - (7)].plist), 0, 0, 0, (yylsp[(1) - (7)]).first_line, (yyloc).last_line);
                  int isExtern = 1;
                  e->setExtern(isExtern);
                  e->targs = (yyvsp[(6) - (7)].tparlist);
                  e->label = new XStr;
                  (yyvsp[(4) - (7)].ntype)->print(*e->label);
                  (yyval.construct) = e;
                }
    break;

  case 73:
#line 341 "xi-grammar.y"
    { if((yyvsp[(3) - (5)].conslist)) (yyvsp[(3) - (5)].conslist)->recurse<int&>((yyvsp[(1) - (5)].intval), &Construct::setExtern); (yyval.construct) = (yyvsp[(3) - (5)].conslist); }
    break;

  case 74:
#line 343 "xi-grammar.y"
    { (yyval.construct) = new Scope((yyvsp[(2) - (5)].strval), (yyvsp[(4) - (5)].conslist)); }
    break;

  case 75:
#line 345 "xi-grammar.y"
    { (yyval.construct) = (yyvsp[(1) - (2)].construct); }
    break;

  case 76:
#line 347 "xi-grammar.y"
    {
          ERROR("preceding construct must be semicolon terminated",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
    break;

  case 77:
#line 353 "xi-grammar.y"
    { (yyvsp[(2) - (2)].module)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].module); }
    break;

  case 78:
#line 355 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 79:
#line 357 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 80:
#line 359 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 81:
#line 361 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 82:
#line 363 "xi-grammar.y"
    { (yyvsp[(2) - (2)].templat)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].templat); }
    break;

  case 83:
#line 365 "xi-grammar.y"
    { (yyval.construct) = NULL; }
    break;

  case 84:
#line 367 "xi-grammar.y"
    { (yyval.construct) = NULL; }
    break;

  case 85:
#line 369 "xi-grammar.y"
    { (yyval.construct) = (yyvsp[(1) - (1)].accelBlock); }
    break;

  case 86:
#line 371 "xi-grammar.y"
    {
          ERROR("invalid construct",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
    break;

  case 87:
#line 379 "xi-grammar.y"
    { (yyval.tparam) = new TParamType((yyvsp[(1) - (1)].type)); }
    break;

  case 88:
#line 381 "xi-grammar.y"
    { (yyval.tparam) = new TParamVal((yyvsp[(1) - (1)].strval)); }
    break;

  case 89:
#line 383 "xi-grammar.y"
    { (yyval.tparam) = new TParamVal((yyvsp[(1) - (1)].strval)); }
    break;

  case 90:
#line 387 "xi-grammar.y"
    { (yyval.tparlist) = new TParamList((yyvsp[(1) - (1)].tparam)); }
    break;

  case 91:
#line 389 "xi-grammar.y"
    { (yyval.tparlist) = new TParamList((yyvsp[(1) - (3)].tparam), (yyvsp[(3) - (3)].tparlist)); }
    break;

  case 92:
#line 393 "xi-grammar.y"
    { (yyval.tparlist) = 0; }
    break;

  case 93:
#line 395 "xi-grammar.y"
    { (yyval.tparlist) = (yyvsp[(1) - (1)].tparlist); }
    break;

  case 94:
#line 399 "xi-grammar.y"
    { (yyval.tparlist) = 0; }
    break;

  case 95:
#line 401 "xi-grammar.y"
    { (yyval.tparlist) = (yyvsp[(2) - (3)].tparlist); }
    break;

  case 96:
#line 405 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("int"); }
    break;

  case 97:
#line 407 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long"); }
    break;

  case 98:
#line 409 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("short"); }
    break;

  case 99:
#line 411 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("char"); }
    break;

  case 100:
#line 413 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned int"); }
    break;

  case 101:
#line 415 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long"); }
    break;

  case 102:
#line 417 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long"); }
    break;

  case 103:
#line 419 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long long"); }
    break;

  case 104:
#line 421 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned short"); }
    break;

  case 105:
#line 423 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned char"); }
    break;

  case 106:
#line 425 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long long"); }
    break;

  case 107:
#line 427 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("float"); }
    break;

  case 108:
#line 429 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("double"); }
    break;

  case 109:
#line 431 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long double"); }
    break;

  case 110:
#line 433 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("void"); }
    break;

  case 111:
#line 436 "xi-grammar.y"
    { (yyval.ntype) = new NamedType((yyvsp[(1) - (2)].strval),(yyvsp[(2) - (2)].tparlist)); }
    break;

  case 112:
#line 437 "xi-grammar.y"
    { 
                    const char* basename, *scope;
                    splitScopedName((yyvsp[(1) - (2)].strval), &scope, &basename);
                    (yyval.ntype) = new NamedType(basename, (yyvsp[(2) - (2)].tparlist), scope);
                }
    break;

  case 113:
#line 445 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 114:
#line 447 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ntype); }
    break;

  case 115:
#line 451 "xi-grammar.y"
    { (yyval.ptype) = new PtrType((yyvsp[(1) - (2)].type)); }
    break;

  case 116:
#line 455 "xi-grammar.y"
    { (yyvsp[(1) - (2)].ptype)->indirect(); (yyval.ptype) = (yyvsp[(1) - (2)].ptype); }
    break;

  case 117:
#line 457 "xi-grammar.y"
    { (yyvsp[(1) - (2)].ptype)->indirect(); (yyval.ptype) = (yyvsp[(1) - (2)].ptype); }
    break;

  case 118:
#line 461 "xi-grammar.y"
    { (yyval.ftype) = new FuncType((yyvsp[(1) - (8)].type), (yyvsp[(4) - (8)].strval), (yyvsp[(7) - (8)].plist)); }
    break;

  case 119:
#line 465 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 120:
#line 467 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 121:
#line 469 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 122:
#line 471 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ftype); }
    break;

  case 123:
#line 473 "xi-grammar.y"
    { (yyval.type) = new ScatterType((yyvsp[(2) - (2)].type)); }
    break;

  case 124:
#line 475 "xi-grammar.y"
    { (yyval.type) = new ConstType((yyvsp[(2) - (2)].type)); }
    break;

  case 125:
#line 477 "xi-grammar.y"
    { (yyval.type) = new ConstType((yyvsp[(1) - (2)].type)); }
    break;

  case 126:
#line 481 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 127:
#line 483 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 128:
#line 485 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 129:
#line 487 "xi-grammar.y"
    { (yyval.type) = new ConstType((yyvsp[(2) - (2)].type)); }
    break;

  case 130:
#line 489 "xi-grammar.y"
    { (yyval.type) = new ConstType((yyvsp[(1) - (2)].type)); }
    break;

  case 131:
#line 493 "xi-grammar.y"
    { (yyval.type) = new ReferenceType((yyvsp[(1) - (2)].type)); }
    break;

  case 132:
#line 495 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 133:
#line 499 "xi-grammar.y"
    { (yyval.type) = new ReferenceType((yyvsp[(1) - (2)].type)); }
    break;

  case 134:
#line 501 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 135:
#line 505 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 136:
#line 509 "xi-grammar.y"
    { (yyval.val) = (yyvsp[(2) - (3)].val); }
    break;

  case 137:
#line 513 "xi-grammar.y"
    { (yyval.vallist) = 0; }
    break;

  case 138:
#line 515 "xi-grammar.y"
    { (yyval.vallist) = new ValueList((yyvsp[(1) - (2)].val), (yyvsp[(2) - (2)].vallist)); }
    break;

  case 139:
#line 519 "xi-grammar.y"
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[(2) - (4)].type), (yyvsp[(3) - (4)].strval), (yyvsp[(4) - (4)].vallist)); }
    break;

  case 140:
#line 523 "xi-grammar.y"
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[(3) - (6)].type), (yyvsp[(5) - (6)].strval), (yyvsp[(6) - (6)].vallist), 1); }
    break;

  case 141:
#line 527 "xi-grammar.y"
    { (yyval.intval) = 0;}
    break;

  case 142:
#line 529 "xi-grammar.y"
    { (yyval.intval) = 0;}
    break;

  case 143:
#line 533 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 144:
#line 535 "xi-grammar.y"
    { 
		  /*
		  printf("Warning: Message attributes are being phased out.\n");
		  printf("Warning: Please remove them from interface files.\n");
		  */
		  (yyval.intval) = (yyvsp[(2) - (3)].intval); 
		}
    break;

  case 145:
#line 545 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (1)].intval); }
    break;

  case 146:
#line 547 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (3)].intval) | (yyvsp[(3) - (3)].intval); }
    break;

  case 147:
#line 551 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 148:
#line 553 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 149:
#line 557 "xi-grammar.y"
    { (yyval.cattr) = 0; }
    break;

  case 150:
#line 559 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(2) - (3)].cattr); }
    break;

  case 151:
#line 563 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (1)].cattr); }
    break;

  case 152:
#line 565 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (3)].cattr) | (yyvsp[(3) - (3)].cattr); }
    break;

  case 153:
#line 569 "xi-grammar.y"
    { python_doc = NULL; (yyval.intval) = 0; }
    break;

  case 154:
#line 571 "xi-grammar.y"
    { python_doc = (yyvsp[(1) - (1)].strval); (yyval.intval) = 0; }
    break;

  case 155:
#line 575 "xi-grammar.y"
    { (yyval.cattr) = Chare::CPYTHON; }
    break;

  case 156:
#line 579 "xi-grammar.y"
    { (yyval.cattr) = 0; }
    break;

  case 157:
#line 581 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(2) - (3)].cattr); }
    break;

  case 158:
#line 585 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (1)].cattr); }
    break;

  case 159:
#line 587 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (3)].cattr) | (yyvsp[(3) - (3)].cattr); }
    break;

  case 160:
#line 591 "xi-grammar.y"
    { (yyval.cattr) = Chare::CMIGRATABLE; }
    break;

  case 161:
#line 593 "xi-grammar.y"
    { (yyval.cattr) = Chare::CPYTHON; }
    break;

  case 162:
#line 597 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 163:
#line 599 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 164:
#line 602 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 165:
#line 604 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 166:
#line 607 "xi-grammar.y"
    { (yyval.mv) = new MsgVar((yyvsp[(2) - (5)].type), (yyvsp[(3) - (5)].strval), (yyvsp[(1) - (5)].intval), (yyvsp[(4) - (5)].intval)); }
    break;

  case 167:
#line 611 "xi-grammar.y"
    { (yyval.mvlist) = new MsgVarList((yyvsp[(1) - (1)].mv)); }
    break;

  case 168:
#line 613 "xi-grammar.y"
    { (yyval.mvlist) = new MsgVarList((yyvsp[(1) - (2)].mv), (yyvsp[(2) - (2)].mvlist)); }
    break;

  case 169:
#line 617 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, (yyvsp[(3) - (3)].ntype)); }
    break;

  case 170:
#line 619 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, (yyvsp[(3) - (6)].ntype), (yyvsp[(5) - (6)].mvlist)); }
    break;

  case 171:
#line 623 "xi-grammar.y"
    { (yyval.typelist) = 0; }
    break;

  case 172:
#line 625 "xi-grammar.y"
    { (yyval.typelist) = (yyvsp[(2) - (2)].typelist); }
    break;

  case 173:
#line 629 "xi-grammar.y"
    { (yyval.typelist) = new TypeList((yyvsp[(1) - (1)].ntype)); }
    break;

  case 174:
#line 631 "xi-grammar.y"
    { (yyval.typelist) = new TypeList((yyvsp[(1) - (3)].ntype), (yyvsp[(3) - (3)].typelist)); }
    break;

  case 175:
#line 635 "xi-grammar.y"
    { (yyval.chare) = new Chare(lineno, (yyvsp[(2) - (5)].cattr)|Chare::CCHARE, (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 176:
#line 637 "xi-grammar.y"
    { (yyval.chare) = new MainChare(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 177:
#line 641 "xi-grammar.y"
    { (yyval.chare) = new Group(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 178:
#line 645 "xi-grammar.y"
    { (yyval.chare) = new NodeGroup(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 179:
#line 649 "xi-grammar.y"
    {/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",(yyvsp[(2) - (4)].strval));
			(yyval.ntype) = new NamedType(buf); 
		}
    break;

  case 180:
#line 655 "xi-grammar.y"
    { (yyval.ntype) = new NamedType((yyvsp[(2) - (3)].strval)); }
    break;

  case 181:
#line 659 "xi-grammar.y"
    {  (yyval.chare) = new Array(lineno, (yyvsp[(2) - (6)].cattr), (yyvsp[(3) - (6)].ntype), (yyvsp[(4) - (6)].ntype), (yyvsp[(5) - (6)].typelist), (yyvsp[(6) - (6)].mbrlist)); }
    break;

  case 182:
#line 661 "xi-grammar.y"
    {  (yyval.chare) = new Array(lineno, (yyvsp[(3) - (6)].cattr), (yyvsp[(2) - (6)].ntype), (yyvsp[(4) - (6)].ntype), (yyvsp[(5) - (6)].typelist), (yyvsp[(6) - (6)].mbrlist)); }
    break;

  case 183:
#line 665 "xi-grammar.y"
    { (yyval.chare) = new Chare(lineno, (yyvsp[(2) - (5)].cattr)|Chare::CCHARE, new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist));}
    break;

  case 184:
#line 667 "xi-grammar.y"
    { (yyval.chare) = new MainChare(lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 185:
#line 671 "xi-grammar.y"
    { (yyval.chare) = new Group(lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 186:
#line 675 "xi-grammar.y"
    { (yyval.chare) = new NodeGroup( lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 187:
#line 679 "xi-grammar.y"
    { (yyval.chare) = new Array( lineno, 0, (yyvsp[(2) - (5)].ntype), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 188:
#line 683 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[(3) - (4)].strval))); }
    break;

  case 189:
#line 685 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[(3) - (7)].strval)), (yyvsp[(5) - (7)].mvlist)); }
    break;

  case 190:
#line 689 "xi-grammar.y"
    { (yyval.type) = 0; }
    break;

  case 191:
#line 691 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(2) - (2)].type); }
    break;

  case 192:
#line 695 "xi-grammar.y"
    { (yyval.strval) = 0; }
    break;

  case 193:
#line 697 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(2) - (2)].strval); }
    break;

  case 194:
#line 699 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(2) - (2)].strval); }
    break;

  case 195:
#line 701 "xi-grammar.y"
    {
		  XStr typeStr;
		  (yyvsp[(2) - (2)].ntype)->print(typeStr);
		  char *tmp = strdup(typeStr.get_string());
		  (yyval.strval) = tmp;
		}
    break;

  case 196:
#line 710 "xi-grammar.y"
    { (yyval.tvar) = new TType(new NamedType((yyvsp[(2) - (3)].strval)), (yyvsp[(3) - (3)].type)); }
    break;

  case 197:
#line 712 "xi-grammar.y"
    { (yyval.tvar) = new TFunc((yyvsp[(1) - (2)].ftype), (yyvsp[(2) - (2)].strval)); }
    break;

  case 198:
#line 714 "xi-grammar.y"
    { (yyval.tvar) = new TName((yyvsp[(1) - (3)].type), (yyvsp[(2) - (3)].strval), (yyvsp[(3) - (3)].strval)); }
    break;

  case 199:
#line 718 "xi-grammar.y"
    { (yyval.tvarlist) = new TVarList((yyvsp[(1) - (1)].tvar)); }
    break;

  case 200:
#line 720 "xi-grammar.y"
    { (yyval.tvarlist) = new TVarList((yyvsp[(1) - (3)].tvar), (yyvsp[(3) - (3)].tvarlist)); }
    break;

  case 201:
#line 724 "xi-grammar.y"
    { (yyval.tvarlist) = (yyvsp[(3) - (4)].tvarlist); }
    break;

  case 202:
#line 728 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 203:
#line 730 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 204:
#line 732 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 205:
#line 734 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 206:
#line 736 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].message)); (yyvsp[(2) - (2)].message)->setTemplate((yyval.templat)); }
    break;

  case 207:
#line 740 "xi-grammar.y"
    { (yyval.mbrlist) = 0; }
    break;

  case 208:
#line 742 "xi-grammar.y"
    { (yyval.mbrlist) = (yyvsp[(2) - (4)].mbrlist); }
    break;

  case 209:
#line 746 "xi-grammar.y"
    { 
                  if (!connectEntries.empty()) {
                    (yyval.mbrlist) = new AstChildren<Member>(connectEntries);
		  } else {
		    (yyval.mbrlist) = 0; 
                  }
		}
    break;

  case 210:
#line 754 "xi-grammar.y"
    { (yyval.mbrlist) = new AstChildren<Member>(-1, (yyvsp[(1) - (2)].member), (yyvsp[(2) - (2)].mbrlist)); }
    break;

  case 211:
#line 758 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].readonly); }
    break;

  case 212:
#line 760 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].readonly); }
    break;

  case 214:
#line 763 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].member); }
    break;

  case 215:
#line 765 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(2) - (2)].pupable); }
    break;

  case 216:
#line 767 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(2) - (2)].includeFile); }
    break;

  case 217:
#line 769 "xi-grammar.y"
    { (yyval.member) = new ClassDeclaration(lineno,(yyvsp[(2) - (2)].strval)); }
    break;

  case 218:
#line 773 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 1); }
    break;

  case 219:
#line 775 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 1); }
    break;

  case 220:
#line 777 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[(3) - (9)].strval)) + '<' +
					    ((yyvsp[(5) - (9)].tparlist))->to_string() + '>').c_str()),
				    1);
		}
    break;

  case 221:
#line 783 "xi-grammar.y"
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[(1) - (3)]).first_column, (yylsp[(1) - (3)]).last_column, (yylsp[(1) - (3)]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 1);
		}
    break;

  case 222:
#line 789 "xi-grammar.y"
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[(1) - (6)]).first_column, (yylsp[(1) - (6)]).last_column, (yylsp[(1) - (6)]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 1);
		}
    break;

  case 223:
#line 798 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 0); }
    break;

  case 224:
#line 800 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 0); }
    break;

  case 225:
#line 802 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[(3) - (9)].strval)) + '<' +
					    ((yyvsp[(5) - (9)].tparlist))->to_string() + '>').c_str()),
				    0);
		}
    break;

  case 226:
#line 808 "xi-grammar.y"
    {
                  InitCall* rtn = new InitCall(lineno, (yyvsp[(6) - (9)].strval), 0);
                  rtn->setAccel();
                  (yyval.member) = rtn;
		}
    break;

  case 227:
#line 816 "xi-grammar.y"
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[(1) - (1)].ntype),0); }
    break;

  case 228:
#line 818 "xi-grammar.y"
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[(1) - (3)].ntype),(yyvsp[(3) - (3)].pupable)); }
    break;

  case 229:
#line 821 "xi-grammar.y"
    { (yyval.includeFile) = new IncludeFile(lineno,(yyvsp[(1) - (1)].strval)); }
    break;

  case 230:
#line 825 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].member); }
    break;

  case 231:
#line 829 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].entry); }
    break;

  case 232:
#line 831 "xi-grammar.y"
    {
                  (yyvsp[(2) - (2)].entry)->tspec = (yyvsp[(1) - (2)].tvarlist);
                  (yyval.member) = (yyvsp[(2) - (2)].entry);
                }
    break;

  case 233:
#line 836 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (2)].member); }
    break;

  case 234:
#line 838 "xi-grammar.y"
    {
          ERROR("invalid SDAG member",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
    break;

  case 235:
#line 846 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 236:
#line 848 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 237:
#line 850 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 238:
#line 852 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 239:
#line 854 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 240:
#line 856 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 241:
#line 858 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 242:
#line 860 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 243:
#line 862 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 244:
#line 864 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 245:
#line 866 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 246:
#line 869 "xi-grammar.y"
    { 
                  (yyval.entry) = new Entry(lineno, (yyvsp[(2) - (7)].intval), (yyvsp[(3) - (7)].type), (yyvsp[(4) - (7)].strval), (yyvsp[(5) - (7)].plist), (yyvsp[(6) - (7)].val), (yyvsp[(7) - (7)].sentry), (const char *) NULL, (yylsp[(1) - (7)]).first_line, (yyloc).last_line);
		  if ((yyvsp[(7) - (7)].sentry) != 0) { 
		    (yyvsp[(7) - (7)].sentry)->con1 = new SdagConstruct(SIDENT, (yyvsp[(4) - (7)].strval));
                    (yyvsp[(7) - (7)].sentry)->setEntry((yyval.entry));
                    (yyvsp[(7) - (7)].sentry)->param = new ParamList((yyvsp[(5) - (7)].plist));
                  }
		}
    break;

  case 247:
#line 878 "xi-grammar.y"
    { 
                  Entry *e = new Entry(lineno, (yyvsp[(2) - (5)].intval), 0, (yyvsp[(3) - (5)].strval), (yyvsp[(4) - (5)].plist),  0, (yyvsp[(5) - (5)].sentry), (const char *) NULL, (yylsp[(1) - (5)]).first_line, (yyloc).last_line);
                  if ((yyvsp[(5) - (5)].sentry) != 0) {
		    (yyvsp[(5) - (5)].sentry)->con1 = new SdagConstruct(SIDENT, (yyvsp[(3) - (5)].strval));
                    (yyvsp[(5) - (5)].sentry)->setEntry((yyval.entry));
                    (yyvsp[(5) - (5)].sentry)->param = new ParamList((yyvsp[(4) - (5)].plist));
                  }
		  if (e->param && e->param->isCkMigMsgPtr()) {
		    WARNING("CkMigrateMsg chare constructor is taken for granted",
		            (yyloc).first_column, (yyloc).last_column);
		    (yyval.entry) = NULL;
		  } else {
		    (yyval.entry) = e;
		  }
		}
    break;

  case 248:
#line 894 "xi-grammar.y"
    {
                  int attribs = SACCEL;
                  const char* name = (yyvsp[(6) - (13)].strval);
                  ParamList* paramList = (yyvsp[(7) - (13)].plist);
                  ParamList* accelParamList = (yyvsp[(8) - (13)].plist);
		  XStr* codeBody = new XStr((yyvsp[(10) - (13)].strval));
                  const char* callbackName = (yyvsp[(12) - (13)].strval);

                  (yyval.entry) = new Entry(lineno, attribs, new BuiltinType("void"), name, paramList, 0, 0, 0 );
                  (yyval.entry)->setAccelParam(accelParamList);
                  (yyval.entry)->setAccelCodeBody(codeBody);
                  (yyval.entry)->setAccelCallbackName(new XStr(callbackName));
                }
    break;

  case 249:
#line 910 "xi-grammar.y"
    { (yyval.accelBlock) = new AccelBlock(lineno, new XStr((yyvsp[(3) - (5)].strval))); }
    break;

  case 250:
#line 912 "xi-grammar.y"
    { (yyval.accelBlock) = new AccelBlock(lineno, NULL); }
    break;

  case 251:
#line 916 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 252:
#line 920 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 253:
#line 922 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(2) - (3)].intval); }
    break;

  case 254:
#line 924 "xi-grammar.y"
    { ERROR("invalid entry method attribute list",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
    break;

  case 255:
#line 931 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (1)].intval); }
    break;

  case 256:
#line 933 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (3)].intval) | (yyvsp[(3) - (3)].intval); }
    break;

  case 257:
#line 937 "xi-grammar.y"
    { (yyval.intval) = STHREADED; }
    break;

  case 258:
#line 939 "xi-grammar.y"
    { (yyval.intval) = SSYNC; }
    break;

  case 259:
#line 941 "xi-grammar.y"
    { (yyval.intval) = SIGET; }
    break;

  case 260:
#line 943 "xi-grammar.y"
    { (yyval.intval) = SLOCKED; }
    break;

  case 261:
#line 945 "xi-grammar.y"
    { (yyval.intval) = SCREATEHERE; }
    break;

  case 262:
#line 947 "xi-grammar.y"
    { (yyval.intval) = SCREATEHOME; }
    break;

  case 263:
#line 949 "xi-grammar.y"
    { (yyval.intval) = SNOKEEP; }
    break;

  case 264:
#line 951 "xi-grammar.y"
    { (yyval.intval) = SNOTRACE; }
    break;

  case 265:
#line 953 "xi-grammar.y"
    { (yyval.intval) = SAPPWORK; }
    break;

  case 266:
#line 955 "xi-grammar.y"
    { (yyval.intval) = SIMMEDIATE; }
    break;

  case 267:
#line 957 "xi-grammar.y"
    { (yyval.intval) = SSKIPSCHED; }
    break;

  case 268:
#line 959 "xi-grammar.y"
    { (yyval.intval) = SINLINE; }
    break;

  case 269:
#line 961 "xi-grammar.y"
    { (yyval.intval) = SLOCAL; }
    break;

  case 270:
#line 963 "xi-grammar.y"
    { (yyval.intval) = SPYTHON; }
    break;

  case 271:
#line 965 "xi-grammar.y"
    { (yyval.intval) = SMEM; }
    break;

  case 272:
#line 967 "xi-grammar.y"
    { (yyval.intval) = SREDUCE; }
    break;

  case 273:
#line 969 "xi-grammar.y"
    {
#ifdef CMK_USING_XLC
        WARNING("a known bug in xl compilers (PMR 18366,122,000) currently breaks "
                "aggregate entry methods.\n"
                "Until a fix is released, this tag will be ignored on those compilers.",
                (yylsp[(1) - (1)]).first_column, (yylsp[(1) - (1)]).last_column, (yylsp[(1) - (1)]).first_line);
        (yyval.intval) = 0;
#else
        (yyval.intval) = SAGGREGATE;
#endif
    }
    break;

  case 274:
#line 981 "xi-grammar.y"
    { (yyval.intval) = SSCATTERV; }
    break;

  case 275:
#line 983 "xi-grammar.y"
    {
		  ERROR("invalid entry method attribute",
		        (yylsp[(1) - (1)]).first_column, (yylsp[(1) - (1)]).last_column);
		  yyclearin;
		  yyerrok;
		}
    break;

  case 276:
#line 992 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 277:
#line 994 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 278:
#line 996 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 279:
#line 1000 "xi-grammar.y"
    { (yyval.strval) = ""; }
    break;

  case 280:
#line 1002 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 281:
#line 1004 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[(1) - (3)].strval))+strlen((yyvsp[(3) - (3)].strval))+3];
			sprintf(tmp,"%s, %s", (yyvsp[(1) - (3)].strval), (yyvsp[(3) - (3)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 282:
#line 1012 "xi-grammar.y"
    { (yyval.strval) = ""; }
    break;

  case 283:
#line 1014 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 284:
#line 1016 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s[%s]%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 285:
#line 1022 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s{%s}%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 286:
#line 1028 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s(%s)%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 287:
#line 1034 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(2) - (4)].strval))+strlen((yyvsp[(4) - (4)].strval))+3];
			sprintf(tmp,"(%s)%s", (yyvsp[(2) - (4)].strval), (yyvsp[(4) - (4)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 288:
#line 1042 "xi-grammar.y"
    {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			(yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].type),(yyvsp[(2) - (3)].strval));
		}
    break;

  case 289:
#line 1049 "xi-grammar.y"
    { 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			(yyval.intval) = 0;
		}
    break;

  case 290:
#line 1057 "xi-grammar.y"
    { 
			in_braces=0;
			(yyval.intval) = 0;
		}
    break;

  case 291:
#line 1064 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (1)].type));}
    break;

  case 292:
#line 1066 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].type),(yyvsp[(2) - (3)].strval)); (yyval.pname)->setConditional((yyvsp[(3) - (3)].intval)); }
    break;

  case 293:
#line 1068 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (4)].type),(yyvsp[(2) - (4)].strval),0,(yyvsp[(4) - (4)].val));}
    break;

  case 294:
#line 1070 "xi-grammar.y"
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].pname)->getType(), (yyvsp[(1) - (3)].pname)->getName() ,(yyvsp[(2) - (3)].strval));
		}
    break;

  case 295:
#line 1076 "xi-grammar.y"
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READONLY; }
    break;

  case 296:
#line 1077 "xi-grammar.y"
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READWRITE; }
    break;

  case 297:
#line 1078 "xi-grammar.y"
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_WRITEONLY; }
    break;

  case 298:
#line 1081 "xi-grammar.y"
    { (yyval.xstrptr) = new XStr((yyvsp[(1) - (1)].strval)); }
    break;

  case 299:
#line 1082 "xi-grammar.y"
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "->" << (yyvsp[(4) - (4)].strval); }
    break;

  case 300:
#line 1083 "xi-grammar.y"
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[(1) - (3)].xstrptr)) << "." << (yyvsp[(3) - (3)].strval); }
    break;

  case 301:
#line 1085 "xi-grammar.y"
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "[" << *((yyvsp[(3) - (4)].xstrptr)) << "]";
                  delete (yyvsp[(1) - (4)].xstrptr);
                  delete (yyvsp[(3) - (4)].xstrptr);
                }
    break;

  case 302:
#line 1092 "xi-grammar.y"
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "[" << (yyvsp[(3) - (4)].strval) << "]";
                  delete (yyvsp[(1) - (4)].xstrptr);
                }
    break;

  case 303:
#line 1098 "xi-grammar.y"
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "(" << *((yyvsp[(3) - (4)].xstrptr)) << ")";
                  delete (yyvsp[(1) - (4)].xstrptr);
                  delete (yyvsp[(3) - (4)].xstrptr);
                }
    break;

  case 304:
#line 1107 "xi-grammar.y"
    {
                  in_bracket = 0;
                  (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].pname)->getType(), (yyvsp[(1) - (3)].pname)->getName(), (yyvsp[(2) - (3)].strval));
                }
    break;

  case 305:
#line 1114 "xi-grammar.y"
    {
                  (yyval.pname) = new Parameter(lineno, (yyvsp[(3) - (7)].type), (yyvsp[(4) - (7)].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[(6) - (7)].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[(1) - (7)].intval));
                }
    break;

  case 306:
#line 1120 "xi-grammar.y"
    {
		  (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (5)].type), (yyvsp[(2) - (5)].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[(4) - (5)].xstrptr));
                  (yyval.pname)->setAccelBufferType(Parameter::ACCEL_BUFFER_TYPE_READWRITE);
		}
    break;

  case 307:
#line 1126 "xi-grammar.y"
    {
                  (yyval.pname) = (yyvsp[(3) - (6)].pname);
                  (yyval.pname)->setAccelInstName((yyvsp[(5) - (6)].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[(1) - (6)].intval));
		}
    break;

  case 308:
#line 1134 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (1)].pname)); }
    break;

  case 309:
#line 1136 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (3)].pname),(yyvsp[(3) - (3)].plist)); }
    break;

  case 310:
#line 1140 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (1)].pname)); }
    break;

  case 311:
#line 1142 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (3)].pname),(yyvsp[(3) - (3)].plist)); }
    break;

  case 312:
#line 1146 "xi-grammar.y"
    { (yyval.plist) = (yyvsp[(2) - (3)].plist); }
    break;

  case 313:
#line 1148 "xi-grammar.y"
    { (yyval.plist) = new ParamList(new Parameter(0, new BuiltinType("void"))); }
    break;

  case 314:
#line 1152 "xi-grammar.y"
    { (yyval.plist) = (yyvsp[(2) - (3)].plist); }
    break;

  case 315:
#line 1154 "xi-grammar.y"
    { (yyval.plist) = 0; }
    break;

  case 316:
#line 1158 "xi-grammar.y"
    { (yyval.val) = 0; }
    break;

  case 317:
#line 1160 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(3) - (3)].strval)); }
    break;

  case 318:
#line 1164 "xi-grammar.y"
    { (yyval.sentry) = 0; }
    break;

  case 319:
#line 1166 "xi-grammar.y"
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[(1) - (1)].sc)); }
    break;

  case 320:
#line 1168 "xi-grammar.y"
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[(2) - (4)].slist)); }
    break;

  case 321:
#line 1172 "xi-grammar.y"
    { (yyval.slist) = new SListConstruct((yyvsp[(1) - (1)].sc)); }
    break;

  case 322:
#line 1174 "xi-grammar.y"
    { (yyval.slist) = new SListConstruct((yyvsp[(1) - (2)].sc), (yyvsp[(2) - (2)].slist));  }
    break;

  case 323:
#line 1178 "xi-grammar.y"
    { (yyval.olist) = new OListConstruct((yyvsp[(1) - (1)].sc)); }
    break;

  case 324:
#line 1180 "xi-grammar.y"
    { (yyval.olist) = new OListConstruct((yyvsp[(1) - (2)].sc), (yyvsp[(2) - (2)].slist)); }
    break;

  case 325:
#line 1184 "xi-grammar.y"
    { (yyval.clist) = new CaseListConstruct((yyvsp[(1) - (1)].when)); }
    break;

  case 326:
#line 1186 "xi-grammar.y"
    { (yyval.clist) = new CaseListConstruct((yyvsp[(1) - (2)].when), (yyvsp[(2) - (2)].clist)); }
    break;

  case 327:
#line 1188 "xi-grammar.y"
    {
		  ERROR("case blocks can only contain when clauses",
		        (yylsp[(1) - (1)]).first_column, (yylsp[(1) - (1)]).last_column);
		  (yyval.clist) = 0;
		}
    break;

  case 328:
#line 1196 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 329:
#line 1198 "xi-grammar.y"
    { (yyval.strval) = 0; }
    break;

  case 330:
#line 1202 "xi-grammar.y"
    { (yyval.when) = new WhenConstruct((yyvsp[(2) - (4)].entrylist), 0); }
    break;

  case 331:
#line 1204 "xi-grammar.y"
    { (yyval.when) = new WhenConstruct((yyvsp[(2) - (3)].entrylist), (yyvsp[(3) - (3)].sc)); }
    break;

  case 332:
#line 1206 "xi-grammar.y"
    { (yyval.when) = new WhenConstruct((yyvsp[(2) - (5)].entrylist), (yyvsp[(4) - (5)].slist)); }
    break;

  case 333:
#line 1210 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 334:
#line 1212 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 335:
#line 1214 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 336:
#line 1216 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 337:
#line 1218 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 338:
#line 1220 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 339:
#line 1222 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 340:
#line 1224 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 341:
#line 1226 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 342:
#line 1228 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 343:
#line 1230 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 344:
#line 1232 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 345:
#line 1236 "xi-grammar.y"
    { (yyval.sc) = new AtomicConstruct((yyvsp[(4) - (6)].strval), (yyvsp[(2) - (6)].strval), (yylsp[(3) - (6)]).first_line); }
    break;

  case 346:
#line 1238 "xi-grammar.y"
    { (yyval.sc) = new OverlapConstruct((yyvsp[(3) - (4)].olist)); }
    break;

  case 347:
#line 1240 "xi-grammar.y"
    { (yyval.sc) = (yyvsp[(1) - (1)].when); }
    break;

  case 348:
#line 1242 "xi-grammar.y"
    { (yyval.sc) = new CaseConstruct((yyvsp[(3) - (4)].clist)); }
    break;

  case 349:
#line 1244 "xi-grammar.y"
    { (yyval.sc) = new ForConstruct((yyvsp[(3) - (11)].intexpr), (yyvsp[(5) - (11)].intexpr), (yyvsp[(7) - (11)].intexpr), (yyvsp[(10) - (11)].slist)); }
    break;

  case 350:
#line 1246 "xi-grammar.y"
    { (yyval.sc) = new ForConstruct((yyvsp[(3) - (9)].intexpr), (yyvsp[(5) - (9)].intexpr), (yyvsp[(7) - (9)].intexpr), (yyvsp[(9) - (9)].sc)); }
    break;

  case 351:
#line 1248 "xi-grammar.y"
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[(3) - (12)].strval)), (yyvsp[(6) - (12)].intexpr),
		             (yyvsp[(8) - (12)].intexpr), (yyvsp[(10) - (12)].intexpr), (yyvsp[(12) - (12)].sc)); }
    break;

  case 352:
#line 1251 "xi-grammar.y"
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[(3) - (14)].strval)), (yyvsp[(6) - (14)].intexpr),
		             (yyvsp[(8) - (14)].intexpr), (yyvsp[(10) - (14)].intexpr), (yyvsp[(13) - (14)].slist)); }
    break;

  case 353:
#line 1254 "xi-grammar.y"
    { (yyval.sc) = new IfConstruct((yyvsp[(3) - (6)].intexpr), (yyvsp[(5) - (6)].sc), (yyvsp[(6) - (6)].sc)); }
    break;

  case 354:
#line 1256 "xi-grammar.y"
    { (yyval.sc) = new IfConstruct((yyvsp[(3) - (8)].intexpr), (yyvsp[(6) - (8)].slist), (yyvsp[(8) - (8)].sc)); }
    break;

  case 355:
#line 1258 "xi-grammar.y"
    { (yyval.sc) = new WhileConstruct((yyvsp[(3) - (5)].intexpr), (yyvsp[(5) - (5)].sc)); }
    break;

  case 356:
#line 1260 "xi-grammar.y"
    { (yyval.sc) = new WhileConstruct((yyvsp[(3) - (7)].intexpr), (yyvsp[(6) - (7)].slist)); }
    break;

  case 357:
#line 1262 "xi-grammar.y"
    { (yyval.sc) = new AtomicConstruct((yyvsp[(2) - (4)].strval), NULL, (yyloc).first_line); }
    break;

  case 358:
#line 1264 "xi-grammar.y"
    {
		  ERROR("unknown SDAG construct or malformed entry method declaration.\n"
		        "You may have forgotten to terminate a previous entry method declaration with a"
		        " semicolon or forgotten to mark a block of sequential SDAG code as 'serial'",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
    break;

  case 359:
#line 1274 "xi-grammar.y"
    { (yyval.sc) = 0; }
    break;

  case 360:
#line 1276 "xi-grammar.y"
    { (yyval.sc) = new ElseConstruct((yyvsp[(2) - (2)].sc)); }
    break;

  case 361:
#line 1278 "xi-grammar.y"
    { (yyval.sc) = new ElseConstruct((yyvsp[(3) - (4)].slist)); }
    break;

  case 362:
#line 1282 "xi-grammar.y"
    { (yyval.intexpr) = new IntExprConstruct((yyvsp[(1) - (1)].strval)); }
    break;

  case 363:
#line 1286 "xi-grammar.y"
    { in_int_expr = 0; (yyval.intval) = 0; }
    break;

  case 364:
#line 1290 "xi-grammar.y"
    { in_int_expr = 1; (yyval.intval) = 0; }
    break;

  case 365:
#line 1294 "xi-grammar.y"
    {
		  (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[(1) - (2)].strval), (yyvsp[(2) - (2)].plist), 0, 0, 0, (yyloc).first_line, (yyloc).last_line);
		}
    break;

  case 366:
#line 1298 "xi-grammar.y"
    {
		  (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[(1) - (5)].strval), (yyvsp[(5) - (5)].plist), 0, 0, (yyvsp[(3) - (5)].strval), (yyloc).first_line, (yyloc).last_line);
		}
    break;

  case 367:
#line 1304 "xi-grammar.y"
    { (yyval.entrylist) = new EntryList((yyvsp[(1) - (1)].entry)); }
    break;

  case 368:
#line 1306 "xi-grammar.y"
    { (yyval.entrylist) = new EntryList((yyvsp[(1) - (3)].entry),(yyvsp[(3) - (3)].entrylist)); }
    break;

  case 369:
#line 1310 "xi-grammar.y"
    { in_bracket=1; }
    break;

  case 370:
#line 1313 "xi-grammar.y"
    { in_bracket=0; }
    break;

  case 371:
#line 1317 "xi-grammar.y"
    { if (!macroDefined((yyvsp[(2) - (2)].strval), 1)) in_comment = 1; }
    break;

  case 372:
#line 1321 "xi-grammar.y"
    { if (!macroDefined((yyvsp[(2) - (2)].strval), 0)) in_comment = 1; }
    break;


/* Line 1267 of yacc.c.  */
#line 4492 "y.tab.c"
      default: break;
    }
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;
  *++yylsp = yyloc;

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

  yyerror_range[0] = yylloc;

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
		      yytoken, &yylval, &yylloc);
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

  yyerror_range[0] = yylsp[1-yylen];
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

      yyerror_range[0] = *yylsp;
      yydestruct ("Error: popping",
		  yystos[yystate], yyvsp, yylsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  if (yyn == YYFINAL)
    YYACCEPT;

  *++yyvsp = yylval;

  yyerror_range[1] = yylloc;
  /* Using YYLLOC is tempting, but would change the location of
     the look-ahead.  YYLOC is available though.  */
  YYLLOC_DEFAULT (yyloc, (yyerror_range - 1), 2);
  *++yylsp = yyloc;

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
		 yytoken, &yylval, &yylloc);
  /* Do not reclaim the symbols of the rule which action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
		  yystos[*yyssp], yyvsp, yylsp);
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


#line 1324 "xi-grammar.y"


void yyerror(const char *msg) { }

