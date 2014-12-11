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
     PACKED = 294,
     VARSIZE = 295,
     ENTRY = 296,
     FOR = 297,
     FORALL = 298,
     WHILE = 299,
     WHEN = 300,
     OVERLAP = 301,
     ATOMIC = 302,
     IF = 303,
     ELSE = 304,
     PYTHON = 305,
     LOCAL = 306,
     NAMESPACE = 307,
     USING = 308,
     IDENT = 309,
     NUMBER = 310,
     LITERAL = 311,
     CPROGRAM = 312,
     HASHIF = 313,
     HASHIFDEF = 314,
     INT = 315,
     LONG = 316,
     SHORT = 317,
     CHAR = 318,
     FLOAT = 319,
     DOUBLE = 320,
     UNSIGNED = 321,
     ACCEL = 322,
     READWRITE = 323,
     WRITEONLY = 324,
     ACCELBLOCK = 325,
     MEMCRITICAL = 326,
     REDUCTIONTARGET = 327,
     CASE = 328
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
#define PACKED 294
#define VARSIZE 295
#define ENTRY 296
#define FOR 297
#define FORALL 298
#define WHILE 299
#define WHEN 300
#define OVERLAP 301
#define ATOMIC 302
#define IF 303
#define ELSE 304
#define PYTHON 305
#define LOCAL 306
#define NAMESPACE 307
#define USING 308
#define IDENT 309
#define NUMBER 310
#define LITERAL 311
#define CPROGRAM 312
#define HASHIF 313
#define HASHIFDEF 314
#define INT 315
#define LONG 316
#define SHORT 317
#define CHAR 318
#define FLOAT 319
#define DOUBLE 320
#define UNSIGNED 321
#define ACCEL 322
#define READWRITE 323
#define WRITEONLY 324
#define ACCELBLOCK 325
#define MEMCRITICAL 326
#define REDUCTIONTARGET 327
#define CASE 328




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
#line 333 "y.tab.c"
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
#line 358 "y.tab.c"

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
#define YYFINAL  55
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   1489

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  90
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  116
/* YYNRULES -- Number of rules.  */
#define YYNRULES  366
/* YYNRULES -- Number of states.  */
#define YYNSTATES  712

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
      17,    19,    21,    23,    25,    27,    29,    31,    33,    35,
      37,    39,    41,    43,    45,    47,    49,    51,    53,    55,
      57,    59,    61,    63,    65,    67,    69,    71,    73,    75,
      77,    79,    81,    83,    85,    87,    89,    91,    93,    95,
      97,    99,   101,   103,   105,   107,   109,   111,   116,   120,
     124,   126,   131,   132,   135,   139,   142,   145,   148,   156,
     162,   168,   171,   174,   177,   180,   183,   186,   189,   192,
     194,   196,   198,   200,   202,   204,   206,   208,   212,   213,
     215,   216,   220,   222,   224,   226,   228,   231,   234,   238,
     242,   245,   248,   251,   253,   255,   258,   260,   263,   266,
     268,   270,   273,   276,   279,   288,   290,   292,   294,   296,
     299,   302,   304,   306,   308,   311,   314,   317,   319,   322,
     324,   326,   330,   331,   334,   339,   346,   347,   349,   350,
     354,   356,   360,   362,   364,   365,   369,   371,   375,   376,
     378,   380,   381,   385,   387,   391,   393,   395,   396,   398,
     399,   402,   408,   410,   413,   417,   424,   425,   428,   430,
     434,   440,   446,   452,   458,   463,   467,   474,   481,   487,
     493,   499,   505,   511,   516,   524,   525,   528,   529,   532,
     535,   538,   542,   545,   549,   551,   555,   560,   563,   566,
     569,   572,   575,   577,   582,   583,   586,   588,   590,   592,
     594,   597,   600,   603,   607,   614,   624,   628,   635,   639,
     646,   656,   666,   668,   672,   674,   676,   678,   681,   684,
     686,   688,   690,   692,   694,   696,   698,   700,   702,   704,
     706,   708,   716,   722,   736,   742,   745,   747,   748,   752,
     754,   756,   760,   762,   764,   766,   768,   770,   772,   774,
     776,   778,   780,   782,   784,   786,   789,   791,   793,   795,
     797,   799,   801,   803,   804,   806,   810,   811,   813,   819,
     825,   831,   836,   840,   842,   844,   846,   850,   855,   859,
     861,   863,   865,   867,   872,   876,   881,   886,   891,   895,
     903,   909,   916,   918,   922,   924,   928,   932,   935,   939,
     942,   943,   947,   949,   951,   956,   958,   961,   963,   966,
     968,   971,   973,   975,   976,   981,   985,   991,   997,  1002,
    1007,  1019,  1029,  1042,  1057,  1064,  1073,  1079,  1087,  1091,
    1097,  1102,  1104,  1109,  1121,  1131,  1144,  1159,  1166,  1175,
    1181,  1189,  1193,  1195,  1196,  1199,  1204,  1206,  1208,  1210,
    1213,  1219,  1221,  1225,  1227,  1229,  1232
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
      91,     0,    -1,    92,    -1,    -1,    97,    92,    -1,    -1,
       5,    -1,    -1,    74,    -1,    54,    -1,     3,    -1,     4,
      -1,     5,    -1,     7,    -1,     8,    -1,     9,    -1,    11,
      -1,    12,    -1,    13,    -1,    14,    -1,    15,    -1,    19,
      -1,    20,    -1,    21,    -1,    22,    -1,    23,    -1,    24,
      -1,    25,    -1,    26,    -1,    27,    -1,    28,    -1,    29,
      -1,    30,    -1,    32,    -1,    33,    -1,    34,    -1,    35,
      -1,    36,    -1,    39,    -1,    40,    -1,    41,    -1,    42,
      -1,    43,    -1,    44,    -1,    45,    -1,    46,    -1,    47,
      -1,    48,    -1,    49,    -1,    51,    -1,    53,    -1,    67,
      -1,    70,    -1,    71,    -1,    72,    -1,    73,    -1,    54,
      -1,    96,    75,    75,    54,    -1,     3,    95,    98,    -1,
       4,    95,    98,    -1,    74,    -1,    76,    99,    77,    94,
      -1,    -1,   101,    99,    -1,    53,    52,    96,    -1,    53,
      96,    -1,    93,   158,    -1,    93,   137,    -1,     5,    41,
     168,   108,    95,   105,   185,    -1,    93,    76,    99,    77,
      94,    -1,    52,    95,    76,    99,    77,    -1,   100,    74,
      -1,   100,   165,    -1,    93,    97,    -1,    93,   140,    -1,
      93,   141,    -1,    93,   142,    -1,    93,   144,    -1,    93,
     155,    -1,   204,    -1,   205,    -1,   167,    -1,     1,    -1,
     116,    -1,    55,    -1,    56,    -1,   102,    -1,   102,    78,
     103,    -1,    -1,   103,    -1,    -1,    79,   104,    80,    -1,
      60,    -1,    61,    -1,    62,    -1,    63,    -1,    66,    60,
      -1,    66,    61,    -1,    66,    61,    60,    -1,    66,    61,
      61,    -1,    66,    62,    -1,    66,    63,    -1,    61,    61,
      -1,    64,    -1,    65,    -1,    61,    65,    -1,    37,    -1,
      95,   105,    -1,    96,   105,    -1,   106,    -1,   108,    -1,
     109,    81,    -1,   110,    81,    -1,   111,    81,    -1,   113,
      82,    81,    95,    83,    82,   183,    83,    -1,   109,    -1,
     110,    -1,   111,    -1,   112,    -1,    38,   113,    -1,   113,
      38,    -1,   109,    -1,   110,    -1,   111,    -1,    38,   114,
      -1,   114,    38,    -1,   114,    84,    -1,   114,    -1,   113,
      84,    -1,   113,    -1,   174,    -1,   202,   117,   203,    -1,
      -1,   118,   119,    -1,     6,   116,    96,   119,    -1,     6,
      16,   109,    81,    96,   119,    -1,    -1,    37,    -1,    -1,
      85,   124,    86,    -1,   125,    -1,   125,    78,   124,    -1,
      39,    -1,    40,    -1,    -1,    85,   127,    86,    -1,   132,
      -1,   132,    78,   127,    -1,    -1,    56,    -1,    50,    -1,
      -1,    85,   131,    86,    -1,   129,    -1,   129,    78,   131,
      -1,    30,    -1,    50,    -1,    -1,    17,    -1,    -1,    85,
      86,    -1,   133,   116,    95,   134,    74,    -1,   135,    -1,
     135,   136,    -1,    16,   123,   107,    -1,    16,   123,   107,
      76,   136,    77,    -1,    -1,    75,   139,    -1,   108,    -1,
     108,    78,   139,    -1,    11,   126,   107,   138,   156,    -1,
      12,   126,   107,   138,   156,    -1,    13,   126,   107,   138,
     156,    -1,    14,   126,   107,   138,   156,    -1,    85,    55,
      95,    86,    -1,    85,    95,    86,    -1,    15,   130,   143,
     107,   138,   156,    -1,    15,   143,   130,   107,   138,   156,
      -1,    11,   126,    95,   138,   156,    -1,    12,   126,    95,
     138,   156,    -1,    13,   126,    95,   138,   156,    -1,    14,
     126,    95,   138,   156,    -1,    15,   143,    95,   138,   156,
      -1,    16,   123,    95,    74,    -1,    16,   123,    95,    76,
     136,    77,    74,    -1,    -1,    87,   116,    -1,    -1,    87,
      55,    -1,    87,    56,    -1,    87,   108,    -1,    18,    95,
     150,    -1,   112,   151,    -1,   116,    95,   151,    -1,   152,
      -1,   152,    78,   153,    -1,    22,    79,   153,    80,    -1,
     154,   145,    -1,   154,   146,    -1,   154,   147,    -1,   154,
     148,    -1,   154,   149,    -1,    74,    -1,    76,   157,    77,
      94,    -1,    -1,   163,   157,    -1,   120,    -1,   121,    -1,
     160,    -1,   159,    -1,    10,   161,    -1,    19,   162,    -1,
      18,    95,    -1,     8,   122,    96,    -1,     8,   122,    96,
      82,   122,    83,    -1,     8,   122,    96,    79,   103,    80,
      82,   122,    83,    -1,     7,   122,    96,    -1,     7,   122,
      96,    82,   122,    83,    -1,     9,   122,    96,    -1,     9,
     122,    96,    82,   122,    83,    -1,     9,   122,    96,    79,
     103,    80,    82,   122,    83,    -1,     9,    85,    67,    86,
     122,    96,    82,   122,    83,    -1,   108,    -1,   108,    78,
     161,    -1,    56,    -1,   164,    -1,   166,    -1,   154,   166,
      -1,   158,    74,    -1,     1,    -1,    41,    -1,    77,    -1,
       7,    -1,     8,    -1,     9,    -1,    11,    -1,    12,    -1,
      15,    -1,    13,    -1,    14,    -1,     6,    -1,    41,   169,
     168,    95,   185,   187,   188,    -1,    41,   169,    95,   185,
     188,    -1,    41,    85,    67,    86,    37,    95,   185,   186,
     176,   174,   177,    95,    74,    -1,    70,   176,   174,   177,
      74,    -1,    70,    74,    -1,   115,    -1,    -1,    85,   170,
      86,    -1,     1,    -1,   171,    -1,   171,    78,   170,    -1,
      21,    -1,    23,    -1,    24,    -1,    25,    -1,    32,    -1,
      33,    -1,    34,    -1,    35,    -1,    36,    -1,    26,    -1,
      27,    -1,    28,    -1,    51,    -1,    50,   128,    -1,    71,
      -1,    72,    -1,    31,    -1,     1,    -1,    56,    -1,    55,
      -1,    96,    -1,    -1,    57,    -1,    57,    78,   173,    -1,
      -1,    57,    -1,    57,    85,   174,    86,   174,    -1,    57,
      76,   174,    77,   174,    -1,    57,    82,   173,    83,   174,
      -1,    82,   174,    83,   174,    -1,   116,    95,    85,    -1,
      76,    -1,    77,    -1,   116,    -1,   116,    95,   133,    -1,
     116,    95,    87,   172,    -1,   175,   174,    86,    -1,     6,
      -1,    68,    -1,    69,    -1,    95,    -1,   180,    88,    80,
      95,    -1,   180,    89,    95,    -1,   180,    85,   180,    86,
      -1,   180,    85,    55,    86,    -1,   180,    82,   180,    83,
      -1,   175,   174,    86,    -1,   179,    75,   116,    95,    79,
     180,    80,    -1,   116,    95,    79,   180,    80,    -1,   179,
      75,   181,    79,   180,    80,    -1,   178,    -1,   178,    78,
     183,    -1,   182,    -1,   182,    78,   184,    -1,    82,   183,
      83,    -1,    82,    83,    -1,    85,   184,    86,    -1,    85,
      86,    -1,    -1,    20,    87,    55,    -1,    74,    -1,   195,
      -1,    76,   189,    77,    94,    -1,   195,    -1,   195,   189,
      -1,   195,    -1,   195,   189,    -1,   193,    -1,   193,   191,
      -1,   194,    -1,    56,    -1,    -1,    45,   201,    76,    77,
      -1,    45,   201,   195,    -1,    45,   201,    76,   189,    77,
      -1,    47,   192,   176,   174,   177,    -1,    46,    76,   190,
      77,    -1,    73,    76,   191,    77,    -1,    42,   199,   174,
      74,   174,    74,   174,   198,    76,   189,    77,    -1,    42,
     199,   174,    74,   174,    74,   174,   198,   195,    -1,    43,
      85,    54,    86,   199,   174,    75,   174,    78,   174,   198,
     195,    -1,    43,    85,    54,    86,   199,   174,    75,   174,
      78,   174,   198,    76,   189,    77,    -1,    48,   199,   174,
     198,   195,   196,    -1,    48,   199,   174,   198,    76,   189,
      77,   196,    -1,    44,   199,   174,   198,   195,    -1,    44,
     199,   174,   198,    76,   189,    77,    -1,   176,   174,   177,
      -1,    47,   192,   176,   174,   177,    -1,    46,    76,   190,
      77,    -1,   193,    -1,    73,    76,   191,    77,    -1,    42,
     199,   197,    74,   197,    74,   197,   198,    76,   189,    77,
      -1,    42,   199,   197,    74,   197,    74,   197,   198,   195,
      -1,    43,    85,    54,    86,   199,   197,    75,   197,    78,
     197,   198,   195,    -1,    43,    85,    54,    86,   199,   197,
      75,   197,    78,   197,   198,    76,   189,    77,    -1,    48,
     199,   197,   198,   195,   196,    -1,    48,   199,   197,   198,
      76,   189,    77,   196,    -1,    44,   199,   197,   198,   195,
      -1,    44,   199,   197,   198,    76,   189,    77,    -1,   176,
     174,   177,    -1,     1,    -1,    -1,    49,   195,    -1,    49,
      76,   189,    77,    -1,   174,    -1,    83,    -1,    82,    -1,
      54,   185,    -1,    54,   202,   174,   203,   185,    -1,   200,
      -1,   200,    78,   201,    -1,    85,    -1,    86,    -1,    58,
      95,    -1,    59,    95,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   192,   192,   197,   200,   205,   206,   211,   212,   217,
     219,   220,   221,   223,   224,   225,   227,   228,   229,   230,
     231,   235,   236,   237,   238,   239,   240,   241,   242,   243,
     244,   245,   246,   247,   248,   249,   250,   251,   254,   255,
     256,   257,   258,   259,   260,   261,   262,   263,   264,   266,
     268,   269,   272,   273,   274,   275,   278,   280,   287,   291,
     298,   300,   305,   306,   310,   312,   314,   316,   318,   330,
     332,   334,   336,   342,   344,   346,   348,   350,   352,   354,
     356,   358,   360,   368,   370,   372,   376,   378,   383,   384,
     389,   390,   394,   396,   398,   400,   402,   404,   406,   408,
     410,   412,   414,   416,   418,   420,   422,   426,   427,   434,
     436,   440,   444,   446,   450,   454,   456,   458,   460,   462,
     464,   468,   470,   472,   474,   476,   480,   482,   486,   488,
     492,   496,   501,   502,   506,   510,   515,   516,   521,   522,
     532,   534,   538,   540,   545,   546,   550,   552,   557,   558,
     562,   567,   568,   572,   574,   578,   580,   585,   586,   590,
     591,   594,   598,   600,   604,   606,   611,   612,   616,   618,
     622,   624,   628,   632,   636,   642,   646,   648,   652,   654,
     658,   662,   666,   670,   672,   677,   678,   683,   684,   686,
     688,   697,   699,   701,   705,   707,   711,   715,   717,   719,
     721,   723,   727,   729,   734,   741,   745,   747,   749,   750,
     752,   754,   756,   760,   762,   764,   770,   776,   785,   787,
     789,   795,   803,   805,   808,   812,   816,   818,   823,   825,
     833,   835,   837,   839,   841,   843,   845,   847,   849,   851,
     853,   856,   866,   883,   899,   901,   905,   910,   911,   913,
     920,   922,   926,   928,   930,   932,   934,   936,   938,   940,
     942,   944,   946,   948,   950,   952,   954,   956,   958,   970,
     979,   981,   983,   988,   989,   991,  1000,  1001,  1003,  1009,
    1015,  1021,  1029,  1036,  1044,  1051,  1053,  1055,  1057,  1064,
    1065,  1066,  1069,  1070,  1071,  1072,  1079,  1085,  1094,  1101,
    1107,  1113,  1121,  1123,  1127,  1129,  1133,  1135,  1139,  1141,
    1146,  1147,  1151,  1153,  1155,  1159,  1161,  1165,  1167,  1171,
    1173,  1175,  1183,  1186,  1189,  1191,  1193,  1197,  1199,  1201,
    1203,  1205,  1207,  1209,  1211,  1213,  1215,  1217,  1219,  1223,
    1225,  1227,  1229,  1231,  1233,  1235,  1238,  1241,  1243,  1245,
    1247,  1249,  1251,  1262,  1263,  1265,  1269,  1273,  1277,  1281,
    1285,  1291,  1293,  1297,  1300,  1304,  1308
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
  "VOID", "CONST", "PACKED", "VARSIZE", "ENTRY", "FOR", "FORALL", "WHILE",
  "WHEN", "OVERLAP", "ATOMIC", "IF", "ELSE", "PYTHON", "LOCAL",
  "NAMESPACE", "USING", "IDENT", "NUMBER", "LITERAL", "CPROGRAM", "HASHIF",
  "HASHIFDEF", "INT", "LONG", "SHORT", "CHAR", "FLOAT", "DOUBLE",
  "UNSIGNED", "ACCEL", "READWRITE", "WRITEONLY", "ACCELBLOCK",
  "MEMCRITICAL", "REDUCTIONTARGET", "CASE", "';'", "':'", "'{'", "'}'",
  "','", "'<'", "'>'", "'*'", "'('", "')'", "'&'", "'['", "']'", "'='",
  "'-'", "'.'", "$accept", "File", "ModuleEList", "OptExtern",
  "OptSemiColon", "Name", "QualName", "Module", "ConstructEList",
  "ConstructList", "ConstructSemi", "Construct", "TParam", "TParamList",
  "TParamEList", "OptTParams", "BuiltinType", "NamedType", "QualNamedType",
  "SimpleType", "OnePtrType", "PtrType", "FuncType", "BaseType",
  "BaseDataType", "RestrictedType", "Type", "ArrayDim", "Dim", "DimList",
  "Readonly", "ReadonlyMsg", "OptVoid", "MAttribs", "MAttribList",
  "MAttrib", "CAttribs", "CAttribList", "PythonOptions", "ArrayAttrib",
  "ArrayAttribs", "ArrayAttribList", "CAttrib", "OptConditional",
  "MsgArray", "Var", "VarList", "Message", "OptBaseList", "BaseList",
  "Chare", "Group", "NodeGroup", "ArrayIndexType", "Array", "TChare",
  "TGroup", "TNodeGroup", "TArray", "TMessage", "OptTypeInit",
  "OptNameInit", "TVar", "TVarList", "TemplateSpec", "Template",
  "MemberEList", "MemberList", "NonEntryMember", "InitNode", "InitProc",
  "PUPableClass", "IncludeFile", "Member", "MemberBody", "UnexpectedToken",
  "Entry", "AccelBlock", "EReturn", "EAttribs", "EAttribList", "EAttrib",
  "DefaultParameter", "CPROGRAM_List", "CCode", "ParamBracketStart",
  "ParamBraceStart", "ParamBraceEnd", "Parameter", "AccelBufferType",
  "AccelInstName", "AccelArrayParam", "AccelParameter", "ParamList",
  "AccelParamList", "EParameters", "AccelEParameters", "OptStackSize",
  "OptSdagCode", "Slist", "Olist", "CaseList", "OptTraceName",
  "WhenConstruct", "NonWhenConstruct", "SingleConstruct", "HasElse",
  "IntExpr", "EndIntExpr", "StartIntExpr", "SEntry", "SEntryList",
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
     325,   326,   327,   328,    59,    58,   123,   125,    44,    60,
      62,    42,    40,    41,    38,    91,    93,    61,    45,    46
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    90,    91,    92,    92,    93,    93,    94,    94,    95,
      95,    95,    95,    95,    95,    95,    95,    95,    95,    95,
      95,    95,    95,    95,    95,    95,    95,    95,    95,    95,
      95,    95,    95,    95,    95,    95,    95,    95,    95,    95,
      95,    95,    95,    95,    95,    95,    95,    95,    95,    95,
      95,    95,    95,    95,    95,    95,    96,    96,    97,    97,
      98,    98,    99,    99,   100,   100,   100,   100,   100,   101,
     101,   101,   101,   101,   101,   101,   101,   101,   101,   101,
     101,   101,   101,   102,   102,   102,   103,   103,   104,   104,
     105,   105,   106,   106,   106,   106,   106,   106,   106,   106,
     106,   106,   106,   106,   106,   106,   106,   107,   108,   109,
     109,   110,   111,   111,   112,   113,   113,   113,   113,   113,
     113,   114,   114,   114,   114,   114,   115,   115,   116,   116,
     117,   118,   119,   119,   120,   121,   122,   122,   123,   123,
     124,   124,   125,   125,   126,   126,   127,   127,   128,   128,
     129,   130,   130,   131,   131,   132,   132,   133,   133,   134,
     134,   135,   136,   136,   137,   137,   138,   138,   139,   139,
     140,   140,   141,   142,   143,   143,   144,   144,   145,   145,
     146,   147,   148,   149,   149,   150,   150,   151,   151,   151,
     151,   152,   152,   152,   153,   153,   154,   155,   155,   155,
     155,   155,   156,   156,   157,   157,   158,   158,   158,   158,
     158,   158,   158,   159,   159,   159,   159,   159,   160,   160,
     160,   160,   161,   161,   162,   163,   164,   164,   164,   164,
     165,   165,   165,   165,   165,   165,   165,   165,   165,   165,
     165,   166,   166,   166,   167,   167,   168,   169,   169,   169,
     170,   170,   171,   171,   171,   171,   171,   171,   171,   171,
     171,   171,   171,   171,   171,   171,   171,   171,   171,   171,
     172,   172,   172,   173,   173,   173,   174,   174,   174,   174,
     174,   174,   175,   176,   177,   178,   178,   178,   178,   179,
     179,   179,   180,   180,   180,   180,   180,   180,   181,   182,
     182,   182,   183,   183,   184,   184,   185,   185,   186,   186,
     187,   187,   188,   188,   188,   189,   189,   190,   190,   191,
     191,   191,   192,   192,   193,   193,   193,   194,   194,   194,
     194,   194,   194,   194,   194,   194,   194,   194,   194,   195,
     195,   195,   195,   195,   195,   195,   195,   195,   195,   195,
     195,   195,   195,   196,   196,   196,   197,   198,   199,   200,
     200,   201,   201,   202,   203,   204,   205
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     0,     2,     0,     1,     0,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     4,     3,     3,
       1,     4,     0,     2,     3,     2,     2,     2,     7,     5,
       5,     2,     2,     2,     2,     2,     2,     2,     2,     1,
       1,     1,     1,     1,     1,     1,     1,     3,     0,     1,
       0,     3,     1,     1,     1,     1,     2,     2,     3,     3,
       2,     2,     2,     1,     1,     2,     1,     2,     2,     1,
       1,     2,     2,     2,     8,     1,     1,     1,     1,     2,
       2,     1,     1,     1,     2,     2,     2,     1,     2,     1,
       1,     3,     0,     2,     4,     6,     0,     1,     0,     3,
       1,     3,     1,     1,     0,     3,     1,     3,     0,     1,
       1,     0,     3,     1,     3,     1,     1,     0,     1,     0,
       2,     5,     1,     2,     3,     6,     0,     2,     1,     3,
       5,     5,     5,     5,     4,     3,     6,     6,     5,     5,
       5,     5,     5,     4,     7,     0,     2,     0,     2,     2,
       2,     3,     2,     3,     1,     3,     4,     2,     2,     2,
       2,     2,     1,     4,     0,     2,     1,     1,     1,     1,
       2,     2,     2,     3,     6,     9,     3,     6,     3,     6,
       9,     9,     1,     3,     1,     1,     1,     2,     2,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     7,     5,    13,     5,     2,     1,     0,     3,     1,
       1,     3,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     2,     1,     1,     1,     1,
       1,     1,     1,     0,     1,     3,     0,     1,     5,     5,
       5,     4,     3,     1,     1,     1,     3,     4,     3,     1,
       1,     1,     1,     4,     3,     4,     4,     4,     3,     7,
       5,     6,     1,     3,     1,     3,     3,     2,     3,     2,
       0,     3,     1,     1,     4,     1,     2,     1,     2,     1,
       2,     1,     1,     0,     4,     3,     5,     5,     4,     4,
      11,     9,    12,    14,     6,     8,     5,     7,     3,     5,
       4,     1,     4,    11,     9,    12,    14,     6,     8,     5,
       7,     3,     1,     0,     2,     4,     1,     1,     1,     2,
       5,     1,     3,     1,     1,     2,     2
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       3,     0,     0,     0,     2,     3,    10,    11,    12,    13,
      14,    15,    16,    17,    18,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,    29,    30,    31,    32,    33,
      34,    35,    36,    37,    38,    39,    40,    41,    42,    43,
      44,    45,    46,    47,    48,    49,    50,     9,    51,    52,
      53,    54,    55,     0,     0,     1,     4,    60,     0,    58,
      59,    82,     6,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    81,    79,    80,     0,     0,     0,    56,    65,
     365,   366,   245,   283,   276,     0,   136,   136,   136,     0,
     144,   144,   144,   144,     0,   138,     0,     0,     0,     0,
      73,   206,   207,    67,    74,    75,    76,    77,     0,    78,
      66,   209,   208,     7,   240,   232,   233,   234,   235,   236,
     238,   239,   237,   230,    71,   231,    72,    63,   106,     0,
      92,    93,    94,    95,   103,   104,     0,    90,   109,   110,
     121,   122,   123,   127,   246,     0,     0,    64,     0,   277,
     276,     0,     0,     0,   115,   116,   117,   118,   129,     0,
     137,     0,     0,     0,     0,   222,   210,     0,     0,     0,
       0,     0,     0,     0,   151,     0,     0,   212,   224,   211,
       0,     0,   144,   144,   144,   144,     0,   138,   197,   198,
     199,   200,   201,     8,    61,   124,   102,   105,    96,    97,
     100,   101,    88,   108,   111,   112,   113,   125,   126,     0,
       0,     0,   276,   273,   276,     0,   284,     0,     0,   119,
     120,     0,   128,   132,   216,   213,     0,   218,     0,   155,
     156,     0,   146,    90,   166,   166,   166,   166,   150,     0,
       0,   153,     0,     0,     0,     0,     0,   142,   143,     0,
     140,   164,     0,   118,     0,   194,     0,     7,     0,     0,
       0,     0,     0,     0,    98,    99,    84,    85,    86,    89,
       0,    83,    90,    70,    57,     0,   274,     0,     0,   276,
     244,     0,     0,   363,   132,   134,   276,   136,     0,   136,
     136,     0,   136,   223,   145,     0,   107,     0,     0,     0,
       0,     0,     0,   175,     0,   152,   166,   166,   139,     0,
     157,   185,     0,   192,   187,     0,   196,    69,   166,   166,
     166,   166,   166,     0,     0,    91,     0,   276,   273,   276,
     276,   281,   132,     0,   133,     0,   130,     0,     0,     0,
       0,     0,     0,   147,   168,   167,   202,     0,   170,   171,
     172,   173,   174,   154,     0,     0,   141,   158,     0,   157,
       0,     0,   191,   188,   189,   190,   193,   195,     0,     0,
       0,     0,     0,   183,   157,    87,     0,    68,   279,   275,
     280,   278,   135,     0,   364,   131,   217,     0,   214,     0,
       0,   219,     0,   229,     0,     0,     0,     0,     0,   225,
     226,   176,   177,     0,   163,   165,   186,   178,   179,   180,
     181,   182,     0,   307,   285,   276,   302,     0,     0,   136,
     136,   136,   169,   249,     0,     0,   227,     7,   228,   205,
     159,     0,   157,     0,     0,   306,     0,     0,     0,     0,
     269,   252,   253,   254,   255,   261,   262,   263,   268,   256,
     257,   258,   259,   260,   148,   264,     0,   266,   267,     0,
     250,    56,     0,     0,   203,     0,     0,   184,   282,     0,
     286,   288,   303,   114,   215,   221,   220,   149,   265,     0,
     248,     0,     0,     0,   160,   161,   271,   270,   272,   287,
       0,   251,   352,     0,     0,     0,     0,     0,   323,     0,
       0,   312,     0,   276,   242,   341,   313,   310,     0,   358,
     276,     0,   276,     0,   361,     0,     0,   322,     0,   276,
       0,     0,     0,     0,     0,     0,     0,   356,     0,     0,
       0,   359,   276,     0,     0,   325,     0,     0,   276,     0,
       0,     0,     0,     0,   323,     0,     0,   276,     0,   319,
     321,     7,   316,   351,     0,   241,     0,     0,   276,     0,
     357,     0,     0,   362,   324,     0,   340,   318,     0,     0,
     276,     0,   276,     0,     0,   276,     0,     0,   342,   320,
     314,   311,   289,   290,   291,   309,     0,     0,   304,     0,
     276,     0,   276,     0,   349,     0,   326,   339,     0,   353,
       0,     0,     0,     0,   276,     0,     0,   338,     0,     0,
       0,   308,     0,   276,     0,     0,   360,     0,     0,   347,
     276,     0,     0,   328,     0,     0,   329,     0,     0,   276,
       0,   305,     0,     0,   276,   350,   353,     0,   354,     0,
     276,     0,   336,   327,     0,   353,   292,     0,     0,     0,
       0,     0,     0,     0,   348,     0,   276,     0,     0,     0,
     334,   300,     0,     0,     0,     0,     0,   298,     0,   243,
       0,   344,   276,   355,     0,   276,   337,   353,     0,     0,
       0,     0,   294,     0,   301,     0,     0,     0,     0,   335,
     297,   296,   295,   293,   299,   343,     0,     0,   331,   276,
       0,   345,     0,     0,     0,   330,     0,   346,     0,   332,
       0,   333
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     3,     4,    68,   194,   233,   137,     5,    59,    69,
      70,    71,   268,   269,   270,   203,   138,   234,   139,   154,
     155,   156,   157,   158,   143,   144,   271,   335,   284,   285,
     101,   102,   161,   176,   249,   250,   168,   231,   478,   241,
     173,   242,   232,   358,   466,   359,   360,   103,   298,   345,
     104,   105,   106,   174,   107,   188,   189,   190,   191,   192,
     362,   313,   255,   256,   395,   109,   348,   396,   397,   111,
     112,   166,   179,   398,   399,   126,   400,    72,   145,   425,
     459,   460,   489,   277,   527,   415,   503,   217,   416,   587,
     647,   630,   588,   417,   589,   377,   557,   525,   504,   521,
     536,   548,   518,   505,   550,   522,   619,   528,   561,   510,
     514,   515,   286,   385,    73,    74
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -612
static const yytype_int16 yypact[] =
{
     279,  1315,  1315,    45,  -612,   279,  -612,  -612,  -612,  -612,
    -612,  -612,  -612,  -612,  -612,  -612,  -612,  -612,  -612,  -612,
    -612,  -612,  -612,  -612,  -612,  -612,  -612,  -612,  -612,  -612,
    -612,  -612,  -612,  -612,  -612,  -612,  -612,  -612,  -612,  -612,
    -612,  -612,  -612,  -612,  -612,  -612,  -612,  -612,  -612,  -612,
    -612,  -612,  -612,    83,    83,  -612,  -612,  -612,   425,  -612,
    -612,  -612,     8,  1315,   129,  1315,  1315,   117,   893,    56,
     942,   425,  -612,  -612,  -612,  1405,    40,   111,  -612,    98,
    -612,  -612,  -612,  -612,    58,   963,   143,   143,   -10,   111,
     109,   109,   109,   109,   122,   125,  1315,   167,   138,   425,
    -612,  -612,  -612,  -612,  -612,  -612,  -612,  -612,   252,  -612,
    -612,  -612,  -612,   182,  -612,  -612,  -612,  -612,  -612,  -612,
    -612,  -612,  -612,  -612,  -612,  -612,  -612,  -612,  -612,  1405,
    -612,   131,  -612,  -612,  -612,  -612,   250,   151,  -612,  -612,
     197,   208,   215,    -4,  -612,   111,   425,    98,   218,    41,
      58,   226,   526,  1423,   197,   208,   215,  -612,    10,   111,
    -612,   111,   111,   237,   111,   228,  -612,     7,  1315,  1315,
    1315,  1315,  1102,   222,   223,   106,  1315,  -612,  -612,  -612,
     449,   238,   109,   109,   109,   109,   222,   125,  -612,  -612,
    -612,  -612,  -612,  -612,  -612,   276,  -612,  -612,  -612,   231,
    -612,  -612,  1391,  -612,  -612,  -612,  -612,  -612,  -612,  1315,
     239,   264,    58,   262,    58,   240,  -612,   246,   241,   -16,
    -612,   245,  -612,   -29,    76,   100,   235,   130,   111,  -612,
    -612,   243,   249,   251,   256,   256,   256,   256,  -612,  1315,
     268,   254,   272,  1173,  1315,   285,  1315,  -612,  -612,   273,
     258,   284,  1315,    49,  1315,   286,   282,   182,  1315,  1315,
    1315,  1315,  1315,  1315,  -612,  -612,  -612,  -612,   289,  -612,
     283,  -612,   251,  -612,  -612,   291,   298,   295,   293,    58,
    -612,   111,  1315,  -612,   299,  -612,    58,   143,  1391,   143,
     143,  1391,   143,  -612,  -612,     7,  -612,   111,   157,   157,
     157,   157,   297,  -612,   285,  -612,   256,   256,  -612,   106,
     368,   300,   185,  -612,   308,   449,  -612,  -612,   256,   256,
     256,   256,   256,   178,  1391,  -612,   316,    58,   262,    58,
      58,  -612,   -29,   318,  -612,   313,  -612,   322,   326,   324,
     111,   329,   332,  -612,   333,  -612,  -612,   463,  -612,  -612,
    -612,  -612,  -612,  -612,   157,   157,  -612,  -612,  1423,     6,
     335,  1423,  -612,  -612,  -612,  -612,  -612,  -612,   157,   157,
     157,   157,   157,  -612,   368,  -612,   815,  -612,  -612,  -612,
    -612,  -612,  -612,   334,  -612,  -612,  -612,   336,  -612,    92,
     337,  -612,   111,  -612,   670,   369,   340,   346,   463,  -612,
    -612,  -612,  -612,  1315,  -612,  -612,  -612,  -612,  -612,  -612,
    -612,  -612,   345,  -612,  1315,    58,   347,   344,  1423,   143,
     143,   143,  -612,  -612,   909,  1031,  -612,   182,  -612,  -612,
     338,   371,    27,   356,  1423,  -612,   363,   365,   366,   370,
    -612,  -612,  -612,  -612,  -612,  -612,  -612,  -612,  -612,  -612,
    -612,  -612,  -612,  -612,   395,  -612,   372,  -612,  -612,   373,
     374,   375,   316,  1315,  -612,   380,   381,  -612,  -612,   220,
    -612,  -612,  -612,  -612,  -612,  -612,  -612,  -612,  -612,   417,
    -612,   961,   522,   316,  -612,  -612,  -612,  -612,    98,  -612,
    1315,  -612,  -612,   378,   383,   378,   420,   400,   423,   378,
     404,  -612,   142,    58,  -612,  -612,  -612,   471,   316,  -612,
      58,   440,    58,    39,   418,   559,   577,  -612,   421,    58,
    1368,   422,   327,   226,   411,   522,   431,  -612,   432,   419,
     435,  -612,    58,   420,   304,  -612,   443,   506,    58,   435,
     378,   436,   378,   448,   423,   378,   450,    58,   453,  1368,
    -612,   182,  -612,  -612,   470,  -612,   328,   421,    58,   378,
    -612,   596,   313,  -612,  -612,   454,  -612,  -612,   226,   719,
      58,   473,    58,   577,   421,    58,  1368,   226,  -612,  -612,
    -612,  -612,  -612,  -612,  -612,  -612,  1315,   457,   455,   451,
      58,   460,    58,   142,  -612,   316,  -612,  -612,   142,   486,
     462,   456,   435,   464,    58,   435,   466,  -612,   465,  1423,
    1340,  -612,   226,    58,   480,   468,  -612,   469,   726,  -612,
      58,   378,   737,  -612,   226,   774,  -612,  1315,  1315,    58,
     477,  -612,  1315,   435,    58,  -612,   486,   142,  -612,   483,
      58,   142,  -612,  -612,   142,   486,  -612,    67,    52,   475,
    1315,   485,   785,   461,  -612,   494,    58,   487,   495,   496,
    -612,  -612,  1315,  1244,   497,  1315,  1315,  -612,    81,  -612,
     142,  -612,    58,  -612,   435,    58,  -612,   486,   161,   489,
     212,  1315,  -612,   147,  -612,   499,   435,   792,   503,  -612,
    -612,  -612,  -612,  -612,  -612,  -612,   840,   142,  -612,    58,
     142,  -612,   507,   435,   517,  -612,   847,  -612,   142,  -612,
     523,  -612
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -612,  -612,   594,  -612,  -249,    -1,   -61,   540,   555,   -56,
    -612,  -612,  -612,  -186,  -612,  -195,  -612,  -140,   -75,   -70,
     -69,   -68,  -171,   458,   481,  -612,   -81,  -612,  -612,  -243,
    -612,  -612,   -76,   426,   303,  -612,   -74,   319,  -612,  -612,
     441,   312,  -612,   186,  -612,  -612,  -301,  -612,   -34,   236,
    -612,  -612,  -612,  -123,  -612,  -612,  -612,  -612,  -612,  -612,
    -612,   315,  -612,   321,   562,  -612,   -30,   247,   565,  -612,
    -612,   406,  -612,  -612,  -612,  -612,   242,  -612,   221,  -612,
     166,  -612,  -612,   320,   -82,    42,   -57,  -490,  -612,  -612,
    -611,  -612,  -612,  -321,    44,  -441,  -612,  -612,   127,  -502,
      82,  -523,   112,  -495,  -612,  -397,  -552,  -472,  -526,  -471,
    -612,   124,   145,    97,  -612,  -612
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -318
static const yytype_int16 yytable[] =
{
      53,    54,   151,    79,   159,   140,   141,   142,   317,   253,
      84,   162,   164,   569,   165,   127,   147,   169,   170,   171,
     552,   482,   220,   357,   512,   549,   579,   160,   519,   235,
     236,   237,   565,   553,   207,   567,   251,   229,   296,   668,
     530,   334,   507,   181,   357,    55,   148,   539,   220,    75,
     244,   678,   680,   606,   549,   683,   283,   230,   404,   140,
     141,   142,    76,   262,    80,    81,   221,   526,   215,   570,
     209,   572,   531,   412,   575,   163,   622,   326,   597,   625,
     208,   549,   218,  -162,   654,   506,   591,   607,   592,   382,
     210,   615,   221,   660,   222,   177,   617,   436,   223,   254,
     224,   225,   338,   227,   306,   341,   307,   652,   258,   259,
     260,   261,   468,   472,   469,   149,   146,   212,   535,   537,
     614,   376,   632,   213,   283,   689,   214,  -187,   506,  -187,
     275,   666,   278,   113,   643,   655,   312,   468,   375,   658,
     150,   633,   659,   492,   253,   247,   248,   661,   687,   662,
     640,   148,   663,   165,   616,   664,   665,    57,   287,    58,
     696,   684,   653,   662,   594,    78,   663,   148,   685,   664,
     665,   240,   599,   148,   420,   148,   537,   706,   464,   288,
     160,    77,   289,    78,   493,   494,   495,   496,   497,   498,
     499,    82,   196,    83,   167,   702,   197,   331,   704,  -283,
     686,   299,   300,   301,   336,   148,   710,   172,   272,   291,
     175,   337,   292,   339,   340,   500,   342,   180,    83,  -283,
     332,   638,   344,   178,  -283,   642,   148,   694,   645,   662,
     202,   346,   663,   347,   254,   664,   665,   365,   302,    78,
     363,   364,   240,   662,   690,   378,   663,   380,   381,   664,
     665,   311,   373,   314,   374,   671,   193,   318,   319,   320,
     321,   322,   323,   182,   183,   184,   185,   186,   187,   349,
     350,   351,   354,   355,    78,   486,   487,   403,   204,   389,
     406,   333,     1,     2,   368,   369,   370,   371,   372,   205,
     698,   264,   265,   211,   662,   414,   206,   663,   692,   701,
     664,   665,   580,   216,   226,   492,   228,   243,   245,   709,
     198,   199,   200,   201,   207,   257,   273,   344,   274,   276,
     280,   290,   281,   279,   401,   402,   282,   295,   492,   294,
     202,   297,   304,   433,   582,   238,   309,   414,   407,   408,
     409,   410,   411,   437,   438,   439,   493,   494,   495,   496,
     497,   498,   499,   414,   303,   140,   141,   142,   305,   308,
     310,  -283,   316,   325,   315,   128,   153,   324,   327,   493,
     494,   495,   496,   497,   498,   499,   328,   500,   329,   330,
      83,   564,    78,   352,   283,   357,  -283,   361,   130,   131,
     132,   133,   134,   135,   136,   312,   583,   584,   376,   384,
     500,   383,   430,    83,  -315,   386,   387,   388,   488,   390,
     394,   392,   405,   432,   585,   391,   418,   427,   419,   421,
     428,   523,   431,   465,   462,   434,    61,   435,    -5,    -5,
      62,    -5,    -5,    -5,    -5,    -5,    -5,    -5,    -5,    -5,
      -5,    -5,   471,    -5,    -5,   467,   473,    -5,   474,   475,
     562,   477,   481,   476,   490,   485,   568,    -9,   479,   480,
     509,   538,   483,   547,   393,   577,   484,   252,   511,    85,
      86,    87,    88,    89,   513,   586,   516,    63,    64,   517,
     520,    96,    97,    65,    66,    98,   128,   153,   600,   508,
     602,   524,   547,   605,   529,    67,   533,    83,   554,   551,
     590,    -5,   -62,    78,   394,   559,   558,   492,   612,   130,
     131,   132,   133,   134,   135,   136,   556,   604,   560,   547,
     566,   571,   624,   492,   573,   581,   576,   601,   628,   586,
     578,   596,   609,   610,   613,   618,   620,   611,   639,   672,
    -204,   623,   621,   626,   627,   635,   636,   649,   493,   494,
     495,   496,   497,   498,   499,   634,   650,   656,   657,   669,
     492,   667,   675,   128,   493,   494,   495,   496,   497,   498,
     499,   673,   676,   677,   674,   691,   695,   681,   492,   500,
      78,   699,    83,  -317,   705,   608,   130,   131,   132,   133,
     134,   135,   136,   688,   707,   500,   501,   492,   502,    56,
     711,   493,   494,   495,   496,   497,   498,   499,   100,    60,
     195,   219,   356,   263,   343,   246,   353,   703,   470,   493,
     494,   495,   496,   497,   498,   499,   646,   648,   422,   366,
     108,   651,   500,   110,   293,   534,   367,   426,   493,   494,
     495,   496,   497,   498,   499,   429,   463,   491,   379,   646,
     500,   629,   555,    83,   631,   603,   574,   563,   532,   595,
       0,   646,   646,     0,   682,   646,     0,     0,     0,   500,
       0,   423,   593,  -247,  -247,  -247,     0,  -247,  -247,  -247,
     693,  -247,  -247,  -247,  -247,  -247,     0,     0,     0,  -247,
    -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,
    -247,     0,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,
    -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,
     492,  -247,     0,  -247,  -247,     0,     0,   492,     0,     0,
    -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,   492,     0,
    -247,  -247,  -247,  -247,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   424,     0,     0,     0,     0,
       0,   493,   494,   495,   496,   497,   498,   499,   493,   494,
     495,   496,   497,   498,   499,   492,     0,     0,     0,   493,
     494,   495,   496,   497,   498,   499,   492,     0,     0,     0,
       0,     0,   500,   492,     0,   598,     0,     0,     0,   500,
       0,     0,   637,     0,     0,     0,     0,     0,     0,     0,
     500,     0,     0,   641,     0,     0,   493,   494,   495,   496,
     497,   498,   499,     0,     0,     0,     0,   493,   494,   495,
     496,   497,   498,   499,   493,   494,   495,   496,   497,   498,
     499,   492,     0,     0,     0,     0,     0,   500,   492,     0,
     644,     0,   128,   153,     0,     0,     0,     0,   500,     0,
       0,   670,     0,     0,     0,   500,     0,     0,   697,    78,
       0,     0,     0,     0,     0,   130,   131,   132,   133,   134,
     135,   136,   493,   494,   495,   496,   497,   498,   499,   493,
     494,   495,   496,   497,   498,   499,     1,     2,   413,    85,
      86,    87,    88,    89,    90,    91,    92,    93,    94,    95,
     440,    96,    97,   500,     0,    98,   700,     0,     0,     0,
     500,     0,     0,   708,     0,     0,     0,     0,     0,     0,
     441,     0,   442,   443,   444,   445,   446,   447,     0,     0,
     448,   449,   450,   451,   452,   453,     0,     0,   114,   115,
     116,   117,     0,   118,   119,   120,   121,   122,     0,   454,
     455,     0,   440,     0,     0,     0,     0,     0,     0,    99,
       0,     0,     0,     0,     0,     0,   456,     0,     0,   152,
     457,   458,   441,   123,   442,   443,   444,   445,   446,   447,
       0,     0,   448,   449,   450,   451,   452,   453,     0,     0,
     128,   153,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   454,   455,     0,     0,     0,   124,    78,     0,   125,
       0,     0,     0,   130,   131,   132,   133,   134,   135,   136,
       0,     0,   457,   458,     6,     7,     8,     0,     9,    10,
      11,     0,    12,    13,    14,    15,    16,     0,     0,     0,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,     0,    29,    30,    31,    32,    33,   128,   129,
      34,    35,    36,    37,    38,    39,    40,    41,    42,    43,
      44,     0,    45,     0,    46,   461,     0,     0,     0,     0,
       0,   130,   131,   132,   133,   134,   135,   136,    48,     0,
       0,    49,    50,    51,    52,     6,     7,     8,     0,     9,
      10,    11,     0,    12,    13,    14,    15,    16,     0,     0,
       0,    17,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,     0,    29,    30,    31,    32,    33,     0,
       0,    34,    35,    36,    37,    38,    39,    40,    41,    42,
      43,    44,   238,    45,     0,    46,    47,   239,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    48,
       0,     0,    49,    50,    51,    52,     6,     7,     8,     0,
       9,    10,    11,     0,    12,    13,    14,    15,    16,     0,
       0,     0,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,     0,    29,    30,    31,    32,    33,
       0,     0,    34,    35,    36,    37,    38,    39,    40,    41,
      42,    43,    44,     0,    45,     0,    46,    47,   239,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      48,     0,     0,    49,    50,    51,    52,     6,     7,     8,
       0,     9,    10,    11,     0,    12,    13,    14,    15,    16,
       0,     0,     0,    17,    18,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,     0,    29,    30,    31,    32,
      33,     0,     0,    34,    35,    36,    37,    38,    39,    40,
      41,    42,    43,    44,     0,    45,     0,    46,    47,   679,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    48,     0,     0,    49,    50,    51,    52,     6,     7,
       8,     0,     9,    10,    11,     0,    12,    13,    14,    15,
      16,     0,     0,     0,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,   582,    29,    30,    31,
      32,    33,     0,     0,    34,    35,    36,    37,    38,    39,
      40,    41,    42,    43,    44,     0,    45,     0,    46,    47,
       0,     0,     0,     0,     0,     0,     0,   128,   153,     0,
       0,     0,    48,     0,     0,    49,    50,    51,    52,     0,
       0,     0,     0,     0,    78,     0,     0,     0,     0,     0,
     130,   131,   132,   133,   134,   135,   136,     0,   583,   584,
     540,   541,   542,   496,   543,   544,   545,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   128,   153,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   546,   128,   129,    83,    78,   266,   267,     0,     0,
       0,   130,   131,   132,   133,   134,   135,   136,     0,    78,
     128,   153,     0,     0,     0,   130,   131,   132,   133,   134,
     135,   136,     0,     0,     0,     0,     0,    78,     0,     0,
       0,     0,     0,   130,   131,   132,   133,   134,   135,   136
};

static const yytype_int16 yycheck[] =
{
       1,     2,    84,    64,    85,    75,    75,    75,   257,   180,
      67,    87,    88,   539,    89,    71,    77,    91,    92,    93,
     522,   462,    38,    17,   495,   520,   549,    37,   499,   169,
     170,   171,   534,   523,    38,   537,   176,    30,   233,   650,
     512,   284,   483,    99,    17,     0,    75,   519,    38,    41,
     173,   662,   663,   576,   549,   666,    85,    50,   359,   129,
     129,   129,    63,   186,    65,    66,    82,   508,   150,   540,
     145,   542,   513,   374,   545,    85,   602,   272,   568,   605,
      84,   576,   152,    77,   636,   482,   558,   577,   559,   332,
     146,   593,    82,   645,    84,    96,   598,   418,   159,   180,
     161,   162,   288,   164,   244,   291,   246,   633,   182,   183,
     184,   185,    85,   434,    87,    57,    76,    76,   515,   516,
     592,    82,   612,    82,    85,   677,    85,    78,   525,    80,
     212,    79,   214,    77,   624,   637,    87,    85,   324,   641,
      82,   613,   644,     1,   315,    39,    40,    80,   674,    82,
     621,    75,    85,   228,   595,    88,    89,    74,    82,    76,
     686,    80,   634,    82,   561,    54,    85,    75,   670,    88,
      89,   172,   569,    75,    82,    75,   573,   703,   427,    79,
      37,    52,    82,    54,    42,    43,    44,    45,    46,    47,
      48,    74,    61,    76,    85,   697,    65,   279,   700,    57,
     672,   235,   236,   237,   286,    75,   708,    85,   209,    79,
      85,   287,    82,   289,   290,    73,   292,    79,    76,    77,
     281,   618,   297,    56,    82,   622,    75,    80,   625,    82,
      79,    74,    85,    76,   315,    88,    89,   312,   239,    54,
      55,    56,   243,    82,    83,   327,    85,   329,   330,    88,
      89,   252,    74,   254,    76,   652,    74,   258,   259,   260,
     261,   262,   263,    11,    12,    13,    14,    15,    16,   299,
     300,   301,   306,   307,    54,    55,    56,   358,    81,   340,
     361,   282,     3,     4,   318,   319,   320,   321,   322,    81,
     687,    60,    61,    75,    82,   376,    81,    85,    86,   696,
      88,    89,   551,    77,    67,     1,    78,    85,    85,   706,
      60,    61,    62,    63,    38,    77,    77,   392,    54,    57,
      74,    86,    81,    83,   354,   355,    81,    78,     1,    86,
      79,    75,    78,   415,     6,    50,    78,   418,   368,   369,
     370,   371,   372,   419,   420,   421,    42,    43,    44,    45,
      46,    47,    48,   434,    86,   425,   425,   425,    86,    86,
      76,    57,    80,    80,    78,    37,    38,    78,    77,    42,
      43,    44,    45,    46,    47,    48,    78,    73,    83,    86,
      76,    77,    54,    86,    85,    17,    82,    87,    60,    61,
      62,    63,    64,    65,    66,    87,    68,    69,    82,    86,
      73,    83,   403,    76,    77,    83,    80,    83,   469,    80,
      41,    78,    77,   414,    86,    83,    82,    77,    82,    82,
      74,   503,    77,    85,   425,    78,     1,    83,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    86,    18,    19,    74,    83,    22,    83,    83,
     532,    56,    78,    83,    37,    74,   538,    82,    86,    86,
      82,   518,   463,   520,     1,   547,    86,    18,    85,     6,
       7,     8,     9,    10,    54,   556,    76,    52,    53,    56,
      76,    18,    19,    58,    59,    22,    37,    38,   570,   490,
     572,    20,   549,   575,    54,    70,    78,    76,    87,    77,
     557,    76,    77,    54,    41,    86,    74,     1,   590,    60,
      61,    62,    63,    64,    65,    66,    85,   574,    83,   576,
      77,    85,   604,     1,    76,    55,    76,    54,   609,   610,
      77,    77,    75,    78,    74,    49,    74,    86,   620,    78,
      77,    77,    86,    77,    79,    77,    77,   629,    42,    43,
      44,    45,    46,    47,    48,    75,    79,    74,   640,    74,
       1,    86,    75,    37,    42,    43,    44,    45,    46,    47,
      48,    77,    77,    77,   656,    86,    77,    80,     1,    73,
      54,    78,    76,    77,    77,   586,    60,    61,    62,    63,
      64,    65,    66,   675,    77,    73,    74,     1,    76,     5,
      77,    42,    43,    44,    45,    46,    47,    48,    68,    54,
     129,   153,   309,   187,   295,   174,   304,   699,   432,    42,
      43,    44,    45,    46,    47,    48,   627,   628,   392,   314,
      68,   632,    73,    68,   228,    76,   315,   395,    42,    43,
      44,    45,    46,    47,    48,   398,   425,   481,   328,   650,
      73,   609,   525,    76,   610,   573,   544,   533,   513,   562,
      -1,   662,   663,    -1,   665,   666,    -1,    -1,    -1,    73,
      -1,     1,    76,     3,     4,     5,    -1,     7,     8,     9,
     681,    11,    12,    13,    14,    15,    -1,    -1,    -1,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    -1,    32,    33,    34,    35,    36,    37,    38,    39,
      40,    41,    42,    43,    44,    45,    46,    47,    48,    49,
       1,    51,    -1,    53,    54,    -1,    -1,     1,    -1,    -1,
      60,    61,    62,    63,    64,    65,    66,    67,     1,    -1,
      70,    71,    72,    73,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    85,    -1,    -1,    -1,    -1,
      -1,    42,    43,    44,    45,    46,    47,    48,    42,    43,
      44,    45,    46,    47,    48,     1,    -1,    -1,    -1,    42,
      43,    44,    45,    46,    47,    48,     1,    -1,    -1,    -1,
      -1,    -1,    73,     1,    -1,    76,    -1,    -1,    -1,    73,
      -1,    -1,    76,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      73,    -1,    -1,    76,    -1,    -1,    42,    43,    44,    45,
      46,    47,    48,    -1,    -1,    -1,    -1,    42,    43,    44,
      45,    46,    47,    48,    42,    43,    44,    45,    46,    47,
      48,     1,    -1,    -1,    -1,    -1,    -1,    73,     1,    -1,
      76,    -1,    37,    38,    -1,    -1,    -1,    -1,    73,    -1,
      -1,    76,    -1,    -1,    -1,    73,    -1,    -1,    76,    54,
      -1,    -1,    -1,    -1,    -1,    60,    61,    62,    63,    64,
      65,    66,    42,    43,    44,    45,    46,    47,    48,    42,
      43,    44,    45,    46,    47,    48,     3,     4,    83,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
       1,    18,    19,    73,    -1,    22,    76,    -1,    -1,    -1,
      73,    -1,    -1,    76,    -1,    -1,    -1,    -1,    -1,    -1,
      21,    -1,    23,    24,    25,    26,    27,    28,    -1,    -1,
      31,    32,    33,    34,    35,    36,    -1,    -1,     6,     7,
       8,     9,    -1,    11,    12,    13,    14,    15,    -1,    50,
      51,    -1,     1,    -1,    -1,    -1,    -1,    -1,    -1,    76,
      -1,    -1,    -1,    -1,    -1,    -1,    67,    -1,    -1,    16,
      71,    72,    21,    41,    23,    24,    25,    26,    27,    28,
      -1,    -1,    31,    32,    33,    34,    35,    36,    -1,    -1,
      37,    38,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    50,    51,    -1,    -1,    -1,    74,    54,    -1,    77,
      -1,    -1,    -1,    60,    61,    62,    63,    64,    65,    66,
      -1,    -1,    71,    72,     3,     4,     5,    -1,     7,     8,
       9,    -1,    11,    12,    13,    14,    15,    -1,    -1,    -1,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    -1,    32,    33,    34,    35,    36,    37,    38,
      39,    40,    41,    42,    43,    44,    45,    46,    47,    48,
      49,    -1,    51,    -1,    53,    54,    -1,    -1,    -1,    -1,
      -1,    60,    61,    62,    63,    64,    65,    66,    67,    -1,
      -1,    70,    71,    72,    73,     3,     4,     5,    -1,     7,
       8,     9,    -1,    11,    12,    13,    14,    15,    -1,    -1,
      -1,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      28,    29,    30,    -1,    32,    33,    34,    35,    36,    -1,
      -1,    39,    40,    41,    42,    43,    44,    45,    46,    47,
      48,    49,    50,    51,    -1,    53,    54,    55,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    67,
      -1,    -1,    70,    71,    72,    73,     3,     4,     5,    -1,
       7,     8,     9,    -1,    11,    12,    13,    14,    15,    -1,
      -1,    -1,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    -1,    32,    33,    34,    35,    36,
      -1,    -1,    39,    40,    41,    42,    43,    44,    45,    46,
      47,    48,    49,    -1,    51,    -1,    53,    54,    55,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      67,    -1,    -1,    70,    71,    72,    73,     3,     4,     5,
      -1,     7,     8,     9,    -1,    11,    12,    13,    14,    15,
      -1,    -1,    -1,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    -1,    32,    33,    34,    35,
      36,    -1,    -1,    39,    40,    41,    42,    43,    44,    45,
      46,    47,    48,    49,    -1,    51,    -1,    53,    54,    55,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    67,    -1,    -1,    70,    71,    72,    73,     3,     4,
       5,    -1,     7,     8,     9,    -1,    11,    12,    13,    14,
      15,    -1,    -1,    -1,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,     6,    32,    33,    34,
      35,    36,    -1,    -1,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    -1,    51,    -1,    53,    54,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    37,    38,    -1,
      -1,    -1,    67,    -1,    -1,    70,    71,    72,    73,    -1,
      -1,    -1,    -1,    -1,    54,    -1,    -1,    -1,    -1,    -1,
      60,    61,    62,    63,    64,    65,    66,    -1,    68,    69,
      42,    43,    44,    45,    46,    47,    48,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    37,    38,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    73,    37,    38,    76,    54,    55,    56,    -1,    -1,
      -1,    60,    61,    62,    63,    64,    65,    66,    -1,    54,
      37,    38,    -1,    -1,    -1,    60,    61,    62,    63,    64,
      65,    66,    -1,    -1,    -1,    -1,    -1,    54,    -1,    -1,
      -1,    -1,    -1,    60,    61,    62,    63,    64,    65,    66
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     3,     4,    91,    92,    97,     3,     4,     5,     7,
       8,     9,    11,    12,    13,    14,    15,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    32,
      33,    34,    35,    36,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    51,    53,    54,    67,    70,
      71,    72,    73,    95,    95,     0,    92,    74,    76,    98,
      98,     1,     5,    52,    53,    58,    59,    70,    93,    99,
     100,   101,   167,   204,   205,    41,    95,    52,    54,    96,
      95,    95,    74,    76,   176,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    18,    19,    22,    76,
      97,   120,   121,   137,   140,   141,   142,   144,   154,   155,
     158,   159,   160,    77,     6,     7,     8,     9,    11,    12,
      13,    14,    15,    41,    74,    77,   165,    99,    37,    38,
      60,    61,    62,    63,    64,    65,    66,    96,   106,   108,
     109,   110,   111,   114,   115,   168,    76,    96,    75,    57,
      82,   174,    16,    38,   109,   110,   111,   112,   113,   116,
      37,   122,   122,    85,   122,   108,   161,    85,   126,   126,
     126,   126,    85,   130,   143,    85,   123,    95,    56,   162,
      79,    99,    11,    12,    13,    14,    15,    16,   145,   146,
     147,   148,   149,    74,    94,   114,    61,    65,    60,    61,
      62,    63,    79,   105,    81,    81,    81,    38,    84,   108,
      99,    75,    76,    82,    85,   174,    77,   177,   109,   113,
      38,    82,    84,    96,    96,    96,    67,    96,    78,    30,
      50,   127,   132,    95,   107,   107,   107,   107,    50,    55,
      95,   129,   131,    85,   143,    85,   130,    39,    40,   124,
     125,   107,    18,   112,   116,   152,   153,    77,   126,   126,
     126,   126,   143,   123,    60,    61,    55,    56,   102,   103,
     104,   116,    95,    77,    54,   174,    57,   173,   174,    83,
      74,    81,    81,    85,   118,   119,   202,    82,    79,    82,
      86,    79,    82,   161,    86,    78,   105,    75,   138,   138,
     138,   138,    95,    86,    78,    86,   107,   107,    86,    78,
      76,    95,    87,   151,    95,    78,    80,    94,    95,    95,
      95,    95,    95,    95,    78,    80,   105,    77,    78,    83,
      86,   174,    96,    95,   119,   117,   174,   122,   103,   122,
     122,   103,   122,   127,   108,   139,    74,    76,   156,   156,
     156,   156,    86,   131,   138,   138,   124,    17,   133,   135,
     136,    87,   150,    55,    56,   108,   151,   153,   138,   138,
     138,   138,   138,    74,    76,   103,    82,   185,   174,   173,
     174,   174,   119,    83,    86,   203,    83,    80,    83,    96,
      80,    83,    78,     1,    41,   154,   157,   158,   163,   164,
     166,   156,   156,   116,   136,    77,   116,   156,   156,   156,
     156,   156,   136,    83,   116,   175,   178,   183,    82,    82,
      82,    82,   139,     1,    85,   169,   166,    77,    74,   157,
      95,    77,    95,   174,    78,    83,   183,   122,   122,   122,
       1,    21,    23,    24,    25,    26,    27,    28,    31,    32,
      33,    34,    35,    36,    50,    51,    67,    71,    72,   170,
     171,    54,    95,   168,    94,    85,   134,    74,    85,    87,
     133,    86,   183,    83,    83,    83,    83,    56,   128,    86,
      86,    78,   185,    95,    86,    74,    55,    56,    96,   172,
      37,   170,     1,    42,    43,    44,    45,    46,    47,    48,
      73,    74,    76,   176,   188,   193,   195,   185,    95,    82,
     199,    85,   199,    54,   200,   201,    76,    56,   192,   199,
      76,   189,   195,   174,    20,   187,   185,   174,   197,    54,
     197,   185,   202,    78,    76,   195,   190,   195,   176,   197,
      42,    43,    44,    46,    47,    48,    73,   176,   191,   193,
     194,    77,   189,   177,    87,   188,    85,   186,    74,    86,
      83,   198,   174,   201,    77,   189,    77,   189,   174,   198,
     199,    85,   199,    76,   192,   199,    76,   174,    77,   191,
      94,    55,     6,    68,    69,    86,   116,   179,   182,   184,
     176,   197,   199,    76,   195,   203,    77,   177,    76,   195,
     174,    54,   174,   190,   176,   174,   191,   177,    95,    75,
      78,    86,   174,    74,   197,   189,   185,   189,    49,   196,
      74,    86,   198,    77,   174,   198,    77,    79,   116,   175,
     181,   184,   177,   197,    75,    77,    77,    76,   195,   174,
     199,    76,   195,   177,    76,   195,    95,   180,    95,   174,
      79,    95,   198,   197,   196,   189,    74,   174,   189,   189,
     196,    80,    82,    85,    88,    89,    79,    86,   180,    74,
      76,   195,    78,    77,   174,    75,    77,    77,   180,    55,
     180,    80,    95,   180,    80,   189,   197,   198,   174,   196,
      83,    86,    86,    95,    80,    77,   198,    76,   195,    78,
      76,   195,   189,   174,   189,    77,   198,    77,    76,   195,
     189,    77
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
#line 193 "xi-grammar.y"
    { (yyval.modlist) = (yyvsp[(1) - (1)].modlist); modlist = (yyvsp[(1) - (1)].modlist); }
    break;

  case 3:
#line 197 "xi-grammar.y"
    { 
		  (yyval.modlist) = 0; 
		}
    break;

  case 4:
#line 201 "xi-grammar.y"
    { (yyval.modlist) = new AstChildren<Module>(lineno, (yyvsp[(1) - (2)].module), (yyvsp[(2) - (2)].modlist)); }
    break;

  case 5:
#line 205 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 6:
#line 207 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 7:
#line 211 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 8:
#line 213 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 9:
#line 218 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 10:
#line 219 "xi-grammar.y"
    { ReservedWord(MODULE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 11:
#line 220 "xi-grammar.y"
    { ReservedWord(MAINMODULE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 12:
#line 221 "xi-grammar.y"
    { ReservedWord(EXTERN, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 13:
#line 223 "xi-grammar.y"
    { ReservedWord(INITCALL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 14:
#line 224 "xi-grammar.y"
    { ReservedWord(INITNODE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 15:
#line 225 "xi-grammar.y"
    { ReservedWord(INITPROC, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 16:
#line 227 "xi-grammar.y"
    { ReservedWord(CHARE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 17:
#line 228 "xi-grammar.y"
    { ReservedWord(MAINCHARE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 18:
#line 229 "xi-grammar.y"
    { ReservedWord(GROUP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 19:
#line 230 "xi-grammar.y"
    { ReservedWord(NODEGROUP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 20:
#line 231 "xi-grammar.y"
    { ReservedWord(ARRAY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 21:
#line 235 "xi-grammar.y"
    { ReservedWord(INCLUDE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 22:
#line 236 "xi-grammar.y"
    { ReservedWord(STACKSIZE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 23:
#line 237 "xi-grammar.y"
    { ReservedWord(THREADED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 24:
#line 238 "xi-grammar.y"
    { ReservedWord(TEMPLATE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 25:
#line 239 "xi-grammar.y"
    { ReservedWord(SYNC, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 26:
#line 240 "xi-grammar.y"
    { ReservedWord(IGET, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 27:
#line 241 "xi-grammar.y"
    { ReservedWord(EXCLUSIVE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 28:
#line 242 "xi-grammar.y"
    { ReservedWord(IMMEDIATE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 29:
#line 243 "xi-grammar.y"
    { ReservedWord(SKIPSCHED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 30:
#line 244 "xi-grammar.y"
    { ReservedWord(INLINE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 31:
#line 245 "xi-grammar.y"
    { ReservedWord(VIRTUAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 32:
#line 246 "xi-grammar.y"
    { ReservedWord(MIGRATABLE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 33:
#line 247 "xi-grammar.y"
    { ReservedWord(CREATEHERE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 34:
#line 248 "xi-grammar.y"
    { ReservedWord(CREATEHOME, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 35:
#line 249 "xi-grammar.y"
    { ReservedWord(NOKEEP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 36:
#line 250 "xi-grammar.y"
    { ReservedWord(NOTRACE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 37:
#line 251 "xi-grammar.y"
    { ReservedWord(APPWORK, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 38:
#line 254 "xi-grammar.y"
    { ReservedWord(PACKED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 39:
#line 255 "xi-grammar.y"
    { ReservedWord(VARSIZE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 40:
#line 256 "xi-grammar.y"
    { ReservedWord(ENTRY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 41:
#line 257 "xi-grammar.y"
    { ReservedWord(FOR, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 42:
#line 258 "xi-grammar.y"
    { ReservedWord(FORALL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 43:
#line 259 "xi-grammar.y"
    { ReservedWord(WHILE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 44:
#line 260 "xi-grammar.y"
    { ReservedWord(WHEN, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 45:
#line 261 "xi-grammar.y"
    { ReservedWord(OVERLAP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 46:
#line 262 "xi-grammar.y"
    { ReservedWord(ATOMIC, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 47:
#line 263 "xi-grammar.y"
    { ReservedWord(IF, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 48:
#line 264 "xi-grammar.y"
    { ReservedWord(ELSE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 49:
#line 266 "xi-grammar.y"
    { ReservedWord(LOCAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 50:
#line 268 "xi-grammar.y"
    { ReservedWord(USING, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 51:
#line 269 "xi-grammar.y"
    { ReservedWord(ACCEL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 52:
#line 272 "xi-grammar.y"
    { ReservedWord(ACCELBLOCK, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 53:
#line 273 "xi-grammar.y"
    { ReservedWord(MEMCRITICAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 54:
#line 274 "xi-grammar.y"
    { ReservedWord(REDUCTIONTARGET, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 55:
#line 275 "xi-grammar.y"
    { ReservedWord(CASE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 56:
#line 279 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 57:
#line 281 "xi-grammar.y"
    {
		  char *tmp = new char[strlen((yyvsp[(1) - (4)].strval))+strlen((yyvsp[(4) - (4)].strval))+3];
		  sprintf(tmp,"%s::%s", (yyvsp[(1) - (4)].strval), (yyvsp[(4) - (4)].strval));
		  (yyval.strval) = tmp;
		}
    break;

  case 58:
#line 288 "xi-grammar.y"
    { 
		    (yyval.module) = new Module(lineno, (yyvsp[(2) - (3)].strval), (yyvsp[(3) - (3)].conslist)); 
		}
    break;

  case 59:
#line 292 "xi-grammar.y"
    {  
		    (yyval.module) = new Module(lineno, (yyvsp[(2) - (3)].strval), (yyvsp[(3) - (3)].conslist)); 
		    (yyval.module)->setMain();
		}
    break;

  case 60:
#line 299 "xi-grammar.y"
    { (yyval.conslist) = 0; }
    break;

  case 61:
#line 301 "xi-grammar.y"
    { (yyval.conslist) = (yyvsp[(2) - (4)].conslist); }
    break;

  case 62:
#line 305 "xi-grammar.y"
    { (yyval.conslist) = 0; }
    break;

  case 63:
#line 307 "xi-grammar.y"
    { (yyval.conslist) = new ConstructList(lineno, (yyvsp[(1) - (2)].construct), (yyvsp[(2) - (2)].conslist)); }
    break;

  case 64:
#line 311 "xi-grammar.y"
    { (yyval.construct) = new UsingScope((yyvsp[(3) - (3)].strval), false); }
    break;

  case 65:
#line 313 "xi-grammar.y"
    { (yyval.construct) = new UsingScope((yyvsp[(2) - (2)].strval), true); }
    break;

  case 66:
#line 315 "xi-grammar.y"
    { (yyvsp[(2) - (2)].member)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].member); }
    break;

  case 67:
#line 317 "xi-grammar.y"
    { (yyvsp[(2) - (2)].message)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].message); }
    break;

  case 68:
#line 319 "xi-grammar.y"
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

  case 69:
#line 331 "xi-grammar.y"
    { if((yyvsp[(3) - (5)].conslist)) (yyvsp[(3) - (5)].conslist)->recurse<int&>((yyvsp[(1) - (5)].intval), &Construct::setExtern); (yyval.construct) = (yyvsp[(3) - (5)].conslist); }
    break;

  case 70:
#line 333 "xi-grammar.y"
    { (yyval.construct) = new Scope((yyvsp[(2) - (5)].strval), (yyvsp[(4) - (5)].conslist)); }
    break;

  case 71:
#line 335 "xi-grammar.y"
    { (yyval.construct) = (yyvsp[(1) - (2)].construct); }
    break;

  case 72:
#line 337 "xi-grammar.y"
    {
          ERROR("preceding construct must be semicolon terminated",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
    break;

  case 73:
#line 343 "xi-grammar.y"
    { (yyvsp[(2) - (2)].module)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].module); }
    break;

  case 74:
#line 345 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 75:
#line 347 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 76:
#line 349 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 77:
#line 351 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 78:
#line 353 "xi-grammar.y"
    { (yyvsp[(2) - (2)].templat)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].templat); }
    break;

  case 79:
#line 355 "xi-grammar.y"
    { (yyval.construct) = NULL; }
    break;

  case 80:
#line 357 "xi-grammar.y"
    { (yyval.construct) = NULL; }
    break;

  case 81:
#line 359 "xi-grammar.y"
    { (yyval.construct) = (yyvsp[(1) - (1)].accelBlock); }
    break;

  case 82:
#line 361 "xi-grammar.y"
    {
          ERROR("invalid construct",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
    break;

  case 83:
#line 369 "xi-grammar.y"
    { (yyval.tparam) = new TParamType((yyvsp[(1) - (1)].type)); }
    break;

  case 84:
#line 371 "xi-grammar.y"
    { (yyval.tparam) = new TParamVal((yyvsp[(1) - (1)].strval)); }
    break;

  case 85:
#line 373 "xi-grammar.y"
    { (yyval.tparam) = new TParamVal((yyvsp[(1) - (1)].strval)); }
    break;

  case 86:
#line 377 "xi-grammar.y"
    { (yyval.tparlist) = new TParamList((yyvsp[(1) - (1)].tparam)); }
    break;

  case 87:
#line 379 "xi-grammar.y"
    { (yyval.tparlist) = new TParamList((yyvsp[(1) - (3)].tparam), (yyvsp[(3) - (3)].tparlist)); }
    break;

  case 88:
#line 383 "xi-grammar.y"
    { (yyval.tparlist) = 0; }
    break;

  case 89:
#line 385 "xi-grammar.y"
    { (yyval.tparlist) = (yyvsp[(1) - (1)].tparlist); }
    break;

  case 90:
#line 389 "xi-grammar.y"
    { (yyval.tparlist) = 0; }
    break;

  case 91:
#line 391 "xi-grammar.y"
    { (yyval.tparlist) = (yyvsp[(2) - (3)].tparlist); }
    break;

  case 92:
#line 395 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("int"); }
    break;

  case 93:
#line 397 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long"); }
    break;

  case 94:
#line 399 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("short"); }
    break;

  case 95:
#line 401 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("char"); }
    break;

  case 96:
#line 403 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned int"); }
    break;

  case 97:
#line 405 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long"); }
    break;

  case 98:
#line 407 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long"); }
    break;

  case 99:
#line 409 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long long"); }
    break;

  case 100:
#line 411 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned short"); }
    break;

  case 101:
#line 413 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned char"); }
    break;

  case 102:
#line 415 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long long"); }
    break;

  case 103:
#line 417 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("float"); }
    break;

  case 104:
#line 419 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("double"); }
    break;

  case 105:
#line 421 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long double"); }
    break;

  case 106:
#line 423 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("void"); }
    break;

  case 107:
#line 426 "xi-grammar.y"
    { (yyval.ntype) = new NamedType((yyvsp[(1) - (2)].strval),(yyvsp[(2) - (2)].tparlist)); }
    break;

  case 108:
#line 427 "xi-grammar.y"
    { 
                    const char* basename, *scope;
                    splitScopedName((yyvsp[(1) - (2)].strval), &scope, &basename);
                    (yyval.ntype) = new NamedType(basename, (yyvsp[(2) - (2)].tparlist), scope);
                }
    break;

  case 109:
#line 435 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 110:
#line 437 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ntype); }
    break;

  case 111:
#line 441 "xi-grammar.y"
    { (yyval.ptype) = new PtrType((yyvsp[(1) - (2)].type)); }
    break;

  case 112:
#line 445 "xi-grammar.y"
    { (yyvsp[(1) - (2)].ptype)->indirect(); (yyval.ptype) = (yyvsp[(1) - (2)].ptype); }
    break;

  case 113:
#line 447 "xi-grammar.y"
    { (yyvsp[(1) - (2)].ptype)->indirect(); (yyval.ptype) = (yyvsp[(1) - (2)].ptype); }
    break;

  case 114:
#line 451 "xi-grammar.y"
    { (yyval.ftype) = new FuncType((yyvsp[(1) - (8)].type), (yyvsp[(4) - (8)].strval), (yyvsp[(7) - (8)].plist)); }
    break;

  case 115:
#line 455 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 116:
#line 457 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 117:
#line 459 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 118:
#line 461 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ftype); }
    break;

  case 119:
#line 463 "xi-grammar.y"
    { (yyval.type) = new ConstType((yyvsp[(2) - (2)].type)); }
    break;

  case 120:
#line 465 "xi-grammar.y"
    { (yyval.type) = new ConstType((yyvsp[(1) - (2)].type)); }
    break;

  case 121:
#line 469 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 122:
#line 471 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 123:
#line 473 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
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
    { (yyval.type) = new ReferenceType((yyvsp[(1) - (2)].type)); }
    break;

  case 127:
#line 483 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 128:
#line 487 "xi-grammar.y"
    { (yyval.type) = new ReferenceType((yyvsp[(1) - (2)].type)); }
    break;

  case 129:
#line 489 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 130:
#line 493 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 131:
#line 497 "xi-grammar.y"
    { (yyval.val) = (yyvsp[(2) - (3)].val); }
    break;

  case 132:
#line 501 "xi-grammar.y"
    { (yyval.vallist) = 0; }
    break;

  case 133:
#line 503 "xi-grammar.y"
    { (yyval.vallist) = new ValueList((yyvsp[(1) - (2)].val), (yyvsp[(2) - (2)].vallist)); }
    break;

  case 134:
#line 507 "xi-grammar.y"
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[(2) - (4)].type), (yyvsp[(3) - (4)].strval), (yyvsp[(4) - (4)].vallist)); }
    break;

  case 135:
#line 511 "xi-grammar.y"
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[(3) - (6)].type), (yyvsp[(5) - (6)].strval), (yyvsp[(6) - (6)].vallist), 1); }
    break;

  case 136:
#line 515 "xi-grammar.y"
    { (yyval.intval) = 0;}
    break;

  case 137:
#line 517 "xi-grammar.y"
    { (yyval.intval) = 0;}
    break;

  case 138:
#line 521 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 139:
#line 523 "xi-grammar.y"
    { 
		  /*
		  printf("Warning: Message attributes are being phased out.\n");
		  printf("Warning: Please remove them from interface files.\n");
		  */
		  (yyval.intval) = (yyvsp[(2) - (3)].intval); 
		}
    break;

  case 140:
#line 533 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (1)].intval); }
    break;

  case 141:
#line 535 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (3)].intval) | (yyvsp[(3) - (3)].intval); }
    break;

  case 142:
#line 539 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 143:
#line 541 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 144:
#line 545 "xi-grammar.y"
    { (yyval.cattr) = 0; }
    break;

  case 145:
#line 547 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(2) - (3)].cattr); }
    break;

  case 146:
#line 551 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (1)].cattr); }
    break;

  case 147:
#line 553 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (3)].cattr) | (yyvsp[(3) - (3)].cattr); }
    break;

  case 148:
#line 557 "xi-grammar.y"
    { python_doc = NULL; (yyval.intval) = 0; }
    break;

  case 149:
#line 559 "xi-grammar.y"
    { python_doc = (yyvsp[(1) - (1)].strval); (yyval.intval) = 0; }
    break;

  case 150:
#line 563 "xi-grammar.y"
    { (yyval.cattr) = Chare::CPYTHON; }
    break;

  case 151:
#line 567 "xi-grammar.y"
    { (yyval.cattr) = 0; }
    break;

  case 152:
#line 569 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(2) - (3)].cattr); }
    break;

  case 153:
#line 573 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (1)].cattr); }
    break;

  case 154:
#line 575 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (3)].cattr) | (yyvsp[(3) - (3)].cattr); }
    break;

  case 155:
#line 579 "xi-grammar.y"
    { (yyval.cattr) = Chare::CMIGRATABLE; }
    break;

  case 156:
#line 581 "xi-grammar.y"
    { (yyval.cattr) = Chare::CPYTHON; }
    break;

  case 157:
#line 585 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 158:
#line 587 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 159:
#line 590 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 160:
#line 592 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 161:
#line 595 "xi-grammar.y"
    { (yyval.mv) = new MsgVar((yyvsp[(2) - (5)].type), (yyvsp[(3) - (5)].strval), (yyvsp[(1) - (5)].intval), (yyvsp[(4) - (5)].intval)); }
    break;

  case 162:
#line 599 "xi-grammar.y"
    { (yyval.mvlist) = new MsgVarList((yyvsp[(1) - (1)].mv)); }
    break;

  case 163:
#line 601 "xi-grammar.y"
    { (yyval.mvlist) = new MsgVarList((yyvsp[(1) - (2)].mv), (yyvsp[(2) - (2)].mvlist)); }
    break;

  case 164:
#line 605 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, (yyvsp[(3) - (3)].ntype)); }
    break;

  case 165:
#line 607 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, (yyvsp[(3) - (6)].ntype), (yyvsp[(5) - (6)].mvlist)); }
    break;

  case 166:
#line 611 "xi-grammar.y"
    { (yyval.typelist) = 0; }
    break;

  case 167:
#line 613 "xi-grammar.y"
    { (yyval.typelist) = (yyvsp[(2) - (2)].typelist); }
    break;

  case 168:
#line 617 "xi-grammar.y"
    { (yyval.typelist) = new TypeList((yyvsp[(1) - (1)].ntype)); }
    break;

  case 169:
#line 619 "xi-grammar.y"
    { (yyval.typelist) = new TypeList((yyvsp[(1) - (3)].ntype), (yyvsp[(3) - (3)].typelist)); }
    break;

  case 170:
#line 623 "xi-grammar.y"
    { (yyval.chare) = new Chare(lineno, (yyvsp[(2) - (5)].cattr)|Chare::CCHARE, (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 171:
#line 625 "xi-grammar.y"
    { (yyval.chare) = new MainChare(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 172:
#line 629 "xi-grammar.y"
    { (yyval.chare) = new Group(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 173:
#line 633 "xi-grammar.y"
    { (yyval.chare) = new NodeGroup(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 174:
#line 637 "xi-grammar.y"
    {/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",(yyvsp[(2) - (4)].strval));
			(yyval.ntype) = new NamedType(buf); 
		}
    break;

  case 175:
#line 643 "xi-grammar.y"
    { (yyval.ntype) = new NamedType((yyvsp[(2) - (3)].strval)); }
    break;

  case 176:
#line 647 "xi-grammar.y"
    {  (yyval.chare) = new Array(lineno, (yyvsp[(2) - (6)].cattr), (yyvsp[(3) - (6)].ntype), (yyvsp[(4) - (6)].ntype), (yyvsp[(5) - (6)].typelist), (yyvsp[(6) - (6)].mbrlist)); }
    break;

  case 177:
#line 649 "xi-grammar.y"
    {  (yyval.chare) = new Array(lineno, (yyvsp[(3) - (6)].cattr), (yyvsp[(2) - (6)].ntype), (yyvsp[(4) - (6)].ntype), (yyvsp[(5) - (6)].typelist), (yyvsp[(6) - (6)].mbrlist)); }
    break;

  case 178:
#line 653 "xi-grammar.y"
    { (yyval.chare) = new Chare(lineno, (yyvsp[(2) - (5)].cattr)|Chare::CCHARE, new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist));}
    break;

  case 179:
#line 655 "xi-grammar.y"
    { (yyval.chare) = new MainChare(lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 180:
#line 659 "xi-grammar.y"
    { (yyval.chare) = new Group(lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 181:
#line 663 "xi-grammar.y"
    { (yyval.chare) = new NodeGroup( lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 182:
#line 667 "xi-grammar.y"
    { (yyval.chare) = new Array( lineno, 0, (yyvsp[(2) - (5)].ntype), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 183:
#line 671 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[(3) - (4)].strval))); }
    break;

  case 184:
#line 673 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[(3) - (7)].strval)), (yyvsp[(5) - (7)].mvlist)); }
    break;

  case 185:
#line 677 "xi-grammar.y"
    { (yyval.type) = 0; }
    break;

  case 186:
#line 679 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(2) - (2)].type); }
    break;

  case 187:
#line 683 "xi-grammar.y"
    { (yyval.strval) = 0; }
    break;

  case 188:
#line 685 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(2) - (2)].strval); }
    break;

  case 189:
#line 687 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(2) - (2)].strval); }
    break;

  case 190:
#line 689 "xi-grammar.y"
    {
		  XStr typeStr;
		  (yyvsp[(2) - (2)].ntype)->print(typeStr);
		  char *tmp = strdup(typeStr.get_string());
		  (yyval.strval) = tmp;
		}
    break;

  case 191:
#line 698 "xi-grammar.y"
    { (yyval.tvar) = new TType(new NamedType((yyvsp[(2) - (3)].strval)), (yyvsp[(3) - (3)].type)); }
    break;

  case 192:
#line 700 "xi-grammar.y"
    { (yyval.tvar) = new TFunc((yyvsp[(1) - (2)].ftype), (yyvsp[(2) - (2)].strval)); }
    break;

  case 193:
#line 702 "xi-grammar.y"
    { (yyval.tvar) = new TName((yyvsp[(1) - (3)].type), (yyvsp[(2) - (3)].strval), (yyvsp[(3) - (3)].strval)); }
    break;

  case 194:
#line 706 "xi-grammar.y"
    { (yyval.tvarlist) = new TVarList((yyvsp[(1) - (1)].tvar)); }
    break;

  case 195:
#line 708 "xi-grammar.y"
    { (yyval.tvarlist) = new TVarList((yyvsp[(1) - (3)].tvar), (yyvsp[(3) - (3)].tvarlist)); }
    break;

  case 196:
#line 712 "xi-grammar.y"
    { (yyval.tvarlist) = (yyvsp[(3) - (4)].tvarlist); }
    break;

  case 197:
#line 716 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 198:
#line 718 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 199:
#line 720 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 200:
#line 722 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 201:
#line 724 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].message)); (yyvsp[(2) - (2)].message)->setTemplate((yyval.templat)); }
    break;

  case 202:
#line 728 "xi-grammar.y"
    { (yyval.mbrlist) = 0; }
    break;

  case 203:
#line 730 "xi-grammar.y"
    { (yyval.mbrlist) = (yyvsp[(2) - (4)].mbrlist); }
    break;

  case 204:
#line 734 "xi-grammar.y"
    { 
                  if (!connectEntries.empty()) {
                    (yyval.mbrlist) = new AstChildren<Member>(connectEntries);
		  } else {
		    (yyval.mbrlist) = 0; 
                  }
		}
    break;

  case 205:
#line 742 "xi-grammar.y"
    { (yyval.mbrlist) = new AstChildren<Member>(-1, (yyvsp[(1) - (2)].member), (yyvsp[(2) - (2)].mbrlist)); }
    break;

  case 206:
#line 746 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].readonly); }
    break;

  case 207:
#line 748 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].readonly); }
    break;

  case 209:
#line 751 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].member); }
    break;

  case 210:
#line 753 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(2) - (2)].pupable); }
    break;

  case 211:
#line 755 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(2) - (2)].includeFile); }
    break;

  case 212:
#line 757 "xi-grammar.y"
    { (yyval.member) = new ClassDeclaration(lineno,(yyvsp[(2) - (2)].strval)); }
    break;

  case 213:
#line 761 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 1); }
    break;

  case 214:
#line 763 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 1); }
    break;

  case 215:
#line 765 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[(3) - (9)].strval)) + '<' +
					    ((yyvsp[(5) - (9)].tparlist))->to_string() + '>').c_str()),
				    1);
		}
    break;

  case 216:
#line 771 "xi-grammar.y"
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[(1) - (3)]).first_column, (yylsp[(1) - (3)]).last_column, (yylsp[(1) - (3)]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 1);
		}
    break;

  case 217:
#line 777 "xi-grammar.y"
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[(1) - (6)]).first_column, (yylsp[(1) - (6)]).last_column, (yylsp[(1) - (6)]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 1);
		}
    break;

  case 218:
#line 786 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 0); }
    break;

  case 219:
#line 788 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 0); }
    break;

  case 220:
#line 790 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[(3) - (9)].strval)) + '<' +
					    ((yyvsp[(5) - (9)].tparlist))->to_string() + '>').c_str()),
				    0);
		}
    break;

  case 221:
#line 796 "xi-grammar.y"
    {
                  InitCall* rtn = new InitCall(lineno, (yyvsp[(6) - (9)].strval), 0);
                  rtn->setAccel();
                  (yyval.member) = rtn;
		}
    break;

  case 222:
#line 804 "xi-grammar.y"
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[(1) - (1)].ntype),0); }
    break;

  case 223:
#line 806 "xi-grammar.y"
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[(1) - (3)].ntype),(yyvsp[(3) - (3)].pupable)); }
    break;

  case 224:
#line 809 "xi-grammar.y"
    { (yyval.includeFile) = new IncludeFile(lineno,(yyvsp[(1) - (1)].strval)); }
    break;

  case 225:
#line 813 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].member); }
    break;

  case 226:
#line 817 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].entry); }
    break;

  case 227:
#line 819 "xi-grammar.y"
    {
                  (yyvsp[(2) - (2)].entry)->tspec = (yyvsp[(1) - (2)].tvarlist);
                  (yyval.member) = (yyvsp[(2) - (2)].entry);
                }
    break;

  case 228:
#line 824 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (2)].member); }
    break;

  case 229:
#line 826 "xi-grammar.y"
    {
          ERROR("invalid SDAG member",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
    break;

  case 230:
#line 834 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 231:
#line 836 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 232:
#line 838 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 233:
#line 840 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 234:
#line 842 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 235:
#line 844 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 236:
#line 846 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 237:
#line 848 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 238:
#line 850 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 239:
#line 852 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 240:
#line 854 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 241:
#line 857 "xi-grammar.y"
    { 
                  (yyval.entry) = new Entry(lineno, (yyvsp[(2) - (7)].intval), (yyvsp[(3) - (7)].type), (yyvsp[(4) - (7)].strval), (yyvsp[(5) - (7)].plist), (yyvsp[(6) - (7)].val), (yyvsp[(7) - (7)].sentry), (const char *) NULL, (yylsp[(1) - (7)]).first_line, (yyloc).last_line);
		  if ((yyvsp[(7) - (7)].sentry) != 0) { 
		    (yyvsp[(7) - (7)].sentry)->con1 = new SdagConstruct(SIDENT, (yyvsp[(4) - (7)].strval));
                    (yyvsp[(7) - (7)].sentry)->entry = (yyval.entry);
                    (yyvsp[(7) - (7)].sentry)->con1->entry = (yyval.entry);
                    (yyvsp[(7) - (7)].sentry)->param = new ParamList((yyvsp[(5) - (7)].plist));
                  }
		}
    break;

  case 242:
#line 867 "xi-grammar.y"
    { 
                  Entry *e = new Entry(lineno, (yyvsp[(2) - (5)].intval), 0, (yyvsp[(3) - (5)].strval), (yyvsp[(4) - (5)].plist),  0, (yyvsp[(5) - (5)].sentry), (const char *) NULL, (yylsp[(1) - (5)]).first_line, (yyloc).last_line);
                  if ((yyvsp[(5) - (5)].sentry) != 0) {
		    (yyvsp[(5) - (5)].sentry)->con1 = new SdagConstruct(SIDENT, (yyvsp[(3) - (5)].strval));
                    (yyvsp[(5) - (5)].sentry)->entry = e;
                    (yyvsp[(5) - (5)].sentry)->con1->entry = e;
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

  case 243:
#line 884 "xi-grammar.y"
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

  case 244:
#line 900 "xi-grammar.y"
    { (yyval.accelBlock) = new AccelBlock(lineno, new XStr((yyvsp[(3) - (5)].strval))); }
    break;

  case 245:
#line 902 "xi-grammar.y"
    { (yyval.accelBlock) = new AccelBlock(lineno, NULL); }
    break;

  case 246:
#line 906 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 247:
#line 910 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 248:
#line 912 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(2) - (3)].intval); }
    break;

  case 249:
#line 914 "xi-grammar.y"
    { ERROR("invalid entry method attribute list",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
    break;

  case 250:
#line 921 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (1)].intval); }
    break;

  case 251:
#line 923 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (3)].intval) | (yyvsp[(3) - (3)].intval); }
    break;

  case 252:
#line 927 "xi-grammar.y"
    { (yyval.intval) = STHREADED; }
    break;

  case 253:
#line 929 "xi-grammar.y"
    { (yyval.intval) = SSYNC; }
    break;

  case 254:
#line 931 "xi-grammar.y"
    { (yyval.intval) = SIGET; }
    break;

  case 255:
#line 933 "xi-grammar.y"
    { (yyval.intval) = SLOCKED; }
    break;

  case 256:
#line 935 "xi-grammar.y"
    { (yyval.intval) = SCREATEHERE; }
    break;

  case 257:
#line 937 "xi-grammar.y"
    { (yyval.intval) = SCREATEHOME; }
    break;

  case 258:
#line 939 "xi-grammar.y"
    { (yyval.intval) = SNOKEEP; }
    break;

  case 259:
#line 941 "xi-grammar.y"
    { (yyval.intval) = SNOTRACE; }
    break;

  case 260:
#line 943 "xi-grammar.y"
    { (yyval.intval) = SAPPWORK; }
    break;

  case 261:
#line 945 "xi-grammar.y"
    { (yyval.intval) = SIMMEDIATE; }
    break;

  case 262:
#line 947 "xi-grammar.y"
    { (yyval.intval) = SSKIPSCHED; }
    break;

  case 263:
#line 949 "xi-grammar.y"
    { (yyval.intval) = SINLINE; }
    break;

  case 264:
#line 951 "xi-grammar.y"
    { (yyval.intval) = SLOCAL; }
    break;

  case 265:
#line 953 "xi-grammar.y"
    { (yyval.intval) = SPYTHON; }
    break;

  case 266:
#line 955 "xi-grammar.y"
    { (yyval.intval) = SMEM; }
    break;

  case 267:
#line 957 "xi-grammar.y"
    { (yyval.intval) = SREDUCE; }
    break;

  case 268:
#line 959 "xi-grammar.y"
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

  case 269:
#line 971 "xi-grammar.y"
    {
		  ERROR("invalid entry method attribute",
		        (yylsp[(1) - (1)]).first_column, (yylsp[(1) - (1)]).last_column);
		  yyclearin;
		  yyerrok;
		}
    break;

  case 270:
#line 980 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 271:
#line 982 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 272:
#line 984 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 273:
#line 988 "xi-grammar.y"
    { (yyval.strval) = ""; }
    break;

  case 274:
#line 990 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 275:
#line 992 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[(1) - (3)].strval))+strlen((yyvsp[(3) - (3)].strval))+3];
			sprintf(tmp,"%s, %s", (yyvsp[(1) - (3)].strval), (yyvsp[(3) - (3)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 276:
#line 1000 "xi-grammar.y"
    { (yyval.strval) = ""; }
    break;

  case 277:
#line 1002 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 278:
#line 1004 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s[%s]%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 279:
#line 1010 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s{%s}%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 280:
#line 1016 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s(%s)%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 281:
#line 1022 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(2) - (4)].strval))+strlen((yyvsp[(4) - (4)].strval))+3];
			sprintf(tmp,"(%s)%s", (yyvsp[(2) - (4)].strval), (yyvsp[(4) - (4)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 282:
#line 1030 "xi-grammar.y"
    {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			(yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].type),(yyvsp[(2) - (3)].strval));
		}
    break;

  case 283:
#line 1037 "xi-grammar.y"
    { 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			(yyval.intval) = 0;
		}
    break;

  case 284:
#line 1045 "xi-grammar.y"
    { 
			in_braces=0;
			(yyval.intval) = 0;
		}
    break;

  case 285:
#line 1052 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (1)].type));}
    break;

  case 286:
#line 1054 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].type),(yyvsp[(2) - (3)].strval)); (yyval.pname)->setConditional((yyvsp[(3) - (3)].intval)); }
    break;

  case 287:
#line 1056 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (4)].type),(yyvsp[(2) - (4)].strval),0,(yyvsp[(4) - (4)].val));}
    break;

  case 288:
#line 1058 "xi-grammar.y"
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].pname)->getType(), (yyvsp[(1) - (3)].pname)->getName() ,(yyvsp[(2) - (3)].strval));
		}
    break;

  case 289:
#line 1064 "xi-grammar.y"
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READONLY; }
    break;

  case 290:
#line 1065 "xi-grammar.y"
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READWRITE; }
    break;

  case 291:
#line 1066 "xi-grammar.y"
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_WRITEONLY; }
    break;

  case 292:
#line 1069 "xi-grammar.y"
    { (yyval.xstrptr) = new XStr((yyvsp[(1) - (1)].strval)); }
    break;

  case 293:
#line 1070 "xi-grammar.y"
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "->" << (yyvsp[(4) - (4)].strval); }
    break;

  case 294:
#line 1071 "xi-grammar.y"
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[(1) - (3)].xstrptr)) << "." << (yyvsp[(3) - (3)].strval); }
    break;

  case 295:
#line 1073 "xi-grammar.y"
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "[" << *((yyvsp[(3) - (4)].xstrptr)) << "]";
                  delete (yyvsp[(1) - (4)].xstrptr);
                  delete (yyvsp[(3) - (4)].xstrptr);
                }
    break;

  case 296:
#line 1080 "xi-grammar.y"
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "[" << (yyvsp[(3) - (4)].strval) << "]";
                  delete (yyvsp[(1) - (4)].xstrptr);
                }
    break;

  case 297:
#line 1086 "xi-grammar.y"
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "(" << *((yyvsp[(3) - (4)].xstrptr)) << ")";
                  delete (yyvsp[(1) - (4)].xstrptr);
                  delete (yyvsp[(3) - (4)].xstrptr);
                }
    break;

  case 298:
#line 1095 "xi-grammar.y"
    {
                  in_bracket = 0;
                  (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].pname)->getType(), (yyvsp[(1) - (3)].pname)->getName(), (yyvsp[(2) - (3)].strval));
                }
    break;

  case 299:
#line 1102 "xi-grammar.y"
    {
                  (yyval.pname) = new Parameter(lineno, (yyvsp[(3) - (7)].type), (yyvsp[(4) - (7)].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[(6) - (7)].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[(1) - (7)].intval));
                }
    break;

  case 300:
#line 1108 "xi-grammar.y"
    {
		  (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (5)].type), (yyvsp[(2) - (5)].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[(4) - (5)].xstrptr));
                  (yyval.pname)->setAccelBufferType(Parameter::ACCEL_BUFFER_TYPE_READWRITE);
		}
    break;

  case 301:
#line 1114 "xi-grammar.y"
    {
                  (yyval.pname) = (yyvsp[(3) - (6)].pname);
                  (yyval.pname)->setAccelInstName((yyvsp[(5) - (6)].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[(1) - (6)].intval));
		}
    break;

  case 302:
#line 1122 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (1)].pname)); }
    break;

  case 303:
#line 1124 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (3)].pname),(yyvsp[(3) - (3)].plist)); }
    break;

  case 304:
#line 1128 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (1)].pname)); }
    break;

  case 305:
#line 1130 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (3)].pname),(yyvsp[(3) - (3)].plist)); }
    break;

  case 306:
#line 1134 "xi-grammar.y"
    { (yyval.plist) = (yyvsp[(2) - (3)].plist); }
    break;

  case 307:
#line 1136 "xi-grammar.y"
    { (yyval.plist) = new ParamList(new Parameter(0, new BuiltinType("void"))); }
    break;

  case 308:
#line 1140 "xi-grammar.y"
    { (yyval.plist) = (yyvsp[(2) - (3)].plist); }
    break;

  case 309:
#line 1142 "xi-grammar.y"
    { (yyval.plist) = 0; }
    break;

  case 310:
#line 1146 "xi-grammar.y"
    { (yyval.val) = 0; }
    break;

  case 311:
#line 1148 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(3) - (3)].strval)); }
    break;

  case 312:
#line 1152 "xi-grammar.y"
    { (yyval.sentry) = 0; }
    break;

  case 313:
#line 1154 "xi-grammar.y"
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[(1) - (1)].sc)); }
    break;

  case 314:
#line 1156 "xi-grammar.y"
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[(2) - (4)].slist)); }
    break;

  case 315:
#line 1160 "xi-grammar.y"
    { (yyval.slist) = new SListConstruct((yyvsp[(1) - (1)].sc)); }
    break;

  case 316:
#line 1162 "xi-grammar.y"
    { (yyval.slist) = new SListConstruct((yyvsp[(1) - (2)].sc), (yyvsp[(2) - (2)].slist));  }
    break;

  case 317:
#line 1166 "xi-grammar.y"
    { (yyval.olist) = new OListConstruct((yyvsp[(1) - (1)].sc)); }
    break;

  case 318:
#line 1168 "xi-grammar.y"
    { (yyval.olist) = new OListConstruct((yyvsp[(1) - (2)].sc), (yyvsp[(2) - (2)].slist)); }
    break;

  case 319:
#line 1172 "xi-grammar.y"
    { (yyval.clist) = new CaseListConstruct((yyvsp[(1) - (1)].when)); }
    break;

  case 320:
#line 1174 "xi-grammar.y"
    { (yyval.clist) = new CaseListConstruct((yyvsp[(1) - (2)].when), (yyvsp[(2) - (2)].clist)); }
    break;

  case 321:
#line 1176 "xi-grammar.y"
    {
		  ERROR("case blocks can only contain when clauses",
		        (yylsp[(1) - (1)]).first_column, (yylsp[(1) - (1)]).last_column);
		  (yyval.clist) = 0;
		}
    break;

  case 322:
#line 1184 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 323:
#line 1186 "xi-grammar.y"
    { (yyval.strval) = 0; }
    break;

  case 324:
#line 1190 "xi-grammar.y"
    { (yyval.when) = new WhenConstruct((yyvsp[(2) - (4)].entrylist), 0); }
    break;

  case 325:
#line 1192 "xi-grammar.y"
    { (yyval.when) = new WhenConstruct((yyvsp[(2) - (3)].entrylist), (yyvsp[(3) - (3)].sc)); }
    break;

  case 326:
#line 1194 "xi-grammar.y"
    { (yyval.when) = new WhenConstruct((yyvsp[(2) - (5)].entrylist), (yyvsp[(4) - (5)].slist)); }
    break;

  case 327:
#line 1198 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 328:
#line 1200 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 329:
#line 1202 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 330:
#line 1204 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 331:
#line 1206 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 332:
#line 1208 "xi-grammar.y"
    { (yyval.when) = 0; }
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
#line 1224 "xi-grammar.y"
    { (yyval.sc) = new AtomicConstruct((yyvsp[(4) - (5)].strval), (yyvsp[(2) - (5)].strval), (yylsp[(3) - (5)]).first_line); }
    break;

  case 340:
#line 1226 "xi-grammar.y"
    { (yyval.sc) = new OverlapConstruct((yyvsp[(3) - (4)].olist)); }
    break;

  case 341:
#line 1228 "xi-grammar.y"
    { (yyval.sc) = (yyvsp[(1) - (1)].when); }
    break;

  case 342:
#line 1230 "xi-grammar.y"
    { (yyval.sc) = new CaseConstruct((yyvsp[(3) - (4)].clist)); }
    break;

  case 343:
#line 1232 "xi-grammar.y"
    { (yyval.sc) = new ForConstruct((yyvsp[(3) - (11)].intexpr), (yyvsp[(5) - (11)].intexpr), (yyvsp[(7) - (11)].intexpr), (yyvsp[(10) - (11)].slist)); }
    break;

  case 344:
#line 1234 "xi-grammar.y"
    { (yyval.sc) = new ForConstruct((yyvsp[(3) - (9)].intexpr), (yyvsp[(5) - (9)].intexpr), (yyvsp[(7) - (9)].intexpr), (yyvsp[(9) - (9)].sc)); }
    break;

  case 345:
#line 1236 "xi-grammar.y"
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[(3) - (12)].strval)), (yyvsp[(6) - (12)].intexpr),
		             (yyvsp[(8) - (12)].intexpr), (yyvsp[(10) - (12)].intexpr), (yyvsp[(12) - (12)].sc)); }
    break;

  case 346:
#line 1239 "xi-grammar.y"
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[(3) - (14)].strval)), (yyvsp[(6) - (14)].intexpr),
		             (yyvsp[(8) - (14)].intexpr), (yyvsp[(10) - (14)].intexpr), (yyvsp[(13) - (14)].slist)); }
    break;

  case 347:
#line 1242 "xi-grammar.y"
    { (yyval.sc) = new IfConstruct((yyvsp[(3) - (6)].intexpr), (yyvsp[(5) - (6)].sc), (yyvsp[(6) - (6)].sc)); }
    break;

  case 348:
#line 1244 "xi-grammar.y"
    { (yyval.sc) = new IfConstruct((yyvsp[(3) - (8)].intexpr), (yyvsp[(6) - (8)].slist), (yyvsp[(8) - (8)].sc)); }
    break;

  case 349:
#line 1246 "xi-grammar.y"
    { (yyval.sc) = new WhileConstruct((yyvsp[(3) - (5)].intexpr), (yyvsp[(5) - (5)].sc)); }
    break;

  case 350:
#line 1248 "xi-grammar.y"
    { (yyval.sc) = new WhileConstruct((yyvsp[(3) - (7)].intexpr), (yyvsp[(6) - (7)].slist)); }
    break;

  case 351:
#line 1250 "xi-grammar.y"
    { (yyval.sc) = new AtomicConstruct((yyvsp[(2) - (3)].strval), NULL, (yyloc).first_line); }
    break;

  case 352:
#line 1252 "xi-grammar.y"
    {
		  ERROR("unknown SDAG construct or malformed entry method declaration.\n"
		        "You may have forgotten to terminate a previous entry method declaration with a"
		        " semicolon or forgotten to mark a block of sequential SDAG code as 'serial'",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
    break;

  case 353:
#line 1262 "xi-grammar.y"
    { (yyval.sc) = 0; }
    break;

  case 354:
#line 1264 "xi-grammar.y"
    { (yyval.sc) = new ElseConstruct((yyvsp[(2) - (2)].sc)); }
    break;

  case 355:
#line 1266 "xi-grammar.y"
    { (yyval.sc) = new ElseConstruct((yyvsp[(3) - (4)].slist)); }
    break;

  case 356:
#line 1270 "xi-grammar.y"
    { (yyval.intexpr) = new IntExprConstruct((yyvsp[(1) - (1)].strval)); }
    break;

  case 357:
#line 1274 "xi-grammar.y"
    { in_int_expr = 0; (yyval.intval) = 0; }
    break;

  case 358:
#line 1278 "xi-grammar.y"
    { in_int_expr = 1; (yyval.intval) = 0; }
    break;

  case 359:
#line 1282 "xi-grammar.y"
    {
		  (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[(1) - (2)].strval), (yyvsp[(2) - (2)].plist), 0, 0, 0, (yyloc).first_line, (yyloc).last_line);
		}
    break;

  case 360:
#line 1286 "xi-grammar.y"
    {
		  (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[(1) - (5)].strval), (yyvsp[(5) - (5)].plist), 0, 0, (yyvsp[(3) - (5)].strval), (yyloc).first_line, (yyloc).last_line);
		}
    break;

  case 361:
#line 1292 "xi-grammar.y"
    { (yyval.entrylist) = new EntryList((yyvsp[(1) - (1)].entry)); }
    break;

  case 362:
#line 1294 "xi-grammar.y"
    { (yyval.entrylist) = new EntryList((yyvsp[(1) - (3)].entry),(yyvsp[(3) - (3)].entrylist)); }
    break;

  case 363:
#line 1298 "xi-grammar.y"
    { in_bracket=1; }
    break;

  case 364:
#line 1301 "xi-grammar.y"
    { in_bracket=0; }
    break;

  case 365:
#line 1305 "xi-grammar.y"
    { if (!macroDefined((yyvsp[(2) - (2)].strval), 1)) in_comment = 1; }
    break;

  case 366:
#line 1309 "xi-grammar.y"
    { if (!macroDefined((yyvsp[(2) - (2)].strval), 0)) in_comment = 1; }
    break;


/* Line 1267 of yacc.c.  */
#line 4440 "y.tab.c"
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


#line 1312 "xi-grammar.y"


void yyerror(const char *msg) { }

