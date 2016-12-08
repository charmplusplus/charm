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
     SCATTERV = 328,
     CASE = 329
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
#define SCATTERV 328
#define CASE 329




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
#line 335 "y.tab.c"
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
#line 360 "y.tab.c"

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
#define YYFINAL  56
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   1513

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  91
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  117
/* YYNRULES -- Number of rules.  */
#define YYNRULES  371
/* YYNRULES -- Number of states.  */
#define YYNSTATES  722

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   329

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    85,     2,
      83,    84,    82,     2,    79,    89,    90,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    76,    75,
      80,    88,    81,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    86,     2,    87,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    77,     2,    78,     2,     2,     2,     2,
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
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74
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
     118,   123,   127,   131,   133,   138,   139,   142,   146,   149,
     152,   155,   163,   169,   175,   178,   181,   184,   187,   190,
     193,   196,   199,   201,   203,   205,   207,   209,   211,   213,
     215,   219,   220,   222,   223,   227,   229,   231,   233,   235,
     238,   241,   245,   249,   252,   255,   258,   260,   262,   265,
     267,   270,   273,   275,   277,   280,   283,   286,   295,   297,
     299,   301,   303,   306,   309,   312,   314,   316,   318,   321,
     324,   327,   329,   332,   334,   336,   340,   341,   344,   349,
     356,   357,   359,   360,   364,   366,   370,   372,   374,   375,
     379,   381,   385,   386,   388,   390,   391,   395,   397,   401,
     403,   405,   406,   408,   409,   412,   418,   420,   423,   427,
     434,   435,   438,   440,   444,   450,   456,   462,   468,   473,
     477,   484,   491,   497,   503,   509,   515,   521,   526,   534,
     535,   538,   539,   542,   545,   548,   552,   555,   559,   561,
     565,   570,   573,   576,   579,   582,   585,   587,   592,   593,
     596,   598,   600,   602,   604,   607,   610,   613,   617,   624,
     634,   638,   645,   649,   656,   666,   676,   678,   682,   684,
     686,   688,   691,   694,   696,   698,   700,   702,   704,   706,
     708,   710,   712,   714,   716,   718,   726,   732,   746,   752,
     755,   757,   758,   762,   764,   766,   770,   772,   774,   776,
     778,   780,   782,   784,   786,   788,   790,   792,   794,   796,
     799,   801,   803,   805,   807,   809,   811,   813,   815,   816,
     818,   822,   823,   825,   831,   837,   843,   848,   852,   854,
     856,   858,   862,   867,   871,   873,   875,   877,   879,   884,
     888,   893,   898,   903,   907,   915,   921,   928,   930,   934,
     936,   940,   944,   947,   951,   954,   955,   959,   961,   963,
     968,   970,   973,   975,   978,   980,   983,   985,   987,   988,
     993,   997,  1003,  1010,  1015,  1020,  1032,  1042,  1055,  1070,
    1077,  1086,  1092,  1100,  1105,  1112,  1117,  1119,  1124,  1136,
    1146,  1159,  1174,  1181,  1190,  1196,  1204,  1209,  1211,  1212,
    1215,  1220,  1222,  1224,  1226,  1229,  1235,  1237,  1241,  1243,
    1245,  1248
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
      92,     0,    -1,    93,    -1,    -1,    99,    93,    -1,    -1,
       5,    -1,    75,    -1,    95,    75,    -1,    -1,    95,    -1,
      54,    -1,     3,    -1,     4,    -1,     5,    -1,     7,    -1,
       8,    -1,     9,    -1,    11,    -1,    12,    -1,    13,    -1,
      14,    -1,    15,    -1,    19,    -1,    20,    -1,    21,    -1,
      22,    -1,    23,    -1,    24,    -1,    25,    -1,    26,    -1,
      27,    -1,    28,    -1,    29,    -1,    30,    -1,    32,    -1,
      33,    -1,    34,    -1,    35,    -1,    36,    -1,    39,    -1,
      40,    -1,    41,    -1,    42,    -1,    43,    -1,    44,    -1,
      45,    -1,    46,    -1,    47,    -1,    48,    -1,    49,    -1,
      51,    -1,    53,    -1,    67,    -1,    70,    -1,    71,    -1,
      72,    -1,    73,    -1,    74,    -1,    54,    -1,    98,    76,
      76,    54,    -1,     3,    97,   100,    -1,     4,    97,   100,
      -1,    95,    -1,    77,   101,    78,    96,    -1,    -1,   103,
     101,    -1,    53,    52,    98,    -1,    53,    98,    -1,    94,
     160,    -1,    94,   139,    -1,     5,    41,   170,   110,    97,
     107,   187,    -1,    94,    77,   101,    78,    96,    -1,    52,
      97,    77,   101,    78,    -1,   102,    95,    -1,   102,   167,
      -1,    94,    99,    -1,    94,   142,    -1,    94,   143,    -1,
      94,   144,    -1,    94,   146,    -1,    94,   157,    -1,   206,
      -1,   207,    -1,   169,    -1,     1,    -1,   118,    -1,    55,
      -1,    56,    -1,   104,    -1,   104,    79,   105,    -1,    -1,
     105,    -1,    -1,    80,   106,    81,    -1,    60,    -1,    61,
      -1,    62,    -1,    63,    -1,    66,    60,    -1,    66,    61,
      -1,    66,    61,    60,    -1,    66,    61,    61,    -1,    66,
      62,    -1,    66,    63,    -1,    61,    61,    -1,    64,    -1,
      65,    -1,    61,    65,    -1,    37,    -1,    97,   107,    -1,
      98,   107,    -1,   108,    -1,   110,    -1,   111,    82,    -1,
     112,    82,    -1,   113,    82,    -1,   115,    83,    82,    97,
      84,    83,   185,    84,    -1,   111,    -1,   112,    -1,   113,
      -1,   114,    -1,    73,   111,    -1,    38,   115,    -1,   115,
      38,    -1,   111,    -1,   112,    -1,   113,    -1,    38,   116,
      -1,   116,    38,    -1,   116,    85,    -1,   116,    -1,   115,
      85,    -1,   115,    -1,   176,    -1,   204,   119,   205,    -1,
      -1,   120,   121,    -1,     6,   118,    98,   121,    -1,     6,
      16,   111,    82,    98,   121,    -1,    -1,    37,    -1,    -1,
      86,   126,    87,    -1,   127,    -1,   127,    79,   126,    -1,
      39,    -1,    40,    -1,    -1,    86,   129,    87,    -1,   134,
      -1,   134,    79,   129,    -1,    -1,    56,    -1,    50,    -1,
      -1,    86,   133,    87,    -1,   131,    -1,   131,    79,   133,
      -1,    30,    -1,    50,    -1,    -1,    17,    -1,    -1,    86,
      87,    -1,   135,   118,    97,   136,    95,    -1,   137,    -1,
     137,   138,    -1,    16,   125,   109,    -1,    16,   125,   109,
      77,   138,    78,    -1,    -1,    76,   141,    -1,   110,    -1,
     110,    79,   141,    -1,    11,   128,   109,   140,   158,    -1,
      12,   128,   109,   140,   158,    -1,    13,   128,   109,   140,
     158,    -1,    14,   128,   109,   140,   158,    -1,    86,    55,
      97,    87,    -1,    86,    97,    87,    -1,    15,   132,   145,
     109,   140,   158,    -1,    15,   145,   132,   109,   140,   158,
      -1,    11,   128,    97,   140,   158,    -1,    12,   128,    97,
     140,   158,    -1,    13,   128,    97,   140,   158,    -1,    14,
     128,    97,   140,   158,    -1,    15,   145,    97,   140,   158,
      -1,    16,   125,    97,    95,    -1,    16,   125,    97,    77,
     138,    78,    95,    -1,    -1,    88,   118,    -1,    -1,    88,
      55,    -1,    88,    56,    -1,    88,   110,    -1,    18,    97,
     152,    -1,   114,   153,    -1,   118,    97,   153,    -1,   154,
      -1,   154,    79,   155,    -1,    22,    80,   155,    81,    -1,
     156,   147,    -1,   156,   148,    -1,   156,   149,    -1,   156,
     150,    -1,   156,   151,    -1,    95,    -1,    77,   159,    78,
      96,    -1,    -1,   165,   159,    -1,   122,    -1,   123,    -1,
     162,    -1,   161,    -1,    10,   163,    -1,    19,   164,    -1,
      18,    97,    -1,     8,   124,    98,    -1,     8,   124,    98,
      83,   124,    84,    -1,     8,   124,    98,    80,   105,    81,
      83,   124,    84,    -1,     7,   124,    98,    -1,     7,   124,
      98,    83,   124,    84,    -1,     9,   124,    98,    -1,     9,
     124,    98,    83,   124,    84,    -1,     9,   124,    98,    80,
     105,    81,    83,   124,    84,    -1,     9,    86,    67,    87,
     124,    98,    83,   124,    84,    -1,   110,    -1,   110,    79,
     163,    -1,    56,    -1,   166,    -1,   168,    -1,   156,   168,
      -1,   160,    95,    -1,     1,    -1,    41,    -1,    78,    -1,
       7,    -1,     8,    -1,     9,    -1,    11,    -1,    12,    -1,
      15,    -1,    13,    -1,    14,    -1,     6,    -1,    41,   171,
     170,    97,   187,   189,   190,    -1,    41,   171,    97,   187,
     190,    -1,    41,    86,    67,    87,    37,    97,   187,   188,
     178,   176,   179,    97,    95,    -1,    70,   178,   176,   179,
      95,    -1,    70,    95,    -1,   117,    -1,    -1,    86,   172,
      87,    -1,     1,    -1,   173,    -1,   173,    79,   172,    -1,
      21,    -1,    23,    -1,    24,    -1,    25,    -1,    32,    -1,
      33,    -1,    34,    -1,    35,    -1,    36,    -1,    26,    -1,
      27,    -1,    28,    -1,    51,    -1,    50,   130,    -1,    71,
      -1,    72,    -1,    31,    -1,    73,    -1,     1,    -1,    56,
      -1,    55,    -1,    98,    -1,    -1,    57,    -1,    57,    79,
     175,    -1,    -1,    57,    -1,    57,    86,   176,    87,   176,
      -1,    57,    77,   176,    78,   176,    -1,    57,    83,   175,
      84,   176,    -1,    83,   176,    84,   176,    -1,   118,    97,
      86,    -1,    77,    -1,    78,    -1,   118,    -1,   118,    97,
     135,    -1,   118,    97,    88,   174,    -1,   177,   176,    87,
      -1,     6,    -1,    68,    -1,    69,    -1,    97,    -1,   182,
      89,    81,    97,    -1,   182,    90,    97,    -1,   182,    86,
     182,    87,    -1,   182,    86,    55,    87,    -1,   182,    83,
     182,    84,    -1,   177,   176,    87,    -1,   181,    76,   118,
      97,    80,   182,    81,    -1,   118,    97,    80,   182,    81,
      -1,   181,    76,   183,    80,   182,    81,    -1,   180,    -1,
     180,    79,   185,    -1,   184,    -1,   184,    79,   186,    -1,
      83,   185,    84,    -1,    83,    84,    -1,    86,   186,    87,
      -1,    86,    87,    -1,    -1,    20,    88,    55,    -1,    95,
      -1,   197,    -1,    77,   191,    78,    96,    -1,   197,    -1,
     197,   191,    -1,   197,    -1,   197,   191,    -1,   195,    -1,
     195,   193,    -1,   196,    -1,    56,    -1,    -1,    45,   203,
      77,    78,    -1,    45,   203,   197,    -1,    45,   203,    77,
     191,    78,    -1,    47,   194,   178,   176,   179,    96,    -1,
      46,    77,   192,    78,    -1,    74,    77,   193,    78,    -1,
      42,   201,   176,    75,   176,    75,   176,   200,    77,   191,
      78,    -1,    42,   201,   176,    75,   176,    75,   176,   200,
     197,    -1,    43,    86,    54,    87,   201,   176,    76,   176,
      79,   176,   200,   197,    -1,    43,    86,    54,    87,   201,
     176,    76,   176,    79,   176,   200,    77,   191,    78,    -1,
      48,   201,   176,   200,   197,   198,    -1,    48,   201,   176,
     200,    77,   191,    78,   198,    -1,    44,   201,   176,   200,
     197,    -1,    44,   201,   176,   200,    77,   191,    78,    -1,
     178,   176,   179,    96,    -1,    47,   194,   178,   176,   179,
      96,    -1,    46,    77,   192,    78,    -1,   195,    -1,    74,
      77,   193,    78,    -1,    42,   201,   199,    75,   199,    75,
     199,   200,    77,   191,    78,    -1,    42,   201,   199,    75,
     199,    75,   199,   200,   197,    -1,    43,    86,    54,    87,
     201,   199,    76,   199,    79,   199,   200,   197,    -1,    43,
      86,    54,    87,   201,   199,    76,   199,    79,   199,   200,
      77,   191,    78,    -1,    48,   201,   199,   200,   197,   198,
      -1,    48,   201,   199,   200,    77,   191,    78,   198,    -1,
      44,   201,   199,   200,   197,    -1,    44,   201,   199,   200,
      77,   191,    78,    -1,   178,   176,   179,    96,    -1,     1,
      -1,    -1,    49,   197,    -1,    49,    77,   191,    78,    -1,
     176,    -1,    84,    -1,    83,    -1,    54,   187,    -1,    54,
     204,   176,   205,   187,    -1,   202,    -1,   202,    79,   203,
      -1,    86,    -1,    87,    -1,    58,    97,    -1,    59,    97,
      -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   193,   193,   198,   201,   206,   207,   211,   213,   218,
     219,   224,   226,   227,   228,   230,   231,   232,   234,   235,
     236,   237,   238,   242,   243,   244,   245,   246,   247,   248,
     249,   250,   251,   252,   253,   254,   255,   256,   257,   258,
     261,   262,   263,   264,   265,   266,   267,   268,   269,   270,
     271,   273,   275,   276,   279,   280,   281,   282,   283,   286,
     288,   295,   299,   306,   308,   313,   314,   318,   320,   322,
     324,   326,   338,   340,   342,   344,   350,   352,   354,   356,
     358,   360,   362,   364,   366,   368,   376,   378,   380,   384,
     386,   391,   392,   397,   398,   402,   404,   406,   408,   410,
     412,   414,   416,   418,   420,   422,   424,   426,   428,   430,
     434,   435,   442,   444,   448,   452,   454,   458,   462,   464,
     466,   468,   470,   472,   474,   478,   480,   482,   484,   486,
     490,   492,   496,   498,   502,   506,   511,   512,   516,   520,
     525,   526,   531,   532,   542,   544,   548,   550,   555,   556,
     560,   562,   567,   568,   572,   577,   578,   582,   584,   588,
     590,   595,   596,   600,   601,   604,   608,   610,   614,   616,
     621,   622,   626,   628,   632,   634,   638,   642,   646,   652,
     656,   658,   662,   664,   668,   672,   676,   680,   682,   687,
     688,   693,   694,   696,   698,   707,   709,   711,   715,   717,
     721,   725,   727,   729,   731,   733,   737,   739,   744,   751,
     755,   757,   759,   760,   762,   764,   766,   770,   772,   774,
     780,   786,   795,   797,   799,   805,   813,   815,   818,   822,
     826,   828,   833,   835,   843,   845,   847,   849,   851,   853,
     855,   857,   859,   861,   863,   866,   875,   891,   907,   909,
     913,   918,   919,   921,   928,   930,   934,   936,   938,   940,
     942,   944,   946,   948,   950,   952,   954,   956,   958,   960,
     962,   964,   966,   978,   980,   989,   991,   993,   998,   999,
    1001,  1010,  1011,  1013,  1019,  1025,  1031,  1039,  1046,  1054,
    1061,  1063,  1065,  1067,  1074,  1075,  1076,  1079,  1080,  1081,
    1082,  1089,  1095,  1104,  1111,  1117,  1123,  1131,  1133,  1137,
    1139,  1143,  1145,  1149,  1151,  1156,  1157,  1161,  1163,  1165,
    1169,  1171,  1175,  1177,  1181,  1183,  1185,  1193,  1196,  1199,
    1201,  1203,  1207,  1209,  1211,  1213,  1215,  1217,  1219,  1221,
    1223,  1225,  1227,  1229,  1233,  1235,  1237,  1239,  1241,  1243,
    1245,  1248,  1251,  1253,  1255,  1257,  1259,  1261,  1272,  1273,
    1275,  1279,  1283,  1287,  1291,  1295,  1301,  1303,  1307,  1310,
    1314,  1318
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
     325,   326,   327,   328,   329,    59,    58,   123,   125,    44,
      60,    62,    42,    40,    41,    38,    91,    93,    61,    45,
      46
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    91,    92,    93,    93,    94,    94,    95,    95,    96,
      96,    97,    97,    97,    97,    97,    97,    97,    97,    97,
      97,    97,    97,    97,    97,    97,    97,    97,    97,    97,
      97,    97,    97,    97,    97,    97,    97,    97,    97,    97,
      97,    97,    97,    97,    97,    97,    97,    97,    97,    97,
      97,    97,    97,    97,    97,    97,    97,    97,    97,    98,
      98,    99,    99,   100,   100,   101,   101,   102,   102,   102,
     102,   102,   103,   103,   103,   103,   103,   103,   103,   103,
     103,   103,   103,   103,   103,   103,   104,   104,   104,   105,
     105,   106,   106,   107,   107,   108,   108,   108,   108,   108,
     108,   108,   108,   108,   108,   108,   108,   108,   108,   108,
     109,   110,   111,   111,   112,   113,   113,   114,   115,   115,
     115,   115,   115,   115,   115,   116,   116,   116,   116,   116,
     117,   117,   118,   118,   119,   120,   121,   121,   122,   123,
     124,   124,   125,   125,   126,   126,   127,   127,   128,   128,
     129,   129,   130,   130,   131,   132,   132,   133,   133,   134,
     134,   135,   135,   136,   136,   137,   138,   138,   139,   139,
     140,   140,   141,   141,   142,   142,   143,   144,   145,   145,
     146,   146,   147,   147,   148,   149,   150,   151,   151,   152,
     152,   153,   153,   153,   153,   154,   154,   154,   155,   155,
     156,   157,   157,   157,   157,   157,   158,   158,   159,   159,
     160,   160,   160,   160,   160,   160,   160,   161,   161,   161,
     161,   161,   162,   162,   162,   162,   163,   163,   164,   165,
     166,   166,   166,   166,   167,   167,   167,   167,   167,   167,
     167,   167,   167,   167,   167,   168,   168,   168,   169,   169,
     170,   171,   171,   171,   172,   172,   173,   173,   173,   173,
     173,   173,   173,   173,   173,   173,   173,   173,   173,   173,
     173,   173,   173,   173,   173,   174,   174,   174,   175,   175,
     175,   176,   176,   176,   176,   176,   176,   177,   178,   179,
     180,   180,   180,   180,   181,   181,   181,   182,   182,   182,
     182,   182,   182,   183,   184,   184,   184,   185,   185,   186,
     186,   187,   187,   188,   188,   189,   189,   190,   190,   190,
     191,   191,   192,   192,   193,   193,   193,   194,   194,   195,
     195,   195,   196,   196,   196,   196,   196,   196,   196,   196,
     196,   196,   196,   196,   197,   197,   197,   197,   197,   197,
     197,   197,   197,   197,   197,   197,   197,   197,   198,   198,
     198,   199,   200,   201,   202,   202,   203,   203,   204,   205,
     206,   207
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
       4,     3,     3,     1,     4,     0,     2,     3,     2,     2,
       2,     7,     5,     5,     2,     2,     2,     2,     2,     2,
       2,     2,     1,     1,     1,     1,     1,     1,     1,     1,
       3,     0,     1,     0,     3,     1,     1,     1,     1,     2,
       2,     3,     3,     2,     2,     2,     1,     1,     2,     1,
       2,     2,     1,     1,     2,     2,     2,     8,     1,     1,
       1,     1,     2,     2,     2,     1,     1,     1,     2,     2,
       2,     1,     2,     1,     1,     3,     0,     2,     4,     6,
       0,     1,     0,     3,     1,     3,     1,     1,     0,     3,
       1,     3,     0,     1,     1,     0,     3,     1,     3,     1,
       1,     0,     1,     0,     2,     5,     1,     2,     3,     6,
       0,     2,     1,     3,     5,     5,     5,     5,     4,     3,
       6,     6,     5,     5,     5,     5,     5,     4,     7,     0,
       2,     0,     2,     2,     2,     3,     2,     3,     1,     3,
       4,     2,     2,     2,     2,     2,     1,     4,     0,     2,
       1,     1,     1,     1,     2,     2,     2,     3,     6,     9,
       3,     6,     3,     6,     9,     9,     1,     3,     1,     1,
       1,     2,     2,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     7,     5,    13,     5,     2,
       1,     0,     3,     1,     1,     3,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     2,
       1,     1,     1,     1,     1,     1,     1,     1,     0,     1,
       3,     0,     1,     5,     5,     5,     4,     3,     1,     1,
       1,     3,     4,     3,     1,     1,     1,     1,     4,     3,
       4,     4,     4,     3,     7,     5,     6,     1,     3,     1,
       3,     3,     2,     3,     2,     0,     3,     1,     1,     4,
       1,     2,     1,     2,     1,     2,     1,     1,     0,     4,
       3,     5,     6,     4,     4,    11,     9,    12,    14,     6,
       8,     5,     7,     4,     6,     4,     1,     4,    11,     9,
      12,    14,     6,     8,     5,     7,     4,     1,     0,     2,
       4,     1,     1,     1,     2,     5,     1,     3,     1,     1,
       2,     2
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       3,     0,     0,     0,     2,     3,    12,    13,    14,    15,
      16,    17,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    31,    32,    33,    34,    35,
      36,    37,    38,    39,    40,    41,    42,    43,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    11,    53,    54,
      55,    56,    57,    58,     0,     0,     1,     4,     7,     0,
      63,    61,    62,    85,     6,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    84,    82,    83,     8,     0,     0,
       0,    59,    68,   370,   371,   288,   249,   281,     0,   140,
     140,   140,     0,   148,   148,   148,   148,     0,   142,     0,
       0,     0,     0,    76,   210,   211,    70,    77,    78,    79,
      80,     0,    81,    69,   213,   212,     9,   244,   236,   237,
     238,   239,   240,   242,   243,   241,   234,   235,    74,    75,
      66,   109,     0,    95,    96,    97,    98,   106,   107,     0,
      93,   112,   113,   125,   126,   127,   131,   250,     0,     0,
      67,     0,   282,   281,     0,     0,     0,     0,   118,   119,
     120,   121,   133,     0,   141,     0,     0,     0,     0,   226,
     214,     0,     0,     0,     0,     0,     0,     0,   155,     0,
       0,   216,   228,   215,     0,     0,   148,   148,   148,   148,
       0,   142,   201,   202,   203,   204,   205,    10,    64,   128,
     105,   108,    99,   100,   103,   104,    91,   111,   114,   115,
     116,   129,   130,     0,     0,     0,   281,   278,   281,     0,
     289,     0,     0,   123,   122,   124,     0,   132,   136,   220,
     217,     0,   222,     0,   159,   160,     0,   150,    93,   170,
     170,   170,   170,   154,     0,     0,   157,     0,     0,     0,
       0,     0,   146,   147,     0,   144,   168,     0,   121,     0,
     198,     0,     9,     0,     0,     0,     0,     0,     0,   101,
     102,    87,    88,    89,    92,     0,    86,    93,    73,    60,
       0,   279,     0,     0,   281,   248,     0,     0,   368,   136,
     138,   281,   140,     0,   140,   140,     0,   140,   227,   149,
       0,   110,     0,     0,     0,     0,     0,     0,   179,     0,
     156,   170,   170,   143,     0,   161,   189,     0,   196,   191,
       0,   200,    72,   170,   170,   170,   170,   170,     0,     0,
      94,     0,   281,   278,   281,   281,   286,   136,     0,   137,
       0,   134,     0,     0,     0,     0,     0,     0,   151,   172,
     171,     0,   206,   174,   175,   176,   177,   178,   158,     0,
       0,   145,   162,     0,   161,     0,     0,   195,   192,   193,
     194,   197,   199,     0,     0,     0,     0,     0,   161,   187,
      90,     0,    71,   284,   280,   285,   283,   139,     0,   369,
     135,   221,     0,   218,     0,     0,   223,     0,   233,     0,
       0,     0,     0,     0,   229,   230,   180,   181,     0,   167,
     169,   190,   182,   183,   184,   185,   186,     0,   312,   290,
     281,   307,     0,     0,   140,   140,   140,   173,   253,     0,
       0,   231,     9,   232,   209,   163,     0,   161,     0,     0,
     311,     0,     0,     0,     0,   274,   256,   257,   258,   259,
     265,   266,   267,   272,   260,   261,   262,   263,   264,   152,
     268,     0,   270,   271,   273,     0,   254,    59,     0,     0,
     207,     0,     0,   188,   287,     0,   291,   293,   308,   117,
     219,   225,   224,   153,   269,     0,   252,     0,     0,     0,
     164,   165,   276,   275,   277,   292,     0,   255,   357,     0,
       0,     0,     0,     0,   328,     0,     0,     0,   317,   281,
     246,   346,   318,   315,     0,   363,   281,     0,   281,     0,
     366,     0,     0,   327,     0,   281,     0,     0,     0,     0,
       0,     0,     0,   361,     0,     0,     0,   364,   281,     0,
       0,   330,     0,     0,   281,     0,     0,     0,     0,     0,
     328,     0,     0,   281,     0,   324,   326,     9,   321,     9,
       0,   245,     0,     0,   281,     0,   362,     0,     0,   367,
     329,     0,   345,   323,     0,     0,   281,     0,   281,     0,
       0,   281,     0,     0,   347,   325,   319,   356,   316,   294,
     295,   296,   314,     0,     0,   309,     0,   281,     0,   281,
       0,   354,     0,   331,     9,     0,   358,     0,     0,     0,
       0,   281,     0,     0,     9,     0,     0,     0,   313,     0,
     281,     0,     0,   365,   344,     0,     0,   352,   281,     0,
       0,   333,     0,     0,   334,   343,     0,     0,   281,     0,
     310,     0,     0,   281,   355,   358,     0,   359,     0,   281,
       0,   341,     9,     0,   358,   297,     0,     0,     0,     0,
       0,     0,     0,   353,     0,   281,     0,     0,   332,     0,
     339,   305,     0,     0,     0,     0,     0,   303,     0,   247,
       0,   349,   281,   360,     0,   281,   342,   358,     0,     0,
       0,     0,   299,     0,   306,     0,     0,     0,     0,   340,
     302,   301,   300,   298,   304,   348,     0,     0,   336,   281,
       0,   350,     0,     0,     0,   335,     0,   351,     0,   337,
       0,   338
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     3,     4,    70,   352,   198,   238,   140,     5,    61,
      71,    72,    73,   273,   274,   275,   207,   141,   239,   142,
     158,   159,   160,   161,   162,   146,   147,   276,   340,   289,
     290,   104,   105,   165,   180,   254,   255,   172,   236,   484,
     246,   177,   247,   237,   363,   472,   364,   365,   106,   303,
     350,   107,   108,   109,   178,   110,   192,   193,   194,   195,
     196,   367,   318,   260,   261,   400,   112,   353,   401,   402,
     114,   115,   170,   183,   403,   404,   129,   405,    74,   148,
     430,   465,   466,   495,   282,   533,   420,   509,   221,   421,
     594,   656,   639,   595,   422,   596,   382,   563,   531,   510,
     527,   542,   554,   524,   511,   556,   528,   627,   534,   567,
     516,   520,   521,   291,   390,    75,    76
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -536
static const yytype_int16 yypact[] =
{
     150,  1337,  1337,    23,  -536,   150,  -536,  -536,  -536,  -536,
    -536,  -536,  -536,  -536,  -536,  -536,  -536,  -536,  -536,  -536,
    -536,  -536,  -536,  -536,  -536,  -536,  -536,  -536,  -536,  -536,
    -536,  -536,  -536,  -536,  -536,  -536,  -536,  -536,  -536,  -536,
    -536,  -536,  -536,  -536,  -536,  -536,  -536,  -536,  -536,  -536,
    -536,  -536,  -536,  -536,   121,   121,  -536,  -536,  -536,   764,
     -40,  -536,  -536,  -536,    14,  1337,   248,  1337,  1337,   236,
     925,    22,   904,   764,  -536,  -536,  -536,  -536,   568,    48,
      87,  -536,    90,  -536,  -536,  -536,   -40,    72,  1380,   148,
     148,    -6,    87,   113,   113,   113,   113,   122,   127,  1337,
     160,   164,   764,  -536,  -536,  -536,  -536,  -536,  -536,  -536,
    -536,   359,  -536,  -536,  -536,  -536,   174,  -536,  -536,  -536,
    -536,  -536,  -536,  -536,  -536,  -536,  -536,  -536,   -40,  -536,
    -536,  -536,   568,  -536,   170,  -536,  -536,  -536,  -536,   216,
     165,  -536,  -536,   177,   186,   200,    18,  -536,    87,   764,
      90,   176,    91,    72,   175,   255,  1440,   255,   177,   186,
     200,  -536,    76,    87,  -536,    87,    87,   220,    87,   204,
    -536,    -5,  1337,  1337,  1337,  1337,  1121,   213,   221,   185,
    1337,  -536,  -536,  -536,    58,   234,   113,   113,   113,   113,
     213,   127,  -536,  -536,  -536,  -536,  -536,   -40,  -536,   285,
    -536,  -536,  -536,   172,  -536,  -536,  1410,  -536,  -536,  -536,
    -536,  -536,  -536,  1337,   247,   279,    72,   278,    72,   253,
    -536,   174,   256,     5,  -536,  -536,   259,  -536,    66,    81,
     -26,   264,   100,    87,  -536,  -536,   280,   263,   266,   268,
     268,   268,   268,  -536,  1337,   281,   287,   294,  1193,  1337,
     332,  1337,  -536,  -536,   296,   306,   309,  1337,   123,  1337,
     308,   317,   174,  1337,  1337,  1337,  1337,  1337,  1337,  -536,
    -536,  -536,  -536,   311,  -536,   319,  -536,   266,  -536,  -536,
     324,   325,   326,   316,    72,   -40,    87,  1337,  -536,   322,
    -536,    72,   148,  1410,   148,   148,  1410,   148,  -536,  -536,
      -5,  -536,    87,   249,   249,   249,   249,   318,  -536,   332,
    -536,   268,   268,  -536,   185,   392,   327,   235,  -536,   328,
      58,  -536,  -536,   268,   268,   268,   268,   268,   257,  1410,
    -536,   329,    72,   278,    72,    72,  -536,    66,   330,  -536,
     335,  -536,   333,   338,   344,    87,   350,   351,  -536,   334,
    -536,   321,   -40,  -536,  -536,  -536,  -536,  -536,  -536,   249,
     249,  -536,  -536,  1440,    15,   355,  1440,  -536,  -536,  -536,
    -536,  -536,  -536,   249,   249,   249,   249,   249,   392,   -40,
    -536,  1395,  -536,  -536,  -536,  -536,  -536,  -536,   353,  -536,
    -536,  -536,   356,  -536,   112,   358,  -536,    87,  -536,   680,
     393,   360,   174,   321,  -536,  -536,  -536,  -536,  1337,  -536,
    -536,  -536,  -536,  -536,  -536,  -536,  -536,   364,  -536,  1337,
      72,   371,   369,  1440,   148,   148,   148,  -536,  -536,   941,
    1049,  -536,   174,   -40,  -536,   372,   174,    25,   368,  1440,
    -536,   373,   378,   379,   380,  -536,  -536,  -536,  -536,  -536,
    -536,  -536,  -536,  -536,  -536,  -536,  -536,  -536,  -536,   409,
    -536,   391,  -536,  -536,  -536,   394,   387,   384,   329,  1337,
    -536,   395,   174,   -40,  -536,   242,  -536,  -536,  -536,  -536,
    -536,  -536,  -536,  -536,  -536,   443,  -536,   994,   478,   329,
    -536,   -40,  -536,  -536,    90,  -536,  1337,  -536,  -536,   402,
     400,   402,   434,   412,   435,   402,   416,   227,   -40,    72,
    -536,  -536,  -536,   476,   329,  -536,    72,   445,    72,    -8,
     421,   503,   540,  -536,   424,    72,   525,   427,   429,   175,
     414,   478,   422,  -536,   437,   423,   430,  -536,    72,   434,
     349,  -536,   438,   486,    72,   430,   402,   432,   402,   442,
     435,   402,   458,    72,   459,   525,  -536,   174,  -536,   174,
     481,  -536,   383,   424,    72,   402,  -536,   599,   335,  -536,
    -536,   462,  -536,  -536,   175,   747,    72,   488,    72,   540,
     424,    72,   525,   175,  -536,  -536,  -536,  -536,  -536,  -536,
    -536,  -536,  -536,  1337,   467,   479,   470,    72,   484,    72,
     227,  -536,   329,  -536,   174,   227,   512,   487,   489,   430,
     496,    72,   430,   497,   174,   498,  1440,  1362,  -536,   175,
      72,   513,   501,  -536,  -536,   518,   754,  -536,    72,   402,
     761,  -536,   175,   809,  -536,  -536,  1337,  1337,    72,   510,
    -536,  1337,   430,    72,  -536,   512,   227,  -536,   516,    72,
     227,  -536,   174,   227,   512,  -536,   103,    63,   511,  1337,
     174,   817,   524,  -536,   519,    72,   531,   530,  -536,   532,
    -536,  -536,  1337,  1265,   528,  1337,  1337,  -536,   120,   -40,
     227,  -536,    72,  -536,   430,    72,  -536,   512,   171,   526,
     269,  1337,  -536,   140,  -536,   533,   430,   824,   536,  -536,
    -536,  -536,  -536,  -536,  -536,  -536,   831,   227,  -536,    72,
     227,  -536,   538,   430,   541,  -536,   879,  -536,   227,  -536,
     542,  -536
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -536,  -536,   613,  -536,   -48,  -251,    -1,   -64,   551,   569,
     -39,  -536,  -536,  -536,  -158,  -536,  -200,  -536,  -134,   -83,
     -73,   -70,   -61,  -172,   471,   491,  -536,   -78,  -536,  -536,
    -267,  -536,  -536,   -77,   446,   312,  -536,    51,   339,  -536,
    -536,   460,   341,  -536,   214,  -536,  -536,  -234,  -536,    53,
     258,  -536,  -536,  -536,  -117,  -536,  -536,  -536,  -536,  -536,
    -536,  -536,   337,  -536,   340,   582,  -536,   289,   250,   584,
    -536,  -536,   426,  -536,  -536,  -536,  -536,   261,  -536,   237,
    -536,   181,  -536,  -536,   336,   -84,    41,   -65,  -493,  -536,
    -536,  -494,  -536,  -536,  -386,    60,  -440,  -536,  -536,   139,
    -513,    99,  -535,   129,  -497,  -536,  -470,  -518,  -492,  -526,
    -457,  -536,   141,   163,   118,  -536,  -536
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -323
static const yytype_int16 yytable[] =
{
      54,    55,    82,   154,    87,   143,    60,    60,   144,   169,
     163,   322,   258,   166,   168,   558,   150,   145,   512,   575,
     585,    86,   339,    56,   128,   234,   536,   571,   488,   555,
     573,   164,   362,   545,   130,    77,   559,   441,   301,   240,
     241,   242,   362,   225,   518,   235,   256,   613,   525,   513,
     151,   541,   543,   478,   293,    78,   211,   294,   555,   143,
     249,   512,   144,   185,    79,   213,    83,    84,   197,   219,
     387,   145,   598,   267,   532,   381,   257,   331,   288,   537,
     167,   604,   222,   630,   224,   555,   633,   622,   226,   576,
     614,   578,   625,  -166,   581,   131,   156,   601,   181,   228,
     116,   229,   230,   212,   232,   606,   259,   621,   599,   543,
     214,   474,    81,   475,   225,   311,   661,   312,   133,   134,
     135,   136,   137,   138,   139,   149,   641,   663,   642,   152,
     409,   157,   280,   664,   283,   343,   670,   667,   346,   652,
     669,    81,   151,   676,   417,   173,   174,   175,   258,   474,
     169,   662,   288,     1,     2,   153,   647,   151,   697,   226,
     651,   227,   623,   654,   292,   678,   151,   695,   216,   699,
     706,   380,   649,   285,   217,   245,   151,   218,   688,   690,
     296,   470,   693,   297,   671,   164,   672,   716,   151,   673,
     696,   681,   674,   675,   712,   425,    58,   714,    59,   171,
     336,   694,  -191,   672,  -191,   720,   673,   341,   176,   674,
     675,   317,   277,   179,   197,   342,   182,   344,   345,   349,
     347,   704,   337,   672,   252,   253,   673,   708,   498,   674,
     675,   200,   269,   270,   370,   201,   711,   263,   264,   265,
     266,   151,   259,   307,   184,   206,   719,   245,   383,    58,
     385,   386,   215,   220,   672,   700,   316,   673,   319,   208,
     674,   675,   323,   324,   325,   326,   327,   328,   209,   499,
     500,   501,   502,   503,   504,   505,   202,   203,   204,   205,
     379,   394,   210,   233,  -288,   408,   338,   231,   411,    81,
     368,   369,   131,   304,   305,   306,    81,   492,   493,   248,
      80,   506,    81,   419,    85,  -288,   586,   250,   587,    81,
    -288,    58,   262,    85,   349,   133,   134,   135,   136,   137,
     138,   139,   398,   211,    58,   278,   351,    88,    89,    90,
      91,    92,    58,   279,   378,   281,   438,   284,   286,    99,
     100,   287,   300,   101,   302,   419,   206,   442,   443,   444,
     498,   295,   672,   624,   433,   673,   702,   143,   674,   675,
     144,   419,   399,   635,   359,   360,   309,   299,   308,   145,
     186,   187,   188,   189,   190,   191,   373,   374,   375,   376,
     377,   310,   243,   313,   197,   314,   315,   320,   473,   589,
     329,   499,   500,   501,   502,   503,   504,   505,   321,  -208,
     330,   668,   332,   335,   333,   357,  -288,   435,   288,   362,
     334,   494,   381,   397,   388,   366,   317,   391,   437,   392,
     131,   156,   389,   506,   491,   529,    85,   570,   393,   468,
     498,   395,  -288,   410,   399,   396,   423,    81,   432,   424,
     508,   426,   436,   133,   134,   135,   136,   137,   138,   139,
     439,   590,   591,   440,   568,   477,   157,   479,   471,   544,
     574,   553,   480,   481,   482,   483,   487,   -11,   489,   583,
     592,   499,   500,   501,   502,   503,   504,   505,   485,   498,
     496,   486,   490,   508,   593,   515,   517,   498,   519,   522,
     553,   523,   607,   526,   609,   514,   530,   612,   597,   535,
     539,    85,   560,   506,   498,   557,    85,  -320,   562,   197,
     565,   197,   564,   619,   566,   611,   572,   553,   577,   579,
     499,   500,   501,   502,   503,   504,   505,   632,   499,   500,
     501,   502,   503,   504,   505,   582,   588,   584,   637,   593,
     603,   498,   608,   616,   648,   499,   500,   501,   502,   503,
     504,   505,   506,    58,   658,   507,   197,   618,   617,   620,
     506,   626,   628,    85,  -322,   666,   197,   546,   547,   548,
     502,   549,   550,   551,   631,   634,   629,   506,   636,   644,
     540,   684,   499,   500,   501,   502,   503,   504,   505,   643,
     659,   665,   615,   354,   355,   356,   645,   683,   677,   552,
     498,   698,    85,   682,   197,   131,   132,   685,   686,   691,
     687,   705,   679,   701,   506,   709,   715,    85,    57,   717,
     721,   103,    81,   199,    62,   713,   361,   223,   133,   134,
     135,   136,   137,   138,   139,   655,   657,   268,   251,   348,
     660,   499,   500,   501,   502,   503,   504,   505,   406,   407,
     358,   476,   111,   434,   113,   427,   371,   638,   655,   298,
     372,   431,   412,   413,   414,   415,   416,   469,   497,   384,
     561,   655,   655,   506,   692,   655,   600,   640,   610,   580,
     569,   428,   538,  -251,  -251,  -251,   602,  -251,  -251,  -251,
     703,  -251,  -251,  -251,  -251,  -251,     0,     0,     0,  -251,
    -251,  -251,  -251,  -251,  -251,  -251,  -251,  -251,  -251,  -251,
    -251,     0,  -251,  -251,  -251,  -251,  -251,  -251,  -251,  -251,
    -251,  -251,  -251,  -251,  -251,  -251,  -251,  -251,  -251,  -251,
       0,  -251,     0,  -251,  -251,     0,     0,     0,     0,     0,
    -251,  -251,  -251,  -251,  -251,  -251,  -251,  -251,   498,     0,
    -251,  -251,  -251,  -251,  -251,   498,     0,     0,     0,     0,
       0,     0,   498,     0,     0,    63,   429,    -5,    -5,    64,
      -5,    -5,    -5,    -5,    -5,    -5,    -5,    -5,    -5,    -5,
      -5,     0,    -5,    -5,     0,     0,    -5,     0,     0,   499,
     500,   501,   502,   503,   504,   505,   499,   500,   501,   502,
     503,   504,   505,   499,   500,   501,   502,   503,   504,   505,
     498,     0,     0,     0,     0,     0,    65,    66,   498,     0,
       0,   506,    67,    68,   605,   498,     0,     0,   506,     0,
       0,   646,   498,     0,    69,   506,     0,     0,   650,     0,
       0,    -5,   -65,     0,     0,     0,     0,     0,     0,     0,
       0,   499,   500,   501,   502,   503,   504,   505,     0,   499,
     500,   501,   502,   503,   504,   505,   499,   500,   501,   502,
     503,   504,   505,   499,   500,   501,   502,   503,   504,   505,
     498,     0,     0,   506,     0,     0,   653,     0,     0,     0,
       0,   506,     0,     0,   680,     0,     0,     0,   506,     0,
       0,   707,     0,     0,     0,   506,     0,     0,   710,     0,
     117,   118,   119,   120,     0,   121,   122,   123,   124,   125,
       0,   499,   500,   501,   502,   503,   504,   505,     1,     2,
       0,    88,    89,    90,    91,    92,    93,    94,    95,    96,
      97,    98,   445,    99,   100,   126,     0,   101,     0,     0,
       0,     0,     0,   506,     0,     0,   718,     0,     0,     0,
       0,     0,   446,     0,   447,   448,   449,   450,   451,   452,
       0,     0,   453,   454,   455,   456,   457,   458,     0,    58,
       0,     0,   127,     0,     0,     0,     0,     0,     0,     0,
       0,   459,   460,     0,     0,   445,     0,     0,     0,     0,
       0,     0,   102,     0,     0,     0,     0,     0,   461,     0,
       0,     0,   462,   463,   464,   446,     0,   447,   448,   449,
     450,   451,   452,     0,     0,   453,   454,   455,   456,   457,
     458,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   459,   460,     0,     0,     0,     0,
       0,     0,     6,     7,     8,     0,     9,    10,    11,     0,
      12,    13,    14,    15,    16,   462,   463,   464,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
       0,    29,    30,    31,    32,    33,   131,   132,    34,    35,
      36,    37,    38,    39,    40,    41,    42,    43,    44,     0,
      45,     0,    46,   467,     0,     0,     0,     0,     0,   133,
     134,   135,   136,   137,   138,   139,    48,     0,     0,    49,
      50,    51,    52,    53,     6,     7,     8,     0,     9,    10,
      11,     0,    12,    13,    14,    15,    16,     0,     0,     0,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,     0,    29,    30,    31,    32,    33,     0,     0,
      34,    35,    36,    37,    38,    39,    40,    41,    42,    43,
      44,   243,    45,     0,    46,    47,   244,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    48,     0,
       0,    49,    50,    51,    52,    53,     6,     7,     8,     0,
       9,    10,    11,     0,    12,    13,    14,    15,    16,     0,
       0,     0,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,     0,    29,    30,    31,    32,    33,
       0,     0,    34,    35,    36,    37,    38,    39,    40,    41,
      42,    43,    44,     0,    45,     0,    46,    47,   244,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      48,     0,     0,    49,    50,    51,    52,    53,     6,     7,
       8,     0,     9,    10,    11,     0,    12,    13,    14,    15,
      16,     0,     0,     0,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,     0,    29,    30,    31,
      32,    33,     0,     0,    34,    35,    36,    37,    38,    39,
      40,    41,    42,    43,    44,     0,    45,     0,    46,    47,
     689,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    48,     0,     0,    49,    50,    51,    52,    53,
       6,     7,     8,     0,     9,    10,    11,     0,    12,    13,
      14,    15,    16,     0,     0,     0,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,   589,    29,
      30,    31,    32,    33,     0,     0,    34,    35,    36,    37,
      38,    39,    40,    41,    42,    43,    44,     0,    45,     0,
      46,    47,     0,     0,     0,     0,   155,     0,     0,   131,
     156,     0,     0,     0,    48,     0,     0,    49,    50,    51,
      52,    53,     0,     0,     0,     0,    81,   131,   156,     0,
       0,     0,   133,   134,   135,   136,   137,   138,   139,     0,
     590,   591,   131,   156,    81,   157,     0,     0,     0,     0,
     133,   134,   135,   136,   137,   138,   139,   131,   156,    81,
       0,     0,     0,   157,     0,   133,   134,   135,   136,   137,
     138,   139,     0,     0,    81,   271,   272,     0,   157,     0,
     133,   134,   135,   136,   137,   138,   139,   131,   156,   418,
       0,     0,     0,   157,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    81,     0,     0,     0,     0,     0,
     133,   134,   135,   136,   137,   138,   139,     0,     0,     0,
       0,     0,     0,   157
};

static const yytype_int16 yycheck[] =
{
       1,     2,    66,    87,    69,    78,    54,    55,    78,    92,
      88,   262,   184,    90,    91,   528,    80,    78,   488,   545,
     555,    69,   289,     0,    72,    30,   518,   540,   468,   526,
     543,    37,    17,   525,    73,    75,   529,   423,   238,   173,
     174,   175,    17,    38,   501,    50,   180,   582,   505,   489,
      76,   521,   522,   439,    80,    41,    38,    83,   555,   132,
     177,   531,   132,   102,    65,   148,    67,    68,   116,   153,
     337,   132,   564,   190,   514,    83,    18,   277,    86,   519,
      86,   574,   155,   609,   157,   582,   612,   600,    83,   546,
     583,   548,   605,    78,   551,    37,    38,   567,    99,   163,
      78,   165,   166,    85,   168,   575,   184,   599,   565,   579,
     149,    86,    54,    88,    38,   249,   642,   251,    60,    61,
      62,    63,    64,    65,    66,    77,   619,   645,   620,    57,
     364,    73,   216,   646,   218,   293,   654,   650,   296,   632,
     653,    54,    76,    80,   378,    94,    95,    96,   320,    86,
     233,   643,    86,     3,     4,    83,   626,    76,   684,    83,
     630,    85,   602,   633,    83,   659,    76,   680,    77,   687,
     696,   329,   629,   221,    83,   176,    76,    86,   672,   673,
      80,   432,   676,    83,    81,    37,    83,   713,    76,    86,
     682,   661,    89,    90,   707,    83,    75,   710,    77,    86,
     284,    81,    79,    83,    81,   718,    86,   291,    86,    89,
      90,    88,   213,    86,   262,   292,    56,   294,   295,   302,
     297,    81,   286,    83,    39,    40,    86,   697,     1,    89,
      90,    61,    60,    61,   317,    65,   706,   186,   187,   188,
     189,    76,   320,   244,    80,    80,   716,   248,   332,    75,
     334,   335,    76,    78,    83,    84,   257,    86,   259,    82,
      89,    90,   263,   264,   265,   266,   267,   268,    82,    42,
      43,    44,    45,    46,    47,    48,    60,    61,    62,    63,
     328,   345,    82,    79,    57,   363,   287,    67,   366,    54,
      55,    56,    37,   240,   241,   242,    54,    55,    56,    86,
      52,    74,    54,   381,    77,    78,   557,    86,   559,    54,
      83,    75,    78,    77,   397,    60,    61,    62,    63,    64,
      65,    66,     1,    38,    75,    78,    77,     6,     7,     8,
       9,    10,    75,    54,    77,    57,   420,    84,    82,    18,
      19,    82,    79,    22,    76,   423,    80,   424,   425,   426,
       1,    87,    83,   604,   402,    86,    87,   430,    89,    90,
     430,   439,    41,   614,   311,   312,    79,    87,    87,   430,
      11,    12,    13,    14,    15,    16,   323,   324,   325,   326,
     327,    87,    50,    87,   432,    79,    77,    79,   436,     6,
      79,    42,    43,    44,    45,    46,    47,    48,    81,    78,
      81,   652,    78,    87,    79,    87,    57,   408,    86,    17,
      84,   475,    83,    79,    84,    88,    88,    84,   419,    81,
      37,    38,    87,    74,   472,   509,    77,    78,    84,   430,
       1,    81,    83,    78,    41,    84,    83,    54,    78,    83,
     488,    83,    78,    60,    61,    62,    63,    64,    65,    66,
      79,    68,    69,    84,   538,    87,    73,    84,    86,   524,
     544,   526,    84,    84,    84,    56,    79,    83,   469,   553,
      87,    42,    43,    44,    45,    46,    47,    48,    87,     1,
      37,    87,    87,   531,   562,    83,    86,     1,    54,    77,
     555,    56,   576,    77,   578,   496,    20,   581,   563,    54,
      79,    77,    88,    74,     1,    78,    77,    78,    86,   557,
      87,   559,    75,   597,    84,   580,    78,   582,    86,    77,
      42,    43,    44,    45,    46,    47,    48,   611,    42,    43,
      44,    45,    46,    47,    48,    77,    55,    78,   616,   617,
      78,     1,    54,    76,   628,    42,    43,    44,    45,    46,
      47,    48,    74,    75,   638,    77,   604,    87,    79,    75,
      74,    49,    75,    77,    78,   649,   614,    42,    43,    44,
      45,    46,    47,    48,    78,    78,    87,    74,    80,    78,
      77,   665,    42,    43,    44,    45,    46,    47,    48,    76,
      80,    75,   593,   304,   305,   306,    78,    78,    87,    74,
       1,   685,    77,    79,   652,    37,    38,    76,    78,    81,
      78,    78,   660,    87,    74,    79,    78,    77,     5,    78,
      78,    70,    54,   132,    55,   709,   314,   156,    60,    61,
      62,    63,    64,    65,    66,   636,   637,   191,   178,   300,
     641,    42,    43,    44,    45,    46,    47,    48,   359,   360,
     309,   437,    70,   403,    70,   397,   319,   616,   659,   233,
     320,   400,   373,   374,   375,   376,   377,   430,   487,   333,
     531,   672,   673,    74,   675,   676,    77,   617,   579,   550,
     539,     1,   519,     3,     4,     5,   568,     7,     8,     9,
     691,    11,    12,    13,    14,    15,    -1,    -1,    -1,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    -1,    32,    33,    34,    35,    36,    37,    38,    39,
      40,    41,    42,    43,    44,    45,    46,    47,    48,    49,
      -1,    51,    -1,    53,    54,    -1,    -1,    -1,    -1,    -1,
      60,    61,    62,    63,    64,    65,    66,    67,     1,    -1,
      70,    71,    72,    73,    74,     1,    -1,    -1,    -1,    -1,
      -1,    -1,     1,    -1,    -1,     1,    86,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    14,    15,
      16,    -1,    18,    19,    -1,    -1,    22,    -1,    -1,    42,
      43,    44,    45,    46,    47,    48,    42,    43,    44,    45,
      46,    47,    48,    42,    43,    44,    45,    46,    47,    48,
       1,    -1,    -1,    -1,    -1,    -1,    52,    53,     1,    -1,
      -1,    74,    58,    59,    77,     1,    -1,    -1,    74,    -1,
      -1,    77,     1,    -1,    70,    74,    -1,    -1,    77,    -1,
      -1,    77,    78,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    42,    43,    44,    45,    46,    47,    48,    -1,    42,
      43,    44,    45,    46,    47,    48,    42,    43,    44,    45,
      46,    47,    48,    42,    43,    44,    45,    46,    47,    48,
       1,    -1,    -1,    74,    -1,    -1,    77,    -1,    -1,    -1,
      -1,    74,    -1,    -1,    77,    -1,    -1,    -1,    74,    -1,
      -1,    77,    -1,    -1,    -1,    74,    -1,    -1,    77,    -1,
       6,     7,     8,     9,    -1,    11,    12,    13,    14,    15,
      -1,    42,    43,    44,    45,    46,    47,    48,     3,     4,
      -1,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,     1,    18,    19,    41,    -1,    22,    -1,    -1,
      -1,    -1,    -1,    74,    -1,    -1,    77,    -1,    -1,    -1,
      -1,    -1,    21,    -1,    23,    24,    25,    26,    27,    28,
      -1,    -1,    31,    32,    33,    34,    35,    36,    -1,    75,
      -1,    -1,    78,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    50,    51,    -1,    -1,     1,    -1,    -1,    -1,    -1,
      -1,    -1,    77,    -1,    -1,    -1,    -1,    -1,    67,    -1,
      -1,    -1,    71,    72,    73,    21,    -1,    23,    24,    25,
      26,    27,    28,    -1,    -1,    31,    32,    33,    34,    35,
      36,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    50,    51,    -1,    -1,    -1,    -1,
      -1,    -1,     3,     4,     5,    -1,     7,     8,     9,    -1,
      11,    12,    13,    14,    15,    71,    72,    73,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      -1,    32,    33,    34,    35,    36,    37,    38,    39,    40,
      41,    42,    43,    44,    45,    46,    47,    48,    49,    -1,
      51,    -1,    53,    54,    -1,    -1,    -1,    -1,    -1,    60,
      61,    62,    63,    64,    65,    66,    67,    -1,    -1,    70,
      71,    72,    73,    74,     3,     4,     5,    -1,     7,     8,
       9,    -1,    11,    12,    13,    14,    15,    -1,    -1,    -1,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    -1,    32,    33,    34,    35,    36,    -1,    -1,
      39,    40,    41,    42,    43,    44,    45,    46,    47,    48,
      49,    50,    51,    -1,    53,    54,    55,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    67,    -1,
      -1,    70,    71,    72,    73,    74,     3,     4,     5,    -1,
       7,     8,     9,    -1,    11,    12,    13,    14,    15,    -1,
      -1,    -1,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    -1,    32,    33,    34,    35,    36,
      -1,    -1,    39,    40,    41,    42,    43,    44,    45,    46,
      47,    48,    49,    -1,    51,    -1,    53,    54,    55,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      67,    -1,    -1,    70,    71,    72,    73,    74,     3,     4,
       5,    -1,     7,     8,     9,    -1,    11,    12,    13,    14,
      15,    -1,    -1,    -1,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    -1,    32,    33,    34,
      35,    36,    -1,    -1,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    -1,    51,    -1,    53,    54,
      55,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    67,    -1,    -1,    70,    71,    72,    73,    74,
       3,     4,     5,    -1,     7,     8,     9,    -1,    11,    12,
      13,    14,    15,    -1,    -1,    -1,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,     6,    32,
      33,    34,    35,    36,    -1,    -1,    39,    40,    41,    42,
      43,    44,    45,    46,    47,    48,    49,    -1,    51,    -1,
      53,    54,    -1,    -1,    -1,    -1,    16,    -1,    -1,    37,
      38,    -1,    -1,    -1,    67,    -1,    -1,    70,    71,    72,
      73,    74,    -1,    -1,    -1,    -1,    54,    37,    38,    -1,
      -1,    -1,    60,    61,    62,    63,    64,    65,    66,    -1,
      68,    69,    37,    38,    54,    73,    -1,    -1,    -1,    -1,
      60,    61,    62,    63,    64,    65,    66,    37,    38,    54,
      -1,    -1,    -1,    73,    -1,    60,    61,    62,    63,    64,
      65,    66,    -1,    -1,    54,    55,    56,    -1,    73,    -1,
      60,    61,    62,    63,    64,    65,    66,    37,    38,    84,
      -1,    -1,    -1,    73,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    54,    -1,    -1,    -1,    -1,    -1,
      60,    61,    62,    63,    64,    65,    66,    -1,    -1,    -1,
      -1,    -1,    -1,    73
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     3,     4,    92,    93,    99,     3,     4,     5,     7,
       8,     9,    11,    12,    13,    14,    15,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    32,
      33,    34,    35,    36,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    51,    53,    54,    67,    70,
      71,    72,    73,    74,    97,    97,     0,    93,    75,    77,
      95,   100,   100,     1,     5,    52,    53,    58,    59,    70,
      94,   101,   102,   103,   169,   206,   207,    75,    41,    97,
      52,    54,    98,    97,    97,    77,    95,   178,     6,     7,
       8,     9,    10,    11,    12,    13,    14,    15,    16,    18,
      19,    22,    77,    99,   122,   123,   139,   142,   143,   144,
     146,   156,   157,   160,   161,   162,    78,     6,     7,     8,
       9,    11,    12,    13,    14,    15,    41,    78,    95,   167,
     101,    37,    38,    60,    61,    62,    63,    64,    65,    66,
      98,   108,   110,   111,   112,   113,   116,   117,   170,    77,
      98,    76,    57,    83,   176,    16,    38,    73,   111,   112,
     113,   114,   115,   118,    37,   124,   124,    86,   124,   110,
     163,    86,   128,   128,   128,   128,    86,   132,   145,    86,
     125,    97,    56,   164,    80,   101,    11,    12,    13,    14,
      15,    16,   147,   148,   149,   150,   151,    95,    96,   116,
      61,    65,    60,    61,    62,    63,    80,   107,    82,    82,
      82,    38,    85,   110,   101,    76,    77,    83,    86,   176,
      78,   179,   111,   115,   111,    38,    83,    85,    98,    98,
      98,    67,    98,    79,    30,    50,   129,   134,    97,   109,
     109,   109,   109,    50,    55,    97,   131,   133,    86,   145,
      86,   132,    39,    40,   126,   127,   109,    18,   114,   118,
     154,   155,    78,   128,   128,   128,   128,   145,   125,    60,
      61,    55,    56,   104,   105,   106,   118,    97,    78,    54,
     176,    57,   175,   176,    84,    95,    82,    82,    86,   120,
     121,   204,    83,    80,    83,    87,    80,    83,   163,    87,
      79,   107,    76,   140,   140,   140,   140,    97,    87,    79,
      87,   109,   109,    87,    79,    77,    97,    88,   153,    97,
      79,    81,    96,    97,    97,    97,    97,    97,    97,    79,
      81,   107,    78,    79,    84,    87,   176,    98,    97,   121,
     119,   176,   124,   105,   124,   124,   105,   124,   129,   110,
     141,    77,    95,   158,   158,   158,   158,    87,   133,   140,
     140,   126,    17,   135,   137,   138,    88,   152,    55,    56,
     110,   153,   155,   140,   140,   140,   140,   140,    77,    95,
     105,    83,   187,   176,   175,   176,   176,   121,    84,    87,
     205,    84,    81,    84,    98,    81,    84,    79,     1,    41,
     156,   159,   160,   165,   166,   168,   158,   158,   118,   138,
      78,   118,   158,   158,   158,   158,   158,   138,    84,   118,
     177,   180,   185,    83,    83,    83,    83,   141,     1,    86,
     171,   168,    78,    95,   159,    97,    78,    97,   176,    79,
      84,   185,   124,   124,   124,     1,    21,    23,    24,    25,
      26,    27,    28,    31,    32,    33,    34,    35,    36,    50,
      51,    67,    71,    72,    73,   172,   173,    54,    97,   170,
      96,    86,   136,    95,    86,    88,   135,    87,   185,    84,
      84,    84,    84,    56,   130,    87,    87,    79,   187,    97,
      87,    95,    55,    56,    98,   174,    37,   172,     1,    42,
      43,    44,    45,    46,    47,    48,    74,    77,    95,   178,
     190,   195,   197,   187,    97,    83,   201,    86,   201,    54,
     202,   203,    77,    56,   194,   201,    77,   191,   197,   176,
      20,   189,   187,   176,   199,    54,   199,   187,   204,    79,
      77,   197,   192,   197,   178,   199,    42,    43,    44,    46,
      47,    48,    74,   178,   193,   195,   196,    78,   191,   179,
      88,   190,    86,   188,    75,    87,    84,   200,   176,   203,
      78,   191,    78,   191,   176,   200,   201,    86,   201,    77,
     194,   201,    77,   176,    78,   193,    96,    96,    55,     6,
      68,    69,    87,   118,   181,   184,   186,   178,   199,   201,
      77,   197,   205,    78,   179,    77,   197,   176,    54,   176,
     192,   178,   176,   193,   179,    97,    76,    79,    87,   176,
      75,   199,   191,   187,    96,   191,    49,   198,    75,    87,
     200,    78,   176,   200,    78,    96,    80,   118,   177,   183,
     186,   179,   199,    76,    78,    78,    77,   197,   176,   201,
      77,   197,   179,    77,   197,    97,   182,    97,   176,    80,
      97,   200,   199,   198,   191,    75,   176,   191,    96,   191,
     198,    81,    83,    86,    89,    90,    80,    87,   182,    95,
      77,   197,    79,    78,   176,    76,    78,    78,   182,    55,
     182,    81,    97,   182,    81,   191,   199,   200,   176,   198,
      84,    87,    87,    97,    81,    78,   200,    77,   197,    79,
      77,   197,   191,   176,   191,    78,   200,    78,    77,   197,
     191,    78
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
#line 194 "xi-grammar.y"
    { (yyval.modlist) = (yyvsp[(1) - (1)].modlist); modlist = (yyvsp[(1) - (1)].modlist); }
    break;

  case 3:
#line 198 "xi-grammar.y"
    { 
		  (yyval.modlist) = 0; 
		}
    break;

  case 4:
#line 202 "xi-grammar.y"
    { (yyval.modlist) = new AstChildren<Module>(lineno, (yyvsp[(1) - (2)].module), (yyvsp[(2) - (2)].modlist)); }
    break;

  case 5:
#line 206 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 6:
#line 208 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 7:
#line 212 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 8:
#line 214 "xi-grammar.y"
    { (yyval.intval) = 2; }
    break;

  case 9:
#line 218 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 10:
#line 220 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 11:
#line 225 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 12:
#line 226 "xi-grammar.y"
    { ReservedWord(MODULE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 13:
#line 227 "xi-grammar.y"
    { ReservedWord(MAINMODULE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 14:
#line 228 "xi-grammar.y"
    { ReservedWord(EXTERN, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 15:
#line 230 "xi-grammar.y"
    { ReservedWord(INITCALL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 16:
#line 231 "xi-grammar.y"
    { ReservedWord(INITNODE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 17:
#line 232 "xi-grammar.y"
    { ReservedWord(INITPROC, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 18:
#line 234 "xi-grammar.y"
    { ReservedWord(CHARE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 19:
#line 235 "xi-grammar.y"
    { ReservedWord(MAINCHARE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 20:
#line 236 "xi-grammar.y"
    { ReservedWord(GROUP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 21:
#line 237 "xi-grammar.y"
    { ReservedWord(NODEGROUP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 22:
#line 238 "xi-grammar.y"
    { ReservedWord(ARRAY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 23:
#line 242 "xi-grammar.y"
    { ReservedWord(INCLUDE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 24:
#line 243 "xi-grammar.y"
    { ReservedWord(STACKSIZE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 25:
#line 244 "xi-grammar.y"
    { ReservedWord(THREADED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 26:
#line 245 "xi-grammar.y"
    { ReservedWord(TEMPLATE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 27:
#line 246 "xi-grammar.y"
    { ReservedWord(SYNC, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 28:
#line 247 "xi-grammar.y"
    { ReservedWord(IGET, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 29:
#line 248 "xi-grammar.y"
    { ReservedWord(EXCLUSIVE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 30:
#line 249 "xi-grammar.y"
    { ReservedWord(IMMEDIATE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 31:
#line 250 "xi-grammar.y"
    { ReservedWord(SKIPSCHED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 32:
#line 251 "xi-grammar.y"
    { ReservedWord(INLINE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 33:
#line 252 "xi-grammar.y"
    { ReservedWord(VIRTUAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 34:
#line 253 "xi-grammar.y"
    { ReservedWord(MIGRATABLE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 35:
#line 254 "xi-grammar.y"
    { ReservedWord(CREATEHERE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 36:
#line 255 "xi-grammar.y"
    { ReservedWord(CREATEHOME, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 37:
#line 256 "xi-grammar.y"
    { ReservedWord(NOKEEP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 38:
#line 257 "xi-grammar.y"
    { ReservedWord(NOTRACE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 39:
#line 258 "xi-grammar.y"
    { ReservedWord(APPWORK, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 40:
#line 261 "xi-grammar.y"
    { ReservedWord(PACKED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 41:
#line 262 "xi-grammar.y"
    { ReservedWord(VARSIZE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 42:
#line 263 "xi-grammar.y"
    { ReservedWord(ENTRY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 43:
#line 264 "xi-grammar.y"
    { ReservedWord(FOR, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 44:
#line 265 "xi-grammar.y"
    { ReservedWord(FORALL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 45:
#line 266 "xi-grammar.y"
    { ReservedWord(WHILE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 46:
#line 267 "xi-grammar.y"
    { ReservedWord(WHEN, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 47:
#line 268 "xi-grammar.y"
    { ReservedWord(OVERLAP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 48:
#line 269 "xi-grammar.y"
    { ReservedWord(ATOMIC, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 49:
#line 270 "xi-grammar.y"
    { ReservedWord(IF, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 50:
#line 271 "xi-grammar.y"
    { ReservedWord(ELSE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 51:
#line 273 "xi-grammar.y"
    { ReservedWord(LOCAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 52:
#line 275 "xi-grammar.y"
    { ReservedWord(USING, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 53:
#line 276 "xi-grammar.y"
    { ReservedWord(ACCEL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 54:
#line 279 "xi-grammar.y"
    { ReservedWord(ACCELBLOCK, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 55:
#line 280 "xi-grammar.y"
    { ReservedWord(MEMCRITICAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 56:
#line 281 "xi-grammar.y"
    { ReservedWord(REDUCTIONTARGET, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 57:
#line 282 "xi-grammar.y"
    { ReservedWord(SCATTERV, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 58:
#line 283 "xi-grammar.y"
    { ReservedWord(CASE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 59:
#line 287 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 60:
#line 289 "xi-grammar.y"
    {
		  char *tmp = new char[strlen((yyvsp[(1) - (4)].strval))+strlen((yyvsp[(4) - (4)].strval))+3];
		  sprintf(tmp,"%s::%s", (yyvsp[(1) - (4)].strval), (yyvsp[(4) - (4)].strval));
		  (yyval.strval) = tmp;
		}
    break;

  case 61:
#line 296 "xi-grammar.y"
    { 
		    (yyval.module) = new Module(lineno, (yyvsp[(2) - (3)].strval), (yyvsp[(3) - (3)].conslist)); 
		}
    break;

  case 62:
#line 300 "xi-grammar.y"
    {  
		    (yyval.module) = new Module(lineno, (yyvsp[(2) - (3)].strval), (yyvsp[(3) - (3)].conslist)); 
		    (yyval.module)->setMain();
		}
    break;

  case 63:
#line 307 "xi-grammar.y"
    { (yyval.conslist) = 0; }
    break;

  case 64:
#line 309 "xi-grammar.y"
    { (yyval.conslist) = (yyvsp[(2) - (4)].conslist); }
    break;

  case 65:
#line 313 "xi-grammar.y"
    { (yyval.conslist) = 0; }
    break;

  case 66:
#line 315 "xi-grammar.y"
    { (yyval.conslist) = new ConstructList(lineno, (yyvsp[(1) - (2)].construct), (yyvsp[(2) - (2)].conslist)); }
    break;

  case 67:
#line 319 "xi-grammar.y"
    { (yyval.construct) = new UsingScope((yyvsp[(3) - (3)].strval), false); }
    break;

  case 68:
#line 321 "xi-grammar.y"
    { (yyval.construct) = new UsingScope((yyvsp[(2) - (2)].strval), true); }
    break;

  case 69:
#line 323 "xi-grammar.y"
    { (yyvsp[(2) - (2)].member)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].member); }
    break;

  case 70:
#line 325 "xi-grammar.y"
    { (yyvsp[(2) - (2)].message)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].message); }
    break;

  case 71:
#line 327 "xi-grammar.y"
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

  case 72:
#line 339 "xi-grammar.y"
    { if((yyvsp[(3) - (5)].conslist)) (yyvsp[(3) - (5)].conslist)->recurse<int&>((yyvsp[(1) - (5)].intval), &Construct::setExtern); (yyval.construct) = (yyvsp[(3) - (5)].conslist); }
    break;

  case 73:
#line 341 "xi-grammar.y"
    { (yyval.construct) = new Scope((yyvsp[(2) - (5)].strval), (yyvsp[(4) - (5)].conslist)); }
    break;

  case 74:
#line 343 "xi-grammar.y"
    { (yyval.construct) = (yyvsp[(1) - (2)].construct); }
    break;

  case 75:
#line 345 "xi-grammar.y"
    {
          ERROR("preceding construct must be semicolon terminated",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
    break;

  case 76:
#line 351 "xi-grammar.y"
    { (yyvsp[(2) - (2)].module)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].module); }
    break;

  case 77:
#line 353 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
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
    { (yyvsp[(2) - (2)].templat)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].templat); }
    break;

  case 82:
#line 363 "xi-grammar.y"
    { (yyval.construct) = NULL; }
    break;

  case 83:
#line 365 "xi-grammar.y"
    { (yyval.construct) = NULL; }
    break;

  case 84:
#line 367 "xi-grammar.y"
    { (yyval.construct) = (yyvsp[(1) - (1)].accelBlock); }
    break;

  case 85:
#line 369 "xi-grammar.y"
    {
          ERROR("invalid construct",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
    break;

  case 86:
#line 377 "xi-grammar.y"
    { (yyval.tparam) = new TParamType((yyvsp[(1) - (1)].type)); }
    break;

  case 87:
#line 379 "xi-grammar.y"
    { (yyval.tparam) = new TParamVal((yyvsp[(1) - (1)].strval)); }
    break;

  case 88:
#line 381 "xi-grammar.y"
    { (yyval.tparam) = new TParamVal((yyvsp[(1) - (1)].strval)); }
    break;

  case 89:
#line 385 "xi-grammar.y"
    { (yyval.tparlist) = new TParamList((yyvsp[(1) - (1)].tparam)); }
    break;

  case 90:
#line 387 "xi-grammar.y"
    { (yyval.tparlist) = new TParamList((yyvsp[(1) - (3)].tparam), (yyvsp[(3) - (3)].tparlist)); }
    break;

  case 91:
#line 391 "xi-grammar.y"
    { (yyval.tparlist) = 0; }
    break;

  case 92:
#line 393 "xi-grammar.y"
    { (yyval.tparlist) = (yyvsp[(1) - (1)].tparlist); }
    break;

  case 93:
#line 397 "xi-grammar.y"
    { (yyval.tparlist) = 0; }
    break;

  case 94:
#line 399 "xi-grammar.y"
    { (yyval.tparlist) = (yyvsp[(2) - (3)].tparlist); }
    break;

  case 95:
#line 403 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("int"); }
    break;

  case 96:
#line 405 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long"); }
    break;

  case 97:
#line 407 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("short"); }
    break;

  case 98:
#line 409 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("char"); }
    break;

  case 99:
#line 411 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned int"); }
    break;

  case 100:
#line 413 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long"); }
    break;

  case 101:
#line 415 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long"); }
    break;

  case 102:
#line 417 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long long"); }
    break;

  case 103:
#line 419 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned short"); }
    break;

  case 104:
#line 421 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned char"); }
    break;

  case 105:
#line 423 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long long"); }
    break;

  case 106:
#line 425 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("float"); }
    break;

  case 107:
#line 427 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("double"); }
    break;

  case 108:
#line 429 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long double"); }
    break;

  case 109:
#line 431 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("void"); }
    break;

  case 110:
#line 434 "xi-grammar.y"
    { (yyval.ntype) = new NamedType((yyvsp[(1) - (2)].strval),(yyvsp[(2) - (2)].tparlist)); }
    break;

  case 111:
#line 435 "xi-grammar.y"
    { 
                    const char* basename, *scope;
                    splitScopedName((yyvsp[(1) - (2)].strval), &scope, &basename);
                    (yyval.ntype) = new NamedType(basename, (yyvsp[(2) - (2)].tparlist), scope);
                }
    break;

  case 112:
#line 443 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 113:
#line 445 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ntype); }
    break;

  case 114:
#line 449 "xi-grammar.y"
    { (yyval.ptype) = new PtrType((yyvsp[(1) - (2)].type)); }
    break;

  case 115:
#line 453 "xi-grammar.y"
    { (yyvsp[(1) - (2)].ptype)->indirect(); (yyval.ptype) = (yyvsp[(1) - (2)].ptype); }
    break;

  case 116:
#line 455 "xi-grammar.y"
    { (yyvsp[(1) - (2)].ptype)->indirect(); (yyval.ptype) = (yyvsp[(1) - (2)].ptype); }
    break;

  case 117:
#line 459 "xi-grammar.y"
    { (yyval.ftype) = new FuncType((yyvsp[(1) - (8)].type), (yyvsp[(4) - (8)].strval), (yyvsp[(7) - (8)].plist)); }
    break;

  case 118:
#line 463 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 119:
#line 465 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 120:
#line 467 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 121:
#line 469 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ftype); }
    break;

  case 122:
#line 471 "xi-grammar.y"
    { (yyval.type) = new ScattervType((yyvsp[(2) - (2)].type)); }
    break;

  case 123:
#line 473 "xi-grammar.y"
    { (yyval.type) = new ConstType((yyvsp[(2) - (2)].type)); }
    break;

  case 124:
#line 475 "xi-grammar.y"
    { (yyval.type) = new ConstType((yyvsp[(1) - (2)].type)); }
    break;

  case 125:
#line 479 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 126:
#line 481 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 127:
#line 483 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 128:
#line 485 "xi-grammar.y"
    { (yyval.type) = new ConstType((yyvsp[(2) - (2)].type)); }
    break;

  case 129:
#line 487 "xi-grammar.y"
    { (yyval.type) = new ConstType((yyvsp[(1) - (2)].type)); }
    break;

  case 130:
#line 491 "xi-grammar.y"
    { (yyval.type) = new ReferenceType((yyvsp[(1) - (2)].type)); }
    break;

  case 131:
#line 493 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 132:
#line 497 "xi-grammar.y"
    { (yyval.type) = new ReferenceType((yyvsp[(1) - (2)].type)); }
    break;

  case 133:
#line 499 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 134:
#line 503 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 135:
#line 507 "xi-grammar.y"
    { (yyval.val) = (yyvsp[(2) - (3)].val); }
    break;

  case 136:
#line 511 "xi-grammar.y"
    { (yyval.vallist) = 0; }
    break;

  case 137:
#line 513 "xi-grammar.y"
    { (yyval.vallist) = new ValueList((yyvsp[(1) - (2)].val), (yyvsp[(2) - (2)].vallist)); }
    break;

  case 138:
#line 517 "xi-grammar.y"
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[(2) - (4)].type), (yyvsp[(3) - (4)].strval), (yyvsp[(4) - (4)].vallist)); }
    break;

  case 139:
#line 521 "xi-grammar.y"
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[(3) - (6)].type), (yyvsp[(5) - (6)].strval), (yyvsp[(6) - (6)].vallist), 1); }
    break;

  case 140:
#line 525 "xi-grammar.y"
    { (yyval.intval) = 0;}
    break;

  case 141:
#line 527 "xi-grammar.y"
    { (yyval.intval) = 0;}
    break;

  case 142:
#line 531 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 143:
#line 533 "xi-grammar.y"
    { 
		  /*
		  printf("Warning: Message attributes are being phased out.\n");
		  printf("Warning: Please remove them from interface files.\n");
		  */
		  (yyval.intval) = (yyvsp[(2) - (3)].intval); 
		}
    break;

  case 144:
#line 543 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (1)].intval); }
    break;

  case 145:
#line 545 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (3)].intval) | (yyvsp[(3) - (3)].intval); }
    break;

  case 146:
#line 549 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 147:
#line 551 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 148:
#line 555 "xi-grammar.y"
    { (yyval.cattr) = 0; }
    break;

  case 149:
#line 557 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(2) - (3)].cattr); }
    break;

  case 150:
#line 561 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (1)].cattr); }
    break;

  case 151:
#line 563 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (3)].cattr) | (yyvsp[(3) - (3)].cattr); }
    break;

  case 152:
#line 567 "xi-grammar.y"
    { python_doc = NULL; (yyval.intval) = 0; }
    break;

  case 153:
#line 569 "xi-grammar.y"
    { python_doc = (yyvsp[(1) - (1)].strval); (yyval.intval) = 0; }
    break;

  case 154:
#line 573 "xi-grammar.y"
    { (yyval.cattr) = Chare::CPYTHON; }
    break;

  case 155:
#line 577 "xi-grammar.y"
    { (yyval.cattr) = 0; }
    break;

  case 156:
#line 579 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(2) - (3)].cattr); }
    break;

  case 157:
#line 583 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (1)].cattr); }
    break;

  case 158:
#line 585 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (3)].cattr) | (yyvsp[(3) - (3)].cattr); }
    break;

  case 159:
#line 589 "xi-grammar.y"
    { (yyval.cattr) = Chare::CMIGRATABLE; }
    break;

  case 160:
#line 591 "xi-grammar.y"
    { (yyval.cattr) = Chare::CPYTHON; }
    break;

  case 161:
#line 595 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 162:
#line 597 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 163:
#line 600 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 164:
#line 602 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 165:
#line 605 "xi-grammar.y"
    { (yyval.mv) = new MsgVar((yyvsp[(2) - (5)].type), (yyvsp[(3) - (5)].strval), (yyvsp[(1) - (5)].intval), (yyvsp[(4) - (5)].intval)); }
    break;

  case 166:
#line 609 "xi-grammar.y"
    { (yyval.mvlist) = new MsgVarList((yyvsp[(1) - (1)].mv)); }
    break;

  case 167:
#line 611 "xi-grammar.y"
    { (yyval.mvlist) = new MsgVarList((yyvsp[(1) - (2)].mv), (yyvsp[(2) - (2)].mvlist)); }
    break;

  case 168:
#line 615 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, (yyvsp[(3) - (3)].ntype)); }
    break;

  case 169:
#line 617 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, (yyvsp[(3) - (6)].ntype), (yyvsp[(5) - (6)].mvlist)); }
    break;

  case 170:
#line 621 "xi-grammar.y"
    { (yyval.typelist) = 0; }
    break;

  case 171:
#line 623 "xi-grammar.y"
    { (yyval.typelist) = (yyvsp[(2) - (2)].typelist); }
    break;

  case 172:
#line 627 "xi-grammar.y"
    { (yyval.typelist) = new TypeList((yyvsp[(1) - (1)].ntype)); }
    break;

  case 173:
#line 629 "xi-grammar.y"
    { (yyval.typelist) = new TypeList((yyvsp[(1) - (3)].ntype), (yyvsp[(3) - (3)].typelist)); }
    break;

  case 174:
#line 633 "xi-grammar.y"
    { (yyval.chare) = new Chare(lineno, (yyvsp[(2) - (5)].cattr)|Chare::CCHARE, (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 175:
#line 635 "xi-grammar.y"
    { (yyval.chare) = new MainChare(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 176:
#line 639 "xi-grammar.y"
    { (yyval.chare) = new Group(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 177:
#line 643 "xi-grammar.y"
    { (yyval.chare) = new NodeGroup(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 178:
#line 647 "xi-grammar.y"
    {/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",(yyvsp[(2) - (4)].strval));
			(yyval.ntype) = new NamedType(buf); 
		}
    break;

  case 179:
#line 653 "xi-grammar.y"
    { (yyval.ntype) = new NamedType((yyvsp[(2) - (3)].strval)); }
    break;

  case 180:
#line 657 "xi-grammar.y"
    {  (yyval.chare) = new Array(lineno, (yyvsp[(2) - (6)].cattr), (yyvsp[(3) - (6)].ntype), (yyvsp[(4) - (6)].ntype), (yyvsp[(5) - (6)].typelist), (yyvsp[(6) - (6)].mbrlist)); }
    break;

  case 181:
#line 659 "xi-grammar.y"
    {  (yyval.chare) = new Array(lineno, (yyvsp[(3) - (6)].cattr), (yyvsp[(2) - (6)].ntype), (yyvsp[(4) - (6)].ntype), (yyvsp[(5) - (6)].typelist), (yyvsp[(6) - (6)].mbrlist)); }
    break;

  case 182:
#line 663 "xi-grammar.y"
    { (yyval.chare) = new Chare(lineno, (yyvsp[(2) - (5)].cattr)|Chare::CCHARE, new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist));}
    break;

  case 183:
#line 665 "xi-grammar.y"
    { (yyval.chare) = new MainChare(lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 184:
#line 669 "xi-grammar.y"
    { (yyval.chare) = new Group(lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 185:
#line 673 "xi-grammar.y"
    { (yyval.chare) = new NodeGroup( lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 186:
#line 677 "xi-grammar.y"
    { (yyval.chare) = new Array( lineno, 0, (yyvsp[(2) - (5)].ntype), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 187:
#line 681 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[(3) - (4)].strval))); }
    break;

  case 188:
#line 683 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[(3) - (7)].strval)), (yyvsp[(5) - (7)].mvlist)); }
    break;

  case 189:
#line 687 "xi-grammar.y"
    { (yyval.type) = 0; }
    break;

  case 190:
#line 689 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(2) - (2)].type); }
    break;

  case 191:
#line 693 "xi-grammar.y"
    { (yyval.strval) = 0; }
    break;

  case 192:
#line 695 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(2) - (2)].strval); }
    break;

  case 193:
#line 697 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(2) - (2)].strval); }
    break;

  case 194:
#line 699 "xi-grammar.y"
    {
		  XStr typeStr;
		  (yyvsp[(2) - (2)].ntype)->print(typeStr);
		  char *tmp = strdup(typeStr.get_string());
		  (yyval.strval) = tmp;
		}
    break;

  case 195:
#line 708 "xi-grammar.y"
    { (yyval.tvar) = new TType(new NamedType((yyvsp[(2) - (3)].strval)), (yyvsp[(3) - (3)].type)); }
    break;

  case 196:
#line 710 "xi-grammar.y"
    { (yyval.tvar) = new TFunc((yyvsp[(1) - (2)].ftype), (yyvsp[(2) - (2)].strval)); }
    break;

  case 197:
#line 712 "xi-grammar.y"
    { (yyval.tvar) = new TName((yyvsp[(1) - (3)].type), (yyvsp[(2) - (3)].strval), (yyvsp[(3) - (3)].strval)); }
    break;

  case 198:
#line 716 "xi-grammar.y"
    { (yyval.tvarlist) = new TVarList((yyvsp[(1) - (1)].tvar)); }
    break;

  case 199:
#line 718 "xi-grammar.y"
    { (yyval.tvarlist) = new TVarList((yyvsp[(1) - (3)].tvar), (yyvsp[(3) - (3)].tvarlist)); }
    break;

  case 200:
#line 722 "xi-grammar.y"
    { (yyval.tvarlist) = (yyvsp[(3) - (4)].tvarlist); }
    break;

  case 201:
#line 726 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
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
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].message)); (yyvsp[(2) - (2)].message)->setTemplate((yyval.templat)); }
    break;

  case 206:
#line 738 "xi-grammar.y"
    { (yyval.mbrlist) = 0; }
    break;

  case 207:
#line 740 "xi-grammar.y"
    { (yyval.mbrlist) = (yyvsp[(2) - (4)].mbrlist); }
    break;

  case 208:
#line 744 "xi-grammar.y"
    { 
                  if (!connectEntries.empty()) {
                    (yyval.mbrlist) = new AstChildren<Member>(connectEntries);
		  } else {
		    (yyval.mbrlist) = 0; 
                  }
		}
    break;

  case 209:
#line 752 "xi-grammar.y"
    { (yyval.mbrlist) = new AstChildren<Member>(-1, (yyvsp[(1) - (2)].member), (yyvsp[(2) - (2)].mbrlist)); }
    break;

  case 210:
#line 756 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].readonly); }
    break;

  case 211:
#line 758 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].readonly); }
    break;

  case 213:
#line 761 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].member); }
    break;

  case 214:
#line 763 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(2) - (2)].pupable); }
    break;

  case 215:
#line 765 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(2) - (2)].includeFile); }
    break;

  case 216:
#line 767 "xi-grammar.y"
    { (yyval.member) = new ClassDeclaration(lineno,(yyvsp[(2) - (2)].strval)); }
    break;

  case 217:
#line 771 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 1); }
    break;

  case 218:
#line 773 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 1); }
    break;

  case 219:
#line 775 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[(3) - (9)].strval)) + '<' +
					    ((yyvsp[(5) - (9)].tparlist))->to_string() + '>').c_str()),
				    1);
		}
    break;

  case 220:
#line 781 "xi-grammar.y"
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[(1) - (3)]).first_column, (yylsp[(1) - (3)]).last_column, (yylsp[(1) - (3)]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 1);
		}
    break;

  case 221:
#line 787 "xi-grammar.y"
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[(1) - (6)]).first_column, (yylsp[(1) - (6)]).last_column, (yylsp[(1) - (6)]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 1);
		}
    break;

  case 222:
#line 796 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 0); }
    break;

  case 223:
#line 798 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 0); }
    break;

  case 224:
#line 800 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[(3) - (9)].strval)) + '<' +
					    ((yyvsp[(5) - (9)].tparlist))->to_string() + '>').c_str()),
				    0);
		}
    break;

  case 225:
#line 806 "xi-grammar.y"
    {
                  InitCall* rtn = new InitCall(lineno, (yyvsp[(6) - (9)].strval), 0);
                  rtn->setAccel();
                  (yyval.member) = rtn;
		}
    break;

  case 226:
#line 814 "xi-grammar.y"
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[(1) - (1)].ntype),0); }
    break;

  case 227:
#line 816 "xi-grammar.y"
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[(1) - (3)].ntype),(yyvsp[(3) - (3)].pupable)); }
    break;

  case 228:
#line 819 "xi-grammar.y"
    { (yyval.includeFile) = new IncludeFile(lineno,(yyvsp[(1) - (1)].strval)); }
    break;

  case 229:
#line 823 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].member); }
    break;

  case 230:
#line 827 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].entry); }
    break;

  case 231:
#line 829 "xi-grammar.y"
    {
                  (yyvsp[(2) - (2)].entry)->tspec = (yyvsp[(1) - (2)].tvarlist);
                  (yyval.member) = (yyvsp[(2) - (2)].entry);
                }
    break;

  case 232:
#line 834 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (2)].member); }
    break;

  case 233:
#line 836 "xi-grammar.y"
    {
          ERROR("invalid SDAG member",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
    break;

  case 234:
#line 844 "xi-grammar.y"
    { (yyval.member) = 0; }
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
#line 867 "xi-grammar.y"
    { 
                  (yyval.entry) = new Entry(lineno, (yyvsp[(2) - (7)].intval), (yyvsp[(3) - (7)].type), (yyvsp[(4) - (7)].strval), (yyvsp[(5) - (7)].plist), (yyvsp[(6) - (7)].val), (yyvsp[(7) - (7)].sentry), (const char *) NULL, (yylsp[(1) - (7)]).first_line, (yyloc).last_line);
		  if ((yyvsp[(7) - (7)].sentry) != 0) { 
		    (yyvsp[(7) - (7)].sentry)->con1 = new SdagConstruct(SIDENT, (yyvsp[(4) - (7)].strval));
                    (yyvsp[(7) - (7)].sentry)->setEntry((yyval.entry));
                    (yyvsp[(7) - (7)].sentry)->param = new ParamList((yyvsp[(5) - (7)].plist));
                  }
		}
    break;

  case 246:
#line 876 "xi-grammar.y"
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

  case 247:
#line 892 "xi-grammar.y"
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

  case 248:
#line 908 "xi-grammar.y"
    { (yyval.accelBlock) = new AccelBlock(lineno, new XStr((yyvsp[(3) - (5)].strval))); }
    break;

  case 249:
#line 910 "xi-grammar.y"
    { (yyval.accelBlock) = new AccelBlock(lineno, NULL); }
    break;

  case 250:
#line 914 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 251:
#line 918 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 252:
#line 920 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(2) - (3)].intval); }
    break;

  case 253:
#line 922 "xi-grammar.y"
    { ERROR("invalid entry method attribute list",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
    break;

  case 254:
#line 929 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (1)].intval); }
    break;

  case 255:
#line 931 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (3)].intval) | (yyvsp[(3) - (3)].intval); }
    break;

  case 256:
#line 935 "xi-grammar.y"
    { (yyval.intval) = STHREADED; }
    break;

  case 257:
#line 937 "xi-grammar.y"
    { (yyval.intval) = SSYNC; }
    break;

  case 258:
#line 939 "xi-grammar.y"
    { (yyval.intval) = SIGET; }
    break;

  case 259:
#line 941 "xi-grammar.y"
    { (yyval.intval) = SLOCKED; }
    break;

  case 260:
#line 943 "xi-grammar.y"
    { (yyval.intval) = SCREATEHERE; }
    break;

  case 261:
#line 945 "xi-grammar.y"
    { (yyval.intval) = SCREATEHOME; }
    break;

  case 262:
#line 947 "xi-grammar.y"
    { (yyval.intval) = SNOKEEP; }
    break;

  case 263:
#line 949 "xi-grammar.y"
    { (yyval.intval) = SNOTRACE; }
    break;

  case 264:
#line 951 "xi-grammar.y"
    { (yyval.intval) = SAPPWORK; }
    break;

  case 265:
#line 953 "xi-grammar.y"
    { (yyval.intval) = SIMMEDIATE; }
    break;

  case 266:
#line 955 "xi-grammar.y"
    { (yyval.intval) = SSKIPSCHED; }
    break;

  case 267:
#line 957 "xi-grammar.y"
    { (yyval.intval) = SINLINE; }
    break;

  case 268:
#line 959 "xi-grammar.y"
    { (yyval.intval) = SLOCAL; }
    break;

  case 269:
#line 961 "xi-grammar.y"
    { (yyval.intval) = SPYTHON; }
    break;

  case 270:
#line 963 "xi-grammar.y"
    { (yyval.intval) = SMEM; }
    break;

  case 271:
#line 965 "xi-grammar.y"
    { (yyval.intval) = SREDUCE; }
    break;

  case 272:
#line 967 "xi-grammar.y"
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

  case 273:
#line 979 "xi-grammar.y"
    { (yyval.intval) = SSCATTERV; }
    break;

  case 274:
#line 981 "xi-grammar.y"
    {
		  ERROR("invalid entry method attribute",
		        (yylsp[(1) - (1)]).first_column, (yylsp[(1) - (1)]).last_column);
		  yyclearin;
		  yyerrok;
		}
    break;

  case 275:
#line 990 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
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
#line 998 "xi-grammar.y"
    { (yyval.strval) = ""; }
    break;

  case 279:
#line 1000 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 280:
#line 1002 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[(1) - (3)].strval))+strlen((yyvsp[(3) - (3)].strval))+3];
			sprintf(tmp,"%s, %s", (yyvsp[(1) - (3)].strval), (yyvsp[(3) - (3)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 281:
#line 1010 "xi-grammar.y"
    { (yyval.strval) = ""; }
    break;

  case 282:
#line 1012 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 283:
#line 1014 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s[%s]%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 284:
#line 1020 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s{%s}%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 285:
#line 1026 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s(%s)%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 286:
#line 1032 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(2) - (4)].strval))+strlen((yyvsp[(4) - (4)].strval))+3];
			sprintf(tmp,"(%s)%s", (yyvsp[(2) - (4)].strval), (yyvsp[(4) - (4)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 287:
#line 1040 "xi-grammar.y"
    {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			(yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].type),(yyvsp[(2) - (3)].strval));
		}
    break;

  case 288:
#line 1047 "xi-grammar.y"
    { 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			(yyval.intval) = 0;
		}
    break;

  case 289:
#line 1055 "xi-grammar.y"
    { 
			in_braces=0;
			(yyval.intval) = 0;
		}
    break;

  case 290:
#line 1062 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (1)].type));}
    break;

  case 291:
#line 1064 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].type),(yyvsp[(2) - (3)].strval)); (yyval.pname)->setConditional((yyvsp[(3) - (3)].intval)); }
    break;

  case 292:
#line 1066 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (4)].type),(yyvsp[(2) - (4)].strval),0,(yyvsp[(4) - (4)].val));}
    break;

  case 293:
#line 1068 "xi-grammar.y"
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].pname)->getType(), (yyvsp[(1) - (3)].pname)->getName() ,(yyvsp[(2) - (3)].strval));
		}
    break;

  case 294:
#line 1074 "xi-grammar.y"
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READONLY; }
    break;

  case 295:
#line 1075 "xi-grammar.y"
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READWRITE; }
    break;

  case 296:
#line 1076 "xi-grammar.y"
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_WRITEONLY; }
    break;

  case 297:
#line 1079 "xi-grammar.y"
    { (yyval.xstrptr) = new XStr((yyvsp[(1) - (1)].strval)); }
    break;

  case 298:
#line 1080 "xi-grammar.y"
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "->" << (yyvsp[(4) - (4)].strval); }
    break;

  case 299:
#line 1081 "xi-grammar.y"
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[(1) - (3)].xstrptr)) << "." << (yyvsp[(3) - (3)].strval); }
    break;

  case 300:
#line 1083 "xi-grammar.y"
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "[" << *((yyvsp[(3) - (4)].xstrptr)) << "]";
                  delete (yyvsp[(1) - (4)].xstrptr);
                  delete (yyvsp[(3) - (4)].xstrptr);
                }
    break;

  case 301:
#line 1090 "xi-grammar.y"
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "[" << (yyvsp[(3) - (4)].strval) << "]";
                  delete (yyvsp[(1) - (4)].xstrptr);
                }
    break;

  case 302:
#line 1096 "xi-grammar.y"
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "(" << *((yyvsp[(3) - (4)].xstrptr)) << ")";
                  delete (yyvsp[(1) - (4)].xstrptr);
                  delete (yyvsp[(3) - (4)].xstrptr);
                }
    break;

  case 303:
#line 1105 "xi-grammar.y"
    {
                  in_bracket = 0;
                  (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].pname)->getType(), (yyvsp[(1) - (3)].pname)->getName(), (yyvsp[(2) - (3)].strval));
                }
    break;

  case 304:
#line 1112 "xi-grammar.y"
    {
                  (yyval.pname) = new Parameter(lineno, (yyvsp[(3) - (7)].type), (yyvsp[(4) - (7)].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[(6) - (7)].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[(1) - (7)].intval));
                }
    break;

  case 305:
#line 1118 "xi-grammar.y"
    {
		  (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (5)].type), (yyvsp[(2) - (5)].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[(4) - (5)].xstrptr));
                  (yyval.pname)->setAccelBufferType(Parameter::ACCEL_BUFFER_TYPE_READWRITE);
		}
    break;

  case 306:
#line 1124 "xi-grammar.y"
    {
                  (yyval.pname) = (yyvsp[(3) - (6)].pname);
                  (yyval.pname)->setAccelInstName((yyvsp[(5) - (6)].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[(1) - (6)].intval));
		}
    break;

  case 307:
#line 1132 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (1)].pname)); }
    break;

  case 308:
#line 1134 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (3)].pname),(yyvsp[(3) - (3)].plist)); }
    break;

  case 309:
#line 1138 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (1)].pname)); }
    break;

  case 310:
#line 1140 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (3)].pname),(yyvsp[(3) - (3)].plist)); }
    break;

  case 311:
#line 1144 "xi-grammar.y"
    { (yyval.plist) = (yyvsp[(2) - (3)].plist); }
    break;

  case 312:
#line 1146 "xi-grammar.y"
    { (yyval.plist) = new ParamList(new Parameter(0, new BuiltinType("void"))); }
    break;

  case 313:
#line 1150 "xi-grammar.y"
    { (yyval.plist) = (yyvsp[(2) - (3)].plist); }
    break;

  case 314:
#line 1152 "xi-grammar.y"
    { (yyval.plist) = 0; }
    break;

  case 315:
#line 1156 "xi-grammar.y"
    { (yyval.val) = 0; }
    break;

  case 316:
#line 1158 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(3) - (3)].strval)); }
    break;

  case 317:
#line 1162 "xi-grammar.y"
    { (yyval.sentry) = 0; }
    break;

  case 318:
#line 1164 "xi-grammar.y"
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[(1) - (1)].sc)); }
    break;

  case 319:
#line 1166 "xi-grammar.y"
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[(2) - (4)].slist)); }
    break;

  case 320:
#line 1170 "xi-grammar.y"
    { (yyval.slist) = new SListConstruct((yyvsp[(1) - (1)].sc)); }
    break;

  case 321:
#line 1172 "xi-grammar.y"
    { (yyval.slist) = new SListConstruct((yyvsp[(1) - (2)].sc), (yyvsp[(2) - (2)].slist));  }
    break;

  case 322:
#line 1176 "xi-grammar.y"
    { (yyval.olist) = new OListConstruct((yyvsp[(1) - (1)].sc)); }
    break;

  case 323:
#line 1178 "xi-grammar.y"
    { (yyval.olist) = new OListConstruct((yyvsp[(1) - (2)].sc), (yyvsp[(2) - (2)].slist)); }
    break;

  case 324:
#line 1182 "xi-grammar.y"
    { (yyval.clist) = new CaseListConstruct((yyvsp[(1) - (1)].when)); }
    break;

  case 325:
#line 1184 "xi-grammar.y"
    { (yyval.clist) = new CaseListConstruct((yyvsp[(1) - (2)].when), (yyvsp[(2) - (2)].clist)); }
    break;

  case 326:
#line 1186 "xi-grammar.y"
    {
		  ERROR("case blocks can only contain when clauses",
		        (yylsp[(1) - (1)]).first_column, (yylsp[(1) - (1)]).last_column);
		  (yyval.clist) = 0;
		}
    break;

  case 327:
#line 1194 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 328:
#line 1196 "xi-grammar.y"
    { (yyval.strval) = 0; }
    break;

  case 329:
#line 1200 "xi-grammar.y"
    { (yyval.when) = new WhenConstruct((yyvsp[(2) - (4)].entrylist), 0); }
    break;

  case 330:
#line 1202 "xi-grammar.y"
    { (yyval.when) = new WhenConstruct((yyvsp[(2) - (3)].entrylist), (yyvsp[(3) - (3)].sc)); }
    break;

  case 331:
#line 1204 "xi-grammar.y"
    { (yyval.when) = new WhenConstruct((yyvsp[(2) - (5)].entrylist), (yyvsp[(4) - (5)].slist)); }
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
#line 1234 "xi-grammar.y"
    { (yyval.sc) = new AtomicConstruct((yyvsp[(4) - (6)].strval), (yyvsp[(2) - (6)].strval), (yylsp[(3) - (6)]).first_line); }
    break;

  case 345:
#line 1236 "xi-grammar.y"
    { (yyval.sc) = new OverlapConstruct((yyvsp[(3) - (4)].olist)); }
    break;

  case 346:
#line 1238 "xi-grammar.y"
    { (yyval.sc) = (yyvsp[(1) - (1)].when); }
    break;

  case 347:
#line 1240 "xi-grammar.y"
    { (yyval.sc) = new CaseConstruct((yyvsp[(3) - (4)].clist)); }
    break;

  case 348:
#line 1242 "xi-grammar.y"
    { (yyval.sc) = new ForConstruct((yyvsp[(3) - (11)].intexpr), (yyvsp[(5) - (11)].intexpr), (yyvsp[(7) - (11)].intexpr), (yyvsp[(10) - (11)].slist)); }
    break;

  case 349:
#line 1244 "xi-grammar.y"
    { (yyval.sc) = new ForConstruct((yyvsp[(3) - (9)].intexpr), (yyvsp[(5) - (9)].intexpr), (yyvsp[(7) - (9)].intexpr), (yyvsp[(9) - (9)].sc)); }
    break;

  case 350:
#line 1246 "xi-grammar.y"
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[(3) - (12)].strval)), (yyvsp[(6) - (12)].intexpr),
		             (yyvsp[(8) - (12)].intexpr), (yyvsp[(10) - (12)].intexpr), (yyvsp[(12) - (12)].sc)); }
    break;

  case 351:
#line 1249 "xi-grammar.y"
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[(3) - (14)].strval)), (yyvsp[(6) - (14)].intexpr),
		             (yyvsp[(8) - (14)].intexpr), (yyvsp[(10) - (14)].intexpr), (yyvsp[(13) - (14)].slist)); }
    break;

  case 352:
#line 1252 "xi-grammar.y"
    { (yyval.sc) = new IfConstruct((yyvsp[(3) - (6)].intexpr), (yyvsp[(5) - (6)].sc), (yyvsp[(6) - (6)].sc)); }
    break;

  case 353:
#line 1254 "xi-grammar.y"
    { (yyval.sc) = new IfConstruct((yyvsp[(3) - (8)].intexpr), (yyvsp[(6) - (8)].slist), (yyvsp[(8) - (8)].sc)); }
    break;

  case 354:
#line 1256 "xi-grammar.y"
    { (yyval.sc) = new WhileConstruct((yyvsp[(3) - (5)].intexpr), (yyvsp[(5) - (5)].sc)); }
    break;

  case 355:
#line 1258 "xi-grammar.y"
    { (yyval.sc) = new WhileConstruct((yyvsp[(3) - (7)].intexpr), (yyvsp[(6) - (7)].slist)); }
    break;

  case 356:
#line 1260 "xi-grammar.y"
    { (yyval.sc) = new AtomicConstruct((yyvsp[(2) - (4)].strval), NULL, (yyloc).first_line); }
    break;

  case 357:
#line 1262 "xi-grammar.y"
    {
		  ERROR("unknown SDAG construct or malformed entry method declaration.\n"
		        "You may have forgotten to terminate a previous entry method declaration with a"
		        " semicolon or forgotten to mark a block of sequential SDAG code as 'serial'",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
    break;

  case 358:
#line 1272 "xi-grammar.y"
    { (yyval.sc) = 0; }
    break;

  case 359:
#line 1274 "xi-grammar.y"
    { (yyval.sc) = new ElseConstruct((yyvsp[(2) - (2)].sc)); }
    break;

  case 360:
#line 1276 "xi-grammar.y"
    { (yyval.sc) = new ElseConstruct((yyvsp[(3) - (4)].slist)); }
    break;

  case 361:
#line 1280 "xi-grammar.y"
    { (yyval.intexpr) = new IntExprConstruct((yyvsp[(1) - (1)].strval)); }
    break;

  case 362:
#line 1284 "xi-grammar.y"
    { in_int_expr = 0; (yyval.intval) = 0; }
    break;

  case 363:
#line 1288 "xi-grammar.y"
    { in_int_expr = 1; (yyval.intval) = 0; }
    break;

  case 364:
#line 1292 "xi-grammar.y"
    {
		  (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[(1) - (2)].strval), (yyvsp[(2) - (2)].plist), 0, 0, 0, (yyloc).first_line, (yyloc).last_line);
		}
    break;

  case 365:
#line 1296 "xi-grammar.y"
    {
		  (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[(1) - (5)].strval), (yyvsp[(5) - (5)].plist), 0, 0, (yyvsp[(3) - (5)].strval), (yyloc).first_line, (yyloc).last_line);
		}
    break;

  case 366:
#line 1302 "xi-grammar.y"
    { (yyval.entrylist) = new EntryList((yyvsp[(1) - (1)].entry)); }
    break;

  case 367:
#line 1304 "xi-grammar.y"
    { (yyval.entrylist) = new EntryList((yyvsp[(1) - (3)].entry),(yyvsp[(3) - (3)].entrylist)); }
    break;

  case 368:
#line 1308 "xi-grammar.y"
    { in_bracket=1; }
    break;

  case 369:
#line 1311 "xi-grammar.y"
    { in_bracket=0; }
    break;

  case 370:
#line 1315 "xi-grammar.y"
    { if (!macroDefined((yyvsp[(2) - (2)].strval), 1)) in_comment = 1; }
    break;

  case 371:
#line 1319 "xi-grammar.y"
    { if (!macroDefined((yyvsp[(2) - (2)].strval), 0)) in_comment = 1; }
    break;


/* Line 1267 of yacc.c.  */
#line 4482 "y.tab.c"
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


#line 1322 "xi-grammar.y"


void yyerror(const char *msg) { }

