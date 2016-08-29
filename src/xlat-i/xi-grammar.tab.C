/* A Bison parser, made by GNU Bison 2.5.  */

/* Bison implementation for Yacc-like parsers in C
   
      Copyright (C) 1984, 1989-1990, 2000-2011 Free Software Foundation, Inc.
   
   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   
   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

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
#define YYBISON_VERSION "2.5"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1

/* Using locations.  */
#define YYLSP_NEEDED 1



/* Copy the first part of user declarations.  */

/* Line 268 of yacc.c  */
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


/* Line 268 of yacc.c  */
#line 119 "y.tab.c"

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
     APPWORK = 290,
     VOID = 291,
     CONST = 292,
     PACKED = 293,
     VARSIZE = 294,
     ENTRY = 295,
     FOR = 296,
     FORALL = 297,
     WHILE = 298,
     WHEN = 299,
     OVERLAP = 300,
     ATOMIC = 301,
     IF = 302,
     ELSE = 303,
     PYTHON = 304,
     LOCAL = 305,
     NAMESPACE = 306,
     USING = 307,
     IDENT = 308,
     NUMBER = 309,
     LITERAL = 310,
     CPROGRAM = 311,
     HASHIF = 312,
     HASHIFDEF = 313,
     INT = 314,
     LONG = 315,
     SHORT = 316,
     CHAR = 317,
     FLOAT = 318,
     DOUBLE = 319,
     UNSIGNED = 320,
     ACCEL = 321,
     READWRITE = 322,
     WRITEONLY = 323,
     ACCELBLOCK = 324,
     MEMCRITICAL = 325,
     REDUCTIONTARGET = 326,
     CASE = 327
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
#define APPWORK 290
#define VOID 291
#define CONST 292
#define PACKED 293
#define VARSIZE 294
#define ENTRY 295
#define FOR 296
#define FORALL 297
#define WHILE 298
#define WHEN 299
#define OVERLAP 300
#define ATOMIC 301
#define IF 302
#define ELSE 303
#define PYTHON 304
#define LOCAL 305
#define NAMESPACE 306
#define USING 307
#define IDENT 308
#define NUMBER 309
#define LITERAL 310
#define CPROGRAM 311
#define HASHIF 312
#define HASHIFDEF 313
#define INT 314
#define LONG 315
#define SHORT 316
#define CHAR 317
#define FLOAT 318
#define DOUBLE 319
#define UNSIGNED 320
#define ACCEL 321
#define READWRITE 322
#define WRITEONLY 323
#define ACCELBLOCK 324
#define MEMCRITICAL 325
#define REDUCTIONTARGET 326
#define CASE 327




#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE
{

/* Line 293 of yacc.c  */
#line 51 "xi-grammar.y"

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



/* Line 293 of yacc.c  */
#line 345 "y.tab.c"
} YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
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


/* Line 343 of yacc.c  */
#line 370 "y.tab.c"

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
YYID (int yyi)
#else
static int
YYID (yyi)
    int yyi;
#endif
{
  return yyi;
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
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
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
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined YYMALLOC || defined malloc) \
	     && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS && (defined __STDC__ || defined __C99__FUNC__ \
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
  yytype_int16 yyss_alloc;
  YYSTYPE yyvs_alloc;
  YYLTYPE yyls_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE) + sizeof (YYLTYPE)) \
      + 2 * YYSTACK_GAP_MAXIMUM)

# define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)				\
    do									\
      {									\
	YYSIZE_T yynewbytes;						\
	YYCOPY (&yyptr->Stack_alloc, Stack, yysize);			\
	Stack = &yyptr->Stack_alloc;					\
	yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
	yyptr += yynewbytes / sizeof (*yyptr);				\
      }									\
    while (YYID (0))

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
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
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  55
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   1457

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  89
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  116
/* YYNRULES -- Number of rules.  */
#define YYNRULES  365
/* YYNRULES -- Number of states.  */
#define YYNSTATES  711

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   327

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    83,     2,
      81,    82,    80,     2,    77,    87,    88,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    74,    73,
      78,    86,    79,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    84,     2,    85,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    75,     2,    76,     2,     2,     2,     2,
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
      65,    66,    67,    68,    69,    70,    71,    72
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
     797,   799,   801,   802,   804,   808,   809,   811,   817,   823,
     829,   834,   838,   840,   842,   844,   848,   853,   857,   859,
     861,   863,   865,   870,   874,   879,   884,   889,   893,   901,
     907,   914,   916,   920,   922,   926,   930,   933,   937,   940,
     941,   945,   947,   949,   954,   956,   959,   961,   964,   966,
     969,   971,   973,   974,   979,   983,   989,   995,  1000,  1005,
    1017,  1027,  1040,  1055,  1062,  1071,  1077,  1085,  1089,  1095,
    1100,  1102,  1107,  1119,  1129,  1142,  1157,  1164,  1173,  1179,
    1187,  1191,  1193,  1194,  1197,  1202,  1204,  1206,  1208,  1211,
    1217,  1219,  1223,  1225,  1227,  1230
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
      90,     0,    -1,    91,    -1,    -1,    96,    91,    -1,    -1,
       5,    -1,    -1,    73,    -1,    53,    -1,     3,    -1,     4,
      -1,     5,    -1,     7,    -1,     8,    -1,     9,    -1,    11,
      -1,    12,    -1,    13,    -1,    14,    -1,    15,    -1,    19,
      -1,    20,    -1,    21,    -1,    22,    -1,    23,    -1,    24,
      -1,    25,    -1,    26,    -1,    27,    -1,    28,    -1,    29,
      -1,    30,    -1,    31,    -1,    32,    -1,    33,    -1,    34,
      -1,    35,    -1,    38,    -1,    39,    -1,    40,    -1,    41,
      -1,    42,    -1,    43,    -1,    44,    -1,    45,    -1,    46,
      -1,    47,    -1,    48,    -1,    50,    -1,    52,    -1,    66,
      -1,    69,    -1,    70,    -1,    71,    -1,    72,    -1,    53,
      -1,    95,    74,    74,    53,    -1,     3,    94,    97,    -1,
       4,    94,    97,    -1,    73,    -1,    75,    98,    76,    93,
      -1,    -1,   100,    98,    -1,    52,    51,    95,    -1,    52,
      95,    -1,    92,   157,    -1,    92,   136,    -1,     5,    40,
     167,   107,    94,   104,   184,    -1,    92,    75,    98,    76,
      93,    -1,    51,    94,    75,    98,    76,    -1,    99,    73,
      -1,    99,   164,    -1,    92,    96,    -1,    92,   139,    -1,
      92,   140,    -1,    92,   141,    -1,    92,   143,    -1,    92,
     154,    -1,   203,    -1,   204,    -1,   166,    -1,     1,    -1,
     115,    -1,    54,    -1,    55,    -1,   101,    -1,   101,    77,
     102,    -1,    -1,   102,    -1,    -1,    78,   103,    79,    -1,
      59,    -1,    60,    -1,    61,    -1,    62,    -1,    65,    59,
      -1,    65,    60,    -1,    65,    60,    59,    -1,    65,    60,
      60,    -1,    65,    61,    -1,    65,    62,    -1,    60,    60,
      -1,    63,    -1,    64,    -1,    60,    64,    -1,    36,    -1,
      94,   104,    -1,    95,   104,    -1,   105,    -1,   107,    -1,
     108,    80,    -1,   109,    80,    -1,   110,    80,    -1,   112,
      81,    80,    94,    82,    81,   182,    82,    -1,   108,    -1,
     109,    -1,   110,    -1,   111,    -1,    37,   112,    -1,   112,
      37,    -1,   108,    -1,   109,    -1,   110,    -1,    37,   113,
      -1,   113,    37,    -1,   113,    83,    -1,   113,    -1,   112,
      83,    -1,   112,    -1,   173,    -1,   201,   116,   202,    -1,
      -1,   117,   118,    -1,     6,   115,    95,   118,    -1,     6,
      16,   108,    80,    95,   118,    -1,    -1,    36,    -1,    -1,
      84,   123,    85,    -1,   124,    -1,   124,    77,   123,    -1,
      38,    -1,    39,    -1,    -1,    84,   126,    85,    -1,   131,
      -1,   131,    77,   126,    -1,    -1,    55,    -1,    49,    -1,
      -1,    84,   130,    85,    -1,   128,    -1,   128,    77,   130,
      -1,    30,    -1,    49,    -1,    -1,    17,    -1,    -1,    84,
      85,    -1,   132,   115,    94,   133,    73,    -1,   134,    -1,
     134,   135,    -1,    16,   122,   106,    -1,    16,   122,   106,
      75,   135,    76,    -1,    -1,    74,   138,    -1,   107,    -1,
     107,    77,   138,    -1,    11,   125,   106,   137,   155,    -1,
      12,   125,   106,   137,   155,    -1,    13,   125,   106,   137,
     155,    -1,    14,   125,   106,   137,   155,    -1,    84,    54,
      94,    85,    -1,    84,    94,    85,    -1,    15,   129,   142,
     106,   137,   155,    -1,    15,   142,   129,   106,   137,   155,
      -1,    11,   125,    94,   137,   155,    -1,    12,   125,    94,
     137,   155,    -1,    13,   125,    94,   137,   155,    -1,    14,
     125,    94,   137,   155,    -1,    15,   142,    94,   137,   155,
      -1,    16,   122,    94,    73,    -1,    16,   122,    94,    75,
     135,    76,    73,    -1,    -1,    86,   115,    -1,    -1,    86,
      54,    -1,    86,    55,    -1,    86,   107,    -1,    18,    94,
     149,    -1,   111,   150,    -1,   115,    94,   150,    -1,   151,
      -1,   151,    77,   152,    -1,    22,    78,   152,    79,    -1,
     153,   144,    -1,   153,   145,    -1,   153,   146,    -1,   153,
     147,    -1,   153,   148,    -1,    73,    -1,    75,   156,    76,
      93,    -1,    -1,   162,   156,    -1,   119,    -1,   120,    -1,
     159,    -1,   158,    -1,    10,   160,    -1,    19,   161,    -1,
      18,    94,    -1,     8,   121,    95,    -1,     8,   121,    95,
      81,   121,    82,    -1,     8,   121,    95,    78,   102,    79,
      81,   121,    82,    -1,     7,   121,    95,    -1,     7,   121,
      95,    81,   121,    82,    -1,     9,   121,    95,    -1,     9,
     121,    95,    81,   121,    82,    -1,     9,   121,    95,    78,
     102,    79,    81,   121,    82,    -1,     9,    84,    66,    85,
     121,    95,    81,   121,    82,    -1,   107,    -1,   107,    77,
     160,    -1,    55,    -1,   163,    -1,   165,    -1,   153,   165,
      -1,   157,    73,    -1,     1,    -1,    40,    -1,    76,    -1,
       7,    -1,     8,    -1,     9,    -1,    11,    -1,    12,    -1,
      15,    -1,    13,    -1,    14,    -1,     6,    -1,    40,   168,
     167,    94,   184,   186,   187,    -1,    40,   168,    94,   184,
     187,    -1,    40,    84,    66,    85,    36,    94,   184,   185,
     175,   173,   176,    94,    73,    -1,    69,   175,   173,   176,
      73,    -1,    69,    73,    -1,   114,    -1,    -1,    84,   169,
      85,    -1,     1,    -1,   170,    -1,   170,    77,   169,    -1,
      21,    -1,    23,    -1,    24,    -1,    25,    -1,    31,    -1,
      32,    -1,    33,    -1,    34,    -1,    35,    -1,    26,    -1,
      27,    -1,    28,    -1,    50,    -1,    49,   127,    -1,    70,
      -1,    71,    -1,     1,    -1,    55,    -1,    54,    -1,    95,
      -1,    -1,    56,    -1,    56,    77,   172,    -1,    -1,    56,
      -1,    56,    84,   173,    85,   173,    -1,    56,    75,   173,
      76,   173,    -1,    56,    81,   172,    82,   173,    -1,    81,
     173,    82,   173,    -1,   115,    94,    84,    -1,    75,    -1,
      76,    -1,   115,    -1,   115,    94,   132,    -1,   115,    94,
      86,   171,    -1,   174,   173,    85,    -1,     6,    -1,    67,
      -1,    68,    -1,    94,    -1,   179,    87,    79,    94,    -1,
     179,    88,    94,    -1,   179,    84,   179,    85,    -1,   179,
      84,    54,    85,    -1,   179,    81,   179,    82,    -1,   174,
     173,    85,    -1,   178,    74,   115,    94,    78,   179,    79,
      -1,   115,    94,    78,   179,    79,    -1,   178,    74,   180,
      78,   179,    79,    -1,   177,    -1,   177,    77,   182,    -1,
     181,    -1,   181,    77,   183,    -1,    81,   182,    82,    -1,
      81,    82,    -1,    84,   183,    85,    -1,    84,    85,    -1,
      -1,    20,    86,    54,    -1,    73,    -1,   194,    -1,    75,
     188,    76,    93,    -1,   194,    -1,   194,   188,    -1,   194,
      -1,   194,   188,    -1,   192,    -1,   192,   190,    -1,   193,
      -1,    55,    -1,    -1,    44,   200,    75,    76,    -1,    44,
     200,   194,    -1,    44,   200,    75,   188,    76,    -1,    46,
     191,   175,   173,   176,    -1,    45,    75,   189,    76,    -1,
      72,    75,   190,    76,    -1,    41,   198,   173,    73,   173,
      73,   173,   197,    75,   188,    76,    -1,    41,   198,   173,
      73,   173,    73,   173,   197,   194,    -1,    42,    84,    53,
      85,   198,   173,    74,   173,    77,   173,   197,   194,    -1,
      42,    84,    53,    85,   198,   173,    74,   173,    77,   173,
     197,    75,   188,    76,    -1,    47,   198,   173,   197,   194,
     195,    -1,    47,   198,   173,   197,    75,   188,    76,   195,
      -1,    43,   198,   173,   197,   194,    -1,    43,   198,   173,
     197,    75,   188,    76,    -1,   175,   173,   176,    -1,    46,
     191,   175,   173,   176,    -1,    45,    75,   189,    76,    -1,
     192,    -1,    72,    75,   190,    76,    -1,    41,   198,   196,
      73,   196,    73,   196,   197,    75,   188,    76,    -1,    41,
     198,   196,    73,   196,    73,   196,   197,   194,    -1,    42,
      84,    53,    85,   198,   196,    74,   196,    77,   196,   197,
     194,    -1,    42,    84,    53,    85,   198,   196,    74,   196,
      77,   196,   197,    75,   188,    76,    -1,    47,   198,   196,
     197,   194,   195,    -1,    47,   198,   196,   197,    75,   188,
      76,   195,    -1,    43,   198,   196,   197,   194,    -1,    43,
     198,   196,   197,    75,   188,    76,    -1,   175,   173,   176,
      -1,     1,    -1,    -1,    48,   194,    -1,    48,    75,   188,
      76,    -1,   173,    -1,    82,    -1,    81,    -1,    53,   184,
      -1,    53,   201,   173,   202,   184,    -1,   199,    -1,   199,
      77,   200,    -1,    84,    -1,    85,    -1,    57,    94,    -1,
      58,    94,    -1
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
     942,   944,   946,   948,   950,   952,   954,   956,   958,   967,
     969,   971,   976,   977,   979,   988,   989,   991,   997,  1003,
    1009,  1017,  1024,  1032,  1039,  1041,  1043,  1045,  1052,  1053,
    1054,  1057,  1058,  1059,  1060,  1067,  1073,  1082,  1089,  1095,
    1101,  1109,  1111,  1115,  1117,  1121,  1123,  1127,  1129,  1134,
    1135,  1139,  1141,  1143,  1147,  1149,  1153,  1155,  1159,  1161,
    1163,  1171,  1174,  1177,  1179,  1181,  1185,  1187,  1189,  1191,
    1193,  1195,  1197,  1199,  1201,  1203,  1205,  1207,  1211,  1213,
    1215,  1217,  1219,  1221,  1223,  1226,  1229,  1231,  1233,  1235,
    1237,  1239,  1250,  1251,  1253,  1257,  1261,  1265,  1269,  1273,
    1279,  1281,  1285,  1288,  1292,  1296
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
  "CREATEHERE", "CREATEHOME", "NOKEEP", "NOTRACE", "APPWORK", "VOID",
  "CONST", "PACKED", "VARSIZE", "ENTRY", "FOR", "FORALL", "WHILE", "WHEN",
  "OVERLAP", "ATOMIC", "IF", "ELSE", "PYTHON", "LOCAL", "NAMESPACE",
  "USING", "IDENT", "NUMBER", "LITERAL", "CPROGRAM", "HASHIF", "HASHIFDEF",
  "INT", "LONG", "SHORT", "CHAR", "FLOAT", "DOUBLE", "UNSIGNED", "ACCEL",
  "READWRITE", "WRITEONLY", "ACCELBLOCK", "MEMCRITICAL", "REDUCTIONTARGET",
  "CASE", "';'", "':'", "'{'", "'}'", "','", "'<'", "'>'", "'*'", "'('",
  "')'", "'&'", "'['", "']'", "'='", "'-'", "'.'", "$accept", "File",
  "ModuleEList", "OptExtern", "OptSemiColon", "Name", "QualName", "Module",
  "ConstructEList", "ConstructList", "ConstructSemi", "Construct",
  "TParam", "TParamList", "TParamEList", "OptTParams", "BuiltinType",
  "NamedType", "QualNamedType", "SimpleType", "OnePtrType", "PtrType",
  "FuncType", "BaseType", "BaseDataType", "RestrictedType", "Type",
  "ArrayDim", "Dim", "DimList", "Readonly", "ReadonlyMsg", "OptVoid",
  "MAttribs", "MAttribList", "MAttrib", "CAttribs", "CAttribList",
  "PythonOptions", "ArrayAttrib", "ArrayAttribs", "ArrayAttribList",
  "CAttrib", "OptConditional", "MsgArray", "Var", "VarList", "Message",
  "OptBaseList", "BaseList", "Chare", "Group", "NodeGroup",
  "ArrayIndexType", "Array", "TChare", "TGroup", "TNodeGroup", "TArray",
  "TMessage", "OptTypeInit", "OptNameInit", "TVar", "TVarList",
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
     325,   326,   327,    59,    58,   123,   125,    44,    60,    62,
      42,    40,    41,    38,    91,    93,    61,    45,    46
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    89,    90,    91,    91,    92,    92,    93,    93,    94,
      94,    94,    94,    94,    94,    94,    94,    94,    94,    94,
      94,    94,    94,    94,    94,    94,    94,    94,    94,    94,
      94,    94,    94,    94,    94,    94,    94,    94,    94,    94,
      94,    94,    94,    94,    94,    94,    94,    94,    94,    94,
      94,    94,    94,    94,    94,    94,    95,    95,    96,    96,
      97,    97,    98,    98,    99,    99,    99,    99,    99,   100,
     100,   100,   100,   100,   100,   100,   100,   100,   100,   100,
     100,   100,   100,   101,   101,   101,   102,   102,   103,   103,
     104,   104,   105,   105,   105,   105,   105,   105,   105,   105,
     105,   105,   105,   105,   105,   105,   105,   106,   107,   108,
     108,   109,   110,   110,   111,   112,   112,   112,   112,   112,
     112,   113,   113,   113,   113,   113,   114,   114,   115,   115,
     116,   117,   118,   118,   119,   120,   121,   121,   122,   122,
     123,   123,   124,   124,   125,   125,   126,   126,   127,   127,
     128,   129,   129,   130,   130,   131,   131,   132,   132,   133,
     133,   134,   135,   135,   136,   136,   137,   137,   138,   138,
     139,   139,   140,   141,   142,   142,   143,   143,   144,   144,
     145,   146,   147,   148,   148,   149,   149,   150,   150,   150,
     150,   151,   151,   151,   152,   152,   153,   154,   154,   154,
     154,   154,   155,   155,   156,   156,   157,   157,   157,   157,
     157,   157,   157,   158,   158,   158,   158,   158,   159,   159,
     159,   159,   160,   160,   161,   162,   163,   163,   163,   163,
     164,   164,   164,   164,   164,   164,   164,   164,   164,   164,
     164,   165,   165,   165,   166,   166,   167,   168,   168,   168,
     169,   169,   170,   170,   170,   170,   170,   170,   170,   170,
     170,   170,   170,   170,   170,   170,   170,   170,   170,   171,
     171,   171,   172,   172,   172,   173,   173,   173,   173,   173,
     173,   174,   175,   176,   177,   177,   177,   177,   178,   178,
     178,   179,   179,   179,   179,   179,   179,   180,   181,   181,
     181,   182,   182,   183,   183,   184,   184,   185,   185,   186,
     186,   187,   187,   187,   188,   188,   189,   189,   190,   190,
     190,   191,   191,   192,   192,   192,   193,   193,   193,   193,
     193,   193,   193,   193,   193,   193,   193,   193,   194,   194,
     194,   194,   194,   194,   194,   194,   194,   194,   194,   194,
     194,   194,   195,   195,   195,   196,   197,   198,   199,   199,
     200,   200,   201,   202,   203,   204
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
       1,     1,     0,     1,     3,     0,     1,     5,     5,     5,
       4,     3,     1,     1,     1,     3,     4,     3,     1,     1,
       1,     1,     4,     3,     4,     4,     4,     3,     7,     5,
       6,     1,     3,     1,     3,     3,     2,     3,     2,     0,
       3,     1,     1,     4,     1,     2,     1,     2,     1,     2,
       1,     1,     0,     4,     3,     5,     5,     4,     4,    11,
       9,    12,    14,     6,     8,     5,     7,     3,     5,     4,
       1,     4,    11,     9,    12,    14,     6,     8,     5,     7,
       3,     1,     0,     2,     4,     1,     1,     1,     2,     5,
       1,     3,     1,     1,     2,     2
};

/* YYDEFACT[STATE-NAME] -- Default reduction number in state STATE-NUM.
   Performed when YYTABLE doesn't specify something else to do.  Zero
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
     364,   365,   245,   282,   275,     0,   136,   136,   136,     0,
     144,   144,   144,   144,     0,   138,     0,     0,     0,     0,
      73,   206,   207,    67,    74,    75,    76,    77,     0,    78,
      66,   209,   208,     7,   240,   232,   233,   234,   235,   236,
     238,   239,   237,   230,    71,   231,    72,    63,   106,     0,
      92,    93,    94,    95,   103,   104,     0,    90,   109,   110,
     121,   122,   123,   127,   246,     0,     0,    64,     0,   276,
     275,     0,     0,     0,   115,   116,   117,   118,   129,     0,
     137,     0,     0,     0,     0,   222,   210,     0,     0,     0,
       0,     0,     0,     0,   151,     0,     0,   212,   224,   211,
       0,     0,   144,   144,   144,   144,     0,   138,   197,   198,
     199,   200,   201,     8,    61,   124,   102,   105,    96,    97,
     100,   101,    88,   108,   111,   112,   113,   125,   126,     0,
       0,     0,   275,   272,   275,     0,   283,     0,     0,   119,
     120,     0,   128,   132,   216,   213,     0,   218,     0,   155,
     156,     0,   146,    90,   166,   166,   166,   166,   150,     0,
       0,   153,     0,     0,     0,     0,     0,   142,   143,     0,
     140,   164,     0,   118,     0,   194,     0,     7,     0,     0,
       0,     0,     0,     0,    98,    99,    84,    85,    86,    89,
       0,    83,    90,    70,    57,     0,   273,     0,     0,   275,
     244,     0,     0,   362,   132,   134,   275,   136,     0,   136,
     136,     0,   136,   223,   145,     0,   107,     0,     0,     0,
       0,     0,     0,   175,     0,   152,   166,   166,   139,     0,
     157,   185,     0,   192,   187,     0,   196,    69,   166,   166,
     166,   166,   166,     0,     0,    91,     0,   275,   272,   275,
     275,   280,   132,     0,   133,     0,   130,     0,     0,     0,
       0,     0,     0,   147,   168,   167,   202,     0,   170,   171,
     172,   173,   174,   154,     0,     0,   141,   158,     0,   157,
       0,     0,   191,   188,   189,   190,   193,   195,     0,     0,
       0,     0,     0,   183,   157,    87,     0,    68,   278,   274,
     279,   277,   135,     0,   363,   131,   217,     0,   214,     0,
       0,   219,     0,   229,     0,     0,     0,     0,     0,   225,
     226,   176,   177,     0,   163,   165,   186,   178,   179,   180,
     181,   182,     0,   306,   284,   275,   301,     0,     0,   136,
     136,   136,   169,   249,     0,     0,   227,     7,   228,   205,
     159,     0,   157,     0,     0,   305,     0,     0,     0,     0,
     268,   252,   253,   254,   255,   261,   262,   263,   256,   257,
     258,   259,   260,   148,   264,     0,   266,   267,     0,   250,
      56,     0,     0,   203,     0,     0,   184,   281,     0,   285,
     287,   302,   114,   215,   221,   220,   149,   265,     0,   248,
       0,     0,     0,   160,   161,   270,   269,   271,   286,     0,
     251,   351,     0,     0,     0,     0,     0,   322,     0,     0,
     311,     0,   275,   242,   340,   312,   309,     0,   357,   275,
       0,   275,     0,   360,     0,     0,   321,     0,   275,     0,
       0,     0,     0,     0,     0,     0,   355,     0,     0,     0,
     358,   275,     0,     0,   324,     0,     0,   275,     0,     0,
       0,     0,     0,   322,     0,     0,   275,     0,   318,   320,
       7,   315,   350,     0,   241,     0,     0,   275,     0,   356,
       0,     0,   361,   323,     0,   339,   317,     0,     0,   275,
       0,   275,     0,     0,   275,     0,     0,   341,   319,   313,
     310,   288,   289,   290,   308,     0,     0,   303,     0,   275,
       0,   275,     0,   348,     0,   325,   338,     0,   352,     0,
       0,     0,     0,   275,     0,     0,   337,     0,     0,     0,
     307,     0,   275,     0,     0,   359,     0,     0,   346,   275,
       0,     0,   327,     0,     0,   328,     0,     0,   275,     0,
     304,     0,     0,   275,   349,   352,     0,   353,     0,   275,
       0,   335,   326,     0,   352,   291,     0,     0,     0,     0,
       0,     0,     0,   347,     0,   275,     0,     0,     0,   333,
     299,     0,     0,     0,     0,     0,   297,     0,   243,     0,
     343,   275,   354,     0,   275,   336,   352,     0,     0,     0,
       0,   293,     0,   300,     0,     0,     0,     0,   334,   296,
     295,   294,   292,   298,   342,     0,     0,   330,   275,     0,
     344,     0,     0,     0,   329,     0,   345,     0,   331,     0,
     332
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     3,     4,    68,   194,   233,   137,     5,    59,    69,
      70,    71,   268,   269,   270,   203,   138,   234,   139,   154,
     155,   156,   157,   158,   143,   144,   271,   335,   284,   285,
     101,   102,   161,   176,   249,   250,   168,   231,   477,   241,
     173,   242,   232,   358,   465,   359,   360,   103,   298,   345,
     104,   105,   106,   174,   107,   188,   189,   190,   191,   192,
     362,   313,   255,   256,   395,   109,   348,   396,   397,   111,
     112,   166,   179,   398,   399,   126,   400,    72,   145,   425,
     458,   459,   488,   277,   526,   415,   502,   217,   416,   586,
     646,   629,   587,   417,   588,   377,   556,   524,   503,   520,
     535,   547,   517,   504,   549,   521,   618,   527,   560,   509,
     513,   514,   286,   385,    73,    74
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -561
static const yytype_int16 yypact[] =
{
     267,  1281,  1281,    26,  -561,   267,  -561,  -561,  -561,  -561,
    -561,  -561,  -561,  -561,  -561,  -561,  -561,  -561,  -561,  -561,
    -561,  -561,  -561,  -561,  -561,  -561,  -561,  -561,  -561,  -561,
    -561,  -561,  -561,  -561,  -561,  -561,  -561,  -561,  -561,  -561,
    -561,  -561,  -561,  -561,  -561,  -561,  -561,  -561,  -561,  -561,
    -561,  -561,  -561,   -29,   -29,  -561,  -561,  -561,   425,  -561,
    -561,  -561,    87,  1281,    61,  1281,  1281,   121,   881,    69,
     541,   425,  -561,  -561,  -561,  1378,    62,   114,  -561,   117,
    -561,  -561,  -561,  -561,   -33,   925,   139,   139,     9,   114,
     115,   115,   115,   115,   143,   147,  1281,   180,   166,   425,
    -561,  -561,  -561,  -561,  -561,  -561,  -561,  -561,   403,  -561,
    -561,  -561,  -561,   167,  -561,  -561,  -561,  -561,  -561,  -561,
    -561,  -561,  -561,  -561,  -561,  -561,  -561,  -561,  -561,  1378,
    -561,   101,  -561,  -561,  -561,  -561,   253,    96,  -561,  -561,
     172,   174,   176,    -3,  -561,   114,   425,   117,   215,   -26,
     -33,   220,   684,  1392,   172,   174,   176,  -561,    40,   114,
    -561,   114,   114,   234,   114,   225,  -561,    24,  1281,  1281,
    1281,  1281,  1071,   224,   232,   244,  1281,  -561,  -561,  -561,
    1325,   246,   115,   115,   115,   115,   224,   147,  -561,  -561,
    -561,  -561,  -561,  -561,  -561,   294,  -561,  -561,  -561,   265,
    -561,  -561,  1358,  -561,  -561,  -561,  -561,  -561,  -561,  1281,
     256,   281,   -33,   279,   -33,   254,  -561,   268,   260,    16,
    -561,   274,  -561,    31,   -24,    85,   273,   158,   114,  -561,
    -561,   277,   282,   292,   297,   297,   297,   297,  -561,  1281,
     287,   298,   291,  1141,  1281,   329,  1281,  -561,  -561,   299,
     302,   307,  1281,    72,  1281,   306,   313,   167,  1281,  1281,
    1281,  1281,  1281,  1281,  -561,  -561,  -561,  -561,   308,  -561,
     314,  -561,   292,  -561,  -561,   319,   323,   317,   316,   -33,
    -561,   114,  1281,  -561,   321,  -561,   -33,   139,  1358,   139,
     139,  1358,   139,  -561,  -561,    24,  -561,   114,   134,   134,
     134,   134,   318,  -561,   329,  -561,   297,   297,  -561,   244,
     387,   324,   221,  -561,   325,  1325,  -561,  -561,   297,   297,
     297,   297,   297,   151,  1358,  -561,   327,   -33,   279,   -33,
     -33,  -561,    31,   330,  -561,   336,  -561,   340,   344,   343,
     114,   348,   360,  -561,   332,  -561,  -561,   320,  -561,  -561,
    -561,  -561,  -561,  -561,   134,   134,  -561,  -561,  1392,     5,
     369,  1392,  -561,  -561,  -561,  -561,  -561,  -561,   134,   134,
     134,   134,   134,  -561,   387,  -561,  1345,  -561,  -561,  -561,
    -561,  -561,  -561,   365,  -561,  -561,  -561,   367,  -561,    43,
     370,  -561,   114,  -561,   669,   366,   374,   379,   320,  -561,
    -561,  -561,  -561,  1281,  -561,  -561,  -561,  -561,  -561,  -561,
    -561,  -561,   377,  -561,  1281,   -33,   380,   372,  1392,   139,
     139,   139,  -561,  -561,   897,  1001,  -561,   167,  -561,  -561,
     375,   383,     8,   373,  1392,  -561,   381,   384,   385,   388,
    -561,  -561,  -561,  -561,  -561,  -561,  -561,  -561,  -561,  -561,
    -561,  -561,  -561,   410,  -561,   386,  -561,  -561,   390,   392,
     391,   327,  1281,  -561,   393,   400,  -561,  -561,   238,  -561,
    -561,  -561,  -561,  -561,  -561,  -561,  -561,  -561,   443,  -561,
     948,   489,   327,  -561,  -561,  -561,  -561,   117,  -561,  1281,
    -561,  -561,   399,   397,   399,   431,   411,   430,   399,   418,
    -561,   222,   -33,  -561,  -561,  -561,   475,   327,  -561,   -33,
     444,   -33,    71,   419,   524,   559,  -561,   423,   -33,   829,
     426,   322,   220,   417,   489,   420,  -561,   432,   421,   433,
    -561,   -33,   431,   305,  -561,   441,   467,   -33,   433,   399,
     435,   399,   445,   430,   399,   447,   -33,   448,   829,  -561,
     167,  -561,  -561,   469,  -561,   120,   423,   -33,   399,  -561,
     577,   336,  -561,  -561,   450,  -561,  -561,   220,   594,   -33,
     476,   -33,   559,   423,   -33,   829,   220,  -561,  -561,  -561,
    -561,  -561,  -561,  -561,  -561,  1281,   464,   463,   456,   -33,
     471,   -33,   222,  -561,   327,  -561,  -561,   222,   497,   478,
     473,   433,   483,   -33,   433,   487,  -561,   494,  1392,  1312,
    -561,   220,   -33,   500,   499,  -561,   501,   717,  -561,   -33,
     399,   724,  -561,   220,   735,  -561,  1281,  1281,   -33,   498,
    -561,  1281,   433,   -33,  -561,   497,   222,  -561,   506,   -33,
     222,  -561,  -561,   222,   497,  -561,    59,    47,   495,  1281,
     509,   771,   508,  -561,   507,   -33,   512,   511,   513,  -561,
    -561,  1281,  1211,   514,  1281,  1281,  -561,   131,  -561,   222,
    -561,   -33,  -561,   433,   -33,  -561,   497,   223,   503,   105,
    1281,  -561,   162,  -561,   515,   433,   782,   517,  -561,  -561,
    -561,  -561,  -561,  -561,  -561,   789,   222,  -561,   -33,   222,
    -561,   521,   433,   522,  -561,   836,  -561,   222,  -561,   531,
    -561
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -561,  -561,   585,  -561,  -249,    -1,   -61,   540,   555,   -56,
    -561,  -561,  -561,  -155,  -561,  -170,  -561,  -140,   -75,   -70,
     -69,   -68,  -171,   457,   482,  -561,   -81,  -561,  -561,  -243,
    -561,  -561,   -76,   428,   303,  -561,   -74,   333,  -561,  -561,
     439,   328,  -561,   195,  -561,  -561,  -322,  -561,   -34,   237,
    -561,  -561,  -561,  -135,  -561,  -561,  -561,  -561,  -561,  -561,
    -561,   331,  -561,   335,   565,  -561,    19,   245,   574,  -561,
    -561,   416,  -561,  -561,  -561,  -561,   251,  -561,   226,  -561,
     173,  -561,  -561,   326,   -82,    39,   -57,  -489,  -561,  -561,
    -432,  -561,  -561,  -305,    46,  -440,  -561,  -561,   132,  -501,
      86,  -509,   116,  -492,  -561,  -396,  -560,  -471,  -525,  -470,
    -561,   125,   150,   104,  -561,  -561
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -317
static const yytype_int16 yytable[] =
{
      53,    54,   151,    79,   159,   140,   141,   142,   317,   253,
      84,   162,   164,   568,   165,   127,   147,   169,   170,   171,
     551,   481,   357,   149,   511,   357,    55,   548,   518,   235,
     236,   237,   564,   552,   207,   566,   251,   404,   244,   578,
     529,   334,   506,   181,    57,   160,    58,   538,   150,   212,
     148,   262,   412,   220,   229,   213,   548,   287,   214,   140,
     141,   142,    76,   296,    80,    81,   605,   525,   215,   569,
     209,   571,   530,   230,   574,   653,   621,   220,   596,   624,
     208,  -162,   218,   548,   659,   505,   590,   606,   591,   382,
     210,   614,   467,   163,   468,   177,   616,   221,   223,   254,
     224,   225,   326,   227,   306,   148,   307,   651,   258,   259,
     260,   261,    77,   436,    78,   283,   688,   148,   534,   536,
     613,   221,   631,   222,   420,   665,   581,    75,   505,   471,
     275,   467,   278,   338,   642,   654,   341,   146,   660,   657,
     661,   632,   658,   662,   253,   113,   663,   664,   686,  -187,
     639,  -187,   376,   165,   615,   283,   128,   153,   312,   148,
     695,   196,   652,   288,   593,   197,   289,    78,   684,   375,
     148,   240,   598,    78,   202,   160,   536,   705,   463,   130,
     131,   132,   133,   134,   135,   136,   661,   582,   583,   662,
     691,   148,   663,   664,    82,   701,    83,   331,   703,   167,
     685,   299,   300,   301,   336,   584,   709,   346,   272,   347,
     683,   337,   661,   339,   340,   662,   342,   667,   663,   664,
     332,   637,   344,   491,   373,   641,   374,   172,   644,   677,
     679,   175,   148,   682,   254,   178,   291,   365,   302,   292,
     193,   693,   240,   661,   180,   378,   662,   380,   381,   663,
     664,   311,   204,   314,   205,   670,   206,   318,   319,   320,
     321,   322,   323,   492,   493,   494,   495,   496,   497,   498,
       1,     2,   354,   355,    78,   363,   364,   403,  -282,   389,
     406,   333,   247,   248,   368,   369,   370,   371,   372,   211,
     697,    78,   485,   486,   499,   414,   216,    83,  -282,   700,
     226,   579,   228,  -282,   661,   689,   491,   662,   243,   708,
     663,   664,   198,   199,   200,   201,   245,   344,   349,   350,
     351,   393,   257,   491,   264,   265,    85,    86,    87,    88,
      89,   207,   273,   433,   274,   276,   279,   414,    96,    97,
     281,   280,    98,   437,   438,   439,   492,   493,   494,   495,
     496,   497,   498,   414,   282,   140,   141,   142,   290,   295,
     394,  -282,   294,   492,   493,   494,   495,   496,   497,   498,
     202,   297,   303,   401,   402,   304,   305,   499,   238,   309,
      83,   563,   310,   315,   308,   324,  -282,   407,   408,   409,
     410,   411,   316,   325,   499,   327,  -204,    83,  -314,   329,
     328,   330,   430,   352,   357,   283,   394,   487,   376,   392,
     361,   312,   383,   432,   182,   183,   184,   185,   186,   187,
     522,   384,   386,   387,   461,   388,    61,   390,    -5,    -5,
      62,    -5,    -5,    -5,    -5,    -5,    -5,    -5,    -5,    -5,
      -5,    -5,   391,    -5,    -5,   405,   418,    -5,   419,   561,
     427,   421,   428,   431,   435,   567,   466,   434,   470,   464,
     537,   482,   546,   472,   576,   476,   473,   474,   491,   480,
     475,   478,    -9,   484,   585,   479,    63,    64,   483,   489,
     508,   510,    65,    66,   512,   516,   515,   599,   507,   601,
     491,   546,   604,   519,    67,   523,   532,   528,    83,   589,
      -5,   -62,   550,   553,   555,   557,   558,   611,   492,   493,
     494,   495,   496,   497,   498,   559,   603,   565,   546,   570,
     572,   623,   575,   580,   577,   491,   595,   627,   585,   600,
     492,   493,   494,   495,   496,   497,   498,   638,   608,   499,
     609,   610,    83,  -316,   612,   617,   648,   114,   115,   116,
     117,   619,   118,   119,   120,   121,   122,   656,   620,   622,
     491,   499,   500,   625,   501,   492,   493,   494,   495,   496,
     497,   498,   626,   673,   633,   634,   649,   635,   491,   655,
     666,   123,   668,   672,   607,   671,   674,   675,   690,   676,
      56,   694,   687,   680,   698,   491,   499,   704,   706,   533,
     492,   493,   494,   495,   496,   497,   498,   710,   100,    60,
     219,   195,   356,   246,   124,   263,   702,   125,   492,   493,
     494,   495,   496,   497,   498,   645,   647,   469,   343,   422,
     650,   499,   353,   108,    83,   492,   493,   494,   495,   496,
     497,   498,   110,   429,   293,   366,   426,   628,   645,   499,
     367,   462,   592,   490,   379,   630,   554,   562,   602,   573,
     645,   645,   531,   681,   645,   594,   499,     0,     0,   597,
     423,     0,  -247,  -247,  -247,     0,  -247,  -247,  -247,   692,
    -247,  -247,  -247,  -247,  -247,     0,     0,     0,  -247,  -247,
    -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,
    -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,
    -247,  -247,  -247,  -247,  -247,  -247,  -247,  -247,   491,  -247,
     128,  -247,  -247,     0,     0,   491,     0,     0,  -247,  -247,
    -247,  -247,  -247,  -247,  -247,  -247,   491,    78,  -247,  -247,
    -247,  -247,     0,   130,   131,   132,   133,   134,   135,   136,
       0,     0,     0,   424,     0,     0,     0,     0,   492,   493,
     494,   495,   496,   497,   498,   492,   493,   494,   495,   496,
     497,   498,   491,     0,     0,     0,   492,   493,   494,   495,
     496,   497,   498,   491,     0,     0,     0,     0,     0,   499,
     491,     0,   636,     0,     0,     0,   499,     0,     0,   640,
       0,     0,     0,     0,     0,     0,     0,   499,     0,     0,
     643,     0,   492,   493,   494,   495,   496,   497,   498,     0,
       0,     0,     0,   492,   493,   494,   495,   496,   497,   498,
     492,   493,   494,   495,   496,   497,   498,   491,     0,     0,
       0,     0,     0,   499,     0,     0,   669,     0,     0,     0,
       0,     0,     0,     0,   499,     0,     0,   696,     0,     0,
       0,   499,     0,     0,   699,     0,     0,     0,     0,     0,
     539,   540,   541,   495,   542,   543,   544,   492,   493,   494,
     495,   496,   497,   498,     1,     2,     0,    85,    86,    87,
      88,    89,    90,    91,    92,    93,    94,    95,   440,    96,
      97,   545,     0,    98,    83,     0,     0,     0,   499,     0,
       0,   707,     0,     0,     0,     0,     0,     0,   441,     0,
     442,   443,   444,   445,   446,   447,     0,     0,   448,   449,
     450,   451,   452,     0,     0,     0,     0,     0,     0,     0,
       0,   152,     0,     0,     0,     0,   453,   454,     0,   440,
       0,     0,     0,     0,     0,     0,    99,     0,     0,     0,
       0,   128,   153,   455,     0,     0,     0,   456,   457,   441,
       0,   442,   443,   444,   445,   446,   447,     0,    78,   448,
     449,   450,   451,   452,   130,   131,   132,   133,   134,   135,
     136,     0,     0,     0,     0,     0,     0,   453,   454,     0,
       0,     0,     0,     0,     6,     7,     8,     0,     9,    10,
      11,     0,    12,    13,    14,    15,    16,     0,   456,   457,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,   128,   129,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
       0,    45,     0,    46,   460,     0,     0,     0,     0,     0,
     130,   131,   132,   133,   134,   135,   136,    48,     0,     0,
      49,    50,    51,    52,     6,     7,     8,     0,     9,    10,
      11,     0,    12,    13,    14,    15,    16,     0,     0,     0,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,     0,     0,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
     238,    45,     0,    46,    47,   239,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    48,     0,     0,
      49,    50,    51,    52,     6,     7,     8,     0,     9,    10,
      11,     0,    12,    13,    14,    15,    16,     0,     0,     0,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,     0,     0,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
       0,    45,     0,    46,    47,   239,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    48,     0,     0,
      49,    50,    51,    52,     6,     7,     8,     0,     9,    10,
      11,     0,    12,    13,    14,    15,    16,     0,     0,     0,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,     0,     0,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
       0,    45,     0,    46,    47,   678,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    48,     0,     0,
      49,    50,    51,    52,     6,     7,     8,     0,     9,    10,
      11,     0,    12,    13,    14,    15,    16,     0,     0,     0,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,     0,   581,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
       0,    45,     0,    46,    47,     0,     0,     0,     0,     0,
       0,     0,     0,   252,     0,     0,     0,    48,   128,   153,
      49,    50,    51,    52,     0,     0,     0,     0,     0,     0,
       0,   128,   153,     0,     0,    78,     0,     0,     0,     0,
       0,   130,   131,   132,   133,   134,   135,   136,    78,   582,
     583,   128,   153,     0,   130,   131,   132,   133,   134,   135,
     136,     0,     0,     0,   128,   153,     0,     0,    78,     0,
       0,     0,     0,     0,   130,   131,   132,   133,   134,   135,
     136,    78,   266,   267,   128,   129,     0,   130,   131,   132,
     133,   134,   135,   136,     0,     0,     0,   413,   128,   153,
       0,    78,     0,     0,     0,     0,     0,   130,   131,   132,
     133,   134,   135,   136,     0,    78,     0,     0,     0,     0,
       0,   130,   131,   132,   133,   134,   135,   136
};

#define yypact_value_is_default(yystate) \
  ((yystate) == (-561))

#define yytable_value_is_error(yytable_value) \
  YYID (0)

static const yytype_int16 yycheck[] =
{
       1,     2,    84,    64,    85,    75,    75,    75,   257,   180,
      67,    87,    88,   538,    89,    71,    77,    91,    92,    93,
     521,   461,    17,    56,   494,    17,     0,   519,   498,   169,
     170,   171,   533,   522,    37,   536,   176,   359,   173,   548,
     511,   284,   482,    99,    73,    36,    75,   518,    81,    75,
      74,   186,   374,    37,    30,    81,   548,    81,    84,   129,
     129,   129,    63,   233,    65,    66,   575,   507,   150,   539,
     145,   541,   512,    49,   544,   635,   601,    37,   567,   604,
      83,    76,   152,   575,   644,   481,   557,   576,   558,   332,
     146,   592,    84,    84,    86,    96,   597,    81,   159,   180,
     161,   162,   272,   164,   244,    74,   246,   632,   182,   183,
     184,   185,    51,   418,    53,    84,   676,    74,   514,   515,
     591,    81,   611,    83,    81,    78,     6,    40,   524,   434,
     212,    84,   214,   288,   623,   636,   291,    75,    79,   640,
      81,   612,   643,    84,   315,    76,    87,    88,   673,    77,
     620,    79,    81,   228,   594,    84,    36,    37,    86,    74,
     685,    60,   633,    78,   560,    64,    81,    53,   669,   324,
      74,   172,   568,    53,    78,    36,   572,   702,   427,    59,
      60,    61,    62,    63,    64,    65,    81,    67,    68,    84,
      85,    74,    87,    88,    73,   696,    75,   279,   699,    84,
     671,   235,   236,   237,   286,    85,   707,    73,   209,    75,
      79,   287,    81,   289,   290,    84,   292,   649,    87,    88,
     281,   617,   297,     1,    73,   621,    75,    84,   624,   661,
     662,    84,    74,   665,   315,    55,    78,   312,   239,    81,
      73,    79,   243,    81,    78,   327,    84,   329,   330,    87,
      88,   252,    80,   254,    80,   651,    80,   258,   259,   260,
     261,   262,   263,    41,    42,    43,    44,    45,    46,    47,
       3,     4,   306,   307,    53,    54,    55,   358,    56,   340,
     361,   282,    38,    39,   318,   319,   320,   321,   322,    74,
     686,    53,    54,    55,    72,   376,    76,    75,    76,   695,
      66,   550,    77,    81,    81,    82,     1,    84,    84,   705,
      87,    88,    59,    60,    61,    62,    84,   392,   299,   300,
     301,     1,    76,     1,    59,    60,     6,     7,     8,     9,
      10,    37,    76,   415,    53,    56,    82,   418,    18,    19,
      80,    73,    22,   419,   420,   421,    41,    42,    43,    44,
      45,    46,    47,   434,    80,   425,   425,   425,    85,    77,
      40,    56,    85,    41,    42,    43,    44,    45,    46,    47,
      78,    74,    85,   354,   355,    77,    85,    72,    49,    77,
      75,    76,    75,    77,    85,    77,    81,   368,   369,   370,
     371,   372,    79,    79,    72,    76,    76,    75,    76,    82,
      77,    85,   403,    85,    17,    84,    40,   468,    81,    77,
      86,    86,    82,   414,    11,    12,    13,    14,    15,    16,
     502,    85,    82,    79,   425,    82,     1,    79,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    82,    18,    19,    76,    81,    22,    81,   531,
      76,    81,    73,    76,    82,   537,    73,    77,    85,    84,
     517,   462,   519,    82,   546,    55,    82,    82,     1,    77,
      82,    85,    81,    73,   555,    85,    51,    52,    85,    36,
      81,    84,    57,    58,    53,    55,    75,   569,   489,   571,
       1,   548,   574,    75,    69,    20,    77,    53,    75,   556,
      75,    76,    76,    86,    84,    73,    85,   589,    41,    42,
      43,    44,    45,    46,    47,    82,   573,    76,   575,    84,
      75,   603,    75,    54,    76,     1,    76,   608,   609,    53,
      41,    42,    43,    44,    45,    46,    47,   619,    74,    72,
      77,    85,    75,    76,    73,    48,   628,     6,     7,     8,
       9,    73,    11,    12,    13,    14,    15,   639,    85,    76,
       1,    72,    73,    76,    75,    41,    42,    43,    44,    45,
      46,    47,    78,   655,    74,    76,    78,    76,     1,    73,
      85,    40,    73,    76,   585,    77,    74,    76,    85,    76,
       5,    76,   674,    79,    77,     1,    72,    76,    76,    75,
      41,    42,    43,    44,    45,    46,    47,    76,    68,    54,
     153,   129,   309,   174,    73,   187,   698,    76,    41,    42,
      43,    44,    45,    46,    47,   626,   627,   432,   295,   392,
     631,    72,   304,    68,    75,    41,    42,    43,    44,    45,
      46,    47,    68,   398,   228,   314,   395,   608,   649,    72,
     315,   425,    75,   480,   328,   609,   524,   532,   572,   543,
     661,   662,   512,   664,   665,   561,    72,    -1,    -1,    75,
       1,    -1,     3,     4,     5,    -1,     7,     8,     9,   680,
      11,    12,    13,    14,    15,    -1,    -1,    -1,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    34,    35,    36,    37,    38,    39,    40,
      41,    42,    43,    44,    45,    46,    47,    48,     1,    50,
      36,    52,    53,    -1,    -1,     1,    -1,    -1,    59,    60,
      61,    62,    63,    64,    65,    66,     1,    53,    69,    70,
      71,    72,    -1,    59,    60,    61,    62,    63,    64,    65,
      -1,    -1,    -1,    84,    -1,    -1,    -1,    -1,    41,    42,
      43,    44,    45,    46,    47,    41,    42,    43,    44,    45,
      46,    47,     1,    -1,    -1,    -1,    41,    42,    43,    44,
      45,    46,    47,     1,    -1,    -1,    -1,    -1,    -1,    72,
       1,    -1,    75,    -1,    -1,    -1,    72,    -1,    -1,    75,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    72,    -1,    -1,
      75,    -1,    41,    42,    43,    44,    45,    46,    47,    -1,
      -1,    -1,    -1,    41,    42,    43,    44,    45,    46,    47,
      41,    42,    43,    44,    45,    46,    47,     1,    -1,    -1,
      -1,    -1,    -1,    72,    -1,    -1,    75,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    72,    -1,    -1,    75,    -1,    -1,
      -1,    72,    -1,    -1,    75,    -1,    -1,    -1,    -1,    -1,
      41,    42,    43,    44,    45,    46,    47,    41,    42,    43,
      44,    45,    46,    47,     3,     4,    -1,     6,     7,     8,
       9,    10,    11,    12,    13,    14,    15,    16,     1,    18,
      19,    72,    -1,    22,    75,    -1,    -1,    -1,    72,    -1,
      -1,    75,    -1,    -1,    -1,    -1,    -1,    -1,    21,    -1,
      23,    24,    25,    26,    27,    28,    -1,    -1,    31,    32,
      33,    34,    35,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    16,    -1,    -1,    -1,    -1,    49,    50,    -1,     1,
      -1,    -1,    -1,    -1,    -1,    -1,    75,    -1,    -1,    -1,
      -1,    36,    37,    66,    -1,    -1,    -1,    70,    71,    21,
      -1,    23,    24,    25,    26,    27,    28,    -1,    53,    31,
      32,    33,    34,    35,    59,    60,    61,    62,    63,    64,
      65,    -1,    -1,    -1,    -1,    -1,    -1,    49,    50,    -1,
      -1,    -1,    -1,    -1,     3,     4,     5,    -1,     7,     8,
       9,    -1,    11,    12,    13,    14,    15,    -1,    70,    71,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    34,    35,    36,    37,    38,
      39,    40,    41,    42,    43,    44,    45,    46,    47,    48,
      -1,    50,    -1,    52,    53,    -1,    -1,    -1,    -1,    -1,
      59,    60,    61,    62,    63,    64,    65,    66,    -1,    -1,
      69,    70,    71,    72,     3,     4,     5,    -1,     7,     8,
       9,    -1,    11,    12,    13,    14,    15,    -1,    -1,    -1,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    34,    35,    -1,    -1,    38,
      39,    40,    41,    42,    43,    44,    45,    46,    47,    48,
      49,    50,    -1,    52,    53,    54,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    66,    -1,    -1,
      69,    70,    71,    72,     3,     4,     5,    -1,     7,     8,
       9,    -1,    11,    12,    13,    14,    15,    -1,    -1,    -1,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    34,    35,    -1,    -1,    38,
      39,    40,    41,    42,    43,    44,    45,    46,    47,    48,
      -1,    50,    -1,    52,    53,    54,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    66,    -1,    -1,
      69,    70,    71,    72,     3,     4,     5,    -1,     7,     8,
       9,    -1,    11,    12,    13,    14,    15,    -1,    -1,    -1,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    34,    35,    -1,    -1,    38,
      39,    40,    41,    42,    43,    44,    45,    46,    47,    48,
      -1,    50,    -1,    52,    53,    54,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    66,    -1,    -1,
      69,    70,    71,    72,     3,     4,     5,    -1,     7,     8,
       9,    -1,    11,    12,    13,    14,    15,    -1,    -1,    -1,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    34,    35,    -1,     6,    38,
      39,    40,    41,    42,    43,    44,    45,    46,    47,    48,
      -1,    50,    -1,    52,    53,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    18,    -1,    -1,    -1,    66,    36,    37,
      69,    70,    71,    72,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    36,    37,    -1,    -1,    53,    -1,    -1,    -1,    -1,
      -1,    59,    60,    61,    62,    63,    64,    65,    53,    67,
      68,    36,    37,    -1,    59,    60,    61,    62,    63,    64,
      65,    -1,    -1,    -1,    36,    37,    -1,    -1,    53,    -1,
      -1,    -1,    -1,    -1,    59,    60,    61,    62,    63,    64,
      65,    53,    54,    55,    36,    37,    -1,    59,    60,    61,
      62,    63,    64,    65,    -1,    -1,    -1,    82,    36,    37,
      -1,    53,    -1,    -1,    -1,    -1,    -1,    59,    60,    61,
      62,    63,    64,    65,    -1,    53,    -1,    -1,    -1,    -1,
      -1,    59,    60,    61,    62,    63,    64,    65
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     3,     4,    90,    91,    96,     3,     4,     5,     7,
       8,     9,    11,    12,    13,    14,    15,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    34,    35,    38,    39,    40,    41,    42,    43,
      44,    45,    46,    47,    48,    50,    52,    53,    66,    69,
      70,    71,    72,    94,    94,     0,    91,    73,    75,    97,
      97,     1,     5,    51,    52,    57,    58,    69,    92,    98,
      99,   100,   166,   203,   204,    40,    94,    51,    53,    95,
      94,    94,    73,    75,   175,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    18,    19,    22,    75,
      96,   119,   120,   136,   139,   140,   141,   143,   153,   154,
     157,   158,   159,    76,     6,     7,     8,     9,    11,    12,
      13,    14,    15,    40,    73,    76,   164,    98,    36,    37,
      59,    60,    61,    62,    63,    64,    65,    95,   105,   107,
     108,   109,   110,   113,   114,   167,    75,    95,    74,    56,
      81,   173,    16,    37,   108,   109,   110,   111,   112,   115,
      36,   121,   121,    84,   121,   107,   160,    84,   125,   125,
     125,   125,    84,   129,   142,    84,   122,    94,    55,   161,
      78,    98,    11,    12,    13,    14,    15,    16,   144,   145,
     146,   147,   148,    73,    93,   113,    60,    64,    59,    60,
      61,    62,    78,   104,    80,    80,    80,    37,    83,   107,
      98,    74,    75,    81,    84,   173,    76,   176,   108,   112,
      37,    81,    83,    95,    95,    95,    66,    95,    77,    30,
      49,   126,   131,    94,   106,   106,   106,   106,    49,    54,
      94,   128,   130,    84,   142,    84,   129,    38,    39,   123,
     124,   106,    18,   111,   115,   151,   152,    76,   125,   125,
     125,   125,   142,   122,    59,    60,    54,    55,   101,   102,
     103,   115,    94,    76,    53,   173,    56,   172,   173,    82,
      73,    80,    80,    84,   117,   118,   201,    81,    78,    81,
      85,    78,    81,   160,    85,    77,   104,    74,   137,   137,
     137,   137,    94,    85,    77,    85,   106,   106,    85,    77,
      75,    94,    86,   150,    94,    77,    79,    93,    94,    94,
      94,    94,    94,    94,    77,    79,   104,    76,    77,    82,
      85,   173,    95,    94,   118,   116,   173,   121,   102,   121,
     121,   102,   121,   126,   107,   138,    73,    75,   155,   155,
     155,   155,    85,   130,   137,   137,   123,    17,   132,   134,
     135,    86,   149,    54,    55,   107,   150,   152,   137,   137,
     137,   137,   137,    73,    75,   102,    81,   184,   173,   172,
     173,   173,   118,    82,    85,   202,    82,    79,    82,    95,
      79,    82,    77,     1,    40,   153,   156,   157,   162,   163,
     165,   155,   155,   115,   135,    76,   115,   155,   155,   155,
     155,   155,   135,    82,   115,   174,   177,   182,    81,    81,
      81,    81,   138,     1,    84,   168,   165,    76,    73,   156,
      94,    76,    94,   173,    77,    82,   182,   121,   121,   121,
       1,    21,    23,    24,    25,    26,    27,    28,    31,    32,
      33,    34,    35,    49,    50,    66,    70,    71,   169,   170,
      53,    94,   167,    93,    84,   133,    73,    84,    86,   132,
      85,   182,    82,    82,    82,    82,    55,   127,    85,    85,
      77,   184,    94,    85,    73,    54,    55,    95,   171,    36,
     169,     1,    41,    42,    43,    44,    45,    46,    47,    72,
      73,    75,   175,   187,   192,   194,   184,    94,    81,   198,
      84,   198,    53,   199,   200,    75,    55,   191,   198,    75,
     188,   194,   173,    20,   186,   184,   173,   196,    53,   196,
     184,   201,    77,    75,   194,   189,   194,   175,   196,    41,
      42,    43,    45,    46,    47,    72,   175,   190,   192,   193,
      76,   188,   176,    86,   187,    84,   185,    73,    85,    82,
     197,   173,   200,    76,   188,    76,   188,   173,   197,   198,
      84,   198,    75,   191,   198,    75,   173,    76,   190,    93,
      54,     6,    67,    68,    85,   115,   178,   181,   183,   175,
     196,   198,    75,   194,   202,    76,   176,    75,   194,   173,
      53,   173,   189,   175,   173,   190,   176,    94,    74,    77,
      85,   173,    73,   196,   188,   184,   188,    48,   195,    73,
      85,   197,    76,   173,   197,    76,    78,   115,   174,   180,
     183,   176,   196,    74,    76,    76,    75,   194,   173,   198,
      75,   194,   176,    75,   194,    94,   179,    94,   173,    78,
      94,   197,   196,   195,   188,    73,   173,   188,   188,   195,
      79,    81,    84,    87,    88,    78,    85,   179,    73,    75,
     194,    77,    76,   173,    74,    76,    76,   179,    54,   179,
      79,    94,   179,    79,   188,   196,   197,   173,   195,    82,
      85,    85,    94,    79,    76,   197,    75,   194,    77,    75,
     194,   188,   173,   188,    76,   197,    76,    75,   194,   188,
      76
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
   Once GCC version 2 has supplanted version 1, this can go.  However,
   YYFAIL appears to be in use.  Nevertheless, it is formally deprecated
   in Bison 2.4.2's NEWS entry, where a plan to phase it out is
   discussed.  */

#define YYFAIL		goto yyerrlab
#if defined YYFAIL
  /* This is here to suppress warnings from the GCC cpp's
     -Wunused-macros.  Normally we don't worry about that warning, but
     some users do, and we want to make it easy for users to remove
     YYFAIL uses, which will produce warnings from Bison 2.5.  */
#endif

#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)					\
do								\
  if (yychar == YYEMPTY && yylen == 1)				\
    {								\
      yychar = (Token);						\
      yylval = (Value);						\
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
yy_stack_print (yytype_int16 *yybottom, yytype_int16 *yytop)
#else
static void
yy_stack_print (yybottom, yytop)
    yytype_int16 *yybottom;
    yytype_int16 *yytop;
#endif
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
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
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr, yyrhs[yyprhs[yyrule] + yyi],
		       &(yyvsp[(yyi + 1) - (yynrhs)])
		       , &(yylsp[(yyi + 1) - (yynrhs)])		       );
      YYFPRINTF (stderr, "\n");
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

/* Copy into *YYMSG, which is of size *YYMSG_ALLOC, an error message
   about the unexpected token YYTOKEN for the state stack whose top is
   YYSSP.

   Return 0 if *YYMSG was successfully written.  Return 1 if *YYMSG is
   not large enough to hold the message.  In that case, also set
   *YYMSG_ALLOC to the required number of bytes.  Return 2 if the
   required number of bytes is too large to store.  */
static int
yysyntax_error (YYSIZE_T *yymsg_alloc, char **yymsg,
                yytype_int16 *yyssp, int yytoken)
{
  YYSIZE_T yysize0 = yytnamerr (0, yytname[yytoken]);
  YYSIZE_T yysize = yysize0;
  YYSIZE_T yysize1;
  enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
  /* Internationalized format string. */
  const char *yyformat = 0;
  /* Arguments of yyformat. */
  char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
  /* Number of reported tokens (one for the "unexpected", one per
     "expected"). */
  int yycount = 0;

  /* There are many possibilities here to consider:
     - Assume YYFAIL is not used.  It's too flawed to consider.  See
       <http://lists.gnu.org/archive/html/bison-patches/2009-12/msg00024.html>
       for details.  YYERROR is fine as it does not invoke this
       function.
     - If this state is a consistent state with a default action, then
       the only way this function was invoked is if the default action
       is an error action.  In that case, don't check for expected
       tokens because there are none.
     - The only way there can be no lookahead present (in yychar) is if
       this state is a consistent state with a default action.  Thus,
       detecting the absence of a lookahead is sufficient to determine
       that there is no unexpected or expected token to report.  In that
       case, just report a simple "syntax error".
     - Don't assume there isn't a lookahead just because this state is a
       consistent state with a default action.  There might have been a
       previous inconsistent state, consistent state with a non-default
       action, or user semantic action that manipulated yychar.
     - Of course, the expected token list depends on states to have
       correct lookahead information, and it depends on the parser not
       to perform extra reductions after fetching a lookahead from the
       scanner and before detecting a syntax error.  Thus, state merging
       (from LALR or IELR) and default reductions corrupt the expected
       token list.  However, the list is correct for canonical LR with
       one exception: it will still contain any token that will not be
       accepted due to an error action in a later state.
  */
  if (yytoken != YYEMPTY)
    {
      int yyn = yypact[*yyssp];
      yyarg[yycount++] = yytname[yytoken];
      if (!yypact_value_is_default (yyn))
        {
          /* Start YYX at -YYN if negative to avoid negative indexes in
             YYCHECK.  In other words, skip the first -YYN actions for
             this state because they are default actions.  */
          int yyxbegin = yyn < 0 ? -yyn : 0;
          /* Stay within bounds of both yycheck and yytname.  */
          int yychecklim = YYLAST - yyn + 1;
          int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
          int yyx;

          for (yyx = yyxbegin; yyx < yyxend; ++yyx)
            if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR
                && !yytable_value_is_error (yytable[yyx + yyn]))
              {
                if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
                  {
                    yycount = 1;
                    yysize = yysize0;
                    break;
                  }
                yyarg[yycount++] = yytname[yyx];
                yysize1 = yysize + yytnamerr (0, yytname[yyx]);
                if (! (yysize <= yysize1
                       && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
                  return 2;
                yysize = yysize1;
              }
        }
    }

  switch (yycount)
    {
# define YYCASE_(N, S)                      \
      case N:                               \
        yyformat = S;                       \
      break
      YYCASE_(0, YY_("syntax error"));
      YYCASE_(1, YY_("syntax error, unexpected %s"));
      YYCASE_(2, YY_("syntax error, unexpected %s, expecting %s"));
      YYCASE_(3, YY_("syntax error, unexpected %s, expecting %s or %s"));
      YYCASE_(4, YY_("syntax error, unexpected %s, expecting %s or %s or %s"));
      YYCASE_(5, YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s"));
# undef YYCASE_
    }

  yysize1 = yysize + yystrlen (yyformat);
  if (! (yysize <= yysize1 && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
    return 2;
  yysize = yysize1;

  if (*yymsg_alloc < yysize)
    {
      *yymsg_alloc = 2 * yysize;
      if (! (yysize <= *yymsg_alloc
             && *yymsg_alloc <= YYSTACK_ALLOC_MAXIMUM))
        *yymsg_alloc = YYSTACK_ALLOC_MAXIMUM;
      return 1;
    }

  /* Avoid sprintf, as that infringes on the user's name space.
     Don't have undefined behavior even if the translation
     produced a string with the wrong number of "%s"s.  */
  {
    char *yyp = *yymsg;
    int yyi = 0;
    while ((*yyp = *yyformat) != '\0')
      if (*yyp == '%' && yyformat[1] == 's' && yyi < yycount)
        {
          yyp += yytnamerr (yyp, yyarg[yyi++]);
          yyformat += 2;
        }
      else
        {
          yyp++;
          yyformat++;
        }
  }
  return 0;
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


/* The lookahead symbol.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;

/* Location data for the lookahead symbol.  */
YYLTYPE yylloc;

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
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus;

    /* The stacks and their tools:
       `yyss': related to states.
       `yyvs': related to semantic values.
       `yyls': related to locations.

       Refer to the stacks thru separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* The state stack.  */
    yytype_int16 yyssa[YYINITDEPTH];
    yytype_int16 *yyss;
    yytype_int16 *yyssp;

    /* The semantic value stack.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs;
    YYSTYPE *yyvsp;

    /* The location stack.  */
    YYLTYPE yylsa[YYINITDEPTH];
    YYLTYPE *yyls;
    YYLTYPE *yylsp;

    /* The locations where the error started and ended.  */
    YYLTYPE yyerror_range[3];

    YYSIZE_T yystacksize;

  int yyn;
  int yyresult;
  /* Lookahead token as an internal (translated) token number.  */
  int yytoken;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;
  YYLTYPE yyloc;

#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N), yylsp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  yytoken = 0;
  yyss = yyssa;
  yyvs = yyvsa;
  yyls = yylsa;
  yystacksize = YYINITDEPTH;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY; /* Cause a token to be read.  */

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
  yylloc.first_column = yylloc.last_column = 1;
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
	YYSTACK_RELOCATE (yyss_alloc, yyss);
	YYSTACK_RELOCATE (yyvs_alloc, yyvs);
	YYSTACK_RELOCATE (yyls_alloc, yyls);
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

  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yypact_value_is_default (yyn))
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
      if (yytable_value_is_error (yyn))
        goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token.  */
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

/* Line 1806 of yacc.c  */
#line 193 "xi-grammar.y"
    { (yyval.modlist) = (yyvsp[(1) - (1)].modlist); modlist = (yyvsp[(1) - (1)].modlist); }
    break;

  case 3:

/* Line 1806 of yacc.c  */
#line 197 "xi-grammar.y"
    { 
		  (yyval.modlist) = 0; 
		}
    break;

  case 4:

/* Line 1806 of yacc.c  */
#line 201 "xi-grammar.y"
    { (yyval.modlist) = new AstChildren<Module>(lineno, (yyvsp[(1) - (2)].module), (yyvsp[(2) - (2)].modlist)); }
    break;

  case 5:

/* Line 1806 of yacc.c  */
#line 205 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 6:

/* Line 1806 of yacc.c  */
#line 207 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 7:

/* Line 1806 of yacc.c  */
#line 211 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 8:

/* Line 1806 of yacc.c  */
#line 213 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 9:

/* Line 1806 of yacc.c  */
#line 218 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 10:

/* Line 1806 of yacc.c  */
#line 219 "xi-grammar.y"
    { ReservedWord(MODULE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 11:

/* Line 1806 of yacc.c  */
#line 220 "xi-grammar.y"
    { ReservedWord(MAINMODULE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 12:

/* Line 1806 of yacc.c  */
#line 221 "xi-grammar.y"
    { ReservedWord(EXTERN, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 13:

/* Line 1806 of yacc.c  */
#line 223 "xi-grammar.y"
    { ReservedWord(INITCALL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 14:

/* Line 1806 of yacc.c  */
#line 224 "xi-grammar.y"
    { ReservedWord(INITNODE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 15:

/* Line 1806 of yacc.c  */
#line 225 "xi-grammar.y"
    { ReservedWord(INITPROC, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 16:

/* Line 1806 of yacc.c  */
#line 227 "xi-grammar.y"
    { ReservedWord(CHARE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 17:

/* Line 1806 of yacc.c  */
#line 228 "xi-grammar.y"
    { ReservedWord(MAINCHARE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 18:

/* Line 1806 of yacc.c  */
#line 229 "xi-grammar.y"
    { ReservedWord(GROUP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 19:

/* Line 1806 of yacc.c  */
#line 230 "xi-grammar.y"
    { ReservedWord(NODEGROUP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 20:

/* Line 1806 of yacc.c  */
#line 231 "xi-grammar.y"
    { ReservedWord(ARRAY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 21:

/* Line 1806 of yacc.c  */
#line 235 "xi-grammar.y"
    { ReservedWord(INCLUDE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 22:

/* Line 1806 of yacc.c  */
#line 236 "xi-grammar.y"
    { ReservedWord(STACKSIZE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 23:

/* Line 1806 of yacc.c  */
#line 237 "xi-grammar.y"
    { ReservedWord(THREADED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 24:

/* Line 1806 of yacc.c  */
#line 238 "xi-grammar.y"
    { ReservedWord(TEMPLATE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 25:

/* Line 1806 of yacc.c  */
#line 239 "xi-grammar.y"
    { ReservedWord(SYNC, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 26:

/* Line 1806 of yacc.c  */
#line 240 "xi-grammar.y"
    { ReservedWord(IGET, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 27:

/* Line 1806 of yacc.c  */
#line 241 "xi-grammar.y"
    { ReservedWord(EXCLUSIVE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 28:

/* Line 1806 of yacc.c  */
#line 242 "xi-grammar.y"
    { ReservedWord(IMMEDIATE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 29:

/* Line 1806 of yacc.c  */
#line 243 "xi-grammar.y"
    { ReservedWord(SKIPSCHED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 30:

/* Line 1806 of yacc.c  */
#line 244 "xi-grammar.y"
    { ReservedWord(INLINE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 31:

/* Line 1806 of yacc.c  */
#line 245 "xi-grammar.y"
    { ReservedWord(VIRTUAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 32:

/* Line 1806 of yacc.c  */
#line 246 "xi-grammar.y"
    { ReservedWord(MIGRATABLE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 33:

/* Line 1806 of yacc.c  */
#line 247 "xi-grammar.y"
    { ReservedWord(CREATEHERE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 34:

/* Line 1806 of yacc.c  */
#line 248 "xi-grammar.y"
    { ReservedWord(CREATEHOME, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 35:

/* Line 1806 of yacc.c  */
#line 249 "xi-grammar.y"
    { ReservedWord(NOKEEP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 36:

/* Line 1806 of yacc.c  */
#line 250 "xi-grammar.y"
    { ReservedWord(NOTRACE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 37:

/* Line 1806 of yacc.c  */
#line 251 "xi-grammar.y"
    { ReservedWord(APPWORK, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 38:

/* Line 1806 of yacc.c  */
#line 254 "xi-grammar.y"
    { ReservedWord(PACKED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 39:

/* Line 1806 of yacc.c  */
#line 255 "xi-grammar.y"
    { ReservedWord(VARSIZE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 40:

/* Line 1806 of yacc.c  */
#line 256 "xi-grammar.y"
    { ReservedWord(ENTRY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 41:

/* Line 1806 of yacc.c  */
#line 257 "xi-grammar.y"
    { ReservedWord(FOR, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 42:

/* Line 1806 of yacc.c  */
#line 258 "xi-grammar.y"
    { ReservedWord(FORALL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 43:

/* Line 1806 of yacc.c  */
#line 259 "xi-grammar.y"
    { ReservedWord(WHILE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 44:

/* Line 1806 of yacc.c  */
#line 260 "xi-grammar.y"
    { ReservedWord(WHEN, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 45:

/* Line 1806 of yacc.c  */
#line 261 "xi-grammar.y"
    { ReservedWord(OVERLAP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 46:

/* Line 1806 of yacc.c  */
#line 262 "xi-grammar.y"
    { ReservedWord(ATOMIC, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 47:

/* Line 1806 of yacc.c  */
#line 263 "xi-grammar.y"
    { ReservedWord(IF, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 48:

/* Line 1806 of yacc.c  */
#line 264 "xi-grammar.y"
    { ReservedWord(ELSE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 49:

/* Line 1806 of yacc.c  */
#line 266 "xi-grammar.y"
    { ReservedWord(LOCAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 50:

/* Line 1806 of yacc.c  */
#line 268 "xi-grammar.y"
    { ReservedWord(USING, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 51:

/* Line 1806 of yacc.c  */
#line 269 "xi-grammar.y"
    { ReservedWord(ACCEL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 52:

/* Line 1806 of yacc.c  */
#line 272 "xi-grammar.y"
    { ReservedWord(ACCELBLOCK, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 53:

/* Line 1806 of yacc.c  */
#line 273 "xi-grammar.y"
    { ReservedWord(MEMCRITICAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 54:

/* Line 1806 of yacc.c  */
#line 274 "xi-grammar.y"
    { ReservedWord(REDUCTIONTARGET, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 55:

/* Line 1806 of yacc.c  */
#line 275 "xi-grammar.y"
    { ReservedWord(CASE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 56:

/* Line 1806 of yacc.c  */
#line 279 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 57:

/* Line 1806 of yacc.c  */
#line 281 "xi-grammar.y"
    {
		  char *tmp = new char[strlen((yyvsp[(1) - (4)].strval))+strlen((yyvsp[(4) - (4)].strval))+3];
		  sprintf(tmp,"%s::%s", (yyvsp[(1) - (4)].strval), (yyvsp[(4) - (4)].strval));
		  (yyval.strval) = tmp;
		}
    break;

  case 58:

/* Line 1806 of yacc.c  */
#line 288 "xi-grammar.y"
    { 
		    (yyval.module) = new Module(lineno, (yyvsp[(2) - (3)].strval), (yyvsp[(3) - (3)].conslist)); 
		}
    break;

  case 59:

/* Line 1806 of yacc.c  */
#line 292 "xi-grammar.y"
    {  
		    (yyval.module) = new Module(lineno, (yyvsp[(2) - (3)].strval), (yyvsp[(3) - (3)].conslist)); 
		    (yyval.module)->setMain();
		}
    break;

  case 60:

/* Line 1806 of yacc.c  */
#line 299 "xi-grammar.y"
    { (yyval.conslist) = 0; }
    break;

  case 61:

/* Line 1806 of yacc.c  */
#line 301 "xi-grammar.y"
    { (yyval.conslist) = (yyvsp[(2) - (4)].conslist); }
    break;

  case 62:

/* Line 1806 of yacc.c  */
#line 305 "xi-grammar.y"
    { (yyval.conslist) = 0; }
    break;

  case 63:

/* Line 1806 of yacc.c  */
#line 307 "xi-grammar.y"
    { (yyval.conslist) = new ConstructList(lineno, (yyvsp[(1) - (2)].construct), (yyvsp[(2) - (2)].conslist)); }
    break;

  case 64:

/* Line 1806 of yacc.c  */
#line 311 "xi-grammar.y"
    { (yyval.construct) = new UsingScope((yyvsp[(3) - (3)].strval), false); }
    break;

  case 65:

/* Line 1806 of yacc.c  */
#line 313 "xi-grammar.y"
    { (yyval.construct) = new UsingScope((yyvsp[(2) - (2)].strval), true); }
    break;

  case 66:

/* Line 1806 of yacc.c  */
#line 315 "xi-grammar.y"
    { (yyvsp[(2) - (2)].member)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].member); }
    break;

  case 67:

/* Line 1806 of yacc.c  */
#line 317 "xi-grammar.y"
    { (yyvsp[(2) - (2)].message)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].message); }
    break;

  case 68:

/* Line 1806 of yacc.c  */
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

/* Line 1806 of yacc.c  */
#line 331 "xi-grammar.y"
    { if((yyvsp[(3) - (5)].conslist)) (yyvsp[(3) - (5)].conslist)->recurse<int&>((yyvsp[(1) - (5)].intval), &Construct::setExtern); (yyval.construct) = (yyvsp[(3) - (5)].conslist); }
    break;

  case 70:

/* Line 1806 of yacc.c  */
#line 333 "xi-grammar.y"
    { (yyval.construct) = new Scope((yyvsp[(2) - (5)].strval), (yyvsp[(4) - (5)].conslist)); }
    break;

  case 71:

/* Line 1806 of yacc.c  */
#line 335 "xi-grammar.y"
    { (yyval.construct) = (yyvsp[(1) - (2)].construct); }
    break;

  case 72:

/* Line 1806 of yacc.c  */
#line 337 "xi-grammar.y"
    {
          ERROR("preceding construct must be semicolon terminated",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
    break;

  case 73:

/* Line 1806 of yacc.c  */
#line 343 "xi-grammar.y"
    { (yyvsp[(2) - (2)].module)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].module); }
    break;

  case 74:

/* Line 1806 of yacc.c  */
#line 345 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 75:

/* Line 1806 of yacc.c  */
#line 347 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 76:

/* Line 1806 of yacc.c  */
#line 349 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 77:

/* Line 1806 of yacc.c  */
#line 351 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 78:

/* Line 1806 of yacc.c  */
#line 353 "xi-grammar.y"
    { (yyvsp[(2) - (2)].templat)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].templat); }
    break;

  case 79:

/* Line 1806 of yacc.c  */
#line 355 "xi-grammar.y"
    { (yyval.construct) = NULL; }
    break;

  case 80:

/* Line 1806 of yacc.c  */
#line 357 "xi-grammar.y"
    { (yyval.construct) = NULL; }
    break;

  case 81:

/* Line 1806 of yacc.c  */
#line 359 "xi-grammar.y"
    { (yyval.construct) = (yyvsp[(1) - (1)].accelBlock); }
    break;

  case 82:

/* Line 1806 of yacc.c  */
#line 361 "xi-grammar.y"
    {
          ERROR("invalid construct",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
    break;

  case 83:

/* Line 1806 of yacc.c  */
#line 369 "xi-grammar.y"
    { (yyval.tparam) = new TParamType((yyvsp[(1) - (1)].type)); }
    break;

  case 84:

/* Line 1806 of yacc.c  */
#line 371 "xi-grammar.y"
    { (yyval.tparam) = new TParamVal((yyvsp[(1) - (1)].strval)); }
    break;

  case 85:

/* Line 1806 of yacc.c  */
#line 373 "xi-grammar.y"
    { (yyval.tparam) = new TParamVal((yyvsp[(1) - (1)].strval)); }
    break;

  case 86:

/* Line 1806 of yacc.c  */
#line 377 "xi-grammar.y"
    { (yyval.tparlist) = new TParamList((yyvsp[(1) - (1)].tparam)); }
    break;

  case 87:

/* Line 1806 of yacc.c  */
#line 379 "xi-grammar.y"
    { (yyval.tparlist) = new TParamList((yyvsp[(1) - (3)].tparam), (yyvsp[(3) - (3)].tparlist)); }
    break;

  case 88:

/* Line 1806 of yacc.c  */
#line 383 "xi-grammar.y"
    { (yyval.tparlist) = 0; }
    break;

  case 89:

/* Line 1806 of yacc.c  */
#line 385 "xi-grammar.y"
    { (yyval.tparlist) = (yyvsp[(1) - (1)].tparlist); }
    break;

  case 90:

/* Line 1806 of yacc.c  */
#line 389 "xi-grammar.y"
    { (yyval.tparlist) = 0; }
    break;

  case 91:

/* Line 1806 of yacc.c  */
#line 391 "xi-grammar.y"
    { (yyval.tparlist) = (yyvsp[(2) - (3)].tparlist); }
    break;

  case 92:

/* Line 1806 of yacc.c  */
#line 395 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("int"); }
    break;

  case 93:

/* Line 1806 of yacc.c  */
#line 397 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long"); }
    break;

  case 94:

/* Line 1806 of yacc.c  */
#line 399 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("short"); }
    break;

  case 95:

/* Line 1806 of yacc.c  */
#line 401 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("char"); }
    break;

  case 96:

/* Line 1806 of yacc.c  */
#line 403 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned int"); }
    break;

  case 97:

/* Line 1806 of yacc.c  */
#line 405 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long"); }
    break;

  case 98:

/* Line 1806 of yacc.c  */
#line 407 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long"); }
    break;

  case 99:

/* Line 1806 of yacc.c  */
#line 409 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long long"); }
    break;

  case 100:

/* Line 1806 of yacc.c  */
#line 411 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned short"); }
    break;

  case 101:

/* Line 1806 of yacc.c  */
#line 413 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned char"); }
    break;

  case 102:

/* Line 1806 of yacc.c  */
#line 415 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long long"); }
    break;

  case 103:

/* Line 1806 of yacc.c  */
#line 417 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("float"); }
    break;

  case 104:

/* Line 1806 of yacc.c  */
#line 419 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("double"); }
    break;

  case 105:

/* Line 1806 of yacc.c  */
#line 421 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long double"); }
    break;

  case 106:

/* Line 1806 of yacc.c  */
#line 423 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("void"); }
    break;

  case 107:

/* Line 1806 of yacc.c  */
#line 426 "xi-grammar.y"
    { (yyval.ntype) = new NamedType((yyvsp[(1) - (2)].strval),(yyvsp[(2) - (2)].tparlist)); }
    break;

  case 108:

/* Line 1806 of yacc.c  */
#line 427 "xi-grammar.y"
    { 
                    const char* basename, *scope;
                    splitScopedName((yyvsp[(1) - (2)].strval), &scope, &basename);
                    (yyval.ntype) = new NamedType(basename, (yyvsp[(2) - (2)].tparlist), scope);
                }
    break;

  case 109:

/* Line 1806 of yacc.c  */
#line 435 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 110:

/* Line 1806 of yacc.c  */
#line 437 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ntype); }
    break;

  case 111:

/* Line 1806 of yacc.c  */
#line 441 "xi-grammar.y"
    { (yyval.ptype) = new PtrType((yyvsp[(1) - (2)].type)); }
    break;

  case 112:

/* Line 1806 of yacc.c  */
#line 445 "xi-grammar.y"
    { (yyvsp[(1) - (2)].ptype)->indirect(); (yyval.ptype) = (yyvsp[(1) - (2)].ptype); }
    break;

  case 113:

/* Line 1806 of yacc.c  */
#line 447 "xi-grammar.y"
    { (yyvsp[(1) - (2)].ptype)->indirect(); (yyval.ptype) = (yyvsp[(1) - (2)].ptype); }
    break;

  case 114:

/* Line 1806 of yacc.c  */
#line 451 "xi-grammar.y"
    { (yyval.ftype) = new FuncType((yyvsp[(1) - (8)].type), (yyvsp[(4) - (8)].strval), (yyvsp[(7) - (8)].plist)); }
    break;

  case 115:

/* Line 1806 of yacc.c  */
#line 455 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 116:

/* Line 1806 of yacc.c  */
#line 457 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 117:

/* Line 1806 of yacc.c  */
#line 459 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 118:

/* Line 1806 of yacc.c  */
#line 461 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ftype); }
    break;

  case 119:

/* Line 1806 of yacc.c  */
#line 463 "xi-grammar.y"
    { (yyval.type) = new ConstType((yyvsp[(2) - (2)].type)); }
    break;

  case 120:

/* Line 1806 of yacc.c  */
#line 465 "xi-grammar.y"
    { (yyval.type) = new ConstType((yyvsp[(1) - (2)].type)); }
    break;

  case 121:

/* Line 1806 of yacc.c  */
#line 469 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 122:

/* Line 1806 of yacc.c  */
#line 471 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 123:

/* Line 1806 of yacc.c  */
#line 473 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 124:

/* Line 1806 of yacc.c  */
#line 475 "xi-grammar.y"
    { (yyval.type) = new ConstType((yyvsp[(2) - (2)].type)); }
    break;

  case 125:

/* Line 1806 of yacc.c  */
#line 477 "xi-grammar.y"
    { (yyval.type) = new ConstType((yyvsp[(1) - (2)].type)); }
    break;

  case 126:

/* Line 1806 of yacc.c  */
#line 481 "xi-grammar.y"
    { (yyval.type) = new ReferenceType((yyvsp[(1) - (2)].type)); }
    break;

  case 127:

/* Line 1806 of yacc.c  */
#line 483 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 128:

/* Line 1806 of yacc.c  */
#line 487 "xi-grammar.y"
    { (yyval.type) = new ReferenceType((yyvsp[(1) - (2)].type)); }
    break;

  case 129:

/* Line 1806 of yacc.c  */
#line 489 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 130:

/* Line 1806 of yacc.c  */
#line 493 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 131:

/* Line 1806 of yacc.c  */
#line 497 "xi-grammar.y"
    { (yyval.val) = (yyvsp[(2) - (3)].val); }
    break;

  case 132:

/* Line 1806 of yacc.c  */
#line 501 "xi-grammar.y"
    { (yyval.vallist) = 0; }
    break;

  case 133:

/* Line 1806 of yacc.c  */
#line 503 "xi-grammar.y"
    { (yyval.vallist) = new ValueList((yyvsp[(1) - (2)].val), (yyvsp[(2) - (2)].vallist)); }
    break;

  case 134:

/* Line 1806 of yacc.c  */
#line 507 "xi-grammar.y"
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[(2) - (4)].type), (yyvsp[(3) - (4)].strval), (yyvsp[(4) - (4)].vallist)); }
    break;

  case 135:

/* Line 1806 of yacc.c  */
#line 511 "xi-grammar.y"
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[(3) - (6)].type), (yyvsp[(5) - (6)].strval), (yyvsp[(6) - (6)].vallist), 1); }
    break;

  case 136:

/* Line 1806 of yacc.c  */
#line 515 "xi-grammar.y"
    { (yyval.intval) = 0;}
    break;

  case 137:

/* Line 1806 of yacc.c  */
#line 517 "xi-grammar.y"
    { (yyval.intval) = 0;}
    break;

  case 138:

/* Line 1806 of yacc.c  */
#line 521 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 139:

/* Line 1806 of yacc.c  */
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

/* Line 1806 of yacc.c  */
#line 533 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (1)].intval); }
    break;

  case 141:

/* Line 1806 of yacc.c  */
#line 535 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (3)].intval) | (yyvsp[(3) - (3)].intval); }
    break;

  case 142:

/* Line 1806 of yacc.c  */
#line 539 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 143:

/* Line 1806 of yacc.c  */
#line 541 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 144:

/* Line 1806 of yacc.c  */
#line 545 "xi-grammar.y"
    { (yyval.cattr) = 0; }
    break;

  case 145:

/* Line 1806 of yacc.c  */
#line 547 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(2) - (3)].cattr); }
    break;

  case 146:

/* Line 1806 of yacc.c  */
#line 551 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (1)].cattr); }
    break;

  case 147:

/* Line 1806 of yacc.c  */
#line 553 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (3)].cattr) | (yyvsp[(3) - (3)].cattr); }
    break;

  case 148:

/* Line 1806 of yacc.c  */
#line 557 "xi-grammar.y"
    { python_doc = NULL; (yyval.intval) = 0; }
    break;

  case 149:

/* Line 1806 of yacc.c  */
#line 559 "xi-grammar.y"
    { python_doc = (yyvsp[(1) - (1)].strval); (yyval.intval) = 0; }
    break;

  case 150:

/* Line 1806 of yacc.c  */
#line 563 "xi-grammar.y"
    { (yyval.cattr) = Chare::CPYTHON; }
    break;

  case 151:

/* Line 1806 of yacc.c  */
#line 567 "xi-grammar.y"
    { (yyval.cattr) = 0; }
    break;

  case 152:

/* Line 1806 of yacc.c  */
#line 569 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(2) - (3)].cattr); }
    break;

  case 153:

/* Line 1806 of yacc.c  */
#line 573 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (1)].cattr); }
    break;

  case 154:

/* Line 1806 of yacc.c  */
#line 575 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (3)].cattr) | (yyvsp[(3) - (3)].cattr); }
    break;

  case 155:

/* Line 1806 of yacc.c  */
#line 579 "xi-grammar.y"
    { (yyval.cattr) = Chare::CMIGRATABLE; }
    break;

  case 156:

/* Line 1806 of yacc.c  */
#line 581 "xi-grammar.y"
    { (yyval.cattr) = Chare::CPYTHON; }
    break;

  case 157:

/* Line 1806 of yacc.c  */
#line 585 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 158:

/* Line 1806 of yacc.c  */
#line 587 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 159:

/* Line 1806 of yacc.c  */
#line 590 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 160:

/* Line 1806 of yacc.c  */
#line 592 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 161:

/* Line 1806 of yacc.c  */
#line 595 "xi-grammar.y"
    { (yyval.mv) = new MsgVar((yyvsp[(2) - (5)].type), (yyvsp[(3) - (5)].strval), (yyvsp[(1) - (5)].intval), (yyvsp[(4) - (5)].intval)); }
    break;

  case 162:

/* Line 1806 of yacc.c  */
#line 599 "xi-grammar.y"
    { (yyval.mvlist) = new MsgVarList((yyvsp[(1) - (1)].mv)); }
    break;

  case 163:

/* Line 1806 of yacc.c  */
#line 601 "xi-grammar.y"
    { (yyval.mvlist) = new MsgVarList((yyvsp[(1) - (2)].mv), (yyvsp[(2) - (2)].mvlist)); }
    break;

  case 164:

/* Line 1806 of yacc.c  */
#line 605 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, (yyvsp[(3) - (3)].ntype)); }
    break;

  case 165:

/* Line 1806 of yacc.c  */
#line 607 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, (yyvsp[(3) - (6)].ntype), (yyvsp[(5) - (6)].mvlist)); }
    break;

  case 166:

/* Line 1806 of yacc.c  */
#line 611 "xi-grammar.y"
    { (yyval.typelist) = 0; }
    break;

  case 167:

/* Line 1806 of yacc.c  */
#line 613 "xi-grammar.y"
    { (yyval.typelist) = (yyvsp[(2) - (2)].typelist); }
    break;

  case 168:

/* Line 1806 of yacc.c  */
#line 617 "xi-grammar.y"
    { (yyval.typelist) = new TypeList((yyvsp[(1) - (1)].ntype)); }
    break;

  case 169:

/* Line 1806 of yacc.c  */
#line 619 "xi-grammar.y"
    { (yyval.typelist) = new TypeList((yyvsp[(1) - (3)].ntype), (yyvsp[(3) - (3)].typelist)); }
    break;

  case 170:

/* Line 1806 of yacc.c  */
#line 623 "xi-grammar.y"
    { (yyval.chare) = new Chare(lineno, (yyvsp[(2) - (5)].cattr)|Chare::CCHARE, (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 171:

/* Line 1806 of yacc.c  */
#line 625 "xi-grammar.y"
    { (yyval.chare) = new MainChare(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 172:

/* Line 1806 of yacc.c  */
#line 629 "xi-grammar.y"
    { (yyval.chare) = new Group(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 173:

/* Line 1806 of yacc.c  */
#line 633 "xi-grammar.y"
    { (yyval.chare) = new NodeGroup(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 174:

/* Line 1806 of yacc.c  */
#line 637 "xi-grammar.y"
    {/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",(yyvsp[(2) - (4)].strval));
			(yyval.ntype) = new NamedType(buf); 
		}
    break;

  case 175:

/* Line 1806 of yacc.c  */
#line 643 "xi-grammar.y"
    { (yyval.ntype) = new NamedType((yyvsp[(2) - (3)].strval)); }
    break;

  case 176:

/* Line 1806 of yacc.c  */
#line 647 "xi-grammar.y"
    {  (yyval.chare) = new Array(lineno, (yyvsp[(2) - (6)].cattr), (yyvsp[(3) - (6)].ntype), (yyvsp[(4) - (6)].ntype), (yyvsp[(5) - (6)].typelist), (yyvsp[(6) - (6)].mbrlist)); }
    break;

  case 177:

/* Line 1806 of yacc.c  */
#line 649 "xi-grammar.y"
    {  (yyval.chare) = new Array(lineno, (yyvsp[(3) - (6)].cattr), (yyvsp[(2) - (6)].ntype), (yyvsp[(4) - (6)].ntype), (yyvsp[(5) - (6)].typelist), (yyvsp[(6) - (6)].mbrlist)); }
    break;

  case 178:

/* Line 1806 of yacc.c  */
#line 653 "xi-grammar.y"
    { (yyval.chare) = new Chare(lineno, (yyvsp[(2) - (5)].cattr)|Chare::CCHARE, new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist));}
    break;

  case 179:

/* Line 1806 of yacc.c  */
#line 655 "xi-grammar.y"
    { (yyval.chare) = new MainChare(lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 180:

/* Line 1806 of yacc.c  */
#line 659 "xi-grammar.y"
    { (yyval.chare) = new Group(lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 181:

/* Line 1806 of yacc.c  */
#line 663 "xi-grammar.y"
    { (yyval.chare) = new NodeGroup( lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 182:

/* Line 1806 of yacc.c  */
#line 667 "xi-grammar.y"
    { (yyval.chare) = new Array( lineno, 0, (yyvsp[(2) - (5)].ntype), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 183:

/* Line 1806 of yacc.c  */
#line 671 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[(3) - (4)].strval))); }
    break;

  case 184:

/* Line 1806 of yacc.c  */
#line 673 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[(3) - (7)].strval)), (yyvsp[(5) - (7)].mvlist)); }
    break;

  case 185:

/* Line 1806 of yacc.c  */
#line 677 "xi-grammar.y"
    { (yyval.type) = 0; }
    break;

  case 186:

/* Line 1806 of yacc.c  */
#line 679 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(2) - (2)].type); }
    break;

  case 187:

/* Line 1806 of yacc.c  */
#line 683 "xi-grammar.y"
    { (yyval.strval) = 0; }
    break;

  case 188:

/* Line 1806 of yacc.c  */
#line 685 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(2) - (2)].strval); }
    break;

  case 189:

/* Line 1806 of yacc.c  */
#line 687 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(2) - (2)].strval); }
    break;

  case 190:

/* Line 1806 of yacc.c  */
#line 689 "xi-grammar.y"
    {
		  XStr typeStr;
		  (yyvsp[(2) - (2)].ntype)->print(typeStr);
		  char *tmp = strdup(typeStr.get_string());
		  (yyval.strval) = tmp;
		}
    break;

  case 191:

/* Line 1806 of yacc.c  */
#line 698 "xi-grammar.y"
    { (yyval.tvar) = new TType(new NamedType((yyvsp[(2) - (3)].strval)), (yyvsp[(3) - (3)].type)); }
    break;

  case 192:

/* Line 1806 of yacc.c  */
#line 700 "xi-grammar.y"
    { (yyval.tvar) = new TFunc((yyvsp[(1) - (2)].ftype), (yyvsp[(2) - (2)].strval)); }
    break;

  case 193:

/* Line 1806 of yacc.c  */
#line 702 "xi-grammar.y"
    { (yyval.tvar) = new TName((yyvsp[(1) - (3)].type), (yyvsp[(2) - (3)].strval), (yyvsp[(3) - (3)].strval)); }
    break;

  case 194:

/* Line 1806 of yacc.c  */
#line 706 "xi-grammar.y"
    { (yyval.tvarlist) = new TVarList((yyvsp[(1) - (1)].tvar)); }
    break;

  case 195:

/* Line 1806 of yacc.c  */
#line 708 "xi-grammar.y"
    { (yyval.tvarlist) = new TVarList((yyvsp[(1) - (3)].tvar), (yyvsp[(3) - (3)].tvarlist)); }
    break;

  case 196:

/* Line 1806 of yacc.c  */
#line 712 "xi-grammar.y"
    { (yyval.tvarlist) = (yyvsp[(3) - (4)].tvarlist); }
    break;

  case 197:

/* Line 1806 of yacc.c  */
#line 716 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 198:

/* Line 1806 of yacc.c  */
#line 718 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 199:

/* Line 1806 of yacc.c  */
#line 720 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 200:

/* Line 1806 of yacc.c  */
#line 722 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 201:

/* Line 1806 of yacc.c  */
#line 724 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].message)); (yyvsp[(2) - (2)].message)->setTemplate((yyval.templat)); }
    break;

  case 202:

/* Line 1806 of yacc.c  */
#line 728 "xi-grammar.y"
    { (yyval.mbrlist) = 0; }
    break;

  case 203:

/* Line 1806 of yacc.c  */
#line 730 "xi-grammar.y"
    { (yyval.mbrlist) = (yyvsp[(2) - (4)].mbrlist); }
    break;

  case 204:

/* Line 1806 of yacc.c  */
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

/* Line 1806 of yacc.c  */
#line 742 "xi-grammar.y"
    { (yyval.mbrlist) = new AstChildren<Member>(-1, (yyvsp[(1) - (2)].member), (yyvsp[(2) - (2)].mbrlist)); }
    break;

  case 206:

/* Line 1806 of yacc.c  */
#line 746 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].readonly); }
    break;

  case 207:

/* Line 1806 of yacc.c  */
#line 748 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].readonly); }
    break;

  case 209:

/* Line 1806 of yacc.c  */
#line 751 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].member); }
    break;

  case 210:

/* Line 1806 of yacc.c  */
#line 753 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(2) - (2)].pupable); }
    break;

  case 211:

/* Line 1806 of yacc.c  */
#line 755 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(2) - (2)].includeFile); }
    break;

  case 212:

/* Line 1806 of yacc.c  */
#line 757 "xi-grammar.y"
    { (yyval.member) = new ClassDeclaration(lineno,(yyvsp[(2) - (2)].strval)); }
    break;

  case 213:

/* Line 1806 of yacc.c  */
#line 761 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 1); }
    break;

  case 214:

/* Line 1806 of yacc.c  */
#line 763 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 1); }
    break;

  case 215:

/* Line 1806 of yacc.c  */
#line 765 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[(3) - (9)].strval)) + '<' +
					    ((yyvsp[(5) - (9)].tparlist))->to_string() + '>').c_str()),
				    1);
		}
    break;

  case 216:

/* Line 1806 of yacc.c  */
#line 771 "xi-grammar.y"
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[(1) - (3)]).first_column, (yylsp[(1) - (3)]).last_column, (yylsp[(1) - (3)]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 1);
		}
    break;

  case 217:

/* Line 1806 of yacc.c  */
#line 777 "xi-grammar.y"
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[(1) - (6)]).first_column, (yylsp[(1) - (6)]).last_column, (yylsp[(1) - (6)]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 1);
		}
    break;

  case 218:

/* Line 1806 of yacc.c  */
#line 786 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 0); }
    break;

  case 219:

/* Line 1806 of yacc.c  */
#line 788 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 0); }
    break;

  case 220:

/* Line 1806 of yacc.c  */
#line 790 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[(3) - (9)].strval)) + '<' +
					    ((yyvsp[(5) - (9)].tparlist))->to_string() + '>').c_str()),
				    0);
		}
    break;

  case 221:

/* Line 1806 of yacc.c  */
#line 796 "xi-grammar.y"
    {
                  InitCall* rtn = new InitCall(lineno, (yyvsp[(6) - (9)].strval), 0);
                  rtn->setAccel();
                  (yyval.member) = rtn;
		}
    break;

  case 222:

/* Line 1806 of yacc.c  */
#line 804 "xi-grammar.y"
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[(1) - (1)].ntype),0); }
    break;

  case 223:

/* Line 1806 of yacc.c  */
#line 806 "xi-grammar.y"
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[(1) - (3)].ntype),(yyvsp[(3) - (3)].pupable)); }
    break;

  case 224:

/* Line 1806 of yacc.c  */
#line 809 "xi-grammar.y"
    { (yyval.includeFile) = new IncludeFile(lineno,(yyvsp[(1) - (1)].strval)); }
    break;

  case 225:

/* Line 1806 of yacc.c  */
#line 813 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].member); }
    break;

  case 226:

/* Line 1806 of yacc.c  */
#line 817 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].entry); }
    break;

  case 227:

/* Line 1806 of yacc.c  */
#line 819 "xi-grammar.y"
    {
                  (yyvsp[(2) - (2)].entry)->tspec = (yyvsp[(1) - (2)].tvarlist);
                  (yyval.member) = (yyvsp[(2) - (2)].entry);
                }
    break;

  case 228:

/* Line 1806 of yacc.c  */
#line 824 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (2)].member); }
    break;

  case 229:

/* Line 1806 of yacc.c  */
#line 826 "xi-grammar.y"
    {
          ERROR("invalid SDAG member",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
    break;

  case 230:

/* Line 1806 of yacc.c  */
#line 834 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 231:

/* Line 1806 of yacc.c  */
#line 836 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 232:

/* Line 1806 of yacc.c  */
#line 838 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 233:

/* Line 1806 of yacc.c  */
#line 840 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 234:

/* Line 1806 of yacc.c  */
#line 842 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 235:

/* Line 1806 of yacc.c  */
#line 844 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 236:

/* Line 1806 of yacc.c  */
#line 846 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 237:

/* Line 1806 of yacc.c  */
#line 848 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 238:

/* Line 1806 of yacc.c  */
#line 850 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 239:

/* Line 1806 of yacc.c  */
#line 852 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 240:

/* Line 1806 of yacc.c  */
#line 854 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 241:

/* Line 1806 of yacc.c  */
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

/* Line 1806 of yacc.c  */
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

/* Line 1806 of yacc.c  */
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

/* Line 1806 of yacc.c  */
#line 900 "xi-grammar.y"
    { (yyval.accelBlock) = new AccelBlock(lineno, new XStr((yyvsp[(3) - (5)].strval))); }
    break;

  case 245:

/* Line 1806 of yacc.c  */
#line 902 "xi-grammar.y"
    { (yyval.accelBlock) = new AccelBlock(lineno, NULL); }
    break;

  case 246:

/* Line 1806 of yacc.c  */
#line 906 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 247:

/* Line 1806 of yacc.c  */
#line 910 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 248:

/* Line 1806 of yacc.c  */
#line 912 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(2) - (3)].intval); }
    break;

  case 249:

/* Line 1806 of yacc.c  */
#line 914 "xi-grammar.y"
    { ERROR("invalid entry method attribute list",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
    break;

  case 250:

/* Line 1806 of yacc.c  */
#line 921 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (1)].intval); }
    break;

  case 251:

/* Line 1806 of yacc.c  */
#line 923 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (3)].intval) | (yyvsp[(3) - (3)].intval); }
    break;

  case 252:

/* Line 1806 of yacc.c  */
#line 927 "xi-grammar.y"
    { (yyval.intval) = STHREADED; }
    break;

  case 253:

/* Line 1806 of yacc.c  */
#line 929 "xi-grammar.y"
    { (yyval.intval) = SSYNC; }
    break;

  case 254:

/* Line 1806 of yacc.c  */
#line 931 "xi-grammar.y"
    { (yyval.intval) = SIGET; }
    break;

  case 255:

/* Line 1806 of yacc.c  */
#line 933 "xi-grammar.y"
    { (yyval.intval) = SLOCKED; }
    break;

  case 256:

/* Line 1806 of yacc.c  */
#line 935 "xi-grammar.y"
    { (yyval.intval) = SCREATEHERE; }
    break;

  case 257:

/* Line 1806 of yacc.c  */
#line 937 "xi-grammar.y"
    { (yyval.intval) = SCREATEHOME; }
    break;

  case 258:

/* Line 1806 of yacc.c  */
#line 939 "xi-grammar.y"
    { (yyval.intval) = SNOKEEP; }
    break;

  case 259:

/* Line 1806 of yacc.c  */
#line 941 "xi-grammar.y"
    { (yyval.intval) = SNOTRACE; }
    break;

  case 260:

/* Line 1806 of yacc.c  */
#line 943 "xi-grammar.y"
    { (yyval.intval) = SAPPWORK; }
    break;

  case 261:

/* Line 1806 of yacc.c  */
#line 945 "xi-grammar.y"
    { (yyval.intval) = SIMMEDIATE; }
    break;

  case 262:

/* Line 1806 of yacc.c  */
#line 947 "xi-grammar.y"
    { (yyval.intval) = SSKIPSCHED; }
    break;

  case 263:

/* Line 1806 of yacc.c  */
#line 949 "xi-grammar.y"
    { (yyval.intval) = SINLINE; }
    break;

  case 264:

/* Line 1806 of yacc.c  */
#line 951 "xi-grammar.y"
    { (yyval.intval) = SLOCAL; }
    break;

  case 265:

/* Line 1806 of yacc.c  */
#line 953 "xi-grammar.y"
    { (yyval.intval) = SPYTHON; }
    break;

  case 266:

/* Line 1806 of yacc.c  */
#line 955 "xi-grammar.y"
    { (yyval.intval) = SMEM; }
    break;

  case 267:

/* Line 1806 of yacc.c  */
#line 957 "xi-grammar.y"
    { (yyval.intval) = SREDUCE; }
    break;

  case 268:

/* Line 1806 of yacc.c  */
#line 959 "xi-grammar.y"
    {
		  ERROR("invalid entry method attribute",
		        (yylsp[(1) - (1)]).first_column, (yylsp[(1) - (1)]).last_column);
		  yyclearin;
		  yyerrok;
		}
    break;

  case 269:

/* Line 1806 of yacc.c  */
#line 968 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 270:

/* Line 1806 of yacc.c  */
#line 970 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 271:

/* Line 1806 of yacc.c  */
#line 972 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 272:

/* Line 1806 of yacc.c  */
#line 976 "xi-grammar.y"
    { (yyval.strval) = ""; }
    break;

  case 273:

/* Line 1806 of yacc.c  */
#line 978 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 274:

/* Line 1806 of yacc.c  */
#line 980 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[(1) - (3)].strval))+strlen((yyvsp[(3) - (3)].strval))+3];
			sprintf(tmp,"%s, %s", (yyvsp[(1) - (3)].strval), (yyvsp[(3) - (3)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 275:

/* Line 1806 of yacc.c  */
#line 988 "xi-grammar.y"
    { (yyval.strval) = ""; }
    break;

  case 276:

/* Line 1806 of yacc.c  */
#line 990 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 277:

/* Line 1806 of yacc.c  */
#line 992 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s[%s]%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 278:

/* Line 1806 of yacc.c  */
#line 998 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s{%s}%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 279:

/* Line 1806 of yacc.c  */
#line 1004 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s(%s)%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 280:

/* Line 1806 of yacc.c  */
#line 1010 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(2) - (4)].strval))+strlen((yyvsp[(4) - (4)].strval))+3];
			sprintf(tmp,"(%s)%s", (yyvsp[(2) - (4)].strval), (yyvsp[(4) - (4)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 281:

/* Line 1806 of yacc.c  */
#line 1018 "xi-grammar.y"
    {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			(yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].type),(yyvsp[(2) - (3)].strval));
		}
    break;

  case 282:

/* Line 1806 of yacc.c  */
#line 1025 "xi-grammar.y"
    { 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			(yyval.intval) = 0;
		}
    break;

  case 283:

/* Line 1806 of yacc.c  */
#line 1033 "xi-grammar.y"
    { 
			in_braces=0;
			(yyval.intval) = 0;
		}
    break;

  case 284:

/* Line 1806 of yacc.c  */
#line 1040 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (1)].type));}
    break;

  case 285:

/* Line 1806 of yacc.c  */
#line 1042 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].type),(yyvsp[(2) - (3)].strval)); (yyval.pname)->setConditional((yyvsp[(3) - (3)].intval)); }
    break;

  case 286:

/* Line 1806 of yacc.c  */
#line 1044 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (4)].type),(yyvsp[(2) - (4)].strval),0,(yyvsp[(4) - (4)].val));}
    break;

  case 287:

/* Line 1806 of yacc.c  */
#line 1046 "xi-grammar.y"
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].pname)->getType(), (yyvsp[(1) - (3)].pname)->getName() ,(yyvsp[(2) - (3)].strval));
		}
    break;

  case 288:

/* Line 1806 of yacc.c  */
#line 1052 "xi-grammar.y"
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READONLY; }
    break;

  case 289:

/* Line 1806 of yacc.c  */
#line 1053 "xi-grammar.y"
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READWRITE; }
    break;

  case 290:

/* Line 1806 of yacc.c  */
#line 1054 "xi-grammar.y"
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_WRITEONLY; }
    break;

  case 291:

/* Line 1806 of yacc.c  */
#line 1057 "xi-grammar.y"
    { (yyval.xstrptr) = new XStr((yyvsp[(1) - (1)].strval)); }
    break;

  case 292:

/* Line 1806 of yacc.c  */
#line 1058 "xi-grammar.y"
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "->" << (yyvsp[(4) - (4)].strval); }
    break;

  case 293:

/* Line 1806 of yacc.c  */
#line 1059 "xi-grammar.y"
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[(1) - (3)].xstrptr)) << "." << (yyvsp[(3) - (3)].strval); }
    break;

  case 294:

/* Line 1806 of yacc.c  */
#line 1061 "xi-grammar.y"
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "[" << *((yyvsp[(3) - (4)].xstrptr)) << "]";
                  delete (yyvsp[(1) - (4)].xstrptr);
                  delete (yyvsp[(3) - (4)].xstrptr);
                }
    break;

  case 295:

/* Line 1806 of yacc.c  */
#line 1068 "xi-grammar.y"
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "[" << (yyvsp[(3) - (4)].strval) << "]";
                  delete (yyvsp[(1) - (4)].xstrptr);
                }
    break;

  case 296:

/* Line 1806 of yacc.c  */
#line 1074 "xi-grammar.y"
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "(" << *((yyvsp[(3) - (4)].xstrptr)) << ")";
                  delete (yyvsp[(1) - (4)].xstrptr);
                  delete (yyvsp[(3) - (4)].xstrptr);
                }
    break;

  case 297:

/* Line 1806 of yacc.c  */
#line 1083 "xi-grammar.y"
    {
                  in_bracket = 0;
                  (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].pname)->getType(), (yyvsp[(1) - (3)].pname)->getName(), (yyvsp[(2) - (3)].strval));
                }
    break;

  case 298:

/* Line 1806 of yacc.c  */
#line 1090 "xi-grammar.y"
    {
                  (yyval.pname) = new Parameter(lineno, (yyvsp[(3) - (7)].type), (yyvsp[(4) - (7)].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[(6) - (7)].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[(1) - (7)].intval));
                }
    break;

  case 299:

/* Line 1806 of yacc.c  */
#line 1096 "xi-grammar.y"
    {
		  (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (5)].type), (yyvsp[(2) - (5)].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[(4) - (5)].xstrptr));
                  (yyval.pname)->setAccelBufferType(Parameter::ACCEL_BUFFER_TYPE_READWRITE);
		}
    break;

  case 300:

/* Line 1806 of yacc.c  */
#line 1102 "xi-grammar.y"
    {
                  (yyval.pname) = (yyvsp[(3) - (6)].pname);
                  (yyval.pname)->setAccelInstName((yyvsp[(5) - (6)].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[(1) - (6)].intval));
		}
    break;

  case 301:

/* Line 1806 of yacc.c  */
#line 1110 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (1)].pname)); }
    break;

  case 302:

/* Line 1806 of yacc.c  */
#line 1112 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (3)].pname),(yyvsp[(3) - (3)].plist)); }
    break;

  case 303:

/* Line 1806 of yacc.c  */
#line 1116 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (1)].pname)); }
    break;

  case 304:

/* Line 1806 of yacc.c  */
#line 1118 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (3)].pname),(yyvsp[(3) - (3)].plist)); }
    break;

  case 305:

/* Line 1806 of yacc.c  */
#line 1122 "xi-grammar.y"
    { (yyval.plist) = (yyvsp[(2) - (3)].plist); }
    break;

  case 306:

/* Line 1806 of yacc.c  */
#line 1124 "xi-grammar.y"
    { (yyval.plist) = new ParamList(new Parameter(0, new BuiltinType("void"))); }
    break;

  case 307:

/* Line 1806 of yacc.c  */
#line 1128 "xi-grammar.y"
    { (yyval.plist) = (yyvsp[(2) - (3)].plist); }
    break;

  case 308:

/* Line 1806 of yacc.c  */
#line 1130 "xi-grammar.y"
    { (yyval.plist) = 0; }
    break;

  case 309:

/* Line 1806 of yacc.c  */
#line 1134 "xi-grammar.y"
    { (yyval.val) = 0; }
    break;

  case 310:

/* Line 1806 of yacc.c  */
#line 1136 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(3) - (3)].strval)); }
    break;

  case 311:

/* Line 1806 of yacc.c  */
#line 1140 "xi-grammar.y"
    { (yyval.sentry) = 0; }
    break;

  case 312:

/* Line 1806 of yacc.c  */
#line 1142 "xi-grammar.y"
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[(1) - (1)].sc)); }
    break;

  case 313:

/* Line 1806 of yacc.c  */
#line 1144 "xi-grammar.y"
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[(2) - (4)].slist)); }
    break;

  case 314:

/* Line 1806 of yacc.c  */
#line 1148 "xi-grammar.y"
    { (yyval.slist) = new SListConstruct((yyvsp[(1) - (1)].sc)); }
    break;

  case 315:

/* Line 1806 of yacc.c  */
#line 1150 "xi-grammar.y"
    { (yyval.slist) = new SListConstruct((yyvsp[(1) - (2)].sc), (yyvsp[(2) - (2)].slist));  }
    break;

  case 316:

/* Line 1806 of yacc.c  */
#line 1154 "xi-grammar.y"
    { (yyval.olist) = new OListConstruct((yyvsp[(1) - (1)].sc)); }
    break;

  case 317:

/* Line 1806 of yacc.c  */
#line 1156 "xi-grammar.y"
    { (yyval.olist) = new OListConstruct((yyvsp[(1) - (2)].sc), (yyvsp[(2) - (2)].slist)); }
    break;

  case 318:

/* Line 1806 of yacc.c  */
#line 1160 "xi-grammar.y"
    { (yyval.clist) = new CaseListConstruct((yyvsp[(1) - (1)].when)); }
    break;

  case 319:

/* Line 1806 of yacc.c  */
#line 1162 "xi-grammar.y"
    { (yyval.clist) = new CaseListConstruct((yyvsp[(1) - (2)].when), (yyvsp[(2) - (2)].clist)); }
    break;

  case 320:

/* Line 1806 of yacc.c  */
#line 1164 "xi-grammar.y"
    {
		  ERROR("case blocks can only contain when clauses",
		        (yylsp[(1) - (1)]).first_column, (yylsp[(1) - (1)]).last_column);
		  (yyval.clist) = 0;
		}
    break;

  case 321:

/* Line 1806 of yacc.c  */
#line 1172 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 322:

/* Line 1806 of yacc.c  */
#line 1174 "xi-grammar.y"
    { (yyval.strval) = 0; }
    break;

  case 323:

/* Line 1806 of yacc.c  */
#line 1178 "xi-grammar.y"
    { (yyval.when) = new WhenConstruct((yyvsp[(2) - (4)].entrylist), 0); }
    break;

  case 324:

/* Line 1806 of yacc.c  */
#line 1180 "xi-grammar.y"
    { (yyval.when) = new WhenConstruct((yyvsp[(2) - (3)].entrylist), (yyvsp[(3) - (3)].sc)); }
    break;

  case 325:

/* Line 1806 of yacc.c  */
#line 1182 "xi-grammar.y"
    { (yyval.when) = new WhenConstruct((yyvsp[(2) - (5)].entrylist), (yyvsp[(4) - (5)].slist)); }
    break;

  case 326:

/* Line 1806 of yacc.c  */
#line 1186 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 327:

/* Line 1806 of yacc.c  */
#line 1188 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 328:

/* Line 1806 of yacc.c  */
#line 1190 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 329:

/* Line 1806 of yacc.c  */
#line 1192 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 330:

/* Line 1806 of yacc.c  */
#line 1194 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 331:

/* Line 1806 of yacc.c  */
#line 1196 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 332:

/* Line 1806 of yacc.c  */
#line 1198 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 333:

/* Line 1806 of yacc.c  */
#line 1200 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 334:

/* Line 1806 of yacc.c  */
#line 1202 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 335:

/* Line 1806 of yacc.c  */
#line 1204 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 336:

/* Line 1806 of yacc.c  */
#line 1206 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 337:

/* Line 1806 of yacc.c  */
#line 1208 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 338:

/* Line 1806 of yacc.c  */
#line 1212 "xi-grammar.y"
    { (yyval.sc) = new AtomicConstruct((yyvsp[(4) - (5)].strval), (yyvsp[(2) - (5)].strval), (yylsp[(3) - (5)]).first_line); }
    break;

  case 339:

/* Line 1806 of yacc.c  */
#line 1214 "xi-grammar.y"
    { (yyval.sc) = new OverlapConstruct((yyvsp[(3) - (4)].olist)); }
    break;

  case 340:

/* Line 1806 of yacc.c  */
#line 1216 "xi-grammar.y"
    { (yyval.sc) = (yyvsp[(1) - (1)].when); }
    break;

  case 341:

/* Line 1806 of yacc.c  */
#line 1218 "xi-grammar.y"
    { (yyval.sc) = new CaseConstruct((yyvsp[(3) - (4)].clist)); }
    break;

  case 342:

/* Line 1806 of yacc.c  */
#line 1220 "xi-grammar.y"
    { (yyval.sc) = new ForConstruct((yyvsp[(3) - (11)].intexpr), (yyvsp[(5) - (11)].intexpr), (yyvsp[(7) - (11)].intexpr), (yyvsp[(10) - (11)].slist)); }
    break;

  case 343:

/* Line 1806 of yacc.c  */
#line 1222 "xi-grammar.y"
    { (yyval.sc) = new ForConstruct((yyvsp[(3) - (9)].intexpr), (yyvsp[(5) - (9)].intexpr), (yyvsp[(7) - (9)].intexpr), (yyvsp[(9) - (9)].sc)); }
    break;

  case 344:

/* Line 1806 of yacc.c  */
#line 1224 "xi-grammar.y"
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[(3) - (12)].strval)), (yyvsp[(6) - (12)].intexpr),
		             (yyvsp[(8) - (12)].intexpr), (yyvsp[(10) - (12)].intexpr), (yyvsp[(12) - (12)].sc)); }
    break;

  case 345:

/* Line 1806 of yacc.c  */
#line 1227 "xi-grammar.y"
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[(3) - (14)].strval)), (yyvsp[(6) - (14)].intexpr),
		             (yyvsp[(8) - (14)].intexpr), (yyvsp[(10) - (14)].intexpr), (yyvsp[(13) - (14)].slist)); }
    break;

  case 346:

/* Line 1806 of yacc.c  */
#line 1230 "xi-grammar.y"
    { (yyval.sc) = new IfConstruct((yyvsp[(3) - (6)].intexpr), (yyvsp[(5) - (6)].sc), (yyvsp[(6) - (6)].sc)); }
    break;

  case 347:

/* Line 1806 of yacc.c  */
#line 1232 "xi-grammar.y"
    { (yyval.sc) = new IfConstruct((yyvsp[(3) - (8)].intexpr), (yyvsp[(6) - (8)].slist), (yyvsp[(8) - (8)].sc)); }
    break;

  case 348:

/* Line 1806 of yacc.c  */
#line 1234 "xi-grammar.y"
    { (yyval.sc) = new WhileConstruct((yyvsp[(3) - (5)].intexpr), (yyvsp[(5) - (5)].sc)); }
    break;

  case 349:

/* Line 1806 of yacc.c  */
#line 1236 "xi-grammar.y"
    { (yyval.sc) = new WhileConstruct((yyvsp[(3) - (7)].intexpr), (yyvsp[(6) - (7)].slist)); }
    break;

  case 350:

/* Line 1806 of yacc.c  */
#line 1238 "xi-grammar.y"
    { (yyval.sc) = new AtomicConstruct((yyvsp[(2) - (3)].strval), NULL, (yyloc).first_line); }
    break;

  case 351:

/* Line 1806 of yacc.c  */
#line 1240 "xi-grammar.y"
    {
		  ERROR("unknown SDAG construct or malformed entry method declaration.\n"
		        "You may have forgotten to terminate a previous entry method declaration with a"
		        " semicolon or forgotten to mark a block of sequential SDAG code as 'serial'",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
    break;

  case 352:

/* Line 1806 of yacc.c  */
#line 1250 "xi-grammar.y"
    { (yyval.sc) = 0; }
    break;

  case 353:

/* Line 1806 of yacc.c  */
#line 1252 "xi-grammar.y"
    { (yyval.sc) = new ElseConstruct((yyvsp[(2) - (2)].sc)); }
    break;

  case 354:

/* Line 1806 of yacc.c  */
#line 1254 "xi-grammar.y"
    { (yyval.sc) = new ElseConstruct((yyvsp[(3) - (4)].slist)); }
    break;

  case 355:

/* Line 1806 of yacc.c  */
#line 1258 "xi-grammar.y"
    { (yyval.intexpr) = new IntExprConstruct((yyvsp[(1) - (1)].strval)); }
    break;

  case 356:

/* Line 1806 of yacc.c  */
#line 1262 "xi-grammar.y"
    { in_int_expr = 0; (yyval.intval) = 0; }
    break;

  case 357:

/* Line 1806 of yacc.c  */
#line 1266 "xi-grammar.y"
    { in_int_expr = 1; (yyval.intval) = 0; }
    break;

  case 358:

/* Line 1806 of yacc.c  */
#line 1270 "xi-grammar.y"
    {
		  (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[(1) - (2)].strval), (yyvsp[(2) - (2)].plist), 0, 0, 0, (yyloc).first_line, (yyloc).last_line);
		}
    break;

  case 359:

/* Line 1806 of yacc.c  */
#line 1274 "xi-grammar.y"
    {
		  (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[(1) - (5)].strval), (yyvsp[(5) - (5)].plist), 0, 0, (yyvsp[(3) - (5)].strval), (yyloc).first_line, (yyloc).last_line);
		}
    break;

  case 360:

/* Line 1806 of yacc.c  */
#line 1280 "xi-grammar.y"
    { (yyval.entrylist) = new EntryList((yyvsp[(1) - (1)].entry)); }
    break;

  case 361:

/* Line 1806 of yacc.c  */
#line 1282 "xi-grammar.y"
    { (yyval.entrylist) = new EntryList((yyvsp[(1) - (3)].entry),(yyvsp[(3) - (3)].entrylist)); }
    break;

  case 362:

/* Line 1806 of yacc.c  */
#line 1286 "xi-grammar.y"
    { in_bracket=1; }
    break;

  case 363:

/* Line 1806 of yacc.c  */
#line 1289 "xi-grammar.y"
    { in_bracket=0; }
    break;

  case 364:

/* Line 1806 of yacc.c  */
#line 1293 "xi-grammar.y"
    { if (!macroDefined((yyvsp[(2) - (2)].strval), 1)) in_comment = 1; }
    break;

  case 365:

/* Line 1806 of yacc.c  */
#line 1297 "xi-grammar.y"
    { if (!macroDefined((yyvsp[(2) - (2)].strval), 0)) in_comment = 1; }
    break;



/* Line 1806 of yacc.c  */
#line 5209 "y.tab.c"
      default: break;
    }
  /* User semantic actions sometimes alter yychar, and that requires
     that yytoken be updated with the new translation.  We take the
     approach of translating immediately before every use of yytoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     YYABORT, YYACCEPT, or YYERROR immediately after altering yychar or
     if it invokes YYBACKUP.  In the case of YYABORT or YYACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of YYERROR or YYBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
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
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == YYEMPTY ? YYEMPTY : YYTRANSLATE (yychar);

  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (YY_("syntax error"));
#else
# define YYSYNTAX_ERROR yysyntax_error (&yymsg_alloc, &yymsg, \
                                        yyssp, yytoken)
      {
        char const *yymsgp = YY_("syntax error");
        int yysyntax_error_status;
        yysyntax_error_status = YYSYNTAX_ERROR;
        if (yysyntax_error_status == 0)
          yymsgp = yymsg;
        else if (yysyntax_error_status == 1)
          {
            if (yymsg != yymsgbuf)
              YYSTACK_FREE (yymsg);
            yymsg = (char *) YYSTACK_ALLOC (yymsg_alloc);
            if (!yymsg)
              {
                yymsg = yymsgbuf;
                yymsg_alloc = sizeof yymsgbuf;
                yysyntax_error_status = 2;
              }
            else
              {
                yysyntax_error_status = YYSYNTAX_ERROR;
                yymsgp = yymsg;
              }
          }
        yyerror (yymsgp);
        if (yysyntax_error_status == 2)
          goto yyexhaustedlab;
      }
# undef YYSYNTAX_ERROR
#endif
    }

  yyerror_range[1] = yylloc;

  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
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

  /* Else will try to reuse lookahead token after shifting the error
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

  yyerror_range[1] = yylsp[1-yylen];
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
      if (!yypact_value_is_default (yyn))
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

      yyerror_range[1] = *yylsp;
      yydestruct ("Error: popping",
		  yystos[yystate], yyvsp, yylsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  *++yyvsp = yylval;

  yyerror_range[2] = yylloc;
  /* Using YYLLOC is tempting, but would change the location of
     the lookahead.  YYLOC is available though.  */
  YYLLOC_DEFAULT (yyloc, yyerror_range, 2);
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

#if !defined(yyoverflow) || YYERROR_VERBOSE
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      yytoken = YYTRANSLATE (yychar);
      yydestruct ("Cleanup: discarding lookahead",
                  yytoken, &yylval, &yylloc);
    }
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



/* Line 2067 of yacc.c  */
#line 1300 "xi-grammar.y"


void yyerror(const char *msg) { }

