/* A Bison parser, made by GNU Bison 3.0.4.  */

/* Bison implementation for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015 Free Software Foundation, Inc.

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
#define YYBISON_VERSION "3.0.4"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1




/* Copy the first part of user declarations.  */
#line 2 "xi-grammar.y" /* yacc.c:339  */

#include <iostream>
#include <string>
#include <string.h>
#include "xi-symbol.h"
#include "sdag/constructs/Constructs.h"
#include "EToken.h"

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
void ReservedWord(int token);
}

#line 112 "xi-grammar.tab.C" /* yacc.c:339  */

# ifndef YY_NULLPTR
#  if defined __cplusplus && 201103L <= __cplusplus
#   define YY_NULLPTR nullptr
#  else
#   define YY_NULLPTR 0
#  endif
# endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

/* In a future release of Bison, this section will be replaced
   by #include "xi-grammar.tab.h".  */
#ifndef YY_YY_XI_GRAMMAR_TAB_H_INCLUDED
# define YY_YY_XI_GRAMMAR_TAB_H_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif
#if YYDEBUG
extern int yydebug;
#endif

/* Token type.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
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

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED

union YYSTYPE
{
#line 50 "xi-grammar.y" /* yacc.c:355  */

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
  Chare::attrib_t cattr;
  SdagConstruct *sc;
  IntExprConstruct *intexpr;
  WhenConstruct *when;
  SListConstruct *slist;
  CaseListConstruct *clist;
  OListConstruct *olist;
  SdagEntryConstruct *sentry;
  XStr* xstrptr;
  AccelBlock* accelBlock;

#line 269 "xi-grammar.tab.C" /* yacc.c:355  */
};

typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif

/* Location type.  */
#if ! defined YYLTYPE && ! defined YYLTYPE_IS_DECLARED
typedef struct YYLTYPE YYLTYPE;
struct YYLTYPE
{
  int first_line;
  int first_column;
  int last_line;
  int last_column;
};
# define YYLTYPE_IS_DECLARED 1
# define YYLTYPE_IS_TRIVIAL 1
#endif


extern YYSTYPE yylval;
extern YYLTYPE yylloc;
int yyparse (void);

#endif /* !YY_YY_XI_GRAMMAR_TAB_H_INCLUDED  */

/* Copy the second part of user declarations.  */

#line 300 "xi-grammar.tab.C" /* yacc.c:358  */

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
#else
typedef signed char yytype_int8;
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
# elif ! defined YYSIZE_T
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
#   define YY_(Msgid) dgettext ("bison-runtime", Msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(Msgid) Msgid
# endif
#endif

#ifndef YY_ATTRIBUTE
# if (defined __GNUC__                                               \
      && (2 < __GNUC__ || (__GNUC__ == 2 && 96 <= __GNUC_MINOR__)))  \
     || defined __SUNPRO_C && 0x5110 <= __SUNPRO_C
#  define YY_ATTRIBUTE(Spec) __attribute__(Spec)
# else
#  define YY_ATTRIBUTE(Spec) /* empty */
# endif
#endif

#ifndef YY_ATTRIBUTE_PURE
# define YY_ATTRIBUTE_PURE   YY_ATTRIBUTE ((__pure__))
#endif

#ifndef YY_ATTRIBUTE_UNUSED
# define YY_ATTRIBUTE_UNUSED YY_ATTRIBUTE ((__unused__))
#endif

#if !defined _Noreturn \
     && (!defined __STDC_VERSION__ || __STDC_VERSION__ < 201112)
# if defined _MSC_VER && 1200 <= _MSC_VER
#  define _Noreturn __declspec (noreturn)
# else
#  define _Noreturn YY_ATTRIBUTE ((__noreturn__))
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(E) ((void) (E))
#else
# define YYUSE(E) /* empty */
#endif

#if defined __GNUC__ && 407 <= __GNUC__ * 100 + __GNUC_MINOR__
/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN \
    _Pragma ("GCC diagnostic push") \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")\
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# define YY_IGNORE_MAYBE_UNINITIALIZED_END \
    _Pragma ("GCC diagnostic pop")
#else
# define YY_INITIAL_VALUE(Value) Value
#endif
#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
# define YY_INITIAL_VALUE(Value) /* Nothing. */
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
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
      /* Use EXIT_SUCCESS as a witness for stdlib.h.  */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's 'empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
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
#   if ! defined malloc && ! defined EXIT_SUCCESS
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS
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
# define YYSTACK_RELOCATE(Stack_alloc, Stack)                           \
    do                                                                  \
      {                                                                 \
        YYSIZE_T yynewbytes;                                            \
        YYCOPY (&yyptr->Stack_alloc, Stack, yysize);                    \
        Stack = &yyptr->Stack_alloc;                                    \
        yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
        yyptr += yynewbytes / sizeof (*yyptr);                          \
      }                                                                 \
    while (0)

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, (Count) * sizeof (*(Src)))
#  else
#   define YYCOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          YYSIZE_T yyi;                         \
          for (yyi = 0; yyi < (Count); yyi++)   \
            (Dst)[yyi] = (Src)[yyi];            \
        }                                       \
      while (0)
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  55
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   1473

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  89
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  116
/* YYNRULES -- Number of rules.  */
#define YYNRULES  364
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  710

/* YYTRANSLATE[YYX] -- Symbol number corresponding to YYX as returned
   by yylex, with out-of-bounds checking.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   327

#define YYTRANSLATE(YYX)                                                \
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, without out-of-bounds checking.  */
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
  /* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   191,   191,   196,   199,   204,   205,   210,   211,   216,
     218,   219,   220,   222,   223,   224,   226,   227,   228,   229,
     230,   234,   235,   236,   237,   238,   239,   240,   241,   242,
     243,   244,   245,   246,   247,   248,   249,   250,   253,   254,
     255,   256,   257,   258,   259,   260,   261,   262,   263,   265,
     267,   268,   271,   272,   273,   274,   277,   279,   287,   291,
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
     690,   692,   694,   698,   700,   704,   708,   710,   712,   714,
     716,   720,   722,   727,   734,   738,   740,   742,   743,   745,
     747,   749,   753,   755,   757,   763,   769,   778,   780,   782,
     788,   796,   798,   801,   805,   809,   811,   816,   818,   826,
     828,   830,   832,   834,   836,   838,   840,   842,   844,   846,
     849,   859,   876,   892,   894,   898,   903,   904,   906,   913,
     915,   919,   921,   923,   925,   927,   929,   931,   933,   935,
     937,   939,   941,   943,   945,   947,   949,   951,   960,   962,
     964,   969,   970,   972,   981,   982,   984,   990,   996,  1002,
    1010,  1017,  1025,  1032,  1034,  1036,  1038,  1045,  1046,  1047,
    1050,  1051,  1052,  1053,  1060,  1066,  1075,  1082,  1088,  1094,
    1102,  1104,  1108,  1110,  1114,  1116,  1120,  1122,  1127,  1128,
    1132,  1134,  1136,  1140,  1142,  1146,  1148,  1152,  1154,  1156,
    1164,  1167,  1170,  1172,  1174,  1178,  1180,  1182,  1184,  1186,
    1188,  1190,  1192,  1194,  1196,  1198,  1200,  1204,  1206,  1208,
    1210,  1212,  1214,  1216,  1219,  1222,  1224,  1226,  1228,  1230,
    1232,  1243,  1244,  1246,  1250,  1254,  1258,  1262,  1266,  1272,
    1274,  1278,  1281,  1285,  1289
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || 0
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
  "HashIFDefComment", YY_NULLPTR
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[NUM] -- (External) token number corresponding to the
   (internal) symbol number NUM (which must be that of a token).  */
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

#define YYPACT_NINF -589

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-589)))

#define YYTABLE_NINF -316

#define yytable_value_is_error(Yytable_value) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
     180,  1303,  1303,    54,  -589,   180,  -589,  -589,  -589,  -589,
    -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,
    -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,
    -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,
    -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,
    -589,  -589,  -589,    42,    42,  -589,  -589,  -589,   592,  -589,
    -589,  -589,     2,  1303,   152,  1303,  1303,   134,   887,    -5,
     936,   592,  -589,  -589,  -589,  1390,    51,    78,  -589,    85,
    -589,  -589,  -589,  -589,   -18,   129,    41,    41,   -11,    78,
      63,    63,    63,    63,   112,   118,  1303,   182,   139,   592,
    -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,   252,  -589,
    -589,  -589,  -589,   183,  -589,  -589,  -589,  -589,  -589,  -589,
    -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,  1390,
    -589,   121,  -589,  -589,  -589,  -589,   236,   149,  -589,  -589,
     194,   198,   208,    29,  -589,    78,   592,    85,   161,    59,
     -18,   217,   573,  1408,   194,   198,   208,  -589,     3,    78,
    -589,    78,    78,   235,    78,   225,  -589,    75,  1303,  1303,
    1303,  1303,  1093,   219,   220,   192,  1303,  -589,  -589,  -589,
    1334,   230,    63,    63,    63,    63,   219,   118,  -589,  -589,
    -589,  -589,  -589,  -589,  -589,   270,  -589,  -589,  -589,   181,
    -589,  -589,  1377,  -589,  -589,  -589,  -589,  -589,  -589,  1303,
     232,   257,   -18,   255,   -18,   231,  -589,   239,   234,     8,
    -589,   238,  -589,   -35,    30,    77,   245,    89,    78,  -589,
    -589,   246,   240,   241,   247,   247,   247,   247,  -589,  1303,
     248,   258,   249,  1163,  1303,   266,  1303,  -589,  -589,   254,
     263,   278,  1303,   101,  1303,   280,   244,   183,  1303,  1303,
    1303,  1303,  1303,  1303,  -589,  -589,  -589,  -589,   281,  -589,
     282,  -589,   241,  -589,  -589,   293,   294,   242,   287,   -18,
    -589,    78,  1303,  -589,   286,  -589,   -18,    41,  1377,    41,
      41,  1377,    41,  -589,  -589,    75,  -589,    78,   179,   179,
     179,   179,   289,  -589,   266,  -589,   247,   247,  -589,   192,
     358,   292,   237,  -589,   295,  1334,  -589,  -589,   247,   247,
     247,   247,   247,   214,  1377,  -589,   301,   -18,   255,   -18,
     -18,  -589,   -35,   302,  -589,   298,  -589,   305,   307,   306,
      78,   310,   308,  -589,   314,  -589,  -589,   319,  -589,  -589,
    -589,  -589,  -589,  -589,   179,   179,  -589,  -589,  1408,     4,
     316,  1408,  -589,  -589,  -589,  -589,  -589,   179,   179,   179,
     179,   179,  -589,   358,  -589,  1347,  -589,  -589,  -589,  -589,
    -589,  -589,   313,  -589,  -589,  -589,   317,  -589,    55,   318,
    -589,    78,  -589,   668,   365,   331,   335,   319,  -589,  -589,
    -589,  -589,  1303,  -589,  -589,  -589,  -589,  -589,  -589,  -589,
    -589,   333,  -589,  1303,   -18,   334,   328,  1408,    41,    41,
      41,  -589,  -589,   903,  1023,  -589,   183,  -589,  -589,   329,
     341,    24,   330,  1408,  -589,   336,   339,   340,   342,  -589,
    -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,
    -589,  -589,   361,  -589,   332,  -589,  -589,   348,   360,   344,
     301,  1303,  -589,   353,   366,  -589,  -589,   120,  -589,  -589,
    -589,  -589,  -589,  -589,  -589,  -589,  -589,   411,  -589,   954,
     488,   301,  -589,  -589,  -589,  -589,    85,  -589,  1303,  -589,
    -589,   368,   369,   368,   397,   376,   400,   368,   381,  -589,
     304,   -18,  -589,  -589,  -589,   437,   301,  -589,   -18,   405,
     -18,   155,   385,   496,   507,  -589,   389,   -18,  1316,   390,
     399,   217,   379,   488,   383,  -589,   395,   384,   388,  -589,
     -18,   397,   321,  -589,   396,   435,   -18,   388,   368,   401,
     368,   408,   400,   368,   409,   -18,   416,  1316,  -589,   183,
    -589,  -589,   439,  -589,   367,   389,   -18,   368,  -589,   576,
     298,  -589,  -589,   418,  -589,  -589,   217,   716,   -18,   442,
     -18,   507,   389,   -18,  1316,   217,  -589,  -589,  -589,  -589,
    -589,  -589,  -589,  -589,  1303,   422,   423,   414,   -18,   428,
     -18,   304,  -589,   301,  -589,  -589,   304,   454,   430,   419,
     388,   429,   -18,   388,   433,  -589,   434,  1408,   955,  -589,
     217,   -18,   440,   443,  -589,   445,   723,  -589,   -18,   368,
     734,  -589,   217,   770,  -589,  1303,  1303,   -18,   438,  -589,
    1303,   388,   -18,  -589,   454,   304,  -589,   449,   -18,   304,
    -589,  -589,   304,   454,  -589,    73,   -27,   459,  1303,   450,
     781,   436,  -589,   448,   -18,   444,   452,   470,  -589,  -589,
    1303,  1233,   446,  1303,  1303,  -589,   131,  -589,   304,  -589,
     -18,  -589,   388,   -18,  -589,   454,   162,   462,   188,  1303,
    -589,   145,  -589,   479,   388,   788,   480,  -589,  -589,  -589,
    -589,  -589,  -589,  -589,   835,   304,  -589,   -18,   304,  -589,
     482,   388,   483,  -589,   842,  -589,   304,  -589,   486,  -589
};

  /* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
     Performed when YYTABLE does not specify something else to do.  Zero
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
     363,   364,   244,   281,   274,     0,   136,   136,   136,     0,
     144,   144,   144,   144,     0,   138,     0,     0,     0,     0,
      73,   205,   206,    67,    74,    75,    76,    77,     0,    78,
      66,   208,   207,     7,   239,   231,   232,   233,   234,   235,
     237,   238,   236,   229,    71,   230,    72,    63,   106,     0,
      92,    93,    94,    95,   103,   104,     0,    90,   109,   110,
     121,   122,   123,   127,   245,     0,     0,    64,     0,   275,
     274,     0,     0,     0,   115,   116,   117,   118,   129,     0,
     137,     0,     0,     0,     0,   221,   209,     0,     0,     0,
       0,     0,     0,     0,   151,     0,     0,   211,   223,   210,
       0,     0,   144,   144,   144,   144,     0,   138,   196,   197,
     198,   199,   200,     8,    61,   124,   102,   105,    96,    97,
     100,   101,    88,   108,   111,   112,   113,   125,   126,     0,
       0,     0,   274,   271,   274,     0,   282,     0,     0,   119,
     120,     0,   128,   132,   215,   212,     0,   217,     0,   155,
     156,     0,   146,    90,   166,   166,   166,   166,   150,     0,
       0,   153,     0,     0,     0,     0,     0,   142,   143,     0,
     140,   164,     0,   118,     0,   193,     0,     7,     0,     0,
       0,     0,     0,     0,    98,    99,    84,    85,    86,    89,
       0,    83,    90,    70,    57,     0,   272,     0,     0,   274,
     243,     0,     0,   361,   132,   134,   274,   136,     0,   136,
     136,     0,   136,   222,   145,     0,   107,     0,     0,     0,
       0,     0,     0,   175,     0,   152,   166,   166,   139,     0,
     157,   185,     0,   191,   187,     0,   195,    69,   166,   166,
     166,   166,   166,     0,     0,    91,     0,   274,   271,   274,
     274,   279,   132,     0,   133,     0,   130,     0,     0,     0,
       0,     0,     0,   147,   168,   167,   201,     0,   170,   171,
     172,   173,   174,   154,     0,     0,   141,   158,     0,   157,
       0,     0,   190,   188,   189,   192,   194,     0,     0,     0,
       0,     0,   183,   157,    87,     0,    68,   277,   273,   278,
     276,   135,     0,   362,   131,   216,     0,   213,     0,     0,
     218,     0,   228,     0,     0,     0,     0,     0,   224,   225,
     176,   177,     0,   163,   165,   186,   178,   179,   180,   181,
     182,     0,   305,   283,   274,   300,     0,     0,   136,   136,
     136,   169,   248,     0,     0,   226,     7,   227,   204,   159,
       0,   157,     0,     0,   304,     0,     0,     0,     0,   267,
     251,   252,   253,   254,   260,   261,   262,   255,   256,   257,
     258,   259,   148,   263,     0,   265,   266,     0,   249,    56,
       0,     0,   202,     0,     0,   184,   280,     0,   284,   286,
     301,   114,   214,   220,   219,   149,   264,     0,   247,     0,
       0,     0,   160,   161,   269,   268,   270,   285,     0,   250,
     350,     0,     0,     0,     0,     0,   321,     0,     0,   310,
       0,   274,   241,   339,   311,   308,     0,   356,   274,     0,
     274,     0,   359,     0,     0,   320,     0,   274,     0,     0,
       0,     0,     0,     0,     0,   354,     0,     0,     0,   357,
     274,     0,     0,   323,     0,     0,   274,     0,     0,     0,
       0,     0,   321,     0,     0,   274,     0,   317,   319,     7,
     314,   349,     0,   240,     0,     0,   274,     0,   355,     0,
       0,   360,   322,     0,   338,   316,     0,     0,   274,     0,
     274,     0,     0,   274,     0,     0,   340,   318,   312,   309,
     287,   288,   289,   307,     0,     0,   302,     0,   274,     0,
     274,     0,   347,     0,   324,   337,     0,   351,     0,     0,
       0,     0,   274,     0,     0,   336,     0,     0,     0,   306,
       0,   274,     0,     0,   358,     0,     0,   345,   274,     0,
       0,   326,     0,     0,   327,     0,     0,   274,     0,   303,
       0,     0,   274,   348,   351,     0,   352,     0,   274,     0,
     334,   325,     0,   351,   290,     0,     0,     0,     0,     0,
       0,     0,   346,     0,   274,     0,     0,     0,   332,   298,
       0,     0,     0,     0,     0,   296,     0,   242,     0,   342,
     274,   353,     0,   274,   335,   351,     0,     0,     0,     0,
     292,     0,   299,     0,     0,     0,     0,   333,   295,   294,
     293,   291,   297,   341,     0,     0,   329,   274,     0,   343,
       0,     0,     0,   328,     0,   344,     0,   330,     0,   331
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -589,  -589,   559,  -589,  -249,    -1,   -61,   497,   512,   -49,
    -589,  -589,  -589,  -175,  -589,  -197,  -589,  -152,   -75,   -70,
     -69,   -68,  -171,   417,   447,  -589,   -81,  -589,  -589,  -256,
    -589,  -589,   -76,   380,   260,  -589,   -62,   279,  -589,  -589,
     404,   269,  -589,   144,  -589,  -589,  -325,  -589,   -36,   189,
    -589,  -589,  -589,   -40,  -589,  -589,  -589,  -589,  -589,  -589,
    -589,   267,  -589,   272,   520,  -589,   285,   193,   521,  -589,
    -589,   364,  -589,  -589,  -589,  -589,   200,  -589,   203,  -589,
     133,  -589,  -589,   288,   -82,     6,   -57,  -508,  -589,  -589,
    -523,  -589,  -589,  -380,    20,  -437,  -589,  -589,   107,  -500,
      60,  -495,    99,  -491,  -589,  -395,  -588,  -484,  -522,  -450,
    -589,   111,   135,    97,  -589,  -589
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     3,     4,    68,   194,   233,   137,     5,    59,    69,
      70,    71,   268,   269,   270,   203,   138,   234,   139,   154,
     155,   156,   157,   158,   143,   144,   271,   335,   284,   285,
     101,   102,   161,   176,   249,   250,   168,   231,   476,   241,
     173,   242,   232,   358,   464,   359,   360,   103,   298,   345,
     104,   105,   106,   174,   107,   188,   189,   190,   191,   192,
     362,   313,   255,   256,   394,   109,   348,   395,   396,   111,
     112,   166,   179,   397,   398,   126,   399,    72,   145,   424,
     457,   458,   487,   277,   525,   414,   501,   217,   415,   585,
     645,   628,   586,   416,   587,   376,   555,   523,   502,   519,
     534,   546,   516,   503,   548,   520,   617,   526,   559,   508,
     512,   513,   286,   384,    73,    74
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[] =
{
      53,    54,   151,    79,   159,   140,   141,   142,   317,   253,
      84,   162,   164,   551,   165,   567,   147,   235,   236,   237,
     550,   357,   127,   480,   251,   160,   528,   547,   334,   169,
     170,   171,   563,   537,   403,   565,   296,   435,   149,   148,
     220,   357,    75,   510,   505,   220,   652,   517,   411,   283,
     181,   664,   577,   470,    55,   658,   547,   466,   595,   140,
     141,   142,    76,   150,    80,    81,   207,   605,   215,   524,
     209,   113,   589,   163,   529,   326,   381,   160,   620,   604,
    -162,   623,   218,   547,   221,   504,   222,   687,   568,   221,
     570,   613,   306,   573,   307,   177,   615,   210,   223,   254,
     224,   225,   630,   227,   148,   229,   612,   590,   466,   650,
     467,   287,   208,   338,   641,    57,   341,    58,   533,   535,
     258,   259,   260,   261,   230,   666,   146,   631,   504,   148,
     275,    78,   278,   244,   212,   653,   419,   676,   678,   656,
     213,   681,   657,   214,   253,   152,   262,   167,   651,   374,
     685,   148,   659,   165,   660,   288,   614,   661,   289,   148,
     662,   663,   694,   148,   592,   128,   153,   291,   683,   638,
     292,   240,   597,    78,   484,   485,   535,   462,  -187,   704,
    -187,   196,    78,     1,     2,   197,   684,   312,   130,   131,
     132,   133,   134,   135,   136,   700,   172,   331,   702,   299,
     300,   301,   175,    77,   336,    78,   708,    82,   272,    83,
     682,   337,   660,   339,   340,   661,   342,   180,   662,   663,
     332,   636,   344,   148,   692,   640,   660,   202,   643,   661,
     247,   248,   662,   663,   254,   211,   375,   178,   302,   283,
     264,   265,   240,   660,   688,   377,   661,   379,   380,   662,
     663,   311,   346,   314,   347,   669,   193,   318,   319,   320,
     321,   322,   323,   182,   183,   184,   185,   186,   187,   660,
     354,   355,   661,   690,   204,   662,   663,   402,   205,   388,
     405,   333,   367,   368,   369,   370,   371,   372,   206,   373,
     696,   363,   364,   216,   413,   198,   199,   200,   201,   699,
     578,   226,   228,   243,   245,   490,   257,   207,   273,   707,
     274,   276,   280,   279,   281,   238,   344,   295,   282,   202,
     392,   297,   490,   316,   329,    85,    86,    87,    88,    89,
     290,   294,   432,   303,   305,   304,   413,    96,    97,   308,
     309,    98,   436,   437,   438,   491,   492,   493,   494,   495,
     496,   497,   413,   310,   140,   141,   142,   315,   324,   393,
    -281,   325,   491,   492,   493,   494,   495,   496,   497,   327,
     283,   328,   330,   580,   352,   357,   498,  -281,   361,    83,
    -281,   312,   375,   383,   382,  -281,   386,   385,   387,   389,
     390,   391,   404,   498,   417,  -203,    83,   562,   418,   420,
     490,   429,  -281,   128,   153,   393,   486,   426,   427,   430,
     434,   433,   431,   463,   465,   469,   475,   477,   471,   521,
      78,   472,   473,   460,   474,    -9,   130,   131,   132,   133,
     134,   135,   136,   478,   581,   582,   490,   479,   482,   483,
     491,   492,   493,   494,   495,   496,   497,   488,   560,   507,
     511,   514,   583,   509,   566,   515,   518,   522,   527,   536,
     481,   545,   531,   575,    83,   552,   549,   554,   556,   557,
     558,   498,   564,   584,    83,  -313,   491,   492,   493,   494,
     495,   496,   497,   571,   574,   569,   598,   506,   600,   490,
     545,   603,   576,   579,   594,   599,   607,   490,   588,   609,
     608,   611,   616,   618,   619,   621,   610,   498,   490,   624,
      83,  -315,   625,   670,   632,   602,   648,   545,   673,   633,
     622,   634,   654,   667,   671,   679,   626,   584,   674,   491,
     492,   493,   494,   495,   496,   497,   637,   491,   492,   493,
     494,   495,   496,   497,   665,   647,   675,   689,   491,   492,
     493,   494,   495,   496,   497,   693,   655,   697,   703,   705,
     498,   499,   709,   500,    56,   100,    60,   263,   498,   356,
     219,   532,   672,   353,   343,   468,   195,   490,   246,   498,
     421,   365,    83,   606,   349,   350,   351,   366,   108,   110,
     428,   686,   293,    61,   425,    -5,    -5,    62,    -5,    -5,
      -5,    -5,    -5,    -5,    -5,    -5,    -5,    -5,    -5,   128,
      -5,    -5,   489,   627,    -5,   701,   378,   491,   492,   493,
     494,   495,   496,   497,   644,   646,    78,   461,   629,   649,
     553,   601,   130,   131,   132,   133,   134,   135,   136,   400,
     401,   572,   561,    63,    64,     0,   530,   644,   498,    65,
      66,   591,   406,   407,   408,   409,   410,   593,     0,   644,
     644,    67,   680,   644,     0,     0,     0,    -5,   -62,   422,
       0,  -246,  -246,  -246,     0,  -246,  -246,  -246,   691,  -246,
    -246,  -246,  -246,  -246,     0,     0,     0,  -246,  -246,  -246,
    -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,
    -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,
    -246,  -246,  -246,  -246,  -246,  -246,  -246,   490,  -246,     0,
    -246,  -246,     0,     0,   490,     0,     0,  -246,  -246,  -246,
    -246,  -246,  -246,  -246,  -246,   490,     0,  -246,  -246,  -246,
    -246,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   423,     0,     0,     0,     0,   491,   492,   493,
     494,   495,   496,   497,   491,   492,   493,   494,   495,   496,
     497,   490,     0,     0,     0,   491,   492,   493,   494,   495,
     496,   497,   490,     0,     0,     0,     0,     0,   498,   490,
       0,   596,     0,     0,     0,   498,     0,     0,   635,     0,
       0,     0,     0,     0,     0,     0,   498,     0,     0,   639,
       0,   491,   492,   493,   494,   495,   496,   497,     0,     0,
       0,     0,   491,   492,   493,   494,   495,   496,   497,   491,
     492,   493,   494,   495,   496,   497,   490,     0,     0,     0,
       0,     0,   498,   490,     0,   642,     0,     0,     0,     0,
       0,     0,     0,   498,     0,     0,   668,     0,     0,     0,
     498,     0,     0,   695,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   491,   492,   493,   494,
     495,   496,   497,   491,   492,   493,   494,   495,   496,   497,
       1,     2,     0,    85,    86,    87,    88,    89,    90,    91,
      92,    93,    94,    95,   439,    96,    97,   498,     0,    98,
     698,     0,     0,     0,   498,     0,     0,   706,     0,     0,
       0,     0,     0,     0,   440,     0,   441,   442,   443,   444,
     445,   446,     0,     0,   447,   448,   449,   450,   451,     0,
       0,     0,   114,   115,   116,   117,     0,   118,   119,   120,
     121,   122,   452,   453,     0,   439,     0,     0,     0,     0,
       0,   580,    99,     0,     0,     0,     0,     0,     0,   454,
       0,     0,     0,   455,   456,   440,   123,   441,   442,   443,
     444,   445,   446,     0,     0,   447,   448,   449,   450,   451,
       0,   128,   153,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   452,   453,     0,     0,     0,    78,   124,
       0,     0,   125,     0,   130,   131,   132,   133,   134,   135,
     136,     0,   581,   582,   455,   456,     6,     7,     8,     0,
       9,    10,    11,     0,    12,    13,    14,    15,    16,     0,
       0,     0,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,   128,
     129,    34,    35,    36,    37,    38,    39,    40,    41,    42,
      43,    44,     0,    45,     0,    46,   459,     0,     0,     0,
       0,     0,   130,   131,   132,   133,   134,   135,   136,    48,
       0,     0,    49,    50,    51,    52,     6,     7,     8,     0,
       9,    10,    11,     0,    12,    13,    14,    15,    16,     0,
       0,     0,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,     0,
       0,    34,    35,    36,    37,    38,    39,    40,    41,    42,
      43,    44,   238,    45,     0,    46,    47,   239,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    48,
       0,     0,    49,    50,    51,    52,     6,     7,     8,     0,
       9,    10,    11,     0,    12,    13,    14,    15,    16,     0,
       0,     0,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,     0,
       0,    34,    35,    36,    37,    38,    39,    40,    41,    42,
      43,    44,     0,    45,     0,    46,    47,   239,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    48,
       0,     0,    49,    50,    51,    52,     6,     7,     8,     0,
       9,    10,    11,     0,    12,    13,    14,    15,    16,     0,
       0,     0,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,     0,
       0,    34,    35,    36,    37,    38,    39,    40,    41,    42,
      43,    44,     0,    45,     0,    46,    47,   677,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    48,
       0,     0,    49,    50,    51,    52,     6,     7,     8,     0,
       9,    10,    11,     0,    12,    13,    14,    15,    16,     0,
       0,     0,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,     0,
       0,    34,    35,    36,    37,    38,    39,    40,    41,    42,
      43,    44,   252,    45,     0,    46,    47,   538,   539,   540,
     494,   541,   542,   543,     0,     0,     0,     0,     0,    48,
     128,   153,    49,    50,    51,    52,     0,     0,     0,     0,
       0,     0,     0,   128,   153,     0,     0,    78,   544,     0,
       0,    83,     0,   130,   131,   132,   133,   134,   135,   136,
      78,     0,     0,     0,     0,     0,   130,   131,   132,   133,
     134,   135,   136,   128,   153,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   128,   129,     0,   412,
      78,   266,   267,     0,     0,     0,   130,   131,   132,   133,
     134,   135,   136,    78,   128,   153,     0,     0,     0,   130,
     131,   132,   133,   134,   135,   136,     0,     0,     0,     0,
       0,    78,     0,     0,     0,     0,     0,   130,   131,   132,
     133,   134,   135,   136
};

static const yytype_int16 yycheck[] =
{
       1,     2,    84,    64,    85,    75,    75,    75,   257,   180,
      67,    87,    88,   521,    89,   537,    77,   169,   170,   171,
     520,    17,    71,   460,   176,    36,   510,   518,   284,    91,
      92,    93,   532,   517,   359,   535,   233,   417,    56,    74,
      37,    17,    40,   493,   481,    37,   634,   497,   373,    84,
      99,    78,   547,   433,     0,   643,   547,    84,   566,   129,
     129,   129,    63,    81,    65,    66,    37,   575,   150,   506,
     145,    76,   556,    84,   511,   272,   332,    36,   600,   574,
      76,   603,   152,   574,    81,   480,    83,   675,   538,    81,
     540,   591,   244,   543,   246,    96,   596,   146,   159,   180,
     161,   162,   610,   164,    74,    30,   590,   557,    84,   631,
      86,    81,    83,   288,   622,    73,   291,    75,   513,   514,
     182,   183,   184,   185,    49,   648,    75,   611,   523,    74,
     212,    53,   214,   173,    75,   635,    81,   660,   661,   639,
      81,   664,   642,    84,   315,    16,   186,    84,   632,   324,
     672,    74,    79,   228,    81,    78,   593,    84,    81,    74,
      87,    88,   684,    74,   559,    36,    37,    78,   668,   619,
      81,   172,   567,    53,    54,    55,   571,   426,    77,   701,
      79,    60,    53,     3,     4,    64,   670,    86,    59,    60,
      61,    62,    63,    64,    65,   695,    84,   279,   698,   235,
     236,   237,    84,    51,   286,    53,   706,    73,   209,    75,
      79,   287,    81,   289,   290,    84,   292,    78,    87,    88,
     281,   616,   297,    74,    79,   620,    81,    78,   623,    84,
      38,    39,    87,    88,   315,    74,    81,    55,   239,    84,
      59,    60,   243,    81,    82,   327,    84,   329,   330,    87,
      88,   252,    73,   254,    75,   650,    73,   258,   259,   260,
     261,   262,   263,    11,    12,    13,    14,    15,    16,    81,
     306,   307,    84,    85,    80,    87,    88,   358,    80,   340,
     361,   282,   318,   319,   320,   321,   322,    73,    80,    75,
     685,    54,    55,    76,   375,    59,    60,    61,    62,   694,
     549,    66,    77,    84,    84,     1,    76,    37,    76,   704,
      53,    56,    73,    82,    80,    49,   391,    77,    80,    78,
       1,    74,     1,    79,    82,     6,     7,     8,     9,    10,
      85,    85,   414,    85,    85,    77,   417,    18,    19,    85,
      77,    22,   418,   419,   420,    41,    42,    43,    44,    45,
      46,    47,   433,    75,   424,   424,   424,    77,    77,    40,
      56,    79,    41,    42,    43,    44,    45,    46,    47,    76,
      84,    77,    85,     6,    85,    17,    72,    56,    86,    75,
      76,    86,    81,    85,    82,    81,    79,    82,    82,    79,
      82,    77,    76,    72,    81,    76,    75,    76,    81,    81,
       1,   402,    81,    36,    37,    40,   467,    76,    73,    76,
      82,    77,   413,    84,    73,    85,    55,    85,    82,   501,
      53,    82,    82,   424,    82,    81,    59,    60,    61,    62,
      63,    64,    65,    85,    67,    68,     1,    77,    85,    73,
      41,    42,    43,    44,    45,    46,    47,    36,   530,    81,
      53,    75,    85,    84,   536,    55,    75,    20,    53,   516,
     461,   518,    77,   545,    75,    86,    76,    84,    73,    85,
      82,    72,    76,   554,    75,    76,    41,    42,    43,    44,
      45,    46,    47,    75,    75,    84,   568,   488,   570,     1,
     547,   573,    76,    54,    76,    53,    74,     1,   555,    85,
      77,    73,    48,    73,    85,    76,   588,    72,     1,    76,
      75,    76,    78,    77,    74,   572,    78,   574,    74,    76,
     602,    76,    73,    73,    76,    79,   607,   608,    76,    41,
      42,    43,    44,    45,    46,    47,   618,    41,    42,    43,
      44,    45,    46,    47,    85,   627,    76,    85,    41,    42,
      43,    44,    45,    46,    47,    76,   638,    77,    76,    76,
      72,    73,    76,    75,     5,    68,    54,   187,    72,   309,
     153,    75,   654,   304,   295,   431,   129,     1,   174,    72,
     391,   314,    75,   584,   299,   300,   301,   315,    68,    68,
     397,   673,   228,     1,   394,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    14,    15,    16,    36,
      18,    19,   479,   607,    22,   697,   328,    41,    42,    43,
      44,    45,    46,    47,   625,   626,    53,   424,   608,   630,
     523,   571,    59,    60,    61,    62,    63,    64,    65,   354,
     355,   542,   531,    51,    52,    -1,   511,   648,    72,    57,
      58,    75,   367,   368,   369,   370,   371,   560,    -1,   660,
     661,    69,   663,   664,    -1,    -1,    -1,    75,    76,     1,
      -1,     3,     4,     5,    -1,     7,     8,     9,   679,    11,
      12,    13,    14,    15,    -1,    -1,    -1,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    34,    35,    36,    37,    38,    39,    40,    41,
      42,    43,    44,    45,    46,    47,    48,     1,    50,    -1,
      52,    53,    -1,    -1,     1,    -1,    -1,    59,    60,    61,
      62,    63,    64,    65,    66,     1,    -1,    69,    70,    71,
      72,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    84,    -1,    -1,    -1,    -1,    41,    42,    43,
      44,    45,    46,    47,    41,    42,    43,    44,    45,    46,
      47,     1,    -1,    -1,    -1,    41,    42,    43,    44,    45,
      46,    47,     1,    -1,    -1,    -1,    -1,    -1,    72,     1,
      -1,    75,    -1,    -1,    -1,    72,    -1,    -1,    75,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    72,    -1,    -1,    75,
      -1,    41,    42,    43,    44,    45,    46,    47,    -1,    -1,
      -1,    -1,    41,    42,    43,    44,    45,    46,    47,    41,
      42,    43,    44,    45,    46,    47,     1,    -1,    -1,    -1,
      -1,    -1,    72,     1,    -1,    75,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    72,    -1,    -1,    75,    -1,    -1,    -1,
      72,    -1,    -1,    75,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    41,    42,    43,    44,
      45,    46,    47,    41,    42,    43,    44,    45,    46,    47,
       3,     4,    -1,     6,     7,     8,     9,    10,    11,    12,
      13,    14,    15,    16,     1,    18,    19,    72,    -1,    22,
      75,    -1,    -1,    -1,    72,    -1,    -1,    75,    -1,    -1,
      -1,    -1,    -1,    -1,    21,    -1,    23,    24,    25,    26,
      27,    28,    -1,    -1,    31,    32,    33,    34,    35,    -1,
      -1,    -1,     6,     7,     8,     9,    -1,    11,    12,    13,
      14,    15,    49,    50,    -1,     1,    -1,    -1,    -1,    -1,
      -1,     6,    75,    -1,    -1,    -1,    -1,    -1,    -1,    66,
      -1,    -1,    -1,    70,    71,    21,    40,    23,    24,    25,
      26,    27,    28,    -1,    -1,    31,    32,    33,    34,    35,
      -1,    36,    37,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    49,    50,    -1,    -1,    -1,    53,    73,
      -1,    -1,    76,    -1,    59,    60,    61,    62,    63,    64,
      65,    -1,    67,    68,    70,    71,     3,     4,     5,    -1,
       7,     8,     9,    -1,    11,    12,    13,    14,    15,    -1,
      -1,    -1,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    36,
      37,    38,    39,    40,    41,    42,    43,    44,    45,    46,
      47,    48,    -1,    50,    -1,    52,    53,    -1,    -1,    -1,
      -1,    -1,    59,    60,    61,    62,    63,    64,    65,    66,
      -1,    -1,    69,    70,    71,    72,     3,     4,     5,    -1,
       7,     8,     9,    -1,    11,    12,    13,    14,    15,    -1,
      -1,    -1,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    -1,
      -1,    38,    39,    40,    41,    42,    43,    44,    45,    46,
      47,    48,    49,    50,    -1,    52,    53,    54,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    66,
      -1,    -1,    69,    70,    71,    72,     3,     4,     5,    -1,
       7,     8,     9,    -1,    11,    12,    13,    14,    15,    -1,
      -1,    -1,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    -1,
      -1,    38,    39,    40,    41,    42,    43,    44,    45,    46,
      47,    48,    -1,    50,    -1,    52,    53,    54,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    66,
      -1,    -1,    69,    70,    71,    72,     3,     4,     5,    -1,
       7,     8,     9,    -1,    11,    12,    13,    14,    15,    -1,
      -1,    -1,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    -1,
      -1,    38,    39,    40,    41,    42,    43,    44,    45,    46,
      47,    48,    -1,    50,    -1,    52,    53,    54,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    66,
      -1,    -1,    69,    70,    71,    72,     3,     4,     5,    -1,
       7,     8,     9,    -1,    11,    12,    13,    14,    15,    -1,
      -1,    -1,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    -1,
      -1,    38,    39,    40,    41,    42,    43,    44,    45,    46,
      47,    48,    18,    50,    -1,    52,    53,    41,    42,    43,
      44,    45,    46,    47,    -1,    -1,    -1,    -1,    -1,    66,
      36,    37,    69,    70,    71,    72,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    36,    37,    -1,    -1,    53,    72,    -1,
      -1,    75,    -1,    59,    60,    61,    62,    63,    64,    65,
      53,    -1,    -1,    -1,    -1,    -1,    59,    60,    61,    62,
      63,    64,    65,    36,    37,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    36,    37,    -1,    82,
      53,    54,    55,    -1,    -1,    -1,    59,    60,    61,    62,
      63,    64,    65,    53,    36,    37,    -1,    -1,    -1,    59,
      60,    61,    62,    63,    64,    65,    -1,    -1,    -1,    -1,
      -1,    53,    -1,    -1,    -1,    -1,    -1,    59,    60,    61,
      62,    63,    64,    65
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
     135,    86,   149,    54,    55,   150,   152,   137,   137,   137,
     137,   137,    73,    75,   102,    81,   184,   173,   172,   173,
     173,   118,    82,    85,   202,    82,    79,    82,    95,    79,
      82,    77,     1,    40,   153,   156,   157,   162,   163,   165,
     155,   155,   115,   135,    76,   115,   155,   155,   155,   155,
     155,   135,    82,   115,   174,   177,   182,    81,    81,    81,
      81,   138,     1,    84,   168,   165,    76,    73,   156,    94,
      76,    94,   173,    77,    82,   182,   121,   121,   121,     1,
      21,    23,    24,    25,    26,    27,    28,    31,    32,    33,
      34,    35,    49,    50,    66,    70,    71,   169,   170,    53,
      94,   167,    93,    84,   133,    73,    84,    86,   132,    85,
     182,    82,    82,    82,    82,    55,   127,    85,    85,    77,
     184,    94,    85,    73,    54,    55,    95,   171,    36,   169,
       1,    41,    42,    43,    44,    45,    46,    47,    72,    73,
      75,   175,   187,   192,   194,   184,    94,    81,   198,    84,
     198,    53,   199,   200,    75,    55,   191,   198,    75,   188,
     194,   173,    20,   186,   184,   173,   196,    53,   196,   184,
     201,    77,    75,   194,   189,   194,   175,   196,    41,    42,
      43,    45,    46,    47,    72,   175,   190,   192,   193,    76,
     188,   176,    86,   187,    84,   185,    73,    85,    82,   197,
     173,   200,    76,   188,    76,   188,   173,   197,   198,    84,
     198,    75,   191,   198,    75,   173,    76,   190,    93,    54,
       6,    67,    68,    85,   115,   178,   181,   183,   175,   196,
     198,    75,   194,   202,    76,   176,    75,   194,   173,    53,
     173,   189,   175,   173,   190,   176,    94,    74,    77,    85,
     173,    73,   196,   188,   184,   188,    48,   195,    73,    85,
     197,    76,   173,   197,    76,    78,   115,   174,   180,   183,
     176,   196,    74,    76,    76,    75,   194,   173,   198,    75,
     194,   176,    75,   194,    94,   179,    94,   173,    78,    94,
     197,   196,   195,   188,    73,   173,   188,   188,   195,    79,
      81,    84,    87,    88,    78,    85,   179,    73,    75,   194,
      77,    76,   173,    74,    76,    76,   179,    54,   179,    79,
      94,   179,    79,   188,   196,   197,   173,   195,    82,    85,
      85,    94,    79,    76,   197,    75,   194,    77,    75,   194,
     188,   173,   188,    76,   197,    76,    75,   194,   188,    76
};

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
     151,   151,   151,   152,   152,   153,   154,   154,   154,   154,
     154,   155,   155,   156,   156,   157,   157,   157,   157,   157,
     157,   157,   158,   158,   158,   158,   158,   159,   159,   159,
     159,   160,   160,   161,   162,   163,   163,   163,   163,   164,
     164,   164,   164,   164,   164,   164,   164,   164,   164,   164,
     165,   165,   165,   166,   166,   167,   168,   168,   168,   169,
     169,   170,   170,   170,   170,   170,   170,   170,   170,   170,
     170,   170,   170,   170,   170,   170,   170,   170,   171,   171,
     171,   172,   172,   172,   173,   173,   173,   173,   173,   173,
     174,   175,   176,   177,   177,   177,   177,   178,   178,   178,
     179,   179,   179,   179,   179,   179,   180,   181,   181,   181,
     182,   182,   183,   183,   184,   184,   185,   185,   186,   186,
     187,   187,   187,   188,   188,   189,   189,   190,   190,   190,
     191,   191,   192,   192,   192,   193,   193,   193,   193,   193,
     193,   193,   193,   193,   193,   193,   193,   194,   194,   194,
     194,   194,   194,   194,   194,   194,   194,   194,   194,   194,
     194,   195,   195,   195,   196,   197,   198,   199,   199,   200,
     200,   201,   202,   203,   204
};

  /* YYR2[YYN] -- Number of symbols on the right hand side of rule YYN.  */
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
       3,     2,     3,     1,     3,     4,     2,     2,     2,     2,
       2,     1,     4,     0,     2,     1,     1,     1,     1,     2,
       2,     2,     3,     6,     9,     3,     6,     3,     6,     9,
       9,     1,     3,     1,     1,     1,     2,     2,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       7,     5,    13,     5,     2,     1,     0,     3,     1,     1,
       3,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     2,     1,     1,     1,     1,     1,
       1,     0,     1,     3,     0,     1,     5,     5,     5,     4,
       3,     1,     1,     1,     3,     4,     3,     1,     1,     1,
       1,     4,     3,     4,     4,     4,     3,     7,     5,     6,
       1,     3,     1,     3,     3,     2,     3,     2,     0,     3,
       1,     1,     4,     1,     2,     1,     2,     1,     2,     1,
       1,     0,     4,     3,     5,     5,     4,     4,    11,     9,
      12,    14,     6,     8,     5,     7,     3,     5,     4,     1,
       4,    11,     9,    12,    14,     6,     8,     5,     7,     3,
       1,     0,     2,     4,     1,     1,     1,     2,     5,     1,
       3,     1,     1,     2,     2
};


#define yyerrok         (yyerrstatus = 0)
#define yyclearin       (yychar = YYEMPTY)
#define YYEMPTY         (-2)
#define YYEOF           0

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab


#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)                                  \
do                                                              \
  if (yychar == YYEMPTY)                                        \
    {                                                           \
      yychar = (Token);                                         \
      yylval = (Value);                                         \
      YYPOPSTACK (yylen);                                       \
      yystate = *yyssp;                                         \
      goto yybackup;                                            \
    }                                                           \
  else                                                          \
    {                                                           \
      yyerror (YY_("syntax error: cannot back up")); \
      YYERROR;                                                  \
    }                                                           \
while (0)

/* Error token number */
#define YYTERROR        1
#define YYERRCODE       256


/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)                                \
    do                                                                  \
      if (N)                                                            \
        {                                                               \
          (Current).first_line   = YYRHSLOC (Rhs, 1).first_line;        \
          (Current).first_column = YYRHSLOC (Rhs, 1).first_column;      \
          (Current).last_line    = YYRHSLOC (Rhs, N).last_line;         \
          (Current).last_column  = YYRHSLOC (Rhs, N).last_column;       \
        }                                                               \
      else                                                              \
        {                                                               \
          (Current).first_line   = (Current).last_line   =              \
            YYRHSLOC (Rhs, 0).last_line;                                \
          (Current).first_column = (Current).last_column =              \
            YYRHSLOC (Rhs, 0).last_column;                              \
        }                                                               \
    while (0)
#endif

#define YYRHSLOC(Rhs, K) ((Rhs)[K])


/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)                        \
do {                                            \
  if (yydebug)                                  \
    YYFPRINTF Args;                             \
} while (0)


/* YY_LOCATION_PRINT -- Print the location on the stream.
   This macro was not mandated originally: define only if we know
   we won't break user code: when these are the locations we know.  */

#ifndef YY_LOCATION_PRINT
# if defined YYLTYPE_IS_TRIVIAL && YYLTYPE_IS_TRIVIAL

/* Print *YYLOCP on YYO.  Private, do not rely on its existence. */

YY_ATTRIBUTE_UNUSED
static unsigned
yy_location_print_ (FILE *yyo, YYLTYPE const * const yylocp)
{
  unsigned res = 0;
  int end_col = 0 != yylocp->last_column ? yylocp->last_column - 1 : 0;
  if (0 <= yylocp->first_line)
    {
      res += YYFPRINTF (yyo, "%d", yylocp->first_line);
      if (0 <= yylocp->first_column)
        res += YYFPRINTF (yyo, ".%d", yylocp->first_column);
    }
  if (0 <= yylocp->last_line)
    {
      if (yylocp->first_line < yylocp->last_line)
        {
          res += YYFPRINTF (yyo, "-%d", yylocp->last_line);
          if (0 <= end_col)
            res += YYFPRINTF (yyo, ".%d", end_col);
        }
      else if (0 <= end_col && yylocp->first_column < end_col)
        res += YYFPRINTF (yyo, "-%d", end_col);
    }
  return res;
 }

#  define YY_LOCATION_PRINT(File, Loc)          \
  yy_location_print_ (File, &(Loc))

# else
#  define YY_LOCATION_PRINT(File, Loc) ((void) 0)
# endif
#endif


# define YY_SYMBOL_PRINT(Title, Type, Value, Location)                    \
do {                                                                      \
  if (yydebug)                                                            \
    {                                                                     \
      YYFPRINTF (stderr, "%s ", Title);                                   \
      yy_symbol_print (stderr,                                            \
                  Type, Value, Location); \
      YYFPRINTF (stderr, "\n");                                           \
    }                                                                     \
} while (0)


/*----------------------------------------.
| Print this symbol's value on YYOUTPUT.  |
`----------------------------------------*/

static void
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep, YYLTYPE const * const yylocationp)
{
  FILE *yyo = yyoutput;
  YYUSE (yyo);
  YYUSE (yylocationp);
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# endif
  YYUSE (yytype);
}


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

static void
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep, YYLTYPE const * const yylocationp)
{
  YYFPRINTF (yyoutput, "%s %s (",
             yytype < YYNTOKENS ? "token" : "nterm", yytname[yytype]);

  YY_LOCATION_PRINT (yyoutput, *yylocationp);
  YYFPRINTF (yyoutput, ": ");
  yy_symbol_value_print (yyoutput, yytype, yyvaluep, yylocationp);
  YYFPRINTF (yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

static void
yy_stack_print (yytype_int16 *yybottom, yytype_int16 *yytop)
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)                            \
do {                                                            \
  if (yydebug)                                                  \
    yy_stack_print ((Bottom), (Top));                           \
} while (0)


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

static void
yy_reduce_print (yytype_int16 *yyssp, YYSTYPE *yyvsp, YYLTYPE *yylsp, int yyrule)
{
  unsigned long int yylno = yyrline[yyrule];
  int yynrhs = yyr2[yyrule];
  int yyi;
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
             yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr,
                       yystos[yyssp[yyi + 1 - yynrhs]],
                       &(yyvsp[(yyi + 1) - (yynrhs)])
                       , &(yylsp[(yyi + 1) - (yynrhs)])                       );
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)          \
do {                                    \
  if (yydebug)                          \
    yy_reduce_print (yyssp, yyvsp, yylsp, Rule); \
} while (0)

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
#ifndef YYINITDEPTH
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
static YYSIZE_T
yystrlen (const char *yystr)
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
static char *
yystpcpy (char *yydest, const char *yysrc)
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
  YYSIZE_T yysize0 = yytnamerr (YY_NULLPTR, yytname[yytoken]);
  YYSIZE_T yysize = yysize0;
  enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
  /* Internationalized format string. */
  const char *yyformat = YY_NULLPTR;
  /* Arguments of yyformat. */
  char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
  /* Number of reported tokens (one for the "unexpected", one per
     "expected"). */
  int yycount = 0;

  /* There are many possibilities here to consider:
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
                {
                  YYSIZE_T yysize1 = yysize + yytnamerr (YY_NULLPTR, yytname[yyx]);
                  if (! (yysize <= yysize1
                         && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
                    return 2;
                  yysize = yysize1;
                }
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

  {
    YYSIZE_T yysize1 = yysize + yystrlen (yyformat);
    if (! (yysize <= yysize1 && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
      return 2;
    yysize = yysize1;
  }

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

static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep, YYLTYPE *yylocationp)
{
  YYUSE (yyvaluep);
  YYUSE (yylocationp);
  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YYUSE (yytype);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}




/* The lookahead symbol.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;
/* Location data for the lookahead symbol.  */
YYLTYPE yylloc
# if defined YYLTYPE_IS_TRIVIAL && YYLTYPE_IS_TRIVIAL
  = { 1, 1, 1, 1 }
# endif
;
/* Number of syntax errors so far.  */
int yynerrs;


/*----------.
| yyparse.  |
`----------*/

int
yyparse (void)
{
    int yystate;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus;

    /* The stacks and their tools:
       'yyss': related to states.
       'yyvs': related to semantic values.
       'yyls': related to locations.

       Refer to the stacks through separate pointers, to allow yyoverflow
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
  int yytoken = 0;
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

  yyssp = yyss = yyssa;
  yyvsp = yyvs = yyvsa;
  yylsp = yyls = yylsa;
  yystacksize = YYINITDEPTH;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY; /* Cause a token to be read.  */
  yylsp[0] = yylloc;
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
      yychar = yylex ();
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
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END
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
     '$$ = $1'.

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
#line 192 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.modlist) = (yyvsp[0].modlist); modlist = (yyvsp[0].modlist); }
#line 2145 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 3:
#line 196 "xi-grammar.y" /* yacc.c:1646  */
    { 
		  (yyval.modlist) = 0; 
		}
#line 2153 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 4:
#line 200 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.modlist) = new AstChildren<Module>(lineno, (yyvsp[-1].module), (yyvsp[0].modlist)); }
#line 2159 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 5:
#line 204 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 2165 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 6:
#line 206 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 2171 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 7:
#line 210 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 2177 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 8:
#line 212 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 2183 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 9:
#line 217 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 2189 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 10:
#line 218 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MODULE); YYABORT; }
#line 2195 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 11:
#line 219 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MAINMODULE); YYABORT; }
#line 2201 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 12:
#line 220 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(EXTERN); YYABORT; }
#line 2207 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 13:
#line 222 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITCALL); YYABORT; }
#line 2213 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 14:
#line 223 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITNODE); YYABORT; }
#line 2219 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 15:
#line 224 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITPROC); YYABORT; }
#line 2225 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 16:
#line 226 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CHARE); }
#line 2231 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 17:
#line 227 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MAINCHARE); }
#line 2237 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 18:
#line 228 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(GROUP); }
#line 2243 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 19:
#line 229 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NODEGROUP); }
#line 2249 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 20:
#line 230 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ARRAY); }
#line 2255 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 21:
#line 234 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INCLUDE); YYABORT; }
#line 2261 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 22:
#line 235 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(STACKSIZE); YYABORT; }
#line 2267 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 23:
#line 236 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(THREADED); YYABORT; }
#line 2273 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 24:
#line 237 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(TEMPLATE); YYABORT; }
#line 2279 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 25:
#line 238 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(SYNC); YYABORT; }
#line 2285 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 26:
#line 239 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IGET); YYABORT; }
#line 2291 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 27:
#line 240 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(EXCLUSIVE); YYABORT; }
#line 2297 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 28:
#line 241 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IMMEDIATE); YYABORT; }
#line 2303 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 29:
#line 242 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(SKIPSCHED); YYABORT; }
#line 2309 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 30:
#line 243 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INLINE); YYABORT; }
#line 2315 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 31:
#line 244 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(VIRTUAL); YYABORT; }
#line 2321 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 32:
#line 245 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MIGRATABLE); YYABORT; }
#line 2327 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 33:
#line 246 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CREATEHERE); YYABORT; }
#line 2333 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 34:
#line 247 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CREATEHOME); YYABORT; }
#line 2339 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 35:
#line 248 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOKEEP); YYABORT; }
#line 2345 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 36:
#line 249 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOTRACE); YYABORT; }
#line 2351 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 37:
#line 250 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(APPWORK); YYABORT; }
#line 2357 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 38:
#line 253 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(PACKED); YYABORT; }
#line 2363 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 39:
#line 254 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(VARSIZE); YYABORT; }
#line 2369 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 40:
#line 255 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ENTRY); YYABORT; }
#line 2375 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 41:
#line 256 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(FOR); YYABORT; }
#line 2381 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 42:
#line 257 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(FORALL); YYABORT; }
#line 2387 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 43:
#line 258 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(WHILE); YYABORT; }
#line 2393 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 44:
#line 259 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(WHEN); YYABORT; }
#line 2399 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 45:
#line 260 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(OVERLAP); YYABORT; }
#line 2405 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 46:
#line 261 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ATOMIC); YYABORT; }
#line 2411 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 47:
#line 262 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IF); YYABORT; }
#line 2417 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 48:
#line 263 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ELSE); YYABORT; }
#line 2423 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 49:
#line 265 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(LOCAL); YYABORT; }
#line 2429 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 50:
#line 267 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(USING); YYABORT; }
#line 2435 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 51:
#line 268 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ACCEL); YYABORT; }
#line 2441 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 52:
#line 271 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ACCELBLOCK); YYABORT; }
#line 2447 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 53:
#line 272 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MEMCRITICAL); YYABORT; }
#line 2453 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 54:
#line 273 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(REDUCTIONTARGET); YYABORT; }
#line 2459 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 55:
#line 274 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CASE); YYABORT; }
#line 2465 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 56:
#line 278 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 2471 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 57:
#line 280 "xi-grammar.y" /* yacc.c:1646  */
    {
		  char *tmp = new char[strlen((yyvsp[-3].strval))+strlen((yyvsp[0].strval))+3];
		  sprintf(tmp,"%s::%s", (yyvsp[-3].strval), (yyvsp[0].strval));
		  (yyval.strval) = tmp;
		}
#line 2481 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 58:
#line 288 "xi-grammar.y" /* yacc.c:1646  */
    { 
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist)); 
		}
#line 2489 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 59:
#line 292 "xi-grammar.y" /* yacc.c:1646  */
    {  
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist)); 
		    (yyval.module)->setMain();
		}
#line 2498 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 60:
#line 299 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = 0; }
#line 2504 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 61:
#line 301 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = (yyvsp[-2].conslist); }
#line 2510 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 62:
#line 305 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = 0; }
#line 2516 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 63:
#line 307 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = new ConstructList(lineno, (yyvsp[-1].construct), (yyvsp[0].conslist)); }
#line 2522 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 64:
#line 311 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new UsingScope((yyvsp[0].strval), false); }
#line 2528 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 65:
#line 313 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new UsingScope((yyvsp[0].strval), true); }
#line 2534 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 66:
#line 315 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].member)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].member); }
#line 2540 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 67:
#line 317 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].message)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].message); }
#line 2546 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 68:
#line 319 "xi-grammar.y" /* yacc.c:1646  */
    {
                  Entry *e = new Entry(lineno, 0, (yyvsp[-4].type), (yyvsp[-2].strval), (yyvsp[0].plist), 0, 0, 0, (yylsp[-6]).first_line, (yyloc).last_line);
                  int isExtern = 1;
                  e->setExtern(isExtern);
                  e->targs = (yyvsp[-1].tparlist);
                  e->label = new XStr;
                  (yyvsp[-3].ntype)->print(*e->label);
                  (yyval.construct) = e;
                }
#line 2560 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 69:
#line 331 "xi-grammar.y" /* yacc.c:1646  */
    { if((yyvsp[-2].conslist)) (yyvsp[-2].conslist)->recurse<int&>((yyvsp[-4].intval), &Construct::setExtern); (yyval.construct) = (yyvsp[-2].conslist); }
#line 2566 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 70:
#line 333 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new Scope((yyvsp[-3].strval), (yyvsp[-1].conslist)); }
#line 2572 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 71:
#line 335 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = (yyvsp[-1].construct); }
#line 2578 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 72:
#line 337 "xi-grammar.y" /* yacc.c:1646  */
    {
          ERROR("preceding construct must be semicolon terminated",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 2588 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 73:
#line 343 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].module)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].module); }
#line 2594 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 74:
#line 345 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2600 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 75:
#line 347 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2606 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 76:
#line 349 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2612 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 77:
#line 351 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2618 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 78:
#line 353 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].templat)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].templat); }
#line 2624 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 79:
#line 355 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = NULL; }
#line 2630 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 80:
#line 357 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = NULL; }
#line 2636 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 81:
#line 359 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = (yyvsp[0].accelBlock); }
#line 2642 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 82:
#line 361 "xi-grammar.y" /* yacc.c:1646  */
    {
          ERROR("invalid construct",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 2652 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 83:
#line 369 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamType((yyvsp[0].type)); }
#line 2658 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 84:
#line 371 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
#line 2664 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 85:
#line 373 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
#line 2670 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 86:
#line 377 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = new TParamList((yyvsp[0].tparam)); }
#line 2676 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 87:
#line 379 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = new TParamList((yyvsp[-2].tparam), (yyvsp[0].tparlist)); }
#line 2682 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 88:
#line 383 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = 0; }
#line 2688 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 89:
#line 385 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = (yyvsp[0].tparlist); }
#line 2694 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 90:
#line 389 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = 0; }
#line 2700 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 91:
#line 391 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = (yyvsp[-1].tparlist); }
#line 2706 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 92:
#line 395 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("int"); }
#line 2712 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 93:
#line 397 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long"); }
#line 2718 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 94:
#line 399 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("short"); }
#line 2724 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 95:
#line 401 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("char"); }
#line 2730 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 96:
#line 403 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned int"); }
#line 2736 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 97:
#line 405 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long"); }
#line 2742 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 98:
#line 407 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long"); }
#line 2748 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 99:
#line 409 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long long"); }
#line 2754 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 100:
#line 411 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned short"); }
#line 2760 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 101:
#line 413 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned char"); }
#line 2766 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 102:
#line 415 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long long"); }
#line 2772 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 103:
#line 417 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("float"); }
#line 2778 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 104:
#line 419 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("double"); }
#line 2784 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 105:
#line 421 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long double"); }
#line 2790 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 106:
#line 423 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("void"); }
#line 2796 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 107:
#line 426 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ntype) = new NamedType((yyvsp[-1].strval),(yyvsp[0].tparlist)); }
#line 2802 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 108:
#line 427 "xi-grammar.y" /* yacc.c:1646  */
    { 
                    const char* basename, *scope;
                    splitScopedName((yyvsp[-1].strval), &scope, &basename);
                    (yyval.ntype) = new NamedType(basename, (yyvsp[0].tparlist), scope);
                }
#line 2812 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 109:
#line 435 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 2818 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 110:
#line 437 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ntype); }
#line 2824 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 111:
#line 441 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ptype) = new PtrType((yyvsp[-1].type)); }
#line 2830 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 112:
#line 445 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
#line 2836 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 113:
#line 447 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
#line 2842 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 114:
#line 451 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ftype) = new FuncType((yyvsp[-7].type), (yyvsp[-4].strval), (yyvsp[-1].plist)); }
#line 2848 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 115:
#line 455 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 2854 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 116:
#line 457 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 2860 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 117:
#line 459 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 2866 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 118:
#line 461 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ftype); }
#line 2872 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 119:
#line 463 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[0].type)); }
#line 2878 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 120:
#line 465 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[-1].type)); }
#line 2884 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 121:
#line 469 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 2890 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 122:
#line 471 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 2896 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 123:
#line 473 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 2902 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 124:
#line 475 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[0].type)); }
#line 2908 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 125:
#line 477 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[-1].type)); }
#line 2914 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 126:
#line 481 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ReferenceType((yyvsp[-1].type)); }
#line 2920 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 127:
#line 483 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 2926 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 128:
#line 487 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ReferenceType((yyvsp[-1].type)); }
#line 2932 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 129:
#line 489 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 2938 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 130:
#line 493 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 2944 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 131:
#line 497 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = (yyvsp[-1].val); }
#line 2950 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 132:
#line 501 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.vallist) = 0; }
#line 2956 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 133:
#line 503 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.vallist) = new ValueList((yyvsp[-1].val), (yyvsp[0].vallist)); }
#line 2962 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 134:
#line 507 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].vallist)); }
#line 2968 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 135:
#line 511 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[-3].type), (yyvsp[-1].strval), (yyvsp[0].vallist), 1); }
#line 2974 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 136:
#line 515 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0;}
#line 2980 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 137:
#line 517 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0;}
#line 2986 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 138:
#line 521 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 2992 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 139:
#line 523 "xi-grammar.y" /* yacc.c:1646  */
    { 
		  /*
		  printf("Warning: Message attributes are being phased out.\n");
		  printf("Warning: Please remove them from interface files.\n");
		  */
		  (yyval.intval) = (yyvsp[-1].intval); 
		}
#line 3004 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 140:
#line 533 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[0].intval); }
#line 3010 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 141:
#line 535 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[-2].intval) | (yyvsp[0].intval); }
#line 3016 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 142:
#line 539 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3022 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 143:
#line 541 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3028 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 144:
#line 545 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = 0; }
#line 3034 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 145:
#line 547 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-1].cattr); }
#line 3040 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 146:
#line 551 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[0].cattr); }
#line 3046 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 147:
#line 553 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
#line 3052 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 148:
#line 557 "xi-grammar.y" /* yacc.c:1646  */
    { python_doc = NULL; (yyval.intval) = 0; }
#line 3058 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 149:
#line 559 "xi-grammar.y" /* yacc.c:1646  */
    { python_doc = (yyvsp[0].strval); (yyval.intval) = 0; }
#line 3064 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 150:
#line 563 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CPYTHON; }
#line 3070 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 151:
#line 567 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = 0; }
#line 3076 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 152:
#line 569 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-1].cattr); }
#line 3082 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 153:
#line 573 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[0].cattr); }
#line 3088 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 154:
#line 575 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
#line 3094 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 155:
#line 579 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CMIGRATABLE; }
#line 3100 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 156:
#line 581 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CPYTHON; }
#line 3106 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 157:
#line 585 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3112 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 158:
#line 587 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 3118 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 159:
#line 590 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3124 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 160:
#line 592 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 3130 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 161:
#line 595 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mv) = new MsgVar((yyvsp[-3].type), (yyvsp[-2].strval), (yyvsp[-4].intval), (yyvsp[-1].intval)); }
#line 3136 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 162:
#line 599 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mvlist) = new MsgVarList((yyvsp[0].mv)); }
#line 3142 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 163:
#line 601 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mvlist) = new MsgVarList((yyvsp[-1].mv), (yyvsp[0].mvlist)); }
#line 3148 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 164:
#line 605 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, (yyvsp[0].ntype)); }
#line 3154 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 165:
#line 607 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, (yyvsp[-3].ntype), (yyvsp[-1].mvlist)); }
#line 3160 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 166:
#line 611 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = 0; }
#line 3166 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 167:
#line 613 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = (yyvsp[0].typelist); }
#line 3172 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 168:
#line 617 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = new TypeList((yyvsp[0].ntype)); }
#line 3178 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 169:
#line 619 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = new TypeList((yyvsp[-2].ntype), (yyvsp[0].typelist)); }
#line 3184 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 170:
#line 623 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr)|Chare::CCHARE, (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3190 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 171:
#line 625 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3196 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 172:
#line 629 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3202 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 173:
#line 633 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new NodeGroup(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3208 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 174:
#line 637 "xi-grammar.y" /* yacc.c:1646  */
    {/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",(yyvsp[-2].strval));
			(yyval.ntype) = new NamedType(buf); 
		}
#line 3218 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 175:
#line 643 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ntype) = new NamedType((yyvsp[-1].strval)); }
#line 3224 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 176:
#line 647 "xi-grammar.y" /* yacc.c:1646  */
    {  (yyval.chare) = new Array(lineno, (yyvsp[-4].cattr), (yyvsp[-3].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3230 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 177:
#line 649 "xi-grammar.y" /* yacc.c:1646  */
    {  (yyval.chare) = new Array(lineno, (yyvsp[-3].cattr), (yyvsp[-4].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3236 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 178:
#line 653 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr)|Chare::CCHARE, new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist));}
#line 3242 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 179:
#line 655 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3248 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 180:
#line 659 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3254 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 181:
#line 663 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new NodeGroup( lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3260 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 182:
#line 667 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Array( lineno, 0, (yyvsp[-3].ntype), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3266 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 183:
#line 671 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-1].strval))); }
#line 3272 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 184:
#line 673 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-4].strval)), (yyvsp[-2].mvlist)); }
#line 3278 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 185:
#line 677 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = 0; }
#line 3284 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 186:
#line 679 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3290 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 187:
#line 683 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = 0; }
#line 3296 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 188:
#line 685 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3302 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 189:
#line 687 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3308 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 190:
#line 691 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TType(new NamedType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3314 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 191:
#line 693 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TFunc((yyvsp[-1].ftype), (yyvsp[0].strval)); }
#line 3320 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 192:
#line 695 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TName((yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].strval)); }
#line 3326 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 193:
#line 699 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = new TVarList((yyvsp[0].tvar)); }
#line 3332 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 194:
#line 701 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = new TVarList((yyvsp[-2].tvar), (yyvsp[0].tvarlist)); }
#line 3338 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 195:
#line 705 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = (yyvsp[-1].tvarlist); }
#line 3344 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 196:
#line 709 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3350 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 197:
#line 711 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3356 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 198:
#line 713 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3362 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 199:
#line 715 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3368 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 200:
#line 717 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].message)); (yyvsp[0].message)->setTemplate((yyval.templat)); }
#line 3374 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 201:
#line 721 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = 0; }
#line 3380 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 202:
#line 723 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = (yyvsp[-2].mbrlist); }
#line 3386 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 203:
#line 727 "xi-grammar.y" /* yacc.c:1646  */
    { 
                  if (!connectEntries.empty()) {
                    (yyval.mbrlist) = new AstChildren<Member>(connectEntries);
		  } else {
		    (yyval.mbrlist) = 0; 
                  }
		}
#line 3398 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 204:
#line 735 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = new AstChildren<Member>(-1, (yyvsp[-1].member), (yyvsp[0].mbrlist)); }
#line 3404 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 205:
#line 739 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].readonly); }
#line 3410 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 206:
#line 741 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].readonly); }
#line 3416 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 208:
#line 744 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].member); }
#line 3422 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 209:
#line 746 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].pupable); }
#line 3428 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 210:
#line 748 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].includeFile); }
#line 3434 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 211:
#line 750 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new ClassDeclaration(lineno,(yyvsp[0].strval)); }
#line 3440 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 212:
#line 754 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1); }
#line 3446 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 213:
#line 756 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1); }
#line 3452 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 214:
#line 758 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[-6].strval)) + '<' +
					    ((yyvsp[-4].tparlist))->to_string() + '>').c_str()),
				    1);
		}
#line 3462 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 215:
#line 764 "xi-grammar.y" /* yacc.c:1646  */
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[-2]).first_column, (yylsp[-2]).last_column, (yylsp[-2]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1);
		}
#line 3472 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 216:
#line 770 "xi-grammar.y" /* yacc.c:1646  */
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[-5]).first_column, (yylsp[-5]).last_column, (yylsp[-5]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1);
		}
#line 3482 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 217:
#line 779 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 0); }
#line 3488 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 218:
#line 781 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 0); }
#line 3494 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 219:
#line 783 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[-6].strval)) + '<' +
					    ((yyvsp[-4].tparlist))->to_string() + '>').c_str()),
				    0);
		}
#line 3504 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 220:
#line 789 "xi-grammar.y" /* yacc.c:1646  */
    {
                  InitCall* rtn = new InitCall(lineno, (yyvsp[-3].strval), 0);
                  rtn->setAccel();
                  (yyval.member) = rtn;
		}
#line 3514 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 221:
#line 797 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[0].ntype),0); }
#line 3520 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 222:
#line 799 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[-2].ntype),(yyvsp[0].pupable)); }
#line 3526 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 223:
#line 802 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.includeFile) = new IncludeFile(lineno,(yyvsp[0].strval)); }
#line 3532 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 224:
#line 806 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].member); }
#line 3538 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 225:
#line 810 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].entry); }
#line 3544 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 226:
#line 812 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyvsp[0].entry)->tspec = (yyvsp[-1].tvarlist);
                  (yyval.member) = (yyvsp[0].entry);
                }
#line 3553 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 227:
#line 817 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[-1].member); }
#line 3559 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 228:
#line 819 "xi-grammar.y" /* yacc.c:1646  */
    {
          ERROR("invalid SDAG member",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 3569 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 229:
#line 827 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3575 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 230:
#line 829 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3581 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 231:
#line 831 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3587 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 232:
#line 833 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3593 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 233:
#line 835 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3599 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 234:
#line 837 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3605 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 235:
#line 839 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3611 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 236:
#line 841 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3617 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 237:
#line 843 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3623 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 238:
#line 845 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3629 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 239:
#line 847 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3635 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 240:
#line 850 "xi-grammar.y" /* yacc.c:1646  */
    { 
                  (yyval.entry) = new Entry(lineno, (yyvsp[-5].intval), (yyvsp[-4].type), (yyvsp[-3].strval), (yyvsp[-2].plist), (yyvsp[-1].val), (yyvsp[0].sentry), (const char *) NULL, (yylsp[-6]).first_line, (yyloc).last_line);
		  if ((yyvsp[0].sentry) != 0) { 
		    (yyvsp[0].sentry)->con1 = new SdagConstruct(SIDENT, (yyvsp[-3].strval));
                    (yyvsp[0].sentry)->entry = (yyval.entry);
                    (yyvsp[0].sentry)->con1->entry = (yyval.entry);
                    (yyvsp[0].sentry)->param = new ParamList((yyvsp[-2].plist));
                  }
		}
#line 3649 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 241:
#line 860 "xi-grammar.y" /* yacc.c:1646  */
    { 
                  Entry *e = new Entry(lineno, (yyvsp[-3].intval), 0, (yyvsp[-2].strval), (yyvsp[-1].plist),  0, (yyvsp[0].sentry), (const char *) NULL, (yylsp[-4]).first_line, (yyloc).last_line);
                  if ((yyvsp[0].sentry) != 0) {
		    (yyvsp[0].sentry)->con1 = new SdagConstruct(SIDENT, (yyvsp[-2].strval));
                    (yyvsp[0].sentry)->entry = e;
                    (yyvsp[0].sentry)->con1->entry = e;
                    (yyvsp[0].sentry)->param = new ParamList((yyvsp[-1].plist));
                  }
		  if (e->param && e->param->isCkMigMsgPtr()) {
		    WARNING("CkMigrateMsg chare constructor is taken for granted",
		            (yyloc).first_column, (yyloc).last_column);
		    (yyval.entry) = NULL;
		  } else {
		    (yyval.entry) = e;
		  }
		}
#line 3670 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 242:
#line 877 "xi-grammar.y" /* yacc.c:1646  */
    {
                  int attribs = SACCEL;
                  const char* name = (yyvsp[-7].strval);
                  ParamList* paramList = (yyvsp[-6].plist);
                  ParamList* accelParamList = (yyvsp[-5].plist);
		  XStr* codeBody = new XStr((yyvsp[-3].strval));
                  const char* callbackName = (yyvsp[-1].strval);

                  (yyval.entry) = new Entry(lineno, attribs, new BuiltinType("void"), name, paramList, 0, 0, 0 );
                  (yyval.entry)->setAccelParam(accelParamList);
                  (yyval.entry)->setAccelCodeBody(codeBody);
                  (yyval.entry)->setAccelCallbackName(new XStr(callbackName));
                }
#line 3688 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 243:
#line 893 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.accelBlock) = new AccelBlock(lineno, new XStr((yyvsp[-2].strval))); }
#line 3694 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 244:
#line 895 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.accelBlock) = new AccelBlock(lineno, NULL); }
#line 3700 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 245:
#line 899 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3706 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 246:
#line 903 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3712 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 247:
#line 905 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[-1].intval); }
#line 3718 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 248:
#line 907 "xi-grammar.y" /* yacc.c:1646  */
    { ERROR("invalid entry method attribute list",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
#line 3727 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 249:
#line 914 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[0].intval); }
#line 3733 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 250:
#line 916 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[-2].intval) | (yyvsp[0].intval); }
#line 3739 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 251:
#line 920 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = STHREADED; }
#line 3745 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 252:
#line 922 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SSYNC; }
#line 3751 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 253:
#line 924 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SIGET; }
#line 3757 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 254:
#line 926 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SLOCKED; }
#line 3763 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 255:
#line 928 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SCREATEHERE; }
#line 3769 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 256:
#line 930 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SCREATEHOME; }
#line 3775 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 257:
#line 932 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SNOKEEP; }
#line 3781 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 258:
#line 934 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SNOTRACE; }
#line 3787 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 259:
#line 936 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SAPPWORK; }
#line 3793 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 260:
#line 938 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SIMMEDIATE; }
#line 3799 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 261:
#line 940 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SSKIPSCHED; }
#line 3805 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 262:
#line 942 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SINLINE; }
#line 3811 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 263:
#line 944 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SLOCAL; }
#line 3817 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 264:
#line 946 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SPYTHON; }
#line 3823 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 265:
#line 948 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SMEM; }
#line 3829 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 266:
#line 950 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SREDUCE; }
#line 3835 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 267:
#line 952 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("invalid entry method attribute",
		        (yylsp[0]).first_column, (yylsp[0]).last_column);
		  yyclearin;
		  yyerrok;
		}
#line 3846 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 268:
#line 961 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 3852 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 269:
#line 963 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 3858 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 270:
#line 965 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 3864 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 271:
#line 969 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = ""; }
#line 3870 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 272:
#line 971 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3876 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 273:
#line 973 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s, %s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 3886 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 274:
#line 981 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = ""; }
#line 3892 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 275:
#line 983 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3898 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 276:
#line 985 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s[%s]%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 3908 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 277:
#line 991 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s{%s}%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 3918 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 278:
#line 997 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s(%s)%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 3928 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 279:
#line 1003 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"(%s)%s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 3938 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 280:
#line 1011 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval));
		}
#line 3947 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 281:
#line 1018 "xi-grammar.y" /* yacc.c:1646  */
    { 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			(yyval.intval) = 0;
		}
#line 3957 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 282:
#line 1026 "xi-grammar.y" /* yacc.c:1646  */
    { 
			in_braces=0;
			(yyval.intval) = 0;
		}
#line 3966 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 283:
#line 1033 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[0].type));}
#line 3972 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 284:
#line 1035 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval)); (yyval.pname)->setConditional((yyvsp[0].intval)); }
#line 3978 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 285:
#line 1037 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[-3].type),(yyvsp[-2].strval),0,(yyvsp[0].val));}
#line 3984 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 286:
#line 1039 "xi-grammar.y" /* yacc.c:1646  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
		}
#line 3993 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 287:
#line 1045 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READONLY; }
#line 3999 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 288:
#line 1046 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READWRITE; }
#line 4005 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 289:
#line 1047 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_WRITEONLY; }
#line 4011 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 290:
#line 1050 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr((yyvsp[0].strval)); }
#line 4017 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 291:
#line 1051 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "->" << (yyvsp[0].strval); }
#line 4023 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 292:
#line 1052 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[-2].xstrptr)) << "." << (yyvsp[0].strval); }
#line 4029 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 293:
#line 1054 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << *((yyvsp[-1].xstrptr)) << "]";
                  delete (yyvsp[-3].xstrptr);
                  delete (yyvsp[-1].xstrptr);
                }
#line 4040 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 294:
#line 1061 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << (yyvsp[-1].strval) << "]";
                  delete (yyvsp[-3].xstrptr);
                }
#line 4050 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 295:
#line 1067 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "(" << *((yyvsp[-1].xstrptr)) << ")";
                  delete (yyvsp[-3].xstrptr);
                  delete (yyvsp[-1].xstrptr);
                }
#line 4061 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 296:
#line 1076 "xi-grammar.y" /* yacc.c:1646  */
    {
                  in_bracket = 0;
                  (yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName(), (yyvsp[-1].strval));
                }
#line 4070 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 297:
#line 1083 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[-6].intval));
                }
#line 4080 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 298:
#line 1089 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType(Parameter::ACCEL_BUFFER_TYPE_READWRITE);
		}
#line 4090 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 299:
#line 1095 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.pname) = (yyvsp[-3].pname);
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[-5].intval));
		}
#line 4100 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 300:
#line 1103 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 4106 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 301:
#line 1105 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 4112 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 302:
#line 1109 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 4118 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 303:
#line 1111 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 4124 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 304:
#line 1115 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = (yyvsp[-1].plist); }
#line 4130 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 305:
#line 1117 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList(new Parameter(0, new BuiltinType("void"))); }
#line 4136 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 306:
#line 1121 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = (yyvsp[-1].plist); }
#line 4142 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 307:
#line 1123 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = 0; }
#line 4148 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 308:
#line 1127 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = 0; }
#line 4154 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 309:
#line 1129 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4160 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 310:
#line 1133 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = 0; }
#line 4166 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 311:
#line 1135 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[0].sc)); }
#line 4172 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 312:
#line 1137 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[-2].slist)); }
#line 4178 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 313:
#line 1141 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.slist) = new SListConstruct((yyvsp[0].sc)); }
#line 4184 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 314:
#line 1143 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.slist) = new SListConstruct((yyvsp[-1].sc), (yyvsp[0].slist));  }
#line 4190 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 315:
#line 1147 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.olist) = new OListConstruct((yyvsp[0].sc)); }
#line 4196 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 316:
#line 1149 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.olist) = new OListConstruct((yyvsp[-1].sc), (yyvsp[0].slist)); }
#line 4202 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 317:
#line 1153 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.clist) = new CaseListConstruct((yyvsp[0].when)); }
#line 4208 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 318:
#line 1155 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.clist) = new CaseListConstruct((yyvsp[-1].when), (yyvsp[0].clist)); }
#line 4214 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 319:
#line 1157 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("case blocks can only contain when clauses",
		        (yylsp[0]).first_column, (yylsp[0]).last_column);
		  (yyval.clist) = 0;
		}
#line 4224 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 320:
#line 1165 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4230 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 321:
#line 1167 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = 0; }
#line 4236 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 322:
#line 1171 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-2].entrylist), 0); }
#line 4242 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 323:
#line 1173 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-1].entrylist), (yyvsp[0].sc)); }
#line 4248 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 324:
#line 1175 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-3].entrylist), (yyvsp[-1].slist)); }
#line 4254 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 325:
#line 1179 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4260 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 326:
#line 1181 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4266 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 327:
#line 1183 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4272 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 328:
#line 1185 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4278 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 329:
#line 1187 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4284 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 330:
#line 1189 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4290 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 331:
#line 1191 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4296 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 332:
#line 1193 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4302 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 333:
#line 1195 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4308 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 334:
#line 1197 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4314 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 335:
#line 1199 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4320 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 336:
#line 1201 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4326 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 337:
#line 1205 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new AtomicConstruct((yyvsp[-1].strval), (yyvsp[-3].strval), (yylsp[-2]).first_line); }
#line 4332 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 338:
#line 1207 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new OverlapConstruct((yyvsp[-1].olist)); }
#line 4338 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 339:
#line 1209 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = (yyvsp[0].when); }
#line 4344 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 340:
#line 1211 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new CaseConstruct((yyvsp[-1].clist)); }
#line 4350 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 341:
#line 1213 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForConstruct((yyvsp[-8].intexpr), (yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4356 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 342:
#line 1215 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForConstruct((yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4362 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 343:
#line 1217 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-9].strval)), (yyvsp[-6].intexpr),
		             (yyvsp[-4].intexpr), (yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4369 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 344:
#line 1220 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-11].strval)), (yyvsp[-8].intexpr),
		             (yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4376 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 345:
#line 1223 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new IfConstruct((yyvsp[-3].intexpr), (yyvsp[-1].sc), (yyvsp[0].sc)); }
#line 4382 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 346:
#line 1225 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new IfConstruct((yyvsp[-5].intexpr), (yyvsp[-2].slist), (yyvsp[0].sc)); }
#line 4388 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 347:
#line 1227 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new WhileConstruct((yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4394 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 348:
#line 1229 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new WhileConstruct((yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4400 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 349:
#line 1231 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new AtomicConstruct((yyvsp[-1].strval), NULL, (yyloc).first_line); }
#line 4406 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 350:
#line 1233 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("unknown SDAG construct or malformed entry method declaration.\n"
		        "You may have forgotten to terminate a previous entry method declaration with a"
		        " semicolon or forgotten to mark a block of sequential SDAG code as 'atomic'",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
#line 4418 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 351:
#line 1243 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = 0; }
#line 4424 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 352:
#line 1245 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ElseConstruct((yyvsp[0].sc)); }
#line 4430 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 353:
#line 1247 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ElseConstruct((yyvsp[-1].slist)); }
#line 4436 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 354:
#line 1251 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intexpr) = new IntExprConstruct((yyvsp[0].strval)); }
#line 4442 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 355:
#line 1255 "xi-grammar.y" /* yacc.c:1646  */
    { in_int_expr = 0; (yyval.intval) = 0; }
#line 4448 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 356:
#line 1259 "xi-grammar.y" /* yacc.c:1646  */
    { in_int_expr = 1; (yyval.intval) = 0; }
#line 4454 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 357:
#line 1263 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[-1].strval), (yyvsp[0].plist), 0, 0, 0, (yyloc).first_line, (yyloc).last_line);
		}
#line 4462 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 358:
#line 1267 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[-4].strval), (yyvsp[0].plist), 0, 0, (yyvsp[-2].strval), (yyloc).first_line, (yyloc).last_line);
		}
#line 4470 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 359:
#line 1273 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.entrylist) = new EntryList((yyvsp[0].entry)); }
#line 4476 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 360:
#line 1275 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.entrylist) = new EntryList((yyvsp[-2].entry),(yyvsp[0].entrylist)); }
#line 4482 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 361:
#line 1279 "xi-grammar.y" /* yacc.c:1646  */
    { in_bracket=1; }
#line 4488 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 362:
#line 1282 "xi-grammar.y" /* yacc.c:1646  */
    { in_bracket=0; }
#line 4494 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 363:
#line 1286 "xi-grammar.y" /* yacc.c:1646  */
    { if (!macroDefined((yyvsp[0].strval), 1)) in_comment = 1; }
#line 4500 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 364:
#line 1290 "xi-grammar.y" /* yacc.c:1646  */
    { if (!macroDefined((yyvsp[0].strval), 0)) in_comment = 1; }
#line 4506 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;


#line 4510 "xi-grammar.tab.C" /* yacc.c:1646  */
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

  /* Now 'shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*--------------------------------------.
| yyerrlab -- here on detecting error.  |
`--------------------------------------*/
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
  /* Do not reclaim the symbols of the rule whose action triggered
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
  yyerrstatus = 3;      /* Each real token shifted decrements this.  */

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

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END

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

#if !defined yyoverflow || YYERROR_VERBOSE
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
  /* Do not reclaim the symbols of the rule whose action triggered
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
  return yyresult;
}
#line 1293 "xi-grammar.y" /* yacc.c:1906  */


void yyerror(const char *msg) { }
