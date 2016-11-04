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

#line 113 "y.tab.c" /* yacc.c:339  */

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
   by #include "y.tab.h".  */
#ifndef YY_YY_Y_TAB_H_INCLUDED
# define YY_YY_Y_TAB_H_INCLUDED
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

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED

union YYSTYPE
{
#line 51 "xi-grammar.y" /* yacc.c:355  */

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

#line 343 "y.tab.c" /* yacc.c:355  */
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

#endif /* !YY_YY_Y_TAB_H_INCLUDED  */

/* Copy the second part of user declarations.  */

#line 374 "y.tab.c" /* yacc.c:358  */

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
#define YYLAST   1487

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  90
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  117
/* YYNRULES -- Number of rules.  */
#define YYNRULES  368
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  718

/* YYTRANSLATE[YYX] -- Symbol number corresponding to YYX as returned
   by yylex, with out-of-bounds checking.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   328

#define YYTRANSLATE(YYX)                                                \
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, without out-of-bounds checking.  */
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
  /* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   192,   192,   197,   200,   205,   206,   210,   212,   217,
     218,   223,   225,   226,   227,   229,   230,   231,   233,   234,
     235,   236,   237,   241,   242,   243,   244,   245,   246,   247,
     248,   249,   250,   251,   252,   253,   254,   255,   256,   257,
     260,   261,   262,   263,   264,   265,   266,   267,   268,   269,
     270,   272,   274,   275,   278,   279,   280,   281,   284,   286,
     293,   297,   304,   306,   311,   312,   316,   318,   320,   322,
     324,   336,   338,   340,   342,   348,   350,   352,   354,   356,
     358,   360,   362,   364,   366,   374,   376,   378,   382,   384,
     389,   390,   395,   396,   400,   402,   404,   406,   408,   410,
     412,   414,   416,   418,   420,   422,   424,   426,   428,   432,
     433,   440,   442,   446,   450,   452,   456,   460,   462,   464,
     466,   468,   470,   474,   476,   478,   480,   482,   486,   488,
     492,   494,   498,   502,   507,   508,   512,   516,   521,   522,
     527,   528,   538,   540,   544,   546,   551,   552,   556,   558,
     563,   564,   568,   573,   574,   578,   580,   584,   586,   591,
     592,   596,   597,   600,   604,   606,   610,   612,   617,   618,
     622,   624,   628,   630,   634,   638,   642,   648,   652,   654,
     658,   660,   664,   668,   672,   676,   678,   683,   684,   689,
     690,   692,   694,   703,   705,   707,   711,   713,   717,   721,
     723,   725,   727,   729,   733,   735,   740,   747,   751,   753,
     755,   756,   758,   760,   762,   766,   768,   770,   776,   782,
     791,   793,   795,   801,   809,   811,   814,   818,   822,   824,
     829,   831,   839,   841,   843,   845,   847,   849,   851,   853,
     855,   857,   859,   862,   871,   887,   903,   905,   909,   914,
     915,   917,   924,   926,   930,   932,   934,   936,   938,   940,
     942,   944,   946,   948,   950,   952,   954,   956,   958,   960,
     962,   974,   983,   985,   987,   992,   993,   995,  1004,  1005,
    1007,  1013,  1019,  1025,  1033,  1040,  1048,  1055,  1057,  1059,
    1061,  1068,  1069,  1070,  1073,  1074,  1075,  1076,  1083,  1089,
    1098,  1105,  1111,  1117,  1125,  1127,  1131,  1133,  1137,  1139,
    1143,  1145,  1150,  1151,  1155,  1157,  1159,  1163,  1165,  1169,
    1171,  1175,  1177,  1179,  1187,  1190,  1193,  1195,  1197,  1201,
    1203,  1205,  1207,  1209,  1211,  1213,  1215,  1217,  1219,  1221,
    1223,  1227,  1229,  1231,  1233,  1235,  1237,  1239,  1242,  1245,
    1247,  1249,  1251,  1253,  1255,  1266,  1267,  1269,  1273,  1277,
    1281,  1285,  1289,  1295,  1297,  1301,  1304,  1308,  1312
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
  "AGGREGATE", "CREATEHERE", "CREATEHOME", "NOKEEP", "NOTRACE", "APPWORK",
  "VOID", "CONST", "PACKED", "VARSIZE", "ENTRY", "FOR", "FORALL", "WHILE",
  "WHEN", "OVERLAP", "ATOMIC", "IF", "ELSE", "PYTHON", "LOCAL",
  "NAMESPACE", "USING", "IDENT", "NUMBER", "LITERAL", "CPROGRAM", "HASHIF",
  "HASHIFDEF", "INT", "LONG", "SHORT", "CHAR", "FLOAT", "DOUBLE",
  "UNSIGNED", "ACCEL", "READWRITE", "WRITEONLY", "ACCELBLOCK",
  "MEMCRITICAL", "REDUCTIONTARGET", "CASE", "';'", "':'", "'{'", "'}'",
  "','", "'<'", "'>'", "'*'", "'('", "')'", "'&'", "'['", "']'", "'='",
  "'-'", "'.'", "$accept", "File", "ModuleEList", "OptExtern",
  "OneOrMoreSemiColon", "OptSemiColon", "Name", "QualName", "Module",
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
     325,   326,   327,   328,    59,    58,   123,   125,    44,    60,
      62,    42,    40,    41,    38,    91,    93,    61,    45,    46
};
# endif

#define YYPACT_NINF -593

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-593)))

#define YYTABLE_NINF -320

#define yytable_value_is_error(Yytable_value) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
     229,  1326,  1326,    35,  -593,   229,  -593,  -593,  -593,  -593,
    -593,  -593,  -593,  -593,  -593,  -593,  -593,  -593,  -593,  -593,
    -593,  -593,  -593,  -593,  -593,  -593,  -593,  -593,  -593,  -593,
    -593,  -593,  -593,  -593,  -593,  -593,  -593,  -593,  -593,  -593,
    -593,  -593,  -593,  -593,  -593,  -593,  -593,  -593,  -593,  -593,
    -593,  -593,  -593,    51,    51,  -593,  -593,  -593,   759,     5,
    -593,  -593,  -593,    41,  1326,   130,  1326,  1326,   129,   920,
      33,   899,   759,  -593,  -593,  -593,  -593,   777,    52,    86,
    -593,    74,  -593,  -593,  -593,     5,   -27,  1347,   120,   120,
     -15,    86,    82,    82,    82,    82,    85,   108,  1326,   139,
     121,   759,  -593,  -593,  -593,  -593,  -593,  -593,  -593,  -593,
     303,  -593,  -593,  -593,  -593,   135,  -593,  -593,  -593,  -593,
    -593,  -593,  -593,  -593,  -593,  -593,  -593,     5,  -593,  -593,
    -593,   777,  -593,    90,  -593,  -593,  -593,  -593,   292,   104,
    -593,  -593,   142,   149,   160,     2,  -593,    86,   759,    74,
     138,    71,   -27,   166,   100,  1421,   142,   149,   160,  -593,
       1,    86,  -593,    86,    86,   179,    86,   176,  -593,    12,
    1326,  1326,  1326,  1326,  1113,   189,   191,   228,  1326,  -593,
    -593,  -593,  1365,   209,    82,    82,    82,    82,   189,   108,
    -593,  -593,  -593,  -593,  -593,     5,  -593,   250,  -593,  -593,
    -593,   218,  -593,  -593,  1408,  -593,  -593,  -593,  -593,  -593,
    -593,  1326,   212,   247,   -27,   249,   -27,   220,  -593,   135,
     239,     6,  -593,   242,  -593,    56,    34,   119,   241,   205,
      86,  -593,  -593,   251,   258,   261,   268,   268,   268,   268,
    -593,  1326,   262,   269,   264,  1184,  1326,   306,  1326,  -593,
    -593,   275,   284,   290,  1326,   151,  1326,   296,   295,   135,
    1326,  1326,  1326,  1326,  1326,  1326,  -593,  -593,  -593,  -593,
     298,  -593,   297,  -593,   261,  -593,  -593,   301,   305,   310,
     293,   -27,     5,    86,  1326,  -593,   299,  -593,   -27,   120,
    1408,   120,   120,  1408,   120,  -593,  -593,    12,  -593,    86,
     132,   132,   132,   132,   313,  -593,   306,  -593,   268,   268,
    -593,   228,   379,   316,   196,  -593,   318,  1365,  -593,  -593,
     268,   268,   268,   268,   268,   152,  1408,  -593,   324,   -27,
     249,   -27,   -27,  -593,    56,   317,  -593,   322,  -593,   330,
     336,   340,    86,   339,   341,  -593,   354,  -593,   421,     5,
    -593,  -593,  -593,  -593,  -593,  -593,   132,   132,  -593,  -593,
    1421,    -3,   357,  1421,  -593,  -593,  -593,  -593,  -593,  -593,
     132,   132,   132,   132,   132,   379,     5,  -593,  1378,  -593,
    -593,  -593,  -593,  -593,  -593,   351,  -593,  -593,  -593,   353,
    -593,    93,   355,  -593,    86,  -593,   676,   405,   370,   135,
     421,  -593,  -593,  -593,  -593,  1326,  -593,  -593,  -593,  -593,
    -593,  -593,  -593,  -593,   371,  -593,  1326,   -27,   373,   366,
    1421,   120,   120,   120,  -593,  -593,   936,  1042,  -593,   135,
       5,  -593,   367,   135,    14,   368,  1421,  -593,   375,   376,
     377,   378,  -593,  -593,  -593,  -593,  -593,  -593,  -593,  -593,
    -593,  -593,  -593,  -593,  -593,  -593,   397,  -593,   388,  -593,
    -593,   389,   385,   394,   324,  1326,  -593,   391,   135,     5,
    -593,   243,  -593,  -593,  -593,  -593,  -593,  -593,  -593,  -593,
    -593,   441,  -593,   988,   265,   324,  -593,     5,  -593,  -593,
      74,  -593,  1326,  -593,  -593,   399,   400,   399,   428,   408,
     431,   399,   413,   248,     5,   -27,  -593,  -593,  -593,   475,
     324,  -593,   -27,   438,   -27,   -26,   418,   482,   536,  -593,
     423,   -27,  1344,   425,   424,   166,   416,   265,   419,  -593,
     432,   422,   427,  -593,   -27,   428,   325,  -593,   435,   472,
     -27,   427,   399,   436,   399,   446,   431,   399,   455,   -27,
     456,  1344,  -593,   135,  -593,   135,   477,  -593,   326,   423,
     -27,   399,  -593,   571,   322,  -593,  -593,   459,  -593,  -593,
     166,   743,   -27,   484,   -27,   536,   423,   -27,  1344,   166,
    -593,  -593,  -593,  -593,  -593,  -593,  -593,  -593,  -593,  1326,
     464,   463,   457,   -27,   468,   -27,   248,  -593,   324,  -593,
     135,   248,   495,   473,   460,   427,   474,   -27,   427,   476,
     135,   478,  1421,   596,  -593,   166,   -27,   479,   483,  -593,
    -593,   486,   750,  -593,   -27,   399,   757,  -593,   166,   805,
    -593,  -593,  1326,  1326,   -27,   480,  -593,  1326,   427,   -27,
    -593,   495,   248,  -593,   490,   -27,   248,  -593,   135,   248,
     495,  -593,    92,   -25,   470,  1326,   135,   812,   487,  -593,
     489,   -27,   492,   494,  -593,   496,  -593,  -593,  1326,  1255,
     505,  1326,  1326,  -593,   136,     5,   248,  -593,   -27,  -593,
     427,   -27,  -593,   495,   246,   488,   356,  1326,  -593,   329,
    -593,   498,   427,   819,   508,  -593,  -593,  -593,  -593,  -593,
    -593,  -593,   826,   248,  -593,   -27,   248,  -593,   499,   427,
     510,  -593,   874,  -593,   248,  -593,   512,  -593
};

  /* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
     Performed when YYTABLE does not specify something else to do.  Zero
     means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       3,     0,     0,     0,     2,     3,    12,    13,    14,    15,
      16,    17,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    31,    32,    33,    34,    35,
      36,    37,    38,    39,    40,    41,    42,    43,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    11,    53,    54,
      55,    56,    57,     0,     0,     1,     4,     7,     0,    62,
      60,    61,    84,     6,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    83,    81,    82,     8,     0,     0,     0,
      58,    67,   367,   368,   285,   247,   278,     0,   138,   138,
     138,     0,   146,   146,   146,   146,     0,   140,     0,     0,
       0,     0,    75,   208,   209,    69,    76,    77,    78,    79,
       0,    80,    68,   211,   210,     9,   242,   234,   235,   236,
     237,   238,   240,   241,   239,   232,   233,    73,    74,    65,
     108,     0,    94,    95,    96,    97,   105,   106,     0,    92,
     111,   112,   123,   124,   125,   129,   248,     0,     0,    66,
       0,   279,   278,     0,     0,     0,   117,   118,   119,   120,
     131,     0,   139,     0,     0,     0,     0,   224,   212,     0,
       0,     0,     0,     0,     0,     0,   153,     0,     0,   214,
     226,   213,     0,     0,   146,   146,   146,   146,     0,   140,
     199,   200,   201,   202,   203,    10,    63,   126,   104,   107,
      98,    99,   102,   103,    90,   110,   113,   114,   115,   127,
     128,     0,     0,     0,   278,   275,   278,     0,   286,     0,
       0,   121,   122,     0,   130,   134,   218,   215,     0,   220,
       0,   157,   158,     0,   148,    92,   168,   168,   168,   168,
     152,     0,     0,   155,     0,     0,     0,     0,     0,   144,
     145,     0,   142,   166,     0,   120,     0,   196,     0,     9,
       0,     0,     0,     0,     0,     0,   100,   101,    86,    87,
      88,    91,     0,    85,    92,    72,    59,     0,   276,     0,
       0,   278,   246,     0,     0,   365,   134,   136,   278,   138,
       0,   138,   138,     0,   138,   225,   147,     0,   109,     0,
       0,     0,     0,     0,     0,   177,     0,   154,   168,   168,
     141,     0,   159,   187,     0,   194,   189,     0,   198,    71,
     168,   168,   168,   168,   168,     0,     0,    93,     0,   278,
     275,   278,   278,   283,   134,     0,   135,     0,   132,     0,
       0,     0,     0,     0,     0,   149,   170,   169,     0,   204,
     172,   173,   174,   175,   176,   156,     0,     0,   143,   160,
       0,   159,     0,     0,   193,   190,   191,   192,   195,   197,
       0,     0,     0,     0,     0,   159,   185,    89,     0,    70,
     281,   277,   282,   280,   137,     0,   366,   133,   219,     0,
     216,     0,     0,   221,     0,   231,     0,     0,     0,     0,
       0,   227,   228,   178,   179,     0,   165,   167,   188,   180,
     181,   182,   183,   184,     0,   309,   287,   278,   304,     0,
       0,   138,   138,   138,   171,   251,     0,     0,   229,     9,
     230,   207,   161,     0,   159,     0,     0,   308,     0,     0,
       0,     0,   271,   254,   255,   256,   257,   263,   264,   265,
     270,   258,   259,   260,   261,   262,   150,   266,     0,   268,
     269,     0,   252,    58,     0,     0,   205,     0,     0,   186,
     284,     0,   288,   290,   305,   116,   217,   223,   222,   151,
     267,     0,   250,     0,     0,     0,   162,   163,   273,   272,
     274,   289,     0,   253,   354,     0,     0,     0,     0,     0,
     325,     0,     0,     0,   314,   278,   244,   343,   315,   312,
       0,   360,   278,     0,   278,     0,   363,     0,     0,   324,
       0,   278,     0,     0,     0,     0,     0,     0,     0,   358,
       0,     0,     0,   361,   278,     0,     0,   327,     0,     0,
     278,     0,     0,     0,     0,     0,   325,     0,     0,   278,
       0,   321,   323,     9,   318,     9,     0,   243,     0,     0,
     278,     0,   359,     0,     0,   364,   326,     0,   342,   320,
       0,     0,   278,     0,   278,     0,     0,   278,     0,     0,
     344,   322,   316,   353,   313,   291,   292,   293,   311,     0,
       0,   306,     0,   278,     0,   278,     0,   351,     0,   328,
       9,     0,   355,     0,     0,     0,     0,   278,     0,     0,
       9,     0,     0,     0,   310,     0,   278,     0,     0,   362,
     341,     0,     0,   349,   278,     0,     0,   330,     0,     0,
     331,   340,     0,     0,   278,     0,   307,     0,     0,   278,
     352,   355,     0,   356,     0,   278,     0,   338,     9,     0,
     355,   294,     0,     0,     0,     0,     0,     0,     0,   350,
       0,   278,     0,     0,   329,     0,   336,   302,     0,     0,
       0,     0,     0,   300,     0,   245,     0,   346,   278,   357,
       0,   278,   339,   355,     0,     0,     0,     0,   296,     0,
     303,     0,     0,     0,     0,   337,   299,   298,   297,   295,
     301,   345,     0,     0,   333,   278,     0,   347,     0,     0,
       0,   332,     0,   348,     0,   334,     0,   335
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -593,  -593,   585,  -593,   -48,  -251,    -1,   -61,   522,   538,
     -56,  -593,  -593,  -593,  -176,  -593,  -178,  -593,  -135,   -72,
     -70,   -67,   -62,  -171,   439,   462,  -593,   -78,  -593,  -593,
    -262,  -593,  -593,   -77,   406,   285,  -593,    50,   302,  -593,
    -593,   429,   300,  -593,   164,  -593,  -593,  -242,  -593,   -51,
     207,  -593,  -593,  -593,  -147,  -593,  -593,  -593,  -593,  -593,
    -593,  -593,   287,  -593,   294,   535,  -593,   267,   210,   551,
    -593,  -593,   392,  -593,  -593,  -593,  -593,   230,  -593,   180,
    -593,   143,  -593,  -593,   312,   -84,    13,   -65,  -492,  -593,
    -593,  -548,  -593,  -593,  -314,    15,  -439,  -593,  -593,   102,
    -507,    55,  -525,    89,  -501,  -593,  -437,  -592,  -487,  -490,
    -449,  -593,   110,   128,    84,  -593,  -593
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     3,     4,    69,   349,   196,   235,   139,     5,    60,
      70,    71,    72,   270,   271,   272,   205,   140,   236,   141,
     156,   157,   158,   159,   160,   145,   146,   273,   337,   286,
     287,   103,   104,   163,   178,   251,   252,   170,   233,   480,
     243,   175,   244,   234,   360,   468,   361,   362,   105,   300,
     347,   106,   107,   108,   176,   109,   190,   191,   192,   193,
     194,   364,   315,   257,   258,   397,   111,   350,   398,   399,
     113,   114,   168,   181,   400,   401,   128,   402,    73,   147,
     427,   461,   462,   491,   279,   529,   417,   505,   219,   418,
     590,   652,   635,   591,   419,   592,   379,   559,   527,   506,
     523,   538,   550,   520,   507,   552,   524,   623,   530,   563,
     512,   516,   517,   288,   387,    74,    75
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[] =
{
      53,    54,   153,    86,    81,    59,    59,   142,   319,   161,
     143,   255,   164,   166,   359,   144,   129,   554,   149,   167,
      85,   551,   162,   127,   336,   484,   581,   532,   246,   567,
     151,   359,   569,   555,   541,    55,   237,   238,   239,   222,
     209,   264,   231,   253,   222,   183,   509,   508,   514,   659,
     551,   571,   521,   609,   672,   152,   378,   298,   666,   285,
     470,   142,   232,    78,   143,    82,    83,   195,   217,   144,
     165,   528,   384,   594,  -164,   211,   533,   551,   600,    76,
     537,   539,    77,   223,   220,   224,   210,   610,   223,   618,
     508,   695,   212,   572,   621,   574,   328,   179,   577,   470,
     225,   471,   226,   227,   256,   229,   438,   674,   617,   150,
     115,   308,   595,   309,   340,   626,   289,   343,   629,   406,
     684,   686,   474,   637,   689,    57,   597,    58,   148,   638,
     277,   150,   280,   414,   602,   660,   648,   130,   539,   663,
      80,   285,   665,   171,   172,   173,   255,   214,   657,   150,
     377,   198,   658,   215,    80,   199,   216,   162,   167,   619,
     132,   133,   134,   135,   136,   137,   138,   169,   150,   691,
     174,   282,   667,   242,   668,   422,   645,   669,   466,   150,
     670,   671,    79,   204,    80,   643,   301,   302,   303,   647,
     693,   692,   650,   177,   150,   180,   708,   333,   290,   710,
     182,   291,   702,    57,   338,    84,    57,   716,   348,    57,
     274,   195,   339,   213,   341,   342,   690,   344,   668,   712,
     677,   669,   334,   206,   670,   671,    57,   346,   375,  -189,
     207,  -189,     1,     2,   260,   261,   262,   263,   314,   256,
     304,   208,   367,   218,   242,   380,   228,   382,   383,   494,
      80,   365,   366,   313,   230,   316,   704,   356,   357,   320,
     321,   322,   323,   324,   325,   707,   494,   249,   250,   370,
     371,   372,   373,   374,   245,   715,   247,   376,   266,   267,
     150,   391,   405,   335,   293,   408,   259,   294,   209,   275,
     495,   496,   497,   498,   499,   500,   501,    80,   488,   489,
     416,   276,   582,   281,   583,  -285,   278,   495,   496,   497,
     498,   499,   500,   501,   184,   185,   186,   187,   188,   189,
     283,   502,   346,   284,    84,  -285,   494,   292,   668,   696,
    -285,   669,   585,   435,   670,   671,   297,   296,   502,    57,
     204,   503,   416,   299,   439,   440,   441,   306,   305,   620,
     307,   430,   200,   201,   202,   203,   240,   142,   416,   631,
     143,   310,   311,   130,   155,   144,   312,   495,   496,   497,
     498,   499,   500,   501,   317,   318,   326,   327,   329,   332,
      80,   195,  -285,   330,   285,   469,   132,   133,   134,   135,
     136,   137,   138,   331,   586,   587,   359,   664,   502,   354,
     385,    84,   566,   363,   432,   314,   378,  -285,   386,   700,
     490,   668,   588,   388,   669,   434,   389,   670,   671,   392,
     487,   525,   395,   390,   393,   494,   464,    87,    88,    89,
      90,    91,   394,   420,   407,   421,   504,   423,   668,    98,
      99,   669,   698,   100,   670,   671,   396,   429,   433,   437,
     564,   436,   467,   479,   473,   540,   570,   549,   475,   476,
     477,   478,   396,   483,   485,   579,   495,   496,   497,   498,
     499,   500,   501,   494,   481,   482,   -11,   486,   492,   504,
     589,   511,   515,   494,   518,   513,   549,   519,   603,   522,
     605,   510,   531,   608,   593,   526,   535,   502,  -206,    84,
      84,  -317,   553,   556,   558,   195,   560,   195,   561,   615,
     562,   607,   568,   549,   495,   496,   497,   498,   499,   500,
     501,   573,   575,   628,   495,   496,   497,   498,   499,   500,
     501,   578,   584,   580,   633,   589,   599,   494,   604,   612,
     644,   613,   616,   614,   622,   502,   625,   624,    84,  -319,
     654,   627,   195,   630,   639,   502,   673,   632,   536,   655,
     640,   662,   195,   641,   661,   678,   679,   681,   351,   352,
     353,   682,   494,   683,   697,   701,   711,   680,   495,   496,
     497,   498,   499,   500,   501,   687,   705,   713,   611,   717,
      56,   102,    61,   197,   221,   265,   358,   694,   472,   345,
     195,   424,   585,   368,   110,   248,   355,   465,   675,   502,
     431,   369,    84,   495,   496,   497,   498,   499,   500,   501,
     112,   709,   295,   403,   404,   634,   493,   428,   636,   557,
     606,   651,   653,   130,   155,   576,   656,   409,   410,   411,
     412,   413,   381,   534,   502,   565,     0,   596,   598,     0,
      80,     0,     0,     0,   651,     0,   132,   133,   134,   135,
     136,   137,   138,     0,   586,   587,     0,   651,   651,     0,
     688,   651,     0,     0,     0,     0,     0,   425,     0,  -249,
    -249,  -249,     0,  -249,  -249,  -249,   699,  -249,  -249,  -249,
    -249,  -249,     0,     0,     0,  -249,  -249,  -249,  -249,  -249,
    -249,  -249,  -249,  -249,  -249,  -249,  -249,     0,  -249,  -249,
    -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,
    -249,  -249,  -249,  -249,  -249,  -249,     0,  -249,     0,  -249,
    -249,     0,     0,     0,     0,     0,  -249,  -249,  -249,  -249,
    -249,  -249,  -249,  -249,   494,     0,  -249,  -249,  -249,  -249,
       0,   494,     0,     0,     0,     0,     0,     0,   494,     0,
      62,   426,    -5,    -5,    63,    -5,    -5,    -5,    -5,    -5,
      -5,    -5,    -5,    -5,    -5,    -5,     0,    -5,    -5,     0,
       0,    -5,     0,     0,     0,   495,   496,   497,   498,   499,
     500,   501,   495,   496,   497,   498,   499,   500,   501,   495,
     496,   497,   498,   499,   500,   501,   494,     0,     0,     0,
       0,    64,    65,   494,   130,   131,   502,    66,    67,   601,
     494,     0,     0,   502,     0,     0,   642,   494,     0,    68,
     502,    80,     0,   646,     0,    -5,   -64,   132,   133,   134,
     135,   136,   137,   138,     0,     0,     0,   495,   496,   497,
     498,   499,   500,   501,   495,   496,   497,   498,   499,   500,
     501,   495,   496,   497,   498,   499,   500,   501,   495,   496,
     497,   498,   499,   500,   501,   494,     0,     0,   502,     0,
       0,   649,     0,     0,     0,   502,     0,     0,   676,     0,
       0,     0,   502,     0,     0,   703,     0,     0,     0,   502,
       0,     0,   706,     0,     0,   116,   117,   118,   119,     0,
     120,   121,   122,   123,   124,     0,   495,   496,   497,   498,
     499,   500,   501,     1,     2,     0,    87,    88,    89,    90,
      91,    92,    93,    94,    95,    96,    97,   442,    98,    99,
     125,     0,   100,     0,     0,     0,     0,   502,     0,     0,
     714,     0,     0,     0,     0,     0,     0,   443,     0,   444,
     445,   446,   447,   448,   449,     0,     0,   450,   451,   452,
     453,   454,   455,    57,     0,     0,   126,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   456,   457,     0,   442,
       0,     0,     0,     0,     0,     0,   101,     0,     0,     0,
       0,     0,     0,   458,     0,     0,     0,   459,   460,   443,
       0,   444,   445,   446,   447,   448,   449,     0,     0,   450,
     451,   452,   453,   454,   455,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   456,   457,
       0,     0,     0,     0,     0,     6,     7,     8,     0,     9,
      10,    11,     0,    12,    13,    14,    15,    16,     0,   459,
     460,    17,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,     0,    29,    30,    31,    32,    33,   130,
     131,    34,    35,    36,    37,    38,    39,    40,    41,    42,
      43,    44,     0,    45,     0,    46,   463,     0,     0,     0,
       0,     0,   132,   133,   134,   135,   136,   137,   138,    48,
       0,     0,    49,    50,    51,    52,     6,     7,     8,     0,
       9,    10,    11,     0,    12,    13,    14,    15,    16,     0,
       0,     0,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,     0,    29,    30,    31,    32,    33,
       0,     0,    34,    35,    36,    37,    38,    39,    40,    41,
      42,    43,    44,   240,    45,     0,    46,    47,   241,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      48,     0,     0,    49,    50,    51,    52,     6,     7,     8,
       0,     9,    10,    11,     0,    12,    13,    14,    15,    16,
       0,     0,     0,    17,    18,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,     0,    29,    30,    31,    32,
      33,     0,     0,    34,    35,    36,    37,    38,    39,    40,
      41,    42,    43,    44,     0,    45,     0,    46,    47,   241,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    48,     0,     0,    49,    50,    51,    52,     6,     7,
       8,     0,     9,    10,    11,     0,    12,    13,    14,    15,
      16,     0,     0,     0,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,     0,    29,    30,    31,
      32,    33,     0,     0,    34,    35,    36,    37,    38,    39,
      40,    41,    42,    43,    44,     0,    45,     0,    46,    47,
     685,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    48,     0,     0,    49,    50,    51,    52,     6,
       7,     8,     0,     9,    10,    11,     0,    12,    13,    14,
      15,    16,     0,     0,     0,    17,    18,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,     0,    29,    30,
      31,    32,    33,   154,     0,    34,    35,    36,    37,    38,
      39,    40,    41,    42,    43,    44,     0,    45,     0,    46,
      47,     0,     0,   254,   130,   155,   542,   543,   544,   498,
     545,   546,   547,    48,     0,     0,    49,    50,    51,    52,
       0,    80,   130,   155,     0,     0,     0,   132,   133,   134,
     135,   136,   137,   138,     0,   130,   155,   548,     0,    80,
      84,     0,     0,     0,     0,   132,   133,   134,   135,   136,
     137,   138,    80,     0,     0,     0,     0,     0,   132,   133,
     134,   135,   136,   137,   138,   130,   155,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   130,   155,
       0,   415,    80,   268,   269,     0,     0,     0,   132,   133,
     134,   135,   136,   137,   138,    80,     0,     0,     0,     0,
       0,   132,   133,   134,   135,   136,   137,   138
};

static const yytype_int16 yycheck[] =
{
       1,     2,    86,    68,    65,    53,    54,    77,   259,    87,
      77,   182,    89,    90,    17,    77,    72,   524,    79,    91,
      68,   522,    37,    71,   286,   464,   551,   514,   175,   536,
      57,    17,   539,   525,   521,     0,   171,   172,   173,    38,
      38,   188,    30,   178,    38,   101,   485,   484,   497,   641,
     551,   541,   501,   578,    79,    82,    82,   235,   650,    85,
      85,   131,    50,    64,   131,    66,    67,   115,   152,   131,
      85,   510,   334,   560,    77,   147,   515,   578,   570,    74,
     517,   518,    41,    82,   154,    84,    84,   579,    82,   596,
     527,   683,   148,   542,   601,   544,   274,    98,   547,    85,
     161,    87,   163,   164,   182,   166,   420,   655,   595,    75,
      77,   246,   561,   248,   290,   605,    82,   293,   608,   361,
     668,   669,   436,   615,   672,    74,   563,    76,    76,   616,
     214,    75,   216,   375,   571,   642,   628,    37,   575,   646,
      54,    85,   649,    93,    94,    95,   317,    76,   638,    75,
     326,    61,   639,    82,    54,    65,    85,    37,   230,   598,
      60,    61,    62,    63,    64,    65,    66,    85,    75,   676,
      85,   219,    80,   174,    82,    82,   625,    85,   429,    75,
      88,    89,    52,    79,    54,   622,   237,   238,   239,   626,
     680,   678,   629,    85,    75,    56,   703,   281,    79,   706,
      79,    82,   692,    74,   288,    76,    74,   714,    76,    74,
     211,   259,   289,    75,   291,   292,    80,   294,    82,   709,
     657,    85,   283,    81,    88,    89,    74,   299,    76,    78,
      81,    80,     3,     4,   184,   185,   186,   187,    87,   317,
     241,    81,   314,    77,   245,   329,    67,   331,   332,     1,
      54,    55,    56,   254,    78,   256,   693,   308,   309,   260,
     261,   262,   263,   264,   265,   702,     1,    39,    40,   320,
     321,   322,   323,   324,    85,   712,    85,   325,    60,    61,
      75,   342,   360,   284,    79,   363,    77,    82,    38,    77,
      42,    43,    44,    45,    46,    47,    48,    54,    55,    56,
     378,    54,   553,    83,   555,    57,    57,    42,    43,    44,
      45,    46,    47,    48,    11,    12,    13,    14,    15,    16,
      81,    73,   394,    81,    76,    77,     1,    86,    82,    83,
      82,    85,     6,   417,    88,    89,    78,    86,    73,    74,
      79,    76,   420,    75,   421,   422,   423,    78,    86,   600,
      86,   399,    60,    61,    62,    63,    50,   427,   436,   610,
     427,    86,    78,    37,    38,   427,    76,    42,    43,    44,
      45,    46,    47,    48,    78,    80,    78,    80,    77,    86,
      54,   429,    57,    78,    85,   433,    60,    61,    62,    63,
      64,    65,    66,    83,    68,    69,    17,   648,    73,    86,
      83,    76,    77,    87,   405,    87,    82,    82,    86,    80,
     471,    82,    86,    83,    85,   416,    80,    88,    89,    80,
     468,   505,     1,    83,    83,     1,   427,     6,     7,     8,
       9,    10,    78,    82,    77,    82,   484,    82,    82,    18,
      19,    85,    86,    22,    88,    89,    41,    77,    77,    83,
     534,    78,    85,    56,    86,   520,   540,   522,    83,    83,
      83,    83,    41,    78,   465,   549,    42,    43,    44,    45,
      46,    47,    48,     1,    86,    86,    82,    86,    37,   527,
     558,    82,    54,     1,    76,    85,   551,    56,   572,    76,
     574,   492,    54,   577,   559,    20,    78,    73,    77,    76,
      76,    77,    77,    87,    85,   553,    74,   555,    86,   593,
      83,   576,    77,   578,    42,    43,    44,    45,    46,    47,
      48,    85,    76,   607,    42,    43,    44,    45,    46,    47,
      48,    76,    55,    77,   612,   613,    77,     1,    54,    75,
     624,    78,    74,    86,    49,    73,    86,    74,    76,    77,
     634,    77,   600,    77,    75,    73,    86,    79,    76,    79,
      77,   645,   610,    77,    74,    78,    77,    75,   301,   302,
     303,    77,     1,    77,    86,    77,    77,   661,    42,    43,
      44,    45,    46,    47,    48,    80,    78,    77,   589,    77,
       5,    69,    54,   131,   155,   189,   311,   681,   434,   297,
     648,   394,     6,   316,    69,   176,   306,   427,   656,    73,
     400,   317,    76,    42,    43,    44,    45,    46,    47,    48,
      69,   705,   230,   356,   357,   612,   483,   397,   613,   527,
     575,   632,   633,    37,    38,   546,   637,   370,   371,   372,
     373,   374,   330,   515,    73,   535,    -1,    76,   564,    -1,
      54,    -1,    -1,    -1,   655,    -1,    60,    61,    62,    63,
      64,    65,    66,    -1,    68,    69,    -1,   668,   669,    -1,
     671,   672,    -1,    -1,    -1,    -1,    -1,     1,    -1,     3,
       4,     5,    -1,     7,     8,     9,   687,    11,    12,    13,
      14,    15,    -1,    -1,    -1,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,    29,    30,    -1,    32,    33,
      34,    35,    36,    37,    38,    39,    40,    41,    42,    43,
      44,    45,    46,    47,    48,    49,    -1,    51,    -1,    53,
      54,    -1,    -1,    -1,    -1,    -1,    60,    61,    62,    63,
      64,    65,    66,    67,     1,    -1,    70,    71,    72,    73,
      -1,     1,    -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,
       1,    85,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    -1,    18,    19,    -1,
      -1,    22,    -1,    -1,    -1,    42,    43,    44,    45,    46,
      47,    48,    42,    43,    44,    45,    46,    47,    48,    42,
      43,    44,    45,    46,    47,    48,     1,    -1,    -1,    -1,
      -1,    52,    53,     1,    37,    38,    73,    58,    59,    76,
       1,    -1,    -1,    73,    -1,    -1,    76,     1,    -1,    70,
      73,    54,    -1,    76,    -1,    76,    77,    60,    61,    62,
      63,    64,    65,    66,    -1,    -1,    -1,    42,    43,    44,
      45,    46,    47,    48,    42,    43,    44,    45,    46,    47,
      48,    42,    43,    44,    45,    46,    47,    48,    42,    43,
      44,    45,    46,    47,    48,     1,    -1,    -1,    73,    -1,
      -1,    76,    -1,    -1,    -1,    73,    -1,    -1,    76,    -1,
      -1,    -1,    73,    -1,    -1,    76,    -1,    -1,    -1,    73,
      -1,    -1,    76,    -1,    -1,     6,     7,     8,     9,    -1,
      11,    12,    13,    14,    15,    -1,    42,    43,    44,    45,
      46,    47,    48,     3,     4,    -1,     6,     7,     8,     9,
      10,    11,    12,    13,    14,    15,    16,     1,    18,    19,
      41,    -1,    22,    -1,    -1,    -1,    -1,    73,    -1,    -1,
      76,    -1,    -1,    -1,    -1,    -1,    -1,    21,    -1,    23,
      24,    25,    26,    27,    28,    -1,    -1,    31,    32,    33,
      34,    35,    36,    74,    -1,    -1,    77,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    50,    51,    -1,     1,
      -1,    -1,    -1,    -1,    -1,    -1,    76,    -1,    -1,    -1,
      -1,    -1,    -1,    67,    -1,    -1,    -1,    71,    72,    21,
      -1,    23,    24,    25,    26,    27,    28,    -1,    -1,    31,
      32,    33,    34,    35,    36,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    50,    51,
      -1,    -1,    -1,    -1,    -1,     3,     4,     5,    -1,     7,
       8,     9,    -1,    11,    12,    13,    14,    15,    -1,    71,
      72,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      28,    29,    30,    -1,    32,    33,    34,    35,    36,    37,
      38,    39,    40,    41,    42,    43,    44,    45,    46,    47,
      48,    49,    -1,    51,    -1,    53,    54,    -1,    -1,    -1,
      -1,    -1,    60,    61,    62,    63,    64,    65,    66,    67,
      -1,    -1,    70,    71,    72,    73,     3,     4,     5,    -1,
       7,     8,     9,    -1,    11,    12,    13,    14,    15,    -1,
      -1,    -1,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    -1,    32,    33,    34,    35,    36,
      -1,    -1,    39,    40,    41,    42,    43,    44,    45,    46,
      47,    48,    49,    50,    51,    -1,    53,    54,    55,    -1,
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
      25,    26,    27,    28,    29,    30,    -1,    32,    33,    34,
      35,    36,    -1,    -1,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    -1,    51,    -1,    53,    54,
      55,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    67,    -1,    -1,    70,    71,    72,    73,     3,
       4,     5,    -1,     7,     8,     9,    -1,    11,    12,    13,
      14,    15,    -1,    -1,    -1,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,    29,    30,    -1,    32,    33,
      34,    35,    36,    16,    -1,    39,    40,    41,    42,    43,
      44,    45,    46,    47,    48,    49,    -1,    51,    -1,    53,
      54,    -1,    -1,    18,    37,    38,    42,    43,    44,    45,
      46,    47,    48,    67,    -1,    -1,    70,    71,    72,    73,
      -1,    54,    37,    38,    -1,    -1,    -1,    60,    61,    62,
      63,    64,    65,    66,    -1,    37,    38,    73,    -1,    54,
      76,    -1,    -1,    -1,    -1,    60,    61,    62,    63,    64,
      65,    66,    54,    -1,    -1,    -1,    -1,    -1,    60,    61,
      62,    63,    64,    65,    66,    37,    38,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    37,    38,
      -1,    83,    54,    55,    56,    -1,    -1,    -1,    60,    61,
      62,    63,    64,    65,    66,    54,    -1,    -1,    -1,    -1,
      -1,    60,    61,    62,    63,    64,    65,    66
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     3,     4,    91,    92,    98,     3,     4,     5,     7,
       8,     9,    11,    12,    13,    14,    15,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    32,
      33,    34,    35,    36,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    51,    53,    54,    67,    70,
      71,    72,    73,    96,    96,     0,    92,    74,    76,    94,
      99,    99,     1,     5,    52,    53,    58,    59,    70,    93,
     100,   101,   102,   168,   205,   206,    74,    41,    96,    52,
      54,    97,    96,    96,    76,    94,   177,     6,     7,     8,
       9,    10,    11,    12,    13,    14,    15,    16,    18,    19,
      22,    76,    98,   121,   122,   138,   141,   142,   143,   145,
     155,   156,   159,   160,   161,    77,     6,     7,     8,     9,
      11,    12,    13,    14,    15,    41,    77,    94,   166,   100,
      37,    38,    60,    61,    62,    63,    64,    65,    66,    97,
     107,   109,   110,   111,   112,   115,   116,   169,    76,    97,
      75,    57,    82,   175,    16,    38,   110,   111,   112,   113,
     114,   117,    37,   123,   123,    85,   123,   109,   162,    85,
     127,   127,   127,   127,    85,   131,   144,    85,   124,    96,
      56,   163,    79,   100,    11,    12,    13,    14,    15,    16,
     146,   147,   148,   149,   150,    94,    95,   115,    61,    65,
      60,    61,    62,    63,    79,   106,    81,    81,    81,    38,
      84,   109,   100,    75,    76,    82,    85,   175,    77,   178,
     110,   114,    38,    82,    84,    97,    97,    97,    67,    97,
      78,    30,    50,   128,   133,    96,   108,   108,   108,   108,
      50,    55,    96,   130,   132,    85,   144,    85,   131,    39,
      40,   125,   126,   108,    18,   113,   117,   153,   154,    77,
     127,   127,   127,   127,   144,   124,    60,    61,    55,    56,
     103,   104,   105,   117,    96,    77,    54,   175,    57,   174,
     175,    83,    94,    81,    81,    85,   119,   120,   203,    82,
      79,    82,    86,    79,    82,   162,    86,    78,   106,    75,
     139,   139,   139,   139,    96,    86,    78,    86,   108,   108,
      86,    78,    76,    96,    87,   152,    96,    78,    80,    95,
      96,    96,    96,    96,    96,    96,    78,    80,   106,    77,
      78,    83,    86,   175,    97,    96,   120,   118,   175,   123,
     104,   123,   123,   104,   123,   128,   109,   140,    76,    94,
     157,   157,   157,   157,    86,   132,   139,   139,   125,    17,
     134,   136,   137,    87,   151,    55,    56,   109,   152,   154,
     139,   139,   139,   139,   139,    76,    94,   104,    82,   186,
     175,   174,   175,   175,   120,    83,    86,   204,    83,    80,
      83,    97,    80,    83,    78,     1,    41,   155,   158,   159,
     164,   165,   167,   157,   157,   117,   137,    77,   117,   157,
     157,   157,   157,   157,   137,    83,   117,   176,   179,   184,
      82,    82,    82,    82,   140,     1,    85,   170,   167,    77,
      94,   158,    96,    77,    96,   175,    78,    83,   184,   123,
     123,   123,     1,    21,    23,    24,    25,    26,    27,    28,
      31,    32,    33,    34,    35,    36,    50,    51,    67,    71,
      72,   171,   172,    54,    96,   169,    95,    85,   135,    94,
      85,    87,   134,    86,   184,    83,    83,    83,    83,    56,
     129,    86,    86,    78,   186,    96,    86,    94,    55,    56,
      97,   173,    37,   171,     1,    42,    43,    44,    45,    46,
      47,    48,    73,    76,    94,   177,   189,   194,   196,   186,
      96,    82,   200,    85,   200,    54,   201,   202,    76,    56,
     193,   200,    76,   190,   196,   175,    20,   188,   186,   175,
     198,    54,   198,   186,   203,    78,    76,   196,   191,   196,
     177,   198,    42,    43,    44,    46,    47,    48,    73,   177,
     192,   194,   195,    77,   190,   178,    87,   189,    85,   187,
      74,    86,    83,   199,   175,   202,    77,   190,    77,   190,
     175,   199,   200,    85,   200,    76,   193,   200,    76,   175,
      77,   192,    95,    95,    55,     6,    68,    69,    86,   117,
     180,   183,   185,   177,   198,   200,    76,   196,   204,    77,
     178,    76,   196,   175,    54,   175,   191,   177,   175,   192,
     178,    96,    75,    78,    86,   175,    74,   198,   190,   186,
      95,   190,    49,   197,    74,    86,   199,    77,   175,   199,
      77,    95,    79,   117,   176,   182,   185,   178,   198,    75,
      77,    77,    76,   196,   175,   200,    76,   196,   178,    76,
     196,    96,   181,    96,   175,    79,    96,   199,   198,   197,
     190,    74,   175,   190,    95,   190,   197,    80,    82,    85,
      88,    89,    79,    86,   181,    94,    76,   196,    78,    77,
     175,    75,    77,    77,   181,    55,   181,    80,    96,   181,
      80,   190,   198,   199,   175,   197,    83,    86,    86,    96,
      80,    77,   199,    76,   196,    78,    76,   196,   190,   175,
     190,    77,   199,    77,    76,   196,   190,    77
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    90,    91,    92,    92,    93,    93,    94,    94,    95,
      95,    96,    96,    96,    96,    96,    96,    96,    96,    96,
      96,    96,    96,    96,    96,    96,    96,    96,    96,    96,
      96,    96,    96,    96,    96,    96,    96,    96,    96,    96,
      96,    96,    96,    96,    96,    96,    96,    96,    96,    96,
      96,    96,    96,    96,    96,    96,    96,    96,    97,    97,
      98,    98,    99,    99,   100,   100,   101,   101,   101,   101,
     101,   102,   102,   102,   102,   102,   102,   102,   102,   102,
     102,   102,   102,   102,   102,   103,   103,   103,   104,   104,
     105,   105,   106,   106,   107,   107,   107,   107,   107,   107,
     107,   107,   107,   107,   107,   107,   107,   107,   107,   108,
     109,   110,   110,   111,   112,   112,   113,   114,   114,   114,
     114,   114,   114,   115,   115,   115,   115,   115,   116,   116,
     117,   117,   118,   119,   120,   120,   121,   122,   123,   123,
     124,   124,   125,   125,   126,   126,   127,   127,   128,   128,
     129,   129,   130,   131,   131,   132,   132,   133,   133,   134,
     134,   135,   135,   136,   137,   137,   138,   138,   139,   139,
     140,   140,   141,   141,   142,   143,   144,   144,   145,   145,
     146,   146,   147,   148,   149,   150,   150,   151,   151,   152,
     152,   152,   152,   153,   153,   153,   154,   154,   155,   156,
     156,   156,   156,   156,   157,   157,   158,   158,   159,   159,
     159,   159,   159,   159,   159,   160,   160,   160,   160,   160,
     161,   161,   161,   161,   162,   162,   163,   164,   165,   165,
     165,   165,   166,   166,   166,   166,   166,   166,   166,   166,
     166,   166,   166,   167,   167,   167,   168,   168,   169,   170,
     170,   170,   171,   171,   172,   172,   172,   172,   172,   172,
     172,   172,   172,   172,   172,   172,   172,   172,   172,   172,
     172,   172,   173,   173,   173,   174,   174,   174,   175,   175,
     175,   175,   175,   175,   176,   177,   178,   179,   179,   179,
     179,   180,   180,   180,   181,   181,   181,   181,   181,   181,
     182,   183,   183,   183,   184,   184,   185,   185,   186,   186,
     187,   187,   188,   188,   189,   189,   189,   190,   190,   191,
     191,   192,   192,   192,   193,   193,   194,   194,   194,   195,
     195,   195,   195,   195,   195,   195,   195,   195,   195,   195,
     195,   196,   196,   196,   196,   196,   196,   196,   196,   196,
     196,   196,   196,   196,   196,   197,   197,   197,   198,   199,
     200,   201,   201,   202,   202,   203,   204,   205,   206
};

  /* YYR2[YYN] -- Number of symbols on the right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     0,     2,     0,     1,     1,     2,     0,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     4,
       3,     3,     1,     4,     0,     2,     3,     2,     2,     2,
       7,     5,     5,     2,     2,     2,     2,     2,     2,     2,
       2,     1,     1,     1,     1,     1,     1,     1,     1,     3,
       0,     1,     0,     3,     1,     1,     1,     1,     2,     2,
       3,     3,     2,     2,     2,     1,     1,     2,     1,     2,
       2,     1,     1,     2,     2,     2,     8,     1,     1,     1,
       1,     2,     2,     1,     1,     1,     2,     2,     2,     1,
       2,     1,     1,     3,     0,     2,     4,     6,     0,     1,
       0,     3,     1,     3,     1,     1,     0,     3,     1,     3,
       0,     1,     1,     0,     3,     1,     3,     1,     1,     0,
       1,     0,     2,     5,     1,     2,     3,     6,     0,     2,
       1,     3,     5,     5,     5,     5,     4,     3,     6,     6,
       5,     5,     5,     5,     5,     4,     7,     0,     2,     0,
       2,     2,     2,     3,     2,     3,     1,     3,     4,     2,
       2,     2,     2,     2,     1,     4,     0,     2,     1,     1,
       1,     1,     2,     2,     2,     3,     6,     9,     3,     6,
       3,     6,     9,     9,     1,     3,     1,     1,     1,     2,
       2,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     7,     5,    13,     5,     2,     1,     0,
       3,     1,     1,     3,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     2,     1,     1,
       1,     1,     1,     1,     1,     0,     1,     3,     0,     1,
       5,     5,     5,     4,     3,     1,     1,     1,     3,     4,
       3,     1,     1,     1,     1,     4,     3,     4,     4,     4,
       3,     7,     5,     6,     1,     3,     1,     3,     3,     2,
       3,     2,     0,     3,     1,     1,     4,     1,     2,     1,
       2,     1,     2,     1,     1,     0,     4,     3,     5,     6,
       4,     4,    11,     9,    12,    14,     6,     8,     5,     7,
       4,     6,     4,     1,     4,    11,     9,    12,    14,     6,
       8,     5,     7,     4,     1,     0,     2,     4,     1,     1,
       1,     2,     5,     1,     3,     1,     1,     2,     2
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
#line 193 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.modlist) = (yyvsp[0].modlist); modlist = (yyvsp[0].modlist); }
#line 2225 "y.tab.c" /* yacc.c:1646  */
    break;

  case 3:
#line 197 "xi-grammar.y" /* yacc.c:1646  */
    { 
		  (yyval.modlist) = 0; 
		}
#line 2233 "y.tab.c" /* yacc.c:1646  */
    break;

  case 4:
#line 201 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.modlist) = new AstChildren<Module>(lineno, (yyvsp[-1].module), (yyvsp[0].modlist)); }
#line 2239 "y.tab.c" /* yacc.c:1646  */
    break;

  case 5:
#line 205 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 2245 "y.tab.c" /* yacc.c:1646  */
    break;

  case 6:
#line 207 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 2251 "y.tab.c" /* yacc.c:1646  */
    break;

  case 7:
#line 211 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 2257 "y.tab.c" /* yacc.c:1646  */
    break;

  case 8:
#line 213 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 2; }
#line 2263 "y.tab.c" /* yacc.c:1646  */
    break;

  case 9:
#line 217 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 2269 "y.tab.c" /* yacc.c:1646  */
    break;

  case 10:
#line 219 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 2275 "y.tab.c" /* yacc.c:1646  */
    break;

  case 11:
#line 224 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 2281 "y.tab.c" /* yacc.c:1646  */
    break;

  case 12:
#line 225 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MODULE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2287 "y.tab.c" /* yacc.c:1646  */
    break;

  case 13:
#line 226 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MAINMODULE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2293 "y.tab.c" /* yacc.c:1646  */
    break;

  case 14:
#line 227 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(EXTERN, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2299 "y.tab.c" /* yacc.c:1646  */
    break;

  case 15:
#line 229 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITCALL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2305 "y.tab.c" /* yacc.c:1646  */
    break;

  case 16:
#line 230 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITNODE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2311 "y.tab.c" /* yacc.c:1646  */
    break;

  case 17:
#line 231 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITPROC, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2317 "y.tab.c" /* yacc.c:1646  */
    break;

  case 18:
#line 233 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CHARE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2323 "y.tab.c" /* yacc.c:1646  */
    break;

  case 19:
#line 234 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MAINCHARE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2329 "y.tab.c" /* yacc.c:1646  */
    break;

  case 20:
#line 235 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(GROUP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2335 "y.tab.c" /* yacc.c:1646  */
    break;

  case 21:
#line 236 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NODEGROUP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2341 "y.tab.c" /* yacc.c:1646  */
    break;

  case 22:
#line 237 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ARRAY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2347 "y.tab.c" /* yacc.c:1646  */
    break;

  case 23:
#line 241 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INCLUDE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2353 "y.tab.c" /* yacc.c:1646  */
    break;

  case 24:
#line 242 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(STACKSIZE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2359 "y.tab.c" /* yacc.c:1646  */
    break;

  case 25:
#line 243 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(THREADED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2365 "y.tab.c" /* yacc.c:1646  */
    break;

  case 26:
#line 244 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(TEMPLATE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2371 "y.tab.c" /* yacc.c:1646  */
    break;

  case 27:
#line 245 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(SYNC, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2377 "y.tab.c" /* yacc.c:1646  */
    break;

  case 28:
#line 246 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IGET, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2383 "y.tab.c" /* yacc.c:1646  */
    break;

  case 29:
#line 247 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(EXCLUSIVE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2389 "y.tab.c" /* yacc.c:1646  */
    break;

  case 30:
#line 248 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IMMEDIATE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2395 "y.tab.c" /* yacc.c:1646  */
    break;

  case 31:
#line 249 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(SKIPSCHED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2401 "y.tab.c" /* yacc.c:1646  */
    break;

  case 32:
#line 250 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INLINE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2407 "y.tab.c" /* yacc.c:1646  */
    break;

  case 33:
#line 251 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(VIRTUAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2413 "y.tab.c" /* yacc.c:1646  */
    break;

  case 34:
#line 252 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MIGRATABLE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2419 "y.tab.c" /* yacc.c:1646  */
    break;

  case 35:
#line 253 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CREATEHERE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2425 "y.tab.c" /* yacc.c:1646  */
    break;

  case 36:
#line 254 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CREATEHOME, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2431 "y.tab.c" /* yacc.c:1646  */
    break;

  case 37:
#line 255 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOKEEP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2437 "y.tab.c" /* yacc.c:1646  */
    break;

  case 38:
#line 256 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOTRACE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2443 "y.tab.c" /* yacc.c:1646  */
    break;

  case 39:
#line 257 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(APPWORK, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2449 "y.tab.c" /* yacc.c:1646  */
    break;

  case 40:
#line 260 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(PACKED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2455 "y.tab.c" /* yacc.c:1646  */
    break;

  case 41:
#line 261 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(VARSIZE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2461 "y.tab.c" /* yacc.c:1646  */
    break;

  case 42:
#line 262 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ENTRY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2467 "y.tab.c" /* yacc.c:1646  */
    break;

  case 43:
#line 263 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(FOR, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2473 "y.tab.c" /* yacc.c:1646  */
    break;

  case 44:
#line 264 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(FORALL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2479 "y.tab.c" /* yacc.c:1646  */
    break;

  case 45:
#line 265 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(WHILE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2485 "y.tab.c" /* yacc.c:1646  */
    break;

  case 46:
#line 266 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(WHEN, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2491 "y.tab.c" /* yacc.c:1646  */
    break;

  case 47:
#line 267 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(OVERLAP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2497 "y.tab.c" /* yacc.c:1646  */
    break;

  case 48:
#line 268 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ATOMIC, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2503 "y.tab.c" /* yacc.c:1646  */
    break;

  case 49:
#line 269 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IF, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2509 "y.tab.c" /* yacc.c:1646  */
    break;

  case 50:
#line 270 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ELSE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2515 "y.tab.c" /* yacc.c:1646  */
    break;

  case 51:
#line 272 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(LOCAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2521 "y.tab.c" /* yacc.c:1646  */
    break;

  case 52:
#line 274 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(USING, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2527 "y.tab.c" /* yacc.c:1646  */
    break;

  case 53:
#line 275 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ACCEL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2533 "y.tab.c" /* yacc.c:1646  */
    break;

  case 54:
#line 278 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ACCELBLOCK, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2539 "y.tab.c" /* yacc.c:1646  */
    break;

  case 55:
#line 279 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MEMCRITICAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2545 "y.tab.c" /* yacc.c:1646  */
    break;

  case 56:
#line 280 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(REDUCTIONTARGET, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2551 "y.tab.c" /* yacc.c:1646  */
    break;

  case 57:
#line 281 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CASE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2557 "y.tab.c" /* yacc.c:1646  */
    break;

  case 58:
#line 285 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 2563 "y.tab.c" /* yacc.c:1646  */
    break;

  case 59:
#line 287 "xi-grammar.y" /* yacc.c:1646  */
    {
		  char *tmp = new char[strlen((yyvsp[-3].strval))+strlen((yyvsp[0].strval))+3];
		  sprintf(tmp,"%s::%s", (yyvsp[-3].strval), (yyvsp[0].strval));
		  (yyval.strval) = tmp;
		}
#line 2573 "y.tab.c" /* yacc.c:1646  */
    break;

  case 60:
#line 294 "xi-grammar.y" /* yacc.c:1646  */
    { 
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist)); 
		}
#line 2581 "y.tab.c" /* yacc.c:1646  */
    break;

  case 61:
#line 298 "xi-grammar.y" /* yacc.c:1646  */
    {  
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist)); 
		    (yyval.module)->setMain();
		}
#line 2590 "y.tab.c" /* yacc.c:1646  */
    break;

  case 62:
#line 305 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = 0; }
#line 2596 "y.tab.c" /* yacc.c:1646  */
    break;

  case 63:
#line 307 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = (yyvsp[-2].conslist); }
#line 2602 "y.tab.c" /* yacc.c:1646  */
    break;

  case 64:
#line 311 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = 0; }
#line 2608 "y.tab.c" /* yacc.c:1646  */
    break;

  case 65:
#line 313 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = new ConstructList(lineno, (yyvsp[-1].construct), (yyvsp[0].conslist)); }
#line 2614 "y.tab.c" /* yacc.c:1646  */
    break;

  case 66:
#line 317 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new UsingScope((yyvsp[0].strval), false); }
#line 2620 "y.tab.c" /* yacc.c:1646  */
    break;

  case 67:
#line 319 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new UsingScope((yyvsp[0].strval), true); }
#line 2626 "y.tab.c" /* yacc.c:1646  */
    break;

  case 68:
#line 321 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].member)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].member); }
#line 2632 "y.tab.c" /* yacc.c:1646  */
    break;

  case 69:
#line 323 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].message)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].message); }
#line 2638 "y.tab.c" /* yacc.c:1646  */
    break;

  case 70:
#line 325 "xi-grammar.y" /* yacc.c:1646  */
    {
                  Entry *e = new Entry(lineno, 0, (yyvsp[-4].type), (yyvsp[-2].strval), (yyvsp[0].plist), 0, 0, 0, (yylsp[-6]).first_line, (yyloc).last_line);
                  int isExtern = 1;
                  e->setExtern(isExtern);
                  e->targs = (yyvsp[-1].tparlist);
                  e->label = new XStr;
                  (yyvsp[-3].ntype)->print(*e->label);
                  (yyval.construct) = e;
                }
#line 2652 "y.tab.c" /* yacc.c:1646  */
    break;

  case 71:
#line 337 "xi-grammar.y" /* yacc.c:1646  */
    { if((yyvsp[-2].conslist)) (yyvsp[-2].conslist)->recurse<int&>((yyvsp[-4].intval), &Construct::setExtern); (yyval.construct) = (yyvsp[-2].conslist); }
#line 2658 "y.tab.c" /* yacc.c:1646  */
    break;

  case 72:
#line 339 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new Scope((yyvsp[-3].strval), (yyvsp[-1].conslist)); }
#line 2664 "y.tab.c" /* yacc.c:1646  */
    break;

  case 73:
#line 341 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = (yyvsp[-1].construct); }
#line 2670 "y.tab.c" /* yacc.c:1646  */
    break;

  case 74:
#line 343 "xi-grammar.y" /* yacc.c:1646  */
    {
          ERROR("preceding construct must be semicolon terminated",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 2680 "y.tab.c" /* yacc.c:1646  */
    break;

  case 75:
#line 349 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].module)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].module); }
#line 2686 "y.tab.c" /* yacc.c:1646  */
    break;

  case 76:
#line 351 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2692 "y.tab.c" /* yacc.c:1646  */
    break;

  case 77:
#line 353 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2698 "y.tab.c" /* yacc.c:1646  */
    break;

  case 78:
#line 355 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2704 "y.tab.c" /* yacc.c:1646  */
    break;

  case 79:
#line 357 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2710 "y.tab.c" /* yacc.c:1646  */
    break;

  case 80:
#line 359 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].templat)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].templat); }
#line 2716 "y.tab.c" /* yacc.c:1646  */
    break;

  case 81:
#line 361 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = NULL; }
#line 2722 "y.tab.c" /* yacc.c:1646  */
    break;

  case 82:
#line 363 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = NULL; }
#line 2728 "y.tab.c" /* yacc.c:1646  */
    break;

  case 83:
#line 365 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = (yyvsp[0].accelBlock); }
#line 2734 "y.tab.c" /* yacc.c:1646  */
    break;

  case 84:
#line 367 "xi-grammar.y" /* yacc.c:1646  */
    {
          ERROR("invalid construct",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 2744 "y.tab.c" /* yacc.c:1646  */
    break;

  case 85:
#line 375 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamType((yyvsp[0].type)); }
#line 2750 "y.tab.c" /* yacc.c:1646  */
    break;

  case 86:
#line 377 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
#line 2756 "y.tab.c" /* yacc.c:1646  */
    break;

  case 87:
#line 379 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
#line 2762 "y.tab.c" /* yacc.c:1646  */
    break;

  case 88:
#line 383 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = new TParamList((yyvsp[0].tparam)); }
#line 2768 "y.tab.c" /* yacc.c:1646  */
    break;

  case 89:
#line 385 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = new TParamList((yyvsp[-2].tparam), (yyvsp[0].tparlist)); }
#line 2774 "y.tab.c" /* yacc.c:1646  */
    break;

  case 90:
#line 389 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = 0; }
#line 2780 "y.tab.c" /* yacc.c:1646  */
    break;

  case 91:
#line 391 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = (yyvsp[0].tparlist); }
#line 2786 "y.tab.c" /* yacc.c:1646  */
    break;

  case 92:
#line 395 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = 0; }
#line 2792 "y.tab.c" /* yacc.c:1646  */
    break;

  case 93:
#line 397 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = (yyvsp[-1].tparlist); }
#line 2798 "y.tab.c" /* yacc.c:1646  */
    break;

  case 94:
#line 401 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("int"); }
#line 2804 "y.tab.c" /* yacc.c:1646  */
    break;

  case 95:
#line 403 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long"); }
#line 2810 "y.tab.c" /* yacc.c:1646  */
    break;

  case 96:
#line 405 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("short"); }
#line 2816 "y.tab.c" /* yacc.c:1646  */
    break;

  case 97:
#line 407 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("char"); }
#line 2822 "y.tab.c" /* yacc.c:1646  */
    break;

  case 98:
#line 409 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned int"); }
#line 2828 "y.tab.c" /* yacc.c:1646  */
    break;

  case 99:
#line 411 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long"); }
#line 2834 "y.tab.c" /* yacc.c:1646  */
    break;

  case 100:
#line 413 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long"); }
#line 2840 "y.tab.c" /* yacc.c:1646  */
    break;

  case 101:
#line 415 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long long"); }
#line 2846 "y.tab.c" /* yacc.c:1646  */
    break;

  case 102:
#line 417 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned short"); }
#line 2852 "y.tab.c" /* yacc.c:1646  */
    break;

  case 103:
#line 419 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned char"); }
#line 2858 "y.tab.c" /* yacc.c:1646  */
    break;

  case 104:
#line 421 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long long"); }
#line 2864 "y.tab.c" /* yacc.c:1646  */
    break;

  case 105:
#line 423 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("float"); }
#line 2870 "y.tab.c" /* yacc.c:1646  */
    break;

  case 106:
#line 425 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("double"); }
#line 2876 "y.tab.c" /* yacc.c:1646  */
    break;

  case 107:
#line 427 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long double"); }
#line 2882 "y.tab.c" /* yacc.c:1646  */
    break;

  case 108:
#line 429 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("void"); }
#line 2888 "y.tab.c" /* yacc.c:1646  */
    break;

  case 109:
#line 432 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ntype) = new NamedType((yyvsp[-1].strval),(yyvsp[0].tparlist)); }
#line 2894 "y.tab.c" /* yacc.c:1646  */
    break;

  case 110:
#line 433 "xi-grammar.y" /* yacc.c:1646  */
    { 
                    const char* basename, *scope;
                    splitScopedName((yyvsp[-1].strval), &scope, &basename);
                    (yyval.ntype) = new NamedType(basename, (yyvsp[0].tparlist), scope);
                }
#line 2904 "y.tab.c" /* yacc.c:1646  */
    break;

  case 111:
#line 441 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 2910 "y.tab.c" /* yacc.c:1646  */
    break;

  case 112:
#line 443 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ntype); }
#line 2916 "y.tab.c" /* yacc.c:1646  */
    break;

  case 113:
#line 447 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ptype) = new PtrType((yyvsp[-1].type)); }
#line 2922 "y.tab.c" /* yacc.c:1646  */
    break;

  case 114:
#line 451 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
#line 2928 "y.tab.c" /* yacc.c:1646  */
    break;

  case 115:
#line 453 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
#line 2934 "y.tab.c" /* yacc.c:1646  */
    break;

  case 116:
#line 457 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ftype) = new FuncType((yyvsp[-7].type), (yyvsp[-4].strval), (yyvsp[-1].plist)); }
#line 2940 "y.tab.c" /* yacc.c:1646  */
    break;

  case 117:
#line 461 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 2946 "y.tab.c" /* yacc.c:1646  */
    break;

  case 118:
#line 463 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 2952 "y.tab.c" /* yacc.c:1646  */
    break;

  case 119:
#line 465 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 2958 "y.tab.c" /* yacc.c:1646  */
    break;

  case 120:
#line 467 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ftype); }
#line 2964 "y.tab.c" /* yacc.c:1646  */
    break;

  case 121:
#line 469 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[0].type)); }
#line 2970 "y.tab.c" /* yacc.c:1646  */
    break;

  case 122:
#line 471 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[-1].type)); }
#line 2976 "y.tab.c" /* yacc.c:1646  */
    break;

  case 123:
#line 475 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 2982 "y.tab.c" /* yacc.c:1646  */
    break;

  case 124:
#line 477 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 2988 "y.tab.c" /* yacc.c:1646  */
    break;

  case 125:
#line 479 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 2994 "y.tab.c" /* yacc.c:1646  */
    break;

  case 126:
#line 481 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[0].type)); }
#line 3000 "y.tab.c" /* yacc.c:1646  */
    break;

  case 127:
#line 483 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[-1].type)); }
#line 3006 "y.tab.c" /* yacc.c:1646  */
    break;

  case 128:
#line 487 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ReferenceType((yyvsp[-1].type)); }
#line 3012 "y.tab.c" /* yacc.c:1646  */
    break;

  case 129:
#line 489 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3018 "y.tab.c" /* yacc.c:1646  */
    break;

  case 130:
#line 493 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ReferenceType((yyvsp[-1].type)); }
#line 3024 "y.tab.c" /* yacc.c:1646  */
    break;

  case 131:
#line 495 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3030 "y.tab.c" /* yacc.c:1646  */
    break;

  case 132:
#line 499 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 3036 "y.tab.c" /* yacc.c:1646  */
    break;

  case 133:
#line 503 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = (yyvsp[-1].val); }
#line 3042 "y.tab.c" /* yacc.c:1646  */
    break;

  case 134:
#line 507 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.vallist) = 0; }
#line 3048 "y.tab.c" /* yacc.c:1646  */
    break;

  case 135:
#line 509 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.vallist) = new ValueList((yyvsp[-1].val), (yyvsp[0].vallist)); }
#line 3054 "y.tab.c" /* yacc.c:1646  */
    break;

  case 136:
#line 513 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].vallist)); }
#line 3060 "y.tab.c" /* yacc.c:1646  */
    break;

  case 137:
#line 517 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[-3].type), (yyvsp[-1].strval), (yyvsp[0].vallist), 1); }
#line 3066 "y.tab.c" /* yacc.c:1646  */
    break;

  case 138:
#line 521 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0;}
#line 3072 "y.tab.c" /* yacc.c:1646  */
    break;

  case 139:
#line 523 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0;}
#line 3078 "y.tab.c" /* yacc.c:1646  */
    break;

  case 140:
#line 527 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3084 "y.tab.c" /* yacc.c:1646  */
    break;

  case 141:
#line 529 "xi-grammar.y" /* yacc.c:1646  */
    { 
		  /*
		  printf("Warning: Message attributes are being phased out.\n");
		  printf("Warning: Please remove them from interface files.\n");
		  */
		  (yyval.intval) = (yyvsp[-1].intval); 
		}
#line 3096 "y.tab.c" /* yacc.c:1646  */
    break;

  case 142:
#line 539 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[0].intval); }
#line 3102 "y.tab.c" /* yacc.c:1646  */
    break;

  case 143:
#line 541 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[-2].intval) | (yyvsp[0].intval); }
#line 3108 "y.tab.c" /* yacc.c:1646  */
    break;

  case 144:
#line 545 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3114 "y.tab.c" /* yacc.c:1646  */
    break;

  case 145:
#line 547 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3120 "y.tab.c" /* yacc.c:1646  */
    break;

  case 146:
#line 551 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = 0; }
#line 3126 "y.tab.c" /* yacc.c:1646  */
    break;

  case 147:
#line 553 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-1].cattr); }
#line 3132 "y.tab.c" /* yacc.c:1646  */
    break;

  case 148:
#line 557 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[0].cattr); }
#line 3138 "y.tab.c" /* yacc.c:1646  */
    break;

  case 149:
#line 559 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
#line 3144 "y.tab.c" /* yacc.c:1646  */
    break;

  case 150:
#line 563 "xi-grammar.y" /* yacc.c:1646  */
    { python_doc = NULL; (yyval.intval) = 0; }
#line 3150 "y.tab.c" /* yacc.c:1646  */
    break;

  case 151:
#line 565 "xi-grammar.y" /* yacc.c:1646  */
    { python_doc = (yyvsp[0].strval); (yyval.intval) = 0; }
#line 3156 "y.tab.c" /* yacc.c:1646  */
    break;

  case 152:
#line 569 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CPYTHON; }
#line 3162 "y.tab.c" /* yacc.c:1646  */
    break;

  case 153:
#line 573 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = 0; }
#line 3168 "y.tab.c" /* yacc.c:1646  */
    break;

  case 154:
#line 575 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-1].cattr); }
#line 3174 "y.tab.c" /* yacc.c:1646  */
    break;

  case 155:
#line 579 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[0].cattr); }
#line 3180 "y.tab.c" /* yacc.c:1646  */
    break;

  case 156:
#line 581 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
#line 3186 "y.tab.c" /* yacc.c:1646  */
    break;

  case 157:
#line 585 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CMIGRATABLE; }
#line 3192 "y.tab.c" /* yacc.c:1646  */
    break;

  case 158:
#line 587 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CPYTHON; }
#line 3198 "y.tab.c" /* yacc.c:1646  */
    break;

  case 159:
#line 591 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3204 "y.tab.c" /* yacc.c:1646  */
    break;

  case 160:
#line 593 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 3210 "y.tab.c" /* yacc.c:1646  */
    break;

  case 161:
#line 596 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3216 "y.tab.c" /* yacc.c:1646  */
    break;

  case 162:
#line 598 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 3222 "y.tab.c" /* yacc.c:1646  */
    break;

  case 163:
#line 601 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mv) = new MsgVar((yyvsp[-3].type), (yyvsp[-2].strval), (yyvsp[-4].intval), (yyvsp[-1].intval)); }
#line 3228 "y.tab.c" /* yacc.c:1646  */
    break;

  case 164:
#line 605 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mvlist) = new MsgVarList((yyvsp[0].mv)); }
#line 3234 "y.tab.c" /* yacc.c:1646  */
    break;

  case 165:
#line 607 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mvlist) = new MsgVarList((yyvsp[-1].mv), (yyvsp[0].mvlist)); }
#line 3240 "y.tab.c" /* yacc.c:1646  */
    break;

  case 166:
#line 611 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, (yyvsp[0].ntype)); }
#line 3246 "y.tab.c" /* yacc.c:1646  */
    break;

  case 167:
#line 613 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, (yyvsp[-3].ntype), (yyvsp[-1].mvlist)); }
#line 3252 "y.tab.c" /* yacc.c:1646  */
    break;

  case 168:
#line 617 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = 0; }
#line 3258 "y.tab.c" /* yacc.c:1646  */
    break;

  case 169:
#line 619 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = (yyvsp[0].typelist); }
#line 3264 "y.tab.c" /* yacc.c:1646  */
    break;

  case 170:
#line 623 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = new TypeList((yyvsp[0].ntype)); }
#line 3270 "y.tab.c" /* yacc.c:1646  */
    break;

  case 171:
#line 625 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = new TypeList((yyvsp[-2].ntype), (yyvsp[0].typelist)); }
#line 3276 "y.tab.c" /* yacc.c:1646  */
    break;

  case 172:
#line 629 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr)|Chare::CCHARE, (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3282 "y.tab.c" /* yacc.c:1646  */
    break;

  case 173:
#line 631 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3288 "y.tab.c" /* yacc.c:1646  */
    break;

  case 174:
#line 635 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3294 "y.tab.c" /* yacc.c:1646  */
    break;

  case 175:
#line 639 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new NodeGroup(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3300 "y.tab.c" /* yacc.c:1646  */
    break;

  case 176:
#line 643 "xi-grammar.y" /* yacc.c:1646  */
    {/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",(yyvsp[-2].strval));
			(yyval.ntype) = new NamedType(buf); 
		}
#line 3310 "y.tab.c" /* yacc.c:1646  */
    break;

  case 177:
#line 649 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ntype) = new NamedType((yyvsp[-1].strval)); }
#line 3316 "y.tab.c" /* yacc.c:1646  */
    break;

  case 178:
#line 653 "xi-grammar.y" /* yacc.c:1646  */
    {  (yyval.chare) = new Array(lineno, (yyvsp[-4].cattr), (yyvsp[-3].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3322 "y.tab.c" /* yacc.c:1646  */
    break;

  case 179:
#line 655 "xi-grammar.y" /* yacc.c:1646  */
    {  (yyval.chare) = new Array(lineno, (yyvsp[-3].cattr), (yyvsp[-4].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3328 "y.tab.c" /* yacc.c:1646  */
    break;

  case 180:
#line 659 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr)|Chare::CCHARE, new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist));}
#line 3334 "y.tab.c" /* yacc.c:1646  */
    break;

  case 181:
#line 661 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3340 "y.tab.c" /* yacc.c:1646  */
    break;

  case 182:
#line 665 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3346 "y.tab.c" /* yacc.c:1646  */
    break;

  case 183:
#line 669 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new NodeGroup( lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3352 "y.tab.c" /* yacc.c:1646  */
    break;

  case 184:
#line 673 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Array( lineno, 0, (yyvsp[-3].ntype), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3358 "y.tab.c" /* yacc.c:1646  */
    break;

  case 185:
#line 677 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-1].strval))); }
#line 3364 "y.tab.c" /* yacc.c:1646  */
    break;

  case 186:
#line 679 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-4].strval)), (yyvsp[-2].mvlist)); }
#line 3370 "y.tab.c" /* yacc.c:1646  */
    break;

  case 187:
#line 683 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = 0; }
#line 3376 "y.tab.c" /* yacc.c:1646  */
    break;

  case 188:
#line 685 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3382 "y.tab.c" /* yacc.c:1646  */
    break;

  case 189:
#line 689 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = 0; }
#line 3388 "y.tab.c" /* yacc.c:1646  */
    break;

  case 190:
#line 691 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3394 "y.tab.c" /* yacc.c:1646  */
    break;

  case 191:
#line 693 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3400 "y.tab.c" /* yacc.c:1646  */
    break;

  case 192:
#line 695 "xi-grammar.y" /* yacc.c:1646  */
    {
		  XStr typeStr;
		  (yyvsp[0].ntype)->print(typeStr);
		  char *tmp = strdup(typeStr.get_string());
		  (yyval.strval) = tmp;
		}
#line 3411 "y.tab.c" /* yacc.c:1646  */
    break;

  case 193:
#line 704 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TType(new NamedType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3417 "y.tab.c" /* yacc.c:1646  */
    break;

  case 194:
#line 706 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TFunc((yyvsp[-1].ftype), (yyvsp[0].strval)); }
#line 3423 "y.tab.c" /* yacc.c:1646  */
    break;

  case 195:
#line 708 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TName((yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].strval)); }
#line 3429 "y.tab.c" /* yacc.c:1646  */
    break;

  case 196:
#line 712 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = new TVarList((yyvsp[0].tvar)); }
#line 3435 "y.tab.c" /* yacc.c:1646  */
    break;

  case 197:
#line 714 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = new TVarList((yyvsp[-2].tvar), (yyvsp[0].tvarlist)); }
#line 3441 "y.tab.c" /* yacc.c:1646  */
    break;

  case 198:
#line 718 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = (yyvsp[-1].tvarlist); }
#line 3447 "y.tab.c" /* yacc.c:1646  */
    break;

  case 199:
#line 722 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3453 "y.tab.c" /* yacc.c:1646  */
    break;

  case 200:
#line 724 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3459 "y.tab.c" /* yacc.c:1646  */
    break;

  case 201:
#line 726 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3465 "y.tab.c" /* yacc.c:1646  */
    break;

  case 202:
#line 728 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3471 "y.tab.c" /* yacc.c:1646  */
    break;

  case 203:
#line 730 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].message)); (yyvsp[0].message)->setTemplate((yyval.templat)); }
#line 3477 "y.tab.c" /* yacc.c:1646  */
    break;

  case 204:
#line 734 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = 0; }
#line 3483 "y.tab.c" /* yacc.c:1646  */
    break;

  case 205:
#line 736 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = (yyvsp[-2].mbrlist); }
#line 3489 "y.tab.c" /* yacc.c:1646  */
    break;

  case 206:
#line 740 "xi-grammar.y" /* yacc.c:1646  */
    { 
                  if (!connectEntries.empty()) {
                    (yyval.mbrlist) = new AstChildren<Member>(connectEntries);
		  } else {
		    (yyval.mbrlist) = 0; 
                  }
		}
#line 3501 "y.tab.c" /* yacc.c:1646  */
    break;

  case 207:
#line 748 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = new AstChildren<Member>(-1, (yyvsp[-1].member), (yyvsp[0].mbrlist)); }
#line 3507 "y.tab.c" /* yacc.c:1646  */
    break;

  case 208:
#line 752 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].readonly); }
#line 3513 "y.tab.c" /* yacc.c:1646  */
    break;

  case 209:
#line 754 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].readonly); }
#line 3519 "y.tab.c" /* yacc.c:1646  */
    break;

  case 211:
#line 757 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].member); }
#line 3525 "y.tab.c" /* yacc.c:1646  */
    break;

  case 212:
#line 759 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].pupable); }
#line 3531 "y.tab.c" /* yacc.c:1646  */
    break;

  case 213:
#line 761 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].includeFile); }
#line 3537 "y.tab.c" /* yacc.c:1646  */
    break;

  case 214:
#line 763 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new ClassDeclaration(lineno,(yyvsp[0].strval)); }
#line 3543 "y.tab.c" /* yacc.c:1646  */
    break;

  case 215:
#line 767 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1); }
#line 3549 "y.tab.c" /* yacc.c:1646  */
    break;

  case 216:
#line 769 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1); }
#line 3555 "y.tab.c" /* yacc.c:1646  */
    break;

  case 217:
#line 771 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[-6].strval)) + '<' +
					    ((yyvsp[-4].tparlist))->to_string() + '>').c_str()),
				    1);
		}
#line 3565 "y.tab.c" /* yacc.c:1646  */
    break;

  case 218:
#line 777 "xi-grammar.y" /* yacc.c:1646  */
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[-2]).first_column, (yylsp[-2]).last_column, (yylsp[-2]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1);
		}
#line 3575 "y.tab.c" /* yacc.c:1646  */
    break;

  case 219:
#line 783 "xi-grammar.y" /* yacc.c:1646  */
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[-5]).first_column, (yylsp[-5]).last_column, (yylsp[-5]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1);
		}
#line 3585 "y.tab.c" /* yacc.c:1646  */
    break;

  case 220:
#line 792 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 0); }
#line 3591 "y.tab.c" /* yacc.c:1646  */
    break;

  case 221:
#line 794 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 0); }
#line 3597 "y.tab.c" /* yacc.c:1646  */
    break;

  case 222:
#line 796 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[-6].strval)) + '<' +
					    ((yyvsp[-4].tparlist))->to_string() + '>').c_str()),
				    0);
		}
#line 3607 "y.tab.c" /* yacc.c:1646  */
    break;

  case 223:
#line 802 "xi-grammar.y" /* yacc.c:1646  */
    {
                  InitCall* rtn = new InitCall(lineno, (yyvsp[-3].strval), 0);
                  rtn->setAccel();
                  (yyval.member) = rtn;
		}
#line 3617 "y.tab.c" /* yacc.c:1646  */
    break;

  case 224:
#line 810 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[0].ntype),0); }
#line 3623 "y.tab.c" /* yacc.c:1646  */
    break;

  case 225:
#line 812 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[-2].ntype),(yyvsp[0].pupable)); }
#line 3629 "y.tab.c" /* yacc.c:1646  */
    break;

  case 226:
#line 815 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.includeFile) = new IncludeFile(lineno,(yyvsp[0].strval)); }
#line 3635 "y.tab.c" /* yacc.c:1646  */
    break;

  case 227:
#line 819 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].member); }
#line 3641 "y.tab.c" /* yacc.c:1646  */
    break;

  case 228:
#line 823 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].entry); }
#line 3647 "y.tab.c" /* yacc.c:1646  */
    break;

  case 229:
#line 825 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyvsp[0].entry)->tspec = (yyvsp[-1].tvarlist);
                  (yyval.member) = (yyvsp[0].entry);
                }
#line 3656 "y.tab.c" /* yacc.c:1646  */
    break;

  case 230:
#line 830 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[-1].member); }
#line 3662 "y.tab.c" /* yacc.c:1646  */
    break;

  case 231:
#line 832 "xi-grammar.y" /* yacc.c:1646  */
    {
          ERROR("invalid SDAG member",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 3672 "y.tab.c" /* yacc.c:1646  */
    break;

  case 232:
#line 840 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3678 "y.tab.c" /* yacc.c:1646  */
    break;

  case 233:
#line 842 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3684 "y.tab.c" /* yacc.c:1646  */
    break;

  case 234:
#line 844 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3690 "y.tab.c" /* yacc.c:1646  */
    break;

  case 235:
#line 846 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3696 "y.tab.c" /* yacc.c:1646  */
    break;

  case 236:
#line 848 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3702 "y.tab.c" /* yacc.c:1646  */
    break;

  case 237:
#line 850 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3708 "y.tab.c" /* yacc.c:1646  */
    break;

  case 238:
#line 852 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3714 "y.tab.c" /* yacc.c:1646  */
    break;

  case 239:
#line 854 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3720 "y.tab.c" /* yacc.c:1646  */
    break;

  case 240:
#line 856 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3726 "y.tab.c" /* yacc.c:1646  */
    break;

  case 241:
#line 858 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3732 "y.tab.c" /* yacc.c:1646  */
    break;

  case 242:
#line 860 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3738 "y.tab.c" /* yacc.c:1646  */
    break;

  case 243:
#line 863 "xi-grammar.y" /* yacc.c:1646  */
    { 
                  (yyval.entry) = new Entry(lineno, (yyvsp[-5].intval), (yyvsp[-4].type), (yyvsp[-3].strval), (yyvsp[-2].plist), (yyvsp[-1].val), (yyvsp[0].sentry), (const char *) NULL, (yylsp[-6]).first_line, (yyloc).last_line);
		  if ((yyvsp[0].sentry) != 0) { 
		    (yyvsp[0].sentry)->con1 = new SdagConstruct(SIDENT, (yyvsp[-3].strval));
                    (yyvsp[0].sentry)->setEntry((yyval.entry));
                    (yyvsp[0].sentry)->param = new ParamList((yyvsp[-2].plist));
                  }
		}
#line 3751 "y.tab.c" /* yacc.c:1646  */
    break;

  case 244:
#line 872 "xi-grammar.y" /* yacc.c:1646  */
    { 
                  Entry *e = new Entry(lineno, (yyvsp[-3].intval), 0, (yyvsp[-2].strval), (yyvsp[-1].plist),  0, (yyvsp[0].sentry), (const char *) NULL, (yylsp[-4]).first_line, (yyloc).last_line);
                  if ((yyvsp[0].sentry) != 0) {
		    (yyvsp[0].sentry)->con1 = new SdagConstruct(SIDENT, (yyvsp[-2].strval));
                    (yyvsp[0].sentry)->setEntry((yyval.entry));
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
#line 3771 "y.tab.c" /* yacc.c:1646  */
    break;

  case 245:
#line 888 "xi-grammar.y" /* yacc.c:1646  */
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
#line 3789 "y.tab.c" /* yacc.c:1646  */
    break;

  case 246:
#line 904 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.accelBlock) = new AccelBlock(lineno, new XStr((yyvsp[-2].strval))); }
#line 3795 "y.tab.c" /* yacc.c:1646  */
    break;

  case 247:
#line 906 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.accelBlock) = new AccelBlock(lineno, NULL); }
#line 3801 "y.tab.c" /* yacc.c:1646  */
    break;

  case 248:
#line 910 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3807 "y.tab.c" /* yacc.c:1646  */
    break;

  case 249:
#line 914 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3813 "y.tab.c" /* yacc.c:1646  */
    break;

  case 250:
#line 916 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[-1].intval); }
#line 3819 "y.tab.c" /* yacc.c:1646  */
    break;

  case 251:
#line 918 "xi-grammar.y" /* yacc.c:1646  */
    { ERROR("invalid entry method attribute list",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
#line 3828 "y.tab.c" /* yacc.c:1646  */
    break;

  case 252:
#line 925 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[0].intval); }
#line 3834 "y.tab.c" /* yacc.c:1646  */
    break;

  case 253:
#line 927 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[-2].intval) | (yyvsp[0].intval); }
#line 3840 "y.tab.c" /* yacc.c:1646  */
    break;

  case 254:
#line 931 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = STHREADED; }
#line 3846 "y.tab.c" /* yacc.c:1646  */
    break;

  case 255:
#line 933 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SSYNC; }
#line 3852 "y.tab.c" /* yacc.c:1646  */
    break;

  case 256:
#line 935 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SIGET; }
#line 3858 "y.tab.c" /* yacc.c:1646  */
    break;

  case 257:
#line 937 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SLOCKED; }
#line 3864 "y.tab.c" /* yacc.c:1646  */
    break;

  case 258:
#line 939 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SCREATEHERE; }
#line 3870 "y.tab.c" /* yacc.c:1646  */
    break;

  case 259:
#line 941 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SCREATEHOME; }
#line 3876 "y.tab.c" /* yacc.c:1646  */
    break;

  case 260:
#line 943 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SNOKEEP; }
#line 3882 "y.tab.c" /* yacc.c:1646  */
    break;

  case 261:
#line 945 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SNOTRACE; }
#line 3888 "y.tab.c" /* yacc.c:1646  */
    break;

  case 262:
#line 947 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SAPPWORK; }
#line 3894 "y.tab.c" /* yacc.c:1646  */
    break;

  case 263:
#line 949 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SIMMEDIATE; }
#line 3900 "y.tab.c" /* yacc.c:1646  */
    break;

  case 264:
#line 951 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SSKIPSCHED; }
#line 3906 "y.tab.c" /* yacc.c:1646  */
    break;

  case 265:
#line 953 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SINLINE; }
#line 3912 "y.tab.c" /* yacc.c:1646  */
    break;

  case 266:
#line 955 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SLOCAL; }
#line 3918 "y.tab.c" /* yacc.c:1646  */
    break;

  case 267:
#line 957 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SPYTHON; }
#line 3924 "y.tab.c" /* yacc.c:1646  */
    break;

  case 268:
#line 959 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SMEM; }
#line 3930 "y.tab.c" /* yacc.c:1646  */
    break;

  case 269:
#line 961 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SREDUCE; }
#line 3936 "y.tab.c" /* yacc.c:1646  */
    break;

  case 270:
#line 963 "xi-grammar.y" /* yacc.c:1646  */
    {
#ifdef CMK_USING_XLC
        WARNING("a known bug in xl compilers (PMR 18366,122,000) currently breaks "
                "aggregate entry methods.\n"
                "Until a fix is released, this tag will be ignored on those compilers.",
                (yylsp[0]).first_column, (yylsp[0]).last_column, (yylsp[0]).first_line);
        (yyval.intval) = 0;
#else
        (yyval.intval) = SAGGREGATE;
#endif
    }
#line 3952 "y.tab.c" /* yacc.c:1646  */
    break;

  case 271:
#line 975 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("invalid entry method attribute",
		        (yylsp[0]).first_column, (yylsp[0]).last_column);
		  yyclearin;
		  yyerrok;
		}
#line 3963 "y.tab.c" /* yacc.c:1646  */
    break;

  case 272:
#line 984 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 3969 "y.tab.c" /* yacc.c:1646  */
    break;

  case 273:
#line 986 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 3975 "y.tab.c" /* yacc.c:1646  */
    break;

  case 274:
#line 988 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 3981 "y.tab.c" /* yacc.c:1646  */
    break;

  case 275:
#line 992 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = ""; }
#line 3987 "y.tab.c" /* yacc.c:1646  */
    break;

  case 276:
#line 994 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3993 "y.tab.c" /* yacc.c:1646  */
    break;

  case 277:
#line 996 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s, %s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4003 "y.tab.c" /* yacc.c:1646  */
    break;

  case 278:
#line 1004 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = ""; }
#line 4009 "y.tab.c" /* yacc.c:1646  */
    break;

  case 279:
#line 1006 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4015 "y.tab.c" /* yacc.c:1646  */
    break;

  case 280:
#line 1008 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s[%s]%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4025 "y.tab.c" /* yacc.c:1646  */
    break;

  case 281:
#line 1014 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s{%s}%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4035 "y.tab.c" /* yacc.c:1646  */
    break;

  case 282:
#line 1020 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s(%s)%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4045 "y.tab.c" /* yacc.c:1646  */
    break;

  case 283:
#line 1026 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"(%s)%s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4055 "y.tab.c" /* yacc.c:1646  */
    break;

  case 284:
#line 1034 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval));
		}
#line 4064 "y.tab.c" /* yacc.c:1646  */
    break;

  case 285:
#line 1041 "xi-grammar.y" /* yacc.c:1646  */
    { 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			(yyval.intval) = 0;
		}
#line 4074 "y.tab.c" /* yacc.c:1646  */
    break;

  case 286:
#line 1049 "xi-grammar.y" /* yacc.c:1646  */
    { 
			in_braces=0;
			(yyval.intval) = 0;
		}
#line 4083 "y.tab.c" /* yacc.c:1646  */
    break;

  case 287:
#line 1056 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[0].type));}
#line 4089 "y.tab.c" /* yacc.c:1646  */
    break;

  case 288:
#line 1058 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval)); (yyval.pname)->setConditional((yyvsp[0].intval)); }
#line 4095 "y.tab.c" /* yacc.c:1646  */
    break;

  case 289:
#line 1060 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[-3].type),(yyvsp[-2].strval),0,(yyvsp[0].val));}
#line 4101 "y.tab.c" /* yacc.c:1646  */
    break;

  case 290:
#line 1062 "xi-grammar.y" /* yacc.c:1646  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
		}
#line 4110 "y.tab.c" /* yacc.c:1646  */
    break;

  case 291:
#line 1068 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READONLY; }
#line 4116 "y.tab.c" /* yacc.c:1646  */
    break;

  case 292:
#line 1069 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READWRITE; }
#line 4122 "y.tab.c" /* yacc.c:1646  */
    break;

  case 293:
#line 1070 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_WRITEONLY; }
#line 4128 "y.tab.c" /* yacc.c:1646  */
    break;

  case 294:
#line 1073 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr((yyvsp[0].strval)); }
#line 4134 "y.tab.c" /* yacc.c:1646  */
    break;

  case 295:
#line 1074 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "->" << (yyvsp[0].strval); }
#line 4140 "y.tab.c" /* yacc.c:1646  */
    break;

  case 296:
#line 1075 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[-2].xstrptr)) << "." << (yyvsp[0].strval); }
#line 4146 "y.tab.c" /* yacc.c:1646  */
    break;

  case 297:
#line 1077 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << *((yyvsp[-1].xstrptr)) << "]";
                  delete (yyvsp[-3].xstrptr);
                  delete (yyvsp[-1].xstrptr);
                }
#line 4157 "y.tab.c" /* yacc.c:1646  */
    break;

  case 298:
#line 1084 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << (yyvsp[-1].strval) << "]";
                  delete (yyvsp[-3].xstrptr);
                }
#line 4167 "y.tab.c" /* yacc.c:1646  */
    break;

  case 299:
#line 1090 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "(" << *((yyvsp[-1].xstrptr)) << ")";
                  delete (yyvsp[-3].xstrptr);
                  delete (yyvsp[-1].xstrptr);
                }
#line 4178 "y.tab.c" /* yacc.c:1646  */
    break;

  case 300:
#line 1099 "xi-grammar.y" /* yacc.c:1646  */
    {
                  in_bracket = 0;
                  (yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName(), (yyvsp[-1].strval));
                }
#line 4187 "y.tab.c" /* yacc.c:1646  */
    break;

  case 301:
#line 1106 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[-6].intval));
                }
#line 4197 "y.tab.c" /* yacc.c:1646  */
    break;

  case 302:
#line 1112 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType(Parameter::ACCEL_BUFFER_TYPE_READWRITE);
		}
#line 4207 "y.tab.c" /* yacc.c:1646  */
    break;

  case 303:
#line 1118 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.pname) = (yyvsp[-3].pname);
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[-5].intval));
		}
#line 4217 "y.tab.c" /* yacc.c:1646  */
    break;

  case 304:
#line 1126 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 4223 "y.tab.c" /* yacc.c:1646  */
    break;

  case 305:
#line 1128 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 4229 "y.tab.c" /* yacc.c:1646  */
    break;

  case 306:
#line 1132 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 4235 "y.tab.c" /* yacc.c:1646  */
    break;

  case 307:
#line 1134 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 4241 "y.tab.c" /* yacc.c:1646  */
    break;

  case 308:
#line 1138 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = (yyvsp[-1].plist); }
#line 4247 "y.tab.c" /* yacc.c:1646  */
    break;

  case 309:
#line 1140 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList(new Parameter(0, new BuiltinType("void"))); }
#line 4253 "y.tab.c" /* yacc.c:1646  */
    break;

  case 310:
#line 1144 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = (yyvsp[-1].plist); }
#line 4259 "y.tab.c" /* yacc.c:1646  */
    break;

  case 311:
#line 1146 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = 0; }
#line 4265 "y.tab.c" /* yacc.c:1646  */
    break;

  case 312:
#line 1150 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = 0; }
#line 4271 "y.tab.c" /* yacc.c:1646  */
    break;

  case 313:
#line 1152 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4277 "y.tab.c" /* yacc.c:1646  */
    break;

  case 314:
#line 1156 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = 0; }
#line 4283 "y.tab.c" /* yacc.c:1646  */
    break;

  case 315:
#line 1158 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[0].sc)); }
#line 4289 "y.tab.c" /* yacc.c:1646  */
    break;

  case 316:
#line 1160 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[-2].slist)); }
#line 4295 "y.tab.c" /* yacc.c:1646  */
    break;

  case 317:
#line 1164 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.slist) = new SListConstruct((yyvsp[0].sc)); }
#line 4301 "y.tab.c" /* yacc.c:1646  */
    break;

  case 318:
#line 1166 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.slist) = new SListConstruct((yyvsp[-1].sc), (yyvsp[0].slist));  }
#line 4307 "y.tab.c" /* yacc.c:1646  */
    break;

  case 319:
#line 1170 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.olist) = new OListConstruct((yyvsp[0].sc)); }
#line 4313 "y.tab.c" /* yacc.c:1646  */
    break;

  case 320:
#line 1172 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.olist) = new OListConstruct((yyvsp[-1].sc), (yyvsp[0].slist)); }
#line 4319 "y.tab.c" /* yacc.c:1646  */
    break;

  case 321:
#line 1176 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.clist) = new CaseListConstruct((yyvsp[0].when)); }
#line 4325 "y.tab.c" /* yacc.c:1646  */
    break;

  case 322:
#line 1178 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.clist) = new CaseListConstruct((yyvsp[-1].when), (yyvsp[0].clist)); }
#line 4331 "y.tab.c" /* yacc.c:1646  */
    break;

  case 323:
#line 1180 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("case blocks can only contain when clauses",
		        (yylsp[0]).first_column, (yylsp[0]).last_column);
		  (yyval.clist) = 0;
		}
#line 4341 "y.tab.c" /* yacc.c:1646  */
    break;

  case 324:
#line 1188 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4347 "y.tab.c" /* yacc.c:1646  */
    break;

  case 325:
#line 1190 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = 0; }
#line 4353 "y.tab.c" /* yacc.c:1646  */
    break;

  case 326:
#line 1194 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-2].entrylist), 0); }
#line 4359 "y.tab.c" /* yacc.c:1646  */
    break;

  case 327:
#line 1196 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-1].entrylist), (yyvsp[0].sc)); }
#line 4365 "y.tab.c" /* yacc.c:1646  */
    break;

  case 328:
#line 1198 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-3].entrylist), (yyvsp[-1].slist)); }
#line 4371 "y.tab.c" /* yacc.c:1646  */
    break;

  case 329:
#line 1202 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4377 "y.tab.c" /* yacc.c:1646  */
    break;

  case 330:
#line 1204 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4383 "y.tab.c" /* yacc.c:1646  */
    break;

  case 331:
#line 1206 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4389 "y.tab.c" /* yacc.c:1646  */
    break;

  case 332:
#line 1208 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4395 "y.tab.c" /* yacc.c:1646  */
    break;

  case 333:
#line 1210 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4401 "y.tab.c" /* yacc.c:1646  */
    break;

  case 334:
#line 1212 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4407 "y.tab.c" /* yacc.c:1646  */
    break;

  case 335:
#line 1214 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4413 "y.tab.c" /* yacc.c:1646  */
    break;

  case 336:
#line 1216 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4419 "y.tab.c" /* yacc.c:1646  */
    break;

  case 337:
#line 1218 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4425 "y.tab.c" /* yacc.c:1646  */
    break;

  case 338:
#line 1220 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4431 "y.tab.c" /* yacc.c:1646  */
    break;

  case 339:
#line 1222 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4437 "y.tab.c" /* yacc.c:1646  */
    break;

  case 340:
#line 1224 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4443 "y.tab.c" /* yacc.c:1646  */
    break;

  case 341:
#line 1228 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new AtomicConstruct((yyvsp[-2].strval), (yyvsp[-4].strval), (yylsp[-3]).first_line); }
#line 4449 "y.tab.c" /* yacc.c:1646  */
    break;

  case 342:
#line 1230 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new OverlapConstruct((yyvsp[-1].olist)); }
#line 4455 "y.tab.c" /* yacc.c:1646  */
    break;

  case 343:
#line 1232 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = (yyvsp[0].when); }
#line 4461 "y.tab.c" /* yacc.c:1646  */
    break;

  case 344:
#line 1234 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new CaseConstruct((yyvsp[-1].clist)); }
#line 4467 "y.tab.c" /* yacc.c:1646  */
    break;

  case 345:
#line 1236 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForConstruct((yyvsp[-8].intexpr), (yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4473 "y.tab.c" /* yacc.c:1646  */
    break;

  case 346:
#line 1238 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForConstruct((yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4479 "y.tab.c" /* yacc.c:1646  */
    break;

  case 347:
#line 1240 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-9].strval)), (yyvsp[-6].intexpr),
		             (yyvsp[-4].intexpr), (yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4486 "y.tab.c" /* yacc.c:1646  */
    break;

  case 348:
#line 1243 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-11].strval)), (yyvsp[-8].intexpr),
		             (yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4493 "y.tab.c" /* yacc.c:1646  */
    break;

  case 349:
#line 1246 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new IfConstruct((yyvsp[-3].intexpr), (yyvsp[-1].sc), (yyvsp[0].sc)); }
#line 4499 "y.tab.c" /* yacc.c:1646  */
    break;

  case 350:
#line 1248 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new IfConstruct((yyvsp[-5].intexpr), (yyvsp[-2].slist), (yyvsp[0].sc)); }
#line 4505 "y.tab.c" /* yacc.c:1646  */
    break;

  case 351:
#line 1250 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new WhileConstruct((yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4511 "y.tab.c" /* yacc.c:1646  */
    break;

  case 352:
#line 1252 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new WhileConstruct((yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4517 "y.tab.c" /* yacc.c:1646  */
    break;

  case 353:
#line 1254 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new AtomicConstruct((yyvsp[-2].strval), NULL, (yyloc).first_line); }
#line 4523 "y.tab.c" /* yacc.c:1646  */
    break;

  case 354:
#line 1256 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("unknown SDAG construct or malformed entry method declaration.\n"
		        "You may have forgotten to terminate a previous entry method declaration with a"
		        " semicolon or forgotten to mark a block of sequential SDAG code as 'serial'",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
#line 4535 "y.tab.c" /* yacc.c:1646  */
    break;

  case 355:
#line 1266 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = 0; }
#line 4541 "y.tab.c" /* yacc.c:1646  */
    break;

  case 356:
#line 1268 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ElseConstruct((yyvsp[0].sc)); }
#line 4547 "y.tab.c" /* yacc.c:1646  */
    break;

  case 357:
#line 1270 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ElseConstruct((yyvsp[-1].slist)); }
#line 4553 "y.tab.c" /* yacc.c:1646  */
    break;

  case 358:
#line 1274 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intexpr) = new IntExprConstruct((yyvsp[0].strval)); }
#line 4559 "y.tab.c" /* yacc.c:1646  */
    break;

  case 359:
#line 1278 "xi-grammar.y" /* yacc.c:1646  */
    { in_int_expr = 0; (yyval.intval) = 0; }
#line 4565 "y.tab.c" /* yacc.c:1646  */
    break;

  case 360:
#line 1282 "xi-grammar.y" /* yacc.c:1646  */
    { in_int_expr = 1; (yyval.intval) = 0; }
#line 4571 "y.tab.c" /* yacc.c:1646  */
    break;

  case 361:
#line 1286 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[-1].strval), (yyvsp[0].plist), 0, 0, 0, (yyloc).first_line, (yyloc).last_line);
		}
#line 4579 "y.tab.c" /* yacc.c:1646  */
    break;

  case 362:
#line 1290 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[-4].strval), (yyvsp[0].plist), 0, 0, (yyvsp[-2].strval), (yyloc).first_line, (yyloc).last_line);
		}
#line 4587 "y.tab.c" /* yacc.c:1646  */
    break;

  case 363:
#line 1296 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.entrylist) = new EntryList((yyvsp[0].entry)); }
#line 4593 "y.tab.c" /* yacc.c:1646  */
    break;

  case 364:
#line 1298 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.entrylist) = new EntryList((yyvsp[-2].entry),(yyvsp[0].entrylist)); }
#line 4599 "y.tab.c" /* yacc.c:1646  */
    break;

  case 365:
#line 1302 "xi-grammar.y" /* yacc.c:1646  */
    { in_bracket=1; }
#line 4605 "y.tab.c" /* yacc.c:1646  */
    break;

  case 366:
#line 1305 "xi-grammar.y" /* yacc.c:1646  */
    { in_bracket=0; }
#line 4611 "y.tab.c" /* yacc.c:1646  */
    break;

  case 367:
#line 1309 "xi-grammar.y" /* yacc.c:1646  */
    { if (!macroDefined((yyvsp[0].strval), 1)) in_comment = 1; }
#line 4617 "y.tab.c" /* yacc.c:1646  */
    break;

  case 368:
#line 1313 "xi-grammar.y" /* yacc.c:1646  */
    { if (!macroDefined((yyvsp[0].strval), 0)) in_comment = 1; }
#line 4623 "y.tab.c" /* yacc.c:1646  */
    break;


#line 4627 "y.tab.c" /* yacc.c:1646  */
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
#line 1316 "xi-grammar.y" /* yacc.c:1906  */


void yyerror(const char *msg) { }
