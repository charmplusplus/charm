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
extern char* yytext;
AstChildren<Module> *modlist;

void yyerror(const char *);

namespace xi {

const int MAX_NUM_ERRORS = 10;
int num_errors = 0;
bool firstRdma = true;

bool enable_warnings = true;

extern int macroDefined(const char *str, int istrue);
extern const char *python_doc;
extern char *fname;
void splitScopedName(const char* name, const char** scope, const char** basename);
void ReservedWord(int token, int fCol, int lCol);
}

#line 115 "y.tab.c" /* yacc.c:339  */

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
    NOCOPY = 294,
    PACKED = 295,
    VARSIZE = 296,
    ENTRY = 297,
    FOR = 298,
    FORALL = 299,
    WHILE = 300,
    WHEN = 301,
    OVERLAP = 302,
    SERIAL = 303,
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
#define NOCOPY 294
#define PACKED 295
#define VARSIZE 296
#define ENTRY 297
#define FOR 298
#define FORALL 299
#define WHILE 300
#define WHEN 301
#define OVERLAP 302
#define SERIAL 303
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
#define CASE 329

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED

union YYSTYPE
{
#line 53 "xi-grammar.y" /* yacc.c:355  */

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

#line 347 "y.tab.c" /* yacc.c:355  */
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

#line 378 "y.tab.c" /* yacc.c:358  */

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
#define YYFINAL  56
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   1490

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  91
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  117
/* YYNRULES -- Number of rules.  */
#define YYNRULES  374
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  729

/* YYTRANSLATE[YYX] -- Symbol number corresponding to YYX as returned
   by yylex, with out-of-bounds checking.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   329

#define YYTRANSLATE(YYX)                                                \
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, without out-of-bounds checking.  */
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
  /* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   195,   195,   200,   203,   208,   209,   213,   215,   220,
     221,   226,   228,   229,   230,   232,   233,   234,   236,   237,
     238,   239,   240,   244,   245,   246,   247,   248,   249,   250,
     251,   252,   253,   254,   255,   256,   257,   258,   259,   260,
     261,   264,   265,   266,   267,   268,   269,   270,   271,   272,
     273,   274,   276,   278,   279,   282,   283,   284,   285,   289,
     291,   297,   304,   308,   315,   317,   322,   323,   327,   329,
     331,   333,   335,   348,   350,   352,   354,   360,   362,   364,
     366,   368,   370,   372,   374,   376,   378,   386,   388,   390,
     394,   396,   401,   402,   407,   408,   412,   414,   416,   418,
     420,   422,   424,   426,   428,   430,   432,   434,   436,   438,
     440,   444,   445,   452,   454,   458,   462,   464,   468,   472,
     474,   476,   478,   480,   482,   486,   488,   490,   492,   494,
     498,   500,   502,   506,   508,   510,   514,   518,   523,   524,
     528,   532,   537,   538,   543,   544,   554,   556,   560,   562,
     567,   568,   572,   574,   579,   580,   584,   589,   590,   594,
     596,   600,   602,   607,   608,   612,   613,   616,   620,   622,
     626,   628,   630,   635,   636,   640,   642,   646,   648,   652,
     656,   660,   666,   670,   672,   676,   678,   682,   686,   690,
     694,   696,   701,   702,   707,   708,   710,   712,   721,   723,
     725,   729,   731,   735,   739,   741,   743,   745,   747,   751,
     753,   758,   765,   769,   771,   773,   774,   776,   778,   780,
     784,   786,   788,   794,   800,   809,   811,   813,   819,   827,
     829,   832,   836,   840,   842,   847,   849,   857,   859,   861,
     863,   865,   867,   869,   871,   873,   875,   877,   880,   890,
     907,   924,   926,   930,   935,   936,   938,   945,   947,   951,
     953,   955,   957,   959,   961,   963,   965,   967,   969,   971,
     973,   975,   977,   979,   981,   983,   987,   996,   998,  1000,
    1005,  1006,  1008,  1017,  1018,  1020,  1026,  1032,  1038,  1046,
    1053,  1061,  1068,  1070,  1072,  1074,  1079,  1091,  1092,  1093,
    1096,  1097,  1098,  1099,  1106,  1112,  1121,  1128,  1134,  1140,
    1148,  1150,  1154,  1156,  1160,  1162,  1166,  1168,  1173,  1174,
    1178,  1180,  1182,  1186,  1188,  1192,  1194,  1198,  1200,  1202,
    1210,  1213,  1216,  1218,  1220,  1224,  1226,  1228,  1230,  1232,
    1234,  1236,  1238,  1240,  1242,  1244,  1246,  1250,  1252,  1254,
    1256,  1258,  1260,  1262,  1265,  1268,  1270,  1272,  1274,  1276,
    1278,  1289,  1290,  1292,  1296,  1300,  1304,  1308,  1313,  1320,
    1322,  1326,  1329,  1333,  1337
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
  "VOID", "CONST", "NOCOPY", "PACKED", "VARSIZE", "ENTRY", "FOR", "FORALL",
  "WHILE", "WHEN", "OVERLAP", "SERIAL", "IF", "ELSE", "PYTHON", "LOCAL",
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
     325,   326,   327,   328,   329,    59,    58,   123,   125,    44,
      60,    62,    42,    40,    41,    38,    91,    93,    61,    45,
      46
};
# endif

#define YYPACT_NINF -630

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-630)))

#define YYTABLE_NINF -326

#define yytable_value_is_error(Yytable_value) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
     139,  1303,  1303,    54,  -630,   139,  -630,  -630,  -630,  -630,
    -630,  -630,  -630,  -630,  -630,  -630,  -630,  -630,  -630,  -630,
    -630,  -630,  -630,  -630,  -630,  -630,  -630,  -630,  -630,  -630,
    -630,  -630,  -630,  -630,  -630,  -630,  -630,  -630,  -630,  -630,
    -630,  -630,  -630,  -630,  -630,  -630,  -630,  -630,  -630,  -630,
    -630,  -630,  -630,  -630,   121,   121,  -630,  -630,  -630,   771,
     -36,  -630,  -630,  -630,    81,  1303,   180,  1303,  1303,   131,
     872,   -19,   939,   771,  -630,  -630,  -630,  -630,   607,    51,
      96,  -630,    90,  -630,  -630,  -630,   -36,    62,  1344,   136,
     136,   -16,    96,    99,    99,    99,    99,   107,   123,  1303,
     119,   132,   771,  -630,  -630,  -630,  -630,  -630,  -630,  -630,
    -630,   478,  -630,  -630,  -630,  -630,   154,  -630,  -630,  -630,
    -630,  -630,  -630,  -630,  -630,  -630,  -630,  -630,   -36,  -630,
    -630,  -630,   607,  -630,   -10,  -630,  -630,  -630,  -630,   307,
      91,  -630,  -630,   150,   162,   165,    22,  -630,    96,   771,
      90,   176,    50,    62,   192,   284,  1423,   150,   162,   165,
    -630,    34,    96,  -630,    96,    96,   203,    96,   195,  -630,
      -4,  1303,  1303,  1303,  1303,  1087,   189,   191,   160,  1303,
    -630,  -630,  -630,  1366,   212,    99,    99,    99,    99,   189,
     123,  -630,  -630,  -630,  -630,  -630,   -36,  -630,   260,  -630,
    -630,  -630,   196,  -630,  -630,  1410,  -630,  -630,  -630,  -630,
    -630,   220,  1303,   245,     0,    62,   266,    62,   249,  -630,
     154,   253,    71,  -630,   259,   255,    45,    64,   106,   247,
     111,    96,  -630,  -630,   256,   265,   279,   285,   285,   285,
     285,  -630,  1303,   278,   287,   302,  1159,  1303,   339,  1303,
    -630,  -630,   304,   313,   317,  1303,    76,  1303,   316,   315,
     154,  1303,  1303,  1303,  1303,  1303,  1303,  -630,  -630,  -630,
    -630,   322,  -630,   321,  -630,  -630,   279,  -630,  -630,  -630,
     326,   340,   324,   328,    62,   -36,    96,  1303,  -630,  -630,
     343,  -630,    62,   136,  1410,   136,   136,  1410,   136,  -630,
    -630,    -4,  -630,    96,   163,   163,   163,   163,   346,  -630,
     339,  -630,   285,   285,  -630,   160,     7,   342,   252,  -630,
     348,  1366,  -630,  -630,   285,   285,   285,   285,   285,   178,
    1410,  -630,   354,    62,   266,    62,    62,  -630,    45,   356,
    -630,   351,  -630,   364,   370,   377,    96,   381,   379,  -630,
     385,  -630,   310,   -36,  -630,  -630,  -630,  -630,  -630,  -630,
     163,   163,  -630,  -630,  -630,  1423,    14,   387,  1423,  -630,
    -630,  -630,  -630,  -630,  -630,   163,   163,   163,   163,   163,
     451,   -36,  -630,  1379,  -630,  -630,  -630,  -630,  -630,  -630,
     386,  -630,  -630,  -630,   388,  -630,    85,   390,  -630,    96,
    -630,   687,   428,   396,   154,   310,  -630,  -630,  -630,  -630,
    1303,  -630,  -630,  -630,  -630,  -630,  -630,  -630,  -630,   399,
    1423,  -630,  1303,    62,   400,   397,   361,   136,   136,   136,
    -630,  -630,   906,  1015,  -630,   154,   -36,  -630,   394,   154,
    1303,    62,     1,   395,   361,  -630,   402,   411,   413,   417,
    -630,  -630,  -630,  -630,  -630,  -630,  -630,  -630,  -630,  -630,
    -630,  -630,  -630,  -630,   447,  -630,   418,  -630,  -630,   419,
     429,   426,   354,  1303,  -630,   423,   154,   -36,   425,   427,
    -630,   258,  -630,  -630,  -630,  -630,  -630,  -630,  -630,  -630,
    -630,   475,  -630,   959,   482,   354,  -630,   -36,  -630,  -630,
    -630,    90,  -630,  1303,  -630,  -630,   430,   431,   430,   461,
     441,   464,   430,   446,   248,   -36,    62,  -630,  -630,  -630,
     515,   354,  -630,    62,   481,    62,    39,   458,   498,   519,
    -630,   471,    62,   922,   473,   336,   192,   465,   482,   463,
    -630,   477,   467,   474,  -630,    62,   461,   329,  -630,   483,
     410,    62,   474,   430,   469,   430,   493,   464,   430,   496,
      62,   499,   922,  -630,   154,  -630,   154,   518,  -630,   380,
     471,    62,   430,  -630,   537,   351,  -630,  -630,   500,  -630,
    -630,   192,   575,    62,   533,    62,   519,   471,    62,   922,
     192,  -630,  -630,  -630,  -630,  -630,  -630,  -630,  -630,  -630,
    1303,   513,   512,   505,    62,   520,    62,   248,  -630,   354,
    -630,   154,   248,   544,   522,   511,   474,   523,    62,   474,
     524,   154,   525,  1423,  1328,  -630,   192,    62,   527,   526,
    -630,  -630,   528,   589,  -630,    62,   430,   612,  -630,   192,
     755,  -630,  -630,  1303,  1303,    62,   529,  -630,  1303,   474,
      62,  -630,   544,   248,  -630,   535,    62,   248,  -630,   154,
     248,   544,  -630,    63,   -43,   521,  1303,   154,   762,   536,
    -630,   534,    62,   540,   548,  -630,   549,  -630,  -630,  1303,
    1231,   547,  1303,  1303,  -630,    94,   -36,   248,  -630,    62,
    -630,   474,    62,  -630,   544,   183,   542,   193,  1303,  -630,
     153,  -630,   552,   474,   769,   560,  -630,  -630,  -630,  -630,
    -630,  -630,  -630,   818,   248,  -630,    62,   248,  -630,   562,
     474,   563,  -630,   825,  -630,   248,  -630,   568,  -630
};

  /* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
     Performed when YYTABLE does not specify something else to do.  Zero
     means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       3,     0,     0,     0,     2,     3,    12,    13,    14,    15,
      16,    17,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    31,    33,    34,    35,    36,
      37,    38,    39,    40,    32,    41,    42,    43,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    11,    54,
      55,    56,    57,    58,     0,     0,     1,     4,     7,     0,
      64,    62,    63,    86,     6,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    85,    83,    84,     8,     0,     0,
       0,    59,    69,   373,   374,   290,   252,   283,     0,   142,
     142,   142,     0,   150,   150,   150,   150,     0,   144,     0,
       0,     0,     0,    77,   213,   214,    71,    78,    79,    80,
      81,     0,    82,    70,   216,   215,     9,   247,   239,   240,
     241,   242,   243,   245,   246,   244,   237,   238,    75,    76,
      67,   110,     0,    96,    97,    98,    99,   107,   108,     0,
      94,   113,   114,   125,   126,   127,   132,   253,     0,     0,
      68,     0,   284,   283,     0,     0,     0,   119,   120,   121,
     122,   135,     0,   143,     0,     0,     0,     0,   229,   217,
       0,     0,     0,     0,     0,     0,     0,   157,     0,     0,
     219,   231,   218,     0,     0,   150,   150,   150,   150,     0,
     144,   204,   205,   206,   207,   208,    10,    65,   128,   106,
     109,   100,   101,   104,   105,    92,   112,   115,   116,   117,
     129,   131,     0,     0,     0,   283,   280,   283,     0,   291,
       0,     0,   123,   124,     0,   134,   138,   223,   220,     0,
     225,     0,   161,   162,     0,   152,    94,   173,   173,   173,
     173,   156,     0,     0,   159,     0,     0,     0,     0,     0,
     148,   149,     0,   146,   170,     0,   122,     0,   201,     0,
       9,     0,     0,     0,     0,     0,     0,   102,   103,    88,
      89,    90,    93,     0,    87,   130,    94,    74,    61,    60,
       0,   281,     0,     0,   283,   251,     0,     0,   133,   371,
     138,   140,   283,   142,     0,   142,   142,     0,   142,   230,
     151,     0,   111,     0,     0,     0,     0,     0,     0,   182,
       0,   158,   173,   173,   145,     0,   163,   192,     0,   199,
     194,     0,   203,    73,   173,   173,   173,   173,   173,     0,
       0,    95,     0,   283,   280,   283,   283,   288,   138,     0,
     139,     0,   136,     0,     0,     0,     0,     0,     0,   153,
     175,   174,     0,   209,   177,   178,   179,   180,   181,   160,
       0,     0,   147,   164,   171,     0,   163,     0,     0,   198,
     195,   196,   197,   200,   202,     0,     0,     0,     0,     0,
     163,   190,    91,     0,    72,   286,   282,   287,   285,   141,
       0,   372,   137,   224,     0,   221,     0,     0,   226,     0,
     236,     0,     0,     0,     0,     0,   232,   233,   183,   184,
       0,   169,   172,   193,   185,   186,   187,   188,   189,     0,
       0,   315,   292,   283,   310,     0,     0,   142,   142,   142,
     176,   256,     0,     0,   234,     9,   235,   212,   165,     0,
       0,   283,   163,     0,     0,   314,     0,     0,     0,     0,
     276,   259,   260,   261,   262,   268,   269,   270,   275,   263,
     264,   265,   266,   267,   154,   271,     0,   273,   274,     0,
     257,    59,     0,     0,   210,     0,     0,   191,     0,     0,
     289,     0,   293,   295,   311,   118,   222,   228,   227,   155,
     272,     0,   255,     0,     0,     0,   166,   167,   296,   278,
     277,   279,   294,     0,   258,   360,     0,     0,     0,     0,
       0,   331,     0,     0,     0,   320,   283,   249,   349,   321,
     318,     0,   366,   283,     0,   283,     0,   369,     0,     0,
     330,     0,   283,     0,     0,     0,     0,     0,     0,     0,
     364,     0,     0,     0,   367,   283,     0,     0,   333,     0,
       0,   283,     0,     0,     0,     0,     0,   331,     0,     0,
     283,     0,   327,   329,     9,   324,     9,     0,   248,     0,
       0,   283,     0,   365,     0,     0,   370,   332,     0,   348,
     326,     0,     0,   283,     0,   283,     0,     0,   283,     0,
       0,   350,   328,   322,   359,   319,   297,   298,   299,   317,
       0,     0,   312,     0,   283,     0,   283,     0,   357,     0,
     334,     9,     0,   361,     0,     0,     0,     0,   283,     0,
       0,     9,     0,     0,     0,   316,     0,   283,     0,     0,
     368,   347,     0,     0,   355,   283,     0,     0,   336,     0,
       0,   337,   346,     0,     0,   283,     0,   313,     0,     0,
     283,   358,   361,     0,   362,     0,   283,     0,   344,     9,
       0,   361,   300,     0,     0,     0,     0,     0,     0,     0,
     356,     0,   283,     0,     0,   335,     0,   342,   308,     0,
       0,     0,     0,     0,   306,     0,   250,     0,   352,   283,
     363,     0,   283,   345,   361,     0,     0,     0,     0,   302,
       0,   309,     0,     0,     0,     0,   343,   305,   304,   303,
     301,   307,   351,     0,     0,   339,   283,     0,   353,     0,
       0,     0,   338,     0,   354,     0,   340,     0,   341
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -630,  -630,   595,  -630,   -42,  -254,    -1,   -61,   578,   596,
     -31,  -630,  -630,  -630,  -272,  -630,  -207,  -630,   -77,   -72,
     -75,   -70,   -69,  -173,   494,   532,  -630,   -84,  -630,  -630,
    -244,  -630,  -630,   -74,   485,   338,  -630,    17,   353,  -630,
    -630,   490,   366,  -630,   235,  -630,  -630,  -242,  -630,   -24,
     281,  -630,  -630,  -630,  -141,  -630,  -630,  -630,  -630,  -630,
    -630,  -630,   363,  -630,   372,   614,  -630,   545,   280,   617,
    -630,  -630,   472,  -630,  -630,  -630,  -630,   303,  -630,   271,
    -630,   225,  -630,  -630,   404,   -85,  -413,   -55,  -500,  -630,
    -630,  -453,  -630,  -630,  -310,   116,  -444,  -630,  -630,   205,
    -497,   158,  -528,   188,  -489,  -630,  -445,  -629,  -492,  -541,
    -467,  -630,   200,   221,   182,  -630,  -630
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     3,     4,    70,   353,   197,   236,   140,     5,    61,
      71,    72,    73,   271,   272,   273,   206,   141,   237,   142,
     157,   158,   159,   160,   161,   146,   147,   274,   341,   290,
     291,   104,   105,   164,   179,   252,   253,   171,   234,   490,
     244,   176,   245,   235,   365,   476,   366,   367,   106,   304,
     351,   107,   108,   109,   177,   110,   191,   192,   193,   194,
     195,   369,   319,   258,   259,   402,   112,   354,   403,   404,
     114,   115,   169,   182,   405,   406,   129,   407,    74,   148,
     433,   469,   470,   502,   282,   540,   423,   516,   220,   424,
     601,   663,   646,   602,   425,   603,   384,   570,   538,   517,
     534,   549,   561,   531,   518,   563,   535,   634,   541,   574,
     523,   527,   528,   292,   392,    75,    76
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[] =
{
      54,    55,   154,   143,   162,    82,   323,   441,   144,   145,
     256,   582,    60,    60,    87,   278,   165,   167,   363,   150,
     168,   163,   344,   670,   363,   347,   232,    86,   494,   302,
     128,   363,   677,   543,   592,   247,   566,   683,   565,    77,
     552,   525,   130,   480,   562,   532,   340,   233,   265,   519,
     578,   520,   199,   580,    56,   279,   200,   143,   382,   116,
     210,   620,   144,   145,    79,   706,    83,    84,   218,   332,
     166,   184,   223,   562,   196,   637,   212,   539,   640,   605,
     221,   611,   544,   548,   550,   364,   583,   480,   585,   481,
     621,   588,  -168,   519,   389,   238,   239,   240,   180,   257,
     562,   226,   254,   227,   228,   606,   230,   211,   668,   223,
     629,   172,   173,   174,   628,   632,   446,   224,   213,   225,
     152,   151,   383,    78,   411,   289,   648,   215,   149,   608,
     280,   289,   283,   216,   484,   649,   217,   613,   419,   659,
     151,   550,     1,     2,   678,   153,   679,   293,   256,   680,
     704,    81,   681,   682,   224,  -194,   671,  -194,   669,   168,
     674,   151,   713,   676,   318,   630,   151,   151,   428,   656,
     312,   205,   313,   163,   243,   701,   181,   679,   285,   723,
     680,   474,   151,   681,   682,   170,   294,   151,   654,   295,
     702,   297,   658,   175,   298,   661,    58,   703,    59,   337,
     250,   251,   261,   262,   263,   264,    58,   342,    85,   178,
     645,   276,   183,   685,   305,   306,   307,   719,   196,   343,
     721,   345,   346,   688,   348,   338,   695,   697,   727,    58,
     700,   350,   207,    80,   711,    81,   679,   257,    58,   680,
     352,   308,   681,   682,   208,   243,   372,   209,   385,   505,
     387,   388,   214,    58,   317,   380,   320,   267,   268,   715,
     324,   325,   326,   327,   328,   329,   679,   707,   718,   680,
     219,   229,   681,   682,   231,   246,   679,   248,   726,   680,
     709,   410,   681,   682,   413,   396,   339,   381,   360,   361,
     260,   506,   507,   508,   509,   510,   511,   512,   210,   422,
     375,   376,   377,   378,   379,   275,  -290,    81,   370,   371,
     593,   400,   594,    81,   499,   500,    88,    89,    90,    91,
      92,   131,   513,   277,   281,    85,  -290,   350,    99,   100,
     505,  -290,   101,   284,   296,   286,   440,   505,   443,    81,
     288,   287,   422,   300,   301,   133,   134,   135,   136,   137,
     138,   139,   401,   447,   448,   449,   479,   631,   143,   205,
     422,   303,   436,   144,   145,   309,   310,   642,   201,   202,
     203,   204,   506,   507,   508,   509,   510,   511,   512,   506,
     507,   508,   509,   510,   511,   512,   596,  -290,  -211,   311,
     241,   314,   315,   196,   316,   321,   322,   477,   131,   156,
     420,   330,   331,   513,   333,   675,    85,   577,   335,   438,
     513,   505,  -290,    85,  -323,   336,    81,   131,   156,   334,
     501,   442,   133,   134,   135,   136,   137,   138,   139,   289,
     368,   536,   472,   358,   497,    81,   318,   383,   391,   478,
     390,   133,   134,   135,   136,   137,   138,   139,   393,   597,
     598,   394,   515,   506,   507,   508,   509,   510,   511,   512,
     575,   395,   397,   398,   399,   412,   581,   599,   363,   426,
     401,   427,   495,   429,   435,   590,   551,   439,   560,   444,
     475,   445,   483,   505,   513,   600,   485,    85,  -325,   185,
     186,   187,   188,   189,   190,   486,   515,   487,   614,   505,
     616,   488,   521,   619,   489,   491,   492,   560,   493,   -11,
     496,   480,   503,   522,   498,   604,   526,   524,   529,   626,
     505,   530,   196,   533,   196,   506,   507,   508,   509,   510,
     511,   512,   618,   639,   560,   537,   542,   546,   505,   644,
     600,   506,   507,   508,   509,   510,   511,   512,    85,   569,
     655,   564,   571,   567,   572,   584,   513,    58,   573,   514,
     665,   579,   506,   507,   508,   509,   510,   511,   512,   196,
     586,   673,   513,   589,   595,   547,   505,   591,   610,   196,
     506,   507,   508,   509,   510,   511,   512,   691,   615,   623,
     505,   624,   625,   513,   633,   627,    85,   635,   636,   622,
      57,   638,   641,   650,   651,   643,   652,   705,   684,   666,
     672,   513,   690,   505,   607,   689,   692,   196,   506,   507,
     508,   509,   510,   511,   512,   686,   693,   694,   698,   708,
     712,   720,   506,   507,   508,   509,   510,   511,   512,   716,
     722,   724,   662,   664,   131,   132,   728,   667,   103,   513,
     222,    62,   612,   362,   349,   506,   507,   508,   509,   510,
     511,   512,    81,   513,   198,   662,   653,   249,   133,   134,
     135,   136,   137,   138,   139,   266,   359,   482,   662,   662,
     430,   699,   662,   373,   111,   437,   513,   113,   431,   657,
    -254,  -254,  -254,   374,  -254,  -254,  -254,   710,  -254,  -254,
    -254,  -254,  -254,   299,   473,   434,  -254,  -254,  -254,  -254,
    -254,  -254,  -254,  -254,  -254,  -254,  -254,  -254,   504,  -254,
    -254,  -254,  -254,  -254,  -254,  -254,  -254,  -254,  -254,  -254,
    -254,  -254,  -254,  -254,  -254,  -254,  -254,  -254,   386,  -254,
     647,  -254,  -254,   568,   617,   587,   576,   545,  -254,  -254,
    -254,  -254,  -254,  -254,  -254,  -254,   505,   609,  -254,  -254,
    -254,  -254,     0,   505,     0,     0,     0,     0,     0,     0,
     505,     0,    63,   432,    -5,    -5,    64,    -5,    -5,    -5,
      -5,    -5,    -5,    -5,    -5,    -5,    -5,    -5,     0,    -5,
      -5,     0,     0,    -5,     0,     0,     0,     0,   506,   507,
     508,   509,   510,   511,   512,   506,   507,   508,   509,   510,
     511,   512,   506,   507,   508,   509,   510,   511,   512,   505,
       0,     0,     0,     0,    65,    66,   505,     0,     0,   513,
      67,    68,   660,     0,     0,     0,   513,     0,     0,   687,
       0,     0,    69,   513,     0,     0,   714,     0,    -5,   -66,
     355,   356,   357,     0,     0,     0,     0,     0,     0,     0,
       0,   506,   507,   508,   509,   510,   511,   512,   506,   507,
     508,   509,   510,   511,   512,     1,     2,     0,    88,    89,
      90,    91,    92,    93,    94,    95,    96,    97,    98,     0,
      99,   100,   513,     0,   101,   717,     0,     0,     0,   513,
       0,     0,   725,     0,     0,   408,   409,   450,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     414,   415,   416,   417,   418,     0,     0,   451,     0,   452,
     453,   454,   455,   456,   457,     0,     0,   458,   459,   460,
     461,   462,   463,     0,     0,   117,   118,   119,   120,   102,
     121,   122,   123,   124,   125,     0,     0,   464,   465,     0,
     450,     0,     0,     0,     0,   553,   554,   555,   509,   556,
     557,   558,     0,     0,   466,     0,     0,     0,   467,   468,
     451,   126,   452,   453,   454,   455,   456,   457,     0,     0,
     458,   459,   460,   461,   462,   463,   559,     0,     0,    85,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     464,   465,     0,     0,    58,     0,     0,   127,     6,     7,
       8,     0,     9,    10,    11,     0,    12,    13,    14,    15,
      16,   467,   468,     0,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,     0,    29,    30,    31,
      32,    33,   131,   132,    34,    35,    36,    37,    38,    39,
      40,    41,    42,    43,    44,    45,     0,    46,     0,    47,
     471,     0,     0,     0,     0,     0,   133,   134,   135,   136,
     137,   138,   139,    49,     0,     0,    50,    51,    52,    53,
       6,     7,     8,     0,     9,    10,    11,     0,    12,    13,
      14,    15,    16,     0,     0,     0,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,     0,    29,
      30,    31,    32,    33,     0,     0,    34,    35,    36,    37,
      38,    39,    40,    41,    42,    43,    44,    45,   241,    46,
       0,    47,    48,   242,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    49,     0,     0,    50,    51,
      52,    53,     6,     7,     8,     0,     9,    10,    11,     0,
      12,    13,    14,    15,    16,     0,     0,     0,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
       0,    29,    30,    31,    32,    33,     0,     0,    34,    35,
      36,    37,    38,    39,    40,    41,    42,    43,    44,    45,
       0,    46,     0,    47,    48,   242,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    49,     0,     0,
      50,    51,    52,    53,     6,     7,     8,     0,     9,    10,
      11,     0,    12,    13,    14,    15,    16,     0,     0,     0,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,     0,    29,    30,    31,    32,    33,     0,     0,
      34,    35,    36,    37,    38,    39,    40,    41,    42,    43,
      44,    45,     0,    46,     0,    47,    48,   696,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    49,
       0,     0,    50,    51,    52,    53,     6,     7,     8,     0,
       9,    10,    11,     0,    12,    13,    14,    15,    16,     0,
       0,     0,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,   596,    29,    30,    31,    32,    33,
       0,     0,    34,    35,    36,    37,    38,    39,    40,    41,
      42,    43,    44,    45,     0,    46,     0,    47,    48,     0,
     155,     0,     0,     0,     0,   131,   156,     0,     0,     0,
       0,    49,     0,     0,    50,    51,    52,    53,     0,     0,
       0,   131,   156,    81,   255,     0,     0,     0,     0,   133,
     134,   135,   136,   137,   138,   139,     0,   597,   598,    81,
       0,     0,     0,   131,   156,   133,   134,   135,   136,   137,
     138,   139,     0,     0,     0,     0,   131,   156,   420,     0,
       0,    81,     0,     0,     0,     0,     0,   133,   134,   135,
     136,   137,   138,   139,    81,     0,     0,     0,     0,     0,
     133,   134,   135,   136,   137,   138,   139,   131,   156,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     131,   156,     0,   421,     0,    81,   269,   270,     0,     0,
       0,   133,   134,   135,   136,   137,   138,   139,    81,     0,
       0,     0,     0,     0,   133,   134,   135,   136,   137,   138,
     139
};

static const yytype_int16 yycheck[] =
{
       1,     2,    87,    78,    88,    66,   260,   420,    78,    78,
     183,   552,    54,    55,    69,    15,    90,    91,    17,    80,
      92,    37,   294,   652,    17,   297,    30,    69,   472,   236,
      72,    17,   661,   525,   562,   176,   536,    80,   535,    75,
     532,   508,    73,    86,   533,   512,   290,    51,   189,   494,
     547,   495,    62,   550,     0,    55,    66,   132,   330,    78,
      38,   589,   132,   132,    65,   694,    67,    68,   153,   276,
      86,   102,    38,   562,   116,   616,   148,   521,   619,   571,
     155,   581,   526,   528,   529,    78,   553,    86,   555,    88,
     590,   558,    78,   538,   338,   172,   173,   174,    99,   183,
     589,   162,   179,   164,   165,   572,   167,    85,   649,    38,
     607,    94,    95,    96,   606,   612,   426,    83,   149,    85,
      58,    76,    83,    42,   366,    86,   626,    77,    77,   574,
     215,    86,   217,    83,   444,   627,    86,   582,   380,   639,
      76,   586,     3,     4,    81,    83,    83,    83,   321,    86,
     691,    55,    89,    90,    83,    79,   653,    81,   650,   231,
     657,    76,   703,   660,    88,   609,    76,    76,    83,   636,
     247,    80,   249,    37,   175,    81,    57,    83,   220,   720,
      86,   435,    76,    89,    90,    86,    80,    76,   633,    83,
     687,    80,   637,    86,    83,   640,    75,   689,    77,   284,
      40,    41,   185,   186,   187,   188,    75,   292,    77,    86,
     623,   212,    80,   666,   238,   239,   240,   714,   260,   293,
     717,   295,   296,   668,   298,   286,   679,   680,   725,    75,
     683,   303,    82,    53,    81,    55,    83,   321,    75,    86,
      77,   242,    89,    90,    82,   246,   318,    82,   333,     1,
     335,   336,    76,    75,   255,    77,   257,    61,    62,   704,
     261,   262,   263,   264,   265,   266,    83,    84,   713,    86,
      78,    68,    89,    90,    79,    86,    83,    86,   723,    86,
      87,   365,    89,    90,   368,   346,   287,   329,   312,   313,
      78,    43,    44,    45,    46,    47,    48,    49,    38,   383,
     324,   325,   326,   327,   328,    85,    58,    55,    56,    57,
     564,     1,   566,    55,    56,    57,     6,     7,     8,     9,
      10,    37,    74,    78,    58,    77,    78,   399,    18,    19,
       1,    83,    22,    84,    87,    82,   420,     1,   423,    55,
      85,    82,   426,    87,    79,    61,    62,    63,    64,    65,
      66,    67,    42,   427,   428,   429,   441,   611,   433,    80,
     444,    76,   404,   433,   433,    87,    79,   621,    61,    62,
      63,    64,    43,    44,    45,    46,    47,    48,    49,    43,
      44,    45,    46,    47,    48,    49,     6,    58,    78,    87,
      51,    87,    79,   435,    77,    79,    81,   439,    37,    38,
      39,    79,    81,    74,    78,   659,    77,    78,    84,   410,
      74,     1,    83,    77,    78,    87,    55,    37,    38,    79,
     481,   422,    61,    62,    63,    64,    65,    66,    67,    86,
      88,   516,   433,    87,   476,    55,    88,    83,    87,   440,
      84,    61,    62,    63,    64,    65,    66,    67,    84,    69,
      70,    81,   494,    43,    44,    45,    46,    47,    48,    49,
     545,    84,    81,    84,    79,    78,   551,    87,    17,    83,
      42,    83,   473,    83,    78,   560,   531,    78,   533,    79,
      86,    84,    87,     1,    74,   569,    84,    77,    78,    11,
      12,    13,    14,    15,    16,    84,   538,    84,   583,     1,
     585,    84,   503,   588,    57,    87,    87,   562,    79,    83,
      87,    86,    37,    83,    87,   570,    55,    86,    77,   604,
       1,    57,   564,    77,   566,    43,    44,    45,    46,    47,
      48,    49,   587,   618,   589,    20,    55,    79,     1,   623,
     624,    43,    44,    45,    46,    47,    48,    49,    77,    86,
     635,    78,    75,    88,    87,    86,    74,    75,    84,    77,
     645,    78,    43,    44,    45,    46,    47,    48,    49,   611,
      77,   656,    74,    77,    56,    77,     1,    78,    78,   621,
      43,    44,    45,    46,    47,    48,    49,   672,    55,    76,
       1,    79,    87,    74,    50,    75,    77,    75,    87,   600,
       5,    78,    78,    76,    78,    80,    78,   692,    87,    80,
      75,    74,    78,     1,    77,    79,    76,   659,    43,    44,
      45,    46,    47,    48,    49,   667,    78,    78,    81,    87,
      78,   716,    43,    44,    45,    46,    47,    48,    49,    79,
      78,    78,   643,   644,    37,    38,    78,   648,    70,    74,
     156,    55,    77,   315,   301,    43,    44,    45,    46,    47,
      48,    49,    55,    74,   132,   666,    77,   177,    61,    62,
      63,    64,    65,    66,    67,   190,   310,   442,   679,   680,
     399,   682,   683,   320,    70,   405,    74,    70,     1,    77,
       3,     4,     5,   321,     7,     8,     9,   698,    11,    12,
      13,    14,    15,   231,   433,   402,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,   493,    32,
      33,    34,    35,    36,    37,    38,    39,    40,    41,    42,
      43,    44,    45,    46,    47,    48,    49,    50,   334,    52,
     624,    54,    55,   538,   586,   557,   546,   526,    61,    62,
      63,    64,    65,    66,    67,    68,     1,   575,    71,    72,
      73,    74,    -1,     1,    -1,    -1,    -1,    -1,    -1,    -1,
       1,    -1,     1,    86,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    14,    15,    16,    -1,    18,
      19,    -1,    -1,    22,    -1,    -1,    -1,    -1,    43,    44,
      45,    46,    47,    48,    49,    43,    44,    45,    46,    47,
      48,    49,    43,    44,    45,    46,    47,    48,    49,     1,
      -1,    -1,    -1,    -1,    53,    54,     1,    -1,    -1,    74,
      59,    60,    77,    -1,    -1,    -1,    74,    -1,    -1,    77,
      -1,    -1,    71,    74,    -1,    -1,    77,    -1,    77,    78,
     305,   306,   307,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    43,    44,    45,    46,    47,    48,    49,    43,    44,
      45,    46,    47,    48,    49,     3,     4,    -1,     6,     7,
       8,     9,    10,    11,    12,    13,    14,    15,    16,    -1,
      18,    19,    74,    -1,    22,    77,    -1,    -1,    -1,    74,
      -1,    -1,    77,    -1,    -1,   360,   361,     1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     375,   376,   377,   378,   379,    -1,    -1,    21,    -1,    23,
      24,    25,    26,    27,    28,    -1,    -1,    31,    32,    33,
      34,    35,    36,    -1,    -1,     6,     7,     8,     9,    77,
      11,    12,    13,    14,    15,    -1,    -1,    51,    52,    -1,
       1,    -1,    -1,    -1,    -1,    43,    44,    45,    46,    47,
      48,    49,    -1,    -1,    68,    -1,    -1,    -1,    72,    73,
      21,    42,    23,    24,    25,    26,    27,    28,    -1,    -1,
      31,    32,    33,    34,    35,    36,    74,    -1,    -1,    77,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      51,    52,    -1,    -1,    75,    -1,    -1,    78,     3,     4,
       5,    -1,     7,     8,     9,    -1,    11,    12,    13,    14,
      15,    72,    73,    -1,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    -1,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    -1,    52,    -1,    54,
      55,    -1,    -1,    -1,    -1,    -1,    61,    62,    63,    64,
      65,    66,    67,    68,    -1,    -1,    71,    72,    73,    74,
       3,     4,     5,    -1,     7,     8,     9,    -1,    11,    12,
      13,    14,    15,    -1,    -1,    -1,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    -1,    32,
      33,    34,    35,    36,    -1,    -1,    39,    40,    41,    42,
      43,    44,    45,    46,    47,    48,    49,    50,    51,    52,
      -1,    54,    55,    56,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    68,    -1,    -1,    71,    72,
      73,    74,     3,     4,     5,    -1,     7,     8,     9,    -1,
      11,    12,    13,    14,    15,    -1,    -1,    -1,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      -1,    32,    33,    34,    35,    36,    -1,    -1,    39,    40,
      41,    42,    43,    44,    45,    46,    47,    48,    49,    50,
      -1,    52,    -1,    54,    55,    56,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    68,    -1,    -1,
      71,    72,    73,    74,     3,     4,     5,    -1,     7,     8,
       9,    -1,    11,    12,    13,    14,    15,    -1,    -1,    -1,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    -1,    32,    33,    34,    35,    36,    -1,    -1,
      39,    40,    41,    42,    43,    44,    45,    46,    47,    48,
      49,    50,    -1,    52,    -1,    54,    55,    56,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    68,
      -1,    -1,    71,    72,    73,    74,     3,     4,     5,    -1,
       7,     8,     9,    -1,    11,    12,    13,    14,    15,    -1,
      -1,    -1,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,     6,    32,    33,    34,    35,    36,
      -1,    -1,    39,    40,    41,    42,    43,    44,    45,    46,
      47,    48,    49,    50,    -1,    52,    -1,    54,    55,    -1,
      16,    -1,    -1,    -1,    -1,    37,    38,    -1,    -1,    -1,
      -1,    68,    -1,    -1,    71,    72,    73,    74,    -1,    -1,
      -1,    37,    38,    55,    18,    -1,    -1,    -1,    -1,    61,
      62,    63,    64,    65,    66,    67,    -1,    69,    70,    55,
      -1,    -1,    -1,    37,    38,    61,    62,    63,    64,    65,
      66,    67,    -1,    -1,    -1,    -1,    37,    38,    39,    -1,
      -1,    55,    -1,    -1,    -1,    -1,    -1,    61,    62,    63,
      64,    65,    66,    67,    55,    -1,    -1,    -1,    -1,    -1,
      61,    62,    63,    64,    65,    66,    67,    37,    38,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      37,    38,    -1,    84,    -1,    55,    56,    57,    -1,    -1,
      -1,    61,    62,    63,    64,    65,    66,    67,    55,    -1,
      -1,    -1,    -1,    -1,    61,    62,    63,    64,    65,    66,
      67
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     3,     4,    92,    93,    99,     3,     4,     5,     7,
       8,     9,    11,    12,    13,    14,    15,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    32,
      33,    34,    35,    36,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    52,    54,    55,    68,
      71,    72,    73,    74,    97,    97,     0,    93,    75,    77,
      95,   100,   100,     1,     5,    53,    54,    59,    60,    71,
      94,   101,   102,   103,   169,   206,   207,    75,    42,    97,
      53,    55,    98,    97,    97,    77,    95,   178,     6,     7,
       8,     9,    10,    11,    12,    13,    14,    15,    16,    18,
      19,    22,    77,    99,   122,   123,   139,   142,   143,   144,
     146,   156,   157,   160,   161,   162,    78,     6,     7,     8,
       9,    11,    12,    13,    14,    15,    42,    78,    95,   167,
     101,    37,    38,    61,    62,    63,    64,    65,    66,    67,
      98,   108,   110,   111,   112,   113,   116,   117,   170,    77,
      98,    76,    58,    83,   176,    16,    38,   111,   112,   113,
     114,   115,   118,    37,   124,   124,    86,   124,   110,   163,
      86,   128,   128,   128,   128,    86,   132,   145,    86,   125,
      97,    57,   164,    80,   101,    11,    12,    13,    14,    15,
      16,   147,   148,   149,   150,   151,    95,    96,   116,    62,
      66,    61,    62,    63,    64,    80,   107,    82,    82,    82,
      38,    85,   110,   101,    76,    77,    83,    86,   176,    78,
     179,   111,   115,    38,    83,    85,    98,    98,    98,    68,
      98,    79,    30,    51,   129,   134,    97,   109,   109,   109,
     109,    51,    56,    97,   131,   133,    86,   145,    86,   132,
      40,    41,   126,   127,   109,    18,   114,   118,   154,   155,
      78,   128,   128,   128,   128,   145,   125,    61,    62,    56,
      57,   104,   105,   106,   118,    85,    97,    78,    15,    55,
     176,    58,   175,   176,    84,    95,    82,    82,    85,    86,
     120,   121,   204,    83,    80,    83,    87,    80,    83,   163,
      87,    79,   107,    76,   140,   140,   140,   140,    97,    87,
      79,    87,   109,   109,    87,    79,    77,    97,    88,   153,
      97,    79,    81,    96,    97,    97,    97,    97,    97,    97,
      79,    81,   107,    78,    79,    84,    87,   176,    98,    97,
     121,   119,   176,   124,   105,   124,   124,   105,   124,   129,
     110,   141,    77,    95,   158,   158,   158,   158,    87,   133,
     140,   140,   126,    17,    78,   135,   137,   138,    88,   152,
      56,    57,   110,   153,   155,   140,   140,   140,   140,   140,
      77,    95,   105,    83,   187,   176,   175,   176,   176,   121,
      84,    87,   205,    84,    81,    84,    98,    81,    84,    79,
       1,    42,   156,   159,   160,   165,   166,   168,   158,   158,
     118,   138,    78,   118,   158,   158,   158,   158,   158,   138,
      39,    84,   118,   177,   180,   185,    83,    83,    83,    83,
     141,     1,    86,   171,   168,    78,    95,   159,    97,    78,
     118,   177,    97,   176,    79,    84,   185,   124,   124,   124,
       1,    21,    23,    24,    25,    26,    27,    28,    31,    32,
      33,    34,    35,    36,    51,    52,    68,    72,    73,   172,
     173,    55,    97,   170,    96,    86,   136,    95,    97,   176,
      86,    88,   135,    87,   185,    84,    84,    84,    84,    57,
     130,    87,    87,    79,   187,    97,    87,    95,    87,    56,
      57,    98,   174,    37,   172,     1,    43,    44,    45,    46,
      47,    48,    49,    74,    77,    95,   178,   190,   195,   197,
     187,    97,    83,   201,    86,   201,    55,   202,   203,    77,
      57,   194,   201,    77,   191,   197,   176,    20,   189,   187,
     176,   199,    55,   199,   187,   204,    79,    77,   197,   192,
     197,   178,   199,    43,    44,    45,    47,    48,    49,    74,
     178,   193,   195,   196,    78,   191,   179,    88,   190,    86,
     188,    75,    87,    84,   200,   176,   203,    78,   191,    78,
     191,   176,   200,   201,    86,   201,    77,   194,   201,    77,
     176,    78,   193,    96,    96,    56,     6,    69,    70,    87,
     118,   181,   184,   186,   178,   199,   201,    77,   197,   205,
      78,   179,    77,   197,   176,    55,   176,   192,   178,   176,
     193,   179,    97,    76,    79,    87,   176,    75,   199,   191,
     187,    96,   191,    50,   198,    75,    87,   200,    78,   176,
     200,    78,    96,    80,   118,   177,   183,   186,   179,   199,
      76,    78,    78,    77,   197,   176,   201,    77,   197,   179,
      77,   197,    97,   182,    97,   176,    80,    97,   200,   199,
     198,   191,    75,   176,   191,    96,   191,   198,    81,    83,
      86,    89,    90,    80,    87,   182,    95,    77,   197,    79,
      78,   176,    76,    78,    78,   182,    56,   182,    81,    97,
     182,    81,   191,   199,   200,   176,   198,    84,    87,    87,
      97,    81,    78,   200,    77,   197,    79,    77,   197,   191,
     176,   191,    78,   200,    78,    77,   197,   191,    78
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    91,    92,    93,    93,    94,    94,    95,    95,    96,
      96,    97,    97,    97,    97,    97,    97,    97,    97,    97,
      97,    97,    97,    97,    97,    97,    97,    97,    97,    97,
      97,    97,    97,    97,    97,    97,    97,    97,    97,    97,
      97,    97,    97,    97,    97,    97,    97,    97,    97,    97,
      97,    97,    97,    97,    97,    97,    97,    97,    97,    98,
      98,    98,    99,    99,   100,   100,   101,   101,   102,   102,
     102,   102,   102,   103,   103,   103,   103,   103,   103,   103,
     103,   103,   103,   103,   103,   103,   103,   104,   104,   104,
     105,   105,   106,   106,   107,   107,   108,   108,   108,   108,
     108,   108,   108,   108,   108,   108,   108,   108,   108,   108,
     108,   109,   110,   111,   111,   112,   113,   113,   114,   115,
     115,   115,   115,   115,   115,   116,   116,   116,   116,   116,
     117,   117,   117,   118,   118,   118,   119,   120,   121,   121,
     122,   123,   124,   124,   125,   125,   126,   126,   127,   127,
     128,   128,   129,   129,   130,   130,   131,   132,   132,   133,
     133,   134,   134,   135,   135,   136,   136,   137,   138,   138,
     139,   139,   139,   140,   140,   141,   141,   142,   142,   143,
     144,   145,   145,   146,   146,   147,   147,   148,   149,   150,
     151,   151,   152,   152,   153,   153,   153,   153,   154,   154,
     154,   155,   155,   156,   157,   157,   157,   157,   157,   158,
     158,   159,   159,   160,   160,   160,   160,   160,   160,   160,
     161,   161,   161,   161,   161,   162,   162,   162,   162,   163,
     163,   164,   165,   166,   166,   166,   166,   167,   167,   167,
     167,   167,   167,   167,   167,   167,   167,   167,   168,   168,
     168,   169,   169,   170,   171,   171,   171,   172,   172,   173,
     173,   173,   173,   173,   173,   173,   173,   173,   173,   173,
     173,   173,   173,   173,   173,   173,   173,   174,   174,   174,
     175,   175,   175,   176,   176,   176,   176,   176,   176,   177,
     178,   179,   180,   180,   180,   180,   180,   181,   181,   181,
     182,   182,   182,   182,   182,   182,   183,   184,   184,   184,
     185,   185,   186,   186,   187,   187,   188,   188,   189,   189,
     190,   190,   190,   191,   191,   192,   192,   193,   193,   193,
     194,   194,   195,   195,   195,   196,   196,   196,   196,   196,
     196,   196,   196,   196,   196,   196,   196,   197,   197,   197,
     197,   197,   197,   197,   197,   197,   197,   197,   197,   197,
     197,   198,   198,   198,   199,   200,   201,   202,   202,   203,
     203,   204,   205,   206,   207
};

  /* YYR2[YYN] -- Number of symbols on the right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     0,     2,     0,     1,     1,     2,     0,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       4,     4,     3,     3,     1,     4,     0,     2,     3,     2,
       2,     2,     7,     5,     5,     2,     2,     2,     2,     2,
       2,     2,     2,     1,     1,     1,     1,     1,     1,     1,
       1,     3,     0,     1,     0,     3,     1,     1,     1,     1,
       2,     2,     3,     3,     2,     2,     2,     1,     1,     2,
       1,     2,     2,     1,     1,     2,     2,     2,     8,     1,
       1,     1,     1,     2,     2,     1,     1,     1,     2,     2,
       3,     2,     1,     3,     2,     1,     1,     3,     0,     2,
       4,     6,     0,     1,     0,     3,     1,     3,     1,     1,
       0,     3,     1,     3,     0,     1,     1,     0,     3,     1,
       3,     1,     1,     0,     1,     0,     2,     5,     1,     2,
       3,     5,     6,     0,     2,     1,     3,     5,     5,     5,
       5,     4,     3,     6,     6,     5,     5,     5,     5,     5,
       4,     7,     0,     2,     0,     2,     2,     2,     3,     2,
       3,     1,     3,     4,     2,     2,     2,     2,     2,     1,
       4,     0,     2,     1,     1,     1,     1,     2,     2,     2,
       3,     6,     9,     3,     6,     3,     6,     9,     9,     1,
       3,     1,     1,     1,     2,     2,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     7,     5,
      13,     5,     2,     1,     0,     3,     1,     1,     3,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     2,     1,     1,     1,     1,     1,     1,     1,
       0,     1,     3,     0,     1,     5,     5,     5,     4,     3,
       1,     1,     1,     3,     4,     3,     4,     1,     1,     1,
       1,     4,     3,     4,     4,     4,     3,     7,     5,     6,
       1,     3,     1,     3,     3,     2,     3,     2,     0,     3,
       1,     1,     4,     1,     2,     1,     2,     1,     2,     1,
       1,     0,     4,     3,     5,     6,     4,     4,    11,     9,
      12,    14,     6,     8,     5,     7,     4,     6,     4,     1,
       4,    11,     9,    12,    14,     6,     8,     5,     7,     4,
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
#line 196 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.modlist) = (yyvsp[0].modlist); modlist = (yyvsp[0].modlist); }
#line 2238 "y.tab.c" /* yacc.c:1661  */
    break;

  case 3:
#line 200 "xi-grammar.y" /* yacc.c:1661  */
    {
		  (yyval.modlist) = 0;
		}
#line 2246 "y.tab.c" /* yacc.c:1661  */
    break;

  case 4:
#line 204 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.modlist) = new AstChildren<Module>(lineno, (yyvsp[-1].module), (yyvsp[0].modlist)); }
#line 2252 "y.tab.c" /* yacc.c:1661  */
    break;

  case 5:
#line 208 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 0; }
#line 2258 "y.tab.c" /* yacc.c:1661  */
    break;

  case 6:
#line 210 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 1; }
#line 2264 "y.tab.c" /* yacc.c:1661  */
    break;

  case 7:
#line 214 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 1; }
#line 2270 "y.tab.c" /* yacc.c:1661  */
    break;

  case 8:
#line 216 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 2; }
#line 2276 "y.tab.c" /* yacc.c:1661  */
    break;

  case 9:
#line 220 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 0; }
#line 2282 "y.tab.c" /* yacc.c:1661  */
    break;

  case 10:
#line 222 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 1; }
#line 2288 "y.tab.c" /* yacc.c:1661  */
    break;

  case 11:
#line 227 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 2294 "y.tab.c" /* yacc.c:1661  */
    break;

  case 12:
#line 228 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(MODULE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2300 "y.tab.c" /* yacc.c:1661  */
    break;

  case 13:
#line 229 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(MAINMODULE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2306 "y.tab.c" /* yacc.c:1661  */
    break;

  case 14:
#line 230 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(EXTERN, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2312 "y.tab.c" /* yacc.c:1661  */
    break;

  case 15:
#line 232 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(INITCALL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2318 "y.tab.c" /* yacc.c:1661  */
    break;

  case 16:
#line 233 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(INITNODE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2324 "y.tab.c" /* yacc.c:1661  */
    break;

  case 17:
#line 234 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(INITPROC, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2330 "y.tab.c" /* yacc.c:1661  */
    break;

  case 18:
#line 236 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(CHARE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2336 "y.tab.c" /* yacc.c:1661  */
    break;

  case 19:
#line 237 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(MAINCHARE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2342 "y.tab.c" /* yacc.c:1661  */
    break;

  case 20:
#line 238 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(GROUP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2348 "y.tab.c" /* yacc.c:1661  */
    break;

  case 21:
#line 239 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(NODEGROUP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2354 "y.tab.c" /* yacc.c:1661  */
    break;

  case 22:
#line 240 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(ARRAY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2360 "y.tab.c" /* yacc.c:1661  */
    break;

  case 23:
#line 244 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(INCLUDE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2366 "y.tab.c" /* yacc.c:1661  */
    break;

  case 24:
#line 245 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(STACKSIZE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2372 "y.tab.c" /* yacc.c:1661  */
    break;

  case 25:
#line 246 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(THREADED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2378 "y.tab.c" /* yacc.c:1661  */
    break;

  case 26:
#line 247 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(TEMPLATE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2384 "y.tab.c" /* yacc.c:1661  */
    break;

  case 27:
#line 248 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(SYNC, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2390 "y.tab.c" /* yacc.c:1661  */
    break;

  case 28:
#line 249 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(IGET, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2396 "y.tab.c" /* yacc.c:1661  */
    break;

  case 29:
#line 250 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(EXCLUSIVE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2402 "y.tab.c" /* yacc.c:1661  */
    break;

  case 30:
#line 251 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(IMMEDIATE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2408 "y.tab.c" /* yacc.c:1661  */
    break;

  case 31:
#line 252 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(SKIPSCHED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2414 "y.tab.c" /* yacc.c:1661  */
    break;

  case 32:
#line 253 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(NOCOPY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2420 "y.tab.c" /* yacc.c:1661  */
    break;

  case 33:
#line 254 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(INLINE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2426 "y.tab.c" /* yacc.c:1661  */
    break;

  case 34:
#line 255 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(VIRTUAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2432 "y.tab.c" /* yacc.c:1661  */
    break;

  case 35:
#line 256 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(MIGRATABLE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2438 "y.tab.c" /* yacc.c:1661  */
    break;

  case 36:
#line 257 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(CREATEHERE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2444 "y.tab.c" /* yacc.c:1661  */
    break;

  case 37:
#line 258 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(CREATEHOME, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2450 "y.tab.c" /* yacc.c:1661  */
    break;

  case 38:
#line 259 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(NOKEEP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2456 "y.tab.c" /* yacc.c:1661  */
    break;

  case 39:
#line 260 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(NOTRACE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2462 "y.tab.c" /* yacc.c:1661  */
    break;

  case 40:
#line 261 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(APPWORK, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2468 "y.tab.c" /* yacc.c:1661  */
    break;

  case 41:
#line 264 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(PACKED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2474 "y.tab.c" /* yacc.c:1661  */
    break;

  case 42:
#line 265 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(VARSIZE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2480 "y.tab.c" /* yacc.c:1661  */
    break;

  case 43:
#line 266 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(ENTRY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2486 "y.tab.c" /* yacc.c:1661  */
    break;

  case 44:
#line 267 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(FOR, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2492 "y.tab.c" /* yacc.c:1661  */
    break;

  case 45:
#line 268 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(FORALL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2498 "y.tab.c" /* yacc.c:1661  */
    break;

  case 46:
#line 269 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(WHILE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2504 "y.tab.c" /* yacc.c:1661  */
    break;

  case 47:
#line 270 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(WHEN, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2510 "y.tab.c" /* yacc.c:1661  */
    break;

  case 48:
#line 271 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(OVERLAP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2516 "y.tab.c" /* yacc.c:1661  */
    break;

  case 49:
#line 272 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(SERIAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2522 "y.tab.c" /* yacc.c:1661  */
    break;

  case 50:
#line 273 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(IF, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2528 "y.tab.c" /* yacc.c:1661  */
    break;

  case 51:
#line 274 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(ELSE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2534 "y.tab.c" /* yacc.c:1661  */
    break;

  case 52:
#line 276 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(LOCAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2540 "y.tab.c" /* yacc.c:1661  */
    break;

  case 53:
#line 278 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(USING, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2546 "y.tab.c" /* yacc.c:1661  */
    break;

  case 54:
#line 279 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(ACCEL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2552 "y.tab.c" /* yacc.c:1661  */
    break;

  case 55:
#line 282 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(ACCELBLOCK, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2558 "y.tab.c" /* yacc.c:1661  */
    break;

  case 56:
#line 283 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(MEMCRITICAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2564 "y.tab.c" /* yacc.c:1661  */
    break;

  case 57:
#line 284 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(REDUCTIONTARGET, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2570 "y.tab.c" /* yacc.c:1661  */
    break;

  case 58:
#line 285 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(CASE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2576 "y.tab.c" /* yacc.c:1661  */
    break;

  case 59:
#line 290 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 2582 "y.tab.c" /* yacc.c:1661  */
    break;

  case 60:
#line 292 "xi-grammar.y" /* yacc.c:1661  */
    {
		  char *tmp = new char[strlen((yyvsp[-3].strval))+strlen((yyvsp[0].strval))+3];
		  sprintf(tmp,"%s::%s", (yyvsp[-3].strval), (yyvsp[0].strval));
		  (yyval.strval) = tmp;
		}
#line 2592 "y.tab.c" /* yacc.c:1661  */
    break;

  case 61:
#line 298 "xi-grammar.y" /* yacc.c:1661  */
    {
		  char *tmp = new char[strlen((yyvsp[-3].strval))+5+3];
		  sprintf(tmp,"%s::array", (yyvsp[-3].strval));
		  (yyval.strval) = tmp;
		}
#line 2602 "y.tab.c" /* yacc.c:1661  */
    break;

  case 62:
#line 305 "xi-grammar.y" /* yacc.c:1661  */
    {
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist));
		}
#line 2610 "y.tab.c" /* yacc.c:1661  */
    break;

  case 63:
#line 309 "xi-grammar.y" /* yacc.c:1661  */
    {
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist));
		    (yyval.module)->setMain();
		}
#line 2619 "y.tab.c" /* yacc.c:1661  */
    break;

  case 64:
#line 316 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.conslist) = 0; }
#line 2625 "y.tab.c" /* yacc.c:1661  */
    break;

  case 65:
#line 318 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.conslist) = (yyvsp[-2].conslist); }
#line 2631 "y.tab.c" /* yacc.c:1661  */
    break;

  case 66:
#line 322 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.conslist) = 0; }
#line 2637 "y.tab.c" /* yacc.c:1661  */
    break;

  case 67:
#line 324 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.conslist) = new ConstructList(lineno, (yyvsp[-1].construct), (yyvsp[0].conslist)); }
#line 2643 "y.tab.c" /* yacc.c:1661  */
    break;

  case 68:
#line 328 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.construct) = new UsingScope((yyvsp[0].strval), false); }
#line 2649 "y.tab.c" /* yacc.c:1661  */
    break;

  case 69:
#line 330 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.construct) = new UsingScope((yyvsp[0].strval), true); }
#line 2655 "y.tab.c" /* yacc.c:1661  */
    break;

  case 70:
#line 332 "xi-grammar.y" /* yacc.c:1661  */
    { (yyvsp[0].member)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].member); }
#line 2661 "y.tab.c" /* yacc.c:1661  */
    break;

  case 71:
#line 334 "xi-grammar.y" /* yacc.c:1661  */
    { (yyvsp[0].message)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].message); }
#line 2667 "y.tab.c" /* yacc.c:1661  */
    break;

  case 72:
#line 336 "xi-grammar.y" /* yacc.c:1661  */
    {
                  Entry *e = new Entry(lineno, 0, (yyvsp[-4].type), (yyvsp[-2].strval), (yyvsp[0].plist), 0, 0, 0, (yylsp[-6]).first_line, (yyloc).last_line);
                  int isExtern = 1;
                  e->setExtern(isExtern);
                  e->targs = (yyvsp[-1].tparlist);
                  e->label = new XStr;
                  (yyvsp[-3].ntype)->print(*e->label);
                  (yyval.construct) = e;
                  firstRdma = true;
                }
#line 2682 "y.tab.c" /* yacc.c:1661  */
    break;

  case 73:
#line 349 "xi-grammar.y" /* yacc.c:1661  */
    { if((yyvsp[-2].conslist)) (yyvsp[-2].conslist)->recurse<int&>((yyvsp[-4].intval), &Construct::setExtern); (yyval.construct) = (yyvsp[-2].conslist); }
#line 2688 "y.tab.c" /* yacc.c:1661  */
    break;

  case 74:
#line 351 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.construct) = new Scope((yyvsp[-3].strval), (yyvsp[-1].conslist)); }
#line 2694 "y.tab.c" /* yacc.c:1661  */
    break;

  case 75:
#line 353 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.construct) = (yyvsp[-1].construct); }
#line 2700 "y.tab.c" /* yacc.c:1661  */
    break;

  case 76:
#line 355 "xi-grammar.y" /* yacc.c:1661  */
    {
          ERROR("preceding construct must be semicolon terminated",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 2710 "y.tab.c" /* yacc.c:1661  */
    break;

  case 77:
#line 361 "xi-grammar.y" /* yacc.c:1661  */
    { (yyvsp[0].module)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].module); }
#line 2716 "y.tab.c" /* yacc.c:1661  */
    break;

  case 78:
#line 363 "xi-grammar.y" /* yacc.c:1661  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2722 "y.tab.c" /* yacc.c:1661  */
    break;

  case 79:
#line 365 "xi-grammar.y" /* yacc.c:1661  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2728 "y.tab.c" /* yacc.c:1661  */
    break;

  case 80:
#line 367 "xi-grammar.y" /* yacc.c:1661  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2734 "y.tab.c" /* yacc.c:1661  */
    break;

  case 81:
#line 369 "xi-grammar.y" /* yacc.c:1661  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2740 "y.tab.c" /* yacc.c:1661  */
    break;

  case 82:
#line 371 "xi-grammar.y" /* yacc.c:1661  */
    { (yyvsp[0].templat)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].templat); }
#line 2746 "y.tab.c" /* yacc.c:1661  */
    break;

  case 83:
#line 373 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.construct) = NULL; }
#line 2752 "y.tab.c" /* yacc.c:1661  */
    break;

  case 84:
#line 375 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.construct) = NULL; }
#line 2758 "y.tab.c" /* yacc.c:1661  */
    break;

  case 85:
#line 377 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.construct) = (yyvsp[0].accelBlock); }
#line 2764 "y.tab.c" /* yacc.c:1661  */
    break;

  case 86:
#line 379 "xi-grammar.y" /* yacc.c:1661  */
    {
          ERROR("invalid construct",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 2774 "y.tab.c" /* yacc.c:1661  */
    break;

  case 87:
#line 387 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tparam) = new TParamType((yyvsp[0].type)); }
#line 2780 "y.tab.c" /* yacc.c:1661  */
    break;

  case 88:
#line 389 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
#line 2786 "y.tab.c" /* yacc.c:1661  */
    break;

  case 89:
#line 391 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
#line 2792 "y.tab.c" /* yacc.c:1661  */
    break;

  case 90:
#line 395 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tparlist) = new TParamList((yyvsp[0].tparam)); }
#line 2798 "y.tab.c" /* yacc.c:1661  */
    break;

  case 91:
#line 397 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tparlist) = new TParamList((yyvsp[-2].tparam), (yyvsp[0].tparlist)); }
#line 2804 "y.tab.c" /* yacc.c:1661  */
    break;

  case 92:
#line 401 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tparlist) = new TParamList(0); }
#line 2810 "y.tab.c" /* yacc.c:1661  */
    break;

  case 93:
#line 403 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tparlist) = (yyvsp[0].tparlist); }
#line 2816 "y.tab.c" /* yacc.c:1661  */
    break;

  case 94:
#line 407 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tparlist) = 0; }
#line 2822 "y.tab.c" /* yacc.c:1661  */
    break;

  case 95:
#line 409 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tparlist) = (yyvsp[-1].tparlist); }
#line 2828 "y.tab.c" /* yacc.c:1661  */
    break;

  case 96:
#line 413 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("int"); }
#line 2834 "y.tab.c" /* yacc.c:1661  */
    break;

  case 97:
#line 415 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("long"); }
#line 2840 "y.tab.c" /* yacc.c:1661  */
    break;

  case 98:
#line 417 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("short"); }
#line 2846 "y.tab.c" /* yacc.c:1661  */
    break;

  case 99:
#line 419 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("char"); }
#line 2852 "y.tab.c" /* yacc.c:1661  */
    break;

  case 100:
#line 421 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("unsigned int"); }
#line 2858 "y.tab.c" /* yacc.c:1661  */
    break;

  case 101:
#line 423 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("unsigned long"); }
#line 2864 "y.tab.c" /* yacc.c:1661  */
    break;

  case 102:
#line 425 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("unsigned long"); }
#line 2870 "y.tab.c" /* yacc.c:1661  */
    break;

  case 103:
#line 427 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("unsigned long long"); }
#line 2876 "y.tab.c" /* yacc.c:1661  */
    break;

  case 104:
#line 429 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("unsigned short"); }
#line 2882 "y.tab.c" /* yacc.c:1661  */
    break;

  case 105:
#line 431 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("unsigned char"); }
#line 2888 "y.tab.c" /* yacc.c:1661  */
    break;

  case 106:
#line 433 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("long long"); }
#line 2894 "y.tab.c" /* yacc.c:1661  */
    break;

  case 107:
#line 435 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("float"); }
#line 2900 "y.tab.c" /* yacc.c:1661  */
    break;

  case 108:
#line 437 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("double"); }
#line 2906 "y.tab.c" /* yacc.c:1661  */
    break;

  case 109:
#line 439 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("long double"); }
#line 2912 "y.tab.c" /* yacc.c:1661  */
    break;

  case 110:
#line 441 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("void"); }
#line 2918 "y.tab.c" /* yacc.c:1661  */
    break;

  case 111:
#line 444 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.ntype) = new NamedType((yyvsp[-1].strval),(yyvsp[0].tparlist)); }
#line 2924 "y.tab.c" /* yacc.c:1661  */
    break;

  case 112:
#line 445 "xi-grammar.y" /* yacc.c:1661  */
    {
                    const char* basename, *scope;
                    splitScopedName((yyvsp[-1].strval), &scope, &basename);
                    (yyval.ntype) = new NamedType(basename, (yyvsp[0].tparlist), scope);
                }
#line 2934 "y.tab.c" /* yacc.c:1661  */
    break;

  case 113:
#line 453 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].type); }
#line 2940 "y.tab.c" /* yacc.c:1661  */
    break;

  case 114:
#line 455 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].ntype); }
#line 2946 "y.tab.c" /* yacc.c:1661  */
    break;

  case 115:
#line 459 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.ptype) = new PtrType((yyvsp[-1].type)); }
#line 2952 "y.tab.c" /* yacc.c:1661  */
    break;

  case 116:
#line 463 "xi-grammar.y" /* yacc.c:1661  */
    { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
#line 2958 "y.tab.c" /* yacc.c:1661  */
    break;

  case 117:
#line 465 "xi-grammar.y" /* yacc.c:1661  */
    { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
#line 2964 "y.tab.c" /* yacc.c:1661  */
    break;

  case 118:
#line 469 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.ftype) = new FuncType((yyvsp[-7].type), (yyvsp[-4].strval), (yyvsp[-1].plist)); }
#line 2970 "y.tab.c" /* yacc.c:1661  */
    break;

  case 119:
#line 473 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].type); }
#line 2976 "y.tab.c" /* yacc.c:1661  */
    break;

  case 120:
#line 475 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 2982 "y.tab.c" /* yacc.c:1661  */
    break;

  case 121:
#line 477 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 2988 "y.tab.c" /* yacc.c:1661  */
    break;

  case 122:
#line 479 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].ftype); }
#line 2994 "y.tab.c" /* yacc.c:1661  */
    break;

  case 123:
#line 481 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new ConstType((yyvsp[0].type)); }
#line 3000 "y.tab.c" /* yacc.c:1661  */
    break;

  case 124:
#line 483 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new ConstType((yyvsp[-1].type)); }
#line 3006 "y.tab.c" /* yacc.c:1661  */
    break;

  case 125:
#line 487 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3012 "y.tab.c" /* yacc.c:1661  */
    break;

  case 126:
#line 489 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3018 "y.tab.c" /* yacc.c:1661  */
    break;

  case 127:
#line 491 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3024 "y.tab.c" /* yacc.c:1661  */
    break;

  case 128:
#line 493 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new ConstType((yyvsp[0].type)); }
#line 3030 "y.tab.c" /* yacc.c:1661  */
    break;

  case 129:
#line 495 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new ConstType((yyvsp[-1].type)); }
#line 3036 "y.tab.c" /* yacc.c:1661  */
    break;

  case 130:
#line 499 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new RValueReferenceType((yyvsp[-2].type)); }
#line 3042 "y.tab.c" /* yacc.c:1661  */
    break;

  case 131:
#line 501 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new ReferenceType((yyvsp[-1].type)); }
#line 3048 "y.tab.c" /* yacc.c:1661  */
    break;

  case 132:
#line 503 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3054 "y.tab.c" /* yacc.c:1661  */
    break;

  case 133:
#line 507 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new RValueReferenceType((yyvsp[-2].type)); }
#line 3060 "y.tab.c" /* yacc.c:1661  */
    break;

  case 134:
#line 509 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new ReferenceType((yyvsp[-1].type)); }
#line 3066 "y.tab.c" /* yacc.c:1661  */
    break;

  case 135:
#line 511 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3072 "y.tab.c" /* yacc.c:1661  */
    break;

  case 136:
#line 515 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 3078 "y.tab.c" /* yacc.c:1661  */
    break;

  case 137:
#line 519 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.val) = (yyvsp[-1].val); }
#line 3084 "y.tab.c" /* yacc.c:1661  */
    break;

  case 138:
#line 523 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.vallist) = 0; }
#line 3090 "y.tab.c" /* yacc.c:1661  */
    break;

  case 139:
#line 525 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.vallist) = new ValueList((yyvsp[-1].val), (yyvsp[0].vallist)); }
#line 3096 "y.tab.c" /* yacc.c:1661  */
    break;

  case 140:
#line 529 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].vallist)); }
#line 3102 "y.tab.c" /* yacc.c:1661  */
    break;

  case 141:
#line 533 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[-3].type), (yyvsp[-1].strval), (yyvsp[0].vallist), 1); }
#line 3108 "y.tab.c" /* yacc.c:1661  */
    break;

  case 142:
#line 537 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 0;}
#line 3114 "y.tab.c" /* yacc.c:1661  */
    break;

  case 143:
#line 539 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 0;}
#line 3120 "y.tab.c" /* yacc.c:1661  */
    break;

  case 144:
#line 543 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 0; }
#line 3126 "y.tab.c" /* yacc.c:1661  */
    break;

  case 145:
#line 545 "xi-grammar.y" /* yacc.c:1661  */
    {
		  /*
		  printf("Warning: Message attributes are being phased out.\n");
		  printf("Warning: Please remove them from interface files.\n");
		  */
		  (yyval.intval) = (yyvsp[-1].intval);
		}
#line 3138 "y.tab.c" /* yacc.c:1661  */
    break;

  case 146:
#line 555 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = (yyvsp[0].intval); }
#line 3144 "y.tab.c" /* yacc.c:1661  */
    break;

  case 147:
#line 557 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = (yyvsp[-2].intval) | (yyvsp[0].intval); }
#line 3150 "y.tab.c" /* yacc.c:1661  */
    break;

  case 148:
#line 561 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 0; }
#line 3156 "y.tab.c" /* yacc.c:1661  */
    break;

  case 149:
#line 563 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 0; }
#line 3162 "y.tab.c" /* yacc.c:1661  */
    break;

  case 150:
#line 567 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = 0; }
#line 3168 "y.tab.c" /* yacc.c:1661  */
    break;

  case 151:
#line 569 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = (yyvsp[-1].cattr); }
#line 3174 "y.tab.c" /* yacc.c:1661  */
    break;

  case 152:
#line 573 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = (yyvsp[0].cattr); }
#line 3180 "y.tab.c" /* yacc.c:1661  */
    break;

  case 153:
#line 575 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
#line 3186 "y.tab.c" /* yacc.c:1661  */
    break;

  case 154:
#line 579 "xi-grammar.y" /* yacc.c:1661  */
    { python_doc = NULL; (yyval.intval) = 0; }
#line 3192 "y.tab.c" /* yacc.c:1661  */
    break;

  case 155:
#line 581 "xi-grammar.y" /* yacc.c:1661  */
    { python_doc = (yyvsp[0].strval); (yyval.intval) = 0; }
#line 3198 "y.tab.c" /* yacc.c:1661  */
    break;

  case 156:
#line 585 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = Chare::CPYTHON; }
#line 3204 "y.tab.c" /* yacc.c:1661  */
    break;

  case 157:
#line 589 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = 0; }
#line 3210 "y.tab.c" /* yacc.c:1661  */
    break;

  case 158:
#line 591 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = (yyvsp[-1].cattr); }
#line 3216 "y.tab.c" /* yacc.c:1661  */
    break;

  case 159:
#line 595 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = (yyvsp[0].cattr); }
#line 3222 "y.tab.c" /* yacc.c:1661  */
    break;

  case 160:
#line 597 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
#line 3228 "y.tab.c" /* yacc.c:1661  */
    break;

  case 161:
#line 601 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = Chare::CMIGRATABLE; }
#line 3234 "y.tab.c" /* yacc.c:1661  */
    break;

  case 162:
#line 603 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = Chare::CPYTHON; }
#line 3240 "y.tab.c" /* yacc.c:1661  */
    break;

  case 163:
#line 607 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 0; }
#line 3246 "y.tab.c" /* yacc.c:1661  */
    break;

  case 164:
#line 609 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 1; }
#line 3252 "y.tab.c" /* yacc.c:1661  */
    break;

  case 165:
#line 612 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 0; }
#line 3258 "y.tab.c" /* yacc.c:1661  */
    break;

  case 166:
#line 614 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 1; }
#line 3264 "y.tab.c" /* yacc.c:1661  */
    break;

  case 167:
#line 617 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.mv) = new MsgVar((yyvsp[-3].type), (yyvsp[-2].strval), (yyvsp[-4].intval), (yyvsp[-1].intval)); }
#line 3270 "y.tab.c" /* yacc.c:1661  */
    break;

  case 168:
#line 621 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.mvlist) = new MsgVarList((yyvsp[0].mv)); }
#line 3276 "y.tab.c" /* yacc.c:1661  */
    break;

  case 169:
#line 623 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.mvlist) = new MsgVarList((yyvsp[-1].mv), (yyvsp[0].mvlist)); }
#line 3282 "y.tab.c" /* yacc.c:1661  */
    break;

  case 170:
#line 627 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.message) = new Message(lineno, (yyvsp[0].ntype)); }
#line 3288 "y.tab.c" /* yacc.c:1661  */
    break;

  case 171:
#line 629 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.message) = new Message(lineno, (yyvsp[-2].ntype)); }
#line 3294 "y.tab.c" /* yacc.c:1661  */
    break;

  case 172:
#line 631 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.message) = new Message(lineno, (yyvsp[-3].ntype), (yyvsp[-1].mvlist)); }
#line 3300 "y.tab.c" /* yacc.c:1661  */
    break;

  case 173:
#line 635 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.typelist) = 0; }
#line 3306 "y.tab.c" /* yacc.c:1661  */
    break;

  case 174:
#line 637 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.typelist) = (yyvsp[0].typelist); }
#line 3312 "y.tab.c" /* yacc.c:1661  */
    break;

  case 175:
#line 641 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.typelist) = new TypeList((yyvsp[0].ntype)); }
#line 3318 "y.tab.c" /* yacc.c:1661  */
    break;

  case 176:
#line 643 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.typelist) = new TypeList((yyvsp[-2].ntype), (yyvsp[0].typelist)); }
#line 3324 "y.tab.c" /* yacc.c:1661  */
    break;

  case 177:
#line 647 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr)|Chare::CCHARE, (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3330 "y.tab.c" /* yacc.c:1661  */
    break;

  case 178:
#line 649 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3336 "y.tab.c" /* yacc.c:1661  */
    break;

  case 179:
#line 653 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3342 "y.tab.c" /* yacc.c:1661  */
    break;

  case 180:
#line 657 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.chare) = new NodeGroup(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3348 "y.tab.c" /* yacc.c:1661  */
    break;

  case 181:
#line 661 "xi-grammar.y" /* yacc.c:1661  */
    {/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",(yyvsp[-2].strval));
			(yyval.ntype) = new NamedType(buf);
		}
#line 3358 "y.tab.c" /* yacc.c:1661  */
    break;

  case 182:
#line 667 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.ntype) = new NamedType((yyvsp[-1].strval)); }
#line 3364 "y.tab.c" /* yacc.c:1661  */
    break;

  case 183:
#line 671 "xi-grammar.y" /* yacc.c:1661  */
    {  (yyval.chare) = new Array(lineno, (yyvsp[-4].cattr), (yyvsp[-3].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3370 "y.tab.c" /* yacc.c:1661  */
    break;

  case 184:
#line 673 "xi-grammar.y" /* yacc.c:1661  */
    {  (yyval.chare) = new Array(lineno, (yyvsp[-3].cattr), (yyvsp[-4].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3376 "y.tab.c" /* yacc.c:1661  */
    break;

  case 185:
#line 677 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr)|Chare::CCHARE, new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist));}
#line 3382 "y.tab.c" /* yacc.c:1661  */
    break;

  case 186:
#line 679 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3388 "y.tab.c" /* yacc.c:1661  */
    break;

  case 187:
#line 683 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3394 "y.tab.c" /* yacc.c:1661  */
    break;

  case 188:
#line 687 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.chare) = new NodeGroup( lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3400 "y.tab.c" /* yacc.c:1661  */
    break;

  case 189:
#line 691 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.chare) = new Array( lineno, 0, (yyvsp[-3].ntype), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3406 "y.tab.c" /* yacc.c:1661  */
    break;

  case 190:
#line 695 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-1].strval))); }
#line 3412 "y.tab.c" /* yacc.c:1661  */
    break;

  case 191:
#line 697 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-4].strval)), (yyvsp[-2].mvlist)); }
#line 3418 "y.tab.c" /* yacc.c:1661  */
    break;

  case 192:
#line 701 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = 0; }
#line 3424 "y.tab.c" /* yacc.c:1661  */
    break;

  case 193:
#line 703 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3430 "y.tab.c" /* yacc.c:1661  */
    break;

  case 194:
#line 707 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = 0; }
#line 3436 "y.tab.c" /* yacc.c:1661  */
    break;

  case 195:
#line 709 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3442 "y.tab.c" /* yacc.c:1661  */
    break;

  case 196:
#line 711 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3448 "y.tab.c" /* yacc.c:1661  */
    break;

  case 197:
#line 713 "xi-grammar.y" /* yacc.c:1661  */
    {
		  XStr typeStr;
		  (yyvsp[0].ntype)->print(typeStr);
		  char *tmp = strdup(typeStr.get_string());
		  (yyval.strval) = tmp;
		}
#line 3459 "y.tab.c" /* yacc.c:1661  */
    break;

  case 198:
#line 722 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tvar) = new TType(new NamedType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3465 "y.tab.c" /* yacc.c:1661  */
    break;

  case 199:
#line 724 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tvar) = new TFunc((yyvsp[-1].ftype), (yyvsp[0].strval)); }
#line 3471 "y.tab.c" /* yacc.c:1661  */
    break;

  case 200:
#line 726 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tvar) = new TName((yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].strval)); }
#line 3477 "y.tab.c" /* yacc.c:1661  */
    break;

  case 201:
#line 730 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tvarlist) = new TVarList((yyvsp[0].tvar)); }
#line 3483 "y.tab.c" /* yacc.c:1661  */
    break;

  case 202:
#line 732 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tvarlist) = new TVarList((yyvsp[-2].tvar), (yyvsp[0].tvarlist)); }
#line 3489 "y.tab.c" /* yacc.c:1661  */
    break;

  case 203:
#line 736 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tvarlist) = (yyvsp[-1].tvarlist); }
#line 3495 "y.tab.c" /* yacc.c:1661  */
    break;

  case 204:
#line 740 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3501 "y.tab.c" /* yacc.c:1661  */
    break;

  case 205:
#line 742 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3507 "y.tab.c" /* yacc.c:1661  */
    break;

  case 206:
#line 744 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3513 "y.tab.c" /* yacc.c:1661  */
    break;

  case 207:
#line 746 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3519 "y.tab.c" /* yacc.c:1661  */
    break;

  case 208:
#line 748 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].message)); (yyvsp[0].message)->setTemplate((yyval.templat)); }
#line 3525 "y.tab.c" /* yacc.c:1661  */
    break;

  case 209:
#line 752 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.mbrlist) = 0; }
#line 3531 "y.tab.c" /* yacc.c:1661  */
    break;

  case 210:
#line 754 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.mbrlist) = (yyvsp[-2].mbrlist); }
#line 3537 "y.tab.c" /* yacc.c:1661  */
    break;

  case 211:
#line 758 "xi-grammar.y" /* yacc.c:1661  */
    {
                  if (!connectEntries.empty()) {
                    (yyval.mbrlist) = new AstChildren<Member>(connectEntries);
		  } else {
		    (yyval.mbrlist) = 0;
                  }
		}
#line 3549 "y.tab.c" /* yacc.c:1661  */
    break;

  case 212:
#line 766 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.mbrlist) = new AstChildren<Member>(-1, (yyvsp[-1].member), (yyvsp[0].mbrlist)); }
#line 3555 "y.tab.c" /* yacc.c:1661  */
    break;

  case 213:
#line 770 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = (yyvsp[0].readonly); }
#line 3561 "y.tab.c" /* yacc.c:1661  */
    break;

  case 214:
#line 772 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = (yyvsp[0].readonly); }
#line 3567 "y.tab.c" /* yacc.c:1661  */
    break;

  case 216:
#line 775 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = (yyvsp[0].member); }
#line 3573 "y.tab.c" /* yacc.c:1661  */
    break;

  case 217:
#line 777 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = (yyvsp[0].pupable); }
#line 3579 "y.tab.c" /* yacc.c:1661  */
    break;

  case 218:
#line 779 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = (yyvsp[0].includeFile); }
#line 3585 "y.tab.c" /* yacc.c:1661  */
    break;

  case 219:
#line 781 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = new ClassDeclaration(lineno,(yyvsp[0].strval)); }
#line 3591 "y.tab.c" /* yacc.c:1661  */
    break;

  case 220:
#line 785 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1); }
#line 3597 "y.tab.c" /* yacc.c:1661  */
    break;

  case 221:
#line 787 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1); }
#line 3603 "y.tab.c" /* yacc.c:1661  */
    break;

  case 222:
#line 789 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[-6].strval)) + '<' +
					    ((yyvsp[-4].tparlist))->to_string() + '>').c_str()),
				    1);
		}
#line 3613 "y.tab.c" /* yacc.c:1661  */
    break;

  case 223:
#line 795 "xi-grammar.y" /* yacc.c:1661  */
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[-2]).first_column, (yylsp[-2]).last_column, (yylsp[-2]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1);
		}
#line 3623 "y.tab.c" /* yacc.c:1661  */
    break;

  case 224:
#line 801 "xi-grammar.y" /* yacc.c:1661  */
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[-5]).first_column, (yylsp[-5]).last_column, (yylsp[-5]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1);
		}
#line 3633 "y.tab.c" /* yacc.c:1661  */
    break;

  case 225:
#line 810 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 0); }
#line 3639 "y.tab.c" /* yacc.c:1661  */
    break;

  case 226:
#line 812 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 0); }
#line 3645 "y.tab.c" /* yacc.c:1661  */
    break;

  case 227:
#line 814 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[-6].strval)) + '<' +
					    ((yyvsp[-4].tparlist))->to_string() + '>').c_str()),
				    0);
		}
#line 3655 "y.tab.c" /* yacc.c:1661  */
    break;

  case 228:
#line 820 "xi-grammar.y" /* yacc.c:1661  */
    {
                  InitCall* rtn = new InitCall(lineno, (yyvsp[-3].strval), 0);
                  rtn->setAccel();
                  (yyval.member) = rtn;
		}
#line 3665 "y.tab.c" /* yacc.c:1661  */
    break;

  case 229:
#line 828 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[0].ntype),0); }
#line 3671 "y.tab.c" /* yacc.c:1661  */
    break;

  case 230:
#line 830 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[-2].ntype),(yyvsp[0].pupable)); }
#line 3677 "y.tab.c" /* yacc.c:1661  */
    break;

  case 231:
#line 833 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.includeFile) = new IncludeFile(lineno,(yyvsp[0].strval)); }
#line 3683 "y.tab.c" /* yacc.c:1661  */
    break;

  case 232:
#line 837 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = (yyvsp[0].member); }
#line 3689 "y.tab.c" /* yacc.c:1661  */
    break;

  case 233:
#line 841 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = (yyvsp[0].entry); }
#line 3695 "y.tab.c" /* yacc.c:1661  */
    break;

  case 234:
#line 843 "xi-grammar.y" /* yacc.c:1661  */
    {
                  (yyvsp[0].entry)->tspec = (yyvsp[-1].tvarlist);
                  (yyval.member) = (yyvsp[0].entry);
                }
#line 3704 "y.tab.c" /* yacc.c:1661  */
    break;

  case 235:
#line 848 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = (yyvsp[-1].member); }
#line 3710 "y.tab.c" /* yacc.c:1661  */
    break;

  case 236:
#line 850 "xi-grammar.y" /* yacc.c:1661  */
    {
          ERROR("invalid SDAG member",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 3720 "y.tab.c" /* yacc.c:1661  */
    break;

  case 237:
#line 858 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3726 "y.tab.c" /* yacc.c:1661  */
    break;

  case 238:
#line 860 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3732 "y.tab.c" /* yacc.c:1661  */
    break;

  case 239:
#line 862 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3738 "y.tab.c" /* yacc.c:1661  */
    break;

  case 240:
#line 864 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3744 "y.tab.c" /* yacc.c:1661  */
    break;

  case 241:
#line 866 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3750 "y.tab.c" /* yacc.c:1661  */
    break;

  case 242:
#line 868 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3756 "y.tab.c" /* yacc.c:1661  */
    break;

  case 243:
#line 870 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3762 "y.tab.c" /* yacc.c:1661  */
    break;

  case 244:
#line 872 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3768 "y.tab.c" /* yacc.c:1661  */
    break;

  case 245:
#line 874 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3774 "y.tab.c" /* yacc.c:1661  */
    break;

  case 246:
#line 876 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3780 "y.tab.c" /* yacc.c:1661  */
    break;

  case 247:
#line 878 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3786 "y.tab.c" /* yacc.c:1661  */
    break;

  case 248:
#line 881 "xi-grammar.y" /* yacc.c:1661  */
    {
                  (yyval.entry) = new Entry(lineno, (yyvsp[-5].intval), (yyvsp[-4].type), (yyvsp[-3].strval), (yyvsp[-2].plist), (yyvsp[-1].val), (yyvsp[0].sentry), (const char *) NULL, (yylsp[-6]).first_line, (yyloc).last_line);
		  if ((yyvsp[0].sentry) != 0) {
		    (yyvsp[0].sentry)->con1 = new SdagConstruct(SIDENT, (yyvsp[-3].strval));
                    (yyvsp[0].sentry)->setEntry((yyval.entry));
                    (yyvsp[0].sentry)->param = new ParamList((yyvsp[-2].plist));
                  }
                  firstRdma = true;
		}
#line 3800 "y.tab.c" /* yacc.c:1661  */
    break;

  case 249:
#line 891 "xi-grammar.y" /* yacc.c:1661  */
    {
                  Entry *e = new Entry(lineno, (yyvsp[-3].intval), 0, (yyvsp[-2].strval), (yyvsp[-1].plist),  0, (yyvsp[0].sentry), (const char *) NULL, (yylsp[-4]).first_line, (yyloc).last_line);
                  if ((yyvsp[0].sentry) != 0) {
		    (yyvsp[0].sentry)->con1 = new SdagConstruct(SIDENT, (yyvsp[-2].strval));
                    (yyvsp[0].sentry)->setEntry((yyval.entry));
                    (yyvsp[0].sentry)->param = new ParamList((yyvsp[-1].plist));
                  }
                  firstRdma = true;
		  if (e->param && e->param->isCkMigMsgPtr()) {
		    WARNING("CkMigrateMsg chare constructor is taken for granted",
		            (yyloc).first_column, (yyloc).last_column);
		    (yyval.entry) = NULL;
		  } else {
		    (yyval.entry) = e;
		  }
		}
#line 3821 "y.tab.c" /* yacc.c:1661  */
    break;

  case 250:
#line 908 "xi-grammar.y" /* yacc.c:1661  */
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
                  firstRdma = true;
                }
#line 3840 "y.tab.c" /* yacc.c:1661  */
    break;

  case 251:
#line 925 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.accelBlock) = new AccelBlock(lineno, new XStr((yyvsp[-2].strval))); }
#line 3846 "y.tab.c" /* yacc.c:1661  */
    break;

  case 252:
#line 927 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.accelBlock) = new AccelBlock(lineno, NULL); }
#line 3852 "y.tab.c" /* yacc.c:1661  */
    break;

  case 253:
#line 931 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3858 "y.tab.c" /* yacc.c:1661  */
    break;

  case 254:
#line 935 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 0; }
#line 3864 "y.tab.c" /* yacc.c:1661  */
    break;

  case 255:
#line 937 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = (yyvsp[-1].intval); }
#line 3870 "y.tab.c" /* yacc.c:1661  */
    break;

  case 256:
#line 939 "xi-grammar.y" /* yacc.c:1661  */
    { ERROR("invalid entry method attribute list",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
#line 3879 "y.tab.c" /* yacc.c:1661  */
    break;

  case 257:
#line 946 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = (yyvsp[0].intval); }
#line 3885 "y.tab.c" /* yacc.c:1661  */
    break;

  case 258:
#line 948 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = (yyvsp[-2].intval) | (yyvsp[0].intval); }
#line 3891 "y.tab.c" /* yacc.c:1661  */
    break;

  case 259:
#line 952 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = STHREADED; }
#line 3897 "y.tab.c" /* yacc.c:1661  */
    break;

  case 260:
#line 954 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SSYNC; }
#line 3903 "y.tab.c" /* yacc.c:1661  */
    break;

  case 261:
#line 956 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SIGET; }
#line 3909 "y.tab.c" /* yacc.c:1661  */
    break;

  case 262:
#line 958 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SLOCKED; }
#line 3915 "y.tab.c" /* yacc.c:1661  */
    break;

  case 263:
#line 960 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SCREATEHERE; }
#line 3921 "y.tab.c" /* yacc.c:1661  */
    break;

  case 264:
#line 962 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SCREATEHOME; }
#line 3927 "y.tab.c" /* yacc.c:1661  */
    break;

  case 265:
#line 964 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SNOKEEP; }
#line 3933 "y.tab.c" /* yacc.c:1661  */
    break;

  case 266:
#line 966 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SNOTRACE; }
#line 3939 "y.tab.c" /* yacc.c:1661  */
    break;

  case 267:
#line 968 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SAPPWORK; }
#line 3945 "y.tab.c" /* yacc.c:1661  */
    break;

  case 268:
#line 970 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SIMMEDIATE; }
#line 3951 "y.tab.c" /* yacc.c:1661  */
    break;

  case 269:
#line 972 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SSKIPSCHED; }
#line 3957 "y.tab.c" /* yacc.c:1661  */
    break;

  case 270:
#line 974 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SINLINE; }
#line 3963 "y.tab.c" /* yacc.c:1661  */
    break;

  case 271:
#line 976 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SLOCAL; }
#line 3969 "y.tab.c" /* yacc.c:1661  */
    break;

  case 272:
#line 978 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SPYTHON; }
#line 3975 "y.tab.c" /* yacc.c:1661  */
    break;

  case 273:
#line 980 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SMEM; }
#line 3981 "y.tab.c" /* yacc.c:1661  */
    break;

  case 274:
#line 982 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SREDUCE; }
#line 3987 "y.tab.c" /* yacc.c:1661  */
    break;

  case 275:
#line 984 "xi-grammar.y" /* yacc.c:1661  */
    {
        (yyval.intval) = SAGGREGATE;
    }
#line 3995 "y.tab.c" /* yacc.c:1661  */
    break;

  case 276:
#line 988 "xi-grammar.y" /* yacc.c:1661  */
    {
		  ERROR("invalid entry method attribute",
		        (yylsp[0]).first_column, (yylsp[0]).last_column);
		  yyclearin;
		  yyerrok;
		}
#line 4006 "y.tab.c" /* yacc.c:1661  */
    break;

  case 277:
#line 997 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4012 "y.tab.c" /* yacc.c:1661  */
    break;

  case 278:
#line 999 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4018 "y.tab.c" /* yacc.c:1661  */
    break;

  case 279:
#line 1001 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4024 "y.tab.c" /* yacc.c:1661  */
    break;

  case 280:
#line 1005 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = ""; }
#line 4030 "y.tab.c" /* yacc.c:1661  */
    break;

  case 281:
#line 1007 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4036 "y.tab.c" /* yacc.c:1661  */
    break;

  case 282:
#line 1009 "xi-grammar.y" /* yacc.c:1661  */
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s, %s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4046 "y.tab.c" /* yacc.c:1661  */
    break;

  case 283:
#line 1017 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = ""; }
#line 4052 "y.tab.c" /* yacc.c:1661  */
    break;

  case 284:
#line 1019 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4058 "y.tab.c" /* yacc.c:1661  */
    break;

  case 285:
#line 1021 "xi-grammar.y" /* yacc.c:1661  */
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s[%s]%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4068 "y.tab.c" /* yacc.c:1661  */
    break;

  case 286:
#line 1027 "xi-grammar.y" /* yacc.c:1661  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s{%s}%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4078 "y.tab.c" /* yacc.c:1661  */
    break;

  case 287:
#line 1033 "xi-grammar.y" /* yacc.c:1661  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s(%s)%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4088 "y.tab.c" /* yacc.c:1661  */
    break;

  case 288:
#line 1039 "xi-grammar.y" /* yacc.c:1661  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"(%s)%s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4098 "y.tab.c" /* yacc.c:1661  */
    break;

  case 289:
#line 1047 "xi-grammar.y" /* yacc.c:1661  */
    {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval));
		}
#line 4107 "y.tab.c" /* yacc.c:1661  */
    break;

  case 290:
#line 1054 "xi-grammar.y" /* yacc.c:1661  */
    {
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			(yyval.intval) = 0;
		}
#line 4117 "y.tab.c" /* yacc.c:1661  */
    break;

  case 291:
#line 1062 "xi-grammar.y" /* yacc.c:1661  */
    {
			in_braces=0;
			(yyval.intval) = 0;
		}
#line 4126 "y.tab.c" /* yacc.c:1661  */
    break;

  case 292:
#line 1069 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[0].type));}
#line 4132 "y.tab.c" /* yacc.c:1661  */
    break;

  case 293:
#line 1071 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval)); (yyval.pname)->setConditional((yyvsp[0].intval)); }
#line 4138 "y.tab.c" /* yacc.c:1661  */
    break;

  case 294:
#line 1073 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[-3].type),(yyvsp[-2].strval),0,(yyvsp[0].val));}
#line 4144 "y.tab.c" /* yacc.c:1661  */
    break;

  case 295:
#line 1075 "xi-grammar.y" /* yacc.c:1661  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
		}
#line 4153 "y.tab.c" /* yacc.c:1661  */
    break;

  case 296:
#line 1080 "xi-grammar.y" /* yacc.c:1661  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
			(yyval.pname)->setRdma(true);
			if(firstRdma) {
				(yyval.pname)->setFirstRdma(true);
				firstRdma = false;
			}
		}
#line 4167 "y.tab.c" /* yacc.c:1661  */
    break;

  case 297:
#line 1091 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READONLY; }
#line 4173 "y.tab.c" /* yacc.c:1661  */
    break;

  case 298:
#line 1092 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READWRITE; }
#line 4179 "y.tab.c" /* yacc.c:1661  */
    break;

  case 299:
#line 1093 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_WRITEONLY; }
#line 4185 "y.tab.c" /* yacc.c:1661  */
    break;

  case 300:
#line 1096 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.xstrptr) = new XStr((yyvsp[0].strval)); }
#line 4191 "y.tab.c" /* yacc.c:1661  */
    break;

  case 301:
#line 1097 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "->" << (yyvsp[0].strval); }
#line 4197 "y.tab.c" /* yacc.c:1661  */
    break;

  case 302:
#line 1098 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[-2].xstrptr)) << "." << (yyvsp[0].strval); }
#line 4203 "y.tab.c" /* yacc.c:1661  */
    break;

  case 303:
#line 1100 "xi-grammar.y" /* yacc.c:1661  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << *((yyvsp[-1].xstrptr)) << "]";
                  delete (yyvsp[-3].xstrptr);
                  delete (yyvsp[-1].xstrptr);
                }
#line 4214 "y.tab.c" /* yacc.c:1661  */
    break;

  case 304:
#line 1107 "xi-grammar.y" /* yacc.c:1661  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << (yyvsp[-1].strval) << "]";
                  delete (yyvsp[-3].xstrptr);
                }
#line 4224 "y.tab.c" /* yacc.c:1661  */
    break;

  case 305:
#line 1113 "xi-grammar.y" /* yacc.c:1661  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "(" << *((yyvsp[-1].xstrptr)) << ")";
                  delete (yyvsp[-3].xstrptr);
                  delete (yyvsp[-1].xstrptr);
                }
#line 4235 "y.tab.c" /* yacc.c:1661  */
    break;

  case 306:
#line 1122 "xi-grammar.y" /* yacc.c:1661  */
    {
                  in_bracket = 0;
                  (yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName(), (yyvsp[-1].strval));
                }
#line 4244 "y.tab.c" /* yacc.c:1661  */
    break;

  case 307:
#line 1129 "xi-grammar.y" /* yacc.c:1661  */
    {
                  (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[-6].intval));
                }
#line 4254 "y.tab.c" /* yacc.c:1661  */
    break;

  case 308:
#line 1135 "xi-grammar.y" /* yacc.c:1661  */
    {
		  (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType(Parameter::ACCEL_BUFFER_TYPE_READWRITE);
		}
#line 4264 "y.tab.c" /* yacc.c:1661  */
    break;

  case 309:
#line 1141 "xi-grammar.y" /* yacc.c:1661  */
    {
                  (yyval.pname) = (yyvsp[-3].pname);
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[-5].intval));
		}
#line 4274 "y.tab.c" /* yacc.c:1661  */
    break;

  case 310:
#line 1149 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 4280 "y.tab.c" /* yacc.c:1661  */
    break;

  case 311:
#line 1151 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 4286 "y.tab.c" /* yacc.c:1661  */
    break;

  case 312:
#line 1155 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 4292 "y.tab.c" /* yacc.c:1661  */
    break;

  case 313:
#line 1157 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 4298 "y.tab.c" /* yacc.c:1661  */
    break;

  case 314:
#line 1161 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.plist) = (yyvsp[-1].plist); }
#line 4304 "y.tab.c" /* yacc.c:1661  */
    break;

  case 315:
#line 1163 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.plist) = new ParamList(new Parameter(0, new BuiltinType("void"))); }
#line 4310 "y.tab.c" /* yacc.c:1661  */
    break;

  case 316:
#line 1167 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.plist) = (yyvsp[-1].plist); }
#line 4316 "y.tab.c" /* yacc.c:1661  */
    break;

  case 317:
#line 1169 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.plist) = 0; }
#line 4322 "y.tab.c" /* yacc.c:1661  */
    break;

  case 318:
#line 1173 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.val) = 0; }
#line 4328 "y.tab.c" /* yacc.c:1661  */
    break;

  case 319:
#line 1175 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4334 "y.tab.c" /* yacc.c:1661  */
    break;

  case 320:
#line 1179 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sentry) = 0; }
#line 4340 "y.tab.c" /* yacc.c:1661  */
    break;

  case 321:
#line 1181 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[0].sc)); }
#line 4346 "y.tab.c" /* yacc.c:1661  */
    break;

  case 322:
#line 1183 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[-2].slist)); }
#line 4352 "y.tab.c" /* yacc.c:1661  */
    break;

  case 323:
#line 1187 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.slist) = new SListConstruct((yyvsp[0].sc)); }
#line 4358 "y.tab.c" /* yacc.c:1661  */
    break;

  case 324:
#line 1189 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.slist) = new SListConstruct((yyvsp[-1].sc), (yyvsp[0].slist));  }
#line 4364 "y.tab.c" /* yacc.c:1661  */
    break;

  case 325:
#line 1193 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.olist) = new OListConstruct((yyvsp[0].sc)); }
#line 4370 "y.tab.c" /* yacc.c:1661  */
    break;

  case 326:
#line 1195 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.olist) = new OListConstruct((yyvsp[-1].sc), (yyvsp[0].slist)); }
#line 4376 "y.tab.c" /* yacc.c:1661  */
    break;

  case 327:
#line 1199 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.clist) = new CaseListConstruct((yyvsp[0].when)); }
#line 4382 "y.tab.c" /* yacc.c:1661  */
    break;

  case 328:
#line 1201 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.clist) = new CaseListConstruct((yyvsp[-1].when), (yyvsp[0].clist)); }
#line 4388 "y.tab.c" /* yacc.c:1661  */
    break;

  case 329:
#line 1203 "xi-grammar.y" /* yacc.c:1661  */
    {
		  ERROR("case blocks can only contain when clauses",
		        (yylsp[0]).first_column, (yylsp[0]).last_column);
		  (yyval.clist) = 0;
		}
#line 4398 "y.tab.c" /* yacc.c:1661  */
    break;

  case 330:
#line 1211 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4404 "y.tab.c" /* yacc.c:1661  */
    break;

  case 331:
#line 1213 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = 0; }
#line 4410 "y.tab.c" /* yacc.c:1661  */
    break;

  case 332:
#line 1217 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = new WhenConstruct((yyvsp[-2].entrylist), 0); }
#line 4416 "y.tab.c" /* yacc.c:1661  */
    break;

  case 333:
#line 1219 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = new WhenConstruct((yyvsp[-1].entrylist), (yyvsp[0].sc)); }
#line 4422 "y.tab.c" /* yacc.c:1661  */
    break;

  case 334:
#line 1221 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = new WhenConstruct((yyvsp[-3].entrylist), (yyvsp[-1].slist)); }
#line 4428 "y.tab.c" /* yacc.c:1661  */
    break;

  case 335:
#line 1225 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4434 "y.tab.c" /* yacc.c:1661  */
    break;

  case 336:
#line 1227 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4440 "y.tab.c" /* yacc.c:1661  */
    break;

  case 337:
#line 1229 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4446 "y.tab.c" /* yacc.c:1661  */
    break;

  case 338:
#line 1231 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4452 "y.tab.c" /* yacc.c:1661  */
    break;

  case 339:
#line 1233 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4458 "y.tab.c" /* yacc.c:1661  */
    break;

  case 340:
#line 1235 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4464 "y.tab.c" /* yacc.c:1661  */
    break;

  case 341:
#line 1237 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4470 "y.tab.c" /* yacc.c:1661  */
    break;

  case 342:
#line 1239 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4476 "y.tab.c" /* yacc.c:1661  */
    break;

  case 343:
#line 1241 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4482 "y.tab.c" /* yacc.c:1661  */
    break;

  case 344:
#line 1243 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4488 "y.tab.c" /* yacc.c:1661  */
    break;

  case 345:
#line 1245 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4494 "y.tab.c" /* yacc.c:1661  */
    break;

  case 346:
#line 1247 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4500 "y.tab.c" /* yacc.c:1661  */
    break;

  case 347:
#line 1251 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new SerialConstruct((yyvsp[-2].strval), (yyvsp[-4].strval), (yylsp[-3]).first_line); }
#line 4506 "y.tab.c" /* yacc.c:1661  */
    break;

  case 348:
#line 1253 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new OverlapConstruct((yyvsp[-1].olist)); }
#line 4512 "y.tab.c" /* yacc.c:1661  */
    break;

  case 349:
#line 1255 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = (yyvsp[0].when); }
#line 4518 "y.tab.c" /* yacc.c:1661  */
    break;

  case 350:
#line 1257 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new CaseConstruct((yyvsp[-1].clist)); }
#line 4524 "y.tab.c" /* yacc.c:1661  */
    break;

  case 351:
#line 1259 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new ForConstruct((yyvsp[-8].intexpr), (yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4530 "y.tab.c" /* yacc.c:1661  */
    break;

  case 352:
#line 1261 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new ForConstruct((yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4536 "y.tab.c" /* yacc.c:1661  */
    break;

  case 353:
#line 1263 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-9].strval)), (yyvsp[-6].intexpr),
		             (yyvsp[-4].intexpr), (yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4543 "y.tab.c" /* yacc.c:1661  */
    break;

  case 354:
#line 1266 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-11].strval)), (yyvsp[-8].intexpr),
		             (yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4550 "y.tab.c" /* yacc.c:1661  */
    break;

  case 355:
#line 1269 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new IfConstruct((yyvsp[-3].intexpr), (yyvsp[-1].sc), (yyvsp[0].sc)); }
#line 4556 "y.tab.c" /* yacc.c:1661  */
    break;

  case 356:
#line 1271 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new IfConstruct((yyvsp[-5].intexpr), (yyvsp[-2].slist), (yyvsp[0].sc)); }
#line 4562 "y.tab.c" /* yacc.c:1661  */
    break;

  case 357:
#line 1273 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new WhileConstruct((yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4568 "y.tab.c" /* yacc.c:1661  */
    break;

  case 358:
#line 1275 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new WhileConstruct((yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4574 "y.tab.c" /* yacc.c:1661  */
    break;

  case 359:
#line 1277 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new SerialConstruct((yyvsp[-2].strval), NULL, (yyloc).first_line); }
#line 4580 "y.tab.c" /* yacc.c:1661  */
    break;

  case 360:
#line 1279 "xi-grammar.y" /* yacc.c:1661  */
    {
		  ERROR("unknown SDAG construct or malformed entry method declaration.\n"
		        "You may have forgotten to terminate a previous entry method declaration with a"
		        " semicolon or forgotten to mark a block of sequential SDAG code as 'serial'",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
#line 4592 "y.tab.c" /* yacc.c:1661  */
    break;

  case 361:
#line 1289 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = 0; }
#line 4598 "y.tab.c" /* yacc.c:1661  */
    break;

  case 362:
#line 1291 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new ElseConstruct((yyvsp[0].sc)); }
#line 4604 "y.tab.c" /* yacc.c:1661  */
    break;

  case 363:
#line 1293 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new ElseConstruct((yyvsp[-1].slist)); }
#line 4610 "y.tab.c" /* yacc.c:1661  */
    break;

  case 364:
#line 1297 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intexpr) = new IntExprConstruct((yyvsp[0].strval)); }
#line 4616 "y.tab.c" /* yacc.c:1661  */
    break;

  case 365:
#line 1301 "xi-grammar.y" /* yacc.c:1661  */
    { in_int_expr = 0; (yyval.intval) = 0; }
#line 4622 "y.tab.c" /* yacc.c:1661  */
    break;

  case 366:
#line 1305 "xi-grammar.y" /* yacc.c:1661  */
    { in_int_expr = 1; (yyval.intval) = 0; }
#line 4628 "y.tab.c" /* yacc.c:1661  */
    break;

  case 367:
#line 1309 "xi-grammar.y" /* yacc.c:1661  */
    {
		  (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[-1].strval), (yyvsp[0].plist), 0, 0, 0, (yyloc).first_line, (yyloc).last_line);
		  firstRdma = true;
		}
#line 4637 "y.tab.c" /* yacc.c:1661  */
    break;

  case 368:
#line 1314 "xi-grammar.y" /* yacc.c:1661  */
    {
		  (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[-4].strval), (yyvsp[0].plist), 0, 0, (yyvsp[-2].strval), (yyloc).first_line, (yyloc).last_line);
		  firstRdma = true;
		}
#line 4646 "y.tab.c" /* yacc.c:1661  */
    break;

  case 369:
#line 1321 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.entrylist) = new EntryList((yyvsp[0].entry)); }
#line 4652 "y.tab.c" /* yacc.c:1661  */
    break;

  case 370:
#line 1323 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.entrylist) = new EntryList((yyvsp[-2].entry),(yyvsp[0].entrylist)); }
#line 4658 "y.tab.c" /* yacc.c:1661  */
    break;

  case 371:
#line 1327 "xi-grammar.y" /* yacc.c:1661  */
    { in_bracket=1; }
#line 4664 "y.tab.c" /* yacc.c:1661  */
    break;

  case 372:
#line 1330 "xi-grammar.y" /* yacc.c:1661  */
    { in_bracket=0; }
#line 4670 "y.tab.c" /* yacc.c:1661  */
    break;

  case 373:
#line 1334 "xi-grammar.y" /* yacc.c:1661  */
    { if (!macroDefined((yyvsp[0].strval), 1)) in_comment = 1; }
#line 4676 "y.tab.c" /* yacc.c:1661  */
    break;

  case 374:
#line 1338 "xi-grammar.y" /* yacc.c:1661  */
    { if (!macroDefined((yyvsp[0].strval), 0)) in_comment = 1; }
#line 4682 "y.tab.c" /* yacc.c:1661  */
    break;


#line 4686 "y.tab.c" /* yacc.c:1661  */
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
#line 1341 "xi-grammar.y" /* yacc.c:1906  */


void yyerror(const char *s)
{
	fprintf(stderr, "[PARSE-ERROR] Unexpected/missing token at line %d. Current token being parsed: '%s'.\n", lineno, yytext);
}
