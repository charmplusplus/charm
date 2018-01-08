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
#define YYLAST   1501

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  91
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  117
/* YYNRULES -- Number of rules.  */
#define YYNRULES  373
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  728

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
     291,   298,   302,   309,   311,   316,   317,   321,   323,   325,
     327,   329,   342,   344,   346,   348,   354,   356,   358,   360,
     362,   364,   366,   368,   370,   372,   380,   382,   384,   388,
     390,   395,   396,   401,   402,   406,   408,   410,   412,   414,
     416,   418,   420,   422,   424,   426,   428,   430,   432,   434,
     438,   439,   446,   448,   452,   456,   458,   462,   466,   468,
     470,   472,   474,   476,   480,   482,   484,   486,   488,   492,
     494,   496,   500,   502,   504,   508,   512,   517,   518,   522,
     526,   531,   532,   537,   538,   548,   550,   554,   556,   561,
     562,   566,   568,   573,   574,   578,   583,   584,   588,   590,
     594,   596,   601,   602,   606,   607,   610,   614,   616,   620,
     622,   624,   629,   630,   634,   636,   640,   642,   646,   650,
     654,   660,   664,   666,   670,   672,   676,   680,   684,   688,
     690,   695,   696,   701,   702,   704,   706,   715,   717,   719,
     723,   725,   729,   733,   735,   737,   739,   741,   745,   747,
     752,   759,   763,   765,   767,   768,   770,   772,   774,   778,
     780,   782,   788,   794,   803,   805,   807,   813,   821,   823,
     826,   830,   834,   836,   841,   843,   851,   853,   855,   857,
     859,   861,   863,   865,   867,   869,   871,   874,   884,   901,
     918,   920,   924,   929,   930,   932,   939,   941,   945,   947,
     949,   951,   953,   955,   957,   959,   961,   963,   965,   967,
     969,   971,   973,   975,   977,   981,   990,   992,   994,   999,
    1000,  1002,  1011,  1012,  1014,  1020,  1026,  1032,  1040,  1047,
    1055,  1062,  1064,  1066,  1068,  1073,  1085,  1086,  1087,  1090,
    1091,  1092,  1093,  1100,  1106,  1115,  1122,  1128,  1134,  1142,
    1144,  1148,  1150,  1154,  1156,  1160,  1162,  1167,  1168,  1172,
    1174,  1176,  1180,  1182,  1186,  1188,  1192,  1194,  1196,  1204,
    1207,  1210,  1212,  1214,  1218,  1220,  1222,  1224,  1226,  1228,
    1230,  1232,  1234,  1236,  1238,  1240,  1244,  1246,  1248,  1250,
    1252,  1254,  1256,  1259,  1262,  1264,  1266,  1268,  1270,  1272,
    1283,  1284,  1286,  1290,  1294,  1298,  1302,  1307,  1314,  1316,
    1320,  1323,  1327,  1331
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

#define YYPACT_NINF -602

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-602)))

#define YYTABLE_NINF -325

#define yytable_value_is_error(Yytable_value) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
     165,  1346,  1346,    37,  -602,   165,  -602,  -602,  -602,  -602,
    -602,  -602,  -602,  -602,  -602,  -602,  -602,  -602,  -602,  -602,
    -602,  -602,  -602,  -602,  -602,  -602,  -602,  -602,  -602,  -602,
    -602,  -602,  -602,  -602,  -602,  -602,  -602,  -602,  -602,  -602,
    -602,  -602,  -602,  -602,  -602,  -602,  -602,  -602,  -602,  -602,
    -602,  -602,  -602,  -602,    67,    67,  -602,  -602,  -602,   770,
      -6,  -602,  -602,  -602,    95,  1346,   254,  1346,  1346,   114,
     934,    71,   912,   770,  -602,  -602,  -602,  -602,   789,   122,
     117,  -602,   127,  -602,  -602,  -602,    -6,   -32,   606,   171,
     171,   -15,   117,   128,   128,   128,   128,   134,   137,  1346,
     177,   164,   770,  -602,  -602,  -602,  -602,  -602,  -602,  -602,
    -602,   217,  -602,  -602,  -602,  -602,   176,  -602,  -602,  -602,
    -602,  -602,  -602,  -602,  -602,  -602,  -602,  -602,    -6,  -602,
    -602,  -602,   789,  -602,   -13,  -602,  -602,  -602,  -602,   238,
      48,  -602,  -602,   195,   215,   221,    23,  -602,   117,   770,
     127,   198,    99,   -32,   226,   499,  1434,   195,   215,   221,
    -602,    69,   117,  -602,   117,   117,   242,   117,   240,  -602,
       1,  1346,  1346,  1346,  1346,  1130,   246,   248,   155,  1346,
    -602,  -602,  -602,  1366,   258,   128,   128,   128,   128,   246,
     137,  -602,  -602,  -602,  -602,  -602,    -6,  -602,   300,  -602,
    -602,  -602,   206,  -602,  -602,  1400,  -602,  -602,  -602,  -602,
    -602,   255,  1346,   261,   288,   -32,   286,   -32,   263,  -602,
     176,   266,    10,  -602,   270,   269,    34,    42,    77,   271,
     163,   117,  -602,  -602,   277,   278,   276,   290,   290,   290,
     290,  -602,  1346,   281,   293,   296,  1202,  1346,   329,  1346,
    -602,  -602,   301,   308,   313,  1346,    50,  1346,   310,   314,
     176,  1346,  1346,  1346,  1346,  1346,  1346,  -602,  -602,  -602,
    -602,   315,  -602,   316,  -602,  -602,   276,  -602,  -602,   320,
     321,   307,   306,   -32,    -6,   117,  1346,  -602,  -602,   317,
    -602,   -32,   171,  1400,   171,   171,  1400,   171,  -602,  -602,
       1,  -602,   117,   245,   245,   245,   245,   318,  -602,   329,
    -602,   290,   290,  -602,   155,     7,   319,   232,  -602,   326,
    1366,  -602,  -602,   290,   290,   290,   290,   290,   256,  1400,
    -602,   323,   -32,   286,   -32,   -32,  -602,    34,   331,  -602,
     330,  -602,   332,   337,   339,   117,   343,   342,  -602,   348,
    -602,   403,    -6,  -602,  -602,  -602,  -602,  -602,  -602,   245,
     245,  -602,  -602,  -602,  1434,    11,   324,  1434,  -602,  -602,
    -602,  -602,  -602,  -602,   245,   245,   245,   245,   245,   411,
      -6,  -602,  1385,  -602,  -602,  -602,  -602,  -602,  -602,   346,
    -602,  -602,  -602,   349,  -602,    60,   351,  -602,   117,  -602,
     686,   393,   358,   176,   403,  -602,  -602,  -602,  -602,  1346,
    -602,  -602,  -602,  -602,  -602,  -602,  -602,  -602,   359,  1434,
    -602,  1346,   -32,   362,   360,  1421,   171,   171,   171,  -602,
    -602,   950,  1058,  -602,   176,    -6,  -602,   356,   176,  1346,
     -32,     8,   361,  1421,  -602,   363,   365,   366,   368,  -602,
    -602,  -602,  -602,  -602,  -602,  -602,  -602,  -602,  -602,  -602,
    -602,  -602,  -602,   389,  -602,   373,  -602,  -602,   374,   383,
     370,   323,  1346,  -602,   376,   176,    -6,   378,   379,  -602,
     272,  -602,  -602,  -602,  -602,  -602,  -602,  -602,  -602,  -602,
     430,  -602,  1003,   481,   323,  -602,    -6,  -602,  -602,  -602,
     127,  -602,  1346,  -602,  -602,   385,   384,   385,   414,   395,
     416,   385,   399,   247,    -6,   -32,  -602,  -602,  -602,   458,
     323,  -602,   -32,   424,   -32,    26,   401,   536,   546,  -602,
     406,   -32,  1362,   415,   442,   226,   404,   481,   408,  -602,
     421,   413,   419,  -602,   -32,   414,   268,  -602,   426,   497,
     -32,   419,   385,   422,   385,   428,   416,   385,   432,   -32,
     429,  1362,  -602,   176,  -602,   176,   454,  -602,   312,   406,
     -32,   385,  -602,   608,   330,  -602,  -602,   433,  -602,  -602,
     226,   754,   -32,   457,   -32,   546,   406,   -32,  1362,   226,
    -602,  -602,  -602,  -602,  -602,  -602,  -602,  -602,  -602,  1346,
     437,   436,   435,   -32,   459,   -32,   247,  -602,   323,  -602,
     176,   247,   467,   460,   461,   419,   472,   -32,   419,   473,
     176,   477,  1434,   570,  -602,   226,   -32,   476,   475,  -602,
    -602,   489,   761,  -602,   -32,   385,   768,  -602,   226,   817,
    -602,  -602,  1346,  1346,   -32,   492,  -602,  1346,   419,   -32,
    -602,   467,   247,  -602,   494,   -32,   247,  -602,   176,   247,
     467,  -602,   111,    81,   486,  1346,   176,   824,   498,  -602,
     509,   -32,   512,   518,  -602,   519,  -602,  -602,  1346,  1274,
     520,  1346,  1346,  -602,   126,    -6,   247,  -602,   -32,  -602,
     419,   -32,  -602,   467,   169,   513,   192,  1346,  -602,   136,
    -602,   521,   419,   831,   523,  -602,  -602,  -602,  -602,  -602,
    -602,  -602,   838,   247,  -602,   -32,   247,  -602,   525,   419,
     526,  -602,   887,  -602,   247,  -602,   527,  -602
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
      63,    61,    62,    85,     6,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    84,    82,    83,     8,     0,     0,
       0,    59,    68,   372,   373,   289,   251,   282,     0,   141,
     141,   141,     0,   149,   149,   149,   149,     0,   143,     0,
       0,     0,     0,    76,   212,   213,    70,    77,    78,    79,
      80,     0,    81,    69,   215,   214,     9,   246,   238,   239,
     240,   241,   242,   244,   245,   243,   236,   237,    74,    75,
      66,   109,     0,    95,    96,    97,    98,   106,   107,     0,
      93,   112,   113,   124,   125,   126,   131,   252,     0,     0,
      67,     0,   283,   282,     0,     0,     0,   118,   119,   120,
     121,   134,     0,   142,     0,     0,     0,     0,   228,   216,
       0,     0,     0,     0,     0,     0,     0,   156,     0,     0,
     218,   230,   217,     0,     0,   149,   149,   149,   149,     0,
     143,   203,   204,   205,   206,   207,    10,    64,   127,   105,
     108,    99,   100,   103,   104,    91,   111,   114,   115,   116,
     128,   130,     0,     0,     0,   282,   279,   282,     0,   290,
       0,     0,   122,   123,     0,   133,   137,   222,   219,     0,
     224,     0,   160,   161,     0,   151,    93,   172,   172,   172,
     172,   155,     0,     0,   158,     0,     0,     0,     0,     0,
     147,   148,     0,   145,   169,     0,   121,     0,   200,     0,
       9,     0,     0,     0,     0,     0,     0,   101,   102,    87,
      88,    89,    92,     0,    86,   129,    93,    73,    60,     0,
     280,     0,     0,   282,   250,     0,     0,   132,   370,   137,
     139,   282,   141,     0,   141,   141,     0,   141,   229,   150,
       0,   110,     0,     0,     0,     0,     0,     0,   181,     0,
     157,   172,   172,   144,     0,   162,   191,     0,   198,   193,
       0,   202,    72,   172,   172,   172,   172,   172,     0,     0,
      94,     0,   282,   279,   282,   282,   287,   137,     0,   138,
       0,   135,     0,     0,     0,     0,     0,     0,   152,   174,
     173,     0,   208,   176,   177,   178,   179,   180,   159,     0,
       0,   146,   163,   170,     0,   162,     0,     0,   197,   194,
     195,   196,   199,   201,     0,     0,     0,     0,     0,   162,
     189,    90,     0,    71,   285,   281,   286,   284,   140,     0,
     371,   136,   223,     0,   220,     0,     0,   225,     0,   235,
       0,     0,     0,     0,     0,   231,   232,   182,   183,     0,
     168,   171,   192,   184,   185,   186,   187,   188,     0,     0,
     314,   291,   282,   309,     0,     0,   141,   141,   141,   175,
     255,     0,     0,   233,     9,   234,   211,   164,     0,     0,
     282,   162,     0,     0,   313,     0,     0,     0,     0,   275,
     258,   259,   260,   261,   267,   268,   269,   274,   262,   263,
     264,   265,   266,   153,   270,     0,   272,   273,     0,   256,
      59,     0,     0,   209,     0,     0,   190,     0,     0,   288,
       0,   292,   294,   310,   117,   221,   227,   226,   154,   271,
       0,   254,     0,     0,     0,   165,   166,   295,   277,   276,
     278,   293,     0,   257,   359,     0,     0,     0,     0,     0,
     330,     0,     0,     0,   319,   282,   248,   348,   320,   317,
       0,   365,   282,     0,   282,     0,   368,     0,     0,   329,
       0,   282,     0,     0,     0,     0,     0,     0,     0,   363,
       0,     0,     0,   366,   282,     0,     0,   332,     0,     0,
     282,     0,     0,     0,     0,     0,   330,     0,     0,   282,
       0,   326,   328,     9,   323,     9,     0,   247,     0,     0,
     282,     0,   364,     0,     0,   369,   331,     0,   347,   325,
       0,     0,   282,     0,   282,     0,     0,   282,     0,     0,
     349,   327,   321,   358,   318,   296,   297,   298,   316,     0,
       0,   311,     0,   282,     0,   282,     0,   356,     0,   333,
       9,     0,   360,     0,     0,     0,     0,   282,     0,     0,
       9,     0,     0,     0,   315,     0,   282,     0,     0,   367,
     346,     0,     0,   354,   282,     0,     0,   335,     0,     0,
     336,   345,     0,     0,   282,     0,   312,     0,     0,   282,
     357,   360,     0,   361,     0,   282,     0,   343,     9,     0,
     360,   299,     0,     0,     0,     0,     0,     0,     0,   355,
       0,   282,     0,     0,   334,     0,   341,   307,     0,     0,
       0,     0,     0,   305,     0,   249,     0,   351,   282,   362,
       0,   282,   344,   360,     0,     0,     0,     0,   301,     0,
     308,     0,     0,     0,     0,   342,   304,   303,   302,   300,
     306,   350,     0,     0,   338,   282,     0,   352,     0,     0,
       0,   337,     0,   353,     0,   339,     0,   340
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -602,  -602,   607,  -602,   -42,  -257,    -1,   -61,   541,   559,
     -44,  -602,  -602,  -602,  -174,  -602,  -204,  -602,  -133,   -75,
     -72,   -70,   -67,  -173,   462,   483,  -602,   -84,  -602,  -602,
    -274,  -602,  -602,   -57,   427,   305,  -602,    85,   327,  -602,
    -602,   444,   336,  -602,   185,  -602,  -602,  -323,  -602,  -161,
     230,  -602,  -602,  -602,   -74,  -602,  -602,  -602,  -602,  -602,
    -602,  -602,   328,  -602,   309,   568,  -602,    80,   244,   579,
    -602,  -602,   431,  -602,  -602,  -602,  -602,   249,  -602,   227,
    -602,   166,  -602,  -602,   333,   -85,  -412,   -55,  -492,  -602,
    -602,  -495,  -602,  -602,  -389,    40,  -450,  -602,  -602,   123,
    -511,    89,  -543,   109,  -475,  -602,  -446,  -601,  -515,  -531,
    -431,  -602,   130,   151,   105,  -602,  -602
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     3,     4,    70,   352,   197,   236,   140,     5,    61,
      71,    72,    73,   271,   272,   273,   206,   141,   237,   142,
     157,   158,   159,   160,   161,   146,   147,   274,   340,   289,
     290,   104,   105,   164,   179,   252,   253,   171,   234,   489,
     244,   176,   245,   235,   364,   475,   365,   366,   106,   303,
     350,   107,   108,   109,   177,   110,   191,   192,   193,   194,
     195,   368,   318,   258,   259,   401,   112,   353,   402,   403,
     114,   115,   169,   182,   404,   405,   129,   406,    74,   148,
     432,   468,   469,   501,   281,   539,   422,   515,   220,   423,
     600,   662,   645,   601,   424,   602,   383,   569,   537,   516,
     533,   548,   560,   530,   517,   562,   534,   633,   540,   573,
     522,   526,   527,   291,   391,    75,    76
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[] =
{
      54,    55,   154,   322,   162,    82,   143,   440,   144,   542,
     256,   145,    60,    60,    87,   339,   551,   168,   591,   150,
     581,   493,   163,   564,   362,   362,   152,    86,   362,   130,
     128,   232,   301,   165,   167,   577,   445,    56,   579,   238,
     239,   240,   410,   565,   519,   619,   254,   518,   223,   199,
     669,   153,   233,   200,   483,   604,   418,   561,   184,   676,
     143,   210,   144,   388,    79,   145,    83,    84,   218,    77,
     538,   166,   331,   212,   196,   543,   524,   304,   305,   306,
     531,   547,   549,   221,   636,   363,   561,   639,   610,  -167,
     627,   518,   705,   224,   479,   628,   480,   620,   180,   257,
     631,   226,   247,   227,   228,   213,   230,   223,   211,   382,
     151,   648,   288,   561,   311,   265,   312,   667,   151,   343,
     288,   582,   346,   584,   151,   292,   587,   607,   205,  -193,
     279,  -193,   282,   647,   668,   612,   151,    78,   317,   549,
     605,   670,    58,   427,    59,   673,   658,   256,   675,   116,
     359,   360,   224,   151,   225,   381,   168,   293,   629,   703,
     294,   682,   374,   375,   376,   377,   378,   479,     1,     2,
     684,   712,    81,   702,   243,   701,   215,   473,   284,   172,
     173,   174,   216,   694,   696,   217,   653,   699,   722,    58,
     657,    85,   677,   660,   678,   250,   251,   679,   336,   149,
     680,   681,   718,   151,   655,   720,   341,   700,   163,   678,
     644,   276,   679,   726,   170,   680,   681,   710,   196,   678,
     175,   687,   679,   178,   337,   680,   681,   349,   185,   186,
     187,   188,   189,   190,   181,   342,   257,   344,   345,   151,
     347,   307,   371,   296,   183,   243,   297,   384,   504,   386,
     387,    58,   678,   706,   316,   679,   319,   714,   680,   681,
     323,   324,   325,   326,   327,   328,   717,   267,   268,   504,
     261,   262,   263,   264,   214,   678,   725,   207,   679,   708,
     409,   680,   681,   412,   395,   338,   380,    81,   369,   370,
     505,   506,   507,   508,   509,   510,   511,   208,   421,   201,
     202,   203,   204,   209,   219,  -289,   592,    80,   593,    81,
     229,   505,   506,   507,   508,   509,   510,   511,   595,   231,
      58,   512,   351,   349,    85,  -289,  -289,    81,   498,   499,
    -289,    58,   246,   379,   248,   439,   260,   442,   210,   277,
     275,   421,   512,   278,   280,    85,   576,   283,   285,   131,
     156,  -289,   286,   630,   287,   478,   205,   300,   295,   421,
     143,   435,   144,   641,   299,   145,   302,    81,   308,   446,
     447,   448,   309,   133,   134,   135,   136,   137,   138,   139,
     241,   596,   597,   310,   354,   355,   356,   314,   313,   320,
     315,   334,   196,   335,   329,   321,   476,   330,   332,   598,
     333,   674,   411,   288,   399,   357,   382,   367,   437,    88,
      89,    90,    91,    92,   317,   389,   392,   390,   393,   500,
     441,    99,   100,   394,   396,   101,   397,   398,   362,   425,
     535,   471,   426,   496,   428,   400,   434,   438,   477,   407,
     408,   443,   474,   504,   444,   400,   488,   484,   482,   485,
     486,   514,   487,   -11,   413,   414,   415,   416,   417,   574,
     490,   491,   492,   495,   479,   580,   497,   502,   521,   525,
     523,   494,   528,   529,   589,   550,   532,   559,   536,   541,
     545,  -210,   504,    85,   599,   505,   506,   507,   508,   509,
     510,   511,   566,   563,   568,   514,   570,   613,   504,   615,
     571,   520,   618,   572,   578,   585,   559,   590,   583,   588,
     594,   609,   614,   622,   603,   623,   512,   632,   625,    85,
    -322,   196,   624,   196,   505,   506,   507,   508,   509,   510,
     511,   617,   638,   559,   626,   634,   131,   504,   643,   599,
     505,   506,   507,   508,   509,   510,   511,   504,   635,   654,
     637,   640,   649,   650,    81,   512,    58,   642,   513,   664,
     133,   134,   135,   136,   137,   138,   139,   651,   196,   671,
     672,   512,   665,   683,    85,  -324,   595,   688,   196,   505,
     506,   507,   508,   509,   510,   511,   690,   689,   691,   505,
     506,   507,   508,   509,   510,   511,   692,   693,   621,   711,
     707,   697,   715,   721,   723,   727,   704,   131,   156,   504,
     512,   103,    57,   546,    62,   198,   196,   266,   222,   361,
     512,   249,   155,    85,   685,    81,   481,   348,   429,   373,
     719,   133,   134,   135,   136,   137,   138,   139,   111,   596,
     597,   661,   663,   131,   156,   358,   666,   372,   436,   113,
     433,   505,   506,   507,   508,   509,   510,   511,   503,   472,
     567,    81,   298,   646,   661,   586,   385,   133,   134,   135,
     136,   137,   138,   139,   616,   575,   544,   661,   661,   608,
     698,   661,   512,     0,     0,   606,     0,   430,     0,  -253,
    -253,  -253,     0,  -253,  -253,  -253,   709,  -253,  -253,  -253,
    -253,  -253,     0,     0,     0,  -253,  -253,  -253,  -253,  -253,
    -253,  -253,  -253,  -253,  -253,  -253,  -253,     0,  -253,  -253,
    -253,  -253,  -253,  -253,  -253,  -253,  -253,  -253,  -253,  -253,
    -253,  -253,  -253,  -253,  -253,  -253,  -253,     0,  -253,     0,
    -253,  -253,     0,     0,     0,     0,     0,  -253,  -253,  -253,
    -253,  -253,  -253,  -253,  -253,   504,     0,  -253,  -253,  -253,
    -253,     0,   504,     0,     0,     0,     0,     0,     0,   504,
       0,    63,   431,    -5,    -5,    64,    -5,    -5,    -5,    -5,
      -5,    -5,    -5,    -5,    -5,    -5,    -5,     0,    -5,    -5,
       0,     0,    -5,     0,     0,     0,     0,   505,   506,   507,
     508,   509,   510,   511,   505,   506,   507,   508,   509,   510,
     511,   505,   506,   507,   508,   509,   510,   511,   504,     0,
       0,     0,     0,    65,    66,   504,   131,   132,   512,    67,
      68,   611,   504,     0,     0,   512,     0,     0,   652,   504,
       0,    69,   512,     0,    81,   656,     0,    -5,   -65,     0,
     133,   134,   135,   136,   137,   138,   139,     0,     0,     0,
     505,   506,   507,   508,   509,   510,   511,   505,   506,   507,
     508,   509,   510,   511,   505,   506,   507,   508,   509,   510,
     511,   505,   506,   507,   508,   509,   510,   511,   504,     0,
       0,   512,     0,     0,   659,     0,     0,     0,   512,     0,
       0,   686,     0,     0,     0,   512,     0,     0,   713,     0,
       0,     0,   512,     0,     0,   716,     0,     0,   117,   118,
     119,   120,     0,   121,   122,   123,   124,   125,     0,     0,
     505,   506,   507,   508,   509,   510,   511,     1,     2,     0,
      88,    89,    90,    91,    92,    93,    94,    95,    96,    97,
      98,   449,    99,   100,   126,     0,   101,     0,     0,     0,
       0,   512,     0,     0,   724,     0,     0,     0,     0,     0,
       0,   450,     0,   451,   452,   453,   454,   455,   456,     0,
       0,   457,   458,   459,   460,   461,   462,    58,     0,     0,
     127,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   463,   464,     0,   449,     0,     0,     0,     0,     0,
       0,   102,     0,     0,     0,     0,     0,     0,   465,     0,
       0,     0,   466,   467,   450,     0,   451,   452,   453,   454,
     455,   456,     0,     0,   457,   458,   459,   460,   461,   462,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   463,   464,     0,     0,     0,     0,
       0,     6,     7,     8,     0,     9,    10,    11,     0,    12,
      13,    14,    15,    16,     0,   466,   467,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,     0,
      29,    30,    31,    32,    33,   131,   132,    34,    35,    36,
      37,    38,    39,    40,    41,    42,    43,    44,    45,     0,
      46,     0,    47,   470,     0,     0,     0,     0,     0,   133,
     134,   135,   136,   137,   138,   139,    49,     0,     0,    50,
      51,    52,    53,     6,     7,     8,     0,     9,    10,    11,
       0,    12,    13,    14,    15,    16,     0,     0,     0,    17,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      28,     0,    29,    30,    31,    32,    33,     0,     0,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,   241,    46,     0,    47,    48,   242,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    49,     0,
       0,    50,    51,    52,    53,     6,     7,     8,     0,     9,
      10,    11,     0,    12,    13,    14,    15,    16,     0,     0,
       0,    17,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,     0,    29,    30,    31,    32,    33,     0,
       0,    34,    35,    36,    37,    38,    39,    40,    41,    42,
      43,    44,    45,     0,    46,     0,    47,    48,   242,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      49,     0,     0,    50,    51,    52,    53,     6,     7,     8,
       0,     9,    10,    11,     0,    12,    13,    14,    15,    16,
       0,     0,     0,    17,    18,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,     0,    29,    30,    31,    32,
      33,     0,     0,    34,    35,    36,    37,    38,    39,    40,
      41,    42,    43,    44,    45,     0,    46,     0,    47,    48,
     695,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    49,     0,     0,    50,    51,    52,    53,     6,
       7,     8,     0,     9,    10,    11,     0,    12,    13,    14,
      15,    16,     0,     0,     0,    17,    18,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,     0,    29,    30,
      31,    32,    33,     0,   255,    34,    35,    36,    37,    38,
      39,    40,    41,    42,    43,    44,    45,     0,    46,     0,
      47,    48,     0,   131,   156,   552,   553,   554,   508,   555,
     556,   557,     0,     0,    49,     0,     0,    50,    51,    52,
      53,    81,   131,   156,   419,     0,     0,   133,   134,   135,
     136,   137,   138,   139,     0,     0,   558,   131,   156,    85,
      81,     0,     0,     0,     0,     0,   133,   134,   135,   136,
     137,   138,   139,     0,     0,    81,   269,   270,   131,   156,
     419,   133,   134,   135,   136,   137,   138,   139,     0,   420,
       0,   131,   156,     0,     0,     0,    81,     0,     0,     0,
       0,     0,   133,   134,   135,   136,   137,   138,   139,    81,
       0,     0,     0,     0,     0,   133,   134,   135,   136,   137,
     138,   139
};

static const yytype_int16 yycheck[] =
{
       1,     2,    87,   260,    88,    66,    78,   419,    78,   524,
     183,    78,    54,    55,    69,   289,   531,    92,   561,    80,
     551,   471,    37,   534,    17,    17,    58,    69,    17,    73,
      72,    30,   236,    90,    91,   546,   425,     0,   549,   172,
     173,   174,   365,   535,   494,   588,   179,   493,    38,    62,
     651,    83,    51,    66,   443,   570,   379,   532,   102,   660,
     132,    38,   132,   337,    65,   132,    67,    68,   153,    75,
     520,    86,   276,   148,   116,   525,   507,   238,   239,   240,
     511,   527,   528,   155,   615,    78,   561,   618,   580,    78,
     605,   537,   693,    83,    86,   606,    88,   589,    99,   183,
     611,   162,   176,   164,   165,   149,   167,    38,    85,    83,
      76,   626,    86,   588,   247,   189,   249,   648,    76,   293,
      86,   552,   296,   554,    76,    83,   557,   573,    80,    79,
     215,    81,   217,   625,   649,   581,    76,    42,    88,   585,
     571,   652,    75,    83,    77,   656,   638,   320,   659,    78,
     311,   312,    83,    76,    85,   329,   231,    80,   608,   690,
      83,    80,   323,   324,   325,   326,   327,    86,     3,     4,
     665,   702,    55,   688,   175,   686,    77,   434,   220,    94,
      95,    96,    83,   678,   679,    86,   632,   682,   719,    75,
     636,    77,    81,   639,    83,    40,    41,    86,   283,    77,
      89,    90,   713,    76,   635,   716,   291,    81,    37,    83,
     622,   212,    86,   724,    86,    89,    90,    81,   260,    83,
      86,   667,    86,    86,   285,    89,    90,   302,    11,    12,
      13,    14,    15,    16,    57,   292,   320,   294,   295,    76,
     297,   242,   317,    80,    80,   246,    83,   332,     1,   334,
     335,    75,    83,    84,   255,    86,   257,   703,    89,    90,
     261,   262,   263,   264,   265,   266,   712,    61,    62,     1,
     185,   186,   187,   188,    76,    83,   722,    82,    86,    87,
     364,    89,    90,   367,   345,   286,   328,    55,    56,    57,
      43,    44,    45,    46,    47,    48,    49,    82,   382,    61,
      62,    63,    64,    82,    78,    58,   563,    53,   565,    55,
      68,    43,    44,    45,    46,    47,    48,    49,     6,    79,
      75,    74,    77,   398,    77,    78,    58,    55,    56,    57,
      83,    75,    86,    77,    86,   419,    78,   422,    38,    78,
      85,   425,    74,    55,    58,    77,    78,    84,    82,    37,
      38,    83,    82,   610,    85,   440,    80,    79,    87,   443,
     432,   403,   432,   620,    87,   432,    76,    55,    87,   426,
     427,   428,    79,    61,    62,    63,    64,    65,    66,    67,
      51,    69,    70,    87,   304,   305,   306,    79,    87,    79,
      77,    84,   434,    87,    79,    81,   438,    81,    78,    87,
      79,   658,    78,    86,     1,    87,    83,    88,   409,     6,
       7,     8,     9,    10,    88,    84,    84,    87,    81,   480,
     421,    18,    19,    84,    81,    22,    84,    79,    17,    83,
     515,   432,    83,   475,    83,    42,    78,    78,   439,   359,
     360,    79,    86,     1,    84,    42,    57,    84,    87,    84,
      84,   493,    84,    83,   374,   375,   376,   377,   378,   544,
      87,    87,    79,    87,    86,   550,    87,    37,    83,    55,
      86,   472,    77,    57,   559,   530,    77,   532,    20,    55,
      79,    78,     1,    77,   568,    43,    44,    45,    46,    47,
      48,    49,    88,    78,    86,   537,    75,   582,     1,   584,
      87,   502,   587,    84,    78,    77,   561,    78,    86,    77,
      56,    78,    55,    76,   569,    79,    74,    50,   603,    77,
      78,   563,    87,   565,    43,    44,    45,    46,    47,    48,
      49,   586,   617,   588,    75,    75,    37,     1,   622,   623,
      43,    44,    45,    46,    47,    48,    49,     1,    87,   634,
      78,    78,    76,    78,    55,    74,    75,    80,    77,   644,
      61,    62,    63,    64,    65,    66,    67,    78,   610,    75,
     655,    74,    80,    87,    77,    78,     6,    79,   620,    43,
      44,    45,    46,    47,    48,    49,   671,    78,    76,    43,
      44,    45,    46,    47,    48,    49,    78,    78,   599,    78,
      87,    81,    79,    78,    78,    78,   691,    37,    38,     1,
      74,    70,     5,    77,    55,   132,   658,   190,   156,   314,
      74,   177,    16,    77,   666,    55,   441,   300,   398,   320,
     715,    61,    62,    63,    64,    65,    66,    67,    70,    69,
      70,   642,   643,    37,    38,   309,   647,   319,   404,    70,
     401,    43,    44,    45,    46,    47,    48,    49,   492,   432,
     537,    55,   231,   623,   665,   556,   333,    61,    62,    63,
      64,    65,    66,    67,   585,   545,   525,   678,   679,   574,
     681,   682,    74,    -1,    -1,    77,    -1,     1,    -1,     3,
       4,     5,    -1,     7,     8,     9,   697,    11,    12,    13,
      14,    15,    -1,    -1,    -1,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,    29,    30,    -1,    32,    33,
      34,    35,    36,    37,    38,    39,    40,    41,    42,    43,
      44,    45,    46,    47,    48,    49,    50,    -1,    52,    -1,
      54,    55,    -1,    -1,    -1,    -1,    -1,    61,    62,    63,
      64,    65,    66,    67,    68,     1,    -1,    71,    72,    73,
      74,    -1,     1,    -1,    -1,    -1,    -1,    -1,    -1,     1,
      -1,     1,    86,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    14,    15,    16,    -1,    18,    19,
      -1,    -1,    22,    -1,    -1,    -1,    -1,    43,    44,    45,
      46,    47,    48,    49,    43,    44,    45,    46,    47,    48,
      49,    43,    44,    45,    46,    47,    48,    49,     1,    -1,
      -1,    -1,    -1,    53,    54,     1,    37,    38,    74,    59,
      60,    77,     1,    -1,    -1,    74,    -1,    -1,    77,     1,
      -1,    71,    74,    -1,    55,    77,    -1,    77,    78,    -1,
      61,    62,    63,    64,    65,    66,    67,    -1,    -1,    -1,
      43,    44,    45,    46,    47,    48,    49,    43,    44,    45,
      46,    47,    48,    49,    43,    44,    45,    46,    47,    48,
      49,    43,    44,    45,    46,    47,    48,    49,     1,    -1,
      -1,    74,    -1,    -1,    77,    -1,    -1,    -1,    74,    -1,
      -1,    77,    -1,    -1,    -1,    74,    -1,    -1,    77,    -1,
      -1,    -1,    74,    -1,    -1,    77,    -1,    -1,     6,     7,
       8,     9,    -1,    11,    12,    13,    14,    15,    -1,    -1,
      43,    44,    45,    46,    47,    48,    49,     3,     4,    -1,
       6,     7,     8,     9,    10,    11,    12,    13,    14,    15,
      16,     1,    18,    19,    42,    -1,    22,    -1,    -1,    -1,
      -1,    74,    -1,    -1,    77,    -1,    -1,    -1,    -1,    -1,
      -1,    21,    -1,    23,    24,    25,    26,    27,    28,    -1,
      -1,    31,    32,    33,    34,    35,    36,    75,    -1,    -1,
      78,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    51,    52,    -1,     1,    -1,    -1,    -1,    -1,    -1,
      -1,    77,    -1,    -1,    -1,    -1,    -1,    -1,    68,    -1,
      -1,    -1,    72,    73,    21,    -1,    23,    24,    25,    26,
      27,    28,    -1,    -1,    31,    32,    33,    34,    35,    36,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    51,    52,    -1,    -1,    -1,    -1,
      -1,     3,     4,     5,    -1,     7,     8,     9,    -1,    11,
      12,    13,    14,    15,    -1,    72,    73,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    -1,
      32,    33,    34,    35,    36,    37,    38,    39,    40,    41,
      42,    43,    44,    45,    46,    47,    48,    49,    50,    -1,
      52,    -1,    54,    55,    -1,    -1,    -1,    -1,    -1,    61,
      62,    63,    64,    65,    66,    67,    68,    -1,    -1,    71,
      72,    73,    74,     3,     4,     5,    -1,     7,     8,     9,
      -1,    11,    12,    13,    14,    15,    -1,    -1,    -1,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    -1,    32,    33,    34,    35,    36,    -1,    -1,    39,
      40,    41,    42,    43,    44,    45,    46,    47,    48,    49,
      50,    51,    52,    -1,    54,    55,    56,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    68,    -1,
      -1,    71,    72,    73,    74,     3,     4,     5,    -1,     7,
       8,     9,    -1,    11,    12,    13,    14,    15,    -1,    -1,
      -1,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      28,    29,    30,    -1,    32,    33,    34,    35,    36,    -1,
      -1,    39,    40,    41,    42,    43,    44,    45,    46,    47,
      48,    49,    50,    -1,    52,    -1,    54,    55,    56,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      68,    -1,    -1,    71,    72,    73,    74,     3,     4,     5,
      -1,     7,     8,     9,    -1,    11,    12,    13,    14,    15,
      -1,    -1,    -1,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    -1,    32,    33,    34,    35,
      36,    -1,    -1,    39,    40,    41,    42,    43,    44,    45,
      46,    47,    48,    49,    50,    -1,    52,    -1,    54,    55,
      56,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    68,    -1,    -1,    71,    72,    73,    74,     3,
       4,     5,    -1,     7,     8,     9,    -1,    11,    12,    13,
      14,    15,    -1,    -1,    -1,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,    29,    30,    -1,    32,    33,
      34,    35,    36,    -1,    18,    39,    40,    41,    42,    43,
      44,    45,    46,    47,    48,    49,    50,    -1,    52,    -1,
      54,    55,    -1,    37,    38,    43,    44,    45,    46,    47,
      48,    49,    -1,    -1,    68,    -1,    -1,    71,    72,    73,
      74,    55,    37,    38,    39,    -1,    -1,    61,    62,    63,
      64,    65,    66,    67,    -1,    -1,    74,    37,    38,    77,
      55,    -1,    -1,    -1,    -1,    -1,    61,    62,    63,    64,
      65,    66,    67,    -1,    -1,    55,    56,    57,    37,    38,
      39,    61,    62,    63,    64,    65,    66,    67,    -1,    84,
      -1,    37,    38,    -1,    -1,    -1,    55,    -1,    -1,    -1,
      -1,    -1,    61,    62,    63,    64,    65,    66,    67,    55,
      -1,    -1,    -1,    -1,    -1,    61,    62,    63,    64,    65,
      66,    67
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
      57,   104,   105,   106,   118,    85,    97,    78,    55,   176,
      58,   175,   176,    84,    95,    82,    82,    85,    86,   120,
     121,   204,    83,    80,    83,    87,    80,    83,   163,    87,
      79,   107,    76,   140,   140,   140,   140,    97,    87,    79,
      87,   109,   109,    87,    79,    77,    97,    88,   153,    97,
      79,    81,    96,    97,    97,    97,    97,    97,    97,    79,
      81,   107,    78,    79,    84,    87,   176,    98,    97,   121,
     119,   176,   124,   105,   124,   124,   105,   124,   129,   110,
     141,    77,    95,   158,   158,   158,   158,    87,   133,   140,
     140,   126,    17,    78,   135,   137,   138,    88,   152,    56,
      57,   110,   153,   155,   140,   140,   140,   140,   140,    77,
      95,   105,    83,   187,   176,   175,   176,   176,   121,    84,
      87,   205,    84,    81,    84,    98,    81,    84,    79,     1,
      42,   156,   159,   160,   165,   166,   168,   158,   158,   118,
     138,    78,   118,   158,   158,   158,   158,   158,   138,    39,
      84,   118,   177,   180,   185,    83,    83,    83,    83,   141,
       1,    86,   171,   168,    78,    95,   159,    97,    78,   118,
     177,    97,   176,    79,    84,   185,   124,   124,   124,     1,
      21,    23,    24,    25,    26,    27,    28,    31,    32,    33,
      34,    35,    36,    51,    52,    68,    72,    73,   172,   173,
      55,    97,   170,    96,    86,   136,    95,    97,   176,    86,
      88,   135,    87,   185,    84,    84,    84,    84,    57,   130,
      87,    87,    79,   187,    97,    87,    95,    87,    56,    57,
      98,   174,    37,   172,     1,    43,    44,    45,    46,    47,
      48,    49,    74,    77,    95,   178,   190,   195,   197,   187,
      97,    83,   201,    86,   201,    55,   202,   203,    77,    57,
     194,   201,    77,   191,   197,   176,    20,   189,   187,   176,
     199,    55,   199,   187,   204,    79,    77,   197,   192,   197,
     178,   199,    43,    44,    45,    47,    48,    49,    74,   178,
     193,   195,   196,    78,   191,   179,    88,   190,    86,   188,
      75,    87,    84,   200,   176,   203,    78,   191,    78,   191,
     176,   200,   201,    86,   201,    77,   194,   201,    77,   176,
      78,   193,    96,    96,    56,     6,    69,    70,    87,   118,
     181,   184,   186,   178,   199,   201,    77,   197,   205,    78,
     179,    77,   197,   176,    55,   176,   192,   178,   176,   193,
     179,    97,    76,    79,    87,   176,    75,   199,   191,   187,
      96,   191,    50,   198,    75,    87,   200,    78,   176,   200,
      78,    96,    80,   118,   177,   183,   186,   179,   199,    76,
      78,    78,    77,   197,   176,   201,    77,   197,   179,    77,
     197,    97,   182,    97,   176,    80,    97,   200,   199,   198,
     191,    75,   176,   191,    96,   191,   198,    81,    83,    86,
      89,    90,    80,    87,   182,    95,    77,   197,    79,    78,
     176,    76,    78,    78,   182,    56,   182,    81,    97,   182,
      81,   191,   199,   200,   176,   198,    84,    87,    87,    97,
      81,    78,   200,    77,   197,    79,    77,   197,   191,   176,
     191,    78,   200,    78,    77,   197,   191,    78
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
      98,    99,    99,   100,   100,   101,   101,   102,   102,   102,
     102,   102,   103,   103,   103,   103,   103,   103,   103,   103,
     103,   103,   103,   103,   103,   103,   104,   104,   104,   105,
     105,   106,   106,   107,   107,   108,   108,   108,   108,   108,
     108,   108,   108,   108,   108,   108,   108,   108,   108,   108,
     109,   110,   111,   111,   112,   113,   113,   114,   115,   115,
     115,   115,   115,   115,   116,   116,   116,   116,   116,   117,
     117,   117,   118,   118,   118,   119,   120,   121,   121,   122,
     123,   124,   124,   125,   125,   126,   126,   127,   127,   128,
     128,   129,   129,   130,   130,   131,   132,   132,   133,   133,
     134,   134,   135,   135,   136,   136,   137,   138,   138,   139,
     139,   139,   140,   140,   141,   141,   142,   142,   143,   144,
     145,   145,   146,   146,   147,   147,   148,   149,   150,   151,
     151,   152,   152,   153,   153,   153,   153,   154,   154,   154,
     155,   155,   156,   157,   157,   157,   157,   157,   158,   158,
     159,   159,   160,   160,   160,   160,   160,   160,   160,   161,
     161,   161,   161,   161,   162,   162,   162,   162,   163,   163,
     164,   165,   166,   166,   166,   166,   167,   167,   167,   167,
     167,   167,   167,   167,   167,   167,   167,   168,   168,   168,
     169,   169,   170,   171,   171,   171,   172,   172,   173,   173,
     173,   173,   173,   173,   173,   173,   173,   173,   173,   173,
     173,   173,   173,   173,   173,   173,   174,   174,   174,   175,
     175,   175,   176,   176,   176,   176,   176,   176,   177,   178,
     179,   180,   180,   180,   180,   180,   181,   181,   181,   182,
     182,   182,   182,   182,   182,   183,   184,   184,   184,   185,
     185,   186,   186,   187,   187,   188,   188,   189,   189,   190,
     190,   190,   191,   191,   192,   192,   193,   193,   193,   194,
     194,   195,   195,   195,   196,   196,   196,   196,   196,   196,
     196,   196,   196,   196,   196,   196,   197,   197,   197,   197,
     197,   197,   197,   197,   197,   197,   197,   197,   197,   197,
     198,   198,   198,   199,   200,   201,   202,   202,   203,   203,
     204,   205,   206,   207
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
       4,     3,     3,     1,     4,     0,     2,     3,     2,     2,
       2,     7,     5,     5,     2,     2,     2,     2,     2,     2,
       2,     2,     1,     1,     1,     1,     1,     1,     1,     1,
       3,     0,     1,     0,     3,     1,     1,     1,     1,     2,
       2,     3,     3,     2,     2,     2,     1,     1,     2,     1,
       2,     2,     1,     1,     2,     2,     2,     8,     1,     1,
       1,     1,     2,     2,     1,     1,     1,     2,     2,     3,
       2,     1,     3,     2,     1,     1,     3,     0,     2,     4,
       6,     0,     1,     0,     3,     1,     3,     1,     1,     0,
       3,     1,     3,     0,     1,     1,     0,     3,     1,     3,
       1,     1,     0,     1,     0,     2,     5,     1,     2,     3,
       5,     6,     0,     2,     1,     3,     5,     5,     5,     5,
       4,     3,     6,     6,     5,     5,     5,     5,     5,     4,
       7,     0,     2,     0,     2,     2,     2,     3,     2,     3,
       1,     3,     4,     2,     2,     2,     2,     2,     1,     4,
       0,     2,     1,     1,     1,     1,     2,     2,     2,     3,
       6,     9,     3,     6,     3,     6,     9,     9,     1,     3,
       1,     1,     1,     2,     2,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     7,     5,    13,
       5,     2,     1,     0,     3,     1,     1,     3,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     2,     1,     1,     1,     1,     1,     1,     1,     0,
       1,     3,     0,     1,     5,     5,     5,     4,     3,     1,
       1,     1,     3,     4,     3,     4,     1,     1,     1,     1,
       4,     3,     4,     4,     4,     3,     7,     5,     6,     1,
       3,     1,     3,     3,     2,     3,     2,     0,     3,     1,
       1,     4,     1,     2,     1,     2,     1,     2,     1,     1,
       0,     4,     3,     5,     6,     4,     4,    11,     9,    12,
      14,     6,     8,     5,     7,     4,     6,     4,     1,     4,
      11,     9,    12,    14,     6,     8,     5,     7,     4,     1,
       0,     2,     4,     1,     1,     1,     2,     5,     1,     3,
       1,     1,     2,     2
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
#line 2240 "y.tab.c" /* yacc.c:1661  */
    break;

  case 3:
#line 200 "xi-grammar.y" /* yacc.c:1661  */
    {
		  (yyval.modlist) = 0;
		}
#line 2248 "y.tab.c" /* yacc.c:1661  */
    break;

  case 4:
#line 204 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.modlist) = new AstChildren<Module>(lineno, (yyvsp[-1].module), (yyvsp[0].modlist)); }
#line 2254 "y.tab.c" /* yacc.c:1661  */
    break;

  case 5:
#line 208 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 0; }
#line 2260 "y.tab.c" /* yacc.c:1661  */
    break;

  case 6:
#line 210 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 1; }
#line 2266 "y.tab.c" /* yacc.c:1661  */
    break;

  case 7:
#line 214 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 1; }
#line 2272 "y.tab.c" /* yacc.c:1661  */
    break;

  case 8:
#line 216 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 2; }
#line 2278 "y.tab.c" /* yacc.c:1661  */
    break;

  case 9:
#line 220 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 0; }
#line 2284 "y.tab.c" /* yacc.c:1661  */
    break;

  case 10:
#line 222 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 1; }
#line 2290 "y.tab.c" /* yacc.c:1661  */
    break;

  case 11:
#line 227 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 2296 "y.tab.c" /* yacc.c:1661  */
    break;

  case 12:
#line 228 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(MODULE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2302 "y.tab.c" /* yacc.c:1661  */
    break;

  case 13:
#line 229 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(MAINMODULE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2308 "y.tab.c" /* yacc.c:1661  */
    break;

  case 14:
#line 230 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(EXTERN, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2314 "y.tab.c" /* yacc.c:1661  */
    break;

  case 15:
#line 232 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(INITCALL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2320 "y.tab.c" /* yacc.c:1661  */
    break;

  case 16:
#line 233 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(INITNODE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2326 "y.tab.c" /* yacc.c:1661  */
    break;

  case 17:
#line 234 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(INITPROC, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2332 "y.tab.c" /* yacc.c:1661  */
    break;

  case 18:
#line 236 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(CHARE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2338 "y.tab.c" /* yacc.c:1661  */
    break;

  case 19:
#line 237 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(MAINCHARE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2344 "y.tab.c" /* yacc.c:1661  */
    break;

  case 20:
#line 238 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(GROUP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2350 "y.tab.c" /* yacc.c:1661  */
    break;

  case 21:
#line 239 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(NODEGROUP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2356 "y.tab.c" /* yacc.c:1661  */
    break;

  case 22:
#line 240 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(ARRAY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2362 "y.tab.c" /* yacc.c:1661  */
    break;

  case 23:
#line 244 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(INCLUDE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2368 "y.tab.c" /* yacc.c:1661  */
    break;

  case 24:
#line 245 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(STACKSIZE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2374 "y.tab.c" /* yacc.c:1661  */
    break;

  case 25:
#line 246 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(THREADED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2380 "y.tab.c" /* yacc.c:1661  */
    break;

  case 26:
#line 247 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(TEMPLATE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2386 "y.tab.c" /* yacc.c:1661  */
    break;

  case 27:
#line 248 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(SYNC, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2392 "y.tab.c" /* yacc.c:1661  */
    break;

  case 28:
#line 249 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(IGET, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2398 "y.tab.c" /* yacc.c:1661  */
    break;

  case 29:
#line 250 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(EXCLUSIVE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2404 "y.tab.c" /* yacc.c:1661  */
    break;

  case 30:
#line 251 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(IMMEDIATE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2410 "y.tab.c" /* yacc.c:1661  */
    break;

  case 31:
#line 252 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(SKIPSCHED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2416 "y.tab.c" /* yacc.c:1661  */
    break;

  case 32:
#line 253 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(NOCOPY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2422 "y.tab.c" /* yacc.c:1661  */
    break;

  case 33:
#line 254 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(INLINE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2428 "y.tab.c" /* yacc.c:1661  */
    break;

  case 34:
#line 255 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(VIRTUAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2434 "y.tab.c" /* yacc.c:1661  */
    break;

  case 35:
#line 256 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(MIGRATABLE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2440 "y.tab.c" /* yacc.c:1661  */
    break;

  case 36:
#line 257 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(CREATEHERE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2446 "y.tab.c" /* yacc.c:1661  */
    break;

  case 37:
#line 258 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(CREATEHOME, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2452 "y.tab.c" /* yacc.c:1661  */
    break;

  case 38:
#line 259 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(NOKEEP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2458 "y.tab.c" /* yacc.c:1661  */
    break;

  case 39:
#line 260 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(NOTRACE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2464 "y.tab.c" /* yacc.c:1661  */
    break;

  case 40:
#line 261 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(APPWORK, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2470 "y.tab.c" /* yacc.c:1661  */
    break;

  case 41:
#line 264 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(PACKED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2476 "y.tab.c" /* yacc.c:1661  */
    break;

  case 42:
#line 265 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(VARSIZE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2482 "y.tab.c" /* yacc.c:1661  */
    break;

  case 43:
#line 266 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(ENTRY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2488 "y.tab.c" /* yacc.c:1661  */
    break;

  case 44:
#line 267 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(FOR, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2494 "y.tab.c" /* yacc.c:1661  */
    break;

  case 45:
#line 268 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(FORALL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2500 "y.tab.c" /* yacc.c:1661  */
    break;

  case 46:
#line 269 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(WHILE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2506 "y.tab.c" /* yacc.c:1661  */
    break;

  case 47:
#line 270 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(WHEN, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2512 "y.tab.c" /* yacc.c:1661  */
    break;

  case 48:
#line 271 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(OVERLAP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2518 "y.tab.c" /* yacc.c:1661  */
    break;

  case 49:
#line 272 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(SERIAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2524 "y.tab.c" /* yacc.c:1661  */
    break;

  case 50:
#line 273 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(IF, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2530 "y.tab.c" /* yacc.c:1661  */
    break;

  case 51:
#line 274 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(ELSE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2536 "y.tab.c" /* yacc.c:1661  */
    break;

  case 52:
#line 276 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(LOCAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2542 "y.tab.c" /* yacc.c:1661  */
    break;

  case 53:
#line 278 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(USING, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2548 "y.tab.c" /* yacc.c:1661  */
    break;

  case 54:
#line 279 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(ACCEL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2554 "y.tab.c" /* yacc.c:1661  */
    break;

  case 55:
#line 282 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(ACCELBLOCK, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2560 "y.tab.c" /* yacc.c:1661  */
    break;

  case 56:
#line 283 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(MEMCRITICAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2566 "y.tab.c" /* yacc.c:1661  */
    break;

  case 57:
#line 284 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(REDUCTIONTARGET, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2572 "y.tab.c" /* yacc.c:1661  */
    break;

  case 58:
#line 285 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(CASE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2578 "y.tab.c" /* yacc.c:1661  */
    break;

  case 59:
#line 290 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 2584 "y.tab.c" /* yacc.c:1661  */
    break;

  case 60:
#line 292 "xi-grammar.y" /* yacc.c:1661  */
    {
		  char *tmp = new char[strlen((yyvsp[-3].strval))+strlen((yyvsp[0].strval))+3];
		  sprintf(tmp,"%s::%s", (yyvsp[-3].strval), (yyvsp[0].strval));
		  (yyval.strval) = tmp;
		}
#line 2594 "y.tab.c" /* yacc.c:1661  */
    break;

  case 61:
#line 299 "xi-grammar.y" /* yacc.c:1661  */
    {
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist));
		}
#line 2602 "y.tab.c" /* yacc.c:1661  */
    break;

  case 62:
#line 303 "xi-grammar.y" /* yacc.c:1661  */
    {
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist));
		    (yyval.module)->setMain();
		}
#line 2611 "y.tab.c" /* yacc.c:1661  */
    break;

  case 63:
#line 310 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.conslist) = 0; }
#line 2617 "y.tab.c" /* yacc.c:1661  */
    break;

  case 64:
#line 312 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.conslist) = (yyvsp[-2].conslist); }
#line 2623 "y.tab.c" /* yacc.c:1661  */
    break;

  case 65:
#line 316 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.conslist) = 0; }
#line 2629 "y.tab.c" /* yacc.c:1661  */
    break;

  case 66:
#line 318 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.conslist) = new ConstructList(lineno, (yyvsp[-1].construct), (yyvsp[0].conslist)); }
#line 2635 "y.tab.c" /* yacc.c:1661  */
    break;

  case 67:
#line 322 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.construct) = new UsingScope((yyvsp[0].strval), false); }
#line 2641 "y.tab.c" /* yacc.c:1661  */
    break;

  case 68:
#line 324 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.construct) = new UsingScope((yyvsp[0].strval), true); }
#line 2647 "y.tab.c" /* yacc.c:1661  */
    break;

  case 69:
#line 326 "xi-grammar.y" /* yacc.c:1661  */
    { (yyvsp[0].member)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].member); }
#line 2653 "y.tab.c" /* yacc.c:1661  */
    break;

  case 70:
#line 328 "xi-grammar.y" /* yacc.c:1661  */
    { (yyvsp[0].message)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].message); }
#line 2659 "y.tab.c" /* yacc.c:1661  */
    break;

  case 71:
#line 330 "xi-grammar.y" /* yacc.c:1661  */
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
#line 2674 "y.tab.c" /* yacc.c:1661  */
    break;

  case 72:
#line 343 "xi-grammar.y" /* yacc.c:1661  */
    { if((yyvsp[-2].conslist)) (yyvsp[-2].conslist)->recurse<int&>((yyvsp[-4].intval), &Construct::setExtern); (yyval.construct) = (yyvsp[-2].conslist); }
#line 2680 "y.tab.c" /* yacc.c:1661  */
    break;

  case 73:
#line 345 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.construct) = new Scope((yyvsp[-3].strval), (yyvsp[-1].conslist)); }
#line 2686 "y.tab.c" /* yacc.c:1661  */
    break;

  case 74:
#line 347 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.construct) = (yyvsp[-1].construct); }
#line 2692 "y.tab.c" /* yacc.c:1661  */
    break;

  case 75:
#line 349 "xi-grammar.y" /* yacc.c:1661  */
    {
          ERROR("preceding construct must be semicolon terminated",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 2702 "y.tab.c" /* yacc.c:1661  */
    break;

  case 76:
#line 355 "xi-grammar.y" /* yacc.c:1661  */
    { (yyvsp[0].module)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].module); }
#line 2708 "y.tab.c" /* yacc.c:1661  */
    break;

  case 77:
#line 357 "xi-grammar.y" /* yacc.c:1661  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2714 "y.tab.c" /* yacc.c:1661  */
    break;

  case 78:
#line 359 "xi-grammar.y" /* yacc.c:1661  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2720 "y.tab.c" /* yacc.c:1661  */
    break;

  case 79:
#line 361 "xi-grammar.y" /* yacc.c:1661  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2726 "y.tab.c" /* yacc.c:1661  */
    break;

  case 80:
#line 363 "xi-grammar.y" /* yacc.c:1661  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2732 "y.tab.c" /* yacc.c:1661  */
    break;

  case 81:
#line 365 "xi-grammar.y" /* yacc.c:1661  */
    { (yyvsp[0].templat)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].templat); }
#line 2738 "y.tab.c" /* yacc.c:1661  */
    break;

  case 82:
#line 367 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.construct) = NULL; }
#line 2744 "y.tab.c" /* yacc.c:1661  */
    break;

  case 83:
#line 369 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.construct) = NULL; }
#line 2750 "y.tab.c" /* yacc.c:1661  */
    break;

  case 84:
#line 371 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.construct) = (yyvsp[0].accelBlock); }
#line 2756 "y.tab.c" /* yacc.c:1661  */
    break;

  case 85:
#line 373 "xi-grammar.y" /* yacc.c:1661  */
    {
          ERROR("invalid construct",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 2766 "y.tab.c" /* yacc.c:1661  */
    break;

  case 86:
#line 381 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tparam) = new TParamType((yyvsp[0].type)); }
#line 2772 "y.tab.c" /* yacc.c:1661  */
    break;

  case 87:
#line 383 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
#line 2778 "y.tab.c" /* yacc.c:1661  */
    break;

  case 88:
#line 385 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
#line 2784 "y.tab.c" /* yacc.c:1661  */
    break;

  case 89:
#line 389 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tparlist) = new TParamList((yyvsp[0].tparam)); }
#line 2790 "y.tab.c" /* yacc.c:1661  */
    break;

  case 90:
#line 391 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tparlist) = new TParamList((yyvsp[-2].tparam), (yyvsp[0].tparlist)); }
#line 2796 "y.tab.c" /* yacc.c:1661  */
    break;

  case 91:
#line 395 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tparlist) = new TParamList(0); }
#line 2802 "y.tab.c" /* yacc.c:1661  */
    break;

  case 92:
#line 397 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tparlist) = (yyvsp[0].tparlist); }
#line 2808 "y.tab.c" /* yacc.c:1661  */
    break;

  case 93:
#line 401 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tparlist) = 0; }
#line 2814 "y.tab.c" /* yacc.c:1661  */
    break;

  case 94:
#line 403 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tparlist) = (yyvsp[-1].tparlist); }
#line 2820 "y.tab.c" /* yacc.c:1661  */
    break;

  case 95:
#line 407 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("int"); }
#line 2826 "y.tab.c" /* yacc.c:1661  */
    break;

  case 96:
#line 409 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("long"); }
#line 2832 "y.tab.c" /* yacc.c:1661  */
    break;

  case 97:
#line 411 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("short"); }
#line 2838 "y.tab.c" /* yacc.c:1661  */
    break;

  case 98:
#line 413 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("char"); }
#line 2844 "y.tab.c" /* yacc.c:1661  */
    break;

  case 99:
#line 415 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("unsigned int"); }
#line 2850 "y.tab.c" /* yacc.c:1661  */
    break;

  case 100:
#line 417 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("unsigned long"); }
#line 2856 "y.tab.c" /* yacc.c:1661  */
    break;

  case 101:
#line 419 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("unsigned long"); }
#line 2862 "y.tab.c" /* yacc.c:1661  */
    break;

  case 102:
#line 421 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("unsigned long long"); }
#line 2868 "y.tab.c" /* yacc.c:1661  */
    break;

  case 103:
#line 423 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("unsigned short"); }
#line 2874 "y.tab.c" /* yacc.c:1661  */
    break;

  case 104:
#line 425 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("unsigned char"); }
#line 2880 "y.tab.c" /* yacc.c:1661  */
    break;

  case 105:
#line 427 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("long long"); }
#line 2886 "y.tab.c" /* yacc.c:1661  */
    break;

  case 106:
#line 429 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("float"); }
#line 2892 "y.tab.c" /* yacc.c:1661  */
    break;

  case 107:
#line 431 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("double"); }
#line 2898 "y.tab.c" /* yacc.c:1661  */
    break;

  case 108:
#line 433 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("long double"); }
#line 2904 "y.tab.c" /* yacc.c:1661  */
    break;

  case 109:
#line 435 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("void"); }
#line 2910 "y.tab.c" /* yacc.c:1661  */
    break;

  case 110:
#line 438 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.ntype) = new NamedType((yyvsp[-1].strval),(yyvsp[0].tparlist)); }
#line 2916 "y.tab.c" /* yacc.c:1661  */
    break;

  case 111:
#line 439 "xi-grammar.y" /* yacc.c:1661  */
    {
                    const char* basename, *scope;
                    splitScopedName((yyvsp[-1].strval), &scope, &basename);
                    (yyval.ntype) = new NamedType(basename, (yyvsp[0].tparlist), scope);
                }
#line 2926 "y.tab.c" /* yacc.c:1661  */
    break;

  case 112:
#line 447 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].type); }
#line 2932 "y.tab.c" /* yacc.c:1661  */
    break;

  case 113:
#line 449 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].ntype); }
#line 2938 "y.tab.c" /* yacc.c:1661  */
    break;

  case 114:
#line 453 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.ptype) = new PtrType((yyvsp[-1].type)); }
#line 2944 "y.tab.c" /* yacc.c:1661  */
    break;

  case 115:
#line 457 "xi-grammar.y" /* yacc.c:1661  */
    { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
#line 2950 "y.tab.c" /* yacc.c:1661  */
    break;

  case 116:
#line 459 "xi-grammar.y" /* yacc.c:1661  */
    { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
#line 2956 "y.tab.c" /* yacc.c:1661  */
    break;

  case 117:
#line 463 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.ftype) = new FuncType((yyvsp[-7].type), (yyvsp[-4].strval), (yyvsp[-1].plist)); }
#line 2962 "y.tab.c" /* yacc.c:1661  */
    break;

  case 118:
#line 467 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].type); }
#line 2968 "y.tab.c" /* yacc.c:1661  */
    break;

  case 119:
#line 469 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 2974 "y.tab.c" /* yacc.c:1661  */
    break;

  case 120:
#line 471 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 2980 "y.tab.c" /* yacc.c:1661  */
    break;

  case 121:
#line 473 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].ftype); }
#line 2986 "y.tab.c" /* yacc.c:1661  */
    break;

  case 122:
#line 475 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new ConstType((yyvsp[0].type)); }
#line 2992 "y.tab.c" /* yacc.c:1661  */
    break;

  case 123:
#line 477 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new ConstType((yyvsp[-1].type)); }
#line 2998 "y.tab.c" /* yacc.c:1661  */
    break;

  case 124:
#line 481 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3004 "y.tab.c" /* yacc.c:1661  */
    break;

  case 125:
#line 483 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3010 "y.tab.c" /* yacc.c:1661  */
    break;

  case 126:
#line 485 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3016 "y.tab.c" /* yacc.c:1661  */
    break;

  case 127:
#line 487 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new ConstType((yyvsp[0].type)); }
#line 3022 "y.tab.c" /* yacc.c:1661  */
    break;

  case 128:
#line 489 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new ConstType((yyvsp[-1].type)); }
#line 3028 "y.tab.c" /* yacc.c:1661  */
    break;

  case 129:
#line 493 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new RValueReferenceType((yyvsp[-2].type)); }
#line 3034 "y.tab.c" /* yacc.c:1661  */
    break;

  case 130:
#line 495 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new ReferenceType((yyvsp[-1].type)); }
#line 3040 "y.tab.c" /* yacc.c:1661  */
    break;

  case 131:
#line 497 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3046 "y.tab.c" /* yacc.c:1661  */
    break;

  case 132:
#line 501 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new RValueReferenceType((yyvsp[-2].type)); }
#line 3052 "y.tab.c" /* yacc.c:1661  */
    break;

  case 133:
#line 503 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new ReferenceType((yyvsp[-1].type)); }
#line 3058 "y.tab.c" /* yacc.c:1661  */
    break;

  case 134:
#line 505 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3064 "y.tab.c" /* yacc.c:1661  */
    break;

  case 135:
#line 509 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 3070 "y.tab.c" /* yacc.c:1661  */
    break;

  case 136:
#line 513 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.val) = (yyvsp[-1].val); }
#line 3076 "y.tab.c" /* yacc.c:1661  */
    break;

  case 137:
#line 517 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.vallist) = 0; }
#line 3082 "y.tab.c" /* yacc.c:1661  */
    break;

  case 138:
#line 519 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.vallist) = new ValueList((yyvsp[-1].val), (yyvsp[0].vallist)); }
#line 3088 "y.tab.c" /* yacc.c:1661  */
    break;

  case 139:
#line 523 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].vallist)); }
#line 3094 "y.tab.c" /* yacc.c:1661  */
    break;

  case 140:
#line 527 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[-3].type), (yyvsp[-1].strval), (yyvsp[0].vallist), 1); }
#line 3100 "y.tab.c" /* yacc.c:1661  */
    break;

  case 141:
#line 531 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 0;}
#line 3106 "y.tab.c" /* yacc.c:1661  */
    break;

  case 142:
#line 533 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 0;}
#line 3112 "y.tab.c" /* yacc.c:1661  */
    break;

  case 143:
#line 537 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 0; }
#line 3118 "y.tab.c" /* yacc.c:1661  */
    break;

  case 144:
#line 539 "xi-grammar.y" /* yacc.c:1661  */
    {
		  /*
		  printf("Warning: Message attributes are being phased out.\n");
		  printf("Warning: Please remove them from interface files.\n");
		  */
		  (yyval.intval) = (yyvsp[-1].intval);
		}
#line 3130 "y.tab.c" /* yacc.c:1661  */
    break;

  case 145:
#line 549 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = (yyvsp[0].intval); }
#line 3136 "y.tab.c" /* yacc.c:1661  */
    break;

  case 146:
#line 551 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = (yyvsp[-2].intval) | (yyvsp[0].intval); }
#line 3142 "y.tab.c" /* yacc.c:1661  */
    break;

  case 147:
#line 555 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 0; }
#line 3148 "y.tab.c" /* yacc.c:1661  */
    break;

  case 148:
#line 557 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 0; }
#line 3154 "y.tab.c" /* yacc.c:1661  */
    break;

  case 149:
#line 561 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = 0; }
#line 3160 "y.tab.c" /* yacc.c:1661  */
    break;

  case 150:
#line 563 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = (yyvsp[-1].cattr); }
#line 3166 "y.tab.c" /* yacc.c:1661  */
    break;

  case 151:
#line 567 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = (yyvsp[0].cattr); }
#line 3172 "y.tab.c" /* yacc.c:1661  */
    break;

  case 152:
#line 569 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
#line 3178 "y.tab.c" /* yacc.c:1661  */
    break;

  case 153:
#line 573 "xi-grammar.y" /* yacc.c:1661  */
    { python_doc = NULL; (yyval.intval) = 0; }
#line 3184 "y.tab.c" /* yacc.c:1661  */
    break;

  case 154:
#line 575 "xi-grammar.y" /* yacc.c:1661  */
    { python_doc = (yyvsp[0].strval); (yyval.intval) = 0; }
#line 3190 "y.tab.c" /* yacc.c:1661  */
    break;

  case 155:
#line 579 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = Chare::CPYTHON; }
#line 3196 "y.tab.c" /* yacc.c:1661  */
    break;

  case 156:
#line 583 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = 0; }
#line 3202 "y.tab.c" /* yacc.c:1661  */
    break;

  case 157:
#line 585 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = (yyvsp[-1].cattr); }
#line 3208 "y.tab.c" /* yacc.c:1661  */
    break;

  case 158:
#line 589 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = (yyvsp[0].cattr); }
#line 3214 "y.tab.c" /* yacc.c:1661  */
    break;

  case 159:
#line 591 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
#line 3220 "y.tab.c" /* yacc.c:1661  */
    break;

  case 160:
#line 595 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = Chare::CMIGRATABLE; }
#line 3226 "y.tab.c" /* yacc.c:1661  */
    break;

  case 161:
#line 597 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = Chare::CPYTHON; }
#line 3232 "y.tab.c" /* yacc.c:1661  */
    break;

  case 162:
#line 601 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 0; }
#line 3238 "y.tab.c" /* yacc.c:1661  */
    break;

  case 163:
#line 603 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 1; }
#line 3244 "y.tab.c" /* yacc.c:1661  */
    break;

  case 164:
#line 606 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 0; }
#line 3250 "y.tab.c" /* yacc.c:1661  */
    break;

  case 165:
#line 608 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 1; }
#line 3256 "y.tab.c" /* yacc.c:1661  */
    break;

  case 166:
#line 611 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.mv) = new MsgVar((yyvsp[-3].type), (yyvsp[-2].strval), (yyvsp[-4].intval), (yyvsp[-1].intval)); }
#line 3262 "y.tab.c" /* yacc.c:1661  */
    break;

  case 167:
#line 615 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.mvlist) = new MsgVarList((yyvsp[0].mv)); }
#line 3268 "y.tab.c" /* yacc.c:1661  */
    break;

  case 168:
#line 617 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.mvlist) = new MsgVarList((yyvsp[-1].mv), (yyvsp[0].mvlist)); }
#line 3274 "y.tab.c" /* yacc.c:1661  */
    break;

  case 169:
#line 621 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.message) = new Message(lineno, (yyvsp[0].ntype)); }
#line 3280 "y.tab.c" /* yacc.c:1661  */
    break;

  case 170:
#line 623 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.message) = new Message(lineno, (yyvsp[-2].ntype)); }
#line 3286 "y.tab.c" /* yacc.c:1661  */
    break;

  case 171:
#line 625 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.message) = new Message(lineno, (yyvsp[-3].ntype), (yyvsp[-1].mvlist)); }
#line 3292 "y.tab.c" /* yacc.c:1661  */
    break;

  case 172:
#line 629 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.typelist) = 0; }
#line 3298 "y.tab.c" /* yacc.c:1661  */
    break;

  case 173:
#line 631 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.typelist) = (yyvsp[0].typelist); }
#line 3304 "y.tab.c" /* yacc.c:1661  */
    break;

  case 174:
#line 635 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.typelist) = new TypeList((yyvsp[0].ntype)); }
#line 3310 "y.tab.c" /* yacc.c:1661  */
    break;

  case 175:
#line 637 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.typelist) = new TypeList((yyvsp[-2].ntype), (yyvsp[0].typelist)); }
#line 3316 "y.tab.c" /* yacc.c:1661  */
    break;

  case 176:
#line 641 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr)|Chare::CCHARE, (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3322 "y.tab.c" /* yacc.c:1661  */
    break;

  case 177:
#line 643 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3328 "y.tab.c" /* yacc.c:1661  */
    break;

  case 178:
#line 647 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3334 "y.tab.c" /* yacc.c:1661  */
    break;

  case 179:
#line 651 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.chare) = new NodeGroup(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3340 "y.tab.c" /* yacc.c:1661  */
    break;

  case 180:
#line 655 "xi-grammar.y" /* yacc.c:1661  */
    {/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",(yyvsp[-2].strval));
			(yyval.ntype) = new NamedType(buf);
		}
#line 3350 "y.tab.c" /* yacc.c:1661  */
    break;

  case 181:
#line 661 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.ntype) = new NamedType((yyvsp[-1].strval)); }
#line 3356 "y.tab.c" /* yacc.c:1661  */
    break;

  case 182:
#line 665 "xi-grammar.y" /* yacc.c:1661  */
    {  (yyval.chare) = new Array(lineno, (yyvsp[-4].cattr), (yyvsp[-3].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3362 "y.tab.c" /* yacc.c:1661  */
    break;

  case 183:
#line 667 "xi-grammar.y" /* yacc.c:1661  */
    {  (yyval.chare) = new Array(lineno, (yyvsp[-3].cattr), (yyvsp[-4].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3368 "y.tab.c" /* yacc.c:1661  */
    break;

  case 184:
#line 671 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr)|Chare::CCHARE, new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist));}
#line 3374 "y.tab.c" /* yacc.c:1661  */
    break;

  case 185:
#line 673 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3380 "y.tab.c" /* yacc.c:1661  */
    break;

  case 186:
#line 677 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3386 "y.tab.c" /* yacc.c:1661  */
    break;

  case 187:
#line 681 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.chare) = new NodeGroup( lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3392 "y.tab.c" /* yacc.c:1661  */
    break;

  case 188:
#line 685 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.chare) = new Array( lineno, 0, (yyvsp[-3].ntype), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3398 "y.tab.c" /* yacc.c:1661  */
    break;

  case 189:
#line 689 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-1].strval))); }
#line 3404 "y.tab.c" /* yacc.c:1661  */
    break;

  case 190:
#line 691 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-4].strval)), (yyvsp[-2].mvlist)); }
#line 3410 "y.tab.c" /* yacc.c:1661  */
    break;

  case 191:
#line 695 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = 0; }
#line 3416 "y.tab.c" /* yacc.c:1661  */
    break;

  case 192:
#line 697 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3422 "y.tab.c" /* yacc.c:1661  */
    break;

  case 193:
#line 701 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = 0; }
#line 3428 "y.tab.c" /* yacc.c:1661  */
    break;

  case 194:
#line 703 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3434 "y.tab.c" /* yacc.c:1661  */
    break;

  case 195:
#line 705 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3440 "y.tab.c" /* yacc.c:1661  */
    break;

  case 196:
#line 707 "xi-grammar.y" /* yacc.c:1661  */
    {
		  XStr typeStr;
		  (yyvsp[0].ntype)->print(typeStr);
		  char *tmp = strdup(typeStr.get_string());
		  (yyval.strval) = tmp;
		}
#line 3451 "y.tab.c" /* yacc.c:1661  */
    break;

  case 197:
#line 716 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tvar) = new TType(new NamedType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3457 "y.tab.c" /* yacc.c:1661  */
    break;

  case 198:
#line 718 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tvar) = new TFunc((yyvsp[-1].ftype), (yyvsp[0].strval)); }
#line 3463 "y.tab.c" /* yacc.c:1661  */
    break;

  case 199:
#line 720 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tvar) = new TName((yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].strval)); }
#line 3469 "y.tab.c" /* yacc.c:1661  */
    break;

  case 200:
#line 724 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tvarlist) = new TVarList((yyvsp[0].tvar)); }
#line 3475 "y.tab.c" /* yacc.c:1661  */
    break;

  case 201:
#line 726 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tvarlist) = new TVarList((yyvsp[-2].tvar), (yyvsp[0].tvarlist)); }
#line 3481 "y.tab.c" /* yacc.c:1661  */
    break;

  case 202:
#line 730 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tvarlist) = (yyvsp[-1].tvarlist); }
#line 3487 "y.tab.c" /* yacc.c:1661  */
    break;

  case 203:
#line 734 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3493 "y.tab.c" /* yacc.c:1661  */
    break;

  case 204:
#line 736 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3499 "y.tab.c" /* yacc.c:1661  */
    break;

  case 205:
#line 738 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3505 "y.tab.c" /* yacc.c:1661  */
    break;

  case 206:
#line 740 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3511 "y.tab.c" /* yacc.c:1661  */
    break;

  case 207:
#line 742 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].message)); (yyvsp[0].message)->setTemplate((yyval.templat)); }
#line 3517 "y.tab.c" /* yacc.c:1661  */
    break;

  case 208:
#line 746 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.mbrlist) = 0; }
#line 3523 "y.tab.c" /* yacc.c:1661  */
    break;

  case 209:
#line 748 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.mbrlist) = (yyvsp[-2].mbrlist); }
#line 3529 "y.tab.c" /* yacc.c:1661  */
    break;

  case 210:
#line 752 "xi-grammar.y" /* yacc.c:1661  */
    {
                  if (!connectEntries.empty()) {
                    (yyval.mbrlist) = new AstChildren<Member>(connectEntries);
		  } else {
		    (yyval.mbrlist) = 0;
                  }
		}
#line 3541 "y.tab.c" /* yacc.c:1661  */
    break;

  case 211:
#line 760 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.mbrlist) = new AstChildren<Member>(-1, (yyvsp[-1].member), (yyvsp[0].mbrlist)); }
#line 3547 "y.tab.c" /* yacc.c:1661  */
    break;

  case 212:
#line 764 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = (yyvsp[0].readonly); }
#line 3553 "y.tab.c" /* yacc.c:1661  */
    break;

  case 213:
#line 766 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = (yyvsp[0].readonly); }
#line 3559 "y.tab.c" /* yacc.c:1661  */
    break;

  case 215:
#line 769 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = (yyvsp[0].member); }
#line 3565 "y.tab.c" /* yacc.c:1661  */
    break;

  case 216:
#line 771 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = (yyvsp[0].pupable); }
#line 3571 "y.tab.c" /* yacc.c:1661  */
    break;

  case 217:
#line 773 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = (yyvsp[0].includeFile); }
#line 3577 "y.tab.c" /* yacc.c:1661  */
    break;

  case 218:
#line 775 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = new ClassDeclaration(lineno,(yyvsp[0].strval)); }
#line 3583 "y.tab.c" /* yacc.c:1661  */
    break;

  case 219:
#line 779 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1); }
#line 3589 "y.tab.c" /* yacc.c:1661  */
    break;

  case 220:
#line 781 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1); }
#line 3595 "y.tab.c" /* yacc.c:1661  */
    break;

  case 221:
#line 783 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[-6].strval)) + '<' +
					    ((yyvsp[-4].tparlist))->to_string() + '>').c_str()),
				    1);
		}
#line 3605 "y.tab.c" /* yacc.c:1661  */
    break;

  case 222:
#line 789 "xi-grammar.y" /* yacc.c:1661  */
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[-2]).first_column, (yylsp[-2]).last_column, (yylsp[-2]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1);
		}
#line 3615 "y.tab.c" /* yacc.c:1661  */
    break;

  case 223:
#line 795 "xi-grammar.y" /* yacc.c:1661  */
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[-5]).first_column, (yylsp[-5]).last_column, (yylsp[-5]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1);
		}
#line 3625 "y.tab.c" /* yacc.c:1661  */
    break;

  case 224:
#line 804 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 0); }
#line 3631 "y.tab.c" /* yacc.c:1661  */
    break;

  case 225:
#line 806 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 0); }
#line 3637 "y.tab.c" /* yacc.c:1661  */
    break;

  case 226:
#line 808 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[-6].strval)) + '<' +
					    ((yyvsp[-4].tparlist))->to_string() + '>').c_str()),
				    0);
		}
#line 3647 "y.tab.c" /* yacc.c:1661  */
    break;

  case 227:
#line 814 "xi-grammar.y" /* yacc.c:1661  */
    {
                  InitCall* rtn = new InitCall(lineno, (yyvsp[-3].strval), 0);
                  rtn->setAccel();
                  (yyval.member) = rtn;
		}
#line 3657 "y.tab.c" /* yacc.c:1661  */
    break;

  case 228:
#line 822 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[0].ntype),0); }
#line 3663 "y.tab.c" /* yacc.c:1661  */
    break;

  case 229:
#line 824 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[-2].ntype),(yyvsp[0].pupable)); }
#line 3669 "y.tab.c" /* yacc.c:1661  */
    break;

  case 230:
#line 827 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.includeFile) = new IncludeFile(lineno,(yyvsp[0].strval)); }
#line 3675 "y.tab.c" /* yacc.c:1661  */
    break;

  case 231:
#line 831 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = (yyvsp[0].member); }
#line 3681 "y.tab.c" /* yacc.c:1661  */
    break;

  case 232:
#line 835 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = (yyvsp[0].entry); }
#line 3687 "y.tab.c" /* yacc.c:1661  */
    break;

  case 233:
#line 837 "xi-grammar.y" /* yacc.c:1661  */
    {
                  (yyvsp[0].entry)->tspec = (yyvsp[-1].tvarlist);
                  (yyval.member) = (yyvsp[0].entry);
                }
#line 3696 "y.tab.c" /* yacc.c:1661  */
    break;

  case 234:
#line 842 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = (yyvsp[-1].member); }
#line 3702 "y.tab.c" /* yacc.c:1661  */
    break;

  case 235:
#line 844 "xi-grammar.y" /* yacc.c:1661  */
    {
          ERROR("invalid SDAG member",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 3712 "y.tab.c" /* yacc.c:1661  */
    break;

  case 236:
#line 852 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3718 "y.tab.c" /* yacc.c:1661  */
    break;

  case 237:
#line 854 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3724 "y.tab.c" /* yacc.c:1661  */
    break;

  case 238:
#line 856 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3730 "y.tab.c" /* yacc.c:1661  */
    break;

  case 239:
#line 858 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3736 "y.tab.c" /* yacc.c:1661  */
    break;

  case 240:
#line 860 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3742 "y.tab.c" /* yacc.c:1661  */
    break;

  case 241:
#line 862 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3748 "y.tab.c" /* yacc.c:1661  */
    break;

  case 242:
#line 864 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3754 "y.tab.c" /* yacc.c:1661  */
    break;

  case 243:
#line 866 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3760 "y.tab.c" /* yacc.c:1661  */
    break;

  case 244:
#line 868 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3766 "y.tab.c" /* yacc.c:1661  */
    break;

  case 245:
#line 870 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3772 "y.tab.c" /* yacc.c:1661  */
    break;

  case 246:
#line 872 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3778 "y.tab.c" /* yacc.c:1661  */
    break;

  case 247:
#line 875 "xi-grammar.y" /* yacc.c:1661  */
    {
                  (yyval.entry) = new Entry(lineno, (yyvsp[-5].intval), (yyvsp[-4].type), (yyvsp[-3].strval), (yyvsp[-2].plist), (yyvsp[-1].val), (yyvsp[0].sentry), (const char *) NULL, (yylsp[-6]).first_line, (yyloc).last_line);
		  if ((yyvsp[0].sentry) != 0) {
		    (yyvsp[0].sentry)->con1 = new SdagConstruct(SIDENT, (yyvsp[-3].strval));
                    (yyvsp[0].sentry)->setEntry((yyval.entry));
                    (yyvsp[0].sentry)->param = new ParamList((yyvsp[-2].plist));
                  }
                  firstRdma = true;
		}
#line 3792 "y.tab.c" /* yacc.c:1661  */
    break;

  case 248:
#line 885 "xi-grammar.y" /* yacc.c:1661  */
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
#line 3813 "y.tab.c" /* yacc.c:1661  */
    break;

  case 249:
#line 902 "xi-grammar.y" /* yacc.c:1661  */
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
#line 3832 "y.tab.c" /* yacc.c:1661  */
    break;

  case 250:
#line 919 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.accelBlock) = new AccelBlock(lineno, new XStr((yyvsp[-2].strval))); }
#line 3838 "y.tab.c" /* yacc.c:1661  */
    break;

  case 251:
#line 921 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.accelBlock) = new AccelBlock(lineno, NULL); }
#line 3844 "y.tab.c" /* yacc.c:1661  */
    break;

  case 252:
#line 925 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3850 "y.tab.c" /* yacc.c:1661  */
    break;

  case 253:
#line 929 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 0; }
#line 3856 "y.tab.c" /* yacc.c:1661  */
    break;

  case 254:
#line 931 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = (yyvsp[-1].intval); }
#line 3862 "y.tab.c" /* yacc.c:1661  */
    break;

  case 255:
#line 933 "xi-grammar.y" /* yacc.c:1661  */
    { ERROR("invalid entry method attribute list",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
#line 3871 "y.tab.c" /* yacc.c:1661  */
    break;

  case 256:
#line 940 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = (yyvsp[0].intval); }
#line 3877 "y.tab.c" /* yacc.c:1661  */
    break;

  case 257:
#line 942 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = (yyvsp[-2].intval) | (yyvsp[0].intval); }
#line 3883 "y.tab.c" /* yacc.c:1661  */
    break;

  case 258:
#line 946 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = STHREADED; }
#line 3889 "y.tab.c" /* yacc.c:1661  */
    break;

  case 259:
#line 948 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SSYNC; }
#line 3895 "y.tab.c" /* yacc.c:1661  */
    break;

  case 260:
#line 950 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SIGET; }
#line 3901 "y.tab.c" /* yacc.c:1661  */
    break;

  case 261:
#line 952 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SLOCKED; }
#line 3907 "y.tab.c" /* yacc.c:1661  */
    break;

  case 262:
#line 954 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SCREATEHERE; }
#line 3913 "y.tab.c" /* yacc.c:1661  */
    break;

  case 263:
#line 956 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SCREATEHOME; }
#line 3919 "y.tab.c" /* yacc.c:1661  */
    break;

  case 264:
#line 958 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SNOKEEP; }
#line 3925 "y.tab.c" /* yacc.c:1661  */
    break;

  case 265:
#line 960 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SNOTRACE; }
#line 3931 "y.tab.c" /* yacc.c:1661  */
    break;

  case 266:
#line 962 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SAPPWORK; }
#line 3937 "y.tab.c" /* yacc.c:1661  */
    break;

  case 267:
#line 964 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SIMMEDIATE; }
#line 3943 "y.tab.c" /* yacc.c:1661  */
    break;

  case 268:
#line 966 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SSKIPSCHED; }
#line 3949 "y.tab.c" /* yacc.c:1661  */
    break;

  case 269:
#line 968 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SINLINE; }
#line 3955 "y.tab.c" /* yacc.c:1661  */
    break;

  case 270:
#line 970 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SLOCAL; }
#line 3961 "y.tab.c" /* yacc.c:1661  */
    break;

  case 271:
#line 972 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SPYTHON; }
#line 3967 "y.tab.c" /* yacc.c:1661  */
    break;

  case 272:
#line 974 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SMEM; }
#line 3973 "y.tab.c" /* yacc.c:1661  */
    break;

  case 273:
#line 976 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SREDUCE; }
#line 3979 "y.tab.c" /* yacc.c:1661  */
    break;

  case 274:
#line 978 "xi-grammar.y" /* yacc.c:1661  */
    {
        (yyval.intval) = SAGGREGATE;
    }
#line 3987 "y.tab.c" /* yacc.c:1661  */
    break;

  case 275:
#line 982 "xi-grammar.y" /* yacc.c:1661  */
    {
		  ERROR("invalid entry method attribute",
		        (yylsp[0]).first_column, (yylsp[0]).last_column);
		  yyclearin;
		  yyerrok;
		}
#line 3998 "y.tab.c" /* yacc.c:1661  */
    break;

  case 276:
#line 991 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4004 "y.tab.c" /* yacc.c:1661  */
    break;

  case 277:
#line 993 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4010 "y.tab.c" /* yacc.c:1661  */
    break;

  case 278:
#line 995 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4016 "y.tab.c" /* yacc.c:1661  */
    break;

  case 279:
#line 999 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = ""; }
#line 4022 "y.tab.c" /* yacc.c:1661  */
    break;

  case 280:
#line 1001 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4028 "y.tab.c" /* yacc.c:1661  */
    break;

  case 281:
#line 1003 "xi-grammar.y" /* yacc.c:1661  */
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s, %s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4038 "y.tab.c" /* yacc.c:1661  */
    break;

  case 282:
#line 1011 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = ""; }
#line 4044 "y.tab.c" /* yacc.c:1661  */
    break;

  case 283:
#line 1013 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4050 "y.tab.c" /* yacc.c:1661  */
    break;

  case 284:
#line 1015 "xi-grammar.y" /* yacc.c:1661  */
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s[%s]%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4060 "y.tab.c" /* yacc.c:1661  */
    break;

  case 285:
#line 1021 "xi-grammar.y" /* yacc.c:1661  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s{%s}%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4070 "y.tab.c" /* yacc.c:1661  */
    break;

  case 286:
#line 1027 "xi-grammar.y" /* yacc.c:1661  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s(%s)%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4080 "y.tab.c" /* yacc.c:1661  */
    break;

  case 287:
#line 1033 "xi-grammar.y" /* yacc.c:1661  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"(%s)%s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4090 "y.tab.c" /* yacc.c:1661  */
    break;

  case 288:
#line 1041 "xi-grammar.y" /* yacc.c:1661  */
    {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval));
		}
#line 4099 "y.tab.c" /* yacc.c:1661  */
    break;

  case 289:
#line 1048 "xi-grammar.y" /* yacc.c:1661  */
    {
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			(yyval.intval) = 0;
		}
#line 4109 "y.tab.c" /* yacc.c:1661  */
    break;

  case 290:
#line 1056 "xi-grammar.y" /* yacc.c:1661  */
    {
			in_braces=0;
			(yyval.intval) = 0;
		}
#line 4118 "y.tab.c" /* yacc.c:1661  */
    break;

  case 291:
#line 1063 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[0].type));}
#line 4124 "y.tab.c" /* yacc.c:1661  */
    break;

  case 292:
#line 1065 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval)); (yyval.pname)->setConditional((yyvsp[0].intval)); }
#line 4130 "y.tab.c" /* yacc.c:1661  */
    break;

  case 293:
#line 1067 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[-3].type),(yyvsp[-2].strval),0,(yyvsp[0].val));}
#line 4136 "y.tab.c" /* yacc.c:1661  */
    break;

  case 294:
#line 1069 "xi-grammar.y" /* yacc.c:1661  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
		}
#line 4145 "y.tab.c" /* yacc.c:1661  */
    break;

  case 295:
#line 1074 "xi-grammar.y" /* yacc.c:1661  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
			(yyval.pname)->setRdma(true);
			if(firstRdma) {
				(yyval.pname)->setFirstRdma(true);
				firstRdma = false;
			}
		}
#line 4159 "y.tab.c" /* yacc.c:1661  */
    break;

  case 296:
#line 1085 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READONLY; }
#line 4165 "y.tab.c" /* yacc.c:1661  */
    break;

  case 297:
#line 1086 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READWRITE; }
#line 4171 "y.tab.c" /* yacc.c:1661  */
    break;

  case 298:
#line 1087 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_WRITEONLY; }
#line 4177 "y.tab.c" /* yacc.c:1661  */
    break;

  case 299:
#line 1090 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.xstrptr) = new XStr((yyvsp[0].strval)); }
#line 4183 "y.tab.c" /* yacc.c:1661  */
    break;

  case 300:
#line 1091 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "->" << (yyvsp[0].strval); }
#line 4189 "y.tab.c" /* yacc.c:1661  */
    break;

  case 301:
#line 1092 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[-2].xstrptr)) << "." << (yyvsp[0].strval); }
#line 4195 "y.tab.c" /* yacc.c:1661  */
    break;

  case 302:
#line 1094 "xi-grammar.y" /* yacc.c:1661  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << *((yyvsp[-1].xstrptr)) << "]";
                  delete (yyvsp[-3].xstrptr);
                  delete (yyvsp[-1].xstrptr);
                }
#line 4206 "y.tab.c" /* yacc.c:1661  */
    break;

  case 303:
#line 1101 "xi-grammar.y" /* yacc.c:1661  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << (yyvsp[-1].strval) << "]";
                  delete (yyvsp[-3].xstrptr);
                }
#line 4216 "y.tab.c" /* yacc.c:1661  */
    break;

  case 304:
#line 1107 "xi-grammar.y" /* yacc.c:1661  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "(" << *((yyvsp[-1].xstrptr)) << ")";
                  delete (yyvsp[-3].xstrptr);
                  delete (yyvsp[-1].xstrptr);
                }
#line 4227 "y.tab.c" /* yacc.c:1661  */
    break;

  case 305:
#line 1116 "xi-grammar.y" /* yacc.c:1661  */
    {
                  in_bracket = 0;
                  (yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName(), (yyvsp[-1].strval));
                }
#line 4236 "y.tab.c" /* yacc.c:1661  */
    break;

  case 306:
#line 1123 "xi-grammar.y" /* yacc.c:1661  */
    {
                  (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[-6].intval));
                }
#line 4246 "y.tab.c" /* yacc.c:1661  */
    break;

  case 307:
#line 1129 "xi-grammar.y" /* yacc.c:1661  */
    {
		  (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType(Parameter::ACCEL_BUFFER_TYPE_READWRITE);
		}
#line 4256 "y.tab.c" /* yacc.c:1661  */
    break;

  case 308:
#line 1135 "xi-grammar.y" /* yacc.c:1661  */
    {
                  (yyval.pname) = (yyvsp[-3].pname);
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[-5].intval));
		}
#line 4266 "y.tab.c" /* yacc.c:1661  */
    break;

  case 309:
#line 1143 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 4272 "y.tab.c" /* yacc.c:1661  */
    break;

  case 310:
#line 1145 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 4278 "y.tab.c" /* yacc.c:1661  */
    break;

  case 311:
#line 1149 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 4284 "y.tab.c" /* yacc.c:1661  */
    break;

  case 312:
#line 1151 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 4290 "y.tab.c" /* yacc.c:1661  */
    break;

  case 313:
#line 1155 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.plist) = (yyvsp[-1].plist); }
#line 4296 "y.tab.c" /* yacc.c:1661  */
    break;

  case 314:
#line 1157 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.plist) = new ParamList(new Parameter(0, new BuiltinType("void"))); }
#line 4302 "y.tab.c" /* yacc.c:1661  */
    break;

  case 315:
#line 1161 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.plist) = (yyvsp[-1].plist); }
#line 4308 "y.tab.c" /* yacc.c:1661  */
    break;

  case 316:
#line 1163 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.plist) = 0; }
#line 4314 "y.tab.c" /* yacc.c:1661  */
    break;

  case 317:
#line 1167 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.val) = 0; }
#line 4320 "y.tab.c" /* yacc.c:1661  */
    break;

  case 318:
#line 1169 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4326 "y.tab.c" /* yacc.c:1661  */
    break;

  case 319:
#line 1173 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sentry) = 0; }
#line 4332 "y.tab.c" /* yacc.c:1661  */
    break;

  case 320:
#line 1175 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[0].sc)); }
#line 4338 "y.tab.c" /* yacc.c:1661  */
    break;

  case 321:
#line 1177 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[-2].slist)); }
#line 4344 "y.tab.c" /* yacc.c:1661  */
    break;

  case 322:
#line 1181 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.slist) = new SListConstruct((yyvsp[0].sc)); }
#line 4350 "y.tab.c" /* yacc.c:1661  */
    break;

  case 323:
#line 1183 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.slist) = new SListConstruct((yyvsp[-1].sc), (yyvsp[0].slist));  }
#line 4356 "y.tab.c" /* yacc.c:1661  */
    break;

  case 324:
#line 1187 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.olist) = new OListConstruct((yyvsp[0].sc)); }
#line 4362 "y.tab.c" /* yacc.c:1661  */
    break;

  case 325:
#line 1189 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.olist) = new OListConstruct((yyvsp[-1].sc), (yyvsp[0].slist)); }
#line 4368 "y.tab.c" /* yacc.c:1661  */
    break;

  case 326:
#line 1193 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.clist) = new CaseListConstruct((yyvsp[0].when)); }
#line 4374 "y.tab.c" /* yacc.c:1661  */
    break;

  case 327:
#line 1195 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.clist) = new CaseListConstruct((yyvsp[-1].when), (yyvsp[0].clist)); }
#line 4380 "y.tab.c" /* yacc.c:1661  */
    break;

  case 328:
#line 1197 "xi-grammar.y" /* yacc.c:1661  */
    {
		  ERROR("case blocks can only contain when clauses",
		        (yylsp[0]).first_column, (yylsp[0]).last_column);
		  (yyval.clist) = 0;
		}
#line 4390 "y.tab.c" /* yacc.c:1661  */
    break;

  case 329:
#line 1205 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4396 "y.tab.c" /* yacc.c:1661  */
    break;

  case 330:
#line 1207 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = 0; }
#line 4402 "y.tab.c" /* yacc.c:1661  */
    break;

  case 331:
#line 1211 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = new WhenConstruct((yyvsp[-2].entrylist), 0); }
#line 4408 "y.tab.c" /* yacc.c:1661  */
    break;

  case 332:
#line 1213 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = new WhenConstruct((yyvsp[-1].entrylist), (yyvsp[0].sc)); }
#line 4414 "y.tab.c" /* yacc.c:1661  */
    break;

  case 333:
#line 1215 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = new WhenConstruct((yyvsp[-3].entrylist), (yyvsp[-1].slist)); }
#line 4420 "y.tab.c" /* yacc.c:1661  */
    break;

  case 334:
#line 1219 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4426 "y.tab.c" /* yacc.c:1661  */
    break;

  case 335:
#line 1221 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4432 "y.tab.c" /* yacc.c:1661  */
    break;

  case 336:
#line 1223 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4438 "y.tab.c" /* yacc.c:1661  */
    break;

  case 337:
#line 1225 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4444 "y.tab.c" /* yacc.c:1661  */
    break;

  case 338:
#line 1227 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4450 "y.tab.c" /* yacc.c:1661  */
    break;

  case 339:
#line 1229 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4456 "y.tab.c" /* yacc.c:1661  */
    break;

  case 340:
#line 1231 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4462 "y.tab.c" /* yacc.c:1661  */
    break;

  case 341:
#line 1233 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4468 "y.tab.c" /* yacc.c:1661  */
    break;

  case 342:
#line 1235 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4474 "y.tab.c" /* yacc.c:1661  */
    break;

  case 343:
#line 1237 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4480 "y.tab.c" /* yacc.c:1661  */
    break;

  case 344:
#line 1239 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4486 "y.tab.c" /* yacc.c:1661  */
    break;

  case 345:
#line 1241 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4492 "y.tab.c" /* yacc.c:1661  */
    break;

  case 346:
#line 1245 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new SerialConstruct((yyvsp[-2].strval), (yyvsp[-4].strval), (yylsp[-3]).first_line); }
#line 4498 "y.tab.c" /* yacc.c:1661  */
    break;

  case 347:
#line 1247 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new OverlapConstruct((yyvsp[-1].olist)); }
#line 4504 "y.tab.c" /* yacc.c:1661  */
    break;

  case 348:
#line 1249 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = (yyvsp[0].when); }
#line 4510 "y.tab.c" /* yacc.c:1661  */
    break;

  case 349:
#line 1251 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new CaseConstruct((yyvsp[-1].clist)); }
#line 4516 "y.tab.c" /* yacc.c:1661  */
    break;

  case 350:
#line 1253 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new ForConstruct((yyvsp[-8].intexpr), (yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4522 "y.tab.c" /* yacc.c:1661  */
    break;

  case 351:
#line 1255 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new ForConstruct((yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4528 "y.tab.c" /* yacc.c:1661  */
    break;

  case 352:
#line 1257 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-9].strval)), (yyvsp[-6].intexpr),
		             (yyvsp[-4].intexpr), (yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4535 "y.tab.c" /* yacc.c:1661  */
    break;

  case 353:
#line 1260 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-11].strval)), (yyvsp[-8].intexpr),
		             (yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4542 "y.tab.c" /* yacc.c:1661  */
    break;

  case 354:
#line 1263 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new IfConstruct((yyvsp[-3].intexpr), (yyvsp[-1].sc), (yyvsp[0].sc)); }
#line 4548 "y.tab.c" /* yacc.c:1661  */
    break;

  case 355:
#line 1265 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new IfConstruct((yyvsp[-5].intexpr), (yyvsp[-2].slist), (yyvsp[0].sc)); }
#line 4554 "y.tab.c" /* yacc.c:1661  */
    break;

  case 356:
#line 1267 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new WhileConstruct((yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4560 "y.tab.c" /* yacc.c:1661  */
    break;

  case 357:
#line 1269 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new WhileConstruct((yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4566 "y.tab.c" /* yacc.c:1661  */
    break;

  case 358:
#line 1271 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new SerialConstruct((yyvsp[-2].strval), NULL, (yyloc).first_line); }
#line 4572 "y.tab.c" /* yacc.c:1661  */
    break;

  case 359:
#line 1273 "xi-grammar.y" /* yacc.c:1661  */
    {
		  ERROR("unknown SDAG construct or malformed entry method declaration.\n"
		        "You may have forgotten to terminate a previous entry method declaration with a"
		        " semicolon or forgotten to mark a block of sequential SDAG code as 'serial'",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
#line 4584 "y.tab.c" /* yacc.c:1661  */
    break;

  case 360:
#line 1283 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = 0; }
#line 4590 "y.tab.c" /* yacc.c:1661  */
    break;

  case 361:
#line 1285 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new ElseConstruct((yyvsp[0].sc)); }
#line 4596 "y.tab.c" /* yacc.c:1661  */
    break;

  case 362:
#line 1287 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new ElseConstruct((yyvsp[-1].slist)); }
#line 4602 "y.tab.c" /* yacc.c:1661  */
    break;

  case 363:
#line 1291 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intexpr) = new IntExprConstruct((yyvsp[0].strval)); }
#line 4608 "y.tab.c" /* yacc.c:1661  */
    break;

  case 364:
#line 1295 "xi-grammar.y" /* yacc.c:1661  */
    { in_int_expr = 0; (yyval.intval) = 0; }
#line 4614 "y.tab.c" /* yacc.c:1661  */
    break;

  case 365:
#line 1299 "xi-grammar.y" /* yacc.c:1661  */
    { in_int_expr = 1; (yyval.intval) = 0; }
#line 4620 "y.tab.c" /* yacc.c:1661  */
    break;

  case 366:
#line 1303 "xi-grammar.y" /* yacc.c:1661  */
    {
		  (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[-1].strval), (yyvsp[0].plist), 0, 0, 0, (yyloc).first_line, (yyloc).last_line);
		  firstRdma = true;
		}
#line 4629 "y.tab.c" /* yacc.c:1661  */
    break;

  case 367:
#line 1308 "xi-grammar.y" /* yacc.c:1661  */
    {
		  (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[-4].strval), (yyvsp[0].plist), 0, 0, (yyvsp[-2].strval), (yyloc).first_line, (yyloc).last_line);
		  firstRdma = true;
		}
#line 4638 "y.tab.c" /* yacc.c:1661  */
    break;

  case 368:
#line 1315 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.entrylist) = new EntryList((yyvsp[0].entry)); }
#line 4644 "y.tab.c" /* yacc.c:1661  */
    break;

  case 369:
#line 1317 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.entrylist) = new EntryList((yyvsp[-2].entry),(yyvsp[0].entrylist)); }
#line 4650 "y.tab.c" /* yacc.c:1661  */
    break;

  case 370:
#line 1321 "xi-grammar.y" /* yacc.c:1661  */
    { in_bracket=1; }
#line 4656 "y.tab.c" /* yacc.c:1661  */
    break;

  case 371:
#line 1324 "xi-grammar.y" /* yacc.c:1661  */
    { in_bracket=0; }
#line 4662 "y.tab.c" /* yacc.c:1661  */
    break;

  case 372:
#line 1328 "xi-grammar.y" /* yacc.c:1661  */
    { if (!macroDefined((yyvsp[0].strval), 1)) in_comment = 1; }
#line 4668 "y.tab.c" /* yacc.c:1661  */
    break;

  case 373:
#line 1332 "xi-grammar.y" /* yacc.c:1661  */
    { if (!macroDefined((yyvsp[0].strval), 0)) in_comment = 1; }
#line 4674 "y.tab.c" /* yacc.c:1661  */
    break;


#line 4678 "y.tab.c" /* yacc.c:1661  */
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
#line 1335 "xi-grammar.y" /* yacc.c:1906  */


void yyerror(const char *s)
{
	fprintf(stderr, "[PARSE-ERROR] Unexpected/missing token at line %d. Current token being parsed: '%s'.\n", lineno, yytext);
}
