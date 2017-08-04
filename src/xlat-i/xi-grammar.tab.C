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
bool firstRdma = true;

bool enable_warnings = true;

extern int macroDefined(const char *str, int istrue);
extern const char *python_doc;
extern char *fname;
void splitScopedName(const char* name, const char** scope, const char** basename);
void ReservedWord(int token, int fCol, int lCol);
}

#line 114 "y.tab.c" /* yacc.c:339  */

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
#line 52 "xi-grammar.y" /* yacc.c:355  */

  Attribute *attr;
  Attribute::Argument *attrarg;
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

#line 348 "y.tab.c" /* yacc.c:355  */
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

#line 379 "y.tab.c" /* yacc.c:358  */

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
#define YYLAST   1509

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  91
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  119
/* YYNRULES -- Number of rules.  */
#define YYNRULES  376
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  737

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
       0,   199,   199,   204,   207,   212,   213,   217,   219,   224,
     225,   230,   232,   233,   234,   236,   237,   238,   240,   241,
     242,   243,   244,   248,   249,   250,   251,   252,   253,   254,
     255,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   268,   269,   270,   271,   272,   273,   274,   275,   276,
     277,   278,   280,   282,   283,   286,   287,   288,   289,   293,
     295,   302,   306,   313,   315,   320,   321,   325,   327,   329,
     331,   333,   346,   348,   350,   352,   358,   360,   362,   364,
     366,   368,   370,   372,   374,   376,   384,   386,   388,   392,
     394,   399,   400,   405,   406,   410,   412,   414,   416,   418,
     420,   422,   424,   426,   428,   430,   432,   434,   436,   438,
     442,   443,   450,   452,   456,   460,   462,   466,   470,   472,
     474,   476,   478,   480,   484,   486,   488,   490,   492,   496,
     498,   502,   504,   508,   512,   517,   518,   522,   526,   531,
     532,   537,   538,   548,   550,   554,   556,   561,   562,   566,
     568,   573,   574,   578,   583,   584,   588,   590,   594,   596,
     601,   602,   606,   607,   610,   614,   616,   620,   622,   624,
     629,   630,   634,   636,   640,   642,   646,   650,   654,   660,
     664,   666,   670,   672,   676,   680,   684,   688,   690,   695,
     696,   701,   702,   704,   706,   715,   717,   719,   723,   725,
     729,   733,   735,   737,   739,   741,   745,   747,   752,   759,
     763,   765,   767,   768,   770,   772,   774,   778,   780,   782,
     788,   794,   803,   805,   807,   813,   821,   823,   826,   830,
     834,   836,   841,   843,   851,   853,   855,   857,   859,   861,
     863,   865,   867,   869,   871,   874,   884,   901,   918,   920,
     924,   929,   930,   932,   940,   944,   945,   949,   950,   951,
     952,   955,   957,   959,   961,   963,   965,   967,   969,   971,
     973,   975,   977,   979,   981,   983,   985,   987,   999,  1008,
    1010,  1012,  1017,  1018,  1020,  1029,  1030,  1032,  1038,  1044,
    1050,  1058,  1065,  1073,  1080,  1082,  1084,  1086,  1091,  1103,
    1104,  1105,  1108,  1109,  1110,  1111,  1118,  1124,  1133,  1140,
    1146,  1152,  1160,  1162,  1166,  1168,  1172,  1174,  1178,  1180,
    1185,  1186,  1190,  1192,  1194,  1198,  1200,  1204,  1206,  1210,
    1212,  1214,  1222,  1225,  1228,  1230,  1232,  1236,  1238,  1240,
    1242,  1244,  1246,  1248,  1250,  1252,  1254,  1256,  1258,  1262,
    1264,  1266,  1268,  1270,  1272,  1274,  1277,  1280,  1282,  1284,
    1286,  1288,  1290,  1301,  1302,  1304,  1308,  1312,  1316,  1320,
    1325,  1332,  1334,  1338,  1341,  1345,  1349
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
  "EReturn", "EAttribs", "AttributeArg", "AttributeArgList", "EAttribList",
  "EAttrib", "DefaultParameter", "CPROGRAM_List", "CCode",
  "ParamBracketStart", "ParamBraceStart", "ParamBraceEnd", "Parameter",
  "AccelBufferType", "AccelInstName", "AccelArrayParam", "AccelParameter",
  "ParamList", "AccelParamList", "EParameters", "AccelEParameters",
  "OptStackSize", "OptSdagCode", "Slist", "Olist", "CaseList",
  "OptTraceName", "WhenConstruct", "NonWhenConstruct", "SingleConstruct",
  "HasElse", "IntExpr", "EndIntExpr", "StartIntExpr", "SEntry",
  "SEntryList", "SParamBracketStart", "SParamBracketEnd", "HashIFComment",
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

#define YYPACT_NINF -631

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-631)))

#define YYTABLE_NINF -328

#define yytable_value_is_error(Yytable_value) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
     284,  1293,  1293,    46,  -631,   284,  -631,  -631,  -631,  -631,
    -631,  -631,  -631,  -631,  -631,  -631,  -631,  -631,  -631,  -631,
    -631,  -631,  -631,  -631,  -631,  -631,  -631,  -631,  -631,  -631,
    -631,  -631,  -631,  -631,  -631,  -631,  -631,  -631,  -631,  -631,
    -631,  -631,  -631,  -631,  -631,  -631,  -631,  -631,  -631,  -631,
    -631,  -631,  -631,  -631,   118,   118,  -631,  -631,  -631,   779,
     -26,  -631,  -631,  -631,    50,  1293,   154,  1293,  1293,   193,
     880,    36,   929,   779,  -631,  -631,  -631,  -631,  1411,    45,
      95,  -631,    88,  -631,  -631,  -631,   -26,   -38,  1314,   121,
     121,    -3,    95,    98,    98,    98,    98,   131,   142,  1293,
     124,   159,   779,  -631,  -631,  -631,  -631,  -631,  -631,  -631,
    -631,   597,  -631,  -631,  -631,  -631,   103,  -631,  -631,  -631,
    -631,  -631,  -631,  -631,  -631,  -631,  -631,  -631,   -26,  -631,
    -631,  -631,  1411,  -631,   152,  -631,  -631,  -631,  -631,   212,
     175,  -631,  -631,   162,   168,   170,     4,  -631,    95,   779,
      88,   183,    82,   -38,   200,  1442,  1427,   162,   168,   170,
    -631,    44,    95,  -631,    95,    95,   203,    95,   219,  -631,
      72,  1293,  1293,  1293,  1293,  1077,   215,   221,   281,  1293,
    -631,  -631,  -631,  1335,   250,    98,    98,    98,    98,   215,
     142,  -631,  -631,  -631,  -631,  -631,   -26,  -631,   295,  -631,
    -631,  -631,   276,  -631,  -631,  1396,  -631,  -631,  -631,  -631,
    -631,  -631,  1293,   257,   264,   -38,   283,   -38,   261,  -631,
     103,   279,    -9,  -631,   282,  -631,    39,   114,   116,   259,
     160,    95,  -631,  -631,   278,   287,   290,   291,   291,   291,
     291,  -631,  1293,   285,   292,   293,  1149,  1293,   323,  1293,
    -631,  -631,   296,   298,   302,  1293,    92,  1293,   305,   307,
     103,  1293,  1293,  1293,  1293,  1293,  1293,  -631,  -631,  -631,
    -631,   306,  -631,   308,  -631,   290,  -631,  -631,   312,   313,
     310,   299,   -38,   -26,    95,  1293,  -631,   311,  -631,   -38,
     121,  1396,   121,   121,  1396,   121,  -631,  -631,    72,  -631,
      95,   206,   206,   206,   206,   309,  -631,   323,  -631,   291,
     291,  -631,   281,     2,   321,   270,  -631,   322,  1335,  -631,
    -631,   291,   291,   291,   291,   291,   241,  1396,  -631,   315,
     -38,   283,   -38,   -38,  -631,    39,   328,  -631,   326,  -631,
     330,   339,   342,    95,   347,   350,  -631,   344,  -631,   437,
     -26,  -631,  -631,  -631,  -631,  -631,  -631,   206,   206,  -631,
    -631,  -631,  1427,    16,   357,  1427,  -631,  -631,  -631,  -631,
    -631,  -631,   206,   206,   206,   206,   206,   420,   -26,  -631,
    1348,  -631,  -631,  -631,  -631,  -631,  -631,   356,  -631,  -631,
    -631,   358,  -631,   130,   359,  -631,    95,  -631,   695,   398,
     370,   103,   437,  -631,  -631,  -631,  -631,  1293,  -631,  -631,
    -631,  -631,  -631,  -631,  -631,  -631,   371,  1427,  -631,  1293,
     -38,   373,   366,   615,   121,   121,   121,  -631,  -631,   896,
    1005,  -631,   103,   -26,  -631,   367,   103,  1293,   -38,     9,
     385,   615,  -631,   374,   376,   389,   391,  -631,  -631,  -631,
    -631,  -631,  -631,  -631,  -631,  -631,  -631,  -631,  -631,  -631,
    -631,   397,  -631,   390,  -631,  -631,   393,   174,   399,   315,
    1293,  -631,   394,   103,   -26,   405,   407,  -631,   275,  -631,
    -631,  -631,  -631,  -631,  -631,  -631,  -631,  -631,   447,  -631,
     949,  1293,   108,   315,  -631,   -26,  -631,  -631,  -631,    88,
    -631,  1293,  -631,   381,   419,   427,  -631,   421,   417,   421,
     458,   439,   457,   421,   440,   246,   -26,   -38,  -631,  -631,
    -631,   500,   315,   465,  1293,   443,  -631,   -38,   469,   -38,
      51,   446,   507,   518,  -631,   449,   -38,  1373,   453,   265,
     200,   445,   108,   441,  -631,  -631,   949,  -631,   459,   448,
     455,  -631,   -38,   458,   304,  -631,   462,   418,   -38,   455,
     421,   451,   421,   464,   457,   421,   466,   -38,   467,  1373,
    -631,   103,  -631,   103,   488,  -631,   338,   449,  -631,   -38,
     421,  -631,   528,   326,  -631,  -631,   468,  -631,  -631,   200,
     584,   -38,   492,   -38,   518,   449,   -38,  1373,   200,  -631,
    -631,  -631,  -631,  -631,  -631,  -631,  -631,  -631,  1293,   481,
     479,   473,   -38,   493,   -38,   246,  -631,   315,  -631,   103,
     246,   520,   504,   495,   455,   505,   -38,   455,   508,   103,
     509,  1427,   798,  -631,   200,   -38,   511,   512,  -631,  -631,
     513,   598,  -631,   -38,   421,   620,  -631,   200,   763,  -631,
    -631,  1293,  1293,   -38,   514,  -631,  1293,   455,   -38,  -631,
     520,   246,  -631,   522,   -38,   246,  -631,   103,   246,   520,
    -631,    80,   -39,   506,  1293,   103,   770,   519,  -631,   523,
     -38,   524,   525,  -631,   526,  -631,  -631,  1293,  1221,   533,
    1293,  1293,  -631,   341,   -26,   246,  -631,   -38,  -631,   455,
     -38,  -631,   520,   148,   530,   213,  1293,  -631,   416,  -631,
     537,   455,   777,   527,  -631,  -631,  -631,  -631,  -631,  -631,
    -631,   826,   246,  -631,   -38,   246,  -631,   540,   455,   541,
    -631,   833,  -631,   246,  -631,   542,  -631
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
       0,    59,    68,   375,   376,   292,   249,   285,     0,   139,
     139,   139,     0,   147,   147,   147,   147,     0,   141,     0,
       0,     0,     0,    76,   210,   211,    70,    77,    78,    79,
      80,     0,    81,    69,   213,   212,     9,   244,   236,   237,
     238,   239,   240,   242,   243,   241,   234,   235,    74,    75,
      66,   109,     0,    95,    96,    97,    98,   106,   107,     0,
      93,   112,   113,   124,   125,   126,   130,   250,     0,     0,
      67,     0,   286,   285,     0,     0,     0,   118,   119,   120,
     121,   132,     0,   140,     0,     0,     0,     0,   226,   214,
       0,     0,     0,     0,     0,     0,     0,   154,     0,     0,
     216,   228,   215,     0,     0,   147,   147,   147,   147,     0,
     141,   201,   202,   203,   204,   205,    10,    64,   127,   105,
     108,    99,   100,   103,   104,    91,   111,   114,   115,   116,
     128,   129,     0,     0,     0,   285,   282,   285,     0,   293,
       0,     0,   122,   123,     0,   131,   135,   220,   217,     0,
     222,     0,   158,   159,     0,   149,    93,   170,   170,   170,
     170,   153,     0,     0,   156,     0,     0,     0,     0,     0,
     145,   146,     0,   143,   167,     0,   121,     0,   198,     0,
       9,     0,     0,     0,     0,     0,     0,   101,   102,    87,
      88,    89,    92,     0,    86,    93,    73,    60,     0,   283,
       0,     0,   285,   248,     0,     0,   373,   135,   137,   285,
     139,     0,   139,   139,     0,   139,   227,   148,     0,   110,
       0,     0,     0,     0,     0,     0,   179,     0,   155,   170,
     170,   142,     0,   160,   189,     0,   196,   191,     0,   200,
      72,   170,   170,   170,   170,   170,     0,     0,    94,     0,
     285,   282,   285,   285,   290,   135,     0,   136,     0,   133,
       0,     0,     0,     0,     0,     0,   150,   172,   171,     0,
     206,   174,   175,   176,   177,   178,   157,     0,     0,   144,
     161,   168,     0,   160,     0,     0,   195,   192,   193,   194,
     197,   199,     0,     0,     0,     0,     0,   160,   187,    90,
       0,    71,   288,   284,   289,   287,   138,     0,   374,   134,
     221,     0,   218,     0,     0,   223,     0,   233,     0,     0,
       0,     0,     0,   229,   230,   180,   181,     0,   166,   169,
     190,   182,   183,   184,   185,   186,     0,     0,   317,   294,
     285,   312,     0,     0,   139,   139,   139,   173,   253,     0,
       0,   231,     9,   232,   209,   162,     0,     0,   285,   160,
       0,     0,   316,     0,     0,     0,     0,   278,   261,   262,
     263,   264,   270,   271,   272,   277,   265,   266,   267,   268,
     269,   151,   273,     0,   275,   276,     0,   257,    59,     0,
       0,   207,     0,     0,   188,     0,     0,   291,     0,   295,
     297,   313,   117,   219,   225,   224,   152,   274,     0,   252,
       0,     0,     0,     0,   163,   164,   298,   280,   279,   281,
     296,     0,   259,     0,   255,     0,   362,     0,     0,     0,
       0,     0,   333,     0,     0,     0,   322,   285,   246,   351,
     323,   320,     0,     0,     0,   258,   368,   285,     0,   285,
       0,   371,     0,     0,   332,     0,   285,     0,     0,     0,
       0,     0,     0,     0,   254,   256,     0,   366,     0,     0,
       0,   369,   285,     0,     0,   335,     0,     0,   285,     0,
       0,     0,     0,     0,   333,     0,     0,   285,     0,   329,
     331,     9,   326,     9,     0,   245,     0,     0,   260,   285,
       0,   367,     0,     0,   372,   334,     0,   350,   328,     0,
       0,   285,     0,   285,     0,     0,   285,     0,     0,   352,
     330,   324,   361,   321,   299,   300,   301,   319,     0,     0,
     314,     0,   285,     0,   285,     0,   359,     0,   336,     9,
       0,   363,     0,     0,     0,     0,   285,     0,     0,     9,
       0,     0,     0,   318,     0,   285,     0,     0,   370,   349,
       0,     0,   357,   285,     0,     0,   338,     0,     0,   339,
     348,     0,     0,   285,     0,   315,     0,     0,   285,   360,
     363,     0,   364,     0,   285,     0,   346,     9,     0,   363,
     302,     0,     0,     0,     0,     0,     0,     0,   358,     0,
     285,     0,     0,   337,     0,   344,   310,     0,     0,     0,
       0,     0,   308,     0,   247,     0,   354,   285,   365,     0,
     285,   347,   363,     0,     0,     0,     0,   304,     0,   311,
       0,     0,     0,     0,   345,   307,   306,   305,   303,   309,
     353,     0,     0,   341,   285,     0,   355,     0,     0,     0,
     340,     0,   356,     0,   342,     0,   343
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -631,  -631,   617,  -631,   -41,  -256,    -1,   -57,   553,   569,
     -37,  -631,  -631,  -631,  -178,  -631,  -221,  -631,  -117,   -90,
     -71,   -62,   -61,  -172,   480,   503,  -631,   -83,  -631,  -631,
    -275,  -631,  -631,   -69,   435,   325,  -631,   -43,   340,  -631,
    -631,   471,   332,  -631,   210,  -631,  -631,  -242,  -631,  -121,
     260,  -631,  -631,  -631,  -139,  -631,  -631,  -631,  -631,  -631,
    -631,  -631,   343,  -631,   353,   587,  -631,   113,   272,   589,
    -631,  -631,   431,  -631,  -631,  -631,  -631,   286,  -631,   253,
    -631,  -631,   164,  -447,  -631,  -631,   360,   -84,  -411,   -59,
    -508,  -631,  -631,  -458,  -631,  -631,  -303,    52,  -445,  -631,
    -631,   150,  -453,    99,  -534,   137,  -510,  -631,  -454,  -630,
    -511,  -551,  -469,  -631,   158,   165,   129,  -631,  -631
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     3,     4,    70,   350,   197,   236,   140,     5,    61,
      71,    72,    73,   271,   272,   273,   206,   141,   237,   142,
     157,   158,   159,   160,   161,   146,   147,   274,   338,   287,
     288,   104,   105,   164,   179,   252,   253,   171,   234,   487,
     244,   176,   245,   235,   362,   473,   363,   364,   106,   301,
     348,   107,   108,   109,   177,   110,   191,   192,   193,   194,
     195,   366,   316,   258,   259,   399,   112,   351,   400,   401,
     114,   115,   169,   182,   402,   403,   129,   404,    74,   148,
     430,   504,   505,   466,   467,   500,   280,   547,   420,   517,
     220,   421,   609,   671,   654,   610,   422,   611,   381,   577,
     542,   518,   538,   556,   568,   535,   519,   570,   539,   642,
     548,   582,   527,   531,   532,   289,   389,    75,    76
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[] =
{
      54,    55,   168,   154,   320,   162,   438,   143,   590,    82,
      87,   256,   337,    60,    60,   299,   144,   145,   550,   360,
     152,   165,   167,   150,   492,   559,   360,   569,    86,   223,
     678,   128,   573,   360,   163,   600,   130,   247,   520,   685,
     529,   691,   210,   502,   536,   153,    56,   477,   521,    77,
     265,   172,   173,   174,   329,   238,   239,   240,   212,   569,
     386,   143,   254,   628,    79,   184,    83,    84,   613,   218,
     144,   145,   714,   645,   224,   196,   648,   543,   555,   557,
     361,   619,   223,   166,   221,   551,   572,   569,   520,   211,
     629,   591,    78,   593,  -165,   477,   596,   478,   180,   578,
     257,   586,   232,   636,   588,   226,   676,   227,   228,   506,
     230,   614,   213,   341,   116,   151,   344,   302,   303,   304,
     443,   408,   149,   233,   657,   286,   656,   224,   616,   225,
     309,   278,   310,   281,   380,   416,   621,   286,   481,   667,
     557,   168,   261,   262,   263,   264,   256,   677,   712,   379,
      81,   507,   508,   509,   510,   511,   512,   513,   163,   215,
     721,   686,   637,   687,   151,   216,   688,   640,   217,   689,
     690,  -191,   638,  -191,   243,   664,   471,   731,    58,   283,
     315,   181,   514,    58,   170,   515,   711,   662,   357,   358,
     151,   666,   151,    58,   669,    59,   291,   290,   334,   292,
     372,   373,   374,   375,   376,   339,   151,    80,   679,    81,
     347,   275,   682,   425,   199,   684,   693,   175,   200,   196,
     653,   340,   696,   342,   343,   369,   345,   335,   178,   703,
     705,   687,   715,   708,   688,   257,   151,   689,   690,   183,
     294,   305,   710,   295,   207,   243,   382,   506,   384,   385,
     208,   151,   209,   490,   314,   205,   317,   491,   723,   214,
     321,   322,   323,   324,   325,   326,   506,   726,    58,   727,
      85,   229,   729,   201,   202,   203,   204,   734,   219,   407,
     735,    58,   410,   349,   336,   378,   393,     1,     2,   507,
     508,   509,   510,   511,   512,   513,   687,   419,   231,   688,
     717,   246,   689,   690,  -292,   506,   347,   248,   507,   508,
     509,   510,   511,   512,   513,   601,    58,   602,   377,   277,
     514,   250,   251,    85,  -292,    81,   367,   368,   260,  -292,
      81,   497,   498,   210,   437,   276,   440,   267,   268,   514,
     419,   279,    85,  -325,   604,   282,   293,   507,   508,   509,
     510,   511,   512,   513,   476,   444,   445,   446,   419,   143,
     433,   284,  -292,   639,   285,   297,   298,   300,   144,   145,
     205,   307,   306,   650,   241,   131,   156,   312,   514,   313,
     308,    85,   585,   311,   318,   327,   333,  -292,   319,   328,
     330,   196,   331,    81,   332,   474,   355,   286,   380,   133,
     134,   135,   136,   137,   138,   139,   435,   605,   606,   365,
     315,   683,   387,   388,   390,   352,   353,   354,   439,   506,
     391,   499,   709,   396,   687,   607,   392,   688,   394,   469,
     689,   690,   495,   540,   395,   409,   475,   360,   397,   423,
     398,   424,   426,    88,    89,    90,    91,    92,   432,   436,
     442,   516,   441,   472,   486,    99,   100,   523,   482,   101,
     483,   507,   508,   509,   510,   511,   512,   513,   583,   493,
     405,   406,   480,   484,   589,   485,   558,   488,   567,   398,
     489,   494,   -11,   598,   501,   411,   412,   413,   414,   415,
     503,   477,   514,   608,   496,    85,  -327,   719,   524,   687,
     522,   516,   688,   528,   526,   689,   690,   622,   506,   624,
     567,   525,   627,   530,   534,  -208,   533,   537,   612,   506,
     541,   544,   546,   503,   549,   553,    85,   576,   634,   506,
     196,   571,   196,   574,   579,   580,   626,   592,   567,   581,
     587,   594,   647,   597,   603,   599,   618,   623,   652,   608,
     507,   508,   509,   510,   511,   512,   513,   631,   632,   663,
     633,   507,   508,   509,   510,   511,   512,   513,   635,   673,
     641,   507,   508,   509,   510,   511,   512,   513,   196,   643,
     681,   514,   644,   646,   554,   506,   649,   658,   196,   651,
     659,   660,   514,   692,   674,    85,   699,   680,   697,   506,
     700,   698,   514,   701,   702,   615,   724,   630,   185,   186,
     187,   188,   189,   190,   706,   720,   713,   716,   730,   732,
     736,   506,    57,   103,    62,   266,   196,   507,   508,   509,
     510,   511,   512,   513,   694,   198,   222,   359,   346,   356,
     728,   507,   508,   509,   510,   511,   512,   513,   249,   479,
     670,   672,   131,   156,   417,   675,   427,   111,   514,   113,
     370,   620,   296,   507,   508,   509,   510,   511,   512,   513,
      81,   371,   514,   670,   434,   661,   133,   134,   135,   136,
     137,   138,   139,   470,   655,   431,   670,   670,   545,   707,
     670,   383,   575,   625,   514,   552,   428,   665,  -251,  -251,
    -251,   595,  -251,  -251,  -251,   718,  -251,  -251,  -251,  -251,
    -251,   584,   617,     0,  -251,  -251,  -251,  -251,  -251,  -251,
    -251,  -251,  -251,  -251,  -251,  -251,     0,  -251,  -251,  -251,
    -251,  -251,  -251,  -251,  -251,  -251,  -251,  -251,  -251,  -251,
    -251,  -251,  -251,  -251,  -251,  -251,     0,  -251,     0,  -251,
    -251,     0,     0,     0,     0,     0,  -251,  -251,  -251,  -251,
    -251,  -251,  -251,  -251,   506,     0,  -251,  -251,  -251,  -251,
       0,   506,     0,     0,     0,     0,     0,     0,   506,     0,
      63,   429,    -5,    -5,    64,    -5,    -5,    -5,    -5,    -5,
      -5,    -5,    -5,    -5,    -5,    -5,     0,    -5,    -5,     0,
       0,    -5,     0,     0,   604,     0,   507,   508,   509,   510,
     511,   512,   513,   507,   508,   509,   510,   511,   512,   513,
     507,   508,   509,   510,   511,   512,   513,   506,     0,     0,
       0,     0,    65,    66,   506,   131,   156,   514,    67,    68,
     668,     0,     0,     0,   514,     0,     0,   695,     0,     0,
      69,   514,     0,    81,   722,     0,    -5,   -65,     0,   133,
     134,   135,   136,   137,   138,   139,     0,   605,   606,   507,
     508,   509,   510,   511,   512,   513,   507,   508,   509,   510,
     511,   512,   513,     1,     2,     0,    88,    89,    90,    91,
      92,    93,    94,    95,    96,    97,    98,   447,    99,   100,
     514,     0,   101,   725,     0,     0,     0,   514,     0,     0,
     733,     0,     0,     0,     0,     0,     0,   448,     0,   449,
     450,   451,   452,   453,   454,     0,     0,   455,   456,   457,
     458,   459,   460,     0,     0,   117,   118,   119,   120,     0,
     121,   122,   123,   124,   125,     0,     0,   461,   462,     0,
     447,     0,     0,     0,     0,     0,     0,   102,     0,     0,
       0,     0,     0,     0,   463,     0,     0,     0,   464,   465,
     448,   126,   449,   450,   451,   452,   453,   454,     0,     0,
     455,   456,   457,   458,   459,   460,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     461,   462,     0,     0,    58,     0,     0,   127,     6,     7,
       8,     0,     9,    10,    11,     0,    12,    13,    14,    15,
      16,   464,   465,     0,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,     0,    29,    30,    31,
      32,    33,   131,   132,    34,    35,    36,    37,    38,    39,
      40,    41,    42,    43,    44,    45,     0,    46,     0,    47,
     468,     0,     0,     0,     0,     0,   133,   134,   135,   136,
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
      44,    45,     0,    46,     0,    47,    48,   704,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    49,
       0,     0,    50,    51,    52,    53,     6,     7,     8,     0,
       9,    10,    11,     0,    12,    13,    14,    15,    16,     0,
       0,     0,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,     0,    29,    30,    31,    32,    33,
     155,     0,    34,    35,    36,    37,    38,    39,    40,    41,
      42,    43,    44,    45,     0,    46,     0,    47,    48,     0,
       0,   131,   156,   255,     0,     0,     0,     0,     0,     0,
       0,    49,     0,     0,    50,    51,    52,    53,     0,    81,
       0,     0,   131,   156,     0,   133,   134,   135,   136,   137,
     138,   139,     0,     0,     0,   131,   156,   417,     0,     0,
      81,     0,     0,     0,     0,     0,   133,   134,   135,   136,
     137,   138,   139,    81,     0,     0,     0,     0,     0,   133,
     134,   135,   136,   137,   138,   139,   560,   561,   562,   510,
     563,   564,   565,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   418,   131,   156,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   566,   131,   132,
      85,    81,   269,   270,     0,     0,     0,   133,   134,   135,
     136,   137,   138,   139,   131,   156,    81,     0,     0,     0,
       0,     0,   133,   134,   135,   136,   137,   138,   139,   131,
       0,     0,    81,     0,     0,     0,     0,     0,   133,   134,
     135,   136,   137,   138,   139,     0,     0,    81,     0,     0,
       0,     0,     0,   133,   134,   135,   136,   137,   138,   139
};

static const yytype_int16 yycheck[] =
{
       1,     2,    92,    87,   260,    88,   417,    78,   559,    66,
      69,   183,   287,    54,    55,   236,    78,    78,   529,    17,
      58,    90,    91,    80,   469,   536,    17,   537,    69,    38,
     660,    72,   540,    17,    37,   569,    73,   176,   492,   669,
     509,    80,    38,   490,   513,    83,     0,    86,   493,    75,
     189,    94,    95,    96,   275,   172,   173,   174,   148,   569,
     335,   132,   179,   597,    65,   102,    67,    68,   579,   153,
     132,   132,   702,   624,    83,   116,   627,   522,   532,   533,
      78,   589,    38,    86,   155,   530,   539,   597,   542,    85,
     598,   560,    42,   562,    78,    86,   565,    88,    99,   546,
     183,   554,    30,   614,   557,   162,   657,   164,   165,     1,
     167,   580,   149,   291,    78,    76,   294,   238,   239,   240,
     423,   363,    77,    51,   635,    86,   634,    83,   582,    85,
     247,   215,   249,   217,    83,   377,   590,    86,   441,   647,
     594,   231,   185,   186,   187,   188,   318,   658,   699,   327,
      55,    43,    44,    45,    46,    47,    48,    49,    37,    77,
     711,    81,   615,    83,    76,    83,    86,   620,    86,    89,
      90,    79,   617,    81,   175,   644,   432,   728,    75,   220,
      88,    57,    74,    75,    86,    77,   697,   641,   309,   310,
      76,   645,    76,    75,   648,    77,    80,    83,   282,    83,
     321,   322,   323,   324,   325,   289,    76,    53,   661,    55,
     300,   212,   665,    83,    62,   668,   674,    86,    66,   260,
     631,   290,   676,   292,   293,   315,   295,   284,    86,   687,
     688,    83,    84,   691,    86,   318,    76,    89,    90,    80,
      80,   242,   695,    83,    82,   246,   330,     1,   332,   333,
      82,    76,    82,    79,   255,    80,   257,    83,   712,    76,
     261,   262,   263,   264,   265,   266,     1,   721,    75,   722,
      77,    68,   725,    61,    62,    63,    64,   731,    78,   362,
     733,    75,   365,    77,   285,   326,   343,     3,     4,    43,
      44,    45,    46,    47,    48,    49,    83,   380,    79,    86,
      87,    86,    89,    90,    58,     1,   396,    86,    43,    44,
      45,    46,    47,    48,    49,   571,    75,   573,    77,    55,
      74,    40,    41,    77,    78,    55,    56,    57,    78,    83,
      55,    56,    57,    38,   417,    78,   420,    61,    62,    74,
     423,    58,    77,    78,     6,    84,    87,    43,    44,    45,
      46,    47,    48,    49,   438,   424,   425,   426,   441,   430,
     401,    82,    58,   619,    82,    87,    79,    76,   430,   430,
      80,    79,    87,   629,    51,    37,    38,    79,    74,    77,
      87,    77,    78,    87,    79,    79,    87,    83,    81,    81,
      78,   432,    79,    55,    84,   436,    87,    86,    83,    61,
      62,    63,    64,    65,    66,    67,   407,    69,    70,    88,
      88,   667,    84,    87,    84,   302,   303,   304,   419,     1,
      81,   478,    81,    79,    83,    87,    84,    86,    81,   430,
      89,    90,   473,   517,    84,    78,   437,    17,     1,    83,
      42,    83,    83,     6,     7,     8,     9,    10,    78,    78,
      84,   492,    79,    86,    57,    18,    19,    76,    84,    22,
      84,    43,    44,    45,    46,    47,    48,    49,   552,   470,
     357,   358,    87,    84,   558,    84,   535,    87,   537,    42,
      87,    87,    83,   567,    37,   372,   373,   374,   375,   376,
     491,    86,    74,   576,    87,    77,    78,    81,    79,    83,
     501,   542,    86,    86,    83,    89,    90,   591,     1,   593,
     569,    84,   596,    55,    57,    78,    77,    77,   577,     1,
      20,    56,    79,   524,    55,    79,    77,    86,   612,     1,
     571,    78,   573,    88,    75,    87,   595,    86,   597,    84,
      78,    77,   626,    77,    56,    78,    78,    55,   631,   632,
      43,    44,    45,    46,    47,    48,    49,    76,    79,   643,
      87,    43,    44,    45,    46,    47,    48,    49,    75,   653,
      50,    43,    44,    45,    46,    47,    48,    49,   619,    75,
     664,    74,    87,    78,    77,     1,    78,    76,   629,    80,
      78,    78,    74,    87,    80,    77,   680,    75,    79,     1,
      76,    78,    74,    78,    78,    77,    79,   608,    11,    12,
      13,    14,    15,    16,    81,    78,   700,    87,    78,    78,
      78,     1,     5,    70,    55,   190,   667,    43,    44,    45,
      46,    47,    48,    49,   675,   132,   156,   312,   298,   307,
     724,    43,    44,    45,    46,    47,    48,    49,   177,   439,
     651,   652,    37,    38,    39,   656,   396,    70,    74,    70,
     317,    77,   231,    43,    44,    45,    46,    47,    48,    49,
      55,   318,    74,   674,   402,    77,    61,    62,    63,    64,
      65,    66,    67,   430,   632,   399,   687,   688,   524,   690,
     691,   331,   542,   594,    74,   530,     1,    77,     3,     4,
       5,   564,     7,     8,     9,   706,    11,    12,    13,    14,
      15,   553,   583,    -1,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    -1,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    -1,    52,    -1,    54,
      55,    -1,    -1,    -1,    -1,    -1,    61,    62,    63,    64,
      65,    66,    67,    68,     1,    -1,    71,    72,    73,    74,
      -1,     1,    -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,
       1,    86,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    -1,    18,    19,    -1,
      -1,    22,    -1,    -1,     6,    -1,    43,    44,    45,    46,
      47,    48,    49,    43,    44,    45,    46,    47,    48,    49,
      43,    44,    45,    46,    47,    48,    49,     1,    -1,    -1,
      -1,    -1,    53,    54,     1,    37,    38,    74,    59,    60,
      77,    -1,    -1,    -1,    74,    -1,    -1,    77,    -1,    -1,
      71,    74,    -1,    55,    77,    -1,    77,    78,    -1,    61,
      62,    63,    64,    65,    66,    67,    -1,    69,    70,    43,
      44,    45,    46,    47,    48,    49,    43,    44,    45,    46,
      47,    48,    49,     3,     4,    -1,     6,     7,     8,     9,
      10,    11,    12,    13,    14,    15,    16,     1,    18,    19,
      74,    -1,    22,    77,    -1,    -1,    -1,    74,    -1,    -1,
      77,    -1,    -1,    -1,    -1,    -1,    -1,    21,    -1,    23,
      24,    25,    26,    27,    28,    -1,    -1,    31,    32,    33,
      34,    35,    36,    -1,    -1,     6,     7,     8,     9,    -1,
      11,    12,    13,    14,    15,    -1,    -1,    51,    52,    -1,
       1,    -1,    -1,    -1,    -1,    -1,    -1,    77,    -1,    -1,
      -1,    -1,    -1,    -1,    68,    -1,    -1,    -1,    72,    73,
      21,    42,    23,    24,    25,    26,    27,    28,    -1,    -1,
      31,    32,    33,    34,    35,    36,    -1,    -1,    -1,    -1,
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
      27,    28,    29,    30,    -1,    32,    33,    34,    35,    36,
      16,    -1,    39,    40,    41,    42,    43,    44,    45,    46,
      47,    48,    49,    50,    -1,    52,    -1,    54,    55,    -1,
      -1,    37,    38,    18,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    68,    -1,    -1,    71,    72,    73,    74,    -1,    55,
      -1,    -1,    37,    38,    -1,    61,    62,    63,    64,    65,
      66,    67,    -1,    -1,    -1,    37,    38,    39,    -1,    -1,
      55,    -1,    -1,    -1,    -1,    -1,    61,    62,    63,    64,
      65,    66,    67,    55,    -1,    -1,    -1,    -1,    -1,    61,
      62,    63,    64,    65,    66,    67,    43,    44,    45,    46,
      47,    48,    49,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    84,    37,    38,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    74,    37,    38,
      77,    55,    56,    57,    -1,    -1,    -1,    61,    62,    63,
      64,    65,    66,    67,    37,    38,    55,    -1,    -1,    -1,
      -1,    -1,    61,    62,    63,    64,    65,    66,    67,    37,
      -1,    -1,    55,    -1,    -1,    -1,    -1,    -1,    61,    62,
      63,    64,    65,    66,    67,    -1,    -1,    55,    -1,    -1,
      -1,    -1,    -1,    61,    62,    63,    64,    65,    66,    67
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
      94,   101,   102,   103,   169,   208,   209,    75,    42,    97,
      53,    55,    98,    97,    97,    77,    95,   180,     6,     7,
       8,     9,    10,    11,    12,    13,    14,    15,    16,    18,
      19,    22,    77,    99,   122,   123,   139,   142,   143,   144,
     146,   156,   157,   160,   161,   162,    78,     6,     7,     8,
       9,    11,    12,    13,    14,    15,    42,    78,    95,   167,
     101,    37,    38,    61,    62,    63,    64,    65,    66,    67,
      98,   108,   110,   111,   112,   113,   116,   117,   170,    77,
      98,    76,    58,    83,   178,    16,    38,   111,   112,   113,
     114,   115,   118,    37,   124,   124,    86,   124,   110,   163,
      86,   128,   128,   128,   128,    86,   132,   145,    86,   125,
      97,    57,   164,    80,   101,    11,    12,    13,    14,    15,
      16,   147,   148,   149,   150,   151,    95,    96,   116,    62,
      66,    61,    62,    63,    64,    80,   107,    82,    82,    82,
      38,    85,   110,   101,    76,    77,    83,    86,   178,    78,
     181,   111,   115,    38,    83,    85,    98,    98,    98,    68,
      98,    79,    30,    51,   129,   134,    97,   109,   109,   109,
     109,    51,    56,    97,   131,   133,    86,   145,    86,   132,
      40,    41,   126,   127,   109,    18,   114,   118,   154,   155,
      78,   128,   128,   128,   128,   145,   125,    61,    62,    56,
      57,   104,   105,   106,   118,    97,    78,    55,   178,    58,
     177,   178,    84,    95,    82,    82,    86,   120,   121,   206,
      83,    80,    83,    87,    80,    83,   163,    87,    79,   107,
      76,   140,   140,   140,   140,    97,    87,    79,    87,   109,
     109,    87,    79,    77,    97,    88,   153,    97,    79,    81,
      96,    97,    97,    97,    97,    97,    97,    79,    81,   107,
      78,    79,    84,    87,   178,    98,    97,   121,   119,   178,
     124,   105,   124,   124,   105,   124,   129,   110,   141,    77,
      95,   158,   158,   158,   158,    87,   133,   140,   140,   126,
      17,    78,   135,   137,   138,    88,   152,    56,    57,   110,
     153,   155,   140,   140,   140,   140,   140,    77,    95,   105,
      83,   189,   178,   177,   178,   178,   121,    84,    87,   207,
      84,    81,    84,    98,    81,    84,    79,     1,    42,   156,
     159,   160,   165,   166,   168,   158,   158,   118,   138,    78,
     118,   158,   158,   158,   158,   158,   138,    39,    84,   118,
     179,   182,   187,    83,    83,    83,    83,   141,     1,    86,
     171,   168,    78,    95,   159,    97,    78,   118,   179,    97,
     178,    79,    84,   187,   124,   124,   124,     1,    21,    23,
      24,    25,    26,    27,    28,    31,    32,    33,    34,    35,
      36,    51,    52,    68,    72,    73,   174,   175,    55,    97,
     170,    96,    86,   136,    95,    97,   178,    86,    88,   135,
      87,   187,    84,    84,    84,    84,    57,   130,    87,    87,
      79,    83,   189,    97,    87,    95,    87,    56,    57,    98,
     176,    37,   174,    97,   172,   173,     1,    43,    44,    45,
      46,    47,    48,    49,    74,    77,    95,   180,   192,   197,
     199,   189,    97,    76,    79,    84,    83,   203,    86,   203,
      55,   204,   205,    77,    57,   196,   203,    77,   193,   199,
     178,    20,   191,   189,    56,   173,    79,   178,   201,    55,
     201,   189,   206,    79,    77,   199,   194,   199,   180,   201,
      43,    44,    45,    47,    48,    49,    74,   180,   195,   197,
     198,    78,   193,   181,    88,   192,    86,   190,   174,    75,
      87,    84,   202,   178,   205,    78,   193,    78,   193,   178,
     202,   203,    86,   203,    77,   196,   203,    77,   178,    78,
     195,    96,    96,    56,     6,    69,    70,    87,   118,   183,
     186,   188,   180,   201,   203,    77,   199,   207,    78,   181,
      77,   199,   178,    55,   178,   194,   180,   178,   195,   181,
      97,    76,    79,    87,   178,    75,   201,   193,   189,    96,
     193,    50,   200,    75,    87,   202,    78,   178,   202,    78,
      96,    80,   118,   179,   185,   188,   181,   201,    76,    78,
      78,    77,   199,   178,   203,    77,   199,   181,    77,   199,
      97,   184,    97,   178,    80,    97,   202,   201,   200,   193,
      75,   178,   193,    96,   193,   200,    81,    83,    86,    89,
      90,    80,    87,   184,    95,    77,   199,    79,    78,   178,
      76,    78,    78,   184,    56,   184,    81,    97,   184,    81,
     193,   201,   202,   178,   200,    84,    87,    87,    97,    81,
      78,   202,    77,   199,    79,    77,   199,   193,   178,   193,
      78,   202,    78,    77,   199,   193,    78
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
     117,   118,   118,   119,   120,   121,   121,   122,   123,   124,
     124,   125,   125,   126,   126,   127,   127,   128,   128,   129,
     129,   130,   130,   131,   132,   132,   133,   133,   134,   134,
     135,   135,   136,   136,   137,   138,   138,   139,   139,   139,
     140,   140,   141,   141,   142,   142,   143,   144,   145,   145,
     146,   146,   147,   147,   148,   149,   150,   151,   151,   152,
     152,   153,   153,   153,   153,   154,   154,   154,   155,   155,
     156,   157,   157,   157,   157,   157,   158,   158,   159,   159,
     160,   160,   160,   160,   160,   160,   160,   161,   161,   161,
     161,   161,   162,   162,   162,   162,   163,   163,   164,   165,
     166,   166,   166,   166,   167,   167,   167,   167,   167,   167,
     167,   167,   167,   167,   167,   168,   168,   168,   169,   169,
     170,   171,   171,   171,   172,   173,   173,   174,   174,   174,
     174,   175,   175,   175,   175,   175,   175,   175,   175,   175,
     175,   175,   175,   175,   175,   175,   175,   175,   175,   176,
     176,   176,   177,   177,   177,   178,   178,   178,   178,   178,
     178,   179,   180,   181,   182,   182,   182,   182,   182,   183,
     183,   183,   184,   184,   184,   184,   184,   184,   185,   186,
     186,   186,   187,   187,   188,   188,   189,   189,   190,   190,
     191,   191,   192,   192,   192,   193,   193,   194,   194,   195,
     195,   195,   196,   196,   197,   197,   197,   198,   198,   198,
     198,   198,   198,   198,   198,   198,   198,   198,   198,   199,
     199,   199,   199,   199,   199,   199,   199,   199,   199,   199,
     199,   199,   199,   200,   200,   200,   201,   202,   203,   204,
     204,   205,   205,   206,   207,   208,   209
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
       1,     1,     2,     2,     1,     1,     1,     2,     2,     2,
       1,     2,     1,     1,     3,     0,     2,     4,     6,     0,
       1,     0,     3,     1,     3,     1,     1,     0,     3,     1,
       3,     0,     1,     1,     0,     3,     1,     3,     1,     1,
       0,     1,     0,     2,     5,     1,     2,     3,     5,     6,
       0,     2,     1,     3,     5,     5,     5,     5,     4,     3,
       6,     6,     5,     5,     5,     5,     5,     4,     7,     0,
       2,     0,     2,     2,     2,     3,     2,     3,     1,     3,
       4,     2,     2,     2,     2,     2,     1,     4,     0,     2,
       1,     1,     1,     1,     2,     2,     2,     3,     6,     9,
       3,     6,     3,     6,     9,     9,     1,     3,     1,     1,
       1,     2,     2,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     7,     5,    13,     5,     2,
       1,     0,     3,     1,     3,     1,     3,     1,     4,     3,
       6,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     2,     1,     1,     1,     1,     1,
       1,     1,     0,     1,     3,     0,     1,     5,     5,     5,
       4,     3,     1,     1,     1,     3,     4,     3,     4,     1,
       1,     1,     1,     4,     3,     4,     4,     4,     3,     7,
       5,     6,     1,     3,     1,     3,     3,     2,     3,     2,
       0,     3,     1,     1,     4,     1,     2,     1,     2,     1,
       2,     1,     1,     0,     4,     3,     5,     6,     4,     4,
      11,     9,    12,    14,     6,     8,     5,     7,     4,     6,
       4,     1,     4,    11,     9,    12,    14,     6,     8,     5,
       7,     4,     1,     0,     2,     4,     1,     1,     1,     2,
       5,     1,     3,     1,     1,     2,     2
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
#line 200 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.modlist) = (yyvsp[0].modlist); modlist = (yyvsp[0].modlist); }
#line 2244 "y.tab.c" /* yacc.c:1661  */
    break;

  case 3:
#line 204 "xi-grammar.y" /* yacc.c:1661  */
    { 
		  (yyval.modlist) = 0; 
		}
#line 2252 "y.tab.c" /* yacc.c:1661  */
    break;

  case 4:
#line 208 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.modlist) = new AstChildren<Module>(lineno, (yyvsp[-1].module), (yyvsp[0].modlist)); }
#line 2258 "y.tab.c" /* yacc.c:1661  */
    break;

  case 5:
#line 212 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 0; }
#line 2264 "y.tab.c" /* yacc.c:1661  */
    break;

  case 6:
#line 214 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 1; }
#line 2270 "y.tab.c" /* yacc.c:1661  */
    break;

  case 7:
#line 218 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 1; }
#line 2276 "y.tab.c" /* yacc.c:1661  */
    break;

  case 8:
#line 220 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 2; }
#line 2282 "y.tab.c" /* yacc.c:1661  */
    break;

  case 9:
#line 224 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 0; }
#line 2288 "y.tab.c" /* yacc.c:1661  */
    break;

  case 10:
#line 226 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 1; }
#line 2294 "y.tab.c" /* yacc.c:1661  */
    break;

  case 11:
#line 231 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 2300 "y.tab.c" /* yacc.c:1661  */
    break;

  case 12:
#line 232 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(MODULE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2306 "y.tab.c" /* yacc.c:1661  */
    break;

  case 13:
#line 233 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(MAINMODULE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2312 "y.tab.c" /* yacc.c:1661  */
    break;

  case 14:
#line 234 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(EXTERN, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2318 "y.tab.c" /* yacc.c:1661  */
    break;

  case 15:
#line 236 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(INITCALL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2324 "y.tab.c" /* yacc.c:1661  */
    break;

  case 16:
#line 237 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(INITNODE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2330 "y.tab.c" /* yacc.c:1661  */
    break;

  case 17:
#line 238 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(INITPROC, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2336 "y.tab.c" /* yacc.c:1661  */
    break;

  case 18:
#line 240 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(CHARE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2342 "y.tab.c" /* yacc.c:1661  */
    break;

  case 19:
#line 241 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(MAINCHARE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2348 "y.tab.c" /* yacc.c:1661  */
    break;

  case 20:
#line 242 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(GROUP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2354 "y.tab.c" /* yacc.c:1661  */
    break;

  case 21:
#line 243 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(NODEGROUP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2360 "y.tab.c" /* yacc.c:1661  */
    break;

  case 22:
#line 244 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(ARRAY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2366 "y.tab.c" /* yacc.c:1661  */
    break;

  case 23:
#line 248 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(INCLUDE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2372 "y.tab.c" /* yacc.c:1661  */
    break;

  case 24:
#line 249 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(STACKSIZE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2378 "y.tab.c" /* yacc.c:1661  */
    break;

  case 25:
#line 250 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(THREADED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2384 "y.tab.c" /* yacc.c:1661  */
    break;

  case 26:
#line 251 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(TEMPLATE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2390 "y.tab.c" /* yacc.c:1661  */
    break;

  case 27:
#line 252 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(SYNC, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2396 "y.tab.c" /* yacc.c:1661  */
    break;

  case 28:
#line 253 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(IGET, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2402 "y.tab.c" /* yacc.c:1661  */
    break;

  case 29:
#line 254 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(EXCLUSIVE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2408 "y.tab.c" /* yacc.c:1661  */
    break;

  case 30:
#line 255 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(IMMEDIATE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2414 "y.tab.c" /* yacc.c:1661  */
    break;

  case 31:
#line 256 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(SKIPSCHED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2420 "y.tab.c" /* yacc.c:1661  */
    break;

  case 32:
#line 257 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(NOCOPY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2426 "y.tab.c" /* yacc.c:1661  */
    break;

  case 33:
#line 258 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(INLINE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2432 "y.tab.c" /* yacc.c:1661  */
    break;

  case 34:
#line 259 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(VIRTUAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2438 "y.tab.c" /* yacc.c:1661  */
    break;

  case 35:
#line 260 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(MIGRATABLE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2444 "y.tab.c" /* yacc.c:1661  */
    break;

  case 36:
#line 261 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(CREATEHERE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2450 "y.tab.c" /* yacc.c:1661  */
    break;

  case 37:
#line 262 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(CREATEHOME, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2456 "y.tab.c" /* yacc.c:1661  */
    break;

  case 38:
#line 263 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(NOKEEP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2462 "y.tab.c" /* yacc.c:1661  */
    break;

  case 39:
#line 264 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(NOTRACE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2468 "y.tab.c" /* yacc.c:1661  */
    break;

  case 40:
#line 265 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(APPWORK, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2474 "y.tab.c" /* yacc.c:1661  */
    break;

  case 41:
#line 268 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(PACKED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2480 "y.tab.c" /* yacc.c:1661  */
    break;

  case 42:
#line 269 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(VARSIZE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2486 "y.tab.c" /* yacc.c:1661  */
    break;

  case 43:
#line 270 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(ENTRY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2492 "y.tab.c" /* yacc.c:1661  */
    break;

  case 44:
#line 271 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(FOR, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2498 "y.tab.c" /* yacc.c:1661  */
    break;

  case 45:
#line 272 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(FORALL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2504 "y.tab.c" /* yacc.c:1661  */
    break;

  case 46:
#line 273 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(WHILE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2510 "y.tab.c" /* yacc.c:1661  */
    break;

  case 47:
#line 274 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(WHEN, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2516 "y.tab.c" /* yacc.c:1661  */
    break;

  case 48:
#line 275 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(OVERLAP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2522 "y.tab.c" /* yacc.c:1661  */
    break;

  case 49:
#line 276 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(SERIAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2528 "y.tab.c" /* yacc.c:1661  */
    break;

  case 50:
#line 277 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(IF, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2534 "y.tab.c" /* yacc.c:1661  */
    break;

  case 51:
#line 278 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(ELSE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2540 "y.tab.c" /* yacc.c:1661  */
    break;

  case 52:
#line 280 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(LOCAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2546 "y.tab.c" /* yacc.c:1661  */
    break;

  case 53:
#line 282 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(USING, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2552 "y.tab.c" /* yacc.c:1661  */
    break;

  case 54:
#line 283 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(ACCEL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2558 "y.tab.c" /* yacc.c:1661  */
    break;

  case 55:
#line 286 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(ACCELBLOCK, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2564 "y.tab.c" /* yacc.c:1661  */
    break;

  case 56:
#line 287 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(MEMCRITICAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2570 "y.tab.c" /* yacc.c:1661  */
    break;

  case 57:
#line 288 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(REDUCTIONTARGET, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2576 "y.tab.c" /* yacc.c:1661  */
    break;

  case 58:
#line 289 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(CASE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2582 "y.tab.c" /* yacc.c:1661  */
    break;

  case 59:
#line 294 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 2588 "y.tab.c" /* yacc.c:1661  */
    break;

  case 60:
#line 296 "xi-grammar.y" /* yacc.c:1661  */
    {
		  char *tmp = new char[strlen((yyvsp[-3].strval))+strlen((yyvsp[0].strval))+3];
		  sprintf(tmp,"%s::%s", (yyvsp[-3].strval), (yyvsp[0].strval));
		  (yyval.strval) = tmp;
		}
#line 2598 "y.tab.c" /* yacc.c:1661  */
    break;

  case 61:
#line 303 "xi-grammar.y" /* yacc.c:1661  */
    { 
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist)); 
		}
#line 2606 "y.tab.c" /* yacc.c:1661  */
    break;

  case 62:
#line 307 "xi-grammar.y" /* yacc.c:1661  */
    {  
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist)); 
		    (yyval.module)->setMain();
		}
#line 2615 "y.tab.c" /* yacc.c:1661  */
    break;

  case 63:
#line 314 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.conslist) = 0; }
#line 2621 "y.tab.c" /* yacc.c:1661  */
    break;

  case 64:
#line 316 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.conslist) = (yyvsp[-2].conslist); }
#line 2627 "y.tab.c" /* yacc.c:1661  */
    break;

  case 65:
#line 320 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.conslist) = 0; }
#line 2633 "y.tab.c" /* yacc.c:1661  */
    break;

  case 66:
#line 322 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.conslist) = new ConstructList(lineno, (yyvsp[-1].construct), (yyvsp[0].conslist)); }
#line 2639 "y.tab.c" /* yacc.c:1661  */
    break;

  case 67:
#line 326 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.construct) = new UsingScope((yyvsp[0].strval), false); }
#line 2645 "y.tab.c" /* yacc.c:1661  */
    break;

  case 68:
#line 328 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.construct) = new UsingScope((yyvsp[0].strval), true); }
#line 2651 "y.tab.c" /* yacc.c:1661  */
    break;

  case 69:
#line 330 "xi-grammar.y" /* yacc.c:1661  */
    { (yyvsp[0].member)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].member); }
#line 2657 "y.tab.c" /* yacc.c:1661  */
    break;

  case 70:
#line 332 "xi-grammar.y" /* yacc.c:1661  */
    { (yyvsp[0].message)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].message); }
#line 2663 "y.tab.c" /* yacc.c:1661  */
    break;

  case 71:
#line 334 "xi-grammar.y" /* yacc.c:1661  */
    {
                  Entry *e = new Entry(lineno, NULL, (yyvsp[-4].type), (yyvsp[-2].strval), (yyvsp[0].plist), 0, 0, 0, (yylsp[-6]).first_line, (yyloc).last_line);
                  int isExtern = 1;
                  e->setExtern(isExtern);
                  e->targs = (yyvsp[-1].tparlist);
                  e->label = new XStr;
                  (yyvsp[-3].ntype)->print(*e->label);
                  (yyval.construct) = e;
                  firstRdma = true;
                }
#line 2678 "y.tab.c" /* yacc.c:1661  */
    break;

  case 72:
#line 347 "xi-grammar.y" /* yacc.c:1661  */
    { if((yyvsp[-2].conslist)) (yyvsp[-2].conslist)->recurse<int&>((yyvsp[-4].intval), &Construct::setExtern); (yyval.construct) = (yyvsp[-2].conslist); }
#line 2684 "y.tab.c" /* yacc.c:1661  */
    break;

  case 73:
#line 349 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.construct) = new Scope((yyvsp[-3].strval), (yyvsp[-1].conslist)); }
#line 2690 "y.tab.c" /* yacc.c:1661  */
    break;

  case 74:
#line 351 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.construct) = (yyvsp[-1].construct); }
#line 2696 "y.tab.c" /* yacc.c:1661  */
    break;

  case 75:
#line 353 "xi-grammar.y" /* yacc.c:1661  */
    {
          ERROR("preceding construct must be semicolon terminated",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 2706 "y.tab.c" /* yacc.c:1661  */
    break;

  case 76:
#line 359 "xi-grammar.y" /* yacc.c:1661  */
    { (yyvsp[0].module)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].module); }
#line 2712 "y.tab.c" /* yacc.c:1661  */
    break;

  case 77:
#line 361 "xi-grammar.y" /* yacc.c:1661  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2718 "y.tab.c" /* yacc.c:1661  */
    break;

  case 78:
#line 363 "xi-grammar.y" /* yacc.c:1661  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2724 "y.tab.c" /* yacc.c:1661  */
    break;

  case 79:
#line 365 "xi-grammar.y" /* yacc.c:1661  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2730 "y.tab.c" /* yacc.c:1661  */
    break;

  case 80:
#line 367 "xi-grammar.y" /* yacc.c:1661  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2736 "y.tab.c" /* yacc.c:1661  */
    break;

  case 81:
#line 369 "xi-grammar.y" /* yacc.c:1661  */
    { (yyvsp[0].templat)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].templat); }
#line 2742 "y.tab.c" /* yacc.c:1661  */
    break;

  case 82:
#line 371 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.construct) = NULL; }
#line 2748 "y.tab.c" /* yacc.c:1661  */
    break;

  case 83:
#line 373 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.construct) = NULL; }
#line 2754 "y.tab.c" /* yacc.c:1661  */
    break;

  case 84:
#line 375 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.construct) = (yyvsp[0].accelBlock); }
#line 2760 "y.tab.c" /* yacc.c:1661  */
    break;

  case 85:
#line 377 "xi-grammar.y" /* yacc.c:1661  */
    {
          ERROR("invalid construct",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 2770 "y.tab.c" /* yacc.c:1661  */
    break;

  case 86:
#line 385 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tparam) = new TParamType((yyvsp[0].type)); }
#line 2776 "y.tab.c" /* yacc.c:1661  */
    break;

  case 87:
#line 387 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
#line 2782 "y.tab.c" /* yacc.c:1661  */
    break;

  case 88:
#line 389 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
#line 2788 "y.tab.c" /* yacc.c:1661  */
    break;

  case 89:
#line 393 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tparlist) = new TParamList((yyvsp[0].tparam)); }
#line 2794 "y.tab.c" /* yacc.c:1661  */
    break;

  case 90:
#line 395 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tparlist) = new TParamList((yyvsp[-2].tparam), (yyvsp[0].tparlist)); }
#line 2800 "y.tab.c" /* yacc.c:1661  */
    break;

  case 91:
#line 399 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tparlist) = 0; }
#line 2806 "y.tab.c" /* yacc.c:1661  */
    break;

  case 92:
#line 401 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tparlist) = (yyvsp[0].tparlist); }
#line 2812 "y.tab.c" /* yacc.c:1661  */
    break;

  case 93:
#line 405 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tparlist) = 0; }
#line 2818 "y.tab.c" /* yacc.c:1661  */
    break;

  case 94:
#line 407 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tparlist) = (yyvsp[-1].tparlist); }
#line 2824 "y.tab.c" /* yacc.c:1661  */
    break;

  case 95:
#line 411 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("int"); }
#line 2830 "y.tab.c" /* yacc.c:1661  */
    break;

  case 96:
#line 413 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("long"); }
#line 2836 "y.tab.c" /* yacc.c:1661  */
    break;

  case 97:
#line 415 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("short"); }
#line 2842 "y.tab.c" /* yacc.c:1661  */
    break;

  case 98:
#line 417 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("char"); }
#line 2848 "y.tab.c" /* yacc.c:1661  */
    break;

  case 99:
#line 419 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("unsigned int"); }
#line 2854 "y.tab.c" /* yacc.c:1661  */
    break;

  case 100:
#line 421 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("unsigned long"); }
#line 2860 "y.tab.c" /* yacc.c:1661  */
    break;

  case 101:
#line 423 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("unsigned long"); }
#line 2866 "y.tab.c" /* yacc.c:1661  */
    break;

  case 102:
#line 425 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("unsigned long long"); }
#line 2872 "y.tab.c" /* yacc.c:1661  */
    break;

  case 103:
#line 427 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("unsigned short"); }
#line 2878 "y.tab.c" /* yacc.c:1661  */
    break;

  case 104:
#line 429 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("unsigned char"); }
#line 2884 "y.tab.c" /* yacc.c:1661  */
    break;

  case 105:
#line 431 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("long long"); }
#line 2890 "y.tab.c" /* yacc.c:1661  */
    break;

  case 106:
#line 433 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("float"); }
#line 2896 "y.tab.c" /* yacc.c:1661  */
    break;

  case 107:
#line 435 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("double"); }
#line 2902 "y.tab.c" /* yacc.c:1661  */
    break;

  case 108:
#line 437 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("long double"); }
#line 2908 "y.tab.c" /* yacc.c:1661  */
    break;

  case 109:
#line 439 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("void"); }
#line 2914 "y.tab.c" /* yacc.c:1661  */
    break;

  case 110:
#line 442 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.ntype) = new NamedType((yyvsp[-1].strval),(yyvsp[0].tparlist)); }
#line 2920 "y.tab.c" /* yacc.c:1661  */
    break;

  case 111:
#line 443 "xi-grammar.y" /* yacc.c:1661  */
    { 
                    const char* basename, *scope;
                    splitScopedName((yyvsp[-1].strval), &scope, &basename);
                    (yyval.ntype) = new NamedType(basename, (yyvsp[0].tparlist), scope);
                }
#line 2930 "y.tab.c" /* yacc.c:1661  */
    break;

  case 112:
#line 451 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].type); }
#line 2936 "y.tab.c" /* yacc.c:1661  */
    break;

  case 113:
#line 453 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].ntype); }
#line 2942 "y.tab.c" /* yacc.c:1661  */
    break;

  case 114:
#line 457 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.ptype) = new PtrType((yyvsp[-1].type)); }
#line 2948 "y.tab.c" /* yacc.c:1661  */
    break;

  case 115:
#line 461 "xi-grammar.y" /* yacc.c:1661  */
    { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
#line 2954 "y.tab.c" /* yacc.c:1661  */
    break;

  case 116:
#line 463 "xi-grammar.y" /* yacc.c:1661  */
    { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
#line 2960 "y.tab.c" /* yacc.c:1661  */
    break;

  case 117:
#line 467 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.ftype) = new FuncType((yyvsp[-7].type), (yyvsp[-4].strval), (yyvsp[-1].plist)); }
#line 2966 "y.tab.c" /* yacc.c:1661  */
    break;

  case 118:
#line 471 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].type); }
#line 2972 "y.tab.c" /* yacc.c:1661  */
    break;

  case 119:
#line 473 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 2978 "y.tab.c" /* yacc.c:1661  */
    break;

  case 120:
#line 475 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 2984 "y.tab.c" /* yacc.c:1661  */
    break;

  case 121:
#line 477 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].ftype); }
#line 2990 "y.tab.c" /* yacc.c:1661  */
    break;

  case 122:
#line 479 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new ConstType((yyvsp[0].type)); }
#line 2996 "y.tab.c" /* yacc.c:1661  */
    break;

  case 123:
#line 481 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new ConstType((yyvsp[-1].type)); }
#line 3002 "y.tab.c" /* yacc.c:1661  */
    break;

  case 124:
#line 485 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3008 "y.tab.c" /* yacc.c:1661  */
    break;

  case 125:
#line 487 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3014 "y.tab.c" /* yacc.c:1661  */
    break;

  case 126:
#line 489 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3020 "y.tab.c" /* yacc.c:1661  */
    break;

  case 127:
#line 491 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new ConstType((yyvsp[0].type)); }
#line 3026 "y.tab.c" /* yacc.c:1661  */
    break;

  case 128:
#line 493 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new ConstType((yyvsp[-1].type)); }
#line 3032 "y.tab.c" /* yacc.c:1661  */
    break;

  case 129:
#line 497 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new ReferenceType((yyvsp[-1].type)); }
#line 3038 "y.tab.c" /* yacc.c:1661  */
    break;

  case 130:
#line 499 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3044 "y.tab.c" /* yacc.c:1661  */
    break;

  case 131:
#line 503 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new ReferenceType((yyvsp[-1].type)); }
#line 3050 "y.tab.c" /* yacc.c:1661  */
    break;

  case 132:
#line 505 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3056 "y.tab.c" /* yacc.c:1661  */
    break;

  case 133:
#line 509 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 3062 "y.tab.c" /* yacc.c:1661  */
    break;

  case 134:
#line 513 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.val) = (yyvsp[-1].val); }
#line 3068 "y.tab.c" /* yacc.c:1661  */
    break;

  case 135:
#line 517 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.vallist) = 0; }
#line 3074 "y.tab.c" /* yacc.c:1661  */
    break;

  case 136:
#line 519 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.vallist) = new ValueList((yyvsp[-1].val), (yyvsp[0].vallist)); }
#line 3080 "y.tab.c" /* yacc.c:1661  */
    break;

  case 137:
#line 523 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].vallist)); }
#line 3086 "y.tab.c" /* yacc.c:1661  */
    break;

  case 138:
#line 527 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[-3].type), (yyvsp[-1].strval), (yyvsp[0].vallist), 1); }
#line 3092 "y.tab.c" /* yacc.c:1661  */
    break;

  case 139:
#line 531 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 0;}
#line 3098 "y.tab.c" /* yacc.c:1661  */
    break;

  case 140:
#line 533 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 0;}
#line 3104 "y.tab.c" /* yacc.c:1661  */
    break;

  case 141:
#line 537 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 0; }
#line 3110 "y.tab.c" /* yacc.c:1661  */
    break;

  case 142:
#line 539 "xi-grammar.y" /* yacc.c:1661  */
    { 
		  /*
		  printf("Warning: Message attributes are being phased out.\n");
		  printf("Warning: Please remove them from interface files.\n");
		  */
		  (yyval.intval) = (yyvsp[-1].intval); 
		}
#line 3122 "y.tab.c" /* yacc.c:1661  */
    break;

  case 143:
#line 549 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = (yyvsp[0].intval); }
#line 3128 "y.tab.c" /* yacc.c:1661  */
    break;

  case 144:
#line 551 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = (yyvsp[-2].intval) | (yyvsp[0].intval); }
#line 3134 "y.tab.c" /* yacc.c:1661  */
    break;

  case 145:
#line 555 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 0; }
#line 3140 "y.tab.c" /* yacc.c:1661  */
    break;

  case 146:
#line 557 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 0; }
#line 3146 "y.tab.c" /* yacc.c:1661  */
    break;

  case 147:
#line 561 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = 0; }
#line 3152 "y.tab.c" /* yacc.c:1661  */
    break;

  case 148:
#line 563 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = (yyvsp[-1].cattr); }
#line 3158 "y.tab.c" /* yacc.c:1661  */
    break;

  case 149:
#line 567 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = (yyvsp[0].cattr); }
#line 3164 "y.tab.c" /* yacc.c:1661  */
    break;

  case 150:
#line 569 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
#line 3170 "y.tab.c" /* yacc.c:1661  */
    break;

  case 151:
#line 573 "xi-grammar.y" /* yacc.c:1661  */
    { python_doc = NULL; (yyval.intval) = 0; }
#line 3176 "y.tab.c" /* yacc.c:1661  */
    break;

  case 152:
#line 575 "xi-grammar.y" /* yacc.c:1661  */
    { python_doc = (yyvsp[0].strval); (yyval.intval) = 0; }
#line 3182 "y.tab.c" /* yacc.c:1661  */
    break;

  case 153:
#line 579 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = Chare::CPYTHON; }
#line 3188 "y.tab.c" /* yacc.c:1661  */
    break;

  case 154:
#line 583 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = 0; }
#line 3194 "y.tab.c" /* yacc.c:1661  */
    break;

  case 155:
#line 585 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = (yyvsp[-1].cattr); }
#line 3200 "y.tab.c" /* yacc.c:1661  */
    break;

  case 156:
#line 589 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = (yyvsp[0].cattr); }
#line 3206 "y.tab.c" /* yacc.c:1661  */
    break;

  case 157:
#line 591 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
#line 3212 "y.tab.c" /* yacc.c:1661  */
    break;

  case 158:
#line 595 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = Chare::CMIGRATABLE; }
#line 3218 "y.tab.c" /* yacc.c:1661  */
    break;

  case 159:
#line 597 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = Chare::CPYTHON; }
#line 3224 "y.tab.c" /* yacc.c:1661  */
    break;

  case 160:
#line 601 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 0; }
#line 3230 "y.tab.c" /* yacc.c:1661  */
    break;

  case 161:
#line 603 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 1; }
#line 3236 "y.tab.c" /* yacc.c:1661  */
    break;

  case 162:
#line 606 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 0; }
#line 3242 "y.tab.c" /* yacc.c:1661  */
    break;

  case 163:
#line 608 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 1; }
#line 3248 "y.tab.c" /* yacc.c:1661  */
    break;

  case 164:
#line 611 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.mv) = new MsgVar((yyvsp[-3].type), (yyvsp[-2].strval), (yyvsp[-4].intval), (yyvsp[-1].intval)); }
#line 3254 "y.tab.c" /* yacc.c:1661  */
    break;

  case 165:
#line 615 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.mvlist) = new MsgVarList((yyvsp[0].mv)); }
#line 3260 "y.tab.c" /* yacc.c:1661  */
    break;

  case 166:
#line 617 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.mvlist) = new MsgVarList((yyvsp[-1].mv), (yyvsp[0].mvlist)); }
#line 3266 "y.tab.c" /* yacc.c:1661  */
    break;

  case 167:
#line 621 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.message) = new Message(lineno, (yyvsp[0].ntype)); }
#line 3272 "y.tab.c" /* yacc.c:1661  */
    break;

  case 168:
#line 623 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.message) = new Message(lineno, (yyvsp[-2].ntype)); }
#line 3278 "y.tab.c" /* yacc.c:1661  */
    break;

  case 169:
#line 625 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.message) = new Message(lineno, (yyvsp[-3].ntype), (yyvsp[-1].mvlist)); }
#line 3284 "y.tab.c" /* yacc.c:1661  */
    break;

  case 170:
#line 629 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.typelist) = 0; }
#line 3290 "y.tab.c" /* yacc.c:1661  */
    break;

  case 171:
#line 631 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.typelist) = (yyvsp[0].typelist); }
#line 3296 "y.tab.c" /* yacc.c:1661  */
    break;

  case 172:
#line 635 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.typelist) = new TypeList((yyvsp[0].ntype)); }
#line 3302 "y.tab.c" /* yacc.c:1661  */
    break;

  case 173:
#line 637 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.typelist) = new TypeList((yyvsp[-2].ntype), (yyvsp[0].typelist)); }
#line 3308 "y.tab.c" /* yacc.c:1661  */
    break;

  case 174:
#line 641 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr)|Chare::CCHARE, (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3314 "y.tab.c" /* yacc.c:1661  */
    break;

  case 175:
#line 643 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3320 "y.tab.c" /* yacc.c:1661  */
    break;

  case 176:
#line 647 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3326 "y.tab.c" /* yacc.c:1661  */
    break;

  case 177:
#line 651 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.chare) = new NodeGroup(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3332 "y.tab.c" /* yacc.c:1661  */
    break;

  case 178:
#line 655 "xi-grammar.y" /* yacc.c:1661  */
    {/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",(yyvsp[-2].strval));
			(yyval.ntype) = new NamedType(buf); 
		}
#line 3342 "y.tab.c" /* yacc.c:1661  */
    break;

  case 179:
#line 661 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.ntype) = new NamedType((yyvsp[-1].strval)); }
#line 3348 "y.tab.c" /* yacc.c:1661  */
    break;

  case 180:
#line 665 "xi-grammar.y" /* yacc.c:1661  */
    {  (yyval.chare) = new Array(lineno, (yyvsp[-4].cattr), (yyvsp[-3].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3354 "y.tab.c" /* yacc.c:1661  */
    break;

  case 181:
#line 667 "xi-grammar.y" /* yacc.c:1661  */
    {  (yyval.chare) = new Array(lineno, (yyvsp[-3].cattr), (yyvsp[-4].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3360 "y.tab.c" /* yacc.c:1661  */
    break;

  case 182:
#line 671 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr)|Chare::CCHARE, new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist));}
#line 3366 "y.tab.c" /* yacc.c:1661  */
    break;

  case 183:
#line 673 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3372 "y.tab.c" /* yacc.c:1661  */
    break;

  case 184:
#line 677 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3378 "y.tab.c" /* yacc.c:1661  */
    break;

  case 185:
#line 681 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.chare) = new NodeGroup( lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3384 "y.tab.c" /* yacc.c:1661  */
    break;

  case 186:
#line 685 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.chare) = new Array( lineno, 0, (yyvsp[-3].ntype), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3390 "y.tab.c" /* yacc.c:1661  */
    break;

  case 187:
#line 689 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-1].strval))); }
#line 3396 "y.tab.c" /* yacc.c:1661  */
    break;

  case 188:
#line 691 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-4].strval)), (yyvsp[-2].mvlist)); }
#line 3402 "y.tab.c" /* yacc.c:1661  */
    break;

  case 189:
#line 695 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = 0; }
#line 3408 "y.tab.c" /* yacc.c:1661  */
    break;

  case 190:
#line 697 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3414 "y.tab.c" /* yacc.c:1661  */
    break;

  case 191:
#line 701 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = 0; }
#line 3420 "y.tab.c" /* yacc.c:1661  */
    break;

  case 192:
#line 703 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3426 "y.tab.c" /* yacc.c:1661  */
    break;

  case 193:
#line 705 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3432 "y.tab.c" /* yacc.c:1661  */
    break;

  case 194:
#line 707 "xi-grammar.y" /* yacc.c:1661  */
    {
		  XStr typeStr;
		  (yyvsp[0].ntype)->print(typeStr);
		  char *tmp = strdup(typeStr.get_string());
		  (yyval.strval) = tmp;
		}
#line 3443 "y.tab.c" /* yacc.c:1661  */
    break;

  case 195:
#line 716 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tvar) = new TType(new NamedType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3449 "y.tab.c" /* yacc.c:1661  */
    break;

  case 196:
#line 718 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tvar) = new TFunc((yyvsp[-1].ftype), (yyvsp[0].strval)); }
#line 3455 "y.tab.c" /* yacc.c:1661  */
    break;

  case 197:
#line 720 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tvar) = new TName((yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].strval)); }
#line 3461 "y.tab.c" /* yacc.c:1661  */
    break;

  case 198:
#line 724 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tvarlist) = new TVarList((yyvsp[0].tvar)); }
#line 3467 "y.tab.c" /* yacc.c:1661  */
    break;

  case 199:
#line 726 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tvarlist) = new TVarList((yyvsp[-2].tvar), (yyvsp[0].tvarlist)); }
#line 3473 "y.tab.c" /* yacc.c:1661  */
    break;

  case 200:
#line 730 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tvarlist) = (yyvsp[-1].tvarlist); }
#line 3479 "y.tab.c" /* yacc.c:1661  */
    break;

  case 201:
#line 734 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3485 "y.tab.c" /* yacc.c:1661  */
    break;

  case 202:
#line 736 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3491 "y.tab.c" /* yacc.c:1661  */
    break;

  case 203:
#line 738 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3497 "y.tab.c" /* yacc.c:1661  */
    break;

  case 204:
#line 740 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3503 "y.tab.c" /* yacc.c:1661  */
    break;

  case 205:
#line 742 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].message)); (yyvsp[0].message)->setTemplate((yyval.templat)); }
#line 3509 "y.tab.c" /* yacc.c:1661  */
    break;

  case 206:
#line 746 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.mbrlist) = 0; }
#line 3515 "y.tab.c" /* yacc.c:1661  */
    break;

  case 207:
#line 748 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.mbrlist) = (yyvsp[-2].mbrlist); }
#line 3521 "y.tab.c" /* yacc.c:1661  */
    break;

  case 208:
#line 752 "xi-grammar.y" /* yacc.c:1661  */
    { 
                  if (!connectEntries.empty()) {
                    (yyval.mbrlist) = new AstChildren<Member>(connectEntries);
		  } else {
		    (yyval.mbrlist) = 0; 
                  }
		}
#line 3533 "y.tab.c" /* yacc.c:1661  */
    break;

  case 209:
#line 760 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.mbrlist) = new AstChildren<Member>(-1, (yyvsp[-1].member), (yyvsp[0].mbrlist)); }
#line 3539 "y.tab.c" /* yacc.c:1661  */
    break;

  case 210:
#line 764 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = (yyvsp[0].readonly); }
#line 3545 "y.tab.c" /* yacc.c:1661  */
    break;

  case 211:
#line 766 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = (yyvsp[0].readonly); }
#line 3551 "y.tab.c" /* yacc.c:1661  */
    break;

  case 213:
#line 769 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = (yyvsp[0].member); }
#line 3557 "y.tab.c" /* yacc.c:1661  */
    break;

  case 214:
#line 771 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = (yyvsp[0].pupable); }
#line 3563 "y.tab.c" /* yacc.c:1661  */
    break;

  case 215:
#line 773 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = (yyvsp[0].includeFile); }
#line 3569 "y.tab.c" /* yacc.c:1661  */
    break;

  case 216:
#line 775 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = new ClassDeclaration(lineno,(yyvsp[0].strval)); }
#line 3575 "y.tab.c" /* yacc.c:1661  */
    break;

  case 217:
#line 779 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1); }
#line 3581 "y.tab.c" /* yacc.c:1661  */
    break;

  case 218:
#line 781 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1); }
#line 3587 "y.tab.c" /* yacc.c:1661  */
    break;

  case 219:
#line 783 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[-6].strval)) + '<' +
					    ((yyvsp[-4].tparlist))->to_string() + '>').c_str()),
				    1);
		}
#line 3597 "y.tab.c" /* yacc.c:1661  */
    break;

  case 220:
#line 789 "xi-grammar.y" /* yacc.c:1661  */
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[-2]).first_column, (yylsp[-2]).last_column, (yylsp[-2]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1);
		}
#line 3607 "y.tab.c" /* yacc.c:1661  */
    break;

  case 221:
#line 795 "xi-grammar.y" /* yacc.c:1661  */
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[-5]).first_column, (yylsp[-5]).last_column, (yylsp[-5]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1);
		}
#line 3617 "y.tab.c" /* yacc.c:1661  */
    break;

  case 222:
#line 804 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 0); }
#line 3623 "y.tab.c" /* yacc.c:1661  */
    break;

  case 223:
#line 806 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 0); }
#line 3629 "y.tab.c" /* yacc.c:1661  */
    break;

  case 224:
#line 808 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[-6].strval)) + '<' +
					    ((yyvsp[-4].tparlist))->to_string() + '>').c_str()),
				    0);
		}
#line 3639 "y.tab.c" /* yacc.c:1661  */
    break;

  case 225:
#line 814 "xi-grammar.y" /* yacc.c:1661  */
    {
                  InitCall* rtn = new InitCall(lineno, (yyvsp[-3].strval), 0);
                  rtn->setAccel();
                  (yyval.member) = rtn;
		}
#line 3649 "y.tab.c" /* yacc.c:1661  */
    break;

  case 226:
#line 822 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[0].ntype),0); }
#line 3655 "y.tab.c" /* yacc.c:1661  */
    break;

  case 227:
#line 824 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[-2].ntype),(yyvsp[0].pupable)); }
#line 3661 "y.tab.c" /* yacc.c:1661  */
    break;

  case 228:
#line 827 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.includeFile) = new IncludeFile(lineno,(yyvsp[0].strval)); }
#line 3667 "y.tab.c" /* yacc.c:1661  */
    break;

  case 229:
#line 831 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = (yyvsp[0].member); }
#line 3673 "y.tab.c" /* yacc.c:1661  */
    break;

  case 230:
#line 835 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = (yyvsp[0].entry); }
#line 3679 "y.tab.c" /* yacc.c:1661  */
    break;

  case 231:
#line 837 "xi-grammar.y" /* yacc.c:1661  */
    {
                  (yyvsp[0].entry)->tspec = (yyvsp[-1].tvarlist);
                  (yyval.member) = (yyvsp[0].entry);
                }
#line 3688 "y.tab.c" /* yacc.c:1661  */
    break;

  case 232:
#line 842 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = (yyvsp[-1].member); }
#line 3694 "y.tab.c" /* yacc.c:1661  */
    break;

  case 233:
#line 844 "xi-grammar.y" /* yacc.c:1661  */
    {
          ERROR("invalid SDAG member",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 3704 "y.tab.c" /* yacc.c:1661  */
    break;

  case 234:
#line 852 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3710 "y.tab.c" /* yacc.c:1661  */
    break;

  case 235:
#line 854 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3716 "y.tab.c" /* yacc.c:1661  */
    break;

  case 236:
#line 856 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3722 "y.tab.c" /* yacc.c:1661  */
    break;

  case 237:
#line 858 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3728 "y.tab.c" /* yacc.c:1661  */
    break;

  case 238:
#line 860 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3734 "y.tab.c" /* yacc.c:1661  */
    break;

  case 239:
#line 862 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3740 "y.tab.c" /* yacc.c:1661  */
    break;

  case 240:
#line 864 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3746 "y.tab.c" /* yacc.c:1661  */
    break;

  case 241:
#line 866 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3752 "y.tab.c" /* yacc.c:1661  */
    break;

  case 242:
#line 868 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3758 "y.tab.c" /* yacc.c:1661  */
    break;

  case 243:
#line 870 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3764 "y.tab.c" /* yacc.c:1661  */
    break;

  case 244:
#line 872 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3770 "y.tab.c" /* yacc.c:1661  */
    break;

  case 245:
#line 875 "xi-grammar.y" /* yacc.c:1661  */
    { 
                  (yyval.entry) = new Entry(lineno, (yyvsp[-5].attr), (yyvsp[-4].type), (yyvsp[-3].strval), (yyvsp[-2].plist), (yyvsp[-1].val), (yyvsp[0].sentry), (const char *) NULL, (yylsp[-6]).first_line, (yyloc).last_line);
		  if ((yyvsp[0].sentry) != 0) { 
		    (yyvsp[0].sentry)->con1 = new SdagConstruct(SIDENT, (yyvsp[-3].strval));
                    (yyvsp[0].sentry)->setEntry((yyval.entry));
                    (yyvsp[0].sentry)->param = new ParamList((yyvsp[-2].plist));
                  }
                  firstRdma = true;
		}
#line 3784 "y.tab.c" /* yacc.c:1661  */
    break;

  case 246:
#line 885 "xi-grammar.y" /* yacc.c:1661  */
    { 
                  Entry *e = new Entry(lineno, (yyvsp[-3].attr), 0, (yyvsp[-2].strval), (yyvsp[-1].plist),  0, (yyvsp[0].sentry), (const char *) NULL, (yylsp[-4]).first_line, (yyloc).last_line);
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
#line 3805 "y.tab.c" /* yacc.c:1661  */
    break;

  case 247:
#line 902 "xi-grammar.y" /* yacc.c:1661  */
    {
                  Attribute* attribs = new Attribute(SACCEL);
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
#line 3824 "y.tab.c" /* yacc.c:1661  */
    break;

  case 248:
#line 919 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.accelBlock) = new AccelBlock(lineno, new XStr((yyvsp[-2].strval))); }
#line 3830 "y.tab.c" /* yacc.c:1661  */
    break;

  case 249:
#line 921 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.accelBlock) = new AccelBlock(lineno, NULL); }
#line 3836 "y.tab.c" /* yacc.c:1661  */
    break;

  case 250:
#line 925 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3842 "y.tab.c" /* yacc.c:1661  */
    break;

  case 251:
#line 929 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.attr) = NULL; }
#line 3848 "y.tab.c" /* yacc.c:1661  */
    break;

  case 252:
#line 931 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.attr) = (yyvsp[-1].attr); }
#line 3854 "y.tab.c" /* yacc.c:1661  */
    break;

  case 253:
#line 933 "xi-grammar.y" /* yacc.c:1661  */
    { ERROR("invalid entry method attribute list",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
#line 3863 "y.tab.c" /* yacc.c:1661  */
    break;

  case 254:
#line 940 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.attrarg) = new Attribute::Argument((yyvsp[-2].strval), atoi((yyvsp[0].strval))); }
#line 3869 "y.tab.c" /* yacc.c:1661  */
    break;

  case 255:
#line 944 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.attrarg) = (yyvsp[0].attrarg); }
#line 3875 "y.tab.c" /* yacc.c:1661  */
    break;

  case 256:
#line 945 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.attrarg) = (yyvsp[-2].attrarg); (yyvsp[-2].attrarg)->next = (yyvsp[0].attrarg); }
#line 3881 "y.tab.c" /* yacc.c:1661  */
    break;

  case 257:
#line 949 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.attr) = new Attribute((yyvsp[0].intval));           }
#line 3887 "y.tab.c" /* yacc.c:1661  */
    break;

  case 258:
#line 950 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.attr) = new Attribute((yyvsp[-3].intval), (yyvsp[-1].attrarg));       }
#line 3893 "y.tab.c" /* yacc.c:1661  */
    break;

  case 259:
#line 951 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.attr) = new Attribute((yyvsp[-2].intval), NULL, (yyvsp[0].attr)); }
#line 3899 "y.tab.c" /* yacc.c:1661  */
    break;

  case 260:
#line 952 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.attr) = new Attribute((yyvsp[-5].intval), (yyvsp[-3].attrarg), (yyvsp[0].attr));   }
#line 3905 "y.tab.c" /* yacc.c:1661  */
    break;

  case 261:
#line 956 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = STHREADED; }
#line 3911 "y.tab.c" /* yacc.c:1661  */
    break;

  case 262:
#line 958 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SSYNC; }
#line 3917 "y.tab.c" /* yacc.c:1661  */
    break;

  case 263:
#line 960 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SIGET; }
#line 3923 "y.tab.c" /* yacc.c:1661  */
    break;

  case 264:
#line 962 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SLOCKED; }
#line 3929 "y.tab.c" /* yacc.c:1661  */
    break;

  case 265:
#line 964 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SCREATEHERE; }
#line 3935 "y.tab.c" /* yacc.c:1661  */
    break;

  case 266:
#line 966 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SCREATEHOME; }
#line 3941 "y.tab.c" /* yacc.c:1661  */
    break;

  case 267:
#line 968 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SNOKEEP; }
#line 3947 "y.tab.c" /* yacc.c:1661  */
    break;

  case 268:
#line 970 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SNOTRACE; }
#line 3953 "y.tab.c" /* yacc.c:1661  */
    break;

  case 269:
#line 972 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SAPPWORK; }
#line 3959 "y.tab.c" /* yacc.c:1661  */
    break;

  case 270:
#line 974 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SIMMEDIATE; }
#line 3965 "y.tab.c" /* yacc.c:1661  */
    break;

  case 271:
#line 976 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SSKIPSCHED; }
#line 3971 "y.tab.c" /* yacc.c:1661  */
    break;

  case 272:
#line 978 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SINLINE; }
#line 3977 "y.tab.c" /* yacc.c:1661  */
    break;

  case 273:
#line 980 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SLOCAL; }
#line 3983 "y.tab.c" /* yacc.c:1661  */
    break;

  case 274:
#line 982 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SPYTHON; }
#line 3989 "y.tab.c" /* yacc.c:1661  */
    break;

  case 275:
#line 984 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SMEM; }
#line 3995 "y.tab.c" /* yacc.c:1661  */
    break;

  case 276:
#line 986 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SREDUCE; }
#line 4001 "y.tab.c" /* yacc.c:1661  */
    break;

  case 277:
#line 988 "xi-grammar.y" /* yacc.c:1661  */
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
#line 4017 "y.tab.c" /* yacc.c:1661  */
    break;

  case 278:
#line 1000 "xi-grammar.y" /* yacc.c:1661  */
    {
		  ERROR("invalid entry method attribute",
		        (yylsp[0]).first_column, (yylsp[0]).last_column);
		  yyclearin;
		  yyerrok;
		}
#line 4028 "y.tab.c" /* yacc.c:1661  */
    break;

  case 279:
#line 1009 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4034 "y.tab.c" /* yacc.c:1661  */
    break;

  case 280:
#line 1011 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4040 "y.tab.c" /* yacc.c:1661  */
    break;

  case 281:
#line 1013 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4046 "y.tab.c" /* yacc.c:1661  */
    break;

  case 282:
#line 1017 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = ""; }
#line 4052 "y.tab.c" /* yacc.c:1661  */
    break;

  case 283:
#line 1019 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4058 "y.tab.c" /* yacc.c:1661  */
    break;

  case 284:
#line 1021 "xi-grammar.y" /* yacc.c:1661  */
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s, %s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4068 "y.tab.c" /* yacc.c:1661  */
    break;

  case 285:
#line 1029 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = ""; }
#line 4074 "y.tab.c" /* yacc.c:1661  */
    break;

  case 286:
#line 1031 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4080 "y.tab.c" /* yacc.c:1661  */
    break;

  case 287:
#line 1033 "xi-grammar.y" /* yacc.c:1661  */
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s[%s]%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4090 "y.tab.c" /* yacc.c:1661  */
    break;

  case 288:
#line 1039 "xi-grammar.y" /* yacc.c:1661  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s{%s}%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4100 "y.tab.c" /* yacc.c:1661  */
    break;

  case 289:
#line 1045 "xi-grammar.y" /* yacc.c:1661  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s(%s)%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4110 "y.tab.c" /* yacc.c:1661  */
    break;

  case 290:
#line 1051 "xi-grammar.y" /* yacc.c:1661  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"(%s)%s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4120 "y.tab.c" /* yacc.c:1661  */
    break;

  case 291:
#line 1059 "xi-grammar.y" /* yacc.c:1661  */
    {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval));
		}
#line 4129 "y.tab.c" /* yacc.c:1661  */
    break;

  case 292:
#line 1066 "xi-grammar.y" /* yacc.c:1661  */
    { 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			(yyval.intval) = 0;
		}
#line 4139 "y.tab.c" /* yacc.c:1661  */
    break;

  case 293:
#line 1074 "xi-grammar.y" /* yacc.c:1661  */
    { 
			in_braces=0;
			(yyval.intval) = 0;
		}
#line 4148 "y.tab.c" /* yacc.c:1661  */
    break;

  case 294:
#line 1081 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[0].type));}
#line 4154 "y.tab.c" /* yacc.c:1661  */
    break;

  case 295:
#line 1083 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval)); (yyval.pname)->setConditional((yyvsp[0].intval)); }
#line 4160 "y.tab.c" /* yacc.c:1661  */
    break;

  case 296:
#line 1085 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[-3].type),(yyvsp[-2].strval),0,(yyvsp[0].val));}
#line 4166 "y.tab.c" /* yacc.c:1661  */
    break;

  case 297:
#line 1087 "xi-grammar.y" /* yacc.c:1661  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
		}
#line 4175 "y.tab.c" /* yacc.c:1661  */
    break;

  case 298:
#line 1092 "xi-grammar.y" /* yacc.c:1661  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
			(yyval.pname)->setRdma(true);
			if(firstRdma) {
				(yyval.pname)->setFirstRdma(true);
				firstRdma = false;
			}
		}
#line 4189 "y.tab.c" /* yacc.c:1661  */
    break;

  case 299:
#line 1103 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READONLY; }
#line 4195 "y.tab.c" /* yacc.c:1661  */
    break;

  case 300:
#line 1104 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READWRITE; }
#line 4201 "y.tab.c" /* yacc.c:1661  */
    break;

  case 301:
#line 1105 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_WRITEONLY; }
#line 4207 "y.tab.c" /* yacc.c:1661  */
    break;

  case 302:
#line 1108 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.xstrptr) = new XStr((yyvsp[0].strval)); }
#line 4213 "y.tab.c" /* yacc.c:1661  */
    break;

  case 303:
#line 1109 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "->" << (yyvsp[0].strval); }
#line 4219 "y.tab.c" /* yacc.c:1661  */
    break;

  case 304:
#line 1110 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[-2].xstrptr)) << "." << (yyvsp[0].strval); }
#line 4225 "y.tab.c" /* yacc.c:1661  */
    break;

  case 305:
#line 1112 "xi-grammar.y" /* yacc.c:1661  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << *((yyvsp[-1].xstrptr)) << "]";
                  delete (yyvsp[-3].xstrptr);
                  delete (yyvsp[-1].xstrptr);
                }
#line 4236 "y.tab.c" /* yacc.c:1661  */
    break;

  case 306:
#line 1119 "xi-grammar.y" /* yacc.c:1661  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << (yyvsp[-1].strval) << "]";
                  delete (yyvsp[-3].xstrptr);
                }
#line 4246 "y.tab.c" /* yacc.c:1661  */
    break;

  case 307:
#line 1125 "xi-grammar.y" /* yacc.c:1661  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "(" << *((yyvsp[-1].xstrptr)) << ")";
                  delete (yyvsp[-3].xstrptr);
                  delete (yyvsp[-1].xstrptr);
                }
#line 4257 "y.tab.c" /* yacc.c:1661  */
    break;

  case 308:
#line 1134 "xi-grammar.y" /* yacc.c:1661  */
    {
                  in_bracket = 0;
                  (yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName(), (yyvsp[-1].strval));
                }
#line 4266 "y.tab.c" /* yacc.c:1661  */
    break;

  case 309:
#line 1141 "xi-grammar.y" /* yacc.c:1661  */
    {
                  (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[-6].intval));
                }
#line 4276 "y.tab.c" /* yacc.c:1661  */
    break;

  case 310:
#line 1147 "xi-grammar.y" /* yacc.c:1661  */
    {
		  (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType(Parameter::ACCEL_BUFFER_TYPE_READWRITE);
		}
#line 4286 "y.tab.c" /* yacc.c:1661  */
    break;

  case 311:
#line 1153 "xi-grammar.y" /* yacc.c:1661  */
    {
                  (yyval.pname) = (yyvsp[-3].pname);
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[-5].intval));
		}
#line 4296 "y.tab.c" /* yacc.c:1661  */
    break;

  case 312:
#line 1161 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 4302 "y.tab.c" /* yacc.c:1661  */
    break;

  case 313:
#line 1163 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 4308 "y.tab.c" /* yacc.c:1661  */
    break;

  case 314:
#line 1167 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 4314 "y.tab.c" /* yacc.c:1661  */
    break;

  case 315:
#line 1169 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 4320 "y.tab.c" /* yacc.c:1661  */
    break;

  case 316:
#line 1173 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.plist) = (yyvsp[-1].plist); }
#line 4326 "y.tab.c" /* yacc.c:1661  */
    break;

  case 317:
#line 1175 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.plist) = new ParamList(new Parameter(0, new BuiltinType("void"))); }
#line 4332 "y.tab.c" /* yacc.c:1661  */
    break;

  case 318:
#line 1179 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.plist) = (yyvsp[-1].plist); }
#line 4338 "y.tab.c" /* yacc.c:1661  */
    break;

  case 319:
#line 1181 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.plist) = 0; }
#line 4344 "y.tab.c" /* yacc.c:1661  */
    break;

  case 320:
#line 1185 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.val) = 0; }
#line 4350 "y.tab.c" /* yacc.c:1661  */
    break;

  case 321:
#line 1187 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4356 "y.tab.c" /* yacc.c:1661  */
    break;

  case 322:
#line 1191 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sentry) = 0; }
#line 4362 "y.tab.c" /* yacc.c:1661  */
    break;

  case 323:
#line 1193 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[0].sc)); }
#line 4368 "y.tab.c" /* yacc.c:1661  */
    break;

  case 324:
#line 1195 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[-2].slist)); }
#line 4374 "y.tab.c" /* yacc.c:1661  */
    break;

  case 325:
#line 1199 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.slist) = new SListConstruct((yyvsp[0].sc)); }
#line 4380 "y.tab.c" /* yacc.c:1661  */
    break;

  case 326:
#line 1201 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.slist) = new SListConstruct((yyvsp[-1].sc), (yyvsp[0].slist));  }
#line 4386 "y.tab.c" /* yacc.c:1661  */
    break;

  case 327:
#line 1205 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.olist) = new OListConstruct((yyvsp[0].sc)); }
#line 4392 "y.tab.c" /* yacc.c:1661  */
    break;

  case 328:
#line 1207 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.olist) = new OListConstruct((yyvsp[-1].sc), (yyvsp[0].slist)); }
#line 4398 "y.tab.c" /* yacc.c:1661  */
    break;

  case 329:
#line 1211 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.clist) = new CaseListConstruct((yyvsp[0].when)); }
#line 4404 "y.tab.c" /* yacc.c:1661  */
    break;

  case 330:
#line 1213 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.clist) = new CaseListConstruct((yyvsp[-1].when), (yyvsp[0].clist)); }
#line 4410 "y.tab.c" /* yacc.c:1661  */
    break;

  case 331:
#line 1215 "xi-grammar.y" /* yacc.c:1661  */
    {
		  ERROR("case blocks can only contain when clauses",
		        (yylsp[0]).first_column, (yylsp[0]).last_column);
		  (yyval.clist) = 0;
		}
#line 4420 "y.tab.c" /* yacc.c:1661  */
    break;

  case 332:
#line 1223 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4426 "y.tab.c" /* yacc.c:1661  */
    break;

  case 333:
#line 1225 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = 0; }
#line 4432 "y.tab.c" /* yacc.c:1661  */
    break;

  case 334:
#line 1229 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = new WhenConstruct((yyvsp[-2].entrylist), 0); }
#line 4438 "y.tab.c" /* yacc.c:1661  */
    break;

  case 335:
#line 1231 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = new WhenConstruct((yyvsp[-1].entrylist), (yyvsp[0].sc)); }
#line 4444 "y.tab.c" /* yacc.c:1661  */
    break;

  case 336:
#line 1233 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = new WhenConstruct((yyvsp[-3].entrylist), (yyvsp[-1].slist)); }
#line 4450 "y.tab.c" /* yacc.c:1661  */
    break;

  case 337:
#line 1237 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4456 "y.tab.c" /* yacc.c:1661  */
    break;

  case 338:
#line 1239 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4462 "y.tab.c" /* yacc.c:1661  */
    break;

  case 339:
#line 1241 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4468 "y.tab.c" /* yacc.c:1661  */
    break;

  case 340:
#line 1243 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4474 "y.tab.c" /* yacc.c:1661  */
    break;

  case 341:
#line 1245 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4480 "y.tab.c" /* yacc.c:1661  */
    break;

  case 342:
#line 1247 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4486 "y.tab.c" /* yacc.c:1661  */
    break;

  case 343:
#line 1249 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4492 "y.tab.c" /* yacc.c:1661  */
    break;

  case 344:
#line 1251 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4498 "y.tab.c" /* yacc.c:1661  */
    break;

  case 345:
#line 1253 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4504 "y.tab.c" /* yacc.c:1661  */
    break;

  case 346:
#line 1255 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4510 "y.tab.c" /* yacc.c:1661  */
    break;

  case 347:
#line 1257 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4516 "y.tab.c" /* yacc.c:1661  */
    break;

  case 348:
#line 1259 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4522 "y.tab.c" /* yacc.c:1661  */
    break;

  case 349:
#line 1263 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new SerialConstruct((yyvsp[-2].strval), (yyvsp[-4].strval), (yylsp[-3]).first_line); }
#line 4528 "y.tab.c" /* yacc.c:1661  */
    break;

  case 350:
#line 1265 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new OverlapConstruct((yyvsp[-1].olist)); }
#line 4534 "y.tab.c" /* yacc.c:1661  */
    break;

  case 351:
#line 1267 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = (yyvsp[0].when); }
#line 4540 "y.tab.c" /* yacc.c:1661  */
    break;

  case 352:
#line 1269 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new CaseConstruct((yyvsp[-1].clist)); }
#line 4546 "y.tab.c" /* yacc.c:1661  */
    break;

  case 353:
#line 1271 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new ForConstruct((yyvsp[-8].intexpr), (yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4552 "y.tab.c" /* yacc.c:1661  */
    break;

  case 354:
#line 1273 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new ForConstruct((yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4558 "y.tab.c" /* yacc.c:1661  */
    break;

  case 355:
#line 1275 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-9].strval)), (yyvsp[-6].intexpr),
		             (yyvsp[-4].intexpr), (yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4565 "y.tab.c" /* yacc.c:1661  */
    break;

  case 356:
#line 1278 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-11].strval)), (yyvsp[-8].intexpr),
		             (yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4572 "y.tab.c" /* yacc.c:1661  */
    break;

  case 357:
#line 1281 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new IfConstruct((yyvsp[-3].intexpr), (yyvsp[-1].sc), (yyvsp[0].sc)); }
#line 4578 "y.tab.c" /* yacc.c:1661  */
    break;

  case 358:
#line 1283 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new IfConstruct((yyvsp[-5].intexpr), (yyvsp[-2].slist), (yyvsp[0].sc)); }
#line 4584 "y.tab.c" /* yacc.c:1661  */
    break;

  case 359:
#line 1285 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new WhileConstruct((yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4590 "y.tab.c" /* yacc.c:1661  */
    break;

  case 360:
#line 1287 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new WhileConstruct((yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4596 "y.tab.c" /* yacc.c:1661  */
    break;

  case 361:
#line 1289 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new SerialConstruct((yyvsp[-2].strval), NULL, (yyloc).first_line); }
#line 4602 "y.tab.c" /* yacc.c:1661  */
    break;

  case 362:
#line 1291 "xi-grammar.y" /* yacc.c:1661  */
    {
		  ERROR("unknown SDAG construct or malformed entry method declaration.\n"
		        "You may have forgotten to terminate a previous entry method declaration with a"
		        " semicolon or forgotten to mark a block of sequential SDAG code as 'serial'",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
#line 4614 "y.tab.c" /* yacc.c:1661  */
    break;

  case 363:
#line 1301 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = 0; }
#line 4620 "y.tab.c" /* yacc.c:1661  */
    break;

  case 364:
#line 1303 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new ElseConstruct((yyvsp[0].sc)); }
#line 4626 "y.tab.c" /* yacc.c:1661  */
    break;

  case 365:
#line 1305 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new ElseConstruct((yyvsp[-1].slist)); }
#line 4632 "y.tab.c" /* yacc.c:1661  */
    break;

  case 366:
#line 1309 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intexpr) = new IntExprConstruct((yyvsp[0].strval)); }
#line 4638 "y.tab.c" /* yacc.c:1661  */
    break;

  case 367:
#line 1313 "xi-grammar.y" /* yacc.c:1661  */
    { in_int_expr = 0; (yyval.intval) = 0; }
#line 4644 "y.tab.c" /* yacc.c:1661  */
    break;

  case 368:
#line 1317 "xi-grammar.y" /* yacc.c:1661  */
    { in_int_expr = 1; (yyval.intval) = 0; }
#line 4650 "y.tab.c" /* yacc.c:1661  */
    break;

  case 369:
#line 1321 "xi-grammar.y" /* yacc.c:1661  */
    {
		  (yyval.entry) = new Entry(lineno, NULL, 0, (yyvsp[-1].strval), (yyvsp[0].plist), 0, 0, 0, (yyloc).first_line, (yyloc).last_line);
		  firstRdma = true;
		}
#line 4659 "y.tab.c" /* yacc.c:1661  */
    break;

  case 370:
#line 1326 "xi-grammar.y" /* yacc.c:1661  */
    {
		  (yyval.entry) = new Entry(lineno, NULL, 0, (yyvsp[-4].strval), (yyvsp[0].plist), 0, 0, (yyvsp[-2].strval), (yyloc).first_line, (yyloc).last_line);
		  firstRdma = true;
		}
#line 4668 "y.tab.c" /* yacc.c:1661  */
    break;

  case 371:
#line 1333 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.entrylist) = new EntryList((yyvsp[0].entry)); }
#line 4674 "y.tab.c" /* yacc.c:1661  */
    break;

  case 372:
#line 1335 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.entrylist) = new EntryList((yyvsp[-2].entry),(yyvsp[0].entrylist)); }
#line 4680 "y.tab.c" /* yacc.c:1661  */
    break;

  case 373:
#line 1339 "xi-grammar.y" /* yacc.c:1661  */
    { in_bracket=1; }
#line 4686 "y.tab.c" /* yacc.c:1661  */
    break;

  case 374:
#line 1342 "xi-grammar.y" /* yacc.c:1661  */
    { in_bracket=0; }
#line 4692 "y.tab.c" /* yacc.c:1661  */
    break;

  case 375:
#line 1346 "xi-grammar.y" /* yacc.c:1661  */
    { if (!macroDefined((yyvsp[0].strval), 1)) in_comment = 1; }
#line 4698 "y.tab.c" /* yacc.c:1661  */
    break;

  case 376:
#line 1350 "xi-grammar.y" /* yacc.c:1661  */
    { if (!macroDefined((yyvsp[0].strval), 0)) in_comment = 1; }
#line 4704 "y.tab.c" /* yacc.c:1661  */
    break;


#line 4708 "y.tab.c" /* yacc.c:1661  */
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
#line 1353 "xi-grammar.y" /* yacc.c:1906  */


void yyerror(const char *msg) { }
