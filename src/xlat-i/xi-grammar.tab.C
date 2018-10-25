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
    NOCOPYPOST = 295,
    PACKED = 296,
    VARSIZE = 297,
    ENTRY = 298,
    FOR = 299,
    FORALL = 300,
    WHILE = 301,
    WHEN = 302,
    OVERLAP = 303,
    SERIAL = 304,
    IF = 305,
    ELSE = 306,
    PYTHON = 307,
    LOCAL = 308,
    NAMESPACE = 309,
    USING = 310,
    IDENT = 311,
    NUMBER = 312,
    LITERAL = 313,
    CPROGRAM = 314,
    HASHIF = 315,
    HASHIFDEF = 316,
    INT = 317,
    LONG = 318,
    SHORT = 319,
    CHAR = 320,
    FLOAT = 321,
    DOUBLE = 322,
    UNSIGNED = 323,
    ACCEL = 324,
    READWRITE = 325,
    WRITEONLY = 326,
    ACCELBLOCK = 327,
    MEMCRITICAL = 328,
    REDUCTIONTARGET = 329,
    CASE = 330,
    TYPENAME = 331
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
#define NOCOPYPOST 295
#define PACKED 296
#define VARSIZE 297
#define ENTRY 298
#define FOR 299
#define FORALL 300
#define WHILE 301
#define WHEN 302
#define OVERLAP 303
#define SERIAL 304
#define IF 305
#define ELSE 306
#define PYTHON 307
#define LOCAL 308
#define NAMESPACE 309
#define USING 310
#define IDENT 311
#define NUMBER 312
#define LITERAL 313
#define CPROGRAM 314
#define HASHIF 315
#define HASHIFDEF 316
#define INT 317
#define LONG 318
#define SHORT 319
#define CHAR 320
#define FLOAT 321
#define DOUBLE 322
#define UNSIGNED 323
#define ACCEL 324
#define READWRITE 325
#define WRITEONLY 326
#define ACCELBLOCK 327
#define MEMCRITICAL 328
#define REDUCTIONTARGET 329
#define CASE 330
#define TYPENAME 331

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

#line 351 "y.tab.c" /* yacc.c:355  */
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

#line 382 "y.tab.c" /* yacc.c:358  */

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
#define YYFINAL  57
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   1581

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  93
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  117
/* YYNRULES -- Number of rules.  */
#define YYNRULES  386
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  770

/* YYTRANSLATE[YYX] -- Symbol number corresponding to YYX as returned
   by yylex, with out-of-bounds checking.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   331

#define YYTRANSLATE(YYX)                                                \
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, without out-of-bounds checking.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    87,     2,
      85,    86,    84,     2,    81,    92,    88,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    78,    77,
      82,    91,    83,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    89,     2,    90,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    79,     2,    80,     2,     2,     2,     2,
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
      75,    76
};

#if YYDEBUG
  /* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   197,   197,   202,   205,   210,   211,   215,   217,   222,
     223,   228,   230,   231,   232,   234,   235,   236,   238,   239,
     240,   241,   242,   246,   247,   248,   249,   250,   251,   252,
     253,   254,   255,   256,   257,   258,   259,   260,   261,   262,
     263,   264,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   279,   281,   282,   285,   286,   287,   288,
     292,   294,   300,   307,   311,   318,   320,   325,   326,   330,
     332,   334,   336,   338,   351,   353,   355,   357,   363,   365,
     367,   369,   371,   373,   375,   377,   379,   381,   389,   391,
     393,   397,   399,   404,   405,   410,   411,   415,   417,   419,
     421,   423,   425,   427,   429,   431,   433,   435,   437,   439,
     441,   443,   447,   448,   453,   461,   463,   467,   471,   473,
     477,   481,   483,   485,   487,   489,   491,   495,   497,   499,
     501,   503,   507,   509,   511,   513,   515,   517,   521,   523,
     525,   527,   529,   531,   535,   539,   544,   545,   549,   553,
     558,   559,   564,   565,   575,   577,   581,   583,   588,   589,
     593,   595,   600,   601,   605,   610,   611,   615,   617,   621,
     623,   628,   629,   633,   634,   637,   641,   643,   647,   649,
     651,   656,   657,   661,   663,   667,   669,   673,   677,   681,
     687,   691,   693,   697,   699,   703,   707,   711,   715,   717,
     722,   723,   728,   729,   731,   733,   742,   744,   746,   748,
     750,   752,   756,   758,   762,   766,   768,   770,   772,   774,
     778,   780,   785,   792,   796,   798,   800,   801,   803,   805,
     807,   811,   813,   815,   821,   827,   836,   838,   840,   846,
     854,   856,   859,   863,   867,   869,   874,   876,   884,   886,
     888,   890,   892,   894,   896,   898,   900,   902,   904,   907,
     917,   934,   951,   953,   957,   962,   963,   965,   972,   974,
     978,   980,   982,   984,   986,   988,   990,   992,   994,   996,
     998,  1000,  1002,  1004,  1006,  1008,  1010,  1014,  1023,  1025,
    1027,  1032,  1033,  1035,  1044,  1045,  1047,  1053,  1059,  1065,
    1073,  1080,  1088,  1095,  1097,  1099,  1101,  1106,  1116,  1128,
    1129,  1130,  1133,  1134,  1135,  1136,  1143,  1149,  1158,  1165,
    1171,  1177,  1185,  1187,  1191,  1193,  1197,  1199,  1203,  1205,
    1210,  1211,  1215,  1217,  1219,  1223,  1225,  1229,  1231,  1235,
    1237,  1239,  1247,  1250,  1253,  1255,  1257,  1261,  1263,  1265,
    1267,  1269,  1271,  1273,  1275,  1277,  1279,  1281,  1283,  1287,
    1289,  1291,  1293,  1295,  1297,  1299,  1302,  1305,  1307,  1309,
    1311,  1313,  1315,  1326,  1327,  1329,  1333,  1337,  1341,  1345,
    1350,  1357,  1359,  1363,  1366,  1370,  1374
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
  "VOID", "CONST", "NOCOPY", "NOCOPYPOST", "PACKED", "VARSIZE", "ENTRY",
  "FOR", "FORALL", "WHILE", "WHEN", "OVERLAP", "SERIAL", "IF", "ELSE",
  "PYTHON", "LOCAL", "NAMESPACE", "USING", "IDENT", "NUMBER", "LITERAL",
  "CPROGRAM", "HASHIF", "HASHIFDEF", "INT", "LONG", "SHORT", "CHAR",
  "FLOAT", "DOUBLE", "UNSIGNED", "ACCEL", "READWRITE", "WRITEONLY",
  "ACCELBLOCK", "MEMCRITICAL", "REDUCTIONTARGET", "CASE", "TYPENAME",
  "';'", "':'", "'{'", "'}'", "','", "'<'", "'>'", "'*'", "'('", "')'",
  "'&'", "'.'", "'['", "']'", "'='", "'-'", "$accept", "File",
  "ModuleEList", "OptExtern", "OneOrMoreSemiColon", "OptSemiColon", "Name",
  "QualName", "Module", "ConstructEList", "ConstructList", "ConstructSemi",
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
     325,   326,   327,   328,   329,   330,   331,    59,    58,   123,
     125,    44,    60,    62,    42,    40,    41,    38,    46,    91,
      93,    61,    45
};
# endif

#define YYPACT_NINF -568

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-568)))

#define YYTABLE_NINF -338

#define yytable_value_is_error(Yytable_value) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
     145,  1325,  1325,    58,  -568,   145,  -568,  -568,  -568,  -568,
    -568,  -568,  -568,  -568,  -568,  -568,  -568,  -568,  -568,  -568,
    -568,  -568,  -568,  -568,  -568,  -568,  -568,  -568,  -568,  -568,
    -568,  -568,  -568,  -568,  -568,  -568,  -568,  -568,  -568,  -568,
    -568,  -568,  -568,  -568,  -568,  -568,  -568,  -568,  -568,  -568,
    -568,  -568,  -568,  -568,  -568,    46,    46,  -568,  -568,  -568,
     886,    27,  -568,  -568,  -568,   165,  1325,   129,  1325,  1325,
     190,  1086,    56,   473,   886,  -568,  -568,  -568,  -568,   278,
      88,   132,  -568,   128,  -568,  -568,  -568,    27,     2,  1366,
     180,   180,    23,    -3,   134,   134,   134,   134,   137,   140,
    1325,   176,   160,   886,  -568,  -568,  -568,  -568,  -568,  -568,
    -568,  -568,   395,  -568,  -568,  -568,  -568,   173,  -568,  -568,
    -568,  -568,  -568,  -568,  -568,  -568,  -568,  -568,  -568,    27,
    -568,  -568,  -568,  1216,  1473,   886,   128,   169,   120,     2,
     188,  1505,  -568,  1490,  -568,    -8,  -568,  -568,  -568,  -568,
     336,   132,    68,  -568,  -568,   187,   207,   227,  -568,    45,
     132,  -568,   132,   132,   248,   132,   246,  -568,    89,  1325,
    1325,  1325,  1325,   124,   247,   249,   170,  1325,  -568,  -568,
    -568,  1387,   268,   134,   134,   134,   134,   247,   140,  -568,
    -568,  -568,  -568,  -568,    27,  -568,  -568,  -568,  -568,  -568,
    -568,  -568,  -568,  -568,  -568,  -568,  -568,  -568,  -568,  -568,
     292,  -568,  -568,  -568,   261,   274,  1473,   187,   207,   227,
      77,  -568,    -3,   277,    22,     2,   299,     2,   273,  -568,
     173,   279,    13,  -568,  -568,  -568,   177,  -568,  -568,    68,
    1423,  -568,  -568,  -568,  -568,  -568,   280,   237,   281,    40,
      93,   109,   276,   163,    -3,  -568,  -568,   282,   287,   289,
     296,   296,   296,   296,  -568,  1325,   290,   300,   294,    82,
    1325,   330,  1325,  -568,  -568,   295,   309,   334,   800,    -4,
      72,  1325,   338,   337,   173,  1325,  1325,  1325,  1325,  1325,
    1325,  -568,  -568,  -568,  1216,   391,  -568,   244,   342,  1325,
    -568,  -568,  -568,   351,   356,   346,   348,     2,    27,   132,
    -568,  -568,  -568,  -568,  -568,   360,  -568,   372,  -568,  1325,
     364,   374,   379,  -568,   377,  -568,     2,   180,  1423,   180,
     180,  1423,   180,  -568,  -568,    89,  -568,    -3,   218,   218,
     218,   218,   381,  -568,   330,  -568,   296,   296,  -568,   170,
      10,   384,   382,   152,   388,   146,  -568,   386,  1387,  -568,
    -568,   296,   296,   296,   296,   296,   235,  -568,   390,   401,
     402,   289,     2,   299,     2,     2,  -568,    40,  1423,  -568,
     405,   404,   406,  -568,  -568,   403,  -568,   409,   422,   423,
     132,   428,   427,  -568,   434,  -568,   369,    27,  -568,  -568,
    -568,  -568,  -568,  -568,   218,   218,  -568,  -568,  -568,  1490,
      15,   437,   430,  1490,  -568,  -568,   432,  -568,  -568,  -568,
    -568,  -568,   218,   218,   218,   218,   218,   504,    27,   435,
     436,  -568,   440,  -568,  -568,  -568,  -568,  -568,  -568,   442,
     441,  -568,  -568,  -568,  -568,   446,  -568,   136,   449,  -568,
      -3,  -568,   721,   479,   455,   173,   369,  -568,  -568,  -568,
    -568,  1325,  -568,  -568,  1325,  -568,   480,  -568,  -568,  -568,
    -568,  -568,   459,   452,  -568,  1408,  -568,  1458,  -568,   180,
     180,   180,  -568,  1162,  1106,  -568,   173,    27,  -568,   454,
     382,   382,   173,  -568,  1490,  1490,  -568,  1325,     2,   460,
     462,   463,   465,   481,   482,   456,   484,   440,  1325,  -568,
     475,   173,  -568,  -568,    27,  1325,     2,     2,    31,   476,
    1458,  -568,  -568,  -568,  -568,  -568,   534,   260,   440,  -568,
      27,   483,   485,   486,  -568,   225,  -568,  -568,  -568,  1325,
    -568,   488,   489,   488,   523,   501,   524,   488,   502,   511,
      27,     2,  -568,  -568,  -568,   563,  -568,  -568,  -568,  -568,
     128,  -568,   440,  -568,     2,   531,     2,    81,   507,   623,
     632,  -568,   510,     2,   378,   512,   453,   188,   506,   260,
     509,  -568,   516,   505,   514,  -568,     2,   523,   573,  -568,
     521,   615,     2,   514,   488,   513,   488,   526,   524,   488,
     527,     2,   528,   378,  -568,   173,  -568,   173,   546,  -568,
     380,   510,     2,   488,  -568,   836,   403,  -568,  -568,   530,
    -568,  -568,   188,   875,     2,   551,     2,   632,   510,     2,
     378,   188,  -568,  -568,  -568,  -568,  -568,  -568,  -568,  -568,
    -568,  1325,   533,   532,   522,     2,   537,     2,   511,  -568,
     440,  -568,   173,   511,   574,   549,   538,   514,   547,     2,
     514,   550,   173,   552,  1490,  1350,  -568,   188,     2,   557,
     556,  -568,  -568,   558,   925,  -568,     2,   488,   932,  -568,
     188,   941,  -568,  -568,  1325,  1325,     2,   555,  -568,  1325,
     514,     2,  -568,   574,   511,  -568,   562,     2,   511,  -568,
     173,   511,   574,  -568,   211,   138,   539,  1325,   173,   948,
     560,  -568,   564,     2,   565,   566,  -568,   569,  -568,  -568,
    1325,  1325,  1252,   559,  1325,  -568,   264,    27,   511,  -568,
       2,  -568,   514,     2,  -568,   574,   174,  -568,   567,   304,
    1325,   376,  -568,   570,   514,   998,   575,  -568,  -568,  -568,
    -568,  -568,  -568,  -568,  1005,   511,  -568,     2,   511,  -568,
     586,   514,   594,  -568,  1012,  -568,   511,  -568,   605,  -568
};

  /* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
     Performed when YYTABLE does not specify something else to do.  Zero
     means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       3,     0,     0,     0,     2,     3,    12,    13,    14,    15,
      16,    17,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    31,    34,    35,    36,    37,
      38,    39,    40,    41,    32,    33,    42,    43,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    54,    11,
      55,    56,    57,    58,    59,     0,     0,     1,     4,     7,
       0,    65,    63,    64,    87,     6,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    86,    84,    85,     8,     0,
       0,     0,    60,    70,   385,   386,   301,   263,   294,     0,
     150,   150,   150,     0,   158,   158,   158,   158,     0,   152,
       0,     0,     0,     0,    78,   224,   225,    72,    79,    80,
      81,    82,     0,    83,    71,   227,   226,     9,   258,   250,
     251,   252,   253,   254,   256,   257,   255,   248,   249,    76,
      77,    68,   267,     0,     0,     0,    69,     0,   295,   294,
       0,     0,   111,     0,    97,    98,    99,   100,   108,   109,
       0,     0,    95,   115,   116,   121,   122,   123,   124,   143,
       0,   151,     0,     0,     0,     0,   240,   228,     0,     0,
       0,     0,     0,     0,     0,   165,     0,     0,   230,   242,
     229,     0,     0,   158,   158,   158,   158,     0,   152,   215,
     216,   217,   218,   219,    10,    66,   287,   270,   271,   272,
     273,   279,   280,   281,   286,   274,   275,   276,   277,   278,
     162,   282,   284,   285,     0,   268,     0,   127,   128,   129,
     137,   264,     0,     0,     0,   294,   291,   294,     0,   302,
       0,     0,   125,   107,   110,   101,   102,   105,   106,    95,
      93,   113,   117,   118,   119,   126,     0,   142,     0,   146,
     234,   231,     0,   236,     0,   169,   170,     0,   160,    95,
     181,   181,   181,   181,   164,     0,     0,   167,     0,     0,
       0,     0,     0,   156,   157,     0,   154,   178,     0,     0,
     124,     0,   212,     0,     9,     0,     0,     0,     0,     0,
       0,   163,   283,   266,     0,   130,   131,   136,     0,     0,
      75,    62,    61,     0,   292,     0,     0,   294,   262,     0,
     103,   104,   114,    89,    90,    91,    94,     0,    88,     0,
     141,     0,     0,   383,   146,   148,   294,   150,     0,   150,
     150,     0,   150,   241,   159,     0,   112,     0,     0,     0,
       0,     0,     0,   190,     0,   166,   181,   181,   153,     0,
     171,     0,   200,    60,     0,     0,   210,   202,     0,   214,
      74,   181,   181,   181,   181,   181,     0,   269,   135,     0,
       0,    95,   294,   291,   294,   294,   299,   146,     0,    96,
       0,     0,     0,   140,   147,     0,   144,     0,     0,     0,
       0,     0,     0,   161,   183,   182,     0,   220,   185,   186,
     187,   188,   189,   168,     0,     0,   155,   172,   179,     0,
     171,     0,     0,     0,   208,   209,     0,   203,   204,   205,
     211,   213,     0,     0,     0,     0,     0,   171,   198,     0,
       0,   134,     0,   297,   293,   298,   296,   149,    92,     0,
       0,   139,   384,   145,   235,     0,   232,     0,     0,   237,
       0,   247,     0,     0,     0,     0,     0,   243,   244,   191,
     192,     0,   177,   180,     0,   201,     0,   193,   194,   195,
     196,   197,     0,     0,   133,     0,    73,     0,   138,   150,
     150,   150,   184,     0,     0,   245,     9,   246,   223,   173,
     200,   200,     0,   132,     0,     0,   327,   303,   294,   322,
       0,     0,     0,     0,     0,     0,    60,     0,     0,   221,
       0,     0,   206,   207,   199,     0,   294,   294,   171,     0,
       0,   326,   120,   233,   239,   238,     0,     0,     0,   174,
     175,     0,     0,     0,   300,     0,   304,   306,   323,     0,
     372,     0,     0,     0,     0,     0,   343,     0,     0,     0,
     332,   294,   260,   361,   333,   330,   307,   308,   289,   288,
     290,   305,     0,   378,   294,     0,   294,     0,   381,     0,
       0,   342,     0,   294,     0,     0,     0,     0,     0,     0,
       0,   376,     0,     0,     0,   379,   294,     0,     0,   345,
       0,     0,   294,     0,     0,     0,     0,     0,   343,     0,
       0,   294,     0,   339,   341,     9,   336,     9,     0,   259,
       0,     0,   294,     0,   377,     0,     0,   382,   344,     0,
     360,   338,     0,     0,   294,     0,   294,     0,     0,   294,
       0,     0,   362,   340,   334,   371,   331,   309,   310,   311,
     329,     0,     0,   324,     0,   294,     0,   294,     0,   369,
       0,   346,     9,     0,   373,     0,     0,     0,     0,   294,
       0,     0,     9,     0,     0,     0,   328,     0,   294,     0,
       0,   380,   359,     0,     0,   367,   294,     0,     0,   348,
       0,     0,   349,   358,     0,     0,   294,     0,   325,     0,
       0,   294,   370,   373,     0,   374,     0,   294,     0,   356,
       9,     0,   373,   312,     0,     0,     0,     0,     0,     0,
       0,   368,     0,   294,     0,     0,   347,     0,   354,   320,
       0,     0,     0,     0,     0,   318,     0,   261,     0,   364,
     294,   375,     0,   294,   357,   373,     0,   314,     0,     0,
       0,     0,   321,     0,     0,     0,     0,   355,   317,   316,
     315,   313,   319,   363,     0,     0,   351,   294,     0,   365,
       0,     0,     0,   350,     0,   366,     0,   352,     0,   353
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -568,  -568,   640,  -568,   -53,  -279,    -1,   -60,   583,   630,
     -49,  -568,  -568,  -568,  -265,  -568,  -220,  -568,  -130,   -85,
    -123,  -124,  -122,  -168,   544,   477,  -568,   -80,  -568,  -568,
    -280,  -568,  -568,   -76,   503,   340,  -568,    12,   357,  -568,
    -568,   525,   352,  -568,   179,  -568,  -568,  -283,  -568,   -89,
     251,  -568,  -568,  -568,  -131,  -568,  -568,  -568,  -568,  -568,
    -568,  -329,   347,  -568,   341,   634,  -568,  -103,   252,   638,
    -568,  -568,   458,  -568,  -568,  -568,  -568,   250,  -568,   226,
     262,   419,  -568,  -568,   343,   -82,  -472,   -66,  -546,  -568,
    -568,  -492,  -568,  -568,  -439,    50,  -493,  -568,  -568,   139,
    -542,    90,  -558,   133,  -541,  -568,  -499,  -567,  -537,  -543,
    -517,  -568,   150,   171,   111,  -568,  -568
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     3,     4,    71,   397,   195,   259,   152,     5,    62,
      72,    73,    74,   315,   316,   317,   241,   153,   260,   154,
     155,   156,   157,   158,   159,   220,   221,   318,   385,   324,
     325,   105,   106,   162,   177,   275,   276,   169,   257,   292,
     267,   174,   268,   258,   409,   511,   410,   411,   107,   338,
     395,   108,   109,   110,   175,   111,   189,   190,   191,   192,
     193,   414,   356,   282,   283,   453,   113,   398,   454,   455,
     115,   116,   167,   180,   456,   457,   130,   458,    75,   222,
     134,   214,   215,   561,   305,   581,   498,   551,   230,   499,
     642,   704,   687,   643,   500,   644,   476,   611,   579,   552,
     575,   590,   602,   572,   553,   604,   576,   675,   582,   615,
     564,   568,   569,   326,   443,    76,    77
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[] =
{
      55,    56,    61,    61,    88,   360,   140,    83,   166,   160,
     218,   217,   219,   280,   527,   163,   165,    87,   231,   312,
     129,   136,   516,   517,   415,   131,   566,   407,   554,   584,
     573,   607,   407,   603,   606,   555,   593,   301,   501,   336,
     261,   262,   263,   270,   384,   633,   619,   277,   407,   621,
     623,   245,   353,    82,   182,   233,   289,   228,    57,   234,
     161,   138,   603,   388,   194,    80,   391,    84,    85,   580,
     589,   591,   661,   151,   585,   646,   652,   624,   302,   626,
     554,   538,   629,   245,   354,   662,   223,   139,   266,   603,
     408,   239,   218,   217,   219,  -176,   647,   437,   246,   178,
     249,   281,   250,   251,    78,   253,   670,   170,   171,   172,
     669,   673,   164,   438,   678,   296,   649,   681,   137,   255,
     534,   689,   535,    59,   654,    60,   711,   462,   591,   323,
     246,   690,   247,   248,   700,   718,   117,   299,    82,   265,
     346,   256,   347,   303,   472,   306,   137,   709,     1,     2,
     240,   432,   712,  -202,   710,  -202,   715,   671,   151,   717,
     697,   512,   513,   355,   297,   298,   475,   135,   747,   166,
     323,   137,   339,   340,   341,   695,   264,   308,   327,   699,
      82,   265,   702,    81,   266,    82,   743,   137,    82,   745,
     280,   328,   686,   744,   329,   285,   286,   287,   288,   225,
     151,   754,    82,   417,   418,   226,   137,   509,    79,   227,
     729,   273,   274,   760,   137,   726,   762,   161,   764,   239,
     724,   480,   151,   168,   768,   376,   173,   534,   736,   176,
     739,   194,   741,  -200,   179,  -200,   399,   400,   401,   310,
     311,   137,   181,   413,   386,   331,   756,   224,   332,   377,
      59,   387,   394,   389,   390,   759,   392,   404,   405,   720,
     748,   540,   721,   722,   342,   767,   723,    59,   229,    86,
     419,   242,   422,   423,   424,   425,   426,   352,   281,   132,
     357,    82,   558,   559,   361,   362,   363,   364,   365,   366,
     433,   243,   435,   436,   719,    59,   720,   396,   371,   721,
     722,   459,   460,   723,   541,   542,   543,   544,   545,   546,
     547,   244,    59,   428,   427,  -265,  -265,   252,   380,   467,
     468,   469,   470,   471,   320,   321,   634,   254,   635,   461,
     447,   368,   369,   465,  -265,   548,   269,    59,   271,   549,
    -265,  -265,  -265,  -265,  -265,  -265,  -265,   742,   284,   720,
     291,   293,   721,   722,  -265,   294,   723,   300,   304,   307,
     218,   217,   219,   309,   319,   394,   330,   133,   335,   322,
     451,   240,   334,   672,   337,    89,    90,    91,    92,    93,
     343,   344,   264,   683,   345,   348,   637,   100,   101,   720,
     349,   102,   721,   722,   750,   497,   723,   497,   235,   236,
     237,   238,   487,   502,   503,   504,   183,   184,   185,   186,
     187,   188,   452,   350,   515,   515,   519,   142,   143,   358,
     359,   716,   594,   595,   596,   544,   597,   598,   599,   296,
     370,   372,   374,   194,   532,   533,    82,   373,   375,   514,
     497,   378,   144,   145,   146,   147,   148,   149,   150,  -222,
     638,   639,   381,   600,   540,   379,   151,    86,   530,   752,
     489,   720,   382,   490,   721,   722,   323,   383,   723,   577,
     640,   402,   412,   413,   550,   560,   416,   355,   429,   118,
     119,   120,   121,   507,   122,   123,   124,   125,   126,   430,
     431,   439,   440,   442,   441,   444,   518,   541,   542,   543,
     544,   545,   546,   547,   616,   445,   592,   528,   601,   446,
     622,   448,   540,   449,   531,   450,   127,   463,   464,   631,
     466,   407,   452,   473,   474,   475,   550,   477,   548,   478,
     641,   479,    86,  -335,   481,   486,   491,   601,   562,   492,
     493,   520,   655,   510,   657,   645,   526,   660,   521,   522,
      59,   523,   194,   128,   194,   541,   542,   543,   544,   545,
     546,   547,   659,   667,   601,   529,   537,   524,   525,   -11,
    -301,   539,   534,   563,   540,   556,   557,   680,   565,   567,
     570,   574,   571,   578,   685,   641,   548,   583,   587,    86,
      86,  -301,   605,   612,   696,   613,  -301,   608,   610,   194,
     614,   620,   625,   636,   706,   627,   630,   656,   632,   194,
     651,   664,   666,   665,   668,   714,   540,   541,   542,   543,
     544,   545,   546,   547,   540,   674,   676,   679,   677,   725,
     682,   732,  -301,   540,   684,   691,   692,   707,   693,   713,
     663,   730,   740,   733,   731,    58,   734,   194,   548,   735,
     753,   746,    86,   618,   104,   727,   757,   749,  -301,   541,
     542,   543,   544,   545,   546,   547,   763,   541,   542,   543,
     544,   545,   546,   547,   765,   761,   541,   542,   543,   544,
     545,   546,   547,   703,   705,   769,    63,   232,   708,   406,
     548,   290,   393,   295,    86,  -337,   403,   536,   548,   421,
     272,   482,   588,   485,   420,   112,   703,   548,   488,   114,
     508,    86,   333,   367,   484,   688,   434,   658,   609,   703,
     737,   703,   132,   703,  -265,  -265,  -265,   650,  -265,  -265,
    -265,   628,  -265,  -265,  -265,  -265,  -265,   617,   586,   751,
    -265,  -265,  -265,  -265,  -265,  -265,  -265,  -265,  -265,  -265,
    -265,  -265,     0,  -265,  -265,  -265,  -265,  -265,  -265,  -265,
    -265,  -265,  -265,  -265,  -265,  -265,  -265,  -265,  -265,  -265,
    -265,  -265,  -265,     0,  -265,     0,  -265,  -265,     0,     0,
       0,     0,     0,  -265,  -265,  -265,  -265,  -265,  -265,  -265,
    -265,     0,     0,  -265,  -265,  -265,  -265,  -265,     0,     0,
       0,     0,     0,     6,     7,     8,     0,     9,    10,    11,
     483,    12,    13,    14,    15,    16,     0,     0,     0,    17,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      28,     0,    29,    30,    31,    32,    33,   540,     0,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,     0,    47,     0,    48,    49,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    50,
       0,     0,    51,    52,    53,    54,   540,     0,     0,     0,
     541,   542,   543,   544,   545,   546,   547,    64,   351,    -5,
      -5,    65,    -5,    -5,    -5,    -5,    -5,    -5,    -5,    -5,
      -5,    -5,    -5,     0,    -5,    -5,     0,     0,    -5,     0,
       0,   548,     0,     0,     0,   648,     0,     0,     0,   541,
     542,   543,   544,   545,   546,   547,   540,     0,     0,     0,
       0,     0,     0,   540,     0,     0,     0,     0,     0,     0,
      66,    67,   540,     0,     0,     0,    68,    69,     0,   540,
     548,     0,     0,     0,   653,     0,     0,     0,    70,     0,
       0,     0,     0,     0,     0,    -5,   -67,     0,     0,   541,
     542,   543,   544,   545,   546,   547,   541,   542,   543,   544,
     545,   546,   547,     0,     0,   541,   542,   543,   544,   545,
     546,   547,   541,   542,   543,   544,   545,   546,   547,   540,
     548,     0,     0,     0,   694,     0,   540,   548,     0,     0,
       0,   698,     0,   540,     0,     0,   548,     0,     0,     0,
     701,     0,     0,   548,     0,     0,     0,   728,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   541,   542,   543,   544,   545,   546,   547,   541,
     542,   543,   544,   545,   546,   547,   541,   542,   543,   544,
     545,   546,   547,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   548,     0,     0,     0,   755,     0,     0,
     548,     0,     0,     0,   758,     0,     0,   548,     0,     1,
       2,   766,    89,    90,    91,    92,    93,    94,    95,    96,
      97,    98,    99,     0,   100,   101,     0,     0,   102,     6,
       7,     8,     0,     9,    10,    11,     0,    12,    13,    14,
      15,    16,     0,     0,     0,    17,    18,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,     0,    29,    30,
      31,    32,    33,   142,   216,    34,    35,    36,    37,    38,
      39,    40,    41,    42,    43,    44,    45,    46,     0,    47,
       0,    48,   506,   196,     0,   103,     0,     0,   144,   145,
     146,   147,   148,   149,   150,    50,     0,     0,    51,    52,
      53,    54,   151,   197,     0,   198,   199,   200,   201,   202,
     203,     0,     0,   204,   205,   206,   207,   208,   209,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   210,   211,     0,   196,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   505,     0,     0,     0,   212,   213,   197,     0,   198,
     199,   200,   201,   202,   203,     0,     0,   204,   205,   206,
     207,   208,   209,     0,     0,     6,     7,     8,     0,     9,
      10,    11,     0,    12,    13,    14,    15,    16,   210,   211,
       0,    17,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,     0,    29,    30,    31,    32,    33,   212,
     213,    34,    35,    36,    37,    38,    39,    40,    41,    42,
      43,    44,    45,    46,     0,    47,     0,    48,    49,   738,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    50,     0,     0,    51,    52,    53,    54,     6,     7,
       8,     0,     9,    10,    11,     0,    12,    13,    14,    15,
      16,     0,     0,     0,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,   637,    29,    30,    31,
      32,    33,     0,     0,    34,    35,    36,    37,    38,    39,
      40,    41,    42,    43,    44,    45,    46,     0,    47,     0,
      48,    49,   141,     0,     0,     0,     0,   142,   143,     0,
       0,     0,     0,     0,    50,     0,     0,    51,    52,    53,
      54,     0,     0,   142,   143,   278,    82,     0,     0,     0,
       0,     0,   144,   145,   146,   147,   148,   149,   150,     0,
     638,   639,    82,     0,   142,   143,   151,     0,   144,   145,
     146,   147,   148,   149,   150,     0,     0,     0,     0,     0,
       0,     0,   151,    82,     0,   142,   143,   494,   495,   144,
     145,   146,   147,   148,   149,   150,     0,     0,     0,     0,
     142,   143,     0,   279,    82,     0,     0,     0,     0,     0,
     144,   145,   146,   147,   148,   149,   150,     0,     0,    82,
     313,   314,     0,     0,   151,   144,   145,   146,   147,   148,
     149,   150,     0,     0,   496,   142,   143,   494,   495,   151,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     142,   216,     0,     0,    82,     0,     0,     0,     0,     0,
     144,   145,   146,   147,   148,   149,   150,   142,   143,    82,
       0,     0,     0,     0,   151,   144,   145,   146,   147,   148,
     149,   150,   142,     0,     0,     0,    82,     0,     0,   151,
       0,     0,   144,   145,   146,   147,   148,   149,   150,     0,
       0,    82,     0,     0,     0,     0,   151,   144,   145,   146,
     147,   148,   149,   150,     0,     0,     0,     0,     0,     0,
       0,   151
};

static const yytype_int16 yycheck[] =
{
       1,     2,    55,    56,    70,   284,    88,    67,    93,    89,
     134,   134,   134,   181,   507,    91,    92,    70,   141,   239,
      73,    81,   494,   495,   353,    74,   543,    17,   527,   566,
     547,   577,    17,   574,   576,   528,   573,    15,   477,   259,
     170,   171,   172,   174,   324,   603,   588,   177,    17,   591,
     593,    38,    56,    56,   103,    63,   187,   139,     0,    67,
      37,    59,   603,   328,   117,    66,   331,    68,    69,   562,
     569,   570,   630,    76,   567,   612,   622,   594,    56,   596,
     579,   520,   599,    38,    88,   631,   135,    85,   173,   630,
      80,   151,   216,   216,   216,    80,   613,   377,    85,   100,
     160,   181,   162,   163,    77,   165,   648,    95,    96,    97,
     647,   653,    89,   378,   657,    38,   615,   660,    78,    30,
      89,   667,    91,    77,   623,    79,   693,   410,   627,    89,
      85,   668,    87,    88,   680,   702,    80,   222,    56,    57,
     270,    52,   272,   225,   427,   227,    78,   690,     3,     4,
      82,   371,   694,    81,   691,    83,   698,   650,    76,   701,
     677,   490,   491,    91,    87,    88,    85,    79,   735,   254,
      89,    78,   261,   262,   263,   674,    52,   230,    85,   678,
      56,    57,   681,    54,   269,    56,   728,    78,    56,   732,
     358,    82,   664,   730,    85,   183,   184,   185,   186,    79,
      76,   744,    56,    57,    58,    85,    78,   486,    43,    89,
     709,    41,    42,   755,    78,   707,   758,    37,   761,   279,
      82,    85,    76,    89,   766,   307,    89,    89,   720,    89,
     722,   284,   724,    81,    58,    83,   339,   340,   341,    62,
      63,    78,    82,    91,   326,    82,   745,    78,    85,   309,
      77,   327,   337,   329,   330,   754,   332,   346,   347,    85,
      86,     1,    88,    89,   265,   764,    92,    77,    80,    79,
     355,    84,   361,   362,   363,   364,   365,   278,   358,     1,
     281,    56,    57,    58,   285,   286,   287,   288,   289,   290,
     372,    84,   374,   375,    83,    77,    85,    79,   299,    88,
      89,   404,   405,    92,    44,    45,    46,    47,    48,    49,
      50,    84,    77,   366,    79,    37,    38,    69,   319,   422,
     423,   424,   425,   426,    87,    88,   605,    81,   607,   409,
     390,    87,    88,   413,    56,    75,    89,    77,    89,    79,
      62,    63,    64,    65,    66,    67,    68,    83,    80,    85,
      58,    90,    88,    89,    76,    81,    92,    80,    59,    86,
     484,   484,   484,    84,    84,   450,    90,    89,    81,    88,
       1,    82,    90,   652,    78,     6,     7,     8,     9,    10,
      90,    81,    52,   662,    90,    90,     6,    18,    19,    85,
      81,    22,    88,    89,    90,   475,    92,   477,    62,    63,
      64,    65,   455,   479,   480,   481,    11,    12,    13,    14,
      15,    16,    43,    79,   494,   495,   498,    37,    38,    81,
      83,   700,    44,    45,    46,    47,    48,    49,    50,    38,
      88,    80,    86,   486,   516,   517,    56,    81,    90,   492,
     520,    81,    62,    63,    64,    65,    66,    67,    68,    80,
      70,    71,    88,    75,     1,    83,    76,    79,   511,    83,
     461,    85,    88,   464,    88,    89,    89,    88,    92,   551,
      90,    90,    88,    91,   527,   535,    88,    91,    88,     6,
       7,     8,     9,   484,    11,    12,    13,    14,    15,    88,
      88,    86,    88,    90,    88,    86,   497,    44,    45,    46,
      47,    48,    49,    50,   586,    83,   572,   508,   574,    86,
     592,    83,     1,    86,   515,    81,    43,    80,    88,   601,
      88,    17,    43,    88,    88,    85,   579,    85,    75,    88,
     610,    85,    79,    80,    85,    80,    56,   603,   539,    80,
      88,    81,   624,    89,   626,   611,    90,   629,    86,    86,
      77,    86,   605,    80,   607,    44,    45,    46,    47,    48,
      49,    50,   628,   645,   630,    90,    90,    86,    86,    85,
      59,    37,    89,    85,     1,    90,    90,   659,    89,    56,
      79,    79,    58,    20,   664,   665,    75,    56,    81,    79,
      79,    80,    80,    77,   676,    90,    85,    91,    89,   652,
      86,    80,    89,    57,   686,    79,    79,    56,    80,   662,
      80,    78,    90,    81,    77,   697,     1,    44,    45,    46,
      47,    48,    49,    50,     1,    51,    77,    80,    90,    90,
      80,   713,    59,     1,    82,    78,    80,    82,    80,    77,
     641,    81,    83,    78,    80,     5,    80,   700,    75,    80,
      80,   733,    79,    80,    71,   708,    81,    90,    85,    44,
      45,    46,    47,    48,    49,    50,    80,    44,    45,    46,
      47,    48,    49,    50,    80,   757,    44,    45,    46,    47,
      48,    49,    50,   684,   685,    80,    56,   143,   689,   349,
      75,   188,   335,   216,    79,    80,   344,   518,    75,   358,
     175,   450,    79,   453,   357,    71,   707,    75,   456,    71,
     484,    79,   254,   294,   452,   665,   373,   627,   579,   720,
     721,   722,     1,   724,     3,     4,     5,   616,     7,     8,
       9,   598,    11,    12,    13,    14,    15,   587,   567,   740,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    -1,    32,    33,    34,    35,    36,    37,    38,
      39,    40,    41,    42,    43,    44,    45,    46,    47,    48,
      49,    50,    51,    -1,    53,    -1,    55,    56,    -1,    -1,
      -1,    -1,    -1,    62,    63,    64,    65,    66,    67,    68,
      69,    -1,    -1,    72,    73,    74,    75,    76,    -1,    -1,
      -1,    -1,    -1,     3,     4,     5,    -1,     7,     8,     9,
      89,    11,    12,    13,    14,    15,    -1,    -1,    -1,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    -1,    32,    33,    34,    35,    36,     1,    -1,    39,
      40,    41,    42,    43,    44,    45,    46,    47,    48,    49,
      50,    51,    -1,    53,    -1,    55,    56,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    69,
      -1,    -1,    72,    73,    74,    75,     1,    -1,    -1,    -1,
      44,    45,    46,    47,    48,    49,    50,     1,    88,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      14,    15,    16,    -1,    18,    19,    -1,    -1,    22,    -1,
      -1,    75,    -1,    -1,    -1,    79,    -1,    -1,    -1,    44,
      45,    46,    47,    48,    49,    50,     1,    -1,    -1,    -1,
      -1,    -1,    -1,     1,    -1,    -1,    -1,    -1,    -1,    -1,
      54,    55,     1,    -1,    -1,    -1,    60,    61,    -1,     1,
      75,    -1,    -1,    -1,    79,    -1,    -1,    -1,    72,    -1,
      -1,    -1,    -1,    -1,    -1,    79,    80,    -1,    -1,    44,
      45,    46,    47,    48,    49,    50,    44,    45,    46,    47,
      48,    49,    50,    -1,    -1,    44,    45,    46,    47,    48,
      49,    50,    44,    45,    46,    47,    48,    49,    50,     1,
      75,    -1,    -1,    -1,    79,    -1,     1,    75,    -1,    -1,
      -1,    79,    -1,     1,    -1,    -1,    75,    -1,    -1,    -1,
      79,    -1,    -1,    75,    -1,    -1,    -1,    79,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    44,    45,    46,    47,    48,    49,    50,    44,
      45,    46,    47,    48,    49,    50,    44,    45,    46,    47,
      48,    49,    50,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    75,    -1,    -1,    -1,    79,    -1,    -1,
      75,    -1,    -1,    -1,    79,    -1,    -1,    75,    -1,     3,
       4,    79,     6,     7,     8,     9,    10,    11,    12,    13,
      14,    15,    16,    -1,    18,    19,    -1,    -1,    22,     3,
       4,     5,    -1,     7,     8,     9,    -1,    11,    12,    13,
      14,    15,    -1,    -1,    -1,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,    29,    30,    -1,    32,    33,
      34,    35,    36,    37,    38,    39,    40,    41,    42,    43,
      44,    45,    46,    47,    48,    49,    50,    51,    -1,    53,
      -1,    55,    56,     1,    -1,    79,    -1,    -1,    62,    63,
      64,    65,    66,    67,    68,    69,    -1,    -1,    72,    73,
      74,    75,    76,    21,    -1,    23,    24,    25,    26,    27,
      28,    -1,    -1,    31,    32,    33,    34,    35,    36,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    52,    53,    -1,     1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    69,    -1,    -1,    -1,    73,    74,    21,    -1,    23,
      24,    25,    26,    27,    28,    -1,    -1,    31,    32,    33,
      34,    35,    36,    -1,    -1,     3,     4,     5,    -1,     7,
       8,     9,    -1,    11,    12,    13,    14,    15,    52,    53,
      -1,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      28,    29,    30,    -1,    32,    33,    34,    35,    36,    73,
      74,    39,    40,    41,    42,    43,    44,    45,    46,    47,
      48,    49,    50,    51,    -1,    53,    -1,    55,    56,    57,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    69,    -1,    -1,    72,    73,    74,    75,     3,     4,
       5,    -1,     7,     8,     9,    -1,    11,    12,    13,    14,
      15,    -1,    -1,    -1,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,     6,    32,    33,    34,
      35,    36,    -1,    -1,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    -1,    53,    -1,
      55,    56,    16,    -1,    -1,    -1,    -1,    37,    38,    -1,
      -1,    -1,    -1,    -1,    69,    -1,    -1,    72,    73,    74,
      75,    -1,    -1,    37,    38,    18,    56,    -1,    -1,    -1,
      -1,    -1,    62,    63,    64,    65,    66,    67,    68,    -1,
      70,    71,    56,    -1,    37,    38,    76,    -1,    62,    63,
      64,    65,    66,    67,    68,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    76,    56,    -1,    37,    38,    39,    40,    62,
      63,    64,    65,    66,    67,    68,    -1,    -1,    -1,    -1,
      37,    38,    -1,    76,    56,    -1,    -1,    -1,    -1,    -1,
      62,    63,    64,    65,    66,    67,    68,    -1,    -1,    56,
      57,    58,    -1,    -1,    76,    62,    63,    64,    65,    66,
      67,    68,    -1,    -1,    86,    37,    38,    39,    40,    76,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      37,    38,    -1,    -1,    56,    -1,    -1,    -1,    -1,    -1,
      62,    63,    64,    65,    66,    67,    68,    37,    38,    56,
      -1,    -1,    -1,    -1,    76,    62,    63,    64,    65,    66,
      67,    68,    37,    -1,    -1,    -1,    56,    -1,    -1,    76,
      -1,    -1,    62,    63,    64,    65,    66,    67,    68,    -1,
      -1,    56,    -1,    -1,    -1,    -1,    76,    62,    63,    64,
      65,    66,    67,    68,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    76
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     3,     4,    94,    95,   101,     3,     4,     5,     7,
       8,     9,    11,    12,    13,    14,    15,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    32,
      33,    34,    35,    36,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    53,    55,    56,
      69,    72,    73,    74,    75,    99,    99,     0,    95,    77,
      79,    97,   102,   102,     1,     5,    54,    55,    60,    61,
      72,    96,   103,   104,   105,   171,   208,   209,    77,    43,
      99,    54,    56,   100,    99,    99,    79,    97,   180,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      18,    19,    22,    79,   101,   124,   125,   141,   144,   145,
     146,   148,   158,   159,   162,   163,   164,    80,     6,     7,
       8,     9,    11,    12,    13,    14,    15,    43,    80,    97,
     169,   103,     1,    89,   173,    79,   100,    78,    59,    85,
     178,    16,    37,    38,    62,    63,    64,    65,    66,    67,
      68,    76,   100,   110,   112,   113,   114,   115,   116,   117,
     120,    37,   126,   126,    89,   126,   112,   165,    89,   130,
     130,   130,   130,    89,   134,   147,    89,   127,    99,    58,
     166,    82,   103,    11,    12,    13,    14,    15,    16,   149,
     150,   151,   152,   153,    97,    98,     1,    21,    23,    24,
      25,    26,    27,    28,    31,    32,    33,    34,    35,    36,
      52,    53,    73,    74,   174,   175,    38,   113,   114,   115,
     118,   119,   172,   103,    78,    79,    85,    89,   178,    80,
     181,   113,   117,    63,    67,    62,    63,    64,    65,   100,
      82,   109,    84,    84,    84,    38,    85,    87,    88,   100,
     100,   100,    69,   100,    81,    30,    52,   131,   136,    99,
     111,   111,   111,   111,    52,    57,   112,   133,   135,    89,
     147,    89,   134,    41,    42,   128,   129,   111,    18,    76,
     116,   120,   156,   157,    80,   130,   130,   130,   130,   147,
     127,    58,   132,    90,    81,   118,    38,    87,    88,   112,
      80,    15,    56,   178,    59,   177,   178,    86,    97,    84,
      62,    63,   109,    57,    58,   106,   107,   108,   120,    84,
      87,    88,    88,    89,   122,   123,   206,    85,    82,    85,
      90,    82,    85,   165,    90,    81,   109,    78,   142,   142,
     142,   142,    99,    90,    81,    90,   111,   111,    90,    81,
      79,    88,    99,    56,    88,    91,   155,    99,    81,    83,
      98,    99,    99,    99,    99,    99,    99,   174,    87,    88,
      88,    99,    80,    81,    86,    90,   178,   100,    81,    83,
      99,    88,    88,    88,   123,   121,   178,   126,   107,   126,
     126,   107,   126,   131,   112,   143,    79,    97,   160,   160,
     160,   160,    90,   135,   142,   142,   128,    17,    80,   137,
     139,   140,    88,    91,   154,   154,    88,    57,    58,   112,
     155,   157,   142,   142,   142,   142,   142,    79,    97,    88,
      88,    88,   109,   178,   177,   178,   178,   123,   107,    86,
      88,    88,    90,   207,    86,    83,    86,   100,    83,    86,
      81,     1,    43,   158,   161,   162,   167,   168,   170,   160,
     160,   120,   140,    80,    88,   120,    88,   160,   160,   160,
     160,   160,   140,    88,    88,    85,   189,    85,    88,    85,
      85,    85,   143,    89,   173,   170,    80,    97,   161,    99,
      99,    56,    80,    88,    39,    40,    86,   120,   179,   182,
     187,   187,   126,   126,   126,    69,    56,    99,   172,    98,
      89,   138,   154,   154,    97,   120,   179,   179,    99,   178,
      81,    86,    86,    86,    86,    86,    90,   189,    99,    90,
      97,    99,   178,   178,    89,    91,   137,    90,   187,    37,
       1,    44,    45,    46,    47,    48,    49,    50,    75,    79,
      97,   180,   192,   197,   199,   189,    90,    90,    57,    58,
     100,   176,    99,    85,   203,    89,   203,    56,   204,   205,
      79,    58,   196,   203,    79,   193,   199,   178,    20,   191,
     189,   178,   201,    56,   201,   189,   206,    81,    79,   199,
     194,   199,   180,   201,    44,    45,    46,    48,    49,    50,
      75,   180,   195,   197,   198,    80,   193,   181,    91,   192,
      89,   190,    77,    90,    86,   202,   178,   205,    80,   193,
      80,   193,   178,   202,   203,    89,   203,    79,   196,   203,
      79,   178,    80,   195,    98,    98,    57,     6,    70,    71,
      90,   120,   183,   186,   188,   180,   201,   203,    79,   199,
     207,    80,   181,    79,   199,   178,    56,   178,   194,   180,
     178,   195,   181,    99,    78,    81,    90,   178,    77,   201,
     193,   189,    98,   193,    51,   200,    77,    90,   202,    80,
     178,   202,    80,    98,    82,   120,   179,   185,   188,   181,
     201,    78,    80,    80,    79,   199,   178,   203,    79,   199,
     181,    79,   199,    99,   184,    99,   178,    82,    99,   202,
     201,   200,   193,    77,   178,   193,    98,   193,   200,    83,
      85,    88,    89,    92,    82,    90,   184,    97,    79,   199,
      81,    80,   178,    78,    80,    80,   184,    99,    57,   184,
      83,   184,    83,   193,   201,   202,   178,   200,    86,    90,
      90,    99,    83,    80,   202,    79,   199,    81,    79,   199,
     193,   178,   193,    80,   202,    80,    79,   199,   193,    80
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    93,    94,    95,    95,    96,    96,    97,    97,    98,
      98,    99,    99,    99,    99,    99,    99,    99,    99,    99,
      99,    99,    99,    99,    99,    99,    99,    99,    99,    99,
      99,    99,    99,    99,    99,    99,    99,    99,    99,    99,
      99,    99,    99,    99,    99,    99,    99,    99,    99,    99,
      99,    99,    99,    99,    99,    99,    99,    99,    99,    99,
     100,   100,   100,   101,   101,   102,   102,   103,   103,   104,
     104,   104,   104,   104,   105,   105,   105,   105,   105,   105,
     105,   105,   105,   105,   105,   105,   105,   105,   106,   106,
     106,   107,   107,   108,   108,   109,   109,   110,   110,   110,
     110,   110,   110,   110,   110,   110,   110,   110,   110,   110,
     110,   110,   111,   112,   112,   113,   113,   114,   115,   115,
     116,   117,   117,   117,   117,   117,   117,   118,   118,   118,
     118,   118,   119,   119,   119,   119,   119,   119,   120,   120,
     120,   120,   120,   120,   121,   122,   123,   123,   124,   125,
     126,   126,   127,   127,   128,   128,   129,   129,   130,   130,
     131,   131,   132,   132,   133,   134,   134,   135,   135,   136,
     136,   137,   137,   138,   138,   139,   140,   140,   141,   141,
     141,   142,   142,   143,   143,   144,   144,   145,   146,   147,
     147,   148,   148,   149,   149,   150,   151,   152,   153,   153,
     154,   154,   155,   155,   155,   155,   156,   156,   156,   156,
     156,   156,   157,   157,   158,   159,   159,   159,   159,   159,
     160,   160,   161,   161,   162,   162,   162,   162,   162,   162,
     162,   163,   163,   163,   163,   163,   164,   164,   164,   164,
     165,   165,   166,   167,   168,   168,   168,   168,   169,   169,
     169,   169,   169,   169,   169,   169,   169,   169,   169,   170,
     170,   170,   171,   171,   172,   173,   173,   173,   174,   174,
     175,   175,   175,   175,   175,   175,   175,   175,   175,   175,
     175,   175,   175,   175,   175,   175,   175,   175,   176,   176,
     176,   177,   177,   177,   178,   178,   178,   178,   178,   178,
     179,   180,   181,   182,   182,   182,   182,   182,   182,   183,
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
       1,     4,     4,     3,     3,     1,     4,     0,     2,     3,
       2,     2,     2,     8,     5,     5,     2,     2,     2,     2,
       2,     2,     2,     2,     1,     1,     1,     1,     1,     1,
       1,     1,     3,     0,     1,     0,     3,     1,     1,     1,
       1,     2,     2,     3,     3,     2,     2,     2,     1,     1,
       2,     1,     2,     2,     3,     1,     1,     2,     2,     2,
       8,     1,     1,     1,     1,     2,     2,     1,     1,     1,
       2,     2,     6,     5,     4,     3,     2,     1,     6,     5,
       4,     3,     2,     1,     1,     3,     0,     2,     4,     6,
       0,     1,     0,     3,     1,     3,     1,     1,     0,     3,
       1,     3,     0,     1,     1,     0,     3,     1,     3,     1,
       1,     0,     1,     0,     2,     5,     1,     2,     3,     5,
       6,     0,     2,     1,     3,     5,     5,     5,     5,     4,
       3,     6,     6,     5,     5,     5,     5,     5,     4,     7,
       0,     2,     0,     2,     2,     2,     6,     6,     3,     3,
       2,     3,     1,     3,     4,     2,     2,     2,     2,     2,
       1,     4,     0,     2,     1,     1,     1,     1,     2,     2,
       2,     3,     6,     9,     3,     6,     3,     6,     9,     9,
       1,     3,     1,     1,     1,     2,     2,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     7,
       5,    13,     5,     2,     1,     0,     3,     1,     1,     3,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     2,     1,     1,     1,     1,     1,     1,
       1,     0,     1,     3,     0,     1,     5,     5,     5,     4,
       3,     1,     1,     1,     3,     4,     3,     4,     4,     1,
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
#line 198 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.modlist) = (yyvsp[0].modlist); modlist = (yyvsp[0].modlist); }
#line 2276 "y.tab.c" /* yacc.c:1646  */
    break;

  case 3:
#line 202 "xi-grammar.y" /* yacc.c:1646  */
    { 
		  (yyval.modlist) = 0; 
		}
#line 2284 "y.tab.c" /* yacc.c:1646  */
    break;

  case 4:
#line 206 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.modlist) = new AstChildren<Module>(lineno, (yyvsp[-1].module), (yyvsp[0].modlist)); }
#line 2290 "y.tab.c" /* yacc.c:1646  */
    break;

  case 5:
#line 210 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 2296 "y.tab.c" /* yacc.c:1646  */
    break;

  case 6:
#line 212 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 2302 "y.tab.c" /* yacc.c:1646  */
    break;

  case 7:
#line 216 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 2308 "y.tab.c" /* yacc.c:1646  */
    break;

  case 8:
#line 218 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 2; }
#line 2314 "y.tab.c" /* yacc.c:1646  */
    break;

  case 9:
#line 222 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 2320 "y.tab.c" /* yacc.c:1646  */
    break;

  case 10:
#line 224 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 2326 "y.tab.c" /* yacc.c:1646  */
    break;

  case 11:
#line 229 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 2332 "y.tab.c" /* yacc.c:1646  */
    break;

  case 12:
#line 230 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MODULE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2338 "y.tab.c" /* yacc.c:1646  */
    break;

  case 13:
#line 231 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MAINMODULE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2344 "y.tab.c" /* yacc.c:1646  */
    break;

  case 14:
#line 232 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(EXTERN, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2350 "y.tab.c" /* yacc.c:1646  */
    break;

  case 15:
#line 234 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITCALL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2356 "y.tab.c" /* yacc.c:1646  */
    break;

  case 16:
#line 235 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITNODE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2362 "y.tab.c" /* yacc.c:1646  */
    break;

  case 17:
#line 236 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITPROC, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2368 "y.tab.c" /* yacc.c:1646  */
    break;

  case 18:
#line 238 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CHARE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2374 "y.tab.c" /* yacc.c:1646  */
    break;

  case 19:
#line 239 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MAINCHARE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2380 "y.tab.c" /* yacc.c:1646  */
    break;

  case 20:
#line 240 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(GROUP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2386 "y.tab.c" /* yacc.c:1646  */
    break;

  case 21:
#line 241 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NODEGROUP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2392 "y.tab.c" /* yacc.c:1646  */
    break;

  case 22:
#line 242 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ARRAY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2398 "y.tab.c" /* yacc.c:1646  */
    break;

  case 23:
#line 246 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INCLUDE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2404 "y.tab.c" /* yacc.c:1646  */
    break;

  case 24:
#line 247 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(STACKSIZE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2410 "y.tab.c" /* yacc.c:1646  */
    break;

  case 25:
#line 248 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(THREADED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2416 "y.tab.c" /* yacc.c:1646  */
    break;

  case 26:
#line 249 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(TEMPLATE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2422 "y.tab.c" /* yacc.c:1646  */
    break;

  case 27:
#line 250 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(SYNC, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2428 "y.tab.c" /* yacc.c:1646  */
    break;

  case 28:
#line 251 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IGET, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2434 "y.tab.c" /* yacc.c:1646  */
    break;

  case 29:
#line 252 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(EXCLUSIVE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2440 "y.tab.c" /* yacc.c:1646  */
    break;

  case 30:
#line 253 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IMMEDIATE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2446 "y.tab.c" /* yacc.c:1646  */
    break;

  case 31:
#line 254 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(SKIPSCHED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2452 "y.tab.c" /* yacc.c:1646  */
    break;

  case 32:
#line 255 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOCOPY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2458 "y.tab.c" /* yacc.c:1646  */
    break;

  case 33:
#line 256 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOCOPYPOST, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2464 "y.tab.c" /* yacc.c:1646  */
    break;

  case 34:
#line 257 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INLINE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2470 "y.tab.c" /* yacc.c:1646  */
    break;

  case 35:
#line 258 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(VIRTUAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2476 "y.tab.c" /* yacc.c:1646  */
    break;

  case 36:
#line 259 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MIGRATABLE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2482 "y.tab.c" /* yacc.c:1646  */
    break;

  case 37:
#line 260 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CREATEHERE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2488 "y.tab.c" /* yacc.c:1646  */
    break;

  case 38:
#line 261 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CREATEHOME, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2494 "y.tab.c" /* yacc.c:1646  */
    break;

  case 39:
#line 262 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOKEEP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2500 "y.tab.c" /* yacc.c:1646  */
    break;

  case 40:
#line 263 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOTRACE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2506 "y.tab.c" /* yacc.c:1646  */
    break;

  case 41:
#line 264 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(APPWORK, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2512 "y.tab.c" /* yacc.c:1646  */
    break;

  case 42:
#line 267 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(PACKED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2518 "y.tab.c" /* yacc.c:1646  */
    break;

  case 43:
#line 268 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(VARSIZE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2524 "y.tab.c" /* yacc.c:1646  */
    break;

  case 44:
#line 269 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ENTRY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2530 "y.tab.c" /* yacc.c:1646  */
    break;

  case 45:
#line 270 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(FOR, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2536 "y.tab.c" /* yacc.c:1646  */
    break;

  case 46:
#line 271 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(FORALL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2542 "y.tab.c" /* yacc.c:1646  */
    break;

  case 47:
#line 272 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(WHILE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2548 "y.tab.c" /* yacc.c:1646  */
    break;

  case 48:
#line 273 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(WHEN, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2554 "y.tab.c" /* yacc.c:1646  */
    break;

  case 49:
#line 274 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(OVERLAP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2560 "y.tab.c" /* yacc.c:1646  */
    break;

  case 50:
#line 275 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(SERIAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2566 "y.tab.c" /* yacc.c:1646  */
    break;

  case 51:
#line 276 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IF, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2572 "y.tab.c" /* yacc.c:1646  */
    break;

  case 52:
#line 277 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ELSE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2578 "y.tab.c" /* yacc.c:1646  */
    break;

  case 53:
#line 279 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(LOCAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2584 "y.tab.c" /* yacc.c:1646  */
    break;

  case 54:
#line 281 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(USING, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2590 "y.tab.c" /* yacc.c:1646  */
    break;

  case 55:
#line 282 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ACCEL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2596 "y.tab.c" /* yacc.c:1646  */
    break;

  case 56:
#line 285 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ACCELBLOCK, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2602 "y.tab.c" /* yacc.c:1646  */
    break;

  case 57:
#line 286 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MEMCRITICAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2608 "y.tab.c" /* yacc.c:1646  */
    break;

  case 58:
#line 287 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(REDUCTIONTARGET, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2614 "y.tab.c" /* yacc.c:1646  */
    break;

  case 59:
#line 288 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CASE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2620 "y.tab.c" /* yacc.c:1646  */
    break;

  case 60:
#line 293 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 2626 "y.tab.c" /* yacc.c:1646  */
    break;

  case 61:
#line 295 "xi-grammar.y" /* yacc.c:1646  */
    {
		  char *tmp = new char[strlen((yyvsp[-3].strval))+strlen((yyvsp[0].strval))+3];
		  sprintf(tmp,"%s::%s", (yyvsp[-3].strval), (yyvsp[0].strval));
		  (yyval.strval) = tmp;
		}
#line 2636 "y.tab.c" /* yacc.c:1646  */
    break;

  case 62:
#line 301 "xi-grammar.y" /* yacc.c:1646  */
    {
		  char *tmp = new char[strlen((yyvsp[-3].strval))+5+3];
		  sprintf(tmp,"%s::array", (yyvsp[-3].strval));
		  (yyval.strval) = tmp;
		}
#line 2646 "y.tab.c" /* yacc.c:1646  */
    break;

  case 63:
#line 308 "xi-grammar.y" /* yacc.c:1646  */
    { 
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist)); 
		}
#line 2654 "y.tab.c" /* yacc.c:1646  */
    break;

  case 64:
#line 312 "xi-grammar.y" /* yacc.c:1646  */
    {  
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist)); 
		    (yyval.module)->setMain();
		}
#line 2663 "y.tab.c" /* yacc.c:1646  */
    break;

  case 65:
#line 319 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = 0; }
#line 2669 "y.tab.c" /* yacc.c:1646  */
    break;

  case 66:
#line 321 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = (yyvsp[-2].conslist); }
#line 2675 "y.tab.c" /* yacc.c:1646  */
    break;

  case 67:
#line 325 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = 0; }
#line 2681 "y.tab.c" /* yacc.c:1646  */
    break;

  case 68:
#line 327 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = new ConstructList(lineno, (yyvsp[-1].construct), (yyvsp[0].conslist)); }
#line 2687 "y.tab.c" /* yacc.c:1646  */
    break;

  case 69:
#line 331 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new UsingScope((yyvsp[0].strval), false); }
#line 2693 "y.tab.c" /* yacc.c:1646  */
    break;

  case 70:
#line 333 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new UsingScope((yyvsp[0].strval), true); }
#line 2699 "y.tab.c" /* yacc.c:1646  */
    break;

  case 71:
#line 335 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].member)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].member); }
#line 2705 "y.tab.c" /* yacc.c:1646  */
    break;

  case 72:
#line 337 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].message)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].message); }
#line 2711 "y.tab.c" /* yacc.c:1646  */
    break;

  case 73:
#line 339 "xi-grammar.y" /* yacc.c:1646  */
    {
                  Entry *e = new Entry(lineno, (yyvsp[-5].intval), (yyvsp[-4].type), (yyvsp[-2].strval), (yyvsp[0].plist), 0, 0, 0, (yylsp[-7]).first_line, (yyloc).last_line);
                  int isExtern = 1;
                  e->setExtern(isExtern);
                  e->targs = (yyvsp[-1].tparlist);
                  e->label = new XStr;
                  (yyvsp[-3].ntype)->print(*e->label);
                  (yyval.construct) = e;
                  firstRdma = true;
                }
#line 2726 "y.tab.c" /* yacc.c:1646  */
    break;

  case 74:
#line 352 "xi-grammar.y" /* yacc.c:1646  */
    { if((yyvsp[-2].conslist)) (yyvsp[-2].conslist)->recurse<int&>((yyvsp[-4].intval), &Construct::setExtern); (yyval.construct) = (yyvsp[-2].conslist); }
#line 2732 "y.tab.c" /* yacc.c:1646  */
    break;

  case 75:
#line 354 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new Scope((yyvsp[-3].strval), (yyvsp[-1].conslist)); }
#line 2738 "y.tab.c" /* yacc.c:1646  */
    break;

  case 76:
#line 356 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = (yyvsp[-1].construct); }
#line 2744 "y.tab.c" /* yacc.c:1646  */
    break;

  case 77:
#line 358 "xi-grammar.y" /* yacc.c:1646  */
    {
          ERROR("preceding construct must be semicolon terminated",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 2754 "y.tab.c" /* yacc.c:1646  */
    break;

  case 78:
#line 364 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].module)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].module); }
#line 2760 "y.tab.c" /* yacc.c:1646  */
    break;

  case 79:
#line 366 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2766 "y.tab.c" /* yacc.c:1646  */
    break;

  case 80:
#line 368 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2772 "y.tab.c" /* yacc.c:1646  */
    break;

  case 81:
#line 370 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2778 "y.tab.c" /* yacc.c:1646  */
    break;

  case 82:
#line 372 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2784 "y.tab.c" /* yacc.c:1646  */
    break;

  case 83:
#line 374 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].templat)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].templat); }
#line 2790 "y.tab.c" /* yacc.c:1646  */
    break;

  case 84:
#line 376 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = NULL; }
#line 2796 "y.tab.c" /* yacc.c:1646  */
    break;

  case 85:
#line 378 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = NULL; }
#line 2802 "y.tab.c" /* yacc.c:1646  */
    break;

  case 86:
#line 380 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = (yyvsp[0].accelBlock); }
#line 2808 "y.tab.c" /* yacc.c:1646  */
    break;

  case 87:
#line 382 "xi-grammar.y" /* yacc.c:1646  */
    {
          ERROR("invalid construct",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 2818 "y.tab.c" /* yacc.c:1646  */
    break;

  case 88:
#line 390 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamType((yyvsp[0].type)); }
#line 2824 "y.tab.c" /* yacc.c:1646  */
    break;

  case 89:
#line 392 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
#line 2830 "y.tab.c" /* yacc.c:1646  */
    break;

  case 90:
#line 394 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
#line 2836 "y.tab.c" /* yacc.c:1646  */
    break;

  case 91:
#line 398 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = new TParamList((yyvsp[0].tparam)); }
#line 2842 "y.tab.c" /* yacc.c:1646  */
    break;

  case 92:
#line 400 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = new TParamList((yyvsp[-2].tparam), (yyvsp[0].tparlist)); }
#line 2848 "y.tab.c" /* yacc.c:1646  */
    break;

  case 93:
#line 404 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = new TParamList(0); }
#line 2854 "y.tab.c" /* yacc.c:1646  */
    break;

  case 94:
#line 406 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = (yyvsp[0].tparlist); }
#line 2860 "y.tab.c" /* yacc.c:1646  */
    break;

  case 95:
#line 410 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = 0; }
#line 2866 "y.tab.c" /* yacc.c:1646  */
    break;

  case 96:
#line 412 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = (yyvsp[-1].tparlist); }
#line 2872 "y.tab.c" /* yacc.c:1646  */
    break;

  case 97:
#line 416 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("int"); }
#line 2878 "y.tab.c" /* yacc.c:1646  */
    break;

  case 98:
#line 418 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long"); }
#line 2884 "y.tab.c" /* yacc.c:1646  */
    break;

  case 99:
#line 420 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("short"); }
#line 2890 "y.tab.c" /* yacc.c:1646  */
    break;

  case 100:
#line 422 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("char"); }
#line 2896 "y.tab.c" /* yacc.c:1646  */
    break;

  case 101:
#line 424 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned int"); }
#line 2902 "y.tab.c" /* yacc.c:1646  */
    break;

  case 102:
#line 426 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long"); }
#line 2908 "y.tab.c" /* yacc.c:1646  */
    break;

  case 103:
#line 428 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long"); }
#line 2914 "y.tab.c" /* yacc.c:1646  */
    break;

  case 104:
#line 430 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long long"); }
#line 2920 "y.tab.c" /* yacc.c:1646  */
    break;

  case 105:
#line 432 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned short"); }
#line 2926 "y.tab.c" /* yacc.c:1646  */
    break;

  case 106:
#line 434 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned char"); }
#line 2932 "y.tab.c" /* yacc.c:1646  */
    break;

  case 107:
#line 436 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long long"); }
#line 2938 "y.tab.c" /* yacc.c:1646  */
    break;

  case 108:
#line 438 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("float"); }
#line 2944 "y.tab.c" /* yacc.c:1646  */
    break;

  case 109:
#line 440 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("double"); }
#line 2950 "y.tab.c" /* yacc.c:1646  */
    break;

  case 110:
#line 442 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long double"); }
#line 2956 "y.tab.c" /* yacc.c:1646  */
    break;

  case 111:
#line 444 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("void"); }
#line 2962 "y.tab.c" /* yacc.c:1646  */
    break;

  case 112:
#line 447 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ntype) = new NamedType((yyvsp[-1].strval),(yyvsp[0].tparlist)); }
#line 2968 "y.tab.c" /* yacc.c:1646  */
    break;

  case 113:
#line 448 "xi-grammar.y" /* yacc.c:1646  */
    { 
                    const char* basename, *scope;
                    splitScopedName((yyvsp[-1].strval), &scope, &basename);
                    (yyval.ntype) = new NamedType(basename, (yyvsp[0].tparlist), scope);
                }
#line 2978 "y.tab.c" /* yacc.c:1646  */
    break;

  case 114:
#line 454 "xi-grammar.y" /* yacc.c:1646  */
    {
			const char* basename, *scope;
			splitScopedName((yyvsp[-1].strval), &scope, &basename);
			(yyval.ntype) = new NamedType(basename, (yyvsp[0].tparlist), scope, true);
		}
#line 2988 "y.tab.c" /* yacc.c:1646  */
    break;

  case 115:
#line 462 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 2994 "y.tab.c" /* yacc.c:1646  */
    break;

  case 116:
#line 464 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ntype); }
#line 3000 "y.tab.c" /* yacc.c:1646  */
    break;

  case 117:
#line 468 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ptype) = new PtrType((yyvsp[-1].type)); }
#line 3006 "y.tab.c" /* yacc.c:1646  */
    break;

  case 118:
#line 472 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
#line 3012 "y.tab.c" /* yacc.c:1646  */
    break;

  case 119:
#line 474 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
#line 3018 "y.tab.c" /* yacc.c:1646  */
    break;

  case 120:
#line 478 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ftype) = new FuncType((yyvsp[-7].type), (yyvsp[-4].strval), (yyvsp[-1].plist)); }
#line 3024 "y.tab.c" /* yacc.c:1646  */
    break;

  case 121:
#line 482 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3030 "y.tab.c" /* yacc.c:1646  */
    break;

  case 122:
#line 484 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3036 "y.tab.c" /* yacc.c:1646  */
    break;

  case 123:
#line 486 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3042 "y.tab.c" /* yacc.c:1646  */
    break;

  case 124:
#line 488 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ftype); }
#line 3048 "y.tab.c" /* yacc.c:1646  */
    break;

  case 125:
#line 490 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[0].type)); }
#line 3054 "y.tab.c" /* yacc.c:1646  */
    break;

  case 126:
#line 492 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[-1].type)); }
#line 3060 "y.tab.c" /* yacc.c:1646  */
    break;

  case 127:
#line 496 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3066 "y.tab.c" /* yacc.c:1646  */
    break;

  case 128:
#line 498 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3072 "y.tab.c" /* yacc.c:1646  */
    break;

  case 129:
#line 500 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3078 "y.tab.c" /* yacc.c:1646  */
    break;

  case 130:
#line 502 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[0].type)); }
#line 3084 "y.tab.c" /* yacc.c:1646  */
    break;

  case 131:
#line 504 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[-1].type)); }
#line 3090 "y.tab.c" /* yacc.c:1646  */
    break;

  case 132:
#line 508 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType(new RValueReferenceType((yyvsp[-5].type))); }
#line 3096 "y.tab.c" /* yacc.c:1646  */
    break;

  case 133:
#line 510 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType(new ReferenceType((yyvsp[-4].type))); }
#line 3102 "y.tab.c" /* yacc.c:1646  */
    break;

  case 134:
#line 512 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType((yyvsp[-3].type)); }
#line 3108 "y.tab.c" /* yacc.c:1646  */
    break;

  case 135:
#line 514 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new RValueReferenceType((yyvsp[-2].type)); }
#line 3114 "y.tab.c" /* yacc.c:1646  */
    break;

  case 136:
#line 516 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ReferenceType((yyvsp[-1].type)); }
#line 3120 "y.tab.c" /* yacc.c:1646  */
    break;

  case 137:
#line 518 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3126 "y.tab.c" /* yacc.c:1646  */
    break;

  case 138:
#line 522 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType(new RValueReferenceType((yyvsp[-5].type))); }
#line 3132 "y.tab.c" /* yacc.c:1646  */
    break;

  case 139:
#line 524 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType(new ReferenceType((yyvsp[-4].type))); }
#line 3138 "y.tab.c" /* yacc.c:1646  */
    break;

  case 140:
#line 526 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType((yyvsp[-3].type)); }
#line 3144 "y.tab.c" /* yacc.c:1646  */
    break;

  case 141:
#line 528 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new RValueReferenceType((yyvsp[-2].type)); }
#line 3150 "y.tab.c" /* yacc.c:1646  */
    break;

  case 142:
#line 530 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ReferenceType((yyvsp[-1].type)); }
#line 3156 "y.tab.c" /* yacc.c:1646  */
    break;

  case 143:
#line 532 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3162 "y.tab.c" /* yacc.c:1646  */
    break;

  case 144:
#line 536 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 3168 "y.tab.c" /* yacc.c:1646  */
    break;

  case 145:
#line 540 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = (yyvsp[-1].val); }
#line 3174 "y.tab.c" /* yacc.c:1646  */
    break;

  case 146:
#line 544 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.vallist) = 0; }
#line 3180 "y.tab.c" /* yacc.c:1646  */
    break;

  case 147:
#line 546 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.vallist) = new ValueList((yyvsp[-1].val), (yyvsp[0].vallist)); }
#line 3186 "y.tab.c" /* yacc.c:1646  */
    break;

  case 148:
#line 550 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].vallist)); }
#line 3192 "y.tab.c" /* yacc.c:1646  */
    break;

  case 149:
#line 554 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[-3].type), (yyvsp[-1].strval), (yyvsp[0].vallist), 1); }
#line 3198 "y.tab.c" /* yacc.c:1646  */
    break;

  case 150:
#line 558 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0;}
#line 3204 "y.tab.c" /* yacc.c:1646  */
    break;

  case 151:
#line 560 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0;}
#line 3210 "y.tab.c" /* yacc.c:1646  */
    break;

  case 152:
#line 564 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3216 "y.tab.c" /* yacc.c:1646  */
    break;

  case 153:
#line 566 "xi-grammar.y" /* yacc.c:1646  */
    { 
		  /*
		  printf("Warning: Message attributes are being phased out.\n");
		  printf("Warning: Please remove them from interface files.\n");
		  */
		  (yyval.intval) = (yyvsp[-1].intval); 
		}
#line 3228 "y.tab.c" /* yacc.c:1646  */
    break;

  case 154:
#line 576 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[0].intval); }
#line 3234 "y.tab.c" /* yacc.c:1646  */
    break;

  case 155:
#line 578 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[-2].intval) | (yyvsp[0].intval); }
#line 3240 "y.tab.c" /* yacc.c:1646  */
    break;

  case 156:
#line 582 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3246 "y.tab.c" /* yacc.c:1646  */
    break;

  case 157:
#line 584 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3252 "y.tab.c" /* yacc.c:1646  */
    break;

  case 158:
#line 588 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = 0; }
#line 3258 "y.tab.c" /* yacc.c:1646  */
    break;

  case 159:
#line 590 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-1].cattr); }
#line 3264 "y.tab.c" /* yacc.c:1646  */
    break;

  case 160:
#line 594 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[0].cattr); }
#line 3270 "y.tab.c" /* yacc.c:1646  */
    break;

  case 161:
#line 596 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
#line 3276 "y.tab.c" /* yacc.c:1646  */
    break;

  case 162:
#line 600 "xi-grammar.y" /* yacc.c:1646  */
    { python_doc = NULL; (yyval.intval) = 0; }
#line 3282 "y.tab.c" /* yacc.c:1646  */
    break;

  case 163:
#line 602 "xi-grammar.y" /* yacc.c:1646  */
    { python_doc = (yyvsp[0].strval); (yyval.intval) = 0; }
#line 3288 "y.tab.c" /* yacc.c:1646  */
    break;

  case 164:
#line 606 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CPYTHON; }
#line 3294 "y.tab.c" /* yacc.c:1646  */
    break;

  case 165:
#line 610 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = 0; }
#line 3300 "y.tab.c" /* yacc.c:1646  */
    break;

  case 166:
#line 612 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-1].cattr); }
#line 3306 "y.tab.c" /* yacc.c:1646  */
    break;

  case 167:
#line 616 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[0].cattr); }
#line 3312 "y.tab.c" /* yacc.c:1646  */
    break;

  case 168:
#line 618 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
#line 3318 "y.tab.c" /* yacc.c:1646  */
    break;

  case 169:
#line 622 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CMIGRATABLE; }
#line 3324 "y.tab.c" /* yacc.c:1646  */
    break;

  case 170:
#line 624 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CPYTHON; }
#line 3330 "y.tab.c" /* yacc.c:1646  */
    break;

  case 171:
#line 628 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3336 "y.tab.c" /* yacc.c:1646  */
    break;

  case 172:
#line 630 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 3342 "y.tab.c" /* yacc.c:1646  */
    break;

  case 173:
#line 633 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3348 "y.tab.c" /* yacc.c:1646  */
    break;

  case 174:
#line 635 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 3354 "y.tab.c" /* yacc.c:1646  */
    break;

  case 175:
#line 638 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mv) = new MsgVar((yyvsp[-3].type), (yyvsp[-2].strval), (yyvsp[-4].intval), (yyvsp[-1].intval)); }
#line 3360 "y.tab.c" /* yacc.c:1646  */
    break;

  case 176:
#line 642 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mvlist) = new MsgVarList((yyvsp[0].mv)); }
#line 3366 "y.tab.c" /* yacc.c:1646  */
    break;

  case 177:
#line 644 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mvlist) = new MsgVarList((yyvsp[-1].mv), (yyvsp[0].mvlist)); }
#line 3372 "y.tab.c" /* yacc.c:1646  */
    break;

  case 178:
#line 648 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, (yyvsp[0].ntype)); }
#line 3378 "y.tab.c" /* yacc.c:1646  */
    break;

  case 179:
#line 650 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, (yyvsp[-2].ntype)); }
#line 3384 "y.tab.c" /* yacc.c:1646  */
    break;

  case 180:
#line 652 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, (yyvsp[-3].ntype), (yyvsp[-1].mvlist)); }
#line 3390 "y.tab.c" /* yacc.c:1646  */
    break;

  case 181:
#line 656 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = 0; }
#line 3396 "y.tab.c" /* yacc.c:1646  */
    break;

  case 182:
#line 658 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = (yyvsp[0].typelist); }
#line 3402 "y.tab.c" /* yacc.c:1646  */
    break;

  case 183:
#line 662 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = new TypeList((yyvsp[0].ntype)); }
#line 3408 "y.tab.c" /* yacc.c:1646  */
    break;

  case 184:
#line 664 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = new TypeList((yyvsp[-2].ntype), (yyvsp[0].typelist)); }
#line 3414 "y.tab.c" /* yacc.c:1646  */
    break;

  case 185:
#line 668 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr)|Chare::CCHARE, (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3420 "y.tab.c" /* yacc.c:1646  */
    break;

  case 186:
#line 670 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3426 "y.tab.c" /* yacc.c:1646  */
    break;

  case 187:
#line 674 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3432 "y.tab.c" /* yacc.c:1646  */
    break;

  case 188:
#line 678 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new NodeGroup(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3438 "y.tab.c" /* yacc.c:1646  */
    break;

  case 189:
#line 682 "xi-grammar.y" /* yacc.c:1646  */
    {/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",(yyvsp[-2].strval));
			(yyval.ntype) = new NamedType(buf); 
		}
#line 3448 "y.tab.c" /* yacc.c:1646  */
    break;

  case 190:
#line 688 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ntype) = (yyvsp[-1].ntype); }
#line 3454 "y.tab.c" /* yacc.c:1646  */
    break;

  case 191:
#line 692 "xi-grammar.y" /* yacc.c:1646  */
    {  (yyval.chare) = new Array(lineno, (yyvsp[-4].cattr), (yyvsp[-3].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3460 "y.tab.c" /* yacc.c:1646  */
    break;

  case 192:
#line 694 "xi-grammar.y" /* yacc.c:1646  */
    {  (yyval.chare) = new Array(lineno, (yyvsp[-3].cattr), (yyvsp[-4].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3466 "y.tab.c" /* yacc.c:1646  */
    break;

  case 193:
#line 698 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr)|Chare::CCHARE, new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist));}
#line 3472 "y.tab.c" /* yacc.c:1646  */
    break;

  case 194:
#line 700 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3478 "y.tab.c" /* yacc.c:1646  */
    break;

  case 195:
#line 704 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3484 "y.tab.c" /* yacc.c:1646  */
    break;

  case 196:
#line 708 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new NodeGroup( lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3490 "y.tab.c" /* yacc.c:1646  */
    break;

  case 197:
#line 712 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Array( lineno, 0, (yyvsp[-3].ntype), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3496 "y.tab.c" /* yacc.c:1646  */
    break;

  case 198:
#line 716 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-1].strval))); }
#line 3502 "y.tab.c" /* yacc.c:1646  */
    break;

  case 199:
#line 718 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-4].strval)), (yyvsp[-2].mvlist)); }
#line 3508 "y.tab.c" /* yacc.c:1646  */
    break;

  case 200:
#line 722 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = 0; }
#line 3514 "y.tab.c" /* yacc.c:1646  */
    break;

  case 201:
#line 724 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3520 "y.tab.c" /* yacc.c:1646  */
    break;

  case 202:
#line 728 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = 0; }
#line 3526 "y.tab.c" /* yacc.c:1646  */
    break;

  case 203:
#line 730 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3532 "y.tab.c" /* yacc.c:1646  */
    break;

  case 204:
#line 732 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3538 "y.tab.c" /* yacc.c:1646  */
    break;

  case 205:
#line 734 "xi-grammar.y" /* yacc.c:1646  */
    {
		  XStr typeStr;
		  (yyvsp[0].ntype)->print(typeStr);
		  char *tmp = strdup(typeStr.get_string());
		  (yyval.strval) = tmp;
		}
#line 3549 "y.tab.c" /* yacc.c:1646  */
    break;

  case 206:
#line 743 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TTypeEllipsis(new NamedEllipsisType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3555 "y.tab.c" /* yacc.c:1646  */
    break;

  case 207:
#line 745 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TTypeEllipsis(new NamedEllipsisType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3561 "y.tab.c" /* yacc.c:1646  */
    break;

  case 208:
#line 747 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TType(new NamedType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3567 "y.tab.c" /* yacc.c:1646  */
    break;

  case 209:
#line 749 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TType(new NamedType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3573 "y.tab.c" /* yacc.c:1646  */
    break;

  case 210:
#line 751 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TFunc((yyvsp[-1].ftype), (yyvsp[0].strval)); }
#line 3579 "y.tab.c" /* yacc.c:1646  */
    break;

  case 211:
#line 753 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TName((yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].strval)); }
#line 3585 "y.tab.c" /* yacc.c:1646  */
    break;

  case 212:
#line 757 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = new TVarList((yyvsp[0].tvar)); }
#line 3591 "y.tab.c" /* yacc.c:1646  */
    break;

  case 213:
#line 759 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = new TVarList((yyvsp[-2].tvar), (yyvsp[0].tvarlist)); }
#line 3597 "y.tab.c" /* yacc.c:1646  */
    break;

  case 214:
#line 763 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = (yyvsp[-1].tvarlist); }
#line 3603 "y.tab.c" /* yacc.c:1646  */
    break;

  case 215:
#line 767 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3609 "y.tab.c" /* yacc.c:1646  */
    break;

  case 216:
#line 769 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3615 "y.tab.c" /* yacc.c:1646  */
    break;

  case 217:
#line 771 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3621 "y.tab.c" /* yacc.c:1646  */
    break;

  case 218:
#line 773 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3627 "y.tab.c" /* yacc.c:1646  */
    break;

  case 219:
#line 775 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].message)); (yyvsp[0].message)->setTemplate((yyval.templat)); }
#line 3633 "y.tab.c" /* yacc.c:1646  */
    break;

  case 220:
#line 779 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = 0; }
#line 3639 "y.tab.c" /* yacc.c:1646  */
    break;

  case 221:
#line 781 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = (yyvsp[-2].mbrlist); }
#line 3645 "y.tab.c" /* yacc.c:1646  */
    break;

  case 222:
#line 785 "xi-grammar.y" /* yacc.c:1646  */
    { 
                  if (!connectEntries.empty()) {
                    (yyval.mbrlist) = new AstChildren<Member>(connectEntries);
		  } else {
		    (yyval.mbrlist) = 0; 
                  }
		}
#line 3657 "y.tab.c" /* yacc.c:1646  */
    break;

  case 223:
#line 793 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = new AstChildren<Member>(-1, (yyvsp[-1].member), (yyvsp[0].mbrlist)); }
#line 3663 "y.tab.c" /* yacc.c:1646  */
    break;

  case 224:
#line 797 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].readonly); }
#line 3669 "y.tab.c" /* yacc.c:1646  */
    break;

  case 225:
#line 799 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].readonly); }
#line 3675 "y.tab.c" /* yacc.c:1646  */
    break;

  case 227:
#line 802 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].member); }
#line 3681 "y.tab.c" /* yacc.c:1646  */
    break;

  case 228:
#line 804 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].pupable); }
#line 3687 "y.tab.c" /* yacc.c:1646  */
    break;

  case 229:
#line 806 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].includeFile); }
#line 3693 "y.tab.c" /* yacc.c:1646  */
    break;

  case 230:
#line 808 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new ClassDeclaration(lineno,(yyvsp[0].strval)); }
#line 3699 "y.tab.c" /* yacc.c:1646  */
    break;

  case 231:
#line 812 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1); }
#line 3705 "y.tab.c" /* yacc.c:1646  */
    break;

  case 232:
#line 814 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1); }
#line 3711 "y.tab.c" /* yacc.c:1646  */
    break;

  case 233:
#line 816 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[-6].strval)) + '<' +
					    ((yyvsp[-4].tparlist))->to_string() + '>').c_str()),
				    1);
		}
#line 3721 "y.tab.c" /* yacc.c:1646  */
    break;

  case 234:
#line 822 "xi-grammar.y" /* yacc.c:1646  */
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[-2]).first_column, (yylsp[-2]).last_column, (yylsp[-2]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1);
		}
#line 3731 "y.tab.c" /* yacc.c:1646  */
    break;

  case 235:
#line 828 "xi-grammar.y" /* yacc.c:1646  */
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[-5]).first_column, (yylsp[-5]).last_column, (yylsp[-5]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1);
		}
#line 3741 "y.tab.c" /* yacc.c:1646  */
    break;

  case 236:
#line 837 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 0); }
#line 3747 "y.tab.c" /* yacc.c:1646  */
    break;

  case 237:
#line 839 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 0); }
#line 3753 "y.tab.c" /* yacc.c:1646  */
    break;

  case 238:
#line 841 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[-6].strval)) + '<' +
					    ((yyvsp[-4].tparlist))->to_string() + '>').c_str()),
				    0);
		}
#line 3763 "y.tab.c" /* yacc.c:1646  */
    break;

  case 239:
#line 847 "xi-grammar.y" /* yacc.c:1646  */
    {
                  InitCall* rtn = new InitCall(lineno, (yyvsp[-3].strval), 0);
                  rtn->setAccel();
                  (yyval.member) = rtn;
		}
#line 3773 "y.tab.c" /* yacc.c:1646  */
    break;

  case 240:
#line 855 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[0].ntype),0); }
#line 3779 "y.tab.c" /* yacc.c:1646  */
    break;

  case 241:
#line 857 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[-2].ntype),(yyvsp[0].pupable)); }
#line 3785 "y.tab.c" /* yacc.c:1646  */
    break;

  case 242:
#line 860 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.includeFile) = new IncludeFile(lineno,(yyvsp[0].strval)); }
#line 3791 "y.tab.c" /* yacc.c:1646  */
    break;

  case 243:
#line 864 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].member); }
#line 3797 "y.tab.c" /* yacc.c:1646  */
    break;

  case 244:
#line 868 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].entry); }
#line 3803 "y.tab.c" /* yacc.c:1646  */
    break;

  case 245:
#line 870 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyvsp[0].entry)->tspec = (yyvsp[-1].tvarlist);
                  (yyval.member) = (yyvsp[0].entry);
                }
#line 3812 "y.tab.c" /* yacc.c:1646  */
    break;

  case 246:
#line 875 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[-1].member); }
#line 3818 "y.tab.c" /* yacc.c:1646  */
    break;

  case 247:
#line 877 "xi-grammar.y" /* yacc.c:1646  */
    {
          ERROR("invalid SDAG member",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 3828 "y.tab.c" /* yacc.c:1646  */
    break;

  case 248:
#line 885 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3834 "y.tab.c" /* yacc.c:1646  */
    break;

  case 249:
#line 887 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3840 "y.tab.c" /* yacc.c:1646  */
    break;

  case 250:
#line 889 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3846 "y.tab.c" /* yacc.c:1646  */
    break;

  case 251:
#line 891 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3852 "y.tab.c" /* yacc.c:1646  */
    break;

  case 252:
#line 893 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3858 "y.tab.c" /* yacc.c:1646  */
    break;

  case 253:
#line 895 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3864 "y.tab.c" /* yacc.c:1646  */
    break;

  case 254:
#line 897 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3870 "y.tab.c" /* yacc.c:1646  */
    break;

  case 255:
#line 899 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3876 "y.tab.c" /* yacc.c:1646  */
    break;

  case 256:
#line 901 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3882 "y.tab.c" /* yacc.c:1646  */
    break;

  case 257:
#line 903 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3888 "y.tab.c" /* yacc.c:1646  */
    break;

  case 258:
#line 905 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3894 "y.tab.c" /* yacc.c:1646  */
    break;

  case 259:
#line 908 "xi-grammar.y" /* yacc.c:1646  */
    { 
                  (yyval.entry) = new Entry(lineno, (yyvsp[-5].intval), (yyvsp[-4].type), (yyvsp[-3].strval), (yyvsp[-2].plist), (yyvsp[-1].val), (yyvsp[0].sentry), (const char *) NULL, (yylsp[-6]).first_line, (yyloc).last_line);
		  if ((yyvsp[0].sentry) != 0) { 
		    (yyvsp[0].sentry)->con1 = new SdagConstruct(SIDENT, (yyvsp[-3].strval));
                    (yyvsp[0].sentry)->setEntry((yyval.entry));
                    (yyvsp[0].sentry)->param = new ParamList((yyvsp[-2].plist));
                  }
                  firstRdma = true;
		}
#line 3908 "y.tab.c" /* yacc.c:1646  */
    break;

  case 260:
#line 918 "xi-grammar.y" /* yacc.c:1646  */
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
#line 3929 "y.tab.c" /* yacc.c:1646  */
    break;

  case 261:
#line 935 "xi-grammar.y" /* yacc.c:1646  */
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
#line 3948 "y.tab.c" /* yacc.c:1646  */
    break;

  case 262:
#line 952 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.accelBlock) = new AccelBlock(lineno, new XStr((yyvsp[-2].strval))); }
#line 3954 "y.tab.c" /* yacc.c:1646  */
    break;

  case 263:
#line 954 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.accelBlock) = new AccelBlock(lineno, NULL); }
#line 3960 "y.tab.c" /* yacc.c:1646  */
    break;

  case 264:
#line 958 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3966 "y.tab.c" /* yacc.c:1646  */
    break;

  case 265:
#line 962 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3972 "y.tab.c" /* yacc.c:1646  */
    break;

  case 266:
#line 964 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[-1].intval); }
#line 3978 "y.tab.c" /* yacc.c:1646  */
    break;

  case 267:
#line 966 "xi-grammar.y" /* yacc.c:1646  */
    { ERROR("invalid entry method attribute list",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
#line 3987 "y.tab.c" /* yacc.c:1646  */
    break;

  case 268:
#line 973 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[0].intval); }
#line 3993 "y.tab.c" /* yacc.c:1646  */
    break;

  case 269:
#line 975 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[-2].intval) | (yyvsp[0].intval); }
#line 3999 "y.tab.c" /* yacc.c:1646  */
    break;

  case 270:
#line 979 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = STHREADED; }
#line 4005 "y.tab.c" /* yacc.c:1646  */
    break;

  case 271:
#line 981 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SSYNC; }
#line 4011 "y.tab.c" /* yacc.c:1646  */
    break;

  case 272:
#line 983 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SIGET; }
#line 4017 "y.tab.c" /* yacc.c:1646  */
    break;

  case 273:
#line 985 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SLOCKED; }
#line 4023 "y.tab.c" /* yacc.c:1646  */
    break;

  case 274:
#line 987 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SCREATEHERE; }
#line 4029 "y.tab.c" /* yacc.c:1646  */
    break;

  case 275:
#line 989 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SCREATEHOME; }
#line 4035 "y.tab.c" /* yacc.c:1646  */
    break;

  case 276:
#line 991 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SNOKEEP; }
#line 4041 "y.tab.c" /* yacc.c:1646  */
    break;

  case 277:
#line 993 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SNOTRACE; }
#line 4047 "y.tab.c" /* yacc.c:1646  */
    break;

  case 278:
#line 995 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SAPPWORK; }
#line 4053 "y.tab.c" /* yacc.c:1646  */
    break;

  case 279:
#line 997 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SIMMEDIATE; }
#line 4059 "y.tab.c" /* yacc.c:1646  */
    break;

  case 280:
#line 999 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SSKIPSCHED; }
#line 4065 "y.tab.c" /* yacc.c:1646  */
    break;

  case 281:
#line 1001 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SINLINE; }
#line 4071 "y.tab.c" /* yacc.c:1646  */
    break;

  case 282:
#line 1003 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SLOCAL; }
#line 4077 "y.tab.c" /* yacc.c:1646  */
    break;

  case 283:
#line 1005 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SPYTHON; }
#line 4083 "y.tab.c" /* yacc.c:1646  */
    break;

  case 284:
#line 1007 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SMEM; }
#line 4089 "y.tab.c" /* yacc.c:1646  */
    break;

  case 285:
#line 1009 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SREDUCE; }
#line 4095 "y.tab.c" /* yacc.c:1646  */
    break;

  case 286:
#line 1011 "xi-grammar.y" /* yacc.c:1646  */
    {
        (yyval.intval) = SAGGREGATE;
    }
#line 4103 "y.tab.c" /* yacc.c:1646  */
    break;

  case 287:
#line 1015 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("invalid entry method attribute",
		        (yylsp[0]).first_column, (yylsp[0]).last_column);
		  yyclearin;
		  yyerrok;
		}
#line 4114 "y.tab.c" /* yacc.c:1646  */
    break;

  case 288:
#line 1024 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4120 "y.tab.c" /* yacc.c:1646  */
    break;

  case 289:
#line 1026 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4126 "y.tab.c" /* yacc.c:1646  */
    break;

  case 290:
#line 1028 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4132 "y.tab.c" /* yacc.c:1646  */
    break;

  case 291:
#line 1032 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = ""; }
#line 4138 "y.tab.c" /* yacc.c:1646  */
    break;

  case 292:
#line 1034 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4144 "y.tab.c" /* yacc.c:1646  */
    break;

  case 293:
#line 1036 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s, %s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4154 "y.tab.c" /* yacc.c:1646  */
    break;

  case 294:
#line 1044 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = ""; }
#line 4160 "y.tab.c" /* yacc.c:1646  */
    break;

  case 295:
#line 1046 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4166 "y.tab.c" /* yacc.c:1646  */
    break;

  case 296:
#line 1048 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s[%s]%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4176 "y.tab.c" /* yacc.c:1646  */
    break;

  case 297:
#line 1054 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s{%s}%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4186 "y.tab.c" /* yacc.c:1646  */
    break;

  case 298:
#line 1060 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s(%s)%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4196 "y.tab.c" /* yacc.c:1646  */
    break;

  case 299:
#line 1066 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"(%s)%s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4206 "y.tab.c" /* yacc.c:1646  */
    break;

  case 300:
#line 1074 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval));
		}
#line 4215 "y.tab.c" /* yacc.c:1646  */
    break;

  case 301:
#line 1081 "xi-grammar.y" /* yacc.c:1646  */
    { 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			(yyval.intval) = 0;
		}
#line 4225 "y.tab.c" /* yacc.c:1646  */
    break;

  case 302:
#line 1089 "xi-grammar.y" /* yacc.c:1646  */
    { 
			in_braces=0;
			(yyval.intval) = 0;
		}
#line 4234 "y.tab.c" /* yacc.c:1646  */
    break;

  case 303:
#line 1096 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[0].type));}
#line 4240 "y.tab.c" /* yacc.c:1646  */
    break;

  case 304:
#line 1098 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval)); (yyval.pname)->setConditional((yyvsp[0].intval)); }
#line 4246 "y.tab.c" /* yacc.c:1646  */
    break;

  case 305:
#line 1100 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[-3].type),(yyvsp[-2].strval),0,(yyvsp[0].val));}
#line 4252 "y.tab.c" /* yacc.c:1646  */
    break;

  case 306:
#line 1102 "xi-grammar.y" /* yacc.c:1646  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
		}
#line 4261 "y.tab.c" /* yacc.c:1646  */
    break;

  case 307:
#line 1107 "xi-grammar.y" /* yacc.c:1646  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
			(yyval.pname)->setRdma(CMK_ZC_P2P_SEND_MSG);
			if(firstRdma) {
				(yyval.pname)->setFirstRdma(true);
				firstRdma = false;
			}
		}
#line 4275 "y.tab.c" /* yacc.c:1646  */
    break;

  case 308:
#line 1117 "xi-grammar.y" /* yacc.c:1646  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
			(yyval.pname)->setRdma(CMK_ZC_P2P_RECV_MSG);
			if(firstRdma) {
				(yyval.pname)->setFirstRdma(true);
				firstRdma = false;
			}
		}
#line 4289 "y.tab.c" /* yacc.c:1646  */
    break;

  case 309:
#line 1128 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READONLY; }
#line 4295 "y.tab.c" /* yacc.c:1646  */
    break;

  case 310:
#line 1129 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READWRITE; }
#line 4301 "y.tab.c" /* yacc.c:1646  */
    break;

  case 311:
#line 1130 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_WRITEONLY; }
#line 4307 "y.tab.c" /* yacc.c:1646  */
    break;

  case 312:
#line 1133 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr((yyvsp[0].strval)); }
#line 4313 "y.tab.c" /* yacc.c:1646  */
    break;

  case 313:
#line 1134 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "->" << (yyvsp[0].strval); }
#line 4319 "y.tab.c" /* yacc.c:1646  */
    break;

  case 314:
#line 1135 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[-2].xstrptr)) << "." << (yyvsp[0].strval); }
#line 4325 "y.tab.c" /* yacc.c:1646  */
    break;

  case 315:
#line 1137 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << *((yyvsp[-1].xstrptr)) << "]";
                  delete (yyvsp[-3].xstrptr);
                  delete (yyvsp[-1].xstrptr);
                }
#line 4336 "y.tab.c" /* yacc.c:1646  */
    break;

  case 316:
#line 1144 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << (yyvsp[-1].strval) << "]";
                  delete (yyvsp[-3].xstrptr);
                }
#line 4346 "y.tab.c" /* yacc.c:1646  */
    break;

  case 317:
#line 1150 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "(" << *((yyvsp[-1].xstrptr)) << ")";
                  delete (yyvsp[-3].xstrptr);
                  delete (yyvsp[-1].xstrptr);
                }
#line 4357 "y.tab.c" /* yacc.c:1646  */
    break;

  case 318:
#line 1159 "xi-grammar.y" /* yacc.c:1646  */
    {
                  in_bracket = 0;
                  (yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName(), (yyvsp[-1].strval));
                }
#line 4366 "y.tab.c" /* yacc.c:1646  */
    break;

  case 319:
#line 1166 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[-6].intval));
                }
#line 4376 "y.tab.c" /* yacc.c:1646  */
    break;

  case 320:
#line 1172 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType(Parameter::ACCEL_BUFFER_TYPE_READWRITE);
		}
#line 4386 "y.tab.c" /* yacc.c:1646  */
    break;

  case 321:
#line 1178 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.pname) = (yyvsp[-3].pname);
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[-5].intval));
		}
#line 4396 "y.tab.c" /* yacc.c:1646  */
    break;

  case 322:
#line 1186 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 4402 "y.tab.c" /* yacc.c:1646  */
    break;

  case 323:
#line 1188 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 4408 "y.tab.c" /* yacc.c:1646  */
    break;

  case 324:
#line 1192 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 4414 "y.tab.c" /* yacc.c:1646  */
    break;

  case 325:
#line 1194 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 4420 "y.tab.c" /* yacc.c:1646  */
    break;

  case 326:
#line 1198 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = (yyvsp[-1].plist); }
#line 4426 "y.tab.c" /* yacc.c:1646  */
    break;

  case 327:
#line 1200 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList(new Parameter(0, new BuiltinType("void"))); }
#line 4432 "y.tab.c" /* yacc.c:1646  */
    break;

  case 328:
#line 1204 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = (yyvsp[-1].plist); }
#line 4438 "y.tab.c" /* yacc.c:1646  */
    break;

  case 329:
#line 1206 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = 0; }
#line 4444 "y.tab.c" /* yacc.c:1646  */
    break;

  case 330:
#line 1210 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = 0; }
#line 4450 "y.tab.c" /* yacc.c:1646  */
    break;

  case 331:
#line 1212 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4456 "y.tab.c" /* yacc.c:1646  */
    break;

  case 332:
#line 1216 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = 0; }
#line 4462 "y.tab.c" /* yacc.c:1646  */
    break;

  case 333:
#line 1218 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[0].sc)); }
#line 4468 "y.tab.c" /* yacc.c:1646  */
    break;

  case 334:
#line 1220 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[-2].slist)); }
#line 4474 "y.tab.c" /* yacc.c:1646  */
    break;

  case 335:
#line 1224 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.slist) = new SListConstruct((yyvsp[0].sc)); }
#line 4480 "y.tab.c" /* yacc.c:1646  */
    break;

  case 336:
#line 1226 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.slist) = new SListConstruct((yyvsp[-1].sc), (yyvsp[0].slist));  }
#line 4486 "y.tab.c" /* yacc.c:1646  */
    break;

  case 337:
#line 1230 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.olist) = new OListConstruct((yyvsp[0].sc)); }
#line 4492 "y.tab.c" /* yacc.c:1646  */
    break;

  case 338:
#line 1232 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.olist) = new OListConstruct((yyvsp[-1].sc), (yyvsp[0].slist)); }
#line 4498 "y.tab.c" /* yacc.c:1646  */
    break;

  case 339:
#line 1236 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.clist) = new CaseListConstruct((yyvsp[0].when)); }
#line 4504 "y.tab.c" /* yacc.c:1646  */
    break;

  case 340:
#line 1238 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.clist) = new CaseListConstruct((yyvsp[-1].when), (yyvsp[0].clist)); }
#line 4510 "y.tab.c" /* yacc.c:1646  */
    break;

  case 341:
#line 1240 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("case blocks can only contain when clauses",
		        (yylsp[0]).first_column, (yylsp[0]).last_column);
		  (yyval.clist) = 0;
		}
#line 4520 "y.tab.c" /* yacc.c:1646  */
    break;

  case 342:
#line 1248 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4526 "y.tab.c" /* yacc.c:1646  */
    break;

  case 343:
#line 1250 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = 0; }
#line 4532 "y.tab.c" /* yacc.c:1646  */
    break;

  case 344:
#line 1254 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-2].entrylist), 0); }
#line 4538 "y.tab.c" /* yacc.c:1646  */
    break;

  case 345:
#line 1256 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-1].entrylist), (yyvsp[0].sc)); }
#line 4544 "y.tab.c" /* yacc.c:1646  */
    break;

  case 346:
#line 1258 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-3].entrylist), (yyvsp[-1].slist)); }
#line 4550 "y.tab.c" /* yacc.c:1646  */
    break;

  case 347:
#line 1262 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4556 "y.tab.c" /* yacc.c:1646  */
    break;

  case 348:
#line 1264 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4562 "y.tab.c" /* yacc.c:1646  */
    break;

  case 349:
#line 1266 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4568 "y.tab.c" /* yacc.c:1646  */
    break;

  case 350:
#line 1268 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4574 "y.tab.c" /* yacc.c:1646  */
    break;

  case 351:
#line 1270 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4580 "y.tab.c" /* yacc.c:1646  */
    break;

  case 352:
#line 1272 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4586 "y.tab.c" /* yacc.c:1646  */
    break;

  case 353:
#line 1274 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4592 "y.tab.c" /* yacc.c:1646  */
    break;

  case 354:
#line 1276 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4598 "y.tab.c" /* yacc.c:1646  */
    break;

  case 355:
#line 1278 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4604 "y.tab.c" /* yacc.c:1646  */
    break;

  case 356:
#line 1280 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4610 "y.tab.c" /* yacc.c:1646  */
    break;

  case 357:
#line 1282 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4616 "y.tab.c" /* yacc.c:1646  */
    break;

  case 358:
#line 1284 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4622 "y.tab.c" /* yacc.c:1646  */
    break;

  case 359:
#line 1288 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new SerialConstruct((yyvsp[-2].strval), (yyvsp[-4].strval), (yylsp[-3]).first_line); }
#line 4628 "y.tab.c" /* yacc.c:1646  */
    break;

  case 360:
#line 1290 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new OverlapConstruct((yyvsp[-1].olist)); }
#line 4634 "y.tab.c" /* yacc.c:1646  */
    break;

  case 361:
#line 1292 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = (yyvsp[0].when); }
#line 4640 "y.tab.c" /* yacc.c:1646  */
    break;

  case 362:
#line 1294 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new CaseConstruct((yyvsp[-1].clist)); }
#line 4646 "y.tab.c" /* yacc.c:1646  */
    break;

  case 363:
#line 1296 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForConstruct((yyvsp[-8].intexpr), (yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4652 "y.tab.c" /* yacc.c:1646  */
    break;

  case 364:
#line 1298 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForConstruct((yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4658 "y.tab.c" /* yacc.c:1646  */
    break;

  case 365:
#line 1300 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-9].strval)), (yyvsp[-6].intexpr),
		             (yyvsp[-4].intexpr), (yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4665 "y.tab.c" /* yacc.c:1646  */
    break;

  case 366:
#line 1303 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-11].strval)), (yyvsp[-8].intexpr),
		             (yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4672 "y.tab.c" /* yacc.c:1646  */
    break;

  case 367:
#line 1306 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new IfConstruct((yyvsp[-3].intexpr), (yyvsp[-1].sc), (yyvsp[0].sc)); }
#line 4678 "y.tab.c" /* yacc.c:1646  */
    break;

  case 368:
#line 1308 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new IfConstruct((yyvsp[-5].intexpr), (yyvsp[-2].slist), (yyvsp[0].sc)); }
#line 4684 "y.tab.c" /* yacc.c:1646  */
    break;

  case 369:
#line 1310 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new WhileConstruct((yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4690 "y.tab.c" /* yacc.c:1646  */
    break;

  case 370:
#line 1312 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new WhileConstruct((yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4696 "y.tab.c" /* yacc.c:1646  */
    break;

  case 371:
#line 1314 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new SerialConstruct((yyvsp[-2].strval), NULL, (yyloc).first_line); }
#line 4702 "y.tab.c" /* yacc.c:1646  */
    break;

  case 372:
#line 1316 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("unknown SDAG construct or malformed entry method declaration.\n"
		        "You may have forgotten to terminate a previous entry method declaration with a"
		        " semicolon or forgotten to mark a block of sequential SDAG code as 'serial'",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
#line 4714 "y.tab.c" /* yacc.c:1646  */
    break;

  case 373:
#line 1326 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = 0; }
#line 4720 "y.tab.c" /* yacc.c:1646  */
    break;

  case 374:
#line 1328 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ElseConstruct((yyvsp[0].sc)); }
#line 4726 "y.tab.c" /* yacc.c:1646  */
    break;

  case 375:
#line 1330 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ElseConstruct((yyvsp[-1].slist)); }
#line 4732 "y.tab.c" /* yacc.c:1646  */
    break;

  case 376:
#line 1334 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intexpr) = new IntExprConstruct((yyvsp[0].strval)); }
#line 4738 "y.tab.c" /* yacc.c:1646  */
    break;

  case 377:
#line 1338 "xi-grammar.y" /* yacc.c:1646  */
    { in_int_expr = 0; (yyval.intval) = 0; }
#line 4744 "y.tab.c" /* yacc.c:1646  */
    break;

  case 378:
#line 1342 "xi-grammar.y" /* yacc.c:1646  */
    { in_int_expr = 1; (yyval.intval) = 0; }
#line 4750 "y.tab.c" /* yacc.c:1646  */
    break;

  case 379:
#line 1346 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[-1].strval), (yyvsp[0].plist), 0, 0, 0, (yyloc).first_line, (yyloc).last_line);
		  firstRdma = true;
		}
#line 4759 "y.tab.c" /* yacc.c:1646  */
    break;

  case 380:
#line 1351 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[-4].strval), (yyvsp[0].plist), 0, 0, (yyvsp[-2].strval), (yyloc).first_line, (yyloc).last_line);
		  firstRdma = true;
		}
#line 4768 "y.tab.c" /* yacc.c:1646  */
    break;

  case 381:
#line 1358 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.entrylist) = new EntryList((yyvsp[0].entry)); }
#line 4774 "y.tab.c" /* yacc.c:1646  */
    break;

  case 382:
#line 1360 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.entrylist) = new EntryList((yyvsp[-2].entry),(yyvsp[0].entrylist)); }
#line 4780 "y.tab.c" /* yacc.c:1646  */
    break;

  case 383:
#line 1364 "xi-grammar.y" /* yacc.c:1646  */
    { in_bracket=1; }
#line 4786 "y.tab.c" /* yacc.c:1646  */
    break;

  case 384:
#line 1367 "xi-grammar.y" /* yacc.c:1646  */
    { in_bracket=0; }
#line 4792 "y.tab.c" /* yacc.c:1646  */
    break;

  case 385:
#line 1371 "xi-grammar.y" /* yacc.c:1646  */
    { if (!macroDefined((yyvsp[0].strval), 1)) in_comment = 1; }
#line 4798 "y.tab.c" /* yacc.c:1646  */
    break;

  case 386:
#line 1375 "xi-grammar.y" /* yacc.c:1646  */
    { if (!macroDefined((yyvsp[0].strval), 0)) in_comment = 1; }
#line 4804 "y.tab.c" /* yacc.c:1646  */
    break;


#line 4808 "y.tab.c" /* yacc.c:1646  */
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
#line 1378 "xi-grammar.y" /* yacc.c:1906  */


void yyerror(const char *s) 
{
	fprintf(stderr, "[PARSE-ERROR] Unexpected/missing token at line %d. Current token being parsed: '%s'.\n", lineno, yytext);
}
