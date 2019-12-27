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
    NOCOPYDEVICE = 296,
    PACKED = 297,
    VARSIZE = 298,
    ENTRY = 299,
    FOR = 300,
    FORALL = 301,
    WHILE = 302,
    WHEN = 303,
    OVERLAP = 304,
    SERIAL = 305,
    IF = 306,
    ELSE = 307,
    PYTHON = 308,
    LOCAL = 309,
    NAMESPACE = 310,
    USING = 311,
    IDENT = 312,
    NUMBER = 313,
    LITERAL = 314,
    CPROGRAM = 315,
    HASHIF = 316,
    HASHIFDEF = 317,
    INT = 318,
    LONG = 319,
    SHORT = 320,
    CHAR = 321,
    FLOAT = 322,
    DOUBLE = 323,
    UNSIGNED = 324,
    ACCEL = 325,
    READWRITE = 326,
    WRITEONLY = 327,
    ACCELBLOCK = 328,
    MEMCRITICAL = 329,
    REDUCTIONTARGET = 330,
    CASE = 331,
    TYPENAME = 332
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
#define NOCOPYDEVICE 296
#define PACKED 297
#define VARSIZE 298
#define ENTRY 299
#define FOR 300
#define FORALL 301
#define WHILE 302
#define WHEN 303
#define OVERLAP 304
#define SERIAL 305
#define IF 306
#define ELSE 307
#define PYTHON 308
#define LOCAL 309
#define NAMESPACE 310
#define USING 311
#define IDENT 312
#define NUMBER 313
#define LITERAL 314
#define CPROGRAM 315
#define HASHIF 316
#define HASHIFDEF 317
#define INT 318
#define LONG 319
#define SHORT 320
#define CHAR 321
#define FLOAT 322
#define DOUBLE 323
#define UNSIGNED 324
#define ACCEL 325
#define READWRITE 326
#define WRITEONLY 327
#define ACCELBLOCK 328
#define MEMCRITICAL 329
#define REDUCTIONTARGET 330
#define CASE 331
#define TYPENAME 332

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

#line 353 "y.tab.c" /* yacc.c:355  */
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

#line 384 "y.tab.c" /* yacc.c:358  */

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
#define YYFINAL  58
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   1527

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  94
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  117
/* YYNRULES -- Number of rules.  */
#define YYNRULES  388
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  775

/* YYTRANSLATE[YYX] -- Symbol number corresponding to YYX as returned
   by yylex, with out-of-bounds checking.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   332

#define YYTRANSLATE(YYX)                                                \
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, without out-of-bounds checking.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    88,     2,
      86,    87,    85,     2,    82,    93,    89,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    79,    78,
      83,    92,    84,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    90,     2,    91,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    80,     2,    81,     2,     2,     2,     2,
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
      75,    76,    77
};

#if YYDEBUG
  /* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   198,   198,   203,   206,   211,   212,   216,   218,   223,
     224,   229,   231,   232,   233,   235,   236,   237,   239,   240,
     241,   242,   243,   247,   248,   249,   250,   251,   252,   253,
     254,   255,   256,   257,   258,   259,   260,   261,   262,   263,
     264,   265,   266,   269,   270,   271,   272,   273,   274,   275,
     276,   277,   278,   279,   281,   283,   284,   287,   288,   289,
     290,   294,   296,   302,   309,   313,   320,   322,   327,   328,
     332,   334,   336,   338,   340,   353,   355,   357,   359,   365,
     367,   369,   371,   373,   375,   377,   379,   381,   383,   391,
     393,   395,   399,   401,   406,   407,   412,   413,   417,   419,
     421,   423,   425,   427,   429,   431,   433,   435,   437,   439,
     441,   443,   445,   449,   450,   455,   463,   465,   469,   473,
     475,   479,   483,   485,   487,   489,   491,   493,   497,   499,
     501,   503,   505,   509,   511,   513,   515,   517,   519,   523,
     525,   527,   529,   531,   533,   537,   541,   546,   547,   551,
     555,   560,   561,   566,   567,   577,   579,   583,   585,   590,
     591,   595,   597,   602,   603,   607,   612,   613,   617,   619,
     623,   625,   630,   631,   635,   636,   639,   643,   645,   649,
     651,   653,   658,   659,   663,   665,   669,   671,   675,   679,
     683,   689,   693,   695,   699,   701,   705,   709,   713,   717,
     719,   724,   725,   730,   731,   733,   735,   744,   746,   748,
     750,   752,   754,   758,   760,   764,   768,   770,   772,   774,
     776,   780,   782,   787,   794,   798,   800,   802,   803,   805,
     807,   809,   813,   815,   817,   823,   829,   838,   840,   842,
     848,   856,   858,   861,   865,   869,   871,   876,   878,   886,
     888,   890,   892,   894,   896,   898,   900,   902,   904,   906,
     909,   919,   936,   953,   955,   959,   964,   965,   967,   974,
     976,   980,   982,   984,   986,   988,   990,   992,   994,   996,
     998,  1000,  1002,  1004,  1006,  1008,  1010,  1012,  1016,  1025,
    1027,  1029,  1034,  1035,  1037,  1046,  1047,  1049,  1055,  1061,
    1067,  1075,  1082,  1090,  1097,  1099,  1101,  1103,  1108,  1119,
    1130,  1143,  1144,  1145,  1148,  1149,  1150,  1151,  1158,  1164,
    1173,  1180,  1186,  1192,  1200,  1202,  1206,  1208,  1212,  1214,
    1218,  1220,  1225,  1226,  1230,  1232,  1234,  1238,  1240,  1244,
    1246,  1250,  1252,  1254,  1262,  1265,  1268,  1270,  1272,  1276,
    1278,  1280,  1282,  1284,  1286,  1288,  1290,  1292,  1294,  1296,
    1298,  1302,  1304,  1306,  1308,  1310,  1312,  1314,  1317,  1320,
    1322,  1324,  1326,  1328,  1330,  1341,  1342,  1344,  1348,  1352,
    1356,  1360,  1365,  1372,  1374,  1378,  1381,  1385,  1389
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
  "VOID", "CONST", "NOCOPY", "NOCOPYPOST", "NOCOPYDEVICE", "PACKED",
  "VARSIZE", "ENTRY", "FOR", "FORALL", "WHILE", "WHEN", "OVERLAP",
  "SERIAL", "IF", "ELSE", "PYTHON", "LOCAL", "NAMESPACE", "USING", "IDENT",
  "NUMBER", "LITERAL", "CPROGRAM", "HASHIF", "HASHIFDEF", "INT", "LONG",
  "SHORT", "CHAR", "FLOAT", "DOUBLE", "UNSIGNED", "ACCEL", "READWRITE",
  "WRITEONLY", "ACCELBLOCK", "MEMCRITICAL", "REDUCTIONTARGET", "CASE",
  "TYPENAME", "';'", "':'", "'{'", "'}'", "','", "'<'", "'>'", "'*'",
  "'('", "')'", "'&'", "'.'", "'['", "']'", "'='", "'-'", "$accept",
  "File", "ModuleEList", "OptExtern", "OneOrMoreSemiColon", "OptSemiColon",
  "Name", "QualName", "Module", "ConstructEList", "ConstructList",
  "ConstructSemi", "Construct", "TParam", "TParamList", "TParamEList",
  "OptTParams", "BuiltinType", "NamedType", "QualNamedType", "SimpleType",
  "OnePtrType", "PtrType", "FuncType", "BaseType", "BaseDataType",
  "RestrictedType", "Type", "ArrayDim", "Dim", "DimList", "Readonly",
  "ReadonlyMsg", "OptVoid", "MAttribs", "MAttribList", "MAttrib",
  "CAttribs", "CAttribList", "PythonOptions", "ArrayAttrib",
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
     325,   326,   327,   328,   329,   330,   331,   332,    59,    58,
     123,   125,    44,    60,    62,    42,    40,    41,    38,    46,
      91,    93,    61,    45
};
# endif

#define YYPACT_NINF -665

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-665)))

#define YYTABLE_NINF -340

#define yytable_value_is_error(Yytable_value) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
     157,  1285,  1285,    51,  -665,   157,  -665,  -665,  -665,  -665,
    -665,  -665,  -665,  -665,  -665,  -665,  -665,  -665,  -665,  -665,
    -665,  -665,  -665,  -665,  -665,  -665,  -665,  -665,  -665,  -665,
    -665,  -665,  -665,  -665,  -665,  -665,  -665,  -665,  -665,  -665,
    -665,  -665,  -665,  -665,  -665,  -665,  -665,  -665,  -665,  -665,
    -665,  -665,  -665,  -665,  -665,  -665,   122,   122,  -665,  -665,
    -665,   893,   -21,  -665,  -665,  -665,    73,  1285,   246,  1285,
    1285,   240,  1041,    54,   224,   893,  -665,  -665,  -665,  -665,
     274,    96,   133,  -665,   100,  -665,  -665,  -665,   -21,    69,
    1327,   166,   166,    12,   -15,   134,   134,   134,   134,   156,
     168,  1285,   218,   144,   893,  -665,  -665,  -665,  -665,  -665,
    -665,  -665,  -665,   229,  -665,  -665,  -665,  -665,   204,  -665,
    -665,  -665,  -665,  -665,  -665,  -665,  -665,  -665,  -665,  -665,
     -21,  -665,  -665,  -665,  1174,  1435,   893,   100,   235,   214,
      69,   216,   437,  -665,  1450,  -665,   110,  -665,  -665,  -665,
    -665,   282,   133,   200,  -665,  -665,   232,   259,   265,  -665,
      36,   133,  -665,   133,   133,   285,   133,   291,  -665,    23,
    1285,  1285,  1285,  1285,    94,   310,   314,   227,  1285,  -665,
    -665,  -665,  1348,   330,   134,   134,   134,   134,   310,   168,
    -665,  -665,  -665,  -665,  -665,   -21,  -665,  -665,  -665,  -665,
    -665,  -665,  -665,  -665,  -665,  -665,  -665,  -665,  -665,  -665,
    -665,   363,  -665,  -665,  -665,   334,   349,  1435,   232,   259,
     265,    39,  -665,   -15,   351,    32,    69,   374,    69,   359,
    -665,   204,   360,    -3,  -665,  -665,  -665,   261,  -665,  -665,
     200,  1417,  -665,  -665,  -665,  -665,  -665,   380,   273,   382,
      60,   103,   347,   361,   358,   -15,  -665,  -665,   381,   391,
     393,   398,   398,   398,   398,  -665,  1285,   387,   399,   389,
     108,  1285,   430,  1285,  -665,  -665,   395,   405,   408,   806,
     -30,   117,  1285,   410,   407,   204,  1285,  1285,  1285,  1285,
    1285,  1285,  -665,  -665,  -665,  1174,   455,  -665,   288,   418,
    1285,  -665,  -665,  -665,   415,   417,   421,   406,    69,   -21,
     133,  -665,  -665,  -665,  -665,  -665,   428,  -665,   435,  -665,
    1285,   423,   431,   432,  -665,   433,  -665,    69,   166,  1417,
     166,   166,  1417,   166,  -665,  -665,    23,  -665,   -15,   311,
     311,   311,   311,   434,  -665,   430,  -665,   398,   398,  -665,
     227,    15,   440,   438,   231,   442,   137,  -665,   443,  1348,
    -665,  -665,   398,   398,   398,   398,   398,   341,  -665,   445,
     447,   448,   393,    69,   374,    69,    69,  -665,    60,  1417,
    -665,   451,   450,   452,  -665,  -665,   441,  -665,   453,   459,
     458,   133,   462,   460,  -665,   466,  -665,   254,   -21,  -665,
    -665,  -665,  -665,  -665,  -665,   311,   311,  -665,  -665,  -665,
    1450,    24,   469,   464,  1450,  -665,  -665,   467,  -665,  -665,
    -665,  -665,  -665,   311,   311,   311,   311,   311,   505,   -21,
     468,   471,  -665,   472,  -665,  -665,  -665,  -665,  -665,  -665,
     493,   494,  -665,  -665,  -665,  -665,   499,  -665,   125,   508,
    -665,   -15,  -665,   726,   480,   474,   204,   254,  -665,  -665,
    -665,  -665,  1285,  -665,  -665,  1285,  -665,   538,  -665,  -665,
    -665,  -665,  -665,   515,   510,  -665,  1369,  -665,  1402,  -665,
     166,   166,   166,  -665,  1119,  1062,  -665,   204,   -21,  -665,
     514,   438,   438,   204,  -665,  1450,  1450,  1450,  -665,  1285,
      69,   518,   521,   522,   523,   525,   526,   524,   528,   472,
    1285,  -665,   529,   204,  -665,  -665,   -21,  1285,    69,    69,
      69,     9,   532,  1402,  -665,  -665,  -665,  -665,  -665,   581,
     527,   472,  -665,   -21,   542,   540,   543,   544,  -665,   392,
    -665,  -665,  -665,  1285,  -665,   547,   546,   547,   580,   559,
     582,   547,   560,   367,   -21,    69,  -665,  -665,  -665,   622,
    -665,  -665,  -665,  -665,  -665,   100,  -665,   472,  -665,    69,
     586,    69,   194,   562,   579,   601,  -665,   573,    69,   541,
     575,   348,   216,   568,   527,   571,  -665,   592,   583,   584,
    -665,    69,   580,   409,  -665,   591,   517,    69,   584,   547,
     585,   547,   593,   582,   547,   596,    69,   597,   541,  -665,
     204,  -665,   204,   621,  -665,   315,   573,    69,   547,  -665,
     618,   441,  -665,  -665,   599,  -665,  -665,   216,   656,    69,
     626,    69,   601,   573,    69,   541,   216,  -665,  -665,  -665,
    -665,  -665,  -665,  -665,  -665,  -665,  1285,   605,   603,   595,
      69,   609,    69,   367,  -665,   472,  -665,   204,   367,   638,
     613,   604,   584,   611,    69,   584,   615,   204,   614,  1450,
    1310,  -665,   216,    69,   620,   619,  -665,  -665,   627,   842,
    -665,    69,   547,   882,  -665,   216,   933,  -665,  -665,  1285,
    1285,    69,   629,  -665,  1285,   584,    69,  -665,   638,   367,
    -665,   631,    69,   367,  -665,   204,   367,   638,  -665,   136,
     176,   623,  1285,   204,   940,   628,  -665,   632,    69,   636,
     635,  -665,   637,  -665,  -665,  1285,  1285,  1211,   633,  1285,
    -665,   243,   -21,   367,  -665,    69,  -665,   584,    69,  -665,
     638,   270,  -665,   630,   377,  1285,   281,  -665,   639,   584,
     949,   640,  -665,  -665,  -665,  -665,  -665,  -665,  -665,   956,
     367,  -665,    69,   367,  -665,   642,   584,   661,  -665,   992,
    -665,   367,  -665,   662,  -665
};

  /* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
     Performed when YYTABLE does not specify something else to do.  Zero
     means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       3,     0,     0,     0,     2,     3,    12,    13,    14,    15,
      16,    17,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    31,    35,    36,    37,    38,
      39,    40,    41,    42,    32,    33,    34,    43,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
      11,    56,    57,    58,    59,    60,     0,     0,     1,     4,
       7,     0,    66,    64,    65,    88,     6,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    87,    85,    86,     8,
       0,     0,     0,    61,    71,   387,   388,   302,   264,   295,
       0,   151,   151,   151,     0,   159,   159,   159,   159,     0,
     153,     0,     0,     0,     0,    79,   225,   226,    73,    80,
      81,    82,    83,     0,    84,    72,   228,   227,     9,   259,
     251,   252,   253,   254,   255,   257,   258,   256,   249,   250,
      77,    78,    69,   268,     0,     0,     0,    70,     0,   296,
     295,     0,     0,   112,     0,    98,    99,   100,   101,   109,
     110,     0,     0,    96,   116,   117,   122,   123,   124,   125,
     144,     0,   152,     0,     0,     0,     0,   241,   229,     0,
       0,     0,     0,     0,     0,     0,   166,     0,     0,   231,
     243,   230,     0,     0,   159,   159,   159,   159,     0,   153,
     216,   217,   218,   219,   220,    10,    67,   288,   271,   272,
     273,   274,   280,   281,   282,   287,   275,   276,   277,   278,
     279,   163,   283,   285,   286,     0,   269,     0,   128,   129,
     130,   138,   265,     0,     0,     0,   295,   292,   295,     0,
     303,     0,     0,   126,   108,   111,   102,   103,   106,   107,
      96,    94,   114,   118,   119,   120,   127,     0,   143,     0,
     147,   235,   232,     0,   237,     0,   170,   171,     0,   161,
      96,   182,   182,   182,   182,   165,     0,     0,   168,     0,
       0,     0,     0,     0,   157,   158,     0,   155,   179,     0,
       0,   125,     0,   213,     0,     9,     0,     0,     0,     0,
       0,     0,   164,   284,   267,     0,   131,   132,   137,     0,
       0,    76,    63,    62,     0,   293,     0,     0,   295,   263,
       0,   104,   105,   115,    90,    91,    92,    95,     0,    89,
       0,   142,     0,     0,   385,   147,   149,   295,   151,     0,
     151,   151,     0,   151,   242,   160,     0,   113,     0,     0,
       0,     0,     0,     0,   191,     0,   167,   182,   182,   154,
       0,   172,     0,   201,    61,     0,     0,   211,   203,     0,
     215,    75,   182,   182,   182,   182,   182,     0,   270,   136,
       0,     0,    96,   295,   292,   295,   295,   300,   147,     0,
      97,     0,     0,     0,   141,   148,     0,   145,     0,     0,
       0,     0,     0,     0,   162,   184,   183,     0,   221,   186,
     187,   188,   189,   190,   169,     0,     0,   156,   173,   180,
       0,   172,     0,     0,     0,   209,   210,     0,   204,   205,
     206,   212,   214,     0,     0,     0,     0,     0,   172,   199,
       0,     0,   135,     0,   298,   294,   299,   297,   150,    93,
       0,     0,   140,   386,   146,   236,     0,   233,     0,     0,
     238,     0,   248,     0,     0,     0,     0,     0,   244,   245,
     192,   193,     0,   178,   181,     0,   202,     0,   194,   195,
     196,   197,   198,     0,     0,   134,     0,    74,     0,   139,
     151,   151,   151,   185,     0,     0,   246,     9,   247,   224,
     174,   201,   201,     0,   133,     0,     0,     0,   329,   304,
     295,   324,     0,     0,     0,     0,     0,     0,    61,     0,
       0,   222,     0,     0,   207,   208,   200,     0,   295,   295,
     295,   172,     0,     0,   328,   121,   234,   240,   239,     0,
       0,     0,   175,   176,     0,     0,     0,     0,   301,     0,
     305,   307,   325,     0,   374,     0,     0,     0,     0,     0,
     345,     0,     0,     0,   334,   295,   261,   363,   335,   332,
     308,   309,   310,   290,   289,   291,   306,     0,   380,   295,
       0,   295,     0,   383,     0,     0,   344,     0,   295,     0,
       0,     0,     0,     0,     0,     0,   378,     0,     0,     0,
     381,   295,     0,     0,   347,     0,     0,   295,     0,     0,
       0,     0,     0,   345,     0,     0,   295,     0,   341,   343,
       9,   338,     9,     0,   260,     0,     0,   295,     0,   379,
       0,     0,   384,   346,     0,   362,   340,     0,     0,   295,
       0,   295,     0,     0,   295,     0,     0,   364,   342,   336,
     373,   333,   311,   312,   313,   331,     0,     0,   326,     0,
     295,     0,   295,     0,   371,     0,   348,     9,     0,   375,
       0,     0,     0,     0,   295,     0,     0,     9,     0,     0,
       0,   330,     0,   295,     0,     0,   382,   361,     0,     0,
     369,   295,     0,     0,   350,     0,     0,   351,   360,     0,
       0,   295,     0,   327,     0,     0,   295,   372,   375,     0,
     376,     0,   295,     0,   358,     9,     0,   375,   314,     0,
       0,     0,     0,     0,     0,     0,   370,     0,   295,     0,
       0,   349,     0,   356,   322,     0,     0,     0,     0,     0,
     320,     0,   262,     0,   366,   295,   377,     0,   295,   359,
     375,     0,   316,     0,     0,     0,     0,   323,     0,     0,
       0,     0,   357,   319,   318,   317,   315,   321,   365,     0,
       0,   353,   295,     0,   367,     0,     0,     0,   352,     0,
     368,     0,   354,     0,   355
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -665,  -665,   714,  -665,   -51,  -282,    -1,   -57,   685,   722,
     -23,  -665,  -665,  -665,  -295,  -665,  -210,  -665,  -133,   -82,
    -127,  -131,  -119,  -172,   641,   564,  -665,   -88,  -665,  -665,
    -267,  -665,  -665,   -79,   598,   436,  -665,   -42,   461,  -665,
    -665,   608,   463,  -665,   267,  -665,  -665,  -314,  -665,   -56,
     353,  -665,  -665,  -665,   -18,  -665,  -665,  -665,  -665,  -665,
    -665,  -333,   449,  -665,   439,   733,  -665,  -208,   355,   734,
    -665,  -665,   567,  -665,  -665,  -665,  -665,   369,  -665,   339,
     384,   549,  -665,  -665,   485,   -80,  -478,   -64,  -549,  -665,
    -665,  -664,  -665,  -665,  -407,   191,  -487,  -665,  -665,   280,
    -550,   233,  -471,   263,  -515,  -665,  -502,  -628,  -542,  -574,
    -432,  -665,   275,   296,   248,  -665,  -665
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     3,     4,    72,   398,   196,   260,   153,     5,    63,
      73,    74,    75,   316,   317,   318,   242,   154,   261,   155,
     156,   157,   158,   159,   160,   221,   222,   319,   386,   325,
     326,   106,   107,   163,   178,   276,   277,   170,   258,   293,
     268,   175,   269,   259,   410,   513,   411,   412,   108,   339,
     396,   109,   110,   111,   176,   112,   190,   191,   192,   193,
     194,   415,   357,   283,   284,   454,   114,   399,   455,   456,
     116,   117,   168,   181,   457,   458,   131,   459,    76,   223,
     135,   215,   216,   566,   306,   586,   500,   555,   231,   501,
     647,   709,   692,   648,   502,   649,   477,   616,   584,   556,
     580,   595,   607,   577,   557,   609,   581,   680,   587,   620,
     569,   573,   574,   327,   444,    77,    78
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[] =
{
      56,    57,   161,   361,   219,    62,    62,    89,   218,   141,
     281,    84,   167,   164,   166,   232,   220,   518,   519,   520,
      88,   416,   530,   130,   628,   137,   408,   354,   558,   589,
     313,   611,   408,   612,   389,   246,   598,   392,   262,   263,
     264,   408,    83,   624,   559,   278,   626,   302,   731,   162,
     337,    58,   132,   256,   171,   172,   173,    79,   385,   355,
     229,   741,   152,   744,   608,   746,    81,   195,    85,    86,
     716,   503,   594,   596,   246,   651,   257,   297,   657,   723,
     585,   183,   558,   247,   439,   590,   219,   667,   683,   303,
     218,   686,   267,   608,   282,   240,   409,   463,   220,   538,
     179,   539,   165,   675,   250,  -177,   251,   252,   678,   254,
     674,   438,   752,   224,   473,   571,   542,    80,   654,   578,
     608,   714,   247,   694,   248,   249,   659,   298,   299,   139,
     596,   695,   400,   401,   402,   118,   705,   638,   347,   138,
     348,   300,   286,   287,   288,   289,   304,   265,   307,   717,
     324,    83,   266,   720,   715,   140,   722,   271,   514,   515,
       1,     2,   433,   750,   666,    83,   266,   629,   676,   631,
     290,   152,   634,   167,   234,   759,   136,   700,   235,   138,
     309,   704,   138,   748,   707,   152,   652,   281,   267,   328,
      83,   691,   769,   749,    83,   418,   419,   460,   461,  -203,
      60,  -203,    61,   162,   138,   511,   340,   341,   342,   356,
     765,   481,   734,   767,   152,   468,   469,   470,   471,   472,
     724,   773,   725,   240,   169,   726,   727,   182,   377,   728,
     119,   120,   121,   122,   195,   123,   124,   125,   126,   127,
     184,   185,   186,   187,   188,   189,   174,   387,   761,   388,
     702,   390,   391,   378,   393,   452,   395,   764,   177,   729,
      90,    91,    92,    93,    94,   343,   538,   772,   128,   274,
     275,   282,   101,   102,   420,   133,   103,   180,   353,   138,
     476,   358,    60,   241,   324,   362,   363,   364,   365,   366,
     367,   405,   406,   434,   226,   436,   437,   230,   453,   372,
     227,    82,    60,    83,   228,   129,   423,   424,   425,   426,
     427,  -266,  -266,  -201,   225,  -201,   429,   243,    60,   381,
      87,   642,   462,   414,   311,   312,   466,   747,   639,   725,
     640,  -266,   726,   727,   448,  -223,   728,  -266,  -266,  -266,
    -266,  -266,  -266,  -266,   244,   236,   237,   238,   239,   544,
     245,  -266,   143,   144,   219,   253,   725,   753,   218,   726,
     727,   321,   322,   728,   134,   757,   220,   725,   544,   395,
     726,   727,    83,   255,   728,   677,   369,   370,   145,   146,
     147,   148,   149,   150,   151,   688,   643,   644,   499,    60,
     499,   397,   152,   545,   546,   547,   548,   549,   550,   551,
     270,   504,   505,   506,   272,   488,   645,   517,   517,   517,
     544,   285,   545,   546,   547,   548,   549,   550,   551,    60,
     522,   428,   292,   721,   552,   294,   138,  -302,    87,  -337,
     329,   295,   301,   330,   305,   499,   195,   138,   535,   536,
     537,   332,   516,   552,   333,   310,   308,    87,  -302,    83,
     563,   564,   331,  -302,   545,   546,   547,   548,   549,   550,
     551,   490,   533,   725,   491,   320,   726,   727,   755,  -302,
     728,   323,   335,   336,   143,   582,   241,   338,   344,   554,
     346,   345,   565,   265,   509,   552,   349,   350,   351,    87,
     623,   360,   359,   297,    83,  -302,   373,   376,   521,   374,
     145,   146,   147,   148,   149,   150,   151,   371,   375,   531,
     379,   621,   382,   597,   152,   606,   534,   627,   544,   380,
     383,   384,   408,   324,   453,   403,   636,   646,   544,   413,
     414,   417,   443,   554,   430,   356,   431,   432,   440,   441,
     445,   442,   567,   446,   606,   447,   449,   450,   451,   660,
     464,   662,   650,   465,   665,   487,   467,   474,   476,   195,
     475,   195,   545,   546,   547,   548,   549,   550,   551,   664,
     672,   606,   545,   546,   547,   548,   549,   550,   551,   478,
     544,   690,   646,   479,   685,   480,   599,   600,   601,   548,
     602,   603,   604,   552,   482,   492,   493,    87,  -339,   494,
     523,   701,   544,   552,   512,    60,   195,   553,   524,   525,
     526,   711,   527,   528,   -11,   529,   195,   605,   543,   544,
     532,    87,   719,   541,   545,   546,   547,   548,   549,   550,
     551,   560,   538,   568,   561,   562,   570,   572,   737,   575,
     579,   576,   583,   588,   592,   668,   545,   546,   547,   548,
     549,   550,   551,    87,   195,   552,   610,   544,   751,   593,
     613,   615,   732,   545,   546,   547,   548,   549,   550,   551,
     617,   619,   625,   632,   618,   630,   635,   552,   637,   641,
     656,    87,   766,   661,   669,   670,   671,   673,   708,   710,
     679,   681,   684,   713,   552,   682,   687,   689,   653,   696,
     697,   545,   546,   547,   548,   549,   550,   551,   698,   718,
     735,   708,   712,   736,   730,   738,   739,   745,   740,    59,
     758,   754,   762,   768,   708,   742,   708,   133,   708,  -266,
    -266,  -266,   552,  -266,  -266,  -266,   658,  -266,  -266,  -266,
    -266,  -266,   770,   774,   756,  -266,  -266,  -266,  -266,  -266,
    -266,  -266,  -266,  -266,  -266,  -266,  -266,   105,  -266,  -266,
    -266,  -266,  -266,  -266,  -266,  -266,  -266,  -266,  -266,  -266,
    -266,  -266,  -266,  -266,  -266,  -266,  -266,  -266,  -266,    64,
    -266,   296,  -266,  -266,   273,   233,   407,   291,   540,  -266,
    -266,  -266,  -266,  -266,  -266,  -266,  -266,   394,   422,  -266,
    -266,  -266,  -266,  -266,   483,   113,   115,   421,   404,     6,
       7,     8,   489,     9,    10,    11,   484,    12,    13,    14,
      15,    16,   334,   486,   510,    17,    18,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,   485,    29,    30,
      31,    32,    33,   544,   368,    34,    35,    36,    37,    38,
      39,    40,    41,    42,    43,    44,    45,    46,    47,   435,
      48,   693,    49,    50,   614,   663,   633,   622,   591,   655,
       0,     0,     0,     0,     0,     0,    51,     0,     0,    52,
      53,    54,    55,   544,     0,     0,     0,   545,   546,   547,
     548,   549,   550,   551,    65,   352,    -5,    -5,    66,    -5,
      -5,    -5,    -5,    -5,    -5,    -5,    -5,    -5,    -5,    -5,
       0,    -5,    -5,     0,     0,    -5,     0,     0,   552,     0,
       0,     0,   699,     0,     0,     0,     0,   545,   546,   547,
     548,   549,   550,   551,   544,     0,     0,     0,     0,     0,
       0,   544,     0,     0,     0,     0,     0,     0,    67,    68,
     544,     0,     0,     0,    69,    70,     0,   544,   552,     0,
       0,     0,   703,     0,     0,     0,    71,     0,     0,     0,
       0,     0,     0,    -5,   -68,     0,     0,     0,   545,   546,
     547,   548,   549,   550,   551,   545,   546,   547,   548,   549,
     550,   551,     0,   544,   545,   546,   547,   548,   549,   550,
     551,   545,   546,   547,   548,   549,   550,   551,     0,   552,
       0,     0,     0,   706,     0,     0,   552,     0,     0,     0,
     733,     0,     0,     0,     0,   552,     0,     0,     0,   760,
       0,     0,   552,     0,     0,     0,   763,   545,   546,   547,
     548,   549,   550,   551,     1,     2,     0,    90,    91,    92,
      93,    94,    95,    96,    97,    98,    99,   100,     0,   101,
     102,     0,     0,   103,     0,     6,     7,     8,   552,     9,
      10,    11,   771,    12,    13,    14,    15,    16,     0,     0,
       0,    17,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,     0,    29,    30,    31,    32,    33,   143,
     217,    34,    35,    36,    37,    38,    39,    40,    41,    42,
      43,    44,    45,    46,    47,     0,    48,     0,    49,   508,
     197,   104,     0,     0,     0,   145,   146,   147,   148,   149,
     150,   151,    51,     0,     0,    52,    53,    54,    55,   152,
     198,     0,   199,   200,   201,   202,   203,   204,     0,     0,
     205,   206,   207,   208,   209,   210,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   211,   212,     0,   197,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   507,
       0,     0,     0,   213,   214,   198,     0,   199,   200,   201,
     202,   203,   204,     0,     0,   205,   206,   207,   208,   209,
     210,     0,     0,     0,     6,     7,     8,     0,     9,    10,
      11,     0,    12,    13,    14,    15,    16,   211,   212,     0,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,     0,    29,    30,    31,    32,    33,   213,   214,
      34,    35,    36,    37,    38,    39,    40,    41,    42,    43,
      44,    45,    46,    47,     0,    48,     0,    49,    50,   743,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    51,     0,     0,    52,    53,    54,    55,     6,     7,
       8,     0,     9,    10,    11,     0,    12,    13,    14,    15,
      16,     0,     0,     0,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,   642,    29,    30,    31,
      32,    33,     0,     0,    34,    35,    36,    37,    38,    39,
      40,    41,    42,    43,    44,    45,    46,    47,     0,    48,
       0,    49,    50,   142,     0,     0,     0,   143,   144,     0,
       0,     0,     0,     0,     0,    51,     0,     0,    52,    53,
      54,    55,     0,     0,   143,   144,   279,    83,     0,     0,
       0,     0,     0,   145,   146,   147,   148,   149,   150,   151,
       0,   643,   644,     0,    83,   143,   144,   152,     0,     0,
     145,   146,   147,   148,   149,   150,   151,     0,     0,     0,
       0,     0,     0,     0,   152,    83,   143,   144,   495,   496,
     497,   145,   146,   147,   148,   149,   150,   151,     0,     0,
       0,     0,     0,     0,     0,   280,    83,     0,     0,     0,
       0,     0,   145,   146,   147,   148,   149,   150,   151,   143,
     144,   495,   496,   497,     0,     0,   152,     0,     0,     0,
       0,     0,     0,     0,   143,   144,   498,     0,     0,    83,
       0,     0,     0,     0,     0,   145,   146,   147,   148,   149,
     150,   151,   143,   217,    83,   314,   315,     0,     0,   152,
     145,   146,   147,   148,   149,   150,   151,   143,   144,     0,
       0,     0,    83,     0,   152,     0,     0,     0,   145,   146,
     147,   148,   149,   150,   151,     0,     0,    83,     0,     0,
       0,     0,   152,   145,   146,   147,   148,   149,   150,   151,
       0,     0,     0,     0,     0,     0,     0,   152
};

static const yytype_int16 yycheck[] =
{
       1,     2,    90,   285,   135,    56,    57,    71,   135,    89,
     182,    68,    94,    92,    93,   142,   135,   495,   496,   497,
      71,   354,   509,    74,   598,    82,    17,    57,   530,   571,
     240,   581,    17,   582,   329,    38,   578,   332,   171,   172,
     173,    17,    57,   593,   531,   178,   596,    15,   712,    37,
     260,     0,    75,    30,    96,    97,    98,    78,   325,    89,
     140,   725,    77,   727,   579,   729,    67,   118,    69,    70,
     698,   478,   574,   575,    38,   617,    53,    38,   627,   707,
     567,   104,   584,    86,   379,   572,   217,   636,   662,    57,
     217,   665,   174,   608,   182,   152,    81,   411,   217,    90,
     101,    92,    90,   653,   161,    81,   163,   164,   658,   166,
     652,   378,   740,   136,   428,   547,   523,    44,   620,   551,
     635,   695,    86,   672,    88,    89,   628,    88,    89,    60,
     632,   673,   340,   341,   342,    81,   685,   608,   271,    79,
     273,   223,   184,   185,   186,   187,   226,    53,   228,   699,
      90,    57,    58,   703,   696,    86,   706,   175,   491,   492,
       3,     4,   372,   737,   635,    57,    58,   599,   655,   601,
     188,    77,   604,   255,    64,   749,    80,   679,    68,    79,
     231,   683,    79,   733,   686,    77,   618,   359,   270,    86,
      57,   669,   766,   735,    57,    58,    59,   405,   406,    82,
      78,    84,    80,    37,    79,   487,   262,   263,   264,    92,
     760,    86,   714,   763,    77,   423,   424,   425,   426,   427,
      84,   771,    86,   280,    90,    89,    90,    83,   308,    93,
       6,     7,     8,     9,   285,    11,    12,    13,    14,    15,
      11,    12,    13,    14,    15,    16,    90,   327,   750,   328,
     682,   330,   331,   310,   333,     1,   338,   759,    90,    83,
       6,     7,     8,     9,    10,   266,    90,   769,    44,    42,
      43,   359,    18,    19,   356,     1,    22,    59,   279,    79,
      86,   282,    78,    83,    90,   286,   287,   288,   289,   290,
     291,   347,   348,   373,    80,   375,   376,    81,    44,   300,
      86,    55,    78,    57,    90,    81,   362,   363,   364,   365,
     366,    37,    38,    82,    79,    84,   367,    85,    78,   320,
      80,     6,   410,    92,    63,    64,   414,    84,   610,    86,
     612,    57,    89,    90,   391,    81,    93,    63,    64,    65,
      66,    67,    68,    69,    85,    63,    64,    65,    66,     1,
      85,    77,    37,    38,   485,    70,    86,    87,   485,    89,
      90,    88,    89,    93,    90,    84,   485,    86,     1,   451,
      89,    90,    57,    82,    93,   657,    88,    89,    63,    64,
      65,    66,    67,    68,    69,   667,    71,    72,   476,    78,
     478,    80,    77,    45,    46,    47,    48,    49,    50,    51,
      90,   480,   481,   482,    90,   456,    91,   495,   496,   497,
       1,    81,    45,    46,    47,    48,    49,    50,    51,    78,
     500,    80,    59,   705,    76,    91,    79,    60,    80,    81,
      83,    82,    81,    86,    60,   523,   487,    79,   518,   519,
     520,    83,   493,    76,    86,    85,    87,    80,    81,    57,
      58,    59,    91,    86,    45,    46,    47,    48,    49,    50,
      51,   462,   513,    86,   465,    85,    89,    90,    91,    60,
      93,    89,    91,    82,    37,   555,    83,    79,    91,   530,
      91,    82,   539,    53,   485,    76,    91,    82,    80,    80,
      81,    84,    82,    38,    57,    86,    81,    91,   499,    82,
      63,    64,    65,    66,    67,    68,    69,    89,    87,   510,
      82,   591,    89,   577,    77,   579,   517,   597,     1,    84,
      89,    89,    17,    90,    44,    91,   606,   615,     1,    89,
      92,    89,    91,   584,    89,    92,    89,    89,    87,    89,
      87,    89,   543,    84,   608,    87,    84,    87,    82,   629,
      81,   631,   616,    89,   634,    81,    89,    89,    86,   610,
      89,   612,    45,    46,    47,    48,    49,    50,    51,   633,
     650,   635,    45,    46,    47,    48,    49,    50,    51,    86,
       1,   669,   670,    89,   664,    86,    45,    46,    47,    48,
      49,    50,    51,    76,    86,    57,    81,    80,    81,    89,
      82,   681,     1,    76,    90,    78,   657,    80,    87,    87,
      87,   691,    87,    87,    86,    91,   667,    76,    37,     1,
      91,    80,   702,    91,    45,    46,    47,    48,    49,    50,
      51,    91,    90,    86,    91,    91,    90,    57,   718,    80,
      80,    59,    20,    57,    82,   646,    45,    46,    47,    48,
      49,    50,    51,    80,   705,    76,    81,     1,   738,    80,
      92,    90,   713,    45,    46,    47,    48,    49,    50,    51,
      78,    87,    81,    80,    91,    90,    80,    76,    81,    58,
      81,    80,   762,    57,    79,    82,    91,    78,   689,   690,
      52,    78,    81,   694,    76,    91,    81,    83,    80,    79,
      81,    45,    46,    47,    48,    49,    50,    51,    81,    78,
      82,   712,    83,    81,    91,    79,    81,    84,    81,     5,
      81,    91,    82,    81,   725,   726,   727,     1,   729,     3,
       4,     5,    76,     7,     8,     9,    80,    11,    12,    13,
      14,    15,    81,    81,   745,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,    29,    30,    72,    32,    33,
      34,    35,    36,    37,    38,    39,    40,    41,    42,    43,
      44,    45,    46,    47,    48,    49,    50,    51,    52,    57,
      54,   217,    56,    57,   176,   144,   350,   189,   521,    63,
      64,    65,    66,    67,    68,    69,    70,   336,   359,    73,
      74,    75,    76,    77,   451,    72,    72,   358,   345,     3,
       4,     5,   457,     7,     8,     9,    90,    11,    12,    13,
      14,    15,   255,   454,   485,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,    29,    30,   453,    32,    33,
      34,    35,    36,     1,   295,    39,    40,    41,    42,    43,
      44,    45,    46,    47,    48,    49,    50,    51,    52,   374,
      54,   670,    56,    57,   584,   632,   603,   592,   572,   621,
      -1,    -1,    -1,    -1,    -1,    -1,    70,    -1,    -1,    73,
      74,    75,    76,     1,    -1,    -1,    -1,    45,    46,    47,
      48,    49,    50,    51,     1,    89,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      -1,    18,    19,    -1,    -1,    22,    -1,    -1,    76,    -1,
      -1,    -1,    80,    -1,    -1,    -1,    -1,    45,    46,    47,
      48,    49,    50,    51,     1,    -1,    -1,    -1,    -1,    -1,
      -1,     1,    -1,    -1,    -1,    -1,    -1,    -1,    55,    56,
       1,    -1,    -1,    -1,    61,    62,    -1,     1,    76,    -1,
      -1,    -1,    80,    -1,    -1,    -1,    73,    -1,    -1,    -1,
      -1,    -1,    -1,    80,    81,    -1,    -1,    -1,    45,    46,
      47,    48,    49,    50,    51,    45,    46,    47,    48,    49,
      50,    51,    -1,     1,    45,    46,    47,    48,    49,    50,
      51,    45,    46,    47,    48,    49,    50,    51,    -1,    76,
      -1,    -1,    -1,    80,    -1,    -1,    76,    -1,    -1,    -1,
      80,    -1,    -1,    -1,    -1,    76,    -1,    -1,    -1,    80,
      -1,    -1,    76,    -1,    -1,    -1,    80,    45,    46,    47,
      48,    49,    50,    51,     3,     4,    -1,     6,     7,     8,
       9,    10,    11,    12,    13,    14,    15,    16,    -1,    18,
      19,    -1,    -1,    22,    -1,     3,     4,     5,    76,     7,
       8,     9,    80,    11,    12,    13,    14,    15,    -1,    -1,
      -1,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      28,    29,    30,    -1,    32,    33,    34,    35,    36,    37,
      38,    39,    40,    41,    42,    43,    44,    45,    46,    47,
      48,    49,    50,    51,    52,    -1,    54,    -1,    56,    57,
       1,    80,    -1,    -1,    -1,    63,    64,    65,    66,    67,
      68,    69,    70,    -1,    -1,    73,    74,    75,    76,    77,
      21,    -1,    23,    24,    25,    26,    27,    28,    -1,    -1,
      31,    32,    33,    34,    35,    36,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    53,    54,    -1,     1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    70,
      -1,    -1,    -1,    74,    75,    21,    -1,    23,    24,    25,
      26,    27,    28,    -1,    -1,    31,    32,    33,    34,    35,
      36,    -1,    -1,    -1,     3,     4,     5,    -1,     7,     8,
       9,    -1,    11,    12,    13,    14,    15,    53,    54,    -1,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    -1,    32,    33,    34,    35,    36,    74,    75,
      39,    40,    41,    42,    43,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    -1,    54,    -1,    56,    57,    58,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    70,    -1,    -1,    73,    74,    75,    76,     3,     4,
       5,    -1,     7,     8,     9,    -1,    11,    12,    13,    14,
      15,    -1,    -1,    -1,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,     6,    32,    33,    34,
      35,    36,    -1,    -1,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    -1,    54,
      -1,    56,    57,    16,    -1,    -1,    -1,    37,    38,    -1,
      -1,    -1,    -1,    -1,    -1,    70,    -1,    -1,    73,    74,
      75,    76,    -1,    -1,    37,    38,    18,    57,    -1,    -1,
      -1,    -1,    -1,    63,    64,    65,    66,    67,    68,    69,
      -1,    71,    72,    -1,    57,    37,    38,    77,    -1,    -1,
      63,    64,    65,    66,    67,    68,    69,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    77,    57,    37,    38,    39,    40,
      41,    63,    64,    65,    66,    67,    68,    69,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    77,    57,    -1,    -1,    -1,
      -1,    -1,    63,    64,    65,    66,    67,    68,    69,    37,
      38,    39,    40,    41,    -1,    -1,    77,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    37,    38,    87,    -1,    -1,    57,
      -1,    -1,    -1,    -1,    -1,    63,    64,    65,    66,    67,
      68,    69,    37,    38,    57,    58,    59,    -1,    -1,    77,
      63,    64,    65,    66,    67,    68,    69,    37,    38,    -1,
      -1,    -1,    57,    -1,    77,    -1,    -1,    -1,    63,    64,
      65,    66,    67,    68,    69,    -1,    -1,    57,    -1,    -1,
      -1,    -1,    77,    63,    64,    65,    66,    67,    68,    69,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    77
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     3,     4,    95,    96,   102,     3,     4,     5,     7,
       8,     9,    11,    12,    13,    14,    15,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    32,
      33,    34,    35,    36,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    54,    56,
      57,    70,    73,    74,    75,    76,   100,   100,     0,    96,
      78,    80,    98,   103,   103,     1,     5,    55,    56,    61,
      62,    73,    97,   104,   105,   106,   172,   209,   210,    78,
      44,   100,    55,    57,   101,   100,   100,    80,    98,   181,
       6,     7,     8,     9,    10,    11,    12,    13,    14,    15,
      16,    18,    19,    22,    80,   102,   125,   126,   142,   145,
     146,   147,   149,   159,   160,   163,   164,   165,    81,     6,
       7,     8,     9,    11,    12,    13,    14,    15,    44,    81,
      98,   170,   104,     1,    90,   174,    80,   101,    79,    60,
      86,   179,    16,    37,    38,    63,    64,    65,    66,    67,
      68,    69,    77,   101,   111,   113,   114,   115,   116,   117,
     118,   121,    37,   127,   127,    90,   127,   113,   166,    90,
     131,   131,   131,   131,    90,   135,   148,    90,   128,   100,
      59,   167,    83,   104,    11,    12,    13,    14,    15,    16,
     150,   151,   152,   153,   154,    98,    99,     1,    21,    23,
      24,    25,    26,    27,    28,    31,    32,    33,    34,    35,
      36,    53,    54,    74,    75,   175,   176,    38,   114,   115,
     116,   119,   120,   173,   104,    79,    80,    86,    90,   179,
      81,   182,   114,   118,    64,    68,    63,    64,    65,    66,
     101,    83,   110,    85,    85,    85,    38,    86,    88,    89,
     101,   101,   101,    70,   101,    82,    30,    53,   132,   137,
     100,   112,   112,   112,   112,    53,    58,   113,   134,   136,
      90,   148,    90,   135,    42,    43,   129,   130,   112,    18,
      77,   117,   121,   157,   158,    81,   131,   131,   131,   131,
     148,   128,    59,   133,    91,    82,   119,    38,    88,    89,
     113,    81,    15,    57,   179,    60,   178,   179,    87,    98,
      85,    63,    64,   110,    58,    59,   107,   108,   109,   121,
      85,    88,    89,    89,    90,   123,   124,   207,    86,    83,
      86,    91,    83,    86,   166,    91,    82,   110,    79,   143,
     143,   143,   143,   100,    91,    82,    91,   112,   112,    91,
      82,    80,    89,   100,    57,    89,    92,   156,   100,    82,
      84,    99,   100,   100,   100,   100,   100,   100,   175,    88,
      89,    89,   100,    81,    82,    87,    91,   179,   101,    82,
      84,   100,    89,    89,    89,   124,   122,   179,   127,   108,
     127,   127,   108,   127,   132,   113,   144,    80,    98,   161,
     161,   161,   161,    91,   136,   143,   143,   129,    17,    81,
     138,   140,   141,    89,    92,   155,   155,    89,    58,    59,
     113,   156,   158,   143,   143,   143,   143,   143,    80,    98,
      89,    89,    89,   110,   179,   178,   179,   179,   124,   108,
      87,    89,    89,    91,   208,    87,    84,    87,   101,    84,
      87,    82,     1,    44,   159,   162,   163,   168,   169,   171,
     161,   161,   121,   141,    81,    89,   121,    89,   161,   161,
     161,   161,   161,   141,    89,    89,    86,   190,    86,    89,
      86,    86,    86,   144,    90,   174,   171,    81,    98,   162,
     100,   100,    57,    81,    89,    39,    40,    41,    87,   121,
     180,   183,   188,   188,   127,   127,   127,    70,    57,   100,
     173,    99,    90,   139,   155,   155,    98,   121,   180,   180,
     180,   100,   179,    82,    87,    87,    87,    87,    87,    91,
     190,   100,    91,    98,   100,   179,   179,   179,    90,    92,
     138,    91,   188,    37,     1,    45,    46,    47,    48,    49,
      50,    51,    76,    80,    98,   181,   193,   198,   200,   190,
      91,    91,    91,    58,    59,   101,   177,   100,    86,   204,
      90,   204,    57,   205,   206,    80,    59,   197,   204,    80,
     194,   200,   179,    20,   192,   190,   179,   202,    57,   202,
     190,   207,    82,    80,   200,   195,   200,   181,   202,    45,
      46,    47,    49,    50,    51,    76,   181,   196,   198,   199,
      81,   194,   182,    92,   193,    90,   191,    78,    91,    87,
     203,   179,   206,    81,   194,    81,   194,   179,   203,   204,
      90,   204,    80,   197,   204,    80,   179,    81,   196,    99,
      99,    58,     6,    71,    72,    91,   121,   184,   187,   189,
     181,   202,   204,    80,   200,   208,    81,   182,    80,   200,
     179,    57,   179,   195,   181,   179,   196,   182,   100,    79,
      82,    91,   179,    78,   202,   194,   190,    99,   194,    52,
     201,    78,    91,   203,    81,   179,   203,    81,    99,    83,
     121,   180,   186,   189,   182,   202,    79,    81,    81,    80,
     200,   179,   204,    80,   200,   182,    80,   200,   100,   185,
     100,   179,    83,   100,   203,   202,   201,   194,    78,   179,
     194,    99,   194,   201,    84,    86,    89,    90,    93,    83,
      91,   185,    98,    80,   200,    82,    81,   179,    79,    81,
      81,   185,   100,    58,   185,    84,   185,    84,   194,   202,
     203,   179,   201,    87,    91,    91,   100,    84,    81,   203,
      80,   200,    82,    80,   200,   194,   179,   194,    81,   203,
      81,    80,   200,   194,    81
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    94,    95,    96,    96,    97,    97,    98,    98,    99,
      99,   100,   100,   100,   100,   100,   100,   100,   100,   100,
     100,   100,   100,   100,   100,   100,   100,   100,   100,   100,
     100,   100,   100,   100,   100,   100,   100,   100,   100,   100,
     100,   100,   100,   100,   100,   100,   100,   100,   100,   100,
     100,   100,   100,   100,   100,   100,   100,   100,   100,   100,
     100,   101,   101,   101,   102,   102,   103,   103,   104,   104,
     105,   105,   105,   105,   105,   106,   106,   106,   106,   106,
     106,   106,   106,   106,   106,   106,   106,   106,   106,   107,
     107,   107,   108,   108,   109,   109,   110,   110,   111,   111,
     111,   111,   111,   111,   111,   111,   111,   111,   111,   111,
     111,   111,   111,   112,   113,   113,   114,   114,   115,   116,
     116,   117,   118,   118,   118,   118,   118,   118,   119,   119,
     119,   119,   119,   120,   120,   120,   120,   120,   120,   121,
     121,   121,   121,   121,   121,   122,   123,   124,   124,   125,
     126,   127,   127,   128,   128,   129,   129,   130,   130,   131,
     131,   132,   132,   133,   133,   134,   135,   135,   136,   136,
     137,   137,   138,   138,   139,   139,   140,   141,   141,   142,
     142,   142,   143,   143,   144,   144,   145,   145,   146,   147,
     148,   148,   149,   149,   150,   150,   151,   152,   153,   154,
     154,   155,   155,   156,   156,   156,   156,   157,   157,   157,
     157,   157,   157,   158,   158,   159,   160,   160,   160,   160,
     160,   161,   161,   162,   162,   163,   163,   163,   163,   163,
     163,   163,   164,   164,   164,   164,   164,   165,   165,   165,
     165,   166,   166,   167,   168,   169,   169,   169,   169,   170,
     170,   170,   170,   170,   170,   170,   170,   170,   170,   170,
     171,   171,   171,   172,   172,   173,   174,   174,   174,   175,
     175,   176,   176,   176,   176,   176,   176,   176,   176,   176,
     176,   176,   176,   176,   176,   176,   176,   176,   176,   177,
     177,   177,   178,   178,   178,   179,   179,   179,   179,   179,
     179,   180,   181,   182,   183,   183,   183,   183,   183,   183,
     183,   184,   184,   184,   185,   185,   185,   185,   185,   185,
     186,   187,   187,   187,   188,   188,   189,   189,   190,   190,
     191,   191,   192,   192,   193,   193,   193,   194,   194,   195,
     195,   196,   196,   196,   197,   197,   198,   198,   198,   199,
     199,   199,   199,   199,   199,   199,   199,   199,   199,   199,
     199,   200,   200,   200,   200,   200,   200,   200,   200,   200,
     200,   200,   200,   200,   200,   201,   201,   201,   202,   203,
     204,   205,   205,   206,   206,   207,   208,   209,   210
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
       1,     1,     4,     4,     3,     3,     1,     4,     0,     2,
       3,     2,     2,     2,     8,     5,     5,     2,     2,     2,
       2,     2,     2,     2,     2,     1,     1,     1,     1,     1,
       1,     1,     1,     3,     0,     1,     0,     3,     1,     1,
       1,     1,     2,     2,     3,     3,     2,     2,     2,     1,
       1,     2,     1,     2,     2,     3,     1,     1,     2,     2,
       2,     8,     1,     1,     1,     1,     2,     2,     1,     1,
       1,     2,     2,     6,     5,     4,     3,     2,     1,     6,
       5,     4,     3,     2,     1,     1,     3,     0,     2,     4,
       6,     0,     1,     0,     3,     1,     3,     1,     1,     0,
       3,     1,     3,     0,     1,     1,     0,     3,     1,     3,
       1,     1,     0,     1,     0,     2,     5,     1,     2,     3,
       5,     6,     0,     2,     1,     3,     5,     5,     5,     5,
       4,     3,     6,     6,     5,     5,     5,     5,     5,     4,
       7,     0,     2,     0,     2,     2,     2,     6,     6,     3,
       3,     2,     3,     1,     3,     4,     2,     2,     2,     2,
       2,     1,     4,     0,     2,     1,     1,     1,     1,     2,
       2,     2,     3,     6,     9,     3,     6,     3,     6,     9,
       9,     1,     3,     1,     1,     1,     2,     2,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       7,     5,    13,     5,     2,     1,     0,     3,     1,     1,
       3,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     2,     1,     1,     1,     1,     1,
       1,     1,     0,     1,     3,     0,     1,     5,     5,     5,
       4,     3,     1,     1,     1,     3,     4,     3,     4,     4,
       4,     1,     1,     1,     1,     4,     3,     4,     4,     4,
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
#line 199 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.modlist) = (yyvsp[0].modlist); modlist = (yyvsp[0].modlist); }
#line 2269 "y.tab.c" /* yacc.c:1646  */
    break;

  case 3:
#line 203 "xi-grammar.y" /* yacc.c:1646  */
    { 
		  (yyval.modlist) = 0; 
		}
#line 2277 "y.tab.c" /* yacc.c:1646  */
    break;

  case 4:
#line 207 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.modlist) = new AstChildren<Module>(lineno, (yyvsp[-1].module), (yyvsp[0].modlist)); }
#line 2283 "y.tab.c" /* yacc.c:1646  */
    break;

  case 5:
#line 211 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 2289 "y.tab.c" /* yacc.c:1646  */
    break;

  case 6:
#line 213 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 2295 "y.tab.c" /* yacc.c:1646  */
    break;

  case 7:
#line 217 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 2301 "y.tab.c" /* yacc.c:1646  */
    break;

  case 8:
#line 219 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 2; }
#line 2307 "y.tab.c" /* yacc.c:1646  */
    break;

  case 9:
#line 223 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 2313 "y.tab.c" /* yacc.c:1646  */
    break;

  case 10:
#line 225 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 2319 "y.tab.c" /* yacc.c:1646  */
    break;

  case 11:
#line 230 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 2325 "y.tab.c" /* yacc.c:1646  */
    break;

  case 12:
#line 231 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MODULE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2331 "y.tab.c" /* yacc.c:1646  */
    break;

  case 13:
#line 232 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MAINMODULE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2337 "y.tab.c" /* yacc.c:1646  */
    break;

  case 14:
#line 233 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(EXTERN, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2343 "y.tab.c" /* yacc.c:1646  */
    break;

  case 15:
#line 235 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITCALL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2349 "y.tab.c" /* yacc.c:1646  */
    break;

  case 16:
#line 236 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITNODE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2355 "y.tab.c" /* yacc.c:1646  */
    break;

  case 17:
#line 237 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITPROC, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2361 "y.tab.c" /* yacc.c:1646  */
    break;

  case 18:
#line 239 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CHARE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2367 "y.tab.c" /* yacc.c:1646  */
    break;

  case 19:
#line 240 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MAINCHARE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2373 "y.tab.c" /* yacc.c:1646  */
    break;

  case 20:
#line 241 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(GROUP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2379 "y.tab.c" /* yacc.c:1646  */
    break;

  case 21:
#line 242 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NODEGROUP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2385 "y.tab.c" /* yacc.c:1646  */
    break;

  case 22:
#line 243 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ARRAY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2391 "y.tab.c" /* yacc.c:1646  */
    break;

  case 23:
#line 247 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INCLUDE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2397 "y.tab.c" /* yacc.c:1646  */
    break;

  case 24:
#line 248 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(STACKSIZE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2403 "y.tab.c" /* yacc.c:1646  */
    break;

  case 25:
#line 249 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(THREADED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2409 "y.tab.c" /* yacc.c:1646  */
    break;

  case 26:
#line 250 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(TEMPLATE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2415 "y.tab.c" /* yacc.c:1646  */
    break;

  case 27:
#line 251 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(SYNC, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2421 "y.tab.c" /* yacc.c:1646  */
    break;

  case 28:
#line 252 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IGET, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2427 "y.tab.c" /* yacc.c:1646  */
    break;

  case 29:
#line 253 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(EXCLUSIVE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2433 "y.tab.c" /* yacc.c:1646  */
    break;

  case 30:
#line 254 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IMMEDIATE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2439 "y.tab.c" /* yacc.c:1646  */
    break;

  case 31:
#line 255 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(SKIPSCHED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2445 "y.tab.c" /* yacc.c:1646  */
    break;

  case 32:
#line 256 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOCOPY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2451 "y.tab.c" /* yacc.c:1646  */
    break;

  case 33:
#line 257 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOCOPYPOST, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2457 "y.tab.c" /* yacc.c:1646  */
    break;

  case 34:
#line 258 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOCOPYDEVICE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2463 "y.tab.c" /* yacc.c:1646  */
    break;

  case 35:
#line 259 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INLINE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2469 "y.tab.c" /* yacc.c:1646  */
    break;

  case 36:
#line 260 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(VIRTUAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2475 "y.tab.c" /* yacc.c:1646  */
    break;

  case 37:
#line 261 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MIGRATABLE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2481 "y.tab.c" /* yacc.c:1646  */
    break;

  case 38:
#line 262 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CREATEHERE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2487 "y.tab.c" /* yacc.c:1646  */
    break;

  case 39:
#line 263 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CREATEHOME, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2493 "y.tab.c" /* yacc.c:1646  */
    break;

  case 40:
#line 264 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOKEEP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2499 "y.tab.c" /* yacc.c:1646  */
    break;

  case 41:
#line 265 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOTRACE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2505 "y.tab.c" /* yacc.c:1646  */
    break;

  case 42:
#line 266 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(APPWORK, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2511 "y.tab.c" /* yacc.c:1646  */
    break;

  case 43:
#line 269 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(PACKED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2517 "y.tab.c" /* yacc.c:1646  */
    break;

  case 44:
#line 270 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(VARSIZE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2523 "y.tab.c" /* yacc.c:1646  */
    break;

  case 45:
#line 271 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ENTRY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2529 "y.tab.c" /* yacc.c:1646  */
    break;

  case 46:
#line 272 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(FOR, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2535 "y.tab.c" /* yacc.c:1646  */
    break;

  case 47:
#line 273 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(FORALL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2541 "y.tab.c" /* yacc.c:1646  */
    break;

  case 48:
#line 274 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(WHILE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2547 "y.tab.c" /* yacc.c:1646  */
    break;

  case 49:
#line 275 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(WHEN, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2553 "y.tab.c" /* yacc.c:1646  */
    break;

  case 50:
#line 276 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(OVERLAP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2559 "y.tab.c" /* yacc.c:1646  */
    break;

  case 51:
#line 277 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(SERIAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2565 "y.tab.c" /* yacc.c:1646  */
    break;

  case 52:
#line 278 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IF, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2571 "y.tab.c" /* yacc.c:1646  */
    break;

  case 53:
#line 279 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ELSE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2577 "y.tab.c" /* yacc.c:1646  */
    break;

  case 54:
#line 281 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(LOCAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2583 "y.tab.c" /* yacc.c:1646  */
    break;

  case 55:
#line 283 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(USING, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2589 "y.tab.c" /* yacc.c:1646  */
    break;

  case 56:
#line 284 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ACCEL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2595 "y.tab.c" /* yacc.c:1646  */
    break;

  case 57:
#line 287 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ACCELBLOCK, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2601 "y.tab.c" /* yacc.c:1646  */
    break;

  case 58:
#line 288 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MEMCRITICAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2607 "y.tab.c" /* yacc.c:1646  */
    break;

  case 59:
#line 289 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(REDUCTIONTARGET, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2613 "y.tab.c" /* yacc.c:1646  */
    break;

  case 60:
#line 290 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CASE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2619 "y.tab.c" /* yacc.c:1646  */
    break;

  case 61:
#line 295 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 2625 "y.tab.c" /* yacc.c:1646  */
    break;

  case 62:
#line 297 "xi-grammar.y" /* yacc.c:1646  */
    {
		  char *tmp = new char[strlen((yyvsp[-3].strval))+strlen((yyvsp[0].strval))+3];
		  sprintf(tmp,"%s::%s", (yyvsp[-3].strval), (yyvsp[0].strval));
		  (yyval.strval) = tmp;
		}
#line 2635 "y.tab.c" /* yacc.c:1646  */
    break;

  case 63:
#line 303 "xi-grammar.y" /* yacc.c:1646  */
    {
		  char *tmp = new char[strlen((yyvsp[-3].strval))+5+3];
		  sprintf(tmp,"%s::array", (yyvsp[-3].strval));
		  (yyval.strval) = tmp;
		}
#line 2645 "y.tab.c" /* yacc.c:1646  */
    break;

  case 64:
#line 310 "xi-grammar.y" /* yacc.c:1646  */
    { 
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist)); 
		}
#line 2653 "y.tab.c" /* yacc.c:1646  */
    break;

  case 65:
#line 314 "xi-grammar.y" /* yacc.c:1646  */
    {  
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist)); 
		    (yyval.module)->setMain();
		}
#line 2662 "y.tab.c" /* yacc.c:1646  */
    break;

  case 66:
#line 321 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = 0; }
#line 2668 "y.tab.c" /* yacc.c:1646  */
    break;

  case 67:
#line 323 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = (yyvsp[-2].conslist); }
#line 2674 "y.tab.c" /* yacc.c:1646  */
    break;

  case 68:
#line 327 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = 0; }
#line 2680 "y.tab.c" /* yacc.c:1646  */
    break;

  case 69:
#line 329 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = new ConstructList(lineno, (yyvsp[-1].construct), (yyvsp[0].conslist)); }
#line 2686 "y.tab.c" /* yacc.c:1646  */
    break;

  case 70:
#line 333 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new UsingScope((yyvsp[0].strval), false); }
#line 2692 "y.tab.c" /* yacc.c:1646  */
    break;

  case 71:
#line 335 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new UsingScope((yyvsp[0].strval), true); }
#line 2698 "y.tab.c" /* yacc.c:1646  */
    break;

  case 72:
#line 337 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].member)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].member); }
#line 2704 "y.tab.c" /* yacc.c:1646  */
    break;

  case 73:
#line 339 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].message)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].message); }
#line 2710 "y.tab.c" /* yacc.c:1646  */
    break;

  case 74:
#line 341 "xi-grammar.y" /* yacc.c:1646  */
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
#line 2725 "y.tab.c" /* yacc.c:1646  */
    break;

  case 75:
#line 354 "xi-grammar.y" /* yacc.c:1646  */
    { if((yyvsp[-2].conslist)) (yyvsp[-2].conslist)->recurse<int&>((yyvsp[-4].intval), &Construct::setExtern); (yyval.construct) = (yyvsp[-2].conslist); }
#line 2731 "y.tab.c" /* yacc.c:1646  */
    break;

  case 76:
#line 356 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new Scope((yyvsp[-3].strval), (yyvsp[-1].conslist)); }
#line 2737 "y.tab.c" /* yacc.c:1646  */
    break;

  case 77:
#line 358 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = (yyvsp[-1].construct); }
#line 2743 "y.tab.c" /* yacc.c:1646  */
    break;

  case 78:
#line 360 "xi-grammar.y" /* yacc.c:1646  */
    {
          ERROR("preceding construct must be semicolon terminated",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 2753 "y.tab.c" /* yacc.c:1646  */
    break;

  case 79:
#line 366 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].module)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].module); }
#line 2759 "y.tab.c" /* yacc.c:1646  */
    break;

  case 80:
#line 368 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2765 "y.tab.c" /* yacc.c:1646  */
    break;

  case 81:
#line 370 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2771 "y.tab.c" /* yacc.c:1646  */
    break;

  case 82:
#line 372 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2777 "y.tab.c" /* yacc.c:1646  */
    break;

  case 83:
#line 374 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2783 "y.tab.c" /* yacc.c:1646  */
    break;

  case 84:
#line 376 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].templat)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].templat); }
#line 2789 "y.tab.c" /* yacc.c:1646  */
    break;

  case 85:
#line 378 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = NULL; }
#line 2795 "y.tab.c" /* yacc.c:1646  */
    break;

  case 86:
#line 380 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = NULL; }
#line 2801 "y.tab.c" /* yacc.c:1646  */
    break;

  case 87:
#line 382 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = (yyvsp[0].accelBlock); }
#line 2807 "y.tab.c" /* yacc.c:1646  */
    break;

  case 88:
#line 384 "xi-grammar.y" /* yacc.c:1646  */
    {
          ERROR("invalid construct",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 2817 "y.tab.c" /* yacc.c:1646  */
    break;

  case 89:
#line 392 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamType((yyvsp[0].type)); }
#line 2823 "y.tab.c" /* yacc.c:1646  */
    break;

  case 90:
#line 394 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
#line 2829 "y.tab.c" /* yacc.c:1646  */
    break;

  case 91:
#line 396 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
#line 2835 "y.tab.c" /* yacc.c:1646  */
    break;

  case 92:
#line 400 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = new TParamList((yyvsp[0].tparam)); }
#line 2841 "y.tab.c" /* yacc.c:1646  */
    break;

  case 93:
#line 402 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = new TParamList((yyvsp[-2].tparam), (yyvsp[0].tparlist)); }
#line 2847 "y.tab.c" /* yacc.c:1646  */
    break;

  case 94:
#line 406 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = new TParamList(0); }
#line 2853 "y.tab.c" /* yacc.c:1646  */
    break;

  case 95:
#line 408 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = (yyvsp[0].tparlist); }
#line 2859 "y.tab.c" /* yacc.c:1646  */
    break;

  case 96:
#line 412 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = 0; }
#line 2865 "y.tab.c" /* yacc.c:1646  */
    break;

  case 97:
#line 414 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = (yyvsp[-1].tparlist); }
#line 2871 "y.tab.c" /* yacc.c:1646  */
    break;

  case 98:
#line 418 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("int"); }
#line 2877 "y.tab.c" /* yacc.c:1646  */
    break;

  case 99:
#line 420 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long"); }
#line 2883 "y.tab.c" /* yacc.c:1646  */
    break;

  case 100:
#line 422 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("short"); }
#line 2889 "y.tab.c" /* yacc.c:1646  */
    break;

  case 101:
#line 424 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("char"); }
#line 2895 "y.tab.c" /* yacc.c:1646  */
    break;

  case 102:
#line 426 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned int"); }
#line 2901 "y.tab.c" /* yacc.c:1646  */
    break;

  case 103:
#line 428 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long"); }
#line 2907 "y.tab.c" /* yacc.c:1646  */
    break;

  case 104:
#line 430 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long"); }
#line 2913 "y.tab.c" /* yacc.c:1646  */
    break;

  case 105:
#line 432 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long long"); }
#line 2919 "y.tab.c" /* yacc.c:1646  */
    break;

  case 106:
#line 434 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned short"); }
#line 2925 "y.tab.c" /* yacc.c:1646  */
    break;

  case 107:
#line 436 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned char"); }
#line 2931 "y.tab.c" /* yacc.c:1646  */
    break;

  case 108:
#line 438 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long long"); }
#line 2937 "y.tab.c" /* yacc.c:1646  */
    break;

  case 109:
#line 440 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("float"); }
#line 2943 "y.tab.c" /* yacc.c:1646  */
    break;

  case 110:
#line 442 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("double"); }
#line 2949 "y.tab.c" /* yacc.c:1646  */
    break;

  case 111:
#line 444 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long double"); }
#line 2955 "y.tab.c" /* yacc.c:1646  */
    break;

  case 112:
#line 446 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("void"); }
#line 2961 "y.tab.c" /* yacc.c:1646  */
    break;

  case 113:
#line 449 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ntype) = new NamedType((yyvsp[-1].strval),(yyvsp[0].tparlist)); }
#line 2967 "y.tab.c" /* yacc.c:1646  */
    break;

  case 114:
#line 450 "xi-grammar.y" /* yacc.c:1646  */
    { 
                    const char* basename, *scope;
                    splitScopedName((yyvsp[-1].strval), &scope, &basename);
                    (yyval.ntype) = new NamedType(basename, (yyvsp[0].tparlist), scope);
                }
#line 2977 "y.tab.c" /* yacc.c:1646  */
    break;

  case 115:
#line 456 "xi-grammar.y" /* yacc.c:1646  */
    {
			const char* basename, *scope;
			splitScopedName((yyvsp[-1].strval), &scope, &basename);
			(yyval.ntype) = new NamedType(basename, (yyvsp[0].tparlist), scope, true);
		}
#line 2987 "y.tab.c" /* yacc.c:1646  */
    break;

  case 116:
#line 464 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 2993 "y.tab.c" /* yacc.c:1646  */
    break;

  case 117:
#line 466 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ntype); }
#line 2999 "y.tab.c" /* yacc.c:1646  */
    break;

  case 118:
#line 470 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ptype) = new PtrType((yyvsp[-1].type)); }
#line 3005 "y.tab.c" /* yacc.c:1646  */
    break;

  case 119:
#line 474 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
#line 3011 "y.tab.c" /* yacc.c:1646  */
    break;

  case 120:
#line 476 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
#line 3017 "y.tab.c" /* yacc.c:1646  */
    break;

  case 121:
#line 480 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ftype) = new FuncType((yyvsp[-7].type), (yyvsp[-4].strval), (yyvsp[-1].plist)); }
#line 3023 "y.tab.c" /* yacc.c:1646  */
    break;

  case 122:
#line 484 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3029 "y.tab.c" /* yacc.c:1646  */
    break;

  case 123:
#line 486 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3035 "y.tab.c" /* yacc.c:1646  */
    break;

  case 124:
#line 488 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3041 "y.tab.c" /* yacc.c:1646  */
    break;

  case 125:
#line 490 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ftype); }
#line 3047 "y.tab.c" /* yacc.c:1646  */
    break;

  case 126:
#line 492 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[0].type)); }
#line 3053 "y.tab.c" /* yacc.c:1646  */
    break;

  case 127:
#line 494 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[-1].type)); }
#line 3059 "y.tab.c" /* yacc.c:1646  */
    break;

  case 128:
#line 498 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3065 "y.tab.c" /* yacc.c:1646  */
    break;

  case 129:
#line 500 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3071 "y.tab.c" /* yacc.c:1646  */
    break;

  case 130:
#line 502 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3077 "y.tab.c" /* yacc.c:1646  */
    break;

  case 131:
#line 504 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[0].type)); }
#line 3083 "y.tab.c" /* yacc.c:1646  */
    break;

  case 132:
#line 506 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[-1].type)); }
#line 3089 "y.tab.c" /* yacc.c:1646  */
    break;

  case 133:
#line 510 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType(new RValueReferenceType((yyvsp[-5].type))); }
#line 3095 "y.tab.c" /* yacc.c:1646  */
    break;

  case 134:
#line 512 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType(new ReferenceType((yyvsp[-4].type))); }
#line 3101 "y.tab.c" /* yacc.c:1646  */
    break;

  case 135:
#line 514 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType((yyvsp[-3].type)); }
#line 3107 "y.tab.c" /* yacc.c:1646  */
    break;

  case 136:
#line 516 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new RValueReferenceType((yyvsp[-2].type)); }
#line 3113 "y.tab.c" /* yacc.c:1646  */
    break;

  case 137:
#line 518 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ReferenceType((yyvsp[-1].type)); }
#line 3119 "y.tab.c" /* yacc.c:1646  */
    break;

  case 138:
#line 520 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3125 "y.tab.c" /* yacc.c:1646  */
    break;

  case 139:
#line 524 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType(new RValueReferenceType((yyvsp[-5].type))); }
#line 3131 "y.tab.c" /* yacc.c:1646  */
    break;

  case 140:
#line 526 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType(new ReferenceType((yyvsp[-4].type))); }
#line 3137 "y.tab.c" /* yacc.c:1646  */
    break;

  case 141:
#line 528 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType((yyvsp[-3].type)); }
#line 3143 "y.tab.c" /* yacc.c:1646  */
    break;

  case 142:
#line 530 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new RValueReferenceType((yyvsp[-2].type)); }
#line 3149 "y.tab.c" /* yacc.c:1646  */
    break;

  case 143:
#line 532 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ReferenceType((yyvsp[-1].type)); }
#line 3155 "y.tab.c" /* yacc.c:1646  */
    break;

  case 144:
#line 534 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3161 "y.tab.c" /* yacc.c:1646  */
    break;

  case 145:
#line 538 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 3167 "y.tab.c" /* yacc.c:1646  */
    break;

  case 146:
#line 542 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = (yyvsp[-1].val); }
#line 3173 "y.tab.c" /* yacc.c:1646  */
    break;

  case 147:
#line 546 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.vallist) = 0; }
#line 3179 "y.tab.c" /* yacc.c:1646  */
    break;

  case 148:
#line 548 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.vallist) = new ValueList((yyvsp[-1].val), (yyvsp[0].vallist)); }
#line 3185 "y.tab.c" /* yacc.c:1646  */
    break;

  case 149:
#line 552 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].vallist)); }
#line 3191 "y.tab.c" /* yacc.c:1646  */
    break;

  case 150:
#line 556 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[-3].type), (yyvsp[-1].strval), (yyvsp[0].vallist), 1); }
#line 3197 "y.tab.c" /* yacc.c:1646  */
    break;

  case 151:
#line 560 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0;}
#line 3203 "y.tab.c" /* yacc.c:1646  */
    break;

  case 152:
#line 562 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0;}
#line 3209 "y.tab.c" /* yacc.c:1646  */
    break;

  case 153:
#line 566 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3215 "y.tab.c" /* yacc.c:1646  */
    break;

  case 154:
#line 568 "xi-grammar.y" /* yacc.c:1646  */
    { 
		  /*
		  printf("Warning: Message attributes are being phased out.\n");
		  printf("Warning: Please remove them from interface files.\n");
		  */
		  (yyval.intval) = (yyvsp[-1].intval); 
		}
#line 3227 "y.tab.c" /* yacc.c:1646  */
    break;

  case 155:
#line 578 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[0].intval); }
#line 3233 "y.tab.c" /* yacc.c:1646  */
    break;

  case 156:
#line 580 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[-2].intval) | (yyvsp[0].intval); }
#line 3239 "y.tab.c" /* yacc.c:1646  */
    break;

  case 157:
#line 584 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3245 "y.tab.c" /* yacc.c:1646  */
    break;

  case 158:
#line 586 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3251 "y.tab.c" /* yacc.c:1646  */
    break;

  case 159:
#line 590 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = 0; }
#line 3257 "y.tab.c" /* yacc.c:1646  */
    break;

  case 160:
#line 592 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-1].cattr); }
#line 3263 "y.tab.c" /* yacc.c:1646  */
    break;

  case 161:
#line 596 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[0].cattr); }
#line 3269 "y.tab.c" /* yacc.c:1646  */
    break;

  case 162:
#line 598 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
#line 3275 "y.tab.c" /* yacc.c:1646  */
    break;

  case 163:
#line 602 "xi-grammar.y" /* yacc.c:1646  */
    { python_doc = NULL; (yyval.intval) = 0; }
#line 3281 "y.tab.c" /* yacc.c:1646  */
    break;

  case 164:
#line 604 "xi-grammar.y" /* yacc.c:1646  */
    { python_doc = (yyvsp[0].strval); (yyval.intval) = 0; }
#line 3287 "y.tab.c" /* yacc.c:1646  */
    break;

  case 165:
#line 608 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CPYTHON; }
#line 3293 "y.tab.c" /* yacc.c:1646  */
    break;

  case 166:
#line 612 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = 0; }
#line 3299 "y.tab.c" /* yacc.c:1646  */
    break;

  case 167:
#line 614 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-1].cattr); }
#line 3305 "y.tab.c" /* yacc.c:1646  */
    break;

  case 168:
#line 618 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[0].cattr); }
#line 3311 "y.tab.c" /* yacc.c:1646  */
    break;

  case 169:
#line 620 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
#line 3317 "y.tab.c" /* yacc.c:1646  */
    break;

  case 170:
#line 624 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CMIGRATABLE; }
#line 3323 "y.tab.c" /* yacc.c:1646  */
    break;

  case 171:
#line 626 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CPYTHON; }
#line 3329 "y.tab.c" /* yacc.c:1646  */
    break;

  case 172:
#line 630 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3335 "y.tab.c" /* yacc.c:1646  */
    break;

  case 173:
#line 632 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 3341 "y.tab.c" /* yacc.c:1646  */
    break;

  case 174:
#line 635 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3347 "y.tab.c" /* yacc.c:1646  */
    break;

  case 175:
#line 637 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 3353 "y.tab.c" /* yacc.c:1646  */
    break;

  case 176:
#line 640 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mv) = new MsgVar((yyvsp[-3].type), (yyvsp[-2].strval), (yyvsp[-4].intval), (yyvsp[-1].intval)); }
#line 3359 "y.tab.c" /* yacc.c:1646  */
    break;

  case 177:
#line 644 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mvlist) = new MsgVarList((yyvsp[0].mv)); }
#line 3365 "y.tab.c" /* yacc.c:1646  */
    break;

  case 178:
#line 646 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mvlist) = new MsgVarList((yyvsp[-1].mv), (yyvsp[0].mvlist)); }
#line 3371 "y.tab.c" /* yacc.c:1646  */
    break;

  case 179:
#line 650 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, (yyvsp[0].ntype)); }
#line 3377 "y.tab.c" /* yacc.c:1646  */
    break;

  case 180:
#line 652 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, (yyvsp[-2].ntype)); }
#line 3383 "y.tab.c" /* yacc.c:1646  */
    break;

  case 181:
#line 654 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, (yyvsp[-3].ntype), (yyvsp[-1].mvlist)); }
#line 3389 "y.tab.c" /* yacc.c:1646  */
    break;

  case 182:
#line 658 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = 0; }
#line 3395 "y.tab.c" /* yacc.c:1646  */
    break;

  case 183:
#line 660 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = (yyvsp[0].typelist); }
#line 3401 "y.tab.c" /* yacc.c:1646  */
    break;

  case 184:
#line 664 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = new TypeList((yyvsp[0].ntype)); }
#line 3407 "y.tab.c" /* yacc.c:1646  */
    break;

  case 185:
#line 666 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = new TypeList((yyvsp[-2].ntype), (yyvsp[0].typelist)); }
#line 3413 "y.tab.c" /* yacc.c:1646  */
    break;

  case 186:
#line 670 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr)|Chare::CCHARE, (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3419 "y.tab.c" /* yacc.c:1646  */
    break;

  case 187:
#line 672 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3425 "y.tab.c" /* yacc.c:1646  */
    break;

  case 188:
#line 676 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3431 "y.tab.c" /* yacc.c:1646  */
    break;

  case 189:
#line 680 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new NodeGroup(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3437 "y.tab.c" /* yacc.c:1646  */
    break;

  case 190:
#line 684 "xi-grammar.y" /* yacc.c:1646  */
    {/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",(yyvsp[-2].strval));
			(yyval.ntype) = new NamedType(buf); 
		}
#line 3447 "y.tab.c" /* yacc.c:1646  */
    break;

  case 191:
#line 690 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ntype) = (yyvsp[-1].ntype); }
#line 3453 "y.tab.c" /* yacc.c:1646  */
    break;

  case 192:
#line 694 "xi-grammar.y" /* yacc.c:1646  */
    {  (yyval.chare) = new Array(lineno, (yyvsp[-4].cattr), (yyvsp[-3].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3459 "y.tab.c" /* yacc.c:1646  */
    break;

  case 193:
#line 696 "xi-grammar.y" /* yacc.c:1646  */
    {  (yyval.chare) = new Array(lineno, (yyvsp[-3].cattr), (yyvsp[-4].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3465 "y.tab.c" /* yacc.c:1646  */
    break;

  case 194:
#line 700 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr)|Chare::CCHARE, new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist));}
#line 3471 "y.tab.c" /* yacc.c:1646  */
    break;

  case 195:
#line 702 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3477 "y.tab.c" /* yacc.c:1646  */
    break;

  case 196:
#line 706 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3483 "y.tab.c" /* yacc.c:1646  */
    break;

  case 197:
#line 710 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new NodeGroup( lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3489 "y.tab.c" /* yacc.c:1646  */
    break;

  case 198:
#line 714 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Array( lineno, 0, (yyvsp[-3].ntype), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3495 "y.tab.c" /* yacc.c:1646  */
    break;

  case 199:
#line 718 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-1].strval))); }
#line 3501 "y.tab.c" /* yacc.c:1646  */
    break;

  case 200:
#line 720 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-4].strval)), (yyvsp[-2].mvlist)); }
#line 3507 "y.tab.c" /* yacc.c:1646  */
    break;

  case 201:
#line 724 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = 0; }
#line 3513 "y.tab.c" /* yacc.c:1646  */
    break;

  case 202:
#line 726 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3519 "y.tab.c" /* yacc.c:1646  */
    break;

  case 203:
#line 730 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = 0; }
#line 3525 "y.tab.c" /* yacc.c:1646  */
    break;

  case 204:
#line 732 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3531 "y.tab.c" /* yacc.c:1646  */
    break;

  case 205:
#line 734 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3537 "y.tab.c" /* yacc.c:1646  */
    break;

  case 206:
#line 736 "xi-grammar.y" /* yacc.c:1646  */
    {
		  XStr typeStr;
		  (yyvsp[0].ntype)->print(typeStr);
		  char *tmp = strdup(typeStr.get_string());
		  (yyval.strval) = tmp;
		}
#line 3548 "y.tab.c" /* yacc.c:1646  */
    break;

  case 207:
#line 745 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TTypeEllipsis(new NamedEllipsisType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3554 "y.tab.c" /* yacc.c:1646  */
    break;

  case 208:
#line 747 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TTypeEllipsis(new NamedEllipsisType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3560 "y.tab.c" /* yacc.c:1646  */
    break;

  case 209:
#line 749 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TType(new NamedType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3566 "y.tab.c" /* yacc.c:1646  */
    break;

  case 210:
#line 751 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TType(new NamedType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3572 "y.tab.c" /* yacc.c:1646  */
    break;

  case 211:
#line 753 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TFunc((yyvsp[-1].ftype), (yyvsp[0].strval)); }
#line 3578 "y.tab.c" /* yacc.c:1646  */
    break;

  case 212:
#line 755 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TName((yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].strval)); }
#line 3584 "y.tab.c" /* yacc.c:1646  */
    break;

  case 213:
#line 759 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = new TVarList((yyvsp[0].tvar)); }
#line 3590 "y.tab.c" /* yacc.c:1646  */
    break;

  case 214:
#line 761 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = new TVarList((yyvsp[-2].tvar), (yyvsp[0].tvarlist)); }
#line 3596 "y.tab.c" /* yacc.c:1646  */
    break;

  case 215:
#line 765 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = (yyvsp[-1].tvarlist); }
#line 3602 "y.tab.c" /* yacc.c:1646  */
    break;

  case 216:
#line 769 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3608 "y.tab.c" /* yacc.c:1646  */
    break;

  case 217:
#line 771 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3614 "y.tab.c" /* yacc.c:1646  */
    break;

  case 218:
#line 773 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3620 "y.tab.c" /* yacc.c:1646  */
    break;

  case 219:
#line 775 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3626 "y.tab.c" /* yacc.c:1646  */
    break;

  case 220:
#line 777 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].message)); (yyvsp[0].message)->setTemplate((yyval.templat)); }
#line 3632 "y.tab.c" /* yacc.c:1646  */
    break;

  case 221:
#line 781 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = 0; }
#line 3638 "y.tab.c" /* yacc.c:1646  */
    break;

  case 222:
#line 783 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = (yyvsp[-2].mbrlist); }
#line 3644 "y.tab.c" /* yacc.c:1646  */
    break;

  case 223:
#line 787 "xi-grammar.y" /* yacc.c:1646  */
    { 
                  if (!connectEntries.empty()) {
                    (yyval.mbrlist) = new AstChildren<Member>(connectEntries);
		  } else {
		    (yyval.mbrlist) = 0; 
                  }
		}
#line 3656 "y.tab.c" /* yacc.c:1646  */
    break;

  case 224:
#line 795 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = new AstChildren<Member>(-1, (yyvsp[-1].member), (yyvsp[0].mbrlist)); }
#line 3662 "y.tab.c" /* yacc.c:1646  */
    break;

  case 225:
#line 799 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].readonly); }
#line 3668 "y.tab.c" /* yacc.c:1646  */
    break;

  case 226:
#line 801 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].readonly); }
#line 3674 "y.tab.c" /* yacc.c:1646  */
    break;

  case 228:
#line 804 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].member); }
#line 3680 "y.tab.c" /* yacc.c:1646  */
    break;

  case 229:
#line 806 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].pupable); }
#line 3686 "y.tab.c" /* yacc.c:1646  */
    break;

  case 230:
#line 808 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].includeFile); }
#line 3692 "y.tab.c" /* yacc.c:1646  */
    break;

  case 231:
#line 810 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new ClassDeclaration(lineno,(yyvsp[0].strval)); }
#line 3698 "y.tab.c" /* yacc.c:1646  */
    break;

  case 232:
#line 814 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1); }
#line 3704 "y.tab.c" /* yacc.c:1646  */
    break;

  case 233:
#line 816 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1); }
#line 3710 "y.tab.c" /* yacc.c:1646  */
    break;

  case 234:
#line 818 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[-6].strval)) + '<' +
					    ((yyvsp[-4].tparlist))->to_string() + '>').c_str()),
				    1);
		}
#line 3720 "y.tab.c" /* yacc.c:1646  */
    break;

  case 235:
#line 824 "xi-grammar.y" /* yacc.c:1646  */
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[-2]).first_column, (yylsp[-2]).last_column, (yylsp[-2]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1);
		}
#line 3730 "y.tab.c" /* yacc.c:1646  */
    break;

  case 236:
#line 830 "xi-grammar.y" /* yacc.c:1646  */
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[-5]).first_column, (yylsp[-5]).last_column, (yylsp[-5]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1);
		}
#line 3740 "y.tab.c" /* yacc.c:1646  */
    break;

  case 237:
#line 839 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 0); }
#line 3746 "y.tab.c" /* yacc.c:1646  */
    break;

  case 238:
#line 841 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 0); }
#line 3752 "y.tab.c" /* yacc.c:1646  */
    break;

  case 239:
#line 843 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[-6].strval)) + '<' +
					    ((yyvsp[-4].tparlist))->to_string() + '>').c_str()),
				    0);
		}
#line 3762 "y.tab.c" /* yacc.c:1646  */
    break;

  case 240:
#line 849 "xi-grammar.y" /* yacc.c:1646  */
    {
                  InitCall* rtn = new InitCall(lineno, (yyvsp[-3].strval), 0);
                  rtn->setAccel();
                  (yyval.member) = rtn;
		}
#line 3772 "y.tab.c" /* yacc.c:1646  */
    break;

  case 241:
#line 857 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[0].ntype),0); }
#line 3778 "y.tab.c" /* yacc.c:1646  */
    break;

  case 242:
#line 859 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[-2].ntype),(yyvsp[0].pupable)); }
#line 3784 "y.tab.c" /* yacc.c:1646  */
    break;

  case 243:
#line 862 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.includeFile) = new IncludeFile(lineno,(yyvsp[0].strval)); }
#line 3790 "y.tab.c" /* yacc.c:1646  */
    break;

  case 244:
#line 866 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].member); }
#line 3796 "y.tab.c" /* yacc.c:1646  */
    break;

  case 245:
#line 870 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].entry); }
#line 3802 "y.tab.c" /* yacc.c:1646  */
    break;

  case 246:
#line 872 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyvsp[0].entry)->tspec = (yyvsp[-1].tvarlist);
                  (yyval.member) = (yyvsp[0].entry);
                }
#line 3811 "y.tab.c" /* yacc.c:1646  */
    break;

  case 247:
#line 877 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[-1].member); }
#line 3817 "y.tab.c" /* yacc.c:1646  */
    break;

  case 248:
#line 879 "xi-grammar.y" /* yacc.c:1646  */
    {
          ERROR("invalid SDAG member",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 3827 "y.tab.c" /* yacc.c:1646  */
    break;

  case 249:
#line 887 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3833 "y.tab.c" /* yacc.c:1646  */
    break;

  case 250:
#line 889 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3839 "y.tab.c" /* yacc.c:1646  */
    break;

  case 251:
#line 891 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3845 "y.tab.c" /* yacc.c:1646  */
    break;

  case 252:
#line 893 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3851 "y.tab.c" /* yacc.c:1646  */
    break;

  case 253:
#line 895 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3857 "y.tab.c" /* yacc.c:1646  */
    break;

  case 254:
#line 897 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3863 "y.tab.c" /* yacc.c:1646  */
    break;

  case 255:
#line 899 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3869 "y.tab.c" /* yacc.c:1646  */
    break;

  case 256:
#line 901 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3875 "y.tab.c" /* yacc.c:1646  */
    break;

  case 257:
#line 903 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3881 "y.tab.c" /* yacc.c:1646  */
    break;

  case 258:
#line 905 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3887 "y.tab.c" /* yacc.c:1646  */
    break;

  case 259:
#line 907 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3893 "y.tab.c" /* yacc.c:1646  */
    break;

  case 260:
#line 910 "xi-grammar.y" /* yacc.c:1646  */
    { 
                  (yyval.entry) = new Entry(lineno, (yyvsp[-5].intval), (yyvsp[-4].type), (yyvsp[-3].strval), (yyvsp[-2].plist), (yyvsp[-1].val), (yyvsp[0].sentry), (const char *) NULL, (yylsp[-6]).first_line, (yyloc).last_line);
		  if ((yyvsp[0].sentry) != 0) { 
		    (yyvsp[0].sentry)->con1 = new SdagConstruct(SIDENT, (yyvsp[-3].strval));
                    (yyvsp[0].sentry)->setEntry((yyval.entry));
                    (yyvsp[0].sentry)->param = new ParamList((yyvsp[-2].plist));
                  }
                  firstRdma = true;
		}
#line 3907 "y.tab.c" /* yacc.c:1646  */
    break;

  case 261:
#line 920 "xi-grammar.y" /* yacc.c:1646  */
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
#line 3928 "y.tab.c" /* yacc.c:1646  */
    break;

  case 262:
#line 937 "xi-grammar.y" /* yacc.c:1646  */
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
#line 3947 "y.tab.c" /* yacc.c:1646  */
    break;

  case 263:
#line 954 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.accelBlock) = new AccelBlock(lineno, new XStr((yyvsp[-2].strval))); }
#line 3953 "y.tab.c" /* yacc.c:1646  */
    break;

  case 264:
#line 956 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.accelBlock) = new AccelBlock(lineno, NULL); }
#line 3959 "y.tab.c" /* yacc.c:1646  */
    break;

  case 265:
#line 960 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3965 "y.tab.c" /* yacc.c:1646  */
    break;

  case 266:
#line 964 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3971 "y.tab.c" /* yacc.c:1646  */
    break;

  case 267:
#line 966 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[-1].intval); }
#line 3977 "y.tab.c" /* yacc.c:1646  */
    break;

  case 268:
#line 968 "xi-grammar.y" /* yacc.c:1646  */
    { ERROR("invalid entry method attribute list",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
#line 3986 "y.tab.c" /* yacc.c:1646  */
    break;

  case 269:
#line 975 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[0].intval); }
#line 3992 "y.tab.c" /* yacc.c:1646  */
    break;

  case 270:
#line 977 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[-2].intval) | (yyvsp[0].intval); }
#line 3998 "y.tab.c" /* yacc.c:1646  */
    break;

  case 271:
#line 981 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = STHREADED; }
#line 4004 "y.tab.c" /* yacc.c:1646  */
    break;

  case 272:
#line 983 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SSYNC; }
#line 4010 "y.tab.c" /* yacc.c:1646  */
    break;

  case 273:
#line 985 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SIGET; }
#line 4016 "y.tab.c" /* yacc.c:1646  */
    break;

  case 274:
#line 987 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SLOCKED; }
#line 4022 "y.tab.c" /* yacc.c:1646  */
    break;

  case 275:
#line 989 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SCREATEHERE; }
#line 4028 "y.tab.c" /* yacc.c:1646  */
    break;

  case 276:
#line 991 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SCREATEHOME; }
#line 4034 "y.tab.c" /* yacc.c:1646  */
    break;

  case 277:
#line 993 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SNOKEEP; }
#line 4040 "y.tab.c" /* yacc.c:1646  */
    break;

  case 278:
#line 995 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SNOTRACE; }
#line 4046 "y.tab.c" /* yacc.c:1646  */
    break;

  case 279:
#line 997 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SAPPWORK; }
#line 4052 "y.tab.c" /* yacc.c:1646  */
    break;

  case 280:
#line 999 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SIMMEDIATE; }
#line 4058 "y.tab.c" /* yacc.c:1646  */
    break;

  case 281:
#line 1001 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SSKIPSCHED; }
#line 4064 "y.tab.c" /* yacc.c:1646  */
    break;

  case 282:
#line 1003 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SINLINE; }
#line 4070 "y.tab.c" /* yacc.c:1646  */
    break;

  case 283:
#line 1005 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SLOCAL; }
#line 4076 "y.tab.c" /* yacc.c:1646  */
    break;

  case 284:
#line 1007 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SPYTHON; }
#line 4082 "y.tab.c" /* yacc.c:1646  */
    break;

  case 285:
#line 1009 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SMEM; }
#line 4088 "y.tab.c" /* yacc.c:1646  */
    break;

  case 286:
#line 1011 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SREDUCE; }
#line 4094 "y.tab.c" /* yacc.c:1646  */
    break;

  case 287:
#line 1013 "xi-grammar.y" /* yacc.c:1646  */
    {
        (yyval.intval) = SAGGREGATE;
    }
#line 4102 "y.tab.c" /* yacc.c:1646  */
    break;

  case 288:
#line 1017 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("invalid entry method attribute",
		        (yylsp[0]).first_column, (yylsp[0]).last_column);
		  yyclearin;
		  yyerrok;
		}
#line 4113 "y.tab.c" /* yacc.c:1646  */
    break;

  case 289:
#line 1026 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4119 "y.tab.c" /* yacc.c:1646  */
    break;

  case 290:
#line 1028 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4125 "y.tab.c" /* yacc.c:1646  */
    break;

  case 291:
#line 1030 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4131 "y.tab.c" /* yacc.c:1646  */
    break;

  case 292:
#line 1034 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = ""; }
#line 4137 "y.tab.c" /* yacc.c:1646  */
    break;

  case 293:
#line 1036 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4143 "y.tab.c" /* yacc.c:1646  */
    break;

  case 294:
#line 1038 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s, %s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4153 "y.tab.c" /* yacc.c:1646  */
    break;

  case 295:
#line 1046 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = ""; }
#line 4159 "y.tab.c" /* yacc.c:1646  */
    break;

  case 296:
#line 1048 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4165 "y.tab.c" /* yacc.c:1646  */
    break;

  case 297:
#line 1050 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s[%s]%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4175 "y.tab.c" /* yacc.c:1646  */
    break;

  case 298:
#line 1056 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s{%s}%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4185 "y.tab.c" /* yacc.c:1646  */
    break;

  case 299:
#line 1062 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s(%s)%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4195 "y.tab.c" /* yacc.c:1646  */
    break;

  case 300:
#line 1068 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"(%s)%s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4205 "y.tab.c" /* yacc.c:1646  */
    break;

  case 301:
#line 1076 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval));
		}
#line 4214 "y.tab.c" /* yacc.c:1646  */
    break;

  case 302:
#line 1083 "xi-grammar.y" /* yacc.c:1646  */
    { 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			(yyval.intval) = 0;
		}
#line 4224 "y.tab.c" /* yacc.c:1646  */
    break;

  case 303:
#line 1091 "xi-grammar.y" /* yacc.c:1646  */
    { 
			in_braces=0;
			(yyval.intval) = 0;
		}
#line 4233 "y.tab.c" /* yacc.c:1646  */
    break;

  case 304:
#line 1098 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[0].type));}
#line 4239 "y.tab.c" /* yacc.c:1646  */
    break;

  case 305:
#line 1100 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval)); (yyval.pname)->setConditional((yyvsp[0].intval)); }
#line 4245 "y.tab.c" /* yacc.c:1646  */
    break;

  case 306:
#line 1102 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[-3].type),(yyvsp[-2].strval),0,(yyvsp[0].val));}
#line 4251 "y.tab.c" /* yacc.c:1646  */
    break;

  case 307:
#line 1104 "xi-grammar.y" /* yacc.c:1646  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
		}
#line 4260 "y.tab.c" /* yacc.c:1646  */
    break;

  case 308:
#line 1109 "xi-grammar.y" /* yacc.c:1646  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
			(yyval.pname)->setRdma(CMK_ZC_P2P_SEND_MSG);
			(yyval.pname)->setDevice(false);
			if(firstRdma) {
				(yyval.pname)->setFirstRdma(true);
				firstRdma = false;
			}
		}
#line 4275 "y.tab.c" /* yacc.c:1646  */
    break;

  case 309:
#line 1120 "xi-grammar.y" /* yacc.c:1646  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
			(yyval.pname)->setRdma(CMK_ZC_P2P_RECV_MSG);
			(yyval.pname)->setDevice(false);
			if(firstRdma) {
				(yyval.pname)->setFirstRdma(true);
				firstRdma = false;
			}
		}
#line 4290 "y.tab.c" /* yacc.c:1646  */
    break;

  case 310:
#line 1131 "xi-grammar.y" /* yacc.c:1646  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
			(yyval.pname)->setRdma(CMK_ZC_P2P_RECV_MSG);
			(yyval.pname)->setDevice(true);
			if(firstRdma) {
				(yyval.pname)->setFirstRdma(true);
				firstRdma = false;
			}
		}
#line 4305 "y.tab.c" /* yacc.c:1646  */
    break;

  case 311:
#line 1143 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READONLY; }
#line 4311 "y.tab.c" /* yacc.c:1646  */
    break;

  case 312:
#line 1144 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READWRITE; }
#line 4317 "y.tab.c" /* yacc.c:1646  */
    break;

  case 313:
#line 1145 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_WRITEONLY; }
#line 4323 "y.tab.c" /* yacc.c:1646  */
    break;

  case 314:
#line 1148 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr((yyvsp[0].strval)); }
#line 4329 "y.tab.c" /* yacc.c:1646  */
    break;

  case 315:
#line 1149 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "->" << (yyvsp[0].strval); }
#line 4335 "y.tab.c" /* yacc.c:1646  */
    break;

  case 316:
#line 1150 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[-2].xstrptr)) << "." << (yyvsp[0].strval); }
#line 4341 "y.tab.c" /* yacc.c:1646  */
    break;

  case 317:
#line 1152 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << *((yyvsp[-1].xstrptr)) << "]";
                  delete (yyvsp[-3].xstrptr);
                  delete (yyvsp[-1].xstrptr);
                }
#line 4352 "y.tab.c" /* yacc.c:1646  */
    break;

  case 318:
#line 1159 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << (yyvsp[-1].strval) << "]";
                  delete (yyvsp[-3].xstrptr);
                }
#line 4362 "y.tab.c" /* yacc.c:1646  */
    break;

  case 319:
#line 1165 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "(" << *((yyvsp[-1].xstrptr)) << ")";
                  delete (yyvsp[-3].xstrptr);
                  delete (yyvsp[-1].xstrptr);
                }
#line 4373 "y.tab.c" /* yacc.c:1646  */
    break;

  case 320:
#line 1174 "xi-grammar.y" /* yacc.c:1646  */
    {
                  in_bracket = 0;
                  (yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName(), (yyvsp[-1].strval));
                }
#line 4382 "y.tab.c" /* yacc.c:1646  */
    break;

  case 321:
#line 1181 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[-6].intval));
                }
#line 4392 "y.tab.c" /* yacc.c:1646  */
    break;

  case 322:
#line 1187 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType(Parameter::ACCEL_BUFFER_TYPE_READWRITE);
		}
#line 4402 "y.tab.c" /* yacc.c:1646  */
    break;

  case 323:
#line 1193 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.pname) = (yyvsp[-3].pname);
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[-5].intval));
		}
#line 4412 "y.tab.c" /* yacc.c:1646  */
    break;

  case 324:
#line 1201 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 4418 "y.tab.c" /* yacc.c:1646  */
    break;

  case 325:
#line 1203 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 4424 "y.tab.c" /* yacc.c:1646  */
    break;

  case 326:
#line 1207 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 4430 "y.tab.c" /* yacc.c:1646  */
    break;

  case 327:
#line 1209 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 4436 "y.tab.c" /* yacc.c:1646  */
    break;

  case 328:
#line 1213 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = (yyvsp[-1].plist); }
#line 4442 "y.tab.c" /* yacc.c:1646  */
    break;

  case 329:
#line 1215 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList(new Parameter(0, new BuiltinType("void"))); }
#line 4448 "y.tab.c" /* yacc.c:1646  */
    break;

  case 330:
#line 1219 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = (yyvsp[-1].plist); }
#line 4454 "y.tab.c" /* yacc.c:1646  */
    break;

  case 331:
#line 1221 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = 0; }
#line 4460 "y.tab.c" /* yacc.c:1646  */
    break;

  case 332:
#line 1225 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = 0; }
#line 4466 "y.tab.c" /* yacc.c:1646  */
    break;

  case 333:
#line 1227 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4472 "y.tab.c" /* yacc.c:1646  */
    break;

  case 334:
#line 1231 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = 0; }
#line 4478 "y.tab.c" /* yacc.c:1646  */
    break;

  case 335:
#line 1233 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[0].sc)); }
#line 4484 "y.tab.c" /* yacc.c:1646  */
    break;

  case 336:
#line 1235 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[-2].slist)); }
#line 4490 "y.tab.c" /* yacc.c:1646  */
    break;

  case 337:
#line 1239 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.slist) = new SListConstruct((yyvsp[0].sc)); }
#line 4496 "y.tab.c" /* yacc.c:1646  */
    break;

  case 338:
#line 1241 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.slist) = new SListConstruct((yyvsp[-1].sc), (yyvsp[0].slist));  }
#line 4502 "y.tab.c" /* yacc.c:1646  */
    break;

  case 339:
#line 1245 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.olist) = new OListConstruct((yyvsp[0].sc)); }
#line 4508 "y.tab.c" /* yacc.c:1646  */
    break;

  case 340:
#line 1247 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.olist) = new OListConstruct((yyvsp[-1].sc), (yyvsp[0].slist)); }
#line 4514 "y.tab.c" /* yacc.c:1646  */
    break;

  case 341:
#line 1251 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.clist) = new CaseListConstruct((yyvsp[0].when)); }
#line 4520 "y.tab.c" /* yacc.c:1646  */
    break;

  case 342:
#line 1253 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.clist) = new CaseListConstruct((yyvsp[-1].when), (yyvsp[0].clist)); }
#line 4526 "y.tab.c" /* yacc.c:1646  */
    break;

  case 343:
#line 1255 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("case blocks can only contain when clauses",
		        (yylsp[0]).first_column, (yylsp[0]).last_column);
		  (yyval.clist) = 0;
		}
#line 4536 "y.tab.c" /* yacc.c:1646  */
    break;

  case 344:
#line 1263 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4542 "y.tab.c" /* yacc.c:1646  */
    break;

  case 345:
#line 1265 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = 0; }
#line 4548 "y.tab.c" /* yacc.c:1646  */
    break;

  case 346:
#line 1269 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-2].entrylist), 0); }
#line 4554 "y.tab.c" /* yacc.c:1646  */
    break;

  case 347:
#line 1271 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-1].entrylist), (yyvsp[0].sc)); }
#line 4560 "y.tab.c" /* yacc.c:1646  */
    break;

  case 348:
#line 1273 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-3].entrylist), (yyvsp[-1].slist)); }
#line 4566 "y.tab.c" /* yacc.c:1646  */
    break;

  case 349:
#line 1277 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4572 "y.tab.c" /* yacc.c:1646  */
    break;

  case 350:
#line 1279 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4578 "y.tab.c" /* yacc.c:1646  */
    break;

  case 351:
#line 1281 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4584 "y.tab.c" /* yacc.c:1646  */
    break;

  case 352:
#line 1283 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4590 "y.tab.c" /* yacc.c:1646  */
    break;

  case 353:
#line 1285 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4596 "y.tab.c" /* yacc.c:1646  */
    break;

  case 354:
#line 1287 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4602 "y.tab.c" /* yacc.c:1646  */
    break;

  case 355:
#line 1289 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4608 "y.tab.c" /* yacc.c:1646  */
    break;

  case 356:
#line 1291 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4614 "y.tab.c" /* yacc.c:1646  */
    break;

  case 357:
#line 1293 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4620 "y.tab.c" /* yacc.c:1646  */
    break;

  case 358:
#line 1295 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4626 "y.tab.c" /* yacc.c:1646  */
    break;

  case 359:
#line 1297 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4632 "y.tab.c" /* yacc.c:1646  */
    break;

  case 360:
#line 1299 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4638 "y.tab.c" /* yacc.c:1646  */
    break;

  case 361:
#line 1303 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new SerialConstruct((yyvsp[-2].strval), (yyvsp[-4].strval), (yylsp[-3]).first_line); }
#line 4644 "y.tab.c" /* yacc.c:1646  */
    break;

  case 362:
#line 1305 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new OverlapConstruct((yyvsp[-1].olist)); }
#line 4650 "y.tab.c" /* yacc.c:1646  */
    break;

  case 363:
#line 1307 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = (yyvsp[0].when); }
#line 4656 "y.tab.c" /* yacc.c:1646  */
    break;

  case 364:
#line 1309 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new CaseConstruct((yyvsp[-1].clist)); }
#line 4662 "y.tab.c" /* yacc.c:1646  */
    break;

  case 365:
#line 1311 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForConstruct((yyvsp[-8].intexpr), (yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4668 "y.tab.c" /* yacc.c:1646  */
    break;

  case 366:
#line 1313 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForConstruct((yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4674 "y.tab.c" /* yacc.c:1646  */
    break;

  case 367:
#line 1315 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-9].strval)), (yyvsp[-6].intexpr),
		             (yyvsp[-4].intexpr), (yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4681 "y.tab.c" /* yacc.c:1646  */
    break;

  case 368:
#line 1318 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-11].strval)), (yyvsp[-8].intexpr),
		             (yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4688 "y.tab.c" /* yacc.c:1646  */
    break;

  case 369:
#line 1321 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new IfConstruct((yyvsp[-3].intexpr), (yyvsp[-1].sc), (yyvsp[0].sc)); }
#line 4694 "y.tab.c" /* yacc.c:1646  */
    break;

  case 370:
#line 1323 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new IfConstruct((yyvsp[-5].intexpr), (yyvsp[-2].slist), (yyvsp[0].sc)); }
#line 4700 "y.tab.c" /* yacc.c:1646  */
    break;

  case 371:
#line 1325 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new WhileConstruct((yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4706 "y.tab.c" /* yacc.c:1646  */
    break;

  case 372:
#line 1327 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new WhileConstruct((yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4712 "y.tab.c" /* yacc.c:1646  */
    break;

  case 373:
#line 1329 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new SerialConstruct((yyvsp[-2].strval), NULL, (yyloc).first_line); }
#line 4718 "y.tab.c" /* yacc.c:1646  */
    break;

  case 374:
#line 1331 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("unknown SDAG construct or malformed entry method declaration.\n"
		        "You may have forgotten to terminate a previous entry method declaration with a"
		        " semicolon or forgotten to mark a block of sequential SDAG code as 'serial'",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
#line 4730 "y.tab.c" /* yacc.c:1646  */
    break;

  case 375:
#line 1341 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = 0; }
#line 4736 "y.tab.c" /* yacc.c:1646  */
    break;

  case 376:
#line 1343 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ElseConstruct((yyvsp[0].sc)); }
#line 4742 "y.tab.c" /* yacc.c:1646  */
    break;

  case 377:
#line 1345 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ElseConstruct((yyvsp[-1].slist)); }
#line 4748 "y.tab.c" /* yacc.c:1646  */
    break;

  case 378:
#line 1349 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intexpr) = new IntExprConstruct((yyvsp[0].strval)); }
#line 4754 "y.tab.c" /* yacc.c:1646  */
    break;

  case 379:
#line 1353 "xi-grammar.y" /* yacc.c:1646  */
    { in_int_expr = 0; (yyval.intval) = 0; }
#line 4760 "y.tab.c" /* yacc.c:1646  */
    break;

  case 380:
#line 1357 "xi-grammar.y" /* yacc.c:1646  */
    { in_int_expr = 1; (yyval.intval) = 0; }
#line 4766 "y.tab.c" /* yacc.c:1646  */
    break;

  case 381:
#line 1361 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[-1].strval), (yyvsp[0].plist), 0, 0, 0, (yyloc).first_line, (yyloc).last_line);
		  firstRdma = true;
		}
#line 4775 "y.tab.c" /* yacc.c:1646  */
    break;

  case 382:
#line 1366 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[-4].strval), (yyvsp[0].plist), 0, 0, (yyvsp[-2].strval), (yyloc).first_line, (yyloc).last_line);
		  firstRdma = true;
		}
#line 4784 "y.tab.c" /* yacc.c:1646  */
    break;

  case 383:
#line 1373 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.entrylist) = new EntryList((yyvsp[0].entry)); }
#line 4790 "y.tab.c" /* yacc.c:1646  */
    break;

  case 384:
#line 1375 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.entrylist) = new EntryList((yyvsp[-2].entry),(yyvsp[0].entrylist)); }
#line 4796 "y.tab.c" /* yacc.c:1646  */
    break;

  case 385:
#line 1379 "xi-grammar.y" /* yacc.c:1646  */
    { in_bracket=1; }
#line 4802 "y.tab.c" /* yacc.c:1646  */
    break;

  case 386:
#line 1382 "xi-grammar.y" /* yacc.c:1646  */
    { in_bracket=0; }
#line 4808 "y.tab.c" /* yacc.c:1646  */
    break;

  case 387:
#line 1386 "xi-grammar.y" /* yacc.c:1646  */
    { if (!macroDefined((yyvsp[0].strval), 1)) in_comment = 1; }
#line 4814 "y.tab.c" /* yacc.c:1646  */
    break;

  case 388:
#line 1390 "xi-grammar.y" /* yacc.c:1646  */
    { if (!macroDefined((yyvsp[0].strval), 0)) in_comment = 1; }
#line 4820 "y.tab.c" /* yacc.c:1646  */
    break;


#line 4824 "y.tab.c" /* yacc.c:1646  */
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
#line 1393 "xi-grammar.y" /* yacc.c:1906  */


void yyerror(const char *s) 
{
	fprintf(stderr, "[PARSE-ERROR] Unexpected/missing token at line %d. Current token being parsed: '%s'.\n", lineno, yytext);
}
