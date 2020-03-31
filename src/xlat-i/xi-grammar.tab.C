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
    SIZET = 324,
    BOOL = 325,
    ACCEL = 326,
    READWRITE = 327,
    WRITEONLY = 328,
    ACCELBLOCK = 329,
    MEMCRITICAL = 330,
    REDUCTIONTARGET = 331,
    CASE = 332,
    TYPENAME = 333
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
#define SIZET 324
#define BOOL 325
#define ACCEL 326
#define READWRITE 327
#define WRITEONLY 328
#define ACCELBLOCK 329
#define MEMCRITICAL 330
#define REDUCTIONTARGET 331
#define CASE 332
#define TYPENAME 333

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED

union YYSTYPE
{
#line 53 "xi-grammar.y" /* yacc.c:355  */

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

#line 357 "y.tab.c" /* yacc.c:355  */
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

#line 388 "y.tab.c" /* yacc.c:358  */

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
#define YYLAST   1577

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  95
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  119
/* YYNRULES -- Number of rules.  */
#define YYNRULES  393
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  783

/* YYTRANSLATE[YYX] -- Symbol number corresponding to YYX as returned
   by yylex, with out-of-bounds checking.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   333

#define YYTRANSLATE(YYX)                                                \
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, without out-of-bounds checking.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    89,     2,
      87,    88,    86,     2,    83,    94,    90,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    80,    79,
      84,    93,    85,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    91,     2,    92,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    81,     2,    82,     2,     2,     2,     2,
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
      75,    76,    77,    78
};

#if YYDEBUG
  /* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   201,   201,   206,   209,   214,   215,   219,   221,   226,
     227,   232,   234,   235,   236,   238,   239,   240,   242,   243,
     244,   245,   246,   250,   251,   252,   253,   254,   255,   256,
     257,   258,   259,   260,   261,   262,   263,   264,   265,   266,
     267,   268,   271,   272,   273,   274,   275,   276,   277,   278,
     279,   280,   281,   283,   285,   286,   289,   290,   291,   292,
     296,   298,   304,   311,   315,   322,   324,   329,   330,   334,
     336,   338,   340,   342,   355,   357,   359,   361,   367,   369,
     371,   373,   375,   377,   379,   381,   383,   385,   393,   395,
     397,   401,   403,   408,   409,   414,   415,   419,   421,   423,
     425,   427,   429,   431,   433,   435,   437,   439,   441,   443,
     445,   447,   449,   451,   455,   456,   461,   469,   471,   475,
     479,   481,   485,   489,   491,   493,   495,   497,   499,   503,
     505,   507,   509,   511,   515,   517,   519,   521,   523,   525,
     529,   531,   533,   535,   537,   539,   543,   547,   552,   553,
     557,   561,   566,   567,   572,   573,   583,   585,   589,   591,
     596,   597,   601,   603,   608,   609,   613,   618,   619,   623,
     625,   629,   631,   636,   637,   641,   642,   645,   649,   651,
     655,   657,   659,   664,   665,   669,   671,   675,   677,   681,
     685,   689,   695,   699,   701,   705,   707,   711,   715,   719,
     723,   725,   730,   731,   736,   737,   739,   741,   750,   752,
     754,   756,   758,   760,   764,   766,   770,   774,   776,   778,
     780,   782,   786,   788,   793,   800,   804,   806,   808,   809,
     811,   813,   815,   819,   821,   823,   829,   835,   844,   846,
     848,   854,   862,   864,   867,   871,   875,   877,   882,   884,
     892,   894,   896,   898,   900,   902,   904,   906,   908,   910,
     912,   915,   925,   942,   959,   961,   965,   970,   971,   973,
     980,   984,   985,   989,   990,   991,   992,   995,   997,   999,
    1001,  1003,  1005,  1007,  1009,  1011,  1013,  1015,  1017,  1019,
    1021,  1023,  1025,  1027,  1031,  1040,  1042,  1044,  1049,  1050,
    1052,  1061,  1062,  1064,  1070,  1076,  1082,  1090,  1097,  1105,
    1112,  1114,  1116,  1118,  1123,  1133,  1145,  1146,  1147,  1150,
    1151,  1152,  1153,  1160,  1166,  1175,  1182,  1188,  1194,  1202,
    1204,  1208,  1210,  1214,  1216,  1220,  1222,  1227,  1228,  1232,
    1234,  1236,  1240,  1242,  1246,  1248,  1252,  1254,  1256,  1264,
    1267,  1270,  1272,  1274,  1278,  1280,  1282,  1284,  1286,  1288,
    1290,  1292,  1294,  1296,  1298,  1300,  1304,  1306,  1308,  1310,
    1312,  1314,  1316,  1319,  1322,  1324,  1326,  1328,  1330,  1332,
    1343,  1344,  1346,  1350,  1354,  1358,  1362,  1367,  1374,  1376,
    1380,  1383,  1387,  1391
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
  "FLOAT", "DOUBLE", "UNSIGNED", "SIZET", "BOOL", "ACCEL", "READWRITE",
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
  "Entry", "AccelBlock", "EReturn", "EAttribs", "AttributeArg",
  "AttributeArgList", "EAttribList", "EAttrib", "DefaultParameter",
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
     325,   326,   327,   328,   329,   330,   331,   332,   333,    59,
      58,   123,   125,    44,    60,    62,    42,    40,    41,    38,
      46,    91,    93,    61,    45
};
# endif

#define YYPACT_NINF -589

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-589)))

#define YYTABLE_NINF -345

#define yytable_value_is_error(Yytable_value) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
     281,  1329,  1329,    82,  -589,   281,  -589,  -589,  -589,  -589,
    -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,
    -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,
    -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,
    -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,
    -589,  -589,  -589,  -589,  -589,   192,   192,  -589,  -589,  -589,
     903,   -26,  -589,  -589,  -589,    45,  1329,   224,  1329,  1329,
     233,  1041,    59,  1058,   903,  -589,  -589,  -589,  -589,   437,
     114,   133,  -589,   158,  -589,  -589,  -589,   -26,    -7,  1374,
     210,   210,   -10,    56,   190,   190,   190,   190,   203,   214,
    1329,   201,   191,   903,  -589,  -589,  -589,  -589,  -589,  -589,
    -589,  -589,   312,  -589,  -589,  -589,  -589,   238,  -589,  -589,
    -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,   -26,
    -589,  -589,  -589,  1196,  1465,   903,   158,   240,   149,    -7,
     247,  1499,  -589,  1482,  -589,   -20,  -589,  -589,  -589,  -589,
     275,  -589,  -589,   133,   107,  -589,  -589,   259,   279,   282,
    -589,    63,   133,  -589,   133,   133,   276,   133,   299,  -589,
     115,  1329,  1329,  1329,  1329,   108,   317,   318,   346,  1329,
    -589,  -589,  -589,  1391,   303,   190,   190,   190,   190,   317,
     214,  -589,  -589,  -589,  -589,  -589,   -26,  -589,  -589,  -589,
    -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,
    -589,  -589,   345,  -589,  -589,  -589,   323,   127,  1465,   259,
     279,   282,    31,  -589,    56,   334,     1,    -7,   358,    -7,
     330,  -589,   238,   333,    39,  -589,  -589,  -589,   291,  -589,
    -589,   107,  1049,  -589,  -589,  -589,  -589,  -589,   335,   301,
     336,    79,   152,    93,   328,   157,    56,  -589,  -589,   332,
     344,   347,   352,   352,   352,   352,  -589,  1329,   341,   351,
     350,   172,  1329,   384,  1329,  -589,  -589,   357,   356,   359,
     815,    61,   126,  1329,   360,   365,   238,  1329,  1329,  1329,
    1329,  1329,  1329,  -589,  -589,  -589,  1196,  1329,   403,  -589,
     311,   354,  1329,  -589,  -589,  -589,   370,   377,   373,   371,
      -7,   -26,   133,  -589,  -589,  -589,  -589,  -589,   379,  -589,
     380,  -589,  1329,   374,   376,   383,  -589,   385,  -589,    -7,
     210,  1049,   210,   210,  1049,   210,  -589,  -589,   115,  -589,
      56,   254,   254,   254,   254,   375,  -589,   384,  -589,   352,
     352,  -589,   346,    24,   387,   386,   184,   388,   167,  -589,
     389,  1391,  -589,  -589,   352,   352,   352,   352,   352,   262,
    -589,   390,   397,   395,   394,   399,   402,   347,    -7,   358,
      -7,    -7,  -589,    79,  1049,  -589,   406,   407,   418,  -589,
    -589,   393,  -589,   422,   426,   424,   133,   428,   429,  -589,
     431,  -589,    53,   -26,  -589,  -589,  -589,  -589,  -589,  -589,
     254,   254,  -589,  -589,  -589,  1482,    27,   436,   434,  1482,
    -589,  -589,   435,  -589,  -589,  -589,  -589,  -589,   254,   254,
     254,   254,   254,   502,   -26,   469,  1329,   446,   440,   442,
    -589,   447,  -589,  -589,  -589,  -589,  -589,  -589,   448,   443,
    -589,  -589,  -589,  -589,   449,  -589,   168,   453,  -589,    56,
    -589,   734,   505,   467,   238,    53,  -589,  -589,  -589,  -589,
    1329,  -589,  -589,  1329,  -589,   494,  -589,  -589,  -589,  -589,
    -589,   473,  -589,  -589,  1196,   468,  -589,  1411,  -589,  1445,
    -589,   210,   210,   210,  -589,  1057,  1138,  -589,   238,   -26,
    -589,   466,   386,   386,   238,  -589,  -589,  1482,  1482,  -589,
    1329,    -7,   478,   474,   475,   476,   480,   481,   479,   483,
     447,  1329,  -589,   495,   238,  -589,  -589,   -26,  1329,    -7,
      -7,    25,   496,  1445,  -589,  -589,  -589,  -589,  -589,   529,
     497,   447,  -589,   -26,   482,   498,   499,  -589,   349,  -589,
    -589,  -589,  1329,  -589,   485,   503,   485,   548,   524,   549,
     485,   527,   409,   -26,    -7,  -589,  -589,  -589,   589,  -589,
    -589,  -589,  -589,   158,  -589,   447,  -589,    -7,   554,    -7,
     125,   528,   602,   625,  -589,   533,    -7,   154,   537,   348,
     247,   522,   497,   530,  -589,   541,   532,   540,  -589,    -7,
     548,   536,  -589,   543,   552,    -7,   540,   485,   539,   485,
     550,   549,   485,   551,    -7,   553,   154,  -589,   238,  -589,
     238,   579,  -589,   294,   533,    -7,   485,  -589,   644,   393,
    -589,  -589,   555,  -589,  -589,   247,   663,    -7,   582,    -7,
     625,   533,    -7,   154,   247,  -589,  -589,  -589,  -589,  -589,
    -589,  -589,  -589,  -589,  1329,   559,   557,   562,    -7,   563,
      -7,   409,  -589,   447,  -589,   238,   409,   590,   565,   564,
     540,   573,    -7,   540,   575,   238,   574,  1482,  1354,  -589,
     247,    -7,   581,   577,  -589,  -589,   580,   851,  -589,    -7,
     485,   892,  -589,   247,   901,  -589,  -589,  1329,  1329,    -7,
     583,  -589,  1329,   540,    -7,  -589,   590,   409,  -589,   586,
      -7,   409,  -589,   238,   409,   590,  -589,   228,    52,   584,
    1329,   238,   942,   594,  -589,   596,    -7,   600,   599,  -589,
     603,  -589,  -589,  1329,  1329,  1254,   597,  1329,  -589,   261,
     -26,   409,  -589,    -7,  -589,   540,    -7,  -589,   590,   289,
    -589,   592,   212,  1329,   284,  -589,   604,   540,   951,   612,
    -589,  -589,  -589,  -589,  -589,  -589,  -589,   958,   409,  -589,
      -7,   409,  -589,   616,   540,   617,  -589,   965,  -589,   409,
    -589,   618,  -589
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
       0,     0,    60,    70,   392,   393,   308,   265,   301,     0,
     152,   152,   152,     0,   160,   160,   160,   160,     0,   154,
       0,     0,     0,     0,    78,   226,   227,    72,    79,    80,
      81,    82,     0,    83,    71,   229,   228,     9,   260,   252,
     253,   254,   255,   256,   258,   259,   257,   250,   251,    76,
      77,    68,   269,     0,     0,     0,    69,     0,   302,   301,
       0,     0,   111,     0,    97,    98,    99,   100,   108,   109,
       0,   112,   113,     0,    95,   117,   118,   123,   124,   125,
     126,   145,     0,   153,     0,     0,     0,     0,   242,   230,
       0,     0,     0,     0,     0,     0,     0,   167,     0,     0,
     232,   244,   231,     0,     0,   160,   160,   160,   160,     0,
     154,   217,   218,   219,   220,   221,    10,    66,   294,   277,
     278,   279,   280,   286,   287,   288,   293,   281,   282,   283,
     284,   285,   164,   289,   291,   292,     0,   273,     0,   129,
     130,   131,   139,   266,     0,     0,     0,   301,   298,   301,
       0,   309,     0,     0,   127,   107,   110,   101,   102,   105,
     106,    95,    93,   115,   119,   120,   121,   128,     0,   144,
       0,   148,   236,   233,     0,   238,     0,   171,   172,     0,
     162,    95,   183,   183,   183,   183,   166,     0,     0,   169,
       0,     0,     0,     0,     0,   158,   159,     0,   156,   180,
       0,     0,   126,     0,   214,     0,     9,     0,     0,     0,
       0,     0,     0,   165,   290,   268,     0,     0,   132,   133,
     138,     0,     0,    75,    62,    61,     0,   299,     0,     0,
     301,   264,     0,   103,   104,   116,    89,    90,    91,    94,
       0,    88,     0,   143,     0,     0,   390,   148,   150,   301,
     152,     0,   152,   152,     0,   152,   243,   161,     0,   114,
       0,     0,     0,     0,     0,     0,   192,     0,   168,   183,
     183,   155,     0,   173,     0,   202,    60,     0,     0,   212,
     204,     0,   216,    74,   183,   183,   183,   183,   183,     0,
     275,     0,   271,     0,   137,     0,     0,    95,   301,   298,
     301,   301,   306,   148,     0,    96,     0,     0,     0,   142,
     149,     0,   146,     0,     0,     0,     0,     0,     0,   163,
     185,   184,     0,   222,   187,   188,   189,   190,   191,   170,
       0,     0,   157,   174,   181,     0,   173,     0,     0,     0,
     210,   211,     0,   205,   206,   207,   213,   215,     0,     0,
       0,     0,     0,   173,   200,     0,     0,   274,     0,     0,
     136,     0,   304,   300,   305,   303,   151,    92,     0,     0,
     141,   391,   147,   237,     0,   234,     0,     0,   239,     0,
     249,     0,     0,     0,     0,     0,   245,   246,   193,   194,
       0,   179,   182,     0,   203,     0,   195,   196,   197,   198,
     199,     0,   270,   272,     0,     0,   135,     0,    73,     0,
     140,   152,   152,   152,   186,     0,     0,   247,     9,   248,
     225,   175,   202,   202,     0,   276,   134,     0,     0,   334,
     310,   301,   329,     0,     0,     0,     0,     0,     0,    60,
       0,     0,   223,     0,     0,   208,   209,   201,     0,   301,
     301,   173,     0,     0,   333,   122,   235,   241,   240,     0,
       0,     0,   176,   177,     0,     0,     0,   307,     0,   311,
     313,   330,     0,   379,     0,     0,     0,     0,     0,   350,
       0,     0,     0,   339,   301,   262,   368,   340,   337,   314,
     315,   296,   295,   297,   312,     0,   385,   301,     0,   301,
       0,   388,     0,     0,   349,     0,   301,     0,     0,     0,
       0,     0,     0,     0,   383,     0,     0,     0,   386,   301,
       0,     0,   352,     0,     0,   301,     0,     0,     0,     0,
       0,   350,     0,     0,   301,     0,   346,   348,     9,   343,
       9,     0,   261,     0,     0,   301,     0,   384,     0,     0,
     389,   351,     0,   367,   345,     0,     0,   301,     0,   301,
       0,     0,   301,     0,     0,   369,   347,   341,   378,   338,
     316,   317,   318,   336,     0,     0,   331,     0,   301,     0,
     301,     0,   376,     0,   353,     9,     0,   380,     0,     0,
       0,     0,   301,     0,     0,     9,     0,     0,     0,   335,
       0,   301,     0,     0,   387,   366,     0,     0,   374,   301,
       0,     0,   355,     0,     0,   356,   365,     0,     0,   301,
       0,   332,     0,     0,   301,   377,   380,     0,   381,     0,
     301,     0,   363,     9,     0,   380,   319,     0,     0,     0,
       0,     0,     0,     0,   375,     0,   301,     0,     0,   354,
       0,   361,   327,     0,     0,     0,     0,     0,   325,     0,
     263,     0,   371,   301,   382,     0,   301,   364,   380,     0,
     321,     0,     0,     0,     0,   328,     0,     0,     0,     0,
     362,   324,   323,   322,   320,   326,   370,     0,     0,   358,
     301,     0,   372,     0,     0,     0,   357,     0,   373,     0,
     359,     0,   360
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -589,  -589,   661,  -589,   -53,  -276,    -1,   -60,   632,   648,
     -25,  -589,  -589,  -589,  -237,  -589,  -215,  -589,  -134,   -75,
    -126,  -123,  -115,  -169,   571,   487,  -589,   -85,  -589,  -589,
    -292,  -589,  -589,   -79,   525,   364,  -589,   -63,   382,  -589,
    -589,   545,   381,  -589,   186,  -589,  -589,  -250,  -589,   -57,
     264,  -589,  -589,  -589,    -5,  -589,  -589,  -589,  -589,  -589,
    -589,  -334,   366,  -589,   363,   647,  -589,  -168,   265,   656,
    -589,  -589,   509,  -589,  -589,  -589,  -589,   267,  -589,   235,
     290,  -589,   314,  -287,  -589,  -589,   412,   -83,  -484,   -64,
    -565,  -589,  -589,  -463,  -589,  -589,  -404,   110,  -491,  -589,
    -589,   194,  -553,   153,  -588,   181,  -529,  -589,  -509,  -558,
    -549,  -540,  -470,  -589,   195,   226,   165,  -589,  -589
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     3,     4,    71,   403,   197,   261,   154,     5,    62,
      72,    73,    74,   318,   319,   320,   243,   155,   262,   156,
     157,   158,   159,   160,   161,   222,   223,   321,   391,   327,
     328,   105,   106,   164,   179,   277,   278,   171,   259,   294,
     269,   176,   270,   260,   415,   524,   416,   417,   107,   341,
     401,   108,   109,   110,   177,   111,   191,   192,   193,   194,
     195,   420,   359,   284,   285,   462,   113,   404,   463,   464,
     115,   116,   169,   182,   465,   466,   130,   467,    75,   224,
     134,   372,   373,   216,   217,   574,   308,   594,   511,   564,
     232,   512,   655,   717,   700,   656,   513,   657,   488,   624,
     592,   565,   588,   603,   615,   585,   566,   617,   589,   688,
     595,   628,   577,   581,   582,   329,   452,    76,    77
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[] =
{
      55,    56,    61,    61,   162,   140,    88,    83,   219,   370,
     363,   220,   165,   167,   282,   233,   304,    87,   168,   221,
     129,   136,   421,   529,   530,   620,   315,   163,   646,   540,
     597,   567,   172,   173,   174,   390,   619,   606,   263,   264,
     265,   413,   413,   235,   413,   279,   339,   236,   632,   131,
     568,   634,   138,    78,   460,   674,   230,   305,   616,    89,
      90,    91,    92,    93,   196,    80,   636,    84,    85,   299,
     665,   100,   101,   602,   604,   102,   659,   247,   184,   675,
     139,   166,    57,   567,   593,   514,   579,   616,    79,   598,
     586,   446,   219,   241,   394,   220,   461,   397,   283,   180,
     268,   247,   251,   221,   252,   253,   414,   255,   683,  -178,
     225,   682,    82,   686,   616,   702,   547,   356,   548,   662,
     300,   301,   287,   288,   289,   290,   248,   667,   713,   551,
     691,   604,   703,   694,   153,  -224,   737,   637,   349,   639,
     350,   117,   642,   547,   306,   257,   309,   447,   724,   302,
     248,   357,   249,   250,   725,   723,   660,   731,   728,   137,
     266,   730,   441,   722,    82,   267,   471,   258,   525,   526,
     326,   272,   684,   137,   405,   406,   407,   331,   708,   311,
     332,   168,   712,   481,   291,   715,   153,   137,   756,    82,
     760,   242,   282,   699,   757,   135,   268,   505,   607,   608,
     609,   557,   610,   611,   612,   758,   342,   343,   344,  -204,
     296,  -204,   487,   742,   297,   773,   326,   767,   775,   358,
     710,   241,   522,    82,   423,   424,   781,   382,    82,   267,
     227,   613,   137,   196,   777,    86,   228,   137,   137,   330,
     229,   334,   468,   469,   335,   153,   392,   163,   137,   769,
     153,   393,   383,   395,   396,   492,   398,   739,   772,   181,
     476,   477,   478,   479,   480,   400,   345,  -202,   780,  -202,
     749,    59,   752,    60,   754,   183,   283,   419,    81,   355,
      82,   170,   360,   425,     1,     2,   364,   365,   366,   367,
     368,   369,   410,   411,   175,   442,   371,   444,   445,   733,
     650,   377,   734,   735,   763,   178,   736,   428,   429,   430,
     431,   432,    59,   732,    86,   733,   434,    59,   734,   735,
     226,   386,   736,   185,   186,   187,   188,   189,   190,   231,
     470,   142,   143,    59,   474,   402,   456,   237,   238,   239,
     240,    59,   647,   433,   648,   244,   755,   254,   733,   553,
      82,   734,   735,   313,   314,   736,   144,   145,   146,   147,
     148,   149,   150,   151,   152,   245,   651,   652,   246,   765,
     219,   733,   153,   220,   734,   735,   733,   761,   736,   734,
     735,   221,   256,   736,   400,   286,   653,   275,   276,   685,
     323,   324,   554,   555,   556,   557,   558,   559,   560,   696,
     374,   375,   510,   293,   510,    82,   571,   572,   271,   273,
     553,   499,   515,   516,   517,   295,   303,   307,   310,   312,
     333,   322,   528,   528,   337,   561,   325,   338,   532,    86,
    -342,   242,   340,   346,   347,   371,   266,   729,   132,   352,
     353,   299,   348,   361,   376,   196,   545,   546,   510,   351,
     362,   527,   378,   554,   555,   556,   557,   558,   559,   560,
     379,   380,   384,   381,   387,   385,   388,   408,  -308,   501,
     435,   543,   502,   389,  -267,  -267,   326,   418,   422,   419,
     436,   590,   358,   437,   438,   451,   561,   563,   573,   439,
      86,  -308,   440,  -267,   448,   520,  -308,   449,   553,  -267,
    -267,  -267,  -267,  -267,  -267,  -267,  -267,  -267,   450,   531,
     453,   454,   455,   457,   459,  -267,   629,   458,   472,   413,
     541,   605,   635,   614,   473,   475,   482,   544,   133,   484,
     485,   644,   486,   490,   487,   489,   491,   553,   654,   563,
     493,   554,   555,   556,   557,   558,   559,   560,   461,   498,
     503,   575,   614,   553,   668,   504,   670,   523,   506,   673,
     658,   533,   534,   535,   536,   196,   552,   196,   537,   538,
     -11,   539,   576,   547,   561,   680,    59,   672,   562,   614,
     554,   555,   556,   557,   558,   559,   560,   542,   550,   693,
     569,   570,   698,   654,   578,  -308,   554,   555,   556,   557,
     558,   559,   560,   553,   580,   583,   709,   584,   587,   591,
     596,   600,   196,   561,    86,   621,   719,    86,   631,   618,
     625,   623,   196,  -308,   626,   633,   553,   727,   627,   561,
     638,   640,   643,    86,  -344,   645,   649,   664,   669,   677,
     678,   687,   681,   745,   689,   553,   554,   555,   556,   557,
     558,   559,   560,   676,   679,   692,   690,   695,   697,   705,
     196,   704,   706,   759,   553,   726,    58,   720,   740,   554,
     555,   556,   557,   558,   559,   560,   738,   743,   744,   561,
     746,   747,   753,   601,   762,   748,   766,   774,   554,   555,
     556,   557,   558,   559,   560,   770,   716,   718,   776,   778,
     782,   721,   561,   104,    63,   298,    86,   554,   555,   556,
     557,   558,   559,   560,   234,   292,   412,   549,   112,   716,
     399,   561,   274,   494,   427,   661,   426,   114,   409,   497,
     500,   521,   716,   750,   716,   132,   716,  -267,  -267,  -267,
     561,  -267,  -267,  -267,   666,  -267,  -267,  -267,  -267,  -267,
     483,   496,   764,  -267,  -267,  -267,  -267,  -267,  -267,  -267,
    -267,  -267,  -267,  -267,  -267,   336,  -267,  -267,  -267,  -267,
    -267,  -267,  -267,  -267,  -267,  -267,  -267,  -267,  -267,  -267,
    -267,  -267,  -267,  -267,  -267,  -267,   622,  -267,   701,  -267,
    -267,   443,   641,   671,   663,   630,  -267,  -267,  -267,  -267,
    -267,  -267,  -267,  -267,  -267,  -267,   599,     0,  -267,  -267,
    -267,  -267,  -267,     0,     0,     0,     0,     0,     6,     7,
       8,     0,     9,    10,    11,   495,    12,    13,    14,    15,
      16,     0,     0,     0,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,     0,    29,    30,    31,
      32,    33,   553,     0,    34,    35,    36,    37,    38,    39,
      40,    41,    42,    43,    44,    45,    46,     0,    47,     0,
      48,    49,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    50,     0,     0,    51,
      52,    53,    54,   553,     0,   554,   555,   556,   557,   558,
     559,   560,   553,     0,    64,   354,    -5,    -5,    65,    -5,
      -5,    -5,    -5,    -5,    -5,    -5,    -5,    -5,    -5,    -5,
       0,    -5,    -5,     0,     0,    -5,     0,     0,   561,     0,
       0,     0,   707,     0,     0,     0,   554,   555,   556,   557,
     558,   559,   560,   553,     0,   554,   555,   556,   557,   558,
     559,   560,   553,     0,     0,     0,     0,    66,    67,   553,
       0,     0,     0,    68,    69,     0,   553,     0,     0,   561,
       0,     0,     0,   711,     0,     0,     0,    70,   561,     0,
       0,     0,   714,     0,    -5,   -67,   554,   555,   556,   557,
     558,   559,   560,     0,     0,   554,   555,   556,   557,   558,
     559,   560,   554,   555,   556,   557,   558,   559,   560,   554,
     555,   556,   557,   558,   559,   560,     0,     0,     0,   561,
       0,     0,     0,   741,     0,     0,     0,     0,   561,     0,
       0,     0,   768,     0,     0,   561,     0,     0,     0,   771,
       0,     0,   561,     0,     1,     2,   779,    89,    90,    91,
      92,    93,    94,    95,    96,    97,    98,    99,   198,   100,
     101,     0,     0,   102,   118,   119,   120,   121,     0,   122,
     123,   124,   125,   126,     0,     0,     0,     0,   199,     0,
     200,   201,   202,   203,   204,   205,   142,   143,   206,   207,
     208,   209,   210,   211,     0,     0,     0,     0,     0,     0,
       0,   127,     0,     0,     0,    82,   316,   317,     0,   212,
     213,   144,   145,   146,   147,   148,   149,   150,   151,   152,
       0,     0,   103,     0,     0,     0,     0,   153,   518,     0,
       0,     0,   214,   215,     0,     0,     0,    59,     0,     0,
     128,     6,     7,     8,     0,     9,    10,    11,     0,    12,
      13,    14,    15,    16,     0,     0,     0,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,     0,
      29,    30,    31,    32,    33,   142,   218,    34,    35,    36,
      37,    38,    39,    40,    41,    42,    43,    44,    45,    46,
       0,    47,     0,    48,   519,     0,     0,   198,     0,     0,
     144,   145,   146,   147,   148,   149,   150,   151,   152,    50,
       0,     0,    51,    52,    53,    54,   153,   199,     0,   200,
     201,   202,   203,   204,   205,     0,     0,   206,   207,   208,
     209,   210,   211,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   212,   213,
       0,     0,     0,     0,     0,     0,     0,     6,     7,     8,
       0,     9,    10,    11,     0,    12,    13,    14,    15,    16,
       0,   214,   215,    17,    18,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,     0,    29,    30,    31,    32,
      33,     0,     0,    34,    35,    36,    37,    38,    39,    40,
      41,    42,    43,    44,    45,    46,     0,    47,     0,    48,
      49,   751,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    50,     0,     0,    51,    52,
      53,    54,     6,     7,     8,     0,     9,    10,    11,     0,
      12,    13,    14,    15,    16,     0,     0,     0,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
     650,    29,    30,    31,    32,    33,     0,     0,    34,    35,
      36,    37,    38,    39,    40,    41,    42,    43,    44,    45,
      46,     0,    47,     0,    48,    49,     0,     0,     0,     0,
     141,   142,   143,     0,     0,     0,     0,     0,     0,     0,
      50,     0,     0,    51,    52,    53,    54,     0,     0,   280,
      82,   142,   143,     0,     0,     0,   144,   145,   146,   147,
     148,   149,   150,   151,   152,     0,   651,   652,   142,   143,
      82,     0,   153,     0,     0,     0,   144,   145,   146,   147,
     148,   149,   150,   151,   152,     0,     0,    82,   142,   143,
     507,   508,   153,   144,   145,   146,   147,   148,   149,   150,
     151,   152,     0,     0,     0,     0,     0,    82,     0,   281,
       0,     0,     0,   144,   145,   146,   147,   148,   149,   150,
     151,   152,   142,   143,   507,   508,     0,     0,     0,   153,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   509,
       0,    82,   142,   218,     0,     0,     0,   144,   145,   146,
     147,   148,   149,   150,   151,   152,     0,     0,     0,   142,
     143,    82,     0,   153,     0,     0,     0,   144,   145,   146,
     147,   148,   149,   150,   151,   152,   142,     0,    82,     0,
       0,     0,     0,   153,   144,   145,   146,   147,   148,   149,
     150,   151,   152,     0,     0,    82,     0,     0,     0,     0,
     153,   144,   145,   146,   147,   148,   149,   150,   151,   152,
       0,     0,     0,     0,     0,     0,     0,   153
};

static const yytype_int16 yycheck[] =
{
       1,     2,    55,    56,    89,    88,    70,    67,   134,   296,
     286,   134,    91,    92,   183,   141,    15,    70,    93,   134,
      73,    81,   356,   507,   508,   590,   241,    37,   616,   520,
     579,   540,    95,    96,    97,   327,   589,   586,   172,   173,
     174,    17,    17,    63,    17,   179,   261,    67,   601,    74,
     541,   604,    59,    79,     1,   643,   139,    56,   587,     6,
       7,     8,     9,    10,   117,    66,   606,    68,    69,    38,
     635,    18,    19,   582,   583,    22,   625,    38,   103,   644,
      87,    91,     0,   592,   575,   489,   556,   616,    43,   580,
     560,   383,   218,   153,   331,   218,    43,   334,   183,   100,
     175,    38,   162,   218,   164,   165,    82,   167,   661,    82,
     135,   660,    56,   666,   643,   680,    91,    56,    93,   628,
      89,    90,   185,   186,   187,   188,    87,   636,   693,   533,
     670,   640,   681,   673,    78,    82,    84,   607,   272,   609,
     274,    82,   612,    91,   227,    30,   229,   384,   706,   224,
      87,    90,    89,    90,   707,   704,   626,   715,   711,    80,
      52,   714,   377,   703,    56,    57,   416,    52,   502,   503,
      91,   176,   663,    80,   342,   343,   344,    84,   687,   232,
      87,   256,   691,   433,   189,   694,    78,    80,   741,    56,
     748,    84,   361,   677,   743,    81,   271,   484,    44,    45,
      46,    47,    48,    49,    50,   745,   263,   264,   265,    83,
      83,    85,    87,   722,    87,   768,    91,   757,   771,    93,
     690,   281,   498,    56,    57,    58,   779,   310,    56,    57,
      81,    77,    80,   286,   774,    81,    87,    80,    80,    87,
      91,    84,   410,   411,    87,    78,   329,    37,    80,   758,
      78,   330,   312,   332,   333,    87,   335,   720,   767,    58,
     428,   429,   430,   431,   432,   340,   267,    83,   777,    85,
     733,    79,   735,    81,   737,    84,   361,    93,    54,   280,
      56,    91,   283,   358,     3,     4,   287,   288,   289,   290,
     291,   292,   349,   350,    91,   378,   297,   380,   381,    87,
       6,   302,    90,    91,    92,    91,    94,   364,   365,   366,
     367,   368,    79,    85,    81,    87,   369,    79,    90,    91,
      80,   322,    94,    11,    12,    13,    14,    15,    16,    82,
     415,    37,    38,    79,   419,    81,   396,    62,    63,    64,
      65,    79,   618,    81,   620,    86,    85,    71,    87,     1,
      56,    90,    91,    62,    63,    94,    62,    63,    64,    65,
      66,    67,    68,    69,    70,    86,    72,    73,    86,    85,
     496,    87,    78,   496,    90,    91,    87,    88,    94,    90,
      91,   496,    83,    94,   459,    82,    92,    41,    42,   665,
      89,    90,    44,    45,    46,    47,    48,    49,    50,   675,
      89,    90,   487,    58,   489,    56,    57,    58,    91,    91,
       1,   464,   491,   492,   493,    92,    82,    59,    88,    86,
      92,    86,   507,   508,    92,    77,    90,    83,   511,    81,
      82,    84,    80,    92,    83,   436,    52,   713,     1,    83,
      81,    38,    92,    83,    90,   498,   529,   530,   533,    92,
      85,   504,    82,    44,    45,    46,    47,    48,    49,    50,
      83,    88,    83,    92,    90,    85,    90,    92,    59,   470,
      80,   524,   473,    90,    37,    38,    91,    90,    90,    93,
      83,   564,    93,    88,    90,    92,    77,   540,   548,    90,
      81,    82,    90,    56,    88,   496,    87,    90,     1,    62,
      63,    64,    65,    66,    67,    68,    69,    70,    90,   510,
      88,    85,    88,    85,    83,    78,   599,    88,    82,    17,
     521,   585,   605,   587,    90,    90,    57,   528,    91,    83,
      90,   614,    90,    90,    87,    87,    87,     1,   623,   592,
      87,    44,    45,    46,    47,    48,    49,    50,    43,    82,
      56,   552,   616,     1,   637,    82,   639,    91,    90,   642,
     624,    83,    88,    88,    88,   618,    37,   620,    88,    88,
      87,    92,    87,    91,    77,   658,    79,   641,    81,   643,
      44,    45,    46,    47,    48,    49,    50,    92,    92,   672,
      92,    92,   677,   678,    91,    59,    44,    45,    46,    47,
      48,    49,    50,     1,    56,    81,   689,    58,    81,    20,
      56,    83,   665,    77,    81,    93,   699,    81,    82,    82,
      79,    91,   675,    87,    92,    82,     1,   710,    88,    77,
      91,    81,    81,    81,    82,    82,    57,    82,    56,    80,
      83,    51,    79,   726,    79,     1,    44,    45,    46,    47,
      48,    49,    50,   654,    92,    82,    92,    82,    84,    82,
     713,    80,    82,   746,     1,    79,     5,    84,   721,    44,
      45,    46,    47,    48,    49,    50,    92,    83,    82,    77,
      80,    82,    85,    81,    92,    82,    82,   770,    44,    45,
      46,    47,    48,    49,    50,    83,   697,   698,    82,    82,
      82,   702,    77,    71,    56,   218,    81,    44,    45,    46,
      47,    48,    49,    50,   143,   190,   352,   531,    71,   720,
     338,    77,   177,   459,   361,    81,   360,    71,   347,   462,
     465,   496,   733,   734,   735,     1,   737,     3,     4,     5,
      77,     7,     8,     9,    81,    11,    12,    13,    14,    15,
     436,   461,   753,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,   256,    32,    33,    34,    35,
      36,    37,    38,    39,    40,    41,    42,    43,    44,    45,
      46,    47,    48,    49,    50,    51,   592,    53,   678,    55,
      56,   379,   611,   640,   629,   600,    62,    63,    64,    65,
      66,    67,    68,    69,    70,    71,   580,    -1,    74,    75,
      76,    77,    78,    -1,    -1,    -1,    -1,    -1,     3,     4,
       5,    -1,     7,     8,     9,    91,    11,    12,    13,    14,
      15,    -1,    -1,    -1,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    -1,    32,    33,    34,
      35,    36,     1,    -1,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    -1,    53,    -1,
      55,    56,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    71,    -1,    -1,    74,
      75,    76,    77,     1,    -1,    44,    45,    46,    47,    48,
      49,    50,     1,    -1,     1,    90,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      -1,    18,    19,    -1,    -1,    22,    -1,    -1,    77,    -1,
      -1,    -1,    81,    -1,    -1,    -1,    44,    45,    46,    47,
      48,    49,    50,     1,    -1,    44,    45,    46,    47,    48,
      49,    50,     1,    -1,    -1,    -1,    -1,    54,    55,     1,
      -1,    -1,    -1,    60,    61,    -1,     1,    -1,    -1,    77,
      -1,    -1,    -1,    81,    -1,    -1,    -1,    74,    77,    -1,
      -1,    -1,    81,    -1,    81,    82,    44,    45,    46,    47,
      48,    49,    50,    -1,    -1,    44,    45,    46,    47,    48,
      49,    50,    44,    45,    46,    47,    48,    49,    50,    44,
      45,    46,    47,    48,    49,    50,    -1,    -1,    -1,    77,
      -1,    -1,    -1,    81,    -1,    -1,    -1,    -1,    77,    -1,
      -1,    -1,    81,    -1,    -1,    77,    -1,    -1,    -1,    81,
      -1,    -1,    77,    -1,     3,     4,    81,     6,     7,     8,
       9,    10,    11,    12,    13,    14,    15,    16,     1,    18,
      19,    -1,    -1,    22,     6,     7,     8,     9,    -1,    11,
      12,    13,    14,    15,    -1,    -1,    -1,    -1,    21,    -1,
      23,    24,    25,    26,    27,    28,    37,    38,    31,    32,
      33,    34,    35,    36,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    43,    -1,    -1,    -1,    56,    57,    58,    -1,    52,
      53,    62,    63,    64,    65,    66,    67,    68,    69,    70,
      -1,    -1,    81,    -1,    -1,    -1,    -1,    78,    71,    -1,
      -1,    -1,    75,    76,    -1,    -1,    -1,    79,    -1,    -1,
      82,     3,     4,     5,    -1,     7,     8,     9,    -1,    11,
      12,    13,    14,    15,    -1,    -1,    -1,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    -1,
      32,    33,    34,    35,    36,    37,    38,    39,    40,    41,
      42,    43,    44,    45,    46,    47,    48,    49,    50,    51,
      -1,    53,    -1,    55,    56,    -1,    -1,     1,    -1,    -1,
      62,    63,    64,    65,    66,    67,    68,    69,    70,    71,
      -1,    -1,    74,    75,    76,    77,    78,    21,    -1,    23,
      24,    25,    26,    27,    28,    -1,    -1,    31,    32,    33,
      34,    35,    36,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    52,    53,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,     3,     4,     5,
      -1,     7,     8,     9,    -1,    11,    12,    13,    14,    15,
      -1,    75,    76,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    -1,    32,    33,    34,    35,
      36,    -1,    -1,    39,    40,    41,    42,    43,    44,    45,
      46,    47,    48,    49,    50,    51,    -1,    53,    -1,    55,
      56,    57,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    71,    -1,    -1,    74,    75,
      76,    77,     3,     4,     5,    -1,     7,     8,     9,    -1,
      11,    12,    13,    14,    15,    -1,    -1,    -1,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
       6,    32,    33,    34,    35,    36,    -1,    -1,    39,    40,
      41,    42,    43,    44,    45,    46,    47,    48,    49,    50,
      51,    -1,    53,    -1,    55,    56,    -1,    -1,    -1,    -1,
      16,    37,    38,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      71,    -1,    -1,    74,    75,    76,    77,    -1,    -1,    18,
      56,    37,    38,    -1,    -1,    -1,    62,    63,    64,    65,
      66,    67,    68,    69,    70,    -1,    72,    73,    37,    38,
      56,    -1,    78,    -1,    -1,    -1,    62,    63,    64,    65,
      66,    67,    68,    69,    70,    -1,    -1,    56,    37,    38,
      39,    40,    78,    62,    63,    64,    65,    66,    67,    68,
      69,    70,    -1,    -1,    -1,    -1,    -1,    56,    -1,    78,
      -1,    -1,    -1,    62,    63,    64,    65,    66,    67,    68,
      69,    70,    37,    38,    39,    40,    -1,    -1,    -1,    78,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    88,
      -1,    56,    37,    38,    -1,    -1,    -1,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    -1,    -1,    -1,    37,
      38,    56,    -1,    78,    -1,    -1,    -1,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    37,    -1,    56,    -1,
      -1,    -1,    -1,    78,    62,    63,    64,    65,    66,    67,
      68,    69,    70,    -1,    -1,    56,    -1,    -1,    -1,    -1,
      78,    62,    63,    64,    65,    66,    67,    68,    69,    70,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    78
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     3,     4,    96,    97,   103,     3,     4,     5,     7,
       8,     9,    11,    12,    13,    14,    15,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    32,
      33,    34,    35,    36,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    53,    55,    56,
      71,    74,    75,    76,    77,   101,   101,     0,    97,    79,
      81,    99,   104,   104,     1,     5,    54,    55,    60,    61,
      74,    98,   105,   106,   107,   173,   212,   213,    79,    43,
     101,    54,    56,   102,   101,   101,    81,    99,   184,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      18,    19,    22,    81,   103,   126,   127,   143,   146,   147,
     148,   150,   160,   161,   164,   165,   166,    82,     6,     7,
       8,     9,    11,    12,    13,    14,    15,    43,    82,    99,
     171,   105,     1,    91,   175,    81,   102,    80,    59,    87,
     182,    16,    37,    38,    62,    63,    64,    65,    66,    67,
      68,    69,    70,    78,   102,   112,   114,   115,   116,   117,
     118,   119,   122,    37,   128,   128,    91,   128,   114,   167,
      91,   132,   132,   132,   132,    91,   136,   149,    91,   129,
     101,    58,   168,    84,   105,    11,    12,    13,    14,    15,
      16,   151,   152,   153,   154,   155,    99,   100,     1,    21,
      23,    24,    25,    26,    27,    28,    31,    32,    33,    34,
      35,    36,    52,    53,    75,    76,   178,   179,    38,   115,
     116,   117,   120,   121,   174,   105,    80,    81,    87,    91,
     182,    82,   185,   115,   119,    63,    67,    62,    63,    64,
      65,   102,    84,   111,    86,    86,    86,    38,    87,    89,
      90,   102,   102,   102,    71,   102,    83,    30,    52,   133,
     138,   101,   113,   113,   113,   113,    52,    57,   114,   135,
     137,    91,   149,    91,   136,    41,    42,   130,   131,   113,
      18,    78,   118,   122,   158,   159,    82,   132,   132,   132,
     132,   149,   129,    58,   134,    92,    83,    87,   120,    38,
      89,    90,   114,    82,    15,    56,   182,    59,   181,   182,
      88,    99,    86,    62,    63,   111,    57,    58,   108,   109,
     110,   122,    86,    89,    90,    90,    91,   124,   125,   210,
      87,    84,    87,    92,    84,    87,   167,    92,    83,   111,
      80,   144,   144,   144,   144,   101,    92,    83,    92,   113,
     113,    92,    83,    81,    90,   101,    56,    90,    93,   157,
     101,    83,    85,   100,   101,   101,   101,   101,   101,   101,
     178,   101,   176,   177,    89,    90,    90,   101,    82,    83,
      88,    92,   182,   102,    83,    85,   101,    90,    90,    90,
     125,   123,   182,   128,   109,   128,   128,   109,   128,   133,
     114,   145,    81,    99,   162,   162,   162,   162,    92,   137,
     144,   144,   130,    17,    82,   139,   141,   142,    90,    93,
     156,   156,    90,    57,    58,   114,   157,   159,   144,   144,
     144,   144,   144,    81,    99,    80,    83,    88,    90,    90,
      90,   111,   182,   181,   182,   182,   125,   109,    88,    90,
      90,    92,   211,    88,    85,    88,   102,    85,    88,    83,
       1,    43,   160,   163,   164,   169,   170,   172,   162,   162,
     122,   142,    82,    90,   122,    90,   162,   162,   162,   162,
     162,   142,    57,   177,    83,    90,    90,    87,   193,    87,
      90,    87,    87,    87,   145,    91,   175,   172,    82,    99,
     163,   101,   101,    56,    82,   178,    90,    39,    40,    88,
     122,   183,   186,   191,   191,   128,   128,   128,    71,    56,
     101,   174,   100,    91,   140,   156,   156,    99,   122,   183,
     183,   101,   182,    83,    88,    88,    88,    88,    88,    92,
     193,   101,    92,    99,   101,   182,   182,    91,    93,   139,
      92,   191,    37,     1,    44,    45,    46,    47,    48,    49,
      50,    77,    81,    99,   184,   196,   201,   203,   193,    92,
      92,    57,    58,   102,   180,   101,    87,   207,    91,   207,
      56,   208,   209,    81,    58,   200,   207,    81,   197,   203,
     182,    20,   195,   193,   182,   205,    56,   205,   193,   210,
      83,    81,   203,   198,   203,   184,   205,    44,    45,    46,
      48,    49,    50,    77,   184,   199,   201,   202,    82,   197,
     185,    93,   196,    91,   194,    79,    92,    88,   206,   182,
     209,    82,   197,    82,   197,   182,   206,   207,    91,   207,
      81,   200,   207,    81,   182,    82,   199,   100,   100,    57,
       6,    72,    73,    92,   122,   187,   190,   192,   184,   205,
     207,    81,   203,   211,    82,   185,    81,   203,   182,    56,
     182,   198,   184,   182,   199,   185,   101,    80,    83,    92,
     182,    79,   205,   197,   193,   100,   197,    51,   204,    79,
      92,   206,    82,   182,   206,    82,   100,    84,   122,   183,
     189,   192,   185,   205,    80,    82,    82,    81,   203,   182,
     207,    81,   203,   185,    81,   203,   101,   188,   101,   182,
      84,   101,   206,   205,   204,   197,    79,   182,   197,   100,
     197,   204,    85,    87,    90,    91,    94,    84,    92,   188,
      99,    81,   203,    83,    82,   182,    80,    82,    82,   188,
     101,    57,   188,    85,   188,    85,   197,   205,   206,   182,
     204,    88,    92,    92,   101,    85,    82,   206,    81,   203,
      83,    81,   203,   197,   182,   197,    82,   206,    82,    81,
     203,   197,    82
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    95,    96,    97,    97,    98,    98,    99,    99,   100,
     100,   101,   101,   101,   101,   101,   101,   101,   101,   101,
     101,   101,   101,   101,   101,   101,   101,   101,   101,   101,
     101,   101,   101,   101,   101,   101,   101,   101,   101,   101,
     101,   101,   101,   101,   101,   101,   101,   101,   101,   101,
     101,   101,   101,   101,   101,   101,   101,   101,   101,   101,
     102,   102,   102,   103,   103,   104,   104,   105,   105,   106,
     106,   106,   106,   106,   107,   107,   107,   107,   107,   107,
     107,   107,   107,   107,   107,   107,   107,   107,   108,   108,
     108,   109,   109,   110,   110,   111,   111,   112,   112,   112,
     112,   112,   112,   112,   112,   112,   112,   112,   112,   112,
     112,   112,   112,   112,   113,   114,   114,   115,   115,   116,
     117,   117,   118,   119,   119,   119,   119,   119,   119,   120,
     120,   120,   120,   120,   121,   121,   121,   121,   121,   121,
     122,   122,   122,   122,   122,   122,   123,   124,   125,   125,
     126,   127,   128,   128,   129,   129,   130,   130,   131,   131,
     132,   132,   133,   133,   134,   134,   135,   136,   136,   137,
     137,   138,   138,   139,   139,   140,   140,   141,   142,   142,
     143,   143,   143,   144,   144,   145,   145,   146,   146,   147,
     148,   149,   149,   150,   150,   151,   151,   152,   153,   154,
     155,   155,   156,   156,   157,   157,   157,   157,   158,   158,
     158,   158,   158,   158,   159,   159,   160,   161,   161,   161,
     161,   161,   162,   162,   163,   163,   164,   164,   164,   164,
     164,   164,   164,   165,   165,   165,   165,   165,   166,   166,
     166,   166,   167,   167,   168,   169,   170,   170,   170,   170,
     171,   171,   171,   171,   171,   171,   171,   171,   171,   171,
     171,   172,   172,   172,   173,   173,   174,   175,   175,   175,
     176,   177,   177,   178,   178,   178,   178,   179,   179,   179,
     179,   179,   179,   179,   179,   179,   179,   179,   179,   179,
     179,   179,   179,   179,   179,   180,   180,   180,   181,   181,
     181,   182,   182,   182,   182,   182,   182,   183,   184,   185,
     186,   186,   186,   186,   186,   186,   187,   187,   187,   188,
     188,   188,   188,   188,   188,   189,   190,   190,   190,   191,
     191,   192,   192,   193,   193,   194,   194,   195,   195,   196,
     196,   196,   197,   197,   198,   198,   199,   199,   199,   200,
     200,   201,   201,   201,   202,   202,   202,   202,   202,   202,
     202,   202,   202,   202,   202,   202,   203,   203,   203,   203,
     203,   203,   203,   203,   203,   203,   203,   203,   203,   203,
     204,   204,   204,   205,   206,   207,   208,   208,   209,   209,
     210,   211,   212,   213
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
       2,     1,     1,     1,     2,     2,     3,     1,     1,     2,
       2,     2,     8,     1,     1,     1,     1,     2,     2,     1,
       1,     1,     2,     2,     6,     5,     4,     3,     2,     1,
       6,     5,     4,     3,     2,     1,     1,     3,     0,     2,
       4,     6,     0,     1,     0,     3,     1,     3,     1,     1,
       0,     3,     1,     3,     0,     1,     1,     0,     3,     1,
       3,     1,     1,     0,     1,     0,     2,     5,     1,     2,
       3,     5,     6,     0,     2,     1,     3,     5,     5,     5,
       5,     4,     3,     6,     6,     5,     5,     5,     5,     5,
       4,     7,     0,     2,     0,     2,     2,     2,     6,     6,
       3,     3,     2,     3,     1,     3,     4,     2,     2,     2,
       2,     2,     1,     4,     0,     2,     1,     1,     1,     1,
       2,     2,     2,     3,     6,     9,     3,     6,     3,     6,
       9,     9,     1,     3,     1,     1,     1,     2,     2,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     7,     5,    13,     5,     2,     1,     0,     3,     1,
       3,     1,     3,     1,     4,     3,     6,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       2,     1,     1,     1,     1,     1,     1,     1,     0,     1,
       3,     0,     1,     5,     5,     5,     4,     3,     1,     1,
       1,     3,     4,     3,     4,     4,     1,     1,     1,     1,
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
#line 202 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.modlist) = (yyvsp[0].modlist); modlist = (yyvsp[0].modlist); }
#line 2290 "y.tab.c" /* yacc.c:1646  */
    break;

  case 3:
#line 206 "xi-grammar.y" /* yacc.c:1646  */
    { 
		  (yyval.modlist) = 0; 
		}
#line 2298 "y.tab.c" /* yacc.c:1646  */
    break;

  case 4:
#line 210 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.modlist) = new AstChildren<Module>(lineno, (yyvsp[-1].module), (yyvsp[0].modlist)); }
#line 2304 "y.tab.c" /* yacc.c:1646  */
    break;

  case 5:
#line 214 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 2310 "y.tab.c" /* yacc.c:1646  */
    break;

  case 6:
#line 216 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 2316 "y.tab.c" /* yacc.c:1646  */
    break;

  case 7:
#line 220 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 2322 "y.tab.c" /* yacc.c:1646  */
    break;

  case 8:
#line 222 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 2; }
#line 2328 "y.tab.c" /* yacc.c:1646  */
    break;

  case 9:
#line 226 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 2334 "y.tab.c" /* yacc.c:1646  */
    break;

  case 10:
#line 228 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 2340 "y.tab.c" /* yacc.c:1646  */
    break;

  case 11:
#line 233 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 2346 "y.tab.c" /* yacc.c:1646  */
    break;

  case 12:
#line 234 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MODULE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2352 "y.tab.c" /* yacc.c:1646  */
    break;

  case 13:
#line 235 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MAINMODULE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2358 "y.tab.c" /* yacc.c:1646  */
    break;

  case 14:
#line 236 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(EXTERN, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2364 "y.tab.c" /* yacc.c:1646  */
    break;

  case 15:
#line 238 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITCALL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2370 "y.tab.c" /* yacc.c:1646  */
    break;

  case 16:
#line 239 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITNODE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2376 "y.tab.c" /* yacc.c:1646  */
    break;

  case 17:
#line 240 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITPROC, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2382 "y.tab.c" /* yacc.c:1646  */
    break;

  case 18:
#line 242 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CHARE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2388 "y.tab.c" /* yacc.c:1646  */
    break;

  case 19:
#line 243 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MAINCHARE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2394 "y.tab.c" /* yacc.c:1646  */
    break;

  case 20:
#line 244 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(GROUP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2400 "y.tab.c" /* yacc.c:1646  */
    break;

  case 21:
#line 245 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NODEGROUP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2406 "y.tab.c" /* yacc.c:1646  */
    break;

  case 22:
#line 246 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ARRAY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2412 "y.tab.c" /* yacc.c:1646  */
    break;

  case 23:
#line 250 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INCLUDE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2418 "y.tab.c" /* yacc.c:1646  */
    break;

  case 24:
#line 251 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(STACKSIZE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2424 "y.tab.c" /* yacc.c:1646  */
    break;

  case 25:
#line 252 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(THREADED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2430 "y.tab.c" /* yacc.c:1646  */
    break;

  case 26:
#line 253 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(TEMPLATE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2436 "y.tab.c" /* yacc.c:1646  */
    break;

  case 27:
#line 254 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(SYNC, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2442 "y.tab.c" /* yacc.c:1646  */
    break;

  case 28:
#line 255 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IGET, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2448 "y.tab.c" /* yacc.c:1646  */
    break;

  case 29:
#line 256 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(EXCLUSIVE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2454 "y.tab.c" /* yacc.c:1646  */
    break;

  case 30:
#line 257 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IMMEDIATE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2460 "y.tab.c" /* yacc.c:1646  */
    break;

  case 31:
#line 258 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(SKIPSCHED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2466 "y.tab.c" /* yacc.c:1646  */
    break;

  case 32:
#line 259 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOCOPY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2472 "y.tab.c" /* yacc.c:1646  */
    break;

  case 33:
#line 260 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOCOPYPOST, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2478 "y.tab.c" /* yacc.c:1646  */
    break;

  case 34:
#line 261 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INLINE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2484 "y.tab.c" /* yacc.c:1646  */
    break;

  case 35:
#line 262 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(VIRTUAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2490 "y.tab.c" /* yacc.c:1646  */
    break;

  case 36:
#line 263 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MIGRATABLE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2496 "y.tab.c" /* yacc.c:1646  */
    break;

  case 37:
#line 264 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CREATEHERE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2502 "y.tab.c" /* yacc.c:1646  */
    break;

  case 38:
#line 265 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CREATEHOME, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2508 "y.tab.c" /* yacc.c:1646  */
    break;

  case 39:
#line 266 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOKEEP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2514 "y.tab.c" /* yacc.c:1646  */
    break;

  case 40:
#line 267 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOTRACE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2520 "y.tab.c" /* yacc.c:1646  */
    break;

  case 41:
#line 268 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(APPWORK, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2526 "y.tab.c" /* yacc.c:1646  */
    break;

  case 42:
#line 271 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(PACKED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2532 "y.tab.c" /* yacc.c:1646  */
    break;

  case 43:
#line 272 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(VARSIZE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2538 "y.tab.c" /* yacc.c:1646  */
    break;

  case 44:
#line 273 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ENTRY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2544 "y.tab.c" /* yacc.c:1646  */
    break;

  case 45:
#line 274 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(FOR, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2550 "y.tab.c" /* yacc.c:1646  */
    break;

  case 46:
#line 275 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(FORALL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2556 "y.tab.c" /* yacc.c:1646  */
    break;

  case 47:
#line 276 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(WHILE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2562 "y.tab.c" /* yacc.c:1646  */
    break;

  case 48:
#line 277 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(WHEN, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2568 "y.tab.c" /* yacc.c:1646  */
    break;

  case 49:
#line 278 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(OVERLAP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2574 "y.tab.c" /* yacc.c:1646  */
    break;

  case 50:
#line 279 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(SERIAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2580 "y.tab.c" /* yacc.c:1646  */
    break;

  case 51:
#line 280 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IF, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2586 "y.tab.c" /* yacc.c:1646  */
    break;

  case 52:
#line 281 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ELSE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2592 "y.tab.c" /* yacc.c:1646  */
    break;

  case 53:
#line 283 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(LOCAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2598 "y.tab.c" /* yacc.c:1646  */
    break;

  case 54:
#line 285 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(USING, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2604 "y.tab.c" /* yacc.c:1646  */
    break;

  case 55:
#line 286 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ACCEL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2610 "y.tab.c" /* yacc.c:1646  */
    break;

  case 56:
#line 289 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ACCELBLOCK, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2616 "y.tab.c" /* yacc.c:1646  */
    break;

  case 57:
#line 290 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MEMCRITICAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2622 "y.tab.c" /* yacc.c:1646  */
    break;

  case 58:
#line 291 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(REDUCTIONTARGET, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2628 "y.tab.c" /* yacc.c:1646  */
    break;

  case 59:
#line 292 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CASE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2634 "y.tab.c" /* yacc.c:1646  */
    break;

  case 60:
#line 297 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 2640 "y.tab.c" /* yacc.c:1646  */
    break;

  case 61:
#line 299 "xi-grammar.y" /* yacc.c:1646  */
    {
		  char *tmp = new char[strlen((yyvsp[-3].strval))+strlen((yyvsp[0].strval))+3];
		  sprintf(tmp,"%s::%s", (yyvsp[-3].strval), (yyvsp[0].strval));
		  (yyval.strval) = tmp;
		}
#line 2650 "y.tab.c" /* yacc.c:1646  */
    break;

  case 62:
#line 305 "xi-grammar.y" /* yacc.c:1646  */
    {
		  char *tmp = new char[strlen((yyvsp[-3].strval))+5+3];
		  sprintf(tmp,"%s::array", (yyvsp[-3].strval));
		  (yyval.strval) = tmp;
		}
#line 2660 "y.tab.c" /* yacc.c:1646  */
    break;

  case 63:
#line 312 "xi-grammar.y" /* yacc.c:1646  */
    { 
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist)); 
		}
#line 2668 "y.tab.c" /* yacc.c:1646  */
    break;

  case 64:
#line 316 "xi-grammar.y" /* yacc.c:1646  */
    {  
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist)); 
		    (yyval.module)->setMain();
		}
#line 2677 "y.tab.c" /* yacc.c:1646  */
    break;

  case 65:
#line 323 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = 0; }
#line 2683 "y.tab.c" /* yacc.c:1646  */
    break;

  case 66:
#line 325 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = (yyvsp[-2].conslist); }
#line 2689 "y.tab.c" /* yacc.c:1646  */
    break;

  case 67:
#line 329 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = 0; }
#line 2695 "y.tab.c" /* yacc.c:1646  */
    break;

  case 68:
#line 331 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = new ConstructList(lineno, (yyvsp[-1].construct), (yyvsp[0].conslist)); }
#line 2701 "y.tab.c" /* yacc.c:1646  */
    break;

  case 69:
#line 335 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new UsingScope((yyvsp[0].strval), false); }
#line 2707 "y.tab.c" /* yacc.c:1646  */
    break;

  case 70:
#line 337 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new UsingScope((yyvsp[0].strval), true); }
#line 2713 "y.tab.c" /* yacc.c:1646  */
    break;

  case 71:
#line 339 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].member)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].member); }
#line 2719 "y.tab.c" /* yacc.c:1646  */
    break;

  case 72:
#line 341 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].message)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].message); }
#line 2725 "y.tab.c" /* yacc.c:1646  */
    break;

  case 73:
#line 343 "xi-grammar.y" /* yacc.c:1646  */
    {
                  Entry *e = new Entry(lineno, (yyvsp[-5].attr), (yyvsp[-4].type), (yyvsp[-2].strval), (yyvsp[0].plist), 0, 0, 0, (yylsp[-7]).first_line, (yyloc).last_line);
                  int isExtern = 1;
                  e->setExtern(isExtern);
                  e->targs = (yyvsp[-1].tparlist);
                  e->label = new XStr;
                  (yyvsp[-3].ntype)->print(*e->label);
                  (yyval.construct) = e;
                  firstRdma = true;
                }
#line 2740 "y.tab.c" /* yacc.c:1646  */
    break;

  case 74:
#line 356 "xi-grammar.y" /* yacc.c:1646  */
    { if((yyvsp[-2].conslist)) (yyvsp[-2].conslist)->recurse<int&>((yyvsp[-4].intval), &Construct::setExtern); (yyval.construct) = (yyvsp[-2].conslist); }
#line 2746 "y.tab.c" /* yacc.c:1646  */
    break;

  case 75:
#line 358 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new Scope((yyvsp[-3].strval), (yyvsp[-1].conslist)); }
#line 2752 "y.tab.c" /* yacc.c:1646  */
    break;

  case 76:
#line 360 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = (yyvsp[-1].construct); }
#line 2758 "y.tab.c" /* yacc.c:1646  */
    break;

  case 77:
#line 362 "xi-grammar.y" /* yacc.c:1646  */
    {
          ERROR("preceding construct must be semicolon terminated",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 2768 "y.tab.c" /* yacc.c:1646  */
    break;

  case 78:
#line 368 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].module)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].module); }
#line 2774 "y.tab.c" /* yacc.c:1646  */
    break;

  case 79:
#line 370 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2780 "y.tab.c" /* yacc.c:1646  */
    break;

  case 80:
#line 372 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2786 "y.tab.c" /* yacc.c:1646  */
    break;

  case 81:
#line 374 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2792 "y.tab.c" /* yacc.c:1646  */
    break;

  case 82:
#line 376 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2798 "y.tab.c" /* yacc.c:1646  */
    break;

  case 83:
#line 378 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].templat)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].templat); }
#line 2804 "y.tab.c" /* yacc.c:1646  */
    break;

  case 84:
#line 380 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = NULL; }
#line 2810 "y.tab.c" /* yacc.c:1646  */
    break;

  case 85:
#line 382 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = NULL; }
#line 2816 "y.tab.c" /* yacc.c:1646  */
    break;

  case 86:
#line 384 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = (yyvsp[0].accelBlock); }
#line 2822 "y.tab.c" /* yacc.c:1646  */
    break;

  case 87:
#line 386 "xi-grammar.y" /* yacc.c:1646  */
    {
          ERROR("invalid construct",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 2832 "y.tab.c" /* yacc.c:1646  */
    break;

  case 88:
#line 394 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamType((yyvsp[0].type)); }
#line 2838 "y.tab.c" /* yacc.c:1646  */
    break;

  case 89:
#line 396 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
#line 2844 "y.tab.c" /* yacc.c:1646  */
    break;

  case 90:
#line 398 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
#line 2850 "y.tab.c" /* yacc.c:1646  */
    break;

  case 91:
#line 402 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = new TParamList((yyvsp[0].tparam)); }
#line 2856 "y.tab.c" /* yacc.c:1646  */
    break;

  case 92:
#line 404 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = new TParamList((yyvsp[-2].tparam), (yyvsp[0].tparlist)); }
#line 2862 "y.tab.c" /* yacc.c:1646  */
    break;

  case 93:
#line 408 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = new TParamList(0); }
#line 2868 "y.tab.c" /* yacc.c:1646  */
    break;

  case 94:
#line 410 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = (yyvsp[0].tparlist); }
#line 2874 "y.tab.c" /* yacc.c:1646  */
    break;

  case 95:
#line 414 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = 0; }
#line 2880 "y.tab.c" /* yacc.c:1646  */
    break;

  case 96:
#line 416 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = (yyvsp[-1].tparlist); }
#line 2886 "y.tab.c" /* yacc.c:1646  */
    break;

  case 97:
#line 420 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("int"); }
#line 2892 "y.tab.c" /* yacc.c:1646  */
    break;

  case 98:
#line 422 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long"); }
#line 2898 "y.tab.c" /* yacc.c:1646  */
    break;

  case 99:
#line 424 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("short"); }
#line 2904 "y.tab.c" /* yacc.c:1646  */
    break;

  case 100:
#line 426 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("char"); }
#line 2910 "y.tab.c" /* yacc.c:1646  */
    break;

  case 101:
#line 428 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned int"); }
#line 2916 "y.tab.c" /* yacc.c:1646  */
    break;

  case 102:
#line 430 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long"); }
#line 2922 "y.tab.c" /* yacc.c:1646  */
    break;

  case 103:
#line 432 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long"); }
#line 2928 "y.tab.c" /* yacc.c:1646  */
    break;

  case 104:
#line 434 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long long"); }
#line 2934 "y.tab.c" /* yacc.c:1646  */
    break;

  case 105:
#line 436 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned short"); }
#line 2940 "y.tab.c" /* yacc.c:1646  */
    break;

  case 106:
#line 438 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned char"); }
#line 2946 "y.tab.c" /* yacc.c:1646  */
    break;

  case 107:
#line 440 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long long"); }
#line 2952 "y.tab.c" /* yacc.c:1646  */
    break;

  case 108:
#line 442 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("float"); }
#line 2958 "y.tab.c" /* yacc.c:1646  */
    break;

  case 109:
#line 444 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("double"); }
#line 2964 "y.tab.c" /* yacc.c:1646  */
    break;

  case 110:
#line 446 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long double"); }
#line 2970 "y.tab.c" /* yacc.c:1646  */
    break;

  case 111:
#line 448 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("void"); }
#line 2976 "y.tab.c" /* yacc.c:1646  */
    break;

  case 112:
#line 450 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("size_t"); }
#line 2982 "y.tab.c" /* yacc.c:1646  */
    break;

  case 113:
#line 452 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("bool"); }
#line 2988 "y.tab.c" /* yacc.c:1646  */
    break;

  case 114:
#line 455 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ntype) = new NamedType((yyvsp[-1].strval),(yyvsp[0].tparlist)); }
#line 2994 "y.tab.c" /* yacc.c:1646  */
    break;

  case 115:
#line 456 "xi-grammar.y" /* yacc.c:1646  */
    { 
                    const char* basename, *scope;
                    splitScopedName((yyvsp[-1].strval), &scope, &basename);
                    (yyval.ntype) = new NamedType(basename, (yyvsp[0].tparlist), scope);
                }
#line 3004 "y.tab.c" /* yacc.c:1646  */
    break;

  case 116:
#line 462 "xi-grammar.y" /* yacc.c:1646  */
    {
			const char* basename, *scope;
			splitScopedName((yyvsp[-1].strval), &scope, &basename);
			(yyval.ntype) = new NamedType(basename, (yyvsp[0].tparlist), scope, true);
		}
#line 3014 "y.tab.c" /* yacc.c:1646  */
    break;

  case 117:
#line 470 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3020 "y.tab.c" /* yacc.c:1646  */
    break;

  case 118:
#line 472 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ntype); }
#line 3026 "y.tab.c" /* yacc.c:1646  */
    break;

  case 119:
#line 476 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ptype) = new PtrType((yyvsp[-1].type)); }
#line 3032 "y.tab.c" /* yacc.c:1646  */
    break;

  case 120:
#line 480 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
#line 3038 "y.tab.c" /* yacc.c:1646  */
    break;

  case 121:
#line 482 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
#line 3044 "y.tab.c" /* yacc.c:1646  */
    break;

  case 122:
#line 486 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ftype) = new FuncType((yyvsp[-7].type), (yyvsp[-4].strval), (yyvsp[-1].plist)); }
#line 3050 "y.tab.c" /* yacc.c:1646  */
    break;

  case 123:
#line 490 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3056 "y.tab.c" /* yacc.c:1646  */
    break;

  case 124:
#line 492 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3062 "y.tab.c" /* yacc.c:1646  */
    break;

  case 125:
#line 494 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3068 "y.tab.c" /* yacc.c:1646  */
    break;

  case 126:
#line 496 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ftype); }
#line 3074 "y.tab.c" /* yacc.c:1646  */
    break;

  case 127:
#line 498 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[0].type)); }
#line 3080 "y.tab.c" /* yacc.c:1646  */
    break;

  case 128:
#line 500 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[-1].type)); }
#line 3086 "y.tab.c" /* yacc.c:1646  */
    break;

  case 129:
#line 504 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3092 "y.tab.c" /* yacc.c:1646  */
    break;

  case 130:
#line 506 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3098 "y.tab.c" /* yacc.c:1646  */
    break;

  case 131:
#line 508 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3104 "y.tab.c" /* yacc.c:1646  */
    break;

  case 132:
#line 510 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[0].type)); }
#line 3110 "y.tab.c" /* yacc.c:1646  */
    break;

  case 133:
#line 512 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[-1].type)); }
#line 3116 "y.tab.c" /* yacc.c:1646  */
    break;

  case 134:
#line 516 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType(new RValueReferenceType((yyvsp[-5].type))); }
#line 3122 "y.tab.c" /* yacc.c:1646  */
    break;

  case 135:
#line 518 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType(new ReferenceType((yyvsp[-4].type))); }
#line 3128 "y.tab.c" /* yacc.c:1646  */
    break;

  case 136:
#line 520 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType((yyvsp[-3].type)); }
#line 3134 "y.tab.c" /* yacc.c:1646  */
    break;

  case 137:
#line 522 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new RValueReferenceType((yyvsp[-2].type)); }
#line 3140 "y.tab.c" /* yacc.c:1646  */
    break;

  case 138:
#line 524 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ReferenceType((yyvsp[-1].type)); }
#line 3146 "y.tab.c" /* yacc.c:1646  */
    break;

  case 139:
#line 526 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3152 "y.tab.c" /* yacc.c:1646  */
    break;

  case 140:
#line 530 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType(new RValueReferenceType((yyvsp[-5].type))); }
#line 3158 "y.tab.c" /* yacc.c:1646  */
    break;

  case 141:
#line 532 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType(new ReferenceType((yyvsp[-4].type))); }
#line 3164 "y.tab.c" /* yacc.c:1646  */
    break;

  case 142:
#line 534 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType((yyvsp[-3].type)); }
#line 3170 "y.tab.c" /* yacc.c:1646  */
    break;

  case 143:
#line 536 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new RValueReferenceType((yyvsp[-2].type)); }
#line 3176 "y.tab.c" /* yacc.c:1646  */
    break;

  case 144:
#line 538 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ReferenceType((yyvsp[-1].type)); }
#line 3182 "y.tab.c" /* yacc.c:1646  */
    break;

  case 145:
#line 540 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3188 "y.tab.c" /* yacc.c:1646  */
    break;

  case 146:
#line 544 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 3194 "y.tab.c" /* yacc.c:1646  */
    break;

  case 147:
#line 548 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = (yyvsp[-1].val); }
#line 3200 "y.tab.c" /* yacc.c:1646  */
    break;

  case 148:
#line 552 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.vallist) = 0; }
#line 3206 "y.tab.c" /* yacc.c:1646  */
    break;

  case 149:
#line 554 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.vallist) = new ValueList((yyvsp[-1].val), (yyvsp[0].vallist)); }
#line 3212 "y.tab.c" /* yacc.c:1646  */
    break;

  case 150:
#line 558 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].vallist)); }
#line 3218 "y.tab.c" /* yacc.c:1646  */
    break;

  case 151:
#line 562 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[-3].type), (yyvsp[-1].strval), (yyvsp[0].vallist), 1); }
#line 3224 "y.tab.c" /* yacc.c:1646  */
    break;

  case 152:
#line 566 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0;}
#line 3230 "y.tab.c" /* yacc.c:1646  */
    break;

  case 153:
#line 568 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0;}
#line 3236 "y.tab.c" /* yacc.c:1646  */
    break;

  case 154:
#line 572 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3242 "y.tab.c" /* yacc.c:1646  */
    break;

  case 155:
#line 574 "xi-grammar.y" /* yacc.c:1646  */
    { 
		  /*
		  printf("Warning: Message attributes are being phased out.\n");
		  printf("Warning: Please remove them from interface files.\n");
		  */
		  (yyval.intval) = (yyvsp[-1].intval); 
		}
#line 3254 "y.tab.c" /* yacc.c:1646  */
    break;

  case 156:
#line 584 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[0].intval); }
#line 3260 "y.tab.c" /* yacc.c:1646  */
    break;

  case 157:
#line 586 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[-2].intval) | (yyvsp[0].intval); }
#line 3266 "y.tab.c" /* yacc.c:1646  */
    break;

  case 158:
#line 590 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3272 "y.tab.c" /* yacc.c:1646  */
    break;

  case 159:
#line 592 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3278 "y.tab.c" /* yacc.c:1646  */
    break;

  case 160:
#line 596 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = 0; }
#line 3284 "y.tab.c" /* yacc.c:1646  */
    break;

  case 161:
#line 598 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-1].cattr); }
#line 3290 "y.tab.c" /* yacc.c:1646  */
    break;

  case 162:
#line 602 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[0].cattr); }
#line 3296 "y.tab.c" /* yacc.c:1646  */
    break;

  case 163:
#line 604 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
#line 3302 "y.tab.c" /* yacc.c:1646  */
    break;

  case 164:
#line 608 "xi-grammar.y" /* yacc.c:1646  */
    { python_doc = NULL; (yyval.intval) = 0; }
#line 3308 "y.tab.c" /* yacc.c:1646  */
    break;

  case 165:
#line 610 "xi-grammar.y" /* yacc.c:1646  */
    { python_doc = (yyvsp[0].strval); (yyval.intval) = 0; }
#line 3314 "y.tab.c" /* yacc.c:1646  */
    break;

  case 166:
#line 614 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CPYTHON; }
#line 3320 "y.tab.c" /* yacc.c:1646  */
    break;

  case 167:
#line 618 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = 0; }
#line 3326 "y.tab.c" /* yacc.c:1646  */
    break;

  case 168:
#line 620 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-1].cattr); }
#line 3332 "y.tab.c" /* yacc.c:1646  */
    break;

  case 169:
#line 624 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[0].cattr); }
#line 3338 "y.tab.c" /* yacc.c:1646  */
    break;

  case 170:
#line 626 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
#line 3344 "y.tab.c" /* yacc.c:1646  */
    break;

  case 171:
#line 630 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CMIGRATABLE; }
#line 3350 "y.tab.c" /* yacc.c:1646  */
    break;

  case 172:
#line 632 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CPYTHON; }
#line 3356 "y.tab.c" /* yacc.c:1646  */
    break;

  case 173:
#line 636 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3362 "y.tab.c" /* yacc.c:1646  */
    break;

  case 174:
#line 638 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 3368 "y.tab.c" /* yacc.c:1646  */
    break;

  case 175:
#line 641 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3374 "y.tab.c" /* yacc.c:1646  */
    break;

  case 176:
#line 643 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 3380 "y.tab.c" /* yacc.c:1646  */
    break;

  case 177:
#line 646 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mv) = new MsgVar((yyvsp[-3].type), (yyvsp[-2].strval), (yyvsp[-4].intval), (yyvsp[-1].intval)); }
#line 3386 "y.tab.c" /* yacc.c:1646  */
    break;

  case 178:
#line 650 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mvlist) = new MsgVarList((yyvsp[0].mv)); }
#line 3392 "y.tab.c" /* yacc.c:1646  */
    break;

  case 179:
#line 652 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mvlist) = new MsgVarList((yyvsp[-1].mv), (yyvsp[0].mvlist)); }
#line 3398 "y.tab.c" /* yacc.c:1646  */
    break;

  case 180:
#line 656 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, (yyvsp[0].ntype)); }
#line 3404 "y.tab.c" /* yacc.c:1646  */
    break;

  case 181:
#line 658 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, (yyvsp[-2].ntype)); }
#line 3410 "y.tab.c" /* yacc.c:1646  */
    break;

  case 182:
#line 660 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, (yyvsp[-3].ntype), (yyvsp[-1].mvlist)); }
#line 3416 "y.tab.c" /* yacc.c:1646  */
    break;

  case 183:
#line 664 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = 0; }
#line 3422 "y.tab.c" /* yacc.c:1646  */
    break;

  case 184:
#line 666 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = (yyvsp[0].typelist); }
#line 3428 "y.tab.c" /* yacc.c:1646  */
    break;

  case 185:
#line 670 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = new TypeList((yyvsp[0].ntype)); }
#line 3434 "y.tab.c" /* yacc.c:1646  */
    break;

  case 186:
#line 672 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = new TypeList((yyvsp[-2].ntype), (yyvsp[0].typelist)); }
#line 3440 "y.tab.c" /* yacc.c:1646  */
    break;

  case 187:
#line 676 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr)|Chare::CCHARE, (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3446 "y.tab.c" /* yacc.c:1646  */
    break;

  case 188:
#line 678 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3452 "y.tab.c" /* yacc.c:1646  */
    break;

  case 189:
#line 682 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3458 "y.tab.c" /* yacc.c:1646  */
    break;

  case 190:
#line 686 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new NodeGroup(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3464 "y.tab.c" /* yacc.c:1646  */
    break;

  case 191:
#line 690 "xi-grammar.y" /* yacc.c:1646  */
    {/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",(yyvsp[-2].strval));
			(yyval.ntype) = new NamedType(buf); 
		}
#line 3474 "y.tab.c" /* yacc.c:1646  */
    break;

  case 192:
#line 696 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ntype) = (yyvsp[-1].ntype); }
#line 3480 "y.tab.c" /* yacc.c:1646  */
    break;

  case 193:
#line 700 "xi-grammar.y" /* yacc.c:1646  */
    {  (yyval.chare) = new Array(lineno, (yyvsp[-4].cattr), (yyvsp[-3].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3486 "y.tab.c" /* yacc.c:1646  */
    break;

  case 194:
#line 702 "xi-grammar.y" /* yacc.c:1646  */
    {  (yyval.chare) = new Array(lineno, (yyvsp[-3].cattr), (yyvsp[-4].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3492 "y.tab.c" /* yacc.c:1646  */
    break;

  case 195:
#line 706 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr)|Chare::CCHARE, new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist));}
#line 3498 "y.tab.c" /* yacc.c:1646  */
    break;

  case 196:
#line 708 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3504 "y.tab.c" /* yacc.c:1646  */
    break;

  case 197:
#line 712 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3510 "y.tab.c" /* yacc.c:1646  */
    break;

  case 198:
#line 716 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new NodeGroup( lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3516 "y.tab.c" /* yacc.c:1646  */
    break;

  case 199:
#line 720 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Array( lineno, 0, (yyvsp[-3].ntype), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3522 "y.tab.c" /* yacc.c:1646  */
    break;

  case 200:
#line 724 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-1].strval))); }
#line 3528 "y.tab.c" /* yacc.c:1646  */
    break;

  case 201:
#line 726 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-4].strval)), (yyvsp[-2].mvlist)); }
#line 3534 "y.tab.c" /* yacc.c:1646  */
    break;

  case 202:
#line 730 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = 0; }
#line 3540 "y.tab.c" /* yacc.c:1646  */
    break;

  case 203:
#line 732 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3546 "y.tab.c" /* yacc.c:1646  */
    break;

  case 204:
#line 736 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = 0; }
#line 3552 "y.tab.c" /* yacc.c:1646  */
    break;

  case 205:
#line 738 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3558 "y.tab.c" /* yacc.c:1646  */
    break;

  case 206:
#line 740 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3564 "y.tab.c" /* yacc.c:1646  */
    break;

  case 207:
#line 742 "xi-grammar.y" /* yacc.c:1646  */
    {
		  XStr typeStr;
		  (yyvsp[0].ntype)->print(typeStr);
		  char *tmp = strdup(typeStr.get_string());
		  (yyval.strval) = tmp;
		}
#line 3575 "y.tab.c" /* yacc.c:1646  */
    break;

  case 208:
#line 751 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TTypeEllipsis(new NamedEllipsisType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3581 "y.tab.c" /* yacc.c:1646  */
    break;

  case 209:
#line 753 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TTypeEllipsis(new NamedEllipsisType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3587 "y.tab.c" /* yacc.c:1646  */
    break;

  case 210:
#line 755 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TType(new NamedType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3593 "y.tab.c" /* yacc.c:1646  */
    break;

  case 211:
#line 757 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TType(new NamedType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3599 "y.tab.c" /* yacc.c:1646  */
    break;

  case 212:
#line 759 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TFunc((yyvsp[-1].ftype), (yyvsp[0].strval)); }
#line 3605 "y.tab.c" /* yacc.c:1646  */
    break;

  case 213:
#line 761 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TName((yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].strval)); }
#line 3611 "y.tab.c" /* yacc.c:1646  */
    break;

  case 214:
#line 765 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = new TVarList((yyvsp[0].tvar)); }
#line 3617 "y.tab.c" /* yacc.c:1646  */
    break;

  case 215:
#line 767 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = new TVarList((yyvsp[-2].tvar), (yyvsp[0].tvarlist)); }
#line 3623 "y.tab.c" /* yacc.c:1646  */
    break;

  case 216:
#line 771 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = (yyvsp[-1].tvarlist); }
#line 3629 "y.tab.c" /* yacc.c:1646  */
    break;

  case 217:
#line 775 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3635 "y.tab.c" /* yacc.c:1646  */
    break;

  case 218:
#line 777 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3641 "y.tab.c" /* yacc.c:1646  */
    break;

  case 219:
#line 779 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3647 "y.tab.c" /* yacc.c:1646  */
    break;

  case 220:
#line 781 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3653 "y.tab.c" /* yacc.c:1646  */
    break;

  case 221:
#line 783 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].message)); (yyvsp[0].message)->setTemplate((yyval.templat)); }
#line 3659 "y.tab.c" /* yacc.c:1646  */
    break;

  case 222:
#line 787 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = 0; }
#line 3665 "y.tab.c" /* yacc.c:1646  */
    break;

  case 223:
#line 789 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = (yyvsp[-2].mbrlist); }
#line 3671 "y.tab.c" /* yacc.c:1646  */
    break;

  case 224:
#line 793 "xi-grammar.y" /* yacc.c:1646  */
    { 
                  if (!connectEntries.empty()) {
                    (yyval.mbrlist) = new AstChildren<Member>(connectEntries);
		  } else {
		    (yyval.mbrlist) = 0; 
                  }
		}
#line 3683 "y.tab.c" /* yacc.c:1646  */
    break;

  case 225:
#line 801 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = new AstChildren<Member>(-1, (yyvsp[-1].member), (yyvsp[0].mbrlist)); }
#line 3689 "y.tab.c" /* yacc.c:1646  */
    break;

  case 226:
#line 805 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].readonly); }
#line 3695 "y.tab.c" /* yacc.c:1646  */
    break;

  case 227:
#line 807 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].readonly); }
#line 3701 "y.tab.c" /* yacc.c:1646  */
    break;

  case 229:
#line 810 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].member); }
#line 3707 "y.tab.c" /* yacc.c:1646  */
    break;

  case 230:
#line 812 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].pupable); }
#line 3713 "y.tab.c" /* yacc.c:1646  */
    break;

  case 231:
#line 814 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].includeFile); }
#line 3719 "y.tab.c" /* yacc.c:1646  */
    break;

  case 232:
#line 816 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new ClassDeclaration(lineno,(yyvsp[0].strval)); }
#line 3725 "y.tab.c" /* yacc.c:1646  */
    break;

  case 233:
#line 820 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1); }
#line 3731 "y.tab.c" /* yacc.c:1646  */
    break;

  case 234:
#line 822 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1); }
#line 3737 "y.tab.c" /* yacc.c:1646  */
    break;

  case 235:
#line 824 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[-6].strval)) + '<' +
					    ((yyvsp[-4].tparlist))->to_string() + '>').c_str()),
				    1);
		}
#line 3747 "y.tab.c" /* yacc.c:1646  */
    break;

  case 236:
#line 830 "xi-grammar.y" /* yacc.c:1646  */
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[-2]).first_column, (yylsp[-2]).last_column, (yylsp[-2]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1);
		}
#line 3757 "y.tab.c" /* yacc.c:1646  */
    break;

  case 237:
#line 836 "xi-grammar.y" /* yacc.c:1646  */
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[-5]).first_column, (yylsp[-5]).last_column, (yylsp[-5]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1);
		}
#line 3767 "y.tab.c" /* yacc.c:1646  */
    break;

  case 238:
#line 845 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 0); }
#line 3773 "y.tab.c" /* yacc.c:1646  */
    break;

  case 239:
#line 847 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 0); }
#line 3779 "y.tab.c" /* yacc.c:1646  */
    break;

  case 240:
#line 849 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[-6].strval)) + '<' +
					    ((yyvsp[-4].tparlist))->to_string() + '>').c_str()),
				    0);
		}
#line 3789 "y.tab.c" /* yacc.c:1646  */
    break;

  case 241:
#line 855 "xi-grammar.y" /* yacc.c:1646  */
    {
                  InitCall* rtn = new InitCall(lineno, (yyvsp[-3].strval), 0);
                  rtn->setAccel();
                  (yyval.member) = rtn;
		}
#line 3799 "y.tab.c" /* yacc.c:1646  */
    break;

  case 242:
#line 863 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[0].ntype),0); }
#line 3805 "y.tab.c" /* yacc.c:1646  */
    break;

  case 243:
#line 865 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[-2].ntype),(yyvsp[0].pupable)); }
#line 3811 "y.tab.c" /* yacc.c:1646  */
    break;

  case 244:
#line 868 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.includeFile) = new IncludeFile(lineno,(yyvsp[0].strval)); }
#line 3817 "y.tab.c" /* yacc.c:1646  */
    break;

  case 245:
#line 872 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].member); }
#line 3823 "y.tab.c" /* yacc.c:1646  */
    break;

  case 246:
#line 876 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].entry); }
#line 3829 "y.tab.c" /* yacc.c:1646  */
    break;

  case 247:
#line 878 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyvsp[0].entry)->tspec = (yyvsp[-1].tvarlist);
                  (yyval.member) = (yyvsp[0].entry);
                }
#line 3838 "y.tab.c" /* yacc.c:1646  */
    break;

  case 248:
#line 883 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[-1].member); }
#line 3844 "y.tab.c" /* yacc.c:1646  */
    break;

  case 249:
#line 885 "xi-grammar.y" /* yacc.c:1646  */
    {
          ERROR("invalid SDAG member",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 3854 "y.tab.c" /* yacc.c:1646  */
    break;

  case 250:
#line 893 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3860 "y.tab.c" /* yacc.c:1646  */
    break;

  case 251:
#line 895 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3866 "y.tab.c" /* yacc.c:1646  */
    break;

  case 252:
#line 897 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3872 "y.tab.c" /* yacc.c:1646  */
    break;

  case 253:
#line 899 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3878 "y.tab.c" /* yacc.c:1646  */
    break;

  case 254:
#line 901 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3884 "y.tab.c" /* yacc.c:1646  */
    break;

  case 255:
#line 903 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3890 "y.tab.c" /* yacc.c:1646  */
    break;

  case 256:
#line 905 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3896 "y.tab.c" /* yacc.c:1646  */
    break;

  case 257:
#line 907 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3902 "y.tab.c" /* yacc.c:1646  */
    break;

  case 258:
#line 909 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3908 "y.tab.c" /* yacc.c:1646  */
    break;

  case 259:
#line 911 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3914 "y.tab.c" /* yacc.c:1646  */
    break;

  case 260:
#line 913 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3920 "y.tab.c" /* yacc.c:1646  */
    break;

  case 261:
#line 916 "xi-grammar.y" /* yacc.c:1646  */
    { 
                  (yyval.entry) = new Entry(lineno, (yyvsp[-5].attr), (yyvsp[-4].type), (yyvsp[-3].strval), (yyvsp[-2].plist), (yyvsp[-1].val), (yyvsp[0].sentry), (const char *) NULL, (yylsp[-6]).first_line, (yyloc).last_line);
		  if ((yyvsp[0].sentry) != 0) { 
		    (yyvsp[0].sentry)->con1 = new SdagConstruct(SIDENT, (yyvsp[-3].strval));
                    (yyvsp[0].sentry)->setEntry((yyval.entry));
                    (yyvsp[0].sentry)->param = new ParamList((yyvsp[-2].plist));
                  }
                  firstRdma = true;
		}
#line 3934 "y.tab.c" /* yacc.c:1646  */
    break;

  case 262:
#line 926 "xi-grammar.y" /* yacc.c:1646  */
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
#line 3955 "y.tab.c" /* yacc.c:1646  */
    break;

  case 263:
#line 943 "xi-grammar.y" /* yacc.c:1646  */
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
#line 3974 "y.tab.c" /* yacc.c:1646  */
    break;

  case 264:
#line 960 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.accelBlock) = new AccelBlock(lineno, new XStr((yyvsp[-2].strval))); }
#line 3980 "y.tab.c" /* yacc.c:1646  */
    break;

  case 265:
#line 962 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.accelBlock) = new AccelBlock(lineno, NULL); }
#line 3986 "y.tab.c" /* yacc.c:1646  */
    break;

  case 266:
#line 966 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3992 "y.tab.c" /* yacc.c:1646  */
    break;

  case 267:
#line 970 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attr) = 0; }
#line 3998 "y.tab.c" /* yacc.c:1646  */
    break;

  case 268:
#line 972 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attr) = (yyvsp[-1].attr); }
#line 4004 "y.tab.c" /* yacc.c:1646  */
    break;

  case 269:
#line 974 "xi-grammar.y" /* yacc.c:1646  */
    { ERROR("invalid entry method attribute list",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
#line 4013 "y.tab.c" /* yacc.c:1646  */
    break;

  case 270:
#line 980 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attrarg) = new Attribute::Argument((yyvsp[-2].strval), atoi((yyvsp[0].strval))); }
#line 4019 "y.tab.c" /* yacc.c:1646  */
    break;

  case 271:
#line 984 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attrarg) = (yyvsp[0].attrarg); }
#line 4025 "y.tab.c" /* yacc.c:1646  */
    break;

  case 272:
#line 985 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attrarg) = (yyvsp[-2].attrarg); (yyvsp[-2].attrarg)->next = (yyvsp[0].attrarg); }
#line 4031 "y.tab.c" /* yacc.c:1646  */
    break;

  case 273:
#line 989 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attr) = new Attribute((yyvsp[0].intval));           }
#line 4037 "y.tab.c" /* yacc.c:1646  */
    break;

  case 274:
#line 990 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attr) = new Attribute((yyvsp[-3].intval), (yyvsp[-1].attrarg));       }
#line 4043 "y.tab.c" /* yacc.c:1646  */
    break;

  case 275:
#line 991 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attr) = new Attribute((yyvsp[-2].intval), NULL, (yyvsp[0].attr)); }
#line 4049 "y.tab.c" /* yacc.c:1646  */
    break;

  case 276:
#line 992 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attr) = new Attribute((yyvsp[-5].intval), (yyvsp[-3].attrarg), (yyvsp[0].attr));   }
#line 4055 "y.tab.c" /* yacc.c:1646  */
    break;

  case 277:
#line 996 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = STHREADED; }
#line 4061 "y.tab.c" /* yacc.c:1646  */
    break;

  case 278:
#line 998 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SSYNC; }
#line 4067 "y.tab.c" /* yacc.c:1646  */
    break;

  case 279:
#line 1000 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SIGET; }
#line 4073 "y.tab.c" /* yacc.c:1646  */
    break;

  case 280:
#line 1002 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SLOCKED; }
#line 4079 "y.tab.c" /* yacc.c:1646  */
    break;

  case 281:
#line 1004 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SCREATEHERE; }
#line 4085 "y.tab.c" /* yacc.c:1646  */
    break;

  case 282:
#line 1006 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SCREATEHOME; }
#line 4091 "y.tab.c" /* yacc.c:1646  */
    break;

  case 283:
#line 1008 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SNOKEEP; }
#line 4097 "y.tab.c" /* yacc.c:1646  */
    break;

  case 284:
#line 1010 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SNOTRACE; }
#line 4103 "y.tab.c" /* yacc.c:1646  */
    break;

  case 285:
#line 1012 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SAPPWORK; }
#line 4109 "y.tab.c" /* yacc.c:1646  */
    break;

  case 286:
#line 1014 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SIMMEDIATE; }
#line 4115 "y.tab.c" /* yacc.c:1646  */
    break;

  case 287:
#line 1016 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SSKIPSCHED; }
#line 4121 "y.tab.c" /* yacc.c:1646  */
    break;

  case 288:
#line 1018 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SINLINE; }
#line 4127 "y.tab.c" /* yacc.c:1646  */
    break;

  case 289:
#line 1020 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SLOCAL; }
#line 4133 "y.tab.c" /* yacc.c:1646  */
    break;

  case 290:
#line 1022 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SPYTHON; }
#line 4139 "y.tab.c" /* yacc.c:1646  */
    break;

  case 291:
#line 1024 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SMEM; }
#line 4145 "y.tab.c" /* yacc.c:1646  */
    break;

  case 292:
#line 1026 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SREDUCE; }
#line 4151 "y.tab.c" /* yacc.c:1646  */
    break;

  case 293:
#line 1028 "xi-grammar.y" /* yacc.c:1646  */
    {
        (yyval.intval) = SAGGREGATE;
    }
#line 4159 "y.tab.c" /* yacc.c:1646  */
    break;

  case 294:
#line 1032 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("invalid entry method attribute",
		        (yylsp[0]).first_column, (yylsp[0]).last_column);
		  yyclearin;
		  yyerrok;
		}
#line 4170 "y.tab.c" /* yacc.c:1646  */
    break;

  case 295:
#line 1041 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4176 "y.tab.c" /* yacc.c:1646  */
    break;

  case 296:
#line 1043 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4182 "y.tab.c" /* yacc.c:1646  */
    break;

  case 297:
#line 1045 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4188 "y.tab.c" /* yacc.c:1646  */
    break;

  case 298:
#line 1049 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = ""; }
#line 4194 "y.tab.c" /* yacc.c:1646  */
    break;

  case 299:
#line 1051 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4200 "y.tab.c" /* yacc.c:1646  */
    break;

  case 300:
#line 1053 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s, %s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4210 "y.tab.c" /* yacc.c:1646  */
    break;

  case 301:
#line 1061 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = ""; }
#line 4216 "y.tab.c" /* yacc.c:1646  */
    break;

  case 302:
#line 1063 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4222 "y.tab.c" /* yacc.c:1646  */
    break;

  case 303:
#line 1065 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s[%s]%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4232 "y.tab.c" /* yacc.c:1646  */
    break;

  case 304:
#line 1071 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s{%s}%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4242 "y.tab.c" /* yacc.c:1646  */
    break;

  case 305:
#line 1077 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s(%s)%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4252 "y.tab.c" /* yacc.c:1646  */
    break;

  case 306:
#line 1083 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"(%s)%s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4262 "y.tab.c" /* yacc.c:1646  */
    break;

  case 307:
#line 1091 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval));
		}
#line 4271 "y.tab.c" /* yacc.c:1646  */
    break;

  case 308:
#line 1098 "xi-grammar.y" /* yacc.c:1646  */
    { 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			(yyval.intval) = 0;
		}
#line 4281 "y.tab.c" /* yacc.c:1646  */
    break;

  case 309:
#line 1106 "xi-grammar.y" /* yacc.c:1646  */
    { 
			in_braces=0;
			(yyval.intval) = 0;
		}
#line 4290 "y.tab.c" /* yacc.c:1646  */
    break;

  case 310:
#line 1113 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[0].type));}
#line 4296 "y.tab.c" /* yacc.c:1646  */
    break;

  case 311:
#line 1115 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval)); (yyval.pname)->setConditional((yyvsp[0].intval)); }
#line 4302 "y.tab.c" /* yacc.c:1646  */
    break;

  case 312:
#line 1117 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[-3].type),(yyvsp[-2].strval),0,(yyvsp[0].val));}
#line 4308 "y.tab.c" /* yacc.c:1646  */
    break;

  case 313:
#line 1119 "xi-grammar.y" /* yacc.c:1646  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
		}
#line 4317 "y.tab.c" /* yacc.c:1646  */
    break;

  case 314:
#line 1124 "xi-grammar.y" /* yacc.c:1646  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
			(yyval.pname)->setRdma(CMK_ZC_P2P_SEND_MSG);
			if(firstRdma) {
				(yyval.pname)->setFirstRdma(true);
				firstRdma = false;
			}
		}
#line 4331 "y.tab.c" /* yacc.c:1646  */
    break;

  case 315:
#line 1134 "xi-grammar.y" /* yacc.c:1646  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
			(yyval.pname)->setRdma(CMK_ZC_P2P_RECV_MSG);
			if(firstRdma) {
				(yyval.pname)->setFirstRdma(true);
				firstRdma = false;
			}
		}
#line 4345 "y.tab.c" /* yacc.c:1646  */
    break;

  case 316:
#line 1145 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READONLY; }
#line 4351 "y.tab.c" /* yacc.c:1646  */
    break;

  case 317:
#line 1146 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READWRITE; }
#line 4357 "y.tab.c" /* yacc.c:1646  */
    break;

  case 318:
#line 1147 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_WRITEONLY; }
#line 4363 "y.tab.c" /* yacc.c:1646  */
    break;

  case 319:
#line 1150 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr((yyvsp[0].strval)); }
#line 4369 "y.tab.c" /* yacc.c:1646  */
    break;

  case 320:
#line 1151 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "->" << (yyvsp[0].strval); }
#line 4375 "y.tab.c" /* yacc.c:1646  */
    break;

  case 321:
#line 1152 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[-2].xstrptr)) << "." << (yyvsp[0].strval); }
#line 4381 "y.tab.c" /* yacc.c:1646  */
    break;

  case 322:
#line 1154 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << *((yyvsp[-1].xstrptr)) << "]";
                  delete (yyvsp[-3].xstrptr);
                  delete (yyvsp[-1].xstrptr);
                }
#line 4392 "y.tab.c" /* yacc.c:1646  */
    break;

  case 323:
#line 1161 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << (yyvsp[-1].strval) << "]";
                  delete (yyvsp[-3].xstrptr);
                }
#line 4402 "y.tab.c" /* yacc.c:1646  */
    break;

  case 324:
#line 1167 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "(" << *((yyvsp[-1].xstrptr)) << ")";
                  delete (yyvsp[-3].xstrptr);
                  delete (yyvsp[-1].xstrptr);
                }
#line 4413 "y.tab.c" /* yacc.c:1646  */
    break;

  case 325:
#line 1176 "xi-grammar.y" /* yacc.c:1646  */
    {
                  in_bracket = 0;
                  (yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName(), (yyvsp[-1].strval));
                }
#line 4422 "y.tab.c" /* yacc.c:1646  */
    break;

  case 326:
#line 1183 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[-6].intval));
                }
#line 4432 "y.tab.c" /* yacc.c:1646  */
    break;

  case 327:
#line 1189 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType(Parameter::ACCEL_BUFFER_TYPE_READWRITE);
		}
#line 4442 "y.tab.c" /* yacc.c:1646  */
    break;

  case 328:
#line 1195 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.pname) = (yyvsp[-3].pname);
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[-5].intval));
		}
#line 4452 "y.tab.c" /* yacc.c:1646  */
    break;

  case 329:
#line 1203 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 4458 "y.tab.c" /* yacc.c:1646  */
    break;

  case 330:
#line 1205 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 4464 "y.tab.c" /* yacc.c:1646  */
    break;

  case 331:
#line 1209 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 4470 "y.tab.c" /* yacc.c:1646  */
    break;

  case 332:
#line 1211 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 4476 "y.tab.c" /* yacc.c:1646  */
    break;

  case 333:
#line 1215 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = (yyvsp[-1].plist); }
#line 4482 "y.tab.c" /* yacc.c:1646  */
    break;

  case 334:
#line 1217 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList(new Parameter(0, new BuiltinType("void"))); }
#line 4488 "y.tab.c" /* yacc.c:1646  */
    break;

  case 335:
#line 1221 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = (yyvsp[-1].plist); }
#line 4494 "y.tab.c" /* yacc.c:1646  */
    break;

  case 336:
#line 1223 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = 0; }
#line 4500 "y.tab.c" /* yacc.c:1646  */
    break;

  case 337:
#line 1227 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = 0; }
#line 4506 "y.tab.c" /* yacc.c:1646  */
    break;

  case 338:
#line 1229 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4512 "y.tab.c" /* yacc.c:1646  */
    break;

  case 339:
#line 1233 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = 0; }
#line 4518 "y.tab.c" /* yacc.c:1646  */
    break;

  case 340:
#line 1235 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[0].sc)); }
#line 4524 "y.tab.c" /* yacc.c:1646  */
    break;

  case 341:
#line 1237 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[-2].slist)); }
#line 4530 "y.tab.c" /* yacc.c:1646  */
    break;

  case 342:
#line 1241 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.slist) = new SListConstruct((yyvsp[0].sc)); }
#line 4536 "y.tab.c" /* yacc.c:1646  */
    break;

  case 343:
#line 1243 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.slist) = new SListConstruct((yyvsp[-1].sc), (yyvsp[0].slist));  }
#line 4542 "y.tab.c" /* yacc.c:1646  */
    break;

  case 344:
#line 1247 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.olist) = new OListConstruct((yyvsp[0].sc)); }
#line 4548 "y.tab.c" /* yacc.c:1646  */
    break;

  case 345:
#line 1249 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.olist) = new OListConstruct((yyvsp[-1].sc), (yyvsp[0].slist)); }
#line 4554 "y.tab.c" /* yacc.c:1646  */
    break;

  case 346:
#line 1253 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.clist) = new CaseListConstruct((yyvsp[0].when)); }
#line 4560 "y.tab.c" /* yacc.c:1646  */
    break;

  case 347:
#line 1255 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.clist) = new CaseListConstruct((yyvsp[-1].when), (yyvsp[0].clist)); }
#line 4566 "y.tab.c" /* yacc.c:1646  */
    break;

  case 348:
#line 1257 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("case blocks can only contain when clauses",
		        (yylsp[0]).first_column, (yylsp[0]).last_column);
		  (yyval.clist) = 0;
		}
#line 4576 "y.tab.c" /* yacc.c:1646  */
    break;

  case 349:
#line 1265 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4582 "y.tab.c" /* yacc.c:1646  */
    break;

  case 350:
#line 1267 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = 0; }
#line 4588 "y.tab.c" /* yacc.c:1646  */
    break;

  case 351:
#line 1271 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-2].entrylist), 0); }
#line 4594 "y.tab.c" /* yacc.c:1646  */
    break;

  case 352:
#line 1273 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-1].entrylist), (yyvsp[0].sc)); }
#line 4600 "y.tab.c" /* yacc.c:1646  */
    break;

  case 353:
#line 1275 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-3].entrylist), (yyvsp[-1].slist)); }
#line 4606 "y.tab.c" /* yacc.c:1646  */
    break;

  case 354:
#line 1279 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4612 "y.tab.c" /* yacc.c:1646  */
    break;

  case 355:
#line 1281 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4618 "y.tab.c" /* yacc.c:1646  */
    break;

  case 356:
#line 1283 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4624 "y.tab.c" /* yacc.c:1646  */
    break;

  case 357:
#line 1285 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4630 "y.tab.c" /* yacc.c:1646  */
    break;

  case 358:
#line 1287 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4636 "y.tab.c" /* yacc.c:1646  */
    break;

  case 359:
#line 1289 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4642 "y.tab.c" /* yacc.c:1646  */
    break;

  case 360:
#line 1291 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4648 "y.tab.c" /* yacc.c:1646  */
    break;

  case 361:
#line 1293 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4654 "y.tab.c" /* yacc.c:1646  */
    break;

  case 362:
#line 1295 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4660 "y.tab.c" /* yacc.c:1646  */
    break;

  case 363:
#line 1297 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4666 "y.tab.c" /* yacc.c:1646  */
    break;

  case 364:
#line 1299 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4672 "y.tab.c" /* yacc.c:1646  */
    break;

  case 365:
#line 1301 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4678 "y.tab.c" /* yacc.c:1646  */
    break;

  case 366:
#line 1305 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new SerialConstruct((yyvsp[-2].strval), (yyvsp[-4].strval), (yylsp[-3]).first_line); }
#line 4684 "y.tab.c" /* yacc.c:1646  */
    break;

  case 367:
#line 1307 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new OverlapConstruct((yyvsp[-1].olist)); }
#line 4690 "y.tab.c" /* yacc.c:1646  */
    break;

  case 368:
#line 1309 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = (yyvsp[0].when); }
#line 4696 "y.tab.c" /* yacc.c:1646  */
    break;

  case 369:
#line 1311 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new CaseConstruct((yyvsp[-1].clist)); }
#line 4702 "y.tab.c" /* yacc.c:1646  */
    break;

  case 370:
#line 1313 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForConstruct((yyvsp[-8].intexpr), (yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4708 "y.tab.c" /* yacc.c:1646  */
    break;

  case 371:
#line 1315 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForConstruct((yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4714 "y.tab.c" /* yacc.c:1646  */
    break;

  case 372:
#line 1317 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-9].strval)), (yyvsp[-6].intexpr),
		             (yyvsp[-4].intexpr), (yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4721 "y.tab.c" /* yacc.c:1646  */
    break;

  case 373:
#line 1320 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-11].strval)), (yyvsp[-8].intexpr),
		             (yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4728 "y.tab.c" /* yacc.c:1646  */
    break;

  case 374:
#line 1323 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new IfConstruct((yyvsp[-3].intexpr), (yyvsp[-1].sc), (yyvsp[0].sc)); }
#line 4734 "y.tab.c" /* yacc.c:1646  */
    break;

  case 375:
#line 1325 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new IfConstruct((yyvsp[-5].intexpr), (yyvsp[-2].slist), (yyvsp[0].sc)); }
#line 4740 "y.tab.c" /* yacc.c:1646  */
    break;

  case 376:
#line 1327 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new WhileConstruct((yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4746 "y.tab.c" /* yacc.c:1646  */
    break;

  case 377:
#line 1329 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new WhileConstruct((yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4752 "y.tab.c" /* yacc.c:1646  */
    break;

  case 378:
#line 1331 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new SerialConstruct((yyvsp[-2].strval), NULL, (yyloc).first_line); }
#line 4758 "y.tab.c" /* yacc.c:1646  */
    break;

  case 379:
#line 1333 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("unknown SDAG construct or malformed entry method declaration.\n"
		        "You may have forgotten to terminate a previous entry method declaration with a"
		        " semicolon or forgotten to mark a block of sequential SDAG code as 'serial'",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
#line 4770 "y.tab.c" /* yacc.c:1646  */
    break;

  case 380:
#line 1343 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = 0; }
#line 4776 "y.tab.c" /* yacc.c:1646  */
    break;

  case 381:
#line 1345 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ElseConstruct((yyvsp[0].sc)); }
#line 4782 "y.tab.c" /* yacc.c:1646  */
    break;

  case 382:
#line 1347 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ElseConstruct((yyvsp[-1].slist)); }
#line 4788 "y.tab.c" /* yacc.c:1646  */
    break;

  case 383:
#line 1351 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intexpr) = new IntExprConstruct((yyvsp[0].strval)); }
#line 4794 "y.tab.c" /* yacc.c:1646  */
    break;

  case 384:
#line 1355 "xi-grammar.y" /* yacc.c:1646  */
    { in_int_expr = 0; (yyval.intval) = 0; }
#line 4800 "y.tab.c" /* yacc.c:1646  */
    break;

  case 385:
#line 1359 "xi-grammar.y" /* yacc.c:1646  */
    { in_int_expr = 1; (yyval.intval) = 0; }
#line 4806 "y.tab.c" /* yacc.c:1646  */
    break;

  case 386:
#line 1363 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.entry) = new Entry(lineno, NULL, 0, (yyvsp[-1].strval), (yyvsp[0].plist), 0, 0, 0, (yyloc).first_line, (yyloc).last_line);
		  firstRdma = true;
		}
#line 4815 "y.tab.c" /* yacc.c:1646  */
    break;

  case 387:
#line 1368 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.entry) = new Entry(lineno, NULL, 0, (yyvsp[-4].strval), (yyvsp[0].plist), 0, 0, (yyvsp[-2].strval), (yyloc).first_line, (yyloc).last_line);
		  firstRdma = true;
		}
#line 4824 "y.tab.c" /* yacc.c:1646  */
    break;

  case 388:
#line 1375 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.entrylist) = new EntryList((yyvsp[0].entry)); }
#line 4830 "y.tab.c" /* yacc.c:1646  */
    break;

  case 389:
#line 1377 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.entrylist) = new EntryList((yyvsp[-2].entry),(yyvsp[0].entrylist)); }
#line 4836 "y.tab.c" /* yacc.c:1646  */
    break;

  case 390:
#line 1381 "xi-grammar.y" /* yacc.c:1646  */
    { in_bracket=1; }
#line 4842 "y.tab.c" /* yacc.c:1646  */
    break;

  case 391:
#line 1384 "xi-grammar.y" /* yacc.c:1646  */
    { in_bracket=0; }
#line 4848 "y.tab.c" /* yacc.c:1646  */
    break;

  case 392:
#line 1388 "xi-grammar.y" /* yacc.c:1646  */
    { if (!macroDefined((yyvsp[0].strval), 1)) in_comment = 1; }
#line 4854 "y.tab.c" /* yacc.c:1646  */
    break;

  case 393:
#line 1392 "xi-grammar.y" /* yacc.c:1646  */
    { if (!macroDefined((yyvsp[0].strval), 0)) in_comment = 1; }
#line 4860 "y.tab.c" /* yacc.c:1646  */
    break;


#line 4864 "y.tab.c" /* yacc.c:1646  */
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
#line 1395 "xi-grammar.y" /* yacc.c:1906  */


void yyerror(const char *s) 
{
	fprintf(stderr, "[PARSE-ERROR] Unexpected/missing token at line %d. Current token being parsed: '%s'.\n", lineno, yytext);
}
