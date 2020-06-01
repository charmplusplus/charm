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
    WHENIDLE = 278,
    SYNC = 279,
    IGET = 280,
    EXCLUSIVE = 281,
    IMMEDIATE = 282,
    SKIPSCHED = 283,
    INLINE = 284,
    VIRTUAL = 285,
    MIGRATABLE = 286,
    AGGREGATE = 287,
    CREATEHERE = 288,
    CREATEHOME = 289,
    NOKEEP = 290,
    NOTRACE = 291,
    APPWORK = 292,
    VOID = 293,
    CONST = 294,
    NOCOPY = 295,
    NOCOPYPOST = 296,
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
#define WHENIDLE 278
#define SYNC 279
#define IGET 280
#define EXCLUSIVE 281
#define IMMEDIATE 282
#define SKIPSCHED 283
#define INLINE 284
#define VIRTUAL 285
#define MIGRATABLE 286
#define AGGREGATE 287
#define CREATEHERE 288
#define CREATEHOME 289
#define NOKEEP 290
#define NOTRACE 291
#define APPWORK 292
#define VOID 293
#define CONST 294
#define NOCOPY 295
#define NOCOPYPOST 296
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
#define YYLAST   1524

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  94
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  117
/* YYNRULES -- Number of rules.  */
#define YYNRULES  388
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  772

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
       0,   197,   197,   202,   205,   210,   211,   215,   217,   222,
     223,   228,   230,   231,   232,   234,   235,   236,   238,   239,
     240,   241,   242,   246,   247,   248,   249,   250,   251,   252,
     253,   254,   255,   256,   257,   258,   259,   260,   261,   262,
     263,   264,   265,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   280,   282,   283,   286,   287,   288,
     289,   293,   295,   301,   308,   312,   319,   321,   326,   327,
     331,   333,   335,   337,   339,   352,   354,   356,   358,   364,
     366,   368,   370,   372,   374,   376,   378,   380,   382,   390,
     392,   394,   398,   400,   405,   406,   411,   412,   416,   418,
     420,   422,   424,   426,   428,   430,   432,   434,   436,   438,
     440,   442,   444,   448,   449,   454,   462,   464,   468,   472,
     474,   478,   482,   484,   486,   488,   490,   492,   496,   498,
     500,   502,   504,   508,   510,   512,   514,   516,   518,   522,
     524,   526,   528,   530,   532,   536,   540,   545,   546,   550,
     554,   559,   560,   565,   566,   576,   578,   582,   584,   589,
     590,   594,   596,   601,   602,   606,   611,   612,   616,   618,
     622,   624,   629,   630,   634,   635,   638,   642,   644,   648,
     650,   652,   657,   658,   662,   664,   668,   670,   674,   678,
     682,   688,   692,   694,   698,   700,   704,   708,   712,   716,
     718,   723,   724,   729,   730,   732,   734,   743,   745,   747,
     749,   751,   753,   757,   759,   763,   767,   769,   771,   773,
     775,   779,   781,   786,   793,   797,   799,   801,   802,   804,
     806,   808,   812,   814,   816,   822,   828,   837,   839,   841,
     847,   855,   857,   860,   864,   868,   870,   875,   877,   885,
     887,   889,   891,   893,   895,   897,   899,   901,   903,   905,
     908,   918,   935,   952,   954,   958,   963,   964,   966,   973,
     975,   979,   981,   983,   985,   987,   989,   991,   993,   995,
     997,   999,  1001,  1003,  1005,  1007,  1009,  1011,  1013,  1017,
    1026,  1028,  1030,  1035,  1036,  1038,  1047,  1048,  1050,  1056,
    1062,  1068,  1076,  1083,  1091,  1098,  1100,  1102,  1104,  1109,
    1119,  1131,  1132,  1133,  1136,  1137,  1138,  1139,  1146,  1152,
    1161,  1168,  1174,  1180,  1188,  1190,  1194,  1196,  1200,  1202,
    1206,  1208,  1213,  1214,  1218,  1220,  1222,  1226,  1228,  1232,
    1234,  1238,  1240,  1242,  1250,  1253,  1256,  1258,  1260,  1264,
    1266,  1268,  1270,  1272,  1274,  1276,  1278,  1280,  1282,  1284,
    1286,  1290,  1292,  1294,  1296,  1298,  1300,  1302,  1305,  1308,
    1310,  1312,  1314,  1316,  1318,  1329,  1330,  1332,  1336,  1340,
    1344,  1348,  1353,  1360,  1362,  1366,  1369,  1373,  1377
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
  "CLASS", "INCLUDE", "STACKSIZE", "THREADED", "TEMPLATE", "WHENIDLE",
  "SYNC", "IGET", "EXCLUSIVE", "IMMEDIATE", "SKIPSCHED", "INLINE",
  "VIRTUAL", "MIGRATABLE", "AGGREGATE", "CREATEHERE", "CREATEHOME",
  "NOKEEP", "NOTRACE", "APPWORK", "VOID", "CONST", "NOCOPY", "NOCOPYPOST",
  "PACKED", "VARSIZE", "ENTRY", "FOR", "FORALL", "WHILE", "WHEN",
  "OVERLAP", "SERIAL", "IF", "ELSE", "PYTHON", "LOCAL", "NAMESPACE",
  "USING", "IDENT", "NUMBER", "LITERAL", "CPROGRAM", "HASHIF", "HASHIFDEF",
  "INT", "LONG", "SHORT", "CHAR", "FLOAT", "DOUBLE", "UNSIGNED", "ACCEL",
  "READWRITE", "WRITEONLY", "ACCELBLOCK", "MEMCRITICAL", "REDUCTIONTARGET",
  "CASE", "TYPENAME", "';'", "':'", "'{'", "'}'", "','", "'<'", "'>'",
  "'*'", "'('", "')'", "'&'", "'.'", "'['", "']'", "'='", "'-'", "$accept",
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

#define YYPACT_NINF -566

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-566)))

#define YYTABLE_NINF -340

#define yytable_value_is_error(Yytable_value) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
     221,  1298,  1298,    31,  -566,   221,  -566,  -566,  -566,  -566,
    -566,  -566,  -566,  -566,  -566,  -566,  -566,  -566,  -566,  -566,
    -566,  -566,  -566,  -566,  -566,  -566,  -566,  -566,  -566,  -566,
    -566,  -566,  -566,  -566,  -566,  -566,  -566,  -566,  -566,  -566,
    -566,  -566,  -566,  -566,  -566,  -566,  -566,  -566,  -566,  -566,
    -566,  -566,  -566,  -566,  -566,  -566,    66,    66,  -566,  -566,
    -566,   890,    50,  -566,  -566,  -566,   132,  1298,   254,  1298,
    1298,   235,  1035,    57,  1052,   890,  -566,  -566,  -566,  -566,
     239,   106,   145,  -566,   141,  -566,  -566,  -566,    50,   -19,
     283,   190,   190,     4,   -17,   144,   144,   144,   144,   147,
     160,  1298,   188,   170,   890,  -566,  -566,  -566,  -566,  -566,
    -566,  -566,  -566,   545,  -566,  -566,  -566,  -566,   178,  -566,
    -566,  -566,  -566,  -566,  -566,  -566,  -566,  -566,  -566,  -566,
      50,  -566,  -566,  -566,  1188,  1404,   890,   141,   195,   127,
     -19,   183,   426,  -566,  1419,  -566,   -12,  -566,  -566,  -566,
    -566,   260,   145,    85,  -566,  -566,   208,   216,   225,  -566,
      94,   145,  -566,   145,   145,   249,   145,   252,  -566,   110,
    1298,  1298,  1298,  1298,   146,   248,   253,   227,  1298,  -566,
    -566,  -566,  1340,   272,   144,   144,   144,   144,   248,   160,
    -566,  -566,  -566,  -566,  -566,    50,  -566,  -566,  -566,  -566,
    -566,  -566,  -566,  -566,  -566,  -566,  -566,  -566,  -566,  -566,
    -566,  -566,   300,  -566,  -566,  -566,   251,   284,  1404,   208,
     216,   225,    33,  -566,   -17,   286,   122,   -19,   308,   -19,
     299,  -566,   178,   297,    41,  -566,  -566,  -566,   217,  -566,
    -566,    85,  1043,  -566,  -566,  -566,  -566,  -566,   302,   209,
     303,    95,   157,   136,   298,   156,   -17,  -566,  -566,   305,
     309,   311,   319,   319,   319,   319,  -566,  1298,   310,   318,
     324,    96,  1298,   349,  1298,  -566,  -566,   325,   327,   337,
     803,    59,   134,  1298,   345,   335,   178,  1298,  1298,  1298,
    1298,  1298,  1298,  -566,  -566,  -566,  1188,   390,  -566,   229,
     341,  1298,  -566,  -566,  -566,   350,   351,   352,   359,   -19,
      50,   145,  -566,  -566,  -566,  -566,  -566,   360,  -566,   348,
    -566,  1298,   363,   364,   365,  -566,   367,  -566,   -19,   190,
    1043,   190,   190,  1043,   190,  -566,  -566,   110,  -566,   -17,
     250,   250,   250,   250,   369,  -566,   349,  -566,   319,   319,
    -566,   227,    29,   374,   376,   149,   377,   153,  -566,   378,
    1340,  -566,  -566,   319,   319,   319,   319,   319,   261,  -566,
     391,   392,   393,   311,   -19,   308,   -19,   -19,  -566,    95,
    1043,  -566,   354,   397,   407,  -566,  -566,   396,  -566,   412,
     421,   420,   145,   427,   430,  -566,   432,  -566,    69,    50,
    -566,  -566,  -566,  -566,  -566,  -566,   250,   250,  -566,  -566,
    -566,  1419,    34,   439,   434,  1419,  -566,  -566,   435,  -566,
    -566,  -566,  -566,  -566,   250,   250,   250,   250,   250,   508,
      50,   440,   441,  -566,   445,  -566,  -566,  -566,  -566,  -566,
    -566,   456,   454,  -566,  -566,  -566,  -566,   459,  -566,   179,
     462,  -566,   -17,  -566,   723,   506,   470,   178,    69,  -566,
    -566,  -566,  -566,  1298,  -566,  -566,  1298,  -566,   495,  -566,
    -566,  -566,  -566,  -566,   473,   480,  -566,  1372,  -566,  1387,
    -566,   190,   190,   190,  -566,  1051,  1131,  -566,   178,    50,
    -566,   472,   376,   376,   178,  -566,  1419,  1419,  -566,  1298,
     -19,   496,   490,   494,   497,   505,   507,   502,   509,   445,
    1298,  -566,   511,   178,  -566,  -566,    50,  1298,   -19,   -19,
      28,   513,  1387,  -566,  -566,  -566,  -566,  -566,   559,   525,
     445,  -566,    50,   517,   518,   520,  -566,   226,  -566,  -566,
    -566,  1298,  -566,   512,   522,   512,   542,   528,   554,   512,
     534,   375,    50,   -19,  -566,  -566,  -566,   595,  -566,  -566,
    -566,  -566,   141,  -566,   445,  -566,   -19,   562,   -19,   115,
     541,   579,   617,  -566,   551,   -19,  1444,   555,   487,   183,
     530,   525,   547,  -566,   557,   548,   553,  -566,   -19,   542,
     398,  -566,   560,   540,   -19,   553,   512,   556,   512,   552,
     554,   512,   558,   -19,   563,  1444,  -566,   178,  -566,   178,
     585,  -566,   306,   551,   -19,   512,  -566,   633,   396,  -566,
    -566,   564,  -566,  -566,   183,   653,   -19,   590,   -19,   617,
     551,   -19,  1444,   183,  -566,  -566,  -566,  -566,  -566,  -566,
    -566,  -566,  -566,  1298,   570,   568,   561,   -19,   573,   -19,
     375,  -566,   445,  -566,   178,   375,   605,   580,   569,   553,
     588,   -19,   553,   589,   178,   578,  1419,  1324,  -566,   183,
     -19,   592,   591,  -566,  -566,   593,   868,  -566,   -19,   512,
     879,  -566,   183,   886,  -566,  -566,  1298,  1298,   -19,   604,
    -566,  1298,   553,   -19,  -566,   605,   375,  -566,   597,   -19,
     375,  -566,   178,   375,   605,  -566,   295,   185,   582,  1298,
     178,   937,   594,  -566,   607,   -19,   610,   611,  -566,   613,
    -566,  -566,  1298,  1298,  1224,   612,  1298,  -566,   383,    50,
     375,  -566,   -19,  -566,   553,   -19,  -566,   605,   321,  -566,
     600,   411,  1298,   429,  -566,   614,   553,   946,   623,  -566,
    -566,  -566,  -566,  -566,  -566,  -566,   953,   375,  -566,   -19,
     375,  -566,   625,   553,   626,  -566,   960,  -566,   375,  -566,
     629,  -566
};

  /* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
     Performed when YYTABLE does not specify something else to do.  Zero
     means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       3,     0,     0,     0,     2,     3,    12,    13,    14,    15,
      16,    17,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    31,    32,    35,    36,    37,
      38,    39,    40,    41,    42,    33,    34,    43,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
      11,    56,    57,    58,    59,    60,     0,     0,     1,     4,
       7,     0,    66,    64,    65,    88,     6,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    87,    85,    86,     8,
       0,     0,     0,    61,    71,   387,   388,   303,   264,   296,
       0,   151,   151,   151,     0,   159,   159,   159,   159,     0,
     153,     0,     0,     0,     0,    79,   225,   226,    73,    80,
      81,    82,    83,     0,    84,    72,   228,   227,     9,   259,
     251,   252,   253,   254,   255,   257,   258,   256,   249,   250,
      77,    78,    69,   268,     0,     0,     0,    70,     0,   297,
     296,     0,     0,   112,     0,    98,    99,   100,   101,   109,
     110,     0,     0,    96,   116,   117,   122,   123,   124,   125,
     144,     0,   152,     0,     0,     0,     0,   241,   229,     0,
       0,     0,     0,     0,     0,     0,   166,     0,     0,   231,
     243,   230,     0,     0,   159,   159,   159,   159,     0,   153,
     216,   217,   218,   219,   220,    10,    67,   289,   271,   272,
     273,   274,   275,   281,   282,   283,   288,   276,   277,   278,
     279,   280,   163,   284,   286,   287,     0,   269,     0,   128,
     129,   130,   138,   265,     0,     0,     0,   296,   293,   296,
       0,   304,     0,     0,   126,   108,   111,   102,   103,   106,
     107,    96,    94,   114,   118,   119,   120,   127,     0,   143,
       0,   147,   235,   232,     0,   237,     0,   170,   171,     0,
     161,    96,   182,   182,   182,   182,   165,     0,     0,   168,
       0,     0,     0,     0,     0,   157,   158,     0,   155,   179,
       0,     0,   125,     0,   213,     0,     9,     0,     0,     0,
       0,     0,     0,   164,   285,   267,     0,   131,   132,   137,
       0,     0,    76,    63,    62,     0,   294,     0,     0,   296,
     263,     0,   104,   105,   115,    90,    91,    92,    95,     0,
      89,     0,   142,     0,     0,   385,   147,   149,   296,   151,
       0,   151,   151,     0,   151,   242,   160,     0,   113,     0,
       0,     0,     0,     0,     0,   191,     0,   167,   182,   182,
     154,     0,   172,     0,   201,    61,     0,     0,   211,   203,
       0,   215,    75,   182,   182,   182,   182,   182,     0,   270,
     136,     0,     0,    96,   296,   293,   296,   296,   301,   147,
       0,    97,     0,     0,     0,   141,   148,     0,   145,     0,
       0,     0,     0,     0,     0,   162,   184,   183,     0,   221,
     186,   187,   188,   189,   190,   169,     0,     0,   156,   173,
     180,     0,   172,     0,     0,     0,   209,   210,     0,   204,
     205,   206,   212,   214,     0,     0,     0,     0,     0,   172,
     199,     0,     0,   135,     0,   299,   295,   300,   298,   150,
      93,     0,     0,   140,   386,   146,   236,     0,   233,     0,
       0,   238,     0,   248,     0,     0,     0,     0,     0,   244,
     245,   192,   193,     0,   178,   181,     0,   202,     0,   194,
     195,   196,   197,   198,     0,     0,   134,     0,    74,     0,
     139,   151,   151,   151,   185,     0,     0,   246,     9,   247,
     224,   174,   201,   201,     0,   133,     0,     0,   329,   305,
     296,   324,     0,     0,     0,     0,     0,     0,    61,     0,
       0,   222,     0,     0,   207,   208,   200,     0,   296,   296,
     172,     0,     0,   328,   121,   234,   240,   239,     0,     0,
       0,   175,   176,     0,     0,     0,   302,     0,   306,   308,
     325,     0,   374,     0,     0,     0,     0,     0,   345,     0,
       0,     0,   334,   296,   261,   363,   335,   332,   309,   310,
     291,   290,   292,   307,     0,   380,   296,     0,   296,     0,
     383,     0,     0,   344,     0,   296,     0,     0,     0,     0,
       0,     0,     0,   378,     0,     0,     0,   381,   296,     0,
       0,   347,     0,     0,   296,     0,     0,     0,     0,     0,
     345,     0,     0,   296,     0,   341,   343,     9,   338,     9,
       0,   260,     0,     0,   296,     0,   379,     0,     0,   384,
     346,     0,   362,   340,     0,     0,   296,     0,   296,     0,
       0,   296,     0,     0,   364,   342,   336,   373,   333,   311,
     312,   313,   331,     0,     0,   326,     0,   296,     0,   296,
       0,   371,     0,   348,     9,     0,   375,     0,     0,     0,
       0,   296,     0,     0,     9,     0,     0,     0,   330,     0,
     296,     0,     0,   382,   361,     0,     0,   369,   296,     0,
       0,   350,     0,     0,   351,   360,     0,     0,   296,     0,
     327,     0,     0,   296,   372,   375,     0,   376,     0,   296,
       0,   358,     9,     0,   375,   314,     0,     0,     0,     0,
       0,     0,     0,   370,     0,   296,     0,     0,   349,     0,
     356,   322,     0,     0,     0,     0,     0,   320,     0,   262,
       0,   366,   296,   377,     0,   296,   359,   375,     0,   316,
       0,     0,     0,     0,   323,     0,     0,     0,     0,   357,
     319,   318,   317,   315,   321,   365,     0,     0,   353,   296,
       0,   367,     0,     0,     0,   352,     0,   368,     0,   354,
       0,   355
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -566,  -566,   706,  -566,   -54,  -274,    -1,   -60,   640,   657,
     -47,  -566,  -566,  -566,  -271,  -566,  -218,  -566,  -123,   -90,
    -121,  -125,  -122,  -173,   571,   498,  -566,   -84,  -566,  -566,
    -294,  -566,  -566,   -77,   529,   366,  -566,   -61,   382,  -566,
    -566,   544,   394,  -566,   219,  -566,  -566,  -245,  -566,  -104,
     326,  -566,  -566,  -566,    21,  -566,  -566,  -566,  -566,  -566,
    -566,  -336,   417,  -566,   395,   709,  -566,   -70,   336,   710,
    -566,  -566,   527,  -566,  -566,  -566,  -566,   329,  -566,   315,
     331,   499,  -566,  -566,   428,   -82,  -472,   -66,  -550,  -566,
    -566,  -534,  -566,  -566,  -440,   135,  -483,  -566,  -566,   223,
    -560,   176,  -561,   220,  -434,  -566,  -518,  -565,  -541,  -557,
    -484,  -566,   230,   240,   203,  -566,  -566
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     3,     4,    72,   399,   196,   261,   153,     5,    63,
      73,    74,    75,   317,   318,   319,   243,   154,   262,   155,
     156,   157,   158,   159,   160,   222,   223,   320,   387,   326,
     327,   106,   107,   163,   178,   277,   278,   170,   259,   294,
     269,   175,   270,   260,   411,   513,   412,   413,   108,   340,
     397,   109,   110,   111,   176,   112,   190,   191,   192,   193,
     194,   416,   358,   284,   285,   455,   114,   400,   456,   457,
     116,   117,   168,   181,   458,   459,   131,   460,    76,   224,
     135,   216,   217,   563,   307,   583,   500,   553,   232,   501,
     644,   706,   689,   645,   502,   646,   478,   613,   581,   554,
     577,   592,   604,   574,   555,   606,   578,   677,   584,   617,
     566,   570,   571,   328,   445,    77,    78
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[] =
{
      56,    57,    62,    62,   167,    89,   161,   141,    84,   282,
     220,   556,   362,   221,   219,   164,   166,    88,   608,   417,
     130,   233,   137,   314,   518,   519,   529,   586,   132,   609,
     621,    58,   386,   623,   595,   171,   172,   173,   625,   503,
      83,   139,   162,   338,   635,   409,   409,   557,   263,   264,
     265,   409,   235,   591,   593,   279,   236,   183,   230,   390,
     152,   568,   393,   556,   195,   575,    81,   140,    85,    86,
     453,   663,   298,   648,   654,    90,    91,    92,    93,    94,
     247,   582,   540,   664,   268,   439,   587,   101,   102,   225,
     672,   103,   241,   220,   165,   675,   221,   219,   283,   651,
     179,   251,   680,   252,   253,   683,   255,   656,   671,   440,
     410,   593,   626,   454,   628,  -177,   355,   631,   536,   691,
     537,   299,   300,   287,   288,   289,   290,   248,    79,   692,
     713,   649,   702,   247,   301,   711,   714,   303,   118,   720,
     717,   257,   605,   719,    60,   305,    61,   308,   356,   348,
    -223,   349,   712,    83,   267,   434,   514,   515,   697,   341,
     342,   343,   701,   258,   138,   704,   167,   464,   242,   673,
     745,   605,   749,   152,   138,   728,    80,   747,   310,   304,
     248,   268,   249,   250,   474,   325,   136,   282,   738,   756,
     741,   746,   743,   731,   688,   699,   272,   762,   605,   266,
     764,   477,    83,    83,   267,   325,   766,   227,   770,   291,
      83,   419,   420,   228,   511,   138,  -203,   229,  -203,   330,
     138,   241,   331,   152,     1,     2,   357,   378,   162,   758,
     152,  -201,   195,  -201,   169,   138,   138,   174,   761,   333,
     133,   415,   334,   329,   406,   407,   388,   180,   769,   396,
     177,   379,   389,   182,   391,   392,    60,   394,   138,   424,
     425,   426,   427,   428,   231,   482,   344,   421,   726,   275,
     276,   401,   402,   403,   226,   536,   283,  -266,  -266,   354,
     312,   313,   359,    83,   560,   561,   363,   364,   365,   366,
     367,   368,   435,   244,   437,   438,  -266,   322,   323,   142,
     373,   245,  -266,  -266,  -266,  -266,  -266,  -266,  -266,    82,
     246,    83,   639,    60,   430,    87,  -266,   370,   371,   254,
     382,   143,   144,   237,   238,   239,   240,   463,    60,   134,
     398,   467,   449,   636,   256,   637,   461,   462,   271,    60,
      83,   429,   295,   273,   143,   144,   145,   146,   147,   148,
     149,   150,   151,   286,   469,   470,   471,   472,   473,   293,
     152,   220,   396,    83,   221,   219,   296,   302,   306,   145,
     146,   147,   148,   149,   150,   151,   542,   640,   641,   721,
     674,   722,   311,   152,   723,   724,   309,   321,   725,   332,
     685,   337,   324,   499,   242,   499,   336,   642,   339,   542,
     346,   345,   266,   489,   504,   505,   506,   722,   750,   351,
     723,   724,   517,   517,   725,   347,   350,   352,   521,   361,
     543,   544,   545,   546,   547,   548,   549,   360,   718,   298,
     372,   374,   381,   375,   195,  -303,   534,   535,   499,   376,
     516,   441,   380,   543,   544,   545,   546,   547,   548,   549,
     377,   550,   383,   384,   385,    87,  -303,   325,  -303,   532,
     404,  -303,   491,   414,   143,   492,   418,   744,   415,   722,
     357,   579,   723,   724,   550,   552,   725,   562,    87,   620,
     431,   432,   433,    83,  -303,   509,   442,   444,   542,   145,
     146,   147,   148,   149,   150,   151,   443,   722,   520,   446,
     723,   724,   752,   152,   725,   447,   618,   448,   594,   530,
     603,   450,   624,   754,   452,   722,   533,   451,   723,   724,
     465,   633,   725,   466,   468,   409,   542,   552,   643,   475,
     476,   477,   543,   544,   545,   546,   547,   548,   549,   603,
     564,   542,   479,   480,   657,   481,   659,   647,   483,   662,
     454,   488,   493,   195,   494,   195,   184,   185,   186,   187,
     188,   189,   512,   550,   661,   669,   603,    87,  -337,   495,
     543,   544,   545,   546,   547,   548,   549,   523,   522,   682,
     542,   524,   687,   643,   525,   543,   544,   545,   546,   547,
     548,   549,   526,   528,   527,   -11,   698,   541,   565,   569,
     195,   550,   531,    60,   539,   551,   708,   536,   572,   558,
     195,   559,   567,   573,   576,   580,   550,   716,   542,   585,
      87,  -339,   610,   589,   543,   544,   545,   546,   547,   548,
     549,    87,   629,   734,   542,   614,   607,   612,   632,   615,
     616,   622,   665,   638,   634,   653,   627,   658,   195,   666,
     667,   670,   668,   748,   542,   550,   729,   676,   678,   590,
     679,   686,   543,   544,   545,   546,   547,   548,   549,   681,
     684,   693,   694,   727,   695,   715,   732,   763,   543,   544,
     545,   546,   547,   548,   549,   705,   707,   709,   733,   735,
     710,   751,   736,   550,   737,   755,   742,    87,   543,   544,
     545,   546,   547,   548,   549,   759,   765,   767,   705,   550,
     771,    59,   105,   650,    64,   234,   297,   408,   292,   395,
     274,   705,   739,   705,   133,   705,  -266,  -266,  -266,   550,
    -266,  -266,  -266,   655,  -266,  -266,  -266,  -266,  -266,   538,
     405,   753,  -266,  -266,  -266,  -266,  -266,  -266,  -266,  -266,
    -266,  -266,  -266,  -266,  -266,   423,  -266,  -266,  -266,  -266,
    -266,  -266,  -266,  -266,  -266,  -266,  -266,  -266,  -266,  -266,
    -266,  -266,  -266,  -266,  -266,  -266,   422,  -266,   484,  -266,
    -266,   113,   115,   335,   487,   486,  -266,  -266,  -266,  -266,
    -266,  -266,  -266,  -266,   490,   369,  -266,  -266,  -266,  -266,
    -266,   510,   690,   436,   611,   660,     6,     7,     8,   588,
       9,    10,    11,   485,    12,    13,    14,    15,    16,   619,
     630,   652,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,     0,    30,    31,    32,    33,
      34,     0,     0,    35,    36,    37,    38,    39,    40,    41,
      42,    43,    44,    45,    46,    47,     0,    48,     0,    49,
      50,     0,     0,     0,     0,     0,     0,     0,     0,   542,
       0,     0,     0,    51,     0,     0,    52,    53,    54,    55,
     542,     0,     0,     0,     0,     0,     0,   542,     0,     0,
       0,    65,   353,    -5,    -5,    66,    -5,    -5,    -5,    -5,
      -5,    -5,    -5,    -5,    -5,    -5,    -5,     0,    -5,    -5,
       0,     0,    -5,   543,   544,   545,   546,   547,   548,   549,
       0,     0,     0,     0,   543,   544,   545,   546,   547,   548,
     549,   543,   544,   545,   546,   547,   548,   549,   542,     0,
       0,     0,     0,     0,   550,    67,    68,   542,   696,     0,
       0,    69,    70,     0,   542,   550,     0,     0,     0,   700,
       0,   542,   550,    71,     0,     0,   703,     0,     0,     0,
      -5,   -68,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   543,   544,   545,   546,   547,   548,   549,     0,
       0,   543,   544,   545,   546,   547,   548,   549,   543,   544,
     545,   546,   547,   548,   549,   543,   544,   545,   546,   547,
     548,   549,     0,   550,     0,     0,     0,   730,     0,     0,
       0,     0,   550,     0,     0,     0,   757,     0,     0,   550,
       0,     0,     0,   760,     0,     0,   550,     0,     1,     2,
     768,    90,    91,    92,    93,    94,    95,    96,    97,    98,
      99,   100,   197,   101,   102,     0,     0,   103,   119,   120,
     121,   122,     0,   123,   124,   125,   126,   127,     0,     0,
       0,     0,   198,     0,   199,   200,   201,   202,   203,   204,
     205,   143,   144,   206,   207,   208,   209,   210,   211,     0,
       0,     0,     0,     0,     0,     0,   128,     0,     0,     0,
      83,   315,   316,     0,   212,   213,   145,   146,   147,   148,
     149,   150,   151,     0,     0,   104,     0,     0,     0,     0,
     152,   507,     0,     0,     0,   214,   215,     0,     0,     0,
      60,     0,     0,   129,     6,     7,     8,     0,     9,    10,
      11,     0,    12,    13,    14,    15,    16,     0,     0,     0,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,     0,    30,    31,    32,    33,    34,   143,
     218,    35,    36,    37,    38,    39,    40,    41,    42,    43,
      44,    45,    46,    47,     0,    48,     0,    49,   508,   197,
       0,     0,     0,     0,   145,   146,   147,   148,   149,   150,
     151,    51,     0,     0,    52,    53,    54,    55,   152,   198,
       0,   199,   200,   201,   202,   203,   204,   205,     0,     0,
     206,   207,   208,   209,   210,   211,     0,     6,     7,     8,
       0,     9,    10,    11,     0,    12,    13,    14,    15,    16,
       0,   212,   213,    17,    18,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,    29,     0,    30,    31,    32,
      33,    34,   214,   215,    35,    36,    37,    38,    39,    40,
      41,    42,    43,    44,    45,    46,    47,     0,    48,     0,
      49,    50,   740,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    51,     0,     0,    52,    53,    54,
      55,     6,     7,     8,     0,     9,    10,    11,     0,    12,
      13,    14,    15,    16,     0,     0,     0,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
     639,    30,    31,    32,    33,    34,     0,     0,    35,    36,
      37,    38,    39,    40,    41,    42,    43,    44,    45,    46,
      47,     0,    48,     0,    49,    50,     0,     0,   280,     0,
       0,     0,   143,   144,     0,     0,     0,     0,    51,     0,
       0,    52,    53,    54,    55,     0,     0,     0,   143,   144,
       0,    83,     0,     0,     0,     0,     0,   145,   146,   147,
     148,   149,   150,   151,     0,   640,   641,    83,     0,     0,
       0,   152,     0,   145,   146,   147,   148,   149,   150,   151,
     143,   144,   496,   497,     0,     0,     0,   281,     0,     0,
       0,     0,     0,     0,     0,   143,   144,   496,   497,    83,
       0,     0,     0,     0,     0,   145,   146,   147,   148,   149,
     150,   151,   143,   218,    83,     0,     0,     0,     0,   152,
     145,   146,   147,   148,   149,   150,   151,   143,   144,   498,
       0,    83,     0,     0,   152,     0,     0,   145,   146,   147,
     148,   149,   150,   151,     0,     0,    83,     0,     0,     0,
       0,   152,   145,   146,   147,   148,   149,   150,   151,   596,
     597,   598,   546,   599,   600,   601,   152,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     602,     0,     0,     0,    87
};

static const yytype_int16 yycheck[] =
{
       1,     2,    56,    57,    94,    71,    90,    89,    68,   182,
     135,   529,   286,   135,   135,    92,    93,    71,   578,   355,
      74,   142,    82,   241,   496,   497,   509,   568,    75,   579,
     590,     0,   326,   593,   575,    96,    97,    98,   595,   479,
      57,    60,    38,   261,   605,    17,    17,   530,   171,   172,
     173,    17,    64,   571,   572,   178,    68,   104,   140,   330,
      77,   545,   333,   581,   118,   549,    67,    86,    69,    70,
       1,   632,    39,   614,   624,     6,     7,     8,     9,    10,
      39,   564,   522,   633,   174,   379,   569,    18,    19,   136,
     650,    22,   152,   218,    90,   655,   218,   218,   182,   617,
     101,   161,   659,   163,   164,   662,   166,   625,   649,   380,
      81,   629,   596,    44,   598,    81,    57,   601,    90,   669,
      92,    88,    89,   184,   185,   186,   187,    86,    78,   670,
     695,   615,   682,    39,   224,   692,   696,    15,    81,   704,
     700,    31,   576,   703,    78,   227,    80,   229,    89,   272,
      81,   274,   693,    57,    58,   373,   492,   493,   676,   263,
     264,   265,   680,    53,    79,   683,   256,   412,    83,   652,
     730,   605,   737,    77,    79,   709,    44,   734,   232,    57,
      86,   271,    88,    89,   429,    90,    80,   360,   722,   746,
     724,   732,   726,   711,   666,   679,   175,   757,   632,    53,
     760,    86,    57,    57,    58,    90,   763,    80,   768,   188,
      57,    58,    59,    86,   488,    79,    82,    90,    84,    83,
      79,   281,    86,    77,     3,     4,    92,   309,    38,   747,
      77,    82,   286,    84,    90,    79,    79,    90,   756,    83,
       1,    92,    86,    86,   348,   349,   328,    59,   766,   339,
      90,   311,   329,    83,   331,   332,    78,   334,    79,   363,
     364,   365,   366,   367,    81,    86,   267,   357,    83,    42,
      43,   341,   342,   343,    79,    90,   360,    38,    39,   280,
      63,    64,   283,    57,    58,    59,   287,   288,   289,   290,
     291,   292,   374,    85,   376,   377,    57,    88,    89,    16,
     301,    85,    63,    64,    65,    66,    67,    68,    69,    55,
      85,    57,     6,    78,   368,    80,    77,    88,    89,    70,
     321,    38,    39,    63,    64,    65,    66,   411,    78,    90,
      80,   415,   392,   607,    82,   609,   406,   407,    90,    78,
      57,    80,    91,    90,    38,    39,    63,    64,    65,    66,
      67,    68,    69,    81,   424,   425,   426,   427,   428,    59,
      77,   486,   452,    57,   486,   486,    82,    81,    60,    63,
      64,    65,    66,    67,    68,    69,     1,    71,    72,    84,
     654,    86,    85,    77,    89,    90,    87,    85,    93,    91,
     664,    82,    89,   477,    83,   479,    91,    91,    79,     1,
      82,    91,    53,   457,   481,   482,   483,    86,    87,    82,
      89,    90,   496,   497,    93,    91,    91,    80,   500,    84,
      45,    46,    47,    48,    49,    50,    51,    82,   702,    39,
      89,    81,    84,    82,   488,    60,   518,   519,   522,    87,
     494,    87,    82,    45,    46,    47,    48,    49,    50,    51,
      91,    76,    89,    89,    89,    80,    81,    90,    60,   513,
      91,    86,   463,    89,    38,   466,    89,    84,    92,    86,
      92,   553,    89,    90,    76,   529,    93,   537,    80,    81,
      89,    89,    89,    57,    86,   486,    89,    91,     1,    63,
      64,    65,    66,    67,    68,    69,    89,    86,   499,    87,
      89,    90,    91,    77,    93,    84,   588,    87,   574,   510,
     576,    84,   594,    84,    82,    86,   517,    87,    89,    90,
      81,   603,    93,    89,    89,    17,     1,   581,   612,    89,
      89,    86,    45,    46,    47,    48,    49,    50,    51,   605,
     541,     1,    86,    89,   626,    86,   628,   613,    86,   631,
      44,    81,    57,   607,    81,   609,    11,    12,    13,    14,
      15,    16,    90,    76,   630,   647,   632,    80,    81,    89,
      45,    46,    47,    48,    49,    50,    51,    87,    82,   661,
       1,    87,   666,   667,    87,    45,    46,    47,    48,    49,
      50,    51,    87,    91,    87,    86,   678,    38,    86,    57,
     654,    76,    91,    78,    91,    80,   688,    90,    80,    91,
     664,    91,    90,    59,    80,    20,    76,   699,     1,    57,
      80,    81,    92,    82,    45,    46,    47,    48,    49,    50,
      51,    80,    80,   715,     1,    78,    81,    90,    80,    91,
      87,    81,   643,    58,    81,    81,    90,    57,   702,    79,
      82,    78,    91,   735,     1,    76,   710,    52,    78,    80,
      91,    83,    45,    46,    47,    48,    49,    50,    51,    81,
      81,    79,    81,    91,    81,    78,    82,   759,    45,    46,
      47,    48,    49,    50,    51,   686,   687,    83,    81,    79,
     691,    91,    81,    76,    81,    81,    84,    80,    45,    46,
      47,    48,    49,    50,    51,    82,    81,    81,   709,    76,
      81,     5,    72,    80,    57,   144,   218,   351,   189,   337,
     176,   722,   723,   724,     1,   726,     3,     4,     5,    76,
       7,     8,     9,    80,    11,    12,    13,    14,    15,   520,
     346,   742,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,   360,    33,    34,    35,    36,
      37,    38,    39,    40,    41,    42,    43,    44,    45,    46,
      47,    48,    49,    50,    51,    52,   359,    54,   452,    56,
      57,    72,    72,   256,   455,   454,    63,    64,    65,    66,
      67,    68,    69,    70,   458,   296,    73,    74,    75,    76,
      77,   486,   667,   375,   581,   629,     3,     4,     5,   569,
       7,     8,     9,    90,    11,    12,    13,    14,    15,   589,
     600,   618,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    -1,    33,    34,    35,    36,
      37,    -1,    -1,    40,    41,    42,    43,    44,    45,    46,
      47,    48,    49,    50,    51,    52,    -1,    54,    -1,    56,
      57,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     1,
      -1,    -1,    -1,    70,    -1,    -1,    73,    74,    75,    76,
       1,    -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,    -1,
      -1,     1,    89,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    14,    15,    16,    -1,    18,    19,
      -1,    -1,    22,    45,    46,    47,    48,    49,    50,    51,
      -1,    -1,    -1,    -1,    45,    46,    47,    48,    49,    50,
      51,    45,    46,    47,    48,    49,    50,    51,     1,    -1,
      -1,    -1,    -1,    -1,    76,    55,    56,     1,    80,    -1,
      -1,    61,    62,    -1,     1,    76,    -1,    -1,    -1,    80,
      -1,     1,    76,    73,    -1,    -1,    80,    -1,    -1,    -1,
      80,    81,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    45,    46,    47,    48,    49,    50,    51,    -1,
      -1,    45,    46,    47,    48,    49,    50,    51,    45,    46,
      47,    48,    49,    50,    51,    45,    46,    47,    48,    49,
      50,    51,    -1,    76,    -1,    -1,    -1,    80,    -1,    -1,
      -1,    -1,    76,    -1,    -1,    -1,    80,    -1,    -1,    76,
      -1,    -1,    -1,    80,    -1,    -1,    76,    -1,     3,     4,
      80,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,     1,    18,    19,    -1,    -1,    22,     6,     7,
       8,     9,    -1,    11,    12,    13,    14,    15,    -1,    -1,
      -1,    -1,    21,    -1,    23,    24,    25,    26,    27,    28,
      29,    38,    39,    32,    33,    34,    35,    36,    37,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    44,    -1,    -1,    -1,
      57,    58,    59,    -1,    53,    54,    63,    64,    65,    66,
      67,    68,    69,    -1,    -1,    80,    -1,    -1,    -1,    -1,
      77,    70,    -1,    -1,    -1,    74,    75,    -1,    -1,    -1,
      78,    -1,    -1,    81,     3,     4,     5,    -1,     7,     8,
       9,    -1,    11,    12,    13,    14,    15,    -1,    -1,    -1,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,    -1,    33,    34,    35,    36,    37,    38,
      39,    40,    41,    42,    43,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    -1,    54,    -1,    56,    57,     1,
      -1,    -1,    -1,    -1,    63,    64,    65,    66,    67,    68,
      69,    70,    -1,    -1,    73,    74,    75,    76,    77,    21,
      -1,    23,    24,    25,    26,    27,    28,    29,    -1,    -1,
      32,    33,    34,    35,    36,    37,    -1,     3,     4,     5,
      -1,     7,     8,     9,    -1,    11,    12,    13,    14,    15,
      -1,    53,    54,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    31,    -1,    33,    34,    35,
      36,    37,    74,    75,    40,    41,    42,    43,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    -1,    54,    -1,
      56,    57,    58,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    70,    -1,    -1,    73,    74,    75,
      76,     3,     4,     5,    -1,     7,     8,     9,    -1,    11,
      12,    13,    14,    15,    -1,    -1,    -1,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
       6,    33,    34,    35,    36,    37,    -1,    -1,    40,    41,
      42,    43,    44,    45,    46,    47,    48,    49,    50,    51,
      52,    -1,    54,    -1,    56,    57,    -1,    -1,    18,    -1,
      -1,    -1,    38,    39,    -1,    -1,    -1,    -1,    70,    -1,
      -1,    73,    74,    75,    76,    -1,    -1,    -1,    38,    39,
      -1,    57,    -1,    -1,    -1,    -1,    -1,    63,    64,    65,
      66,    67,    68,    69,    -1,    71,    72,    57,    -1,    -1,
      -1,    77,    -1,    63,    64,    65,    66,    67,    68,    69,
      38,    39,    40,    41,    -1,    -1,    -1,    77,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    38,    39,    40,    41,    57,
      -1,    -1,    -1,    -1,    -1,    63,    64,    65,    66,    67,
      68,    69,    38,    39,    57,    -1,    -1,    -1,    -1,    77,
      63,    64,    65,    66,    67,    68,    69,    38,    39,    87,
      -1,    57,    -1,    -1,    77,    -1,    -1,    63,    64,    65,
      66,    67,    68,    69,    -1,    -1,    57,    -1,    -1,    -1,
      -1,    77,    63,    64,    65,    66,    67,    68,    69,    45,
      46,    47,    48,    49,    50,    51,    77,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      76,    -1,    -1,    -1,    80
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     3,     4,    95,    96,   102,     3,     4,     5,     7,
       8,     9,    11,    12,    13,    14,    15,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      33,    34,    35,    36,    37,    40,    41,    42,    43,    44,
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
      86,   179,    16,    38,    39,    63,    64,    65,    66,    67,
      68,    69,    77,   101,   111,   113,   114,   115,   116,   117,
     118,   121,    38,   127,   127,    90,   127,   113,   166,    90,
     131,   131,   131,   131,    90,   135,   148,    90,   128,   100,
      59,   167,    83,   104,    11,    12,    13,    14,    15,    16,
     150,   151,   152,   153,   154,    98,    99,     1,    21,    23,
      24,    25,    26,    27,    28,    29,    32,    33,    34,    35,
      36,    37,    53,    54,    74,    75,   175,   176,    39,   114,
     115,   116,   119,   120,   173,   104,    79,    80,    86,    90,
     179,    81,   182,   114,   118,    64,    68,    63,    64,    65,
      66,   101,    83,   110,    85,    85,    85,    39,    86,    88,
      89,   101,   101,   101,    70,   101,    82,    31,    53,   132,
     137,   100,   112,   112,   112,   112,    53,    58,   113,   134,
     136,    90,   148,    90,   135,    42,    43,   129,   130,   112,
      18,    77,   117,   121,   157,   158,    81,   131,   131,   131,
     131,   148,   128,    59,   133,    91,    82,   119,    39,    88,
      89,   113,    81,    15,    57,   179,    60,   178,   179,    87,
      98,    85,    63,    64,   110,    58,    59,   107,   108,   109,
     121,    85,    88,    89,    89,    90,   123,   124,   207,    86,
      83,    86,    91,    83,    86,   166,    91,    82,   110,    79,
     143,   143,   143,   143,   100,    91,    82,    91,   112,   112,
      91,    82,    80,    89,   100,    57,    89,    92,   156,   100,
      82,    84,    99,   100,   100,   100,   100,   100,   100,   175,
      88,    89,    89,   100,    81,    82,    87,    91,   179,   101,
      82,    84,   100,    89,    89,    89,   124,   122,   179,   127,
     108,   127,   127,   108,   127,   132,   113,   144,    80,    98,
     161,   161,   161,   161,    91,   136,   143,   143,   129,    17,
      81,   138,   140,   141,    89,    92,   155,   155,    89,    58,
      59,   113,   156,   158,   143,   143,   143,   143,   143,    80,
      98,    89,    89,    89,   110,   179,   178,   179,   179,   124,
     108,    87,    89,    89,    91,   208,    87,    84,    87,   101,
      84,    87,    82,     1,    44,   159,   162,   163,   168,   169,
     171,   161,   161,   121,   141,    81,    89,   121,    89,   161,
     161,   161,   161,   161,   141,    89,    89,    86,   190,    86,
      89,    86,    86,    86,   144,    90,   174,   171,    81,    98,
     162,   100,   100,    57,    81,    89,    40,    41,    87,   121,
     180,   183,   188,   188,   127,   127,   127,    70,    57,   100,
     173,    99,    90,   139,   155,   155,    98,   121,   180,   180,
     100,   179,    82,    87,    87,    87,    87,    87,    91,   190,
     100,    91,    98,   100,   179,   179,    90,    92,   138,    91,
     188,    38,     1,    45,    46,    47,    48,    49,    50,    51,
      76,    80,    98,   181,   193,   198,   200,   190,    91,    91,
      58,    59,   101,   177,   100,    86,   204,    90,   204,    57,
     205,   206,    80,    59,   197,   204,    80,   194,   200,   179,
      20,   192,   190,   179,   202,    57,   202,   190,   207,    82,
      80,   200,   195,   200,   181,   202,    45,    46,    47,    49,
      50,    51,    76,   181,   196,   198,   199,    81,   194,   182,
      92,   193,    90,   191,    78,    91,    87,   203,   179,   206,
      81,   194,    81,   194,   179,   203,   204,    90,   204,    80,
     197,   204,    80,   179,    81,   196,    99,    99,    58,     6,
      71,    72,    91,   121,   184,   187,   189,   181,   202,   204,
      80,   200,   208,    81,   182,    80,   200,   179,    57,   179,
     195,   181,   179,   196,   182,   100,    79,    82,    91,   179,
      78,   202,   194,   190,    99,   194,    52,   201,    78,    91,
     203,    81,   179,   203,    81,    99,    83,   121,   180,   186,
     189,   182,   202,    79,    81,    81,    80,   200,   179,   204,
      80,   200,   182,    80,   200,   100,   185,   100,   179,    83,
     100,   203,   202,   201,   194,    78,   179,   194,    99,   194,
     201,    84,    86,    89,    90,    93,    83,    91,   185,    98,
      80,   200,    82,    81,   179,    79,    81,    81,   185,   100,
      58,   185,    84,   185,    84,   194,   202,   203,   179,   201,
      87,    91,    91,   100,    84,    81,   203,    80,   200,    82,
      80,   200,   194,   179,   194,    81,   203,    81,    80,   200,
     194,    81
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
     176,   176,   176,   176,   176,   176,   176,   176,   176,   176,
     177,   177,   177,   178,   178,   178,   179,   179,   179,   179,
     179,   179,   180,   181,   182,   183,   183,   183,   183,   183,
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
       1,     1,     1,     1,     1,     2,     1,     1,     1,     1,
       1,     1,     1,     0,     1,     3,     0,     1,     5,     5,
       5,     4,     3,     1,     1,     1,     3,     4,     3,     4,
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
#line 198 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.modlist) = (yyvsp[0].modlist); modlist = (yyvsp[0].modlist); }
#line 2269 "y.tab.c" /* yacc.c:1646  */
    break;

  case 3:
#line 202 "xi-grammar.y" /* yacc.c:1646  */
    { 
		  (yyval.modlist) = 0; 
		}
#line 2277 "y.tab.c" /* yacc.c:1646  */
    break;

  case 4:
#line 206 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.modlist) = new AstChildren<Module>(lineno, (yyvsp[-1].module), (yyvsp[0].modlist)); }
#line 2283 "y.tab.c" /* yacc.c:1646  */
    break;

  case 5:
#line 210 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 2289 "y.tab.c" /* yacc.c:1646  */
    break;

  case 6:
#line 212 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 2295 "y.tab.c" /* yacc.c:1646  */
    break;

  case 7:
#line 216 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 2301 "y.tab.c" /* yacc.c:1646  */
    break;

  case 8:
#line 218 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 2; }
#line 2307 "y.tab.c" /* yacc.c:1646  */
    break;

  case 9:
#line 222 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 2313 "y.tab.c" /* yacc.c:1646  */
    break;

  case 10:
#line 224 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 2319 "y.tab.c" /* yacc.c:1646  */
    break;

  case 11:
#line 229 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 2325 "y.tab.c" /* yacc.c:1646  */
    break;

  case 12:
#line 230 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MODULE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2331 "y.tab.c" /* yacc.c:1646  */
    break;

  case 13:
#line 231 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MAINMODULE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2337 "y.tab.c" /* yacc.c:1646  */
    break;

  case 14:
#line 232 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(EXTERN, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2343 "y.tab.c" /* yacc.c:1646  */
    break;

  case 15:
#line 234 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITCALL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2349 "y.tab.c" /* yacc.c:1646  */
    break;

  case 16:
#line 235 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITNODE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2355 "y.tab.c" /* yacc.c:1646  */
    break;

  case 17:
#line 236 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITPROC, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2361 "y.tab.c" /* yacc.c:1646  */
    break;

  case 18:
#line 238 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CHARE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2367 "y.tab.c" /* yacc.c:1646  */
    break;

  case 19:
#line 239 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MAINCHARE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2373 "y.tab.c" /* yacc.c:1646  */
    break;

  case 20:
#line 240 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(GROUP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2379 "y.tab.c" /* yacc.c:1646  */
    break;

  case 21:
#line 241 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NODEGROUP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2385 "y.tab.c" /* yacc.c:1646  */
    break;

  case 22:
#line 242 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ARRAY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2391 "y.tab.c" /* yacc.c:1646  */
    break;

  case 23:
#line 246 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INCLUDE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2397 "y.tab.c" /* yacc.c:1646  */
    break;

  case 24:
#line 247 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(STACKSIZE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2403 "y.tab.c" /* yacc.c:1646  */
    break;

  case 25:
#line 248 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(THREADED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2409 "y.tab.c" /* yacc.c:1646  */
    break;

  case 26:
#line 249 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(TEMPLATE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2415 "y.tab.c" /* yacc.c:1646  */
    break;

  case 27:
#line 250 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(WHENIDLE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2421 "y.tab.c" /* yacc.c:1646  */
    break;

  case 28:
#line 251 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(SYNC, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2427 "y.tab.c" /* yacc.c:1646  */
    break;

  case 29:
#line 252 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IGET, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2433 "y.tab.c" /* yacc.c:1646  */
    break;

  case 30:
#line 253 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(EXCLUSIVE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2439 "y.tab.c" /* yacc.c:1646  */
    break;

  case 31:
#line 254 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IMMEDIATE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2445 "y.tab.c" /* yacc.c:1646  */
    break;

  case 32:
#line 255 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(SKIPSCHED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2451 "y.tab.c" /* yacc.c:1646  */
    break;

  case 33:
#line 256 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOCOPY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2457 "y.tab.c" /* yacc.c:1646  */
    break;

  case 34:
#line 257 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOCOPYPOST, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2463 "y.tab.c" /* yacc.c:1646  */
    break;

  case 35:
#line 258 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INLINE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2469 "y.tab.c" /* yacc.c:1646  */
    break;

  case 36:
#line 259 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(VIRTUAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2475 "y.tab.c" /* yacc.c:1646  */
    break;

  case 37:
#line 260 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MIGRATABLE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2481 "y.tab.c" /* yacc.c:1646  */
    break;

  case 38:
#line 261 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CREATEHERE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2487 "y.tab.c" /* yacc.c:1646  */
    break;

  case 39:
#line 262 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CREATEHOME, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2493 "y.tab.c" /* yacc.c:1646  */
    break;

  case 40:
#line 263 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOKEEP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2499 "y.tab.c" /* yacc.c:1646  */
    break;

  case 41:
#line 264 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOTRACE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2505 "y.tab.c" /* yacc.c:1646  */
    break;

  case 42:
#line 265 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(APPWORK, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2511 "y.tab.c" /* yacc.c:1646  */
    break;

  case 43:
#line 268 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(PACKED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2517 "y.tab.c" /* yacc.c:1646  */
    break;

  case 44:
#line 269 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(VARSIZE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2523 "y.tab.c" /* yacc.c:1646  */
    break;

  case 45:
#line 270 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ENTRY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2529 "y.tab.c" /* yacc.c:1646  */
    break;

  case 46:
#line 271 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(FOR, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2535 "y.tab.c" /* yacc.c:1646  */
    break;

  case 47:
#line 272 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(FORALL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2541 "y.tab.c" /* yacc.c:1646  */
    break;

  case 48:
#line 273 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(WHILE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2547 "y.tab.c" /* yacc.c:1646  */
    break;

  case 49:
#line 274 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(WHEN, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2553 "y.tab.c" /* yacc.c:1646  */
    break;

  case 50:
#line 275 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(OVERLAP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2559 "y.tab.c" /* yacc.c:1646  */
    break;

  case 51:
#line 276 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(SERIAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2565 "y.tab.c" /* yacc.c:1646  */
    break;

  case 52:
#line 277 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IF, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2571 "y.tab.c" /* yacc.c:1646  */
    break;

  case 53:
#line 278 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ELSE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2577 "y.tab.c" /* yacc.c:1646  */
    break;

  case 54:
#line 280 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(LOCAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2583 "y.tab.c" /* yacc.c:1646  */
    break;

  case 55:
#line 282 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(USING, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2589 "y.tab.c" /* yacc.c:1646  */
    break;

  case 56:
#line 283 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ACCEL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2595 "y.tab.c" /* yacc.c:1646  */
    break;

  case 57:
#line 286 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ACCELBLOCK, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2601 "y.tab.c" /* yacc.c:1646  */
    break;

  case 58:
#line 287 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MEMCRITICAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2607 "y.tab.c" /* yacc.c:1646  */
    break;

  case 59:
#line 288 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(REDUCTIONTARGET, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2613 "y.tab.c" /* yacc.c:1646  */
    break;

  case 60:
#line 289 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CASE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2619 "y.tab.c" /* yacc.c:1646  */
    break;

  case 61:
#line 294 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 2625 "y.tab.c" /* yacc.c:1646  */
    break;

  case 62:
#line 296 "xi-grammar.y" /* yacc.c:1646  */
    {
		  char *tmp = new char[strlen((yyvsp[-3].strval))+strlen((yyvsp[0].strval))+3];
		  sprintf(tmp,"%s::%s", (yyvsp[-3].strval), (yyvsp[0].strval));
		  (yyval.strval) = tmp;
		}
#line 2635 "y.tab.c" /* yacc.c:1646  */
    break;

  case 63:
#line 302 "xi-grammar.y" /* yacc.c:1646  */
    {
		  char *tmp = new char[strlen((yyvsp[-3].strval))+5+3];
		  sprintf(tmp,"%s::array", (yyvsp[-3].strval));
		  (yyval.strval) = tmp;
		}
#line 2645 "y.tab.c" /* yacc.c:1646  */
    break;

  case 64:
#line 309 "xi-grammar.y" /* yacc.c:1646  */
    { 
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist)); 
		}
#line 2653 "y.tab.c" /* yacc.c:1646  */
    break;

  case 65:
#line 313 "xi-grammar.y" /* yacc.c:1646  */
    {  
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist)); 
		    (yyval.module)->setMain();
		}
#line 2662 "y.tab.c" /* yacc.c:1646  */
    break;

  case 66:
#line 320 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = 0; }
#line 2668 "y.tab.c" /* yacc.c:1646  */
    break;

  case 67:
#line 322 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = (yyvsp[-2].conslist); }
#line 2674 "y.tab.c" /* yacc.c:1646  */
    break;

  case 68:
#line 326 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = 0; }
#line 2680 "y.tab.c" /* yacc.c:1646  */
    break;

  case 69:
#line 328 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = new ConstructList(lineno, (yyvsp[-1].construct), (yyvsp[0].conslist)); }
#line 2686 "y.tab.c" /* yacc.c:1646  */
    break;

  case 70:
#line 332 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new UsingScope((yyvsp[0].strval), false); }
#line 2692 "y.tab.c" /* yacc.c:1646  */
    break;

  case 71:
#line 334 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new UsingScope((yyvsp[0].strval), true); }
#line 2698 "y.tab.c" /* yacc.c:1646  */
    break;

  case 72:
#line 336 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].member)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].member); }
#line 2704 "y.tab.c" /* yacc.c:1646  */
    break;

  case 73:
#line 338 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].message)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].message); }
#line 2710 "y.tab.c" /* yacc.c:1646  */
    break;

  case 74:
#line 340 "xi-grammar.y" /* yacc.c:1646  */
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
#line 353 "xi-grammar.y" /* yacc.c:1646  */
    { if((yyvsp[-2].conslist)) (yyvsp[-2].conslist)->recurse<int&>((yyvsp[-4].intval), &Construct::setExtern); (yyval.construct) = (yyvsp[-2].conslist); }
#line 2731 "y.tab.c" /* yacc.c:1646  */
    break;

  case 76:
#line 355 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new Scope((yyvsp[-3].strval), (yyvsp[-1].conslist)); }
#line 2737 "y.tab.c" /* yacc.c:1646  */
    break;

  case 77:
#line 357 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = (yyvsp[-1].construct); }
#line 2743 "y.tab.c" /* yacc.c:1646  */
    break;

  case 78:
#line 359 "xi-grammar.y" /* yacc.c:1646  */
    {
          ERROR("preceding construct must be semicolon terminated",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 2753 "y.tab.c" /* yacc.c:1646  */
    break;

  case 79:
#line 365 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].module)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].module); }
#line 2759 "y.tab.c" /* yacc.c:1646  */
    break;

  case 80:
#line 367 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2765 "y.tab.c" /* yacc.c:1646  */
    break;

  case 81:
#line 369 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2771 "y.tab.c" /* yacc.c:1646  */
    break;

  case 82:
#line 371 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2777 "y.tab.c" /* yacc.c:1646  */
    break;

  case 83:
#line 373 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2783 "y.tab.c" /* yacc.c:1646  */
    break;

  case 84:
#line 375 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].templat)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].templat); }
#line 2789 "y.tab.c" /* yacc.c:1646  */
    break;

  case 85:
#line 377 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = NULL; }
#line 2795 "y.tab.c" /* yacc.c:1646  */
    break;

  case 86:
#line 379 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = NULL; }
#line 2801 "y.tab.c" /* yacc.c:1646  */
    break;

  case 87:
#line 381 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = (yyvsp[0].accelBlock); }
#line 2807 "y.tab.c" /* yacc.c:1646  */
    break;

  case 88:
#line 383 "xi-grammar.y" /* yacc.c:1646  */
    {
          ERROR("invalid construct",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 2817 "y.tab.c" /* yacc.c:1646  */
    break;

  case 89:
#line 391 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamType((yyvsp[0].type)); }
#line 2823 "y.tab.c" /* yacc.c:1646  */
    break;

  case 90:
#line 393 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
#line 2829 "y.tab.c" /* yacc.c:1646  */
    break;

  case 91:
#line 395 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
#line 2835 "y.tab.c" /* yacc.c:1646  */
    break;

  case 92:
#line 399 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = new TParamList((yyvsp[0].tparam)); }
#line 2841 "y.tab.c" /* yacc.c:1646  */
    break;

  case 93:
#line 401 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = new TParamList((yyvsp[-2].tparam), (yyvsp[0].tparlist)); }
#line 2847 "y.tab.c" /* yacc.c:1646  */
    break;

  case 94:
#line 405 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = new TParamList(0); }
#line 2853 "y.tab.c" /* yacc.c:1646  */
    break;

  case 95:
#line 407 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = (yyvsp[0].tparlist); }
#line 2859 "y.tab.c" /* yacc.c:1646  */
    break;

  case 96:
#line 411 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = 0; }
#line 2865 "y.tab.c" /* yacc.c:1646  */
    break;

  case 97:
#line 413 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = (yyvsp[-1].tparlist); }
#line 2871 "y.tab.c" /* yacc.c:1646  */
    break;

  case 98:
#line 417 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("int"); }
#line 2877 "y.tab.c" /* yacc.c:1646  */
    break;

  case 99:
#line 419 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long"); }
#line 2883 "y.tab.c" /* yacc.c:1646  */
    break;

  case 100:
#line 421 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("short"); }
#line 2889 "y.tab.c" /* yacc.c:1646  */
    break;

  case 101:
#line 423 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("char"); }
#line 2895 "y.tab.c" /* yacc.c:1646  */
    break;

  case 102:
#line 425 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned int"); }
#line 2901 "y.tab.c" /* yacc.c:1646  */
    break;

  case 103:
#line 427 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long"); }
#line 2907 "y.tab.c" /* yacc.c:1646  */
    break;

  case 104:
#line 429 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long"); }
#line 2913 "y.tab.c" /* yacc.c:1646  */
    break;

  case 105:
#line 431 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long long"); }
#line 2919 "y.tab.c" /* yacc.c:1646  */
    break;

  case 106:
#line 433 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned short"); }
#line 2925 "y.tab.c" /* yacc.c:1646  */
    break;

  case 107:
#line 435 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned char"); }
#line 2931 "y.tab.c" /* yacc.c:1646  */
    break;

  case 108:
#line 437 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long long"); }
#line 2937 "y.tab.c" /* yacc.c:1646  */
    break;

  case 109:
#line 439 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("float"); }
#line 2943 "y.tab.c" /* yacc.c:1646  */
    break;

  case 110:
#line 441 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("double"); }
#line 2949 "y.tab.c" /* yacc.c:1646  */
    break;

  case 111:
#line 443 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long double"); }
#line 2955 "y.tab.c" /* yacc.c:1646  */
    break;

  case 112:
#line 445 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("void"); }
#line 2961 "y.tab.c" /* yacc.c:1646  */
    break;

  case 113:
#line 448 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ntype) = new NamedType((yyvsp[-1].strval),(yyvsp[0].tparlist)); }
#line 2967 "y.tab.c" /* yacc.c:1646  */
    break;

  case 114:
#line 449 "xi-grammar.y" /* yacc.c:1646  */
    { 
                    const char* basename, *scope;
                    splitScopedName((yyvsp[-1].strval), &scope, &basename);
                    (yyval.ntype) = new NamedType(basename, (yyvsp[0].tparlist), scope);
                }
#line 2977 "y.tab.c" /* yacc.c:1646  */
    break;

  case 115:
#line 455 "xi-grammar.y" /* yacc.c:1646  */
    {
			const char* basename, *scope;
			splitScopedName((yyvsp[-1].strval), &scope, &basename);
			(yyval.ntype) = new NamedType(basename, (yyvsp[0].tparlist), scope, true);
		}
#line 2987 "y.tab.c" /* yacc.c:1646  */
    break;

  case 116:
#line 463 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 2993 "y.tab.c" /* yacc.c:1646  */
    break;

  case 117:
#line 465 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ntype); }
#line 2999 "y.tab.c" /* yacc.c:1646  */
    break;

  case 118:
#line 469 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ptype) = new PtrType((yyvsp[-1].type)); }
#line 3005 "y.tab.c" /* yacc.c:1646  */
    break;

  case 119:
#line 473 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
#line 3011 "y.tab.c" /* yacc.c:1646  */
    break;

  case 120:
#line 475 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
#line 3017 "y.tab.c" /* yacc.c:1646  */
    break;

  case 121:
#line 479 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ftype) = new FuncType((yyvsp[-7].type), (yyvsp[-4].strval), (yyvsp[-1].plist)); }
#line 3023 "y.tab.c" /* yacc.c:1646  */
    break;

  case 122:
#line 483 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3029 "y.tab.c" /* yacc.c:1646  */
    break;

  case 123:
#line 485 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3035 "y.tab.c" /* yacc.c:1646  */
    break;

  case 124:
#line 487 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3041 "y.tab.c" /* yacc.c:1646  */
    break;

  case 125:
#line 489 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ftype); }
#line 3047 "y.tab.c" /* yacc.c:1646  */
    break;

  case 126:
#line 491 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[0].type)); }
#line 3053 "y.tab.c" /* yacc.c:1646  */
    break;

  case 127:
#line 493 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[-1].type)); }
#line 3059 "y.tab.c" /* yacc.c:1646  */
    break;

  case 128:
#line 497 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3065 "y.tab.c" /* yacc.c:1646  */
    break;

  case 129:
#line 499 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3071 "y.tab.c" /* yacc.c:1646  */
    break;

  case 130:
#line 501 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3077 "y.tab.c" /* yacc.c:1646  */
    break;

  case 131:
#line 503 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[0].type)); }
#line 3083 "y.tab.c" /* yacc.c:1646  */
    break;

  case 132:
#line 505 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[-1].type)); }
#line 3089 "y.tab.c" /* yacc.c:1646  */
    break;

  case 133:
#line 509 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType(new RValueReferenceType((yyvsp[-5].type))); }
#line 3095 "y.tab.c" /* yacc.c:1646  */
    break;

  case 134:
#line 511 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType(new ReferenceType((yyvsp[-4].type))); }
#line 3101 "y.tab.c" /* yacc.c:1646  */
    break;

  case 135:
#line 513 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType((yyvsp[-3].type)); }
#line 3107 "y.tab.c" /* yacc.c:1646  */
    break;

  case 136:
#line 515 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new RValueReferenceType((yyvsp[-2].type)); }
#line 3113 "y.tab.c" /* yacc.c:1646  */
    break;

  case 137:
#line 517 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ReferenceType((yyvsp[-1].type)); }
#line 3119 "y.tab.c" /* yacc.c:1646  */
    break;

  case 138:
#line 519 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3125 "y.tab.c" /* yacc.c:1646  */
    break;

  case 139:
#line 523 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType(new RValueReferenceType((yyvsp[-5].type))); }
#line 3131 "y.tab.c" /* yacc.c:1646  */
    break;

  case 140:
#line 525 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType(new ReferenceType((yyvsp[-4].type))); }
#line 3137 "y.tab.c" /* yacc.c:1646  */
    break;

  case 141:
#line 527 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType((yyvsp[-3].type)); }
#line 3143 "y.tab.c" /* yacc.c:1646  */
    break;

  case 142:
#line 529 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new RValueReferenceType((yyvsp[-2].type)); }
#line 3149 "y.tab.c" /* yacc.c:1646  */
    break;

  case 143:
#line 531 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ReferenceType((yyvsp[-1].type)); }
#line 3155 "y.tab.c" /* yacc.c:1646  */
    break;

  case 144:
#line 533 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3161 "y.tab.c" /* yacc.c:1646  */
    break;

  case 145:
#line 537 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 3167 "y.tab.c" /* yacc.c:1646  */
    break;

  case 146:
#line 541 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = (yyvsp[-1].val); }
#line 3173 "y.tab.c" /* yacc.c:1646  */
    break;

  case 147:
#line 545 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.vallist) = 0; }
#line 3179 "y.tab.c" /* yacc.c:1646  */
    break;

  case 148:
#line 547 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.vallist) = new ValueList((yyvsp[-1].val), (yyvsp[0].vallist)); }
#line 3185 "y.tab.c" /* yacc.c:1646  */
    break;

  case 149:
#line 551 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].vallist)); }
#line 3191 "y.tab.c" /* yacc.c:1646  */
    break;

  case 150:
#line 555 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[-3].type), (yyvsp[-1].strval), (yyvsp[0].vallist), 1); }
#line 3197 "y.tab.c" /* yacc.c:1646  */
    break;

  case 151:
#line 559 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0;}
#line 3203 "y.tab.c" /* yacc.c:1646  */
    break;

  case 152:
#line 561 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0;}
#line 3209 "y.tab.c" /* yacc.c:1646  */
    break;

  case 153:
#line 565 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3215 "y.tab.c" /* yacc.c:1646  */
    break;

  case 154:
#line 567 "xi-grammar.y" /* yacc.c:1646  */
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
#line 577 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[0].intval); }
#line 3233 "y.tab.c" /* yacc.c:1646  */
    break;

  case 156:
#line 579 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[-2].intval) | (yyvsp[0].intval); }
#line 3239 "y.tab.c" /* yacc.c:1646  */
    break;

  case 157:
#line 583 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3245 "y.tab.c" /* yacc.c:1646  */
    break;

  case 158:
#line 585 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3251 "y.tab.c" /* yacc.c:1646  */
    break;

  case 159:
#line 589 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = 0; }
#line 3257 "y.tab.c" /* yacc.c:1646  */
    break;

  case 160:
#line 591 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-1].cattr); }
#line 3263 "y.tab.c" /* yacc.c:1646  */
    break;

  case 161:
#line 595 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[0].cattr); }
#line 3269 "y.tab.c" /* yacc.c:1646  */
    break;

  case 162:
#line 597 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
#line 3275 "y.tab.c" /* yacc.c:1646  */
    break;

  case 163:
#line 601 "xi-grammar.y" /* yacc.c:1646  */
    { python_doc = NULL; (yyval.intval) = 0; }
#line 3281 "y.tab.c" /* yacc.c:1646  */
    break;

  case 164:
#line 603 "xi-grammar.y" /* yacc.c:1646  */
    { python_doc = (yyvsp[0].strval); (yyval.intval) = 0; }
#line 3287 "y.tab.c" /* yacc.c:1646  */
    break;

  case 165:
#line 607 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CPYTHON; }
#line 3293 "y.tab.c" /* yacc.c:1646  */
    break;

  case 166:
#line 611 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = 0; }
#line 3299 "y.tab.c" /* yacc.c:1646  */
    break;

  case 167:
#line 613 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-1].cattr); }
#line 3305 "y.tab.c" /* yacc.c:1646  */
    break;

  case 168:
#line 617 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[0].cattr); }
#line 3311 "y.tab.c" /* yacc.c:1646  */
    break;

  case 169:
#line 619 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
#line 3317 "y.tab.c" /* yacc.c:1646  */
    break;

  case 170:
#line 623 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CMIGRATABLE; }
#line 3323 "y.tab.c" /* yacc.c:1646  */
    break;

  case 171:
#line 625 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CPYTHON; }
#line 3329 "y.tab.c" /* yacc.c:1646  */
    break;

  case 172:
#line 629 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3335 "y.tab.c" /* yacc.c:1646  */
    break;

  case 173:
#line 631 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 3341 "y.tab.c" /* yacc.c:1646  */
    break;

  case 174:
#line 634 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3347 "y.tab.c" /* yacc.c:1646  */
    break;

  case 175:
#line 636 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 3353 "y.tab.c" /* yacc.c:1646  */
    break;

  case 176:
#line 639 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mv) = new MsgVar((yyvsp[-3].type), (yyvsp[-2].strval), (yyvsp[-4].intval), (yyvsp[-1].intval)); }
#line 3359 "y.tab.c" /* yacc.c:1646  */
    break;

  case 177:
#line 643 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mvlist) = new MsgVarList((yyvsp[0].mv)); }
#line 3365 "y.tab.c" /* yacc.c:1646  */
    break;

  case 178:
#line 645 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mvlist) = new MsgVarList((yyvsp[-1].mv), (yyvsp[0].mvlist)); }
#line 3371 "y.tab.c" /* yacc.c:1646  */
    break;

  case 179:
#line 649 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, (yyvsp[0].ntype)); }
#line 3377 "y.tab.c" /* yacc.c:1646  */
    break;

  case 180:
#line 651 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, (yyvsp[-2].ntype)); }
#line 3383 "y.tab.c" /* yacc.c:1646  */
    break;

  case 181:
#line 653 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, (yyvsp[-3].ntype), (yyvsp[-1].mvlist)); }
#line 3389 "y.tab.c" /* yacc.c:1646  */
    break;

  case 182:
#line 657 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = 0; }
#line 3395 "y.tab.c" /* yacc.c:1646  */
    break;

  case 183:
#line 659 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = (yyvsp[0].typelist); }
#line 3401 "y.tab.c" /* yacc.c:1646  */
    break;

  case 184:
#line 663 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = new TypeList((yyvsp[0].ntype)); }
#line 3407 "y.tab.c" /* yacc.c:1646  */
    break;

  case 185:
#line 665 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = new TypeList((yyvsp[-2].ntype), (yyvsp[0].typelist)); }
#line 3413 "y.tab.c" /* yacc.c:1646  */
    break;

  case 186:
#line 669 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr)|Chare::CCHARE, (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3419 "y.tab.c" /* yacc.c:1646  */
    break;

  case 187:
#line 671 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3425 "y.tab.c" /* yacc.c:1646  */
    break;

  case 188:
#line 675 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3431 "y.tab.c" /* yacc.c:1646  */
    break;

  case 189:
#line 679 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new NodeGroup(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3437 "y.tab.c" /* yacc.c:1646  */
    break;

  case 190:
#line 683 "xi-grammar.y" /* yacc.c:1646  */
    {/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",(yyvsp[-2].strval));
			(yyval.ntype) = new NamedType(buf); 
		}
#line 3447 "y.tab.c" /* yacc.c:1646  */
    break;

  case 191:
#line 689 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ntype) = (yyvsp[-1].ntype); }
#line 3453 "y.tab.c" /* yacc.c:1646  */
    break;

  case 192:
#line 693 "xi-grammar.y" /* yacc.c:1646  */
    {  (yyval.chare) = new Array(lineno, (yyvsp[-4].cattr), (yyvsp[-3].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3459 "y.tab.c" /* yacc.c:1646  */
    break;

  case 193:
#line 695 "xi-grammar.y" /* yacc.c:1646  */
    {  (yyval.chare) = new Array(lineno, (yyvsp[-3].cattr), (yyvsp[-4].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3465 "y.tab.c" /* yacc.c:1646  */
    break;

  case 194:
#line 699 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr)|Chare::CCHARE, new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist));}
#line 3471 "y.tab.c" /* yacc.c:1646  */
    break;

  case 195:
#line 701 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3477 "y.tab.c" /* yacc.c:1646  */
    break;

  case 196:
#line 705 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3483 "y.tab.c" /* yacc.c:1646  */
    break;

  case 197:
#line 709 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new NodeGroup( lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3489 "y.tab.c" /* yacc.c:1646  */
    break;

  case 198:
#line 713 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Array( lineno, 0, (yyvsp[-3].ntype), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3495 "y.tab.c" /* yacc.c:1646  */
    break;

  case 199:
#line 717 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-1].strval))); }
#line 3501 "y.tab.c" /* yacc.c:1646  */
    break;

  case 200:
#line 719 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-4].strval)), (yyvsp[-2].mvlist)); }
#line 3507 "y.tab.c" /* yacc.c:1646  */
    break;

  case 201:
#line 723 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = 0; }
#line 3513 "y.tab.c" /* yacc.c:1646  */
    break;

  case 202:
#line 725 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3519 "y.tab.c" /* yacc.c:1646  */
    break;

  case 203:
#line 729 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = 0; }
#line 3525 "y.tab.c" /* yacc.c:1646  */
    break;

  case 204:
#line 731 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3531 "y.tab.c" /* yacc.c:1646  */
    break;

  case 205:
#line 733 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3537 "y.tab.c" /* yacc.c:1646  */
    break;

  case 206:
#line 735 "xi-grammar.y" /* yacc.c:1646  */
    {
		  XStr typeStr;
		  (yyvsp[0].ntype)->print(typeStr);
		  char *tmp = strdup(typeStr.get_string());
		  (yyval.strval) = tmp;
		}
#line 3548 "y.tab.c" /* yacc.c:1646  */
    break;

  case 207:
#line 744 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TTypeEllipsis(new NamedEllipsisType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3554 "y.tab.c" /* yacc.c:1646  */
    break;

  case 208:
#line 746 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TTypeEllipsis(new NamedEllipsisType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3560 "y.tab.c" /* yacc.c:1646  */
    break;

  case 209:
#line 748 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TType(new NamedType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3566 "y.tab.c" /* yacc.c:1646  */
    break;

  case 210:
#line 750 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TType(new NamedType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3572 "y.tab.c" /* yacc.c:1646  */
    break;

  case 211:
#line 752 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TFunc((yyvsp[-1].ftype), (yyvsp[0].strval)); }
#line 3578 "y.tab.c" /* yacc.c:1646  */
    break;

  case 212:
#line 754 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TName((yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].strval)); }
#line 3584 "y.tab.c" /* yacc.c:1646  */
    break;

  case 213:
#line 758 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = new TVarList((yyvsp[0].tvar)); }
#line 3590 "y.tab.c" /* yacc.c:1646  */
    break;

  case 214:
#line 760 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = new TVarList((yyvsp[-2].tvar), (yyvsp[0].tvarlist)); }
#line 3596 "y.tab.c" /* yacc.c:1646  */
    break;

  case 215:
#line 764 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = (yyvsp[-1].tvarlist); }
#line 3602 "y.tab.c" /* yacc.c:1646  */
    break;

  case 216:
#line 768 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3608 "y.tab.c" /* yacc.c:1646  */
    break;

  case 217:
#line 770 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3614 "y.tab.c" /* yacc.c:1646  */
    break;

  case 218:
#line 772 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3620 "y.tab.c" /* yacc.c:1646  */
    break;

  case 219:
#line 774 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3626 "y.tab.c" /* yacc.c:1646  */
    break;

  case 220:
#line 776 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].message)); (yyvsp[0].message)->setTemplate((yyval.templat)); }
#line 3632 "y.tab.c" /* yacc.c:1646  */
    break;

  case 221:
#line 780 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = 0; }
#line 3638 "y.tab.c" /* yacc.c:1646  */
    break;

  case 222:
#line 782 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = (yyvsp[-2].mbrlist); }
#line 3644 "y.tab.c" /* yacc.c:1646  */
    break;

  case 223:
#line 786 "xi-grammar.y" /* yacc.c:1646  */
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
#line 794 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = new AstChildren<Member>(-1, (yyvsp[-1].member), (yyvsp[0].mbrlist)); }
#line 3662 "y.tab.c" /* yacc.c:1646  */
    break;

  case 225:
#line 798 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].readonly); }
#line 3668 "y.tab.c" /* yacc.c:1646  */
    break;

  case 226:
#line 800 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].readonly); }
#line 3674 "y.tab.c" /* yacc.c:1646  */
    break;

  case 228:
#line 803 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].member); }
#line 3680 "y.tab.c" /* yacc.c:1646  */
    break;

  case 229:
#line 805 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].pupable); }
#line 3686 "y.tab.c" /* yacc.c:1646  */
    break;

  case 230:
#line 807 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].includeFile); }
#line 3692 "y.tab.c" /* yacc.c:1646  */
    break;

  case 231:
#line 809 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new ClassDeclaration(lineno,(yyvsp[0].strval)); }
#line 3698 "y.tab.c" /* yacc.c:1646  */
    break;

  case 232:
#line 813 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1); }
#line 3704 "y.tab.c" /* yacc.c:1646  */
    break;

  case 233:
#line 815 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1); }
#line 3710 "y.tab.c" /* yacc.c:1646  */
    break;

  case 234:
#line 817 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[-6].strval)) + '<' +
					    ((yyvsp[-4].tparlist))->to_string() + '>').c_str()),
				    1);
		}
#line 3720 "y.tab.c" /* yacc.c:1646  */
    break;

  case 235:
#line 823 "xi-grammar.y" /* yacc.c:1646  */
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[-2]).first_column, (yylsp[-2]).last_column, (yylsp[-2]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1);
		}
#line 3730 "y.tab.c" /* yacc.c:1646  */
    break;

  case 236:
#line 829 "xi-grammar.y" /* yacc.c:1646  */
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[-5]).first_column, (yylsp[-5]).last_column, (yylsp[-5]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1);
		}
#line 3740 "y.tab.c" /* yacc.c:1646  */
    break;

  case 237:
#line 838 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 0); }
#line 3746 "y.tab.c" /* yacc.c:1646  */
    break;

  case 238:
#line 840 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 0); }
#line 3752 "y.tab.c" /* yacc.c:1646  */
    break;

  case 239:
#line 842 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[-6].strval)) + '<' +
					    ((yyvsp[-4].tparlist))->to_string() + '>').c_str()),
				    0);
		}
#line 3762 "y.tab.c" /* yacc.c:1646  */
    break;

  case 240:
#line 848 "xi-grammar.y" /* yacc.c:1646  */
    {
                  InitCall* rtn = new InitCall(lineno, (yyvsp[-3].strval), 0);
                  rtn->setAccel();
                  (yyval.member) = rtn;
		}
#line 3772 "y.tab.c" /* yacc.c:1646  */
    break;

  case 241:
#line 856 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[0].ntype),0); }
#line 3778 "y.tab.c" /* yacc.c:1646  */
    break;

  case 242:
#line 858 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[-2].ntype),(yyvsp[0].pupable)); }
#line 3784 "y.tab.c" /* yacc.c:1646  */
    break;

  case 243:
#line 861 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.includeFile) = new IncludeFile(lineno,(yyvsp[0].strval)); }
#line 3790 "y.tab.c" /* yacc.c:1646  */
    break;

  case 244:
#line 865 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].member); }
#line 3796 "y.tab.c" /* yacc.c:1646  */
    break;

  case 245:
#line 869 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].entry); }
#line 3802 "y.tab.c" /* yacc.c:1646  */
    break;

  case 246:
#line 871 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyvsp[0].entry)->tspec = (yyvsp[-1].tvarlist);
                  (yyval.member) = (yyvsp[0].entry);
                }
#line 3811 "y.tab.c" /* yacc.c:1646  */
    break;

  case 247:
#line 876 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[-1].member); }
#line 3817 "y.tab.c" /* yacc.c:1646  */
    break;

  case 248:
#line 878 "xi-grammar.y" /* yacc.c:1646  */
    {
          ERROR("invalid SDAG member",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 3827 "y.tab.c" /* yacc.c:1646  */
    break;

  case 249:
#line 886 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3833 "y.tab.c" /* yacc.c:1646  */
    break;

  case 250:
#line 888 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3839 "y.tab.c" /* yacc.c:1646  */
    break;

  case 251:
#line 890 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3845 "y.tab.c" /* yacc.c:1646  */
    break;

  case 252:
#line 892 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3851 "y.tab.c" /* yacc.c:1646  */
    break;

  case 253:
#line 894 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3857 "y.tab.c" /* yacc.c:1646  */
    break;

  case 254:
#line 896 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3863 "y.tab.c" /* yacc.c:1646  */
    break;

  case 255:
#line 898 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3869 "y.tab.c" /* yacc.c:1646  */
    break;

  case 256:
#line 900 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3875 "y.tab.c" /* yacc.c:1646  */
    break;

  case 257:
#line 902 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3881 "y.tab.c" /* yacc.c:1646  */
    break;

  case 258:
#line 904 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3887 "y.tab.c" /* yacc.c:1646  */
    break;

  case 259:
#line 906 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3893 "y.tab.c" /* yacc.c:1646  */
    break;

  case 260:
#line 909 "xi-grammar.y" /* yacc.c:1646  */
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
#line 919 "xi-grammar.y" /* yacc.c:1646  */
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
#line 936 "xi-grammar.y" /* yacc.c:1646  */
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
#line 953 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.accelBlock) = new AccelBlock(lineno, new XStr((yyvsp[-2].strval))); }
#line 3953 "y.tab.c" /* yacc.c:1646  */
    break;

  case 264:
#line 955 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.accelBlock) = new AccelBlock(lineno, NULL); }
#line 3959 "y.tab.c" /* yacc.c:1646  */
    break;

  case 265:
#line 959 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3965 "y.tab.c" /* yacc.c:1646  */
    break;

  case 266:
#line 963 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3971 "y.tab.c" /* yacc.c:1646  */
    break;

  case 267:
#line 965 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[-1].intval); }
#line 3977 "y.tab.c" /* yacc.c:1646  */
    break;

  case 268:
#line 967 "xi-grammar.y" /* yacc.c:1646  */
    { ERROR("invalid entry method attribute list",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
#line 3986 "y.tab.c" /* yacc.c:1646  */
    break;

  case 269:
#line 974 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[0].intval); }
#line 3992 "y.tab.c" /* yacc.c:1646  */
    break;

  case 270:
#line 976 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[-2].intval) | (yyvsp[0].intval); }
#line 3998 "y.tab.c" /* yacc.c:1646  */
    break;

  case 271:
#line 980 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = STHREADED; }
#line 4004 "y.tab.c" /* yacc.c:1646  */
    break;

  case 272:
#line 982 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SWHENIDLE; }
#line 4010 "y.tab.c" /* yacc.c:1646  */
    break;

  case 273:
#line 984 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SSYNC; }
#line 4016 "y.tab.c" /* yacc.c:1646  */
    break;

  case 274:
#line 986 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SIGET; }
#line 4022 "y.tab.c" /* yacc.c:1646  */
    break;

  case 275:
#line 988 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SLOCKED; }
#line 4028 "y.tab.c" /* yacc.c:1646  */
    break;

  case 276:
#line 990 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SCREATEHERE; }
#line 4034 "y.tab.c" /* yacc.c:1646  */
    break;

  case 277:
#line 992 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SCREATEHOME; }
#line 4040 "y.tab.c" /* yacc.c:1646  */
    break;

  case 278:
#line 994 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SNOKEEP; }
#line 4046 "y.tab.c" /* yacc.c:1646  */
    break;

  case 279:
#line 996 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SNOTRACE; }
#line 4052 "y.tab.c" /* yacc.c:1646  */
    break;

  case 280:
#line 998 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SAPPWORK; }
#line 4058 "y.tab.c" /* yacc.c:1646  */
    break;

  case 281:
#line 1000 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SIMMEDIATE; }
#line 4064 "y.tab.c" /* yacc.c:1646  */
    break;

  case 282:
#line 1002 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SSKIPSCHED; }
#line 4070 "y.tab.c" /* yacc.c:1646  */
    break;

  case 283:
#line 1004 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SINLINE; }
#line 4076 "y.tab.c" /* yacc.c:1646  */
    break;

  case 284:
#line 1006 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SLOCAL; }
#line 4082 "y.tab.c" /* yacc.c:1646  */
    break;

  case 285:
#line 1008 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SPYTHON; }
#line 4088 "y.tab.c" /* yacc.c:1646  */
    break;

  case 286:
#line 1010 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SMEM; }
#line 4094 "y.tab.c" /* yacc.c:1646  */
    break;

  case 287:
#line 1012 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SREDUCE; }
#line 4100 "y.tab.c" /* yacc.c:1646  */
    break;

  case 288:
#line 1014 "xi-grammar.y" /* yacc.c:1646  */
    {
        (yyval.intval) = SAGGREGATE;
    }
#line 4108 "y.tab.c" /* yacc.c:1646  */
    break;

  case 289:
#line 1018 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("invalid entry method attribute",
		        (yylsp[0]).first_column, (yylsp[0]).last_column);
		  yyclearin;
		  yyerrok;
		}
#line 4119 "y.tab.c" /* yacc.c:1646  */
    break;

  case 290:
#line 1027 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4125 "y.tab.c" /* yacc.c:1646  */
    break;

  case 291:
#line 1029 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4131 "y.tab.c" /* yacc.c:1646  */
    break;

  case 292:
#line 1031 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4137 "y.tab.c" /* yacc.c:1646  */
    break;

  case 293:
#line 1035 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = ""; }
#line 4143 "y.tab.c" /* yacc.c:1646  */
    break;

  case 294:
#line 1037 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4149 "y.tab.c" /* yacc.c:1646  */
    break;

  case 295:
#line 1039 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s, %s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4159 "y.tab.c" /* yacc.c:1646  */
    break;

  case 296:
#line 1047 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = ""; }
#line 4165 "y.tab.c" /* yacc.c:1646  */
    break;

  case 297:
#line 1049 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4171 "y.tab.c" /* yacc.c:1646  */
    break;

  case 298:
#line 1051 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s[%s]%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4181 "y.tab.c" /* yacc.c:1646  */
    break;

  case 299:
#line 1057 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s{%s}%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4191 "y.tab.c" /* yacc.c:1646  */
    break;

  case 300:
#line 1063 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s(%s)%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4201 "y.tab.c" /* yacc.c:1646  */
    break;

  case 301:
#line 1069 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"(%s)%s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4211 "y.tab.c" /* yacc.c:1646  */
    break;

  case 302:
#line 1077 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval));
		}
#line 4220 "y.tab.c" /* yacc.c:1646  */
    break;

  case 303:
#line 1084 "xi-grammar.y" /* yacc.c:1646  */
    { 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			(yyval.intval) = 0;
		}
#line 4230 "y.tab.c" /* yacc.c:1646  */
    break;

  case 304:
#line 1092 "xi-grammar.y" /* yacc.c:1646  */
    { 
			in_braces=0;
			(yyval.intval) = 0;
		}
#line 4239 "y.tab.c" /* yacc.c:1646  */
    break;

  case 305:
#line 1099 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[0].type));}
#line 4245 "y.tab.c" /* yacc.c:1646  */
    break;

  case 306:
#line 1101 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval)); (yyval.pname)->setConditional((yyvsp[0].intval)); }
#line 4251 "y.tab.c" /* yacc.c:1646  */
    break;

  case 307:
#line 1103 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[-3].type),(yyvsp[-2].strval),0,(yyvsp[0].val));}
#line 4257 "y.tab.c" /* yacc.c:1646  */
    break;

  case 308:
#line 1105 "xi-grammar.y" /* yacc.c:1646  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
		}
#line 4266 "y.tab.c" /* yacc.c:1646  */
    break;

  case 309:
#line 1110 "xi-grammar.y" /* yacc.c:1646  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
			(yyval.pname)->setRdma(CMK_ZC_P2P_SEND_MSG);
			if(firstRdma) {
				(yyval.pname)->setFirstRdma(true);
				firstRdma = false;
			}
		}
#line 4280 "y.tab.c" /* yacc.c:1646  */
    break;

  case 310:
#line 1120 "xi-grammar.y" /* yacc.c:1646  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
			(yyval.pname)->setRdma(CMK_ZC_P2P_RECV_MSG);
			if(firstRdma) {
				(yyval.pname)->setFirstRdma(true);
				firstRdma = false;
			}
		}
#line 4294 "y.tab.c" /* yacc.c:1646  */
    break;

  case 311:
#line 1131 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READONLY; }
#line 4300 "y.tab.c" /* yacc.c:1646  */
    break;

  case 312:
#line 1132 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READWRITE; }
#line 4306 "y.tab.c" /* yacc.c:1646  */
    break;

  case 313:
#line 1133 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_WRITEONLY; }
#line 4312 "y.tab.c" /* yacc.c:1646  */
    break;

  case 314:
#line 1136 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr((yyvsp[0].strval)); }
#line 4318 "y.tab.c" /* yacc.c:1646  */
    break;

  case 315:
#line 1137 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "->" << (yyvsp[0].strval); }
#line 4324 "y.tab.c" /* yacc.c:1646  */
    break;

  case 316:
#line 1138 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[-2].xstrptr)) << "." << (yyvsp[0].strval); }
#line 4330 "y.tab.c" /* yacc.c:1646  */
    break;

  case 317:
#line 1140 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << *((yyvsp[-1].xstrptr)) << "]";
                  delete (yyvsp[-3].xstrptr);
                  delete (yyvsp[-1].xstrptr);
                }
#line 4341 "y.tab.c" /* yacc.c:1646  */
    break;

  case 318:
#line 1147 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << (yyvsp[-1].strval) << "]";
                  delete (yyvsp[-3].xstrptr);
                }
#line 4351 "y.tab.c" /* yacc.c:1646  */
    break;

  case 319:
#line 1153 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "(" << *((yyvsp[-1].xstrptr)) << ")";
                  delete (yyvsp[-3].xstrptr);
                  delete (yyvsp[-1].xstrptr);
                }
#line 4362 "y.tab.c" /* yacc.c:1646  */
    break;

  case 320:
#line 1162 "xi-grammar.y" /* yacc.c:1646  */
    {
                  in_bracket = 0;
                  (yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName(), (yyvsp[-1].strval));
                }
#line 4371 "y.tab.c" /* yacc.c:1646  */
    break;

  case 321:
#line 1169 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[-6].intval));
                }
#line 4381 "y.tab.c" /* yacc.c:1646  */
    break;

  case 322:
#line 1175 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType(Parameter::ACCEL_BUFFER_TYPE_READWRITE);
		}
#line 4391 "y.tab.c" /* yacc.c:1646  */
    break;

  case 323:
#line 1181 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.pname) = (yyvsp[-3].pname);
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[-5].intval));
		}
#line 4401 "y.tab.c" /* yacc.c:1646  */
    break;

  case 324:
#line 1189 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 4407 "y.tab.c" /* yacc.c:1646  */
    break;

  case 325:
#line 1191 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 4413 "y.tab.c" /* yacc.c:1646  */
    break;

  case 326:
#line 1195 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 4419 "y.tab.c" /* yacc.c:1646  */
    break;

  case 327:
#line 1197 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 4425 "y.tab.c" /* yacc.c:1646  */
    break;

  case 328:
#line 1201 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = (yyvsp[-1].plist); }
#line 4431 "y.tab.c" /* yacc.c:1646  */
    break;

  case 329:
#line 1203 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList(new Parameter(0, new BuiltinType("void"))); }
#line 4437 "y.tab.c" /* yacc.c:1646  */
    break;

  case 330:
#line 1207 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = (yyvsp[-1].plist); }
#line 4443 "y.tab.c" /* yacc.c:1646  */
    break;

  case 331:
#line 1209 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = 0; }
#line 4449 "y.tab.c" /* yacc.c:1646  */
    break;

  case 332:
#line 1213 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = 0; }
#line 4455 "y.tab.c" /* yacc.c:1646  */
    break;

  case 333:
#line 1215 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4461 "y.tab.c" /* yacc.c:1646  */
    break;

  case 334:
#line 1219 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = 0; }
#line 4467 "y.tab.c" /* yacc.c:1646  */
    break;

  case 335:
#line 1221 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[0].sc)); }
#line 4473 "y.tab.c" /* yacc.c:1646  */
    break;

  case 336:
#line 1223 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[-2].slist)); }
#line 4479 "y.tab.c" /* yacc.c:1646  */
    break;

  case 337:
#line 1227 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.slist) = new SListConstruct((yyvsp[0].sc)); }
#line 4485 "y.tab.c" /* yacc.c:1646  */
    break;

  case 338:
#line 1229 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.slist) = new SListConstruct((yyvsp[-1].sc), (yyvsp[0].slist));  }
#line 4491 "y.tab.c" /* yacc.c:1646  */
    break;

  case 339:
#line 1233 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.olist) = new OListConstruct((yyvsp[0].sc)); }
#line 4497 "y.tab.c" /* yacc.c:1646  */
    break;

  case 340:
#line 1235 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.olist) = new OListConstruct((yyvsp[-1].sc), (yyvsp[0].slist)); }
#line 4503 "y.tab.c" /* yacc.c:1646  */
    break;

  case 341:
#line 1239 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.clist) = new CaseListConstruct((yyvsp[0].when)); }
#line 4509 "y.tab.c" /* yacc.c:1646  */
    break;

  case 342:
#line 1241 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.clist) = new CaseListConstruct((yyvsp[-1].when), (yyvsp[0].clist)); }
#line 4515 "y.tab.c" /* yacc.c:1646  */
    break;

  case 343:
#line 1243 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("case blocks can only contain when clauses",
		        (yylsp[0]).first_column, (yylsp[0]).last_column);
		  (yyval.clist) = 0;
		}
#line 4525 "y.tab.c" /* yacc.c:1646  */
    break;

  case 344:
#line 1251 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4531 "y.tab.c" /* yacc.c:1646  */
    break;

  case 345:
#line 1253 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = 0; }
#line 4537 "y.tab.c" /* yacc.c:1646  */
    break;

  case 346:
#line 1257 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-2].entrylist), 0); }
#line 4543 "y.tab.c" /* yacc.c:1646  */
    break;

  case 347:
#line 1259 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-1].entrylist), (yyvsp[0].sc)); }
#line 4549 "y.tab.c" /* yacc.c:1646  */
    break;

  case 348:
#line 1261 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-3].entrylist), (yyvsp[-1].slist)); }
#line 4555 "y.tab.c" /* yacc.c:1646  */
    break;

  case 349:
#line 1265 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4561 "y.tab.c" /* yacc.c:1646  */
    break;

  case 350:
#line 1267 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4567 "y.tab.c" /* yacc.c:1646  */
    break;

  case 351:
#line 1269 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4573 "y.tab.c" /* yacc.c:1646  */
    break;

  case 352:
#line 1271 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4579 "y.tab.c" /* yacc.c:1646  */
    break;

  case 353:
#line 1273 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4585 "y.tab.c" /* yacc.c:1646  */
    break;

  case 354:
#line 1275 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4591 "y.tab.c" /* yacc.c:1646  */
    break;

  case 355:
#line 1277 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4597 "y.tab.c" /* yacc.c:1646  */
    break;

  case 356:
#line 1279 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4603 "y.tab.c" /* yacc.c:1646  */
    break;

  case 357:
#line 1281 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4609 "y.tab.c" /* yacc.c:1646  */
    break;

  case 358:
#line 1283 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4615 "y.tab.c" /* yacc.c:1646  */
    break;

  case 359:
#line 1285 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4621 "y.tab.c" /* yacc.c:1646  */
    break;

  case 360:
#line 1287 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4627 "y.tab.c" /* yacc.c:1646  */
    break;

  case 361:
#line 1291 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new SerialConstruct((yyvsp[-2].strval), (yyvsp[-4].strval), (yylsp[-3]).first_line); }
#line 4633 "y.tab.c" /* yacc.c:1646  */
    break;

  case 362:
#line 1293 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new OverlapConstruct((yyvsp[-1].olist)); }
#line 4639 "y.tab.c" /* yacc.c:1646  */
    break;

  case 363:
#line 1295 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = (yyvsp[0].when); }
#line 4645 "y.tab.c" /* yacc.c:1646  */
    break;

  case 364:
#line 1297 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new CaseConstruct((yyvsp[-1].clist)); }
#line 4651 "y.tab.c" /* yacc.c:1646  */
    break;

  case 365:
#line 1299 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForConstruct((yyvsp[-8].intexpr), (yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4657 "y.tab.c" /* yacc.c:1646  */
    break;

  case 366:
#line 1301 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForConstruct((yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4663 "y.tab.c" /* yacc.c:1646  */
    break;

  case 367:
#line 1303 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-9].strval)), (yyvsp[-6].intexpr),
		             (yyvsp[-4].intexpr), (yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4670 "y.tab.c" /* yacc.c:1646  */
    break;

  case 368:
#line 1306 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-11].strval)), (yyvsp[-8].intexpr),
		             (yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4677 "y.tab.c" /* yacc.c:1646  */
    break;

  case 369:
#line 1309 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new IfConstruct((yyvsp[-3].intexpr), (yyvsp[-1].sc), (yyvsp[0].sc)); }
#line 4683 "y.tab.c" /* yacc.c:1646  */
    break;

  case 370:
#line 1311 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new IfConstruct((yyvsp[-5].intexpr), (yyvsp[-2].slist), (yyvsp[0].sc)); }
#line 4689 "y.tab.c" /* yacc.c:1646  */
    break;

  case 371:
#line 1313 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new WhileConstruct((yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4695 "y.tab.c" /* yacc.c:1646  */
    break;

  case 372:
#line 1315 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new WhileConstruct((yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4701 "y.tab.c" /* yacc.c:1646  */
    break;

  case 373:
#line 1317 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new SerialConstruct((yyvsp[-2].strval), NULL, (yyloc).first_line); }
#line 4707 "y.tab.c" /* yacc.c:1646  */
    break;

  case 374:
#line 1319 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("unknown SDAG construct or malformed entry method declaration.\n"
		        "You may have forgotten to terminate a previous entry method declaration with a"
		        " semicolon or forgotten to mark a block of sequential SDAG code as 'serial'",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
#line 4719 "y.tab.c" /* yacc.c:1646  */
    break;

  case 375:
#line 1329 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = 0; }
#line 4725 "y.tab.c" /* yacc.c:1646  */
    break;

  case 376:
#line 1331 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ElseConstruct((yyvsp[0].sc)); }
#line 4731 "y.tab.c" /* yacc.c:1646  */
    break;

  case 377:
#line 1333 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ElseConstruct((yyvsp[-1].slist)); }
#line 4737 "y.tab.c" /* yacc.c:1646  */
    break;

  case 378:
#line 1337 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intexpr) = new IntExprConstruct((yyvsp[0].strval)); }
#line 4743 "y.tab.c" /* yacc.c:1646  */
    break;

  case 379:
#line 1341 "xi-grammar.y" /* yacc.c:1646  */
    { in_int_expr = 0; (yyval.intval) = 0; }
#line 4749 "y.tab.c" /* yacc.c:1646  */
    break;

  case 380:
#line 1345 "xi-grammar.y" /* yacc.c:1646  */
    { in_int_expr = 1; (yyval.intval) = 0; }
#line 4755 "y.tab.c" /* yacc.c:1646  */
    break;

  case 381:
#line 1349 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[-1].strval), (yyvsp[0].plist), 0, 0, 0, (yyloc).first_line, (yyloc).last_line);
		  firstRdma = true;
		}
#line 4764 "y.tab.c" /* yacc.c:1646  */
    break;

  case 382:
#line 1354 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[-4].strval), (yyvsp[0].plist), 0, 0, (yyvsp[-2].strval), (yyloc).first_line, (yyloc).last_line);
		  firstRdma = true;
		}
#line 4773 "y.tab.c" /* yacc.c:1646  */
    break;

  case 383:
#line 1361 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.entrylist) = new EntryList((yyvsp[0].entry)); }
#line 4779 "y.tab.c" /* yacc.c:1646  */
    break;

  case 384:
#line 1363 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.entrylist) = new EntryList((yyvsp[-2].entry),(yyvsp[0].entrylist)); }
#line 4785 "y.tab.c" /* yacc.c:1646  */
    break;

  case 385:
#line 1367 "xi-grammar.y" /* yacc.c:1646  */
    { in_bracket=1; }
#line 4791 "y.tab.c" /* yacc.c:1646  */
    break;

  case 386:
#line 1370 "xi-grammar.y" /* yacc.c:1646  */
    { in_bracket=0; }
#line 4797 "y.tab.c" /* yacc.c:1646  */
    break;

  case 387:
#line 1374 "xi-grammar.y" /* yacc.c:1646  */
    { if (!macroDefined((yyvsp[0].strval), 1)) in_comment = 1; }
#line 4803 "y.tab.c" /* yacc.c:1646  */
    break;

  case 388:
#line 1378 "xi-grammar.y" /* yacc.c:1646  */
    { if (!macroDefined((yyvsp[0].strval), 0)) in_comment = 1; }
#line 4809 "y.tab.c" /* yacc.c:1646  */
    break;


#line 4813 "y.tab.c" /* yacc.c:1646  */
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
#line 1381 "xi-grammar.y" /* yacc.c:1906  */


void yyerror(const char *s) 
{
	fprintf(stderr, "[PARSE-ERROR] Unexpected/missing token at line %d. Current token being parsed: '%s'.\n", lineno, yytext);
}
