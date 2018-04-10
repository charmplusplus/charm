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
    CASE = 329,
    TYPENAME = 330
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
#define TYPENAME 330

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

#line 349 "y.tab.c" /* yacc.c:355  */
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

#line 380 "y.tab.c" /* yacc.c:358  */

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
#define YYLAST   1683

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  92
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  117
/* YYNRULES -- Number of rules.  */
#define YYNRULES  384
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  764

/* YYTRANSLATE[YYX] -- Symbol number corresponding to YYX as returned
   by yylex, with out-of-bounds checking.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   330

#define YYTRANSLATE(YYX)                                                \
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, without out-of-bounds checking.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    86,     2,
      84,    85,    83,     2,    80,    91,    87,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    77,    76,
      81,    90,    82,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    88,     2,    89,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    78,     2,    79,     2,     2,     2,     2,
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
      75
};

#if YYDEBUG
  /* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   196,   196,   201,   204,   209,   210,   214,   216,   221,
     222,   227,   229,   230,   231,   233,   234,   235,   237,   238,
     239,   240,   241,   245,   246,   247,   248,   249,   250,   251,
     252,   253,   254,   255,   256,   257,   258,   259,   260,   261,
     262,   265,   266,   267,   268,   269,   270,   271,   272,   273,
     274,   275,   277,   279,   280,   283,   284,   285,   286,   290,
     292,   298,   305,   309,   316,   318,   323,   324,   328,   330,
     332,   334,   336,   349,   351,   353,   355,   361,   363,   365,
     367,   369,   371,   373,   375,   377,   379,   387,   389,   391,
     395,   397,   402,   403,   408,   409,   413,   415,   417,   419,
     421,   423,   425,   427,   429,   431,   433,   435,   437,   439,
     441,   445,   446,   453,   455,   457,   461,   465,   467,   471,
     475,   477,   479,   481,   483,   485,   489,   491,   493,   495,
     497,   501,   503,   505,   507,   509,   511,   515,   517,   519,
     521,   523,   525,   529,   533,   538,   539,   543,   547,   552,
     553,   558,   559,   569,   571,   575,   577,   582,   583,   587,
     589,   594,   595,   599,   604,   605,   609,   611,   615,   617,
     622,   623,   627,   628,   631,   635,   637,   641,   643,   645,
     650,   651,   655,   657,   661,   663,   667,   671,   675,   681,
     685,   687,   691,   693,   697,   701,   705,   709,   711,   716,
     717,   722,   723,   725,   727,   736,   738,   740,   742,   744,
     746,   750,   752,   756,   760,   762,   764,   766,   768,   772,
     774,   779,   786,   790,   792,   794,   795,   797,   799,   801,
     805,   807,   809,   815,   821,   830,   832,   834,   840,   848,
     850,   853,   857,   861,   863,   868,   870,   878,   880,   882,
     884,   886,   888,   890,   892,   894,   896,   898,   901,   911,
     928,   945,   947,   951,   956,   957,   959,   966,   968,   972,
     974,   976,   978,   980,   982,   984,   986,   988,   990,   992,
     994,   996,   998,  1000,  1002,  1004,  1008,  1017,  1019,  1021,
    1026,  1027,  1029,  1038,  1039,  1041,  1047,  1053,  1059,  1067,
    1074,  1082,  1089,  1091,  1093,  1095,  1100,  1112,  1113,  1114,
    1117,  1118,  1119,  1120,  1127,  1133,  1142,  1149,  1155,  1161,
    1169,  1171,  1175,  1177,  1181,  1183,  1187,  1189,  1194,  1195,
    1199,  1201,  1203,  1207,  1209,  1213,  1215,  1219,  1221,  1223,
    1231,  1234,  1237,  1239,  1241,  1245,  1247,  1249,  1251,  1253,
    1255,  1257,  1259,  1261,  1263,  1265,  1267,  1271,  1273,  1275,
    1277,  1279,  1281,  1283,  1286,  1289,  1291,  1293,  1295,  1297,
    1299,  1310,  1311,  1313,  1317,  1321,  1325,  1329,  1334,  1341,
    1343,  1347,  1350,  1354,  1358
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
  "MEMCRITICAL", "REDUCTIONTARGET", "CASE", "TYPENAME", "';'", "':'",
  "'{'", "'}'", "','", "'<'", "'>'", "'*'", "'('", "')'", "'&'", "'.'",
  "'['", "']'", "'='", "'-'", "$accept", "File", "ModuleEList",
  "OptExtern", "OneOrMoreSemiColon", "OptSemiColon", "Name", "QualName",
  "Module", "ConstructEList", "ConstructList", "ConstructSemi",
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
     325,   326,   327,   328,   329,   330,    59,    58,   123,   125,
      44,    60,    62,    42,    40,    41,    38,    46,    91,    93,
      61,    45
};
# endif

#define YYPACT_NINF -645

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-645)))

#define YYTABLE_NINF -336

#define yytable_value_is_error(Yytable_value) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
     159,  1483,  1483,    48,  -645,   159,  -645,  -645,  -645,  -645,
    -645,  -645,  -645,  -645,  -645,  -645,  -645,  -645,  -645,  -645,
    -645,  -645,  -645,  -645,  -645,  -645,  -645,  -645,  -645,  -645,
    -645,  -645,  -645,  -645,  -645,  -645,  -645,  -645,  -645,  -645,
    -645,  -645,  -645,  -645,  -645,  -645,  -645,  -645,  -645,  -645,
    -645,  -645,  -645,  -645,    49,    49,  -645,  -645,  -645,   878,
     -18,  -645,  -645,  -645,    93,  1483,   139,  1483,  1483,   248,
    1083,    51,  1100,   878,  -645,  -645,  -645,  -645,   274,    64,
     100,  -645,   126,  -645,  -645,  -645,   -18,    45,  1524,   171,
     171,    -9,   100,    82,    82,    82,    82,   140,   152,  1483,
     194,   176,   878,  -645,  -645,  -645,  -645,  -645,  -645,  -645,
    -645,   258,  -645,  -645,  -645,  -645,   185,  -645,  -645,  -645,
    -645,  -645,  -645,  -645,  -645,  -645,  -645,  -645,   -18,  -645,
    -645,  -645,  1232,  1592,   878,   126,   187,   -34,    45,   199,
     381,  -645,  1608,  -645,   164,  -645,  -645,  -645,  -645,   232,
     100,   177,  -645,  -645,   207,   215,   227,  -645,    31,   100,
    -645,   100,   100,   246,   100,   237,  -645,    81,  1483,  1483,
    1483,  1483,  1267,   262,   264,   136,  1483,  -645,  -645,  -645,
    1558,   275,    82,    82,    82,    82,   262,   152,  -645,  -645,
    -645,  -645,  -645,   -18,  -645,  -645,  -645,  -645,  -645,  -645,
    -645,  -645,  -645,  -645,  -645,  -645,  -645,  -645,  -645,   296,
    -645,  -645,  -645,   270,   281,  1592,   207,   215,   227,    72,
    -645,   100,   286,    57,    45,   297,    45,   284,  -645,   185,
     283,     3,  -645,  -645,  -645,   123,  -645,  -645,  -645,  1091,
    -645,  -645,  -645,  -645,  -645,   287,   147,   285,   -39,   -24,
      88,   282,   116,   100,  -645,  -645,   293,   294,   292,   307,
     307,   307,   307,  -645,  1483,   300,   305,   302,  1339,  1483,
     325,  1483,  -645,  -645,   303,   308,   315,   793,   -28,   129,
    1483,   318,   313,   185,  1483,  1483,  1483,  1483,  1483,  1483,
    -645,  -645,  -645,  1232,   372,  -645,   260,   328,  1483,  -645,
    -645,  -645,   347,   348,   344,   338,    45,   -18,   100,  -645,
    -645,  -645,  -645,   350,  -645,   353,  -645,  1483,   351,   354,
     362,  -645,   363,  -645,    45,   171,  1091,   171,   171,  1091,
     171,  -645,  -645,    81,  -645,   100,   266,   266,   266,   266,
     364,  -645,   325,  -645,   307,   307,  -645,   136,     9,   365,
     369,   133,   375,   249,  -645,   373,  1558,  -645,  -645,   307,
     307,   307,   307,   307,   267,  -645,   377,   379,   380,   292,
      45,   297,    45,    45,  -645,   -39,  1091,  -645,   383,   382,
     384,  -645,  -645,   385,  -645,   388,   394,   392,   100,   396,
     395,  -645,   390,  -645,   180,   -18,  -645,  -645,  -645,  -645,
    -645,  -645,   266,   266,  -645,  -645,  -645,  1608,    12,   400,
     402,  1608,  -645,  -645,   403,  -645,  -645,  -645,  -645,  -645,
     266,   266,   266,   266,   266,   474,   -18,   405,   407,  -645,
     411,  -645,  -645,  -645,  -645,  -645,  -645,   412,   413,  -645,
    -645,  -645,  -645,   414,  -645,    37,   417,  -645,   100,  -645,
     715,   460,   424,   185,   180,  -645,  -645,  -645,  -645,  1483,
    -645,  -645,  1483,  -645,   451,  -645,  -645,  -645,  -645,  -645,
     429,   423,  -645,  1543,  -645,  1577,  -645,   171,   171,   171,
    -645,  1099,  1177,  -645,   185,   -18,  -645,   427,   369,   369,
     185,  -645,  1608,  -645,  1483,    45,   432,   431,   434,   435,
     436,   449,   446,   440,   411,  1483,  -645,   447,   185,  -645,
    -645,   -18,  1483,    45,     7,   452,  1577,  -645,  -645,  -645,
    -645,  -645,   506,   593,   411,  -645,   -18,   456,   458,  -645,
     252,  -645,  -645,  -645,  1483,  -645,   461,   463,   461,   494,
     475,   495,   461,   476,   376,   -18,    45,  -645,  -645,  -645,
     535,  -645,  -645,  -645,   126,  -645,   411,  -645,    45,   501,
      45,   172,   479,   613,   829,  -645,   482,    45,   483,   485,
     539,   199,   472,   593,   478,  -645,   491,   480,   486,  -645,
      45,   494,   439,  -645,   489,   576,    45,   486,   461,   487,
     461,   492,   495,   461,   496,    45,   497,   483,  -645,   185,
    -645,   185,   517,  -645,   342,   482,    45,   461,  -645,   867,
     385,  -645,  -645,   499,  -645,  -645,   199,   916,    45,   526,
      45,   829,   482,    45,   483,   199,  -645,  -645,  -645,  -645,
    -645,  -645,  -645,  -645,  -645,  1483,   513,   511,   503,    45,
     520,    45,   376,  -645,   411,  -645,   185,   376,   543,   521,
     509,   486,   522,    45,   486,   523,   185,   519,  1608,  1508,
    -645,   199,    45,   527,   524,  -645,  -645,   528,   923,  -645,
      45,   461,   932,  -645,   199,   939,  -645,  -645,  1483,  1483,
      45,   525,  -645,  1483,   486,    45,  -645,   543,   376,  -645,
     532,    45,   376,  -645,   185,   376,   543,  -645,   243,    38,
     538,  1483,   185,   988,   529,  -645,   533,    45,   534,   536,
    -645,   537,  -645,  -645,  1483,  1483,  1411,   546,  1483,  -645,
     276,   -18,   376,  -645,    45,  -645,   486,    45,  -645,   543,
     157,  -645,   540,   231,  1483,   299,  -645,   551,   486,   995,
     552,  -645,  -645,  -645,  -645,  -645,  -645,  -645,  1002,   376,
    -645,    45,   376,  -645,   554,   486,   556,  -645,  1010,  -645,
     376,  -645,   565,  -645
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
       0,    59,    69,   383,   384,   300,   262,   293,     0,   149,
     149,   149,     0,   157,   157,   157,   157,     0,   151,     0,
       0,     0,     0,    77,   223,   224,    71,    78,    79,    80,
      81,     0,    82,    70,   226,   225,     9,   257,   249,   250,
     251,   252,   253,   255,   256,   254,   247,   248,    75,    76,
      67,   266,     0,     0,     0,    68,     0,   294,   293,     0,
       0,   110,     0,    96,    97,    98,    99,   107,   108,     0,
       0,    94,   113,   115,   120,   121,   122,   123,   142,     0,
     150,     0,     0,     0,     0,   239,   227,     0,     0,     0,
       0,     0,     0,     0,   164,     0,     0,   229,   241,   228,
       0,     0,   157,   157,   157,   157,     0,   151,   214,   215,
     216,   217,   218,    10,    65,   286,   269,   270,   271,   272,
     278,   279,   280,   285,   273,   274,   275,   276,   277,   161,
     281,   283,   284,     0,   267,     0,   126,   127,   128,   136,
     263,     0,     0,     0,   293,   290,   293,     0,   301,     0,
       0,   124,   106,   109,   100,   101,   104,   105,   114,    92,
     112,   116,   117,   118,   125,     0,   141,     0,   145,   233,
     230,     0,   235,     0,   168,   169,     0,   159,    94,   180,
     180,   180,   180,   163,     0,     0,   166,     0,     0,     0,
       0,     0,   155,   156,     0,   153,   177,     0,     0,   123,
       0,   211,     0,     9,     0,     0,     0,     0,     0,     0,
     162,   282,   265,     0,   129,   130,   135,     0,     0,    74,
      61,    60,     0,   291,     0,     0,   293,   261,     0,   102,
     103,    88,    89,    90,    93,     0,    87,     0,   140,     0,
       0,   381,   145,   147,   293,   149,     0,   149,   149,     0,
     149,   240,   158,     0,   111,     0,     0,     0,     0,     0,
       0,   189,     0,   165,   180,   180,   152,     0,   170,     0,
     199,    59,     0,     0,   209,   201,     0,   213,    73,   180,
     180,   180,   180,   180,     0,   268,   134,     0,     0,    94,
     293,   290,   293,   293,   298,   145,     0,    95,     0,     0,
       0,   139,   146,     0,   143,     0,     0,     0,     0,     0,
       0,   160,   182,   181,     0,   219,   184,   185,   186,   187,
     188,   167,     0,     0,   154,   171,   178,     0,   170,     0,
       0,     0,   207,   208,     0,   202,   203,   204,   210,   212,
       0,     0,     0,     0,     0,   170,   197,     0,     0,   133,
       0,   296,   292,   297,   295,   148,    91,     0,     0,   138,
     382,   144,   234,     0,   231,     0,     0,   236,     0,   246,
       0,     0,     0,     0,     0,   242,   243,   190,   191,     0,
     176,   179,     0,   200,     0,   192,   193,   194,   195,   196,
       0,     0,   132,     0,    72,     0,   137,   149,   149,   149,
     183,     0,     0,   244,     9,   245,   222,   172,   199,   199,
       0,   131,     0,   325,   302,   293,   320,     0,     0,     0,
       0,     0,     0,    59,     0,     0,   220,     0,     0,   205,
     206,   198,     0,   293,   170,     0,     0,   324,   119,   232,
     238,   237,     0,     0,     0,   173,   174,     0,     0,   299,
       0,   303,   305,   321,     0,   370,     0,     0,     0,     0,
       0,   341,     0,     0,     0,   330,   293,   259,   359,   331,
     328,   306,   288,   287,   289,   304,     0,   376,   293,     0,
     293,     0,   379,     0,     0,   340,     0,   293,     0,     0,
       0,     0,     0,     0,     0,   374,     0,     0,     0,   377,
     293,     0,     0,   343,     0,     0,   293,     0,     0,     0,
       0,     0,   341,     0,     0,   293,     0,   337,   339,     9,
     334,     9,     0,   258,     0,     0,   293,     0,   375,     0,
       0,   380,   342,     0,   358,   336,     0,     0,   293,     0,
     293,     0,     0,   293,     0,     0,   360,   338,   332,   369,
     329,   307,   308,   309,   327,     0,     0,   322,     0,   293,
       0,   293,     0,   367,     0,   344,     9,     0,   371,     0,
       0,     0,     0,   293,     0,     0,     9,     0,     0,     0,
     326,     0,   293,     0,     0,   378,   357,     0,     0,   365,
     293,     0,     0,   346,     0,     0,   347,   356,     0,     0,
     293,     0,   323,     0,     0,   293,   368,   371,     0,   372,
       0,   293,     0,   354,     9,     0,   371,   310,     0,     0,
       0,     0,     0,     0,     0,   366,     0,   293,     0,     0,
     345,     0,   352,   318,     0,     0,     0,     0,     0,   316,
       0,   260,     0,   362,   293,   373,     0,   293,   355,   371,
       0,   312,     0,     0,     0,     0,   319,     0,     0,     0,
       0,   353,   315,   314,   313,   311,   317,   361,     0,     0,
     349,   293,     0,   363,     0,     0,     0,   348,     0,   364,
       0,   350,     0,   351
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -645,  -645,   626,  -645,   -51,  -278,    -1,   -55,   575,   592,
     -26,  -645,  -645,  -645,  -284,  -645,  -238,  -645,    -3,   -73,
    -126,  -131,  -125,  -165,   507,   433,  -645,   -79,  -645,  -645,
    -282,  -645,  -645,   -78,   465,   306,  -645,   -33,   330,  -645,
    -645,   490,   323,  -645,   154,  -645,  -645,  -286,  -645,  -124,
     218,  -645,  -645,  -645,   -40,  -645,  -645,  -645,  -645,  -645,
    -645,  -328,   317,  -645,   319,   603,  -645,  -121,   220,   606,
    -645,  -645,   426,  -645,  -645,  -645,  -645,   229,  -645,   201,
     234,   393,  -645,  -645,   310,   -81,  -475,   -59,  -520,  -645,
    -645,  -487,  -645,  -645,  -442,    26,  -488,  -645,  -645,   115,
    -548,    68,  -542,    98,  -522,  -645,  -493,  -644,  -528,  -531,
    -507,  -645,   111,   132,    84,  -645,  -645
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     3,     4,    70,   395,   194,   258,   151,     5,    61,
      71,    72,    73,   313,   314,   315,   240,   152,   259,   153,
     154,   155,   156,   157,   158,   219,   220,   316,   383,   322,
     323,   104,   105,   161,   176,   274,   275,   168,   256,   291,
     266,   173,   267,   257,   407,   508,   408,   409,   106,   336,
     393,   107,   108,   109,   174,   110,   188,   189,   190,   191,
     192,   412,   354,   281,   282,   451,   112,   396,   452,   453,
     114,   115,   166,   179,   454,   455,   129,   456,    74,   221,
     133,   213,   214,   555,   304,   575,   495,   546,   229,   496,
     636,   698,   681,   637,   497,   638,   474,   605,   573,   547,
     569,   584,   596,   566,   548,   598,   570,   669,   576,   609,
     558,   562,   563,   324,   441,    75,    76
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[] =
{
      54,    55,   217,    60,    60,   358,   139,   216,   218,   159,
      87,    82,   162,   164,   230,   279,   523,   513,    86,   165,
     334,   128,   600,   413,   405,   135,   405,   351,   160,   405,
     549,   560,   578,   498,   613,   567,   550,   615,   136,   587,
     382,   244,   386,   705,   224,   389,   597,   130,    56,   321,
     225,   601,   712,   136,   226,   627,   617,   227,    77,   352,
     325,   169,   170,   171,    79,   193,    83,    84,   574,   244,
     583,   585,   300,   579,   533,   597,   181,   238,   640,   163,
     549,   618,   655,   620,   217,   741,   623,   245,   406,   216,
     218,  -175,   436,   435,   664,   529,   646,   530,   177,   667,
     641,   280,   597,   137,   248,   656,   249,   250,   222,   252,
     295,   254,   301,   663,   136,   245,   643,   246,   247,   718,
     672,   478,   460,   675,   648,    58,   529,    59,   585,   138,
     116,   430,   255,   269,   684,    78,   337,   338,   339,   470,
     706,   683,   134,   302,   709,   305,   288,   711,   298,   284,
     285,   286,   287,   703,   694,    81,   665,   704,   296,   297,
     509,   510,     1,     2,   691,   136,   260,   261,   262,   326,
     167,   265,   327,   276,   737,   689,   272,   273,   307,   693,
     165,   449,   696,   680,   309,   310,    88,    89,    90,    91,
      92,   279,    80,   136,    81,   739,   738,   329,    99,   100,
     330,   754,   101,   136,   756,   238,   506,   748,   160,  -201,
     723,  -201,   762,  -199,   720,  -199,   397,   398,   399,   353,
     402,   403,   450,   411,   758,   374,   232,   730,   172,   733,
     233,   735,   193,   318,   319,   420,   421,   422,   423,   424,
     175,   714,   742,   384,   715,   716,   750,   385,   717,   387,
     388,   178,   390,   375,   136,   753,   473,   180,   239,  -221,
     321,    58,   392,   340,   223,   761,   344,   265,   345,   182,
     183,   184,   185,   186,   187,   131,   350,   280,   228,   355,
     417,   457,   458,   359,   360,   361,   362,   363,   364,   431,
     241,   433,   434,   234,   235,   236,   237,   369,   242,   465,
     466,   467,   468,   469,    81,   415,   416,    81,   552,   553,
     243,  -264,  -264,   426,   251,   714,   378,   253,   715,   716,
     744,   628,   717,   629,    58,   713,    85,   714,   459,  -264,
     715,   716,   463,   445,   717,  -264,  -264,  -264,  -264,  -264,
    -264,  -264,    58,    58,   394,   425,   366,   367,   631,  -264,
     268,   217,   270,   290,   283,   303,   216,   218,   736,   292,
     714,   293,   132,   715,   716,   299,   308,   717,   666,   306,
     317,   328,   320,   239,   333,   392,   263,   535,   677,   141,
     142,   746,   332,   714,   335,   342,   715,   716,   347,   341,
     717,   343,   346,   348,   494,   357,   494,    81,   356,   499,
     500,   501,   485,   143,   144,   145,   146,   147,   148,   149,
     295,   632,   633,   512,   515,   368,   710,   150,   141,   536,
     537,   538,   539,   540,   541,   542,   370,   373,   371,   372,
     376,   634,   528,   193,  -300,   377,    81,   494,   379,   511,
     535,   380,   143,   144,   145,   146,   147,   148,   149,   381,
     543,   321,   410,   400,    85,  -300,   150,   526,   487,   411,
    -300,   488,   414,   353,   427,   571,   428,   429,   437,   438,
     448,   439,   545,   442,   440,   554,   443,   444,   446,   461,
     447,   504,   536,   537,   538,   539,   540,   541,   542,   462,
     464,   405,   471,   514,   472,   473,   475,  -300,   477,   610,
     476,   479,   450,   484,   524,   616,   489,   586,   490,   595,
     491,   527,   516,   543,   625,   507,   517,    85,   612,   518,
     519,   520,   545,  -300,   -11,   635,   588,   589,   590,   539,
     591,   592,   593,   556,   521,   522,   525,   649,   595,   651,
     535,   532,   654,   534,   529,   557,   639,   551,   193,   561,
     193,   559,   565,   564,   568,   572,   577,   594,   661,   581,
      85,    85,   602,   653,   599,   595,   604,   606,   614,   607,
     621,   608,   674,   630,   624,   619,   626,   535,   645,   679,
     635,   650,   536,   537,   538,   539,   540,   541,   542,   690,
     658,   659,   660,   668,   535,   193,   662,   670,   671,   700,
     678,   673,   676,   686,   685,   193,   701,   687,   707,   724,
     708,   727,   725,   543,   535,   728,   729,    85,  -333,   536,
     537,   538,   539,   540,   541,   542,   726,   719,   734,   743,
     747,    57,   751,   757,   657,   759,   536,   537,   538,   539,
     540,   541,   542,   193,   763,   103,   740,    62,   294,   231,
     543,   721,   289,   404,    85,  -335,   536,   537,   538,   539,
     540,   541,   542,   391,   271,   401,   480,   543,   531,    58,
     755,   544,   418,   111,   486,   419,   113,   697,   699,   331,
     483,   432,   702,   505,   482,   682,   365,   543,   603,   652,
     622,   582,   611,   580,   644,     0,     0,     0,     0,     0,
     697,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   697,   731,   697,   131,   697,  -264,  -264,
    -264,     0,  -264,  -264,  -264,     0,  -264,  -264,  -264,  -264,
    -264,     0,     0,   745,  -264,  -264,  -264,  -264,  -264,  -264,
    -264,  -264,  -264,  -264,  -264,  -264,     0,  -264,  -264,  -264,
    -264,  -264,  -264,  -264,  -264,  -264,  -264,  -264,  -264,  -264,
    -264,  -264,  -264,  -264,  -264,  -264,     0,  -264,     0,  -264,
    -264,     0,     0,     0,     0,     0,  -264,  -264,  -264,  -264,
    -264,  -264,  -264,  -264,     0,     0,  -264,  -264,  -264,  -264,
    -264,     0,     0,     0,     0,     0,     6,     7,     8,     0,
       9,    10,    11,   481,    12,    13,    14,    15,    16,     0,
       0,     0,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,     0,    29,    30,    31,    32,    33,
     535,     0,    34,    35,    36,    37,    38,    39,    40,    41,
      42,    43,    44,    45,     0,    46,     0,    47,    48,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    49,     0,     0,    50,    51,    52,    53,   535,     0,
       0,     0,   536,   537,   538,   539,   540,   541,   542,    63,
     349,    -5,    -5,    64,    -5,    -5,    -5,    -5,    -5,    -5,
      -5,    -5,    -5,    -5,    -5,     0,    -5,    -5,     0,     0,
      -5,     0,     0,   543,     0,     0,     0,    85,     0,     0,
     536,   537,   538,   539,   540,   541,   542,   535,     0,     0,
       0,     0,     0,     0,   535,     0,     0,     0,     0,     0,
       0,    65,    66,   535,     0,     0,     0,    67,    68,     0,
     535,   543,     0,     0,     0,   642,     0,     0,     0,    69,
       0,     0,     0,     0,     0,     0,    -5,   -66,     0,   536,
     537,   538,   539,   540,   541,   542,   536,   537,   538,   539,
     540,   541,   542,     0,     0,   536,   537,   538,   539,   540,
     541,   542,   536,   537,   538,   539,   540,   541,   542,   535,
     543,     0,     0,     0,   647,     0,   535,   543,     0,     0,
       0,   688,     0,   535,     0,     0,   543,     0,     0,     0,
     692,   535,     0,   543,     0,     0,     0,   695,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   536,   537,   538,   539,   540,   541,   542,   536,   537,
     538,   539,   540,   541,   542,   536,   537,   538,   539,   540,
     541,   542,     0,   536,   537,   538,   539,   540,   541,   542,
       0,     0,   543,     0,     0,     0,   722,     0,     0,   543,
       0,     0,     0,   749,     0,     0,   543,     0,     0,     0,
     752,     0,     0,     0,   543,     0,     1,     2,   760,    88,
      89,    90,    91,    92,    93,    94,    95,    96,    97,    98,
     195,    99,   100,     0,     0,   101,   117,   118,   119,   120,
       0,   121,   122,   123,   124,   125,     0,     0,     0,     0,
     196,     0,   197,   198,   199,   200,   201,   202,   141,   142,
     203,   204,   205,   206,   207,   208,     0,     0,     0,     0,
       0,     0,   126,     0,     0,     0,    81,   311,   312,     0,
     209,   210,   143,   144,   145,   146,   147,   148,   149,     0,
       0,   102,     0,     0,     0,     0,   150,   502,     0,     0,
       0,   211,   212,     0,     0,     0,    58,     0,     0,   127,
       6,     7,     8,     0,     9,    10,    11,     0,    12,    13,
      14,    15,    16,     0,     0,     0,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,     0,    29,
      30,    31,    32,    33,   141,   215,    34,    35,    36,    37,
      38,    39,    40,    41,    42,    43,    44,    45,     0,    46,
       0,    47,   503,   195,     0,     0,     0,     0,   143,   144,
     145,   146,   147,   148,   149,    49,     0,     0,    50,    51,
      52,    53,   150,   196,     0,   197,   198,   199,   200,   201,
     202,     0,     0,   203,   204,   205,   206,   207,   208,     0,
       6,     7,     8,     0,     9,    10,    11,     0,    12,    13,
      14,    15,    16,   209,   210,     0,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,     0,    29,
      30,    31,    32,    33,   211,   212,    34,    35,    36,    37,
      38,    39,    40,    41,    42,    43,    44,    45,   263,    46,
       0,    47,    48,   264,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    49,     0,     0,    50,    51,
      52,    53,     6,     7,     8,     0,     9,    10,    11,     0,
      12,    13,    14,    15,    16,     0,     0,     0,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
       0,    29,    30,    31,    32,    33,     0,     0,    34,    35,
      36,    37,    38,    39,    40,    41,    42,    43,    44,    45,
       0,    46,     0,    47,    48,   264,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    49,     0,     0,
      50,    51,    52,    53,     6,     7,     8,     0,     9,    10,
      11,     0,    12,    13,    14,    15,    16,     0,     0,     0,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,     0,    29,    30,    31,    32,    33,     0,     0,
      34,    35,    36,    37,    38,    39,    40,    41,    42,    43,
      44,    45,     0,    46,     0,    47,    48,   732,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    49,
       0,     0,    50,    51,    52,    53,     6,     7,     8,     0,
       9,    10,    11,     0,    12,    13,    14,    15,    16,     0,
       0,     0,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,   631,    29,    30,    31,    32,    33,
       0,     0,    34,    35,    36,    37,    38,    39,    40,    41,
      42,    43,    44,    45,     0,    46,     0,    47,    48,     0,
     140,     0,     0,     0,     0,   141,   142,     0,     0,     0,
       0,    49,     0,     0,    50,    51,    52,    53,     0,     0,
       0,   141,   142,    81,     0,     0,     0,     0,     0,   143,
     144,   145,   146,   147,   148,   149,   277,   632,   633,    81,
     141,   142,   492,   150,     0,   143,   144,   145,   146,   147,
     148,   149,     0,     0,     0,   141,   142,     0,    81,   150,
       0,     0,     0,     0,   143,   144,   145,   146,   147,   148,
     149,     0,     0,    81,   141,   142,   492,     0,   150,   143,
     144,   145,   146,   147,   148,   149,     0,     0,   493,   141,
     215,     0,    81,   278,     0,     0,     0,     0,   143,   144,
     145,   146,   147,   148,   149,   141,   142,    81,     0,     0,
       0,     0,   150,   143,   144,   145,   146,   147,   148,   149,
       0,     0,     0,    81,     0,     0,     0,   150,     0,   143,
     144,   145,   146,   147,   148,   149,     0,     0,     0,     0,
       0,     0,     0,   150
};

static const yytype_int16 yycheck[] =
{
       1,     2,   133,    54,    55,   283,    87,   133,   133,    88,
      69,    66,    90,    91,   140,   180,   504,   492,    69,    92,
     258,    72,   570,   351,    17,    80,    17,    55,    37,    17,
     523,   538,   560,   475,   582,   542,   524,   585,    77,   567,
     322,    38,   326,   687,    78,   329,   568,    73,     0,    88,
      84,   571,   696,    77,    88,   597,   587,   138,    76,    87,
      84,    94,    95,    96,    65,   116,    67,    68,   556,    38,
     563,   564,    15,   561,   516,   597,   102,   150,   606,    88,
     573,   588,   624,   590,   215,   729,   593,    84,    79,   215,
     215,    79,   376,   375,   642,    88,   616,    90,    99,   647,
     607,   180,   624,    58,   159,   625,   161,   162,   134,   164,
      38,    30,    55,   641,    77,    84,   609,    86,    87,    81,
     651,    84,   408,   654,   617,    76,    88,    78,   621,    84,
      79,   369,    51,   173,   662,    42,   260,   261,   262,   425,
     688,   661,    78,   224,   692,   226,   186,   695,   221,   182,
     183,   184,   185,   684,   674,    55,   644,   685,    86,    87,
     488,   489,     3,     4,   671,    77,   169,   170,   171,    81,
      88,   172,    84,   176,   722,   668,    40,    41,   229,   672,
     253,     1,   675,   658,    61,    62,     6,     7,     8,     9,
      10,   356,    53,    77,    55,   726,   724,    81,    18,    19,
      84,   749,    22,    77,   752,   278,   484,   738,    37,    80,
     703,    82,   760,    80,   701,    82,   337,   338,   339,    90,
     344,   345,    42,    90,   755,   306,    62,   714,    88,   716,
      66,   718,   283,    86,    87,   359,   360,   361,   362,   363,
      88,    84,    85,   324,    87,    88,   739,   325,    91,   327,
     328,    57,   330,   308,    77,   748,    84,    81,    81,    79,
      88,    76,   335,   264,    77,   758,   269,   268,   271,    11,
      12,    13,    14,    15,    16,     1,   277,   356,    79,   280,
     353,   402,   403,   284,   285,   286,   287,   288,   289,   370,
      83,   372,   373,    61,    62,    63,    64,   298,    83,   420,
     421,   422,   423,   424,    55,    56,    57,    55,    56,    57,
      83,    37,    38,   364,    68,    84,   317,    80,    87,    88,
      89,   599,    91,   601,    76,    82,    78,    84,   407,    55,
      87,    88,   411,   388,    91,    61,    62,    63,    64,    65,
      66,    67,    76,    76,    78,    78,    86,    87,     6,    75,
      88,   482,    88,    57,    79,    58,   482,   482,    82,    89,
      84,    80,    88,    87,    88,    79,    83,    91,   646,    85,
      83,    89,    87,    81,    80,   448,    51,     1,   656,    37,
      38,    82,    89,    84,    77,    80,    87,    88,    80,    89,
      91,    89,    89,    78,   473,    82,   475,    55,    80,   477,
     478,   479,   453,    61,    62,    63,    64,    65,    66,    67,
      38,    69,    70,   492,   495,    87,   694,    75,    37,    43,
      44,    45,    46,    47,    48,    49,    79,    89,    80,    85,
      80,    89,   513,   484,    58,    82,    55,   516,    87,   490,
       1,    87,    61,    62,    63,    64,    65,    66,    67,    87,
      74,    88,    87,    89,    78,    79,    75,   508,   459,    90,
      84,   462,    87,    90,    87,   546,    87,    87,    85,    87,
      80,    87,   523,    85,    89,   530,    82,    85,    82,    79,
      85,   482,    43,    44,    45,    46,    47,    48,    49,    87,
      87,    17,    87,   494,    87,    84,    84,    58,    84,   580,
      87,    84,    42,    79,   505,   586,    55,   566,    79,   568,
      87,   512,    80,    74,   595,    88,    85,    78,    79,    85,
      85,    85,   573,    84,    84,   604,    43,    44,    45,    46,
      47,    48,    49,   534,    85,    89,    89,   618,   597,   620,
       1,    89,   623,    37,    88,    84,   605,    89,   599,    55,
     601,    88,    57,    78,    78,    20,    55,    74,   639,    80,
      78,    78,    90,   622,    79,   624,    88,    76,    79,    89,
      78,    85,   653,    56,    78,    88,    79,     1,    79,   658,
     659,    55,    43,    44,    45,    46,    47,    48,    49,   670,
      77,    80,    89,    50,     1,   646,    76,    76,    89,   680,
      81,    79,    79,    79,    77,   656,    81,    79,    76,    80,
     691,    77,    79,    74,     1,    79,    79,    78,    79,    43,
      44,    45,    46,    47,    48,    49,   707,    89,    82,    89,
      79,     5,    80,    79,   635,    79,    43,    44,    45,    46,
      47,    48,    49,   694,    79,    70,   727,    55,   215,   142,
      74,   702,   187,   347,    78,    79,    43,    44,    45,    46,
      47,    48,    49,   333,   174,   342,   448,    74,   514,    76,
     751,    78,   355,    70,   454,   356,    70,   678,   679,   253,
     451,   371,   683,   482,   450,   659,   293,    74,   573,   621,
     592,    78,   581,   561,   610,    -1,    -1,    -1,    -1,    -1,
     701,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   714,   715,   716,     1,   718,     3,     4,
       5,    -1,     7,     8,     9,    -1,    11,    12,    13,    14,
      15,    -1,    -1,   734,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    -1,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    -1,    52,    -1,    54,
      55,    -1,    -1,    -1,    -1,    -1,    61,    62,    63,    64,
      65,    66,    67,    68,    -1,    -1,    71,    72,    73,    74,
      75,    -1,    -1,    -1,    -1,    -1,     3,     4,     5,    -1,
       7,     8,     9,    88,    11,    12,    13,    14,    15,    -1,
      -1,    -1,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    -1,    32,    33,    34,    35,    36,
       1,    -1,    39,    40,    41,    42,    43,    44,    45,    46,
      47,    48,    49,    50,    -1,    52,    -1,    54,    55,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    68,    -1,    -1,    71,    72,    73,    74,     1,    -1,
      -1,    -1,    43,    44,    45,    46,    47,    48,    49,     1,
      87,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    14,    15,    16,    -1,    18,    19,    -1,    -1,
      22,    -1,    -1,    74,    -1,    -1,    -1,    78,    -1,    -1,
      43,    44,    45,    46,    47,    48,    49,     1,    -1,    -1,
      -1,    -1,    -1,    -1,     1,    -1,    -1,    -1,    -1,    -1,
      -1,    53,    54,     1,    -1,    -1,    -1,    59,    60,    -1,
       1,    74,    -1,    -1,    -1,    78,    -1,    -1,    -1,    71,
      -1,    -1,    -1,    -1,    -1,    -1,    78,    79,    -1,    43,
      44,    45,    46,    47,    48,    49,    43,    44,    45,    46,
      47,    48,    49,    -1,    -1,    43,    44,    45,    46,    47,
      48,    49,    43,    44,    45,    46,    47,    48,    49,     1,
      74,    -1,    -1,    -1,    78,    -1,     1,    74,    -1,    -1,
      -1,    78,    -1,     1,    -1,    -1,    74,    -1,    -1,    -1,
      78,     1,    -1,    74,    -1,    -1,    -1,    78,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    43,    44,    45,    46,    47,    48,    49,    43,    44,
      45,    46,    47,    48,    49,    43,    44,    45,    46,    47,
      48,    49,    -1,    43,    44,    45,    46,    47,    48,    49,
      -1,    -1,    74,    -1,    -1,    -1,    78,    -1,    -1,    74,
      -1,    -1,    -1,    78,    -1,    -1,    74,    -1,    -1,    -1,
      78,    -1,    -1,    -1,    74,    -1,     3,     4,    78,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
       1,    18,    19,    -1,    -1,    22,     6,     7,     8,     9,
      -1,    11,    12,    13,    14,    15,    -1,    -1,    -1,    -1,
      21,    -1,    23,    24,    25,    26,    27,    28,    37,    38,
      31,    32,    33,    34,    35,    36,    -1,    -1,    -1,    -1,
      -1,    -1,    42,    -1,    -1,    -1,    55,    56,    57,    -1,
      51,    52,    61,    62,    63,    64,    65,    66,    67,    -1,
      -1,    78,    -1,    -1,    -1,    -1,    75,    68,    -1,    -1,
      -1,    72,    73,    -1,    -1,    -1,    76,    -1,    -1,    79,
       3,     4,     5,    -1,     7,     8,     9,    -1,    11,    12,
      13,    14,    15,    -1,    -1,    -1,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    -1,    32,
      33,    34,    35,    36,    37,    38,    39,    40,    41,    42,
      43,    44,    45,    46,    47,    48,    49,    50,    -1,    52,
      -1,    54,    55,     1,    -1,    -1,    -1,    -1,    61,    62,
      63,    64,    65,    66,    67,    68,    -1,    -1,    71,    72,
      73,    74,    75,    21,    -1,    23,    24,    25,    26,    27,
      28,    -1,    -1,    31,    32,    33,    34,    35,    36,    -1,
       3,     4,     5,    -1,     7,     8,     9,    -1,    11,    12,
      13,    14,    15,    51,    52,    -1,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    -1,    32,
      33,    34,    35,    36,    72,    73,    39,    40,    41,    42,
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
      -1,    37,    38,    55,    -1,    -1,    -1,    -1,    -1,    61,
      62,    63,    64,    65,    66,    67,    18,    69,    70,    55,
      37,    38,    39,    75,    -1,    61,    62,    63,    64,    65,
      66,    67,    -1,    -1,    -1,    37,    38,    -1,    55,    75,
      -1,    -1,    -1,    -1,    61,    62,    63,    64,    65,    66,
      67,    -1,    -1,    55,    37,    38,    39,    -1,    75,    61,
      62,    63,    64,    65,    66,    67,    -1,    -1,    85,    37,
      38,    -1,    55,    75,    -1,    -1,    -1,    -1,    61,    62,
      63,    64,    65,    66,    67,    37,    38,    55,    -1,    -1,
      -1,    -1,    75,    61,    62,    63,    64,    65,    66,    67,
      -1,    -1,    -1,    55,    -1,    -1,    -1,    75,    -1,    61,
      62,    63,    64,    65,    66,    67,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    75
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     3,     4,    93,    94,   100,     3,     4,     5,     7,
       8,     9,    11,    12,    13,    14,    15,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    32,
      33,    34,    35,    36,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    52,    54,    55,    68,
      71,    72,    73,    74,    98,    98,     0,    94,    76,    78,
      96,   101,   101,     1,     5,    53,    54,    59,    60,    71,
      95,   102,   103,   104,   170,   207,   208,    76,    42,    98,
      53,    55,    99,    98,    98,    78,    96,   179,     6,     7,
       8,     9,    10,    11,    12,    13,    14,    15,    16,    18,
      19,    22,    78,   100,   123,   124,   140,   143,   144,   145,
     147,   157,   158,   161,   162,   163,    79,     6,     7,     8,
       9,    11,    12,    13,    14,    15,    42,    79,    96,   168,
     102,     1,    88,   172,    78,    99,    77,    58,    84,   177,
      16,    37,    38,    61,    62,    63,    64,    65,    66,    67,
      75,    99,   109,   111,   112,   113,   114,   115,   116,   119,
      37,   125,   125,    88,   125,   111,   164,    88,   129,   129,
     129,   129,    88,   133,   146,    88,   126,    98,    57,   165,
      81,   102,    11,    12,    13,    14,    15,    16,   148,   149,
     150,   151,   152,    96,    97,     1,    21,    23,    24,    25,
      26,    27,    28,    31,    32,    33,    34,    35,    36,    51,
      52,    72,    73,   173,   174,    38,   112,   113,   114,   117,
     118,   171,   102,    77,    78,    84,    88,   177,    79,   180,
     112,   116,    62,    66,    61,    62,    63,    64,   111,    81,
     108,    83,    83,    83,    38,    84,    86,    87,    99,    99,
      99,    68,    99,    80,    30,    51,   130,   135,    98,   110,
     110,   110,   110,    51,    56,    98,   132,   134,    88,   146,
      88,   133,    40,    41,   127,   128,   110,    18,    75,   115,
     119,   155,   156,    79,   129,   129,   129,   129,   146,   126,
      57,   131,    89,    80,   117,    38,    86,    87,   111,    79,
      15,    55,   177,    58,   176,   177,    85,    96,    83,    61,
      62,    56,    57,   105,   106,   107,   119,    83,    86,    87,
      87,    88,   121,   122,   205,    84,    81,    84,    89,    81,
      84,   164,    89,    80,   108,    77,   141,   141,   141,   141,
      98,    89,    80,    89,   110,   110,    89,    80,    78,    87,
      98,    55,    87,    90,   154,    98,    80,    82,    97,    98,
      98,    98,    98,    98,    98,   173,    86,    87,    87,    98,
      79,    80,    85,    89,   177,    99,    80,    82,    98,    87,
      87,    87,   122,   120,   177,   125,   106,   125,   125,   106,
     125,   130,   111,   142,    78,    96,   159,   159,   159,   159,
      89,   134,   141,   141,   127,    17,    79,   136,   138,   139,
      87,    90,   153,   153,    87,    56,    57,   111,   154,   156,
     141,   141,   141,   141,   141,    78,    96,    87,    87,    87,
     108,   177,   176,   177,   177,   122,   106,    85,    87,    87,
      89,   206,    85,    82,    85,    99,    82,    85,    80,     1,
      42,   157,   160,   161,   166,   167,   169,   159,   159,   119,
     139,    79,    87,   119,    87,   159,   159,   159,   159,   159,
     139,    87,    87,    84,   188,    84,    87,    84,    84,    84,
     142,    88,   172,   169,    79,    96,   160,    98,    98,    55,
      79,    87,    39,    85,   119,   178,   181,   186,   186,   125,
     125,   125,    68,    55,    98,   171,    97,    88,   137,   153,
     153,    96,   119,   178,    98,   177,    80,    85,    85,    85,
      85,    85,    89,   188,    98,    89,    96,    98,   177,    88,
      90,   136,    89,   186,    37,     1,    43,    44,    45,    46,
      47,    48,    49,    74,    78,    96,   179,   191,   196,   198,
     188,    89,    56,    57,    99,   175,    98,    84,   202,    88,
     202,    55,   203,   204,    78,    57,   195,   202,    78,   192,
     198,   177,    20,   190,   188,   177,   200,    55,   200,   188,
     205,    80,    78,   198,   193,   198,   179,   200,    43,    44,
      45,    47,    48,    49,    74,   179,   194,   196,   197,    79,
     192,   180,    90,   191,    88,   189,    76,    89,    85,   201,
     177,   204,    79,   192,    79,   192,   177,   201,   202,    88,
     202,    78,   195,   202,    78,   177,    79,   194,    97,    97,
      56,     6,    69,    70,    89,   119,   182,   185,   187,   179,
     200,   202,    78,   198,   206,    79,   180,    78,   198,   177,
      55,   177,   193,   179,   177,   194,   180,    98,    77,    80,
      89,   177,    76,   200,   192,   188,    97,   192,    50,   199,
      76,    89,   201,    79,   177,   201,    79,    97,    81,   119,
     178,   184,   187,   180,   200,    77,    79,    79,    78,   198,
     177,   202,    78,   198,   180,    78,   198,    98,   183,    98,
     177,    81,    98,   201,   200,   199,   192,    76,   177,   192,
      97,   192,   199,    82,    84,    87,    88,    91,    81,    89,
     183,    96,    78,   198,    80,    79,   177,    77,    79,    79,
     183,    98,    56,   183,    82,   183,    82,   192,   200,   201,
     177,   199,    85,    89,    89,    98,    82,    79,   201,    78,
     198,    80,    78,   198,   192,   177,   192,    79,   201,    79,
      78,   198,   192,    79
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    92,    93,    94,    94,    95,    95,    96,    96,    97,
      97,    98,    98,    98,    98,    98,    98,    98,    98,    98,
      98,    98,    98,    98,    98,    98,    98,    98,    98,    98,
      98,    98,    98,    98,    98,    98,    98,    98,    98,    98,
      98,    98,    98,    98,    98,    98,    98,    98,    98,    98,
      98,    98,    98,    98,    98,    98,    98,    98,    98,    99,
      99,    99,   100,   100,   101,   101,   102,   102,   103,   103,
     103,   103,   103,   104,   104,   104,   104,   104,   104,   104,
     104,   104,   104,   104,   104,   104,   104,   105,   105,   105,
     106,   106,   107,   107,   108,   108,   109,   109,   109,   109,
     109,   109,   109,   109,   109,   109,   109,   109,   109,   109,
     109,   110,   111,   112,   112,   112,   113,   114,   114,   115,
     116,   116,   116,   116,   116,   116,   117,   117,   117,   117,
     117,   118,   118,   118,   118,   118,   118,   119,   119,   119,
     119,   119,   119,   120,   121,   122,   122,   123,   124,   125,
     125,   126,   126,   127,   127,   128,   128,   129,   129,   130,
     130,   131,   131,   132,   133,   133,   134,   134,   135,   135,
     136,   136,   137,   137,   138,   139,   139,   140,   140,   140,
     141,   141,   142,   142,   143,   143,   144,   145,   146,   146,
     147,   147,   148,   148,   149,   150,   151,   152,   152,   153,
     153,   154,   154,   154,   154,   155,   155,   155,   155,   155,
     155,   156,   156,   157,   158,   158,   158,   158,   158,   159,
     159,   160,   160,   161,   161,   161,   161,   161,   161,   161,
     162,   162,   162,   162,   162,   163,   163,   163,   163,   164,
     164,   165,   166,   167,   167,   167,   167,   168,   168,   168,
     168,   168,   168,   168,   168,   168,   168,   168,   169,   169,
     169,   170,   170,   171,   172,   172,   172,   173,   173,   174,
     174,   174,   174,   174,   174,   174,   174,   174,   174,   174,
     174,   174,   174,   174,   174,   174,   174,   175,   175,   175,
     176,   176,   176,   177,   177,   177,   177,   177,   177,   178,
     179,   180,   181,   181,   181,   181,   181,   182,   182,   182,
     183,   183,   183,   183,   183,   183,   184,   185,   185,   185,
     186,   186,   187,   187,   188,   188,   189,   189,   190,   190,
     191,   191,   191,   192,   192,   193,   193,   194,   194,   194,
     195,   195,   196,   196,   196,   197,   197,   197,   197,   197,
     197,   197,   197,   197,   197,   197,   197,   198,   198,   198,
     198,   198,   198,   198,   198,   198,   198,   198,   198,   198,
     198,   199,   199,   199,   200,   201,   202,   203,   203,   204,
     204,   205,   206,   207,   208
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
       2,     2,     8,     5,     5,     2,     2,     2,     2,     2,
       2,     2,     2,     1,     1,     1,     1,     1,     1,     1,
       1,     3,     0,     1,     0,     3,     1,     1,     1,     1,
       2,     2,     3,     3,     2,     2,     2,     1,     1,     2,
       1,     2,     2,     1,     2,     1,     2,     2,     2,     8,
       1,     1,     1,     1,     2,     2,     1,     1,     1,     2,
       2,     6,     5,     4,     3,     2,     1,     6,     5,     4,
       3,     2,     1,     1,     3,     0,     2,     4,     6,     0,
       1,     0,     3,     1,     3,     1,     1,     0,     3,     1,
       3,     0,     1,     1,     0,     3,     1,     3,     1,     1,
       0,     1,     0,     2,     5,     1,     2,     3,     5,     6,
       0,     2,     1,     3,     5,     5,     5,     5,     4,     3,
       6,     6,     5,     5,     5,     5,     5,     4,     7,     0,
       2,     0,     2,     2,     2,     6,     6,     3,     3,     2,
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
#line 197 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.modlist) = (yyvsp[0].modlist); modlist = (yyvsp[0].modlist); }
#line 2294 "y.tab.c" /* yacc.c:1661  */
    break;

  case 3:
#line 201 "xi-grammar.y" /* yacc.c:1661  */
    {
		  (yyval.modlist) = 0;
		}
#line 2302 "y.tab.c" /* yacc.c:1661  */
    break;

  case 4:
#line 205 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.modlist) = new AstChildren<Module>(lineno, (yyvsp[-1].module), (yyvsp[0].modlist)); }
#line 2308 "y.tab.c" /* yacc.c:1661  */
    break;

  case 5:
#line 209 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 0; }
#line 2314 "y.tab.c" /* yacc.c:1661  */
    break;

  case 6:
#line 211 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 1; }
#line 2320 "y.tab.c" /* yacc.c:1661  */
    break;

  case 7:
#line 215 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 1; }
#line 2326 "y.tab.c" /* yacc.c:1661  */
    break;

  case 8:
#line 217 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 2; }
#line 2332 "y.tab.c" /* yacc.c:1661  */
    break;

  case 9:
#line 221 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 0; }
#line 2338 "y.tab.c" /* yacc.c:1661  */
    break;

  case 10:
#line 223 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 1; }
#line 2344 "y.tab.c" /* yacc.c:1661  */
    break;

  case 11:
#line 228 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 2350 "y.tab.c" /* yacc.c:1661  */
    break;

  case 12:
#line 229 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(MODULE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2356 "y.tab.c" /* yacc.c:1661  */
    break;

  case 13:
#line 230 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(MAINMODULE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2362 "y.tab.c" /* yacc.c:1661  */
    break;

  case 14:
#line 231 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(EXTERN, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2368 "y.tab.c" /* yacc.c:1661  */
    break;

  case 15:
#line 233 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(INITCALL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2374 "y.tab.c" /* yacc.c:1661  */
    break;

  case 16:
#line 234 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(INITNODE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2380 "y.tab.c" /* yacc.c:1661  */
    break;

  case 17:
#line 235 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(INITPROC, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2386 "y.tab.c" /* yacc.c:1661  */
    break;

  case 18:
#line 237 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(CHARE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2392 "y.tab.c" /* yacc.c:1661  */
    break;

  case 19:
#line 238 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(MAINCHARE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2398 "y.tab.c" /* yacc.c:1661  */
    break;

  case 20:
#line 239 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(GROUP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2404 "y.tab.c" /* yacc.c:1661  */
    break;

  case 21:
#line 240 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(NODEGROUP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2410 "y.tab.c" /* yacc.c:1661  */
    break;

  case 22:
#line 241 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(ARRAY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2416 "y.tab.c" /* yacc.c:1661  */
    break;

  case 23:
#line 245 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(INCLUDE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2422 "y.tab.c" /* yacc.c:1661  */
    break;

  case 24:
#line 246 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(STACKSIZE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2428 "y.tab.c" /* yacc.c:1661  */
    break;

  case 25:
#line 247 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(THREADED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2434 "y.tab.c" /* yacc.c:1661  */
    break;

  case 26:
#line 248 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(TEMPLATE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2440 "y.tab.c" /* yacc.c:1661  */
    break;

  case 27:
#line 249 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(SYNC, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2446 "y.tab.c" /* yacc.c:1661  */
    break;

  case 28:
#line 250 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(IGET, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2452 "y.tab.c" /* yacc.c:1661  */
    break;

  case 29:
#line 251 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(EXCLUSIVE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2458 "y.tab.c" /* yacc.c:1661  */
    break;

  case 30:
#line 252 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(IMMEDIATE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2464 "y.tab.c" /* yacc.c:1661  */
    break;

  case 31:
#line 253 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(SKIPSCHED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2470 "y.tab.c" /* yacc.c:1661  */
    break;

  case 32:
#line 254 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(NOCOPY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2476 "y.tab.c" /* yacc.c:1661  */
    break;

  case 33:
#line 255 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(INLINE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2482 "y.tab.c" /* yacc.c:1661  */
    break;

  case 34:
#line 256 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(VIRTUAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2488 "y.tab.c" /* yacc.c:1661  */
    break;

  case 35:
#line 257 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(MIGRATABLE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2494 "y.tab.c" /* yacc.c:1661  */
    break;

  case 36:
#line 258 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(CREATEHERE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2500 "y.tab.c" /* yacc.c:1661  */
    break;

  case 37:
#line 259 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(CREATEHOME, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2506 "y.tab.c" /* yacc.c:1661  */
    break;

  case 38:
#line 260 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(NOKEEP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2512 "y.tab.c" /* yacc.c:1661  */
    break;

  case 39:
#line 261 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(NOTRACE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2518 "y.tab.c" /* yacc.c:1661  */
    break;

  case 40:
#line 262 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(APPWORK, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2524 "y.tab.c" /* yacc.c:1661  */
    break;

  case 41:
#line 265 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(PACKED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2530 "y.tab.c" /* yacc.c:1661  */
    break;

  case 42:
#line 266 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(VARSIZE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2536 "y.tab.c" /* yacc.c:1661  */
    break;

  case 43:
#line 267 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(ENTRY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2542 "y.tab.c" /* yacc.c:1661  */
    break;

  case 44:
#line 268 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(FOR, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2548 "y.tab.c" /* yacc.c:1661  */
    break;

  case 45:
#line 269 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(FORALL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2554 "y.tab.c" /* yacc.c:1661  */
    break;

  case 46:
#line 270 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(WHILE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2560 "y.tab.c" /* yacc.c:1661  */
    break;

  case 47:
#line 271 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(WHEN, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2566 "y.tab.c" /* yacc.c:1661  */
    break;

  case 48:
#line 272 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(OVERLAP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2572 "y.tab.c" /* yacc.c:1661  */
    break;

  case 49:
#line 273 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(SERIAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2578 "y.tab.c" /* yacc.c:1661  */
    break;

  case 50:
#line 274 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(IF, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2584 "y.tab.c" /* yacc.c:1661  */
    break;

  case 51:
#line 275 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(ELSE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2590 "y.tab.c" /* yacc.c:1661  */
    break;

  case 52:
#line 277 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(LOCAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2596 "y.tab.c" /* yacc.c:1661  */
    break;

  case 53:
#line 279 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(USING, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2602 "y.tab.c" /* yacc.c:1661  */
    break;

  case 54:
#line 280 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(ACCEL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2608 "y.tab.c" /* yacc.c:1661  */
    break;

  case 55:
#line 283 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(ACCELBLOCK, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2614 "y.tab.c" /* yacc.c:1661  */
    break;

  case 56:
#line 284 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(MEMCRITICAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2620 "y.tab.c" /* yacc.c:1661  */
    break;

  case 57:
#line 285 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(REDUCTIONTARGET, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2626 "y.tab.c" /* yacc.c:1661  */
    break;

  case 58:
#line 286 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(CASE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2632 "y.tab.c" /* yacc.c:1661  */
    break;

  case 59:
#line 291 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 2638 "y.tab.c" /* yacc.c:1661  */
    break;

  case 60:
#line 293 "xi-grammar.y" /* yacc.c:1661  */
    {
		  char *tmp = new char[strlen((yyvsp[-3].strval))+strlen((yyvsp[0].strval))+3];
		  sprintf(tmp,"%s::%s", (yyvsp[-3].strval), (yyvsp[0].strval));
		  (yyval.strval) = tmp;
		}
#line 2648 "y.tab.c" /* yacc.c:1661  */
    break;

  case 61:
#line 299 "xi-grammar.y" /* yacc.c:1661  */
    {
		  char *tmp = new char[strlen((yyvsp[-3].strval))+5+3];
		  sprintf(tmp,"%s::array", (yyvsp[-3].strval));
		  (yyval.strval) = tmp;
		}
#line 2658 "y.tab.c" /* yacc.c:1661  */
    break;

  case 62:
#line 306 "xi-grammar.y" /* yacc.c:1661  */
    {
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist));
		}
#line 2666 "y.tab.c" /* yacc.c:1661  */
    break;

  case 63:
#line 310 "xi-grammar.y" /* yacc.c:1661  */
    {
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist));
		    (yyval.module)->setMain();
		}
#line 2675 "y.tab.c" /* yacc.c:1661  */
    break;

  case 64:
#line 317 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.conslist) = 0; }
#line 2681 "y.tab.c" /* yacc.c:1661  */
    break;

  case 65:
#line 319 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.conslist) = (yyvsp[-2].conslist); }
#line 2687 "y.tab.c" /* yacc.c:1661  */
    break;

  case 66:
#line 323 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.conslist) = 0; }
#line 2693 "y.tab.c" /* yacc.c:1661  */
    break;

  case 67:
#line 325 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.conslist) = new ConstructList(lineno, (yyvsp[-1].construct), (yyvsp[0].conslist)); }
#line 2699 "y.tab.c" /* yacc.c:1661  */
    break;

  case 68:
#line 329 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.construct) = new UsingScope((yyvsp[0].strval), false); }
#line 2705 "y.tab.c" /* yacc.c:1661  */
    break;

  case 69:
#line 331 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.construct) = new UsingScope((yyvsp[0].strval), true); }
#line 2711 "y.tab.c" /* yacc.c:1661  */
    break;

  case 70:
#line 333 "xi-grammar.y" /* yacc.c:1661  */
    { (yyvsp[0].member)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].member); }
#line 2717 "y.tab.c" /* yacc.c:1661  */
    break;

  case 71:
#line 335 "xi-grammar.y" /* yacc.c:1661  */
    { (yyvsp[0].message)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].message); }
#line 2723 "y.tab.c" /* yacc.c:1661  */
    break;

  case 72:
#line 337 "xi-grammar.y" /* yacc.c:1661  */
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
#line 2738 "y.tab.c" /* yacc.c:1661  */
    break;

  case 73:
#line 350 "xi-grammar.y" /* yacc.c:1661  */
    { if((yyvsp[-2].conslist)) (yyvsp[-2].conslist)->recurse<int&>((yyvsp[-4].intval), &Construct::setExtern); (yyval.construct) = (yyvsp[-2].conslist); }
#line 2744 "y.tab.c" /* yacc.c:1661  */
    break;

  case 74:
#line 352 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.construct) = new Scope((yyvsp[-3].strval), (yyvsp[-1].conslist)); }
#line 2750 "y.tab.c" /* yacc.c:1661  */
    break;

  case 75:
#line 354 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.construct) = (yyvsp[-1].construct); }
#line 2756 "y.tab.c" /* yacc.c:1661  */
    break;

  case 76:
#line 356 "xi-grammar.y" /* yacc.c:1661  */
    {
          ERROR("preceding construct must be semicolon terminated",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 2766 "y.tab.c" /* yacc.c:1661  */
    break;

  case 77:
#line 362 "xi-grammar.y" /* yacc.c:1661  */
    { (yyvsp[0].module)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].module); }
#line 2772 "y.tab.c" /* yacc.c:1661  */
    break;

  case 78:
#line 364 "xi-grammar.y" /* yacc.c:1661  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2778 "y.tab.c" /* yacc.c:1661  */
    break;

  case 79:
#line 366 "xi-grammar.y" /* yacc.c:1661  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2784 "y.tab.c" /* yacc.c:1661  */
    break;

  case 80:
#line 368 "xi-grammar.y" /* yacc.c:1661  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2790 "y.tab.c" /* yacc.c:1661  */
    break;

  case 81:
#line 370 "xi-grammar.y" /* yacc.c:1661  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2796 "y.tab.c" /* yacc.c:1661  */
    break;

  case 82:
#line 372 "xi-grammar.y" /* yacc.c:1661  */
    { (yyvsp[0].templat)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].templat); }
#line 2802 "y.tab.c" /* yacc.c:1661  */
    break;

  case 83:
#line 374 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.construct) = NULL; }
#line 2808 "y.tab.c" /* yacc.c:1661  */
    break;

  case 84:
#line 376 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.construct) = NULL; }
#line 2814 "y.tab.c" /* yacc.c:1661  */
    break;

  case 85:
#line 378 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.construct) = (yyvsp[0].accelBlock); }
#line 2820 "y.tab.c" /* yacc.c:1661  */
    break;

  case 86:
#line 380 "xi-grammar.y" /* yacc.c:1661  */
    {
          ERROR("invalid construct",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 2830 "y.tab.c" /* yacc.c:1661  */
    break;

  case 87:
#line 388 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tparam) = new TParamType((yyvsp[0].type)); }
#line 2836 "y.tab.c" /* yacc.c:1661  */
    break;

  case 88:
#line 390 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
#line 2842 "y.tab.c" /* yacc.c:1661  */
    break;

  case 89:
#line 392 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
#line 2848 "y.tab.c" /* yacc.c:1661  */
    break;

  case 90:
#line 396 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tparlist) = new TParamList((yyvsp[0].tparam)); }
#line 2854 "y.tab.c" /* yacc.c:1661  */
    break;

  case 91:
#line 398 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tparlist) = new TParamList((yyvsp[-2].tparam), (yyvsp[0].tparlist)); }
#line 2860 "y.tab.c" /* yacc.c:1661  */
    break;

  case 92:
#line 402 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tparlist) = new TParamList(0); }
#line 2866 "y.tab.c" /* yacc.c:1661  */
    break;

  case 93:
#line 404 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tparlist) = (yyvsp[0].tparlist); }
#line 2872 "y.tab.c" /* yacc.c:1661  */
    break;

  case 94:
#line 408 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tparlist) = 0; }
#line 2878 "y.tab.c" /* yacc.c:1661  */
    break;

  case 95:
#line 410 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tparlist) = (yyvsp[-1].tparlist); }
#line 2884 "y.tab.c" /* yacc.c:1661  */
    break;

  case 96:
#line 414 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("int"); }
#line 2890 "y.tab.c" /* yacc.c:1661  */
    break;

  case 97:
#line 416 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("long"); }
#line 2896 "y.tab.c" /* yacc.c:1661  */
    break;

  case 98:
#line 418 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("short"); }
#line 2902 "y.tab.c" /* yacc.c:1661  */
    break;

  case 99:
#line 420 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("char"); }
#line 2908 "y.tab.c" /* yacc.c:1661  */
    break;

  case 100:
#line 422 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("unsigned int"); }
#line 2914 "y.tab.c" /* yacc.c:1661  */
    break;

  case 101:
#line 424 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("unsigned long"); }
#line 2920 "y.tab.c" /* yacc.c:1661  */
    break;

  case 102:
#line 426 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("unsigned long"); }
#line 2926 "y.tab.c" /* yacc.c:1661  */
    break;

  case 103:
#line 428 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("unsigned long long"); }
#line 2932 "y.tab.c" /* yacc.c:1661  */
    break;

  case 104:
#line 430 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("unsigned short"); }
#line 2938 "y.tab.c" /* yacc.c:1661  */
    break;

  case 105:
#line 432 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("unsigned char"); }
#line 2944 "y.tab.c" /* yacc.c:1661  */
    break;

  case 106:
#line 434 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("long long"); }
#line 2950 "y.tab.c" /* yacc.c:1661  */
    break;

  case 107:
#line 436 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("float"); }
#line 2956 "y.tab.c" /* yacc.c:1661  */
    break;

  case 108:
#line 438 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("double"); }
#line 2962 "y.tab.c" /* yacc.c:1661  */
    break;

  case 109:
#line 440 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("long double"); }
#line 2968 "y.tab.c" /* yacc.c:1661  */
    break;

  case 110:
#line 442 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("void"); }
#line 2974 "y.tab.c" /* yacc.c:1661  */
    break;

  case 111:
#line 445 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.ntype) = new NamedType((yyvsp[-1].strval),(yyvsp[0].tparlist)); }
#line 2980 "y.tab.c" /* yacc.c:1661  */
    break;

  case 112:
#line 446 "xi-grammar.y" /* yacc.c:1661  */
    {
                    const char* basename, *scope;
                    splitScopedName((yyvsp[-1].strval), &scope, &basename);
                    (yyval.ntype) = new NamedType(basename, (yyvsp[0].tparlist), scope);
                }
#line 2990 "y.tab.c" /* yacc.c:1661  */
    break;

  case 113:
#line 454 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].type); }
#line 2996 "y.tab.c" /* yacc.c:1661  */
    break;

  case 114:
#line 456 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new TypenameType((yyvsp[0].ntype)); }
#line 3002 "y.tab.c" /* yacc.c:1661  */
    break;

  case 115:
#line 458 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].ntype); }
#line 3008 "y.tab.c" /* yacc.c:1661  */
    break;

  case 116:
#line 462 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.ptype) = new PtrType((yyvsp[-1].type)); }
#line 3014 "y.tab.c" /* yacc.c:1661  */
    break;

  case 117:
#line 466 "xi-grammar.y" /* yacc.c:1661  */
    { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
#line 3020 "y.tab.c" /* yacc.c:1661  */
    break;

  case 118:
#line 468 "xi-grammar.y" /* yacc.c:1661  */
    { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
#line 3026 "y.tab.c" /* yacc.c:1661  */
    break;

  case 119:
#line 472 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.ftype) = new FuncType((yyvsp[-7].type), (yyvsp[-4].strval), (yyvsp[-1].plist)); }
#line 3032 "y.tab.c" /* yacc.c:1661  */
    break;

  case 120:
#line 476 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3038 "y.tab.c" /* yacc.c:1661  */
    break;

  case 121:
#line 478 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3044 "y.tab.c" /* yacc.c:1661  */
    break;

  case 122:
#line 480 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3050 "y.tab.c" /* yacc.c:1661  */
    break;

  case 123:
#line 482 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].ftype); }
#line 3056 "y.tab.c" /* yacc.c:1661  */
    break;

  case 124:
#line 484 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new ConstType((yyvsp[0].type)); }
#line 3062 "y.tab.c" /* yacc.c:1661  */
    break;

  case 125:
#line 486 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new ConstType((yyvsp[-1].type)); }
#line 3068 "y.tab.c" /* yacc.c:1661  */
    break;

  case 126:
#line 490 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3074 "y.tab.c" /* yacc.c:1661  */
    break;

  case 127:
#line 492 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3080 "y.tab.c" /* yacc.c:1661  */
    break;

  case 128:
#line 494 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3086 "y.tab.c" /* yacc.c:1661  */
    break;

  case 129:
#line 496 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new ConstType((yyvsp[0].type)); }
#line 3092 "y.tab.c" /* yacc.c:1661  */
    break;

  case 130:
#line 498 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new ConstType((yyvsp[-1].type)); }
#line 3098 "y.tab.c" /* yacc.c:1661  */
    break;

  case 131:
#line 502 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new EllipsisType(new RValueReferenceType((yyvsp[-5].type))); }
#line 3104 "y.tab.c" /* yacc.c:1661  */
    break;

  case 132:
#line 504 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new EllipsisType(new ReferenceType((yyvsp[-4].type))); }
#line 3110 "y.tab.c" /* yacc.c:1661  */
    break;

  case 133:
#line 506 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new EllipsisType((yyvsp[-3].type)); }
#line 3116 "y.tab.c" /* yacc.c:1661  */
    break;

  case 134:
#line 508 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new RValueReferenceType((yyvsp[-2].type)); }
#line 3122 "y.tab.c" /* yacc.c:1661  */
    break;

  case 135:
#line 510 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new ReferenceType((yyvsp[-1].type)); }
#line 3128 "y.tab.c" /* yacc.c:1661  */
    break;

  case 136:
#line 512 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3134 "y.tab.c" /* yacc.c:1661  */
    break;

  case 137:
#line 516 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new EllipsisType(new RValueReferenceType((yyvsp[-5].type))); }
#line 3140 "y.tab.c" /* yacc.c:1661  */
    break;

  case 138:
#line 518 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new EllipsisType(new ReferenceType((yyvsp[-4].type))); }
#line 3146 "y.tab.c" /* yacc.c:1661  */
    break;

  case 139:
#line 520 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new EllipsisType((yyvsp[-3].type)); }
#line 3152 "y.tab.c" /* yacc.c:1661  */
    break;

  case 140:
#line 522 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new RValueReferenceType((yyvsp[-2].type)); }
#line 3158 "y.tab.c" /* yacc.c:1661  */
    break;

  case 141:
#line 524 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new ReferenceType((yyvsp[-1].type)); }
#line 3164 "y.tab.c" /* yacc.c:1661  */
    break;

  case 142:
#line 526 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3170 "y.tab.c" /* yacc.c:1661  */
    break;

  case 143:
#line 530 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 3176 "y.tab.c" /* yacc.c:1661  */
    break;

  case 144:
#line 534 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.val) = (yyvsp[-1].val); }
#line 3182 "y.tab.c" /* yacc.c:1661  */
    break;

  case 145:
#line 538 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.vallist) = 0; }
#line 3188 "y.tab.c" /* yacc.c:1661  */
    break;

  case 146:
#line 540 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.vallist) = new ValueList((yyvsp[-1].val), (yyvsp[0].vallist)); }
#line 3194 "y.tab.c" /* yacc.c:1661  */
    break;

  case 147:
#line 544 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].vallist)); }
#line 3200 "y.tab.c" /* yacc.c:1661  */
    break;

  case 148:
#line 548 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[-3].type), (yyvsp[-1].strval), (yyvsp[0].vallist), 1); }
#line 3206 "y.tab.c" /* yacc.c:1661  */
    break;

  case 149:
#line 552 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 0;}
#line 3212 "y.tab.c" /* yacc.c:1661  */
    break;

  case 150:
#line 554 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 0;}
#line 3218 "y.tab.c" /* yacc.c:1661  */
    break;

  case 151:
#line 558 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 0; }
#line 3224 "y.tab.c" /* yacc.c:1661  */
    break;

  case 152:
#line 560 "xi-grammar.y" /* yacc.c:1661  */
    {
		  /*
		  printf("Warning: Message attributes are being phased out.\n");
		  printf("Warning: Please remove them from interface files.\n");
		  */
		  (yyval.intval) = (yyvsp[-1].intval);
		}
#line 3236 "y.tab.c" /* yacc.c:1661  */
    break;

  case 153:
#line 570 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = (yyvsp[0].intval); }
#line 3242 "y.tab.c" /* yacc.c:1661  */
    break;

  case 154:
#line 572 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = (yyvsp[-2].intval) | (yyvsp[0].intval); }
#line 3248 "y.tab.c" /* yacc.c:1661  */
    break;

  case 155:
#line 576 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 0; }
#line 3254 "y.tab.c" /* yacc.c:1661  */
    break;

  case 156:
#line 578 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 0; }
#line 3260 "y.tab.c" /* yacc.c:1661  */
    break;

  case 157:
#line 582 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = 0; }
#line 3266 "y.tab.c" /* yacc.c:1661  */
    break;

  case 158:
#line 584 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = (yyvsp[-1].cattr); }
#line 3272 "y.tab.c" /* yacc.c:1661  */
    break;

  case 159:
#line 588 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = (yyvsp[0].cattr); }
#line 3278 "y.tab.c" /* yacc.c:1661  */
    break;

  case 160:
#line 590 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
#line 3284 "y.tab.c" /* yacc.c:1661  */
    break;

  case 161:
#line 594 "xi-grammar.y" /* yacc.c:1661  */
    { python_doc = NULL; (yyval.intval) = 0; }
#line 3290 "y.tab.c" /* yacc.c:1661  */
    break;

  case 162:
#line 596 "xi-grammar.y" /* yacc.c:1661  */
    { python_doc = (yyvsp[0].strval); (yyval.intval) = 0; }
#line 3296 "y.tab.c" /* yacc.c:1661  */
    break;

  case 163:
#line 600 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = Chare::CPYTHON; }
#line 3302 "y.tab.c" /* yacc.c:1661  */
    break;

  case 164:
#line 604 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = 0; }
#line 3308 "y.tab.c" /* yacc.c:1661  */
    break;

  case 165:
#line 606 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = (yyvsp[-1].cattr); }
#line 3314 "y.tab.c" /* yacc.c:1661  */
    break;

  case 166:
#line 610 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = (yyvsp[0].cattr); }
#line 3320 "y.tab.c" /* yacc.c:1661  */
    break;

  case 167:
#line 612 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
#line 3326 "y.tab.c" /* yacc.c:1661  */
    break;

  case 168:
#line 616 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = Chare::CMIGRATABLE; }
#line 3332 "y.tab.c" /* yacc.c:1661  */
    break;

  case 169:
#line 618 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = Chare::CPYTHON; }
#line 3338 "y.tab.c" /* yacc.c:1661  */
    break;

  case 170:
#line 622 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 0; }
#line 3344 "y.tab.c" /* yacc.c:1661  */
    break;

  case 171:
#line 624 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 1; }
#line 3350 "y.tab.c" /* yacc.c:1661  */
    break;

  case 172:
#line 627 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 0; }
#line 3356 "y.tab.c" /* yacc.c:1661  */
    break;

  case 173:
#line 629 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 1; }
#line 3362 "y.tab.c" /* yacc.c:1661  */
    break;

  case 174:
#line 632 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.mv) = new MsgVar((yyvsp[-3].type), (yyvsp[-2].strval), (yyvsp[-4].intval), (yyvsp[-1].intval)); }
#line 3368 "y.tab.c" /* yacc.c:1661  */
    break;

  case 175:
#line 636 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.mvlist) = new MsgVarList((yyvsp[0].mv)); }
#line 3374 "y.tab.c" /* yacc.c:1661  */
    break;

  case 176:
#line 638 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.mvlist) = new MsgVarList((yyvsp[-1].mv), (yyvsp[0].mvlist)); }
#line 3380 "y.tab.c" /* yacc.c:1661  */
    break;

  case 177:
#line 642 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.message) = new Message(lineno, (yyvsp[0].ntype)); }
#line 3386 "y.tab.c" /* yacc.c:1661  */
    break;

  case 178:
#line 644 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.message) = new Message(lineno, (yyvsp[-2].ntype)); }
#line 3392 "y.tab.c" /* yacc.c:1661  */
    break;

  case 179:
#line 646 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.message) = new Message(lineno, (yyvsp[-3].ntype), (yyvsp[-1].mvlist)); }
#line 3398 "y.tab.c" /* yacc.c:1661  */
    break;

  case 180:
#line 650 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.typelist) = 0; }
#line 3404 "y.tab.c" /* yacc.c:1661  */
    break;

  case 181:
#line 652 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.typelist) = (yyvsp[0].typelist); }
#line 3410 "y.tab.c" /* yacc.c:1661  */
    break;

  case 182:
#line 656 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.typelist) = new TypeList((yyvsp[0].ntype)); }
#line 3416 "y.tab.c" /* yacc.c:1661  */
    break;

  case 183:
#line 658 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.typelist) = new TypeList((yyvsp[-2].ntype), (yyvsp[0].typelist)); }
#line 3422 "y.tab.c" /* yacc.c:1661  */
    break;

  case 184:
#line 662 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr)|Chare::CCHARE, (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3428 "y.tab.c" /* yacc.c:1661  */
    break;

  case 185:
#line 664 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3434 "y.tab.c" /* yacc.c:1661  */
    break;

  case 186:
#line 668 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3440 "y.tab.c" /* yacc.c:1661  */
    break;

  case 187:
#line 672 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.chare) = new NodeGroup(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3446 "y.tab.c" /* yacc.c:1661  */
    break;

  case 188:
#line 676 "xi-grammar.y" /* yacc.c:1661  */
    {/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",(yyvsp[-2].strval));
			(yyval.ntype) = new NamedType(buf);
		}
#line 3456 "y.tab.c" /* yacc.c:1661  */
    break;

  case 189:
#line 682 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.ntype) = new NamedType((yyvsp[-1].strval)); }
#line 3462 "y.tab.c" /* yacc.c:1661  */
    break;

  case 190:
#line 686 "xi-grammar.y" /* yacc.c:1661  */
    {  (yyval.chare) = new Array(lineno, (yyvsp[-4].cattr), (yyvsp[-3].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3468 "y.tab.c" /* yacc.c:1661  */
    break;

  case 191:
#line 688 "xi-grammar.y" /* yacc.c:1661  */
    {  (yyval.chare) = new Array(lineno, (yyvsp[-3].cattr), (yyvsp[-4].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3474 "y.tab.c" /* yacc.c:1661  */
    break;

  case 192:
#line 692 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr)|Chare::CCHARE, new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist));}
#line 3480 "y.tab.c" /* yacc.c:1661  */
    break;

  case 193:
#line 694 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3486 "y.tab.c" /* yacc.c:1661  */
    break;

  case 194:
#line 698 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3492 "y.tab.c" /* yacc.c:1661  */
    break;

  case 195:
#line 702 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.chare) = new NodeGroup( lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3498 "y.tab.c" /* yacc.c:1661  */
    break;

  case 196:
#line 706 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.chare) = new Array( lineno, 0, (yyvsp[-3].ntype), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3504 "y.tab.c" /* yacc.c:1661  */
    break;

  case 197:
#line 710 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-1].strval))); }
#line 3510 "y.tab.c" /* yacc.c:1661  */
    break;

  case 198:
#line 712 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-4].strval)), (yyvsp[-2].mvlist)); }
#line 3516 "y.tab.c" /* yacc.c:1661  */
    break;

  case 199:
#line 716 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = 0; }
#line 3522 "y.tab.c" /* yacc.c:1661  */
    break;

  case 200:
#line 718 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3528 "y.tab.c" /* yacc.c:1661  */
    break;

  case 201:
#line 722 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = 0; }
#line 3534 "y.tab.c" /* yacc.c:1661  */
    break;

  case 202:
#line 724 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3540 "y.tab.c" /* yacc.c:1661  */
    break;

  case 203:
#line 726 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3546 "y.tab.c" /* yacc.c:1661  */
    break;

  case 204:
#line 728 "xi-grammar.y" /* yacc.c:1661  */
    {
		  XStr typeStr;
		  (yyvsp[0].ntype)->print(typeStr);
		  char *tmp = strdup(typeStr.get_string());
		  (yyval.strval) = tmp;
		}
#line 3557 "y.tab.c" /* yacc.c:1661  */
    break;

  case 205:
#line 737 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tvar) = new TTypeEllipsis(new NamedEllipsisType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3563 "y.tab.c" /* yacc.c:1661  */
    break;

  case 206:
#line 739 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tvar) = new TTypeEllipsis(new NamedEllipsisType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3569 "y.tab.c" /* yacc.c:1661  */
    break;

  case 207:
#line 741 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tvar) = new TType(new NamedType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3575 "y.tab.c" /* yacc.c:1661  */
    break;

  case 208:
#line 743 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tvar) = new TType(new NamedType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3581 "y.tab.c" /* yacc.c:1661  */
    break;

  case 209:
#line 745 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tvar) = new TFunc((yyvsp[-1].ftype), (yyvsp[0].strval)); }
#line 3587 "y.tab.c" /* yacc.c:1661  */
    break;

  case 210:
#line 747 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tvar) = new TName((yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].strval)); }
#line 3593 "y.tab.c" /* yacc.c:1661  */
    break;

  case 211:
#line 751 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tvarlist) = new TVarList((yyvsp[0].tvar)); }
#line 3599 "y.tab.c" /* yacc.c:1661  */
    break;

  case 212:
#line 753 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tvarlist) = new TVarList((yyvsp[-2].tvar), (yyvsp[0].tvarlist)); }
#line 3605 "y.tab.c" /* yacc.c:1661  */
    break;

  case 213:
#line 757 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tvarlist) = (yyvsp[-1].tvarlist); }
#line 3611 "y.tab.c" /* yacc.c:1661  */
    break;

  case 214:
#line 761 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3617 "y.tab.c" /* yacc.c:1661  */
    break;

  case 215:
#line 763 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3623 "y.tab.c" /* yacc.c:1661  */
    break;

  case 216:
#line 765 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3629 "y.tab.c" /* yacc.c:1661  */
    break;

  case 217:
#line 767 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3635 "y.tab.c" /* yacc.c:1661  */
    break;

  case 218:
#line 769 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].message)); (yyvsp[0].message)->setTemplate((yyval.templat)); }
#line 3641 "y.tab.c" /* yacc.c:1661  */
    break;

  case 219:
#line 773 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.mbrlist) = 0; }
#line 3647 "y.tab.c" /* yacc.c:1661  */
    break;

  case 220:
#line 775 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.mbrlist) = (yyvsp[-2].mbrlist); }
#line 3653 "y.tab.c" /* yacc.c:1661  */
    break;

  case 221:
#line 779 "xi-grammar.y" /* yacc.c:1661  */
    {
                  if (!connectEntries.empty()) {
                    (yyval.mbrlist) = new AstChildren<Member>(connectEntries);
		  } else {
		    (yyval.mbrlist) = 0;
                  }
		}
#line 3665 "y.tab.c" /* yacc.c:1661  */
    break;

  case 222:
#line 787 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.mbrlist) = new AstChildren<Member>(-1, (yyvsp[-1].member), (yyvsp[0].mbrlist)); }
#line 3671 "y.tab.c" /* yacc.c:1661  */
    break;

  case 223:
#line 791 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = (yyvsp[0].readonly); }
#line 3677 "y.tab.c" /* yacc.c:1661  */
    break;

  case 224:
#line 793 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = (yyvsp[0].readonly); }
#line 3683 "y.tab.c" /* yacc.c:1661  */
    break;

  case 226:
#line 796 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = (yyvsp[0].member); }
#line 3689 "y.tab.c" /* yacc.c:1661  */
    break;

  case 227:
#line 798 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = (yyvsp[0].pupable); }
#line 3695 "y.tab.c" /* yacc.c:1661  */
    break;

  case 228:
#line 800 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = (yyvsp[0].includeFile); }
#line 3701 "y.tab.c" /* yacc.c:1661  */
    break;

  case 229:
#line 802 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = new ClassDeclaration(lineno,(yyvsp[0].strval)); }
#line 3707 "y.tab.c" /* yacc.c:1661  */
    break;

  case 230:
#line 806 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1); }
#line 3713 "y.tab.c" /* yacc.c:1661  */
    break;

  case 231:
#line 808 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1); }
#line 3719 "y.tab.c" /* yacc.c:1661  */
    break;

  case 232:
#line 810 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[-6].strval)) + '<' +
					    ((yyvsp[-4].tparlist))->to_string() + '>').c_str()),
				    1);
		}
#line 3729 "y.tab.c" /* yacc.c:1661  */
    break;

  case 233:
#line 816 "xi-grammar.y" /* yacc.c:1661  */
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[-2]).first_column, (yylsp[-2]).last_column, (yylsp[-2]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1);
		}
#line 3739 "y.tab.c" /* yacc.c:1661  */
    break;

  case 234:
#line 822 "xi-grammar.y" /* yacc.c:1661  */
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[-5]).first_column, (yylsp[-5]).last_column, (yylsp[-5]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1);
		}
#line 3749 "y.tab.c" /* yacc.c:1661  */
    break;

  case 235:
#line 831 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 0); }
#line 3755 "y.tab.c" /* yacc.c:1661  */
    break;

  case 236:
#line 833 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 0); }
#line 3761 "y.tab.c" /* yacc.c:1661  */
    break;

  case 237:
#line 835 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[-6].strval)) + '<' +
					    ((yyvsp[-4].tparlist))->to_string() + '>').c_str()),
				    0);
		}
#line 3771 "y.tab.c" /* yacc.c:1661  */
    break;

  case 238:
#line 841 "xi-grammar.y" /* yacc.c:1661  */
    {
                  InitCall* rtn = new InitCall(lineno, (yyvsp[-3].strval), 0);
                  rtn->setAccel();
                  (yyval.member) = rtn;
		}
#line 3781 "y.tab.c" /* yacc.c:1661  */
    break;

  case 239:
#line 849 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[0].ntype),0); }
#line 3787 "y.tab.c" /* yacc.c:1661  */
    break;

  case 240:
#line 851 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[-2].ntype),(yyvsp[0].pupable)); }
#line 3793 "y.tab.c" /* yacc.c:1661  */
    break;

  case 241:
#line 854 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.includeFile) = new IncludeFile(lineno,(yyvsp[0].strval)); }
#line 3799 "y.tab.c" /* yacc.c:1661  */
    break;

  case 242:
#line 858 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = (yyvsp[0].member); }
#line 3805 "y.tab.c" /* yacc.c:1661  */
    break;

  case 243:
#line 862 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = (yyvsp[0].entry); }
#line 3811 "y.tab.c" /* yacc.c:1661  */
    break;

  case 244:
#line 864 "xi-grammar.y" /* yacc.c:1661  */
    {
                  (yyvsp[0].entry)->tspec = (yyvsp[-1].tvarlist);
                  (yyval.member) = (yyvsp[0].entry);
                }
#line 3820 "y.tab.c" /* yacc.c:1661  */
    break;

  case 245:
#line 869 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = (yyvsp[-1].member); }
#line 3826 "y.tab.c" /* yacc.c:1661  */
    break;

  case 246:
#line 871 "xi-grammar.y" /* yacc.c:1661  */
    {
          ERROR("invalid SDAG member",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 3836 "y.tab.c" /* yacc.c:1661  */
    break;

  case 247:
#line 879 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3842 "y.tab.c" /* yacc.c:1661  */
    break;

  case 248:
#line 881 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3848 "y.tab.c" /* yacc.c:1661  */
    break;

  case 249:
#line 883 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3854 "y.tab.c" /* yacc.c:1661  */
    break;

  case 250:
#line 885 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3860 "y.tab.c" /* yacc.c:1661  */
    break;

  case 251:
#line 887 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3866 "y.tab.c" /* yacc.c:1661  */
    break;

  case 252:
#line 889 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3872 "y.tab.c" /* yacc.c:1661  */
    break;

  case 253:
#line 891 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3878 "y.tab.c" /* yacc.c:1661  */
    break;

  case 254:
#line 893 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3884 "y.tab.c" /* yacc.c:1661  */
    break;

  case 255:
#line 895 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3890 "y.tab.c" /* yacc.c:1661  */
    break;

  case 256:
#line 897 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3896 "y.tab.c" /* yacc.c:1661  */
    break;

  case 257:
#line 899 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3902 "y.tab.c" /* yacc.c:1661  */
    break;

  case 258:
#line 902 "xi-grammar.y" /* yacc.c:1661  */
    {
                  (yyval.entry) = new Entry(lineno, (yyvsp[-5].intval), (yyvsp[-4].type), (yyvsp[-3].strval), (yyvsp[-2].plist), (yyvsp[-1].val), (yyvsp[0].sentry), (const char *) NULL, (yylsp[-6]).first_line, (yyloc).last_line);
		  if ((yyvsp[0].sentry) != 0) {
		    (yyvsp[0].sentry)->con1 = new SdagConstruct(SIDENT, (yyvsp[-3].strval));
                    (yyvsp[0].sentry)->setEntry((yyval.entry));
                    (yyvsp[0].sentry)->param = new ParamList((yyvsp[-2].plist));
                  }
                  firstRdma = true;
		}
#line 3916 "y.tab.c" /* yacc.c:1661  */
    break;

  case 259:
#line 912 "xi-grammar.y" /* yacc.c:1661  */
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
#line 3937 "y.tab.c" /* yacc.c:1661  */
    break;

  case 260:
#line 929 "xi-grammar.y" /* yacc.c:1661  */
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
#line 3956 "y.tab.c" /* yacc.c:1661  */
    break;

  case 261:
#line 946 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.accelBlock) = new AccelBlock(lineno, new XStr((yyvsp[-2].strval))); }
#line 3962 "y.tab.c" /* yacc.c:1661  */
    break;

  case 262:
#line 948 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.accelBlock) = new AccelBlock(lineno, NULL); }
#line 3968 "y.tab.c" /* yacc.c:1661  */
    break;

  case 263:
#line 952 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3974 "y.tab.c" /* yacc.c:1661  */
    break;

  case 264:
#line 956 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 0; }
#line 3980 "y.tab.c" /* yacc.c:1661  */
    break;

  case 265:
#line 958 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = (yyvsp[-1].intval); }
#line 3986 "y.tab.c" /* yacc.c:1661  */
    break;

  case 266:
#line 960 "xi-grammar.y" /* yacc.c:1661  */
    { ERROR("invalid entry method attribute list",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
#line 3995 "y.tab.c" /* yacc.c:1661  */
    break;

  case 267:
#line 967 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = (yyvsp[0].intval); }
#line 4001 "y.tab.c" /* yacc.c:1661  */
    break;

  case 268:
#line 969 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = (yyvsp[-2].intval) | (yyvsp[0].intval); }
#line 4007 "y.tab.c" /* yacc.c:1661  */
    break;

  case 269:
#line 973 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = STHREADED; }
#line 4013 "y.tab.c" /* yacc.c:1661  */
    break;

  case 270:
#line 975 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SSYNC; }
#line 4019 "y.tab.c" /* yacc.c:1661  */
    break;

  case 271:
#line 977 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SIGET; }
#line 4025 "y.tab.c" /* yacc.c:1661  */
    break;

  case 272:
#line 979 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SLOCKED; }
#line 4031 "y.tab.c" /* yacc.c:1661  */
    break;

  case 273:
#line 981 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SCREATEHERE; }
#line 4037 "y.tab.c" /* yacc.c:1661  */
    break;

  case 274:
#line 983 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SCREATEHOME; }
#line 4043 "y.tab.c" /* yacc.c:1661  */
    break;

  case 275:
#line 985 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SNOKEEP; }
#line 4049 "y.tab.c" /* yacc.c:1661  */
    break;

  case 276:
#line 987 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SNOTRACE; }
#line 4055 "y.tab.c" /* yacc.c:1661  */
    break;

  case 277:
#line 989 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SAPPWORK; }
#line 4061 "y.tab.c" /* yacc.c:1661  */
    break;

  case 278:
#line 991 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SIMMEDIATE; }
#line 4067 "y.tab.c" /* yacc.c:1661  */
    break;

  case 279:
#line 993 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SSKIPSCHED; }
#line 4073 "y.tab.c" /* yacc.c:1661  */
    break;

  case 280:
#line 995 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SINLINE; }
#line 4079 "y.tab.c" /* yacc.c:1661  */
    break;

  case 281:
#line 997 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SLOCAL; }
#line 4085 "y.tab.c" /* yacc.c:1661  */
    break;

  case 282:
#line 999 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SPYTHON; }
#line 4091 "y.tab.c" /* yacc.c:1661  */
    break;

  case 283:
#line 1001 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SMEM; }
#line 4097 "y.tab.c" /* yacc.c:1661  */
    break;

  case 284:
#line 1003 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SREDUCE; }
#line 4103 "y.tab.c" /* yacc.c:1661  */
    break;

  case 285:
#line 1005 "xi-grammar.y" /* yacc.c:1661  */
    {
        (yyval.intval) = SAGGREGATE;
    }
#line 4111 "y.tab.c" /* yacc.c:1661  */
    break;

  case 286:
#line 1009 "xi-grammar.y" /* yacc.c:1661  */
    {
		  ERROR("invalid entry method attribute",
		        (yylsp[0]).first_column, (yylsp[0]).last_column);
		  yyclearin;
		  yyerrok;
		}
#line 4122 "y.tab.c" /* yacc.c:1661  */
    break;

  case 287:
#line 1018 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4128 "y.tab.c" /* yacc.c:1661  */
    break;

  case 288:
#line 1020 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4134 "y.tab.c" /* yacc.c:1661  */
    break;

  case 289:
#line 1022 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4140 "y.tab.c" /* yacc.c:1661  */
    break;

  case 290:
#line 1026 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = ""; }
#line 4146 "y.tab.c" /* yacc.c:1661  */
    break;

  case 291:
#line 1028 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4152 "y.tab.c" /* yacc.c:1661  */
    break;

  case 292:
#line 1030 "xi-grammar.y" /* yacc.c:1661  */
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s, %s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4162 "y.tab.c" /* yacc.c:1661  */
    break;

  case 293:
#line 1038 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = ""; }
#line 4168 "y.tab.c" /* yacc.c:1661  */
    break;

  case 294:
#line 1040 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4174 "y.tab.c" /* yacc.c:1661  */
    break;

  case 295:
#line 1042 "xi-grammar.y" /* yacc.c:1661  */
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s[%s]%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4184 "y.tab.c" /* yacc.c:1661  */
    break;

  case 296:
#line 1048 "xi-grammar.y" /* yacc.c:1661  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s{%s}%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4194 "y.tab.c" /* yacc.c:1661  */
    break;

  case 297:
#line 1054 "xi-grammar.y" /* yacc.c:1661  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s(%s)%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4204 "y.tab.c" /* yacc.c:1661  */
    break;

  case 298:
#line 1060 "xi-grammar.y" /* yacc.c:1661  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"(%s)%s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4214 "y.tab.c" /* yacc.c:1661  */
    break;

  case 299:
#line 1068 "xi-grammar.y" /* yacc.c:1661  */
    {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval));
		}
#line 4223 "y.tab.c" /* yacc.c:1661  */
    break;

  case 300:
#line 1075 "xi-grammar.y" /* yacc.c:1661  */
    {
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			(yyval.intval) = 0;
		}
#line 4233 "y.tab.c" /* yacc.c:1661  */
    break;

  case 301:
#line 1083 "xi-grammar.y" /* yacc.c:1661  */
    {
			in_braces=0;
			(yyval.intval) = 0;
		}
#line 4242 "y.tab.c" /* yacc.c:1661  */
    break;

  case 302:
#line 1090 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[0].type));}
#line 4248 "y.tab.c" /* yacc.c:1661  */
    break;

  case 303:
#line 1092 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval)); (yyval.pname)->setConditional((yyvsp[0].intval)); }
#line 4254 "y.tab.c" /* yacc.c:1661  */
    break;

  case 304:
#line 1094 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[-3].type),(yyvsp[-2].strval),0,(yyvsp[0].val));}
#line 4260 "y.tab.c" /* yacc.c:1661  */
    break;

  case 305:
#line 1096 "xi-grammar.y" /* yacc.c:1661  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
		}
#line 4269 "y.tab.c" /* yacc.c:1661  */
    break;

  case 306:
#line 1101 "xi-grammar.y" /* yacc.c:1661  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
			(yyval.pname)->setRdma(true);
			if(firstRdma) {
				(yyval.pname)->setFirstRdma(true);
				firstRdma = false;
			}
		}
#line 4283 "y.tab.c" /* yacc.c:1661  */
    break;

  case 307:
#line 1112 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READONLY; }
#line 4289 "y.tab.c" /* yacc.c:1661  */
    break;

  case 308:
#line 1113 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READWRITE; }
#line 4295 "y.tab.c" /* yacc.c:1661  */
    break;

  case 309:
#line 1114 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_WRITEONLY; }
#line 4301 "y.tab.c" /* yacc.c:1661  */
    break;

  case 310:
#line 1117 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.xstrptr) = new XStr((yyvsp[0].strval)); }
#line 4307 "y.tab.c" /* yacc.c:1661  */
    break;

  case 311:
#line 1118 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "->" << (yyvsp[0].strval); }
#line 4313 "y.tab.c" /* yacc.c:1661  */
    break;

  case 312:
#line 1119 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[-2].xstrptr)) << "." << (yyvsp[0].strval); }
#line 4319 "y.tab.c" /* yacc.c:1661  */
    break;

  case 313:
#line 1121 "xi-grammar.y" /* yacc.c:1661  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << *((yyvsp[-1].xstrptr)) << "]";
                  delete (yyvsp[-3].xstrptr);
                  delete (yyvsp[-1].xstrptr);
                }
#line 4330 "y.tab.c" /* yacc.c:1661  */
    break;

  case 314:
#line 1128 "xi-grammar.y" /* yacc.c:1661  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << (yyvsp[-1].strval) << "]";
                  delete (yyvsp[-3].xstrptr);
                }
#line 4340 "y.tab.c" /* yacc.c:1661  */
    break;

  case 315:
#line 1134 "xi-grammar.y" /* yacc.c:1661  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "(" << *((yyvsp[-1].xstrptr)) << ")";
                  delete (yyvsp[-3].xstrptr);
                  delete (yyvsp[-1].xstrptr);
                }
#line 4351 "y.tab.c" /* yacc.c:1661  */
    break;

  case 316:
#line 1143 "xi-grammar.y" /* yacc.c:1661  */
    {
                  in_bracket = 0;
                  (yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName(), (yyvsp[-1].strval));
                }
#line 4360 "y.tab.c" /* yacc.c:1661  */
    break;

  case 317:
#line 1150 "xi-grammar.y" /* yacc.c:1661  */
    {
                  (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[-6].intval));
                }
#line 4370 "y.tab.c" /* yacc.c:1661  */
    break;

  case 318:
#line 1156 "xi-grammar.y" /* yacc.c:1661  */
    {
		  (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType(Parameter::ACCEL_BUFFER_TYPE_READWRITE);
		}
#line 4380 "y.tab.c" /* yacc.c:1661  */
    break;

  case 319:
#line 1162 "xi-grammar.y" /* yacc.c:1661  */
    {
                  (yyval.pname) = (yyvsp[-3].pname);
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[-5].intval));
		}
#line 4390 "y.tab.c" /* yacc.c:1661  */
    break;

  case 320:
#line 1170 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 4396 "y.tab.c" /* yacc.c:1661  */
    break;

  case 321:
#line 1172 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 4402 "y.tab.c" /* yacc.c:1661  */
    break;

  case 322:
#line 1176 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 4408 "y.tab.c" /* yacc.c:1661  */
    break;

  case 323:
#line 1178 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 4414 "y.tab.c" /* yacc.c:1661  */
    break;

  case 324:
#line 1182 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.plist) = (yyvsp[-1].plist); }
#line 4420 "y.tab.c" /* yacc.c:1661  */
    break;

  case 325:
#line 1184 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.plist) = new ParamList(new Parameter(0, new BuiltinType("void"))); }
#line 4426 "y.tab.c" /* yacc.c:1661  */
    break;

  case 326:
#line 1188 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.plist) = (yyvsp[-1].plist); }
#line 4432 "y.tab.c" /* yacc.c:1661  */
    break;

  case 327:
#line 1190 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.plist) = 0; }
#line 4438 "y.tab.c" /* yacc.c:1661  */
    break;

  case 328:
#line 1194 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.val) = 0; }
#line 4444 "y.tab.c" /* yacc.c:1661  */
    break;

  case 329:
#line 1196 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4450 "y.tab.c" /* yacc.c:1661  */
    break;

  case 330:
#line 1200 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sentry) = 0; }
#line 4456 "y.tab.c" /* yacc.c:1661  */
    break;

  case 331:
#line 1202 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[0].sc)); }
#line 4462 "y.tab.c" /* yacc.c:1661  */
    break;

  case 332:
#line 1204 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[-2].slist)); }
#line 4468 "y.tab.c" /* yacc.c:1661  */
    break;

  case 333:
#line 1208 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.slist) = new SListConstruct((yyvsp[0].sc)); }
#line 4474 "y.tab.c" /* yacc.c:1661  */
    break;

  case 334:
#line 1210 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.slist) = new SListConstruct((yyvsp[-1].sc), (yyvsp[0].slist));  }
#line 4480 "y.tab.c" /* yacc.c:1661  */
    break;

  case 335:
#line 1214 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.olist) = new OListConstruct((yyvsp[0].sc)); }
#line 4486 "y.tab.c" /* yacc.c:1661  */
    break;

  case 336:
#line 1216 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.olist) = new OListConstruct((yyvsp[-1].sc), (yyvsp[0].slist)); }
#line 4492 "y.tab.c" /* yacc.c:1661  */
    break;

  case 337:
#line 1220 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.clist) = new CaseListConstruct((yyvsp[0].when)); }
#line 4498 "y.tab.c" /* yacc.c:1661  */
    break;

  case 338:
#line 1222 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.clist) = new CaseListConstruct((yyvsp[-1].when), (yyvsp[0].clist)); }
#line 4504 "y.tab.c" /* yacc.c:1661  */
    break;

  case 339:
#line 1224 "xi-grammar.y" /* yacc.c:1661  */
    {
		  ERROR("case blocks can only contain when clauses",
		        (yylsp[0]).first_column, (yylsp[0]).last_column);
		  (yyval.clist) = 0;
		}
#line 4514 "y.tab.c" /* yacc.c:1661  */
    break;

  case 340:
#line 1232 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4520 "y.tab.c" /* yacc.c:1661  */
    break;

  case 341:
#line 1234 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = 0; }
#line 4526 "y.tab.c" /* yacc.c:1661  */
    break;

  case 342:
#line 1238 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = new WhenConstruct((yyvsp[-2].entrylist), 0); }
#line 4532 "y.tab.c" /* yacc.c:1661  */
    break;

  case 343:
#line 1240 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = new WhenConstruct((yyvsp[-1].entrylist), (yyvsp[0].sc)); }
#line 4538 "y.tab.c" /* yacc.c:1661  */
    break;

  case 344:
#line 1242 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = new WhenConstruct((yyvsp[-3].entrylist), (yyvsp[-1].slist)); }
#line 4544 "y.tab.c" /* yacc.c:1661  */
    break;

  case 345:
#line 1246 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4550 "y.tab.c" /* yacc.c:1661  */
    break;

  case 346:
#line 1248 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4556 "y.tab.c" /* yacc.c:1661  */
    break;

  case 347:
#line 1250 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4562 "y.tab.c" /* yacc.c:1661  */
    break;

  case 348:
#line 1252 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4568 "y.tab.c" /* yacc.c:1661  */
    break;

  case 349:
#line 1254 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4574 "y.tab.c" /* yacc.c:1661  */
    break;

  case 350:
#line 1256 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4580 "y.tab.c" /* yacc.c:1661  */
    break;

  case 351:
#line 1258 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4586 "y.tab.c" /* yacc.c:1661  */
    break;

  case 352:
#line 1260 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4592 "y.tab.c" /* yacc.c:1661  */
    break;

  case 353:
#line 1262 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4598 "y.tab.c" /* yacc.c:1661  */
    break;

  case 354:
#line 1264 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4604 "y.tab.c" /* yacc.c:1661  */
    break;

  case 355:
#line 1266 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4610 "y.tab.c" /* yacc.c:1661  */
    break;

  case 356:
#line 1268 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4616 "y.tab.c" /* yacc.c:1661  */
    break;

  case 357:
#line 1272 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new SerialConstruct((yyvsp[-2].strval), (yyvsp[-4].strval), (yylsp[-3]).first_line); }
#line 4622 "y.tab.c" /* yacc.c:1661  */
    break;

  case 358:
#line 1274 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new OverlapConstruct((yyvsp[-1].olist)); }
#line 4628 "y.tab.c" /* yacc.c:1661  */
    break;

  case 359:
#line 1276 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = (yyvsp[0].when); }
#line 4634 "y.tab.c" /* yacc.c:1661  */
    break;

  case 360:
#line 1278 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new CaseConstruct((yyvsp[-1].clist)); }
#line 4640 "y.tab.c" /* yacc.c:1661  */
    break;

  case 361:
#line 1280 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new ForConstruct((yyvsp[-8].intexpr), (yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4646 "y.tab.c" /* yacc.c:1661  */
    break;

  case 362:
#line 1282 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new ForConstruct((yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4652 "y.tab.c" /* yacc.c:1661  */
    break;

  case 363:
#line 1284 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-9].strval)), (yyvsp[-6].intexpr),
		             (yyvsp[-4].intexpr), (yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4659 "y.tab.c" /* yacc.c:1661  */
    break;

  case 364:
#line 1287 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-11].strval)), (yyvsp[-8].intexpr),
		             (yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4666 "y.tab.c" /* yacc.c:1661  */
    break;

  case 365:
#line 1290 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new IfConstruct((yyvsp[-3].intexpr), (yyvsp[-1].sc), (yyvsp[0].sc)); }
#line 4672 "y.tab.c" /* yacc.c:1661  */
    break;

  case 366:
#line 1292 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new IfConstruct((yyvsp[-5].intexpr), (yyvsp[-2].slist), (yyvsp[0].sc)); }
#line 4678 "y.tab.c" /* yacc.c:1661  */
    break;

  case 367:
#line 1294 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new WhileConstruct((yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4684 "y.tab.c" /* yacc.c:1661  */
    break;

  case 368:
#line 1296 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new WhileConstruct((yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4690 "y.tab.c" /* yacc.c:1661  */
    break;

  case 369:
#line 1298 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new SerialConstruct((yyvsp[-2].strval), NULL, (yyloc).first_line); }
#line 4696 "y.tab.c" /* yacc.c:1661  */
    break;

  case 370:
#line 1300 "xi-grammar.y" /* yacc.c:1661  */
    {
		  ERROR("unknown SDAG construct or malformed entry method declaration.\n"
		        "You may have forgotten to terminate a previous entry method declaration with a"
		        " semicolon or forgotten to mark a block of sequential SDAG code as 'serial'",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
#line 4708 "y.tab.c" /* yacc.c:1661  */
    break;

  case 371:
#line 1310 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = 0; }
#line 4714 "y.tab.c" /* yacc.c:1661  */
    break;

  case 372:
#line 1312 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new ElseConstruct((yyvsp[0].sc)); }
#line 4720 "y.tab.c" /* yacc.c:1661  */
    break;

  case 373:
#line 1314 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new ElseConstruct((yyvsp[-1].slist)); }
#line 4726 "y.tab.c" /* yacc.c:1661  */
    break;

  case 374:
#line 1318 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intexpr) = new IntExprConstruct((yyvsp[0].strval)); }
#line 4732 "y.tab.c" /* yacc.c:1661  */
    break;

  case 375:
#line 1322 "xi-grammar.y" /* yacc.c:1661  */
    { in_int_expr = 0; (yyval.intval) = 0; }
#line 4738 "y.tab.c" /* yacc.c:1661  */
    break;

  case 376:
#line 1326 "xi-grammar.y" /* yacc.c:1661  */
    { in_int_expr = 1; (yyval.intval) = 0; }
#line 4744 "y.tab.c" /* yacc.c:1661  */
    break;

  case 377:
#line 1330 "xi-grammar.y" /* yacc.c:1661  */
    {
		  (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[-1].strval), (yyvsp[0].plist), 0, 0, 0, (yyloc).first_line, (yyloc).last_line);
		  firstRdma = true;
		}
#line 4753 "y.tab.c" /* yacc.c:1661  */
    break;

  case 378:
#line 1335 "xi-grammar.y" /* yacc.c:1661  */
    {
		  (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[-4].strval), (yyvsp[0].plist), 0, 0, (yyvsp[-2].strval), (yyloc).first_line, (yyloc).last_line);
		  firstRdma = true;
		}
#line 4762 "y.tab.c" /* yacc.c:1661  */
    break;

  case 379:
#line 1342 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.entrylist) = new EntryList((yyvsp[0].entry)); }
#line 4768 "y.tab.c" /* yacc.c:1661  */
    break;

  case 380:
#line 1344 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.entrylist) = new EntryList((yyvsp[-2].entry),(yyvsp[0].entrylist)); }
#line 4774 "y.tab.c" /* yacc.c:1661  */
    break;

  case 381:
#line 1348 "xi-grammar.y" /* yacc.c:1661  */
    { in_bracket=1; }
#line 4780 "y.tab.c" /* yacc.c:1661  */
    break;

  case 382:
#line 1351 "xi-grammar.y" /* yacc.c:1661  */
    { in_bracket=0; }
#line 4786 "y.tab.c" /* yacc.c:1661  */
    break;

  case 383:
#line 1355 "xi-grammar.y" /* yacc.c:1661  */
    { if (!macroDefined((yyvsp[0].strval), 1)) in_comment = 1; }
#line 4792 "y.tab.c" /* yacc.c:1661  */
    break;

  case 384:
#line 1359 "xi-grammar.y" /* yacc.c:1661  */
    { if (!macroDefined((yyvsp[0].strval), 0)) in_comment = 1; }
#line 4798 "y.tab.c" /* yacc.c:1661  */
    break;


#line 4802 "y.tab.c" /* yacc.c:1661  */
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
#line 1362 "xi-grammar.y" /* yacc.c:1906  */


void yyerror(const char *s)
{
	fprintf(stderr, "[PARSE-ERROR] Unexpected/missing token at line %d. Current token being parsed: '%s'.\n", lineno, yytext);
}
