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
#define YYLAST   1519

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  92
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  117
/* YYNRULES -- Number of rules.  */
#define YYNRULES  376
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  734

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
      84,    85,    83,     2,    80,    90,    91,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    77,    76,
      81,    89,    82,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    87,     2,    88,     2,     2,     2,     2,     2,     2,
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
     497,   501,   503,   505,   509,   511,   513,   517,   521,   526,
     527,   531,   535,   540,   541,   546,   547,   557,   559,   563,
     565,   570,   571,   575,   577,   582,   583,   587,   592,   593,
     597,   599,   603,   605,   610,   611,   615,   616,   619,   623,
     625,   629,   631,   633,   638,   639,   643,   645,   649,   651,
     655,   659,   663,   669,   673,   675,   679,   681,   685,   689,
     693,   697,   699,   704,   705,   710,   711,   713,   715,   724,
     726,   728,   730,   734,   736,   740,   744,   746,   748,   750,
     752,   756,   758,   763,   770,   774,   776,   778,   779,   781,
     783,   785,   789,   791,   793,   799,   805,   814,   816,   818,
     824,   832,   834,   837,   841,   845,   847,   852,   854,   862,
     864,   866,   868,   870,   872,   874,   876,   878,   880,   882,
     885,   895,   912,   929,   931,   935,   940,   941,   943,   950,
     952,   956,   958,   960,   962,   964,   966,   968,   970,   972,
     974,   976,   978,   980,   982,   984,   986,   988,   992,  1001,
    1003,  1005,  1010,  1011,  1013,  1022,  1023,  1025,  1031,  1037,
    1043,  1051,  1058,  1066,  1073,  1075,  1077,  1079,  1084,  1096,
    1097,  1098,  1101,  1102,  1103,  1104,  1111,  1117,  1126,  1133,
    1139,  1145,  1153,  1155,  1159,  1161,  1165,  1167,  1171,  1173,
    1178,  1179,  1183,  1185,  1187,  1191,  1193,  1197,  1199,  1203,
    1205,  1207,  1215,  1218,  1221,  1223,  1225,  1229,  1231,  1233,
    1235,  1237,  1239,  1241,  1243,  1245,  1247,  1249,  1251,  1255,
    1257,  1259,  1261,  1263,  1265,  1267,  1270,  1273,  1275,  1277,
    1279,  1281,  1283,  1294,  1295,  1297,  1301,  1305,  1309,  1313,
    1318,  1325,  1327,  1331,  1334,  1338,  1342
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
  "'{'", "'}'", "','", "'<'", "'>'", "'*'", "'('", "')'", "'&'", "'['",
  "']'", "'='", "'-'", "'.'", "$accept", "File", "ModuleEList",
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
      44,    60,    62,    42,    40,    41,    38,    91,    93,    61,
      45,    46
};
# endif

#define YYPACT_NINF -545

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-545)))

#define YYTABLE_NINF -328

#define yytable_value_is_error(Yytable_value) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
     144,  1304,  1304,    57,  -545,   144,  -545,  -545,  -545,  -545,
    -545,  -545,  -545,  -545,  -545,  -545,  -545,  -545,  -545,  -545,
    -545,  -545,  -545,  -545,  -545,  -545,  -545,  -545,  -545,  -545,
    -545,  -545,  -545,  -545,  -545,  -545,  -545,  -545,  -545,  -545,
    -545,  -545,  -545,  -545,  -545,  -545,  -545,  -545,  -545,  -545,
    -545,  -545,  -545,  -545,    82,    82,  -545,  -545,  -545,   777,
      -3,  -545,  -545,  -545,    77,  1304,   148,  1304,  1304,   156,
     904,    51,   921,   777,  -545,  -545,  -545,  -545,  1413,    79,
     122,  -545,   108,  -545,  -545,  -545,    -3,   -30,  1345,   151,
     151,    -7,   122,   105,   105,   105,   105,   124,   127,  1304,
     181,   191,   777,  -545,  -545,  -545,  -545,  -545,  -545,  -545,
    -545,   294,  -545,  -545,  -545,  -545,   177,  -545,  -545,  -545,
    -545,  -545,  -545,  -545,  -545,  -545,  -545,  -545,    -3,  -545,
    -545,  -545,  1413,  -545,   -31,  -545,  -545,  -545,  -545,   220,
     122,    50,  -545,  -545,   192,   193,   203,    11,  -545,   122,
     777,   108,   213,    81,   -30,   243,  1444,  1429,   192,   193,
     203,  -545,    17,   122,  -545,   122,   122,   257,   122,   253,
    -545,    75,  1304,  1304,  1304,  1304,  1088,   248,   258,   217,
    1304,  -545,  -545,  -545,  1379,   265,   105,   105,   105,   105,
     248,   127,  -545,  -545,  -545,  -545,  -545,    -3,  -545,   315,
    -545,  -545,  -545,   200,  -545,  -545,  -545,   912,  -545,  -545,
    -545,  -545,  -545,   271,  1304,   280,    22,   -30,   302,   -30,
     277,  -545,   177,   281,     4,  -545,   284,   283,    41,    89,
     109,   282,   165,   122,  -545,  -545,   285,   295,   311,   306,
     306,   306,   306,  -545,  1304,   305,   324,   317,  1160,  1304,
     357,  1304,  -545,  -545,   322,   336,   335,  1304,   363,    90,
    1304,   342,   339,   177,  1304,  1304,  1304,  1304,  1304,  1304,
    -545,  -545,  -545,  -545,   343,  -545,   347,  -545,  -545,   311,
    -545,  -545,  -545,   345,   348,   349,   350,   -30,    -3,   122,
    1304,  -545,  -545,   353,  -545,   -30,   151,   912,   151,   151,
     912,   151,  -545,  -545,    75,  -545,   122,   172,   172,   172,
     172,   359,  -545,   357,  -545,   306,   306,  -545,   217,     2,
     344,   155,   340,  -545,   361,  1379,  -545,  -545,   306,   306,
     306,   306,   306,   236,   912,  -545,   364,   -30,   302,   -30,
     -30,  -545,    41,   366,  -545,   374,  -545,   367,   372,   381,
     122,   385,   383,  -545,   393,  -545,   330,    -3,  -545,  -545,
    -545,  -545,  -545,  -545,   172,   172,  -545,  -545,  -545,  1429,
       7,   351,  1429,  -545,  -545,  -545,  -545,  -545,  -545,  -545,
     172,   172,   172,   172,   172,   457,    -3,  -545,  1364,  -545,
    -545,  -545,  -545,  -545,  -545,   391,  -545,  -545,  -545,   392,
    -545,    94,   395,  -545,   122,  -545,   692,   436,   402,   177,
     330,  -545,  -545,  -545,  -545,  1304,  -545,  -545,  -545,  -545,
    -545,  -545,  -545,  -545,   412,  1429,  -545,  1304,   -30,   413,
     407,  1398,   151,   151,   151,  -545,  -545,   920,   998,  -545,
     177,    -3,  -545,   408,   177,  1304,   -30,     3,   406,  1398,
    -545,   411,   414,   415,   417,  -545,  -545,  -545,  -545,  -545,
    -545,  -545,  -545,  -545,  -545,  -545,  -545,  -545,  -545,   441,
    -545,   418,  -545,  -545,   422,   431,   428,   364,  1304,  -545,
     425,   177,    -3,   430,   427,  -545,   346,  -545,  -545,  -545,
    -545,  -545,  -545,  -545,  -545,  -545,   483,  -545,  1053,   503,
     364,  -545,    -3,  -545,  -545,  -545,   108,  -545,  1304,  -545,
    -545,   437,   435,   437,   472,   451,   473,   437,   453,   333,
      -3,   -30,  -545,  -545,  -545,   512,   364,  -545,   -30,   478,
     -30,   -25,   454,   515,   539,  -545,   458,   -30,   254,   456,
     272,   243,   448,   503,   452,  -545,   465,   455,   468,  -545,
     -30,   472,   341,  -545,   463,   440,   -30,   468,   437,   469,
     437,   479,   473,   437,   488,   -30,   489,   254,  -545,   177,
    -545,   177,   511,  -545,   394,   458,   -30,   437,  -545,   553,
     374,  -545,  -545,   490,  -545,  -545,   243,   594,   -30,   516,
     -30,   539,   458,   -30,   254,   243,  -545,  -545,  -545,  -545,
    -545,  -545,  -545,  -545,  -545,  1304,   495,   493,   486,   -30,
     499,   -30,   333,  -545,   364,  -545,   177,   333,   528,   514,
     506,   468,   524,   -30,   468,   526,   177,   510,  1429,  1329,
    -545,   243,   -30,   529,   530,  -545,  -545,   531,   614,  -545,
     -30,   437,   760,  -545,   243,   767,  -545,  -545,  1304,  1304,
     -30,   527,  -545,  1304,   468,   -30,  -545,   528,   333,  -545,
     535,   -30,   333,  -545,   177,   333,   528,  -545,   107,   135,
     519,  1304,   177,   775,   534,  -545,   537,   -30,   542,   541,
    -545,   543,  -545,  -545,  1304,  1232,   546,  1304,  1304,  -545,
     149,    -3,   333,  -545,   -30,  -545,   468,   -30,  -545,   528,
     239,   533,   355,  1304,  -545,   187,  -545,   544,   468,   816,
     545,  -545,  -545,  -545,  -545,  -545,  -545,  -545,   824,   333,
    -545,   -30,   333,  -545,   550,   468,   551,  -545,   831,  -545,
     333,  -545,   554,  -545
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
       0,    59,    69,   375,   376,   292,   254,   285,     0,   143,
     143,   143,     0,   151,   151,   151,   151,     0,   145,     0,
       0,     0,     0,    77,   215,   216,    71,    78,    79,    80,
      81,     0,    82,    70,   218,   217,     9,   249,   241,   242,
     243,   244,   245,   247,   248,   246,   239,   240,    75,    76,
      67,   110,     0,    96,    97,    98,    99,   107,   108,     0,
       0,    94,   113,   115,   126,   127,   128,   133,   255,     0,
       0,    68,     0,   286,   285,     0,     0,     0,   120,   121,
     122,   123,   136,     0,   144,     0,     0,     0,     0,   231,
     219,     0,     0,     0,     0,     0,     0,     0,   158,     0,
       0,   221,   233,   220,     0,     0,   151,   151,   151,   151,
       0,   145,   206,   207,   208,   209,   210,    10,    65,   129,
     106,   109,   100,   101,   104,   105,   114,    92,   112,   116,
     117,   118,   130,   132,     0,     0,     0,   285,   282,   285,
       0,   293,     0,     0,   124,   125,     0,   135,   139,   225,
     222,     0,   227,     0,   162,   163,     0,   153,    94,   174,
     174,   174,   174,   157,     0,     0,   160,     0,     0,     0,
       0,     0,   149,   150,     0,   147,   171,     0,     0,   123,
       0,   203,     0,     9,     0,     0,     0,     0,     0,     0,
     102,   103,    88,    89,    90,    93,     0,    87,   131,    94,
      74,    61,    60,     0,   283,     0,     0,   285,   253,     0,
       0,   134,   373,   139,   141,   285,   143,     0,   143,   143,
       0,   143,   232,   152,     0,   111,     0,     0,     0,     0,
       0,     0,   183,     0,   159,   174,   174,   146,     0,   164,
     193,    59,     0,   201,   195,     0,   205,    73,   174,   174,
     174,   174,   174,     0,     0,    95,     0,   285,   282,   285,
     285,   290,   139,     0,   140,     0,   137,     0,     0,     0,
       0,     0,     0,   154,   176,   175,     0,   211,   178,   179,
     180,   181,   182,   161,     0,     0,   148,   165,   172,     0,
     164,     0,     0,   199,   200,   196,   197,   198,   202,   204,
       0,     0,     0,     0,     0,   164,   191,    91,     0,    72,
     288,   284,   289,   287,   142,     0,   374,   138,   226,     0,
     223,     0,     0,   228,     0,   238,     0,     0,     0,     0,
       0,   234,   235,   184,   185,     0,   170,   173,   194,   186,
     187,   188,   189,   190,     0,     0,   317,   294,   285,   312,
       0,     0,   143,   143,   143,   177,   258,     0,     0,   236,
       9,   237,   214,   166,     0,     0,   285,   164,     0,     0,
     316,     0,     0,     0,     0,   278,   261,   262,   263,   264,
     270,   271,   272,   277,   265,   266,   267,   268,   269,   155,
     273,     0,   275,   276,     0,   259,    59,     0,     0,   212,
       0,     0,   192,     0,     0,   291,     0,   295,   297,   313,
     119,   224,   230,   229,   156,   274,     0,   257,     0,     0,
       0,   167,   168,   298,   280,   279,   281,   296,     0,   260,
     362,     0,     0,     0,     0,     0,   333,     0,     0,     0,
     322,   285,   251,   351,   323,   320,     0,   368,   285,     0,
     285,     0,   371,     0,     0,   332,     0,   285,     0,     0,
       0,     0,     0,     0,     0,   366,     0,     0,     0,   369,
     285,     0,     0,   335,     0,     0,   285,     0,     0,     0,
       0,     0,   333,     0,     0,   285,     0,   329,   331,     9,
     326,     9,     0,   250,     0,     0,   285,     0,   367,     0,
       0,   372,   334,     0,   350,   328,     0,     0,   285,     0,
     285,     0,     0,   285,     0,     0,   352,   330,   324,   361,
     321,   299,   300,   301,   319,     0,     0,   314,     0,   285,
       0,   285,     0,   359,     0,   336,     9,     0,   363,     0,
       0,     0,     0,   285,     0,     0,     9,     0,     0,     0,
     318,     0,   285,     0,     0,   370,   349,     0,     0,   357,
     285,     0,     0,   338,     0,     0,   339,   348,     0,     0,
     285,     0,   315,     0,     0,   285,   360,   363,     0,   364,
       0,   285,     0,   346,     9,     0,   363,   302,     0,     0,
       0,     0,     0,     0,     0,   358,     0,   285,     0,     0,
     337,     0,   344,   310,     0,     0,     0,     0,     0,   308,
       0,   252,     0,   354,   285,   365,     0,   285,   347,   363,
       0,     0,     0,     0,   304,     0,   311,     0,     0,     0,
       0,   345,   307,   306,   305,   303,   309,   353,     0,     0,
     341,   285,     0,   355,     0,     0,     0,   340,     0,   356,
       0,   342,     0,   343
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -545,  -545,   619,  -545,   -46,  -258,    -1,   -59,   562,   579,
     -55,  -545,  -545,  -545,  -185,  -545,  -180,  -545,  -135,   -77,
     -72,   -67,   -64,  -174,   487,   513,  -545,   -84,  -545,  -545,
    -271,  -545,  -545,   -78,   444,   328,  -545,   -43,   352,  -545,
    -545,   471,   337,  -545,   204,  -545,  -545,  -245,  -545,   -36,
     249,  -545,  -545,  -545,   -53,  -545,  -545,  -545,  -545,  -545,
    -545,   334,   354,  -545,   329,   595,  -545,  -156,   256,   597,
    -545,  -545,   438,  -545,  -545,  -545,  -545,   262,  -545,   226,
    -545,   175,  -545,  -545,   338,   -85,  -409,   -66,  -508,  -545,
    -545,  -521,  -545,  -545,  -329,    45,  -452,  -545,  -545,   132,
    -523,    86,  -533,   117,  -511,  -545,  -458,  -544,  -494,  -513,
    -467,  -545,   129,   150,   102,  -545,  -545
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     3,     4,    70,   357,   198,   238,   141,     5,    61,
      71,    72,    73,   274,   275,   276,   208,   142,   239,   143,
     158,   159,   160,   161,   162,   147,   148,   277,   345,   293,
     294,   104,   105,   165,   180,   254,   255,   172,   236,   495,
     246,   177,   247,   237,   369,   481,   370,   371,   106,   307,
     355,   107,   108,   109,   178,   110,   192,   193,   194,   195,
     196,   373,   323,   261,   262,   407,   112,   358,   408,   409,
     114,   115,   170,   183,   410,   411,   129,   412,    74,   149,
     438,   474,   475,   507,   285,   545,   428,   521,   222,   429,
     606,   668,   651,   607,   430,   608,   389,   575,   543,   522,
     539,   554,   566,   536,   523,   568,   540,   639,   546,   579,
     528,   532,   533,   295,   397,    75,    76
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[] =
{
      54,    55,   155,    87,   163,   327,   144,    82,    60,    60,
     259,   145,   166,   168,   146,   169,   446,   570,   130,   367,
     367,   151,   344,    86,   367,   499,   128,   567,   153,   583,
     164,   200,   585,   571,   597,   201,   548,   281,   240,   241,
     242,   524,   225,   557,   587,   256,   530,   185,   525,   212,
     537,   173,   174,   175,   154,   225,   567,    56,   305,   388,
     144,   625,   292,   206,    79,   145,    83,    84,   146,   220,
     197,   394,   214,    77,   544,   553,   555,   282,   616,   549,
     167,   368,   610,   567,   223,   524,  -169,   626,   226,   634,
     485,   588,   486,   590,   637,   215,   593,   213,   181,   336,
     260,   226,   451,   227,   228,   234,   229,   230,   642,   232,
     611,   645,   348,   675,   315,   351,   316,   633,   152,    78,
     489,   613,   682,   653,   249,   416,   235,   152,   292,   618,
     116,   207,   283,   555,   286,   676,   664,   268,   654,   679,
     424,   673,   681,   264,   265,   266,   267,     1,     2,   387,
     690,   259,   359,   360,   361,   711,   169,   150,    58,   217,
      59,   674,   635,   700,   702,   218,   152,   705,   219,   707,
    -195,   152,  -195,   296,   661,   245,   288,    81,   433,   322,
     659,   206,   479,   709,   663,   152,   152,   666,   164,   683,
     297,   684,   171,   298,   685,   718,   724,   686,   687,   726,
     708,    80,   341,    81,   308,   309,   310,   732,   413,   414,
     346,   176,   728,   279,   179,   693,   688,   197,   347,   650,
     349,   350,   485,   352,   419,   420,   421,   422,   423,   354,
     342,   706,    58,   684,    85,  -193,   685,  -193,   182,   686,
     687,   260,   152,   311,   372,   377,   300,   245,    58,   301,
     356,   720,   390,    58,   392,   393,   320,   252,   253,   324,
     723,   270,   271,   328,   329,   330,   331,   332,   333,   716,
     731,   684,   184,   510,   685,   209,   210,   686,   687,   364,
     365,   202,   203,   204,   205,   415,   211,   386,   418,   343,
     216,   401,   380,   381,   382,   383,   384,   558,   559,   560,
     514,   561,   562,   563,   427,   186,   187,   188,   189,   190,
     191,   598,    58,   599,   385,   511,   512,   513,   514,   515,
     516,   517,   221,   684,   712,   231,   685,   354,   564,   686,
     687,   405,    85,   233,   510,   248,    88,    89,    90,    91,
      92,   445,   510,   448,   263,   250,   518,   427,    99,   100,
      85,  -325,   101,   212,   452,   453,   454,   278,   636,   280,
     284,   484,   287,   441,   289,   427,   144,   290,   647,   291,
     299,   145,   406,   303,   146,   304,   511,   512,   513,   514,
     515,   516,   517,   306,   511,   512,   513,   514,   515,   516,
     517,  -292,   207,   312,   197,    81,   375,   376,   482,  -292,
     601,    81,   504,   505,   313,   314,   680,   518,   243,  -213,
     317,    85,  -292,   319,   443,   518,   318,  -292,   321,    85,
     582,   326,   325,   334,   337,  -292,   447,   506,   338,   335,
     417,   131,   157,   372,   339,   502,   541,   477,   340,   684,
     292,   510,   685,   714,   483,   686,   687,   362,   388,    81,
     322,   395,   398,   520,   399,   133,   134,   135,   136,   137,
     138,   139,   396,   602,   603,   580,   400,   402,   403,   140,
     556,   586,   565,   404,   367,   431,   432,   500,   406,   434,
     595,   440,   604,   511,   512,   513,   514,   515,   516,   517,
     605,   444,   450,   449,   488,   480,   490,   520,   494,   491,
     492,   565,   493,   619,   510,   621,   496,   526,   624,   609,
     497,   498,   -11,   501,   518,   503,   510,   485,    85,  -327,
     508,   527,   529,   197,   631,   197,   623,   531,   565,   534,
     535,   538,   542,   547,   551,   569,    85,   572,   644,   574,
     510,   576,   584,   577,   649,   605,   511,   512,   513,   514,
     515,   516,   517,   578,   510,   660,   589,   591,   511,   512,
     513,   514,   515,   516,   517,   670,   594,   600,   596,   615,
     197,   620,   628,   629,   630,   632,   678,   518,   638,    58,
     197,   519,   511,   512,   513,   514,   515,   516,   517,   518,
     640,   648,   696,   552,   641,   510,   511,   512,   513,   514,
     515,   516,   517,   643,   627,   646,   655,   689,   671,   656,
     657,   677,   710,   518,   694,   510,   695,    85,   197,   697,
     698,   713,   699,   717,    57,   721,   691,   518,   703,   727,
     729,   612,   103,   733,    62,   269,   725,   511,   512,   513,
     514,   515,   516,   517,   224,   199,   366,   667,   669,   251,
     363,   487,   672,   435,   379,   374,   353,   511,   512,   513,
     514,   515,   516,   517,   478,   111,   442,   113,   518,   439,
     667,   302,   617,   509,   652,   573,   391,   622,   378,   592,
     581,   550,   614,   667,   667,     0,   704,   667,   518,     0,
       0,     0,   658,   436,     0,  -256,  -256,  -256,     0,  -256,
    -256,  -256,   715,  -256,  -256,  -256,  -256,  -256,     0,     0,
       0,  -256,  -256,  -256,  -256,  -256,  -256,  -256,  -256,  -256,
    -256,  -256,  -256,     0,  -256,  -256,  -256,  -256,  -256,  -256,
    -256,  -256,  -256,  -256,  -256,  -256,  -256,  -256,  -256,  -256,
    -256,  -256,  -256,     0,  -256,     0,  -256,  -256,     0,     0,
       0,     0,     0,  -256,  -256,  -256,  -256,  -256,  -256,  -256,
    -256,   510,     0,  -256,  -256,  -256,  -256,  -256,   510,     0,
       0,     0,     0,     0,     0,     0,   510,     0,    63,   437,
      -5,    -5,    64,    -5,    -5,    -5,    -5,    -5,    -5,    -5,
      -5,    -5,    -5,    -5,     0,    -5,    -5,     0,     0,    -5,
       0,     0,     0,   511,   512,   513,   514,   515,   516,   517,
     511,   512,   513,   514,   515,   516,   517,   510,   511,   512,
     513,   514,   515,   516,   517,   510,     0,     0,     0,     0,
      65,    66,   510,     0,   518,     0,    67,    68,   662,     0,
       0,   518,     0,     0,     0,   665,     0,     0,    69,   518,
       0,     0,     0,   692,     0,    -5,   -66,     0,     0,   511,
     512,   513,   514,   515,   516,   517,     0,   511,   512,   513,
     514,   515,   516,   517,   511,   512,   513,   514,   515,   516,
     517,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     518,     0,     0,     0,   719,     0,     0,     0,   518,     0,
       0,     0,   722,     0,     0,   518,     0,     1,     2,   730,
      88,    89,    90,    91,    92,    93,    94,    95,    96,    97,
      98,   455,    99,   100,     0,     0,   101,   117,   118,   119,
     120,     0,   121,   122,   123,   124,   125,     0,     0,     0,
       0,   456,     0,   457,   458,   459,   460,   461,   462,   131,
     157,   463,   464,   465,   466,   467,   468,     0,     0,     0,
       0,     0,     0,   126,     0,     0,     0,    81,   272,   273,
       0,   469,   470,   133,   134,   135,   136,   137,   138,   139,
       0,     0,   102,     0,     0,     0,     0,   140,   471,     0,
       0,     0,   472,   473,     0,     0,     0,    58,     0,     0,
     127,     6,     7,     8,     0,     9,    10,    11,     0,    12,
      13,    14,    15,    16,     0,     0,     0,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,     0,
      29,    30,    31,    32,    33,   131,   132,    34,    35,    36,
      37,    38,    39,    40,    41,    42,    43,    44,    45,     0,
      46,     0,    47,   476,   455,     0,     0,     0,     0,   133,
     134,   135,   136,   137,   138,   139,    49,     0,     0,    50,
      51,    52,    53,   140,   456,     0,   457,   458,   459,   460,
     461,   462,     0,     0,   463,   464,   465,   466,   467,   468,
       0,     6,     7,     8,     0,     9,    10,    11,     0,    12,
      13,    14,    15,    16,   469,   470,     0,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,     0,
      29,    30,    31,    32,    33,   472,   473,    34,    35,    36,
      37,    38,    39,    40,    41,    42,    43,    44,    45,   243,
      46,     0,    47,    48,   244,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    49,     0,     0,    50,
      51,    52,    53,     6,     7,     8,     0,     9,    10,    11,
       0,    12,    13,    14,    15,    16,     0,     0,     0,    17,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      28,     0,    29,    30,    31,    32,    33,     0,     0,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,     0,    46,     0,    47,    48,   244,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    49,     0,
       0,    50,    51,    52,    53,     6,     7,     8,     0,     9,
      10,    11,     0,    12,    13,    14,    15,    16,     0,     0,
       0,    17,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,     0,    29,    30,    31,    32,    33,     0,
       0,    34,    35,    36,    37,    38,    39,    40,    41,    42,
      43,    44,    45,     0,    46,     0,    47,    48,   701,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      49,     0,     0,    50,    51,    52,    53,     6,     7,     8,
       0,     9,    10,    11,     0,    12,    13,    14,    15,    16,
       0,     0,     0,    17,    18,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,   601,    29,    30,    31,    32,
      33,     0,     0,    34,    35,    36,    37,    38,    39,    40,
      41,    42,    43,    44,    45,     0,    46,     0,    47,    48,
       0,   156,     0,     0,     0,     0,   131,   157,     0,     0,
       0,     0,    49,     0,     0,    50,    51,    52,    53,     0,
       0,     0,   131,   157,    81,     0,     0,     0,     0,     0,
     133,   134,   135,   136,   137,   138,   139,   257,   602,   603,
      81,   131,   157,   425,   140,     0,   133,   134,   135,   136,
     137,   138,   139,     0,     0,     0,   131,   157,     0,    81,
     140,     0,     0,     0,     0,   133,   134,   135,   136,   137,
     138,   139,     0,     0,    81,   131,   157,   425,     0,   140,
     133,   134,   135,   136,   137,   138,   139,     0,     0,   426,
     131,   132,     0,    81,   258,     0,     0,     0,     0,   133,
     134,   135,   136,   137,   138,   139,   131,   157,    81,     0,
       0,     0,     0,   140,   133,   134,   135,   136,   137,   138,
     139,   131,     0,     0,    81,     0,     0,     0,   140,     0,
     133,   134,   135,   136,   137,   138,   139,     0,     0,    81,
       0,     0,     0,     0,   140,   133,   134,   135,   136,   137,
     138,   139,     0,     0,     0,     0,     0,     0,     0,   140
};

static const yytype_int16 yycheck[] =
{
       1,     2,    87,    69,    88,   263,    78,    66,    54,    55,
     184,    78,    90,    91,    78,    92,   425,   540,    73,    17,
      17,    80,   293,    69,    17,   477,    72,   538,    58,   552,
      37,    62,   555,   541,   567,    66,   530,    15,   173,   174,
     175,   499,    38,   537,   557,   180,   513,   102,   500,    38,
     517,    94,    95,    96,    84,    38,   567,     0,   238,    84,
     132,   594,    87,   140,    65,   132,    67,    68,   132,   154,
     116,   342,   149,    76,   526,   533,   534,    55,   586,   531,
      87,    79,   576,   594,   156,   543,    79,   595,    84,   612,
      87,   558,    89,   560,   617,   150,   563,    86,    99,   279,
     184,    84,   431,    86,   163,    30,   165,   166,   621,   168,
     577,   624,   297,   657,   249,   300,   251,   611,    77,    42,
     449,   579,   666,   631,   177,   370,    51,    77,    87,   587,
      79,    81,   217,   591,   219,   658,   644,   190,   632,   662,
     385,   654,   665,   186,   187,   188,   189,     3,     4,   334,
     671,   325,   308,   309,   310,   699,   233,    78,    76,    78,
      78,   655,   614,   684,   685,    84,    77,   688,    87,   692,
      80,    77,    82,    84,   641,   176,   222,    55,    84,    89,
     638,   258,   440,   696,   642,    77,    77,   645,    37,    82,
      81,    84,    87,    84,    87,   708,   719,    90,    91,   722,
     694,    53,   287,    55,   240,   241,   242,   730,   364,   365,
     295,    87,   725,   214,    87,   673,    81,   263,   296,   628,
     298,   299,    87,   301,   380,   381,   382,   383,   384,   306,
     289,    82,    76,    84,    78,    80,    87,    82,    57,    90,
      91,   325,    77,   244,    89,   322,    81,   248,    76,    84,
      78,   709,   337,    76,   339,   340,   257,    40,    41,   260,
     718,    61,    62,   264,   265,   266,   267,   268,   269,    82,
     728,    84,    81,     1,    87,    83,    83,    90,    91,   315,
     316,    61,    62,    63,    64,   369,    83,   333,   372,   290,
      77,   350,   328,   329,   330,   331,   332,    43,    44,    45,
      46,    47,    48,    49,   388,    11,    12,    13,    14,    15,
      16,   569,    76,   571,    78,    43,    44,    45,    46,    47,
      48,    49,    79,    84,    85,    68,    87,   404,    74,    90,
      91,     1,    78,    80,     1,    87,     6,     7,     8,     9,
      10,   425,     1,   428,    79,    87,    74,   431,    18,    19,
      78,    79,    22,    38,   432,   433,   434,    86,   616,    79,
      58,   446,    85,   409,    83,   449,   438,    83,   626,    86,
      88,   438,    42,    88,   438,    80,    43,    44,    45,    46,
      47,    48,    49,    77,    43,    44,    45,    46,    47,    48,
      49,    58,    81,    88,   440,    55,    56,    57,   444,    58,
       6,    55,    56,    57,    80,    88,   664,    74,    51,    79,
      88,    78,    79,    78,   415,    74,    80,    84,    55,    78,
      79,    82,    80,    80,    79,    84,   427,   486,    80,    82,
      79,    37,    38,    89,    85,   481,   521,   438,    88,    84,
      87,     1,    87,    88,   445,    90,    91,    88,    84,    55,
      89,    85,    85,   499,    82,    61,    62,    63,    64,    65,
      66,    67,    88,    69,    70,   550,    85,    82,    85,    75,
     536,   556,   538,    80,    17,    84,    84,   478,    42,    84,
     565,    79,    88,    43,    44,    45,    46,    47,    48,    49,
     574,    79,    85,    80,    88,    87,    85,   543,    57,    85,
      85,   567,    85,   588,     1,   590,    88,   508,   593,   575,
      88,    80,    84,    88,    74,    88,     1,    87,    78,    79,
      37,    84,    87,   569,   609,   571,   592,    55,   594,    78,
      57,    78,    20,    55,    80,    79,    78,    89,   623,    87,
       1,    76,    79,    88,   628,   629,    43,    44,    45,    46,
      47,    48,    49,    85,     1,   640,    87,    78,    43,    44,
      45,    46,    47,    48,    49,   650,    78,    56,    79,    79,
     616,    55,    77,    80,    88,    76,   661,    74,    50,    76,
     626,    78,    43,    44,    45,    46,    47,    48,    49,    74,
      76,    81,   677,    78,    88,     1,    43,    44,    45,    46,
      47,    48,    49,    79,   605,    79,    77,    88,    81,    79,
      79,    76,   697,    74,    80,     1,    79,    78,   664,    77,
      79,    88,    79,    79,     5,    80,   672,    74,    82,    79,
      79,    78,    70,    79,    55,   191,   721,    43,    44,    45,
      46,    47,    48,    49,   157,   132,   318,   648,   649,   178,
     313,   447,   653,   404,   325,   321,   304,    43,    44,    45,
      46,    47,    48,    49,   438,    70,   410,    70,    74,   407,
     671,   233,    78,   498,   629,   543,   338,   591,   324,   562,
     551,   531,   580,   684,   685,    -1,   687,   688,    74,    -1,
      -1,    -1,    78,     1,    -1,     3,     4,     5,    -1,     7,
       8,     9,   703,    11,    12,    13,    14,    15,    -1,    -1,
      -1,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      28,    29,    30,    -1,    32,    33,    34,    35,    36,    37,
      38,    39,    40,    41,    42,    43,    44,    45,    46,    47,
      48,    49,    50,    -1,    52,    -1,    54,    55,    -1,    -1,
      -1,    -1,    -1,    61,    62,    63,    64,    65,    66,    67,
      68,     1,    -1,    71,    72,    73,    74,    75,     1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,     1,    87,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    14,    15,    16,    -1,    18,    19,    -1,    -1,    22,
      -1,    -1,    -1,    43,    44,    45,    46,    47,    48,    49,
      43,    44,    45,    46,    47,    48,    49,     1,    43,    44,
      45,    46,    47,    48,    49,     1,    -1,    -1,    -1,    -1,
      53,    54,     1,    -1,    74,    -1,    59,    60,    78,    -1,
      -1,    74,    -1,    -1,    -1,    78,    -1,    -1,    71,    74,
      -1,    -1,    -1,    78,    -1,    78,    79,    -1,    -1,    43,
      44,    45,    46,    47,    48,    49,    -1,    43,    44,    45,
      46,    47,    48,    49,    43,    44,    45,    46,    47,    48,
      49,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      74,    -1,    -1,    -1,    78,    -1,    -1,    -1,    74,    -1,
      -1,    -1,    78,    -1,    -1,    74,    -1,     3,     4,    78,
       6,     7,     8,     9,    10,    11,    12,    13,    14,    15,
      16,     1,    18,    19,    -1,    -1,    22,     6,     7,     8,
       9,    -1,    11,    12,    13,    14,    15,    -1,    -1,    -1,
      -1,    21,    -1,    23,    24,    25,    26,    27,    28,    37,
      38,    31,    32,    33,    34,    35,    36,    -1,    -1,    -1,
      -1,    -1,    -1,    42,    -1,    -1,    -1,    55,    56,    57,
      -1,    51,    52,    61,    62,    63,    64,    65,    66,    67,
      -1,    -1,    78,    -1,    -1,    -1,    -1,    75,    68,    -1,
      -1,    -1,    72,    73,    -1,    -1,    -1,    76,    -1,    -1,
      79,     3,     4,     5,    -1,     7,     8,     9,    -1,    11,
      12,    13,    14,    15,    -1,    -1,    -1,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    -1,
      32,    33,    34,    35,    36,    37,    38,    39,    40,    41,
      42,    43,    44,    45,    46,    47,    48,    49,    50,    -1,
      52,    -1,    54,    55,     1,    -1,    -1,    -1,    -1,    61,
      62,    63,    64,    65,    66,    67,    68,    -1,    -1,    71,
      72,    73,    74,    75,    21,    -1,    23,    24,    25,    26,
      27,    28,    -1,    -1,    31,    32,    33,    34,    35,    36,
      -1,     3,     4,     5,    -1,     7,     8,     9,    -1,    11,
      12,    13,    14,    15,    51,    52,    -1,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    -1,
      32,    33,    34,    35,    36,    72,    73,    39,    40,    41,
      42,    43,    44,    45,    46,    47,    48,    49,    50,    51,
      52,    -1,    54,    55,    56,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    68,    -1,    -1,    71,
      72,    73,    74,     3,     4,     5,    -1,     7,     8,     9,
      -1,    11,    12,    13,    14,    15,    -1,    -1,    -1,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    -1,    32,    33,    34,    35,    36,    -1,    -1,    39,
      40,    41,    42,    43,    44,    45,    46,    47,    48,    49,
      50,    -1,    52,    -1,    54,    55,    56,    -1,    -1,    -1,
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
      26,    27,    28,    29,    30,     6,    32,    33,    34,    35,
      36,    -1,    -1,    39,    40,    41,    42,    43,    44,    45,
      46,    47,    48,    49,    50,    -1,    52,    -1,    54,    55,
      -1,    16,    -1,    -1,    -1,    -1,    37,    38,    -1,    -1,
      -1,    -1,    68,    -1,    -1,    71,    72,    73,    74,    -1,
      -1,    -1,    37,    38,    55,    -1,    -1,    -1,    -1,    -1,
      61,    62,    63,    64,    65,    66,    67,    18,    69,    70,
      55,    37,    38,    39,    75,    -1,    61,    62,    63,    64,
      65,    66,    67,    -1,    -1,    -1,    37,    38,    -1,    55,
      75,    -1,    -1,    -1,    -1,    61,    62,    63,    64,    65,
      66,    67,    -1,    -1,    55,    37,    38,    39,    -1,    75,
      61,    62,    63,    64,    65,    66,    67,    -1,    -1,    85,
      37,    38,    -1,    55,    75,    -1,    -1,    -1,    -1,    61,
      62,    63,    64,    65,    66,    67,    37,    38,    55,    -1,
      -1,    -1,    -1,    75,    61,    62,    63,    64,    65,    66,
      67,    37,    -1,    -1,    55,    -1,    -1,    -1,    75,    -1,
      61,    62,    63,    64,    65,    66,    67,    -1,    -1,    55,
      -1,    -1,    -1,    -1,    75,    61,    62,    63,    64,    65,
      66,    67,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    75
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
     102,    37,    38,    61,    62,    63,    64,    65,    66,    67,
      75,    99,   109,   111,   112,   113,   114,   117,   118,   171,
      78,    99,    77,    58,    84,   177,    16,    38,   112,   113,
     114,   115,   116,   119,    37,   125,   125,    87,   125,   111,
     164,    87,   129,   129,   129,   129,    87,   133,   146,    87,
     126,    98,    57,   165,    81,   102,    11,    12,    13,    14,
      15,    16,   148,   149,   150,   151,   152,    96,    97,   117,
      62,    66,    61,    62,    63,    64,   111,    81,   108,    83,
      83,    83,    38,    86,   111,   102,    77,    78,    84,    87,
     177,    79,   180,   112,   116,    38,    84,    86,    99,    99,
      99,    68,    99,    80,    30,    51,   130,   135,    98,   110,
     110,   110,   110,    51,    56,    98,   132,   134,    87,   146,
      87,   133,    40,    41,   127,   128,   110,    18,    75,   115,
     119,   155,   156,    79,   129,   129,   129,   129,   146,   126,
      61,    62,    56,    57,   105,   106,   107,   119,    86,    98,
      79,    15,    55,   177,    58,   176,   177,    85,    96,    83,
      83,    86,    87,   121,   122,   205,    84,    81,    84,    88,
      81,    84,   164,    88,    80,   108,    77,   141,   141,   141,
     141,    98,    88,    80,    88,   110,   110,    88,    80,    78,
      98,    55,    89,   154,    98,    80,    82,    97,    98,    98,
      98,    98,    98,    98,    80,    82,   108,    79,    80,    85,
      88,   177,    99,    98,   122,   120,   177,   125,   106,   125,
     125,   106,   125,   130,   111,   142,    78,    96,   159,   159,
     159,   159,    88,   134,   141,   141,   127,    17,    79,   136,
     138,   139,    89,   153,   153,    56,    57,   111,   154,   156,
     141,   141,   141,   141,   141,    78,    96,   106,    84,   188,
     177,   176,   177,   177,   122,    85,    88,   206,    85,    82,
      85,    99,    82,    85,    80,     1,    42,   157,   160,   161,
     166,   167,   169,   159,   159,   119,   139,    79,   119,   159,
     159,   159,   159,   159,   139,    39,    85,   119,   178,   181,
     186,    84,    84,    84,    84,   142,     1,    87,   172,   169,
      79,    96,   160,    98,    79,   119,   178,    98,   177,    80,
      85,   186,   125,   125,   125,     1,    21,    23,    24,    25,
      26,    27,    28,    31,    32,    33,    34,    35,    36,    51,
      52,    68,    72,    73,   173,   174,    55,    98,   171,    97,
      87,   137,    96,    98,   177,    87,    89,   136,    88,   186,
      85,    85,    85,    85,    57,   131,    88,    88,    80,   188,
      98,    88,    96,    88,    56,    57,    99,   175,    37,   173,
       1,    43,    44,    45,    46,    47,    48,    49,    74,    78,
      96,   179,   191,   196,   198,   188,    98,    84,   202,    87,
     202,    55,   203,   204,    78,    57,   195,   202,    78,   192,
     198,   177,    20,   190,   188,   177,   200,    55,   200,   188,
     205,    80,    78,   198,   193,   198,   179,   200,    43,    44,
      45,    47,    48,    49,    74,   179,   194,   196,   197,    79,
     192,   180,    89,   191,    87,   189,    76,    88,    85,   201,
     177,   204,    79,   192,    79,   192,   177,   201,   202,    87,
     202,    78,   195,   202,    78,   177,    79,   194,    97,    97,
      56,     6,    69,    70,    88,   119,   182,   185,   187,   179,
     200,   202,    78,   198,   206,    79,   180,    78,   198,   177,
      55,   177,   193,   179,   177,   194,   180,    98,    77,    80,
      88,   177,    76,   200,   192,   188,    97,   192,    50,   199,
      76,    88,   201,    79,   177,   201,    79,    97,    81,   119,
     178,   184,   187,   180,   200,    77,    79,    79,    78,   198,
     177,   202,    78,   198,   180,    78,   198,    98,   183,    98,
     177,    81,    98,   201,   200,   199,   192,    76,   177,   192,
      97,   192,   199,    82,    84,    87,    90,    91,    81,    88,
     183,    96,    78,   198,    80,    79,   177,    77,    79,    79,
     183,    56,   183,    82,    98,   183,    82,   192,   200,   201,
     177,   199,    85,    88,    88,    98,    82,    79,   201,    78,
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
     117,   118,   118,   118,   119,   119,   119,   120,   121,   122,
     122,   123,   124,   125,   125,   126,   126,   127,   127,   128,
     128,   129,   129,   130,   130,   131,   131,   132,   133,   133,
     134,   134,   135,   135,   136,   136,   137,   137,   138,   139,
     139,   140,   140,   140,   141,   141,   142,   142,   143,   143,
     144,   145,   146,   146,   147,   147,   148,   148,   149,   150,
     151,   152,   152,   153,   153,   154,   154,   154,   154,   155,
     155,   155,   155,   156,   156,   157,   158,   158,   158,   158,
     158,   159,   159,   160,   160,   161,   161,   161,   161,   161,
     161,   161,   162,   162,   162,   162,   162,   163,   163,   163,
     163,   164,   164,   165,   166,   167,   167,   167,   167,   168,
     168,   168,   168,   168,   168,   168,   168,   168,   168,   168,
     169,   169,   169,   170,   170,   171,   172,   172,   172,   173,
     173,   174,   174,   174,   174,   174,   174,   174,   174,   174,
     174,   174,   174,   174,   174,   174,   174,   174,   174,   175,
     175,   175,   176,   176,   176,   177,   177,   177,   177,   177,
     177,   178,   179,   180,   181,   181,   181,   181,   181,   182,
     182,   182,   183,   183,   183,   183,   183,   183,   184,   185,
     185,   185,   186,   186,   187,   187,   188,   188,   189,   189,
     190,   190,   191,   191,   191,   192,   192,   193,   193,   194,
     194,   194,   195,   195,   196,   196,   196,   197,   197,   197,
     197,   197,   197,   197,   197,   197,   197,   197,   197,   198,
     198,   198,   198,   198,   198,   198,   198,   198,   198,   198,
     198,   198,   198,   199,   199,   199,   200,   201,   202,   203,
     203,   204,   204,   205,   206,   207,   208
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
       1,     2,     2,     1,     2,     1,     2,     2,     2,     8,
       1,     1,     1,     1,     2,     2,     1,     1,     1,     2,
       2,     3,     2,     1,     3,     2,     1,     1,     3,     0,
       2,     4,     6,     0,     1,     0,     3,     1,     3,     1,
       1,     0,     3,     1,     3,     0,     1,     1,     0,     3,
       1,     3,     1,     1,     0,     1,     0,     2,     5,     1,
       2,     3,     5,     6,     0,     2,     1,     3,     5,     5,
       5,     5,     4,     3,     6,     6,     5,     5,     5,     5,
       5,     4,     7,     0,     2,     0,     2,     2,     2,     3,
       3,     2,     3,     1,     3,     4,     2,     2,     2,     2,
       2,     1,     4,     0,     2,     1,     1,     1,     1,     2,
       2,     2,     3,     6,     9,     3,     6,     3,     6,     9,
       9,     1,     3,     1,     1,     1,     2,     2,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       7,     5,    13,     5,     2,     1,     0,     3,     1,     1,
       3,     1,     1,     1,     1,     1,     1,     1,     1,     1,
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
#line 197 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.modlist) = (yyvsp[0].modlist); modlist = (yyvsp[0].modlist); }
#line 2248 "y.tab.c" /* yacc.c:1661  */
    break;

  case 3:
#line 201 "xi-grammar.y" /* yacc.c:1661  */
    {
		  (yyval.modlist) = 0;
		}
#line 2256 "y.tab.c" /* yacc.c:1661  */
    break;

  case 4:
#line 205 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.modlist) = new AstChildren<Module>(lineno, (yyvsp[-1].module), (yyvsp[0].modlist)); }
#line 2262 "y.tab.c" /* yacc.c:1661  */
    break;

  case 5:
#line 209 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 0; }
#line 2268 "y.tab.c" /* yacc.c:1661  */
    break;

  case 6:
#line 211 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 1; }
#line 2274 "y.tab.c" /* yacc.c:1661  */
    break;

  case 7:
#line 215 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 1; }
#line 2280 "y.tab.c" /* yacc.c:1661  */
    break;

  case 8:
#line 217 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 2; }
#line 2286 "y.tab.c" /* yacc.c:1661  */
    break;

  case 9:
#line 221 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 0; }
#line 2292 "y.tab.c" /* yacc.c:1661  */
    break;

  case 10:
#line 223 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 1; }
#line 2298 "y.tab.c" /* yacc.c:1661  */
    break;

  case 11:
#line 228 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 2304 "y.tab.c" /* yacc.c:1661  */
    break;

  case 12:
#line 229 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(MODULE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2310 "y.tab.c" /* yacc.c:1661  */
    break;

  case 13:
#line 230 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(MAINMODULE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2316 "y.tab.c" /* yacc.c:1661  */
    break;

  case 14:
#line 231 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(EXTERN, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2322 "y.tab.c" /* yacc.c:1661  */
    break;

  case 15:
#line 233 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(INITCALL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2328 "y.tab.c" /* yacc.c:1661  */
    break;

  case 16:
#line 234 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(INITNODE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2334 "y.tab.c" /* yacc.c:1661  */
    break;

  case 17:
#line 235 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(INITPROC, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2340 "y.tab.c" /* yacc.c:1661  */
    break;

  case 18:
#line 237 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(CHARE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2346 "y.tab.c" /* yacc.c:1661  */
    break;

  case 19:
#line 238 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(MAINCHARE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2352 "y.tab.c" /* yacc.c:1661  */
    break;

  case 20:
#line 239 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(GROUP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2358 "y.tab.c" /* yacc.c:1661  */
    break;

  case 21:
#line 240 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(NODEGROUP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2364 "y.tab.c" /* yacc.c:1661  */
    break;

  case 22:
#line 241 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(ARRAY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2370 "y.tab.c" /* yacc.c:1661  */
    break;

  case 23:
#line 245 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(INCLUDE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2376 "y.tab.c" /* yacc.c:1661  */
    break;

  case 24:
#line 246 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(STACKSIZE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2382 "y.tab.c" /* yacc.c:1661  */
    break;

  case 25:
#line 247 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(THREADED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2388 "y.tab.c" /* yacc.c:1661  */
    break;

  case 26:
#line 248 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(TEMPLATE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2394 "y.tab.c" /* yacc.c:1661  */
    break;

  case 27:
#line 249 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(SYNC, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2400 "y.tab.c" /* yacc.c:1661  */
    break;

  case 28:
#line 250 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(IGET, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2406 "y.tab.c" /* yacc.c:1661  */
    break;

  case 29:
#line 251 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(EXCLUSIVE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2412 "y.tab.c" /* yacc.c:1661  */
    break;

  case 30:
#line 252 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(IMMEDIATE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2418 "y.tab.c" /* yacc.c:1661  */
    break;

  case 31:
#line 253 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(SKIPSCHED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2424 "y.tab.c" /* yacc.c:1661  */
    break;

  case 32:
#line 254 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(NOCOPY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2430 "y.tab.c" /* yacc.c:1661  */
    break;

  case 33:
#line 255 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(INLINE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2436 "y.tab.c" /* yacc.c:1661  */
    break;

  case 34:
#line 256 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(VIRTUAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2442 "y.tab.c" /* yacc.c:1661  */
    break;

  case 35:
#line 257 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(MIGRATABLE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2448 "y.tab.c" /* yacc.c:1661  */
    break;

  case 36:
#line 258 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(CREATEHERE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2454 "y.tab.c" /* yacc.c:1661  */
    break;

  case 37:
#line 259 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(CREATEHOME, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2460 "y.tab.c" /* yacc.c:1661  */
    break;

  case 38:
#line 260 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(NOKEEP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2466 "y.tab.c" /* yacc.c:1661  */
    break;

  case 39:
#line 261 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(NOTRACE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2472 "y.tab.c" /* yacc.c:1661  */
    break;

  case 40:
#line 262 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(APPWORK, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2478 "y.tab.c" /* yacc.c:1661  */
    break;

  case 41:
#line 265 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(PACKED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2484 "y.tab.c" /* yacc.c:1661  */
    break;

  case 42:
#line 266 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(VARSIZE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2490 "y.tab.c" /* yacc.c:1661  */
    break;

  case 43:
#line 267 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(ENTRY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2496 "y.tab.c" /* yacc.c:1661  */
    break;

  case 44:
#line 268 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(FOR, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2502 "y.tab.c" /* yacc.c:1661  */
    break;

  case 45:
#line 269 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(FORALL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2508 "y.tab.c" /* yacc.c:1661  */
    break;

  case 46:
#line 270 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(WHILE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2514 "y.tab.c" /* yacc.c:1661  */
    break;

  case 47:
#line 271 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(WHEN, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2520 "y.tab.c" /* yacc.c:1661  */
    break;

  case 48:
#line 272 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(OVERLAP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2526 "y.tab.c" /* yacc.c:1661  */
    break;

  case 49:
#line 273 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(SERIAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2532 "y.tab.c" /* yacc.c:1661  */
    break;

  case 50:
#line 274 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(IF, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2538 "y.tab.c" /* yacc.c:1661  */
    break;

  case 51:
#line 275 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(ELSE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2544 "y.tab.c" /* yacc.c:1661  */
    break;

  case 52:
#line 277 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(LOCAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2550 "y.tab.c" /* yacc.c:1661  */
    break;

  case 53:
#line 279 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(USING, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2556 "y.tab.c" /* yacc.c:1661  */
    break;

  case 54:
#line 280 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(ACCEL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2562 "y.tab.c" /* yacc.c:1661  */
    break;

  case 55:
#line 283 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(ACCELBLOCK, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2568 "y.tab.c" /* yacc.c:1661  */
    break;

  case 56:
#line 284 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(MEMCRITICAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2574 "y.tab.c" /* yacc.c:1661  */
    break;

  case 57:
#line 285 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(REDUCTIONTARGET, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2580 "y.tab.c" /* yacc.c:1661  */
    break;

  case 58:
#line 286 "xi-grammar.y" /* yacc.c:1661  */
    { ReservedWord(CASE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2586 "y.tab.c" /* yacc.c:1661  */
    break;

  case 59:
#line 291 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 2592 "y.tab.c" /* yacc.c:1661  */
    break;

  case 60:
#line 293 "xi-grammar.y" /* yacc.c:1661  */
    {
		  char *tmp = new char[strlen((yyvsp[-3].strval))+strlen((yyvsp[0].strval))+3];
		  sprintf(tmp,"%s::%s", (yyvsp[-3].strval), (yyvsp[0].strval));
		  (yyval.strval) = tmp;
		}
#line 2602 "y.tab.c" /* yacc.c:1661  */
    break;

  case 61:
#line 299 "xi-grammar.y" /* yacc.c:1661  */
    {
		  char *tmp = new char[strlen((yyvsp[-3].strval))+5+3];
		  sprintf(tmp,"%s::array", (yyvsp[-3].strval));
		  (yyval.strval) = tmp;
		}
#line 2612 "y.tab.c" /* yacc.c:1661  */
    break;

  case 62:
#line 306 "xi-grammar.y" /* yacc.c:1661  */
    {
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist));
		}
#line 2620 "y.tab.c" /* yacc.c:1661  */
    break;

  case 63:
#line 310 "xi-grammar.y" /* yacc.c:1661  */
    {
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist));
		    (yyval.module)->setMain();
		}
#line 2629 "y.tab.c" /* yacc.c:1661  */
    break;

  case 64:
#line 317 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.conslist) = 0; }
#line 2635 "y.tab.c" /* yacc.c:1661  */
    break;

  case 65:
#line 319 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.conslist) = (yyvsp[-2].conslist); }
#line 2641 "y.tab.c" /* yacc.c:1661  */
    break;

  case 66:
#line 323 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.conslist) = 0; }
#line 2647 "y.tab.c" /* yacc.c:1661  */
    break;

  case 67:
#line 325 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.conslist) = new ConstructList(lineno, (yyvsp[-1].construct), (yyvsp[0].conslist)); }
#line 2653 "y.tab.c" /* yacc.c:1661  */
    break;

  case 68:
#line 329 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.construct) = new UsingScope((yyvsp[0].strval), false); }
#line 2659 "y.tab.c" /* yacc.c:1661  */
    break;

  case 69:
#line 331 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.construct) = new UsingScope((yyvsp[0].strval), true); }
#line 2665 "y.tab.c" /* yacc.c:1661  */
    break;

  case 70:
#line 333 "xi-grammar.y" /* yacc.c:1661  */
    { (yyvsp[0].member)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].member); }
#line 2671 "y.tab.c" /* yacc.c:1661  */
    break;

  case 71:
#line 335 "xi-grammar.y" /* yacc.c:1661  */
    { (yyvsp[0].message)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].message); }
#line 2677 "y.tab.c" /* yacc.c:1661  */
    break;

  case 72:
#line 337 "xi-grammar.y" /* yacc.c:1661  */
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
#line 2692 "y.tab.c" /* yacc.c:1661  */
    break;

  case 73:
#line 350 "xi-grammar.y" /* yacc.c:1661  */
    { if((yyvsp[-2].conslist)) (yyvsp[-2].conslist)->recurse<int&>((yyvsp[-4].intval), &Construct::setExtern); (yyval.construct) = (yyvsp[-2].conslist); }
#line 2698 "y.tab.c" /* yacc.c:1661  */
    break;

  case 74:
#line 352 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.construct) = new Scope((yyvsp[-3].strval), (yyvsp[-1].conslist)); }
#line 2704 "y.tab.c" /* yacc.c:1661  */
    break;

  case 75:
#line 354 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.construct) = (yyvsp[-1].construct); }
#line 2710 "y.tab.c" /* yacc.c:1661  */
    break;

  case 76:
#line 356 "xi-grammar.y" /* yacc.c:1661  */
    {
          ERROR("preceding construct must be semicolon terminated",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 2720 "y.tab.c" /* yacc.c:1661  */
    break;

  case 77:
#line 362 "xi-grammar.y" /* yacc.c:1661  */
    { (yyvsp[0].module)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].module); }
#line 2726 "y.tab.c" /* yacc.c:1661  */
    break;

  case 78:
#line 364 "xi-grammar.y" /* yacc.c:1661  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2732 "y.tab.c" /* yacc.c:1661  */
    break;

  case 79:
#line 366 "xi-grammar.y" /* yacc.c:1661  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2738 "y.tab.c" /* yacc.c:1661  */
    break;

  case 80:
#line 368 "xi-grammar.y" /* yacc.c:1661  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2744 "y.tab.c" /* yacc.c:1661  */
    break;

  case 81:
#line 370 "xi-grammar.y" /* yacc.c:1661  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2750 "y.tab.c" /* yacc.c:1661  */
    break;

  case 82:
#line 372 "xi-grammar.y" /* yacc.c:1661  */
    { (yyvsp[0].templat)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].templat); }
#line 2756 "y.tab.c" /* yacc.c:1661  */
    break;

  case 83:
#line 374 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.construct) = NULL; }
#line 2762 "y.tab.c" /* yacc.c:1661  */
    break;

  case 84:
#line 376 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.construct) = NULL; }
#line 2768 "y.tab.c" /* yacc.c:1661  */
    break;

  case 85:
#line 378 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.construct) = (yyvsp[0].accelBlock); }
#line 2774 "y.tab.c" /* yacc.c:1661  */
    break;

  case 86:
#line 380 "xi-grammar.y" /* yacc.c:1661  */
    {
          ERROR("invalid construct",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 2784 "y.tab.c" /* yacc.c:1661  */
    break;

  case 87:
#line 388 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tparam) = new TParamType((yyvsp[0].type)); }
#line 2790 "y.tab.c" /* yacc.c:1661  */
    break;

  case 88:
#line 390 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
#line 2796 "y.tab.c" /* yacc.c:1661  */
    break;

  case 89:
#line 392 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
#line 2802 "y.tab.c" /* yacc.c:1661  */
    break;

  case 90:
#line 396 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tparlist) = new TParamList((yyvsp[0].tparam)); }
#line 2808 "y.tab.c" /* yacc.c:1661  */
    break;

  case 91:
#line 398 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tparlist) = new TParamList((yyvsp[-2].tparam), (yyvsp[0].tparlist)); }
#line 2814 "y.tab.c" /* yacc.c:1661  */
    break;

  case 92:
#line 402 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tparlist) = new TParamList(0); }
#line 2820 "y.tab.c" /* yacc.c:1661  */
    break;

  case 93:
#line 404 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tparlist) = (yyvsp[0].tparlist); }
#line 2826 "y.tab.c" /* yacc.c:1661  */
    break;

  case 94:
#line 408 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tparlist) = 0; }
#line 2832 "y.tab.c" /* yacc.c:1661  */
    break;

  case 95:
#line 410 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tparlist) = (yyvsp[-1].tparlist); }
#line 2838 "y.tab.c" /* yacc.c:1661  */
    break;

  case 96:
#line 414 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("int"); }
#line 2844 "y.tab.c" /* yacc.c:1661  */
    break;

  case 97:
#line 416 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("long"); }
#line 2850 "y.tab.c" /* yacc.c:1661  */
    break;

  case 98:
#line 418 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("short"); }
#line 2856 "y.tab.c" /* yacc.c:1661  */
    break;

  case 99:
#line 420 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("char"); }
#line 2862 "y.tab.c" /* yacc.c:1661  */
    break;

  case 100:
#line 422 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("unsigned int"); }
#line 2868 "y.tab.c" /* yacc.c:1661  */
    break;

  case 101:
#line 424 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("unsigned long"); }
#line 2874 "y.tab.c" /* yacc.c:1661  */
    break;

  case 102:
#line 426 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("unsigned long"); }
#line 2880 "y.tab.c" /* yacc.c:1661  */
    break;

  case 103:
#line 428 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("unsigned long long"); }
#line 2886 "y.tab.c" /* yacc.c:1661  */
    break;

  case 104:
#line 430 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("unsigned short"); }
#line 2892 "y.tab.c" /* yacc.c:1661  */
    break;

  case 105:
#line 432 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("unsigned char"); }
#line 2898 "y.tab.c" /* yacc.c:1661  */
    break;

  case 106:
#line 434 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("long long"); }
#line 2904 "y.tab.c" /* yacc.c:1661  */
    break;

  case 107:
#line 436 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("float"); }
#line 2910 "y.tab.c" /* yacc.c:1661  */
    break;

  case 108:
#line 438 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("double"); }
#line 2916 "y.tab.c" /* yacc.c:1661  */
    break;

  case 109:
#line 440 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("long double"); }
#line 2922 "y.tab.c" /* yacc.c:1661  */
    break;

  case 110:
#line 442 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new BuiltinType("void"); }
#line 2928 "y.tab.c" /* yacc.c:1661  */
    break;

  case 111:
#line 445 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.ntype) = new NamedType((yyvsp[-1].strval),(yyvsp[0].tparlist)); }
#line 2934 "y.tab.c" /* yacc.c:1661  */
    break;

  case 112:
#line 446 "xi-grammar.y" /* yacc.c:1661  */
    {
                    const char* basename, *scope;
                    splitScopedName((yyvsp[-1].strval), &scope, &basename);
                    (yyval.ntype) = new NamedType(basename, (yyvsp[0].tparlist), scope);
                }
#line 2944 "y.tab.c" /* yacc.c:1661  */
    break;

  case 113:
#line 454 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].type); }
#line 2950 "y.tab.c" /* yacc.c:1661  */
    break;

  case 114:
#line 456 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new TypenameType((yyvsp[0].ntype)); }
#line 2956 "y.tab.c" /* yacc.c:1661  */
    break;

  case 115:
#line 458 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].ntype); }
#line 2962 "y.tab.c" /* yacc.c:1661  */
    break;

  case 116:
#line 462 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.ptype) = new PtrType((yyvsp[-1].type)); }
#line 2968 "y.tab.c" /* yacc.c:1661  */
    break;

  case 117:
#line 466 "xi-grammar.y" /* yacc.c:1661  */
    { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
#line 2974 "y.tab.c" /* yacc.c:1661  */
    break;

  case 118:
#line 468 "xi-grammar.y" /* yacc.c:1661  */
    { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
#line 2980 "y.tab.c" /* yacc.c:1661  */
    break;

  case 119:
#line 472 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.ftype) = new FuncType((yyvsp[-7].type), (yyvsp[-4].strval), (yyvsp[-1].plist)); }
#line 2986 "y.tab.c" /* yacc.c:1661  */
    break;

  case 120:
#line 476 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].type); }
#line 2992 "y.tab.c" /* yacc.c:1661  */
    break;

  case 121:
#line 478 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 2998 "y.tab.c" /* yacc.c:1661  */
    break;

  case 122:
#line 480 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3004 "y.tab.c" /* yacc.c:1661  */
    break;

  case 123:
#line 482 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].ftype); }
#line 3010 "y.tab.c" /* yacc.c:1661  */
    break;

  case 124:
#line 484 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new ConstType((yyvsp[0].type)); }
#line 3016 "y.tab.c" /* yacc.c:1661  */
    break;

  case 125:
#line 486 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new ConstType((yyvsp[-1].type)); }
#line 3022 "y.tab.c" /* yacc.c:1661  */
    break;

  case 126:
#line 490 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3028 "y.tab.c" /* yacc.c:1661  */
    break;

  case 127:
#line 492 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3034 "y.tab.c" /* yacc.c:1661  */
    break;

  case 128:
#line 494 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3040 "y.tab.c" /* yacc.c:1661  */
    break;

  case 129:
#line 496 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new ConstType((yyvsp[0].type)); }
#line 3046 "y.tab.c" /* yacc.c:1661  */
    break;

  case 130:
#line 498 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new ConstType((yyvsp[-1].type)); }
#line 3052 "y.tab.c" /* yacc.c:1661  */
    break;

  case 131:
#line 502 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new RValueReferenceType((yyvsp[-2].type)); }
#line 3058 "y.tab.c" /* yacc.c:1661  */
    break;

  case 132:
#line 504 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new ReferenceType((yyvsp[-1].type)); }
#line 3064 "y.tab.c" /* yacc.c:1661  */
    break;

  case 133:
#line 506 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3070 "y.tab.c" /* yacc.c:1661  */
    break;

  case 134:
#line 510 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new RValueReferenceType((yyvsp[-2].type)); }
#line 3076 "y.tab.c" /* yacc.c:1661  */
    break;

  case 135:
#line 512 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = new ReferenceType((yyvsp[-1].type)); }
#line 3082 "y.tab.c" /* yacc.c:1661  */
    break;

  case 136:
#line 514 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3088 "y.tab.c" /* yacc.c:1661  */
    break;

  case 137:
#line 518 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 3094 "y.tab.c" /* yacc.c:1661  */
    break;

  case 138:
#line 522 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.val) = (yyvsp[-1].val); }
#line 3100 "y.tab.c" /* yacc.c:1661  */
    break;

  case 139:
#line 526 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.vallist) = 0; }
#line 3106 "y.tab.c" /* yacc.c:1661  */
    break;

  case 140:
#line 528 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.vallist) = new ValueList((yyvsp[-1].val), (yyvsp[0].vallist)); }
#line 3112 "y.tab.c" /* yacc.c:1661  */
    break;

  case 141:
#line 532 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].vallist)); }
#line 3118 "y.tab.c" /* yacc.c:1661  */
    break;

  case 142:
#line 536 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[-3].type), (yyvsp[-1].strval), (yyvsp[0].vallist), 1); }
#line 3124 "y.tab.c" /* yacc.c:1661  */
    break;

  case 143:
#line 540 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 0;}
#line 3130 "y.tab.c" /* yacc.c:1661  */
    break;

  case 144:
#line 542 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 0;}
#line 3136 "y.tab.c" /* yacc.c:1661  */
    break;

  case 145:
#line 546 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 0; }
#line 3142 "y.tab.c" /* yacc.c:1661  */
    break;

  case 146:
#line 548 "xi-grammar.y" /* yacc.c:1661  */
    {
		  /*
		  printf("Warning: Message attributes are being phased out.\n");
		  printf("Warning: Please remove them from interface files.\n");
		  */
		  (yyval.intval) = (yyvsp[-1].intval);
		}
#line 3154 "y.tab.c" /* yacc.c:1661  */
    break;

  case 147:
#line 558 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = (yyvsp[0].intval); }
#line 3160 "y.tab.c" /* yacc.c:1661  */
    break;

  case 148:
#line 560 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = (yyvsp[-2].intval) | (yyvsp[0].intval); }
#line 3166 "y.tab.c" /* yacc.c:1661  */
    break;

  case 149:
#line 564 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 0; }
#line 3172 "y.tab.c" /* yacc.c:1661  */
    break;

  case 150:
#line 566 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 0; }
#line 3178 "y.tab.c" /* yacc.c:1661  */
    break;

  case 151:
#line 570 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = 0; }
#line 3184 "y.tab.c" /* yacc.c:1661  */
    break;

  case 152:
#line 572 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = (yyvsp[-1].cattr); }
#line 3190 "y.tab.c" /* yacc.c:1661  */
    break;

  case 153:
#line 576 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = (yyvsp[0].cattr); }
#line 3196 "y.tab.c" /* yacc.c:1661  */
    break;

  case 154:
#line 578 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
#line 3202 "y.tab.c" /* yacc.c:1661  */
    break;

  case 155:
#line 582 "xi-grammar.y" /* yacc.c:1661  */
    { python_doc = NULL; (yyval.intval) = 0; }
#line 3208 "y.tab.c" /* yacc.c:1661  */
    break;

  case 156:
#line 584 "xi-grammar.y" /* yacc.c:1661  */
    { python_doc = (yyvsp[0].strval); (yyval.intval) = 0; }
#line 3214 "y.tab.c" /* yacc.c:1661  */
    break;

  case 157:
#line 588 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = Chare::CPYTHON; }
#line 3220 "y.tab.c" /* yacc.c:1661  */
    break;

  case 158:
#line 592 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = 0; }
#line 3226 "y.tab.c" /* yacc.c:1661  */
    break;

  case 159:
#line 594 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = (yyvsp[-1].cattr); }
#line 3232 "y.tab.c" /* yacc.c:1661  */
    break;

  case 160:
#line 598 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = (yyvsp[0].cattr); }
#line 3238 "y.tab.c" /* yacc.c:1661  */
    break;

  case 161:
#line 600 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
#line 3244 "y.tab.c" /* yacc.c:1661  */
    break;

  case 162:
#line 604 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = Chare::CMIGRATABLE; }
#line 3250 "y.tab.c" /* yacc.c:1661  */
    break;

  case 163:
#line 606 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.cattr) = Chare::CPYTHON; }
#line 3256 "y.tab.c" /* yacc.c:1661  */
    break;

  case 164:
#line 610 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 0; }
#line 3262 "y.tab.c" /* yacc.c:1661  */
    break;

  case 165:
#line 612 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 1; }
#line 3268 "y.tab.c" /* yacc.c:1661  */
    break;

  case 166:
#line 615 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 0; }
#line 3274 "y.tab.c" /* yacc.c:1661  */
    break;

  case 167:
#line 617 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 1; }
#line 3280 "y.tab.c" /* yacc.c:1661  */
    break;

  case 168:
#line 620 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.mv) = new MsgVar((yyvsp[-3].type), (yyvsp[-2].strval), (yyvsp[-4].intval), (yyvsp[-1].intval)); }
#line 3286 "y.tab.c" /* yacc.c:1661  */
    break;

  case 169:
#line 624 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.mvlist) = new MsgVarList((yyvsp[0].mv)); }
#line 3292 "y.tab.c" /* yacc.c:1661  */
    break;

  case 170:
#line 626 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.mvlist) = new MsgVarList((yyvsp[-1].mv), (yyvsp[0].mvlist)); }
#line 3298 "y.tab.c" /* yacc.c:1661  */
    break;

  case 171:
#line 630 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.message) = new Message(lineno, (yyvsp[0].ntype)); }
#line 3304 "y.tab.c" /* yacc.c:1661  */
    break;

  case 172:
#line 632 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.message) = new Message(lineno, (yyvsp[-2].ntype)); }
#line 3310 "y.tab.c" /* yacc.c:1661  */
    break;

  case 173:
#line 634 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.message) = new Message(lineno, (yyvsp[-3].ntype), (yyvsp[-1].mvlist)); }
#line 3316 "y.tab.c" /* yacc.c:1661  */
    break;

  case 174:
#line 638 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.typelist) = 0; }
#line 3322 "y.tab.c" /* yacc.c:1661  */
    break;

  case 175:
#line 640 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.typelist) = (yyvsp[0].typelist); }
#line 3328 "y.tab.c" /* yacc.c:1661  */
    break;

  case 176:
#line 644 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.typelist) = new TypeList((yyvsp[0].ntype)); }
#line 3334 "y.tab.c" /* yacc.c:1661  */
    break;

  case 177:
#line 646 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.typelist) = new TypeList((yyvsp[-2].ntype), (yyvsp[0].typelist)); }
#line 3340 "y.tab.c" /* yacc.c:1661  */
    break;

  case 178:
#line 650 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr)|Chare::CCHARE, (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3346 "y.tab.c" /* yacc.c:1661  */
    break;

  case 179:
#line 652 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3352 "y.tab.c" /* yacc.c:1661  */
    break;

  case 180:
#line 656 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3358 "y.tab.c" /* yacc.c:1661  */
    break;

  case 181:
#line 660 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.chare) = new NodeGroup(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3364 "y.tab.c" /* yacc.c:1661  */
    break;

  case 182:
#line 664 "xi-grammar.y" /* yacc.c:1661  */
    {/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",(yyvsp[-2].strval));
			(yyval.ntype) = new NamedType(buf);
		}
#line 3374 "y.tab.c" /* yacc.c:1661  */
    break;

  case 183:
#line 670 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.ntype) = new NamedType((yyvsp[-1].strval)); }
#line 3380 "y.tab.c" /* yacc.c:1661  */
    break;

  case 184:
#line 674 "xi-grammar.y" /* yacc.c:1661  */
    {  (yyval.chare) = new Array(lineno, (yyvsp[-4].cattr), (yyvsp[-3].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3386 "y.tab.c" /* yacc.c:1661  */
    break;

  case 185:
#line 676 "xi-grammar.y" /* yacc.c:1661  */
    {  (yyval.chare) = new Array(lineno, (yyvsp[-3].cattr), (yyvsp[-4].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3392 "y.tab.c" /* yacc.c:1661  */
    break;

  case 186:
#line 680 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr)|Chare::CCHARE, new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist));}
#line 3398 "y.tab.c" /* yacc.c:1661  */
    break;

  case 187:
#line 682 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3404 "y.tab.c" /* yacc.c:1661  */
    break;

  case 188:
#line 686 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3410 "y.tab.c" /* yacc.c:1661  */
    break;

  case 189:
#line 690 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.chare) = new NodeGroup( lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3416 "y.tab.c" /* yacc.c:1661  */
    break;

  case 190:
#line 694 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.chare) = new Array( lineno, 0, (yyvsp[-3].ntype), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3422 "y.tab.c" /* yacc.c:1661  */
    break;

  case 191:
#line 698 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-1].strval))); }
#line 3428 "y.tab.c" /* yacc.c:1661  */
    break;

  case 192:
#line 700 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-4].strval)), (yyvsp[-2].mvlist)); }
#line 3434 "y.tab.c" /* yacc.c:1661  */
    break;

  case 193:
#line 704 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = 0; }
#line 3440 "y.tab.c" /* yacc.c:1661  */
    break;

  case 194:
#line 706 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3446 "y.tab.c" /* yacc.c:1661  */
    break;

  case 195:
#line 710 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = 0; }
#line 3452 "y.tab.c" /* yacc.c:1661  */
    break;

  case 196:
#line 712 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3458 "y.tab.c" /* yacc.c:1661  */
    break;

  case 197:
#line 714 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3464 "y.tab.c" /* yacc.c:1661  */
    break;

  case 198:
#line 716 "xi-grammar.y" /* yacc.c:1661  */
    {
		  XStr typeStr;
		  (yyvsp[0].ntype)->print(typeStr);
		  char *tmp = strdup(typeStr.get_string());
		  (yyval.strval) = tmp;
		}
#line 3475 "y.tab.c" /* yacc.c:1661  */
    break;

  case 199:
#line 725 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tvar) = new TType(new NamedType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3481 "y.tab.c" /* yacc.c:1661  */
    break;

  case 200:
#line 727 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tvar) = new TType(new NamedType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3487 "y.tab.c" /* yacc.c:1661  */
    break;

  case 201:
#line 729 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tvar) = new TFunc((yyvsp[-1].ftype), (yyvsp[0].strval)); }
#line 3493 "y.tab.c" /* yacc.c:1661  */
    break;

  case 202:
#line 731 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tvar) = new TName((yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].strval)); }
#line 3499 "y.tab.c" /* yacc.c:1661  */
    break;

  case 203:
#line 735 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tvarlist) = new TVarList((yyvsp[0].tvar)); }
#line 3505 "y.tab.c" /* yacc.c:1661  */
    break;

  case 204:
#line 737 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tvarlist) = new TVarList((yyvsp[-2].tvar), (yyvsp[0].tvarlist)); }
#line 3511 "y.tab.c" /* yacc.c:1661  */
    break;

  case 205:
#line 741 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.tvarlist) = (yyvsp[-1].tvarlist); }
#line 3517 "y.tab.c" /* yacc.c:1661  */
    break;

  case 206:
#line 745 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3523 "y.tab.c" /* yacc.c:1661  */
    break;

  case 207:
#line 747 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3529 "y.tab.c" /* yacc.c:1661  */
    break;

  case 208:
#line 749 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3535 "y.tab.c" /* yacc.c:1661  */
    break;

  case 209:
#line 751 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3541 "y.tab.c" /* yacc.c:1661  */
    break;

  case 210:
#line 753 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].message)); (yyvsp[0].message)->setTemplate((yyval.templat)); }
#line 3547 "y.tab.c" /* yacc.c:1661  */
    break;

  case 211:
#line 757 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.mbrlist) = 0; }
#line 3553 "y.tab.c" /* yacc.c:1661  */
    break;

  case 212:
#line 759 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.mbrlist) = (yyvsp[-2].mbrlist); }
#line 3559 "y.tab.c" /* yacc.c:1661  */
    break;

  case 213:
#line 763 "xi-grammar.y" /* yacc.c:1661  */
    {
                  if (!connectEntries.empty()) {
                    (yyval.mbrlist) = new AstChildren<Member>(connectEntries);
		  } else {
		    (yyval.mbrlist) = 0;
                  }
		}
#line 3571 "y.tab.c" /* yacc.c:1661  */
    break;

  case 214:
#line 771 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.mbrlist) = new AstChildren<Member>(-1, (yyvsp[-1].member), (yyvsp[0].mbrlist)); }
#line 3577 "y.tab.c" /* yacc.c:1661  */
    break;

  case 215:
#line 775 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = (yyvsp[0].readonly); }
#line 3583 "y.tab.c" /* yacc.c:1661  */
    break;

  case 216:
#line 777 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = (yyvsp[0].readonly); }
#line 3589 "y.tab.c" /* yacc.c:1661  */
    break;

  case 218:
#line 780 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = (yyvsp[0].member); }
#line 3595 "y.tab.c" /* yacc.c:1661  */
    break;

  case 219:
#line 782 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = (yyvsp[0].pupable); }
#line 3601 "y.tab.c" /* yacc.c:1661  */
    break;

  case 220:
#line 784 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = (yyvsp[0].includeFile); }
#line 3607 "y.tab.c" /* yacc.c:1661  */
    break;

  case 221:
#line 786 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = new ClassDeclaration(lineno,(yyvsp[0].strval)); }
#line 3613 "y.tab.c" /* yacc.c:1661  */
    break;

  case 222:
#line 790 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1); }
#line 3619 "y.tab.c" /* yacc.c:1661  */
    break;

  case 223:
#line 792 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1); }
#line 3625 "y.tab.c" /* yacc.c:1661  */
    break;

  case 224:
#line 794 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[-6].strval)) + '<' +
					    ((yyvsp[-4].tparlist))->to_string() + '>').c_str()),
				    1);
		}
#line 3635 "y.tab.c" /* yacc.c:1661  */
    break;

  case 225:
#line 800 "xi-grammar.y" /* yacc.c:1661  */
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[-2]).first_column, (yylsp[-2]).last_column, (yylsp[-2]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1);
		}
#line 3645 "y.tab.c" /* yacc.c:1661  */
    break;

  case 226:
#line 806 "xi-grammar.y" /* yacc.c:1661  */
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[-5]).first_column, (yylsp[-5]).last_column, (yylsp[-5]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1);
		}
#line 3655 "y.tab.c" /* yacc.c:1661  */
    break;

  case 227:
#line 815 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 0); }
#line 3661 "y.tab.c" /* yacc.c:1661  */
    break;

  case 228:
#line 817 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 0); }
#line 3667 "y.tab.c" /* yacc.c:1661  */
    break;

  case 229:
#line 819 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[-6].strval)) + '<' +
					    ((yyvsp[-4].tparlist))->to_string() + '>').c_str()),
				    0);
		}
#line 3677 "y.tab.c" /* yacc.c:1661  */
    break;

  case 230:
#line 825 "xi-grammar.y" /* yacc.c:1661  */
    {
                  InitCall* rtn = new InitCall(lineno, (yyvsp[-3].strval), 0);
                  rtn->setAccel();
                  (yyval.member) = rtn;
		}
#line 3687 "y.tab.c" /* yacc.c:1661  */
    break;

  case 231:
#line 833 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[0].ntype),0); }
#line 3693 "y.tab.c" /* yacc.c:1661  */
    break;

  case 232:
#line 835 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[-2].ntype),(yyvsp[0].pupable)); }
#line 3699 "y.tab.c" /* yacc.c:1661  */
    break;

  case 233:
#line 838 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.includeFile) = new IncludeFile(lineno,(yyvsp[0].strval)); }
#line 3705 "y.tab.c" /* yacc.c:1661  */
    break;

  case 234:
#line 842 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = (yyvsp[0].member); }
#line 3711 "y.tab.c" /* yacc.c:1661  */
    break;

  case 235:
#line 846 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = (yyvsp[0].entry); }
#line 3717 "y.tab.c" /* yacc.c:1661  */
    break;

  case 236:
#line 848 "xi-grammar.y" /* yacc.c:1661  */
    {
                  (yyvsp[0].entry)->tspec = (yyvsp[-1].tvarlist);
                  (yyval.member) = (yyvsp[0].entry);
                }
#line 3726 "y.tab.c" /* yacc.c:1661  */
    break;

  case 237:
#line 853 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = (yyvsp[-1].member); }
#line 3732 "y.tab.c" /* yacc.c:1661  */
    break;

  case 238:
#line 855 "xi-grammar.y" /* yacc.c:1661  */
    {
          ERROR("invalid SDAG member",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 3742 "y.tab.c" /* yacc.c:1661  */
    break;

  case 239:
#line 863 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3748 "y.tab.c" /* yacc.c:1661  */
    break;

  case 240:
#line 865 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3754 "y.tab.c" /* yacc.c:1661  */
    break;

  case 241:
#line 867 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3760 "y.tab.c" /* yacc.c:1661  */
    break;

  case 242:
#line 869 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3766 "y.tab.c" /* yacc.c:1661  */
    break;

  case 243:
#line 871 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3772 "y.tab.c" /* yacc.c:1661  */
    break;

  case 244:
#line 873 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3778 "y.tab.c" /* yacc.c:1661  */
    break;

  case 245:
#line 875 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3784 "y.tab.c" /* yacc.c:1661  */
    break;

  case 246:
#line 877 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3790 "y.tab.c" /* yacc.c:1661  */
    break;

  case 247:
#line 879 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3796 "y.tab.c" /* yacc.c:1661  */
    break;

  case 248:
#line 881 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3802 "y.tab.c" /* yacc.c:1661  */
    break;

  case 249:
#line 883 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.member) = 0; }
#line 3808 "y.tab.c" /* yacc.c:1661  */
    break;

  case 250:
#line 886 "xi-grammar.y" /* yacc.c:1661  */
    {
                  (yyval.entry) = new Entry(lineno, (yyvsp[-5].intval), (yyvsp[-4].type), (yyvsp[-3].strval), (yyvsp[-2].plist), (yyvsp[-1].val), (yyvsp[0].sentry), (const char *) NULL, (yylsp[-6]).first_line, (yyloc).last_line);
		  if ((yyvsp[0].sentry) != 0) {
		    (yyvsp[0].sentry)->con1 = new SdagConstruct(SIDENT, (yyvsp[-3].strval));
                    (yyvsp[0].sentry)->setEntry((yyval.entry));
                    (yyvsp[0].sentry)->param = new ParamList((yyvsp[-2].plist));
                  }
                  firstRdma = true;
		}
#line 3822 "y.tab.c" /* yacc.c:1661  */
    break;

  case 251:
#line 896 "xi-grammar.y" /* yacc.c:1661  */
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
#line 3843 "y.tab.c" /* yacc.c:1661  */
    break;

  case 252:
#line 913 "xi-grammar.y" /* yacc.c:1661  */
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
#line 3862 "y.tab.c" /* yacc.c:1661  */
    break;

  case 253:
#line 930 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.accelBlock) = new AccelBlock(lineno, new XStr((yyvsp[-2].strval))); }
#line 3868 "y.tab.c" /* yacc.c:1661  */
    break;

  case 254:
#line 932 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.accelBlock) = new AccelBlock(lineno, NULL); }
#line 3874 "y.tab.c" /* yacc.c:1661  */
    break;

  case 255:
#line 936 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3880 "y.tab.c" /* yacc.c:1661  */
    break;

  case 256:
#line 940 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = 0; }
#line 3886 "y.tab.c" /* yacc.c:1661  */
    break;

  case 257:
#line 942 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = (yyvsp[-1].intval); }
#line 3892 "y.tab.c" /* yacc.c:1661  */
    break;

  case 258:
#line 944 "xi-grammar.y" /* yacc.c:1661  */
    { ERROR("invalid entry method attribute list",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
#line 3901 "y.tab.c" /* yacc.c:1661  */
    break;

  case 259:
#line 951 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = (yyvsp[0].intval); }
#line 3907 "y.tab.c" /* yacc.c:1661  */
    break;

  case 260:
#line 953 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = (yyvsp[-2].intval) | (yyvsp[0].intval); }
#line 3913 "y.tab.c" /* yacc.c:1661  */
    break;

  case 261:
#line 957 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = STHREADED; }
#line 3919 "y.tab.c" /* yacc.c:1661  */
    break;

  case 262:
#line 959 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SSYNC; }
#line 3925 "y.tab.c" /* yacc.c:1661  */
    break;

  case 263:
#line 961 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SIGET; }
#line 3931 "y.tab.c" /* yacc.c:1661  */
    break;

  case 264:
#line 963 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SLOCKED; }
#line 3937 "y.tab.c" /* yacc.c:1661  */
    break;

  case 265:
#line 965 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SCREATEHERE; }
#line 3943 "y.tab.c" /* yacc.c:1661  */
    break;

  case 266:
#line 967 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SCREATEHOME; }
#line 3949 "y.tab.c" /* yacc.c:1661  */
    break;

  case 267:
#line 969 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SNOKEEP; }
#line 3955 "y.tab.c" /* yacc.c:1661  */
    break;

  case 268:
#line 971 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SNOTRACE; }
#line 3961 "y.tab.c" /* yacc.c:1661  */
    break;

  case 269:
#line 973 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SAPPWORK; }
#line 3967 "y.tab.c" /* yacc.c:1661  */
    break;

  case 270:
#line 975 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SIMMEDIATE; }
#line 3973 "y.tab.c" /* yacc.c:1661  */
    break;

  case 271:
#line 977 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SSKIPSCHED; }
#line 3979 "y.tab.c" /* yacc.c:1661  */
    break;

  case 272:
#line 979 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SINLINE; }
#line 3985 "y.tab.c" /* yacc.c:1661  */
    break;

  case 273:
#line 981 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SLOCAL; }
#line 3991 "y.tab.c" /* yacc.c:1661  */
    break;

  case 274:
#line 983 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SPYTHON; }
#line 3997 "y.tab.c" /* yacc.c:1661  */
    break;

  case 275:
#line 985 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SMEM; }
#line 4003 "y.tab.c" /* yacc.c:1661  */
    break;

  case 276:
#line 987 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = SREDUCE; }
#line 4009 "y.tab.c" /* yacc.c:1661  */
    break;

  case 277:
#line 989 "xi-grammar.y" /* yacc.c:1661  */
    {
        (yyval.intval) = SAGGREGATE;
    }
#line 4017 "y.tab.c" /* yacc.c:1661  */
    break;

  case 278:
#line 993 "xi-grammar.y" /* yacc.c:1661  */
    {
		  ERROR("invalid entry method attribute",
		        (yylsp[0]).first_column, (yylsp[0]).last_column);
		  yyclearin;
		  yyerrok;
		}
#line 4028 "y.tab.c" /* yacc.c:1661  */
    break;

  case 279:
#line 1002 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4034 "y.tab.c" /* yacc.c:1661  */
    break;

  case 280:
#line 1004 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4040 "y.tab.c" /* yacc.c:1661  */
    break;

  case 281:
#line 1006 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4046 "y.tab.c" /* yacc.c:1661  */
    break;

  case 282:
#line 1010 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = ""; }
#line 4052 "y.tab.c" /* yacc.c:1661  */
    break;

  case 283:
#line 1012 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4058 "y.tab.c" /* yacc.c:1661  */
    break;

  case 284:
#line 1014 "xi-grammar.y" /* yacc.c:1661  */
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s, %s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4068 "y.tab.c" /* yacc.c:1661  */
    break;

  case 285:
#line 1022 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = ""; }
#line 4074 "y.tab.c" /* yacc.c:1661  */
    break;

  case 286:
#line 1024 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4080 "y.tab.c" /* yacc.c:1661  */
    break;

  case 287:
#line 1026 "xi-grammar.y" /* yacc.c:1661  */
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s[%s]%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4090 "y.tab.c" /* yacc.c:1661  */
    break;

  case 288:
#line 1032 "xi-grammar.y" /* yacc.c:1661  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s{%s}%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4100 "y.tab.c" /* yacc.c:1661  */
    break;

  case 289:
#line 1038 "xi-grammar.y" /* yacc.c:1661  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s(%s)%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4110 "y.tab.c" /* yacc.c:1661  */
    break;

  case 290:
#line 1044 "xi-grammar.y" /* yacc.c:1661  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"(%s)%s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4120 "y.tab.c" /* yacc.c:1661  */
    break;

  case 291:
#line 1052 "xi-grammar.y" /* yacc.c:1661  */
    {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval));
		}
#line 4129 "y.tab.c" /* yacc.c:1661  */
    break;

  case 292:
#line 1059 "xi-grammar.y" /* yacc.c:1661  */
    {
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			(yyval.intval) = 0;
		}
#line 4139 "y.tab.c" /* yacc.c:1661  */
    break;

  case 293:
#line 1067 "xi-grammar.y" /* yacc.c:1661  */
    {
			in_braces=0;
			(yyval.intval) = 0;
		}
#line 4148 "y.tab.c" /* yacc.c:1661  */
    break;

  case 294:
#line 1074 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[0].type));}
#line 4154 "y.tab.c" /* yacc.c:1661  */
    break;

  case 295:
#line 1076 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval)); (yyval.pname)->setConditional((yyvsp[0].intval)); }
#line 4160 "y.tab.c" /* yacc.c:1661  */
    break;

  case 296:
#line 1078 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[-3].type),(yyvsp[-2].strval),0,(yyvsp[0].val));}
#line 4166 "y.tab.c" /* yacc.c:1661  */
    break;

  case 297:
#line 1080 "xi-grammar.y" /* yacc.c:1661  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
		}
#line 4175 "y.tab.c" /* yacc.c:1661  */
    break;

  case 298:
#line 1085 "xi-grammar.y" /* yacc.c:1661  */
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
#line 1096 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READONLY; }
#line 4195 "y.tab.c" /* yacc.c:1661  */
    break;

  case 300:
#line 1097 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READWRITE; }
#line 4201 "y.tab.c" /* yacc.c:1661  */
    break;

  case 301:
#line 1098 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_WRITEONLY; }
#line 4207 "y.tab.c" /* yacc.c:1661  */
    break;

  case 302:
#line 1101 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.xstrptr) = new XStr((yyvsp[0].strval)); }
#line 4213 "y.tab.c" /* yacc.c:1661  */
    break;

  case 303:
#line 1102 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "->" << (yyvsp[0].strval); }
#line 4219 "y.tab.c" /* yacc.c:1661  */
    break;

  case 304:
#line 1103 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[-2].xstrptr)) << "." << (yyvsp[0].strval); }
#line 4225 "y.tab.c" /* yacc.c:1661  */
    break;

  case 305:
#line 1105 "xi-grammar.y" /* yacc.c:1661  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << *((yyvsp[-1].xstrptr)) << "]";
                  delete (yyvsp[-3].xstrptr);
                  delete (yyvsp[-1].xstrptr);
                }
#line 4236 "y.tab.c" /* yacc.c:1661  */
    break;

  case 306:
#line 1112 "xi-grammar.y" /* yacc.c:1661  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << (yyvsp[-1].strval) << "]";
                  delete (yyvsp[-3].xstrptr);
                }
#line 4246 "y.tab.c" /* yacc.c:1661  */
    break;

  case 307:
#line 1118 "xi-grammar.y" /* yacc.c:1661  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "(" << *((yyvsp[-1].xstrptr)) << ")";
                  delete (yyvsp[-3].xstrptr);
                  delete (yyvsp[-1].xstrptr);
                }
#line 4257 "y.tab.c" /* yacc.c:1661  */
    break;

  case 308:
#line 1127 "xi-grammar.y" /* yacc.c:1661  */
    {
                  in_bracket = 0;
                  (yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName(), (yyvsp[-1].strval));
                }
#line 4266 "y.tab.c" /* yacc.c:1661  */
    break;

  case 309:
#line 1134 "xi-grammar.y" /* yacc.c:1661  */
    {
                  (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[-6].intval));
                }
#line 4276 "y.tab.c" /* yacc.c:1661  */
    break;

  case 310:
#line 1140 "xi-grammar.y" /* yacc.c:1661  */
    {
		  (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType(Parameter::ACCEL_BUFFER_TYPE_READWRITE);
		}
#line 4286 "y.tab.c" /* yacc.c:1661  */
    break;

  case 311:
#line 1146 "xi-grammar.y" /* yacc.c:1661  */
    {
                  (yyval.pname) = (yyvsp[-3].pname);
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[-5].intval));
		}
#line 4296 "y.tab.c" /* yacc.c:1661  */
    break;

  case 312:
#line 1154 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 4302 "y.tab.c" /* yacc.c:1661  */
    break;

  case 313:
#line 1156 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 4308 "y.tab.c" /* yacc.c:1661  */
    break;

  case 314:
#line 1160 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 4314 "y.tab.c" /* yacc.c:1661  */
    break;

  case 315:
#line 1162 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 4320 "y.tab.c" /* yacc.c:1661  */
    break;

  case 316:
#line 1166 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.plist) = (yyvsp[-1].plist); }
#line 4326 "y.tab.c" /* yacc.c:1661  */
    break;

  case 317:
#line 1168 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.plist) = new ParamList(new Parameter(0, new BuiltinType("void"))); }
#line 4332 "y.tab.c" /* yacc.c:1661  */
    break;

  case 318:
#line 1172 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.plist) = (yyvsp[-1].plist); }
#line 4338 "y.tab.c" /* yacc.c:1661  */
    break;

  case 319:
#line 1174 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.plist) = 0; }
#line 4344 "y.tab.c" /* yacc.c:1661  */
    break;

  case 320:
#line 1178 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.val) = 0; }
#line 4350 "y.tab.c" /* yacc.c:1661  */
    break;

  case 321:
#line 1180 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4356 "y.tab.c" /* yacc.c:1661  */
    break;

  case 322:
#line 1184 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sentry) = 0; }
#line 4362 "y.tab.c" /* yacc.c:1661  */
    break;

  case 323:
#line 1186 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[0].sc)); }
#line 4368 "y.tab.c" /* yacc.c:1661  */
    break;

  case 324:
#line 1188 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[-2].slist)); }
#line 4374 "y.tab.c" /* yacc.c:1661  */
    break;

  case 325:
#line 1192 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.slist) = new SListConstruct((yyvsp[0].sc)); }
#line 4380 "y.tab.c" /* yacc.c:1661  */
    break;

  case 326:
#line 1194 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.slist) = new SListConstruct((yyvsp[-1].sc), (yyvsp[0].slist));  }
#line 4386 "y.tab.c" /* yacc.c:1661  */
    break;

  case 327:
#line 1198 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.olist) = new OListConstruct((yyvsp[0].sc)); }
#line 4392 "y.tab.c" /* yacc.c:1661  */
    break;

  case 328:
#line 1200 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.olist) = new OListConstruct((yyvsp[-1].sc), (yyvsp[0].slist)); }
#line 4398 "y.tab.c" /* yacc.c:1661  */
    break;

  case 329:
#line 1204 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.clist) = new CaseListConstruct((yyvsp[0].when)); }
#line 4404 "y.tab.c" /* yacc.c:1661  */
    break;

  case 330:
#line 1206 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.clist) = new CaseListConstruct((yyvsp[-1].when), (yyvsp[0].clist)); }
#line 4410 "y.tab.c" /* yacc.c:1661  */
    break;

  case 331:
#line 1208 "xi-grammar.y" /* yacc.c:1661  */
    {
		  ERROR("case blocks can only contain when clauses",
		        (yylsp[0]).first_column, (yylsp[0]).last_column);
		  (yyval.clist) = 0;
		}
#line 4420 "y.tab.c" /* yacc.c:1661  */
    break;

  case 332:
#line 1216 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4426 "y.tab.c" /* yacc.c:1661  */
    break;

  case 333:
#line 1218 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.strval) = 0; }
#line 4432 "y.tab.c" /* yacc.c:1661  */
    break;

  case 334:
#line 1222 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = new WhenConstruct((yyvsp[-2].entrylist), 0); }
#line 4438 "y.tab.c" /* yacc.c:1661  */
    break;

  case 335:
#line 1224 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = new WhenConstruct((yyvsp[-1].entrylist), (yyvsp[0].sc)); }
#line 4444 "y.tab.c" /* yacc.c:1661  */
    break;

  case 336:
#line 1226 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = new WhenConstruct((yyvsp[-3].entrylist), (yyvsp[-1].slist)); }
#line 4450 "y.tab.c" /* yacc.c:1661  */
    break;

  case 337:
#line 1230 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4456 "y.tab.c" /* yacc.c:1661  */
    break;

  case 338:
#line 1232 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4462 "y.tab.c" /* yacc.c:1661  */
    break;

  case 339:
#line 1234 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4468 "y.tab.c" /* yacc.c:1661  */
    break;

  case 340:
#line 1236 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4474 "y.tab.c" /* yacc.c:1661  */
    break;

  case 341:
#line 1238 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4480 "y.tab.c" /* yacc.c:1661  */
    break;

  case 342:
#line 1240 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4486 "y.tab.c" /* yacc.c:1661  */
    break;

  case 343:
#line 1242 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4492 "y.tab.c" /* yacc.c:1661  */
    break;

  case 344:
#line 1244 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4498 "y.tab.c" /* yacc.c:1661  */
    break;

  case 345:
#line 1246 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4504 "y.tab.c" /* yacc.c:1661  */
    break;

  case 346:
#line 1248 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4510 "y.tab.c" /* yacc.c:1661  */
    break;

  case 347:
#line 1250 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4516 "y.tab.c" /* yacc.c:1661  */
    break;

  case 348:
#line 1252 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.when) = 0; }
#line 4522 "y.tab.c" /* yacc.c:1661  */
    break;

  case 349:
#line 1256 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new SerialConstruct((yyvsp[-2].strval), (yyvsp[-4].strval), (yylsp[-3]).first_line); }
#line 4528 "y.tab.c" /* yacc.c:1661  */
    break;

  case 350:
#line 1258 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new OverlapConstruct((yyvsp[-1].olist)); }
#line 4534 "y.tab.c" /* yacc.c:1661  */
    break;

  case 351:
#line 1260 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = (yyvsp[0].when); }
#line 4540 "y.tab.c" /* yacc.c:1661  */
    break;

  case 352:
#line 1262 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new CaseConstruct((yyvsp[-1].clist)); }
#line 4546 "y.tab.c" /* yacc.c:1661  */
    break;

  case 353:
#line 1264 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new ForConstruct((yyvsp[-8].intexpr), (yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4552 "y.tab.c" /* yacc.c:1661  */
    break;

  case 354:
#line 1266 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new ForConstruct((yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4558 "y.tab.c" /* yacc.c:1661  */
    break;

  case 355:
#line 1268 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-9].strval)), (yyvsp[-6].intexpr),
		             (yyvsp[-4].intexpr), (yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4565 "y.tab.c" /* yacc.c:1661  */
    break;

  case 356:
#line 1271 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-11].strval)), (yyvsp[-8].intexpr),
		             (yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4572 "y.tab.c" /* yacc.c:1661  */
    break;

  case 357:
#line 1274 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new IfConstruct((yyvsp[-3].intexpr), (yyvsp[-1].sc), (yyvsp[0].sc)); }
#line 4578 "y.tab.c" /* yacc.c:1661  */
    break;

  case 358:
#line 1276 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new IfConstruct((yyvsp[-5].intexpr), (yyvsp[-2].slist), (yyvsp[0].sc)); }
#line 4584 "y.tab.c" /* yacc.c:1661  */
    break;

  case 359:
#line 1278 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new WhileConstruct((yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4590 "y.tab.c" /* yacc.c:1661  */
    break;

  case 360:
#line 1280 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new WhileConstruct((yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4596 "y.tab.c" /* yacc.c:1661  */
    break;

  case 361:
#line 1282 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new SerialConstruct((yyvsp[-2].strval), NULL, (yyloc).first_line); }
#line 4602 "y.tab.c" /* yacc.c:1661  */
    break;

  case 362:
#line 1284 "xi-grammar.y" /* yacc.c:1661  */
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
#line 1294 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = 0; }
#line 4620 "y.tab.c" /* yacc.c:1661  */
    break;

  case 364:
#line 1296 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new ElseConstruct((yyvsp[0].sc)); }
#line 4626 "y.tab.c" /* yacc.c:1661  */
    break;

  case 365:
#line 1298 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.sc) = new ElseConstruct((yyvsp[-1].slist)); }
#line 4632 "y.tab.c" /* yacc.c:1661  */
    break;

  case 366:
#line 1302 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.intexpr) = new IntExprConstruct((yyvsp[0].strval)); }
#line 4638 "y.tab.c" /* yacc.c:1661  */
    break;

  case 367:
#line 1306 "xi-grammar.y" /* yacc.c:1661  */
    { in_int_expr = 0; (yyval.intval) = 0; }
#line 4644 "y.tab.c" /* yacc.c:1661  */
    break;

  case 368:
#line 1310 "xi-grammar.y" /* yacc.c:1661  */
    { in_int_expr = 1; (yyval.intval) = 0; }
#line 4650 "y.tab.c" /* yacc.c:1661  */
    break;

  case 369:
#line 1314 "xi-grammar.y" /* yacc.c:1661  */
    {
		  (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[-1].strval), (yyvsp[0].plist), 0, 0, 0, (yyloc).first_line, (yyloc).last_line);
		  firstRdma = true;
		}
#line 4659 "y.tab.c" /* yacc.c:1661  */
    break;

  case 370:
#line 1319 "xi-grammar.y" /* yacc.c:1661  */
    {
		  (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[-4].strval), (yyvsp[0].plist), 0, 0, (yyvsp[-2].strval), (yyloc).first_line, (yyloc).last_line);
		  firstRdma = true;
		}
#line 4668 "y.tab.c" /* yacc.c:1661  */
    break;

  case 371:
#line 1326 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.entrylist) = new EntryList((yyvsp[0].entry)); }
#line 4674 "y.tab.c" /* yacc.c:1661  */
    break;

  case 372:
#line 1328 "xi-grammar.y" /* yacc.c:1661  */
    { (yyval.entrylist) = new EntryList((yyvsp[-2].entry),(yyvsp[0].entrylist)); }
#line 4680 "y.tab.c" /* yacc.c:1661  */
    break;

  case 373:
#line 1332 "xi-grammar.y" /* yacc.c:1661  */
    { in_bracket=1; }
#line 4686 "y.tab.c" /* yacc.c:1661  */
    break;

  case 374:
#line 1335 "xi-grammar.y" /* yacc.c:1661  */
    { in_bracket=0; }
#line 4692 "y.tab.c" /* yacc.c:1661  */
    break;

  case 375:
#line 1339 "xi-grammar.y" /* yacc.c:1661  */
    { if (!macroDefined((yyvsp[0].strval), 1)) in_comment = 1; }
#line 4698 "y.tab.c" /* yacc.c:1661  */
    break;

  case 376:
#line 1343 "xi-grammar.y" /* yacc.c:1661  */
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
#line 1346 "xi-grammar.y" /* yacc.c:1906  */


void yyerror(const char *s)
{
	fprintf(stderr, "[PARSE-ERROR] Unexpected/missing token at line %d. Current token being parsed: '%s'.\n", lineno, yytext);
}
