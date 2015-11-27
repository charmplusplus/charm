/* A Bison parser, made by GNU Bison 3.0.2.  */

/* Bison implementation for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2013 Free Software Foundation, Inc.

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
#define YYBISON_VERSION "3.0.2"

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
void ReservedWord(int token);
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
    CREATEHERE = 286,
    CREATEHOME = 287,
    NOKEEP = 288,
    NOTRACE = 289,
    APPWORK = 290,
    VOID = 291,
    CONST = 292,
    PACKED = 293,
    VARSIZE = 294,
    ENTRY = 295,
    FOR = 296,
    FORALL = 297,
    WHILE = 298,
    WHEN = 299,
    OVERLAP = 300,
    ATOMIC = 301,
    IF = 302,
    ELSE = 303,
    PYTHON = 304,
    LOCAL = 305,
    NAMESPACE = 306,
    USING = 307,
    IDENT = 308,
    NUMBER = 309,
    LITERAL = 310,
    CPROGRAM = 311,
    HASHIF = 312,
    HASHIFDEF = 313,
    INT = 314,
    LONG = 315,
    SHORT = 316,
    CHAR = 317,
    FLOAT = 318,
    DOUBLE = 319,
    UNSIGNED = 320,
    ACCEL = 321,
    READWRITE = 322,
    WRITEONLY = 323,
    ACCELBLOCK = 324,
    MEMCRITICAL = 325,
    DISKPREFETCH = 326,
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
#define CREATEHERE 286
#define CREATEHOME 287
#define NOKEEP 288
#define NOTRACE 289
#define APPWORK 290
#define VOID 291
#define CONST 292
#define PACKED 293
#define VARSIZE 294
#define ENTRY 295
#define FOR 296
#define FORALL 297
#define WHILE 298
#define WHEN 299
#define OVERLAP 300
#define ATOMIC 301
#define IF 302
#define ELSE 303
#define PYTHON 304
#define LOCAL 305
#define NAMESPACE 306
#define USING 307
#define IDENT 308
#define NUMBER 309
#define LITERAL 310
#define CPROGRAM 311
#define HASHIF 312
#define HASHIFDEF 313
#define INT 314
#define LONG 315
#define SHORT 316
#define CHAR 317
#define FLOAT 318
#define DOUBLE 319
#define UNSIGNED 320
#define ACCEL 321
#define READWRITE 322
#define WRITEONLY 323
#define ACCELBLOCK 324
#define MEMCRITICAL 325
#define DISKPREFETCH 326
#define REDUCTIONTARGET 327
#define CASE 328

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE YYSTYPE;
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

#line 372 "y.tab.c" /* yacc.c:358  */

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
#define YYLAST   1444

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  90
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  120
/* YYNRULES -- Number of rules.  */
#define YYNRULES  376
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  737

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
       0,   193,   193,   198,   201,   206,   207,   212,   213,   218,
     220,   221,   222,   224,   225,   226,   228,   229,   230,   231,
     232,   236,   237,   238,   239,   240,   241,   242,   243,   244,
     245,   246,   247,   248,   249,   250,   251,   252,   255,   256,
     257,   258,   259,   260,   261,   262,   263,   264,   265,   267,
     269,   270,   273,   274,   275,   276,   277,   280,   282,   289,
     293,   300,   302,   307,   308,   312,   314,   316,   318,   320,
     332,   334,   336,   338,   344,   346,   348,   350,   352,   354,
     356,   358,   360,   362,   370,   372,   374,   378,   380,   385,
     386,   391,   392,   396,   398,   400,   402,   404,   406,   408,
     410,   412,   414,   416,   418,   420,   422,   424,   428,   429,
     436,   438,   442,   446,   448,   452,   456,   458,   460,   462,
     464,   466,   470,   472,   474,   476,   478,   482,   484,   488,
     490,   494,   498,   503,   504,   508,   512,   517,   518,   523,
     524,   534,   536,   540,   542,   547,   548,   552,   554,   559,
     560,   564,   569,   570,   574,   576,   580,   582,   587,   588,
     592,   593,   596,   600,   602,   606,   608,   613,   614,   618,
     620,   624,   626,   630,   634,   638,   644,   648,   650,   654,
     656,   660,   664,   668,   672,   674,   679,   680,   685,   686,
     688,   690,   699,   701,   703,   707,   709,   713,   717,   719,
     721,   723,   725,   729,   731,   736,   743,   747,   749,   751,
     752,   754,   756,   758,   762,   764,   766,   772,   778,   787,
     789,   791,   797,   805,   807,   810,   814,   818,   820,   825,
     827,   835,   837,   839,   841,   843,   845,   847,   849,   851,
     853,   855,   858,   868,   885,   899,   908,   910,   914,   919,
     920,   922,   929,   931,   935,   937,   939,   941,   943,   945,
     947,   949,   951,   953,   955,   957,   959,   961,   963,   965,
     967,   976,   978,   980,   985,   986,   988,   997,   998,  1000,
    1006,  1012,  1018,  1026,  1033,  1041,  1048,  1050,  1052,  1054,
    1061,  1062,  1063,  1066,  1067,  1068,  1071,  1072,  1073,  1074,
    1081,  1087,  1096,  1103,  1109,  1115,  1123,  1130,  1137,  1144,
    1146,  1150,  1152,  1156,  1158,  1162,  1164,  1168,  1170,  1174,
    1179,  1180,  1184,  1186,  1188,  1192,  1194,  1198,  1200,  1204,
    1206,  1208,  1216,  1219,  1222,  1224,  1226,  1230,  1232,  1234,
    1236,  1238,  1240,  1242,  1244,  1246,  1248,  1250,  1252,  1256,
    1258,  1260,  1262,  1264,  1266,  1268,  1271,  1274,  1276,  1278,
    1280,  1282,  1284,  1295,  1296,  1298,  1302,  1306,  1310,  1314,
    1318,  1324,  1326,  1330,  1333,  1337,  1341
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
  "CREATEHERE", "CREATEHOME", "NOKEEP", "NOTRACE", "APPWORK", "VOID",
  "CONST", "PACKED", "VARSIZE", "ENTRY", "FOR", "FORALL", "WHILE", "WHEN",
  "OVERLAP", "ATOMIC", "IF", "ELSE", "PYTHON", "LOCAL", "NAMESPACE",
  "USING", "IDENT", "NUMBER", "LITERAL", "CPROGRAM", "HASHIF", "HASHIFDEF",
  "INT", "LONG", "SHORT", "CHAR", "FLOAT", "DOUBLE", "UNSIGNED", "ACCEL",
  "READWRITE", "WRITEONLY", "ACCELBLOCK", "MEMCRITICAL", "DISKPREFETCH",
  "REDUCTIONTARGET", "CASE", "';'", "':'", "'{'", "'}'", "','", "'<'",
  "'>'", "'*'", "'('", "')'", "'&'", "'['", "']'", "'='", "'-'", "'.'",
  "$accept", "File", "ModuleEList", "OptExtern", "OptSemiColon", "Name",
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
  "ParamBraceEnd", "Parameter", "AccelBufferType", "OOCBufferType",
  "AccelInstName", "AccelArrayParam", "AccelParameter", "OOCParameter",
  "ParamList", "AccelParamList", "OOCParamList", "EParameters",
  "AccelEParameters", "OOCEParameters", "OptStackSize", "OptSdagCode",
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

#define YYPACT_NINF -634

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-634)))

#define YYTABLE_NINF -328

#define yytable_value_is_error(Yytable_value) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
     190,  1292,  1292,    24,  -634,   190,  -634,  -634,  -634,  -634,
    -634,  -634,  -634,  -634,  -634,  -634,  -634,  -634,  -634,  -634,
    -634,  -634,  -634,  -634,  -634,  -634,  -634,  -634,  -634,  -634,
    -634,  -634,  -634,  -634,  -634,  -634,  -634,  -634,  -634,  -634,
    -634,  -634,  -634,  -634,  -634,  -634,  -634,  -634,  -634,  -634,
    -634,  -634,  -634,  -634,   -29,   -29,  -634,  -634,  -634,   426,
    -634,  -634,  -634,    16,  1292,   225,  1292,  1292,   209,   386,
      32,   933,   426,  -634,  -634,  -634,  1364,    67,    98,  -634,
      81,  -634,  -634,  -634,  -634,   -33,  1313,   137,   137,     3,
      98,   118,   118,   118,   118,   145,   152,  1292,   151,   161,
     426,  -634,  -634,  -634,  -634,  -634,  -634,  -634,  -634,   298,
    -634,  -634,  -634,  -634,   170,  -634,  -634,  -634,  -634,  -634,
    -634,  -634,  -634,  -634,  -634,  -634,  -634,  -634,  -634,  -634,
    1364,  -634,    82,  -634,  -634,  -634,  -634,   264,    99,  -634,
    -634,   174,   194,   198,    -8,  -634,    98,   426,    81,   228,
      95,   -33,   172,   777,  1379,   174,   194,   198,  -634,    71,
      98,  -634,    98,    98,   234,    98,   253,  -634,     4,  1292,
    1292,  1292,  1292,  1079,   247,   249,   255,  1292,  -634,  -634,
    -634,   940,   258,   118,   118,   118,   118,   247,   152,  -634,
    -634,  -634,  -634,  -634,  -634,  -634,   299,  -634,  -634,  -634,
     260,  -634,  -634,  1344,  -634,  -634,  -634,  -634,  -634,  -634,
    1292,   263,   288,   -33,   296,   -33,   271,  -634,   285,   287,
      -5,  -634,   290,  -634,   -37,   100,    35,   286,    84,    98,
    -634,  -634,   292,   295,   297,   307,   307,   307,   307,  -634,
    1292,   293,   305,   301,  1150,  1292,   325,  1292,  -634,  -634,
     320,   306,   309,  1292,   121,  1292,   313,   327,   170,  1292,
    1292,  1292,  1292,  1292,  1292,  -634,  -634,  -634,  -634,   331,
    -634,   332,  -634,   297,  -634,  -634,   334,   335,   339,   338,
     -33,  -634,    98,  1292,  -634,   341,  -634,   -33,   137,  1344,
     137,   137,  1344,   137,  -634,  -634,     4,  -634,    98,   230,
     230,   230,   230,   342,  -634,   325,  -634,   307,   307,  -634,
     255,   429,   356,   243,  -634,   362,   940,  -634,  -634,   307,
     307,   307,   307,   307,   241,  1344,  -634,   370,   -33,   296,
     -33,   -33,  -634,   -37,   372,  -634,   367,  -634,   373,   378,
     381,    98,   385,   383,  -634,   389,  -634,  -634,   503,  -634,
    -634,  -634,  -634,  -634,  -634,   230,   230,  -634,  -634,  1379,
       8,   384,  1379,  -634,  -634,  -634,  -634,  -634,  -634,   230,
     230,   230,   230,   230,  -634,   429,  -634,  1331,  -634,  -634,
    -634,  -634,  -634,  -634,   388,  -634,  -634,  -634,   390,  -634,
     132,   391,  -634,    98,  -634,   686,   428,   394,   400,   503,
    -634,  -634,  -634,  -634,  1292,  -634,  -634,  -634,  -634,  -634,
    -634,  -634,  -634,   398,  -634,  1292,   -33,   402,   393,  1379,
     137,   137,   137,  -634,  -634,   884,  1008,  -634,   170,  -634,
    -634,   396,   408,    10,   399,  1379,  -634,   404,   405,   407,
     410,  -634,  -634,  -634,  -634,  -634,  -634,  -634,  -634,  -634,
    -634,  -634,  -634,  -634,   443,  -634,   414,  -634,   415,  -634,
     419,   430,   425,   370,  1292,  -634,   431,   441,  -634,  -634,
     275,  -634,  -634,  -634,  -634,  -634,  -634,  -634,  -634,  -634,
     483,   484,  -634,   936,   505,   370,  -634,  -634,  -634,  -634,
      81,  -634,  1292,  1292,  -634,  -634,   451,   438,   451,   481,
     460,   482,   451,   462,  -634,   245,   -33,  -634,  -634,  -634,
     519,   370,   370,  -634,   -33,   487,   -33,   120,   466,   523,
     559,  -634,   469,   -33,   226,   476,   374,   172,   468,   505,
     471,   472,  -634,   497,   473,   489,  -634,   -33,   481,   304,
    -634,   496,   485,   -33,   489,   451,   490,   451,   498,   482,
     451,   506,   -33,   510,   226,  -634,   170,  -634,  -634,   529,
    -634,   302,   469,    13,   516,   -33,   451,  -634,   576,   367,
    -634,  -634,   514,  -634,  -634,   172,   734,   -33,   540,   -33,
     559,   469,   -33,   226,   172,  -634,  -634,  -634,  -634,  -634,
    -634,  -634,  -634,  1292,   520,   530,   511,   -33,  -634,  -634,
    -634,   532,   531,   512,  -634,   536,   -33,   245,  -634,   370,
    -634,  -634,   245,   563,   538,   527,   489,   537,   -33,   489,
     547,  -634,   546,  1379,   610,  -634,   172,  1292,    13,  -634,
     -33,   552,   551,  -634,   553,   741,  -634,   -33,   451,   752,
    -634,   172,   759,  -634,  1292,  1292,   -33,   550,  -634,  1292,
     135,  -634,   489,   -33,  -634,   563,   245,  -634,   557,   -33,
     245,  -634,  -634,   245,   563,  -634,    56,   139,   548,  1292,
     562,   -33,   -33,   807,   555,  -634,   560,   -33,   567,   561,
     568,  -634,  -634,  1292,  1221,   570,  1292,  1292,  -634,    80,
    -634,   489,   367,   245,  -634,   -33,  -634,   489,   -33,  -634,
     563,   168,   565,   143,  1292,  -634,   153,  -634,  -634,  -634,
     581,   489,   814,   582,  -634,  -634,  -634,  -634,  -634,  -634,
    -634,   821,   245,  -634,   -33,   245,  -634,   584,   489,   585,
    -634,   828,  -634,   245,  -634,   587,  -634
};

  /* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
     Performed when YYTABLE does not specify something else to do.  Zero
     means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       3,     0,     0,     0,     2,     3,    10,    11,    12,    13,
      14,    15,    16,    17,    18,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,    29,    30,    31,    32,    33,
      34,    35,    36,    37,    38,    39,    40,    41,    42,    43,
      44,    45,    46,    47,    48,    49,    50,     9,    51,    52,
      53,    54,    55,    56,     0,     0,     1,     4,    61,     0,
      59,    60,    83,     6,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    82,    80,    81,     0,     0,     0,    57,
      66,   375,   376,   247,   284,   277,     0,   137,   137,   137,
       0,   145,   145,   145,   145,     0,   139,     0,     0,     0,
       0,    74,   207,   208,    68,    75,    76,    77,    78,     0,
      79,    67,   210,   209,     7,   241,   233,   234,   235,   236,
     237,   239,   240,   238,   231,    72,   232,    73,    64,   107,
       0,    93,    94,    95,    96,   104,   105,     0,    91,   110,
     111,   122,   123,   124,   128,   248,     0,     0,    65,     0,
     278,   277,     0,     0,     0,   116,   117,   118,   119,   130,
       0,   138,     0,     0,     0,     0,   223,   211,     0,     0,
       0,     0,     0,     0,     0,   152,     0,     0,   213,   225,
     212,     0,     0,   145,   145,   145,   145,     0,   139,   198,
     199,   200,   201,   202,     8,    62,   125,   103,   106,    97,
      98,   101,   102,    89,   109,   112,   113,   114,   126,   127,
       0,     0,     0,   277,   274,   277,     0,   285,     0,     0,
     120,   121,     0,   129,   133,   217,   214,     0,   219,     0,
     156,   157,     0,   147,    91,   167,   167,   167,   167,   151,
       0,     0,   154,     0,     0,     0,     0,     0,   143,   144,
       0,   141,   165,     0,   119,     0,   195,     0,     7,     0,
       0,     0,     0,     0,     0,    99,   100,    85,    86,    87,
      90,     0,    84,    91,    71,    58,     0,   275,     0,     0,
     277,   246,     0,     0,   373,   133,   135,   277,   137,     0,
     137,   137,     0,   137,   224,   146,     0,   108,     0,     0,
       0,     0,     0,     0,   176,     0,   153,   167,   167,   140,
       0,   158,   186,     0,   193,   188,     0,   197,    70,   167,
     167,   167,   167,   167,     0,     0,    92,     0,   277,   274,
     277,   277,   282,   133,     0,   134,     0,   131,     0,     0,
       0,     0,     0,     0,   148,   169,   168,   203,     0,   171,
     172,   173,   174,   175,   155,     0,     0,   142,   159,     0,
     158,     0,     0,   192,   189,   190,   191,   194,   196,     0,
       0,     0,     0,     0,   184,   158,    88,     0,    69,   280,
     276,   281,   279,   136,     0,   374,   132,   218,     0,   215,
       0,     0,   220,     0,   230,     0,     0,     0,     0,     0,
     226,   227,   177,   178,     0,   164,   166,   187,   179,   180,
     181,   182,   183,     0,   316,   286,   277,   309,     0,     0,
     137,   137,   137,   170,   251,     0,     0,   228,     7,   229,
     206,   160,     0,   158,     0,     0,   315,     0,     0,     0,
       0,   270,   254,   255,   256,   257,   263,   264,   265,   258,
     259,   260,   261,   262,   149,   266,     0,   268,     0,   269,
       0,   252,    57,     0,     0,   204,     0,     0,   185,   283,
       0,   287,   289,   310,   115,   216,   222,   221,   150,   267,
       0,     0,   250,     0,     0,     0,   161,   162,   272,   271,
     273,   288,     0,     0,   253,   362,     0,     0,     0,     0,
       0,   333,     0,     0,   322,     0,   277,   243,   351,   323,
     320,     0,     0,   368,   277,     0,   277,     0,   371,     0,
       0,   332,     0,   277,     0,     0,     0,     0,     0,     0,
       0,     0,   366,     0,     0,     0,   369,   277,     0,     0,
     335,     0,     0,   277,     0,     0,     0,     0,     0,   333,
       0,     0,   277,     0,   329,   331,     7,   326,   361,     0,
     242,     0,     0,     0,     0,   277,     0,   367,     0,     0,
     372,   334,     0,   350,   328,     0,     0,   277,     0,   277,
       0,     0,   277,     0,     0,   352,   330,   324,   321,   290,
     291,   292,   318,     0,     0,   311,     0,   277,   293,   294,
     295,     0,   313,     0,   245,     0,   277,     0,   359,     0,
     336,   349,     0,   363,     0,     0,     0,     0,   277,     0,
       0,   348,     0,     0,     0,   317,     0,     0,     0,   319,
     277,     0,     0,   370,     0,     0,   357,   277,     0,     0,
     338,     0,     0,   339,     0,     0,   277,     0,   312,     0,
     308,   314,     0,   277,   360,   363,     0,   364,     0,   277,
       0,   346,   337,     0,   363,   296,     0,     0,     0,     0,
       0,   277,   277,     0,     0,   358,     0,   277,     0,     0,
       0,   344,   304,     0,     0,     0,     0,     0,   302,     0,
     244,     0,     0,     0,   354,   277,   365,     0,   277,   347,
     363,     0,     0,     0,     0,   298,     0,   305,   307,   306,
       0,     0,     0,     0,   345,   301,   300,   299,   297,   303,
     353,     0,     0,   341,   277,     0,   355,     0,     0,     0,
     340,     0,   356,     0,   342,     0,   343
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -634,  -634,   654,  -634,  -249,    -1,   -60,   596,   611,   -56,
    -634,  -634,  -634,  -155,  -634,  -183,  -634,    19,   -77,   -70,
     -69,   -68,  -169,   513,   549,  -634,   -82,  -634,  -634,  -259,
    -634,  -634,   -78,   488,   371,  -634,     0,   392,  -634,  -634,
     509,   375,  -634,   259,  -634,  -634,  -256,  -634,  -195,   303,
    -634,  -634,  -634,  -128,  -634,  -634,  -634,  -634,  -634,  -634,
    -634,   387,  -634,   421,   635,  -634,   284,   344,   671,  -634,
    -634,   515,  -634,  -634,  -634,  -634,   345,  -634,   328,  -634,
     278,  -634,  -634,   433,   -83,   140,   -65,  -477,  -634,  -634,
    -634,  -629,  -634,  -634,  -634,  -304,   141,   136,  -433,  -634,
    -634,  -634,   237,  -506,   187,  -519,   219,  -467,  -634,  -447,
    -633,  -495,  -530,  -427,  -634,   231,  -502,  -552,  -634,  -634
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     3,     4,    69,   195,   234,   138,     5,    60,    70,
      71,    72,   269,   270,   271,   204,   139,   235,   140,   155,
     156,   157,   158,   159,   144,   145,   272,   336,   285,   286,
     102,   103,   162,   177,   250,   251,   169,   232,   479,   242,
     174,   243,   233,   359,   467,   360,   361,   104,   299,   346,
     105,   106,   107,   175,   108,   189,   190,   191,   192,   193,
     363,   314,   256,   257,   396,   110,   349,   397,   398,   112,
     113,   167,   180,   399,   400,   127,   401,    73,   146,   426,
     460,   461,   491,   278,   532,   416,   506,   218,   417,   594,
     601,   666,   647,   595,   602,   418,   596,   603,   378,   562,
     564,   529,   507,   525,   541,   553,   522,   508,   555,   526,
     636,   533,   568,   514,   518,   519,   287,   386,    74,    75
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[] =
{
      54,    55,   152,    85,   160,    80,   141,   142,   143,   318,
     163,   165,   254,   166,   576,   537,   128,   609,   148,   598,
     557,   535,   675,   150,    56,   358,   335,   358,   544,   208,
     484,   681,   221,   572,   230,   586,   574,   509,   149,   161,
     689,   300,   301,   302,   182,    58,   245,    59,   284,   151,
     558,   297,   510,   231,   701,   703,    76,   554,   706,   263,
     141,   142,   143,    77,   620,    81,    82,   714,   216,   210,
     605,   516,   540,   542,   383,   523,   209,   222,   530,   531,
     599,   600,   509,   219,   536,  -163,   639,   554,   164,   642,
     327,   211,   170,   171,   172,   469,   178,   470,   611,   255,
     224,   632,   225,   226,   405,   228,   634,   621,   221,   114,
     149,   631,   355,   356,   289,   437,   554,   290,   577,   413,
     579,   608,   673,   582,   369,   370,   371,   372,   373,   613,
     276,   473,   279,   542,   339,   652,   682,   342,   683,   606,
     709,   684,   197,   147,   685,   686,   198,   254,   672,   649,
     676,    79,   166,   222,   679,   223,   149,   680,   674,   149,
     707,   708,   683,   292,   662,   684,   293,   712,   685,   686,
     376,   213,   241,   161,   149,   149,   633,   214,   203,   465,
     215,   721,   288,   259,   260,   261,   262,   710,   657,   236,
     237,   238,   661,     1,     2,   664,   252,   332,   731,  -188,
     711,  -188,   377,   168,   337,   284,   179,   149,   313,   273,
     338,   659,   340,   341,   421,   343,   727,   513,   687,   729,
     284,   345,   333,   671,   469,   683,   694,   735,   684,   717,
     173,   685,   686,   719,   255,   683,   366,   176,   684,   303,
     181,   685,   686,   241,   194,   379,   495,   381,   382,   217,
     683,   715,   312,   684,   315,   205,   685,   686,   319,   320,
     321,   322,   323,   324,   307,   723,   308,   545,   546,   547,
     499,   548,   549,   550,   726,   206,    78,   404,    79,   207,
     407,   390,   334,    83,   734,    84,   496,   497,   498,   499,
     500,   501,   502,   248,   249,   415,    79,   364,   365,   551,
     227,  -284,    84,   212,   347,   495,   348,   587,   589,   183,
     184,   185,   186,   187,   188,   374,   345,   375,   503,   265,
     266,    84,  -284,   199,   200,   201,   202,  -284,    79,   488,
     489,   229,   244,   434,   246,   258,   208,   415,   129,   154,
     274,   275,   438,   439,   440,   496,   497,   498,   499,   500,
     501,   502,   277,   415,   280,    79,   141,   142,   143,   281,
    -284,   131,   132,   133,   134,   135,   136,   137,   282,   590,
     591,   283,   291,   296,   239,   495,   203,   503,   295,   304,
      84,   571,   298,   305,   310,   311,  -284,   306,   592,     1,
       2,   316,    86,    87,    88,    89,    90,    91,    92,    93,
      94,    95,    96,   431,    97,    98,   309,   317,    99,   325,
     490,   328,   326,   329,   433,   496,   497,   498,   499,   500,
     501,   502,   330,   527,   331,   463,   284,    62,   353,    -5,
      -5,    63,    -5,    -5,    -5,    -5,    -5,    -5,    -5,    -5,
      -5,    -5,    -5,   362,    -5,    -5,   358,   503,    -5,   313,
      84,  -325,   377,   385,   569,   384,   387,   543,   388,   552,
     575,   406,   100,   485,   389,   391,   392,   393,   395,   584,
     419,   428,   420,   422,   429,   432,   436,    64,    65,   593,
     435,   466,   468,    66,    67,   472,   495,   474,   475,   552,
     476,   511,   512,   477,   614,    68,   616,   597,   478,   619,
     480,   481,    -5,   -63,   394,   482,   495,    -9,   483,    86,
      87,    88,    89,    90,   626,   487,   618,   486,   552,   492,
     493,    97,    98,   515,   495,    99,   496,   497,   498,   499,
     500,   501,   502,   513,   517,   641,   520,   521,   524,   528,
     534,   645,   593,   395,   538,    84,   496,   497,   498,   499,
     500,   501,   502,   556,   658,   559,   561,   563,   503,   566,
     495,    84,  -327,   668,   496,   497,   498,   499,   500,   501,
     502,   565,   567,   573,   580,   578,   678,   495,   503,   504,
    -205,   505,   583,   588,   350,   351,   352,   585,   691,   692,
     604,   610,   622,   615,   697,   623,   503,   625,   629,   539,
     496,   497,   498,   499,   500,   501,   502,   627,   624,   628,
     630,   635,   637,   638,   640,   713,   589,   496,   497,   498,
     499,   500,   501,   502,   643,   644,   650,   653,   654,   669,
     655,   677,   503,   695,   688,    84,   690,   696,   699,   402,
     403,   728,   698,   665,   667,   700,   129,   154,   670,   503,
     704,   716,   607,   408,   409,   410,   411,   412,   720,    57,
     724,   730,   732,    79,   736,   101,    61,   220,   665,   131,
     132,   133,   134,   135,   136,   137,   264,   590,   591,   196,
     354,   357,   665,   665,   247,   705,   665,   424,   344,  -249,
    -249,  -249,   471,  -249,  -249,  -249,   423,  -249,  -249,  -249,
    -249,  -249,   367,   718,   109,  -249,  -249,  -249,  -249,  -249,
    -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,
    -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,  -249,
    -249,  -249,  -249,  -249,  -249,   495,  -249,   368,  -249,  -249,
     111,   427,   495,   430,   294,  -249,  -249,  -249,  -249,  -249,
    -249,  -249,  -249,   495,   464,  -249,  -249,  -249,  -249,  -249,
     495,   494,   380,   646,   651,   648,   560,   617,   581,   570,
       0,   425,     0,     0,     0,   496,   497,   498,   499,   500,
     501,   502,   496,   497,   498,   499,   500,   501,   502,     0,
       0,     0,     0,   496,   497,   498,   499,   500,   501,   502,
     496,   497,   498,   499,   500,   501,   502,   503,   495,     0,
     612,     0,     0,   129,   503,   495,     0,   656,     0,     0,
       0,     0,   495,     0,     0,   503,     0,     0,   660,   495,
      79,     0,   503,     0,     0,   663,   131,   132,   133,   134,
     135,   136,   137,     0,     0,     0,     0,     0,   496,   497,
     498,   499,   500,   501,   502,   496,   497,   498,   499,   500,
     501,   502,   496,   497,   498,   499,   500,   501,   502,   496,
     497,   498,   499,   500,   501,   502,     0,     0,     0,     0,
     503,     0,     0,   693,     0,   441,     0,   503,     0,     0,
     722,     0,     0,     0,   503,     0,     0,   725,     0,     0,
       0,   503,     0,     0,   733,   442,     0,   443,   444,   445,
     446,   447,   448,     0,     0,   449,   450,   451,   452,   453,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   454,   455,     0,     0,   441,     0,   115,
     116,   117,   118,     0,   119,   120,   121,   122,   123,     0,
     456,     0,     0,     0,   457,   458,   459,   442,   253,   443,
     444,   445,   446,   447,   448,     0,     0,   449,   450,   451,
     452,   453,     0,   124,     0,     0,   129,   154,     0,     0,
       0,     0,     0,     0,     0,   454,   455,     0,     0,     0,
       0,     0,     0,    79,     0,     0,     0,     0,     0,   131,
     132,   133,   134,   135,   136,   137,   457,   125,   459,     0,
     126,     6,     7,     8,     0,     9,    10,    11,     0,    12,
      13,    14,    15,    16,     0,     0,     0,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    31,    32,    33,   129,   130,    34,    35,    36,    37,
      38,    39,    40,    41,    42,    43,    44,     0,    45,     0,
      46,   462,     0,     0,     0,     0,     0,   131,   132,   133,
     134,   135,   136,   137,    48,     0,     0,    49,    50,    51,
      52,    53,     6,     7,     8,     0,     9,    10,    11,     0,
      12,    13,    14,    15,    16,     0,     0,     0,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,     0,     0,    34,    35,    36,
      37,    38,    39,    40,    41,    42,    43,    44,   239,    45,
       0,    46,    47,   240,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    48,     0,     0,    49,    50,
      51,    52,    53,     6,     7,     8,     0,     9,    10,    11,
       0,    12,    13,    14,    15,    16,     0,     0,     0,    17,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      28,    29,    30,    31,    32,    33,     0,     0,    34,    35,
      36,    37,    38,    39,    40,    41,    42,    43,    44,     0,
      45,     0,    46,    47,   240,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    48,     0,     0,    49,
      50,    51,    52,    53,     6,     7,     8,     0,     9,    10,
      11,     0,    12,    13,    14,    15,    16,     0,     0,     0,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,     0,     0,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
       0,    45,     0,    46,    47,   702,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    48,     0,     0,
      49,    50,    51,    52,    53,     6,     7,     8,     0,     9,
      10,    11,     0,    12,    13,    14,    15,    16,     0,     0,
       0,    17,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    31,    32,    33,     0,   153,
      34,    35,    36,    37,    38,    39,    40,    41,    42,    43,
      44,     0,    45,     0,    46,    47,     0,     0,     0,   129,
     154,     0,     0,     0,     0,     0,     0,     0,    48,     0,
       0,    49,    50,    51,    52,    53,    79,   129,   154,     0,
       0,     0,   131,   132,   133,   134,   135,   136,   137,     0,
     129,   154,     0,     0,    79,     0,     0,     0,     0,     0,
     131,   132,   133,   134,   135,   136,   137,    79,   267,   268,
     129,   130,     0,   131,   132,   133,   134,   135,   136,   137,
       0,     0,     0,     0,   414,   129,   154,    79,     0,     0,
       0,     0,     0,   131,   132,   133,   134,   135,   136,   137,
       0,     0,    79,     0,     0,     0,     0,     0,   131,   132,
     133,   134,   135,   136,   137
};

static const yytype_int16 yycheck[] =
{
       1,     2,    85,    68,    86,    65,    76,    76,    76,   258,
      88,    89,   181,    90,   544,   517,    72,   569,    78,     6,
     526,   516,   655,    56,     0,    17,   285,    17,   523,    37,
     463,   664,    37,   539,    30,   554,   542,   484,    75,    36,
     669,   236,   237,   238,   100,    74,   174,    76,    85,    82,
     527,   234,   485,    49,   683,   684,    40,   524,   687,   187,
     130,   130,   130,    64,   583,    66,    67,   700,   151,   146,
     565,   498,   519,   520,   333,   502,    84,    82,   511,   512,
      67,    68,   529,   153,   517,    77,   616,   554,    85,   619,
     273,   147,    92,    93,    94,    85,    97,    87,   575,   181,
     160,   607,   162,   163,   360,   165,   612,   584,    37,    77,
      75,   606,   307,   308,    79,   419,   583,    82,   545,   375,
     547,   568,   652,   550,   319,   320,   321,   322,   323,   576,
     213,   435,   215,   580,   289,   630,    80,   292,    82,   566,
     692,    85,    60,    76,    88,    89,    64,   316,   650,   626,
     656,    53,   229,    82,   660,    84,    75,   663,   653,    75,
      80,   691,    82,    79,   641,    85,    82,   697,    88,    89,
     325,    76,   173,    36,    75,    75,   609,    82,    79,   428,
      85,   711,    82,   183,   184,   185,   186,   693,   635,   170,
     171,   172,   639,     3,     4,   642,   177,   280,   728,    78,
     695,    80,    82,    85,   287,    85,    55,    75,    87,   210,
     288,   638,   290,   291,    82,   293,   722,    82,    79,   725,
      85,   298,   282,   650,    85,    82,   673,   733,    85,    86,
      85,    88,    89,    80,   316,    82,   313,    85,    85,   240,
      79,    88,    89,   244,    74,   328,     1,   330,   331,    77,
      82,    83,   253,    85,   255,    81,    88,    89,   259,   260,
     261,   262,   263,   264,   245,   712,   247,    41,    42,    43,
      44,    45,    46,    47,   721,    81,    51,   359,    53,    81,
     362,   341,   283,    74,   731,    76,    41,    42,    43,    44,
      45,    46,    47,    38,    39,   377,    53,    54,    55,    73,
      66,    56,    76,    75,    74,     1,    76,   556,     6,    11,
      12,    13,    14,    15,    16,    74,   393,    76,    73,    59,
      60,    76,    77,    59,    60,    61,    62,    82,    53,    54,
      55,    78,    85,   416,    85,    77,    37,   419,    36,    37,
      77,    53,   420,   421,   422,    41,    42,    43,    44,    45,
      46,    47,    56,   435,    83,    53,   426,   426,   426,    74,
      56,    59,    60,    61,    62,    63,    64,    65,    81,    67,
      68,    81,    86,    78,    49,     1,    79,    73,    86,    86,
      76,    77,    75,    78,    78,    76,    82,    86,    86,     3,
       4,    78,     6,     7,     8,     9,    10,    11,    12,    13,
      14,    15,    16,   404,    18,    19,    86,    80,    22,    78,
     470,    77,    80,    78,   415,    41,    42,    43,    44,    45,
      46,    47,    83,   506,    86,   426,    85,     1,    86,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      14,    15,    16,    87,    18,    19,    17,    73,    22,    87,
      76,    77,    82,    86,   537,    83,    83,   522,    80,   524,
     543,    77,    76,   464,    83,    80,    83,    78,    40,   552,
      82,    77,    82,    82,    74,    77,    83,    51,    52,   561,
      78,    85,    74,    57,    58,    86,     1,    83,    83,   554,
      83,   492,   493,    83,   577,    69,   579,   562,    55,   582,
      86,    86,    76,    77,     1,    86,     1,    82,    78,     6,
       7,     8,     9,    10,   597,    74,   581,    86,   583,    36,
      36,    18,    19,    85,     1,    22,    41,    42,    43,    44,
      45,    46,    47,    82,    53,   618,    76,    55,    76,    20,
      53,   623,   624,    40,    78,    76,    41,    42,    43,    44,
      45,    46,    47,    77,   637,    87,    85,    85,    73,    86,
       1,    76,    77,   646,    41,    42,    43,    44,    45,    46,
      47,    74,    83,    77,    76,    85,   659,     1,    73,    74,
      77,    76,    76,    54,   300,   301,   302,    77,   671,   672,
      74,    77,   593,    53,   677,    75,    73,    86,    86,    76,
      41,    42,    43,    44,    45,    46,    47,    75,    78,    78,
      74,    48,    74,    86,    77,   698,     6,    41,    42,    43,
      44,    45,    46,    47,    77,    79,   627,    75,    77,    79,
      77,    74,    73,    78,    86,    76,    74,    77,    77,   355,
     356,   724,    75,   644,   645,    77,    36,    37,   649,    73,
      80,    86,    76,   369,   370,   371,   372,   373,    77,     5,
      78,    77,    77,    53,    77,    69,    55,   154,   669,    59,
      60,    61,    62,    63,    64,    65,   188,    67,    68,   130,
     305,   310,   683,   684,   175,   686,   687,     1,   296,     3,
       4,     5,   433,     7,     8,     9,   393,    11,    12,    13,
      14,    15,   315,   704,    69,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,    29,    30,    31,    32,    33,
      34,    35,    36,    37,    38,    39,    40,    41,    42,    43,
      44,    45,    46,    47,    48,     1,    50,   316,    52,    53,
      69,   396,     1,   399,   229,    59,    60,    61,    62,    63,
      64,    65,    66,     1,   426,    69,    70,    71,    72,    73,
       1,   483,   329,   623,   628,   624,   529,   580,   549,   538,
      -1,    85,    -1,    -1,    -1,    41,    42,    43,    44,    45,
      46,    47,    41,    42,    43,    44,    45,    46,    47,    -1,
      -1,    -1,    -1,    41,    42,    43,    44,    45,    46,    47,
      41,    42,    43,    44,    45,    46,    47,    73,     1,    -1,
      76,    -1,    -1,    36,    73,     1,    -1,    76,    -1,    -1,
      -1,    -1,     1,    -1,    -1,    73,    -1,    -1,    76,     1,
      53,    -1,    73,    -1,    -1,    76,    59,    60,    61,    62,
      63,    64,    65,    -1,    -1,    -1,    -1,    -1,    41,    42,
      43,    44,    45,    46,    47,    41,    42,    43,    44,    45,
      46,    47,    41,    42,    43,    44,    45,    46,    47,    41,
      42,    43,    44,    45,    46,    47,    -1,    -1,    -1,    -1,
      73,    -1,    -1,    76,    -1,     1,    -1,    73,    -1,    -1,
      76,    -1,    -1,    -1,    73,    -1,    -1,    76,    -1,    -1,
      -1,    73,    -1,    -1,    76,    21,    -1,    23,    24,    25,
      26,    27,    28,    -1,    -1,    31,    32,    33,    34,    35,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    49,    50,    -1,    -1,     1,    -1,     6,
       7,     8,     9,    -1,    11,    12,    13,    14,    15,    -1,
      66,    -1,    -1,    -1,    70,    71,    72,    21,    18,    23,
      24,    25,    26,    27,    28,    -1,    -1,    31,    32,    33,
      34,    35,    -1,    40,    -1,    -1,    36,    37,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    49,    50,    -1,    -1,    -1,
      -1,    -1,    -1,    53,    -1,    -1,    -1,    -1,    -1,    59,
      60,    61,    62,    63,    64,    65,    70,    74,    72,    -1,
      77,     3,     4,     5,    -1,     7,     8,     9,    -1,    11,
      12,    13,    14,    15,    -1,    -1,    -1,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    34,    35,    36,    37,    38,    39,    40,    41,
      42,    43,    44,    45,    46,    47,    48,    -1,    50,    -1,
      52,    53,    -1,    -1,    -1,    -1,    -1,    59,    60,    61,
      62,    63,    64,    65,    66,    -1,    -1,    69,    70,    71,
      72,    73,     3,     4,     5,    -1,     7,     8,     9,    -1,
      11,    12,    13,    14,    15,    -1,    -1,    -1,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    34,    35,    -1,    -1,    38,    39,    40,
      41,    42,    43,    44,    45,    46,    47,    48,    49,    50,
      -1,    52,    53,    54,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    66,    -1,    -1,    69,    70,
      71,    72,    73,     3,     4,     5,    -1,     7,     8,     9,
      -1,    11,    12,    13,    14,    15,    -1,    -1,    -1,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    31,    32,    33,    34,    35,    -1,    -1,    38,    39,
      40,    41,    42,    43,    44,    45,    46,    47,    48,    -1,
      50,    -1,    52,    53,    54,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    66,    -1,    -1,    69,
      70,    71,    72,    73,     3,     4,     5,    -1,     7,     8,
       9,    -1,    11,    12,    13,    14,    15,    -1,    -1,    -1,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    34,    35,    -1,    -1,    38,
      39,    40,    41,    42,    43,    44,    45,    46,    47,    48,
      -1,    50,    -1,    52,    53,    54,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    66,    -1,    -1,
      69,    70,    71,    72,    73,     3,     4,     5,    -1,     7,
       8,     9,    -1,    11,    12,    13,    14,    15,    -1,    -1,
      -1,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      28,    29,    30,    31,    32,    33,    34,    35,    -1,    16,
      38,    39,    40,    41,    42,    43,    44,    45,    46,    47,
      48,    -1,    50,    -1,    52,    53,    -1,    -1,    -1,    36,
      37,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    66,    -1,
      -1,    69,    70,    71,    72,    73,    53,    36,    37,    -1,
      -1,    -1,    59,    60,    61,    62,    63,    64,    65,    -1,
      36,    37,    -1,    -1,    53,    -1,    -1,    -1,    -1,    -1,
      59,    60,    61,    62,    63,    64,    65,    53,    54,    55,
      36,    37,    -1,    59,    60,    61,    62,    63,    64,    65,
      -1,    -1,    -1,    -1,    83,    36,    37,    53,    -1,    -1,
      -1,    -1,    -1,    59,    60,    61,    62,    63,    64,    65,
      -1,    -1,    53,    -1,    -1,    -1,    -1,    -1,    59,    60,
      61,    62,    63,    64,    65
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     3,     4,    91,    92,    97,     3,     4,     5,     7,
       8,     9,    11,    12,    13,    14,    15,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    34,    35,    38,    39,    40,    41,    42,    43,
      44,    45,    46,    47,    48,    50,    52,    53,    66,    69,
      70,    71,    72,    73,    95,    95,     0,    92,    74,    76,
      98,    98,     1,     5,    51,    52,    57,    58,    69,    93,
      99,   100,   101,   167,   208,   209,    40,    95,    51,    53,
      96,    95,    95,    74,    76,   176,     6,     7,     8,     9,
      10,    11,    12,    13,    14,    15,    16,    18,    19,    22,
      76,    97,   120,   121,   137,   140,   141,   142,   144,   154,
     155,   158,   159,   160,    77,     6,     7,     8,     9,    11,
      12,    13,    14,    15,    40,    74,    77,   165,    99,    36,
      37,    59,    60,    61,    62,    63,    64,    65,    96,   106,
     108,   109,   110,   111,   114,   115,   168,    76,    96,    75,
      56,    82,   174,    16,    37,   109,   110,   111,   112,   113,
     116,    36,   122,   122,    85,   122,   108,   161,    85,   126,
     126,   126,   126,    85,   130,   143,    85,   123,    95,    55,
     162,    79,    99,    11,    12,    13,    14,    15,    16,   145,
     146,   147,   148,   149,    74,    94,   114,    60,    64,    59,
      60,    61,    62,    79,   105,    81,    81,    81,    37,    84,
     108,    99,    75,    76,    82,    85,   174,    77,   177,   109,
     113,    37,    82,    84,    96,    96,    96,    66,    96,    78,
      30,    49,   127,   132,    95,   107,   107,   107,   107,    49,
      54,    95,   129,   131,    85,   143,    85,   130,    38,    39,
     124,   125,   107,    18,   112,   116,   152,   153,    77,   126,
     126,   126,   126,   143,   123,    59,    60,    54,    55,   102,
     103,   104,   116,    95,    77,    53,   174,    56,   173,   174,
      83,    74,    81,    81,    85,   118,   119,   206,    82,    79,
      82,    86,    79,    82,   161,    86,    78,   105,    75,   138,
     138,   138,   138,    95,    86,    78,    86,   107,   107,    86,
      78,    76,    95,    87,   151,    95,    78,    80,    94,    95,
      95,    95,    95,    95,    95,    78,    80,   105,    77,    78,
      83,    86,   174,    96,    95,   119,   117,   174,   122,   103,
     122,   122,   103,   122,   127,   108,   139,    74,    76,   156,
     156,   156,   156,    86,   131,   138,   138,   124,    17,   133,
     135,   136,    87,   150,    54,    55,   108,   151,   153,   138,
     138,   138,   138,   138,    74,    76,   103,    82,   188,   174,
     173,   174,   174,   119,    83,    86,   207,    83,    80,    83,
      96,    80,    83,    78,     1,    40,   154,   157,   158,   163,
     164,   166,   156,   156,   116,   136,    77,   116,   156,   156,
     156,   156,   156,   136,    83,   116,   175,   178,   185,    82,
      82,    82,    82,   139,     1,    85,   169,   166,    77,    74,
     157,    95,    77,    95,   174,    78,    83,   185,   122,   122,
     122,     1,    21,    23,    24,    25,    26,    27,    28,    31,
      32,    33,    34,    35,    49,    50,    66,    70,    71,    72,
     170,   171,    53,    95,   168,    94,    85,   134,    74,    85,
      87,   133,    86,   185,    83,    83,    83,    83,    55,   128,
      86,    86,    86,    78,   188,    95,    86,    74,    54,    55,
      96,   172,    36,    36,   170,     1,    41,    42,    43,    44,
      45,    46,    47,    73,    74,    76,   176,   192,   197,   199,
     188,    95,    95,    82,   203,    85,   203,    53,   204,   205,
      76,    55,   196,   203,    76,   193,   199,   174,    20,   191,
     188,   188,   174,   201,    53,   201,   188,   206,    78,    76,
     199,   194,   199,   176,   201,    41,    42,    43,    45,    46,
      47,    73,   176,   195,   197,   198,    77,   193,   177,    87,
     192,    85,   189,    85,   190,    74,    86,    83,   202,   174,
     205,    77,   193,    77,   193,   174,   202,   203,    85,   203,
      76,   196,   203,    76,   174,    77,   195,    94,    54,     6,
      67,    68,    86,   116,   179,   183,   186,   176,     6,    67,
      68,   180,   184,   187,    74,   201,   203,    76,   199,   207,
      77,   177,    76,   199,   174,    53,   174,   194,   176,   174,
     195,   177,    95,    75,    78,    86,   174,    75,    78,    86,
      74,   201,   193,   188,   193,    48,   200,    74,    86,   202,
      77,   174,   202,    77,    79,   116,   175,   182,   186,   177,
      95,   187,   201,    75,    77,    77,    76,   199,   174,   203,
      76,   199,   177,    76,   199,    95,   181,    95,   174,    79,
      95,   203,   206,   202,   201,   200,   193,    74,   174,   193,
     193,   200,    80,    82,    85,    88,    89,    79,    86,   181,
      74,   174,   174,    76,   199,    78,    77,   174,    75,    77,
      77,   181,    54,   181,    80,    95,   181,    80,   202,   207,
     193,   201,   202,   174,   200,    83,    86,    86,    95,    80,
      77,   202,    76,   199,    78,    76,   199,   193,   174,   193,
      77,   202,    77,    76,   199,   193,    77
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    90,    91,    92,    92,    93,    93,    94,    94,    95,
      95,    95,    95,    95,    95,    95,    95,    95,    95,    95,
      95,    95,    95,    95,    95,    95,    95,    95,    95,    95,
      95,    95,    95,    95,    95,    95,    95,    95,    95,    95,
      95,    95,    95,    95,    95,    95,    95,    95,    95,    95,
      95,    95,    95,    95,    95,    95,    95,    96,    96,    97,
      97,    98,    98,    99,    99,   100,   100,   100,   100,   100,
     101,   101,   101,   101,   101,   101,   101,   101,   101,   101,
     101,   101,   101,   101,   102,   102,   102,   103,   103,   104,
     104,   105,   105,   106,   106,   106,   106,   106,   106,   106,
     106,   106,   106,   106,   106,   106,   106,   106,   107,   108,
     109,   109,   110,   111,   111,   112,   113,   113,   113,   113,
     113,   113,   114,   114,   114,   114,   114,   115,   115,   116,
     116,   117,   118,   119,   119,   120,   121,   122,   122,   123,
     123,   124,   124,   125,   125,   126,   126,   127,   127,   128,
     128,   129,   130,   130,   131,   131,   132,   132,   133,   133,
     134,   134,   135,   136,   136,   137,   137,   138,   138,   139,
     139,   140,   140,   141,   142,   143,   143,   144,   144,   145,
     145,   146,   147,   148,   149,   149,   150,   150,   151,   151,
     151,   151,   152,   152,   152,   153,   153,   154,   155,   155,
     155,   155,   155,   156,   156,   157,   157,   158,   158,   158,
     158,   158,   158,   158,   159,   159,   159,   159,   159,   160,
     160,   160,   160,   161,   161,   162,   163,   164,   164,   164,
     164,   165,   165,   165,   165,   165,   165,   165,   165,   165,
     165,   165,   166,   166,   166,   166,   167,   167,   168,   169,
     169,   169,   170,   170,   171,   171,   171,   171,   171,   171,
     171,   171,   171,   171,   171,   171,   171,   171,   171,   171,
     171,   172,   172,   172,   173,   173,   173,   174,   174,   174,
     174,   174,   174,   175,   176,   177,   178,   178,   178,   178,
     179,   179,   179,   180,   180,   180,   181,   181,   181,   181,
     181,   181,   182,   183,   183,   183,   184,   184,   184,   185,
     185,   186,   186,   187,   187,   188,   188,   189,   189,   190,
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
       0,     2,     1,     0,     2,     0,     1,     0,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     4,     3,
       3,     1,     4,     0,     2,     3,     2,     2,     2,     7,
       5,     5,     2,     2,     2,     2,     2,     2,     2,     2,
       1,     1,     1,     1,     1,     1,     1,     1,     3,     0,
       1,     0,     3,     1,     1,     1,     1,     2,     2,     3,
       3,     2,     2,     2,     1,     1,     2,     1,     2,     2,
       1,     1,     2,     2,     2,     8,     1,     1,     1,     1,
       2,     2,     1,     1,     1,     2,     2,     2,     1,     2,
       1,     1,     3,     0,     2,     4,     6,     0,     1,     0,
       3,     1,     3,     1,     1,     0,     3,     1,     3,     0,
       1,     1,     0,     3,     1,     3,     1,     1,     0,     1,
       0,     2,     5,     1,     2,     3,     6,     0,     2,     1,
       3,     5,     5,     5,     5,     4,     3,     6,     6,     5,
       5,     5,     5,     5,     4,     7,     0,     2,     0,     2,
       2,     2,     3,     2,     3,     1,     3,     4,     2,     2,
       2,     2,     2,     1,     4,     0,     2,     1,     1,     1,
       1,     2,     2,     2,     3,     6,     9,     3,     6,     3,
       6,     9,     9,     1,     3,     1,     1,     1,     2,     2,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     7,     5,    13,     9,     5,     2,     1,     0,
       3,     1,     1,     3,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     2,     1,     1,
       1,     1,     1,     1,     0,     1,     3,     0,     1,     5,
       5,     5,     4,     3,     1,     1,     1,     3,     4,     3,
       1,     1,     1,     1,     1,     1,     1,     4,     3,     4,
       4,     4,     3,     7,     5,     6,     6,     6,     3,     1,
       3,     1,     3,     1,     3,     3,     2,     3,     2,     3,
       0,     3,     1,     1,     4,     1,     2,     1,     2,     1,
       2,     1,     1,     0,     4,     3,     5,     5,     4,     4,
      11,     9,    12,    14,     6,     8,     5,     7,     3,     5,
       4,     1,     4,    11,     9,    12,    14,     6,     8,     5,
       7,     3,     1,     0,     2,     4,     1,     1,     1,     2,
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
#line 194 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.modlist) = (yyvsp[0].modlist); modlist = (yyvsp[0].modlist); }
#line 2224 "y.tab.c" /* yacc.c:1646  */
    break;

  case 3:
#line 198 "xi-grammar.y" /* yacc.c:1646  */
    { 
		  (yyval.modlist) = 0; 
		}
#line 2232 "y.tab.c" /* yacc.c:1646  */
    break;

  case 4:
#line 202 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.modlist) = new AstChildren<Module>(lineno, (yyvsp[-1].module), (yyvsp[0].modlist)); }
#line 2238 "y.tab.c" /* yacc.c:1646  */
    break;

  case 5:
#line 206 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 2244 "y.tab.c" /* yacc.c:1646  */
    break;

  case 6:
#line 208 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 2250 "y.tab.c" /* yacc.c:1646  */
    break;

  case 7:
#line 212 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 2256 "y.tab.c" /* yacc.c:1646  */
    break;

  case 8:
#line 214 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 2262 "y.tab.c" /* yacc.c:1646  */
    break;

  case 9:
#line 219 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 2268 "y.tab.c" /* yacc.c:1646  */
    break;

  case 10:
#line 220 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MODULE); YYABORT; }
#line 2274 "y.tab.c" /* yacc.c:1646  */
    break;

  case 11:
#line 221 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MAINMODULE); YYABORT; }
#line 2280 "y.tab.c" /* yacc.c:1646  */
    break;

  case 12:
#line 222 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(EXTERN); YYABORT; }
#line 2286 "y.tab.c" /* yacc.c:1646  */
    break;

  case 13:
#line 224 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITCALL); YYABORT; }
#line 2292 "y.tab.c" /* yacc.c:1646  */
    break;

  case 14:
#line 225 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITNODE); YYABORT; }
#line 2298 "y.tab.c" /* yacc.c:1646  */
    break;

  case 15:
#line 226 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITPROC); YYABORT; }
#line 2304 "y.tab.c" /* yacc.c:1646  */
    break;

  case 16:
#line 228 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CHARE); }
#line 2310 "y.tab.c" /* yacc.c:1646  */
    break;

  case 17:
#line 229 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MAINCHARE); }
#line 2316 "y.tab.c" /* yacc.c:1646  */
    break;

  case 18:
#line 230 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(GROUP); }
#line 2322 "y.tab.c" /* yacc.c:1646  */
    break;

  case 19:
#line 231 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NODEGROUP); }
#line 2328 "y.tab.c" /* yacc.c:1646  */
    break;

  case 20:
#line 232 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ARRAY); }
#line 2334 "y.tab.c" /* yacc.c:1646  */
    break;

  case 21:
#line 236 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INCLUDE); YYABORT; }
#line 2340 "y.tab.c" /* yacc.c:1646  */
    break;

  case 22:
#line 237 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(STACKSIZE); YYABORT; }
#line 2346 "y.tab.c" /* yacc.c:1646  */
    break;

  case 23:
#line 238 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(THREADED); YYABORT; }
#line 2352 "y.tab.c" /* yacc.c:1646  */
    break;

  case 24:
#line 239 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(TEMPLATE); YYABORT; }
#line 2358 "y.tab.c" /* yacc.c:1646  */
    break;

  case 25:
#line 240 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(SYNC); YYABORT; }
#line 2364 "y.tab.c" /* yacc.c:1646  */
    break;

  case 26:
#line 241 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IGET); YYABORT; }
#line 2370 "y.tab.c" /* yacc.c:1646  */
    break;

  case 27:
#line 242 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(EXCLUSIVE); YYABORT; }
#line 2376 "y.tab.c" /* yacc.c:1646  */
    break;

  case 28:
#line 243 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IMMEDIATE); YYABORT; }
#line 2382 "y.tab.c" /* yacc.c:1646  */
    break;

  case 29:
#line 244 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(SKIPSCHED); YYABORT; }
#line 2388 "y.tab.c" /* yacc.c:1646  */
    break;

  case 30:
#line 245 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INLINE); YYABORT; }
#line 2394 "y.tab.c" /* yacc.c:1646  */
    break;

  case 31:
#line 246 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(VIRTUAL); YYABORT; }
#line 2400 "y.tab.c" /* yacc.c:1646  */
    break;

  case 32:
#line 247 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MIGRATABLE); YYABORT; }
#line 2406 "y.tab.c" /* yacc.c:1646  */
    break;

  case 33:
#line 248 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CREATEHERE); YYABORT; }
#line 2412 "y.tab.c" /* yacc.c:1646  */
    break;

  case 34:
#line 249 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CREATEHOME); YYABORT; }
#line 2418 "y.tab.c" /* yacc.c:1646  */
    break;

  case 35:
#line 250 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOKEEP); YYABORT; }
#line 2424 "y.tab.c" /* yacc.c:1646  */
    break;

  case 36:
#line 251 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOTRACE); YYABORT; }
#line 2430 "y.tab.c" /* yacc.c:1646  */
    break;

  case 37:
#line 252 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(APPWORK); YYABORT; }
#line 2436 "y.tab.c" /* yacc.c:1646  */
    break;

  case 38:
#line 255 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(PACKED); YYABORT; }
#line 2442 "y.tab.c" /* yacc.c:1646  */
    break;

  case 39:
#line 256 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(VARSIZE); YYABORT; }
#line 2448 "y.tab.c" /* yacc.c:1646  */
    break;

  case 40:
#line 257 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ENTRY); YYABORT; }
#line 2454 "y.tab.c" /* yacc.c:1646  */
    break;

  case 41:
#line 258 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(FOR); YYABORT; }
#line 2460 "y.tab.c" /* yacc.c:1646  */
    break;

  case 42:
#line 259 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(FORALL); YYABORT; }
#line 2466 "y.tab.c" /* yacc.c:1646  */
    break;

  case 43:
#line 260 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(WHILE); YYABORT; }
#line 2472 "y.tab.c" /* yacc.c:1646  */
    break;

  case 44:
#line 261 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(WHEN); YYABORT; }
#line 2478 "y.tab.c" /* yacc.c:1646  */
    break;

  case 45:
#line 262 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(OVERLAP); YYABORT; }
#line 2484 "y.tab.c" /* yacc.c:1646  */
    break;

  case 46:
#line 263 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ATOMIC); YYABORT; }
#line 2490 "y.tab.c" /* yacc.c:1646  */
    break;

  case 47:
#line 264 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IF); YYABORT; }
#line 2496 "y.tab.c" /* yacc.c:1646  */
    break;

  case 48:
#line 265 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ELSE); YYABORT; }
#line 2502 "y.tab.c" /* yacc.c:1646  */
    break;

  case 49:
#line 267 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(LOCAL); YYABORT; }
#line 2508 "y.tab.c" /* yacc.c:1646  */
    break;

  case 50:
#line 269 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(USING); YYABORT; }
#line 2514 "y.tab.c" /* yacc.c:1646  */
    break;

  case 51:
#line 270 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ACCEL); YYABORT; }
#line 2520 "y.tab.c" /* yacc.c:1646  */
    break;

  case 52:
#line 273 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ACCELBLOCK); YYABORT; }
#line 2526 "y.tab.c" /* yacc.c:1646  */
    break;

  case 53:
#line 274 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MEMCRITICAL); YYABORT; }
#line 2532 "y.tab.c" /* yacc.c:1646  */
    break;

  case 54:
#line 275 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(DISKPREFETCH); YYABORT; }
#line 2538 "y.tab.c" /* yacc.c:1646  */
    break;

  case 55:
#line 276 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(REDUCTIONTARGET); YYABORT; }
#line 2544 "y.tab.c" /* yacc.c:1646  */
    break;

  case 56:
#line 277 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CASE); YYABORT; }
#line 2550 "y.tab.c" /* yacc.c:1646  */
    break;

  case 57:
#line 281 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 2556 "y.tab.c" /* yacc.c:1646  */
    break;

  case 58:
#line 283 "xi-grammar.y" /* yacc.c:1646  */
    {
		  char *tmp = new char[strlen((yyvsp[-3].strval))+strlen((yyvsp[0].strval))+3];
		  sprintf(tmp,"%s::%s", (yyvsp[-3].strval), (yyvsp[0].strval));
		  (yyval.strval) = tmp;
		}
#line 2566 "y.tab.c" /* yacc.c:1646  */
    break;

  case 59:
#line 290 "xi-grammar.y" /* yacc.c:1646  */
    { 
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist)); 
		}
#line 2574 "y.tab.c" /* yacc.c:1646  */
    break;

  case 60:
#line 294 "xi-grammar.y" /* yacc.c:1646  */
    {  
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist)); 
		    (yyval.module)->setMain();
		}
#line 2583 "y.tab.c" /* yacc.c:1646  */
    break;

  case 61:
#line 301 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = 0; }
#line 2589 "y.tab.c" /* yacc.c:1646  */
    break;

  case 62:
#line 303 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = (yyvsp[-2].conslist); }
#line 2595 "y.tab.c" /* yacc.c:1646  */
    break;

  case 63:
#line 307 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = 0; }
#line 2601 "y.tab.c" /* yacc.c:1646  */
    break;

  case 64:
#line 309 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = new ConstructList(lineno, (yyvsp[-1].construct), (yyvsp[0].conslist)); }
#line 2607 "y.tab.c" /* yacc.c:1646  */
    break;

  case 65:
#line 313 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new UsingScope((yyvsp[0].strval), false); }
#line 2613 "y.tab.c" /* yacc.c:1646  */
    break;

  case 66:
#line 315 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new UsingScope((yyvsp[0].strval), true); }
#line 2619 "y.tab.c" /* yacc.c:1646  */
    break;

  case 67:
#line 317 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].member)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].member); }
#line 2625 "y.tab.c" /* yacc.c:1646  */
    break;

  case 68:
#line 319 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].message)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].message); }
#line 2631 "y.tab.c" /* yacc.c:1646  */
    break;

  case 69:
#line 321 "xi-grammar.y" /* yacc.c:1646  */
    {
                  Entry *e = new Entry(lineno, 0, (yyvsp[-4].type), (yyvsp[-2].strval), (yyvsp[0].plist), 0, 0, 0, (yylsp[-6]).first_line, (yyloc).last_line);
                  int isExtern = 1;
                  e->setExtern(isExtern);
                  e->targs = (yyvsp[-1].tparlist);
                  e->label = new XStr;
                  (yyvsp[-3].ntype)->print(*e->label);
                  (yyval.construct) = e;
                }
#line 2645 "y.tab.c" /* yacc.c:1646  */
    break;

  case 70:
#line 333 "xi-grammar.y" /* yacc.c:1646  */
    { if((yyvsp[-2].conslist)) (yyvsp[-2].conslist)->recurse<int&>((yyvsp[-4].intval), &Construct::setExtern); (yyval.construct) = (yyvsp[-2].conslist); }
#line 2651 "y.tab.c" /* yacc.c:1646  */
    break;

  case 71:
#line 335 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new Scope((yyvsp[-3].strval), (yyvsp[-1].conslist)); }
#line 2657 "y.tab.c" /* yacc.c:1646  */
    break;

  case 72:
#line 337 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = (yyvsp[-1].construct); }
#line 2663 "y.tab.c" /* yacc.c:1646  */
    break;

  case 73:
#line 339 "xi-grammar.y" /* yacc.c:1646  */
    {
          ERROR("preceding construct must be semicolon terminated",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 2673 "y.tab.c" /* yacc.c:1646  */
    break;

  case 74:
#line 345 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].module)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].module); }
#line 2679 "y.tab.c" /* yacc.c:1646  */
    break;

  case 75:
#line 347 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2685 "y.tab.c" /* yacc.c:1646  */
    break;

  case 76:
#line 349 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2691 "y.tab.c" /* yacc.c:1646  */
    break;

  case 77:
#line 351 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2697 "y.tab.c" /* yacc.c:1646  */
    break;

  case 78:
#line 353 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2703 "y.tab.c" /* yacc.c:1646  */
    break;

  case 79:
#line 355 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].templat)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].templat); }
#line 2709 "y.tab.c" /* yacc.c:1646  */
    break;

  case 80:
#line 357 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = NULL; }
#line 2715 "y.tab.c" /* yacc.c:1646  */
    break;

  case 81:
#line 359 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = NULL; }
#line 2721 "y.tab.c" /* yacc.c:1646  */
    break;

  case 82:
#line 361 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = (yyvsp[0].accelBlock); }
#line 2727 "y.tab.c" /* yacc.c:1646  */
    break;

  case 83:
#line 363 "xi-grammar.y" /* yacc.c:1646  */
    {
          ERROR("invalid construct",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 2737 "y.tab.c" /* yacc.c:1646  */
    break;

  case 84:
#line 371 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamType((yyvsp[0].type)); }
#line 2743 "y.tab.c" /* yacc.c:1646  */
    break;

  case 85:
#line 373 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
#line 2749 "y.tab.c" /* yacc.c:1646  */
    break;

  case 86:
#line 375 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
#line 2755 "y.tab.c" /* yacc.c:1646  */
    break;

  case 87:
#line 379 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = new TParamList((yyvsp[0].tparam)); }
#line 2761 "y.tab.c" /* yacc.c:1646  */
    break;

  case 88:
#line 381 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = new TParamList((yyvsp[-2].tparam), (yyvsp[0].tparlist)); }
#line 2767 "y.tab.c" /* yacc.c:1646  */
    break;

  case 89:
#line 385 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = 0; }
#line 2773 "y.tab.c" /* yacc.c:1646  */
    break;

  case 90:
#line 387 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = (yyvsp[0].tparlist); }
#line 2779 "y.tab.c" /* yacc.c:1646  */
    break;

  case 91:
#line 391 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = 0; }
#line 2785 "y.tab.c" /* yacc.c:1646  */
    break;

  case 92:
#line 393 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = (yyvsp[-1].tparlist); }
#line 2791 "y.tab.c" /* yacc.c:1646  */
    break;

  case 93:
#line 397 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("int"); }
#line 2797 "y.tab.c" /* yacc.c:1646  */
    break;

  case 94:
#line 399 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long"); }
#line 2803 "y.tab.c" /* yacc.c:1646  */
    break;

  case 95:
#line 401 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("short"); }
#line 2809 "y.tab.c" /* yacc.c:1646  */
    break;

  case 96:
#line 403 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("char"); }
#line 2815 "y.tab.c" /* yacc.c:1646  */
    break;

  case 97:
#line 405 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned int"); }
#line 2821 "y.tab.c" /* yacc.c:1646  */
    break;

  case 98:
#line 407 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long"); }
#line 2827 "y.tab.c" /* yacc.c:1646  */
    break;

  case 99:
#line 409 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long"); }
#line 2833 "y.tab.c" /* yacc.c:1646  */
    break;

  case 100:
#line 411 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long long"); }
#line 2839 "y.tab.c" /* yacc.c:1646  */
    break;

  case 101:
#line 413 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned short"); }
#line 2845 "y.tab.c" /* yacc.c:1646  */
    break;

  case 102:
#line 415 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned char"); }
#line 2851 "y.tab.c" /* yacc.c:1646  */
    break;

  case 103:
#line 417 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long long"); }
#line 2857 "y.tab.c" /* yacc.c:1646  */
    break;

  case 104:
#line 419 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("float"); }
#line 2863 "y.tab.c" /* yacc.c:1646  */
    break;

  case 105:
#line 421 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("double"); }
#line 2869 "y.tab.c" /* yacc.c:1646  */
    break;

  case 106:
#line 423 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long double"); }
#line 2875 "y.tab.c" /* yacc.c:1646  */
    break;

  case 107:
#line 425 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("void"); }
#line 2881 "y.tab.c" /* yacc.c:1646  */
    break;

  case 108:
#line 428 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ntype) = new NamedType((yyvsp[-1].strval),(yyvsp[0].tparlist)); }
#line 2887 "y.tab.c" /* yacc.c:1646  */
    break;

  case 109:
#line 429 "xi-grammar.y" /* yacc.c:1646  */
    { 
                    const char* basename, *scope;
                    splitScopedName((yyvsp[-1].strval), &scope, &basename);
                    (yyval.ntype) = new NamedType(basename, (yyvsp[0].tparlist), scope);
                }
#line 2897 "y.tab.c" /* yacc.c:1646  */
    break;

  case 110:
#line 437 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 2903 "y.tab.c" /* yacc.c:1646  */
    break;

  case 111:
#line 439 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ntype); }
#line 2909 "y.tab.c" /* yacc.c:1646  */
    break;

  case 112:
#line 443 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ptype) = new PtrType((yyvsp[-1].type)); }
#line 2915 "y.tab.c" /* yacc.c:1646  */
    break;

  case 113:
#line 447 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
#line 2921 "y.tab.c" /* yacc.c:1646  */
    break;

  case 114:
#line 449 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
#line 2927 "y.tab.c" /* yacc.c:1646  */
    break;

  case 115:
#line 453 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ftype) = new FuncType((yyvsp[-7].type), (yyvsp[-4].strval), (yyvsp[-1].plist)); }
#line 2933 "y.tab.c" /* yacc.c:1646  */
    break;

  case 116:
#line 457 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 2939 "y.tab.c" /* yacc.c:1646  */
    break;

  case 117:
#line 459 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 2945 "y.tab.c" /* yacc.c:1646  */
    break;

  case 118:
#line 461 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 2951 "y.tab.c" /* yacc.c:1646  */
    break;

  case 119:
#line 463 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ftype); }
#line 2957 "y.tab.c" /* yacc.c:1646  */
    break;

  case 120:
#line 465 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[0].type)); }
#line 2963 "y.tab.c" /* yacc.c:1646  */
    break;

  case 121:
#line 467 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[-1].type)); }
#line 2969 "y.tab.c" /* yacc.c:1646  */
    break;

  case 122:
#line 471 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 2975 "y.tab.c" /* yacc.c:1646  */
    break;

  case 123:
#line 473 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 2981 "y.tab.c" /* yacc.c:1646  */
    break;

  case 124:
#line 475 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 2987 "y.tab.c" /* yacc.c:1646  */
    break;

  case 125:
#line 477 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[0].type)); }
#line 2993 "y.tab.c" /* yacc.c:1646  */
    break;

  case 126:
#line 479 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[-1].type)); }
#line 2999 "y.tab.c" /* yacc.c:1646  */
    break;

  case 127:
#line 483 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ReferenceType((yyvsp[-1].type)); }
#line 3005 "y.tab.c" /* yacc.c:1646  */
    break;

  case 128:
#line 485 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3011 "y.tab.c" /* yacc.c:1646  */
    break;

  case 129:
#line 489 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ReferenceType((yyvsp[-1].type)); }
#line 3017 "y.tab.c" /* yacc.c:1646  */
    break;

  case 130:
#line 491 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3023 "y.tab.c" /* yacc.c:1646  */
    break;

  case 131:
#line 495 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 3029 "y.tab.c" /* yacc.c:1646  */
    break;

  case 132:
#line 499 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = (yyvsp[-1].val); }
#line 3035 "y.tab.c" /* yacc.c:1646  */
    break;

  case 133:
#line 503 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.vallist) = 0; }
#line 3041 "y.tab.c" /* yacc.c:1646  */
    break;

  case 134:
#line 505 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.vallist) = new ValueList((yyvsp[-1].val), (yyvsp[0].vallist)); }
#line 3047 "y.tab.c" /* yacc.c:1646  */
    break;

  case 135:
#line 509 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].vallist)); }
#line 3053 "y.tab.c" /* yacc.c:1646  */
    break;

  case 136:
#line 513 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[-3].type), (yyvsp[-1].strval), (yyvsp[0].vallist), 1); }
#line 3059 "y.tab.c" /* yacc.c:1646  */
    break;

  case 137:
#line 517 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0;}
#line 3065 "y.tab.c" /* yacc.c:1646  */
    break;

  case 138:
#line 519 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0;}
#line 3071 "y.tab.c" /* yacc.c:1646  */
    break;

  case 139:
#line 523 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3077 "y.tab.c" /* yacc.c:1646  */
    break;

  case 140:
#line 525 "xi-grammar.y" /* yacc.c:1646  */
    { 
		  /*
		  printf("Warning: Message attributes are being phased out.\n");
		  printf("Warning: Please remove them from interface files.\n");
		  */
		  (yyval.intval) = (yyvsp[-1].intval); 
		}
#line 3089 "y.tab.c" /* yacc.c:1646  */
    break;

  case 141:
#line 535 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[0].intval); }
#line 3095 "y.tab.c" /* yacc.c:1646  */
    break;

  case 142:
#line 537 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[-2].intval) | (yyvsp[0].intval); }
#line 3101 "y.tab.c" /* yacc.c:1646  */
    break;

  case 143:
#line 541 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3107 "y.tab.c" /* yacc.c:1646  */
    break;

  case 144:
#line 543 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3113 "y.tab.c" /* yacc.c:1646  */
    break;

  case 145:
#line 547 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = 0; }
#line 3119 "y.tab.c" /* yacc.c:1646  */
    break;

  case 146:
#line 549 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-1].cattr); }
#line 3125 "y.tab.c" /* yacc.c:1646  */
    break;

  case 147:
#line 553 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[0].cattr); }
#line 3131 "y.tab.c" /* yacc.c:1646  */
    break;

  case 148:
#line 555 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
#line 3137 "y.tab.c" /* yacc.c:1646  */
    break;

  case 149:
#line 559 "xi-grammar.y" /* yacc.c:1646  */
    { python_doc = NULL; (yyval.intval) = 0; }
#line 3143 "y.tab.c" /* yacc.c:1646  */
    break;

  case 150:
#line 561 "xi-grammar.y" /* yacc.c:1646  */
    { python_doc = (yyvsp[0].strval); (yyval.intval) = 0; }
#line 3149 "y.tab.c" /* yacc.c:1646  */
    break;

  case 151:
#line 565 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CPYTHON; }
#line 3155 "y.tab.c" /* yacc.c:1646  */
    break;

  case 152:
#line 569 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = 0; }
#line 3161 "y.tab.c" /* yacc.c:1646  */
    break;

  case 153:
#line 571 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-1].cattr); }
#line 3167 "y.tab.c" /* yacc.c:1646  */
    break;

  case 154:
#line 575 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[0].cattr); }
#line 3173 "y.tab.c" /* yacc.c:1646  */
    break;

  case 155:
#line 577 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
#line 3179 "y.tab.c" /* yacc.c:1646  */
    break;

  case 156:
#line 581 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CMIGRATABLE; }
#line 3185 "y.tab.c" /* yacc.c:1646  */
    break;

  case 157:
#line 583 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CPYTHON; }
#line 3191 "y.tab.c" /* yacc.c:1646  */
    break;

  case 158:
#line 587 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3197 "y.tab.c" /* yacc.c:1646  */
    break;

  case 159:
#line 589 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 3203 "y.tab.c" /* yacc.c:1646  */
    break;

  case 160:
#line 592 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3209 "y.tab.c" /* yacc.c:1646  */
    break;

  case 161:
#line 594 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 3215 "y.tab.c" /* yacc.c:1646  */
    break;

  case 162:
#line 597 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mv) = new MsgVar((yyvsp[-3].type), (yyvsp[-2].strval), (yyvsp[-4].intval), (yyvsp[-1].intval)); }
#line 3221 "y.tab.c" /* yacc.c:1646  */
    break;

  case 163:
#line 601 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mvlist) = new MsgVarList((yyvsp[0].mv)); }
#line 3227 "y.tab.c" /* yacc.c:1646  */
    break;

  case 164:
#line 603 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mvlist) = new MsgVarList((yyvsp[-1].mv), (yyvsp[0].mvlist)); }
#line 3233 "y.tab.c" /* yacc.c:1646  */
    break;

  case 165:
#line 607 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, (yyvsp[0].ntype)); }
#line 3239 "y.tab.c" /* yacc.c:1646  */
    break;

  case 166:
#line 609 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, (yyvsp[-3].ntype), (yyvsp[-1].mvlist)); }
#line 3245 "y.tab.c" /* yacc.c:1646  */
    break;

  case 167:
#line 613 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = 0; }
#line 3251 "y.tab.c" /* yacc.c:1646  */
    break;

  case 168:
#line 615 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = (yyvsp[0].typelist); }
#line 3257 "y.tab.c" /* yacc.c:1646  */
    break;

  case 169:
#line 619 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = new TypeList((yyvsp[0].ntype)); }
#line 3263 "y.tab.c" /* yacc.c:1646  */
    break;

  case 170:
#line 621 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = new TypeList((yyvsp[-2].ntype), (yyvsp[0].typelist)); }
#line 3269 "y.tab.c" /* yacc.c:1646  */
    break;

  case 171:
#line 625 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr)|Chare::CCHARE, (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3275 "y.tab.c" /* yacc.c:1646  */
    break;

  case 172:
#line 627 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3281 "y.tab.c" /* yacc.c:1646  */
    break;

  case 173:
#line 631 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3287 "y.tab.c" /* yacc.c:1646  */
    break;

  case 174:
#line 635 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new NodeGroup(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3293 "y.tab.c" /* yacc.c:1646  */
    break;

  case 175:
#line 639 "xi-grammar.y" /* yacc.c:1646  */
    {/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",(yyvsp[-2].strval));
			(yyval.ntype) = new NamedType(buf); 
		}
#line 3303 "y.tab.c" /* yacc.c:1646  */
    break;

  case 176:
#line 645 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ntype) = new NamedType((yyvsp[-1].strval)); }
#line 3309 "y.tab.c" /* yacc.c:1646  */
    break;

  case 177:
#line 649 "xi-grammar.y" /* yacc.c:1646  */
    {  (yyval.chare) = new Array(lineno, (yyvsp[-4].cattr), (yyvsp[-3].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3315 "y.tab.c" /* yacc.c:1646  */
    break;

  case 178:
#line 651 "xi-grammar.y" /* yacc.c:1646  */
    {  (yyval.chare) = new Array(lineno, (yyvsp[-3].cattr), (yyvsp[-4].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3321 "y.tab.c" /* yacc.c:1646  */
    break;

  case 179:
#line 655 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr)|Chare::CCHARE, new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist));}
#line 3327 "y.tab.c" /* yacc.c:1646  */
    break;

  case 180:
#line 657 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3333 "y.tab.c" /* yacc.c:1646  */
    break;

  case 181:
#line 661 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3339 "y.tab.c" /* yacc.c:1646  */
    break;

  case 182:
#line 665 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new NodeGroup( lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3345 "y.tab.c" /* yacc.c:1646  */
    break;

  case 183:
#line 669 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Array( lineno, 0, (yyvsp[-3].ntype), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3351 "y.tab.c" /* yacc.c:1646  */
    break;

  case 184:
#line 673 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-1].strval))); }
#line 3357 "y.tab.c" /* yacc.c:1646  */
    break;

  case 185:
#line 675 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-4].strval)), (yyvsp[-2].mvlist)); }
#line 3363 "y.tab.c" /* yacc.c:1646  */
    break;

  case 186:
#line 679 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = 0; }
#line 3369 "y.tab.c" /* yacc.c:1646  */
    break;

  case 187:
#line 681 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3375 "y.tab.c" /* yacc.c:1646  */
    break;

  case 188:
#line 685 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = 0; }
#line 3381 "y.tab.c" /* yacc.c:1646  */
    break;

  case 189:
#line 687 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3387 "y.tab.c" /* yacc.c:1646  */
    break;

  case 190:
#line 689 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3393 "y.tab.c" /* yacc.c:1646  */
    break;

  case 191:
#line 691 "xi-grammar.y" /* yacc.c:1646  */
    {
		  XStr typeStr;
		  (yyvsp[0].ntype)->print(typeStr);
		  char *tmp = strdup(typeStr.get_string());
		  (yyval.strval) = tmp;
		}
#line 3404 "y.tab.c" /* yacc.c:1646  */
    break;

  case 192:
#line 700 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TType(new NamedType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3410 "y.tab.c" /* yacc.c:1646  */
    break;

  case 193:
#line 702 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TFunc((yyvsp[-1].ftype), (yyvsp[0].strval)); }
#line 3416 "y.tab.c" /* yacc.c:1646  */
    break;

  case 194:
#line 704 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TName((yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].strval)); }
#line 3422 "y.tab.c" /* yacc.c:1646  */
    break;

  case 195:
#line 708 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = new TVarList((yyvsp[0].tvar)); }
#line 3428 "y.tab.c" /* yacc.c:1646  */
    break;

  case 196:
#line 710 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = new TVarList((yyvsp[-2].tvar), (yyvsp[0].tvarlist)); }
#line 3434 "y.tab.c" /* yacc.c:1646  */
    break;

  case 197:
#line 714 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = (yyvsp[-1].tvarlist); }
#line 3440 "y.tab.c" /* yacc.c:1646  */
    break;

  case 198:
#line 718 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3446 "y.tab.c" /* yacc.c:1646  */
    break;

  case 199:
#line 720 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3452 "y.tab.c" /* yacc.c:1646  */
    break;

  case 200:
#line 722 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3458 "y.tab.c" /* yacc.c:1646  */
    break;

  case 201:
#line 724 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3464 "y.tab.c" /* yacc.c:1646  */
    break;

  case 202:
#line 726 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].message)); (yyvsp[0].message)->setTemplate((yyval.templat)); }
#line 3470 "y.tab.c" /* yacc.c:1646  */
    break;

  case 203:
#line 730 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = 0; }
#line 3476 "y.tab.c" /* yacc.c:1646  */
    break;

  case 204:
#line 732 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = (yyvsp[-2].mbrlist); }
#line 3482 "y.tab.c" /* yacc.c:1646  */
    break;

  case 205:
#line 736 "xi-grammar.y" /* yacc.c:1646  */
    { 
                  if (!connectEntries.empty()) {
                    (yyval.mbrlist) = new AstChildren<Member>(connectEntries);
		  } else {
		    (yyval.mbrlist) = 0; 
                  }
		}
#line 3494 "y.tab.c" /* yacc.c:1646  */
    break;

  case 206:
#line 744 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = new AstChildren<Member>(-1, (yyvsp[-1].member), (yyvsp[0].mbrlist)); }
#line 3500 "y.tab.c" /* yacc.c:1646  */
    break;

  case 207:
#line 748 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].readonly); }
#line 3506 "y.tab.c" /* yacc.c:1646  */
    break;

  case 208:
#line 750 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].readonly); }
#line 3512 "y.tab.c" /* yacc.c:1646  */
    break;

  case 210:
#line 753 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].member); }
#line 3518 "y.tab.c" /* yacc.c:1646  */
    break;

  case 211:
#line 755 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].pupable); }
#line 3524 "y.tab.c" /* yacc.c:1646  */
    break;

  case 212:
#line 757 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].includeFile); }
#line 3530 "y.tab.c" /* yacc.c:1646  */
    break;

  case 213:
#line 759 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new ClassDeclaration(lineno,(yyvsp[0].strval)); }
#line 3536 "y.tab.c" /* yacc.c:1646  */
    break;

  case 214:
#line 763 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1); }
#line 3542 "y.tab.c" /* yacc.c:1646  */
    break;

  case 215:
#line 765 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1); }
#line 3548 "y.tab.c" /* yacc.c:1646  */
    break;

  case 216:
#line 767 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[-6].strval)) + '<' +
					    ((yyvsp[-4].tparlist))->to_string() + '>').c_str()),
				    1);
		}
#line 3558 "y.tab.c" /* yacc.c:1646  */
    break;

  case 217:
#line 773 "xi-grammar.y" /* yacc.c:1646  */
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[-2]).first_column, (yylsp[-2]).last_column, (yylsp[-2]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1);
		}
#line 3568 "y.tab.c" /* yacc.c:1646  */
    break;

  case 218:
#line 779 "xi-grammar.y" /* yacc.c:1646  */
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[-5]).first_column, (yylsp[-5]).last_column, (yylsp[-5]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1);
		}
#line 3578 "y.tab.c" /* yacc.c:1646  */
    break;

  case 219:
#line 788 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 0); }
#line 3584 "y.tab.c" /* yacc.c:1646  */
    break;

  case 220:
#line 790 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 0); }
#line 3590 "y.tab.c" /* yacc.c:1646  */
    break;

  case 221:
#line 792 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[-6].strval)) + '<' +
					    ((yyvsp[-4].tparlist))->to_string() + '>').c_str()),
				    0);
		}
#line 3600 "y.tab.c" /* yacc.c:1646  */
    break;

  case 222:
#line 798 "xi-grammar.y" /* yacc.c:1646  */
    {
                  InitCall* rtn = new InitCall(lineno, (yyvsp[-3].strval), 0);
                  rtn->setAccel();
                  (yyval.member) = rtn;
		}
#line 3610 "y.tab.c" /* yacc.c:1646  */
    break;

  case 223:
#line 806 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[0].ntype),0); }
#line 3616 "y.tab.c" /* yacc.c:1646  */
    break;

  case 224:
#line 808 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[-2].ntype),(yyvsp[0].pupable)); }
#line 3622 "y.tab.c" /* yacc.c:1646  */
    break;

  case 225:
#line 811 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.includeFile) = new IncludeFile(lineno,(yyvsp[0].strval)); }
#line 3628 "y.tab.c" /* yacc.c:1646  */
    break;

  case 226:
#line 815 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].member); }
#line 3634 "y.tab.c" /* yacc.c:1646  */
    break;

  case 227:
#line 819 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].entry); }
#line 3640 "y.tab.c" /* yacc.c:1646  */
    break;

  case 228:
#line 821 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyvsp[0].entry)->tspec = (yyvsp[-1].tvarlist);
                  (yyval.member) = (yyvsp[0].entry);
                }
#line 3649 "y.tab.c" /* yacc.c:1646  */
    break;

  case 229:
#line 826 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[-1].member); }
#line 3655 "y.tab.c" /* yacc.c:1646  */
    break;

  case 230:
#line 828 "xi-grammar.y" /* yacc.c:1646  */
    {
          ERROR("invalid SDAG member",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 3665 "y.tab.c" /* yacc.c:1646  */
    break;

  case 231:
#line 836 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3671 "y.tab.c" /* yacc.c:1646  */
    break;

  case 232:
#line 838 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3677 "y.tab.c" /* yacc.c:1646  */
    break;

  case 233:
#line 840 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3683 "y.tab.c" /* yacc.c:1646  */
    break;

  case 234:
#line 842 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3689 "y.tab.c" /* yacc.c:1646  */
    break;

  case 235:
#line 844 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3695 "y.tab.c" /* yacc.c:1646  */
    break;

  case 236:
#line 846 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3701 "y.tab.c" /* yacc.c:1646  */
    break;

  case 237:
#line 848 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3707 "y.tab.c" /* yacc.c:1646  */
    break;

  case 238:
#line 850 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3713 "y.tab.c" /* yacc.c:1646  */
    break;

  case 239:
#line 852 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3719 "y.tab.c" /* yacc.c:1646  */
    break;

  case 240:
#line 854 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3725 "y.tab.c" /* yacc.c:1646  */
    break;

  case 241:
#line 856 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3731 "y.tab.c" /* yacc.c:1646  */
    break;

  case 242:
#line 859 "xi-grammar.y" /* yacc.c:1646  */
    { 
                  (yyval.entry) = new Entry(lineno, (yyvsp[-5].intval), (yyvsp[-4].type), (yyvsp[-3].strval), (yyvsp[-2].plist), (yyvsp[-1].val), (yyvsp[0].sentry), (const char *) NULL, (yylsp[-6]).first_line, (yyloc).last_line);
		  if ((yyvsp[0].sentry) != 0) { 
		    (yyvsp[0].sentry)->con1 = new SdagConstruct(SIDENT, (yyvsp[-3].strval));
                    (yyvsp[0].sentry)->entry = (yyval.entry);
                    (yyvsp[0].sentry)->con1->entry = (yyval.entry);
                    (yyvsp[0].sentry)->param = new ParamList((yyvsp[-2].plist));
                  }
		}
#line 3745 "y.tab.c" /* yacc.c:1646  */
    break;

  case 243:
#line 869 "xi-grammar.y" /* yacc.c:1646  */
    { 
                  Entry *e = new Entry(lineno, (yyvsp[-3].intval), 0, (yyvsp[-2].strval), (yyvsp[-1].plist),  0, (yyvsp[0].sentry), (const char *) NULL, (yylsp[-4]).first_line, (yyloc).last_line);
                  if ((yyvsp[0].sentry) != 0) {
		    (yyvsp[0].sentry)->con1 = new SdagConstruct(SIDENT, (yyvsp[-2].strval));
                    (yyvsp[0].sentry)->entry = e;
                    (yyvsp[0].sentry)->con1->entry = e;
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
#line 3766 "y.tab.c" /* yacc.c:1646  */
    break;

  case 244:
#line 886 "xi-grammar.y" /* yacc.c:1646  */
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
#line 3784 "y.tab.c" /* yacc.c:1646  */
    break;

  case 245:
#line 900 "xi-grammar.y" /* yacc.c:1646  */
    {
                  const char* name = (yyvsp[-3].strval);
                  ParamList* paramList = (yyvsp[-2].plist);
                  (yyval.entry) = new Entry(lineno, SDISK, new BuiltinType("void"), name, paramList, 0, 0, 0 );
                  (yyval.entry)->setOOCParam((yyvsp[-1].plist));
                }
#line 3795 "y.tab.c" /* yacc.c:1646  */
    break;

  case 246:
#line 909 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.accelBlock) = new AccelBlock(lineno, new XStr((yyvsp[-2].strval))); }
#line 3801 "y.tab.c" /* yacc.c:1646  */
    break;

  case 247:
#line 911 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.accelBlock) = new AccelBlock(lineno, NULL); }
#line 3807 "y.tab.c" /* yacc.c:1646  */
    break;

  case 248:
#line 915 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3813 "y.tab.c" /* yacc.c:1646  */
    break;

  case 249:
#line 919 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3819 "y.tab.c" /* yacc.c:1646  */
    break;

  case 250:
#line 921 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[-1].intval); }
#line 3825 "y.tab.c" /* yacc.c:1646  */
    break;

  case 251:
#line 923 "xi-grammar.y" /* yacc.c:1646  */
    { ERROR("invalid entry method attribute list",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
#line 3834 "y.tab.c" /* yacc.c:1646  */
    break;

  case 252:
#line 930 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[0].intval); }
#line 3840 "y.tab.c" /* yacc.c:1646  */
    break;

  case 253:
#line 932 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[-2].intval) | (yyvsp[0].intval); }
#line 3846 "y.tab.c" /* yacc.c:1646  */
    break;

  case 254:
#line 936 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = STHREADED; }
#line 3852 "y.tab.c" /* yacc.c:1646  */
    break;

  case 255:
#line 938 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SSYNC; }
#line 3858 "y.tab.c" /* yacc.c:1646  */
    break;

  case 256:
#line 940 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SIGET; }
#line 3864 "y.tab.c" /* yacc.c:1646  */
    break;

  case 257:
#line 942 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SLOCKED; }
#line 3870 "y.tab.c" /* yacc.c:1646  */
    break;

  case 258:
#line 944 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SCREATEHERE; }
#line 3876 "y.tab.c" /* yacc.c:1646  */
    break;

  case 259:
#line 946 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SCREATEHOME; }
#line 3882 "y.tab.c" /* yacc.c:1646  */
    break;

  case 260:
#line 948 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SNOKEEP; }
#line 3888 "y.tab.c" /* yacc.c:1646  */
    break;

  case 261:
#line 950 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SNOTRACE; }
#line 3894 "y.tab.c" /* yacc.c:1646  */
    break;

  case 262:
#line 952 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SAPPWORK; }
#line 3900 "y.tab.c" /* yacc.c:1646  */
    break;

  case 263:
#line 954 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SIMMEDIATE; }
#line 3906 "y.tab.c" /* yacc.c:1646  */
    break;

  case 264:
#line 956 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SSKIPSCHED; }
#line 3912 "y.tab.c" /* yacc.c:1646  */
    break;

  case 265:
#line 958 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SINLINE; }
#line 3918 "y.tab.c" /* yacc.c:1646  */
    break;

  case 266:
#line 960 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SLOCAL; }
#line 3924 "y.tab.c" /* yacc.c:1646  */
    break;

  case 267:
#line 962 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SPYTHON; }
#line 3930 "y.tab.c" /* yacc.c:1646  */
    break;

  case 268:
#line 964 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SMEM; }
#line 3936 "y.tab.c" /* yacc.c:1646  */
    break;

  case 269:
#line 966 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SREDUCE; }
#line 3942 "y.tab.c" /* yacc.c:1646  */
    break;

  case 270:
#line 968 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("invalid entry method attribute",
		        (yylsp[0]).first_column, (yylsp[0]).last_column);
		  yyclearin;
		  yyerrok;
		}
#line 3953 "y.tab.c" /* yacc.c:1646  */
    break;

  case 271:
#line 977 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 3959 "y.tab.c" /* yacc.c:1646  */
    break;

  case 272:
#line 979 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 3965 "y.tab.c" /* yacc.c:1646  */
    break;

  case 273:
#line 981 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 3971 "y.tab.c" /* yacc.c:1646  */
    break;

  case 274:
#line 985 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = ""; }
#line 3977 "y.tab.c" /* yacc.c:1646  */
    break;

  case 275:
#line 987 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3983 "y.tab.c" /* yacc.c:1646  */
    break;

  case 276:
#line 989 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s, %s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 3993 "y.tab.c" /* yacc.c:1646  */
    break;

  case 277:
#line 997 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = ""; }
#line 3999 "y.tab.c" /* yacc.c:1646  */
    break;

  case 278:
#line 999 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4005 "y.tab.c" /* yacc.c:1646  */
    break;

  case 279:
#line 1001 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s[%s]%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4015 "y.tab.c" /* yacc.c:1646  */
    break;

  case 280:
#line 1007 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s{%s}%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4025 "y.tab.c" /* yacc.c:1646  */
    break;

  case 281:
#line 1013 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s(%s)%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4035 "y.tab.c" /* yacc.c:1646  */
    break;

  case 282:
#line 1019 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"(%s)%s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4045 "y.tab.c" /* yacc.c:1646  */
    break;

  case 283:
#line 1027 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval));
		}
#line 4054 "y.tab.c" /* yacc.c:1646  */
    break;

  case 284:
#line 1034 "xi-grammar.y" /* yacc.c:1646  */
    { 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			(yyval.intval) = 0;
		}
#line 4064 "y.tab.c" /* yacc.c:1646  */
    break;

  case 285:
#line 1042 "xi-grammar.y" /* yacc.c:1646  */
    { 
			in_braces=0;
			(yyval.intval) = 0;
		}
#line 4073 "y.tab.c" /* yacc.c:1646  */
    break;

  case 286:
#line 1049 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[0].type));}
#line 4079 "y.tab.c" /* yacc.c:1646  */
    break;

  case 287:
#line 1051 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval)); (yyval.pname)->setConditional((yyvsp[0].intval)); }
#line 4085 "y.tab.c" /* yacc.c:1646  */
    break;

  case 288:
#line 1053 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[-3].type),(yyvsp[-2].strval),0,(yyvsp[0].val));}
#line 4091 "y.tab.c" /* yacc.c:1646  */
    break;

  case 289:
#line 1055 "xi-grammar.y" /* yacc.c:1646  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
		}
#line 4100 "y.tab.c" /* yacc.c:1646  */
    break;

  case 290:
#line 1061 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READONLY; }
#line 4106 "y.tab.c" /* yacc.c:1646  */
    break;

  case 291:
#line 1062 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READWRITE; }
#line 4112 "y.tab.c" /* yacc.c:1646  */
    break;

  case 292:
#line 1063 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_WRITEONLY; }
#line 4118 "y.tab.c" /* yacc.c:1646  */
    break;

  case 293:
#line 1066 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::OOC_BUFFER_TYPE_READONLY; }
#line 4124 "y.tab.c" /* yacc.c:1646  */
    break;

  case 294:
#line 1067 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::OOC_BUFFER_TYPE_READWRITE; }
#line 4130 "y.tab.c" /* yacc.c:1646  */
    break;

  case 295:
#line 1068 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::OOC_BUFFER_TYPE_WRITEONLY; }
#line 4136 "y.tab.c" /* yacc.c:1646  */
    break;

  case 296:
#line 1071 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr((yyvsp[0].strval)); }
#line 4142 "y.tab.c" /* yacc.c:1646  */
    break;

  case 297:
#line 1072 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "->" << (yyvsp[0].strval); }
#line 4148 "y.tab.c" /* yacc.c:1646  */
    break;

  case 298:
#line 1073 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[-2].xstrptr)) << "." << (yyvsp[0].strval); }
#line 4154 "y.tab.c" /* yacc.c:1646  */
    break;

  case 299:
#line 1075 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << *((yyvsp[-1].xstrptr)) << "]";
                  delete (yyvsp[-3].xstrptr);
                  delete (yyvsp[-1].xstrptr);
                }
#line 4165 "y.tab.c" /* yacc.c:1646  */
    break;

  case 300:
#line 1082 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << (yyvsp[-1].strval) << "]";
                  delete (yyvsp[-3].xstrptr);
                }
#line 4175 "y.tab.c" /* yacc.c:1646  */
    break;

  case 301:
#line 1088 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "(" << *((yyvsp[-1].xstrptr)) << ")";
                  delete (yyvsp[-3].xstrptr);
                  delete (yyvsp[-1].xstrptr);
                }
#line 4186 "y.tab.c" /* yacc.c:1646  */
    break;

  case 302:
#line 1097 "xi-grammar.y" /* yacc.c:1646  */
    {
                  in_bracket = 0;
                  (yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName(), (yyvsp[-1].strval));
                }
#line 4195 "y.tab.c" /* yacc.c:1646  */
    break;

  case 303:
#line 1104 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[-6].intval));
                }
#line 4205 "y.tab.c" /* yacc.c:1646  */
    break;

  case 304:
#line 1110 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType(Parameter::ACCEL_BUFFER_TYPE_READWRITE);
		}
#line 4215 "y.tab.c" /* yacc.c:1646  */
    break;

  case 305:
#line 1116 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.pname) = (yyvsp[-3].pname);
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[-5].intval));
		}
#line 4225 "y.tab.c" /* yacc.c:1646  */
    break;

  case 306:
#line 1124 "xi-grammar.y" /* yacc.c:1646  */
    {
                  char *tmp = new char[strlen((yyvsp[-3].strval))+strlen((yyvsp[-1].strval))+3];
                  sprintf(tmp,"%s[%s]", (yyvsp[-3].strval), (yyvsp[-1].strval));
                  (yyval.pname) = new Parameter(lineno, tmp);
                  (yyval.pname)->setOOCBufferType((yyvsp[-5].intval));
                }
#line 4236 "y.tab.c" /* yacc.c:1646  */
    break;

  case 307:
#line 1131 "xi-grammar.y" /* yacc.c:1646  */
    {
                  char *tmp = new char[strlen((yyvsp[-3].strval))+strlen((yyvsp[-1].strval))+3];
                  sprintf(tmp,"%s(%s)", (yyvsp[-3].strval), (yyvsp[-1].strval));
                  (yyval.pname) = new Parameter(lineno, tmp);
                  (yyval.pname)->setOOCBufferType((yyvsp[-5].intval));
                }
#line 4247 "y.tab.c" /* yacc.c:1646  */
    break;

  case 308:
#line 1138 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.pname) = new Parameter(lineno,(yyvsp[0].strval));
                  (yyval.pname)->setOOCBufferType((yyvsp[-2].intval));
                }
#line 4256 "y.tab.c" /* yacc.c:1646  */
    break;

  case 309:
#line 1145 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 4262 "y.tab.c" /* yacc.c:1646  */
    break;

  case 310:
#line 1147 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 4268 "y.tab.c" /* yacc.c:1646  */
    break;

  case 311:
#line 1151 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 4274 "y.tab.c" /* yacc.c:1646  */
    break;

  case 312:
#line 1153 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 4280 "y.tab.c" /* yacc.c:1646  */
    break;

  case 313:
#line 1157 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 4286 "y.tab.c" /* yacc.c:1646  */
    break;

  case 314:
#line 1159 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 4292 "y.tab.c" /* yacc.c:1646  */
    break;

  case 315:
#line 1163 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = (yyvsp[-1].plist); }
#line 4298 "y.tab.c" /* yacc.c:1646  */
    break;

  case 316:
#line 1165 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList(new Parameter(0, new BuiltinType("void"))); }
#line 4304 "y.tab.c" /* yacc.c:1646  */
    break;

  case 317:
#line 1169 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = (yyvsp[-1].plist); }
#line 4310 "y.tab.c" /* yacc.c:1646  */
    break;

  case 318:
#line 1171 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = 0; }
#line 4316 "y.tab.c" /* yacc.c:1646  */
    break;

  case 319:
#line 1175 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = (yyvsp[-1].plist);}
#line 4322 "y.tab.c" /* yacc.c:1646  */
    break;

  case 320:
#line 1179 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = 0; }
#line 4328 "y.tab.c" /* yacc.c:1646  */
    break;

  case 321:
#line 1181 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4334 "y.tab.c" /* yacc.c:1646  */
    break;

  case 322:
#line 1185 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = 0; }
#line 4340 "y.tab.c" /* yacc.c:1646  */
    break;

  case 323:
#line 1187 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[0].sc)); }
#line 4346 "y.tab.c" /* yacc.c:1646  */
    break;

  case 324:
#line 1189 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[-2].slist)); }
#line 4352 "y.tab.c" /* yacc.c:1646  */
    break;

  case 325:
#line 1193 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.slist) = new SListConstruct((yyvsp[0].sc)); }
#line 4358 "y.tab.c" /* yacc.c:1646  */
    break;

  case 326:
#line 1195 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.slist) = new SListConstruct((yyvsp[-1].sc), (yyvsp[0].slist));  }
#line 4364 "y.tab.c" /* yacc.c:1646  */
    break;

  case 327:
#line 1199 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.olist) = new OListConstruct((yyvsp[0].sc)); }
#line 4370 "y.tab.c" /* yacc.c:1646  */
    break;

  case 328:
#line 1201 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.olist) = new OListConstruct((yyvsp[-1].sc), (yyvsp[0].slist)); }
#line 4376 "y.tab.c" /* yacc.c:1646  */
    break;

  case 329:
#line 1205 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.clist) = new CaseListConstruct((yyvsp[0].when)); }
#line 4382 "y.tab.c" /* yacc.c:1646  */
    break;

  case 330:
#line 1207 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.clist) = new CaseListConstruct((yyvsp[-1].when), (yyvsp[0].clist)); }
#line 4388 "y.tab.c" /* yacc.c:1646  */
    break;

  case 331:
#line 1209 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("case blocks can only contain when clauses",
		        (yylsp[0]).first_column, (yylsp[0]).last_column);
		  (yyval.clist) = 0;
		}
#line 4398 "y.tab.c" /* yacc.c:1646  */
    break;

  case 332:
#line 1217 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4404 "y.tab.c" /* yacc.c:1646  */
    break;

  case 333:
#line 1219 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = 0; }
#line 4410 "y.tab.c" /* yacc.c:1646  */
    break;

  case 334:
#line 1223 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-2].entrylist), 0); }
#line 4416 "y.tab.c" /* yacc.c:1646  */
    break;

  case 335:
#line 1225 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-1].entrylist), (yyvsp[0].sc)); }
#line 4422 "y.tab.c" /* yacc.c:1646  */
    break;

  case 336:
#line 1227 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-3].entrylist), (yyvsp[-1].slist)); }
#line 4428 "y.tab.c" /* yacc.c:1646  */
    break;

  case 337:
#line 1231 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4434 "y.tab.c" /* yacc.c:1646  */
    break;

  case 338:
#line 1233 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4440 "y.tab.c" /* yacc.c:1646  */
    break;

  case 339:
#line 1235 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4446 "y.tab.c" /* yacc.c:1646  */
    break;

  case 340:
#line 1237 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4452 "y.tab.c" /* yacc.c:1646  */
    break;

  case 341:
#line 1239 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4458 "y.tab.c" /* yacc.c:1646  */
    break;

  case 342:
#line 1241 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4464 "y.tab.c" /* yacc.c:1646  */
    break;

  case 343:
#line 1243 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4470 "y.tab.c" /* yacc.c:1646  */
    break;

  case 344:
#line 1245 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4476 "y.tab.c" /* yacc.c:1646  */
    break;

  case 345:
#line 1247 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4482 "y.tab.c" /* yacc.c:1646  */
    break;

  case 346:
#line 1249 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4488 "y.tab.c" /* yacc.c:1646  */
    break;

  case 347:
#line 1251 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4494 "y.tab.c" /* yacc.c:1646  */
    break;

  case 348:
#line 1253 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4500 "y.tab.c" /* yacc.c:1646  */
    break;

  case 349:
#line 1257 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new AtomicConstruct((yyvsp[-1].strval), (yyvsp[-3].strval), (yylsp[-2]).first_line); }
#line 4506 "y.tab.c" /* yacc.c:1646  */
    break;

  case 350:
#line 1259 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new OverlapConstruct((yyvsp[-1].olist)); }
#line 4512 "y.tab.c" /* yacc.c:1646  */
    break;

  case 351:
#line 1261 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = (yyvsp[0].when); }
#line 4518 "y.tab.c" /* yacc.c:1646  */
    break;

  case 352:
#line 1263 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new CaseConstruct((yyvsp[-1].clist)); }
#line 4524 "y.tab.c" /* yacc.c:1646  */
    break;

  case 353:
#line 1265 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForConstruct((yyvsp[-8].intexpr), (yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4530 "y.tab.c" /* yacc.c:1646  */
    break;

  case 354:
#line 1267 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForConstruct((yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4536 "y.tab.c" /* yacc.c:1646  */
    break;

  case 355:
#line 1269 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-9].strval)), (yyvsp[-6].intexpr),
		             (yyvsp[-4].intexpr), (yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4543 "y.tab.c" /* yacc.c:1646  */
    break;

  case 356:
#line 1272 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-11].strval)), (yyvsp[-8].intexpr),
		             (yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4550 "y.tab.c" /* yacc.c:1646  */
    break;

  case 357:
#line 1275 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new IfConstruct((yyvsp[-3].intexpr), (yyvsp[-1].sc), (yyvsp[0].sc)); }
#line 4556 "y.tab.c" /* yacc.c:1646  */
    break;

  case 358:
#line 1277 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new IfConstruct((yyvsp[-5].intexpr), (yyvsp[-2].slist), (yyvsp[0].sc)); }
#line 4562 "y.tab.c" /* yacc.c:1646  */
    break;

  case 359:
#line 1279 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new WhileConstruct((yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4568 "y.tab.c" /* yacc.c:1646  */
    break;

  case 360:
#line 1281 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new WhileConstruct((yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4574 "y.tab.c" /* yacc.c:1646  */
    break;

  case 361:
#line 1283 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new AtomicConstruct((yyvsp[-1].strval), NULL, (yyloc).first_line); }
#line 4580 "y.tab.c" /* yacc.c:1646  */
    break;

  case 362:
#line 1285 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("unknown SDAG construct or malformed entry method declaration.\n"
		        "You may have forgotten to terminate a previous entry method declaration with a"
		        " semicolon or forgotten to mark a block of sequential SDAG code as 'serial'",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
#line 4592 "y.tab.c" /* yacc.c:1646  */
    break;

  case 363:
#line 1295 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = 0; }
#line 4598 "y.tab.c" /* yacc.c:1646  */
    break;

  case 364:
#line 1297 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ElseConstruct((yyvsp[0].sc)); }
#line 4604 "y.tab.c" /* yacc.c:1646  */
    break;

  case 365:
#line 1299 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ElseConstruct((yyvsp[-1].slist)); }
#line 4610 "y.tab.c" /* yacc.c:1646  */
    break;

  case 366:
#line 1303 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intexpr) = new IntExprConstruct((yyvsp[0].strval)); }
#line 4616 "y.tab.c" /* yacc.c:1646  */
    break;

  case 367:
#line 1307 "xi-grammar.y" /* yacc.c:1646  */
    { in_int_expr = 0; (yyval.intval) = 0; }
#line 4622 "y.tab.c" /* yacc.c:1646  */
    break;

  case 368:
#line 1311 "xi-grammar.y" /* yacc.c:1646  */
    { in_int_expr = 1; (yyval.intval) = 0; }
#line 4628 "y.tab.c" /* yacc.c:1646  */
    break;

  case 369:
#line 1315 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[-1].strval), (yyvsp[0].plist), 0, 0, 0, (yyloc).first_line, (yyloc).last_line);
		}
#line 4636 "y.tab.c" /* yacc.c:1646  */
    break;

  case 370:
#line 1319 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[-4].strval), (yyvsp[0].plist), 0, 0, (yyvsp[-2].strval), (yyloc).first_line, (yyloc).last_line);
		}
#line 4644 "y.tab.c" /* yacc.c:1646  */
    break;

  case 371:
#line 1325 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.entrylist) = new EntryList((yyvsp[0].entry)); }
#line 4650 "y.tab.c" /* yacc.c:1646  */
    break;

  case 372:
#line 1327 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.entrylist) = new EntryList((yyvsp[-2].entry),(yyvsp[0].entrylist)); }
#line 4656 "y.tab.c" /* yacc.c:1646  */
    break;

  case 373:
#line 1331 "xi-grammar.y" /* yacc.c:1646  */
    { in_bracket=1; }
#line 4662 "y.tab.c" /* yacc.c:1646  */
    break;

  case 374:
#line 1334 "xi-grammar.y" /* yacc.c:1646  */
    { in_bracket=0; }
#line 4668 "y.tab.c" /* yacc.c:1646  */
    break;

  case 375:
#line 1338 "xi-grammar.y" /* yacc.c:1646  */
    { if (!macroDefined((yyvsp[0].strval), 1)) in_comment = 1; }
#line 4674 "y.tab.c" /* yacc.c:1646  */
    break;

  case 376:
#line 1342 "xi-grammar.y" /* yacc.c:1646  */
    { if (!macroDefined((yyvsp[0].strval), 0)) in_comment = 1; }
#line 4680 "y.tab.c" /* yacc.c:1646  */
    break;


#line 4684 "y.tab.c" /* yacc.c:1646  */
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
#line 1345 "xi-grammar.y" /* yacc.c:1906  */


void yyerror(const char *msg) { }
