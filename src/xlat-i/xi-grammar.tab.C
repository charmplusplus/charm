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
bool firstDeviceRdma = true;

bool enable_warnings = true;

extern int macroDefined(const char *str, int istrue);
extern const char *python_doc;
extern char *fname;
void splitScopedName(const char* name, const char** scope, const char** basename);
void ReservedWord(int token, int fCol, int lCol);
}

#line 116 "y.tab.c" /* yacc.c:339  */

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
    NOCOPYDEVICE = 297,
    PACKED = 298,
    VARSIZE = 299,
    ENTRY = 300,
    FOR = 301,
    FORALL = 302,
    WHILE = 303,
    WHEN = 304,
    OVERLAP = 305,
    SERIAL = 306,
    IF = 307,
    ELSE = 308,
    PYTHON = 309,
    LOCAL = 310,
    NAMESPACE = 311,
    USING = 312,
    IDENT = 313,
    NUMBER = 314,
    LITERAL = 315,
    CPROGRAM = 316,
    HASHIF = 317,
    HASHIFDEF = 318,
    INT = 319,
    LONG = 320,
    SHORT = 321,
    CHAR = 322,
    FLOAT = 323,
    DOUBLE = 324,
    UNSIGNED = 325,
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
#define NOCOPYDEVICE 297
#define PACKED 298
#define VARSIZE 299
#define ENTRY 300
#define FOR 301
#define FORALL 302
#define WHILE 303
#define WHEN 304
#define OVERLAP 305
#define SERIAL 306
#define IF 307
#define ELSE 308
#define PYTHON 309
#define LOCAL 310
#define NAMESPACE 311
#define USING 312
#define IDENT 313
#define NUMBER 314
#define LITERAL 315
#define CPROGRAM 316
#define HASHIF 317
#define HASHIFDEF 318
#define INT 319
#define LONG 320
#define SHORT 321
#define CHAR 322
#define FLOAT 323
#define DOUBLE 324
#define UNSIGNED 325
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
#line 54 "xi-grammar.y" /* yacc.c:355  */

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

#line 358 "y.tab.c" /* yacc.c:355  */
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

#line 389 "y.tab.c" /* yacc.c:358  */

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
#define YYFINAL  59
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   1523

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  95
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  119
/* YYNRULES -- Number of rules.  */
#define YYNRULES  400
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  793

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
       0,   203,   203,   208,   211,   216,   217,   221,   223,   228,
     229,   234,   236,   237,   238,   240,   241,   242,   244,   245,
     246,   247,   248,   252,   253,   254,   255,   256,   257,   258,
     259,   260,   261,   262,   263,   264,   265,   266,   267,   268,
     269,   270,   271,   272,   275,   276,   277,   278,   279,   280,
     281,   282,   283,   284,   285,   287,   289,   290,   293,   294,
     295,   296,   300,   302,   308,   315,   319,   326,   328,   333,
     334,   338,   340,   342,   344,   346,   360,   362,   364,   366,
     372,   374,   376,   378,   380,   382,   384,   386,   388,   390,
     398,   400,   402,   406,   408,   413,   414,   419,   420,   424,
     426,   428,   430,   432,   434,   436,   438,   440,   442,   444,
     446,   448,   450,   452,   454,   456,   458,   460,   462,   466,
     467,   472,   480,   482,   486,   490,   492,   496,   500,   502,
     504,   506,   508,   510,   514,   516,   518,   520,   522,   526,
     528,   530,   532,   534,   536,   540,   542,   544,   546,   548,
     550,   554,   558,   563,   564,   568,   572,   577,   578,   583,
     584,   594,   596,   600,   602,   607,   608,   612,   614,   619,
     620,   624,   629,   630,   634,   636,   640,   642,   647,   648,
     652,   653,   656,   660,   662,   666,   668,   670,   675,   676,
     680,   682,   686,   688,   692,   696,   700,   706,   710,   712,
     716,   718,   722,   726,   730,   734,   736,   741,   742,   747,
     748,   750,   752,   761,   763,   765,   767,   769,   771,   775,
     777,   781,   785,   787,   789,   791,   793,   797,   799,   804,
     811,   815,   817,   819,   820,   822,   824,   826,   830,   832,
     834,   840,   846,   855,   857,   859,   865,   873,   875,   878,
     882,   886,   888,   893,   895,   903,   905,   907,   909,   911,
     913,   915,   917,   919,   921,   923,   926,   937,   955,   973,
     975,   979,   984,   985,   987,   994,   998,   999,  1003,  1004,
    1005,  1006,  1009,  1011,  1013,  1015,  1017,  1019,  1021,  1023,
    1025,  1027,  1029,  1031,  1033,  1035,  1037,  1039,  1041,  1043,
    1047,  1056,  1058,  1060,  1065,  1066,  1068,  1077,  1078,  1080,
    1086,  1092,  1098,  1106,  1113,  1121,  1128,  1130,  1132,  1134,
    1139,  1149,  1159,  1171,  1172,  1173,  1176,  1177,  1178,  1179,
    1186,  1192,  1201,  1208,  1214,  1220,  1228,  1230,  1234,  1236,
    1240,  1242,  1246,  1248,  1253,  1254,  1258,  1260,  1262,  1266,
    1268,  1272,  1274,  1278,  1280,  1282,  1290,  1293,  1296,  1298,
    1300,  1304,  1306,  1308,  1310,  1312,  1314,  1316,  1318,  1320,
    1322,  1324,  1326,  1330,  1332,  1334,  1336,  1338,  1340,  1342,
    1345,  1348,  1350,  1352,  1354,  1356,  1358,  1369,  1370,  1372,
    1376,  1380,  1384,  1388,  1394,  1402,  1404,  1408,  1411,  1415,
    1419
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
  "NOCOPYDEVICE", "PACKED", "VARSIZE", "ENTRY", "FOR", "FORALL", "WHILE",
  "WHEN", "OVERLAP", "SERIAL", "IF", "ELSE", "PYTHON", "LOCAL",
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
     325,   326,   327,   328,   329,   330,   331,   332,   333,    59,
      58,   123,   125,    44,    60,    62,    42,    40,    41,    38,
      46,    91,    93,    61,    45
};
# endif

#define YYPACT_NINF -601

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-601)))

#define YYTABLE_NINF -352

#define yytable_value_is_error(Yytable_value) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
     197,  1300,  1300,    61,  -601,   197,  -601,  -601,  -601,  -601,
    -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,
    -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,
    -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,
    -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,
    -601,  -601,  -601,  -601,  -601,  -601,  -601,   162,   162,  -601,
    -601,  -601,   913,    -3,  -601,  -601,  -601,    62,  1300,   222,
    1300,  1300,   207,  1054,    50,   992,   913,  -601,  -601,  -601,
    -601,   206,    70,    96,  -601,    91,  -601,  -601,  -601,    -3,
     -24,  1343,   149,   149,     6,   -18,   118,   118,   118,   118,
     121,   155,  1300,   134,   167,   913,  -601,  -601,  -601,  -601,
    -601,  -601,  -601,  -601,   529,  -601,  -601,  -601,  -601,   175,
    -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,
    -601,    -3,  -601,  -601,  -601,  1188,  1427,   913,    91,   185,
      77,   -24,   186,   329,  -601,  1445,  -601,   198,   217,  -601,
    -601,  -601,   324,    96,   168,  -601,  -601,   209,   216,   234,
    -601,    51,    96,  -601,    96,    96,   246,    96,   250,  -601,
      23,  1300,  1300,  1300,  1300,   103,   252,   270,   176,  1300,
    -601,  -601,  -601,   655,   301,   118,   118,   118,   118,   252,
     155,  -601,  -601,  -601,  -601,  -601,    -3,  -601,  -601,  -601,
    -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,
    -601,  -601,  -601,   341,  -601,  -601,  -601,   311,   200,  1427,
     209,   216,   234,    34,  -601,   -18,   327,    64,   -24,   351,
     -24,   325,  -601,   175,   328,    -9,  -601,   360,  -601,  -601,
    -601,   287,   361,  -601,   168,  1412,  -601,  -601,  -601,  -601,
    -601,   344,   296,   342,   -33,   -30,   111,   339,   336,   -18,
    -601,  -601,   347,   359,   364,   363,   363,   363,   363,  -601,
    1300,   352,   362,   378,   107,  1300,   420,  1300,  -601,  -601,
     388,   398,   403,   825,    -6,    89,  1300,   402,   404,   175,
    1300,  1300,  1300,  1300,  1300,  1300,  -601,  -601,  -601,  1188,
    1300,   451,  -601,   332,   406,  1300,  -601,  -601,  -601,   422,
     424,   421,   413,   -24,    -3,    96,  -601,  -601,   444,  -601,
    -601,  -601,  -601,   429,  -601,   426,  -601,  1300,   423,   425,
     435,  -601,   437,  -601,   -24,   149,  1412,   149,   149,  1412,
     149,  -601,  -601,    23,  -601,   -18,   224,   224,   224,   224,
     434,  -601,   420,  -601,   363,   363,  -601,   176,    22,   439,
     440,   157,   442,   145,  -601,   443,   655,  -601,  -601,   363,
     363,   363,   363,   363,   299,  -601,   458,   431,   461,   460,
     464,   465,   364,   -24,   351,   -24,   -24,  -601,   -33,  -601,
    1412,  -601,   463,   466,   467,  -601,  -601,   469,  -601,   470,
     474,   475,    96,   480,   481,  -601,   485,  -601,   428,    -3,
    -601,  -601,  -601,  -601,  -601,  -601,   224,   224,  -601,  -601,
    -601,  1445,    26,   488,   482,  1445,  -601,  -601,   494,  -601,
    -601,  -601,  -601,  -601,   224,   224,   224,   224,   224,   554,
      -3,   526,  1300,   503,   497,   498,  -601,   502,  -601,  -601,
    -601,  -601,  -601,  -601,   505,   500,  -601,  -601,  -601,  -601,
     508,  -601,    55,   509,  -601,   -18,  -601,   744,   553,   517,
     175,   428,  -601,  -601,  -601,  -601,  1300,  -601,  -601,  1300,
    -601,   544,  -601,  -601,  -601,  -601,  -601,   521,  -601,  -601,
    1188,   514,  -601,  1376,  -601,  1391,  -601,   149,   149,   149,
    -601,  1132,  1074,  -601,   175,    -3,  -601,   515,   440,   440,
     175,  -601,  -601,  1445,  1445,  1445,  -601,  1300,   -24,   522,
     524,   525,   527,   528,   531,   518,   534,   502,  1300,  -601,
     533,   175,  -601,  -601,    -3,  1300,   -24,   -24,   -24,    19,
     535,  1391,  -601,  -601,  -601,  -601,  -601,   579,   530,   502,
    -601,    -3,   532,   536,   537,   539,  -601,   306,  -601,  -601,
    -601,  1300,  -601,   546,   543,   546,   560,   541,   566,   546,
     555,   323,    -3,   -24,  -601,  -601,  -601,   617,  -601,  -601,
    -601,  -601,  -601,    91,  -601,   502,  -601,   -24,   580,   -24,
     240,   563,   593,   607,  -601,   567,   -24,   417,   565,   260,
     186,   556,   530,   559,  -601,   581,   569,   574,  -601,   -24,
     560,   410,  -601,   582,   471,   -24,   574,   546,   575,   546,
     584,   566,   546,   586,   -24,   587,   417,  -601,   175,  -601,
     175,   613,  -601,   290,   567,   -24,   546,  -601,   631,   469,
    -601,  -601,   603,  -601,  -601,   186,   651,   -24,   628,   -24,
     607,   567,   -24,   417,   186,  -601,  -601,  -601,  -601,  -601,
    -601,  -601,  -601,  -601,  1300,   595,   604,   597,   -24,   611,
     -24,   323,  -601,   502,  -601,   175,   323,   638,   625,   600,
     574,   614,   -24,   574,   623,   175,   626,  1445,  1326,  -601,
     186,   -24,   629,   632,  -601,  -601,   633,   890,  -601,   -24,
     546,   897,  -601,   186,   904,  -601,  -601,  1300,  1300,   -24,
     634,  -601,  1300,   574,   -24,  -601,   638,   323,  -601,   637,
     -24,   323,  -601,   175,   323,   638,  -601,   245,    86,   635,
    1300,   175,   911,   643,  -601,   648,   -24,   654,   649,  -601,
     653,  -601,  -601,  1300,  1300,  1225,   652,  1300,  -601,   259,
      -3,   323,  -601,   -24,  -601,   574,   -24,  -601,   638,   412,
    -601,   644,   231,  1300,   392,  -601,   656,   574,   963,   657,
    -601,  -601,  -601,  -601,  -601,  -601,  -601,   971,   323,  -601,
     -24,   323,  -601,   659,   574,   668,  -601,   978,  -601,   323,
    -601,   672,  -601
};

  /* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
     Performed when YYTABLE does not specify something else to do.  Zero
     means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       3,     0,     0,     0,     2,     3,    12,    13,    14,    15,
      16,    17,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    31,    32,    36,    37,    38,
      39,    40,    41,    42,    43,    33,    34,    35,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
      56,    11,    57,    58,    59,    60,    61,     0,     0,     1,
       4,     7,     0,    67,    65,    66,    89,     6,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    88,    86,    87,
       8,     0,     0,     0,    62,    72,   399,   400,   314,   270,
     307,     0,   157,   157,   157,     0,   165,   165,   165,   165,
       0,   159,     0,     0,     0,     0,    80,   231,   232,    74,
      81,    82,    83,    84,     0,    85,    73,   234,   233,     9,
     265,   257,   258,   259,   260,   261,   263,   264,   262,   255,
     256,    78,    79,    70,   274,     0,     0,     0,    71,     0,
     308,   307,     0,     0,   118,     0,    99,   100,   102,   104,
     115,   116,     0,     0,    97,   122,   123,   128,   129,   130,
     131,   150,     0,   158,     0,     0,     0,     0,   247,   235,
       0,     0,     0,     0,     0,     0,     0,   172,     0,     0,
     237,   249,   236,     0,     0,   165,   165,   165,   165,     0,
     159,   222,   223,   224,   225,   226,    10,    68,   300,   282,
     283,   284,   285,   286,   292,   293,   294,   299,   287,   288,
     289,   290,   291,   169,   295,   297,   298,     0,   278,     0,
     134,   135,   136,   144,   271,     0,     0,     0,   307,   304,
     307,     0,   315,     0,     0,   132,   101,   113,   117,   103,
     105,   106,   110,   112,    97,    95,   120,   124,   125,   126,
     133,     0,   149,     0,   153,   241,   238,     0,   243,     0,
     176,   177,     0,   167,    97,   188,   188,   188,   188,   171,
       0,     0,   174,     0,     0,     0,     0,     0,   163,   164,
       0,   161,   185,     0,     0,   131,     0,   219,     0,     9,
       0,     0,     0,     0,     0,     0,   170,   296,   273,     0,
       0,   137,   138,   143,     0,     0,    77,    64,    63,     0,
     305,     0,     0,   307,   269,     0,   114,   107,   108,   111,
     121,    91,    92,    93,    96,     0,    90,     0,   148,     0,
       0,   397,   153,   155,   307,   157,     0,   157,   157,     0,
     157,   248,   166,     0,   119,     0,     0,     0,     0,     0,
       0,   197,     0,   173,   188,   188,   160,     0,   178,     0,
     207,    62,     0,     0,   217,   209,     0,   221,    76,   188,
     188,   188,   188,   188,     0,   280,     0,   276,     0,   142,
       0,     0,    97,   307,   304,   307,   307,   312,   153,   109,
       0,    98,     0,     0,     0,   147,   154,     0,   151,     0,
       0,     0,     0,     0,     0,   168,   190,   189,     0,   227,
     192,   193,   194,   195,   196,   175,     0,     0,   162,   179,
     186,     0,   178,     0,     0,     0,   215,   216,     0,   210,
     211,   212,   218,   220,     0,     0,     0,     0,     0,   178,
     205,     0,     0,   279,     0,     0,   141,     0,   310,   306,
     311,   309,   156,    94,     0,     0,   146,   398,   152,   242,
       0,   239,     0,     0,   244,     0,   254,     0,     0,     0,
       0,     0,   250,   251,   198,   199,     0,   184,   187,     0,
     208,     0,   200,   201,   202,   203,   204,     0,   275,   277,
       0,     0,   140,     0,    75,     0,   145,   157,   157,   157,
     191,     0,     0,   252,     9,   253,   230,   180,   207,   207,
       0,   281,   139,     0,     0,     0,   341,   316,   307,   336,
       0,     0,     0,     0,     0,     0,    62,     0,     0,   228,
       0,     0,   213,   214,   206,     0,   307,   307,   307,   178,
       0,     0,   340,   127,   240,   246,   245,     0,     0,     0,
     181,   182,     0,     0,     0,     0,   313,     0,   317,   319,
     337,     0,   386,     0,     0,     0,     0,     0,   357,     0,
       0,     0,   346,   307,   267,   375,   347,   344,   320,   321,
     322,   302,   301,   303,   318,     0,   392,   307,     0,   307,
       0,   395,     0,     0,   356,     0,   307,     0,     0,     0,
       0,     0,     0,     0,   390,     0,     0,     0,   393,   307,
       0,     0,   359,     0,     0,   307,     0,     0,     0,     0,
       0,   357,     0,     0,   307,     0,   353,   355,     9,   350,
       9,     0,   266,     0,     0,   307,     0,   391,     0,     0,
     396,   358,     0,   374,   352,     0,     0,   307,     0,   307,
       0,     0,   307,     0,     0,   376,   354,   348,   385,   345,
     323,   324,   325,   343,     0,     0,   338,     0,   307,     0,
     307,     0,   383,     0,   360,     9,     0,   387,     0,     0,
       0,     0,   307,     0,     0,     9,     0,     0,     0,   342,
       0,   307,     0,     0,   394,   373,     0,     0,   381,   307,
       0,     0,   362,     0,     0,   363,   372,     0,     0,   307,
       0,   339,     0,     0,   307,   384,   387,     0,   388,     0,
     307,     0,   370,     9,     0,   387,   326,     0,     0,     0,
       0,     0,     0,     0,   382,     0,   307,     0,     0,   361,
       0,   368,   334,     0,     0,     0,     0,     0,   332,     0,
     268,     0,   378,   307,   389,     0,   307,   371,   387,     0,
     328,     0,     0,     0,     0,   335,     0,     0,     0,     0,
     369,   331,   330,   329,   327,   333,   377,     0,     0,   365,
     307,     0,   379,     0,     0,     0,   364,     0,   380,     0,
     366,     0,   367
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -601,  -601,   712,  -601,   -55,  -283,    -1,   -62,   666,   702,
     -17,  -601,  -601,  -601,  -221,  -601,  -219,  -601,  -141,   -86,
    -125,  -126,  -121,  -164,   616,   557,  -601,   -87,  -601,  -601,
    -267,  -601,  -601,   -80,   608,   446,  -601,   128,   457,  -601,
    -601,   627,   453,  -601,   267,  -601,  -601,  -354,  -601,  -140,
     358,  -601,  -601,  -601,   -29,  -601,  -601,  -601,  -601,  -601,
    -601,  -319,   452,  -601,   441,   743,  -601,  -199,   353,   752,
    -601,  -601,   568,  -601,  -601,  -601,  -601,   373,  -601,   340,
     376,  -601,   384,  -291,  -601,  -601,   447,   -85,  -491,   -60,
    -559,  -601,  -601,  -537,  -601,  -601,  -442,   169,  -498,  -601,
    -601,   261,  -565,   214,  -578,   258,  -571,  -601,  -521,  -579,
    -561,  -600,  -503,  -601,   271,   294,   247,  -601,  -601
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     3,     4,    73,   409,   197,   264,   154,     5,    64,
      74,    75,    76,   323,   324,   325,   246,   155,   265,   156,
     157,   158,   159,   160,   161,   223,   224,   326,   397,   332,
     333,   107,   108,   164,   179,   280,   281,   171,   262,   297,
     272,   176,   273,   263,   421,   531,   422,   423,   109,   346,
     407,   110,   111,   112,   177,   113,   191,   192,   193,   194,
     195,   426,   364,   287,   288,   468,   115,   410,   469,   470,
     117,   118,   169,   182,   471,   472,   132,   473,    77,   225,
     136,   377,   378,   217,   218,   584,   311,   604,   518,   573,
     233,   519,   665,   727,   710,   666,   520,   667,   494,   634,
     602,   574,   598,   613,   625,   595,   575,   627,   599,   698,
     605,   638,   587,   591,   592,   334,   458,    78,    79
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[] =
{
      57,    58,    63,    63,   162,   142,   368,    85,   375,   168,
     221,   220,    90,   165,   167,   222,   646,    89,   234,   285,
     131,   138,   536,   537,   538,   320,   626,   576,   607,   548,
     250,   266,   267,   268,   629,   616,   419,   140,   282,   419,
      84,   630,   427,   419,   163,   344,   642,   139,   656,   644,
     139,   577,   361,   521,   260,   626,   231,   335,   331,   133,
     153,    59,   589,   141,   196,   396,   596,    82,   477,    86,
      87,   612,   614,   302,   669,   684,    80,   261,   251,   307,
     701,   576,   626,   704,   362,   487,   675,   603,   184,   271,
     250,   244,   608,   221,   220,   685,   286,   166,   222,   560,
     254,   180,   255,   256,   420,   258,   693,    81,  -183,   692,
     556,   696,   557,   732,   647,   400,   649,   672,   403,   652,
     226,   452,   308,   303,   304,   677,   347,   348,   349,   614,
     713,   712,   119,   670,   354,   139,   355,   734,   251,   305,
     252,   253,   498,   309,   723,   312,   741,   275,   411,   412,
     413,   137,   735,   733,    84,   768,   738,   269,   228,   740,
     294,    84,   270,   447,   229,    84,   270,   777,   230,   453,
     747,   139,  -209,   168,  -209,   694,   718,   556,   314,   770,
     722,   153,   363,   725,   787,   153,   766,   163,   271,   532,
     533,   139,   767,   749,   181,   336,   709,   720,   337,   511,
       1,     2,   285,    84,   429,   430,   759,   134,   762,   170,
     764,   752,   175,   783,   416,   417,   785,   474,   475,   278,
     279,   529,   244,   153,   791,   172,   173,   174,   387,   434,
     435,   436,   437,   438,   196,   482,   483,   484,   485,   486,
    -207,    61,  -207,    62,  -272,  -272,   178,   779,   139,   398,
     425,   183,   245,   388,    61,   399,   782,   401,   402,   406,
     404,   562,   236,   237,  -272,   227,   790,   238,   232,   350,
    -272,  -272,  -272,  -272,  -272,  -272,  -272,   431,    83,   286,
      84,   239,   360,   299,  -272,   365,    61,   300,    88,   369,
     370,   371,   372,   373,   374,   247,   660,   135,   448,   376,
     450,   451,   248,    61,   382,   408,   563,   564,   565,   566,
     567,   568,   569,   290,   291,   292,   293,   257,   743,   440,
     249,   744,   745,   773,   562,   746,   392,   493,   144,   145,
     742,   331,   743,   259,   476,   744,   745,   570,   480,   746,
     462,    88,  -349,   274,   765,   657,   743,   658,    84,   744,
     745,   317,   318,   746,   146,   147,   148,   149,   150,   151,
     152,   276,   661,   662,    84,   581,   582,   144,   153,   563,
     564,   565,   566,   567,   568,   569,   221,   220,    61,   406,
     439,   222,   663,   289,  -314,   328,   329,    84,   240,   241,
     242,   243,   695,   146,   147,   148,   149,   150,   151,   152,
     570,   296,   706,   298,    88,  -314,   517,   153,   517,   306,
    -314,   562,   310,   313,   315,   505,   139,   522,   523,   524,
     339,   379,   380,   340,   316,   319,   535,   535,   535,   466,
     327,   338,   330,   540,    91,    92,    93,    94,    95,   342,
     739,   376,   343,   345,   351,   352,   102,   103,   245,   196,
     104,   553,   554,   555,   517,   534,   563,   564,   565,   566,
     567,   568,   569,   617,   618,   619,   566,   620,   621,   622,
     353,  -314,   562,   467,   269,   507,   551,   775,   508,   743,
     356,   357,   744,   745,   358,   366,   746,   570,   600,   367,
     302,    88,   641,   572,   623,   583,   381,  -314,    88,   743,
     771,   527,   744,   745,   383,   386,   746,   384,   389,   385,
    -229,   391,   390,   393,   442,   394,   539,   563,   564,   565,
     566,   567,   568,   569,   639,   395,   414,   549,   331,   424,
     645,   562,   428,   425,   552,   615,   363,   624,   441,   654,
     185,   186,   187,   188,   189,   190,   664,   572,   570,   443,
     444,   454,    88,  -351,   445,   446,   455,   456,   459,   460,
     585,   457,   678,   461,   680,   463,   624,   683,   465,   464,
     478,   419,   479,   196,   668,   196,   563,   564,   565,   566,
     567,   568,   569,   690,   481,   488,   490,   491,   492,   493,
     496,   682,   495,   624,   562,   497,   499,   703,   467,   504,
     708,   664,   509,   510,   512,   541,   530,   570,   562,    61,
     547,   571,   542,   543,   719,   544,   545,   561,   590,   546,
     196,   -11,   593,   556,   729,   550,   594,   559,   578,   579,
     196,   580,   562,   586,   588,   737,   597,   601,   606,   563,
     564,   565,   566,   567,   568,   569,   610,   628,    88,   631,
     633,   755,   562,   563,   564,   565,   566,   567,   568,   569,
     635,   636,   637,   686,   643,   650,   648,   653,   196,   655,
     570,   769,   659,   283,   611,   687,   750,   563,   564,   565,
     566,   567,   568,   569,   570,   674,   679,   688,    88,   689,
     691,   697,   700,   144,   145,   784,   702,   563,   564,   565,
     566,   567,   568,   569,   699,   705,   726,   728,   570,   714,
     707,   731,   671,    84,   715,   716,   736,    60,   730,   146,
     147,   148,   149,   150,   151,   152,   753,   748,   570,   726,
     754,   757,   676,   284,   756,   758,   772,   763,   776,   106,
     780,   786,   726,   760,   726,   134,   726,  -272,  -272,  -272,
     788,  -272,  -272,  -272,   792,  -272,  -272,  -272,  -272,  -272,
      65,   235,   774,  -272,  -272,  -272,  -272,  -272,  -272,  -272,
    -272,  -272,  -272,  -272,  -272,  -272,   301,  -272,  -272,  -272,
    -272,  -272,  -272,  -272,  -272,  -272,  -272,  -272,  -272,  -272,
    -272,  -272,  -272,  -272,  -272,  -272,  -272,  -272,   295,  -272,
     405,  -272,  -272,   418,   277,   415,   558,   433,  -272,  -272,
    -272,  -272,  -272,  -272,  -272,  -272,   114,   432,  -272,  -272,
    -272,  -272,  -272,   500,   506,   116,   489,   341,     6,     7,
       8,   449,     9,    10,    11,   501,    12,    13,    14,    15,
      16,   503,   528,   502,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,   711,    30,    31,
      32,    33,    34,   632,   681,    35,    36,    37,    38,    39,
      40,    41,    42,    43,    44,    45,    46,    47,    48,   651,
      49,   640,    50,    51,   609,     0,   673,     0,     0,     0,
       0,   562,     0,     0,     0,     0,    52,     0,   562,    53,
      54,    55,    56,     0,     0,   562,     0,     0,     0,     0,
       0,     0,   562,     0,    66,   359,    -5,    -5,    67,    -5,
      -5,    -5,    -5,    -5,    -5,    -5,    -5,    -5,    -5,    -5,
       0,    -5,    -5,     0,     0,    -5,   563,   564,   565,   566,
     567,   568,   569,   563,   564,   565,   566,   567,   568,   569,
     563,   564,   565,   566,   567,   568,   569,   563,   564,   565,
     566,   567,   568,   569,   562,     0,     0,   570,     0,    68,
      69,   717,   562,     0,   570,    70,    71,     0,   721,   562,
       0,   570,     0,     0,     0,   724,     0,    72,   570,     0,
       0,     0,   751,     0,    -5,   -69,     0,     0,   120,   121,
     122,   123,     0,   124,   125,   126,   127,   128,     0,   563,
     564,   565,   566,   567,   568,   569,     0,   563,   564,   565,
     566,   567,   568,   569,   563,   564,   565,   566,   567,   568,
     569,     0,     0,     0,     0,     0,     0,   129,     0,     0,
     570,     0,     0,     0,   778,     0,     0,     0,   570,     0,
       0,     0,   781,     0,     0,   570,     0,     1,     2,   789,
      91,    92,    93,    94,    95,    96,    97,    98,    99,   100,
     101,    61,   102,   103,   130,     0,   104,     6,     7,     8,
       0,     9,    10,    11,     0,    12,    13,    14,    15,    16,
       0,     0,     0,    17,    18,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,    29,     0,    30,    31,    32,
      33,    34,   144,   219,    35,    36,    37,    38,    39,    40,
      41,    42,    43,    44,    45,    46,    47,    48,     0,    49,
       0,    50,   526,   198,     0,   105,     0,     0,   146,   147,
     148,   149,   150,   151,   152,    52,     0,     0,    53,    54,
      55,    56,   153,   199,     0,   200,   201,   202,   203,   204,
     205,   206,     0,     0,   207,   208,   209,   210,   211,   212,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   213,   214,     0,   198,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   525,     0,     0,     0,   215,   216,   199,
       0,   200,   201,   202,   203,   204,   205,   206,     0,     0,
     207,   208,   209,   210,   211,   212,     0,     0,     6,     7,
       8,     0,     9,    10,    11,     0,    12,    13,    14,    15,
      16,     0,   213,   214,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,     0,    30,    31,
      32,    33,    34,   215,   216,    35,    36,    37,    38,    39,
      40,    41,    42,    43,    44,    45,    46,    47,    48,     0,
      49,     0,    50,    51,   761,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    52,     0,     0,    53,
      54,    55,    56,     6,     7,     8,     0,     9,    10,    11,
       0,    12,    13,    14,    15,    16,     0,     0,     0,    17,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      28,    29,   660,    30,    31,    32,    33,    34,     0,     0,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,     0,    49,     0,    50,    51,   143,
       0,     0,     0,     0,   144,   145,     0,     0,     0,     0,
       0,    52,     0,     0,    53,    54,    55,    56,     0,     0,
       0,   144,   145,     0,    84,     0,     0,     0,     0,     0,
     146,   147,   148,   149,   150,   151,   152,     0,   661,   662,
       0,    84,     0,     0,   153,     0,     0,   146,   147,   148,
     149,   150,   151,   152,   144,   145,   513,   514,   515,     0,
       0,   153,     0,     0,     0,     0,     0,     0,     0,   144,
     145,   513,   514,   515,    84,     0,     0,     0,     0,     0,
     146,   147,   148,   149,   150,   151,   152,     0,     0,    84,
     144,   145,     0,     0,   153,   146,   147,   148,   149,   150,
     151,   152,     0,     0,   516,   144,   219,     0,     0,   153,
      84,   321,   322,     0,     0,     0,   146,   147,   148,   149,
     150,   151,   152,   144,   145,    84,     0,     0,     0,     0,
     153,   146,   147,   148,   149,   150,   151,   152,     0,     0,
       0,     0,     0,    84,     0,   153,     0,     0,     0,   146,
     147,   148,   149,   150,   151,   152,     0,     0,     0,     0,
       0,     0,     0,   153
};

static const yytype_int16 yycheck[] =
{
       1,     2,    57,    58,    91,    90,   289,    69,   299,    95,
     136,   136,    72,    93,    94,   136,   616,    72,   143,   183,
      75,    83,   513,   514,   515,   244,   597,   548,   589,   527,
      39,   172,   173,   174,   599,   596,    17,    61,   179,    17,
      58,   600,   361,    17,    38,   264,   611,    80,   626,   614,
      80,   549,    58,   495,    31,   626,   141,    87,    91,    76,
      78,     0,   565,    87,   119,   332,   569,    68,   422,    70,
      71,   592,   593,    39,   635,   653,    79,    54,    87,    15,
     680,   602,   653,   683,    90,   439,   645,   585,   105,   175,
      39,   153,   590,   219,   219,   654,   183,    91,   219,   541,
     162,   102,   164,   165,    82,   167,   671,    45,    82,   670,
      91,   676,    93,   713,   617,   336,   619,   638,   339,   622,
     137,   388,    58,    89,    90,   646,   266,   267,   268,   650,
     691,   690,    82,   636,   275,    80,   277,   716,    87,   225,
      89,    90,    87,   228,   703,   230,   725,   176,   347,   348,
     349,    81,   717,   714,    58,   755,   721,    54,    81,   724,
     189,    58,    59,   382,    87,    58,    59,   767,    91,   390,
      84,    80,    83,   259,    85,   673,   697,    91,   233,   758,
     701,    78,    93,   704,   784,    78,   751,    38,   274,   508,
     509,    80,   753,   730,    60,    84,   687,   700,    87,   490,
       3,     4,   366,    58,    59,    60,   743,     1,   745,    91,
     747,   732,    91,   778,   354,   355,   781,   416,   417,    43,
      44,   504,   284,    78,   789,    97,    98,    99,   313,   369,
     370,   371,   372,   373,   289,   434,   435,   436,   437,   438,
      83,    79,    85,    81,    38,    39,    91,   768,    80,   334,
      93,    84,    84,   315,    79,   335,   777,   337,   338,   345,
     340,     1,    64,    65,    58,    80,   787,    69,    82,   270,
      64,    65,    66,    67,    68,    69,    70,   363,    56,   366,
      58,    64,   283,    83,    78,   286,    79,    87,    81,   290,
     291,   292,   293,   294,   295,    86,     6,    91,   383,   300,
     385,   386,    86,    79,   305,    81,    46,    47,    48,    49,
      50,    51,    52,   185,   186,   187,   188,    71,    87,   374,
      86,    90,    91,    92,     1,    94,   327,    87,    38,    39,
      85,    91,    87,    83,   421,    90,    91,    77,   425,    94,
     402,    81,    82,    91,    85,   628,    87,   630,    58,    90,
      91,    64,    65,    94,    64,    65,    66,    67,    68,    69,
      70,    91,    72,    73,    58,    59,    60,    38,    78,    46,
      47,    48,    49,    50,    51,    52,   502,   502,    79,   465,
      81,   502,    92,    82,    61,    89,    90,    58,    64,    65,
      66,    67,   675,    64,    65,    66,    67,    68,    69,    70,
      77,    60,   685,    92,    81,    82,   493,    78,   495,    82,
      87,     1,    61,    88,    86,   470,    80,   497,   498,   499,
      84,    89,    90,    87,    64,    64,   513,   514,   515,     1,
      86,    92,    90,   518,     6,     7,     8,     9,    10,    92,
     723,   442,    83,    80,    92,    83,    18,    19,    84,   504,
      22,   536,   537,   538,   541,   510,    46,    47,    48,    49,
      50,    51,    52,    46,    47,    48,    49,    50,    51,    52,
      92,    61,     1,    45,    54,   476,   531,    85,   479,    87,
      92,    83,    90,    91,    81,    83,    94,    77,   573,    85,
      39,    81,    82,   548,    77,   557,    90,    87,    81,    87,
      88,   502,    90,    91,    82,    92,    94,    83,    64,    88,
      82,    85,    83,    90,    83,    90,   517,    46,    47,    48,
      49,    50,    51,    52,   609,    90,    92,   528,    91,    90,
     615,     1,    90,    93,   535,   595,    93,   597,    80,   624,
      11,    12,    13,    14,    15,    16,   633,   602,    77,    88,
      90,    88,    81,    82,    90,    90,    90,    90,    88,    85,
     561,    92,   647,    88,   649,    85,   626,   652,    83,    88,
      82,    17,    90,   628,   634,   630,    46,    47,    48,    49,
      50,    51,    52,   668,    90,    59,    83,    90,    90,    87,
      90,   651,    87,   653,     1,    87,    87,   682,    45,    82,
     687,   688,    58,    82,    90,    83,    91,    77,     1,    79,
      92,    81,    88,    88,   699,    88,    88,    38,    58,    88,
     675,    87,    81,    91,   709,    92,    60,    92,    92,    92,
     685,    92,     1,    87,    91,   720,    81,    20,    58,    46,
      47,    48,    49,    50,    51,    52,    83,    82,    81,    93,
      91,   736,     1,    46,    47,    48,    49,    50,    51,    52,
      79,    92,    88,   664,    82,    81,    91,    81,   723,    82,
      77,   756,    59,    18,    81,    80,   731,    46,    47,    48,
      49,    50,    51,    52,    77,    82,    58,    83,    81,    92,
      79,    53,    92,    38,    39,   780,    82,    46,    47,    48,
      49,    50,    51,    52,    79,    82,   707,   708,    77,    80,
      84,   712,    81,    58,    82,    82,    79,     5,    84,    64,
      65,    66,    67,    68,    69,    70,    83,    92,    77,   730,
      82,    82,    81,    78,    80,    82,    92,    85,    82,    73,
      83,    82,   743,   744,   745,     1,   747,     3,     4,     5,
      82,     7,     8,     9,    82,    11,    12,    13,    14,    15,
      58,   145,   763,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    31,   219,    33,    34,    35,
      36,    37,    38,    39,    40,    41,    42,    43,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    53,   190,    55,
     343,    57,    58,   357,   177,   352,   539,   366,    64,    65,
      66,    67,    68,    69,    70,    71,    73,   365,    74,    75,
      76,    77,    78,   465,   471,    73,   442,   259,     3,     4,
       5,   384,     7,     8,     9,    91,    11,    12,    13,    14,
      15,   468,   502,   467,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,   688,    33,    34,
      35,    36,    37,   602,   650,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,   621,
      55,   610,    57,    58,   590,    -1,   639,    -1,    -1,    -1,
      -1,     1,    -1,    -1,    -1,    -1,    71,    -1,     1,    74,
      75,    76,    77,    -1,    -1,     1,    -1,    -1,    -1,    -1,
      -1,    -1,     1,    -1,     1,    90,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      -1,    18,    19,    -1,    -1,    22,    46,    47,    48,    49,
      50,    51,    52,    46,    47,    48,    49,    50,    51,    52,
      46,    47,    48,    49,    50,    51,    52,    46,    47,    48,
      49,    50,    51,    52,     1,    -1,    -1,    77,    -1,    56,
      57,    81,     1,    -1,    77,    62,    63,    -1,    81,     1,
      -1,    77,    -1,    -1,    -1,    81,    -1,    74,    77,    -1,
      -1,    -1,    81,    -1,    81,    82,    -1,    -1,     6,     7,
       8,     9,    -1,    11,    12,    13,    14,    15,    -1,    46,
      47,    48,    49,    50,    51,    52,    -1,    46,    47,    48,
      49,    50,    51,    52,    46,    47,    48,    49,    50,    51,
      52,    -1,    -1,    -1,    -1,    -1,    -1,    45,    -1,    -1,
      77,    -1,    -1,    -1,    81,    -1,    -1,    -1,    77,    -1,
      -1,    -1,    81,    -1,    -1,    77,    -1,     3,     4,    81,
       6,     7,     8,     9,    10,    11,    12,    13,    14,    15,
      16,    79,    18,    19,    82,    -1,    22,     3,     4,     5,
      -1,     7,     8,     9,    -1,    11,    12,    13,    14,    15,
      -1,    -1,    -1,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    31,    -1,    33,    34,    35,
      36,    37,    38,    39,    40,    41,    42,    43,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    -1,    55,
      -1,    57,    58,     1,    -1,    81,    -1,    -1,    64,    65,
      66,    67,    68,    69,    70,    71,    -1,    -1,    74,    75,
      76,    77,    78,    21,    -1,    23,    24,    25,    26,    27,
      28,    29,    -1,    -1,    32,    33,    34,    35,    36,    37,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    54,    55,    -1,     1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    71,    -1,    -1,    -1,    75,    76,    21,
      -1,    23,    24,    25,    26,    27,    28,    29,    -1,    -1,
      32,    33,    34,    35,    36,    37,    -1,    -1,     3,     4,
       5,    -1,     7,     8,     9,    -1,    11,    12,    13,    14,
      15,    -1,    54,    55,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    -1,    33,    34,
      35,    36,    37,    75,    76,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    -1,
      55,    -1,    57,    58,    59,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    71,    -1,    -1,    74,
      75,    76,    77,     3,     4,     5,    -1,     7,     8,     9,
      -1,    11,    12,    13,    14,    15,    -1,    -1,    -1,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    31,     6,    33,    34,    35,    36,    37,    -1,    -1,
      40,    41,    42,    43,    44,    45,    46,    47,    48,    49,
      50,    51,    52,    53,    -1,    55,    -1,    57,    58,    16,
      -1,    -1,    -1,    -1,    38,    39,    -1,    -1,    -1,    -1,
      -1,    71,    -1,    -1,    74,    75,    76,    77,    -1,    -1,
      -1,    38,    39,    -1,    58,    -1,    -1,    -1,    -1,    -1,
      64,    65,    66,    67,    68,    69,    70,    -1,    72,    73,
      -1,    58,    -1,    -1,    78,    -1,    -1,    64,    65,    66,
      67,    68,    69,    70,    38,    39,    40,    41,    42,    -1,
      -1,    78,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    38,
      39,    40,    41,    42,    58,    -1,    -1,    -1,    -1,    -1,
      64,    65,    66,    67,    68,    69,    70,    -1,    -1,    58,
      38,    39,    -1,    -1,    78,    64,    65,    66,    67,    68,
      69,    70,    -1,    -1,    88,    38,    39,    -1,    -1,    78,
      58,    59,    60,    -1,    -1,    -1,    64,    65,    66,    67,
      68,    69,    70,    38,    39,    58,    -1,    -1,    -1,    -1,
      78,    64,    65,    66,    67,    68,    69,    70,    -1,    -1,
      -1,    -1,    -1,    58,    -1,    78,    -1,    -1,    -1,    64,
      65,    66,    67,    68,    69,    70,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    78
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     3,     4,    96,    97,   103,     3,     4,     5,     7,
       8,     9,    11,    12,    13,    14,    15,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      33,    34,    35,    36,    37,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    55,
      57,    58,    71,    74,    75,    76,    77,   101,   101,     0,
      97,    79,    81,    99,   104,   104,     1,     5,    56,    57,
      62,    63,    74,    98,   105,   106,   107,   173,   212,   213,
      79,    45,   101,    56,    58,   102,   101,   101,    81,    99,
     184,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    18,    19,    22,    81,   103,   126,   127,   143,
     146,   147,   148,   150,   160,   161,   164,   165,   166,    82,
       6,     7,     8,     9,    11,    12,    13,    14,    15,    45,
      82,    99,   171,   105,     1,    91,   175,    81,   102,    80,
      61,    87,   182,    16,    38,    39,    64,    65,    66,    67,
      68,    69,    70,    78,   102,   112,   114,   115,   116,   117,
     118,   119,   122,    38,   128,   128,    91,   128,   114,   167,
      91,   132,   132,   132,   132,    91,   136,   149,    91,   129,
     101,    60,   168,    84,   105,    11,    12,    13,    14,    15,
      16,   151,   152,   153,   154,   155,    99,   100,     1,    21,
      23,    24,    25,    26,    27,    28,    29,    32,    33,    34,
      35,    36,    37,    54,    55,    75,    76,   178,   179,    39,
     115,   116,   117,   120,   121,   174,   105,    80,    81,    87,
      91,   182,    82,   185,   115,   119,    64,    65,    69,    64,
      64,    65,    66,    67,   102,    84,   111,    86,    86,    86,
      39,    87,    89,    90,   102,   102,   102,    71,   102,    83,
      31,    54,   133,   138,   101,   113,   113,   113,   113,    54,
      59,   114,   135,   137,    91,   149,    91,   136,    43,    44,
     130,   131,   113,    18,    78,   118,   122,   158,   159,    82,
     132,   132,   132,   132,   149,   129,    60,   134,    92,    83,
      87,   120,    39,    89,    90,   114,    82,    15,    58,   182,
      61,   181,   182,    88,    99,    86,    64,    64,    65,    64,
     111,    59,    60,   108,   109,   110,   122,    86,    89,    90,
      90,    91,   124,   125,   210,    87,    84,    87,    92,    84,
      87,   167,    92,    83,   111,    80,   144,   144,   144,   144,
     101,    92,    83,    92,   113,   113,    92,    83,    81,    90,
     101,    58,    90,    93,   157,   101,    83,    85,   100,   101,
     101,   101,   101,   101,   101,   178,   101,   176,   177,    89,
      90,    90,   101,    82,    83,    88,    92,   182,   102,    64,
      83,    85,   101,    90,    90,    90,   125,   123,   182,   128,
     109,   128,   128,   109,   128,   133,   114,   145,    81,    99,
     162,   162,   162,   162,    92,   137,   144,   144,   130,    17,
      82,   139,   141,   142,    90,    93,   156,   156,    90,    59,
      60,   114,   157,   159,   144,   144,   144,   144,   144,    81,
      99,    80,    83,    88,    90,    90,    90,   111,   182,   181,
     182,   182,   125,   109,    88,    90,    90,    92,   211,    88,
      85,    88,   102,    85,    88,    83,     1,    45,   160,   163,
     164,   169,   170,   172,   162,   162,   122,   142,    82,    90,
     122,    90,   162,   162,   162,   162,   162,   142,    59,   177,
      83,    90,    90,    87,   193,    87,    90,    87,    87,    87,
     145,    91,   175,   172,    82,    99,   163,   101,   101,    58,
      82,   178,    90,    40,    41,    42,    88,   122,   183,   186,
     191,   191,   128,   128,   128,    71,    58,   101,   174,   100,
      91,   140,   156,   156,    99,   122,   183,   183,   183,   101,
     182,    83,    88,    88,    88,    88,    88,    92,   193,   101,
      92,    99,   101,   182,   182,   182,    91,    93,   139,    92,
     191,    38,     1,    46,    47,    48,    49,    50,    51,    52,
      77,    81,    99,   184,   196,   201,   203,   193,    92,    92,
      92,    59,    60,   102,   180,   101,    87,   207,    91,   207,
      58,   208,   209,    81,    60,   200,   207,    81,   197,   203,
     182,    20,   195,   193,   182,   205,    58,   205,   193,   210,
      83,    81,   203,   198,   203,   184,   205,    46,    47,    48,
      50,    51,    52,    77,   184,   199,   201,   202,    82,   197,
     185,    93,   196,    91,   194,    79,    92,    88,   206,   182,
     209,    82,   197,    82,   197,   182,   206,   207,    91,   207,
      81,   200,   207,    81,   182,    82,   199,   100,   100,    59,
       6,    72,    73,    92,   122,   187,   190,   192,   184,   205,
     207,    81,   203,   211,    82,   185,    81,   203,   182,    58,
     182,   198,   184,   182,   199,   185,   101,    80,    83,    92,
     182,    79,   205,   197,   193,   100,   197,    53,   204,    79,
      92,   206,    82,   182,   206,    82,   100,    84,   122,   183,
     189,   192,   185,   205,    80,    82,    82,    81,   203,   182,
     207,    81,   203,   185,    81,   203,   101,   188,   101,   182,
      84,   101,   206,   205,   204,   197,    79,   182,   197,   100,
     197,   204,    85,    87,    90,    91,    94,    84,    92,   188,
      99,    81,   203,    83,    82,   182,    80,    82,    82,   188,
     101,    59,   188,    85,   188,    85,   197,   205,   206,   182,
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
     101,   101,   102,   102,   102,   103,   103,   104,   104,   105,
     105,   106,   106,   106,   106,   106,   107,   107,   107,   107,
     107,   107,   107,   107,   107,   107,   107,   107,   107,   107,
     108,   108,   108,   109,   109,   110,   110,   111,   111,   112,
     112,   112,   112,   112,   112,   112,   112,   112,   112,   112,
     112,   112,   112,   112,   112,   112,   112,   112,   112,   113,
     114,   114,   115,   115,   116,   117,   117,   118,   119,   119,
     119,   119,   119,   119,   120,   120,   120,   120,   120,   121,
     121,   121,   121,   121,   121,   122,   122,   122,   122,   122,
     122,   123,   124,   125,   125,   126,   127,   128,   128,   129,
     129,   130,   130,   131,   131,   132,   132,   133,   133,   134,
     134,   135,   136,   136,   137,   137,   138,   138,   139,   139,
     140,   140,   141,   142,   142,   143,   143,   143,   144,   144,
     145,   145,   146,   146,   147,   148,   149,   149,   150,   150,
     151,   151,   152,   153,   154,   155,   155,   156,   156,   157,
     157,   157,   157,   158,   158,   158,   158,   158,   158,   159,
     159,   160,   161,   161,   161,   161,   161,   162,   162,   163,
     163,   164,   164,   164,   164,   164,   164,   164,   165,   165,
     165,   165,   165,   166,   166,   166,   166,   167,   167,   168,
     169,   170,   170,   170,   170,   171,   171,   171,   171,   171,
     171,   171,   171,   171,   171,   171,   172,   172,   172,   173,
     173,   174,   175,   175,   175,   176,   177,   177,   178,   178,
     178,   178,   179,   179,   179,   179,   179,   179,   179,   179,
     179,   179,   179,   179,   179,   179,   179,   179,   179,   179,
     179,   180,   180,   180,   181,   181,   181,   182,   182,   182,
     182,   182,   182,   183,   184,   185,   186,   186,   186,   186,
     186,   186,   186,   187,   187,   187,   188,   188,   188,   188,
     188,   188,   189,   190,   190,   190,   191,   191,   192,   192,
     193,   193,   194,   194,   195,   195,   196,   196,   196,   197,
     197,   198,   198,   199,   199,   199,   200,   200,   201,   201,
     201,   202,   202,   202,   202,   202,   202,   202,   202,   202,
     202,   202,   202,   203,   203,   203,   203,   203,   203,   203,
     203,   203,   203,   203,   203,   203,   203,   204,   204,   204,
     205,   206,   207,   208,   208,   209,   209,   210,   211,   212,
     213
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
       1,     1,     1,     4,     4,     3,     3,     1,     4,     0,
       2,     3,     2,     2,     2,     8,     5,     5,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     1,     1,     1,
       1,     1,     1,     1,     3,     0,     1,     0,     3,     1,
       1,     2,     1,     2,     1,     2,     2,     3,     3,     4,
       2,     3,     2,     2,     3,     1,     1,     2,     1,     2,
       2,     3,     1,     1,     2,     2,     2,     8,     1,     1,
       1,     1,     2,     2,     1,     1,     1,     2,     2,     6,
       5,     4,     3,     2,     1,     6,     5,     4,     3,     2,
       1,     1,     3,     0,     2,     4,     6,     0,     1,     0,
       3,     1,     3,     1,     1,     0,     3,     1,     3,     0,
       1,     1,     0,     3,     1,     3,     1,     1,     0,     1,
       0,     2,     5,     1,     2,     3,     5,     6,     0,     2,
       1,     3,     5,     5,     5,     5,     4,     3,     6,     6,
       5,     5,     5,     5,     5,     4,     7,     0,     2,     0,
       2,     2,     2,     6,     6,     3,     3,     2,     3,     1,
       3,     4,     2,     2,     2,     2,     2,     1,     4,     0,
       2,     1,     1,     1,     1,     2,     2,     2,     3,     6,
       9,     3,     6,     3,     6,     9,     9,     1,     3,     1,
       1,     1,     2,     2,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     7,     5,    13,     5,
       2,     1,     0,     3,     1,     3,     1,     3,     1,     4,
       3,     6,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     2,     1,     1,     1,
       1,     1,     1,     1,     0,     1,     3,     0,     1,     5,
       5,     5,     4,     3,     1,     1,     1,     3,     4,     3,
       4,     4,     4,     1,     1,     1,     1,     4,     3,     4,
       4,     4,     3,     7,     5,     6,     1,     3,     1,     3,
       3,     2,     3,     2,     0,     3,     1,     1,     4,     1,
       2,     1,     2,     1,     2,     1,     1,     0,     4,     3,
       5,     6,     4,     4,    11,     9,    12,    14,     6,     8,
       5,     7,     4,     6,     4,     1,     4,    11,     9,    12,
      14,     6,     8,     5,     7,     4,     1,     0,     2,     4,
       1,     1,     1,     2,     5,     1,     3,     1,     1,     2,
       2
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
#line 204 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.modlist) = (yyvsp[0].modlist); modlist = (yyvsp[0].modlist); }
#line 2287 "y.tab.c" /* yacc.c:1646  */
    break;

  case 3:
#line 208 "xi-grammar.y" /* yacc.c:1646  */
    { 
		  (yyval.modlist) = 0; 
		}
#line 2295 "y.tab.c" /* yacc.c:1646  */
    break;

  case 4:
#line 212 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.modlist) = new AstChildren<Module>(lineno, (yyvsp[-1].module), (yyvsp[0].modlist)); }
#line 2301 "y.tab.c" /* yacc.c:1646  */
    break;

  case 5:
#line 216 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 2307 "y.tab.c" /* yacc.c:1646  */
    break;

  case 6:
#line 218 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 2313 "y.tab.c" /* yacc.c:1646  */
    break;

  case 7:
#line 222 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 2319 "y.tab.c" /* yacc.c:1646  */
    break;

  case 8:
#line 224 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 2; }
#line 2325 "y.tab.c" /* yacc.c:1646  */
    break;

  case 9:
#line 228 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 2331 "y.tab.c" /* yacc.c:1646  */
    break;

  case 10:
#line 230 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 2337 "y.tab.c" /* yacc.c:1646  */
    break;

  case 11:
#line 235 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 2343 "y.tab.c" /* yacc.c:1646  */
    break;

  case 12:
#line 236 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MODULE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2349 "y.tab.c" /* yacc.c:1646  */
    break;

  case 13:
#line 237 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MAINMODULE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2355 "y.tab.c" /* yacc.c:1646  */
    break;

  case 14:
#line 238 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(EXTERN, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2361 "y.tab.c" /* yacc.c:1646  */
    break;

  case 15:
#line 240 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITCALL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2367 "y.tab.c" /* yacc.c:1646  */
    break;

  case 16:
#line 241 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITNODE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2373 "y.tab.c" /* yacc.c:1646  */
    break;

  case 17:
#line 242 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITPROC, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2379 "y.tab.c" /* yacc.c:1646  */
    break;

  case 18:
#line 244 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CHARE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2385 "y.tab.c" /* yacc.c:1646  */
    break;

  case 19:
#line 245 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MAINCHARE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2391 "y.tab.c" /* yacc.c:1646  */
    break;

  case 20:
#line 246 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(GROUP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2397 "y.tab.c" /* yacc.c:1646  */
    break;

  case 21:
#line 247 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NODEGROUP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2403 "y.tab.c" /* yacc.c:1646  */
    break;

  case 22:
#line 248 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ARRAY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2409 "y.tab.c" /* yacc.c:1646  */
    break;

  case 23:
#line 252 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INCLUDE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2415 "y.tab.c" /* yacc.c:1646  */
    break;

  case 24:
#line 253 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(STACKSIZE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2421 "y.tab.c" /* yacc.c:1646  */
    break;

  case 25:
#line 254 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(THREADED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2427 "y.tab.c" /* yacc.c:1646  */
    break;

  case 26:
#line 255 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(TEMPLATE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2433 "y.tab.c" /* yacc.c:1646  */
    break;

  case 27:
#line 256 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(WHENIDLE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2439 "y.tab.c" /* yacc.c:1646  */
    break;

  case 28:
#line 257 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(SYNC, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2445 "y.tab.c" /* yacc.c:1646  */
    break;

  case 29:
#line 258 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IGET, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2451 "y.tab.c" /* yacc.c:1646  */
    break;

  case 30:
#line 259 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(EXCLUSIVE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2457 "y.tab.c" /* yacc.c:1646  */
    break;

  case 31:
#line 260 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IMMEDIATE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2463 "y.tab.c" /* yacc.c:1646  */
    break;

  case 32:
#line 261 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(SKIPSCHED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2469 "y.tab.c" /* yacc.c:1646  */
    break;

  case 33:
#line 262 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOCOPY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2475 "y.tab.c" /* yacc.c:1646  */
    break;

  case 34:
#line 263 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOCOPYPOST, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2481 "y.tab.c" /* yacc.c:1646  */
    break;

  case 35:
#line 264 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOCOPYDEVICE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2487 "y.tab.c" /* yacc.c:1646  */
    break;

  case 36:
#line 265 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INLINE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2493 "y.tab.c" /* yacc.c:1646  */
    break;

  case 37:
#line 266 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(VIRTUAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2499 "y.tab.c" /* yacc.c:1646  */
    break;

  case 38:
#line 267 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MIGRATABLE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2505 "y.tab.c" /* yacc.c:1646  */
    break;

  case 39:
#line 268 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CREATEHERE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2511 "y.tab.c" /* yacc.c:1646  */
    break;

  case 40:
#line 269 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CREATEHOME, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2517 "y.tab.c" /* yacc.c:1646  */
    break;

  case 41:
#line 270 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOKEEP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2523 "y.tab.c" /* yacc.c:1646  */
    break;

  case 42:
#line 271 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOTRACE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2529 "y.tab.c" /* yacc.c:1646  */
    break;

  case 43:
#line 272 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(APPWORK, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2535 "y.tab.c" /* yacc.c:1646  */
    break;

  case 44:
#line 275 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(PACKED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2541 "y.tab.c" /* yacc.c:1646  */
    break;

  case 45:
#line 276 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(VARSIZE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2547 "y.tab.c" /* yacc.c:1646  */
    break;

  case 46:
#line 277 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ENTRY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2553 "y.tab.c" /* yacc.c:1646  */
    break;

  case 47:
#line 278 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(FOR, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2559 "y.tab.c" /* yacc.c:1646  */
    break;

  case 48:
#line 279 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(FORALL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2565 "y.tab.c" /* yacc.c:1646  */
    break;

  case 49:
#line 280 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(WHILE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2571 "y.tab.c" /* yacc.c:1646  */
    break;

  case 50:
#line 281 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(WHEN, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2577 "y.tab.c" /* yacc.c:1646  */
    break;

  case 51:
#line 282 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(OVERLAP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2583 "y.tab.c" /* yacc.c:1646  */
    break;

  case 52:
#line 283 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(SERIAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2589 "y.tab.c" /* yacc.c:1646  */
    break;

  case 53:
#line 284 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IF, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2595 "y.tab.c" /* yacc.c:1646  */
    break;

  case 54:
#line 285 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ELSE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2601 "y.tab.c" /* yacc.c:1646  */
    break;

  case 55:
#line 287 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(LOCAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2607 "y.tab.c" /* yacc.c:1646  */
    break;

  case 56:
#line 289 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(USING, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2613 "y.tab.c" /* yacc.c:1646  */
    break;

  case 57:
#line 290 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ACCEL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2619 "y.tab.c" /* yacc.c:1646  */
    break;

  case 58:
#line 293 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ACCELBLOCK, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2625 "y.tab.c" /* yacc.c:1646  */
    break;

  case 59:
#line 294 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MEMCRITICAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2631 "y.tab.c" /* yacc.c:1646  */
    break;

  case 60:
#line 295 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(REDUCTIONTARGET, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2637 "y.tab.c" /* yacc.c:1646  */
    break;

  case 61:
#line 296 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CASE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2643 "y.tab.c" /* yacc.c:1646  */
    break;

  case 62:
#line 301 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 2649 "y.tab.c" /* yacc.c:1646  */
    break;

  case 63:
#line 303 "xi-grammar.y" /* yacc.c:1646  */
    {
		  char *tmp = new char[strlen((yyvsp[-3].strval))+strlen((yyvsp[0].strval))+3];
		  sprintf(tmp,"%s::%s", (yyvsp[-3].strval), (yyvsp[0].strval));
		  (yyval.strval) = tmp;
		}
#line 2659 "y.tab.c" /* yacc.c:1646  */
    break;

  case 64:
#line 309 "xi-grammar.y" /* yacc.c:1646  */
    {
		  char *tmp = new char[strlen((yyvsp[-3].strval))+5+3];
		  sprintf(tmp,"%s::array", (yyvsp[-3].strval));
		  (yyval.strval) = tmp;
		}
#line 2669 "y.tab.c" /* yacc.c:1646  */
    break;

  case 65:
#line 316 "xi-grammar.y" /* yacc.c:1646  */
    { 
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist)); 
		}
#line 2677 "y.tab.c" /* yacc.c:1646  */
    break;

  case 66:
#line 320 "xi-grammar.y" /* yacc.c:1646  */
    {  
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist)); 
		    (yyval.module)->setMain();
		}
#line 2686 "y.tab.c" /* yacc.c:1646  */
    break;

  case 67:
#line 327 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = 0; }
#line 2692 "y.tab.c" /* yacc.c:1646  */
    break;

  case 68:
#line 329 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = (yyvsp[-2].conslist); }
#line 2698 "y.tab.c" /* yacc.c:1646  */
    break;

  case 69:
#line 333 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = 0; }
#line 2704 "y.tab.c" /* yacc.c:1646  */
    break;

  case 70:
#line 335 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = new ConstructList(lineno, (yyvsp[-1].construct), (yyvsp[0].conslist)); }
#line 2710 "y.tab.c" /* yacc.c:1646  */
    break;

  case 71:
#line 339 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new UsingScope((yyvsp[0].strval), false); }
#line 2716 "y.tab.c" /* yacc.c:1646  */
    break;

  case 72:
#line 341 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new UsingScope((yyvsp[0].strval), true); }
#line 2722 "y.tab.c" /* yacc.c:1646  */
    break;

  case 73:
#line 343 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].member)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].member); }
#line 2728 "y.tab.c" /* yacc.c:1646  */
    break;

  case 74:
#line 345 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].message)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].message); }
#line 2734 "y.tab.c" /* yacc.c:1646  */
    break;

  case 75:
#line 347 "xi-grammar.y" /* yacc.c:1646  */
    {
                  Entry *e = new Entry(lineno, (yyvsp[-5].attr), (yyvsp[-4].type), (yyvsp[-2].strval), (yyvsp[0].plist), 0, 0, 0, (yylsp[-7]).first_line, (yyloc).last_line);
                  int isExtern = 1;
                  e->setExtern(isExtern);
                  e->targs = (yyvsp[-1].tparlist);
                  e->label = new XStr;
                  (yyvsp[-3].ntype)->print(*e->label);
                  (yyval.construct) = e;
                  firstRdma = true;
                  firstDeviceRdma = true;
                }
#line 2750 "y.tab.c" /* yacc.c:1646  */
    break;

  case 76:
#line 361 "xi-grammar.y" /* yacc.c:1646  */
    { if((yyvsp[-2].conslist)) (yyvsp[-2].conslist)->recurse<int&>((yyvsp[-4].intval), &Construct::setExtern); (yyval.construct) = (yyvsp[-2].conslist); }
#line 2756 "y.tab.c" /* yacc.c:1646  */
    break;

  case 77:
#line 363 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new Scope((yyvsp[-3].strval), (yyvsp[-1].conslist)); }
#line 2762 "y.tab.c" /* yacc.c:1646  */
    break;

  case 78:
#line 365 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = (yyvsp[-1].construct); }
#line 2768 "y.tab.c" /* yacc.c:1646  */
    break;

  case 79:
#line 367 "xi-grammar.y" /* yacc.c:1646  */
    {
          ERROR("preceding construct must be semicolon terminated",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 2778 "y.tab.c" /* yacc.c:1646  */
    break;

  case 80:
#line 373 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].module)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].module); }
#line 2784 "y.tab.c" /* yacc.c:1646  */
    break;

  case 81:
#line 375 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2790 "y.tab.c" /* yacc.c:1646  */
    break;

  case 82:
#line 377 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2796 "y.tab.c" /* yacc.c:1646  */
    break;

  case 83:
#line 379 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2802 "y.tab.c" /* yacc.c:1646  */
    break;

  case 84:
#line 381 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2808 "y.tab.c" /* yacc.c:1646  */
    break;

  case 85:
#line 383 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].templat)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].templat); }
#line 2814 "y.tab.c" /* yacc.c:1646  */
    break;

  case 86:
#line 385 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = NULL; }
#line 2820 "y.tab.c" /* yacc.c:1646  */
    break;

  case 87:
#line 387 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = NULL; }
#line 2826 "y.tab.c" /* yacc.c:1646  */
    break;

  case 88:
#line 389 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = (yyvsp[0].accelBlock); }
#line 2832 "y.tab.c" /* yacc.c:1646  */
    break;

  case 89:
#line 391 "xi-grammar.y" /* yacc.c:1646  */
    {
          ERROR("invalid construct",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 2842 "y.tab.c" /* yacc.c:1646  */
    break;

  case 90:
#line 399 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamType((yyvsp[0].type)); }
#line 2848 "y.tab.c" /* yacc.c:1646  */
    break;

  case 91:
#line 401 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
#line 2854 "y.tab.c" /* yacc.c:1646  */
    break;

  case 92:
#line 403 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
#line 2860 "y.tab.c" /* yacc.c:1646  */
    break;

  case 93:
#line 407 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = new TParamList((yyvsp[0].tparam)); }
#line 2866 "y.tab.c" /* yacc.c:1646  */
    break;

  case 94:
#line 409 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = new TParamList((yyvsp[-2].tparam), (yyvsp[0].tparlist)); }
#line 2872 "y.tab.c" /* yacc.c:1646  */
    break;

  case 95:
#line 413 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = new TParamList(0); }
#line 2878 "y.tab.c" /* yacc.c:1646  */
    break;

  case 96:
#line 415 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = (yyvsp[0].tparlist); }
#line 2884 "y.tab.c" /* yacc.c:1646  */
    break;

  case 97:
#line 419 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = 0; }
#line 2890 "y.tab.c" /* yacc.c:1646  */
    break;

  case 98:
#line 421 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = (yyvsp[-1].tparlist); }
#line 2896 "y.tab.c" /* yacc.c:1646  */
    break;

  case 99:
#line 425 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("int"); }
#line 2902 "y.tab.c" /* yacc.c:1646  */
    break;

  case 100:
#line 427 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long"); }
#line 2908 "y.tab.c" /* yacc.c:1646  */
    break;

  case 101:
#line 429 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long int"); }
#line 2914 "y.tab.c" /* yacc.c:1646  */
    break;

  case 102:
#line 431 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("short"); }
#line 2920 "y.tab.c" /* yacc.c:1646  */
    break;

  case 103:
#line 433 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("short int"); }
#line 2926 "y.tab.c" /* yacc.c:1646  */
    break;

  case 104:
#line 435 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("char"); }
#line 2932 "y.tab.c" /* yacc.c:1646  */
    break;

  case 105:
#line 437 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned int"); }
#line 2938 "y.tab.c" /* yacc.c:1646  */
    break;

  case 106:
#line 439 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long"); }
#line 2944 "y.tab.c" /* yacc.c:1646  */
    break;

  case 107:
#line 441 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long int"); }
#line 2950 "y.tab.c" /* yacc.c:1646  */
    break;

  case 108:
#line 443 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long long"); }
#line 2956 "y.tab.c" /* yacc.c:1646  */
    break;

  case 109:
#line 445 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long long int"); }
#line 2962 "y.tab.c" /* yacc.c:1646  */
    break;

  case 110:
#line 447 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned short"); }
#line 2968 "y.tab.c" /* yacc.c:1646  */
    break;

  case 111:
#line 449 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned short int"); }
#line 2974 "y.tab.c" /* yacc.c:1646  */
    break;

  case 112:
#line 451 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned char"); }
#line 2980 "y.tab.c" /* yacc.c:1646  */
    break;

  case 113:
#line 453 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long long"); }
#line 2986 "y.tab.c" /* yacc.c:1646  */
    break;

  case 114:
#line 455 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long long int"); }
#line 2992 "y.tab.c" /* yacc.c:1646  */
    break;

  case 115:
#line 457 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("float"); }
#line 2998 "y.tab.c" /* yacc.c:1646  */
    break;

  case 116:
#line 459 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("double"); }
#line 3004 "y.tab.c" /* yacc.c:1646  */
    break;

  case 117:
#line 461 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long double"); }
#line 3010 "y.tab.c" /* yacc.c:1646  */
    break;

  case 118:
#line 463 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("void"); }
#line 3016 "y.tab.c" /* yacc.c:1646  */
    break;

  case 119:
#line 466 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ntype) = new NamedType((yyvsp[-1].strval),(yyvsp[0].tparlist)); }
#line 3022 "y.tab.c" /* yacc.c:1646  */
    break;

  case 120:
#line 467 "xi-grammar.y" /* yacc.c:1646  */
    { 
                    const char* basename, *scope;
                    splitScopedName((yyvsp[-1].strval), &scope, &basename);
                    (yyval.ntype) = new NamedType(basename, (yyvsp[0].tparlist), scope);
                }
#line 3032 "y.tab.c" /* yacc.c:1646  */
    break;

  case 121:
#line 473 "xi-grammar.y" /* yacc.c:1646  */
    {
			const char* basename, *scope;
			splitScopedName((yyvsp[-1].strval), &scope, &basename);
			(yyval.ntype) = new NamedType(basename, (yyvsp[0].tparlist), scope, true);
		}
#line 3042 "y.tab.c" /* yacc.c:1646  */
    break;

  case 122:
#line 481 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3048 "y.tab.c" /* yacc.c:1646  */
    break;

  case 123:
#line 483 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ntype); }
#line 3054 "y.tab.c" /* yacc.c:1646  */
    break;

  case 124:
#line 487 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ptype) = new PtrType((yyvsp[-1].type)); }
#line 3060 "y.tab.c" /* yacc.c:1646  */
    break;

  case 125:
#line 491 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
#line 3066 "y.tab.c" /* yacc.c:1646  */
    break;

  case 126:
#line 493 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
#line 3072 "y.tab.c" /* yacc.c:1646  */
    break;

  case 127:
#line 497 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ftype) = new FuncType((yyvsp[-7].type), (yyvsp[-4].strval), (yyvsp[-1].plist)); }
#line 3078 "y.tab.c" /* yacc.c:1646  */
    break;

  case 128:
#line 501 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3084 "y.tab.c" /* yacc.c:1646  */
    break;

  case 129:
#line 503 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3090 "y.tab.c" /* yacc.c:1646  */
    break;

  case 130:
#line 505 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3096 "y.tab.c" /* yacc.c:1646  */
    break;

  case 131:
#line 507 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ftype); }
#line 3102 "y.tab.c" /* yacc.c:1646  */
    break;

  case 132:
#line 509 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[0].type)); }
#line 3108 "y.tab.c" /* yacc.c:1646  */
    break;

  case 133:
#line 511 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[-1].type)); }
#line 3114 "y.tab.c" /* yacc.c:1646  */
    break;

  case 134:
#line 515 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3120 "y.tab.c" /* yacc.c:1646  */
    break;

  case 135:
#line 517 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3126 "y.tab.c" /* yacc.c:1646  */
    break;

  case 136:
#line 519 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3132 "y.tab.c" /* yacc.c:1646  */
    break;

  case 137:
#line 521 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[0].type)); }
#line 3138 "y.tab.c" /* yacc.c:1646  */
    break;

  case 138:
#line 523 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[-1].type)); }
#line 3144 "y.tab.c" /* yacc.c:1646  */
    break;

  case 139:
#line 527 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType(new RValueReferenceType((yyvsp[-5].type))); }
#line 3150 "y.tab.c" /* yacc.c:1646  */
    break;

  case 140:
#line 529 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType(new ReferenceType((yyvsp[-4].type))); }
#line 3156 "y.tab.c" /* yacc.c:1646  */
    break;

  case 141:
#line 531 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType((yyvsp[-3].type)); }
#line 3162 "y.tab.c" /* yacc.c:1646  */
    break;

  case 142:
#line 533 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new RValueReferenceType((yyvsp[-2].type)); }
#line 3168 "y.tab.c" /* yacc.c:1646  */
    break;

  case 143:
#line 535 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ReferenceType((yyvsp[-1].type)); }
#line 3174 "y.tab.c" /* yacc.c:1646  */
    break;

  case 144:
#line 537 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3180 "y.tab.c" /* yacc.c:1646  */
    break;

  case 145:
#line 541 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType(new RValueReferenceType((yyvsp[-5].type))); }
#line 3186 "y.tab.c" /* yacc.c:1646  */
    break;

  case 146:
#line 543 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType(new ReferenceType((yyvsp[-4].type))); }
#line 3192 "y.tab.c" /* yacc.c:1646  */
    break;

  case 147:
#line 545 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType((yyvsp[-3].type)); }
#line 3198 "y.tab.c" /* yacc.c:1646  */
    break;

  case 148:
#line 547 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new RValueReferenceType((yyvsp[-2].type)); }
#line 3204 "y.tab.c" /* yacc.c:1646  */
    break;

  case 149:
#line 549 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ReferenceType((yyvsp[-1].type)); }
#line 3210 "y.tab.c" /* yacc.c:1646  */
    break;

  case 150:
#line 551 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3216 "y.tab.c" /* yacc.c:1646  */
    break;

  case 151:
#line 555 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 3222 "y.tab.c" /* yacc.c:1646  */
    break;

  case 152:
#line 559 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = (yyvsp[-1].val); }
#line 3228 "y.tab.c" /* yacc.c:1646  */
    break;

  case 153:
#line 563 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.vallist) = 0; }
#line 3234 "y.tab.c" /* yacc.c:1646  */
    break;

  case 154:
#line 565 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.vallist) = new ValueList((yyvsp[-1].val), (yyvsp[0].vallist)); }
#line 3240 "y.tab.c" /* yacc.c:1646  */
    break;

  case 155:
#line 569 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].vallist)); }
#line 3246 "y.tab.c" /* yacc.c:1646  */
    break;

  case 156:
#line 573 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[-3].type), (yyvsp[-1].strval), (yyvsp[0].vallist), 1); }
#line 3252 "y.tab.c" /* yacc.c:1646  */
    break;

  case 157:
#line 577 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0;}
#line 3258 "y.tab.c" /* yacc.c:1646  */
    break;

  case 158:
#line 579 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0;}
#line 3264 "y.tab.c" /* yacc.c:1646  */
    break;

  case 159:
#line 583 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3270 "y.tab.c" /* yacc.c:1646  */
    break;

  case 160:
#line 585 "xi-grammar.y" /* yacc.c:1646  */
    { 
		  /*
		  printf("Warning: Message attributes are being phased out.\n");
		  printf("Warning: Please remove them from interface files.\n");
		  */
		  (yyval.intval) = (yyvsp[-1].intval); 
		}
#line 3282 "y.tab.c" /* yacc.c:1646  */
    break;

  case 161:
#line 595 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[0].intval); }
#line 3288 "y.tab.c" /* yacc.c:1646  */
    break;

  case 162:
#line 597 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[-2].intval) | (yyvsp[0].intval); }
#line 3294 "y.tab.c" /* yacc.c:1646  */
    break;

  case 163:
#line 601 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3300 "y.tab.c" /* yacc.c:1646  */
    break;

  case 164:
#line 603 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3306 "y.tab.c" /* yacc.c:1646  */
    break;

  case 165:
#line 607 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = 0; }
#line 3312 "y.tab.c" /* yacc.c:1646  */
    break;

  case 166:
#line 609 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-1].cattr); }
#line 3318 "y.tab.c" /* yacc.c:1646  */
    break;

  case 167:
#line 613 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[0].cattr); }
#line 3324 "y.tab.c" /* yacc.c:1646  */
    break;

  case 168:
#line 615 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
#line 3330 "y.tab.c" /* yacc.c:1646  */
    break;

  case 169:
#line 619 "xi-grammar.y" /* yacc.c:1646  */
    { python_doc = NULL; (yyval.intval) = 0; }
#line 3336 "y.tab.c" /* yacc.c:1646  */
    break;

  case 170:
#line 621 "xi-grammar.y" /* yacc.c:1646  */
    { python_doc = (yyvsp[0].strval); (yyval.intval) = 0; }
#line 3342 "y.tab.c" /* yacc.c:1646  */
    break;

  case 171:
#line 625 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CPYTHON; }
#line 3348 "y.tab.c" /* yacc.c:1646  */
    break;

  case 172:
#line 629 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = 0; }
#line 3354 "y.tab.c" /* yacc.c:1646  */
    break;

  case 173:
#line 631 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-1].cattr); }
#line 3360 "y.tab.c" /* yacc.c:1646  */
    break;

  case 174:
#line 635 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[0].cattr); }
#line 3366 "y.tab.c" /* yacc.c:1646  */
    break;

  case 175:
#line 637 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
#line 3372 "y.tab.c" /* yacc.c:1646  */
    break;

  case 176:
#line 641 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CMIGRATABLE; }
#line 3378 "y.tab.c" /* yacc.c:1646  */
    break;

  case 177:
#line 643 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CPYTHON; }
#line 3384 "y.tab.c" /* yacc.c:1646  */
    break;

  case 178:
#line 647 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3390 "y.tab.c" /* yacc.c:1646  */
    break;

  case 179:
#line 649 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 3396 "y.tab.c" /* yacc.c:1646  */
    break;

  case 180:
#line 652 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3402 "y.tab.c" /* yacc.c:1646  */
    break;

  case 181:
#line 654 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 3408 "y.tab.c" /* yacc.c:1646  */
    break;

  case 182:
#line 657 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mv) = new MsgVar((yyvsp[-3].type), (yyvsp[-2].strval), (yyvsp[-4].intval), (yyvsp[-1].intval)); }
#line 3414 "y.tab.c" /* yacc.c:1646  */
    break;

  case 183:
#line 661 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mvlist) = new MsgVarList((yyvsp[0].mv)); }
#line 3420 "y.tab.c" /* yacc.c:1646  */
    break;

  case 184:
#line 663 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mvlist) = new MsgVarList((yyvsp[-1].mv), (yyvsp[0].mvlist)); }
#line 3426 "y.tab.c" /* yacc.c:1646  */
    break;

  case 185:
#line 667 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, (yyvsp[0].ntype)); }
#line 3432 "y.tab.c" /* yacc.c:1646  */
    break;

  case 186:
#line 669 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, (yyvsp[-2].ntype)); }
#line 3438 "y.tab.c" /* yacc.c:1646  */
    break;

  case 187:
#line 671 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, (yyvsp[-3].ntype), (yyvsp[-1].mvlist)); }
#line 3444 "y.tab.c" /* yacc.c:1646  */
    break;

  case 188:
#line 675 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = 0; }
#line 3450 "y.tab.c" /* yacc.c:1646  */
    break;

  case 189:
#line 677 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = (yyvsp[0].typelist); }
#line 3456 "y.tab.c" /* yacc.c:1646  */
    break;

  case 190:
#line 681 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = new TypeList((yyvsp[0].ntype)); }
#line 3462 "y.tab.c" /* yacc.c:1646  */
    break;

  case 191:
#line 683 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = new TypeList((yyvsp[-2].ntype), (yyvsp[0].typelist)); }
#line 3468 "y.tab.c" /* yacc.c:1646  */
    break;

  case 192:
#line 687 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr)|Chare::CCHARE, (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3474 "y.tab.c" /* yacc.c:1646  */
    break;

  case 193:
#line 689 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3480 "y.tab.c" /* yacc.c:1646  */
    break;

  case 194:
#line 693 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3486 "y.tab.c" /* yacc.c:1646  */
    break;

  case 195:
#line 697 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new NodeGroup(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3492 "y.tab.c" /* yacc.c:1646  */
    break;

  case 196:
#line 701 "xi-grammar.y" /* yacc.c:1646  */
    {/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",(yyvsp[-2].strval));
			(yyval.ntype) = new NamedType(buf); 
		}
#line 3502 "y.tab.c" /* yacc.c:1646  */
    break;

  case 197:
#line 707 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ntype) = (yyvsp[-1].ntype); }
#line 3508 "y.tab.c" /* yacc.c:1646  */
    break;

  case 198:
#line 711 "xi-grammar.y" /* yacc.c:1646  */
    {  (yyval.chare) = new Array(lineno, (yyvsp[-4].cattr), (yyvsp[-3].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3514 "y.tab.c" /* yacc.c:1646  */
    break;

  case 199:
#line 713 "xi-grammar.y" /* yacc.c:1646  */
    {  (yyval.chare) = new Array(lineno, (yyvsp[-3].cattr), (yyvsp[-4].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3520 "y.tab.c" /* yacc.c:1646  */
    break;

  case 200:
#line 717 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr)|Chare::CCHARE, new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist));}
#line 3526 "y.tab.c" /* yacc.c:1646  */
    break;

  case 201:
#line 719 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3532 "y.tab.c" /* yacc.c:1646  */
    break;

  case 202:
#line 723 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3538 "y.tab.c" /* yacc.c:1646  */
    break;

  case 203:
#line 727 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new NodeGroup( lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3544 "y.tab.c" /* yacc.c:1646  */
    break;

  case 204:
#line 731 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Array( lineno, 0, (yyvsp[-3].ntype), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3550 "y.tab.c" /* yacc.c:1646  */
    break;

  case 205:
#line 735 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-1].strval))); }
#line 3556 "y.tab.c" /* yacc.c:1646  */
    break;

  case 206:
#line 737 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-4].strval)), (yyvsp[-2].mvlist)); }
#line 3562 "y.tab.c" /* yacc.c:1646  */
    break;

  case 207:
#line 741 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = 0; }
#line 3568 "y.tab.c" /* yacc.c:1646  */
    break;

  case 208:
#line 743 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3574 "y.tab.c" /* yacc.c:1646  */
    break;

  case 209:
#line 747 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = 0; }
#line 3580 "y.tab.c" /* yacc.c:1646  */
    break;

  case 210:
#line 749 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3586 "y.tab.c" /* yacc.c:1646  */
    break;

  case 211:
#line 751 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3592 "y.tab.c" /* yacc.c:1646  */
    break;

  case 212:
#line 753 "xi-grammar.y" /* yacc.c:1646  */
    {
		  XStr typeStr;
		  (yyvsp[0].ntype)->print(typeStr);
		  char *tmp = strdup(typeStr.get_string());
		  (yyval.strval) = tmp;
		}
#line 3603 "y.tab.c" /* yacc.c:1646  */
    break;

  case 213:
#line 762 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TTypeEllipsis(new NamedEllipsisType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3609 "y.tab.c" /* yacc.c:1646  */
    break;

  case 214:
#line 764 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TTypeEllipsis(new NamedEllipsisType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3615 "y.tab.c" /* yacc.c:1646  */
    break;

  case 215:
#line 766 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TType(new NamedType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3621 "y.tab.c" /* yacc.c:1646  */
    break;

  case 216:
#line 768 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TType(new NamedType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3627 "y.tab.c" /* yacc.c:1646  */
    break;

  case 217:
#line 770 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TFunc((yyvsp[-1].ftype), (yyvsp[0].strval)); }
#line 3633 "y.tab.c" /* yacc.c:1646  */
    break;

  case 218:
#line 772 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TName((yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].strval)); }
#line 3639 "y.tab.c" /* yacc.c:1646  */
    break;

  case 219:
#line 776 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = new TVarList((yyvsp[0].tvar)); }
#line 3645 "y.tab.c" /* yacc.c:1646  */
    break;

  case 220:
#line 778 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = new TVarList((yyvsp[-2].tvar), (yyvsp[0].tvarlist)); }
#line 3651 "y.tab.c" /* yacc.c:1646  */
    break;

  case 221:
#line 782 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = (yyvsp[-1].tvarlist); }
#line 3657 "y.tab.c" /* yacc.c:1646  */
    break;

  case 222:
#line 786 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3663 "y.tab.c" /* yacc.c:1646  */
    break;

  case 223:
#line 788 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3669 "y.tab.c" /* yacc.c:1646  */
    break;

  case 224:
#line 790 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3675 "y.tab.c" /* yacc.c:1646  */
    break;

  case 225:
#line 792 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3681 "y.tab.c" /* yacc.c:1646  */
    break;

  case 226:
#line 794 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].message)); (yyvsp[0].message)->setTemplate((yyval.templat)); }
#line 3687 "y.tab.c" /* yacc.c:1646  */
    break;

  case 227:
#line 798 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = 0; }
#line 3693 "y.tab.c" /* yacc.c:1646  */
    break;

  case 228:
#line 800 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = (yyvsp[-2].mbrlist); }
#line 3699 "y.tab.c" /* yacc.c:1646  */
    break;

  case 229:
#line 804 "xi-grammar.y" /* yacc.c:1646  */
    { 
                  if (!connectEntries.empty()) {
                    (yyval.mbrlist) = new AstChildren<Member>(connectEntries);
		  } else {
		    (yyval.mbrlist) = 0; 
                  }
		}
#line 3711 "y.tab.c" /* yacc.c:1646  */
    break;

  case 230:
#line 812 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = new AstChildren<Member>(-1, (yyvsp[-1].member), (yyvsp[0].mbrlist)); }
#line 3717 "y.tab.c" /* yacc.c:1646  */
    break;

  case 231:
#line 816 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].readonly); }
#line 3723 "y.tab.c" /* yacc.c:1646  */
    break;

  case 232:
#line 818 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].readonly); }
#line 3729 "y.tab.c" /* yacc.c:1646  */
    break;

  case 234:
#line 821 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].member); }
#line 3735 "y.tab.c" /* yacc.c:1646  */
    break;

  case 235:
#line 823 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].pupable); }
#line 3741 "y.tab.c" /* yacc.c:1646  */
    break;

  case 236:
#line 825 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].includeFile); }
#line 3747 "y.tab.c" /* yacc.c:1646  */
    break;

  case 237:
#line 827 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new ClassDeclaration(lineno,(yyvsp[0].strval)); }
#line 3753 "y.tab.c" /* yacc.c:1646  */
    break;

  case 238:
#line 831 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1); }
#line 3759 "y.tab.c" /* yacc.c:1646  */
    break;

  case 239:
#line 833 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1); }
#line 3765 "y.tab.c" /* yacc.c:1646  */
    break;

  case 240:
#line 835 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[-6].strval)) + '<' +
					    ((yyvsp[-4].tparlist))->to_string() + '>').c_str()),
				    1);
		}
#line 3775 "y.tab.c" /* yacc.c:1646  */
    break;

  case 241:
#line 841 "xi-grammar.y" /* yacc.c:1646  */
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[-2]).first_column, (yylsp[-2]).last_column, (yylsp[-2]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1);
		}
#line 3785 "y.tab.c" /* yacc.c:1646  */
    break;

  case 242:
#line 847 "xi-grammar.y" /* yacc.c:1646  */
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[-5]).first_column, (yylsp[-5]).last_column, (yylsp[-5]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1);
		}
#line 3795 "y.tab.c" /* yacc.c:1646  */
    break;

  case 243:
#line 856 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 0); }
#line 3801 "y.tab.c" /* yacc.c:1646  */
    break;

  case 244:
#line 858 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 0); }
#line 3807 "y.tab.c" /* yacc.c:1646  */
    break;

  case 245:
#line 860 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[-6].strval)) + '<' +
					    ((yyvsp[-4].tparlist))->to_string() + '>').c_str()),
				    0);
		}
#line 3817 "y.tab.c" /* yacc.c:1646  */
    break;

  case 246:
#line 866 "xi-grammar.y" /* yacc.c:1646  */
    {
                  InitCall* rtn = new InitCall(lineno, (yyvsp[-3].strval), 0);
                  rtn->setAccel();
                  (yyval.member) = rtn;
		}
#line 3827 "y.tab.c" /* yacc.c:1646  */
    break;

  case 247:
#line 874 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[0].ntype),0); }
#line 3833 "y.tab.c" /* yacc.c:1646  */
    break;

  case 248:
#line 876 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[-2].ntype),(yyvsp[0].pupable)); }
#line 3839 "y.tab.c" /* yacc.c:1646  */
    break;

  case 249:
#line 879 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.includeFile) = new IncludeFile(lineno,(yyvsp[0].strval)); }
#line 3845 "y.tab.c" /* yacc.c:1646  */
    break;

  case 250:
#line 883 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].member); }
#line 3851 "y.tab.c" /* yacc.c:1646  */
    break;

  case 251:
#line 887 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].entry); }
#line 3857 "y.tab.c" /* yacc.c:1646  */
    break;

  case 252:
#line 889 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyvsp[0].entry)->tspec = (yyvsp[-1].tvarlist);
                  (yyval.member) = (yyvsp[0].entry);
                }
#line 3866 "y.tab.c" /* yacc.c:1646  */
    break;

  case 253:
#line 894 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[-1].member); }
#line 3872 "y.tab.c" /* yacc.c:1646  */
    break;

  case 254:
#line 896 "xi-grammar.y" /* yacc.c:1646  */
    {
          ERROR("invalid SDAG member",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 3882 "y.tab.c" /* yacc.c:1646  */
    break;

  case 255:
#line 904 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3888 "y.tab.c" /* yacc.c:1646  */
    break;

  case 256:
#line 906 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3894 "y.tab.c" /* yacc.c:1646  */
    break;

  case 257:
#line 908 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3900 "y.tab.c" /* yacc.c:1646  */
    break;

  case 258:
#line 910 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3906 "y.tab.c" /* yacc.c:1646  */
    break;

  case 259:
#line 912 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3912 "y.tab.c" /* yacc.c:1646  */
    break;

  case 260:
#line 914 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3918 "y.tab.c" /* yacc.c:1646  */
    break;

  case 261:
#line 916 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3924 "y.tab.c" /* yacc.c:1646  */
    break;

  case 262:
#line 918 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3930 "y.tab.c" /* yacc.c:1646  */
    break;

  case 263:
#line 920 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3936 "y.tab.c" /* yacc.c:1646  */
    break;

  case 264:
#line 922 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3942 "y.tab.c" /* yacc.c:1646  */
    break;

  case 265:
#line 924 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3948 "y.tab.c" /* yacc.c:1646  */
    break;

  case 266:
#line 927 "xi-grammar.y" /* yacc.c:1646  */
    { 
                  (yyval.entry) = new Entry(lineno, (yyvsp[-5].attr), (yyvsp[-4].type), (yyvsp[-3].strval), (yyvsp[-2].plist), (yyvsp[-1].val), (yyvsp[0].sentry), (const char *) NULL, (yylsp[-6]).first_line, (yyloc).last_line);
		  if ((yyvsp[0].sentry) != 0) { 
		    (yyvsp[0].sentry)->con1 = new SdagConstruct(SIDENT, (yyvsp[-3].strval));
                    (yyvsp[0].sentry)->setEntry((yyval.entry));
                    (yyvsp[0].sentry)->param = new ParamList((yyvsp[-2].plist));
                  }
                  firstRdma = true;
                  firstDeviceRdma = true;
		}
#line 3963 "y.tab.c" /* yacc.c:1646  */
    break;

  case 267:
#line 938 "xi-grammar.y" /* yacc.c:1646  */
    { 
                  Entry *e = new Entry(lineno, (yyvsp[-3].attr), 0, (yyvsp[-2].strval), (yyvsp[-1].plist),  0, (yyvsp[0].sentry), (const char *) NULL, (yylsp[-4]).first_line, (yyloc).last_line);
                  if ((yyvsp[0].sentry) != 0) {
		    (yyvsp[0].sentry)->con1 = new SdagConstruct(SIDENT, (yyvsp[-2].strval));
                    (yyvsp[0].sentry)->setEntry((yyval.entry));
                    (yyvsp[0].sentry)->param = new ParamList((yyvsp[-1].plist));
                  }
                  firstRdma = true;
                  firstDeviceRdma = true;
		  if (e->param && e->param->isCkMigMsgPtr()) {
		    WARNING("CkMigrateMsg chare constructor is taken for granted",
		            (yyloc).first_column, (yyloc).last_column);
		    (yyval.entry) = NULL;
		  } else {
		    (yyval.entry) = e;
		  }
		}
#line 3985 "y.tab.c" /* yacc.c:1646  */
    break;

  case 268:
#line 956 "xi-grammar.y" /* yacc.c:1646  */
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
                  firstDeviceRdma = true;
                }
#line 4005 "y.tab.c" /* yacc.c:1646  */
    break;

  case 269:
#line 974 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.accelBlock) = new AccelBlock(lineno, new XStr((yyvsp[-2].strval))); }
#line 4011 "y.tab.c" /* yacc.c:1646  */
    break;

  case 270:
#line 976 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.accelBlock) = new AccelBlock(lineno, NULL); }
#line 4017 "y.tab.c" /* yacc.c:1646  */
    break;

  case 271:
#line 980 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 4023 "y.tab.c" /* yacc.c:1646  */
    break;

  case 272:
#line 984 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attr) = 0; }
#line 4029 "y.tab.c" /* yacc.c:1646  */
    break;

  case 273:
#line 986 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attr) = (yyvsp[-1].attr); }
#line 4035 "y.tab.c" /* yacc.c:1646  */
    break;

  case 274:
#line 988 "xi-grammar.y" /* yacc.c:1646  */
    { ERROR("invalid entry method attribute list",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
#line 4044 "y.tab.c" /* yacc.c:1646  */
    break;

  case 275:
#line 994 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attrarg) = new Attribute::Argument((yyvsp[-2].strval), atoi((yyvsp[0].strval))); }
#line 4050 "y.tab.c" /* yacc.c:1646  */
    break;

  case 276:
#line 998 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attrarg) = (yyvsp[0].attrarg); }
#line 4056 "y.tab.c" /* yacc.c:1646  */
    break;

  case 277:
#line 999 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attrarg) = (yyvsp[-2].attrarg); (yyvsp[-2].attrarg)->next = (yyvsp[0].attrarg); }
#line 4062 "y.tab.c" /* yacc.c:1646  */
    break;

  case 278:
#line 1003 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attr) = new Attribute((yyvsp[0].intval));           }
#line 4068 "y.tab.c" /* yacc.c:1646  */
    break;

  case 279:
#line 1004 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attr) = new Attribute((yyvsp[-3].intval), (yyvsp[-1].attrarg));       }
#line 4074 "y.tab.c" /* yacc.c:1646  */
    break;

  case 280:
#line 1005 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attr) = new Attribute((yyvsp[-2].intval), NULL, (yyvsp[0].attr)); }
#line 4080 "y.tab.c" /* yacc.c:1646  */
    break;

  case 281:
#line 1006 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attr) = new Attribute((yyvsp[-5].intval), (yyvsp[-3].attrarg), (yyvsp[0].attr));   }
#line 4086 "y.tab.c" /* yacc.c:1646  */
    break;

  case 282:
#line 1010 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = STHREADED; }
#line 4092 "y.tab.c" /* yacc.c:1646  */
    break;

  case 283:
#line 1012 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SWHENIDLE; }
#line 4098 "y.tab.c" /* yacc.c:1646  */
    break;

  case 284:
#line 1014 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SSYNC; }
#line 4104 "y.tab.c" /* yacc.c:1646  */
    break;

  case 285:
#line 1016 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SIGET; }
#line 4110 "y.tab.c" /* yacc.c:1646  */
    break;

  case 286:
#line 1018 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SLOCKED; }
#line 4116 "y.tab.c" /* yacc.c:1646  */
    break;

  case 287:
#line 1020 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SCREATEHERE; }
#line 4122 "y.tab.c" /* yacc.c:1646  */
    break;

  case 288:
#line 1022 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SCREATEHOME; }
#line 4128 "y.tab.c" /* yacc.c:1646  */
    break;

  case 289:
#line 1024 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SNOKEEP; }
#line 4134 "y.tab.c" /* yacc.c:1646  */
    break;

  case 290:
#line 1026 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SNOTRACE; }
#line 4140 "y.tab.c" /* yacc.c:1646  */
    break;

  case 291:
#line 1028 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SAPPWORK; }
#line 4146 "y.tab.c" /* yacc.c:1646  */
    break;

  case 292:
#line 1030 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SIMMEDIATE; }
#line 4152 "y.tab.c" /* yacc.c:1646  */
    break;

  case 293:
#line 1032 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SSKIPSCHED; }
#line 4158 "y.tab.c" /* yacc.c:1646  */
    break;

  case 294:
#line 1034 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SINLINE; }
#line 4164 "y.tab.c" /* yacc.c:1646  */
    break;

  case 295:
#line 1036 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SLOCAL; }
#line 4170 "y.tab.c" /* yacc.c:1646  */
    break;

  case 296:
#line 1038 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SPYTHON; }
#line 4176 "y.tab.c" /* yacc.c:1646  */
    break;

  case 297:
#line 1040 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SMEM; }
#line 4182 "y.tab.c" /* yacc.c:1646  */
    break;

  case 298:
#line 1042 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SREDUCE; }
#line 4188 "y.tab.c" /* yacc.c:1646  */
    break;

  case 299:
#line 1044 "xi-grammar.y" /* yacc.c:1646  */
    {
        (yyval.intval) = SAGGREGATE;
    }
#line 4196 "y.tab.c" /* yacc.c:1646  */
    break;

  case 300:
#line 1048 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("invalid entry method attribute",
		        (yylsp[0]).first_column, (yylsp[0]).last_column);
		  yyclearin;
		  yyerrok;
		}
#line 4207 "y.tab.c" /* yacc.c:1646  */
    break;

  case 301:
#line 1057 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4213 "y.tab.c" /* yacc.c:1646  */
    break;

  case 302:
#line 1059 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4219 "y.tab.c" /* yacc.c:1646  */
    break;

  case 303:
#line 1061 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4225 "y.tab.c" /* yacc.c:1646  */
    break;

  case 304:
#line 1065 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = ""; }
#line 4231 "y.tab.c" /* yacc.c:1646  */
    break;

  case 305:
#line 1067 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4237 "y.tab.c" /* yacc.c:1646  */
    break;

  case 306:
#line 1069 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s, %s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4247 "y.tab.c" /* yacc.c:1646  */
    break;

  case 307:
#line 1077 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = ""; }
#line 4253 "y.tab.c" /* yacc.c:1646  */
    break;

  case 308:
#line 1079 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4259 "y.tab.c" /* yacc.c:1646  */
    break;

  case 309:
#line 1081 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s[%s]%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4269 "y.tab.c" /* yacc.c:1646  */
    break;

  case 310:
#line 1087 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s{%s}%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4279 "y.tab.c" /* yacc.c:1646  */
    break;

  case 311:
#line 1093 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s(%s)%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4289 "y.tab.c" /* yacc.c:1646  */
    break;

  case 312:
#line 1099 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"(%s)%s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4299 "y.tab.c" /* yacc.c:1646  */
    break;

  case 313:
#line 1107 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval));
		}
#line 4308 "y.tab.c" /* yacc.c:1646  */
    break;

  case 314:
#line 1114 "xi-grammar.y" /* yacc.c:1646  */
    { 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			(yyval.intval) = 0;
		}
#line 4318 "y.tab.c" /* yacc.c:1646  */
    break;

  case 315:
#line 1122 "xi-grammar.y" /* yacc.c:1646  */
    { 
			in_braces=0;
			(yyval.intval) = 0;
		}
#line 4327 "y.tab.c" /* yacc.c:1646  */
    break;

  case 316:
#line 1129 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[0].type));}
#line 4333 "y.tab.c" /* yacc.c:1646  */
    break;

  case 317:
#line 1131 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval)); (yyval.pname)->setConditional((yyvsp[0].intval)); }
#line 4339 "y.tab.c" /* yacc.c:1646  */
    break;

  case 318:
#line 1133 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[-3].type),(yyvsp[-2].strval),0,(yyvsp[0].val));}
#line 4345 "y.tab.c" /* yacc.c:1646  */
    break;

  case 319:
#line 1135 "xi-grammar.y" /* yacc.c:1646  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
		}
#line 4354 "y.tab.c" /* yacc.c:1646  */
    break;

  case 320:
#line 1140 "xi-grammar.y" /* yacc.c:1646  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
			(yyval.pname)->setRdma(CMK_ZC_P2P_SEND_MSG);
			if(firstRdma) {
				(yyval.pname)->setFirstRdma(true);
				firstRdma = false;
			}
		}
#line 4368 "y.tab.c" /* yacc.c:1646  */
    break;

  case 321:
#line 1150 "xi-grammar.y" /* yacc.c:1646  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
			(yyval.pname)->setRdma(CMK_ZC_P2P_RECV_MSG);
			if(firstRdma) {
				(yyval.pname)->setFirstRdma(true);
				firstRdma = false;
			}
		}
#line 4382 "y.tab.c" /* yacc.c:1646  */
    break;

  case 322:
#line 1160 "xi-grammar.y" /* yacc.c:1646  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
			(yyval.pname)->setRdma(CMK_ZC_DEVICE_MSG);
			if (firstDeviceRdma) {
				(yyval.pname)->setFirstDeviceRdma(true);
				firstDeviceRdma = false;
			}
		}
#line 4396 "y.tab.c" /* yacc.c:1646  */
    break;

  case 323:
#line 1171 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READONLY; }
#line 4402 "y.tab.c" /* yacc.c:1646  */
    break;

  case 324:
#line 1172 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READWRITE; }
#line 4408 "y.tab.c" /* yacc.c:1646  */
    break;

  case 325:
#line 1173 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_WRITEONLY; }
#line 4414 "y.tab.c" /* yacc.c:1646  */
    break;

  case 326:
#line 1176 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr((yyvsp[0].strval)); }
#line 4420 "y.tab.c" /* yacc.c:1646  */
    break;

  case 327:
#line 1177 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "->" << (yyvsp[0].strval); }
#line 4426 "y.tab.c" /* yacc.c:1646  */
    break;

  case 328:
#line 1178 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[-2].xstrptr)) << "." << (yyvsp[0].strval); }
#line 4432 "y.tab.c" /* yacc.c:1646  */
    break;

  case 329:
#line 1180 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << *((yyvsp[-1].xstrptr)) << "]";
                  delete (yyvsp[-3].xstrptr);
                  delete (yyvsp[-1].xstrptr);
                }
#line 4443 "y.tab.c" /* yacc.c:1646  */
    break;

  case 330:
#line 1187 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << (yyvsp[-1].strval) << "]";
                  delete (yyvsp[-3].xstrptr);
                }
#line 4453 "y.tab.c" /* yacc.c:1646  */
    break;

  case 331:
#line 1193 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "(" << *((yyvsp[-1].xstrptr)) << ")";
                  delete (yyvsp[-3].xstrptr);
                  delete (yyvsp[-1].xstrptr);
                }
#line 4464 "y.tab.c" /* yacc.c:1646  */
    break;

  case 332:
#line 1202 "xi-grammar.y" /* yacc.c:1646  */
    {
                  in_bracket = 0;
                  (yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName(), (yyvsp[-1].strval));
                }
#line 4473 "y.tab.c" /* yacc.c:1646  */
    break;

  case 333:
#line 1209 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[-6].intval));
                }
#line 4483 "y.tab.c" /* yacc.c:1646  */
    break;

  case 334:
#line 1215 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType(Parameter::ACCEL_BUFFER_TYPE_READWRITE);
		}
#line 4493 "y.tab.c" /* yacc.c:1646  */
    break;

  case 335:
#line 1221 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.pname) = (yyvsp[-3].pname);
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[-5].intval));
		}
#line 4503 "y.tab.c" /* yacc.c:1646  */
    break;

  case 336:
#line 1229 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 4509 "y.tab.c" /* yacc.c:1646  */
    break;

  case 337:
#line 1231 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 4515 "y.tab.c" /* yacc.c:1646  */
    break;

  case 338:
#line 1235 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 4521 "y.tab.c" /* yacc.c:1646  */
    break;

  case 339:
#line 1237 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 4527 "y.tab.c" /* yacc.c:1646  */
    break;

  case 340:
#line 1241 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = (yyvsp[-1].plist); }
#line 4533 "y.tab.c" /* yacc.c:1646  */
    break;

  case 341:
#line 1243 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList(new Parameter(0, new BuiltinType("void"))); }
#line 4539 "y.tab.c" /* yacc.c:1646  */
    break;

  case 342:
#line 1247 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = (yyvsp[-1].plist); }
#line 4545 "y.tab.c" /* yacc.c:1646  */
    break;

  case 343:
#line 1249 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = 0; }
#line 4551 "y.tab.c" /* yacc.c:1646  */
    break;

  case 344:
#line 1253 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = 0; }
#line 4557 "y.tab.c" /* yacc.c:1646  */
    break;

  case 345:
#line 1255 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4563 "y.tab.c" /* yacc.c:1646  */
    break;

  case 346:
#line 1259 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = 0; }
#line 4569 "y.tab.c" /* yacc.c:1646  */
    break;

  case 347:
#line 1261 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[0].sc)); }
#line 4575 "y.tab.c" /* yacc.c:1646  */
    break;

  case 348:
#line 1263 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[-2].slist)); }
#line 4581 "y.tab.c" /* yacc.c:1646  */
    break;

  case 349:
#line 1267 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.slist) = new SListConstruct((yyvsp[0].sc)); }
#line 4587 "y.tab.c" /* yacc.c:1646  */
    break;

  case 350:
#line 1269 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.slist) = new SListConstruct((yyvsp[-1].sc), (yyvsp[0].slist));  }
#line 4593 "y.tab.c" /* yacc.c:1646  */
    break;

  case 351:
#line 1273 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.olist) = new OListConstruct((yyvsp[0].sc)); }
#line 4599 "y.tab.c" /* yacc.c:1646  */
    break;

  case 352:
#line 1275 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.olist) = new OListConstruct((yyvsp[-1].sc), (yyvsp[0].slist)); }
#line 4605 "y.tab.c" /* yacc.c:1646  */
    break;

  case 353:
#line 1279 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.clist) = new CaseListConstruct((yyvsp[0].when)); }
#line 4611 "y.tab.c" /* yacc.c:1646  */
    break;

  case 354:
#line 1281 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.clist) = new CaseListConstruct((yyvsp[-1].when), (yyvsp[0].clist)); }
#line 4617 "y.tab.c" /* yacc.c:1646  */
    break;

  case 355:
#line 1283 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("case blocks can only contain when clauses",
		        (yylsp[0]).first_column, (yylsp[0]).last_column);
		  (yyval.clist) = 0;
		}
#line 4627 "y.tab.c" /* yacc.c:1646  */
    break;

  case 356:
#line 1291 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4633 "y.tab.c" /* yacc.c:1646  */
    break;

  case 357:
#line 1293 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = 0; }
#line 4639 "y.tab.c" /* yacc.c:1646  */
    break;

  case 358:
#line 1297 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-2].entrylist), 0); }
#line 4645 "y.tab.c" /* yacc.c:1646  */
    break;

  case 359:
#line 1299 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-1].entrylist), (yyvsp[0].sc)); }
#line 4651 "y.tab.c" /* yacc.c:1646  */
    break;

  case 360:
#line 1301 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-3].entrylist), (yyvsp[-1].slist)); }
#line 4657 "y.tab.c" /* yacc.c:1646  */
    break;

  case 361:
#line 1305 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4663 "y.tab.c" /* yacc.c:1646  */
    break;

  case 362:
#line 1307 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4669 "y.tab.c" /* yacc.c:1646  */
    break;

  case 363:
#line 1309 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4675 "y.tab.c" /* yacc.c:1646  */
    break;

  case 364:
#line 1311 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4681 "y.tab.c" /* yacc.c:1646  */
    break;

  case 365:
#line 1313 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4687 "y.tab.c" /* yacc.c:1646  */
    break;

  case 366:
#line 1315 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4693 "y.tab.c" /* yacc.c:1646  */
    break;

  case 367:
#line 1317 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4699 "y.tab.c" /* yacc.c:1646  */
    break;

  case 368:
#line 1319 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4705 "y.tab.c" /* yacc.c:1646  */
    break;

  case 369:
#line 1321 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4711 "y.tab.c" /* yacc.c:1646  */
    break;

  case 370:
#line 1323 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4717 "y.tab.c" /* yacc.c:1646  */
    break;

  case 371:
#line 1325 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4723 "y.tab.c" /* yacc.c:1646  */
    break;

  case 372:
#line 1327 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4729 "y.tab.c" /* yacc.c:1646  */
    break;

  case 373:
#line 1331 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new SerialConstruct((yyvsp[-2].strval), (yyvsp[-4].strval), (yylsp[-3]).first_line); }
#line 4735 "y.tab.c" /* yacc.c:1646  */
    break;

  case 374:
#line 1333 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new OverlapConstruct((yyvsp[-1].olist)); }
#line 4741 "y.tab.c" /* yacc.c:1646  */
    break;

  case 375:
#line 1335 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = (yyvsp[0].when); }
#line 4747 "y.tab.c" /* yacc.c:1646  */
    break;

  case 376:
#line 1337 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new CaseConstruct((yyvsp[-1].clist)); }
#line 4753 "y.tab.c" /* yacc.c:1646  */
    break;

  case 377:
#line 1339 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForConstruct((yyvsp[-8].intexpr), (yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4759 "y.tab.c" /* yacc.c:1646  */
    break;

  case 378:
#line 1341 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForConstruct((yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4765 "y.tab.c" /* yacc.c:1646  */
    break;

  case 379:
#line 1343 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-9].strval)), (yyvsp[-6].intexpr),
		             (yyvsp[-4].intexpr), (yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4772 "y.tab.c" /* yacc.c:1646  */
    break;

  case 380:
#line 1346 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-11].strval)), (yyvsp[-8].intexpr),
		             (yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4779 "y.tab.c" /* yacc.c:1646  */
    break;

  case 381:
#line 1349 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new IfConstruct((yyvsp[-3].intexpr), (yyvsp[-1].sc), (yyvsp[0].sc)); }
#line 4785 "y.tab.c" /* yacc.c:1646  */
    break;

  case 382:
#line 1351 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new IfConstruct((yyvsp[-5].intexpr), (yyvsp[-2].slist), (yyvsp[0].sc)); }
#line 4791 "y.tab.c" /* yacc.c:1646  */
    break;

  case 383:
#line 1353 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new WhileConstruct((yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4797 "y.tab.c" /* yacc.c:1646  */
    break;

  case 384:
#line 1355 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new WhileConstruct((yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4803 "y.tab.c" /* yacc.c:1646  */
    break;

  case 385:
#line 1357 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new SerialConstruct((yyvsp[-2].strval), NULL, (yyloc).first_line); }
#line 4809 "y.tab.c" /* yacc.c:1646  */
    break;

  case 386:
#line 1359 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("unknown SDAG construct or malformed entry method declaration.\n"
		        "You may have forgotten to terminate a previous entry method declaration with a"
		        " semicolon or forgotten to mark a block of sequential SDAG code as 'serial'",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
#line 4821 "y.tab.c" /* yacc.c:1646  */
    break;

  case 387:
#line 1369 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = 0; }
#line 4827 "y.tab.c" /* yacc.c:1646  */
    break;

  case 388:
#line 1371 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ElseConstruct((yyvsp[0].sc)); }
#line 4833 "y.tab.c" /* yacc.c:1646  */
    break;

  case 389:
#line 1373 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ElseConstruct((yyvsp[-1].slist)); }
#line 4839 "y.tab.c" /* yacc.c:1646  */
    break;

  case 390:
#line 1377 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intexpr) = new IntExprConstruct((yyvsp[0].strval)); }
#line 4845 "y.tab.c" /* yacc.c:1646  */
    break;

  case 391:
#line 1381 "xi-grammar.y" /* yacc.c:1646  */
    { in_int_expr = 0; (yyval.intval) = 0; }
#line 4851 "y.tab.c" /* yacc.c:1646  */
    break;

  case 392:
#line 1385 "xi-grammar.y" /* yacc.c:1646  */
    { in_int_expr = 1; (yyval.intval) = 0; }
#line 4857 "y.tab.c" /* yacc.c:1646  */
    break;

  case 393:
#line 1389 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.entry) = new Entry(lineno, NULL, 0, (yyvsp[-1].strval), (yyvsp[0].plist), 0, 0, 0, (yyloc).first_line, (yyloc).last_line);
		  firstRdma = true;
		  firstDeviceRdma = true;
		}
#line 4867 "y.tab.c" /* yacc.c:1646  */
    break;

  case 394:
#line 1395 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.entry) = new Entry(lineno, NULL, 0, (yyvsp[-4].strval), (yyvsp[0].plist), 0, 0, (yyvsp[-2].strval), (yyloc).first_line, (yyloc).last_line);
		  firstRdma = true;
		  firstDeviceRdma = true;
		}
#line 4877 "y.tab.c" /* yacc.c:1646  */
    break;

  case 395:
#line 1403 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.entrylist) = new EntryList((yyvsp[0].entry)); }
#line 4883 "y.tab.c" /* yacc.c:1646  */
    break;

  case 396:
#line 1405 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.entrylist) = new EntryList((yyvsp[-2].entry),(yyvsp[0].entrylist)); }
#line 4889 "y.tab.c" /* yacc.c:1646  */
    break;

  case 397:
#line 1409 "xi-grammar.y" /* yacc.c:1646  */
    { in_bracket=1; }
#line 4895 "y.tab.c" /* yacc.c:1646  */
    break;

  case 398:
#line 1412 "xi-grammar.y" /* yacc.c:1646  */
    { in_bracket=0; }
#line 4901 "y.tab.c" /* yacc.c:1646  */
    break;

  case 399:
#line 1416 "xi-grammar.y" /* yacc.c:1646  */
    { if (!macroDefined((yyvsp[0].strval), 1)) in_comment = 1; }
#line 4907 "y.tab.c" /* yacc.c:1646  */
    break;

  case 400:
#line 1420 "xi-grammar.y" /* yacc.c:1646  */
    { if (!macroDefined((yyvsp[0].strval), 0)) in_comment = 1; }
#line 4913 "y.tab.c" /* yacc.c:1646  */
    break;


#line 4917 "y.tab.c" /* yacc.c:1646  */
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
#line 1423 "xi-grammar.y" /* yacc.c:1906  */


void yyerror(const char *s) 
{
	fprintf(stderr, "[PARSE-ERROR] Unexpected/missing token at line %d. Current token being parsed: '%s'.\n", lineno, yytext);
}
