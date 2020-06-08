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
    DEVICE = 297,
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
#define DEVICE 297
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

#line 356 "y.tab.c" /* yacc.c:355  */
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

#line 387 "y.tab.c" /* yacc.c:358  */

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
#define YYLAST   1524

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  95
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  117
/* YYNRULES -- Number of rules.  */
#define YYNRULES  390
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  777

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
       0,   199,   199,   204,   207,   212,   213,   217,   219,   224,
     225,   230,   232,   233,   234,   236,   237,   238,   240,   241,
     242,   243,   244,   248,   249,   250,   251,   252,   253,   254,
     255,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   271,   272,   273,   274,   275,   276,
     277,   278,   279,   280,   281,   283,   285,   286,   289,   290,
     291,   292,   296,   298,   304,   311,   315,   322,   324,   329,
     330,   334,   336,   338,   340,   342,   356,   358,   360,   362,
     368,   370,   372,   374,   376,   378,   380,   382,   384,   386,
     394,   396,   398,   402,   404,   409,   410,   415,   416,   420,
     422,   424,   426,   428,   430,   432,   434,   436,   438,   440,
     442,   444,   446,   448,   452,   453,   458,   466,   468,   472,
     476,   478,   482,   486,   488,   490,   492,   494,   496,   500,
     502,   504,   506,   508,   512,   514,   516,   518,   520,   522,
     526,   528,   530,   532,   534,   536,   540,   544,   549,   550,
     554,   558,   563,   564,   569,   570,   580,   582,   586,   588,
     593,   594,   598,   600,   605,   606,   610,   615,   616,   620,
     622,   626,   628,   633,   634,   638,   639,   642,   646,   648,
     652,   654,   656,   661,   662,   666,   668,   672,   674,   678,
     682,   686,   692,   696,   698,   702,   704,   708,   712,   716,
     720,   722,   727,   728,   733,   734,   736,   738,   747,   749,
     751,   753,   755,   757,   761,   763,   767,   771,   773,   775,
     777,   779,   783,   785,   790,   797,   801,   803,   805,   806,
     808,   810,   812,   816,   818,   820,   826,   832,   841,   843,
     845,   851,   859,   861,   864,   868,   872,   874,   879,   881,
     889,   891,   893,   895,   897,   899,   901,   903,   905,   907,
     909,   912,   923,   941,   959,   961,   965,   970,   971,   973,
     980,   982,   986,   988,   990,   992,   994,   996,   998,  1000,
    1002,  1004,  1006,  1008,  1010,  1012,  1014,  1016,  1018,  1020,
    1024,  1033,  1035,  1037,  1042,  1043,  1045,  1054,  1055,  1057,
    1063,  1069,  1075,  1083,  1090,  1098,  1105,  1107,  1109,  1111,
    1116,  1126,  1136,  1148,  1149,  1150,  1153,  1154,  1155,  1156,
    1163,  1169,  1178,  1185,  1191,  1197,  1205,  1207,  1211,  1213,
    1217,  1219,  1223,  1225,  1230,  1231,  1235,  1237,  1239,  1243,
    1245,  1249,  1251,  1255,  1257,  1259,  1267,  1270,  1273,  1275,
    1277,  1281,  1283,  1285,  1287,  1289,  1291,  1293,  1295,  1297,
    1299,  1301,  1303,  1307,  1309,  1311,  1313,  1315,  1317,  1319,
    1322,  1325,  1327,  1329,  1331,  1333,  1335,  1346,  1347,  1349,
    1353,  1357,  1361,  1365,  1371,  1379,  1381,  1385,  1388,  1392,
    1396
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
  "DEVICE", "PACKED", "VARSIZE", "ENTRY", "FOR", "FORALL", "WHILE", "WHEN",
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
     325,   326,   327,   328,   329,   330,   331,   332,   333,    59,
      58,   123,   125,    44,    60,    62,    42,    40,    41,    38,
      46,    91,    93,    61,    45
};
# endif

#define YYPACT_NINF -660

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-660)))

#define YYTABLE_NINF -342

#define yytable_value_is_error(Yytable_value) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
      77,  1318,  1318,    59,  -660,    77,  -660,  -660,  -660,  -660,
    -660,  -660,  -660,  -660,  -660,  -660,  -660,  -660,  -660,  -660,
    -660,  -660,  -660,  -660,  -660,  -660,  -660,  -660,  -660,  -660,
    -660,  -660,  -660,  -660,  -660,  -660,  -660,  -660,  -660,  -660,
    -660,  -660,  -660,  -660,  -660,  -660,  -660,  -660,  -660,  -660,
    -660,  -660,  -660,  -660,  -660,  -660,  -660,   -23,   -23,  -660,
    -660,  -660,   897,    52,  -660,  -660,  -660,    99,  1318,    95,
    1318,  1318,   200,  1071,    80,   637,   897,  -660,  -660,  -660,
    -660,   281,    89,   125,  -660,   135,  -660,  -660,  -660,    52,
     -16,   153,   161,   161,     6,    60,   138,   138,   138,   138,
     141,   146,  1318,   185,   168,   897,  -660,  -660,  -660,  -660,
    -660,  -660,  -660,  -660,   382,  -660,  -660,  -660,  -660,   187,
    -660,  -660,  -660,  -660,  -660,  -660,  -660,  -660,  -660,  -660,
    -660,    52,  -660,  -660,  -660,  1206,   987,   897,   135,   189,
      93,   -16,   214,  1446,  -660,  1431,  -660,   169,  -660,  -660,
    -660,  -660,   211,   125,   160,  -660,  -660,   221,   227,   236,
    -660,    38,   125,  -660,   125,   125,   259,   125,   257,  -660,
      20,  1318,  1318,  1318,  1318,    82,   262,   264,   145,  1318,
    -660,  -660,  -660,   456,   283,   138,   138,   138,   138,   262,
     146,  -660,  -660,  -660,  -660,  -660,    52,  -660,  -660,  -660,
    -660,  -660,  -660,  -660,  -660,  -660,  -660,  -660,  -660,  -660,
    -660,  -660,  -660,   284,  -660,  -660,  -660,   271,   285,   987,
     221,   227,   236,    14,  -660,    60,   289,    33,   -16,   314,
     -16,   292,  -660,   187,   296,    15,  -660,  -660,  -660,   240,
    -660,  -660,   160,  1413,  -660,  -660,  -660,  -660,  -660,   297,
     237,   294,   -25,    43,   113,   299,   184,    60,  -660,  -660,
     307,   305,   316,   309,   309,   309,   309,  -660,  1318,   310,
     318,   320,   136,  1318,   352,  1318,  -660,  -660,   326,   338,
     342,   809,    57,   119,  1318,   344,   343,   187,  1318,  1318,
    1318,  1318,  1318,  1318,  -660,  -660,  -660,  1206,   404,  -660,
     253,   365,  1318,  -660,  -660,  -660,   375,   377,   370,   376,
     -16,    52,   125,  -660,  -660,  -660,  -660,  -660,   379,  -660,
     385,  -660,  1318,   386,   388,   389,  -660,   378,  -660,   -16,
     161,  1413,   161,   161,  1413,   161,  -660,  -660,    20,  -660,
      60,   244,   244,   244,   244,   380,  -660,   352,  -660,   309,
     309,  -660,   145,     3,   390,   374,   150,   392,   107,  -660,
     394,   456,  -660,  -660,   309,   309,   309,   309,   309,   254,
    -660,   393,   395,   398,   316,   -16,   314,   -16,   -16,  -660,
     -25,  1413,  -660,   383,   400,   401,  -660,  -660,   405,  -660,
     387,   408,   410,   125,   411,   420,  -660,   409,  -660,   407,
      52,  -660,  -660,  -660,  -660,  -660,  -660,   244,   244,  -660,
    -660,  -660,  1431,     8,   417,   419,  1431,  -660,  -660,   422,
    -660,  -660,  -660,  -660,  -660,   244,   244,   244,   244,   244,
     493,    52,   437,   441,  -660,   446,  -660,  -660,  -660,  -660,
    -660,  -660,   451,   449,  -660,  -660,  -660,  -660,   453,  -660,
      55,   454,  -660,    60,  -660,   728,   498,   463,   187,   407,
    -660,  -660,  -660,  -660,  1318,  -660,  -660,  1318,  -660,   490,
    -660,  -660,  -660,  -660,  -660,   467,   460,  -660,  1365,  -660,
    1398,  -660,   161,   161,   161,  -660,  1150,  1092,  -660,   187,
      52,  -660,   461,   374,   374,   187,  -660,  1431,  1431,  1431,
    -660,  1318,   -16,   472,   469,   470,   471,   474,   476,   468,
     478,   446,  1318,  -660,   475,   187,  -660,  -660,    52,  1318,
     -16,   -16,   -16,    23,   477,  1398,  -660,  -660,  -660,  -660,
    -660,   528,   621,   446,  -660,    52,   479,   482,   495,   496,
    -660,   226,  -660,  -660,  -660,  1318,  -660,   481,   491,   481,
     531,   510,   539,   481,   519,   455,    52,   -16,  -660,  -660,
    -660,   581,  -660,  -660,  -660,  -660,  -660,   135,  -660,   446,
    -660,   -16,   544,   -16,   159,   521,   384,   628,  -660,   524,
     -16,  1142,   525,   546,   214,   516,   621,   523,  -660,   533,
     527,   532,  -660,   -16,   531,   529,  -660,   535,   584,   -16,
     532,   481,   530,   481,   534,   539,   481,   545,   -16,   543,
    1142,  -660,   187,  -660,   187,   578,  -660,   381,   524,   -16,
     481,  -660,   874,   405,  -660,  -660,   556,  -660,  -660,   214,
     881,   -16,   583,   -16,   628,   524,   -16,  1142,   214,  -660,
    -660,  -660,  -660,  -660,  -660,  -660,  -660,  -660,  1318,   559,
     570,   550,   -16,   575,   -16,   455,  -660,   446,  -660,   187,
     455,   602,   579,   565,   532,   577,   -16,   532,   580,   187,
     597,  1431,  1344,  -660,   214,   -16,   603,   604,  -660,  -660,
     605,   888,  -660,   -16,   481,   895,  -660,   214,   947,  -660,
    -660,  1318,  1318,   -16,   601,  -660,  1318,   532,   -16,  -660,
     602,   455,  -660,   609,   -16,   455,  -660,   187,   455,   602,
    -660,   247,   114,   571,  1318,   187,   955,   606,  -660,   610,
     -16,   613,   612,  -660,   614,  -660,  -660,  1318,  1318,  1243,
     616,  1318,  -660,   267,    52,   455,  -660,   -16,  -660,   532,
     -16,  -660,   602,   212,  -660,   607,   287,  1318,   279,  -660,
     615,   532,   962,   620,  -660,  -660,  -660,  -660,  -660,  -660,
    -660,   969,   455,  -660,   -16,   455,  -660,   622,   532,   624,
    -660,  1021,  -660,   455,  -660,   625,  -660
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
       8,     0,     0,     0,    62,    72,   389,   390,   304,   265,
     297,     0,   152,   152,   152,     0,   160,   160,   160,   160,
       0,   154,     0,     0,     0,     0,    80,   226,   227,    74,
      81,    82,    83,    84,     0,    85,    73,   229,   228,     9,
     260,   252,   253,   254,   255,   256,   258,   259,   257,   250,
     251,    78,    79,    70,   269,     0,     0,     0,    71,     0,
     298,   297,     0,     0,   113,     0,    99,   100,   101,   102,
     110,   111,     0,     0,    97,   117,   118,   123,   124,   125,
     126,   145,     0,   153,     0,     0,     0,     0,   242,   230,
       0,     0,     0,     0,     0,     0,     0,   167,     0,     0,
     232,   244,   231,     0,     0,   160,   160,   160,   160,     0,
     154,   217,   218,   219,   220,   221,    10,    68,   290,   272,
     273,   274,   275,   276,   282,   283,   284,   289,   277,   278,
     279,   280,   281,   164,   285,   287,   288,     0,   270,     0,
     129,   130,   131,   139,   266,     0,     0,     0,   297,   294,
     297,     0,   305,     0,     0,   127,   109,   112,   103,   104,
     107,   108,    97,    95,   115,   119,   120,   121,   128,     0,
     144,     0,   148,   236,   233,     0,   238,     0,   171,   172,
       0,   162,    97,   183,   183,   183,   183,   166,     0,     0,
     169,     0,     0,     0,     0,     0,   158,   159,     0,   156,
     180,     0,     0,   126,     0,   214,     0,     9,     0,     0,
       0,     0,     0,     0,   165,   286,   268,     0,   132,   133,
     138,     0,     0,    77,    64,    63,     0,   295,     0,     0,
     297,   264,     0,   105,   106,   116,    91,    92,    93,    96,
       0,    90,     0,   143,     0,     0,   387,   148,   150,   297,
     152,     0,   152,   152,     0,   152,   243,   161,     0,   114,
       0,     0,     0,     0,     0,     0,   192,     0,   168,   183,
     183,   155,     0,   173,     0,   202,    62,     0,     0,   212,
     204,     0,   216,    76,   183,   183,   183,   183,   183,     0,
     271,   137,     0,     0,    97,   297,   294,   297,   297,   302,
     148,     0,    98,     0,     0,     0,   142,   149,     0,   146,
       0,     0,     0,     0,     0,     0,   163,   185,   184,     0,
     222,   187,   188,   189,   190,   191,   170,     0,     0,   157,
     174,   181,     0,   173,     0,     0,     0,   210,   211,     0,
     205,   206,   207,   213,   215,     0,     0,     0,     0,     0,
     173,   200,     0,     0,   136,     0,   300,   296,   301,   299,
     151,    94,     0,     0,   141,   388,   147,   237,     0,   234,
       0,     0,   239,     0,   249,     0,     0,     0,     0,     0,
     245,   246,   193,   194,     0,   179,   182,     0,   203,     0,
     195,   196,   197,   198,   199,     0,     0,   135,     0,    75,
       0,   140,   152,   152,   152,   186,     0,     0,   247,     9,
     248,   225,   175,   202,   202,     0,   134,     0,     0,     0,
     331,   306,   297,   326,     0,     0,     0,     0,     0,     0,
      62,     0,     0,   223,     0,     0,   208,   209,   201,     0,
     297,   297,   297,   173,     0,     0,   330,   122,   235,   241,
     240,     0,     0,     0,   176,   177,     0,     0,     0,     0,
     303,     0,   307,   309,   327,     0,   376,     0,     0,     0,
       0,     0,   347,     0,     0,     0,   336,   297,   262,   365,
     337,   334,   310,   311,   312,   292,   291,   293,   308,     0,
     382,   297,     0,   297,     0,   385,     0,     0,   346,     0,
     297,     0,     0,     0,     0,     0,     0,     0,   380,     0,
       0,     0,   383,   297,     0,     0,   349,     0,     0,   297,
       0,     0,     0,     0,     0,   347,     0,     0,   297,     0,
     343,   345,     9,   340,     9,     0,   261,     0,     0,   297,
       0,   381,     0,     0,   386,   348,     0,   364,   342,     0,
       0,   297,     0,   297,     0,     0,   297,     0,     0,   366,
     344,   338,   375,   335,   313,   314,   315,   333,     0,     0,
     328,     0,   297,     0,   297,     0,   373,     0,   350,     9,
       0,   377,     0,     0,     0,     0,   297,     0,     0,     9,
       0,     0,     0,   332,     0,   297,     0,     0,   384,   363,
       0,     0,   371,   297,     0,     0,   352,     0,     0,   353,
     362,     0,     0,   297,     0,   329,     0,     0,   297,   374,
     377,     0,   378,     0,   297,     0,   360,     9,     0,   377,
     316,     0,     0,     0,     0,     0,     0,     0,   372,     0,
     297,     0,     0,   351,     0,   358,   324,     0,     0,     0,
       0,     0,   322,     0,   263,     0,   368,   297,   379,     0,
     297,   361,   377,     0,   318,     0,     0,     0,     0,   325,
       0,     0,     0,     0,   359,   321,   320,   319,   317,   323,
     367,     0,     0,   355,   297,     0,   369,     0,     0,     0,
     354,     0,   370,     0,   356,     0,   357
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -660,  -660,   703,  -660,   -51,  -283,    -1,   -57,   638,   652,
     -45,  -660,  -660,  -660,  -269,  -660,  -229,  -660,  -136,   -93,
    -125,  -131,  -120,  -174,   567,   499,  -660,   -88,  -660,  -660,
    -280,  -660,  -660,   -79,   548,   362,  -660,   129,   396,  -660,
    -660,   538,   373,  -660,   194,  -660,  -660,  -267,  -660,   -56,
     268,  -660,  -660,  -660,   -18,  -660,  -660,  -660,  -660,  -660,
    -660,  -339,   363,  -660,   361,   651,  -660,  -166,   266,   671,
    -660,  -660,   488,  -660,  -660,  -660,  -660,   304,  -660,   295,
     329,   492,  -660,  -660,   412,   -80,  -470,   -64,  -554,  -660,
    -660,  -457,  -660,  -660,  -438,   115,  -476,  -660,  -660,   204,
    -549,   157,  -488,   195,  -558,  -660,  -513,  -659,  -541,  -578,
    -477,  -660,   207,   233,   186,  -660,  -660
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     3,     4,    73,   400,   197,   262,   154,     5,    64,
      74,    75,    76,   318,   319,   320,   244,   155,   263,   156,
     157,   158,   159,   160,   161,   223,   224,   321,   388,   327,
     328,   107,   108,   164,   179,   278,   279,   171,   260,   295,
     270,   176,   271,   261,   412,   515,   413,   414,   109,   341,
     398,   110,   111,   112,   177,   113,   191,   192,   193,   194,
     195,   417,   359,   285,   286,   456,   115,   401,   457,   458,
     117,   118,   169,   182,   459,   460,   132,   461,    77,   225,
     136,   217,   218,   568,   308,   588,   502,   557,   233,   503,
     649,   711,   694,   650,   504,   651,   479,   618,   586,   558,
     582,   597,   609,   579,   559,   611,   583,   682,   589,   622,
     571,   575,   576,   329,   446,    78,    79
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[] =
{
      57,    58,   168,   162,   363,   221,    63,    63,    90,   283,
     142,   220,    85,   315,   165,   167,   222,   418,   234,   560,
     410,    89,   630,   610,   131,   410,   138,   520,   521,   522,
     614,   133,   591,   339,   613,   532,   264,   265,   266,   600,
     410,   718,   505,   280,   163,   140,   626,   387,   304,   628,
     725,   258,   610,   299,   248,   139,    61,   561,    62,    59,
     184,   231,   391,   596,   598,   394,   326,    82,   196,    86,
      87,   141,   573,   560,   259,   659,   580,   248,   653,   610,
       1,     2,   269,   754,   669,   411,   685,   544,   221,   688,
    -178,   305,   226,   587,   220,   284,   242,   166,   592,   222,
     440,   180,   249,   300,   301,   252,   677,   253,   254,   656,
     256,   680,   441,   676,   540,   356,   541,   661,    84,   716,
     696,   598,   640,   139,   631,   249,   633,   250,   251,   636,
     330,    80,   302,   707,   697,   139,   267,   349,   153,   350,
      84,   268,   483,   654,    81,   435,   465,   357,   306,   668,
     309,    83,   719,    84,   516,   517,   722,   717,   273,   724,
     153,   752,   119,   475,   168,    84,   420,   421,   702,   143,
     137,   292,   706,   761,   228,   709,   402,   403,   404,   269,
     229,   678,   311,    84,   230,   153,   750,   283,   276,   277,
     771,   144,   145,   139,    84,   268,   751,   331,   731,   163,
     332,   693,  -204,   736,  -204,   540,   513,   704,   342,   343,
     344,    84,   358,   767,   153,   139,   769,   146,   147,   148,
     149,   150,   151,   152,   775,   242,   172,   173,   174,   170,
     379,   153,   175,  -202,   236,  -202,   196,   178,   237,   763,
     139,   462,   463,   416,   243,   181,   478,   397,   766,   389,
     326,   390,   183,   392,   393,   380,   395,   733,   774,   470,
     471,   472,   473,   474,   139,   422,    61,   345,   334,   227,
     743,   335,   746,   284,   748,   238,   239,   240,   241,    61,
     355,    88,   134,   360,    84,   565,   566,   364,   365,   366,
     367,   368,   369,   407,   408,   436,   232,   438,   439,   727,
     755,   374,   728,   729,   313,   314,   730,   245,   425,   426,
     427,   428,   429,   246,   288,   289,   290,   291,   431,  -267,
    -267,   383,   247,    61,   464,   399,   323,   324,   468,   641,
     255,   642,   726,    61,   727,   430,   450,   728,   729,  -267,
     257,   730,   371,   372,   294,  -267,  -267,  -267,  -267,  -267,
    -267,  -267,   749,   272,   727,   274,   221,   728,   729,  -267,
     397,   730,   220,   296,   759,   287,   727,   222,   297,   728,
     729,   303,   135,   730,   727,   307,   679,   728,   729,   757,
     310,   730,   312,   322,   325,   546,   690,   644,   338,   340,
     501,   333,   501,   185,   186,   187,   188,   189,   190,   337,
     243,   347,   346,   506,   507,   508,   267,   490,   454,   519,
     519,   519,   348,    91,    92,    93,    94,    95,   351,   144,
     145,   352,   524,   353,   723,   102,   103,   361,   362,   104,
     547,   548,   549,   550,   551,   552,   553,   501,   196,    84,
     537,   538,   539,   299,   518,   146,   147,   148,   149,   150,
     151,   152,   455,   645,   646,   373,   546,   375,   377,   153,
     376,   554,   381,   492,   535,   595,   493,   416,   378,   326,
     382,   442,   405,   647,   281,   447,   384,   584,   385,   386,
     415,   556,   419,   432,   567,   433,   511,   358,   434,  -224,
     443,   444,   453,   448,   144,   145,   451,   445,   449,   466,
     523,   547,   548,   549,   550,   551,   552,   553,   452,   467,
     410,   533,   469,   623,    84,   599,  -304,   608,   536,   629,
     146,   147,   148,   149,   150,   151,   152,   476,   638,   648,
     546,   477,   554,   478,   282,   556,    88,  -304,   480,   481,
     482,   484,  -304,   455,   569,   489,   608,   546,   494,   495,
     496,   662,   514,   664,   652,   525,   667,   526,   527,   528,
     531,   196,   529,   196,   530,   -11,   545,   534,   570,   543,
     540,   666,   674,   608,   562,   547,   548,   549,   550,   551,
     552,   553,   572,   692,   648,   546,   687,   563,   564,   574,
    -304,   577,   547,   548,   549,   550,   551,   552,   553,   578,
     581,   585,   590,   703,   594,    88,   554,   612,   196,   615,
      88,   625,   619,   713,   617,   634,  -304,   627,   196,   620,
     621,   632,   546,   554,   721,   639,   637,    88,  -339,   546,
     547,   548,   549,   550,   551,   552,   553,   643,   658,   671,
     739,   663,   673,   120,   121,   122,   123,   670,   124,   125,
     126,   127,   128,   672,   675,   681,   196,   684,   683,   686,
     753,   554,   689,   732,   734,    88,  -341,   547,   548,   549,
     550,   551,   552,   553,   547,   548,   549,   550,   551,   552,
     553,   691,   129,   698,   768,   714,   699,   700,   720,   737,
     710,   712,   738,   740,   741,   715,   742,   760,   554,   756,
      61,   747,   555,   764,   770,   554,   772,   776,    60,    88,
      65,   106,   235,   710,   409,   275,    61,   542,   298,   130,
     406,   485,   424,   423,   114,   491,   710,   744,   710,   134,
     710,  -267,  -267,  -267,   396,  -267,  -267,  -267,   293,  -267,
    -267,  -267,  -267,  -267,   116,   336,   758,  -267,  -267,  -267,
    -267,  -267,  -267,  -267,  -267,  -267,  -267,  -267,  -267,  -267,
     488,  -267,  -267,  -267,  -267,  -267,  -267,  -267,  -267,  -267,
    -267,  -267,  -267,  -267,  -267,  -267,  -267,  -267,  -267,  -267,
    -267,  -267,   512,  -267,   487,  -267,  -267,   695,   437,   370,
     616,   665,  -267,  -267,  -267,  -267,  -267,  -267,  -267,  -267,
     635,   624,  -267,  -267,  -267,  -267,  -267,   593,     0,   657,
       0,     0,     6,     7,     8,     0,     9,    10,    11,   486,
      12,    13,    14,    15,    16,     0,     0,     0,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,     0,    30,    31,    32,    33,    34,     0,     0,    35,
      36,    37,    38,    39,    40,    41,    42,    43,    44,    45,
      46,    47,    48,     0,    49,     0,    50,    51,     0,     0,
       0,     0,     0,     0,     0,   546,     0,     0,     0,     0,
      52,     0,   546,    53,    54,    55,    56,     0,     0,   546,
       0,     0,     0,     0,     0,     0,   546,     0,    66,   354,
      -5,    -5,    67,    -5,    -5,    -5,    -5,    -5,    -5,    -5,
      -5,    -5,    -5,    -5,     0,    -5,    -5,     0,     0,    -5,
     547,   548,   549,   550,   551,   552,   553,   547,   548,   549,
     550,   551,   552,   553,   547,   548,   549,   550,   551,   552,
     553,   547,   548,   549,   550,   551,   552,   553,   546,     0,
       0,   554,     0,    68,    69,   655,   546,     0,   554,    70,
      71,     0,   660,   546,     0,   554,     0,     0,     0,   701,
     546,    72,   554,     0,     0,     0,   705,     0,    -5,   -69,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   547,   548,   549,   550,   551,   552,   553,
       0,   547,   548,   549,   550,   551,   552,   553,   547,   548,
     549,   550,   551,   552,   553,   547,   548,   549,   550,   551,
     552,   553,   546,     0,   554,   144,   219,     0,   708,     0,
       0,     0,   554,     0,     0,     0,   735,     0,     0,   554,
       0,     0,     0,   762,     0,    84,   554,     0,     0,     0,
     765,   146,   147,   148,   149,   150,   151,   152,     0,     0,
       0,     0,     0,     0,     0,   153,     0,   547,   548,   549,
     550,   551,   552,   553,     1,     2,     0,    91,    92,    93,
      94,    95,    96,    97,    98,    99,   100,   101,     0,   102,
     103,     0,     0,   104,     0,     6,     7,     8,   554,     9,
      10,    11,   773,    12,    13,    14,    15,    16,     0,     0,
       0,    17,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,     0,    30,    31,    32,    33,    34,
     144,   219,    35,    36,    37,    38,    39,    40,    41,    42,
      43,    44,    45,    46,    47,    48,     0,    49,     0,    50,
     510,   198,   105,     0,     0,     0,   146,   147,   148,   149,
     150,   151,   152,    52,     0,     0,    53,    54,    55,    56,
     153,   199,     0,   200,   201,   202,   203,   204,   205,   206,
       0,     0,   207,   208,   209,   210,   211,   212,   601,   602,
     603,   550,   604,   605,   606,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   213,   214,     0,   198,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   607,
       0,   509,     0,    88,     0,   215,   216,   199,     0,   200,
     201,   202,   203,   204,   205,   206,     0,     0,   207,   208,
     209,   210,   211,   212,     0,     0,     6,     7,     8,     0,
       9,    10,    11,     0,    12,    13,    14,    15,    16,     0,
     213,   214,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,     0,    30,    31,    32,    33,
      34,   215,   216,    35,    36,    37,    38,    39,    40,    41,
      42,    43,    44,    45,    46,    47,    48,     0,    49,     0,
      50,    51,   745,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    52,     0,     0,    53,    54,    55,
      56,     6,     7,     8,     0,     9,    10,    11,     0,    12,
      13,    14,    15,    16,     0,     0,     0,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
     644,    30,    31,    32,    33,    34,     0,     0,    35,    36,
      37,    38,    39,    40,    41,    42,    43,    44,    45,    46,
      47,    48,     0,    49,     0,    50,    51,     0,     0,     0,
       0,     0,   144,   145,     0,     0,     0,     0,     0,    52,
       0,     0,    53,    54,    55,    56,     0,     0,     0,     0,
       0,     0,    84,   144,   145,   497,   498,   499,   146,   147,
     148,   149,   150,   151,   152,     0,   645,   646,     0,     0,
       0,     0,   153,    84,     0,     0,     0,     0,     0,   146,
     147,   148,   149,   150,   151,   152,   144,   145,   497,   498,
     499,     0,     0,   153,     0,     0,     0,     0,     0,     0,
       0,   144,   145,   500,     0,     0,    84,     0,     0,     0,
       0,     0,   146,   147,   148,   149,   150,   151,   152,   144,
     145,    84,   316,   317,     0,     0,   153,   146,   147,   148,
     149,   150,   151,   152,   144,     0,     0,     0,     0,    84,
       0,   153,     0,     0,     0,   146,   147,   148,   149,   150,
     151,   152,     0,     0,    84,     0,     0,     0,     0,   153,
     146,   147,   148,   149,   150,   151,   152,     0,     0,     0,
       0,     0,     0,     0,   153
};

static const yytype_int16 yycheck[] =
{
       1,     2,    95,    91,   287,   136,    57,    58,    72,   183,
      90,   136,    69,   242,    93,    94,   136,   356,   143,   532,
      17,    72,   600,   581,    75,    17,    83,   497,   498,   499,
     584,    76,   573,   262,   583,   511,   172,   173,   174,   580,
      17,   700,   480,   179,    38,    61,   595,   327,    15,   598,
     709,    31,   610,    39,    39,    80,    79,   533,    81,     0,
     105,   141,   331,   576,   577,   334,    91,    68,   119,    70,
      71,    87,   549,   586,    54,   629,   553,    39,   619,   637,
       3,     4,   175,   742,   638,    82,   664,   525,   219,   667,
      82,    58,   137,   569,   219,   183,   153,    91,   574,   219,
     380,   102,    87,    89,    90,   162,   655,   164,   165,   622,
     167,   660,   381,   654,    91,    58,    93,   630,    58,   697,
     674,   634,   610,    80,   601,    87,   603,    89,    90,   606,
      87,    79,   225,   687,   675,    80,    54,   273,    78,   275,
      58,    59,    87,   620,    45,   374,   413,    90,   228,   637,
     230,    56,   701,    58,   493,   494,   705,   698,   176,   708,
      78,   739,    82,   430,   257,    58,    59,    60,   681,    16,
      81,   189,   685,   751,    81,   688,   342,   343,   344,   272,
      87,   657,   233,    58,    91,    78,   735,   361,    43,    44,
     768,    38,    39,    80,    58,    59,   737,    84,    84,    38,
      87,   671,    83,   716,    85,    91,   489,   684,   264,   265,
     266,    58,    93,   762,    78,    80,   765,    64,    65,    66,
      67,    68,    69,    70,   773,   282,    97,    98,    99,    91,
     310,    78,    91,    83,    65,    85,   287,    91,    69,   752,
      80,   407,   408,    93,    84,    60,    87,   340,   761,   329,
      91,   330,    84,   332,   333,   312,   335,   714,   771,   425,
     426,   427,   428,   429,    80,   358,    79,   268,    84,    80,
     727,    87,   729,   361,   731,    64,    65,    66,    67,    79,
     281,    81,     1,   284,    58,    59,    60,   288,   289,   290,
     291,   292,   293,   349,   350,   375,    82,   377,   378,    87,
      88,   302,    90,    91,    64,    65,    94,    86,   364,   365,
     366,   367,   368,    86,   185,   186,   187,   188,   369,    38,
      39,   322,    86,    79,   412,    81,    89,    90,   416,   612,
      71,   614,    85,    79,    87,    81,   393,    90,    91,    58,
      83,    94,    89,    90,    60,    64,    65,    66,    67,    68,
      69,    70,    85,    91,    87,    91,   487,    90,    91,    78,
     453,    94,   487,    92,    85,    82,    87,   487,    83,    90,
      91,    82,    91,    94,    87,    61,   659,    90,    91,    92,
      88,    94,    86,    86,    90,     1,   669,     6,    83,    80,
     478,    92,   480,    11,    12,    13,    14,    15,    16,    92,
      84,    83,    92,   482,   483,   484,    54,   458,     1,   497,
     498,   499,    92,     6,     7,     8,     9,    10,    92,    38,
      39,    83,   502,    81,   707,    18,    19,    83,    85,    22,
      46,    47,    48,    49,    50,    51,    52,   525,   489,    58,
     520,   521,   522,    39,   495,    64,    65,    66,    67,    68,
      69,    70,    45,    72,    73,    90,     1,    82,    88,    78,
      83,    77,    83,   464,   515,    81,   467,    93,    92,    91,
      85,    88,    92,    92,    18,    88,    90,   557,    90,    90,
      90,   532,    90,    90,   541,    90,   487,    93,    90,    82,
      90,    90,    83,    85,    38,    39,    85,    92,    88,    82,
     501,    46,    47,    48,    49,    50,    51,    52,    88,    90,
      17,   512,    90,   593,    58,   579,    61,   581,   519,   599,
      64,    65,    66,    67,    68,    69,    70,    90,   608,   617,
       1,    90,    77,    87,    78,   586,    81,    82,    87,    90,
      87,    87,    87,    45,   545,    82,   610,     1,    58,    82,
      90,   631,    91,   633,   618,    83,   636,    88,    88,    88,
      92,   612,    88,   614,    88,    87,    38,    92,    87,    92,
      91,   635,   652,   637,    92,    46,    47,    48,    49,    50,
      51,    52,    91,   671,   672,     1,   666,    92,    92,    58,
      61,    81,    46,    47,    48,    49,    50,    51,    52,    60,
      81,    20,    58,   683,    83,    81,    77,    82,   659,    93,
      81,    82,    79,   693,    91,    81,    87,    82,   669,    92,
      88,    91,     1,    77,   704,    82,    81,    81,    82,     1,
      46,    47,    48,    49,    50,    51,    52,    59,    82,    80,
     720,    58,    92,     6,     7,     8,     9,   648,    11,    12,
      13,    14,    15,    83,    79,    53,   707,    92,    79,    82,
     740,    77,    82,    92,   715,    81,    82,    46,    47,    48,
      49,    50,    51,    52,    46,    47,    48,    49,    50,    51,
      52,    84,    45,    80,   764,    84,    82,    82,    79,    83,
     691,   692,    82,    80,    82,   696,    82,    82,    77,    92,
      79,    85,    81,    83,    82,    77,    82,    82,     5,    81,
      58,    73,   145,   714,   352,   177,    79,   523,   219,    82,
     347,   453,   361,   360,    73,   459,   727,   728,   729,     1,
     731,     3,     4,     5,   338,     7,     8,     9,   190,    11,
      12,    13,    14,    15,    73,   257,   747,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
     456,    33,    34,    35,    36,    37,    38,    39,    40,    41,
      42,    43,    44,    45,    46,    47,    48,    49,    50,    51,
      52,    53,   487,    55,   455,    57,    58,   672,   376,   297,
     586,   634,    64,    65,    66,    67,    68,    69,    70,    71,
     605,   594,    74,    75,    76,    77,    78,   574,    -1,   623,
      -1,    -1,     3,     4,     5,    -1,     7,     8,     9,    91,
      11,    12,    13,    14,    15,    -1,    -1,    -1,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    -1,    33,    34,    35,    36,    37,    -1,    -1,    40,
      41,    42,    43,    44,    45,    46,    47,    48,    49,    50,
      51,    52,    53,    -1,    55,    -1,    57,    58,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,     1,    -1,    -1,    -1,    -1,
      71,    -1,     1,    74,    75,    76,    77,    -1,    -1,     1,
      -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,     1,    90,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    14,    15,    16,    -1,    18,    19,    -1,    -1,    22,
      46,    47,    48,    49,    50,    51,    52,    46,    47,    48,
      49,    50,    51,    52,    46,    47,    48,    49,    50,    51,
      52,    46,    47,    48,    49,    50,    51,    52,     1,    -1,
      -1,    77,    -1,    56,    57,    81,     1,    -1,    77,    62,
      63,    -1,    81,     1,    -1,    77,    -1,    -1,    -1,    81,
       1,    74,    77,    -1,    -1,    -1,    81,    -1,    81,    82,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    46,    47,    48,    49,    50,    51,    52,
      -1,    46,    47,    48,    49,    50,    51,    52,    46,    47,
      48,    49,    50,    51,    52,    46,    47,    48,    49,    50,
      51,    52,     1,    -1,    77,    38,    39,    -1,    81,    -1,
      -1,    -1,    77,    -1,    -1,    -1,    81,    -1,    -1,    77,
      -1,    -1,    -1,    81,    -1,    58,    77,    -1,    -1,    -1,
      81,    64,    65,    66,    67,    68,    69,    70,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    78,    -1,    46,    47,    48,
      49,    50,    51,    52,     3,     4,    -1,     6,     7,     8,
       9,    10,    11,    12,    13,    14,    15,    16,    -1,    18,
      19,    -1,    -1,    22,    -1,     3,     4,     5,    77,     7,
       8,     9,    81,    11,    12,    13,    14,    15,    -1,    -1,
      -1,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      28,    29,    30,    31,    -1,    33,    34,    35,    36,    37,
      38,    39,    40,    41,    42,    43,    44,    45,    46,    47,
      48,    49,    50,    51,    52,    53,    -1,    55,    -1,    57,
      58,     1,    81,    -1,    -1,    -1,    64,    65,    66,    67,
      68,    69,    70,    71,    -1,    -1,    74,    75,    76,    77,
      78,    21,    -1,    23,    24,    25,    26,    27,    28,    29,
      -1,    -1,    32,    33,    34,    35,    36,    37,    46,    47,
      48,    49,    50,    51,    52,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    54,    55,    -1,     1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    77,
      -1,    71,    -1,    81,    -1,    75,    76,    21,    -1,    23,
      24,    25,    26,    27,    28,    29,    -1,    -1,    32,    33,
      34,    35,    36,    37,    -1,    -1,     3,     4,     5,    -1,
       7,     8,     9,    -1,    11,    12,    13,    14,    15,    -1,
      54,    55,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    -1,    33,    34,    35,    36,
      37,    75,    76,    40,    41,    42,    43,    44,    45,    46,
      47,    48,    49,    50,    51,    52,    53,    -1,    55,    -1,
      57,    58,    59,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    71,    -1,    -1,    74,    75,    76,
      77,     3,     4,     5,    -1,     7,     8,     9,    -1,    11,
      12,    13,    14,    15,    -1,    -1,    -1,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
       6,    33,    34,    35,    36,    37,    -1,    -1,    40,    41,
      42,    43,    44,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    -1,    55,    -1,    57,    58,    -1,    -1,    -1,
      -1,    -1,    38,    39,    -1,    -1,    -1,    -1,    -1,    71,
      -1,    -1,    74,    75,    76,    77,    -1,    -1,    -1,    -1,
      -1,    -1,    58,    38,    39,    40,    41,    42,    64,    65,
      66,    67,    68,    69,    70,    -1,    72,    73,    -1,    -1,
      -1,    -1,    78,    58,    -1,    -1,    -1,    -1,    -1,    64,
      65,    66,    67,    68,    69,    70,    38,    39,    40,    41,
      42,    -1,    -1,    78,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    38,    39,    88,    -1,    -1,    58,    -1,    -1,    -1,
      -1,    -1,    64,    65,    66,    67,    68,    69,    70,    38,
      39,    58,    59,    60,    -1,    -1,    78,    64,    65,    66,
      67,    68,    69,    70,    38,    -1,    -1,    -1,    -1,    58,
      -1,    78,    -1,    -1,    -1,    64,    65,    66,    67,    68,
      69,    70,    -1,    -1,    58,    -1,    -1,    -1,    -1,    78,
      64,    65,    66,    67,    68,    69,    70,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    78
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
      62,    63,    74,    98,   105,   106,   107,   173,   210,   211,
      79,    45,   101,    56,    58,   102,   101,   101,    81,    99,
     182,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    18,    19,    22,    81,   103,   126,   127,   143,
     146,   147,   148,   150,   160,   161,   164,   165,   166,    82,
       6,     7,     8,     9,    11,    12,    13,    14,    15,    45,
      82,    99,   171,   105,     1,    91,   175,    81,   102,    80,
      61,    87,   180,    16,    38,    39,    64,    65,    66,    67,
      68,    69,    70,    78,   102,   112,   114,   115,   116,   117,
     118,   119,   122,    38,   128,   128,    91,   128,   114,   167,
      91,   132,   132,   132,   132,    91,   136,   149,    91,   129,
     101,    60,   168,    84,   105,    11,    12,    13,    14,    15,
      16,   151,   152,   153,   154,   155,    99,   100,     1,    21,
      23,    24,    25,    26,    27,    28,    29,    32,    33,    34,
      35,    36,    37,    54,    55,    75,    76,   176,   177,    39,
     115,   116,   117,   120,   121,   174,   105,    80,    81,    87,
      91,   180,    82,   183,   115,   119,    65,    69,    64,    65,
      66,    67,   102,    84,   111,    86,    86,    86,    39,    87,
      89,    90,   102,   102,   102,    71,   102,    83,    31,    54,
     133,   138,   101,   113,   113,   113,   113,    54,    59,   114,
     135,   137,    91,   149,    91,   136,    43,    44,   130,   131,
     113,    18,    78,   118,   122,   158,   159,    82,   132,   132,
     132,   132,   149,   129,    60,   134,    92,    83,   120,    39,
      89,    90,   114,    82,    15,    58,   180,    61,   179,   180,
      88,    99,    86,    64,    65,   111,    59,    60,   108,   109,
     110,   122,    86,    89,    90,    90,    91,   124,   125,   208,
      87,    84,    87,    92,    84,    87,   167,    92,    83,   111,
      80,   144,   144,   144,   144,   101,    92,    83,    92,   113,
     113,    92,    83,    81,    90,   101,    58,    90,    93,   157,
     101,    83,    85,   100,   101,   101,   101,   101,   101,   101,
     176,    89,    90,    90,   101,    82,    83,    88,    92,   180,
     102,    83,    85,   101,    90,    90,    90,   125,   123,   180,
     128,   109,   128,   128,   109,   128,   133,   114,   145,    81,
      99,   162,   162,   162,   162,    92,   137,   144,   144,   130,
      17,    82,   139,   141,   142,    90,    93,   156,   156,    90,
      59,    60,   114,   157,   159,   144,   144,   144,   144,   144,
      81,    99,    90,    90,    90,   111,   180,   179,   180,   180,
     125,   109,    88,    90,    90,    92,   209,    88,    85,    88,
     102,    85,    88,    83,     1,    45,   160,   163,   164,   169,
     170,   172,   162,   162,   122,   142,    82,    90,   122,    90,
     162,   162,   162,   162,   162,   142,    90,    90,    87,   191,
      87,    90,    87,    87,    87,   145,    91,   175,   172,    82,
      99,   163,   101,   101,    58,    82,    90,    40,    41,    42,
      88,   122,   181,   184,   189,   189,   128,   128,   128,    71,
      58,   101,   174,   100,    91,   140,   156,   156,    99,   122,
     181,   181,   181,   101,   180,    83,    88,    88,    88,    88,
      88,    92,   191,   101,    92,    99,   101,   180,   180,   180,
      91,    93,   139,    92,   189,    38,     1,    46,    47,    48,
      49,    50,    51,    52,    77,    81,    99,   182,   194,   199,
     201,   191,    92,    92,    92,    59,    60,   102,   178,   101,
      87,   205,    91,   205,    58,   206,   207,    81,    60,   198,
     205,    81,   195,   201,   180,    20,   193,   191,   180,   203,
      58,   203,   191,   208,    83,    81,   201,   196,   201,   182,
     203,    46,    47,    48,    50,    51,    52,    77,   182,   197,
     199,   200,    82,   195,   183,    93,   194,    91,   192,    79,
      92,    88,   204,   180,   207,    82,   195,    82,   195,   180,
     204,   205,    91,   205,    81,   198,   205,    81,   180,    82,
     197,   100,   100,    59,     6,    72,    73,    92,   122,   185,
     188,   190,   182,   203,   205,    81,   201,   209,    82,   183,
      81,   201,   180,    58,   180,   196,   182,   180,   197,   183,
     101,    80,    83,    92,   180,    79,   203,   195,   191,   100,
     195,    53,   202,    79,    92,   204,    82,   180,   204,    82,
     100,    84,   122,   181,   187,   190,   183,   203,    80,    82,
      82,    81,   201,   180,   205,    81,   201,   183,    81,   201,
     101,   186,   101,   180,    84,   101,   204,   203,   202,   195,
      79,   180,   195,   100,   195,   202,    85,    87,    90,    91,
      94,    84,    92,   186,    99,    81,   201,    83,    82,   180,
      80,    82,    82,   186,   101,    59,   186,    85,   186,    85,
     195,   203,   204,   180,   202,    88,    92,    92,   101,    85,
      82,   204,    81,   201,    83,    81,   201,   195,   180,   195,
      82,   204,    82,    81,   201,   195,    82
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
     176,   176,   177,   177,   177,   177,   177,   177,   177,   177,
     177,   177,   177,   177,   177,   177,   177,   177,   177,   177,
     177,   178,   178,   178,   179,   179,   179,   180,   180,   180,
     180,   180,   180,   181,   182,   183,   184,   184,   184,   184,
     184,   184,   184,   185,   185,   185,   186,   186,   186,   186,
     186,   186,   187,   188,   188,   188,   189,   189,   190,   190,
     191,   191,   192,   192,   193,   193,   194,   194,   194,   195,
     195,   196,   196,   197,   197,   197,   198,   198,   199,   199,
     199,   200,   200,   200,   200,   200,   200,   200,   200,   200,
     200,   200,   200,   201,   201,   201,   201,   201,   201,   201,
     201,   201,   201,   201,   201,   201,   201,   202,   202,   202,
     203,   204,   205,   206,   206,   207,   207,   208,   209,   210,
     211
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
       1,     1,     1,     2,     2,     3,     3,     2,     2,     2,
       1,     1,     2,     1,     2,     2,     3,     1,     1,     2,
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
       1,     3,     1,     1,     1,     1,     1,     1,     1,     1,
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
#line 200 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.modlist) = (yyvsp[0].modlist); modlist = (yyvsp[0].modlist); }
#line 2275 "y.tab.c" /* yacc.c:1646  */
    break;

  case 3:
#line 204 "xi-grammar.y" /* yacc.c:1646  */
    { 
		  (yyval.modlist) = 0; 
		}
#line 2283 "y.tab.c" /* yacc.c:1646  */
    break;

  case 4:
#line 208 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.modlist) = new AstChildren<Module>(lineno, (yyvsp[-1].module), (yyvsp[0].modlist)); }
#line 2289 "y.tab.c" /* yacc.c:1646  */
    break;

  case 5:
#line 212 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 2295 "y.tab.c" /* yacc.c:1646  */
    break;

  case 6:
#line 214 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 2301 "y.tab.c" /* yacc.c:1646  */
    break;

  case 7:
#line 218 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 2307 "y.tab.c" /* yacc.c:1646  */
    break;

  case 8:
#line 220 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 2; }
#line 2313 "y.tab.c" /* yacc.c:1646  */
    break;

  case 9:
#line 224 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 2319 "y.tab.c" /* yacc.c:1646  */
    break;

  case 10:
#line 226 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 2325 "y.tab.c" /* yacc.c:1646  */
    break;

  case 11:
#line 231 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 2331 "y.tab.c" /* yacc.c:1646  */
    break;

  case 12:
#line 232 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MODULE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2337 "y.tab.c" /* yacc.c:1646  */
    break;

  case 13:
#line 233 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MAINMODULE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2343 "y.tab.c" /* yacc.c:1646  */
    break;

  case 14:
#line 234 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(EXTERN, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2349 "y.tab.c" /* yacc.c:1646  */
    break;

  case 15:
#line 236 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITCALL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2355 "y.tab.c" /* yacc.c:1646  */
    break;

  case 16:
#line 237 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITNODE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2361 "y.tab.c" /* yacc.c:1646  */
    break;

  case 17:
#line 238 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITPROC, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2367 "y.tab.c" /* yacc.c:1646  */
    break;

  case 18:
#line 240 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CHARE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2373 "y.tab.c" /* yacc.c:1646  */
    break;

  case 19:
#line 241 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MAINCHARE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2379 "y.tab.c" /* yacc.c:1646  */
    break;

  case 20:
#line 242 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(GROUP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2385 "y.tab.c" /* yacc.c:1646  */
    break;

  case 21:
#line 243 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NODEGROUP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2391 "y.tab.c" /* yacc.c:1646  */
    break;

  case 22:
#line 244 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ARRAY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2397 "y.tab.c" /* yacc.c:1646  */
    break;

  case 23:
#line 248 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INCLUDE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2403 "y.tab.c" /* yacc.c:1646  */
    break;

  case 24:
#line 249 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(STACKSIZE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2409 "y.tab.c" /* yacc.c:1646  */
    break;

  case 25:
#line 250 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(THREADED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2415 "y.tab.c" /* yacc.c:1646  */
    break;

  case 26:
#line 251 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(TEMPLATE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2421 "y.tab.c" /* yacc.c:1646  */
    break;

  case 27:
#line 252 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(WHENIDLE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2427 "y.tab.c" /* yacc.c:1646  */
    break;

  case 28:
#line 253 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(SYNC, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2433 "y.tab.c" /* yacc.c:1646  */
    break;

  case 29:
#line 254 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IGET, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2439 "y.tab.c" /* yacc.c:1646  */
    break;

  case 30:
#line 255 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(EXCLUSIVE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2445 "y.tab.c" /* yacc.c:1646  */
    break;

  case 31:
#line 256 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IMMEDIATE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2451 "y.tab.c" /* yacc.c:1646  */
    break;

  case 32:
#line 257 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(SKIPSCHED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2457 "y.tab.c" /* yacc.c:1646  */
    break;

  case 33:
#line 258 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOCOPY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2463 "y.tab.c" /* yacc.c:1646  */
    break;

  case 34:
#line 259 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOCOPYPOST, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2469 "y.tab.c" /* yacc.c:1646  */
    break;

  case 35:
#line 260 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(DEVICE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2475 "y.tab.c" /* yacc.c:1646  */
    break;

  case 36:
#line 261 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INLINE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2481 "y.tab.c" /* yacc.c:1646  */
    break;

  case 37:
#line 262 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(VIRTUAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2487 "y.tab.c" /* yacc.c:1646  */
    break;

  case 38:
#line 263 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MIGRATABLE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2493 "y.tab.c" /* yacc.c:1646  */
    break;

  case 39:
#line 264 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CREATEHERE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2499 "y.tab.c" /* yacc.c:1646  */
    break;

  case 40:
#line 265 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CREATEHOME, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2505 "y.tab.c" /* yacc.c:1646  */
    break;

  case 41:
#line 266 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOKEEP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2511 "y.tab.c" /* yacc.c:1646  */
    break;

  case 42:
#line 267 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOTRACE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2517 "y.tab.c" /* yacc.c:1646  */
    break;

  case 43:
#line 268 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(APPWORK, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2523 "y.tab.c" /* yacc.c:1646  */
    break;

  case 44:
#line 271 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(PACKED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2529 "y.tab.c" /* yacc.c:1646  */
    break;

  case 45:
#line 272 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(VARSIZE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2535 "y.tab.c" /* yacc.c:1646  */
    break;

  case 46:
#line 273 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ENTRY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2541 "y.tab.c" /* yacc.c:1646  */
    break;

  case 47:
#line 274 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(FOR, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2547 "y.tab.c" /* yacc.c:1646  */
    break;

  case 48:
#line 275 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(FORALL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2553 "y.tab.c" /* yacc.c:1646  */
    break;

  case 49:
#line 276 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(WHILE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2559 "y.tab.c" /* yacc.c:1646  */
    break;

  case 50:
#line 277 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(WHEN, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2565 "y.tab.c" /* yacc.c:1646  */
    break;

  case 51:
#line 278 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(OVERLAP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2571 "y.tab.c" /* yacc.c:1646  */
    break;

  case 52:
#line 279 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(SERIAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2577 "y.tab.c" /* yacc.c:1646  */
    break;

  case 53:
#line 280 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IF, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2583 "y.tab.c" /* yacc.c:1646  */
    break;

  case 54:
#line 281 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ELSE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2589 "y.tab.c" /* yacc.c:1646  */
    break;

  case 55:
#line 283 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(LOCAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2595 "y.tab.c" /* yacc.c:1646  */
    break;

  case 56:
#line 285 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(USING, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2601 "y.tab.c" /* yacc.c:1646  */
    break;

  case 57:
#line 286 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ACCEL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2607 "y.tab.c" /* yacc.c:1646  */
    break;

  case 58:
#line 289 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ACCELBLOCK, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2613 "y.tab.c" /* yacc.c:1646  */
    break;

  case 59:
#line 290 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MEMCRITICAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2619 "y.tab.c" /* yacc.c:1646  */
    break;

  case 60:
#line 291 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(REDUCTIONTARGET, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2625 "y.tab.c" /* yacc.c:1646  */
    break;

  case 61:
#line 292 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CASE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2631 "y.tab.c" /* yacc.c:1646  */
    break;

  case 62:
#line 297 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 2637 "y.tab.c" /* yacc.c:1646  */
    break;

  case 63:
#line 299 "xi-grammar.y" /* yacc.c:1646  */
    {
		  char *tmp = new char[strlen((yyvsp[-3].strval))+strlen((yyvsp[0].strval))+3];
		  sprintf(tmp,"%s::%s", (yyvsp[-3].strval), (yyvsp[0].strval));
		  (yyval.strval) = tmp;
		}
#line 2647 "y.tab.c" /* yacc.c:1646  */
    break;

  case 64:
#line 305 "xi-grammar.y" /* yacc.c:1646  */
    {
		  char *tmp = new char[strlen((yyvsp[-3].strval))+5+3];
		  sprintf(tmp,"%s::array", (yyvsp[-3].strval));
		  (yyval.strval) = tmp;
		}
#line 2657 "y.tab.c" /* yacc.c:1646  */
    break;

  case 65:
#line 312 "xi-grammar.y" /* yacc.c:1646  */
    { 
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist)); 
		}
#line 2665 "y.tab.c" /* yacc.c:1646  */
    break;

  case 66:
#line 316 "xi-grammar.y" /* yacc.c:1646  */
    {  
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist)); 
		    (yyval.module)->setMain();
		}
#line 2674 "y.tab.c" /* yacc.c:1646  */
    break;

  case 67:
#line 323 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = 0; }
#line 2680 "y.tab.c" /* yacc.c:1646  */
    break;

  case 68:
#line 325 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = (yyvsp[-2].conslist); }
#line 2686 "y.tab.c" /* yacc.c:1646  */
    break;

  case 69:
#line 329 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = 0; }
#line 2692 "y.tab.c" /* yacc.c:1646  */
    break;

  case 70:
#line 331 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = new ConstructList(lineno, (yyvsp[-1].construct), (yyvsp[0].conslist)); }
#line 2698 "y.tab.c" /* yacc.c:1646  */
    break;

  case 71:
#line 335 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new UsingScope((yyvsp[0].strval), false); }
#line 2704 "y.tab.c" /* yacc.c:1646  */
    break;

  case 72:
#line 337 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new UsingScope((yyvsp[0].strval), true); }
#line 2710 "y.tab.c" /* yacc.c:1646  */
    break;

  case 73:
#line 339 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].member)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].member); }
#line 2716 "y.tab.c" /* yacc.c:1646  */
    break;

  case 74:
#line 341 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].message)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].message); }
#line 2722 "y.tab.c" /* yacc.c:1646  */
    break;

  case 75:
#line 343 "xi-grammar.y" /* yacc.c:1646  */
    {
                  Entry *e = new Entry(lineno, (yyvsp[-5].intval), (yyvsp[-4].type), (yyvsp[-2].strval), (yyvsp[0].plist), 0, 0, 0, (yylsp[-7]).first_line, (yyloc).last_line);
                  int isExtern = 1;
                  e->setExtern(isExtern);
                  e->targs = (yyvsp[-1].tparlist);
                  e->label = new XStr;
                  (yyvsp[-3].ntype)->print(*e->label);
                  (yyval.construct) = e;
                  firstRdma = true;
                  firstDeviceRdma = true;
                }
#line 2738 "y.tab.c" /* yacc.c:1646  */
    break;

  case 76:
#line 357 "xi-grammar.y" /* yacc.c:1646  */
    { if((yyvsp[-2].conslist)) (yyvsp[-2].conslist)->recurse<int&>((yyvsp[-4].intval), &Construct::setExtern); (yyval.construct) = (yyvsp[-2].conslist); }
#line 2744 "y.tab.c" /* yacc.c:1646  */
    break;

  case 77:
#line 359 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new Scope((yyvsp[-3].strval), (yyvsp[-1].conslist)); }
#line 2750 "y.tab.c" /* yacc.c:1646  */
    break;

  case 78:
#line 361 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = (yyvsp[-1].construct); }
#line 2756 "y.tab.c" /* yacc.c:1646  */
    break;

  case 79:
#line 363 "xi-grammar.y" /* yacc.c:1646  */
    {
          ERROR("preceding construct must be semicolon terminated",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 2766 "y.tab.c" /* yacc.c:1646  */
    break;

  case 80:
#line 369 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].module)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].module); }
#line 2772 "y.tab.c" /* yacc.c:1646  */
    break;

  case 81:
#line 371 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2778 "y.tab.c" /* yacc.c:1646  */
    break;

  case 82:
#line 373 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2784 "y.tab.c" /* yacc.c:1646  */
    break;

  case 83:
#line 375 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2790 "y.tab.c" /* yacc.c:1646  */
    break;

  case 84:
#line 377 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2796 "y.tab.c" /* yacc.c:1646  */
    break;

  case 85:
#line 379 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].templat)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].templat); }
#line 2802 "y.tab.c" /* yacc.c:1646  */
    break;

  case 86:
#line 381 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = NULL; }
#line 2808 "y.tab.c" /* yacc.c:1646  */
    break;

  case 87:
#line 383 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = NULL; }
#line 2814 "y.tab.c" /* yacc.c:1646  */
    break;

  case 88:
#line 385 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = (yyvsp[0].accelBlock); }
#line 2820 "y.tab.c" /* yacc.c:1646  */
    break;

  case 89:
#line 387 "xi-grammar.y" /* yacc.c:1646  */
    {
          ERROR("invalid construct",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 2830 "y.tab.c" /* yacc.c:1646  */
    break;

  case 90:
#line 395 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamType((yyvsp[0].type)); }
#line 2836 "y.tab.c" /* yacc.c:1646  */
    break;

  case 91:
#line 397 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
#line 2842 "y.tab.c" /* yacc.c:1646  */
    break;

  case 92:
#line 399 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
#line 2848 "y.tab.c" /* yacc.c:1646  */
    break;

  case 93:
#line 403 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = new TParamList((yyvsp[0].tparam)); }
#line 2854 "y.tab.c" /* yacc.c:1646  */
    break;

  case 94:
#line 405 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = new TParamList((yyvsp[-2].tparam), (yyvsp[0].tparlist)); }
#line 2860 "y.tab.c" /* yacc.c:1646  */
    break;

  case 95:
#line 409 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = new TParamList(0); }
#line 2866 "y.tab.c" /* yacc.c:1646  */
    break;

  case 96:
#line 411 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = (yyvsp[0].tparlist); }
#line 2872 "y.tab.c" /* yacc.c:1646  */
    break;

  case 97:
#line 415 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = 0; }
#line 2878 "y.tab.c" /* yacc.c:1646  */
    break;

  case 98:
#line 417 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = (yyvsp[-1].tparlist); }
#line 2884 "y.tab.c" /* yacc.c:1646  */
    break;

  case 99:
#line 421 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("int"); }
#line 2890 "y.tab.c" /* yacc.c:1646  */
    break;

  case 100:
#line 423 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long"); }
#line 2896 "y.tab.c" /* yacc.c:1646  */
    break;

  case 101:
#line 425 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("short"); }
#line 2902 "y.tab.c" /* yacc.c:1646  */
    break;

  case 102:
#line 427 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("char"); }
#line 2908 "y.tab.c" /* yacc.c:1646  */
    break;

  case 103:
#line 429 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned int"); }
#line 2914 "y.tab.c" /* yacc.c:1646  */
    break;

  case 104:
#line 431 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long"); }
#line 2920 "y.tab.c" /* yacc.c:1646  */
    break;

  case 105:
#line 433 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long"); }
#line 2926 "y.tab.c" /* yacc.c:1646  */
    break;

  case 106:
#line 435 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long long"); }
#line 2932 "y.tab.c" /* yacc.c:1646  */
    break;

  case 107:
#line 437 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned short"); }
#line 2938 "y.tab.c" /* yacc.c:1646  */
    break;

  case 108:
#line 439 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned char"); }
#line 2944 "y.tab.c" /* yacc.c:1646  */
    break;

  case 109:
#line 441 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long long"); }
#line 2950 "y.tab.c" /* yacc.c:1646  */
    break;

  case 110:
#line 443 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("float"); }
#line 2956 "y.tab.c" /* yacc.c:1646  */
    break;

  case 111:
#line 445 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("double"); }
#line 2962 "y.tab.c" /* yacc.c:1646  */
    break;

  case 112:
#line 447 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long double"); }
#line 2968 "y.tab.c" /* yacc.c:1646  */
    break;

  case 113:
#line 449 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("void"); }
#line 2974 "y.tab.c" /* yacc.c:1646  */
    break;

  case 114:
#line 452 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ntype) = new NamedType((yyvsp[-1].strval),(yyvsp[0].tparlist)); }
#line 2980 "y.tab.c" /* yacc.c:1646  */
    break;

  case 115:
#line 453 "xi-grammar.y" /* yacc.c:1646  */
    { 
                    const char* basename, *scope;
                    splitScopedName((yyvsp[-1].strval), &scope, &basename);
                    (yyval.ntype) = new NamedType(basename, (yyvsp[0].tparlist), scope);
                }
#line 2990 "y.tab.c" /* yacc.c:1646  */
    break;

  case 116:
#line 459 "xi-grammar.y" /* yacc.c:1646  */
    {
			const char* basename, *scope;
			splitScopedName((yyvsp[-1].strval), &scope, &basename);
			(yyval.ntype) = new NamedType(basename, (yyvsp[0].tparlist), scope, true);
		}
#line 3000 "y.tab.c" /* yacc.c:1646  */
    break;

  case 117:
#line 467 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3006 "y.tab.c" /* yacc.c:1646  */
    break;

  case 118:
#line 469 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ntype); }
#line 3012 "y.tab.c" /* yacc.c:1646  */
    break;

  case 119:
#line 473 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ptype) = new PtrType((yyvsp[-1].type)); }
#line 3018 "y.tab.c" /* yacc.c:1646  */
    break;

  case 120:
#line 477 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
#line 3024 "y.tab.c" /* yacc.c:1646  */
    break;

  case 121:
#line 479 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
#line 3030 "y.tab.c" /* yacc.c:1646  */
    break;

  case 122:
#line 483 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ftype) = new FuncType((yyvsp[-7].type), (yyvsp[-4].strval), (yyvsp[-1].plist)); }
#line 3036 "y.tab.c" /* yacc.c:1646  */
    break;

  case 123:
#line 487 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3042 "y.tab.c" /* yacc.c:1646  */
    break;

  case 124:
#line 489 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3048 "y.tab.c" /* yacc.c:1646  */
    break;

  case 125:
#line 491 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3054 "y.tab.c" /* yacc.c:1646  */
    break;

  case 126:
#line 493 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ftype); }
#line 3060 "y.tab.c" /* yacc.c:1646  */
    break;

  case 127:
#line 495 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[0].type)); }
#line 3066 "y.tab.c" /* yacc.c:1646  */
    break;

  case 128:
#line 497 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[-1].type)); }
#line 3072 "y.tab.c" /* yacc.c:1646  */
    break;

  case 129:
#line 501 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3078 "y.tab.c" /* yacc.c:1646  */
    break;

  case 130:
#line 503 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3084 "y.tab.c" /* yacc.c:1646  */
    break;

  case 131:
#line 505 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3090 "y.tab.c" /* yacc.c:1646  */
    break;

  case 132:
#line 507 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[0].type)); }
#line 3096 "y.tab.c" /* yacc.c:1646  */
    break;

  case 133:
#line 509 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[-1].type)); }
#line 3102 "y.tab.c" /* yacc.c:1646  */
    break;

  case 134:
#line 513 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType(new RValueReferenceType((yyvsp[-5].type))); }
#line 3108 "y.tab.c" /* yacc.c:1646  */
    break;

  case 135:
#line 515 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType(new ReferenceType((yyvsp[-4].type))); }
#line 3114 "y.tab.c" /* yacc.c:1646  */
    break;

  case 136:
#line 517 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType((yyvsp[-3].type)); }
#line 3120 "y.tab.c" /* yacc.c:1646  */
    break;

  case 137:
#line 519 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new RValueReferenceType((yyvsp[-2].type)); }
#line 3126 "y.tab.c" /* yacc.c:1646  */
    break;

  case 138:
#line 521 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ReferenceType((yyvsp[-1].type)); }
#line 3132 "y.tab.c" /* yacc.c:1646  */
    break;

  case 139:
#line 523 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3138 "y.tab.c" /* yacc.c:1646  */
    break;

  case 140:
#line 527 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType(new RValueReferenceType((yyvsp[-5].type))); }
#line 3144 "y.tab.c" /* yacc.c:1646  */
    break;

  case 141:
#line 529 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType(new ReferenceType((yyvsp[-4].type))); }
#line 3150 "y.tab.c" /* yacc.c:1646  */
    break;

  case 142:
#line 531 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType((yyvsp[-3].type)); }
#line 3156 "y.tab.c" /* yacc.c:1646  */
    break;

  case 143:
#line 533 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new RValueReferenceType((yyvsp[-2].type)); }
#line 3162 "y.tab.c" /* yacc.c:1646  */
    break;

  case 144:
#line 535 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ReferenceType((yyvsp[-1].type)); }
#line 3168 "y.tab.c" /* yacc.c:1646  */
    break;

  case 145:
#line 537 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3174 "y.tab.c" /* yacc.c:1646  */
    break;

  case 146:
#line 541 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 3180 "y.tab.c" /* yacc.c:1646  */
    break;

  case 147:
#line 545 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = (yyvsp[-1].val); }
#line 3186 "y.tab.c" /* yacc.c:1646  */
    break;

  case 148:
#line 549 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.vallist) = 0; }
#line 3192 "y.tab.c" /* yacc.c:1646  */
    break;

  case 149:
#line 551 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.vallist) = new ValueList((yyvsp[-1].val), (yyvsp[0].vallist)); }
#line 3198 "y.tab.c" /* yacc.c:1646  */
    break;

  case 150:
#line 555 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].vallist)); }
#line 3204 "y.tab.c" /* yacc.c:1646  */
    break;

  case 151:
#line 559 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[-3].type), (yyvsp[-1].strval), (yyvsp[0].vallist), 1); }
#line 3210 "y.tab.c" /* yacc.c:1646  */
    break;

  case 152:
#line 563 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0;}
#line 3216 "y.tab.c" /* yacc.c:1646  */
    break;

  case 153:
#line 565 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0;}
#line 3222 "y.tab.c" /* yacc.c:1646  */
    break;

  case 154:
#line 569 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3228 "y.tab.c" /* yacc.c:1646  */
    break;

  case 155:
#line 571 "xi-grammar.y" /* yacc.c:1646  */
    { 
		  /*
		  printf("Warning: Message attributes are being phased out.\n");
		  printf("Warning: Please remove them from interface files.\n");
		  */
		  (yyval.intval) = (yyvsp[-1].intval); 
		}
#line 3240 "y.tab.c" /* yacc.c:1646  */
    break;

  case 156:
#line 581 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[0].intval); }
#line 3246 "y.tab.c" /* yacc.c:1646  */
    break;

  case 157:
#line 583 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[-2].intval) | (yyvsp[0].intval); }
#line 3252 "y.tab.c" /* yacc.c:1646  */
    break;

  case 158:
#line 587 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3258 "y.tab.c" /* yacc.c:1646  */
    break;

  case 159:
#line 589 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3264 "y.tab.c" /* yacc.c:1646  */
    break;

  case 160:
#line 593 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = 0; }
#line 3270 "y.tab.c" /* yacc.c:1646  */
    break;

  case 161:
#line 595 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-1].cattr); }
#line 3276 "y.tab.c" /* yacc.c:1646  */
    break;

  case 162:
#line 599 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[0].cattr); }
#line 3282 "y.tab.c" /* yacc.c:1646  */
    break;

  case 163:
#line 601 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
#line 3288 "y.tab.c" /* yacc.c:1646  */
    break;

  case 164:
#line 605 "xi-grammar.y" /* yacc.c:1646  */
    { python_doc = NULL; (yyval.intval) = 0; }
#line 3294 "y.tab.c" /* yacc.c:1646  */
    break;

  case 165:
#line 607 "xi-grammar.y" /* yacc.c:1646  */
    { python_doc = (yyvsp[0].strval); (yyval.intval) = 0; }
#line 3300 "y.tab.c" /* yacc.c:1646  */
    break;

  case 166:
#line 611 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CPYTHON; }
#line 3306 "y.tab.c" /* yacc.c:1646  */
    break;

  case 167:
#line 615 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = 0; }
#line 3312 "y.tab.c" /* yacc.c:1646  */
    break;

  case 168:
#line 617 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-1].cattr); }
#line 3318 "y.tab.c" /* yacc.c:1646  */
    break;

  case 169:
#line 621 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[0].cattr); }
#line 3324 "y.tab.c" /* yacc.c:1646  */
    break;

  case 170:
#line 623 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
#line 3330 "y.tab.c" /* yacc.c:1646  */
    break;

  case 171:
#line 627 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CMIGRATABLE; }
#line 3336 "y.tab.c" /* yacc.c:1646  */
    break;

  case 172:
#line 629 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CPYTHON; }
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
    { (yyval.intval) = 0; }
#line 3360 "y.tab.c" /* yacc.c:1646  */
    break;

  case 176:
#line 640 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 3366 "y.tab.c" /* yacc.c:1646  */
    break;

  case 177:
#line 643 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mv) = new MsgVar((yyvsp[-3].type), (yyvsp[-2].strval), (yyvsp[-4].intval), (yyvsp[-1].intval)); }
#line 3372 "y.tab.c" /* yacc.c:1646  */
    break;

  case 178:
#line 647 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mvlist) = new MsgVarList((yyvsp[0].mv)); }
#line 3378 "y.tab.c" /* yacc.c:1646  */
    break;

  case 179:
#line 649 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mvlist) = new MsgVarList((yyvsp[-1].mv), (yyvsp[0].mvlist)); }
#line 3384 "y.tab.c" /* yacc.c:1646  */
    break;

  case 180:
#line 653 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, (yyvsp[0].ntype)); }
#line 3390 "y.tab.c" /* yacc.c:1646  */
    break;

  case 181:
#line 655 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, (yyvsp[-2].ntype)); }
#line 3396 "y.tab.c" /* yacc.c:1646  */
    break;

  case 182:
#line 657 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, (yyvsp[-3].ntype), (yyvsp[-1].mvlist)); }
#line 3402 "y.tab.c" /* yacc.c:1646  */
    break;

  case 183:
#line 661 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = 0; }
#line 3408 "y.tab.c" /* yacc.c:1646  */
    break;

  case 184:
#line 663 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = (yyvsp[0].typelist); }
#line 3414 "y.tab.c" /* yacc.c:1646  */
    break;

  case 185:
#line 667 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = new TypeList((yyvsp[0].ntype)); }
#line 3420 "y.tab.c" /* yacc.c:1646  */
    break;

  case 186:
#line 669 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = new TypeList((yyvsp[-2].ntype), (yyvsp[0].typelist)); }
#line 3426 "y.tab.c" /* yacc.c:1646  */
    break;

  case 187:
#line 673 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr)|Chare::CCHARE, (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3432 "y.tab.c" /* yacc.c:1646  */
    break;

  case 188:
#line 675 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3438 "y.tab.c" /* yacc.c:1646  */
    break;

  case 189:
#line 679 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3444 "y.tab.c" /* yacc.c:1646  */
    break;

  case 190:
#line 683 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new NodeGroup(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3450 "y.tab.c" /* yacc.c:1646  */
    break;

  case 191:
#line 687 "xi-grammar.y" /* yacc.c:1646  */
    {/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",(yyvsp[-2].strval));
			(yyval.ntype) = new NamedType(buf); 
		}
#line 3460 "y.tab.c" /* yacc.c:1646  */
    break;

  case 192:
#line 693 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ntype) = (yyvsp[-1].ntype); }
#line 3466 "y.tab.c" /* yacc.c:1646  */
    break;

  case 193:
#line 697 "xi-grammar.y" /* yacc.c:1646  */
    {  (yyval.chare) = new Array(lineno, (yyvsp[-4].cattr), (yyvsp[-3].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3472 "y.tab.c" /* yacc.c:1646  */
    break;

  case 194:
#line 699 "xi-grammar.y" /* yacc.c:1646  */
    {  (yyval.chare) = new Array(lineno, (yyvsp[-3].cattr), (yyvsp[-4].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3478 "y.tab.c" /* yacc.c:1646  */
    break;

  case 195:
#line 703 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr)|Chare::CCHARE, new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist));}
#line 3484 "y.tab.c" /* yacc.c:1646  */
    break;

  case 196:
#line 705 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3490 "y.tab.c" /* yacc.c:1646  */
    break;

  case 197:
#line 709 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3496 "y.tab.c" /* yacc.c:1646  */
    break;

  case 198:
#line 713 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new NodeGroup( lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3502 "y.tab.c" /* yacc.c:1646  */
    break;

  case 199:
#line 717 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Array( lineno, 0, (yyvsp[-3].ntype), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3508 "y.tab.c" /* yacc.c:1646  */
    break;

  case 200:
#line 721 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-1].strval))); }
#line 3514 "y.tab.c" /* yacc.c:1646  */
    break;

  case 201:
#line 723 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-4].strval)), (yyvsp[-2].mvlist)); }
#line 3520 "y.tab.c" /* yacc.c:1646  */
    break;

  case 202:
#line 727 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = 0; }
#line 3526 "y.tab.c" /* yacc.c:1646  */
    break;

  case 203:
#line 729 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3532 "y.tab.c" /* yacc.c:1646  */
    break;

  case 204:
#line 733 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = 0; }
#line 3538 "y.tab.c" /* yacc.c:1646  */
    break;

  case 205:
#line 735 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3544 "y.tab.c" /* yacc.c:1646  */
    break;

  case 206:
#line 737 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3550 "y.tab.c" /* yacc.c:1646  */
    break;

  case 207:
#line 739 "xi-grammar.y" /* yacc.c:1646  */
    {
		  XStr typeStr;
		  (yyvsp[0].ntype)->print(typeStr);
		  char *tmp = strdup(typeStr.get_string());
		  (yyval.strval) = tmp;
		}
#line 3561 "y.tab.c" /* yacc.c:1646  */
    break;

  case 208:
#line 748 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TTypeEllipsis(new NamedEllipsisType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3567 "y.tab.c" /* yacc.c:1646  */
    break;

  case 209:
#line 750 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TTypeEllipsis(new NamedEllipsisType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3573 "y.tab.c" /* yacc.c:1646  */
    break;

  case 210:
#line 752 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TType(new NamedType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3579 "y.tab.c" /* yacc.c:1646  */
    break;

  case 211:
#line 754 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TType(new NamedType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3585 "y.tab.c" /* yacc.c:1646  */
    break;

  case 212:
#line 756 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TFunc((yyvsp[-1].ftype), (yyvsp[0].strval)); }
#line 3591 "y.tab.c" /* yacc.c:1646  */
    break;

  case 213:
#line 758 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TName((yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].strval)); }
#line 3597 "y.tab.c" /* yacc.c:1646  */
    break;

  case 214:
#line 762 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = new TVarList((yyvsp[0].tvar)); }
#line 3603 "y.tab.c" /* yacc.c:1646  */
    break;

  case 215:
#line 764 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = new TVarList((yyvsp[-2].tvar), (yyvsp[0].tvarlist)); }
#line 3609 "y.tab.c" /* yacc.c:1646  */
    break;

  case 216:
#line 768 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = (yyvsp[-1].tvarlist); }
#line 3615 "y.tab.c" /* yacc.c:1646  */
    break;

  case 217:
#line 772 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3621 "y.tab.c" /* yacc.c:1646  */
    break;

  case 218:
#line 774 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3627 "y.tab.c" /* yacc.c:1646  */
    break;

  case 219:
#line 776 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3633 "y.tab.c" /* yacc.c:1646  */
    break;

  case 220:
#line 778 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3639 "y.tab.c" /* yacc.c:1646  */
    break;

  case 221:
#line 780 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].message)); (yyvsp[0].message)->setTemplate((yyval.templat)); }
#line 3645 "y.tab.c" /* yacc.c:1646  */
    break;

  case 222:
#line 784 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = 0; }
#line 3651 "y.tab.c" /* yacc.c:1646  */
    break;

  case 223:
#line 786 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = (yyvsp[-2].mbrlist); }
#line 3657 "y.tab.c" /* yacc.c:1646  */
    break;

  case 224:
#line 790 "xi-grammar.y" /* yacc.c:1646  */
    { 
                  if (!connectEntries.empty()) {
                    (yyval.mbrlist) = new AstChildren<Member>(connectEntries);
		  } else {
		    (yyval.mbrlist) = 0; 
                  }
		}
#line 3669 "y.tab.c" /* yacc.c:1646  */
    break;

  case 225:
#line 798 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = new AstChildren<Member>(-1, (yyvsp[-1].member), (yyvsp[0].mbrlist)); }
#line 3675 "y.tab.c" /* yacc.c:1646  */
    break;

  case 226:
#line 802 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].readonly); }
#line 3681 "y.tab.c" /* yacc.c:1646  */
    break;

  case 227:
#line 804 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].readonly); }
#line 3687 "y.tab.c" /* yacc.c:1646  */
    break;

  case 229:
#line 807 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].member); }
#line 3693 "y.tab.c" /* yacc.c:1646  */
    break;

  case 230:
#line 809 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].pupable); }
#line 3699 "y.tab.c" /* yacc.c:1646  */
    break;

  case 231:
#line 811 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].includeFile); }
#line 3705 "y.tab.c" /* yacc.c:1646  */
    break;

  case 232:
#line 813 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new ClassDeclaration(lineno,(yyvsp[0].strval)); }
#line 3711 "y.tab.c" /* yacc.c:1646  */
    break;

  case 233:
#line 817 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1); }
#line 3717 "y.tab.c" /* yacc.c:1646  */
    break;

  case 234:
#line 819 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1); }
#line 3723 "y.tab.c" /* yacc.c:1646  */
    break;

  case 235:
#line 821 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[-6].strval)) + '<' +
					    ((yyvsp[-4].tparlist))->to_string() + '>').c_str()),
				    1);
		}
#line 3733 "y.tab.c" /* yacc.c:1646  */
    break;

  case 236:
#line 827 "xi-grammar.y" /* yacc.c:1646  */
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[-2]).first_column, (yylsp[-2]).last_column, (yylsp[-2]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1);
		}
#line 3743 "y.tab.c" /* yacc.c:1646  */
    break;

  case 237:
#line 833 "xi-grammar.y" /* yacc.c:1646  */
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[-5]).first_column, (yylsp[-5]).last_column, (yylsp[-5]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1);
		}
#line 3753 "y.tab.c" /* yacc.c:1646  */
    break;

  case 238:
#line 842 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 0); }
#line 3759 "y.tab.c" /* yacc.c:1646  */
    break;

  case 239:
#line 844 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 0); }
#line 3765 "y.tab.c" /* yacc.c:1646  */
    break;

  case 240:
#line 846 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[-6].strval)) + '<' +
					    ((yyvsp[-4].tparlist))->to_string() + '>').c_str()),
				    0);
		}
#line 3775 "y.tab.c" /* yacc.c:1646  */
    break;

  case 241:
#line 852 "xi-grammar.y" /* yacc.c:1646  */
    {
                  InitCall* rtn = new InitCall(lineno, (yyvsp[-3].strval), 0);
                  rtn->setAccel();
                  (yyval.member) = rtn;
		}
#line 3785 "y.tab.c" /* yacc.c:1646  */
    break;

  case 242:
#line 860 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[0].ntype),0); }
#line 3791 "y.tab.c" /* yacc.c:1646  */
    break;

  case 243:
#line 862 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[-2].ntype),(yyvsp[0].pupable)); }
#line 3797 "y.tab.c" /* yacc.c:1646  */
    break;

  case 244:
#line 865 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.includeFile) = new IncludeFile(lineno,(yyvsp[0].strval)); }
#line 3803 "y.tab.c" /* yacc.c:1646  */
    break;

  case 245:
#line 869 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].member); }
#line 3809 "y.tab.c" /* yacc.c:1646  */
    break;

  case 246:
#line 873 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].entry); }
#line 3815 "y.tab.c" /* yacc.c:1646  */
    break;

  case 247:
#line 875 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyvsp[0].entry)->tspec = (yyvsp[-1].tvarlist);
                  (yyval.member) = (yyvsp[0].entry);
                }
#line 3824 "y.tab.c" /* yacc.c:1646  */
    break;

  case 248:
#line 880 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[-1].member); }
#line 3830 "y.tab.c" /* yacc.c:1646  */
    break;

  case 249:
#line 882 "xi-grammar.y" /* yacc.c:1646  */
    {
          ERROR("invalid SDAG member",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 3840 "y.tab.c" /* yacc.c:1646  */
    break;

  case 250:
#line 890 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3846 "y.tab.c" /* yacc.c:1646  */
    break;

  case 251:
#line 892 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3852 "y.tab.c" /* yacc.c:1646  */
    break;

  case 252:
#line 894 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3858 "y.tab.c" /* yacc.c:1646  */
    break;

  case 253:
#line 896 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3864 "y.tab.c" /* yacc.c:1646  */
    break;

  case 254:
#line 898 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3870 "y.tab.c" /* yacc.c:1646  */
    break;

  case 255:
#line 900 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3876 "y.tab.c" /* yacc.c:1646  */
    break;

  case 256:
#line 902 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3882 "y.tab.c" /* yacc.c:1646  */
    break;

  case 257:
#line 904 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3888 "y.tab.c" /* yacc.c:1646  */
    break;

  case 258:
#line 906 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3894 "y.tab.c" /* yacc.c:1646  */
    break;

  case 259:
#line 908 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3900 "y.tab.c" /* yacc.c:1646  */
    break;

  case 260:
#line 910 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3906 "y.tab.c" /* yacc.c:1646  */
    break;

  case 261:
#line 913 "xi-grammar.y" /* yacc.c:1646  */
    { 
                  (yyval.entry) = new Entry(lineno, (yyvsp[-5].intval), (yyvsp[-4].type), (yyvsp[-3].strval), (yyvsp[-2].plist), (yyvsp[-1].val), (yyvsp[0].sentry), (const char *) NULL, (yylsp[-6]).first_line, (yyloc).last_line);
		  if ((yyvsp[0].sentry) != 0) { 
		    (yyvsp[0].sentry)->con1 = new SdagConstruct(SIDENT, (yyvsp[-3].strval));
                    (yyvsp[0].sentry)->setEntry((yyval.entry));
                    (yyvsp[0].sentry)->param = new ParamList((yyvsp[-2].plist));
                  }
                  firstRdma = true;
                  firstDeviceRdma = true;
		}
#line 3921 "y.tab.c" /* yacc.c:1646  */
    break;

  case 262:
#line 924 "xi-grammar.y" /* yacc.c:1646  */
    { 
                  Entry *e = new Entry(lineno, (yyvsp[-3].intval), 0, (yyvsp[-2].strval), (yyvsp[-1].plist),  0, (yyvsp[0].sentry), (const char *) NULL, (yylsp[-4]).first_line, (yyloc).last_line);
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
#line 3943 "y.tab.c" /* yacc.c:1646  */
    break;

  case 263:
#line 942 "xi-grammar.y" /* yacc.c:1646  */
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
                  firstDeviceRdma = true;
                }
#line 3963 "y.tab.c" /* yacc.c:1646  */
    break;

  case 264:
#line 960 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.accelBlock) = new AccelBlock(lineno, new XStr((yyvsp[-2].strval))); }
#line 3969 "y.tab.c" /* yacc.c:1646  */
    break;

  case 265:
#line 962 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.accelBlock) = new AccelBlock(lineno, NULL); }
#line 3975 "y.tab.c" /* yacc.c:1646  */
    break;

  case 266:
#line 966 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3981 "y.tab.c" /* yacc.c:1646  */
    break;

  case 267:
#line 970 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3987 "y.tab.c" /* yacc.c:1646  */
    break;

  case 268:
#line 972 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[-1].intval); }
#line 3993 "y.tab.c" /* yacc.c:1646  */
    break;

  case 269:
#line 974 "xi-grammar.y" /* yacc.c:1646  */
    { ERROR("invalid entry method attribute list",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
#line 4002 "y.tab.c" /* yacc.c:1646  */
    break;

  case 270:
#line 981 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[0].intval); }
#line 4008 "y.tab.c" /* yacc.c:1646  */
    break;

  case 271:
#line 983 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[-2].intval) | (yyvsp[0].intval); }
#line 4014 "y.tab.c" /* yacc.c:1646  */
    break;

  case 272:
#line 987 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = STHREADED; }
#line 4020 "y.tab.c" /* yacc.c:1646  */
    break;

  case 273:
#line 989 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SWHENIDLE; }
#line 4026 "y.tab.c" /* yacc.c:1646  */
    break;

  case 274:
#line 991 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SSYNC; }
#line 4032 "y.tab.c" /* yacc.c:1646  */
    break;

  case 275:
#line 993 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SIGET; }
#line 4038 "y.tab.c" /* yacc.c:1646  */
    break;

  case 276:
#line 995 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SLOCKED; }
#line 4044 "y.tab.c" /* yacc.c:1646  */
    break;

  case 277:
#line 997 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SCREATEHERE; }
#line 4050 "y.tab.c" /* yacc.c:1646  */
    break;

  case 278:
#line 999 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SCREATEHOME; }
#line 4056 "y.tab.c" /* yacc.c:1646  */
    break;

  case 279:
#line 1001 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SNOKEEP; }
#line 4062 "y.tab.c" /* yacc.c:1646  */
    break;

  case 280:
#line 1003 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SNOTRACE; }
#line 4068 "y.tab.c" /* yacc.c:1646  */
    break;

  case 281:
#line 1005 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SAPPWORK; }
#line 4074 "y.tab.c" /* yacc.c:1646  */
    break;

  case 282:
#line 1007 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SIMMEDIATE; }
#line 4080 "y.tab.c" /* yacc.c:1646  */
    break;

  case 283:
#line 1009 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SSKIPSCHED; }
#line 4086 "y.tab.c" /* yacc.c:1646  */
    break;

  case 284:
#line 1011 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SINLINE; }
#line 4092 "y.tab.c" /* yacc.c:1646  */
    break;

  case 285:
#line 1013 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SLOCAL; }
#line 4098 "y.tab.c" /* yacc.c:1646  */
    break;

  case 286:
#line 1015 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SPYTHON; }
#line 4104 "y.tab.c" /* yacc.c:1646  */
    break;

  case 287:
#line 1017 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SMEM; }
#line 4110 "y.tab.c" /* yacc.c:1646  */
    break;

  case 288:
#line 1019 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SREDUCE; }
#line 4116 "y.tab.c" /* yacc.c:1646  */
    break;

  case 289:
#line 1021 "xi-grammar.y" /* yacc.c:1646  */
    {
        (yyval.intval) = SAGGREGATE;
    }
#line 4124 "y.tab.c" /* yacc.c:1646  */
    break;

  case 290:
#line 1025 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("invalid entry method attribute",
		        (yylsp[0]).first_column, (yylsp[0]).last_column);
		  yyclearin;
		  yyerrok;
		}
#line 4135 "y.tab.c" /* yacc.c:1646  */
    break;

  case 291:
#line 1034 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4141 "y.tab.c" /* yacc.c:1646  */
    break;

  case 292:
#line 1036 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4147 "y.tab.c" /* yacc.c:1646  */
    break;

  case 293:
#line 1038 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4153 "y.tab.c" /* yacc.c:1646  */
    break;

  case 294:
#line 1042 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = ""; }
#line 4159 "y.tab.c" /* yacc.c:1646  */
    break;

  case 295:
#line 1044 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4165 "y.tab.c" /* yacc.c:1646  */
    break;

  case 296:
#line 1046 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s, %s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4175 "y.tab.c" /* yacc.c:1646  */
    break;

  case 297:
#line 1054 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = ""; }
#line 4181 "y.tab.c" /* yacc.c:1646  */
    break;

  case 298:
#line 1056 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4187 "y.tab.c" /* yacc.c:1646  */
    break;

  case 299:
#line 1058 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s[%s]%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4197 "y.tab.c" /* yacc.c:1646  */
    break;

  case 300:
#line 1064 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s{%s}%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4207 "y.tab.c" /* yacc.c:1646  */
    break;

  case 301:
#line 1070 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s(%s)%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4217 "y.tab.c" /* yacc.c:1646  */
    break;

  case 302:
#line 1076 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"(%s)%s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4227 "y.tab.c" /* yacc.c:1646  */
    break;

  case 303:
#line 1084 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval));
		}
#line 4236 "y.tab.c" /* yacc.c:1646  */
    break;

  case 304:
#line 1091 "xi-grammar.y" /* yacc.c:1646  */
    { 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			(yyval.intval) = 0;
		}
#line 4246 "y.tab.c" /* yacc.c:1646  */
    break;

  case 305:
#line 1099 "xi-grammar.y" /* yacc.c:1646  */
    { 
			in_braces=0;
			(yyval.intval) = 0;
		}
#line 4255 "y.tab.c" /* yacc.c:1646  */
    break;

  case 306:
#line 1106 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[0].type));}
#line 4261 "y.tab.c" /* yacc.c:1646  */
    break;

  case 307:
#line 1108 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval)); (yyval.pname)->setConditional((yyvsp[0].intval)); }
#line 4267 "y.tab.c" /* yacc.c:1646  */
    break;

  case 308:
#line 1110 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[-3].type),(yyvsp[-2].strval),0,(yyvsp[0].val));}
#line 4273 "y.tab.c" /* yacc.c:1646  */
    break;

  case 309:
#line 1112 "xi-grammar.y" /* yacc.c:1646  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
		}
#line 4282 "y.tab.c" /* yacc.c:1646  */
    break;

  case 310:
#line 1117 "xi-grammar.y" /* yacc.c:1646  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
			(yyval.pname)->setRdma(CMK_ZC_P2P_SEND_MSG);
			if(firstRdma) {
				(yyval.pname)->setFirstRdma(true);
				firstRdma = false;
			}
		}
#line 4296 "y.tab.c" /* yacc.c:1646  */
    break;

  case 311:
#line 1127 "xi-grammar.y" /* yacc.c:1646  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
			(yyval.pname)->setRdma(CMK_ZC_P2P_RECV_MSG);
			if(firstRdma) {
				(yyval.pname)->setFirstRdma(true);
				firstRdma = false;
			}
		}
#line 4310 "y.tab.c" /* yacc.c:1646  */
    break;

  case 312:
#line 1137 "xi-grammar.y" /* yacc.c:1646  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
			(yyval.pname)->setRdma(CMK_ZC_DEVICE_MSG);
			if (firstDeviceRdma) {
				(yyval.pname)->setFirstDeviceRdma(true);
				firstDeviceRdma = false;
			}
		}
#line 4324 "y.tab.c" /* yacc.c:1646  */
    break;

  case 313:
#line 1148 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READONLY; }
#line 4330 "y.tab.c" /* yacc.c:1646  */
    break;

  case 314:
#line 1149 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READWRITE; }
#line 4336 "y.tab.c" /* yacc.c:1646  */
    break;

  case 315:
#line 1150 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_WRITEONLY; }
#line 4342 "y.tab.c" /* yacc.c:1646  */
    break;

  case 316:
#line 1153 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr((yyvsp[0].strval)); }
#line 4348 "y.tab.c" /* yacc.c:1646  */
    break;

  case 317:
#line 1154 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "->" << (yyvsp[0].strval); }
#line 4354 "y.tab.c" /* yacc.c:1646  */
    break;

  case 318:
#line 1155 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[-2].xstrptr)) << "." << (yyvsp[0].strval); }
#line 4360 "y.tab.c" /* yacc.c:1646  */
    break;

  case 319:
#line 1157 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << *((yyvsp[-1].xstrptr)) << "]";
                  delete (yyvsp[-3].xstrptr);
                  delete (yyvsp[-1].xstrptr);
                }
#line 4371 "y.tab.c" /* yacc.c:1646  */
    break;

  case 320:
#line 1164 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << (yyvsp[-1].strval) << "]";
                  delete (yyvsp[-3].xstrptr);
                }
#line 4381 "y.tab.c" /* yacc.c:1646  */
    break;

  case 321:
#line 1170 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "(" << *((yyvsp[-1].xstrptr)) << ")";
                  delete (yyvsp[-3].xstrptr);
                  delete (yyvsp[-1].xstrptr);
                }
#line 4392 "y.tab.c" /* yacc.c:1646  */
    break;

  case 322:
#line 1179 "xi-grammar.y" /* yacc.c:1646  */
    {
                  in_bracket = 0;
                  (yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName(), (yyvsp[-1].strval));
                }
#line 4401 "y.tab.c" /* yacc.c:1646  */
    break;

  case 323:
#line 1186 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[-6].intval));
                }
#line 4411 "y.tab.c" /* yacc.c:1646  */
    break;

  case 324:
#line 1192 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType(Parameter::ACCEL_BUFFER_TYPE_READWRITE);
		}
#line 4421 "y.tab.c" /* yacc.c:1646  */
    break;

  case 325:
#line 1198 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.pname) = (yyvsp[-3].pname);
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[-5].intval));
		}
#line 4431 "y.tab.c" /* yacc.c:1646  */
    break;

  case 326:
#line 1206 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 4437 "y.tab.c" /* yacc.c:1646  */
    break;

  case 327:
#line 1208 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 4443 "y.tab.c" /* yacc.c:1646  */
    break;

  case 328:
#line 1212 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 4449 "y.tab.c" /* yacc.c:1646  */
    break;

  case 329:
#line 1214 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 4455 "y.tab.c" /* yacc.c:1646  */
    break;

  case 330:
#line 1218 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = (yyvsp[-1].plist); }
#line 4461 "y.tab.c" /* yacc.c:1646  */
    break;

  case 331:
#line 1220 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList(new Parameter(0, new BuiltinType("void"))); }
#line 4467 "y.tab.c" /* yacc.c:1646  */
    break;

  case 332:
#line 1224 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = (yyvsp[-1].plist); }
#line 4473 "y.tab.c" /* yacc.c:1646  */
    break;

  case 333:
#line 1226 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = 0; }
#line 4479 "y.tab.c" /* yacc.c:1646  */
    break;

  case 334:
#line 1230 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = 0; }
#line 4485 "y.tab.c" /* yacc.c:1646  */
    break;

  case 335:
#line 1232 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4491 "y.tab.c" /* yacc.c:1646  */
    break;

  case 336:
#line 1236 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = 0; }
#line 4497 "y.tab.c" /* yacc.c:1646  */
    break;

  case 337:
#line 1238 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[0].sc)); }
#line 4503 "y.tab.c" /* yacc.c:1646  */
    break;

  case 338:
#line 1240 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[-2].slist)); }
#line 4509 "y.tab.c" /* yacc.c:1646  */
    break;

  case 339:
#line 1244 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.slist) = new SListConstruct((yyvsp[0].sc)); }
#line 4515 "y.tab.c" /* yacc.c:1646  */
    break;

  case 340:
#line 1246 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.slist) = new SListConstruct((yyvsp[-1].sc), (yyvsp[0].slist));  }
#line 4521 "y.tab.c" /* yacc.c:1646  */
    break;

  case 341:
#line 1250 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.olist) = new OListConstruct((yyvsp[0].sc)); }
#line 4527 "y.tab.c" /* yacc.c:1646  */
    break;

  case 342:
#line 1252 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.olist) = new OListConstruct((yyvsp[-1].sc), (yyvsp[0].slist)); }
#line 4533 "y.tab.c" /* yacc.c:1646  */
    break;

  case 343:
#line 1256 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.clist) = new CaseListConstruct((yyvsp[0].when)); }
#line 4539 "y.tab.c" /* yacc.c:1646  */
    break;

  case 344:
#line 1258 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.clist) = new CaseListConstruct((yyvsp[-1].when), (yyvsp[0].clist)); }
#line 4545 "y.tab.c" /* yacc.c:1646  */
    break;

  case 345:
#line 1260 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("case blocks can only contain when clauses",
		        (yylsp[0]).first_column, (yylsp[0]).last_column);
		  (yyval.clist) = 0;
		}
#line 4555 "y.tab.c" /* yacc.c:1646  */
    break;

  case 346:
#line 1268 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4561 "y.tab.c" /* yacc.c:1646  */
    break;

  case 347:
#line 1270 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = 0; }
#line 4567 "y.tab.c" /* yacc.c:1646  */
    break;

  case 348:
#line 1274 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-2].entrylist), 0); }
#line 4573 "y.tab.c" /* yacc.c:1646  */
    break;

  case 349:
#line 1276 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-1].entrylist), (yyvsp[0].sc)); }
#line 4579 "y.tab.c" /* yacc.c:1646  */
    break;

  case 350:
#line 1278 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-3].entrylist), (yyvsp[-1].slist)); }
#line 4585 "y.tab.c" /* yacc.c:1646  */
    break;

  case 351:
#line 1282 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4591 "y.tab.c" /* yacc.c:1646  */
    break;

  case 352:
#line 1284 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4597 "y.tab.c" /* yacc.c:1646  */
    break;

  case 353:
#line 1286 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4603 "y.tab.c" /* yacc.c:1646  */
    break;

  case 354:
#line 1288 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4609 "y.tab.c" /* yacc.c:1646  */
    break;

  case 355:
#line 1290 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4615 "y.tab.c" /* yacc.c:1646  */
    break;

  case 356:
#line 1292 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4621 "y.tab.c" /* yacc.c:1646  */
    break;

  case 357:
#line 1294 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4627 "y.tab.c" /* yacc.c:1646  */
    break;

  case 358:
#line 1296 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4633 "y.tab.c" /* yacc.c:1646  */
    break;

  case 359:
#line 1298 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4639 "y.tab.c" /* yacc.c:1646  */
    break;

  case 360:
#line 1300 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4645 "y.tab.c" /* yacc.c:1646  */
    break;

  case 361:
#line 1302 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4651 "y.tab.c" /* yacc.c:1646  */
    break;

  case 362:
#line 1304 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4657 "y.tab.c" /* yacc.c:1646  */
    break;

  case 363:
#line 1308 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new SerialConstruct((yyvsp[-2].strval), (yyvsp[-4].strval), (yylsp[-3]).first_line); }
#line 4663 "y.tab.c" /* yacc.c:1646  */
    break;

  case 364:
#line 1310 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new OverlapConstruct((yyvsp[-1].olist)); }
#line 4669 "y.tab.c" /* yacc.c:1646  */
    break;

  case 365:
#line 1312 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = (yyvsp[0].when); }
#line 4675 "y.tab.c" /* yacc.c:1646  */
    break;

  case 366:
#line 1314 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new CaseConstruct((yyvsp[-1].clist)); }
#line 4681 "y.tab.c" /* yacc.c:1646  */
    break;

  case 367:
#line 1316 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForConstruct((yyvsp[-8].intexpr), (yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4687 "y.tab.c" /* yacc.c:1646  */
    break;

  case 368:
#line 1318 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForConstruct((yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4693 "y.tab.c" /* yacc.c:1646  */
    break;

  case 369:
#line 1320 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-9].strval)), (yyvsp[-6].intexpr),
		             (yyvsp[-4].intexpr), (yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4700 "y.tab.c" /* yacc.c:1646  */
    break;

  case 370:
#line 1323 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-11].strval)), (yyvsp[-8].intexpr),
		             (yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4707 "y.tab.c" /* yacc.c:1646  */
    break;

  case 371:
#line 1326 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new IfConstruct((yyvsp[-3].intexpr), (yyvsp[-1].sc), (yyvsp[0].sc)); }
#line 4713 "y.tab.c" /* yacc.c:1646  */
    break;

  case 372:
#line 1328 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new IfConstruct((yyvsp[-5].intexpr), (yyvsp[-2].slist), (yyvsp[0].sc)); }
#line 4719 "y.tab.c" /* yacc.c:1646  */
    break;

  case 373:
#line 1330 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new WhileConstruct((yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4725 "y.tab.c" /* yacc.c:1646  */
    break;

  case 374:
#line 1332 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new WhileConstruct((yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4731 "y.tab.c" /* yacc.c:1646  */
    break;

  case 375:
#line 1334 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new SerialConstruct((yyvsp[-2].strval), NULL, (yyloc).first_line); }
#line 4737 "y.tab.c" /* yacc.c:1646  */
    break;

  case 376:
#line 1336 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("unknown SDAG construct or malformed entry method declaration.\n"
		        "You may have forgotten to terminate a previous entry method declaration with a"
		        " semicolon or forgotten to mark a block of sequential SDAG code as 'serial'",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
#line 4749 "y.tab.c" /* yacc.c:1646  */
    break;

  case 377:
#line 1346 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = 0; }
#line 4755 "y.tab.c" /* yacc.c:1646  */
    break;

  case 378:
#line 1348 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ElseConstruct((yyvsp[0].sc)); }
#line 4761 "y.tab.c" /* yacc.c:1646  */
    break;

  case 379:
#line 1350 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ElseConstruct((yyvsp[-1].slist)); }
#line 4767 "y.tab.c" /* yacc.c:1646  */
    break;

  case 380:
#line 1354 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intexpr) = new IntExprConstruct((yyvsp[0].strval)); }
#line 4773 "y.tab.c" /* yacc.c:1646  */
    break;

  case 381:
#line 1358 "xi-grammar.y" /* yacc.c:1646  */
    { in_int_expr = 0; (yyval.intval) = 0; }
#line 4779 "y.tab.c" /* yacc.c:1646  */
    break;

  case 382:
#line 1362 "xi-grammar.y" /* yacc.c:1646  */
    { in_int_expr = 1; (yyval.intval) = 0; }
#line 4785 "y.tab.c" /* yacc.c:1646  */
    break;

  case 383:
#line 1366 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[-1].strval), (yyvsp[0].plist), 0, 0, 0, (yyloc).first_line, (yyloc).last_line);
		  firstRdma = true;
		  firstDeviceRdma = true;
		}
#line 4795 "y.tab.c" /* yacc.c:1646  */
    break;

  case 384:
#line 1372 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[-4].strval), (yyvsp[0].plist), 0, 0, (yyvsp[-2].strval), (yyloc).first_line, (yyloc).last_line);
		  firstRdma = true;
		  firstDeviceRdma = true;
		}
#line 4805 "y.tab.c" /* yacc.c:1646  */
    break;

  case 385:
#line 1380 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.entrylist) = new EntryList((yyvsp[0].entry)); }
#line 4811 "y.tab.c" /* yacc.c:1646  */
    break;

  case 386:
#line 1382 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.entrylist) = new EntryList((yyvsp[-2].entry),(yyvsp[0].entrylist)); }
#line 4817 "y.tab.c" /* yacc.c:1646  */
    break;

  case 387:
#line 1386 "xi-grammar.y" /* yacc.c:1646  */
    { in_bracket=1; }
#line 4823 "y.tab.c" /* yacc.c:1646  */
    break;

  case 388:
#line 1389 "xi-grammar.y" /* yacc.c:1646  */
    { in_bracket=0; }
#line 4829 "y.tab.c" /* yacc.c:1646  */
    break;

  case 389:
#line 1393 "xi-grammar.y" /* yacc.c:1646  */
    { if (!macroDefined((yyvsp[0].strval), 1)) in_comment = 1; }
#line 4835 "y.tab.c" /* yacc.c:1646  */
    break;

  case 390:
#line 1397 "xi-grammar.y" /* yacc.c:1646  */
    { if (!macroDefined((yyvsp[0].strval), 0)) in_comment = 1; }
#line 4841 "y.tab.c" /* yacc.c:1646  */
    break;


#line 4845 "y.tab.c" /* yacc.c:1646  */
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
#line 1400 "xi-grammar.y" /* yacc.c:1906  */


void yyerror(const char *s) 
{
	fprintf(stderr, "[PARSE-ERROR] Unexpected/missing token at line %d. Current token being parsed: '%s'.\n", lineno, yytext);
}
