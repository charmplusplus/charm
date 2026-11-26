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
    INBOX = 293,
    COROUTINE = 294,
    VOID = 295,
    CONST = 296,
    NOCOPY = 297,
    NOCOPYPOST = 298,
    NOCOPYDEVICE = 299,
    PACKED = 300,
    VARSIZE = 301,
    ENTRY = 302,
    FOR = 303,
    FORALL = 304,
    WHILE = 305,
    WHEN = 306,
    OVERLAP = 307,
    SERIAL = 308,
    IF = 309,
    ELSE = 310,
    PYTHON = 311,
    LOCAL = 312,
    NAMESPACE = 313,
    USING = 314,
    IDENT = 315,
    NUMBER = 316,
    LITERAL = 317,
    CPROGRAM = 318,
    HASHIF = 319,
    HASHIFDEF = 320,
    INT = 321,
    LONG = 322,
    SHORT = 323,
    CHAR = 324,
    FLOAT = 325,
    DOUBLE = 326,
    UNSIGNED = 327,
    ACCEL = 328,
    READWRITE = 329,
    WRITEONLY = 330,
    ACCELBLOCK = 331,
    MEMCRITICAL = 332,
    REDUCTIONTARGET = 333,
    CASE = 334,
    TYPENAME = 335
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
#define INBOX 293
#define COROUTINE 294
#define VOID 295
#define CONST 296
#define NOCOPY 297
#define NOCOPYPOST 298
#define NOCOPYDEVICE 299
#define PACKED 300
#define VARSIZE 301
#define ENTRY 302
#define FOR 303
#define FORALL 304
#define WHILE 305
#define WHEN 306
#define OVERLAP 307
#define SERIAL 308
#define IF 309
#define ELSE 310
#define PYTHON 311
#define LOCAL 312
#define NAMESPACE 313
#define USING 314
#define IDENT 315
#define NUMBER 316
#define LITERAL 317
#define CPROGRAM 318
#define HASHIF 319
#define HASHIFDEF 320
#define INT 321
#define LONG 322
#define SHORT 323
#define CHAR 324
#define FLOAT 325
#define DOUBLE 326
#define UNSIGNED 327
#define ACCEL 328
#define READWRITE 329
#define WRITEONLY 330
#define ACCELBLOCK 331
#define MEMCRITICAL 332
#define REDUCTIONTARGET 333
#define CASE 334
#define TYPENAME 335

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

#line 362 "y.tab.c" /* yacc.c:355  */
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

#line 393 "y.tab.c" /* yacc.c:358  */

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
#define YYLAST   1574

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  97
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  119
/* YYNRULES -- Number of rules.  */
#define YYNRULES  402
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  795

/* YYTRANSLATE[YYX] -- Symbol number corresponding to YYX as returned
   by yylex, with out-of-bounds checking.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   335

#define YYTRANSLATE(YYX)                                                \
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, without out-of-bounds checking.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    91,     2,
      89,    90,    88,     2,    85,    96,    92,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    82,    81,
      86,    95,    87,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    93,     2,    94,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    83,     2,    84,     2,     2,     2,     2,
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
      75,    76,    77,    78,    79,    80
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
     295,   296,   300,   302,   309,   317,   321,   328,   330,   335,
     336,   340,   342,   344,   346,   348,   362,   364,   366,   368,
     374,   376,   378,   380,   382,   384,   386,   388,   390,   392,
     400,   402,   404,   408,   410,   415,   416,   421,   422,   426,
     428,   430,   432,   434,   436,   438,   440,   442,   444,   446,
     448,   450,   452,   454,   456,   458,   460,   462,   464,   468,
     469,   474,   482,   484,   488,   492,   494,   498,   502,   504,
     506,   508,   510,   512,   516,   518,   520,   522,   524,   528,
     530,   532,   534,   536,   538,   542,   544,   546,   548,   550,
     552,   556,   560,   565,   566,   570,   574,   579,   580,   585,
     586,   596,   598,   602,   604,   609,   610,   614,   616,   621,
     622,   626,   631,   632,   636,   638,   642,   644,   649,   650,
     654,   655,   658,   662,   664,   668,   670,   672,   677,   678,
     682,   684,   688,   690,   694,   698,   702,   708,   712,   714,
     718,   720,   724,   728,   732,   736,   738,   743,   744,   749,
     750,   752,   754,   763,   765,   767,   769,   771,   773,   777,
     779,   783,   787,   789,   791,   793,   795,   799,   801,   806,
     813,   817,   819,   821,   822,   824,   826,   828,   832,   834,
     836,   842,   848,   857,   859,   861,   867,   875,   877,   880,
     884,   888,   890,   895,   897,   905,   907,   909,   911,   913,
     915,   917,   919,   921,   923,   925,   928,   939,   957,   975,
     977,   981,   986,   987,   989,   996,  1000,  1001,  1005,  1006,
    1007,  1008,  1011,  1013,  1015,  1017,  1019,  1021,  1023,  1025,
    1027,  1029,  1031,  1033,  1035,  1037,  1039,  1041,  1043,  1045,
    1047,  1049,  1053,  1062,  1064,  1066,  1071,  1072,  1074,  1084,
    1085,  1087,  1094,  1101,  1108,  1117,  1124,  1132,  1139,  1141,
    1143,  1145,  1150,  1160,  1170,  1182,  1183,  1184,  1187,  1188,
    1189,  1190,  1197,  1203,  1212,  1219,  1225,  1231,  1239,  1241,
    1245,  1247,  1251,  1253,  1257,  1259,  1264,  1265,  1269,  1271,
    1273,  1277,  1279,  1283,  1285,  1289,  1291,  1293,  1301,  1304,
    1307,  1309,  1311,  1315,  1317,  1319,  1321,  1323,  1325,  1327,
    1329,  1331,  1333,  1335,  1337,  1341,  1343,  1345,  1347,  1349,
    1351,  1353,  1356,  1359,  1361,  1363,  1365,  1367,  1369,  1380,
    1381,  1383,  1387,  1391,  1395,  1399,  1405,  1413,  1415,  1419,
    1422,  1426,  1430
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
  "NOKEEP", "NOTRACE", "APPWORK", "INBOX", "COROUTINE", "VOID", "CONST",
  "NOCOPY", "NOCOPYPOST", "NOCOPYDEVICE", "PACKED", "VARSIZE", "ENTRY",
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
     325,   326,   327,   328,   329,   330,   331,   332,   333,   334,
     335,    59,    58,   123,   125,    44,    60,    62,    42,    40,
      41,    38,    46,    91,    93,    61,    45
};
# endif

#define YYPACT_NINF -599

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-599)))

#define YYTABLE_NINF -354

#define yytable_value_is_error(Yytable_value) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
     205,  1370,  1370,   121,  -599,   205,  -599,  -599,  -599,  -599,
    -599,  -599,  -599,  -599,  -599,  -599,  -599,  -599,  -599,  -599,
    -599,  -599,  -599,  -599,  -599,  -599,  -599,  -599,  -599,  -599,
    -599,  -599,  -599,  -599,  -599,  -599,  -599,  -599,  -599,  -599,
    -599,  -599,  -599,  -599,  -599,  -599,  -599,  -599,  -599,  -599,
    -599,  -599,  -599,  -599,  -599,  -599,  -599,    91,    91,  -599,
    -599,  -599,   919,    52,  -599,  -599,  -599,   101,  1370,   225,
    1370,  1370,   207,  1071,    77,   360,   919,  -599,  -599,  -599,
    -599,   206,    81,   160,  -599,   107,  -599,  -599,  -599,    52,
       5,   652,   195,   195,    -7,    78,   159,   159,   159,   159,
     174,   177,  1370,   218,   236,   919,  -599,  -599,  -599,  -599,
    -599,  -599,  -599,  -599,   411,  -599,  -599,  -599,  -599,   245,
    -599,  -599,  -599,  -599,  -599,  -599,  -599,  -599,  -599,  -599,
    -599,    52,  -599,  -599,  -599,  1254,  1479,   919,   107,   257,
     122,     5,   262,  1178,  -599,  1494,  -599,   198,   275,  -599,
    -599,  -599,   332,   160,   162,  -599,  -599,   260,   268,   291,
    -599,    25,   160,  -599,   160,   160,   286,   160,   305,  -599,
      22,  1370,  1370,  1370,  1370,   123,   298,   310,   252,  1370,
    -599,  -599,  -599,  1413,   320,   159,   159,   159,   159,   298,
     177,  -599,  -599,  -599,  -599,  -599,    52,  -599,  -599,  -599,
    -599,  -599,  -599,  -599,  -599,  -599,  -599,  -599,  -599,  -599,
    -599,  -599,  -599,  -599,  -599,   350,  -599,  -599,  -599,   319,
     165,  1479,   260,   268,   291,    85,  -599,    78,   330,    37,
       5,   352,     5,   326,  -599,   245,   343,    10,  -599,   366,
    -599,  -599,  -599,   271,   367,  -599,   162,  1051,  -599,  -599,
    -599,  -599,  -599,   348,   270,   345,    58,   115,   -28,   324,
      70,    78,  -599,  -599,   344,   354,   362,   368,   368,   368,
     368,  -599,  1370,   358,   380,   372,   -17,  1370,   412,  1370,
    -599,  -599,   373,   384,   391,   829,    62,   147,  1370,   385,
     388,   245,  1370,  1370,  1370,  1370,  1370,  1370,  -599,  -599,
    -599,  1254,  1370,   435,  -599,   292,   396,  1370,  -599,  -599,
    -599,   395,   406,   402,   413,     5,    52,   160,  -599,  -599,
     434,  -599,  -599,  -599,  -599,   419,  -599,   421,  -599,  1370,
     414,   417,   418,  -599,   420,  -599,     5,   195,  1051,   195,
     195,  1051,   195,  -599,  -599,    22,  -599,    78,   229,   229,
     229,   229,   423,  -599,   412,  -599,   368,   368,  -599,   252,
      14,   422,   416,   158,   436,   138,  -599,   432,  1413,  -599,
    -599,   368,   368,   368,   368,   368,   244,  -599,   448,   430,
     443,   442,   446,   450,   362,     5,   352,     5,     5,  -599,
      58,  -599,  1051,  -599,   445,   451,   452,  -599,  -599,   453,
    -599,   455,   459,   461,   160,   465,   463,  -599,   472,  -599,
     387,    52,  -599,  -599,  -599,  -599,  -599,  -599,   229,   229,
    -599,  -599,  -599,  1494,    31,   456,   464,  1494,  -599,  -599,
     466,  -599,  -599,  -599,  -599,  -599,   229,   229,   229,   229,
     229,   542,    52,   499,  1370,   476,   473,   475,  -599,   474,
    -599,  -599,  -599,  -599,  -599,  -599,   481,   479,  -599,  -599,
    -599,  -599,   483,  -599,   130,   484,  -599,    78,  -599,   746,
     527,   502,   245,   387,  -599,  -599,  -599,  -599,  1370,  -599,
    -599,  1370,  -599,   528,  -599,  -599,  -599,  -599,  -599,   503,
    -599,  -599,  1254,   497,  -599,  1446,  -599,  1461,  -599,   195,
     195,   195,  -599,  1196,  1136,  -599,   245,    52,  -599,   498,
     416,   416,   245,  -599,  -599,  1494,  1494,  1494,  -599,  1370,
       5,   505,   506,   507,   508,   510,   511,   512,   515,   474,
    1370,  -599,   514,   245,  -599,  -599,    52,  1370,     5,     5,
       5,    15,   516,  1461,  -599,  -599,  -599,  -599,  -599,   552,
     593,   474,  -599,    52,   518,   521,   523,   524,  -599,   258,
    -599,  -599,  -599,  1370,  -599,   531,   519,   531,   545,   536,
     559,   531,   540,   281,    52,     5,  -599,  -599,  -599,   604,
    -599,  -599,  -599,  -599,  -599,   107,  -599,   474,  -599,     5,
     565,     5,   220,   543,   606,   650,  -599,   544,     5,   433,
     546,   471,   262,   534,   593,   538,  -599,   553,   539,   548,
    -599,     5,   545,   410,  -599,   551,   530,     5,   548,   531,
     547,   531,   556,   559,   531,   566,     5,   564,   433,  -599,
     245,  -599,   245,   575,  -599,   283,   544,     5,   531,  -599,
     896,   453,  -599,  -599,   568,  -599,  -599,   262,   903,     5,
     590,     5,   650,   544,     5,   433,   262,  -599,  -599,  -599,
    -599,  -599,  -599,  -599,  -599,  -599,  1370,   579,   577,   569,
       5,   583,     5,   281,  -599,   474,  -599,   245,   281,   611,
     586,   581,   548,   585,     5,   548,   587,   245,   591,  1494,
    1396,  -599,   262,     5,   597,   596,  -599,  -599,   598,   910,
    -599,     5,   531,   917,  -599,   262,   971,  -599,  -599,  1370,
    1370,     5,   595,  -599,  1370,   548,     5,  -599,   611,   281,
    -599,   602,     5,   281,  -599,   245,   281,   611,  -599,    99,
     170,   592,  1370,   245,   979,   599,  -599,   603,     5,   608,
     607,  -599,   610,  -599,  -599,  1370,  1370,  1293,   601,  1370,
    -599,   293,    52,   281,  -599,     5,  -599,   548,     5,  -599,
     611,   215,  -599,   612,   353,  1370,   409,  -599,   621,   548,
     986,   622,  -599,  -599,  -599,  -599,  -599,  -599,  -599,   993,
     281,  -599,     5,   281,  -599,   626,   548,   627,  -599,  1047,
    -599,   281,  -599,   630,  -599
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
       8,     0,     0,     0,    62,    72,   401,   402,   316,   270,
     309,     0,   157,   157,   157,     0,   165,   165,   165,   165,
       0,   159,     0,     0,     0,     0,    80,   231,   232,    74,
      81,    82,    83,    84,     0,    85,    73,   234,   233,     9,
     265,   257,   258,   259,   260,   261,   263,   264,   262,   255,
     256,    78,    79,    70,   274,     0,     0,     0,    71,     0,
     310,   309,     0,     0,   118,     0,    99,   100,   102,   104,
     115,   116,     0,     0,    97,   122,   123,   128,   129,   130,
     131,   150,     0,   158,     0,     0,     0,     0,   247,   235,
       0,     0,     0,     0,     0,     0,     0,   172,     0,     0,
     237,   249,   236,     0,     0,   165,   165,   165,   165,     0,
     159,   222,   223,   224,   225,   226,    10,    68,   302,   282,
     283,   284,   285,   286,   294,   295,   296,   301,   287,   288,
     289,   292,   293,   290,   291,   169,   297,   299,   300,     0,
     278,     0,   134,   135,   136,   144,   271,     0,     0,     0,
     309,   306,   309,     0,   317,     0,     0,   132,   101,   113,
     117,   103,   105,   106,   110,   112,    97,    95,   120,   124,
     125,   126,   133,     0,   149,     0,   153,   241,   238,     0,
     243,     0,   176,   177,     0,   167,    97,   188,   188,   188,
     188,   171,     0,     0,   174,     0,     0,     0,     0,     0,
     163,   164,     0,   161,   185,     0,     0,   131,     0,   219,
       0,     9,     0,     0,     0,     0,     0,     0,   170,   298,
     273,     0,     0,   137,   138,   143,     0,     0,    77,    64,
      63,     0,   307,     0,     0,   309,   269,     0,   114,   107,
     108,   111,   121,    91,    92,    93,    96,     0,    90,     0,
     148,     0,     0,   399,   153,   155,   309,   157,     0,   157,
     157,     0,   157,   248,   166,     0,   119,     0,     0,     0,
       0,     0,     0,   197,     0,   173,   188,   188,   160,     0,
     178,     0,   207,    62,     0,     0,   217,   209,     0,   221,
      76,   188,   188,   188,   188,   188,     0,   280,     0,   276,
       0,   142,     0,     0,    97,   309,   306,   309,   309,   314,
     153,   109,     0,    98,     0,     0,     0,   147,   154,     0,
     151,     0,     0,     0,     0,     0,     0,   168,   190,   189,
       0,   227,   192,   193,   194,   195,   196,   175,     0,     0,
     162,   179,   186,     0,   178,     0,     0,     0,   215,   216,
       0,   210,   211,   212,   218,   220,     0,     0,     0,     0,
       0,   178,   205,     0,     0,   279,     0,     0,   141,     0,
     312,   308,   313,   311,   156,    94,     0,     0,   146,   400,
     152,   242,     0,   239,     0,     0,   244,     0,   254,     0,
       0,     0,     0,     0,   250,   251,   198,   199,     0,   184,
     187,     0,   208,     0,   200,   201,   202,   203,   204,     0,
     275,   277,     0,     0,   140,     0,    75,     0,   145,   157,
     157,   157,   191,     0,     0,   252,     9,   253,   230,   180,
     207,   207,     0,   281,   139,     0,     0,     0,   343,   318,
     309,   338,     0,     0,     0,     0,     0,     0,    62,     0,
       0,   228,     0,     0,   213,   214,   206,     0,   309,   309,
     309,   178,     0,     0,   342,   127,   240,   246,   245,     0,
       0,     0,   181,   182,     0,     0,     0,     0,   315,     0,
     319,   321,   339,     0,   388,     0,     0,     0,     0,     0,
     359,     0,     0,     0,   348,   309,   267,   377,   349,   346,
     322,   323,   324,   304,   303,   305,   320,     0,   394,   309,
       0,   309,     0,   397,     0,     0,   358,     0,   309,     0,
       0,     0,     0,     0,     0,     0,   392,     0,     0,     0,
     395,   309,     0,     0,   361,     0,     0,   309,     0,     0,
       0,     0,     0,   359,     0,     0,   309,     0,   355,   357,
       9,   352,     9,     0,   266,     0,     0,   309,     0,   393,
       0,     0,   398,   360,     0,   376,   354,     0,     0,   309,
       0,   309,     0,     0,   309,     0,     0,   378,   356,   350,
     387,   347,   325,   326,   327,   345,     0,     0,   340,     0,
     309,     0,   309,     0,   385,     0,   362,     9,     0,   389,
       0,     0,     0,     0,   309,     0,     0,     9,     0,     0,
       0,   344,     0,   309,     0,     0,   396,   375,     0,     0,
     383,   309,     0,     0,   364,     0,     0,   365,   374,     0,
       0,   309,     0,   341,     0,     0,   309,   386,   389,     0,
     390,     0,   309,     0,   372,     9,     0,   389,   328,     0,
       0,     0,     0,     0,     0,     0,   384,     0,   309,     0,
       0,   363,     0,   370,   336,     0,     0,     0,     0,     0,
     334,     0,   268,     0,   380,   309,   391,     0,   309,   373,
     389,     0,   330,     0,     0,     0,     0,   337,     0,     0,
       0,     0,   371,   333,   332,   331,   329,   335,   379,     0,
       0,   367,   309,     0,   381,     0,     0,     0,   366,     0,
     382,     0,   368,     0,   369
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -599,  -599,   690,  -599,   -55,  -285,    -1,   -62,   623,   657,
     -50,  -599,  -599,  -599,  -279,  -599,  -221,  -599,  -133,   -86,
    -128,  -126,  -117,  -167,   571,   496,  -599,   -87,  -599,  -599,
    -298,  -599,  -599,   -80,   535,   369,  -599,   129,   381,  -599,
    -599,   550,   376,  -599,   193,  -599,  -599,  -364,  -599,  -134,
     269,  -599,  -599,  -599,   -46,  -599,  -599,  -599,  -599,  -599,
    -599,  -345,   370,  -599,   371,   662,  -599,  -269,   265,   667,
    -599,  -599,   480,  -599,  -599,  -599,  -599,   272,  -599,   239,
     287,  -599,   308,  -290,  -599,  -599,   377,   -85,  -493,   -60,
    -568,  -599,  -599,  -516,  -599,  -599,  -425,    72,  -502,  -599,
    -599,   180,  -566,   126,  -590,   179,  -571,  -599,  -521,  -598,
    -561,  -576,  -496,  -599,   173,   212,   166,  -599,  -599
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     3,     4,    73,   411,   197,   266,   154,     5,    64,
      74,    75,    76,   325,   326,   327,   248,   155,   267,   156,
     157,   158,   159,   160,   161,   225,   226,   328,   399,   334,
     335,   107,   108,   164,   179,   282,   283,   171,   264,   299,
     274,   176,   275,   265,   423,   533,   424,   425,   109,   348,
     409,   110,   111,   112,   177,   113,   191,   192,   193,   194,
     195,   428,   366,   289,   290,   470,   115,   412,   471,   472,
     117,   118,   169,   182,   473,   474,   132,   475,    77,   227,
     136,   379,   380,   219,   220,   586,   313,   606,   520,   575,
     235,   521,   667,   729,   712,   668,   522,   669,   496,   636,
     604,   576,   600,   615,   627,   597,   577,   629,   601,   700,
     607,   640,   589,   593,   594,   336,   460,    78,    79
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[] =
{
      57,    58,    63,    63,   162,   142,   370,    85,   222,   168,
     223,   377,    90,   165,   167,   236,   287,    89,   429,   224,
     131,   138,   538,   539,   540,   322,   133,   550,   628,   578,
     609,   421,   421,   163,   632,   631,   398,   618,   658,   268,
     269,   270,   648,    84,   272,   346,   284,   644,   421,   579,
     646,   252,   309,   262,   139,   184,   233,   628,   338,   402,
     479,   339,   405,   153,   196,   686,   252,    82,   140,    86,
      87,   591,   523,   614,   616,   598,   671,   489,   263,   677,
     413,   414,   415,   578,   628,   605,   166,   228,   687,   273,
     610,   246,   454,   222,   141,   223,   288,   310,   422,   253,
     256,   180,   257,   258,   224,   260,   703,   695,   558,   706,
     559,   694,   698,   455,   253,  -183,   254,   255,   562,   674,
     736,    59,   363,   649,   714,   651,   304,   679,   654,   743,
     277,   616,   715,    80,   349,   350,   351,   725,    84,   734,
     139,   307,   672,   296,   356,   311,   357,   314,    81,   476,
     477,   333,   139,   737,   364,   735,   341,   740,   153,   342,
     742,   119,   772,   449,   137,   534,   535,   484,   485,   486,
     487,   488,    61,   696,    62,   168,   305,   306,   720,   271,
     316,   770,   724,    84,   272,   727,   744,   768,   745,   139,
     273,   746,   747,   779,   769,   748,   711,   139,    84,   431,
     432,   287,   513,   153,   337,   230,   722,   134,     1,     2,
     789,   231,   139,   754,   785,   232,   751,   787,   153,   500,
      84,   531,   418,   419,   246,   793,   172,   173,   174,   761,
     389,   764,  -209,   766,  -209,   163,   196,   436,   437,   438,
     439,   440,   365,  -207,   139,  -207,  -272,  -272,   247,   781,
     301,   400,   170,   427,   302,   390,   749,   401,   784,   403,
     404,   408,   406,   558,   238,   239,  -272,   175,   792,   240,
     178,   352,  -272,  -272,  -272,  -272,  -272,  -272,  -272,   433,
     181,   288,   564,    83,   362,    84,  -272,   367,    61,   662,
      88,   371,   372,   373,   374,   375,   376,   280,   281,   135,
     450,   378,   452,   453,   745,   773,   384,   746,   747,   495,
      61,   748,   410,   333,   292,   293,   294,   295,    84,   583,
     584,   442,   183,   144,   145,    61,    61,   441,   394,   565,
     566,   567,   568,   569,   570,   571,   478,   319,   320,   229,
     482,   241,   464,    84,  -316,   659,   234,   660,   249,   146,
     147,   148,   149,   150,   151,   152,   250,   663,   664,   259,
     572,   330,   331,   153,    88,  -316,   120,   121,   122,   123,
    -316,   124,   125,   126,   127,   128,   222,   665,   223,   251,
     767,   408,   745,   381,   382,   746,   747,   224,   468,   748,
     261,   276,   697,    91,    92,    93,    94,    95,   242,   243,
     244,   245,   708,   278,   291,   102,   103,   129,   519,   104,
     519,   564,   298,   300,   308,   312,   315,   507,   340,   524,
     525,   526,   185,   186,   187,   188,   189,   190,   537,   537,
     537,   317,   318,   321,   469,   542,   329,   332,   344,   345,
     741,    61,   745,   378,   130,   746,   747,   775,   247,   748,
     347,   196,   353,   555,   556,   557,   519,   536,   565,   566,
     567,   568,   569,   570,   571,   354,   355,   358,   271,   359,
     368,  -229,   564,  -316,   360,   369,   304,   509,   553,   385,
     510,   619,   620,   621,   568,   622,   623,   624,   383,   572,
     602,   386,   387,    88,   643,   574,   777,   585,   745,  -316,
     391,   746,   747,   529,   392,   748,   395,   388,   393,   396,
     397,   427,   625,   333,   426,   444,    88,   416,   541,   565,
     566,   567,   568,   569,   570,   571,   641,   365,   430,   551,
     443,   564,   647,   445,   446,   456,   554,   617,   447,   626,
     480,   656,   448,   457,   458,   461,   462,   459,   666,   574,
     572,   463,   465,   466,    88,  -351,   481,   467,   483,   421,
     490,   492,   587,   495,   680,   493,   682,   494,   626,   685,
     497,   498,   499,   501,   469,   196,   670,   196,   565,   566,
     567,   568,   569,   570,   571,   692,   506,   512,   511,   514,
     543,   532,   563,   684,   564,   626,   544,   545,   546,   705,
     547,   548,   710,   666,   -11,   592,   549,   564,   552,   572,
     561,   558,   590,    88,  -353,   580,   721,   581,   582,   595,
     588,   596,   196,   599,   603,   608,   731,    88,   612,   633,
     630,   635,   196,   638,   637,   645,   661,   739,   639,   652,
     650,   565,   566,   567,   568,   569,   570,   571,   657,   655,
     681,   564,   676,   757,   565,   566,   567,   568,   569,   570,
     571,   689,   690,   691,   693,   688,   699,   701,   143,   704,
     196,   707,   572,   771,    61,   702,   573,   709,   752,   716,
     717,   732,   718,   738,   755,   572,   750,   756,   765,   613,
     758,   759,   144,   145,   760,    60,   106,   786,   565,   566,
     567,   568,   569,   570,   571,   778,   774,   782,   728,   730,
     788,   790,    84,   733,   794,    65,   237,   303,   146,   147,
     148,   149,   150,   151,   152,   297,   407,   279,   420,   572,
     417,   728,   153,    88,   560,   114,   502,   434,   508,   435,
     116,   343,   505,   530,   728,   762,   728,   134,   728,  -272,
    -272,  -272,   491,  -272,  -272,  -272,   504,  -272,  -272,  -272,
    -272,  -272,   713,   451,   776,  -272,  -272,  -272,  -272,  -272,
    -272,  -272,  -272,  -272,  -272,  -272,  -272,  -272,   683,  -272,
    -272,  -272,  -272,  -272,   634,   642,  -272,  -272,  -272,  -272,
    -272,  -272,  -272,  -272,  -272,  -272,  -272,  -272,  -272,  -272,
    -272,  -272,   653,  -272,   611,  -272,  -272,   675,     0,     0,
       0,     0,  -272,  -272,  -272,  -272,  -272,  -272,  -272,  -272,
       0,     0,  -272,  -272,  -272,  -272,  -272,     0,     0,     0,
       0,     0,     6,     7,     8,     0,     9,    10,    11,   503,
      12,    13,    14,    15,    16,     0,     0,     0,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,     0,    30,    31,    32,    33,    34,     0,     0,     0,
       0,    35,    36,    37,    38,    39,    40,    41,    42,    43,
      44,    45,    46,    47,    48,     0,    49,     0,    50,    51,
       0,     0,     0,     0,     0,     0,     0,   564,     0,     0,
       0,     0,    52,     0,   564,    53,    54,    55,    56,     0,
       0,   564,     0,     0,     0,     0,     0,     0,   564,     0,
      66,   361,    -5,    -5,    67,    -5,    -5,    -5,    -5,    -5,
      -5,    -5,    -5,    -5,    -5,    -5,     0,    -5,    -5,     0,
       0,    -5,     0,     0,   565,   566,   567,   568,   569,   570,
     571,   565,   566,   567,   568,   569,   570,   571,   565,   566,
     567,   568,   569,   570,   571,   565,   566,   567,   568,   569,
     570,   571,   564,     0,     0,   572,     0,    68,    69,   673,
     564,     0,   572,    70,    71,     0,   678,   564,     0,   572,
       0,     0,     0,   719,   564,    72,   572,     0,     0,     0,
     723,     0,    -5,   -69,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   565,
     566,   567,   568,   569,   570,   571,     0,   565,   566,   567,
     568,   569,   570,   571,   565,   566,   567,   568,   569,   570,
     571,   565,   566,   567,   568,   569,   570,   571,   564,     0,
     572,     0,     0,     0,   726,     0,     0,     0,   572,     0,
       0,     0,   753,     0,     0,   572,     0,     0,     0,   780,
       0,     0,   572,     0,     1,     2,   783,    91,    92,    93,
      94,    95,    96,    97,    98,    99,   100,   101,     0,   102,
     103,   144,   145,   104,     0,   565,   566,   567,   568,   569,
     570,   571,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    84,   323,   324,     0,     0,     0,   146,   147,   148,
     149,   150,   151,   152,     0,     0,   572,     0,     0,     0,
     791,   153,     0,     0,     0,     0,     0,     0,     0,     6,
       7,     8,     0,     9,    10,    11,     0,    12,    13,    14,
      15,    16,     0,     0,   105,    17,    18,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,     0,    30,
      31,    32,    33,    34,     0,     0,   144,   221,    35,    36,
      37,    38,    39,    40,    41,    42,    43,    44,    45,    46,
      47,    48,     0,    49,     0,    50,   528,   198,     0,     0,
       0,     0,   146,   147,   148,   149,   150,   151,   152,    52,
       0,     0,    53,    54,    55,    56,   153,   199,   144,   200,
     201,   202,   203,   204,   205,   206,     0,     0,   207,   208,
     209,   210,   211,   212,   213,   214,     0,     0,    84,     0,
       0,     0,     0,     0,   146,   147,   148,   149,   150,   151,
     152,     0,   215,   216,     0,   198,     0,     0,   153,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   527,
       0,     0,     0,   217,   218,   199,     0,   200,   201,   202,
     203,   204,   205,   206,     0,     0,   207,   208,   209,   210,
     211,   212,   213,   214,     0,     0,     6,     7,     8,     0,
       9,    10,    11,     0,    12,    13,    14,    15,    16,     0,
     215,   216,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,     0,    30,    31,    32,    33,
      34,   217,   218,     0,     0,    35,    36,    37,    38,    39,
      40,    41,    42,    43,    44,    45,    46,    47,    48,     0,
      49,     0,    50,    51,   763,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    52,     0,     0,    53,
      54,    55,    56,     6,     7,     8,     0,     9,    10,    11,
       0,    12,    13,    14,    15,    16,     0,     0,     0,    17,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      28,    29,   662,    30,    31,    32,    33,    34,     0,     0,
       0,     0,    35,    36,    37,    38,    39,    40,    41,    42,
      43,    44,    45,    46,    47,    48,     0,    49,     0,    50,
      51,   285,     0,     0,     0,     0,   144,   145,     0,     0,
       0,     0,     0,    52,     0,     0,    53,    54,    55,    56,
       0,     0,     0,   144,   145,     0,    84,     0,     0,     0,
       0,     0,   146,   147,   148,   149,   150,   151,   152,     0,
     663,   664,     0,    84,     0,     0,   153,     0,     0,   146,
     147,   148,   149,   150,   151,   152,   144,   145,   515,   516,
     517,     0,     0,   286,     0,     0,     0,     0,     0,     0,
       0,   144,   145,   515,   516,   517,    84,     0,     0,     0,
       0,     0,   146,   147,   148,   149,   150,   151,   152,   144,
     221,    84,     0,     0,     0,     0,   153,   146,   147,   148,
     149,   150,   151,   152,   144,   145,   518,     0,     0,    84,
       0,   153,     0,     0,     0,   146,   147,   148,   149,   150,
     151,   152,     0,     0,    84,     0,     0,     0,     0,   153,
     146,   147,   148,   149,   150,   151,   152,     0,     0,     0,
       0,     0,     0,     0,   153
};

static const yytype_int16 yycheck[] =
{
       1,     2,    57,    58,    91,    90,   291,    69,   136,    95,
     136,   301,    72,    93,    94,   143,   183,    72,   363,   136,
      75,    83,   515,   516,   517,   246,    76,   529,   599,   550,
     591,    17,    17,    40,   602,   601,   334,   598,   628,   172,
     173,   174,   618,    60,    61,   266,   179,   613,    17,   551,
     616,    41,    15,    31,    82,   105,   141,   628,    86,   338,
     424,    89,   341,    80,   119,   655,    41,    68,    63,    70,
      71,   567,   497,   594,   595,   571,   637,   441,    56,   647,
     349,   350,   351,   604,   655,   587,    93,   137,   656,   175,
     592,   153,   390,   221,    89,   221,   183,    60,    84,    89,
     162,   102,   164,   165,   221,   167,   682,   673,    93,   685,
      95,   672,   678,   392,    89,    84,    91,    92,   543,   640,
     718,     0,    60,   619,   692,   621,    41,   648,   624,   727,
     176,   652,   693,    81,   268,   269,   270,   705,    60,   715,
      82,   227,   638,   189,   277,   230,   279,   232,    47,   418,
     419,    93,    82,   719,    92,   716,    86,   723,    80,    89,
     726,    84,   760,   384,    83,   510,   511,   436,   437,   438,
     439,   440,    81,   675,    83,   261,    91,    92,   699,    56,
     235,   757,   703,    60,    61,   706,    87,   753,    89,    82,
     276,    92,    93,   769,   755,    96,   689,    82,    60,    61,
      62,   368,   492,    80,    89,    83,   702,     1,     3,     4,
     786,    89,    82,   734,   780,    93,   732,   783,    80,    89,
      60,   506,   356,   357,   286,   791,    97,    98,    99,   745,
     315,   747,    85,   749,    87,    40,   291,   371,   372,   373,
     374,   375,    95,    85,    82,    87,    40,    41,    86,   770,
      85,   336,    93,    95,    89,   317,    86,   337,   779,   339,
     340,   347,   342,    93,    66,    67,    60,    93,   789,    71,
      93,   272,    66,    67,    68,    69,    70,    71,    72,   365,
      62,   368,     1,    58,   285,    60,    80,   288,    81,     6,
      83,   292,   293,   294,   295,   296,   297,    45,    46,    93,
     385,   302,   387,   388,    89,    90,   307,    92,    93,    89,
      81,    96,    83,    93,   185,   186,   187,   188,    60,    61,
      62,   376,    86,    40,    41,    81,    81,    83,   329,    48,
      49,    50,    51,    52,    53,    54,   423,    66,    67,    82,
     427,    66,   404,    60,    63,   630,    84,   632,    88,    66,
      67,    68,    69,    70,    71,    72,    88,    74,    75,    73,
      79,    91,    92,    80,    83,    84,     6,     7,     8,     9,
      89,    11,    12,    13,    14,    15,   504,    94,   504,    88,
      87,   467,    89,    91,    92,    92,    93,   504,     1,    96,
      85,    93,   677,     6,     7,     8,     9,    10,    66,    67,
      68,    69,   687,    93,    84,    18,    19,    47,   495,    22,
     497,     1,    62,    94,    84,    63,    90,   472,    94,   499,
     500,   501,    11,    12,    13,    14,    15,    16,   515,   516,
     517,    88,    66,    66,    47,   520,    88,    92,    94,    85,
     725,    81,    89,   444,    84,    92,    93,    94,    86,    96,
      82,   506,    94,   538,   539,   540,   543,   512,    48,    49,
      50,    51,    52,    53,    54,    85,    94,    94,    56,    85,
      85,    84,     1,    63,    83,    87,    41,   478,   533,    84,
     481,    48,    49,    50,    51,    52,    53,    54,    92,    79,
     575,    85,    90,    83,    84,   550,    87,   559,    89,    89,
      66,    92,    93,   504,    85,    96,    92,    94,    87,    92,
      92,    95,    79,    93,    92,    85,    83,    94,   519,    48,
      49,    50,    51,    52,    53,    54,   611,    95,    92,   530,
      82,     1,   617,    90,    92,    90,   537,   597,    92,   599,
      84,   626,    92,    92,    92,    90,    87,    94,   635,   604,
      79,    90,    87,    90,    83,    84,    92,    85,    92,    17,
      61,    85,   563,    89,   649,    92,   651,    92,   628,   654,
      89,    92,    89,    89,    47,   630,   636,   632,    48,    49,
      50,    51,    52,    53,    54,   670,    84,    84,    60,    92,
      85,    93,    40,   653,     1,   655,    90,    90,    90,   684,
      90,    90,   689,   690,    89,    60,    94,     1,    94,    79,
      94,    93,    93,    83,    84,    94,   701,    94,    94,    83,
      89,    62,   677,    83,    20,    60,   711,    83,    85,    95,
      84,    93,   687,    94,    81,    84,    61,   722,    90,    83,
      93,    48,    49,    50,    51,    52,    53,    54,    84,    83,
      60,     1,    84,   738,    48,    49,    50,    51,    52,    53,
      54,    82,    85,    94,    81,   666,    55,    81,    16,    84,
     725,    84,    79,   758,    81,    94,    83,    86,   733,    82,
      84,    86,    84,    81,    85,    79,    94,    84,    87,    83,
      82,    84,    40,    41,    84,     5,    73,   782,    48,    49,
      50,    51,    52,    53,    54,    84,    94,    85,   709,   710,
      84,    84,    60,   714,    84,    58,   145,   221,    66,    67,
      68,    69,    70,    71,    72,   190,   345,   177,   359,    79,
     354,   732,    80,    83,   541,    73,   467,   367,   473,   368,
      73,   261,   470,   504,   745,   746,   747,     1,   749,     3,
       4,     5,   444,     7,     8,     9,   469,    11,    12,    13,
      14,    15,   690,   386,   765,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,    29,    30,    31,   652,    33,
      34,    35,    36,    37,   604,   612,    40,    41,    42,    43,
      44,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      54,    55,   623,    57,   592,    59,    60,   641,    -1,    -1,
      -1,    -1,    66,    67,    68,    69,    70,    71,    72,    73,
      -1,    -1,    76,    77,    78,    79,    80,    -1,    -1,    -1,
      -1,    -1,     3,     4,     5,    -1,     7,     8,     9,    93,
      11,    12,    13,    14,    15,    -1,    -1,    -1,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    -1,    33,    34,    35,    36,    37,    -1,    -1,    -1,
      -1,    42,    43,    44,    45,    46,    47,    48,    49,    50,
      51,    52,    53,    54,    55,    -1,    57,    -1,    59,    60,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,    -1,
      -1,    -1,    73,    -1,     1,    76,    77,    78,    79,    -1,
      -1,     1,    -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,
       1,    92,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    -1,    18,    19,    -1,
      -1,    22,    -1,    -1,    48,    49,    50,    51,    52,    53,
      54,    48,    49,    50,    51,    52,    53,    54,    48,    49,
      50,    51,    52,    53,    54,    48,    49,    50,    51,    52,
      53,    54,     1,    -1,    -1,    79,    -1,    58,    59,    83,
       1,    -1,    79,    64,    65,    -1,    83,     1,    -1,    79,
      -1,    -1,    -1,    83,     1,    76,    79,    -1,    -1,    -1,
      83,    -1,    83,    84,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    48,
      49,    50,    51,    52,    53,    54,    -1,    48,    49,    50,
      51,    52,    53,    54,    48,    49,    50,    51,    52,    53,
      54,    48,    49,    50,    51,    52,    53,    54,     1,    -1,
      79,    -1,    -1,    -1,    83,    -1,    -1,    -1,    79,    -1,
      -1,    -1,    83,    -1,    -1,    79,    -1,    -1,    -1,    83,
      -1,    -1,    79,    -1,     3,     4,    83,     6,     7,     8,
       9,    10,    11,    12,    13,    14,    15,    16,    -1,    18,
      19,    40,    41,    22,    -1,    48,    49,    50,    51,    52,
      53,    54,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    60,    61,    62,    -1,    -1,    -1,    66,    67,    68,
      69,    70,    71,    72,    -1,    -1,    79,    -1,    -1,    -1,
      83,    80,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     3,
       4,     5,    -1,     7,     8,     9,    -1,    11,    12,    13,
      14,    15,    -1,    -1,    83,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,    29,    30,    31,    -1,    33,
      34,    35,    36,    37,    -1,    -1,    40,    41,    42,    43,
      44,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      54,    55,    -1,    57,    -1,    59,    60,     1,    -1,    -1,
      -1,    -1,    66,    67,    68,    69,    70,    71,    72,    73,
      -1,    -1,    76,    77,    78,    79,    80,    21,    40,    23,
      24,    25,    26,    27,    28,    29,    -1,    -1,    32,    33,
      34,    35,    36,    37,    38,    39,    -1,    -1,    60,    -1,
      -1,    -1,    -1,    -1,    66,    67,    68,    69,    70,    71,
      72,    -1,    56,    57,    -1,     1,    -1,    -1,    80,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    73,
      -1,    -1,    -1,    77,    78,    21,    -1,    23,    24,    25,
      26,    27,    28,    29,    -1,    -1,    32,    33,    34,    35,
      36,    37,    38,    39,    -1,    -1,     3,     4,     5,    -1,
       7,     8,     9,    -1,    11,    12,    13,    14,    15,    -1,
      56,    57,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    -1,    33,    34,    35,    36,
      37,    77,    78,    -1,    -1,    42,    43,    44,    45,    46,
      47,    48,    49,    50,    51,    52,    53,    54,    55,    -1,
      57,    -1,    59,    60,    61,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    73,    -1,    -1,    76,
      77,    78,    79,     3,     4,     5,    -1,     7,     8,     9,
      -1,    11,    12,    13,    14,    15,    -1,    -1,    -1,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    31,     6,    33,    34,    35,    36,    37,    -1,    -1,
      -1,    -1,    42,    43,    44,    45,    46,    47,    48,    49,
      50,    51,    52,    53,    54,    55,    -1,    57,    -1,    59,
      60,    18,    -1,    -1,    -1,    -1,    40,    41,    -1,    -1,
      -1,    -1,    -1,    73,    -1,    -1,    76,    77,    78,    79,
      -1,    -1,    -1,    40,    41,    -1,    60,    -1,    -1,    -1,
      -1,    -1,    66,    67,    68,    69,    70,    71,    72,    -1,
      74,    75,    -1,    60,    -1,    -1,    80,    -1,    -1,    66,
      67,    68,    69,    70,    71,    72,    40,    41,    42,    43,
      44,    -1,    -1,    80,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    40,    41,    42,    43,    44,    60,    -1,    -1,    -1,
      -1,    -1,    66,    67,    68,    69,    70,    71,    72,    40,
      41,    60,    -1,    -1,    -1,    -1,    80,    66,    67,    68,
      69,    70,    71,    72,    40,    41,    90,    -1,    -1,    60,
      -1,    80,    -1,    -1,    -1,    66,    67,    68,    69,    70,
      71,    72,    -1,    -1,    60,    -1,    -1,    -1,    -1,    80,
      66,    67,    68,    69,    70,    71,    72,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    80
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     3,     4,    98,    99,   105,     3,     4,     5,     7,
       8,     9,    11,    12,    13,    14,    15,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      33,    34,    35,    36,    37,    42,    43,    44,    45,    46,
      47,    48,    49,    50,    51,    52,    53,    54,    55,    57,
      59,    60,    73,    76,    77,    78,    79,   103,   103,     0,
      99,    81,    83,   101,   106,   106,     1,     5,    58,    59,
      64,    65,    76,   100,   107,   108,   109,   175,   214,   215,
      81,    47,   103,    58,    60,   104,   103,   103,    83,   101,
     186,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    18,    19,    22,    83,   105,   128,   129,   145,
     148,   149,   150,   152,   162,   163,   166,   167,   168,    84,
       6,     7,     8,     9,    11,    12,    13,    14,    15,    47,
      84,   101,   173,   107,     1,    93,   177,    83,   104,    82,
      63,    89,   184,    16,    40,    41,    66,    67,    68,    69,
      70,    71,    72,    80,   104,   114,   116,   117,   118,   119,
     120,   121,   124,    40,   130,   130,    93,   130,   116,   169,
      93,   134,   134,   134,   134,    93,   138,   151,    93,   131,
     103,    62,   170,    86,   107,    11,    12,    13,    14,    15,
      16,   153,   154,   155,   156,   157,   101,   102,     1,    21,
      23,    24,    25,    26,    27,    28,    29,    32,    33,    34,
      35,    36,    37,    38,    39,    56,    57,    77,    78,   180,
     181,    41,   117,   118,   119,   122,   123,   176,   107,    82,
      83,    89,    93,   184,    84,   187,   117,   121,    66,    67,
      71,    66,    66,    67,    68,    69,   104,    86,   113,    88,
      88,    88,    41,    89,    91,    92,   104,   104,   104,    73,
     104,    85,    31,    56,   135,   140,   103,   115,   115,   115,
     115,    56,    61,   116,   137,   139,    93,   151,    93,   138,
      45,    46,   132,   133,   115,    18,    80,   120,   124,   160,
     161,    84,   134,   134,   134,   134,   151,   131,    62,   136,
      94,    85,    89,   122,    41,    91,    92,   116,    84,    15,
      60,   184,    63,   183,   184,    90,   101,    88,    66,    66,
      67,    66,   113,    61,    62,   110,   111,   112,   124,    88,
      91,    92,    92,    93,   126,   127,   212,    89,    86,    89,
      94,    86,    89,   169,    94,    85,   113,    82,   146,   146,
     146,   146,   103,    94,    85,    94,   115,   115,    94,    85,
      83,    92,   103,    60,    92,    95,   159,   103,    85,    87,
     102,   103,   103,   103,   103,   103,   103,   180,   103,   178,
     179,    91,    92,    92,   103,    84,    85,    90,    94,   184,
     104,    66,    85,    87,   103,    92,    92,    92,   127,   125,
     184,   130,   111,   130,   130,   111,   130,   135,   116,   147,
      83,   101,   164,   164,   164,   164,    94,   139,   146,   146,
     132,    17,    84,   141,   143,   144,    92,    95,   158,   158,
      92,    61,    62,   116,   159,   161,   146,   146,   146,   146,
     146,    83,   101,    82,    85,    90,    92,    92,    92,   113,
     184,   183,   184,   184,   127,   111,    90,    92,    92,    94,
     213,    90,    87,    90,   104,    87,    90,    85,     1,    47,
     162,   165,   166,   171,   172,   174,   164,   164,   124,   144,
      84,    92,   124,    92,   164,   164,   164,   164,   164,   144,
      61,   179,    85,    92,    92,    89,   195,    89,    92,    89,
      89,    89,   147,    93,   177,   174,    84,   101,   165,   103,
     103,    60,    84,   180,    92,    42,    43,    44,    90,   124,
     185,   188,   193,   193,   130,   130,   130,    73,    60,   103,
     176,   102,    93,   142,   158,   158,   101,   124,   185,   185,
     185,   103,   184,    85,    90,    90,    90,    90,    90,    94,
     195,   103,    94,   101,   103,   184,   184,   184,    93,    95,
     141,    94,   193,    40,     1,    48,    49,    50,    51,    52,
      53,    54,    79,    83,   101,   186,   198,   203,   205,   195,
      94,    94,    94,    61,    62,   104,   182,   103,    89,   209,
      93,   209,    60,   210,   211,    83,    62,   202,   209,    83,
     199,   205,   184,    20,   197,   195,   184,   207,    60,   207,
     195,   212,    85,    83,   205,   200,   205,   186,   207,    48,
      49,    50,    52,    53,    54,    79,   186,   201,   203,   204,
      84,   199,   187,    95,   198,    93,   196,    81,    94,    90,
     208,   184,   211,    84,   199,    84,   199,   184,   208,   209,
      93,   209,    83,   202,   209,    83,   184,    84,   201,   102,
     102,    61,     6,    74,    75,    94,   124,   189,   192,   194,
     186,   207,   209,    83,   205,   213,    84,   187,    83,   205,
     184,    60,   184,   200,   186,   184,   201,   187,   103,    82,
      85,    94,   184,    81,   207,   199,   195,   102,   199,    55,
     206,    81,    94,   208,    84,   184,   208,    84,   102,    86,
     124,   185,   191,   194,   187,   207,    82,    84,    84,    83,
     205,   184,   209,    83,   205,   187,    83,   205,   103,   190,
     103,   184,    86,   103,   208,   207,   206,   199,    81,   184,
     199,   102,   199,   206,    87,    89,    92,    93,    96,    86,
      94,   190,   101,    83,   205,    85,    84,   184,    82,    84,
      84,   190,   103,    61,   190,    87,   190,    87,   199,   207,
     208,   184,   206,    90,    94,    94,   103,    87,    84,   208,
      83,   205,    85,    83,   205,   199,   184,   199,    84,   208,
      84,    83,   205,   199,    84
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    97,    98,    99,    99,   100,   100,   101,   101,   102,
     102,   103,   103,   103,   103,   103,   103,   103,   103,   103,
     103,   103,   103,   103,   103,   103,   103,   103,   103,   103,
     103,   103,   103,   103,   103,   103,   103,   103,   103,   103,
     103,   103,   103,   103,   103,   103,   103,   103,   103,   103,
     103,   103,   103,   103,   103,   103,   103,   103,   103,   103,
     103,   103,   104,   104,   104,   105,   105,   106,   106,   107,
     107,   108,   108,   108,   108,   108,   109,   109,   109,   109,
     109,   109,   109,   109,   109,   109,   109,   109,   109,   109,
     110,   110,   110,   111,   111,   112,   112,   113,   113,   114,
     114,   114,   114,   114,   114,   114,   114,   114,   114,   114,
     114,   114,   114,   114,   114,   114,   114,   114,   114,   115,
     116,   116,   117,   117,   118,   119,   119,   120,   121,   121,
     121,   121,   121,   121,   122,   122,   122,   122,   122,   123,
     123,   123,   123,   123,   123,   124,   124,   124,   124,   124,
     124,   125,   126,   127,   127,   128,   129,   130,   130,   131,
     131,   132,   132,   133,   133,   134,   134,   135,   135,   136,
     136,   137,   138,   138,   139,   139,   140,   140,   141,   141,
     142,   142,   143,   144,   144,   145,   145,   145,   146,   146,
     147,   147,   148,   148,   149,   150,   151,   151,   152,   152,
     153,   153,   154,   155,   156,   157,   157,   158,   158,   159,
     159,   159,   159,   160,   160,   160,   160,   160,   160,   161,
     161,   162,   163,   163,   163,   163,   163,   164,   164,   165,
     165,   166,   166,   166,   166,   166,   166,   166,   167,   167,
     167,   167,   167,   168,   168,   168,   168,   169,   169,   170,
     171,   172,   172,   172,   172,   173,   173,   173,   173,   173,
     173,   173,   173,   173,   173,   173,   174,   174,   174,   175,
     175,   176,   177,   177,   177,   178,   179,   179,   180,   180,
     180,   180,   181,   181,   181,   181,   181,   181,   181,   181,
     181,   181,   181,   181,   181,   181,   181,   181,   181,   181,
     181,   181,   181,   182,   182,   182,   183,   183,   183,   184,
     184,   184,   184,   184,   184,   185,   186,   187,   188,   188,
     188,   188,   188,   188,   188,   189,   189,   189,   190,   190,
     190,   190,   190,   190,   191,   192,   192,   192,   193,   193,
     194,   194,   195,   195,   196,   196,   197,   197,   198,   198,
     198,   199,   199,   200,   200,   201,   201,   201,   202,   202,
     203,   203,   203,   204,   204,   204,   204,   204,   204,   204,
     204,   204,   204,   204,   204,   205,   205,   205,   205,   205,
     205,   205,   205,   205,   205,   205,   205,   205,   205,   206,
     206,   206,   207,   208,   209,   210,   210,   211,   211,   212,
     213,   214,   215
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
       1,     1,     1,     1,     1,     1,     1,     1,     2,     1,
       1,     1,     1,     1,     1,     1,     0,     1,     3,     0,
       1,     5,     5,     5,     4,     3,     1,     1,     1,     3,
       4,     3,     4,     4,     4,     1,     1,     1,     1,     4,
       3,     4,     4,     4,     3,     7,     5,     6,     1,     3,
       1,     3,     3,     2,     3,     2,     0,     3,     1,     1,
       4,     1,     2,     1,     2,     1,     2,     1,     1,     0,
       4,     3,     5,     6,     4,     4,    11,     9,    12,    14,
       6,     8,     5,     7,     4,     6,     4,     1,     4,    11,
       9,    12,    14,     6,     8,     5,     7,     4,     1,     0,
       2,     4,     1,     1,     1,     2,     5,     1,     3,     1,
       1,     2,     2
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
#line 2301 "y.tab.c" /* yacc.c:1646  */
    break;

  case 3:
#line 208 "xi-grammar.y" /* yacc.c:1646  */
    { 
		  (yyval.modlist) = 0; 
		}
#line 2309 "y.tab.c" /* yacc.c:1646  */
    break;

  case 4:
#line 212 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.modlist) = new AstChildren<Module>(lineno, (yyvsp[-1].module), (yyvsp[0].modlist)); }
#line 2315 "y.tab.c" /* yacc.c:1646  */
    break;

  case 5:
#line 216 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 2321 "y.tab.c" /* yacc.c:1646  */
    break;

  case 6:
#line 218 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 2327 "y.tab.c" /* yacc.c:1646  */
    break;

  case 7:
#line 222 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 2333 "y.tab.c" /* yacc.c:1646  */
    break;

  case 8:
#line 224 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 2; }
#line 2339 "y.tab.c" /* yacc.c:1646  */
    break;

  case 9:
#line 228 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 2345 "y.tab.c" /* yacc.c:1646  */
    break;

  case 10:
#line 230 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 2351 "y.tab.c" /* yacc.c:1646  */
    break;

  case 11:
#line 235 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 2357 "y.tab.c" /* yacc.c:1646  */
    break;

  case 12:
#line 236 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MODULE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2363 "y.tab.c" /* yacc.c:1646  */
    break;

  case 13:
#line 237 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MAINMODULE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2369 "y.tab.c" /* yacc.c:1646  */
    break;

  case 14:
#line 238 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(EXTERN, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2375 "y.tab.c" /* yacc.c:1646  */
    break;

  case 15:
#line 240 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITCALL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2381 "y.tab.c" /* yacc.c:1646  */
    break;

  case 16:
#line 241 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITNODE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2387 "y.tab.c" /* yacc.c:1646  */
    break;

  case 17:
#line 242 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITPROC, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2393 "y.tab.c" /* yacc.c:1646  */
    break;

  case 18:
#line 244 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CHARE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2399 "y.tab.c" /* yacc.c:1646  */
    break;

  case 19:
#line 245 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MAINCHARE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2405 "y.tab.c" /* yacc.c:1646  */
    break;

  case 20:
#line 246 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(GROUP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2411 "y.tab.c" /* yacc.c:1646  */
    break;

  case 21:
#line 247 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NODEGROUP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2417 "y.tab.c" /* yacc.c:1646  */
    break;

  case 22:
#line 248 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ARRAY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2423 "y.tab.c" /* yacc.c:1646  */
    break;

  case 23:
#line 252 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INCLUDE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2429 "y.tab.c" /* yacc.c:1646  */
    break;

  case 24:
#line 253 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(STACKSIZE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2435 "y.tab.c" /* yacc.c:1646  */
    break;

  case 25:
#line 254 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(THREADED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2441 "y.tab.c" /* yacc.c:1646  */
    break;

  case 26:
#line 255 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(TEMPLATE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2447 "y.tab.c" /* yacc.c:1646  */
    break;

  case 27:
#line 256 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(WHENIDLE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2453 "y.tab.c" /* yacc.c:1646  */
    break;

  case 28:
#line 257 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(SYNC, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2459 "y.tab.c" /* yacc.c:1646  */
    break;

  case 29:
#line 258 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IGET, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2465 "y.tab.c" /* yacc.c:1646  */
    break;

  case 30:
#line 259 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(EXCLUSIVE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2471 "y.tab.c" /* yacc.c:1646  */
    break;

  case 31:
#line 260 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IMMEDIATE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2477 "y.tab.c" /* yacc.c:1646  */
    break;

  case 32:
#line 261 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(SKIPSCHED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2483 "y.tab.c" /* yacc.c:1646  */
    break;

  case 33:
#line 262 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOCOPY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2489 "y.tab.c" /* yacc.c:1646  */
    break;

  case 34:
#line 263 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOCOPYPOST, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2495 "y.tab.c" /* yacc.c:1646  */
    break;

  case 35:
#line 264 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOCOPYDEVICE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2501 "y.tab.c" /* yacc.c:1646  */
    break;

  case 36:
#line 265 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INLINE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2507 "y.tab.c" /* yacc.c:1646  */
    break;

  case 37:
#line 266 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(VIRTUAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2513 "y.tab.c" /* yacc.c:1646  */
    break;

  case 38:
#line 267 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MIGRATABLE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2519 "y.tab.c" /* yacc.c:1646  */
    break;

  case 39:
#line 268 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CREATEHERE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2525 "y.tab.c" /* yacc.c:1646  */
    break;

  case 40:
#line 269 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CREATEHOME, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2531 "y.tab.c" /* yacc.c:1646  */
    break;

  case 41:
#line 270 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOKEEP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2537 "y.tab.c" /* yacc.c:1646  */
    break;

  case 42:
#line 271 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOTRACE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2543 "y.tab.c" /* yacc.c:1646  */
    break;

  case 43:
#line 272 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(APPWORK, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2549 "y.tab.c" /* yacc.c:1646  */
    break;

  case 44:
#line 275 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(PACKED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2555 "y.tab.c" /* yacc.c:1646  */
    break;

  case 45:
#line 276 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(VARSIZE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2561 "y.tab.c" /* yacc.c:1646  */
    break;

  case 46:
#line 277 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ENTRY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2567 "y.tab.c" /* yacc.c:1646  */
    break;

  case 47:
#line 278 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(FOR, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2573 "y.tab.c" /* yacc.c:1646  */
    break;

  case 48:
#line 279 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(FORALL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2579 "y.tab.c" /* yacc.c:1646  */
    break;

  case 49:
#line 280 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(WHILE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2585 "y.tab.c" /* yacc.c:1646  */
    break;

  case 50:
#line 281 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(WHEN, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2591 "y.tab.c" /* yacc.c:1646  */
    break;

  case 51:
#line 282 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(OVERLAP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2597 "y.tab.c" /* yacc.c:1646  */
    break;

  case 52:
#line 283 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(SERIAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2603 "y.tab.c" /* yacc.c:1646  */
    break;

  case 53:
#line 284 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IF, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2609 "y.tab.c" /* yacc.c:1646  */
    break;

  case 54:
#line 285 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ELSE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2615 "y.tab.c" /* yacc.c:1646  */
    break;

  case 55:
#line 287 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(LOCAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2621 "y.tab.c" /* yacc.c:1646  */
    break;

  case 56:
#line 289 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(USING, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2627 "y.tab.c" /* yacc.c:1646  */
    break;

  case 57:
#line 290 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ACCEL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2633 "y.tab.c" /* yacc.c:1646  */
    break;

  case 58:
#line 293 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ACCELBLOCK, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2639 "y.tab.c" /* yacc.c:1646  */
    break;

  case 59:
#line 294 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MEMCRITICAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2645 "y.tab.c" /* yacc.c:1646  */
    break;

  case 60:
#line 295 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(REDUCTIONTARGET, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2651 "y.tab.c" /* yacc.c:1646  */
    break;

  case 61:
#line 296 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CASE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2657 "y.tab.c" /* yacc.c:1646  */
    break;

  case 62:
#line 301 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 2663 "y.tab.c" /* yacc.c:1646  */
    break;

  case 63:
#line 303 "xi-grammar.y" /* yacc.c:1646  */
    {
		  int len = strlen((yyvsp[-3].strval))+strlen((yyvsp[0].strval))+3;
		  char *tmp = new char[len];
		  snprintf(tmp,len,"%s::%s", (yyvsp[-3].strval), (yyvsp[0].strval));
		  (yyval.strval) = tmp;
		}
#line 2674 "y.tab.c" /* yacc.c:1646  */
    break;

  case 64:
#line 310 "xi-grammar.y" /* yacc.c:1646  */
    {
		  int len = strlen((yyvsp[-3].strval))+5+3;
		  char *tmp = new char[len];
		  snprintf(tmp, len, "%s::array", (yyvsp[-3].strval));
		  (yyval.strval) = tmp;
		}
#line 2685 "y.tab.c" /* yacc.c:1646  */
    break;

  case 65:
#line 318 "xi-grammar.y" /* yacc.c:1646  */
    { 
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist)); 
		}
#line 2693 "y.tab.c" /* yacc.c:1646  */
    break;

  case 66:
#line 322 "xi-grammar.y" /* yacc.c:1646  */
    {  
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist)); 
		    (yyval.module)->setMain();
		}
#line 2702 "y.tab.c" /* yacc.c:1646  */
    break;

  case 67:
#line 329 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = 0; }
#line 2708 "y.tab.c" /* yacc.c:1646  */
    break;

  case 68:
#line 331 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = (yyvsp[-2].conslist); }
#line 2714 "y.tab.c" /* yacc.c:1646  */
    break;

  case 69:
#line 335 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = 0; }
#line 2720 "y.tab.c" /* yacc.c:1646  */
    break;

  case 70:
#line 337 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = new ConstructList(lineno, (yyvsp[-1].construct), (yyvsp[0].conslist)); }
#line 2726 "y.tab.c" /* yacc.c:1646  */
    break;

  case 71:
#line 341 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new UsingScope((yyvsp[0].strval), false); }
#line 2732 "y.tab.c" /* yacc.c:1646  */
    break;

  case 72:
#line 343 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new UsingScope((yyvsp[0].strval), true); }
#line 2738 "y.tab.c" /* yacc.c:1646  */
    break;

  case 73:
#line 345 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].member)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].member); }
#line 2744 "y.tab.c" /* yacc.c:1646  */
    break;

  case 74:
#line 347 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].message)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].message); }
#line 2750 "y.tab.c" /* yacc.c:1646  */
    break;

  case 75:
#line 349 "xi-grammar.y" /* yacc.c:1646  */
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
#line 2766 "y.tab.c" /* yacc.c:1646  */
    break;

  case 76:
#line 363 "xi-grammar.y" /* yacc.c:1646  */
    { if((yyvsp[-2].conslist)) (yyvsp[-2].conslist)->recurse<int&>((yyvsp[-4].intval), &Construct::setExtern); (yyval.construct) = (yyvsp[-2].conslist); }
#line 2772 "y.tab.c" /* yacc.c:1646  */
    break;

  case 77:
#line 365 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new Scope((yyvsp[-3].strval), (yyvsp[-1].conslist)); }
#line 2778 "y.tab.c" /* yacc.c:1646  */
    break;

  case 78:
#line 367 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = (yyvsp[-1].construct); }
#line 2784 "y.tab.c" /* yacc.c:1646  */
    break;

  case 79:
#line 369 "xi-grammar.y" /* yacc.c:1646  */
    {
          ERROR("preceding construct must be semicolon terminated",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 2794 "y.tab.c" /* yacc.c:1646  */
    break;

  case 80:
#line 375 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].module)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].module); }
#line 2800 "y.tab.c" /* yacc.c:1646  */
    break;

  case 81:
#line 377 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2806 "y.tab.c" /* yacc.c:1646  */
    break;

  case 82:
#line 379 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2812 "y.tab.c" /* yacc.c:1646  */
    break;

  case 83:
#line 381 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2818 "y.tab.c" /* yacc.c:1646  */
    break;

  case 84:
#line 383 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2824 "y.tab.c" /* yacc.c:1646  */
    break;

  case 85:
#line 385 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].templat)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].templat); }
#line 2830 "y.tab.c" /* yacc.c:1646  */
    break;

  case 86:
#line 387 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = NULL; }
#line 2836 "y.tab.c" /* yacc.c:1646  */
    break;

  case 87:
#line 389 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = NULL; }
#line 2842 "y.tab.c" /* yacc.c:1646  */
    break;

  case 88:
#line 391 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = (yyvsp[0].accelBlock); }
#line 2848 "y.tab.c" /* yacc.c:1646  */
    break;

  case 89:
#line 393 "xi-grammar.y" /* yacc.c:1646  */
    {
          ERROR("invalid construct",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 2858 "y.tab.c" /* yacc.c:1646  */
    break;

  case 90:
#line 401 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamType((yyvsp[0].type)); }
#line 2864 "y.tab.c" /* yacc.c:1646  */
    break;

  case 91:
#line 403 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
#line 2870 "y.tab.c" /* yacc.c:1646  */
    break;

  case 92:
#line 405 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
#line 2876 "y.tab.c" /* yacc.c:1646  */
    break;

  case 93:
#line 409 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = new TParamList((yyvsp[0].tparam)); }
#line 2882 "y.tab.c" /* yacc.c:1646  */
    break;

  case 94:
#line 411 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = new TParamList((yyvsp[-2].tparam), (yyvsp[0].tparlist)); }
#line 2888 "y.tab.c" /* yacc.c:1646  */
    break;

  case 95:
#line 415 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = new TParamList(0); }
#line 2894 "y.tab.c" /* yacc.c:1646  */
    break;

  case 96:
#line 417 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = (yyvsp[0].tparlist); }
#line 2900 "y.tab.c" /* yacc.c:1646  */
    break;

  case 97:
#line 421 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = 0; }
#line 2906 "y.tab.c" /* yacc.c:1646  */
    break;

  case 98:
#line 423 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = (yyvsp[-1].tparlist); }
#line 2912 "y.tab.c" /* yacc.c:1646  */
    break;

  case 99:
#line 427 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("int"); }
#line 2918 "y.tab.c" /* yacc.c:1646  */
    break;

  case 100:
#line 429 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long"); }
#line 2924 "y.tab.c" /* yacc.c:1646  */
    break;

  case 101:
#line 431 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long int"); }
#line 2930 "y.tab.c" /* yacc.c:1646  */
    break;

  case 102:
#line 433 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("short"); }
#line 2936 "y.tab.c" /* yacc.c:1646  */
    break;

  case 103:
#line 435 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("short int"); }
#line 2942 "y.tab.c" /* yacc.c:1646  */
    break;

  case 104:
#line 437 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("char"); }
#line 2948 "y.tab.c" /* yacc.c:1646  */
    break;

  case 105:
#line 439 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned int"); }
#line 2954 "y.tab.c" /* yacc.c:1646  */
    break;

  case 106:
#line 441 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long"); }
#line 2960 "y.tab.c" /* yacc.c:1646  */
    break;

  case 107:
#line 443 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long int"); }
#line 2966 "y.tab.c" /* yacc.c:1646  */
    break;

  case 108:
#line 445 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long long"); }
#line 2972 "y.tab.c" /* yacc.c:1646  */
    break;

  case 109:
#line 447 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long long int"); }
#line 2978 "y.tab.c" /* yacc.c:1646  */
    break;

  case 110:
#line 449 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned short"); }
#line 2984 "y.tab.c" /* yacc.c:1646  */
    break;

  case 111:
#line 451 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned short int"); }
#line 2990 "y.tab.c" /* yacc.c:1646  */
    break;

  case 112:
#line 453 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned char"); }
#line 2996 "y.tab.c" /* yacc.c:1646  */
    break;

  case 113:
#line 455 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long long"); }
#line 3002 "y.tab.c" /* yacc.c:1646  */
    break;

  case 114:
#line 457 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long long int"); }
#line 3008 "y.tab.c" /* yacc.c:1646  */
    break;

  case 115:
#line 459 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("float"); }
#line 3014 "y.tab.c" /* yacc.c:1646  */
    break;

  case 116:
#line 461 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("double"); }
#line 3020 "y.tab.c" /* yacc.c:1646  */
    break;

  case 117:
#line 463 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long double"); }
#line 3026 "y.tab.c" /* yacc.c:1646  */
    break;

  case 118:
#line 465 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("void"); }
#line 3032 "y.tab.c" /* yacc.c:1646  */
    break;

  case 119:
#line 468 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ntype) = new NamedType((yyvsp[-1].strval),(yyvsp[0].tparlist)); }
#line 3038 "y.tab.c" /* yacc.c:1646  */
    break;

  case 120:
#line 469 "xi-grammar.y" /* yacc.c:1646  */
    { 
                    const char* basename, *scope;
                    splitScopedName((yyvsp[-1].strval), &scope, &basename);
                    (yyval.ntype) = new NamedType(basename, (yyvsp[0].tparlist), scope);
                }
#line 3048 "y.tab.c" /* yacc.c:1646  */
    break;

  case 121:
#line 475 "xi-grammar.y" /* yacc.c:1646  */
    {
			const char* basename, *scope;
			splitScopedName((yyvsp[-1].strval), &scope, &basename);
			(yyval.ntype) = new NamedType(basename, (yyvsp[0].tparlist), scope, true);
		}
#line 3058 "y.tab.c" /* yacc.c:1646  */
    break;

  case 122:
#line 483 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3064 "y.tab.c" /* yacc.c:1646  */
    break;

  case 123:
#line 485 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ntype); }
#line 3070 "y.tab.c" /* yacc.c:1646  */
    break;

  case 124:
#line 489 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ptype) = new PtrType((yyvsp[-1].type)); }
#line 3076 "y.tab.c" /* yacc.c:1646  */
    break;

  case 125:
#line 493 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
#line 3082 "y.tab.c" /* yacc.c:1646  */
    break;

  case 126:
#line 495 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
#line 3088 "y.tab.c" /* yacc.c:1646  */
    break;

  case 127:
#line 499 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ftype) = new FuncType((yyvsp[-7].type), (yyvsp[-4].strval), (yyvsp[-1].plist)); }
#line 3094 "y.tab.c" /* yacc.c:1646  */
    break;

  case 128:
#line 503 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3100 "y.tab.c" /* yacc.c:1646  */
    break;

  case 129:
#line 505 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3106 "y.tab.c" /* yacc.c:1646  */
    break;

  case 130:
#line 507 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3112 "y.tab.c" /* yacc.c:1646  */
    break;

  case 131:
#line 509 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ftype); }
#line 3118 "y.tab.c" /* yacc.c:1646  */
    break;

  case 132:
#line 511 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[0].type)); }
#line 3124 "y.tab.c" /* yacc.c:1646  */
    break;

  case 133:
#line 513 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[-1].type)); }
#line 3130 "y.tab.c" /* yacc.c:1646  */
    break;

  case 134:
#line 517 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3136 "y.tab.c" /* yacc.c:1646  */
    break;

  case 135:
#line 519 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3142 "y.tab.c" /* yacc.c:1646  */
    break;

  case 136:
#line 521 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3148 "y.tab.c" /* yacc.c:1646  */
    break;

  case 137:
#line 523 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[0].type)); }
#line 3154 "y.tab.c" /* yacc.c:1646  */
    break;

  case 138:
#line 525 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[-1].type)); }
#line 3160 "y.tab.c" /* yacc.c:1646  */
    break;

  case 139:
#line 529 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType(new RValueReferenceType((yyvsp[-5].type))); }
#line 3166 "y.tab.c" /* yacc.c:1646  */
    break;

  case 140:
#line 531 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType(new ReferenceType((yyvsp[-4].type))); }
#line 3172 "y.tab.c" /* yacc.c:1646  */
    break;

  case 141:
#line 533 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType((yyvsp[-3].type)); }
#line 3178 "y.tab.c" /* yacc.c:1646  */
    break;

  case 142:
#line 535 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new RValueReferenceType((yyvsp[-2].type)); }
#line 3184 "y.tab.c" /* yacc.c:1646  */
    break;

  case 143:
#line 537 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ReferenceType((yyvsp[-1].type)); }
#line 3190 "y.tab.c" /* yacc.c:1646  */
    break;

  case 144:
#line 539 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3196 "y.tab.c" /* yacc.c:1646  */
    break;

  case 145:
#line 543 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType(new RValueReferenceType((yyvsp[-5].type))); }
#line 3202 "y.tab.c" /* yacc.c:1646  */
    break;

  case 146:
#line 545 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType(new ReferenceType((yyvsp[-4].type))); }
#line 3208 "y.tab.c" /* yacc.c:1646  */
    break;

  case 147:
#line 547 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType((yyvsp[-3].type)); }
#line 3214 "y.tab.c" /* yacc.c:1646  */
    break;

  case 148:
#line 549 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new RValueReferenceType((yyvsp[-2].type)); }
#line 3220 "y.tab.c" /* yacc.c:1646  */
    break;

  case 149:
#line 551 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ReferenceType((yyvsp[-1].type)); }
#line 3226 "y.tab.c" /* yacc.c:1646  */
    break;

  case 150:
#line 553 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3232 "y.tab.c" /* yacc.c:1646  */
    break;

  case 151:
#line 557 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 3238 "y.tab.c" /* yacc.c:1646  */
    break;

  case 152:
#line 561 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = (yyvsp[-1].val); }
#line 3244 "y.tab.c" /* yacc.c:1646  */
    break;

  case 153:
#line 565 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.vallist) = 0; }
#line 3250 "y.tab.c" /* yacc.c:1646  */
    break;

  case 154:
#line 567 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.vallist) = new ValueList((yyvsp[-1].val), (yyvsp[0].vallist)); }
#line 3256 "y.tab.c" /* yacc.c:1646  */
    break;

  case 155:
#line 571 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].vallist)); }
#line 3262 "y.tab.c" /* yacc.c:1646  */
    break;

  case 156:
#line 575 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[-3].type), (yyvsp[-1].strval), (yyvsp[0].vallist), 1); }
#line 3268 "y.tab.c" /* yacc.c:1646  */
    break;

  case 157:
#line 579 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0;}
#line 3274 "y.tab.c" /* yacc.c:1646  */
    break;

  case 158:
#line 581 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0;}
#line 3280 "y.tab.c" /* yacc.c:1646  */
    break;

  case 159:
#line 585 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3286 "y.tab.c" /* yacc.c:1646  */
    break;

  case 160:
#line 587 "xi-grammar.y" /* yacc.c:1646  */
    { 
		  /*
		  printf("Warning: Message attributes are being phased out.\n");
		  printf("Warning: Please remove them from interface files.\n");
		  */
		  (yyval.intval) = (yyvsp[-1].intval); 
		}
#line 3298 "y.tab.c" /* yacc.c:1646  */
    break;

  case 161:
#line 597 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[0].intval); }
#line 3304 "y.tab.c" /* yacc.c:1646  */
    break;

  case 162:
#line 599 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[-2].intval) | (yyvsp[0].intval); }
#line 3310 "y.tab.c" /* yacc.c:1646  */
    break;

  case 163:
#line 603 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3316 "y.tab.c" /* yacc.c:1646  */
    break;

  case 164:
#line 605 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3322 "y.tab.c" /* yacc.c:1646  */
    break;

  case 165:
#line 609 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = 0; }
#line 3328 "y.tab.c" /* yacc.c:1646  */
    break;

  case 166:
#line 611 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-1].cattr); }
#line 3334 "y.tab.c" /* yacc.c:1646  */
    break;

  case 167:
#line 615 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[0].cattr); }
#line 3340 "y.tab.c" /* yacc.c:1646  */
    break;

  case 168:
#line 617 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
#line 3346 "y.tab.c" /* yacc.c:1646  */
    break;

  case 169:
#line 621 "xi-grammar.y" /* yacc.c:1646  */
    { python_doc = NULL; (yyval.intval) = 0; }
#line 3352 "y.tab.c" /* yacc.c:1646  */
    break;

  case 170:
#line 623 "xi-grammar.y" /* yacc.c:1646  */
    { python_doc = (yyvsp[0].strval); (yyval.intval) = 0; }
#line 3358 "y.tab.c" /* yacc.c:1646  */
    break;

  case 171:
#line 627 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CPYTHON; }
#line 3364 "y.tab.c" /* yacc.c:1646  */
    break;

  case 172:
#line 631 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = 0; }
#line 3370 "y.tab.c" /* yacc.c:1646  */
    break;

  case 173:
#line 633 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-1].cattr); }
#line 3376 "y.tab.c" /* yacc.c:1646  */
    break;

  case 174:
#line 637 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[0].cattr); }
#line 3382 "y.tab.c" /* yacc.c:1646  */
    break;

  case 175:
#line 639 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
#line 3388 "y.tab.c" /* yacc.c:1646  */
    break;

  case 176:
#line 643 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CMIGRATABLE; }
#line 3394 "y.tab.c" /* yacc.c:1646  */
    break;

  case 177:
#line 645 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CPYTHON; }
#line 3400 "y.tab.c" /* yacc.c:1646  */
    break;

  case 178:
#line 649 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3406 "y.tab.c" /* yacc.c:1646  */
    break;

  case 179:
#line 651 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 3412 "y.tab.c" /* yacc.c:1646  */
    break;

  case 180:
#line 654 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3418 "y.tab.c" /* yacc.c:1646  */
    break;

  case 181:
#line 656 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 3424 "y.tab.c" /* yacc.c:1646  */
    break;

  case 182:
#line 659 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mv) = new MsgVar((yyvsp[-3].type), (yyvsp[-2].strval), (yyvsp[-4].intval), (yyvsp[-1].intval)); }
#line 3430 "y.tab.c" /* yacc.c:1646  */
    break;

  case 183:
#line 663 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mvlist) = new MsgVarList((yyvsp[0].mv)); }
#line 3436 "y.tab.c" /* yacc.c:1646  */
    break;

  case 184:
#line 665 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mvlist) = new MsgVarList((yyvsp[-1].mv), (yyvsp[0].mvlist)); }
#line 3442 "y.tab.c" /* yacc.c:1646  */
    break;

  case 185:
#line 669 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, (yyvsp[0].ntype)); }
#line 3448 "y.tab.c" /* yacc.c:1646  */
    break;

  case 186:
#line 671 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, (yyvsp[-2].ntype)); }
#line 3454 "y.tab.c" /* yacc.c:1646  */
    break;

  case 187:
#line 673 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, (yyvsp[-3].ntype), (yyvsp[-1].mvlist)); }
#line 3460 "y.tab.c" /* yacc.c:1646  */
    break;

  case 188:
#line 677 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = 0; }
#line 3466 "y.tab.c" /* yacc.c:1646  */
    break;

  case 189:
#line 679 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = (yyvsp[0].typelist); }
#line 3472 "y.tab.c" /* yacc.c:1646  */
    break;

  case 190:
#line 683 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = new TypeList((yyvsp[0].ntype)); }
#line 3478 "y.tab.c" /* yacc.c:1646  */
    break;

  case 191:
#line 685 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = new TypeList((yyvsp[-2].ntype), (yyvsp[0].typelist)); }
#line 3484 "y.tab.c" /* yacc.c:1646  */
    break;

  case 192:
#line 689 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr)|Chare::CCHARE, (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3490 "y.tab.c" /* yacc.c:1646  */
    break;

  case 193:
#line 691 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3496 "y.tab.c" /* yacc.c:1646  */
    break;

  case 194:
#line 695 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3502 "y.tab.c" /* yacc.c:1646  */
    break;

  case 195:
#line 699 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new NodeGroup(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3508 "y.tab.c" /* yacc.c:1646  */
    break;

  case 196:
#line 703 "xi-grammar.y" /* yacc.c:1646  */
    {/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			snprintf(buf,40,"%sD",(yyvsp[-2].strval));
			(yyval.ntype) = new NamedType(buf); 
		}
#line 3518 "y.tab.c" /* yacc.c:1646  */
    break;

  case 197:
#line 709 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ntype) = (yyvsp[-1].ntype); }
#line 3524 "y.tab.c" /* yacc.c:1646  */
    break;

  case 198:
#line 713 "xi-grammar.y" /* yacc.c:1646  */
    {  (yyval.chare) = new Array(lineno, (yyvsp[-4].cattr), (yyvsp[-3].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3530 "y.tab.c" /* yacc.c:1646  */
    break;

  case 199:
#line 715 "xi-grammar.y" /* yacc.c:1646  */
    {  (yyval.chare) = new Array(lineno, (yyvsp[-3].cattr), (yyvsp[-4].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3536 "y.tab.c" /* yacc.c:1646  */
    break;

  case 200:
#line 719 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr)|Chare::CCHARE, new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist));}
#line 3542 "y.tab.c" /* yacc.c:1646  */
    break;

  case 201:
#line 721 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3548 "y.tab.c" /* yacc.c:1646  */
    break;

  case 202:
#line 725 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3554 "y.tab.c" /* yacc.c:1646  */
    break;

  case 203:
#line 729 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new NodeGroup( lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3560 "y.tab.c" /* yacc.c:1646  */
    break;

  case 204:
#line 733 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Array( lineno, 0, (yyvsp[-3].ntype), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3566 "y.tab.c" /* yacc.c:1646  */
    break;

  case 205:
#line 737 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-1].strval))); }
#line 3572 "y.tab.c" /* yacc.c:1646  */
    break;

  case 206:
#line 739 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-4].strval)), (yyvsp[-2].mvlist)); }
#line 3578 "y.tab.c" /* yacc.c:1646  */
    break;

  case 207:
#line 743 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = 0; }
#line 3584 "y.tab.c" /* yacc.c:1646  */
    break;

  case 208:
#line 745 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3590 "y.tab.c" /* yacc.c:1646  */
    break;

  case 209:
#line 749 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = 0; }
#line 3596 "y.tab.c" /* yacc.c:1646  */
    break;

  case 210:
#line 751 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3602 "y.tab.c" /* yacc.c:1646  */
    break;

  case 211:
#line 753 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3608 "y.tab.c" /* yacc.c:1646  */
    break;

  case 212:
#line 755 "xi-grammar.y" /* yacc.c:1646  */
    {
		  XStr typeStr;
		  (yyvsp[0].ntype)->print(typeStr);
		  char *tmp = strdup(typeStr.get_string());
		  (yyval.strval) = tmp;
		}
#line 3619 "y.tab.c" /* yacc.c:1646  */
    break;

  case 213:
#line 764 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TTypeEllipsis(new NamedEllipsisType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3625 "y.tab.c" /* yacc.c:1646  */
    break;

  case 214:
#line 766 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TTypeEllipsis(new NamedEllipsisType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3631 "y.tab.c" /* yacc.c:1646  */
    break;

  case 215:
#line 768 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TType(new NamedType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3637 "y.tab.c" /* yacc.c:1646  */
    break;

  case 216:
#line 770 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TType(new NamedType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3643 "y.tab.c" /* yacc.c:1646  */
    break;

  case 217:
#line 772 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TFunc((yyvsp[-1].ftype), (yyvsp[0].strval)); }
#line 3649 "y.tab.c" /* yacc.c:1646  */
    break;

  case 218:
#line 774 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TName((yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].strval)); }
#line 3655 "y.tab.c" /* yacc.c:1646  */
    break;

  case 219:
#line 778 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = new TVarList((yyvsp[0].tvar)); }
#line 3661 "y.tab.c" /* yacc.c:1646  */
    break;

  case 220:
#line 780 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = new TVarList((yyvsp[-2].tvar), (yyvsp[0].tvarlist)); }
#line 3667 "y.tab.c" /* yacc.c:1646  */
    break;

  case 221:
#line 784 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = (yyvsp[-1].tvarlist); }
#line 3673 "y.tab.c" /* yacc.c:1646  */
    break;

  case 222:
#line 788 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3679 "y.tab.c" /* yacc.c:1646  */
    break;

  case 223:
#line 790 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3685 "y.tab.c" /* yacc.c:1646  */
    break;

  case 224:
#line 792 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3691 "y.tab.c" /* yacc.c:1646  */
    break;

  case 225:
#line 794 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3697 "y.tab.c" /* yacc.c:1646  */
    break;

  case 226:
#line 796 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].message)); (yyvsp[0].message)->setTemplate((yyval.templat)); }
#line 3703 "y.tab.c" /* yacc.c:1646  */
    break;

  case 227:
#line 800 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = 0; }
#line 3709 "y.tab.c" /* yacc.c:1646  */
    break;

  case 228:
#line 802 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = (yyvsp[-2].mbrlist); }
#line 3715 "y.tab.c" /* yacc.c:1646  */
    break;

  case 229:
#line 806 "xi-grammar.y" /* yacc.c:1646  */
    { 
                  if (!connectEntries.empty()) {
                    (yyval.mbrlist) = new AstChildren<Member>(connectEntries);
		  } else {
		    (yyval.mbrlist) = 0; 
                  }
		}
#line 3727 "y.tab.c" /* yacc.c:1646  */
    break;

  case 230:
#line 814 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = new AstChildren<Member>(-1, (yyvsp[-1].member), (yyvsp[0].mbrlist)); }
#line 3733 "y.tab.c" /* yacc.c:1646  */
    break;

  case 231:
#line 818 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].readonly); }
#line 3739 "y.tab.c" /* yacc.c:1646  */
    break;

  case 232:
#line 820 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].readonly); }
#line 3745 "y.tab.c" /* yacc.c:1646  */
    break;

  case 234:
#line 823 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].member); }
#line 3751 "y.tab.c" /* yacc.c:1646  */
    break;

  case 235:
#line 825 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].pupable); }
#line 3757 "y.tab.c" /* yacc.c:1646  */
    break;

  case 236:
#line 827 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].includeFile); }
#line 3763 "y.tab.c" /* yacc.c:1646  */
    break;

  case 237:
#line 829 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new ClassDeclaration(lineno,(yyvsp[0].strval)); }
#line 3769 "y.tab.c" /* yacc.c:1646  */
    break;

  case 238:
#line 833 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1); }
#line 3775 "y.tab.c" /* yacc.c:1646  */
    break;

  case 239:
#line 835 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1); }
#line 3781 "y.tab.c" /* yacc.c:1646  */
    break;

  case 240:
#line 837 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[-6].strval)) + '<' +
					    ((yyvsp[-4].tparlist))->to_string() + '>').c_str()),
				    1);
		}
#line 3791 "y.tab.c" /* yacc.c:1646  */
    break;

  case 241:
#line 843 "xi-grammar.y" /* yacc.c:1646  */
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[-2]).first_column, (yylsp[-2]).last_column, (yylsp[-2]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1);
		}
#line 3801 "y.tab.c" /* yacc.c:1646  */
    break;

  case 242:
#line 849 "xi-grammar.y" /* yacc.c:1646  */
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[-5]).first_column, (yylsp[-5]).last_column, (yylsp[-5]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1);
		}
#line 3811 "y.tab.c" /* yacc.c:1646  */
    break;

  case 243:
#line 858 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 0); }
#line 3817 "y.tab.c" /* yacc.c:1646  */
    break;

  case 244:
#line 860 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 0); }
#line 3823 "y.tab.c" /* yacc.c:1646  */
    break;

  case 245:
#line 862 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[-6].strval)) + '<' +
					    ((yyvsp[-4].tparlist))->to_string() + '>').c_str()),
				    0);
		}
#line 3833 "y.tab.c" /* yacc.c:1646  */
    break;

  case 246:
#line 868 "xi-grammar.y" /* yacc.c:1646  */
    {
                  InitCall* rtn = new InitCall(lineno, (yyvsp[-3].strval), 0);
                  rtn->setAccel();
                  (yyval.member) = rtn;
		}
#line 3843 "y.tab.c" /* yacc.c:1646  */
    break;

  case 247:
#line 876 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[0].ntype),0); }
#line 3849 "y.tab.c" /* yacc.c:1646  */
    break;

  case 248:
#line 878 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[-2].ntype),(yyvsp[0].pupable)); }
#line 3855 "y.tab.c" /* yacc.c:1646  */
    break;

  case 249:
#line 881 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.includeFile) = new IncludeFile(lineno,(yyvsp[0].strval)); }
#line 3861 "y.tab.c" /* yacc.c:1646  */
    break;

  case 250:
#line 885 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].member); }
#line 3867 "y.tab.c" /* yacc.c:1646  */
    break;

  case 251:
#line 889 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].entry); }
#line 3873 "y.tab.c" /* yacc.c:1646  */
    break;

  case 252:
#line 891 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyvsp[0].entry)->tspec = (yyvsp[-1].tvarlist);
                  (yyval.member) = (yyvsp[0].entry);
                }
#line 3882 "y.tab.c" /* yacc.c:1646  */
    break;

  case 253:
#line 896 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[-1].member); }
#line 3888 "y.tab.c" /* yacc.c:1646  */
    break;

  case 254:
#line 898 "xi-grammar.y" /* yacc.c:1646  */
    {
          ERROR("invalid SDAG member",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 3898 "y.tab.c" /* yacc.c:1646  */
    break;

  case 255:
#line 906 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3904 "y.tab.c" /* yacc.c:1646  */
    break;

  case 256:
#line 908 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3910 "y.tab.c" /* yacc.c:1646  */
    break;

  case 257:
#line 910 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3916 "y.tab.c" /* yacc.c:1646  */
    break;

  case 258:
#line 912 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3922 "y.tab.c" /* yacc.c:1646  */
    break;

  case 259:
#line 914 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3928 "y.tab.c" /* yacc.c:1646  */
    break;

  case 260:
#line 916 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3934 "y.tab.c" /* yacc.c:1646  */
    break;

  case 261:
#line 918 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3940 "y.tab.c" /* yacc.c:1646  */
    break;

  case 262:
#line 920 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3946 "y.tab.c" /* yacc.c:1646  */
    break;

  case 263:
#line 922 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3952 "y.tab.c" /* yacc.c:1646  */
    break;

  case 264:
#line 924 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3958 "y.tab.c" /* yacc.c:1646  */
    break;

  case 265:
#line 926 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3964 "y.tab.c" /* yacc.c:1646  */
    break;

  case 266:
#line 929 "xi-grammar.y" /* yacc.c:1646  */
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
#line 3979 "y.tab.c" /* yacc.c:1646  */
    break;

  case 267:
#line 940 "xi-grammar.y" /* yacc.c:1646  */
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
#line 4001 "y.tab.c" /* yacc.c:1646  */
    break;

  case 268:
#line 958 "xi-grammar.y" /* yacc.c:1646  */
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
#line 4021 "y.tab.c" /* yacc.c:1646  */
    break;

  case 269:
#line 976 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.accelBlock) = new AccelBlock(lineno, new XStr((yyvsp[-2].strval))); }
#line 4027 "y.tab.c" /* yacc.c:1646  */
    break;

  case 270:
#line 978 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.accelBlock) = new AccelBlock(lineno, NULL); }
#line 4033 "y.tab.c" /* yacc.c:1646  */
    break;

  case 271:
#line 982 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 4039 "y.tab.c" /* yacc.c:1646  */
    break;

  case 272:
#line 986 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attr) = 0; }
#line 4045 "y.tab.c" /* yacc.c:1646  */
    break;

  case 273:
#line 988 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attr) = (yyvsp[-1].attr); }
#line 4051 "y.tab.c" /* yacc.c:1646  */
    break;

  case 274:
#line 990 "xi-grammar.y" /* yacc.c:1646  */
    { ERROR("invalid entry method attribute list",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
#line 4060 "y.tab.c" /* yacc.c:1646  */
    break;

  case 275:
#line 996 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attrarg) = new Attribute::Argument((yyvsp[-2].strval), atoi((yyvsp[0].strval))); }
#line 4066 "y.tab.c" /* yacc.c:1646  */
    break;

  case 276:
#line 1000 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attrarg) = (yyvsp[0].attrarg); }
#line 4072 "y.tab.c" /* yacc.c:1646  */
    break;

  case 277:
#line 1001 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attrarg) = (yyvsp[-2].attrarg); (yyvsp[-2].attrarg)->next = (yyvsp[0].attrarg); }
#line 4078 "y.tab.c" /* yacc.c:1646  */
    break;

  case 278:
#line 1005 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attr) = new Attribute((yyvsp[0].intval));           }
#line 4084 "y.tab.c" /* yacc.c:1646  */
    break;

  case 279:
#line 1006 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attr) = new Attribute((yyvsp[-3].intval), (yyvsp[-1].attrarg));       }
#line 4090 "y.tab.c" /* yacc.c:1646  */
    break;

  case 280:
#line 1007 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attr) = new Attribute((yyvsp[-2].intval), NULL, (yyvsp[0].attr)); }
#line 4096 "y.tab.c" /* yacc.c:1646  */
    break;

  case 281:
#line 1008 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attr) = new Attribute((yyvsp[-5].intval), (yyvsp[-3].attrarg), (yyvsp[0].attr));   }
#line 4102 "y.tab.c" /* yacc.c:1646  */
    break;

  case 282:
#line 1012 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = STHREADED; }
#line 4108 "y.tab.c" /* yacc.c:1646  */
    break;

  case 283:
#line 1014 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SWHENIDLE; }
#line 4114 "y.tab.c" /* yacc.c:1646  */
    break;

  case 284:
#line 1016 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SSYNC; }
#line 4120 "y.tab.c" /* yacc.c:1646  */
    break;

  case 285:
#line 1018 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SIGET; }
#line 4126 "y.tab.c" /* yacc.c:1646  */
    break;

  case 286:
#line 1020 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SLOCKED; }
#line 4132 "y.tab.c" /* yacc.c:1646  */
    break;

  case 287:
#line 1022 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SCREATEHERE; }
#line 4138 "y.tab.c" /* yacc.c:1646  */
    break;

  case 288:
#line 1024 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SCREATEHOME; }
#line 4144 "y.tab.c" /* yacc.c:1646  */
    break;

  case 289:
#line 1026 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SNOKEEP; }
#line 4150 "y.tab.c" /* yacc.c:1646  */
    break;

  case 290:
#line 1028 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SINBOX; }
#line 4156 "y.tab.c" /* yacc.c:1646  */
    break;

  case 291:
#line 1030 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SCOROUTINE; }
#line 4162 "y.tab.c" /* yacc.c:1646  */
    break;

  case 292:
#line 1032 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SNOTRACE; }
#line 4168 "y.tab.c" /* yacc.c:1646  */
    break;

  case 293:
#line 1034 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SAPPWORK; }
#line 4174 "y.tab.c" /* yacc.c:1646  */
    break;

  case 294:
#line 1036 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SIMMEDIATE; }
#line 4180 "y.tab.c" /* yacc.c:1646  */
    break;

  case 295:
#line 1038 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SSKIPSCHED; }
#line 4186 "y.tab.c" /* yacc.c:1646  */
    break;

  case 296:
#line 1040 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SINLINE; }
#line 4192 "y.tab.c" /* yacc.c:1646  */
    break;

  case 297:
#line 1042 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SLOCAL; }
#line 4198 "y.tab.c" /* yacc.c:1646  */
    break;

  case 298:
#line 1044 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SPYTHON; }
#line 4204 "y.tab.c" /* yacc.c:1646  */
    break;

  case 299:
#line 1046 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SMEM; }
#line 4210 "y.tab.c" /* yacc.c:1646  */
    break;

  case 300:
#line 1048 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SREDUCE; }
#line 4216 "y.tab.c" /* yacc.c:1646  */
    break;

  case 301:
#line 1050 "xi-grammar.y" /* yacc.c:1646  */
    {
        (yyval.intval) = SAGGREGATE;
    }
#line 4224 "y.tab.c" /* yacc.c:1646  */
    break;

  case 302:
#line 1054 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("invalid entry method attribute",
		        (yylsp[0]).first_column, (yylsp[0]).last_column);
		  yyclearin;
		  yyerrok;
		}
#line 4235 "y.tab.c" /* yacc.c:1646  */
    break;

  case 303:
#line 1063 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4241 "y.tab.c" /* yacc.c:1646  */
    break;

  case 304:
#line 1065 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4247 "y.tab.c" /* yacc.c:1646  */
    break;

  case 305:
#line 1067 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4253 "y.tab.c" /* yacc.c:1646  */
    break;

  case 306:
#line 1071 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = ""; }
#line 4259 "y.tab.c" /* yacc.c:1646  */
    break;

  case 307:
#line 1073 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4265 "y.tab.c" /* yacc.c:1646  */
    break;

  case 308:
#line 1075 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Returned only when in_bracket*/
			int len = strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3;
			char *tmp = new char[len];
			snprintf(tmp,len,"%s, %s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4276 "y.tab.c" /* yacc.c:1646  */
    break;

  case 309:
#line 1084 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = ""; }
#line 4282 "y.tab.c" /* yacc.c:1646  */
    break;

  case 310:
#line 1086 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4288 "y.tab.c" /* yacc.c:1646  */
    break;

  case 311:
#line 1088 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Returned only when in_bracket*/
			int len = strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3;
			char *tmp = new char[len];
			snprintf(tmp, len, "%s[%s]%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4299 "y.tab.c" /* yacc.c:1646  */
    break;

  case 312:
#line 1095 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			int len = strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3;
			char *tmp = new char[len];
			snprintf(tmp, len, "%s{%s}%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4310 "y.tab.c" /* yacc.c:1646  */
    break;

  case 313:
#line 1102 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			int len = strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3;
			char *tmp = new char[len];
			snprintf(tmp, len, "%s(%s)%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4321 "y.tab.c" /* yacc.c:1646  */
    break;

  case 314:
#line 1109 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			int len = strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3;
			char *tmp = new char[len];
			snprintf(tmp, len, "(%s)%s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4332 "y.tab.c" /* yacc.c:1646  */
    break;

  case 315:
#line 1118 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval));
		}
#line 4341 "y.tab.c" /* yacc.c:1646  */
    break;

  case 316:
#line 1125 "xi-grammar.y" /* yacc.c:1646  */
    { 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			(yyval.intval) = 0;
		}
#line 4351 "y.tab.c" /* yacc.c:1646  */
    break;

  case 317:
#line 1133 "xi-grammar.y" /* yacc.c:1646  */
    { 
			in_braces=0;
			(yyval.intval) = 0;
		}
#line 4360 "y.tab.c" /* yacc.c:1646  */
    break;

  case 318:
#line 1140 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[0].type));}
#line 4366 "y.tab.c" /* yacc.c:1646  */
    break;

  case 319:
#line 1142 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval)); (yyval.pname)->setConditional((yyvsp[0].intval)); }
#line 4372 "y.tab.c" /* yacc.c:1646  */
    break;

  case 320:
#line 1144 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[-3].type),(yyvsp[-2].strval),0,(yyvsp[0].val));}
#line 4378 "y.tab.c" /* yacc.c:1646  */
    break;

  case 321:
#line 1146 "xi-grammar.y" /* yacc.c:1646  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
		}
#line 4387 "y.tab.c" /* yacc.c:1646  */
    break;

  case 322:
#line 1151 "xi-grammar.y" /* yacc.c:1646  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
			(yyval.pname)->setRdma(CMK_ZC_P2P_SEND_MSG);
			if(firstRdma) {
				(yyval.pname)->setFirstRdma(true);
				firstRdma = false;
			}
		}
#line 4401 "y.tab.c" /* yacc.c:1646  */
    break;

  case 323:
#line 1161 "xi-grammar.y" /* yacc.c:1646  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
			(yyval.pname)->setRdma(CMK_ZC_P2P_RECV_MSG);
			if(firstRdma) {
				(yyval.pname)->setFirstRdma(true);
				firstRdma = false;
			}
		}
#line 4415 "y.tab.c" /* yacc.c:1646  */
    break;

  case 324:
#line 1171 "xi-grammar.y" /* yacc.c:1646  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
			(yyval.pname)->setRdma(CMK_ZC_DEVICE_MSG);
			if (firstDeviceRdma) {
				(yyval.pname)->setFirstDeviceRdma(true);
				firstDeviceRdma = false;
			}
		}
#line 4429 "y.tab.c" /* yacc.c:1646  */
    break;

  case 325:
#line 1182 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READONLY; }
#line 4435 "y.tab.c" /* yacc.c:1646  */
    break;

  case 326:
#line 1183 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READWRITE; }
#line 4441 "y.tab.c" /* yacc.c:1646  */
    break;

  case 327:
#line 1184 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_WRITEONLY; }
#line 4447 "y.tab.c" /* yacc.c:1646  */
    break;

  case 328:
#line 1187 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr((yyvsp[0].strval)); }
#line 4453 "y.tab.c" /* yacc.c:1646  */
    break;

  case 329:
#line 1188 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "->" << (yyvsp[0].strval); }
#line 4459 "y.tab.c" /* yacc.c:1646  */
    break;

  case 330:
#line 1189 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[-2].xstrptr)) << "." << (yyvsp[0].strval); }
#line 4465 "y.tab.c" /* yacc.c:1646  */
    break;

  case 331:
#line 1191 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << *((yyvsp[-1].xstrptr)) << "]";
                  delete (yyvsp[-3].xstrptr);
                  delete (yyvsp[-1].xstrptr);
                }
#line 4476 "y.tab.c" /* yacc.c:1646  */
    break;

  case 332:
#line 1198 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << (yyvsp[-1].strval) << "]";
                  delete (yyvsp[-3].xstrptr);
                }
#line 4486 "y.tab.c" /* yacc.c:1646  */
    break;

  case 333:
#line 1204 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "(" << *((yyvsp[-1].xstrptr)) << ")";
                  delete (yyvsp[-3].xstrptr);
                  delete (yyvsp[-1].xstrptr);
                }
#line 4497 "y.tab.c" /* yacc.c:1646  */
    break;

  case 334:
#line 1213 "xi-grammar.y" /* yacc.c:1646  */
    {
                  in_bracket = 0;
                  (yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName(), (yyvsp[-1].strval));
                }
#line 4506 "y.tab.c" /* yacc.c:1646  */
    break;

  case 335:
#line 1220 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[-6].intval));
                }
#line 4516 "y.tab.c" /* yacc.c:1646  */
    break;

  case 336:
#line 1226 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType(Parameter::ACCEL_BUFFER_TYPE_READWRITE);
		}
#line 4526 "y.tab.c" /* yacc.c:1646  */
    break;

  case 337:
#line 1232 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.pname) = (yyvsp[-3].pname);
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[-5].intval));
		}
#line 4536 "y.tab.c" /* yacc.c:1646  */
    break;

  case 338:
#line 1240 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 4542 "y.tab.c" /* yacc.c:1646  */
    break;

  case 339:
#line 1242 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 4548 "y.tab.c" /* yacc.c:1646  */
    break;

  case 340:
#line 1246 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 4554 "y.tab.c" /* yacc.c:1646  */
    break;

  case 341:
#line 1248 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 4560 "y.tab.c" /* yacc.c:1646  */
    break;

  case 342:
#line 1252 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = (yyvsp[-1].plist); }
#line 4566 "y.tab.c" /* yacc.c:1646  */
    break;

  case 343:
#line 1254 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList(new Parameter(0, new BuiltinType("void"))); }
#line 4572 "y.tab.c" /* yacc.c:1646  */
    break;

  case 344:
#line 1258 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = (yyvsp[-1].plist); }
#line 4578 "y.tab.c" /* yacc.c:1646  */
    break;

  case 345:
#line 1260 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = 0; }
#line 4584 "y.tab.c" /* yacc.c:1646  */
    break;

  case 346:
#line 1264 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = 0; }
#line 4590 "y.tab.c" /* yacc.c:1646  */
    break;

  case 347:
#line 1266 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4596 "y.tab.c" /* yacc.c:1646  */
    break;

  case 348:
#line 1270 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = 0; }
#line 4602 "y.tab.c" /* yacc.c:1646  */
    break;

  case 349:
#line 1272 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[0].sc)); }
#line 4608 "y.tab.c" /* yacc.c:1646  */
    break;

  case 350:
#line 1274 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[-2].slist)); }
#line 4614 "y.tab.c" /* yacc.c:1646  */
    break;

  case 351:
#line 1278 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.slist) = new SListConstruct((yyvsp[0].sc)); }
#line 4620 "y.tab.c" /* yacc.c:1646  */
    break;

  case 352:
#line 1280 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.slist) = new SListConstruct((yyvsp[-1].sc), (yyvsp[0].slist));  }
#line 4626 "y.tab.c" /* yacc.c:1646  */
    break;

  case 353:
#line 1284 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.olist) = new OListConstruct((yyvsp[0].sc)); }
#line 4632 "y.tab.c" /* yacc.c:1646  */
    break;

  case 354:
#line 1286 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.olist) = new OListConstruct((yyvsp[-1].sc), (yyvsp[0].slist)); }
#line 4638 "y.tab.c" /* yacc.c:1646  */
    break;

  case 355:
#line 1290 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.clist) = new CaseListConstruct((yyvsp[0].when)); }
#line 4644 "y.tab.c" /* yacc.c:1646  */
    break;

  case 356:
#line 1292 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.clist) = new CaseListConstruct((yyvsp[-1].when), (yyvsp[0].clist)); }
#line 4650 "y.tab.c" /* yacc.c:1646  */
    break;

  case 357:
#line 1294 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("case blocks can only contain when clauses",
		        (yylsp[0]).first_column, (yylsp[0]).last_column);
		  (yyval.clist) = 0;
		}
#line 4660 "y.tab.c" /* yacc.c:1646  */
    break;

  case 358:
#line 1302 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4666 "y.tab.c" /* yacc.c:1646  */
    break;

  case 359:
#line 1304 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = 0; }
#line 4672 "y.tab.c" /* yacc.c:1646  */
    break;

  case 360:
#line 1308 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-2].entrylist), 0); }
#line 4678 "y.tab.c" /* yacc.c:1646  */
    break;

  case 361:
#line 1310 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-1].entrylist), (yyvsp[0].sc)); }
#line 4684 "y.tab.c" /* yacc.c:1646  */
    break;

  case 362:
#line 1312 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-3].entrylist), (yyvsp[-1].slist)); }
#line 4690 "y.tab.c" /* yacc.c:1646  */
    break;

  case 363:
#line 1316 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4696 "y.tab.c" /* yacc.c:1646  */
    break;

  case 364:
#line 1318 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4702 "y.tab.c" /* yacc.c:1646  */
    break;

  case 365:
#line 1320 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4708 "y.tab.c" /* yacc.c:1646  */
    break;

  case 366:
#line 1322 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4714 "y.tab.c" /* yacc.c:1646  */
    break;

  case 367:
#line 1324 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4720 "y.tab.c" /* yacc.c:1646  */
    break;

  case 368:
#line 1326 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4726 "y.tab.c" /* yacc.c:1646  */
    break;

  case 369:
#line 1328 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4732 "y.tab.c" /* yacc.c:1646  */
    break;

  case 370:
#line 1330 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4738 "y.tab.c" /* yacc.c:1646  */
    break;

  case 371:
#line 1332 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4744 "y.tab.c" /* yacc.c:1646  */
    break;

  case 372:
#line 1334 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4750 "y.tab.c" /* yacc.c:1646  */
    break;

  case 373:
#line 1336 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4756 "y.tab.c" /* yacc.c:1646  */
    break;

  case 374:
#line 1338 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4762 "y.tab.c" /* yacc.c:1646  */
    break;

  case 375:
#line 1342 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new SerialConstruct((yyvsp[-2].strval), (yyvsp[-4].strval), (yylsp[-3]).first_line); }
#line 4768 "y.tab.c" /* yacc.c:1646  */
    break;

  case 376:
#line 1344 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new OverlapConstruct((yyvsp[-1].olist)); }
#line 4774 "y.tab.c" /* yacc.c:1646  */
    break;

  case 377:
#line 1346 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = (yyvsp[0].when); }
#line 4780 "y.tab.c" /* yacc.c:1646  */
    break;

  case 378:
#line 1348 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new CaseConstruct((yyvsp[-1].clist)); }
#line 4786 "y.tab.c" /* yacc.c:1646  */
    break;

  case 379:
#line 1350 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForConstruct((yyvsp[-8].intexpr), (yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4792 "y.tab.c" /* yacc.c:1646  */
    break;

  case 380:
#line 1352 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForConstruct((yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4798 "y.tab.c" /* yacc.c:1646  */
    break;

  case 381:
#line 1354 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-9].strval)), (yyvsp[-6].intexpr),
		             (yyvsp[-4].intexpr), (yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4805 "y.tab.c" /* yacc.c:1646  */
    break;

  case 382:
#line 1357 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-11].strval)), (yyvsp[-8].intexpr),
		             (yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4812 "y.tab.c" /* yacc.c:1646  */
    break;

  case 383:
#line 1360 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new IfConstruct((yyvsp[-3].intexpr), (yyvsp[-1].sc), (yyvsp[0].sc)); }
#line 4818 "y.tab.c" /* yacc.c:1646  */
    break;

  case 384:
#line 1362 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new IfConstruct((yyvsp[-5].intexpr), (yyvsp[-2].slist), (yyvsp[0].sc)); }
#line 4824 "y.tab.c" /* yacc.c:1646  */
    break;

  case 385:
#line 1364 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new WhileConstruct((yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4830 "y.tab.c" /* yacc.c:1646  */
    break;

  case 386:
#line 1366 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new WhileConstruct((yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4836 "y.tab.c" /* yacc.c:1646  */
    break;

  case 387:
#line 1368 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new SerialConstruct((yyvsp[-2].strval), NULL, (yyloc).first_line); }
#line 4842 "y.tab.c" /* yacc.c:1646  */
    break;

  case 388:
#line 1370 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("unknown SDAG construct or malformed entry method declaration.\n"
		        "You may have forgotten to terminate a previous entry method declaration with a"
		        " semicolon or forgotten to mark a block of sequential SDAG code as 'serial'",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
#line 4854 "y.tab.c" /* yacc.c:1646  */
    break;

  case 389:
#line 1380 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = 0; }
#line 4860 "y.tab.c" /* yacc.c:1646  */
    break;

  case 390:
#line 1382 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ElseConstruct((yyvsp[0].sc)); }
#line 4866 "y.tab.c" /* yacc.c:1646  */
    break;

  case 391:
#line 1384 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ElseConstruct((yyvsp[-1].slist)); }
#line 4872 "y.tab.c" /* yacc.c:1646  */
    break;

  case 392:
#line 1388 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intexpr) = new IntExprConstruct((yyvsp[0].strval)); }
#line 4878 "y.tab.c" /* yacc.c:1646  */
    break;

  case 393:
#line 1392 "xi-grammar.y" /* yacc.c:1646  */
    { in_int_expr = 0; (yyval.intval) = 0; }
#line 4884 "y.tab.c" /* yacc.c:1646  */
    break;

  case 394:
#line 1396 "xi-grammar.y" /* yacc.c:1646  */
    { in_int_expr = 1; (yyval.intval) = 0; }
#line 4890 "y.tab.c" /* yacc.c:1646  */
    break;

  case 395:
#line 1400 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.entry) = new Entry(lineno, NULL, 0, (yyvsp[-1].strval), (yyvsp[0].plist), 0, 0, 0, (yyloc).first_line, (yyloc).last_line);
		  firstRdma = true;
		  firstDeviceRdma = true;
		}
#line 4900 "y.tab.c" /* yacc.c:1646  */
    break;

  case 396:
#line 1406 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.entry) = new Entry(lineno, NULL, 0, (yyvsp[-4].strval), (yyvsp[0].plist), 0, 0, (yyvsp[-2].strval), (yyloc).first_line, (yyloc).last_line);
		  firstRdma = true;
		  firstDeviceRdma = true;
		}
#line 4910 "y.tab.c" /* yacc.c:1646  */
    break;

  case 397:
#line 1414 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.entrylist) = new EntryList((yyvsp[0].entry)); }
#line 4916 "y.tab.c" /* yacc.c:1646  */
    break;

  case 398:
#line 1416 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.entrylist) = new EntryList((yyvsp[-2].entry),(yyvsp[0].entrylist)); }
#line 4922 "y.tab.c" /* yacc.c:1646  */
    break;

  case 399:
#line 1420 "xi-grammar.y" /* yacc.c:1646  */
    { in_bracket=1; }
#line 4928 "y.tab.c" /* yacc.c:1646  */
    break;

  case 400:
#line 1423 "xi-grammar.y" /* yacc.c:1646  */
    { in_bracket=0; }
#line 4934 "y.tab.c" /* yacc.c:1646  */
    break;

  case 401:
#line 1427 "xi-grammar.y" /* yacc.c:1646  */
    { if (!macroDefined((yyvsp[0].strval), 1)) in_comment = 1; }
#line 4940 "y.tab.c" /* yacc.c:1646  */
    break;

  case 402:
#line 1431 "xi-grammar.y" /* yacc.c:1646  */
    { if (!macroDefined((yyvsp[0].strval), 0)) in_comment = 1; }
#line 4946 "y.tab.c" /* yacc.c:1646  */
    break;


#line 4950 "y.tab.c" /* yacc.c:1646  */
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
#line 1434 "xi-grammar.y" /* yacc.c:1906  */


void yyerror(const char *s) 
{
	fprintf(stderr, "[PARSE-ERROR] Unexpected/missing token at line %d. Current token being parsed: '%s'.\n", lineno, yytext);
}
