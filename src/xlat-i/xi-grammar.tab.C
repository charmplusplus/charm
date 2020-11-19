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
    SIZET = 326,
    BOOL = 327,
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
#define SIZET 326
#define BOOL 327
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
#define YYLAST   1653

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  97
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  119
/* YYNRULES -- Number of rules.  */
#define YYNRULES  397
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  790

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
     295,   296,   300,   302,   308,   315,   319,   326,   328,   333,
     334,   338,   340,   342,   344,   346,   360,   362,   364,   366,
     372,   374,   376,   378,   380,   382,   384,   386,   388,   390,
     398,   400,   402,   406,   408,   413,   414,   419,   420,   424,
     426,   428,   430,   432,   434,   436,   438,   440,   442,   444,
     446,   448,   450,   452,   454,   456,   460,   461,   466,   474,
     476,   480,   484,   486,   490,   494,   496,   498,   500,   502,
     504,   508,   510,   512,   514,   516,   520,   522,   524,   526,
     528,   530,   534,   536,   538,   540,   542,   544,   548,   552,
     557,   558,   562,   566,   571,   572,   577,   578,   588,   590,
     594,   596,   601,   602,   606,   608,   613,   614,   618,   623,
     624,   628,   630,   634,   636,   641,   642,   646,   647,   650,
     654,   656,   660,   662,   664,   669,   670,   674,   676,   680,
     682,   686,   690,   694,   700,   704,   706,   710,   712,   716,
     720,   724,   728,   730,   735,   736,   741,   742,   744,   746,
     755,   757,   759,   761,   763,   765,   769,   771,   775,   779,
     781,   783,   785,   787,   791,   793,   798,   805,   809,   811,
     813,   814,   816,   818,   820,   824,   826,   828,   834,   840,
     849,   851,   853,   859,   867,   869,   872,   876,   880,   882,
     887,   889,   897,   899,   901,   903,   905,   907,   909,   911,
     913,   915,   917,   920,   931,   949,   967,   969,   973,   978,
     979,   981,   988,   992,   993,   997,   998,   999,  1000,  1003,
    1005,  1007,  1009,  1011,  1013,  1015,  1017,  1019,  1021,  1023,
    1025,  1027,  1029,  1031,  1033,  1035,  1037,  1041,  1050,  1052,
    1054,  1059,  1060,  1062,  1071,  1072,  1074,  1080,  1086,  1092,
    1100,  1107,  1115,  1122,  1124,  1126,  1128,  1133,  1143,  1153,
    1165,  1166,  1167,  1170,  1171,  1172,  1173,  1180,  1186,  1195,
    1202,  1208,  1214,  1222,  1224,  1228,  1230,  1234,  1236,  1240,
    1242,  1247,  1248,  1252,  1254,  1256,  1260,  1262,  1266,  1268,
    1272,  1274,  1276,  1284,  1287,  1290,  1292,  1294,  1298,  1300,
    1302,  1304,  1306,  1308,  1310,  1312,  1314,  1316,  1318,  1320,
    1324,  1326,  1328,  1330,  1332,  1334,  1336,  1339,  1342,  1344,
    1346,  1348,  1350,  1352,  1363,  1364,  1366,  1370,  1374,  1378,
    1382,  1388,  1396,  1398,  1402,  1405,  1409,  1413
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
  "UNSIGNED", "SIZET", "BOOL", "ACCEL", "READWRITE", "WRITEONLY",
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

#define YYPACT_NINF -596

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-596)))

#define YYTABLE_NINF -349

#define yytable_value_is_error(Yytable_value) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
     147,  1401,  1401,    53,  -596,   147,  -596,  -596,  -596,  -596,
    -596,  -596,  -596,  -596,  -596,  -596,  -596,  -596,  -596,  -596,
    -596,  -596,  -596,  -596,  -596,  -596,  -596,  -596,  -596,  -596,
    -596,  -596,  -596,  -596,  -596,  -596,  -596,  -596,  -596,  -596,
    -596,  -596,  -596,  -596,  -596,  -596,  -596,  -596,  -596,  -596,
    -596,  -596,  -596,  -596,  -596,  -596,  -596,    55,    55,  -596,
    -596,  -596,   914,   -25,  -596,  -596,  -596,    38,  1401,   143,
    1401,  1401,   195,  1081,    36,  1057,   914,  -596,  -596,  -596,
    -596,   242,    24,    98,  -596,   114,  -596,  -596,  -596,   -25,
      66,  1445,   149,   149,     3,     0,   109,   109,   109,   109,
     120,   135,  1401,   180,   165,   914,  -596,  -596,  -596,  -596,
    -596,  -596,  -596,  -596,   503,  -596,  -596,  -596,  -596,   176,
    -596,  -596,  -596,  -596,  -596,  -596,  -596,  -596,  -596,  -596,
    -596,   -25,  -596,  -596,  -596,  1264,  1556,   914,   114,   185,
     105,    66,   191,   630,  -596,  1573,  -596,   142,  -596,  -596,
    -596,  -596,   277,  -596,  -596,    98,   188,  -596,  -596,   200,
     208,   217,  -596,    -5,    98,  -596,    98,    98,   255,    98,
     246,  -596,    17,  1401,  1401,  1401,  1401,    74,   269,   271,
     258,  1401,  -596,  -596,  -596,  1482,   262,   109,   109,   109,
     109,   269,   135,  -596,  -596,  -596,  -596,  -596,   -25,  -596,
    -596,  -596,  -596,  -596,  -596,  -596,  -596,  -596,  -596,  -596,
    -596,  -596,  -596,  -596,  -596,   305,  -596,  -596,  -596,   272,
     130,  1556,   200,   208,   217,    33,  -596,     0,   285,    63,
      66,   310,    66,   292,  -596,   176,   289,    13,  -596,  -596,
    -596,   102,  -596,  -596,   188,   329,  -596,  -596,  -596,  -596,
    -596,   296,   224,   298,   -31,    96,   163,   309,   243,     0,
    -596,  -596,   312,   319,   322,   328,   328,   328,   328,  -596,
    1401,   317,   339,   334,   101,  1401,   359,  1401,  -596,  -596,
     337,   351,   356,   824,   -13,   123,  1401,   357,   354,   176,
    1401,  1401,  1401,  1401,  1401,  1401,  -596,  -596,  -596,  1264,
    1401,   404,  -596,   264,   352,  1401,  -596,  -596,  -596,   361,
     366,   367,   362,    66,   -25,    98,  -596,  -596,  -596,  -596,
    -596,   373,  -596,   375,  -596,  1401,   377,   378,   379,  -596,
     370,  -596,    66,   149,   329,   149,   149,   329,   149,  -596,
    -596,    17,  -596,     0,   203,   203,   203,   203,   380,  -596,
     359,  -596,   328,   328,  -596,   258,    14,   386,   384,   129,
     389,   124,  -596,   387,  1482,  -596,  -596,   328,   328,   328,
     328,   328,   253,  -596,   401,   400,   396,   395,   397,   399,
     322,    66,   310,    66,    66,  -596,   -31,   329,  -596,   398,
     402,   403,  -596,  -596,   405,  -596,   406,   410,   411,    98,
     413,   414,  -596,   407,  -596,   458,   -25,  -596,  -596,  -596,
    -596,  -596,  -596,   203,   203,  -596,  -596,  -596,  1573,    30,
     421,   429,  1573,  -596,  -596,   430,  -596,  -596,  -596,  -596,
    -596,   203,   203,   203,   203,   203,   506,   -25,   469,  1401,
     444,   438,   440,  -596,   445,  -596,  -596,  -596,  -596,  -596,
    -596,   447,   441,  -596,  -596,  -596,  -596,   448,  -596,   155,
     449,  -596,     0,  -596,   741,   495,   462,   176,   458,  -596,
    -596,  -596,  -596,  1401,  -596,  -596,  1401,  -596,   496,  -596,
    -596,  -596,  -596,  -596,   471,  -596,  -596,  1264,   467,  -596,
    1503,  -596,  1538,  -596,   149,   149,   149,  -596,  1206,  1146,
    -596,   176,   -25,  -596,   468,   384,   384,   176,  -596,  -596,
    1573,  1573,  1573,  -596,  1401,    66,   477,   475,   476,   478,
     479,   481,   473,   484,   445,  1401,  -596,   482,   176,  -596,
    -596,   -25,  1401,    66,    66,    66,    23,   483,  1538,  -596,
    -596,  -596,  -596,  -596,   537,   603,   445,  -596,   -25,   485,
     488,   500,   502,  -596,   260,  -596,  -596,  -596,  1401,  -596,
     508,   505,   508,   543,   512,   542,   508,   520,   302,   -25,
      66,  -596,  -596,  -596,   585,  -596,  -596,  -596,  -596,  -596,
     114,  -596,   445,  -596,    66,   550,    66,   194,   524,   460,
     628,  -596,   528,    66,  1069,   529,   501,   191,   517,   603,
     521,  -596,   534,   522,   531,  -596,    66,   543,   371,  -596,
     538,   540,    66,   531,   508,   525,   508,   545,   542,   508,
     547,    66,   541,  1069,  -596,   176,  -596,   176,   582,  -596,
     568,   528,    66,   508,  -596,   668,   405,  -596,  -596,   560,
    -596,  -596,   191,   861,    66,   587,    66,   628,   528,    66,
    1069,   191,  -596,  -596,  -596,  -596,  -596,  -596,  -596,  -596,
    -596,  1401,   564,   571,   563,    66,   577,    66,   302,  -596,
     445,  -596,   176,   302,   606,   580,   569,   531,   586,    66,
     531,   588,   176,   578,  1573,  1427,  -596,   191,    66,   584,
     597,  -596,  -596,   599,   903,  -596,    66,   508,   913,  -596,
     191,   955,  -596,  -596,  1401,  1401,    66,   601,  -596,  1401,
     531,    66,  -596,   606,   302,  -596,   590,    66,   302,  -596,
     176,   302,   606,  -596,   133,   175,   591,  1401,   176,   965,
     604,  -596,   608,    66,   611,   621,  -596,   622,  -596,  -596,
    1401,  1401,  1324,   625,  1401,  -596,   234,   -25,   302,  -596,
      66,  -596,   531,    66,  -596,   606,   146,  -596,   596,   341,
    1401,   287,  -596,   629,   531,   972,   624,  -596,  -596,  -596,
    -596,  -596,  -596,  -596,   979,   302,  -596,    66,   302,  -596,
     637,   531,   638,  -596,  1031,  -596,   302,  -596,   639,  -596
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
       8,     0,     0,     0,    62,    72,   396,   397,   311,   267,
     304,     0,   154,   154,   154,     0,   162,   162,   162,   162,
       0,   156,     0,     0,     0,     0,    80,   228,   229,    74,
      81,    82,    83,    84,     0,    85,    73,   231,   230,     9,
     262,   254,   255,   256,   257,   258,   260,   261,   259,   252,
     253,    78,    79,    70,   271,     0,     0,     0,    71,     0,
     305,   304,     0,     0,   113,     0,    99,   100,   101,   102,
     110,   111,     0,   114,   115,     0,    97,   119,   120,   125,
     126,   127,   128,   147,     0,   155,     0,     0,     0,     0,
     244,   232,     0,     0,     0,     0,     0,     0,     0,   169,
       0,     0,   234,   246,   233,     0,     0,   162,   162,   162,
     162,     0,   156,   219,   220,   221,   222,   223,    10,    68,
     297,   279,   280,   281,   282,   283,   289,   290,   291,   296,
     284,   285,   286,   287,   288,   166,   292,   294,   295,     0,
     275,     0,   131,   132,   133,   141,   268,     0,     0,     0,
     304,   301,   304,     0,   312,     0,     0,   129,   109,   112,
     103,   104,   107,   108,    97,    95,   117,   121,   122,   123,
     130,     0,   146,     0,   150,   238,   235,     0,   240,     0,
     173,   174,     0,   164,    97,   185,   185,   185,   185,   168,
       0,     0,   171,     0,     0,     0,     0,     0,   160,   161,
       0,   158,   182,     0,     0,   128,     0,   216,     0,     9,
       0,     0,     0,     0,     0,     0,   167,   293,   270,     0,
       0,   134,   135,   140,     0,     0,    77,    64,    63,     0,
     302,     0,     0,   304,   266,     0,   105,   106,   118,    91,
      92,    93,    96,     0,    90,     0,   145,     0,     0,   394,
     150,   152,   304,   154,     0,   154,   154,     0,   154,   245,
     163,     0,   116,     0,     0,     0,     0,     0,     0,   194,
       0,   170,   185,   185,   157,     0,   175,     0,   204,    62,
       0,     0,   214,   206,     0,   218,    76,   185,   185,   185,
     185,   185,     0,   277,     0,   273,     0,   139,     0,     0,
      97,   304,   301,   304,   304,   309,   150,     0,    98,     0,
       0,     0,   144,   151,     0,   148,     0,     0,     0,     0,
       0,     0,   165,   187,   186,     0,   224,   189,   190,   191,
     192,   193,   172,     0,     0,   159,   176,   183,     0,   175,
       0,     0,     0,   212,   213,     0,   207,   208,   209,   215,
     217,     0,     0,     0,     0,     0,   175,   202,     0,     0,
     276,     0,     0,   138,     0,   307,   303,   308,   306,   153,
      94,     0,     0,   143,   395,   149,   239,     0,   236,     0,
       0,   241,     0,   251,     0,     0,     0,     0,     0,   247,
     248,   195,   196,     0,   181,   184,     0,   205,     0,   197,
     198,   199,   200,   201,     0,   272,   274,     0,     0,   137,
       0,    75,     0,   142,   154,   154,   154,   188,     0,     0,
     249,     9,   250,   227,   177,   204,   204,     0,   278,   136,
       0,     0,     0,   338,   313,   304,   333,     0,     0,     0,
       0,     0,     0,    62,     0,     0,   225,     0,     0,   210,
     211,   203,     0,   304,   304,   304,   175,     0,     0,   337,
     124,   237,   243,   242,     0,     0,     0,   178,   179,     0,
       0,     0,     0,   310,     0,   314,   316,   334,     0,   383,
       0,     0,     0,     0,     0,   354,     0,     0,     0,   343,
     304,   264,   372,   344,   341,   317,   318,   319,   299,   298,
     300,   315,     0,   389,   304,     0,   304,     0,   392,     0,
       0,   353,     0,   304,     0,     0,     0,     0,     0,     0,
       0,   387,     0,     0,     0,   390,   304,     0,     0,   356,
       0,     0,   304,     0,     0,     0,     0,     0,   354,     0,
       0,   304,     0,   350,   352,     9,   347,     9,     0,   263,
       0,     0,   304,     0,   388,     0,     0,   393,   355,     0,
     371,   349,     0,     0,   304,     0,   304,     0,     0,   304,
       0,     0,   373,   351,   345,   382,   342,   320,   321,   322,
     340,     0,     0,   335,     0,   304,     0,   304,     0,   380,
       0,   357,     9,     0,   384,     0,     0,     0,     0,   304,
       0,     0,     9,     0,     0,     0,   339,     0,   304,     0,
       0,   391,   370,     0,     0,   378,   304,     0,     0,   359,
       0,     0,   360,   369,     0,     0,   304,     0,   336,     0,
       0,   304,   381,   384,     0,   385,     0,   304,     0,   367,
       9,     0,   384,   323,     0,     0,     0,     0,     0,     0,
       0,   379,     0,   304,     0,     0,   358,     0,   365,   331,
       0,     0,     0,     0,     0,   329,     0,   265,     0,   375,
     304,   386,     0,   304,   368,   384,     0,   325,     0,     0,
       0,     0,   332,     0,     0,     0,     0,   366,   328,   327,
     326,   324,   330,   374,     0,     0,   362,   304,     0,   376,
       0,     0,     0,   361,     0,   377,     0,   363,     0,   364
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -596,  -596,   719,  -596,   -55,  -280,    -1,   -61,   652,   669,
     -46,  -596,  -596,  -596,  -277,  -596,  -215,  -596,  -138,   -84,
    -124,  -129,  -126,  -169,   583,   509,  -596,   -85,  -596,  -596,
    -297,  -596,  -596,   -80,   539,   374,  -596,    43,   391,  -596,
    -596,   554,   385,  -596,   198,  -596,  -596,  -272,  -596,  -105,
     274,  -596,  -596,  -596,  -128,  -596,  -596,  -596,  -596,  -596,
    -596,  -333,   394,  -596,   409,   664,  -596,   -74,   270,   685,
    -596,  -596,   536,  -596,  -596,  -596,  -596,   332,  -596,   301,
     338,  -596,   364,  -284,  -596,  -596,   419,   -86,  -487,   -67,
    -529,  -596,  -596,  -551,  -596,  -596,  -448,   119,  -470,  -596,
    -596,   216,  -569,   169,  -577,   204,  -562,  -596,  -524,  -587,
    -558,  -595,  -485,  -596,   218,   236,   190,  -596,  -596
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     3,     4,    73,   406,   199,   264,   156,     5,    64,
      74,    75,    76,   321,   322,   323,   246,   157,   265,   158,
     159,   160,   161,   162,   163,   225,   226,   324,   394,   330,
     331,   107,   108,   166,   181,   280,   281,   173,   262,   297,
     272,   178,   273,   263,   418,   528,   419,   420,   109,   344,
     404,   110,   111,   112,   179,   113,   193,   194,   195,   196,
     197,   423,   362,   287,   288,   465,   115,   407,   466,   467,
     117,   118,   171,   184,   468,   469,   132,   470,    77,   227,
     136,   375,   376,   219,   220,   581,   311,   601,   515,   570,
     235,   516,   662,   724,   707,   663,   517,   664,   491,   631,
     599,   571,   595,   610,   622,   592,   572,   624,   596,   695,
     602,   635,   584,   588,   589,   332,   455,    78,    79
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[] =
{
      57,    58,    63,    63,   142,    90,   164,   223,    85,   366,
     224,   170,   222,   167,   169,   373,   285,    89,   643,   236,
     131,   573,   138,   533,   534,   535,   424,   626,   604,   318,
     133,   416,   623,   393,   250,   613,   266,   267,   268,   639,
     416,   165,   641,   282,   518,   359,   653,   416,   260,   342,
     275,   139,   250,    59,   545,   233,    80,   397,    84,   186,
     400,   623,   329,   294,   198,   609,   611,    82,   627,    86,
      87,   261,   302,   681,   666,   573,   574,   586,   307,   360,
     155,   593,   698,    81,   251,   701,   252,   253,   623,   449,
     557,   228,   223,   271,   244,   224,   168,   222,   417,   690,
     286,   182,   251,   254,   693,   255,   256,   137,   258,   689,
     450,   669,   600,   672,  -180,   729,   553,   605,   554,   674,
     119,   308,   682,   611,   303,   304,   731,   140,   269,   644,
     710,   646,    84,   270,   649,   738,    61,   352,    62,   353,
     174,   175,   176,   305,   309,   732,   312,   474,   667,   735,
       1,     2,   737,   730,   155,   141,    84,   765,   709,    84,
     270,   345,   346,   347,   484,   444,   316,   317,   767,   774,
     715,   720,   529,   530,   719,   170,   746,   722,   139,   763,
     314,   155,    84,   426,   427,   333,   784,   165,   230,   756,
     271,   759,   764,   761,   231,   285,   139,   706,   232,    83,
     691,    84,   172,   508,   155,   749,   780,   238,  -206,   782,
    -206,   239,   717,   177,  -204,   299,  -204,   788,   361,   300,
     739,   526,   740,   244,   422,   741,   742,   385,   180,   743,
     290,   291,   292,   293,   198,   740,   768,   139,   741,   742,
     183,   776,   743,   134,   495,   139,   395,   413,   414,   334,
     779,   185,   335,   396,   386,   398,   399,    61,   401,   403,
     787,   744,   431,   432,   433,   434,   435,   229,   553,   348,
     139,   408,   409,   410,   245,   234,    61,   428,    88,   286,
    -269,  -269,   358,   490,    61,   363,   405,   329,   247,   367,
     368,   369,   370,   371,   372,   445,   248,   447,   448,   374,
    -269,   278,   279,   559,   380,   249,  -269,  -269,  -269,  -269,
    -269,  -269,  -269,  -269,  -269,   326,   327,   437,    84,   578,
     579,   762,  -269,   740,   389,   139,   741,   742,   257,   337,
     743,   259,   338,   473,    61,   135,   436,   477,   459,   471,
     472,   240,   241,   242,   243,   654,   289,   655,   560,   561,
     562,   563,   564,   565,   566,   377,   378,   479,   480,   481,
     482,   483,   274,  -311,   276,   296,   298,   144,   145,   306,
     223,   310,   559,   224,   772,   222,   740,   315,   403,   741,
     742,   567,   313,   743,   325,    88,  -311,    84,   319,   320,
     328,  -311,   692,   146,   147,   148,   149,   150,   151,   152,
     153,   154,   703,   336,   341,   514,   340,   514,   245,   155,
     343,   349,   502,   269,   519,   520,   521,   560,   561,   562,
     563,   564,   565,   566,   350,   532,   532,   532,   351,   537,
     740,   354,  -311,   741,   742,   770,   355,   743,   374,   356,
     736,   365,   364,   302,   379,   381,   198,   550,   551,   552,
     567,   382,   531,   514,    88,   638,   384,   383,   387,   463,
    -311,   559,   388,   329,    91,    92,    93,    94,    95,   390,
     391,   392,   504,   548,   411,   505,   102,   103,   421,   422,
     104,   425,   361,   438,   597,   439,   440,   441,   451,   442,
     569,   443,   462,   580,   452,   453,   456,   457,   524,   454,
     460,   458,   559,   464,   461,   475,   560,   561,   562,   563,
     564,   565,   566,   536,   187,   188,   189,   190,   191,   192,
     636,   476,   478,   416,   546,   612,   642,   621,   485,   487,
     488,   549,   489,   493,   490,   651,   492,   494,   496,   567,
     464,   559,  -226,   608,   569,   661,   501,   560,   561,   562,
     563,   564,   565,   566,   506,   507,   621,   582,   675,   509,
     677,   527,   538,   680,   665,   539,   540,   544,   541,   542,
     198,   543,   198,   -11,   657,   558,   547,   556,   553,   687,
     567,   679,   575,   621,    88,  -346,   560,   561,   562,   563,
     564,   565,   566,   700,   576,   590,   577,   583,   585,   705,
     661,   587,   591,   594,   559,   598,   144,   145,   603,   607,
     716,    88,   628,   625,   630,   632,   633,   198,   645,   567,
     726,   634,   640,    88,  -348,   652,    84,   198,   647,   559,
     650,   734,   146,   147,   148,   149,   150,   151,   152,   153,
     154,   656,   658,   659,   671,   676,   684,   752,   155,   560,
     561,   562,   563,   564,   565,   566,   685,   686,   688,   694,
     683,   696,   660,   697,   704,   198,   711,   766,   144,   559,
     699,   733,   702,   747,   560,   561,   562,   563,   564,   565,
     566,   712,   567,   713,    61,   745,   568,   727,    84,   750,
     769,   781,   751,   753,   146,   147,   148,   149,   150,   151,
     152,   153,   154,   723,   725,   754,   755,   567,   728,   777,
     155,    88,   760,   773,   560,   561,   562,   563,   564,   565,
     566,   783,   785,   789,    60,   106,   723,    65,   237,   415,
     301,   295,   402,   277,   555,   412,   497,   114,   503,   723,
     757,   723,   134,   723,  -269,  -269,  -269,   567,  -269,  -269,
    -269,   668,  -269,  -269,  -269,  -269,  -269,   429,   116,   771,
    -269,  -269,  -269,  -269,  -269,  -269,  -269,  -269,  -269,  -269,
    -269,  -269,  -269,   430,  -269,  -269,  -269,  -269,  -269,  -269,
    -269,  -269,  -269,  -269,  -269,  -269,  -269,  -269,  -269,  -269,
    -269,  -269,  -269,  -269,  -269,   339,  -269,   500,  -269,  -269,
     525,   446,   499,   486,   708,  -269,  -269,  -269,  -269,  -269,
    -269,  -269,  -269,  -269,  -269,   629,   678,  -269,  -269,  -269,
    -269,  -269,   648,   606,     0,   637,   670,     6,     7,     8,
       0,     9,    10,    11,   498,    12,    13,    14,    15,    16,
       0,     0,     0,    17,    18,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,    29,     0,    30,    31,    32,
      33,    34,   559,     0,    35,    36,    37,    38,    39,    40,
      41,    42,    43,    44,    45,    46,    47,    48,     0,    49,
       0,    50,    51,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    52,     0,     0,
      53,    54,    55,    56,   559,     0,     0,   560,   561,   562,
     563,   564,   565,   566,   559,    66,   357,    -5,    -5,    67,
      -5,    -5,    -5,    -5,    -5,    -5,    -5,    -5,    -5,    -5,
      -5,     0,    -5,    -5,     0,     0,    -5,     0,     0,     0,
     567,     0,     0,     0,   673,     0,     0,     0,     0,   560,
     561,   562,   563,   564,   565,   566,   559,     0,     0,   560,
     561,   562,   563,   564,   565,   566,   559,     0,     0,     0,
      68,    69,     0,   559,     0,     0,    70,    71,     0,     0,
     559,     0,   567,     0,     0,     0,   714,     0,     0,     0,
      72,     0,   567,     0,     0,     0,   718,    -5,   -69,     0,
       0,   560,   561,   562,   563,   564,   565,   566,     0,     0,
       0,   560,   561,   562,   563,   564,   565,   566,   560,   561,
     562,   563,   564,   565,   566,   560,   561,   562,   563,   564,
     565,   566,   559,     0,   567,     0,     0,     0,   721,     0,
       0,     0,     0,     0,   567,     0,     0,     0,   748,     0,
       0,   567,     0,     0,     0,   775,     0,     0,   567,     0,
       0,     0,   778,   120,   121,   122,   123,     0,   124,   125,
     126,   127,   128,     0,     0,     0,     0,   560,   561,   562,
     563,   564,   565,   566,     1,     2,     0,    91,    92,    93,
      94,    95,    96,    97,    98,    99,   100,   101,     0,   102,
     103,     0,   129,   104,     0,     0,     0,     0,     0,     0,
     567,     0,     0,     0,   786,   614,   615,   616,   563,   617,
     618,   619,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    61,     0,
       0,   130,     0,     0,     0,     0,     0,     0,   620,     6,
       7,     8,    88,     9,    10,    11,     0,    12,    13,    14,
      15,    16,     0,     0,   105,    17,    18,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,     0,    30,
      31,    32,    33,    34,   144,   221,    35,    36,    37,    38,
      39,    40,    41,    42,    43,    44,    45,    46,    47,    48,
       0,    49,     0,    50,   523,     0,     0,   200,     0,     0,
     146,   147,   148,   149,   150,   151,   152,   153,   154,    52,
       0,     0,    53,    54,    55,    56,   155,   201,     0,   202,
     203,   204,   205,   206,   207,   208,     0,     0,   209,   210,
     211,   212,   213,   214,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     215,   216,     0,     0,     0,   200,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   522,
       0,     0,     0,   217,   218,   201,     0,   202,   203,   204,
     205,   206,   207,   208,     0,     0,   209,   210,   211,   212,
     213,   214,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   215,   216,
       0,     0,     0,     0,     0,     0,     0,     6,     7,     8,
       0,     9,    10,    11,     0,    12,    13,    14,    15,    16,
       0,   217,   218,    17,    18,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,    29,     0,    30,    31,    32,
      33,    34,     0,     0,    35,    36,    37,    38,    39,    40,
      41,    42,    43,    44,    45,    46,    47,    48,     0,    49,
       0,    50,    51,   758,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    52,     0,     0,
      53,    54,    55,    56,     6,     7,     8,     0,     9,    10,
      11,     0,    12,    13,    14,    15,    16,     0,     0,     0,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,   657,    30,    31,    32,    33,    34,     0,
       0,    35,    36,    37,    38,    39,    40,    41,    42,    43,
      44,    45,    46,    47,    48,     0,    49,     0,    50,    51,
       0,   143,     0,     0,     0,   144,   145,     0,     0,     0,
       0,     0,     0,     0,    52,     0,     0,    53,    54,    55,
      56,     0,     0,   144,   145,    84,     0,     0,     0,     0,
       0,   146,   147,   148,   149,   150,   151,   152,   153,   154,
     283,   658,   659,    84,     0,     0,     0,   155,     0,   146,
     147,   148,   149,   150,   151,   152,   153,   154,     0,     0,
     144,   145,     0,     0,     0,   155,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      84,   144,   145,   510,   511,   512,   146,   147,   148,   149,
     150,   151,   152,   153,   154,     0,     0,     0,     0,     0,
       0,    84,   284,     0,     0,     0,     0,   146,   147,   148,
     149,   150,   151,   152,   153,   154,   144,   145,   510,   511,
     512,     0,     0,   155,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   513,   144,   221,    84,     0,     0,     0,
       0,     0,   146,   147,   148,   149,   150,   151,   152,   153,
     154,   144,   145,     0,    84,     0,     0,     0,   155,     0,
     146,   147,   148,   149,   150,   151,   152,   153,   154,     0,
       0,    84,     0,     0,     0,     0,   155,   146,   147,   148,
     149,   150,   151,   152,   153,   154,     0,     0,     0,     0,
       0,     0,     0,   155
};

static const yytype_int16 yycheck[] =
{
       1,     2,    57,    58,    90,    72,    91,   136,    69,   289,
     136,    95,   136,    93,    94,   299,   185,    72,   613,   143,
      75,   545,    83,   510,   511,   512,   359,   596,   586,   244,
      76,    17,   594,   330,    39,   593,   174,   175,   176,   608,
      17,    38,   611,   181,   492,    58,   623,    17,    31,   264,
     178,    82,    39,     0,   524,   141,    81,   334,    58,   105,
     337,   623,    93,   191,   119,   589,   590,    68,   597,    70,
      71,    54,    39,   650,   632,   599,   546,   562,    15,    92,
      80,   566,   677,    45,    89,   680,    91,    92,   650,   386,
     538,   137,   221,   177,   155,   221,    93,   221,    84,   668,
     185,   102,    89,   164,   673,   166,   167,    83,   169,   667,
     387,   635,   582,   642,    84,   710,    93,   587,    95,   643,
      84,    58,   651,   647,    91,    92,   713,    61,    54,   614,
     688,   616,    58,    59,   619,   722,    81,   275,    83,   277,
      97,    98,    99,   227,   230,   714,   232,   419,   633,   718,
       3,     4,   721,   711,    80,    89,    58,   752,   687,    58,
      59,   266,   267,   268,   436,   380,    64,    65,   755,   764,
     694,   700,   505,   506,   698,   259,   727,   701,    82,   748,
     235,    80,    58,    59,    60,    89,   781,    38,    83,   740,
     274,   742,   750,   744,    89,   364,    82,   684,    93,    56,
     670,    58,    93,   487,    80,   729,   775,    65,    85,   778,
      87,    69,   697,    93,    85,    85,    87,   786,    95,    89,
      87,   501,    89,   284,    95,    92,    93,   313,    93,    96,
     187,   188,   189,   190,   289,    89,    90,    82,    92,    93,
      60,   765,    96,     1,    89,    82,   332,   352,   353,    86,
     774,    86,    89,   333,   315,   335,   336,    81,   338,   343,
     784,    86,   367,   368,   369,   370,   371,    82,    93,   270,
      82,   345,   346,   347,    86,    84,    81,   361,    83,   364,
      38,    39,   283,    89,    81,   286,    83,    93,    88,   290,
     291,   292,   293,   294,   295,   381,    88,   383,   384,   300,
      58,    43,    44,     1,   305,    88,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    91,    92,   372,    58,    59,
      60,    87,    80,    89,   325,    82,    92,    93,    73,    86,
      96,    85,    89,   418,    81,    93,    83,   422,   399,   413,
     414,    64,    65,    66,    67,   625,    84,   627,    46,    47,
      48,    49,    50,    51,    52,    91,    92,   431,   432,   433,
     434,   435,    93,    61,    93,    60,    94,    38,    39,    84,
     499,    61,     1,   499,    87,   499,    89,    88,   462,    92,
      93,    79,    90,    96,    88,    83,    84,    58,    59,    60,
      92,    89,   672,    64,    65,    66,    67,    68,    69,    70,
      71,    72,   682,    94,    85,   490,    94,   492,    86,    80,
      82,    94,   467,    54,   494,   495,   496,    46,    47,    48,
      49,    50,    51,    52,    85,   510,   511,   512,    94,   515,
      89,    94,    61,    92,    93,    94,    85,    96,   439,    83,
     720,    87,    85,    39,    92,    84,   501,   533,   534,   535,
      79,    85,   507,   538,    83,    84,    94,    90,    85,     1,
      89,     1,    87,    93,     6,     7,     8,     9,    10,    92,
      92,    92,   473,   528,    94,   476,    18,    19,    92,    95,
      22,    92,    95,    82,   570,    85,    90,    92,    90,    92,
     545,    92,    85,   554,    92,    92,    90,    87,   499,    94,
      87,    90,     1,    45,    90,    84,    46,    47,    48,    49,
      50,    51,    52,   514,    11,    12,    13,    14,    15,    16,
     606,    92,    92,    17,   525,   592,   612,   594,    59,    85,
      92,   532,    92,    92,    89,   621,    89,    89,    89,    79,
      45,     1,    84,    83,   599,   630,    84,    46,    47,    48,
      49,    50,    51,    52,    58,    84,   623,   558,   644,    92,
     646,    93,    85,   649,   631,    90,    90,    94,    90,    90,
     625,    90,   627,    89,     6,    38,    94,    94,    93,   665,
      79,   648,    94,   650,    83,    84,    46,    47,    48,    49,
      50,    51,    52,   679,    94,    83,    94,    89,    93,   684,
     685,    58,    60,    83,     1,    20,    38,    39,    58,    85,
     696,    83,    95,    84,    93,    81,    94,   672,    93,    79,
     706,    90,    84,    83,    84,    84,    58,   682,    83,     1,
      83,   717,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    59,    74,    75,    84,    58,    82,   733,    80,    46,
      47,    48,    49,    50,    51,    52,    85,    94,    81,    53,
     661,    81,    94,    94,    86,   720,    82,   753,    38,     1,
      84,    81,    84,   728,    46,    47,    48,    49,    50,    51,
      52,    84,    79,    84,    81,    94,    83,    86,    58,    85,
      94,   777,    84,    82,    64,    65,    66,    67,    68,    69,
      70,    71,    72,   704,   705,    84,    84,    79,   709,    85,
      80,    83,    87,    84,    46,    47,    48,    49,    50,    51,
      52,    84,    84,    84,     5,    73,   727,    58,   145,   355,
     221,   192,   341,   179,   536,   350,   462,    73,   468,   740,
     741,   742,     1,   744,     3,     4,     5,    79,     7,     8,
       9,    83,    11,    12,    13,    14,    15,   363,    73,   760,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,   364,    33,    34,    35,    36,    37,    38,
      39,    40,    41,    42,    43,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,   259,    55,   465,    57,    58,
     499,   382,   464,   439,   685,    64,    65,    66,    67,    68,
      69,    70,    71,    72,    73,   599,   647,    76,    77,    78,
      79,    80,   618,   587,    -1,   607,   636,     3,     4,     5,
      -1,     7,     8,     9,    93,    11,    12,    13,    14,    15,
      -1,    -1,    -1,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    31,    -1,    33,    34,    35,
      36,    37,     1,    -1,    40,    41,    42,    43,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    -1,    55,
      -1,    57,    58,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    73,    -1,    -1,
      76,    77,    78,    79,     1,    -1,    -1,    46,    47,    48,
      49,    50,    51,    52,     1,     1,    92,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    14,    15,
      16,    -1,    18,    19,    -1,    -1,    22,    -1,    -1,    -1,
      79,    -1,    -1,    -1,    83,    -1,    -1,    -1,    -1,    46,
      47,    48,    49,    50,    51,    52,     1,    -1,    -1,    46,
      47,    48,    49,    50,    51,    52,     1,    -1,    -1,    -1,
      56,    57,    -1,     1,    -1,    -1,    62,    63,    -1,    -1,
       1,    -1,    79,    -1,    -1,    -1,    83,    -1,    -1,    -1,
      76,    -1,    79,    -1,    -1,    -1,    83,    83,    84,    -1,
      -1,    46,    47,    48,    49,    50,    51,    52,    -1,    -1,
      -1,    46,    47,    48,    49,    50,    51,    52,    46,    47,
      48,    49,    50,    51,    52,    46,    47,    48,    49,    50,
      51,    52,     1,    -1,    79,    -1,    -1,    -1,    83,    -1,
      -1,    -1,    -1,    -1,    79,    -1,    -1,    -1,    83,    -1,
      -1,    79,    -1,    -1,    -1,    83,    -1,    -1,    79,    -1,
      -1,    -1,    83,     6,     7,     8,     9,    -1,    11,    12,
      13,    14,    15,    -1,    -1,    -1,    -1,    46,    47,    48,
      49,    50,    51,    52,     3,     4,    -1,     6,     7,     8,
       9,    10,    11,    12,    13,    14,    15,    16,    -1,    18,
      19,    -1,    45,    22,    -1,    -1,    -1,    -1,    -1,    -1,
      79,    -1,    -1,    -1,    83,    46,    47,    48,    49,    50,
      51,    52,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    81,    -1,
      -1,    84,    -1,    -1,    -1,    -1,    -1,    -1,    79,     3,
       4,     5,    83,     7,     8,     9,    -1,    11,    12,    13,
      14,    15,    -1,    -1,    83,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,    29,    30,    31,    -1,    33,
      34,    35,    36,    37,    38,    39,    40,    41,    42,    43,
      44,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      -1,    55,    -1,    57,    58,    -1,    -1,     1,    -1,    -1,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      -1,    -1,    76,    77,    78,    79,    80,    21,    -1,    23,
      24,    25,    26,    27,    28,    29,    -1,    -1,    32,    33,
      34,    35,    36,    37,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      54,    55,    -1,    -1,    -1,     1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    73,
      -1,    -1,    -1,    77,    78,    21,    -1,    23,    24,    25,
      26,    27,    28,    29,    -1,    -1,    32,    33,    34,    35,
      36,    37,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    54,    55,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,     3,     4,     5,
      -1,     7,     8,     9,    -1,    11,    12,    13,    14,    15,
      -1,    77,    78,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    31,    -1,    33,    34,    35,
      36,    37,    -1,    -1,    40,    41,    42,    43,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    -1,    55,
      -1,    57,    58,    59,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    73,    -1,    -1,
      76,    77,    78,    79,     3,     4,     5,    -1,     7,     8,
       9,    -1,    11,    12,    13,    14,    15,    -1,    -1,    -1,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,     6,    33,    34,    35,    36,    37,    -1,
      -1,    40,    41,    42,    43,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    -1,    55,    -1,    57,    58,
      -1,    16,    -1,    -1,    -1,    38,    39,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    73,    -1,    -1,    76,    77,    78,
      79,    -1,    -1,    38,    39,    58,    -1,    -1,    -1,    -1,
      -1,    64,    65,    66,    67,    68,    69,    70,    71,    72,
      18,    74,    75,    58,    -1,    -1,    -1,    80,    -1,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    -1,    -1,
      38,    39,    -1,    -1,    -1,    80,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      58,    38,    39,    40,    41,    42,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    -1,    -1,    -1,    -1,    -1,
      -1,    58,    80,    -1,    -1,    -1,    -1,    64,    65,    66,
      67,    68,    69,    70,    71,    72,    38,    39,    40,    41,
      42,    -1,    -1,    80,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    90,    38,    39,    58,    -1,    -1,    -1,
      -1,    -1,    64,    65,    66,    67,    68,    69,    70,    71,
      72,    38,    39,    -1,    58,    -1,    -1,    -1,    80,    -1,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    -1,
      -1,    58,    -1,    -1,    -1,    -1,    80,    64,    65,    66,
      67,    68,    69,    70,    71,    72,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    80
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     3,     4,    98,    99,   105,     3,     4,     5,     7,
       8,     9,    11,    12,    13,    14,    15,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      33,    34,    35,    36,    37,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    55,
      57,    58,    73,    76,    77,    78,    79,   103,   103,     0,
      99,    81,    83,   101,   106,   106,     1,     5,    56,    57,
      62,    63,    76,   100,   107,   108,   109,   175,   214,   215,
      81,    45,   103,    56,    58,   104,   103,   103,    83,   101,
     186,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    18,    19,    22,    83,   105,   128,   129,   145,
     148,   149,   150,   152,   162,   163,   166,   167,   168,    84,
       6,     7,     8,     9,    11,    12,    13,    14,    15,    45,
      84,   101,   173,   107,     1,    93,   177,    83,   104,    82,
      61,    89,   184,    16,    38,    39,    64,    65,    66,    67,
      68,    69,    70,    71,    72,    80,   104,   114,   116,   117,
     118,   119,   120,   121,   124,    38,   130,   130,    93,   130,
     116,   169,    93,   134,   134,   134,   134,    93,   138,   151,
      93,   131,   103,    60,   170,    86,   107,    11,    12,    13,
      14,    15,    16,   153,   154,   155,   156,   157,   101,   102,
       1,    21,    23,    24,    25,    26,    27,    28,    29,    32,
      33,    34,    35,    36,    37,    54,    55,    77,    78,   180,
     181,    39,   117,   118,   119,   122,   123,   176,   107,    82,
      83,    89,    93,   184,    84,   187,   117,   121,    65,    69,
      64,    65,    66,    67,   104,    86,   113,    88,    88,    88,
      39,    89,    91,    92,   104,   104,   104,    73,   104,    85,
      31,    54,   135,   140,   103,   115,   115,   115,   115,    54,
      59,   116,   137,   139,    93,   151,    93,   138,    43,    44,
     132,   133,   115,    18,    80,   120,   124,   160,   161,    84,
     134,   134,   134,   134,   151,   131,    60,   136,    94,    85,
      89,   122,    39,    91,    92,   116,    84,    15,    58,   184,
      61,   183,   184,    90,   101,    88,    64,    65,   113,    59,
      60,   110,   111,   112,   124,    88,    91,    92,    92,    93,
     126,   127,   212,    89,    86,    89,    94,    86,    89,   169,
      94,    85,   113,    82,   146,   146,   146,   146,   103,    94,
      85,    94,   115,   115,    94,    85,    83,    92,   103,    58,
      92,    95,   159,   103,    85,    87,   102,   103,   103,   103,
     103,   103,   103,   180,   103,   178,   179,    91,    92,    92,
     103,    84,    85,    90,    94,   184,   104,    85,    87,   103,
      92,    92,    92,   127,   125,   184,   130,   111,   130,   130,
     111,   130,   135,   116,   147,    83,   101,   164,   164,   164,
     164,    94,   139,   146,   146,   132,    17,    84,   141,   143,
     144,    92,    95,   158,   158,    92,    59,    60,   116,   159,
     161,   146,   146,   146,   146,   146,    83,   101,    82,    85,
      90,    92,    92,    92,   113,   184,   183,   184,   184,   127,
     111,    90,    92,    92,    94,   213,    90,    87,    90,   104,
      87,    90,    85,     1,    45,   162,   165,   166,   171,   172,
     174,   164,   164,   124,   144,    84,    92,   124,    92,   164,
     164,   164,   164,   164,   144,    59,   179,    85,    92,    92,
      89,   195,    89,    92,    89,    89,    89,   147,    93,   177,
     174,    84,   101,   165,   103,   103,    58,    84,   180,    92,
      40,    41,    42,    90,   124,   185,   188,   193,   193,   130,
     130,   130,    73,    58,   103,   176,   102,    93,   142,   158,
     158,   101,   124,   185,   185,   185,   103,   184,    85,    90,
      90,    90,    90,    90,    94,   195,   103,    94,   101,   103,
     184,   184,   184,    93,    95,   141,    94,   193,    38,     1,
      46,    47,    48,    49,    50,    51,    52,    79,    83,   101,
     186,   198,   203,   205,   195,    94,    94,    94,    59,    60,
     104,   182,   103,    89,   209,    93,   209,    58,   210,   211,
      83,    60,   202,   209,    83,   199,   205,   184,    20,   197,
     195,   184,   207,    58,   207,   195,   212,    85,    83,   205,
     200,   205,   186,   207,    46,    47,    48,    50,    51,    52,
      79,   186,   201,   203,   204,    84,   199,   187,    95,   198,
      93,   196,    81,    94,    90,   208,   184,   211,    84,   199,
      84,   199,   184,   208,   209,    93,   209,    83,   202,   209,
      83,   184,    84,   201,   102,   102,    59,     6,    74,    75,
      94,   124,   189,   192,   194,   186,   207,   209,    83,   205,
     213,    84,   187,    83,   205,   184,    58,   184,   200,   186,
     184,   201,   187,   103,    82,    85,    94,   184,    81,   207,
     199,   195,   102,   199,    53,   206,    81,    94,   208,    84,
     184,   208,    84,   102,    86,   124,   185,   191,   194,   187,
     207,    82,    84,    84,    83,   205,   184,   209,    83,   205,
     187,    83,   205,   103,   190,   103,   184,    86,   103,   208,
     207,   206,   199,    81,   184,   199,   102,   199,   206,    87,
      89,    92,    93,    96,    86,    94,   190,   101,    83,   205,
      85,    84,   184,    82,    84,    84,   190,   103,    59,   190,
      87,   190,    87,   199,   207,   208,   184,   206,    90,    94,
      94,   103,    87,    84,   208,    83,   205,    85,    83,   205,
     199,   184,   199,    84,   208,    84,    83,   205,   199,    84
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
     114,   114,   114,   114,   114,   114,   115,   116,   116,   117,
     117,   118,   119,   119,   120,   121,   121,   121,   121,   121,
     121,   122,   122,   122,   122,   122,   123,   123,   123,   123,
     123,   123,   124,   124,   124,   124,   124,   124,   125,   126,
     127,   127,   128,   129,   130,   130,   131,   131,   132,   132,
     133,   133,   134,   134,   135,   135,   136,   136,   137,   138,
     138,   139,   139,   140,   140,   141,   141,   142,   142,   143,
     144,   144,   145,   145,   145,   146,   146,   147,   147,   148,
     148,   149,   150,   151,   151,   152,   152,   153,   153,   154,
     155,   156,   157,   157,   158,   158,   159,   159,   159,   159,
     160,   160,   160,   160,   160,   160,   161,   161,   162,   163,
     163,   163,   163,   163,   164,   164,   165,   165,   166,   166,
     166,   166,   166,   166,   166,   167,   167,   167,   167,   167,
     168,   168,   168,   168,   169,   169,   170,   171,   172,   172,
     172,   172,   173,   173,   173,   173,   173,   173,   173,   173,
     173,   173,   173,   174,   174,   174,   175,   175,   176,   177,
     177,   177,   178,   179,   179,   180,   180,   180,   180,   181,
     181,   181,   181,   181,   181,   181,   181,   181,   181,   181,
     181,   181,   181,   181,   181,   181,   181,   181,   182,   182,
     182,   183,   183,   183,   184,   184,   184,   184,   184,   184,
     185,   186,   187,   188,   188,   188,   188,   188,   188,   188,
     189,   189,   189,   190,   190,   190,   190,   190,   190,   191,
     192,   192,   192,   193,   193,   194,   194,   195,   195,   196,
     196,   197,   197,   198,   198,   198,   199,   199,   200,   200,
     201,   201,   201,   202,   202,   203,   203,   203,   204,   204,
     204,   204,   204,   204,   204,   204,   204,   204,   204,   204,
     205,   205,   205,   205,   205,   205,   205,   205,   205,   205,
     205,   205,   205,   205,   206,   206,   206,   207,   208,   209,
     210,   210,   211,   211,   212,   213,   214,   215
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
       1,     1,     2,     1,     1,     1,     2,     2,     3,     1,
       1,     2,     2,     2,     8,     1,     1,     1,     1,     2,
       2,     1,     1,     1,     2,     2,     6,     5,     4,     3,
       2,     1,     6,     5,     4,     3,     2,     1,     1,     3,
       0,     2,     4,     6,     0,     1,     0,     3,     1,     3,
       1,     1,     0,     3,     1,     3,     0,     1,     1,     0,
       3,     1,     3,     1,     1,     0,     1,     0,     2,     5,
       1,     2,     3,     5,     6,     0,     2,     1,     3,     5,
       5,     5,     5,     4,     3,     6,     6,     5,     5,     5,
       5,     5,     4,     7,     0,     2,     0,     2,     2,     2,
       6,     6,     3,     3,     2,     3,     1,     3,     4,     2,
       2,     2,     2,     2,     1,     4,     0,     2,     1,     1,
       1,     1,     2,     2,     2,     3,     6,     9,     3,     6,
       3,     6,     9,     9,     1,     3,     1,     1,     1,     2,
       2,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     7,     5,    13,     5,     2,     1,     0,
       3,     1,     3,     1,     3,     1,     4,     3,     6,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     2,     1,     1,     1,     1,     1,     1,
       1,     0,     1,     3,     0,     1,     5,     5,     5,     4,
       3,     1,     1,     1,     3,     4,     3,     4,     4,     4,
       1,     1,     1,     1,     4,     3,     4,     4,     4,     3,
       7,     5,     6,     1,     3,     1,     3,     3,     2,     3,
       2,     0,     3,     1,     1,     4,     1,     2,     1,     2,
       1,     2,     1,     1,     0,     4,     3,     5,     6,     4,
       4,    11,     9,    12,    14,     6,     8,     5,     7,     4,
       6,     4,     1,     4,    11,     9,    12,    14,     6,     8,
       5,     7,     4,     1,     0,     2,     4,     1,     1,     1,
       2,     5,     1,     3,     1,     1,     2,     2
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
#line 2311 "y.tab.c" /* yacc.c:1646  */
    break;

  case 3:
#line 208 "xi-grammar.y" /* yacc.c:1646  */
    { 
		  (yyval.modlist) = 0; 
		}
#line 2319 "y.tab.c" /* yacc.c:1646  */
    break;

  case 4:
#line 212 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.modlist) = new AstChildren<Module>(lineno, (yyvsp[-1].module), (yyvsp[0].modlist)); }
#line 2325 "y.tab.c" /* yacc.c:1646  */
    break;

  case 5:
#line 216 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 2331 "y.tab.c" /* yacc.c:1646  */
    break;

  case 6:
#line 218 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 2337 "y.tab.c" /* yacc.c:1646  */
    break;

  case 7:
#line 222 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 2343 "y.tab.c" /* yacc.c:1646  */
    break;

  case 8:
#line 224 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 2; }
#line 2349 "y.tab.c" /* yacc.c:1646  */
    break;

  case 9:
#line 228 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 2355 "y.tab.c" /* yacc.c:1646  */
    break;

  case 10:
#line 230 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 2361 "y.tab.c" /* yacc.c:1646  */
    break;

  case 11:
#line 235 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 2367 "y.tab.c" /* yacc.c:1646  */
    break;

  case 12:
#line 236 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MODULE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2373 "y.tab.c" /* yacc.c:1646  */
    break;

  case 13:
#line 237 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MAINMODULE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2379 "y.tab.c" /* yacc.c:1646  */
    break;

  case 14:
#line 238 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(EXTERN, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2385 "y.tab.c" /* yacc.c:1646  */
    break;

  case 15:
#line 240 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITCALL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2391 "y.tab.c" /* yacc.c:1646  */
    break;

  case 16:
#line 241 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITNODE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2397 "y.tab.c" /* yacc.c:1646  */
    break;

  case 17:
#line 242 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITPROC, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2403 "y.tab.c" /* yacc.c:1646  */
    break;

  case 18:
#line 244 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CHARE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2409 "y.tab.c" /* yacc.c:1646  */
    break;

  case 19:
#line 245 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MAINCHARE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2415 "y.tab.c" /* yacc.c:1646  */
    break;

  case 20:
#line 246 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(GROUP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2421 "y.tab.c" /* yacc.c:1646  */
    break;

  case 21:
#line 247 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NODEGROUP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2427 "y.tab.c" /* yacc.c:1646  */
    break;

  case 22:
#line 248 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ARRAY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2433 "y.tab.c" /* yacc.c:1646  */
    break;

  case 23:
#line 252 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INCLUDE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2439 "y.tab.c" /* yacc.c:1646  */
    break;

  case 24:
#line 253 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(STACKSIZE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2445 "y.tab.c" /* yacc.c:1646  */
    break;

  case 25:
#line 254 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(THREADED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2451 "y.tab.c" /* yacc.c:1646  */
    break;

  case 26:
#line 255 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(TEMPLATE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2457 "y.tab.c" /* yacc.c:1646  */
    break;

  case 27:
#line 256 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(WHENIDLE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2463 "y.tab.c" /* yacc.c:1646  */
    break;

  case 28:
#line 257 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(SYNC, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2469 "y.tab.c" /* yacc.c:1646  */
    break;

  case 29:
#line 258 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IGET, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2475 "y.tab.c" /* yacc.c:1646  */
    break;

  case 30:
#line 259 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(EXCLUSIVE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2481 "y.tab.c" /* yacc.c:1646  */
    break;

  case 31:
#line 260 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IMMEDIATE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2487 "y.tab.c" /* yacc.c:1646  */
    break;

  case 32:
#line 261 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(SKIPSCHED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2493 "y.tab.c" /* yacc.c:1646  */
    break;

  case 33:
#line 262 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOCOPY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2499 "y.tab.c" /* yacc.c:1646  */
    break;

  case 34:
#line 263 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOCOPYPOST, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2505 "y.tab.c" /* yacc.c:1646  */
    break;

  case 35:
#line 264 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOCOPYDEVICE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2511 "y.tab.c" /* yacc.c:1646  */
    break;

  case 36:
#line 265 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INLINE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2517 "y.tab.c" /* yacc.c:1646  */
    break;

  case 37:
#line 266 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(VIRTUAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2523 "y.tab.c" /* yacc.c:1646  */
    break;

  case 38:
#line 267 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MIGRATABLE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2529 "y.tab.c" /* yacc.c:1646  */
    break;

  case 39:
#line 268 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CREATEHERE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2535 "y.tab.c" /* yacc.c:1646  */
    break;

  case 40:
#line 269 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CREATEHOME, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2541 "y.tab.c" /* yacc.c:1646  */
    break;

  case 41:
#line 270 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOKEEP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2547 "y.tab.c" /* yacc.c:1646  */
    break;

  case 42:
#line 271 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOTRACE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2553 "y.tab.c" /* yacc.c:1646  */
    break;

  case 43:
#line 272 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(APPWORK, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2559 "y.tab.c" /* yacc.c:1646  */
    break;

  case 44:
#line 275 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(PACKED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2565 "y.tab.c" /* yacc.c:1646  */
    break;

  case 45:
#line 276 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(VARSIZE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2571 "y.tab.c" /* yacc.c:1646  */
    break;

  case 46:
#line 277 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ENTRY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2577 "y.tab.c" /* yacc.c:1646  */
    break;

  case 47:
#line 278 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(FOR, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2583 "y.tab.c" /* yacc.c:1646  */
    break;

  case 48:
#line 279 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(FORALL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2589 "y.tab.c" /* yacc.c:1646  */
    break;

  case 49:
#line 280 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(WHILE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2595 "y.tab.c" /* yacc.c:1646  */
    break;

  case 50:
#line 281 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(WHEN, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2601 "y.tab.c" /* yacc.c:1646  */
    break;

  case 51:
#line 282 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(OVERLAP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2607 "y.tab.c" /* yacc.c:1646  */
    break;

  case 52:
#line 283 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(SERIAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2613 "y.tab.c" /* yacc.c:1646  */
    break;

  case 53:
#line 284 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IF, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2619 "y.tab.c" /* yacc.c:1646  */
    break;

  case 54:
#line 285 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ELSE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2625 "y.tab.c" /* yacc.c:1646  */
    break;

  case 55:
#line 287 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(LOCAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2631 "y.tab.c" /* yacc.c:1646  */
    break;

  case 56:
#line 289 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(USING, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2637 "y.tab.c" /* yacc.c:1646  */
    break;

  case 57:
#line 290 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ACCEL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2643 "y.tab.c" /* yacc.c:1646  */
    break;

  case 58:
#line 293 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ACCELBLOCK, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2649 "y.tab.c" /* yacc.c:1646  */
    break;

  case 59:
#line 294 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MEMCRITICAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2655 "y.tab.c" /* yacc.c:1646  */
    break;

  case 60:
#line 295 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(REDUCTIONTARGET, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2661 "y.tab.c" /* yacc.c:1646  */
    break;

  case 61:
#line 296 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CASE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2667 "y.tab.c" /* yacc.c:1646  */
    break;

  case 62:
#line 301 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 2673 "y.tab.c" /* yacc.c:1646  */
    break;

  case 63:
#line 303 "xi-grammar.y" /* yacc.c:1646  */
    {
		  char *tmp = new char[strlen((yyvsp[-3].strval))+strlen((yyvsp[0].strval))+3];
		  sprintf(tmp,"%s::%s", (yyvsp[-3].strval), (yyvsp[0].strval));
		  (yyval.strval) = tmp;
		}
#line 2683 "y.tab.c" /* yacc.c:1646  */
    break;

  case 64:
#line 309 "xi-grammar.y" /* yacc.c:1646  */
    {
		  char *tmp = new char[strlen((yyvsp[-3].strval))+5+3];
		  sprintf(tmp,"%s::array", (yyvsp[-3].strval));
		  (yyval.strval) = tmp;
		}
#line 2693 "y.tab.c" /* yacc.c:1646  */
    break;

  case 65:
#line 316 "xi-grammar.y" /* yacc.c:1646  */
    { 
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist)); 
		}
#line 2701 "y.tab.c" /* yacc.c:1646  */
    break;

  case 66:
#line 320 "xi-grammar.y" /* yacc.c:1646  */
    {  
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist)); 
		    (yyval.module)->setMain();
		}
#line 2710 "y.tab.c" /* yacc.c:1646  */
    break;

  case 67:
#line 327 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = 0; }
#line 2716 "y.tab.c" /* yacc.c:1646  */
    break;

  case 68:
#line 329 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = (yyvsp[-2].conslist); }
#line 2722 "y.tab.c" /* yacc.c:1646  */
    break;

  case 69:
#line 333 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = 0; }
#line 2728 "y.tab.c" /* yacc.c:1646  */
    break;

  case 70:
#line 335 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = new ConstructList(lineno, (yyvsp[-1].construct), (yyvsp[0].conslist)); }
#line 2734 "y.tab.c" /* yacc.c:1646  */
    break;

  case 71:
#line 339 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new UsingScope((yyvsp[0].strval), false); }
#line 2740 "y.tab.c" /* yacc.c:1646  */
    break;

  case 72:
#line 341 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new UsingScope((yyvsp[0].strval), true); }
#line 2746 "y.tab.c" /* yacc.c:1646  */
    break;

  case 73:
#line 343 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].member)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].member); }
#line 2752 "y.tab.c" /* yacc.c:1646  */
    break;

  case 74:
#line 345 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].message)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].message); }
#line 2758 "y.tab.c" /* yacc.c:1646  */
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
#line 2774 "y.tab.c" /* yacc.c:1646  */
    break;

  case 76:
#line 361 "xi-grammar.y" /* yacc.c:1646  */
    { if((yyvsp[-2].conslist)) (yyvsp[-2].conslist)->recurse<int&>((yyvsp[-4].intval), &Construct::setExtern); (yyval.construct) = (yyvsp[-2].conslist); }
#line 2780 "y.tab.c" /* yacc.c:1646  */
    break;

  case 77:
#line 363 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new Scope((yyvsp[-3].strval), (yyvsp[-1].conslist)); }
#line 2786 "y.tab.c" /* yacc.c:1646  */
    break;

  case 78:
#line 365 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = (yyvsp[-1].construct); }
#line 2792 "y.tab.c" /* yacc.c:1646  */
    break;

  case 79:
#line 367 "xi-grammar.y" /* yacc.c:1646  */
    {
          ERROR("preceding construct must be semicolon terminated",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 2802 "y.tab.c" /* yacc.c:1646  */
    break;

  case 80:
#line 373 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].module)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].module); }
#line 2808 "y.tab.c" /* yacc.c:1646  */
    break;

  case 81:
#line 375 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2814 "y.tab.c" /* yacc.c:1646  */
    break;

  case 82:
#line 377 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2820 "y.tab.c" /* yacc.c:1646  */
    break;

  case 83:
#line 379 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2826 "y.tab.c" /* yacc.c:1646  */
    break;

  case 84:
#line 381 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2832 "y.tab.c" /* yacc.c:1646  */
    break;

  case 85:
#line 383 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].templat)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].templat); }
#line 2838 "y.tab.c" /* yacc.c:1646  */
    break;

  case 86:
#line 385 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = NULL; }
#line 2844 "y.tab.c" /* yacc.c:1646  */
    break;

  case 87:
#line 387 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = NULL; }
#line 2850 "y.tab.c" /* yacc.c:1646  */
    break;

  case 88:
#line 389 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = (yyvsp[0].accelBlock); }
#line 2856 "y.tab.c" /* yacc.c:1646  */
    break;

  case 89:
#line 391 "xi-grammar.y" /* yacc.c:1646  */
    {
          ERROR("invalid construct",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 2866 "y.tab.c" /* yacc.c:1646  */
    break;

  case 90:
#line 399 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamType((yyvsp[0].type)); }
#line 2872 "y.tab.c" /* yacc.c:1646  */
    break;

  case 91:
#line 401 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
#line 2878 "y.tab.c" /* yacc.c:1646  */
    break;

  case 92:
#line 403 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
#line 2884 "y.tab.c" /* yacc.c:1646  */
    break;

  case 93:
#line 407 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = new TParamList((yyvsp[0].tparam)); }
#line 2890 "y.tab.c" /* yacc.c:1646  */
    break;

  case 94:
#line 409 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = new TParamList((yyvsp[-2].tparam), (yyvsp[0].tparlist)); }
#line 2896 "y.tab.c" /* yacc.c:1646  */
    break;

  case 95:
#line 413 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = new TParamList(0); }
#line 2902 "y.tab.c" /* yacc.c:1646  */
    break;

  case 96:
#line 415 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = (yyvsp[0].tparlist); }
#line 2908 "y.tab.c" /* yacc.c:1646  */
    break;

  case 97:
#line 419 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = 0; }
#line 2914 "y.tab.c" /* yacc.c:1646  */
    break;

  case 98:
#line 421 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = (yyvsp[-1].tparlist); }
#line 2920 "y.tab.c" /* yacc.c:1646  */
    break;

  case 99:
#line 425 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("int"); }
#line 2926 "y.tab.c" /* yacc.c:1646  */
    break;

  case 100:
#line 427 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long"); }
#line 2932 "y.tab.c" /* yacc.c:1646  */
    break;

  case 101:
#line 429 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("short"); }
#line 2938 "y.tab.c" /* yacc.c:1646  */
    break;

  case 102:
#line 431 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("char"); }
#line 2944 "y.tab.c" /* yacc.c:1646  */
    break;

  case 103:
#line 433 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned int"); }
#line 2950 "y.tab.c" /* yacc.c:1646  */
    break;

  case 104:
#line 435 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long"); }
#line 2956 "y.tab.c" /* yacc.c:1646  */
    break;

  case 105:
#line 437 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long"); }
#line 2962 "y.tab.c" /* yacc.c:1646  */
    break;

  case 106:
#line 439 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long long"); }
#line 2968 "y.tab.c" /* yacc.c:1646  */
    break;

  case 107:
#line 441 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned short"); }
#line 2974 "y.tab.c" /* yacc.c:1646  */
    break;

  case 108:
#line 443 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned char"); }
#line 2980 "y.tab.c" /* yacc.c:1646  */
    break;

  case 109:
#line 445 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long long"); }
#line 2986 "y.tab.c" /* yacc.c:1646  */
    break;

  case 110:
#line 447 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("float"); }
#line 2992 "y.tab.c" /* yacc.c:1646  */
    break;

  case 111:
#line 449 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("double"); }
#line 2998 "y.tab.c" /* yacc.c:1646  */
    break;

  case 112:
#line 451 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long double"); }
#line 3004 "y.tab.c" /* yacc.c:1646  */
    break;

  case 113:
#line 453 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("void"); }
#line 3010 "y.tab.c" /* yacc.c:1646  */
    break;

  case 114:
#line 455 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("size_t"); }
#line 3016 "y.tab.c" /* yacc.c:1646  */
    break;

  case 115:
#line 457 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("bool"); }
#line 3022 "y.tab.c" /* yacc.c:1646  */
    break;

  case 116:
#line 460 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ntype) = new NamedType((yyvsp[-1].strval),(yyvsp[0].tparlist)); }
#line 3028 "y.tab.c" /* yacc.c:1646  */
    break;

  case 117:
#line 461 "xi-grammar.y" /* yacc.c:1646  */
    { 
                    const char* basename, *scope;
                    splitScopedName((yyvsp[-1].strval), &scope, &basename);
                    (yyval.ntype) = new NamedType(basename, (yyvsp[0].tparlist), scope);
                }
#line 3038 "y.tab.c" /* yacc.c:1646  */
    break;

  case 118:
#line 467 "xi-grammar.y" /* yacc.c:1646  */
    {
			const char* basename, *scope;
			splitScopedName((yyvsp[-1].strval), &scope, &basename);
			(yyval.ntype) = new NamedType(basename, (yyvsp[0].tparlist), scope, true);
		}
#line 3048 "y.tab.c" /* yacc.c:1646  */
    break;

  case 119:
#line 475 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3054 "y.tab.c" /* yacc.c:1646  */
    break;

  case 120:
#line 477 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ntype); }
#line 3060 "y.tab.c" /* yacc.c:1646  */
    break;

  case 121:
#line 481 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ptype) = new PtrType((yyvsp[-1].type)); }
#line 3066 "y.tab.c" /* yacc.c:1646  */
    break;

  case 122:
#line 485 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
#line 3072 "y.tab.c" /* yacc.c:1646  */
    break;

  case 123:
#line 487 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
#line 3078 "y.tab.c" /* yacc.c:1646  */
    break;

  case 124:
#line 491 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ftype) = new FuncType((yyvsp[-7].type), (yyvsp[-4].strval), (yyvsp[-1].plist)); }
#line 3084 "y.tab.c" /* yacc.c:1646  */
    break;

  case 125:
#line 495 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3090 "y.tab.c" /* yacc.c:1646  */
    break;

  case 126:
#line 497 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3096 "y.tab.c" /* yacc.c:1646  */
    break;

  case 127:
#line 499 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3102 "y.tab.c" /* yacc.c:1646  */
    break;

  case 128:
#line 501 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ftype); }
#line 3108 "y.tab.c" /* yacc.c:1646  */
    break;

  case 129:
#line 503 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[0].type)); }
#line 3114 "y.tab.c" /* yacc.c:1646  */
    break;

  case 130:
#line 505 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[-1].type)); }
#line 3120 "y.tab.c" /* yacc.c:1646  */
    break;

  case 131:
#line 509 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3126 "y.tab.c" /* yacc.c:1646  */
    break;

  case 132:
#line 511 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3132 "y.tab.c" /* yacc.c:1646  */
    break;

  case 133:
#line 513 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3138 "y.tab.c" /* yacc.c:1646  */
    break;

  case 134:
#line 515 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[0].type)); }
#line 3144 "y.tab.c" /* yacc.c:1646  */
    break;

  case 135:
#line 517 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[-1].type)); }
#line 3150 "y.tab.c" /* yacc.c:1646  */
    break;

  case 136:
#line 521 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType(new RValueReferenceType((yyvsp[-5].type))); }
#line 3156 "y.tab.c" /* yacc.c:1646  */
    break;

  case 137:
#line 523 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType(new ReferenceType((yyvsp[-4].type))); }
#line 3162 "y.tab.c" /* yacc.c:1646  */
    break;

  case 138:
#line 525 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType((yyvsp[-3].type)); }
#line 3168 "y.tab.c" /* yacc.c:1646  */
    break;

  case 139:
#line 527 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new RValueReferenceType((yyvsp[-2].type)); }
#line 3174 "y.tab.c" /* yacc.c:1646  */
    break;

  case 140:
#line 529 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ReferenceType((yyvsp[-1].type)); }
#line 3180 "y.tab.c" /* yacc.c:1646  */
    break;

  case 141:
#line 531 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3186 "y.tab.c" /* yacc.c:1646  */
    break;

  case 142:
#line 535 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType(new RValueReferenceType((yyvsp[-5].type))); }
#line 3192 "y.tab.c" /* yacc.c:1646  */
    break;

  case 143:
#line 537 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType(new ReferenceType((yyvsp[-4].type))); }
#line 3198 "y.tab.c" /* yacc.c:1646  */
    break;

  case 144:
#line 539 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType((yyvsp[-3].type)); }
#line 3204 "y.tab.c" /* yacc.c:1646  */
    break;

  case 145:
#line 541 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new RValueReferenceType((yyvsp[-2].type)); }
#line 3210 "y.tab.c" /* yacc.c:1646  */
    break;

  case 146:
#line 543 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ReferenceType((yyvsp[-1].type)); }
#line 3216 "y.tab.c" /* yacc.c:1646  */
    break;

  case 147:
#line 545 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3222 "y.tab.c" /* yacc.c:1646  */
    break;

  case 148:
#line 549 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 3228 "y.tab.c" /* yacc.c:1646  */
    break;

  case 149:
#line 553 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = (yyvsp[-1].val); }
#line 3234 "y.tab.c" /* yacc.c:1646  */
    break;

  case 150:
#line 557 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.vallist) = 0; }
#line 3240 "y.tab.c" /* yacc.c:1646  */
    break;

  case 151:
#line 559 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.vallist) = new ValueList((yyvsp[-1].val), (yyvsp[0].vallist)); }
#line 3246 "y.tab.c" /* yacc.c:1646  */
    break;

  case 152:
#line 563 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].vallist)); }
#line 3252 "y.tab.c" /* yacc.c:1646  */
    break;

  case 153:
#line 567 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[-3].type), (yyvsp[-1].strval), (yyvsp[0].vallist), 1); }
#line 3258 "y.tab.c" /* yacc.c:1646  */
    break;

  case 154:
#line 571 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0;}
#line 3264 "y.tab.c" /* yacc.c:1646  */
    break;

  case 155:
#line 573 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0;}
#line 3270 "y.tab.c" /* yacc.c:1646  */
    break;

  case 156:
#line 577 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3276 "y.tab.c" /* yacc.c:1646  */
    break;

  case 157:
#line 579 "xi-grammar.y" /* yacc.c:1646  */
    { 
		  /*
		  printf("Warning: Message attributes are being phased out.\n");
		  printf("Warning: Please remove them from interface files.\n");
		  */
		  (yyval.intval) = (yyvsp[-1].intval); 
		}
#line 3288 "y.tab.c" /* yacc.c:1646  */
    break;

  case 158:
#line 589 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[0].intval); }
#line 3294 "y.tab.c" /* yacc.c:1646  */
    break;

  case 159:
#line 591 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[-2].intval) | (yyvsp[0].intval); }
#line 3300 "y.tab.c" /* yacc.c:1646  */
    break;

  case 160:
#line 595 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3306 "y.tab.c" /* yacc.c:1646  */
    break;

  case 161:
#line 597 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3312 "y.tab.c" /* yacc.c:1646  */
    break;

  case 162:
#line 601 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = 0; }
#line 3318 "y.tab.c" /* yacc.c:1646  */
    break;

  case 163:
#line 603 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-1].cattr); }
#line 3324 "y.tab.c" /* yacc.c:1646  */
    break;

  case 164:
#line 607 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[0].cattr); }
#line 3330 "y.tab.c" /* yacc.c:1646  */
    break;

  case 165:
#line 609 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
#line 3336 "y.tab.c" /* yacc.c:1646  */
    break;

  case 166:
#line 613 "xi-grammar.y" /* yacc.c:1646  */
    { python_doc = NULL; (yyval.intval) = 0; }
#line 3342 "y.tab.c" /* yacc.c:1646  */
    break;

  case 167:
#line 615 "xi-grammar.y" /* yacc.c:1646  */
    { python_doc = (yyvsp[0].strval); (yyval.intval) = 0; }
#line 3348 "y.tab.c" /* yacc.c:1646  */
    break;

  case 168:
#line 619 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CPYTHON; }
#line 3354 "y.tab.c" /* yacc.c:1646  */
    break;

  case 169:
#line 623 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = 0; }
#line 3360 "y.tab.c" /* yacc.c:1646  */
    break;

  case 170:
#line 625 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-1].cattr); }
#line 3366 "y.tab.c" /* yacc.c:1646  */
    break;

  case 171:
#line 629 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[0].cattr); }
#line 3372 "y.tab.c" /* yacc.c:1646  */
    break;

  case 172:
#line 631 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
#line 3378 "y.tab.c" /* yacc.c:1646  */
    break;

  case 173:
#line 635 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CMIGRATABLE; }
#line 3384 "y.tab.c" /* yacc.c:1646  */
    break;

  case 174:
#line 637 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CPYTHON; }
#line 3390 "y.tab.c" /* yacc.c:1646  */
    break;

  case 175:
#line 641 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3396 "y.tab.c" /* yacc.c:1646  */
    break;

  case 176:
#line 643 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 3402 "y.tab.c" /* yacc.c:1646  */
    break;

  case 177:
#line 646 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3408 "y.tab.c" /* yacc.c:1646  */
    break;

  case 178:
#line 648 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 3414 "y.tab.c" /* yacc.c:1646  */
    break;

  case 179:
#line 651 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mv) = new MsgVar((yyvsp[-3].type), (yyvsp[-2].strval), (yyvsp[-4].intval), (yyvsp[-1].intval)); }
#line 3420 "y.tab.c" /* yacc.c:1646  */
    break;

  case 180:
#line 655 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mvlist) = new MsgVarList((yyvsp[0].mv)); }
#line 3426 "y.tab.c" /* yacc.c:1646  */
    break;

  case 181:
#line 657 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mvlist) = new MsgVarList((yyvsp[-1].mv), (yyvsp[0].mvlist)); }
#line 3432 "y.tab.c" /* yacc.c:1646  */
    break;

  case 182:
#line 661 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, (yyvsp[0].ntype)); }
#line 3438 "y.tab.c" /* yacc.c:1646  */
    break;

  case 183:
#line 663 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, (yyvsp[-2].ntype)); }
#line 3444 "y.tab.c" /* yacc.c:1646  */
    break;

  case 184:
#line 665 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, (yyvsp[-3].ntype), (yyvsp[-1].mvlist)); }
#line 3450 "y.tab.c" /* yacc.c:1646  */
    break;

  case 185:
#line 669 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = 0; }
#line 3456 "y.tab.c" /* yacc.c:1646  */
    break;

  case 186:
#line 671 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = (yyvsp[0].typelist); }
#line 3462 "y.tab.c" /* yacc.c:1646  */
    break;

  case 187:
#line 675 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = new TypeList((yyvsp[0].ntype)); }
#line 3468 "y.tab.c" /* yacc.c:1646  */
    break;

  case 188:
#line 677 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = new TypeList((yyvsp[-2].ntype), (yyvsp[0].typelist)); }
#line 3474 "y.tab.c" /* yacc.c:1646  */
    break;

  case 189:
#line 681 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr)|Chare::CCHARE, (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3480 "y.tab.c" /* yacc.c:1646  */
    break;

  case 190:
#line 683 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3486 "y.tab.c" /* yacc.c:1646  */
    break;

  case 191:
#line 687 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3492 "y.tab.c" /* yacc.c:1646  */
    break;

  case 192:
#line 691 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new NodeGroup(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3498 "y.tab.c" /* yacc.c:1646  */
    break;

  case 193:
#line 695 "xi-grammar.y" /* yacc.c:1646  */
    {/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",(yyvsp[-2].strval));
			(yyval.ntype) = new NamedType(buf); 
		}
#line 3508 "y.tab.c" /* yacc.c:1646  */
    break;

  case 194:
#line 701 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ntype) = (yyvsp[-1].ntype); }
#line 3514 "y.tab.c" /* yacc.c:1646  */
    break;

  case 195:
#line 705 "xi-grammar.y" /* yacc.c:1646  */
    {  (yyval.chare) = new Array(lineno, (yyvsp[-4].cattr), (yyvsp[-3].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3520 "y.tab.c" /* yacc.c:1646  */
    break;

  case 196:
#line 707 "xi-grammar.y" /* yacc.c:1646  */
    {  (yyval.chare) = new Array(lineno, (yyvsp[-3].cattr), (yyvsp[-4].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3526 "y.tab.c" /* yacc.c:1646  */
    break;

  case 197:
#line 711 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr)|Chare::CCHARE, new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist));}
#line 3532 "y.tab.c" /* yacc.c:1646  */
    break;

  case 198:
#line 713 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3538 "y.tab.c" /* yacc.c:1646  */
    break;

  case 199:
#line 717 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3544 "y.tab.c" /* yacc.c:1646  */
    break;

  case 200:
#line 721 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new NodeGroup( lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3550 "y.tab.c" /* yacc.c:1646  */
    break;

  case 201:
#line 725 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Array( lineno, 0, (yyvsp[-3].ntype), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3556 "y.tab.c" /* yacc.c:1646  */
    break;

  case 202:
#line 729 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-1].strval))); }
#line 3562 "y.tab.c" /* yacc.c:1646  */
    break;

  case 203:
#line 731 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-4].strval)), (yyvsp[-2].mvlist)); }
#line 3568 "y.tab.c" /* yacc.c:1646  */
    break;

  case 204:
#line 735 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = 0; }
#line 3574 "y.tab.c" /* yacc.c:1646  */
    break;

  case 205:
#line 737 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3580 "y.tab.c" /* yacc.c:1646  */
    break;

  case 206:
#line 741 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = 0; }
#line 3586 "y.tab.c" /* yacc.c:1646  */
    break;

  case 207:
#line 743 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3592 "y.tab.c" /* yacc.c:1646  */
    break;

  case 208:
#line 745 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3598 "y.tab.c" /* yacc.c:1646  */
    break;

  case 209:
#line 747 "xi-grammar.y" /* yacc.c:1646  */
    {
		  XStr typeStr;
		  (yyvsp[0].ntype)->print(typeStr);
		  char *tmp = strdup(typeStr.get_string());
		  (yyval.strval) = tmp;
		}
#line 3609 "y.tab.c" /* yacc.c:1646  */
    break;

  case 210:
#line 756 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TTypeEllipsis(new NamedEllipsisType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3615 "y.tab.c" /* yacc.c:1646  */
    break;

  case 211:
#line 758 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TTypeEllipsis(new NamedEllipsisType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3621 "y.tab.c" /* yacc.c:1646  */
    break;

  case 212:
#line 760 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TType(new NamedType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3627 "y.tab.c" /* yacc.c:1646  */
    break;

  case 213:
#line 762 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TType(new NamedType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3633 "y.tab.c" /* yacc.c:1646  */
    break;

  case 214:
#line 764 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TFunc((yyvsp[-1].ftype), (yyvsp[0].strval)); }
#line 3639 "y.tab.c" /* yacc.c:1646  */
    break;

  case 215:
#line 766 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TName((yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].strval)); }
#line 3645 "y.tab.c" /* yacc.c:1646  */
    break;

  case 216:
#line 770 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = new TVarList((yyvsp[0].tvar)); }
#line 3651 "y.tab.c" /* yacc.c:1646  */
    break;

  case 217:
#line 772 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = new TVarList((yyvsp[-2].tvar), (yyvsp[0].tvarlist)); }
#line 3657 "y.tab.c" /* yacc.c:1646  */
    break;

  case 218:
#line 776 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = (yyvsp[-1].tvarlist); }
#line 3663 "y.tab.c" /* yacc.c:1646  */
    break;

  case 219:
#line 780 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3669 "y.tab.c" /* yacc.c:1646  */
    break;

  case 220:
#line 782 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3675 "y.tab.c" /* yacc.c:1646  */
    break;

  case 221:
#line 784 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3681 "y.tab.c" /* yacc.c:1646  */
    break;

  case 222:
#line 786 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3687 "y.tab.c" /* yacc.c:1646  */
    break;

  case 223:
#line 788 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].message)); (yyvsp[0].message)->setTemplate((yyval.templat)); }
#line 3693 "y.tab.c" /* yacc.c:1646  */
    break;

  case 224:
#line 792 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = 0; }
#line 3699 "y.tab.c" /* yacc.c:1646  */
    break;

  case 225:
#line 794 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = (yyvsp[-2].mbrlist); }
#line 3705 "y.tab.c" /* yacc.c:1646  */
    break;

  case 226:
#line 798 "xi-grammar.y" /* yacc.c:1646  */
    { 
                  if (!connectEntries.empty()) {
                    (yyval.mbrlist) = new AstChildren<Member>(connectEntries);
		  } else {
		    (yyval.mbrlist) = 0; 
                  }
		}
#line 3717 "y.tab.c" /* yacc.c:1646  */
    break;

  case 227:
#line 806 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = new AstChildren<Member>(-1, (yyvsp[-1].member), (yyvsp[0].mbrlist)); }
#line 3723 "y.tab.c" /* yacc.c:1646  */
    break;

  case 228:
#line 810 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].readonly); }
#line 3729 "y.tab.c" /* yacc.c:1646  */
    break;

  case 229:
#line 812 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].readonly); }
#line 3735 "y.tab.c" /* yacc.c:1646  */
    break;

  case 231:
#line 815 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].member); }
#line 3741 "y.tab.c" /* yacc.c:1646  */
    break;

  case 232:
#line 817 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].pupable); }
#line 3747 "y.tab.c" /* yacc.c:1646  */
    break;

  case 233:
#line 819 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].includeFile); }
#line 3753 "y.tab.c" /* yacc.c:1646  */
    break;

  case 234:
#line 821 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new ClassDeclaration(lineno,(yyvsp[0].strval)); }
#line 3759 "y.tab.c" /* yacc.c:1646  */
    break;

  case 235:
#line 825 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1); }
#line 3765 "y.tab.c" /* yacc.c:1646  */
    break;

  case 236:
#line 827 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1); }
#line 3771 "y.tab.c" /* yacc.c:1646  */
    break;

  case 237:
#line 829 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[-6].strval)) + '<' +
					    ((yyvsp[-4].tparlist))->to_string() + '>').c_str()),
				    1);
		}
#line 3781 "y.tab.c" /* yacc.c:1646  */
    break;

  case 238:
#line 835 "xi-grammar.y" /* yacc.c:1646  */
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[-2]).first_column, (yylsp[-2]).last_column, (yylsp[-2]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1);
		}
#line 3791 "y.tab.c" /* yacc.c:1646  */
    break;

  case 239:
#line 841 "xi-grammar.y" /* yacc.c:1646  */
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[-5]).first_column, (yylsp[-5]).last_column, (yylsp[-5]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1);
		}
#line 3801 "y.tab.c" /* yacc.c:1646  */
    break;

  case 240:
#line 850 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 0); }
#line 3807 "y.tab.c" /* yacc.c:1646  */
    break;

  case 241:
#line 852 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 0); }
#line 3813 "y.tab.c" /* yacc.c:1646  */
    break;

  case 242:
#line 854 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[-6].strval)) + '<' +
					    ((yyvsp[-4].tparlist))->to_string() + '>').c_str()),
				    0);
		}
#line 3823 "y.tab.c" /* yacc.c:1646  */
    break;

  case 243:
#line 860 "xi-grammar.y" /* yacc.c:1646  */
    {
                  InitCall* rtn = new InitCall(lineno, (yyvsp[-3].strval), 0);
                  rtn->setAccel();
                  (yyval.member) = rtn;
		}
#line 3833 "y.tab.c" /* yacc.c:1646  */
    break;

  case 244:
#line 868 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[0].ntype),0); }
#line 3839 "y.tab.c" /* yacc.c:1646  */
    break;

  case 245:
#line 870 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[-2].ntype),(yyvsp[0].pupable)); }
#line 3845 "y.tab.c" /* yacc.c:1646  */
    break;

  case 246:
#line 873 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.includeFile) = new IncludeFile(lineno,(yyvsp[0].strval)); }
#line 3851 "y.tab.c" /* yacc.c:1646  */
    break;

  case 247:
#line 877 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].member); }
#line 3857 "y.tab.c" /* yacc.c:1646  */
    break;

  case 248:
#line 881 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].entry); }
#line 3863 "y.tab.c" /* yacc.c:1646  */
    break;

  case 249:
#line 883 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyvsp[0].entry)->tspec = (yyvsp[-1].tvarlist);
                  (yyval.member) = (yyvsp[0].entry);
                }
#line 3872 "y.tab.c" /* yacc.c:1646  */
    break;

  case 250:
#line 888 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[-1].member); }
#line 3878 "y.tab.c" /* yacc.c:1646  */
    break;

  case 251:
#line 890 "xi-grammar.y" /* yacc.c:1646  */
    {
          ERROR("invalid SDAG member",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 3888 "y.tab.c" /* yacc.c:1646  */
    break;

  case 252:
#line 898 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3894 "y.tab.c" /* yacc.c:1646  */
    break;

  case 253:
#line 900 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3900 "y.tab.c" /* yacc.c:1646  */
    break;

  case 254:
#line 902 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3906 "y.tab.c" /* yacc.c:1646  */
    break;

  case 255:
#line 904 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3912 "y.tab.c" /* yacc.c:1646  */
    break;

  case 256:
#line 906 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3918 "y.tab.c" /* yacc.c:1646  */
    break;

  case 257:
#line 908 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3924 "y.tab.c" /* yacc.c:1646  */
    break;

  case 258:
#line 910 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3930 "y.tab.c" /* yacc.c:1646  */
    break;

  case 259:
#line 912 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3936 "y.tab.c" /* yacc.c:1646  */
    break;

  case 260:
#line 914 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3942 "y.tab.c" /* yacc.c:1646  */
    break;

  case 261:
#line 916 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3948 "y.tab.c" /* yacc.c:1646  */
    break;

  case 262:
#line 918 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3954 "y.tab.c" /* yacc.c:1646  */
    break;

  case 263:
#line 921 "xi-grammar.y" /* yacc.c:1646  */
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
#line 3969 "y.tab.c" /* yacc.c:1646  */
    break;

  case 264:
#line 932 "xi-grammar.y" /* yacc.c:1646  */
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
#line 3991 "y.tab.c" /* yacc.c:1646  */
    break;

  case 265:
#line 950 "xi-grammar.y" /* yacc.c:1646  */
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
#line 4011 "y.tab.c" /* yacc.c:1646  */
    break;

  case 266:
#line 968 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.accelBlock) = new AccelBlock(lineno, new XStr((yyvsp[-2].strval))); }
#line 4017 "y.tab.c" /* yacc.c:1646  */
    break;

  case 267:
#line 970 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.accelBlock) = new AccelBlock(lineno, NULL); }
#line 4023 "y.tab.c" /* yacc.c:1646  */
    break;

  case 268:
#line 974 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 4029 "y.tab.c" /* yacc.c:1646  */
    break;

  case 269:
#line 978 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attr) = 0; }
#line 4035 "y.tab.c" /* yacc.c:1646  */
    break;

  case 270:
#line 980 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attr) = (yyvsp[-1].attr); }
#line 4041 "y.tab.c" /* yacc.c:1646  */
    break;

  case 271:
#line 982 "xi-grammar.y" /* yacc.c:1646  */
    { ERROR("invalid entry method attribute list",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
#line 4050 "y.tab.c" /* yacc.c:1646  */
    break;

  case 272:
#line 988 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attrarg) = new Attribute::Argument((yyvsp[-2].strval), atoi((yyvsp[0].strval))); }
#line 4056 "y.tab.c" /* yacc.c:1646  */
    break;

  case 273:
#line 992 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attrarg) = (yyvsp[0].attrarg); }
#line 4062 "y.tab.c" /* yacc.c:1646  */
    break;

  case 274:
#line 993 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attrarg) = (yyvsp[-2].attrarg); (yyvsp[-2].attrarg)->next = (yyvsp[0].attrarg); }
#line 4068 "y.tab.c" /* yacc.c:1646  */
    break;

  case 275:
#line 997 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attr) = new Attribute((yyvsp[0].intval));           }
#line 4074 "y.tab.c" /* yacc.c:1646  */
    break;

  case 276:
#line 998 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attr) = new Attribute((yyvsp[-3].intval), (yyvsp[-1].attrarg));       }
#line 4080 "y.tab.c" /* yacc.c:1646  */
    break;

  case 277:
#line 999 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attr) = new Attribute((yyvsp[-2].intval), NULL, (yyvsp[0].attr)); }
#line 4086 "y.tab.c" /* yacc.c:1646  */
    break;

  case 278:
#line 1000 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attr) = new Attribute((yyvsp[-5].intval), (yyvsp[-3].attrarg), (yyvsp[0].attr));   }
#line 4092 "y.tab.c" /* yacc.c:1646  */
    break;

  case 279:
#line 1004 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = STHREADED; }
#line 4098 "y.tab.c" /* yacc.c:1646  */
    break;

  case 280:
#line 1006 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SWHENIDLE; }
#line 4104 "y.tab.c" /* yacc.c:1646  */
    break;

  case 281:
#line 1008 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SSYNC; }
#line 4110 "y.tab.c" /* yacc.c:1646  */
    break;

  case 282:
#line 1010 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SIGET; }
#line 4116 "y.tab.c" /* yacc.c:1646  */
    break;

  case 283:
#line 1012 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SLOCKED; }
#line 4122 "y.tab.c" /* yacc.c:1646  */
    break;

  case 284:
#line 1014 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SCREATEHERE; }
#line 4128 "y.tab.c" /* yacc.c:1646  */
    break;

  case 285:
#line 1016 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SCREATEHOME; }
#line 4134 "y.tab.c" /* yacc.c:1646  */
    break;

  case 286:
#line 1018 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SNOKEEP; }
#line 4140 "y.tab.c" /* yacc.c:1646  */
    break;

  case 287:
#line 1020 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SNOTRACE; }
#line 4146 "y.tab.c" /* yacc.c:1646  */
    break;

  case 288:
#line 1022 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SAPPWORK; }
#line 4152 "y.tab.c" /* yacc.c:1646  */
    break;

  case 289:
#line 1024 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SIMMEDIATE; }
#line 4158 "y.tab.c" /* yacc.c:1646  */
    break;

  case 290:
#line 1026 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SSKIPSCHED; }
#line 4164 "y.tab.c" /* yacc.c:1646  */
    break;

  case 291:
#line 1028 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SINLINE; }
#line 4170 "y.tab.c" /* yacc.c:1646  */
    break;

  case 292:
#line 1030 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SLOCAL; }
#line 4176 "y.tab.c" /* yacc.c:1646  */
    break;

  case 293:
#line 1032 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SPYTHON; }
#line 4182 "y.tab.c" /* yacc.c:1646  */
    break;

  case 294:
#line 1034 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SMEM; }
#line 4188 "y.tab.c" /* yacc.c:1646  */
    break;

  case 295:
#line 1036 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SREDUCE; }
#line 4194 "y.tab.c" /* yacc.c:1646  */
    break;

  case 296:
#line 1038 "xi-grammar.y" /* yacc.c:1646  */
    {
        (yyval.intval) = SAGGREGATE;
    }
#line 4202 "y.tab.c" /* yacc.c:1646  */
    break;

  case 297:
#line 1042 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("invalid entry method attribute",
		        (yylsp[0]).first_column, (yylsp[0]).last_column);
		  yyclearin;
		  yyerrok;
		}
#line 4213 "y.tab.c" /* yacc.c:1646  */
    break;

  case 298:
#line 1051 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4219 "y.tab.c" /* yacc.c:1646  */
    break;

  case 299:
#line 1053 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4225 "y.tab.c" /* yacc.c:1646  */
    break;

  case 300:
#line 1055 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4231 "y.tab.c" /* yacc.c:1646  */
    break;

  case 301:
#line 1059 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = ""; }
#line 4237 "y.tab.c" /* yacc.c:1646  */
    break;

  case 302:
#line 1061 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4243 "y.tab.c" /* yacc.c:1646  */
    break;

  case 303:
#line 1063 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s, %s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4253 "y.tab.c" /* yacc.c:1646  */
    break;

  case 304:
#line 1071 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = ""; }
#line 4259 "y.tab.c" /* yacc.c:1646  */
    break;

  case 305:
#line 1073 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4265 "y.tab.c" /* yacc.c:1646  */
    break;

  case 306:
#line 1075 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s[%s]%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4275 "y.tab.c" /* yacc.c:1646  */
    break;

  case 307:
#line 1081 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s{%s}%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4285 "y.tab.c" /* yacc.c:1646  */
    break;

  case 308:
#line 1087 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s(%s)%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4295 "y.tab.c" /* yacc.c:1646  */
    break;

  case 309:
#line 1093 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"(%s)%s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4305 "y.tab.c" /* yacc.c:1646  */
    break;

  case 310:
#line 1101 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval));
		}
#line 4314 "y.tab.c" /* yacc.c:1646  */
    break;

  case 311:
#line 1108 "xi-grammar.y" /* yacc.c:1646  */
    { 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			(yyval.intval) = 0;
		}
#line 4324 "y.tab.c" /* yacc.c:1646  */
    break;

  case 312:
#line 1116 "xi-grammar.y" /* yacc.c:1646  */
    { 
			in_braces=0;
			(yyval.intval) = 0;
		}
#line 4333 "y.tab.c" /* yacc.c:1646  */
    break;

  case 313:
#line 1123 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[0].type));}
#line 4339 "y.tab.c" /* yacc.c:1646  */
    break;

  case 314:
#line 1125 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval)); (yyval.pname)->setConditional((yyvsp[0].intval)); }
#line 4345 "y.tab.c" /* yacc.c:1646  */
    break;

  case 315:
#line 1127 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[-3].type),(yyvsp[-2].strval),0,(yyvsp[0].val));}
#line 4351 "y.tab.c" /* yacc.c:1646  */
    break;

  case 316:
#line 1129 "xi-grammar.y" /* yacc.c:1646  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
		}
#line 4360 "y.tab.c" /* yacc.c:1646  */
    break;

  case 317:
#line 1134 "xi-grammar.y" /* yacc.c:1646  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
			(yyval.pname)->setRdma(CMK_ZC_P2P_SEND_MSG);
			if(firstRdma) {
				(yyval.pname)->setFirstRdma(true);
				firstRdma = false;
			}
		}
#line 4374 "y.tab.c" /* yacc.c:1646  */
    break;

  case 318:
#line 1144 "xi-grammar.y" /* yacc.c:1646  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
			(yyval.pname)->setRdma(CMK_ZC_P2P_RECV_MSG);
			if(firstRdma) {
				(yyval.pname)->setFirstRdma(true);
				firstRdma = false;
			}
		}
#line 4388 "y.tab.c" /* yacc.c:1646  */
    break;

  case 319:
#line 1154 "xi-grammar.y" /* yacc.c:1646  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
			(yyval.pname)->setRdma(CMK_ZC_DEVICE_MSG);
			if (firstDeviceRdma) {
				(yyval.pname)->setFirstDeviceRdma(true);
				firstDeviceRdma = false;
			}
		}
#line 4402 "y.tab.c" /* yacc.c:1646  */
    break;

  case 320:
#line 1165 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READONLY; }
#line 4408 "y.tab.c" /* yacc.c:1646  */
    break;

  case 321:
#line 1166 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READWRITE; }
#line 4414 "y.tab.c" /* yacc.c:1646  */
    break;

  case 322:
#line 1167 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_WRITEONLY; }
#line 4420 "y.tab.c" /* yacc.c:1646  */
    break;

  case 323:
#line 1170 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr((yyvsp[0].strval)); }
#line 4426 "y.tab.c" /* yacc.c:1646  */
    break;

  case 324:
#line 1171 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "->" << (yyvsp[0].strval); }
#line 4432 "y.tab.c" /* yacc.c:1646  */
    break;

  case 325:
#line 1172 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[-2].xstrptr)) << "." << (yyvsp[0].strval); }
#line 4438 "y.tab.c" /* yacc.c:1646  */
    break;

  case 326:
#line 1174 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << *((yyvsp[-1].xstrptr)) << "]";
                  delete (yyvsp[-3].xstrptr);
                  delete (yyvsp[-1].xstrptr);
                }
#line 4449 "y.tab.c" /* yacc.c:1646  */
    break;

  case 327:
#line 1181 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << (yyvsp[-1].strval) << "]";
                  delete (yyvsp[-3].xstrptr);
                }
#line 4459 "y.tab.c" /* yacc.c:1646  */
    break;

  case 328:
#line 1187 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "(" << *((yyvsp[-1].xstrptr)) << ")";
                  delete (yyvsp[-3].xstrptr);
                  delete (yyvsp[-1].xstrptr);
                }
#line 4470 "y.tab.c" /* yacc.c:1646  */
    break;

  case 329:
#line 1196 "xi-grammar.y" /* yacc.c:1646  */
    {
                  in_bracket = 0;
                  (yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName(), (yyvsp[-1].strval));
                }
#line 4479 "y.tab.c" /* yacc.c:1646  */
    break;

  case 330:
#line 1203 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[-6].intval));
                }
#line 4489 "y.tab.c" /* yacc.c:1646  */
    break;

  case 331:
#line 1209 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType(Parameter::ACCEL_BUFFER_TYPE_READWRITE);
		}
#line 4499 "y.tab.c" /* yacc.c:1646  */
    break;

  case 332:
#line 1215 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.pname) = (yyvsp[-3].pname);
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[-5].intval));
		}
#line 4509 "y.tab.c" /* yacc.c:1646  */
    break;

  case 333:
#line 1223 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 4515 "y.tab.c" /* yacc.c:1646  */
    break;

  case 334:
#line 1225 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 4521 "y.tab.c" /* yacc.c:1646  */
    break;

  case 335:
#line 1229 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 4527 "y.tab.c" /* yacc.c:1646  */
    break;

  case 336:
#line 1231 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 4533 "y.tab.c" /* yacc.c:1646  */
    break;

  case 337:
#line 1235 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = (yyvsp[-1].plist); }
#line 4539 "y.tab.c" /* yacc.c:1646  */
    break;

  case 338:
#line 1237 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList(new Parameter(0, new BuiltinType("void"))); }
#line 4545 "y.tab.c" /* yacc.c:1646  */
    break;

  case 339:
#line 1241 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = (yyvsp[-1].plist); }
#line 4551 "y.tab.c" /* yacc.c:1646  */
    break;

  case 340:
#line 1243 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = 0; }
#line 4557 "y.tab.c" /* yacc.c:1646  */
    break;

  case 341:
#line 1247 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = 0; }
#line 4563 "y.tab.c" /* yacc.c:1646  */
    break;

  case 342:
#line 1249 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4569 "y.tab.c" /* yacc.c:1646  */
    break;

  case 343:
#line 1253 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = 0; }
#line 4575 "y.tab.c" /* yacc.c:1646  */
    break;

  case 344:
#line 1255 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[0].sc)); }
#line 4581 "y.tab.c" /* yacc.c:1646  */
    break;

  case 345:
#line 1257 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[-2].slist)); }
#line 4587 "y.tab.c" /* yacc.c:1646  */
    break;

  case 346:
#line 1261 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.slist) = new SListConstruct((yyvsp[0].sc)); }
#line 4593 "y.tab.c" /* yacc.c:1646  */
    break;

  case 347:
#line 1263 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.slist) = new SListConstruct((yyvsp[-1].sc), (yyvsp[0].slist));  }
#line 4599 "y.tab.c" /* yacc.c:1646  */
    break;

  case 348:
#line 1267 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.olist) = new OListConstruct((yyvsp[0].sc)); }
#line 4605 "y.tab.c" /* yacc.c:1646  */
    break;

  case 349:
#line 1269 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.olist) = new OListConstruct((yyvsp[-1].sc), (yyvsp[0].slist)); }
#line 4611 "y.tab.c" /* yacc.c:1646  */
    break;

  case 350:
#line 1273 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.clist) = new CaseListConstruct((yyvsp[0].when)); }
#line 4617 "y.tab.c" /* yacc.c:1646  */
    break;

  case 351:
#line 1275 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.clist) = new CaseListConstruct((yyvsp[-1].when), (yyvsp[0].clist)); }
#line 4623 "y.tab.c" /* yacc.c:1646  */
    break;

  case 352:
#line 1277 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("case blocks can only contain when clauses",
		        (yylsp[0]).first_column, (yylsp[0]).last_column);
		  (yyval.clist) = 0;
		}
#line 4633 "y.tab.c" /* yacc.c:1646  */
    break;

  case 353:
#line 1285 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4639 "y.tab.c" /* yacc.c:1646  */
    break;

  case 354:
#line 1287 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = 0; }
#line 4645 "y.tab.c" /* yacc.c:1646  */
    break;

  case 355:
#line 1291 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-2].entrylist), 0); }
#line 4651 "y.tab.c" /* yacc.c:1646  */
    break;

  case 356:
#line 1293 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-1].entrylist), (yyvsp[0].sc)); }
#line 4657 "y.tab.c" /* yacc.c:1646  */
    break;

  case 357:
#line 1295 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-3].entrylist), (yyvsp[-1].slist)); }
#line 4663 "y.tab.c" /* yacc.c:1646  */
    break;

  case 358:
#line 1299 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4669 "y.tab.c" /* yacc.c:1646  */
    break;

  case 359:
#line 1301 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4675 "y.tab.c" /* yacc.c:1646  */
    break;

  case 360:
#line 1303 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4681 "y.tab.c" /* yacc.c:1646  */
    break;

  case 361:
#line 1305 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4687 "y.tab.c" /* yacc.c:1646  */
    break;

  case 362:
#line 1307 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4693 "y.tab.c" /* yacc.c:1646  */
    break;

  case 363:
#line 1309 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4699 "y.tab.c" /* yacc.c:1646  */
    break;

  case 364:
#line 1311 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4705 "y.tab.c" /* yacc.c:1646  */
    break;

  case 365:
#line 1313 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4711 "y.tab.c" /* yacc.c:1646  */
    break;

  case 366:
#line 1315 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4717 "y.tab.c" /* yacc.c:1646  */
    break;

  case 367:
#line 1317 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4723 "y.tab.c" /* yacc.c:1646  */
    break;

  case 368:
#line 1319 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4729 "y.tab.c" /* yacc.c:1646  */
    break;

  case 369:
#line 1321 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4735 "y.tab.c" /* yacc.c:1646  */
    break;

  case 370:
#line 1325 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new SerialConstruct((yyvsp[-2].strval), (yyvsp[-4].strval), (yylsp[-3]).first_line); }
#line 4741 "y.tab.c" /* yacc.c:1646  */
    break;

  case 371:
#line 1327 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new OverlapConstruct((yyvsp[-1].olist)); }
#line 4747 "y.tab.c" /* yacc.c:1646  */
    break;

  case 372:
#line 1329 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = (yyvsp[0].when); }
#line 4753 "y.tab.c" /* yacc.c:1646  */
    break;

  case 373:
#line 1331 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new CaseConstruct((yyvsp[-1].clist)); }
#line 4759 "y.tab.c" /* yacc.c:1646  */
    break;

  case 374:
#line 1333 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForConstruct((yyvsp[-8].intexpr), (yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4765 "y.tab.c" /* yacc.c:1646  */
    break;

  case 375:
#line 1335 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForConstruct((yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4771 "y.tab.c" /* yacc.c:1646  */
    break;

  case 376:
#line 1337 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-9].strval)), (yyvsp[-6].intexpr),
		             (yyvsp[-4].intexpr), (yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4778 "y.tab.c" /* yacc.c:1646  */
    break;

  case 377:
#line 1340 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-11].strval)), (yyvsp[-8].intexpr),
		             (yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4785 "y.tab.c" /* yacc.c:1646  */
    break;

  case 378:
#line 1343 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new IfConstruct((yyvsp[-3].intexpr), (yyvsp[-1].sc), (yyvsp[0].sc)); }
#line 4791 "y.tab.c" /* yacc.c:1646  */
    break;

  case 379:
#line 1345 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new IfConstruct((yyvsp[-5].intexpr), (yyvsp[-2].slist), (yyvsp[0].sc)); }
#line 4797 "y.tab.c" /* yacc.c:1646  */
    break;

  case 380:
#line 1347 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new WhileConstruct((yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4803 "y.tab.c" /* yacc.c:1646  */
    break;

  case 381:
#line 1349 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new WhileConstruct((yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4809 "y.tab.c" /* yacc.c:1646  */
    break;

  case 382:
#line 1351 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new SerialConstruct((yyvsp[-2].strval), NULL, (yyloc).first_line); }
#line 4815 "y.tab.c" /* yacc.c:1646  */
    break;

  case 383:
#line 1353 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("unknown SDAG construct or malformed entry method declaration.\n"
		        "You may have forgotten to terminate a previous entry method declaration with a"
		        " semicolon or forgotten to mark a block of sequential SDAG code as 'serial'",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
#line 4827 "y.tab.c" /* yacc.c:1646  */
    break;

  case 384:
#line 1363 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = 0; }
#line 4833 "y.tab.c" /* yacc.c:1646  */
    break;

  case 385:
#line 1365 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ElseConstruct((yyvsp[0].sc)); }
#line 4839 "y.tab.c" /* yacc.c:1646  */
    break;

  case 386:
#line 1367 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ElseConstruct((yyvsp[-1].slist)); }
#line 4845 "y.tab.c" /* yacc.c:1646  */
    break;

  case 387:
#line 1371 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intexpr) = new IntExprConstruct((yyvsp[0].strval)); }
#line 4851 "y.tab.c" /* yacc.c:1646  */
    break;

  case 388:
#line 1375 "xi-grammar.y" /* yacc.c:1646  */
    { in_int_expr = 0; (yyval.intval) = 0; }
#line 4857 "y.tab.c" /* yacc.c:1646  */
    break;

  case 389:
#line 1379 "xi-grammar.y" /* yacc.c:1646  */
    { in_int_expr = 1; (yyval.intval) = 0; }
#line 4863 "y.tab.c" /* yacc.c:1646  */
    break;

  case 390:
#line 1383 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.entry) = new Entry(lineno, NULL, 0, (yyvsp[-1].strval), (yyvsp[0].plist), 0, 0, 0, (yyloc).first_line, (yyloc).last_line);
		  firstRdma = true;
		  firstDeviceRdma = true;
		}
#line 4873 "y.tab.c" /* yacc.c:1646  */
    break;

  case 391:
#line 1389 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.entry) = new Entry(lineno, NULL, 0, (yyvsp[-4].strval), (yyvsp[0].plist), 0, 0, (yyvsp[-2].strval), (yyloc).first_line, (yyloc).last_line);
		  firstRdma = true;
		  firstDeviceRdma = true;
		}
#line 4883 "y.tab.c" /* yacc.c:1646  */
    break;

  case 392:
#line 1397 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.entrylist) = new EntryList((yyvsp[0].entry)); }
#line 4889 "y.tab.c" /* yacc.c:1646  */
    break;

  case 393:
#line 1399 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.entrylist) = new EntryList((yyvsp[-2].entry),(yyvsp[0].entrylist)); }
#line 4895 "y.tab.c" /* yacc.c:1646  */
    break;

  case 394:
#line 1403 "xi-grammar.y" /* yacc.c:1646  */
    { in_bracket=1; }
#line 4901 "y.tab.c" /* yacc.c:1646  */
    break;

  case 395:
#line 1406 "xi-grammar.y" /* yacc.c:1646  */
    { in_bracket=0; }
#line 4907 "y.tab.c" /* yacc.c:1646  */
    break;

  case 396:
#line 1410 "xi-grammar.y" /* yacc.c:1646  */
    { if (!macroDefined((yyvsp[0].strval), 1)) in_comment = 1; }
#line 4913 "y.tab.c" /* yacc.c:1646  */
    break;

  case 397:
#line 1414 "xi-grammar.y" /* yacc.c:1646  */
    { if (!macroDefined((yyvsp[0].strval), 0)) in_comment = 1; }
#line 4919 "y.tab.c" /* yacc.c:1646  */
    break;


#line 4923 "y.tab.c" /* yacc.c:1646  */
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
#line 1417 "xi-grammar.y" /* yacc.c:1906  */


void yyerror(const char *s) 
{
	fprintf(stderr, "[PARSE-ERROR] Unexpected/missing token at line %d. Current token being parsed: '%s'.\n", lineno, yytext);
}
