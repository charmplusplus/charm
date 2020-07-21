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
    SIZET = 325,
    BOOL = 326,
    ACCEL = 327,
    READWRITE = 328,
    WRITEONLY = 329,
    ACCELBLOCK = 330,
    MEMCRITICAL = 331,
    REDUCTIONTARGET = 332,
    CASE = 333,
    TYPENAME = 334
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
#define SIZET 325
#define BOOL 326
#define ACCEL 327
#define READWRITE 328
#define WRITEONLY 329
#define ACCELBLOCK 330
#define MEMCRITICAL 331
#define REDUCTIONTARGET 332
#define CASE 333
#define TYPENAME 334

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

#line 359 "y.tab.c" /* yacc.c:355  */
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

#line 390 "y.tab.c" /* yacc.c:358  */

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
#define YYLAST   1576

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  96
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  119
/* YYNRULES -- Number of rules.  */
#define YYNRULES  395
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  785

/* YYTRANSLATE[YYX] -- Symbol number corresponding to YYX as returned
   by yylex, with out-of-bounds checking.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   334

#define YYTRANSLATE(YYX)                                                \
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, without out-of-bounds checking.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    90,     2,
      88,    89,    87,     2,    84,    95,    91,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    81,    80,
      85,    94,    86,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    92,     2,    93,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    82,     2,    83,     2,     2,     2,     2,
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
      75,    76,    77,    78,    79
};

#if YYDEBUG
  /* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   201,   201,   206,   209,   214,   215,   219,   221,   226,
     227,   232,   234,   235,   236,   238,   239,   240,   242,   243,
     244,   245,   246,   250,   251,   252,   253,   254,   255,   256,
     257,   258,   259,   260,   261,   262,   263,   264,   265,   266,
     267,   268,   269,   272,   273,   274,   275,   276,   277,   278,
     279,   280,   281,   282,   284,   286,   287,   290,   291,   292,
     293,   297,   299,   305,   312,   316,   323,   325,   330,   331,
     335,   337,   339,   341,   343,   356,   358,   360,   362,   368,
     370,   372,   374,   376,   378,   380,   382,   384,   386,   394,
     396,   398,   402,   404,   409,   410,   415,   416,   420,   422,
     424,   426,   428,   430,   432,   434,   436,   438,   440,   442,
     444,   446,   448,   450,   452,   456,   457,   462,   470,   472,
     476,   480,   482,   486,   490,   492,   494,   496,   498,   500,
     504,   506,   508,   510,   512,   516,   518,   520,   522,   524,
     526,   530,   532,   534,   536,   538,   540,   544,   548,   553,
     554,   558,   562,   567,   568,   573,   574,   584,   586,   590,
     592,   597,   598,   602,   604,   609,   610,   614,   619,   620,
     624,   626,   630,   632,   637,   638,   642,   643,   646,   650,
     652,   656,   658,   660,   665,   666,   670,   672,   676,   678,
     682,   686,   690,   696,   700,   702,   706,   708,   712,   716,
     720,   724,   726,   731,   732,   737,   738,   740,   742,   751,
     753,   755,   757,   759,   761,   765,   767,   771,   775,   777,
     779,   781,   783,   787,   789,   794,   801,   805,   807,   809,
     810,   812,   814,   816,   820,   822,   824,   830,   836,   845,
     847,   849,   855,   863,   865,   868,   872,   876,   878,   883,
     885,   893,   895,   897,   899,   901,   903,   905,   907,   909,
     911,   913,   916,   926,   943,   960,   962,   966,   971,   972,
     974,   981,   985,   986,   990,   991,   992,   993,   996,   998,
    1000,  1002,  1004,  1006,  1008,  1010,  1012,  1014,  1016,  1018,
    1020,  1022,  1024,  1026,  1028,  1030,  1034,  1043,  1045,  1047,
    1052,  1053,  1055,  1064,  1065,  1067,  1073,  1079,  1085,  1093,
    1100,  1108,  1115,  1117,  1119,  1121,  1126,  1136,  1148,  1149,
    1150,  1153,  1154,  1155,  1156,  1163,  1169,  1178,  1185,  1191,
    1197,  1205,  1207,  1211,  1213,  1217,  1219,  1223,  1225,  1230,
    1231,  1235,  1237,  1239,  1243,  1245,  1249,  1251,  1255,  1257,
    1259,  1267,  1270,  1273,  1275,  1277,  1281,  1283,  1285,  1287,
    1289,  1291,  1293,  1295,  1297,  1299,  1301,  1303,  1307,  1309,
    1311,  1313,  1315,  1317,  1319,  1322,  1325,  1327,  1329,  1331,
    1333,  1335,  1346,  1347,  1349,  1353,  1357,  1361,  1365,  1370,
    1377,  1379,  1383,  1386,  1390,  1394
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
  "INT", "LONG", "SHORT", "CHAR", "FLOAT", "DOUBLE", "UNSIGNED", "SIZET",
  "BOOL", "ACCEL", "READWRITE", "WRITEONLY", "ACCELBLOCK", "MEMCRITICAL",
  "REDUCTIONTARGET", "CASE", "TYPENAME", "';'", "':'", "'{'", "'}'", "','",
  "'<'", "'>'", "'*'", "'('", "')'", "'&'", "'.'", "'['", "']'", "'='",
  "'-'", "$accept", "File", "ModuleEList", "OptExtern",
  "OneOrMoreSemiColon", "OptSemiColon", "Name", "QualName", "Module",
  "ConstructEList", "ConstructList", "ConstructSemi", "Construct",
  "TParam", "TParamList", "TParamEList", "OptTParams", "BuiltinType",
  "NamedType", "QualNamedType", "SimpleType", "OnePtrType", "PtrType",
  "FuncType", "BaseType", "BaseDataType", "RestrictedType", "Type",
  "ArrayDim", "Dim", "DimList", "Readonly", "ReadonlyMsg", "OptVoid",
  "MAttribs", "MAttribList", "MAttrib", "CAttribs", "CAttribList",
  "PythonOptions", "ArrayAttrib", "ArrayAttribs", "ArrayAttribList",
  "CAttrib", "OptConditional", "MsgArray", "Var", "VarList", "Message",
  "OptBaseList", "BaseList", "Chare", "Group", "NodeGroup",
  "ArrayIndexType", "Array", "TChare", "TGroup", "TNodeGroup", "TArray",
  "TMessage", "OptTypeInit", "OptNameInit", "TVar", "TVarList",
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
      59,    58,   123,   125,    44,    60,    62,    42,    40,    41,
      38,    46,    91,    93,    61,    45
};
# endif

#define YYPACT_NINF -662

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-662)))

#define YYTABLE_NINF -347

#define yytable_value_is_error(Yytable_value) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
     263,  1340,  1340,    43,  -662,   263,  -662,  -662,  -662,  -662,
    -662,  -662,  -662,  -662,  -662,  -662,  -662,  -662,  -662,  -662,
    -662,  -662,  -662,  -662,  -662,  -662,  -662,  -662,  -662,  -662,
    -662,  -662,  -662,  -662,  -662,  -662,  -662,  -662,  -662,  -662,
    -662,  -662,  -662,  -662,  -662,  -662,  -662,  -662,  -662,  -662,
    -662,  -662,  -662,  -662,  -662,  -662,   148,   148,  -662,  -662,
    -662,   907,   -28,  -662,  -662,  -662,    11,  1340,   223,  1340,
    1340,   231,  1048,   -25,  1065,   907,  -662,  -662,  -662,  -662,
     286,    -3,    33,  -662,    34,  -662,  -662,  -662,   -28,    -7,
     648,   136,   136,   -10,    70,    87,    87,    87,    87,    90,
      93,  1340,   152,   158,   907,  -662,  -662,  -662,  -662,  -662,
    -662,  -662,  -662,   534,  -662,  -662,  -662,  -662,   156,  -662,
    -662,  -662,  -662,  -662,  -662,  -662,  -662,  -662,  -662,  -662,
     -28,  -662,  -662,  -662,  1205,  1463,   907,    34,   150,    47,
      -7,   167,  1497,  -662,  1480,  -662,    91,  -662,  -662,  -662,
    -662,   282,  -662,  -662,    33,   103,  -662,  -662,   220,   234,
     240,  -662,    81,    33,  -662,    33,    33,   229,    33,   254,
    -662,   113,  1340,  1340,  1340,  1340,   146,   267,   269,   321,
    1340,  -662,  -662,  -662,  1389,   292,    87,    87,    87,    87,
     267,    93,  -662,  -662,  -662,  -662,  -662,   -28,  -662,  -662,
    -662,  -662,  -662,  -662,  -662,  -662,  -662,  -662,  -662,  -662,
    -662,  -662,  -662,  -662,   322,  -662,  -662,  -662,   300,   114,
    1463,   220,   234,   240,    86,  -662,    70,   302,    20,    -7,
     336,    -7,   314,  -662,   156,   320,    -8,  -662,  -662,  -662,
     305,  -662,  -662,   103,  1056,  -662,  -662,  -662,  -662,  -662,
     324,   308,   318,   -19,    53,   289,   325,   295,    70,  -662,
    -662,   327,   329,   332,   340,   340,   340,   340,  -662,  1340,
     333,   338,   334,   175,  1340,   372,  1340,  -662,  -662,   346,
     347,   351,   818,   -20,   151,  1340,   356,   357,   156,  1340,
    1340,  1340,  1340,  1340,  1340,  -662,  -662,  -662,  1205,  1340,
     405,  -662,   315,   359,  1340,  -662,  -662,  -662,   364,   367,
     368,   363,    -7,   -28,    33,  -662,  -662,  -662,  -662,  -662,
     374,  -662,   382,  -662,  1340,   393,   394,   395,  -662,   381,
    -662,    -7,   136,  1056,   136,   136,  1056,   136,  -662,  -662,
     113,  -662,    70,   246,   246,   246,   246,   397,  -662,   372,
    -662,   340,   340,  -662,   321,    19,   415,   402,   165,   416,
      99,  -662,   414,  1389,  -662,  -662,   340,   340,   340,   340,
     340,   278,  -662,   406,   426,   425,   429,   430,   436,   332,
      -7,   336,    -7,    -7,  -662,   -19,  1056,  -662,   439,   440,
     443,  -662,  -662,   437,  -662,   446,   453,   452,    33,   456,
     462,  -662,   459,  -662,   200,   -28,  -662,  -662,  -662,  -662,
    -662,  -662,   246,   246,  -662,  -662,  -662,  1480,    22,   469,
     464,  1480,  -662,  -662,   466,  -662,  -662,  -662,  -662,  -662,
     246,   246,   246,   246,   246,   542,   -28,   502,  1340,   479,
     473,   474,  -662,   481,  -662,  -662,  -662,  -662,  -662,  -662,
     490,   476,  -662,  -662,  -662,  -662,   492,  -662,    62,   504,
    -662,    70,  -662,   736,   546,   510,   156,   200,  -662,  -662,
    -662,  -662,  1340,  -662,  -662,  1340,  -662,   538,  -662,  -662,
    -662,  -662,  -662,   511,  -662,  -662,  1205,   505,  -662,  1409,
    -662,  1443,  -662,   136,   136,   136,  -662,  1064,  1146,  -662,
     156,   -28,  -662,   508,   402,   402,   156,  -662,  -662,  1480,
    1480,  -662,  1340,    -7,   513,   512,   515,   517,   520,   521,
     518,   524,   481,  1340,  -662,   523,   156,  -662,  -662,   -28,
    1340,    -7,    -7,    24,   527,  1443,  -662,  -662,  -662,  -662,
    -662,   576,   525,   481,  -662,   -28,   529,   531,   532,  -662,
     237,  -662,  -662,  -662,  1340,  -662,   540,   530,   540,   560,
     544,   571,   540,   549,   226,   -28,    -7,  -662,  -662,  -662,
     612,  -662,  -662,  -662,  -662,    34,  -662,   481,  -662,    -7,
     583,    -7,   214,   557,   537,   588,  -662,   561,    -7,   431,
     559,   341,   167,   550,   525,   562,  -662,   573,   563,   568,
    -662,    -7,   560,   284,  -662,   575,   454,    -7,   568,   540,
     567,   540,   578,   571,   540,   580,    -7,   584,   431,  -662,
     156,  -662,   156,   605,  -662,   396,   561,    -7,   540,  -662,
     601,   437,  -662,  -662,   585,  -662,  -662,   167,   626,    -7,
     621,    -7,   588,   561,    -7,   431,   167,  -662,  -662,  -662,
    -662,  -662,  -662,  -662,  -662,  -662,  1340,   599,   597,   589,
      -7,   604,    -7,   226,  -662,   481,  -662,   156,   226,   633,
     608,   598,   568,   607,    -7,   568,   609,   156,   610,  1480,
    1366,  -662,   167,    -7,   613,   614,  -662,  -662,   617,   855,
    -662,    -7,   540,   896,  -662,   167,   906,  -662,  -662,  1340,
    1340,    -7,   611,  -662,  1340,   568,    -7,  -662,   633,   226,
    -662,   622,    -7,   226,  -662,   156,   226,   633,  -662,   224,
     131,   600,  1340,   156,   947,   623,  -662,   618,    -7,   625,
     627,  -662,   637,  -662,  -662,  1340,  1340,  1264,   636,  1340,
    -662,   309,   -28,   226,  -662,    -7,  -662,   568,    -7,  -662,
     633,   403,  -662,   616,   424,  1340,   350,  -662,   640,   568,
     957,   641,  -662,  -662,  -662,  -662,  -662,  -662,  -662,   964,
     226,  -662,    -7,   226,  -662,   643,   568,   645,  -662,   971,
    -662,   226,  -662,   646,  -662
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
       0,     0,     0,    61,    71,   394,   395,   310,   266,   303,
       0,   153,   153,   153,     0,   161,   161,   161,   161,     0,
     155,     0,     0,     0,     0,    79,   227,   228,    73,    80,
      81,    82,    83,     0,    84,    72,   230,   229,     9,   261,
     253,   254,   255,   256,   257,   259,   260,   258,   251,   252,
      77,    78,    69,   270,     0,     0,     0,    70,     0,   304,
     303,     0,     0,   112,     0,    98,    99,   100,   101,   109,
     110,     0,   113,   114,     0,    96,   118,   119,   124,   125,
     126,   127,   146,     0,   154,     0,     0,     0,     0,   243,
     231,     0,     0,     0,     0,     0,     0,     0,   168,     0,
       0,   233,   245,   232,     0,     0,   161,   161,   161,   161,
       0,   155,   218,   219,   220,   221,   222,    10,    67,   296,
     278,   279,   280,   281,   282,   288,   289,   290,   295,   283,
     284,   285,   286,   287,   165,   291,   293,   294,     0,   274,
       0,   130,   131,   132,   140,   267,     0,     0,     0,   303,
     300,   303,     0,   311,     0,     0,   128,   108,   111,   102,
     103,   106,   107,    96,    94,   116,   120,   121,   122,   129,
       0,   145,     0,   149,   237,   234,     0,   239,     0,   172,
     173,     0,   163,    96,   184,   184,   184,   184,   167,     0,
       0,   170,     0,     0,     0,     0,     0,   159,   160,     0,
     157,   181,     0,     0,   127,     0,   215,     0,     9,     0,
       0,     0,     0,     0,     0,   166,   292,   269,     0,     0,
     133,   134,   139,     0,     0,    76,    63,    62,     0,   301,
       0,     0,   303,   265,     0,   104,   105,   117,    90,    91,
      92,    95,     0,    89,     0,   144,     0,     0,   392,   149,
     151,   303,   153,     0,   153,   153,     0,   153,   244,   162,
       0,   115,     0,     0,     0,     0,     0,     0,   193,     0,
     169,   184,   184,   156,     0,   174,     0,   203,    61,     0,
       0,   213,   205,     0,   217,    75,   184,   184,   184,   184,
     184,     0,   276,     0,   272,     0,   138,     0,     0,    96,
     303,   300,   303,   303,   308,   149,     0,    97,     0,     0,
       0,   143,   150,     0,   147,     0,     0,     0,     0,     0,
       0,   164,   186,   185,     0,   223,   188,   189,   190,   191,
     192,   171,     0,     0,   158,   175,   182,     0,   174,     0,
       0,     0,   211,   212,     0,   206,   207,   208,   214,   216,
       0,     0,     0,     0,     0,   174,   201,     0,     0,   275,
       0,     0,   137,     0,   306,   302,   307,   305,   152,    93,
       0,     0,   142,   393,   148,   238,     0,   235,     0,     0,
     240,     0,   250,     0,     0,     0,     0,     0,   246,   247,
     194,   195,     0,   180,   183,     0,   204,     0,   196,   197,
     198,   199,   200,     0,   271,   273,     0,     0,   136,     0,
      74,     0,   141,   153,   153,   153,   187,     0,     0,   248,
       9,   249,   226,   176,   203,   203,     0,   277,   135,     0,
       0,   336,   312,   303,   331,     0,     0,     0,     0,     0,
       0,    61,     0,     0,   224,     0,     0,   209,   210,   202,
       0,   303,   303,   174,     0,     0,   335,   123,   236,   242,
     241,     0,     0,     0,   177,   178,     0,     0,     0,   309,
       0,   313,   315,   332,     0,   381,     0,     0,     0,     0,
       0,   352,     0,     0,     0,   341,   303,   263,   370,   342,
     339,   316,   317,   298,   297,   299,   314,     0,   387,   303,
       0,   303,     0,   390,     0,     0,   351,     0,   303,     0,
       0,     0,     0,     0,     0,     0,   385,     0,     0,     0,
     388,   303,     0,     0,   354,     0,     0,   303,     0,     0,
       0,     0,     0,   352,     0,     0,   303,     0,   348,   350,
       9,   345,     9,     0,   262,     0,     0,   303,     0,   386,
       0,     0,   391,   353,     0,   369,   347,     0,     0,   303,
       0,   303,     0,     0,   303,     0,     0,   371,   349,   343,
     380,   340,   318,   319,   320,   338,     0,     0,   333,     0,
     303,     0,   303,     0,   378,     0,   355,     9,     0,   382,
       0,     0,     0,     0,   303,     0,     0,     9,     0,     0,
       0,   337,     0,   303,     0,     0,   389,   368,     0,     0,
     376,   303,     0,     0,   357,     0,     0,   358,   367,     0,
       0,   303,     0,   334,     0,     0,   303,   379,   382,     0,
     383,     0,   303,     0,   365,     9,     0,   382,   321,     0,
       0,     0,     0,     0,     0,     0,   377,     0,   303,     0,
       0,   356,     0,   363,   329,     0,     0,     0,     0,     0,
     327,     0,   264,     0,   373,   303,   384,     0,   303,   366,
     382,     0,   323,     0,     0,     0,     0,   330,     0,     0,
       0,     0,   364,   326,   325,   324,   322,   328,   372,     0,
       0,   360,   303,     0,   374,     0,     0,     0,   359,     0,
     375,     0,   361,     0,   362
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -662,  -662,   719,  -662,   -54,  -283,    -1,   -57,   658,   674,
     -26,  -662,  -662,  -662,  -273,  -662,  -209,  -662,  -129,   -90,
    -119,  -125,  -116,  -176,   602,   522,  -662,   -81,  -662,  -662,
    -289,  -662,  -662,   -79,   541,   379,  -662,    26,   412,  -662,
    -662,   590,   404,  -662,   256,  -662,  -662,  -368,  -662,  -105,
     330,  -662,  -662,  -662,    -4,  -662,  -662,  -662,  -662,  -662,
    -662,  -340,   432,  -662,   433,   723,  -662,  -192,   331,   725,
    -662,  -662,   551,  -662,  -662,  -662,  -662,   352,  -662,   312,
     354,  -662,   380,  -286,  -662,  -662,   438,   -83,  -488,   -64,
    -563,  -662,  -662,  -417,  -662,  -662,  -459,   140,  -484,  -662,
    -662,   230,  -576,   192,  -570,   222,  -519,  -662,  -500,  -661,
    -555,  -584,  -497,  -662,   248,   275,   205,  -662,  -662
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     3,     4,    72,   405,   198,   263,   155,     5,    63,
      73,    74,    75,   320,   321,   322,   245,   156,   264,   157,
     158,   159,   160,   161,   162,   224,   225,   323,   393,   329,
     330,   106,   107,   165,   180,   279,   280,   172,   261,   296,
     271,   177,   272,   262,   417,   526,   418,   419,   108,   343,
     403,   109,   110,   111,   178,   112,   192,   193,   194,   195,
     196,   422,   361,   286,   287,   464,   114,   406,   465,   466,
     116,   117,   170,   183,   467,   468,   131,   469,    76,   226,
     135,   374,   375,   218,   219,   576,   310,   596,   513,   566,
     234,   514,   657,   719,   702,   658,   515,   659,   490,   626,
     594,   567,   590,   605,   617,   587,   568,   619,   591,   690,
     597,   630,   579,   583,   584,   331,   454,    77,    78
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[] =
{
      56,    57,    62,    62,   169,   365,   141,    89,   284,   163,
     222,    84,   372,   166,   168,   621,   221,    88,   423,   223,
     130,   531,   532,   235,   638,   137,   599,   634,   164,   622,
     636,   249,   516,   608,   317,   306,   415,   358,   542,   415,
     392,   415,   569,    58,   265,   266,   267,   726,   648,   132,
     473,   281,    79,   139,   341,    80,   733,   232,   118,   570,
     396,   581,   138,   399,   197,   588,    81,   483,    85,    86,
     618,   359,   661,   328,   667,   676,   553,   307,   185,   136,
     250,   140,   167,   677,   604,   606,   270,   685,   693,   762,
      83,   696,   688,   595,   569,   222,   448,   243,   600,   618,
     181,   221,   416,   285,   223,  -179,   253,   684,   254,   255,
     227,   257,   639,   449,   641,   138,   549,   644,   550,   704,
     249,   724,   173,   174,   175,   301,   618,    83,   705,   229,
     664,   662,   715,   727,   138,   230,   304,   730,   669,   231,
     732,   332,   606,   138,   259,   351,   308,   352,   311,   154,
     494,   725,   407,   408,   409,   237,    83,   425,   426,   238,
     344,   345,   346,   760,   527,   528,   260,   758,   169,   250,
     443,   251,   252,   274,   164,   769,   302,   303,   154,   171,
     313,   686,   176,   270,   138,   179,   293,   284,   244,   710,
     759,   701,   779,   714,   775,   712,   717,   777,   298,   268,
     507,   462,   299,    83,   269,   783,    90,    91,    92,    93,
      94,   182,   289,   290,   291,   292,   739,   524,   101,   102,
     470,   471,   103,   549,   744,   154,   243,   555,    60,   384,
      61,   228,    83,   269,   197,  -205,    60,  -205,   478,   479,
     480,   481,   482,   184,   463,   360,   412,   413,   394,  -203,
     233,  -203,   402,   395,   154,   397,   398,   385,   400,   421,
     771,   430,   431,   432,   433,   434,     1,     2,   347,   774,
     427,   556,   557,   558,   559,   560,   561,   562,    82,   782,
      83,   357,   285,  -225,   362,   555,  -310,   133,   366,   367,
     368,   369,   370,   371,    83,   573,   574,   444,   373,   446,
     447,   256,   489,   379,   563,   741,   328,   246,    87,  -310,
     734,    60,   735,    87,  -310,   736,   737,   436,   751,   738,
     754,   247,   756,   388,  -268,  -268,    60,   248,   404,   556,
     557,   558,   559,   560,   561,   562,   472,   649,   258,   650,
     476,   458,   555,  -268,  -310,   239,   240,   241,   242,  -268,
    -268,  -268,  -268,  -268,  -268,  -268,  -268,  -268,    60,   273,
     435,   275,   563,   277,   278,  -268,    87,   633,   315,   316,
     138,   402,  -310,   222,   333,   288,   138,   334,   134,   221,
     336,   295,   223,   337,   687,   305,   556,   557,   558,   559,
     560,   561,   562,   297,   698,   757,   309,   735,   325,   326,
     736,   737,   652,   312,   738,   376,   377,   314,   512,   327,
     512,   324,   501,   340,   517,   518,   519,   244,   335,   563,
     339,   342,   349,    87,  -344,   268,   348,   350,   530,   530,
     534,   354,   731,   355,   143,   144,   767,   373,   735,   353,
     363,   736,   737,   364,   301,   738,   197,   380,   547,   548,
     378,   381,   529,    83,   512,   555,   383,   382,   386,   145,
     146,   147,   148,   149,   150,   151,   152,   153,   387,   653,
     654,   503,   545,   328,   504,   154,   609,   610,   611,   559,
     612,   613,   614,   592,   389,   390,   391,   437,   565,   655,
     410,   735,   763,   575,   736,   737,   421,   522,   738,   556,
     557,   558,   559,   560,   561,   562,   420,   424,   360,   615,
     438,   533,   735,    87,   439,   736,   737,   765,   631,   738,
     440,   441,   543,   607,   637,   616,   555,   442,   450,   546,
     453,   451,   563,   646,   452,   455,    87,  -346,   555,   456,
     565,   457,   459,   461,   656,   186,   187,   188,   189,   190,
     191,   460,   474,   577,   616,   475,   670,   477,   672,   415,
     484,   675,   660,   486,   487,   488,   197,   492,   197,   489,
     556,   557,   558,   559,   560,   561,   562,   682,   491,   674,
     493,   616,   556,   557,   558,   559,   560,   561,   562,   555,
     463,   695,   495,   500,   506,   505,   508,   535,   700,   656,
     525,   536,   555,   563,   537,    60,   538,   564,   711,   539,
     540,   541,   -11,   197,   554,   563,   544,   582,   721,   603,
     552,   549,   580,   197,   571,   572,   585,   555,   578,   729,
     586,   589,   593,   556,   557,   558,   559,   560,   561,   562,
     598,   602,   620,    87,   623,   747,   556,   557,   558,   559,
     560,   561,   562,   627,   625,   678,   628,   629,   635,   640,
     642,   197,   645,   651,   142,   761,   563,   647,   666,   742,
      87,   556,   557,   558,   559,   560,   561,   562,   671,   563,
     679,   680,   681,   663,   683,   689,   143,   144,   691,   776,
     694,   692,   697,   740,   706,   699,   722,   707,   718,   720,
     708,   746,   728,   723,   563,    83,   748,   745,   668,   764,
     749,   145,   146,   147,   148,   149,   150,   151,   152,   153,
     750,   718,   755,   768,    59,   772,   778,   154,   780,   784,
     105,    64,   294,   414,   718,   752,   718,   133,   718,  -268,
    -268,  -268,   300,  -268,  -268,  -268,   236,  -268,  -268,  -268,
    -268,  -268,   401,   411,   766,  -268,  -268,  -268,  -268,  -268,
    -268,  -268,  -268,  -268,  -268,  -268,  -268,  -268,   276,  -268,
    -268,  -268,  -268,  -268,  -268,  -268,  -268,  -268,  -268,  -268,
    -268,  -268,  -268,  -268,  -268,  -268,  -268,  -268,  -268,   551,
    -268,   496,  -268,  -268,   428,   113,   429,   115,   502,  -268,
    -268,  -268,  -268,  -268,  -268,  -268,  -268,  -268,  -268,   338,
     523,  -268,  -268,  -268,  -268,  -268,   499,   498,   485,   445,
     703,     6,     7,     8,   624,     9,    10,    11,   497,    12,
      13,    14,    15,    16,   673,   643,   665,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
     632,    30,    31,    32,    33,    34,   555,   601,    35,    36,
      37,    38,    39,    40,    41,    42,    43,    44,    45,    46,
      47,     0,    48,     0,    49,    50,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      51,     0,     0,    52,    53,    54,    55,   555,     0,     0,
     556,   557,   558,   559,   560,   561,   562,   555,    65,   356,
      -5,    -5,    66,    -5,    -5,    -5,    -5,    -5,    -5,    -5,
      -5,    -5,    -5,    -5,     0,    -5,    -5,     0,     0,    -5,
       0,     0,     0,   563,     0,     0,     0,   709,     0,     0,
       0,   556,   557,   558,   559,   560,   561,   562,   555,     0,
       0,   556,   557,   558,   559,   560,   561,   562,   555,     0,
       0,     0,    67,    68,     0,   555,     0,     0,    69,    70,
       0,     0,   555,     0,   563,     0,     0,     0,   713,     0,
       0,     0,    71,     0,   563,     0,     0,     0,   716,    -5,
     -68,     0,   556,   557,   558,   559,   560,   561,   562,     0,
       0,     0,   556,   557,   558,   559,   560,   561,   562,   556,
     557,   558,   559,   560,   561,   562,   556,   557,   558,   559,
     560,   561,   562,     0,     0,   563,     0,     0,     0,   743,
       0,     0,     0,     0,     0,   563,     0,     0,     0,   770,
       0,     0,   563,     0,     0,     0,   773,     0,     0,   563,
       0,     1,     2,   781,    90,    91,    92,    93,    94,    95,
      96,    97,    98,    99,   100,   199,   101,   102,     0,     0,
     103,   119,   120,   121,   122,     0,   123,   124,   125,   126,
     127,     0,     0,     0,     0,   200,     0,   201,   202,   203,
     204,   205,   206,   207,   143,   144,   208,   209,   210,   211,
     212,   213,     0,     0,     0,     0,     0,     0,     0,   128,
       0,     0,     0,    83,   318,   319,     0,   214,   215,   145,
     146,   147,   148,   149,   150,   151,   152,   153,     0,     0,
     104,     0,     0,     0,     0,   154,   520,     0,     0,     0,
     216,   217,     0,     0,     0,    60,     0,     0,   129,     6,
       7,     8,     0,     9,    10,    11,     0,    12,    13,    14,
      15,    16,     0,     0,     0,    17,    18,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,     0,    30,
      31,    32,    33,    34,   143,   220,    35,    36,    37,    38,
      39,    40,    41,    42,    43,    44,    45,    46,    47,     0,
      48,     0,    49,   521,     0,     0,   199,     0,     0,   145,
     146,   147,   148,   149,   150,   151,   152,   153,    51,     0,
       0,    52,    53,    54,    55,   154,   200,     0,   201,   202,
     203,   204,   205,   206,   207,     0,     0,   208,   209,   210,
     211,   212,   213,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   214,   215,
       0,     0,     0,     0,     0,     0,     0,     6,     7,     8,
       0,     9,    10,    11,     0,    12,    13,    14,    15,    16,
       0,   216,   217,    17,    18,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,    29,     0,    30,    31,    32,
      33,    34,     0,     0,    35,    36,    37,    38,    39,    40,
      41,    42,    43,    44,    45,    46,    47,     0,    48,     0,
      49,    50,   753,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    51,     0,     0,    52,
      53,    54,    55,     6,     7,     8,     0,     9,    10,    11,
       0,    12,    13,    14,    15,    16,     0,     0,     0,    17,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      28,    29,   652,    30,    31,    32,    33,    34,     0,     0,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,     0,    48,     0,    49,    50,     0,     0,
       0,     0,     0,     0,   143,   144,     0,   282,     0,     0,
       0,     0,    51,     0,     0,    52,    53,    54,    55,     0,
       0,     0,     0,    83,     0,     0,     0,   143,   144,   145,
     146,   147,   148,   149,   150,   151,   152,   153,     0,   653,
     654,     0,     0,     0,     0,   154,    83,   143,   144,   509,
     510,     0,   145,   146,   147,   148,   149,   150,   151,   152,
     153,     0,     0,     0,     0,     0,    83,     0,   283,     0,
       0,     0,   145,   146,   147,   148,   149,   150,   151,   152,
     153,   143,   144,   509,   510,     0,     0,     0,   154,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   511,     0,
      83,   143,   220,     0,     0,     0,   145,   146,   147,   148,
     149,   150,   151,   152,   153,     0,     0,     0,   143,   144,
      83,     0,   154,     0,     0,     0,   145,   146,   147,   148,
     149,   150,   151,   152,   153,   143,     0,    83,     0,     0,
       0,     0,   154,   145,   146,   147,   148,   149,   150,   151,
     152,   153,     0,     0,    83,     0,     0,     0,     0,   154,
     145,   146,   147,   148,   149,   150,   151,   152,   153,     0,
       0,     0,     0,     0,     0,     0,   154
};

static const yytype_int16 yycheck[] =
{
       1,     2,    56,    57,    94,   288,    89,    71,   184,    90,
     135,    68,   298,    92,    93,   591,   135,    71,   358,   135,
      74,   509,   510,   142,   608,    82,   581,   603,    38,   592,
     606,    39,   491,   588,   243,    15,    17,    57,   522,    17,
     329,    17,   542,     0,   173,   174,   175,   708,   618,    75,
     418,   180,    80,    60,   263,    44,   717,   140,    83,   543,
     333,   558,    81,   336,   118,   562,    67,   435,    69,    70,
     589,    91,   627,    92,   637,   645,   535,    57,   104,    82,
      88,    88,    92,   646,   584,   585,   176,   663,   672,   750,
      57,   675,   668,   577,   594,   220,   385,   154,   582,   618,
     101,   220,    83,   184,   220,    83,   163,   662,   165,   166,
     136,   168,   609,   386,   611,    81,    92,   614,    94,   682,
      39,   705,    96,    97,    98,    39,   645,    57,   683,    82,
     630,   628,   695,   709,    81,    88,   226,   713,   638,    92,
     716,    88,   642,    81,    31,   274,   229,   276,   231,    79,
      88,   706,   344,   345,   346,    64,    57,    58,    59,    68,
     265,   266,   267,   747,   504,   505,    53,   743,   258,    88,
     379,    90,    91,   177,    38,   759,    90,    91,    79,    92,
     234,   665,    92,   273,    81,    92,   190,   363,    85,   689,
     745,   679,   776,   693,   770,   692,   696,   773,    84,    53,
     486,     1,    88,    57,    58,   781,     6,     7,     8,     9,
      10,    59,   186,   187,   188,   189,    85,   500,    18,    19,
     412,   413,    22,    92,   724,    79,   283,     1,    80,   312,
      82,    81,    57,    58,   288,    84,    80,    86,   430,   431,
     432,   433,   434,    85,    44,    94,   351,   352,   331,    84,
      83,    86,   342,   332,    79,   334,   335,   314,   337,    94,
     760,   366,   367,   368,   369,   370,     3,     4,   269,   769,
     360,    45,    46,    47,    48,    49,    50,    51,    55,   779,
      57,   282,   363,    83,   285,     1,    60,     1,   289,   290,
     291,   292,   293,   294,    57,    58,    59,   380,   299,   382,
     383,    72,    88,   304,    78,   722,    92,    87,    82,    83,
      86,    80,    88,    82,    88,    91,    92,   371,   735,    95,
     737,    87,   739,   324,    38,    39,    80,    87,    82,    45,
      46,    47,    48,    49,    50,    51,   417,   620,    84,   622,
     421,   398,     1,    57,    60,    63,    64,    65,    66,    63,
      64,    65,    66,    67,    68,    69,    70,    71,    80,    92,
      82,    92,    78,    42,    43,    79,    82,    83,    63,    64,
      81,   461,    88,   498,    85,    83,    81,    88,    92,   498,
      85,    59,   498,    88,   667,    83,    45,    46,    47,    48,
      49,    50,    51,    93,   677,    86,    60,    88,    90,    91,
      91,    92,     6,    89,    95,    90,    91,    87,   489,    91,
     491,    87,   466,    84,   493,   494,   495,    85,    93,    78,
      93,    81,    84,    82,    83,    53,    93,    93,   509,   510,
     513,    84,   715,    82,    38,    39,    86,   438,    88,    93,
      84,    91,    92,    86,    39,    95,   500,    83,   531,   532,
      91,    84,   506,    57,   535,     1,    93,    89,    84,    63,
      64,    65,    66,    67,    68,    69,    70,    71,    86,    73,
      74,   472,   526,    92,   475,    79,    45,    46,    47,    48,
      49,    50,    51,   566,    91,    91,    91,    81,   542,    93,
      93,    88,    89,   550,    91,    92,    94,   498,    95,    45,
      46,    47,    48,    49,    50,    51,    91,    91,    94,    78,
      84,   512,    88,    82,    89,    91,    92,    93,   601,    95,
      91,    91,   523,   587,   607,   589,     1,    91,    89,   530,
      93,    91,    78,   616,    91,    89,    82,    83,     1,    86,
     594,    89,    86,    84,   625,    11,    12,    13,    14,    15,
      16,    89,    83,   554,   618,    91,   639,    91,   641,    17,
      58,   644,   626,    84,    91,    91,   620,    91,   622,    88,
      45,    46,    47,    48,    49,    50,    51,   660,    88,   643,
      88,   645,    45,    46,    47,    48,    49,    50,    51,     1,
      44,   674,    88,    83,    83,    57,    91,    84,   679,   680,
      92,    89,     1,    78,    89,    80,    89,    82,   691,    89,
      89,    93,    88,   667,    38,    78,    93,    57,   701,    82,
      93,    92,    92,   677,    93,    93,    82,     1,    88,   712,
      59,    82,    20,    45,    46,    47,    48,    49,    50,    51,
      57,    84,    83,    82,    94,   728,    45,    46,    47,    48,
      49,    50,    51,    80,    92,   656,    93,    89,    83,    92,
      82,   715,    82,    58,    16,   748,    78,    83,    83,   723,
      82,    45,    46,    47,    48,    49,    50,    51,    57,    78,
      81,    84,    93,    82,    80,    52,    38,    39,    80,   772,
      83,    93,    83,    93,    81,    85,    85,    83,   699,   700,
      83,    83,    80,   704,    78,    57,    81,    84,    82,    93,
      83,    63,    64,    65,    66,    67,    68,    69,    70,    71,
      83,   722,    86,    83,     5,    84,    83,    79,    83,    83,
      72,    57,   191,   354,   735,   736,   737,     1,   739,     3,
       4,     5,   220,     7,     8,     9,   144,    11,    12,    13,
      14,    15,   340,   349,   755,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,    29,    30,    31,   178,    33,
      34,    35,    36,    37,    38,    39,    40,    41,    42,    43,
      44,    45,    46,    47,    48,    49,    50,    51,    52,   533,
      54,   461,    56,    57,   362,    72,   363,    72,   467,    63,
      64,    65,    66,    67,    68,    69,    70,    71,    72,   258,
     498,    75,    76,    77,    78,    79,   464,   463,   438,   381,
     680,     3,     4,     5,   594,     7,     8,     9,    92,    11,
      12,    13,    14,    15,   642,   613,   631,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
     602,    33,    34,    35,    36,    37,     1,   582,    40,    41,
      42,    43,    44,    45,    46,    47,    48,    49,    50,    51,
      52,    -1,    54,    -1,    56,    57,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      72,    -1,    -1,    75,    76,    77,    78,     1,    -1,    -1,
      45,    46,    47,    48,    49,    50,    51,     1,     1,    91,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    14,    15,    16,    -1,    18,    19,    -1,    -1,    22,
      -1,    -1,    -1,    78,    -1,    -1,    -1,    82,    -1,    -1,
      -1,    45,    46,    47,    48,    49,    50,    51,     1,    -1,
      -1,    45,    46,    47,    48,    49,    50,    51,     1,    -1,
      -1,    -1,    55,    56,    -1,     1,    -1,    -1,    61,    62,
      -1,    -1,     1,    -1,    78,    -1,    -1,    -1,    82,    -1,
      -1,    -1,    75,    -1,    78,    -1,    -1,    -1,    82,    82,
      83,    -1,    45,    46,    47,    48,    49,    50,    51,    -1,
      -1,    -1,    45,    46,    47,    48,    49,    50,    51,    45,
      46,    47,    48,    49,    50,    51,    45,    46,    47,    48,
      49,    50,    51,    -1,    -1,    78,    -1,    -1,    -1,    82,
      -1,    -1,    -1,    -1,    -1,    78,    -1,    -1,    -1,    82,
      -1,    -1,    78,    -1,    -1,    -1,    82,    -1,    -1,    78,
      -1,     3,     4,    82,     6,     7,     8,     9,    10,    11,
      12,    13,    14,    15,    16,     1,    18,    19,    -1,    -1,
      22,     6,     7,     8,     9,    -1,    11,    12,    13,    14,
      15,    -1,    -1,    -1,    -1,    21,    -1,    23,    24,    25,
      26,    27,    28,    29,    38,    39,    32,    33,    34,    35,
      36,    37,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    44,
      -1,    -1,    -1,    57,    58,    59,    -1,    53,    54,    63,
      64,    65,    66,    67,    68,    69,    70,    71,    -1,    -1,
      82,    -1,    -1,    -1,    -1,    79,    72,    -1,    -1,    -1,
      76,    77,    -1,    -1,    -1,    80,    -1,    -1,    83,     3,
       4,     5,    -1,     7,     8,     9,    -1,    11,    12,    13,
      14,    15,    -1,    -1,    -1,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,    29,    30,    31,    -1,    33,
      34,    35,    36,    37,    38,    39,    40,    41,    42,    43,
      44,    45,    46,    47,    48,    49,    50,    51,    52,    -1,
      54,    -1,    56,    57,    -1,    -1,     1,    -1,    -1,    63,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    -1,
      -1,    75,    76,    77,    78,    79,    21,    -1,    23,    24,
      25,    26,    27,    28,    29,    -1,    -1,    32,    33,    34,
      35,    36,    37,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    53,    54,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,     3,     4,     5,
      -1,     7,     8,     9,    -1,    11,    12,    13,    14,    15,
      -1,    76,    77,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    31,    -1,    33,    34,    35,
      36,    37,    -1,    -1,    40,    41,    42,    43,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    -1,    54,    -1,
      56,    57,    58,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    72,    -1,    -1,    75,
      76,    77,    78,     3,     4,     5,    -1,     7,     8,     9,
      -1,    11,    12,    13,    14,    15,    -1,    -1,    -1,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    31,     6,    33,    34,    35,    36,    37,    -1,    -1,
      40,    41,    42,    43,    44,    45,    46,    47,    48,    49,
      50,    51,    52,    -1,    54,    -1,    56,    57,    -1,    -1,
      -1,    -1,    -1,    -1,    38,    39,    -1,    18,    -1,    -1,
      -1,    -1,    72,    -1,    -1,    75,    76,    77,    78,    -1,
      -1,    -1,    -1,    57,    -1,    -1,    -1,    38,    39,    63,
      64,    65,    66,    67,    68,    69,    70,    71,    -1,    73,
      74,    -1,    -1,    -1,    -1,    79,    57,    38,    39,    40,
      41,    -1,    63,    64,    65,    66,    67,    68,    69,    70,
      71,    -1,    -1,    -1,    -1,    -1,    57,    -1,    79,    -1,
      -1,    -1,    63,    64,    65,    66,    67,    68,    69,    70,
      71,    38,    39,    40,    41,    -1,    -1,    -1,    79,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    89,    -1,
      57,    38,    39,    -1,    -1,    -1,    63,    64,    65,    66,
      67,    68,    69,    70,    71,    -1,    -1,    -1,    38,    39,
      57,    -1,    79,    -1,    -1,    -1,    63,    64,    65,    66,
      67,    68,    69,    70,    71,    38,    -1,    57,    -1,    -1,
      -1,    -1,    79,    63,    64,    65,    66,    67,    68,    69,
      70,    71,    -1,    -1,    57,    -1,    -1,    -1,    -1,    79,
      63,    64,    65,    66,    67,    68,    69,    70,    71,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    79
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     3,     4,    97,    98,   104,     3,     4,     5,     7,
       8,     9,    11,    12,    13,    14,    15,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      33,    34,    35,    36,    37,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    54,    56,
      57,    72,    75,    76,    77,    78,   102,   102,     0,    98,
      80,    82,   100,   105,   105,     1,     5,    55,    56,    61,
      62,    75,    99,   106,   107,   108,   174,   213,   214,    80,
      44,   102,    55,    57,   103,   102,   102,    82,   100,   185,
       6,     7,     8,     9,    10,    11,    12,    13,    14,    15,
      16,    18,    19,    22,    82,   104,   127,   128,   144,   147,
     148,   149,   151,   161,   162,   165,   166,   167,    83,     6,
       7,     8,     9,    11,    12,    13,    14,    15,    44,    83,
     100,   172,   106,     1,    92,   176,    82,   103,    81,    60,
      88,   183,    16,    38,    39,    63,    64,    65,    66,    67,
      68,    69,    70,    71,    79,   103,   113,   115,   116,   117,
     118,   119,   120,   123,    38,   129,   129,    92,   129,   115,
     168,    92,   133,   133,   133,   133,    92,   137,   150,    92,
     130,   102,    59,   169,    85,   106,    11,    12,    13,    14,
      15,    16,   152,   153,   154,   155,   156,   100,   101,     1,
      21,    23,    24,    25,    26,    27,    28,    29,    32,    33,
      34,    35,    36,    37,    53,    54,    76,    77,   179,   180,
      39,   116,   117,   118,   121,   122,   175,   106,    81,    82,
      88,    92,   183,    83,   186,   116,   120,    64,    68,    63,
      64,    65,    66,   103,    85,   112,    87,    87,    87,    39,
      88,    90,    91,   103,   103,   103,    72,   103,    84,    31,
      53,   134,   139,   102,   114,   114,   114,   114,    53,    58,
     115,   136,   138,    92,   150,    92,   137,    42,    43,   131,
     132,   114,    18,    79,   119,   123,   159,   160,    83,   133,
     133,   133,   133,   150,   130,    59,   135,    93,    84,    88,
     121,    39,    90,    91,   115,    83,    15,    57,   183,    60,
     182,   183,    89,   100,    87,    63,    64,   112,    58,    59,
     109,   110,   111,   123,    87,    90,    91,    91,    92,   125,
     126,   211,    88,    85,    88,    93,    85,    88,   168,    93,
      84,   112,    81,   145,   145,   145,   145,   102,    93,    84,
      93,   114,   114,    93,    84,    82,    91,   102,    57,    91,
      94,   158,   102,    84,    86,   101,   102,   102,   102,   102,
     102,   102,   179,   102,   177,   178,    90,    91,    91,   102,
      83,    84,    89,    93,   183,   103,    84,    86,   102,    91,
      91,    91,   126,   124,   183,   129,   110,   129,   129,   110,
     129,   134,   115,   146,    82,   100,   163,   163,   163,   163,
      93,   138,   145,   145,   131,    17,    83,   140,   142,   143,
      91,    94,   157,   157,    91,    58,    59,   115,   158,   160,
     145,   145,   145,   145,   145,    82,   100,    81,    84,    89,
      91,    91,    91,   112,   183,   182,   183,   183,   126,   110,
      89,    91,    91,    93,   212,    89,    86,    89,   103,    86,
      89,    84,     1,    44,   161,   164,   165,   170,   171,   173,
     163,   163,   123,   143,    83,    91,   123,    91,   163,   163,
     163,   163,   163,   143,    58,   178,    84,    91,    91,    88,
     194,    88,    91,    88,    88,    88,   146,    92,   176,   173,
      83,   100,   164,   102,   102,    57,    83,   179,    91,    40,
      41,    89,   123,   184,   187,   192,   192,   129,   129,   129,
      72,    57,   102,   175,   101,    92,   141,   157,   157,   100,
     123,   184,   184,   102,   183,    84,    89,    89,    89,    89,
      89,    93,   194,   102,    93,   100,   102,   183,   183,    92,
      94,   140,    93,   192,    38,     1,    45,    46,    47,    48,
      49,    50,    51,    78,    82,   100,   185,   197,   202,   204,
     194,    93,    93,    58,    59,   103,   181,   102,    88,   208,
      92,   208,    57,   209,   210,    82,    59,   201,   208,    82,
     198,   204,   183,    20,   196,   194,   183,   206,    57,   206,
     194,   211,    84,    82,   204,   199,   204,   185,   206,    45,
      46,    47,    49,    50,    51,    78,   185,   200,   202,   203,
      83,   198,   186,    94,   197,    92,   195,    80,    93,    89,
     207,   183,   210,    83,   198,    83,   198,   183,   207,   208,
      92,   208,    82,   201,   208,    82,   183,    83,   200,   101,
     101,    58,     6,    73,    74,    93,   123,   188,   191,   193,
     185,   206,   208,    82,   204,   212,    83,   186,    82,   204,
     183,    57,   183,   199,   185,   183,   200,   186,   102,    81,
      84,    93,   183,    80,   206,   198,   194,   101,   198,    52,
     205,    80,    93,   207,    83,   183,   207,    83,   101,    85,
     123,   184,   190,   193,   186,   206,    81,    83,    83,    82,
     204,   183,   208,    82,   204,   186,    82,   204,   102,   189,
     102,   183,    85,   102,   207,   206,   205,   198,    80,   183,
     198,   101,   198,   205,    86,    88,    91,    92,    95,    85,
      93,   189,   100,    82,   204,    84,    83,   183,    81,    83,
      83,   189,   102,    58,   189,    86,   189,    86,   198,   206,
     207,   183,   205,    89,    93,    93,   102,    86,    83,   207,
      82,   204,    84,    82,   204,   198,   183,   198,    83,   207,
      83,    82,   204,   198,    83
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    96,    97,    98,    98,    99,    99,   100,   100,   101,
     101,   102,   102,   102,   102,   102,   102,   102,   102,   102,
     102,   102,   102,   102,   102,   102,   102,   102,   102,   102,
     102,   102,   102,   102,   102,   102,   102,   102,   102,   102,
     102,   102,   102,   102,   102,   102,   102,   102,   102,   102,
     102,   102,   102,   102,   102,   102,   102,   102,   102,   102,
     102,   103,   103,   103,   104,   104,   105,   105,   106,   106,
     107,   107,   107,   107,   107,   108,   108,   108,   108,   108,
     108,   108,   108,   108,   108,   108,   108,   108,   108,   109,
     109,   109,   110,   110,   111,   111,   112,   112,   113,   113,
     113,   113,   113,   113,   113,   113,   113,   113,   113,   113,
     113,   113,   113,   113,   113,   114,   115,   115,   116,   116,
     117,   118,   118,   119,   120,   120,   120,   120,   120,   120,
     121,   121,   121,   121,   121,   122,   122,   122,   122,   122,
     122,   123,   123,   123,   123,   123,   123,   124,   125,   126,
     126,   127,   128,   129,   129,   130,   130,   131,   131,   132,
     132,   133,   133,   134,   134,   135,   135,   136,   137,   137,
     138,   138,   139,   139,   140,   140,   141,   141,   142,   143,
     143,   144,   144,   144,   145,   145,   146,   146,   147,   147,
     148,   149,   150,   150,   151,   151,   152,   152,   153,   154,
     155,   156,   156,   157,   157,   158,   158,   158,   158,   159,
     159,   159,   159,   159,   159,   160,   160,   161,   162,   162,
     162,   162,   162,   163,   163,   164,   164,   165,   165,   165,
     165,   165,   165,   165,   166,   166,   166,   166,   166,   167,
     167,   167,   167,   168,   168,   169,   170,   171,   171,   171,
     171,   172,   172,   172,   172,   172,   172,   172,   172,   172,
     172,   172,   173,   173,   173,   174,   174,   175,   176,   176,
     176,   177,   178,   178,   179,   179,   179,   179,   180,   180,
     180,   180,   180,   180,   180,   180,   180,   180,   180,   180,
     180,   180,   180,   180,   180,   180,   180,   181,   181,   181,
     182,   182,   182,   183,   183,   183,   183,   183,   183,   184,
     185,   186,   187,   187,   187,   187,   187,   187,   188,   188,
     188,   189,   189,   189,   189,   189,   189,   190,   191,   191,
     191,   192,   192,   193,   193,   194,   194,   195,   195,   196,
     196,   197,   197,   197,   198,   198,   199,   199,   200,   200,
     200,   201,   201,   202,   202,   202,   203,   203,   203,   203,
     203,   203,   203,   203,   203,   203,   203,   203,   204,   204,
     204,   204,   204,   204,   204,   204,   204,   204,   204,   204,
     204,   204,   205,   205,   205,   206,   207,   208,   209,   209,
     210,   210,   211,   212,   213,   214
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
       1,     2,     1,     1,     1,     2,     2,     3,     1,     1,
       2,     2,     2,     8,     1,     1,     1,     1,     2,     2,
       1,     1,     1,     2,     2,     6,     5,     4,     3,     2,
       1,     6,     5,     4,     3,     2,     1,     1,     3,     0,
       2,     4,     6,     0,     1,     0,     3,     1,     3,     1,
       1,     0,     3,     1,     3,     0,     1,     1,     0,     3,
       1,     3,     1,     1,     0,     1,     0,     2,     5,     1,
       2,     3,     5,     6,     0,     2,     1,     3,     5,     5,
       5,     5,     4,     3,     6,     6,     5,     5,     5,     5,
       5,     4,     7,     0,     2,     0,     2,     2,     2,     6,
       6,     3,     3,     2,     3,     1,     3,     4,     2,     2,
       2,     2,     2,     1,     4,     0,     2,     1,     1,     1,
       1,     2,     2,     2,     3,     6,     9,     3,     6,     3,
       6,     9,     9,     1,     3,     1,     1,     1,     2,     2,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     7,     5,    13,     5,     2,     1,     0,     3,
       1,     3,     1,     3,     1,     4,     3,     6,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     2,     1,     1,     1,     1,     1,     1,     1,
       0,     1,     3,     0,     1,     5,     5,     5,     4,     3,
       1,     1,     1,     3,     4,     3,     4,     4,     1,     1,
       1,     1,     4,     3,     4,     4,     4,     3,     7,     5,
       6,     1,     3,     1,     3,     3,     2,     3,     2,     0,
       3,     1,     1,     4,     1,     2,     1,     2,     1,     2,
       1,     1,     0,     4,     3,     5,     6,     4,     4,    11,
       9,    12,    14,     6,     8,     5,     7,     4,     6,     4,
       1,     4,    11,     9,    12,    14,     6,     8,     5,     7,
       4,     1,     0,     2,     4,     1,     1,     1,     2,     5,
       1,     3,     1,     1,     2,     2
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
#line 2292 "y.tab.c" /* yacc.c:1646  */
    break;

  case 3:
#line 206 "xi-grammar.y" /* yacc.c:1646  */
    { 
		  (yyval.modlist) = 0; 
		}
#line 2300 "y.tab.c" /* yacc.c:1646  */
    break;

  case 4:
#line 210 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.modlist) = new AstChildren<Module>(lineno, (yyvsp[-1].module), (yyvsp[0].modlist)); }
#line 2306 "y.tab.c" /* yacc.c:1646  */
    break;

  case 5:
#line 214 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 2312 "y.tab.c" /* yacc.c:1646  */
    break;

  case 6:
#line 216 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 2318 "y.tab.c" /* yacc.c:1646  */
    break;

  case 7:
#line 220 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 2324 "y.tab.c" /* yacc.c:1646  */
    break;

  case 8:
#line 222 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 2; }
#line 2330 "y.tab.c" /* yacc.c:1646  */
    break;

  case 9:
#line 226 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 2336 "y.tab.c" /* yacc.c:1646  */
    break;

  case 10:
#line 228 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 2342 "y.tab.c" /* yacc.c:1646  */
    break;

  case 11:
#line 233 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 2348 "y.tab.c" /* yacc.c:1646  */
    break;

  case 12:
#line 234 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MODULE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2354 "y.tab.c" /* yacc.c:1646  */
    break;

  case 13:
#line 235 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MAINMODULE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2360 "y.tab.c" /* yacc.c:1646  */
    break;

  case 14:
#line 236 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(EXTERN, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2366 "y.tab.c" /* yacc.c:1646  */
    break;

  case 15:
#line 238 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITCALL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2372 "y.tab.c" /* yacc.c:1646  */
    break;

  case 16:
#line 239 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITNODE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2378 "y.tab.c" /* yacc.c:1646  */
    break;

  case 17:
#line 240 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITPROC, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2384 "y.tab.c" /* yacc.c:1646  */
    break;

  case 18:
#line 242 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CHARE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2390 "y.tab.c" /* yacc.c:1646  */
    break;

  case 19:
#line 243 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MAINCHARE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2396 "y.tab.c" /* yacc.c:1646  */
    break;

  case 20:
#line 244 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(GROUP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2402 "y.tab.c" /* yacc.c:1646  */
    break;

  case 21:
#line 245 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NODEGROUP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2408 "y.tab.c" /* yacc.c:1646  */
    break;

  case 22:
#line 246 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ARRAY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2414 "y.tab.c" /* yacc.c:1646  */
    break;

  case 23:
#line 250 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INCLUDE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2420 "y.tab.c" /* yacc.c:1646  */
    break;

  case 24:
#line 251 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(STACKSIZE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2426 "y.tab.c" /* yacc.c:1646  */
    break;

  case 25:
#line 252 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(THREADED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2432 "y.tab.c" /* yacc.c:1646  */
    break;

  case 26:
#line 253 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(TEMPLATE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2438 "y.tab.c" /* yacc.c:1646  */
    break;

  case 27:
#line 254 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(WHENIDLE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2444 "y.tab.c" /* yacc.c:1646  */
    break;

  case 28:
#line 255 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(SYNC, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2450 "y.tab.c" /* yacc.c:1646  */
    break;

  case 29:
#line 256 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IGET, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2456 "y.tab.c" /* yacc.c:1646  */
    break;

  case 30:
#line 257 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(EXCLUSIVE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2462 "y.tab.c" /* yacc.c:1646  */
    break;

  case 31:
#line 258 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IMMEDIATE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2468 "y.tab.c" /* yacc.c:1646  */
    break;

  case 32:
#line 259 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(SKIPSCHED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2474 "y.tab.c" /* yacc.c:1646  */
    break;

  case 33:
#line 260 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOCOPY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2480 "y.tab.c" /* yacc.c:1646  */
    break;

  case 34:
#line 261 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOCOPYPOST, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2486 "y.tab.c" /* yacc.c:1646  */
    break;

  case 35:
#line 262 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INLINE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2492 "y.tab.c" /* yacc.c:1646  */
    break;

  case 36:
#line 263 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(VIRTUAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2498 "y.tab.c" /* yacc.c:1646  */
    break;

  case 37:
#line 264 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MIGRATABLE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2504 "y.tab.c" /* yacc.c:1646  */
    break;

  case 38:
#line 265 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CREATEHERE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2510 "y.tab.c" /* yacc.c:1646  */
    break;

  case 39:
#line 266 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CREATEHOME, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2516 "y.tab.c" /* yacc.c:1646  */
    break;

  case 40:
#line 267 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOKEEP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2522 "y.tab.c" /* yacc.c:1646  */
    break;

  case 41:
#line 268 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOTRACE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2528 "y.tab.c" /* yacc.c:1646  */
    break;

  case 42:
#line 269 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(APPWORK, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2534 "y.tab.c" /* yacc.c:1646  */
    break;

  case 43:
#line 272 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(PACKED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2540 "y.tab.c" /* yacc.c:1646  */
    break;

  case 44:
#line 273 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(VARSIZE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2546 "y.tab.c" /* yacc.c:1646  */
    break;

  case 45:
#line 274 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ENTRY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2552 "y.tab.c" /* yacc.c:1646  */
    break;

  case 46:
#line 275 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(FOR, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2558 "y.tab.c" /* yacc.c:1646  */
    break;

  case 47:
#line 276 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(FORALL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2564 "y.tab.c" /* yacc.c:1646  */
    break;

  case 48:
#line 277 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(WHILE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2570 "y.tab.c" /* yacc.c:1646  */
    break;

  case 49:
#line 278 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(WHEN, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2576 "y.tab.c" /* yacc.c:1646  */
    break;

  case 50:
#line 279 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(OVERLAP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2582 "y.tab.c" /* yacc.c:1646  */
    break;

  case 51:
#line 280 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(SERIAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2588 "y.tab.c" /* yacc.c:1646  */
    break;

  case 52:
#line 281 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IF, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2594 "y.tab.c" /* yacc.c:1646  */
    break;

  case 53:
#line 282 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ELSE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2600 "y.tab.c" /* yacc.c:1646  */
    break;

  case 54:
#line 284 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(LOCAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2606 "y.tab.c" /* yacc.c:1646  */
    break;

  case 55:
#line 286 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(USING, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2612 "y.tab.c" /* yacc.c:1646  */
    break;

  case 56:
#line 287 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ACCEL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2618 "y.tab.c" /* yacc.c:1646  */
    break;

  case 57:
#line 290 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ACCELBLOCK, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2624 "y.tab.c" /* yacc.c:1646  */
    break;

  case 58:
#line 291 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MEMCRITICAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2630 "y.tab.c" /* yacc.c:1646  */
    break;

  case 59:
#line 292 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(REDUCTIONTARGET, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2636 "y.tab.c" /* yacc.c:1646  */
    break;

  case 60:
#line 293 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CASE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2642 "y.tab.c" /* yacc.c:1646  */
    break;

  case 61:
#line 298 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 2648 "y.tab.c" /* yacc.c:1646  */
    break;

  case 62:
#line 300 "xi-grammar.y" /* yacc.c:1646  */
    {
		  char *tmp = new char[strlen((yyvsp[-3].strval))+strlen((yyvsp[0].strval))+3];
		  sprintf(tmp,"%s::%s", (yyvsp[-3].strval), (yyvsp[0].strval));
		  (yyval.strval) = tmp;
		}
#line 2658 "y.tab.c" /* yacc.c:1646  */
    break;

  case 63:
#line 306 "xi-grammar.y" /* yacc.c:1646  */
    {
		  char *tmp = new char[strlen((yyvsp[-3].strval))+5+3];
		  sprintf(tmp,"%s::array", (yyvsp[-3].strval));
		  (yyval.strval) = tmp;
		}
#line 2668 "y.tab.c" /* yacc.c:1646  */
    break;

  case 64:
#line 313 "xi-grammar.y" /* yacc.c:1646  */
    { 
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist)); 
		}
#line 2676 "y.tab.c" /* yacc.c:1646  */
    break;

  case 65:
#line 317 "xi-grammar.y" /* yacc.c:1646  */
    {  
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist)); 
		    (yyval.module)->setMain();
		}
#line 2685 "y.tab.c" /* yacc.c:1646  */
    break;

  case 66:
#line 324 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = 0; }
#line 2691 "y.tab.c" /* yacc.c:1646  */
    break;

  case 67:
#line 326 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = (yyvsp[-2].conslist); }
#line 2697 "y.tab.c" /* yacc.c:1646  */
    break;

  case 68:
#line 330 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = 0; }
#line 2703 "y.tab.c" /* yacc.c:1646  */
    break;

  case 69:
#line 332 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = new ConstructList(lineno, (yyvsp[-1].construct), (yyvsp[0].conslist)); }
#line 2709 "y.tab.c" /* yacc.c:1646  */
    break;

  case 70:
#line 336 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new UsingScope((yyvsp[0].strval), false); }
#line 2715 "y.tab.c" /* yacc.c:1646  */
    break;

  case 71:
#line 338 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new UsingScope((yyvsp[0].strval), true); }
#line 2721 "y.tab.c" /* yacc.c:1646  */
    break;

  case 72:
#line 340 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].member)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].member); }
#line 2727 "y.tab.c" /* yacc.c:1646  */
    break;

  case 73:
#line 342 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].message)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].message); }
#line 2733 "y.tab.c" /* yacc.c:1646  */
    break;

  case 74:
#line 344 "xi-grammar.y" /* yacc.c:1646  */
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
#line 2748 "y.tab.c" /* yacc.c:1646  */
    break;

  case 75:
#line 357 "xi-grammar.y" /* yacc.c:1646  */
    { if((yyvsp[-2].conslist)) (yyvsp[-2].conslist)->recurse<int&>((yyvsp[-4].intval), &Construct::setExtern); (yyval.construct) = (yyvsp[-2].conslist); }
#line 2754 "y.tab.c" /* yacc.c:1646  */
    break;

  case 76:
#line 359 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new Scope((yyvsp[-3].strval), (yyvsp[-1].conslist)); }
#line 2760 "y.tab.c" /* yacc.c:1646  */
    break;

  case 77:
#line 361 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = (yyvsp[-1].construct); }
#line 2766 "y.tab.c" /* yacc.c:1646  */
    break;

  case 78:
#line 363 "xi-grammar.y" /* yacc.c:1646  */
    {
          ERROR("preceding construct must be semicolon terminated",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 2776 "y.tab.c" /* yacc.c:1646  */
    break;

  case 79:
#line 369 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].module)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].module); }
#line 2782 "y.tab.c" /* yacc.c:1646  */
    break;

  case 80:
#line 371 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2788 "y.tab.c" /* yacc.c:1646  */
    break;

  case 81:
#line 373 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2794 "y.tab.c" /* yacc.c:1646  */
    break;

  case 82:
#line 375 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2800 "y.tab.c" /* yacc.c:1646  */
    break;

  case 83:
#line 377 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2806 "y.tab.c" /* yacc.c:1646  */
    break;

  case 84:
#line 379 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].templat)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].templat); }
#line 2812 "y.tab.c" /* yacc.c:1646  */
    break;

  case 85:
#line 381 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = NULL; }
#line 2818 "y.tab.c" /* yacc.c:1646  */
    break;

  case 86:
#line 383 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = NULL; }
#line 2824 "y.tab.c" /* yacc.c:1646  */
    break;

  case 87:
#line 385 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = (yyvsp[0].accelBlock); }
#line 2830 "y.tab.c" /* yacc.c:1646  */
    break;

  case 88:
#line 387 "xi-grammar.y" /* yacc.c:1646  */
    {
          ERROR("invalid construct",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 2840 "y.tab.c" /* yacc.c:1646  */
    break;

  case 89:
#line 395 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamType((yyvsp[0].type)); }
#line 2846 "y.tab.c" /* yacc.c:1646  */
    break;

  case 90:
#line 397 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
#line 2852 "y.tab.c" /* yacc.c:1646  */
    break;

  case 91:
#line 399 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
#line 2858 "y.tab.c" /* yacc.c:1646  */
    break;

  case 92:
#line 403 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = new TParamList((yyvsp[0].tparam)); }
#line 2864 "y.tab.c" /* yacc.c:1646  */
    break;

  case 93:
#line 405 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = new TParamList((yyvsp[-2].tparam), (yyvsp[0].tparlist)); }
#line 2870 "y.tab.c" /* yacc.c:1646  */
    break;

  case 94:
#line 409 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = new TParamList(0); }
#line 2876 "y.tab.c" /* yacc.c:1646  */
    break;

  case 95:
#line 411 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = (yyvsp[0].tparlist); }
#line 2882 "y.tab.c" /* yacc.c:1646  */
    break;

  case 96:
#line 415 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = 0; }
#line 2888 "y.tab.c" /* yacc.c:1646  */
    break;

  case 97:
#line 417 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = (yyvsp[-1].tparlist); }
#line 2894 "y.tab.c" /* yacc.c:1646  */
    break;

  case 98:
#line 421 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("int"); }
#line 2900 "y.tab.c" /* yacc.c:1646  */
    break;

  case 99:
#line 423 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long"); }
#line 2906 "y.tab.c" /* yacc.c:1646  */
    break;

  case 100:
#line 425 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("short"); }
#line 2912 "y.tab.c" /* yacc.c:1646  */
    break;

  case 101:
#line 427 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("char"); }
#line 2918 "y.tab.c" /* yacc.c:1646  */
    break;

  case 102:
#line 429 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned int"); }
#line 2924 "y.tab.c" /* yacc.c:1646  */
    break;

  case 103:
#line 431 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long"); }
#line 2930 "y.tab.c" /* yacc.c:1646  */
    break;

  case 104:
#line 433 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long"); }
#line 2936 "y.tab.c" /* yacc.c:1646  */
    break;

  case 105:
#line 435 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long long"); }
#line 2942 "y.tab.c" /* yacc.c:1646  */
    break;

  case 106:
#line 437 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned short"); }
#line 2948 "y.tab.c" /* yacc.c:1646  */
    break;

  case 107:
#line 439 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned char"); }
#line 2954 "y.tab.c" /* yacc.c:1646  */
    break;

  case 108:
#line 441 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long long"); }
#line 2960 "y.tab.c" /* yacc.c:1646  */
    break;

  case 109:
#line 443 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("float"); }
#line 2966 "y.tab.c" /* yacc.c:1646  */
    break;

  case 110:
#line 445 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("double"); }
#line 2972 "y.tab.c" /* yacc.c:1646  */
    break;

  case 111:
#line 447 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long double"); }
#line 2978 "y.tab.c" /* yacc.c:1646  */
    break;

  case 112:
#line 449 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("void"); }
#line 2984 "y.tab.c" /* yacc.c:1646  */
    break;

  case 113:
#line 451 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("size_t"); }
#line 2990 "y.tab.c" /* yacc.c:1646  */
    break;

  case 114:
#line 453 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("bool"); }
#line 2996 "y.tab.c" /* yacc.c:1646  */
    break;

  case 115:
#line 456 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ntype) = new NamedType((yyvsp[-1].strval),(yyvsp[0].tparlist)); }
#line 3002 "y.tab.c" /* yacc.c:1646  */
    break;

  case 116:
#line 457 "xi-grammar.y" /* yacc.c:1646  */
    { 
                    const char* basename, *scope;
                    splitScopedName((yyvsp[-1].strval), &scope, &basename);
                    (yyval.ntype) = new NamedType(basename, (yyvsp[0].tparlist), scope);
                }
#line 3012 "y.tab.c" /* yacc.c:1646  */
    break;

  case 117:
#line 463 "xi-grammar.y" /* yacc.c:1646  */
    {
			const char* basename, *scope;
			splitScopedName((yyvsp[-1].strval), &scope, &basename);
			(yyval.ntype) = new NamedType(basename, (yyvsp[0].tparlist), scope, true);
		}
#line 3022 "y.tab.c" /* yacc.c:1646  */
    break;

  case 118:
#line 471 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3028 "y.tab.c" /* yacc.c:1646  */
    break;

  case 119:
#line 473 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ntype); }
#line 3034 "y.tab.c" /* yacc.c:1646  */
    break;

  case 120:
#line 477 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ptype) = new PtrType((yyvsp[-1].type)); }
#line 3040 "y.tab.c" /* yacc.c:1646  */
    break;

  case 121:
#line 481 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
#line 3046 "y.tab.c" /* yacc.c:1646  */
    break;

  case 122:
#line 483 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
#line 3052 "y.tab.c" /* yacc.c:1646  */
    break;

  case 123:
#line 487 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ftype) = new FuncType((yyvsp[-7].type), (yyvsp[-4].strval), (yyvsp[-1].plist)); }
#line 3058 "y.tab.c" /* yacc.c:1646  */
    break;

  case 124:
#line 491 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3064 "y.tab.c" /* yacc.c:1646  */
    break;

  case 125:
#line 493 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3070 "y.tab.c" /* yacc.c:1646  */
    break;

  case 126:
#line 495 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3076 "y.tab.c" /* yacc.c:1646  */
    break;

  case 127:
#line 497 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ftype); }
#line 3082 "y.tab.c" /* yacc.c:1646  */
    break;

  case 128:
#line 499 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[0].type)); }
#line 3088 "y.tab.c" /* yacc.c:1646  */
    break;

  case 129:
#line 501 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[-1].type)); }
#line 3094 "y.tab.c" /* yacc.c:1646  */
    break;

  case 130:
#line 505 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3100 "y.tab.c" /* yacc.c:1646  */
    break;

  case 131:
#line 507 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3106 "y.tab.c" /* yacc.c:1646  */
    break;

  case 132:
#line 509 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3112 "y.tab.c" /* yacc.c:1646  */
    break;

  case 133:
#line 511 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[0].type)); }
#line 3118 "y.tab.c" /* yacc.c:1646  */
    break;

  case 134:
#line 513 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[-1].type)); }
#line 3124 "y.tab.c" /* yacc.c:1646  */
    break;

  case 135:
#line 517 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType(new RValueReferenceType((yyvsp[-5].type))); }
#line 3130 "y.tab.c" /* yacc.c:1646  */
    break;

  case 136:
#line 519 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType(new ReferenceType((yyvsp[-4].type))); }
#line 3136 "y.tab.c" /* yacc.c:1646  */
    break;

  case 137:
#line 521 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType((yyvsp[-3].type)); }
#line 3142 "y.tab.c" /* yacc.c:1646  */
    break;

  case 138:
#line 523 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new RValueReferenceType((yyvsp[-2].type)); }
#line 3148 "y.tab.c" /* yacc.c:1646  */
    break;

  case 139:
#line 525 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ReferenceType((yyvsp[-1].type)); }
#line 3154 "y.tab.c" /* yacc.c:1646  */
    break;

  case 140:
#line 527 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3160 "y.tab.c" /* yacc.c:1646  */
    break;

  case 141:
#line 531 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType(new RValueReferenceType((yyvsp[-5].type))); }
#line 3166 "y.tab.c" /* yacc.c:1646  */
    break;

  case 142:
#line 533 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType(new ReferenceType((yyvsp[-4].type))); }
#line 3172 "y.tab.c" /* yacc.c:1646  */
    break;

  case 143:
#line 535 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType((yyvsp[-3].type)); }
#line 3178 "y.tab.c" /* yacc.c:1646  */
    break;

  case 144:
#line 537 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new RValueReferenceType((yyvsp[-2].type)); }
#line 3184 "y.tab.c" /* yacc.c:1646  */
    break;

  case 145:
#line 539 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ReferenceType((yyvsp[-1].type)); }
#line 3190 "y.tab.c" /* yacc.c:1646  */
    break;

  case 146:
#line 541 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3196 "y.tab.c" /* yacc.c:1646  */
    break;

  case 147:
#line 545 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 3202 "y.tab.c" /* yacc.c:1646  */
    break;

  case 148:
#line 549 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = (yyvsp[-1].val); }
#line 3208 "y.tab.c" /* yacc.c:1646  */
    break;

  case 149:
#line 553 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.vallist) = 0; }
#line 3214 "y.tab.c" /* yacc.c:1646  */
    break;

  case 150:
#line 555 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.vallist) = new ValueList((yyvsp[-1].val), (yyvsp[0].vallist)); }
#line 3220 "y.tab.c" /* yacc.c:1646  */
    break;

  case 151:
#line 559 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].vallist)); }
#line 3226 "y.tab.c" /* yacc.c:1646  */
    break;

  case 152:
#line 563 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[-3].type), (yyvsp[-1].strval), (yyvsp[0].vallist), 1); }
#line 3232 "y.tab.c" /* yacc.c:1646  */
    break;

  case 153:
#line 567 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0;}
#line 3238 "y.tab.c" /* yacc.c:1646  */
    break;

  case 154:
#line 569 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0;}
#line 3244 "y.tab.c" /* yacc.c:1646  */
    break;

  case 155:
#line 573 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3250 "y.tab.c" /* yacc.c:1646  */
    break;

  case 156:
#line 575 "xi-grammar.y" /* yacc.c:1646  */
    { 
		  /*
		  printf("Warning: Message attributes are being phased out.\n");
		  printf("Warning: Please remove them from interface files.\n");
		  */
		  (yyval.intval) = (yyvsp[-1].intval); 
		}
#line 3262 "y.tab.c" /* yacc.c:1646  */
    break;

  case 157:
#line 585 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[0].intval); }
#line 3268 "y.tab.c" /* yacc.c:1646  */
    break;

  case 158:
#line 587 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[-2].intval) | (yyvsp[0].intval); }
#line 3274 "y.tab.c" /* yacc.c:1646  */
    break;

  case 159:
#line 591 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3280 "y.tab.c" /* yacc.c:1646  */
    break;

  case 160:
#line 593 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3286 "y.tab.c" /* yacc.c:1646  */
    break;

  case 161:
#line 597 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = 0; }
#line 3292 "y.tab.c" /* yacc.c:1646  */
    break;

  case 162:
#line 599 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-1].cattr); }
#line 3298 "y.tab.c" /* yacc.c:1646  */
    break;

  case 163:
#line 603 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[0].cattr); }
#line 3304 "y.tab.c" /* yacc.c:1646  */
    break;

  case 164:
#line 605 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
#line 3310 "y.tab.c" /* yacc.c:1646  */
    break;

  case 165:
#line 609 "xi-grammar.y" /* yacc.c:1646  */
    { python_doc = NULL; (yyval.intval) = 0; }
#line 3316 "y.tab.c" /* yacc.c:1646  */
    break;

  case 166:
#line 611 "xi-grammar.y" /* yacc.c:1646  */
    { python_doc = (yyvsp[0].strval); (yyval.intval) = 0; }
#line 3322 "y.tab.c" /* yacc.c:1646  */
    break;

  case 167:
#line 615 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CPYTHON; }
#line 3328 "y.tab.c" /* yacc.c:1646  */
    break;

  case 168:
#line 619 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = 0; }
#line 3334 "y.tab.c" /* yacc.c:1646  */
    break;

  case 169:
#line 621 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-1].cattr); }
#line 3340 "y.tab.c" /* yacc.c:1646  */
    break;

  case 170:
#line 625 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[0].cattr); }
#line 3346 "y.tab.c" /* yacc.c:1646  */
    break;

  case 171:
#line 627 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
#line 3352 "y.tab.c" /* yacc.c:1646  */
    break;

  case 172:
#line 631 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CMIGRATABLE; }
#line 3358 "y.tab.c" /* yacc.c:1646  */
    break;

  case 173:
#line 633 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CPYTHON; }
#line 3364 "y.tab.c" /* yacc.c:1646  */
    break;

  case 174:
#line 637 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3370 "y.tab.c" /* yacc.c:1646  */
    break;

  case 175:
#line 639 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 3376 "y.tab.c" /* yacc.c:1646  */
    break;

  case 176:
#line 642 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3382 "y.tab.c" /* yacc.c:1646  */
    break;

  case 177:
#line 644 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 3388 "y.tab.c" /* yacc.c:1646  */
    break;

  case 178:
#line 647 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mv) = new MsgVar((yyvsp[-3].type), (yyvsp[-2].strval), (yyvsp[-4].intval), (yyvsp[-1].intval)); }
#line 3394 "y.tab.c" /* yacc.c:1646  */
    break;

  case 179:
#line 651 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mvlist) = new MsgVarList((yyvsp[0].mv)); }
#line 3400 "y.tab.c" /* yacc.c:1646  */
    break;

  case 180:
#line 653 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mvlist) = new MsgVarList((yyvsp[-1].mv), (yyvsp[0].mvlist)); }
#line 3406 "y.tab.c" /* yacc.c:1646  */
    break;

  case 181:
#line 657 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, (yyvsp[0].ntype)); }
#line 3412 "y.tab.c" /* yacc.c:1646  */
    break;

  case 182:
#line 659 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, (yyvsp[-2].ntype)); }
#line 3418 "y.tab.c" /* yacc.c:1646  */
    break;

  case 183:
#line 661 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, (yyvsp[-3].ntype), (yyvsp[-1].mvlist)); }
#line 3424 "y.tab.c" /* yacc.c:1646  */
    break;

  case 184:
#line 665 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = 0; }
#line 3430 "y.tab.c" /* yacc.c:1646  */
    break;

  case 185:
#line 667 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = (yyvsp[0].typelist); }
#line 3436 "y.tab.c" /* yacc.c:1646  */
    break;

  case 186:
#line 671 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = new TypeList((yyvsp[0].ntype)); }
#line 3442 "y.tab.c" /* yacc.c:1646  */
    break;

  case 187:
#line 673 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = new TypeList((yyvsp[-2].ntype), (yyvsp[0].typelist)); }
#line 3448 "y.tab.c" /* yacc.c:1646  */
    break;

  case 188:
#line 677 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr)|Chare::CCHARE, (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3454 "y.tab.c" /* yacc.c:1646  */
    break;

  case 189:
#line 679 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3460 "y.tab.c" /* yacc.c:1646  */
    break;

  case 190:
#line 683 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3466 "y.tab.c" /* yacc.c:1646  */
    break;

  case 191:
#line 687 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new NodeGroup(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3472 "y.tab.c" /* yacc.c:1646  */
    break;

  case 192:
#line 691 "xi-grammar.y" /* yacc.c:1646  */
    {/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",(yyvsp[-2].strval));
			(yyval.ntype) = new NamedType(buf); 
		}
#line 3482 "y.tab.c" /* yacc.c:1646  */
    break;

  case 193:
#line 697 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ntype) = (yyvsp[-1].ntype); }
#line 3488 "y.tab.c" /* yacc.c:1646  */
    break;

  case 194:
#line 701 "xi-grammar.y" /* yacc.c:1646  */
    {  (yyval.chare) = new Array(lineno, (yyvsp[-4].cattr), (yyvsp[-3].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3494 "y.tab.c" /* yacc.c:1646  */
    break;

  case 195:
#line 703 "xi-grammar.y" /* yacc.c:1646  */
    {  (yyval.chare) = new Array(lineno, (yyvsp[-3].cattr), (yyvsp[-4].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3500 "y.tab.c" /* yacc.c:1646  */
    break;

  case 196:
#line 707 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr)|Chare::CCHARE, new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist));}
#line 3506 "y.tab.c" /* yacc.c:1646  */
    break;

  case 197:
#line 709 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3512 "y.tab.c" /* yacc.c:1646  */
    break;

  case 198:
#line 713 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3518 "y.tab.c" /* yacc.c:1646  */
    break;

  case 199:
#line 717 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new NodeGroup( lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3524 "y.tab.c" /* yacc.c:1646  */
    break;

  case 200:
#line 721 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Array( lineno, 0, (yyvsp[-3].ntype), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3530 "y.tab.c" /* yacc.c:1646  */
    break;

  case 201:
#line 725 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-1].strval))); }
#line 3536 "y.tab.c" /* yacc.c:1646  */
    break;

  case 202:
#line 727 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-4].strval)), (yyvsp[-2].mvlist)); }
#line 3542 "y.tab.c" /* yacc.c:1646  */
    break;

  case 203:
#line 731 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = 0; }
#line 3548 "y.tab.c" /* yacc.c:1646  */
    break;

  case 204:
#line 733 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3554 "y.tab.c" /* yacc.c:1646  */
    break;

  case 205:
#line 737 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = 0; }
#line 3560 "y.tab.c" /* yacc.c:1646  */
    break;

  case 206:
#line 739 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3566 "y.tab.c" /* yacc.c:1646  */
    break;

  case 207:
#line 741 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3572 "y.tab.c" /* yacc.c:1646  */
    break;

  case 208:
#line 743 "xi-grammar.y" /* yacc.c:1646  */
    {
		  XStr typeStr;
		  (yyvsp[0].ntype)->print(typeStr);
		  char *tmp = strdup(typeStr.get_string());
		  (yyval.strval) = tmp;
		}
#line 3583 "y.tab.c" /* yacc.c:1646  */
    break;

  case 209:
#line 752 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TTypeEllipsis(new NamedEllipsisType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3589 "y.tab.c" /* yacc.c:1646  */
    break;

  case 210:
#line 754 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TTypeEllipsis(new NamedEllipsisType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3595 "y.tab.c" /* yacc.c:1646  */
    break;

  case 211:
#line 756 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TType(new NamedType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3601 "y.tab.c" /* yacc.c:1646  */
    break;

  case 212:
#line 758 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TType(new NamedType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3607 "y.tab.c" /* yacc.c:1646  */
    break;

  case 213:
#line 760 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TFunc((yyvsp[-1].ftype), (yyvsp[0].strval)); }
#line 3613 "y.tab.c" /* yacc.c:1646  */
    break;

  case 214:
#line 762 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TName((yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].strval)); }
#line 3619 "y.tab.c" /* yacc.c:1646  */
    break;

  case 215:
#line 766 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = new TVarList((yyvsp[0].tvar)); }
#line 3625 "y.tab.c" /* yacc.c:1646  */
    break;

  case 216:
#line 768 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = new TVarList((yyvsp[-2].tvar), (yyvsp[0].tvarlist)); }
#line 3631 "y.tab.c" /* yacc.c:1646  */
    break;

  case 217:
#line 772 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = (yyvsp[-1].tvarlist); }
#line 3637 "y.tab.c" /* yacc.c:1646  */
    break;

  case 218:
#line 776 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3643 "y.tab.c" /* yacc.c:1646  */
    break;

  case 219:
#line 778 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3649 "y.tab.c" /* yacc.c:1646  */
    break;

  case 220:
#line 780 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3655 "y.tab.c" /* yacc.c:1646  */
    break;

  case 221:
#line 782 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3661 "y.tab.c" /* yacc.c:1646  */
    break;

  case 222:
#line 784 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].message)); (yyvsp[0].message)->setTemplate((yyval.templat)); }
#line 3667 "y.tab.c" /* yacc.c:1646  */
    break;

  case 223:
#line 788 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = 0; }
#line 3673 "y.tab.c" /* yacc.c:1646  */
    break;

  case 224:
#line 790 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = (yyvsp[-2].mbrlist); }
#line 3679 "y.tab.c" /* yacc.c:1646  */
    break;

  case 225:
#line 794 "xi-grammar.y" /* yacc.c:1646  */
    { 
                  if (!connectEntries.empty()) {
                    (yyval.mbrlist) = new AstChildren<Member>(connectEntries);
		  } else {
		    (yyval.mbrlist) = 0; 
                  }
		}
#line 3691 "y.tab.c" /* yacc.c:1646  */
    break;

  case 226:
#line 802 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = new AstChildren<Member>(-1, (yyvsp[-1].member), (yyvsp[0].mbrlist)); }
#line 3697 "y.tab.c" /* yacc.c:1646  */
    break;

  case 227:
#line 806 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].readonly); }
#line 3703 "y.tab.c" /* yacc.c:1646  */
    break;

  case 228:
#line 808 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].readonly); }
#line 3709 "y.tab.c" /* yacc.c:1646  */
    break;

  case 230:
#line 811 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].member); }
#line 3715 "y.tab.c" /* yacc.c:1646  */
    break;

  case 231:
#line 813 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].pupable); }
#line 3721 "y.tab.c" /* yacc.c:1646  */
    break;

  case 232:
#line 815 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].includeFile); }
#line 3727 "y.tab.c" /* yacc.c:1646  */
    break;

  case 233:
#line 817 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new ClassDeclaration(lineno,(yyvsp[0].strval)); }
#line 3733 "y.tab.c" /* yacc.c:1646  */
    break;

  case 234:
#line 821 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1); }
#line 3739 "y.tab.c" /* yacc.c:1646  */
    break;

  case 235:
#line 823 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1); }
#line 3745 "y.tab.c" /* yacc.c:1646  */
    break;

  case 236:
#line 825 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[-6].strval)) + '<' +
					    ((yyvsp[-4].tparlist))->to_string() + '>').c_str()),
				    1);
		}
#line 3755 "y.tab.c" /* yacc.c:1646  */
    break;

  case 237:
#line 831 "xi-grammar.y" /* yacc.c:1646  */
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[-2]).first_column, (yylsp[-2]).last_column, (yylsp[-2]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1);
		}
#line 3765 "y.tab.c" /* yacc.c:1646  */
    break;

  case 238:
#line 837 "xi-grammar.y" /* yacc.c:1646  */
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[-5]).first_column, (yylsp[-5]).last_column, (yylsp[-5]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1);
		}
#line 3775 "y.tab.c" /* yacc.c:1646  */
    break;

  case 239:
#line 846 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 0); }
#line 3781 "y.tab.c" /* yacc.c:1646  */
    break;

  case 240:
#line 848 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 0); }
#line 3787 "y.tab.c" /* yacc.c:1646  */
    break;

  case 241:
#line 850 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[-6].strval)) + '<' +
					    ((yyvsp[-4].tparlist))->to_string() + '>').c_str()),
				    0);
		}
#line 3797 "y.tab.c" /* yacc.c:1646  */
    break;

  case 242:
#line 856 "xi-grammar.y" /* yacc.c:1646  */
    {
                  InitCall* rtn = new InitCall(lineno, (yyvsp[-3].strval), 0);
                  rtn->setAccel();
                  (yyval.member) = rtn;
		}
#line 3807 "y.tab.c" /* yacc.c:1646  */
    break;

  case 243:
#line 864 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[0].ntype),0); }
#line 3813 "y.tab.c" /* yacc.c:1646  */
    break;

  case 244:
#line 866 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[-2].ntype),(yyvsp[0].pupable)); }
#line 3819 "y.tab.c" /* yacc.c:1646  */
    break;

  case 245:
#line 869 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.includeFile) = new IncludeFile(lineno,(yyvsp[0].strval)); }
#line 3825 "y.tab.c" /* yacc.c:1646  */
    break;

  case 246:
#line 873 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].member); }
#line 3831 "y.tab.c" /* yacc.c:1646  */
    break;

  case 247:
#line 877 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].entry); }
#line 3837 "y.tab.c" /* yacc.c:1646  */
    break;

  case 248:
#line 879 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyvsp[0].entry)->tspec = (yyvsp[-1].tvarlist);
                  (yyval.member) = (yyvsp[0].entry);
                }
#line 3846 "y.tab.c" /* yacc.c:1646  */
    break;

  case 249:
#line 884 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[-1].member); }
#line 3852 "y.tab.c" /* yacc.c:1646  */
    break;

  case 250:
#line 886 "xi-grammar.y" /* yacc.c:1646  */
    {
          ERROR("invalid SDAG member",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 3862 "y.tab.c" /* yacc.c:1646  */
    break;

  case 251:
#line 894 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3868 "y.tab.c" /* yacc.c:1646  */
    break;

  case 252:
#line 896 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3874 "y.tab.c" /* yacc.c:1646  */
    break;

  case 253:
#line 898 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3880 "y.tab.c" /* yacc.c:1646  */
    break;

  case 254:
#line 900 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3886 "y.tab.c" /* yacc.c:1646  */
    break;

  case 255:
#line 902 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3892 "y.tab.c" /* yacc.c:1646  */
    break;

  case 256:
#line 904 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3898 "y.tab.c" /* yacc.c:1646  */
    break;

  case 257:
#line 906 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3904 "y.tab.c" /* yacc.c:1646  */
    break;

  case 258:
#line 908 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3910 "y.tab.c" /* yacc.c:1646  */
    break;

  case 259:
#line 910 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3916 "y.tab.c" /* yacc.c:1646  */
    break;

  case 260:
#line 912 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3922 "y.tab.c" /* yacc.c:1646  */
    break;

  case 261:
#line 914 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3928 "y.tab.c" /* yacc.c:1646  */
    break;

  case 262:
#line 917 "xi-grammar.y" /* yacc.c:1646  */
    { 
                  (yyval.entry) = new Entry(lineno, (yyvsp[-5].attr), (yyvsp[-4].type), (yyvsp[-3].strval), (yyvsp[-2].plist), (yyvsp[-1].val), (yyvsp[0].sentry), (const char *) NULL, (yylsp[-6]).first_line, (yyloc).last_line);
		  if ((yyvsp[0].sentry) != 0) { 
		    (yyvsp[0].sentry)->con1 = new SdagConstruct(SIDENT, (yyvsp[-3].strval));
                    (yyvsp[0].sentry)->setEntry((yyval.entry));
                    (yyvsp[0].sentry)->param = new ParamList((yyvsp[-2].plist));
                  }
                  firstRdma = true;
		}
#line 3942 "y.tab.c" /* yacc.c:1646  */
    break;

  case 263:
#line 927 "xi-grammar.y" /* yacc.c:1646  */
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
#line 3963 "y.tab.c" /* yacc.c:1646  */
    break;

  case 264:
#line 944 "xi-grammar.y" /* yacc.c:1646  */
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
#line 3982 "y.tab.c" /* yacc.c:1646  */
    break;

  case 265:
#line 961 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.accelBlock) = new AccelBlock(lineno, new XStr((yyvsp[-2].strval))); }
#line 3988 "y.tab.c" /* yacc.c:1646  */
    break;

  case 266:
#line 963 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.accelBlock) = new AccelBlock(lineno, NULL); }
#line 3994 "y.tab.c" /* yacc.c:1646  */
    break;

  case 267:
#line 967 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 4000 "y.tab.c" /* yacc.c:1646  */
    break;

  case 268:
#line 971 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attr) = 0; }
#line 4006 "y.tab.c" /* yacc.c:1646  */
    break;

  case 269:
#line 973 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attr) = (yyvsp[-1].attr); }
#line 4012 "y.tab.c" /* yacc.c:1646  */
    break;

  case 270:
#line 975 "xi-grammar.y" /* yacc.c:1646  */
    { ERROR("invalid entry method attribute list",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
#line 4021 "y.tab.c" /* yacc.c:1646  */
    break;

  case 271:
#line 981 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attrarg) = new Attribute::Argument((yyvsp[-2].strval), atoi((yyvsp[0].strval))); }
#line 4027 "y.tab.c" /* yacc.c:1646  */
    break;

  case 272:
#line 985 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attrarg) = (yyvsp[0].attrarg); }
#line 4033 "y.tab.c" /* yacc.c:1646  */
    break;

  case 273:
#line 986 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attrarg) = (yyvsp[-2].attrarg); (yyvsp[-2].attrarg)->next = (yyvsp[0].attrarg); }
#line 4039 "y.tab.c" /* yacc.c:1646  */
    break;

  case 274:
#line 990 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attr) = new Attribute((yyvsp[0].intval));           }
#line 4045 "y.tab.c" /* yacc.c:1646  */
    break;

  case 275:
#line 991 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attr) = new Attribute((yyvsp[-3].intval), (yyvsp[-1].attrarg));       }
#line 4051 "y.tab.c" /* yacc.c:1646  */
    break;

  case 276:
#line 992 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attr) = new Attribute((yyvsp[-2].intval), NULL, (yyvsp[0].attr)); }
#line 4057 "y.tab.c" /* yacc.c:1646  */
    break;

  case 277:
#line 993 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attr) = new Attribute((yyvsp[-5].intval), (yyvsp[-3].attrarg), (yyvsp[0].attr));   }
#line 4063 "y.tab.c" /* yacc.c:1646  */
    break;

  case 278:
#line 997 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = STHREADED; }
#line 4069 "y.tab.c" /* yacc.c:1646  */
    break;

  case 279:
#line 999 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SWHENIDLE; }
#line 4075 "y.tab.c" /* yacc.c:1646  */
    break;

  case 280:
#line 1001 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SSYNC; }
#line 4081 "y.tab.c" /* yacc.c:1646  */
    break;

  case 281:
#line 1003 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SIGET; }
#line 4087 "y.tab.c" /* yacc.c:1646  */
    break;

  case 282:
#line 1005 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SLOCKED; }
#line 4093 "y.tab.c" /* yacc.c:1646  */
    break;

  case 283:
#line 1007 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SCREATEHERE; }
#line 4099 "y.tab.c" /* yacc.c:1646  */
    break;

  case 284:
#line 1009 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SCREATEHOME; }
#line 4105 "y.tab.c" /* yacc.c:1646  */
    break;

  case 285:
#line 1011 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SNOKEEP; }
#line 4111 "y.tab.c" /* yacc.c:1646  */
    break;

  case 286:
#line 1013 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SNOTRACE; }
#line 4117 "y.tab.c" /* yacc.c:1646  */
    break;

  case 287:
#line 1015 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SAPPWORK; }
#line 4123 "y.tab.c" /* yacc.c:1646  */
    break;

  case 288:
#line 1017 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SIMMEDIATE; }
#line 4129 "y.tab.c" /* yacc.c:1646  */
    break;

  case 289:
#line 1019 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SSKIPSCHED; }
#line 4135 "y.tab.c" /* yacc.c:1646  */
    break;

  case 290:
#line 1021 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SINLINE; }
#line 4141 "y.tab.c" /* yacc.c:1646  */
    break;

  case 291:
#line 1023 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SLOCAL; }
#line 4147 "y.tab.c" /* yacc.c:1646  */
    break;

  case 292:
#line 1025 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SPYTHON; }
#line 4153 "y.tab.c" /* yacc.c:1646  */
    break;

  case 293:
#line 1027 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SMEM; }
#line 4159 "y.tab.c" /* yacc.c:1646  */
    break;

  case 294:
#line 1029 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SREDUCE; }
#line 4165 "y.tab.c" /* yacc.c:1646  */
    break;

  case 295:
#line 1031 "xi-grammar.y" /* yacc.c:1646  */
    {
        (yyval.intval) = SAGGREGATE;
    }
#line 4173 "y.tab.c" /* yacc.c:1646  */
    break;

  case 296:
#line 1035 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("invalid entry method attribute",
		        (yylsp[0]).first_column, (yylsp[0]).last_column);
		  yyclearin;
		  yyerrok;
		}
#line 4184 "y.tab.c" /* yacc.c:1646  */
    break;

  case 297:
#line 1044 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4190 "y.tab.c" /* yacc.c:1646  */
    break;

  case 298:
#line 1046 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4196 "y.tab.c" /* yacc.c:1646  */
    break;

  case 299:
#line 1048 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4202 "y.tab.c" /* yacc.c:1646  */
    break;

  case 300:
#line 1052 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = ""; }
#line 4208 "y.tab.c" /* yacc.c:1646  */
    break;

  case 301:
#line 1054 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4214 "y.tab.c" /* yacc.c:1646  */
    break;

  case 302:
#line 1056 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s, %s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4224 "y.tab.c" /* yacc.c:1646  */
    break;

  case 303:
#line 1064 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = ""; }
#line 4230 "y.tab.c" /* yacc.c:1646  */
    break;

  case 304:
#line 1066 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4236 "y.tab.c" /* yacc.c:1646  */
    break;

  case 305:
#line 1068 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s[%s]%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4246 "y.tab.c" /* yacc.c:1646  */
    break;

  case 306:
#line 1074 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s{%s}%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4256 "y.tab.c" /* yacc.c:1646  */
    break;

  case 307:
#line 1080 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s(%s)%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4266 "y.tab.c" /* yacc.c:1646  */
    break;

  case 308:
#line 1086 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"(%s)%s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4276 "y.tab.c" /* yacc.c:1646  */
    break;

  case 309:
#line 1094 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval));
		}
#line 4285 "y.tab.c" /* yacc.c:1646  */
    break;

  case 310:
#line 1101 "xi-grammar.y" /* yacc.c:1646  */
    { 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			(yyval.intval) = 0;
		}
#line 4295 "y.tab.c" /* yacc.c:1646  */
    break;

  case 311:
#line 1109 "xi-grammar.y" /* yacc.c:1646  */
    { 
			in_braces=0;
			(yyval.intval) = 0;
		}
#line 4304 "y.tab.c" /* yacc.c:1646  */
    break;

  case 312:
#line 1116 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[0].type));}
#line 4310 "y.tab.c" /* yacc.c:1646  */
    break;

  case 313:
#line 1118 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval)); (yyval.pname)->setConditional((yyvsp[0].intval)); }
#line 4316 "y.tab.c" /* yacc.c:1646  */
    break;

  case 314:
#line 1120 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[-3].type),(yyvsp[-2].strval),0,(yyvsp[0].val));}
#line 4322 "y.tab.c" /* yacc.c:1646  */
    break;

  case 315:
#line 1122 "xi-grammar.y" /* yacc.c:1646  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
		}
#line 4331 "y.tab.c" /* yacc.c:1646  */
    break;

  case 316:
#line 1127 "xi-grammar.y" /* yacc.c:1646  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
			(yyval.pname)->setRdma(CMK_ZC_P2P_SEND_MSG);
			if(firstRdma) {
				(yyval.pname)->setFirstRdma(true);
				firstRdma = false;
			}
		}
#line 4345 "y.tab.c" /* yacc.c:1646  */
    break;

  case 317:
#line 1137 "xi-grammar.y" /* yacc.c:1646  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
			(yyval.pname)->setRdma(CMK_ZC_P2P_RECV_MSG);
			if(firstRdma) {
				(yyval.pname)->setFirstRdma(true);
				firstRdma = false;
			}
		}
#line 4359 "y.tab.c" /* yacc.c:1646  */
    break;

  case 318:
#line 1148 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READONLY; }
#line 4365 "y.tab.c" /* yacc.c:1646  */
    break;

  case 319:
#line 1149 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READWRITE; }
#line 4371 "y.tab.c" /* yacc.c:1646  */
    break;

  case 320:
#line 1150 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_WRITEONLY; }
#line 4377 "y.tab.c" /* yacc.c:1646  */
    break;

  case 321:
#line 1153 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr((yyvsp[0].strval)); }
#line 4383 "y.tab.c" /* yacc.c:1646  */
    break;

  case 322:
#line 1154 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "->" << (yyvsp[0].strval); }
#line 4389 "y.tab.c" /* yacc.c:1646  */
    break;

  case 323:
#line 1155 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[-2].xstrptr)) << "." << (yyvsp[0].strval); }
#line 4395 "y.tab.c" /* yacc.c:1646  */
    break;

  case 324:
#line 1157 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << *((yyvsp[-1].xstrptr)) << "]";
                  delete (yyvsp[-3].xstrptr);
                  delete (yyvsp[-1].xstrptr);
                }
#line 4406 "y.tab.c" /* yacc.c:1646  */
    break;

  case 325:
#line 1164 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << (yyvsp[-1].strval) << "]";
                  delete (yyvsp[-3].xstrptr);
                }
#line 4416 "y.tab.c" /* yacc.c:1646  */
    break;

  case 326:
#line 1170 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "(" << *((yyvsp[-1].xstrptr)) << ")";
                  delete (yyvsp[-3].xstrptr);
                  delete (yyvsp[-1].xstrptr);
                }
#line 4427 "y.tab.c" /* yacc.c:1646  */
    break;

  case 327:
#line 1179 "xi-grammar.y" /* yacc.c:1646  */
    {
                  in_bracket = 0;
                  (yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName(), (yyvsp[-1].strval));
                }
#line 4436 "y.tab.c" /* yacc.c:1646  */
    break;

  case 328:
#line 1186 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[-6].intval));
                }
#line 4446 "y.tab.c" /* yacc.c:1646  */
    break;

  case 329:
#line 1192 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType(Parameter::ACCEL_BUFFER_TYPE_READWRITE);
		}
#line 4456 "y.tab.c" /* yacc.c:1646  */
    break;

  case 330:
#line 1198 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.pname) = (yyvsp[-3].pname);
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[-5].intval));
		}
#line 4466 "y.tab.c" /* yacc.c:1646  */
    break;

  case 331:
#line 1206 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 4472 "y.tab.c" /* yacc.c:1646  */
    break;

  case 332:
#line 1208 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 4478 "y.tab.c" /* yacc.c:1646  */
    break;

  case 333:
#line 1212 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 4484 "y.tab.c" /* yacc.c:1646  */
    break;

  case 334:
#line 1214 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 4490 "y.tab.c" /* yacc.c:1646  */
    break;

  case 335:
#line 1218 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = (yyvsp[-1].plist); }
#line 4496 "y.tab.c" /* yacc.c:1646  */
    break;

  case 336:
#line 1220 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList(new Parameter(0, new BuiltinType("void"))); }
#line 4502 "y.tab.c" /* yacc.c:1646  */
    break;

  case 337:
#line 1224 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = (yyvsp[-1].plist); }
#line 4508 "y.tab.c" /* yacc.c:1646  */
    break;

  case 338:
#line 1226 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = 0; }
#line 4514 "y.tab.c" /* yacc.c:1646  */
    break;

  case 339:
#line 1230 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = 0; }
#line 4520 "y.tab.c" /* yacc.c:1646  */
    break;

  case 340:
#line 1232 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4526 "y.tab.c" /* yacc.c:1646  */
    break;

  case 341:
#line 1236 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = 0; }
#line 4532 "y.tab.c" /* yacc.c:1646  */
    break;

  case 342:
#line 1238 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[0].sc)); }
#line 4538 "y.tab.c" /* yacc.c:1646  */
    break;

  case 343:
#line 1240 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[-2].slist)); }
#line 4544 "y.tab.c" /* yacc.c:1646  */
    break;

  case 344:
#line 1244 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.slist) = new SListConstruct((yyvsp[0].sc)); }
#line 4550 "y.tab.c" /* yacc.c:1646  */
    break;

  case 345:
#line 1246 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.slist) = new SListConstruct((yyvsp[-1].sc), (yyvsp[0].slist));  }
#line 4556 "y.tab.c" /* yacc.c:1646  */
    break;

  case 346:
#line 1250 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.olist) = new OListConstruct((yyvsp[0].sc)); }
#line 4562 "y.tab.c" /* yacc.c:1646  */
    break;

  case 347:
#line 1252 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.olist) = new OListConstruct((yyvsp[-1].sc), (yyvsp[0].slist)); }
#line 4568 "y.tab.c" /* yacc.c:1646  */
    break;

  case 348:
#line 1256 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.clist) = new CaseListConstruct((yyvsp[0].when)); }
#line 4574 "y.tab.c" /* yacc.c:1646  */
    break;

  case 349:
#line 1258 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.clist) = new CaseListConstruct((yyvsp[-1].when), (yyvsp[0].clist)); }
#line 4580 "y.tab.c" /* yacc.c:1646  */
    break;

  case 350:
#line 1260 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("case blocks can only contain when clauses",
		        (yylsp[0]).first_column, (yylsp[0]).last_column);
		  (yyval.clist) = 0;
		}
#line 4590 "y.tab.c" /* yacc.c:1646  */
    break;

  case 351:
#line 1268 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4596 "y.tab.c" /* yacc.c:1646  */
    break;

  case 352:
#line 1270 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = 0; }
#line 4602 "y.tab.c" /* yacc.c:1646  */
    break;

  case 353:
#line 1274 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-2].entrylist), 0); }
#line 4608 "y.tab.c" /* yacc.c:1646  */
    break;

  case 354:
#line 1276 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-1].entrylist), (yyvsp[0].sc)); }
#line 4614 "y.tab.c" /* yacc.c:1646  */
    break;

  case 355:
#line 1278 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-3].entrylist), (yyvsp[-1].slist)); }
#line 4620 "y.tab.c" /* yacc.c:1646  */
    break;

  case 356:
#line 1282 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4626 "y.tab.c" /* yacc.c:1646  */
    break;

  case 357:
#line 1284 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4632 "y.tab.c" /* yacc.c:1646  */
    break;

  case 358:
#line 1286 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4638 "y.tab.c" /* yacc.c:1646  */
    break;

  case 359:
#line 1288 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4644 "y.tab.c" /* yacc.c:1646  */
    break;

  case 360:
#line 1290 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4650 "y.tab.c" /* yacc.c:1646  */
    break;

  case 361:
#line 1292 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4656 "y.tab.c" /* yacc.c:1646  */
    break;

  case 362:
#line 1294 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4662 "y.tab.c" /* yacc.c:1646  */
    break;

  case 363:
#line 1296 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4668 "y.tab.c" /* yacc.c:1646  */
    break;

  case 364:
#line 1298 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4674 "y.tab.c" /* yacc.c:1646  */
    break;

  case 365:
#line 1300 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4680 "y.tab.c" /* yacc.c:1646  */
    break;

  case 366:
#line 1302 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4686 "y.tab.c" /* yacc.c:1646  */
    break;

  case 367:
#line 1304 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4692 "y.tab.c" /* yacc.c:1646  */
    break;

  case 368:
#line 1308 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new SerialConstruct((yyvsp[-2].strval), (yyvsp[-4].strval), (yylsp[-3]).first_line); }
#line 4698 "y.tab.c" /* yacc.c:1646  */
    break;

  case 369:
#line 1310 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new OverlapConstruct((yyvsp[-1].olist)); }
#line 4704 "y.tab.c" /* yacc.c:1646  */
    break;

  case 370:
#line 1312 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = (yyvsp[0].when); }
#line 4710 "y.tab.c" /* yacc.c:1646  */
    break;

  case 371:
#line 1314 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new CaseConstruct((yyvsp[-1].clist)); }
#line 4716 "y.tab.c" /* yacc.c:1646  */
    break;

  case 372:
#line 1316 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForConstruct((yyvsp[-8].intexpr), (yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4722 "y.tab.c" /* yacc.c:1646  */
    break;

  case 373:
#line 1318 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForConstruct((yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4728 "y.tab.c" /* yacc.c:1646  */
    break;

  case 374:
#line 1320 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-9].strval)), (yyvsp[-6].intexpr),
		             (yyvsp[-4].intexpr), (yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4735 "y.tab.c" /* yacc.c:1646  */
    break;

  case 375:
#line 1323 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-11].strval)), (yyvsp[-8].intexpr),
		             (yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4742 "y.tab.c" /* yacc.c:1646  */
    break;

  case 376:
#line 1326 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new IfConstruct((yyvsp[-3].intexpr), (yyvsp[-1].sc), (yyvsp[0].sc)); }
#line 4748 "y.tab.c" /* yacc.c:1646  */
    break;

  case 377:
#line 1328 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new IfConstruct((yyvsp[-5].intexpr), (yyvsp[-2].slist), (yyvsp[0].sc)); }
#line 4754 "y.tab.c" /* yacc.c:1646  */
    break;

  case 378:
#line 1330 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new WhileConstruct((yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4760 "y.tab.c" /* yacc.c:1646  */
    break;

  case 379:
#line 1332 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new WhileConstruct((yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4766 "y.tab.c" /* yacc.c:1646  */
    break;

  case 380:
#line 1334 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new SerialConstruct((yyvsp[-2].strval), NULL, (yyloc).first_line); }
#line 4772 "y.tab.c" /* yacc.c:1646  */
    break;

  case 381:
#line 1336 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("unknown SDAG construct or malformed entry method declaration.\n"
		        "You may have forgotten to terminate a previous entry method declaration with a"
		        " semicolon or forgotten to mark a block of sequential SDAG code as 'serial'",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
#line 4784 "y.tab.c" /* yacc.c:1646  */
    break;

  case 382:
#line 1346 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = 0; }
#line 4790 "y.tab.c" /* yacc.c:1646  */
    break;

  case 383:
#line 1348 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ElseConstruct((yyvsp[0].sc)); }
#line 4796 "y.tab.c" /* yacc.c:1646  */
    break;

  case 384:
#line 1350 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ElseConstruct((yyvsp[-1].slist)); }
#line 4802 "y.tab.c" /* yacc.c:1646  */
    break;

  case 385:
#line 1354 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intexpr) = new IntExprConstruct((yyvsp[0].strval)); }
#line 4808 "y.tab.c" /* yacc.c:1646  */
    break;

  case 386:
#line 1358 "xi-grammar.y" /* yacc.c:1646  */
    { in_int_expr = 0; (yyval.intval) = 0; }
#line 4814 "y.tab.c" /* yacc.c:1646  */
    break;

  case 387:
#line 1362 "xi-grammar.y" /* yacc.c:1646  */
    { in_int_expr = 1; (yyval.intval) = 0; }
#line 4820 "y.tab.c" /* yacc.c:1646  */
    break;

  case 388:
#line 1366 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.entry) = new Entry(lineno, NULL, 0, (yyvsp[-1].strval), (yyvsp[0].plist), 0, 0, 0, (yyloc).first_line, (yyloc).last_line);
		  firstRdma = true;
		}
#line 4829 "y.tab.c" /* yacc.c:1646  */
    break;

  case 389:
#line 1371 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.entry) = new Entry(lineno, NULL, 0, (yyvsp[-4].strval), (yyvsp[0].plist), 0, 0, (yyvsp[-2].strval), (yyloc).first_line, (yyloc).last_line);
		  firstRdma = true;
		}
#line 4838 "y.tab.c" /* yacc.c:1646  */
    break;

  case 390:
#line 1378 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.entrylist) = new EntryList((yyvsp[0].entry)); }
#line 4844 "y.tab.c" /* yacc.c:1646  */
    break;

  case 391:
#line 1380 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.entrylist) = new EntryList((yyvsp[-2].entry),(yyvsp[0].entrylist)); }
#line 4850 "y.tab.c" /* yacc.c:1646  */
    break;

  case 392:
#line 1384 "xi-grammar.y" /* yacc.c:1646  */
    { in_bracket=1; }
#line 4856 "y.tab.c" /* yacc.c:1646  */
    break;

  case 393:
#line 1387 "xi-grammar.y" /* yacc.c:1646  */
    { in_bracket=0; }
#line 4862 "y.tab.c" /* yacc.c:1646  */
    break;

  case 394:
#line 1391 "xi-grammar.y" /* yacc.c:1646  */
    { if (!macroDefined((yyvsp[0].strval), 1)) in_comment = 1; }
#line 4868 "y.tab.c" /* yacc.c:1646  */
    break;

  case 395:
#line 1395 "xi-grammar.y" /* yacc.c:1646  */
    { if (!macroDefined((yyvsp[0].strval), 0)) in_comment = 1; }
#line 4874 "y.tab.c" /* yacc.c:1646  */
    break;


#line 4878 "y.tab.c" /* yacc.c:1646  */
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
#line 1398 "xi-grammar.y" /* yacc.c:1906  */


void yyerror(const char *s) 
{
	fprintf(stderr, "[PARSE-ERROR] Unexpected/missing token at line %d. Current token being parsed: '%s'.\n", lineno, yytext);
}
