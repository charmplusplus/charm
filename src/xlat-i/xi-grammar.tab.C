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
#define YYFINAL  61
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   1577

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  97
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  119
/* YYNRULES -- Number of rules.  */
#define YYNRULES  404
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  797

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
     269,   270,   271,   272,   273,   274,   277,   278,   279,   280,
     281,   282,   283,   284,   285,   286,   287,   289,   291,   292,
     295,   296,   297,   298,   302,   304,   311,   319,   323,   330,
     332,   337,   338,   342,   344,   346,   348,   350,   364,   366,
     368,   370,   376,   378,   380,   382,   384,   386,   388,   390,
     392,   394,   402,   404,   406,   410,   412,   417,   418,   423,
     424,   428,   430,   432,   434,   436,   438,   440,   442,   444,
     446,   448,   450,   452,   454,   456,   458,   460,   462,   464,
     466,   470,   471,   476,   484,   486,   490,   494,   496,   500,
     504,   506,   508,   510,   512,   514,   518,   520,   522,   524,
     526,   530,   532,   534,   536,   538,   540,   544,   546,   548,
     550,   552,   554,   558,   562,   567,   568,   572,   576,   581,
     582,   587,   588,   598,   600,   604,   606,   611,   612,   616,
     618,   623,   624,   628,   633,   634,   638,   640,   644,   646,
     651,   652,   656,   657,   660,   664,   666,   670,   672,   674,
     679,   680,   684,   686,   690,   692,   696,   700,   704,   710,
     714,   716,   720,   722,   726,   730,   734,   738,   740,   745,
     746,   751,   752,   754,   756,   765,   767,   769,   771,   773,
     775,   779,   781,   785,   789,   791,   793,   795,   797,   801,
     803,   808,   815,   819,   821,   823,   824,   826,   828,   830,
     834,   836,   838,   844,   850,   859,   861,   863,   869,   877,
     879,   882,   886,   890,   892,   897,   899,   907,   909,   911,
     913,   915,   917,   919,   921,   923,   925,   927,   930,   941,
     959,   977,   979,   983,   988,   989,   991,   998,  1002,  1003,
    1007,  1008,  1009,  1010,  1013,  1015,  1017,  1019,  1021,  1023,
    1025,  1027,  1029,  1031,  1033,  1035,  1037,  1039,  1041,  1043,
    1045,  1047,  1049,  1051,  1055,  1064,  1066,  1068,  1073,  1074,
    1076,  1086,  1087,  1089,  1096,  1103,  1110,  1119,  1126,  1134,
    1141,  1143,  1145,  1147,  1152,  1162,  1172,  1184,  1185,  1186,
    1189,  1190,  1191,  1192,  1199,  1205,  1214,  1221,  1227,  1233,
    1241,  1243,  1247,  1249,  1253,  1255,  1259,  1261,  1266,  1267,
    1271,  1273,  1275,  1279,  1281,  1285,  1287,  1291,  1293,  1295,
    1303,  1306,  1309,  1311,  1313,  1317,  1319,  1321,  1323,  1325,
    1327,  1329,  1331,  1333,  1335,  1337,  1339,  1343,  1345,  1347,
    1349,  1351,  1353,  1355,  1358,  1361,  1363,  1365,  1367,  1369,
    1371,  1382,  1383,  1385,  1389,  1393,  1397,  1401,  1407,  1415,
    1417,  1421,  1424,  1428,  1432
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

#define YYPACT_NINF -601

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-601)))

#define YYTABLE_NINF -356

#define yytable_value_is_error(Yytable_value) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
     169,  1345,  1345,    49,  -601,   169,  -601,  -601,  -601,  -601,
    -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,
    -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,
    -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,
    -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,
    -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,   182,
     182,  -601,  -601,  -601,   921,   -29,  -601,  -601,  -601,    10,
    1345,   230,  1345,  1345,   256,  1025,    -6,  1001,   921,  -601,
    -601,  -601,  -601,   174,   -20,    74,  -601,    55,  -601,  -601,
    -601,   -29,   -21,   590,   102,   102,    52,    -5,    59,    59,
      59,    59,    67,    88,  1345,   128,   107,   921,  -601,  -601,
    -601,  -601,  -601,  -601,  -601,  -601,   370,  -601,  -601,  -601,
    -601,   123,  -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,
    -601,  -601,  -601,   -29,  -601,  -601,  -601,  1208,  1454,   921,
      55,   137,   144,   -21,   147,  1132,  -601,  1469,  -601,   269,
     173,  -601,  -601,  -601,   347,    74,   195,  -601,  -601,   194,
     199,   239,  -601,   114,    74,  -601,    74,    74,   206,    74,
     263,  -601,   180,  1345,  1345,  1345,  1345,   191,   231,   261,
     210,  1345,  -601,  -601,  -601,  1388,   271,    59,    59,    59,
      59,   231,    88,  -601,  -601,  -601,  -601,  -601,   -29,  -601,
    -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,
    -601,  -601,  -601,  -601,  -601,  -601,  -601,   296,  -601,  -601,
    -601,   272,   260,  1454,   194,   199,   239,   109,  -601,    -5,
     285,   152,   -21,   308,   -21,   282,  -601,   123,   287,    73,
    -601,   310,  -601,  -601,  -601,   234,   325,  -601,   195,  1436,
    -601,  -601,  -601,  -601,  -601,   306,   215,   309,   -34,    57,
     288,   317,   311,    -5,  -601,  -601,   323,   343,   316,   358,
     358,   358,   358,  -601,  1345,   348,   356,   353,   -16,  1345,
     390,  1345,  -601,  -601,   360,   378,   382,   831,    78,   185,
    1345,   381,   387,   123,  1345,  1345,  1345,  1345,  1345,  1345,
    -601,  -601,  -601,  1208,  1345,   434,  -601,   223,   384,  1345,
    -601,  -601,  -601,   393,   399,   388,   391,   -21,   -29,    74,
    -601,  -601,   422,  -601,  -601,  -601,  -601,   406,  -601,   407,
    -601,  1345,   401,   403,   404,  -601,   408,  -601,   -21,   102,
    1436,   102,   102,  1436,   102,  -601,  -601,   180,  -601,    -5,
     346,   346,   346,   346,   410,  -601,   390,  -601,   358,   358,
    -601,   210,    13,   415,   405,   189,   416,   188,  -601,   424,
    1388,  -601,  -601,   358,   358,   358,   358,   358,   352,  -601,
     435,   438,   437,   432,   440,   441,   316,   -21,   308,   -21,
     -21,  -601,   -34,  -601,  1436,  -601,   439,   443,   444,  -601,
    -601,   436,  -601,   447,   455,   456,    74,   460,   458,  -601,
     464,  -601,   442,   -29,  -601,  -601,  -601,  -601,  -601,  -601,
     346,   346,  -601,  -601,  -601,  1469,    24,   468,   469,  1469,
    -601,  -601,   470,  -601,  -601,  -601,  -601,  -601,   346,   346,
     346,   346,   346,   546,   -29,   504,  1345,   482,   483,   488,
    -601,   492,  -601,  -601,  -601,  -601,  -601,  -601,   493,   491,
    -601,  -601,  -601,  -601,   496,  -601,   131,   497,  -601,    -5,
    -601,   748,   542,   506,   123,   442,  -601,  -601,  -601,  -601,
    1345,  -601,  -601,  1345,  -601,   531,  -601,  -601,  -601,  -601,
    -601,   508,  -601,  -601,  1208,   501,  -601,  1421,  -601,  1020,
    -601,   102,   102,   102,  -601,  1150,  1090,  -601,   123,   -29,
    -601,   507,   405,   405,   123,  -601,  -601,  1469,  1469,  1469,
    -601,  1345,   -21,   509,   512,   513,   524,   525,   526,   523,
     510,   492,  1345,  -601,   527,   123,  -601,  -601,   -29,  1345,
     -21,   -21,   -21,    17,   528,  1020,  -601,  -601,  -601,  -601,
    -601,   558,    75,   492,  -601,   -29,   530,   532,   533,   535,
    -601,   364,  -601,  -601,  -601,  1345,  -601,   536,   539,   536,
     560,   550,   557,   536,   553,   268,   -29,   -21,  -601,  -601,
    -601,   617,  -601,  -601,  -601,  -601,  -601,    55,  -601,   492,
    -601,   -21,   580,   -21,   257,   556,   505,   559,  -601,   568,
     -21,  1494,   570,   419,   147,   569,    75,   572,  -601,   571,
     574,   573,  -601,   -21,   560,   355,  -601,   582,   461,   -21,
     573,   536,   576,   536,   588,   557,   536,   594,   -21,   589,
    1494,  -601,   123,  -601,   123,   618,  -601,   293,   568,   -21,
     536,  -601,   595,   436,  -601,  -601,   597,  -601,  -601,   147,
     634,   -21,   629,   -21,   559,   568,   -21,  1494,   147,  -601,
    -601,  -601,  -601,  -601,  -601,  -601,  -601,  -601,  1345,   608,
     606,   598,   -21,   612,   -21,   268,  -601,   492,  -601,   123,
     268,   639,   614,   602,   573,   613,   -21,   573,   623,   123,
     622,  1469,  1371,  -601,   147,   -21,   616,   625,  -601,  -601,
     628,   652,  -601,   -21,   536,   675,  -601,   147,   898,  -601,
    -601,  1345,  1345,   -21,   630,  -601,  1345,   573,   -21,  -601,
     639,   268,  -601,   633,   -21,   268,  -601,   123,   268,   639,
    -601,   303,   142,   624,  1345,   123,   905,   635,  -601,   637,
     -21,   640,   646,  -601,   648,  -601,  -601,  1345,  1345,  1268,
     632,  1345,  -601,   394,   -29,   268,  -601,   -21,  -601,   573,
     -21,  -601,   639,   236,  -601,   642,   480,  1345,   429,  -601,
     650,   573,   912,   653,  -601,  -601,  -601,  -601,  -601,  -601,
    -601,   919,   268,  -601,   -21,   268,  -601,   655,   573,   656,
    -601,   973,  -601,   268,  -601,   657,  -601
};

  /* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
     Performed when YYTABLE does not specify something else to do.  Zero
     means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       3,     0,     0,     0,     2,     3,    12,    13,    14,    15,
      16,    17,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    31,    32,    36,    37,    38,
      39,    40,    41,    44,    45,    42,    43,    33,    34,    35,
      46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
      56,    57,    58,    11,    59,    60,    61,    62,    63,     0,
       0,     1,     4,     7,     0,    69,    67,    68,    91,     6,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    90,
      88,    89,     8,     0,     0,     0,    64,    74,   403,   404,
     318,   272,   311,     0,   159,   159,   159,     0,   167,   167,
     167,   167,     0,   161,     0,     0,     0,     0,    82,   233,
     234,    76,    83,    84,    85,    86,     0,    87,    75,   236,
     235,     9,   267,   259,   260,   261,   262,   263,   265,   266,
     264,   257,   258,    80,    81,    72,   276,     0,     0,     0,
      73,     0,   312,   311,     0,     0,   120,     0,   101,   102,
     104,   106,   117,   118,     0,     0,    99,   124,   125,   130,
     131,   132,   133,   152,     0,   160,     0,     0,     0,     0,
     249,   237,     0,     0,     0,     0,     0,     0,     0,   174,
       0,     0,   239,   251,   238,     0,     0,   167,   167,   167,
     167,     0,   161,   224,   225,   226,   227,   228,    10,    70,
     304,   284,   285,   286,   287,   288,   296,   297,   298,   303,
     289,   290,   291,   294,   295,   292,   293,   171,   299,   301,
     302,     0,   280,     0,   136,   137,   138,   146,   273,     0,
       0,     0,   311,   308,   311,     0,   319,     0,     0,   134,
     103,   115,   119,   105,   107,   108,   112,   114,    99,    97,
     122,   126,   127,   128,   135,     0,   151,     0,   155,   243,
     240,     0,   245,     0,   178,   179,     0,   169,    99,   190,
     190,   190,   190,   173,     0,     0,   176,     0,     0,     0,
       0,     0,   165,   166,     0,   163,   187,     0,     0,   133,
       0,   221,     0,     9,     0,     0,     0,     0,     0,     0,
     172,   300,   275,     0,     0,   139,   140,   145,     0,     0,
      79,    66,    65,     0,   309,     0,     0,   311,   271,     0,
     116,   109,   110,   113,   123,    93,    94,    95,    98,     0,
      92,     0,   150,     0,     0,   401,   155,   157,   311,   159,
       0,   159,   159,     0,   159,   250,   168,     0,   121,     0,
       0,     0,     0,     0,     0,   199,     0,   175,   190,   190,
     162,     0,   180,     0,   209,    64,     0,     0,   219,   211,
       0,   223,    78,   190,   190,   190,   190,   190,     0,   282,
       0,   278,     0,   144,     0,     0,    99,   311,   308,   311,
     311,   316,   155,   111,     0,   100,     0,     0,     0,   149,
     156,     0,   153,     0,     0,     0,     0,     0,     0,   170,
     192,   191,     0,   229,   194,   195,   196,   197,   198,   177,
       0,     0,   164,   181,   188,     0,   180,     0,     0,     0,
     217,   218,     0,   212,   213,   214,   220,   222,     0,     0,
       0,     0,     0,   180,   207,     0,     0,   281,     0,     0,
     143,     0,   314,   310,   315,   313,   158,    96,     0,     0,
     148,   402,   154,   244,     0,   241,     0,     0,   246,     0,
     256,     0,     0,     0,     0,     0,   252,   253,   200,   201,
       0,   186,   189,     0,   210,     0,   202,   203,   204,   205,
     206,     0,   277,   279,     0,     0,   142,     0,    77,     0,
     147,   159,   159,   159,   193,     0,     0,   254,     9,   255,
     232,   182,   209,   209,     0,   283,   141,     0,     0,     0,
     345,   320,   311,   340,     0,     0,     0,     0,     0,     0,
      64,     0,     0,   230,     0,     0,   215,   216,   208,     0,
     311,   311,   311,   180,     0,     0,   344,   129,   242,   248,
     247,     0,     0,     0,   183,   184,     0,     0,     0,     0,
     317,     0,   321,   323,   341,     0,   390,     0,     0,     0,
       0,     0,   361,     0,     0,     0,   350,   311,   269,   379,
     351,   348,   324,   325,   326,   306,   305,   307,   322,     0,
     396,   311,     0,   311,     0,   399,     0,     0,   360,     0,
     311,     0,     0,     0,     0,     0,     0,     0,   394,     0,
       0,     0,   397,   311,     0,     0,   363,     0,     0,   311,
       0,     0,     0,     0,     0,   361,     0,     0,   311,     0,
     357,   359,     9,   354,     9,     0,   268,     0,     0,   311,
       0,   395,     0,     0,   400,   362,     0,   378,   356,     0,
       0,   311,     0,   311,     0,     0,   311,     0,     0,   380,
     358,   352,   389,   349,   327,   328,   329,   347,     0,     0,
     342,     0,   311,     0,   311,     0,   387,     0,   364,     9,
       0,   391,     0,     0,     0,     0,   311,     0,     0,     9,
       0,     0,     0,   346,     0,   311,     0,     0,   398,   377,
       0,     0,   385,   311,     0,     0,   366,     0,     0,   367,
     376,     0,     0,   311,     0,   343,     0,     0,   311,   388,
     391,     0,   392,     0,   311,     0,   374,     9,     0,   391,
     330,     0,     0,     0,     0,     0,     0,     0,   386,     0,
     311,     0,     0,   365,     0,   372,   338,     0,     0,     0,
       0,     0,   336,     0,   270,     0,   382,   311,   393,     0,
     311,   375,   391,     0,   332,     0,     0,     0,     0,   339,
       0,     0,     0,     0,   373,   335,   334,   333,   331,   337,
     381,     0,     0,   369,   311,     0,   383,     0,     0,     0,
     368,     0,   384,     0,   370,     0,   371
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -601,  -601,   732,  -601,   -55,  -291,    -1,   -62,   667,   683,
      -8,  -601,  -601,  -601,  -293,  -601,  -235,  -601,  -138,   -89,
    -128,  -127,  -117,  -173,   659,   521,  -601,   -87,  -601,  -601,
    -307,  -601,  -601,   -80,   619,   448,  -601,    20,   398,  -601,
    -601,   585,   409,  -601,   237,  -601,  -601,  -364,  -601,  -190,
     335,  -601,  -601,  -601,  -118,  -601,  -601,  -601,  -601,  -601,
    -601,  -347,   453,  -601,   459,   735,  -601,  -129,   337,   738,
    -601,  -601,   567,  -601,  -601,  -601,  -601,   351,  -601,   326,
     362,  -601,   385,  -300,  -601,  -601,   449,   -85,  -493,   -60,
    -572,  -601,  -601,  -571,  -601,  -601,  -366,   155,  -500,  -601,
    -601,   242,  -564,   209,  -590,   224,  -574,  -601,  -506,  -504,
    -565,  -600,  -508,  -601,   258,   277,   244,  -601,  -601
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     3,     4,    75,   413,   199,   268,   156,     5,    66,
      76,    77,    78,   327,   328,   329,   250,   157,   269,   158,
     159,   160,   161,   162,   163,   227,   228,   330,   401,   336,
     337,   109,   110,   166,   181,   284,   285,   173,   266,   301,
     276,   178,   277,   267,   425,   535,   426,   427,   111,   350,
     411,   112,   113,   114,   179,   115,   193,   194,   195,   196,
     197,   430,   368,   291,   292,   472,   117,   414,   473,   474,
     119,   120,   171,   184,   475,   476,   134,   477,    79,   229,
     138,   381,   382,   221,   222,   588,   315,   608,   522,   577,
     237,   523,   669,   731,   714,   670,   524,   671,   498,   638,
     606,   578,   602,   617,   629,   599,   579,   631,   603,   702,
     609,   642,   591,   595,   596,   338,   462,    80,    81
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[] =
{
      59,    60,   372,   379,    65,    65,   164,   144,   170,    87,
     224,   225,   289,   324,    92,   167,   169,   238,   431,    91,
     650,   226,   133,   140,   540,   541,   542,   630,   611,   400,
     423,   552,   634,   348,   423,   620,   270,   271,   272,   633,
     660,   423,   142,   286,    86,   274,   580,   404,   141,    61,
     407,   646,    82,   581,   648,    86,   630,    83,   235,   335,
     279,   593,   481,   139,   155,   600,   198,   688,   143,    84,
     135,    88,    89,   298,   673,   155,   566,   679,   121,   491,
     351,   352,   353,   630,   705,   456,   689,   708,   275,   607,
     616,   618,   165,   248,   612,   224,   225,   424,   290,   186,
     580,   457,   258,   182,   259,   260,   226,   262,  -185,   696,
     560,   697,   561,   651,   254,   653,   700,   736,   656,   174,
     175,   176,   716,   567,   568,   569,   570,   571,   572,   573,
     717,   230,   674,   525,    86,   727,   676,   141,   365,   141,
     309,   358,   165,   359,   681,   168,   339,   313,   618,   316,
     306,   451,   172,   737,   574,   254,    63,   739,   575,   772,
     177,   742,   255,   753,   744,   536,   537,   311,   420,   421,
     366,   781,     1,     2,   170,   136,   763,   698,   766,   564,
     768,   180,   318,   438,   439,   440,   441,   442,   791,   275,
     183,   770,   771,   185,   515,   722,   724,   289,   713,   726,
     307,   308,   729,   255,    63,   256,   257,   294,   295,   296,
     297,   264,   312,   141,  -274,  -274,   738,   533,   787,   231,
     502,   789,   415,   416,   417,   745,   248,   232,   751,   795,
     756,   236,   391,   233,  -274,   560,   265,   234,   198,   243,
    -274,  -274,  -274,  -274,  -274,  -274,  -274,   273,    86,   433,
     434,    86,   274,   402,  -274,   282,   283,   392,   774,   403,
     410,   405,   406,    63,   408,    64,   783,   137,   155,   566,
    -211,   155,  -211,   354,  -209,   786,  -209,   141,   435,   261,
     367,   249,   251,   290,   429,   794,   364,   252,    85,   369,
      86,   478,   479,   373,   374,   375,   376,   377,   378,   664,
     321,   322,   452,   380,   454,   455,   332,   333,   386,   486,
     487,   488,   489,   490,   383,   384,   567,   568,   569,   570,
     571,   572,   573,   444,   278,   747,   775,   253,   748,   749,
     396,  -318,   750,   146,   147,   240,   241,    63,   480,    90,
     242,   661,   484,   662,   466,   303,   497,   574,   263,   304,
     335,    90,  -318,    86,   280,   293,   566,  -318,   300,   148,
     149,   150,   151,   152,   153,   154,   302,   665,   666,   310,
     141,   314,   317,   155,   340,   319,   320,   341,   224,   225,
     410,   187,   188,   189,   190,   191,   192,   667,   699,   226,
     746,   323,   747,   141,   331,   748,   749,   343,   710,   750,
     344,   334,   249,   567,   568,   569,   570,   571,   572,   573,
     521,   342,   521,   244,   245,   246,   247,   346,  -318,   509,
     566,   526,   527,   528,    86,   585,   586,    63,   347,   412,
     539,   539,   539,    63,   574,   443,   743,   544,    90,   645,
     349,   356,   355,   470,  -318,   380,   273,   357,    93,    94,
      95,    96,    97,   198,   360,   557,   558,   559,   521,   538,
     104,   105,   566,   361,   106,   362,   370,   567,   568,   569,
     570,   571,   572,   573,   371,   306,   385,   387,   389,   511,
     555,   769,   512,   747,   388,   390,   748,   749,   393,   471,
     750,   394,   604,   397,   395,   398,   399,   576,   574,   587,
     429,   335,    90,  -353,   418,   531,   566,   428,   432,   567,
     568,   569,   570,   571,   572,   573,   779,   445,   747,   367,
     543,   748,   749,   446,   448,   750,  -231,   447,   643,   458,
     461,   553,   449,   450,   649,   459,   460,   463,   556,   619,
     574,   628,   464,   658,    90,  -355,   465,   467,   468,   469,
     668,   576,   482,   567,   568,   569,   570,   571,   572,   573,
     566,   483,   485,   423,   589,   492,   682,   494,   684,   747,
     628,   687,   748,   749,   777,   495,   750,   198,   672,   198,
     496,   497,   499,   500,   574,   501,   503,   694,   615,   471,
     508,   513,   514,   516,   545,   686,   566,   628,   565,   -11,
     534,   707,   546,   547,   712,   668,   145,   567,   568,   569,
     570,   571,   572,   573,   548,   549,   550,   551,   723,   598,
     594,   554,   563,   560,   198,   590,   582,   583,   733,   584,
     146,   147,   592,   597,   198,   566,   601,   605,   574,   741,
     610,   614,    90,   567,   568,   569,   570,   571,   572,   573,
      86,    90,   639,   566,   632,   759,   148,   149,   150,   151,
     152,   153,   154,   641,   635,   637,   647,   690,   640,   652,
     155,   654,   198,   659,   574,   773,   566,   657,   675,   663,
     754,   678,   567,   568,   569,   570,   571,   572,   573,   683,
     691,   692,   693,   695,   701,   703,   704,   706,   718,   788,
     567,   568,   569,   570,   571,   572,   573,   709,   711,   719,
     730,   732,   720,   574,   740,   735,   734,   680,   752,   767,
     757,   758,   760,   567,   568,   569,   570,   571,   572,   573,
     761,   574,   762,   730,   780,   721,   776,    62,   784,   790,
     792,   796,   108,    67,   305,   409,   730,   764,   730,   136,
     730,  -274,  -274,  -274,   574,  -274,  -274,  -274,   725,  -274,
    -274,  -274,  -274,  -274,   281,   419,   778,  -274,  -274,  -274,
    -274,  -274,  -274,  -274,  -274,  -274,  -274,  -274,  -274,  -274,
     562,  -274,  -274,  -274,  -274,  -274,  -274,  -274,  -274,  -274,
    -274,  -274,  -274,  -274,  -274,  -274,  -274,  -274,  -274,  -274,
    -274,  -274,  -274,  -274,   504,  -274,   239,  -274,  -274,   422,
     116,   299,   510,   118,  -274,  -274,  -274,  -274,  -274,  -274,
    -274,  -274,   436,   507,  -274,  -274,  -274,  -274,  -274,   437,
     345,   493,   532,   506,     6,     7,     8,   453,     9,    10,
      11,   505,    12,    13,    14,    15,    16,   715,   636,   655,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,   685,    30,    31,    32,    33,    34,    35,
      36,   613,   644,    37,    38,    39,    40,    41,    42,    43,
      44,    45,    46,    47,    48,    49,    50,   677,    51,     0,
      52,    53,     0,     0,     0,     0,     0,     0,     0,   566,
       0,     0,     0,     0,    54,     0,   566,    55,    56,    57,
      58,     0,     0,   566,     0,     0,     0,     0,     0,     0,
     566,     0,    68,   363,    -5,    -5,    69,    -5,    -5,    -5,
      -5,    -5,    -5,    -5,    -5,    -5,    -5,    -5,     0,    -5,
      -5,     0,     0,    -5,     0,     0,   567,   568,   569,   570,
     571,   572,   573,   567,   568,   569,   570,   571,   572,   573,
     567,   568,   569,   570,   571,   572,   573,   567,   568,   569,
     570,   571,   572,   573,   566,     0,     0,   574,     0,    70,
      71,   728,     0,     0,   574,    72,    73,     0,   755,     0,
       0,   574,     0,     0,     0,   782,     0,    74,   574,     0,
       0,     0,   785,     0,    -5,   -71,     0,   122,   123,   124,
     125,     0,   126,   127,   128,   129,   130,     0,     0,     0,
       0,   567,   568,   569,   570,   571,   572,   573,     1,     2,
       0,    93,    94,    95,    96,    97,    98,    99,   100,   101,
     102,   103,     0,   104,   105,     0,     0,   106,   131,     0,
       0,     0,   574,     0,     0,     0,   793,     0,     0,     0,
     146,   147,   517,   518,   519,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      86,     0,    63,     0,     0,   132,   148,   149,   150,   151,
     152,   153,   154,     6,     7,     8,     0,     9,    10,    11,
     155,    12,    13,    14,    15,    16,     0,     0,   107,    17,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      28,    29,     0,    30,    31,    32,    33,    34,    35,    36,
     146,   223,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,     0,    51,     0,    52,
     530,   200,     0,     0,     0,     0,   148,   149,   150,   151,
     152,   153,   154,    54,     0,     0,    55,    56,    57,    58,
     155,   201,   146,   202,   203,   204,   205,   206,   207,   208,
       0,     0,   209,   210,   211,   212,   213,   214,   215,   216,
       0,     0,    86,     0,     0,     0,     0,     0,   148,   149,
     150,   151,   152,   153,   154,     0,   217,   218,     0,   200,
       0,     0,   155,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   529,     0,     0,     0,   219,   220,   201,
       0,   202,   203,   204,   205,   206,   207,   208,     0,     0,
     209,   210,   211,   212,   213,   214,   215,   216,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   217,   218,     0,     0,     0,     0,
       0,     6,     7,     8,     0,     9,    10,    11,     0,    12,
      13,    14,    15,    16,     0,   219,   220,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
       0,    30,    31,    32,    33,    34,    35,    36,     0,     0,
      37,    38,    39,    40,    41,    42,    43,    44,    45,    46,
      47,    48,    49,    50,     0,    51,     0,    52,    53,   765,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    54,     0,     0,    55,    56,    57,    58,     6,     7,
       8,     0,     9,    10,    11,     0,    12,    13,    14,    15,
      16,     0,     0,     0,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,   664,    30,    31,
      32,    33,    34,    35,    36,     0,     0,    37,    38,    39,
      40,    41,    42,    43,    44,    45,    46,    47,    48,    49,
      50,     0,    51,     0,    52,    53,   287,     0,     0,     0,
       0,   146,   147,     0,     0,     0,     0,     0,    54,     0,
       0,    55,    56,    57,    58,     0,     0,     0,   146,   147,
       0,    86,     0,     0,     0,     0,     0,   148,   149,   150,
     151,   152,   153,   154,     0,   665,   666,     0,    86,     0,
       0,   155,     0,     0,   148,   149,   150,   151,   152,   153,
     154,   146,   147,   517,   518,   519,     0,     0,   288,     0,
       0,     0,     0,     0,     0,     0,   146,   147,     0,     0,
       0,    86,     0,     0,     0,     0,     0,   148,   149,   150,
     151,   152,   153,   154,   146,   223,    86,   325,   326,     0,
       0,   155,   148,   149,   150,   151,   152,   153,   154,   146,
     147,   520,     0,     0,    86,     0,   155,     0,     0,     0,
     148,   149,   150,   151,   152,   153,   154,     0,     0,    86,
       0,     0,     0,     0,   155,   148,   149,   150,   151,   152,
     153,   154,   621,   622,   623,   570,   624,   625,   626,   155,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   627,     0,     0,     0,    90
};

static const yytype_int16 yycheck[] =
{
       1,     2,   293,   303,    59,    60,    93,    92,    97,    71,
     138,   138,   185,   248,    74,    95,    96,   145,   365,    74,
     620,   138,    77,    85,   517,   518,   519,   601,   593,   336,
      17,   531,   604,   268,    17,   600,   174,   175,   176,   603,
     630,    17,    63,   181,    60,    61,   552,   340,    82,     0,
     343,   615,    81,   553,   618,    60,   630,    47,   143,    93,
     178,   569,   426,    83,    80,   573,   121,   657,    89,    70,
      78,    72,    73,   191,   639,    80,     1,   649,    84,   443,
     270,   271,   272,   657,   684,   392,   658,   687,   177,   589,
     596,   597,    40,   155,   594,   223,   223,    84,   185,   107,
     606,   394,   164,   104,   166,   167,   223,   169,    84,   674,
      93,   675,    95,   621,    41,   623,   680,   717,   626,    99,
     100,   101,   694,    48,    49,    50,    51,    52,    53,    54,
     695,   139,   640,   499,    60,   707,   642,    82,    60,    82,
     229,   279,    40,   281,   650,    93,    89,   232,   654,   234,
      41,   386,    93,   718,    79,    41,    81,   721,    83,   759,
      93,   725,    89,   734,   728,   512,   513,    15,   358,   359,
      92,   771,     3,     4,   263,     1,   747,   677,   749,   545,
     751,    93,   237,   373,   374,   375,   376,   377,   788,   278,
      62,   755,   757,    86,   494,   701,   704,   370,   691,   705,
      91,    92,   708,    89,    81,    91,    92,   187,   188,   189,
     190,    31,    60,    82,    40,    41,   720,   508,   782,    82,
      89,   785,   351,   352,   353,   729,   288,    83,    86,   793,
     736,    84,   317,    89,    60,    93,    56,    93,   293,    66,
      66,    67,    68,    69,    70,    71,    72,    56,    60,    61,
      62,    60,    61,   338,    80,    45,    46,   319,   762,   339,
     349,   341,   342,    81,   344,    83,   772,    93,    80,     1,
      85,    80,    87,   274,    85,   781,    87,    82,   367,    73,
      95,    86,    88,   370,    95,   791,   287,    88,    58,   290,
      60,   420,   421,   294,   295,   296,   297,   298,   299,     6,
      66,    67,   387,   304,   389,   390,    91,    92,   309,   438,
     439,   440,   441,   442,    91,    92,    48,    49,    50,    51,
      52,    53,    54,   378,    93,    89,    90,    88,    92,    93,
     331,    63,    96,    40,    41,    66,    67,    81,   425,    83,
      71,   632,   429,   634,   406,    85,    89,    79,    85,    89,
      93,    83,    84,    60,    93,    84,     1,    89,    62,    66,
      67,    68,    69,    70,    71,    72,    94,    74,    75,    84,
      82,    63,    90,    80,    86,    88,    66,    89,   506,   506,
     469,    11,    12,    13,    14,    15,    16,    94,   679,   506,
      87,    66,    89,    82,    88,    92,    93,    86,   689,    96,
      89,    92,    86,    48,    49,    50,    51,    52,    53,    54,
     497,    94,   499,    66,    67,    68,    69,    94,    63,   474,
       1,   501,   502,   503,    60,    61,    62,    81,    85,    83,
     517,   518,   519,    81,    79,    83,   727,   522,    83,    84,
      82,    85,    94,     1,    89,   446,    56,    94,     6,     7,
       8,     9,    10,   508,    94,   540,   541,   542,   545,   514,
      18,    19,     1,    85,    22,    83,    85,    48,    49,    50,
      51,    52,    53,    54,    87,    41,    92,    84,    90,   480,
     535,    87,   483,    89,    85,    94,    92,    93,    66,    47,
      96,    85,   577,    92,    87,    92,    92,   552,    79,   561,
      95,    93,    83,    84,    94,   506,     1,    92,    92,    48,
      49,    50,    51,    52,    53,    54,    87,    82,    89,    95,
     521,    92,    93,    85,    92,    96,    84,    90,   613,    90,
      94,   532,    92,    92,   619,    92,    92,    90,   539,   599,
      79,   601,    87,   628,    83,    84,    90,    87,    90,    85,
     637,   606,    84,    48,    49,    50,    51,    52,    53,    54,
       1,    92,    92,    17,   565,    61,   651,    85,   653,    89,
     630,   656,    92,    93,    94,    92,    96,   632,   638,   634,
      92,    89,    89,    92,    79,    89,    89,   672,    83,    47,
      84,    60,    84,    92,    85,   655,     1,   657,    40,    89,
      93,   686,    90,    90,   691,   692,    16,    48,    49,    50,
      51,    52,    53,    54,    90,    90,    90,    94,   703,    62,
      60,    94,    94,    93,   679,    89,    94,    94,   713,    94,
      40,    41,    93,    83,   689,     1,    83,    20,    79,   724,
      60,    85,    83,    48,    49,    50,    51,    52,    53,    54,
      60,    83,    81,     1,    84,   740,    66,    67,    68,    69,
      70,    71,    72,    90,    95,    93,    84,   668,    94,    93,
      80,    83,   727,    84,    79,   760,     1,    83,    83,    61,
     735,    84,    48,    49,    50,    51,    52,    53,    54,    60,
      82,    85,    94,    81,    55,    81,    94,    84,    82,   784,
      48,    49,    50,    51,    52,    53,    54,    84,    86,    84,
     711,   712,    84,    79,    81,   716,    86,    83,    94,    87,
      85,    84,    82,    48,    49,    50,    51,    52,    53,    54,
      84,    79,    84,   734,    84,    83,    94,     5,    85,    84,
      84,    84,    75,    60,   223,   347,   747,   748,   749,     1,
     751,     3,     4,     5,    79,     7,     8,     9,    83,    11,
      12,    13,    14,    15,   179,   356,   767,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
     543,    33,    34,    35,    36,    37,    38,    39,    40,    41,
      42,    43,    44,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,    55,   469,    57,   147,    59,    60,   361,
      75,   192,   475,    75,    66,    67,    68,    69,    70,    71,
      72,    73,   369,   472,    76,    77,    78,    79,    80,   370,
     263,   446,   506,   471,     3,     4,     5,   388,     7,     8,
       9,    93,    11,    12,    13,    14,    15,   692,   606,   625,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,   654,    33,    34,    35,    36,    37,    38,
      39,   594,   614,    42,    43,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    55,   643,    57,    -1,
      59,    60,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     1,
      -1,    -1,    -1,    -1,    73,    -1,     1,    76,    77,    78,
      79,    -1,    -1,     1,    -1,    -1,    -1,    -1,    -1,    -1,
       1,    -1,     1,    92,     3,     4,     5,     6,     7,     8,
       9,    10,    11,    12,    13,    14,    15,    16,    -1,    18,
      19,    -1,    -1,    22,    -1,    -1,    48,    49,    50,    51,
      52,    53,    54,    48,    49,    50,    51,    52,    53,    54,
      48,    49,    50,    51,    52,    53,    54,    48,    49,    50,
      51,    52,    53,    54,     1,    -1,    -1,    79,    -1,    58,
      59,    83,    -1,    -1,    79,    64,    65,    -1,    83,    -1,
      -1,    79,    -1,    -1,    -1,    83,    -1,    76,    79,    -1,
      -1,    -1,    83,    -1,    83,    84,    -1,     6,     7,     8,
       9,    -1,    11,    12,    13,    14,    15,    -1,    -1,    -1,
      -1,    48,    49,    50,    51,    52,    53,    54,     3,     4,
      -1,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    -1,    18,    19,    -1,    -1,    22,    47,    -1,
      -1,    -1,    79,    -1,    -1,    -1,    83,    -1,    -1,    -1,
      40,    41,    42,    43,    44,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      60,    -1,    81,    -1,    -1,    84,    66,    67,    68,    69,
      70,    71,    72,     3,     4,     5,    -1,     7,     8,     9,
      80,    11,    12,    13,    14,    15,    -1,    -1,    83,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    31,    -1,    33,    34,    35,    36,    37,    38,    39,
      40,    41,    42,    43,    44,    45,    46,    47,    48,    49,
      50,    51,    52,    53,    54,    55,    -1,    57,    -1,    59,
      60,     1,    -1,    -1,    -1,    -1,    66,    67,    68,    69,
      70,    71,    72,    73,    -1,    -1,    76,    77,    78,    79,
      80,    21,    40,    23,    24,    25,    26,    27,    28,    29,
      -1,    -1,    32,    33,    34,    35,    36,    37,    38,    39,
      -1,    -1,    60,    -1,    -1,    -1,    -1,    -1,    66,    67,
      68,    69,    70,    71,    72,    -1,    56,    57,    -1,     1,
      -1,    -1,    80,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    73,    -1,    -1,    -1,    77,    78,    21,
      -1,    23,    24,    25,    26,    27,    28,    29,    -1,    -1,
      32,    33,    34,    35,    36,    37,    38,    39,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    56,    57,    -1,    -1,    -1,    -1,
      -1,     3,     4,     5,    -1,     7,     8,     9,    -1,    11,
      12,    13,    14,    15,    -1,    77,    78,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      -1,    33,    34,    35,    36,    37,    38,    39,    -1,    -1,
      42,    43,    44,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,    55,    -1,    57,    -1,    59,    60,    61,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    73,    -1,    -1,    76,    77,    78,    79,     3,     4,
       5,    -1,     7,     8,     9,    -1,    11,    12,    13,    14,
      15,    -1,    -1,    -1,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,     6,    33,    34,
      35,    36,    37,    38,    39,    -1,    -1,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    -1,    57,    -1,    59,    60,    18,    -1,    -1,    -1,
      -1,    40,    41,    -1,    -1,    -1,    -1,    -1,    73,    -1,
      -1,    76,    77,    78,    79,    -1,    -1,    -1,    40,    41,
      -1,    60,    -1,    -1,    -1,    -1,    -1,    66,    67,    68,
      69,    70,    71,    72,    -1,    74,    75,    -1,    60,    -1,
      -1,    80,    -1,    -1,    66,    67,    68,    69,    70,    71,
      72,    40,    41,    42,    43,    44,    -1,    -1,    80,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    40,    41,    -1,    -1,
      -1,    60,    -1,    -1,    -1,    -1,    -1,    66,    67,    68,
      69,    70,    71,    72,    40,    41,    60,    61,    62,    -1,
      -1,    80,    66,    67,    68,    69,    70,    71,    72,    40,
      41,    90,    -1,    -1,    60,    -1,    80,    -1,    -1,    -1,
      66,    67,    68,    69,    70,    71,    72,    -1,    -1,    60,
      -1,    -1,    -1,    -1,    80,    66,    67,    68,    69,    70,
      71,    72,    48,    49,    50,    51,    52,    53,    54,    80,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    79,    -1,    -1,    -1,    83
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     3,     4,    98,    99,   105,     3,     4,     5,     7,
       8,     9,    11,    12,    13,    14,    15,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      33,    34,    35,    36,    37,    38,    39,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    57,    59,    60,    73,    76,    77,    78,    79,   103,
     103,     0,    99,    81,    83,   101,   106,   106,     1,     5,
      58,    59,    64,    65,    76,   100,   107,   108,   109,   175,
     214,   215,    81,    47,   103,    58,    60,   104,   103,   103,
      83,   101,   186,     6,     7,     8,     9,    10,    11,    12,
      13,    14,    15,    16,    18,    19,    22,    83,   105,   128,
     129,   145,   148,   149,   150,   152,   162,   163,   166,   167,
     168,    84,     6,     7,     8,     9,    11,    12,    13,    14,
      15,    47,    84,   101,   173,   107,     1,    93,   177,    83,
     104,    82,    63,    89,   184,    16,    40,    41,    66,    67,
      68,    69,    70,    71,    72,    80,   104,   114,   116,   117,
     118,   119,   120,   121,   124,    40,   130,   130,    93,   130,
     116,   169,    93,   134,   134,   134,   134,    93,   138,   151,
      93,   131,   103,    62,   170,    86,   107,    11,    12,    13,
      14,    15,    16,   153,   154,   155,   156,   157,   101,   102,
       1,    21,    23,    24,    25,    26,    27,    28,    29,    32,
      33,    34,    35,    36,    37,    38,    39,    56,    57,    77,
      78,   180,   181,    41,   117,   118,   119,   122,   123,   176,
     107,    82,    83,    89,    93,   184,    84,   187,   117,   121,
      66,    67,    71,    66,    66,    67,    68,    69,   104,    86,
     113,    88,    88,    88,    41,    89,    91,    92,   104,   104,
     104,    73,   104,    85,    31,    56,   135,   140,   103,   115,
     115,   115,   115,    56,    61,   116,   137,   139,    93,   151,
      93,   138,    45,    46,   132,   133,   115,    18,    80,   120,
     124,   160,   161,    84,   134,   134,   134,   134,   151,   131,
      62,   136,    94,    85,    89,   122,    41,    91,    92,   116,
      84,    15,    60,   184,    63,   183,   184,    90,   101,    88,
      66,    66,    67,    66,   113,    61,    62,   110,   111,   112,
     124,    88,    91,    92,    92,    93,   126,   127,   212,    89,
      86,    89,    94,    86,    89,   169,    94,    85,   113,    82,
     146,   146,   146,   146,   103,    94,    85,    94,   115,   115,
      94,    85,    83,    92,   103,    60,    92,    95,   159,   103,
      85,    87,   102,   103,   103,   103,   103,   103,   103,   180,
     103,   178,   179,    91,    92,    92,   103,    84,    85,    90,
      94,   184,   104,    66,    85,    87,   103,    92,    92,    92,
     127,   125,   184,   130,   111,   130,   130,   111,   130,   135,
     116,   147,    83,   101,   164,   164,   164,   164,    94,   139,
     146,   146,   132,    17,    84,   141,   143,   144,    92,    95,
     158,   158,    92,    61,    62,   116,   159,   161,   146,   146,
     146,   146,   146,    83,   101,    82,    85,    90,    92,    92,
      92,   113,   184,   183,   184,   184,   127,   111,    90,    92,
      92,    94,   213,    90,    87,    90,   104,    87,    90,    85,
       1,    47,   162,   165,   166,   171,   172,   174,   164,   164,
     124,   144,    84,    92,   124,    92,   164,   164,   164,   164,
     164,   144,    61,   179,    85,    92,    92,    89,   195,    89,
      92,    89,    89,    89,   147,    93,   177,   174,    84,   101,
     165,   103,   103,    60,    84,   180,    92,    42,    43,    44,
      90,   124,   185,   188,   193,   193,   130,   130,   130,    73,
      60,   103,   176,   102,    93,   142,   158,   158,   101,   124,
     185,   185,   185,   103,   184,    85,    90,    90,    90,    90,
      90,    94,   195,   103,    94,   101,   103,   184,   184,   184,
      93,    95,   141,    94,   193,    40,     1,    48,    49,    50,
      51,    52,    53,    54,    79,    83,   101,   186,   198,   203,
     205,   195,    94,    94,    94,    61,    62,   104,   182,   103,
      89,   209,    93,   209,    60,   210,   211,    83,    62,   202,
     209,    83,   199,   205,   184,    20,   197,   195,   184,   207,
      60,   207,   195,   212,    85,    83,   205,   200,   205,   186,
     207,    48,    49,    50,    52,    53,    54,    79,   186,   201,
     203,   204,    84,   199,   187,    95,   198,    93,   196,    81,
      94,    90,   208,   184,   211,    84,   199,    84,   199,   184,
     208,   209,    93,   209,    83,   202,   209,    83,   184,    84,
     201,   102,   102,    61,     6,    74,    75,    94,   124,   189,
     192,   194,   186,   207,   209,    83,   205,   213,    84,   187,
      83,   205,   184,    60,   184,   200,   186,   184,   201,   187,
     103,    82,    85,    94,   184,    81,   207,   199,   195,   102,
     199,    55,   206,    81,    94,   208,    84,   184,   208,    84,
     102,    86,   124,   185,   191,   194,   187,   207,    82,    84,
      84,    83,   205,   184,   209,    83,   205,   187,    83,   205,
     103,   190,   103,   184,    86,   103,   208,   207,   206,   199,
      81,   184,   199,   102,   199,   206,    87,    89,    92,    93,
      96,    86,    94,   190,   101,    83,   205,    85,    84,   184,
      82,    84,    84,   190,   103,    61,   190,    87,   190,    87,
     199,   207,   208,   184,   206,    90,    94,    94,   103,    87,
      84,   208,    83,   205,    85,    83,   205,   199,   184,   199,
      84,   208,    84,    83,   205,   199,    84
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
     103,   103,   103,   103,   104,   104,   104,   105,   105,   106,
     106,   107,   107,   108,   108,   108,   108,   108,   109,   109,
     109,   109,   109,   109,   109,   109,   109,   109,   109,   109,
     109,   109,   110,   110,   110,   111,   111,   112,   112,   113,
     113,   114,   114,   114,   114,   114,   114,   114,   114,   114,
     114,   114,   114,   114,   114,   114,   114,   114,   114,   114,
     114,   115,   116,   116,   117,   117,   118,   119,   119,   120,
     121,   121,   121,   121,   121,   121,   122,   122,   122,   122,
     122,   123,   123,   123,   123,   123,   123,   124,   124,   124,
     124,   124,   124,   125,   126,   127,   127,   128,   129,   130,
     130,   131,   131,   132,   132,   133,   133,   134,   134,   135,
     135,   136,   136,   137,   138,   138,   139,   139,   140,   140,
     141,   141,   142,   142,   143,   144,   144,   145,   145,   145,
     146,   146,   147,   147,   148,   148,   149,   150,   151,   151,
     152,   152,   153,   153,   154,   155,   156,   157,   157,   158,
     158,   159,   159,   159,   159,   160,   160,   160,   160,   160,
     160,   161,   161,   162,   163,   163,   163,   163,   163,   164,
     164,   165,   165,   166,   166,   166,   166,   166,   166,   166,
     167,   167,   167,   167,   167,   168,   168,   168,   168,   169,
     169,   170,   171,   172,   172,   172,   172,   173,   173,   173,
     173,   173,   173,   173,   173,   173,   173,   173,   174,   174,
     174,   175,   175,   176,   177,   177,   177,   178,   179,   179,
     180,   180,   180,   180,   181,   181,   181,   181,   181,   181,
     181,   181,   181,   181,   181,   181,   181,   181,   181,   181,
     181,   181,   181,   181,   181,   182,   182,   182,   183,   183,
     183,   184,   184,   184,   184,   184,   184,   185,   186,   187,
     188,   188,   188,   188,   188,   188,   188,   189,   189,   189,
     190,   190,   190,   190,   190,   190,   191,   192,   192,   192,
     193,   193,   194,   194,   195,   195,   196,   196,   197,   197,
     198,   198,   198,   199,   199,   200,   200,   201,   201,   201,
     202,   202,   203,   203,   203,   204,   204,   204,   204,   204,
     204,   204,   204,   204,   204,   204,   204,   205,   205,   205,
     205,   205,   205,   205,   205,   205,   205,   205,   205,   205,
     205,   206,   206,   206,   207,   208,   209,   210,   210,   211,
     211,   212,   213,   214,   215
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
       1,     1,     1,     1,     1,     4,     4,     3,     3,     1,
       4,     0,     2,     3,     2,     2,     2,     8,     5,     5,
       2,     2,     2,     2,     2,     2,     2,     2,     1,     1,
       1,     1,     1,     1,     1,     1,     3,     0,     1,     0,
       3,     1,     1,     2,     1,     2,     1,     2,     2,     3,
       3,     4,     2,     3,     2,     2,     3,     1,     1,     2,
       1,     2,     2,     3,     1,     1,     2,     2,     2,     8,
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
      13,     5,     2,     1,     0,     3,     1,     3,     1,     3,
       1,     4,     3,     6,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       2,     1,     1,     1,     1,     1,     1,     1,     0,     1,
       3,     0,     1,     5,     5,     5,     4,     3,     1,     1,
       1,     3,     4,     3,     4,     4,     4,     1,     1,     1,
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
    { (yyval.strval) = strdup("inbox"); }
#line 2543 "y.tab.c" /* yacc.c:1646  */
    break;

  case 43:
#line 272 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = strdup("coroutine"); }
#line 2549 "y.tab.c" /* yacc.c:1646  */
    break;

  case 44:
#line 273 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOTRACE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2555 "y.tab.c" /* yacc.c:1646  */
    break;

  case 45:
#line 274 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(APPWORK, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2561 "y.tab.c" /* yacc.c:1646  */
    break;

  case 46:
#line 277 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(PACKED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2567 "y.tab.c" /* yacc.c:1646  */
    break;

  case 47:
#line 278 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(VARSIZE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2573 "y.tab.c" /* yacc.c:1646  */
    break;

  case 48:
#line 279 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ENTRY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2579 "y.tab.c" /* yacc.c:1646  */
    break;

  case 49:
#line 280 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(FOR, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2585 "y.tab.c" /* yacc.c:1646  */
    break;

  case 50:
#line 281 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(FORALL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2591 "y.tab.c" /* yacc.c:1646  */
    break;

  case 51:
#line 282 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(WHILE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2597 "y.tab.c" /* yacc.c:1646  */
    break;

  case 52:
#line 283 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(WHEN, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2603 "y.tab.c" /* yacc.c:1646  */
    break;

  case 53:
#line 284 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(OVERLAP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2609 "y.tab.c" /* yacc.c:1646  */
    break;

  case 54:
#line 285 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(SERIAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2615 "y.tab.c" /* yacc.c:1646  */
    break;

  case 55:
#line 286 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IF, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2621 "y.tab.c" /* yacc.c:1646  */
    break;

  case 56:
#line 287 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ELSE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2627 "y.tab.c" /* yacc.c:1646  */
    break;

  case 57:
#line 289 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(LOCAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2633 "y.tab.c" /* yacc.c:1646  */
    break;

  case 58:
#line 291 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(USING, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2639 "y.tab.c" /* yacc.c:1646  */
    break;

  case 59:
#line 292 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ACCEL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2645 "y.tab.c" /* yacc.c:1646  */
    break;

  case 60:
#line 295 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ACCELBLOCK, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2651 "y.tab.c" /* yacc.c:1646  */
    break;

  case 61:
#line 296 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MEMCRITICAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2657 "y.tab.c" /* yacc.c:1646  */
    break;

  case 62:
#line 297 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(REDUCTIONTARGET, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2663 "y.tab.c" /* yacc.c:1646  */
    break;

  case 63:
#line 298 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CASE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2669 "y.tab.c" /* yacc.c:1646  */
    break;

  case 64:
#line 303 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 2675 "y.tab.c" /* yacc.c:1646  */
    break;

  case 65:
#line 305 "xi-grammar.y" /* yacc.c:1646  */
    {
		  int len = strlen((yyvsp[-3].strval))+strlen((yyvsp[0].strval))+3;
		  char *tmp = new char[len];
		  snprintf(tmp,len,"%s::%s", (yyvsp[-3].strval), (yyvsp[0].strval));
		  (yyval.strval) = tmp;
		}
#line 2686 "y.tab.c" /* yacc.c:1646  */
    break;

  case 66:
#line 312 "xi-grammar.y" /* yacc.c:1646  */
    {
		  int len = strlen((yyvsp[-3].strval))+5+3;
		  char *tmp = new char[len];
		  snprintf(tmp, len, "%s::array", (yyvsp[-3].strval));
		  (yyval.strval) = tmp;
		}
#line 2697 "y.tab.c" /* yacc.c:1646  */
    break;

  case 67:
#line 320 "xi-grammar.y" /* yacc.c:1646  */
    { 
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist)); 
		}
#line 2705 "y.tab.c" /* yacc.c:1646  */
    break;

  case 68:
#line 324 "xi-grammar.y" /* yacc.c:1646  */
    {  
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist)); 
		    (yyval.module)->setMain();
		}
#line 2714 "y.tab.c" /* yacc.c:1646  */
    break;

  case 69:
#line 331 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = 0; }
#line 2720 "y.tab.c" /* yacc.c:1646  */
    break;

  case 70:
#line 333 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = (yyvsp[-2].conslist); }
#line 2726 "y.tab.c" /* yacc.c:1646  */
    break;

  case 71:
#line 337 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = 0; }
#line 2732 "y.tab.c" /* yacc.c:1646  */
    break;

  case 72:
#line 339 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = new ConstructList(lineno, (yyvsp[-1].construct), (yyvsp[0].conslist)); }
#line 2738 "y.tab.c" /* yacc.c:1646  */
    break;

  case 73:
#line 343 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new UsingScope((yyvsp[0].strval), false); }
#line 2744 "y.tab.c" /* yacc.c:1646  */
    break;

  case 74:
#line 345 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new UsingScope((yyvsp[0].strval), true); }
#line 2750 "y.tab.c" /* yacc.c:1646  */
    break;

  case 75:
#line 347 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].member)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].member); }
#line 2756 "y.tab.c" /* yacc.c:1646  */
    break;

  case 76:
#line 349 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].message)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].message); }
#line 2762 "y.tab.c" /* yacc.c:1646  */
    break;

  case 77:
#line 351 "xi-grammar.y" /* yacc.c:1646  */
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
#line 2778 "y.tab.c" /* yacc.c:1646  */
    break;

  case 78:
#line 365 "xi-grammar.y" /* yacc.c:1646  */
    { if((yyvsp[-2].conslist)) (yyvsp[-2].conslist)->recurse<int&>((yyvsp[-4].intval), &Construct::setExtern); (yyval.construct) = (yyvsp[-2].conslist); }
#line 2784 "y.tab.c" /* yacc.c:1646  */
    break;

  case 79:
#line 367 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new Scope((yyvsp[-3].strval), (yyvsp[-1].conslist)); }
#line 2790 "y.tab.c" /* yacc.c:1646  */
    break;

  case 80:
#line 369 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = (yyvsp[-1].construct); }
#line 2796 "y.tab.c" /* yacc.c:1646  */
    break;

  case 81:
#line 371 "xi-grammar.y" /* yacc.c:1646  */
    {
          ERROR("preceding construct must be semicolon terminated",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 2806 "y.tab.c" /* yacc.c:1646  */
    break;

  case 82:
#line 377 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].module)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].module); }
#line 2812 "y.tab.c" /* yacc.c:1646  */
    break;

  case 83:
#line 379 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2818 "y.tab.c" /* yacc.c:1646  */
    break;

  case 84:
#line 381 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2824 "y.tab.c" /* yacc.c:1646  */
    break;

  case 85:
#line 383 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2830 "y.tab.c" /* yacc.c:1646  */
    break;

  case 86:
#line 385 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2836 "y.tab.c" /* yacc.c:1646  */
    break;

  case 87:
#line 387 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].templat)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].templat); }
#line 2842 "y.tab.c" /* yacc.c:1646  */
    break;

  case 88:
#line 389 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = NULL; }
#line 2848 "y.tab.c" /* yacc.c:1646  */
    break;

  case 89:
#line 391 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = NULL; }
#line 2854 "y.tab.c" /* yacc.c:1646  */
    break;

  case 90:
#line 393 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = (yyvsp[0].accelBlock); }
#line 2860 "y.tab.c" /* yacc.c:1646  */
    break;

  case 91:
#line 395 "xi-grammar.y" /* yacc.c:1646  */
    {
          ERROR("invalid construct",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 2870 "y.tab.c" /* yacc.c:1646  */
    break;

  case 92:
#line 403 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamType((yyvsp[0].type)); }
#line 2876 "y.tab.c" /* yacc.c:1646  */
    break;

  case 93:
#line 405 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
#line 2882 "y.tab.c" /* yacc.c:1646  */
    break;

  case 94:
#line 407 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
#line 2888 "y.tab.c" /* yacc.c:1646  */
    break;

  case 95:
#line 411 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = new TParamList((yyvsp[0].tparam)); }
#line 2894 "y.tab.c" /* yacc.c:1646  */
    break;

  case 96:
#line 413 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = new TParamList((yyvsp[-2].tparam), (yyvsp[0].tparlist)); }
#line 2900 "y.tab.c" /* yacc.c:1646  */
    break;

  case 97:
#line 417 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = new TParamList(0); }
#line 2906 "y.tab.c" /* yacc.c:1646  */
    break;

  case 98:
#line 419 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = (yyvsp[0].tparlist); }
#line 2912 "y.tab.c" /* yacc.c:1646  */
    break;

  case 99:
#line 423 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = 0; }
#line 2918 "y.tab.c" /* yacc.c:1646  */
    break;

  case 100:
#line 425 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = (yyvsp[-1].tparlist); }
#line 2924 "y.tab.c" /* yacc.c:1646  */
    break;

  case 101:
#line 429 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("int"); }
#line 2930 "y.tab.c" /* yacc.c:1646  */
    break;

  case 102:
#line 431 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long"); }
#line 2936 "y.tab.c" /* yacc.c:1646  */
    break;

  case 103:
#line 433 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long int"); }
#line 2942 "y.tab.c" /* yacc.c:1646  */
    break;

  case 104:
#line 435 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("short"); }
#line 2948 "y.tab.c" /* yacc.c:1646  */
    break;

  case 105:
#line 437 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("short int"); }
#line 2954 "y.tab.c" /* yacc.c:1646  */
    break;

  case 106:
#line 439 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("char"); }
#line 2960 "y.tab.c" /* yacc.c:1646  */
    break;

  case 107:
#line 441 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned int"); }
#line 2966 "y.tab.c" /* yacc.c:1646  */
    break;

  case 108:
#line 443 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long"); }
#line 2972 "y.tab.c" /* yacc.c:1646  */
    break;

  case 109:
#line 445 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long int"); }
#line 2978 "y.tab.c" /* yacc.c:1646  */
    break;

  case 110:
#line 447 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long long"); }
#line 2984 "y.tab.c" /* yacc.c:1646  */
    break;

  case 111:
#line 449 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long long int"); }
#line 2990 "y.tab.c" /* yacc.c:1646  */
    break;

  case 112:
#line 451 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned short"); }
#line 2996 "y.tab.c" /* yacc.c:1646  */
    break;

  case 113:
#line 453 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned short int"); }
#line 3002 "y.tab.c" /* yacc.c:1646  */
    break;

  case 114:
#line 455 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned char"); }
#line 3008 "y.tab.c" /* yacc.c:1646  */
    break;

  case 115:
#line 457 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long long"); }
#line 3014 "y.tab.c" /* yacc.c:1646  */
    break;

  case 116:
#line 459 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long long int"); }
#line 3020 "y.tab.c" /* yacc.c:1646  */
    break;

  case 117:
#line 461 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("float"); }
#line 3026 "y.tab.c" /* yacc.c:1646  */
    break;

  case 118:
#line 463 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("double"); }
#line 3032 "y.tab.c" /* yacc.c:1646  */
    break;

  case 119:
#line 465 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long double"); }
#line 3038 "y.tab.c" /* yacc.c:1646  */
    break;

  case 120:
#line 467 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("void"); }
#line 3044 "y.tab.c" /* yacc.c:1646  */
    break;

  case 121:
#line 470 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ntype) = new NamedType((yyvsp[-1].strval),(yyvsp[0].tparlist)); }
#line 3050 "y.tab.c" /* yacc.c:1646  */
    break;

  case 122:
#line 471 "xi-grammar.y" /* yacc.c:1646  */
    { 
                    const char* basename, *scope;
                    splitScopedName((yyvsp[-1].strval), &scope, &basename);
                    (yyval.ntype) = new NamedType(basename, (yyvsp[0].tparlist), scope);
                }
#line 3060 "y.tab.c" /* yacc.c:1646  */
    break;

  case 123:
#line 477 "xi-grammar.y" /* yacc.c:1646  */
    {
			const char* basename, *scope;
			splitScopedName((yyvsp[-1].strval), &scope, &basename);
			(yyval.ntype) = new NamedType(basename, (yyvsp[0].tparlist), scope, true);
		}
#line 3070 "y.tab.c" /* yacc.c:1646  */
    break;

  case 124:
#line 485 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3076 "y.tab.c" /* yacc.c:1646  */
    break;

  case 125:
#line 487 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ntype); }
#line 3082 "y.tab.c" /* yacc.c:1646  */
    break;

  case 126:
#line 491 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ptype) = new PtrType((yyvsp[-1].type)); }
#line 3088 "y.tab.c" /* yacc.c:1646  */
    break;

  case 127:
#line 495 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
#line 3094 "y.tab.c" /* yacc.c:1646  */
    break;

  case 128:
#line 497 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
#line 3100 "y.tab.c" /* yacc.c:1646  */
    break;

  case 129:
#line 501 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ftype) = new FuncType((yyvsp[-7].type), (yyvsp[-4].strval), (yyvsp[-1].plist)); }
#line 3106 "y.tab.c" /* yacc.c:1646  */
    break;

  case 130:
#line 505 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3112 "y.tab.c" /* yacc.c:1646  */
    break;

  case 131:
#line 507 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3118 "y.tab.c" /* yacc.c:1646  */
    break;

  case 132:
#line 509 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3124 "y.tab.c" /* yacc.c:1646  */
    break;

  case 133:
#line 511 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ftype); }
#line 3130 "y.tab.c" /* yacc.c:1646  */
    break;

  case 134:
#line 513 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[0].type)); }
#line 3136 "y.tab.c" /* yacc.c:1646  */
    break;

  case 135:
#line 515 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[-1].type)); }
#line 3142 "y.tab.c" /* yacc.c:1646  */
    break;

  case 136:
#line 519 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3148 "y.tab.c" /* yacc.c:1646  */
    break;

  case 137:
#line 521 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3154 "y.tab.c" /* yacc.c:1646  */
    break;

  case 138:
#line 523 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3160 "y.tab.c" /* yacc.c:1646  */
    break;

  case 139:
#line 525 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[0].type)); }
#line 3166 "y.tab.c" /* yacc.c:1646  */
    break;

  case 140:
#line 527 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[-1].type)); }
#line 3172 "y.tab.c" /* yacc.c:1646  */
    break;

  case 141:
#line 531 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType(new RValueReferenceType((yyvsp[-5].type))); }
#line 3178 "y.tab.c" /* yacc.c:1646  */
    break;

  case 142:
#line 533 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType(new ReferenceType((yyvsp[-4].type))); }
#line 3184 "y.tab.c" /* yacc.c:1646  */
    break;

  case 143:
#line 535 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType((yyvsp[-3].type)); }
#line 3190 "y.tab.c" /* yacc.c:1646  */
    break;

  case 144:
#line 537 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new RValueReferenceType((yyvsp[-2].type)); }
#line 3196 "y.tab.c" /* yacc.c:1646  */
    break;

  case 145:
#line 539 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ReferenceType((yyvsp[-1].type)); }
#line 3202 "y.tab.c" /* yacc.c:1646  */
    break;

  case 146:
#line 541 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3208 "y.tab.c" /* yacc.c:1646  */
    break;

  case 147:
#line 545 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType(new RValueReferenceType((yyvsp[-5].type))); }
#line 3214 "y.tab.c" /* yacc.c:1646  */
    break;

  case 148:
#line 547 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType(new ReferenceType((yyvsp[-4].type))); }
#line 3220 "y.tab.c" /* yacc.c:1646  */
    break;

  case 149:
#line 549 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new EllipsisType((yyvsp[-3].type)); }
#line 3226 "y.tab.c" /* yacc.c:1646  */
    break;

  case 150:
#line 551 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new RValueReferenceType((yyvsp[-2].type)); }
#line 3232 "y.tab.c" /* yacc.c:1646  */
    break;

  case 151:
#line 553 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ReferenceType((yyvsp[-1].type)); }
#line 3238 "y.tab.c" /* yacc.c:1646  */
    break;

  case 152:
#line 555 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3244 "y.tab.c" /* yacc.c:1646  */
    break;

  case 153:
#line 559 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 3250 "y.tab.c" /* yacc.c:1646  */
    break;

  case 154:
#line 563 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = (yyvsp[-1].val); }
#line 3256 "y.tab.c" /* yacc.c:1646  */
    break;

  case 155:
#line 567 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.vallist) = 0; }
#line 3262 "y.tab.c" /* yacc.c:1646  */
    break;

  case 156:
#line 569 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.vallist) = new ValueList((yyvsp[-1].val), (yyvsp[0].vallist)); }
#line 3268 "y.tab.c" /* yacc.c:1646  */
    break;

  case 157:
#line 573 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].vallist)); }
#line 3274 "y.tab.c" /* yacc.c:1646  */
    break;

  case 158:
#line 577 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[-3].type), (yyvsp[-1].strval), (yyvsp[0].vallist), 1); }
#line 3280 "y.tab.c" /* yacc.c:1646  */
    break;

  case 159:
#line 581 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0;}
#line 3286 "y.tab.c" /* yacc.c:1646  */
    break;

  case 160:
#line 583 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0;}
#line 3292 "y.tab.c" /* yacc.c:1646  */
    break;

  case 161:
#line 587 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3298 "y.tab.c" /* yacc.c:1646  */
    break;

  case 162:
#line 589 "xi-grammar.y" /* yacc.c:1646  */
    { 
		  /*
		  printf("Warning: Message attributes are being phased out.\n");
		  printf("Warning: Please remove them from interface files.\n");
		  */
		  (yyval.intval) = (yyvsp[-1].intval); 
		}
#line 3310 "y.tab.c" /* yacc.c:1646  */
    break;

  case 163:
#line 599 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[0].intval); }
#line 3316 "y.tab.c" /* yacc.c:1646  */
    break;

  case 164:
#line 601 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[-2].intval) | (yyvsp[0].intval); }
#line 3322 "y.tab.c" /* yacc.c:1646  */
    break;

  case 165:
#line 605 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3328 "y.tab.c" /* yacc.c:1646  */
    break;

  case 166:
#line 607 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3334 "y.tab.c" /* yacc.c:1646  */
    break;

  case 167:
#line 611 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = 0; }
#line 3340 "y.tab.c" /* yacc.c:1646  */
    break;

  case 168:
#line 613 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-1].cattr); }
#line 3346 "y.tab.c" /* yacc.c:1646  */
    break;

  case 169:
#line 617 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[0].cattr); }
#line 3352 "y.tab.c" /* yacc.c:1646  */
    break;

  case 170:
#line 619 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
#line 3358 "y.tab.c" /* yacc.c:1646  */
    break;

  case 171:
#line 623 "xi-grammar.y" /* yacc.c:1646  */
    { python_doc = NULL; (yyval.intval) = 0; }
#line 3364 "y.tab.c" /* yacc.c:1646  */
    break;

  case 172:
#line 625 "xi-grammar.y" /* yacc.c:1646  */
    { python_doc = (yyvsp[0].strval); (yyval.intval) = 0; }
#line 3370 "y.tab.c" /* yacc.c:1646  */
    break;

  case 173:
#line 629 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CPYTHON; }
#line 3376 "y.tab.c" /* yacc.c:1646  */
    break;

  case 174:
#line 633 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = 0; }
#line 3382 "y.tab.c" /* yacc.c:1646  */
    break;

  case 175:
#line 635 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-1].cattr); }
#line 3388 "y.tab.c" /* yacc.c:1646  */
    break;

  case 176:
#line 639 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[0].cattr); }
#line 3394 "y.tab.c" /* yacc.c:1646  */
    break;

  case 177:
#line 641 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
#line 3400 "y.tab.c" /* yacc.c:1646  */
    break;

  case 178:
#line 645 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CMIGRATABLE; }
#line 3406 "y.tab.c" /* yacc.c:1646  */
    break;

  case 179:
#line 647 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CPYTHON; }
#line 3412 "y.tab.c" /* yacc.c:1646  */
    break;

  case 180:
#line 651 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3418 "y.tab.c" /* yacc.c:1646  */
    break;

  case 181:
#line 653 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 3424 "y.tab.c" /* yacc.c:1646  */
    break;

  case 182:
#line 656 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3430 "y.tab.c" /* yacc.c:1646  */
    break;

  case 183:
#line 658 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 3436 "y.tab.c" /* yacc.c:1646  */
    break;

  case 184:
#line 661 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mv) = new MsgVar((yyvsp[-3].type), (yyvsp[-2].strval), (yyvsp[-4].intval), (yyvsp[-1].intval)); }
#line 3442 "y.tab.c" /* yacc.c:1646  */
    break;

  case 185:
#line 665 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mvlist) = new MsgVarList((yyvsp[0].mv)); }
#line 3448 "y.tab.c" /* yacc.c:1646  */
    break;

  case 186:
#line 667 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mvlist) = new MsgVarList((yyvsp[-1].mv), (yyvsp[0].mvlist)); }
#line 3454 "y.tab.c" /* yacc.c:1646  */
    break;

  case 187:
#line 671 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, (yyvsp[0].ntype)); }
#line 3460 "y.tab.c" /* yacc.c:1646  */
    break;

  case 188:
#line 673 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, (yyvsp[-2].ntype)); }
#line 3466 "y.tab.c" /* yacc.c:1646  */
    break;

  case 189:
#line 675 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, (yyvsp[-3].ntype), (yyvsp[-1].mvlist)); }
#line 3472 "y.tab.c" /* yacc.c:1646  */
    break;

  case 190:
#line 679 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = 0; }
#line 3478 "y.tab.c" /* yacc.c:1646  */
    break;

  case 191:
#line 681 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = (yyvsp[0].typelist); }
#line 3484 "y.tab.c" /* yacc.c:1646  */
    break;

  case 192:
#line 685 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = new TypeList((yyvsp[0].ntype)); }
#line 3490 "y.tab.c" /* yacc.c:1646  */
    break;

  case 193:
#line 687 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = new TypeList((yyvsp[-2].ntype), (yyvsp[0].typelist)); }
#line 3496 "y.tab.c" /* yacc.c:1646  */
    break;

  case 194:
#line 691 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr)|Chare::CCHARE, (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3502 "y.tab.c" /* yacc.c:1646  */
    break;

  case 195:
#line 693 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3508 "y.tab.c" /* yacc.c:1646  */
    break;

  case 196:
#line 697 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3514 "y.tab.c" /* yacc.c:1646  */
    break;

  case 197:
#line 701 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new NodeGroup(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3520 "y.tab.c" /* yacc.c:1646  */
    break;

  case 198:
#line 705 "xi-grammar.y" /* yacc.c:1646  */
    {/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			snprintf(buf,40,"%sD",(yyvsp[-2].strval));
			(yyval.ntype) = new NamedType(buf); 
		}
#line 3530 "y.tab.c" /* yacc.c:1646  */
    break;

  case 199:
#line 711 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ntype) = (yyvsp[-1].ntype); }
#line 3536 "y.tab.c" /* yacc.c:1646  */
    break;

  case 200:
#line 715 "xi-grammar.y" /* yacc.c:1646  */
    {  (yyval.chare) = new Array(lineno, (yyvsp[-4].cattr), (yyvsp[-3].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3542 "y.tab.c" /* yacc.c:1646  */
    break;

  case 201:
#line 717 "xi-grammar.y" /* yacc.c:1646  */
    {  (yyval.chare) = new Array(lineno, (yyvsp[-3].cattr), (yyvsp[-4].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3548 "y.tab.c" /* yacc.c:1646  */
    break;

  case 202:
#line 721 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr)|Chare::CCHARE, new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist));}
#line 3554 "y.tab.c" /* yacc.c:1646  */
    break;

  case 203:
#line 723 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3560 "y.tab.c" /* yacc.c:1646  */
    break;

  case 204:
#line 727 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3566 "y.tab.c" /* yacc.c:1646  */
    break;

  case 205:
#line 731 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new NodeGroup( lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3572 "y.tab.c" /* yacc.c:1646  */
    break;

  case 206:
#line 735 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Array( lineno, 0, (yyvsp[-3].ntype), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3578 "y.tab.c" /* yacc.c:1646  */
    break;

  case 207:
#line 739 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-1].strval))); }
#line 3584 "y.tab.c" /* yacc.c:1646  */
    break;

  case 208:
#line 741 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-4].strval)), (yyvsp[-2].mvlist)); }
#line 3590 "y.tab.c" /* yacc.c:1646  */
    break;

  case 209:
#line 745 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = 0; }
#line 3596 "y.tab.c" /* yacc.c:1646  */
    break;

  case 210:
#line 747 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3602 "y.tab.c" /* yacc.c:1646  */
    break;

  case 211:
#line 751 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = 0; }
#line 3608 "y.tab.c" /* yacc.c:1646  */
    break;

  case 212:
#line 753 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3614 "y.tab.c" /* yacc.c:1646  */
    break;

  case 213:
#line 755 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3620 "y.tab.c" /* yacc.c:1646  */
    break;

  case 214:
#line 757 "xi-grammar.y" /* yacc.c:1646  */
    {
		  XStr typeStr;
		  (yyvsp[0].ntype)->print(typeStr);
		  char *tmp = strdup(typeStr.get_string());
		  (yyval.strval) = tmp;
		}
#line 3631 "y.tab.c" /* yacc.c:1646  */
    break;

  case 215:
#line 766 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TTypeEllipsis(new NamedEllipsisType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3637 "y.tab.c" /* yacc.c:1646  */
    break;

  case 216:
#line 768 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TTypeEllipsis(new NamedEllipsisType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3643 "y.tab.c" /* yacc.c:1646  */
    break;

  case 217:
#line 770 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TType(new NamedType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3649 "y.tab.c" /* yacc.c:1646  */
    break;

  case 218:
#line 772 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TType(new NamedType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3655 "y.tab.c" /* yacc.c:1646  */
    break;

  case 219:
#line 774 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TFunc((yyvsp[-1].ftype), (yyvsp[0].strval)); }
#line 3661 "y.tab.c" /* yacc.c:1646  */
    break;

  case 220:
#line 776 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TName((yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].strval)); }
#line 3667 "y.tab.c" /* yacc.c:1646  */
    break;

  case 221:
#line 780 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = new TVarList((yyvsp[0].tvar)); }
#line 3673 "y.tab.c" /* yacc.c:1646  */
    break;

  case 222:
#line 782 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = new TVarList((yyvsp[-2].tvar), (yyvsp[0].tvarlist)); }
#line 3679 "y.tab.c" /* yacc.c:1646  */
    break;

  case 223:
#line 786 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = (yyvsp[-1].tvarlist); }
#line 3685 "y.tab.c" /* yacc.c:1646  */
    break;

  case 224:
#line 790 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3691 "y.tab.c" /* yacc.c:1646  */
    break;

  case 225:
#line 792 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3697 "y.tab.c" /* yacc.c:1646  */
    break;

  case 226:
#line 794 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3703 "y.tab.c" /* yacc.c:1646  */
    break;

  case 227:
#line 796 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3709 "y.tab.c" /* yacc.c:1646  */
    break;

  case 228:
#line 798 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].message)); (yyvsp[0].message)->setTemplate((yyval.templat)); }
#line 3715 "y.tab.c" /* yacc.c:1646  */
    break;

  case 229:
#line 802 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = 0; }
#line 3721 "y.tab.c" /* yacc.c:1646  */
    break;

  case 230:
#line 804 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = (yyvsp[-2].mbrlist); }
#line 3727 "y.tab.c" /* yacc.c:1646  */
    break;

  case 231:
#line 808 "xi-grammar.y" /* yacc.c:1646  */
    { 
                  if (!connectEntries.empty()) {
                    (yyval.mbrlist) = new AstChildren<Member>(connectEntries);
		  } else {
		    (yyval.mbrlist) = 0; 
                  }
		}
#line 3739 "y.tab.c" /* yacc.c:1646  */
    break;

  case 232:
#line 816 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = new AstChildren<Member>(-1, (yyvsp[-1].member), (yyvsp[0].mbrlist)); }
#line 3745 "y.tab.c" /* yacc.c:1646  */
    break;

  case 233:
#line 820 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].readonly); }
#line 3751 "y.tab.c" /* yacc.c:1646  */
    break;

  case 234:
#line 822 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].readonly); }
#line 3757 "y.tab.c" /* yacc.c:1646  */
    break;

  case 236:
#line 825 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].member); }
#line 3763 "y.tab.c" /* yacc.c:1646  */
    break;

  case 237:
#line 827 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].pupable); }
#line 3769 "y.tab.c" /* yacc.c:1646  */
    break;

  case 238:
#line 829 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].includeFile); }
#line 3775 "y.tab.c" /* yacc.c:1646  */
    break;

  case 239:
#line 831 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new ClassDeclaration(lineno,(yyvsp[0].strval)); }
#line 3781 "y.tab.c" /* yacc.c:1646  */
    break;

  case 240:
#line 835 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1); }
#line 3787 "y.tab.c" /* yacc.c:1646  */
    break;

  case 241:
#line 837 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1); }
#line 3793 "y.tab.c" /* yacc.c:1646  */
    break;

  case 242:
#line 839 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[-6].strval)) + '<' +
					    ((yyvsp[-4].tparlist))->to_string() + '>').c_str()),
				    1);
		}
#line 3803 "y.tab.c" /* yacc.c:1646  */
    break;

  case 243:
#line 845 "xi-grammar.y" /* yacc.c:1646  */
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[-2]).first_column, (yylsp[-2]).last_column, (yylsp[-2]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1);
		}
#line 3813 "y.tab.c" /* yacc.c:1646  */
    break;

  case 244:
#line 851 "xi-grammar.y" /* yacc.c:1646  */
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[-5]).first_column, (yylsp[-5]).last_column, (yylsp[-5]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1);
		}
#line 3823 "y.tab.c" /* yacc.c:1646  */
    break;

  case 245:
#line 860 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 0); }
#line 3829 "y.tab.c" /* yacc.c:1646  */
    break;

  case 246:
#line 862 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 0); }
#line 3835 "y.tab.c" /* yacc.c:1646  */
    break;

  case 247:
#line 864 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[-6].strval)) + '<' +
					    ((yyvsp[-4].tparlist))->to_string() + '>').c_str()),
				    0);
		}
#line 3845 "y.tab.c" /* yacc.c:1646  */
    break;

  case 248:
#line 870 "xi-grammar.y" /* yacc.c:1646  */
    {
                  InitCall* rtn = new InitCall(lineno, (yyvsp[-3].strval), 0);
                  rtn->setAccel();
                  (yyval.member) = rtn;
		}
#line 3855 "y.tab.c" /* yacc.c:1646  */
    break;

  case 249:
#line 878 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[0].ntype),0); }
#line 3861 "y.tab.c" /* yacc.c:1646  */
    break;

  case 250:
#line 880 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[-2].ntype),(yyvsp[0].pupable)); }
#line 3867 "y.tab.c" /* yacc.c:1646  */
    break;

  case 251:
#line 883 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.includeFile) = new IncludeFile(lineno,(yyvsp[0].strval)); }
#line 3873 "y.tab.c" /* yacc.c:1646  */
    break;

  case 252:
#line 887 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].member); }
#line 3879 "y.tab.c" /* yacc.c:1646  */
    break;

  case 253:
#line 891 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].entry); }
#line 3885 "y.tab.c" /* yacc.c:1646  */
    break;

  case 254:
#line 893 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyvsp[0].entry)->tspec = (yyvsp[-1].tvarlist);
                  (yyval.member) = (yyvsp[0].entry);
                }
#line 3894 "y.tab.c" /* yacc.c:1646  */
    break;

  case 255:
#line 898 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[-1].member); }
#line 3900 "y.tab.c" /* yacc.c:1646  */
    break;

  case 256:
#line 900 "xi-grammar.y" /* yacc.c:1646  */
    {
          ERROR("invalid SDAG member",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 3910 "y.tab.c" /* yacc.c:1646  */
    break;

  case 257:
#line 908 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3916 "y.tab.c" /* yacc.c:1646  */
    break;

  case 258:
#line 910 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3922 "y.tab.c" /* yacc.c:1646  */
    break;

  case 259:
#line 912 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3928 "y.tab.c" /* yacc.c:1646  */
    break;

  case 260:
#line 914 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3934 "y.tab.c" /* yacc.c:1646  */
    break;

  case 261:
#line 916 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3940 "y.tab.c" /* yacc.c:1646  */
    break;

  case 262:
#line 918 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3946 "y.tab.c" /* yacc.c:1646  */
    break;

  case 263:
#line 920 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3952 "y.tab.c" /* yacc.c:1646  */
    break;

  case 264:
#line 922 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3958 "y.tab.c" /* yacc.c:1646  */
    break;

  case 265:
#line 924 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3964 "y.tab.c" /* yacc.c:1646  */
    break;

  case 266:
#line 926 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3970 "y.tab.c" /* yacc.c:1646  */
    break;

  case 267:
#line 928 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3976 "y.tab.c" /* yacc.c:1646  */
    break;

  case 268:
#line 931 "xi-grammar.y" /* yacc.c:1646  */
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
#line 3991 "y.tab.c" /* yacc.c:1646  */
    break;

  case 269:
#line 942 "xi-grammar.y" /* yacc.c:1646  */
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
#line 4013 "y.tab.c" /* yacc.c:1646  */
    break;

  case 270:
#line 960 "xi-grammar.y" /* yacc.c:1646  */
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
#line 4033 "y.tab.c" /* yacc.c:1646  */
    break;

  case 271:
#line 978 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.accelBlock) = new AccelBlock(lineno, new XStr((yyvsp[-2].strval))); }
#line 4039 "y.tab.c" /* yacc.c:1646  */
    break;

  case 272:
#line 980 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.accelBlock) = new AccelBlock(lineno, NULL); }
#line 4045 "y.tab.c" /* yacc.c:1646  */
    break;

  case 273:
#line 984 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 4051 "y.tab.c" /* yacc.c:1646  */
    break;

  case 274:
#line 988 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attr) = 0; }
#line 4057 "y.tab.c" /* yacc.c:1646  */
    break;

  case 275:
#line 990 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attr) = (yyvsp[-1].attr); }
#line 4063 "y.tab.c" /* yacc.c:1646  */
    break;

  case 276:
#line 992 "xi-grammar.y" /* yacc.c:1646  */
    { ERROR("invalid entry method attribute list",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
#line 4072 "y.tab.c" /* yacc.c:1646  */
    break;

  case 277:
#line 998 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attrarg) = new Attribute::Argument((yyvsp[-2].strval), atoi((yyvsp[0].strval))); }
#line 4078 "y.tab.c" /* yacc.c:1646  */
    break;

  case 278:
#line 1002 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attrarg) = (yyvsp[0].attrarg); }
#line 4084 "y.tab.c" /* yacc.c:1646  */
    break;

  case 279:
#line 1003 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attrarg) = (yyvsp[-2].attrarg); (yyvsp[-2].attrarg)->next = (yyvsp[0].attrarg); }
#line 4090 "y.tab.c" /* yacc.c:1646  */
    break;

  case 280:
#line 1007 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attr) = new Attribute((yyvsp[0].intval));           }
#line 4096 "y.tab.c" /* yacc.c:1646  */
    break;

  case 281:
#line 1008 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attr) = new Attribute((yyvsp[-3].intval), (yyvsp[-1].attrarg));       }
#line 4102 "y.tab.c" /* yacc.c:1646  */
    break;

  case 282:
#line 1009 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attr) = new Attribute((yyvsp[-2].intval), NULL, (yyvsp[0].attr)); }
#line 4108 "y.tab.c" /* yacc.c:1646  */
    break;

  case 283:
#line 1010 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.attr) = new Attribute((yyvsp[-5].intval), (yyvsp[-3].attrarg), (yyvsp[0].attr));   }
#line 4114 "y.tab.c" /* yacc.c:1646  */
    break;

  case 284:
#line 1014 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = STHREADED; }
#line 4120 "y.tab.c" /* yacc.c:1646  */
    break;

  case 285:
#line 1016 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SWHENIDLE; }
#line 4126 "y.tab.c" /* yacc.c:1646  */
    break;

  case 286:
#line 1018 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SSYNC; }
#line 4132 "y.tab.c" /* yacc.c:1646  */
    break;

  case 287:
#line 1020 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SIGET; }
#line 4138 "y.tab.c" /* yacc.c:1646  */
    break;

  case 288:
#line 1022 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SLOCKED; }
#line 4144 "y.tab.c" /* yacc.c:1646  */
    break;

  case 289:
#line 1024 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SCREATEHERE; }
#line 4150 "y.tab.c" /* yacc.c:1646  */
    break;

  case 290:
#line 1026 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SCREATEHOME; }
#line 4156 "y.tab.c" /* yacc.c:1646  */
    break;

  case 291:
#line 1028 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SNOKEEP; }
#line 4162 "y.tab.c" /* yacc.c:1646  */
    break;

  case 292:
#line 1030 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SINBOX; }
#line 4168 "y.tab.c" /* yacc.c:1646  */
    break;

  case 293:
#line 1032 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SCOROUTINE; }
#line 4174 "y.tab.c" /* yacc.c:1646  */
    break;

  case 294:
#line 1034 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SNOTRACE; }
#line 4180 "y.tab.c" /* yacc.c:1646  */
    break;

  case 295:
#line 1036 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SAPPWORK; }
#line 4186 "y.tab.c" /* yacc.c:1646  */
    break;

  case 296:
#line 1038 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SIMMEDIATE; }
#line 4192 "y.tab.c" /* yacc.c:1646  */
    break;

  case 297:
#line 1040 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SSKIPSCHED; }
#line 4198 "y.tab.c" /* yacc.c:1646  */
    break;

  case 298:
#line 1042 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SINLINE; }
#line 4204 "y.tab.c" /* yacc.c:1646  */
    break;

  case 299:
#line 1044 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SLOCAL; }
#line 4210 "y.tab.c" /* yacc.c:1646  */
    break;

  case 300:
#line 1046 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SPYTHON; }
#line 4216 "y.tab.c" /* yacc.c:1646  */
    break;

  case 301:
#line 1048 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SMEM; }
#line 4222 "y.tab.c" /* yacc.c:1646  */
    break;

  case 302:
#line 1050 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SREDUCE; }
#line 4228 "y.tab.c" /* yacc.c:1646  */
    break;

  case 303:
#line 1052 "xi-grammar.y" /* yacc.c:1646  */
    {
        (yyval.intval) = SAGGREGATE;
    }
#line 4236 "y.tab.c" /* yacc.c:1646  */
    break;

  case 304:
#line 1056 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("invalid entry method attribute",
		        (yylsp[0]).first_column, (yylsp[0]).last_column);
		  yyclearin;
		  yyerrok;
		}
#line 4247 "y.tab.c" /* yacc.c:1646  */
    break;

  case 305:
#line 1065 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4253 "y.tab.c" /* yacc.c:1646  */
    break;

  case 306:
#line 1067 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4259 "y.tab.c" /* yacc.c:1646  */
    break;

  case 307:
#line 1069 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4265 "y.tab.c" /* yacc.c:1646  */
    break;

  case 308:
#line 1073 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = ""; }
#line 4271 "y.tab.c" /* yacc.c:1646  */
    break;

  case 309:
#line 1075 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4277 "y.tab.c" /* yacc.c:1646  */
    break;

  case 310:
#line 1077 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Returned only when in_bracket*/
			int len = strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3;
			char *tmp = new char[len];
			snprintf(tmp,len,"%s, %s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4288 "y.tab.c" /* yacc.c:1646  */
    break;

  case 311:
#line 1086 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = ""; }
#line 4294 "y.tab.c" /* yacc.c:1646  */
    break;

  case 312:
#line 1088 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4300 "y.tab.c" /* yacc.c:1646  */
    break;

  case 313:
#line 1090 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Returned only when in_bracket*/
			int len = strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3;
			char *tmp = new char[len];
			snprintf(tmp, len, "%s[%s]%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4311 "y.tab.c" /* yacc.c:1646  */
    break;

  case 314:
#line 1097 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			int len = strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3;
			char *tmp = new char[len];
			snprintf(tmp, len, "%s{%s}%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4322 "y.tab.c" /* yacc.c:1646  */
    break;

  case 315:
#line 1104 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			int len = strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3;
			char *tmp = new char[len];
			snprintf(tmp, len, "%s(%s)%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4333 "y.tab.c" /* yacc.c:1646  */
    break;

  case 316:
#line 1111 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			int len = strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3;
			char *tmp = new char[len];
			snprintf(tmp, len, "(%s)%s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4344 "y.tab.c" /* yacc.c:1646  */
    break;

  case 317:
#line 1120 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval));
		}
#line 4353 "y.tab.c" /* yacc.c:1646  */
    break;

  case 318:
#line 1127 "xi-grammar.y" /* yacc.c:1646  */
    { 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			(yyval.intval) = 0;
		}
#line 4363 "y.tab.c" /* yacc.c:1646  */
    break;

  case 319:
#line 1135 "xi-grammar.y" /* yacc.c:1646  */
    { 
			in_braces=0;
			(yyval.intval) = 0;
		}
#line 4372 "y.tab.c" /* yacc.c:1646  */
    break;

  case 320:
#line 1142 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[0].type));}
#line 4378 "y.tab.c" /* yacc.c:1646  */
    break;

  case 321:
#line 1144 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval)); (yyval.pname)->setConditional((yyvsp[0].intval)); }
#line 4384 "y.tab.c" /* yacc.c:1646  */
    break;

  case 322:
#line 1146 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[-3].type),(yyvsp[-2].strval),0,(yyvsp[0].val));}
#line 4390 "y.tab.c" /* yacc.c:1646  */
    break;

  case 323:
#line 1148 "xi-grammar.y" /* yacc.c:1646  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
		}
#line 4399 "y.tab.c" /* yacc.c:1646  */
    break;

  case 324:
#line 1153 "xi-grammar.y" /* yacc.c:1646  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
			(yyval.pname)->setRdma(CMK_ZC_P2P_SEND_MSG);
			if(firstRdma) {
				(yyval.pname)->setFirstRdma(true);
				firstRdma = false;
			}
		}
#line 4413 "y.tab.c" /* yacc.c:1646  */
    break;

  case 325:
#line 1163 "xi-grammar.y" /* yacc.c:1646  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
			(yyval.pname)->setRdma(CMK_ZC_P2P_RECV_MSG);
			if(firstRdma) {
				(yyval.pname)->setFirstRdma(true);
				firstRdma = false;
			}
		}
#line 4427 "y.tab.c" /* yacc.c:1646  */
    break;

  case 326:
#line 1173 "xi-grammar.y" /* yacc.c:1646  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
			(yyval.pname)->setRdma(CMK_ZC_DEVICE_MSG);
			if (firstDeviceRdma) {
				(yyval.pname)->setFirstDeviceRdma(true);
				firstDeviceRdma = false;
			}
		}
#line 4441 "y.tab.c" /* yacc.c:1646  */
    break;

  case 327:
#line 1184 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READONLY; }
#line 4447 "y.tab.c" /* yacc.c:1646  */
    break;

  case 328:
#line 1185 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READWRITE; }
#line 4453 "y.tab.c" /* yacc.c:1646  */
    break;

  case 329:
#line 1186 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_WRITEONLY; }
#line 4459 "y.tab.c" /* yacc.c:1646  */
    break;

  case 330:
#line 1189 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr((yyvsp[0].strval)); }
#line 4465 "y.tab.c" /* yacc.c:1646  */
    break;

  case 331:
#line 1190 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "->" << (yyvsp[0].strval); }
#line 4471 "y.tab.c" /* yacc.c:1646  */
    break;

  case 332:
#line 1191 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[-2].xstrptr)) << "." << (yyvsp[0].strval); }
#line 4477 "y.tab.c" /* yacc.c:1646  */
    break;

  case 333:
#line 1193 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << *((yyvsp[-1].xstrptr)) << "]";
                  delete (yyvsp[-3].xstrptr);
                  delete (yyvsp[-1].xstrptr);
                }
#line 4488 "y.tab.c" /* yacc.c:1646  */
    break;

  case 334:
#line 1200 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << (yyvsp[-1].strval) << "]";
                  delete (yyvsp[-3].xstrptr);
                }
#line 4498 "y.tab.c" /* yacc.c:1646  */
    break;

  case 335:
#line 1206 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "(" << *((yyvsp[-1].xstrptr)) << ")";
                  delete (yyvsp[-3].xstrptr);
                  delete (yyvsp[-1].xstrptr);
                }
#line 4509 "y.tab.c" /* yacc.c:1646  */
    break;

  case 336:
#line 1215 "xi-grammar.y" /* yacc.c:1646  */
    {
                  in_bracket = 0;
                  (yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName(), (yyvsp[-1].strval));
                }
#line 4518 "y.tab.c" /* yacc.c:1646  */
    break;

  case 337:
#line 1222 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[-6].intval));
                }
#line 4528 "y.tab.c" /* yacc.c:1646  */
    break;

  case 338:
#line 1228 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType(Parameter::ACCEL_BUFFER_TYPE_READWRITE);
		}
#line 4538 "y.tab.c" /* yacc.c:1646  */
    break;

  case 339:
#line 1234 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.pname) = (yyvsp[-3].pname);
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[-5].intval));
		}
#line 4548 "y.tab.c" /* yacc.c:1646  */
    break;

  case 340:
#line 1242 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 4554 "y.tab.c" /* yacc.c:1646  */
    break;

  case 341:
#line 1244 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 4560 "y.tab.c" /* yacc.c:1646  */
    break;

  case 342:
#line 1248 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 4566 "y.tab.c" /* yacc.c:1646  */
    break;

  case 343:
#line 1250 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 4572 "y.tab.c" /* yacc.c:1646  */
    break;

  case 344:
#line 1254 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = (yyvsp[-1].plist); }
#line 4578 "y.tab.c" /* yacc.c:1646  */
    break;

  case 345:
#line 1256 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList(new Parameter(0, new BuiltinType("void"))); }
#line 4584 "y.tab.c" /* yacc.c:1646  */
    break;

  case 346:
#line 1260 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = (yyvsp[-1].plist); }
#line 4590 "y.tab.c" /* yacc.c:1646  */
    break;

  case 347:
#line 1262 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = 0; }
#line 4596 "y.tab.c" /* yacc.c:1646  */
    break;

  case 348:
#line 1266 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = 0; }
#line 4602 "y.tab.c" /* yacc.c:1646  */
    break;

  case 349:
#line 1268 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4608 "y.tab.c" /* yacc.c:1646  */
    break;

  case 350:
#line 1272 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = 0; }
#line 4614 "y.tab.c" /* yacc.c:1646  */
    break;

  case 351:
#line 1274 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[0].sc)); }
#line 4620 "y.tab.c" /* yacc.c:1646  */
    break;

  case 352:
#line 1276 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[-2].slist)); }
#line 4626 "y.tab.c" /* yacc.c:1646  */
    break;

  case 353:
#line 1280 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.slist) = new SListConstruct((yyvsp[0].sc)); }
#line 4632 "y.tab.c" /* yacc.c:1646  */
    break;

  case 354:
#line 1282 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.slist) = new SListConstruct((yyvsp[-1].sc), (yyvsp[0].slist));  }
#line 4638 "y.tab.c" /* yacc.c:1646  */
    break;

  case 355:
#line 1286 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.olist) = new OListConstruct((yyvsp[0].sc)); }
#line 4644 "y.tab.c" /* yacc.c:1646  */
    break;

  case 356:
#line 1288 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.olist) = new OListConstruct((yyvsp[-1].sc), (yyvsp[0].slist)); }
#line 4650 "y.tab.c" /* yacc.c:1646  */
    break;

  case 357:
#line 1292 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.clist) = new CaseListConstruct((yyvsp[0].when)); }
#line 4656 "y.tab.c" /* yacc.c:1646  */
    break;

  case 358:
#line 1294 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.clist) = new CaseListConstruct((yyvsp[-1].when), (yyvsp[0].clist)); }
#line 4662 "y.tab.c" /* yacc.c:1646  */
    break;

  case 359:
#line 1296 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("case blocks can only contain when clauses",
		        (yylsp[0]).first_column, (yylsp[0]).last_column);
		  (yyval.clist) = 0;
		}
#line 4672 "y.tab.c" /* yacc.c:1646  */
    break;

  case 360:
#line 1304 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4678 "y.tab.c" /* yacc.c:1646  */
    break;

  case 361:
#line 1306 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = 0; }
#line 4684 "y.tab.c" /* yacc.c:1646  */
    break;

  case 362:
#line 1310 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-2].entrylist), 0); }
#line 4690 "y.tab.c" /* yacc.c:1646  */
    break;

  case 363:
#line 1312 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-1].entrylist), (yyvsp[0].sc)); }
#line 4696 "y.tab.c" /* yacc.c:1646  */
    break;

  case 364:
#line 1314 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-3].entrylist), (yyvsp[-1].slist)); }
#line 4702 "y.tab.c" /* yacc.c:1646  */
    break;

  case 365:
#line 1318 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4708 "y.tab.c" /* yacc.c:1646  */
    break;

  case 366:
#line 1320 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4714 "y.tab.c" /* yacc.c:1646  */
    break;

  case 367:
#line 1322 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4720 "y.tab.c" /* yacc.c:1646  */
    break;

  case 368:
#line 1324 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4726 "y.tab.c" /* yacc.c:1646  */
    break;

  case 369:
#line 1326 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4732 "y.tab.c" /* yacc.c:1646  */
    break;

  case 370:
#line 1328 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4738 "y.tab.c" /* yacc.c:1646  */
    break;

  case 371:
#line 1330 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4744 "y.tab.c" /* yacc.c:1646  */
    break;

  case 372:
#line 1332 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4750 "y.tab.c" /* yacc.c:1646  */
    break;

  case 373:
#line 1334 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4756 "y.tab.c" /* yacc.c:1646  */
    break;

  case 374:
#line 1336 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4762 "y.tab.c" /* yacc.c:1646  */
    break;

  case 375:
#line 1338 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4768 "y.tab.c" /* yacc.c:1646  */
    break;

  case 376:
#line 1340 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4774 "y.tab.c" /* yacc.c:1646  */
    break;

  case 377:
#line 1344 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new SerialConstruct((yyvsp[-2].strval), (yyvsp[-4].strval), (yylsp[-3]).first_line); }
#line 4780 "y.tab.c" /* yacc.c:1646  */
    break;

  case 378:
#line 1346 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new OverlapConstruct((yyvsp[-1].olist)); }
#line 4786 "y.tab.c" /* yacc.c:1646  */
    break;

  case 379:
#line 1348 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = (yyvsp[0].when); }
#line 4792 "y.tab.c" /* yacc.c:1646  */
    break;

  case 380:
#line 1350 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new CaseConstruct((yyvsp[-1].clist)); }
#line 4798 "y.tab.c" /* yacc.c:1646  */
    break;

  case 381:
#line 1352 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForConstruct((yyvsp[-8].intexpr), (yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4804 "y.tab.c" /* yacc.c:1646  */
    break;

  case 382:
#line 1354 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForConstruct((yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4810 "y.tab.c" /* yacc.c:1646  */
    break;

  case 383:
#line 1356 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-9].strval)), (yyvsp[-6].intexpr),
		             (yyvsp[-4].intexpr), (yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4817 "y.tab.c" /* yacc.c:1646  */
    break;

  case 384:
#line 1359 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-11].strval)), (yyvsp[-8].intexpr),
		             (yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4824 "y.tab.c" /* yacc.c:1646  */
    break;

  case 385:
#line 1362 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new IfConstruct((yyvsp[-3].intexpr), (yyvsp[-1].sc), (yyvsp[0].sc)); }
#line 4830 "y.tab.c" /* yacc.c:1646  */
    break;

  case 386:
#line 1364 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new IfConstruct((yyvsp[-5].intexpr), (yyvsp[-2].slist), (yyvsp[0].sc)); }
#line 4836 "y.tab.c" /* yacc.c:1646  */
    break;

  case 387:
#line 1366 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new WhileConstruct((yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4842 "y.tab.c" /* yacc.c:1646  */
    break;

  case 388:
#line 1368 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new WhileConstruct((yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4848 "y.tab.c" /* yacc.c:1646  */
    break;

  case 389:
#line 1370 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new SerialConstruct((yyvsp[-2].strval), NULL, (yyloc).first_line); }
#line 4854 "y.tab.c" /* yacc.c:1646  */
    break;

  case 390:
#line 1372 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("unknown SDAG construct or malformed entry method declaration.\n"
		        "You may have forgotten to terminate a previous entry method declaration with a"
		        " semicolon or forgotten to mark a block of sequential SDAG code as 'serial'",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
#line 4866 "y.tab.c" /* yacc.c:1646  */
    break;

  case 391:
#line 1382 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = 0; }
#line 4872 "y.tab.c" /* yacc.c:1646  */
    break;

  case 392:
#line 1384 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ElseConstruct((yyvsp[0].sc)); }
#line 4878 "y.tab.c" /* yacc.c:1646  */
    break;

  case 393:
#line 1386 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ElseConstruct((yyvsp[-1].slist)); }
#line 4884 "y.tab.c" /* yacc.c:1646  */
    break;

  case 394:
#line 1390 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intexpr) = new IntExprConstruct((yyvsp[0].strval)); }
#line 4890 "y.tab.c" /* yacc.c:1646  */
    break;

  case 395:
#line 1394 "xi-grammar.y" /* yacc.c:1646  */
    { in_int_expr = 0; (yyval.intval) = 0; }
#line 4896 "y.tab.c" /* yacc.c:1646  */
    break;

  case 396:
#line 1398 "xi-grammar.y" /* yacc.c:1646  */
    { in_int_expr = 1; (yyval.intval) = 0; }
#line 4902 "y.tab.c" /* yacc.c:1646  */
    break;

  case 397:
#line 1402 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.entry) = new Entry(lineno, NULL, 0, (yyvsp[-1].strval), (yyvsp[0].plist), 0, 0, 0, (yyloc).first_line, (yyloc).last_line);
		  firstRdma = true;
		  firstDeviceRdma = true;
		}
#line 4912 "y.tab.c" /* yacc.c:1646  */
    break;

  case 398:
#line 1408 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.entry) = new Entry(lineno, NULL, 0, (yyvsp[-4].strval), (yyvsp[0].plist), 0, 0, (yyvsp[-2].strval), (yyloc).first_line, (yyloc).last_line);
		  firstRdma = true;
		  firstDeviceRdma = true;
		}
#line 4922 "y.tab.c" /* yacc.c:1646  */
    break;

  case 399:
#line 1416 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.entrylist) = new EntryList((yyvsp[0].entry)); }
#line 4928 "y.tab.c" /* yacc.c:1646  */
    break;

  case 400:
#line 1418 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.entrylist) = new EntryList((yyvsp[-2].entry),(yyvsp[0].entrylist)); }
#line 4934 "y.tab.c" /* yacc.c:1646  */
    break;

  case 401:
#line 1422 "xi-grammar.y" /* yacc.c:1646  */
    { in_bracket=1; }
#line 4940 "y.tab.c" /* yacc.c:1646  */
    break;

  case 402:
#line 1425 "xi-grammar.y" /* yacc.c:1646  */
    { in_bracket=0; }
#line 4946 "y.tab.c" /* yacc.c:1646  */
    break;

  case 403:
#line 1429 "xi-grammar.y" /* yacc.c:1646  */
    { if (!macroDefined((yyvsp[0].strval), 1)) in_comment = 1; }
#line 4952 "y.tab.c" /* yacc.c:1646  */
    break;

  case 404:
#line 1433 "xi-grammar.y" /* yacc.c:1646  */
    { if (!macroDefined((yyvsp[0].strval), 0)) in_comment = 1; }
#line 4958 "y.tab.c" /* yacc.c:1646  */
    break;


#line 4962 "y.tab.c" /* yacc.c:1646  */
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
#line 1436 "xi-grammar.y" /* yacc.c:1906  */


void yyerror(const char *s) 
{
	fprintf(stderr, "[PARSE-ERROR] Unexpected/missing token at line %d. Current token being parsed: '%s'.\n", lineno, yytext);
}
