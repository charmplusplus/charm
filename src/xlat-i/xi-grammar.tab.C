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
void ReservedWord(int token, int fCol, int lCol);
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
    AGGREGATE = 286,
    CREATEHERE = 287,
    CREATEHOME = 288,
    NOKEEP = 289,
    NOTRACE = 290,
    APPWORK = 291,
    VOID = 292,
    CONST = 293,
    RDMA = 294,
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
    CASE = 329
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
#define RDMA 294
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

#line 345 "y.tab.c" /* yacc.c:355  */
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

#line 374 "y.tab.c" /* yacc.c:358  */

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
#define YYLAST   1516

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  91
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  117
/* YYNRULES -- Number of rules.  */
#define YYNRULES  370
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  725

/* YYTRANSLATE[YYX] -- Symbol number corresponding to YYX as returned
   by yylex, with out-of-bounds checking.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   329

#define YYTRANSLATE(YYX)                                                \
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, without out-of-bounds checking.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    85,     2,
      83,    84,    82,     2,    79,    89,    90,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    76,    75,
      80,    88,    81,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    86,     2,    87,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    77,     2,    78,     2,     2,     2,     2,
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
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74
};

#if YYDEBUG
  /* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   193,   193,   198,   201,   206,   207,   211,   213,   218,
     219,   224,   226,   227,   228,   230,   231,   232,   234,   235,
     236,   237,   238,   242,   243,   244,   245,   246,   247,   248,
     249,   250,   251,   252,   253,   254,   255,   256,   257,   258,
     259,   262,   263,   264,   265,   266,   267,   268,   269,   270,
     271,   272,   274,   276,   277,   280,   281,   282,   283,   287,
     289,   296,   300,   307,   309,   314,   315,   319,   321,   323,
     325,   327,   339,   341,   343,   345,   351,   353,   355,   357,
     359,   361,   363,   365,   367,   369,   377,   379,   381,   385,
     387,   392,   393,   398,   399,   403,   405,   407,   409,   411,
     413,   415,   417,   419,   421,   423,   425,   427,   429,   431,
     435,   436,   443,   445,   449,   453,   455,   459,   463,   465,
     467,   469,   471,   473,   477,   479,   481,   483,   485,   489,
     491,   495,   497,   501,   505,   510,   511,   515,   519,   524,
     525,   530,   531,   541,   543,   547,   549,   554,   555,   559,
     561,   566,   567,   571,   576,   577,   581,   583,   587,   589,
     594,   595,   599,   600,   603,   607,   609,   613,   615,   620,
     621,   625,   627,   631,   633,   637,   641,   645,   651,   655,
     657,   661,   663,   667,   671,   675,   679,   681,   686,   687,
     692,   693,   695,   697,   706,   708,   710,   714,   716,   720,
     724,   726,   728,   730,   732,   736,   738,   743,   750,   754,
     756,   758,   759,   761,   763,   765,   769,   771,   773,   779,
     785,   794,   796,   798,   804,   812,   814,   817,   821,   825,
     827,   832,   834,   842,   844,   846,   848,   850,   852,   854,
     856,   858,   860,   862,   865,   874,   890,   906,   908,   912,
     917,   918,   920,   927,   929,   933,   935,   937,   939,   941,
     943,   945,   947,   949,   951,   953,   955,   957,   959,   961,
     963,   965,   977,   986,   988,   990,   995,   996,   998,  1007,
    1008,  1010,  1016,  1022,  1028,  1036,  1043,  1051,  1058,  1060,
    1062,  1064,  1069,  1077,  1078,  1079,  1082,  1083,  1084,  1085,
    1092,  1098,  1107,  1114,  1120,  1126,  1134,  1136,  1140,  1142,
    1146,  1148,  1152,  1154,  1159,  1160,  1164,  1166,  1168,  1172,
    1174,  1178,  1180,  1184,  1186,  1188,  1196,  1199,  1202,  1204,
    1206,  1210,  1212,  1214,  1216,  1218,  1220,  1222,  1224,  1226,
    1228,  1230,  1232,  1236,  1238,  1240,  1242,  1244,  1246,  1248,
    1251,  1254,  1256,  1258,  1260,  1262,  1264,  1275,  1276,  1278,
    1282,  1286,  1290,  1294,  1298,  1304,  1306,  1310,  1313,  1317,
    1321
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
  "VOID", "CONST", "RDMA", "PACKED", "VARSIZE", "ENTRY", "FOR", "FORALL",
  "WHILE", "WHEN", "OVERLAP", "SERIAL", "IF", "ELSE", "PYTHON", "LOCAL",
  "NAMESPACE", "USING", "IDENT", "NUMBER", "LITERAL", "CPROGRAM", "HASHIF",
  "HASHIFDEF", "INT", "LONG", "SHORT", "CHAR", "FLOAT", "DOUBLE",
  "UNSIGNED", "ACCEL", "READWRITE", "WRITEONLY", "ACCELBLOCK",
  "MEMCRITICAL", "REDUCTIONTARGET", "CASE", "';'", "':'", "'{'", "'}'",
  "','", "'<'", "'>'", "'*'", "'('", "')'", "'&'", "'['", "']'", "'='",
  "'-'", "'.'", "$accept", "File", "ModuleEList", "OptExtern",
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
     325,   326,   327,   328,   329,    59,    58,   123,   125,    44,
      60,    62,    42,    40,    41,    38,    91,    93,    61,    45,
      46
};
# endif

#define YYPACT_NINF -617

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-617)))

#define YYTABLE_NINF -322

#define yytable_value_is_error(Yytable_value) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
     150,  1350,  1350,    50,  -617,   150,  -617,  -617,  -617,  -617,
    -617,  -617,  -617,  -617,  -617,  -617,  -617,  -617,  -617,  -617,
    -617,  -617,  -617,  -617,  -617,  -617,  -617,  -617,  -617,  -617,
    -617,  -617,  -617,  -617,  -617,  -617,  -617,  -617,  -617,  -617,
    -617,  -617,  -617,  -617,  -617,  -617,  -617,  -617,  -617,  -617,
    -617,  -617,  -617,  -617,   106,   106,  -617,  -617,  -617,   767,
     -38,  -617,  -617,  -617,    49,  1350,   185,  1350,  1350,   172,
     938,    48,   479,   767,  -617,  -617,  -617,  -617,   786,    52,
      88,  -617,    90,  -617,  -617,  -617,   -38,    37,  1371,   140,
     140,     8,    88,    98,    98,    98,    98,   103,   109,  1350,
     135,   122,   767,  -617,  -617,  -617,  -617,  -617,  -617,  -617,
    -617,   620,  -617,  -617,  -617,  -617,   133,  -617,  -617,  -617,
    -617,  -617,  -617,  -617,  -617,  -617,  -617,  -617,   -38,  -617,
    -617,  -617,   786,  -617,    -7,  -617,  -617,  -617,  -617,   166,
      91,  -617,  -617,   134,   157,   170,     0,  -617,    88,   767,
      90,   192,    87,    37,   193,  1449,   936,   134,   157,   170,
    -617,   -10,    88,  -617,    88,    88,   206,    88,   197,  -617,
     -11,  1350,  1350,  1350,  1350,  1134,   195,   196,   179,  1350,
    -617,  -617,  -617,  1392,   200,    98,    98,    98,    98,   195,
     109,  -617,  -617,  -617,  -617,  -617,   -38,  -617,   241,  -617,
    -617,  -617,   173,  -617,  -617,   859,  -617,  -617,  -617,  -617,
    -617,  -617,  1350,   221,   251,    37,   247,    37,   223,  -617,
     133,   226,    14,  -617,   228,  -617,    45,   -29,    61,   225,
      99,    88,  -617,  -617,   230,   236,   238,   244,   244,   244,
     244,  -617,  1350,   234,   253,   237,  1206,  1350,   283,  1350,
    -617,  -617,   255,   259,   269,  1350,    51,  1350,   272,   267,
     133,  1350,  1350,  1350,  1350,  1350,  1350,  -617,  -617,  -617,
    -617,   279,  -617,   281,  -617,   238,  -617,  -617,   285,   289,
     266,   277,    37,   -38,    88,  1350,  -617,   284,  -617,    37,
     140,   859,   140,   140,   859,   140,  -617,  -617,   -11,  -617,
      88,   178,   178,   178,   178,   282,  -617,   283,  -617,   244,
     244,  -617,   179,   354,   291,   168,  -617,   293,  1392,  -617,
    -617,   244,   244,   244,   244,   244,   182,   859,  -617,   299,
      37,   247,    37,    37,  -617,    45,   301,  -617,   296,  -617,
     302,   307,   312,    88,   316,   314,  -617,   321,  -617,   518,
     -38,  -617,  -617,  -617,  -617,  -617,  -617,   178,   178,  -617,
    -617,   936,    12,   323,   936,  -617,  -617,  -617,  -617,  -617,
    -617,   178,   178,   178,   178,   178,   354,   -38,  -617,  1405,
    -617,  -617,  -617,  -617,  -617,  -617,   319,  -617,  -617,  -617,
     325,  -617,    89,   326,  -617,    88,  -617,   683,   368,   333,
     133,   518,  -617,  -617,  -617,  -617,  1350,  -617,  -617,  -617,
    -617,  -617,  -617,  -617,  -617,   335,   936,  -617,  1350,    37,
     336,   330,  1436,   140,   140,   140,  -617,  -617,   954,  1062,
    -617,   133,   -38,  -617,   332,   133,  1350,    37,    25,   338,
    1436,  -617,   342,   343,   345,   347,  -617,  -617,  -617,  -617,
    -617,  -617,  -617,  -617,  -617,  -617,  -617,  -617,  -617,  -617,
     359,  -617,   349,  -617,  -617,   350,   353,   339,   299,  1350,
    -617,   351,   133,   -38,   348,   370,  -617,   240,  -617,  -617,
    -617,  -617,  -617,  -617,  -617,  -617,  -617,   396,  -617,  1007,
     544,   299,  -617,   -38,  -617,  -617,  -617,    90,  -617,  1350,
    -617,  -617,   357,   369,   357,   403,   383,   404,   357,   385,
     329,   -38,    37,  -617,  -617,  -617,   443,   299,  -617,    37,
     412,    37,    41,   390,   580,   602,  -617,   393,    37,   405,
     395,   245,   193,   389,   544,   392,  -617,   414,   384,   416,
    -617,    37,   403,   398,  -617,   402,   346,    37,   416,   357,
     410,   357,   424,   404,   357,   427,    37,   428,   405,  -617,
     133,  -617,   133,   449,  -617,   510,   393,    37,   357,  -617,
     751,   296,  -617,  -617,   429,  -617,  -617,   193,   758,    37,
     453,    37,   602,   393,    37,   405,   193,  -617,  -617,  -617,
    -617,  -617,  -617,  -617,  -617,  -617,  1350,   434,   433,   426,
      37,   439,    37,   329,  -617,   299,  -617,   133,   329,   465,
     442,   436,   416,   451,    37,   416,   452,   133,   454,   936,
     603,  -617,   193,    37,   455,   457,  -617,  -617,   463,   765,
    -617,    37,   357,   814,  -617,   193,   821,  -617,  -617,  1350,
    1350,    37,   462,  -617,  1350,   416,    37,  -617,   465,   329,
    -617,   458,    37,   329,  -617,   133,   329,   465,  -617,   107,
      71,   456,  1350,   133,   828,   467,  -617,   466,    37,   474,
     473,  -617,   475,  -617,  -617,  1350,  1278,   471,  1350,  1350,
    -617,   250,   -38,   329,  -617,    37,  -617,   416,    37,  -617,
     465,   115,   468,   183,  1350,  -617,   271,  -617,   480,   416,
     835,   482,  -617,  -617,  -617,  -617,  -617,  -617,  -617,   884,
     329,  -617,    37,   329,  -617,   484,   416,   485,  -617,   891,
    -617,   329,  -617,   486,  -617
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
      63,    61,    62,    85,     6,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    84,    82,    83,     8,     0,     0,
       0,    59,    68,   369,   370,   286,   248,   279,     0,   139,
     139,   139,     0,   147,   147,   147,   147,     0,   141,     0,
       0,     0,     0,    76,   209,   210,    70,    77,    78,    79,
      80,     0,    81,    69,   212,   211,     9,   243,   235,   236,
     237,   238,   239,   241,   242,   240,   233,   234,    74,    75,
      66,   109,     0,    95,    96,    97,    98,   106,   107,     0,
      93,   112,   113,   124,   125,   126,   130,   249,     0,     0,
      67,     0,   280,   279,     0,     0,     0,   118,   119,   120,
     121,   132,     0,   140,     0,     0,     0,     0,   225,   213,
       0,     0,     0,     0,     0,     0,     0,   154,     0,     0,
     215,   227,   214,     0,     0,   147,   147,   147,   147,     0,
     141,   200,   201,   202,   203,   204,    10,    64,   127,   105,
     108,    99,   100,   103,   104,    91,   111,   114,   115,   116,
     128,   129,     0,     0,     0,   279,   276,   279,     0,   287,
       0,     0,   122,   123,     0,   131,   135,   219,   216,     0,
     221,     0,   158,   159,     0,   149,    93,   169,   169,   169,
     169,   153,     0,     0,   156,     0,     0,     0,     0,     0,
     145,   146,     0,   143,   167,     0,   121,     0,   197,     0,
       9,     0,     0,     0,     0,     0,     0,   101,   102,    87,
      88,    89,    92,     0,    86,    93,    73,    60,     0,   277,
       0,     0,   279,   247,     0,     0,   367,   135,   137,   279,
     139,     0,   139,   139,     0,   139,   226,   148,     0,   110,
       0,     0,     0,     0,     0,     0,   178,     0,   155,   169,
     169,   142,     0,   160,   188,     0,   195,   190,     0,   199,
      72,   169,   169,   169,   169,   169,     0,     0,    94,     0,
     279,   276,   279,   279,   284,   135,     0,   136,     0,   133,
       0,     0,     0,     0,     0,     0,   150,   171,   170,     0,
     205,   173,   174,   175,   176,   177,   157,     0,     0,   144,
     161,     0,   160,     0,     0,   194,   191,   192,   193,   196,
     198,     0,     0,     0,     0,     0,   160,   186,    90,     0,
      71,   282,   278,   283,   281,   138,     0,   368,   134,   220,
       0,   217,     0,     0,   222,     0,   232,     0,     0,     0,
       0,     0,   228,   229,   179,   180,     0,   166,   168,   189,
     181,   182,   183,   184,   185,     0,     0,   311,   288,   279,
     306,     0,     0,   139,   139,   139,   172,   252,     0,     0,
     230,     9,   231,   208,   162,     0,     0,   279,   160,     0,
       0,   310,     0,     0,     0,     0,   272,   255,   256,   257,
     258,   264,   265,   266,   271,   259,   260,   261,   262,   263,
     151,   267,     0,   269,   270,     0,   253,    59,     0,     0,
     206,     0,     0,   187,     0,     0,   285,     0,   289,   291,
     307,   117,   218,   224,   223,   152,   268,     0,   251,     0,
       0,     0,   163,   164,   292,   274,   273,   275,   290,     0,
     254,   356,     0,     0,     0,     0,     0,   327,     0,     0,
       0,   316,   279,   245,   345,   317,   314,     0,   362,   279,
       0,   279,     0,   365,     0,     0,   326,     0,   279,     0,
       0,     0,     0,     0,     0,     0,   360,     0,     0,     0,
     363,   279,     0,     0,   329,     0,     0,   279,     0,     0,
       0,     0,     0,   327,     0,     0,   279,     0,   323,   325,
       9,   320,     9,     0,   244,     0,     0,   279,     0,   361,
       0,     0,   366,   328,     0,   344,   322,     0,     0,   279,
       0,   279,     0,     0,   279,     0,     0,   346,   324,   318,
     355,   315,   293,   294,   295,   313,     0,     0,   308,     0,
     279,     0,   279,     0,   353,     0,   330,     9,     0,   357,
       0,     0,     0,     0,   279,     0,     0,     9,     0,     0,
       0,   312,     0,   279,     0,     0,   364,   343,     0,     0,
     351,   279,     0,     0,   332,     0,     0,   333,   342,     0,
       0,   279,     0,   309,     0,     0,   279,   354,   357,     0,
     358,     0,   279,     0,   340,     9,     0,   357,   296,     0,
       0,     0,     0,     0,     0,     0,   352,     0,   279,     0,
       0,   331,     0,   338,   304,     0,     0,     0,     0,     0,
     302,     0,   246,     0,   348,   279,   359,     0,   279,   341,
     357,     0,     0,     0,     0,   298,     0,   305,     0,     0,
       0,     0,   339,   301,   300,   299,   297,   303,   347,     0,
       0,   335,   279,     0,   349,     0,     0,     0,   334,     0,
     350,     0,   336,     0,   337
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -617,  -617,   562,  -617,   -51,  -251,    -1,   -58,   498,   514,
     -39,  -617,  -617,  -617,  -179,  -617,  -175,  -617,   -86,   -79,
     -76,   -64,   -62,  -171,   422,   450,  -617,   -81,  -617,  -617,
    -257,  -617,  -617,   -80,   394,   273,  -617,   -69,   300,  -617,
    -617,   406,   287,  -617,   161,  -617,  -617,  -319,  -617,     4,
     205,  -617,  -617,  -617,  -145,  -617,  -617,  -617,  -617,  -617,
    -617,  -617,   288,  -617,   290,   531,  -617,   -71,   201,   537,
    -617,  -617,   379,  -617,  -617,  -617,  -617,   213,  -617,   184,
    -617,   125,  -617,  -617,   286,   -82,  -401,   -63,  -497,  -617,
    -617,  -517,  -617,  -617,  -389,    -5,  -445,  -617,  -617,    82,
    -507,    38,  -435,    69,  -509,  -617,  -442,  -616,  -468,  -531,
    -446,  -617,    95,   120,    73,  -617,  -617
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     3,     4,    70,   350,   197,   236,   140,     5,    61,
      71,    72,    73,   271,   272,   273,   206,   141,   237,   142,
     157,   158,   159,   160,   161,   146,   147,   274,   338,   287,
     288,   104,   105,   164,   179,   252,   253,   171,   234,   486,
     244,   176,   245,   235,   361,   472,   362,   363,   106,   301,
     348,   107,   108,   109,   177,   110,   191,   192,   193,   194,
     195,   365,   316,   258,   259,   398,   112,   351,   399,   400,
     114,   115,   169,   182,   401,   402,   129,   403,    74,   148,
     429,   465,   466,   498,   280,   536,   419,   512,   220,   420,
     597,   659,   642,   598,   421,   599,   380,   566,   534,   513,
     530,   545,   557,   527,   514,   559,   531,   630,   537,   570,
     519,   523,   524,   289,   388,    75,    76
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[] =
{
      54,    55,   143,    60,    60,   154,    87,   162,    82,   320,
     165,   167,   256,   168,   144,   437,   145,   578,    86,   232,
     558,   128,   150,   490,   561,   172,   173,   174,   223,   360,
     337,   247,   666,   442,   130,   562,   574,    77,   210,   576,
     233,   673,   360,   407,   265,   163,   516,   151,   515,   558,
      56,   480,   223,   539,   290,   199,   143,   415,   521,   200,
     548,   299,   528,   184,    79,   196,    83,    84,   144,   212,
     145,   218,   535,   224,   702,   225,   558,   540,   385,   221,
     607,   633,   544,   546,   636,   211,   238,   239,   240,   617,
    -165,    78,   515,   254,   166,   152,   625,   224,   180,   601,
     329,   628,   257,   579,   226,   581,   227,   228,   584,   230,
     213,   476,   341,   477,   664,   344,   261,   262,   263,   264,
     153,   151,   602,   588,   379,   644,   116,   286,   604,   149,
    -190,   286,  -190,   278,   624,   281,   609,   151,   655,   315,
     546,   291,   667,    81,   292,   681,   670,   256,   378,   672,
     616,   679,   168,     1,     2,   645,   700,   476,   691,   693,
     626,   309,   696,   310,   215,   151,   151,   151,   709,   283,
     216,   205,   424,   217,   243,   151,   698,   163,   665,   294,
     470,    58,   295,    59,   170,   719,   652,   650,   674,   175,
     675,   654,   181,   676,   657,   178,   677,   678,   675,   703,
     334,   676,   183,   715,   677,   678,   717,   339,    58,   196,
     340,   275,   342,   343,   723,   345,   207,   699,   641,   250,
     251,   347,   684,    81,   366,   367,   335,   201,   202,   203,
     204,   352,   353,   354,   267,   268,   368,   257,    80,   208,
      81,   305,   302,   303,   304,   243,   501,    58,   381,    85,
     383,   384,   209,    58,   314,   349,   317,    58,   711,   376,
     321,   322,   323,   324,   325,   326,   675,   714,   214,   676,
     705,   219,   677,   678,   229,   377,   231,   722,   260,   210,
     406,   246,   248,   409,   336,   392,   404,   405,   502,   503,
     504,   505,   506,   507,   508,    81,   495,   496,   418,   276,
     410,   411,   412,   413,   414,   279,   277,   282,   284,   589,
     285,   590,   293,   357,   358,   298,   347,   297,   205,   509,
     300,   306,    85,  -319,   308,   371,   372,   373,   374,   375,
     501,   697,   307,   675,   241,   436,   676,   439,   312,   677,
     678,   418,   311,   443,   444,   445,   313,   501,   319,   432,
     332,   318,   707,   143,   675,   475,   627,   676,   327,   418,
     677,   678,   328,   330,   333,   144,   638,   145,   331,   355,
     286,   360,   502,   503,   504,   505,   506,   507,   508,   364,
     196,   315,   379,   387,   473,   386,   389,  -286,   390,   502,
     503,   504,   505,   506,   507,   508,   391,   393,   394,   501,
     395,   408,   422,   509,   671,   434,    85,  -286,   423,   425,
     397,   431,  -286,   435,   441,   440,   485,   438,   471,   497,
     509,   493,   -11,    85,  -321,   479,   481,   482,   468,   483,
     532,   484,   489,   499,   476,   474,   487,   488,   492,   511,
     518,   502,   503,   504,   505,   506,   507,   508,   549,   550,
     551,   505,   552,   553,   554,   520,  -286,   494,   522,   571,
     525,   526,   529,   533,   547,   577,   556,   538,   491,   542,
      85,   568,   509,   560,   586,    85,   573,   563,   565,   555,
     575,  -286,    85,   511,   596,   117,   118,   119,   120,   567,
     121,   122,   123,   124,   125,   556,   580,   610,   517,   612,
     569,   582,   615,   600,   585,   591,   587,   606,   611,   196,
     619,   196,   620,   621,   623,   629,   592,   631,   622,   396,
     614,   126,   556,   632,    88,    89,    90,    91,    92,   634,
     637,   646,   635,   668,   639,   647,    99,   100,   640,   596,
     101,   648,   662,   680,   686,   501,   685,   131,   156,   651,
     688,   689,   694,   690,    58,   704,   196,   127,   708,   661,
     397,   712,   718,   720,   724,    81,   196,    57,   103,    62,
     669,   133,   134,   135,   136,   137,   138,   139,   222,   593,
     594,   501,   198,   249,   266,   359,   687,   502,   503,   504,
     505,   506,   507,   508,   356,   618,  -207,   595,   346,   478,
     426,   111,   433,   501,   196,   369,   701,   113,   370,   592,
     296,   430,   682,   469,   500,   643,   564,   382,   509,    58,
     613,   510,   583,   502,   503,   504,   505,   506,   507,   508,
     716,   185,   186,   187,   188,   189,   190,   572,   658,   660,
     131,   156,   541,   663,   605,   502,   503,   504,   505,   506,
     507,   508,     0,     0,   509,     0,     0,   543,    81,     0,
       0,   658,     0,     0,   133,   134,   135,   136,   137,   138,
     139,     0,   593,   594,   658,   658,   509,   695,   658,    85,
       0,     0,     0,     0,   427,     0,  -250,  -250,  -250,     0,
    -250,  -250,  -250,   706,  -250,  -250,  -250,  -250,  -250,     0,
       0,     0,  -250,  -250,  -250,  -250,  -250,  -250,  -250,  -250,
    -250,  -250,  -250,  -250,     0,  -250,  -250,  -250,  -250,  -250,
    -250,  -250,  -250,  -250,  -250,  -250,  -250,  -250,  -250,  -250,
    -250,  -250,  -250,  -250,     0,  -250,     0,  -250,  -250,     0,
       0,     0,     0,     0,  -250,  -250,  -250,  -250,  -250,  -250,
    -250,  -250,   501,     0,  -250,  -250,  -250,  -250,     0,   501,
       0,     0,     0,     0,     0,     0,   501,     0,    63,   428,
      -5,    -5,    64,    -5,    -5,    -5,    -5,    -5,    -5,    -5,
      -5,    -5,    -5,    -5,     0,    -5,    -5,     0,     0,    -5,
       0,     0,     0,     0,   502,   503,   504,   505,   506,   507,
     508,   502,   503,   504,   505,   506,   507,   508,   502,   503,
     504,   505,   506,   507,   508,   501,     0,     0,     0,     0,
      65,    66,   501,   131,   132,   509,    67,    68,   603,   501,
       0,     0,   509,     0,     0,   608,   501,     0,    69,   509,
       0,    81,   649,     0,    -5,   -65,     0,   133,   134,   135,
     136,   137,   138,   139,     0,     0,     0,   502,   503,   504,
     505,   506,   507,   508,   502,   503,   504,   505,   506,   507,
     508,   502,   503,   504,   505,   506,   507,   508,   502,   503,
     504,   505,   506,   507,   508,   501,     0,     0,   509,     0,
       0,   653,   501,     0,     0,   509,   131,   156,   656,     0,
       0,     0,   509,     0,     0,   683,     0,     0,     0,   509,
       0,     0,   710,     0,    81,   269,   270,     0,     0,     0,
     133,   134,   135,   136,   137,   138,   139,   502,   503,   504,
     505,   506,   507,   508,   502,   503,   504,   505,   506,   507,
     508,     1,     2,     0,    88,    89,    90,    91,    92,    93,
      94,    95,    96,    97,    98,   446,    99,   100,   509,     0,
     101,   713,     0,     0,     0,   509,     0,     0,   721,     0,
       0,     0,     0,   131,   156,   447,     0,   448,   449,   450,
     451,   452,   453,     0,     0,   454,   455,   456,   457,   458,
     459,    81,     0,     0,     0,     0,     0,   133,   134,   135,
     136,   137,   138,   139,     0,   460,   461,     0,   446,     0,
       0,     0,     0,     0,     0,   102,     0,     0,     0,     0,
       0,     0,   462,     0,     0,     0,   463,   464,   447,     0,
     448,   449,   450,   451,   452,   453,     0,     0,   454,   455,
     456,   457,   458,   459,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   460,   461,
       0,     0,     0,     0,     0,     6,     7,     8,     0,     9,
      10,    11,     0,    12,    13,    14,    15,    16,     0,   463,
     464,    17,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,     0,    29,    30,    31,    32,    33,   131,
     132,    34,    35,    36,    37,    38,    39,    40,    41,    42,
      43,    44,    45,     0,    46,     0,    47,   467,     0,     0,
       0,     0,     0,   133,   134,   135,   136,   137,   138,   139,
      49,     0,     0,    50,    51,    52,    53,     6,     7,     8,
       0,     9,    10,    11,     0,    12,    13,    14,    15,    16,
       0,     0,     0,    17,    18,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,     0,    29,    30,    31,    32,
      33,     0,     0,    34,    35,    36,    37,    38,    39,    40,
      41,    42,    43,    44,    45,   241,    46,     0,    47,    48,
     242,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    49,     0,     0,    50,    51,    52,    53,     6,
       7,     8,     0,     9,    10,    11,     0,    12,    13,    14,
      15,    16,     0,     0,     0,    17,    18,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,     0,    29,    30,
      31,    32,    33,     0,     0,    34,    35,    36,    37,    38,
      39,    40,    41,    42,    43,    44,    45,     0,    46,     0,
      47,    48,   242,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    49,     0,     0,    50,    51,    52,
      53,     6,     7,     8,     0,     9,    10,    11,     0,    12,
      13,    14,    15,    16,     0,     0,     0,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,     0,
      29,    30,    31,    32,    33,     0,     0,    34,    35,    36,
      37,    38,    39,    40,    41,    42,    43,    44,    45,     0,
      46,     0,    47,    48,   692,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    49,     0,     0,    50,
      51,    52,    53,     6,     7,     8,     0,     9,    10,    11,
       0,    12,    13,    14,    15,    16,     0,     0,     0,    17,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      28,     0,    29,    30,    31,    32,    33,   155,     0,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,     0,    46,     0,    47,    48,     0,     0,   131,   156,
     255,     0,     0,     0,     0,     0,     0,     0,    49,     0,
       0,    50,    51,    52,    53,     0,    81,     0,     0,   131,
     156,     0,   133,   134,   135,   136,   137,   138,   139,     0,
       0,     0,   131,   156,   416,     0,     0,    81,     0,     0,
       0,     0,     0,   133,   134,   135,   136,   137,   138,   139,
      81,     0,     0,     0,     0,     0,   133,   134,   135,   136,
     137,   138,   139,   131,   156,   416,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   131,     0,     0,   417,
       0,    81,     0,     0,     0,     0,     0,   133,   134,   135,
     136,   137,   138,   139,    81,     0,     0,     0,     0,     0,
     133,   134,   135,   136,   137,   138,   139
};

static const yytype_int16 yycheck[] =
{
       1,     2,    78,    54,    55,    87,    69,    88,    66,   260,
      90,    91,   183,    92,    78,   416,    78,   548,    69,    30,
     529,    72,    80,   468,   531,    94,    95,    96,    38,    17,
     287,   176,   648,   422,    73,   532,   543,    75,    38,   546,
      51,   657,    17,   362,   189,    37,   491,    76,   490,   558,
       0,   440,    38,   521,    83,    62,   132,   376,   504,    66,
     528,   236,   508,   102,    65,   116,    67,    68,   132,   148,
     132,   153,   517,    83,   690,    85,   585,   522,   335,   155,
     577,   612,   524,   525,   615,    85,   172,   173,   174,   586,
      78,    42,   534,   179,    86,    58,   603,    83,    99,   567,
     275,   608,   183,   549,   162,   551,   164,   165,   554,   167,
     149,    86,   291,    88,   645,   294,   185,   186,   187,   188,
      83,    76,   568,   558,    83,   622,    78,    86,   570,    77,
      79,    86,    81,   215,   602,   217,   578,    76,   635,    88,
     582,    80,   649,    55,    83,   662,   653,   318,   327,   656,
     585,    80,   231,     3,     4,   623,   687,    86,   675,   676,
     605,   247,   679,   249,    77,    76,    76,    76,   699,   220,
      83,    80,    83,    86,   175,    76,   683,    37,   646,    80,
     431,    75,    83,    77,    86,   716,   632,   629,    81,    86,
      83,   633,    57,    86,   636,    86,    89,    90,    83,    84,
     282,    86,    80,   710,    89,    90,   713,   289,    75,   260,
     290,   212,   292,   293,   721,   295,    82,   685,   619,    40,
      41,   300,   664,    55,    56,    57,   284,    61,    62,    63,
      64,   302,   303,   304,    61,    62,   315,   318,    53,    82,
      55,   242,   238,   239,   240,   246,     1,    75,   330,    77,
     332,   333,    82,    75,   255,    77,   257,    75,   700,    77,
     261,   262,   263,   264,   265,   266,    83,   709,    76,    86,
      87,    78,    89,    90,    68,   326,    79,   719,    78,    38,
     361,    86,    86,   364,   285,   343,   357,   358,    43,    44,
      45,    46,    47,    48,    49,    55,    56,    57,   379,    78,
     371,   372,   373,   374,   375,    58,    55,    84,    82,   560,
      82,   562,    87,   309,   310,    79,   395,    87,    80,    74,
      76,    87,    77,    78,    87,   321,   322,   323,   324,   325,
       1,    81,    79,    83,    51,   416,    86,   419,    79,    89,
      90,   422,    87,   423,   424,   425,    77,     1,    81,   400,
      84,    79,    81,   429,    83,   437,   607,    86,    79,   440,
      89,    90,    81,    78,    87,   429,   617,   429,    79,    87,
      86,    17,    43,    44,    45,    46,    47,    48,    49,    88,
     431,    88,    83,    87,   435,    84,    84,    58,    81,    43,
      44,    45,    46,    47,    48,    49,    84,    81,    84,     1,
      79,    78,    83,    74,   655,   406,    77,    78,    83,    83,
      42,    78,    83,    78,    84,    79,    57,   418,    86,   477,
      74,   472,    83,    77,    78,    87,    84,    84,   429,    84,
     512,    84,    79,    37,    86,   436,    87,    87,    87,   490,
      83,    43,    44,    45,    46,    47,    48,    49,    43,    44,
      45,    46,    47,    48,    49,    86,    58,    87,    55,   541,
      77,    57,    77,    20,   527,   547,   529,    55,   469,    79,
      77,    87,    74,    78,   556,    77,    78,    88,    86,    74,
      78,    83,    77,   534,   565,     6,     7,     8,     9,    75,
      11,    12,    13,    14,    15,   558,    86,   579,   499,   581,
      84,    77,   584,   566,    77,    56,    78,    78,    55,   560,
      76,   562,    79,    87,    75,    50,     6,    75,   600,     1,
     583,    42,   585,    87,     6,     7,     8,     9,    10,    78,
      78,    76,   614,    75,    80,    78,    18,    19,   619,   620,
      22,    78,    80,    87,    78,     1,    79,    37,    38,   631,
      76,    78,    81,    78,    75,    87,   607,    78,    78,   641,
      42,    79,    78,    78,    78,    55,   617,     5,    70,    55,
     652,    61,    62,    63,    64,    65,    66,    67,   156,    69,
      70,     1,   132,   177,   190,   312,   668,    43,    44,    45,
      46,    47,    48,    49,   307,   596,    78,    87,   298,   438,
     395,    70,   401,     1,   655,   317,   688,    70,   318,     6,
     231,   398,   663,   429,   489,   620,   534,   331,    74,    75,
     582,    77,   553,    43,    44,    45,    46,    47,    48,    49,
     712,    11,    12,    13,    14,    15,    16,   542,   639,   640,
      37,    38,   522,   644,   571,    43,    44,    45,    46,    47,
      48,    49,    -1,    -1,    74,    -1,    -1,    77,    55,    -1,
      -1,   662,    -1,    -1,    61,    62,    63,    64,    65,    66,
      67,    -1,    69,    70,   675,   676,    74,   678,   679,    77,
      -1,    -1,    -1,    -1,     1,    -1,     3,     4,     5,    -1,
       7,     8,     9,   694,    11,    12,    13,    14,    15,    -1,
      -1,    -1,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    -1,    32,    33,    34,    35,    36,
      37,    38,    39,    40,    41,    42,    43,    44,    45,    46,
      47,    48,    49,    50,    -1,    52,    -1,    54,    55,    -1,
      -1,    -1,    -1,    -1,    61,    62,    63,    64,    65,    66,
      67,    68,     1,    -1,    71,    72,    73,    74,    -1,     1,
      -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,     1,    86,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    14,    15,    16,    -1,    18,    19,    -1,    -1,    22,
      -1,    -1,    -1,    -1,    43,    44,    45,    46,    47,    48,
      49,    43,    44,    45,    46,    47,    48,    49,    43,    44,
      45,    46,    47,    48,    49,     1,    -1,    -1,    -1,    -1,
      53,    54,     1,    37,    38,    74,    59,    60,    77,     1,
      -1,    -1,    74,    -1,    -1,    77,     1,    -1,    71,    74,
      -1,    55,    77,    -1,    77,    78,    -1,    61,    62,    63,
      64,    65,    66,    67,    -1,    -1,    -1,    43,    44,    45,
      46,    47,    48,    49,    43,    44,    45,    46,    47,    48,
      49,    43,    44,    45,    46,    47,    48,    49,    43,    44,
      45,    46,    47,    48,    49,     1,    -1,    -1,    74,    -1,
      -1,    77,     1,    -1,    -1,    74,    37,    38,    77,    -1,
      -1,    -1,    74,    -1,    -1,    77,    -1,    -1,    -1,    74,
      -1,    -1,    77,    -1,    55,    56,    57,    -1,    -1,    -1,
      61,    62,    63,    64,    65,    66,    67,    43,    44,    45,
      46,    47,    48,    49,    43,    44,    45,    46,    47,    48,
      49,     3,     4,    -1,     6,     7,     8,     9,    10,    11,
      12,    13,    14,    15,    16,     1,    18,    19,    74,    -1,
      22,    77,    -1,    -1,    -1,    74,    -1,    -1,    77,    -1,
      -1,    -1,    -1,    37,    38,    21,    -1,    23,    24,    25,
      26,    27,    28,    -1,    -1,    31,    32,    33,    34,    35,
      36,    55,    -1,    -1,    -1,    -1,    -1,    61,    62,    63,
      64,    65,    66,    67,    -1,    51,    52,    -1,     1,    -1,
      -1,    -1,    -1,    -1,    -1,    77,    -1,    -1,    -1,    -1,
      -1,    -1,    68,    -1,    -1,    -1,    72,    73,    21,    -1,
      23,    24,    25,    26,    27,    28,    -1,    -1,    31,    32,
      33,    34,    35,    36,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    51,    52,
      -1,    -1,    -1,    -1,    -1,     3,     4,     5,    -1,     7,
       8,     9,    -1,    11,    12,    13,    14,    15,    -1,    72,
      73,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      28,    29,    30,    -1,    32,    33,    34,    35,    36,    37,
      38,    39,    40,    41,    42,    43,    44,    45,    46,    47,
      48,    49,    50,    -1,    52,    -1,    54,    55,    -1,    -1,
      -1,    -1,    -1,    61,    62,    63,    64,    65,    66,    67,
      68,    -1,    -1,    71,    72,    73,    74,     3,     4,     5,
      -1,     7,     8,     9,    -1,    11,    12,    13,    14,    15,
      -1,    -1,    -1,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    -1,    32,    33,    34,    35,
      36,    -1,    -1,    39,    40,    41,    42,    43,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    -1,    54,    55,
      56,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    68,    -1,    -1,    71,    72,    73,    74,     3,
       4,     5,    -1,     7,     8,     9,    -1,    11,    12,    13,
      14,    15,    -1,    -1,    -1,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,    29,    30,    -1,    32,    33,
      34,    35,    36,    -1,    -1,    39,    40,    41,    42,    43,
      44,    45,    46,    47,    48,    49,    50,    -1,    52,    -1,
      54,    55,    56,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    68,    -1,    -1,    71,    72,    73,
      74,     3,     4,     5,    -1,     7,     8,     9,    -1,    11,
      12,    13,    14,    15,    -1,    -1,    -1,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    -1,
      32,    33,    34,    35,    36,    -1,    -1,    39,    40,    41,
      42,    43,    44,    45,    46,    47,    48,    49,    50,    -1,
      52,    -1,    54,    55,    56,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    68,    -1,    -1,    71,
      72,    73,    74,     3,     4,     5,    -1,     7,     8,     9,
      -1,    11,    12,    13,    14,    15,    -1,    -1,    -1,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    -1,    32,    33,    34,    35,    36,    16,    -1,    39,
      40,    41,    42,    43,    44,    45,    46,    47,    48,    49,
      50,    -1,    52,    -1,    54,    55,    -1,    -1,    37,    38,
      18,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    68,    -1,
      -1,    71,    72,    73,    74,    -1,    55,    -1,    -1,    37,
      38,    -1,    61,    62,    63,    64,    65,    66,    67,    -1,
      -1,    -1,    37,    38,    39,    -1,    -1,    55,    -1,    -1,
      -1,    -1,    -1,    61,    62,    63,    64,    65,    66,    67,
      55,    -1,    -1,    -1,    -1,    -1,    61,    62,    63,    64,
      65,    66,    67,    37,    38,    39,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    37,    -1,    -1,    84,
      -1,    55,    -1,    -1,    -1,    -1,    -1,    61,    62,    63,
      64,    65,    66,    67,    55,    -1,    -1,    -1,    -1,    -1,
      61,    62,    63,    64,    65,    66,    67
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     3,     4,    92,    93,    99,     3,     4,     5,     7,
       8,     9,    11,    12,    13,    14,    15,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    32,
      33,    34,    35,    36,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    52,    54,    55,    68,
      71,    72,    73,    74,    97,    97,     0,    93,    75,    77,
      95,   100,   100,     1,     5,    53,    54,    59,    60,    71,
      94,   101,   102,   103,   169,   206,   207,    75,    42,    97,
      53,    55,    98,    97,    97,    77,    95,   178,     6,     7,
       8,     9,    10,    11,    12,    13,    14,    15,    16,    18,
      19,    22,    77,    99,   122,   123,   139,   142,   143,   144,
     146,   156,   157,   160,   161,   162,    78,     6,     7,     8,
       9,    11,    12,    13,    14,    15,    42,    78,    95,   167,
     101,    37,    38,    61,    62,    63,    64,    65,    66,    67,
      98,   108,   110,   111,   112,   113,   116,   117,   170,    77,
      98,    76,    58,    83,   176,    16,    38,   111,   112,   113,
     114,   115,   118,    37,   124,   124,    86,   124,   110,   163,
      86,   128,   128,   128,   128,    86,   132,   145,    86,   125,
      97,    57,   164,    80,   101,    11,    12,    13,    14,    15,
      16,   147,   148,   149,   150,   151,    95,    96,   116,    62,
      66,    61,    62,    63,    64,    80,   107,    82,    82,    82,
      38,    85,   110,   101,    76,    77,    83,    86,   176,    78,
     179,   111,   115,    38,    83,    85,    98,    98,    98,    68,
      98,    79,    30,    51,   129,   134,    97,   109,   109,   109,
     109,    51,    56,    97,   131,   133,    86,   145,    86,   132,
      40,    41,   126,   127,   109,    18,   114,   118,   154,   155,
      78,   128,   128,   128,   128,   145,   125,    61,    62,    56,
      57,   104,   105,   106,   118,    97,    78,    55,   176,    58,
     175,   176,    84,    95,    82,    82,    86,   120,   121,   204,
      83,    80,    83,    87,    80,    83,   163,    87,    79,   107,
      76,   140,   140,   140,   140,    97,    87,    79,    87,   109,
     109,    87,    79,    77,    97,    88,   153,    97,    79,    81,
      96,    97,    97,    97,    97,    97,    97,    79,    81,   107,
      78,    79,    84,    87,   176,    98,    97,   121,   119,   176,
     124,   105,   124,   124,   105,   124,   129,   110,   141,    77,
      95,   158,   158,   158,   158,    87,   133,   140,   140,   126,
      17,   135,   137,   138,    88,   152,    56,    57,   110,   153,
     155,   140,   140,   140,   140,   140,    77,    95,   105,    83,
     187,   176,   175,   176,   176,   121,    84,    87,   205,    84,
      81,    84,    98,    81,    84,    79,     1,    42,   156,   159,
     160,   165,   166,   168,   158,   158,   118,   138,    78,   118,
     158,   158,   158,   158,   158,   138,    39,    84,   118,   177,
     180,   185,    83,    83,    83,    83,   141,     1,    86,   171,
     168,    78,    95,   159,    97,    78,   118,   177,    97,   176,
      79,    84,   185,   124,   124,   124,     1,    21,    23,    24,
      25,    26,    27,    28,    31,    32,    33,    34,    35,    36,
      51,    52,    68,    72,    73,   172,   173,    55,    97,   170,
      96,    86,   136,    95,    97,   176,    86,    88,   135,    87,
     185,    84,    84,    84,    84,    57,   130,    87,    87,    79,
     187,    97,    87,    95,    87,    56,    57,    98,   174,    37,
     172,     1,    43,    44,    45,    46,    47,    48,    49,    74,
      77,    95,   178,   190,   195,   197,   187,    97,    83,   201,
      86,   201,    55,   202,   203,    77,    57,   194,   201,    77,
     191,   197,   176,    20,   189,   187,   176,   199,    55,   199,
     187,   204,    79,    77,   197,   192,   197,   178,   199,    43,
      44,    45,    47,    48,    49,    74,   178,   193,   195,   196,
      78,   191,   179,    88,   190,    86,   188,    75,    87,    84,
     200,   176,   203,    78,   191,    78,   191,   176,   200,   201,
      86,   201,    77,   194,   201,    77,   176,    78,   193,    96,
      96,    56,     6,    69,    70,    87,   118,   181,   184,   186,
     178,   199,   201,    77,   197,   205,    78,   179,    77,   197,
     176,    55,   176,   192,   178,   176,   193,   179,    97,    76,
      79,    87,   176,    75,   199,   191,   187,    96,   191,    50,
     198,    75,    87,   200,    78,   176,   200,    78,    96,    80,
     118,   177,   183,   186,   179,   199,    76,    78,    78,    77,
     197,   176,   201,    77,   197,   179,    77,   197,    97,   182,
      97,   176,    80,    97,   200,   199,   198,   191,    75,   176,
     191,    96,   191,   198,    81,    83,    86,    89,    90,    80,
      87,   182,    95,    77,   197,    79,    78,   176,    76,    78,
      78,   182,    56,   182,    81,    97,   182,    81,   191,   199,
     200,   176,   198,    84,    87,    87,    97,    81,    78,   200,
      77,   197,    79,    77,   197,   191,   176,   191,    78,   200,
      78,    77,   197,   191,    78
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    91,    92,    93,    93,    94,    94,    95,    95,    96,
      96,    97,    97,    97,    97,    97,    97,    97,    97,    97,
      97,    97,    97,    97,    97,    97,    97,    97,    97,    97,
      97,    97,    97,    97,    97,    97,    97,    97,    97,    97,
      97,    97,    97,    97,    97,    97,    97,    97,    97,    97,
      97,    97,    97,    97,    97,    97,    97,    97,    97,    98,
      98,    99,    99,   100,   100,   101,   101,   102,   102,   102,
     102,   102,   103,   103,   103,   103,   103,   103,   103,   103,
     103,   103,   103,   103,   103,   103,   104,   104,   104,   105,
     105,   106,   106,   107,   107,   108,   108,   108,   108,   108,
     108,   108,   108,   108,   108,   108,   108,   108,   108,   108,
     109,   110,   111,   111,   112,   113,   113,   114,   115,   115,
     115,   115,   115,   115,   116,   116,   116,   116,   116,   117,
     117,   118,   118,   119,   120,   121,   121,   122,   123,   124,
     124,   125,   125,   126,   126,   127,   127,   128,   128,   129,
     129,   130,   130,   131,   132,   132,   133,   133,   134,   134,
     135,   135,   136,   136,   137,   138,   138,   139,   139,   140,
     140,   141,   141,   142,   142,   143,   144,   145,   145,   146,
     146,   147,   147,   148,   149,   150,   151,   151,   152,   152,
     153,   153,   153,   153,   154,   154,   154,   155,   155,   156,
     157,   157,   157,   157,   157,   158,   158,   159,   159,   160,
     160,   160,   160,   160,   160,   160,   161,   161,   161,   161,
     161,   162,   162,   162,   162,   163,   163,   164,   165,   166,
     166,   166,   166,   167,   167,   167,   167,   167,   167,   167,
     167,   167,   167,   167,   168,   168,   168,   169,   169,   170,
     171,   171,   171,   172,   172,   173,   173,   173,   173,   173,
     173,   173,   173,   173,   173,   173,   173,   173,   173,   173,
     173,   173,   173,   174,   174,   174,   175,   175,   175,   176,
     176,   176,   176,   176,   176,   177,   178,   179,   180,   180,
     180,   180,   180,   181,   181,   181,   182,   182,   182,   182,
     182,   182,   183,   184,   184,   184,   185,   185,   186,   186,
     187,   187,   188,   188,   189,   189,   190,   190,   190,   191,
     191,   192,   192,   193,   193,   193,   194,   194,   195,   195,
     195,   196,   196,   196,   196,   196,   196,   196,   196,   196,
     196,   196,   196,   197,   197,   197,   197,   197,   197,   197,
     197,   197,   197,   197,   197,   197,   197,   198,   198,   198,
     199,   200,   201,   202,   202,   203,   203,   204,   205,   206,
     207
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
       4,     3,     3,     1,     4,     0,     2,     3,     2,     2,
       2,     7,     5,     5,     2,     2,     2,     2,     2,     2,
       2,     2,     1,     1,     1,     1,     1,     1,     1,     1,
       3,     0,     1,     0,     3,     1,     1,     1,     1,     2,
       2,     3,     3,     2,     2,     2,     1,     1,     2,     1,
       2,     2,     1,     1,     2,     2,     2,     8,     1,     1,
       1,     1,     2,     2,     1,     1,     1,     2,     2,     2,
       1,     2,     1,     1,     3,     0,     2,     4,     6,     0,
       1,     0,     3,     1,     3,     1,     1,     0,     3,     1,
       3,     0,     1,     1,     0,     3,     1,     3,     1,     1,
       0,     1,     0,     2,     5,     1,     2,     3,     6,     0,
       2,     1,     3,     5,     5,     5,     5,     4,     3,     6,
       6,     5,     5,     5,     5,     5,     4,     7,     0,     2,
       0,     2,     2,     2,     3,     2,     3,     1,     3,     4,
       2,     2,     2,     2,     2,     1,     4,     0,     2,     1,
       1,     1,     1,     2,     2,     2,     3,     6,     9,     3,
       6,     3,     6,     9,     9,     1,     3,     1,     1,     1,
       2,     2,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     7,     5,    13,     5,     2,     1,
       0,     3,     1,     1,     3,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     2,     1,
       1,     1,     1,     1,     1,     1,     0,     1,     3,     0,
       1,     5,     5,     5,     4,     3,     1,     1,     1,     3,
       4,     3,     4,     1,     1,     1,     1,     4,     3,     4,
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
#line 194 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.modlist) = (yyvsp[0].modlist); modlist = (yyvsp[0].modlist); }
#line 2238 "y.tab.c" /* yacc.c:1646  */
    break;

  case 3:
#line 198 "xi-grammar.y" /* yacc.c:1646  */
    { 
		  (yyval.modlist) = 0; 
		}
#line 2246 "y.tab.c" /* yacc.c:1646  */
    break;

  case 4:
#line 202 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.modlist) = new AstChildren<Module>(lineno, (yyvsp[-1].module), (yyvsp[0].modlist)); }
#line 2252 "y.tab.c" /* yacc.c:1646  */
    break;

  case 5:
#line 206 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 2258 "y.tab.c" /* yacc.c:1646  */
    break;

  case 6:
#line 208 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 2264 "y.tab.c" /* yacc.c:1646  */
    break;

  case 7:
#line 212 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 2270 "y.tab.c" /* yacc.c:1646  */
    break;

  case 8:
#line 214 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 2; }
#line 2276 "y.tab.c" /* yacc.c:1646  */
    break;

  case 9:
#line 218 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 2282 "y.tab.c" /* yacc.c:1646  */
    break;

  case 10:
#line 220 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 2288 "y.tab.c" /* yacc.c:1646  */
    break;

  case 11:
#line 225 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 2294 "y.tab.c" /* yacc.c:1646  */
    break;

  case 12:
#line 226 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MODULE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2300 "y.tab.c" /* yacc.c:1646  */
    break;

  case 13:
#line 227 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MAINMODULE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2306 "y.tab.c" /* yacc.c:1646  */
    break;

  case 14:
#line 228 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(EXTERN, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2312 "y.tab.c" /* yacc.c:1646  */
    break;

  case 15:
#line 230 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITCALL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2318 "y.tab.c" /* yacc.c:1646  */
    break;

  case 16:
#line 231 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITNODE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2324 "y.tab.c" /* yacc.c:1646  */
    break;

  case 17:
#line 232 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITPROC, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2330 "y.tab.c" /* yacc.c:1646  */
    break;

  case 18:
#line 234 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CHARE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2336 "y.tab.c" /* yacc.c:1646  */
    break;

  case 19:
#line 235 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MAINCHARE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2342 "y.tab.c" /* yacc.c:1646  */
    break;

  case 20:
#line 236 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(GROUP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2348 "y.tab.c" /* yacc.c:1646  */
    break;

  case 21:
#line 237 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NODEGROUP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2354 "y.tab.c" /* yacc.c:1646  */
    break;

  case 22:
#line 238 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ARRAY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2360 "y.tab.c" /* yacc.c:1646  */
    break;

  case 23:
#line 242 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INCLUDE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2366 "y.tab.c" /* yacc.c:1646  */
    break;

  case 24:
#line 243 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(STACKSIZE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2372 "y.tab.c" /* yacc.c:1646  */
    break;

  case 25:
#line 244 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(THREADED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2378 "y.tab.c" /* yacc.c:1646  */
    break;

  case 26:
#line 245 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(TEMPLATE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2384 "y.tab.c" /* yacc.c:1646  */
    break;

  case 27:
#line 246 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(SYNC, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2390 "y.tab.c" /* yacc.c:1646  */
    break;

  case 28:
#line 247 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IGET, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2396 "y.tab.c" /* yacc.c:1646  */
    break;

  case 29:
#line 248 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(EXCLUSIVE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2402 "y.tab.c" /* yacc.c:1646  */
    break;

  case 30:
#line 249 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IMMEDIATE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2408 "y.tab.c" /* yacc.c:1646  */
    break;

  case 31:
#line 250 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(SKIPSCHED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2414 "y.tab.c" /* yacc.c:1646  */
    break;

  case 32:
#line 251 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(RDMA, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2420 "y.tab.c" /* yacc.c:1646  */
    break;

  case 33:
#line 252 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INLINE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2426 "y.tab.c" /* yacc.c:1646  */
    break;

  case 34:
#line 253 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(VIRTUAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2432 "y.tab.c" /* yacc.c:1646  */
    break;

  case 35:
#line 254 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MIGRATABLE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2438 "y.tab.c" /* yacc.c:1646  */
    break;

  case 36:
#line 255 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CREATEHERE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2444 "y.tab.c" /* yacc.c:1646  */
    break;

  case 37:
#line 256 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CREATEHOME, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2450 "y.tab.c" /* yacc.c:1646  */
    break;

  case 38:
#line 257 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOKEEP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2456 "y.tab.c" /* yacc.c:1646  */
    break;

  case 39:
#line 258 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOTRACE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2462 "y.tab.c" /* yacc.c:1646  */
    break;

  case 40:
#line 259 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(APPWORK, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2468 "y.tab.c" /* yacc.c:1646  */
    break;

  case 41:
#line 262 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(PACKED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2474 "y.tab.c" /* yacc.c:1646  */
    break;

  case 42:
#line 263 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(VARSIZE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2480 "y.tab.c" /* yacc.c:1646  */
    break;

  case 43:
#line 264 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ENTRY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2486 "y.tab.c" /* yacc.c:1646  */
    break;

  case 44:
#line 265 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(FOR, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2492 "y.tab.c" /* yacc.c:1646  */
    break;

  case 45:
#line 266 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(FORALL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2498 "y.tab.c" /* yacc.c:1646  */
    break;

  case 46:
#line 267 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(WHILE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2504 "y.tab.c" /* yacc.c:1646  */
    break;

  case 47:
#line 268 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(WHEN, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2510 "y.tab.c" /* yacc.c:1646  */
    break;

  case 48:
#line 269 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(OVERLAP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2516 "y.tab.c" /* yacc.c:1646  */
    break;

  case 49:
#line 270 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(SERIAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2522 "y.tab.c" /* yacc.c:1646  */
    break;

  case 50:
#line 271 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IF, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2528 "y.tab.c" /* yacc.c:1646  */
    break;

  case 51:
#line 272 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ELSE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2534 "y.tab.c" /* yacc.c:1646  */
    break;

  case 52:
#line 274 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(LOCAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2540 "y.tab.c" /* yacc.c:1646  */
    break;

  case 53:
#line 276 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(USING, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2546 "y.tab.c" /* yacc.c:1646  */
    break;

  case 54:
#line 277 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ACCEL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2552 "y.tab.c" /* yacc.c:1646  */
    break;

  case 55:
#line 280 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ACCELBLOCK, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2558 "y.tab.c" /* yacc.c:1646  */
    break;

  case 56:
#line 281 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MEMCRITICAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2564 "y.tab.c" /* yacc.c:1646  */
    break;

  case 57:
#line 282 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(REDUCTIONTARGET, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2570 "y.tab.c" /* yacc.c:1646  */
    break;

  case 58:
#line 283 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CASE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
#line 2576 "y.tab.c" /* yacc.c:1646  */
    break;

  case 59:
#line 288 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 2582 "y.tab.c" /* yacc.c:1646  */
    break;

  case 60:
#line 290 "xi-grammar.y" /* yacc.c:1646  */
    {
		  char *tmp = new char[strlen((yyvsp[-3].strval))+strlen((yyvsp[0].strval))+3];
		  sprintf(tmp,"%s::%s", (yyvsp[-3].strval), (yyvsp[0].strval));
		  (yyval.strval) = tmp;
		}
#line 2592 "y.tab.c" /* yacc.c:1646  */
    break;

  case 61:
#line 297 "xi-grammar.y" /* yacc.c:1646  */
    { 
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist)); 
		}
#line 2600 "y.tab.c" /* yacc.c:1646  */
    break;

  case 62:
#line 301 "xi-grammar.y" /* yacc.c:1646  */
    {  
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist)); 
		    (yyval.module)->setMain();
		}
#line 2609 "y.tab.c" /* yacc.c:1646  */
    break;

  case 63:
#line 308 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = 0; }
#line 2615 "y.tab.c" /* yacc.c:1646  */
    break;

  case 64:
#line 310 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = (yyvsp[-2].conslist); }
#line 2621 "y.tab.c" /* yacc.c:1646  */
    break;

  case 65:
#line 314 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = 0; }
#line 2627 "y.tab.c" /* yacc.c:1646  */
    break;

  case 66:
#line 316 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = new ConstructList(lineno, (yyvsp[-1].construct), (yyvsp[0].conslist)); }
#line 2633 "y.tab.c" /* yacc.c:1646  */
    break;

  case 67:
#line 320 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new UsingScope((yyvsp[0].strval), false); }
#line 2639 "y.tab.c" /* yacc.c:1646  */
    break;

  case 68:
#line 322 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new UsingScope((yyvsp[0].strval), true); }
#line 2645 "y.tab.c" /* yacc.c:1646  */
    break;

  case 69:
#line 324 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].member)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].member); }
#line 2651 "y.tab.c" /* yacc.c:1646  */
    break;

  case 70:
#line 326 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].message)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].message); }
#line 2657 "y.tab.c" /* yacc.c:1646  */
    break;

  case 71:
#line 328 "xi-grammar.y" /* yacc.c:1646  */
    {
                  Entry *e = new Entry(lineno, 0, (yyvsp[-4].type), (yyvsp[-2].strval), (yyvsp[0].plist), 0, 0, 0, (yylsp[-6]).first_line, (yyloc).last_line);
                  int isExtern = 1;
                  e->setExtern(isExtern);
                  e->targs = (yyvsp[-1].tparlist);
                  e->label = new XStr;
                  (yyvsp[-3].ntype)->print(*e->label);
                  (yyval.construct) = e;
                }
#line 2671 "y.tab.c" /* yacc.c:1646  */
    break;

  case 72:
#line 340 "xi-grammar.y" /* yacc.c:1646  */
    { if((yyvsp[-2].conslist)) (yyvsp[-2].conslist)->recurse<int&>((yyvsp[-4].intval), &Construct::setExtern); (yyval.construct) = (yyvsp[-2].conslist); }
#line 2677 "y.tab.c" /* yacc.c:1646  */
    break;

  case 73:
#line 342 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new Scope((yyvsp[-3].strval), (yyvsp[-1].conslist)); }
#line 2683 "y.tab.c" /* yacc.c:1646  */
    break;

  case 74:
#line 344 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = (yyvsp[-1].construct); }
#line 2689 "y.tab.c" /* yacc.c:1646  */
    break;

  case 75:
#line 346 "xi-grammar.y" /* yacc.c:1646  */
    {
          ERROR("preceding construct must be semicolon terminated",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 2699 "y.tab.c" /* yacc.c:1646  */
    break;

  case 76:
#line 352 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].module)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].module); }
#line 2705 "y.tab.c" /* yacc.c:1646  */
    break;

  case 77:
#line 354 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2711 "y.tab.c" /* yacc.c:1646  */
    break;

  case 78:
#line 356 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2717 "y.tab.c" /* yacc.c:1646  */
    break;

  case 79:
#line 358 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2723 "y.tab.c" /* yacc.c:1646  */
    break;

  case 80:
#line 360 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2729 "y.tab.c" /* yacc.c:1646  */
    break;

  case 81:
#line 362 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].templat)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].templat); }
#line 2735 "y.tab.c" /* yacc.c:1646  */
    break;

  case 82:
#line 364 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = NULL; }
#line 2741 "y.tab.c" /* yacc.c:1646  */
    break;

  case 83:
#line 366 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = NULL; }
#line 2747 "y.tab.c" /* yacc.c:1646  */
    break;

  case 84:
#line 368 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = (yyvsp[0].accelBlock); }
#line 2753 "y.tab.c" /* yacc.c:1646  */
    break;

  case 85:
#line 370 "xi-grammar.y" /* yacc.c:1646  */
    {
          ERROR("invalid construct",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 2763 "y.tab.c" /* yacc.c:1646  */
    break;

  case 86:
#line 378 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamType((yyvsp[0].type)); }
#line 2769 "y.tab.c" /* yacc.c:1646  */
    break;

  case 87:
#line 380 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
#line 2775 "y.tab.c" /* yacc.c:1646  */
    break;

  case 88:
#line 382 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
#line 2781 "y.tab.c" /* yacc.c:1646  */
    break;

  case 89:
#line 386 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = new TParamList((yyvsp[0].tparam)); }
#line 2787 "y.tab.c" /* yacc.c:1646  */
    break;

  case 90:
#line 388 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = new TParamList((yyvsp[-2].tparam), (yyvsp[0].tparlist)); }
#line 2793 "y.tab.c" /* yacc.c:1646  */
    break;

  case 91:
#line 392 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = 0; }
#line 2799 "y.tab.c" /* yacc.c:1646  */
    break;

  case 92:
#line 394 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = (yyvsp[0].tparlist); }
#line 2805 "y.tab.c" /* yacc.c:1646  */
    break;

  case 93:
#line 398 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = 0; }
#line 2811 "y.tab.c" /* yacc.c:1646  */
    break;

  case 94:
#line 400 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = (yyvsp[-1].tparlist); }
#line 2817 "y.tab.c" /* yacc.c:1646  */
    break;

  case 95:
#line 404 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("int"); }
#line 2823 "y.tab.c" /* yacc.c:1646  */
    break;

  case 96:
#line 406 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long"); }
#line 2829 "y.tab.c" /* yacc.c:1646  */
    break;

  case 97:
#line 408 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("short"); }
#line 2835 "y.tab.c" /* yacc.c:1646  */
    break;

  case 98:
#line 410 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("char"); }
#line 2841 "y.tab.c" /* yacc.c:1646  */
    break;

  case 99:
#line 412 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned int"); }
#line 2847 "y.tab.c" /* yacc.c:1646  */
    break;

  case 100:
#line 414 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long"); }
#line 2853 "y.tab.c" /* yacc.c:1646  */
    break;

  case 101:
#line 416 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long"); }
#line 2859 "y.tab.c" /* yacc.c:1646  */
    break;

  case 102:
#line 418 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long long"); }
#line 2865 "y.tab.c" /* yacc.c:1646  */
    break;

  case 103:
#line 420 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned short"); }
#line 2871 "y.tab.c" /* yacc.c:1646  */
    break;

  case 104:
#line 422 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned char"); }
#line 2877 "y.tab.c" /* yacc.c:1646  */
    break;

  case 105:
#line 424 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long long"); }
#line 2883 "y.tab.c" /* yacc.c:1646  */
    break;

  case 106:
#line 426 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("float"); }
#line 2889 "y.tab.c" /* yacc.c:1646  */
    break;

  case 107:
#line 428 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("double"); }
#line 2895 "y.tab.c" /* yacc.c:1646  */
    break;

  case 108:
#line 430 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long double"); }
#line 2901 "y.tab.c" /* yacc.c:1646  */
    break;

  case 109:
#line 432 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("void"); }
#line 2907 "y.tab.c" /* yacc.c:1646  */
    break;

  case 110:
#line 435 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ntype) = new NamedType((yyvsp[-1].strval),(yyvsp[0].tparlist)); }
#line 2913 "y.tab.c" /* yacc.c:1646  */
    break;

  case 111:
#line 436 "xi-grammar.y" /* yacc.c:1646  */
    { 
                    const char* basename, *scope;
                    splitScopedName((yyvsp[-1].strval), &scope, &basename);
                    (yyval.ntype) = new NamedType(basename, (yyvsp[0].tparlist), scope);
                }
#line 2923 "y.tab.c" /* yacc.c:1646  */
    break;

  case 112:
#line 444 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 2929 "y.tab.c" /* yacc.c:1646  */
    break;

  case 113:
#line 446 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ntype); }
#line 2935 "y.tab.c" /* yacc.c:1646  */
    break;

  case 114:
#line 450 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ptype) = new PtrType((yyvsp[-1].type)); }
#line 2941 "y.tab.c" /* yacc.c:1646  */
    break;

  case 115:
#line 454 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
#line 2947 "y.tab.c" /* yacc.c:1646  */
    break;

  case 116:
#line 456 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
#line 2953 "y.tab.c" /* yacc.c:1646  */
    break;

  case 117:
#line 460 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ftype) = new FuncType((yyvsp[-7].type), (yyvsp[-4].strval), (yyvsp[-1].plist)); }
#line 2959 "y.tab.c" /* yacc.c:1646  */
    break;

  case 118:
#line 464 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 2965 "y.tab.c" /* yacc.c:1646  */
    break;

  case 119:
#line 466 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 2971 "y.tab.c" /* yacc.c:1646  */
    break;

  case 120:
#line 468 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 2977 "y.tab.c" /* yacc.c:1646  */
    break;

  case 121:
#line 470 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ftype); }
#line 2983 "y.tab.c" /* yacc.c:1646  */
    break;

  case 122:
#line 472 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[0].type)); }
#line 2989 "y.tab.c" /* yacc.c:1646  */
    break;

  case 123:
#line 474 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[-1].type)); }
#line 2995 "y.tab.c" /* yacc.c:1646  */
    break;

  case 124:
#line 478 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3001 "y.tab.c" /* yacc.c:1646  */
    break;

  case 125:
#line 480 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3007 "y.tab.c" /* yacc.c:1646  */
    break;

  case 126:
#line 482 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3013 "y.tab.c" /* yacc.c:1646  */
    break;

  case 127:
#line 484 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[0].type)); }
#line 3019 "y.tab.c" /* yacc.c:1646  */
    break;

  case 128:
#line 486 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[-1].type)); }
#line 3025 "y.tab.c" /* yacc.c:1646  */
    break;

  case 129:
#line 490 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ReferenceType((yyvsp[-1].type)); }
#line 3031 "y.tab.c" /* yacc.c:1646  */
    break;

  case 130:
#line 492 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3037 "y.tab.c" /* yacc.c:1646  */
    break;

  case 131:
#line 496 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ReferenceType((yyvsp[-1].type)); }
#line 3043 "y.tab.c" /* yacc.c:1646  */
    break;

  case 132:
#line 498 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3049 "y.tab.c" /* yacc.c:1646  */
    break;

  case 133:
#line 502 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 3055 "y.tab.c" /* yacc.c:1646  */
    break;

  case 134:
#line 506 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = (yyvsp[-1].val); }
#line 3061 "y.tab.c" /* yacc.c:1646  */
    break;

  case 135:
#line 510 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.vallist) = 0; }
#line 3067 "y.tab.c" /* yacc.c:1646  */
    break;

  case 136:
#line 512 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.vallist) = new ValueList((yyvsp[-1].val), (yyvsp[0].vallist)); }
#line 3073 "y.tab.c" /* yacc.c:1646  */
    break;

  case 137:
#line 516 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].vallist)); }
#line 3079 "y.tab.c" /* yacc.c:1646  */
    break;

  case 138:
#line 520 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[-3].type), (yyvsp[-1].strval), (yyvsp[0].vallist), 1); }
#line 3085 "y.tab.c" /* yacc.c:1646  */
    break;

  case 139:
#line 524 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0;}
#line 3091 "y.tab.c" /* yacc.c:1646  */
    break;

  case 140:
#line 526 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0;}
#line 3097 "y.tab.c" /* yacc.c:1646  */
    break;

  case 141:
#line 530 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3103 "y.tab.c" /* yacc.c:1646  */
    break;

  case 142:
#line 532 "xi-grammar.y" /* yacc.c:1646  */
    { 
		  /*
		  printf("Warning: Message attributes are being phased out.\n");
		  printf("Warning: Please remove them from interface files.\n");
		  */
		  (yyval.intval) = (yyvsp[-1].intval); 
		}
#line 3115 "y.tab.c" /* yacc.c:1646  */
    break;

  case 143:
#line 542 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[0].intval); }
#line 3121 "y.tab.c" /* yacc.c:1646  */
    break;

  case 144:
#line 544 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[-2].intval) | (yyvsp[0].intval); }
#line 3127 "y.tab.c" /* yacc.c:1646  */
    break;

  case 145:
#line 548 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3133 "y.tab.c" /* yacc.c:1646  */
    break;

  case 146:
#line 550 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3139 "y.tab.c" /* yacc.c:1646  */
    break;

  case 147:
#line 554 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = 0; }
#line 3145 "y.tab.c" /* yacc.c:1646  */
    break;

  case 148:
#line 556 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-1].cattr); }
#line 3151 "y.tab.c" /* yacc.c:1646  */
    break;

  case 149:
#line 560 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[0].cattr); }
#line 3157 "y.tab.c" /* yacc.c:1646  */
    break;

  case 150:
#line 562 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
#line 3163 "y.tab.c" /* yacc.c:1646  */
    break;

  case 151:
#line 566 "xi-grammar.y" /* yacc.c:1646  */
    { python_doc = NULL; (yyval.intval) = 0; }
#line 3169 "y.tab.c" /* yacc.c:1646  */
    break;

  case 152:
#line 568 "xi-grammar.y" /* yacc.c:1646  */
    { python_doc = (yyvsp[0].strval); (yyval.intval) = 0; }
#line 3175 "y.tab.c" /* yacc.c:1646  */
    break;

  case 153:
#line 572 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CPYTHON; }
#line 3181 "y.tab.c" /* yacc.c:1646  */
    break;

  case 154:
#line 576 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = 0; }
#line 3187 "y.tab.c" /* yacc.c:1646  */
    break;

  case 155:
#line 578 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-1].cattr); }
#line 3193 "y.tab.c" /* yacc.c:1646  */
    break;

  case 156:
#line 582 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[0].cattr); }
#line 3199 "y.tab.c" /* yacc.c:1646  */
    break;

  case 157:
#line 584 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
#line 3205 "y.tab.c" /* yacc.c:1646  */
    break;

  case 158:
#line 588 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CMIGRATABLE; }
#line 3211 "y.tab.c" /* yacc.c:1646  */
    break;

  case 159:
#line 590 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CPYTHON; }
#line 3217 "y.tab.c" /* yacc.c:1646  */
    break;

  case 160:
#line 594 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3223 "y.tab.c" /* yacc.c:1646  */
    break;

  case 161:
#line 596 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 3229 "y.tab.c" /* yacc.c:1646  */
    break;

  case 162:
#line 599 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3235 "y.tab.c" /* yacc.c:1646  */
    break;

  case 163:
#line 601 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 3241 "y.tab.c" /* yacc.c:1646  */
    break;

  case 164:
#line 604 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mv) = new MsgVar((yyvsp[-3].type), (yyvsp[-2].strval), (yyvsp[-4].intval), (yyvsp[-1].intval)); }
#line 3247 "y.tab.c" /* yacc.c:1646  */
    break;

  case 165:
#line 608 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mvlist) = new MsgVarList((yyvsp[0].mv)); }
#line 3253 "y.tab.c" /* yacc.c:1646  */
    break;

  case 166:
#line 610 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mvlist) = new MsgVarList((yyvsp[-1].mv), (yyvsp[0].mvlist)); }
#line 3259 "y.tab.c" /* yacc.c:1646  */
    break;

  case 167:
#line 614 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, (yyvsp[0].ntype)); }
#line 3265 "y.tab.c" /* yacc.c:1646  */
    break;

  case 168:
#line 616 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, (yyvsp[-3].ntype), (yyvsp[-1].mvlist)); }
#line 3271 "y.tab.c" /* yacc.c:1646  */
    break;

  case 169:
#line 620 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = 0; }
#line 3277 "y.tab.c" /* yacc.c:1646  */
    break;

  case 170:
#line 622 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = (yyvsp[0].typelist); }
#line 3283 "y.tab.c" /* yacc.c:1646  */
    break;

  case 171:
#line 626 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = new TypeList((yyvsp[0].ntype)); }
#line 3289 "y.tab.c" /* yacc.c:1646  */
    break;

  case 172:
#line 628 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = new TypeList((yyvsp[-2].ntype), (yyvsp[0].typelist)); }
#line 3295 "y.tab.c" /* yacc.c:1646  */
    break;

  case 173:
#line 632 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr)|Chare::CCHARE, (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3301 "y.tab.c" /* yacc.c:1646  */
    break;

  case 174:
#line 634 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3307 "y.tab.c" /* yacc.c:1646  */
    break;

  case 175:
#line 638 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3313 "y.tab.c" /* yacc.c:1646  */
    break;

  case 176:
#line 642 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new NodeGroup(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3319 "y.tab.c" /* yacc.c:1646  */
    break;

  case 177:
#line 646 "xi-grammar.y" /* yacc.c:1646  */
    {/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",(yyvsp[-2].strval));
			(yyval.ntype) = new NamedType(buf); 
		}
#line 3329 "y.tab.c" /* yacc.c:1646  */
    break;

  case 178:
#line 652 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ntype) = new NamedType((yyvsp[-1].strval)); }
#line 3335 "y.tab.c" /* yacc.c:1646  */
    break;

  case 179:
#line 656 "xi-grammar.y" /* yacc.c:1646  */
    {  (yyval.chare) = new Array(lineno, (yyvsp[-4].cattr), (yyvsp[-3].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3341 "y.tab.c" /* yacc.c:1646  */
    break;

  case 180:
#line 658 "xi-grammar.y" /* yacc.c:1646  */
    {  (yyval.chare) = new Array(lineno, (yyvsp[-3].cattr), (yyvsp[-4].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3347 "y.tab.c" /* yacc.c:1646  */
    break;

  case 181:
#line 662 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr)|Chare::CCHARE, new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist));}
#line 3353 "y.tab.c" /* yacc.c:1646  */
    break;

  case 182:
#line 664 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3359 "y.tab.c" /* yacc.c:1646  */
    break;

  case 183:
#line 668 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3365 "y.tab.c" /* yacc.c:1646  */
    break;

  case 184:
#line 672 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new NodeGroup( lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3371 "y.tab.c" /* yacc.c:1646  */
    break;

  case 185:
#line 676 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Array( lineno, 0, (yyvsp[-3].ntype), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3377 "y.tab.c" /* yacc.c:1646  */
    break;

  case 186:
#line 680 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-1].strval))); }
#line 3383 "y.tab.c" /* yacc.c:1646  */
    break;

  case 187:
#line 682 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-4].strval)), (yyvsp[-2].mvlist)); }
#line 3389 "y.tab.c" /* yacc.c:1646  */
    break;

  case 188:
#line 686 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = 0; }
#line 3395 "y.tab.c" /* yacc.c:1646  */
    break;

  case 189:
#line 688 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3401 "y.tab.c" /* yacc.c:1646  */
    break;

  case 190:
#line 692 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = 0; }
#line 3407 "y.tab.c" /* yacc.c:1646  */
    break;

  case 191:
#line 694 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3413 "y.tab.c" /* yacc.c:1646  */
    break;

  case 192:
#line 696 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3419 "y.tab.c" /* yacc.c:1646  */
    break;

  case 193:
#line 698 "xi-grammar.y" /* yacc.c:1646  */
    {
		  XStr typeStr;
		  (yyvsp[0].ntype)->print(typeStr);
		  char *tmp = strdup(typeStr.get_string());
		  (yyval.strval) = tmp;
		}
#line 3430 "y.tab.c" /* yacc.c:1646  */
    break;

  case 194:
#line 707 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TType(new NamedType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3436 "y.tab.c" /* yacc.c:1646  */
    break;

  case 195:
#line 709 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TFunc((yyvsp[-1].ftype), (yyvsp[0].strval)); }
#line 3442 "y.tab.c" /* yacc.c:1646  */
    break;

  case 196:
#line 711 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TName((yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].strval)); }
#line 3448 "y.tab.c" /* yacc.c:1646  */
    break;

  case 197:
#line 715 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = new TVarList((yyvsp[0].tvar)); }
#line 3454 "y.tab.c" /* yacc.c:1646  */
    break;

  case 198:
#line 717 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = new TVarList((yyvsp[-2].tvar), (yyvsp[0].tvarlist)); }
#line 3460 "y.tab.c" /* yacc.c:1646  */
    break;

  case 199:
#line 721 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = (yyvsp[-1].tvarlist); }
#line 3466 "y.tab.c" /* yacc.c:1646  */
    break;

  case 200:
#line 725 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3472 "y.tab.c" /* yacc.c:1646  */
    break;

  case 201:
#line 727 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3478 "y.tab.c" /* yacc.c:1646  */
    break;

  case 202:
#line 729 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3484 "y.tab.c" /* yacc.c:1646  */
    break;

  case 203:
#line 731 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3490 "y.tab.c" /* yacc.c:1646  */
    break;

  case 204:
#line 733 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].message)); (yyvsp[0].message)->setTemplate((yyval.templat)); }
#line 3496 "y.tab.c" /* yacc.c:1646  */
    break;

  case 205:
#line 737 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = 0; }
#line 3502 "y.tab.c" /* yacc.c:1646  */
    break;

  case 206:
#line 739 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = (yyvsp[-2].mbrlist); }
#line 3508 "y.tab.c" /* yacc.c:1646  */
    break;

  case 207:
#line 743 "xi-grammar.y" /* yacc.c:1646  */
    { 
                  if (!connectEntries.empty()) {
                    (yyval.mbrlist) = new AstChildren<Member>(connectEntries);
		  } else {
		    (yyval.mbrlist) = 0; 
                  }
		}
#line 3520 "y.tab.c" /* yacc.c:1646  */
    break;

  case 208:
#line 751 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = new AstChildren<Member>(-1, (yyvsp[-1].member), (yyvsp[0].mbrlist)); }
#line 3526 "y.tab.c" /* yacc.c:1646  */
    break;

  case 209:
#line 755 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].readonly); }
#line 3532 "y.tab.c" /* yacc.c:1646  */
    break;

  case 210:
#line 757 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].readonly); }
#line 3538 "y.tab.c" /* yacc.c:1646  */
    break;

  case 212:
#line 760 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].member); }
#line 3544 "y.tab.c" /* yacc.c:1646  */
    break;

  case 213:
#line 762 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].pupable); }
#line 3550 "y.tab.c" /* yacc.c:1646  */
    break;

  case 214:
#line 764 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].includeFile); }
#line 3556 "y.tab.c" /* yacc.c:1646  */
    break;

  case 215:
#line 766 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new ClassDeclaration(lineno,(yyvsp[0].strval)); }
#line 3562 "y.tab.c" /* yacc.c:1646  */
    break;

  case 216:
#line 770 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1); }
#line 3568 "y.tab.c" /* yacc.c:1646  */
    break;

  case 217:
#line 772 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1); }
#line 3574 "y.tab.c" /* yacc.c:1646  */
    break;

  case 218:
#line 774 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[-6].strval)) + '<' +
					    ((yyvsp[-4].tparlist))->to_string() + '>').c_str()),
				    1);
		}
#line 3584 "y.tab.c" /* yacc.c:1646  */
    break;

  case 219:
#line 780 "xi-grammar.y" /* yacc.c:1646  */
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[-2]).first_column, (yylsp[-2]).last_column, (yylsp[-2]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1);
		}
#line 3594 "y.tab.c" /* yacc.c:1646  */
    break;

  case 220:
#line 786 "xi-grammar.y" /* yacc.c:1646  */
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[-5]).first_column, (yylsp[-5]).last_column, (yylsp[-5]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1);
		}
#line 3604 "y.tab.c" /* yacc.c:1646  */
    break;

  case 221:
#line 795 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 0); }
#line 3610 "y.tab.c" /* yacc.c:1646  */
    break;

  case 222:
#line 797 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 0); }
#line 3616 "y.tab.c" /* yacc.c:1646  */
    break;

  case 223:
#line 799 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[-6].strval)) + '<' +
					    ((yyvsp[-4].tparlist))->to_string() + '>').c_str()),
				    0);
		}
#line 3626 "y.tab.c" /* yacc.c:1646  */
    break;

  case 224:
#line 805 "xi-grammar.y" /* yacc.c:1646  */
    {
                  InitCall* rtn = new InitCall(lineno, (yyvsp[-3].strval), 0);
                  rtn->setAccel();
                  (yyval.member) = rtn;
		}
#line 3636 "y.tab.c" /* yacc.c:1646  */
    break;

  case 225:
#line 813 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[0].ntype),0); }
#line 3642 "y.tab.c" /* yacc.c:1646  */
    break;

  case 226:
#line 815 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[-2].ntype),(yyvsp[0].pupable)); }
#line 3648 "y.tab.c" /* yacc.c:1646  */
    break;

  case 227:
#line 818 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.includeFile) = new IncludeFile(lineno,(yyvsp[0].strval)); }
#line 3654 "y.tab.c" /* yacc.c:1646  */
    break;

  case 228:
#line 822 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].member); }
#line 3660 "y.tab.c" /* yacc.c:1646  */
    break;

  case 229:
#line 826 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].entry); }
#line 3666 "y.tab.c" /* yacc.c:1646  */
    break;

  case 230:
#line 828 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyvsp[0].entry)->tspec = (yyvsp[-1].tvarlist);
                  (yyval.member) = (yyvsp[0].entry);
                }
#line 3675 "y.tab.c" /* yacc.c:1646  */
    break;

  case 231:
#line 833 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[-1].member); }
#line 3681 "y.tab.c" /* yacc.c:1646  */
    break;

  case 232:
#line 835 "xi-grammar.y" /* yacc.c:1646  */
    {
          ERROR("invalid SDAG member",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 3691 "y.tab.c" /* yacc.c:1646  */
    break;

  case 233:
#line 843 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3697 "y.tab.c" /* yacc.c:1646  */
    break;

  case 234:
#line 845 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3703 "y.tab.c" /* yacc.c:1646  */
    break;

  case 235:
#line 847 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3709 "y.tab.c" /* yacc.c:1646  */
    break;

  case 236:
#line 849 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3715 "y.tab.c" /* yacc.c:1646  */
    break;

  case 237:
#line 851 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3721 "y.tab.c" /* yacc.c:1646  */
    break;

  case 238:
#line 853 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3727 "y.tab.c" /* yacc.c:1646  */
    break;

  case 239:
#line 855 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3733 "y.tab.c" /* yacc.c:1646  */
    break;

  case 240:
#line 857 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3739 "y.tab.c" /* yacc.c:1646  */
    break;

  case 241:
#line 859 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3745 "y.tab.c" /* yacc.c:1646  */
    break;

  case 242:
#line 861 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3751 "y.tab.c" /* yacc.c:1646  */
    break;

  case 243:
#line 863 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3757 "y.tab.c" /* yacc.c:1646  */
    break;

  case 244:
#line 866 "xi-grammar.y" /* yacc.c:1646  */
    { 
                  (yyval.entry) = new Entry(lineno, (yyvsp[-5].intval), (yyvsp[-4].type), (yyvsp[-3].strval), (yyvsp[-2].plist), (yyvsp[-1].val), (yyvsp[0].sentry), (const char *) NULL, (yylsp[-6]).first_line, (yyloc).last_line);
		  if ((yyvsp[0].sentry) != 0) { 
		    (yyvsp[0].sentry)->con1 = new SdagConstruct(SIDENT, (yyvsp[-3].strval));
                    (yyvsp[0].sentry)->setEntry((yyval.entry));
                    (yyvsp[0].sentry)->param = new ParamList((yyvsp[-2].plist));
                  }
		}
#line 3770 "y.tab.c" /* yacc.c:1646  */
    break;

  case 245:
#line 875 "xi-grammar.y" /* yacc.c:1646  */
    { 
                  Entry *e = new Entry(lineno, (yyvsp[-3].intval), 0, (yyvsp[-2].strval), (yyvsp[-1].plist),  0, (yyvsp[0].sentry), (const char *) NULL, (yylsp[-4]).first_line, (yyloc).last_line);
                  if ((yyvsp[0].sentry) != 0) {
		    (yyvsp[0].sentry)->con1 = new SdagConstruct(SIDENT, (yyvsp[-2].strval));
                    (yyvsp[0].sentry)->setEntry((yyval.entry));
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
#line 3790 "y.tab.c" /* yacc.c:1646  */
    break;

  case 246:
#line 891 "xi-grammar.y" /* yacc.c:1646  */
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
#line 3808 "y.tab.c" /* yacc.c:1646  */
    break;

  case 247:
#line 907 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.accelBlock) = new AccelBlock(lineno, new XStr((yyvsp[-2].strval))); }
#line 3814 "y.tab.c" /* yacc.c:1646  */
    break;

  case 248:
#line 909 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.accelBlock) = new AccelBlock(lineno, NULL); }
#line 3820 "y.tab.c" /* yacc.c:1646  */
    break;

  case 249:
#line 913 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3826 "y.tab.c" /* yacc.c:1646  */
    break;

  case 250:
#line 917 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3832 "y.tab.c" /* yacc.c:1646  */
    break;

  case 251:
#line 919 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[-1].intval); }
#line 3838 "y.tab.c" /* yacc.c:1646  */
    break;

  case 252:
#line 921 "xi-grammar.y" /* yacc.c:1646  */
    { ERROR("invalid entry method attribute list",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
#line 3847 "y.tab.c" /* yacc.c:1646  */
    break;

  case 253:
#line 928 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[0].intval); }
#line 3853 "y.tab.c" /* yacc.c:1646  */
    break;

  case 254:
#line 930 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[-2].intval) | (yyvsp[0].intval); }
#line 3859 "y.tab.c" /* yacc.c:1646  */
    break;

  case 255:
#line 934 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = STHREADED; }
#line 3865 "y.tab.c" /* yacc.c:1646  */
    break;

  case 256:
#line 936 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SSYNC; }
#line 3871 "y.tab.c" /* yacc.c:1646  */
    break;

  case 257:
#line 938 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SIGET; }
#line 3877 "y.tab.c" /* yacc.c:1646  */
    break;

  case 258:
#line 940 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SLOCKED; }
#line 3883 "y.tab.c" /* yacc.c:1646  */
    break;

  case 259:
#line 942 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SCREATEHERE; }
#line 3889 "y.tab.c" /* yacc.c:1646  */
    break;

  case 260:
#line 944 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SCREATEHOME; }
#line 3895 "y.tab.c" /* yacc.c:1646  */
    break;

  case 261:
#line 946 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SNOKEEP; }
#line 3901 "y.tab.c" /* yacc.c:1646  */
    break;

  case 262:
#line 948 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SNOTRACE; }
#line 3907 "y.tab.c" /* yacc.c:1646  */
    break;

  case 263:
#line 950 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SAPPWORK; }
#line 3913 "y.tab.c" /* yacc.c:1646  */
    break;

  case 264:
#line 952 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SIMMEDIATE; }
#line 3919 "y.tab.c" /* yacc.c:1646  */
    break;

  case 265:
#line 954 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SSKIPSCHED; }
#line 3925 "y.tab.c" /* yacc.c:1646  */
    break;

  case 266:
#line 956 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SINLINE; }
#line 3931 "y.tab.c" /* yacc.c:1646  */
    break;

  case 267:
#line 958 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SLOCAL; }
#line 3937 "y.tab.c" /* yacc.c:1646  */
    break;

  case 268:
#line 960 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SPYTHON; }
#line 3943 "y.tab.c" /* yacc.c:1646  */
    break;

  case 269:
#line 962 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SMEM; }
#line 3949 "y.tab.c" /* yacc.c:1646  */
    break;

  case 270:
#line 964 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SREDUCE; }
#line 3955 "y.tab.c" /* yacc.c:1646  */
    break;

  case 271:
#line 966 "xi-grammar.y" /* yacc.c:1646  */
    {
#ifdef CMK_USING_XLC
        WARNING("a known bug in xl compilers (PMR 18366,122,000) currently breaks "
                "aggregate entry methods.\n"
                "Until a fix is released, this tag will be ignored on those compilers.",
                (yylsp[0]).first_column, (yylsp[0]).last_column, (yylsp[0]).first_line);
        (yyval.intval) = 0;
#else
        (yyval.intval) = SAGGREGATE;
#endif
    }
#line 3971 "y.tab.c" /* yacc.c:1646  */
    break;

  case 272:
#line 978 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("invalid entry method attribute",
		        (yylsp[0]).first_column, (yylsp[0]).last_column);
		  yyclearin;
		  yyerrok;
		}
#line 3982 "y.tab.c" /* yacc.c:1646  */
    break;

  case 273:
#line 987 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 3988 "y.tab.c" /* yacc.c:1646  */
    break;

  case 274:
#line 989 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 3994 "y.tab.c" /* yacc.c:1646  */
    break;

  case 275:
#line 991 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4000 "y.tab.c" /* yacc.c:1646  */
    break;

  case 276:
#line 995 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = ""; }
#line 4006 "y.tab.c" /* yacc.c:1646  */
    break;

  case 277:
#line 997 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4012 "y.tab.c" /* yacc.c:1646  */
    break;

  case 278:
#line 999 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s, %s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4022 "y.tab.c" /* yacc.c:1646  */
    break;

  case 279:
#line 1007 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = ""; }
#line 4028 "y.tab.c" /* yacc.c:1646  */
    break;

  case 280:
#line 1009 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4034 "y.tab.c" /* yacc.c:1646  */
    break;

  case 281:
#line 1011 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s[%s]%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4044 "y.tab.c" /* yacc.c:1646  */
    break;

  case 282:
#line 1017 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s{%s}%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4054 "y.tab.c" /* yacc.c:1646  */
    break;

  case 283:
#line 1023 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s(%s)%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4064 "y.tab.c" /* yacc.c:1646  */
    break;

  case 284:
#line 1029 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"(%s)%s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4074 "y.tab.c" /* yacc.c:1646  */
    break;

  case 285:
#line 1037 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval));
		}
#line 4083 "y.tab.c" /* yacc.c:1646  */
    break;

  case 286:
#line 1044 "xi-grammar.y" /* yacc.c:1646  */
    { 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			(yyval.intval) = 0;
		}
#line 4093 "y.tab.c" /* yacc.c:1646  */
    break;

  case 287:
#line 1052 "xi-grammar.y" /* yacc.c:1646  */
    { 
			in_braces=0;
			(yyval.intval) = 0;
		}
#line 4102 "y.tab.c" /* yacc.c:1646  */
    break;

  case 288:
#line 1059 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[0].type));}
#line 4108 "y.tab.c" /* yacc.c:1646  */
    break;

  case 289:
#line 1061 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval)); (yyval.pname)->setConditional((yyvsp[0].intval)); }
#line 4114 "y.tab.c" /* yacc.c:1646  */
    break;

  case 290:
#line 1063 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[-3].type),(yyvsp[-2].strval),0,(yyvsp[0].val));}
#line 4120 "y.tab.c" /* yacc.c:1646  */
    break;

  case 291:
#line 1065 "xi-grammar.y" /* yacc.c:1646  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
		}
#line 4129 "y.tab.c" /* yacc.c:1646  */
    break;

  case 292:
#line 1070 "xi-grammar.y" /* yacc.c:1646  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
			(yyval.pname)->setRdma(true);
		}
#line 4139 "y.tab.c" /* yacc.c:1646  */
    break;

  case 293:
#line 1077 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READONLY; }
#line 4145 "y.tab.c" /* yacc.c:1646  */
    break;

  case 294:
#line 1078 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READWRITE; }
#line 4151 "y.tab.c" /* yacc.c:1646  */
    break;

  case 295:
#line 1079 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_WRITEONLY; }
#line 4157 "y.tab.c" /* yacc.c:1646  */
    break;

  case 296:
#line 1082 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr((yyvsp[0].strval)); }
#line 4163 "y.tab.c" /* yacc.c:1646  */
    break;

  case 297:
#line 1083 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "->" << (yyvsp[0].strval); }
#line 4169 "y.tab.c" /* yacc.c:1646  */
    break;

  case 298:
#line 1084 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[-2].xstrptr)) << "." << (yyvsp[0].strval); }
#line 4175 "y.tab.c" /* yacc.c:1646  */
    break;

  case 299:
#line 1086 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << *((yyvsp[-1].xstrptr)) << "]";
                  delete (yyvsp[-3].xstrptr);
                  delete (yyvsp[-1].xstrptr);
                }
#line 4186 "y.tab.c" /* yacc.c:1646  */
    break;

  case 300:
#line 1093 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << (yyvsp[-1].strval) << "]";
                  delete (yyvsp[-3].xstrptr);
                }
#line 4196 "y.tab.c" /* yacc.c:1646  */
    break;

  case 301:
#line 1099 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "(" << *((yyvsp[-1].xstrptr)) << ")";
                  delete (yyvsp[-3].xstrptr);
                  delete (yyvsp[-1].xstrptr);
                }
#line 4207 "y.tab.c" /* yacc.c:1646  */
    break;

  case 302:
#line 1108 "xi-grammar.y" /* yacc.c:1646  */
    {
                  in_bracket = 0;
                  (yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName(), (yyvsp[-1].strval));
                }
#line 4216 "y.tab.c" /* yacc.c:1646  */
    break;

  case 303:
#line 1115 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[-6].intval));
                }
#line 4226 "y.tab.c" /* yacc.c:1646  */
    break;

  case 304:
#line 1121 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType(Parameter::ACCEL_BUFFER_TYPE_READWRITE);
		}
#line 4236 "y.tab.c" /* yacc.c:1646  */
    break;

  case 305:
#line 1127 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.pname) = (yyvsp[-3].pname);
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[-5].intval));
		}
#line 4246 "y.tab.c" /* yacc.c:1646  */
    break;

  case 306:
#line 1135 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 4252 "y.tab.c" /* yacc.c:1646  */
    break;

  case 307:
#line 1137 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 4258 "y.tab.c" /* yacc.c:1646  */
    break;

  case 308:
#line 1141 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 4264 "y.tab.c" /* yacc.c:1646  */
    break;

  case 309:
#line 1143 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 4270 "y.tab.c" /* yacc.c:1646  */
    break;

  case 310:
#line 1147 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = (yyvsp[-1].plist); }
#line 4276 "y.tab.c" /* yacc.c:1646  */
    break;

  case 311:
#line 1149 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList(new Parameter(0, new BuiltinType("void"))); }
#line 4282 "y.tab.c" /* yacc.c:1646  */
    break;

  case 312:
#line 1153 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = (yyvsp[-1].plist); }
#line 4288 "y.tab.c" /* yacc.c:1646  */
    break;

  case 313:
#line 1155 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = 0; }
#line 4294 "y.tab.c" /* yacc.c:1646  */
    break;

  case 314:
#line 1159 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = 0; }
#line 4300 "y.tab.c" /* yacc.c:1646  */
    break;

  case 315:
#line 1161 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4306 "y.tab.c" /* yacc.c:1646  */
    break;

  case 316:
#line 1165 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = 0; }
#line 4312 "y.tab.c" /* yacc.c:1646  */
    break;

  case 317:
#line 1167 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[0].sc)); }
#line 4318 "y.tab.c" /* yacc.c:1646  */
    break;

  case 318:
#line 1169 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[-2].slist)); }
#line 4324 "y.tab.c" /* yacc.c:1646  */
    break;

  case 319:
#line 1173 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.slist) = new SListConstruct((yyvsp[0].sc)); }
#line 4330 "y.tab.c" /* yacc.c:1646  */
    break;

  case 320:
#line 1175 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.slist) = new SListConstruct((yyvsp[-1].sc), (yyvsp[0].slist));  }
#line 4336 "y.tab.c" /* yacc.c:1646  */
    break;

  case 321:
#line 1179 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.olist) = new OListConstruct((yyvsp[0].sc)); }
#line 4342 "y.tab.c" /* yacc.c:1646  */
    break;

  case 322:
#line 1181 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.olist) = new OListConstruct((yyvsp[-1].sc), (yyvsp[0].slist)); }
#line 4348 "y.tab.c" /* yacc.c:1646  */
    break;

  case 323:
#line 1185 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.clist) = new CaseListConstruct((yyvsp[0].when)); }
#line 4354 "y.tab.c" /* yacc.c:1646  */
    break;

  case 324:
#line 1187 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.clist) = new CaseListConstruct((yyvsp[-1].when), (yyvsp[0].clist)); }
#line 4360 "y.tab.c" /* yacc.c:1646  */
    break;

  case 325:
#line 1189 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("case blocks can only contain when clauses",
		        (yylsp[0]).first_column, (yylsp[0]).last_column);
		  (yyval.clist) = 0;
		}
#line 4370 "y.tab.c" /* yacc.c:1646  */
    break;

  case 326:
#line 1197 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4376 "y.tab.c" /* yacc.c:1646  */
    break;

  case 327:
#line 1199 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = 0; }
#line 4382 "y.tab.c" /* yacc.c:1646  */
    break;

  case 328:
#line 1203 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-2].entrylist), 0); }
#line 4388 "y.tab.c" /* yacc.c:1646  */
    break;

  case 329:
#line 1205 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-1].entrylist), (yyvsp[0].sc)); }
#line 4394 "y.tab.c" /* yacc.c:1646  */
    break;

  case 330:
#line 1207 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-3].entrylist), (yyvsp[-1].slist)); }
#line 4400 "y.tab.c" /* yacc.c:1646  */
    break;

  case 331:
#line 1211 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4406 "y.tab.c" /* yacc.c:1646  */
    break;

  case 332:
#line 1213 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4412 "y.tab.c" /* yacc.c:1646  */
    break;

  case 333:
#line 1215 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4418 "y.tab.c" /* yacc.c:1646  */
    break;

  case 334:
#line 1217 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4424 "y.tab.c" /* yacc.c:1646  */
    break;

  case 335:
#line 1219 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4430 "y.tab.c" /* yacc.c:1646  */
    break;

  case 336:
#line 1221 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4436 "y.tab.c" /* yacc.c:1646  */
    break;

  case 337:
#line 1223 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4442 "y.tab.c" /* yacc.c:1646  */
    break;

  case 338:
#line 1225 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4448 "y.tab.c" /* yacc.c:1646  */
    break;

  case 339:
#line 1227 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4454 "y.tab.c" /* yacc.c:1646  */
    break;

  case 340:
#line 1229 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4460 "y.tab.c" /* yacc.c:1646  */
    break;

  case 341:
#line 1231 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4466 "y.tab.c" /* yacc.c:1646  */
    break;

  case 342:
#line 1233 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4472 "y.tab.c" /* yacc.c:1646  */
    break;

  case 343:
#line 1237 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new SerialConstruct((yyvsp[-2].strval), (yyvsp[-4].strval), (yylsp[-3]).first_line); }
#line 4478 "y.tab.c" /* yacc.c:1646  */
    break;

  case 344:
#line 1239 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new OverlapConstruct((yyvsp[-1].olist)); }
#line 4484 "y.tab.c" /* yacc.c:1646  */
    break;

  case 345:
#line 1241 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = (yyvsp[0].when); }
#line 4490 "y.tab.c" /* yacc.c:1646  */
    break;

  case 346:
#line 1243 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new CaseConstruct((yyvsp[-1].clist)); }
#line 4496 "y.tab.c" /* yacc.c:1646  */
    break;

  case 347:
#line 1245 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForConstruct((yyvsp[-8].intexpr), (yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4502 "y.tab.c" /* yacc.c:1646  */
    break;

  case 348:
#line 1247 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForConstruct((yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4508 "y.tab.c" /* yacc.c:1646  */
    break;

  case 349:
#line 1249 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-9].strval)), (yyvsp[-6].intexpr),
		             (yyvsp[-4].intexpr), (yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4515 "y.tab.c" /* yacc.c:1646  */
    break;

  case 350:
#line 1252 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-11].strval)), (yyvsp[-8].intexpr),
		             (yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4522 "y.tab.c" /* yacc.c:1646  */
    break;

  case 351:
#line 1255 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new IfConstruct((yyvsp[-3].intexpr), (yyvsp[-1].sc), (yyvsp[0].sc)); }
#line 4528 "y.tab.c" /* yacc.c:1646  */
    break;

  case 352:
#line 1257 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new IfConstruct((yyvsp[-5].intexpr), (yyvsp[-2].slist), (yyvsp[0].sc)); }
#line 4534 "y.tab.c" /* yacc.c:1646  */
    break;

  case 353:
#line 1259 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new WhileConstruct((yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4540 "y.tab.c" /* yacc.c:1646  */
    break;

  case 354:
#line 1261 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new WhileConstruct((yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4546 "y.tab.c" /* yacc.c:1646  */
    break;

  case 355:
#line 1263 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new SerialConstruct((yyvsp[-2].strval), NULL, (yyloc).first_line); }
#line 4552 "y.tab.c" /* yacc.c:1646  */
    break;

  case 356:
#line 1265 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("unknown SDAG construct or malformed entry method declaration.\n"
		        "You may have forgotten to terminate a previous entry method declaration with a"
		        " semicolon or forgotten to mark a block of sequential SDAG code as 'serial'",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
#line 4564 "y.tab.c" /* yacc.c:1646  */
    break;

  case 357:
#line 1275 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = 0; }
#line 4570 "y.tab.c" /* yacc.c:1646  */
    break;

  case 358:
#line 1277 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ElseConstruct((yyvsp[0].sc)); }
#line 4576 "y.tab.c" /* yacc.c:1646  */
    break;

  case 359:
#line 1279 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ElseConstruct((yyvsp[-1].slist)); }
#line 4582 "y.tab.c" /* yacc.c:1646  */
    break;

  case 360:
#line 1283 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intexpr) = new IntExprConstruct((yyvsp[0].strval)); }
#line 4588 "y.tab.c" /* yacc.c:1646  */
    break;

  case 361:
#line 1287 "xi-grammar.y" /* yacc.c:1646  */
    { in_int_expr = 0; (yyval.intval) = 0; }
#line 4594 "y.tab.c" /* yacc.c:1646  */
    break;

  case 362:
#line 1291 "xi-grammar.y" /* yacc.c:1646  */
    { in_int_expr = 1; (yyval.intval) = 0; }
#line 4600 "y.tab.c" /* yacc.c:1646  */
    break;

  case 363:
#line 1295 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[-1].strval), (yyvsp[0].plist), 0, 0, 0, (yyloc).first_line, (yyloc).last_line);
		}
#line 4608 "y.tab.c" /* yacc.c:1646  */
    break;

  case 364:
#line 1299 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[-4].strval), (yyvsp[0].plist), 0, 0, (yyvsp[-2].strval), (yyloc).first_line, (yyloc).last_line);
		}
#line 4616 "y.tab.c" /* yacc.c:1646  */
    break;

  case 365:
#line 1305 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.entrylist) = new EntryList((yyvsp[0].entry)); }
#line 4622 "y.tab.c" /* yacc.c:1646  */
    break;

  case 366:
#line 1307 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.entrylist) = new EntryList((yyvsp[-2].entry),(yyvsp[0].entrylist)); }
#line 4628 "y.tab.c" /* yacc.c:1646  */
    break;

  case 367:
#line 1311 "xi-grammar.y" /* yacc.c:1646  */
    { in_bracket=1; }
#line 4634 "y.tab.c" /* yacc.c:1646  */
    break;

  case 368:
#line 1314 "xi-grammar.y" /* yacc.c:1646  */
    { in_bracket=0; }
#line 4640 "y.tab.c" /* yacc.c:1646  */
    break;

  case 369:
#line 1318 "xi-grammar.y" /* yacc.c:1646  */
    { if (!macroDefined((yyvsp[0].strval), 1)) in_comment = 1; }
#line 4646 "y.tab.c" /* yacc.c:1646  */
    break;

  case 370:
#line 1322 "xi-grammar.y" /* yacc.c:1646  */
    { if (!macroDefined((yyvsp[0].strval), 0)) in_comment = 1; }
#line 4652 "y.tab.c" /* yacc.c:1646  */
    break;


#line 4656 "y.tab.c" /* yacc.c:1646  */
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
#line 1325 "xi-grammar.y" /* yacc.c:1906  */


void yyerror(const char *msg) { }
