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
    REDUCTIONTARGET = 326,
    CASE = 327
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
#define REDUCTIONTARGET 326
#define CASE 327

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

#line 341 "y.tab.c" /* yacc.c:355  */
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

#line 370 "y.tab.c" /* yacc.c:358  */

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
#define YYFINAL  55
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   1473

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  89
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  116
/* YYNRULES -- Number of rules.  */
#define YYNRULES  364
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  710

/* YYTRANSLATE[YYX] -- Symbol number corresponding to YYX as returned
   by yylex, with out-of-bounds checking.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   327

#define YYTRANSLATE(YYX)                                                \
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, without out-of-bounds checking.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    83,     2,
      81,    82,    80,     2,    77,    87,    88,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    74,    73,
      78,    86,    79,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    84,     2,    85,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    75,     2,    76,     2,     2,     2,     2,
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
      65,    66,    67,    68,    69,    70,    71,    72
};

#if YYDEBUG
  /* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   192,   192,   197,   200,   205,   206,   211,   212,   217,
     219,   220,   221,   223,   224,   225,   227,   228,   229,   230,
     231,   235,   236,   237,   238,   239,   240,   241,   242,   243,
     244,   245,   246,   247,   248,   249,   250,   251,   254,   255,
     256,   257,   258,   259,   260,   261,   262,   263,   264,   266,
     268,   269,   272,   273,   274,   275,   278,   280,   288,   292,
     299,   301,   306,   307,   311,   313,   315,   317,   319,   331,
     333,   335,   337,   343,   345,   347,   349,   351,   353,   355,
     357,   359,   361,   369,   371,   373,   377,   379,   384,   385,
     390,   391,   395,   397,   399,   401,   403,   405,   407,   409,
     411,   413,   415,   417,   419,   421,   423,   427,   428,   435,
     437,   441,   445,   447,   451,   455,   457,   459,   461,   463,
     465,   469,   471,   473,   475,   477,   481,   483,   487,   489,
     493,   497,   502,   503,   507,   511,   516,   517,   522,   523,
     533,   535,   539,   541,   546,   547,   551,   553,   558,   559,
     563,   568,   569,   573,   575,   579,   581,   586,   587,   591,
     592,   595,   599,   601,   605,   607,   612,   613,   617,   619,
     623,   625,   629,   633,   637,   643,   647,   649,   653,   655,
     659,   663,   667,   671,   673,   678,   679,   684,   685,   687,
     691,   693,   695,   699,   701,   705,   709,   711,   713,   715,
     717,   721,   723,   728,   735,   739,   741,   743,   744,   746,
     748,   750,   754,   756,   758,   764,   770,   779,   781,   783,
     789,   797,   799,   802,   806,   810,   812,   817,   819,   827,
     829,   831,   833,   835,   837,   839,   841,   843,   845,   847,
     850,   860,   877,   893,   895,   899,   904,   905,   907,   914,
     916,   920,   922,   924,   926,   928,   930,   932,   934,   936,
     938,   940,   942,   944,   946,   948,   950,   952,   961,   963,
     965,   970,   971,   973,   982,   983,   985,   991,   997,  1003,
    1011,  1018,  1026,  1033,  1035,  1037,  1039,  1046,  1047,  1048,
    1051,  1052,  1053,  1054,  1061,  1067,  1076,  1083,  1089,  1095,
    1103,  1105,  1109,  1111,  1115,  1117,  1121,  1123,  1128,  1129,
    1133,  1135,  1137,  1141,  1143,  1147,  1149,  1153,  1155,  1157,
    1165,  1168,  1171,  1173,  1175,  1179,  1181,  1183,  1185,  1187,
    1189,  1191,  1193,  1195,  1197,  1199,  1201,  1205,  1207,  1209,
    1211,  1213,  1215,  1217,  1220,  1223,  1225,  1227,  1229,  1231,
    1233,  1244,  1245,  1247,  1251,  1255,  1259,  1263,  1267,  1273,
    1275,  1279,  1282,  1286,  1290
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
  "READWRITE", "WRITEONLY", "ACCELBLOCK", "MEMCRITICAL", "REDUCTIONTARGET",
  "CASE", "';'", "':'", "'{'", "'}'", "','", "'<'", "'>'", "'*'", "'('",
  "')'", "'&'", "'['", "']'", "'='", "'-'", "'.'", "$accept", "File",
  "ModuleEList", "OptExtern", "OptSemiColon", "Name", "QualName", "Module",
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
     325,   326,   327,    59,    58,   123,   125,    44,    60,    62,
      42,    40,    41,    38,    91,    93,    61,    45,    46
};
# endif

#define YYPACT_NINF -589

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-589)))

#define YYTABLE_NINF -316

#define yytable_value_is_error(Yytable_value) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
     180,  1303,  1303,    54,  -589,   180,  -589,  -589,  -589,  -589,
    -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,
    -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,
    -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,
    -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,
    -589,  -589,  -589,    42,    42,  -589,  -589,  -589,   592,  -589,
    -589,  -589,     2,  1303,   152,  1303,  1303,   134,   887,    -5,
     936,   592,  -589,  -589,  -589,  1390,    51,    78,  -589,    85,
    -589,  -589,  -589,  -589,   -18,   129,    41,    41,   -11,    78,
      63,    63,    63,    63,   112,   118,  1303,   182,   139,   592,
    -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,   252,  -589,
    -589,  -589,  -589,   183,  -589,  -589,  -589,  -589,  -589,  -589,
    -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,  1390,
    -589,   121,  -589,  -589,  -589,  -589,   236,   149,  -589,  -589,
     194,   198,   208,    29,  -589,    78,   592,    85,   161,    59,
     -18,   217,   573,  1408,   194,   198,   208,  -589,     3,    78,
    -589,    78,    78,   235,    78,   225,  -589,    75,  1303,  1303,
    1303,  1303,  1093,   219,   220,   192,  1303,  -589,  -589,  -589,
    1334,   230,    63,    63,    63,    63,   219,   118,  -589,  -589,
    -589,  -589,  -589,  -589,  -589,   270,  -589,  -589,  -589,   181,
    -589,  -589,  1377,  -589,  -589,  -589,  -589,  -589,  -589,  1303,
     232,   257,   -18,   255,   -18,   231,  -589,   239,   234,     8,
    -589,   238,  -589,   -35,    30,    77,   245,    89,    78,  -589,
    -589,   246,   240,   241,   247,   247,   247,   247,  -589,  1303,
     248,   258,   249,  1163,  1303,   266,  1303,  -589,  -589,   254,
     263,   278,  1303,   101,  1303,   280,   244,   183,  1303,  1303,
    1303,  1303,  1303,  1303,  -589,  -589,  -589,  -589,   281,  -589,
     282,  -589,   241,  -589,  -589,   293,   294,   242,   287,   -18,
    -589,    78,  1303,  -589,   286,  -589,   -18,    41,  1377,    41,
      41,  1377,    41,  -589,  -589,    75,  -589,    78,   179,   179,
     179,   179,   289,  -589,   266,  -589,   247,   247,  -589,   192,
     358,   292,   237,  -589,   295,  1334,  -589,  -589,   247,   247,
     247,   247,   247,   214,  1377,  -589,   301,   -18,   255,   -18,
     -18,  -589,   -35,   302,  -589,   298,  -589,   305,   307,   306,
      78,   310,   308,  -589,   314,  -589,  -589,   319,  -589,  -589,
    -589,  -589,  -589,  -589,   179,   179,  -589,  -589,  1408,     4,
     316,  1408,  -589,  -589,  -589,  -589,  -589,   179,   179,   179,
     179,   179,  -589,   358,  -589,  1347,  -589,  -589,  -589,  -589,
    -589,  -589,   313,  -589,  -589,  -589,   317,  -589,    55,   318,
    -589,    78,  -589,   668,   365,   331,   335,   319,  -589,  -589,
    -589,  -589,  1303,  -589,  -589,  -589,  -589,  -589,  -589,  -589,
    -589,   333,  -589,  1303,   -18,   334,   328,  1408,    41,    41,
      41,  -589,  -589,   903,  1023,  -589,   183,  -589,  -589,   329,
     341,    24,   330,  1408,  -589,   336,   339,   340,   342,  -589,
    -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,  -589,
    -589,  -589,   361,  -589,   332,  -589,  -589,   348,   360,   344,
     301,  1303,  -589,   353,   366,  -589,  -589,   120,  -589,  -589,
    -589,  -589,  -589,  -589,  -589,  -589,  -589,   411,  -589,   954,
     488,   301,  -589,  -589,  -589,  -589,    85,  -589,  1303,  -589,
    -589,   368,   369,   368,   397,   376,   400,   368,   381,  -589,
     304,   -18,  -589,  -589,  -589,   437,   301,  -589,   -18,   405,
     -18,   155,   385,   496,   507,  -589,   389,   -18,  1316,   390,
     399,   217,   379,   488,   383,  -589,   395,   384,   388,  -589,
     -18,   397,   321,  -589,   396,   435,   -18,   388,   368,   401,
     368,   408,   400,   368,   409,   -18,   416,  1316,  -589,   183,
    -589,  -589,   439,  -589,   367,   389,   -18,   368,  -589,   576,
     298,  -589,  -589,   418,  -589,  -589,   217,   716,   -18,   442,
     -18,   507,   389,   -18,  1316,   217,  -589,  -589,  -589,  -589,
    -589,  -589,  -589,  -589,  1303,   422,   423,   414,   -18,   428,
     -18,   304,  -589,   301,  -589,  -589,   304,   454,   430,   419,
     388,   429,   -18,   388,   433,  -589,   434,  1408,   955,  -589,
     217,   -18,   440,   443,  -589,   445,   723,  -589,   -18,   368,
     734,  -589,   217,   770,  -589,  1303,  1303,   -18,   438,  -589,
    1303,   388,   -18,  -589,   454,   304,  -589,   449,   -18,   304,
    -589,  -589,   304,   454,  -589,    73,   -27,   459,  1303,   450,
     781,   436,  -589,   448,   -18,   444,   452,   470,  -589,  -589,
    1303,  1233,   446,  1303,  1303,  -589,   131,  -589,   304,  -589,
     -18,  -589,   388,   -18,  -589,   454,   162,   462,   188,  1303,
    -589,   145,  -589,   479,   388,   788,   480,  -589,  -589,  -589,
    -589,  -589,  -589,  -589,   835,   304,  -589,   -18,   304,  -589,
     482,   388,   483,  -589,   842,  -589,   304,  -589,   486,  -589
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
      53,    54,    55,     0,     0,     1,     4,    60,     0,    58,
      59,    82,     6,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    81,    79,    80,     0,     0,     0,    56,    65,
     363,   364,   244,   281,   274,     0,   136,   136,   136,     0,
     144,   144,   144,   144,     0,   138,     0,     0,     0,     0,
      73,   205,   206,    67,    74,    75,    76,    77,     0,    78,
      66,   208,   207,     7,   239,   231,   232,   233,   234,   235,
     237,   238,   236,   229,    71,   230,    72,    63,   106,     0,
      92,    93,    94,    95,   103,   104,     0,    90,   109,   110,
     121,   122,   123,   127,   245,     0,     0,    64,     0,   275,
     274,     0,     0,     0,   115,   116,   117,   118,   129,     0,
     137,     0,     0,     0,     0,   221,   209,     0,     0,     0,
       0,     0,     0,     0,   151,     0,     0,   211,   223,   210,
       0,     0,   144,   144,   144,   144,     0,   138,   196,   197,
     198,   199,   200,     8,    61,   124,   102,   105,    96,    97,
     100,   101,    88,   108,   111,   112,   113,   125,   126,     0,
       0,     0,   274,   271,   274,     0,   282,     0,     0,   119,
     120,     0,   128,   132,   215,   212,     0,   217,     0,   155,
     156,     0,   146,    90,   166,   166,   166,   166,   150,     0,
       0,   153,     0,     0,     0,     0,     0,   142,   143,     0,
     140,   164,     0,   118,     0,   193,     0,     7,     0,     0,
       0,     0,     0,     0,    98,    99,    84,    85,    86,    89,
       0,    83,    90,    70,    57,     0,   272,     0,     0,   274,
     243,     0,     0,   361,   132,   134,   274,   136,     0,   136,
     136,     0,   136,   222,   145,     0,   107,     0,     0,     0,
       0,     0,     0,   175,     0,   152,   166,   166,   139,     0,
     157,   185,     0,   191,   187,     0,   195,    69,   166,   166,
     166,   166,   166,     0,     0,    91,     0,   274,   271,   274,
     274,   279,   132,     0,   133,     0,   130,     0,     0,     0,
       0,     0,     0,   147,   168,   167,   201,     0,   170,   171,
     172,   173,   174,   154,     0,     0,   141,   158,     0,   157,
       0,     0,   190,   188,   189,   192,   194,     0,     0,     0,
       0,     0,   183,   157,    87,     0,    68,   277,   273,   278,
     276,   135,     0,   362,   131,   216,     0,   213,     0,     0,
     218,     0,   228,     0,     0,     0,     0,     0,   224,   225,
     176,   177,     0,   163,   165,   186,   178,   179,   180,   181,
     182,     0,   305,   283,   274,   300,     0,     0,   136,   136,
     136,   169,   248,     0,     0,   226,     7,   227,   204,   159,
       0,   157,     0,     0,   304,     0,     0,     0,     0,   267,
     251,   252,   253,   254,   260,   261,   262,   255,   256,   257,
     258,   259,   148,   263,     0,   265,   266,     0,   249,    56,
       0,     0,   202,     0,     0,   184,   280,     0,   284,   286,
     301,   114,   214,   220,   219,   149,   264,     0,   247,     0,
       0,     0,   160,   161,   269,   268,   270,   285,     0,   250,
     350,     0,     0,     0,     0,     0,   321,     0,     0,   310,
       0,   274,   241,   339,   311,   308,     0,   356,   274,     0,
     274,     0,   359,     0,     0,   320,     0,   274,     0,     0,
       0,     0,     0,     0,     0,   354,     0,     0,     0,   357,
     274,     0,     0,   323,     0,     0,   274,     0,     0,     0,
       0,     0,   321,     0,     0,   274,     0,   317,   319,     7,
     314,   349,     0,   240,     0,     0,   274,     0,   355,     0,
       0,   360,   322,     0,   338,   316,     0,     0,   274,     0,
     274,     0,     0,   274,     0,     0,   340,   318,   312,   309,
     287,   288,   289,   307,     0,     0,   302,     0,   274,     0,
     274,     0,   347,     0,   324,   337,     0,   351,     0,     0,
       0,     0,   274,     0,     0,   336,     0,     0,     0,   306,
       0,   274,     0,     0,   358,     0,     0,   345,   274,     0,
       0,   326,     0,     0,   327,     0,     0,   274,     0,   303,
       0,     0,   274,   348,   351,     0,   352,     0,   274,     0,
     334,   325,     0,   351,   290,     0,     0,     0,     0,     0,
       0,     0,   346,     0,   274,     0,     0,     0,   332,   298,
       0,     0,     0,     0,     0,   296,     0,   242,     0,   342,
     274,   353,     0,   274,   335,   351,     0,     0,     0,     0,
     292,     0,   299,     0,     0,     0,     0,   333,   295,   294,
     293,   291,   297,   341,     0,     0,   329,   274,     0,   343,
       0,     0,     0,   328,     0,   344,     0,   330,     0,   331
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -589,  -589,   559,  -589,  -249,    -1,   -61,   497,   512,   -49,
    -589,  -589,  -589,  -175,  -589,  -197,  -589,  -152,   -75,   -70,
     -69,   -68,  -171,   417,   447,  -589,   -81,  -589,  -589,  -256,
    -589,  -589,   -76,   380,   260,  -589,   -62,   279,  -589,  -589,
     404,   269,  -589,   144,  -589,  -589,  -325,  -589,   -36,   189,
    -589,  -589,  -589,   -40,  -589,  -589,  -589,  -589,  -589,  -589,
    -589,   267,  -589,   272,   520,  -589,   285,   193,   521,  -589,
    -589,   364,  -589,  -589,  -589,  -589,   200,  -589,   203,  -589,
     133,  -589,  -589,   288,   -82,     6,   -57,  -508,  -589,  -589,
    -523,  -589,  -589,  -380,    20,  -437,  -589,  -589,   107,  -500,
      60,  -495,    99,  -491,  -589,  -395,  -588,  -484,  -522,  -450,
    -589,   111,   135,    97,  -589,  -589
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     3,     4,    68,   194,   233,   137,     5,    59,    69,
      70,    71,   268,   269,   270,   203,   138,   234,   139,   154,
     155,   156,   157,   158,   143,   144,   271,   335,   284,   285,
     101,   102,   161,   176,   249,   250,   168,   231,   476,   241,
     173,   242,   232,   358,   464,   359,   360,   103,   298,   345,
     104,   105,   106,   174,   107,   188,   189,   190,   191,   192,
     362,   313,   255,   256,   394,   109,   348,   395,   396,   111,
     112,   166,   179,   397,   398,   126,   399,    72,   145,   424,
     457,   458,   487,   277,   525,   414,   501,   217,   415,   585,
     645,   628,   586,   416,   587,   376,   555,   523,   502,   519,
     534,   546,   516,   503,   548,   520,   617,   526,   559,   508,
     512,   513,   286,   384,    73,    74
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[] =
{
      53,    54,   151,    79,   159,   140,   141,   142,   317,   253,
      84,   162,   164,   551,   165,   567,   147,   235,   236,   237,
     550,   357,   127,   480,   251,   160,   528,   547,   334,   169,
     170,   171,   563,   537,   403,   565,   296,   435,   149,   148,
     220,   357,    75,   510,   505,   220,   652,   517,   411,   283,
     181,   664,   577,   470,    55,   658,   547,   466,   595,   140,
     141,   142,    76,   150,    80,    81,   207,   605,   215,   524,
     209,   113,   589,   163,   529,   326,   381,   160,   620,   604,
    -162,   623,   218,   547,   221,   504,   222,   687,   568,   221,
     570,   613,   306,   573,   307,   177,   615,   210,   223,   254,
     224,   225,   630,   227,   148,   229,   612,   590,   466,   650,
     467,   287,   208,   338,   641,    57,   341,    58,   533,   535,
     258,   259,   260,   261,   230,   666,   146,   631,   504,   148,
     275,    78,   278,   244,   212,   653,   419,   676,   678,   656,
     213,   681,   657,   214,   253,   152,   262,   167,   651,   374,
     685,   148,   659,   165,   660,   288,   614,   661,   289,   148,
     662,   663,   694,   148,   592,   128,   153,   291,   683,   638,
     292,   240,   597,    78,   484,   485,   535,   462,  -187,   704,
    -187,   196,    78,     1,     2,   197,   684,   312,   130,   131,
     132,   133,   134,   135,   136,   700,   172,   331,   702,   299,
     300,   301,   175,    77,   336,    78,   708,    82,   272,    83,
     682,   337,   660,   339,   340,   661,   342,   180,   662,   663,
     332,   636,   344,   148,   692,   640,   660,   202,   643,   661,
     247,   248,   662,   663,   254,   211,   375,   178,   302,   283,
     264,   265,   240,   660,   688,   377,   661,   379,   380,   662,
     663,   311,   346,   314,   347,   669,   193,   318,   319,   320,
     321,   322,   323,   182,   183,   184,   185,   186,   187,   660,
     354,   355,   661,   690,   204,   662,   663,   402,   205,   388,
     405,   333,   367,   368,   369,   370,   371,   372,   206,   373,
     696,   363,   364,   216,   413,   198,   199,   200,   201,   699,
     578,   226,   228,   243,   245,   490,   257,   207,   273,   707,
     274,   276,   280,   279,   281,   238,   344,   295,   282,   202,
     392,   297,   490,   316,   329,    85,    86,    87,    88,    89,
     290,   294,   432,   303,   305,   304,   413,    96,    97,   308,
     309,    98,   436,   437,   438,   491,   492,   493,   494,   495,
     496,   497,   413,   310,   140,   141,   142,   315,   324,   393,
    -281,   325,   491,   492,   493,   494,   495,   496,   497,   327,
     283,   328,   330,   580,   352,   357,   498,  -281,   361,    83,
    -281,   312,   375,   383,   382,  -281,   386,   385,   387,   389,
     390,   391,   404,   498,   417,  -203,    83,   562,   418,   420,
     490,   429,  -281,   128,   153,   393,   486,   426,   427,   430,
     434,   433,   431,   463,   465,   469,   475,   477,   471,   521,
      78,   472,   473,   460,   474,    -9,   130,   131,   132,   133,
     134,   135,   136,   478,   581,   582,   490,   479,   482,   483,
     491,   492,   493,   494,   495,   496,   497,   488,   560,   507,
     511,   514,   583,   509,   566,   515,   518,   522,   527,   536,
     481,   545,   531,   575,    83,   552,   549,   554,   556,   557,
     558,   498,   564,   584,    83,  -313,   491,   492,   493,   494,
     495,   496,   497,   571,   574,   569,   598,   506,   600,   490,
     545,   603,   576,   579,   594,   599,   607,   490,   588,   609,
     608,   611,   616,   618,   619,   621,   610,   498,   490,   624,
      83,  -315,   625,   670,   632,   602,   648,   545,   673,   633,
     622,   634,   654,   667,   671,   679,   626,   584,   674,   491,
     492,   493,   494,   495,   496,   497,   637,   491,   492,   493,
     494,   495,   496,   497,   665,   647,   675,   689,   491,   492,
     493,   494,   495,   496,   497,   693,   655,   697,   703,   705,
     498,   499,   709,   500,    56,   100,    60,   263,   498,   356,
     219,   532,   672,   353,   343,   468,   195,   490,   246,   498,
     421,   365,    83,   606,   349,   350,   351,   366,   108,   110,
     428,   686,   293,    61,   425,    -5,    -5,    62,    -5,    -5,
      -5,    -5,    -5,    -5,    -5,    -5,    -5,    -5,    -5,   128,
      -5,    -5,   489,   627,    -5,   701,   378,   491,   492,   493,
     494,   495,   496,   497,   644,   646,    78,   461,   629,   649,
     553,   601,   130,   131,   132,   133,   134,   135,   136,   400,
     401,   572,   561,    63,    64,     0,   530,   644,   498,    65,
      66,   591,   406,   407,   408,   409,   410,   593,     0,   644,
     644,    67,   680,   644,     0,     0,     0,    -5,   -62,   422,
       0,  -246,  -246,  -246,     0,  -246,  -246,  -246,   691,  -246,
    -246,  -246,  -246,  -246,     0,     0,     0,  -246,  -246,  -246,
    -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,
    -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,
    -246,  -246,  -246,  -246,  -246,  -246,  -246,   490,  -246,     0,
    -246,  -246,     0,     0,   490,     0,     0,  -246,  -246,  -246,
    -246,  -246,  -246,  -246,  -246,   490,     0,  -246,  -246,  -246,
    -246,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   423,     0,     0,     0,     0,   491,   492,   493,
     494,   495,   496,   497,   491,   492,   493,   494,   495,   496,
     497,   490,     0,     0,     0,   491,   492,   493,   494,   495,
     496,   497,   490,     0,     0,     0,     0,     0,   498,   490,
       0,   596,     0,     0,     0,   498,     0,     0,   635,     0,
       0,     0,     0,     0,     0,     0,   498,     0,     0,   639,
       0,   491,   492,   493,   494,   495,   496,   497,     0,     0,
       0,     0,   491,   492,   493,   494,   495,   496,   497,   491,
     492,   493,   494,   495,   496,   497,   490,     0,     0,     0,
       0,     0,   498,   490,     0,   642,     0,     0,     0,     0,
       0,     0,     0,   498,     0,     0,   668,     0,     0,     0,
     498,     0,     0,   695,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   491,   492,   493,   494,
     495,   496,   497,   491,   492,   493,   494,   495,   496,   497,
       1,     2,     0,    85,    86,    87,    88,    89,    90,    91,
      92,    93,    94,    95,   439,    96,    97,   498,     0,    98,
     698,     0,     0,     0,   498,     0,     0,   706,     0,     0,
       0,     0,     0,     0,   440,     0,   441,   442,   443,   444,
     445,   446,     0,     0,   447,   448,   449,   450,   451,     0,
       0,     0,   114,   115,   116,   117,     0,   118,   119,   120,
     121,   122,   452,   453,     0,   439,     0,     0,     0,     0,
       0,   580,    99,     0,     0,     0,     0,     0,     0,   454,
       0,     0,     0,   455,   456,   440,   123,   441,   442,   443,
     444,   445,   446,     0,     0,   447,   448,   449,   450,   451,
       0,   128,   153,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   452,   453,     0,     0,     0,    78,   124,
       0,     0,   125,     0,   130,   131,   132,   133,   134,   135,
     136,     0,   581,   582,   455,   456,     6,     7,     8,     0,
       9,    10,    11,     0,    12,    13,    14,    15,    16,     0,
       0,     0,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,   128,
     129,    34,    35,    36,    37,    38,    39,    40,    41,    42,
      43,    44,     0,    45,     0,    46,   459,     0,     0,     0,
       0,     0,   130,   131,   132,   133,   134,   135,   136,    48,
       0,     0,    49,    50,    51,    52,     6,     7,     8,     0,
       9,    10,    11,     0,    12,    13,    14,    15,    16,     0,
       0,     0,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,     0,
       0,    34,    35,    36,    37,    38,    39,    40,    41,    42,
      43,    44,   238,    45,     0,    46,    47,   239,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    48,
       0,     0,    49,    50,    51,    52,     6,     7,     8,     0,
       9,    10,    11,     0,    12,    13,    14,    15,    16,     0,
       0,     0,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,     0,
       0,    34,    35,    36,    37,    38,    39,    40,    41,    42,
      43,    44,     0,    45,     0,    46,    47,   239,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    48,
       0,     0,    49,    50,    51,    52,     6,     7,     8,     0,
       9,    10,    11,     0,    12,    13,    14,    15,    16,     0,
       0,     0,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,     0,
       0,    34,    35,    36,    37,    38,    39,    40,    41,    42,
      43,    44,     0,    45,     0,    46,    47,   677,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    48,
       0,     0,    49,    50,    51,    52,     6,     7,     8,     0,
       9,    10,    11,     0,    12,    13,    14,    15,    16,     0,
       0,     0,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,     0,
       0,    34,    35,    36,    37,    38,    39,    40,    41,    42,
      43,    44,   252,    45,     0,    46,    47,   538,   539,   540,
     494,   541,   542,   543,     0,     0,     0,     0,     0,    48,
     128,   153,    49,    50,    51,    52,     0,     0,     0,     0,
       0,     0,     0,   128,   153,     0,     0,    78,   544,     0,
       0,    83,     0,   130,   131,   132,   133,   134,   135,   136,
      78,     0,     0,     0,     0,     0,   130,   131,   132,   133,
     134,   135,   136,   128,   153,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   128,   129,     0,   412,
      78,   266,   267,     0,     0,     0,   130,   131,   132,   133,
     134,   135,   136,    78,   128,   153,     0,     0,     0,   130,
     131,   132,   133,   134,   135,   136,     0,     0,     0,     0,
       0,    78,     0,     0,     0,     0,     0,   130,   131,   132,
     133,   134,   135,   136
};

static const yytype_int16 yycheck[] =
{
       1,     2,    84,    64,    85,    75,    75,    75,   257,   180,
      67,    87,    88,   521,    89,   537,    77,   169,   170,   171,
     520,    17,    71,   460,   176,    36,   510,   518,   284,    91,
      92,    93,   532,   517,   359,   535,   233,   417,    56,    74,
      37,    17,    40,   493,   481,    37,   634,   497,   373,    84,
      99,    78,   547,   433,     0,   643,   547,    84,   566,   129,
     129,   129,    63,    81,    65,    66,    37,   575,   150,   506,
     145,    76,   556,    84,   511,   272,   332,    36,   600,   574,
      76,   603,   152,   574,    81,   480,    83,   675,   538,    81,
     540,   591,   244,   543,   246,    96,   596,   146,   159,   180,
     161,   162,   610,   164,    74,    30,   590,   557,    84,   631,
      86,    81,    83,   288,   622,    73,   291,    75,   513,   514,
     182,   183,   184,   185,    49,   648,    75,   611,   523,    74,
     212,    53,   214,   173,    75,   635,    81,   660,   661,   639,
      81,   664,   642,    84,   315,    16,   186,    84,   632,   324,
     672,    74,    79,   228,    81,    78,   593,    84,    81,    74,
      87,    88,   684,    74,   559,    36,    37,    78,   668,   619,
      81,   172,   567,    53,    54,    55,   571,   426,    77,   701,
      79,    60,    53,     3,     4,    64,   670,    86,    59,    60,
      61,    62,    63,    64,    65,   695,    84,   279,   698,   235,
     236,   237,    84,    51,   286,    53,   706,    73,   209,    75,
      79,   287,    81,   289,   290,    84,   292,    78,    87,    88,
     281,   616,   297,    74,    79,   620,    81,    78,   623,    84,
      38,    39,    87,    88,   315,    74,    81,    55,   239,    84,
      59,    60,   243,    81,    82,   327,    84,   329,   330,    87,
      88,   252,    73,   254,    75,   650,    73,   258,   259,   260,
     261,   262,   263,    11,    12,    13,    14,    15,    16,    81,
     306,   307,    84,    85,    80,    87,    88,   358,    80,   340,
     361,   282,   318,   319,   320,   321,   322,    73,    80,    75,
     685,    54,    55,    76,   375,    59,    60,    61,    62,   694,
     549,    66,    77,    84,    84,     1,    76,    37,    76,   704,
      53,    56,    73,    82,    80,    49,   391,    77,    80,    78,
       1,    74,     1,    79,    82,     6,     7,     8,     9,    10,
      85,    85,   414,    85,    85,    77,   417,    18,    19,    85,
      77,    22,   418,   419,   420,    41,    42,    43,    44,    45,
      46,    47,   433,    75,   424,   424,   424,    77,    77,    40,
      56,    79,    41,    42,    43,    44,    45,    46,    47,    76,
      84,    77,    85,     6,    85,    17,    72,    56,    86,    75,
      76,    86,    81,    85,    82,    81,    79,    82,    82,    79,
      82,    77,    76,    72,    81,    76,    75,    76,    81,    81,
       1,   402,    81,    36,    37,    40,   467,    76,    73,    76,
      82,    77,   413,    84,    73,    85,    55,    85,    82,   501,
      53,    82,    82,   424,    82,    81,    59,    60,    61,    62,
      63,    64,    65,    85,    67,    68,     1,    77,    85,    73,
      41,    42,    43,    44,    45,    46,    47,    36,   530,    81,
      53,    75,    85,    84,   536,    55,    75,    20,    53,   516,
     461,   518,    77,   545,    75,    86,    76,    84,    73,    85,
      82,    72,    76,   554,    75,    76,    41,    42,    43,    44,
      45,    46,    47,    75,    75,    84,   568,   488,   570,     1,
     547,   573,    76,    54,    76,    53,    74,     1,   555,    85,
      77,    73,    48,    73,    85,    76,   588,    72,     1,    76,
      75,    76,    78,    77,    74,   572,    78,   574,    74,    76,
     602,    76,    73,    73,    76,    79,   607,   608,    76,    41,
      42,    43,    44,    45,    46,    47,   618,    41,    42,    43,
      44,    45,    46,    47,    85,   627,    76,    85,    41,    42,
      43,    44,    45,    46,    47,    76,   638,    77,    76,    76,
      72,    73,    76,    75,     5,    68,    54,   187,    72,   309,
     153,    75,   654,   304,   295,   431,   129,     1,   174,    72,
     391,   314,    75,   584,   299,   300,   301,   315,    68,    68,
     397,   673,   228,     1,   394,     3,     4,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    14,    15,    16,    36,
      18,    19,   479,   607,    22,   697,   328,    41,    42,    43,
      44,    45,    46,    47,   625,   626,    53,   424,   608,   630,
     523,   571,    59,    60,    61,    62,    63,    64,    65,   354,
     355,   542,   531,    51,    52,    -1,   511,   648,    72,    57,
      58,    75,   367,   368,   369,   370,   371,   560,    -1,   660,
     661,    69,   663,   664,    -1,    -1,    -1,    75,    76,     1,
      -1,     3,     4,     5,    -1,     7,     8,     9,   679,    11,
      12,    13,    14,    15,    -1,    -1,    -1,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    34,    35,    36,    37,    38,    39,    40,    41,
      42,    43,    44,    45,    46,    47,    48,     1,    50,    -1,
      52,    53,    -1,    -1,     1,    -1,    -1,    59,    60,    61,
      62,    63,    64,    65,    66,     1,    -1,    69,    70,    71,
      72,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    84,    -1,    -1,    -1,    -1,    41,    42,    43,
      44,    45,    46,    47,    41,    42,    43,    44,    45,    46,
      47,     1,    -1,    -1,    -1,    41,    42,    43,    44,    45,
      46,    47,     1,    -1,    -1,    -1,    -1,    -1,    72,     1,
      -1,    75,    -1,    -1,    -1,    72,    -1,    -1,    75,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    72,    -1,    -1,    75,
      -1,    41,    42,    43,    44,    45,    46,    47,    -1,    -1,
      -1,    -1,    41,    42,    43,    44,    45,    46,    47,    41,
      42,    43,    44,    45,    46,    47,     1,    -1,    -1,    -1,
      -1,    -1,    72,     1,    -1,    75,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    72,    -1,    -1,    75,    -1,    -1,    -1,
      72,    -1,    -1,    75,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    41,    42,    43,    44,
      45,    46,    47,    41,    42,    43,    44,    45,    46,    47,
       3,     4,    -1,     6,     7,     8,     9,    10,    11,    12,
      13,    14,    15,    16,     1,    18,    19,    72,    -1,    22,
      75,    -1,    -1,    -1,    72,    -1,    -1,    75,    -1,    -1,
      -1,    -1,    -1,    -1,    21,    -1,    23,    24,    25,    26,
      27,    28,    -1,    -1,    31,    32,    33,    34,    35,    -1,
      -1,    -1,     6,     7,     8,     9,    -1,    11,    12,    13,
      14,    15,    49,    50,    -1,     1,    -1,    -1,    -1,    -1,
      -1,     6,    75,    -1,    -1,    -1,    -1,    -1,    -1,    66,
      -1,    -1,    -1,    70,    71,    21,    40,    23,    24,    25,
      26,    27,    28,    -1,    -1,    31,    32,    33,    34,    35,
      -1,    36,    37,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    49,    50,    -1,    -1,    -1,    53,    73,
      -1,    -1,    76,    -1,    59,    60,    61,    62,    63,    64,
      65,    -1,    67,    68,    70,    71,     3,     4,     5,    -1,
       7,     8,     9,    -1,    11,    12,    13,    14,    15,    -1,
      -1,    -1,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    36,
      37,    38,    39,    40,    41,    42,    43,    44,    45,    46,
      47,    48,    -1,    50,    -1,    52,    53,    -1,    -1,    -1,
      -1,    -1,    59,    60,    61,    62,    63,    64,    65,    66,
      -1,    -1,    69,    70,    71,    72,     3,     4,     5,    -1,
       7,     8,     9,    -1,    11,    12,    13,    14,    15,    -1,
      -1,    -1,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    -1,
      -1,    38,    39,    40,    41,    42,    43,    44,    45,    46,
      47,    48,    49,    50,    -1,    52,    53,    54,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    66,
      -1,    -1,    69,    70,    71,    72,     3,     4,     5,    -1,
       7,     8,     9,    -1,    11,    12,    13,    14,    15,    -1,
      -1,    -1,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    -1,
      -1,    38,    39,    40,    41,    42,    43,    44,    45,    46,
      47,    48,    -1,    50,    -1,    52,    53,    54,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    66,
      -1,    -1,    69,    70,    71,    72,     3,     4,     5,    -1,
       7,     8,     9,    -1,    11,    12,    13,    14,    15,    -1,
      -1,    -1,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    -1,
      -1,    38,    39,    40,    41,    42,    43,    44,    45,    46,
      47,    48,    -1,    50,    -1,    52,    53,    54,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    66,
      -1,    -1,    69,    70,    71,    72,     3,     4,     5,    -1,
       7,     8,     9,    -1,    11,    12,    13,    14,    15,    -1,
      -1,    -1,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    -1,
      -1,    38,    39,    40,    41,    42,    43,    44,    45,    46,
      47,    48,    18,    50,    -1,    52,    53,    41,    42,    43,
      44,    45,    46,    47,    -1,    -1,    -1,    -1,    -1,    66,
      36,    37,    69,    70,    71,    72,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    36,    37,    -1,    -1,    53,    72,    -1,
      -1,    75,    -1,    59,    60,    61,    62,    63,    64,    65,
      53,    -1,    -1,    -1,    -1,    -1,    59,    60,    61,    62,
      63,    64,    65,    36,    37,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    36,    37,    -1,    82,
      53,    54,    55,    -1,    -1,    -1,    59,    60,    61,    62,
      63,    64,    65,    53,    36,    37,    -1,    -1,    -1,    59,
      60,    61,    62,    63,    64,    65,    -1,    -1,    -1,    -1,
      -1,    53,    -1,    -1,    -1,    -1,    -1,    59,    60,    61,
      62,    63,    64,    65
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     3,     4,    90,    91,    96,     3,     4,     5,     7,
       8,     9,    11,    12,    13,    14,    15,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    34,    35,    38,    39,    40,    41,    42,    43,
      44,    45,    46,    47,    48,    50,    52,    53,    66,    69,
      70,    71,    72,    94,    94,     0,    91,    73,    75,    97,
      97,     1,     5,    51,    52,    57,    58,    69,    92,    98,
      99,   100,   166,   203,   204,    40,    94,    51,    53,    95,
      94,    94,    73,    75,   175,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    18,    19,    22,    75,
      96,   119,   120,   136,   139,   140,   141,   143,   153,   154,
     157,   158,   159,    76,     6,     7,     8,     9,    11,    12,
      13,    14,    15,    40,    73,    76,   164,    98,    36,    37,
      59,    60,    61,    62,    63,    64,    65,    95,   105,   107,
     108,   109,   110,   113,   114,   167,    75,    95,    74,    56,
      81,   173,    16,    37,   108,   109,   110,   111,   112,   115,
      36,   121,   121,    84,   121,   107,   160,    84,   125,   125,
     125,   125,    84,   129,   142,    84,   122,    94,    55,   161,
      78,    98,    11,    12,    13,    14,    15,    16,   144,   145,
     146,   147,   148,    73,    93,   113,    60,    64,    59,    60,
      61,    62,    78,   104,    80,    80,    80,    37,    83,   107,
      98,    74,    75,    81,    84,   173,    76,   176,   108,   112,
      37,    81,    83,    95,    95,    95,    66,    95,    77,    30,
      49,   126,   131,    94,   106,   106,   106,   106,    49,    54,
      94,   128,   130,    84,   142,    84,   129,    38,    39,   123,
     124,   106,    18,   111,   115,   151,   152,    76,   125,   125,
     125,   125,   142,   122,    59,    60,    54,    55,   101,   102,
     103,   115,    94,    76,    53,   173,    56,   172,   173,    82,
      73,    80,    80,    84,   117,   118,   201,    81,    78,    81,
      85,    78,    81,   160,    85,    77,   104,    74,   137,   137,
     137,   137,    94,    85,    77,    85,   106,   106,    85,    77,
      75,    94,    86,   150,    94,    77,    79,    93,    94,    94,
      94,    94,    94,    94,    77,    79,   104,    76,    77,    82,
      85,   173,    95,    94,   118,   116,   173,   121,   102,   121,
     121,   102,   121,   126,   107,   138,    73,    75,   155,   155,
     155,   155,    85,   130,   137,   137,   123,    17,   132,   134,
     135,    86,   149,    54,    55,   150,   152,   137,   137,   137,
     137,   137,    73,    75,   102,    81,   184,   173,   172,   173,
     173,   118,    82,    85,   202,    82,    79,    82,    95,    79,
      82,    77,     1,    40,   153,   156,   157,   162,   163,   165,
     155,   155,   115,   135,    76,   115,   155,   155,   155,   155,
     155,   135,    82,   115,   174,   177,   182,    81,    81,    81,
      81,   138,     1,    84,   168,   165,    76,    73,   156,    94,
      76,    94,   173,    77,    82,   182,   121,   121,   121,     1,
      21,    23,    24,    25,    26,    27,    28,    31,    32,    33,
      34,    35,    49,    50,    66,    70,    71,   169,   170,    53,
      94,   167,    93,    84,   133,    73,    84,    86,   132,    85,
     182,    82,    82,    82,    82,    55,   127,    85,    85,    77,
     184,    94,    85,    73,    54,    55,    95,   171,    36,   169,
       1,    41,    42,    43,    44,    45,    46,    47,    72,    73,
      75,   175,   187,   192,   194,   184,    94,    81,   198,    84,
     198,    53,   199,   200,    75,    55,   191,   198,    75,   188,
     194,   173,    20,   186,   184,   173,   196,    53,   196,   184,
     201,    77,    75,   194,   189,   194,   175,   196,    41,    42,
      43,    45,    46,    47,    72,   175,   190,   192,   193,    76,
     188,   176,    86,   187,    84,   185,    73,    85,    82,   197,
     173,   200,    76,   188,    76,   188,   173,   197,   198,    84,
     198,    75,   191,   198,    75,   173,    76,   190,    93,    54,
       6,    67,    68,    85,   115,   178,   181,   183,   175,   196,
     198,    75,   194,   202,    76,   176,    75,   194,   173,    53,
     173,   189,   175,   173,   190,   176,    94,    74,    77,    85,
     173,    73,   196,   188,   184,   188,    48,   195,    73,    85,
     197,    76,   173,   197,    76,    78,   115,   174,   180,   183,
     176,   196,    74,    76,    76,    75,   194,   173,   198,    75,
     194,   176,    75,   194,    94,   179,    94,   173,    78,    94,
     197,   196,   195,   188,    73,   173,   188,   188,   195,    79,
      81,    84,    87,    88,    78,    85,   179,    73,    75,   194,
      77,    76,   173,    74,    76,    76,   179,    54,   179,    79,
      94,   179,    79,   188,   196,   197,   173,   195,    82,    85,
      85,    94,    79,    76,   197,    75,   194,    77,    75,   194,
     188,   173,   188,    76,   197,    76,    75,   194,   188,    76
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    89,    90,    91,    91,    92,    92,    93,    93,    94,
      94,    94,    94,    94,    94,    94,    94,    94,    94,    94,
      94,    94,    94,    94,    94,    94,    94,    94,    94,    94,
      94,    94,    94,    94,    94,    94,    94,    94,    94,    94,
      94,    94,    94,    94,    94,    94,    94,    94,    94,    94,
      94,    94,    94,    94,    94,    94,    95,    95,    96,    96,
      97,    97,    98,    98,    99,    99,    99,    99,    99,   100,
     100,   100,   100,   100,   100,   100,   100,   100,   100,   100,
     100,   100,   100,   101,   101,   101,   102,   102,   103,   103,
     104,   104,   105,   105,   105,   105,   105,   105,   105,   105,
     105,   105,   105,   105,   105,   105,   105,   106,   107,   108,
     108,   109,   110,   110,   111,   112,   112,   112,   112,   112,
     112,   113,   113,   113,   113,   113,   114,   114,   115,   115,
     116,   117,   118,   118,   119,   120,   121,   121,   122,   122,
     123,   123,   124,   124,   125,   125,   126,   126,   127,   127,
     128,   129,   129,   130,   130,   131,   131,   132,   132,   133,
     133,   134,   135,   135,   136,   136,   137,   137,   138,   138,
     139,   139,   140,   141,   142,   142,   143,   143,   144,   144,
     145,   146,   147,   148,   148,   149,   149,   150,   150,   150,
     151,   151,   151,   152,   152,   153,   154,   154,   154,   154,
     154,   155,   155,   156,   156,   157,   157,   157,   157,   157,
     157,   157,   158,   158,   158,   158,   158,   159,   159,   159,
     159,   160,   160,   161,   162,   163,   163,   163,   163,   164,
     164,   164,   164,   164,   164,   164,   164,   164,   164,   164,
     165,   165,   165,   166,   166,   167,   168,   168,   168,   169,
     169,   170,   170,   170,   170,   170,   170,   170,   170,   170,
     170,   170,   170,   170,   170,   170,   170,   170,   171,   171,
     171,   172,   172,   172,   173,   173,   173,   173,   173,   173,
     174,   175,   176,   177,   177,   177,   177,   178,   178,   178,
     179,   179,   179,   179,   179,   179,   180,   181,   181,   181,
     182,   182,   183,   183,   184,   184,   185,   185,   186,   186,
     187,   187,   187,   188,   188,   189,   189,   190,   190,   190,
     191,   191,   192,   192,   192,   193,   193,   193,   193,   193,
     193,   193,   193,   193,   193,   193,   193,   194,   194,   194,
     194,   194,   194,   194,   194,   194,   194,   194,   194,   194,
     194,   195,   195,   195,   196,   197,   198,   199,   199,   200,
     200,   201,   202,   203,   204
};

  /* YYR2[YYN] -- Number of symbols on the right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     0,     2,     0,     1,     0,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     4,     3,     3,
       1,     4,     0,     2,     3,     2,     2,     2,     7,     5,
       5,     2,     2,     2,     2,     2,     2,     2,     2,     1,
       1,     1,     1,     1,     1,     1,     1,     3,     0,     1,
       0,     3,     1,     1,     1,     1,     2,     2,     3,     3,
       2,     2,     2,     1,     1,     2,     1,     2,     2,     1,
       1,     2,     2,     2,     8,     1,     1,     1,     1,     2,
       2,     1,     1,     1,     2,     2,     2,     1,     2,     1,
       1,     3,     0,     2,     4,     6,     0,     1,     0,     3,
       1,     3,     1,     1,     0,     3,     1,     3,     0,     1,
       1,     0,     3,     1,     3,     1,     1,     0,     1,     0,
       2,     5,     1,     2,     3,     6,     0,     2,     1,     3,
       5,     5,     5,     5,     4,     3,     6,     6,     5,     5,
       5,     5,     5,     4,     7,     0,     2,     0,     2,     2,
       3,     2,     3,     1,     3,     4,     2,     2,     2,     2,
       2,     1,     4,     0,     2,     1,     1,     1,     1,     2,
       2,     2,     3,     6,     9,     3,     6,     3,     6,     9,
       9,     1,     3,     1,     1,     1,     2,     2,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       7,     5,    13,     5,     2,     1,     0,     3,     1,     1,
       3,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     2,     1,     1,     1,     1,     1,
       1,     0,     1,     3,     0,     1,     5,     5,     5,     4,
       3,     1,     1,     1,     3,     4,     3,     1,     1,     1,
       1,     4,     3,     4,     4,     4,     3,     7,     5,     6,
       1,     3,     1,     3,     3,     2,     3,     2,     0,     3,
       1,     1,     4,     1,     2,     1,     2,     1,     2,     1,
       1,     0,     4,     3,     5,     5,     4,     4,    11,     9,
      12,    14,     6,     8,     5,     7,     3,     5,     4,     1,
       4,    11,     9,    12,    14,     6,     8,     5,     7,     3,
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
#line 193 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.modlist) = (yyvsp[0].modlist); modlist = (yyvsp[0].modlist); }
#line 2215 "y.tab.c" /* yacc.c:1646  */
    break;

  case 3:
#line 197 "xi-grammar.y" /* yacc.c:1646  */
    { 
		  (yyval.modlist) = 0; 
		}
#line 2223 "y.tab.c" /* yacc.c:1646  */
    break;

  case 4:
#line 201 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.modlist) = new AstChildren<Module>(lineno, (yyvsp[-1].module), (yyvsp[0].modlist)); }
#line 2229 "y.tab.c" /* yacc.c:1646  */
    break;

  case 5:
#line 205 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 2235 "y.tab.c" /* yacc.c:1646  */
    break;

  case 6:
#line 207 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 2241 "y.tab.c" /* yacc.c:1646  */
    break;

  case 7:
#line 211 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 2247 "y.tab.c" /* yacc.c:1646  */
    break;

  case 8:
#line 213 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 2253 "y.tab.c" /* yacc.c:1646  */
    break;

  case 9:
#line 218 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 2259 "y.tab.c" /* yacc.c:1646  */
    break;

  case 10:
#line 219 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MODULE); YYABORT; }
#line 2265 "y.tab.c" /* yacc.c:1646  */
    break;

  case 11:
#line 220 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MAINMODULE); YYABORT; }
#line 2271 "y.tab.c" /* yacc.c:1646  */
    break;

  case 12:
#line 221 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(EXTERN); YYABORT; }
#line 2277 "y.tab.c" /* yacc.c:1646  */
    break;

  case 13:
#line 223 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITCALL); YYABORT; }
#line 2283 "y.tab.c" /* yacc.c:1646  */
    break;

  case 14:
#line 224 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITNODE); YYABORT; }
#line 2289 "y.tab.c" /* yacc.c:1646  */
    break;

  case 15:
#line 225 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITPROC); YYABORT; }
#line 2295 "y.tab.c" /* yacc.c:1646  */
    break;

  case 16:
#line 227 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CHARE); }
#line 2301 "y.tab.c" /* yacc.c:1646  */
    break;

  case 17:
#line 228 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MAINCHARE); }
#line 2307 "y.tab.c" /* yacc.c:1646  */
    break;

  case 18:
#line 229 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(GROUP); }
#line 2313 "y.tab.c" /* yacc.c:1646  */
    break;

  case 19:
#line 230 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NODEGROUP); }
#line 2319 "y.tab.c" /* yacc.c:1646  */
    break;

  case 20:
#line 231 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ARRAY); }
#line 2325 "y.tab.c" /* yacc.c:1646  */
    break;

  case 21:
#line 235 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INCLUDE); YYABORT; }
#line 2331 "y.tab.c" /* yacc.c:1646  */
    break;

  case 22:
#line 236 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(STACKSIZE); YYABORT; }
#line 2337 "y.tab.c" /* yacc.c:1646  */
    break;

  case 23:
#line 237 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(THREADED); YYABORT; }
#line 2343 "y.tab.c" /* yacc.c:1646  */
    break;

  case 24:
#line 238 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(TEMPLATE); YYABORT; }
#line 2349 "y.tab.c" /* yacc.c:1646  */
    break;

  case 25:
#line 239 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(SYNC); YYABORT; }
#line 2355 "y.tab.c" /* yacc.c:1646  */
    break;

  case 26:
#line 240 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IGET); YYABORT; }
#line 2361 "y.tab.c" /* yacc.c:1646  */
    break;

  case 27:
#line 241 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(EXCLUSIVE); YYABORT; }
#line 2367 "y.tab.c" /* yacc.c:1646  */
    break;

  case 28:
#line 242 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IMMEDIATE); YYABORT; }
#line 2373 "y.tab.c" /* yacc.c:1646  */
    break;

  case 29:
#line 243 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(SKIPSCHED); YYABORT; }
#line 2379 "y.tab.c" /* yacc.c:1646  */
    break;

  case 30:
#line 244 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INLINE); YYABORT; }
#line 2385 "y.tab.c" /* yacc.c:1646  */
    break;

  case 31:
#line 245 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(VIRTUAL); YYABORT; }
#line 2391 "y.tab.c" /* yacc.c:1646  */
    break;

  case 32:
#line 246 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MIGRATABLE); YYABORT; }
#line 2397 "y.tab.c" /* yacc.c:1646  */
    break;

  case 33:
#line 247 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CREATEHERE); YYABORT; }
#line 2403 "y.tab.c" /* yacc.c:1646  */
    break;

  case 34:
#line 248 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CREATEHOME); YYABORT; }
#line 2409 "y.tab.c" /* yacc.c:1646  */
    break;

  case 35:
#line 249 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOKEEP); YYABORT; }
#line 2415 "y.tab.c" /* yacc.c:1646  */
    break;

  case 36:
#line 250 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOTRACE); YYABORT; }
#line 2421 "y.tab.c" /* yacc.c:1646  */
    break;

  case 37:
#line 251 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(APPWORK); YYABORT; }
#line 2427 "y.tab.c" /* yacc.c:1646  */
    break;

  case 38:
#line 254 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(PACKED); YYABORT; }
#line 2433 "y.tab.c" /* yacc.c:1646  */
    break;

  case 39:
#line 255 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(VARSIZE); YYABORT; }
#line 2439 "y.tab.c" /* yacc.c:1646  */
    break;

  case 40:
#line 256 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ENTRY); YYABORT; }
#line 2445 "y.tab.c" /* yacc.c:1646  */
    break;

  case 41:
#line 257 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(FOR); YYABORT; }
#line 2451 "y.tab.c" /* yacc.c:1646  */
    break;

  case 42:
#line 258 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(FORALL); YYABORT; }
#line 2457 "y.tab.c" /* yacc.c:1646  */
    break;

  case 43:
#line 259 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(WHILE); YYABORT; }
#line 2463 "y.tab.c" /* yacc.c:1646  */
    break;

  case 44:
#line 260 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(WHEN); YYABORT; }
#line 2469 "y.tab.c" /* yacc.c:1646  */
    break;

  case 45:
#line 261 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(OVERLAP); YYABORT; }
#line 2475 "y.tab.c" /* yacc.c:1646  */
    break;

  case 46:
#line 262 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ATOMIC); YYABORT; }
#line 2481 "y.tab.c" /* yacc.c:1646  */
    break;

  case 47:
#line 263 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IF); YYABORT; }
#line 2487 "y.tab.c" /* yacc.c:1646  */
    break;

  case 48:
#line 264 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ELSE); YYABORT; }
#line 2493 "y.tab.c" /* yacc.c:1646  */
    break;

  case 49:
#line 266 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(LOCAL); YYABORT; }
#line 2499 "y.tab.c" /* yacc.c:1646  */
    break;

  case 50:
#line 268 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(USING); YYABORT; }
#line 2505 "y.tab.c" /* yacc.c:1646  */
    break;

  case 51:
#line 269 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ACCEL); YYABORT; }
#line 2511 "y.tab.c" /* yacc.c:1646  */
    break;

  case 52:
#line 272 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ACCELBLOCK); YYABORT; }
#line 2517 "y.tab.c" /* yacc.c:1646  */
    break;

  case 53:
#line 273 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MEMCRITICAL); YYABORT; }
#line 2523 "y.tab.c" /* yacc.c:1646  */
    break;

  case 54:
#line 274 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(REDUCTIONTARGET); YYABORT; }
#line 2529 "y.tab.c" /* yacc.c:1646  */
    break;

  case 55:
#line 275 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CASE); YYABORT; }
#line 2535 "y.tab.c" /* yacc.c:1646  */
    break;

  case 56:
#line 279 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 2541 "y.tab.c" /* yacc.c:1646  */
    break;

  case 57:
#line 281 "xi-grammar.y" /* yacc.c:1646  */
    {
		  char *tmp = new char[strlen((yyvsp[-3].strval))+strlen((yyvsp[0].strval))+3];
		  sprintf(tmp,"%s::%s", (yyvsp[-3].strval), (yyvsp[0].strval));
		  (yyval.strval) = tmp;
		}
#line 2551 "y.tab.c" /* yacc.c:1646  */
    break;

  case 58:
#line 289 "xi-grammar.y" /* yacc.c:1646  */
    { 
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist)); 
		}
#line 2559 "y.tab.c" /* yacc.c:1646  */
    break;

  case 59:
#line 293 "xi-grammar.y" /* yacc.c:1646  */
    {  
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist)); 
		    (yyval.module)->setMain();
		}
#line 2568 "y.tab.c" /* yacc.c:1646  */
    break;

  case 60:
#line 300 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = 0; }
#line 2574 "y.tab.c" /* yacc.c:1646  */
    break;

  case 61:
#line 302 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = (yyvsp[-2].conslist); }
#line 2580 "y.tab.c" /* yacc.c:1646  */
    break;

  case 62:
#line 306 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = 0; }
#line 2586 "y.tab.c" /* yacc.c:1646  */
    break;

  case 63:
#line 308 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = new ConstructList(lineno, (yyvsp[-1].construct), (yyvsp[0].conslist)); }
#line 2592 "y.tab.c" /* yacc.c:1646  */
    break;

  case 64:
#line 312 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new UsingScope((yyvsp[0].strval), false); }
#line 2598 "y.tab.c" /* yacc.c:1646  */
    break;

  case 65:
#line 314 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new UsingScope((yyvsp[0].strval), true); }
#line 2604 "y.tab.c" /* yacc.c:1646  */
    break;

  case 66:
#line 316 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].member)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].member); }
#line 2610 "y.tab.c" /* yacc.c:1646  */
    break;

  case 67:
#line 318 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].message)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].message); }
#line 2616 "y.tab.c" /* yacc.c:1646  */
    break;

  case 68:
#line 320 "xi-grammar.y" /* yacc.c:1646  */
    {
                  Entry *e = new Entry(lineno, 0, (yyvsp[-4].type), (yyvsp[-2].strval), (yyvsp[0].plist), 0, 0, 0, (yylsp[-6]).first_line, (yyloc).last_line);
                  int isExtern = 1;
                  e->setExtern(isExtern);
                  e->targs = (yyvsp[-1].tparlist);
                  e->label = new XStr;
                  (yyvsp[-3].ntype)->print(*e->label);
                  (yyval.construct) = e;
                }
#line 2630 "y.tab.c" /* yacc.c:1646  */
    break;

  case 69:
#line 332 "xi-grammar.y" /* yacc.c:1646  */
    { if((yyvsp[-2].conslist)) (yyvsp[-2].conslist)->recurse<int&>((yyvsp[-4].intval), &Construct::setExtern); (yyval.construct) = (yyvsp[-2].conslist); }
#line 2636 "y.tab.c" /* yacc.c:1646  */
    break;

  case 70:
#line 334 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new Scope((yyvsp[-3].strval), (yyvsp[-1].conslist)); }
#line 2642 "y.tab.c" /* yacc.c:1646  */
    break;

  case 71:
#line 336 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = (yyvsp[-1].construct); }
#line 2648 "y.tab.c" /* yacc.c:1646  */
    break;

  case 72:
#line 338 "xi-grammar.y" /* yacc.c:1646  */
    {
          ERROR("preceding construct must be semicolon terminated",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 2658 "y.tab.c" /* yacc.c:1646  */
    break;

  case 73:
#line 344 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].module)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].module); }
#line 2664 "y.tab.c" /* yacc.c:1646  */
    break;

  case 74:
#line 346 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2670 "y.tab.c" /* yacc.c:1646  */
    break;

  case 75:
#line 348 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2676 "y.tab.c" /* yacc.c:1646  */
    break;

  case 76:
#line 350 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2682 "y.tab.c" /* yacc.c:1646  */
    break;

  case 77:
#line 352 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2688 "y.tab.c" /* yacc.c:1646  */
    break;

  case 78:
#line 354 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].templat)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].templat); }
#line 2694 "y.tab.c" /* yacc.c:1646  */
    break;

  case 79:
#line 356 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = NULL; }
#line 2700 "y.tab.c" /* yacc.c:1646  */
    break;

  case 80:
#line 358 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = NULL; }
#line 2706 "y.tab.c" /* yacc.c:1646  */
    break;

  case 81:
#line 360 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = (yyvsp[0].accelBlock); }
#line 2712 "y.tab.c" /* yacc.c:1646  */
    break;

  case 82:
#line 362 "xi-grammar.y" /* yacc.c:1646  */
    {
          ERROR("invalid construct",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 2722 "y.tab.c" /* yacc.c:1646  */
    break;

  case 83:
#line 370 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamType((yyvsp[0].type)); }
#line 2728 "y.tab.c" /* yacc.c:1646  */
    break;

  case 84:
#line 372 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
#line 2734 "y.tab.c" /* yacc.c:1646  */
    break;

  case 85:
#line 374 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
#line 2740 "y.tab.c" /* yacc.c:1646  */
    break;

  case 86:
#line 378 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = new TParamList((yyvsp[0].tparam)); }
#line 2746 "y.tab.c" /* yacc.c:1646  */
    break;

  case 87:
#line 380 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = new TParamList((yyvsp[-2].tparam), (yyvsp[0].tparlist)); }
#line 2752 "y.tab.c" /* yacc.c:1646  */
    break;

  case 88:
#line 384 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = 0; }
#line 2758 "y.tab.c" /* yacc.c:1646  */
    break;

  case 89:
#line 386 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = (yyvsp[0].tparlist); }
#line 2764 "y.tab.c" /* yacc.c:1646  */
    break;

  case 90:
#line 390 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = 0; }
#line 2770 "y.tab.c" /* yacc.c:1646  */
    break;

  case 91:
#line 392 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = (yyvsp[-1].tparlist); }
#line 2776 "y.tab.c" /* yacc.c:1646  */
    break;

  case 92:
#line 396 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("int"); }
#line 2782 "y.tab.c" /* yacc.c:1646  */
    break;

  case 93:
#line 398 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long"); }
#line 2788 "y.tab.c" /* yacc.c:1646  */
    break;

  case 94:
#line 400 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("short"); }
#line 2794 "y.tab.c" /* yacc.c:1646  */
    break;

  case 95:
#line 402 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("char"); }
#line 2800 "y.tab.c" /* yacc.c:1646  */
    break;

  case 96:
#line 404 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned int"); }
#line 2806 "y.tab.c" /* yacc.c:1646  */
    break;

  case 97:
#line 406 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long"); }
#line 2812 "y.tab.c" /* yacc.c:1646  */
    break;

  case 98:
#line 408 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long"); }
#line 2818 "y.tab.c" /* yacc.c:1646  */
    break;

  case 99:
#line 410 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long long"); }
#line 2824 "y.tab.c" /* yacc.c:1646  */
    break;

  case 100:
#line 412 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned short"); }
#line 2830 "y.tab.c" /* yacc.c:1646  */
    break;

  case 101:
#line 414 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned char"); }
#line 2836 "y.tab.c" /* yacc.c:1646  */
    break;

  case 102:
#line 416 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long long"); }
#line 2842 "y.tab.c" /* yacc.c:1646  */
    break;

  case 103:
#line 418 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("float"); }
#line 2848 "y.tab.c" /* yacc.c:1646  */
    break;

  case 104:
#line 420 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("double"); }
#line 2854 "y.tab.c" /* yacc.c:1646  */
    break;

  case 105:
#line 422 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long double"); }
#line 2860 "y.tab.c" /* yacc.c:1646  */
    break;

  case 106:
#line 424 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("void"); }
#line 2866 "y.tab.c" /* yacc.c:1646  */
    break;

  case 107:
#line 427 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ntype) = new NamedType((yyvsp[-1].strval),(yyvsp[0].tparlist)); }
#line 2872 "y.tab.c" /* yacc.c:1646  */
    break;

  case 108:
#line 428 "xi-grammar.y" /* yacc.c:1646  */
    { 
                    const char* basename, *scope;
                    splitScopedName((yyvsp[-1].strval), &scope, &basename);
                    (yyval.ntype) = new NamedType(basename, (yyvsp[0].tparlist), scope);
                }
#line 2882 "y.tab.c" /* yacc.c:1646  */
    break;

  case 109:
#line 436 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 2888 "y.tab.c" /* yacc.c:1646  */
    break;

  case 110:
#line 438 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ntype); }
#line 2894 "y.tab.c" /* yacc.c:1646  */
    break;

  case 111:
#line 442 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ptype) = new PtrType((yyvsp[-1].type)); }
#line 2900 "y.tab.c" /* yacc.c:1646  */
    break;

  case 112:
#line 446 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
#line 2906 "y.tab.c" /* yacc.c:1646  */
    break;

  case 113:
#line 448 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
#line 2912 "y.tab.c" /* yacc.c:1646  */
    break;

  case 114:
#line 452 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ftype) = new FuncType((yyvsp[-7].type), (yyvsp[-4].strval), (yyvsp[-1].plist)); }
#line 2918 "y.tab.c" /* yacc.c:1646  */
    break;

  case 115:
#line 456 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 2924 "y.tab.c" /* yacc.c:1646  */
    break;

  case 116:
#line 458 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 2930 "y.tab.c" /* yacc.c:1646  */
    break;

  case 117:
#line 460 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 2936 "y.tab.c" /* yacc.c:1646  */
    break;

  case 118:
#line 462 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ftype); }
#line 2942 "y.tab.c" /* yacc.c:1646  */
    break;

  case 119:
#line 464 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[0].type)); }
#line 2948 "y.tab.c" /* yacc.c:1646  */
    break;

  case 120:
#line 466 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[-1].type)); }
#line 2954 "y.tab.c" /* yacc.c:1646  */
    break;

  case 121:
#line 470 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 2960 "y.tab.c" /* yacc.c:1646  */
    break;

  case 122:
#line 472 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 2966 "y.tab.c" /* yacc.c:1646  */
    break;

  case 123:
#line 474 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 2972 "y.tab.c" /* yacc.c:1646  */
    break;

  case 124:
#line 476 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[0].type)); }
#line 2978 "y.tab.c" /* yacc.c:1646  */
    break;

  case 125:
#line 478 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[-1].type)); }
#line 2984 "y.tab.c" /* yacc.c:1646  */
    break;

  case 126:
#line 482 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ReferenceType((yyvsp[-1].type)); }
#line 2990 "y.tab.c" /* yacc.c:1646  */
    break;

  case 127:
#line 484 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 2996 "y.tab.c" /* yacc.c:1646  */
    break;

  case 128:
#line 488 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ReferenceType((yyvsp[-1].type)); }
#line 3002 "y.tab.c" /* yacc.c:1646  */
    break;

  case 129:
#line 490 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3008 "y.tab.c" /* yacc.c:1646  */
    break;

  case 130:
#line 494 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 3014 "y.tab.c" /* yacc.c:1646  */
    break;

  case 131:
#line 498 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = (yyvsp[-1].val); }
#line 3020 "y.tab.c" /* yacc.c:1646  */
    break;

  case 132:
#line 502 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.vallist) = 0; }
#line 3026 "y.tab.c" /* yacc.c:1646  */
    break;

  case 133:
#line 504 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.vallist) = new ValueList((yyvsp[-1].val), (yyvsp[0].vallist)); }
#line 3032 "y.tab.c" /* yacc.c:1646  */
    break;

  case 134:
#line 508 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].vallist)); }
#line 3038 "y.tab.c" /* yacc.c:1646  */
    break;

  case 135:
#line 512 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[-3].type), (yyvsp[-1].strval), (yyvsp[0].vallist), 1); }
#line 3044 "y.tab.c" /* yacc.c:1646  */
    break;

  case 136:
#line 516 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0;}
#line 3050 "y.tab.c" /* yacc.c:1646  */
    break;

  case 137:
#line 518 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0;}
#line 3056 "y.tab.c" /* yacc.c:1646  */
    break;

  case 138:
#line 522 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3062 "y.tab.c" /* yacc.c:1646  */
    break;

  case 139:
#line 524 "xi-grammar.y" /* yacc.c:1646  */
    { 
		  /*
		  printf("Warning: Message attributes are being phased out.\n");
		  printf("Warning: Please remove them from interface files.\n");
		  */
		  (yyval.intval) = (yyvsp[-1].intval); 
		}
#line 3074 "y.tab.c" /* yacc.c:1646  */
    break;

  case 140:
#line 534 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[0].intval); }
#line 3080 "y.tab.c" /* yacc.c:1646  */
    break;

  case 141:
#line 536 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[-2].intval) | (yyvsp[0].intval); }
#line 3086 "y.tab.c" /* yacc.c:1646  */
    break;

  case 142:
#line 540 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3092 "y.tab.c" /* yacc.c:1646  */
    break;

  case 143:
#line 542 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3098 "y.tab.c" /* yacc.c:1646  */
    break;

  case 144:
#line 546 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = 0; }
#line 3104 "y.tab.c" /* yacc.c:1646  */
    break;

  case 145:
#line 548 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-1].cattr); }
#line 3110 "y.tab.c" /* yacc.c:1646  */
    break;

  case 146:
#line 552 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[0].cattr); }
#line 3116 "y.tab.c" /* yacc.c:1646  */
    break;

  case 147:
#line 554 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
#line 3122 "y.tab.c" /* yacc.c:1646  */
    break;

  case 148:
#line 558 "xi-grammar.y" /* yacc.c:1646  */
    { python_doc = NULL; (yyval.intval) = 0; }
#line 3128 "y.tab.c" /* yacc.c:1646  */
    break;

  case 149:
#line 560 "xi-grammar.y" /* yacc.c:1646  */
    { python_doc = (yyvsp[0].strval); (yyval.intval) = 0; }
#line 3134 "y.tab.c" /* yacc.c:1646  */
    break;

  case 150:
#line 564 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CPYTHON; }
#line 3140 "y.tab.c" /* yacc.c:1646  */
    break;

  case 151:
#line 568 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = 0; }
#line 3146 "y.tab.c" /* yacc.c:1646  */
    break;

  case 152:
#line 570 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-1].cattr); }
#line 3152 "y.tab.c" /* yacc.c:1646  */
    break;

  case 153:
#line 574 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[0].cattr); }
#line 3158 "y.tab.c" /* yacc.c:1646  */
    break;

  case 154:
#line 576 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
#line 3164 "y.tab.c" /* yacc.c:1646  */
    break;

  case 155:
#line 580 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CMIGRATABLE; }
#line 3170 "y.tab.c" /* yacc.c:1646  */
    break;

  case 156:
#line 582 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CPYTHON; }
#line 3176 "y.tab.c" /* yacc.c:1646  */
    break;

  case 157:
#line 586 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3182 "y.tab.c" /* yacc.c:1646  */
    break;

  case 158:
#line 588 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 3188 "y.tab.c" /* yacc.c:1646  */
    break;

  case 159:
#line 591 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3194 "y.tab.c" /* yacc.c:1646  */
    break;

  case 160:
#line 593 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 3200 "y.tab.c" /* yacc.c:1646  */
    break;

  case 161:
#line 596 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mv) = new MsgVar((yyvsp[-3].type), (yyvsp[-2].strval), (yyvsp[-4].intval), (yyvsp[-1].intval)); }
#line 3206 "y.tab.c" /* yacc.c:1646  */
    break;

  case 162:
#line 600 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mvlist) = new MsgVarList((yyvsp[0].mv)); }
#line 3212 "y.tab.c" /* yacc.c:1646  */
    break;

  case 163:
#line 602 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mvlist) = new MsgVarList((yyvsp[-1].mv), (yyvsp[0].mvlist)); }
#line 3218 "y.tab.c" /* yacc.c:1646  */
    break;

  case 164:
#line 606 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, (yyvsp[0].ntype)); }
#line 3224 "y.tab.c" /* yacc.c:1646  */
    break;

  case 165:
#line 608 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, (yyvsp[-3].ntype), (yyvsp[-1].mvlist)); }
#line 3230 "y.tab.c" /* yacc.c:1646  */
    break;

  case 166:
#line 612 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = 0; }
#line 3236 "y.tab.c" /* yacc.c:1646  */
    break;

  case 167:
#line 614 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = (yyvsp[0].typelist); }
#line 3242 "y.tab.c" /* yacc.c:1646  */
    break;

  case 168:
#line 618 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = new TypeList((yyvsp[0].ntype)); }
#line 3248 "y.tab.c" /* yacc.c:1646  */
    break;

  case 169:
#line 620 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = new TypeList((yyvsp[-2].ntype), (yyvsp[0].typelist)); }
#line 3254 "y.tab.c" /* yacc.c:1646  */
    break;

  case 170:
#line 624 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr)|Chare::CCHARE, (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3260 "y.tab.c" /* yacc.c:1646  */
    break;

  case 171:
#line 626 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3266 "y.tab.c" /* yacc.c:1646  */
    break;

  case 172:
#line 630 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3272 "y.tab.c" /* yacc.c:1646  */
    break;

  case 173:
#line 634 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new NodeGroup(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3278 "y.tab.c" /* yacc.c:1646  */
    break;

  case 174:
#line 638 "xi-grammar.y" /* yacc.c:1646  */
    {/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",(yyvsp[-2].strval));
			(yyval.ntype) = new NamedType(buf); 
		}
#line 3288 "y.tab.c" /* yacc.c:1646  */
    break;

  case 175:
#line 644 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ntype) = new NamedType((yyvsp[-1].strval)); }
#line 3294 "y.tab.c" /* yacc.c:1646  */
    break;

  case 176:
#line 648 "xi-grammar.y" /* yacc.c:1646  */
    {  (yyval.chare) = new Array(lineno, (yyvsp[-4].cattr), (yyvsp[-3].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3300 "y.tab.c" /* yacc.c:1646  */
    break;

  case 177:
#line 650 "xi-grammar.y" /* yacc.c:1646  */
    {  (yyval.chare) = new Array(lineno, (yyvsp[-3].cattr), (yyvsp[-4].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3306 "y.tab.c" /* yacc.c:1646  */
    break;

  case 178:
#line 654 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr)|Chare::CCHARE, new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist));}
#line 3312 "y.tab.c" /* yacc.c:1646  */
    break;

  case 179:
#line 656 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3318 "y.tab.c" /* yacc.c:1646  */
    break;

  case 180:
#line 660 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3324 "y.tab.c" /* yacc.c:1646  */
    break;

  case 181:
#line 664 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new NodeGroup( lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3330 "y.tab.c" /* yacc.c:1646  */
    break;

  case 182:
#line 668 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Array( lineno, 0, (yyvsp[-3].ntype), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3336 "y.tab.c" /* yacc.c:1646  */
    break;

  case 183:
#line 672 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-1].strval))); }
#line 3342 "y.tab.c" /* yacc.c:1646  */
    break;

  case 184:
#line 674 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-4].strval)), (yyvsp[-2].mvlist)); }
#line 3348 "y.tab.c" /* yacc.c:1646  */
    break;

  case 185:
#line 678 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = 0; }
#line 3354 "y.tab.c" /* yacc.c:1646  */
    break;

  case 186:
#line 680 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3360 "y.tab.c" /* yacc.c:1646  */
    break;

  case 187:
#line 684 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = 0; }
#line 3366 "y.tab.c" /* yacc.c:1646  */
    break;

  case 188:
#line 686 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3372 "y.tab.c" /* yacc.c:1646  */
    break;

  case 189:
#line 688 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3378 "y.tab.c" /* yacc.c:1646  */
    break;

  case 190:
#line 692 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TType(new NamedType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3384 "y.tab.c" /* yacc.c:1646  */
    break;

  case 191:
#line 694 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TFunc((yyvsp[-1].ftype), (yyvsp[0].strval)); }
#line 3390 "y.tab.c" /* yacc.c:1646  */
    break;

  case 192:
#line 696 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TName((yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].strval)); }
#line 3396 "y.tab.c" /* yacc.c:1646  */
    break;

  case 193:
#line 700 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = new TVarList((yyvsp[0].tvar)); }
#line 3402 "y.tab.c" /* yacc.c:1646  */
    break;

  case 194:
#line 702 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = new TVarList((yyvsp[-2].tvar), (yyvsp[0].tvarlist)); }
#line 3408 "y.tab.c" /* yacc.c:1646  */
    break;

  case 195:
#line 706 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = (yyvsp[-1].tvarlist); }
#line 3414 "y.tab.c" /* yacc.c:1646  */
    break;

  case 196:
#line 710 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3420 "y.tab.c" /* yacc.c:1646  */
    break;

  case 197:
#line 712 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3426 "y.tab.c" /* yacc.c:1646  */
    break;

  case 198:
#line 714 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3432 "y.tab.c" /* yacc.c:1646  */
    break;

  case 199:
#line 716 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3438 "y.tab.c" /* yacc.c:1646  */
    break;

  case 200:
#line 718 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].message)); (yyvsp[0].message)->setTemplate((yyval.templat)); }
#line 3444 "y.tab.c" /* yacc.c:1646  */
    break;

  case 201:
#line 722 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = 0; }
#line 3450 "y.tab.c" /* yacc.c:1646  */
    break;

  case 202:
#line 724 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = (yyvsp[-2].mbrlist); }
#line 3456 "y.tab.c" /* yacc.c:1646  */
    break;

  case 203:
#line 728 "xi-grammar.y" /* yacc.c:1646  */
    { 
                  if (!connectEntries.empty()) {
                    (yyval.mbrlist) = new AstChildren<Member>(connectEntries);
		  } else {
		    (yyval.mbrlist) = 0; 
                  }
		}
#line 3468 "y.tab.c" /* yacc.c:1646  */
    break;

  case 204:
#line 736 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = new AstChildren<Member>(-1, (yyvsp[-1].member), (yyvsp[0].mbrlist)); }
#line 3474 "y.tab.c" /* yacc.c:1646  */
    break;

  case 205:
#line 740 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].readonly); }
#line 3480 "y.tab.c" /* yacc.c:1646  */
    break;

  case 206:
#line 742 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].readonly); }
#line 3486 "y.tab.c" /* yacc.c:1646  */
    break;

  case 208:
#line 745 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].member); }
#line 3492 "y.tab.c" /* yacc.c:1646  */
    break;

  case 209:
#line 747 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].pupable); }
#line 3498 "y.tab.c" /* yacc.c:1646  */
    break;

  case 210:
#line 749 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].includeFile); }
#line 3504 "y.tab.c" /* yacc.c:1646  */
    break;

  case 211:
#line 751 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new ClassDeclaration(lineno,(yyvsp[0].strval)); }
#line 3510 "y.tab.c" /* yacc.c:1646  */
    break;

  case 212:
#line 755 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1); }
#line 3516 "y.tab.c" /* yacc.c:1646  */
    break;

  case 213:
#line 757 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1); }
#line 3522 "y.tab.c" /* yacc.c:1646  */
    break;

  case 214:
#line 759 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[-6].strval)) + '<' +
					    ((yyvsp[-4].tparlist))->to_string() + '>').c_str()),
				    1);
		}
#line 3532 "y.tab.c" /* yacc.c:1646  */
    break;

  case 215:
#line 765 "xi-grammar.y" /* yacc.c:1646  */
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[-2]).first_column, (yylsp[-2]).last_column, (yylsp[-2]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1);
		}
#line 3542 "y.tab.c" /* yacc.c:1646  */
    break;

  case 216:
#line 771 "xi-grammar.y" /* yacc.c:1646  */
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[-5]).first_column, (yylsp[-5]).last_column, (yylsp[-5]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1);
		}
#line 3552 "y.tab.c" /* yacc.c:1646  */
    break;

  case 217:
#line 780 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 0); }
#line 3558 "y.tab.c" /* yacc.c:1646  */
    break;

  case 218:
#line 782 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 0); }
#line 3564 "y.tab.c" /* yacc.c:1646  */
    break;

  case 219:
#line 784 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[-6].strval)) + '<' +
					    ((yyvsp[-4].tparlist))->to_string() + '>').c_str()),
				    0);
		}
#line 3574 "y.tab.c" /* yacc.c:1646  */
    break;

  case 220:
#line 790 "xi-grammar.y" /* yacc.c:1646  */
    {
                  InitCall* rtn = new InitCall(lineno, (yyvsp[-3].strval), 0);
                  rtn->setAccel();
                  (yyval.member) = rtn;
		}
#line 3584 "y.tab.c" /* yacc.c:1646  */
    break;

  case 221:
#line 798 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[0].ntype),0); }
#line 3590 "y.tab.c" /* yacc.c:1646  */
    break;

  case 222:
#line 800 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[-2].ntype),(yyvsp[0].pupable)); }
#line 3596 "y.tab.c" /* yacc.c:1646  */
    break;

  case 223:
#line 803 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.includeFile) = new IncludeFile(lineno,(yyvsp[0].strval)); }
#line 3602 "y.tab.c" /* yacc.c:1646  */
    break;

  case 224:
#line 807 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].member); }
#line 3608 "y.tab.c" /* yacc.c:1646  */
    break;

  case 225:
#line 811 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].entry); }
#line 3614 "y.tab.c" /* yacc.c:1646  */
    break;

  case 226:
#line 813 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyvsp[0].entry)->tspec = (yyvsp[-1].tvarlist);
                  (yyval.member) = (yyvsp[0].entry);
                }
#line 3623 "y.tab.c" /* yacc.c:1646  */
    break;

  case 227:
#line 818 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[-1].member); }
#line 3629 "y.tab.c" /* yacc.c:1646  */
    break;

  case 228:
#line 820 "xi-grammar.y" /* yacc.c:1646  */
    {
          ERROR("invalid SDAG member",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 3639 "y.tab.c" /* yacc.c:1646  */
    break;

  case 229:
#line 828 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3645 "y.tab.c" /* yacc.c:1646  */
    break;

  case 230:
#line 830 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3651 "y.tab.c" /* yacc.c:1646  */
    break;

  case 231:
#line 832 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3657 "y.tab.c" /* yacc.c:1646  */
    break;

  case 232:
#line 834 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3663 "y.tab.c" /* yacc.c:1646  */
    break;

  case 233:
#line 836 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3669 "y.tab.c" /* yacc.c:1646  */
    break;

  case 234:
#line 838 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3675 "y.tab.c" /* yacc.c:1646  */
    break;

  case 235:
#line 840 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3681 "y.tab.c" /* yacc.c:1646  */
    break;

  case 236:
#line 842 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3687 "y.tab.c" /* yacc.c:1646  */
    break;

  case 237:
#line 844 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3693 "y.tab.c" /* yacc.c:1646  */
    break;

  case 238:
#line 846 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3699 "y.tab.c" /* yacc.c:1646  */
    break;

  case 239:
#line 848 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3705 "y.tab.c" /* yacc.c:1646  */
    break;

  case 240:
#line 851 "xi-grammar.y" /* yacc.c:1646  */
    { 
                  (yyval.entry) = new Entry(lineno, (yyvsp[-5].intval), (yyvsp[-4].type), (yyvsp[-3].strval), (yyvsp[-2].plist), (yyvsp[-1].val), (yyvsp[0].sentry), (const char *) NULL, (yylsp[-6]).first_line, (yyloc).last_line);
		  if ((yyvsp[0].sentry) != 0) { 
		    (yyvsp[0].sentry)->con1 = new SdagConstruct(SIDENT, (yyvsp[-3].strval));
                    (yyvsp[0].sentry)->entry = (yyval.entry);
                    (yyvsp[0].sentry)->con1->entry = (yyval.entry);
                    (yyvsp[0].sentry)->param = new ParamList((yyvsp[-2].plist));
                  }
		}
#line 3719 "y.tab.c" /* yacc.c:1646  */
    break;

  case 241:
#line 861 "xi-grammar.y" /* yacc.c:1646  */
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
#line 3740 "y.tab.c" /* yacc.c:1646  */
    break;

  case 242:
#line 878 "xi-grammar.y" /* yacc.c:1646  */
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
#line 3758 "y.tab.c" /* yacc.c:1646  */
    break;

  case 243:
#line 894 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.accelBlock) = new AccelBlock(lineno, new XStr((yyvsp[-2].strval))); }
#line 3764 "y.tab.c" /* yacc.c:1646  */
    break;

  case 244:
#line 896 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.accelBlock) = new AccelBlock(lineno, NULL); }
#line 3770 "y.tab.c" /* yacc.c:1646  */
    break;

  case 245:
#line 900 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3776 "y.tab.c" /* yacc.c:1646  */
    break;

  case 246:
#line 904 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3782 "y.tab.c" /* yacc.c:1646  */
    break;

  case 247:
#line 906 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[-1].intval); }
#line 3788 "y.tab.c" /* yacc.c:1646  */
    break;

  case 248:
#line 908 "xi-grammar.y" /* yacc.c:1646  */
    { ERROR("invalid entry method attribute list",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
#line 3797 "y.tab.c" /* yacc.c:1646  */
    break;

  case 249:
#line 915 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[0].intval); }
#line 3803 "y.tab.c" /* yacc.c:1646  */
    break;

  case 250:
#line 917 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[-2].intval) | (yyvsp[0].intval); }
#line 3809 "y.tab.c" /* yacc.c:1646  */
    break;

  case 251:
#line 921 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = STHREADED; }
#line 3815 "y.tab.c" /* yacc.c:1646  */
    break;

  case 252:
#line 923 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SSYNC; }
#line 3821 "y.tab.c" /* yacc.c:1646  */
    break;

  case 253:
#line 925 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SIGET; }
#line 3827 "y.tab.c" /* yacc.c:1646  */
    break;

  case 254:
#line 927 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SLOCKED; }
#line 3833 "y.tab.c" /* yacc.c:1646  */
    break;

  case 255:
#line 929 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SCREATEHERE; }
#line 3839 "y.tab.c" /* yacc.c:1646  */
    break;

  case 256:
#line 931 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SCREATEHOME; }
#line 3845 "y.tab.c" /* yacc.c:1646  */
    break;

  case 257:
#line 933 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SNOKEEP; }
#line 3851 "y.tab.c" /* yacc.c:1646  */
    break;

  case 258:
#line 935 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SNOTRACE; }
#line 3857 "y.tab.c" /* yacc.c:1646  */
    break;

  case 259:
#line 937 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SAPPWORK; }
#line 3863 "y.tab.c" /* yacc.c:1646  */
    break;

  case 260:
#line 939 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SIMMEDIATE; }
#line 3869 "y.tab.c" /* yacc.c:1646  */
    break;

  case 261:
#line 941 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SSKIPSCHED; }
#line 3875 "y.tab.c" /* yacc.c:1646  */
    break;

  case 262:
#line 943 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SINLINE; }
#line 3881 "y.tab.c" /* yacc.c:1646  */
    break;

  case 263:
#line 945 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SLOCAL; }
#line 3887 "y.tab.c" /* yacc.c:1646  */
    break;

  case 264:
#line 947 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SPYTHON; }
#line 3893 "y.tab.c" /* yacc.c:1646  */
    break;

  case 265:
#line 949 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SMEM; }
#line 3899 "y.tab.c" /* yacc.c:1646  */
    break;

  case 266:
#line 951 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SREDUCE; }
#line 3905 "y.tab.c" /* yacc.c:1646  */
    break;

  case 267:
#line 953 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("invalid entry method attribute",
		        (yylsp[0]).first_column, (yylsp[0]).last_column);
		  yyclearin;
		  yyerrok;
		}
#line 3916 "y.tab.c" /* yacc.c:1646  */
    break;

  case 268:
#line 962 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 3922 "y.tab.c" /* yacc.c:1646  */
    break;

  case 269:
#line 964 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 3928 "y.tab.c" /* yacc.c:1646  */
    break;

  case 270:
#line 966 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 3934 "y.tab.c" /* yacc.c:1646  */
    break;

  case 271:
#line 970 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = ""; }
#line 3940 "y.tab.c" /* yacc.c:1646  */
    break;

  case 272:
#line 972 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3946 "y.tab.c" /* yacc.c:1646  */
    break;

  case 273:
#line 974 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s, %s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 3956 "y.tab.c" /* yacc.c:1646  */
    break;

  case 274:
#line 982 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = ""; }
#line 3962 "y.tab.c" /* yacc.c:1646  */
    break;

  case 275:
#line 984 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3968 "y.tab.c" /* yacc.c:1646  */
    break;

  case 276:
#line 986 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s[%s]%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 3978 "y.tab.c" /* yacc.c:1646  */
    break;

  case 277:
#line 992 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s{%s}%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 3988 "y.tab.c" /* yacc.c:1646  */
    break;

  case 278:
#line 998 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s(%s)%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 3998 "y.tab.c" /* yacc.c:1646  */
    break;

  case 279:
#line 1004 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"(%s)%s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 4008 "y.tab.c" /* yacc.c:1646  */
    break;

  case 280:
#line 1012 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval));
		}
#line 4017 "y.tab.c" /* yacc.c:1646  */
    break;

  case 281:
#line 1019 "xi-grammar.y" /* yacc.c:1646  */
    { 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			(yyval.intval) = 0;
		}
#line 4027 "y.tab.c" /* yacc.c:1646  */
    break;

  case 282:
#line 1027 "xi-grammar.y" /* yacc.c:1646  */
    { 
			in_braces=0;
			(yyval.intval) = 0;
		}
#line 4036 "y.tab.c" /* yacc.c:1646  */
    break;

  case 283:
#line 1034 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[0].type));}
#line 4042 "y.tab.c" /* yacc.c:1646  */
    break;

  case 284:
#line 1036 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval)); (yyval.pname)->setConditional((yyvsp[0].intval)); }
#line 4048 "y.tab.c" /* yacc.c:1646  */
    break;

  case 285:
#line 1038 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[-3].type),(yyvsp[-2].strval),0,(yyvsp[0].val));}
#line 4054 "y.tab.c" /* yacc.c:1646  */
    break;

  case 286:
#line 1040 "xi-grammar.y" /* yacc.c:1646  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
		}
#line 4063 "y.tab.c" /* yacc.c:1646  */
    break;

  case 287:
#line 1046 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READONLY; }
#line 4069 "y.tab.c" /* yacc.c:1646  */
    break;

  case 288:
#line 1047 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READWRITE; }
#line 4075 "y.tab.c" /* yacc.c:1646  */
    break;

  case 289:
#line 1048 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_WRITEONLY; }
#line 4081 "y.tab.c" /* yacc.c:1646  */
    break;

  case 290:
#line 1051 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr((yyvsp[0].strval)); }
#line 4087 "y.tab.c" /* yacc.c:1646  */
    break;

  case 291:
#line 1052 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "->" << (yyvsp[0].strval); }
#line 4093 "y.tab.c" /* yacc.c:1646  */
    break;

  case 292:
#line 1053 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[-2].xstrptr)) << "." << (yyvsp[0].strval); }
#line 4099 "y.tab.c" /* yacc.c:1646  */
    break;

  case 293:
#line 1055 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << *((yyvsp[-1].xstrptr)) << "]";
                  delete (yyvsp[-3].xstrptr);
                  delete (yyvsp[-1].xstrptr);
                }
#line 4110 "y.tab.c" /* yacc.c:1646  */
    break;

  case 294:
#line 1062 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << (yyvsp[-1].strval) << "]";
                  delete (yyvsp[-3].xstrptr);
                }
#line 4120 "y.tab.c" /* yacc.c:1646  */
    break;

  case 295:
#line 1068 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "(" << *((yyvsp[-1].xstrptr)) << ")";
                  delete (yyvsp[-3].xstrptr);
                  delete (yyvsp[-1].xstrptr);
                }
#line 4131 "y.tab.c" /* yacc.c:1646  */
    break;

  case 296:
#line 1077 "xi-grammar.y" /* yacc.c:1646  */
    {
                  in_bracket = 0;
                  (yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName(), (yyvsp[-1].strval));
                }
#line 4140 "y.tab.c" /* yacc.c:1646  */
    break;

  case 297:
#line 1084 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[-6].intval));
                }
#line 4150 "y.tab.c" /* yacc.c:1646  */
    break;

  case 298:
#line 1090 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType(Parameter::ACCEL_BUFFER_TYPE_READWRITE);
		}
#line 4160 "y.tab.c" /* yacc.c:1646  */
    break;

  case 299:
#line 1096 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.pname) = (yyvsp[-3].pname);
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[-5].intval));
		}
#line 4170 "y.tab.c" /* yacc.c:1646  */
    break;

  case 300:
#line 1104 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 4176 "y.tab.c" /* yacc.c:1646  */
    break;

  case 301:
#line 1106 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 4182 "y.tab.c" /* yacc.c:1646  */
    break;

  case 302:
#line 1110 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 4188 "y.tab.c" /* yacc.c:1646  */
    break;

  case 303:
#line 1112 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 4194 "y.tab.c" /* yacc.c:1646  */
    break;

  case 304:
#line 1116 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = (yyvsp[-1].plist); }
#line 4200 "y.tab.c" /* yacc.c:1646  */
    break;

  case 305:
#line 1118 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList(new Parameter(0, new BuiltinType("void"))); }
#line 4206 "y.tab.c" /* yacc.c:1646  */
    break;

  case 306:
#line 1122 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = (yyvsp[-1].plist); }
#line 4212 "y.tab.c" /* yacc.c:1646  */
    break;

  case 307:
#line 1124 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = 0; }
#line 4218 "y.tab.c" /* yacc.c:1646  */
    break;

  case 308:
#line 1128 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = 0; }
#line 4224 "y.tab.c" /* yacc.c:1646  */
    break;

  case 309:
#line 1130 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4230 "y.tab.c" /* yacc.c:1646  */
    break;

  case 310:
#line 1134 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = 0; }
#line 4236 "y.tab.c" /* yacc.c:1646  */
    break;

  case 311:
#line 1136 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[0].sc)); }
#line 4242 "y.tab.c" /* yacc.c:1646  */
    break;

  case 312:
#line 1138 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[-2].slist)); }
#line 4248 "y.tab.c" /* yacc.c:1646  */
    break;

  case 313:
#line 1142 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.slist) = new SListConstruct((yyvsp[0].sc)); }
#line 4254 "y.tab.c" /* yacc.c:1646  */
    break;

  case 314:
#line 1144 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.slist) = new SListConstruct((yyvsp[-1].sc), (yyvsp[0].slist));  }
#line 4260 "y.tab.c" /* yacc.c:1646  */
    break;

  case 315:
#line 1148 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.olist) = new OListConstruct((yyvsp[0].sc)); }
#line 4266 "y.tab.c" /* yacc.c:1646  */
    break;

  case 316:
#line 1150 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.olist) = new OListConstruct((yyvsp[-1].sc), (yyvsp[0].slist)); }
#line 4272 "y.tab.c" /* yacc.c:1646  */
    break;

  case 317:
#line 1154 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.clist) = new CaseListConstruct((yyvsp[0].when)); }
#line 4278 "y.tab.c" /* yacc.c:1646  */
    break;

  case 318:
#line 1156 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.clist) = new CaseListConstruct((yyvsp[-1].when), (yyvsp[0].clist)); }
#line 4284 "y.tab.c" /* yacc.c:1646  */
    break;

  case 319:
#line 1158 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("case blocks can only contain when clauses",
		        (yylsp[0]).first_column, (yylsp[0]).last_column);
		  (yyval.clist) = 0;
		}
#line 4294 "y.tab.c" /* yacc.c:1646  */
    break;

  case 320:
#line 1166 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4300 "y.tab.c" /* yacc.c:1646  */
    break;

  case 321:
#line 1168 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = 0; }
#line 4306 "y.tab.c" /* yacc.c:1646  */
    break;

  case 322:
#line 1172 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-2].entrylist), 0); }
#line 4312 "y.tab.c" /* yacc.c:1646  */
    break;

  case 323:
#line 1174 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-1].entrylist), (yyvsp[0].sc)); }
#line 4318 "y.tab.c" /* yacc.c:1646  */
    break;

  case 324:
#line 1176 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-3].entrylist), (yyvsp[-1].slist)); }
#line 4324 "y.tab.c" /* yacc.c:1646  */
    break;

  case 325:
#line 1180 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4330 "y.tab.c" /* yacc.c:1646  */
    break;

  case 326:
#line 1182 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4336 "y.tab.c" /* yacc.c:1646  */
    break;

  case 327:
#line 1184 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4342 "y.tab.c" /* yacc.c:1646  */
    break;

  case 328:
#line 1186 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4348 "y.tab.c" /* yacc.c:1646  */
    break;

  case 329:
#line 1188 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4354 "y.tab.c" /* yacc.c:1646  */
    break;

  case 330:
#line 1190 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4360 "y.tab.c" /* yacc.c:1646  */
    break;

  case 331:
#line 1192 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4366 "y.tab.c" /* yacc.c:1646  */
    break;

  case 332:
#line 1194 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4372 "y.tab.c" /* yacc.c:1646  */
    break;

  case 333:
#line 1196 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4378 "y.tab.c" /* yacc.c:1646  */
    break;

  case 334:
#line 1198 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4384 "y.tab.c" /* yacc.c:1646  */
    break;

  case 335:
#line 1200 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4390 "y.tab.c" /* yacc.c:1646  */
    break;

  case 336:
#line 1202 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4396 "y.tab.c" /* yacc.c:1646  */
    break;

  case 337:
#line 1206 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new AtomicConstruct((yyvsp[-1].strval), (yyvsp[-3].strval), (yylsp[-2]).first_line); }
#line 4402 "y.tab.c" /* yacc.c:1646  */
    break;

  case 338:
#line 1208 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new OverlapConstruct((yyvsp[-1].olist)); }
#line 4408 "y.tab.c" /* yacc.c:1646  */
    break;

  case 339:
#line 1210 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = (yyvsp[0].when); }
#line 4414 "y.tab.c" /* yacc.c:1646  */
    break;

  case 340:
#line 1212 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new CaseConstruct((yyvsp[-1].clist)); }
#line 4420 "y.tab.c" /* yacc.c:1646  */
    break;

  case 341:
#line 1214 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForConstruct((yyvsp[-8].intexpr), (yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4426 "y.tab.c" /* yacc.c:1646  */
    break;

  case 342:
#line 1216 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForConstruct((yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4432 "y.tab.c" /* yacc.c:1646  */
    break;

  case 343:
#line 1218 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-9].strval)), (yyvsp[-6].intexpr),
		             (yyvsp[-4].intexpr), (yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4439 "y.tab.c" /* yacc.c:1646  */
    break;

  case 344:
#line 1221 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-11].strval)), (yyvsp[-8].intexpr),
		             (yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4446 "y.tab.c" /* yacc.c:1646  */
    break;

  case 345:
#line 1224 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new IfConstruct((yyvsp[-3].intexpr), (yyvsp[-1].sc), (yyvsp[0].sc)); }
#line 4452 "y.tab.c" /* yacc.c:1646  */
    break;

  case 346:
#line 1226 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new IfConstruct((yyvsp[-5].intexpr), (yyvsp[-2].slist), (yyvsp[0].sc)); }
#line 4458 "y.tab.c" /* yacc.c:1646  */
    break;

  case 347:
#line 1228 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new WhileConstruct((yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4464 "y.tab.c" /* yacc.c:1646  */
    break;

  case 348:
#line 1230 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new WhileConstruct((yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4470 "y.tab.c" /* yacc.c:1646  */
    break;

  case 349:
#line 1232 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new AtomicConstruct((yyvsp[-1].strval), NULL, (yyloc).first_line); }
#line 4476 "y.tab.c" /* yacc.c:1646  */
    break;

  case 350:
#line 1234 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("unknown SDAG construct or malformed entry method declaration.\n"
		        "You may have forgotten to terminate a previous entry method declaration with a"
		        " semicolon or forgotten to mark a block of sequential SDAG code as 'atomic'",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
#line 4488 "y.tab.c" /* yacc.c:1646  */
    break;

  case 351:
#line 1244 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = 0; }
#line 4494 "y.tab.c" /* yacc.c:1646  */
    break;

  case 352:
#line 1246 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ElseConstruct((yyvsp[0].sc)); }
#line 4500 "y.tab.c" /* yacc.c:1646  */
    break;

  case 353:
#line 1248 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ElseConstruct((yyvsp[-1].slist)); }
#line 4506 "y.tab.c" /* yacc.c:1646  */
    break;

  case 354:
#line 1252 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intexpr) = new IntExprConstruct((yyvsp[0].strval)); }
#line 4512 "y.tab.c" /* yacc.c:1646  */
    break;

  case 355:
#line 1256 "xi-grammar.y" /* yacc.c:1646  */
    { in_int_expr = 0; (yyval.intval) = 0; }
#line 4518 "y.tab.c" /* yacc.c:1646  */
    break;

  case 356:
#line 1260 "xi-grammar.y" /* yacc.c:1646  */
    { in_int_expr = 1; (yyval.intval) = 0; }
#line 4524 "y.tab.c" /* yacc.c:1646  */
    break;

  case 357:
#line 1264 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[-1].strval), (yyvsp[0].plist), 0, 0, 0, (yyloc).first_line, (yyloc).last_line);
		}
#line 4532 "y.tab.c" /* yacc.c:1646  */
    break;

  case 358:
#line 1268 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[-4].strval), (yyvsp[0].plist), 0, 0, (yyvsp[-2].strval), (yyloc).first_line, (yyloc).last_line);
		}
#line 4540 "y.tab.c" /* yacc.c:1646  */
    break;

  case 359:
#line 1274 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.entrylist) = new EntryList((yyvsp[0].entry)); }
#line 4546 "y.tab.c" /* yacc.c:1646  */
    break;

  case 360:
#line 1276 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.entrylist) = new EntryList((yyvsp[-2].entry),(yyvsp[0].entrylist)); }
#line 4552 "y.tab.c" /* yacc.c:1646  */
    break;

  case 361:
#line 1280 "xi-grammar.y" /* yacc.c:1646  */
    { in_bracket=1; }
#line 4558 "y.tab.c" /* yacc.c:1646  */
    break;

  case 362:
#line 1283 "xi-grammar.y" /* yacc.c:1646  */
    { in_bracket=0; }
#line 4564 "y.tab.c" /* yacc.c:1646  */
    break;

  case 363:
#line 1287 "xi-grammar.y" /* yacc.c:1646  */
    { if (!macroDefined((yyvsp[0].strval), 1)) in_comment = 1; }
#line 4570 "y.tab.c" /* yacc.c:1646  */
    break;

  case 364:
#line 1291 "xi-grammar.y" /* yacc.c:1646  */
    { if (!macroDefined((yyvsp[0].strval), 0)) in_comment = 1; }
#line 4576 "y.tab.c" /* yacc.c:1646  */
    break;


#line 4580 "y.tab.c" /* yacc.c:1646  */
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
#line 1294 "xi-grammar.y" /* yacc.c:1906  */


void yyerror(const char *msg) { }
