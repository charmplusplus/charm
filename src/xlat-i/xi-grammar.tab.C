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
#include <vector>
#include <string>
#include <string.h>
#include "xi-symbol.h"
#include "sdag/constructs/Constructs.h"
#include "EToken.h"

// Has to be a macro since YYABORT can only be used within rule actions.
#define ERROR(...) \
  if (xi::num_errors++ == xi::MAX_NUM_ERRORS) { \
    YYABORT;                                    \
  } else {                                      \
    pretty_msg("error", __VA_ARGS__);           \
  }

#define WARNING(...) \
  if (enable_warnings) {                \
    pretty_msg("warning", __VA_ARGS__); \
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

extern std::vector<std::string> inputBuffer;
const int MAX_NUM_ERRORS = 10;
int num_errors = 0;

bool enable_warnings = true;

extern int macroDefined(const char *str, int istrue);
extern const char *python_doc;
extern char *fname;
void splitScopedName(const char* name, const char** scope, const char** basename);
void ReservedWord(int token);
}

#line 114 "xi-grammar.tab.C" /* yacc.c:339  */

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
   by #include "xi-grammar.tab.h".  */
#ifndef YY_YY_XI_GRAMMAR_TAB_H_INCLUDED
# define YY_YY_XI_GRAMMAR_TAB_H_INCLUDED
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

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED

union YYSTYPE
{
#line 52 "xi-grammar.y" /* yacc.c:355  */

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
  Chare::attrib_t cattr;
  SdagConstruct *sc;
  IntExprConstruct *intexpr;
  WhenConstruct *when;
  SListConstruct *slist;
  CaseListConstruct *clist;
  OListConstruct *olist;
  SdagEntryConstruct *sentry;
  XStr* xstrptr;
  AccelBlock* accelBlock;

#line 271 "xi-grammar.tab.C" /* yacc.c:355  */
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

#endif /* !YY_YY_XI_GRAMMAR_TAB_H_INCLUDED  */

/* Copy the second part of user declarations.  */

#line 302 "xi-grammar.tab.C" /* yacc.c:358  */

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
#define YYLAST   1470

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  89
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  116
/* YYNRULES -- Number of rules.  */
#define YYNRULES  364
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  707

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
       0,   193,   193,   198,   201,   206,   207,   212,   213,   218,
     220,   221,   222,   224,   225,   226,   228,   229,   230,   231,
     232,   236,   237,   238,   239,   240,   241,   242,   243,   244,
     245,   246,   247,   248,   249,   250,   251,   252,   255,   256,
     257,   258,   259,   260,   261,   262,   263,   264,   265,   267,
     269,   270,   273,   274,   275,   276,   279,   281,   289,   293,
     300,   302,   307,   308,   312,   314,   316,   318,   320,   332,
     334,   336,   338,   344,   346,   348,   350,   352,   354,   356,
     358,   360,   362,   370,   372,   374,   378,   380,   385,   386,
     391,   392,   396,   398,   400,   402,   404,   406,   408,   410,
     412,   414,   416,   418,   420,   422,   424,   428,   429,   436,
     438,   442,   446,   448,   452,   456,   458,   460,   462,   464,
     466,   470,   472,   474,   476,   478,   482,   484,   488,   490,
     494,   498,   503,   504,   508,   512,   517,   518,   523,   524,
     534,   536,   540,   542,   547,   548,   552,   554,   559,   560,
     564,   569,   570,   574,   576,   580,   582,   587,   588,   592,
     593,   596,   600,   602,   606,   608,   613,   614,   618,   620,
     624,   626,   630,   634,   638,   644,   648,   650,   654,   656,
     660,   664,   668,   672,   674,   679,   680,   685,   686,   688,
     692,   694,   696,   700,   702,   706,   710,   712,   714,   716,
     718,   722,   724,   729,   736,   740,   742,   744,   745,   747,
     749,   751,   755,   757,   759,   765,   771,   780,   782,   784,
     790,   798,   800,   803,   807,   809,   817,   819,   824,   828,
     830,   832,   834,   836,   838,   840,   842,   844,   846,   848,
     851,   861,   878,   894,   896,   900,   905,   906,   908,   915,
     917,   921,   923,   925,   927,   929,   931,   933,   935,   937,
     939,   941,   943,   945,   947,   949,   951,   953,   962,   964,
     966,   971,   972,   974,   983,   984,   986,   992,   998,  1004,
    1012,  1019,  1027,  1034,  1036,  1038,  1040,  1047,  1048,  1049,
    1052,  1053,  1054,  1055,  1062,  1068,  1077,  1084,  1090,  1096,
    1104,  1106,  1110,  1112,  1116,  1118,  1122,  1124,  1129,  1130,
    1135,  1136,  1138,  1142,  1144,  1148,  1150,  1154,  1156,  1158,
    1166,  1169,  1172,  1174,  1176,  1180,  1182,  1184,  1186,  1188,
    1190,  1192,  1194,  1196,  1198,  1200,  1202,  1206,  1208,  1210,
    1212,  1214,  1216,  1218,  1221,  1224,  1226,  1228,  1230,  1232,
    1234,  1245,  1246,  1248,  1252,  1256,  1260,  1264,  1268,  1274,
    1276,  1280,  1283,  1287,  1291
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

#define YYPACT_NINF -527

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-527)))

#define YYTABLE_NINF -316

#define yytable_value_is_error(Yytable_value) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
      90,  1278,  1278,    53,  -527,    90,  -527,  -527,  -527,  -527,
    -527,  -527,  -527,  -527,  -527,  -527,  -527,  -527,  -527,  -527,
    -527,  -527,  -527,  -527,  -527,  -527,  -527,  -527,  -527,  -527,
    -527,  -527,  -527,  -527,  -527,  -527,  -527,  -527,  -527,  -527,
    -527,  -527,  -527,  -527,  -527,  -527,  -527,  -527,  -527,  -527,
    -527,  -527,  -527,   177,   177,  -527,  -527,  -527,   420,  -527,
    -527,  -527,    92,  1278,   178,  1278,  1278,   181,   877,     6,
     857,   420,  -527,  -527,  -527,  1387,    68,    83,  -527,    82,
    -527,  -527,  -527,  -527,   -26,  1322,   137,   137,   -12,    83,
      85,    85,    85,    85,   100,   111,  1278,   154,   123,   420,
    -527,  -527,  -527,  -527,  -527,  -527,  -527,  -527,   370,  -527,
    -527,  -527,  -527,   131,  -527,  -527,  -527,  -527,  -527,  -527,
    -527,  -527,  -527,  -527,  -527,  -527,  -527,  -527,  -527,  1387,
    -527,   -27,  -527,  -527,  -527,  -527,   223,   -36,  -527,  -527,
     159,   167,   183,     9,  -527,    83,   420,    82,   195,   110,
     -26,   194,   680,  1405,   159,   167,   183,  -527,    66,    83,
    -527,    83,    83,   207,    83,   209,  -527,    -9,  1278,  1278,
    1278,  1278,  1068,   210,   211,   113,  1278,  -527,  -527,  -527,
    1342,   217,    85,    85,    85,    85,   210,   111,  -527,  -527,
    -527,  -527,  -527,  -527,  -527,   259,  -527,  -527,  -527,   160,
    -527,  -527,  1374,  -527,  -527,  -527,  -527,  -527,  -527,  1278,
     235,   260,   -26,   256,   -26,   233,  -527,   244,   247,    19,
    -527,   250,  -527,    44,    42,   125,   246,   133,    83,  -527,
    -527,   251,   266,   267,   273,   273,   273,   273,  -527,  1278,
     252,   271,   264,  1138,  1278,   302,  1278,  -527,  -527,   268,
     275,   281,  1278,   149,  1278,   310,   309,   131,  1278,  1278,
    1278,  1278,  1278,  1278,  -527,  -527,  -527,  -527,   313,  -527,
     312,  -527,   267,  -527,  -527,   318,   320,   319,   321,   -26,
    -527,    83,  1278,  -527,   311,  -527,   -26,   137,  1374,   137,
     137,  1374,   137,  -527,  -527,    -9,  -527,    83,   201,   201,
     201,   201,   325,  -527,   302,  -527,   273,   273,  -527,   113,
     385,   334,   169,  -527,   351,  1342,  -527,  -527,   273,   273,
     273,   273,   273,   204,  1374,  -527,   322,   -26,   256,   -26,
     -26,  -527,    44,   358,  -527,   356,  -527,   362,   366,   367,
      83,   372,   368,  -527,   327,  -527,  -527,  1390,  -527,  -527,
    -527,  -527,  -527,  -527,   201,   201,  -527,  -527,  1405,     2,
     376,  1405,  -527,  -527,  -527,  -527,  -527,   201,   201,   201,
     201,   201,  -527,   385,  -527,   876,  -527,  -527,  -527,  -527,
    -527,  -527,   375,  -527,  -527,  -527,   377,  -527,    61,   378,
    -527,    83,   665,   417,   386,  -527,  1390,   976,  -527,  -527,
    -527,  1278,  -527,  -527,  -527,  -527,  -527,  -527,  -527,  -527,
     388,  -527,  1278,   -26,   384,   383,  1405,   137,   137,   137,
    -527,  -527,   893,   998,  -527,   131,  -527,  -527,  -527,   382,
     394,     1,   389,  1405,  -527,   387,   391,   393,   397,  -527,
    -527,  -527,  -527,  -527,  -527,  -527,  -527,  -527,  -527,  -527,
    -527,  -527,   413,  -527,   395,  -527,  -527,   396,   399,   401,
     322,  1278,  -527,   398,   412,  -527,  -527,   236,  -527,  -527,
    -527,  -527,  -527,  -527,  -527,  -527,  -527,   454,  -527,   944,
     519,   322,  -527,  -527,  -527,  -527,    82,  -527,  1278,  -527,
    -527,   416,   409,   416,   441,   423,   444,   416,   425,   263,
     -26,  -527,  -527,  -527,   481,   322,  -527,   -26,   449,   -26,
      38,   426,   505,   556,  -527,   429,   -26,   324,   431,   333,
     194,   422,   519,   428,  -527,   437,   430,   432,  -527,   -26,
     441,   317,  -527,   440,   371,   -26,   432,   416,   433,   416,
     438,   444,   416,   443,   -26,   445,   324,  -527,  -527,  -527,
    -527,   470,  -527,   261,   429,   -26,   416,  -527,   574,   356,
    -527,  -527,   453,  -527,  -527,   194,   592,   -26,   483,   -26,
     556,   429,   -26,   324,   194,  -527,  -527,  -527,  -527,  -527,
    -527,  -527,  1278,   463,   461,   455,   -26,   466,   -26,   263,
    -527,   322,  -527,  -527,   263,   493,   469,   458,   432,   477,
     -26,   432,   478,  -527,   467,  1405,  1309,  -527,   194,   -26,
     482,   491,  -527,   492,   713,  -527,   -26,   416,   720,  -527,
     194,   731,  -527,  1278,  1278,   -26,   480,  -527,  1278,   432,
     -26,  -527,   493,   263,  -527,   496,   -26,   263,  -527,  -527,
     263,   493,  -527,    79,    77,   485,  1278,  -527,   767,   495,
    -527,   497,   -26,   500,   502,   503,  -527,  -527,  1278,  1208,
     504,  1278,  1278,  -527,   109,   263,  -527,   -26,  -527,   432,
     -26,  -527,   493,   184,   499,   156,  1278,  -527,   146,  -527,
     506,   432,   778,   508,  -527,  -527,  -527,  -527,  -527,  -527,
    -527,   785,   263,  -527,   -26,   263,  -527,   510,   432,   511,
    -527,   832,  -527,   263,  -527,   512,  -527
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
       0,     0,     0,   147,   168,   167,   201,   203,   170,   171,
     172,   173,   174,   154,     0,     0,   141,   158,     0,   157,
       0,     0,   190,   188,   189,   192,   194,     0,     0,     0,
       0,     0,   183,   157,    87,     0,    68,   277,   273,   278,
     276,   135,     0,   362,   131,   216,     0,   213,     0,     0,
     218,     0,     0,     0,     0,   228,   203,     0,   226,   176,
     177,     0,   163,   165,   186,   178,   179,   180,   181,   182,
       0,   305,   283,   274,   300,     0,     0,   136,   136,   136,
     169,   248,     0,     0,   227,     7,   204,   224,   225,   159,
       0,   157,     0,     0,   304,     0,     0,     0,     0,   267,
     251,   252,   253,   254,   260,   261,   262,   255,   256,   257,
     258,   259,   148,   263,     0,   265,   266,     0,   249,    56,
       0,     0,   202,     0,     0,   184,   280,     0,   284,   286,
     301,   114,   214,   220,   219,   149,   264,     0,   247,     0,
       0,     0,   160,   161,   269,   268,   270,   285,     0,   250,
     350,     0,     0,     0,     0,     0,   321,     0,     0,     0,
     274,   241,   339,   311,   308,     0,   356,   274,     0,   274,
       0,   359,     0,     0,   320,     0,   274,     0,     0,     0,
       0,     0,     0,     0,   354,     0,     0,     0,   357,   274,
       0,     0,   323,     0,     0,   274,     0,     0,     0,     0,
       0,   321,     0,     0,   274,     0,   317,   319,   312,   314,
     349,     0,   240,     0,     0,   274,     0,   355,     0,     0,
     360,   322,     0,   338,   316,     0,     0,   274,     0,   274,
       0,     0,   274,     0,     0,   340,   318,   309,   287,   288,
     289,   307,     0,     0,   302,     0,   274,     0,   274,     0,
     347,     0,   324,   337,     0,   351,     0,     0,     0,     0,
     274,     0,     0,   336,     0,     0,     0,   306,     0,   274,
       0,     0,   358,     0,     0,   345,   274,     0,     0,   326,
       0,     0,   327,     0,     0,   274,     0,   303,     0,     0,
     274,   348,   351,     0,   352,     0,   274,     0,   334,   325,
       0,   351,   290,     0,     0,     0,     0,   242,     0,     0,
     346,     0,   274,     0,     0,     0,   332,   298,     0,     0,
       0,     0,     0,   296,     0,     0,   342,   274,   353,     0,
     274,   335,   351,     0,     0,     0,     0,   292,     0,   299,
       0,     0,     0,     0,   333,   295,   294,   293,   291,   297,
     341,     0,     0,   329,   274,     0,   343,     0,     0,     0,
     328,     0,   344,     0,   330,     0,   331
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -527,  -527,   571,  -527,  -249,    -1,   -60,   522,   542,   -56,
    -527,  -527,  -527,  -266,  -527,  -213,  -527,  -119,   -75,   -69,
     -68,   -66,  -167,   451,   476,  -527,   -83,  -527,  -527,  -243,
    -527,  -527,   -77,   419,   298,  -527,   -44,   314,  -527,  -527,
     434,   306,  -527,   180,  -527,  -527,  -252,  -527,   -19,   221,
    -527,  -527,  -527,  -141,  -527,  -527,  -527,  -527,  -527,  -527,
    -527,   300,  -527,   315,   557,  -527,  -190,   228,   558,  -527,
    -527,   404,  -527,  -527,  -527,   232,   248,  -527,   219,  -527,
     161,  -527,  -527,   316,   -81,    43,   -62,  -494,  -527,  -527,
    -476,  -527,  -527,  -320,    37,  -437,  -527,  -527,   128,  -503,
      81,  -519,   106,  -429,  -527,  -446,  -526,  -480,  -524,  -458,
    -527,   122,   143,    95,  -527,  -527
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
     362,   313,   255,   256,   393,   109,   348,   394,   395,   111,
     112,   166,   179,   396,   397,   126,   398,    72,   145,   423,
     457,   458,   487,   277,   524,   413,   500,   217,   414,   583,
     643,   626,   584,   415,   585,   376,   554,   522,   501,   518,
     533,   545,   515,   502,   547,   519,   615,   525,   558,   507,
     511,   512,   286,   384,    73,    74
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[] =
{
      53,    54,   159,   151,    79,    84,   140,   141,   317,   142,
     162,   164,   566,   253,   165,   127,   549,   147,   357,   357,
     296,   229,   338,   480,   160,   341,   550,   576,   562,   527,
     149,   564,   244,   196,   503,   509,   536,   197,   148,   516,
     230,   334,   202,   181,   504,   262,   207,   169,   170,   171,
     235,   236,   237,    55,   602,   150,   220,   251,   374,   326,
     140,   141,    76,   142,    80,    81,   532,   534,   523,   215,
     209,   593,   163,   528,   618,   587,   503,   621,  -162,   567,
     603,   569,   113,   218,   572,   466,   611,   467,   546,   381,
     210,   613,   208,     1,     2,   177,   435,   254,   588,   223,
     221,   224,   225,   220,   227,   648,   650,   402,   610,   349,
     350,   351,   590,   470,   628,   656,   148,   546,   148,   375,
     595,   410,   283,   287,   534,   306,   639,   307,   283,   629,
     651,   275,    75,   278,   654,   148,    78,   655,   258,   259,
     260,   261,   418,   146,   546,   682,   684,   221,   253,   222,
     649,   247,   248,   165,   612,   662,   148,   691,   657,   636,
     658,   466,   680,   659,   399,   400,   660,   661,   634,   167,
     664,   240,   638,   160,   701,   641,   462,   405,   406,   407,
     408,   409,   673,   675,   172,   212,   678,   681,   679,   697,
     658,   213,   699,   659,   214,   175,   660,   661,   331,   148,
     705,   180,   666,   288,   193,   336,   289,   148,   272,   178,
     337,   291,   339,   340,   292,   342,   299,   300,   301,   264,
     265,   332,   344,   363,   364,   689,  -187,   658,  -187,    77,
     659,    78,   254,   660,   661,   312,   693,   658,   302,   204,
     659,   687,   240,   660,   661,   696,   377,   205,   379,   380,
      57,   311,    58,   314,    82,   704,    83,   318,   319,   320,
     321,   322,   323,   206,   490,   658,   685,   578,   659,   211,
     216,   660,   661,   226,   346,   401,   347,   372,   404,   373,
     388,   333,   198,   199,   200,   201,   228,   354,   355,    78,
     484,   485,   412,   257,   243,   245,   207,   128,   153,   367,
     368,   369,   370,   371,   491,   492,   493,   494,   495,   496,
     497,   273,   276,   274,    78,   279,   344,   280,   490,  -281,
     130,   131,   132,   133,   134,   135,   136,   281,   579,   580,
     282,   290,   432,   412,   490,   498,   294,   303,    83,  -281,
     436,   437,   438,   295,  -281,   202,   581,   297,   304,   305,
     412,   238,   309,   308,   140,   141,   310,   142,   491,   492,
     493,   494,   495,   496,   497,   537,   538,   539,   494,   540,
     541,   542,   490,  -281,   491,   492,   493,   494,   495,   496,
     497,   182,   183,   184,   185,   186,   187,   315,   316,   498,
     324,   325,    83,   561,   327,   283,   543,   328,  -281,    83,
     429,   329,   357,   375,   391,   498,   330,   486,    83,  -313,
     352,   431,   491,   492,   493,   494,   495,   496,   497,   520,
     361,    61,   460,    -5,    -5,    62,    -5,    -5,    -5,    -5,
      -5,    -5,    -5,    -5,    -5,    -5,    -5,   312,    -5,    -5,
     382,   383,    -5,   498,   385,   386,    83,  -315,   559,   387,
     390,   389,   403,   535,   565,   544,   416,   392,   417,   419,
     481,   433,   425,   574,   430,   434,   463,   465,   475,   471,
     582,    63,    64,   472,   469,   473,   479,    65,    66,   474,
     477,   478,    -9,   482,   544,   483,   596,   505,   598,    67,
     488,   601,   586,   508,   510,    -5,   -62,   506,   513,   514,
     517,   521,   526,   530,    83,   608,   490,   548,   551,   600,
     555,   544,   553,   570,   557,   556,   563,   568,   573,   620,
     490,   575,   624,   582,   577,  -310,  -310,  -310,  -310,   592,
    -310,  -310,  -310,  -310,  -310,   635,   597,   605,   606,   609,
     607,   614,   616,   617,   645,   623,   491,   492,   493,   494,
     495,   496,   497,   619,   622,   653,   630,   490,   646,  -310,
     491,   492,   493,   494,   495,   496,   497,   631,   632,   652,
     663,   669,   667,   668,   670,   490,    56,   498,   671,   672,
     531,   604,   690,   676,   686,   694,   700,   702,   706,   683,
     100,   498,  -310,   490,   499,  -310,    60,   491,   492,   493,
     494,   495,   496,   497,   219,   195,   263,   356,   246,   343,
     353,   468,   420,   698,   365,   491,   492,   493,   494,   495,
     496,   497,   642,   644,   426,   108,   110,   647,   498,   428,
     366,    83,   293,   491,   492,   493,   494,   495,   496,   497,
     489,   424,   461,   627,   378,   642,   498,   571,   625,   589,
     552,   599,   560,   529,   591,     0,     0,   642,   642,     0,
     677,   642,     0,     0,   498,     0,   421,   594,  -246,  -246,
    -246,     0,  -246,  -246,  -246,   688,  -246,  -246,  -246,  -246,
    -246,     0,     0,     0,  -246,  -246,  -246,  -246,  -246,  -246,
    -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,
    -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,  -246,
    -246,  -246,  -246,  -246,   490,  -246,   128,  -246,  -246,     0,
       0,   490,     0,     0,  -246,  -246,  -246,  -246,  -246,  -246,
    -246,  -246,   490,    78,  -246,  -246,  -246,  -246,     0,   130,
     131,   132,   133,   134,   135,   136,     0,     0,     0,   422,
       0,     0,     0,     0,   491,   492,   493,   494,   495,   496,
     497,   491,   492,   493,   494,   495,   496,   497,   490,     0,
       0,     0,   491,   492,   493,   494,   495,   496,   497,   490,
       0,     0,     0,     0,     0,   498,   490,     0,   633,     0,
       0,     0,   498,     0,     0,   637,     0,     0,     0,     0,
       0,     0,     0,   498,     0,     0,   640,     0,   491,   492,
     493,   494,   495,   496,   497,     0,     0,     0,     0,   491,
     492,   493,   494,   495,   496,   497,   491,   492,   493,   494,
     495,   496,   497,   490,     0,     0,     0,     0,     0,   498,
       0,     0,   665,     0,     0,     0,     0,     0,     0,     0,
     498,     0,     0,   692,     0,     0,     0,   498,     0,     0,
     695,     0,     0,   114,   115,   116,   117,     0,   118,   119,
     120,   121,   122,   491,   492,   493,   494,   495,   496,   497,
       1,     2,     0,    85,    86,    87,    88,    89,    90,    91,
      92,    93,    94,    95,   439,    96,    97,   123,     0,    98,
       0,     0,     0,     0,   498,     0,     0,   703,     0,     0,
       0,     0,   128,   153,   440,     0,   441,   442,   443,   444,
     445,   446,     0,     0,   447,   448,   449,   450,   451,    78,
     124,     0,     0,   125,     0,   130,   131,   132,   133,   134,
     135,   136,   452,   453,     0,   439,     0,     0,     0,     0,
       0,     0,    99,     0,     0,     0,     0,     0,   411,   454,
       0,     0,     0,   455,   456,   440,     0,   441,   442,   443,
     444,   445,   446,     0,     0,   447,   448,   449,   450,   451,
       0,     0,   114,   115,   116,   117,     0,   118,   119,   120,
     121,   122,     0,   452,   453,     0,     0,     0,     0,     0,
       0,     6,     7,     8,     0,     9,    10,    11,     0,    12,
      13,    14,    15,    16,   455,   456,   123,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    31,    32,    33,   128,   129,    34,    35,    36,    37,
      38,    39,    40,    41,    42,    43,    44,     0,    45,   427,
      46,   459,   125,     0,     0,     0,     0,   130,   131,   132,
     133,   134,   135,   136,    48,     0,     0,    49,    50,    51,
      52,     6,     7,     8,     0,     9,    10,    11,     0,    12,
      13,    14,    15,    16,     0,     0,     0,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    31,    32,    33,     0,     0,    34,    35,    36,    37,
      38,    39,    40,    41,    42,    43,    44,   238,    45,     0,
      46,    47,   239,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    48,     0,     0,    49,    50,    51,
      52,     6,     7,     8,     0,     9,    10,    11,     0,    12,
      13,    14,    15,    16,     0,     0,     0,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    31,    32,    33,     0,     0,    34,    35,    36,    37,
      38,    39,    40,    41,    42,    43,    44,     0,    45,     0,
      46,    47,   239,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    48,     0,     0,    49,    50,    51,
      52,     6,     7,     8,     0,     9,    10,    11,     0,    12,
      13,    14,    15,    16,     0,     0,     0,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    31,    32,    33,     0,     0,    34,    35,    36,    37,
      38,    39,    40,    41,    42,    43,    44,     0,    45,     0,
      46,    47,   674,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    48,     0,     0,    49,    50,    51,
      52,     6,     7,     8,     0,     9,    10,    11,     0,    12,
      13,    14,    15,    16,     0,     0,     0,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    31,    32,    33,     0,   578,    34,    35,    36,    37,
      38,    39,    40,    41,    42,    43,    44,     0,    45,     0,
      46,    47,     0,     0,     0,     0,     0,     0,   152,     0,
       0,     0,     0,     0,    48,   128,   153,    49,    50,    51,
      52,     0,     0,     0,     0,     0,     0,     0,   128,   153,
     252,     0,    78,     0,     0,     0,     0,     0,   130,   131,
     132,   133,   134,   135,   136,    78,   579,   580,   128,   153,
       0,   130,   131,   132,   133,   134,   135,   136,     0,     0,
       0,     0,     0,     0,     0,    78,    85,    86,    87,    88,
      89,   130,   131,   132,   133,   134,   135,   136,    96,    97,
     128,   153,    98,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   128,   129,     0,     0,    78,   266,   267,
     392,     0,     0,   130,   131,   132,   133,   134,   135,   136,
      78,   128,   153,     0,     0,     0,   130,   131,   132,   133,
     134,   135,   136,     0,     0,     0,     0,     0,    78,     0,
       0,     0,     0,     0,   130,   131,   132,   133,   134,   135,
     136
};

static const yytype_int16 yycheck[] =
{
       1,     2,    85,    84,    64,    67,    75,    75,   257,    75,
      87,    88,   536,   180,    89,    71,   519,    77,    17,    17,
     233,    30,   288,   460,    36,   291,   520,   546,   531,   509,
      56,   534,   173,    60,   480,   493,   516,    64,    74,   497,
      49,   284,    78,    99,   481,   186,    37,    91,    92,    93,
     169,   170,   171,     0,   573,    81,    37,   176,   324,   272,
     129,   129,    63,   129,    65,    66,   512,   513,   505,   150,
     145,   565,    84,   510,   598,   555,   522,   601,    76,   537,
     574,   539,    76,   152,   542,    84,   589,    86,   517,   332,
     146,   594,    83,     3,     4,    96,   416,   180,   556,   159,
      81,   161,   162,    37,   164,   629,   632,   359,   588,   299,
     300,   301,   558,   433,   608,   641,    74,   546,    74,    81,
     566,   373,    84,    81,   570,   244,   620,   246,    84,   609,
     633,   212,    40,   214,   637,    74,    53,   640,   182,   183,
     184,   185,    81,    75,   573,   669,   672,    81,   315,    83,
     630,    38,    39,   228,   591,    78,    74,   681,    79,   617,
      81,    84,   665,    84,   354,   355,    87,    88,   614,    84,
     646,   172,   618,    36,   698,   621,   425,   367,   368,   369,
     370,   371,   658,   659,    84,    75,   662,   667,    79,   692,
      81,    81,   695,    84,    84,    84,    87,    88,   279,    74,
     703,    78,   648,    78,    73,   286,    81,    74,   209,    55,
     287,    78,   289,   290,    81,   292,   235,   236,   237,    59,
      60,   281,   297,    54,    55,    79,    77,    81,    79,    51,
      84,    53,   315,    87,    88,    86,   682,    81,   239,    80,
      84,    85,   243,    87,    88,   691,   327,    80,   329,   330,
      73,   252,    75,   254,    73,   701,    75,   258,   259,   260,
     261,   262,   263,    80,     1,    81,    82,     6,    84,    74,
      76,    87,    88,    66,    73,   358,    75,    73,   361,    75,
     340,   282,    59,    60,    61,    62,    77,   306,   307,    53,
      54,    55,   375,    76,    84,    84,    37,    36,    37,   318,
     319,   320,   321,   322,    41,    42,    43,    44,    45,    46,
      47,    76,    56,    53,    53,    82,   391,    73,     1,    56,
      59,    60,    61,    62,    63,    64,    65,    80,    67,    68,
      80,    85,   413,   416,     1,    72,    85,    85,    75,    76,
     417,   418,   419,    77,    81,    78,    85,    74,    77,    85,
     433,    49,    77,    85,   423,   423,    75,   423,    41,    42,
      43,    44,    45,    46,    47,    41,    42,    43,    44,    45,
      46,    47,     1,    56,    41,    42,    43,    44,    45,    46,
      47,    11,    12,    13,    14,    15,    16,    77,    79,    72,
      77,    79,    75,    76,    76,    84,    72,    77,    81,    75,
     401,    82,    17,    81,    77,    72,    85,   467,    75,    76,
      85,   412,    41,    42,    43,    44,    45,    46,    47,   500,
      86,     1,   423,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    14,    15,    16,    86,    18,    19,
      82,    85,    22,    72,    82,    79,    75,    76,   529,    82,
      82,    79,    76,   515,   535,   517,    81,    40,    81,    81,
     461,    77,    76,   544,    76,    82,    84,    73,    55,    82,
     553,    51,    52,    82,    85,    82,    77,    57,    58,    82,
      85,    85,    81,    85,   546,    73,   567,   488,   569,    69,
      36,   572,   554,    84,    53,    75,    76,    81,    75,    55,
      75,    20,    53,    77,    75,   586,     1,    76,    86,   571,
      73,   573,    84,    75,    82,    85,    76,    84,    75,   600,
       1,    76,   605,   606,    54,     6,     7,     8,     9,    76,
      11,    12,    13,    14,    15,   616,    53,    74,    77,    73,
      85,    48,    73,    85,   625,    78,    41,    42,    43,    44,
      45,    46,    47,    76,    76,   636,    74,     1,    78,    40,
      41,    42,    43,    44,    45,    46,    47,    76,    76,    73,
      85,   652,    77,    76,    74,     1,     5,    72,    76,    76,
      75,   582,    76,    79,    85,    77,    76,    76,    76,   670,
      68,    72,    73,     1,    75,    76,    54,    41,    42,    43,
      44,    45,    46,    47,   153,   129,   187,   309,   174,   295,
     304,   431,   391,   694,   314,    41,    42,    43,    44,    45,
      46,    47,   623,   624,   396,    68,    68,   628,    72,   397,
     315,    75,   228,    41,    42,    43,    44,    45,    46,    47,
     479,   393,   423,   606,   328,   646,    72,   541,   605,    75,
     522,   570,   530,   510,   559,    -1,    -1,   658,   659,    -1,
     661,   662,    -1,    -1,    72,    -1,     1,    75,     3,     4,
       5,    -1,     7,     8,     9,   676,    11,    12,    13,    14,
      15,    -1,    -1,    -1,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,     1,    50,    36,    52,    53,    -1,
      -1,     1,    -1,    -1,    59,    60,    61,    62,    63,    64,
      65,    66,     1,    53,    69,    70,    71,    72,    -1,    59,
      60,    61,    62,    63,    64,    65,    -1,    -1,    -1,    84,
      -1,    -1,    -1,    -1,    41,    42,    43,    44,    45,    46,
      47,    41,    42,    43,    44,    45,    46,    47,     1,    -1,
      -1,    -1,    41,    42,    43,    44,    45,    46,    47,     1,
      -1,    -1,    -1,    -1,    -1,    72,     1,    -1,    75,    -1,
      -1,    -1,    72,    -1,    -1,    75,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    72,    -1,    -1,    75,    -1,    41,    42,
      43,    44,    45,    46,    47,    -1,    -1,    -1,    -1,    41,
      42,    43,    44,    45,    46,    47,    41,    42,    43,    44,
      45,    46,    47,     1,    -1,    -1,    -1,    -1,    -1,    72,
      -1,    -1,    75,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      72,    -1,    -1,    75,    -1,    -1,    -1,    72,    -1,    -1,
      75,    -1,    -1,     6,     7,     8,     9,    -1,    11,    12,
      13,    14,    15,    41,    42,    43,    44,    45,    46,    47,
       3,     4,    -1,     6,     7,     8,     9,    10,    11,    12,
      13,    14,    15,    16,     1,    18,    19,    40,    -1,    22,
      -1,    -1,    -1,    -1,    72,    -1,    -1,    75,    -1,    -1,
      -1,    -1,    36,    37,    21,    -1,    23,    24,    25,    26,
      27,    28,    -1,    -1,    31,    32,    33,    34,    35,    53,
      73,    -1,    -1,    76,    -1,    59,    60,    61,    62,    63,
      64,    65,    49,    50,    -1,     1,    -1,    -1,    -1,    -1,
      -1,    -1,    75,    -1,    -1,    -1,    -1,    -1,    82,    66,
      -1,    -1,    -1,    70,    71,    21,    -1,    23,    24,    25,
      26,    27,    28,    -1,    -1,    31,    32,    33,    34,    35,
      -1,    -1,     6,     7,     8,     9,    -1,    11,    12,    13,
      14,    15,    -1,    49,    50,    -1,    -1,    -1,    -1,    -1,
      -1,     3,     4,     5,    -1,     7,     8,     9,    -1,    11,
      12,    13,    14,    15,    70,    71,    40,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    34,    35,    36,    37,    38,    39,    40,    41,
      42,    43,    44,    45,    46,    47,    48,    -1,    50,    73,
      52,    53,    76,    -1,    -1,    -1,    -1,    59,    60,    61,
      62,    63,    64,    65,    66,    -1,    -1,    69,    70,    71,
      72,     3,     4,     5,    -1,     7,     8,     9,    -1,    11,
      12,    13,    14,    15,    -1,    -1,    -1,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    34,    35,    -1,    -1,    38,    39,    40,    41,
      42,    43,    44,    45,    46,    47,    48,    49,    50,    -1,
      52,    53,    54,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    66,    -1,    -1,    69,    70,    71,
      72,     3,     4,     5,    -1,     7,     8,     9,    -1,    11,
      12,    13,    14,    15,    -1,    -1,    -1,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    34,    35,    -1,    -1,    38,    39,    40,    41,
      42,    43,    44,    45,    46,    47,    48,    -1,    50,    -1,
      52,    53,    54,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    66,    -1,    -1,    69,    70,    71,
      72,     3,     4,     5,    -1,     7,     8,     9,    -1,    11,
      12,    13,    14,    15,    -1,    -1,    -1,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    34,    35,    -1,    -1,    38,    39,    40,    41,
      42,    43,    44,    45,    46,    47,    48,    -1,    50,    -1,
      52,    53,    54,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    66,    -1,    -1,    69,    70,    71,
      72,     3,     4,     5,    -1,     7,     8,     9,    -1,    11,
      12,    13,    14,    15,    -1,    -1,    -1,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    34,    35,    -1,     6,    38,    39,    40,    41,
      42,    43,    44,    45,    46,    47,    48,    -1,    50,    -1,
      52,    53,    -1,    -1,    -1,    -1,    -1,    -1,    16,    -1,
      -1,    -1,    -1,    -1,    66,    36,    37,    69,    70,    71,
      72,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    36,    37,
      18,    -1,    53,    -1,    -1,    -1,    -1,    -1,    59,    60,
      61,    62,    63,    64,    65,    53,    67,    68,    36,    37,
      -1,    59,    60,    61,    62,    63,    64,    65,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    53,     6,     7,     8,     9,
      10,    59,    60,    61,    62,    63,    64,    65,    18,    19,
      36,    37,    22,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    36,    37,    -1,    -1,    53,    54,    55,
      40,    -1,    -1,    59,    60,    61,    62,    63,    64,    65,
      53,    36,    37,    -1,    -1,    -1,    59,    60,    61,    62,
      63,    64,    65,    -1,    -1,    -1,    -1,    -1,    53,    -1,
      -1,    -1,    -1,    -1,    59,    60,    61,    62,    63,    64,
      65
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
      82,    77,    40,   153,   156,   157,   162,   163,   165,   155,
     155,   115,   135,    76,   115,   155,   155,   155,   155,   155,
     135,    82,   115,   174,   177,   182,    81,    81,    81,    81,
     138,     1,    84,   168,   165,    76,   156,    73,   164,    94,
      76,    94,   173,    77,    82,   182,   121,   121,   121,     1,
      21,    23,    24,    25,    26,    27,    28,    31,    32,    33,
      34,    35,    49,    50,    66,    70,    71,   169,   170,    53,
      94,   167,    93,    84,   133,    73,    84,    86,   132,    85,
     182,    82,    82,    82,    82,    55,   127,    85,    85,    77,
     184,    94,    85,    73,    54,    55,    95,   171,    36,   169,
       1,    41,    42,    43,    44,    45,    46,    47,    72,    75,
     175,   187,   192,   194,   184,    94,    81,   198,    84,   198,
      53,   199,   200,    75,    55,   191,   198,    75,   188,   194,
     173,    20,   186,   184,   173,   196,    53,   196,   184,   201,
      77,    75,   194,   189,   194,   175,   196,    41,    42,    43,
      45,    46,    47,    72,   175,   190,   192,   193,    76,   188,
     176,    86,   187,    84,   185,    73,    85,    82,   197,   173,
     200,    76,   188,    76,   188,   173,   197,   198,    84,   198,
      75,   191,   198,    75,   173,    76,   190,    54,     6,    67,
      68,    85,   115,   178,   181,   183,   175,   196,   198,    75,
     194,   202,    76,   176,    75,   194,   173,    53,   173,   189,
     175,   173,   190,   176,    94,    74,    77,    85,   173,    73,
     196,   188,   184,   188,    48,   195,    73,    85,   197,    76,
     173,   197,    76,    78,   115,   174,   180,   183,   176,   196,
      74,    76,    76,    75,   194,   173,   198,    75,   194,   176,
      75,   194,    94,   179,    94,   173,    78,    94,   197,   196,
     195,   188,    73,   173,   188,   188,   195,    79,    81,    84,
      87,    88,    78,    85,   179,    75,   194,    77,    76,   173,
      74,    76,    76,   179,    54,   179,    79,    94,   179,    79,
     188,   196,   197,   173,   195,    82,    85,    85,    94,    79,
      76,   197,    75,   194,    77,    75,   194,   188,   173,   188,
      76,   197,    76,    75,   194,   188,    76
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
     159,   160,   160,   161,   162,   162,   163,   163,   163,   164,
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
       9,     1,     3,     1,     2,     2,     1,     2,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       7,     5,    12,     5,     2,     1,     0,     3,     1,     1,
       3,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     2,     1,     1,     1,     1,     1,
       1,     0,     1,     3,     0,     1,     5,     5,     5,     4,
       3,     1,     1,     1,     3,     4,     3,     1,     1,     1,
       1,     4,     3,     4,     4,     4,     3,     7,     5,     6,
       1,     3,     1,     3,     3,     2,     3,     2,     0,     3,
       0,     1,     3,     1,     2,     1,     2,     1,     2,     1,
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
#line 194 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.modlist) = (yyvsp[0].modlist); modlist = (yyvsp[0].modlist); }
#line 2147 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 3:
#line 198 "xi-grammar.y" /* yacc.c:1646  */
    { 
		  (yyval.modlist) = 0; 
		}
#line 2155 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 4:
#line 202 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.modlist) = new AstChildren<Module>(lineno, (yyvsp[-1].module), (yyvsp[0].modlist)); }
#line 2161 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 5:
#line 206 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 2167 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 6:
#line 208 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 2173 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 7:
#line 212 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 2179 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 8:
#line 214 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 2185 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 9:
#line 219 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 2191 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 10:
#line 220 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MODULE); YYABORT; }
#line 2197 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 11:
#line 221 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MAINMODULE); YYABORT; }
#line 2203 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 12:
#line 222 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(EXTERN); YYABORT; }
#line 2209 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 13:
#line 224 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITCALL); YYABORT; }
#line 2215 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 14:
#line 225 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITNODE); YYABORT; }
#line 2221 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 15:
#line 226 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITPROC); YYABORT; }
#line 2227 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 16:
#line 228 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CHARE); }
#line 2233 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 17:
#line 229 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MAINCHARE); }
#line 2239 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 18:
#line 230 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(GROUP); }
#line 2245 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 19:
#line 231 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NODEGROUP); }
#line 2251 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 20:
#line 232 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ARRAY); }
#line 2257 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 21:
#line 236 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INCLUDE); YYABORT; }
#line 2263 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 22:
#line 237 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(STACKSIZE); YYABORT; }
#line 2269 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 23:
#line 238 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(THREADED); YYABORT; }
#line 2275 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 24:
#line 239 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(TEMPLATE); YYABORT; }
#line 2281 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 25:
#line 240 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(SYNC); YYABORT; }
#line 2287 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 26:
#line 241 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IGET); YYABORT; }
#line 2293 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 27:
#line 242 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(EXCLUSIVE); YYABORT; }
#line 2299 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 28:
#line 243 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IMMEDIATE); YYABORT; }
#line 2305 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 29:
#line 244 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(SKIPSCHED); YYABORT; }
#line 2311 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 30:
#line 245 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INLINE); YYABORT; }
#line 2317 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 31:
#line 246 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(VIRTUAL); YYABORT; }
#line 2323 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 32:
#line 247 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MIGRATABLE); YYABORT; }
#line 2329 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 33:
#line 248 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CREATEHERE); YYABORT; }
#line 2335 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 34:
#line 249 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CREATEHOME); YYABORT; }
#line 2341 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 35:
#line 250 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOKEEP); YYABORT; }
#line 2347 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 36:
#line 251 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOTRACE); YYABORT; }
#line 2353 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 37:
#line 252 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(APPWORK); YYABORT; }
#line 2359 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 38:
#line 255 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(PACKED); YYABORT; }
#line 2365 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 39:
#line 256 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(VARSIZE); YYABORT; }
#line 2371 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 40:
#line 257 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ENTRY); YYABORT; }
#line 2377 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 41:
#line 258 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(FOR); YYABORT; }
#line 2383 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 42:
#line 259 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(FORALL); YYABORT; }
#line 2389 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 43:
#line 260 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(WHILE); YYABORT; }
#line 2395 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 44:
#line 261 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(WHEN); YYABORT; }
#line 2401 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 45:
#line 262 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(OVERLAP); YYABORT; }
#line 2407 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 46:
#line 263 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ATOMIC); YYABORT; }
#line 2413 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 47:
#line 264 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IF); YYABORT; }
#line 2419 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 48:
#line 265 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ELSE); YYABORT; }
#line 2425 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 49:
#line 267 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(LOCAL); YYABORT; }
#line 2431 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 50:
#line 269 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(USING); YYABORT; }
#line 2437 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 51:
#line 270 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ACCEL); YYABORT; }
#line 2443 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 52:
#line 273 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ACCELBLOCK); YYABORT; }
#line 2449 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 53:
#line 274 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MEMCRITICAL); YYABORT; }
#line 2455 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 54:
#line 275 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(REDUCTIONTARGET); YYABORT; }
#line 2461 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 55:
#line 276 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CASE); YYABORT; }
#line 2467 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 56:
#line 280 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 2473 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 57:
#line 282 "xi-grammar.y" /* yacc.c:1646  */
    {
		  char *tmp = new char[strlen((yyvsp[-3].strval))+strlen((yyvsp[0].strval))+3];
		  sprintf(tmp,"%s::%s", (yyvsp[-3].strval), (yyvsp[0].strval));
		  (yyval.strval) = tmp;
		}
#line 2483 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 58:
#line 290 "xi-grammar.y" /* yacc.c:1646  */
    { 
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist)); 
		}
#line 2491 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 59:
#line 294 "xi-grammar.y" /* yacc.c:1646  */
    {  
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist)); 
		    (yyval.module)->setMain();
		}
#line 2500 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 60:
#line 301 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = 0; }
#line 2506 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 61:
#line 303 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = (yyvsp[-2].conslist); }
#line 2512 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 62:
#line 307 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = 0; }
#line 2518 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 63:
#line 309 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = new ConstructList(lineno, (yyvsp[-1].construct), (yyvsp[0].conslist)); }
#line 2524 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 64:
#line 313 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new UsingScope((yyvsp[0].strval), false); }
#line 2530 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 65:
#line 315 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new UsingScope((yyvsp[0].strval), true); }
#line 2536 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 66:
#line 317 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].member)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].member); }
#line 2542 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 67:
#line 319 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].message)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].message); }
#line 2548 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 68:
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
#line 2562 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 69:
#line 333 "xi-grammar.y" /* yacc.c:1646  */
    { if((yyvsp[-2].conslist)) (yyvsp[-2].conslist)->recurse<int&>((yyvsp[-4].intval), &Construct::setExtern); (yyval.construct) = (yyvsp[-2].conslist); }
#line 2568 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 70:
#line 335 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new Scope((yyvsp[-3].strval), (yyvsp[-1].conslist)); }
#line 2574 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 71:
#line 337 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = (yyvsp[-1].construct); }
#line 2580 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 72:
#line 339 "xi-grammar.y" /* yacc.c:1646  */
    {
          ERROR("preceding construct must be semicolon terminated",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 2590 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 73:
#line 345 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].module)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].module); }
#line 2596 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 74:
#line 347 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2602 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 75:
#line 349 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2608 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 76:
#line 351 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2614 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 77:
#line 353 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2620 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 78:
#line 355 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].templat)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].templat); }
#line 2626 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 79:
#line 357 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = NULL; }
#line 2632 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 80:
#line 359 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = NULL; }
#line 2638 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 81:
#line 361 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = (yyvsp[0].accelBlock); }
#line 2644 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 82:
#line 363 "xi-grammar.y" /* yacc.c:1646  */
    {
          ERROR("invalid construct",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 2654 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 83:
#line 371 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamType((yyvsp[0].type)); }
#line 2660 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 84:
#line 373 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
#line 2666 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 85:
#line 375 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
#line 2672 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 86:
#line 379 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = new TParamList((yyvsp[0].tparam)); }
#line 2678 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 87:
#line 381 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = new TParamList((yyvsp[-2].tparam), (yyvsp[0].tparlist)); }
#line 2684 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 88:
#line 385 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = 0; }
#line 2690 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 89:
#line 387 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = (yyvsp[0].tparlist); }
#line 2696 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 90:
#line 391 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = 0; }
#line 2702 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 91:
#line 393 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = (yyvsp[-1].tparlist); }
#line 2708 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 92:
#line 397 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("int"); }
#line 2714 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 93:
#line 399 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long"); }
#line 2720 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 94:
#line 401 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("short"); }
#line 2726 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 95:
#line 403 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("char"); }
#line 2732 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 96:
#line 405 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned int"); }
#line 2738 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 97:
#line 407 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long"); }
#line 2744 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 98:
#line 409 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long"); }
#line 2750 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 99:
#line 411 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long long"); }
#line 2756 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 100:
#line 413 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned short"); }
#line 2762 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 101:
#line 415 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned char"); }
#line 2768 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 102:
#line 417 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long long"); }
#line 2774 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 103:
#line 419 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("float"); }
#line 2780 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 104:
#line 421 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("double"); }
#line 2786 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 105:
#line 423 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long double"); }
#line 2792 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 106:
#line 425 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("void"); }
#line 2798 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 107:
#line 428 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ntype) = new NamedType((yyvsp[-1].strval),(yyvsp[0].tparlist)); }
#line 2804 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 108:
#line 429 "xi-grammar.y" /* yacc.c:1646  */
    { 
                    const char* basename, *scope;
                    splitScopedName((yyvsp[-1].strval), &scope, &basename);
                    (yyval.ntype) = new NamedType(basename, (yyvsp[0].tparlist), scope);
                }
#line 2814 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 109:
#line 437 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 2820 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 110:
#line 439 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ntype); }
#line 2826 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 111:
#line 443 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ptype) = new PtrType((yyvsp[-1].type)); }
#line 2832 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 112:
#line 447 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
#line 2838 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 113:
#line 449 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
#line 2844 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 114:
#line 453 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ftype) = new FuncType((yyvsp[-7].type), (yyvsp[-4].strval), (yyvsp[-1].plist)); }
#line 2850 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 115:
#line 457 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 2856 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 116:
#line 459 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 2862 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 117:
#line 461 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 2868 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 118:
#line 463 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ftype); }
#line 2874 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 119:
#line 465 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[0].type)); }
#line 2880 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 120:
#line 467 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[-1].type)); }
#line 2886 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 121:
#line 471 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 2892 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 122:
#line 473 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 2898 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 123:
#line 475 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 2904 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 124:
#line 477 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[0].type)); }
#line 2910 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 125:
#line 479 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[-1].type)); }
#line 2916 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 126:
#line 483 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ReferenceType((yyvsp[-1].type)); }
#line 2922 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 127:
#line 485 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 2928 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 128:
#line 489 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ReferenceType((yyvsp[-1].type)); }
#line 2934 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 129:
#line 491 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 2940 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 130:
#line 495 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 2946 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 131:
#line 499 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = (yyvsp[-1].val); }
#line 2952 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 132:
#line 503 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.vallist) = 0; }
#line 2958 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 133:
#line 505 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.vallist) = new ValueList((yyvsp[-1].val), (yyvsp[0].vallist)); }
#line 2964 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 134:
#line 509 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].vallist)); }
#line 2970 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 135:
#line 513 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[-3].type), (yyvsp[-1].strval), (yyvsp[0].vallist), 1); }
#line 2976 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 136:
#line 517 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0;}
#line 2982 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 137:
#line 519 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0;}
#line 2988 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 138:
#line 523 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 2994 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 139:
#line 525 "xi-grammar.y" /* yacc.c:1646  */
    { 
		  /*
		  printf("Warning: Message attributes are being phased out.\n");
		  printf("Warning: Please remove them from interface files.\n");
		  */
		  (yyval.intval) = (yyvsp[-1].intval); 
		}
#line 3006 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 140:
#line 535 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[0].intval); }
#line 3012 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 141:
#line 537 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[-2].intval) | (yyvsp[0].intval); }
#line 3018 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 142:
#line 541 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3024 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 143:
#line 543 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3030 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 144:
#line 547 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = 0; }
#line 3036 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 145:
#line 549 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-1].cattr); }
#line 3042 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 146:
#line 553 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[0].cattr); }
#line 3048 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 147:
#line 555 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
#line 3054 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 148:
#line 559 "xi-grammar.y" /* yacc.c:1646  */
    { python_doc = NULL; (yyval.intval) = 0; }
#line 3060 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 149:
#line 561 "xi-grammar.y" /* yacc.c:1646  */
    { python_doc = (yyvsp[0].strval); (yyval.intval) = 0; }
#line 3066 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 150:
#line 565 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CPYTHON; }
#line 3072 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 151:
#line 569 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = 0; }
#line 3078 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 152:
#line 571 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-1].cattr); }
#line 3084 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 153:
#line 575 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[0].cattr); }
#line 3090 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 154:
#line 577 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
#line 3096 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 155:
#line 581 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CMIGRATABLE; }
#line 3102 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 156:
#line 583 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CPYTHON; }
#line 3108 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 157:
#line 587 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3114 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 158:
#line 589 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 3120 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 159:
#line 592 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3126 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 160:
#line 594 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 3132 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 161:
#line 597 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mv) = new MsgVar((yyvsp[-3].type), (yyvsp[-2].strval), (yyvsp[-4].intval), (yyvsp[-1].intval)); }
#line 3138 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 162:
#line 601 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mvlist) = new MsgVarList((yyvsp[0].mv)); }
#line 3144 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 163:
#line 603 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mvlist) = new MsgVarList((yyvsp[-1].mv), (yyvsp[0].mvlist)); }
#line 3150 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 164:
#line 607 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, (yyvsp[0].ntype)); }
#line 3156 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 165:
#line 609 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, (yyvsp[-3].ntype), (yyvsp[-1].mvlist)); }
#line 3162 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 166:
#line 613 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = 0; }
#line 3168 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 167:
#line 615 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = (yyvsp[0].typelist); }
#line 3174 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 168:
#line 619 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = new TypeList((yyvsp[0].ntype)); }
#line 3180 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 169:
#line 621 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = new TypeList((yyvsp[-2].ntype), (yyvsp[0].typelist)); }
#line 3186 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 170:
#line 625 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr)|Chare::CCHARE, (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3192 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 171:
#line 627 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3198 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 172:
#line 631 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3204 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 173:
#line 635 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new NodeGroup(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3210 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 174:
#line 639 "xi-grammar.y" /* yacc.c:1646  */
    {/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",(yyvsp[-2].strval));
			(yyval.ntype) = new NamedType(buf); 
		}
#line 3220 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 175:
#line 645 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ntype) = new NamedType((yyvsp[-1].strval)); }
#line 3226 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 176:
#line 649 "xi-grammar.y" /* yacc.c:1646  */
    {  (yyval.chare) = new Array(lineno, (yyvsp[-4].cattr), (yyvsp[-3].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3232 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 177:
#line 651 "xi-grammar.y" /* yacc.c:1646  */
    {  (yyval.chare) = new Array(lineno, (yyvsp[-3].cattr), (yyvsp[-4].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3238 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 178:
#line 655 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr)|Chare::CCHARE, new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist));}
#line 3244 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 179:
#line 657 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3250 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 180:
#line 661 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3256 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 181:
#line 665 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new NodeGroup( lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3262 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 182:
#line 669 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Array( lineno, 0, (yyvsp[-3].ntype), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3268 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 183:
#line 673 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-1].strval))); }
#line 3274 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 184:
#line 675 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-4].strval)), (yyvsp[-2].mvlist)); }
#line 3280 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 185:
#line 679 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = 0; }
#line 3286 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 186:
#line 681 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3292 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 187:
#line 685 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = 0; }
#line 3298 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 188:
#line 687 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3304 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 189:
#line 689 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3310 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 190:
#line 693 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TType(new NamedType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3316 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 191:
#line 695 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TFunc((yyvsp[-1].ftype), (yyvsp[0].strval)); }
#line 3322 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 192:
#line 697 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TName((yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].strval)); }
#line 3328 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 193:
#line 701 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = new TVarList((yyvsp[0].tvar)); }
#line 3334 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 194:
#line 703 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = new TVarList((yyvsp[-2].tvar), (yyvsp[0].tvarlist)); }
#line 3340 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 195:
#line 707 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = (yyvsp[-1].tvarlist); }
#line 3346 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 196:
#line 711 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3352 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 197:
#line 713 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3358 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 198:
#line 715 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3364 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 199:
#line 717 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3370 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 200:
#line 719 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].message)); (yyvsp[0].message)->setTemplate((yyval.templat)); }
#line 3376 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 201:
#line 723 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = 0; }
#line 3382 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 202:
#line 725 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = (yyvsp[-2].mbrlist); }
#line 3388 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 203:
#line 729 "xi-grammar.y" /* yacc.c:1646  */
    { 
                  if (!connectEntries.empty()) {
                    (yyval.mbrlist) = new AstChildren<Member>(connectEntries);
		  } else {
		    (yyval.mbrlist) = 0; 
                  }
		}
#line 3400 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 204:
#line 737 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = new AstChildren<Member>(-1, (yyvsp[-1].member), (yyvsp[0].mbrlist)); }
#line 3406 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 205:
#line 741 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].readonly); }
#line 3412 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 206:
#line 743 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].readonly); }
#line 3418 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 208:
#line 746 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].member); }
#line 3424 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 209:
#line 748 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].pupable); }
#line 3430 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 210:
#line 750 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].includeFile); }
#line 3436 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 211:
#line 752 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new ClassDeclaration(lineno,(yyvsp[0].strval)); }
#line 3442 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 212:
#line 756 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1); }
#line 3448 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 213:
#line 758 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1); }
#line 3454 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 214:
#line 760 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[-6].strval)) + '<' +
					    ((yyvsp[-4].tparlist))->to_string() + '>').c_str()),
				    1);
		}
#line 3464 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 215:
#line 766 "xi-grammar.y" /* yacc.c:1646  */
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[-2]).first_column, (yylsp[-2]).last_column, (yylsp[-2]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1);
		}
#line 3474 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 216:
#line 772 "xi-grammar.y" /* yacc.c:1646  */
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[-5]).first_column, (yylsp[-5]).last_column, (yylsp[-5]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1);
		}
#line 3484 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 217:
#line 781 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 0); }
#line 3490 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 218:
#line 783 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 0); }
#line 3496 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 219:
#line 785 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[-6].strval)) + '<' +
					    ((yyvsp[-4].tparlist))->to_string() + '>').c_str()),
				    0);
		}
#line 3506 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 220:
#line 791 "xi-grammar.y" /* yacc.c:1646  */
    {
                  InitCall* rtn = new InitCall(lineno, (yyvsp[-3].strval), 0);
                  rtn->setAccel();
                  (yyval.member) = rtn;
		}
#line 3516 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 221:
#line 799 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[0].ntype),0); }
#line 3522 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 222:
#line 801 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[-2].ntype),(yyvsp[0].pupable)); }
#line 3528 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 223:
#line 804 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.includeFile) = new IncludeFile(lineno,(yyvsp[0].strval)); }
#line 3534 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 224:
#line 808 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[-1].member); }
#line 3540 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 225:
#line 810 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("preceding entry method declaration must be semicolon-terminated",
		        (yylsp[0]).first_column, (yylsp[0]).last_column);
		  YYABORT;
		}
#line 3550 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 226:
#line 818 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].entry); }
#line 3556 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 227:
#line 820 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyvsp[0].entry)->tspec = (yyvsp[-1].tvarlist);
                  (yyval.member) = (yyvsp[0].entry);
                }
#line 3565 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 228:
#line 825 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].member); }
#line 3571 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 229:
#line 829 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3577 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 230:
#line 831 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3583 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 231:
#line 833 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3589 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 232:
#line 835 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3595 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 233:
#line 837 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3601 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 234:
#line 839 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3607 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 235:
#line 841 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3613 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 236:
#line 843 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3619 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 237:
#line 845 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3625 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 238:
#line 847 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3631 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 239:
#line 849 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3637 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 240:
#line 852 "xi-grammar.y" /* yacc.c:1646  */
    { 
                  (yyval.entry) = new Entry(lineno, (yyvsp[-5].intval), (yyvsp[-4].type), (yyvsp[-3].strval), (yyvsp[-2].plist), (yyvsp[-1].val), (yyvsp[0].sentry), (const char *) NULL, (yylsp[-6]).first_line, (yyloc).last_line);
		  if ((yyvsp[0].sentry) != 0) { 
		    (yyvsp[0].sentry)->con1 = new SdagConstruct(SIDENT, (yyvsp[-3].strval));
                    (yyvsp[0].sentry)->entry = (yyval.entry);
                    (yyvsp[0].sentry)->con1->entry = (yyval.entry);
                    (yyvsp[0].sentry)->param = new ParamList((yyvsp[-2].plist));
                  }
		}
#line 3651 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 241:
#line 862 "xi-grammar.y" /* yacc.c:1646  */
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
#line 3672 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 242:
#line 879 "xi-grammar.y" /* yacc.c:1646  */
    {
                  int attribs = SACCEL;
                  const char* name = (yyvsp[-6].strval);
                  ParamList* paramList = (yyvsp[-5].plist);
                  ParamList* accelParamList = (yyvsp[-4].plist);
		  XStr* codeBody = new XStr((yyvsp[-2].strval));
                  const char* callbackName = (yyvsp[0].strval);

                  (yyval.entry) = new Entry(lineno, attribs, new BuiltinType("void"), name, paramList, 0, 0, 0 );
                  (yyval.entry)->setAccelParam(accelParamList);
                  (yyval.entry)->setAccelCodeBody(codeBody);
                  (yyval.entry)->setAccelCallbackName(new XStr(callbackName));
                }
#line 3690 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 243:
#line 895 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.accelBlock) = new AccelBlock(lineno, new XStr((yyvsp[-2].strval))); }
#line 3696 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 244:
#line 897 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.accelBlock) = new AccelBlock(lineno, NULL); }
#line 3702 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 245:
#line 901 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3708 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 246:
#line 905 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3714 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 247:
#line 907 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[-1].intval); }
#line 3720 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 248:
#line 909 "xi-grammar.y" /* yacc.c:1646  */
    { ERROR("invalid entry method attribute list",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
#line 3729 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 249:
#line 916 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[0].intval); }
#line 3735 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 250:
#line 918 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[-2].intval) | (yyvsp[0].intval); }
#line 3741 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 251:
#line 922 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = STHREADED; }
#line 3747 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 252:
#line 924 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SSYNC; }
#line 3753 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 253:
#line 926 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SIGET; }
#line 3759 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 254:
#line 928 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SLOCKED; }
#line 3765 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 255:
#line 930 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SCREATEHERE; }
#line 3771 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 256:
#line 932 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SCREATEHOME; }
#line 3777 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 257:
#line 934 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SNOKEEP; }
#line 3783 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 258:
#line 936 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SNOTRACE; }
#line 3789 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 259:
#line 938 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SAPPWORK; }
#line 3795 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 260:
#line 940 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SIMMEDIATE; }
#line 3801 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 261:
#line 942 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SSKIPSCHED; }
#line 3807 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 262:
#line 944 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SINLINE; }
#line 3813 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 263:
#line 946 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SLOCAL; }
#line 3819 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 264:
#line 948 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SPYTHON; }
#line 3825 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 265:
#line 950 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SMEM; }
#line 3831 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 266:
#line 952 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SREDUCE; }
#line 3837 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 267:
#line 954 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("invalid entry method attribute",
		        (yylsp[0]).first_column, (yylsp[0]).last_column);
		  yyclearin;
		  yyerrok;
		}
#line 3848 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 268:
#line 963 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 3854 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 269:
#line 965 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 3860 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 270:
#line 967 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 3866 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 271:
#line 971 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = ""; }
#line 3872 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 272:
#line 973 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3878 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 273:
#line 975 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s, %s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 3888 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 274:
#line 983 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = ""; }
#line 3894 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 275:
#line 985 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3900 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 276:
#line 987 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s[%s]%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 3910 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 277:
#line 993 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s{%s}%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 3920 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 278:
#line 999 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s(%s)%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 3930 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 279:
#line 1005 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"(%s)%s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 3940 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 280:
#line 1013 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval));
		}
#line 3949 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 281:
#line 1020 "xi-grammar.y" /* yacc.c:1646  */
    { 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			(yyval.intval) = 0;
		}
#line 3959 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 282:
#line 1028 "xi-grammar.y" /* yacc.c:1646  */
    { 
			in_braces=0;
			(yyval.intval) = 0;
		}
#line 3968 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 283:
#line 1035 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[0].type));}
#line 3974 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 284:
#line 1037 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval)); (yyval.pname)->setConditional((yyvsp[0].intval)); }
#line 3980 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 285:
#line 1039 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[-3].type),(yyvsp[-2].strval),0,(yyvsp[0].val));}
#line 3986 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 286:
#line 1041 "xi-grammar.y" /* yacc.c:1646  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
		}
#line 3995 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 287:
#line 1047 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READONLY; }
#line 4001 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 288:
#line 1048 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READWRITE; }
#line 4007 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 289:
#line 1049 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_WRITEONLY; }
#line 4013 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 290:
#line 1052 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr((yyvsp[0].strval)); }
#line 4019 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 291:
#line 1053 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "->" << (yyvsp[0].strval); }
#line 4025 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 292:
#line 1054 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[-2].xstrptr)) << "." << (yyvsp[0].strval); }
#line 4031 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 293:
#line 1056 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << *((yyvsp[-1].xstrptr)) << "]";
                  delete (yyvsp[-3].xstrptr);
                  delete (yyvsp[-1].xstrptr);
                }
#line 4042 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 294:
#line 1063 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << (yyvsp[-1].strval) << "]";
                  delete (yyvsp[-3].xstrptr);
                }
#line 4052 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 295:
#line 1069 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "(" << *((yyvsp[-1].xstrptr)) << ")";
                  delete (yyvsp[-3].xstrptr);
                  delete (yyvsp[-1].xstrptr);
                }
#line 4063 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 296:
#line 1078 "xi-grammar.y" /* yacc.c:1646  */
    {
                  in_bracket = 0;
                  (yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName(), (yyvsp[-1].strval));
                }
#line 4072 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 297:
#line 1085 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[-6].intval));
                }
#line 4082 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 298:
#line 1091 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType(Parameter::ACCEL_BUFFER_TYPE_READWRITE);
		}
#line 4092 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 299:
#line 1097 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.pname) = (yyvsp[-3].pname);
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[-5].intval));
		}
#line 4102 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 300:
#line 1105 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 4108 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 301:
#line 1107 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 4114 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 302:
#line 1111 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 4120 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 303:
#line 1113 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 4126 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 304:
#line 1117 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = (yyvsp[-1].plist); }
#line 4132 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 305:
#line 1119 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList(new Parameter(0, new BuiltinType("void"))); }
#line 4138 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 306:
#line 1123 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = (yyvsp[-1].plist); }
#line 4144 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 307:
#line 1125 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = 0; }
#line 4150 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 308:
#line 1129 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = 0; }
#line 4156 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 309:
#line 1131 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4162 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 310:
#line 1135 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = 0; }
#line 4168 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 311:
#line 1137 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[0].sc)); }
#line 4174 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 312:
#line 1139 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[-1].slist)); }
#line 4180 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 313:
#line 1143 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.slist) = new SListConstruct((yyvsp[0].sc)); }
#line 4186 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 314:
#line 1145 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.slist) = new SListConstruct((yyvsp[-1].sc), (yyvsp[0].slist));  }
#line 4192 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 315:
#line 1149 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.olist) = new OListConstruct((yyvsp[0].sc)); }
#line 4198 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 316:
#line 1151 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.olist) = new OListConstruct((yyvsp[-1].sc), (yyvsp[0].slist)); }
#line 4204 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 317:
#line 1155 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.clist) = new CaseListConstruct((yyvsp[0].when)); }
#line 4210 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 318:
#line 1157 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.clist) = new CaseListConstruct((yyvsp[-1].when), (yyvsp[0].clist)); }
#line 4216 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 319:
#line 1159 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("case blocks can only contain when clauses",
		        (yylsp[0]).first_column, (yylsp[0]).last_column);
		  (yyval.clist) = 0;
		}
#line 4226 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 320:
#line 1167 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4232 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 321:
#line 1169 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = 0; }
#line 4238 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 322:
#line 1173 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-2].entrylist), 0); }
#line 4244 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 323:
#line 1175 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-1].entrylist), (yyvsp[0].sc)); }
#line 4250 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 324:
#line 1177 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-3].entrylist), (yyvsp[-1].slist)); }
#line 4256 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 325:
#line 1181 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4262 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 326:
#line 1183 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4268 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 327:
#line 1185 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4274 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 328:
#line 1187 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4280 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 329:
#line 1189 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4286 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 330:
#line 1191 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4292 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 331:
#line 1193 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4298 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 332:
#line 1195 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4304 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 333:
#line 1197 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4310 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 334:
#line 1199 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4316 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 335:
#line 1201 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4322 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 336:
#line 1203 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4328 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 337:
#line 1207 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new AtomicConstruct((yyvsp[-1].strval), (yyvsp[-3].strval)); }
#line 4334 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 338:
#line 1209 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new OverlapConstruct((yyvsp[-1].olist)); }
#line 4340 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 339:
#line 1211 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = (yyvsp[0].when); }
#line 4346 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 340:
#line 1213 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new CaseConstruct((yyvsp[-1].clist)); }
#line 4352 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 341:
#line 1215 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForConstruct((yyvsp[-8].intexpr), (yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4358 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 342:
#line 1217 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForConstruct((yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4364 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 343:
#line 1219 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-9].strval)), (yyvsp[-6].intexpr),
		             (yyvsp[-4].intexpr), (yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4371 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 344:
#line 1222 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-11].strval)), (yyvsp[-8].intexpr),
		             (yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4378 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 345:
#line 1225 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new IfConstruct((yyvsp[-3].intexpr), (yyvsp[-1].sc), (yyvsp[0].sc)); }
#line 4384 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 346:
#line 1227 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new IfConstruct((yyvsp[-5].intexpr), (yyvsp[-2].slist), (yyvsp[0].sc)); }
#line 4390 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 347:
#line 1229 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new WhileConstruct((yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4396 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 348:
#line 1231 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new WhileConstruct((yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4402 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 349:
#line 1233 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new AtomicConstruct((yyvsp[-1].strval), NULL); }
#line 4408 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 350:
#line 1235 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("unknown SDAG construct or malformed entry method definition.\n"
		        "You may have forgotten to terminate a previous entry method definition with a"
		        " semicolon or forgotten to mark a block of sequential SDAG code as 'atomic'",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
#line 4420 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 351:
#line 1245 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = 0; }
#line 4426 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 352:
#line 1247 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ElseConstruct((yyvsp[0].sc)); }
#line 4432 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 353:
#line 1249 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ElseConstruct((yyvsp[-1].slist)); }
#line 4438 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 354:
#line 1253 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intexpr) = new IntExprConstruct((yyvsp[0].strval)); }
#line 4444 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 355:
#line 1257 "xi-grammar.y" /* yacc.c:1646  */
    { in_int_expr = 0; (yyval.intval) = 0; }
#line 4450 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 356:
#line 1261 "xi-grammar.y" /* yacc.c:1646  */
    { in_int_expr = 1; (yyval.intval) = 0; }
#line 4456 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 357:
#line 1265 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[-1].strval), (yyvsp[0].plist), 0, 0, 0, (yyloc).first_line, (yyloc).last_line);
		}
#line 4464 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 358:
#line 1269 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[-4].strval), (yyvsp[0].plist), 0, 0, (yyvsp[-2].strval), (yyloc).first_line, (yyloc).last_line);
		}
#line 4472 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 359:
#line 1275 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.entrylist) = new EntryList((yyvsp[0].entry)); }
#line 4478 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 360:
#line 1277 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.entrylist) = new EntryList((yyvsp[-2].entry),(yyvsp[0].entrylist)); }
#line 4484 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 361:
#line 1281 "xi-grammar.y" /* yacc.c:1646  */
    { in_bracket=1; }
#line 4490 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 362:
#line 1284 "xi-grammar.y" /* yacc.c:1646  */
    { in_bracket=0; }
#line 4496 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 363:
#line 1288 "xi-grammar.y" /* yacc.c:1646  */
    { if (!macroDefined((yyvsp[0].strval), 1)) in_comment = 1; }
#line 4502 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 364:
#line 1292 "xi-grammar.y" /* yacc.c:1646  */
    { if (!macroDefined((yyvsp[0].strval), 0)) in_comment = 1; }
#line 4508 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;


#line 4512 "xi-grammar.tab.C" /* yacc.c:1646  */
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
#line 1295 "xi-grammar.y" /* yacc.c:1906  */


std::string _get_caret_line(int err_line_start, int first_col, int last_col)
{
  std::string caret_line(first_col - err_line_start - 1, ' ');
  caret_line += std::string(last_col - first_col + 1, '^');

  return caret_line;
}

void _pretty_header(std::string type, std::string msg, int first_col, int last_col, int first_line, int last_line)
{
  std::cerr << cur_file << ":" << first_line << ":";

  if (first_col != -1)
    std::cerr << first_col << "-" << last_col << ": ";

  std::cerr << type << ": " << msg << std::endl;
}

void _pretty_print(std::string type, std::string msg, int first_col, int last_col, int first_line, int last_line)
{
  _pretty_header(type, msg, first_col, last_col, first_line, last_line);

  std::string err_line = inputBuffer[first_line-1];

  if (err_line.length() != 0) {
    int err_line_start = err_line.find_first_not_of(" \t\r\n");
    err_line.erase(0, err_line_start);

    std::string caret_line;
    if (first_col != -1)
      caret_line = _get_caret_line(err_line_start, first_col, last_col);

    std::cerr << "  " << err_line << std::endl;

    if (first_col != -1)
      std::cerr << "  " << caret_line;
    std::cerr << std::endl;
  }
}

void pretty_msg(std::string type, std::string msg, int first_col, int last_col, int first_line, int last_line)
{
  if (first_line == -1) first_line = lineno;
  if (last_line  == -1)  last_line = lineno;
  _pretty_print(type, msg, first_col, last_col, first_line, last_line);
}

void pretty_msg_noline(std::string type, std::string msg, int first_col, int last_col, int first_line, int last_line)
{
  if (first_line == -1) first_line = lineno;
  if (last_line  == -1)  last_line = lineno;
  _pretty_header(type, msg, first_col, last_col, first_line, last_line);
}

void yyerror(const char *msg) { }
