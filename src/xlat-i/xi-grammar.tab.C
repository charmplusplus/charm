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
#define YYLAST   1248

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  89
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  114
/* YYNRULES -- Number of rules.  */
#define YYNRULES  352
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  646

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
     438,   442,   446,   448,   452,   456,   458,   460,   462,   465,
     467,   471,   473,   477,   481,   486,   487,   491,   495,   500,
     501,   506,   507,   517,   519,   523,   525,   530,   531,   535,
     537,   542,   543,   547,   552,   553,   557,   559,   563,   565,
     570,   571,   575,   576,   579,   583,   585,   589,   591,   596,
     597,   601,   603,   607,   609,   613,   617,   621,   627,   631,
     633,   637,   639,   643,   647,   651,   655,   657,   662,   663,
     668,   669,   671,   675,   677,   679,   683,   685,   689,   693,
     695,   697,   699,   701,   705,   707,   712,   719,   723,   725,
     727,   728,   730,   732,   734,   738,   740,   742,   748,   754,
     763,   765,   767,   773,   781,   783,   786,   790,   792,   800,
     802,   807,   811,   813,   815,   817,   819,   821,   823,   825,
     827,   829,   831,   834,   844,   861,   877,   879,   883,   885,
     890,   891,   893,   900,   902,   906,   908,   910,   912,   914,
     916,   918,   920,   922,   924,   926,   928,   930,   932,   934,
     936,   938,   947,   949,   951,   956,   957,   959,   968,   969,
     971,   977,   983,   989,   997,  1004,  1012,  1019,  1021,  1023,
    1025,  1032,  1033,  1034,  1037,  1038,  1039,  1040,  1047,  1053,
    1062,  1069,  1075,  1081,  1089,  1091,  1095,  1097,  1101,  1103,
    1107,  1109,  1114,  1115,  1120,  1121,  1123,  1127,  1129,  1133,
    1135,  1139,  1141,  1143,  1151,  1154,  1157,  1159,  1161,  1165,
    1167,  1169,  1171,  1173,  1175,  1179,  1181,  1183,  1185,  1187,
    1189,  1191,  1194,  1197,  1199,  1201,  1203,  1205,  1207,  1218,
    1219,  1221,  1225,  1229,  1233,  1237,  1239,  1243,  1245,  1249,
    1252,  1256,  1260
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
  "FuncType", "BaseType", "Type", "ArrayDim", "Dim", "DimList", "Readonly",
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
     325,   326,   327,    59,    58,   123,   125,    44,    60,    62,
      42,    40,    41,    38,    91,    93,    61,    45,    46
};
# endif

#define YYPACT_NINF -514

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-514)))

#define YYTABLE_NINF -310

#define yytable_value_is_error(Yytable_value) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
     157,  1118,  1118,    27,  -514,   157,  -514,  -514,  -514,  -514,
    -514,  -514,  -514,  -514,  -514,  -514,  -514,  -514,  -514,  -514,
    -514,  -514,  -514,  -514,  -514,  -514,  -514,  -514,  -514,  -514,
    -514,  -514,  -514,  -514,  -514,  -514,  -514,  -514,  -514,  -514,
    -514,  -514,  -514,  -514,  -514,  -514,  -514,  -514,  -514,  -514,
    -514,  -514,  -514,   145,   145,  -514,  -514,  -514,   349,  -514,
    -514,  -514,    -4,  1118,   172,  1118,  1118,   160,   702,   -27,
      44,   349,  -514,  -514,  -514,   427,    18,     7,  -514,    40,
    -514,  -514,  -514,  -514,   -41,   620,    99,    99,   -15,     7,
      62,    62,    62,    62,   100,   124,  1118,   156,   183,   349,
    -514,  -514,  -514,  -514,  -514,  -514,  -514,  -514,   597,  -514,
    -514,  -514,  -514,   161,  -514,  -514,  -514,  -514,  -514,  -514,
    -514,  -514,  -514,  -514,  -514,  -514,  -514,  -514,   169,  -514,
      83,  -514,  -514,  -514,  -514,   336,    96,  -514,  -514,   190,
    -514,     7,   349,    40,   200,   101,   -41,   205,  1183,  -514,
     178,   190,   212,   214,  -514,    30,     7,  -514,     7,     7,
     217,     7,   208,  -514,    53,  1118,  1118,  1118,  1118,   908,
     211,   215,   134,  1118,  -514,  -514,  -514,   448,   228,    62,
      62,    62,    62,   211,   124,  -514,  -514,  -514,  -514,  -514,
    -514,  -514,  -514,  -514,  -514,   170,  -514,  -514,  1170,  -514,
    -514,  1118,   232,   245,   -41,   244,   -41,   236,  -514,   248,
     246,   -13,  -514,  -514,  -514,   250,  -514,    85,   -43,    60,
     239,    97,     7,  -514,  -514,   249,   259,   262,   274,   274,
     274,   274,  -514,  1118,   264,   289,   266,   978,  1118,   321,
    1118,  -514,  -514,   287,   296,   303,  1118,   194,  1118,   307,
     306,   161,  1118,  1118,  1118,  1118,  1118,  1118,  -514,  -514,
    -514,  -514,   322,  -514,   323,  -514,   262,  -514,  -514,   328,
     342,   340,   338,   -41,  -514,     7,  1118,  -514,   343,  -514,
     -41,    99,  1170,    99,    99,  1170,    99,  -514,  -514,    53,
    -514,     7,   171,   171,   171,   171,   341,  -514,   321,  -514,
     274,   274,  -514,   134,   411,   346,   224,  -514,   356,   448,
    -514,  -514,   274,   274,   274,   274,   274,   175,  1170,  -514,
     350,   -41,   244,   -41,   -41,  -514,    85,   363,  -514,   370,
    -514,   374,   378,   376,     7,   381,   379,  -514,   399,  -514,
    -514,  1186,  -514,  -514,  -514,  -514,  -514,  -514,   171,   171,
    -514,  -514,   178,    -3,   401,   178,  -514,  -514,  -514,  -514,
    -514,   171,   171,   171,   171,   171,  -514,   411,  -514,  1138,
    -514,  -514,  -514,  -514,  -514,  -514,   397,  -514,  -514,  -514,
     398,  -514,   131,   412,  -514,     7,   514,   442,   418,  -514,
    1186,   751,  -514,  -514,  -514,  1118,  -514,  -514,  -514,  -514,
    -514,  -514,  -514,  -514,   419,  -514,  1118,   -41,   420,   414,
     178,    99,    99,    99,  -514,  -514,   718,   838,  -514,   161,
    -514,  -514,  -514,   421,   429,     6,   431,   178,  -514,   422,
     432,   438,   449,  -514,  -514,  -514,  -514,  -514,  -514,  -514,
    -514,  -514,  -514,  -514,  -514,  -514,   469,  -514,   447,  -514,
    -514,   466,   486,   487,   350,  1118,  -514,   484,   497,  -514,
    -514,   135,  -514,  -514,  -514,  -514,  -514,  -514,  -514,  -514,
    -514,   535,  -514,   769,   368,   350,  -514,  -514,  -514,  -514,
      40,  -514,  1118,  -514,  -514,   491,   498,   491,   534,   513,
     538,   491,   515,   111,   -41,  -514,  -514,  -514,   574,   350,
    -514,   -41,   542,   -41,   126,   522,   428,   580,  -514,   525,
     -41,   406,   526,   345,   205,   517,   368,   521,  -514,   528,
     531,   524,  -514,   -41,   534,   221,  -514,   543,   392,   -41,
     524,  -514,  -514,  -514,  -514,  -514,  -514,   544,   406,  -514,
    -514,  -514,  -514,   599,  -514,   252,   525,   -41,   491,  -514,
     588,   370,  -514,  -514,   578,  -514,  -514,   205,   596,  -514,
    -514,  -514,  -514,  -514,  -514,  -514,  1118,   585,   584,   577,
     -41,   591,   -41,   111,  -514,   350,  -514,  -514,   111,   617,
     589,   178,   770,  -514,   205,   -41,   592,   593,  -514,   594,
     603,  -514,  1118,  1118,   -41,   598,  -514,  1118,   524,   -41,
    -514,   617,   111,  -514,  -514,   140,    31,   587,  1118,  -514,
     650,   600,  -514,   610,  -514,  1118,  1048,   595,  1118,  1118,
    -514,   203,   111,  -514,   -41,  -514,   241,   602,   254,  1118,
    -514,   222,  -514,   612,   524,  -514,  -514,  -514,  -514,  -514,
    -514,   657,   111,  -514,   613,  -514
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
     351,   352,   237,   275,   268,     0,   129,   129,   129,     0,
     137,   137,   137,   137,     0,   131,     0,     0,     0,     0,
      73,   198,   199,    67,    74,    75,    76,    77,     0,    78,
      66,   201,   200,     7,   232,   224,   225,   226,   227,   228,
     230,   231,   229,   222,    71,   223,    72,    63,   238,    92,
      93,    94,    95,   103,   104,     0,    90,   109,   110,     0,
     239,     0,     0,    64,     0,   269,   268,     0,     0,   106,
       0,   115,   116,   117,   118,   122,     0,   130,     0,     0,
       0,     0,   214,   202,     0,     0,     0,     0,     0,     0,
       0,   144,     0,     0,   204,   216,   203,     0,     0,   137,
     137,   137,   137,     0,   131,   189,   190,   191,   192,   193,
       8,    61,   102,   105,    96,    97,   100,   101,    88,   108,
     111,     0,     0,     0,   268,   265,   268,     0,   276,     0,
       0,   119,   112,   113,   120,     0,   121,   125,   208,   205,
       0,   210,     0,   148,   149,     0,   139,    90,   159,   159,
     159,   159,   143,     0,     0,   146,     0,     0,     0,     0,
       0,   135,   136,     0,   133,   157,     0,   118,     0,   186,
       0,     7,     0,     0,     0,     0,     0,     0,    98,    99,
      84,    85,    86,    89,     0,    83,    90,    70,    57,     0,
     266,     0,     0,   268,   236,     0,     0,   349,   125,   127,
     268,   129,     0,   129,   129,     0,   129,   215,   138,     0,
     107,     0,     0,     0,     0,     0,     0,   168,     0,   145,
     159,   159,   132,     0,   150,   178,     0,   184,   180,     0,
     188,    69,   159,   159,   159,   159,   159,     0,     0,    91,
       0,   268,   265,   268,   268,   273,   125,     0,   126,     0,
     123,     0,     0,     0,     0,     0,     0,   140,   161,   160,
     194,   196,   163,   164,   165,   166,   167,   147,     0,     0,
     134,   151,     0,   150,     0,     0,   183,   181,   182,   185,
     187,     0,     0,     0,     0,     0,   176,   150,    87,     0,
      68,   271,   267,   272,   270,   128,     0,   350,   124,   209,
       0,   206,     0,     0,   211,     0,     0,     0,     0,   221,
     196,     0,   219,   169,   170,     0,   156,   158,   179,   171,
     172,   173,   174,   175,     0,   299,   277,   268,   294,     0,
       0,   129,   129,   129,   162,   242,     0,     0,   220,     7,
     197,   217,   218,   152,     0,   150,     0,     0,   298,     0,
       0,     0,     0,   261,   245,   246,   247,   248,   254,   255,
     256,   249,   250,   251,   252,   253,   141,   257,     0,   259,
     260,     0,   243,    56,     0,     0,   195,     0,     0,   177,
     274,     0,   278,   280,   295,   114,   207,   213,   212,   142,
     258,     0,   241,     0,     0,     0,   153,   154,   263,   262,
     264,   279,     0,   244,   338,     0,     0,     0,     0,     0,
     315,     0,     0,     0,   268,   234,   327,   305,   302,     0,
     344,   268,     0,   268,     0,   347,     0,     0,   314,     0,
     268,     0,     0,     0,     0,     0,     0,     0,   342,     0,
       0,     0,   345,   268,     0,     0,   317,     0,     0,   268,
       0,   321,   322,   324,   320,   319,   323,     0,   311,   313,
     306,   308,   337,     0,   233,     0,     0,   268,     0,   343,
       0,     0,   348,   316,     0,   326,   310,     0,     0,   328,
     312,   303,   281,   282,   283,   301,     0,     0,   296,     0,
     268,     0,   268,     0,   335,     0,   318,   325,     0,   339,
       0,     0,     0,   300,     0,   268,     0,     0,   346,     0,
       0,   333,     0,     0,   268,     0,   297,     0,     0,   268,
     336,   339,     0,   340,   284,     0,     0,     0,     0,   235,
       0,     0,   334,     0,   292,     0,     0,     0,     0,     0,
     290,     0,     0,   330,   268,   341,     0,     0,     0,     0,
     286,     0,   293,     0,     0,   289,   288,   287,   285,   291,
     329,     0,     0,   331,     0,   332
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -514,  -514,   685,  -514,  -242,    -1,   -58,   639,   669,   -53,
    -514,  -514,  -514,  -211,  -514,  -180,  -514,  -134,   -78,   -72,
     -70,  -514,  -164,   576,   -83,  -514,  -514,  -249,  -514,  -514,
     -80,   546,   424,  -514,   -50,   439,  -514,  -514,   560,   435,
    -514,   309,  -514,  -514,  -245,  -514,  -150,   351,  -514,  -514,
    -514,   -47,  -514,  -514,  -514,  -514,  -514,  -514,  -514,   430,
    -514,   426,   672,  -514,  -168,   347,   679,  -514,  -514,   532,
    -514,  -514,  -514,   357,   369,  -514,   344,  -514,   282,  -514,
    -514,   450,   -64,   188,   -63,  -466,  -514,  -514,  -417,  -514,
    -514,  -294,   189,  -438,  -514,  -514,   257,  -503,  -514,   237,
    -514,  -433,  -514,  -462,   173,  -475,  -513,  -461,  -514,   255,
     276,   227,  -514,  -514
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     3,     4,    68,   191,   227,   136,     5,    59,    69,
      70,    71,   262,   263,   264,   199,   137,   228,   138,   151,
     152,   153,   154,   155,   265,   329,   278,   279,   101,   102,
     158,   173,   243,   244,   165,   225,   470,   235,   170,   236,
     226,   352,   458,   353,   354,   103,   292,   339,   104,   105,
     106,   171,   107,   185,   186,   187,   188,   189,   356,   307,
     249,   250,   387,   109,   342,   388,   389,   111,   112,   163,
     176,   390,   391,   126,   392,    72,   141,   417,   451,   452,
     481,   271,   518,   407,   494,   209,   408,   567,   605,   595,
     568,   409,   569,   370,   546,   516,   495,   512,   527,   537,
     509,   496,   539,   513,   591,   519,   550,   501,   505,   506,
     280,   378,    73,    74
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[] =
{
      53,    54,   156,   139,    84,   140,    79,   159,   161,   311,
     541,   162,   497,   247,   351,   145,   474,   558,   127,   143,
     147,   157,   554,   351,   214,   556,   503,    55,   521,   328,
     510,   144,   229,   230,   231,   530,    75,   498,   281,   245,
     146,   166,   167,   168,   526,   528,   178,   290,   542,   113,
     114,   115,   116,   117,   497,   118,   119,   120,   121,   122,
      78,   517,    76,   201,    80,    81,   522,   214,   215,   160,
     587,   332,   571,  -155,   335,   589,   210,   375,   538,   293,
     294,   295,   207,   223,   123,   610,   320,   572,   574,   202,
     460,   577,   461,   142,   248,   174,   579,   586,   217,   613,
     218,   219,   224,   221,   300,   538,   301,   368,   396,   619,
     598,   215,   484,   216,   144,   460,   429,   124,   597,   633,
     125,   641,   404,   238,   611,   343,   344,   345,   603,   252,
     253,   254,   255,   464,   144,   157,   256,   588,   282,   644,
     269,   283,   272,   192,   162,   247,   164,   193,   623,   634,
     348,   349,   485,   486,   487,   488,   489,   490,   491,   144,
       1,     2,   361,   362,   363,   364,   365,  -275,   234,   277,
     144,   144,   241,   242,   198,   285,   204,   456,   286,   643,
     393,   394,   205,   492,   169,   206,    83,  -275,    78,   478,
     479,   621,  -275,   399,   400,   401,   402,   403,   626,   628,
     266,   331,   631,   333,   334,   144,   336,   369,   172,   325,
     277,   175,   412,   338,   149,   150,   330,   326,    57,   614,
      58,   615,   484,    77,   616,    78,   248,   617,   618,   258,
     259,    78,   296,    82,   190,    83,   234,   129,   130,   131,
     132,   133,   134,   135,   340,   305,   341,   308,   366,  -106,
     367,   312,   313,   314,   315,   316,   317,   371,   562,   373,
     374,   177,   485,   486,   487,   488,   489,   490,   491,   395,
     200,  -180,   398,  -180,   203,   327,   382,  -275,   357,   358,
     306,   208,   632,   220,   615,   222,   406,   616,   149,   150,
     617,   618,   212,   492,   213,   237,    83,   553,   268,   239,
     270,   639,  -275,   615,   251,    78,   616,   338,   267,   617,
     618,   129,   130,   131,   132,   133,   134,   135,   273,   563,
     564,   274,   615,   635,   284,   616,   275,   406,   617,   618,
     276,   430,   431,   432,   288,   615,   289,   565,   616,   637,
     198,   617,   618,   426,   406,   139,   484,   140,   291,   297,
      61,   299,    -5,    -5,    62,    -5,    -5,    -5,    -5,    -5,
      -5,    -5,    -5,    -5,    -5,    -5,   298,    -5,    -5,   484,
     232,    -5,   302,   303,  -304,  -304,  -304,  -304,   304,  -304,
    -304,  -304,  -304,  -304,   309,   310,   485,   486,   487,   488,
     489,   490,   491,   484,   423,   194,   195,   196,   197,   318,
      63,    64,   319,   480,   321,   425,    65,    66,  -304,   485,
     486,   487,   488,   489,   490,   491,   454,   492,    67,   322,
      83,  -307,   323,   324,    -5,   -62,   346,   277,   351,   484,
     514,   369,   355,   485,   486,   487,   488,   489,   490,   491,
     492,  -304,   306,   493,  -304,   376,   529,   531,   532,   533,
     488,   534,   535,   536,   475,   377,   379,   380,   381,   551,
     383,   384,   566,   128,   492,   557,   246,    83,  -309,   485,
     486,   487,   488,   489,   490,   491,   385,   397,   410,   411,
      78,   499,   386,   570,   149,   150,   129,   130,   131,   132,
     133,   134,   135,   413,   419,   424,   428,   427,   593,   566,
     492,    78,   459,   525,   465,   457,   584,   129,   130,   131,
     132,   133,   134,   135,   466,   415,   463,  -240,  -240,  -240,
     467,  -240,  -240,  -240,   469,  -240,  -240,  -240,  -240,  -240,
     607,   468,   471,  -240,  -240,  -240,  -240,  -240,  -240,  -240,
    -240,  -240,  -240,  -240,  -240,  -240,  -240,  -240,  -240,  -240,
    -240,   472,  -240,  -240,  -240,  -240,  -240,  -240,  -240,  -240,
    -240,  -240,  -240,   473,  -240,   580,  -240,  -240,    -9,   476,
     477,   482,   500,  -240,  -240,  -240,  -240,  -240,  -240,  -240,
    -240,   484,   502,  -240,  -240,  -240,  -240,   504,   507,   484,
     511,   604,   606,   508,   515,   520,   609,   484,   416,   524,
      83,   547,   540,   543,   484,   545,   549,   604,   179,   180,
     181,   182,   183,   184,   604,   604,   548,   630,   604,   555,
     559,   485,   486,   487,   488,   489,   490,   491,   638,   485,
     486,   487,   488,   489,   490,   491,   148,   485,   486,   487,
     488,   489,   490,   491,   485,   486,   487,   488,   489,   490,
     491,   484,   492,   561,   576,    83,   149,   150,   484,   581,
     492,   582,   583,   573,   585,   590,   599,   592,   492,   600,
     601,   578,   620,    78,   629,   492,   608,   624,   602,   129,
     130,   131,   132,   133,   134,   135,   625,   636,   640,   645,
      56,   485,   486,   487,   488,   489,   490,   491,   485,   486,
     487,   488,   489,   490,   491,     1,     2,   100,    85,    86,
      87,    88,    89,    90,    91,    92,    93,    94,    95,   433,
      96,    97,   492,    60,    98,   622,   211,   350,   337,   492,
     257,   240,   642,   347,   462,   360,   414,   420,   359,   434,
     108,   435,   436,   437,   438,   439,   440,   110,   422,   441,
     442,   443,   444,   445,   287,   483,   418,   114,   115,   116,
     117,   455,   118,   119,   120,   121,   122,   446,   447,   594,
     433,   596,   372,   544,   612,   560,   562,    99,   575,   552,
     523,     0,     0,     0,   448,     0,     0,     0,   449,   450,
     434,   123,   435,   436,   437,   438,   439,   440,     0,     0,
     441,   442,   443,   444,   445,     0,   149,   150,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   446,   447,
       0,     0,     0,    78,   421,     0,     0,   125,     0,   129,
     130,   131,   132,   133,   134,   135,     0,   563,   564,   449,
     450,     6,     7,     8,     0,     9,    10,    11,     0,    12,
      13,    14,    15,    16,     0,     0,     0,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    31,    32,    33,   128,     0,    34,    35,    36,    37,
      38,    39,    40,    41,    42,    43,    44,     0,    45,     0,
      46,   453,     0,     0,     0,     0,     0,   129,   130,   131,
     132,   133,   134,   135,    48,     0,     0,    49,    50,    51,
      52,     6,     7,     8,     0,     9,    10,    11,     0,    12,
      13,    14,    15,    16,     0,     0,     0,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    31,    32,    33,     0,     0,    34,    35,    36,    37,
      38,    39,    40,    41,    42,    43,    44,   232,    45,     0,
      46,    47,   233,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    48,     0,     0,    49,    50,    51,
      52,     6,     7,     8,     0,     9,    10,    11,     0,    12,
      13,    14,    15,    16,     0,     0,     0,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    31,    32,    33,     0,     0,    34,    35,    36,    37,
      38,    39,    40,    41,    42,    43,    44,     0,    45,     0,
      46,    47,   233,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    48,     0,     0,    49,    50,    51,
      52,     6,     7,     8,     0,     9,    10,    11,     0,    12,
      13,    14,    15,    16,     0,     0,     0,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    31,    32,    33,     0,     0,    34,    35,    36,    37,
      38,    39,    40,    41,    42,    43,    44,     0,    45,     0,
      46,    47,   627,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    48,     0,     0,    49,    50,    51,
      52,     6,     7,     8,     0,     9,    10,    11,     0,    12,
      13,    14,    15,    16,     0,     0,     0,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    31,    32,    33,     0,     0,    34,    35,    36,    37,
      38,    39,    40,    41,    42,    43,    44,     0,    45,     0,
      46,    47,     0,     0,   149,   150,     0,     0,     0,     0,
       0,     0,     0,     0,    48,     0,     0,    49,    50,    51,
      52,    78,    85,    86,    87,    88,    89,   129,   130,   131,
     132,   133,   134,   135,    96,    97,   149,   150,    98,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   149,
     405,     0,     0,    78,   260,   261,   386,     0,     0,   129,
     130,   131,   132,   133,   134,   135,    78,     0,     0,     0,
       0,     0,   129,   130,   131,   132,   133,   134,   135
};

static const yytype_int16 yycheck[] =
{
       1,     2,    85,    75,    67,    75,    64,    87,    88,   251,
     513,    89,   474,   177,    17,    56,   454,   530,    71,    77,
      84,    36,   525,    17,    37,   528,   487,     0,   503,   278,
     491,    74,   166,   167,   168,   510,    40,   475,    81,   173,
      81,    91,    92,    93,   506,   507,    99,   227,   514,    76,
       6,     7,     8,     9,   516,    11,    12,    13,    14,    15,
      53,   499,    63,   141,    65,    66,   504,    37,    81,    84,
     573,   282,   547,    76,   285,   578,   148,   326,   511,   229,
     230,   231,   146,    30,    40,   598,   266,   548,   550,   142,
      84,   557,    86,    75,   177,    96,   558,   572,   156,   602,
     158,   159,    49,   161,   238,   538,   240,   318,   353,    78,
     585,    81,     1,    83,    74,    84,   410,    73,   584,   622,
      76,   634,   367,   170,   599,   293,   294,   295,   590,   179,
     180,   181,   182,   427,    74,    36,   183,   575,    78,   642,
     204,    81,   206,    60,   222,   309,    84,    64,   610,   624,
     300,   301,    41,    42,    43,    44,    45,    46,    47,    74,
       3,     4,   312,   313,   314,   315,   316,    56,   169,    84,
      74,    74,    38,    39,    78,    78,    75,   419,    81,   641,
     348,   349,    81,    72,    84,    84,    75,    76,    53,    54,
      55,   608,    81,   361,   362,   363,   364,   365,   615,   616,
     201,   281,   619,   283,   284,    74,   286,    81,    84,   273,
      84,    55,    81,   291,    36,    37,   280,   275,    73,    79,
      75,    81,     1,    51,    84,    53,   309,    87,    88,    59,
      60,    53,   233,    73,    73,    75,   237,    59,    60,    61,
      62,    63,    64,    65,    73,   246,    75,   248,    73,    80,
      75,   252,   253,   254,   255,   256,   257,   321,     6,   323,
     324,    78,    41,    42,    43,    44,    45,    46,    47,   352,
      80,    77,   355,    79,    74,   276,   334,    56,    54,    55,
      86,    76,    79,    66,    81,    77,   369,    84,    36,    37,
      87,    88,    80,    72,    80,    84,    75,    76,    53,    84,
      56,    79,    81,    81,    76,    53,    84,   385,    76,    87,
      88,    59,    60,    61,    62,    63,    64,    65,    82,    67,
      68,    73,    81,    82,    85,    84,    80,   410,    87,    88,
      80,   411,   412,   413,    85,    81,    77,    85,    84,    85,
      78,    87,    88,   407,   427,   417,     1,   417,    74,    85,
       1,    85,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    77,    18,    19,     1,
      49,    22,    85,    77,     6,     7,     8,     9,    75,    11,
      12,    13,    14,    15,    77,    79,    41,    42,    43,    44,
      45,    46,    47,     1,   395,    59,    60,    61,    62,    77,
      51,    52,    79,   461,    76,   406,    57,    58,    40,    41,
      42,    43,    44,    45,    46,    47,   417,    72,    69,    77,
      75,    76,    82,    85,    75,    76,    85,    84,    17,     1,
     494,    81,    86,    41,    42,    43,    44,    45,    46,    47,
      72,    73,    86,    75,    76,    82,   509,    41,    42,    43,
      44,    45,    46,    47,   455,    85,    82,    79,    82,   523,
      79,    82,   545,    36,    72,   529,    18,    75,    76,    41,
      42,    43,    44,    45,    46,    47,    77,    76,    81,    81,
      53,   482,    40,   546,    36,    37,    59,    60,    61,    62,
      63,    64,    65,    81,    76,    76,    82,    77,   581,   582,
      72,    53,    73,    75,    82,    84,   570,    59,    60,    61,
      62,    63,    64,    65,    82,     1,    85,     3,     4,     5,
      82,     7,     8,     9,    55,    11,    12,    13,    14,    15,
     594,    82,    85,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    31,    32,    33,    34,    35,
      36,    85,    38,    39,    40,    41,    42,    43,    44,    45,
      46,    47,    48,    77,    50,   566,    52,    53,    81,    85,
      73,    36,    81,    59,    60,    61,    62,    63,    64,    65,
      66,     1,    84,    69,    70,    71,    72,    53,    75,     1,
      75,   592,   593,    55,    20,    53,   597,     1,    84,    77,
      75,    73,    76,    86,     1,    84,    82,   608,    11,    12,
      13,    14,    15,    16,   615,   616,    85,   618,   619,    76,
      76,    41,    42,    43,    44,    45,    46,    47,   629,    41,
      42,    43,    44,    45,    46,    47,    16,    41,    42,    43,
      44,    45,    46,    47,    41,    42,    43,    44,    45,    46,
      47,     1,    72,    54,    76,    75,    36,    37,     1,    74,
      72,    77,    85,    75,    73,    48,    74,    78,    72,    76,
      76,    75,    85,    53,    79,    72,    78,    77,    75,    59,
      60,    61,    62,    63,    64,    65,    76,    85,    76,    76,
       5,    41,    42,    43,    44,    45,    46,    47,    41,    42,
      43,    44,    45,    46,    47,     3,     4,    68,     6,     7,
       8,     9,    10,    11,    12,    13,    14,    15,    16,     1,
      18,    19,    72,    54,    22,    75,   150,   303,   289,    72,
     184,   171,    75,   298,   425,   309,   385,   390,   308,    21,
      68,    23,    24,    25,    26,    27,    28,    68,   391,    31,
      32,    33,    34,    35,   222,   473,   387,     6,     7,     8,
       9,   417,    11,    12,    13,    14,    15,    49,    50,   581,
       1,   582,   322,   516,   601,   538,     6,    75,   551,   524,
     504,    -1,    -1,    -1,    66,    -1,    -1,    -1,    70,    71,
      21,    40,    23,    24,    25,    26,    27,    28,    -1,    -1,
      31,    32,    33,    34,    35,    -1,    36,    37,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    49,    50,
      -1,    -1,    -1,    53,    73,    -1,    -1,    76,    -1,    59,
      60,    61,    62,    63,    64,    65,    -1,    67,    68,    70,
      71,     3,     4,     5,    -1,     7,     8,     9,    -1,    11,
      12,    13,    14,    15,    -1,    -1,    -1,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    34,    35,    36,    -1,    38,    39,    40,    41,
      42,    43,    44,    45,    46,    47,    48,    -1,    50,    -1,
      52,    53,    -1,    -1,    -1,    -1,    -1,    59,    60,    61,
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
      32,    33,    34,    35,    -1,    -1,    38,    39,    40,    41,
      42,    43,    44,    45,    46,    47,    48,    -1,    50,    -1,
      52,    53,    -1,    -1,    36,    37,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    66,    -1,    -1,    69,    70,    71,
      72,    53,     6,     7,     8,     9,    10,    59,    60,    61,
      62,    63,    64,    65,    18,    19,    36,    37,    22,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    36,
      82,    -1,    -1,    53,    54,    55,    40,    -1,    -1,    59,
      60,    61,    62,    63,    64,    65,    53,    -1,    -1,    -1,
      -1,    -1,    59,    60,    61,    62,    63,    64,    65
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
      99,   100,   164,   201,   202,    40,    94,    51,    53,    95,
      94,    94,    73,    75,   173,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    18,    19,    22,    75,
      96,   117,   118,   134,   137,   138,   139,   141,   151,   152,
     155,   156,   157,    76,     6,     7,     8,     9,    11,    12,
      13,    14,    15,    40,    73,    76,   162,    98,    36,    59,
      60,    61,    62,    63,    64,    65,    95,   105,   107,   108,
     109,   165,    75,    95,    74,    56,    81,   171,    16,    36,
      37,   108,   109,   110,   111,   112,   113,    36,   119,   119,
      84,   119,   107,   158,    84,   123,   123,   123,   123,    84,
     127,   140,    84,   120,    94,    55,   159,    78,    98,    11,
      12,    13,    14,    15,    16,   142,   143,   144,   145,   146,
      73,    93,    60,    64,    59,    60,    61,    62,    78,   104,
      80,   107,    98,    74,    75,    81,    84,   171,    76,   174,
     108,   112,    80,    80,    37,    81,    83,    95,    95,    95,
      66,    95,    77,    30,    49,   124,   129,    94,   106,   106,
     106,   106,    49,    54,    94,   126,   128,    84,   140,    84,
     127,    38,    39,   121,   122,   106,    18,   111,   113,   149,
     150,    76,   123,   123,   123,   123,   140,   120,    59,    60,
      54,    55,   101,   102,   103,   113,    94,    76,    53,   171,
      56,   170,   171,    82,    73,    80,    80,    84,   115,   116,
     199,    81,    78,    81,    85,    78,    81,   158,    85,    77,
     104,    74,   135,   135,   135,   135,    94,    85,    77,    85,
     106,   106,    85,    77,    75,    94,    86,   148,    94,    77,
      79,    93,    94,    94,    94,    94,    94,    94,    77,    79,
     104,    76,    77,    82,    85,   171,    95,    94,   116,   114,
     171,   119,   102,   119,   119,   102,   119,   124,   107,   136,
      73,    75,   153,   153,   153,   153,    85,   128,   135,   135,
     121,    17,   130,   132,   133,    86,   147,    54,    55,   148,
     150,   135,   135,   135,   135,   135,    73,    75,   102,    81,
     182,   171,   170,   171,   171,   116,    82,    85,   200,    82,
      79,    82,    95,    79,    82,    77,    40,   151,   154,   155,
     160,   161,   163,   153,   153,   113,   133,    76,   113,   153,
     153,   153,   153,   153,   133,    82,   113,   172,   175,   180,
      81,    81,    81,    81,   136,     1,    84,   166,   163,    76,
     154,    73,   162,    94,    76,    94,   171,    77,    82,   180,
     119,   119,   119,     1,    21,    23,    24,    25,    26,    27,
      28,    31,    32,    33,    34,    35,    49,    50,    66,    70,
      71,   167,   168,    53,    94,   165,    93,    84,   131,    73,
      84,    86,   130,    85,   180,    82,    82,    82,    82,    55,
     125,    85,    85,    77,   182,    94,    85,    73,    54,    55,
      95,   169,    36,   167,     1,    41,    42,    43,    44,    45,
      46,    47,    72,    75,   173,   185,   190,   192,   182,    94,
      81,   196,    84,   196,    53,   197,   198,    75,    55,   189,
     196,    75,   186,   192,   171,    20,   184,   182,   171,   194,
      53,   194,   182,   199,    77,    75,   192,   187,   192,   173,
     194,    41,    42,    43,    45,    46,    47,   188,   190,   191,
      76,   186,   174,    86,   185,    84,   183,    73,    85,    82,
     195,   171,   198,    76,   186,    76,   186,   171,   195,    76,
     188,    54,     6,    67,    68,    85,   113,   176,   179,   181,
     173,   194,   196,    75,   192,   200,    76,   174,    75,   192,
      94,    74,    77,    85,   171,    73,   194,   186,   182,   186,
      48,   193,    78,   113,   172,   178,   181,   174,   194,    74,
      76,    76,    75,   192,    94,   177,    94,   171,    78,    94,
     195,   194,   193,   186,    79,    81,    84,    87,    88,    78,
      85,   177,    75,   192,    77,    76,   177,    54,   177,    79,
      94,   177,    79,   186,   194,    82,    85,    85,    94,    79,
      76,   195,    75,   192,   186,    76
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
     112,   113,   113,   114,   115,   116,   116,   117,   118,   119,
     119,   120,   120,   121,   121,   122,   122,   123,   123,   124,
     124,   125,   125,   126,   127,   127,   128,   128,   129,   129,
     130,   130,   131,   131,   132,   133,   133,   134,   134,   135,
     135,   136,   136,   137,   137,   138,   139,   140,   140,   141,
     141,   142,   142,   143,   144,   145,   146,   146,   147,   147,
     148,   148,   148,   149,   149,   149,   150,   150,   151,   152,
     152,   152,   152,   152,   153,   153,   154,   154,   155,   155,
     155,   155,   155,   155,   155,   156,   156,   156,   156,   156,
     157,   157,   157,   157,   158,   158,   159,   160,   160,   161,
     161,   161,   162,   162,   162,   162,   162,   162,   162,   162,
     162,   162,   162,   163,   163,   163,   164,   164,   165,   165,
     166,   166,   166,   167,   167,   168,   168,   168,   168,   168,
     168,   168,   168,   168,   168,   168,   168,   168,   168,   168,
     168,   168,   169,   169,   169,   170,   170,   170,   171,   171,
     171,   171,   171,   171,   172,   173,   174,   175,   175,   175,
     175,   176,   176,   176,   177,   177,   177,   177,   177,   177,
     178,   179,   179,   179,   180,   180,   181,   181,   182,   182,
     183,   183,   184,   184,   185,   185,   185,   186,   186,   187,
     187,   188,   188,   188,   189,   189,   190,   190,   190,   191,
     191,   191,   191,   191,   191,   192,   192,   192,   192,   192,
     192,   192,   192,   192,   192,   192,   192,   192,   192,   193,
     193,   193,   194,   195,   196,   197,   197,   198,   198,   199,
     200,   201,   202
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
       2,     2,     1,     1,     3,     0,     2,     4,     6,     0,
       1,     0,     3,     1,     3,     1,     1,     0,     3,     1,
       3,     0,     1,     1,     0,     3,     1,     3,     1,     1,
       0,     1,     0,     2,     5,     1,     2,     3,     6,     0,
       2,     1,     3,     5,     5,     5,     5,     4,     3,     6,
       6,     5,     5,     5,     5,     5,     4,     7,     0,     2,
       0,     2,     2,     3,     2,     3,     1,     3,     4,     2,
       2,     2,     2,     2,     1,     4,     0,     2,     1,     1,
       1,     1,     2,     2,     2,     3,     6,     9,     3,     6,
       3,     6,     9,     9,     1,     3,     1,     2,     2,     1,
       2,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     7,     5,    12,     5,     2,     1,     1,
       0,     3,     1,     1,     3,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     2,     1,
       1,     1,     1,     1,     1,     0,     1,     3,     0,     1,
       5,     5,     5,     4,     3,     1,     1,     1,     3,     4,
       3,     1,     1,     1,     1,     4,     3,     4,     4,     4,
       3,     7,     5,     6,     1,     3,     1,     3,     3,     2,
       3,     2,     0,     3,     0,     1,     3,     1,     2,     1,
       2,     1,     2,     1,     1,     0,     4,     3,     5,     1,
       1,     1,     1,     1,     1,     5,     4,     1,     4,    11,
       9,    12,    14,     6,     8,     5,     7,     3,     1,     0,
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
#line 194 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.modlist) = (yyvsp[0].modlist); modlist = (yyvsp[0].modlist); }
#line 2079 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 3:
#line 198 "xi-grammar.y" /* yacc.c:1646  */
    { 
		  (yyval.modlist) = 0; 
		}
#line 2087 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 4:
#line 202 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.modlist) = new AstChildren<Module>(lineno, (yyvsp[-1].module), (yyvsp[0].modlist)); }
#line 2093 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 5:
#line 206 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 2099 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 6:
#line 208 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 2105 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 7:
#line 212 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 2111 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 8:
#line 214 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 2117 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 9:
#line 219 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 2123 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 10:
#line 220 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MODULE); YYABORT; }
#line 2129 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 11:
#line 221 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MAINMODULE); YYABORT; }
#line 2135 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 12:
#line 222 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(EXTERN); YYABORT; }
#line 2141 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 13:
#line 224 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITCALL); YYABORT; }
#line 2147 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 14:
#line 225 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITNODE); YYABORT; }
#line 2153 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 15:
#line 226 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITPROC); YYABORT; }
#line 2159 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 16:
#line 228 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CHARE); }
#line 2165 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 17:
#line 229 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MAINCHARE); }
#line 2171 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 18:
#line 230 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(GROUP); }
#line 2177 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 19:
#line 231 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NODEGROUP); }
#line 2183 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 20:
#line 232 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ARRAY); }
#line 2189 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 21:
#line 236 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INCLUDE); YYABORT; }
#line 2195 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 22:
#line 237 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(STACKSIZE); YYABORT; }
#line 2201 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 23:
#line 238 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(THREADED); YYABORT; }
#line 2207 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 24:
#line 239 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(TEMPLATE); YYABORT; }
#line 2213 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 25:
#line 240 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(SYNC); YYABORT; }
#line 2219 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 26:
#line 241 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IGET); YYABORT; }
#line 2225 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 27:
#line 242 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(EXCLUSIVE); YYABORT; }
#line 2231 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 28:
#line 243 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IMMEDIATE); YYABORT; }
#line 2237 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 29:
#line 244 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(SKIPSCHED); YYABORT; }
#line 2243 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 30:
#line 245 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INLINE); YYABORT; }
#line 2249 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 31:
#line 246 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(VIRTUAL); YYABORT; }
#line 2255 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 32:
#line 247 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MIGRATABLE); YYABORT; }
#line 2261 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 33:
#line 248 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CREATEHERE); YYABORT; }
#line 2267 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 34:
#line 249 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CREATEHOME); YYABORT; }
#line 2273 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 35:
#line 250 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOKEEP); YYABORT; }
#line 2279 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 36:
#line 251 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOTRACE); YYABORT; }
#line 2285 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 37:
#line 252 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(APPWORK); YYABORT; }
#line 2291 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 38:
#line 255 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(PACKED); YYABORT; }
#line 2297 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 39:
#line 256 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(VARSIZE); YYABORT; }
#line 2303 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 40:
#line 257 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ENTRY); YYABORT; }
#line 2309 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 41:
#line 258 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(FOR); YYABORT; }
#line 2315 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 42:
#line 259 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(FORALL); YYABORT; }
#line 2321 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 43:
#line 260 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(WHILE); YYABORT; }
#line 2327 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 44:
#line 261 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(WHEN); YYABORT; }
#line 2333 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 45:
#line 262 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(OVERLAP); YYABORT; }
#line 2339 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 46:
#line 263 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ATOMIC); YYABORT; }
#line 2345 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 47:
#line 264 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IF); YYABORT; }
#line 2351 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 48:
#line 265 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ELSE); YYABORT; }
#line 2357 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 49:
#line 267 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(LOCAL); YYABORT; }
#line 2363 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 50:
#line 269 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(USING); YYABORT; }
#line 2369 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 51:
#line 270 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ACCEL); YYABORT; }
#line 2375 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 52:
#line 273 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ACCELBLOCK); YYABORT; }
#line 2381 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 53:
#line 274 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MEMCRITICAL); YYABORT; }
#line 2387 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 54:
#line 275 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(REDUCTIONTARGET); YYABORT; }
#line 2393 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 55:
#line 276 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CASE); YYABORT; }
#line 2399 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 56:
#line 280 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 2405 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 57:
#line 282 "xi-grammar.y" /* yacc.c:1646  */
    {
		  char *tmp = new char[strlen((yyvsp[-3].strval))+strlen((yyvsp[0].strval))+3];
		  sprintf(tmp,"%s::%s", (yyvsp[-3].strval), (yyvsp[0].strval));
		  (yyval.strval) = tmp;
		}
#line 2415 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 58:
#line 290 "xi-grammar.y" /* yacc.c:1646  */
    { 
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist)); 
		}
#line 2423 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 59:
#line 294 "xi-grammar.y" /* yacc.c:1646  */
    {  
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist)); 
		    (yyval.module)->setMain();
		}
#line 2432 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 60:
#line 301 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = 0; }
#line 2438 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 61:
#line 303 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = (yyvsp[-2].conslist); }
#line 2444 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 62:
#line 307 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = 0; }
#line 2450 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 63:
#line 309 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = new ConstructList(lineno, (yyvsp[-1].construct), (yyvsp[0].conslist)); }
#line 2456 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 64:
#line 313 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new UsingScope((yyvsp[0].strval), false); }
#line 2462 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 65:
#line 315 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new UsingScope((yyvsp[0].strval), true); }
#line 2468 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 66:
#line 317 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].member)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].member); }
#line 2474 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 67:
#line 319 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].message)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].message); }
#line 2480 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 68:
#line 321 "xi-grammar.y" /* yacc.c:1646  */
    {
                  Entry *e = new Entry(lineno, 0, (yyvsp[-4].type), (yyvsp[-2].strval), (yyvsp[0].plist), 0, 0, 0);
                  int isExtern = 1;
                  e->setExtern(isExtern);
                  e->targs = (yyvsp[-1].tparlist);
                  e->label = new XStr;
                  (yyvsp[-3].ntype)->print(*e->label);
                  (yyval.construct) = e;
                }
#line 2494 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 69:
#line 333 "xi-grammar.y" /* yacc.c:1646  */
    { if((yyvsp[-2].conslist)) (yyvsp[-2].conslist)->recurse<int&>((yyvsp[-4].intval), &Construct::setExtern); (yyval.construct) = (yyvsp[-2].conslist); }
#line 2500 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 70:
#line 335 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new Scope((yyvsp[-3].strval), (yyvsp[-1].conslist)); }
#line 2506 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 71:
#line 337 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = (yyvsp[-1].construct); }
#line 2512 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 72:
#line 339 "xi-grammar.y" /* yacc.c:1646  */
    {
          ERROR("preceding construct must be semicolon terminated",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 2522 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 73:
#line 345 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].module)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].module); }
#line 2528 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 74:
#line 347 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2534 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 75:
#line 349 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2540 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 76:
#line 351 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2546 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 77:
#line 353 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2552 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 78:
#line 355 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].templat)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].templat); }
#line 2558 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 79:
#line 357 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = NULL; }
#line 2564 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 80:
#line 359 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = NULL; }
#line 2570 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 81:
#line 361 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = (yyvsp[0].accelBlock); }
#line 2576 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 82:
#line 363 "xi-grammar.y" /* yacc.c:1646  */
    {
          ERROR("invalid construct",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
#line 2586 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 83:
#line 371 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamType((yyvsp[0].type)); }
#line 2592 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 84:
#line 373 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
#line 2598 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 85:
#line 375 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
#line 2604 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 86:
#line 379 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = new TParamList((yyvsp[0].tparam)); }
#line 2610 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 87:
#line 381 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = new TParamList((yyvsp[-2].tparam), (yyvsp[0].tparlist)); }
#line 2616 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 88:
#line 385 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = 0; }
#line 2622 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 89:
#line 387 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = (yyvsp[0].tparlist); }
#line 2628 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 90:
#line 391 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = 0; }
#line 2634 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 91:
#line 393 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = (yyvsp[-1].tparlist); }
#line 2640 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 92:
#line 397 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("int"); }
#line 2646 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 93:
#line 399 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long"); }
#line 2652 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 94:
#line 401 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("short"); }
#line 2658 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 95:
#line 403 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("char"); }
#line 2664 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 96:
#line 405 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned int"); }
#line 2670 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 97:
#line 407 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long"); }
#line 2676 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 98:
#line 409 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long"); }
#line 2682 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 99:
#line 411 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long long"); }
#line 2688 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 100:
#line 413 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned short"); }
#line 2694 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 101:
#line 415 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned char"); }
#line 2700 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 102:
#line 417 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long long"); }
#line 2706 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 103:
#line 419 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("float"); }
#line 2712 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 104:
#line 421 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("double"); }
#line 2718 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 105:
#line 423 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long double"); }
#line 2724 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 106:
#line 425 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("void"); }
#line 2730 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 107:
#line 428 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ntype) = new NamedType((yyvsp[-1].strval),(yyvsp[0].tparlist)); }
#line 2736 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 108:
#line 429 "xi-grammar.y" /* yacc.c:1646  */
    { 
                    const char* basename, *scope;
                    splitScopedName((yyvsp[-1].strval), &scope, &basename);
                    (yyval.ntype) = new NamedType(basename, (yyvsp[0].tparlist), scope);
                }
#line 2746 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 109:
#line 437 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 2752 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 110:
#line 439 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ntype); }
#line 2758 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 111:
#line 443 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ptype) = new PtrType((yyvsp[-1].type)); }
#line 2764 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 112:
#line 447 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
#line 2770 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 113:
#line 449 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
#line 2776 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 114:
#line 453 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ftype) = new FuncType((yyvsp[-7].type), (yyvsp[-4].strval), (yyvsp[-1].plist)); }
#line 2782 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 115:
#line 457 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 2788 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 116:
#line 459 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 2794 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 117:
#line 461 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 2800 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 118:
#line 463 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ftype); }
#line 2806 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 119:
#line 466 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[0].type)); }
#line 2812 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 120:
#line 468 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[-1].type)); }
#line 2818 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 121:
#line 472 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ReferenceType((yyvsp[-1].type)); }
#line 2824 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 122:
#line 474 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 2830 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 123:
#line 478 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 2836 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 124:
#line 482 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = (yyvsp[-1].val); }
#line 2842 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 125:
#line 486 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.vallist) = 0; }
#line 2848 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 126:
#line 488 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.vallist) = new ValueList((yyvsp[-1].val), (yyvsp[0].vallist)); }
#line 2854 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 127:
#line 492 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].vallist)); }
#line 2860 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 128:
#line 496 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[-3].type), (yyvsp[-1].strval), (yyvsp[0].vallist), 1); }
#line 2866 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 129:
#line 500 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0;}
#line 2872 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 130:
#line 502 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0;}
#line 2878 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 131:
#line 506 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 2884 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 132:
#line 508 "xi-grammar.y" /* yacc.c:1646  */
    { 
		  /*
		  printf("Warning: Message attributes are being phased out.\n");
		  printf("Warning: Please remove them from interface files.\n");
		  */
		  (yyval.intval) = (yyvsp[-1].intval); 
		}
#line 2896 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 133:
#line 518 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[0].intval); }
#line 2902 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 134:
#line 520 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[-2].intval) | (yyvsp[0].intval); }
#line 2908 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 135:
#line 524 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 2914 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 136:
#line 526 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 2920 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 137:
#line 530 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = 0; }
#line 2926 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 138:
#line 532 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-1].cattr); }
#line 2932 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 139:
#line 536 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[0].cattr); }
#line 2938 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 140:
#line 538 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
#line 2944 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 141:
#line 542 "xi-grammar.y" /* yacc.c:1646  */
    { python_doc = NULL; (yyval.intval) = 0; }
#line 2950 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 142:
#line 544 "xi-grammar.y" /* yacc.c:1646  */
    { python_doc = (yyvsp[0].strval); (yyval.intval) = 0; }
#line 2956 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 143:
#line 548 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CPYTHON; }
#line 2962 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 144:
#line 552 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = 0; }
#line 2968 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 145:
#line 554 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-1].cattr); }
#line 2974 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 146:
#line 558 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[0].cattr); }
#line 2980 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 147:
#line 560 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
#line 2986 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 148:
#line 564 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CMIGRATABLE; }
#line 2992 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 149:
#line 566 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CPYTHON; }
#line 2998 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 150:
#line 570 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3004 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 151:
#line 572 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 3010 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 152:
#line 575 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3016 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 153:
#line 577 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 3022 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 154:
#line 580 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mv) = new MsgVar((yyvsp[-3].type), (yyvsp[-2].strval), (yyvsp[-4].intval), (yyvsp[-1].intval)); }
#line 3028 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 155:
#line 584 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mvlist) = new MsgVarList((yyvsp[0].mv)); }
#line 3034 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 156:
#line 586 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mvlist) = new MsgVarList((yyvsp[-1].mv), (yyvsp[0].mvlist)); }
#line 3040 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 157:
#line 590 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, (yyvsp[0].ntype)); }
#line 3046 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 158:
#line 592 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, (yyvsp[-3].ntype), (yyvsp[-1].mvlist)); }
#line 3052 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 159:
#line 596 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = 0; }
#line 3058 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 160:
#line 598 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = (yyvsp[0].typelist); }
#line 3064 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 161:
#line 602 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = new TypeList((yyvsp[0].ntype)); }
#line 3070 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 162:
#line 604 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = new TypeList((yyvsp[-2].ntype), (yyvsp[0].typelist)); }
#line 3076 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 163:
#line 608 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr)|Chare::CCHARE, (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3082 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 164:
#line 610 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3088 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 165:
#line 614 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3094 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 166:
#line 618 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new NodeGroup(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3100 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 167:
#line 622 "xi-grammar.y" /* yacc.c:1646  */
    {/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",(yyvsp[-2].strval));
			(yyval.ntype) = new NamedType(buf); 
		}
#line 3110 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 168:
#line 628 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ntype) = new NamedType((yyvsp[-1].strval)); }
#line 3116 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 169:
#line 632 "xi-grammar.y" /* yacc.c:1646  */
    {  (yyval.chare) = new Array(lineno, (yyvsp[-4].cattr), (yyvsp[-3].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3122 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 170:
#line 634 "xi-grammar.y" /* yacc.c:1646  */
    {  (yyval.chare) = new Array(lineno, (yyvsp[-3].cattr), (yyvsp[-4].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3128 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 171:
#line 638 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr)|Chare::CCHARE, new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist));}
#line 3134 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 172:
#line 640 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3140 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 173:
#line 644 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3146 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 174:
#line 648 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new NodeGroup( lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3152 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 175:
#line 652 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Array( lineno, 0, (yyvsp[-3].ntype), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3158 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 176:
#line 656 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-1].strval))); }
#line 3164 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 177:
#line 658 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-4].strval)), (yyvsp[-2].mvlist)); }
#line 3170 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 178:
#line 662 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = 0; }
#line 3176 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 179:
#line 664 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3182 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 180:
#line 668 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = 0; }
#line 3188 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 181:
#line 670 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3194 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 182:
#line 672 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3200 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 183:
#line 676 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TType(new NamedType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3206 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 184:
#line 678 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TFunc((yyvsp[-1].ftype), (yyvsp[0].strval)); }
#line 3212 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 185:
#line 680 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TName((yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].strval)); }
#line 3218 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 186:
#line 684 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = new TVarList((yyvsp[0].tvar)); }
#line 3224 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 187:
#line 686 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = new TVarList((yyvsp[-2].tvar), (yyvsp[0].tvarlist)); }
#line 3230 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 188:
#line 690 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = (yyvsp[-1].tvarlist); }
#line 3236 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 189:
#line 694 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3242 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 190:
#line 696 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3248 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 191:
#line 698 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3254 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 192:
#line 700 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3260 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 193:
#line 702 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].message)); (yyvsp[0].message)->setTemplate((yyval.templat)); }
#line 3266 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 194:
#line 706 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = 0; }
#line 3272 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 195:
#line 708 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = (yyvsp[-2].mbrlist); }
#line 3278 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 196:
#line 712 "xi-grammar.y" /* yacc.c:1646  */
    { 
                  if (!connectEntries.empty()) {
                    (yyval.mbrlist) = new AstChildren<Member>(connectEntries);
		  } else {
		    (yyval.mbrlist) = 0; 
                  }
		}
#line 3290 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 197:
#line 720 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = new AstChildren<Member>(-1, (yyvsp[-1].member), (yyvsp[0].mbrlist)); }
#line 3296 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 198:
#line 724 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].readonly); }
#line 3302 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 199:
#line 726 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].readonly); }
#line 3308 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 201:
#line 729 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].member); }
#line 3314 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 202:
#line 731 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].pupable); }
#line 3320 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 203:
#line 733 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].includeFile); }
#line 3326 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 204:
#line 735 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new ClassDeclaration(lineno,(yyvsp[0].strval)); }
#line 3332 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 205:
#line 739 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1); }
#line 3338 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 206:
#line 741 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1); }
#line 3344 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 207:
#line 743 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[-6].strval)) + '<' +
					    ((yyvsp[-4].tparlist))->to_string() + '>').c_str()),
				    1);
		}
#line 3354 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 208:
#line 749 "xi-grammar.y" /* yacc.c:1646  */
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[-2]).first_column, (yylsp[-2]).last_column, (yylsp[-2]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1);
		}
#line 3364 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 209:
#line 755 "xi-grammar.y" /* yacc.c:1646  */
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[-5]).first_column, (yylsp[-5]).last_column, (yylsp[-5]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1);
		}
#line 3374 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 210:
#line 764 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 0); }
#line 3380 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 211:
#line 766 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 0); }
#line 3386 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 212:
#line 768 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[-6].strval)) + '<' +
					    ((yyvsp[-4].tparlist))->to_string() + '>').c_str()),
				    0);
		}
#line 3396 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 213:
#line 774 "xi-grammar.y" /* yacc.c:1646  */
    {
                  InitCall* rtn = new InitCall(lineno, (yyvsp[-3].strval), 0);
                  rtn->setAccel();
                  (yyval.member) = rtn;
		}
#line 3406 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 214:
#line 782 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[0].ntype),0); }
#line 3412 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 215:
#line 784 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[-2].ntype),(yyvsp[0].pupable)); }
#line 3418 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 216:
#line 787 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.includeFile) = new IncludeFile(lineno,(yyvsp[0].strval)); }
#line 3424 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 217:
#line 791 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[-1].member); }
#line 3430 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 218:
#line 793 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("preceding entry method declaration must be semicolon-terminated",
		        (yylsp[0]).first_column, (yylsp[0]).last_column);
		  YYABORT;
		}
#line 3440 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 219:
#line 801 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].entry); }
#line 3446 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 220:
#line 803 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyvsp[0].entry)->tspec = (yyvsp[-1].tvarlist);
                  (yyval.member) = (yyvsp[0].entry);
                }
#line 3455 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 221:
#line 808 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].member); }
#line 3461 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 222:
#line 812 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3467 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 223:
#line 814 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3473 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 224:
#line 816 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3479 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 225:
#line 818 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3485 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 226:
#line 820 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3491 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 227:
#line 822 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3497 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 228:
#line 824 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3503 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 229:
#line 826 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3509 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 230:
#line 828 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3515 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 231:
#line 830 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3521 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 232:
#line 832 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3527 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 233:
#line 835 "xi-grammar.y" /* yacc.c:1646  */
    { 
                  (yyval.entry) = new Entry(lineno, (yyvsp[-5].intval), (yyvsp[-4].type), (yyvsp[-3].strval), (yyvsp[-2].plist), (yyvsp[-1].val), (yyvsp[0].sentry)); 
		  if ((yyvsp[0].sentry) != 0) { 
		    (yyvsp[0].sentry)->con1 = new SdagConstruct(SIDENT, (yyvsp[-3].strval));
                    (yyvsp[0].sentry)->entry = (yyval.entry);
                    (yyvsp[0].sentry)->con1->entry = (yyval.entry);
                    (yyvsp[0].sentry)->param = new ParamList((yyvsp[-2].plist));
                  }
		}
#line 3541 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 234:
#line 845 "xi-grammar.y" /* yacc.c:1646  */
    { 
                  Entry *e = new Entry(lineno, (yyvsp[-3].intval), 0, (yyvsp[-2].strval), (yyvsp[-1].plist),  0, (yyvsp[0].sentry));
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
#line 3562 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 235:
#line 862 "xi-grammar.y" /* yacc.c:1646  */
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
#line 3580 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 236:
#line 878 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.accelBlock) = new AccelBlock(lineno, new XStr((yyvsp[-2].strval))); }
#line 3586 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 237:
#line 880 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.accelBlock) = new AccelBlock(lineno, NULL); }
#line 3592 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 238:
#line 884 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("void"); }
#line 3598 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 239:
#line 886 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3604 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 240:
#line 890 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3610 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 241:
#line 892 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[-1].intval); }
#line 3616 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 242:
#line 894 "xi-grammar.y" /* yacc.c:1646  */
    { ERROR("invalid entry method attribute list",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
#line 3625 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 243:
#line 901 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[0].intval); }
#line 3631 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 244:
#line 903 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[-2].intval) | (yyvsp[0].intval); }
#line 3637 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 245:
#line 907 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = STHREADED; }
#line 3643 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 246:
#line 909 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SSYNC; }
#line 3649 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 247:
#line 911 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SIGET; }
#line 3655 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 248:
#line 913 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SLOCKED; }
#line 3661 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 249:
#line 915 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SCREATEHERE; }
#line 3667 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 250:
#line 917 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SCREATEHOME; }
#line 3673 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 251:
#line 919 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SNOKEEP; }
#line 3679 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 252:
#line 921 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SNOTRACE; }
#line 3685 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 253:
#line 923 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SAPPWORK; }
#line 3691 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 254:
#line 925 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SIMMEDIATE; }
#line 3697 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 255:
#line 927 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SSKIPSCHED; }
#line 3703 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 256:
#line 929 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SINLINE; }
#line 3709 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 257:
#line 931 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SLOCAL; }
#line 3715 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 258:
#line 933 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SPYTHON; }
#line 3721 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 259:
#line 935 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SMEM; }
#line 3727 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 260:
#line 937 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SREDUCE; }
#line 3733 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 261:
#line 939 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("invalid entry method attribute",
		        (yylsp[0]).first_column, (yylsp[0]).last_column);
		  yyclearin;
		  yyerrok;
		}
#line 3744 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 262:
#line 948 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 3750 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 263:
#line 950 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 3756 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 264:
#line 952 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 3762 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 265:
#line 956 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = ""; }
#line 3768 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 266:
#line 958 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3774 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 267:
#line 960 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s, %s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 3784 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 268:
#line 968 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = ""; }
#line 3790 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 269:
#line 970 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3796 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 270:
#line 972 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s[%s]%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 3806 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 271:
#line 978 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s{%s}%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 3816 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 272:
#line 984 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s(%s)%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 3826 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 273:
#line 990 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"(%s)%s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 3836 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 274:
#line 998 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval));
		}
#line 3845 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 275:
#line 1005 "xi-grammar.y" /* yacc.c:1646  */
    { 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			(yyval.intval) = 0;
		}
#line 3855 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 276:
#line 1013 "xi-grammar.y" /* yacc.c:1646  */
    { 
			in_braces=0;
			(yyval.intval) = 0;
		}
#line 3864 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 277:
#line 1020 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[0].type));}
#line 3870 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 278:
#line 1022 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval)); (yyval.pname)->setConditional((yyvsp[0].intval)); }
#line 3876 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 279:
#line 1024 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[-3].type),(yyvsp[-2].strval),0,(yyvsp[0].val));}
#line 3882 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 280:
#line 1026 "xi-grammar.y" /* yacc.c:1646  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
		}
#line 3891 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 281:
#line 1032 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READONLY; }
#line 3897 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 282:
#line 1033 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READWRITE; }
#line 3903 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 283:
#line 1034 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_WRITEONLY; }
#line 3909 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 284:
#line 1037 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr((yyvsp[0].strval)); }
#line 3915 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 285:
#line 1038 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "->" << (yyvsp[0].strval); }
#line 3921 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 286:
#line 1039 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[-2].xstrptr)) << "." << (yyvsp[0].strval); }
#line 3927 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 287:
#line 1041 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << *((yyvsp[-1].xstrptr)) << "]";
                  delete (yyvsp[-3].xstrptr);
                  delete (yyvsp[-1].xstrptr);
                }
#line 3938 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 288:
#line 1048 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << (yyvsp[-1].strval) << "]";
                  delete (yyvsp[-3].xstrptr);
                }
#line 3948 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 289:
#line 1054 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "(" << *((yyvsp[-1].xstrptr)) << ")";
                  delete (yyvsp[-3].xstrptr);
                  delete (yyvsp[-1].xstrptr);
                }
#line 3959 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 290:
#line 1063 "xi-grammar.y" /* yacc.c:1646  */
    {
                  in_bracket = 0;
                  (yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName(), (yyvsp[-1].strval));
                }
#line 3968 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 291:
#line 1070 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[-6].intval));
                }
#line 3978 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 292:
#line 1076 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType(Parameter::ACCEL_BUFFER_TYPE_READWRITE);
		}
#line 3988 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 293:
#line 1082 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.pname) = (yyvsp[-3].pname);
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[-5].intval));
		}
#line 3998 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 294:
#line 1090 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 4004 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 295:
#line 1092 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 4010 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 296:
#line 1096 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 4016 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 297:
#line 1098 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 4022 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 298:
#line 1102 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = (yyvsp[-1].plist); }
#line 4028 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 299:
#line 1104 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList(new Parameter(0, new BuiltinType("void"))); }
#line 4034 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 300:
#line 1108 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = (yyvsp[-1].plist); }
#line 4040 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 301:
#line 1110 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = 0; }
#line 4046 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 302:
#line 1114 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = 0; }
#line 4052 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 303:
#line 1116 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 4058 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 304:
#line 1120 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = 0; }
#line 4064 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 305:
#line 1122 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[0].sc)); }
#line 4070 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 306:
#line 1124 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[-1].slist)); }
#line 4076 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 307:
#line 1128 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.slist) = new SListConstruct((yyvsp[0].sc)); }
#line 4082 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 308:
#line 1130 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.slist) = new SListConstruct((yyvsp[-1].sc), (yyvsp[0].slist));  }
#line 4088 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 309:
#line 1134 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.olist) = new OListConstruct((yyvsp[0].sc)); }
#line 4094 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 310:
#line 1136 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.olist) = new OListConstruct((yyvsp[-1].sc), (yyvsp[0].slist)); }
#line 4100 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 311:
#line 1140 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.clist) = new CaseListConstruct((yyvsp[0].when)); }
#line 4106 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 312:
#line 1142 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.clist) = new CaseListConstruct((yyvsp[-1].when), (yyvsp[0].clist)); }
#line 4112 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 313:
#line 1144 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("case blocks can only contain when clauses",
		        (yylsp[0]).first_column, (yylsp[0]).last_column);
		  (yyval.clist) = 0;
		}
#line 4122 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 314:
#line 1152 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 4128 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 315:
#line 1154 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = 0; }
#line 4134 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 316:
#line 1158 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-2].entrylist), 0); }
#line 4140 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 317:
#line 1160 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-1].entrylist), (yyvsp[0].sc)); }
#line 4146 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 318:
#line 1162 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-3].entrylist), (yyvsp[-1].slist)); }
#line 4152 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 319:
#line 1166 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4158 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 320:
#line 1168 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4164 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 321:
#line 1170 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4170 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 322:
#line 1172 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4176 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 323:
#line 1174 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4182 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 324:
#line 1176 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4188 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 325:
#line 1180 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new AtomicConstruct((yyvsp[-1].strval), (yyvsp[-3].strval)); }
#line 4194 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 326:
#line 1182 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new OverlapConstruct((yyvsp[-1].olist)); }
#line 4200 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 327:
#line 1184 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = (yyvsp[0].when); }
#line 4206 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 328:
#line 1186 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new CaseConstruct((yyvsp[-1].clist)); }
#line 4212 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 329:
#line 1188 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForConstruct((yyvsp[-8].intexpr), (yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4218 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 330:
#line 1190 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForConstruct((yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4224 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 331:
#line 1192 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-9].strval)), (yyvsp[-6].intexpr),
		             (yyvsp[-4].intexpr), (yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4231 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 332:
#line 1195 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-11].strval)), (yyvsp[-8].intexpr),
		             (yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4238 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 333:
#line 1198 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new IfConstruct((yyvsp[-3].intexpr), (yyvsp[-1].sc), (yyvsp[0].sc)); }
#line 4244 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 334:
#line 1200 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new IfConstruct((yyvsp[-5].intexpr), (yyvsp[-2].slist), (yyvsp[0].sc)); }
#line 4250 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 335:
#line 1202 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new WhileConstruct((yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4256 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 336:
#line 1204 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new WhileConstruct((yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4262 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 337:
#line 1206 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new AtomicConstruct((yyvsp[-1].strval), NULL); }
#line 4268 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 338:
#line 1208 "xi-grammar.y" /* yacc.c:1646  */
    {
		  ERROR("unknown SDAG construct or malformed entry method definition.\n"
		        "You may have forgotten to terminate a previous entry method definition with a"
		        " semicolon or forgotten to mark a block of sequential SDAG code as 'atomic'",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
#line 4280 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 339:
#line 1218 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = 0; }
#line 4286 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 340:
#line 1220 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ElseConstruct((yyvsp[0].sc)); }
#line 4292 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 341:
#line 1222 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ElseConstruct((yyvsp[-1].slist)); }
#line 4298 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 342:
#line 1226 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intexpr) = new IntExprConstruct((yyvsp[0].strval)); }
#line 4304 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 343:
#line 1230 "xi-grammar.y" /* yacc.c:1646  */
    { in_int_expr = 0; (yyval.intval) = 0; }
#line 4310 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 344:
#line 1234 "xi-grammar.y" /* yacc.c:1646  */
    { in_int_expr = 1; (yyval.intval) = 0; }
#line 4316 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 345:
#line 1238 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[-1].strval), (yyvsp[0].plist), 0, 0, 0); }
#line 4322 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 346:
#line 1240 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[-4].strval), (yyvsp[0].plist), 0, 0, (yyvsp[-2].strval)); }
#line 4328 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 347:
#line 1244 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.entrylist) = new EntryList((yyvsp[0].entry)); }
#line 4334 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 348:
#line 1246 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.entrylist) = new EntryList((yyvsp[-2].entry),(yyvsp[0].entrylist)); }
#line 4340 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 349:
#line 1250 "xi-grammar.y" /* yacc.c:1646  */
    { in_bracket=1; }
#line 4346 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 350:
#line 1253 "xi-grammar.y" /* yacc.c:1646  */
    { in_bracket=0; }
#line 4352 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 351:
#line 1257 "xi-grammar.y" /* yacc.c:1646  */
    { if (!macroDefined((yyvsp[0].strval), 1)) in_comment = 1; }
#line 4358 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;

  case 352:
#line 1261 "xi-grammar.y" /* yacc.c:1646  */
    { if (!macroDefined((yyvsp[0].strval), 0)) in_comment = 1; }
#line 4364 "xi-grammar.tab.C" /* yacc.c:1646  */
    break;


#line 4368 "xi-grammar.tab.C" /* yacc.c:1646  */
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
#line 1264 "xi-grammar.y" /* yacc.c:1906  */


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
