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
using namespace xi;
extern int yylex (void) ;
extern unsigned char in_comment;
void yyerror(const char *);
extern unsigned int lineno;
extern int in_bracket,in_braces,in_int_expr;
extern std::list<Entry *> connectEntries;
AstChildren<Module> *modlist;
namespace xi {
extern int macroDefined(const char *str, int istrue);
extern const char *python_doc;
void splitScopedName(const char* name, const char** scope, const char** basename);
void ReservedWord(int token);
}

#line 89 "xi-grammar.tab.c" /* yacc.c:339  */

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
#line 25 "xi-grammar.y" /* yacc.c:355  */

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
  WhenConstruct *when;
  SListConstruct *slist;
  CaseListConstruct *clist;
  OListConstruct *olist;
  XStr* xstrptr;
  AccelBlock* accelBlock;

#line 241 "xi-grammar.tab.c" /* yacc.c:355  */
};

typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif


extern YYSTYPE yylval;

int yyparse (void);



/* Copy the second part of user declarations.  */

#line 258 "xi-grammar.tab.c" /* yacc.c:358  */

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
         || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yytype_int16 yyss_alloc;
  YYSTYPE yyvs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

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
#define YYLAST   1293

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  89
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  113
/* YYNRULES -- Number of rules.  */
#define YYNRULES  351
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  644

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
       0,   162,   162,   167,   170,   175,   176,   181,   182,   187,
     189,   190,   191,   193,   194,   195,   197,   198,   199,   200,
     201,   205,   206,   207,   208,   209,   210,   211,   212,   213,
     214,   215,   216,   217,   218,   219,   220,   221,   224,   225,
     226,   227,   228,   229,   230,   231,   232,   233,   234,   236,
     238,   239,   242,   243,   244,   245,   248,   250,   258,   262,
     269,   271,   276,   277,   281,   283,   285,   287,   289,   301,
     303,   305,   307,   309,   311,   313,   315,   317,   319,   321,
     323,   325,   327,   331,   333,   335,   339,   341,   346,   347,
     352,   353,   357,   359,   361,   363,   365,   367,   369,   371,
     373,   375,   377,   379,   381,   383,   385,   389,   390,   397,
     399,   403,   407,   409,   413,   417,   419,   421,   423,   426,
     428,   432,   434,   438,   442,   447,   448,   452,   456,   461,
     462,   467,   468,   478,   480,   484,   486,   491,   492,   496,
     498,   503,   504,   508,   513,   514,   518,   520,   524,   526,
     531,   532,   536,   537,   540,   544,   546,   550,   552,   557,
     558,   562,   564,   568,   570,   574,   578,   582,   588,   592,
     594,   598,   600,   604,   608,   612,   616,   618,   623,   624,
     629,   630,   632,   636,   638,   640,   644,   646,   650,   654,
     656,   658,   660,   662,   666,   668,   673,   680,   684,   686,
     688,   689,   691,   693,   695,   699,   701,   703,   709,   712,
     717,   719,   721,   727,   735,   737,   740,   744,   747,   751,
     753,   758,   762,   764,   766,   768,   770,   772,   774,   776,
     778,   780,   782,   785,   795,   810,   826,   828,   832,   834,
     839,   840,   842,   846,   848,   852,   854,   856,   858,   860,
     862,   864,   866,   868,   870,   872,   874,   876,   878,   880,
     882,   884,   888,   890,   892,   897,   898,   900,   909,   910,
     912,   918,   924,   930,   938,   945,   953,   960,   962,   964,
     966,   973,   974,   975,   978,   979,   980,   981,   988,   994,
    1003,  1010,  1016,  1022,  1030,  1032,  1036,  1038,  1042,  1044,
    1048,  1050,  1055,  1056,  1061,  1062,  1064,  1068,  1070,  1074,
    1076,  1080,  1082,  1084,  1088,  1091,  1094,  1096,  1098,  1102,
    1104,  1106,  1108,  1110,  1112,  1116,  1118,  1120,  1122,  1124,
    1126,  1128,  1131,  1134,  1136,  1138,  1140,  1142,  1144,  1151,
    1152,  1154,  1158,  1162,  1166,  1168,  1172,  1174,  1178,  1181,
    1185,  1189
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
      38,  1118,  1118,    56,  -514,    38,  -514,  -514,  -514,  -514,
    -514,  -514,  -514,  -514,  -514,  -514,  -514,  -514,  -514,  -514,
    -514,  -514,  -514,  -514,  -514,  -514,  -514,  -514,  -514,  -514,
    -514,  -514,  -514,  -514,  -514,  -514,  -514,  -514,  -514,  -514,
    -514,  -514,  -514,  -514,  -514,  -514,  -514,  -514,  -514,  -514,
    -514,  -514,  -514,   145,   145,  -514,  -514,  -514,   624,  -514,
    -514,  -514,    34,  1118,   119,  1118,  1118,   150,   717,     4,
     697,   624,  -514,  -514,  -514,   423,    15,    50,  -514,    24,
    -514,  -514,  -514,  -514,   -22,  1162,    95,    95,    -7,    50,
      55,    55,    55,    55,    60,    66,  1118,   121,   106,   624,
    -514,  -514,  -514,  -514,  -514,  -514,  -514,  -514,   246,  -514,
    -514,  -514,  -514,   124,  -514,  -514,  -514,  -514,  -514,  -514,
    -514,  -514,  -514,  -514,  -514,  -514,  -514,  -514,   123,  -514,
      53,  -514,  -514,  -514,  -514,   255,    97,  -514,  -514,   137,
    -514,    50,   624,    24,   148,    -2,   -22,   170,  1228,  -514,
    1215,   137,   187,   190,  -514,    31,    50,  -514,    50,    50,
     216,    50,   206,  -514,     3,  1118,  1118,  1118,  1118,   908,
     241,   244,    77,  1118,  -514,  -514,  -514,  1182,   236,    55,
      55,    55,    55,   241,    66,  -514,  -514,  -514,  -514,  -514,
    -514,  -514,  -514,  -514,  -514,   243,  -514,  -514,  1195,  -514,
    -514,  1118,   253,   274,   -22,   285,   -22,   248,  -514,   269,
     264,    -6,  -514,  -514,  -514,   268,  -514,   -27,   114,    11,
     266,   109,    50,  -514,  -514,   271,   281,   276,   286,   286,
     286,   286,  -514,  1118,   278,   284,   279,   978,  1118,   313,
    1118,  -514,  -514,   291,   300,   304,  1118,    88,  1118,   303,
     302,   124,  1118,  1118,  1118,  1118,  1118,  1118,  -514,  -514,
    -514,  -514,   305,  -514,   315,  -514,   276,  -514,  -514,   307,
     329,   310,   320,   -22,  -514,  1118,  1118,  -514,   330,  -514,
     -22,    95,  1195,    95,    95,  1195,    95,  -514,  -514,     3,
    -514,    50,   164,   164,   164,   164,   332,  -514,   313,  -514,
     286,   286,  -514,    77,   401,   334,   251,  -514,   336,  1182,
    -514,  -514,   286,   286,   286,   286,   286,   193,  1195,  -514,
     342,   -22,   285,   -22,   -22,  -514,  -514,   348,  -514,   339,
    -514,   349,   353,   351,    50,   355,   357,  -514,   358,  -514,
    -514,   574,  -514,  -514,  -514,  -514,  -514,  -514,   164,   164,
    -514,  -514,  1215,    10,   360,  1215,  -514,  -514,  -514,  -514,
    -514,   164,   164,   164,   164,   164,  -514,   401,  -514,   716,
    -514,  -514,  -514,  -514,  -514,   356,  -514,  -514,  -514,   361,
    -514,   120,   362,  -514,    50,   507,   404,   370,  -514,   574,
     816,  -514,  -514,  -514,  1118,  -514,  -514,  -514,  -514,  -514,
    -514,  -514,  -514,   371,  -514,  1118,   -22,   373,   369,  1215,
      95,    95,    95,  -514,  -514,   733,   838,  -514,   124,  -514,
    -514,  -514,   364,   379,     7,   372,  1215,  -514,   374,   376,
     393,   395,  -514,  -514,  -514,  -514,  -514,  -514,  -514,  -514,
    -514,  -514,  -514,  -514,  -514,   400,  -514,   405,  -514,  -514,
     406,   412,   397,   342,  1118,  -514,   408,   421,  -514,  -514,
     195,  -514,  -514,  -514,  -514,  -514,  -514,  -514,  -514,  -514,
     462,  -514,   784,   643,   342,  -514,  -514,  -514,  -514,    24,
    -514,  1118,  -514,  -514,   419,   417,   419,   450,   429,   451,
     419,   430,   110,   -22,  -514,  -514,  -514,   487,   342,  -514,
     -22,   456,   -22,   115,   436,   293,   344,  -514,   442,   -22,
     328,   447,   168,   170,   438,   643,   441,   471,   473,   474,
    -514,   -22,   450,   117,  -514,   485,   277,   -22,   474,  -514,
    -514,  -514,  -514,  -514,  -514,   486,   328,  -514,  -514,  -514,
    -514,   510,  -514,   228,   442,   -22,   419,  -514,   354,   339,
    -514,  -514,   489,  -514,  -514,   170,   366,  -514,  -514,  -514,
    -514,  -514,  -514,  -514,  1118,   500,   498,   501,   -22,   512,
     -22,   110,  -514,   342,  -514,  -514,   110,   539,   517,  1215,
    1149,  -514,   170,   -22,   514,   521,  -514,   522,   420,  -514,
    1118,  1118,   -22,   523,  -514,  1118,   474,   -22,  -514,   539,
     110,  -514,  -514,   140,   -24,   515,  1118,  -514,   427,   525,
    -514,   527,  -514,  1118,  1048,   520,  1118,  1118,  -514,   154,
     110,  -514,   -22,  -514,    45,   519,   223,  1118,  -514,   192,
    -514,   530,   474,  -514,  -514,  -514,  -514,  -514,  -514,   619,
     110,  -514,   531,  -514
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
     350,   351,   237,   275,   268,     0,   129,   129,   129,     0,
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
     266,     0,     0,   268,   236,     0,     0,   348,   125,   127,
     268,   129,     0,   129,   129,     0,   129,   215,   138,     0,
     107,     0,     0,     0,     0,     0,     0,   168,     0,   145,
     159,   159,   132,     0,   150,   178,     0,   184,   180,     0,
     188,    69,   159,   159,   159,   159,   159,     0,     0,    91,
       0,   268,   265,   268,   268,   273,   128,     0,   126,     0,
     123,     0,     0,     0,     0,     0,     0,   140,   161,   160,
     194,   196,   163,   164,   165,   166,   167,   147,     0,     0,
     134,   151,     0,   150,     0,     0,   183,   181,   182,   185,
     187,     0,     0,     0,     0,     0,   176,   150,    87,     0,
      68,   271,   267,   272,   270,     0,   349,   124,   209,     0,
     206,     0,     0,   211,     0,     0,     0,     0,   221,   196,
       0,   219,   169,   170,     0,   156,   158,   179,   171,   172,
     173,   174,   175,     0,   299,   277,   268,   294,     0,     0,
     129,   129,   129,   162,   242,     0,     0,   220,     7,   197,
     217,   218,   152,     0,   150,     0,     0,   298,     0,     0,
       0,     0,   261,   245,   246,   247,   248,   254,   255,   256,
     249,   250,   251,   252,   253,   141,   257,     0,   259,   260,
       0,   243,    56,     0,     0,   195,     0,     0,   177,   274,
       0,   278,   280,   295,   114,   207,   213,   212,   142,   258,
       0,   241,     0,     0,     0,   153,   154,   263,   262,   264,
     279,     0,   244,   338,     0,     0,     0,     0,     0,   315,
       0,     0,     0,   268,   234,   327,   305,   302,     0,   343,
     268,     0,   268,     0,   346,     0,     0,   314,     0,   268,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     344,   268,     0,     0,   317,     0,     0,   268,     0,   321,
     322,   324,   320,   319,   323,     0,   311,   313,   306,   308,
     337,     0,   233,     0,     0,   268,     0,   342,     0,     0,
     347,   316,     0,   326,   310,     0,     0,   328,   312,   303,
     281,   282,   283,   301,     0,     0,   296,     0,   268,     0,
     268,     0,   335,     0,   318,   325,     0,   339,     0,     0,
       0,   300,     0,   268,     0,     0,   345,     0,     0,   333,
       0,     0,   268,     0,   297,     0,     0,   268,   336,   339,
       0,   340,   284,     0,     0,     0,     0,   235,     0,     0,
     334,     0,   292,     0,     0,     0,     0,     0,   290,     0,
       0,   330,   268,   341,     0,     0,     0,     0,   286,     0,
     293,     0,     0,   289,   288,   287,   285,   291,   329,     0,
       0,   331,     0,   332
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -514,  -514,   603,  -514,  -241,    -1,   -57,   541,   556,   -55,
    -514,  -514,  -514,  -234,  -514,  -197,  -514,   -32,   -75,   -70,
     -69,  -514,  -166,   461,   -83,  -514,  -514,   340,  -514,  -514,
     -79,   433,   316,  -514,   -74,   333,  -514,  -514,   452,   323,
    -514,   200,  -514,  -514,  -318,  -514,  -191,   257,  -514,  -514,
    -514,  -133,  -514,  -514,  -514,  -514,  -514,  -514,  -514,   337,
    -514,   338,   580,  -514,   -64,   270,   585,  -514,  -514,   445,
    -514,  -514,  -514,   280,   282,  -514,   256,  -514,   197,  -514,
    -514,   352,  -143,    92,   -63,  -485,  -514,  -514,  -468,  -514,
    -514,  -373,    93,  -431,  -514,  -514,   162,  -500,  -514,   142,
    -514,  -478,  -514,  -460,    80,  -513,  -465,  -514,   158,   189,
     146,  -514,  -514
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     3,     4,    68,   191,   227,   136,     5,    59,    69,
      70,    71,   262,   263,   264,   199,   137,   228,   138,   151,
     152,   153,   154,   155,   265,   329,   278,   279,   101,   102,
     158,   173,   243,   244,   165,   225,   469,   235,   170,   236,
     226,   352,   457,   353,   354,   103,   292,   339,   104,   105,
     106,   171,   107,   185,   186,   187,   188,   189,   356,   307,
     249,   250,   386,   109,   342,   387,   388,   111,   112,   163,
     176,   389,   390,   126,   391,    72,   141,   416,   450,   451,
     480,   271,   147,   406,   493,   209,   407,   565,   603,   593,
     566,   408,   567,   370,   544,   515,   494,   511,   525,   535,
     508,   495,   537,   512,   589,   548,   500,   504,   505,   280,
     377,    73,    74
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[] =
{
      53,    54,   156,   207,    84,   139,   140,    79,   159,   161,
     311,   247,   539,   496,   162,   556,   127,   166,   167,   168,
     143,   502,   473,   552,   351,   509,   554,   351,   540,   157,
     290,   214,   536,   223,   145,   395,   428,   238,   293,   294,
     295,     1,     2,   497,   178,   524,   526,   144,   332,   403,
     256,   335,   224,   463,   617,   496,    55,   277,   536,   146,
     459,   269,    76,   272,    80,    81,   201,   516,   214,   320,
     575,   585,   520,   204,    75,   215,   587,   160,   210,   205,
     113,   570,   206,   608,   368,   144,  -155,   202,   572,   282,
     142,   459,   283,   460,   248,   174,   577,   595,   144,   217,
     611,   218,   219,    78,   221,   252,   253,   254,   255,   348,
     349,   483,   215,   192,   216,   241,   242,   193,   483,   639,
     631,   361,   362,   363,   364,   365,   613,   633,   601,   614,
     325,   157,   615,   616,   229,   230,   231,   330,   619,   164,
     642,   245,   586,   247,   169,   624,   626,   162,   621,   629,
     172,   484,   485,   486,   487,   488,   489,   490,   484,   485,
     486,   487,   488,   489,   490,  -180,  -275,  -180,   234,   483,
      77,   144,    78,  -275,   306,   198,   175,   455,   371,   641,
     373,   374,   491,   144,   177,    83,  -275,   285,   144,   491,
     286,  -275,    83,   551,   144,   281,   369,   190,  -275,   277,
     266,   411,   331,  -106,   333,   334,   300,   336,   301,   484,
     485,   486,   487,   488,   489,   490,   338,   200,    57,   612,
      58,   613,   203,    82,   614,    83,   248,   615,   616,   343,
     344,   345,   296,   630,   560,   613,   234,   340,   614,   341,
     491,   615,   616,    83,  -307,   305,   208,   308,    78,   477,
     478,   312,   313,   314,   315,   316,   317,   179,   180,   181,
     182,   183,   184,   425,   149,   150,   366,   212,   367,   394,
     213,   637,   397,   613,   326,   327,   614,   381,   483,   615,
     616,    78,   220,   222,   392,   393,   405,   129,   130,   131,
     132,   133,   134,   135,   483,   561,   562,   398,   399,   400,
     401,   402,   258,   259,   613,   357,   358,   614,   635,   338,
     615,   616,   251,   563,   194,   195,   196,   197,   484,   485,
     486,   487,   488,   489,   490,   237,   405,   268,   239,   267,
     273,   429,   430,   431,   484,   485,   486,   487,   488,   489,
     490,   270,   274,   405,   275,   483,   139,   140,   276,   491,
     513,   284,    83,  -309,   198,   483,   288,   517,   289,   519,
     291,   298,   232,   297,   299,   491,   528,   483,   523,   529,
     530,   531,   487,   532,   533,   534,   302,   303,   549,   304,
     309,   310,   318,   321,   555,   484,   485,   486,   487,   488,
     489,   490,   323,   422,   319,   484,   485,   486,   487,   488,
     489,   490,   569,   479,   424,   324,   322,   484,   485,   486,
     487,   488,   489,   490,   277,   453,   491,   346,   351,    83,
     355,   483,   306,   369,   376,   582,   491,   584,   483,   571,
     375,   378,   379,   380,   382,   384,   396,   409,   491,   383,
     596,   576,   410,   412,   385,   527,   418,   423,   456,   605,
     426,   427,   458,   474,   609,   468,   464,   462,   465,   128,
     564,   484,   485,   486,   487,   488,   489,   490,   484,   485,
     486,   487,   488,   489,   490,   466,    78,   467,    -9,   632,
     498,   568,   129,   130,   131,   132,   133,   134,   135,   472,
     470,   471,   491,   475,   476,   600,   591,   564,   481,   491,
     499,   501,   620,   503,   506,   510,   507,   514,   414,   518,
    -240,  -240,  -240,   522,  -240,  -240,  -240,    83,  -240,  -240,
    -240,  -240,  -240,   538,   541,   543,  -240,  -240,  -240,  -240,
    -240,  -240,  -240,  -240,  -240,  -240,  -240,  -240,  -240,  -240,
    -240,  -240,  -240,  -240,   545,  -240,  -240,  -240,  -240,  -240,
    -240,  -240,  -240,  -240,  -240,  -240,   547,  -240,   546,  -240,
    -240,   553,   557,   578,   559,   574,  -240,  -240,  -240,  -240,
    -240,  -240,  -240,  -240,   579,   580,  -240,  -240,  -240,  -240,
      85,    86,    87,    88,    89,   583,   581,   588,   597,   602,
     604,   415,    96,    97,   607,   590,    98,   598,   599,   627,
     618,   606,   622,   623,   634,   602,   638,   643,    56,   100,
      60,   211,   602,   602,   385,   628,   602,   257,   328,   350,
     483,   347,   337,   240,   461,    61,   636,    -5,    -5,    62,
      -5,    -5,    -5,    -5,    -5,    -5,    -5,    -5,    -5,    -5,
      -5,   413,    -5,    -5,   483,   359,    -5,   360,   108,  -304,
    -304,  -304,  -304,   110,  -304,  -304,  -304,  -304,  -304,   419,
     484,   485,   486,   487,   488,   489,   490,   287,   417,   482,
     421,   592,   454,   594,   372,    63,    64,   542,   558,   610,
     550,    65,    66,  -304,   484,   485,   486,   487,   488,   489,
     490,   491,   521,    67,   640,   573,     0,     0,     0,    -5,
     -62,     0,     0,   114,   115,   116,   117,     0,   118,   119,
     120,   121,   122,     0,     0,   491,  -304,     0,   492,  -304,
       1,     2,     0,    85,    86,    87,    88,    89,    90,    91,
      92,    93,    94,    95,   432,    96,    97,   123,     0,    98,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   149,   150,   433,     0,   434,   435,   436,   437,
     438,   439,     0,     0,   440,   441,   442,   443,   444,    78,
     124,     0,     0,   125,     0,   129,   130,   131,   132,   133,
     134,   135,   445,   446,     0,   432,     0,     0,     0,     0,
       0,     0,    99,     0,     0,     0,     0,     0,   404,   447,
       0,     0,     0,   448,   449,   433,     0,   434,   435,   436,
     437,   438,   439,     0,     0,   440,   441,   442,   443,   444,
       0,     0,   114,   115,   116,   117,     0,   118,   119,   120,
     121,   122,     0,   445,   446,     0,     0,     0,     0,     0,
       0,     6,     7,     8,     0,     9,    10,    11,     0,    12,
      13,    14,    15,    16,   448,   449,   123,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    31,    32,    33,   128,     0,    34,    35,    36,    37,
      38,    39,    40,    41,    42,    43,    44,     0,    45,   420,
      46,   452,   125,     0,     0,     0,     0,   129,   130,   131,
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
      46,    47,   625,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    48,     0,     0,    49,    50,    51,
      52,     6,     7,     8,     0,     9,    10,    11,     0,    12,
      13,    14,    15,    16,     0,     0,     0,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    31,    32,    33,     0,   560,    34,    35,    36,    37,
      38,    39,    40,    41,    42,    43,    44,     0,    45,     0,
      46,    47,     0,     0,     0,     0,     0,     0,   148,     0,
       0,     0,     0,     0,    48,   149,   150,    49,    50,    51,
      52,     0,     0,     0,     0,     0,     0,     0,   149,   150,
     246,     0,    78,     0,     0,     0,     0,     0,   129,   130,
     131,   132,   133,   134,   135,    78,   561,   562,   149,   150,
       0,   129,   130,   131,   132,   133,   134,   135,     0,     0,
       0,   149,   150,     0,     0,    78,     0,     0,     0,     0,
       0,   129,   130,   131,   132,   133,   134,   135,    78,   260,
     261,   149,   150,     0,   129,   130,   131,   132,   133,   134,
     135,     0,     0,     0,   149,     0,     0,     0,    78,     0,
       0,     0,     0,     0,   129,   130,   131,   132,   133,   134,
     135,    78,     0,     0,     0,     0,     0,   129,   130,   131,
     132,   133,   134,   135
};

static const yytype_int16 yycheck[] =
{
       1,     2,    85,   146,    67,    75,    75,    64,    87,    88,
     251,   177,   512,   473,    89,   528,    71,    91,    92,    93,
      77,   486,   453,   523,    17,   490,   526,    17,   513,    36,
     227,    37,   510,    30,    56,   353,   409,   170,   229,   230,
     231,     3,     4,   474,    99,   505,   506,    74,   282,   367,
     183,   285,    49,   426,    78,   515,     0,    84,   536,    81,
      84,   204,    63,   206,    65,    66,   141,   498,    37,   266,
     555,   571,   503,    75,    40,    81,   576,    84,   148,    81,
      76,   546,    84,   596,   318,    74,    76,   142,   548,    78,
      75,    84,    81,    86,   177,    96,   556,   582,    74,   156,
     600,   158,   159,    53,   161,   179,   180,   181,   182,   300,
     301,     1,    81,    60,    83,    38,    39,    64,     1,   632,
     620,   312,   313,   314,   315,   316,    81,    82,   588,    84,
     273,    36,    87,    88,   166,   167,   168,   280,   606,    84,
     640,   173,   573,   309,    84,   613,   614,   222,   608,   617,
      84,    41,    42,    43,    44,    45,    46,    47,    41,    42,
      43,    44,    45,    46,    47,    77,    56,    79,   169,     1,
      51,    74,    53,    56,    86,    78,    55,   418,   321,   639,
     323,   324,    72,    74,    78,    75,    76,    78,    74,    72,
      81,    81,    75,    76,    74,    81,    81,    73,    81,    84,
     201,    81,   281,    80,   283,   284,   238,   286,   240,    41,
      42,    43,    44,    45,    46,    47,   291,    80,    73,    79,
      75,    81,    74,    73,    84,    75,   309,    87,    88,   293,
     294,   295,   233,    79,     6,    81,   237,    73,    84,    75,
      72,    87,    88,    75,    76,   246,    76,   248,    53,    54,
      55,   252,   253,   254,   255,   256,   257,    11,    12,    13,
      14,    15,    16,   406,    36,    37,    73,    80,    75,   352,
      80,    79,   355,    81,   275,   276,    84,   334,     1,    87,
      88,    53,    66,    77,   348,   349,   369,    59,    60,    61,
      62,    63,    64,    65,     1,    67,    68,   361,   362,   363,
     364,   365,    59,    60,    81,    54,    55,    84,    85,   384,
      87,    88,    76,    85,    59,    60,    61,    62,    41,    42,
      43,    44,    45,    46,    47,    84,   409,    53,    84,    76,
      82,   410,   411,   412,    41,    42,    43,    44,    45,    46,
      47,    56,    73,   426,    80,     1,   416,   416,    80,    72,
     493,    85,    75,    76,    78,     1,    85,   500,    77,   502,
      74,    77,    49,    85,    85,    72,   509,     1,    75,    41,
      42,    43,    44,    45,    46,    47,    85,    77,   521,    75,
      77,    79,    77,    76,   527,    41,    42,    43,    44,    45,
      46,    47,    82,   394,    79,    41,    42,    43,    44,    45,
      46,    47,   545,   460,   405,    85,    77,    41,    42,    43,
      44,    45,    46,    47,    84,   416,    72,    85,    17,    75,
      86,     1,    86,    81,    85,   568,    72,   570,     1,    75,
      82,    82,    79,    82,    79,    77,    76,    81,    72,    82,
     583,    75,    81,    81,    40,   508,    76,    76,    84,   592,
      77,    82,    73,   454,   597,    55,    82,    85,    82,    36,
     543,    41,    42,    43,    44,    45,    46,    47,    41,    42,
      43,    44,    45,    46,    47,    82,    53,    82,    81,   622,
     481,   544,    59,    60,    61,    62,    63,    64,    65,    77,
      85,    85,    72,    85,    73,    75,   579,   580,    36,    72,
      81,    84,    75,    53,    75,    75,    55,    20,     1,    53,
       3,     4,     5,    77,     7,     8,     9,    75,    11,    12,
      13,    14,    15,    76,    86,    84,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    73,    38,    39,    40,    41,    42,
      43,    44,    45,    46,    47,    48,    82,    50,    85,    52,
      53,    76,    76,   564,    54,    76,    59,    60,    61,    62,
      63,    64,    65,    66,    74,    77,    69,    70,    71,    72,
       6,     7,     8,     9,    10,    73,    85,    48,    74,   590,
     591,    84,    18,    19,   595,    78,    22,    76,    76,    79,
      85,    78,    77,    76,    85,   606,    76,    76,     5,    68,
      54,   150,   613,   614,    40,   616,   617,   184,   278,   303,
       1,   298,   289,   171,   424,     1,   627,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    14,    15,
      16,   384,    18,    19,     1,   308,    22,   309,    68,     6,
       7,     8,     9,    68,    11,    12,    13,    14,    15,   389,
      41,    42,    43,    44,    45,    46,    47,   222,   386,   472,
     390,   579,   416,   580,   322,    51,    52,   515,   536,   599,
     522,    57,    58,    40,    41,    42,    43,    44,    45,    46,
      47,    72,   503,    69,    75,   549,    -1,    -1,    -1,    75,
      76,    -1,    -1,     6,     7,     8,     9,    -1,    11,    12,
      13,    14,    15,    -1,    -1,    72,    73,    -1,    75,    76,
       3,     4,    -1,     6,     7,     8,     9,    10,    11,    12,
      13,    14,    15,    16,     1,    18,    19,    40,    -1,    22,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
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
      32,    33,    34,    35,    36,    -1,    38,    39,    40,    41,
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
      -1,    36,    37,    -1,    -1,    53,    -1,    -1,    -1,    -1,
      -1,    59,    60,    61,    62,    63,    64,    65,    53,    54,
      55,    36,    37,    -1,    59,    60,    61,    62,    63,    64,
      65,    -1,    -1,    -1,    36,    -1,    -1,    -1,    53,    -1,
      -1,    -1,    -1,    -1,    59,    60,    61,    62,    63,    64,
      65,    53,    -1,    -1,    -1,    -1,    -1,    59,    60,    61,
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
      99,   100,   164,   200,   201,    40,    94,    51,    53,    95,
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
     198,    81,    78,    81,    85,    78,    81,   158,    85,    77,
     104,    74,   135,   135,   135,   135,    94,    85,    77,    85,
     106,   106,    85,    77,    75,    94,    86,   148,    94,    77,
      79,    93,    94,    94,    94,    94,    94,    94,    77,    79,
     104,    76,    77,    82,    85,   171,    94,    94,   116,   114,
     171,   119,   102,   119,   119,   102,   119,   124,   107,   136,
      73,    75,   153,   153,   153,   153,    85,   128,   135,   135,
     121,    17,   130,   132,   133,    86,   147,    54,    55,   148,
     150,   135,   135,   135,   135,   135,    73,    75,   102,    81,
     182,   171,   170,   171,   171,    82,    85,   199,    82,    79,
      82,    95,    79,    82,    77,    40,   151,   154,   155,   160,
     161,   163,   153,   153,   113,   133,    76,   113,   153,   153,
     153,   153,   153,   133,    82,   113,   172,   175,   180,    81,
      81,    81,    81,   136,     1,    84,   166,   163,    76,   154,
      73,   162,    94,    76,    94,   171,    77,    82,   180,   119,
     119,   119,     1,    21,    23,    24,    25,    26,    27,    28,
      31,    32,    33,    34,    35,    49,    50,    66,    70,    71,
     167,   168,    53,    94,   165,    93,    84,   131,    73,    84,
      86,   130,    85,   180,    82,    82,    82,    82,    55,   125,
      85,    85,    77,   182,    94,    85,    73,    54,    55,    95,
     169,    36,   167,     1,    41,    42,    43,    44,    45,    46,
      47,    72,    75,   173,   185,   190,   192,   182,    94,    81,
     195,    84,   195,    53,   196,   197,    75,    55,   189,   195,
      75,   186,   192,   171,    20,   184,   182,   171,    53,   171,
     182,   198,    77,    75,   192,   187,   192,   173,   171,    41,
      42,    43,    45,    46,    47,   188,   190,   191,    76,   186,
     174,    86,   185,    84,   183,    73,    85,    82,   194,   171,
     197,    76,   186,    76,   186,   171,   194,    76,   188,    54,
       6,    67,    68,    85,   113,   176,   179,   181,   173,   171,
     195,    75,   192,   199,    76,   174,    75,   192,    94,    74,
      77,    85,   171,    73,   171,   186,   182,   186,    48,   193,
      78,   113,   172,   178,   181,   174,   171,    74,    76,    76,
      75,   192,    94,   177,    94,   171,    78,    94,   194,   171,
     193,   186,    79,    81,    84,    87,    88,    78,    85,   177,
      75,   192,    77,    76,   177,    54,   177,    79,    94,   177,
      79,   186,   171,    82,    85,    85,    94,    79,    76,   194,
      75,   192,   186,    76
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
     193,   193,   194,   195,   196,   196,   197,   197,   198,   199,
     200,   201
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
       2,     2,     1,     1,     3,     0,     2,     4,     5,     0,
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
       2,     4,     1,     1,     2,     5,     1,     3,     1,     1,
       2,     2
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

/* This macro is provided for backward compatibility. */
#ifndef YY_LOCATION_PRINT
# define YY_LOCATION_PRINT(File, Loc) ((void) 0)
#endif


# define YY_SYMBOL_PRINT(Title, Type, Value, Location)                    \
do {                                                                      \
  if (yydebug)                                                            \
    {                                                                     \
      YYFPRINTF (stderr, "%s ", Title);                                   \
      yy_symbol_print (stderr,                                            \
                  Type, Value); \
      YYFPRINTF (stderr, "\n");                                           \
    }                                                                     \
} while (0)


/*----------------------------------------.
| Print this symbol's value on YYOUTPUT.  |
`----------------------------------------*/

static void
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
{
  FILE *yyo = yyoutput;
  YYUSE (yyo);
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
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
{
  YYFPRINTF (yyoutput, "%s %s (",
             yytype < YYNTOKENS ? "token" : "nterm", yytname[yytype]);

  yy_symbol_value_print (yyoutput, yytype, yyvaluep);
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
yy_reduce_print (yytype_int16 *yyssp, YYSTYPE *yyvsp, int yyrule)
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
                                              );
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)          \
do {                                    \
  if (yydebug)                          \
    yy_reduce_print (yyssp, yyvsp, Rule); \
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
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep)
{
  YYUSE (yyvaluep);
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

    YYSIZE_T yystacksize;

  int yyn;
  int yyresult;
  /* Lookahead token as an internal (translated) token number.  */
  int yytoken = 0;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;

#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  yyssp = yyss = yyssa;
  yyvsp = yyvs = yyvsa;
  yystacksize = YYINITDEPTH;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY; /* Cause a token to be read.  */
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

        /* Each stack pointer address is followed by the size of the
           data in use in that stack, in bytes.  This used to be a
           conditional around just the two extra args, but that might
           be undefined if yyoverflow is a macro.  */
        yyoverflow (YY_("memory exhausted"),
                    &yyss1, yysize * sizeof (*yyssp),
                    &yyvs1, yysize * sizeof (*yyvsp),
                    &yystacksize);

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
#  undef YYSTACK_RELOCATE
        if (yyss1 != yyssa)
          YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;

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


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 2:
#line 163 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.modlist) = (yyvsp[0].modlist); modlist = (yyvsp[0].modlist); }
#line 1950 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 3:
#line 167 "xi-grammar.y" /* yacc.c:1646  */
    { 
		  (yyval.modlist) = 0; 
		}
#line 1958 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 4:
#line 171 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.modlist) = new AstChildren<Module>(lineno, (yyvsp[-1].module), (yyvsp[0].modlist)); }
#line 1964 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 5:
#line 175 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 1970 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 6:
#line 177 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 1976 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 7:
#line 181 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 1982 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 8:
#line 183 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 1988 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 9:
#line 188 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 1994 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 10:
#line 189 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MODULE); YYABORT; }
#line 2000 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 11:
#line 190 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MAINMODULE); YYABORT; }
#line 2006 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 12:
#line 191 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(EXTERN); YYABORT; }
#line 2012 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 13:
#line 193 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITCALL); YYABORT; }
#line 2018 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 14:
#line 194 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITNODE); YYABORT; }
#line 2024 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 15:
#line 195 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITPROC); YYABORT; }
#line 2030 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 16:
#line 197 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CHARE); }
#line 2036 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 17:
#line 198 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MAINCHARE); }
#line 2042 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 18:
#line 199 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(GROUP); }
#line 2048 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 19:
#line 200 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NODEGROUP); }
#line 2054 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 20:
#line 201 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ARRAY); }
#line 2060 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 21:
#line 205 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INCLUDE); YYABORT; }
#line 2066 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 22:
#line 206 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(STACKSIZE); YYABORT; }
#line 2072 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 23:
#line 207 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(THREADED); YYABORT; }
#line 2078 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 24:
#line 208 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(TEMPLATE); YYABORT; }
#line 2084 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 25:
#line 209 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(SYNC); YYABORT; }
#line 2090 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 26:
#line 210 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IGET); YYABORT; }
#line 2096 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 27:
#line 211 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(EXCLUSIVE); YYABORT; }
#line 2102 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 28:
#line 212 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IMMEDIATE); YYABORT; }
#line 2108 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 29:
#line 213 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(SKIPSCHED); YYABORT; }
#line 2114 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 30:
#line 214 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INLINE); YYABORT; }
#line 2120 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 31:
#line 215 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(VIRTUAL); YYABORT; }
#line 2126 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 32:
#line 216 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MIGRATABLE); YYABORT; }
#line 2132 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 33:
#line 217 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CREATEHERE); YYABORT; }
#line 2138 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 34:
#line 218 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CREATEHOME); YYABORT; }
#line 2144 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 35:
#line 219 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOKEEP); YYABORT; }
#line 2150 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 36:
#line 220 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOTRACE); YYABORT; }
#line 2156 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 37:
#line 221 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(APPWORK); YYABORT; }
#line 2162 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 38:
#line 224 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(PACKED); YYABORT; }
#line 2168 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 39:
#line 225 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(VARSIZE); YYABORT; }
#line 2174 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 40:
#line 226 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ENTRY); YYABORT; }
#line 2180 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 41:
#line 227 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(FOR); YYABORT; }
#line 2186 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 42:
#line 228 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(FORALL); YYABORT; }
#line 2192 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 43:
#line 229 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(WHILE); YYABORT; }
#line 2198 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 44:
#line 230 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(WHEN); YYABORT; }
#line 2204 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 45:
#line 231 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(OVERLAP); YYABORT; }
#line 2210 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 46:
#line 232 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ATOMIC); YYABORT; }
#line 2216 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 47:
#line 233 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IF); YYABORT; }
#line 2222 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 48:
#line 234 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ELSE); YYABORT; }
#line 2228 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 49:
#line 236 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(LOCAL); YYABORT; }
#line 2234 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 50:
#line 238 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(USING); YYABORT; }
#line 2240 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 51:
#line 239 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ACCEL); YYABORT; }
#line 2246 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 52:
#line 242 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ACCELBLOCK); YYABORT; }
#line 2252 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 53:
#line 243 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MEMCRITICAL); YYABORT; }
#line 2258 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 54:
#line 244 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(REDUCTIONTARGET); YYABORT; }
#line 2264 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 55:
#line 245 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CASE); YYABORT; }
#line 2270 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 56:
#line 249 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 2276 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 57:
#line 251 "xi-grammar.y" /* yacc.c:1646  */
    {
		  char *tmp = new char[strlen((yyvsp[-3].strval))+strlen((yyvsp[0].strval))+3];
		  sprintf(tmp,"%s::%s", (yyvsp[-3].strval), (yyvsp[0].strval));
		  (yyval.strval) = tmp;
		}
#line 2286 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 58:
#line 259 "xi-grammar.y" /* yacc.c:1646  */
    { 
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist)); 
		}
#line 2294 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 59:
#line 263 "xi-grammar.y" /* yacc.c:1646  */
    {  
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist)); 
		    (yyval.module)->setMain();
		}
#line 2303 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 60:
#line 270 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = 0; }
#line 2309 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 61:
#line 272 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = (yyvsp[-2].conslist); }
#line 2315 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 62:
#line 276 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = 0; }
#line 2321 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 63:
#line 278 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = new ConstructList(lineno, (yyvsp[-1].construct), (yyvsp[0].conslist)); }
#line 2327 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 64:
#line 282 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new UsingScope((yyvsp[0].strval), false); }
#line 2333 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 65:
#line 284 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new UsingScope((yyvsp[0].strval), true); }
#line 2339 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 66:
#line 286 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].member)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].member); }
#line 2345 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 67:
#line 288 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].message)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].message); }
#line 2351 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 68:
#line 290 "xi-grammar.y" /* yacc.c:1646  */
    {
                  Entry *e = new Entry(lineno, 0, (yyvsp[-4].type), (yyvsp[-2].strval), (yyvsp[0].plist), 0, 0, 0);
                  int isExtern = 1;
                  e->setExtern(isExtern);
                  e->targs = (yyvsp[-1].tparlist);
                  e->label = new XStr;
                  (yyvsp[-3].ntype)->print(*e->label);
                  (yyval.construct) = e;
                }
#line 2365 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 69:
#line 302 "xi-grammar.y" /* yacc.c:1646  */
    { if((yyvsp[-2].conslist)) (yyvsp[-2].conslist)->recurse<int&>((yyvsp[-4].intval), &Construct::setExtern); (yyval.construct) = (yyvsp[-2].conslist); }
#line 2371 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 70:
#line 304 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new Scope((yyvsp[-3].strval), (yyvsp[-1].conslist)); }
#line 2377 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 71:
#line 306 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = (yyvsp[-1].construct); }
#line 2383 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 72:
#line 308 "xi-grammar.y" /* yacc.c:1646  */
    { yyerror("The preceding construct must be semicolon terminated"); YYABORT; }
#line 2389 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 73:
#line 310 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].module)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].module); }
#line 2395 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 74:
#line 312 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2401 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 75:
#line 314 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2407 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 76:
#line 316 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2413 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 77:
#line 318 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2419 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 78:
#line 320 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].templat)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].templat); }
#line 2425 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 79:
#line 322 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = NULL; }
#line 2431 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 80:
#line 324 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = NULL; }
#line 2437 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 81:
#line 326 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = (yyvsp[0].accelBlock); }
#line 2443 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 82:
#line 328 "xi-grammar.y" /* yacc.c:1646  */
    { printf("Invalid construct\n"); YYABORT; }
#line 2449 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 83:
#line 332 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamType((yyvsp[0].type)); }
#line 2455 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 84:
#line 334 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
#line 2461 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 85:
#line 336 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
#line 2467 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 86:
#line 340 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = new TParamList((yyvsp[0].tparam)); }
#line 2473 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 87:
#line 342 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = new TParamList((yyvsp[-2].tparam), (yyvsp[0].tparlist)); }
#line 2479 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 88:
#line 346 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = 0; }
#line 2485 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 89:
#line 348 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = (yyvsp[0].tparlist); }
#line 2491 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 90:
#line 352 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = 0; }
#line 2497 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 91:
#line 354 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = (yyvsp[-1].tparlist); }
#line 2503 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 92:
#line 358 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("int"); }
#line 2509 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 93:
#line 360 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long"); }
#line 2515 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 94:
#line 362 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("short"); }
#line 2521 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 95:
#line 364 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("char"); }
#line 2527 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 96:
#line 366 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned int"); }
#line 2533 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 97:
#line 368 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long"); }
#line 2539 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 98:
#line 370 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long"); }
#line 2545 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 99:
#line 372 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long long"); }
#line 2551 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 100:
#line 374 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned short"); }
#line 2557 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 101:
#line 376 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned char"); }
#line 2563 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 102:
#line 378 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long long"); }
#line 2569 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 103:
#line 380 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("float"); }
#line 2575 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 104:
#line 382 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("double"); }
#line 2581 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 105:
#line 384 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long double"); }
#line 2587 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 106:
#line 386 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("void"); }
#line 2593 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 107:
#line 389 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ntype) = new NamedType((yyvsp[-1].strval),(yyvsp[0].tparlist)); }
#line 2599 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 108:
#line 390 "xi-grammar.y" /* yacc.c:1646  */
    { 
                    const char* basename, *scope;
                    splitScopedName((yyvsp[-1].strval), &scope, &basename);
                    (yyval.ntype) = new NamedType(basename, (yyvsp[0].tparlist), scope);
                }
#line 2609 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 109:
#line 398 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 2615 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 110:
#line 400 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ntype); }
#line 2621 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 111:
#line 404 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ptype) = new PtrType((yyvsp[-1].type)); }
#line 2627 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 112:
#line 408 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
#line 2633 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 113:
#line 410 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
#line 2639 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 114:
#line 414 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ftype) = new FuncType((yyvsp[-7].type), (yyvsp[-4].strval), (yyvsp[-1].plist)); }
#line 2645 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 115:
#line 418 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 2651 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 116:
#line 420 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 2657 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 117:
#line 422 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 2663 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 118:
#line 424 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ftype); }
#line 2669 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 119:
#line 427 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[0].type)); }
#line 2675 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 120:
#line 429 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[-1].type)); }
#line 2681 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 121:
#line 433 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ReferenceType((yyvsp[-1].type)); }
#line 2687 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 122:
#line 435 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 2693 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 123:
#line 439 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 2699 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 124:
#line 443 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = (yyvsp[-1].val); }
#line 2705 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 125:
#line 447 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.vallist) = 0; }
#line 2711 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 126:
#line 449 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.vallist) = new ValueList((yyvsp[-1].val), (yyvsp[0].vallist)); }
#line 2717 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 127:
#line 453 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].vallist)); }
#line 2723 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 128:
#line 457 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[-2].type), (yyvsp[0].strval), 0, 1); }
#line 2729 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 129:
#line 461 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0;}
#line 2735 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 130:
#line 463 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0;}
#line 2741 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 131:
#line 467 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 2747 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 132:
#line 469 "xi-grammar.y" /* yacc.c:1646  */
    { 
		  /*
		  printf("Warning: Message attributes are being phased out.\n");
		  printf("Warning: Please remove them from interface files.\n");
		  */
		  (yyval.intval) = (yyvsp[-1].intval); 
		}
#line 2759 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 133:
#line 479 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[0].intval); }
#line 2765 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 134:
#line 481 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[-2].intval) | (yyvsp[0].intval); }
#line 2771 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 135:
#line 485 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 2777 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 136:
#line 487 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 2783 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 137:
#line 491 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = 0; }
#line 2789 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 138:
#line 493 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-1].cattr); }
#line 2795 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 139:
#line 497 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[0].cattr); }
#line 2801 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 140:
#line 499 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
#line 2807 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 141:
#line 503 "xi-grammar.y" /* yacc.c:1646  */
    { python_doc = NULL; (yyval.intval) = 0; }
#line 2813 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 142:
#line 505 "xi-grammar.y" /* yacc.c:1646  */
    { python_doc = (yyvsp[0].strval); (yyval.intval) = 0; }
#line 2819 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 143:
#line 509 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CPYTHON; }
#line 2825 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 144:
#line 513 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = 0; }
#line 2831 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 145:
#line 515 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-1].cattr); }
#line 2837 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 146:
#line 519 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[0].cattr); }
#line 2843 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 147:
#line 521 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
#line 2849 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 148:
#line 525 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CMIGRATABLE; }
#line 2855 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 149:
#line 527 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CPYTHON; }
#line 2861 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 150:
#line 531 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 2867 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 151:
#line 533 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 2873 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 152:
#line 536 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 2879 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 153:
#line 538 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 2885 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 154:
#line 541 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mv) = new MsgVar((yyvsp[-3].type), (yyvsp[-2].strval), (yyvsp[-4].intval), (yyvsp[-1].intval)); }
#line 2891 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 155:
#line 545 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mvlist) = new MsgVarList((yyvsp[0].mv)); }
#line 2897 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 156:
#line 547 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mvlist) = new MsgVarList((yyvsp[-1].mv), (yyvsp[0].mvlist)); }
#line 2903 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 157:
#line 551 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, (yyvsp[0].ntype)); }
#line 2909 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 158:
#line 553 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, (yyvsp[-3].ntype), (yyvsp[-1].mvlist)); }
#line 2915 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 159:
#line 557 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = 0; }
#line 2921 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 160:
#line 559 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = (yyvsp[0].typelist); }
#line 2927 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 161:
#line 563 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = new TypeList((yyvsp[0].ntype)); }
#line 2933 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 162:
#line 565 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = new TypeList((yyvsp[-2].ntype), (yyvsp[0].typelist)); }
#line 2939 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 163:
#line 569 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr)|Chare::CCHARE, (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 2945 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 164:
#line 571 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 2951 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 165:
#line 575 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 2957 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 166:
#line 579 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new NodeGroup(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 2963 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 167:
#line 583 "xi-grammar.y" /* yacc.c:1646  */
    {/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",(yyvsp[-2].strval));
			(yyval.ntype) = new NamedType(buf); 
		}
#line 2973 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 168:
#line 589 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ntype) = new NamedType((yyvsp[-1].strval)); }
#line 2979 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 169:
#line 593 "xi-grammar.y" /* yacc.c:1646  */
    {  (yyval.chare) = new Array(lineno, (yyvsp[-4].cattr), (yyvsp[-3].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 2985 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 170:
#line 595 "xi-grammar.y" /* yacc.c:1646  */
    {  (yyval.chare) = new Array(lineno, (yyvsp[-3].cattr), (yyvsp[-4].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 2991 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 171:
#line 599 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr)|Chare::CCHARE, new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist));}
#line 2997 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 172:
#line 601 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3003 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 173:
#line 605 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3009 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 174:
#line 609 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new NodeGroup( lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3015 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 175:
#line 613 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Array( lineno, 0, (yyvsp[-3].ntype), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3021 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 176:
#line 617 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-1].strval))); }
#line 3027 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 177:
#line 619 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-4].strval)), (yyvsp[-2].mvlist)); }
#line 3033 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 178:
#line 623 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = 0; }
#line 3039 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 179:
#line 625 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3045 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 180:
#line 629 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = 0; }
#line 3051 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 181:
#line 631 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3057 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 182:
#line 633 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3063 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 183:
#line 637 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TType(new NamedType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3069 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 184:
#line 639 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TFunc((yyvsp[-1].ftype), (yyvsp[0].strval)); }
#line 3075 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 185:
#line 641 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TName((yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].strval)); }
#line 3081 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 186:
#line 645 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = new TVarList((yyvsp[0].tvar)); }
#line 3087 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 187:
#line 647 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = new TVarList((yyvsp[-2].tvar), (yyvsp[0].tvarlist)); }
#line 3093 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 188:
#line 651 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = (yyvsp[-1].tvarlist); }
#line 3099 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 189:
#line 655 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3105 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 190:
#line 657 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3111 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 191:
#line 659 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3117 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 192:
#line 661 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3123 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 193:
#line 663 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].message)); (yyvsp[0].message)->setTemplate((yyval.templat)); }
#line 3129 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 194:
#line 667 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = 0; }
#line 3135 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 195:
#line 669 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = (yyvsp[-2].mbrlist); }
#line 3141 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 196:
#line 673 "xi-grammar.y" /* yacc.c:1646  */
    { 
                  if (!connectEntries.empty()) {
                    (yyval.mbrlist) = new AstChildren<Member>(connectEntries);
		  } else {
		    (yyval.mbrlist) = 0; 
                  }
		}
#line 3153 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 197:
#line 681 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = new AstChildren<Member>(-1, (yyvsp[-1].member), (yyvsp[0].mbrlist)); }
#line 3159 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 198:
#line 685 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].readonly); }
#line 3165 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 199:
#line 687 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].readonly); }
#line 3171 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 201:
#line 690 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].member); }
#line 3177 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 202:
#line 692 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].pupable); }
#line 3183 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 203:
#line 694 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].includeFile); }
#line 3189 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 204:
#line 696 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new ClassDeclaration(lineno,(yyvsp[0].strval)); }
#line 3195 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 205:
#line 700 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1); }
#line 3201 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 206:
#line 702 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1); }
#line 3207 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 207:
#line 704 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[-6].strval)) + '<' +
					    ((yyvsp[-4].tparlist))->to_string() + '>').c_str()),
				    1);
		}
#line 3217 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 208:
#line 710 "xi-grammar.y" /* yacc.c:1646  */
    { printf("Warning: deprecated use of initcall. Use initnode or initproc instead.\n"); 
		  (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1); }
#line 3224 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 209:
#line 713 "xi-grammar.y" /* yacc.c:1646  */
    { printf("Warning: deprecated use of initcall. Use initnode or initproc instead.\n");
		  (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1); }
#line 3231 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 210:
#line 718 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 0); }
#line 3237 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 211:
#line 720 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 0); }
#line 3243 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 212:
#line 722 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[-6].strval)) + '<' +
					    ((yyvsp[-4].tparlist))->to_string() + '>').c_str()),
				    0);
		}
#line 3253 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 213:
#line 728 "xi-grammar.y" /* yacc.c:1646  */
    {
                  InitCall* rtn = new InitCall(lineno, (yyvsp[-3].strval), 0);
                  rtn->setAccel();
                  (yyval.member) = rtn;
		}
#line 3263 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 214:
#line 736 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[0].ntype),0); }
#line 3269 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 215:
#line 738 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[-2].ntype),(yyvsp[0].pupable)); }
#line 3275 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 216:
#line 741 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.includeFile) = new IncludeFile(lineno,(yyvsp[0].strval)); }
#line 3281 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 217:
#line 745 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[-1].member); }
#line 3287 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 218:
#line 748 "xi-grammar.y" /* yacc.c:1646  */
    { yyerror("The preceding entry method declaration must be semicolon-terminated."); YYABORT; }
#line 3293 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 219:
#line 752 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].entry); }
#line 3299 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 220:
#line 754 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyvsp[0].entry)->tspec = (yyvsp[-1].tvarlist);
                  (yyval.member) = (yyvsp[0].entry);
                }
#line 3308 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 221:
#line 759 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].member); }
#line 3314 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 222:
#line 763 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3320 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 223:
#line 765 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3326 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 224:
#line 767 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3332 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 225:
#line 769 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3338 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 226:
#line 771 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3344 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 227:
#line 773 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3350 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 228:
#line 775 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3356 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 229:
#line 777 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3362 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 230:
#line 779 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3368 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 231:
#line 781 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3374 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 232:
#line 783 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3380 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 233:
#line 786 "xi-grammar.y" /* yacc.c:1646  */
    { 
                  (yyval.entry) = new Entry(lineno, (yyvsp[-5].intval), (yyvsp[-4].type), (yyvsp[-3].strval), (yyvsp[-2].plist), (yyvsp[-1].val), (yyvsp[0].sc)); 
		  if ((yyvsp[0].sc) != 0) { 
		    (yyvsp[0].sc)->con1 = new SdagConstruct(SIDENT, (yyvsp[-3].strval));
                    (yyvsp[0].sc)->entry = (yyval.entry);
                    (yyvsp[0].sc)->con1->entry = (yyval.entry);
                    (yyvsp[0].sc)->param = new ParamList((yyvsp[-2].plist));
                  }
		}
#line 3394 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 234:
#line 796 "xi-grammar.y" /* yacc.c:1646  */
    { 
                  Entry *e = new Entry(lineno, (yyvsp[-3].intval), 0, (yyvsp[-2].strval), (yyvsp[-1].plist),  0, (yyvsp[0].sc));
                  if ((yyvsp[0].sc) != 0) {
		    (yyvsp[0].sc)->con1 = new SdagConstruct(SIDENT, (yyvsp[-2].strval));
                    (yyvsp[0].sc)->entry = e;
                    (yyvsp[0].sc)->con1->entry = e;
                    (yyvsp[0].sc)->param = new ParamList((yyvsp[-1].plist));
                  }
		  if (e->param && e->param->isCkMigMsgPtr()) {
		    yyerror("Charm++ takes a CkMigrateMsg chare constructor for granted, but continuing anyway");
		    (yyval.entry) = NULL;
		  } else
		    (yyval.entry) = e;
		}
#line 3413 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 235:
#line 811 "xi-grammar.y" /* yacc.c:1646  */
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
#line 3431 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 236:
#line 827 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.accelBlock) = new AccelBlock(lineno, new XStr((yyvsp[-2].strval))); }
#line 3437 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 237:
#line 829 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.accelBlock) = new AccelBlock(lineno, NULL); }
#line 3443 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 238:
#line 833 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("void"); }
#line 3449 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 239:
#line 835 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3455 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 240:
#line 839 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3461 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 241:
#line 841 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[-1].intval); }
#line 3467 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 242:
#line 843 "xi-grammar.y" /* yacc.c:1646  */
    { printf("Invalid entry method attribute list\n"); YYABORT; }
#line 3473 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 243:
#line 847 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[0].intval); }
#line 3479 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 244:
#line 849 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[-2].intval) | (yyvsp[0].intval); }
#line 3485 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 245:
#line 853 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = STHREADED; }
#line 3491 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 246:
#line 855 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SSYNC; }
#line 3497 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 247:
#line 857 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SIGET; }
#line 3503 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 248:
#line 859 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SLOCKED; }
#line 3509 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 249:
#line 861 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SCREATEHERE; }
#line 3515 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 250:
#line 863 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SCREATEHOME; }
#line 3521 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 251:
#line 865 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SNOKEEP; }
#line 3527 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 252:
#line 867 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SNOTRACE; }
#line 3533 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 253:
#line 869 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SAPPWORK; }
#line 3539 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 254:
#line 871 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SIMMEDIATE; }
#line 3545 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 255:
#line 873 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SSKIPSCHED; }
#line 3551 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 256:
#line 875 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SINLINE; }
#line 3557 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 257:
#line 877 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SLOCAL; }
#line 3563 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 258:
#line 879 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SPYTHON; }
#line 3569 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 259:
#line 881 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SMEM; }
#line 3575 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 260:
#line 883 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SREDUCE; }
#line 3581 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 261:
#line 885 "xi-grammar.y" /* yacc.c:1646  */
    { printf("Invalid entry method attribute: %s\n", yylval); YYABORT; }
#line 3587 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 262:
#line 889 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 3593 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 263:
#line 891 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 3599 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 264:
#line 893 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 3605 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 265:
#line 897 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = ""; }
#line 3611 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 266:
#line 899 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3617 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 267:
#line 901 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s, %s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 3627 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 268:
#line 909 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = ""; }
#line 3633 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 269:
#line 911 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3639 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 270:
#line 913 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s[%s]%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 3649 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 271:
#line 919 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s{%s}%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 3659 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 272:
#line 925 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s(%s)%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 3669 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 273:
#line 931 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"(%s)%s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 3679 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 274:
#line 939 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval));
		}
#line 3688 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 275:
#line 946 "xi-grammar.y" /* yacc.c:1646  */
    { 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			(yyval.intval) = 0;
		}
#line 3698 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 276:
#line 954 "xi-grammar.y" /* yacc.c:1646  */
    { 
			in_braces=0;
			(yyval.intval) = 0;
		}
#line 3707 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 277:
#line 961 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[0].type));}
#line 3713 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 278:
#line 963 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval)); (yyval.pname)->setConditional((yyvsp[0].intval)); }
#line 3719 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 279:
#line 965 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[-3].type),(yyvsp[-2].strval),0,(yyvsp[0].val));}
#line 3725 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 280:
#line 967 "xi-grammar.y" /* yacc.c:1646  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
		}
#line 3734 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 281:
#line 973 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READONLY; }
#line 3740 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 282:
#line 974 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READWRITE; }
#line 3746 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 283:
#line 975 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_WRITEONLY; }
#line 3752 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 284:
#line 978 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr((yyvsp[0].strval)); }
#line 3758 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 285:
#line 979 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "->" << (yyvsp[0].strval); }
#line 3764 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 286:
#line 980 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[-2].xstrptr)) << "." << (yyvsp[0].strval); }
#line 3770 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 287:
#line 982 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << *((yyvsp[-1].xstrptr)) << "]";
                  delete (yyvsp[-3].xstrptr);
                  delete (yyvsp[-1].xstrptr);
                }
#line 3781 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 288:
#line 989 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << (yyvsp[-1].strval) << "]";
                  delete (yyvsp[-3].xstrptr);
                }
#line 3791 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 289:
#line 995 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "(" << *((yyvsp[-1].xstrptr)) << ")";
                  delete (yyvsp[-3].xstrptr);
                  delete (yyvsp[-1].xstrptr);
                }
#line 3802 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 290:
#line 1004 "xi-grammar.y" /* yacc.c:1646  */
    {
                  in_bracket = 0;
                  (yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName(), (yyvsp[-1].strval));
                }
#line 3811 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 291:
#line 1011 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[-6].intval));
                }
#line 3821 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 292:
#line 1017 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType(Parameter::ACCEL_BUFFER_TYPE_READWRITE);
		}
#line 3831 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 293:
#line 1023 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.pname) = (yyvsp[-3].pname);
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[-5].intval));
		}
#line 3841 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 294:
#line 1031 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 3847 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 295:
#line 1033 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 3853 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 296:
#line 1037 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 3859 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 297:
#line 1039 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 3865 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 298:
#line 1043 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = (yyvsp[-1].plist); }
#line 3871 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 299:
#line 1045 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList(new Parameter(0, new BuiltinType("void"))); }
#line 3877 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 300:
#line 1049 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = (yyvsp[-1].plist); }
#line 3883 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 301:
#line 1051 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = 0; }
#line 3889 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 302:
#line 1055 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = 0; }
#line 3895 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 303:
#line 1057 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 3901 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 304:
#line 1061 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = 0; }
#line 3907 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 305:
#line 1063 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new SdagConstruct(SSDAGENTRY, (yyvsp[0].sc)); }
#line 3913 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 306:
#line 1065 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new SdagConstruct(SSDAGENTRY, (yyvsp[-1].slist)); }
#line 3919 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 307:
#line 1069 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.slist) = new SListConstruct((yyvsp[0].sc)); }
#line 3925 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 308:
#line 1071 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.slist) = new SListConstruct((yyvsp[-1].sc), (yyvsp[0].slist));  }
#line 3931 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 309:
#line 1075 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.olist) = new OListConstruct((yyvsp[0].sc)); }
#line 3937 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 310:
#line 1077 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.olist) = new OListConstruct((yyvsp[-1].sc), (yyvsp[0].slist)); }
#line 3943 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 311:
#line 1081 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.clist) = new CaseListConstruct((yyvsp[0].when)); }
#line 3949 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 312:
#line 1083 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.clist) = new CaseListConstruct((yyvsp[-1].when), (yyvsp[0].clist)); }
#line 3955 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 313:
#line 1085 "xi-grammar.y" /* yacc.c:1646  */
    { yyerror("Case blocks in SDAG can only contain when clauses."); YYABORT; }
#line 3961 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 314:
#line 1089 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3967 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 315:
#line 1091 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = 0; }
#line 3973 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 316:
#line 1095 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-2].entrylist), 0); }
#line 3979 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 317:
#line 1097 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-1].entrylist), (yyvsp[0].sc)); }
#line 3985 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 318:
#line 1099 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-3].entrylist), (yyvsp[-1].slist)); }
#line 3991 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 319:
#line 1103 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 3997 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 320:
#line 1105 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4003 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 321:
#line 1107 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4009 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 322:
#line 1109 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4015 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 323:
#line 1111 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4021 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 324:
#line 1113 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4027 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 325:
#line 1117 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new AtomicConstruct((yyvsp[-1].strval), (yyvsp[-3].strval)); }
#line 4033 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 326:
#line 1119 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new OverlapConstruct((yyvsp[-1].olist)); }
#line 4039 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 327:
#line 1121 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = (yyvsp[0].when); }
#line 4045 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 328:
#line 1123 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new CaseConstruct((yyvsp[-1].clist)); }
#line 4051 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 329:
#line 1125 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForConstruct(new SdagConstruct(SINT_EXPR, (yyvsp[-8].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[-6].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[-4].strval)), (yyvsp[-1].slist)); }
#line 4057 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 330:
#line 1127 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForConstruct(new SdagConstruct(SINT_EXPR, (yyvsp[-6].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[-4].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[-2].strval)), (yyvsp[0].sc)); }
#line 4063 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 331:
#line 1129 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-9].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[-6].strval)), 
		             new SdagConstruct(SINT_EXPR, (yyvsp[-4].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[-2].strval)), (yyvsp[0].sc)); }
#line 4070 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 332:
#line 1132 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-11].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[-8].strval)), 
		             new SdagConstruct(SINT_EXPR, (yyvsp[-6].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[-4].strval)), (yyvsp[-1].slist)); }
#line 4077 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 333:
#line 1135 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new IfConstruct(new SdagConstruct(SINT_EXPR, (yyvsp[-3].strval)), (yyvsp[-1].sc), (yyvsp[0].sc)); }
#line 4083 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 334:
#line 1137 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new IfConstruct(new SdagConstruct(SINT_EXPR, (yyvsp[-5].strval)), (yyvsp[-2].slist), (yyvsp[0].sc)); }
#line 4089 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 335:
#line 1139 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new WhileConstruct(new SdagConstruct(SINT_EXPR, (yyvsp[-2].strval)), (yyvsp[0].sc)); }
#line 4095 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 336:
#line 1141 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new WhileConstruct(new SdagConstruct(SINT_EXPR, (yyvsp[-4].strval)), (yyvsp[-1].slist)); }
#line 4101 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 337:
#line 1143 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new AtomicConstruct((yyvsp[-1].strval), NULL); }
#line 4107 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 338:
#line 1145 "xi-grammar.y" /* yacc.c:1646  */
    { printf("Unknown SDAG construct or malformed entry method definition.\n"
                         "You may have forgotten to terminate an entry method definition with a"
                         " semicolon or forgotten to mark a block of sequential SDAG code as 'atomic'\n"); YYABORT; }
#line 4115 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 339:
#line 1151 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = 0; }
#line 4121 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 340:
#line 1153 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ElseConstruct((yyvsp[0].sc)); }
#line 4127 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 341:
#line 1155 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ElseConstruct((yyvsp[-1].slist)); }
#line 4133 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 342:
#line 1159 "xi-grammar.y" /* yacc.c:1646  */
    { in_int_expr = 0; (yyval.intval) = 0; }
#line 4139 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 343:
#line 1163 "xi-grammar.y" /* yacc.c:1646  */
    { in_int_expr = 1; (yyval.intval) = 0; }
#line 4145 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 344:
#line 1167 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[-1].strval), (yyvsp[0].plist), 0, 0, 0); }
#line 4151 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 345:
#line 1169 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[-4].strval), (yyvsp[0].plist), 0, 0, (yyvsp[-2].strval)); }
#line 4157 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 346:
#line 1173 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.entrylist) = new EntryList((yyvsp[0].entry)); }
#line 4163 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 347:
#line 1175 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.entrylist) = new EntryList((yyvsp[-2].entry),(yyvsp[0].entrylist)); }
#line 4169 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 348:
#line 1179 "xi-grammar.y" /* yacc.c:1646  */
    { in_bracket=1; }
#line 4175 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 349:
#line 1182 "xi-grammar.y" /* yacc.c:1646  */
    { in_bracket=0; }
#line 4181 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 350:
#line 1186 "xi-grammar.y" /* yacc.c:1646  */
    { if (!macroDefined((yyvsp[0].strval), 1)) in_comment = 1; }
#line 4187 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 351:
#line 1190 "xi-grammar.y" /* yacc.c:1646  */
    { if (!macroDefined((yyvsp[0].strval), 0)) in_comment = 1; }
#line 4193 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;


#line 4197 "xi-grammar.tab.c" /* yacc.c:1646  */
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
                      yytoken, &yylval);
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


      yydestruct ("Error: popping",
                  yystos[yystate], yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END


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
                  yytoken, &yylval);
    }
  /* Do not reclaim the symbols of the rule whose action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
                  yystos[*yyssp], yyvsp);
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
#line 1193 "xi-grammar.y" /* yacc.c:1906  */

void yyerror(const char *mesg)
{
    std::cerr << cur_file<<":"<<lineno<<": Charmxi syntax error> "
	      << mesg << std::endl;
}
