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
  IntExprConstruct *intexpr;
  WhenConstruct *when;
  SListConstruct *slist;
  CaseListConstruct *clist;
  OListConstruct *olist;
  SdagEntryConstruct *sentry;
  XStr* xstrptr;
  AccelBlock* accelBlock;

#line 243 "xi-grammar.tab.c" /* yacc.c:355  */
};

typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif


extern YYSTYPE yylval;

int yyparse (void);



/* Copy the second part of user declarations.  */

#line 260 "xi-grammar.tab.c" /* yacc.c:358  */

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
#define YYLAST   1261

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  89
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  114
/* YYNRULES -- Number of rules.  */
#define YYNRULES  352
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  645

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
       0,   166,   166,   171,   174,   179,   180,   185,   186,   191,
     193,   194,   195,   197,   198,   199,   201,   202,   203,   204,
     205,   209,   210,   211,   212,   213,   214,   215,   216,   217,
     218,   219,   220,   221,   222,   223,   224,   225,   228,   229,
     230,   231,   232,   233,   234,   235,   236,   237,   238,   240,
     242,   243,   246,   247,   248,   249,   252,   254,   262,   266,
     273,   275,   280,   281,   285,   287,   289,   291,   293,   305,
     307,   309,   311,   313,   315,   317,   319,   321,   323,   325,
     327,   329,   331,   335,   337,   339,   343,   345,   350,   351,
     356,   357,   361,   363,   365,   367,   369,   371,   373,   375,
     377,   379,   381,   383,   385,   387,   389,   393,   394,   401,
     403,   407,   411,   413,   417,   421,   423,   425,   427,   430,
     432,   436,   438,   442,   446,   451,   452,   456,   460,   465,
     466,   471,   472,   482,   484,   488,   490,   495,   496,   500,
     502,   507,   508,   512,   517,   518,   522,   524,   528,   530,
     535,   536,   540,   541,   544,   548,   550,   554,   556,   561,
     562,   566,   568,   572,   574,   578,   582,   586,   592,   596,
     598,   602,   604,   608,   612,   616,   620,   622,   627,   628,
     633,   634,   636,   640,   642,   644,   648,   650,   654,   658,
     660,   662,   664,   666,   670,   672,   677,   684,   688,   690,
     692,   693,   695,   697,   699,   703,   705,   707,   713,   716,
     721,   723,   725,   731,   739,   741,   744,   748,   751,   755,
     757,   762,   766,   768,   770,   772,   774,   776,   778,   780,
     782,   784,   786,   789,   799,   814,   830,   832,   836,   838,
     843,   844,   846,   850,   852,   856,   858,   860,   862,   864,
     866,   868,   870,   872,   874,   876,   878,   880,   882,   884,
     886,   888,   892,   894,   896,   901,   902,   904,   913,   914,
     916,   922,   928,   934,   942,   949,   957,   964,   966,   968,
     970,   977,   978,   979,   982,   983,   984,   985,   992,   998,
    1007,  1014,  1020,  1026,  1034,  1036,  1040,  1042,  1046,  1048,
    1052,  1054,  1059,  1060,  1065,  1066,  1068,  1072,  1074,  1078,
    1080,  1084,  1086,  1088,  1092,  1095,  1098,  1100,  1102,  1106,
    1108,  1110,  1112,  1114,  1116,  1120,  1122,  1124,  1126,  1128,
    1130,  1132,  1135,  1138,  1140,  1142,  1144,  1146,  1148,  1155,
    1156,  1158,  1162,  1166,  1170,  1174,  1176,  1180,  1182,  1186,
    1189,  1193,  1197
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

#define YYPACT_NINF -518

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-518)))

#define YYTABLE_NINF -310

#define yytable_value_is_error(Yytable_value) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
     106,  1125,  1125,    53,  -518,   106,  -518,  -518,  -518,  -518,
    -518,  -518,  -518,  -518,  -518,  -518,  -518,  -518,  -518,  -518,
    -518,  -518,  -518,  -518,  -518,  -518,  -518,  -518,  -518,  -518,
    -518,  -518,  -518,  -518,  -518,  -518,  -518,  -518,  -518,  -518,
    -518,  -518,  -518,  -518,  -518,  -518,  -518,  -518,  -518,  -518,
    -518,  -518,  -518,    82,    82,  -518,  -518,  -518,   348,  -518,
    -518,  -518,    -2,  1125,   178,  1125,  1125,   175,   672,     5,
     721,   348,  -518,  -518,  -518,   131,    68,   144,  -518,    86,
    -518,  -518,  -518,  -518,   -33,  1145,   145,   145,    -7,   144,
     103,   103,   103,   103,   115,   139,  1125,   180,   183,   348,
    -518,  -518,  -518,  -518,  -518,  -518,  -518,  -518,   596,  -518,
    -518,  -518,  -518,   197,  -518,  -518,  -518,  -518,  -518,  -518,
    -518,  -518,  -518,  -518,  -518,  -518,  -518,  -518,   227,  -518,
      75,  -518,  -518,  -518,  -518,   159,   124,  -518,  -518,   228,
    -518,   144,   348,    86,   235,    91,   -33,   209,   426,  -518,
    1196,   228,   230,   243,  -518,     8,   144,  -518,   144,   144,
     261,   144,   251,  -518,    26,  1125,  1125,  1125,  1125,   915,
     245,   250,   205,  1125,  -518,  -518,  -518,   447,   257,   103,
     103,   103,   103,   245,   139,  -518,  -518,  -518,  -518,  -518,
    -518,  -518,  -518,  -518,  -518,   241,  -518,  -518,  1176,  -518,
    -518,  1125,   259,   283,   -33,   282,   -33,   258,  -518,   266,
     267,   -24,  -518,  -518,  -518,   268,  -518,   -30,    99,   168,
     256,   213,   144,  -518,  -518,   265,   288,   291,   297,   297,
     297,   297,  -518,  1125,   287,   300,   298,   985,  1125,   335,
    1125,  -518,  -518,   309,   318,   321,  1125,   128,  1125,   320,
     319,   197,  1125,  1125,  1125,  1125,  1125,  1125,  -518,  -518,
    -518,  -518,   324,  -518,   339,  -518,   291,  -518,  -518,   327,
     345,   343,   336,   -33,  -518,  1125,  1125,  -518,   342,  -518,
     -33,   145,  1176,   145,   145,  1176,   145,  -518,  -518,    26,
    -518,   144,   198,   198,   198,   198,   346,  -518,   335,  -518,
     297,   297,  -518,   205,   410,   355,   249,  -518,   358,   447,
    -518,  -518,   297,   297,   297,   297,   297,   208,  1176,  -518,
     349,   -33,   282,   -33,   -33,  -518,  -518,   364,  -518,   363,
    -518,   369,   371,   372,   144,   378,   377,  -518,   383,  -518,
    -518,   794,  -518,  -518,  -518,  -518,  -518,  -518,   198,   198,
    -518,  -518,  1196,    10,   399,  1196,  -518,  -518,  -518,  -518,
    -518,   198,   198,   198,   198,   198,  -518,   410,  -518,  1163,
    -518,  -518,  -518,  -518,  -518,   395,  -518,  -518,  -518,   396,
    -518,   134,   397,  -518,   144,   513,   441,   416,  -518,   794,
     771,  -518,  -518,  -518,  1125,  -518,  -518,  -518,  -518,  -518,
    -518,  -518,  -518,   417,  -518,  1125,   -33,   418,   412,  1196,
     145,   145,   145,  -518,  -518,   688,   845,  -518,   197,  -518,
    -518,  -518,   419,   423,    38,   428,  1196,  -518,   422,   433,
     437,   448,  -518,  -518,  -518,  -518,  -518,  -518,  -518,  -518,
    -518,  -518,  -518,  -518,  -518,   446,  -518,   438,  -518,  -518,
     465,   454,   481,   349,  1125,  -518,   482,   495,  -518,  -518,
     225,  -518,  -518,  -518,  -518,  -518,  -518,  -518,  -518,  -518,
     533,  -518,   739,   367,   349,  -518,  -518,  -518,  -518,    86,
    -518,  1125,  -518,  -518,   489,   487,   489,   528,   511,   532,
     489,   514,   107,   -33,  -518,  -518,  -518,   572,   349,  -518,
     -33,   540,   -33,    42,   517,    27,    87,  -518,   523,   -33,
     654,   524,   344,   209,   515,   367,   518,  -518,   526,   519,
     536,  -518,   -33,   528,   221,  -518,   529,   391,   -33,   536,
    -518,  -518,  -518,  -518,  -518,  -518,   539,   654,  -518,  -518,
    -518,  -518,   565,  -518,   252,   523,   -33,   489,  -518,   427,
     363,  -518,  -518,   559,  -518,  -518,   209,   579,  -518,  -518,
    -518,  -518,  -518,  -518,  -518,  1125,   576,   575,   568,   -33,
     582,   -33,   107,  -518,   349,  -518,  -518,   107,   608,   580,
    1196,   762,  -518,   209,   -33,   583,   584,  -518,   585,   587,
    -518,  1125,  1125,   -33,   586,  -518,  1125,   536,   -33,  -518,
     608,   107,  -518,  -518,    77,    12,   578,  1125,  -518,   595,
     588,  -518,   590,  -518,  1125,  1055,   589,  1125,  1125,  -518,
     211,   107,  -518,   -33,  -518,   146,   607,   237,  1125,  -518,
     368,  -518,   593,   536,  -518,  -518,  -518,  -518,  -518,  -518,
     602,   107,  -518,   597,  -518
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
       0,   268,   265,   268,   268,   273,   128,     0,   126,     0,
     123,     0,     0,     0,     0,     0,     0,   140,   161,   160,
     194,   196,   163,   164,   165,   166,   167,   147,     0,     0,
     134,   151,     0,   150,     0,     0,   183,   181,   182,   185,
     187,     0,     0,     0,     0,     0,   176,   150,    87,     0,
      68,   271,   267,   272,   270,     0,   350,   124,   209,     0,
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
       0,     0,     0,   268,   234,   327,   305,   302,     0,   344,
     268,     0,   268,     0,   347,     0,     0,   314,     0,   268,
       0,     0,     0,     0,     0,     0,     0,   342,     0,     0,
       0,   345,   268,     0,     0,   317,     0,     0,   268,     0,
     321,   322,   324,   320,   319,   323,     0,   311,   313,   306,
     308,   337,     0,   233,     0,     0,   268,     0,   343,     0,
       0,   348,   316,     0,   326,   310,     0,     0,   328,   312,
     303,   281,   282,   283,   301,     0,     0,   296,     0,   268,
       0,   268,     0,   335,     0,   318,   325,     0,   339,     0,
       0,     0,   300,     0,   268,     0,     0,   346,     0,     0,
     333,     0,     0,   268,     0,   297,     0,     0,   268,   336,
     339,     0,   340,   284,     0,     0,     0,     0,   235,     0,
       0,   334,     0,   292,     0,     0,     0,     0,     0,   290,
       0,     0,   330,   268,   341,     0,     0,     0,     0,   286,
       0,   293,     0,     0,   289,   288,   287,   285,   291,   329,
       0,     0,   331,     0,   332
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -518,  -518,   666,  -518,  -241,    -1,   -58,   604,   639,   -50,
    -518,  -518,  -518,  -144,  -518,  -187,  -518,    44,   -78,   -72,
     -70,  -518,  -163,   552,   -83,  -518,  -518,   425,  -518,  -518,
     -80,   520,   402,  -518,   -67,   421,  -518,  -518,   535,   409,
    -518,   284,  -518,  -518,  -321,  -518,  -195,   333,  -518,  -518,
    -518,  -131,  -518,  -518,  -518,  -518,  -518,  -518,  -518,   431,
    -518,   415,   650,  -518,  -124,   337,   657,  -518,  -518,   509,
    -518,  -518,  -518,   351,   356,  -518,   328,  -518,   271,  -518,
    -518,   424,   -64,   165,   -63,  -496,  -518,  -518,  -429,  -518,
    -518,  -359,   167,  -437,  -518,  -518,   234,  -494,  -518,   214,
    -518,  -479,  -518,  -464,   150,  -487,  -517,  -443,  -518,   229,
     253,   203,  -518,  -518
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
     480,   271,   517,   406,   493,   209,   407,   566,   604,   594,
     567,   408,   568,   370,   545,   515,   494,   511,   526,   536,
     508,   495,   538,   512,   590,   518,   549,   500,   504,   505,
     280,   377,    73,    74
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[] =
{
      53,    54,   156,   139,    84,   140,    79,   159,   161,   496,
     311,   162,   557,   214,   247,   520,   473,   541,   540,   143,
     147,   127,   529,   145,   166,   167,   168,   351,   483,   157,
     553,   537,   395,   555,   293,   294,   295,   497,    75,   238,
     290,   525,   527,   502,   144,   214,   403,   509,   146,   178,
     428,   496,   256,    55,   277,   351,   223,   215,   537,   570,
     576,   516,    76,   201,    80,    81,   521,   463,   484,   485,
     486,   487,   488,   489,   490,   224,   210,   160,   586,   320,
     609,   113,   207,   588,   585,   573,  -155,   596,   483,   215,
     618,   216,   202,   578,   248,   174,   459,   597,   217,   491,
     218,   219,   524,   221,   571,   348,   349,   612,   483,     1,
       2,   610,   252,   253,   254,   255,   640,   361,   362,   363,
     364,   365,   459,   369,   460,   602,   277,   632,   484,   485,
     486,   487,   488,   489,   490,   192,   633,   587,   332,   193,
     269,   335,   272,   142,   162,   622,   247,   643,   484,   485,
     486,   487,   488,   489,   490,    57,   613,    58,   614,   491,
     144,   615,    83,  -275,   616,   617,   204,   128,   234,   343,
     344,   345,   205,   144,   368,   206,   642,   455,   620,   491,
     281,   157,    83,  -275,    78,   625,   627,   164,  -275,   630,
     129,   130,   131,   132,   133,   134,   135,    78,   144,   169,
     266,   331,   198,   333,   334,  -180,   336,  -180,   144,   325,
     229,   230,   231,   338,   306,   411,   330,   245,   194,   195,
     196,   197,   483,   172,   392,   393,   248,   614,   634,    77,
     615,    78,   296,   616,   617,   175,   234,   398,   399,   400,
     401,   402,   144,   241,   242,   305,   282,   308,    82,   283,
      83,   312,   313,   314,   315,   316,   317,   371,   561,   373,
     374,   177,   484,   485,   486,   487,   488,   489,   490,   394,
     190,   340,   397,   341,   326,   327,   381,  -275,    78,   477,
     478,   366,   300,   367,   301,   208,   405,   144,   149,   150,
     631,   285,   614,   491,   286,   615,    83,   552,   616,   617,
     258,   259,  -275,   357,   358,    78,   338,  -106,   200,   203,
     212,   129,   130,   131,   132,   133,   134,   135,   614,   562,
     563,   615,   636,   213,   616,   617,   405,   220,   222,   237,
     429,   430,   431,   251,   239,   267,   268,   564,   270,   274,
     273,   284,   425,   405,   139,   483,   140,   275,   276,    61,
     288,    -5,    -5,    62,    -5,    -5,    -5,    -5,    -5,    -5,
      -5,    -5,    -5,    -5,    -5,   289,    -5,    -5,   483,   198,
      -5,   291,   297,  -304,  -304,  -304,  -304,   298,  -304,  -304,
    -304,  -304,  -304,   299,   232,   484,   485,   486,   487,   488,
     489,   490,   483,   422,   302,   303,   304,   309,   310,    63,
      64,   318,   479,   321,   424,    65,    66,  -304,   484,   485,
     486,   487,   488,   489,   490,   453,   491,    67,   319,    83,
    -307,   324,   322,    -5,   -62,   323,   277,   351,   483,   513,
     369,   346,   484,   485,   486,   487,   488,   489,   490,   491,
    -304,   355,   492,  -304,   306,   528,   375,   638,   376,   614,
     379,   378,   615,   474,   380,   616,   617,   382,   550,   383,
     384,   565,   149,   491,   556,   246,    83,  -309,   484,   485,
     486,   487,   488,   489,   490,   396,   409,   410,   412,    78,
     498,   385,   569,   149,   150,   129,   130,   131,   132,   133,
     134,   135,   418,   423,   427,   426,   458,   592,   565,   491,
      78,   468,   572,   456,   464,   583,   129,   130,   131,   132,
     133,   134,   135,   462,   414,   465,  -240,  -240,  -240,   466,
    -240,  -240,  -240,   470,  -240,  -240,  -240,  -240,  -240,   606,
     467,   472,  -240,  -240,  -240,  -240,  -240,  -240,  -240,  -240,
    -240,  -240,  -240,  -240,  -240,  -240,  -240,  -240,  -240,  -240,
     471,  -240,  -240,  -240,  -240,  -240,  -240,  -240,  -240,  -240,
    -240,  -240,    -9,  -240,   579,  -240,  -240,   475,   476,   481,
     499,   501,  -240,  -240,  -240,  -240,  -240,  -240,  -240,  -240,
     483,   503,  -240,  -240,  -240,  -240,   506,   507,   483,   510,
     603,   605,   514,   519,   523,   608,   483,   415,    83,   546,
     539,   542,   544,   483,   547,   554,   603,   179,   180,   181,
     182,   183,   184,   603,   603,   558,   629,   603,   548,   560,
     484,   485,   486,   487,   488,   489,   490,   637,   484,   485,
     486,   487,   488,   489,   490,   575,   484,   485,   486,   487,
     488,   489,   490,   484,   485,   486,   487,   488,   489,   490,
     580,   491,   581,   582,   577,   584,   589,   598,   591,   491,
     599,   600,   601,   619,   607,   623,   624,   491,   628,   639,
     621,    56,   100,   644,   491,     1,     2,   641,    85,    86,
      87,    88,    89,    90,    91,    92,    93,    94,    95,   432,
      96,    97,   635,    60,    98,   530,   531,   532,   487,   533,
     534,   535,   211,   328,   257,   350,   240,   347,   461,   433,
     337,   434,   435,   436,   437,   438,   439,   413,   108,   440,
     441,   442,   443,   444,   360,   110,   419,   114,   115,   116,
     117,   287,   118,   119,   120,   121,   122,   445,   446,   359,
     432,   421,   417,   482,   454,   593,   372,    99,   595,   543,
     611,   559,   551,   574,   447,     0,   522,     0,   448,   449,
     433,   123,   434,   435,   436,   437,   438,   439,   561,     0,
     440,   441,   442,   443,   444,     0,     0,   114,   115,   116,
     117,     0,   118,   119,   120,   121,   122,     0,   445,   446,
       0,     0,     0,     0,   124,     0,     0,   125,   149,   150,
      85,    86,    87,    88,    89,     0,     0,     0,     0,   448,
     449,   123,    96,    97,     0,    78,    98,     0,     0,     0,
       0,   129,   130,   131,   132,   133,   134,   135,     0,   562,
     563,     0,     0,     0,   385,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   420,     0,     0,   125,     6,     7,
       8,     0,     9,    10,    11,     0,    12,    13,    14,    15,
      16,     0,     0,     0,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,   128,     0,    34,    35,    36,    37,    38,    39,    40,
      41,    42,    43,    44,     0,    45,     0,    46,   452,     0,
       0,     0,     0,     0,   129,   130,   131,   132,   133,   134,
     135,    48,     0,     0,    49,    50,    51,    52,     6,     7,
       8,     0,     9,    10,    11,     0,    12,    13,    14,    15,
      16,     0,     0,     0,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,     0,     0,    34,    35,    36,    37,    38,    39,    40,
      41,    42,    43,    44,   232,    45,     0,    46,    47,   233,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    48,     0,     0,    49,    50,    51,    52,     6,     7,
       8,     0,     9,    10,    11,     0,    12,    13,    14,    15,
      16,     0,     0,     0,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,     0,     0,    34,    35,    36,    37,    38,    39,    40,
      41,    42,    43,    44,     0,    45,     0,    46,    47,   233,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    48,     0,     0,    49,    50,    51,    52,     6,     7,
       8,     0,     9,    10,    11,     0,    12,    13,    14,    15,
      16,     0,     0,     0,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,     0,     0,    34,    35,    36,    37,    38,    39,    40,
      41,    42,    43,    44,     0,    45,     0,    46,    47,   626,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    48,     0,     0,    49,    50,    51,    52,     6,     7,
       8,     0,     9,    10,    11,     0,    12,    13,    14,    15,
      16,     0,     0,     0,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,   148,     0,    34,    35,    36,    37,    38,    39,    40,
      41,    42,    43,    44,     0,    45,     0,    46,    47,     0,
       0,   149,   150,     0,     0,     0,     0,     0,     0,     0,
       0,    48,     0,     0,    49,    50,    51,    52,    78,   149,
     150,     0,     0,     0,   129,   130,   131,   132,   133,   134,
     135,     0,   149,   150,     0,     0,    78,     0,     0,     0,
       0,     0,   129,   130,   131,   132,   133,   134,   135,    78,
     260,   261,   149,   150,     0,   129,   130,   131,   132,   133,
     134,   135,     0,     0,     0,   404,     0,     0,     0,    78,
       0,     0,     0,     0,     0,   129,   130,   131,   132,   133,
     134,   135
};

static const yytype_int16 yycheck[] =
{
       1,     2,    85,    75,    67,    75,    64,    87,    88,   473,
     251,    89,   529,    37,   177,   502,   453,   513,   512,    77,
      84,    71,   509,    56,    91,    92,    93,    17,     1,    36,
     524,   510,   353,   527,   229,   230,   231,   474,    40,   170,
     227,   505,   506,   486,    74,    37,   367,   490,    81,    99,
     409,   515,   183,     0,    84,    17,    30,    81,   537,   546,
     556,   498,    63,   141,    65,    66,   503,   426,    41,    42,
      43,    44,    45,    46,    47,    49,   148,    84,   572,   266,
     597,    76,   146,   577,   571,   549,    76,   583,     1,    81,
      78,    83,   142,   557,   177,    96,    84,   584,   156,    72,
     158,   159,    75,   161,   547,   300,   301,   601,     1,     3,
       4,   598,   179,   180,   181,   182,   633,   312,   313,   314,
     315,   316,    84,    81,    86,   589,    84,   621,    41,    42,
      43,    44,    45,    46,    47,    60,   623,   574,   282,    64,
     204,   285,   206,    75,   222,   609,   309,   641,    41,    42,
      43,    44,    45,    46,    47,    73,    79,    75,    81,    72,
      74,    84,    75,    56,    87,    88,    75,    36,   169,   293,
     294,   295,    81,    74,   318,    84,   640,   418,   607,    72,
      81,    36,    75,    76,    53,   614,   615,    84,    81,   618,
      59,    60,    61,    62,    63,    64,    65,    53,    74,    84,
     201,   281,    78,   283,   284,    77,   286,    79,    74,   273,
     166,   167,   168,   291,    86,    81,   280,   173,    59,    60,
      61,    62,     1,    84,   348,   349,   309,    81,    82,    51,
      84,    53,   233,    87,    88,    55,   237,   361,   362,   363,
     364,   365,    74,    38,    39,   246,    78,   248,    73,    81,
      75,   252,   253,   254,   255,   256,   257,   321,     6,   323,
     324,    78,    41,    42,    43,    44,    45,    46,    47,   352,
      73,    73,   355,    75,   275,   276,   334,    56,    53,    54,
      55,    73,   238,    75,   240,    76,   369,    74,    36,    37,
      79,    78,    81,    72,    81,    84,    75,    76,    87,    88,
      59,    60,    81,    54,    55,    53,   384,    80,    80,    74,
      80,    59,    60,    61,    62,    63,    64,    65,    81,    67,
      68,    84,    85,    80,    87,    88,   409,    66,    77,    84,
     410,   411,   412,    76,    84,    76,    53,    85,    56,    73,
      82,    85,   406,   426,   416,     1,   416,    80,    80,     1,
      85,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    14,    15,    16,    77,    18,    19,     1,    78,
      22,    74,    85,     6,     7,     8,     9,    77,    11,    12,
      13,    14,    15,    85,    49,    41,    42,    43,    44,    45,
      46,    47,     1,   394,    85,    77,    75,    77,    79,    51,
      52,    77,   460,    76,   405,    57,    58,    40,    41,    42,
      43,    44,    45,    46,    47,   416,    72,    69,    79,    75,
      76,    85,    77,    75,    76,    82,    84,    17,     1,   493,
      81,    85,    41,    42,    43,    44,    45,    46,    47,    72,
      73,    86,    75,    76,    86,   508,    82,    79,    85,    81,
      79,    82,    84,   454,    82,    87,    88,    79,   522,    82,
      77,   544,    36,    72,   528,    18,    75,    76,    41,    42,
      43,    44,    45,    46,    47,    76,    81,    81,    81,    53,
     481,    40,   545,    36,    37,    59,    60,    61,    62,    63,
      64,    65,    76,    76,    82,    77,    73,   580,   581,    72,
      53,    55,    75,    84,    82,   569,    59,    60,    61,    62,
      63,    64,    65,    85,     1,    82,     3,     4,     5,    82,
       7,     8,     9,    85,    11,    12,    13,    14,    15,   593,
      82,    77,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    36,
      85,    38,    39,    40,    41,    42,    43,    44,    45,    46,
      47,    48,    81,    50,   565,    52,    53,    85,    73,    36,
      81,    84,    59,    60,    61,    62,    63,    64,    65,    66,
       1,    53,    69,    70,    71,    72,    75,    55,     1,    75,
     591,   592,    20,    53,    77,   596,     1,    84,    75,    73,
      76,    86,    84,     1,    85,    76,   607,    11,    12,    13,
      14,    15,    16,   614,   615,    76,   617,   618,    82,    54,
      41,    42,    43,    44,    45,    46,    47,   628,    41,    42,
      43,    44,    45,    46,    47,    76,    41,    42,    43,    44,
      45,    46,    47,    41,    42,    43,    44,    45,    46,    47,
      74,    72,    77,    85,    75,    73,    48,    74,    78,    72,
      76,    76,    75,    85,    78,    77,    76,    72,    79,    76,
      75,     5,    68,    76,    72,     3,     4,    75,     6,     7,
       8,     9,    10,    11,    12,    13,    14,    15,    16,     1,
      18,    19,    85,    54,    22,    41,    42,    43,    44,    45,
      46,    47,   150,   278,   184,   303,   171,   298,   424,    21,
     289,    23,    24,    25,    26,    27,    28,   384,    68,    31,
      32,    33,    34,    35,   309,    68,   389,     6,     7,     8,
       9,   222,    11,    12,    13,    14,    15,    49,    50,   308,
       1,   390,   386,   472,   416,   580,   322,    75,   581,   515,
     600,   537,   523,   550,    66,    -1,   503,    -1,    70,    71,
      21,    40,    23,    24,    25,    26,    27,    28,     6,    -1,
      31,    32,    33,    34,    35,    -1,    -1,     6,     7,     8,
       9,    -1,    11,    12,    13,    14,    15,    -1,    49,    50,
      -1,    -1,    -1,    -1,    73,    -1,    -1,    76,    36,    37,
       6,     7,     8,     9,    10,    -1,    -1,    -1,    -1,    70,
      71,    40,    18,    19,    -1,    53,    22,    -1,    -1,    -1,
      -1,    59,    60,    61,    62,    63,    64,    65,    -1,    67,
      68,    -1,    -1,    -1,    40,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    73,    -1,    -1,    76,     3,     4,
       5,    -1,     7,     8,     9,    -1,    11,    12,    13,    14,
      15,    -1,    -1,    -1,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    -1,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    -1,    50,    -1,    52,    53,    -1,
      -1,    -1,    -1,    -1,    59,    60,    61,    62,    63,    64,
      65,    66,    -1,    -1,    69,    70,    71,    72,     3,     4,
       5,    -1,     7,     8,     9,    -1,    11,    12,    13,    14,
      15,    -1,    -1,    -1,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    -1,    -1,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    -1,    52,    53,    54,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    66,    -1,    -1,    69,    70,    71,    72,     3,     4,
       5,    -1,     7,     8,     9,    -1,    11,    12,    13,    14,
      15,    -1,    -1,    -1,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    -1,    -1,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    -1,    50,    -1,    52,    53,    54,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    66,    -1,    -1,    69,    70,    71,    72,     3,     4,
       5,    -1,     7,     8,     9,    -1,    11,    12,    13,    14,
      15,    -1,    -1,    -1,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    -1,    -1,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    -1,    50,    -1,    52,    53,    54,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    66,    -1,    -1,    69,    70,    71,    72,     3,     4,
       5,    -1,     7,     8,     9,    -1,    11,    12,    13,    14,
      15,    -1,    -1,    -1,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    16,    -1,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    -1,    50,    -1,    52,    53,    -1,
      -1,    36,    37,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    66,    -1,    -1,    69,    70,    71,    72,    53,    36,
      37,    -1,    -1,    -1,    59,    60,    61,    62,    63,    64,
      65,    -1,    36,    37,    -1,    -1,    53,    -1,    -1,    -1,
      -1,    -1,    59,    60,    61,    62,    63,    64,    65,    53,
      54,    55,    36,    37,    -1,    59,    60,    61,    62,    63,
      64,    65,    -1,    -1,    -1,    82,    -1,    -1,    -1,    53,
      -1,    -1,    -1,    -1,    -1,    59,    60,    61,    62,    63,
      64,    65
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
     104,    76,    77,    82,    85,   171,    94,    94,   116,   114,
     171,   119,   102,   119,   119,   102,   119,   124,   107,   136,
      73,    75,   153,   153,   153,   153,    85,   128,   135,   135,
     121,    17,   130,   132,   133,    86,   147,    54,    55,   148,
     150,   135,   135,   135,   135,   135,    73,    75,   102,    81,
     182,   171,   170,   171,   171,    82,    85,   200,    82,    79,
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
     196,    84,   196,    53,   197,   198,    75,    55,   189,   196,
      75,   186,   192,   171,    20,   184,   182,   171,   194,    53,
     194,   182,   199,    77,    75,   192,   187,   192,   173,   194,
      41,    42,    43,    45,    46,    47,   188,   190,   191,    76,
     186,   174,    86,   185,    84,   183,    73,    85,    82,   195,
     171,   198,    76,   186,    76,   186,   171,   195,    76,   188,
      54,     6,    67,    68,    85,   113,   176,   179,   181,   173,
     194,   196,    75,   192,   200,    76,   174,    75,   192,    94,
      74,    77,    85,   171,    73,   194,   186,   182,   186,    48,
     193,    78,   113,   172,   178,   181,   174,   194,    74,    76,
      76,    75,   192,    94,   177,    94,   171,    78,    94,   195,
     194,   193,   186,    79,    81,    84,    87,    88,    78,    85,
     177,    75,   192,    77,    76,   177,    54,   177,    79,    94,
     177,    79,   186,   194,    82,    85,    85,    94,    79,    76,
     195,    75,   192,   186,    76
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
#line 167 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.modlist) = (yyvsp[0].modlist); modlist = (yyvsp[0].modlist); }
#line 1946 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 3:
#line 171 "xi-grammar.y" /* yacc.c:1646  */
    { 
		  (yyval.modlist) = 0; 
		}
#line 1954 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 4:
#line 175 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.modlist) = new AstChildren<Module>(lineno, (yyvsp[-1].module), (yyvsp[0].modlist)); }
#line 1960 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 5:
#line 179 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 1966 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 6:
#line 181 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 1972 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 7:
#line 185 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 1978 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 8:
#line 187 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 1984 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 9:
#line 192 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 1990 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 10:
#line 193 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MODULE); YYABORT; }
#line 1996 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 11:
#line 194 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MAINMODULE); YYABORT; }
#line 2002 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 12:
#line 195 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(EXTERN); YYABORT; }
#line 2008 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 13:
#line 197 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITCALL); YYABORT; }
#line 2014 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 14:
#line 198 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITNODE); YYABORT; }
#line 2020 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 15:
#line 199 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INITPROC); YYABORT; }
#line 2026 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 16:
#line 201 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CHARE); }
#line 2032 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 17:
#line 202 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MAINCHARE); }
#line 2038 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 18:
#line 203 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(GROUP); }
#line 2044 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 19:
#line 204 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NODEGROUP); }
#line 2050 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 20:
#line 205 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ARRAY); }
#line 2056 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 21:
#line 209 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INCLUDE); YYABORT; }
#line 2062 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 22:
#line 210 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(STACKSIZE); YYABORT; }
#line 2068 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 23:
#line 211 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(THREADED); YYABORT; }
#line 2074 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 24:
#line 212 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(TEMPLATE); YYABORT; }
#line 2080 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 25:
#line 213 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(SYNC); YYABORT; }
#line 2086 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 26:
#line 214 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IGET); YYABORT; }
#line 2092 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 27:
#line 215 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(EXCLUSIVE); YYABORT; }
#line 2098 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 28:
#line 216 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IMMEDIATE); YYABORT; }
#line 2104 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 29:
#line 217 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(SKIPSCHED); YYABORT; }
#line 2110 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 30:
#line 218 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(INLINE); YYABORT; }
#line 2116 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 31:
#line 219 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(VIRTUAL); YYABORT; }
#line 2122 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 32:
#line 220 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MIGRATABLE); YYABORT; }
#line 2128 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 33:
#line 221 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CREATEHERE); YYABORT; }
#line 2134 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 34:
#line 222 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CREATEHOME); YYABORT; }
#line 2140 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 35:
#line 223 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOKEEP); YYABORT; }
#line 2146 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 36:
#line 224 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(NOTRACE); YYABORT; }
#line 2152 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 37:
#line 225 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(APPWORK); YYABORT; }
#line 2158 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 38:
#line 228 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(PACKED); YYABORT; }
#line 2164 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 39:
#line 229 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(VARSIZE); YYABORT; }
#line 2170 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 40:
#line 230 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ENTRY); YYABORT; }
#line 2176 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 41:
#line 231 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(FOR); YYABORT; }
#line 2182 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 42:
#line 232 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(FORALL); YYABORT; }
#line 2188 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 43:
#line 233 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(WHILE); YYABORT; }
#line 2194 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 44:
#line 234 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(WHEN); YYABORT; }
#line 2200 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 45:
#line 235 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(OVERLAP); YYABORT; }
#line 2206 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 46:
#line 236 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ATOMIC); YYABORT; }
#line 2212 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 47:
#line 237 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(IF); YYABORT; }
#line 2218 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 48:
#line 238 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ELSE); YYABORT; }
#line 2224 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 49:
#line 240 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(LOCAL); YYABORT; }
#line 2230 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 50:
#line 242 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(USING); YYABORT; }
#line 2236 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 51:
#line 243 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ACCEL); YYABORT; }
#line 2242 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 52:
#line 246 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(ACCELBLOCK); YYABORT; }
#line 2248 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 53:
#line 247 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(MEMCRITICAL); YYABORT; }
#line 2254 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 54:
#line 248 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(REDUCTIONTARGET); YYABORT; }
#line 2260 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 55:
#line 249 "xi-grammar.y" /* yacc.c:1646  */
    { ReservedWord(CASE); YYABORT; }
#line 2266 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 56:
#line 253 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 2272 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 57:
#line 255 "xi-grammar.y" /* yacc.c:1646  */
    {
		  char *tmp = new char[strlen((yyvsp[-3].strval))+strlen((yyvsp[0].strval))+3];
		  sprintf(tmp,"%s::%s", (yyvsp[-3].strval), (yyvsp[0].strval));
		  (yyval.strval) = tmp;
		}
#line 2282 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 58:
#line 263 "xi-grammar.y" /* yacc.c:1646  */
    { 
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist)); 
		}
#line 2290 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 59:
#line 267 "xi-grammar.y" /* yacc.c:1646  */
    {  
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist)); 
		    (yyval.module)->setMain();
		}
#line 2299 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 60:
#line 274 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = 0; }
#line 2305 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 61:
#line 276 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = (yyvsp[-2].conslist); }
#line 2311 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 62:
#line 280 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = 0; }
#line 2317 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 63:
#line 282 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.conslist) = new ConstructList(lineno, (yyvsp[-1].construct), (yyvsp[0].conslist)); }
#line 2323 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 64:
#line 286 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new UsingScope((yyvsp[0].strval), false); }
#line 2329 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 65:
#line 288 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new UsingScope((yyvsp[0].strval), true); }
#line 2335 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 66:
#line 290 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].member)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].member); }
#line 2341 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 67:
#line 292 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].message)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].message); }
#line 2347 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 68:
#line 294 "xi-grammar.y" /* yacc.c:1646  */
    {
                  Entry *e = new Entry(lineno, 0, (yyvsp[-4].type), (yyvsp[-2].strval), (yyvsp[0].plist), 0, 0, 0);
                  int isExtern = 1;
                  e->setExtern(isExtern);
                  e->targs = (yyvsp[-1].tparlist);
                  e->label = new XStr;
                  (yyvsp[-3].ntype)->print(*e->label);
                  (yyval.construct) = e;
                }
#line 2361 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 69:
#line 306 "xi-grammar.y" /* yacc.c:1646  */
    { if((yyvsp[-2].conslist)) (yyvsp[-2].conslist)->recurse<int&>((yyvsp[-4].intval), &Construct::setExtern); (yyval.construct) = (yyvsp[-2].conslist); }
#line 2367 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 70:
#line 308 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = new Scope((yyvsp[-3].strval), (yyvsp[-1].conslist)); }
#line 2373 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 71:
#line 310 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = (yyvsp[-1].construct); }
#line 2379 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 72:
#line 312 "xi-grammar.y" /* yacc.c:1646  */
    { yyerror("The preceding construct must be semicolon terminated"); YYABORT; }
#line 2385 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 73:
#line 314 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].module)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].module); }
#line 2391 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 74:
#line 316 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2397 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 75:
#line 318 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2403 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 76:
#line 320 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2409 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 77:
#line 322 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
#line 2415 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 78:
#line 324 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[0].templat)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].templat); }
#line 2421 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 79:
#line 326 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = NULL; }
#line 2427 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 80:
#line 328 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = NULL; }
#line 2433 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 81:
#line 330 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.construct) = (yyvsp[0].accelBlock); }
#line 2439 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 82:
#line 332 "xi-grammar.y" /* yacc.c:1646  */
    { printf("Invalid construct\n"); YYABORT; }
#line 2445 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 83:
#line 336 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamType((yyvsp[0].type)); }
#line 2451 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 84:
#line 338 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
#line 2457 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 85:
#line 340 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
#line 2463 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 86:
#line 344 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = new TParamList((yyvsp[0].tparam)); }
#line 2469 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 87:
#line 346 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = new TParamList((yyvsp[-2].tparam), (yyvsp[0].tparlist)); }
#line 2475 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 88:
#line 350 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = 0; }
#line 2481 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 89:
#line 352 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = (yyvsp[0].tparlist); }
#line 2487 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 90:
#line 356 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = 0; }
#line 2493 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 91:
#line 358 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tparlist) = (yyvsp[-1].tparlist); }
#line 2499 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 92:
#line 362 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("int"); }
#line 2505 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 93:
#line 364 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long"); }
#line 2511 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 94:
#line 366 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("short"); }
#line 2517 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 95:
#line 368 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("char"); }
#line 2523 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 96:
#line 370 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned int"); }
#line 2529 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 97:
#line 372 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long"); }
#line 2535 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 98:
#line 374 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long"); }
#line 2541 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 99:
#line 376 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned long long"); }
#line 2547 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 100:
#line 378 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned short"); }
#line 2553 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 101:
#line 380 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("unsigned char"); }
#line 2559 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 102:
#line 382 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long long"); }
#line 2565 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 103:
#line 384 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("float"); }
#line 2571 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 104:
#line 386 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("double"); }
#line 2577 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 105:
#line 388 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("long double"); }
#line 2583 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 106:
#line 390 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("void"); }
#line 2589 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 107:
#line 393 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ntype) = new NamedType((yyvsp[-1].strval),(yyvsp[0].tparlist)); }
#line 2595 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 108:
#line 394 "xi-grammar.y" /* yacc.c:1646  */
    { 
                    const char* basename, *scope;
                    splitScopedName((yyvsp[-1].strval), &scope, &basename);
                    (yyval.ntype) = new NamedType(basename, (yyvsp[0].tparlist), scope);
                }
#line 2605 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 109:
#line 402 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 2611 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 110:
#line 404 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ntype); }
#line 2617 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 111:
#line 408 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ptype) = new PtrType((yyvsp[-1].type)); }
#line 2623 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 112:
#line 412 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
#line 2629 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 113:
#line 414 "xi-grammar.y" /* yacc.c:1646  */
    { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
#line 2635 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 114:
#line 418 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ftype) = new FuncType((yyvsp[-7].type), (yyvsp[-4].strval), (yyvsp[-1].plist)); }
#line 2641 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 115:
#line 422 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 2647 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 116:
#line 424 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 2653 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 117:
#line 426 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 2659 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 118:
#line 428 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ftype); }
#line 2665 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 119:
#line 431 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[0].type)); }
#line 2671 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 120:
#line 433 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ConstType((yyvsp[-1].type)); }
#line 2677 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 121:
#line 437 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new ReferenceType((yyvsp[-1].type)); }
#line 2683 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 122:
#line 439 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 2689 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 123:
#line 443 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 2695 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 124:
#line 447 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = (yyvsp[-1].val); }
#line 2701 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 125:
#line 451 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.vallist) = 0; }
#line 2707 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 126:
#line 453 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.vallist) = new ValueList((yyvsp[-1].val), (yyvsp[0].vallist)); }
#line 2713 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 127:
#line 457 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].vallist)); }
#line 2719 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 128:
#line 461 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[-2].type), (yyvsp[0].strval), 0, 1); }
#line 2725 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 129:
#line 465 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0;}
#line 2731 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 130:
#line 467 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0;}
#line 2737 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 131:
#line 471 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 2743 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 132:
#line 473 "xi-grammar.y" /* yacc.c:1646  */
    { 
		  /*
		  printf("Warning: Message attributes are being phased out.\n");
		  printf("Warning: Please remove them from interface files.\n");
		  */
		  (yyval.intval) = (yyvsp[-1].intval); 
		}
#line 2755 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 133:
#line 483 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[0].intval); }
#line 2761 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 134:
#line 485 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[-2].intval) | (yyvsp[0].intval); }
#line 2767 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 135:
#line 489 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 2773 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 136:
#line 491 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 2779 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 137:
#line 495 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = 0; }
#line 2785 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 138:
#line 497 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-1].cattr); }
#line 2791 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 139:
#line 501 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[0].cattr); }
#line 2797 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 140:
#line 503 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
#line 2803 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 141:
#line 507 "xi-grammar.y" /* yacc.c:1646  */
    { python_doc = NULL; (yyval.intval) = 0; }
#line 2809 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 142:
#line 509 "xi-grammar.y" /* yacc.c:1646  */
    { python_doc = (yyvsp[0].strval); (yyval.intval) = 0; }
#line 2815 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 143:
#line 513 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CPYTHON; }
#line 2821 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 144:
#line 517 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = 0; }
#line 2827 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 145:
#line 519 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-1].cattr); }
#line 2833 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 146:
#line 523 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[0].cattr); }
#line 2839 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 147:
#line 525 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
#line 2845 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 148:
#line 529 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CMIGRATABLE; }
#line 2851 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 149:
#line 531 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.cattr) = Chare::CPYTHON; }
#line 2857 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 150:
#line 535 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 2863 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 151:
#line 537 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 2869 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 152:
#line 540 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 2875 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 153:
#line 542 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 1; }
#line 2881 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 154:
#line 545 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mv) = new MsgVar((yyvsp[-3].type), (yyvsp[-2].strval), (yyvsp[-4].intval), (yyvsp[-1].intval)); }
#line 2887 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 155:
#line 549 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mvlist) = new MsgVarList((yyvsp[0].mv)); }
#line 2893 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 156:
#line 551 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mvlist) = new MsgVarList((yyvsp[-1].mv), (yyvsp[0].mvlist)); }
#line 2899 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 157:
#line 555 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, (yyvsp[0].ntype)); }
#line 2905 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 158:
#line 557 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, (yyvsp[-3].ntype), (yyvsp[-1].mvlist)); }
#line 2911 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 159:
#line 561 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = 0; }
#line 2917 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 160:
#line 563 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = (yyvsp[0].typelist); }
#line 2923 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 161:
#line 567 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = new TypeList((yyvsp[0].ntype)); }
#line 2929 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 162:
#line 569 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.typelist) = new TypeList((yyvsp[-2].ntype), (yyvsp[0].typelist)); }
#line 2935 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 163:
#line 573 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr)|Chare::CCHARE, (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 2941 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 164:
#line 575 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 2947 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 165:
#line 579 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 2953 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 166:
#line 583 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new NodeGroup(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 2959 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 167:
#line 587 "xi-grammar.y" /* yacc.c:1646  */
    {/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",(yyvsp[-2].strval));
			(yyval.ntype) = new NamedType(buf); 
		}
#line 2969 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 168:
#line 593 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.ntype) = new NamedType((yyvsp[-1].strval)); }
#line 2975 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 169:
#line 597 "xi-grammar.y" /* yacc.c:1646  */
    {  (yyval.chare) = new Array(lineno, (yyvsp[-4].cattr), (yyvsp[-3].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 2981 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 170:
#line 599 "xi-grammar.y" /* yacc.c:1646  */
    {  (yyval.chare) = new Array(lineno, (yyvsp[-3].cattr), (yyvsp[-4].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 2987 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 171:
#line 603 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr)|Chare::CCHARE, new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist));}
#line 2993 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 172:
#line 605 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 2999 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 173:
#line 609 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3005 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 174:
#line 613 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new NodeGroup( lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3011 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 175:
#line 617 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.chare) = new Array( lineno, 0, (yyvsp[-3].ntype), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
#line 3017 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 176:
#line 621 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-1].strval))); }
#line 3023 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 177:
#line 623 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-4].strval)), (yyvsp[-2].mvlist)); }
#line 3029 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 178:
#line 627 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = 0; }
#line 3035 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 179:
#line 629 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].type); }
#line 3041 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 180:
#line 633 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = 0; }
#line 3047 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 181:
#line 635 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3053 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 182:
#line 637 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3059 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 183:
#line 641 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TType(new NamedType((yyvsp[-1].strval)), (yyvsp[0].type)); }
#line 3065 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 184:
#line 643 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TFunc((yyvsp[-1].ftype), (yyvsp[0].strval)); }
#line 3071 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 185:
#line 645 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvar) = new TName((yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].strval)); }
#line 3077 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 186:
#line 649 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = new TVarList((yyvsp[0].tvar)); }
#line 3083 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 187:
#line 651 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = new TVarList((yyvsp[-2].tvar), (yyvsp[0].tvarlist)); }
#line 3089 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 188:
#line 655 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.tvarlist) = (yyvsp[-1].tvarlist); }
#line 3095 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 189:
#line 659 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3101 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 190:
#line 661 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3107 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 191:
#line 663 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3113 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 192:
#line 665 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
#line 3119 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 193:
#line 667 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].message)); (yyvsp[0].message)->setTemplate((yyval.templat)); }
#line 3125 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 194:
#line 671 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = 0; }
#line 3131 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 195:
#line 673 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = (yyvsp[-2].mbrlist); }
#line 3137 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 196:
#line 677 "xi-grammar.y" /* yacc.c:1646  */
    { 
                  if (!connectEntries.empty()) {
                    (yyval.mbrlist) = new AstChildren<Member>(connectEntries);
		  } else {
		    (yyval.mbrlist) = 0; 
                  }
		}
#line 3149 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 197:
#line 685 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.mbrlist) = new AstChildren<Member>(-1, (yyvsp[-1].member), (yyvsp[0].mbrlist)); }
#line 3155 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 198:
#line 689 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].readonly); }
#line 3161 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 199:
#line 691 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].readonly); }
#line 3167 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 201:
#line 694 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].member); }
#line 3173 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 202:
#line 696 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].pupable); }
#line 3179 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 203:
#line 698 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].includeFile); }
#line 3185 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 204:
#line 700 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new ClassDeclaration(lineno,(yyvsp[0].strval)); }
#line 3191 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 205:
#line 704 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1); }
#line 3197 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 206:
#line 706 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1); }
#line 3203 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 207:
#line 708 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[-6].strval)) + '<' +
					    ((yyvsp[-4].tparlist))->to_string() + '>').c_str()),
				    1);
		}
#line 3213 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 208:
#line 714 "xi-grammar.y" /* yacc.c:1646  */
    { printf("Warning: deprecated use of initcall. Use initnode or initproc instead.\n"); 
		  (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1); }
#line 3220 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 209:
#line 717 "xi-grammar.y" /* yacc.c:1646  */
    { printf("Warning: deprecated use of initcall. Use initnode or initproc instead.\n");
		  (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1); }
#line 3227 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 210:
#line 722 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 0); }
#line 3233 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 211:
#line 724 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 0); }
#line 3239 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 212:
#line 726 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[-6].strval)) + '<' +
					    ((yyvsp[-4].tparlist))->to_string() + '>').c_str()),
				    0);
		}
#line 3249 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 213:
#line 732 "xi-grammar.y" /* yacc.c:1646  */
    {
                  InitCall* rtn = new InitCall(lineno, (yyvsp[-3].strval), 0);
                  rtn->setAccel();
                  (yyval.member) = rtn;
		}
#line 3259 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 214:
#line 740 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[0].ntype),0); }
#line 3265 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 215:
#line 742 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[-2].ntype),(yyvsp[0].pupable)); }
#line 3271 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 216:
#line 745 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.includeFile) = new IncludeFile(lineno,(yyvsp[0].strval)); }
#line 3277 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 217:
#line 749 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[-1].member); }
#line 3283 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 218:
#line 752 "xi-grammar.y" /* yacc.c:1646  */
    { yyerror("The preceding entry method declaration must be semicolon-terminated."); YYABORT; }
#line 3289 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 219:
#line 756 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].entry); }
#line 3295 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 220:
#line 758 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyvsp[0].entry)->tspec = (yyvsp[-1].tvarlist);
                  (yyval.member) = (yyvsp[0].entry);
                }
#line 3304 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 221:
#line 763 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = (yyvsp[0].member); }
#line 3310 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 222:
#line 767 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3316 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 223:
#line 769 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3322 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 224:
#line 771 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3328 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 225:
#line 773 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3334 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 226:
#line 775 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3340 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 227:
#line 777 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3346 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 228:
#line 779 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3352 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 229:
#line 781 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3358 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 230:
#line 783 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3364 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 231:
#line 785 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3370 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 232:
#line 787 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.member) = 0; }
#line 3376 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 233:
#line 790 "xi-grammar.y" /* yacc.c:1646  */
    { 
                  (yyval.entry) = new Entry(lineno, (yyvsp[-5].intval), (yyvsp[-4].type), (yyvsp[-3].strval), (yyvsp[-2].plist), (yyvsp[-1].val), (yyvsp[0].sentry)); 
		  if ((yyvsp[0].sentry) != 0) { 
		    (yyvsp[0].sentry)->con1 = new SdagConstruct(SIDENT, (yyvsp[-3].strval));
                    (yyvsp[0].sentry)->entry = (yyval.entry);
                    (yyvsp[0].sentry)->con1->entry = (yyval.entry);
                    (yyvsp[0].sentry)->param = new ParamList((yyvsp[-2].plist));
                  }
		}
#line 3390 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 234:
#line 800 "xi-grammar.y" /* yacc.c:1646  */
    { 
                  Entry *e = new Entry(lineno, (yyvsp[-3].intval), 0, (yyvsp[-2].strval), (yyvsp[-1].plist),  0, (yyvsp[0].sentry));
                  if ((yyvsp[0].sentry) != 0) {
		    (yyvsp[0].sentry)->con1 = new SdagConstruct(SIDENT, (yyvsp[-2].strval));
                    (yyvsp[0].sentry)->entry = e;
                    (yyvsp[0].sentry)->con1->entry = e;
                    (yyvsp[0].sentry)->param = new ParamList((yyvsp[-1].plist));
                  }
		  if (e->param && e->param->isCkMigMsgPtr()) {
		    yyerror("Charm++ takes a CkMigrateMsg chare constructor for granted, but continuing anyway");
		    (yyval.entry) = NULL;
		  } else
		    (yyval.entry) = e;
		}
#line 3409 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 235:
#line 815 "xi-grammar.y" /* yacc.c:1646  */
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
#line 3427 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 236:
#line 831 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.accelBlock) = new AccelBlock(lineno, new XStr((yyvsp[-2].strval))); }
#line 3433 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 237:
#line 833 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.accelBlock) = new AccelBlock(lineno, NULL); }
#line 3439 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 238:
#line 837 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = new BuiltinType("void"); }
#line 3445 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 239:
#line 839 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.type) = (yyvsp[0].ptype); }
#line 3451 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 240:
#line 843 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = 0; }
#line 3457 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 241:
#line 845 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[-1].intval); }
#line 3463 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 242:
#line 847 "xi-grammar.y" /* yacc.c:1646  */
    { printf("Invalid entry method attribute list\n"); YYABORT; }
#line 3469 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 243:
#line 851 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[0].intval); }
#line 3475 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 244:
#line 853 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = (yyvsp[-2].intval) | (yyvsp[0].intval); }
#line 3481 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 245:
#line 857 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = STHREADED; }
#line 3487 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 246:
#line 859 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SSYNC; }
#line 3493 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 247:
#line 861 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SIGET; }
#line 3499 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 248:
#line 863 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SLOCKED; }
#line 3505 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 249:
#line 865 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SCREATEHERE; }
#line 3511 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 250:
#line 867 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SCREATEHOME; }
#line 3517 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 251:
#line 869 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SNOKEEP; }
#line 3523 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 252:
#line 871 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SNOTRACE; }
#line 3529 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 253:
#line 873 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SAPPWORK; }
#line 3535 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 254:
#line 875 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SIMMEDIATE; }
#line 3541 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 255:
#line 877 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SSKIPSCHED; }
#line 3547 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 256:
#line 879 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SINLINE; }
#line 3553 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 257:
#line 881 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SLOCAL; }
#line 3559 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 258:
#line 883 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SPYTHON; }
#line 3565 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 259:
#line 885 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SMEM; }
#line 3571 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 260:
#line 887 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = SREDUCE; }
#line 3577 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 261:
#line 889 "xi-grammar.y" /* yacc.c:1646  */
    { printf("Invalid entry method attribute: %s\n", yylval); YYABORT; }
#line 3583 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 262:
#line 893 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 3589 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 263:
#line 895 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 3595 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 264:
#line 897 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 3601 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 265:
#line 901 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = ""; }
#line 3607 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 266:
#line 903 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3613 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 267:
#line 905 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s, %s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 3623 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 268:
#line 913 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = ""; }
#line 3629 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 269:
#line 915 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3635 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 270:
#line 917 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s[%s]%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 3645 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 271:
#line 923 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s{%s}%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 3655 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 272:
#line 929 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s(%s)%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 3665 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 273:
#line 935 "xi-grammar.y" /* yacc.c:1646  */
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"(%s)%s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
#line 3675 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 274:
#line 943 "xi-grammar.y" /* yacc.c:1646  */
    {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval));
		}
#line 3684 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 275:
#line 950 "xi-grammar.y" /* yacc.c:1646  */
    { 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			(yyval.intval) = 0;
		}
#line 3694 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 276:
#line 958 "xi-grammar.y" /* yacc.c:1646  */
    { 
			in_braces=0;
			(yyval.intval) = 0;
		}
#line 3703 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 277:
#line 965 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[0].type));}
#line 3709 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 278:
#line 967 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval)); (yyval.pname)->setConditional((yyvsp[0].intval)); }
#line 3715 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 279:
#line 969 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.pname) = new Parameter(lineno, (yyvsp[-3].type),(yyvsp[-2].strval),0,(yyvsp[0].val));}
#line 3721 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 280:
#line 971 "xi-grammar.y" /* yacc.c:1646  */
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
		}
#line 3730 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 281:
#line 977 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READONLY; }
#line 3736 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 282:
#line 978 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READWRITE; }
#line 3742 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 283:
#line 979 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_WRITEONLY; }
#line 3748 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 284:
#line 982 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr((yyvsp[0].strval)); }
#line 3754 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 285:
#line 983 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "->" << (yyvsp[0].strval); }
#line 3760 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 286:
#line 984 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[-2].xstrptr)) << "." << (yyvsp[0].strval); }
#line 3766 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 287:
#line 986 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << *((yyvsp[-1].xstrptr)) << "]";
                  delete (yyvsp[-3].xstrptr);
                  delete (yyvsp[-1].xstrptr);
                }
#line 3777 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 288:
#line 993 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "[" << (yyvsp[-1].strval) << "]";
                  delete (yyvsp[-3].xstrptr);
                }
#line 3787 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 289:
#line 999 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[-3].xstrptr)) << "(" << *((yyvsp[-1].xstrptr)) << ")";
                  delete (yyvsp[-3].xstrptr);
                  delete (yyvsp[-1].xstrptr);
                }
#line 3798 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 290:
#line 1008 "xi-grammar.y" /* yacc.c:1646  */
    {
                  in_bracket = 0;
                  (yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName(), (yyvsp[-1].strval));
                }
#line 3807 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 291:
#line 1015 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[-6].intval));
                }
#line 3817 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 292:
#line 1021 "xi-grammar.y" /* yacc.c:1646  */
    {
		  (yyval.pname) = new Parameter(lineno, (yyvsp[-4].type), (yyvsp[-3].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType(Parameter::ACCEL_BUFFER_TYPE_READWRITE);
		}
#line 3827 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 293:
#line 1027 "xi-grammar.y" /* yacc.c:1646  */
    {
                  (yyval.pname) = (yyvsp[-3].pname);
                  (yyval.pname)->setAccelInstName((yyvsp[-1].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[-5].intval));
		}
#line 3837 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 294:
#line 1035 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 3843 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 295:
#line 1037 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 3849 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 296:
#line 1041 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
#line 3855 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 297:
#line 1043 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
#line 3861 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 298:
#line 1047 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = (yyvsp[-1].plist); }
#line 3867 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 299:
#line 1049 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = new ParamList(new Parameter(0, new BuiltinType("void"))); }
#line 3873 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 300:
#line 1053 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = (yyvsp[-1].plist); }
#line 3879 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 301:
#line 1055 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.plist) = 0; }
#line 3885 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 302:
#line 1059 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = 0; }
#line 3891 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 303:
#line 1061 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.val) = new Value((yyvsp[0].strval)); }
#line 3897 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 304:
#line 1065 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = 0; }
#line 3903 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 305:
#line 1067 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[0].sc)); }
#line 3909 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 306:
#line 1069 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[-1].slist)); }
#line 3915 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 307:
#line 1073 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.slist) = new SListConstruct((yyvsp[0].sc)); }
#line 3921 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 308:
#line 1075 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.slist) = new SListConstruct((yyvsp[-1].sc), (yyvsp[0].slist));  }
#line 3927 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 309:
#line 1079 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.olist) = new OListConstruct((yyvsp[0].sc)); }
#line 3933 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 310:
#line 1081 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.olist) = new OListConstruct((yyvsp[-1].sc), (yyvsp[0].slist)); }
#line 3939 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 311:
#line 1085 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.clist) = new CaseListConstruct((yyvsp[0].when)); }
#line 3945 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 312:
#line 1087 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.clist) = new CaseListConstruct((yyvsp[-1].when), (yyvsp[0].clist)); }
#line 3951 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 313:
#line 1089 "xi-grammar.y" /* yacc.c:1646  */
    { yyerror("Case blocks in SDAG can only contain when clauses."); YYABORT; }
#line 3957 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 314:
#line 1093 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = (yyvsp[0].strval); }
#line 3963 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 315:
#line 1095 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.strval) = 0; }
#line 3969 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 316:
#line 1099 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-2].entrylist), 0); }
#line 3975 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 317:
#line 1101 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-1].entrylist), (yyvsp[0].sc)); }
#line 3981 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 318:
#line 1103 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = new WhenConstruct((yyvsp[-3].entrylist), (yyvsp[-1].slist)); }
#line 3987 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 319:
#line 1107 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 3993 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 320:
#line 1109 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 3999 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 321:
#line 1111 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4005 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 322:
#line 1113 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4011 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 323:
#line 1115 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4017 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 324:
#line 1117 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.when) = 0; }
#line 4023 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 325:
#line 1121 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new AtomicConstruct((yyvsp[-1].strval), (yyvsp[-3].strval)); }
#line 4029 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 326:
#line 1123 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new OverlapConstruct((yyvsp[-1].olist)); }
#line 4035 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 327:
#line 1125 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = (yyvsp[0].when); }
#line 4041 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 328:
#line 1127 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new CaseConstruct((yyvsp[-1].clist)); }
#line 4047 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 329:
#line 1129 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForConstruct((yyvsp[-8].intexpr), (yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4053 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 330:
#line 1131 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForConstruct((yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4059 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 331:
#line 1133 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-9].strval)), (yyvsp[-6].intexpr),
		             (yyvsp[-4].intexpr), (yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4066 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 332:
#line 1136 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[-11].strval)), (yyvsp[-8].intexpr),
		             (yyvsp[-6].intexpr), (yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4073 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 333:
#line 1139 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new IfConstruct((yyvsp[-3].intexpr), (yyvsp[-1].sc), (yyvsp[0].sc)); }
#line 4079 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 334:
#line 1141 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new IfConstruct((yyvsp[-5].intexpr), (yyvsp[-2].slist), (yyvsp[0].sc)); }
#line 4085 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 335:
#line 1143 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new WhileConstruct((yyvsp[-2].intexpr), (yyvsp[0].sc)); }
#line 4091 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 336:
#line 1145 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new WhileConstruct((yyvsp[-4].intexpr), (yyvsp[-1].slist)); }
#line 4097 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 337:
#line 1147 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new AtomicConstruct((yyvsp[-1].strval), NULL); }
#line 4103 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 338:
#line 1149 "xi-grammar.y" /* yacc.c:1646  */
    { printf("Unknown SDAG construct or malformed entry method definition.\n"
				 "You may have forgotten to terminate an entry method definition with a"
				 " semicolon or forgotten to mark a block of sequential SDAG code as 'atomic'\n"); YYABORT; }
#line 4111 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 339:
#line 1155 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = 0; }
#line 4117 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 340:
#line 1157 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ElseConstruct((yyvsp[0].sc)); }
#line 4123 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 341:
#line 1159 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.sc) = new ElseConstruct((yyvsp[-1].slist)); }
#line 4129 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 342:
#line 1163 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.intexpr) = new IntExprConstruct((yyvsp[0].strval)); }
#line 4135 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 343:
#line 1167 "xi-grammar.y" /* yacc.c:1646  */
    { in_int_expr = 0; (yyval.intval) = 0; }
#line 4141 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 344:
#line 1171 "xi-grammar.y" /* yacc.c:1646  */
    { in_int_expr = 1; (yyval.intval) = 0; }
#line 4147 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 345:
#line 1175 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[-1].strval), (yyvsp[0].plist), 0, 0, 0); }
#line 4153 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 346:
#line 1177 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[-4].strval), (yyvsp[0].plist), 0, 0, (yyvsp[-2].strval)); }
#line 4159 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 347:
#line 1181 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.entrylist) = new EntryList((yyvsp[0].entry)); }
#line 4165 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 348:
#line 1183 "xi-grammar.y" /* yacc.c:1646  */
    { (yyval.entrylist) = new EntryList((yyvsp[-2].entry),(yyvsp[0].entrylist)); }
#line 4171 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 349:
#line 1187 "xi-grammar.y" /* yacc.c:1646  */
    { in_bracket=1; }
#line 4177 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 350:
#line 1190 "xi-grammar.y" /* yacc.c:1646  */
    { in_bracket=0; }
#line 4183 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 351:
#line 1194 "xi-grammar.y" /* yacc.c:1646  */
    { if (!macroDefined((yyvsp[0].strval), 1)) in_comment = 1; }
#line 4189 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;

  case 352:
#line 1198 "xi-grammar.y" /* yacc.c:1646  */
    { if (!macroDefined((yyvsp[0].strval), 0)) in_comment = 1; }
#line 4195 "xi-grammar.tab.c" /* yacc.c:1646  */
    break;


#line 4199 "xi-grammar.tab.c" /* yacc.c:1646  */
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
#line 1201 "xi-grammar.y" /* yacc.c:1906  */

void yyerror(const char *mesg)
{
    std::cerr << cur_file<<":"<<lineno<<": Charmxi syntax error> "
	      << mesg << std::endl;
}
