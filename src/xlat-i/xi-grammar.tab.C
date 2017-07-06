/* A Bison parser, made by GNU Bison 2.5.  */

/* Bison implementation for Yacc-like parsers in C
   
      Copyright (C) 1984, 1989-1990, 2000-2011 Free Software Foundation, Inc.
   
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
#define YYBISON_VERSION "2.5"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1

/* Using locations.  */
#define YYLSP_NEEDED 1



/* Copy the first part of user declarations.  */

/* Line 268 of yacc.c  */
#line 2 "xi-grammar.y"

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
bool firstRdma = true;

bool enable_warnings = true;

extern int macroDefined(const char *str, int istrue);
extern const char *python_doc;
extern char *fname;
void splitScopedName(const char* name, const char** scope, const char** basename);
void ReservedWord(int token, int fCol, int lCol);
}


/* Line 268 of yacc.c  */
#line 120 "y.tab.c"

/* Enabling traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

/* Enabling the token table.  */
#ifndef YYTOKEN_TABLE
# define YYTOKEN_TABLE 0
#endif


/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum yytokentype {
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
     NOCOPY = 294,
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
#define NOCOPY 294
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




#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE
{

/* Line 293 of yacc.c  */
#line 52 "xi-grammar.y"

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



/* Line 293 of yacc.c  */
#line 350 "y.tab.c"
} YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
#endif

#if ! defined YYLTYPE && ! defined YYLTYPE_IS_DECLARED
typedef struct YYLTYPE
{
  int first_line;
  int first_column;
  int last_line;
  int last_column;
} YYLTYPE;
# define yyltype YYLTYPE /* obsolescent; will be withdrawn */
# define YYLTYPE_IS_DECLARED 1
# define YYLTYPE_IS_TRIVIAL 1
#endif


/* Copy the second part of user declarations.  */


/* Line 343 of yacc.c  */
#line 375 "y.tab.c"

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
#elif (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
typedef signed char yytype_int8;
#else
typedef short int yytype_int8;
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
# elif ! defined YYSIZE_T && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
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
#   define YY_(msgid) dgettext ("bison-runtime", msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(msgid) msgid
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(e) ((void) (e))
#else
# define YYUSE(e) /* empty */
#endif

/* Identity function, used to suppress warnings about constant conditions.  */
#ifndef lint
# define YYID(n) (n)
#else
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static int
YYID (int yyi)
#else
static int
YYID (yyi)
    int yyi;
#endif
{
  return yyi;
}
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
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's `empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (YYID (0))
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
#   if ! defined malloc && ! defined EXIT_SUCCESS && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
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
# define YYSTACK_RELOCATE(Stack_alloc, Stack)				\
    do									\
      {									\
	YYSIZE_T yynewbytes;						\
	YYCOPY (&yyptr->Stack_alloc, Stack, yysize);			\
	Stack = &yyptr->Stack_alloc;					\
	yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
	yyptr += yynewbytes / sizeof (*yyptr);				\
      }									\
    while (YYID (0))

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from FROM to TO.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(To, From, Count) \
      __builtin_memcpy (To, From, (Count) * sizeof (*(From)))
#  else
#   define YYCOPY(To, From, Count)		\
      do					\
	{					\
	  YYSIZE_T yyi;				\
	  for (yyi = 0; yyi < (Count); yyi++)	\
	    (To)[yyi] = (From)[yyi];		\
	}					\
      while (YYID (0))
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  56
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   1517

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  91
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  117
/* YYNRULES -- Number of rules.  */
#define YYNRULES  371
/* YYNRULES -- Number of states.  */
#define YYNSTATES  726

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   329

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
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
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     5,     6,     9,    10,    12,    14,    17,
      18,    20,    22,    24,    26,    28,    30,    32,    34,    36,
      38,    40,    42,    44,    46,    48,    50,    52,    54,    56,
      58,    60,    62,    64,    66,    68,    70,    72,    74,    76,
      78,    80,    82,    84,    86,    88,    90,    92,    94,    96,
      98,   100,   102,   104,   106,   108,   110,   112,   114,   116,
     118,   123,   127,   131,   133,   138,   139,   142,   146,   149,
     152,   155,   163,   169,   175,   178,   181,   184,   187,   190,
     193,   196,   199,   201,   203,   205,   207,   209,   211,   213,
     215,   219,   220,   222,   223,   227,   229,   231,   233,   235,
     238,   241,   245,   249,   252,   255,   258,   260,   262,   265,
     267,   270,   273,   275,   277,   280,   283,   286,   295,   297,
     299,   301,   303,   306,   309,   311,   313,   315,   318,   321,
     324,   326,   329,   331,   333,   337,   338,   341,   346,   353,
     354,   356,   357,   361,   363,   367,   369,   371,   372,   376,
     378,   382,   383,   385,   387,   388,   392,   394,   398,   400,
     402,   403,   405,   406,   409,   415,   417,   420,   424,   430,
     437,   438,   441,   443,   447,   453,   459,   465,   471,   476,
     480,   487,   494,   500,   506,   512,   518,   524,   529,   537,
     538,   541,   542,   545,   548,   551,   555,   558,   562,   564,
     568,   573,   576,   579,   582,   585,   588,   590,   595,   596,
     599,   601,   603,   605,   607,   610,   613,   616,   620,   627,
     637,   641,   648,   652,   659,   669,   679,   681,   685,   687,
     689,   691,   694,   697,   699,   701,   703,   705,   707,   709,
     711,   713,   715,   717,   719,   721,   729,   735,   749,   755,
     758,   760,   761,   765,   767,   769,   773,   775,   777,   779,
     781,   783,   785,   787,   789,   791,   793,   795,   797,   799,
     802,   804,   806,   808,   810,   812,   814,   816,   817,   819,
     823,   824,   826,   832,   838,   844,   849,   853,   855,   857,
     859,   863,   868,   872,   877,   879,   881,   883,   885,   890,
     894,   899,   904,   909,   913,   921,   927,   934,   936,   940,
     942,   946,   950,   953,   957,   960,   961,   965,   967,   969,
     974,   976,   979,   981,   984,   986,   989,   991,   993,   994,
     999,  1003,  1009,  1016,  1021,  1026,  1038,  1048,  1061,  1076,
    1083,  1092,  1098,  1106,  1111,  1118,  1123,  1125,  1130,  1142,
    1152,  1165,  1180,  1187,  1196,  1202,  1210,  1215,  1217,  1218,
    1221,  1226,  1228,  1230,  1232,  1235,  1241,  1243,  1247,  1249,
    1251,  1254
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
      92,     0,    -1,    93,    -1,    -1,    99,    93,    -1,    -1,
       5,    -1,    75,    -1,    95,    75,    -1,    -1,    95,    -1,
      55,    -1,     3,    -1,     4,    -1,     5,    -1,     7,    -1,
       8,    -1,     9,    -1,    11,    -1,    12,    -1,    13,    -1,
      14,    -1,    15,    -1,    19,    -1,    20,    -1,    21,    -1,
      22,    -1,    23,    -1,    24,    -1,    25,    -1,    26,    -1,
      27,    -1,    39,    -1,    28,    -1,    29,    -1,    30,    -1,
      32,    -1,    33,    -1,    34,    -1,    35,    -1,    36,    -1,
      40,    -1,    41,    -1,    42,    -1,    43,    -1,    44,    -1,
      45,    -1,    46,    -1,    47,    -1,    48,    -1,    49,    -1,
      50,    -1,    52,    -1,    54,    -1,    68,    -1,    71,    -1,
      72,    -1,    73,    -1,    74,    -1,    55,    -1,    98,    76,
      76,    55,    -1,     3,    97,   100,    -1,     4,    97,   100,
      -1,    95,    -1,    77,   101,    78,    96,    -1,    -1,   103,
     101,    -1,    54,    53,    98,    -1,    54,    98,    -1,    94,
     160,    -1,    94,   139,    -1,     5,    42,   170,   110,    97,
     107,   187,    -1,    94,    77,   101,    78,    96,    -1,    53,
      97,    77,   101,    78,    -1,   102,    95,    -1,   102,   167,
      -1,    94,    99,    -1,    94,   142,    -1,    94,   143,    -1,
      94,   144,    -1,    94,   146,    -1,    94,   157,    -1,   206,
      -1,   207,    -1,   169,    -1,     1,    -1,   118,    -1,    56,
      -1,    57,    -1,   104,    -1,   104,    79,   105,    -1,    -1,
     105,    -1,    -1,    80,   106,    81,    -1,    61,    -1,    62,
      -1,    63,    -1,    64,    -1,    67,    61,    -1,    67,    62,
      -1,    67,    62,    61,    -1,    67,    62,    62,    -1,    67,
      63,    -1,    67,    64,    -1,    62,    62,    -1,    65,    -1,
      66,    -1,    62,    66,    -1,    37,    -1,    97,   107,    -1,
      98,   107,    -1,   108,    -1,   110,    -1,   111,    82,    -1,
     112,    82,    -1,   113,    82,    -1,   115,    83,    82,    97,
      84,    83,   185,    84,    -1,   111,    -1,   112,    -1,   113,
      -1,   114,    -1,    38,   115,    -1,   115,    38,    -1,   111,
      -1,   112,    -1,   113,    -1,    38,   116,    -1,   116,    38,
      -1,   116,    85,    -1,   116,    -1,   115,    85,    -1,   115,
      -1,   176,    -1,   204,   119,   205,    -1,    -1,   120,   121,
      -1,     6,   118,    98,   121,    -1,     6,    16,   111,    82,
      98,   121,    -1,    -1,    37,    -1,    -1,    86,   126,    87,
      -1,   127,    -1,   127,    79,   126,    -1,    40,    -1,    41,
      -1,    -1,    86,   129,    87,    -1,   134,    -1,   134,    79,
     129,    -1,    -1,    57,    -1,    51,    -1,    -1,    86,   133,
      87,    -1,   131,    -1,   131,    79,   133,    -1,    30,    -1,
      51,    -1,    -1,    17,    -1,    -1,    86,    87,    -1,   135,
     118,    97,   136,    95,    -1,   137,    -1,   137,   138,    -1,
      16,   125,   109,    -1,    16,   125,   109,    77,    78,    -1,
      16,   125,   109,    77,   138,    78,    -1,    -1,    76,   141,
      -1,   110,    -1,   110,    79,   141,    -1,    11,   128,   109,
     140,   158,    -1,    12,   128,   109,   140,   158,    -1,    13,
     128,   109,   140,   158,    -1,    14,   128,   109,   140,   158,
      -1,    86,    56,    97,    87,    -1,    86,    97,    87,    -1,
      15,   132,   145,   109,   140,   158,    -1,    15,   145,   132,
     109,   140,   158,    -1,    11,   128,    97,   140,   158,    -1,
      12,   128,    97,   140,   158,    -1,    13,   128,    97,   140,
     158,    -1,    14,   128,    97,   140,   158,    -1,    15,   145,
      97,   140,   158,    -1,    16,   125,    97,    95,    -1,    16,
     125,    97,    77,   138,    78,    95,    -1,    -1,    88,   118,
      -1,    -1,    88,    56,    -1,    88,    57,    -1,    88,   110,
      -1,    18,    97,   152,    -1,   114,   153,    -1,   118,    97,
     153,    -1,   154,    -1,   154,    79,   155,    -1,    22,    80,
     155,    81,    -1,   156,   147,    -1,   156,   148,    -1,   156,
     149,    -1,   156,   150,    -1,   156,   151,    -1,    95,    -1,
      77,   159,    78,    96,    -1,    -1,   165,   159,    -1,   122,
      -1,   123,    -1,   162,    -1,   161,    -1,    10,   163,    -1,
      19,   164,    -1,    18,    97,    -1,     8,   124,    98,    -1,
       8,   124,    98,    83,   124,    84,    -1,     8,   124,    98,
      80,   105,    81,    83,   124,    84,    -1,     7,   124,    98,
      -1,     7,   124,    98,    83,   124,    84,    -1,     9,   124,
      98,    -1,     9,   124,    98,    83,   124,    84,    -1,     9,
     124,    98,    80,   105,    81,    83,   124,    84,    -1,     9,
      86,    68,    87,   124,    98,    83,   124,    84,    -1,   110,
      -1,   110,    79,   163,    -1,    57,    -1,   166,    -1,   168,
      -1,   156,   168,    -1,   160,    95,    -1,     1,    -1,    42,
      -1,    78,    -1,     7,    -1,     8,    -1,     9,    -1,    11,
      -1,    12,    -1,    15,    -1,    13,    -1,    14,    -1,     6,
      -1,    42,   171,   170,    97,   187,   189,   190,    -1,    42,
     171,    97,   187,   190,    -1,    42,    86,    68,    87,    37,
      97,   187,   188,   178,   176,   179,    97,    95,    -1,    71,
     178,   176,   179,    95,    -1,    71,    95,    -1,   117,    -1,
      -1,    86,   172,    87,    -1,     1,    -1,   173,    -1,   173,
      79,   172,    -1,    21,    -1,    23,    -1,    24,    -1,    25,
      -1,    32,    -1,    33,    -1,    34,    -1,    35,    -1,    36,
      -1,    26,    -1,    27,    -1,    28,    -1,    52,    -1,    51,
     130,    -1,    72,    -1,    73,    -1,    31,    -1,     1,    -1,
      57,    -1,    56,    -1,    98,    -1,    -1,    58,    -1,    58,
      79,   175,    -1,    -1,    58,    -1,    58,    86,   176,    87,
     176,    -1,    58,    77,   176,    78,   176,    -1,    58,    83,
     175,    84,   176,    -1,    83,   176,    84,   176,    -1,   118,
      97,    86,    -1,    77,    -1,    78,    -1,   118,    -1,   118,
      97,   135,    -1,   118,    97,    88,   174,    -1,   177,   176,
      87,    -1,    39,   177,   176,    87,    -1,     6,    -1,    69,
      -1,    70,    -1,    97,    -1,   182,    89,    81,    97,    -1,
     182,    90,    97,    -1,   182,    86,   182,    87,    -1,   182,
      86,    56,    87,    -1,   182,    83,   182,    84,    -1,   177,
     176,    87,    -1,   181,    76,   118,    97,    80,   182,    81,
      -1,   118,    97,    80,   182,    81,    -1,   181,    76,   183,
      80,   182,    81,    -1,   180,    -1,   180,    79,   185,    -1,
     184,    -1,   184,    79,   186,    -1,    83,   185,    84,    -1,
      83,    84,    -1,    86,   186,    87,    -1,    86,    87,    -1,
      -1,    20,    88,    56,    -1,    95,    -1,   197,    -1,    77,
     191,    78,    96,    -1,   197,    -1,   197,   191,    -1,   197,
      -1,   197,   191,    -1,   195,    -1,   195,   193,    -1,   196,
      -1,    57,    -1,    -1,    46,   203,    77,    78,    -1,    46,
     203,   197,    -1,    46,   203,    77,   191,    78,    -1,    48,
     194,   178,   176,   179,    96,    -1,    47,    77,   192,    78,
      -1,    74,    77,   193,    78,    -1,    43,   201,   176,    75,
     176,    75,   176,   200,    77,   191,    78,    -1,    43,   201,
     176,    75,   176,    75,   176,   200,   197,    -1,    44,    86,
      55,    87,   201,   176,    76,   176,    79,   176,   200,   197,
      -1,    44,    86,    55,    87,   201,   176,    76,   176,    79,
     176,   200,    77,   191,    78,    -1,    49,   201,   176,   200,
     197,   198,    -1,    49,   201,   176,   200,    77,   191,    78,
     198,    -1,    45,   201,   176,   200,   197,    -1,    45,   201,
     176,   200,    77,   191,    78,    -1,   178,   176,   179,    96,
      -1,    48,   194,   178,   176,   179,    96,    -1,    47,    77,
     192,    78,    -1,   195,    -1,    74,    77,   193,    78,    -1,
      43,   201,   199,    75,   199,    75,   199,   200,    77,   191,
      78,    -1,    43,   201,   199,    75,   199,    75,   199,   200,
     197,    -1,    44,    86,    55,    87,   201,   199,    76,   199,
      79,   199,   200,   197,    -1,    44,    86,    55,    87,   201,
     199,    76,   199,    79,   199,   200,    77,   191,    78,    -1,
      49,   201,   199,   200,   197,   198,    -1,    49,   201,   199,
     200,    77,   191,    78,   198,    -1,    45,   201,   199,   200,
     197,    -1,    45,   201,   199,   200,    77,   191,    78,    -1,
     178,   176,   179,    96,    -1,     1,    -1,    -1,    50,   197,
      -1,    50,    77,   191,    78,    -1,   176,    -1,    84,    -1,
      83,    -1,    55,   187,    -1,    55,   204,   176,   205,   187,
      -1,   202,    -1,   202,    79,   203,    -1,    86,    -1,    87,
      -1,    59,    97,    -1,    60,    97,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   194,   194,   199,   202,   207,   208,   212,   214,   219,
     220,   225,   227,   228,   229,   231,   232,   233,   235,   236,
     237,   238,   239,   243,   244,   245,   246,   247,   248,   249,
     250,   251,   252,   253,   254,   255,   256,   257,   258,   259,
     260,   263,   264,   265,   266,   267,   268,   269,   270,   271,
     272,   273,   275,   277,   278,   281,   282,   283,   284,   288,
     290,   297,   301,   308,   310,   315,   316,   320,   322,   324,
     326,   328,   341,   343,   345,   347,   353,   355,   357,   359,
     361,   363,   365,   367,   369,   371,   379,   381,   383,   387,
     389,   394,   395,   400,   401,   405,   407,   409,   411,   413,
     415,   417,   419,   421,   423,   425,   427,   429,   431,   433,
     437,   438,   445,   447,   451,   455,   457,   461,   465,   467,
     469,   471,   473,   475,   479,   481,   483,   485,   487,   491,
     493,   497,   499,   503,   507,   512,   513,   517,   521,   526,
     527,   532,   533,   543,   545,   549,   551,   556,   557,   561,
     563,   568,   569,   573,   578,   579,   583,   585,   589,   591,
     596,   597,   601,   602,   605,   609,   611,   615,   617,   619,
     624,   625,   629,   631,   635,   637,   641,   645,   649,   655,
     659,   661,   665,   667,   671,   675,   679,   683,   685,   690,
     691,   696,   697,   699,   701,   710,   712,   714,   718,   720,
     724,   728,   730,   732,   734,   736,   740,   742,   747,   754,
     758,   760,   762,   763,   765,   767,   769,   773,   775,   777,
     783,   789,   798,   800,   802,   808,   816,   818,   821,   825,
     829,   831,   836,   838,   846,   848,   850,   852,   854,   856,
     858,   860,   862,   864,   866,   869,   879,   896,   913,   915,
     919,   924,   925,   927,   934,   936,   940,   942,   944,   946,
     948,   950,   952,   954,   956,   958,   960,   962,   964,   966,
     968,   970,   972,   984,   993,   995,   997,  1002,  1003,  1005,
    1014,  1015,  1017,  1023,  1029,  1035,  1043,  1050,  1058,  1065,
    1067,  1069,  1071,  1076,  1088,  1089,  1090,  1093,  1094,  1095,
    1096,  1103,  1109,  1118,  1125,  1131,  1137,  1145,  1147,  1151,
    1153,  1157,  1159,  1163,  1165,  1170,  1171,  1175,  1177,  1179,
    1183,  1185,  1189,  1191,  1195,  1197,  1199,  1207,  1210,  1213,
    1215,  1217,  1221,  1223,  1225,  1227,  1229,  1231,  1233,  1235,
    1237,  1239,  1241,  1243,  1247,  1249,  1251,  1253,  1255,  1257,
    1259,  1262,  1265,  1267,  1269,  1271,  1273,  1275,  1286,  1287,
    1289,  1293,  1297,  1301,  1305,  1310,  1317,  1319,  1323,  1326,
    1330,  1334
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
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
  "VOID", "CONST", "NOCOPY", "PACKED", "VARSIZE", "ENTRY", "FOR", "FORALL",
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
  "HashIFDefComment", 0
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
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
     135,   135,   136,   136,   137,   138,   138,   139,   139,   139,
     140,   140,   141,   141,   142,   142,   143,   144,   145,   145,
     146,   146,   147,   147,   148,   149,   150,   151,   151,   152,
     152,   153,   153,   153,   153,   154,   154,   154,   155,   155,
     156,   157,   157,   157,   157,   157,   158,   158,   159,   159,
     160,   160,   160,   160,   160,   160,   160,   161,   161,   161,
     161,   161,   162,   162,   162,   162,   163,   163,   164,   165,
     166,   166,   166,   166,   167,   167,   167,   167,   167,   167,
     167,   167,   167,   167,   167,   168,   168,   168,   169,   169,
     170,   171,   171,   171,   172,   172,   173,   173,   173,   173,
     173,   173,   173,   173,   173,   173,   173,   173,   173,   173,
     173,   173,   173,   173,   174,   174,   174,   175,   175,   175,
     176,   176,   176,   176,   176,   176,   177,   178,   179,   180,
     180,   180,   180,   180,   181,   181,   181,   182,   182,   182,
     182,   182,   182,   183,   184,   184,   184,   185,   185,   186,
     186,   187,   187,   188,   188,   189,   189,   190,   190,   190,
     191,   191,   192,   192,   193,   193,   193,   194,   194,   195,
     195,   195,   196,   196,   196,   196,   196,   196,   196,   196,
     196,   196,   196,   196,   197,   197,   197,   197,   197,   197,
     197,   197,   197,   197,   197,   197,   197,   197,   198,   198,
     198,   199,   200,   201,   202,   202,   203,   203,   204,   205,
     206,   207
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
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
       0,     1,     0,     2,     5,     1,     2,     3,     5,     6,
       0,     2,     1,     3,     5,     5,     5,     5,     4,     3,
       6,     6,     5,     5,     5,     5,     5,     4,     7,     0,
       2,     0,     2,     2,     2,     3,     2,     3,     1,     3,
       4,     2,     2,     2,     2,     2,     1,     4,     0,     2,
       1,     1,     1,     1,     2,     2,     2,     3,     6,     9,
       3,     6,     3,     6,     9,     9,     1,     3,     1,     1,
       1,     2,     2,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     7,     5,    13,     5,     2,
       1,     0,     3,     1,     1,     3,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     2,
       1,     1,     1,     1,     1,     1,     1,     0,     1,     3,
       0,     1,     5,     5,     5,     4,     3,     1,     1,     1,
       3,     4,     3,     4,     1,     1,     1,     1,     4,     3,
       4,     4,     4,     3,     7,     5,     6,     1,     3,     1,
       3,     3,     2,     3,     2,     0,     3,     1,     1,     4,
       1,     2,     1,     2,     1,     2,     1,     1,     0,     4,
       3,     5,     6,     4,     4,    11,     9,    12,    14,     6,
       8,     5,     7,     4,     6,     4,     1,     4,    11,     9,
      12,    14,     6,     8,     5,     7,     4,     1,     0,     2,
       4,     1,     1,     1,     2,     5,     1,     3,     1,     1,
       2,     2
};

/* YYDEFACT[STATE-NAME] -- Default reduction number in state STATE-NUM.
   Performed when YYTABLE doesn't specify something else to do.  Zero
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
       0,    59,    68,   370,   371,   287,   249,   280,     0,   139,
     139,   139,     0,   147,   147,   147,   147,     0,   141,     0,
       0,     0,     0,    76,   210,   211,    70,    77,    78,    79,
      80,     0,    81,    69,   213,   212,     9,   244,   236,   237,
     238,   239,   240,   242,   243,   241,   234,   235,    74,    75,
      66,   109,     0,    95,    96,    97,    98,   106,   107,     0,
      93,   112,   113,   124,   125,   126,   130,   250,     0,     0,
      67,     0,   281,   280,     0,     0,     0,   118,   119,   120,
     121,   132,     0,   140,     0,     0,     0,     0,   226,   214,
       0,     0,     0,     0,     0,     0,     0,   154,     0,     0,
     216,   228,   215,     0,     0,   147,   147,   147,   147,     0,
     141,   201,   202,   203,   204,   205,    10,    64,   127,   105,
     108,    99,   100,   103,   104,    91,   111,   114,   115,   116,
     128,   129,     0,     0,     0,   280,   277,   280,     0,   288,
       0,     0,   122,   123,     0,   131,   135,   220,   217,     0,
     222,     0,   158,   159,     0,   149,    93,   170,   170,   170,
     170,   153,     0,     0,   156,     0,     0,     0,     0,     0,
     145,   146,     0,   143,   167,     0,   121,     0,   198,     0,
       9,     0,     0,     0,     0,     0,     0,   101,   102,    87,
      88,    89,    92,     0,    86,    93,    73,    60,     0,   278,
       0,     0,   280,   248,     0,     0,   368,   135,   137,   280,
     139,     0,   139,   139,     0,   139,   227,   148,     0,   110,
       0,     0,     0,     0,     0,     0,   179,     0,   155,   170,
     170,   142,     0,   160,   189,     0,   196,   191,     0,   200,
      72,   170,   170,   170,   170,   170,     0,     0,    94,     0,
     280,   277,   280,   280,   285,   135,     0,   136,     0,   133,
       0,     0,     0,     0,     0,     0,   150,   172,   171,     0,
     206,   174,   175,   176,   177,   178,   157,     0,     0,   144,
     161,   168,     0,   160,     0,     0,   195,   192,   193,   194,
     197,   199,     0,     0,     0,     0,     0,   160,   187,    90,
       0,    71,   283,   279,   284,   282,   138,     0,   369,   134,
     221,     0,   218,     0,     0,   223,     0,   233,     0,     0,
       0,     0,     0,   229,   230,   180,   181,     0,   166,   169,
     190,   182,   183,   184,   185,   186,     0,     0,   312,   289,
     280,   307,     0,     0,   139,   139,   139,   173,   253,     0,
       0,   231,     9,   232,   209,   162,     0,     0,   280,   160,
       0,     0,   311,     0,     0,     0,     0,   273,   256,   257,
     258,   259,   265,   266,   267,   272,   260,   261,   262,   263,
     264,   151,   268,     0,   270,   271,     0,   254,    59,     0,
       0,   207,     0,     0,   188,     0,     0,   286,     0,   290,
     292,   308,   117,   219,   225,   224,   152,   269,     0,   252,
       0,     0,     0,   163,   164,   293,   275,   274,   276,   291,
       0,   255,   357,     0,     0,     0,     0,     0,   328,     0,
       0,     0,   317,   280,   246,   346,   318,   315,     0,   363,
     280,     0,   280,     0,   366,     0,     0,   327,     0,   280,
       0,     0,     0,     0,     0,     0,     0,   361,     0,     0,
       0,   364,   280,     0,     0,   330,     0,     0,   280,     0,
       0,     0,     0,     0,   328,     0,     0,   280,     0,   324,
     326,     9,   321,     9,     0,   245,     0,     0,   280,     0,
     362,     0,     0,   367,   329,     0,   345,   323,     0,     0,
     280,     0,   280,     0,     0,   280,     0,     0,   347,   325,
     319,   356,   316,   294,   295,   296,   314,     0,     0,   309,
       0,   280,     0,   280,     0,   354,     0,   331,     9,     0,
     358,     0,     0,     0,     0,   280,     0,     0,     9,     0,
       0,     0,   313,     0,   280,     0,     0,   365,   344,     0,
       0,   352,   280,     0,     0,   333,     0,     0,   334,   343,
       0,     0,   280,     0,   310,     0,     0,   280,   355,   358,
       0,   359,     0,   280,     0,   341,     9,     0,   358,   297,
       0,     0,     0,     0,     0,     0,     0,   353,     0,   280,
       0,     0,   332,     0,   339,   305,     0,     0,     0,     0,
       0,   303,     0,   247,     0,   349,   280,   360,     0,   280,
     342,   358,     0,     0,     0,     0,   299,     0,   306,     0,
       0,     0,     0,   340,   302,   301,   300,   298,   304,   348,
       0,     0,   336,   280,     0,   350,     0,     0,     0,   335,
       0,   351,     0,   337,     0,   338
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     3,     4,    70,   350,   197,   236,   140,     5,    61,
      71,    72,    73,   271,   272,   273,   206,   141,   237,   142,
     157,   158,   159,   160,   161,   146,   147,   274,   338,   287,
     288,   104,   105,   164,   179,   252,   253,   171,   234,   487,
     244,   176,   245,   235,   362,   473,   363,   364,   106,   301,
     348,   107,   108,   109,   177,   110,   191,   192,   193,   194,
     195,   366,   316,   258,   259,   399,   112,   351,   400,   401,
     114,   115,   169,   182,   402,   403,   129,   404,    74,   148,
     430,   466,   467,   499,   280,   537,   420,   513,   220,   421,
     598,   660,   643,   599,   422,   600,   381,   567,   535,   514,
     531,   546,   558,   528,   515,   560,   532,   631,   538,   571,
     520,   524,   525,   289,   389,    75,    76
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -603
static const yytype_int16 yypact[] =
{
     249,  1344,  1344,    34,  -603,   249,  -603,  -603,  -603,  -603,
    -603,  -603,  -603,  -603,  -603,  -603,  -603,  -603,  -603,  -603,
    -603,  -603,  -603,  -603,  -603,  -603,  -603,  -603,  -603,  -603,
    -603,  -603,  -603,  -603,  -603,  -603,  -603,  -603,  -603,  -603,
    -603,  -603,  -603,  -603,  -603,  -603,  -603,  -603,  -603,  -603,
    -603,  -603,  -603,  -603,    42,    42,  -603,  -603,  -603,   768,
     -35,  -603,  -603,  -603,     7,  1344,   149,  1344,  1344,   142,
     932,   -23,   910,   768,  -603,  -603,  -603,  -603,  1431,   -17,
     101,  -603,   116,  -603,  -603,  -603,   -35,   -21,   604,   136,
     136,    -9,   101,   138,   138,   138,   138,   160,   169,  1344,
     171,   155,   768,  -603,  -603,  -603,  -603,  -603,  -603,  -603,
    -603,   257,  -603,  -603,  -603,  -603,   182,  -603,  -603,  -603,
    -603,  -603,  -603,  -603,  -603,  -603,  -603,  -603,   -35,  -603,
    -603,  -603,  1431,  -603,    97,  -603,  -603,  -603,  -603,   176,
      58,  -603,  -603,   184,   192,   196,    15,  -603,   101,   768,
     116,   207,    68,   -21,   208,   931,  1450,   184,   192,   196,
    -603,    20,   101,  -603,   101,   101,   229,   101,   221,  -603,
      21,  1344,  1344,  1344,  1344,  1128,   222,   223,   255,  1344,
    -603,  -603,  -603,  1364,   233,   138,   138,   138,   138,   222,
     169,  -603,  -603,  -603,  -603,  -603,   -35,  -603,   277,  -603,
    -603,  -603,   237,  -603,  -603,  1396,  -603,  -603,  -603,  -603,
    -603,  -603,  1344,   252,   278,   -21,   276,   -21,   256,  -603,
     182,   260,    25,  -603,   261,  -603,    46,    38,   125,   266,
     140,   101,  -603,  -603,   267,   268,   259,   272,   272,   272,
     272,  -603,  1344,   271,   281,   274,  1200,  1344,   311,  1344,
    -603,  -603,   283,   284,   287,  1344,    62,  1344,   286,   290,
     182,  1344,  1344,  1344,  1344,  1344,  1344,  -603,  -603,  -603,
    -603,   293,  -603,   292,  -603,   259,  -603,  -603,   301,   309,
     296,   302,   -21,   -35,   101,  1344,  -603,   305,  -603,   -21,
     136,  1396,   136,   136,  1396,   136,  -603,  -603,    21,  -603,
     101,   154,   154,   154,   154,   313,  -603,   311,  -603,   272,
     272,  -603,   255,     0,   304,   225,  -603,   314,  1364,  -603,
    -603,   272,   272,   272,   272,   272,   172,  1396,  -603,   318,
     -21,   276,   -21,   -21,  -603,    46,   319,  -603,   317,  -603,
     323,   328,   327,   101,   331,   329,  -603,   335,  -603,   368,
     -35,  -603,  -603,  -603,  -603,  -603,  -603,   154,   154,  -603,
    -603,  -603,  1450,     9,   337,  1450,  -603,  -603,  -603,  -603,
    -603,  -603,   154,   154,   154,   154,   154,   399,   -35,  -603,
    1383,  -603,  -603,  -603,  -603,  -603,  -603,   334,  -603,  -603,
    -603,   336,  -603,    96,   338,  -603,   101,  -603,   684,   381,
     347,   182,   368,  -603,  -603,  -603,  -603,  1344,  -603,  -603,
    -603,  -603,  -603,  -603,  -603,  -603,   348,  1450,  -603,  1344,
     -21,   353,   351,  1417,   136,   136,   136,  -603,  -603,   948,
    1056,  -603,   182,   -35,  -603,   355,   182,  1344,   -21,     2,
     352,  1417,  -603,   358,   359,   360,   361,  -603,  -603,  -603,
    -603,  -603,  -603,  -603,  -603,  -603,  -603,  -603,  -603,  -603,
    -603,   377,  -603,   362,  -603,  -603,   363,   369,   364,   318,
    1344,  -603,   370,   182,   -35,   365,   371,  -603,   236,  -603,
    -603,  -603,  -603,  -603,  -603,  -603,  -603,  -603,   422,  -603,
    1001,   535,   318,  -603,   -35,  -603,  -603,  -603,   116,  -603,
    1344,  -603,  -603,   380,   378,   380,   413,   393,   414,   380,
     395,   258,   -35,   -21,  -603,  -603,  -603,   453,   318,  -603,
     -21,   419,   -21,   107,   397,   545,   555,  -603,   400,   -21,
     275,   403,   481,   208,   390,   535,   406,  -603,   408,   415,
     411,  -603,   -21,   413,   350,  -603,   423,   496,   -21,   411,
     380,   420,   380,   428,   414,   380,   430,   -21,   431,   275,
    -603,   182,  -603,   182,   452,  -603,   424,   400,   -21,   380,
    -603,   607,   317,  -603,  -603,   435,  -603,  -603,   208,   752,
     -21,   459,   -21,   555,   400,   -21,   275,   208,  -603,  -603,
    -603,  -603,  -603,  -603,  -603,  -603,  -603,  1344,   439,   437,
     433,   -21,   442,   -21,   258,  -603,   318,  -603,   182,   258,
     468,   447,   444,   411,   454,   -21,   411,   460,   182,   457,
    1450,   787,  -603,   208,   -21,   471,   470,  -603,  -603,   473,
     759,  -603,   -21,   380,   766,  -603,   208,   815,  -603,  -603,
    1344,  1344,   -21,   469,  -603,  1344,   411,   -21,  -603,   468,
     258,  -603,   477,   -21,   258,  -603,   182,   258,   468,  -603,
      72,   -48,   466,  1344,   182,   822,   475,  -603,   483,   -21,
     486,   485,  -603,   487,  -603,  -603,  1344,  1272,   488,  1344,
    1344,  -603,    94,   -35,   258,  -603,   -21,  -603,   411,   -21,
    -603,   468,    81,   479,    99,  1344,  -603,   144,  -603,   490,
     411,   829,   493,  -603,  -603,  -603,  -603,  -603,  -603,  -603,
     836,   258,  -603,   -21,   258,  -603,   497,   411,   498,  -603,
     885,  -603,   258,  -603,   499,  -603
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -603,  -603,   559,  -603,   -51,  -251,    -1,   -58,   515,   531,
     -50,  -603,  -603,  -603,  -179,  -603,  -216,  -603,  -129,   -79,
     -71,   -64,   -62,  -171,   441,   463,  -603,   -86,  -603,  -603,
    -262,  -603,  -603,   -80,   416,   299,  -603,   102,   316,  -603,
    -603,   438,   310,  -603,   177,  -603,  -603,  -238,  -603,     4,
     227,  -603,  -603,  -603,   -66,  -603,  -603,  -603,  -603,  -603,
    -603,  -603,   307,  -603,   300,   551,  -603,    80,   224,   557,
    -603,  -603,   394,  -603,  -603,  -603,  -603,   231,  -603,   198,
    -603,   143,  -603,  -603,   303,   -82,  -402,   -63,  -492,  -603,
    -603,  -550,  -603,  -603,  -312,    14,  -438,  -603,  -603,   103,
    -508,    53,  -529,    83,  -484,  -603,  -443,  -602,  -487,  -522,
    -476,  -603,   100,   122,    74,  -603,  -603
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -323
static const yytype_int16 yytable[] =
{
      54,    55,   162,    60,    60,   154,    87,   143,    82,   320,
     165,   167,   256,   168,   144,   438,   145,   360,    86,   360,
     299,   128,   150,   130,   562,   337,   360,   579,   163,   522,
     589,   491,   680,   529,    56,   540,   575,   152,   477,   577,
      77,   563,   549,   238,   239,   240,   559,   667,   516,    78,
     254,   232,   184,   210,   517,   116,   674,   617,   223,   329,
     149,   143,   153,   223,    79,   196,    83,    84,   144,   212,
     145,   218,   233,   386,   580,   559,   582,   166,   361,   585,
     536,   602,   545,   547,   221,   541,   608,  -165,   477,   703,
     478,   634,   516,   603,   637,   618,   626,   257,   180,   213,
     211,   629,   559,   224,   226,   225,   227,   228,   224,   230,
     247,   443,   341,   682,   151,   344,   625,    58,   309,    59,
     310,   290,   151,   265,   665,   408,   692,   694,   605,   481,
     697,   645,   286,   278,   151,   281,   610,   646,   205,   416,
     547,  -191,   668,  -191,   656,   215,   671,   256,   379,   673,
     315,   216,   168,   675,   217,   676,    81,   653,   677,   199,
     666,   678,   679,   200,   676,   704,   701,   677,   627,   283,
     678,   679,   151,   163,   243,   698,   699,   676,   710,   425,
     677,   471,   676,   678,   679,   677,   706,   651,   678,   679,
     380,   655,   151,   286,   658,   720,   172,   173,   174,   700,
     334,   151,    80,   716,    81,   291,   718,   339,   292,   196,
     340,   275,   342,   343,   724,   345,   151,    58,   642,    85,
     294,   347,   685,   295,   170,   708,   335,   676,   181,    58,
     677,   349,   257,   678,   679,   183,   369,   201,   202,   203,
     204,   305,   302,   303,   304,   243,   175,    58,   382,   377,
     384,   385,     1,     2,   314,   178,   317,    58,   712,   502,
     321,   322,   323,   324,   325,   326,   207,   715,   185,   186,
     187,   188,   189,   190,   208,   378,   407,   723,   209,   410,
      81,   367,   368,   214,   336,   393,   219,   261,   262,   263,
     264,    81,   496,   497,   419,   250,   251,   229,   267,   268,
     231,   503,   504,   505,   506,   507,   508,   509,   246,   248,
     590,   260,   591,   357,   358,   210,  -287,   347,   550,   551,
     552,   506,   553,   554,   555,   372,   373,   374,   375,   376,
     276,   437,   510,   277,   279,    85,  -287,   419,   440,   205,
     282,  -287,   284,   285,   444,   445,   446,   298,   300,   556,
     433,   502,    85,   293,   297,   419,   476,   628,   306,   143,
     307,   308,   241,   312,   313,   318,   144,   639,   145,   397,
     311,   319,   327,   328,    88,    89,    90,    91,    92,   330,
     332,   196,   352,   353,   354,   474,    99,   100,   331,   333,
     101,   286,   365,   503,   504,   505,   506,   507,   508,   509,
     355,   380,   315,   387,   388,   672,   435,   390,  -287,   391,
     398,   392,   394,   395,   396,   409,   360,   423,   439,   424,
     498,   426,   494,   398,   510,   432,   436,    85,   574,   469,
     593,   533,   441,  -287,   486,   442,   475,   405,   406,   480,
     512,   472,   482,   483,   484,   485,  -208,   -11,   490,   488,
     489,   477,   411,   412,   413,   414,   415,   493,   495,   500,
     572,   131,   156,   519,   521,   548,   578,   557,   523,   492,
     526,   527,   530,   534,   539,   587,   543,    85,   564,    81,
     597,   561,   502,   568,   512,   133,   134,   135,   136,   137,
     138,   139,   566,   594,   595,   570,   557,   502,   611,   518,
     613,   576,   569,   616,   601,   583,   581,   586,   592,   588,
     196,   596,   196,   607,   612,   620,   621,   624,   630,   623,
     622,   615,   632,   557,   503,   504,   505,   506,   507,   508,
     509,   633,   635,   636,   641,   597,   502,   640,   638,   503,
     504,   505,   506,   507,   508,   509,   502,   647,   648,   663,
     652,   649,   669,   681,   686,   510,   502,   196,    85,  -320,
     662,   687,   689,   690,    57,   691,   705,   196,   709,   695,
     510,   670,   713,    85,  -322,   719,   721,   725,   503,   504,
     505,   506,   507,   508,   509,   103,    62,   688,   503,   504,
     505,   506,   507,   508,   509,   198,   619,   222,   503,   504,
     505,   506,   507,   508,   509,   196,   266,   702,   502,   510,
      58,   359,   511,   683,   346,   249,   479,   356,   371,   510,
     155,   111,   544,   427,   370,   296,   434,   113,   470,   510,
     431,   717,    85,   501,   383,   644,   614,   584,   565,   659,
     661,   131,   156,   573,   664,   542,   606,     0,     0,     0,
     503,   504,   505,   506,   507,   508,   509,     0,     0,    81,
       0,     0,   659,     0,     0,   133,   134,   135,   136,   137,
     138,   139,     0,     0,     0,   659,   659,     0,   696,   659,
       0,   510,     0,     0,   604,   428,     0,  -251,  -251,  -251,
       0,  -251,  -251,  -251,   707,  -251,  -251,  -251,  -251,  -251,
       0,     0,     0,  -251,  -251,  -251,  -251,  -251,  -251,  -251,
    -251,  -251,  -251,  -251,  -251,     0,  -251,  -251,  -251,  -251,
    -251,  -251,  -251,  -251,  -251,  -251,  -251,  -251,  -251,  -251,
    -251,  -251,  -251,  -251,  -251,     0,  -251,     0,  -251,  -251,
       0,     0,     0,     0,     0,  -251,  -251,  -251,  -251,  -251,
    -251,  -251,  -251,   502,     0,  -251,  -251,  -251,  -251,     0,
     502,     0,     0,     0,     0,     0,     0,   502,     0,    63,
     429,    -5,    -5,    64,    -5,    -5,    -5,    -5,    -5,    -5,
      -5,    -5,    -5,    -5,    -5,     0,    -5,    -5,     0,     0,
      -5,     0,     0,   593,     0,   503,   504,   505,   506,   507,
     508,   509,   503,   504,   505,   506,   507,   508,   509,   503,
     504,   505,   506,   507,   508,   509,   502,     0,     0,     0,
       0,    65,    66,   502,   131,   156,   510,    67,    68,   609,
     502,     0,     0,   510,     0,     0,   650,   502,     0,    69,
     510,     0,    81,   654,     0,    -5,   -65,     0,   133,   134,
     135,   136,   137,   138,   139,     0,   594,   595,   503,   504,
     505,   506,   507,   508,   509,   503,   504,   505,   506,   507,
     508,   509,   503,   504,   505,   506,   507,   508,   509,   503,
     504,   505,   506,   507,   508,   509,   502,     0,     0,   510,
       0,     0,   657,     0,     0,     0,   510,     0,     0,   684,
       0,     0,     0,   510,     0,     0,   711,     0,     0,     0,
     510,     0,     0,   714,     0,     0,   117,   118,   119,   120,
       0,   121,   122,   123,   124,   125,     0,     0,   503,   504,
     505,   506,   507,   508,   509,     1,     2,     0,    88,    89,
      90,    91,    92,    93,    94,    95,    96,    97,    98,   447,
      99,   100,   126,     0,   101,     0,     0,     0,     0,   510,
       0,     0,   722,     0,     0,     0,     0,     0,   131,   448,
       0,   449,   450,   451,   452,   453,   454,     0,     0,   455,
     456,   457,   458,   459,   460,    58,    81,     0,   127,     0,
       0,     0,   133,   134,   135,   136,   137,   138,   139,   461,
     462,     0,   447,     0,     0,     0,     0,     0,     0,   102,
       0,     0,     0,     0,     0,     0,   463,     0,     0,     0,
     464,   465,   448,     0,   449,   450,   451,   452,   453,   454,
       0,     0,   455,   456,   457,   458,   459,   460,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   461,   462,     0,     0,     0,     0,     0,     6,
       7,     8,     0,     9,    10,    11,     0,    12,    13,    14,
      15,    16,     0,   464,   465,    17,    18,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,     0,    29,    30,
      31,    32,    33,   131,   132,    34,    35,    36,    37,    38,
      39,    40,    41,    42,    43,    44,    45,     0,    46,     0,
      47,   468,     0,     0,     0,     0,     0,   133,   134,   135,
     136,   137,   138,   139,    49,     0,     0,    50,    51,    52,
      53,     6,     7,     8,     0,     9,    10,    11,     0,    12,
      13,    14,    15,    16,     0,     0,     0,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,     0,
      29,    30,    31,    32,    33,     0,     0,    34,    35,    36,
      37,    38,    39,    40,    41,    42,    43,    44,    45,   241,
      46,     0,    47,    48,   242,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    49,     0,     0,    50,
      51,    52,    53,     6,     7,     8,     0,     9,    10,    11,
       0,    12,    13,    14,    15,    16,     0,     0,     0,    17,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      28,     0,    29,    30,    31,    32,    33,     0,     0,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,     0,    46,     0,    47,    48,   242,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    49,     0,
       0,    50,    51,    52,    53,     6,     7,     8,     0,     9,
      10,    11,     0,    12,    13,    14,    15,    16,     0,     0,
       0,    17,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,     0,    29,    30,    31,    32,    33,     0,
       0,    34,    35,    36,    37,    38,    39,    40,    41,    42,
      43,    44,    45,     0,    46,     0,    47,    48,   693,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      49,     0,     0,    50,    51,    52,    53,     6,     7,     8,
       0,     9,    10,    11,     0,    12,    13,    14,    15,    16,
       0,     0,     0,    17,    18,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,     0,    29,    30,    31,    32,
      33,     0,   255,    34,    35,    36,    37,    38,    39,    40,
      41,    42,    43,    44,    45,     0,    46,     0,    47,    48,
       0,   131,   156,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    49,     0,     0,    50,    51,    52,    53,    81,
     131,   156,   417,     0,     0,   133,   134,   135,   136,   137,
     138,   139,     0,   131,   156,     0,     0,     0,    81,     0,
       0,     0,     0,     0,   133,   134,   135,   136,   137,   138,
     139,    81,   269,   270,   131,   156,   417,   133,   134,   135,
     136,   137,   138,   139,     0,     0,     0,   418,   131,   132,
       0,     0,    81,     0,     0,     0,     0,     0,   133,   134,
     135,   136,   137,   138,   139,     0,    81,   131,   156,     0,
       0,     0,   133,   134,   135,   136,   137,   138,   139,     0,
       0,     0,     0,     0,     0,    81,     0,     0,     0,     0,
       0,   133,   134,   135,   136,   137,   138,   139
};

#define yypact_value_is_default(yystate) \
  ((yystate) == (-603))

#define yytable_value_is_error(yytable_value) \
  YYID (0)

static const yytype_int16 yycheck[] =
{
       1,     2,    88,    54,    55,    87,    69,    78,    66,   260,
      90,    91,   183,    92,    78,   417,    78,    17,    69,    17,
     236,    72,    80,    73,   532,   287,    17,   549,    37,   505,
     559,   469,    80,   509,     0,   522,   544,    58,    86,   547,
      75,   533,   529,   172,   173,   174,   530,   649,   491,    42,
     179,    30,   102,    38,   492,    78,   658,   586,    38,   275,
      77,   132,    83,    38,    65,   116,    67,    68,   132,   148,
     132,   153,    51,   335,   550,   559,   552,    86,    78,   555,
     518,   568,   525,   526,   155,   523,   578,    78,    86,   691,
      88,   613,   535,   569,   616,   587,   604,   183,    99,   149,
      85,   609,   586,    83,   162,    85,   164,   165,    83,   167,
     176,   423,   291,   663,    76,   294,   603,    75,   247,    77,
     249,    83,    76,   189,   646,   363,   676,   677,   571,   441,
     680,   623,    86,   215,    76,   217,   579,   624,    80,   377,
     583,    79,   650,    81,   636,    77,   654,   318,   327,   657,
      88,    83,   231,    81,    86,    83,    55,   633,    86,    62,
     647,    89,    90,    66,    83,    84,   688,    86,   606,   220,
      89,    90,    76,    37,   175,    81,   684,    83,   700,    83,
      86,   432,    83,    89,    90,    86,    87,   630,    89,    90,
      83,   634,    76,    86,   637,   717,    94,    95,    96,   686,
     282,    76,    53,   711,    55,    80,   714,   289,    83,   260,
     290,   212,   292,   293,   722,   295,    76,    75,   620,    77,
      80,   300,   665,    83,    86,    81,   284,    83,    57,    75,
      86,    77,   318,    89,    90,    80,   315,    61,    62,    63,
      64,   242,   238,   239,   240,   246,    86,    75,   330,    77,
     332,   333,     3,     4,   255,    86,   257,    75,   701,     1,
     261,   262,   263,   264,   265,   266,    82,   710,    11,    12,
      13,    14,    15,    16,    82,   326,   362,   720,    82,   365,
      55,    56,    57,    76,   285,   343,    78,   185,   186,   187,
     188,    55,    56,    57,   380,    40,    41,    68,    61,    62,
      79,    43,    44,    45,    46,    47,    48,    49,    86,    86,
     561,    78,   563,   309,   310,    38,    58,   396,    43,    44,
      45,    46,    47,    48,    49,   321,   322,   323,   324,   325,
      78,   417,    74,    55,    58,    77,    78,   423,   420,    80,
      84,    83,    82,    82,   424,   425,   426,    79,    76,    74,
     401,     1,    77,    87,    87,   441,   438,   608,    87,   430,
      79,    87,    51,    79,    77,    79,   430,   618,   430,     1,
      87,    81,    79,    81,     6,     7,     8,     9,    10,    78,
      84,   432,   302,   303,   304,   436,    18,    19,    79,    87,
      22,    86,    88,    43,    44,    45,    46,    47,    48,    49,
      87,    83,    88,    84,    87,   656,   407,    84,    58,    81,
      42,    84,    81,    84,    79,    78,    17,    83,   419,    83,
     478,    83,   473,    42,    74,    78,    78,    77,    78,   430,
       6,   513,    79,    83,    57,    84,   437,   357,   358,    87,
     491,    86,    84,    84,    84,    84,    78,    83,    79,    87,
      87,    86,   372,   373,   374,   375,   376,    87,    87,    37,
     542,    37,    38,    83,    86,   528,   548,   530,    55,   470,
      77,    57,    77,    20,    55,   557,    79,    77,    88,    55,
     566,    78,     1,    75,   535,    61,    62,    63,    64,    65,
      66,    67,    86,    69,    70,    84,   559,     1,   580,   500,
     582,    78,    87,   585,   567,    77,    86,    77,    56,    78,
     561,    87,   563,    78,    55,    76,    79,    75,    50,   601,
      87,   584,    75,   586,    43,    44,    45,    46,    47,    48,
      49,    87,    78,   615,   620,   621,     1,    80,    78,    43,
      44,    45,    46,    47,    48,    49,     1,    76,    78,    80,
     632,    78,    75,    87,    79,    74,     1,   608,    77,    78,
     642,    78,    76,    78,     5,    78,    87,   618,    78,    81,
      74,   653,    79,    77,    78,    78,    78,    78,    43,    44,
      45,    46,    47,    48,    49,    70,    55,   669,    43,    44,
      45,    46,    47,    48,    49,   132,   597,   156,    43,    44,
      45,    46,    47,    48,    49,   656,   190,   689,     1,    74,
      75,   312,    77,   664,   298,   177,   439,   307,   318,    74,
      16,    70,    77,   396,   317,   231,   402,    70,   430,    74,
     399,   713,    77,   490,   331,   621,   583,   554,   535,   640,
     641,    37,    38,   543,   645,   523,   572,    -1,    -1,    -1,
      43,    44,    45,    46,    47,    48,    49,    -1,    -1,    55,
      -1,    -1,   663,    -1,    -1,    61,    62,    63,    64,    65,
      66,    67,    -1,    -1,    -1,   676,   677,    -1,   679,   680,
      -1,    74,    -1,    -1,    77,     1,    -1,     3,     4,     5,
      -1,     7,     8,     9,   695,    11,    12,    13,    14,    15,
      -1,    -1,    -1,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    -1,    32,    33,    34,    35,
      36,    37,    38,    39,    40,    41,    42,    43,    44,    45,
      46,    47,    48,    49,    50,    -1,    52,    -1,    54,    55,
      -1,    -1,    -1,    -1,    -1,    61,    62,    63,    64,    65,
      66,    67,    68,     1,    -1,    71,    72,    73,    74,    -1,
       1,    -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,     1,
      86,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    14,    15,    16,    -1,    18,    19,    -1,    -1,
      22,    -1,    -1,     6,    -1,    43,    44,    45,    46,    47,
      48,    49,    43,    44,    45,    46,    47,    48,    49,    43,
      44,    45,    46,    47,    48,    49,     1,    -1,    -1,    -1,
      -1,    53,    54,     1,    37,    38,    74,    59,    60,    77,
       1,    -1,    -1,    74,    -1,    -1,    77,     1,    -1,    71,
      74,    -1,    55,    77,    -1,    77,    78,    -1,    61,    62,
      63,    64,    65,    66,    67,    -1,    69,    70,    43,    44,
      45,    46,    47,    48,    49,    43,    44,    45,    46,    47,
      48,    49,    43,    44,    45,    46,    47,    48,    49,    43,
      44,    45,    46,    47,    48,    49,     1,    -1,    -1,    74,
      -1,    -1,    77,    -1,    -1,    -1,    74,    -1,    -1,    77,
      -1,    -1,    -1,    74,    -1,    -1,    77,    -1,    -1,    -1,
      74,    -1,    -1,    77,    -1,    -1,     6,     7,     8,     9,
      -1,    11,    12,    13,    14,    15,    -1,    -1,    43,    44,
      45,    46,    47,    48,    49,     3,     4,    -1,     6,     7,
       8,     9,    10,    11,    12,    13,    14,    15,    16,     1,
      18,    19,    42,    -1,    22,    -1,    -1,    -1,    -1,    74,
      -1,    -1,    77,    -1,    -1,    -1,    -1,    -1,    37,    21,
      -1,    23,    24,    25,    26,    27,    28,    -1,    -1,    31,
      32,    33,    34,    35,    36,    75,    55,    -1,    78,    -1,
      -1,    -1,    61,    62,    63,    64,    65,    66,    67,    51,
      52,    -1,     1,    -1,    -1,    -1,    -1,    -1,    -1,    77,
      -1,    -1,    -1,    -1,    -1,    -1,    68,    -1,    -1,    -1,
      72,    73,    21,    -1,    23,    24,    25,    26,    27,    28,
      -1,    -1,    31,    32,    33,    34,    35,    36,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    51,    52,    -1,    -1,    -1,    -1,    -1,     3,
       4,     5,    -1,     7,     8,     9,    -1,    11,    12,    13,
      14,    15,    -1,    72,    73,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,    29,    30,    -1,    32,    33,
      34,    35,    36,    37,    38,    39,    40,    41,    42,    43,
      44,    45,    46,    47,    48,    49,    50,    -1,    52,    -1,
      54,    55,    -1,    -1,    -1,    -1,    -1,    61,    62,    63,
      64,    65,    66,    67,    68,    -1,    -1,    71,    72,    73,
      74,     3,     4,     5,    -1,     7,     8,     9,    -1,    11,
      12,    13,    14,    15,    -1,    -1,    -1,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    -1,
      32,    33,    34,    35,    36,    -1,    -1,    39,    40,    41,
      42,    43,    44,    45,    46,    47,    48,    49,    50,    51,
      52,    -1,    54,    55,    56,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    68,    -1,    -1,    71,
      72,    73,    74,     3,     4,     5,    -1,     7,     8,     9,
      -1,    11,    12,    13,    14,    15,    -1,    -1,    -1,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    -1,    32,    33,    34,    35,    36,    -1,    -1,    39,
      40,    41,    42,    43,    44,    45,    46,    47,    48,    49,
      50,    -1,    52,    -1,    54,    55,    56,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    68,    -1,
      -1,    71,    72,    73,    74,     3,     4,     5,    -1,     7,
       8,     9,    -1,    11,    12,    13,    14,    15,    -1,    -1,
      -1,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      28,    29,    30,    -1,    32,    33,    34,    35,    36,    -1,
      -1,    39,    40,    41,    42,    43,    44,    45,    46,    47,
      48,    49,    50,    -1,    52,    -1,    54,    55,    56,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      68,    -1,    -1,    71,    72,    73,    74,     3,     4,     5,
      -1,     7,     8,     9,    -1,    11,    12,    13,    14,    15,
      -1,    -1,    -1,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    -1,    32,    33,    34,    35,
      36,    -1,    18,    39,    40,    41,    42,    43,    44,    45,
      46,    47,    48,    49,    50,    -1,    52,    -1,    54,    55,
      -1,    37,    38,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    68,    -1,    -1,    71,    72,    73,    74,    55,
      37,    38,    39,    -1,    -1,    61,    62,    63,    64,    65,
      66,    67,    -1,    37,    38,    -1,    -1,    -1,    55,    -1,
      -1,    -1,    -1,    -1,    61,    62,    63,    64,    65,    66,
      67,    55,    56,    57,    37,    38,    39,    61,    62,    63,
      64,    65,    66,    67,    -1,    -1,    -1,    84,    37,    38,
      -1,    -1,    55,    -1,    -1,    -1,    -1,    -1,    61,    62,
      63,    64,    65,    66,    67,    -1,    55,    37,    38,    -1,
      -1,    -1,    61,    62,    63,    64,    65,    66,    67,    -1,
      -1,    -1,    -1,    -1,    -1,    55,    -1,    -1,    -1,    -1,
      -1,    61,    62,    63,    64,    65,    66,    67
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
      17,    78,   135,   137,   138,    88,   152,    56,    57,   110,
     153,   155,   140,   140,   140,   140,   140,    77,    95,   105,
      83,   187,   176,   175,   176,   176,   121,    84,    87,   205,
      84,    81,    84,    98,    81,    84,    79,     1,    42,   156,
     159,   160,   165,   166,   168,   158,   158,   118,   138,    78,
     118,   158,   158,   158,   158,   158,   138,    39,    84,   118,
     177,   180,   185,    83,    83,    83,    83,   141,     1,    86,
     171,   168,    78,    95,   159,    97,    78,   118,   177,    97,
     176,    79,    84,   185,   124,   124,   124,     1,    21,    23,
      24,    25,    26,    27,    28,    31,    32,    33,    34,    35,
      36,    51,    52,    68,    72,    73,   172,   173,    55,    97,
     170,    96,    86,   136,    95,    97,   176,    86,    88,   135,
      87,   185,    84,    84,    84,    84,    57,   130,    87,    87,
      79,   187,    97,    87,    95,    87,    56,    57,    98,   174,
      37,   172,     1,    43,    44,    45,    46,    47,    48,    49,
      74,    77,    95,   178,   190,   195,   197,   187,    97,    83,
     201,    86,   201,    55,   202,   203,    77,    57,   194,   201,
      77,   191,   197,   176,    20,   189,   187,   176,   199,    55,
     199,   187,   204,    79,    77,   197,   192,   197,   178,   199,
      43,    44,    45,    47,    48,    49,    74,   178,   193,   195,
     196,    78,   191,   179,    88,   190,    86,   188,    75,    87,
      84,   200,   176,   203,    78,   191,    78,   191,   176,   200,
     201,    86,   201,    77,   194,   201,    77,   176,    78,   193,
      96,    96,    56,     6,    69,    70,    87,   118,   181,   184,
     186,   178,   199,   201,    77,   197,   205,    78,   179,    77,
     197,   176,    55,   176,   192,   178,   176,   193,   179,    97,
      76,    79,    87,   176,    75,   199,   191,   187,    96,   191,
      50,   198,    75,    87,   200,    78,   176,   200,    78,    96,
      80,   118,   177,   183,   186,   179,   199,    76,    78,    78,
      77,   197,   176,   201,    77,   197,   179,    77,   197,    97,
     182,    97,   176,    80,    97,   200,   199,   198,   191,    75,
     176,   191,    96,   191,   198,    81,    83,    86,    89,    90,
      80,    87,   182,    95,    77,   197,    79,    78,   176,    76,
      78,    78,   182,    56,   182,    81,    97,   182,    81,   191,
     199,   200,   176,   198,    84,    87,    87,    97,    81,    78,
     200,    77,   197,    79,    77,   197,   191,   176,   191,    78,
     200,    78,    77,   197,   191,    78
};

#define yyerrok		(yyerrstatus = 0)
#define yyclearin	(yychar = YYEMPTY)
#define YYEMPTY		(-2)
#define YYEOF		0

#define YYACCEPT	goto yyacceptlab
#define YYABORT		goto yyabortlab
#define YYERROR		goto yyerrorlab


/* Like YYERROR except do call yyerror.  This remains here temporarily
   to ease the transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  However,
   YYFAIL appears to be in use.  Nevertheless, it is formally deprecated
   in Bison 2.4.2's NEWS entry, where a plan to phase it out is
   discussed.  */

#define YYFAIL		goto yyerrlab
#if defined YYFAIL
  /* This is here to suppress warnings from the GCC cpp's
     -Wunused-macros.  Normally we don't worry about that warning, but
     some users do, and we want to make it easy for users to remove
     YYFAIL uses, which will produce warnings from Bison 2.5.  */
#endif

#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)					\
do								\
  if (yychar == YYEMPTY && yylen == 1)				\
    {								\
      yychar = (Token);						\
      yylval = (Value);						\
      YYPOPSTACK (1);						\
      goto yybackup;						\
    }								\
  else								\
    {								\
      yyerror (YY_("syntax error: cannot back up")); \
      YYERROR;							\
    }								\
while (YYID (0))


#define YYTERROR	1
#define YYERRCODE	256


/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#define YYRHSLOC(Rhs, K) ((Rhs)[K])
#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)				\
    do									\
      if (YYID (N))                                                    \
	{								\
	  (Current).first_line   = YYRHSLOC (Rhs, 1).first_line;	\
	  (Current).first_column = YYRHSLOC (Rhs, 1).first_column;	\
	  (Current).last_line    = YYRHSLOC (Rhs, N).last_line;		\
	  (Current).last_column  = YYRHSLOC (Rhs, N).last_column;	\
	}								\
      else								\
	{								\
	  (Current).first_line   = (Current).last_line   =		\
	    YYRHSLOC (Rhs, 0).last_line;				\
	  (Current).first_column = (Current).last_column =		\
	    YYRHSLOC (Rhs, 0).last_column;				\
	}								\
    while (YYID (0))
#endif


/* YY_LOCATION_PRINT -- Print the location on the stream.
   This macro was not mandated originally: define only if we know
   we won't break user code: when these are the locations we know.  */

#ifndef YY_LOCATION_PRINT
# if defined YYLTYPE_IS_TRIVIAL && YYLTYPE_IS_TRIVIAL
#  define YY_LOCATION_PRINT(File, Loc)			\
     fprintf (File, "%d.%d-%d.%d",			\
	      (Loc).first_line, (Loc).first_column,	\
	      (Loc).last_line,  (Loc).last_column)
# else
#  define YY_LOCATION_PRINT(File, Loc) ((void) 0)
# endif
#endif


/* YYLEX -- calling `yylex' with the right arguments.  */

#ifdef YYLEX_PARAM
# define YYLEX yylex (YYLEX_PARAM)
#else
# define YYLEX yylex ()
#endif

/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)			\
do {						\
  if (yydebug)					\
    YYFPRINTF Args;				\
} while (YYID (0))

# define YY_SYMBOL_PRINT(Title, Type, Value, Location)			  \
do {									  \
  if (yydebug)								  \
    {									  \
      YYFPRINTF (stderr, "%s ", Title);					  \
      yy_symbol_print (stderr,						  \
		  Type, Value, Location); \
      YYFPRINTF (stderr, "\n");						  \
    }									  \
} while (YYID (0))


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep, YYLTYPE const * const yylocationp)
#else
static void
yy_symbol_value_print (yyoutput, yytype, yyvaluep, yylocationp)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
    YYLTYPE const * const yylocationp;
#endif
{
  if (!yyvaluep)
    return;
  YYUSE (yylocationp);
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# else
  YYUSE (yyoutput);
# endif
  switch (yytype)
    {
      default:
	break;
    }
}


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep, YYLTYPE const * const yylocationp)
#else
static void
yy_symbol_print (yyoutput, yytype, yyvaluep, yylocationp)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
    YYLTYPE const * const yylocationp;
#endif
{
  if (yytype < YYNTOKENS)
    YYFPRINTF (yyoutput, "token %s (", yytname[yytype]);
  else
    YYFPRINTF (yyoutput, "nterm %s (", yytname[yytype]);

  YY_LOCATION_PRINT (yyoutput, *yylocationp);
  YYFPRINTF (yyoutput, ": ");
  yy_symbol_value_print (yyoutput, yytype, yyvaluep, yylocationp);
  YYFPRINTF (yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_stack_print (yytype_int16 *yybottom, yytype_int16 *yytop)
#else
static void
yy_stack_print (yybottom, yytop)
    yytype_int16 *yybottom;
    yytype_int16 *yytop;
#endif
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)				\
do {								\
  if (yydebug)							\
    yy_stack_print ((Bottom), (Top));				\
} while (YYID (0))


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_reduce_print (YYSTYPE *yyvsp, YYLTYPE *yylsp, int yyrule)
#else
static void
yy_reduce_print (yyvsp, yylsp, yyrule)
    YYSTYPE *yyvsp;
    YYLTYPE *yylsp;
    int yyrule;
#endif
{
  int yynrhs = yyr2[yyrule];
  int yyi;
  unsigned long int yylno = yyrline[yyrule];
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
	     yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr, yyrhs[yyprhs[yyrule] + yyi],
		       &(yyvsp[(yyi + 1) - (yynrhs)])
		       , &(yylsp[(yyi + 1) - (yynrhs)])		       );
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)		\
do {					\
  if (yydebug)				\
    yy_reduce_print (yyvsp, yylsp, Rule); \
} while (YYID (0))

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
#ifndef	YYINITDEPTH
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
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static YYSIZE_T
yystrlen (const char *yystr)
#else
static YYSIZE_T
yystrlen (yystr)
    const char *yystr;
#endif
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
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static char *
yystpcpy (char *yydest, const char *yysrc)
#else
static char *
yystpcpy (yydest, yysrc)
    char *yydest;
    const char *yysrc;
#endif
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
  YYSIZE_T yysize0 = yytnamerr (0, yytname[yytoken]);
  YYSIZE_T yysize = yysize0;
  YYSIZE_T yysize1;
  enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
  /* Internationalized format string. */
  const char *yyformat = 0;
  /* Arguments of yyformat. */
  char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
  /* Number of reported tokens (one for the "unexpected", one per
     "expected"). */
  int yycount = 0;

  /* There are many possibilities here to consider:
     - Assume YYFAIL is not used.  It's too flawed to consider.  See
       <http://lists.gnu.org/archive/html/bison-patches/2009-12/msg00024.html>
       for details.  YYERROR is fine as it does not invoke this
       function.
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
                yysize1 = yysize + yytnamerr (0, yytname[yyx]);
                if (! (yysize <= yysize1
                       && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
                  return 2;
                yysize = yysize1;
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

  yysize1 = yysize + yystrlen (yyformat);
  if (! (yysize <= yysize1 && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
    return 2;
  yysize = yysize1;

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

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep, YYLTYPE *yylocationp)
#else
static void
yydestruct (yymsg, yytype, yyvaluep, yylocationp)
    const char *yymsg;
    int yytype;
    YYSTYPE *yyvaluep;
    YYLTYPE *yylocationp;
#endif
{
  YYUSE (yyvaluep);
  YYUSE (yylocationp);

  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  switch (yytype)
    {

      default:
	break;
    }
}


/* Prevent warnings from -Wmissing-prototypes.  */
#ifdef YYPARSE_PARAM
#if defined __STDC__ || defined __cplusplus
int yyparse (void *YYPARSE_PARAM);
#else
int yyparse ();
#endif
#else /* ! YYPARSE_PARAM */
#if defined __STDC__ || defined __cplusplus
int yyparse (void);
#else
int yyparse ();
#endif
#endif /* ! YYPARSE_PARAM */


/* The lookahead symbol.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;

/* Location data for the lookahead symbol.  */
YYLTYPE yylloc;

/* Number of syntax errors so far.  */
int yynerrs;


/*----------.
| yyparse.  |
`----------*/

#ifdef YYPARSE_PARAM
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void *YYPARSE_PARAM)
#else
int
yyparse (YYPARSE_PARAM)
    void *YYPARSE_PARAM;
#endif
#else /* ! YYPARSE_PARAM */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void)
#else
int
yyparse ()

#endif
#endif
{
    int yystate;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus;

    /* The stacks and their tools:
       `yyss': related to states.
       `yyvs': related to semantic values.
       `yyls': related to locations.

       Refer to the stacks thru separate pointers, to allow yyoverflow
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
  int yytoken;
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

  yytoken = 0;
  yyss = yyssa;
  yyvs = yyvsa;
  yyls = yylsa;
  yystacksize = YYINITDEPTH;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY; /* Cause a token to be read.  */

  /* Initialize stack pointers.
     Waste one element of value and location stack
     so that they stay on the same level as the state stack.
     The wasted elements are never initialized.  */
  yyssp = yyss;
  yyvsp = yyvs;
  yylsp = yyls;

#if defined YYLTYPE_IS_TRIVIAL && YYLTYPE_IS_TRIVIAL
  /* Initialize the default location before parsing starts.  */
  yylloc.first_line   = yylloc.last_line   = 1;
  yylloc.first_column = yylloc.last_column = 1;
#endif

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
      yychar = YYLEX;
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
  *++yyvsp = yylval;
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
     `$$ = $1'.

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

/* Line 1806 of yacc.c  */
#line 195 "xi-grammar.y"
    { (yyval.modlist) = (yyvsp[(1) - (1)].modlist); modlist = (yyvsp[(1) - (1)].modlist); }
    break;

  case 3:

/* Line 1806 of yacc.c  */
#line 199 "xi-grammar.y"
    { 
		  (yyval.modlist) = 0; 
		}
    break;

  case 4:

/* Line 1806 of yacc.c  */
#line 203 "xi-grammar.y"
    { (yyval.modlist) = new AstChildren<Module>(lineno, (yyvsp[(1) - (2)].module), (yyvsp[(2) - (2)].modlist)); }
    break;

  case 5:

/* Line 1806 of yacc.c  */
#line 207 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 6:

/* Line 1806 of yacc.c  */
#line 209 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 7:

/* Line 1806 of yacc.c  */
#line 213 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 8:

/* Line 1806 of yacc.c  */
#line 215 "xi-grammar.y"
    { (yyval.intval) = 2; }
    break;

  case 9:

/* Line 1806 of yacc.c  */
#line 219 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 10:

/* Line 1806 of yacc.c  */
#line 221 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 11:

/* Line 1806 of yacc.c  */
#line 226 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 12:

/* Line 1806 of yacc.c  */
#line 227 "xi-grammar.y"
    { ReservedWord(MODULE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 13:

/* Line 1806 of yacc.c  */
#line 228 "xi-grammar.y"
    { ReservedWord(MAINMODULE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 14:

/* Line 1806 of yacc.c  */
#line 229 "xi-grammar.y"
    { ReservedWord(EXTERN, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 15:

/* Line 1806 of yacc.c  */
#line 231 "xi-grammar.y"
    { ReservedWord(INITCALL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 16:

/* Line 1806 of yacc.c  */
#line 232 "xi-grammar.y"
    { ReservedWord(INITNODE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 17:

/* Line 1806 of yacc.c  */
#line 233 "xi-grammar.y"
    { ReservedWord(INITPROC, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 18:

/* Line 1806 of yacc.c  */
#line 235 "xi-grammar.y"
    { ReservedWord(CHARE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 19:

/* Line 1806 of yacc.c  */
#line 236 "xi-grammar.y"
    { ReservedWord(MAINCHARE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 20:

/* Line 1806 of yacc.c  */
#line 237 "xi-grammar.y"
    { ReservedWord(GROUP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 21:

/* Line 1806 of yacc.c  */
#line 238 "xi-grammar.y"
    { ReservedWord(NODEGROUP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 22:

/* Line 1806 of yacc.c  */
#line 239 "xi-grammar.y"
    { ReservedWord(ARRAY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 23:

/* Line 1806 of yacc.c  */
#line 243 "xi-grammar.y"
    { ReservedWord(INCLUDE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 24:

/* Line 1806 of yacc.c  */
#line 244 "xi-grammar.y"
    { ReservedWord(STACKSIZE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 25:

/* Line 1806 of yacc.c  */
#line 245 "xi-grammar.y"
    { ReservedWord(THREADED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 26:

/* Line 1806 of yacc.c  */
#line 246 "xi-grammar.y"
    { ReservedWord(TEMPLATE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 27:

/* Line 1806 of yacc.c  */
#line 247 "xi-grammar.y"
    { ReservedWord(SYNC, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 28:

/* Line 1806 of yacc.c  */
#line 248 "xi-grammar.y"
    { ReservedWord(IGET, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 29:

/* Line 1806 of yacc.c  */
#line 249 "xi-grammar.y"
    { ReservedWord(EXCLUSIVE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 30:

/* Line 1806 of yacc.c  */
#line 250 "xi-grammar.y"
    { ReservedWord(IMMEDIATE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 31:

/* Line 1806 of yacc.c  */
#line 251 "xi-grammar.y"
    { ReservedWord(SKIPSCHED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 32:

/* Line 1806 of yacc.c  */
#line 252 "xi-grammar.y"
    { ReservedWord(NOCOPY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 33:

/* Line 1806 of yacc.c  */
#line 253 "xi-grammar.y"
    { ReservedWord(INLINE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 34:

/* Line 1806 of yacc.c  */
#line 254 "xi-grammar.y"
    { ReservedWord(VIRTUAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 35:

/* Line 1806 of yacc.c  */
#line 255 "xi-grammar.y"
    { ReservedWord(MIGRATABLE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 36:

/* Line 1806 of yacc.c  */
#line 256 "xi-grammar.y"
    { ReservedWord(CREATEHERE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 37:

/* Line 1806 of yacc.c  */
#line 257 "xi-grammar.y"
    { ReservedWord(CREATEHOME, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 38:

/* Line 1806 of yacc.c  */
#line 258 "xi-grammar.y"
    { ReservedWord(NOKEEP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 39:

/* Line 1806 of yacc.c  */
#line 259 "xi-grammar.y"
    { ReservedWord(NOTRACE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 40:

/* Line 1806 of yacc.c  */
#line 260 "xi-grammar.y"
    { ReservedWord(APPWORK, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 41:

/* Line 1806 of yacc.c  */
#line 263 "xi-grammar.y"
    { ReservedWord(PACKED, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 42:

/* Line 1806 of yacc.c  */
#line 264 "xi-grammar.y"
    { ReservedWord(VARSIZE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 43:

/* Line 1806 of yacc.c  */
#line 265 "xi-grammar.y"
    { ReservedWord(ENTRY, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 44:

/* Line 1806 of yacc.c  */
#line 266 "xi-grammar.y"
    { ReservedWord(FOR, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 45:

/* Line 1806 of yacc.c  */
#line 267 "xi-grammar.y"
    { ReservedWord(FORALL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 46:

/* Line 1806 of yacc.c  */
#line 268 "xi-grammar.y"
    { ReservedWord(WHILE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 47:

/* Line 1806 of yacc.c  */
#line 269 "xi-grammar.y"
    { ReservedWord(WHEN, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 48:

/* Line 1806 of yacc.c  */
#line 270 "xi-grammar.y"
    { ReservedWord(OVERLAP, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 49:

/* Line 1806 of yacc.c  */
#line 271 "xi-grammar.y"
    { ReservedWord(SERIAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 50:

/* Line 1806 of yacc.c  */
#line 272 "xi-grammar.y"
    { ReservedWord(IF, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 51:

/* Line 1806 of yacc.c  */
#line 273 "xi-grammar.y"
    { ReservedWord(ELSE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 52:

/* Line 1806 of yacc.c  */
#line 275 "xi-grammar.y"
    { ReservedWord(LOCAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 53:

/* Line 1806 of yacc.c  */
#line 277 "xi-grammar.y"
    { ReservedWord(USING, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 54:

/* Line 1806 of yacc.c  */
#line 278 "xi-grammar.y"
    { ReservedWord(ACCEL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 55:

/* Line 1806 of yacc.c  */
#line 281 "xi-grammar.y"
    { ReservedWord(ACCELBLOCK, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 56:

/* Line 1806 of yacc.c  */
#line 282 "xi-grammar.y"
    { ReservedWord(MEMCRITICAL, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 57:

/* Line 1806 of yacc.c  */
#line 283 "xi-grammar.y"
    { ReservedWord(REDUCTIONTARGET, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 58:

/* Line 1806 of yacc.c  */
#line 284 "xi-grammar.y"
    { ReservedWord(CASE, (yyloc).first_column, (yyloc).last_column); YYABORT; }
    break;

  case 59:

/* Line 1806 of yacc.c  */
#line 289 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 60:

/* Line 1806 of yacc.c  */
#line 291 "xi-grammar.y"
    {
		  char *tmp = new char[strlen((yyvsp[(1) - (4)].strval))+strlen((yyvsp[(4) - (4)].strval))+3];
		  sprintf(tmp,"%s::%s", (yyvsp[(1) - (4)].strval), (yyvsp[(4) - (4)].strval));
		  (yyval.strval) = tmp;
		}
    break;

  case 61:

/* Line 1806 of yacc.c  */
#line 298 "xi-grammar.y"
    { 
		    (yyval.module) = new Module(lineno, (yyvsp[(2) - (3)].strval), (yyvsp[(3) - (3)].conslist)); 
		}
    break;

  case 62:

/* Line 1806 of yacc.c  */
#line 302 "xi-grammar.y"
    {  
		    (yyval.module) = new Module(lineno, (yyvsp[(2) - (3)].strval), (yyvsp[(3) - (3)].conslist)); 
		    (yyval.module)->setMain();
		}
    break;

  case 63:

/* Line 1806 of yacc.c  */
#line 309 "xi-grammar.y"
    { (yyval.conslist) = 0; }
    break;

  case 64:

/* Line 1806 of yacc.c  */
#line 311 "xi-grammar.y"
    { (yyval.conslist) = (yyvsp[(2) - (4)].conslist); }
    break;

  case 65:

/* Line 1806 of yacc.c  */
#line 315 "xi-grammar.y"
    { (yyval.conslist) = 0; }
    break;

  case 66:

/* Line 1806 of yacc.c  */
#line 317 "xi-grammar.y"
    { (yyval.conslist) = new ConstructList(lineno, (yyvsp[(1) - (2)].construct), (yyvsp[(2) - (2)].conslist)); }
    break;

  case 67:

/* Line 1806 of yacc.c  */
#line 321 "xi-grammar.y"
    { (yyval.construct) = new UsingScope((yyvsp[(3) - (3)].strval), false); }
    break;

  case 68:

/* Line 1806 of yacc.c  */
#line 323 "xi-grammar.y"
    { (yyval.construct) = new UsingScope((yyvsp[(2) - (2)].strval), true); }
    break;

  case 69:

/* Line 1806 of yacc.c  */
#line 325 "xi-grammar.y"
    { (yyvsp[(2) - (2)].member)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].member); }
    break;

  case 70:

/* Line 1806 of yacc.c  */
#line 327 "xi-grammar.y"
    { (yyvsp[(2) - (2)].message)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].message); }
    break;

  case 71:

/* Line 1806 of yacc.c  */
#line 329 "xi-grammar.y"
    {
                  Entry *e = new Entry(lineno, 0, (yyvsp[(3) - (7)].type), (yyvsp[(5) - (7)].strval), (yyvsp[(7) - (7)].plist), 0, 0, 0, (yylsp[(1) - (7)]).first_line, (yyloc).last_line);
                  int isExtern = 1;
                  e->setExtern(isExtern);
                  e->targs = (yyvsp[(6) - (7)].tparlist);
                  e->label = new XStr;
                  (yyvsp[(4) - (7)].ntype)->print(*e->label);
                  (yyval.construct) = e;
                  firstRdma = true;
                }
    break;

  case 72:

/* Line 1806 of yacc.c  */
#line 342 "xi-grammar.y"
    { if((yyvsp[(3) - (5)].conslist)) (yyvsp[(3) - (5)].conslist)->recurse<int&>((yyvsp[(1) - (5)].intval), &Construct::setExtern); (yyval.construct) = (yyvsp[(3) - (5)].conslist); }
    break;

  case 73:

/* Line 1806 of yacc.c  */
#line 344 "xi-grammar.y"
    { (yyval.construct) = new Scope((yyvsp[(2) - (5)].strval), (yyvsp[(4) - (5)].conslist)); }
    break;

  case 74:

/* Line 1806 of yacc.c  */
#line 346 "xi-grammar.y"
    { (yyval.construct) = (yyvsp[(1) - (2)].construct); }
    break;

  case 75:

/* Line 1806 of yacc.c  */
#line 348 "xi-grammar.y"
    {
          ERROR("preceding construct must be semicolon terminated",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
    break;

  case 76:

/* Line 1806 of yacc.c  */
#line 354 "xi-grammar.y"
    { (yyvsp[(2) - (2)].module)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].module); }
    break;

  case 77:

/* Line 1806 of yacc.c  */
#line 356 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 78:

/* Line 1806 of yacc.c  */
#line 358 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 79:

/* Line 1806 of yacc.c  */
#line 360 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 80:

/* Line 1806 of yacc.c  */
#line 362 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 81:

/* Line 1806 of yacc.c  */
#line 364 "xi-grammar.y"
    { (yyvsp[(2) - (2)].templat)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].templat); }
    break;

  case 82:

/* Line 1806 of yacc.c  */
#line 366 "xi-grammar.y"
    { (yyval.construct) = NULL; }
    break;

  case 83:

/* Line 1806 of yacc.c  */
#line 368 "xi-grammar.y"
    { (yyval.construct) = NULL; }
    break;

  case 84:

/* Line 1806 of yacc.c  */
#line 370 "xi-grammar.y"
    { (yyval.construct) = (yyvsp[(1) - (1)].accelBlock); }
    break;

  case 85:

/* Line 1806 of yacc.c  */
#line 372 "xi-grammar.y"
    {
          ERROR("invalid construct",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
    break;

  case 86:

/* Line 1806 of yacc.c  */
#line 380 "xi-grammar.y"
    { (yyval.tparam) = new TParamType((yyvsp[(1) - (1)].type)); }
    break;

  case 87:

/* Line 1806 of yacc.c  */
#line 382 "xi-grammar.y"
    { (yyval.tparam) = new TParamVal((yyvsp[(1) - (1)].strval)); }
    break;

  case 88:

/* Line 1806 of yacc.c  */
#line 384 "xi-grammar.y"
    { (yyval.tparam) = new TParamVal((yyvsp[(1) - (1)].strval)); }
    break;

  case 89:

/* Line 1806 of yacc.c  */
#line 388 "xi-grammar.y"
    { (yyval.tparlist) = new TParamList((yyvsp[(1) - (1)].tparam)); }
    break;

  case 90:

/* Line 1806 of yacc.c  */
#line 390 "xi-grammar.y"
    { (yyval.tparlist) = new TParamList((yyvsp[(1) - (3)].tparam), (yyvsp[(3) - (3)].tparlist)); }
    break;

  case 91:

/* Line 1806 of yacc.c  */
#line 394 "xi-grammar.y"
    { (yyval.tparlist) = 0; }
    break;

  case 92:

/* Line 1806 of yacc.c  */
#line 396 "xi-grammar.y"
    { (yyval.tparlist) = (yyvsp[(1) - (1)].tparlist); }
    break;

  case 93:

/* Line 1806 of yacc.c  */
#line 400 "xi-grammar.y"
    { (yyval.tparlist) = 0; }
    break;

  case 94:

/* Line 1806 of yacc.c  */
#line 402 "xi-grammar.y"
    { (yyval.tparlist) = (yyvsp[(2) - (3)].tparlist); }
    break;

  case 95:

/* Line 1806 of yacc.c  */
#line 406 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("int"); }
    break;

  case 96:

/* Line 1806 of yacc.c  */
#line 408 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long"); }
    break;

  case 97:

/* Line 1806 of yacc.c  */
#line 410 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("short"); }
    break;

  case 98:

/* Line 1806 of yacc.c  */
#line 412 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("char"); }
    break;

  case 99:

/* Line 1806 of yacc.c  */
#line 414 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned int"); }
    break;

  case 100:

/* Line 1806 of yacc.c  */
#line 416 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long"); }
    break;

  case 101:

/* Line 1806 of yacc.c  */
#line 418 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long"); }
    break;

  case 102:

/* Line 1806 of yacc.c  */
#line 420 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long long"); }
    break;

  case 103:

/* Line 1806 of yacc.c  */
#line 422 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned short"); }
    break;

  case 104:

/* Line 1806 of yacc.c  */
#line 424 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned char"); }
    break;

  case 105:

/* Line 1806 of yacc.c  */
#line 426 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long long"); }
    break;

  case 106:

/* Line 1806 of yacc.c  */
#line 428 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("float"); }
    break;

  case 107:

/* Line 1806 of yacc.c  */
#line 430 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("double"); }
    break;

  case 108:

/* Line 1806 of yacc.c  */
#line 432 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long double"); }
    break;

  case 109:

/* Line 1806 of yacc.c  */
#line 434 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("void"); }
    break;

  case 110:

/* Line 1806 of yacc.c  */
#line 437 "xi-grammar.y"
    { (yyval.ntype) = new NamedType((yyvsp[(1) - (2)].strval),(yyvsp[(2) - (2)].tparlist)); }
    break;

  case 111:

/* Line 1806 of yacc.c  */
#line 438 "xi-grammar.y"
    { 
                    const char* basename, *scope;
                    splitScopedName((yyvsp[(1) - (2)].strval), &scope, &basename);
                    (yyval.ntype) = new NamedType(basename, (yyvsp[(2) - (2)].tparlist), scope);
                }
    break;

  case 112:

/* Line 1806 of yacc.c  */
#line 446 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 113:

/* Line 1806 of yacc.c  */
#line 448 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ntype); }
    break;

  case 114:

/* Line 1806 of yacc.c  */
#line 452 "xi-grammar.y"
    { (yyval.ptype) = new PtrType((yyvsp[(1) - (2)].type)); }
    break;

  case 115:

/* Line 1806 of yacc.c  */
#line 456 "xi-grammar.y"
    { (yyvsp[(1) - (2)].ptype)->indirect(); (yyval.ptype) = (yyvsp[(1) - (2)].ptype); }
    break;

  case 116:

/* Line 1806 of yacc.c  */
#line 458 "xi-grammar.y"
    { (yyvsp[(1) - (2)].ptype)->indirect(); (yyval.ptype) = (yyvsp[(1) - (2)].ptype); }
    break;

  case 117:

/* Line 1806 of yacc.c  */
#line 462 "xi-grammar.y"
    { (yyval.ftype) = new FuncType((yyvsp[(1) - (8)].type), (yyvsp[(4) - (8)].strval), (yyvsp[(7) - (8)].plist)); }
    break;

  case 118:

/* Line 1806 of yacc.c  */
#line 466 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 119:

/* Line 1806 of yacc.c  */
#line 468 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 120:

/* Line 1806 of yacc.c  */
#line 470 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 121:

/* Line 1806 of yacc.c  */
#line 472 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ftype); }
    break;

  case 122:

/* Line 1806 of yacc.c  */
#line 474 "xi-grammar.y"
    { (yyval.type) = new ConstType((yyvsp[(2) - (2)].type)); }
    break;

  case 123:

/* Line 1806 of yacc.c  */
#line 476 "xi-grammar.y"
    { (yyval.type) = new ConstType((yyvsp[(1) - (2)].type)); }
    break;

  case 124:

/* Line 1806 of yacc.c  */
#line 480 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 125:

/* Line 1806 of yacc.c  */
#line 482 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 126:

/* Line 1806 of yacc.c  */
#line 484 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 127:

/* Line 1806 of yacc.c  */
#line 486 "xi-grammar.y"
    { (yyval.type) = new ConstType((yyvsp[(2) - (2)].type)); }
    break;

  case 128:

/* Line 1806 of yacc.c  */
#line 488 "xi-grammar.y"
    { (yyval.type) = new ConstType((yyvsp[(1) - (2)].type)); }
    break;

  case 129:

/* Line 1806 of yacc.c  */
#line 492 "xi-grammar.y"
    { (yyval.type) = new ReferenceType((yyvsp[(1) - (2)].type)); }
    break;

  case 130:

/* Line 1806 of yacc.c  */
#line 494 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 131:

/* Line 1806 of yacc.c  */
#line 498 "xi-grammar.y"
    { (yyval.type) = new ReferenceType((yyvsp[(1) - (2)].type)); }
    break;

  case 132:

/* Line 1806 of yacc.c  */
#line 500 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 133:

/* Line 1806 of yacc.c  */
#line 504 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 134:

/* Line 1806 of yacc.c  */
#line 508 "xi-grammar.y"
    { (yyval.val) = (yyvsp[(2) - (3)].val); }
    break;

  case 135:

/* Line 1806 of yacc.c  */
#line 512 "xi-grammar.y"
    { (yyval.vallist) = 0; }
    break;

  case 136:

/* Line 1806 of yacc.c  */
#line 514 "xi-grammar.y"
    { (yyval.vallist) = new ValueList((yyvsp[(1) - (2)].val), (yyvsp[(2) - (2)].vallist)); }
    break;

  case 137:

/* Line 1806 of yacc.c  */
#line 518 "xi-grammar.y"
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[(2) - (4)].type), (yyvsp[(3) - (4)].strval), (yyvsp[(4) - (4)].vallist)); }
    break;

  case 138:

/* Line 1806 of yacc.c  */
#line 522 "xi-grammar.y"
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[(3) - (6)].type), (yyvsp[(5) - (6)].strval), (yyvsp[(6) - (6)].vallist), 1); }
    break;

  case 139:

/* Line 1806 of yacc.c  */
#line 526 "xi-grammar.y"
    { (yyval.intval) = 0;}
    break;

  case 140:

/* Line 1806 of yacc.c  */
#line 528 "xi-grammar.y"
    { (yyval.intval) = 0;}
    break;

  case 141:

/* Line 1806 of yacc.c  */
#line 532 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 142:

/* Line 1806 of yacc.c  */
#line 534 "xi-grammar.y"
    { 
		  /*
		  printf("Warning: Message attributes are being phased out.\n");
		  printf("Warning: Please remove them from interface files.\n");
		  */
		  (yyval.intval) = (yyvsp[(2) - (3)].intval); 
		}
    break;

  case 143:

/* Line 1806 of yacc.c  */
#line 544 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (1)].intval); }
    break;

  case 144:

/* Line 1806 of yacc.c  */
#line 546 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (3)].intval) | (yyvsp[(3) - (3)].intval); }
    break;

  case 145:

/* Line 1806 of yacc.c  */
#line 550 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 146:

/* Line 1806 of yacc.c  */
#line 552 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 147:

/* Line 1806 of yacc.c  */
#line 556 "xi-grammar.y"
    { (yyval.cattr) = 0; }
    break;

  case 148:

/* Line 1806 of yacc.c  */
#line 558 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(2) - (3)].cattr); }
    break;

  case 149:

/* Line 1806 of yacc.c  */
#line 562 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (1)].cattr); }
    break;

  case 150:

/* Line 1806 of yacc.c  */
#line 564 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (3)].cattr) | (yyvsp[(3) - (3)].cattr); }
    break;

  case 151:

/* Line 1806 of yacc.c  */
#line 568 "xi-grammar.y"
    { python_doc = NULL; (yyval.intval) = 0; }
    break;

  case 152:

/* Line 1806 of yacc.c  */
#line 570 "xi-grammar.y"
    { python_doc = (yyvsp[(1) - (1)].strval); (yyval.intval) = 0; }
    break;

  case 153:

/* Line 1806 of yacc.c  */
#line 574 "xi-grammar.y"
    { (yyval.cattr) = Chare::CPYTHON; }
    break;

  case 154:

/* Line 1806 of yacc.c  */
#line 578 "xi-grammar.y"
    { (yyval.cattr) = 0; }
    break;

  case 155:

/* Line 1806 of yacc.c  */
#line 580 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(2) - (3)].cattr); }
    break;

  case 156:

/* Line 1806 of yacc.c  */
#line 584 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (1)].cattr); }
    break;

  case 157:

/* Line 1806 of yacc.c  */
#line 586 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (3)].cattr) | (yyvsp[(3) - (3)].cattr); }
    break;

  case 158:

/* Line 1806 of yacc.c  */
#line 590 "xi-grammar.y"
    { (yyval.cattr) = Chare::CMIGRATABLE; }
    break;

  case 159:

/* Line 1806 of yacc.c  */
#line 592 "xi-grammar.y"
    { (yyval.cattr) = Chare::CPYTHON; }
    break;

  case 160:

/* Line 1806 of yacc.c  */
#line 596 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 161:

/* Line 1806 of yacc.c  */
#line 598 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 162:

/* Line 1806 of yacc.c  */
#line 601 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 163:

/* Line 1806 of yacc.c  */
#line 603 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 164:

/* Line 1806 of yacc.c  */
#line 606 "xi-grammar.y"
    { (yyval.mv) = new MsgVar((yyvsp[(2) - (5)].type), (yyvsp[(3) - (5)].strval), (yyvsp[(1) - (5)].intval), (yyvsp[(4) - (5)].intval)); }
    break;

  case 165:

/* Line 1806 of yacc.c  */
#line 610 "xi-grammar.y"
    { (yyval.mvlist) = new MsgVarList((yyvsp[(1) - (1)].mv)); }
    break;

  case 166:

/* Line 1806 of yacc.c  */
#line 612 "xi-grammar.y"
    { (yyval.mvlist) = new MsgVarList((yyvsp[(1) - (2)].mv), (yyvsp[(2) - (2)].mvlist)); }
    break;

  case 167:

/* Line 1806 of yacc.c  */
#line 616 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, (yyvsp[(3) - (3)].ntype)); }
    break;

  case 168:

/* Line 1806 of yacc.c  */
#line 618 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, (yyvsp[(3) - (5)].ntype)); }
    break;

  case 169:

/* Line 1806 of yacc.c  */
#line 620 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, (yyvsp[(3) - (6)].ntype), (yyvsp[(5) - (6)].mvlist)); }
    break;

  case 170:

/* Line 1806 of yacc.c  */
#line 624 "xi-grammar.y"
    { (yyval.typelist) = 0; }
    break;

  case 171:

/* Line 1806 of yacc.c  */
#line 626 "xi-grammar.y"
    { (yyval.typelist) = (yyvsp[(2) - (2)].typelist); }
    break;

  case 172:

/* Line 1806 of yacc.c  */
#line 630 "xi-grammar.y"
    { (yyval.typelist) = new TypeList((yyvsp[(1) - (1)].ntype)); }
    break;

  case 173:

/* Line 1806 of yacc.c  */
#line 632 "xi-grammar.y"
    { (yyval.typelist) = new TypeList((yyvsp[(1) - (3)].ntype), (yyvsp[(3) - (3)].typelist)); }
    break;

  case 174:

/* Line 1806 of yacc.c  */
#line 636 "xi-grammar.y"
    { (yyval.chare) = new Chare(lineno, (yyvsp[(2) - (5)].cattr)|Chare::CCHARE, (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 175:

/* Line 1806 of yacc.c  */
#line 638 "xi-grammar.y"
    { (yyval.chare) = new MainChare(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 176:

/* Line 1806 of yacc.c  */
#line 642 "xi-grammar.y"
    { (yyval.chare) = new Group(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 177:

/* Line 1806 of yacc.c  */
#line 646 "xi-grammar.y"
    { (yyval.chare) = new NodeGroup(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 178:

/* Line 1806 of yacc.c  */
#line 650 "xi-grammar.y"
    {/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",(yyvsp[(2) - (4)].strval));
			(yyval.ntype) = new NamedType(buf); 
		}
    break;

  case 179:

/* Line 1806 of yacc.c  */
#line 656 "xi-grammar.y"
    { (yyval.ntype) = new NamedType((yyvsp[(2) - (3)].strval)); }
    break;

  case 180:

/* Line 1806 of yacc.c  */
#line 660 "xi-grammar.y"
    {  (yyval.chare) = new Array(lineno, (yyvsp[(2) - (6)].cattr), (yyvsp[(3) - (6)].ntype), (yyvsp[(4) - (6)].ntype), (yyvsp[(5) - (6)].typelist), (yyvsp[(6) - (6)].mbrlist)); }
    break;

  case 181:

/* Line 1806 of yacc.c  */
#line 662 "xi-grammar.y"
    {  (yyval.chare) = new Array(lineno, (yyvsp[(3) - (6)].cattr), (yyvsp[(2) - (6)].ntype), (yyvsp[(4) - (6)].ntype), (yyvsp[(5) - (6)].typelist), (yyvsp[(6) - (6)].mbrlist)); }
    break;

  case 182:

/* Line 1806 of yacc.c  */
#line 666 "xi-grammar.y"
    { (yyval.chare) = new Chare(lineno, (yyvsp[(2) - (5)].cattr)|Chare::CCHARE, new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist));}
    break;

  case 183:

/* Line 1806 of yacc.c  */
#line 668 "xi-grammar.y"
    { (yyval.chare) = new MainChare(lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 184:

/* Line 1806 of yacc.c  */
#line 672 "xi-grammar.y"
    { (yyval.chare) = new Group(lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 185:

/* Line 1806 of yacc.c  */
#line 676 "xi-grammar.y"
    { (yyval.chare) = new NodeGroup( lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 186:

/* Line 1806 of yacc.c  */
#line 680 "xi-grammar.y"
    { (yyval.chare) = new Array( lineno, 0, (yyvsp[(2) - (5)].ntype), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 187:

/* Line 1806 of yacc.c  */
#line 684 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[(3) - (4)].strval))); }
    break;

  case 188:

/* Line 1806 of yacc.c  */
#line 686 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[(3) - (7)].strval)), (yyvsp[(5) - (7)].mvlist)); }
    break;

  case 189:

/* Line 1806 of yacc.c  */
#line 690 "xi-grammar.y"
    { (yyval.type) = 0; }
    break;

  case 190:

/* Line 1806 of yacc.c  */
#line 692 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(2) - (2)].type); }
    break;

  case 191:

/* Line 1806 of yacc.c  */
#line 696 "xi-grammar.y"
    { (yyval.strval) = 0; }
    break;

  case 192:

/* Line 1806 of yacc.c  */
#line 698 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(2) - (2)].strval); }
    break;

  case 193:

/* Line 1806 of yacc.c  */
#line 700 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(2) - (2)].strval); }
    break;

  case 194:

/* Line 1806 of yacc.c  */
#line 702 "xi-grammar.y"
    {
		  XStr typeStr;
		  (yyvsp[(2) - (2)].ntype)->print(typeStr);
		  char *tmp = strdup(typeStr.get_string());
		  (yyval.strval) = tmp;
		}
    break;

  case 195:

/* Line 1806 of yacc.c  */
#line 711 "xi-grammar.y"
    { (yyval.tvar) = new TType(new NamedType((yyvsp[(2) - (3)].strval)), (yyvsp[(3) - (3)].type)); }
    break;

  case 196:

/* Line 1806 of yacc.c  */
#line 713 "xi-grammar.y"
    { (yyval.tvar) = new TFunc((yyvsp[(1) - (2)].ftype), (yyvsp[(2) - (2)].strval)); }
    break;

  case 197:

/* Line 1806 of yacc.c  */
#line 715 "xi-grammar.y"
    { (yyval.tvar) = new TName((yyvsp[(1) - (3)].type), (yyvsp[(2) - (3)].strval), (yyvsp[(3) - (3)].strval)); }
    break;

  case 198:

/* Line 1806 of yacc.c  */
#line 719 "xi-grammar.y"
    { (yyval.tvarlist) = new TVarList((yyvsp[(1) - (1)].tvar)); }
    break;

  case 199:

/* Line 1806 of yacc.c  */
#line 721 "xi-grammar.y"
    { (yyval.tvarlist) = new TVarList((yyvsp[(1) - (3)].tvar), (yyvsp[(3) - (3)].tvarlist)); }
    break;

  case 200:

/* Line 1806 of yacc.c  */
#line 725 "xi-grammar.y"
    { (yyval.tvarlist) = (yyvsp[(3) - (4)].tvarlist); }
    break;

  case 201:

/* Line 1806 of yacc.c  */
#line 729 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 202:

/* Line 1806 of yacc.c  */
#line 731 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 203:

/* Line 1806 of yacc.c  */
#line 733 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 204:

/* Line 1806 of yacc.c  */
#line 735 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 205:

/* Line 1806 of yacc.c  */
#line 737 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].message)); (yyvsp[(2) - (2)].message)->setTemplate((yyval.templat)); }
    break;

  case 206:

/* Line 1806 of yacc.c  */
#line 741 "xi-grammar.y"
    { (yyval.mbrlist) = 0; }
    break;

  case 207:

/* Line 1806 of yacc.c  */
#line 743 "xi-grammar.y"
    { (yyval.mbrlist) = (yyvsp[(2) - (4)].mbrlist); }
    break;

  case 208:

/* Line 1806 of yacc.c  */
#line 747 "xi-grammar.y"
    { 
                  if (!connectEntries.empty()) {
                    (yyval.mbrlist) = new AstChildren<Member>(connectEntries);
		  } else {
		    (yyval.mbrlist) = 0; 
                  }
		}
    break;

  case 209:

/* Line 1806 of yacc.c  */
#line 755 "xi-grammar.y"
    { (yyval.mbrlist) = new AstChildren<Member>(-1, (yyvsp[(1) - (2)].member), (yyvsp[(2) - (2)].mbrlist)); }
    break;

  case 210:

/* Line 1806 of yacc.c  */
#line 759 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].readonly); }
    break;

  case 211:

/* Line 1806 of yacc.c  */
#line 761 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].readonly); }
    break;

  case 213:

/* Line 1806 of yacc.c  */
#line 764 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].member); }
    break;

  case 214:

/* Line 1806 of yacc.c  */
#line 766 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(2) - (2)].pupable); }
    break;

  case 215:

/* Line 1806 of yacc.c  */
#line 768 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(2) - (2)].includeFile); }
    break;

  case 216:

/* Line 1806 of yacc.c  */
#line 770 "xi-grammar.y"
    { (yyval.member) = new ClassDeclaration(lineno,(yyvsp[(2) - (2)].strval)); }
    break;

  case 217:

/* Line 1806 of yacc.c  */
#line 774 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 1); }
    break;

  case 218:

/* Line 1806 of yacc.c  */
#line 776 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 1); }
    break;

  case 219:

/* Line 1806 of yacc.c  */
#line 778 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[(3) - (9)].strval)) + '<' +
					    ((yyvsp[(5) - (9)].tparlist))->to_string() + '>').c_str()),
				    1);
		}
    break;

  case 220:

/* Line 1806 of yacc.c  */
#line 784 "xi-grammar.y"
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[(1) - (3)]).first_column, (yylsp[(1) - (3)]).last_column, (yylsp[(1) - (3)]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 1);
		}
    break;

  case 221:

/* Line 1806 of yacc.c  */
#line 790 "xi-grammar.y"
    {
		  WARNING("deprecated use of initcall. Use initnode or initproc instead",
		          (yylsp[(1) - (6)]).first_column, (yylsp[(1) - (6)]).last_column, (yylsp[(1) - (6)]).first_line);
		  (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 1);
		}
    break;

  case 222:

/* Line 1806 of yacc.c  */
#line 799 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 0); }
    break;

  case 223:

/* Line 1806 of yacc.c  */
#line 801 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 0); }
    break;

  case 224:

/* Line 1806 of yacc.c  */
#line 803 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[(3) - (9)].strval)) + '<' +
					    ((yyvsp[(5) - (9)].tparlist))->to_string() + '>').c_str()),
				    0);
		}
    break;

  case 225:

/* Line 1806 of yacc.c  */
#line 809 "xi-grammar.y"
    {
                  InitCall* rtn = new InitCall(lineno, (yyvsp[(6) - (9)].strval), 0);
                  rtn->setAccel();
                  (yyval.member) = rtn;
		}
    break;

  case 226:

/* Line 1806 of yacc.c  */
#line 817 "xi-grammar.y"
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[(1) - (1)].ntype),0); }
    break;

  case 227:

/* Line 1806 of yacc.c  */
#line 819 "xi-grammar.y"
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[(1) - (3)].ntype),(yyvsp[(3) - (3)].pupable)); }
    break;

  case 228:

/* Line 1806 of yacc.c  */
#line 822 "xi-grammar.y"
    { (yyval.includeFile) = new IncludeFile(lineno,(yyvsp[(1) - (1)].strval)); }
    break;

  case 229:

/* Line 1806 of yacc.c  */
#line 826 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].member); }
    break;

  case 230:

/* Line 1806 of yacc.c  */
#line 830 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].entry); }
    break;

  case 231:

/* Line 1806 of yacc.c  */
#line 832 "xi-grammar.y"
    {
                  (yyvsp[(2) - (2)].entry)->tspec = (yyvsp[(1) - (2)].tvarlist);
                  (yyval.member) = (yyvsp[(2) - (2)].entry);
                }
    break;

  case 232:

/* Line 1806 of yacc.c  */
#line 837 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (2)].member); }
    break;

  case 233:

/* Line 1806 of yacc.c  */
#line 839 "xi-grammar.y"
    {
          ERROR("invalid SDAG member",
                (yyloc).first_column, (yyloc).last_column);
          YYABORT;
        }
    break;

  case 234:

/* Line 1806 of yacc.c  */
#line 847 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 235:

/* Line 1806 of yacc.c  */
#line 849 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 236:

/* Line 1806 of yacc.c  */
#line 851 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 237:

/* Line 1806 of yacc.c  */
#line 853 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 238:

/* Line 1806 of yacc.c  */
#line 855 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 239:

/* Line 1806 of yacc.c  */
#line 857 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 240:

/* Line 1806 of yacc.c  */
#line 859 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 241:

/* Line 1806 of yacc.c  */
#line 861 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 242:

/* Line 1806 of yacc.c  */
#line 863 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 243:

/* Line 1806 of yacc.c  */
#line 865 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 244:

/* Line 1806 of yacc.c  */
#line 867 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 245:

/* Line 1806 of yacc.c  */
#line 870 "xi-grammar.y"
    { 
                  (yyval.entry) = new Entry(lineno, (yyvsp[(2) - (7)].intval), (yyvsp[(3) - (7)].type), (yyvsp[(4) - (7)].strval), (yyvsp[(5) - (7)].plist), (yyvsp[(6) - (7)].val), (yyvsp[(7) - (7)].sentry), (const char *) NULL, (yylsp[(1) - (7)]).first_line, (yyloc).last_line);
		  if ((yyvsp[(7) - (7)].sentry) != 0) { 
		    (yyvsp[(7) - (7)].sentry)->con1 = new SdagConstruct(SIDENT, (yyvsp[(4) - (7)].strval));
                    (yyvsp[(7) - (7)].sentry)->setEntry((yyval.entry));
                    (yyvsp[(7) - (7)].sentry)->param = new ParamList((yyvsp[(5) - (7)].plist));
                  }
                  firstRdma = true;
		}
    break;

  case 246:

/* Line 1806 of yacc.c  */
#line 880 "xi-grammar.y"
    { 
                  Entry *e = new Entry(lineno, (yyvsp[(2) - (5)].intval), 0, (yyvsp[(3) - (5)].strval), (yyvsp[(4) - (5)].plist),  0, (yyvsp[(5) - (5)].sentry), (const char *) NULL, (yylsp[(1) - (5)]).first_line, (yyloc).last_line);
                  if ((yyvsp[(5) - (5)].sentry) != 0) {
		    (yyvsp[(5) - (5)].sentry)->con1 = new SdagConstruct(SIDENT, (yyvsp[(3) - (5)].strval));
                    (yyvsp[(5) - (5)].sentry)->setEntry((yyval.entry));
                    (yyvsp[(5) - (5)].sentry)->param = new ParamList((yyvsp[(4) - (5)].plist));
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
    break;

  case 247:

/* Line 1806 of yacc.c  */
#line 897 "xi-grammar.y"
    {
                  int attribs = SACCEL;
                  const char* name = (yyvsp[(6) - (13)].strval);
                  ParamList* paramList = (yyvsp[(7) - (13)].plist);
                  ParamList* accelParamList = (yyvsp[(8) - (13)].plist);
		  XStr* codeBody = new XStr((yyvsp[(10) - (13)].strval));
                  const char* callbackName = (yyvsp[(12) - (13)].strval);

                  (yyval.entry) = new Entry(lineno, attribs, new BuiltinType("void"), name, paramList, 0, 0, 0 );
                  (yyval.entry)->setAccelParam(accelParamList);
                  (yyval.entry)->setAccelCodeBody(codeBody);
                  (yyval.entry)->setAccelCallbackName(new XStr(callbackName));
                  firstRdma = true;
                }
    break;

  case 248:

/* Line 1806 of yacc.c  */
#line 914 "xi-grammar.y"
    { (yyval.accelBlock) = new AccelBlock(lineno, new XStr((yyvsp[(3) - (5)].strval))); }
    break;

  case 249:

/* Line 1806 of yacc.c  */
#line 916 "xi-grammar.y"
    { (yyval.accelBlock) = new AccelBlock(lineno, NULL); }
    break;

  case 250:

/* Line 1806 of yacc.c  */
#line 920 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 251:

/* Line 1806 of yacc.c  */
#line 924 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 252:

/* Line 1806 of yacc.c  */
#line 926 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(2) - (3)].intval); }
    break;

  case 253:

/* Line 1806 of yacc.c  */
#line 928 "xi-grammar.y"
    { ERROR("invalid entry method attribute list",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
    break;

  case 254:

/* Line 1806 of yacc.c  */
#line 935 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (1)].intval); }
    break;

  case 255:

/* Line 1806 of yacc.c  */
#line 937 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (3)].intval) | (yyvsp[(3) - (3)].intval); }
    break;

  case 256:

/* Line 1806 of yacc.c  */
#line 941 "xi-grammar.y"
    { (yyval.intval) = STHREADED; }
    break;

  case 257:

/* Line 1806 of yacc.c  */
#line 943 "xi-grammar.y"
    { (yyval.intval) = SSYNC; }
    break;

  case 258:

/* Line 1806 of yacc.c  */
#line 945 "xi-grammar.y"
    { (yyval.intval) = SIGET; }
    break;

  case 259:

/* Line 1806 of yacc.c  */
#line 947 "xi-grammar.y"
    { (yyval.intval) = SLOCKED; }
    break;

  case 260:

/* Line 1806 of yacc.c  */
#line 949 "xi-grammar.y"
    { (yyval.intval) = SCREATEHERE; }
    break;

  case 261:

/* Line 1806 of yacc.c  */
#line 951 "xi-grammar.y"
    { (yyval.intval) = SCREATEHOME; }
    break;

  case 262:

/* Line 1806 of yacc.c  */
#line 953 "xi-grammar.y"
    { (yyval.intval) = SNOKEEP; }
    break;

  case 263:

/* Line 1806 of yacc.c  */
#line 955 "xi-grammar.y"
    { (yyval.intval) = SNOTRACE; }
    break;

  case 264:

/* Line 1806 of yacc.c  */
#line 957 "xi-grammar.y"
    { (yyval.intval) = SAPPWORK; }
    break;

  case 265:

/* Line 1806 of yacc.c  */
#line 959 "xi-grammar.y"
    { (yyval.intval) = SIMMEDIATE; }
    break;

  case 266:

/* Line 1806 of yacc.c  */
#line 961 "xi-grammar.y"
    { (yyval.intval) = SSKIPSCHED; }
    break;

  case 267:

/* Line 1806 of yacc.c  */
#line 963 "xi-grammar.y"
    { (yyval.intval) = SINLINE; }
    break;

  case 268:

/* Line 1806 of yacc.c  */
#line 965 "xi-grammar.y"
    { (yyval.intval) = SLOCAL; }
    break;

  case 269:

/* Line 1806 of yacc.c  */
#line 967 "xi-grammar.y"
    { (yyval.intval) = SPYTHON; }
    break;

  case 270:

/* Line 1806 of yacc.c  */
#line 969 "xi-grammar.y"
    { (yyval.intval) = SMEM; }
    break;

  case 271:

/* Line 1806 of yacc.c  */
#line 971 "xi-grammar.y"
    { (yyval.intval) = SREDUCE; }
    break;

  case 272:

/* Line 1806 of yacc.c  */
#line 973 "xi-grammar.y"
    {
#ifdef CMK_USING_XLC
        WARNING("a known bug in xl compilers (PMR 18366,122,000) currently breaks "
                "aggregate entry methods.\n"
                "Until a fix is released, this tag will be ignored on those compilers.",
                (yylsp[(1) - (1)]).first_column, (yylsp[(1) - (1)]).last_column, (yylsp[(1) - (1)]).first_line);
        (yyval.intval) = 0;
#else
        (yyval.intval) = SAGGREGATE;
#endif
    }
    break;

  case 273:

/* Line 1806 of yacc.c  */
#line 985 "xi-grammar.y"
    {
		  ERROR("invalid entry method attribute",
		        (yylsp[(1) - (1)]).first_column, (yylsp[(1) - (1)]).last_column);
		  yyclearin;
		  yyerrok;
		}
    break;

  case 274:

/* Line 1806 of yacc.c  */
#line 994 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 275:

/* Line 1806 of yacc.c  */
#line 996 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 276:

/* Line 1806 of yacc.c  */
#line 998 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 277:

/* Line 1806 of yacc.c  */
#line 1002 "xi-grammar.y"
    { (yyval.strval) = ""; }
    break;

  case 278:

/* Line 1806 of yacc.c  */
#line 1004 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 279:

/* Line 1806 of yacc.c  */
#line 1006 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[(1) - (3)].strval))+strlen((yyvsp[(3) - (3)].strval))+3];
			sprintf(tmp,"%s, %s", (yyvsp[(1) - (3)].strval), (yyvsp[(3) - (3)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 280:

/* Line 1806 of yacc.c  */
#line 1014 "xi-grammar.y"
    { (yyval.strval) = ""; }
    break;

  case 281:

/* Line 1806 of yacc.c  */
#line 1016 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 282:

/* Line 1806 of yacc.c  */
#line 1018 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s[%s]%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 283:

/* Line 1806 of yacc.c  */
#line 1024 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s{%s}%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 284:

/* Line 1806 of yacc.c  */
#line 1030 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s(%s)%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 285:

/* Line 1806 of yacc.c  */
#line 1036 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(2) - (4)].strval))+strlen((yyvsp[(4) - (4)].strval))+3];
			sprintf(tmp,"(%s)%s", (yyvsp[(2) - (4)].strval), (yyvsp[(4) - (4)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 286:

/* Line 1806 of yacc.c  */
#line 1044 "xi-grammar.y"
    {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			(yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].type),(yyvsp[(2) - (3)].strval));
		}
    break;

  case 287:

/* Line 1806 of yacc.c  */
#line 1051 "xi-grammar.y"
    { 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			(yyval.intval) = 0;
		}
    break;

  case 288:

/* Line 1806 of yacc.c  */
#line 1059 "xi-grammar.y"
    { 
			in_braces=0;
			(yyval.intval) = 0;
		}
    break;

  case 289:

/* Line 1806 of yacc.c  */
#line 1066 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (1)].type));}
    break;

  case 290:

/* Line 1806 of yacc.c  */
#line 1068 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].type),(yyvsp[(2) - (3)].strval)); (yyval.pname)->setConditional((yyvsp[(3) - (3)].intval)); }
    break;

  case 291:

/* Line 1806 of yacc.c  */
#line 1070 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (4)].type),(yyvsp[(2) - (4)].strval),0,(yyvsp[(4) - (4)].val));}
    break;

  case 292:

/* Line 1806 of yacc.c  */
#line 1072 "xi-grammar.y"
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].pname)->getType(), (yyvsp[(1) - (3)].pname)->getName() ,(yyvsp[(2) - (3)].strval));
		}
    break;

  case 293:

/* Line 1806 of yacc.c  */
#line 1077 "xi-grammar.y"
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[(2) - (4)].pname)->getType(), (yyvsp[(2) - (4)].pname)->getName() ,(yyvsp[(3) - (4)].strval));
			(yyval.pname)->setRdma(true);
			if(firstRdma) {
				(yyval.pname)->setFirstRdma(true);
				firstRdma = false;
			}
		}
    break;

  case 294:

/* Line 1806 of yacc.c  */
#line 1088 "xi-grammar.y"
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READONLY; }
    break;

  case 295:

/* Line 1806 of yacc.c  */
#line 1089 "xi-grammar.y"
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READWRITE; }
    break;

  case 296:

/* Line 1806 of yacc.c  */
#line 1090 "xi-grammar.y"
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_WRITEONLY; }
    break;

  case 297:

/* Line 1806 of yacc.c  */
#line 1093 "xi-grammar.y"
    { (yyval.xstrptr) = new XStr((yyvsp[(1) - (1)].strval)); }
    break;

  case 298:

/* Line 1806 of yacc.c  */
#line 1094 "xi-grammar.y"
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "->" << (yyvsp[(4) - (4)].strval); }
    break;

  case 299:

/* Line 1806 of yacc.c  */
#line 1095 "xi-grammar.y"
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[(1) - (3)].xstrptr)) << "." << (yyvsp[(3) - (3)].strval); }
    break;

  case 300:

/* Line 1806 of yacc.c  */
#line 1097 "xi-grammar.y"
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "[" << *((yyvsp[(3) - (4)].xstrptr)) << "]";
                  delete (yyvsp[(1) - (4)].xstrptr);
                  delete (yyvsp[(3) - (4)].xstrptr);
                }
    break;

  case 301:

/* Line 1806 of yacc.c  */
#line 1104 "xi-grammar.y"
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "[" << (yyvsp[(3) - (4)].strval) << "]";
                  delete (yyvsp[(1) - (4)].xstrptr);
                }
    break;

  case 302:

/* Line 1806 of yacc.c  */
#line 1110 "xi-grammar.y"
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "(" << *((yyvsp[(3) - (4)].xstrptr)) << ")";
                  delete (yyvsp[(1) - (4)].xstrptr);
                  delete (yyvsp[(3) - (4)].xstrptr);
                }
    break;

  case 303:

/* Line 1806 of yacc.c  */
#line 1119 "xi-grammar.y"
    {
                  in_bracket = 0;
                  (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].pname)->getType(), (yyvsp[(1) - (3)].pname)->getName(), (yyvsp[(2) - (3)].strval));
                }
    break;

  case 304:

/* Line 1806 of yacc.c  */
#line 1126 "xi-grammar.y"
    {
                  (yyval.pname) = new Parameter(lineno, (yyvsp[(3) - (7)].type), (yyvsp[(4) - (7)].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[(6) - (7)].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[(1) - (7)].intval));
                }
    break;

  case 305:

/* Line 1806 of yacc.c  */
#line 1132 "xi-grammar.y"
    {
		  (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (5)].type), (yyvsp[(2) - (5)].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[(4) - (5)].xstrptr));
                  (yyval.pname)->setAccelBufferType(Parameter::ACCEL_BUFFER_TYPE_READWRITE);
		}
    break;

  case 306:

/* Line 1806 of yacc.c  */
#line 1138 "xi-grammar.y"
    {
                  (yyval.pname) = (yyvsp[(3) - (6)].pname);
                  (yyval.pname)->setAccelInstName((yyvsp[(5) - (6)].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[(1) - (6)].intval));
		}
    break;

  case 307:

/* Line 1806 of yacc.c  */
#line 1146 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (1)].pname)); }
    break;

  case 308:

/* Line 1806 of yacc.c  */
#line 1148 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (3)].pname),(yyvsp[(3) - (3)].plist)); }
    break;

  case 309:

/* Line 1806 of yacc.c  */
#line 1152 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (1)].pname)); }
    break;

  case 310:

/* Line 1806 of yacc.c  */
#line 1154 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (3)].pname),(yyvsp[(3) - (3)].plist)); }
    break;

  case 311:

/* Line 1806 of yacc.c  */
#line 1158 "xi-grammar.y"
    { (yyval.plist) = (yyvsp[(2) - (3)].plist); }
    break;

  case 312:

/* Line 1806 of yacc.c  */
#line 1160 "xi-grammar.y"
    { (yyval.plist) = new ParamList(new Parameter(0, new BuiltinType("void"))); }
    break;

  case 313:

/* Line 1806 of yacc.c  */
#line 1164 "xi-grammar.y"
    { (yyval.plist) = (yyvsp[(2) - (3)].plist); }
    break;

  case 314:

/* Line 1806 of yacc.c  */
#line 1166 "xi-grammar.y"
    { (yyval.plist) = 0; }
    break;

  case 315:

/* Line 1806 of yacc.c  */
#line 1170 "xi-grammar.y"
    { (yyval.val) = 0; }
    break;

  case 316:

/* Line 1806 of yacc.c  */
#line 1172 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(3) - (3)].strval)); }
    break;

  case 317:

/* Line 1806 of yacc.c  */
#line 1176 "xi-grammar.y"
    { (yyval.sentry) = 0; }
    break;

  case 318:

/* Line 1806 of yacc.c  */
#line 1178 "xi-grammar.y"
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[(1) - (1)].sc)); }
    break;

  case 319:

/* Line 1806 of yacc.c  */
#line 1180 "xi-grammar.y"
    { (yyval.sentry) = new SdagEntryConstruct((yyvsp[(2) - (4)].slist)); }
    break;

  case 320:

/* Line 1806 of yacc.c  */
#line 1184 "xi-grammar.y"
    { (yyval.slist) = new SListConstruct((yyvsp[(1) - (1)].sc)); }
    break;

  case 321:

/* Line 1806 of yacc.c  */
#line 1186 "xi-grammar.y"
    { (yyval.slist) = new SListConstruct((yyvsp[(1) - (2)].sc), (yyvsp[(2) - (2)].slist));  }
    break;

  case 322:

/* Line 1806 of yacc.c  */
#line 1190 "xi-grammar.y"
    { (yyval.olist) = new OListConstruct((yyvsp[(1) - (1)].sc)); }
    break;

  case 323:

/* Line 1806 of yacc.c  */
#line 1192 "xi-grammar.y"
    { (yyval.olist) = new OListConstruct((yyvsp[(1) - (2)].sc), (yyvsp[(2) - (2)].slist)); }
    break;

  case 324:

/* Line 1806 of yacc.c  */
#line 1196 "xi-grammar.y"
    { (yyval.clist) = new CaseListConstruct((yyvsp[(1) - (1)].when)); }
    break;

  case 325:

/* Line 1806 of yacc.c  */
#line 1198 "xi-grammar.y"
    { (yyval.clist) = new CaseListConstruct((yyvsp[(1) - (2)].when), (yyvsp[(2) - (2)].clist)); }
    break;

  case 326:

/* Line 1806 of yacc.c  */
#line 1200 "xi-grammar.y"
    {
		  ERROR("case blocks can only contain when clauses",
		        (yylsp[(1) - (1)]).first_column, (yylsp[(1) - (1)]).last_column);
		  (yyval.clist) = 0;
		}
    break;

  case 327:

/* Line 1806 of yacc.c  */
#line 1208 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 328:

/* Line 1806 of yacc.c  */
#line 1210 "xi-grammar.y"
    { (yyval.strval) = 0; }
    break;

  case 329:

/* Line 1806 of yacc.c  */
#line 1214 "xi-grammar.y"
    { (yyval.when) = new WhenConstruct((yyvsp[(2) - (4)].entrylist), 0); }
    break;

  case 330:

/* Line 1806 of yacc.c  */
#line 1216 "xi-grammar.y"
    { (yyval.when) = new WhenConstruct((yyvsp[(2) - (3)].entrylist), (yyvsp[(3) - (3)].sc)); }
    break;

  case 331:

/* Line 1806 of yacc.c  */
#line 1218 "xi-grammar.y"
    { (yyval.when) = new WhenConstruct((yyvsp[(2) - (5)].entrylist), (yyvsp[(4) - (5)].slist)); }
    break;

  case 332:

/* Line 1806 of yacc.c  */
#line 1222 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 333:

/* Line 1806 of yacc.c  */
#line 1224 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 334:

/* Line 1806 of yacc.c  */
#line 1226 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 335:

/* Line 1806 of yacc.c  */
#line 1228 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 336:

/* Line 1806 of yacc.c  */
#line 1230 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 337:

/* Line 1806 of yacc.c  */
#line 1232 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 338:

/* Line 1806 of yacc.c  */
#line 1234 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 339:

/* Line 1806 of yacc.c  */
#line 1236 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 340:

/* Line 1806 of yacc.c  */
#line 1238 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 341:

/* Line 1806 of yacc.c  */
#line 1240 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 342:

/* Line 1806 of yacc.c  */
#line 1242 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 343:

/* Line 1806 of yacc.c  */
#line 1244 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 344:

/* Line 1806 of yacc.c  */
#line 1248 "xi-grammar.y"
    { (yyval.sc) = new SerialConstruct((yyvsp[(4) - (6)].strval), (yyvsp[(2) - (6)].strval), (yylsp[(3) - (6)]).first_line); }
    break;

  case 345:

/* Line 1806 of yacc.c  */
#line 1250 "xi-grammar.y"
    { (yyval.sc) = new OverlapConstruct((yyvsp[(3) - (4)].olist)); }
    break;

  case 346:

/* Line 1806 of yacc.c  */
#line 1252 "xi-grammar.y"
    { (yyval.sc) = (yyvsp[(1) - (1)].when); }
    break;

  case 347:

/* Line 1806 of yacc.c  */
#line 1254 "xi-grammar.y"
    { (yyval.sc) = new CaseConstruct((yyvsp[(3) - (4)].clist)); }
    break;

  case 348:

/* Line 1806 of yacc.c  */
#line 1256 "xi-grammar.y"
    { (yyval.sc) = new ForConstruct((yyvsp[(3) - (11)].intexpr), (yyvsp[(5) - (11)].intexpr), (yyvsp[(7) - (11)].intexpr), (yyvsp[(10) - (11)].slist)); }
    break;

  case 349:

/* Line 1806 of yacc.c  */
#line 1258 "xi-grammar.y"
    { (yyval.sc) = new ForConstruct((yyvsp[(3) - (9)].intexpr), (yyvsp[(5) - (9)].intexpr), (yyvsp[(7) - (9)].intexpr), (yyvsp[(9) - (9)].sc)); }
    break;

  case 350:

/* Line 1806 of yacc.c  */
#line 1260 "xi-grammar.y"
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[(3) - (12)].strval)), (yyvsp[(6) - (12)].intexpr),
		             (yyvsp[(8) - (12)].intexpr), (yyvsp[(10) - (12)].intexpr), (yyvsp[(12) - (12)].sc)); }
    break;

  case 351:

/* Line 1806 of yacc.c  */
#line 1263 "xi-grammar.y"
    { (yyval.sc) = new ForallConstruct(new SdagConstruct(SIDENT, (yyvsp[(3) - (14)].strval)), (yyvsp[(6) - (14)].intexpr),
		             (yyvsp[(8) - (14)].intexpr), (yyvsp[(10) - (14)].intexpr), (yyvsp[(13) - (14)].slist)); }
    break;

  case 352:

/* Line 1806 of yacc.c  */
#line 1266 "xi-grammar.y"
    { (yyval.sc) = new IfConstruct((yyvsp[(3) - (6)].intexpr), (yyvsp[(5) - (6)].sc), (yyvsp[(6) - (6)].sc)); }
    break;

  case 353:

/* Line 1806 of yacc.c  */
#line 1268 "xi-grammar.y"
    { (yyval.sc) = new IfConstruct((yyvsp[(3) - (8)].intexpr), (yyvsp[(6) - (8)].slist), (yyvsp[(8) - (8)].sc)); }
    break;

  case 354:

/* Line 1806 of yacc.c  */
#line 1270 "xi-grammar.y"
    { (yyval.sc) = new WhileConstruct((yyvsp[(3) - (5)].intexpr), (yyvsp[(5) - (5)].sc)); }
    break;

  case 355:

/* Line 1806 of yacc.c  */
#line 1272 "xi-grammar.y"
    { (yyval.sc) = new WhileConstruct((yyvsp[(3) - (7)].intexpr), (yyvsp[(6) - (7)].slist)); }
    break;

  case 356:

/* Line 1806 of yacc.c  */
#line 1274 "xi-grammar.y"
    { (yyval.sc) = new SerialConstruct((yyvsp[(2) - (4)].strval), NULL, (yyloc).first_line); }
    break;

  case 357:

/* Line 1806 of yacc.c  */
#line 1276 "xi-grammar.y"
    {
		  ERROR("unknown SDAG construct or malformed entry method declaration.\n"
		        "You may have forgotten to terminate a previous entry method declaration with a"
		        " semicolon or forgotten to mark a block of sequential SDAG code as 'serial'",
		        (yyloc).first_column, (yyloc).last_column);
		  YYABORT;
		}
    break;

  case 358:

/* Line 1806 of yacc.c  */
#line 1286 "xi-grammar.y"
    { (yyval.sc) = 0; }
    break;

  case 359:

/* Line 1806 of yacc.c  */
#line 1288 "xi-grammar.y"
    { (yyval.sc) = new ElseConstruct((yyvsp[(2) - (2)].sc)); }
    break;

  case 360:

/* Line 1806 of yacc.c  */
#line 1290 "xi-grammar.y"
    { (yyval.sc) = new ElseConstruct((yyvsp[(3) - (4)].slist)); }
    break;

  case 361:

/* Line 1806 of yacc.c  */
#line 1294 "xi-grammar.y"
    { (yyval.intexpr) = new IntExprConstruct((yyvsp[(1) - (1)].strval)); }
    break;

  case 362:

/* Line 1806 of yacc.c  */
#line 1298 "xi-grammar.y"
    { in_int_expr = 0; (yyval.intval) = 0; }
    break;

  case 363:

/* Line 1806 of yacc.c  */
#line 1302 "xi-grammar.y"
    { in_int_expr = 1; (yyval.intval) = 0; }
    break;

  case 364:

/* Line 1806 of yacc.c  */
#line 1306 "xi-grammar.y"
    {
		  (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[(1) - (2)].strval), (yyvsp[(2) - (2)].plist), 0, 0, 0, (yyloc).first_line, (yyloc).last_line);
		  firstRdma = true;
		}
    break;

  case 365:

/* Line 1806 of yacc.c  */
#line 1311 "xi-grammar.y"
    {
		  (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[(1) - (5)].strval), (yyvsp[(5) - (5)].plist), 0, 0, (yyvsp[(3) - (5)].strval), (yyloc).first_line, (yyloc).last_line);
		  firstRdma = true;
		}
    break;

  case 366:

/* Line 1806 of yacc.c  */
#line 1318 "xi-grammar.y"
    { (yyval.entrylist) = new EntryList((yyvsp[(1) - (1)].entry)); }
    break;

  case 367:

/* Line 1806 of yacc.c  */
#line 1320 "xi-grammar.y"
    { (yyval.entrylist) = new EntryList((yyvsp[(1) - (3)].entry),(yyvsp[(3) - (3)].entrylist)); }
    break;

  case 368:

/* Line 1806 of yacc.c  */
#line 1324 "xi-grammar.y"
    { in_bracket=1; }
    break;

  case 369:

/* Line 1806 of yacc.c  */
#line 1327 "xi-grammar.y"
    { in_bracket=0; }
    break;

  case 370:

/* Line 1806 of yacc.c  */
#line 1331 "xi-grammar.y"
    { if (!macroDefined((yyvsp[(2) - (2)].strval), 1)) in_comment = 1; }
    break;

  case 371:

/* Line 1806 of yacc.c  */
#line 1335 "xi-grammar.y"
    { if (!macroDefined((yyvsp[(2) - (2)].strval), 0)) in_comment = 1; }
    break;



/* Line 1806 of yacc.c  */
#line 5301 "y.tab.c"
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

  /* Now `shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*------------------------------------.
| yyerrlab -- here on detecting error |
`------------------------------------*/
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
  /* Do not reclaim the symbols of the rule which action triggered
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
  yyerrstatus = 3;	/* Each real token shifted decrements this.  */

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

  *++yyvsp = yylval;

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

#if !defined(yyoverflow) || YYERROR_VERBOSE
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
  /* Do not reclaim the symbols of the rule which action triggered
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
  /* Make sure YYID is used.  */
  return YYID (yyresult);
}



/* Line 2067 of yacc.c  */
#line 1338 "xi-grammar.y"


void yyerror(const char *msg) { }

