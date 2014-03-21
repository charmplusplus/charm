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
#define YYLSP_NEEDED 0



/* Copy the first part of user declarations.  */

/* Line 268 of yacc.c  */
#line 2 "xi-grammar.y"

#include <iostream>
#include <string>
#include <string.h>
#include "xi-symbol.h"
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


/* Line 268 of yacc.c  */
#line 94 "y.tab.c"

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




#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE
{

/* Line 293 of yacc.c  */
#line 24 "xi-grammar.y"

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
  XStr* xstrptr;
  AccelBlock* accelBlock;



/* Line 293 of yacc.c  */
#line 315 "y.tab.c"
} YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
#endif


/* Copy the second part of user declarations.  */


/* Line 343 of yacc.c  */
#line 327 "y.tab.c"

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
#define YYFINAL  55
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   1307

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  89
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  113
/* YYNRULES -- Number of rules.  */
#define YYNRULES  352
/* YYNRULES -- Number of states.  */
#define YYNSTATES  645

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   327

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
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
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     5,     6,     9,    10,    12,    13,    15,
      17,    19,    21,    23,    25,    27,    29,    31,    33,    35,
      37,    39,    41,    43,    45,    47,    49,    51,    53,    55,
      57,    59,    61,    63,    65,    67,    69,    71,    73,    75,
      77,    79,    81,    83,    85,    87,    89,    91,    93,    95,
      97,    99,   101,   103,   105,   107,   109,   111,   116,   120,
     124,   126,   131,   132,   135,   139,   142,   145,   148,   156,
     162,   168,   171,   174,   177,   180,   183,   186,   189,   192,
     194,   196,   198,   200,   202,   204,   206,   208,   212,   213,
     215,   216,   220,   222,   224,   226,   228,   231,   234,   238,
     242,   245,   248,   251,   253,   255,   258,   260,   263,   266,
     268,   270,   273,   276,   279,   288,   290,   292,   294,   296,
     299,   302,   305,   307,   309,   311,   315,   316,   319,   324,
     330,   331,   333,   334,   338,   340,   344,   346,   348,   349,
     353,   355,   359,   360,   362,   364,   365,   369,   371,   375,
     377,   379,   380,   382,   383,   386,   392,   394,   397,   401,
     408,   409,   412,   414,   418,   424,   430,   436,   442,   447,
     451,   458,   465,   471,   477,   483,   489,   495,   500,   508,
     509,   512,   513,   516,   519,   523,   526,   530,   532,   536,
     541,   544,   547,   550,   553,   556,   558,   563,   564,   567,
     569,   571,   573,   575,   578,   581,   584,   588,   595,   605,
     609,   616,   620,   627,   637,   647,   649,   653,   655,   658,
     661,   663,   666,   668,   670,   672,   674,   676,   678,   680,
     682,   684,   686,   688,   690,   698,   704,   717,   723,   726,
     728,   730,   731,   735,   737,   739,   743,   745,   747,   749,
     751,   753,   755,   757,   759,   761,   763,   765,   767,   769,
     772,   774,   776,   778,   780,   782,   784,   785,   787,   791,
     792,   794,   800,   806,   812,   817,   821,   823,   825,   827,
     831,   836,   840,   842,   844,   846,   848,   853,   857,   862,
     867,   872,   876,   884,   890,   897,   899,   903,   905,   909,
     913,   916,   920,   923,   924,   928,   929,   931,   935,   937,
     940,   942,   945,   947,   950,   952,   954,   955,   960,   964,
     970,   972,   974,   976,   978,   980,   982,   988,   993,   995,
    1000,  1012,  1022,  1035,  1050,  1057,  1066,  1072,  1080,  1084,
    1086,  1087,  1090,  1095,  1097,  1099,  1102,  1108,  1110,  1114,
    1116,  1118,  1121
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
      90,     0,    -1,    91,    -1,    -1,    96,    91,    -1,    -1,
       5,    -1,    -1,    73,    -1,    53,    -1,     3,    -1,     4,
      -1,     5,    -1,     7,    -1,     8,    -1,     9,    -1,    11,
      -1,    12,    -1,    13,    -1,    14,    -1,    15,    -1,    19,
      -1,    20,    -1,    21,    -1,    22,    -1,    23,    -1,    24,
      -1,    25,    -1,    26,    -1,    27,    -1,    28,    -1,    29,
      -1,    30,    -1,    31,    -1,    32,    -1,    33,    -1,    34,
      -1,    35,    -1,    38,    -1,    39,    -1,    40,    -1,    41,
      -1,    42,    -1,    43,    -1,    44,    -1,    45,    -1,    46,
      -1,    47,    -1,    48,    -1,    50,    -1,    52,    -1,    66,
      -1,    69,    -1,    70,    -1,    71,    -1,    72,    -1,    53,
      -1,    95,    74,    74,    53,    -1,     3,    94,    97,    -1,
       4,    94,    97,    -1,    73,    -1,    75,    98,    76,    93,
      -1,    -1,   100,    98,    -1,    52,    51,    95,    -1,    52,
      95,    -1,    92,   155,    -1,    92,   134,    -1,     5,    40,
     165,   107,    94,   104,   182,    -1,    92,    75,    98,    76,
      93,    -1,    51,    94,    75,    98,    76,    -1,    99,    73,
      -1,    99,   162,    -1,    92,    96,    -1,    92,   137,    -1,
      92,   138,    -1,    92,   139,    -1,    92,   141,    -1,    92,
     152,    -1,   200,    -1,   201,    -1,   164,    -1,     1,    -1,
     113,    -1,    54,    -1,    55,    -1,   101,    -1,   101,    77,
     102,    -1,    -1,   102,    -1,    -1,    78,   103,    79,    -1,
      59,    -1,    60,    -1,    61,    -1,    62,    -1,    65,    59,
      -1,    65,    60,    -1,    65,    60,    59,    -1,    65,    60,
      60,    -1,    65,    61,    -1,    65,    62,    -1,    60,    60,
      -1,    63,    -1,    64,    -1,    60,    64,    -1,    36,    -1,
      94,   104,    -1,    95,   104,    -1,   105,    -1,   107,    -1,
     108,    80,    -1,   109,    80,    -1,   110,    80,    -1,   112,
      81,    80,    94,    82,    81,   180,    82,    -1,   108,    -1,
     109,    -1,   110,    -1,   111,    -1,    37,   112,    -1,   112,
      37,    -1,   112,    83,    -1,   112,    -1,    54,    -1,    95,
      -1,    84,   114,    85,    -1,    -1,   115,   116,    -1,     6,
     113,    95,   116,    -1,     6,    16,   108,    80,    94,    -1,
      -1,    36,    -1,    -1,    84,   121,    85,    -1,   122,    -1,
     122,    77,   121,    -1,    38,    -1,    39,    -1,    -1,    84,
     124,    85,    -1,   129,    -1,   129,    77,   124,    -1,    -1,
      55,    -1,    49,    -1,    -1,    84,   128,    85,    -1,   126,
      -1,   126,    77,   128,    -1,    30,    -1,    49,    -1,    -1,
      17,    -1,    -1,    84,    85,    -1,   130,   113,    94,   131,
      73,    -1,   132,    -1,   132,   133,    -1,    16,   120,   106,
      -1,    16,   120,   106,    75,   133,    76,    -1,    -1,    74,
     136,    -1,   107,    -1,   107,    77,   136,    -1,    11,   123,
     106,   135,   153,    -1,    12,   123,   106,   135,   153,    -1,
      13,   123,   106,   135,   153,    -1,    14,   123,   106,   135,
     153,    -1,    84,    54,    94,    85,    -1,    84,    94,    85,
      -1,    15,   127,   140,   106,   135,   153,    -1,    15,   140,
     127,   106,   135,   153,    -1,    11,   123,    94,   135,   153,
      -1,    12,   123,    94,   135,   153,    -1,    13,   123,    94,
     135,   153,    -1,    14,   123,    94,   135,   153,    -1,    15,
     140,    94,   135,   153,    -1,    16,   120,    94,    73,    -1,
      16,   120,    94,    75,   133,    76,    73,    -1,    -1,    86,
     113,    -1,    -1,    86,    54,    -1,    86,    55,    -1,    18,
      94,   147,    -1,   111,   148,    -1,   113,    94,   148,    -1,
     149,    -1,   149,    77,   150,    -1,    22,    78,   150,    79,
      -1,   151,   142,    -1,   151,   143,    -1,   151,   144,    -1,
     151,   145,    -1,   151,   146,    -1,    73,    -1,    75,   154,
      76,    93,    -1,    -1,   160,   154,    -1,   117,    -1,   118,
      -1,   157,    -1,   156,    -1,    10,   158,    -1,    19,   159,
      -1,    18,    94,    -1,     8,   119,    95,    -1,     8,   119,
      95,    81,   119,    82,    -1,     8,   119,    95,    78,   102,
      79,    81,   119,    82,    -1,     7,   119,    95,    -1,     7,
     119,    95,    81,   119,    82,    -1,     9,   119,    95,    -1,
       9,   119,    95,    81,   119,    82,    -1,     9,   119,    95,
      78,   102,    79,    81,   119,    82,    -1,     9,    84,    66,
      85,   119,    95,    81,   119,    82,    -1,   107,    -1,   107,
      77,   158,    -1,    55,    -1,   161,    73,    -1,   161,   162,
      -1,   163,    -1,   151,   163,    -1,   155,    -1,    40,    -1,
      76,    -1,     7,    -1,     8,    -1,     9,    -1,    11,    -1,
      12,    -1,    15,    -1,    13,    -1,    14,    -1,     6,    -1,
      40,   166,   165,    94,   182,   184,   185,    -1,    40,   166,
      94,   182,   185,    -1,    40,    84,    66,    85,    36,    94,
     182,   183,   173,   171,   174,    94,    -1,    69,   173,   171,
     174,    73,    -1,    69,    73,    -1,    36,    -1,   109,    -1,
      -1,    84,   167,    85,    -1,     1,    -1,   168,    -1,   168,
      77,   167,    -1,    21,    -1,    23,    -1,    24,    -1,    25,
      -1,    31,    -1,    32,    -1,    33,    -1,    34,    -1,    35,
      -1,    26,    -1,    27,    -1,    28,    -1,    50,    -1,    49,
     125,    -1,    70,    -1,    71,    -1,     1,    -1,    55,    -1,
      54,    -1,    95,    -1,    -1,    56,    -1,    56,    77,   170,
      -1,    -1,    56,    -1,    56,    84,   171,    85,   171,    -1,
      56,    75,   171,    76,   171,    -1,    56,    81,   170,    82,
     171,    -1,    81,   171,    82,   171,    -1,   113,    94,    84,
      -1,    75,    -1,    76,    -1,   113,    -1,   113,    94,   130,
      -1,   113,    94,    86,   169,    -1,   172,   171,    85,    -1,
       6,    -1,    67,    -1,    68,    -1,    94,    -1,   177,    87,
      79,    94,    -1,   177,    88,    94,    -1,   177,    84,   177,
      85,    -1,   177,    84,    54,    85,    -1,   177,    81,   177,
      82,    -1,   172,   171,    85,    -1,   176,    74,   113,    94,
      78,   177,    79,    -1,   113,    94,    78,   177,    79,    -1,
     176,    74,   178,    78,   177,    79,    -1,   175,    -1,   175,
      77,   180,    -1,   179,    -1,   179,    77,   181,    -1,    81,
     180,    82,    -1,    81,    82,    -1,    84,   181,    85,    -1,
      84,    85,    -1,    -1,    20,    86,    54,    -1,    -1,   192,
      -1,    75,   186,    76,    -1,   192,    -1,   192,   186,    -1,
     192,    -1,   192,   186,    -1,   190,    -1,   190,   188,    -1,
     191,    -1,    55,    -1,    -1,    44,   197,    75,    76,    -1,
      44,   197,   192,    -1,    44,   197,    75,   186,    76,    -1,
      46,    -1,    45,    -1,    41,    -1,    42,    -1,    47,    -1,
      43,    -1,    46,   189,   173,   171,   174,    -1,    45,    75,
     187,    76,    -1,   190,    -1,    72,    75,   188,    76,    -1,
      41,   195,   171,    73,   171,    73,   171,   194,    75,   186,
      76,    -1,    41,   195,   171,    73,   171,    73,   171,   194,
     192,    -1,    42,    84,    53,    85,   195,   171,    74,   171,
      77,   171,   194,   192,    -1,    42,    84,    53,    85,   195,
     171,    74,   171,    77,   171,   194,    75,   186,    76,    -1,
      47,   195,   171,   194,   192,   193,    -1,    47,   195,   171,
     194,    75,   186,    76,   193,    -1,    43,   195,   171,   194,
     192,    -1,    43,   195,   171,   194,    75,   186,    76,    -1,
     173,   171,   174,    -1,     1,    -1,    -1,    48,   192,    -1,
      48,    75,   186,    76,    -1,    82,    -1,    81,    -1,    53,
     182,    -1,    53,   198,   171,   199,   182,    -1,   196,    -1,
     196,    77,   197,    -1,    84,    -1,    85,    -1,    57,    94,
      -1,    58,    94,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   155,   155,   160,   163,   168,   169,   174,   175,   180,
     182,   183,   184,   186,   187,   188,   190,   191,   192,   193,
     194,   198,   199,   200,   201,   202,   203,   204,   205,   206,
     207,   208,   209,   210,   211,   212,   213,   214,   217,   218,
     219,   220,   221,   222,   223,   224,   225,   226,   227,   229,
     231,   232,   235,   236,   237,   238,   241,   243,   251,   255,
     262,   264,   269,   270,   274,   276,   278,   280,   282,   294,
     296,   298,   300,   302,   304,   306,   308,   310,   312,   314,
     316,   318,   320,   324,   326,   328,   332,   334,   339,   340,
     345,   346,   350,   352,   354,   356,   358,   360,   362,   364,
     366,   368,   370,   372,   374,   376,   378,   382,   383,   390,
     392,   396,   400,   402,   406,   410,   412,   414,   416,   419,
     421,   425,   427,   431,   433,   437,   442,   443,   447,   451,
     456,   457,   462,   463,   473,   475,   479,   481,   486,   487,
     491,   493,   498,   499,   503,   508,   509,   513,   515,   519,
     521,   526,   527,   531,   532,   535,   539,   541,   545,   547,
     552,   553,   557,   559,   563,   565,   569,   573,   577,   583,
     587,   589,   593,   595,   599,   603,   607,   611,   613,   618,
     619,   624,   625,   627,   631,   633,   635,   639,   641,   645,
     649,   651,   653,   655,   657,   661,   663,   668,   675,   679,
     681,   683,   684,   686,   688,   690,   694,   696,   698,   704,
     707,   712,   714,   716,   722,   730,   732,   735,   739,   742,
     746,   748,   753,   757,   759,   761,   763,   765,   767,   769,
     771,   773,   775,   777,   780,   790,   805,   821,   823,   827,
     829,   834,   835,   837,   841,   843,   847,   849,   851,   853,
     855,   857,   859,   861,   863,   865,   867,   869,   871,   873,
     875,   877,   879,   883,   885,   887,   892,   893,   895,   904,
     905,   907,   913,   919,   925,   933,   940,   948,   955,   957,
     959,   961,   968,   969,   970,   973,   974,   975,   976,   983,
     989,   998,  1005,  1011,  1017,  1025,  1027,  1031,  1033,  1037,
    1039,  1043,  1045,  1050,  1051,  1056,  1057,  1059,  1063,  1065,
    1069,  1071,  1075,  1077,  1079,  1083,  1086,  1089,  1091,  1093,
    1097,  1099,  1101,  1103,  1105,  1107,  1111,  1113,  1115,  1117,
    1119,  1122,  1125,  1128,  1131,  1133,  1135,  1137,  1139,  1141,
    1148,  1149,  1151,  1155,  1159,  1163,  1165,  1169,  1171,  1175,
    1178,  1182,  1186
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
     325,   326,   327,    59,    58,   123,   125,    44,    60,    62,
      42,    40,    41,    38,    91,    93,    61,    45,    46
};
# endif

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
     112,   113,   113,   114,   114,   115,   116,   116,   117,   118,
     119,   119,   120,   120,   121,   121,   122,   122,   123,   123,
     124,   124,   125,   125,   126,   127,   127,   128,   128,   129,
     129,   130,   130,   131,   131,   132,   133,   133,   134,   134,
     135,   135,   136,   136,   137,   137,   138,   139,   140,   140,
     141,   141,   142,   142,   143,   144,   145,   146,   146,   147,
     147,   148,   148,   148,   149,   149,   149,   150,   150,   151,
     152,   152,   152,   152,   152,   153,   153,   154,   154,   155,
     155,   155,   155,   155,   155,   155,   156,   156,   156,   156,
     156,   157,   157,   157,   157,   158,   158,   159,   160,   160,
     161,   161,   161,   162,   162,   162,   162,   162,   162,   162,
     162,   162,   162,   162,   163,   163,   163,   164,   164,   165,
     165,   166,   166,   166,   167,   167,   168,   168,   168,   168,
     168,   168,   168,   168,   168,   168,   168,   168,   168,   168,
     168,   168,   168,   169,   169,   169,   170,   170,   170,   171,
     171,   171,   171,   171,   171,   172,   173,   174,   175,   175,
     175,   175,   176,   176,   176,   177,   177,   177,   177,   177,
     177,   178,   179,   179,   179,   180,   180,   181,   181,   182,
     182,   183,   183,   184,   184,   185,   185,   185,   186,   186,
     187,   187,   188,   188,   188,   189,   189,   190,   190,   190,
     191,   191,   191,   191,   191,   191,   192,   192,   192,   192,
     192,   192,   192,   192,   192,   192,   192,   192,   192,   192,
     193,   193,   193,   194,   195,   196,   196,   197,   197,   198,
     199,   200,   201
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
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
       2,     2,     1,     1,     1,     3,     0,     2,     4,     5,
       0,     1,     0,     3,     1,     3,     1,     1,     0,     3,
       1,     3,     0,     1,     1,     0,     3,     1,     3,     1,
       1,     0,     1,     0,     2,     5,     1,     2,     3,     6,
       0,     2,     1,     3,     5,     5,     5,     5,     4,     3,
       6,     6,     5,     5,     5,     5,     5,     4,     7,     0,
       2,     0,     2,     2,     3,     2,     3,     1,     3,     4,
       2,     2,     2,     2,     2,     1,     4,     0,     2,     1,
       1,     1,     1,     2,     2,     2,     3,     6,     9,     3,
       6,     3,     6,     9,     9,     1,     3,     1,     2,     2,
       1,     2,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     7,     5,    12,     5,     2,     1,
       1,     0,     3,     1,     1,     3,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     2,
       1,     1,     1,     1,     1,     1,     0,     1,     3,     0,
       1,     5,     5,     5,     4,     3,     1,     1,     1,     3,
       4,     3,     1,     1,     1,     1,     4,     3,     4,     4,
       4,     3,     7,     5,     6,     1,     3,     1,     3,     3,
       2,     3,     2,     0,     3,     0,     1,     3,     1,     2,
       1,     2,     1,     2,     1,     1,     0,     4,     3,     5,
       1,     1,     1,     1,     1,     1,     5,     4,     1,     4,
      11,     9,    12,    14,     6,     8,     5,     7,     3,     1,
       0,     2,     4,     1,     1,     2,     5,     1,     3,     1,
       1,     2,     2
};

/* YYDEFACT[STATE-NAME] -- Default reduction number in state STATE-NUM.
   Performed when YYTABLE doesn't specify something else to do.  Zero
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
     351,   352,   238,   276,   269,     0,   130,   130,   130,     0,
     138,   138,   138,   138,     0,   132,     0,     0,     0,     0,
      73,   199,   200,    67,    74,    75,    76,    77,     0,    78,
      66,   202,   201,     7,   233,   225,   226,   227,   228,   229,
     231,   232,   230,   223,    71,   224,    72,    63,   239,    92,
      93,    94,    95,   103,   104,     0,    90,   109,   110,     0,
     240,     0,     0,    64,     0,   270,   269,     0,     0,   106,
       0,   115,   116,   117,   118,   122,     0,   131,     0,     0,
       0,     0,   215,   203,     0,     0,     0,     0,     0,     0,
       0,   145,     0,     0,   205,   217,   204,     0,     0,   138,
     138,   138,   138,     0,   132,   190,   191,   192,   193,   194,
       8,    61,   102,   105,    96,    97,   100,   101,    88,   108,
     111,     0,     0,     0,   269,   266,   269,     0,   277,     0,
       0,   119,   112,   113,   120,     0,   121,   126,   209,   206,
       0,   211,     0,   149,   150,     0,   140,    90,   160,   160,
     160,   160,   144,     0,     0,   147,     0,     0,     0,     0,
       0,   136,   137,     0,   134,   158,     0,   118,     0,   187,
       0,     7,     0,     0,     0,     0,     0,     0,    98,    99,
      84,    85,    86,    89,     0,    83,    90,    70,    57,     0,
     267,     0,     0,   269,   237,     0,     0,     0,   126,   128,
     130,     0,   130,   130,     0,   130,   216,   139,     0,   107,
       0,     0,     0,     0,     0,     0,   169,     0,   146,   160,
     160,   133,     0,   151,   179,     0,   185,   181,     0,   189,
      69,   160,   160,   160,   160,   160,     0,     0,    91,     0,
     269,   266,   269,   269,   274,   129,     0,   123,   124,     0,
     127,     0,     0,     0,     0,     0,     0,   141,   162,   161,
     195,   197,   164,   165,   166,   167,   168,   148,     0,     0,
     135,   152,     0,   151,     0,     0,   184,   182,   183,   186,
     188,     0,     0,     0,     0,     0,   177,   151,    87,     0,
      68,   272,   268,   273,   271,     0,   125,   210,     0,   207,
       0,     0,   212,     0,     0,     0,     0,   222,   197,     0,
     220,   170,   171,     0,   157,   159,   180,   172,   173,   174,
     175,   176,     0,   300,   278,   269,   295,     0,     0,   130,
     130,   130,   163,   243,     0,     0,   221,     7,   198,   218,
     219,   153,     0,   151,     0,     0,   299,     0,     0,     0,
       0,   262,   246,   247,   248,   249,   255,   256,   257,   250,
     251,   252,   253,   254,   142,   258,     0,   260,   261,     0,
     244,    56,     0,     0,   196,     0,     0,   178,   275,     0,
     279,   281,   296,   114,   208,   214,   213,   143,   259,     0,
     242,     0,     0,     0,   154,   155,   264,   263,   265,   280,
       0,   245,   339,     0,     0,     0,     0,     0,   316,     0,
       0,     0,   269,   235,   328,   306,   303,     0,   344,   269,
       0,   269,     0,   347,     0,     0,   315,     0,   269,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   349,
     345,   269,     0,     0,   318,     0,     0,   269,     0,   322,
     323,   325,   321,   320,   324,     0,   312,   314,   307,   309,
     338,     0,   234,     0,     0,   269,     0,   343,     0,     0,
     348,   317,     0,   327,   311,     0,     0,   329,   313,   304,
     282,   283,   284,   302,     0,     0,   297,     0,   269,     0,
     269,     0,   336,   350,     0,   319,   326,     0,   340,     0,
       0,     0,   301,     0,   269,     0,     0,   346,     0,     0,
     334,     0,     0,   269,     0,   298,     0,     0,   269,   337,
     340,     0,   341,   285,     0,     0,     0,     0,   236,     0,
       0,   335,     0,   293,     0,     0,     0,     0,     0,   291,
       0,     0,   331,   269,   342,     0,     0,     0,     0,   287,
       0,   294,     0,     0,   290,   289,   288,   286,   292,   330,
       0,     0,   332,     0,   333
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     3,     4,    68,   191,   227,   136,     5,    59,    69,
      70,    71,   262,   263,   264,   199,   137,   228,   138,   151,
     152,   153,   154,   155,   265,   329,   278,   279,   101,   102,
     158,   173,   243,   244,   165,   225,   468,   235,   170,   236,
     226,   352,   456,   353,   354,   103,   291,   339,   104,   105,
     106,   171,   107,   185,   186,   187,   188,   189,   356,   306,
     249,   250,   385,   109,   342,   386,   387,   111,   112,   163,
     176,   388,   389,   126,   390,    72,   141,   415,   449,   450,
     479,   271,   147,   405,   492,   209,   406,   565,   604,   594,
     566,   407,   567,   370,   544,   514,   493,   510,   525,   535,
     507,   494,   537,   511,   590,   548,   499,   503,   504,   521,
     574,    73,    74
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -517
static const yytype_int16 yypact[] =
{
     101,  1128,  1128,    44,  -517,   101,  -517,  -517,  -517,  -517,
    -517,  -517,  -517,  -517,  -517,  -517,  -517,  -517,  -517,  -517,
    -517,  -517,  -517,  -517,  -517,  -517,  -517,  -517,  -517,  -517,
    -517,  -517,  -517,  -517,  -517,  -517,  -517,  -517,  -517,  -517,
    -517,  -517,  -517,  -517,  -517,  -517,  -517,  -517,  -517,  -517,
    -517,  -517,  -517,   132,   132,  -517,  -517,  -517,   567,  -517,
    -517,  -517,    37,  1128,   173,  1128,  1128,   160,   712,    53,
     459,   567,  -517,  -517,  -517,   241,    35,    87,  -517,    75,
    -517,  -517,  -517,  -517,   -22,  1148,   117,   117,   -15,    87,
      90,    90,    90,    90,   105,   110,  1128,   142,   140,   567,
    -517,  -517,  -517,  -517,  -517,  -517,  -517,  -517,   339,  -517,
    -517,  -517,  -517,   164,  -517,  -517,  -517,  -517,  -517,  -517,
    -517,  -517,  -517,  -517,  -517,  -517,  -517,  -517,   166,  -517,
     124,  -517,  -517,  -517,  -517,   161,    18,  -517,  -517,   188,
    -517,    87,   567,    75,   167,   -35,   -22,   197,  1242,  -517,
    1224,   188,   207,   210,  -517,   -10,    87,  -517,    87,    87,
     245,    87,   230,  -517,    21,  1128,  1128,  1128,  1128,   918,
     231,   233,   226,  1128,  -517,  -517,  -517,  1168,   234,    90,
      90,    90,    90,   231,   110,  -517,  -517,  -517,  -517,  -517,
    -517,  -517,  -517,  -517,  -517,   211,  -517,  -517,  1211,  -517,
    -517,  1128,   248,   267,   -22,   265,   -22,   247,  -517,   260,
     254,    -7,  -517,  -517,  -517,   263,  -517,   -32,    89,    50,
     259,    67,    87,  -517,  -517,   261,   271,   279,   285,   285,
     285,   285,  -517,  1128,   275,   284,   277,   988,  1128,   314,
    1128,  -517,  -517,   282,   291,   296,  1128,    32,  1128,   295,
     297,   164,  1128,  1128,  1128,  1128,  1128,  1128,  -517,  -517,
    -517,  -517,   300,  -517,   301,  -517,   279,  -517,  -517,   298,
     302,   299,   288,   -22,  -517,  1128,  1128,   243,   303,  -517,
     117,  1211,   117,   117,  1211,   117,  -517,  -517,    21,  -517,
      87,   169,   169,   169,   169,   304,  -517,   314,  -517,   285,
     285,  -517,   226,   368,   305,   244,  -517,   307,  1168,  -517,
    -517,   285,   285,   285,   285,   285,   184,  1211,  -517,   309,
     -22,   265,   -22,   -22,  -517,  -517,   312,  -517,    75,   315,
    -517,   329,   333,   331,    87,   343,   341,  -517,   347,  -517,
    -517,   221,  -517,  -517,  -517,  -517,  -517,  -517,   169,   169,
    -517,  -517,  1224,     3,   310,  1224,  -517,  -517,  -517,  -517,
    -517,   169,   169,   169,   169,   169,  -517,   368,  -517,  1181,
    -517,  -517,  -517,  -517,  -517,   345,  -517,  -517,   355,  -517,
      91,   356,  -517,    87,   481,   400,   366,  -517,   221,   761,
    -517,  -517,  -517,  1128,  -517,  -517,  -517,  -517,  -517,  -517,
    -517,  -517,   367,  -517,  1128,   -22,   370,   369,  1224,   117,
     117,   117,  -517,  -517,   728,   848,  -517,   164,  -517,  -517,
    -517,   361,   375,     1,   371,  1224,  -517,   372,   376,   379,
     381,  -517,  -517,  -517,  -517,  -517,  -517,  -517,  -517,  -517,
    -517,  -517,  -517,  -517,   398,  -517,   384,  -517,  -517,   390,
     380,   383,   309,  1128,  -517,   391,   404,  -517,  -517,   195,
    -517,  -517,  -517,  -517,  -517,  -517,  -517,  -517,  -517,   442,
    -517,   779,   638,   309,  -517,  -517,  -517,  -517,    75,  -517,
    1128,  -517,  -517,   402,   403,   402,   438,   443,   475,   402,
     461,   115,   -22,  -517,  -517,  -517,   517,   309,  -517,   -22,
     485,   -22,    85,   462,    92,   374,  -517,   473,   -22,   694,
     478,   294,   197,   463,   638,   471,   483,   474,   476,  -517,
    -517,   -22,   438,   237,  -517,   484,   363,   -22,   476,  -517,
    -517,  -517,  -517,  -517,  -517,   486,   694,  -517,  -517,  -517,
    -517,   507,  -517,   149,   473,   -22,   402,  -517,   387,   479,
    -517,  -517,   490,  -517,  -517,   197,   556,  -517,  -517,  -517,
    -517,  -517,  -517,  -517,  1128,   493,   492,   499,   -22,   514,
     -22,   115,  -517,  -517,   309,  -517,  -517,   115,   540,   515,
    1224,   780,  -517,   197,   -22,   518,   520,  -517,   528,   614,
    -517,  1128,  1128,   -22,   516,  -517,  1128,   476,   -22,  -517,
     540,   115,  -517,  -517,    94,     2,   522,  1128,  -517,   621,
     531,  -517,   529,  -517,  1128,  1058,   530,  1128,  1128,  -517,
     179,   115,  -517,   -22,  -517,   111,   525,   204,  1128,  -517,
     235,  -517,   535,   476,  -517,  -517,  -517,  -517,  -517,  -517,
     628,   115,  -517,   536,  -517
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -517,  -517,   615,  -517,  -241,    -1,   -58,   553,   569,   -54,
    -517,  -517,  -517,  -249,  -517,  -191,  -517,  -113,   -75,   -70,
     -68,  -517,  -166,   480,   -83,  -517,  -517,   348,  -517,  -517,
     -79,   448,   332,  -517,   -67,   349,  -517,  -517,   464,   336,
    -517,   215,  -517,  -517,  -270,  -517,  -192,   257,  -517,  -517,
    -517,   -81,  -517,  -517,  -517,  -517,  -517,  -517,  -517,   334,
    -517,   340,   586,  -517,    34,   289,   608,  -517,  -517,   465,
    -517,  -517,  -517,   306,   313,  -517,   273,  -517,   219,  -517,
    -517,   373,  -143,   112,   -63,  -499,  -517,  -517,  -464,  -517,
    -517,  -367,   116,  -430,  -517,  -517,   177,  -495,  -517,   163,
    -517,  -486,  -517,  -457,   102,  -516,  -456,  -517,   182,  -517,
    -517,  -517,  -517
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -311
static const yytype_int16 yytable[] =
{
      53,    54,   156,   207,    84,   139,    79,   140,   159,   161,
     310,   247,   556,   540,   162,   495,   539,   127,   351,   143,
     351,   157,   472,   536,   166,   167,   168,   214,   552,   501,
     214,   554,   332,   508,   145,   335,   289,   292,   293,   294,
     204,   427,   144,   496,    55,   178,   205,   524,   526,   206,
     536,   223,   277,   229,   230,   231,   576,   495,   462,   146,
     245,   269,    76,   272,    80,    81,   201,   515,   368,   160,
     224,   215,   520,   216,   215,   319,   586,    75,   210,  -156,
     618,   609,   588,   394,   596,   458,   458,   459,   202,   238,
     570,   572,   144,   482,   248,   174,   198,   402,   217,   578,
     218,   219,   256,   221,     1,     2,   612,   348,   349,  -181,
     142,  -181,   252,   253,   254,   255,   482,   640,   305,   361,
     362,   363,   364,   365,   144,   299,   632,   300,   281,   113,
     324,   282,   602,   483,   484,   485,   486,   487,   488,   489,
      78,   144,   247,   620,   587,   284,   643,   162,   285,   144,
     625,   627,   622,   157,   630,   560,   483,   484,   485,   486,
     487,   488,   489,   144,   490,   144,   369,   523,   234,   519,
     280,  -276,   410,   613,   164,   614,   454,   371,   615,   373,
     374,   616,   617,   642,   192,   149,   150,   490,   193,   169,
      83,  -276,   614,   634,   172,   615,  -276,   175,   616,   617,
     266,   331,    78,   333,   334,    57,   336,    58,   129,   130,
     131,   132,   133,   134,   135,   338,   561,   562,   177,   328,
     194,   195,   196,   197,    77,   248,    78,    85,    86,    87,
      88,    89,   295,    82,   563,    83,   234,   190,   482,    96,
      97,   203,   340,    98,   341,   304,  -106,   307,    78,   476,
     477,   311,   312,   313,   314,   315,   316,   366,   631,   367,
     614,   384,   424,   615,   241,   242,   616,   617,   200,   393,
     258,   259,   396,   208,   325,   326,   380,   128,   483,   484,
     485,   486,   487,   488,   489,   614,   404,   212,   615,   636,
     213,   616,   617,  -276,    78,   482,    78,   327,   357,   358,
     129,   130,   131,   132,   133,   134,   135,   222,   338,   490,
     251,   220,    83,   551,   638,   237,   614,   239,  -276,   615,
     268,   270,   616,   617,   267,   404,   343,   344,   345,   273,
     428,   429,   430,   274,   275,   483,   484,   485,   486,   487,
     488,   489,   404,   276,   283,   139,   287,   140,   288,   512,
     179,   180,   181,   182,   183,   184,   516,   198,   518,   290,
     296,   297,   298,   232,   482,   528,   490,   301,   302,    83,
    -308,   303,   308,   323,   320,   482,   309,   317,   549,   321,
     318,   322,   391,   392,   555,   351,   395,   277,   482,   346,
     369,   355,   421,   305,   375,   397,   398,   399,   400,   401,
     376,   478,   569,   423,   483,   484,   485,   486,   487,   488,
     489,   377,   378,   379,   452,   483,   484,   485,   486,   487,
     488,   489,   381,   382,   383,   583,   408,   585,   483,   484,
     485,   486,   487,   488,   489,   490,   409,   411,    83,  -310,
     384,   597,   417,   422,   527,   455,   490,   425,   457,    83,
     606,   426,   473,   467,   463,   610,   461,   471,   464,   490,
     564,   465,   571,   466,    -9,   114,   115,   116,   117,   469,
     118,   119,   120,   121,   122,   470,   474,   475,   480,   497,
     633,   568,   413,   498,  -241,  -241,  -241,   500,  -241,  -241,
    -241,   502,  -241,  -241,  -241,  -241,  -241,   592,   564,   123,
    -241,  -241,  -241,  -241,  -241,  -241,  -241,  -241,  -241,  -241,
    -241,  -241,  -241,  -241,  -241,  -241,  -241,  -241,   505,  -241,
    -241,  -241,  -241,  -241,  -241,  -241,  -241,  -241,  -241,  -241,
     506,  -241,   124,  -241,  -241,   125,   509,   513,   517,   522,
    -241,  -241,  -241,  -241,  -241,  -241,  -241,  -241,    83,   541,
    -241,  -241,  -241,  -241,   538,   543,   545,   482,   547,   546,
     553,   559,   557,   579,   573,   414,   575,   580,    61,   581,
      -5,    -5,    62,    -5,    -5,    -5,    -5,    -5,    -5,    -5,
      -5,    -5,    -5,    -5,   582,    -5,    -5,   584,   589,    -5,
     603,   605,   598,   591,   607,   608,   599,   483,   484,   485,
     486,   487,   488,   489,   600,   624,   603,   619,   623,   628,
     635,   639,   644,   603,   603,   482,   629,   603,    63,    64,
      56,   100,   482,    60,    65,    66,   330,   637,   490,   482,
     211,   577,   257,   347,   350,   240,    67,   337,   460,   482,
     412,   359,    -5,   -62,  -305,  -305,  -305,  -305,   360,  -305,
    -305,  -305,  -305,  -305,   108,   483,   484,   485,   486,   487,
     488,   489,   483,   484,   485,   486,   487,   488,   489,   483,
     484,   485,   486,   487,   488,   489,   110,   418,  -305,   483,
     484,   485,   486,   487,   488,   489,   490,   286,   453,   601,
     481,   542,   593,   490,   372,   420,   621,   595,   416,   558,
     490,     0,   611,   641,   550,     0,     0,     0,     0,     0,
     490,  -305,     0,   491,  -305,     1,     2,     0,    85,    86,
      87,    88,    89,    90,    91,    92,    93,    94,    95,   431,
      96,    97,     0,     0,    98,   529,   530,   531,   486,   532,
     533,   534,     0,     0,     0,     0,     0,     0,     0,   432,
       0,   433,   434,   435,   436,   437,   438,     0,     0,   439,
     440,   441,   442,   443,     0,     0,     0,   114,   115,   116,
     117,     0,   118,   119,   120,   121,   122,   444,   445,     0,
     431,     0,     0,     0,     0,     0,   560,    99,     0,     0,
       0,     0,     0,     0,   446,     0,     0,     0,   447,   448,
     432,   123,   433,   434,   435,   436,   437,   438,     0,     0,
     439,   440,   441,   442,   443,     0,   149,   150,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   444,   445,
       0,     0,     0,    78,   419,     0,     0,   125,     0,   129,
     130,   131,   132,   133,   134,   135,     0,   561,   562,   447,
     448,     6,     7,     8,     0,     9,    10,    11,     0,    12,
      13,    14,    15,    16,     0,     0,     0,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    31,    32,    33,   128,     0,    34,    35,    36,    37,
      38,    39,    40,    41,    42,    43,    44,     0,    45,     0,
      46,   451,     0,     0,     0,     0,     0,   129,   130,   131,
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
      46,    47,   626,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    48,     0,     0,    49,    50,    51,
      52,     6,     7,     8,     0,     9,    10,    11,     0,    12,
      13,    14,    15,    16,     0,     0,     0,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    31,    32,    33,   148,     0,    34,    35,    36,    37,
      38,    39,    40,    41,    42,    43,    44,     0,    45,     0,
      46,    47,     0,     0,   149,   150,   246,     0,     0,     0,
       0,     0,     0,     0,    48,     0,     0,    49,    50,    51,
      52,    78,     0,     0,   149,   150,     0,   129,   130,   131,
     132,   133,   134,   135,     0,     0,     0,   149,   150,     0,
       0,    78,     0,     0,     0,     0,     0,   129,   130,   131,
     132,   133,   134,   135,    78,     0,     0,     0,     0,     0,
     129,   130,   131,   132,   133,   134,   135,   149,   150,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     149,   150,     0,   403,    78,   260,   261,     0,     0,     0,
     129,   130,   131,   132,   133,   134,   135,    78,   149,     0,
       0,     0,     0,   129,   130,   131,   132,   133,   134,   135,
       0,     0,     0,     0,     0,    78,     0,     0,     0,     0,
       0,   129,   130,   131,   132,   133,   134,   135
};

#define yypact_value_is_default(yystate) \
  ((yystate) == (-517))

#define yytable_value_is_error(yytable_value) \
  YYID (0)

static const yytype_int16 yycheck[] =
{
       1,     2,    85,   146,    67,    75,    64,    75,    87,    88,
     251,   177,   528,   512,    89,   472,   511,    71,    17,    77,
      17,    36,   452,   509,    91,    92,    93,    37,   523,   485,
      37,   526,   281,   489,    56,   284,   227,   229,   230,   231,
      75,   408,    74,   473,     0,    99,    81,   504,   505,    84,
     536,    30,    84,   166,   167,   168,   555,   514,   425,    81,
     173,   204,    63,   206,    65,    66,   141,   497,   317,    84,
      49,    81,   502,    83,    81,   266,   571,    40,   148,    76,
      78,   597,   577,   353,   583,    84,    84,    86,   142,   170,
     546,   548,    74,     1,   177,    96,    78,   367,   156,   556,
     158,   159,   183,   161,     3,     4,   601,   299,   300,    77,
      75,    79,   179,   180,   181,   182,     1,   633,    86,   311,
     312,   313,   314,   315,    74,   238,   621,   240,    78,    76,
     273,    81,   589,    41,    42,    43,    44,    45,    46,    47,
      53,    74,   308,   607,   574,    78,   641,   222,    81,    74,
     614,   615,   609,    36,   618,     6,    41,    42,    43,    44,
      45,    46,    47,    74,    72,    74,    81,    75,   169,    84,
      81,    56,    81,    79,    84,    81,   417,   320,    84,   322,
     323,    87,    88,   640,    60,    36,    37,    72,    64,    84,
      75,    76,    81,    82,    84,    84,    81,    55,    87,    88,
     201,   280,    53,   282,   283,    73,   285,    75,    59,    60,
      61,    62,    63,    64,    65,   290,    67,    68,    78,   277,
      59,    60,    61,    62,    51,   308,    53,     6,     7,     8,
       9,    10,   233,    73,    85,    75,   237,    73,     1,    18,
      19,    74,    73,    22,    75,   246,    80,   248,    53,    54,
      55,   252,   253,   254,   255,   256,   257,    73,    79,    75,
      81,    40,   405,    84,    38,    39,    87,    88,    80,   352,
      59,    60,   355,    76,   275,   276,   334,    36,    41,    42,
      43,    44,    45,    46,    47,    81,   369,    80,    84,    85,
      80,    87,    88,    56,    53,     1,    53,    54,    54,    55,
      59,    60,    61,    62,    63,    64,    65,    77,   383,    72,
      76,    66,    75,    76,    79,    84,    81,    84,    81,    84,
      53,    56,    87,    88,    76,   408,   292,   293,   294,    82,
     409,   410,   411,    73,    80,    41,    42,    43,    44,    45,
      46,    47,   425,    80,    85,   415,    85,   415,    77,   492,
      11,    12,    13,    14,    15,    16,   499,    78,   501,    74,
      85,    77,    85,    49,     1,   508,    72,    85,    77,    75,
      76,    75,    77,    85,    76,     1,    79,    77,   521,    77,
      79,    82,   348,   349,   527,    17,    76,    84,     1,    85,
      81,    86,   393,    86,    82,   361,   362,   363,   364,   365,
      85,   459,   545,   404,    41,    42,    43,    44,    45,    46,
      47,    82,    79,    82,   415,    41,    42,    43,    44,    45,
      46,    47,    79,    82,    77,   568,    81,   570,    41,    42,
      43,    44,    45,    46,    47,    72,    81,    81,    75,    76,
      40,   584,    76,    76,   507,    84,    72,    77,    73,    75,
     593,    82,   453,    55,    82,   598,    85,    77,    82,    72,
     543,    82,    75,    82,    81,     6,     7,     8,     9,    85,
      11,    12,    13,    14,    15,    85,    85,    73,    36,   480,
     623,   544,     1,    81,     3,     4,     5,    84,     7,     8,
       9,    53,    11,    12,    13,    14,    15,   580,   581,    40,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    34,    35,    36,    75,    38,
      39,    40,    41,    42,    43,    44,    45,    46,    47,    48,
      55,    50,    73,    52,    53,    76,    75,    20,    53,    77,
      59,    60,    61,    62,    63,    64,    65,    66,    75,    86,
      69,    70,    71,    72,    76,    84,    73,     1,    82,    85,
      76,    54,    76,   564,    85,    84,    76,    74,     1,    77,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    14,    15,    16,    85,    18,    19,    73,    48,    22,
     591,   592,    74,    78,    78,   596,    76,    41,    42,    43,
      44,    45,    46,    47,    76,    76,   607,    85,    77,    79,
      85,    76,    76,   614,   615,     1,   617,   618,    51,    52,
       5,    68,     1,    54,    57,    58,   278,   628,    72,     1,
     150,    75,   184,   297,   302,   171,    69,   288,   423,     1,
     383,   307,    75,    76,     6,     7,     8,     9,   308,    11,
      12,    13,    14,    15,    68,    41,    42,    43,    44,    45,
      46,    47,    41,    42,    43,    44,    45,    46,    47,    41,
      42,    43,    44,    45,    46,    47,    68,   388,    40,    41,
      42,    43,    44,    45,    46,    47,    72,   222,   415,    75,
     471,   514,   580,    72,   321,   389,    75,   581,   385,   536,
      72,    -1,   600,    75,   522,    -1,    -1,    -1,    -1,    -1,
      72,    73,    -1,    75,    76,     3,     4,    -1,     6,     7,
       8,     9,    10,    11,    12,    13,    14,    15,    16,     1,
      18,    19,    -1,    -1,    22,    41,    42,    43,    44,    45,
      46,    47,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    21,
      -1,    23,    24,    25,    26,    27,    28,    -1,    -1,    31,
      32,    33,    34,    35,    -1,    -1,    -1,     6,     7,     8,
       9,    -1,    11,    12,    13,    14,    15,    49,    50,    -1,
       1,    -1,    -1,    -1,    -1,    -1,     6,    75,    -1,    -1,
      -1,    -1,    -1,    -1,    66,    -1,    -1,    -1,    70,    71,
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
      32,    33,    34,    35,    16,    -1,    38,    39,    40,    41,
      42,    43,    44,    45,    46,    47,    48,    -1,    50,    -1,
      52,    53,    -1,    -1,    36,    37,    18,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    66,    -1,    -1,    69,    70,    71,
      72,    53,    -1,    -1,    36,    37,    -1,    59,    60,    61,
      62,    63,    64,    65,    -1,    -1,    -1,    36,    37,    -1,
      -1,    53,    -1,    -1,    -1,    -1,    -1,    59,    60,    61,
      62,    63,    64,    65,    53,    -1,    -1,    -1,    -1,    -1,
      59,    60,    61,    62,    63,    64,    65,    36,    37,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      36,    37,    -1,    82,    53,    54,    55,    -1,    -1,    -1,
      59,    60,    61,    62,    63,    64,    65,    53,    36,    -1,
      -1,    -1,    -1,    59,    60,    61,    62,    63,    64,    65,
      -1,    -1,    -1,    -1,    -1,    53,    -1,    -1,    -1,    -1,
      -1,    59,    60,    61,    62,    63,    64,    65
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
      81,    78,    81,    85,    78,    81,   158,    85,    77,   104,
      74,   135,   135,   135,   135,    94,    85,    77,    85,   106,
     106,    85,    77,    75,    94,    86,   148,    94,    77,    79,
      93,    94,    94,    94,    94,    94,    94,    77,    79,   104,
      76,    77,    82,    85,   171,    94,    94,    54,    95,   114,
     116,   119,   102,   119,   119,   102,   119,   124,   107,   136,
      73,    75,   153,   153,   153,   153,    85,   128,   135,   135,
     121,    17,   130,   132,   133,    86,   147,    54,    55,   148,
     150,   135,   135,   135,   135,   135,    73,    75,   102,    81,
     182,   171,   170,   171,   171,    82,    85,    82,    79,    82,
      95,    79,    82,    77,    40,   151,   154,   155,   160,   161,
     163,   153,   153,   113,   133,    76,   113,   153,   153,   153,
     153,   153,   133,    82,   113,   172,   175,   180,    81,    81,
      81,    81,   136,     1,    84,   166,   163,    76,   154,    73,
     162,    94,    76,    94,   171,    77,    82,   180,   119,   119,
     119,     1,    21,    23,    24,    25,    26,    27,    28,    31,
      32,    33,    34,    35,    49,    50,    66,    70,    71,   167,
     168,    53,    94,   165,    93,    84,   131,    73,    84,    86,
     130,    85,   180,    82,    82,    82,    82,    55,   125,    85,
      85,    77,   182,    94,    85,    73,    54,    55,    95,   169,
      36,   167,     1,    41,    42,    43,    44,    45,    46,    47,
      72,    75,   173,   185,   190,   192,   182,    94,    81,   195,
      84,   195,    53,   196,   197,    75,    55,   189,   195,    75,
     186,   192,   171,    20,   184,   182,   171,    53,   171,    84,
     182,   198,    77,    75,   192,   187,   192,   173,   171,    41,
      42,    43,    45,    46,    47,   188,   190,   191,    76,   186,
     174,    86,   185,    84,   183,    73,    85,    82,   194,   171,
     197,    76,   186,    76,   186,   171,   194,    76,   188,    54,
       6,    67,    68,    85,   113,   176,   179,   181,   173,   171,
     195,    75,   192,    85,   199,    76,   174,    75,   192,    94,
      74,    77,    85,   171,    73,   171,   186,   182,   186,    48,
     193,    78,   113,   172,   178,   181,   174,   171,    74,    76,
      76,    75,   192,    94,   177,    94,   171,    78,    94,   194,
     171,   193,   186,    79,    81,    84,    87,    88,    78,    85,
     177,    75,   192,    77,    76,   177,    54,   177,    79,    94,
     177,    79,   186,   171,    82,    85,    85,    94,    79,    76,
     194,    75,   192,   186,    76
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


/* This macro is provided for backward compatibility. */

#ifndef YY_LOCATION_PRINT
# define YY_LOCATION_PRINT(File, Loc) ((void) 0)
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
		  Type, Value); \
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
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_value_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (!yyvaluep)
    return;
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
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (yytype < YYNTOKENS)
    YYFPRINTF (yyoutput, "token %s (", yytname[yytype]);
  else
    YYFPRINTF (yyoutput, "nterm %s (", yytname[yytype]);

  yy_symbol_value_print (yyoutput, yytype, yyvaluep);
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
yy_reduce_print (YYSTYPE *yyvsp, int yyrule)
#else
static void
yy_reduce_print (yyvsp, yyrule)
    YYSTYPE *yyvsp;
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
		       		       );
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)		\
do {					\
  if (yydebug)				\
    yy_reduce_print (yyvsp, Rule); \
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
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep)
#else
static void
yydestruct (yymsg, yytype, yyvaluep)
    const char *yymsg;
    int yytype;
    YYSTYPE *yyvaluep;
#endif
{
  YYUSE (yyvaluep);

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

    YYSIZE_T yystacksize;

  int yyn;
  int yyresult;
  /* Lookahead token as an internal (translated) token number.  */
  int yytoken;
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

  yytoken = 0;
  yyss = yyssa;
  yyvs = yyvsa;
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


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 2:

/* Line 1806 of yacc.c  */
#line 156 "xi-grammar.y"
    { (yyval.modlist) = (yyvsp[(1) - (1)].modlist); modlist = (yyvsp[(1) - (1)].modlist); }
    break;

  case 3:

/* Line 1806 of yacc.c  */
#line 160 "xi-grammar.y"
    { 
		  (yyval.modlist) = 0; 
		}
    break;

  case 4:

/* Line 1806 of yacc.c  */
#line 164 "xi-grammar.y"
    { (yyval.modlist) = new AstChildren<Module>(lineno, (yyvsp[(1) - (2)].module), (yyvsp[(2) - (2)].modlist)); }
    break;

  case 5:

/* Line 1806 of yacc.c  */
#line 168 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 6:

/* Line 1806 of yacc.c  */
#line 170 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 7:

/* Line 1806 of yacc.c  */
#line 174 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 8:

/* Line 1806 of yacc.c  */
#line 176 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 9:

/* Line 1806 of yacc.c  */
#line 181 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 10:

/* Line 1806 of yacc.c  */
#line 182 "xi-grammar.y"
    { ReservedWord(MODULE); YYABORT; }
    break;

  case 11:

/* Line 1806 of yacc.c  */
#line 183 "xi-grammar.y"
    { ReservedWord(MAINMODULE); YYABORT; }
    break;

  case 12:

/* Line 1806 of yacc.c  */
#line 184 "xi-grammar.y"
    { ReservedWord(EXTERN); YYABORT; }
    break;

  case 13:

/* Line 1806 of yacc.c  */
#line 186 "xi-grammar.y"
    { ReservedWord(INITCALL); YYABORT; }
    break;

  case 14:

/* Line 1806 of yacc.c  */
#line 187 "xi-grammar.y"
    { ReservedWord(INITNODE); YYABORT; }
    break;

  case 15:

/* Line 1806 of yacc.c  */
#line 188 "xi-grammar.y"
    { ReservedWord(INITPROC); YYABORT; }
    break;

  case 16:

/* Line 1806 of yacc.c  */
#line 190 "xi-grammar.y"
    { ReservedWord(CHARE); }
    break;

  case 17:

/* Line 1806 of yacc.c  */
#line 191 "xi-grammar.y"
    { ReservedWord(MAINCHARE); }
    break;

  case 18:

/* Line 1806 of yacc.c  */
#line 192 "xi-grammar.y"
    { ReservedWord(GROUP); }
    break;

  case 19:

/* Line 1806 of yacc.c  */
#line 193 "xi-grammar.y"
    { ReservedWord(NODEGROUP); }
    break;

  case 20:

/* Line 1806 of yacc.c  */
#line 194 "xi-grammar.y"
    { ReservedWord(ARRAY); }
    break;

  case 21:

/* Line 1806 of yacc.c  */
#line 198 "xi-grammar.y"
    { ReservedWord(INCLUDE); YYABORT; }
    break;

  case 22:

/* Line 1806 of yacc.c  */
#line 199 "xi-grammar.y"
    { ReservedWord(STACKSIZE); YYABORT; }
    break;

  case 23:

/* Line 1806 of yacc.c  */
#line 200 "xi-grammar.y"
    { ReservedWord(THREADED); YYABORT; }
    break;

  case 24:

/* Line 1806 of yacc.c  */
#line 201 "xi-grammar.y"
    { ReservedWord(TEMPLATE); YYABORT; }
    break;

  case 25:

/* Line 1806 of yacc.c  */
#line 202 "xi-grammar.y"
    { ReservedWord(SYNC); YYABORT; }
    break;

  case 26:

/* Line 1806 of yacc.c  */
#line 203 "xi-grammar.y"
    { ReservedWord(IGET); YYABORT; }
    break;

  case 27:

/* Line 1806 of yacc.c  */
#line 204 "xi-grammar.y"
    { ReservedWord(EXCLUSIVE); YYABORT; }
    break;

  case 28:

/* Line 1806 of yacc.c  */
#line 205 "xi-grammar.y"
    { ReservedWord(IMMEDIATE); YYABORT; }
    break;

  case 29:

/* Line 1806 of yacc.c  */
#line 206 "xi-grammar.y"
    { ReservedWord(SKIPSCHED); YYABORT; }
    break;

  case 30:

/* Line 1806 of yacc.c  */
#line 207 "xi-grammar.y"
    { ReservedWord(INLINE); YYABORT; }
    break;

  case 31:

/* Line 1806 of yacc.c  */
#line 208 "xi-grammar.y"
    { ReservedWord(VIRTUAL); YYABORT; }
    break;

  case 32:

/* Line 1806 of yacc.c  */
#line 209 "xi-grammar.y"
    { ReservedWord(MIGRATABLE); YYABORT; }
    break;

  case 33:

/* Line 1806 of yacc.c  */
#line 210 "xi-grammar.y"
    { ReservedWord(CREATEHERE); YYABORT; }
    break;

  case 34:

/* Line 1806 of yacc.c  */
#line 211 "xi-grammar.y"
    { ReservedWord(CREATEHOME); YYABORT; }
    break;

  case 35:

/* Line 1806 of yacc.c  */
#line 212 "xi-grammar.y"
    { ReservedWord(NOKEEP); YYABORT; }
    break;

  case 36:

/* Line 1806 of yacc.c  */
#line 213 "xi-grammar.y"
    { ReservedWord(NOTRACE); YYABORT; }
    break;

  case 37:

/* Line 1806 of yacc.c  */
#line 214 "xi-grammar.y"
    { ReservedWord(APPWORK); YYABORT; }
    break;

  case 38:

/* Line 1806 of yacc.c  */
#line 217 "xi-grammar.y"
    { ReservedWord(PACKED); YYABORT; }
    break;

  case 39:

/* Line 1806 of yacc.c  */
#line 218 "xi-grammar.y"
    { ReservedWord(VARSIZE); YYABORT; }
    break;

  case 40:

/* Line 1806 of yacc.c  */
#line 219 "xi-grammar.y"
    { ReservedWord(ENTRY); YYABORT; }
    break;

  case 41:

/* Line 1806 of yacc.c  */
#line 220 "xi-grammar.y"
    { ReservedWord(FOR); YYABORT; }
    break;

  case 42:

/* Line 1806 of yacc.c  */
#line 221 "xi-grammar.y"
    { ReservedWord(FORALL); YYABORT; }
    break;

  case 43:

/* Line 1806 of yacc.c  */
#line 222 "xi-grammar.y"
    { ReservedWord(WHILE); YYABORT; }
    break;

  case 44:

/* Line 1806 of yacc.c  */
#line 223 "xi-grammar.y"
    { ReservedWord(WHEN); YYABORT; }
    break;

  case 45:

/* Line 1806 of yacc.c  */
#line 224 "xi-grammar.y"
    { ReservedWord(OVERLAP); YYABORT; }
    break;

  case 46:

/* Line 1806 of yacc.c  */
#line 225 "xi-grammar.y"
    { ReservedWord(ATOMIC); YYABORT; }
    break;

  case 47:

/* Line 1806 of yacc.c  */
#line 226 "xi-grammar.y"
    { ReservedWord(IF); YYABORT; }
    break;

  case 48:

/* Line 1806 of yacc.c  */
#line 227 "xi-grammar.y"
    { ReservedWord(ELSE); YYABORT; }
    break;

  case 49:

/* Line 1806 of yacc.c  */
#line 229 "xi-grammar.y"
    { ReservedWord(LOCAL); YYABORT; }
    break;

  case 50:

/* Line 1806 of yacc.c  */
#line 231 "xi-grammar.y"
    { ReservedWord(USING); YYABORT; }
    break;

  case 51:

/* Line 1806 of yacc.c  */
#line 232 "xi-grammar.y"
    { ReservedWord(ACCEL); YYABORT; }
    break;

  case 52:

/* Line 1806 of yacc.c  */
#line 235 "xi-grammar.y"
    { ReservedWord(ACCELBLOCK); YYABORT; }
    break;

  case 53:

/* Line 1806 of yacc.c  */
#line 236 "xi-grammar.y"
    { ReservedWord(MEMCRITICAL); YYABORT; }
    break;

  case 54:

/* Line 1806 of yacc.c  */
#line 237 "xi-grammar.y"
    { ReservedWord(REDUCTIONTARGET); YYABORT; }
    break;

  case 55:

/* Line 1806 of yacc.c  */
#line 238 "xi-grammar.y"
    { ReservedWord(CASE); YYABORT; }
    break;

  case 56:

/* Line 1806 of yacc.c  */
#line 242 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 57:

/* Line 1806 of yacc.c  */
#line 244 "xi-grammar.y"
    {
		  char *tmp = new char[strlen((yyvsp[(1) - (4)].strval))+strlen((yyvsp[(4) - (4)].strval))+3];
		  sprintf(tmp,"%s::%s", (yyvsp[(1) - (4)].strval), (yyvsp[(4) - (4)].strval));
		  (yyval.strval) = tmp;
		}
    break;

  case 58:

/* Line 1806 of yacc.c  */
#line 252 "xi-grammar.y"
    { 
		    (yyval.module) = new Module(lineno, (yyvsp[(2) - (3)].strval), (yyvsp[(3) - (3)].conslist)); 
		}
    break;

  case 59:

/* Line 1806 of yacc.c  */
#line 256 "xi-grammar.y"
    {  
		    (yyval.module) = new Module(lineno, (yyvsp[(2) - (3)].strval), (yyvsp[(3) - (3)].conslist)); 
		    (yyval.module)->setMain();
		}
    break;

  case 60:

/* Line 1806 of yacc.c  */
#line 263 "xi-grammar.y"
    { (yyval.conslist) = 0; }
    break;

  case 61:

/* Line 1806 of yacc.c  */
#line 265 "xi-grammar.y"
    { (yyval.conslist) = (yyvsp[(2) - (4)].conslist); }
    break;

  case 62:

/* Line 1806 of yacc.c  */
#line 269 "xi-grammar.y"
    { (yyval.conslist) = 0; }
    break;

  case 63:

/* Line 1806 of yacc.c  */
#line 271 "xi-grammar.y"
    { (yyval.conslist) = new ConstructList(lineno, (yyvsp[(1) - (2)].construct), (yyvsp[(2) - (2)].conslist)); }
    break;

  case 64:

/* Line 1806 of yacc.c  */
#line 275 "xi-grammar.y"
    { (yyval.construct) = new UsingScope((yyvsp[(3) - (3)].strval), false); }
    break;

  case 65:

/* Line 1806 of yacc.c  */
#line 277 "xi-grammar.y"
    { (yyval.construct) = new UsingScope((yyvsp[(2) - (2)].strval), true); }
    break;

  case 66:

/* Line 1806 of yacc.c  */
#line 279 "xi-grammar.y"
    { (yyvsp[(2) - (2)].member)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].member); }
    break;

  case 67:

/* Line 1806 of yacc.c  */
#line 281 "xi-grammar.y"
    { (yyvsp[(2) - (2)].message)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].message); }
    break;

  case 68:

/* Line 1806 of yacc.c  */
#line 283 "xi-grammar.y"
    {
                  Entry *e = new Entry(lineno, 0, (yyvsp[(3) - (7)].type), (yyvsp[(5) - (7)].strval), (yyvsp[(7) - (7)].plist), 0, 0, 0);
                  int isExtern = 1;
                  e->setExtern(isExtern);
                  e->targs = (yyvsp[(6) - (7)].tparlist);
                  e->label = new XStr;
                  (yyvsp[(4) - (7)].ntype)->print(*e->label);
                  (yyval.construct) = e;
                }
    break;

  case 69:

/* Line 1806 of yacc.c  */
#line 295 "xi-grammar.y"
    { if((yyvsp[(3) - (5)].conslist)) (yyvsp[(3) - (5)].conslist)->recurse<int&>((yyvsp[(1) - (5)].intval), &Construct::setExtern); (yyval.construct) = (yyvsp[(3) - (5)].conslist); }
    break;

  case 70:

/* Line 1806 of yacc.c  */
#line 297 "xi-grammar.y"
    { (yyval.construct) = new Scope((yyvsp[(2) - (5)].strval), (yyvsp[(4) - (5)].conslist)); }
    break;

  case 71:

/* Line 1806 of yacc.c  */
#line 299 "xi-grammar.y"
    { (yyval.construct) = (yyvsp[(1) - (2)].construct); }
    break;

  case 72:

/* Line 1806 of yacc.c  */
#line 301 "xi-grammar.y"
    { yyerror("The preceding construct must be semicolon terminated"); YYABORT; }
    break;

  case 73:

/* Line 1806 of yacc.c  */
#line 303 "xi-grammar.y"
    { (yyvsp[(2) - (2)].module)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].module); }
    break;

  case 74:

/* Line 1806 of yacc.c  */
#line 305 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 75:

/* Line 1806 of yacc.c  */
#line 307 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 76:

/* Line 1806 of yacc.c  */
#line 309 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 77:

/* Line 1806 of yacc.c  */
#line 311 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 78:

/* Line 1806 of yacc.c  */
#line 313 "xi-grammar.y"
    { (yyvsp[(2) - (2)].templat)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].templat); }
    break;

  case 79:

/* Line 1806 of yacc.c  */
#line 315 "xi-grammar.y"
    { (yyval.construct) = NULL; }
    break;

  case 80:

/* Line 1806 of yacc.c  */
#line 317 "xi-grammar.y"
    { (yyval.construct) = NULL; }
    break;

  case 81:

/* Line 1806 of yacc.c  */
#line 319 "xi-grammar.y"
    { (yyval.construct) = (yyvsp[(1) - (1)].accelBlock); }
    break;

  case 82:

/* Line 1806 of yacc.c  */
#line 321 "xi-grammar.y"
    { printf("Invalid construct\n"); YYABORT; }
    break;

  case 83:

/* Line 1806 of yacc.c  */
#line 325 "xi-grammar.y"
    { (yyval.tparam) = new TParamType((yyvsp[(1) - (1)].type)); }
    break;

  case 84:

/* Line 1806 of yacc.c  */
#line 327 "xi-grammar.y"
    { (yyval.tparam) = new TParamVal((yyvsp[(1) - (1)].strval)); }
    break;

  case 85:

/* Line 1806 of yacc.c  */
#line 329 "xi-grammar.y"
    { (yyval.tparam) = new TParamVal((yyvsp[(1) - (1)].strval)); }
    break;

  case 86:

/* Line 1806 of yacc.c  */
#line 333 "xi-grammar.y"
    { (yyval.tparlist) = new TParamList((yyvsp[(1) - (1)].tparam)); }
    break;

  case 87:

/* Line 1806 of yacc.c  */
#line 335 "xi-grammar.y"
    { (yyval.tparlist) = new TParamList((yyvsp[(1) - (3)].tparam), (yyvsp[(3) - (3)].tparlist)); }
    break;

  case 88:

/* Line 1806 of yacc.c  */
#line 339 "xi-grammar.y"
    { (yyval.tparlist) = 0; }
    break;

  case 89:

/* Line 1806 of yacc.c  */
#line 341 "xi-grammar.y"
    { (yyval.tparlist) = (yyvsp[(1) - (1)].tparlist); }
    break;

  case 90:

/* Line 1806 of yacc.c  */
#line 345 "xi-grammar.y"
    { (yyval.tparlist) = 0; }
    break;

  case 91:

/* Line 1806 of yacc.c  */
#line 347 "xi-grammar.y"
    { (yyval.tparlist) = (yyvsp[(2) - (3)].tparlist); }
    break;

  case 92:

/* Line 1806 of yacc.c  */
#line 351 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("int"); }
    break;

  case 93:

/* Line 1806 of yacc.c  */
#line 353 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long"); }
    break;

  case 94:

/* Line 1806 of yacc.c  */
#line 355 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("short"); }
    break;

  case 95:

/* Line 1806 of yacc.c  */
#line 357 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("char"); }
    break;

  case 96:

/* Line 1806 of yacc.c  */
#line 359 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned int"); }
    break;

  case 97:

/* Line 1806 of yacc.c  */
#line 361 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long"); }
    break;

  case 98:

/* Line 1806 of yacc.c  */
#line 363 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long"); }
    break;

  case 99:

/* Line 1806 of yacc.c  */
#line 365 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long long"); }
    break;

  case 100:

/* Line 1806 of yacc.c  */
#line 367 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned short"); }
    break;

  case 101:

/* Line 1806 of yacc.c  */
#line 369 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned char"); }
    break;

  case 102:

/* Line 1806 of yacc.c  */
#line 371 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long long"); }
    break;

  case 103:

/* Line 1806 of yacc.c  */
#line 373 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("float"); }
    break;

  case 104:

/* Line 1806 of yacc.c  */
#line 375 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("double"); }
    break;

  case 105:

/* Line 1806 of yacc.c  */
#line 377 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long double"); }
    break;

  case 106:

/* Line 1806 of yacc.c  */
#line 379 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("void"); }
    break;

  case 107:

/* Line 1806 of yacc.c  */
#line 382 "xi-grammar.y"
    { (yyval.ntype) = new NamedType((yyvsp[(1) - (2)].strval),(yyvsp[(2) - (2)].tparlist)); }
    break;

  case 108:

/* Line 1806 of yacc.c  */
#line 383 "xi-grammar.y"
    { 
                    const char* basename, *scope;
                    splitScopedName((yyvsp[(1) - (2)].strval), &scope, &basename);
                    (yyval.ntype) = new NamedType(basename, (yyvsp[(2) - (2)].tparlist), scope);
                }
    break;

  case 109:

/* Line 1806 of yacc.c  */
#line 391 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 110:

/* Line 1806 of yacc.c  */
#line 393 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ntype); }
    break;

  case 111:

/* Line 1806 of yacc.c  */
#line 397 "xi-grammar.y"
    { (yyval.ptype) = new PtrType((yyvsp[(1) - (2)].type)); }
    break;

  case 112:

/* Line 1806 of yacc.c  */
#line 401 "xi-grammar.y"
    { (yyvsp[(1) - (2)].ptype)->indirect(); (yyval.ptype) = (yyvsp[(1) - (2)].ptype); }
    break;

  case 113:

/* Line 1806 of yacc.c  */
#line 403 "xi-grammar.y"
    { (yyvsp[(1) - (2)].ptype)->indirect(); (yyval.ptype) = (yyvsp[(1) - (2)].ptype); }
    break;

  case 114:

/* Line 1806 of yacc.c  */
#line 407 "xi-grammar.y"
    { (yyval.ftype) = new FuncType((yyvsp[(1) - (8)].type), (yyvsp[(4) - (8)].strval), (yyvsp[(7) - (8)].plist)); }
    break;

  case 115:

/* Line 1806 of yacc.c  */
#line 411 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 116:

/* Line 1806 of yacc.c  */
#line 413 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 117:

/* Line 1806 of yacc.c  */
#line 415 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 118:

/* Line 1806 of yacc.c  */
#line 417 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ftype); }
    break;

  case 119:

/* Line 1806 of yacc.c  */
#line 420 "xi-grammar.y"
    { (yyval.type) = new ConstType((yyvsp[(2) - (2)].type)); }
    break;

  case 120:

/* Line 1806 of yacc.c  */
#line 422 "xi-grammar.y"
    { (yyval.type) = new ConstType((yyvsp[(1) - (2)].type)); }
    break;

  case 121:

/* Line 1806 of yacc.c  */
#line 426 "xi-grammar.y"
    { (yyval.type) = new ReferenceType((yyvsp[(1) - (2)].type)); }
    break;

  case 122:

/* Line 1806 of yacc.c  */
#line 428 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 123:

/* Line 1806 of yacc.c  */
#line 432 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 124:

/* Line 1806 of yacc.c  */
#line 434 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 125:

/* Line 1806 of yacc.c  */
#line 438 "xi-grammar.y"
    { (yyval.val) = (yyvsp[(2) - (3)].val); }
    break;

  case 126:

/* Line 1806 of yacc.c  */
#line 442 "xi-grammar.y"
    { (yyval.vallist) = 0; }
    break;

  case 127:

/* Line 1806 of yacc.c  */
#line 444 "xi-grammar.y"
    { (yyval.vallist) = new ValueList((yyvsp[(1) - (2)].val), (yyvsp[(2) - (2)].vallist)); }
    break;

  case 128:

/* Line 1806 of yacc.c  */
#line 448 "xi-grammar.y"
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[(2) - (4)].type), (yyvsp[(3) - (4)].strval), (yyvsp[(4) - (4)].vallist)); }
    break;

  case 129:

/* Line 1806 of yacc.c  */
#line 452 "xi-grammar.y"
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[(3) - (5)].type), (yyvsp[(5) - (5)].strval), 0, 1); }
    break;

  case 130:

/* Line 1806 of yacc.c  */
#line 456 "xi-grammar.y"
    { (yyval.intval) = 0;}
    break;

  case 131:

/* Line 1806 of yacc.c  */
#line 458 "xi-grammar.y"
    { (yyval.intval) = 0;}
    break;

  case 132:

/* Line 1806 of yacc.c  */
#line 462 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 133:

/* Line 1806 of yacc.c  */
#line 464 "xi-grammar.y"
    { 
		  /*
		  printf("Warning: Message attributes are being phased out.\n");
		  printf("Warning: Please remove them from interface files.\n");
		  */
		  (yyval.intval) = (yyvsp[(2) - (3)].intval); 
		}
    break;

  case 134:

/* Line 1806 of yacc.c  */
#line 474 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (1)].intval); }
    break;

  case 135:

/* Line 1806 of yacc.c  */
#line 476 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (3)].intval) | (yyvsp[(3) - (3)].intval); }
    break;

  case 136:

/* Line 1806 of yacc.c  */
#line 480 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 137:

/* Line 1806 of yacc.c  */
#line 482 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 138:

/* Line 1806 of yacc.c  */
#line 486 "xi-grammar.y"
    { (yyval.cattr) = 0; }
    break;

  case 139:

/* Line 1806 of yacc.c  */
#line 488 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(2) - (3)].cattr); }
    break;

  case 140:

/* Line 1806 of yacc.c  */
#line 492 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (1)].cattr); }
    break;

  case 141:

/* Line 1806 of yacc.c  */
#line 494 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (3)].cattr) | (yyvsp[(3) - (3)].cattr); }
    break;

  case 142:

/* Line 1806 of yacc.c  */
#line 498 "xi-grammar.y"
    { python_doc = NULL; (yyval.intval) = 0; }
    break;

  case 143:

/* Line 1806 of yacc.c  */
#line 500 "xi-grammar.y"
    { python_doc = (yyvsp[(1) - (1)].strval); (yyval.intval) = 0; }
    break;

  case 144:

/* Line 1806 of yacc.c  */
#line 504 "xi-grammar.y"
    { (yyval.cattr) = Chare::CPYTHON; }
    break;

  case 145:

/* Line 1806 of yacc.c  */
#line 508 "xi-grammar.y"
    { (yyval.cattr) = 0; }
    break;

  case 146:

/* Line 1806 of yacc.c  */
#line 510 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(2) - (3)].cattr); }
    break;

  case 147:

/* Line 1806 of yacc.c  */
#line 514 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (1)].cattr); }
    break;

  case 148:

/* Line 1806 of yacc.c  */
#line 516 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (3)].cattr) | (yyvsp[(3) - (3)].cattr); }
    break;

  case 149:

/* Line 1806 of yacc.c  */
#line 520 "xi-grammar.y"
    { (yyval.cattr) = Chare::CMIGRATABLE; }
    break;

  case 150:

/* Line 1806 of yacc.c  */
#line 522 "xi-grammar.y"
    { (yyval.cattr) = Chare::CPYTHON; }
    break;

  case 151:

/* Line 1806 of yacc.c  */
#line 526 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 152:

/* Line 1806 of yacc.c  */
#line 528 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 153:

/* Line 1806 of yacc.c  */
#line 531 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 154:

/* Line 1806 of yacc.c  */
#line 533 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 155:

/* Line 1806 of yacc.c  */
#line 536 "xi-grammar.y"
    { (yyval.mv) = new MsgVar((yyvsp[(2) - (5)].type), (yyvsp[(3) - (5)].strval), (yyvsp[(1) - (5)].intval), (yyvsp[(4) - (5)].intval)); }
    break;

  case 156:

/* Line 1806 of yacc.c  */
#line 540 "xi-grammar.y"
    { (yyval.mvlist) = new MsgVarList((yyvsp[(1) - (1)].mv)); }
    break;

  case 157:

/* Line 1806 of yacc.c  */
#line 542 "xi-grammar.y"
    { (yyval.mvlist) = new MsgVarList((yyvsp[(1) - (2)].mv), (yyvsp[(2) - (2)].mvlist)); }
    break;

  case 158:

/* Line 1806 of yacc.c  */
#line 546 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, (yyvsp[(3) - (3)].ntype)); }
    break;

  case 159:

/* Line 1806 of yacc.c  */
#line 548 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, (yyvsp[(3) - (6)].ntype), (yyvsp[(5) - (6)].mvlist)); }
    break;

  case 160:

/* Line 1806 of yacc.c  */
#line 552 "xi-grammar.y"
    { (yyval.typelist) = 0; }
    break;

  case 161:

/* Line 1806 of yacc.c  */
#line 554 "xi-grammar.y"
    { (yyval.typelist) = (yyvsp[(2) - (2)].typelist); }
    break;

  case 162:

/* Line 1806 of yacc.c  */
#line 558 "xi-grammar.y"
    { (yyval.typelist) = new TypeList((yyvsp[(1) - (1)].ntype)); }
    break;

  case 163:

/* Line 1806 of yacc.c  */
#line 560 "xi-grammar.y"
    { (yyval.typelist) = new TypeList((yyvsp[(1) - (3)].ntype), (yyvsp[(3) - (3)].typelist)); }
    break;

  case 164:

/* Line 1806 of yacc.c  */
#line 564 "xi-grammar.y"
    { (yyval.chare) = new Chare(lineno, (yyvsp[(2) - (5)].cattr)|Chare::CCHARE, (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 165:

/* Line 1806 of yacc.c  */
#line 566 "xi-grammar.y"
    { (yyval.chare) = new MainChare(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 166:

/* Line 1806 of yacc.c  */
#line 570 "xi-grammar.y"
    { (yyval.chare) = new Group(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 167:

/* Line 1806 of yacc.c  */
#line 574 "xi-grammar.y"
    { (yyval.chare) = new NodeGroup(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 168:

/* Line 1806 of yacc.c  */
#line 578 "xi-grammar.y"
    {/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",(yyvsp[(2) - (4)].strval));
			(yyval.ntype) = new NamedType(buf); 
		}
    break;

  case 169:

/* Line 1806 of yacc.c  */
#line 584 "xi-grammar.y"
    { (yyval.ntype) = new NamedType((yyvsp[(2) - (3)].strval)); }
    break;

  case 170:

/* Line 1806 of yacc.c  */
#line 588 "xi-grammar.y"
    {  (yyval.chare) = new Array(lineno, (yyvsp[(2) - (6)].cattr), (yyvsp[(3) - (6)].ntype), (yyvsp[(4) - (6)].ntype), (yyvsp[(5) - (6)].typelist), (yyvsp[(6) - (6)].mbrlist)); }
    break;

  case 171:

/* Line 1806 of yacc.c  */
#line 590 "xi-grammar.y"
    {  (yyval.chare) = new Array(lineno, (yyvsp[(3) - (6)].cattr), (yyvsp[(2) - (6)].ntype), (yyvsp[(4) - (6)].ntype), (yyvsp[(5) - (6)].typelist), (yyvsp[(6) - (6)].mbrlist)); }
    break;

  case 172:

/* Line 1806 of yacc.c  */
#line 594 "xi-grammar.y"
    { (yyval.chare) = new Chare(lineno, (yyvsp[(2) - (5)].cattr)|Chare::CCHARE, new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist));}
    break;

  case 173:

/* Line 1806 of yacc.c  */
#line 596 "xi-grammar.y"
    { (yyval.chare) = new MainChare(lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 174:

/* Line 1806 of yacc.c  */
#line 600 "xi-grammar.y"
    { (yyval.chare) = new Group(lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 175:

/* Line 1806 of yacc.c  */
#line 604 "xi-grammar.y"
    { (yyval.chare) = new NodeGroup( lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 176:

/* Line 1806 of yacc.c  */
#line 608 "xi-grammar.y"
    { (yyval.chare) = new Array( lineno, 0, (yyvsp[(2) - (5)].ntype), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 177:

/* Line 1806 of yacc.c  */
#line 612 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[(3) - (4)].strval))); }
    break;

  case 178:

/* Line 1806 of yacc.c  */
#line 614 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[(3) - (7)].strval)), (yyvsp[(5) - (7)].mvlist)); }
    break;

  case 179:

/* Line 1806 of yacc.c  */
#line 618 "xi-grammar.y"
    { (yyval.type) = 0; }
    break;

  case 180:

/* Line 1806 of yacc.c  */
#line 620 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(2) - (2)].type); }
    break;

  case 181:

/* Line 1806 of yacc.c  */
#line 624 "xi-grammar.y"
    { (yyval.strval) = 0; }
    break;

  case 182:

/* Line 1806 of yacc.c  */
#line 626 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(2) - (2)].strval); }
    break;

  case 183:

/* Line 1806 of yacc.c  */
#line 628 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(2) - (2)].strval); }
    break;

  case 184:

/* Line 1806 of yacc.c  */
#line 632 "xi-grammar.y"
    { (yyval.tvar) = new TType(new NamedType((yyvsp[(2) - (3)].strval)), (yyvsp[(3) - (3)].type)); }
    break;

  case 185:

/* Line 1806 of yacc.c  */
#line 634 "xi-grammar.y"
    { (yyval.tvar) = new TFunc((yyvsp[(1) - (2)].ftype), (yyvsp[(2) - (2)].strval)); }
    break;

  case 186:

/* Line 1806 of yacc.c  */
#line 636 "xi-grammar.y"
    { (yyval.tvar) = new TName((yyvsp[(1) - (3)].type), (yyvsp[(2) - (3)].strval), (yyvsp[(3) - (3)].strval)); }
    break;

  case 187:

/* Line 1806 of yacc.c  */
#line 640 "xi-grammar.y"
    { (yyval.tvarlist) = new TVarList((yyvsp[(1) - (1)].tvar)); }
    break;

  case 188:

/* Line 1806 of yacc.c  */
#line 642 "xi-grammar.y"
    { (yyval.tvarlist) = new TVarList((yyvsp[(1) - (3)].tvar), (yyvsp[(3) - (3)].tvarlist)); }
    break;

  case 189:

/* Line 1806 of yacc.c  */
#line 646 "xi-grammar.y"
    { (yyval.tvarlist) = (yyvsp[(3) - (4)].tvarlist); }
    break;

  case 190:

/* Line 1806 of yacc.c  */
#line 650 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 191:

/* Line 1806 of yacc.c  */
#line 652 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 192:

/* Line 1806 of yacc.c  */
#line 654 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 193:

/* Line 1806 of yacc.c  */
#line 656 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 194:

/* Line 1806 of yacc.c  */
#line 658 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].message)); (yyvsp[(2) - (2)].message)->setTemplate((yyval.templat)); }
    break;

  case 195:

/* Line 1806 of yacc.c  */
#line 662 "xi-grammar.y"
    { (yyval.mbrlist) = 0; }
    break;

  case 196:

/* Line 1806 of yacc.c  */
#line 664 "xi-grammar.y"
    { (yyval.mbrlist) = (yyvsp[(2) - (4)].mbrlist); }
    break;

  case 197:

/* Line 1806 of yacc.c  */
#line 668 "xi-grammar.y"
    { 
                  if (!connectEntries.empty()) {
                    (yyval.mbrlist) = new AstChildren<Member>(connectEntries);
		  } else {
		    (yyval.mbrlist) = 0; 
                  }
		}
    break;

  case 198:

/* Line 1806 of yacc.c  */
#line 676 "xi-grammar.y"
    { (yyval.mbrlist) = new AstChildren<Member>(-1, (yyvsp[(1) - (2)].member), (yyvsp[(2) - (2)].mbrlist)); }
    break;

  case 199:

/* Line 1806 of yacc.c  */
#line 680 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].readonly); }
    break;

  case 200:

/* Line 1806 of yacc.c  */
#line 682 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].readonly); }
    break;

  case 202:

/* Line 1806 of yacc.c  */
#line 685 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].member); }
    break;

  case 203:

/* Line 1806 of yacc.c  */
#line 687 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(2) - (2)].pupable); }
    break;

  case 204:

/* Line 1806 of yacc.c  */
#line 689 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(2) - (2)].includeFile); }
    break;

  case 205:

/* Line 1806 of yacc.c  */
#line 691 "xi-grammar.y"
    { (yyval.member) = new ClassDeclaration(lineno,(yyvsp[(2) - (2)].strval)); }
    break;

  case 206:

/* Line 1806 of yacc.c  */
#line 695 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 1); }
    break;

  case 207:

/* Line 1806 of yacc.c  */
#line 697 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 1); }
    break;

  case 208:

/* Line 1806 of yacc.c  */
#line 699 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[(3) - (9)].strval)) + '<' +
					    ((yyvsp[(5) - (9)].tparlist))->to_string() + '>').c_str()),
				    1);
		}
    break;

  case 209:

/* Line 1806 of yacc.c  */
#line 705 "xi-grammar.y"
    { printf("Warning: deprecated use of initcall. Use initnode or initproc instead.\n"); 
		  (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 1); }
    break;

  case 210:

/* Line 1806 of yacc.c  */
#line 708 "xi-grammar.y"
    { printf("Warning: deprecated use of initcall. Use initnode or initproc instead.\n");
		  (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 1); }
    break;

  case 211:

/* Line 1806 of yacc.c  */
#line 713 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 0); }
    break;

  case 212:

/* Line 1806 of yacc.c  */
#line 715 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 0); }
    break;

  case 213:

/* Line 1806 of yacc.c  */
#line 717 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[(3) - (9)].strval)) + '<' +
					    ((yyvsp[(5) - (9)].tparlist))->to_string() + '>').c_str()),
				    0);
		}
    break;

  case 214:

/* Line 1806 of yacc.c  */
#line 723 "xi-grammar.y"
    {
                  InitCall* rtn = new InitCall(lineno, (yyvsp[(6) - (9)].strval), 0);
                  rtn->setAccel();
                  (yyval.member) = rtn;
		}
    break;

  case 215:

/* Line 1806 of yacc.c  */
#line 731 "xi-grammar.y"
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[(1) - (1)].ntype),0); }
    break;

  case 216:

/* Line 1806 of yacc.c  */
#line 733 "xi-grammar.y"
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[(1) - (3)].ntype),(yyvsp[(3) - (3)].pupable)); }
    break;

  case 217:

/* Line 1806 of yacc.c  */
#line 736 "xi-grammar.y"
    { (yyval.includeFile) = new IncludeFile(lineno,(yyvsp[(1) - (1)].strval)); }
    break;

  case 218:

/* Line 1806 of yacc.c  */
#line 740 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (2)].member); }
    break;

  case 219:

/* Line 1806 of yacc.c  */
#line 743 "xi-grammar.y"
    { yyerror("The preceding entry method declaration must be semicolon-terminated."); YYABORT; }
    break;

  case 220:

/* Line 1806 of yacc.c  */
#line 747 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].entry); }
    break;

  case 221:

/* Line 1806 of yacc.c  */
#line 749 "xi-grammar.y"
    {
                  (yyvsp[(2) - (2)].entry)->tspec = (yyvsp[(1) - (2)].tvarlist);
                  (yyval.member) = (yyvsp[(2) - (2)].entry);
                }
    break;

  case 222:

/* Line 1806 of yacc.c  */
#line 754 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].member); }
    break;

  case 223:

/* Line 1806 of yacc.c  */
#line 758 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 224:

/* Line 1806 of yacc.c  */
#line 760 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 225:

/* Line 1806 of yacc.c  */
#line 762 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 226:

/* Line 1806 of yacc.c  */
#line 764 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 227:

/* Line 1806 of yacc.c  */
#line 766 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 228:

/* Line 1806 of yacc.c  */
#line 768 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 229:

/* Line 1806 of yacc.c  */
#line 770 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 230:

/* Line 1806 of yacc.c  */
#line 772 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 231:

/* Line 1806 of yacc.c  */
#line 774 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 232:

/* Line 1806 of yacc.c  */
#line 776 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 233:

/* Line 1806 of yacc.c  */
#line 778 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 234:

/* Line 1806 of yacc.c  */
#line 781 "xi-grammar.y"
    { 
                  (yyval.entry) = new Entry(lineno, (yyvsp[(2) - (7)].intval), (yyvsp[(3) - (7)].type), (yyvsp[(4) - (7)].strval), (yyvsp[(5) - (7)].plist), (yyvsp[(6) - (7)].val), (yyvsp[(7) - (7)].sc)); 
		  if ((yyvsp[(7) - (7)].sc) != 0) { 
		    (yyvsp[(7) - (7)].sc)->con1 = new SdagConstruct(SIDENT, (yyvsp[(4) - (7)].strval));
                    (yyvsp[(7) - (7)].sc)->entry = (yyval.entry);
                    (yyvsp[(7) - (7)].sc)->con1->entry = (yyval.entry);
                    (yyvsp[(7) - (7)].sc)->param = new ParamList((yyvsp[(5) - (7)].plist));
                  }
		}
    break;

  case 235:

/* Line 1806 of yacc.c  */
#line 791 "xi-grammar.y"
    { 
                  Entry *e = new Entry(lineno, (yyvsp[(2) - (5)].intval), 0, (yyvsp[(3) - (5)].strval), (yyvsp[(4) - (5)].plist),  0, (yyvsp[(5) - (5)].sc));
                  if ((yyvsp[(5) - (5)].sc) != 0) {
		    (yyvsp[(5) - (5)].sc)->con1 = new SdagConstruct(SIDENT, (yyvsp[(3) - (5)].strval));
                    (yyvsp[(5) - (5)].sc)->entry = e;
                    (yyvsp[(5) - (5)].sc)->con1->entry = e;
                    (yyvsp[(5) - (5)].sc)->param = new ParamList((yyvsp[(4) - (5)].plist));
                  }
		  if (e->param && e->param->isCkMigMsgPtr()) {
		    yyerror("Charm++ takes a CkMigrateMsg chare constructor for granted, but continuing anyway");
		    (yyval.entry) = NULL;
		  } else
		    (yyval.entry) = e;
		}
    break;

  case 236:

/* Line 1806 of yacc.c  */
#line 806 "xi-grammar.y"
    {
                  int attribs = SACCEL;
                  const char* name = (yyvsp[(6) - (12)].strval);
                  ParamList* paramList = (yyvsp[(7) - (12)].plist);
                  ParamList* accelParamList = (yyvsp[(8) - (12)].plist);
		  XStr* codeBody = new XStr((yyvsp[(10) - (12)].strval));
                  const char* callbackName = (yyvsp[(12) - (12)].strval);

                  (yyval.entry) = new Entry(lineno, attribs, new BuiltinType("void"), name, paramList, 0, 0, 0 );
                  (yyval.entry)->setAccelParam(accelParamList);
                  (yyval.entry)->setAccelCodeBody(codeBody);
                  (yyval.entry)->setAccelCallbackName(new XStr(callbackName));
                }
    break;

  case 237:

/* Line 1806 of yacc.c  */
#line 822 "xi-grammar.y"
    { (yyval.accelBlock) = new AccelBlock(lineno, new XStr((yyvsp[(3) - (5)].strval))); }
    break;

  case 238:

/* Line 1806 of yacc.c  */
#line 824 "xi-grammar.y"
    { (yyval.accelBlock) = new AccelBlock(lineno, NULL); }
    break;

  case 239:

/* Line 1806 of yacc.c  */
#line 828 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("void"); }
    break;

  case 240:

/* Line 1806 of yacc.c  */
#line 830 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 241:

/* Line 1806 of yacc.c  */
#line 834 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 242:

/* Line 1806 of yacc.c  */
#line 836 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(2) - (3)].intval); }
    break;

  case 243:

/* Line 1806 of yacc.c  */
#line 838 "xi-grammar.y"
    { printf("Invalid entry method attribute list\n"); YYABORT; }
    break;

  case 244:

/* Line 1806 of yacc.c  */
#line 842 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (1)].intval); }
    break;

  case 245:

/* Line 1806 of yacc.c  */
#line 844 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (3)].intval) | (yyvsp[(3) - (3)].intval); }
    break;

  case 246:

/* Line 1806 of yacc.c  */
#line 848 "xi-grammar.y"
    { (yyval.intval) = STHREADED; }
    break;

  case 247:

/* Line 1806 of yacc.c  */
#line 850 "xi-grammar.y"
    { (yyval.intval) = SSYNC; }
    break;

  case 248:

/* Line 1806 of yacc.c  */
#line 852 "xi-grammar.y"
    { (yyval.intval) = SIGET; }
    break;

  case 249:

/* Line 1806 of yacc.c  */
#line 854 "xi-grammar.y"
    { (yyval.intval) = SLOCKED; }
    break;

  case 250:

/* Line 1806 of yacc.c  */
#line 856 "xi-grammar.y"
    { (yyval.intval) = SCREATEHERE; }
    break;

  case 251:

/* Line 1806 of yacc.c  */
#line 858 "xi-grammar.y"
    { (yyval.intval) = SCREATEHOME; }
    break;

  case 252:

/* Line 1806 of yacc.c  */
#line 860 "xi-grammar.y"
    { (yyval.intval) = SNOKEEP; }
    break;

  case 253:

/* Line 1806 of yacc.c  */
#line 862 "xi-grammar.y"
    { (yyval.intval) = SNOTRACE; }
    break;

  case 254:

/* Line 1806 of yacc.c  */
#line 864 "xi-grammar.y"
    { (yyval.intval) = SAPPWORK; }
    break;

  case 255:

/* Line 1806 of yacc.c  */
#line 866 "xi-grammar.y"
    { (yyval.intval) = SIMMEDIATE; }
    break;

  case 256:

/* Line 1806 of yacc.c  */
#line 868 "xi-grammar.y"
    { (yyval.intval) = SSKIPSCHED; }
    break;

  case 257:

/* Line 1806 of yacc.c  */
#line 870 "xi-grammar.y"
    { (yyval.intval) = SINLINE; }
    break;

  case 258:

/* Line 1806 of yacc.c  */
#line 872 "xi-grammar.y"
    { (yyval.intval) = SLOCAL; }
    break;

  case 259:

/* Line 1806 of yacc.c  */
#line 874 "xi-grammar.y"
    { (yyval.intval) = SPYTHON; }
    break;

  case 260:

/* Line 1806 of yacc.c  */
#line 876 "xi-grammar.y"
    { (yyval.intval) = SMEM; }
    break;

  case 261:

/* Line 1806 of yacc.c  */
#line 878 "xi-grammar.y"
    { (yyval.intval) = SREDUCE; }
    break;

  case 262:

/* Line 1806 of yacc.c  */
#line 880 "xi-grammar.y"
    { printf("Invalid entry method attribute: %s\n", yylval); YYABORT; }
    break;

  case 263:

/* Line 1806 of yacc.c  */
#line 884 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 264:

/* Line 1806 of yacc.c  */
#line 886 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 265:

/* Line 1806 of yacc.c  */
#line 888 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 266:

/* Line 1806 of yacc.c  */
#line 892 "xi-grammar.y"
    { (yyval.strval) = ""; }
    break;

  case 267:

/* Line 1806 of yacc.c  */
#line 894 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 268:

/* Line 1806 of yacc.c  */
#line 896 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[(1) - (3)].strval))+strlen((yyvsp[(3) - (3)].strval))+3];
			sprintf(tmp,"%s, %s", (yyvsp[(1) - (3)].strval), (yyvsp[(3) - (3)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 269:

/* Line 1806 of yacc.c  */
#line 904 "xi-grammar.y"
    { (yyval.strval) = ""; }
    break;

  case 270:

/* Line 1806 of yacc.c  */
#line 906 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 271:

/* Line 1806 of yacc.c  */
#line 908 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s[%s]%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 272:

/* Line 1806 of yacc.c  */
#line 914 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s{%s}%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 273:

/* Line 1806 of yacc.c  */
#line 920 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s(%s)%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 274:

/* Line 1806 of yacc.c  */
#line 926 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(2) - (4)].strval))+strlen((yyvsp[(4) - (4)].strval))+3];
			sprintf(tmp,"(%s)%s", (yyvsp[(2) - (4)].strval), (yyvsp[(4) - (4)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 275:

/* Line 1806 of yacc.c  */
#line 934 "xi-grammar.y"
    {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			(yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].type),(yyvsp[(2) - (3)].strval));
		}
    break;

  case 276:

/* Line 1806 of yacc.c  */
#line 941 "xi-grammar.y"
    { 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			(yyval.intval) = 0;
		}
    break;

  case 277:

/* Line 1806 of yacc.c  */
#line 949 "xi-grammar.y"
    { 
			in_braces=0;
			(yyval.intval) = 0;
		}
    break;

  case 278:

/* Line 1806 of yacc.c  */
#line 956 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (1)].type));}
    break;

  case 279:

/* Line 1806 of yacc.c  */
#line 958 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].type),(yyvsp[(2) - (3)].strval)); (yyval.pname)->setConditional((yyvsp[(3) - (3)].intval)); }
    break;

  case 280:

/* Line 1806 of yacc.c  */
#line 960 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (4)].type),(yyvsp[(2) - (4)].strval),0,(yyvsp[(4) - (4)].val));}
    break;

  case 281:

/* Line 1806 of yacc.c  */
#line 962 "xi-grammar.y"
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].pname)->getType(), (yyvsp[(1) - (3)].pname)->getName() ,(yyvsp[(2) - (3)].strval));
		}
    break;

  case 282:

/* Line 1806 of yacc.c  */
#line 968 "xi-grammar.y"
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READONLY; }
    break;

  case 283:

/* Line 1806 of yacc.c  */
#line 969 "xi-grammar.y"
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READWRITE; }
    break;

  case 284:

/* Line 1806 of yacc.c  */
#line 970 "xi-grammar.y"
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_WRITEONLY; }
    break;

  case 285:

/* Line 1806 of yacc.c  */
#line 973 "xi-grammar.y"
    { (yyval.xstrptr) = new XStr((yyvsp[(1) - (1)].strval)); }
    break;

  case 286:

/* Line 1806 of yacc.c  */
#line 974 "xi-grammar.y"
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "->" << (yyvsp[(4) - (4)].strval); }
    break;

  case 287:

/* Line 1806 of yacc.c  */
#line 975 "xi-grammar.y"
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[(1) - (3)].xstrptr)) << "." << (yyvsp[(3) - (3)].strval); }
    break;

  case 288:

/* Line 1806 of yacc.c  */
#line 977 "xi-grammar.y"
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "[" << *((yyvsp[(3) - (4)].xstrptr)) << "]";
                  delete (yyvsp[(1) - (4)].xstrptr);
                  delete (yyvsp[(3) - (4)].xstrptr);
                }
    break;

  case 289:

/* Line 1806 of yacc.c  */
#line 984 "xi-grammar.y"
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "[" << (yyvsp[(3) - (4)].strval) << "]";
                  delete (yyvsp[(1) - (4)].xstrptr);
                }
    break;

  case 290:

/* Line 1806 of yacc.c  */
#line 990 "xi-grammar.y"
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "(" << *((yyvsp[(3) - (4)].xstrptr)) << ")";
                  delete (yyvsp[(1) - (4)].xstrptr);
                  delete (yyvsp[(3) - (4)].xstrptr);
                }
    break;

  case 291:

/* Line 1806 of yacc.c  */
#line 999 "xi-grammar.y"
    {
                  in_bracket = 0;
                  (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].pname)->getType(), (yyvsp[(1) - (3)].pname)->getName(), (yyvsp[(2) - (3)].strval));
                }
    break;

  case 292:

/* Line 1806 of yacc.c  */
#line 1006 "xi-grammar.y"
    {
                  (yyval.pname) = new Parameter(lineno, (yyvsp[(3) - (7)].type), (yyvsp[(4) - (7)].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[(6) - (7)].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[(1) - (7)].intval));
                }
    break;

  case 293:

/* Line 1806 of yacc.c  */
#line 1012 "xi-grammar.y"
    {
		  (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (5)].type), (yyvsp[(2) - (5)].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[(4) - (5)].xstrptr));
                  (yyval.pname)->setAccelBufferType(Parameter::ACCEL_BUFFER_TYPE_READWRITE);
		}
    break;

  case 294:

/* Line 1806 of yacc.c  */
#line 1018 "xi-grammar.y"
    {
                  (yyval.pname) = (yyvsp[(3) - (6)].pname);
                  (yyval.pname)->setAccelInstName((yyvsp[(5) - (6)].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[(1) - (6)].intval));
		}
    break;

  case 295:

/* Line 1806 of yacc.c  */
#line 1026 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (1)].pname)); }
    break;

  case 296:

/* Line 1806 of yacc.c  */
#line 1028 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (3)].pname),(yyvsp[(3) - (3)].plist)); }
    break;

  case 297:

/* Line 1806 of yacc.c  */
#line 1032 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (1)].pname)); }
    break;

  case 298:

/* Line 1806 of yacc.c  */
#line 1034 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (3)].pname),(yyvsp[(3) - (3)].plist)); }
    break;

  case 299:

/* Line 1806 of yacc.c  */
#line 1038 "xi-grammar.y"
    { (yyval.plist) = (yyvsp[(2) - (3)].plist); }
    break;

  case 300:

/* Line 1806 of yacc.c  */
#line 1040 "xi-grammar.y"
    { (yyval.plist) = new ParamList(new Parameter(0, new BuiltinType("void"))); }
    break;

  case 301:

/* Line 1806 of yacc.c  */
#line 1044 "xi-grammar.y"
    { (yyval.plist) = (yyvsp[(2) - (3)].plist); }
    break;

  case 302:

/* Line 1806 of yacc.c  */
#line 1046 "xi-grammar.y"
    { (yyval.plist) = 0; }
    break;

  case 303:

/* Line 1806 of yacc.c  */
#line 1050 "xi-grammar.y"
    { (yyval.val) = 0; }
    break;

  case 304:

/* Line 1806 of yacc.c  */
#line 1052 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(3) - (3)].strval)); }
    break;

  case 305:

/* Line 1806 of yacc.c  */
#line 1056 "xi-grammar.y"
    { (yyval.sc) = 0; }
    break;

  case 306:

/* Line 1806 of yacc.c  */
#line 1058 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSDAGENTRY, (yyvsp[(1) - (1)].sc)); }
    break;

  case 307:

/* Line 1806 of yacc.c  */
#line 1060 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSDAGENTRY, (yyvsp[(2) - (3)].sc)); }
    break;

  case 308:

/* Line 1806 of yacc.c  */
#line 1064 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSLIST, (yyvsp[(1) - (1)].sc)); }
    break;

  case 309:

/* Line 1806 of yacc.c  */
#line 1066 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSLIST, (yyvsp[(1) - (2)].sc), (yyvsp[(2) - (2)].sc));  }
    break;

  case 310:

/* Line 1806 of yacc.c  */
#line 1070 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SOLIST, (yyvsp[(1) - (1)].sc)); }
    break;

  case 311:

/* Line 1806 of yacc.c  */
#line 1072 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SOLIST, (yyvsp[(1) - (2)].sc), (yyvsp[(2) - (2)].sc)); }
    break;

  case 312:

/* Line 1806 of yacc.c  */
#line 1076 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SCASELIST, (yyvsp[(1) - (1)].when)); }
    break;

  case 313:

/* Line 1806 of yacc.c  */
#line 1078 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SCASELIST, (yyvsp[(1) - (2)].when), (yyvsp[(2) - (2)].sc)); }
    break;

  case 314:

/* Line 1806 of yacc.c  */
#line 1080 "xi-grammar.y"
    { yyerror("Case blocks in SDAG can only contain when clauses."); YYABORT; }
    break;

  case 315:

/* Line 1806 of yacc.c  */
#line 1084 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 316:

/* Line 1806 of yacc.c  */
#line 1086 "xi-grammar.y"
    { (yyval.strval) = 0; }
    break;

  case 317:

/* Line 1806 of yacc.c  */
#line 1090 "xi-grammar.y"
    { (yyval.when) = new WhenConstruct((yyvsp[(2) - (4)].entrylist), 0); }
    break;

  case 318:

/* Line 1806 of yacc.c  */
#line 1092 "xi-grammar.y"
    { (yyval.when) = new WhenConstruct((yyvsp[(2) - (3)].entrylist), (yyvsp[(3) - (3)].sc)); }
    break;

  case 319:

/* Line 1806 of yacc.c  */
#line 1094 "xi-grammar.y"
    { (yyval.when) = new WhenConstruct((yyvsp[(2) - (5)].entrylist), (yyvsp[(4) - (5)].sc)); }
    break;

  case 320:

/* Line 1806 of yacc.c  */
#line 1098 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 321:

/* Line 1806 of yacc.c  */
#line 1100 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 322:

/* Line 1806 of yacc.c  */
#line 1102 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 323:

/* Line 1806 of yacc.c  */
#line 1104 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 324:

/* Line 1806 of yacc.c  */
#line 1106 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 325:

/* Line 1806 of yacc.c  */
#line 1108 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 326:

/* Line 1806 of yacc.c  */
#line 1112 "xi-grammar.y"
    { (yyval.sc) = new AtomicConstruct((yyvsp[(4) - (5)].strval), (yyvsp[(2) - (5)].strval)); }
    break;

  case 327:

/* Line 1806 of yacc.c  */
#line 1114 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SOVERLAP,0, 0,0,0,0,(yyvsp[(3) - (4)].sc), 0); }
    break;

  case 328:

/* Line 1806 of yacc.c  */
#line 1116 "xi-grammar.y"
    { (yyval.sc) = (yyvsp[(1) - (1)].when); }
    break;

  case 329:

/* Line 1806 of yacc.c  */
#line 1118 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SCASE, 0, 0, 0, 0, 0, (yyvsp[(3) - (4)].sc), 0); }
    break;

  case 330:

/* Line 1806 of yacc.c  */
#line 1120 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFOR, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (11)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(5) - (11)].strval)),
		             new SdagConstruct(SINT_EXPR, (yyvsp[(7) - (11)].strval)), 0, (yyvsp[(10) - (11)].sc), 0); }
    break;

  case 331:

/* Line 1806 of yacc.c  */
#line 1123 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFOR, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (9)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(5) - (9)].strval)), 
		         new SdagConstruct(SINT_EXPR, (yyvsp[(7) - (9)].strval)), 0, (yyvsp[(9) - (9)].sc), 0); }
    break;

  case 332:

/* Line 1806 of yacc.c  */
#line 1126 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFORALL, 0, new SdagConstruct(SIDENT, (yyvsp[(3) - (12)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(6) - (12)].strval)), 
		             new SdagConstruct(SINT_EXPR, (yyvsp[(8) - (12)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(10) - (12)].strval)), (yyvsp[(12) - (12)].sc), 0); }
    break;

  case 333:

/* Line 1806 of yacc.c  */
#line 1129 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFORALL, 0, new SdagConstruct(SIDENT, (yyvsp[(3) - (14)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(6) - (14)].strval)), 
		                 new SdagConstruct(SINT_EXPR, (yyvsp[(8) - (14)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(10) - (14)].strval)), (yyvsp[(13) - (14)].sc), 0); }
    break;

  case 334:

/* Line 1806 of yacc.c  */
#line 1132 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (6)].strval)), (yyvsp[(6) - (6)].sc),0,0,(yyvsp[(5) - (6)].sc),0); }
    break;

  case 335:

/* Line 1806 of yacc.c  */
#line 1134 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (8)].strval)), (yyvsp[(8) - (8)].sc),0,0,(yyvsp[(6) - (8)].sc),0); }
    break;

  case 336:

/* Line 1806 of yacc.c  */
#line 1136 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SWHILE, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (5)].strval)), 0,0,0,(yyvsp[(5) - (5)].sc),0); }
    break;

  case 337:

/* Line 1806 of yacc.c  */
#line 1138 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SWHILE, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (7)].strval)), 0,0,0,(yyvsp[(6) - (7)].sc),0); }
    break;

  case 338:

/* Line 1806 of yacc.c  */
#line 1140 "xi-grammar.y"
    { (yyval.sc) = new AtomicConstruct((yyvsp[(2) - (3)].strval), NULL); }
    break;

  case 339:

/* Line 1806 of yacc.c  */
#line 1142 "xi-grammar.y"
    { printf("Unknown SDAG construct or malformed entry method definition.\n"
                         "You may have forgotten to terminate an entry method definition with a"
                         " semicolon or forgotten to mark a block of sequential SDAG code as 'atomic'\n"); YYABORT; }
    break;

  case 340:

/* Line 1806 of yacc.c  */
#line 1148 "xi-grammar.y"
    { (yyval.sc) = 0; }
    break;

  case 341:

/* Line 1806 of yacc.c  */
#line 1150 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SELSE, 0,0,0,0,0, (yyvsp[(2) - (2)].sc),0); }
    break;

  case 342:

/* Line 1806 of yacc.c  */
#line 1152 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SELSE, 0,0,0,0,0, (yyvsp[(3) - (4)].sc),0); }
    break;

  case 343:

/* Line 1806 of yacc.c  */
#line 1156 "xi-grammar.y"
    { in_int_expr = 0; (yyval.intval) = 0; }
    break;

  case 344:

/* Line 1806 of yacc.c  */
#line 1160 "xi-grammar.y"
    { in_int_expr = 1; (yyval.intval) = 0; }
    break;

  case 345:

/* Line 1806 of yacc.c  */
#line 1164 "xi-grammar.y"
    { (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[(1) - (2)].strval), (yyvsp[(2) - (2)].plist), 0, 0, 0); }
    break;

  case 346:

/* Line 1806 of yacc.c  */
#line 1166 "xi-grammar.y"
    { (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[(1) - (5)].strval), (yyvsp[(5) - (5)].plist), 0, 0, (yyvsp[(3) - (5)].strval)); }
    break;

  case 347:

/* Line 1806 of yacc.c  */
#line 1170 "xi-grammar.y"
    { (yyval.entrylist) = new EntryList((yyvsp[(1) - (1)].entry)); }
    break;

  case 348:

/* Line 1806 of yacc.c  */
#line 1172 "xi-grammar.y"
    { (yyval.entrylist) = new EntryList((yyvsp[(1) - (3)].entry),(yyvsp[(3) - (3)].entrylist)); }
    break;

  case 349:

/* Line 1806 of yacc.c  */
#line 1176 "xi-grammar.y"
    { in_bracket=1; }
    break;

  case 350:

/* Line 1806 of yacc.c  */
#line 1179 "xi-grammar.y"
    { in_bracket=0; }
    break;

  case 351:

/* Line 1806 of yacc.c  */
#line 1183 "xi-grammar.y"
    { if (!macroDefined((yyvsp[(2) - (2)].strval), 1)) in_comment = 1; }
    break;

  case 352:

/* Line 1806 of yacc.c  */
#line 1187 "xi-grammar.y"
    { if (!macroDefined((yyvsp[(2) - (2)].strval), 0)) in_comment = 1; }
    break;



/* Line 1806 of yacc.c  */
#line 4919 "y.tab.c"
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


      yydestruct ("Error: popping",
		  yystos[yystate], yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  *++yyvsp = yylval;


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
                  yytoken, &yylval);
    }
  /* Do not reclaim the symbols of the rule which action triggered
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
  /* Make sure YYID is used.  */
  return YYID (yyresult);
}



/* Line 2067 of yacc.c  */
#line 1190 "xi-grammar.y"

void yyerror(const char *mesg)
{
    std::cerr << cur_file<<":"<<lineno<<": Charmxi syntax error> "
	      << mesg << std::endl;
}

