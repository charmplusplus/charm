
/* A Bison parser, made by GNU Bison 2.4.1.  */

/* Skeleton implementation for Bison's Yacc-like parsers in C
   
      Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003, 2004, 2005, 2006
   Free Software Foundation, Inc.
   
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
#define YYBISON_VERSION "2.4.1"

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

/* Line 189 of yacc.c  */
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
ModuleList *modlist;
namespace xi {
extern int macroDefined(const char *str, int istrue);
extern const char *python_doc;
void splitScopedName(const char* name, const char** scope, const char** basename);
}


/* Line 189 of yacc.c  */
#line 95 "y.tab.c"

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
     VOID = 290,
     CONST = 291,
     PACKED = 292,
     VARSIZE = 293,
     ENTRY = 294,
     FOR = 295,
     FORALL = 296,
     WHILE = 297,
     WHEN = 298,
     OVERLAP = 299,
     ATOMIC = 300,
     FORWARD = 301,
     IF = 302,
     ELSE = 303,
     CONNECT = 304,
     PUBLISHES = 305,
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
     REDUCTIONTARGET = 328
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
#define VOID 290
#define CONST 291
#define PACKED 292
#define VARSIZE 293
#define ENTRY 294
#define FOR 295
#define FORALL 296
#define WHILE 297
#define WHEN 298
#define OVERLAP 299
#define ATOMIC 300
#define FORWARD 301
#define IF 302
#define ELSE 303
#define CONNECT 304
#define PUBLISHES 305
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




#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE
{

/* Line 214 of yacc.c  */
#line 23 "xi-grammar.y"

  ModuleList *modlist;
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
  MemberList *mbrlist;
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
  XStr* xstrptr;
  AccelBlock* accelBlock;



/* Line 214 of yacc.c  */
#line 317 "y.tab.c"
} YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
#endif


/* Copy the second part of user declarations.  */


/* Line 264 of yacc.c  */
#line 329 "y.tab.c"

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
# if YYENABLE_NLS
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
#    if ! defined _ALLOCA_H && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#     ifndef _STDLIB_H
#      define _STDLIB_H 1
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
#  if (defined __cplusplus && ! defined _STDLIB_H \
       && ! ((defined YYMALLOC || defined malloc) \
	     && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef _STDLIB_H
#    define _STDLIB_H 1
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
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

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  9
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   916

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  90
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  113
/* YYNRULES -- Number of rules.  */
#define YYNRULES  302
/* YYNRULES -- Number of states.  */
#define YYNSTATES  606

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   328

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    84,     2,
      82,    83,    81,     2,    78,    88,    89,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    75,    74,
      79,    87,    80,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    85,     2,    86,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    76,     2,    77,     2,     2,     2,     2,
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
      65,    66,    67,    68,    69,    70,    71,    72,    73
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     5,     6,     9,    10,    12,    13,    15,
      17,    19,    24,    28,    32,    34,    39,    40,    43,    47,
      50,    53,    56,    64,    70,    76,    79,    82,    85,    88,
      91,    94,    97,   100,   102,   104,   106,   108,   110,   112,
     114,   116,   120,   121,   123,   124,   128,   130,   132,   134,
     136,   139,   142,   146,   150,   153,   156,   159,   161,   163,
     166,   168,   171,   174,   176,   178,   181,   184,   187,   196,
     198,   200,   202,   204,   207,   210,   213,   215,   217,   219,
     223,   224,   227,   232,   238,   239,   241,   242,   246,   248,
     252,   254,   256,   257,   261,   263,   267,   268,   270,   272,
     273,   277,   279,   283,   285,   287,   288,   290,   291,   294,
     300,   302,   305,   309,   316,   317,   320,   322,   326,   332,
     338,   344,   350,   355,   359,   366,   373,   379,   385,   391,
     397,   403,   408,   416,   417,   420,   421,   424,   427,   431,
     434,   438,   440,   444,   449,   452,   455,   458,   461,   464,
     466,   471,   472,   475,   477,   479,   481,   483,   486,   489,
     492,   496,   503,   513,   517,   524,   528,   535,   545,   555,
     557,   561,   563,   566,   569,   571,   574,   576,   578,   580,
     582,   584,   586,   588,   590,   592,   594,   596,   598,   606,
     612,   625,   631,   634,   636,   638,   639,   643,   645,   647,
     651,   653,   655,   657,   659,   661,   663,   665,   667,   669,
     671,   673,   675,   678,   680,   682,   684,   686,   688,   690,
     691,   693,   697,   698,   700,   706,   712,   718,   723,   727,
     729,   731,   733,   737,   742,   746,   748,   750,   752,   754,
     759,   763,   768,   773,   778,   782,   790,   796,   803,   805,
     809,   811,   815,   819,   822,   826,   829,   830,   834,   835,
     837,   841,   843,   846,   848,   851,   852,   857,   859,   863,
     865,   866,   873,   882,   887,   891,   897,   902,   914,   924,
     937,   952,   959,   968,   974,   982,   986,   990,   992,   993,
     996,  1001,  1003,  1007,  1009,  1011,  1014,  1020,  1022,  1026,
    1028,  1030,  1033
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
      91,     0,    -1,    92,    -1,    -1,    97,    92,    -1,    -1,
       5,    -1,    -1,    74,    -1,    55,    -1,    55,    -1,    96,
      75,    75,    55,    -1,     3,    95,    98,    -1,     4,    95,
      98,    -1,    74,    -1,    76,    99,    77,    94,    -1,    -1,
     101,    99,    -1,    54,    53,    96,    -1,    54,    96,    -1,
      93,   156,    -1,    93,   135,    -1,     5,    39,   166,   108,
      95,   105,   183,    -1,    93,    76,    99,    77,    94,    -1,
      53,    95,    76,    99,    77,    -1,   100,    74,    -1,   100,
     163,    -1,    93,    97,    -1,    93,   138,    -1,    93,   139,
      -1,    93,   140,    -1,    93,   142,    -1,    93,   153,    -1,
     201,    -1,   202,    -1,   165,    -1,     1,    -1,   114,    -1,
      56,    -1,    57,    -1,   102,    -1,   102,    78,   103,    -1,
      -1,   103,    -1,    -1,    79,   104,    80,    -1,    61,    -1,
      62,    -1,    63,    -1,    64,    -1,    67,    61,    -1,    67,
      62,    -1,    67,    62,    61,    -1,    67,    62,    62,    -1,
      67,    63,    -1,    67,    64,    -1,    62,    62,    -1,    65,
      -1,    66,    -1,    62,    66,    -1,    35,    -1,    95,   105,
      -1,    96,   105,    -1,   106,    -1,   108,    -1,   109,    81,
      -1,   110,    81,    -1,   111,    81,    -1,   113,    82,    81,
      95,    83,    82,   181,    83,    -1,   109,    -1,   110,    -1,
     111,    -1,   112,    -1,    36,   113,    -1,   113,    36,    -1,
     113,    84,    -1,   113,    -1,    56,    -1,    96,    -1,    85,
     115,    86,    -1,    -1,   116,   117,    -1,     6,   114,    96,
     117,    -1,     6,    16,   109,    81,    95,    -1,    -1,    35,
      -1,    -1,    85,   122,    86,    -1,   123,    -1,   123,    78,
     122,    -1,    37,    -1,    38,    -1,    -1,    85,   125,    86,
      -1,   130,    -1,   130,    78,   125,    -1,    -1,    57,    -1,
      51,    -1,    -1,    85,   129,    86,    -1,   127,    -1,   127,
      78,   129,    -1,    30,    -1,    51,    -1,    -1,    17,    -1,
      -1,    85,    86,    -1,   131,   114,    95,   132,    74,    -1,
     133,    -1,   133,   134,    -1,    16,   121,   107,    -1,    16,
     121,   107,    76,   134,    77,    -1,    -1,    75,   137,    -1,
     108,    -1,   108,    78,   137,    -1,    11,   124,   107,   136,
     154,    -1,    12,   124,   107,   136,   154,    -1,    13,   124,
     107,   136,   154,    -1,    14,   124,   107,   136,   154,    -1,
      85,    56,    95,    86,    -1,    85,    95,    86,    -1,    15,
     128,   141,   107,   136,   154,    -1,    15,   141,   128,   107,
     136,   154,    -1,    11,   124,    95,   136,   154,    -1,    12,
     124,    95,   136,   154,    -1,    13,   124,    95,   136,   154,
      -1,    14,   124,    95,   136,   154,    -1,    15,   141,    95,
     136,   154,    -1,    16,   121,    95,    74,    -1,    16,   121,
      95,    76,   134,    77,    74,    -1,    -1,    87,   114,    -1,
      -1,    87,    56,    -1,    87,    57,    -1,    18,    95,   148,
      -1,   112,   149,    -1,   114,    95,   149,    -1,   150,    -1,
     150,    78,   151,    -1,    22,    79,   151,    80,    -1,   152,
     143,    -1,   152,   144,    -1,   152,   145,    -1,   152,   146,
      -1,   152,   147,    -1,    74,    -1,    76,   155,    77,    94,
      -1,    -1,   161,   155,    -1,   118,    -1,   119,    -1,   158,
      -1,   157,    -1,    10,   159,    -1,    19,   160,    -1,    18,
      95,    -1,     8,   120,    96,    -1,     8,   120,    96,    82,
     120,    83,    -1,     8,   120,    96,    79,   103,    80,    82,
     120,    83,    -1,     7,   120,    96,    -1,     7,   120,    96,
      82,   120,    83,    -1,     9,   120,    96,    -1,     9,   120,
      96,    82,   120,    83,    -1,     9,   120,    96,    79,   103,
      80,    82,   120,    83,    -1,     9,    85,    68,    86,   120,
      96,    82,   120,    83,    -1,   108,    -1,   108,    78,   159,
      -1,    57,    -1,   162,    74,    -1,   162,   163,    -1,   164,
      -1,   152,   164,    -1,   156,    -1,    39,    -1,    77,    -1,
       7,    -1,     8,    -1,     9,    -1,    11,    -1,    12,    -1,
      15,    -1,    13,    -1,    14,    -1,     6,    -1,    39,   167,
     166,    95,   183,   185,   186,    -1,    39,   167,    95,   183,
     186,    -1,    39,    85,    68,    86,    35,    95,   183,   184,
     174,   172,   175,    95,    -1,    71,   174,   172,   175,    74,
      -1,    71,    74,    -1,    35,    -1,   110,    -1,    -1,    85,
     168,    86,    -1,     1,    -1,   169,    -1,   169,    78,   168,
      -1,    21,    -1,    23,    -1,    24,    -1,    25,    -1,    31,
      -1,    32,    -1,    33,    -1,    34,    -1,    26,    -1,    27,
      -1,    28,    -1,    52,    -1,    51,   126,    -1,    72,    -1,
      73,    -1,     1,    -1,    57,    -1,    56,    -1,    96,    -1,
      -1,    58,    -1,    58,    78,   171,    -1,    -1,    58,    -1,
      58,    85,   172,    86,   172,    -1,    58,    76,   172,    77,
     172,    -1,    58,    82,   171,    83,   172,    -1,    82,   172,
      83,   172,    -1,   114,    95,    85,    -1,    76,    -1,    77,
      -1,   114,    -1,   114,    95,   131,    -1,   114,    95,    87,
     170,    -1,   173,   172,    86,    -1,     6,    -1,    69,    -1,
      70,    -1,    95,    -1,   178,    88,    80,    95,    -1,   178,
      89,    95,    -1,   178,    85,   178,    86,    -1,   178,    85,
      56,    86,    -1,   178,    82,   178,    83,    -1,   173,   172,
      86,    -1,   177,    75,   114,    95,    79,   178,    80,    -1,
     114,    95,    79,   178,    80,    -1,   177,    75,   179,    79,
     178,    80,    -1,   176,    -1,   176,    78,   181,    -1,   180,
      -1,   180,    78,   182,    -1,    82,   181,    83,    -1,    82,
      83,    -1,    85,   182,    86,    -1,    85,    86,    -1,    -1,
      20,    87,    56,    -1,    -1,   192,    -1,    76,   187,    77,
      -1,   192,    -1,   192,   187,    -1,   192,    -1,   192,   187,
      -1,    -1,    50,    82,   190,    83,    -1,    55,    -1,    55,
      78,   190,    -1,    57,    -1,    -1,    45,   191,   174,   172,
     175,   189,    -1,    49,    82,    55,   183,    83,   174,   172,
      77,    -1,    43,   198,    76,    77,    -1,    43,   198,   192,
      -1,    43,   198,    76,   187,    77,    -1,    44,    76,   188,
      77,    -1,    40,   196,   172,    74,   172,    74,   172,   195,
      76,   187,    77,    -1,    40,   196,   172,    74,   172,    74,
     172,   195,   192,    -1,    41,    85,    55,    86,   196,   172,
      75,   172,    78,   172,   195,   192,    -1,    41,    85,    55,
      86,   196,   172,    75,   172,    78,   172,   195,    76,   187,
      77,    -1,    47,   196,   172,   195,   192,   193,    -1,    47,
     196,   172,   195,    76,   187,    77,   193,    -1,    42,   196,
     172,   195,   192,    -1,    42,   196,   172,   195,    76,   187,
      77,    -1,    46,   194,    74,    -1,   174,   172,   175,    -1,
       1,    -1,    -1,    48,   192,    -1,    48,    76,   187,    77,
      -1,    55,    -1,    55,    78,   194,    -1,    83,    -1,    82,
      -1,    55,   183,    -1,    55,   199,   172,   200,   183,    -1,
     197,    -1,   197,    78,   198,    -1,    85,    -1,    86,    -1,
      59,    95,    -1,    60,    95,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   154,   154,   159,   162,   167,   168,   173,   174,   178,
     182,   184,   192,   196,   203,   205,   210,   211,   215,   217,
     219,   221,   223,   235,   237,   239,   241,   243,   245,   247,
     249,   251,   253,   255,   257,   259,   261,   265,   267,   269,
     273,   275,   280,   281,   286,   287,   291,   293,   295,   297,
     299,   301,   303,   305,   307,   309,   311,   313,   315,   317,
     319,   323,   324,   331,   333,   337,   341,   343,   347,   351,
     353,   355,   357,   360,   362,   366,   368,   372,   374,   378,
     383,   384,   388,   392,   397,   398,   403,   404,   414,   416,
     420,   422,   427,   428,   432,   434,   439,   440,   444,   449,
     450,   454,   456,   460,   462,   467,   468,   472,   473,   476,
     480,   482,   486,   488,   493,   494,   498,   500,   504,   506,
     510,   514,   518,   524,   528,   530,   534,   536,   540,   544,
     548,   552,   554,   559,   560,   565,   566,   568,   572,   574,
     576,   580,   582,   586,   590,   592,   594,   596,   598,   602,
     604,   609,   616,   620,   622,   624,   625,   627,   629,   631,
     635,   637,   639,   645,   648,   653,   655,   657,   663,   671,
     673,   676,   680,   683,   687,   689,   694,   698,   700,   702,
     704,   706,   708,   710,   712,   714,   716,   718,   721,   729,
     742,   760,   762,   766,   768,   773,   774,   776,   780,   782,
     786,   788,   790,   792,   794,   796,   798,   800,   802,   804,
     806,   808,   810,   812,   814,   816,   820,   822,   824,   829,
     830,   832,   841,   842,   844,   850,   856,   862,   870,   877,
     885,   892,   894,   896,   898,   905,   906,   907,   910,   911,
     912,   913,   920,   926,   935,   942,   948,   954,   962,   964,
     968,   970,   974,   976,   980,   982,   987,   988,   993,   994,
     996,  1000,  1002,  1006,  1008,  1013,  1014,  1018,  1020,  1024,
    1027,  1030,  1034,  1048,  1050,  1052,  1054,  1056,  1059,  1062,
    1065,  1068,  1070,  1072,  1074,  1076,  1078,  1080,  1087,  1088,
    1090,  1093,  1095,  1099,  1103,  1107,  1109,  1113,  1115,  1119,
    1122,  1126,  1130
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
  "CREATEHERE", "CREATEHOME", "NOKEEP", "NOTRACE", "VOID", "CONST",
  "PACKED", "VARSIZE", "ENTRY", "FOR", "FORALL", "WHILE", "WHEN",
  "OVERLAP", "ATOMIC", "FORWARD", "IF", "ELSE", "CONNECT", "PUBLISHES",
  "PYTHON", "LOCAL", "NAMESPACE", "USING", "IDENT", "NUMBER", "LITERAL",
  "CPROGRAM", "HASHIF", "HASHIFDEF", "INT", "LONG", "SHORT", "CHAR",
  "FLOAT", "DOUBLE", "UNSIGNED", "ACCEL", "READWRITE", "WRITEONLY",
  "ACCELBLOCK", "MEMCRITICAL", "REDUCTIONTARGET", "';'", "':'", "'{'",
  "'}'", "','", "'<'", "'>'", "'*'", "'('", "')'", "'&'", "'['", "']'",
  "'='", "'-'", "'.'", "$accept", "File", "ModuleEList", "OptExtern",
  "OptSemiColon", "Name", "QualName", "Module", "ConstructEList",
  "ConstructList", "ConstructSemi", "Construct", "TParam", "TParamList",
  "TParamEList", "OptTParams", "BuiltinType", "NamedType", "QualNamedType",
  "SimpleType", "OnePtrType", "PtrType", "FuncType", "BaseType", "Type",
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
  "Slist", "Olist", "OptPubList", "PublishesList", "OptTraceName",
  "SingleConstruct", "HasElse", "ForwardList", "EndIntExpr",
  "StartIntExpr", "SEntry", "SEntryList", "SParamBracketStart",
  "SParamBracketEnd", "HashIFComment", "HashIFDefComment", 0
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
     325,   326,   327,   328,    59,    58,   123,   125,    44,    60,
      62,    42,    40,    41,    38,    91,    93,    61,    45,    46
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    90,    91,    92,    92,    93,    93,    94,    94,    95,
      96,    96,    97,    97,    98,    98,    99,    99,   100,   100,
     100,   100,   100,   101,   101,   101,   101,   101,   101,   101,
     101,   101,   101,   101,   101,   101,   101,   102,   102,   102,
     103,   103,   104,   104,   105,   105,   106,   106,   106,   106,
     106,   106,   106,   106,   106,   106,   106,   106,   106,   106,
     106,   107,   108,   109,   109,   110,   111,   111,   112,   113,
     113,   113,   113,   113,   113,   114,   114,   115,   115,   116,
     117,   117,   118,   119,   120,   120,   121,   121,   122,   122,
     123,   123,   124,   124,   125,   125,   126,   126,   127,   128,
     128,   129,   129,   130,   130,   131,   131,   132,   132,   133,
     134,   134,   135,   135,   136,   136,   137,   137,   138,   138,
     139,   140,   141,   141,   142,   142,   143,   143,   144,   145,
     146,   147,   147,   148,   148,   149,   149,   149,   150,   150,
     150,   151,   151,   152,   153,   153,   153,   153,   153,   154,
     154,   155,   155,   156,   156,   156,   156,   156,   156,   156,
     157,   157,   157,   157,   157,   158,   158,   158,   158,   159,
     159,   160,   161,   161,   162,   162,   162,   163,   163,   163,
     163,   163,   163,   163,   163,   163,   163,   163,   164,   164,
     164,   165,   165,   166,   166,   167,   167,   167,   168,   168,
     169,   169,   169,   169,   169,   169,   169,   169,   169,   169,
     169,   169,   169,   169,   169,   169,   170,   170,   170,   171,
     171,   171,   172,   172,   172,   172,   172,   172,   173,   174,
     175,   176,   176,   176,   176,   177,   177,   177,   178,   178,
     178,   178,   178,   178,   179,   180,   180,   180,   181,   181,
     182,   182,   183,   183,   184,   184,   185,   185,   186,   186,
     186,   187,   187,   188,   188,   189,   189,   190,   190,   191,
     191,   192,   192,   192,   192,   192,   192,   192,   192,   192,
     192,   192,   192,   192,   192,   192,   192,   192,   193,   193,
     193,   194,   194,   195,   196,   197,   197,   198,   198,   199,
     200,   201,   202
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     0,     2,     0,     1,     0,     1,     1,
       1,     4,     3,     3,     1,     4,     0,     2,     3,     2,
       2,     2,     7,     5,     5,     2,     2,     2,     2,     2,
       2,     2,     2,     1,     1,     1,     1,     1,     1,     1,
       1,     3,     0,     1,     0,     3,     1,     1,     1,     1,
       2,     2,     3,     3,     2,     2,     2,     1,     1,     2,
       1,     2,     2,     1,     1,     2,     2,     2,     8,     1,
       1,     1,     1,     2,     2,     2,     1,     1,     1,     3,
       0,     2,     4,     5,     0,     1,     0,     3,     1,     3,
       1,     1,     0,     3,     1,     3,     0,     1,     1,     0,
       3,     1,     3,     1,     1,     0,     1,     0,     2,     5,
       1,     2,     3,     6,     0,     2,     1,     3,     5,     5,
       5,     5,     4,     3,     6,     6,     5,     5,     5,     5,
       5,     4,     7,     0,     2,     0,     2,     2,     3,     2,
       3,     1,     3,     4,     2,     2,     2,     2,     2,     1,
       4,     0,     2,     1,     1,     1,     1,     2,     2,     2,
       3,     6,     9,     3,     6,     3,     6,     9,     9,     1,
       3,     1,     2,     2,     1,     2,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     7,     5,
      12,     5,     2,     1,     1,     0,     3,     1,     1,     3,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     2,     1,     1,     1,     1,     1,     1,     0,
       1,     3,     0,     1,     5,     5,     5,     4,     3,     1,
       1,     1,     3,     4,     3,     1,     1,     1,     1,     4,
       3,     4,     4,     4,     3,     7,     5,     6,     1,     3,
       1,     3,     3,     2,     3,     2,     0,     3,     0,     1,
       3,     1,     2,     1,     2,     0,     4,     1,     3,     1,
       0,     6,     8,     4,     3,     5,     4,    11,     9,    12,
      14,     6,     8,     5,     7,     3,     3,     1,     0,     2,
       4,     1,     3,     1,     1,     2,     5,     1,     3,     1,
       1,     2,     2
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       3,     0,     0,     0,     2,     3,     9,     0,     0,     1,
       4,    14,     0,    12,    13,    36,     6,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    35,    33,    34,     0,
       0,     0,    10,    19,   301,   302,   192,   229,   222,     0,
      84,    84,    84,     0,    92,    92,    92,    92,     0,    86,
       0,     0,     0,     0,    27,   153,   154,    21,    28,    29,
      30,    31,     0,    32,    20,   156,   155,     7,   187,   179,
     180,   181,   182,   183,   185,   186,   184,   177,    25,   178,
      26,    17,    60,    46,    47,    48,    49,    57,    58,     0,
      44,    63,    64,     0,   194,     0,     0,    18,     0,   223,
     222,     0,     0,    60,     0,    69,    70,    71,    72,    76,
       0,    85,     0,     0,     0,     0,   169,   157,     0,     0,
       0,     0,     0,     0,     0,    99,     0,     0,   159,   171,
     158,     0,     0,    92,    92,    92,    92,     0,    86,   144,
     145,   146,   147,   148,     8,    15,    56,    59,    50,    51,
      54,    55,    42,    62,    65,     0,     0,     0,   222,   219,
     222,     0,   230,     0,     0,    73,    66,    67,    74,     0,
      75,    80,   163,   160,     0,   165,     0,   103,   104,     0,
      94,    44,   114,   114,   114,   114,    98,     0,     0,   101,
       0,     0,     0,     0,     0,    90,    91,     0,    88,   112,
       0,    72,     0,   141,     0,     7,     0,     0,     0,     0,
       0,     0,    52,    53,    38,    39,    40,    43,     0,    37,
      44,    24,    11,     0,   220,     0,     0,   222,   191,     0,
       0,     0,    80,    82,    84,     0,    84,    84,     0,    84,
     170,    93,     0,    61,     0,     0,     0,     0,     0,     0,
     123,     0,   100,   114,   114,    87,     0,   105,   133,     0,
     139,   135,     0,   143,    23,   114,   114,   114,   114,   114,
       0,     0,    45,     0,   222,   219,   222,   222,   227,    83,
       0,    77,    78,     0,    81,     0,     0,     0,     0,     0,
       0,    95,   116,   115,   149,   151,   118,   119,   120,   121,
     122,   102,     0,     0,    89,   106,     0,   105,     0,     0,
     138,   136,   137,   140,   142,     0,     0,     0,     0,     0,
     131,   105,    41,     0,    22,   225,   221,   226,   224,     0,
      79,   164,     0,   161,     0,     0,   166,     0,     0,     0,
       0,   176,   151,     0,   174,   124,   125,     0,   111,   113,
     134,   126,   127,   128,   129,   130,     0,   253,   231,   222,
     248,     0,     0,    84,    84,    84,   117,   197,     0,     0,
     175,     7,   152,   172,   173,   107,     0,   105,     0,     0,
     252,     0,     0,     0,     0,   215,   200,   201,   202,   203,
     208,   209,   210,   204,   205,   206,   207,    96,   211,     0,
     213,   214,     0,   198,    10,     0,     0,   150,     0,     0,
     132,   228,     0,   232,   234,   249,    68,   162,   168,   167,
      97,   212,     0,   196,     0,     0,     0,   108,   109,   217,
     216,   218,   233,     0,   199,   287,     0,     0,     0,     0,
       0,   270,     0,     0,     0,     0,   222,   189,   259,   256,
       0,   294,   222,     0,   222,     0,   297,     0,     0,   269,
       0,   291,     0,   222,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   299,   295,   222,     0,     0,   274,
       0,     0,   222,     0,   285,     0,     0,   260,   262,   286,
       0,   188,     0,     0,   222,     0,   293,     0,     0,   298,
     273,     0,   276,   264,     0,   292,     0,     0,   257,   235,
     236,   237,   255,     0,     0,   250,     0,   222,     0,   222,
       0,   283,   300,     0,   275,   265,     0,   288,     0,     0,
       0,     0,   254,     0,   222,     0,     0,   296,     0,   271,
       0,     0,   281,   222,     0,     0,   222,     0,   251,     0,
       0,   222,   284,     0,   288,     0,   289,     0,   238,     0,
       0,     0,     0,   190,     0,     0,   267,     0,   282,     0,
     272,   246,     0,     0,     0,     0,     0,   244,     0,     0,
     278,   222,     0,   266,   290,     0,     0,     0,     0,   240,
       0,   247,     0,     0,   268,   243,   242,   241,   239,   245,
     277,     0,     0,   279,     0,   280
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     3,     4,    22,   145,   181,    90,     5,    13,    23,
      24,    25,   216,   217,   218,   153,    91,   182,    92,   105,
     106,   107,   108,   109,   219,   283,   232,   233,    55,    56,
     112,   127,   197,   198,   119,   179,   421,   189,   124,   190,
     180,   306,   409,   307,   308,    57,   245,   293,    58,    59,
      60,   125,    61,   139,   140,   141,   142,   143,   310,   260,
     203,   204,   339,    63,   296,   340,   341,    65,    66,   117,
     130,   342,   343,    80,   344,    26,    95,   369,   402,   403,
     432,   225,   101,   359,   446,   163,   360,   514,   559,   547,
     515,   361,   516,   324,   493,   469,   447,   465,   480,   539,
     567,   460,   466,   542,   462,   497,   452,   456,   457,   476,
     523,    27,    28
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -474
static const yytype_int16 yypact[] =
{
     114,   -31,   -31,    66,  -474,   114,  -474,    71,    71,  -474,
    -474,  -474,   457,  -474,  -474,  -474,    69,   -31,    31,   -31,
     -31,   120,   657,    49,   681,   457,  -474,  -474,  -474,   816,
      80,   129,  -474,   100,  -474,  -474,  -474,  -474,   -30,   568,
     152,   152,   -14,   129,   108,   108,   108,   108,   117,   131,
     -31,   166,   161,   457,  -474,  -474,  -474,  -474,  -474,  -474,
    -474,  -474,   349,  -474,  -474,  -474,  -474,   169,  -474,  -474,
    -474,  -474,  -474,  -474,  -474,  -474,  -474,  -474,  -474,  -474,
    -474,  -474,   191,  -474,   111,  -474,  -474,  -474,  -474,   235,
     138,  -474,  -474,   171,  -474,   129,   457,   100,   175,    21,
     -30,   184,   834,  -474,   801,   171,   195,   205,  -474,    25,
     129,  -474,   129,   129,   220,   129,   215,  -474,     3,   -31,
     -31,   -31,   -31,   227,   216,   217,   124,   -31,  -474,  -474,
    -474,   760,   226,   108,   108,   108,   108,   216,   131,  -474,
    -474,  -474,  -474,  -474,  -474,  -474,  -474,  -474,  -474,   203,
    -474,  -474,   783,  -474,  -474,   -31,   239,   249,   -30,   255,
     -30,   234,  -474,   254,   248,    10,  -474,  -474,  -474,   251,
    -474,     8,    48,   113,   244,   136,   129,  -474,  -474,   258,
     263,   268,   284,   284,   284,   284,  -474,   -31,   280,   294,
     287,   219,   -31,   323,   -31,  -474,  -474,   289,   298,   301,
     -31,    11,   -31,   300,   302,   169,   -31,   -31,   -31,   -31,
     -31,   -31,  -474,  -474,  -474,  -474,   303,  -474,   304,  -474,
     268,  -474,  -474,   306,   307,   296,   310,   -30,  -474,   -31,
     -31,   229,   314,  -474,   152,   783,   152,   152,   783,   152,
    -474,  -474,     3,  -474,   129,   145,   145,   145,   145,   316,
    -474,   323,  -474,   284,   284,  -474,   124,   383,   317,   212,
    -474,   319,   760,  -474,  -474,   284,   284,   284,   284,   284,
     173,   783,  -474,   305,   -30,   255,   -30,   -30,  -474,  -474,
     320,  -474,   100,   321,  -474,   325,   329,   335,   129,   340,
     339,  -474,   347,  -474,  -474,   795,  -474,  -474,  -474,  -474,
    -474,  -474,   145,   145,  -474,  -474,   801,    -3,   350,   801,
    -474,  -474,  -474,  -474,  -474,   145,   145,   145,   145,   145,
    -474,   383,  -474,   745,  -474,  -474,  -474,  -474,  -474,   344,
    -474,  -474,   346,  -474,    54,   351,  -474,   129,   104,   390,
     353,  -474,   795,   758,  -474,  -474,  -474,   -31,  -474,  -474,
    -474,  -474,  -474,  -474,  -474,  -474,   354,  -474,   -31,   -30,
     356,   352,   801,   152,   152,   152,  -474,  -474,   676,   849,
    -474,   169,  -474,  -474,  -474,   355,   362,     0,   357,   801,
    -474,   359,   361,   365,   366,  -474,  -474,  -474,  -474,  -474,
    -474,  -474,  -474,  -474,  -474,  -474,  -474,   380,  -474,   367,
    -474,  -474,   368,   373,   363,   305,   -31,  -474,   391,   378,
    -474,  -474,     1,  -474,  -474,  -474,  -474,  -474,  -474,  -474,
    -474,  -474,   404,  -474,   690,   551,   305,  -474,  -474,  -474,
    -474,   100,  -474,   -31,  -474,  -474,   377,   393,   377,   425,
     406,   426,   429,   377,   405,   190,   -30,  -474,  -474,   466,
     305,  -474,   -30,   433,   -30,   -50,   411,   348,   370,  -474,
     414,   413,   418,   -30,   438,   427,   278,   184,   419,   551,
     422,   434,   428,   430,  -474,  -474,   -30,   425,   213,  -474,
     441,   293,   -30,   429,  -474,   430,   305,  -474,  -474,  -474,
     459,  -474,   245,   414,   -30,   377,  -474,   456,   444,  -474,
    -474,   442,  -474,  -474,   184,  -474,   480,   448,  -474,  -474,
    -474,  -474,  -474,   -31,   460,   458,   452,   -30,   465,   -30,
     190,  -474,  -474,   305,  -474,   490,   190,   493,   414,   463,
     801,   724,  -474,   184,   -30,   474,   473,  -474,   469,  -474,
     476,   536,  -474,   -30,   -31,   -31,   -30,   475,  -474,   -31,
     430,   -30,  -474,   500,   493,   190,  -474,   491,  -474,    39,
      55,   481,   -31,  -474,   598,   492,   495,   486,  -474,   509,
    -474,  -474,   -31,   236,   508,   -31,   -31,  -474,    75,   190,
    -474,   -30,   500,  -474,  -474,   115,   503,   156,   -31,  -474,
      94,  -474,   528,   430,  -474,  -474,  -474,  -474,  -474,  -474,
    -474,   608,   190,  -474,   529,  -474
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -474,  -474,   602,  -474,  -199,    -1,   -11,   586,   603,    -2,
    -474,  -474,  -474,  -191,  -474,  -150,  -474,   -82,   -32,   -24,
     -21,  -474,  -121,   506,   -36,  -474,  -474,   381,  -474,  -474,
     -12,   482,   400,  -474,   -20,   379,  -474,  -474,   494,   371,
    -474,   247,  -474,  -474,  -248,  -474,  -116,   309,  -474,  -474,
    -474,   -49,  -474,  -474,  -474,  -474,  -474,  -474,  -474,   375,
    -474,   364,   615,  -474,   299,   336,   636,  -474,  -474,   483,
    -474,  -474,  -474,   337,   342,  -474,   313,  -474,   238,  -474,
    -474,   408,   -96,   155,   -19,  -454,  -474,  -474,  -430,  -474,
    -474,  -307,   167,  -390,  -474,  -474,   237,  -444,  -474,  -474,
     123,  -474,  -416,   132,   242,  -473,  -395,  -474,   252,  -474,
    -474,  -474,  -474
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -264
static const yytype_int16 yytable[] =
{
       7,     8,    38,   110,   161,    93,   264,    33,    94,   448,
     201,   116,   506,   489,   305,   425,    30,   305,    34,    35,
      97,   111,   488,    81,     6,   120,   121,   122,    99,   113,
     115,   243,   323,   177,   501,   474,   449,   503,   183,   184,
     185,   479,   481,   454,   286,   199,   168,   289,   463,   128,
     525,   132,   100,   448,   178,   381,    32,   429,   430,   348,
     470,   168,   223,   155,   226,   475,     9,   246,   247,   248,
     273,   114,   415,   356,  -110,   192,   536,   564,   164,   549,
     322,   521,   540,    98,    31,   411,    32,   412,   210,  -135,
     527,  -135,   169,   231,   156,   202,   507,   158,   259,   171,
     519,   172,   173,   159,   175,   367,   160,   169,    29,   170,
     253,   569,   254,   206,   207,   208,   209,     1,     2,   571,
     601,   572,   188,    98,   573,   556,    67,   574,   575,    98,
     234,   278,   578,   537,   576,   592,   364,   302,   303,  -195,
     411,   201,   585,   587,   116,    11,   590,    12,   580,   315,
     316,   317,   318,   319,   220,   591,    96,   572,   604,  -195,
     573,   195,   196,   574,   575,  -195,  -195,  -195,  -195,  -195,
    -195,  -195,   407,   146,   599,    98,   572,   147,   325,   573,
     327,   328,   574,   575,    32,   603,   249,   111,    98,   368,
     188,   435,   235,   118,    36,   236,    37,   572,   595,   258,
     573,   261,   123,   574,   575,   265,   266,   267,   268,   269,
     270,    98,   292,    98,   435,   238,   126,   152,   239,   294,
     282,   295,   285,   129,   287,   288,   202,   290,   279,   280,
     436,   437,   438,   439,   440,   441,   442,   443,   572,   444,
     131,   573,   597,   144,   574,   575,  -193,   320,  -229,   321,
     157,   509,   154,   436,   437,   438,   439,   440,   441,   442,
     443,   162,   444,   378,   212,   213,    37,  -229,   311,   312,
     347,  -229,  -229,   350,     6,   187,   166,   334,   186,   435,
     103,   104,     6,   187,    32,   281,   167,   358,   174,    37,
     500,     6,   586,   176,   435,  -229,   148,   149,   150,   151,
      32,   191,   193,   205,   222,   292,    83,    84,    85,    86,
      87,    88,    89,   224,   510,   511,   221,   227,   436,   437,
     438,   439,   440,   441,   442,   443,   358,   444,   228,   229,
     237,   512,   230,   436,   437,   438,   439,   440,   441,   442,
     443,   242,   444,   358,   241,    93,   375,   152,    94,   435,
     467,   382,   383,   384,    37,  -261,   471,   377,   473,   244,
     133,   134,   135,   136,   137,   138,   250,   485,   405,    37,
    -263,   435,   251,   252,   186,   255,   256,   257,   262,   276,
     498,   271,   263,   274,   272,   275,   504,   323,   436,   437,
     438,   439,   440,   441,   442,   443,   277,   444,   518,   231,
     305,   431,   300,   329,   309,   426,   259,   330,   331,   332,
     436,   437,   438,   439,   440,   441,   442,   443,   333,   444,
     335,   533,   336,   535,   478,   337,   362,   349,   363,   338,
     371,   376,   450,   365,   379,   380,   410,   420,   550,   433,
     408,   482,   416,   414,   417,    -9,    37,   557,   418,   419,
     561,   424,   428,   422,   423,   565,   513,   435,    15,   451,
      -5,    -5,    16,    -5,    -5,    -5,    -5,    -5,    -5,    -5,
      -5,    -5,    -5,    -5,   517,    -5,    -5,   427,   453,    -5,
     455,   435,   458,   459,   461,   593,   468,   464,   472,   477,
      37,   483,   484,   486,   545,   513,   436,   437,   438,   439,
     440,   441,   442,   443,   487,   444,   490,   492,   494,   543,
      17,    18,   529,   496,   495,   508,    19,    20,   502,   524,
     436,   437,   438,   439,   440,   441,   442,   443,    21,   444,
     522,   528,   520,    -5,   -16,   530,   531,   435,   532,   534,
     538,   541,   544,   558,   560,   297,   298,   299,   563,   551,
     552,   553,   435,   554,   562,   566,   526,  -258,  -258,  -258,
    -258,   558,  -258,  -258,  -258,  -258,  -258,   577,   570,   583,
     581,   558,   558,   582,   589,   558,   436,   437,   438,   439,
     440,   441,   442,   443,   102,   444,   584,   598,   588,   596,
    -258,   436,   437,   438,   439,   440,   441,   442,   443,   435,
     444,   345,   346,   103,   104,   600,   605,    10,    54,   435,
     165,    14,   555,   284,   351,   352,   353,   354,   355,   194,
     211,   291,   301,    32,   413,  -258,   314,   445,  -258,    83,
      84,    85,    86,    87,    88,    89,   313,    62,   436,   437,
     438,   439,   440,   441,   442,   443,   366,   444,   436,   437,
     438,   439,   440,   441,   442,   443,   304,   444,    64,   240,
       1,     2,   434,    39,    40,    41,    42,    43,    44,    45,
      46,    47,    48,    49,   579,    50,    51,   385,   372,    52,
     374,   370,   406,   326,   602,   546,   568,    68,    69,    70,
      71,   385,    72,    73,    74,    75,    76,   386,   548,   387,
     388,   389,   390,   391,   392,   594,   491,   393,   394,   395,
     396,   386,     0,   387,   388,   389,   390,   391,   392,     0,
      77,   393,   394,   395,   396,   505,     0,   397,   398,   499,
     509,     0,     0,    53,     0,     0,     0,     0,     0,     0,
       0,   397,   398,     0,   399,     0,     0,     0,   400,   401,
       0,     0,     0,     0,     0,    78,     0,     0,    79,   103,
     104,     0,   400,   401,    68,    69,    70,    71,     0,    72,
      73,    74,    75,    76,     0,     0,     0,     0,   200,    32,
     103,   104,     0,     0,     0,    83,    84,    85,    86,    87,
      88,    89,     0,   510,   511,   103,   104,    77,     0,     0,
      32,    39,    40,    41,    42,    43,    83,    84,    85,    86,
      87,    88,    89,    50,    51,    32,     0,    52,   103,   104,
       0,    83,    84,    85,    86,    87,    88,    89,   357,     0,
       0,     0,   373,     0,   338,    79,   103,   104,    32,   214,
     215,     0,     0,     0,    83,    84,    85,    86,    87,    88,
      89,    82,     0,     0,     0,     0,    32,     0,     0,     0,
       0,     0,    83,    84,    85,    86,    87,    88,    89,   103,
       0,    32,     0,     0,     0,     0,     0,    83,    84,    85,
      86,    87,    88,    89,    82,     0,     0,     0,     0,    32,
       0,     0,     0,     0,     0,    83,    84,    85,    86,    87,
      88,    89,     0,     0,   404,     0,     0,     0,     0,     0,
      83,    84,    85,    86,    87,    88,    89
};

static const yytype_int16 yycheck[] =
{
       1,     2,    21,    39,   100,    29,   205,    18,    29,   425,
     131,    43,   485,   467,    17,   405,    17,    17,    19,    20,
      31,    35,   466,    25,    55,    45,    46,    47,    58,    41,
      42,   181,    82,    30,   478,    85,   426,   481,   120,   121,
     122,   457,   458,   438,   235,   127,    36,   238,   443,    50,
     504,    53,    82,   469,    51,   362,    55,    56,    57,   307,
     450,    36,   158,    95,   160,   455,     0,   183,   184,   185,
     220,    85,   379,   321,    77,   124,   520,   550,   102,   533,
     271,   497,   526,    75,    53,    85,    55,    87,   137,    78,
     506,    80,    82,    85,    96,   131,   486,    76,    87,   110,
     495,   112,   113,    82,   115,     1,    85,    82,    39,    84,
     192,   555,   194,   133,   134,   135,   136,     3,     4,    80,
     593,    82,   123,    75,    85,   541,    77,    88,    89,    75,
      82,   227,   562,   523,    79,   579,    82,   253,   254,    35,
      85,   262,   572,   573,   176,    74,   576,    76,   564,   265,
     266,   267,   268,   269,   155,    80,    76,    82,   602,    55,
      85,    37,    38,    88,    89,    61,    62,    63,    64,    65,
      66,    67,   371,    62,    80,    75,    82,    66,   274,    85,
     276,   277,    88,    89,    55,   601,   187,    35,    75,    85,
     191,     1,    79,    85,    74,    82,    76,    82,    83,   200,
      85,   202,    85,    88,    89,   206,   207,   208,   209,   210,
     211,    75,   244,    75,     1,    79,    85,    79,    82,    74,
     231,    76,   234,    57,   236,   237,   262,   239,   229,   230,
      40,    41,    42,    43,    44,    45,    46,    47,    82,    49,
      79,    85,    86,    74,    88,    89,    55,    74,    58,    76,
      75,     6,    81,    40,    41,    42,    43,    44,    45,    46,
      47,    77,    49,   359,    61,    62,    76,    77,    56,    57,
     306,    58,    82,   309,    55,    56,    81,   288,    51,     1,
      35,    36,    55,    56,    55,    56,    81,   323,    68,    76,
      77,    55,    56,    78,     1,    82,    61,    62,    63,    64,
      55,    85,    85,    77,    55,   337,    61,    62,    63,    64,
      65,    66,    67,    58,    69,    70,    77,    83,    40,    41,
      42,    43,    44,    45,    46,    47,   362,    49,    74,    81,
      86,    86,    81,    40,    41,    42,    43,    44,    45,    46,
      47,    78,    49,   379,    86,   369,   347,    79,   369,     1,
     446,   363,   364,   365,    76,    77,   452,   358,   454,    75,
      11,    12,    13,    14,    15,    16,    86,   463,   369,    76,
      77,     1,    78,    86,    51,    86,    78,    76,    78,    83,
     476,    78,    80,    77,    80,    78,   482,    82,    40,    41,
      42,    43,    44,    45,    46,    47,    86,    49,   494,    85,
      17,   412,    86,    83,    87,   406,    87,    86,    83,    80,
      40,    41,    42,    43,    44,    45,    46,    47,    83,    49,
      80,   517,    83,   519,    76,    78,    82,    77,    82,    39,
      77,    77,   433,    82,    78,    83,    74,    57,   534,    35,
      85,   460,    83,    86,    83,    82,    76,   543,    83,    83,
     546,    78,    74,    86,    86,   551,   492,     1,     1,    82,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    14,    15,    16,   493,    18,    19,    86,    85,    22,
      55,     1,    76,    57,    55,   581,    20,    82,    55,    78,
      76,    78,    74,    55,   530,   531,    40,    41,    42,    43,
      44,    45,    46,    47,    77,    49,    87,    85,    74,   528,
      53,    54,   513,    83,    86,    56,    59,    60,    77,    77,
      40,    41,    42,    43,    44,    45,    46,    47,    71,    49,
      86,    83,    76,    76,    77,    75,    78,     1,    86,    74,
      50,    48,    79,   544,   545,   246,   247,   248,   549,    75,
      77,    82,     1,    77,    79,    55,    76,     6,     7,     8,
       9,   562,    11,    12,    13,    14,    15,    86,    77,    83,
      78,   572,   573,    78,   575,   576,    40,    41,    42,    43,
      44,    45,    46,    47,    16,    49,    77,   588,    80,    86,
      39,    40,    41,    42,    43,    44,    45,    46,    47,     1,
      49,   302,   303,    35,    36,    77,    77,     5,    22,     1,
     104,     8,    76,   232,   315,   316,   317,   318,   319,   125,
     138,   242,   251,    55,   377,    74,   262,    76,    77,    61,
      62,    63,    64,    65,    66,    67,   261,    22,    40,    41,
      42,    43,    44,    45,    46,    47,   337,    49,    40,    41,
      42,    43,    44,    45,    46,    47,   256,    49,    22,   176,
       3,     4,   424,     6,     7,     8,     9,    10,    11,    12,
      13,    14,    15,    16,    76,    18,    19,     1,   342,    22,
     343,   339,   369,   275,    76,   530,   554,     6,     7,     8,
       9,     1,    11,    12,    13,    14,    15,    21,   531,    23,
      24,    25,    26,    27,    28,   582,   469,    31,    32,    33,
      34,    21,    -1,    23,    24,    25,    26,    27,    28,    -1,
      39,    31,    32,    33,    34,   483,    -1,    51,    52,   477,
       6,    -1,    -1,    76,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    51,    52,    -1,    68,    -1,    -1,    -1,    72,    73,
      -1,    -1,    -1,    -1,    -1,    74,    -1,    -1,    77,    35,
      36,    -1,    72,    73,     6,     7,     8,     9,    -1,    11,
      12,    13,    14,    15,    -1,    -1,    -1,    -1,    18,    55,
      35,    36,    -1,    -1,    -1,    61,    62,    63,    64,    65,
      66,    67,    -1,    69,    70,    35,    36,    39,    -1,    -1,
      55,     6,     7,     8,     9,    10,    61,    62,    63,    64,
      65,    66,    67,    18,    19,    55,    -1,    22,    35,    36,
      -1,    61,    62,    63,    64,    65,    66,    67,    83,    -1,
      -1,    -1,    74,    -1,    39,    77,    35,    36,    55,    56,
      57,    -1,    -1,    -1,    61,    62,    63,    64,    65,    66,
      67,    35,    -1,    -1,    -1,    -1,    55,    -1,    -1,    -1,
      -1,    -1,    61,    62,    63,    64,    65,    66,    67,    35,
      -1,    55,    -1,    -1,    -1,    -1,    -1,    61,    62,    63,
      64,    65,    66,    67,    35,    -1,    -1,    -1,    -1,    55,
      -1,    -1,    -1,    -1,    -1,    61,    62,    63,    64,    65,
      66,    67,    -1,    -1,    55,    -1,    -1,    -1,    -1,    -1,
      61,    62,    63,    64,    65,    66,    67
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     3,     4,    91,    92,    97,    55,    95,    95,     0,
      92,    74,    76,    98,    98,     1,     5,    53,    54,    59,
      60,    71,    93,    99,   100,   101,   165,   201,   202,    39,
      95,    53,    55,    96,    95,    95,    74,    76,   174,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      18,    19,    22,    76,    97,   118,   119,   135,   138,   139,
     140,   142,   152,   153,   156,   157,   158,    77,     6,     7,
       8,     9,    11,    12,    13,    14,    15,    39,    74,    77,
     163,    99,    35,    61,    62,    63,    64,    65,    66,    67,
      96,   106,   108,   109,   110,   166,    76,    96,    75,    58,
      82,   172,    16,    35,    36,   109,   110,   111,   112,   113,
     114,    35,   120,   120,    85,   120,   108,   159,    85,   124,
     124,   124,   124,    85,   128,   141,    85,   121,    95,    57,
     160,    79,    99,    11,    12,    13,    14,    15,    16,   143,
     144,   145,   146,   147,    74,    94,    62,    66,    61,    62,
      63,    64,    79,   105,    81,   108,    99,    75,    76,    82,
      85,   172,    77,   175,   109,   113,    81,    81,    36,    82,
      84,    96,    96,    96,    68,    96,    78,    30,    51,   125,
     130,    95,   107,   107,   107,   107,    51,    56,    95,   127,
     129,    85,   141,    85,   128,    37,    38,   122,   123,   107,
      18,   112,   114,   150,   151,    77,   124,   124,   124,   124,
     141,   121,    61,    62,    56,    57,   102,   103,   104,   114,
      95,    77,    55,   172,    58,   171,   172,    83,    74,    81,
      81,    85,   116,   117,    82,    79,    82,    86,    79,    82,
     159,    86,    78,   105,    75,   136,   136,   136,   136,    95,
      86,    78,    86,   107,   107,    86,    78,    76,    95,    87,
     149,    95,    78,    80,    94,    95,    95,    95,    95,    95,
      95,    78,    80,   105,    77,    78,    83,    86,   172,    95,
      95,    56,    96,   115,   117,   120,   103,   120,   120,   103,
     120,   125,   108,   137,    74,    76,   154,   154,   154,   154,
      86,   129,   136,   136,   122,    17,   131,   133,   134,    87,
     148,    56,    57,   149,   151,   136,   136,   136,   136,   136,
      74,    76,   103,    82,   183,   172,   171,   172,   172,    83,
      86,    83,    80,    83,    96,    80,    83,    78,    39,   152,
     155,   156,   161,   162,   164,   154,   154,   114,   134,    77,
     114,   154,   154,   154,   154,   154,   134,    83,   114,   173,
     176,   181,    82,    82,    82,    82,   137,     1,    85,   167,
     164,    77,   155,    74,   163,    95,    77,    95,   172,    78,
      83,   181,   120,   120,   120,     1,    21,    23,    24,    25,
      26,    27,    28,    31,    32,    33,    34,    51,    52,    68,
      72,    73,   168,   169,    55,    95,   166,    94,    85,   132,
      74,    85,    87,   131,    86,   181,    83,    83,    83,    83,
      57,   126,    86,    86,    78,   183,    95,    86,    74,    56,
      57,    96,   170,    35,   168,     1,    40,    41,    42,    43,
      44,    45,    46,    47,    49,    76,   174,   186,   192,   183,
      95,    82,   196,    85,   196,    55,   197,   198,    76,    57,
     191,    55,   194,   196,    82,   187,   192,   172,    20,   185,
     183,   172,    55,   172,    85,   183,   199,    78,    76,   192,
     188,   192,   174,    78,    74,   172,    55,    77,   187,   175,
      87,   186,    85,   184,    74,    86,    83,   195,   172,   198,
      77,   187,    77,   187,   172,   194,   195,   183,    56,     6,
      69,    70,    86,   114,   177,   180,   182,   174,   172,   196,
      76,   192,    86,   200,    77,   175,    76,   192,    83,    95,
      75,    78,    86,   172,    74,   172,   187,   183,    50,   189,
     187,    48,   193,   174,    79,   114,   173,   179,   182,   175,
     172,    75,    77,    82,    77,    76,   192,   172,    95,   178,
      95,   172,    79,    95,   195,   172,    55,   190,   193,   187,
      77,    80,    82,    85,    88,    89,    79,    86,   178,    76,
     192,    78,    78,    83,    77,   178,    56,   178,    80,    95,
     178,    80,   187,   172,   190,    83,    86,    86,    95,    80,
      77,   195,    76,   192,   187,    77
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
   Once GCC version 2 has supplanted version 1, this can go.  */

#define YYFAIL		goto yyerrlab

#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)					\
do								\
  if (yychar == YYEMPTY && yylen == 1)				\
    {								\
      yychar = (Token);						\
      yylval = (Value);						\
      yytoken = YYTRANSLATE (yychar);				\
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
# if YYLTYPE_IS_TRIVIAL
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

/* Copy into YYRESULT an error message about the unexpected token
   YYCHAR while in state YYSTATE.  Return the number of bytes copied,
   including the terminating null byte.  If YYRESULT is null, do not
   copy anything; just return the number of bytes that would be
   copied.  As a special case, return 0 if an ordinary "syntax error"
   message will do.  Return YYSIZE_MAXIMUM if overflow occurs during
   size calculation.  */
static YYSIZE_T
yysyntax_error (char *yyresult, int yystate, int yychar)
{
  int yyn = yypact[yystate];

  if (! (YYPACT_NINF < yyn && yyn <= YYLAST))
    return 0;
  else
    {
      int yytype = YYTRANSLATE (yychar);
      YYSIZE_T yysize0 = yytnamerr (0, yytname[yytype]);
      YYSIZE_T yysize = yysize0;
      YYSIZE_T yysize1;
      int yysize_overflow = 0;
      enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
      char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
      int yyx;

# if 0
      /* This is so xgettext sees the translatable formats that are
	 constructed on the fly.  */
      YY_("syntax error, unexpected %s");
      YY_("syntax error, unexpected %s, expecting %s");
      YY_("syntax error, unexpected %s, expecting %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s");
# endif
      char *yyfmt;
      char const *yyf;
      static char const yyunexpected[] = "syntax error, unexpected %s";
      static char const yyexpecting[] = ", expecting %s";
      static char const yyor[] = " or %s";
      char yyformat[sizeof yyunexpected
		    + sizeof yyexpecting - 1
		    + ((YYERROR_VERBOSE_ARGS_MAXIMUM - 2)
		       * (sizeof yyor - 1))];
      char const *yyprefix = yyexpecting;

      /* Start YYX at -YYN if negative to avoid negative indexes in
	 YYCHECK.  */
      int yyxbegin = yyn < 0 ? -yyn : 0;

      /* Stay within bounds of both yycheck and yytname.  */
      int yychecklim = YYLAST - yyn + 1;
      int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
      int yycount = 1;

      yyarg[0] = yytname[yytype];
      yyfmt = yystpcpy (yyformat, yyunexpected);

      for (yyx = yyxbegin; yyx < yyxend; ++yyx)
	if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR)
	  {
	    if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
	      {
		yycount = 1;
		yysize = yysize0;
		yyformat[sizeof yyunexpected - 1] = '\0';
		break;
	      }
	    yyarg[yycount++] = yytname[yyx];
	    yysize1 = yysize + yytnamerr (0, yytname[yyx]);
	    yysize_overflow |= (yysize1 < yysize);
	    yysize = yysize1;
	    yyfmt = yystpcpy (yyfmt, yyprefix);
	    yyprefix = yyor;
	  }

      yyf = YY_(yyformat);
      yysize1 = yysize + yystrlen (yyf);
      yysize_overflow |= (yysize1 < yysize);
      yysize = yysize1;

      if (yysize_overflow)
	return YYSIZE_MAXIMUM;

      if (yyresult)
	{
	  /* Avoid sprintf, as that infringes on the user's name space.
	     Don't have undefined behavior even if the translation
	     produced a string with the wrong number of "%s"s.  */
	  char *yyp = yyresult;
	  int yyi = 0;
	  while ((*yyp = *yyf) != '\0')
	    {
	      if (*yyp == '%' && yyf[1] == 's' && yyi < yycount)
		{
		  yyp += yytnamerr (yyp, yyarg[yyi++]);
		  yyf += 2;
		}
	      else
		{
		  yyp++;
		  yyf++;
		}
	    }
	}
      return yysize;
    }
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



/*-------------------------.
| yyparse or yypush_parse.  |
`-------------------------*/

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
  if (yyn == YYPACT_NINF)
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
      if (yyn == 0 || yyn == YYTABLE_NINF)
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

/* Line 1455 of yacc.c  */
#line 155 "xi-grammar.y"
    { (yyval.modlist) = (yyvsp[(1) - (1)].modlist); modlist = (yyvsp[(1) - (1)].modlist); }
    break;

  case 3:

/* Line 1455 of yacc.c  */
#line 159 "xi-grammar.y"
    { 
		  (yyval.modlist) = 0; 
		}
    break;

  case 4:

/* Line 1455 of yacc.c  */
#line 163 "xi-grammar.y"
    { (yyval.modlist) = new ModuleList(lineno, (yyvsp[(1) - (2)].module), (yyvsp[(2) - (2)].modlist)); }
    break;

  case 5:

/* Line 1455 of yacc.c  */
#line 167 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 6:

/* Line 1455 of yacc.c  */
#line 169 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 7:

/* Line 1455 of yacc.c  */
#line 173 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 8:

/* Line 1455 of yacc.c  */
#line 175 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 9:

/* Line 1455 of yacc.c  */
#line 179 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 10:

/* Line 1455 of yacc.c  */
#line 183 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 11:

/* Line 1455 of yacc.c  */
#line 185 "xi-grammar.y"
    {
		  char *tmp = new char[strlen((yyvsp[(1) - (4)].strval))+strlen((yyvsp[(4) - (4)].strval))+3];
		  sprintf(tmp,"%s::%s", (yyvsp[(1) - (4)].strval), (yyvsp[(4) - (4)].strval));
		  (yyval.strval) = tmp;
		}
    break;

  case 12:

/* Line 1455 of yacc.c  */
#line 193 "xi-grammar.y"
    { 
		    (yyval.module) = new Module(lineno, (yyvsp[(2) - (3)].strval), (yyvsp[(3) - (3)].conslist)); 
		}
    break;

  case 13:

/* Line 1455 of yacc.c  */
#line 197 "xi-grammar.y"
    {  
		    (yyval.module) = new Module(lineno, (yyvsp[(2) - (3)].strval), (yyvsp[(3) - (3)].conslist)); 
		    (yyval.module)->setMain();
		}
    break;

  case 14:

/* Line 1455 of yacc.c  */
#line 204 "xi-grammar.y"
    { (yyval.conslist) = 0; }
    break;

  case 15:

/* Line 1455 of yacc.c  */
#line 206 "xi-grammar.y"
    { (yyval.conslist) = (yyvsp[(2) - (4)].conslist); }
    break;

  case 16:

/* Line 1455 of yacc.c  */
#line 210 "xi-grammar.y"
    { (yyval.conslist) = 0; }
    break;

  case 17:

/* Line 1455 of yacc.c  */
#line 212 "xi-grammar.y"
    { (yyval.conslist) = new ConstructList(lineno, (yyvsp[(1) - (2)].construct), (yyvsp[(2) - (2)].conslist)); }
    break;

  case 18:

/* Line 1455 of yacc.c  */
#line 216 "xi-grammar.y"
    { (yyval.construct) = new UsingScope((yyvsp[(3) - (3)].strval), false); }
    break;

  case 19:

/* Line 1455 of yacc.c  */
#line 218 "xi-grammar.y"
    { (yyval.construct) = new UsingScope((yyvsp[(2) - (2)].strval), true); }
    break;

  case 20:

/* Line 1455 of yacc.c  */
#line 220 "xi-grammar.y"
    { (yyvsp[(2) - (2)].member)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].member); }
    break;

  case 21:

/* Line 1455 of yacc.c  */
#line 222 "xi-grammar.y"
    { (yyvsp[(2) - (2)].message)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].message); }
    break;

  case 22:

/* Line 1455 of yacc.c  */
#line 224 "xi-grammar.y"
    {
                  Entry *e = new Entry(lineno, 0, (yyvsp[(3) - (7)].type), (yyvsp[(5) - (7)].strval), (yyvsp[(7) - (7)].plist), 0, 0, 0, 0, 0);
                  int isExtern = 1;
                  e->setExtern(isExtern);
                  e->targs = (yyvsp[(6) - (7)].tparlist);
                  e->label = new XStr;
                  (yyvsp[(4) - (7)].ntype)->print(*e->label);
                  (yyval.construct) = e;
                }
    break;

  case 23:

/* Line 1455 of yacc.c  */
#line 236 "xi-grammar.y"
    { if((yyvsp[(3) - (5)].conslist)) (yyvsp[(3) - (5)].conslist)->setExtern((yyvsp[(1) - (5)].intval)); (yyval.construct) = (yyvsp[(3) - (5)].conslist); }
    break;

  case 24:

/* Line 1455 of yacc.c  */
#line 238 "xi-grammar.y"
    { (yyval.construct) = new Scope((yyvsp[(2) - (5)].strval), (yyvsp[(4) - (5)].conslist)); }
    break;

  case 25:

/* Line 1455 of yacc.c  */
#line 240 "xi-grammar.y"
    { (yyval.construct) = (yyvsp[(1) - (2)].construct); }
    break;

  case 26:

/* Line 1455 of yacc.c  */
#line 242 "xi-grammar.y"
    { yyerror("The preceding construct must be semicolon terminated"); YYABORT; }
    break;

  case 27:

/* Line 1455 of yacc.c  */
#line 244 "xi-grammar.y"
    { (yyvsp[(2) - (2)].module)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].module); }
    break;

  case 28:

/* Line 1455 of yacc.c  */
#line 246 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 29:

/* Line 1455 of yacc.c  */
#line 248 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 30:

/* Line 1455 of yacc.c  */
#line 250 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 31:

/* Line 1455 of yacc.c  */
#line 252 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 32:

/* Line 1455 of yacc.c  */
#line 254 "xi-grammar.y"
    { (yyvsp[(2) - (2)].templat)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].templat); }
    break;

  case 33:

/* Line 1455 of yacc.c  */
#line 256 "xi-grammar.y"
    { (yyval.construct) = NULL; }
    break;

  case 34:

/* Line 1455 of yacc.c  */
#line 258 "xi-grammar.y"
    { (yyval.construct) = NULL; }
    break;

  case 35:

/* Line 1455 of yacc.c  */
#line 260 "xi-grammar.y"
    { (yyval.construct) = (yyvsp[(1) - (1)].accelBlock); }
    break;

  case 36:

/* Line 1455 of yacc.c  */
#line 262 "xi-grammar.y"
    { printf("Invalid construct\n"); YYABORT; }
    break;

  case 37:

/* Line 1455 of yacc.c  */
#line 266 "xi-grammar.y"
    { (yyval.tparam) = new TParamType((yyvsp[(1) - (1)].type)); }
    break;

  case 38:

/* Line 1455 of yacc.c  */
#line 268 "xi-grammar.y"
    { (yyval.tparam) = new TParamVal((yyvsp[(1) - (1)].strval)); }
    break;

  case 39:

/* Line 1455 of yacc.c  */
#line 270 "xi-grammar.y"
    { (yyval.tparam) = new TParamVal((yyvsp[(1) - (1)].strval)); }
    break;

  case 40:

/* Line 1455 of yacc.c  */
#line 274 "xi-grammar.y"
    { (yyval.tparlist) = new TParamList((yyvsp[(1) - (1)].tparam)); }
    break;

  case 41:

/* Line 1455 of yacc.c  */
#line 276 "xi-grammar.y"
    { (yyval.tparlist) = new TParamList((yyvsp[(1) - (3)].tparam), (yyvsp[(3) - (3)].tparlist)); }
    break;

  case 42:

/* Line 1455 of yacc.c  */
#line 280 "xi-grammar.y"
    { (yyval.tparlist) = 0; }
    break;

  case 43:

/* Line 1455 of yacc.c  */
#line 282 "xi-grammar.y"
    { (yyval.tparlist) = (yyvsp[(1) - (1)].tparlist); }
    break;

  case 44:

/* Line 1455 of yacc.c  */
#line 286 "xi-grammar.y"
    { (yyval.tparlist) = 0; }
    break;

  case 45:

/* Line 1455 of yacc.c  */
#line 288 "xi-grammar.y"
    { (yyval.tparlist) = (yyvsp[(2) - (3)].tparlist); }
    break;

  case 46:

/* Line 1455 of yacc.c  */
#line 292 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("int"); }
    break;

  case 47:

/* Line 1455 of yacc.c  */
#line 294 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long"); }
    break;

  case 48:

/* Line 1455 of yacc.c  */
#line 296 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("short"); }
    break;

  case 49:

/* Line 1455 of yacc.c  */
#line 298 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("char"); }
    break;

  case 50:

/* Line 1455 of yacc.c  */
#line 300 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned int"); }
    break;

  case 51:

/* Line 1455 of yacc.c  */
#line 302 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long"); }
    break;

  case 52:

/* Line 1455 of yacc.c  */
#line 304 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long"); }
    break;

  case 53:

/* Line 1455 of yacc.c  */
#line 306 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long long"); }
    break;

  case 54:

/* Line 1455 of yacc.c  */
#line 308 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned short"); }
    break;

  case 55:

/* Line 1455 of yacc.c  */
#line 310 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned char"); }
    break;

  case 56:

/* Line 1455 of yacc.c  */
#line 312 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long long"); }
    break;

  case 57:

/* Line 1455 of yacc.c  */
#line 314 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("float"); }
    break;

  case 58:

/* Line 1455 of yacc.c  */
#line 316 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("double"); }
    break;

  case 59:

/* Line 1455 of yacc.c  */
#line 318 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long double"); }
    break;

  case 60:

/* Line 1455 of yacc.c  */
#line 320 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("void"); }
    break;

  case 61:

/* Line 1455 of yacc.c  */
#line 323 "xi-grammar.y"
    { (yyval.ntype) = new NamedType((yyvsp[(1) - (2)].strval),(yyvsp[(2) - (2)].tparlist)); }
    break;

  case 62:

/* Line 1455 of yacc.c  */
#line 324 "xi-grammar.y"
    { 
                    const char* basename, *scope;
                    splitScopedName((yyvsp[(1) - (2)].strval), &scope, &basename);
                    (yyval.ntype) = new NamedType(basename, (yyvsp[(2) - (2)].tparlist), scope);
                }
    break;

  case 63:

/* Line 1455 of yacc.c  */
#line 332 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 64:

/* Line 1455 of yacc.c  */
#line 334 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ntype); }
    break;

  case 65:

/* Line 1455 of yacc.c  */
#line 338 "xi-grammar.y"
    { (yyval.ptype) = new PtrType((yyvsp[(1) - (2)].type)); }
    break;

  case 66:

/* Line 1455 of yacc.c  */
#line 342 "xi-grammar.y"
    { (yyvsp[(1) - (2)].ptype)->indirect(); (yyval.ptype) = (yyvsp[(1) - (2)].ptype); }
    break;

  case 67:

/* Line 1455 of yacc.c  */
#line 344 "xi-grammar.y"
    { (yyvsp[(1) - (2)].ptype)->indirect(); (yyval.ptype) = (yyvsp[(1) - (2)].ptype); }
    break;

  case 68:

/* Line 1455 of yacc.c  */
#line 348 "xi-grammar.y"
    { (yyval.ftype) = new FuncType((yyvsp[(1) - (8)].type), (yyvsp[(4) - (8)].strval), (yyvsp[(7) - (8)].plist)); }
    break;

  case 69:

/* Line 1455 of yacc.c  */
#line 352 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 70:

/* Line 1455 of yacc.c  */
#line 354 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 71:

/* Line 1455 of yacc.c  */
#line 356 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 72:

/* Line 1455 of yacc.c  */
#line 358 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ftype); }
    break;

  case 73:

/* Line 1455 of yacc.c  */
#line 361 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(2) - (2)].type); }
    break;

  case 74:

/* Line 1455 of yacc.c  */
#line 363 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (2)].type); }
    break;

  case 75:

/* Line 1455 of yacc.c  */
#line 367 "xi-grammar.y"
    { (yyval.type) = new ReferenceType((yyvsp[(1) - (2)].type)); }
    break;

  case 76:

/* Line 1455 of yacc.c  */
#line 369 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 77:

/* Line 1455 of yacc.c  */
#line 373 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 78:

/* Line 1455 of yacc.c  */
#line 375 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 79:

/* Line 1455 of yacc.c  */
#line 379 "xi-grammar.y"
    { (yyval.val) = (yyvsp[(2) - (3)].val); }
    break;

  case 80:

/* Line 1455 of yacc.c  */
#line 383 "xi-grammar.y"
    { (yyval.vallist) = 0; }
    break;

  case 81:

/* Line 1455 of yacc.c  */
#line 385 "xi-grammar.y"
    { (yyval.vallist) = new ValueList((yyvsp[(1) - (2)].val), (yyvsp[(2) - (2)].vallist)); }
    break;

  case 82:

/* Line 1455 of yacc.c  */
#line 389 "xi-grammar.y"
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[(2) - (4)].type), (yyvsp[(3) - (4)].strval), (yyvsp[(4) - (4)].vallist)); }
    break;

  case 83:

/* Line 1455 of yacc.c  */
#line 393 "xi-grammar.y"
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[(3) - (5)].type), (yyvsp[(5) - (5)].strval), 0, 1); }
    break;

  case 84:

/* Line 1455 of yacc.c  */
#line 397 "xi-grammar.y"
    { (yyval.intval) = 0;}
    break;

  case 85:

/* Line 1455 of yacc.c  */
#line 399 "xi-grammar.y"
    { (yyval.intval) = 0;}
    break;

  case 86:

/* Line 1455 of yacc.c  */
#line 403 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 87:

/* Line 1455 of yacc.c  */
#line 405 "xi-grammar.y"
    { 
		  /*
		  printf("Warning: Message attributes are being phased out.\n");
		  printf("Warning: Please remove them from interface files.\n");
		  */
		  (yyval.intval) = (yyvsp[(2) - (3)].intval); 
		}
    break;

  case 88:

/* Line 1455 of yacc.c  */
#line 415 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (1)].intval); }
    break;

  case 89:

/* Line 1455 of yacc.c  */
#line 417 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (3)].intval) | (yyvsp[(3) - (3)].intval); }
    break;

  case 90:

/* Line 1455 of yacc.c  */
#line 421 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 91:

/* Line 1455 of yacc.c  */
#line 423 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 92:

/* Line 1455 of yacc.c  */
#line 427 "xi-grammar.y"
    { (yyval.cattr) = 0; }
    break;

  case 93:

/* Line 1455 of yacc.c  */
#line 429 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(2) - (3)].cattr); }
    break;

  case 94:

/* Line 1455 of yacc.c  */
#line 433 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (1)].cattr); }
    break;

  case 95:

/* Line 1455 of yacc.c  */
#line 435 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (3)].cattr) | (yyvsp[(3) - (3)].cattr); }
    break;

  case 96:

/* Line 1455 of yacc.c  */
#line 439 "xi-grammar.y"
    { python_doc = NULL; (yyval.intval) = 0; }
    break;

  case 97:

/* Line 1455 of yacc.c  */
#line 441 "xi-grammar.y"
    { python_doc = (yyvsp[(1) - (1)].strval); (yyval.intval) = 0; }
    break;

  case 98:

/* Line 1455 of yacc.c  */
#line 445 "xi-grammar.y"
    { (yyval.cattr) = Chare::CPYTHON; }
    break;

  case 99:

/* Line 1455 of yacc.c  */
#line 449 "xi-grammar.y"
    { (yyval.cattr) = 0; }
    break;

  case 100:

/* Line 1455 of yacc.c  */
#line 451 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(2) - (3)].cattr); }
    break;

  case 101:

/* Line 1455 of yacc.c  */
#line 455 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (1)].cattr); }
    break;

  case 102:

/* Line 1455 of yacc.c  */
#line 457 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (3)].cattr) | (yyvsp[(3) - (3)].cattr); }
    break;

  case 103:

/* Line 1455 of yacc.c  */
#line 461 "xi-grammar.y"
    { (yyval.cattr) = Chare::CMIGRATABLE; }
    break;

  case 104:

/* Line 1455 of yacc.c  */
#line 463 "xi-grammar.y"
    { (yyval.cattr) = Chare::CPYTHON; }
    break;

  case 105:

/* Line 1455 of yacc.c  */
#line 467 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 106:

/* Line 1455 of yacc.c  */
#line 469 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 107:

/* Line 1455 of yacc.c  */
#line 472 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 108:

/* Line 1455 of yacc.c  */
#line 474 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 109:

/* Line 1455 of yacc.c  */
#line 477 "xi-grammar.y"
    { (yyval.mv) = new MsgVar((yyvsp[(2) - (5)].type), (yyvsp[(3) - (5)].strval), (yyvsp[(1) - (5)].intval), (yyvsp[(4) - (5)].intval)); }
    break;

  case 110:

/* Line 1455 of yacc.c  */
#line 481 "xi-grammar.y"
    { (yyval.mvlist) = new MsgVarList((yyvsp[(1) - (1)].mv)); }
    break;

  case 111:

/* Line 1455 of yacc.c  */
#line 483 "xi-grammar.y"
    { (yyval.mvlist) = new MsgVarList((yyvsp[(1) - (2)].mv), (yyvsp[(2) - (2)].mvlist)); }
    break;

  case 112:

/* Line 1455 of yacc.c  */
#line 487 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, (yyvsp[(3) - (3)].ntype)); }
    break;

  case 113:

/* Line 1455 of yacc.c  */
#line 489 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, (yyvsp[(3) - (6)].ntype), (yyvsp[(5) - (6)].mvlist)); }
    break;

  case 114:

/* Line 1455 of yacc.c  */
#line 493 "xi-grammar.y"
    { (yyval.typelist) = 0; }
    break;

  case 115:

/* Line 1455 of yacc.c  */
#line 495 "xi-grammar.y"
    { (yyval.typelist) = (yyvsp[(2) - (2)].typelist); }
    break;

  case 116:

/* Line 1455 of yacc.c  */
#line 499 "xi-grammar.y"
    { (yyval.typelist) = new TypeList((yyvsp[(1) - (1)].ntype)); }
    break;

  case 117:

/* Line 1455 of yacc.c  */
#line 501 "xi-grammar.y"
    { (yyval.typelist) = new TypeList((yyvsp[(1) - (3)].ntype), (yyvsp[(3) - (3)].typelist)); }
    break;

  case 118:

/* Line 1455 of yacc.c  */
#line 505 "xi-grammar.y"
    { (yyval.chare) = new Chare(lineno, (yyvsp[(2) - (5)].cattr)|Chare::CCHARE, (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 119:

/* Line 1455 of yacc.c  */
#line 507 "xi-grammar.y"
    { (yyval.chare) = new MainChare(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 120:

/* Line 1455 of yacc.c  */
#line 511 "xi-grammar.y"
    { (yyval.chare) = new Group(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 121:

/* Line 1455 of yacc.c  */
#line 515 "xi-grammar.y"
    { (yyval.chare) = new NodeGroup(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 122:

/* Line 1455 of yacc.c  */
#line 519 "xi-grammar.y"
    {/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",(yyvsp[(2) - (4)].strval));
			(yyval.ntype) = new NamedType(buf); 
		}
    break;

  case 123:

/* Line 1455 of yacc.c  */
#line 525 "xi-grammar.y"
    { (yyval.ntype) = new NamedType((yyvsp[(2) - (3)].strval)); }
    break;

  case 124:

/* Line 1455 of yacc.c  */
#line 529 "xi-grammar.y"
    {  (yyval.chare) = new Array(lineno, (yyvsp[(2) - (6)].cattr), (yyvsp[(3) - (6)].ntype), (yyvsp[(4) - (6)].ntype), (yyvsp[(5) - (6)].typelist), (yyvsp[(6) - (6)].mbrlist)); }
    break;

  case 125:

/* Line 1455 of yacc.c  */
#line 531 "xi-grammar.y"
    {  (yyval.chare) = new Array(lineno, (yyvsp[(3) - (6)].cattr), (yyvsp[(2) - (6)].ntype), (yyvsp[(4) - (6)].ntype), (yyvsp[(5) - (6)].typelist), (yyvsp[(6) - (6)].mbrlist)); }
    break;

  case 126:

/* Line 1455 of yacc.c  */
#line 535 "xi-grammar.y"
    { (yyval.chare) = new Chare(lineno, (yyvsp[(2) - (5)].cattr)|Chare::CCHARE, new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist));}
    break;

  case 127:

/* Line 1455 of yacc.c  */
#line 537 "xi-grammar.y"
    { (yyval.chare) = new MainChare(lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 128:

/* Line 1455 of yacc.c  */
#line 541 "xi-grammar.y"
    { (yyval.chare) = new Group(lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 129:

/* Line 1455 of yacc.c  */
#line 545 "xi-grammar.y"
    { (yyval.chare) = new NodeGroup( lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 130:

/* Line 1455 of yacc.c  */
#line 549 "xi-grammar.y"
    { (yyval.chare) = new Array( lineno, 0, (yyvsp[(2) - (5)].ntype), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 131:

/* Line 1455 of yacc.c  */
#line 553 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[(3) - (4)].strval))); }
    break;

  case 132:

/* Line 1455 of yacc.c  */
#line 555 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[(3) - (7)].strval)), (yyvsp[(5) - (7)].mvlist)); }
    break;

  case 133:

/* Line 1455 of yacc.c  */
#line 559 "xi-grammar.y"
    { (yyval.type) = 0; }
    break;

  case 134:

/* Line 1455 of yacc.c  */
#line 561 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(2) - (2)].type); }
    break;

  case 135:

/* Line 1455 of yacc.c  */
#line 565 "xi-grammar.y"
    { (yyval.strval) = 0; }
    break;

  case 136:

/* Line 1455 of yacc.c  */
#line 567 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(2) - (2)].strval); }
    break;

  case 137:

/* Line 1455 of yacc.c  */
#line 569 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(2) - (2)].strval); }
    break;

  case 138:

/* Line 1455 of yacc.c  */
#line 573 "xi-grammar.y"
    { (yyval.tvar) = new TType(new NamedType((yyvsp[(2) - (3)].strval)), (yyvsp[(3) - (3)].type)); }
    break;

  case 139:

/* Line 1455 of yacc.c  */
#line 575 "xi-grammar.y"
    { (yyval.tvar) = new TFunc((yyvsp[(1) - (2)].ftype), (yyvsp[(2) - (2)].strval)); }
    break;

  case 140:

/* Line 1455 of yacc.c  */
#line 577 "xi-grammar.y"
    { (yyval.tvar) = new TName((yyvsp[(1) - (3)].type), (yyvsp[(2) - (3)].strval), (yyvsp[(3) - (3)].strval)); }
    break;

  case 141:

/* Line 1455 of yacc.c  */
#line 581 "xi-grammar.y"
    { (yyval.tvarlist) = new TVarList((yyvsp[(1) - (1)].tvar)); }
    break;

  case 142:

/* Line 1455 of yacc.c  */
#line 583 "xi-grammar.y"
    { (yyval.tvarlist) = new TVarList((yyvsp[(1) - (3)].tvar), (yyvsp[(3) - (3)].tvarlist)); }
    break;

  case 143:

/* Line 1455 of yacc.c  */
#line 587 "xi-grammar.y"
    { (yyval.tvarlist) = (yyvsp[(3) - (4)].tvarlist); }
    break;

  case 144:

/* Line 1455 of yacc.c  */
#line 591 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 145:

/* Line 1455 of yacc.c  */
#line 593 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 146:

/* Line 1455 of yacc.c  */
#line 595 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 147:

/* Line 1455 of yacc.c  */
#line 597 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 148:

/* Line 1455 of yacc.c  */
#line 599 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].message)); (yyvsp[(2) - (2)].message)->setTemplate((yyval.templat)); }
    break;

  case 149:

/* Line 1455 of yacc.c  */
#line 603 "xi-grammar.y"
    { (yyval.mbrlist) = 0; }
    break;

  case 150:

/* Line 1455 of yacc.c  */
#line 605 "xi-grammar.y"
    { (yyval.mbrlist) = (yyvsp[(2) - (4)].mbrlist); }
    break;

  case 151:

/* Line 1455 of yacc.c  */
#line 609 "xi-grammar.y"
    { 
                  if (!connectEntries.empty()) {
                    (yyval.mbrlist) = new MemberList(connectEntries);
		  } else {
		    (yyval.mbrlist) = 0; 
                  }
		}
    break;

  case 152:

/* Line 1455 of yacc.c  */
#line 617 "xi-grammar.y"
    { (yyval.mbrlist) = new MemberList((yyvsp[(1) - (2)].member), (yyvsp[(2) - (2)].mbrlist)); }
    break;

  case 153:

/* Line 1455 of yacc.c  */
#line 621 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].readonly); }
    break;

  case 154:

/* Line 1455 of yacc.c  */
#line 623 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].readonly); }
    break;

  case 156:

/* Line 1455 of yacc.c  */
#line 626 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].member); }
    break;

  case 157:

/* Line 1455 of yacc.c  */
#line 628 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(2) - (2)].pupable); }
    break;

  case 158:

/* Line 1455 of yacc.c  */
#line 630 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(2) - (2)].includeFile); }
    break;

  case 159:

/* Line 1455 of yacc.c  */
#line 632 "xi-grammar.y"
    { (yyval.member) = new ClassDeclaration(lineno,(yyvsp[(2) - (2)].strval)); }
    break;

  case 160:

/* Line 1455 of yacc.c  */
#line 636 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 1); }
    break;

  case 161:

/* Line 1455 of yacc.c  */
#line 638 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 1); }
    break;

  case 162:

/* Line 1455 of yacc.c  */
#line 640 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[(3) - (9)].strval)) + '<' +
					    ((yyvsp[(5) - (9)].tparlist))->to_string() + '>').c_str()),
				    1);
		}
    break;

  case 163:

/* Line 1455 of yacc.c  */
#line 646 "xi-grammar.y"
    { printf("Warning: deprecated use of initcall. Use initnode or initproc instead.\n"); 
		  (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 1); }
    break;

  case 164:

/* Line 1455 of yacc.c  */
#line 649 "xi-grammar.y"
    { printf("Warning: deprecated use of initcall. Use initnode or initproc instead.\n");
		  (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 1); }
    break;

  case 165:

/* Line 1455 of yacc.c  */
#line 654 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 0); }
    break;

  case 166:

/* Line 1455 of yacc.c  */
#line 656 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 0); }
    break;

  case 167:

/* Line 1455 of yacc.c  */
#line 658 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[(3) - (9)].strval)) + '<' +
					    ((yyvsp[(5) - (9)].tparlist))->to_string() + '>').c_str()),
				    0);
		}
    break;

  case 168:

/* Line 1455 of yacc.c  */
#line 664 "xi-grammar.y"
    {
                  InitCall* rtn = new InitCall(lineno, (yyvsp[(6) - (9)].strval), 0);
                  rtn->setAccel();
                  (yyval.member) = rtn;
		}
    break;

  case 169:

/* Line 1455 of yacc.c  */
#line 672 "xi-grammar.y"
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[(1) - (1)].ntype),0); }
    break;

  case 170:

/* Line 1455 of yacc.c  */
#line 674 "xi-grammar.y"
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[(1) - (3)].ntype),(yyvsp[(3) - (3)].pupable)); }
    break;

  case 171:

/* Line 1455 of yacc.c  */
#line 677 "xi-grammar.y"
    { (yyval.includeFile) = new IncludeFile(lineno,(yyvsp[(1) - (1)].strval)); }
    break;

  case 172:

/* Line 1455 of yacc.c  */
#line 681 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (2)].member); }
    break;

  case 173:

/* Line 1455 of yacc.c  */
#line 684 "xi-grammar.y"
    { yyerror("The preceding entry method declaration must be semicolon-terminated."); YYABORT; }
    break;

  case 174:

/* Line 1455 of yacc.c  */
#line 688 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].entry); }
    break;

  case 175:

/* Line 1455 of yacc.c  */
#line 690 "xi-grammar.y"
    {
                  (yyvsp[(2) - (2)].entry)->tspec = (yyvsp[(1) - (2)].tvarlist);
                  (yyval.member) = (yyvsp[(2) - (2)].entry);
                }
    break;

  case 176:

/* Line 1455 of yacc.c  */
#line 695 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].member); }
    break;

  case 177:

/* Line 1455 of yacc.c  */
#line 699 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 178:

/* Line 1455 of yacc.c  */
#line 701 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 179:

/* Line 1455 of yacc.c  */
#line 703 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 180:

/* Line 1455 of yacc.c  */
#line 705 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 181:

/* Line 1455 of yacc.c  */
#line 707 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 182:

/* Line 1455 of yacc.c  */
#line 709 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 183:

/* Line 1455 of yacc.c  */
#line 711 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 184:

/* Line 1455 of yacc.c  */
#line 713 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 185:

/* Line 1455 of yacc.c  */
#line 715 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 186:

/* Line 1455 of yacc.c  */
#line 717 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 187:

/* Line 1455 of yacc.c  */
#line 719 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 188:

/* Line 1455 of yacc.c  */
#line 722 "xi-grammar.y"
    { 
		  if ((yyvsp[(7) - (7)].sc) != 0) { 
		    (yyvsp[(7) - (7)].sc)->con1 = new SdagConstruct(SIDENT, (yyvsp[(4) - (7)].strval));
                    (yyvsp[(7) - (7)].sc)->param = new ParamList((yyvsp[(5) - (7)].plist));
                  }
		  (yyval.entry) = new Entry(lineno, (yyvsp[(2) - (7)].intval), (yyvsp[(3) - (7)].type), (yyvsp[(4) - (7)].strval), (yyvsp[(5) - (7)].plist), (yyvsp[(6) - (7)].val), (yyvsp[(7) - (7)].sc), 0, 0); 
		}
    break;

  case 189:

/* Line 1455 of yacc.c  */
#line 730 "xi-grammar.y"
    { 
		  if ((yyvsp[(5) - (5)].sc) != 0) {
		    (yyvsp[(5) - (5)].sc)->con1 = new SdagConstruct(SIDENT, (yyvsp[(3) - (5)].strval));
                    (yyvsp[(5) - (5)].sc)->param = new ParamList((yyvsp[(4) - (5)].plist));
                  }
		  Entry *e = new Entry(lineno, (yyvsp[(2) - (5)].intval),     0, (yyvsp[(3) - (5)].strval), (yyvsp[(4) - (5)].plist),  0, (yyvsp[(5) - (5)].sc), 0, 0);
		  if (e->param && e->param->isCkMigMsgPtr()) {
		    yyerror("Charm++ takes a CkMigrateMsg chare constructor for granted, but continuing anyway");
		    (yyval.entry) = NULL;
		  } else
		    (yyval.entry) = e;
		}
    break;

  case 190:

/* Line 1455 of yacc.c  */
#line 743 "xi-grammar.y"
    {
                  int attribs = SACCEL;
                  const char* name = (yyvsp[(6) - (12)].strval);
                  ParamList* paramList = (yyvsp[(7) - (12)].plist);
                  ParamList* accelParamList = (yyvsp[(8) - (12)].plist);
		  XStr* codeBody = new XStr((yyvsp[(10) - (12)].strval));
                  const char* callbackName = (yyvsp[(12) - (12)].strval);

                  (yyval.entry) = new Entry(lineno, attribs, new BuiltinType("void"), name, paramList,
                                 0, 0, 0, 0, 0
                                );
                  (yyval.entry)->setAccelParam(accelParamList);
                  (yyval.entry)->setAccelCodeBody(codeBody);
                  (yyval.entry)->setAccelCallbackName(new XStr(callbackName));
                }
    break;

  case 191:

/* Line 1455 of yacc.c  */
#line 761 "xi-grammar.y"
    { (yyval.accelBlock) = new AccelBlock(lineno, new XStr((yyvsp[(3) - (5)].strval))); }
    break;

  case 192:

/* Line 1455 of yacc.c  */
#line 763 "xi-grammar.y"
    { (yyval.accelBlock) = new AccelBlock(lineno, NULL); }
    break;

  case 193:

/* Line 1455 of yacc.c  */
#line 767 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("void"); }
    break;

  case 194:

/* Line 1455 of yacc.c  */
#line 769 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 195:

/* Line 1455 of yacc.c  */
#line 773 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 196:

/* Line 1455 of yacc.c  */
#line 775 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(2) - (3)].intval); }
    break;

  case 197:

/* Line 1455 of yacc.c  */
#line 777 "xi-grammar.y"
    { printf("Invalid entry method attribute list\n"); YYABORT; }
    break;

  case 198:

/* Line 1455 of yacc.c  */
#line 781 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (1)].intval); }
    break;

  case 199:

/* Line 1455 of yacc.c  */
#line 783 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (3)].intval) | (yyvsp[(3) - (3)].intval); }
    break;

  case 200:

/* Line 1455 of yacc.c  */
#line 787 "xi-grammar.y"
    { (yyval.intval) = STHREADED; }
    break;

  case 201:

/* Line 1455 of yacc.c  */
#line 789 "xi-grammar.y"
    { (yyval.intval) = SSYNC; }
    break;

  case 202:

/* Line 1455 of yacc.c  */
#line 791 "xi-grammar.y"
    { (yyval.intval) = SIGET; }
    break;

  case 203:

/* Line 1455 of yacc.c  */
#line 793 "xi-grammar.y"
    { (yyval.intval) = SLOCKED; }
    break;

  case 204:

/* Line 1455 of yacc.c  */
#line 795 "xi-grammar.y"
    { (yyval.intval) = SCREATEHERE; }
    break;

  case 205:

/* Line 1455 of yacc.c  */
#line 797 "xi-grammar.y"
    { (yyval.intval) = SCREATEHOME; }
    break;

  case 206:

/* Line 1455 of yacc.c  */
#line 799 "xi-grammar.y"
    { (yyval.intval) = SNOKEEP; }
    break;

  case 207:

/* Line 1455 of yacc.c  */
#line 801 "xi-grammar.y"
    { (yyval.intval) = SNOTRACE; }
    break;

  case 208:

/* Line 1455 of yacc.c  */
#line 803 "xi-grammar.y"
    { (yyval.intval) = SIMMEDIATE; }
    break;

  case 209:

/* Line 1455 of yacc.c  */
#line 805 "xi-grammar.y"
    { (yyval.intval) = SSKIPSCHED; }
    break;

  case 210:

/* Line 1455 of yacc.c  */
#line 807 "xi-grammar.y"
    { (yyval.intval) = SINLINE; }
    break;

  case 211:

/* Line 1455 of yacc.c  */
#line 809 "xi-grammar.y"
    { (yyval.intval) = SLOCAL; }
    break;

  case 212:

/* Line 1455 of yacc.c  */
#line 811 "xi-grammar.y"
    { (yyval.intval) = SPYTHON; }
    break;

  case 213:

/* Line 1455 of yacc.c  */
#line 813 "xi-grammar.y"
    { (yyval.intval) = SMEM; }
    break;

  case 214:

/* Line 1455 of yacc.c  */
#line 815 "xi-grammar.y"
    { (yyval.intval) = SREDUCE; }
    break;

  case 215:

/* Line 1455 of yacc.c  */
#line 817 "xi-grammar.y"
    { printf("Invalid entry method attribute: %s\n", yylval); YYABORT; }
    break;

  case 216:

/* Line 1455 of yacc.c  */
#line 821 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 217:

/* Line 1455 of yacc.c  */
#line 823 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 218:

/* Line 1455 of yacc.c  */
#line 825 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 219:

/* Line 1455 of yacc.c  */
#line 829 "xi-grammar.y"
    { (yyval.strval) = ""; }
    break;

  case 220:

/* Line 1455 of yacc.c  */
#line 831 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 221:

/* Line 1455 of yacc.c  */
#line 833 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[(1) - (3)].strval))+strlen((yyvsp[(3) - (3)].strval))+3];
			sprintf(tmp,"%s, %s", (yyvsp[(1) - (3)].strval), (yyvsp[(3) - (3)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 222:

/* Line 1455 of yacc.c  */
#line 841 "xi-grammar.y"
    { (yyval.strval) = ""; }
    break;

  case 223:

/* Line 1455 of yacc.c  */
#line 843 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 224:

/* Line 1455 of yacc.c  */
#line 845 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s[%s]%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 225:

/* Line 1455 of yacc.c  */
#line 851 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s{%s}%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 226:

/* Line 1455 of yacc.c  */
#line 857 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s(%s)%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 227:

/* Line 1455 of yacc.c  */
#line 863 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(2) - (4)].strval))+strlen((yyvsp[(4) - (4)].strval))+3];
			sprintf(tmp,"(%s)%s", (yyvsp[(2) - (4)].strval), (yyvsp[(4) - (4)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 228:

/* Line 1455 of yacc.c  */
#line 871 "xi-grammar.y"
    {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			(yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].type),(yyvsp[(2) - (3)].strval));
		}
    break;

  case 229:

/* Line 1455 of yacc.c  */
#line 878 "xi-grammar.y"
    { 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			(yyval.intval) = 0;
		}
    break;

  case 230:

/* Line 1455 of yacc.c  */
#line 886 "xi-grammar.y"
    { 
			in_braces=0;
			(yyval.intval) = 0;
		}
    break;

  case 231:

/* Line 1455 of yacc.c  */
#line 893 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (1)].type));}
    break;

  case 232:

/* Line 1455 of yacc.c  */
#line 895 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].type),(yyvsp[(2) - (3)].strval)); (yyval.pname)->setConditional((yyvsp[(3) - (3)].intval)); }
    break;

  case 233:

/* Line 1455 of yacc.c  */
#line 897 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (4)].type),(yyvsp[(2) - (4)].strval),0,(yyvsp[(4) - (4)].val));}
    break;

  case 234:

/* Line 1455 of yacc.c  */
#line 899 "xi-grammar.y"
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].pname)->getType(), (yyvsp[(1) - (3)].pname)->getName() ,(yyvsp[(2) - (3)].strval));
		}
    break;

  case 235:

/* Line 1455 of yacc.c  */
#line 905 "xi-grammar.y"
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READONLY; }
    break;

  case 236:

/* Line 1455 of yacc.c  */
#line 906 "xi-grammar.y"
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READWRITE; }
    break;

  case 237:

/* Line 1455 of yacc.c  */
#line 907 "xi-grammar.y"
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_WRITEONLY; }
    break;

  case 238:

/* Line 1455 of yacc.c  */
#line 910 "xi-grammar.y"
    { (yyval.xstrptr) = new XStr((yyvsp[(1) - (1)].strval)); }
    break;

  case 239:

/* Line 1455 of yacc.c  */
#line 911 "xi-grammar.y"
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "->" << (yyvsp[(4) - (4)].strval); }
    break;

  case 240:

/* Line 1455 of yacc.c  */
#line 912 "xi-grammar.y"
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[(1) - (3)].xstrptr)) << "." << (yyvsp[(3) - (3)].strval); }
    break;

  case 241:

/* Line 1455 of yacc.c  */
#line 914 "xi-grammar.y"
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "[" << *((yyvsp[(3) - (4)].xstrptr)) << "]";
                  delete (yyvsp[(1) - (4)].xstrptr);
                  delete (yyvsp[(3) - (4)].xstrptr);
                }
    break;

  case 242:

/* Line 1455 of yacc.c  */
#line 921 "xi-grammar.y"
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "[" << (yyvsp[(3) - (4)].strval) << "]";
                  delete (yyvsp[(1) - (4)].xstrptr);
                }
    break;

  case 243:

/* Line 1455 of yacc.c  */
#line 927 "xi-grammar.y"
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "(" << *((yyvsp[(3) - (4)].xstrptr)) << ")";
                  delete (yyvsp[(1) - (4)].xstrptr);
                  delete (yyvsp[(3) - (4)].xstrptr);
                }
    break;

  case 244:

/* Line 1455 of yacc.c  */
#line 936 "xi-grammar.y"
    {
                  in_bracket = 0;
                  (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].pname)->getType(), (yyvsp[(1) - (3)].pname)->getName(), (yyvsp[(2) - (3)].strval));
                }
    break;

  case 245:

/* Line 1455 of yacc.c  */
#line 943 "xi-grammar.y"
    {
                  (yyval.pname) = new Parameter(lineno, (yyvsp[(3) - (7)].type), (yyvsp[(4) - (7)].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[(6) - (7)].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[(1) - (7)].intval));
                }
    break;

  case 246:

/* Line 1455 of yacc.c  */
#line 949 "xi-grammar.y"
    {
		  (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (5)].type), (yyvsp[(2) - (5)].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[(4) - (5)].xstrptr));
                  (yyval.pname)->setAccelBufferType(Parameter::ACCEL_BUFFER_TYPE_READWRITE);
		}
    break;

  case 247:

/* Line 1455 of yacc.c  */
#line 955 "xi-grammar.y"
    {
                  (yyval.pname) = (yyvsp[(3) - (6)].pname);
                  (yyval.pname)->setAccelInstName((yyvsp[(5) - (6)].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[(1) - (6)].intval));
		}
    break;

  case 248:

/* Line 1455 of yacc.c  */
#line 963 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (1)].pname)); }
    break;

  case 249:

/* Line 1455 of yacc.c  */
#line 965 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (3)].pname),(yyvsp[(3) - (3)].plist)); }
    break;

  case 250:

/* Line 1455 of yacc.c  */
#line 969 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (1)].pname)); }
    break;

  case 251:

/* Line 1455 of yacc.c  */
#line 971 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (3)].pname),(yyvsp[(3) - (3)].plist)); }
    break;

  case 252:

/* Line 1455 of yacc.c  */
#line 975 "xi-grammar.y"
    { (yyval.plist) = (yyvsp[(2) - (3)].plist); }
    break;

  case 253:

/* Line 1455 of yacc.c  */
#line 977 "xi-grammar.y"
    { (yyval.plist) = new ParamList(new Parameter(0, new BuiltinType("void"))); }
    break;

  case 254:

/* Line 1455 of yacc.c  */
#line 981 "xi-grammar.y"
    { (yyval.plist) = (yyvsp[(2) - (3)].plist); }
    break;

  case 255:

/* Line 1455 of yacc.c  */
#line 983 "xi-grammar.y"
    { (yyval.plist) = 0; }
    break;

  case 256:

/* Line 1455 of yacc.c  */
#line 987 "xi-grammar.y"
    { (yyval.val) = 0; }
    break;

  case 257:

/* Line 1455 of yacc.c  */
#line 989 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(3) - (3)].strval)); }
    break;

  case 258:

/* Line 1455 of yacc.c  */
#line 993 "xi-grammar.y"
    { (yyval.sc) = 0; }
    break;

  case 259:

/* Line 1455 of yacc.c  */
#line 995 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSDAGENTRY, (yyvsp[(1) - (1)].sc)); }
    break;

  case 260:

/* Line 1455 of yacc.c  */
#line 997 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSDAGENTRY, (yyvsp[(2) - (3)].sc)); }
    break;

  case 261:

/* Line 1455 of yacc.c  */
#line 1001 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSLIST, (yyvsp[(1) - (1)].sc)); }
    break;

  case 262:

/* Line 1455 of yacc.c  */
#line 1003 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSLIST, (yyvsp[(1) - (2)].sc), (yyvsp[(2) - (2)].sc));  }
    break;

  case 263:

/* Line 1455 of yacc.c  */
#line 1007 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SOLIST, (yyvsp[(1) - (1)].sc)); }
    break;

  case 264:

/* Line 1455 of yacc.c  */
#line 1009 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SOLIST, (yyvsp[(1) - (2)].sc), (yyvsp[(2) - (2)].sc)); }
    break;

  case 265:

/* Line 1455 of yacc.c  */
#line 1013 "xi-grammar.y"
    { (yyval.sc) = 0; }
    break;

  case 266:

/* Line 1455 of yacc.c  */
#line 1015 "xi-grammar.y"
    { (yyval.sc) = (yyvsp[(3) - (4)].sc); }
    break;

  case 267:

/* Line 1455 of yacc.c  */
#line 1019 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SPUBLISHES, new SdagConstruct(SIDENT, (yyvsp[(1) - (1)].strval))); }
    break;

  case 268:

/* Line 1455 of yacc.c  */
#line 1021 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SPUBLISHES, new SdagConstruct(SIDENT, (yyvsp[(1) - (3)].strval)), (yyvsp[(3) - (3)].sc));  }
    break;

  case 269:

/* Line 1455 of yacc.c  */
#line 1025 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 270:

/* Line 1455 of yacc.c  */
#line 1027 "xi-grammar.y"
    { (yyval.strval) = 0; }
    break;

  case 271:

/* Line 1455 of yacc.c  */
#line 1031 "xi-grammar.y"
    {
		   (yyval.sc) = buildAtomic((yyvsp[(4) - (6)].strval), (yyvsp[(6) - (6)].sc), (yyvsp[(2) - (6)].strval));
		 }
    break;

  case 272:

/* Line 1455 of yacc.c  */
#line 1035 "xi-grammar.y"
    {  
		   in_braces = 0;
		   if (((yyvsp[(4) - (8)].plist)->isVoid() == 0) && ((yyvsp[(4) - (8)].plist)->isMessage() == 0))
                   {
		      connectEntries.push_back(new Entry(0, 0, new BuiltinType("void"), (yyvsp[(3) - (8)].strval),
	 	 			new ParamList(new Parameter(lineno, new PtrType( 
                                        new NamedType("CkMarshallMsg")), "_msg")), 0, 0, 0, 1, (yyvsp[(4) - (8)].plist)));
		   }
		   else  {
		      connectEntries.push_back(new Entry(0, 0, new BuiltinType("void"), (yyvsp[(3) - (8)].strval), (yyvsp[(4) - (8)].plist), 0, 0, 0, 1, (yyvsp[(4) - (8)].plist)));
                   }
                   (yyval.sc) = new SdagConstruct(SCONNECT, (yyvsp[(3) - (8)].strval), (yyvsp[(7) - (8)].strval), (yyvsp[(4) - (8)].plist));
		}
    break;

  case 273:

/* Line 1455 of yacc.c  */
#line 1049 "xi-grammar.y"
    { (yyval.sc) = new WhenConstruct((yyvsp[(2) - (4)].entrylist), 0); }
    break;

  case 274:

/* Line 1455 of yacc.c  */
#line 1051 "xi-grammar.y"
    { (yyval.sc) = new WhenConstruct((yyvsp[(2) - (3)].entrylist), (yyvsp[(3) - (3)].sc)); }
    break;

  case 275:

/* Line 1455 of yacc.c  */
#line 1053 "xi-grammar.y"
    { (yyval.sc) = new WhenConstruct((yyvsp[(2) - (5)].entrylist), (yyvsp[(4) - (5)].sc)); }
    break;

  case 276:

/* Line 1455 of yacc.c  */
#line 1055 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SOVERLAP,0, 0,0,0,0,(yyvsp[(3) - (4)].sc), 0); }
    break;

  case 277:

/* Line 1455 of yacc.c  */
#line 1057 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFOR, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (11)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(5) - (11)].strval)),
		             new SdagConstruct(SINT_EXPR, (yyvsp[(7) - (11)].strval)), 0, (yyvsp[(10) - (11)].sc), 0); }
    break;

  case 278:

/* Line 1455 of yacc.c  */
#line 1060 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFOR, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (9)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(5) - (9)].strval)), 
		         new SdagConstruct(SINT_EXPR, (yyvsp[(7) - (9)].strval)), 0, (yyvsp[(9) - (9)].sc), 0); }
    break;

  case 279:

/* Line 1455 of yacc.c  */
#line 1063 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFORALL, 0, new SdagConstruct(SIDENT, (yyvsp[(3) - (12)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(6) - (12)].strval)), 
		             new SdagConstruct(SINT_EXPR, (yyvsp[(8) - (12)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(10) - (12)].strval)), (yyvsp[(12) - (12)].sc), 0); }
    break;

  case 280:

/* Line 1455 of yacc.c  */
#line 1066 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFORALL, 0, new SdagConstruct(SIDENT, (yyvsp[(3) - (14)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(6) - (14)].strval)), 
		                 new SdagConstruct(SINT_EXPR, (yyvsp[(8) - (14)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(10) - (14)].strval)), (yyvsp[(13) - (14)].sc), 0); }
    break;

  case 281:

/* Line 1455 of yacc.c  */
#line 1069 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (6)].strval)), (yyvsp[(6) - (6)].sc),0,0,(yyvsp[(5) - (6)].sc),0); }
    break;

  case 282:

/* Line 1455 of yacc.c  */
#line 1071 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (8)].strval)), (yyvsp[(8) - (8)].sc),0,0,(yyvsp[(6) - (8)].sc),0); }
    break;

  case 283:

/* Line 1455 of yacc.c  */
#line 1073 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (5)].strval)), 0,0,0,(yyvsp[(5) - (5)].sc),0); }
    break;

  case 284:

/* Line 1455 of yacc.c  */
#line 1075 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SWHILE, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (7)].strval)), 0,0,0,(yyvsp[(6) - (7)].sc),0); }
    break;

  case 285:

/* Line 1455 of yacc.c  */
#line 1077 "xi-grammar.y"
    { (yyval.sc) = (yyvsp[(2) - (3)].sc); }
    break;

  case 286:

/* Line 1455 of yacc.c  */
#line 1079 "xi-grammar.y"
    { (yyval.sc) = buildAtomic((yyvsp[(2) - (3)].strval), NULL, NULL); }
    break;

  case 287:

/* Line 1455 of yacc.c  */
#line 1081 "xi-grammar.y"
    { printf("Unknown SDAG construct or malformed entry method definition.\n"
                         "You may have forgotten to terminate an entry method definition with a"
                         " semicolon or forgotten to mark a block of sequential SDAG code as 'atomic'\n"); YYABORT; }
    break;

  case 288:

/* Line 1455 of yacc.c  */
#line 1087 "xi-grammar.y"
    { (yyval.sc) = 0; }
    break;

  case 289:

/* Line 1455 of yacc.c  */
#line 1089 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SELSE, 0,0,0,0,0, (yyvsp[(2) - (2)].sc),0); }
    break;

  case 290:

/* Line 1455 of yacc.c  */
#line 1091 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SELSE, 0,0,0,0,0, (yyvsp[(3) - (4)].sc),0); }
    break;

  case 291:

/* Line 1455 of yacc.c  */
#line 1094 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFORWARD, new SdagConstruct(SIDENT, (yyvsp[(1) - (1)].strval))); }
    break;

  case 292:

/* Line 1455 of yacc.c  */
#line 1096 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFORWARD, new SdagConstruct(SIDENT, (yyvsp[(1) - (3)].strval)), (yyvsp[(3) - (3)].sc));  }
    break;

  case 293:

/* Line 1455 of yacc.c  */
#line 1100 "xi-grammar.y"
    { in_int_expr = 0; (yyval.intval) = 0; }
    break;

  case 294:

/* Line 1455 of yacc.c  */
#line 1104 "xi-grammar.y"
    { in_int_expr = 1; (yyval.intval) = 0; }
    break;

  case 295:

/* Line 1455 of yacc.c  */
#line 1108 "xi-grammar.y"
    { (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[(1) - (2)].strval), (yyvsp[(2) - (2)].plist), 0, 0, 0, 0); }
    break;

  case 296:

/* Line 1455 of yacc.c  */
#line 1110 "xi-grammar.y"
    { (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[(1) - (5)].strval), (yyvsp[(5) - (5)].plist), 0, 0, (yyvsp[(3) - (5)].strval), 0); }
    break;

  case 297:

/* Line 1455 of yacc.c  */
#line 1114 "xi-grammar.y"
    { (yyval.entrylist) = new EntryList((yyvsp[(1) - (1)].entry)); }
    break;

  case 298:

/* Line 1455 of yacc.c  */
#line 1116 "xi-grammar.y"
    { (yyval.entrylist) = new EntryList((yyvsp[(1) - (3)].entry),(yyvsp[(3) - (3)].entrylist)); }
    break;

  case 299:

/* Line 1455 of yacc.c  */
#line 1120 "xi-grammar.y"
    { in_bracket=1; }
    break;

  case 300:

/* Line 1455 of yacc.c  */
#line 1123 "xi-grammar.y"
    { in_bracket=0; }
    break;

  case 301:

/* Line 1455 of yacc.c  */
#line 1127 "xi-grammar.y"
    { if (!macroDefined((yyvsp[(2) - (2)].strval), 1)) in_comment = 1; }
    break;

  case 302:

/* Line 1455 of yacc.c  */
#line 1131 "xi-grammar.y"
    { if (!macroDefined((yyvsp[(2) - (2)].strval), 0)) in_comment = 1; }
    break;



/* Line 1455 of yacc.c  */
#line 4432 "y.tab.c"
      default: break;
    }
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
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (YY_("syntax error"));
#else
      {
	YYSIZE_T yysize = yysyntax_error (0, yystate, yychar);
	if (yymsg_alloc < yysize && yymsg_alloc < YYSTACK_ALLOC_MAXIMUM)
	  {
	    YYSIZE_T yyalloc = 2 * yysize;
	    if (! (yysize <= yyalloc && yyalloc <= YYSTACK_ALLOC_MAXIMUM))
	      yyalloc = YYSTACK_ALLOC_MAXIMUM;
	    if (yymsg != yymsgbuf)
	      YYSTACK_FREE (yymsg);
	    yymsg = (char *) YYSTACK_ALLOC (yyalloc);
	    if (yymsg)
	      yymsg_alloc = yyalloc;
	    else
	      {
		yymsg = yymsgbuf;
		yymsg_alloc = sizeof yymsgbuf;
	      }
	  }

	if (0 < yysize && yysize <= yymsg_alloc)
	  {
	    (void) yysyntax_error (yymsg, yystate, yychar);
	    yyerror (yymsg);
	  }
	else
	  {
	    yyerror (YY_("syntax error"));
	    if (yysize != 0)
	      goto yyexhaustedlab;
	  }
      }
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
      if (yyn != YYPACT_NINF)
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
     yydestruct ("Cleanup: discarding lookahead",
		 yytoken, &yylval);
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



/* Line 1675 of yacc.c  */
#line 1134 "xi-grammar.y"

void yyerror(const char *mesg)
{
    std::cerr << cur_file<<":"<<lineno<<": Charmxi syntax error> "
	      << mesg << std::endl;
}

