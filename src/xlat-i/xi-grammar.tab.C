
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
#define CASE 329




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
  WhenConstruct *when;
  XStr* xstrptr;
  AccelBlock* accelBlock;



/* Line 214 of yacc.c  */
#line 320 "y.tab.c"
} YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
#endif


/* Copy the second part of user declarations.  */


/* Line 264 of yacc.c  */
#line 332 "y.tab.c"

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
#define YYLAST   940

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  91
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  116
/* YYNRULES -- Number of rules.  */
#define YYNRULES  315
/* YYNRULES -- Number of states.  */
#define YYNSTATES  622

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
     837,   841,   843,   846,   848,   851,   853,   856,   858,   859,
     864,   866,   870,   872,   873,   878,   882,   888,   890,   892,
     894,   896,   898,   900,   902,   904,   911,   920,   925,   927,
     932,   944,   954,   967,   982,   989,   998,  1004,  1012,  1016,
    1020,  1022,  1023,  1026,  1031,  1033,  1037,  1039,  1041,  1044,
    1050,  1052,  1056,  1058,  1060,  1063
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
      92,     0,    -1,    93,    -1,    -1,    98,    93,    -1,    -1,
       5,    -1,    -1,    75,    -1,    55,    -1,    55,    -1,    97,
      76,    76,    55,    -1,     3,    96,    99,    -1,     4,    96,
      99,    -1,    75,    -1,    77,   100,    78,    95,    -1,    -1,
     102,   100,    -1,    54,    53,    97,    -1,    54,    97,    -1,
      94,   157,    -1,    94,   136,    -1,     5,    39,   167,   109,
      96,   106,   184,    -1,    94,    77,   100,    78,    95,    -1,
      53,    96,    77,   100,    78,    -1,   101,    75,    -1,   101,
     164,    -1,    94,    98,    -1,    94,   139,    -1,    94,   140,
      -1,    94,   141,    -1,    94,   143,    -1,    94,   154,    -1,
     205,    -1,   206,    -1,   166,    -1,     1,    -1,   115,    -1,
      56,    -1,    57,    -1,   103,    -1,   103,    79,   104,    -1,
      -1,   104,    -1,    -1,    80,   105,    81,    -1,    61,    -1,
      62,    -1,    63,    -1,    64,    -1,    67,    61,    -1,    67,
      62,    -1,    67,    62,    61,    -1,    67,    62,    62,    -1,
      67,    63,    -1,    67,    64,    -1,    62,    62,    -1,    65,
      -1,    66,    -1,    62,    66,    -1,    35,    -1,    96,   106,
      -1,    97,   106,    -1,   107,    -1,   109,    -1,   110,    82,
      -1,   111,    82,    -1,   112,    82,    -1,   114,    83,    82,
      96,    84,    83,   182,    84,    -1,   110,    -1,   111,    -1,
     112,    -1,   113,    -1,    36,   114,    -1,   114,    36,    -1,
     114,    85,    -1,   114,    -1,    56,    -1,    97,    -1,    86,
     116,    87,    -1,    -1,   117,   118,    -1,     6,   115,    97,
     118,    -1,     6,    16,   110,    82,    96,    -1,    -1,    35,
      -1,    -1,    86,   123,    87,    -1,   124,    -1,   124,    79,
     123,    -1,    37,    -1,    38,    -1,    -1,    86,   126,    87,
      -1,   131,    -1,   131,    79,   126,    -1,    -1,    57,    -1,
      51,    -1,    -1,    86,   130,    87,    -1,   128,    -1,   128,
      79,   130,    -1,    30,    -1,    51,    -1,    -1,    17,    -1,
      -1,    86,    87,    -1,   132,   115,    96,   133,    75,    -1,
     134,    -1,   134,   135,    -1,    16,   122,   108,    -1,    16,
     122,   108,    77,   135,    78,    -1,    -1,    76,   138,    -1,
     109,    -1,   109,    79,   138,    -1,    11,   125,   108,   137,
     155,    -1,    12,   125,   108,   137,   155,    -1,    13,   125,
     108,   137,   155,    -1,    14,   125,   108,   137,   155,    -1,
      86,    56,    96,    87,    -1,    86,    96,    87,    -1,    15,
     129,   142,   108,   137,   155,    -1,    15,   142,   129,   108,
     137,   155,    -1,    11,   125,    96,   137,   155,    -1,    12,
     125,    96,   137,   155,    -1,    13,   125,    96,   137,   155,
      -1,    14,   125,    96,   137,   155,    -1,    15,   142,    96,
     137,   155,    -1,    16,   122,    96,    75,    -1,    16,   122,
      96,    77,   135,    78,    75,    -1,    -1,    88,   115,    -1,
      -1,    88,    56,    -1,    88,    57,    -1,    18,    96,   149,
      -1,   113,   150,    -1,   115,    96,   150,    -1,   151,    -1,
     151,    79,   152,    -1,    22,    80,   152,    81,    -1,   153,
     144,    -1,   153,   145,    -1,   153,   146,    -1,   153,   147,
      -1,   153,   148,    -1,    75,    -1,    77,   156,    78,    95,
      -1,    -1,   162,   156,    -1,   119,    -1,   120,    -1,   159,
      -1,   158,    -1,    10,   160,    -1,    19,   161,    -1,    18,
      96,    -1,     8,   121,    97,    -1,     8,   121,    97,    83,
     121,    84,    -1,     8,   121,    97,    80,   104,    81,    83,
     121,    84,    -1,     7,   121,    97,    -1,     7,   121,    97,
      83,   121,    84,    -1,     9,   121,    97,    -1,     9,   121,
      97,    83,   121,    84,    -1,     9,   121,    97,    80,   104,
      81,    83,   121,    84,    -1,     9,    86,    68,    87,   121,
      97,    83,   121,    84,    -1,   109,    -1,   109,    79,   160,
      -1,    57,    -1,   163,    75,    -1,   163,   164,    -1,   165,
      -1,   153,   165,    -1,   157,    -1,    39,    -1,    78,    -1,
       7,    -1,     8,    -1,     9,    -1,    11,    -1,    12,    -1,
      15,    -1,    13,    -1,    14,    -1,     6,    -1,    39,   168,
     167,    96,   184,   186,   187,    -1,    39,   168,    96,   184,
     187,    -1,    39,    86,    68,    87,    35,    96,   184,   185,
     175,   173,   176,    96,    -1,    71,   175,   173,   176,    75,
      -1,    71,    75,    -1,    35,    -1,   111,    -1,    -1,    86,
     169,    87,    -1,     1,    -1,   170,    -1,   170,    79,   169,
      -1,    21,    -1,    23,    -1,    24,    -1,    25,    -1,    31,
      -1,    32,    -1,    33,    -1,    34,    -1,    26,    -1,    27,
      -1,    28,    -1,    52,    -1,    51,   127,    -1,    72,    -1,
      73,    -1,     1,    -1,    57,    -1,    56,    -1,    97,    -1,
      -1,    58,    -1,    58,    79,   172,    -1,    -1,    58,    -1,
      58,    86,   173,    87,   173,    -1,    58,    77,   173,    78,
     173,    -1,    58,    83,   172,    84,   173,    -1,    83,   173,
      84,   173,    -1,   115,    96,    86,    -1,    77,    -1,    78,
      -1,   115,    -1,   115,    96,   132,    -1,   115,    96,    88,
     171,    -1,   174,   173,    87,    -1,     6,    -1,    69,    -1,
      70,    -1,    96,    -1,   179,    89,    81,    96,    -1,   179,
      90,    96,    -1,   179,    86,   179,    87,    -1,   179,    86,
      56,    87,    -1,   179,    83,   179,    84,    -1,   174,   173,
      87,    -1,   178,    76,   115,    96,    80,   179,    81,    -1,
     115,    96,    80,   179,    81,    -1,   178,    76,   180,    80,
     179,    81,    -1,   177,    -1,   177,    79,   182,    -1,   181,
      -1,   181,    79,   183,    -1,    83,   182,    84,    -1,    83,
      84,    -1,    86,   183,    87,    -1,    86,    87,    -1,    -1,
      20,    88,    56,    -1,    -1,   196,    -1,    77,   188,    78,
      -1,   196,    -1,   196,   188,    -1,   196,    -1,   196,   188,
      -1,   194,    -1,   194,   190,    -1,   195,    -1,    -1,    50,
      83,   192,    84,    -1,    55,    -1,    55,    79,   192,    -1,
      57,    -1,    -1,    43,   202,    77,    78,    -1,    43,   202,
     196,    -1,    43,   202,    77,   188,    78,    -1,    45,    -1,
      49,    -1,    44,    -1,    40,    -1,    41,    -1,    47,    -1,
      42,    -1,    46,    -1,    45,   193,   175,   173,   176,   191,
      -1,    49,    83,    55,   184,    84,   175,   173,    78,    -1,
      44,    77,   189,    78,    -1,   194,    -1,    74,    77,   190,
      78,    -1,    40,   200,   173,    75,   173,    75,   173,   199,
      77,   188,    78,    -1,    40,   200,   173,    75,   173,    75,
     173,   199,   196,    -1,    41,    86,    55,    87,   200,   173,
      76,   173,    79,   173,   199,   196,    -1,    41,    86,    55,
      87,   200,   173,    76,   173,    79,   173,   199,    77,   188,
      78,    -1,    47,   200,   173,   199,   196,   197,    -1,    47,
     200,   173,   199,    77,   188,    78,   197,    -1,    42,   200,
     173,   199,   196,    -1,    42,   200,   173,   199,    77,   188,
      78,    -1,    46,   198,    75,    -1,   175,   173,   176,    -1,
       1,    -1,    -1,    48,   196,    -1,    48,    77,   188,    78,
      -1,    55,    -1,    55,    79,   198,    -1,    84,    -1,    83,
      -1,    55,   184,    -1,    55,   203,   173,   204,   184,    -1,
     201,    -1,   201,    79,   202,    -1,    86,    -1,    87,    -1,
      59,    96,    -1,    60,    96,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   157,   157,   162,   165,   170,   171,   176,   177,   181,
     185,   187,   195,   199,   206,   208,   213,   214,   218,   220,
     222,   224,   226,   238,   240,   242,   244,   246,   248,   250,
     252,   254,   256,   258,   260,   262,   264,   268,   270,   272,
     276,   278,   283,   284,   289,   290,   294,   296,   298,   300,
     302,   304,   306,   308,   310,   312,   314,   316,   318,   320,
     322,   326,   327,   334,   336,   340,   344,   346,   350,   354,
     356,   358,   360,   363,   365,   369,   371,   375,   377,   381,
     386,   387,   391,   395,   400,   401,   406,   407,   417,   419,
     423,   425,   430,   431,   435,   437,   442,   443,   447,   452,
     453,   457,   459,   463,   465,   470,   471,   475,   476,   479,
     483,   485,   489,   491,   496,   497,   501,   503,   507,   509,
     513,   517,   521,   527,   531,   533,   537,   539,   543,   547,
     551,   555,   557,   562,   563,   568,   569,   571,   575,   577,
     579,   583,   585,   589,   593,   595,   597,   599,   601,   605,
     607,   612,   619,   623,   625,   627,   628,   630,   632,   634,
     638,   640,   642,   648,   651,   656,   658,   660,   666,   674,
     676,   679,   683,   686,   690,   692,   697,   701,   703,   705,
     707,   709,   711,   713,   715,   717,   719,   721,   724,   732,
     745,   763,   765,   769,   771,   776,   777,   779,   783,   785,
     789,   791,   793,   795,   797,   799,   801,   803,   805,   807,
     809,   811,   813,   815,   817,   819,   823,   825,   827,   832,
     833,   835,   844,   845,   847,   853,   859,   865,   873,   880,
     888,   895,   897,   899,   901,   908,   909,   910,   913,   914,
     915,   916,   923,   929,   938,   945,   951,   957,   965,   967,
     971,   973,   977,   979,   983,   985,   990,   991,   996,   997,
     999,  1003,  1005,  1009,  1011,  1015,  1017,  1019,  1024,  1025,
    1029,  1031,  1035,  1038,  1041,  1043,  1045,  1049,  1051,  1053,
    1055,  1057,  1059,  1061,  1063,  1067,  1071,  1085,  1087,  1089,
    1091,  1094,  1097,  1100,  1103,  1105,  1107,  1109,  1111,  1113,
    1115,  1122,  1123,  1125,  1128,  1130,  1134,  1138,  1142,  1144,
    1148,  1150,  1154,  1157,  1161,  1165
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
  "ACCELBLOCK", "MEMCRITICAL", "REDUCTIONTARGET", "CASE", "';'", "':'",
  "'{'", "'}'", "','", "'<'", "'>'", "'*'", "'('", "')'", "'&'", "'['",
  "']'", "'='", "'-'", "'.'", "$accept", "File", "ModuleEList",
  "OptExtern", "OptSemiColon", "Name", "QualName", "Module",
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
  "OptSdagCode", "Slist", "Olist", "CaseList", "OptPubList",
  "PublishesList", "OptTraceName", "WhenConstruct", "NonWhenConstruct",
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
     325,   326,   327,   328,   329,    59,    58,   123,   125,    44,
      60,    62,    42,    40,    41,    38,    91,    93,    61,    45,
      46
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    91,    92,    93,    93,    94,    94,    95,    95,    96,
      97,    97,    98,    98,    99,    99,   100,   100,   101,   101,
     101,   101,   101,   102,   102,   102,   102,   102,   102,   102,
     102,   102,   102,   102,   102,   102,   102,   103,   103,   103,
     104,   104,   105,   105,   106,   106,   107,   107,   107,   107,
     107,   107,   107,   107,   107,   107,   107,   107,   107,   107,
     107,   108,   109,   110,   110,   111,   112,   112,   113,   114,
     114,   114,   114,   114,   114,   115,   115,   116,   116,   117,
     118,   118,   119,   120,   121,   121,   122,   122,   123,   123,
     124,   124,   125,   125,   126,   126,   127,   127,   128,   129,
     129,   130,   130,   131,   131,   132,   132,   133,   133,   134,
     135,   135,   136,   136,   137,   137,   138,   138,   139,   139,
     140,   141,   142,   142,   143,   143,   144,   144,   145,   146,
     147,   148,   148,   149,   149,   150,   150,   150,   151,   151,
     151,   152,   152,   153,   154,   154,   154,   154,   154,   155,
     155,   156,   156,   157,   157,   157,   157,   157,   157,   157,
     158,   158,   158,   158,   158,   159,   159,   159,   159,   160,
     160,   161,   162,   162,   163,   163,   163,   164,   164,   164,
     164,   164,   164,   164,   164,   164,   164,   164,   165,   165,
     165,   166,   166,   167,   167,   168,   168,   168,   169,   169,
     170,   170,   170,   170,   170,   170,   170,   170,   170,   170,
     170,   170,   170,   170,   170,   170,   171,   171,   171,   172,
     172,   172,   173,   173,   173,   173,   173,   173,   174,   175,
     176,   177,   177,   177,   177,   178,   178,   178,   179,   179,
     179,   179,   179,   179,   180,   181,   181,   181,   182,   182,
     183,   183,   184,   184,   185,   185,   186,   186,   187,   187,
     187,   188,   188,   189,   189,   190,   190,   190,   191,   191,
     192,   192,   193,   193,   194,   194,   194,   195,   195,   195,
     195,   195,   195,   195,   195,   196,   196,   196,   196,   196,
     196,   196,   196,   196,   196,   196,   196,   196,   196,   196,
     196,   197,   197,   197,   198,   198,   199,   200,   201,   201,
     202,   202,   203,   204,   205,   206
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
       3,     1,     2,     1,     2,     1,     2,     1,     0,     4,
       1,     3,     1,     0,     4,     3,     5,     1,     1,     1,
       1,     1,     1,     1,     1,     6,     8,     4,     1,     4,
      11,     9,    12,    14,     6,     8,     5,     7,     3,     3,
       1,     0,     2,     4,     1,     3,     1,     1,     2,     5,
       1,     3,     1,     1,     2,     2
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       3,     0,     0,     0,     2,     3,     9,     0,     0,     1,
       4,    14,     0,    12,    13,    36,     6,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    35,    33,    34,     0,
       0,     0,    10,    19,   314,   315,   192,   229,   222,     0,
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
     216,   218,   233,     0,   199,   300,     0,     0,     0,     0,
       0,   273,     0,     0,     0,     0,     0,   222,   189,   288,
     259,   256,     0,   307,   222,     0,   222,     0,   310,     0,
       0,   272,     0,   304,     0,   222,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   312,   308,   222,
       0,     0,   275,     0,     0,   222,     0,   298,     0,     0,
     280,   281,   283,   279,   277,   284,   282,   278,     0,   265,
     267,   260,   262,   299,     0,   188,     0,     0,   222,     0,
     306,     0,     0,   311,   274,     0,   287,   264,     0,   305,
       0,     0,   289,   266,   257,   235,   236,   237,   255,     0,
       0,   250,     0,   222,     0,   222,     0,   296,   313,     0,
     276,   268,     0,   301,     0,     0,     0,     0,   254,     0,
     222,     0,     0,   309,     0,   285,     0,     0,   294,   222,
       0,     0,   222,     0,   251,     0,     0,   222,   297,     0,
     301,     0,   302,     0,   238,     0,     0,     0,     0,   190,
       0,     0,   270,     0,   295,     0,   286,   246,     0,     0,
       0,     0,     0,   244,     0,     0,   291,   222,     0,   269,
     303,     0,     0,     0,     0,   240,     0,   247,     0,     0,
     271,   243,   242,   241,   239,   245,   290,     0,     0,   292,
       0,   293
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
     432,   225,   101,   359,   447,   163,   360,   530,   575,   563,
     531,   361,   532,   324,   507,   472,   448,   468,   483,   498,
     555,   583,   462,   449,   500,   469,   558,   464,   511,   454,
     458,   459,   479,   539,    27,    28
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -474
static const yytype_int16 yypact[] =
{
     116,    18,    18,    51,  -474,   116,  -474,    17,    17,  -474,
    -474,  -474,   539,  -474,  -474,  -474,    48,    18,    93,    18,
      18,   161,   720,    25,   741,   539,  -474,  -474,  -474,   274,
      57,    85,  -474,    87,  -474,  -474,  -474,  -474,   -27,   195,
     149,   149,    -5,    85,    90,    90,    90,    90,   101,   109,
      18,   143,   135,   539,  -474,  -474,  -474,  -474,  -474,  -474,
    -474,  -474,   433,  -474,  -474,  -474,  -474,   152,  -474,  -474,
    -474,  -474,  -474,  -474,  -474,  -474,  -474,  -474,  -474,  -474,
    -474,  -474,   163,  -474,   -14,  -474,  -474,  -474,  -474,   185,
      30,  -474,  -474,   216,  -474,    85,   539,    87,   254,    35,
     -27,   256,   852,  -474,   839,   216,   260,   277,  -474,   -13,
      85,  -474,    85,    85,   264,    85,   299,  -474,     3,    18,
      18,    18,    18,   138,   294,   295,   131,    18,  -474,  -474,
    -474,   469,   304,    90,    90,    90,    90,   294,   109,  -474,
    -474,  -474,  -474,  -474,  -474,  -474,  -474,  -474,  -474,   142,
    -474,  -474,   815,  -474,  -474,    18,   306,   331,   -27,   329,
     -27,   310,  -474,   313,   318,    -4,  -474,  -474,  -474,   320,
    -474,    14,   -17,   141,   316,   224,    85,  -474,  -474,   319,
     332,   341,   333,   333,   333,   333,  -474,    18,   323,   355,
     348,   298,    18,   387,    18,  -474,  -474,   353,   362,   365,
      18,    94,    18,   372,   371,   152,    18,    18,    18,    18,
      18,    18,  -474,  -474,  -474,  -474,   374,  -474,   375,  -474,
     341,  -474,  -474,   377,   378,   376,   380,   -27,  -474,    18,
      18,   309,   379,  -474,   149,   815,   149,   149,   815,   149,
    -474,  -474,     3,  -474,    85,   243,   243,   243,   243,   381,
    -474,   387,  -474,   333,   333,  -474,   131,   447,   385,   265,
    -474,   386,   469,  -474,  -474,   333,   333,   333,   333,   333,
     248,   815,  -474,   389,   -27,   329,   -27,   -27,  -474,  -474,
     399,  -474,    87,   382,  -474,   401,   405,   413,    85,   418,
     416,  -474,   423,  -474,  -474,   562,  -474,  -474,  -474,  -474,
    -474,  -474,   243,   243,  -474,  -474,   839,    11,   425,   839,
    -474,  -474,  -474,  -474,  -474,   243,   243,   243,   243,   243,
    -474,   447,  -474,   800,  -474,  -474,  -474,  -474,  -474,   424,
    -474,  -474,   431,  -474,    15,   432,  -474,    85,   178,   467,
     430,  -474,   562,   818,  -474,  -474,  -474,    18,  -474,  -474,
    -474,  -474,  -474,  -474,  -474,  -474,   438,  -474,    18,   -27,
     439,   436,   839,   149,   149,   149,  -474,  -474,   736,   873,
    -474,   152,  -474,  -474,  -474,   441,   442,    -3,   434,   839,
    -474,   445,   453,   454,   455,  -474,  -474,  -474,  -474,  -474,
    -474,  -474,  -474,  -474,  -474,  -474,  -474,   462,  -474,   475,
    -474,  -474,   476,   443,   458,   389,    18,  -474,   478,   481,
    -474,  -474,   246,  -474,  -474,  -474,  -474,  -474,  -474,  -474,
    -474,  -474,   531,  -474,   750,   384,   389,  -474,  -474,  -474,
    -474,    87,  -474,    18,  -474,  -474,   484,   487,   484,   519,
     498,   521,   527,   484,   493,   506,    83,   -27,  -474,  -474,
    -474,   565,   389,  -474,   -27,   540,   -27,    91,   507,   270,
     449,  -474,   517,   518,   529,   -27,   541,   373,   524,   330,
     256,   520,   384,   523,   530,   525,   522,  -474,  -474,   -27,
     519,   250,  -474,   533,   435,   -27,   527,  -474,   522,   389,
    -474,  -474,  -474,  -474,  -474,  -474,  -474,  -474,   535,   373,
    -474,  -474,  -474,  -474,   551,  -474,   219,   517,   -27,   484,
    -474,   578,   528,  -474,  -474,   536,  -474,  -474,   256,  -474,
     588,   542,  -474,  -474,  -474,  -474,  -474,  -474,  -474,    18,
     560,   568,   563,   -27,   576,   -27,    83,  -474,  -474,   389,
    -474,   603,    83,   606,   517,   577,   839,   779,  -474,   256,
     -27,   580,   581,  -474,   575,  -474,   582,   599,  -474,   -27,
      18,    18,   -27,   583,  -474,    18,   522,   -27,  -474,   609,
     606,    83,  -474,   589,  -474,    81,    53,   574,    18,  -474,
     637,   587,   590,   584,  -474,   592,  -474,  -474,    18,   311,
     591,    18,    18,  -474,   102,    83,  -474,   -27,   609,  -474,
    -474,   266,   598,   189,    18,  -474,   133,  -474,   593,   522,
    -474,  -474,  -474,  -474,  -474,  -474,  -474,   648,    83,  -474,
     596,  -474
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -474,  -474,   670,  -474,  -199,    -1,   -11,   665,   688,    21,
    -474,  -474,  -474,  -185,  -474,  -140,  -474,   -85,   -32,   -24,
     -21,  -474,  -119,   594,   -36,  -474,  -474,   468,  -474,  -474,
      -2,   561,   446,  -474,   -20,   459,  -474,  -474,   579,   452,
    -474,   328,  -474,  -474,  -260,  -474,  -116,   369,  -474,  -474,
    -474,   -66,  -474,  -474,  -474,  -474,  -474,  -474,  -474,   448,
    -474,   450,   685,  -474,   -50,   366,   691,  -474,  -474,   534,
    -474,  -474,  -474,   397,   402,  -474,   346,  -474,   292,  -474,
    -474,   444,   -96,   171,   -19,  -441,  -474,  -474,  -433,  -474,
    -474,  -324,   173,  -392,  -474,  -474,   249,  -460,  -474,   244,
    -474,   120,  -474,  -424,  -474,  -415,   174,   259,  -473,  -421,
    -474,   278,  -474,  -474,  -474,  -474
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -264
static const yytype_int16 yytable[] =
{
       7,     8,    38,   110,   161,    93,   264,    33,    94,   502,
     450,   116,   201,   425,   305,   520,    30,   456,    34,    35,
      97,   515,   465,   168,   517,   120,   121,   122,   305,   503,
     111,    99,   168,   177,   451,   183,   184,   185,   381,   113,
     115,   243,   199,   499,   482,   484,    81,   348,   146,   128,
     286,     9,   147,   289,   178,   415,   100,   450,   192,    98,
     473,   356,   223,   155,   226,   478,   234,   246,   247,   248,
     169,   210,   170,     6,   132,   499,   552,   541,   164,   169,
     273,   114,   556,   411,   435,   412,   322,    29,   535,  -110,
      98,    98,    11,   580,    12,   202,   537,   521,   364,   171,
     231,   172,   173,    67,   175,   543,    98,   253,   565,   254,
     152,   585,   158,   206,   207,   208,   209,   156,   159,     1,
       2,   160,   188,   436,   437,   438,   439,   440,   441,   442,
     443,   278,   444,   592,    96,   608,   617,   302,   303,   411,
      32,  -229,   572,   201,   116,   594,    31,   553,    32,   315,
     316,   317,   318,   319,   220,   601,   603,   445,   620,   606,
      37,  -229,   587,    98,   588,   596,  -229,   589,   195,   196,
     590,   591,   407,  -135,   323,  -135,   118,   477,   325,   367,
     327,   328,   259,   607,   111,   588,   249,   123,   589,   186,
     188,   590,   591,     6,   187,   126,   297,   298,   299,   258,
     129,   261,   619,   212,   213,   265,   266,   267,   268,   269,
     270,   102,   292,  -195,   615,   131,   588,    98,  -193,   589,
     282,   235,   590,   591,   236,   525,   202,   144,   279,   280,
     103,   104,   285,  -195,   287,   288,    36,   290,    37,  -195,
    -195,  -195,  -195,  -195,  -195,  -195,   148,   149,   150,   151,
      32,   435,   345,   346,   103,   104,    83,    84,    85,    86,
      87,    88,    89,   378,   368,   351,   352,   353,   354,   355,
     347,   435,   588,   350,    32,   589,   613,   334,   590,   591,
      83,    84,    85,    86,    87,    88,    89,   358,   526,   527,
     436,   437,   438,   439,   440,   441,   442,   443,   154,   444,
      98,    32,   429,   430,   238,   292,   528,   239,  -229,    82,
     436,   437,   438,   439,   440,   441,   442,   443,   294,   444,
     295,   311,   312,   320,   445,   321,   358,    37,   514,    32,
     157,   435,   174,  -229,   162,    83,    84,    85,    86,    87,
      88,    89,   166,   358,   445,    93,   375,   481,    94,   588,
     611,   470,   589,     6,   187,   590,   591,   377,   474,   167,
     476,   382,   383,   384,    32,   281,     6,   602,   405,   488,
     436,   437,   438,   439,   440,   441,   442,   443,   176,   444,
     191,   193,   205,   512,   221,   435,   222,   224,   228,   518,
    -258,  -258,  -258,  -258,   227,  -258,  -258,  -258,  -258,  -258,
     229,   431,   230,   237,   445,   426,   241,    37,  -261,   244,
     250,   242,   534,   490,   491,   492,   439,   493,   494,   495,
     496,   152,   497,  -258,   436,   437,   438,   439,   440,   441,
     442,   443,   452,   444,   251,   252,   435,   549,   186,   551,
     255,   256,   257,   485,   133,   134,   135,   136,   137,   138,
     435,   262,   263,   271,   566,   274,   272,   275,   445,  -258,
     276,   446,  -258,   573,   305,   231,   577,   277,   300,   330,
     529,   581,   323,   309,   259,   436,   437,   438,   439,   440,
     441,   442,   443,   329,   444,   331,   332,   200,   533,   436,
     437,   438,   439,   440,   441,   442,   443,   333,   444,   335,
     336,   609,   337,   349,   103,   104,   338,   362,   371,   445,
     561,   529,    37,  -263,   363,   365,   376,   410,   379,   420,
     380,   414,   424,   445,    32,   559,    37,   408,   545,   416,
      83,    84,    85,    86,    87,    88,    89,   417,   418,   419,
      15,    -9,    -5,    -5,    16,    -5,    -5,    -5,    -5,    -5,
      -5,    -5,    -5,    -5,    -5,    -5,   428,    -5,    -5,   574,
     576,    -5,   422,   423,   579,   427,   433,   453,    39,    40,
      41,    42,    43,   455,   457,   460,   466,   574,   461,   435,
      50,    51,   463,   467,    52,   471,   480,   574,   574,   435,
     605,   574,    17,    18,    37,   475,   489,   486,    19,    20,
     435,   338,   501,   614,   487,   508,   510,   524,   504,   506,
      21,   516,   509,   522,   540,   538,    -5,   -16,   436,   437,
     438,   439,   440,   441,   442,   443,   544,   444,   436,   437,
     438,   439,   440,   441,   442,   443,   546,   444,   435,   436,
     437,   438,   439,   440,   441,   442,   443,   547,   444,   435,
     548,   550,   445,   554,   557,   536,   567,   560,   569,   568,
     570,   593,   445,   578,   582,   542,   597,   586,   599,   598,
     600,   616,   604,   445,   621,    10,   571,   436,   437,   438,
     439,   440,   441,   442,   443,   612,   444,    54,   436,   437,
     438,   439,   440,   441,   442,   443,    14,   444,   165,   211,
     284,   291,   304,   301,   194,   413,   366,    62,   372,   313,
     240,   445,   314,    64,   595,   406,   434,   562,   610,   326,
     564,   505,   445,     1,     2,   618,    39,    40,    41,    42,
      43,    44,    45,    46,    47,    48,    49,   385,    50,    51,
     374,   370,    52,   523,   584,   519,     0,    68,    69,    70,
      71,   385,    72,    73,    74,    75,    76,   386,   513,   387,
     388,   389,   390,   391,   392,     0,     0,   393,   394,   395,
     396,   386,     0,   387,   388,   389,   390,   391,   392,     0,
      77,   393,   394,   395,   396,   525,     0,   397,   398,     0,
       0,     0,     0,     0,     0,     0,     0,    53,     0,     0,
       0,   397,   398,     0,   399,     0,     0,     0,   400,   401,
       0,     0,     0,     0,   103,   104,    78,     0,     0,    79,
       0,     0,   400,   401,    68,    69,    70,    71,     0,    72,
      73,    74,    75,    76,    32,   103,   104,     0,     0,     0,
      83,    84,    85,    86,    87,    88,    89,     0,   526,   527,
     103,   104,     0,     0,     0,    32,     0,    77,     0,     0,
       0,    83,    84,    85,    86,    87,    88,    89,     0,     0,
      32,   214,   215,     0,   103,   104,    83,    84,    85,    86,
      87,    88,    89,     0,   357,     0,     0,   103,     0,     0,
       0,     0,     0,   373,    32,     0,    79,     0,     0,     0,
      83,    84,    85,    86,    87,    88,    89,    32,    82,     0,
       0,     0,     0,    83,    84,    85,    86,    87,    88,    89,
       0,     0,     0,     0,     0,     0,     0,     0,   404,     0,
       0,     0,     0,     0,    83,    84,    85,    86,    87,    88,
      89
};

static const yytype_int16 yycheck[] =
{
       1,     2,    21,    39,   100,    29,   205,    18,    29,   469,
     425,    43,   131,   405,    17,   488,    17,   438,    19,    20,
      31,   481,   443,    36,   484,    45,    46,    47,    17,   470,
      35,    58,    36,    30,   426,   120,   121,   122,   362,    41,
      42,   181,   127,   467,   459,   460,    25,   307,    62,    50,
     235,     0,    66,   238,    51,   379,    83,   472,   124,    76,
     452,   321,   158,    95,   160,   457,    83,   183,   184,   185,
      83,   137,    85,    55,    53,   499,   536,   518,   102,    83,
     220,    86,   542,    86,     1,    88,   271,    39,   509,    78,
      76,    76,    75,   566,    77,   131,   511,   489,    83,   110,
      86,   112,   113,    78,   115,   520,    76,   192,   549,   194,
      80,   571,    77,   133,   134,   135,   136,    96,    83,     3,
       4,    86,   123,    40,    41,    42,    43,    44,    45,    46,
      47,   227,    49,    80,    77,   595,   609,   253,   254,    86,
      55,    58,   557,   262,   176,   578,    53,   539,    55,   265,
     266,   267,   268,   269,   155,   588,   589,    74,   618,   592,
      77,    78,    81,    76,    83,   580,    83,    86,    37,    38,
      89,    90,   371,    79,    83,    81,    86,    86,   274,     1,
     276,   277,    88,    81,    35,    83,   187,    86,    86,    51,
     191,    89,    90,    55,    56,    86,   246,   247,   248,   200,
      57,   202,   617,    61,    62,   206,   207,   208,   209,   210,
     211,    16,   244,    35,    81,    80,    83,    76,    55,    86,
     231,    80,    89,    90,    83,     6,   262,    75,   229,   230,
      35,    36,   234,    55,   236,   237,    75,   239,    77,    61,
      62,    63,    64,    65,    66,    67,    61,    62,    63,    64,
      55,     1,   302,   303,    35,    36,    61,    62,    63,    64,
      65,    66,    67,   359,    86,   315,   316,   317,   318,   319,
     306,     1,    83,   309,    55,    86,    87,   288,    89,    90,
      61,    62,    63,    64,    65,    66,    67,   323,    69,    70,
      40,    41,    42,    43,    44,    45,    46,    47,    82,    49,
      76,    55,    56,    57,    80,   337,    87,    83,    58,    35,
      40,    41,    42,    43,    44,    45,    46,    47,    75,    49,
      77,    56,    57,    75,    74,    77,   362,    77,    78,    55,
      76,     1,    68,    83,    78,    61,    62,    63,    64,    65,
      66,    67,    82,   379,    74,   369,   347,    77,   369,    83,
      84,   447,    86,    55,    56,    89,    90,   358,   454,    82,
     456,   363,   364,   365,    55,    56,    55,    56,   369,   465,
      40,    41,    42,    43,    44,    45,    46,    47,    79,    49,
      86,    86,    78,   479,    78,     1,    55,    58,    75,   485,
       6,     7,     8,     9,    84,    11,    12,    13,    14,    15,
      82,   412,    82,    87,    74,   406,    87,    77,    78,    76,
      87,    79,   508,    40,    41,    42,    43,    44,    45,    46,
      47,    80,    49,    39,    40,    41,    42,    43,    44,    45,
      46,    47,   433,    49,    79,    87,     1,   533,    51,   535,
      87,    79,    77,   462,    11,    12,    13,    14,    15,    16,
       1,    79,    81,    79,   550,    78,    81,    79,    74,    75,
      84,    77,    78,   559,    17,    86,   562,    87,    87,    87,
     506,   567,    83,    88,    88,    40,    41,    42,    43,    44,
      45,    46,    47,    84,    49,    84,    81,    18,   507,    40,
      41,    42,    43,    44,    45,    46,    47,    84,    49,    81,
      84,   597,    79,    78,    35,    36,    39,    83,    78,    74,
     546,   547,    77,    78,    83,    83,    78,    75,    79,    57,
      84,    87,    79,    74,    55,   544,    77,    86,   529,    84,
      61,    62,    63,    64,    65,    66,    67,    84,    84,    84,
       1,    83,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    75,    18,    19,   560,
     561,    22,    87,    87,   565,    87,    35,    83,     6,     7,
       8,     9,    10,    86,    55,    77,    83,   578,    57,     1,
      18,    19,    55,    77,    22,    20,    79,   588,   589,     1,
     591,   592,    53,    54,    77,    55,    55,    79,    59,    60,
       1,    39,    78,   604,    75,    75,    84,    56,    88,    86,
      71,    78,    87,    78,    78,    87,    77,    78,    40,    41,
      42,    43,    44,    45,    46,    47,    84,    49,    40,    41,
      42,    43,    44,    45,    46,    47,    76,    49,     1,    40,
      41,    42,    43,    44,    45,    46,    47,    79,    49,     1,
      87,    75,    74,    50,    48,    77,    76,    80,    83,    78,
      78,    87,    74,    80,    55,    77,    79,    78,    84,    79,
      78,    78,    81,    74,    78,     5,    77,    40,    41,    42,
      43,    44,    45,    46,    47,    87,    49,    22,    40,    41,
      42,    43,    44,    45,    46,    47,     8,    49,   104,   138,
     232,   242,   256,   251,   125,   377,   337,    22,   342,   261,
     176,    74,   262,    22,    77,   369,   424,   546,   598,   275,
     547,   472,    74,     3,     4,    77,     6,     7,     8,     9,
      10,    11,    12,    13,    14,    15,    16,     1,    18,    19,
     343,   339,    22,   499,   570,   486,    -1,     6,     7,     8,
       9,     1,    11,    12,    13,    14,    15,    21,   480,    23,
      24,    25,    26,    27,    28,    -1,    -1,    31,    32,    33,
      34,    21,    -1,    23,    24,    25,    26,    27,    28,    -1,
      39,    31,    32,    33,    34,     6,    -1,    51,    52,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    77,    -1,    -1,
      -1,    51,    52,    -1,    68,    -1,    -1,    -1,    72,    73,
      -1,    -1,    -1,    -1,    35,    36,    75,    -1,    -1,    78,
      -1,    -1,    72,    73,     6,     7,     8,     9,    -1,    11,
      12,    13,    14,    15,    55,    35,    36,    -1,    -1,    -1,
      61,    62,    63,    64,    65,    66,    67,    -1,    69,    70,
      35,    36,    -1,    -1,    -1,    55,    -1,    39,    -1,    -1,
      -1,    61,    62,    63,    64,    65,    66,    67,    -1,    -1,
      55,    56,    57,    -1,    35,    36,    61,    62,    63,    64,
      65,    66,    67,    -1,    84,    -1,    -1,    35,    -1,    -1,
      -1,    -1,    -1,    75,    55,    -1,    78,    -1,    -1,    -1,
      61,    62,    63,    64,    65,    66,    67,    55,    35,    -1,
      -1,    -1,    -1,    61,    62,    63,    64,    65,    66,    67,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    55,    -1,
      -1,    -1,    -1,    -1,    61,    62,    63,    64,    65,    66,
      67
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     3,     4,    92,    93,    98,    55,    96,    96,     0,
      93,    75,    77,    99,    99,     1,     5,    53,    54,    59,
      60,    71,    94,   100,   101,   102,   166,   205,   206,    39,
      96,    53,    55,    97,    96,    96,    75,    77,   175,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      18,    19,    22,    77,    98,   119,   120,   136,   139,   140,
     141,   143,   153,   154,   157,   158,   159,    78,     6,     7,
       8,     9,    11,    12,    13,    14,    15,    39,    75,    78,
     164,   100,    35,    61,    62,    63,    64,    65,    66,    67,
      97,   107,   109,   110,   111,   167,    77,    97,    76,    58,
      83,   173,    16,    35,    36,   110,   111,   112,   113,   114,
     115,    35,   121,   121,    86,   121,   109,   160,    86,   125,
     125,   125,   125,    86,   129,   142,    86,   122,    96,    57,
     161,    80,   100,    11,    12,    13,    14,    15,    16,   144,
     145,   146,   147,   148,    75,    95,    62,    66,    61,    62,
      63,    64,    80,   106,    82,   109,   100,    76,    77,    83,
      86,   173,    78,   176,   110,   114,    82,    82,    36,    83,
      85,    97,    97,    97,    68,    97,    79,    30,    51,   126,
     131,    96,   108,   108,   108,   108,    51,    56,    96,   128,
     130,    86,   142,    86,   129,    37,    38,   123,   124,   108,
      18,   113,   115,   151,   152,    78,   125,   125,   125,   125,
     142,   122,    61,    62,    56,    57,   103,   104,   105,   115,
      96,    78,    55,   173,    58,   172,   173,    84,    75,    82,
      82,    86,   117,   118,    83,    80,    83,    87,    80,    83,
     160,    87,    79,   106,    76,   137,   137,   137,   137,    96,
      87,    79,    87,   108,   108,    87,    79,    77,    96,    88,
     150,    96,    79,    81,    95,    96,    96,    96,    96,    96,
      96,    79,    81,   106,    78,    79,    84,    87,   173,    96,
      96,    56,    97,   116,   118,   121,   104,   121,   121,   104,
     121,   126,   109,   138,    75,    77,   155,   155,   155,   155,
      87,   130,   137,   137,   123,    17,   132,   134,   135,    88,
     149,    56,    57,   150,   152,   137,   137,   137,   137,   137,
      75,    77,   104,    83,   184,   173,   172,   173,   173,    84,
      87,    84,    81,    84,    97,    81,    84,    79,    39,   153,
     156,   157,   162,   163,   165,   155,   155,   115,   135,    78,
     115,   155,   155,   155,   155,   155,   135,    84,   115,   174,
     177,   182,    83,    83,    83,    83,   138,     1,    86,   168,
     165,    78,   156,    75,   164,    96,    78,    96,   173,    79,
      84,   182,   121,   121,   121,     1,    21,    23,    24,    25,
      26,    27,    28,    31,    32,    33,    34,    51,    52,    68,
      72,    73,   169,   170,    55,    96,   167,    95,    86,   133,
      75,    86,    88,   132,    87,   182,    84,    84,    84,    84,
      57,   127,    87,    87,    79,   184,    96,    87,    75,    56,
      57,    97,   171,    35,   169,     1,    40,    41,    42,    43,
      44,    45,    46,    47,    49,    74,    77,   175,   187,   194,
     196,   184,    96,    83,   200,    86,   200,    55,   201,   202,
      77,    57,   193,    55,   198,   200,    83,    77,   188,   196,
     173,    20,   186,   184,   173,    55,   173,    86,   184,   203,
      79,    77,   196,   189,   196,   175,    79,    75,   173,    55,
      40,    41,    42,    44,    45,    46,    47,    49,   190,   194,
     195,    78,   188,   176,    88,   187,    86,   185,    75,    87,
      84,   199,   173,   202,    78,   188,    78,   188,   173,   198,
     199,   184,    78,   190,    56,     6,    69,    70,    87,   115,
     178,   181,   183,   175,   173,   200,    77,   196,    87,   204,
      78,   176,    77,   196,    84,    96,    76,    79,    87,   173,
      75,   173,   188,   184,    50,   191,   188,    48,   197,   175,
      80,   115,   174,   180,   183,   176,   173,    76,    78,    83,
      78,    77,   196,   173,    96,   179,    96,   173,    80,    96,
     199,   173,    55,   192,   197,   188,    78,    81,    83,    86,
      89,    90,    80,    87,   179,    77,   196,    79,    79,    84,
      78,   179,    56,   179,    81,    96,   179,    81,   188,   173,
     192,    84,    87,    87,    96,    81,    78,   199,    77,   196,
     188,    78
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
#line 158 "xi-grammar.y"
    { (yyval.modlist) = (yyvsp[(1) - (1)].modlist); modlist = (yyvsp[(1) - (1)].modlist); }
    break;

  case 3:

/* Line 1455 of yacc.c  */
#line 162 "xi-grammar.y"
    { 
		  (yyval.modlist) = 0; 
		}
    break;

  case 4:

/* Line 1455 of yacc.c  */
#line 166 "xi-grammar.y"
    { (yyval.modlist) = new ModuleList(lineno, (yyvsp[(1) - (2)].module), (yyvsp[(2) - (2)].modlist)); }
    break;

  case 5:

/* Line 1455 of yacc.c  */
#line 170 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 6:

/* Line 1455 of yacc.c  */
#line 172 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 7:

/* Line 1455 of yacc.c  */
#line 176 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 8:

/* Line 1455 of yacc.c  */
#line 178 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 9:

/* Line 1455 of yacc.c  */
#line 182 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 10:

/* Line 1455 of yacc.c  */
#line 186 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 11:

/* Line 1455 of yacc.c  */
#line 188 "xi-grammar.y"
    {
		  char *tmp = new char[strlen((yyvsp[(1) - (4)].strval))+strlen((yyvsp[(4) - (4)].strval))+3];
		  sprintf(tmp,"%s::%s", (yyvsp[(1) - (4)].strval), (yyvsp[(4) - (4)].strval));
		  (yyval.strval) = tmp;
		}
    break;

  case 12:

/* Line 1455 of yacc.c  */
#line 196 "xi-grammar.y"
    { 
		    (yyval.module) = new Module(lineno, (yyvsp[(2) - (3)].strval), (yyvsp[(3) - (3)].conslist)); 
		}
    break;

  case 13:

/* Line 1455 of yacc.c  */
#line 200 "xi-grammar.y"
    {  
		    (yyval.module) = new Module(lineno, (yyvsp[(2) - (3)].strval), (yyvsp[(3) - (3)].conslist)); 
		    (yyval.module)->setMain();
		}
    break;

  case 14:

/* Line 1455 of yacc.c  */
#line 207 "xi-grammar.y"
    { (yyval.conslist) = 0; }
    break;

  case 15:

/* Line 1455 of yacc.c  */
#line 209 "xi-grammar.y"
    { (yyval.conslist) = (yyvsp[(2) - (4)].conslist); }
    break;

  case 16:

/* Line 1455 of yacc.c  */
#line 213 "xi-grammar.y"
    { (yyval.conslist) = 0; }
    break;

  case 17:

/* Line 1455 of yacc.c  */
#line 215 "xi-grammar.y"
    { (yyval.conslist) = new ConstructList(lineno, (yyvsp[(1) - (2)].construct), (yyvsp[(2) - (2)].conslist)); }
    break;

  case 18:

/* Line 1455 of yacc.c  */
#line 219 "xi-grammar.y"
    { (yyval.construct) = new UsingScope((yyvsp[(3) - (3)].strval), false); }
    break;

  case 19:

/* Line 1455 of yacc.c  */
#line 221 "xi-grammar.y"
    { (yyval.construct) = new UsingScope((yyvsp[(2) - (2)].strval), true); }
    break;

  case 20:

/* Line 1455 of yacc.c  */
#line 223 "xi-grammar.y"
    { (yyvsp[(2) - (2)].member)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].member); }
    break;

  case 21:

/* Line 1455 of yacc.c  */
#line 225 "xi-grammar.y"
    { (yyvsp[(2) - (2)].message)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].message); }
    break;

  case 22:

/* Line 1455 of yacc.c  */
#line 227 "xi-grammar.y"
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
#line 239 "xi-grammar.y"
    { if((yyvsp[(3) - (5)].conslist)) (yyvsp[(3) - (5)].conslist)->setExtern((yyvsp[(1) - (5)].intval)); (yyval.construct) = (yyvsp[(3) - (5)].conslist); }
    break;

  case 24:

/* Line 1455 of yacc.c  */
#line 241 "xi-grammar.y"
    { (yyval.construct) = new Scope((yyvsp[(2) - (5)].strval), (yyvsp[(4) - (5)].conslist)); }
    break;

  case 25:

/* Line 1455 of yacc.c  */
#line 243 "xi-grammar.y"
    { (yyval.construct) = (yyvsp[(1) - (2)].construct); }
    break;

  case 26:

/* Line 1455 of yacc.c  */
#line 245 "xi-grammar.y"
    { yyerror("The preceding construct must be semicolon terminated"); YYABORT; }
    break;

  case 27:

/* Line 1455 of yacc.c  */
#line 247 "xi-grammar.y"
    { (yyvsp[(2) - (2)].module)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].module); }
    break;

  case 28:

/* Line 1455 of yacc.c  */
#line 249 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 29:

/* Line 1455 of yacc.c  */
#line 251 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 30:

/* Line 1455 of yacc.c  */
#line 253 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 31:

/* Line 1455 of yacc.c  */
#line 255 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 32:

/* Line 1455 of yacc.c  */
#line 257 "xi-grammar.y"
    { (yyvsp[(2) - (2)].templat)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].templat); }
    break;

  case 33:

/* Line 1455 of yacc.c  */
#line 259 "xi-grammar.y"
    { (yyval.construct) = NULL; }
    break;

  case 34:

/* Line 1455 of yacc.c  */
#line 261 "xi-grammar.y"
    { (yyval.construct) = NULL; }
    break;

  case 35:

/* Line 1455 of yacc.c  */
#line 263 "xi-grammar.y"
    { (yyval.construct) = (yyvsp[(1) - (1)].accelBlock); }
    break;

  case 36:

/* Line 1455 of yacc.c  */
#line 265 "xi-grammar.y"
    { printf("Invalid construct\n"); YYABORT; }
    break;

  case 37:

/* Line 1455 of yacc.c  */
#line 269 "xi-grammar.y"
    { (yyval.tparam) = new TParamType((yyvsp[(1) - (1)].type)); }
    break;

  case 38:

/* Line 1455 of yacc.c  */
#line 271 "xi-grammar.y"
    { (yyval.tparam) = new TParamVal((yyvsp[(1) - (1)].strval)); }
    break;

  case 39:

/* Line 1455 of yacc.c  */
#line 273 "xi-grammar.y"
    { (yyval.tparam) = new TParamVal((yyvsp[(1) - (1)].strval)); }
    break;

  case 40:

/* Line 1455 of yacc.c  */
#line 277 "xi-grammar.y"
    { (yyval.tparlist) = new TParamList((yyvsp[(1) - (1)].tparam)); }
    break;

  case 41:

/* Line 1455 of yacc.c  */
#line 279 "xi-grammar.y"
    { (yyval.tparlist) = new TParamList((yyvsp[(1) - (3)].tparam), (yyvsp[(3) - (3)].tparlist)); }
    break;

  case 42:

/* Line 1455 of yacc.c  */
#line 283 "xi-grammar.y"
    { (yyval.tparlist) = 0; }
    break;

  case 43:

/* Line 1455 of yacc.c  */
#line 285 "xi-grammar.y"
    { (yyval.tparlist) = (yyvsp[(1) - (1)].tparlist); }
    break;

  case 44:

/* Line 1455 of yacc.c  */
#line 289 "xi-grammar.y"
    { (yyval.tparlist) = 0; }
    break;

  case 45:

/* Line 1455 of yacc.c  */
#line 291 "xi-grammar.y"
    { (yyval.tparlist) = (yyvsp[(2) - (3)].tparlist); }
    break;

  case 46:

/* Line 1455 of yacc.c  */
#line 295 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("int"); }
    break;

  case 47:

/* Line 1455 of yacc.c  */
#line 297 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long"); }
    break;

  case 48:

/* Line 1455 of yacc.c  */
#line 299 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("short"); }
    break;

  case 49:

/* Line 1455 of yacc.c  */
#line 301 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("char"); }
    break;

  case 50:

/* Line 1455 of yacc.c  */
#line 303 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned int"); }
    break;

  case 51:

/* Line 1455 of yacc.c  */
#line 305 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long"); }
    break;

  case 52:

/* Line 1455 of yacc.c  */
#line 307 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long"); }
    break;

  case 53:

/* Line 1455 of yacc.c  */
#line 309 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long long"); }
    break;

  case 54:

/* Line 1455 of yacc.c  */
#line 311 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned short"); }
    break;

  case 55:

/* Line 1455 of yacc.c  */
#line 313 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned char"); }
    break;

  case 56:

/* Line 1455 of yacc.c  */
#line 315 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long long"); }
    break;

  case 57:

/* Line 1455 of yacc.c  */
#line 317 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("float"); }
    break;

  case 58:

/* Line 1455 of yacc.c  */
#line 319 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("double"); }
    break;

  case 59:

/* Line 1455 of yacc.c  */
#line 321 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long double"); }
    break;

  case 60:

/* Line 1455 of yacc.c  */
#line 323 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("void"); }
    break;

  case 61:

/* Line 1455 of yacc.c  */
#line 326 "xi-grammar.y"
    { (yyval.ntype) = new NamedType((yyvsp[(1) - (2)].strval),(yyvsp[(2) - (2)].tparlist)); }
    break;

  case 62:

/* Line 1455 of yacc.c  */
#line 327 "xi-grammar.y"
    { 
                    const char* basename, *scope;
                    splitScopedName((yyvsp[(1) - (2)].strval), &scope, &basename);
                    (yyval.ntype) = new NamedType(basename, (yyvsp[(2) - (2)].tparlist), scope);
                }
    break;

  case 63:

/* Line 1455 of yacc.c  */
#line 335 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 64:

/* Line 1455 of yacc.c  */
#line 337 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ntype); }
    break;

  case 65:

/* Line 1455 of yacc.c  */
#line 341 "xi-grammar.y"
    { (yyval.ptype) = new PtrType((yyvsp[(1) - (2)].type)); }
    break;

  case 66:

/* Line 1455 of yacc.c  */
#line 345 "xi-grammar.y"
    { (yyvsp[(1) - (2)].ptype)->indirect(); (yyval.ptype) = (yyvsp[(1) - (2)].ptype); }
    break;

  case 67:

/* Line 1455 of yacc.c  */
#line 347 "xi-grammar.y"
    { (yyvsp[(1) - (2)].ptype)->indirect(); (yyval.ptype) = (yyvsp[(1) - (2)].ptype); }
    break;

  case 68:

/* Line 1455 of yacc.c  */
#line 351 "xi-grammar.y"
    { (yyval.ftype) = new FuncType((yyvsp[(1) - (8)].type), (yyvsp[(4) - (8)].strval), (yyvsp[(7) - (8)].plist)); }
    break;

  case 69:

/* Line 1455 of yacc.c  */
#line 355 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 70:

/* Line 1455 of yacc.c  */
#line 357 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 71:

/* Line 1455 of yacc.c  */
#line 359 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 72:

/* Line 1455 of yacc.c  */
#line 361 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ftype); }
    break;

  case 73:

/* Line 1455 of yacc.c  */
#line 364 "xi-grammar.y"
    { (yyval.type) = new ConstType((yyvsp[(2) - (2)].type)); }
    break;

  case 74:

/* Line 1455 of yacc.c  */
#line 366 "xi-grammar.y"
    { (yyval.type) = new ConstType((yyvsp[(1) - (2)].type)); }
    break;

  case 75:

/* Line 1455 of yacc.c  */
#line 370 "xi-grammar.y"
    { (yyval.type) = new ReferenceType((yyvsp[(1) - (2)].type)); }
    break;

  case 76:

/* Line 1455 of yacc.c  */
#line 372 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 77:

/* Line 1455 of yacc.c  */
#line 376 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 78:

/* Line 1455 of yacc.c  */
#line 378 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 79:

/* Line 1455 of yacc.c  */
#line 382 "xi-grammar.y"
    { (yyval.val) = (yyvsp[(2) - (3)].val); }
    break;

  case 80:

/* Line 1455 of yacc.c  */
#line 386 "xi-grammar.y"
    { (yyval.vallist) = 0; }
    break;

  case 81:

/* Line 1455 of yacc.c  */
#line 388 "xi-grammar.y"
    { (yyval.vallist) = new ValueList((yyvsp[(1) - (2)].val), (yyvsp[(2) - (2)].vallist)); }
    break;

  case 82:

/* Line 1455 of yacc.c  */
#line 392 "xi-grammar.y"
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[(2) - (4)].type), (yyvsp[(3) - (4)].strval), (yyvsp[(4) - (4)].vallist)); }
    break;

  case 83:

/* Line 1455 of yacc.c  */
#line 396 "xi-grammar.y"
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[(3) - (5)].type), (yyvsp[(5) - (5)].strval), 0, 1); }
    break;

  case 84:

/* Line 1455 of yacc.c  */
#line 400 "xi-grammar.y"
    { (yyval.intval) = 0;}
    break;

  case 85:

/* Line 1455 of yacc.c  */
#line 402 "xi-grammar.y"
    { (yyval.intval) = 0;}
    break;

  case 86:

/* Line 1455 of yacc.c  */
#line 406 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 87:

/* Line 1455 of yacc.c  */
#line 408 "xi-grammar.y"
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
#line 418 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (1)].intval); }
    break;

  case 89:

/* Line 1455 of yacc.c  */
#line 420 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (3)].intval) | (yyvsp[(3) - (3)].intval); }
    break;

  case 90:

/* Line 1455 of yacc.c  */
#line 424 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 91:

/* Line 1455 of yacc.c  */
#line 426 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 92:

/* Line 1455 of yacc.c  */
#line 430 "xi-grammar.y"
    { (yyval.cattr) = 0; }
    break;

  case 93:

/* Line 1455 of yacc.c  */
#line 432 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(2) - (3)].cattr); }
    break;

  case 94:

/* Line 1455 of yacc.c  */
#line 436 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (1)].cattr); }
    break;

  case 95:

/* Line 1455 of yacc.c  */
#line 438 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (3)].cattr) | (yyvsp[(3) - (3)].cattr); }
    break;

  case 96:

/* Line 1455 of yacc.c  */
#line 442 "xi-grammar.y"
    { python_doc = NULL; (yyval.intval) = 0; }
    break;

  case 97:

/* Line 1455 of yacc.c  */
#line 444 "xi-grammar.y"
    { python_doc = (yyvsp[(1) - (1)].strval); (yyval.intval) = 0; }
    break;

  case 98:

/* Line 1455 of yacc.c  */
#line 448 "xi-grammar.y"
    { (yyval.cattr) = Chare::CPYTHON; }
    break;

  case 99:

/* Line 1455 of yacc.c  */
#line 452 "xi-grammar.y"
    { (yyval.cattr) = 0; }
    break;

  case 100:

/* Line 1455 of yacc.c  */
#line 454 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(2) - (3)].cattr); }
    break;

  case 101:

/* Line 1455 of yacc.c  */
#line 458 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (1)].cattr); }
    break;

  case 102:

/* Line 1455 of yacc.c  */
#line 460 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (3)].cattr) | (yyvsp[(3) - (3)].cattr); }
    break;

  case 103:

/* Line 1455 of yacc.c  */
#line 464 "xi-grammar.y"
    { (yyval.cattr) = Chare::CMIGRATABLE; }
    break;

  case 104:

/* Line 1455 of yacc.c  */
#line 466 "xi-grammar.y"
    { (yyval.cattr) = Chare::CPYTHON; }
    break;

  case 105:

/* Line 1455 of yacc.c  */
#line 470 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 106:

/* Line 1455 of yacc.c  */
#line 472 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 107:

/* Line 1455 of yacc.c  */
#line 475 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 108:

/* Line 1455 of yacc.c  */
#line 477 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 109:

/* Line 1455 of yacc.c  */
#line 480 "xi-grammar.y"
    { (yyval.mv) = new MsgVar((yyvsp[(2) - (5)].type), (yyvsp[(3) - (5)].strval), (yyvsp[(1) - (5)].intval), (yyvsp[(4) - (5)].intval)); }
    break;

  case 110:

/* Line 1455 of yacc.c  */
#line 484 "xi-grammar.y"
    { (yyval.mvlist) = new MsgVarList((yyvsp[(1) - (1)].mv)); }
    break;

  case 111:

/* Line 1455 of yacc.c  */
#line 486 "xi-grammar.y"
    { (yyval.mvlist) = new MsgVarList((yyvsp[(1) - (2)].mv), (yyvsp[(2) - (2)].mvlist)); }
    break;

  case 112:

/* Line 1455 of yacc.c  */
#line 490 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, (yyvsp[(3) - (3)].ntype)); }
    break;

  case 113:

/* Line 1455 of yacc.c  */
#line 492 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, (yyvsp[(3) - (6)].ntype), (yyvsp[(5) - (6)].mvlist)); }
    break;

  case 114:

/* Line 1455 of yacc.c  */
#line 496 "xi-grammar.y"
    { (yyval.typelist) = 0; }
    break;

  case 115:

/* Line 1455 of yacc.c  */
#line 498 "xi-grammar.y"
    { (yyval.typelist) = (yyvsp[(2) - (2)].typelist); }
    break;

  case 116:

/* Line 1455 of yacc.c  */
#line 502 "xi-grammar.y"
    { (yyval.typelist) = new TypeList((yyvsp[(1) - (1)].ntype)); }
    break;

  case 117:

/* Line 1455 of yacc.c  */
#line 504 "xi-grammar.y"
    { (yyval.typelist) = new TypeList((yyvsp[(1) - (3)].ntype), (yyvsp[(3) - (3)].typelist)); }
    break;

  case 118:

/* Line 1455 of yacc.c  */
#line 508 "xi-grammar.y"
    { (yyval.chare) = new Chare(lineno, (yyvsp[(2) - (5)].cattr)|Chare::CCHARE, (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 119:

/* Line 1455 of yacc.c  */
#line 510 "xi-grammar.y"
    { (yyval.chare) = new MainChare(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 120:

/* Line 1455 of yacc.c  */
#line 514 "xi-grammar.y"
    { (yyval.chare) = new Group(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 121:

/* Line 1455 of yacc.c  */
#line 518 "xi-grammar.y"
    { (yyval.chare) = new NodeGroup(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 122:

/* Line 1455 of yacc.c  */
#line 522 "xi-grammar.y"
    {/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",(yyvsp[(2) - (4)].strval));
			(yyval.ntype) = new NamedType(buf); 
		}
    break;

  case 123:

/* Line 1455 of yacc.c  */
#line 528 "xi-grammar.y"
    { (yyval.ntype) = new NamedType((yyvsp[(2) - (3)].strval)); }
    break;

  case 124:

/* Line 1455 of yacc.c  */
#line 532 "xi-grammar.y"
    {  (yyval.chare) = new Array(lineno, (yyvsp[(2) - (6)].cattr), (yyvsp[(3) - (6)].ntype), (yyvsp[(4) - (6)].ntype), (yyvsp[(5) - (6)].typelist), (yyvsp[(6) - (6)].mbrlist)); }
    break;

  case 125:

/* Line 1455 of yacc.c  */
#line 534 "xi-grammar.y"
    {  (yyval.chare) = new Array(lineno, (yyvsp[(3) - (6)].cattr), (yyvsp[(2) - (6)].ntype), (yyvsp[(4) - (6)].ntype), (yyvsp[(5) - (6)].typelist), (yyvsp[(6) - (6)].mbrlist)); }
    break;

  case 126:

/* Line 1455 of yacc.c  */
#line 538 "xi-grammar.y"
    { (yyval.chare) = new Chare(lineno, (yyvsp[(2) - (5)].cattr)|Chare::CCHARE, new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist));}
    break;

  case 127:

/* Line 1455 of yacc.c  */
#line 540 "xi-grammar.y"
    { (yyval.chare) = new MainChare(lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 128:

/* Line 1455 of yacc.c  */
#line 544 "xi-grammar.y"
    { (yyval.chare) = new Group(lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 129:

/* Line 1455 of yacc.c  */
#line 548 "xi-grammar.y"
    { (yyval.chare) = new NodeGroup( lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 130:

/* Line 1455 of yacc.c  */
#line 552 "xi-grammar.y"
    { (yyval.chare) = new Array( lineno, 0, (yyvsp[(2) - (5)].ntype), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 131:

/* Line 1455 of yacc.c  */
#line 556 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[(3) - (4)].strval))); }
    break;

  case 132:

/* Line 1455 of yacc.c  */
#line 558 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[(3) - (7)].strval)), (yyvsp[(5) - (7)].mvlist)); }
    break;

  case 133:

/* Line 1455 of yacc.c  */
#line 562 "xi-grammar.y"
    { (yyval.type) = 0; }
    break;

  case 134:

/* Line 1455 of yacc.c  */
#line 564 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(2) - (2)].type); }
    break;

  case 135:

/* Line 1455 of yacc.c  */
#line 568 "xi-grammar.y"
    { (yyval.strval) = 0; }
    break;

  case 136:

/* Line 1455 of yacc.c  */
#line 570 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(2) - (2)].strval); }
    break;

  case 137:

/* Line 1455 of yacc.c  */
#line 572 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(2) - (2)].strval); }
    break;

  case 138:

/* Line 1455 of yacc.c  */
#line 576 "xi-grammar.y"
    { (yyval.tvar) = new TType(new NamedType((yyvsp[(2) - (3)].strval)), (yyvsp[(3) - (3)].type)); }
    break;

  case 139:

/* Line 1455 of yacc.c  */
#line 578 "xi-grammar.y"
    { (yyval.tvar) = new TFunc((yyvsp[(1) - (2)].ftype), (yyvsp[(2) - (2)].strval)); }
    break;

  case 140:

/* Line 1455 of yacc.c  */
#line 580 "xi-grammar.y"
    { (yyval.tvar) = new TName((yyvsp[(1) - (3)].type), (yyvsp[(2) - (3)].strval), (yyvsp[(3) - (3)].strval)); }
    break;

  case 141:

/* Line 1455 of yacc.c  */
#line 584 "xi-grammar.y"
    { (yyval.tvarlist) = new TVarList((yyvsp[(1) - (1)].tvar)); }
    break;

  case 142:

/* Line 1455 of yacc.c  */
#line 586 "xi-grammar.y"
    { (yyval.tvarlist) = new TVarList((yyvsp[(1) - (3)].tvar), (yyvsp[(3) - (3)].tvarlist)); }
    break;

  case 143:

/* Line 1455 of yacc.c  */
#line 590 "xi-grammar.y"
    { (yyval.tvarlist) = (yyvsp[(3) - (4)].tvarlist); }
    break;

  case 144:

/* Line 1455 of yacc.c  */
#line 594 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 145:

/* Line 1455 of yacc.c  */
#line 596 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 146:

/* Line 1455 of yacc.c  */
#line 598 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 147:

/* Line 1455 of yacc.c  */
#line 600 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 148:

/* Line 1455 of yacc.c  */
#line 602 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].message)); (yyvsp[(2) - (2)].message)->setTemplate((yyval.templat)); }
    break;

  case 149:

/* Line 1455 of yacc.c  */
#line 606 "xi-grammar.y"
    { (yyval.mbrlist) = 0; }
    break;

  case 150:

/* Line 1455 of yacc.c  */
#line 608 "xi-grammar.y"
    { (yyval.mbrlist) = (yyvsp[(2) - (4)].mbrlist); }
    break;

  case 151:

/* Line 1455 of yacc.c  */
#line 612 "xi-grammar.y"
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
#line 620 "xi-grammar.y"
    { (yyval.mbrlist) = new MemberList((yyvsp[(1) - (2)].member), (yyvsp[(2) - (2)].mbrlist)); }
    break;

  case 153:

/* Line 1455 of yacc.c  */
#line 624 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].readonly); }
    break;

  case 154:

/* Line 1455 of yacc.c  */
#line 626 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].readonly); }
    break;

  case 156:

/* Line 1455 of yacc.c  */
#line 629 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].member); }
    break;

  case 157:

/* Line 1455 of yacc.c  */
#line 631 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(2) - (2)].pupable); }
    break;

  case 158:

/* Line 1455 of yacc.c  */
#line 633 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(2) - (2)].includeFile); }
    break;

  case 159:

/* Line 1455 of yacc.c  */
#line 635 "xi-grammar.y"
    { (yyval.member) = new ClassDeclaration(lineno,(yyvsp[(2) - (2)].strval)); }
    break;

  case 160:

/* Line 1455 of yacc.c  */
#line 639 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 1); }
    break;

  case 161:

/* Line 1455 of yacc.c  */
#line 641 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 1); }
    break;

  case 162:

/* Line 1455 of yacc.c  */
#line 643 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[(3) - (9)].strval)) + '<' +
					    ((yyvsp[(5) - (9)].tparlist))->to_string() + '>').c_str()),
				    1);
		}
    break;

  case 163:

/* Line 1455 of yacc.c  */
#line 649 "xi-grammar.y"
    { printf("Warning: deprecated use of initcall. Use initnode or initproc instead.\n"); 
		  (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 1); }
    break;

  case 164:

/* Line 1455 of yacc.c  */
#line 652 "xi-grammar.y"
    { printf("Warning: deprecated use of initcall. Use initnode or initproc instead.\n");
		  (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 1); }
    break;

  case 165:

/* Line 1455 of yacc.c  */
#line 657 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 0); }
    break;

  case 166:

/* Line 1455 of yacc.c  */
#line 659 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 0); }
    break;

  case 167:

/* Line 1455 of yacc.c  */
#line 661 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[(3) - (9)].strval)) + '<' +
					    ((yyvsp[(5) - (9)].tparlist))->to_string() + '>').c_str()),
				    0);
		}
    break;

  case 168:

/* Line 1455 of yacc.c  */
#line 667 "xi-grammar.y"
    {
                  InitCall* rtn = new InitCall(lineno, (yyvsp[(6) - (9)].strval), 0);
                  rtn->setAccel();
                  (yyval.member) = rtn;
		}
    break;

  case 169:

/* Line 1455 of yacc.c  */
#line 675 "xi-grammar.y"
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[(1) - (1)].ntype),0); }
    break;

  case 170:

/* Line 1455 of yacc.c  */
#line 677 "xi-grammar.y"
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[(1) - (3)].ntype),(yyvsp[(3) - (3)].pupable)); }
    break;

  case 171:

/* Line 1455 of yacc.c  */
#line 680 "xi-grammar.y"
    { (yyval.includeFile) = new IncludeFile(lineno,(yyvsp[(1) - (1)].strval)); }
    break;

  case 172:

/* Line 1455 of yacc.c  */
#line 684 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (2)].member); }
    break;

  case 173:

/* Line 1455 of yacc.c  */
#line 687 "xi-grammar.y"
    { yyerror("The preceding entry method declaration must be semicolon-terminated."); YYABORT; }
    break;

  case 174:

/* Line 1455 of yacc.c  */
#line 691 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].entry); }
    break;

  case 175:

/* Line 1455 of yacc.c  */
#line 693 "xi-grammar.y"
    {
                  (yyvsp[(2) - (2)].entry)->tspec = (yyvsp[(1) - (2)].tvarlist);
                  (yyval.member) = (yyvsp[(2) - (2)].entry);
                }
    break;

  case 176:

/* Line 1455 of yacc.c  */
#line 698 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].member); }
    break;

  case 177:

/* Line 1455 of yacc.c  */
#line 702 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 178:

/* Line 1455 of yacc.c  */
#line 704 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 179:

/* Line 1455 of yacc.c  */
#line 706 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 180:

/* Line 1455 of yacc.c  */
#line 708 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 181:

/* Line 1455 of yacc.c  */
#line 710 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 182:

/* Line 1455 of yacc.c  */
#line 712 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 183:

/* Line 1455 of yacc.c  */
#line 714 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 184:

/* Line 1455 of yacc.c  */
#line 716 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 185:

/* Line 1455 of yacc.c  */
#line 718 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 186:

/* Line 1455 of yacc.c  */
#line 720 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 187:

/* Line 1455 of yacc.c  */
#line 722 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 188:

/* Line 1455 of yacc.c  */
#line 725 "xi-grammar.y"
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
#line 733 "xi-grammar.y"
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
#line 746 "xi-grammar.y"
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
#line 764 "xi-grammar.y"
    { (yyval.accelBlock) = new AccelBlock(lineno, new XStr((yyvsp[(3) - (5)].strval))); }
    break;

  case 192:

/* Line 1455 of yacc.c  */
#line 766 "xi-grammar.y"
    { (yyval.accelBlock) = new AccelBlock(lineno, NULL); }
    break;

  case 193:

/* Line 1455 of yacc.c  */
#line 770 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("void"); }
    break;

  case 194:

/* Line 1455 of yacc.c  */
#line 772 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 195:

/* Line 1455 of yacc.c  */
#line 776 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 196:

/* Line 1455 of yacc.c  */
#line 778 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(2) - (3)].intval); }
    break;

  case 197:

/* Line 1455 of yacc.c  */
#line 780 "xi-grammar.y"
    { printf("Invalid entry method attribute list\n"); YYABORT; }
    break;

  case 198:

/* Line 1455 of yacc.c  */
#line 784 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (1)].intval); }
    break;

  case 199:

/* Line 1455 of yacc.c  */
#line 786 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (3)].intval) | (yyvsp[(3) - (3)].intval); }
    break;

  case 200:

/* Line 1455 of yacc.c  */
#line 790 "xi-grammar.y"
    { (yyval.intval) = STHREADED; }
    break;

  case 201:

/* Line 1455 of yacc.c  */
#line 792 "xi-grammar.y"
    { (yyval.intval) = SSYNC; }
    break;

  case 202:

/* Line 1455 of yacc.c  */
#line 794 "xi-grammar.y"
    { (yyval.intval) = SIGET; }
    break;

  case 203:

/* Line 1455 of yacc.c  */
#line 796 "xi-grammar.y"
    { (yyval.intval) = SLOCKED; }
    break;

  case 204:

/* Line 1455 of yacc.c  */
#line 798 "xi-grammar.y"
    { (yyval.intval) = SCREATEHERE; }
    break;

  case 205:

/* Line 1455 of yacc.c  */
#line 800 "xi-grammar.y"
    { (yyval.intval) = SCREATEHOME; }
    break;

  case 206:

/* Line 1455 of yacc.c  */
#line 802 "xi-grammar.y"
    { (yyval.intval) = SNOKEEP; }
    break;

  case 207:

/* Line 1455 of yacc.c  */
#line 804 "xi-grammar.y"
    { (yyval.intval) = SNOTRACE; }
    break;

  case 208:

/* Line 1455 of yacc.c  */
#line 806 "xi-grammar.y"
    { (yyval.intval) = SIMMEDIATE; }
    break;

  case 209:

/* Line 1455 of yacc.c  */
#line 808 "xi-grammar.y"
    { (yyval.intval) = SSKIPSCHED; }
    break;

  case 210:

/* Line 1455 of yacc.c  */
#line 810 "xi-grammar.y"
    { (yyval.intval) = SINLINE; }
    break;

  case 211:

/* Line 1455 of yacc.c  */
#line 812 "xi-grammar.y"
    { (yyval.intval) = SLOCAL; }
    break;

  case 212:

/* Line 1455 of yacc.c  */
#line 814 "xi-grammar.y"
    { (yyval.intval) = SPYTHON; }
    break;

  case 213:

/* Line 1455 of yacc.c  */
#line 816 "xi-grammar.y"
    { (yyval.intval) = SMEM; }
    break;

  case 214:

/* Line 1455 of yacc.c  */
#line 818 "xi-grammar.y"
    { (yyval.intval) = SREDUCE; }
    break;

  case 215:

/* Line 1455 of yacc.c  */
#line 820 "xi-grammar.y"
    { printf("Invalid entry method attribute: %s\n", yylval); YYABORT; }
    break;

  case 216:

/* Line 1455 of yacc.c  */
#line 824 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 217:

/* Line 1455 of yacc.c  */
#line 826 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 218:

/* Line 1455 of yacc.c  */
#line 828 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 219:

/* Line 1455 of yacc.c  */
#line 832 "xi-grammar.y"
    { (yyval.strval) = ""; }
    break;

  case 220:

/* Line 1455 of yacc.c  */
#line 834 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 221:

/* Line 1455 of yacc.c  */
#line 836 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[(1) - (3)].strval))+strlen((yyvsp[(3) - (3)].strval))+3];
			sprintf(tmp,"%s, %s", (yyvsp[(1) - (3)].strval), (yyvsp[(3) - (3)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 222:

/* Line 1455 of yacc.c  */
#line 844 "xi-grammar.y"
    { (yyval.strval) = ""; }
    break;

  case 223:

/* Line 1455 of yacc.c  */
#line 846 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 224:

/* Line 1455 of yacc.c  */
#line 848 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s[%s]%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 225:

/* Line 1455 of yacc.c  */
#line 854 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s{%s}%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 226:

/* Line 1455 of yacc.c  */
#line 860 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s(%s)%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 227:

/* Line 1455 of yacc.c  */
#line 866 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(2) - (4)].strval))+strlen((yyvsp[(4) - (4)].strval))+3];
			sprintf(tmp,"(%s)%s", (yyvsp[(2) - (4)].strval), (yyvsp[(4) - (4)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 228:

/* Line 1455 of yacc.c  */
#line 874 "xi-grammar.y"
    {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			(yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].type),(yyvsp[(2) - (3)].strval));
		}
    break;

  case 229:

/* Line 1455 of yacc.c  */
#line 881 "xi-grammar.y"
    { 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			(yyval.intval) = 0;
		}
    break;

  case 230:

/* Line 1455 of yacc.c  */
#line 889 "xi-grammar.y"
    { 
			in_braces=0;
			(yyval.intval) = 0;
		}
    break;

  case 231:

/* Line 1455 of yacc.c  */
#line 896 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (1)].type));}
    break;

  case 232:

/* Line 1455 of yacc.c  */
#line 898 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].type),(yyvsp[(2) - (3)].strval)); (yyval.pname)->setConditional((yyvsp[(3) - (3)].intval)); }
    break;

  case 233:

/* Line 1455 of yacc.c  */
#line 900 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (4)].type),(yyvsp[(2) - (4)].strval),0,(yyvsp[(4) - (4)].val));}
    break;

  case 234:

/* Line 1455 of yacc.c  */
#line 902 "xi-grammar.y"
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].pname)->getType(), (yyvsp[(1) - (3)].pname)->getName() ,(yyvsp[(2) - (3)].strval));
		}
    break;

  case 235:

/* Line 1455 of yacc.c  */
#line 908 "xi-grammar.y"
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READONLY; }
    break;

  case 236:

/* Line 1455 of yacc.c  */
#line 909 "xi-grammar.y"
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READWRITE; }
    break;

  case 237:

/* Line 1455 of yacc.c  */
#line 910 "xi-grammar.y"
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_WRITEONLY; }
    break;

  case 238:

/* Line 1455 of yacc.c  */
#line 913 "xi-grammar.y"
    { (yyval.xstrptr) = new XStr((yyvsp[(1) - (1)].strval)); }
    break;

  case 239:

/* Line 1455 of yacc.c  */
#line 914 "xi-grammar.y"
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "->" << (yyvsp[(4) - (4)].strval); }
    break;

  case 240:

/* Line 1455 of yacc.c  */
#line 915 "xi-grammar.y"
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[(1) - (3)].xstrptr)) << "." << (yyvsp[(3) - (3)].strval); }
    break;

  case 241:

/* Line 1455 of yacc.c  */
#line 917 "xi-grammar.y"
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "[" << *((yyvsp[(3) - (4)].xstrptr)) << "]";
                  delete (yyvsp[(1) - (4)].xstrptr);
                  delete (yyvsp[(3) - (4)].xstrptr);
                }
    break;

  case 242:

/* Line 1455 of yacc.c  */
#line 924 "xi-grammar.y"
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "[" << (yyvsp[(3) - (4)].strval) << "]";
                  delete (yyvsp[(1) - (4)].xstrptr);
                }
    break;

  case 243:

/* Line 1455 of yacc.c  */
#line 930 "xi-grammar.y"
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "(" << *((yyvsp[(3) - (4)].xstrptr)) << ")";
                  delete (yyvsp[(1) - (4)].xstrptr);
                  delete (yyvsp[(3) - (4)].xstrptr);
                }
    break;

  case 244:

/* Line 1455 of yacc.c  */
#line 939 "xi-grammar.y"
    {
                  in_bracket = 0;
                  (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].pname)->getType(), (yyvsp[(1) - (3)].pname)->getName(), (yyvsp[(2) - (3)].strval));
                }
    break;

  case 245:

/* Line 1455 of yacc.c  */
#line 946 "xi-grammar.y"
    {
                  (yyval.pname) = new Parameter(lineno, (yyvsp[(3) - (7)].type), (yyvsp[(4) - (7)].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[(6) - (7)].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[(1) - (7)].intval));
                }
    break;

  case 246:

/* Line 1455 of yacc.c  */
#line 952 "xi-grammar.y"
    {
		  (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (5)].type), (yyvsp[(2) - (5)].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[(4) - (5)].xstrptr));
                  (yyval.pname)->setAccelBufferType(Parameter::ACCEL_BUFFER_TYPE_READWRITE);
		}
    break;

  case 247:

/* Line 1455 of yacc.c  */
#line 958 "xi-grammar.y"
    {
                  (yyval.pname) = (yyvsp[(3) - (6)].pname);
                  (yyval.pname)->setAccelInstName((yyvsp[(5) - (6)].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[(1) - (6)].intval));
		}
    break;

  case 248:

/* Line 1455 of yacc.c  */
#line 966 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (1)].pname)); }
    break;

  case 249:

/* Line 1455 of yacc.c  */
#line 968 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (3)].pname),(yyvsp[(3) - (3)].plist)); }
    break;

  case 250:

/* Line 1455 of yacc.c  */
#line 972 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (1)].pname)); }
    break;

  case 251:

/* Line 1455 of yacc.c  */
#line 974 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (3)].pname),(yyvsp[(3) - (3)].plist)); }
    break;

  case 252:

/* Line 1455 of yacc.c  */
#line 978 "xi-grammar.y"
    { (yyval.plist) = (yyvsp[(2) - (3)].plist); }
    break;

  case 253:

/* Line 1455 of yacc.c  */
#line 980 "xi-grammar.y"
    { (yyval.plist) = new ParamList(new Parameter(0, new BuiltinType("void"))); }
    break;

  case 254:

/* Line 1455 of yacc.c  */
#line 984 "xi-grammar.y"
    { (yyval.plist) = (yyvsp[(2) - (3)].plist); }
    break;

  case 255:

/* Line 1455 of yacc.c  */
#line 986 "xi-grammar.y"
    { (yyval.plist) = 0; }
    break;

  case 256:

/* Line 1455 of yacc.c  */
#line 990 "xi-grammar.y"
    { (yyval.val) = 0; }
    break;

  case 257:

/* Line 1455 of yacc.c  */
#line 992 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(3) - (3)].strval)); }
    break;

  case 258:

/* Line 1455 of yacc.c  */
#line 996 "xi-grammar.y"
    { (yyval.sc) = 0; }
    break;

  case 259:

/* Line 1455 of yacc.c  */
#line 998 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSDAGENTRY, (yyvsp[(1) - (1)].sc)); }
    break;

  case 260:

/* Line 1455 of yacc.c  */
#line 1000 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSDAGENTRY, (yyvsp[(2) - (3)].sc)); }
    break;

  case 261:

/* Line 1455 of yacc.c  */
#line 1004 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSLIST, (yyvsp[(1) - (1)].sc)); }
    break;

  case 262:

/* Line 1455 of yacc.c  */
#line 1006 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSLIST, (yyvsp[(1) - (2)].sc), (yyvsp[(2) - (2)].sc));  }
    break;

  case 263:

/* Line 1455 of yacc.c  */
#line 1010 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SOLIST, (yyvsp[(1) - (1)].sc)); }
    break;

  case 264:

/* Line 1455 of yacc.c  */
#line 1012 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SOLIST, (yyvsp[(1) - (2)].sc), (yyvsp[(2) - (2)].sc)); }
    break;

  case 265:

/* Line 1455 of yacc.c  */
#line 1016 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SCASELIST, (yyvsp[(1) - (1)].when)); }
    break;

  case 266:

/* Line 1455 of yacc.c  */
#line 1018 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SCASELIST, (yyvsp[(1) - (2)].when), (yyvsp[(2) - (2)].sc)); }
    break;

  case 267:

/* Line 1455 of yacc.c  */
#line 1020 "xi-grammar.y"
    { yyerror("Case blocks in SDAG can only contain when clauses."); YYABORT; }
    break;

  case 268:

/* Line 1455 of yacc.c  */
#line 1024 "xi-grammar.y"
    { (yyval.sc) = 0; }
    break;

  case 269:

/* Line 1455 of yacc.c  */
#line 1026 "xi-grammar.y"
    { (yyval.sc) = (yyvsp[(3) - (4)].sc); }
    break;

  case 270:

/* Line 1455 of yacc.c  */
#line 1030 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SPUBLISHES, new SdagConstruct(SIDENT, (yyvsp[(1) - (1)].strval))); }
    break;

  case 271:

/* Line 1455 of yacc.c  */
#line 1032 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SPUBLISHES, new SdagConstruct(SIDENT, (yyvsp[(1) - (3)].strval)), (yyvsp[(3) - (3)].sc));  }
    break;

  case 272:

/* Line 1455 of yacc.c  */
#line 1036 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 273:

/* Line 1455 of yacc.c  */
#line 1038 "xi-grammar.y"
    { (yyval.strval) = 0; }
    break;

  case 274:

/* Line 1455 of yacc.c  */
#line 1042 "xi-grammar.y"
    { (yyval.when) = new WhenConstruct((yyvsp[(2) - (4)].entrylist), 0); }
    break;

  case 275:

/* Line 1455 of yacc.c  */
#line 1044 "xi-grammar.y"
    { (yyval.when) = new WhenConstruct((yyvsp[(2) - (3)].entrylist), (yyvsp[(3) - (3)].sc)); }
    break;

  case 276:

/* Line 1455 of yacc.c  */
#line 1046 "xi-grammar.y"
    { (yyval.when) = new WhenConstruct((yyvsp[(2) - (5)].entrylist), (yyvsp[(4) - (5)].sc)); }
    break;

  case 277:

/* Line 1455 of yacc.c  */
#line 1050 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 278:

/* Line 1455 of yacc.c  */
#line 1052 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 279:

/* Line 1455 of yacc.c  */
#line 1054 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 280:

/* Line 1455 of yacc.c  */
#line 1056 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 281:

/* Line 1455 of yacc.c  */
#line 1058 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 282:

/* Line 1455 of yacc.c  */
#line 1060 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 283:

/* Line 1455 of yacc.c  */
#line 1062 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 284:

/* Line 1455 of yacc.c  */
#line 1064 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 285:

/* Line 1455 of yacc.c  */
#line 1068 "xi-grammar.y"
    {
		   (yyval.sc) = new AtomicConstruct((yyvsp[(4) - (6)].strval), (yyvsp[(6) - (6)].sc), (yyvsp[(2) - (6)].strval));
		 }
    break;

  case 286:

/* Line 1455 of yacc.c  */
#line 1072 "xi-grammar.y"
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

  case 287:

/* Line 1455 of yacc.c  */
#line 1086 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SOVERLAP,0, 0,0,0,0,(yyvsp[(3) - (4)].sc), 0); }
    break;

  case 288:

/* Line 1455 of yacc.c  */
#line 1088 "xi-grammar.y"
    { (yyval.sc) = (yyvsp[(1) - (1)].when); }
    break;

  case 289:

/* Line 1455 of yacc.c  */
#line 1090 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SCASE, 0, 0, 0, 0, 0, (yyvsp[(3) - (4)].sc), 0); }
    break;

  case 290:

/* Line 1455 of yacc.c  */
#line 1092 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFOR, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (11)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(5) - (11)].strval)),
		             new SdagConstruct(SINT_EXPR, (yyvsp[(7) - (11)].strval)), 0, (yyvsp[(10) - (11)].sc), 0); }
    break;

  case 291:

/* Line 1455 of yacc.c  */
#line 1095 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFOR, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (9)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(5) - (9)].strval)), 
		         new SdagConstruct(SINT_EXPR, (yyvsp[(7) - (9)].strval)), 0, (yyvsp[(9) - (9)].sc), 0); }
    break;

  case 292:

/* Line 1455 of yacc.c  */
#line 1098 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFORALL, 0, new SdagConstruct(SIDENT, (yyvsp[(3) - (12)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(6) - (12)].strval)), 
		             new SdagConstruct(SINT_EXPR, (yyvsp[(8) - (12)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(10) - (12)].strval)), (yyvsp[(12) - (12)].sc), 0); }
    break;

  case 293:

/* Line 1455 of yacc.c  */
#line 1101 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFORALL, 0, new SdagConstruct(SIDENT, (yyvsp[(3) - (14)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(6) - (14)].strval)), 
		                 new SdagConstruct(SINT_EXPR, (yyvsp[(8) - (14)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(10) - (14)].strval)), (yyvsp[(13) - (14)].sc), 0); }
    break;

  case 294:

/* Line 1455 of yacc.c  */
#line 1104 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (6)].strval)), (yyvsp[(6) - (6)].sc),0,0,(yyvsp[(5) - (6)].sc),0); }
    break;

  case 295:

/* Line 1455 of yacc.c  */
#line 1106 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (8)].strval)), (yyvsp[(8) - (8)].sc),0,0,(yyvsp[(6) - (8)].sc),0); }
    break;

  case 296:

/* Line 1455 of yacc.c  */
#line 1108 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SWHILE, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (5)].strval)), 0,0,0,(yyvsp[(5) - (5)].sc),0); }
    break;

  case 297:

/* Line 1455 of yacc.c  */
#line 1110 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SWHILE, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (7)].strval)), 0,0,0,(yyvsp[(6) - (7)].sc),0); }
    break;

  case 298:

/* Line 1455 of yacc.c  */
#line 1112 "xi-grammar.y"
    { (yyval.sc) = (yyvsp[(2) - (3)].sc); }
    break;

  case 299:

/* Line 1455 of yacc.c  */
#line 1114 "xi-grammar.y"
    { (yyval.sc) = new AtomicConstruct((yyvsp[(2) - (3)].strval), NULL, NULL); }
    break;

  case 300:

/* Line 1455 of yacc.c  */
#line 1116 "xi-grammar.y"
    { printf("Unknown SDAG construct or malformed entry method definition.\n"
                         "You may have forgotten to terminate an entry method definition with a"
                         " semicolon or forgotten to mark a block of sequential SDAG code as 'atomic'\n"); YYABORT; }
    break;

  case 301:

/* Line 1455 of yacc.c  */
#line 1122 "xi-grammar.y"
    { (yyval.sc) = 0; }
    break;

  case 302:

/* Line 1455 of yacc.c  */
#line 1124 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SELSE, 0,0,0,0,0, (yyvsp[(2) - (2)].sc),0); }
    break;

  case 303:

/* Line 1455 of yacc.c  */
#line 1126 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SELSE, 0,0,0,0,0, (yyvsp[(3) - (4)].sc),0); }
    break;

  case 304:

/* Line 1455 of yacc.c  */
#line 1129 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFORWARD, new SdagConstruct(SIDENT, (yyvsp[(1) - (1)].strval))); }
    break;

  case 305:

/* Line 1455 of yacc.c  */
#line 1131 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFORWARD, new SdagConstruct(SIDENT, (yyvsp[(1) - (3)].strval)), (yyvsp[(3) - (3)].sc));  }
    break;

  case 306:

/* Line 1455 of yacc.c  */
#line 1135 "xi-grammar.y"
    { in_int_expr = 0; (yyval.intval) = 0; }
    break;

  case 307:

/* Line 1455 of yacc.c  */
#line 1139 "xi-grammar.y"
    { in_int_expr = 1; (yyval.intval) = 0; }
    break;

  case 308:

/* Line 1455 of yacc.c  */
#line 1143 "xi-grammar.y"
    { (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[(1) - (2)].strval), (yyvsp[(2) - (2)].plist), 0, 0, 0, 0); }
    break;

  case 309:

/* Line 1455 of yacc.c  */
#line 1145 "xi-grammar.y"
    { (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[(1) - (5)].strval), (yyvsp[(5) - (5)].plist), 0, 0, (yyvsp[(3) - (5)].strval), 0); }
    break;

  case 310:

/* Line 1455 of yacc.c  */
#line 1149 "xi-grammar.y"
    { (yyval.entrylist) = new EntryList((yyvsp[(1) - (1)].entry)); }
    break;

  case 311:

/* Line 1455 of yacc.c  */
#line 1151 "xi-grammar.y"
    { (yyval.entrylist) = new EntryList((yyvsp[(1) - (3)].entry),(yyvsp[(3) - (3)].entrylist)); }
    break;

  case 312:

/* Line 1455 of yacc.c  */
#line 1155 "xi-grammar.y"
    { in_bracket=1; }
    break;

  case 313:

/* Line 1455 of yacc.c  */
#line 1158 "xi-grammar.y"
    { in_bracket=0; }
    break;

  case 314:

/* Line 1455 of yacc.c  */
#line 1162 "xi-grammar.y"
    { if (!macroDefined((yyvsp[(2) - (2)].strval), 1)) in_comment = 1; }
    break;

  case 315:

/* Line 1455 of yacc.c  */
#line 1166 "xi-grammar.y"
    { if (!macroDefined((yyvsp[(2) - (2)].strval), 0)) in_comment = 1; }
    break;



/* Line 1455 of yacc.c  */
#line 4547 "y.tab.c"
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
#line 1169 "xi-grammar.y"

void yyerror(const char *mesg)
{
    std::cerr << cur_file<<":"<<lineno<<": Charmxi syntax error> "
	      << mesg << std::endl;
}

