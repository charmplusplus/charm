
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
extern TList<Entry *> *connectEntries;
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
#define YYLAST   741

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  90
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  110
/* YYNRULES -- Number of rules.  */
#define YYNRULES  283
/* YYNRULES -- Number of states.  */
#define YYNSTATES  598

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
      17,    19,    24,    28,    32,    34,    39,    40,    43,    49,
      55,    60,    64,    67,    70,    74,    77,    80,    83,    86,
      89,    98,   100,   102,   104,   106,   108,   110,   112,   116,
     117,   119,   120,   124,   126,   128,   130,   132,   135,   138,
     142,   146,   149,   152,   155,   157,   159,   162,   164,   167,
     170,   172,   174,   177,   180,   183,   192,   194,   196,   198,
     200,   203,   206,   209,   211,   213,   215,   219,   220,   223,
     228,   234,   235,   237,   238,   242,   244,   248,   250,   252,
     253,   257,   259,   263,   264,   266,   268,   269,   273,   275,
     279,   281,   283,   284,   286,   287,   290,   296,   298,   301,
     305,   312,   313,   316,   318,   322,   328,   334,   340,   346,
     351,   355,   362,   369,   375,   381,   387,   393,   399,   404,
     412,   413,   416,   417,   420,   423,   427,   430,   434,   436,
     440,   445,   448,   451,   454,   457,   460,   462,   467,   468,
     471,   474,   477,   480,   483,   487,   491,   495,   499,   506,
     516,   520,   527,   531,   538,   548,   558,   560,   564,   566,
     569,   573,   575,   583,   589,   602,   608,   611,   613,   615,
     616,   620,   622,   626,   628,   630,   632,   634,   636,   638,
     640,   642,   644,   646,   648,   650,   653,   655,   657,   659,
     661,   663,   664,   666,   670,   671,   673,   679,   685,   691,
     696,   700,   702,   704,   706,   710,   715,   719,   721,   723,
     725,   727,   732,   736,   741,   746,   751,   755,   763,   769,
     776,   778,   782,   784,   788,   792,   795,   799,   802,   803,
     807,   808,   810,   814,   816,   819,   821,   824,   825,   830,
     832,   836,   838,   839,   846,   855,   860,   864,   870,   875,
     887,   897,   910,   925,   932,   941,   947,   955,   959,   963,
     964,   967,   972,   974,   978,   980,   982,   985,   991,   993,
     997,   999,  1001,  1004
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
      91,     0,    -1,    92,    -1,    -1,    97,    92,    -1,    -1,
       5,    -1,    -1,    74,    -1,    55,    -1,    55,    -1,    96,
      75,    75,    55,    -1,     3,    95,    98,    -1,     4,    95,
      98,    -1,    74,    -1,    76,    99,    77,    94,    -1,    -1,
     100,    99,    -1,    93,    76,    99,    77,    94,    -1,    53,
      95,    76,    99,    77,    -1,    54,    53,    96,    74,    -1,
      54,    96,    74,    -1,    93,    97,    -1,    93,   155,    -1,
      93,   134,    74,    -1,    93,   137,    -1,    93,   138,    -1,
      93,   139,    -1,    93,   141,    -1,    93,   152,    -1,     5,
      39,   163,   107,    95,   104,   180,    74,    -1,   198,    -1,
     199,    -1,   162,    -1,   113,    -1,    56,    -1,    57,    -1,
     101,    -1,   101,    78,   102,    -1,    -1,   102,    -1,    -1,
      79,   103,    80,    -1,    61,    -1,    62,    -1,    63,    -1,
      64,    -1,    67,    61,    -1,    67,    62,    -1,    67,    62,
      61,    -1,    67,    62,    62,    -1,    67,    63,    -1,    67,
      64,    -1,    62,    62,    -1,    65,    -1,    66,    -1,    62,
      66,    -1,    35,    -1,    95,   104,    -1,    96,   104,    -1,
     105,    -1,   107,    -1,   108,    81,    -1,   109,    81,    -1,
     110,    81,    -1,   112,    82,    81,    95,    83,    82,   178,
      83,    -1,   108,    -1,   109,    -1,   110,    -1,   111,    -1,
      36,   112,    -1,   112,    36,    -1,   112,    84,    -1,   112,
      -1,    56,    -1,    96,    -1,    85,   114,    86,    -1,    -1,
     115,   116,    -1,     6,   113,    96,   116,    -1,     6,    16,
     108,    81,    95,    -1,    -1,    35,    -1,    -1,    85,   121,
      86,    -1,   122,    -1,   122,    78,   121,    -1,    37,    -1,
      38,    -1,    -1,    85,   124,    86,    -1,   129,    -1,   129,
      78,   124,    -1,    -1,    57,    -1,    51,    -1,    -1,    85,
     128,    86,    -1,   126,    -1,   126,    78,   128,    -1,    30,
      -1,    51,    -1,    -1,    17,    -1,    -1,    85,    86,    -1,
     130,   113,    95,   131,    74,    -1,   132,    -1,   132,   133,
      -1,    16,   120,   106,    -1,    16,   120,   106,    76,   133,
      77,    -1,    -1,    75,   136,    -1,   107,    -1,   107,    78,
     136,    -1,    11,   123,   106,   135,   153,    -1,    12,   123,
     106,   135,   153,    -1,    13,   123,   106,   135,   153,    -1,
      14,   123,   106,   135,   153,    -1,    85,    56,    95,    86,
      -1,    85,    95,    86,    -1,    15,   127,   140,   106,   135,
     153,    -1,    15,   140,   127,   106,   135,   153,    -1,    11,
     123,    95,   135,   153,    -1,    12,   123,    95,   135,   153,
      -1,    13,   123,    95,   135,   153,    -1,    14,   123,    95,
     135,   153,    -1,    15,   140,    95,   135,   153,    -1,    16,
     120,    95,    74,    -1,    16,   120,    95,    76,   133,    77,
      74,    -1,    -1,    87,   113,    -1,    -1,    87,    56,    -1,
      87,    57,    -1,    18,    95,   147,    -1,   111,   148,    -1,
     113,    95,   148,    -1,   149,    -1,   149,    78,   150,    -1,
      22,    79,   150,    80,    -1,   151,   142,    -1,   151,   143,
      -1,   151,   144,    -1,   151,   145,    -1,   151,   146,    -1,
      74,    -1,    76,   154,    77,    94,    -1,    -1,   160,   154,
      -1,   117,    74,    -1,   118,    74,    -1,   157,    74,    -1,
     156,    74,    -1,    10,   158,    74,    -1,    19,   159,    74,
      -1,    18,    95,    74,    -1,     8,   119,    96,    -1,     8,
     119,    96,    82,   119,    83,    -1,     8,   119,    96,    79,
     102,    80,    82,   119,    83,    -1,     7,   119,    96,    -1,
       7,   119,    96,    82,   119,    83,    -1,     9,   119,    96,
      -1,     9,   119,    96,    82,   119,    83,    -1,     9,   119,
      96,    79,   102,    80,    82,   119,    83,    -1,     9,    85,
      68,    86,   119,    96,    82,   119,    83,    -1,   107,    -1,
     107,    78,   158,    -1,    57,    -1,   161,    74,    -1,   151,
     161,    74,    -1,   155,    -1,    39,   164,   163,    95,   180,
     182,   183,    -1,    39,   164,    95,   180,   183,    -1,    39,
      85,    68,    86,    35,    95,   180,   181,   171,   169,   172,
      95,    -1,    71,   171,   169,   172,    74,    -1,    71,    74,
      -1,    35,    -1,   109,    -1,    -1,    85,   165,    86,    -1,
     166,    -1,   166,    78,   165,    -1,    21,    -1,    23,    -1,
      24,    -1,    25,    -1,    31,    -1,    32,    -1,    33,    -1,
      34,    -1,    26,    -1,    27,    -1,    28,    -1,    52,    -1,
      51,   125,    -1,    72,    -1,    73,    -1,    57,    -1,    56,
      -1,    96,    -1,    -1,    58,    -1,    58,    78,   168,    -1,
      -1,    58,    -1,    58,    85,   169,    86,   169,    -1,    58,
      76,   169,    77,   169,    -1,    58,    82,   168,    83,   169,
      -1,    82,   169,    83,   169,    -1,   113,    95,    85,    -1,
      76,    -1,    77,    -1,   113,    -1,   113,    95,   130,    -1,
     113,    95,    87,   167,    -1,   170,   169,    86,    -1,     6,
      -1,    69,    -1,    70,    -1,    95,    -1,   175,    88,    80,
      95,    -1,   175,    89,    95,    -1,   175,    85,   175,    86,
      -1,   175,    85,    56,    86,    -1,   175,    82,   175,    83,
      -1,   170,   169,    86,    -1,   174,    75,   113,    95,    79,
     175,    80,    -1,   113,    95,    79,   175,    80,    -1,   174,
      75,   176,    79,   175,    80,    -1,   173,    -1,   173,    78,
     178,    -1,   177,    -1,   177,    78,   179,    -1,    82,   178,
      83,    -1,    82,    83,    -1,    85,   179,    86,    -1,    85,
      86,    -1,    -1,    20,    87,    56,    -1,    -1,   189,    -1,
      76,   184,    77,    -1,   189,    -1,   189,   184,    -1,   189,
      -1,   189,   184,    -1,    -1,    50,    82,   187,    83,    -1,
      55,    -1,    55,    78,   187,    -1,    57,    -1,    -1,    45,
     188,   171,   169,   172,   186,    -1,    49,    82,    55,   180,
      83,   171,   169,    77,    -1,    43,   195,    76,    77,    -1,
      43,   195,   189,    -1,    43,   195,    76,   184,    77,    -1,
      44,    76,   185,    77,    -1,    40,   193,   169,    74,   169,
      74,   169,   192,    76,   184,    77,    -1,    40,   193,   169,
      74,   169,    74,   169,   192,   189,    -1,    41,    85,    55,
      86,   193,   169,    75,   169,    78,   169,   192,   189,    -1,
      41,    85,    55,    86,   193,   169,    75,   169,    78,   169,
     192,    76,   184,    77,    -1,    47,   193,   169,   192,   189,
     190,    -1,    47,   193,   169,   192,    76,   184,    77,   190,
      -1,    42,   193,   169,   192,   189,    -1,    42,   193,   169,
     192,    76,   184,    77,    -1,    46,   191,    74,    -1,   171,
     169,   172,    -1,    -1,    48,   189,    -1,    48,    76,   184,
      77,    -1,    55,    -1,    55,    78,   191,    -1,    83,    -1,
      82,    -1,    55,   180,    -1,    55,   196,   169,   197,   180,
      -1,   194,    -1,   194,    78,   195,    -1,    85,    -1,    86,
      -1,    59,    95,    -1,    60,    95,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   154,   154,   159,   162,   167,   168,   173,   174,   178,
     182,   184,   192,   196,   203,   205,   210,   211,   215,   217,
     219,   221,   223,   225,   227,   229,   231,   233,   235,   237,
     239,   249,   251,   253,   257,   259,   261,   265,   267,   272,
     273,   278,   279,   283,   285,   287,   289,   291,   293,   295,
     297,   299,   301,   303,   305,   307,   309,   311,   315,   316,
     323,   325,   329,   333,   335,   339,   343,   345,   347,   349,
     352,   354,   358,   360,   364,   366,   370,   375,   376,   380,
     384,   389,   390,   395,   396,   406,   408,   412,   414,   419,
     420,   424,   426,   431,   432,   436,   441,   442,   446,   448,
     452,   454,   459,   460,   464,   465,   468,   472,   474,   478,
     480,   485,   486,   490,   492,   496,   498,   502,   506,   510,
     516,   520,   522,   526,   528,   532,   536,   540,   544,   546,
     551,   552,   557,   558,   560,   564,   566,   568,   572,   574,
     578,   582,   584,   586,   588,   590,   594,   596,   601,   619,
     623,   625,   627,   628,   630,   632,   634,   638,   640,   642,
     648,   651,   656,   658,   660,   666,   674,   676,   679,   683,
     685,   690,   694,   702,   715,   733,   735,   739,   741,   746,
     747,   751,   753,   757,   759,   761,   763,   765,   767,   769,
     771,   773,   775,   777,   779,   781,   783,   785,   789,   791,
     793,   798,   799,   801,   810,   811,   813,   819,   825,   831,
     839,   846,   854,   861,   863,   865,   867,   874,   875,   876,
     879,   880,   881,   882,   889,   895,   904,   911,   917,   923,
     931,   933,   937,   939,   943,   945,   949,   951,   956,   957,
     962,   963,   965,   969,   971,   975,   977,   982,   983,   987,
     989,   993,   996,   999,  1003,  1017,  1019,  1021,  1023,  1025,
    1028,  1031,  1034,  1037,  1039,  1041,  1043,  1045,  1047,  1051,
    1052,  1054,  1057,  1059,  1063,  1067,  1071,  1073,  1077,  1079,
    1083,  1086,  1090,  1094
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
  "ConstructList", "Construct", "TParam", "TParamList", "TParamEList",
  "OptTParams", "BuiltinType", "NamedType", "QualNamedType", "SimpleType",
  "OnePtrType", "PtrType", "FuncType", "BaseType", "Type", "ArrayDim",
  "Dim", "DimList", "Readonly", "ReadonlyMsg", "OptVoid", "MAttribs",
  "MAttribList", "MAttrib", "CAttribs", "CAttribList", "PythonOptions",
  "ArrayAttrib", "ArrayAttribs", "ArrayAttribList", "CAttrib",
  "OptConditional", "MsgArray", "Var", "VarList", "Message", "OptBaseList",
  "BaseList", "Chare", "Group", "NodeGroup", "ArrayIndexType", "Array",
  "TChare", "TGroup", "TNodeGroup", "TArray", "TMessage", "OptTypeInit",
  "OptNameInit", "TVar", "TVarList", "TemplateSpec", "Template",
  "MemberEList", "MemberList", "NonEntryMember", "InitNode", "InitProc",
  "PUPableClass", "IncludeFile", "Member", "Entry", "AccelBlock",
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
     100,   100,   100,   100,   100,   100,   100,   100,   100,   100,
     100,   100,   100,   100,   101,   101,   101,   102,   102,   103,
     103,   104,   104,   105,   105,   105,   105,   105,   105,   105,
     105,   105,   105,   105,   105,   105,   105,   105,   106,   107,
     108,   108,   109,   110,   110,   111,   112,   112,   112,   112,
     112,   112,   113,   113,   114,   114,   115,   116,   116,   117,
     118,   119,   119,   120,   120,   121,   121,   122,   122,   123,
     123,   124,   124,   125,   125,   126,   127,   127,   128,   128,
     129,   129,   130,   130,   131,   131,   132,   133,   133,   134,
     134,   135,   135,   136,   136,   137,   137,   138,   139,   140,
     140,   141,   141,   142,   142,   143,   144,   145,   146,   146,
     147,   147,   148,   148,   148,   149,   149,   149,   150,   150,
     151,   152,   152,   152,   152,   152,   153,   153,   154,   154,
     155,   155,   155,   155,   155,   155,   155,   156,   156,   156,
     156,   156,   157,   157,   157,   157,   158,   158,   159,   160,
     160,   160,   161,   161,   161,   162,   162,   163,   163,   164,
     164,   165,   165,   166,   166,   166,   166,   166,   166,   166,
     166,   166,   166,   166,   166,   166,   166,   166,   167,   167,
     167,   168,   168,   168,   169,   169,   169,   169,   169,   169,
     170,   171,   172,   173,   173,   173,   173,   174,   174,   174,
     175,   175,   175,   175,   175,   175,   176,   177,   177,   177,
     178,   178,   179,   179,   180,   180,   181,   181,   182,   182,
     183,   183,   183,   184,   184,   185,   185,   186,   186,   187,
     187,   188,   188,   189,   189,   189,   189,   189,   189,   189,
     189,   189,   189,   189,   189,   189,   189,   189,   189,   190,
     190,   190,   191,   191,   192,   193,   194,   194,   195,   195,
     196,   197,   198,   199
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     0,     2,     0,     1,     0,     1,     1,
       1,     4,     3,     3,     1,     4,     0,     2,     5,     5,
       4,     3,     2,     2,     3,     2,     2,     2,     2,     2,
       8,     1,     1,     1,     1,     1,     1,     1,     3,     0,
       1,     0,     3,     1,     1,     1,     1,     2,     2,     3,
       3,     2,     2,     2,     1,     1,     2,     1,     2,     2,
       1,     1,     2,     2,     2,     8,     1,     1,     1,     1,
       2,     2,     2,     1,     1,     1,     3,     0,     2,     4,
       5,     0,     1,     0,     3,     1,     3,     1,     1,     0,
       3,     1,     3,     0,     1,     1,     0,     3,     1,     3,
       1,     1,     0,     1,     0,     2,     5,     1,     2,     3,
       6,     0,     2,     1,     3,     5,     5,     5,     5,     4,
       3,     6,     6,     5,     5,     5,     5,     5,     4,     7,
       0,     2,     0,     2,     2,     3,     2,     3,     1,     3,
       4,     2,     2,     2,     2,     2,     1,     4,     0,     2,
       2,     2,     2,     2,     3,     3,     3,     3,     6,     9,
       3,     6,     3,     6,     9,     9,     1,     3,     1,     2,
       3,     1,     7,     5,    12,     5,     2,     1,     1,     0,
       3,     1,     3,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     2,     1,     1,     1,     1,
       1,     0,     1,     3,     0,     1,     5,     5,     5,     4,
       3,     1,     1,     1,     3,     4,     3,     1,     1,     1,
       1,     4,     3,     4,     4,     4,     3,     7,     5,     6,
       1,     3,     1,     3,     3,     2,     3,     2,     0,     3,
       0,     1,     3,     1,     2,     1,     2,     0,     4,     1,
       3,     1,     0,     6,     8,     4,     3,     5,     4,    11,
       9,    12,    14,     6,     8,     5,     7,     3,     3,     0,
       2,     4,     1,     3,     1,     1,     2,     5,     1,     3,
       1,     1,     2,     2
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       3,     0,     0,     0,     2,     3,     9,     0,     0,     1,
       4,    14,     5,    12,    13,     6,     0,     0,     0,     0,
       0,     0,     0,     5,    33,    31,    32,     0,     0,     0,
      10,     0,   282,   283,   176,   211,   204,     0,    81,    81,
      81,     0,    89,    89,    89,    89,     0,    83,     0,     0,
       0,     5,    22,     0,     0,     0,    25,    26,    27,    28,
       0,    29,    23,     0,     0,     7,    17,    57,    43,    44,
      45,    46,    54,    55,     0,    41,    60,    61,     0,   178,
       0,     5,     0,    21,     0,   205,   204,     0,     0,    57,
       0,    66,    67,    68,    69,    73,     0,    82,     0,     0,
       0,     0,   166,     0,     0,     0,     0,     0,     0,     0,
       0,    96,     0,     0,     0,   168,     0,     0,     0,   150,
     151,    24,    89,    89,    89,    89,     0,    83,   141,   142,
     143,   144,   145,   153,   152,     8,    15,    53,    56,    47,
      48,    51,    52,    39,    59,    62,     0,     0,    20,     0,
     204,   201,   204,     0,   212,     0,     0,    70,    63,    64,
      71,     0,    72,    77,   160,   157,     0,   162,     0,   154,
     100,   101,     0,    91,    41,   111,   111,   111,   111,    95,
       0,     0,    98,     0,     0,     0,     0,     0,    87,    88,
       0,    85,   109,   156,   155,     0,    69,     0,   138,     0,
       7,     0,     0,     0,     0,     0,     0,    49,    50,    35,
      36,    37,    40,     0,    34,    41,    19,    11,     0,   202,
       0,     0,   204,   175,     0,     0,     0,    77,    79,    81,
       0,    81,    81,     0,    81,   167,    90,     0,    58,     0,
       0,     0,     0,     0,     0,   120,     0,    97,   111,   111,
      84,     0,   102,   130,     0,   136,   132,     0,   140,    18,
     111,   111,   111,   111,   111,     0,     0,    42,     0,   204,
     201,   204,   204,   209,    80,     0,    74,    75,     0,    78,
       0,     0,     0,     0,     0,     0,    92,   113,   112,   146,
     148,   115,   116,   117,   118,   119,    99,     0,     0,    86,
     103,     0,   102,     0,     0,   135,   133,   134,   137,   139,
       0,     0,     0,     0,     0,   128,   102,    38,     0,     0,
     207,   203,   208,   206,     0,    76,   161,     0,   158,     0,
       0,   163,     0,   179,     0,     0,   171,   148,     0,   121,
     122,     0,   108,   110,   131,   123,   124,   125,   126,   127,
       0,   235,   213,   204,   230,     0,    30,     0,    81,    81,
      81,   114,     0,     0,     0,     7,   149,   169,   104,     0,
     102,     0,     0,   234,     0,     0,     0,     0,   183,   184,
     185,   186,   191,   192,   193,   187,   188,   189,   190,    93,
     194,     0,   196,   197,     0,   181,    10,     0,     0,   170,
     147,     0,     0,   129,   210,     0,   214,   216,   231,    65,
     159,   165,   164,    94,   195,     0,   180,     0,   240,     0,
     105,   106,   199,   198,   200,   215,     0,   182,     0,     0,
       0,     0,     0,   252,     0,     0,     0,   211,   204,   173,
     241,   238,     0,   275,   204,     0,   204,     0,   278,     0,
       0,   251,     0,   272,     0,   204,     0,     0,   243,     0,
       0,   240,     0,     0,     0,     0,   280,   276,   204,     0,
     211,   256,     0,   245,   204,     0,   267,     0,     0,   242,
     244,   268,     0,   172,     0,     0,   204,     0,   274,     0,
       0,   279,   255,     0,   258,   246,     0,   273,     0,     0,
     239,   217,   218,   219,   237,     0,     0,   232,     0,   204,
       0,   204,   211,   265,   281,     0,   257,   247,   211,   269,
       0,     0,     0,     0,   236,     0,   204,     0,     0,   277,
       0,   253,     0,     0,   263,   204,     0,     0,   204,     0,
     233,     0,     0,   204,   266,     0,   269,   211,   270,     0,
     220,     0,     0,     0,     0,   174,     0,     0,   249,     0,
     264,     0,   254,   228,     0,     0,     0,     0,     0,   226,
       0,   211,   260,   204,     0,   248,   271,     0,     0,     0,
       0,   222,     0,   229,     0,     0,   250,   225,   224,   223,
     221,   227,   259,     0,   211,   261,     0,   262
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     3,     4,    21,   136,   174,    75,     5,    13,    22,
      23,   211,   212,   213,   144,    76,   175,    77,    91,    92,
      93,    94,    95,   214,   278,   227,   228,    53,    54,    98,
     113,   190,   191,   105,   172,   414,   182,   110,   183,   173,
     301,   402,   302,   303,    55,   240,   288,    56,    57,    58,
     111,    59,   128,   129,   130,   131,   132,   305,   255,   198,
     199,   334,    61,   291,   335,   336,    63,    64,   103,   116,
     337,   338,    24,    80,   363,   394,   395,   425,   220,    87,
     353,   438,   155,   354,   506,   551,   539,   507,   355,   508,
     319,   485,   461,   439,   457,   472,   531,   559,   452,   458,
     534,   454,   489,   444,   448,   449,   468,   515,    25,    26
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -462
static const yytype_int16 yypact[] =
{
     306,    -2,    -2,    36,  -462,   306,  -462,    73,    73,  -462,
    -462,  -462,    19,  -462,  -462,    24,    -2,   175,    -2,    -2,
     203,   239,    43,    19,  -462,  -462,  -462,   494,    50,    83,
    -462,   271,  -462,  -462,  -462,  -462,   -26,   357,   123,   123,
      17,    83,    87,    87,    87,    87,    95,   100,    -2,   140,
     127,    19,  -462,   159,   162,   185,  -462,  -462,  -462,  -462,
     323,  -462,  -462,   198,   206,   223,  -462,   231,  -462,   156,
    -462,  -462,  -462,  -462,   261,   187,  -462,  -462,   221,  -462,
      83,    19,   282,  -462,   241,    52,   -26,   249,   600,  -462,
     582,   221,   247,   262,  -462,    81,    83,  -462,    83,    83,
     279,    83,   276,   275,     3,    -2,    -2,    -2,    -2,   165,
     294,   299,   344,    -2,   291,  -462,   316,   409,   319,  -462,
    -462,  -462,    87,    87,    87,    87,   294,   100,  -462,  -462,
    -462,  -462,  -462,  -462,  -462,  -462,  -462,  -462,  -462,  -462,
     327,  -462,  -462,   567,  -462,  -462,    -2,   322,  -462,   347,
     -26,   353,   -26,   325,  -462,   356,   336,    40,  -462,  -462,
    -462,   351,  -462,   -15,    66,    77,   360,   181,    83,  -462,
    -462,  -462,   361,   355,   352,   367,   367,   367,   367,  -462,
      -2,   362,   371,   364,   345,    -2,   400,    -2,  -462,  -462,
     370,   375,   382,  -462,  -462,    -2,    53,    -2,   381,   383,
     223,    -2,    -2,    -2,    -2,    -2,    -2,  -462,  -462,  -462,
    -462,   384,  -462,   385,  -462,   352,  -462,  -462,   389,   390,
     378,   399,   -26,  -462,    -2,    -2,   349,   402,  -462,   123,
     567,   123,   123,   567,   123,  -462,  -462,     3,  -462,    83,
     209,   209,   209,   209,   403,  -462,   400,  -462,   367,   367,
    -462,   344,   452,   401,   350,  -462,   404,   409,  -462,  -462,
     367,   367,   367,   367,   367,   213,   567,  -462,   423,   -26,
     353,   -26,   -26,  -462,  -462,   424,  -462,   431,   422,  -462,
     426,   430,   428,    83,   432,   440,  -462,   447,  -462,  -462,
     311,  -462,  -462,  -462,  -462,  -462,  -462,   209,   209,  -462,
    -462,   582,     6,   449,   582,  -462,  -462,  -462,  -462,  -462,
     209,   209,   209,   209,   209,  -462,   452,  -462,   374,   453,
    -462,  -462,  -462,  -462,   446,  -462,  -462,   450,  -462,    91,
     451,  -462,    83,   467,   492,   457,  -462,   311,   463,  -462,
    -462,    -2,  -462,  -462,  -462,  -462,  -462,  -462,  -462,  -462,
     461,  -462,    -2,   -26,   476,   456,  -462,   582,   123,   123,
     123,  -462,   280,   615,   488,   223,  -462,  -462,   480,   514,
      10,   482,   582,  -462,   515,   517,   518,   521,  -462,  -462,
    -462,  -462,  -462,  -462,  -462,  -462,  -462,  -462,  -462,   557,
    -462,   533,  -462,  -462,   534,   543,   545,   423,    -2,  -462,
    -462,   539,   562,  -462,  -462,   212,  -462,  -462,  -462,  -462,
    -462,  -462,  -462,  -462,  -462,   603,  -462,   343,   168,   423,
    -462,  -462,  -462,  -462,   431,  -462,    -2,  -462,   558,   554,
     558,   586,   575,   595,   598,   558,   572,   454,   -26,  -462,
    -462,   636,   423,  -462,   -26,   602,   -26,    75,   580,   475,
     454,  -462,   583,   590,   597,   -26,   605,   592,   454,   249,
     585,   168,   588,   601,   599,   591,  -462,  -462,   -26,   586,
     437,  -462,   606,   454,   -26,   598,  -462,   591,   423,  -462,
    -462,  -462,   628,  -462,    49,   583,   -26,   558,  -462,   501,
     604,  -462,  -462,   609,  -462,  -462,   249,  -462,   529,   608,
    -462,  -462,  -462,  -462,  -462,    -2,   612,   610,   607,   -26,
     618,   -26,   454,  -462,  -462,   423,  -462,   639,   454,   646,
     583,   616,   582,   229,  -462,   249,   -26,   621,   620,  -462,
     617,  -462,   623,   540,  -462,   -26,    -2,    -2,   -26,   619,
    -462,    -2,   591,   -26,  -462,   647,   646,   454,  -462,   624,
    -462,   102,   -20,   622,    -2,  -462,   550,   625,   626,   627,
    -462,   629,  -462,  -462,    -2,   358,   631,    -2,    -2,  -462,
     193,   454,  -462,   -26,   647,  -462,  -462,   110,   630,    82,
      -2,  -462,   298,  -462,   632,   591,  -462,  -462,  -462,  -462,
    -462,  -462,  -462,   566,   454,  -462,   635,  -462
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -462,  -462,   700,  -462,  -196,    -1,    -7,   686,   705,   -10,
    -462,  -462,  -202,  -462,   -86,  -462,   -62,   -32,   -22,   -19,
    -462,  -106,   633,   -30,  -462,  -462,   487,  -462,  -462,     0,
     593,   464,  -462,   -24,   481,  -462,  -462,   611,   471,  -462,
     354,  -462,  -462,  -241,  -462,   -23,   387,  -462,  -462,  -462,
     -68,  -462,  -462,  -462,  -462,  -462,  -462,  -462,   465,  -462,
     468,   706,  -462,  -136,   391,   708,  -462,  -462,   563,  -462,
    -462,   392,  -462,   369,  -462,   313,  -462,  -462,   466,   -83,
     211,   -18,  -416,  -462,  -462,  -422,  -462,  -462,  -322,   214,
    -385,  -462,  -462,   273,  -444,  -462,  -462,   161,  -462,  -412,
     192,   264,  -461,  -405,  -462,   272,  -462,  -462,  -462,  -462
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -178
static const yytype_int16 yytable[] =
{
       7,     8,    36,   153,   259,    78,   440,    96,    79,   102,
      31,   196,   418,    66,   480,    28,   498,    32,    33,   106,
     107,   108,    82,   300,    15,   446,   493,   300,   281,   495,
     455,   284,    85,   170,   441,   374,     9,   471,   473,    99,
     101,   118,   185,   481,   176,   177,   178,   114,   146,   440,
     408,   192,    97,     6,   171,   501,    86,   462,   205,   568,
      84,   342,   467,    27,   317,   404,   156,   218,   528,   221,
     226,   147,    16,    17,   532,   350,   160,   513,    18,    19,
     517,   556,   511,  -107,    89,    90,   519,   197,   238,   163,
      20,   164,   165,   499,   167,   404,   -16,   405,   201,   202,
     203,   204,   100,   561,    30,   292,   293,   294,   181,   541,
      68,    69,    70,    71,    72,    73,    74,   160,   502,   503,
      65,   548,   161,   248,   593,   249,    81,   584,   150,   268,
     529,  -132,   570,  -132,   151,   504,   102,   152,    30,   273,
     254,    84,   577,   579,   572,   215,   582,    11,   229,    12,
     596,   196,    84,   241,   242,   243,   230,   318,    97,   231,
     466,   339,   340,   161,   564,   162,    84,   565,   589,   400,
     566,   567,   104,   359,   345,   346,   347,   348,   349,   244,
     109,   595,   563,   181,   564,   112,   320,   565,   322,   323,
     566,   567,   564,   587,   253,   565,   256,   115,   566,   567,
     260,   261,   262,   263,   264,   265,   117,   287,   428,   429,
     430,   431,   432,   433,   434,   435,   179,   436,   137,   277,
       6,   180,   138,   274,   275,   297,   298,   197,    29,   280,
      30,   282,   283,   119,   285,   501,   120,   310,   311,   312,
     313,   314,     1,     2,   437,    37,    38,    39,    40,    41,
      42,    43,    44,    45,    46,    47,    84,    48,    49,   121,
     233,    50,    84,   234,    89,    90,   143,    30,   422,   423,
     371,   341,   133,   583,   344,   564,   329,    34,   565,    35,
     134,   566,   567,   289,    30,   290,  -177,   315,   352,   316,
      68,    69,    70,    71,    72,    73,    74,   135,   502,   503,
     287,   378,   145,   379,   380,   381,   382,   383,   384,     1,
       2,   385,   386,   387,   388,    51,   149,    37,    38,    39,
      40,    41,   139,   140,   141,   142,   154,   352,   158,    48,
      49,   389,   390,    50,   122,   123,   124,   125,   126,   127,
     368,    78,   352,   159,    79,    83,    84,   166,   391,   169,
     333,   370,   392,   393,   168,   459,   148,    84,   375,   376,
     377,   463,   397,   465,   378,   193,   379,   380,   381,   382,
     383,   384,   477,    88,   385,   386,   387,   388,   591,   184,
     564,   188,   189,   565,   186,   490,   566,   567,   207,   208,
     194,   496,    89,    90,   389,   390,   200,   419,   424,   216,
       6,   180,   217,   510,    30,   276,   306,   307,   222,    89,
      90,   219,    30,     6,   578,   392,   393,   224,    68,    69,
      70,    71,    72,    73,    74,   442,   525,   195,   527,    30,
     223,   143,   225,   237,   474,    68,    69,    70,    71,    72,
      73,    74,   239,   542,    89,    90,   232,   236,   245,   246,
     247,   179,   549,   251,   505,   553,   250,   351,   252,   257,
     557,   271,   266,   258,    30,   267,   269,   509,   270,   300,
      68,    69,    70,    71,    72,    73,    74,   428,   429,   430,
     431,   432,   433,   434,   435,   272,   436,   226,   304,   295,
     585,   254,   537,   505,   428,   429,   430,   431,   432,   433,
     434,   435,   535,   436,   521,   318,    84,   324,   325,   326,
     327,   328,   330,    35,   492,   428,   429,   430,   431,   432,
     433,   434,   435,   331,   436,   332,   343,   356,   357,    67,
      35,   333,   358,   360,   365,   550,   552,   367,   369,   373,
     555,   428,   429,   430,   431,   432,   433,   434,   435,    30,
     436,   470,   362,   550,   372,    68,    69,    70,    71,    72,
      73,    74,   399,   550,   550,   401,   581,   550,   407,   428,
     429,   430,   431,   432,   433,   434,   435,   512,   436,   590,
     428,   429,   430,   431,   432,   433,   434,   435,   403,   436,
     428,   429,   430,   431,   432,   433,   434,   435,   409,   436,
     410,   411,    89,    90,   412,   518,   428,   429,   430,   431,
     432,   433,   434,   435,   413,   436,   547,    89,    90,   415,
     416,   417,    30,   209,   210,   420,   571,    -9,    68,    69,
      70,    71,    72,    73,    74,    89,   421,    30,   426,   445,
     443,   447,   594,    68,    69,    70,    71,    72,    73,    74,
      67,   450,   451,   453,   456,    30,   460,   464,   469,    35,
     478,    68,    69,    70,    71,    72,    73,    74,   475,   479,
     396,   476,   482,   484,   488,   486,    68,    69,    70,    71,
      72,    73,    74,   494,   500,   487,   516,   522,   523,   530,
     514,   520,   526,   524,   533,   536,   543,   544,   554,   545,
     546,   562,   558,   573,   574,    10,   576,    52,   569,   592,
     575,   580,   597,    14,   279,   299,   588,   296,   286,   361,
     206,   308,   187,   157,   406,   309,   364,    60,   366,    62,
     427,   235,   398,   538,   483,   586,   321,   540,   560,   497,
       0,   491
};

static const yytype_int16 yycheck[] =
{
       1,     2,    20,    86,   200,    27,   418,    37,    27,    41,
      17,   117,   397,    23,   458,    16,   477,    18,    19,    43,
      44,    45,    29,    17,     5,   430,   470,    17,   230,   473,
     435,   233,    58,    30,   419,   357,     0,   449,   450,    39,
      40,    51,   110,   459,   106,   107,   108,    48,    80,   461,
     372,   113,    35,    55,    51,     6,    82,   442,   126,    79,
      75,   302,   447,    39,   266,    85,    88,   150,   512,   152,
      85,    81,    53,    54,   518,   316,    36,   489,    59,    60,
     496,   542,   487,    77,    35,    36,   498,   117,   174,    96,
      71,    98,    99,   478,   101,    85,    77,    87,   122,   123,
     124,   125,    85,   547,    55,   241,   242,   243,   109,   525,
      61,    62,    63,    64,    65,    66,    67,    36,    69,    70,
      77,   533,    82,   185,   585,   187,    76,   571,    76,   215,
     515,    78,   554,    80,    82,    86,   168,    85,    55,   222,
      87,    75,   564,   565,   556,   146,   568,    74,    82,    76,
     594,   257,    75,   176,   177,   178,    79,    82,    35,    82,
      85,   297,   298,    82,    82,    84,    75,    85,    86,   365,
      88,    89,    85,    82,   310,   311,   312,   313,   314,   180,
      85,   593,    80,   184,    82,    85,   269,    85,   271,   272,
      88,    89,    82,    83,   195,    85,   197,    57,    88,    89,
     201,   202,   203,   204,   205,   206,    79,   239,    40,    41,
      42,    43,    44,    45,    46,    47,    51,    49,    62,   226,
      55,    56,    66,   224,   225,   248,   249,   257,    53,   229,
      55,   231,   232,    74,   234,     6,    74,   260,   261,   262,
     263,   264,     3,     4,    76,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    75,    18,    19,    74,
      79,    22,    75,    82,    35,    36,    79,    55,    56,    57,
     353,   301,    74,    80,   304,    82,   283,    74,    85,    76,
      74,    88,    89,    74,    55,    76,    55,    74,   318,    76,
      61,    62,    63,    64,    65,    66,    67,    74,    69,    70,
     332,    21,    81,    23,    24,    25,    26,    27,    28,     3,
       4,    31,    32,    33,    34,    76,    75,     6,     7,     8,
       9,    10,    61,    62,    63,    64,    77,   357,    81,    18,
      19,    51,    52,    22,    11,    12,    13,    14,    15,    16,
     341,   363,   372,    81,   363,    74,    75,    68,    68,    74,
      39,   352,    72,    73,    78,   438,    74,    75,   358,   359,
     360,   444,   363,   446,    21,    74,    23,    24,    25,    26,
      27,    28,   455,    16,    31,    32,    33,    34,    80,    85,
      82,    37,    38,    85,    85,   468,    88,    89,    61,    62,
      74,   474,    35,    36,    51,    52,    77,   398,   405,    77,
      55,    56,    55,   486,    55,    56,    56,    57,    83,    35,
      36,    58,    55,    55,    56,    72,    73,    81,    61,    62,
      63,    64,    65,    66,    67,   426,   509,    18,   511,    55,
      74,    79,    81,    78,   452,    61,    62,    63,    64,    65,
      66,    67,    75,   526,    35,    36,    86,    86,    86,    78,
      86,    51,   535,    78,   484,   538,    86,    83,    76,    78,
     543,    83,    78,    80,    55,    80,    77,   485,    78,    17,
      61,    62,    63,    64,    65,    66,    67,    40,    41,    42,
      43,    44,    45,    46,    47,    86,    49,    85,    87,    86,
     573,    87,   522,   523,    40,    41,    42,    43,    44,    45,
      46,    47,   520,    49,   505,    82,    75,    83,    86,    83,
      80,    83,    80,    76,    77,    40,    41,    42,    43,    44,
      45,    46,    47,    83,    49,    78,    77,    74,    82,    35,
      76,    39,    82,    82,    77,   536,   537,    74,    77,    83,
     541,    40,    41,    42,    43,    44,    45,    46,    47,    55,
      49,    76,    85,   554,    78,    61,    62,    63,    64,    65,
      66,    67,    74,   564,   565,    85,   567,   568,    86,    40,
      41,    42,    43,    44,    45,    46,    47,    76,    49,   580,
      40,    41,    42,    43,    44,    45,    46,    47,    74,    49,
      40,    41,    42,    43,    44,    45,    46,    47,    83,    49,
      83,    83,    35,    36,    83,    76,    40,    41,    42,    43,
      44,    45,    46,    47,    57,    49,    76,    35,    36,    86,
      86,    78,    55,    56,    57,    86,    76,    82,    61,    62,
      63,    64,    65,    66,    67,    35,    74,    55,    35,    85,
      82,    55,    76,    61,    62,    63,    64,    65,    66,    67,
      35,    76,    57,    55,    82,    55,    20,    55,    78,    76,
      55,    61,    62,    63,    64,    65,    66,    67,    78,    77,
      55,    74,    87,    85,    83,    74,    61,    62,    63,    64,
      65,    66,    67,    77,    56,    86,    77,    75,    78,    50,
      86,    83,    74,    86,    48,    79,    75,    77,    79,    82,
      77,    77,    55,    78,    78,     5,    77,    21,    86,    77,
      83,    80,    77,     8,   227,   251,    86,   246,   237,   332,
     127,   256,   111,    90,   370,   257,   334,    21,   337,    21,
     417,   168,   363,   522,   461,   574,   270,   523,   546,   475,
      -1,   469
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     3,     4,    91,    92,    97,    55,    95,    95,     0,
      92,    74,    76,    98,    98,     5,    53,    54,    59,    60,
      71,    93,    99,   100,   162,   198,   199,    39,    95,    53,
      55,    96,    95,    95,    74,    76,   171,     6,     7,     8,
       9,    10,    11,    12,    13,    14,    15,    16,    18,    19,
      22,    76,    97,   117,   118,   134,   137,   138,   139,   141,
     151,   152,   155,   156,   157,    77,    99,    35,    61,    62,
      63,    64,    65,    66,    67,    96,   105,   107,   108,   109,
     163,    76,    96,    74,    75,    58,    82,   169,    16,    35,
      36,   108,   109,   110,   111,   112,   113,    35,   119,   119,
      85,   119,   107,   158,    85,   123,   123,   123,   123,    85,
     127,   140,    85,   120,    95,    57,   159,    79,    99,    74,
      74,    74,    11,    12,    13,    14,    15,    16,   142,   143,
     144,   145,   146,    74,    74,    74,    94,    62,    66,    61,
      62,    63,    64,    79,   104,    81,   107,    99,    74,    75,
      76,    82,    85,   169,    77,   172,   108,   112,    81,    81,
      36,    82,    84,    96,    96,    96,    68,    96,    78,    74,
      30,    51,   124,   129,    95,   106,   106,   106,   106,    51,
      56,    95,   126,   128,    85,   140,    85,   127,    37,    38,
     121,   122,   106,    74,    74,    18,   111,   113,   149,   150,
      77,   123,   123,   123,   123,   140,   120,    61,    62,    56,
      57,   101,   102,   103,   113,    95,    77,    55,   169,    58,
     168,   169,    83,    74,    81,    81,    85,   115,   116,    82,
      79,    82,    86,    79,    82,   158,    86,    78,   104,    75,
     135,   135,   135,   135,    95,    86,    78,    86,   106,   106,
      86,    78,    76,    95,    87,   148,    95,    78,    80,    94,
      95,    95,    95,    95,    95,    95,    78,    80,   104,    77,
      78,    83,    86,   169,    95,    95,    56,    96,   114,   116,
     119,   102,   119,   119,   102,   119,   124,   107,   136,    74,
      76,   153,   153,   153,   153,    86,   128,   135,   135,   121,
      17,   130,   132,   133,    87,   147,    56,    57,   148,   150,
     135,   135,   135,   135,   135,    74,    76,   102,    82,   180,
     169,   168,   169,   169,    83,    86,    83,    80,    83,    96,
      80,    83,    78,    39,   151,   154,   155,   160,   161,   153,
     153,   113,   133,    77,   113,   153,   153,   153,   153,   153,
     133,    83,   113,   170,   173,   178,    74,    82,    82,    82,
      82,   136,    85,   164,   161,    77,   154,    74,    95,    77,
      95,   169,    78,    83,   178,   119,   119,   119,    21,    23,
      24,    25,    26,    27,    28,    31,    32,    33,    34,    51,
      52,    68,    72,    73,   165,   166,    55,    95,   163,    74,
      94,    85,   131,    74,    85,    87,   130,    86,   178,    83,
      83,    83,    83,    57,   125,    86,    86,    78,   180,    95,
      86,    74,    56,    57,    96,   167,    35,   165,    40,    41,
      42,    43,    44,    45,    46,    47,    49,    76,   171,   183,
     189,   180,    95,    82,   193,    85,   193,    55,   194,   195,
      76,    57,   188,    55,   191,   193,    82,   184,   189,   169,
      20,   182,   180,   169,    55,   169,    85,   180,   196,    78,
      76,   189,   185,   189,   171,    78,    74,   169,    55,    77,
     184,   172,    87,   183,    85,   181,    74,    86,    83,   192,
     169,   195,    77,   184,    77,   184,   169,   191,   192,   180,
      56,     6,    69,    70,    86,   113,   174,   177,   179,   171,
     169,   193,    76,   189,    86,   197,    77,   172,    76,   189,
      83,    95,    75,    78,    86,   169,    74,   169,   184,   180,
      50,   186,   184,    48,   190,   171,    79,   113,   170,   176,
     179,   172,   169,    75,    77,    82,    77,    76,   189,   169,
      95,   175,    95,   169,    79,    95,   192,   169,    55,   187,
     190,   184,    77,    80,    82,    85,    88,    89,    79,    86,
     175,    76,   189,    78,    78,    83,    77,   175,    56,   175,
      80,    95,   175,    80,   184,   169,   187,    83,    86,    86,
      95,    80,    77,   192,    76,   189,   184,    77
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
    { if((yyvsp[(3) - (5)].conslist)) (yyvsp[(3) - (5)].conslist)->setExtern((yyvsp[(1) - (5)].intval)); (yyval.construct) = (yyvsp[(3) - (5)].conslist); }
    break;

  case 19:

/* Line 1455 of yacc.c  */
#line 218 "xi-grammar.y"
    { (yyval.construct) = new Scope((yyvsp[(2) - (5)].strval), (yyvsp[(4) - (5)].conslist)); }
    break;

  case 20:

/* Line 1455 of yacc.c  */
#line 220 "xi-grammar.y"
    { (yyval.construct) = new UsingScope((yyvsp[(3) - (4)].strval), false); }
    break;

  case 21:

/* Line 1455 of yacc.c  */
#line 222 "xi-grammar.y"
    { (yyval.construct) = new UsingScope((yyvsp[(2) - (3)].strval), true); }
    break;

  case 22:

/* Line 1455 of yacc.c  */
#line 224 "xi-grammar.y"
    { (yyvsp[(2) - (2)].module)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].module); }
    break;

  case 23:

/* Line 1455 of yacc.c  */
#line 226 "xi-grammar.y"
    { (yyvsp[(2) - (2)].member)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].member); }
    break;

  case 24:

/* Line 1455 of yacc.c  */
#line 228 "xi-grammar.y"
    { (yyvsp[(2) - (3)].message)->setExtern((yyvsp[(1) - (3)].intval)); (yyval.construct) = (yyvsp[(2) - (3)].message); }
    break;

  case 25:

/* Line 1455 of yacc.c  */
#line 230 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 26:

/* Line 1455 of yacc.c  */
#line 232 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 27:

/* Line 1455 of yacc.c  */
#line 234 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 28:

/* Line 1455 of yacc.c  */
#line 236 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 29:

/* Line 1455 of yacc.c  */
#line 238 "xi-grammar.y"
    { (yyvsp[(2) - (2)].templat)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].templat); }
    break;

  case 30:

/* Line 1455 of yacc.c  */
#line 240 "xi-grammar.y"
    {
          Entry *e = new Entry(lineno, 0, (yyvsp[(3) - (8)].type), (yyvsp[(5) - (8)].strval), (yyvsp[(7) - (8)].plist), 0, 0, 0, 0, 0);
          int isExtern = 1;
          e->setExtern(isExtern);
          e->targs = (yyvsp[(6) - (8)].tparlist);
          e->label = new XStr;
          (yyvsp[(4) - (8)].ntype)->print(*e->label);
          (yyval.construct) = e;
        }
    break;

  case 31:

/* Line 1455 of yacc.c  */
#line 250 "xi-grammar.y"
    { (yyval.construct) = NULL; }
    break;

  case 32:

/* Line 1455 of yacc.c  */
#line 252 "xi-grammar.y"
    { (yyval.construct) = NULL; }
    break;

  case 33:

/* Line 1455 of yacc.c  */
#line 254 "xi-grammar.y"
    { (yyval.construct) = (yyvsp[(1) - (1)].accelBlock); }
    break;

  case 34:

/* Line 1455 of yacc.c  */
#line 258 "xi-grammar.y"
    { (yyval.tparam) = new TParamType((yyvsp[(1) - (1)].type)); }
    break;

  case 35:

/* Line 1455 of yacc.c  */
#line 260 "xi-grammar.y"
    { (yyval.tparam) = new TParamVal((yyvsp[(1) - (1)].strval)); }
    break;

  case 36:

/* Line 1455 of yacc.c  */
#line 262 "xi-grammar.y"
    { (yyval.tparam) = new TParamVal((yyvsp[(1) - (1)].strval)); }
    break;

  case 37:

/* Line 1455 of yacc.c  */
#line 266 "xi-grammar.y"
    { (yyval.tparlist) = new TParamList((yyvsp[(1) - (1)].tparam)); }
    break;

  case 38:

/* Line 1455 of yacc.c  */
#line 268 "xi-grammar.y"
    { (yyval.tparlist) = new TParamList((yyvsp[(1) - (3)].tparam), (yyvsp[(3) - (3)].tparlist)); }
    break;

  case 39:

/* Line 1455 of yacc.c  */
#line 272 "xi-grammar.y"
    { (yyval.tparlist) = 0; }
    break;

  case 40:

/* Line 1455 of yacc.c  */
#line 274 "xi-grammar.y"
    { (yyval.tparlist) = (yyvsp[(1) - (1)].tparlist); }
    break;

  case 41:

/* Line 1455 of yacc.c  */
#line 278 "xi-grammar.y"
    { (yyval.tparlist) = 0; }
    break;

  case 42:

/* Line 1455 of yacc.c  */
#line 280 "xi-grammar.y"
    { (yyval.tparlist) = (yyvsp[(2) - (3)].tparlist); }
    break;

  case 43:

/* Line 1455 of yacc.c  */
#line 284 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("int"); }
    break;

  case 44:

/* Line 1455 of yacc.c  */
#line 286 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long"); }
    break;

  case 45:

/* Line 1455 of yacc.c  */
#line 288 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("short"); }
    break;

  case 46:

/* Line 1455 of yacc.c  */
#line 290 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("char"); }
    break;

  case 47:

/* Line 1455 of yacc.c  */
#line 292 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned int"); }
    break;

  case 48:

/* Line 1455 of yacc.c  */
#line 294 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long"); }
    break;

  case 49:

/* Line 1455 of yacc.c  */
#line 296 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long"); }
    break;

  case 50:

/* Line 1455 of yacc.c  */
#line 298 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long long"); }
    break;

  case 51:

/* Line 1455 of yacc.c  */
#line 300 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned short"); }
    break;

  case 52:

/* Line 1455 of yacc.c  */
#line 302 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned char"); }
    break;

  case 53:

/* Line 1455 of yacc.c  */
#line 304 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long long"); }
    break;

  case 54:

/* Line 1455 of yacc.c  */
#line 306 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("float"); }
    break;

  case 55:

/* Line 1455 of yacc.c  */
#line 308 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("double"); }
    break;

  case 56:

/* Line 1455 of yacc.c  */
#line 310 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long double"); }
    break;

  case 57:

/* Line 1455 of yacc.c  */
#line 312 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("void"); }
    break;

  case 58:

/* Line 1455 of yacc.c  */
#line 315 "xi-grammar.y"
    { (yyval.ntype) = new NamedType((yyvsp[(1) - (2)].strval),(yyvsp[(2) - (2)].tparlist)); }
    break;

  case 59:

/* Line 1455 of yacc.c  */
#line 316 "xi-grammar.y"
    { 
                    const char* basename, *scope;
                    splitScopedName((yyvsp[(1) - (2)].strval), &scope, &basename);
                    (yyval.ntype) = new NamedType(basename, (yyvsp[(2) - (2)].tparlist), scope);
                }
    break;

  case 60:

/* Line 1455 of yacc.c  */
#line 324 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 61:

/* Line 1455 of yacc.c  */
#line 326 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ntype); }
    break;

  case 62:

/* Line 1455 of yacc.c  */
#line 330 "xi-grammar.y"
    { (yyval.ptype) = new PtrType((yyvsp[(1) - (2)].type)); }
    break;

  case 63:

/* Line 1455 of yacc.c  */
#line 334 "xi-grammar.y"
    { (yyvsp[(1) - (2)].ptype)->indirect(); (yyval.ptype) = (yyvsp[(1) - (2)].ptype); }
    break;

  case 64:

/* Line 1455 of yacc.c  */
#line 336 "xi-grammar.y"
    { (yyvsp[(1) - (2)].ptype)->indirect(); (yyval.ptype) = (yyvsp[(1) - (2)].ptype); }
    break;

  case 65:

/* Line 1455 of yacc.c  */
#line 340 "xi-grammar.y"
    { (yyval.ftype) = new FuncType((yyvsp[(1) - (8)].type), (yyvsp[(4) - (8)].strval), (yyvsp[(7) - (8)].plist)); }
    break;

  case 66:

/* Line 1455 of yacc.c  */
#line 344 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 67:

/* Line 1455 of yacc.c  */
#line 346 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 68:

/* Line 1455 of yacc.c  */
#line 348 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 69:

/* Line 1455 of yacc.c  */
#line 350 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ftype); }
    break;

  case 70:

/* Line 1455 of yacc.c  */
#line 353 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(2) - (2)].type); }
    break;

  case 71:

/* Line 1455 of yacc.c  */
#line 355 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (2)].type); }
    break;

  case 72:

/* Line 1455 of yacc.c  */
#line 359 "xi-grammar.y"
    { (yyval.type) = new ReferenceType((yyvsp[(1) - (2)].type)); }
    break;

  case 73:

/* Line 1455 of yacc.c  */
#line 361 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 74:

/* Line 1455 of yacc.c  */
#line 365 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 75:

/* Line 1455 of yacc.c  */
#line 367 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 76:

/* Line 1455 of yacc.c  */
#line 371 "xi-grammar.y"
    { (yyval.val) = (yyvsp[(2) - (3)].val); }
    break;

  case 77:

/* Line 1455 of yacc.c  */
#line 375 "xi-grammar.y"
    { (yyval.vallist) = 0; }
    break;

  case 78:

/* Line 1455 of yacc.c  */
#line 377 "xi-grammar.y"
    { (yyval.vallist) = new ValueList((yyvsp[(1) - (2)].val), (yyvsp[(2) - (2)].vallist)); }
    break;

  case 79:

/* Line 1455 of yacc.c  */
#line 381 "xi-grammar.y"
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[(2) - (4)].type), (yyvsp[(3) - (4)].strval), (yyvsp[(4) - (4)].vallist)); }
    break;

  case 80:

/* Line 1455 of yacc.c  */
#line 385 "xi-grammar.y"
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[(3) - (5)].type), (yyvsp[(5) - (5)].strval), 0, 1); }
    break;

  case 81:

/* Line 1455 of yacc.c  */
#line 389 "xi-grammar.y"
    { (yyval.intval) = 0;}
    break;

  case 82:

/* Line 1455 of yacc.c  */
#line 391 "xi-grammar.y"
    { (yyval.intval) = 0;}
    break;

  case 83:

/* Line 1455 of yacc.c  */
#line 395 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 84:

/* Line 1455 of yacc.c  */
#line 397 "xi-grammar.y"
    { 
		  /*
		  printf("Warning: Message attributes are being phased out.\n");
		  printf("Warning: Please remove them from interface files.\n");
		  */
		  (yyval.intval) = (yyvsp[(2) - (3)].intval); 
		}
    break;

  case 85:

/* Line 1455 of yacc.c  */
#line 407 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (1)].intval); }
    break;

  case 86:

/* Line 1455 of yacc.c  */
#line 409 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (3)].intval) | (yyvsp[(3) - (3)].intval); }
    break;

  case 87:

/* Line 1455 of yacc.c  */
#line 413 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 88:

/* Line 1455 of yacc.c  */
#line 415 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 89:

/* Line 1455 of yacc.c  */
#line 419 "xi-grammar.y"
    { (yyval.cattr) = 0; }
    break;

  case 90:

/* Line 1455 of yacc.c  */
#line 421 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(2) - (3)].cattr); }
    break;

  case 91:

/* Line 1455 of yacc.c  */
#line 425 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (1)].cattr); }
    break;

  case 92:

/* Line 1455 of yacc.c  */
#line 427 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (3)].cattr) | (yyvsp[(3) - (3)].cattr); }
    break;

  case 93:

/* Line 1455 of yacc.c  */
#line 431 "xi-grammar.y"
    { python_doc = NULL; (yyval.intval) = 0; }
    break;

  case 94:

/* Line 1455 of yacc.c  */
#line 433 "xi-grammar.y"
    { python_doc = (yyvsp[(1) - (1)].strval); (yyval.intval) = 0; }
    break;

  case 95:

/* Line 1455 of yacc.c  */
#line 437 "xi-grammar.y"
    { (yyval.cattr) = Chare::CPYTHON; }
    break;

  case 96:

/* Line 1455 of yacc.c  */
#line 441 "xi-grammar.y"
    { (yyval.cattr) = 0; }
    break;

  case 97:

/* Line 1455 of yacc.c  */
#line 443 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(2) - (3)].cattr); }
    break;

  case 98:

/* Line 1455 of yacc.c  */
#line 447 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (1)].cattr); }
    break;

  case 99:

/* Line 1455 of yacc.c  */
#line 449 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (3)].cattr) | (yyvsp[(3) - (3)].cattr); }
    break;

  case 100:

/* Line 1455 of yacc.c  */
#line 453 "xi-grammar.y"
    { (yyval.cattr) = Chare::CMIGRATABLE; }
    break;

  case 101:

/* Line 1455 of yacc.c  */
#line 455 "xi-grammar.y"
    { (yyval.cattr) = Chare::CPYTHON; }
    break;

  case 102:

/* Line 1455 of yacc.c  */
#line 459 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 103:

/* Line 1455 of yacc.c  */
#line 461 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 104:

/* Line 1455 of yacc.c  */
#line 464 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 105:

/* Line 1455 of yacc.c  */
#line 466 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 106:

/* Line 1455 of yacc.c  */
#line 469 "xi-grammar.y"
    { (yyval.mv) = new MsgVar((yyvsp[(2) - (5)].type), (yyvsp[(3) - (5)].strval), (yyvsp[(1) - (5)].intval), (yyvsp[(4) - (5)].intval)); }
    break;

  case 107:

/* Line 1455 of yacc.c  */
#line 473 "xi-grammar.y"
    { (yyval.mvlist) = new MsgVarList((yyvsp[(1) - (1)].mv)); }
    break;

  case 108:

/* Line 1455 of yacc.c  */
#line 475 "xi-grammar.y"
    { (yyval.mvlist) = new MsgVarList((yyvsp[(1) - (2)].mv), (yyvsp[(2) - (2)].mvlist)); }
    break;

  case 109:

/* Line 1455 of yacc.c  */
#line 479 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, (yyvsp[(3) - (3)].ntype)); }
    break;

  case 110:

/* Line 1455 of yacc.c  */
#line 481 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, (yyvsp[(3) - (6)].ntype), (yyvsp[(5) - (6)].mvlist)); }
    break;

  case 111:

/* Line 1455 of yacc.c  */
#line 485 "xi-grammar.y"
    { (yyval.typelist) = 0; }
    break;

  case 112:

/* Line 1455 of yacc.c  */
#line 487 "xi-grammar.y"
    { (yyval.typelist) = (yyvsp[(2) - (2)].typelist); }
    break;

  case 113:

/* Line 1455 of yacc.c  */
#line 491 "xi-grammar.y"
    { (yyval.typelist) = new TypeList((yyvsp[(1) - (1)].ntype)); }
    break;

  case 114:

/* Line 1455 of yacc.c  */
#line 493 "xi-grammar.y"
    { (yyval.typelist) = new TypeList((yyvsp[(1) - (3)].ntype), (yyvsp[(3) - (3)].typelist)); }
    break;

  case 115:

/* Line 1455 of yacc.c  */
#line 497 "xi-grammar.y"
    { (yyval.chare) = new Chare(lineno, (yyvsp[(2) - (5)].cattr)|Chare::CCHARE, (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 116:

/* Line 1455 of yacc.c  */
#line 499 "xi-grammar.y"
    { (yyval.chare) = new MainChare(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 117:

/* Line 1455 of yacc.c  */
#line 503 "xi-grammar.y"
    { (yyval.chare) = new Group(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 118:

/* Line 1455 of yacc.c  */
#line 507 "xi-grammar.y"
    { (yyval.chare) = new NodeGroup(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 119:

/* Line 1455 of yacc.c  */
#line 511 "xi-grammar.y"
    {/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",(yyvsp[(2) - (4)].strval));
			(yyval.ntype) = new NamedType(buf); 
		}
    break;

  case 120:

/* Line 1455 of yacc.c  */
#line 517 "xi-grammar.y"
    { (yyval.ntype) = new NamedType((yyvsp[(2) - (3)].strval)); }
    break;

  case 121:

/* Line 1455 of yacc.c  */
#line 521 "xi-grammar.y"
    {  (yyval.chare) = new Array(lineno, (yyvsp[(2) - (6)].cattr), (yyvsp[(3) - (6)].ntype), (yyvsp[(4) - (6)].ntype), (yyvsp[(5) - (6)].typelist), (yyvsp[(6) - (6)].mbrlist)); }
    break;

  case 122:

/* Line 1455 of yacc.c  */
#line 523 "xi-grammar.y"
    {  (yyval.chare) = new Array(lineno, (yyvsp[(3) - (6)].cattr), (yyvsp[(2) - (6)].ntype), (yyvsp[(4) - (6)].ntype), (yyvsp[(5) - (6)].typelist), (yyvsp[(6) - (6)].mbrlist)); }
    break;

  case 123:

/* Line 1455 of yacc.c  */
#line 527 "xi-grammar.y"
    { (yyval.chare) = new Chare(lineno, (yyvsp[(2) - (5)].cattr)|Chare::CCHARE, new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist));}
    break;

  case 124:

/* Line 1455 of yacc.c  */
#line 529 "xi-grammar.y"
    { (yyval.chare) = new MainChare(lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 125:

/* Line 1455 of yacc.c  */
#line 533 "xi-grammar.y"
    { (yyval.chare) = new Group(lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 126:

/* Line 1455 of yacc.c  */
#line 537 "xi-grammar.y"
    { (yyval.chare) = new NodeGroup( lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 127:

/* Line 1455 of yacc.c  */
#line 541 "xi-grammar.y"
    { (yyval.chare) = new Array( lineno, 0, (yyvsp[(2) - (5)].ntype), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 128:

/* Line 1455 of yacc.c  */
#line 545 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[(3) - (4)].strval))); }
    break;

  case 129:

/* Line 1455 of yacc.c  */
#line 547 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[(3) - (7)].strval)), (yyvsp[(5) - (7)].mvlist)); }
    break;

  case 130:

/* Line 1455 of yacc.c  */
#line 551 "xi-grammar.y"
    { (yyval.type) = 0; }
    break;

  case 131:

/* Line 1455 of yacc.c  */
#line 553 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(2) - (2)].type); }
    break;

  case 132:

/* Line 1455 of yacc.c  */
#line 557 "xi-grammar.y"
    { (yyval.strval) = 0; }
    break;

  case 133:

/* Line 1455 of yacc.c  */
#line 559 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(2) - (2)].strval); }
    break;

  case 134:

/* Line 1455 of yacc.c  */
#line 561 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(2) - (2)].strval); }
    break;

  case 135:

/* Line 1455 of yacc.c  */
#line 565 "xi-grammar.y"
    { (yyval.tvar) = new TType(new NamedType((yyvsp[(2) - (3)].strval)), (yyvsp[(3) - (3)].type)); }
    break;

  case 136:

/* Line 1455 of yacc.c  */
#line 567 "xi-grammar.y"
    { (yyval.tvar) = new TFunc((yyvsp[(1) - (2)].ftype), (yyvsp[(2) - (2)].strval)); }
    break;

  case 137:

/* Line 1455 of yacc.c  */
#line 569 "xi-grammar.y"
    { (yyval.tvar) = new TName((yyvsp[(1) - (3)].type), (yyvsp[(2) - (3)].strval), (yyvsp[(3) - (3)].strval)); }
    break;

  case 138:

/* Line 1455 of yacc.c  */
#line 573 "xi-grammar.y"
    { (yyval.tvarlist) = new TVarList((yyvsp[(1) - (1)].tvar)); }
    break;

  case 139:

/* Line 1455 of yacc.c  */
#line 575 "xi-grammar.y"
    { (yyval.tvarlist) = new TVarList((yyvsp[(1) - (3)].tvar), (yyvsp[(3) - (3)].tvarlist)); }
    break;

  case 140:

/* Line 1455 of yacc.c  */
#line 579 "xi-grammar.y"
    { (yyval.tvarlist) = (yyvsp[(3) - (4)].tvarlist); }
    break;

  case 141:

/* Line 1455 of yacc.c  */
#line 583 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 142:

/* Line 1455 of yacc.c  */
#line 585 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 143:

/* Line 1455 of yacc.c  */
#line 587 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 144:

/* Line 1455 of yacc.c  */
#line 589 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 145:

/* Line 1455 of yacc.c  */
#line 591 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].message)); (yyvsp[(2) - (2)].message)->setTemplate((yyval.templat)); }
    break;

  case 146:

/* Line 1455 of yacc.c  */
#line 595 "xi-grammar.y"
    { (yyval.mbrlist) = 0; }
    break;

  case 147:

/* Line 1455 of yacc.c  */
#line 597 "xi-grammar.y"
    { (yyval.mbrlist) = (yyvsp[(2) - (4)].mbrlist); }
    break;

  case 148:

/* Line 1455 of yacc.c  */
#line 601 "xi-grammar.y"
    { 
		  Entry *tempEntry;
		  if (!connectEntries->empty()) {
		    tempEntry = connectEntries->begin();
		    MemberList *ml;
		    ml = new MemberList(tempEntry, 0);
		    tempEntry = connectEntries->next();
		    for(; !connectEntries->end(); tempEntry = connectEntries->next()) {
                      ml->appendMember(tempEntry); 
		    }
		    while (!connectEntries->empty())
		      connectEntries->pop();
                    (yyval.mbrlist) = ml; 
		  }
		  else {
		    (yyval.mbrlist) = 0; 
                  }
		}
    break;

  case 149:

/* Line 1455 of yacc.c  */
#line 620 "xi-grammar.y"
    { (yyval.mbrlist) = new MemberList((yyvsp[(1) - (2)].member), (yyvsp[(2) - (2)].mbrlist)); }
    break;

  case 150:

/* Line 1455 of yacc.c  */
#line 624 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (2)].readonly); }
    break;

  case 151:

/* Line 1455 of yacc.c  */
#line 626 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (2)].readonly); }
    break;

  case 153:

/* Line 1455 of yacc.c  */
#line 629 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (2)].member); }
    break;

  case 154:

/* Line 1455 of yacc.c  */
#line 631 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(2) - (3)].pupable); }
    break;

  case 155:

/* Line 1455 of yacc.c  */
#line 633 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(2) - (3)].includeFile); }
    break;

  case 156:

/* Line 1455 of yacc.c  */
#line 635 "xi-grammar.y"
    { (yyval.member) = new ClassDeclaration(lineno,(yyvsp[(2) - (3)].strval)); }
    break;

  case 157:

/* Line 1455 of yacc.c  */
#line 639 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 1); }
    break;

  case 158:

/* Line 1455 of yacc.c  */
#line 641 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 1); }
    break;

  case 159:

/* Line 1455 of yacc.c  */
#line 643 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[(3) - (9)].strval)) + '<' +
					    ((yyvsp[(5) - (9)].tparlist))->to_string() + '>').c_str()),
				    1);
		}
    break;

  case 160:

/* Line 1455 of yacc.c  */
#line 649 "xi-grammar.y"
    { printf("Warning: deprecated use of initcall. Use initnode or initproc instead.\n"); 
		  (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 1); }
    break;

  case 161:

/* Line 1455 of yacc.c  */
#line 652 "xi-grammar.y"
    { printf("Warning: deprecated use of initcall. Use initnode or initproc instead.\n");
		  (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 1); }
    break;

  case 162:

/* Line 1455 of yacc.c  */
#line 657 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 0); }
    break;

  case 163:

/* Line 1455 of yacc.c  */
#line 659 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 0); }
    break;

  case 164:

/* Line 1455 of yacc.c  */
#line 661 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[(3) - (9)].strval)) + '<' +
					    ((yyvsp[(5) - (9)].tparlist))->to_string() + '>').c_str()),
				    0);
		}
    break;

  case 165:

/* Line 1455 of yacc.c  */
#line 667 "xi-grammar.y"
    {
                  InitCall* rtn = new InitCall(lineno, (yyvsp[(6) - (9)].strval), 0);
                  rtn->setAccel();
                  (yyval.member) = rtn;
		}
    break;

  case 166:

/* Line 1455 of yacc.c  */
#line 675 "xi-grammar.y"
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[(1) - (1)].ntype),0); }
    break;

  case 167:

/* Line 1455 of yacc.c  */
#line 677 "xi-grammar.y"
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[(1) - (3)].ntype),(yyvsp[(3) - (3)].pupable)); }
    break;

  case 168:

/* Line 1455 of yacc.c  */
#line 680 "xi-grammar.y"
    { (yyval.includeFile) = new IncludeFile(lineno,(yyvsp[(1) - (1)].strval)); }
    break;

  case 169:

/* Line 1455 of yacc.c  */
#line 684 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (2)].entry); }
    break;

  case 170:

/* Line 1455 of yacc.c  */
#line 686 "xi-grammar.y"
    {
                  (yyvsp[(2) - (3)].entry)->tspec = (yyvsp[(1) - (3)].tvarlist);
                  (yyval.member) = (yyvsp[(2) - (3)].entry);
                }
    break;

  case 171:

/* Line 1455 of yacc.c  */
#line 691 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].member); }
    break;

  case 172:

/* Line 1455 of yacc.c  */
#line 695 "xi-grammar.y"
    { 
		  if ((yyvsp[(7) - (7)].sc) != 0) { 
		    (yyvsp[(7) - (7)].sc)->con1 = new SdagConstruct(SIDENT, (yyvsp[(4) - (7)].strval));
                    (yyvsp[(7) - (7)].sc)->param = new ParamList((yyvsp[(5) - (7)].plist));
                  }
		  (yyval.entry) = new Entry(lineno, (yyvsp[(2) - (7)].intval), (yyvsp[(3) - (7)].type), (yyvsp[(4) - (7)].strval), (yyvsp[(5) - (7)].plist), (yyvsp[(6) - (7)].val), (yyvsp[(7) - (7)].sc), 0, 0); 
		}
    break;

  case 173:

/* Line 1455 of yacc.c  */
#line 703 "xi-grammar.y"
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

  case 174:

/* Line 1455 of yacc.c  */
#line 716 "xi-grammar.y"
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

  case 175:

/* Line 1455 of yacc.c  */
#line 734 "xi-grammar.y"
    { (yyval.accelBlock) = new AccelBlock(lineno, new XStr((yyvsp[(3) - (5)].strval))); }
    break;

  case 176:

/* Line 1455 of yacc.c  */
#line 736 "xi-grammar.y"
    { (yyval.accelBlock) = new AccelBlock(lineno, NULL); }
    break;

  case 177:

/* Line 1455 of yacc.c  */
#line 740 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("void"); }
    break;

  case 178:

/* Line 1455 of yacc.c  */
#line 742 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 179:

/* Line 1455 of yacc.c  */
#line 746 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 180:

/* Line 1455 of yacc.c  */
#line 748 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(2) - (3)].intval); }
    break;

  case 181:

/* Line 1455 of yacc.c  */
#line 752 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (1)].intval); }
    break;

  case 182:

/* Line 1455 of yacc.c  */
#line 754 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (3)].intval) | (yyvsp[(3) - (3)].intval); }
    break;

  case 183:

/* Line 1455 of yacc.c  */
#line 758 "xi-grammar.y"
    { (yyval.intval) = STHREADED; }
    break;

  case 184:

/* Line 1455 of yacc.c  */
#line 760 "xi-grammar.y"
    { (yyval.intval) = SSYNC; }
    break;

  case 185:

/* Line 1455 of yacc.c  */
#line 762 "xi-grammar.y"
    { (yyval.intval) = SIGET; }
    break;

  case 186:

/* Line 1455 of yacc.c  */
#line 764 "xi-grammar.y"
    { (yyval.intval) = SLOCKED; }
    break;

  case 187:

/* Line 1455 of yacc.c  */
#line 766 "xi-grammar.y"
    { (yyval.intval) = SCREATEHERE; }
    break;

  case 188:

/* Line 1455 of yacc.c  */
#line 768 "xi-grammar.y"
    { (yyval.intval) = SCREATEHOME; }
    break;

  case 189:

/* Line 1455 of yacc.c  */
#line 770 "xi-grammar.y"
    { (yyval.intval) = SNOKEEP; }
    break;

  case 190:

/* Line 1455 of yacc.c  */
#line 772 "xi-grammar.y"
    { (yyval.intval) = SNOTRACE; }
    break;

  case 191:

/* Line 1455 of yacc.c  */
#line 774 "xi-grammar.y"
    { (yyval.intval) = SIMMEDIATE; }
    break;

  case 192:

/* Line 1455 of yacc.c  */
#line 776 "xi-grammar.y"
    { (yyval.intval) = SSKIPSCHED; }
    break;

  case 193:

/* Line 1455 of yacc.c  */
#line 778 "xi-grammar.y"
    { (yyval.intval) = SINLINE; }
    break;

  case 194:

/* Line 1455 of yacc.c  */
#line 780 "xi-grammar.y"
    { (yyval.intval) = SLOCAL; }
    break;

  case 195:

/* Line 1455 of yacc.c  */
#line 782 "xi-grammar.y"
    { (yyval.intval) = SPYTHON; }
    break;

  case 196:

/* Line 1455 of yacc.c  */
#line 784 "xi-grammar.y"
    { (yyval.intval) = SMEM; }
    break;

  case 197:

/* Line 1455 of yacc.c  */
#line 786 "xi-grammar.y"
    { (yyval.intval) = SREDUCE; }
    break;

  case 198:

/* Line 1455 of yacc.c  */
#line 790 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 199:

/* Line 1455 of yacc.c  */
#line 792 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 200:

/* Line 1455 of yacc.c  */
#line 794 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 201:

/* Line 1455 of yacc.c  */
#line 798 "xi-grammar.y"
    { (yyval.strval) = ""; }
    break;

  case 202:

/* Line 1455 of yacc.c  */
#line 800 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 203:

/* Line 1455 of yacc.c  */
#line 802 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[(1) - (3)].strval))+strlen((yyvsp[(3) - (3)].strval))+3];
			sprintf(tmp,"%s, %s", (yyvsp[(1) - (3)].strval), (yyvsp[(3) - (3)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 204:

/* Line 1455 of yacc.c  */
#line 810 "xi-grammar.y"
    { (yyval.strval) = ""; }
    break;

  case 205:

/* Line 1455 of yacc.c  */
#line 812 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 206:

/* Line 1455 of yacc.c  */
#line 814 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s[%s]%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 207:

/* Line 1455 of yacc.c  */
#line 820 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s{%s}%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 208:

/* Line 1455 of yacc.c  */
#line 826 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s(%s)%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 209:

/* Line 1455 of yacc.c  */
#line 832 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(2) - (4)].strval))+strlen((yyvsp[(4) - (4)].strval))+3];
			sprintf(tmp,"(%s)%s", (yyvsp[(2) - (4)].strval), (yyvsp[(4) - (4)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 210:

/* Line 1455 of yacc.c  */
#line 840 "xi-grammar.y"
    {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			(yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].type),(yyvsp[(2) - (3)].strval));
		}
    break;

  case 211:

/* Line 1455 of yacc.c  */
#line 847 "xi-grammar.y"
    { 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			(yyval.intval) = 0;
		}
    break;

  case 212:

/* Line 1455 of yacc.c  */
#line 855 "xi-grammar.y"
    { 
			in_braces=0;
			(yyval.intval) = 0;
		}
    break;

  case 213:

/* Line 1455 of yacc.c  */
#line 862 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (1)].type));}
    break;

  case 214:

/* Line 1455 of yacc.c  */
#line 864 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].type),(yyvsp[(2) - (3)].strval)); (yyval.pname)->setConditional((yyvsp[(3) - (3)].intval)); }
    break;

  case 215:

/* Line 1455 of yacc.c  */
#line 866 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (4)].type),(yyvsp[(2) - (4)].strval),0,(yyvsp[(4) - (4)].val));}
    break;

  case 216:

/* Line 1455 of yacc.c  */
#line 868 "xi-grammar.y"
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].pname)->getType(), (yyvsp[(1) - (3)].pname)->getName() ,(yyvsp[(2) - (3)].strval));
		}
    break;

  case 217:

/* Line 1455 of yacc.c  */
#line 874 "xi-grammar.y"
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READONLY; }
    break;

  case 218:

/* Line 1455 of yacc.c  */
#line 875 "xi-grammar.y"
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READWRITE; }
    break;

  case 219:

/* Line 1455 of yacc.c  */
#line 876 "xi-grammar.y"
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_WRITEONLY; }
    break;

  case 220:

/* Line 1455 of yacc.c  */
#line 879 "xi-grammar.y"
    { (yyval.xstrptr) = new XStr((yyvsp[(1) - (1)].strval)); }
    break;

  case 221:

/* Line 1455 of yacc.c  */
#line 880 "xi-grammar.y"
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "->" << (yyvsp[(4) - (4)].strval); }
    break;

  case 222:

/* Line 1455 of yacc.c  */
#line 881 "xi-grammar.y"
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[(1) - (3)].xstrptr)) << "." << (yyvsp[(3) - (3)].strval); }
    break;

  case 223:

/* Line 1455 of yacc.c  */
#line 883 "xi-grammar.y"
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "[" << *((yyvsp[(3) - (4)].xstrptr)) << "]";
                  delete (yyvsp[(1) - (4)].xstrptr);
                  delete (yyvsp[(3) - (4)].xstrptr);
                }
    break;

  case 224:

/* Line 1455 of yacc.c  */
#line 890 "xi-grammar.y"
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "[" << (yyvsp[(3) - (4)].strval) << "]";
                  delete (yyvsp[(1) - (4)].xstrptr);
                }
    break;

  case 225:

/* Line 1455 of yacc.c  */
#line 896 "xi-grammar.y"
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "(" << *((yyvsp[(3) - (4)].xstrptr)) << ")";
                  delete (yyvsp[(1) - (4)].xstrptr);
                  delete (yyvsp[(3) - (4)].xstrptr);
                }
    break;

  case 226:

/* Line 1455 of yacc.c  */
#line 905 "xi-grammar.y"
    {
                  in_bracket = 0;
                  (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].pname)->getType(), (yyvsp[(1) - (3)].pname)->getName(), (yyvsp[(2) - (3)].strval));
                }
    break;

  case 227:

/* Line 1455 of yacc.c  */
#line 912 "xi-grammar.y"
    {
                  (yyval.pname) = new Parameter(lineno, (yyvsp[(3) - (7)].type), (yyvsp[(4) - (7)].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[(6) - (7)].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[(1) - (7)].intval));
                }
    break;

  case 228:

/* Line 1455 of yacc.c  */
#line 918 "xi-grammar.y"
    {
		  (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (5)].type), (yyvsp[(2) - (5)].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[(4) - (5)].xstrptr));
                  (yyval.pname)->setAccelBufferType(Parameter::ACCEL_BUFFER_TYPE_READWRITE);
		}
    break;

  case 229:

/* Line 1455 of yacc.c  */
#line 924 "xi-grammar.y"
    {
                  (yyval.pname) = (yyvsp[(3) - (6)].pname);
                  (yyval.pname)->setAccelInstName((yyvsp[(5) - (6)].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[(1) - (6)].intval));
		}
    break;

  case 230:

/* Line 1455 of yacc.c  */
#line 932 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (1)].pname)); }
    break;

  case 231:

/* Line 1455 of yacc.c  */
#line 934 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (3)].pname),(yyvsp[(3) - (3)].plist)); }
    break;

  case 232:

/* Line 1455 of yacc.c  */
#line 938 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (1)].pname)); }
    break;

  case 233:

/* Line 1455 of yacc.c  */
#line 940 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (3)].pname),(yyvsp[(3) - (3)].plist)); }
    break;

  case 234:

/* Line 1455 of yacc.c  */
#line 944 "xi-grammar.y"
    { (yyval.plist) = (yyvsp[(2) - (3)].plist); }
    break;

  case 235:

/* Line 1455 of yacc.c  */
#line 946 "xi-grammar.y"
    { (yyval.plist) = new ParamList(new Parameter(0, new BuiltinType("void"))); }
    break;

  case 236:

/* Line 1455 of yacc.c  */
#line 950 "xi-grammar.y"
    { (yyval.plist) = (yyvsp[(2) - (3)].plist); }
    break;

  case 237:

/* Line 1455 of yacc.c  */
#line 952 "xi-grammar.y"
    { (yyval.plist) = 0; }
    break;

  case 238:

/* Line 1455 of yacc.c  */
#line 956 "xi-grammar.y"
    { (yyval.val) = 0; }
    break;

  case 239:

/* Line 1455 of yacc.c  */
#line 958 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(3) - (3)].strval)); }
    break;

  case 240:

/* Line 1455 of yacc.c  */
#line 962 "xi-grammar.y"
    { (yyval.sc) = 0; }
    break;

  case 241:

/* Line 1455 of yacc.c  */
#line 964 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSDAGENTRY, (yyvsp[(1) - (1)].sc)); }
    break;

  case 242:

/* Line 1455 of yacc.c  */
#line 966 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSDAGENTRY, (yyvsp[(2) - (3)].sc)); }
    break;

  case 243:

/* Line 1455 of yacc.c  */
#line 970 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSLIST, (yyvsp[(1) - (1)].sc)); }
    break;

  case 244:

/* Line 1455 of yacc.c  */
#line 972 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSLIST, (yyvsp[(1) - (2)].sc), (yyvsp[(2) - (2)].sc));  }
    break;

  case 245:

/* Line 1455 of yacc.c  */
#line 976 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SOLIST, (yyvsp[(1) - (1)].sc)); }
    break;

  case 246:

/* Line 1455 of yacc.c  */
#line 978 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SOLIST, (yyvsp[(1) - (2)].sc), (yyvsp[(2) - (2)].sc)); }
    break;

  case 247:

/* Line 1455 of yacc.c  */
#line 982 "xi-grammar.y"
    { (yyval.sc) = 0; }
    break;

  case 248:

/* Line 1455 of yacc.c  */
#line 984 "xi-grammar.y"
    { (yyval.sc) = (yyvsp[(3) - (4)].sc); }
    break;

  case 249:

/* Line 1455 of yacc.c  */
#line 988 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SPUBLISHES, new SdagConstruct(SIDENT, (yyvsp[(1) - (1)].strval))); }
    break;

  case 250:

/* Line 1455 of yacc.c  */
#line 990 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SPUBLISHES, new SdagConstruct(SIDENT, (yyvsp[(1) - (3)].strval)), (yyvsp[(3) - (3)].sc));  }
    break;

  case 251:

/* Line 1455 of yacc.c  */
#line 994 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 252:

/* Line 1455 of yacc.c  */
#line 996 "xi-grammar.y"
    { (yyval.strval) = 0; }
    break;

  case 253:

/* Line 1455 of yacc.c  */
#line 1000 "xi-grammar.y"
    {
		   (yyval.sc) = buildAtomic((yyvsp[(4) - (6)].strval), (yyvsp[(6) - (6)].sc), (yyvsp[(2) - (6)].strval));
		 }
    break;

  case 254:

/* Line 1455 of yacc.c  */
#line 1004 "xi-grammar.y"
    {  
		   in_braces = 0;
		   if (((yyvsp[(4) - (8)].plist)->isVoid() == 0) && ((yyvsp[(4) - (8)].plist)->isMessage() == 0))
                   {
		      connectEntries->append(new Entry(0, 0, new BuiltinType("void"), (yyvsp[(3) - (8)].strval), 
	 	 			new ParamList(new Parameter(lineno, new PtrType( 
                                        new NamedType("CkMarshallMsg")), "_msg")), 0, 0, 0, 1, (yyvsp[(4) - (8)].plist)));
		   }
		   else  {
		      connectEntries->append(new Entry(0, 0, new BuiltinType("void"), (yyvsp[(3) - (8)].strval), (yyvsp[(4) - (8)].plist), 0, 0, 0, 1, (yyvsp[(4) - (8)].plist)));
                   }
                   (yyval.sc) = new SdagConstruct(SCONNECT, (yyvsp[(3) - (8)].strval), (yyvsp[(7) - (8)].strval), (yyvsp[(4) - (8)].plist));
		}
    break;

  case 255:

/* Line 1455 of yacc.c  */
#line 1018 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SWHEN, 0, 0, 0,0,0, 0,  (yyvsp[(2) - (4)].entrylist)); }
    break;

  case 256:

/* Line 1455 of yacc.c  */
#line 1020 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SWHEN, 0, 0, 0,0,0, (yyvsp[(3) - (3)].sc), (yyvsp[(2) - (3)].entrylist)); }
    break;

  case 257:

/* Line 1455 of yacc.c  */
#line 1022 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SWHEN, 0, 0, 0,0,0, (yyvsp[(4) - (5)].sc), (yyvsp[(2) - (5)].entrylist)); }
    break;

  case 258:

/* Line 1455 of yacc.c  */
#line 1024 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SOVERLAP,0, 0,0,0,0,(yyvsp[(3) - (4)].sc), 0); }
    break;

  case 259:

/* Line 1455 of yacc.c  */
#line 1026 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFOR, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (11)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(5) - (11)].strval)),
		             new SdagConstruct(SINT_EXPR, (yyvsp[(7) - (11)].strval)), 0, (yyvsp[(10) - (11)].sc), 0); }
    break;

  case 260:

/* Line 1455 of yacc.c  */
#line 1029 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFOR, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (9)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(5) - (9)].strval)), 
		         new SdagConstruct(SINT_EXPR, (yyvsp[(7) - (9)].strval)), 0, (yyvsp[(9) - (9)].sc), 0); }
    break;

  case 261:

/* Line 1455 of yacc.c  */
#line 1032 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFORALL, 0, new SdagConstruct(SIDENT, (yyvsp[(3) - (12)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(6) - (12)].strval)), 
		             new SdagConstruct(SINT_EXPR, (yyvsp[(8) - (12)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(10) - (12)].strval)), (yyvsp[(12) - (12)].sc), 0); }
    break;

  case 262:

/* Line 1455 of yacc.c  */
#line 1035 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFORALL, 0, new SdagConstruct(SIDENT, (yyvsp[(3) - (14)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(6) - (14)].strval)), 
		                 new SdagConstruct(SINT_EXPR, (yyvsp[(8) - (14)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(10) - (14)].strval)), (yyvsp[(13) - (14)].sc), 0); }
    break;

  case 263:

/* Line 1455 of yacc.c  */
#line 1038 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (6)].strval)), (yyvsp[(6) - (6)].sc),0,0,(yyvsp[(5) - (6)].sc),0); }
    break;

  case 264:

/* Line 1455 of yacc.c  */
#line 1040 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (8)].strval)), (yyvsp[(8) - (8)].sc),0,0,(yyvsp[(6) - (8)].sc),0); }
    break;

  case 265:

/* Line 1455 of yacc.c  */
#line 1042 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (5)].strval)), 0,0,0,(yyvsp[(5) - (5)].sc),0); }
    break;

  case 266:

/* Line 1455 of yacc.c  */
#line 1044 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SWHILE, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (7)].strval)), 0,0,0,(yyvsp[(6) - (7)].sc),0); }
    break;

  case 267:

/* Line 1455 of yacc.c  */
#line 1046 "xi-grammar.y"
    { (yyval.sc) = (yyvsp[(2) - (3)].sc); }
    break;

  case 268:

/* Line 1455 of yacc.c  */
#line 1048 "xi-grammar.y"
    { (yyval.sc) = buildAtomic((yyvsp[(2) - (3)].strval), NULL, NULL); }
    break;

  case 269:

/* Line 1455 of yacc.c  */
#line 1051 "xi-grammar.y"
    { (yyval.sc) = 0; }
    break;

  case 270:

/* Line 1455 of yacc.c  */
#line 1053 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SELSE, 0,0,0,0,0, (yyvsp[(2) - (2)].sc),0); }
    break;

  case 271:

/* Line 1455 of yacc.c  */
#line 1055 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SELSE, 0,0,0,0,0, (yyvsp[(3) - (4)].sc),0); }
    break;

  case 272:

/* Line 1455 of yacc.c  */
#line 1058 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFORWARD, new SdagConstruct(SIDENT, (yyvsp[(1) - (1)].strval))); }
    break;

  case 273:

/* Line 1455 of yacc.c  */
#line 1060 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFORWARD, new SdagConstruct(SIDENT, (yyvsp[(1) - (3)].strval)), (yyvsp[(3) - (3)].sc));  }
    break;

  case 274:

/* Line 1455 of yacc.c  */
#line 1064 "xi-grammar.y"
    { in_int_expr = 0; (yyval.intval) = 0; }
    break;

  case 275:

/* Line 1455 of yacc.c  */
#line 1068 "xi-grammar.y"
    { in_int_expr = 1; (yyval.intval) = 0; }
    break;

  case 276:

/* Line 1455 of yacc.c  */
#line 1072 "xi-grammar.y"
    { (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[(1) - (2)].strval), (yyvsp[(2) - (2)].plist), 0, 0, 0, 0); }
    break;

  case 277:

/* Line 1455 of yacc.c  */
#line 1074 "xi-grammar.y"
    { (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[(1) - (5)].strval), (yyvsp[(5) - (5)].plist), 0, 0, (yyvsp[(3) - (5)].strval), 0); }
    break;

  case 278:

/* Line 1455 of yacc.c  */
#line 1078 "xi-grammar.y"
    { (yyval.entrylist) = new EntryList((yyvsp[(1) - (1)].entry)); }
    break;

  case 279:

/* Line 1455 of yacc.c  */
#line 1080 "xi-grammar.y"
    { (yyval.entrylist) = new EntryList((yyvsp[(1) - (3)].entry),(yyvsp[(3) - (3)].entrylist)); }
    break;

  case 280:

/* Line 1455 of yacc.c  */
#line 1084 "xi-grammar.y"
    { in_bracket=1; }
    break;

  case 281:

/* Line 1455 of yacc.c  */
#line 1087 "xi-grammar.y"
    { in_bracket=0; }
    break;

  case 282:

/* Line 1455 of yacc.c  */
#line 1091 "xi-grammar.y"
    { if (!macroDefined((yyvsp[(2) - (2)].strval), 1)) in_comment = 1; }
    break;

  case 283:

/* Line 1455 of yacc.c  */
#line 1095 "xi-grammar.y"
    { if (!macroDefined((yyvsp[(2) - (2)].strval), 0)) in_comment = 1; }
    break;



/* Line 1455 of yacc.c  */
#line 4257 "y.tab.c"
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
#line 1098 "xi-grammar.y"

void yyerror(const char *mesg)
{
    std::cerr << cur_file<<":"<<lineno<<": Charmxi syntax error> "
	      << mesg << std::endl;
}

