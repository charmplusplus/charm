/* A Bison parser, made by GNU Bison 2.3.  */

/* Skeleton implementation for Bison's Yacc-like parsers in C

   Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003, 2004, 2005, 2006
   Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301, USA.  */

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
#define YYBISON_VERSION "2.3"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Using locations.  */
#define YYLSP_NEEDED 0



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
     IF = 301,
     ELSE = 302,
     PYTHON = 303,
     LOCAL = 304,
     NAMESPACE = 305,
     USING = 306,
     IDENT = 307,
     NUMBER = 308,
     LITERAL = 309,
     CPROGRAM = 310,
     HASHIF = 311,
     HASHIFDEF = 312,
     INT = 313,
     LONG = 314,
     SHORT = 315,
     CHAR = 316,
     FLOAT = 317,
     DOUBLE = 318,
     UNSIGNED = 319,
     ACCEL = 320,
     READWRITE = 321,
     WRITEONLY = 322,
     ACCELBLOCK = 323,
     MEMCRITICAL = 324,
     REDUCTIONTARGET = 325,
     CASE = 326
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
#define IF 301
#define ELSE 302
#define PYTHON 303
#define LOCAL 304
#define NAMESPACE 305
#define USING 306
#define IDENT 307
#define NUMBER 308
#define LITERAL 309
#define CPROGRAM 310
#define HASHIF 311
#define HASHIFDEF 312
#define INT 313
#define LONG 314
#define SHORT 315
#define CHAR 316
#define FLOAT 317
#define DOUBLE 318
#define UNSIGNED 319
#define ACCEL 320
#define READWRITE 321
#define WRITEONLY 322
#define ACCELBLOCK 323
#define MEMCRITICAL 324
#define REDUCTIONTARGET 325
#define CASE 326




/* Copy the first part of user declarations.  */
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
}


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

#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE
#line 23 "xi-grammar.y"
{
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
}
/* Line 193 of yacc.c.  */
#line 298 "y.tab.c"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 311 "y.tab.c"

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
YYID (int i)
#else
static int
YYID (i)
    int i;
#endif
{
  return i;
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
  yytype_int16 yyss;
  YYSTYPE yyvs;
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
# define YYSTACK_RELOCATE(Stack)					\
    do									\
      {									\
	YYSIZE_T yynewbytes;						\
	YYCOPY (&yyptr->Stack, Stack, yysize);				\
	Stack = &yyptr->Stack;						\
	yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
	yyptr += yynewbytes / sizeof (*yyptr);				\
      }									\
    while (YYID (0))

#endif

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  9
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   852

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  88
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  113
/* YYNRULES -- Number of rules.  */
#define YYNRULES  305
/* YYNRULES -- Number of states.  */
#define YYNSTATES  598

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   326

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    82,     2,
      80,    81,    79,     2,    76,    86,    87,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    73,    72,
      77,    85,    78,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    83,     2,    84,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    74,     2,    75,     2,     2,     2,     2,
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
      65,    66,    67,    68,    69,    70,    71
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
     837,   841,   843,   846,   848,   851,   853,   856,   858,   860,
     861,   866,   870,   876,   878,   880,   882,   884,   886,   888,
     894,   899,   901,   906,   918,   928,   941,   956,   963,   972,
     978,   986,   990,   992,   993,   996,  1001,  1003,  1005,  1008,
    1014,  1016,  1020,  1022,  1024,  1027
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
      89,     0,    -1,    90,    -1,    -1,    95,    90,    -1,    -1,
       5,    -1,    -1,    72,    -1,    52,    -1,    52,    -1,    94,
      73,    73,    52,    -1,     3,    93,    96,    -1,     4,    93,
      96,    -1,    72,    -1,    74,    97,    75,    92,    -1,    -1,
      99,    97,    -1,    51,    50,    94,    -1,    51,    94,    -1,
      91,   154,    -1,    91,   133,    -1,     5,    39,   164,   106,
      93,   103,   181,    -1,    91,    74,    97,    75,    92,    -1,
      50,    93,    74,    97,    75,    -1,    98,    72,    -1,    98,
     161,    -1,    91,    95,    -1,    91,   136,    -1,    91,   137,
      -1,    91,   138,    -1,    91,   140,    -1,    91,   151,    -1,
     199,    -1,   200,    -1,   163,    -1,     1,    -1,   112,    -1,
      53,    -1,    54,    -1,   100,    -1,   100,    76,   101,    -1,
      -1,   101,    -1,    -1,    77,   102,    78,    -1,    58,    -1,
      59,    -1,    60,    -1,    61,    -1,    64,    58,    -1,    64,
      59,    -1,    64,    59,    58,    -1,    64,    59,    59,    -1,
      64,    60,    -1,    64,    61,    -1,    59,    59,    -1,    62,
      -1,    63,    -1,    59,    63,    -1,    35,    -1,    93,   103,
      -1,    94,   103,    -1,   104,    -1,   106,    -1,   107,    79,
      -1,   108,    79,    -1,   109,    79,    -1,   111,    80,    79,
      93,    81,    80,   179,    81,    -1,   107,    -1,   108,    -1,
     109,    -1,   110,    -1,    36,   111,    -1,   111,    36,    -1,
     111,    82,    -1,   111,    -1,    53,    -1,    94,    -1,    83,
     113,    84,    -1,    -1,   114,   115,    -1,     6,   112,    94,
     115,    -1,     6,    16,   107,    79,    93,    -1,    -1,    35,
      -1,    -1,    83,   120,    84,    -1,   121,    -1,   121,    76,
     120,    -1,    37,    -1,    38,    -1,    -1,    83,   123,    84,
      -1,   128,    -1,   128,    76,   123,    -1,    -1,    54,    -1,
      48,    -1,    -1,    83,   127,    84,    -1,   125,    -1,   125,
      76,   127,    -1,    30,    -1,    48,    -1,    -1,    17,    -1,
      -1,    83,    84,    -1,   129,   112,    93,   130,    72,    -1,
     131,    -1,   131,   132,    -1,    16,   119,   105,    -1,    16,
     119,   105,    74,   132,    75,    -1,    -1,    73,   135,    -1,
     106,    -1,   106,    76,   135,    -1,    11,   122,   105,   134,
     152,    -1,    12,   122,   105,   134,   152,    -1,    13,   122,
     105,   134,   152,    -1,    14,   122,   105,   134,   152,    -1,
      83,    53,    93,    84,    -1,    83,    93,    84,    -1,    15,
     126,   139,   105,   134,   152,    -1,    15,   139,   126,   105,
     134,   152,    -1,    11,   122,    93,   134,   152,    -1,    12,
     122,    93,   134,   152,    -1,    13,   122,    93,   134,   152,
      -1,    14,   122,    93,   134,   152,    -1,    15,   139,    93,
     134,   152,    -1,    16,   119,    93,    72,    -1,    16,   119,
      93,    74,   132,    75,    72,    -1,    -1,    85,   112,    -1,
      -1,    85,    53,    -1,    85,    54,    -1,    18,    93,   146,
      -1,   110,   147,    -1,   112,    93,   147,    -1,   148,    -1,
     148,    76,   149,    -1,    22,    77,   149,    78,    -1,   150,
     141,    -1,   150,   142,    -1,   150,   143,    -1,   150,   144,
      -1,   150,   145,    -1,    72,    -1,    74,   153,    75,    92,
      -1,    -1,   159,   153,    -1,   116,    -1,   117,    -1,   156,
      -1,   155,    -1,    10,   157,    -1,    19,   158,    -1,    18,
      93,    -1,     8,   118,    94,    -1,     8,   118,    94,    80,
     118,    81,    -1,     8,   118,    94,    77,   101,    78,    80,
     118,    81,    -1,     7,   118,    94,    -1,     7,   118,    94,
      80,   118,    81,    -1,     9,   118,    94,    -1,     9,   118,
      94,    80,   118,    81,    -1,     9,   118,    94,    77,   101,
      78,    80,   118,    81,    -1,     9,    83,    65,    84,   118,
      94,    80,   118,    81,    -1,   106,    -1,   106,    76,   157,
      -1,    54,    -1,   160,    72,    -1,   160,   161,    -1,   162,
      -1,   150,   162,    -1,   154,    -1,    39,    -1,    75,    -1,
       7,    -1,     8,    -1,     9,    -1,    11,    -1,    12,    -1,
      15,    -1,    13,    -1,    14,    -1,     6,    -1,    39,   165,
     164,    93,   181,   183,   184,    -1,    39,   165,    93,   181,
     184,    -1,    39,    83,    65,    84,    35,    93,   181,   182,
     172,   170,   173,    93,    -1,    68,   172,   170,   173,    72,
      -1,    68,    72,    -1,    35,    -1,   108,    -1,    -1,    83,
     166,    84,    -1,     1,    -1,   167,    -1,   167,    76,   166,
      -1,    21,    -1,    23,    -1,    24,    -1,    25,    -1,    31,
      -1,    32,    -1,    33,    -1,    34,    -1,    26,    -1,    27,
      -1,    28,    -1,    49,    -1,    48,   124,    -1,    69,    -1,
      70,    -1,     1,    -1,    54,    -1,    53,    -1,    94,    -1,
      -1,    55,    -1,    55,    76,   169,    -1,    -1,    55,    -1,
      55,    83,   170,    84,   170,    -1,    55,    74,   170,    75,
     170,    -1,    55,    80,   169,    81,   170,    -1,    80,   170,
      81,   170,    -1,   112,    93,    83,    -1,    74,    -1,    75,
      -1,   112,    -1,   112,    93,   129,    -1,   112,    93,    85,
     168,    -1,   171,   170,    84,    -1,     6,    -1,    66,    -1,
      67,    -1,    93,    -1,   176,    86,    78,    93,    -1,   176,
      87,    93,    -1,   176,    83,   176,    84,    -1,   176,    83,
      53,    84,    -1,   176,    80,   176,    81,    -1,   171,   170,
      84,    -1,   175,    73,   112,    93,    77,   176,    78,    -1,
     112,    93,    77,   176,    78,    -1,   175,    73,   177,    77,
     176,    78,    -1,   174,    -1,   174,    76,   179,    -1,   178,
      -1,   178,    76,   180,    -1,    80,   179,    81,    -1,    80,
      81,    -1,    83,   180,    84,    -1,    83,    84,    -1,    -1,
      20,    85,    53,    -1,    -1,   191,    -1,    74,   185,    75,
      -1,   191,    -1,   191,   185,    -1,   191,    -1,   191,   185,
      -1,   189,    -1,   189,   187,    -1,   190,    -1,    54,    -1,
      -1,    43,   196,    74,    75,    -1,    43,   196,   191,    -1,
      43,   196,    74,   185,    75,    -1,    45,    -1,    44,    -1,
      40,    -1,    41,    -1,    46,    -1,    42,    -1,    45,   188,
     172,   170,   173,    -1,    44,    74,   186,    75,    -1,   189,
      -1,    71,    74,   187,    75,    -1,    40,   194,   170,    72,
     170,    72,   170,   193,    74,   185,    75,    -1,    40,   194,
     170,    72,   170,    72,   170,   193,   191,    -1,    41,    83,
      52,    84,   194,   170,    73,   170,    76,   170,   193,   191,
      -1,    41,    83,    52,    84,   194,   170,    73,   170,    76,
     170,   193,    74,   185,    75,    -1,    46,   194,   170,   193,
     191,   192,    -1,    46,   194,   170,   193,    74,   185,    75,
     192,    -1,    42,   194,   170,   193,   191,    -1,    42,   194,
     170,   193,    74,   185,    75,    -1,   172,   170,   173,    -1,
       1,    -1,    -1,    47,   191,    -1,    47,    74,   185,    75,
      -1,    81,    -1,    80,    -1,    52,   181,    -1,    52,   197,
     170,   198,   181,    -1,   195,    -1,   195,    76,   196,    -1,
      83,    -1,    84,    -1,    56,    93,    -1,    57,    93,    -1
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
     704,   706,   708,   710,   712,   714,   716,   718,   721,   731,
     746,   762,   764,   768,   770,   775,   776,   778,   782,   784,
     788,   790,   792,   794,   796,   798,   800,   802,   804,   806,
     808,   810,   812,   814,   816,   818,   822,   824,   826,   831,
     832,   834,   843,   844,   846,   852,   858,   864,   872,   879,
     887,   894,   896,   898,   900,   907,   908,   909,   912,   913,
     914,   915,   922,   928,   937,   944,   950,   956,   964,   966,
     970,   972,   976,   978,   982,   984,   989,   990,   995,   996,
     998,  1002,  1004,  1008,  1010,  1014,  1016,  1018,  1022,  1025,
    1028,  1030,  1032,  1036,  1038,  1040,  1042,  1044,  1046,  1050,
    1052,  1054,  1056,  1058,  1061,  1064,  1067,  1070,  1072,  1074,
    1076,  1078,  1080,  1087,  1088,  1090,  1094,  1098,  1102,  1104,
    1108,  1110,  1114,  1117,  1121,  1125
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
     325,   326,    59,    58,   123,   125,    44,    60,    62,    42,
      40,    41,    38,    91,    93,    61,    45,    46
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    88,    89,    90,    90,    91,    91,    92,    92,    93,
      94,    94,    95,    95,    96,    96,    97,    97,    98,    98,
      98,    98,    98,    99,    99,    99,    99,    99,    99,    99,
      99,    99,    99,    99,    99,    99,    99,   100,   100,   100,
     101,   101,   102,   102,   103,   103,   104,   104,   104,   104,
     104,   104,   104,   104,   104,   104,   104,   104,   104,   104,
     104,   105,   106,   107,   107,   108,   109,   109,   110,   111,
     111,   111,   111,   111,   111,   112,   112,   113,   113,   114,
     115,   115,   116,   117,   118,   118,   119,   119,   120,   120,
     121,   121,   122,   122,   123,   123,   124,   124,   125,   126,
     126,   127,   127,   128,   128,   129,   129,   130,   130,   131,
     132,   132,   133,   133,   134,   134,   135,   135,   136,   136,
     137,   138,   139,   139,   140,   140,   141,   141,   142,   143,
     144,   145,   145,   146,   146,   147,   147,   147,   148,   148,
     148,   149,   149,   150,   151,   151,   151,   151,   151,   152,
     152,   153,   153,   154,   154,   154,   154,   154,   154,   154,
     155,   155,   155,   155,   155,   156,   156,   156,   156,   157,
     157,   158,   159,   159,   160,   160,   160,   161,   161,   161,
     161,   161,   161,   161,   161,   161,   161,   161,   162,   162,
     162,   163,   163,   164,   164,   165,   165,   165,   166,   166,
     167,   167,   167,   167,   167,   167,   167,   167,   167,   167,
     167,   167,   167,   167,   167,   167,   168,   168,   168,   169,
     169,   169,   170,   170,   170,   170,   170,   170,   171,   172,
     173,   174,   174,   174,   174,   175,   175,   175,   176,   176,
     176,   176,   176,   176,   177,   178,   178,   178,   179,   179,
     180,   180,   181,   181,   182,   182,   183,   183,   184,   184,
     184,   185,   185,   186,   186,   187,   187,   187,   188,   188,
     189,   189,   189,   190,   190,   190,   190,   190,   190,   191,
     191,   191,   191,   191,   191,   191,   191,   191,   191,   191,
     191,   191,   191,   192,   192,   192,   193,   194,   195,   195,
     196,   196,   197,   198,   199,   200
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
       3,     1,     2,     1,     2,     1,     2,     1,     1,     0,
       4,     3,     5,     1,     1,     1,     1,     1,     1,     5,
       4,     1,     4,    11,     9,    12,    14,     6,     8,     5,
       7,     3,     1,     0,     2,     4,     1,     1,     2,     5,
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
       0,     0,    10,    19,   304,   305,   192,   229,   222,     0,
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
     216,   218,   233,     0,   199,   292,     0,     0,     0,     0,
       0,   269,     0,     0,     0,   222,   189,   281,   259,   256,
       0,   297,   222,     0,   222,     0,   300,     0,     0,   268,
       0,   222,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   302,   298,   222,     0,     0,   271,     0,     0,
     222,     0,   275,   276,   278,   274,   273,   277,     0,   265,
     267,   260,   262,   291,     0,   188,     0,     0,   222,     0,
     296,     0,     0,   301,   270,     0,   280,   264,     0,     0,
     282,   266,   257,   235,   236,   237,   255,     0,     0,   250,
       0,   222,     0,   222,     0,   289,   303,     0,   272,   279,
       0,   293,     0,     0,     0,   254,     0,   222,     0,     0,
     299,     0,     0,   287,     0,     0,   222,     0,   251,     0,
       0,   222,   290,   293,     0,   294,   238,     0,     0,     0,
       0,   190,     0,     0,   288,     0,   246,     0,     0,     0,
       0,     0,   244,     0,     0,   284,   222,   295,     0,     0,
       0,     0,   240,     0,   247,     0,     0,   243,   242,   241,
     239,   245,   283,     0,     0,   285,     0,   286
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
     432,   225,   101,   359,   445,   163,   360,   518,   557,   547,
     519,   361,   520,   324,   497,   467,   446,   463,   478,   488,
     460,   447,   490,   464,   543,   501,   452,   456,   457,   474,
     527,    27,    28
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -468
static const yytype_int16 yypact[] =
{
      84,    -9,    -9,    95,  -468,    84,  -468,    68,    68,  -468,
    -468,  -468,   325,  -468,  -468,  -468,    88,    -9,   121,    -9,
      -9,   170,   636,    78,   654,   325,  -468,  -468,  -468,   328,
     -20,   116,  -468,    -6,  -468,  -468,  -468,  -468,   -34,   702,
     148,   148,   -12,   116,   115,   115,   115,   115,   128,   144,
      -9,   183,   125,   325,  -468,  -468,  -468,  -468,  -468,  -468,
    -468,  -468,   549,  -468,  -468,  -468,  -468,   206,  -468,  -468,
    -468,  -468,  -468,  -468,  -468,  -468,  -468,  -468,  -468,  -468,
    -468,  -468,   210,  -468,    67,  -468,  -468,  -468,  -468,   158,
     119,  -468,  -468,   217,  -468,   116,   325,    -6,   212,    77,
     -34,   244,   568,  -468,   239,   217,   225,   248,  -468,    -3,
     116,  -468,   116,   116,   227,   116,   247,  -468,    -4,    -9,
      -9,    -9,    -9,    76,   271,   288,   156,    -9,  -468,  -468,
    -468,   732,   289,   115,   115,   115,   115,   271,   144,  -468,
    -468,  -468,  -468,  -468,  -468,  -468,  -468,  -468,  -468,   189,
    -468,  -468,   775,  -468,  -468,    -9,   299,   331,   -34,   322,
     -34,   304,  -468,   323,   315,   -23,  -468,  -468,  -468,   317,
    -468,    -7,    79,    30,   313,    99,   116,  -468,  -468,   314,
     327,   329,   334,   334,   334,   334,  -468,    -9,   320,   332,
     333,   199,    -9,   368,    -9,  -468,  -468,   342,   352,   355,
      -9,   137,    -9,   359,   353,   206,    -9,    -9,    -9,    -9,
      -9,    -9,  -468,  -468,  -468,  -468,   362,  -468,   361,  -468,
     329,  -468,  -468,   371,   372,   366,   367,   -34,  -468,    -9,
      -9,   214,   373,  -468,   148,   775,   148,   148,   775,   148,
    -468,  -468,    -4,  -468,   116,   250,   250,   250,   250,   370,
    -468,   368,  -468,   334,   334,  -468,   156,   440,   376,   306,
    -468,   377,   732,  -468,  -468,   334,   334,   334,   334,   334,
     281,   775,  -468,   383,   -34,   322,   -34,   -34,  -468,  -468,
     394,  -468,    -6,   374,  -468,   395,   386,   396,   116,   400,
     401,  -468,   407,  -468,  -468,   737,  -468,  -468,  -468,  -468,
    -468,  -468,   250,   250,  -468,  -468,   239,    -5,   416,   239,
    -468,  -468,  -468,  -468,  -468,   250,   250,   250,   250,   250,
    -468,   440,  -468,   745,  -468,  -468,  -468,  -468,  -468,   412,
    -468,  -468,   413,  -468,   109,   414,  -468,   116,    54,   456,
     431,  -468,   737,   700,  -468,  -468,  -468,    -9,  -468,  -468,
    -468,  -468,  -468,  -468,  -468,  -468,   432,  -468,    -9,   -34,
     441,   438,   239,   148,   148,   148,  -468,  -468,   652,   788,
    -468,   206,  -468,  -468,  -468,   445,   455,    15,   451,   239,
    -468,   448,   457,   460,   465,  -468,  -468,  -468,  -468,  -468,
    -468,  -468,  -468,  -468,  -468,  -468,  -468,   482,  -468,   453,
    -468,  -468,   463,   464,   469,   383,    -9,  -468,   466,   481,
    -468,  -468,   229,  -468,  -468,  -468,  -468,  -468,  -468,  -468,
    -468,  -468,   520,  -468,   671,   459,   383,  -468,  -468,  -468,
    -468,    -6,  -468,    -9,  -468,  -468,   476,   474,   476,   506,
     504,   525,   476,   507,   190,   -34,  -468,  -468,  -468,   562,
     383,  -468,   -34,   546,   -34,    28,   523,   444,   468,  -468,
     509,   -34,   690,   526,   369,   244,   515,   459,   521,   534,
     530,   535,  -468,  -468,   -34,   506,   213,  -468,   542,   378,
     -34,   535,  -468,  -468,  -468,  -468,  -468,  -468,   548,   690,
    -468,  -468,  -468,  -468,   566,  -468,   103,   509,   -34,   476,
    -468,   480,   537,  -468,  -468,   558,  -468,  -468,   244,   531,
    -468,  -468,  -468,  -468,  -468,  -468,  -468,    -9,   561,   559,
     540,   -34,   564,   -34,   190,  -468,  -468,   383,  -468,  -468,
     190,   590,   579,   239,   254,  -468,   244,   -34,   584,   589,
    -468,   595,   544,  -468,    -9,    -9,   -34,   582,  -468,    -9,
     535,   -34,  -468,   590,   190,  -468,  -468,   117,   -41,   587,
      -9,  -468,   551,   598,  -468,   606,  -468,    -9,   309,   604,
      -9,    -9,  -468,   163,   190,  -468,   -34,  -468,   286,   603,
     350,    -9,  -468,   193,  -468,   613,   535,  -468,  -468,  -468,
    -468,  -468,  -468,   567,   190,  -468,   614,  -468
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -468,  -468,   685,  -468,  -196,    -1,   -11,   669,   708,     0,
    -468,  -468,  -468,  -198,  -468,  -164,  -468,   -36,   -32,   -24,
     -21,  -468,  -121,   619,   -37,  -468,  -468,   492,  -468,  -468,
     -13,   610,   471,  -468,    14,   483,  -468,  -468,   617,   477,
    -468,   375,  -468,  -468,  -249,  -468,  -133,   420,  -468,  -468,
    -468,   -90,  -468,  -468,  -468,  -468,  -468,  -468,  -468,   488,
    -468,   489,   731,  -468,    -8,   427,   736,  -468,  -468,   594,
    -468,  -468,  -468,   428,   434,  -468,   405,  -468,   354,  -468,
    -468,   502,   -96,   246,   -18,  -443,  -468,  -468,  -383,  -468,
    -468,  -282,   249,  -381,  -468,  -468,   318,  -449,  -468,   293,
    -468,  -421,  -468,  -419,   233,  -467,  -407,  -468,   312,  -468,
    -468,  -468,  -468
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -264
static const yytype_int16 yytable[] =
{
       7,     8,   110,    38,   161,    93,   448,    33,    94,   264,
     201,   116,   305,   168,   509,   492,    30,   243,    34,    35,
      97,    99,   493,   111,   425,    81,   177,   505,   113,   115,
     507,   454,   305,   168,   192,   461,   571,   286,   477,   479,
     289,   489,   411,     6,   178,   449,   100,   210,   448,   128,
     246,   247,   248,   132,    96,   367,   273,   169,   348,   120,
     121,   122,   223,   155,   226,   529,    98,    98,   489,   468,
    -110,   114,   356,   322,   473,   539,   231,   169,   164,   170,
     381,   541,   525,   562,   183,   184,   185,     1,     2,  -195,
     531,   199,   523,   549,   202,     9,   156,   415,   411,   171,
     412,   172,   173,    98,   175,   565,  -195,   235,   323,   513,
     236,   472,  -195,  -195,  -195,  -195,  -195,  -195,  -195,   593,
     302,   303,   188,   555,   186,   585,   146,    29,     6,   187,
     147,   278,   315,   316,   317,   318,   319,   368,   103,   104,
      11,   201,    12,   575,   116,   596,   540,   206,   207,   208,
     209,   158,    98,    67,   220,    32,   253,   159,   254,   234,
     160,    83,    84,    85,    86,    87,    88,    89,    32,   514,
     515,    31,    98,    32,   595,   407,   238,   573,   325,   239,
     327,   328,    98,   111,   578,   580,   249,   516,   583,   364,
     188,   435,    98,   195,   196,   566,   152,   567,   118,   258,
     568,   261,   131,   569,   570,   265,   266,   267,   268,   269,
     270,   123,   292,  -135,   435,  -135,   148,   149,   150,   151,
     282,   285,   259,   287,   288,   202,   290,   126,   279,   280,
     436,   437,   438,   439,   440,   441,   442,   129,   297,   298,
     299,   584,    36,   567,    37,  -229,   568,   212,   213,   569,
     570,     6,   187,   436,   437,   438,   439,   440,   441,   442,
     513,   443,  -193,   378,    37,  -229,    32,   281,  -229,   347,
    -229,   591,   350,   567,   103,   104,   568,   334,   144,   569,
     570,    32,   429,   430,   443,   157,   358,    37,   504,   103,
     104,    32,   174,  -229,   345,   346,   154,    83,    84,    85,
      86,    87,    88,    89,   166,   292,    32,   351,   352,   353,
     354,   355,    83,    84,    85,    86,    87,    88,    89,   162,
     514,   515,   294,   176,   295,   358,    15,   167,    -5,    -5,
      16,    -5,    -5,    -5,    -5,    -5,    -5,    -5,    -5,    -5,
      -5,    -5,   358,    -5,    -5,    93,   375,    -5,    94,   465,
     382,   383,   384,   320,   191,   321,   469,   377,   471,   311,
     312,     6,   579,    82,   205,   481,   567,   587,   405,   568,
     435,   193,   569,   570,   221,    17,    18,   224,   502,   435,
      32,    19,    20,   222,   508,   227,    83,    84,    85,    86,
      87,    88,    89,    21,   229,   228,   230,   237,   241,    -5,
     -16,   431,   522,   242,   250,   426,   152,   244,   251,   436,
     437,   438,   439,   440,   441,   442,   186,   252,   436,   437,
     438,   439,   440,   441,   442,   536,   255,   538,   256,   257,
     567,   263,   450,   568,   589,   262,   569,   570,   271,   272,
     443,   550,   480,    37,  -261,   435,   274,   276,   275,   443,
     559,   277,    37,  -263,   300,   563,   231,   305,   330,   517,
     435,   309,   259,   323,   332,  -258,  -258,  -258,  -258,   435,
    -258,  -258,  -258,  -258,  -258,   329,   331,   333,   335,   521,
     586,   435,   336,   337,   436,   437,   438,   439,   440,   441,
     442,   349,   362,   363,   365,   338,   545,   517,  -258,   436,
     437,   438,   439,   440,   441,   442,   371,   376,   436,   437,
     438,   439,   440,   441,   442,   443,   532,   379,   476,   380,
     436,   437,   438,   439,   440,   441,   442,   410,   408,   416,
     443,  -258,   435,   444,  -258,   414,   420,   422,   417,   443,
     424,   418,    37,   556,   558,   435,   419,   423,   561,    -9,
     427,   443,   435,   428,   524,   433,   451,   453,   455,   556,
     133,   134,   135,   136,   137,   138,   556,   556,   435,   582,
     556,   436,   437,   438,   439,   440,   441,   442,   458,   459,
     590,   462,   466,    37,   436,   437,   438,   439,   440,   441,
     442,   436,   437,   438,   439,   440,   441,   442,   470,   475,
     494,   491,   443,   103,   496,   530,   498,   436,   437,   438,
     439,   440,   441,   442,   499,   443,   500,   506,   554,   512,
      32,   526,   443,   510,   535,   574,    83,    84,    85,    86,
      87,    88,    89,   528,   533,   534,   537,   542,   443,     1,
       2,   594,    39,    40,    41,    42,    43,    44,    45,    46,
      47,    48,    49,   385,    50,    51,   544,   551,    52,   560,
      68,    69,    70,    71,   552,    72,    73,    74,    75,    76,
     553,   572,   385,   386,   576,   387,   388,   389,   390,   391,
     392,   577,   581,   393,   394,   395,   396,   588,   592,   597,
      10,    54,   386,    77,   387,   388,   389,   390,   391,   392,
     397,   398,   393,   394,   395,   396,    68,    69,    70,    71,
      53,    72,    73,    74,    75,    76,    14,   399,   102,   397,
     398,   400,   401,   165,   284,   291,    78,   304,   301,    79,
     482,   483,   484,   439,   485,   486,   487,   103,   104,    77,
     400,   401,   194,    39,    40,    41,    42,    43,   211,   313,
     200,   314,   413,    62,    32,    50,    51,   366,    64,    52,
      83,    84,    85,    86,    87,    88,    89,   103,   104,   372,
     240,   374,   373,   370,   406,    79,   338,   326,   434,   546,
     103,   104,   511,   548,    32,   495,   564,   503,     0,     0,
      83,    84,    85,    86,    87,    88,    89,    32,     0,     0,
       0,     0,     0,    83,    84,    85,    86,    87,    88,    89,
     103,   104,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    82,     0,     0,   357,    32,   214,   215,
       0,     0,     0,    83,    84,    85,    86,    87,    88,    89,
     404,     0,     0,     0,     0,     0,    83,    84,    85,    86,
      87,    88,    89
};

static const yytype_int16 yycheck[] =
{
       1,     2,    39,    21,   100,    29,   425,    18,    29,   205,
     131,    43,    17,    36,   481,   464,    17,   181,    19,    20,
      31,    55,   465,    35,   405,    25,    30,   476,    41,    42,
     479,   438,    17,    36,   124,   442,    77,   235,   457,   458,
     238,   462,    83,    52,    48,   426,    80,   137,   467,    50,
     183,   184,   185,    53,    74,     1,   220,    80,   307,    45,
      46,    47,   158,    95,   160,   508,    73,    73,   489,   450,
      75,    83,   321,   271,   455,   524,    83,    80,   102,    82,
     362,   530,   501,   550,   120,   121,   122,     3,     4,    35,
     509,   127,   499,   536,   131,     0,    96,   379,    83,   110,
      85,   112,   113,    73,   115,   554,    52,    77,    80,     6,
      80,    83,    58,    59,    60,    61,    62,    63,    64,   586,
     253,   254,   123,   542,    48,   574,    59,    39,    52,    53,
      63,   227,   265,   266,   267,   268,   269,    83,    35,    36,
      72,   262,    74,   562,   176,   594,   527,   133,   134,   135,
     136,    74,    73,    75,   155,    52,   192,    80,   194,    80,
      83,    58,    59,    60,    61,    62,    63,    64,    52,    66,
      67,    50,    73,    52,   593,   371,    77,   560,   274,    80,
     276,   277,    73,    35,   567,   568,   187,    84,   571,    80,
     191,     1,    73,    37,    38,    78,    77,    80,    83,   200,
      83,   202,    77,    86,    87,   206,   207,   208,   209,   210,
     211,    83,   244,    76,     1,    78,    58,    59,    60,    61,
     231,   234,    85,   236,   237,   262,   239,    83,   229,   230,
      40,    41,    42,    43,    44,    45,    46,    54,   246,   247,
     248,    78,    72,    80,    74,    55,    83,    58,    59,    86,
      87,    52,    53,    40,    41,    42,    43,    44,    45,    46,
       6,    71,    52,   359,    74,    75,    52,    53,    55,   306,
      80,    78,   309,    80,    35,    36,    83,   288,    72,    86,
      87,    52,    53,    54,    71,    73,   323,    74,    75,    35,
      36,    52,    65,    80,   302,   303,    79,    58,    59,    60,
      61,    62,    63,    64,    79,   337,    52,   315,   316,   317,
     318,   319,    58,    59,    60,    61,    62,    63,    64,    75,
      66,    67,    72,    76,    74,   362,     1,    79,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,   379,    18,    19,   369,   347,    22,   369,   445,
     363,   364,   365,    72,    83,    74,   452,   358,   454,    53,
      54,    52,    53,    35,    75,   461,    80,    81,   369,    83,
       1,    83,    86,    87,    75,    50,    51,    55,   474,     1,
      52,    56,    57,    52,   480,    81,    58,    59,    60,    61,
      62,    63,    64,    68,    79,    72,    79,    84,    84,    74,
      75,   412,   498,    76,    84,   406,    77,    73,    76,    40,
      41,    42,    43,    44,    45,    46,    48,    84,    40,    41,
      42,    43,    44,    45,    46,   521,    84,   523,    76,    74,
      80,    78,   433,    83,    84,    76,    86,    87,    76,    78,
      71,   537,   460,    74,    75,     1,    75,    81,    76,    71,
     546,    84,    74,    75,    84,   551,    83,    17,    84,   496,
       1,    85,    85,    80,    78,     6,     7,     8,     9,     1,
      11,    12,    13,    14,    15,    81,    81,    81,    78,   497,
     576,     1,    81,    76,    40,    41,    42,    43,    44,    45,
      46,    75,    80,    80,    80,    39,   533,   534,    39,    40,
      41,    42,    43,    44,    45,    46,    75,    75,    40,    41,
      42,    43,    44,    45,    46,    71,   517,    76,    74,    81,
      40,    41,    42,    43,    44,    45,    46,    72,    83,    81,
      71,    72,     1,    74,    75,    84,    54,    84,    81,    71,
      76,    81,    74,   544,   545,     1,    81,    84,   549,    80,
      84,    71,     1,    72,    74,    35,    80,    83,    52,   560,
      11,    12,    13,    14,    15,    16,   567,   568,     1,   570,
     571,    40,    41,    42,    43,    44,    45,    46,    74,    54,
     581,    74,    20,    74,    40,    41,    42,    43,    44,    45,
      46,    40,    41,    42,    43,    44,    45,    46,    52,    76,
      85,    75,    71,    35,    83,    74,    72,    40,    41,    42,
      43,    44,    45,    46,    84,    71,    81,    75,    74,    53,
      52,    84,    71,    75,    84,    74,    58,    59,    60,    61,
      62,    63,    64,    75,    73,    76,    72,    47,    71,     3,
       4,    74,     6,     7,     8,     9,    10,    11,    12,    13,
      14,    15,    16,     1,    18,    19,    77,    73,    22,    77,
       6,     7,     8,     9,    75,    11,    12,    13,    14,    15,
      75,    84,     1,    21,    76,    23,    24,    25,    26,    27,
      28,    75,    78,    31,    32,    33,    34,    84,    75,    75,
       5,    22,    21,    39,    23,    24,    25,    26,    27,    28,
      48,    49,    31,    32,    33,    34,     6,     7,     8,     9,
      74,    11,    12,    13,    14,    15,     8,    65,    16,    48,
      49,    69,    70,   104,   232,   242,    72,   256,   251,    75,
      40,    41,    42,    43,    44,    45,    46,    35,    36,    39,
      69,    70,   125,     6,     7,     8,     9,    10,   138,   261,
      18,   262,   377,    22,    52,    18,    19,   337,    22,    22,
      58,    59,    60,    61,    62,    63,    64,    35,    36,   342,
     176,   343,    72,   339,   369,    75,    39,   275,   424,   533,
      35,    36,   489,   534,    52,   467,   553,   475,    -1,    -1,
      58,    59,    60,    61,    62,    63,    64,    52,    -1,    -1,
      -1,    -1,    -1,    58,    59,    60,    61,    62,    63,    64,
      35,    36,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    35,    -1,    -1,    81,    52,    53,    54,
      -1,    -1,    -1,    58,    59,    60,    61,    62,    63,    64,
      52,    -1,    -1,    -1,    -1,    -1,    58,    59,    60,    61,
      62,    63,    64
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     3,     4,    89,    90,    95,    52,    93,    93,     0,
      90,    72,    74,    96,    96,     1,     5,    50,    51,    56,
      57,    68,    91,    97,    98,    99,   163,   199,   200,    39,
      93,    50,    52,    94,    93,    93,    72,    74,   172,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      18,    19,    22,    74,    95,   116,   117,   133,   136,   137,
     138,   140,   150,   151,   154,   155,   156,    75,     6,     7,
       8,     9,    11,    12,    13,    14,    15,    39,    72,    75,
     161,    97,    35,    58,    59,    60,    61,    62,    63,    64,
      94,   104,   106,   107,   108,   164,    74,    94,    73,    55,
      80,   170,    16,    35,    36,   107,   108,   109,   110,   111,
     112,    35,   118,   118,    83,   118,   106,   157,    83,   122,
     122,   122,   122,    83,   126,   139,    83,   119,    93,    54,
     158,    77,    97,    11,    12,    13,    14,    15,    16,   141,
     142,   143,   144,   145,    72,    92,    59,    63,    58,    59,
      60,    61,    77,   103,    79,   106,    97,    73,    74,    80,
      83,   170,    75,   173,   107,   111,    79,    79,    36,    80,
      82,    94,    94,    94,    65,    94,    76,    30,    48,   123,
     128,    93,   105,   105,   105,   105,    48,    53,    93,   125,
     127,    83,   139,    83,   126,    37,    38,   120,   121,   105,
      18,   110,   112,   148,   149,    75,   122,   122,   122,   122,
     139,   119,    58,    59,    53,    54,   100,   101,   102,   112,
      93,    75,    52,   170,    55,   169,   170,    81,    72,    79,
      79,    83,   114,   115,    80,    77,    80,    84,    77,    80,
     157,    84,    76,   103,    73,   134,   134,   134,   134,    93,
      84,    76,    84,   105,   105,    84,    76,    74,    93,    85,
     147,    93,    76,    78,    92,    93,    93,    93,    93,    93,
      93,    76,    78,   103,    75,    76,    81,    84,   170,    93,
      93,    53,    94,   113,   115,   118,   101,   118,   118,   101,
     118,   123,   106,   135,    72,    74,   152,   152,   152,   152,
      84,   127,   134,   134,   120,    17,   129,   131,   132,    85,
     146,    53,    54,   147,   149,   134,   134,   134,   134,   134,
      72,    74,   101,    80,   181,   170,   169,   170,   170,    81,
      84,    81,    78,    81,    94,    78,    81,    76,    39,   150,
     153,   154,   159,   160,   162,   152,   152,   112,   132,    75,
     112,   152,   152,   152,   152,   152,   132,    81,   112,   171,
     174,   179,    80,    80,    80,    80,   135,     1,    83,   165,
     162,    75,   153,    72,   161,    93,    75,    93,   170,    76,
      81,   179,   118,   118,   118,     1,    21,    23,    24,    25,
      26,    27,    28,    31,    32,    33,    34,    48,    49,    65,
      69,    70,   166,   167,    52,    93,   164,    92,    83,   130,
      72,    83,    85,   129,    84,   179,    81,    81,    81,    81,
      54,   124,    84,    84,    76,   181,    93,    84,    72,    53,
      54,    94,   168,    35,   166,     1,    40,    41,    42,    43,
      44,    45,    46,    71,    74,   172,   184,   189,   191,   181,
      93,    80,   194,    83,   194,    52,   195,   196,    74,    54,
     188,   194,    74,   185,   191,   170,    20,   183,   181,   170,
      52,   170,    83,   181,   197,    76,    74,   191,   186,   191,
     172,   170,    40,    41,    42,    44,    45,    46,   187,   189,
     190,    75,   185,   173,    85,   184,    83,   182,    72,    84,
      81,   193,   170,   196,    75,   185,    75,   185,   170,   193,
      75,   187,    53,     6,    66,    67,    84,   112,   175,   178,
     180,   172,   170,   194,    74,   191,    84,   198,    75,   173,
      74,   191,    93,    73,    76,    84,   170,    72,   170,   185,
     181,   185,    47,   192,    77,   112,   171,   177,   180,   173,
     170,    73,    75,    75,    74,   191,    93,   176,    93,   170,
      77,    93,   193,   170,   192,   185,    78,    80,    83,    86,
      87,    77,    84,   176,    74,   191,    76,    75,   176,    53,
     176,    78,    93,   176,    78,   185,   170,    81,    84,    84,
      93,    78,    75,   193,    74,   191,   185,    75
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
yy_stack_print (yytype_int16 *bottom, yytype_int16 *top)
#else
static void
yy_stack_print (bottom, top)
    yytype_int16 *bottom;
    yytype_int16 *top;
#endif
{
  YYFPRINTF (stderr, "Stack now");
  for (; bottom <= top; ++bottom)
    YYFPRINTF (stderr, " %d", *bottom);
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
      fprintf (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr, yyrhs[yyprhs[yyrule] + yyi],
		       &(yyvsp[(yyi + 1) - (yynrhs)])
		       		       );
      fprintf (stderr, "\n");
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



/* The look-ahead symbol.  */
int yychar;

/* The semantic value of the look-ahead symbol.  */
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
  int yyn;
  int yyresult;
  /* Number of tokens to shift before error messages enabled.  */
  int yyerrstatus;
  /* Look-ahead token as an internal (translated) token number.  */
  int yytoken = 0;
#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

  /* Three stacks and their tools:
     `yyss': related to states,
     `yyvs': related to semantic values,
     `yyls': related to locations.

     Refer to the stacks thru separate pointers, to allow yyoverflow
     to reallocate them elsewhere.  */

  /* The state stack.  */
  yytype_int16 yyssa[YYINITDEPTH];
  yytype_int16 *yyss = yyssa;
  yytype_int16 *yyssp;

  /* The semantic value stack.  */
  YYSTYPE yyvsa[YYINITDEPTH];
  YYSTYPE *yyvs = yyvsa;
  YYSTYPE *yyvsp;



#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  YYSIZE_T yystacksize = YYINITDEPTH;

  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;


  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY;		/* Cause a token to be read.  */

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
	YYSTACK_RELOCATE (yyss);
	YYSTACK_RELOCATE (yyvs);

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

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     look-ahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to look-ahead token.  */
  yyn = yypact[yystate];
  if (yyn == YYPACT_NINF)
    goto yydefault;

  /* Not known => get a look-ahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid look-ahead symbol.  */
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

  if (yyn == YYFINAL)
    YYACCEPT;

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the look-ahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token unless it is eof.  */
  if (yychar != YYEOF)
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
#line 155 "xi-grammar.y"
    { (yyval.modlist) = (yyvsp[(1) - (1)].modlist); modlist = (yyvsp[(1) - (1)].modlist); }
    break;

  case 3:
#line 159 "xi-grammar.y"
    { 
		  (yyval.modlist) = 0; 
		}
    break;

  case 4:
#line 163 "xi-grammar.y"
    { (yyval.modlist) = new AstChildren<Module>(lineno, (yyvsp[(1) - (2)].module), (yyvsp[(2) - (2)].modlist)); }
    break;

  case 5:
#line 167 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 6:
#line 169 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 7:
#line 173 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 8:
#line 175 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 9:
#line 179 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 10:
#line 183 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 11:
#line 185 "xi-grammar.y"
    {
		  char *tmp = new char[strlen((yyvsp[(1) - (4)].strval))+strlen((yyvsp[(4) - (4)].strval))+3];
		  sprintf(tmp,"%s::%s", (yyvsp[(1) - (4)].strval), (yyvsp[(4) - (4)].strval));
		  (yyval.strval) = tmp;
		}
    break;

  case 12:
#line 193 "xi-grammar.y"
    { 
		    (yyval.module) = new Module(lineno, (yyvsp[(2) - (3)].strval), (yyvsp[(3) - (3)].conslist)); 
		}
    break;

  case 13:
#line 197 "xi-grammar.y"
    {  
		    (yyval.module) = new Module(lineno, (yyvsp[(2) - (3)].strval), (yyvsp[(3) - (3)].conslist)); 
		    (yyval.module)->setMain();
		}
    break;

  case 14:
#line 204 "xi-grammar.y"
    { (yyval.conslist) = 0; }
    break;

  case 15:
#line 206 "xi-grammar.y"
    { (yyval.conslist) = (yyvsp[(2) - (4)].conslist); }
    break;

  case 16:
#line 210 "xi-grammar.y"
    { (yyval.conslist) = 0; }
    break;

  case 17:
#line 212 "xi-grammar.y"
    { (yyval.conslist) = new ConstructList(lineno, (yyvsp[(1) - (2)].construct), (yyvsp[(2) - (2)].conslist)); }
    break;

  case 18:
#line 216 "xi-grammar.y"
    { (yyval.construct) = new UsingScope((yyvsp[(3) - (3)].strval), false); }
    break;

  case 19:
#line 218 "xi-grammar.y"
    { (yyval.construct) = new UsingScope((yyvsp[(2) - (2)].strval), true); }
    break;

  case 20:
#line 220 "xi-grammar.y"
    { (yyvsp[(2) - (2)].member)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].member); }
    break;

  case 21:
#line 222 "xi-grammar.y"
    { (yyvsp[(2) - (2)].message)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].message); }
    break;

  case 22:
#line 224 "xi-grammar.y"
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

  case 23:
#line 236 "xi-grammar.y"
    { if((yyvsp[(3) - (5)].conslist)) (yyvsp[(3) - (5)].conslist)->recurse<int&>((yyvsp[(1) - (5)].intval), &Construct::setExtern); (yyval.construct) = (yyvsp[(3) - (5)].conslist); }
    break;

  case 24:
#line 238 "xi-grammar.y"
    { (yyval.construct) = new Scope((yyvsp[(2) - (5)].strval), (yyvsp[(4) - (5)].conslist)); }
    break;

  case 25:
#line 240 "xi-grammar.y"
    { (yyval.construct) = (yyvsp[(1) - (2)].construct); }
    break;

  case 26:
#line 242 "xi-grammar.y"
    { yyerror("The preceding construct must be semicolon terminated"); YYABORT; }
    break;

  case 27:
#line 244 "xi-grammar.y"
    { (yyvsp[(2) - (2)].module)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].module); }
    break;

  case 28:
#line 246 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 29:
#line 248 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 30:
#line 250 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 31:
#line 252 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 32:
#line 254 "xi-grammar.y"
    { (yyvsp[(2) - (2)].templat)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].templat); }
    break;

  case 33:
#line 256 "xi-grammar.y"
    { (yyval.construct) = NULL; }
    break;

  case 34:
#line 258 "xi-grammar.y"
    { (yyval.construct) = NULL; }
    break;

  case 35:
#line 260 "xi-grammar.y"
    { (yyval.construct) = (yyvsp[(1) - (1)].accelBlock); }
    break;

  case 36:
#line 262 "xi-grammar.y"
    { printf("Invalid construct\n"); YYABORT; }
    break;

  case 37:
#line 266 "xi-grammar.y"
    { (yyval.tparam) = new TParamType((yyvsp[(1) - (1)].type)); }
    break;

  case 38:
#line 268 "xi-grammar.y"
    { (yyval.tparam) = new TParamVal((yyvsp[(1) - (1)].strval)); }
    break;

  case 39:
#line 270 "xi-grammar.y"
    { (yyval.tparam) = new TParamVal((yyvsp[(1) - (1)].strval)); }
    break;

  case 40:
#line 274 "xi-grammar.y"
    { (yyval.tparlist) = new TParamList((yyvsp[(1) - (1)].tparam)); }
    break;

  case 41:
#line 276 "xi-grammar.y"
    { (yyval.tparlist) = new TParamList((yyvsp[(1) - (3)].tparam), (yyvsp[(3) - (3)].tparlist)); }
    break;

  case 42:
#line 280 "xi-grammar.y"
    { (yyval.tparlist) = 0; }
    break;

  case 43:
#line 282 "xi-grammar.y"
    { (yyval.tparlist) = (yyvsp[(1) - (1)].tparlist); }
    break;

  case 44:
#line 286 "xi-grammar.y"
    { (yyval.tparlist) = 0; }
    break;

  case 45:
#line 288 "xi-grammar.y"
    { (yyval.tparlist) = (yyvsp[(2) - (3)].tparlist); }
    break;

  case 46:
#line 292 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("int"); }
    break;

  case 47:
#line 294 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long"); }
    break;

  case 48:
#line 296 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("short"); }
    break;

  case 49:
#line 298 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("char"); }
    break;

  case 50:
#line 300 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned int"); }
    break;

  case 51:
#line 302 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long"); }
    break;

  case 52:
#line 304 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long"); }
    break;

  case 53:
#line 306 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long long"); }
    break;

  case 54:
#line 308 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned short"); }
    break;

  case 55:
#line 310 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned char"); }
    break;

  case 56:
#line 312 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long long"); }
    break;

  case 57:
#line 314 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("float"); }
    break;

  case 58:
#line 316 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("double"); }
    break;

  case 59:
#line 318 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long double"); }
    break;

  case 60:
#line 320 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("void"); }
    break;

  case 61:
#line 323 "xi-grammar.y"
    { (yyval.ntype) = new NamedType((yyvsp[(1) - (2)].strval),(yyvsp[(2) - (2)].tparlist)); }
    break;

  case 62:
#line 324 "xi-grammar.y"
    { 
                    const char* basename, *scope;
                    splitScopedName((yyvsp[(1) - (2)].strval), &scope, &basename);
                    (yyval.ntype) = new NamedType(basename, (yyvsp[(2) - (2)].tparlist), scope);
                }
    break;

  case 63:
#line 332 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 64:
#line 334 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ntype); }
    break;

  case 65:
#line 338 "xi-grammar.y"
    { (yyval.ptype) = new PtrType((yyvsp[(1) - (2)].type)); }
    break;

  case 66:
#line 342 "xi-grammar.y"
    { (yyvsp[(1) - (2)].ptype)->indirect(); (yyval.ptype) = (yyvsp[(1) - (2)].ptype); }
    break;

  case 67:
#line 344 "xi-grammar.y"
    { (yyvsp[(1) - (2)].ptype)->indirect(); (yyval.ptype) = (yyvsp[(1) - (2)].ptype); }
    break;

  case 68:
#line 348 "xi-grammar.y"
    { (yyval.ftype) = new FuncType((yyvsp[(1) - (8)].type), (yyvsp[(4) - (8)].strval), (yyvsp[(7) - (8)].plist)); }
    break;

  case 69:
#line 352 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 70:
#line 354 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 71:
#line 356 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 72:
#line 358 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ftype); }
    break;

  case 73:
#line 361 "xi-grammar.y"
    { (yyval.type) = new ConstType((yyvsp[(2) - (2)].type)); }
    break;

  case 74:
#line 363 "xi-grammar.y"
    { (yyval.type) = new ConstType((yyvsp[(1) - (2)].type)); }
    break;

  case 75:
#line 367 "xi-grammar.y"
    { (yyval.type) = new ReferenceType((yyvsp[(1) - (2)].type)); }
    break;

  case 76:
#line 369 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 77:
#line 373 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 78:
#line 375 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 79:
#line 379 "xi-grammar.y"
    { (yyval.val) = (yyvsp[(2) - (3)].val); }
    break;

  case 80:
#line 383 "xi-grammar.y"
    { (yyval.vallist) = 0; }
    break;

  case 81:
#line 385 "xi-grammar.y"
    { (yyval.vallist) = new ValueList((yyvsp[(1) - (2)].val), (yyvsp[(2) - (2)].vallist)); }
    break;

  case 82:
#line 389 "xi-grammar.y"
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[(2) - (4)].type), (yyvsp[(3) - (4)].strval), (yyvsp[(4) - (4)].vallist)); }
    break;

  case 83:
#line 393 "xi-grammar.y"
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[(3) - (5)].type), (yyvsp[(5) - (5)].strval), 0, 1); }
    break;

  case 84:
#line 397 "xi-grammar.y"
    { (yyval.intval) = 0;}
    break;

  case 85:
#line 399 "xi-grammar.y"
    { (yyval.intval) = 0;}
    break;

  case 86:
#line 403 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 87:
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
#line 415 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (1)].intval); }
    break;

  case 89:
#line 417 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (3)].intval) | (yyvsp[(3) - (3)].intval); }
    break;

  case 90:
#line 421 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 91:
#line 423 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 92:
#line 427 "xi-grammar.y"
    { (yyval.cattr) = 0; }
    break;

  case 93:
#line 429 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(2) - (3)].cattr); }
    break;

  case 94:
#line 433 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (1)].cattr); }
    break;

  case 95:
#line 435 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (3)].cattr) | (yyvsp[(3) - (3)].cattr); }
    break;

  case 96:
#line 439 "xi-grammar.y"
    { python_doc = NULL; (yyval.intval) = 0; }
    break;

  case 97:
#line 441 "xi-grammar.y"
    { python_doc = (yyvsp[(1) - (1)].strval); (yyval.intval) = 0; }
    break;

  case 98:
#line 445 "xi-grammar.y"
    { (yyval.cattr) = Chare::CPYTHON; }
    break;

  case 99:
#line 449 "xi-grammar.y"
    { (yyval.cattr) = 0; }
    break;

  case 100:
#line 451 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(2) - (3)].cattr); }
    break;

  case 101:
#line 455 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (1)].cattr); }
    break;

  case 102:
#line 457 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (3)].cattr) | (yyvsp[(3) - (3)].cattr); }
    break;

  case 103:
#line 461 "xi-grammar.y"
    { (yyval.cattr) = Chare::CMIGRATABLE; }
    break;

  case 104:
#line 463 "xi-grammar.y"
    { (yyval.cattr) = Chare::CPYTHON; }
    break;

  case 105:
#line 467 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 106:
#line 469 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 107:
#line 472 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 108:
#line 474 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 109:
#line 477 "xi-grammar.y"
    { (yyval.mv) = new MsgVar((yyvsp[(2) - (5)].type), (yyvsp[(3) - (5)].strval), (yyvsp[(1) - (5)].intval), (yyvsp[(4) - (5)].intval)); }
    break;

  case 110:
#line 481 "xi-grammar.y"
    { (yyval.mvlist) = new MsgVarList((yyvsp[(1) - (1)].mv)); }
    break;

  case 111:
#line 483 "xi-grammar.y"
    { (yyval.mvlist) = new MsgVarList((yyvsp[(1) - (2)].mv), (yyvsp[(2) - (2)].mvlist)); }
    break;

  case 112:
#line 487 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, (yyvsp[(3) - (3)].ntype)); }
    break;

  case 113:
#line 489 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, (yyvsp[(3) - (6)].ntype), (yyvsp[(5) - (6)].mvlist)); }
    break;

  case 114:
#line 493 "xi-grammar.y"
    { (yyval.typelist) = 0; }
    break;

  case 115:
#line 495 "xi-grammar.y"
    { (yyval.typelist) = (yyvsp[(2) - (2)].typelist); }
    break;

  case 116:
#line 499 "xi-grammar.y"
    { (yyval.typelist) = new TypeList((yyvsp[(1) - (1)].ntype)); }
    break;

  case 117:
#line 501 "xi-grammar.y"
    { (yyval.typelist) = new TypeList((yyvsp[(1) - (3)].ntype), (yyvsp[(3) - (3)].typelist)); }
    break;

  case 118:
#line 505 "xi-grammar.y"
    { (yyval.chare) = new Chare(lineno, (yyvsp[(2) - (5)].cattr)|Chare::CCHARE, (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 119:
#line 507 "xi-grammar.y"
    { (yyval.chare) = new MainChare(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 120:
#line 511 "xi-grammar.y"
    { (yyval.chare) = new Group(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 121:
#line 515 "xi-grammar.y"
    { (yyval.chare) = new NodeGroup(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 122:
#line 519 "xi-grammar.y"
    {/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",(yyvsp[(2) - (4)].strval));
			(yyval.ntype) = new NamedType(buf); 
		}
    break;

  case 123:
#line 525 "xi-grammar.y"
    { (yyval.ntype) = new NamedType((yyvsp[(2) - (3)].strval)); }
    break;

  case 124:
#line 529 "xi-grammar.y"
    {  (yyval.chare) = new Array(lineno, (yyvsp[(2) - (6)].cattr), (yyvsp[(3) - (6)].ntype), (yyvsp[(4) - (6)].ntype), (yyvsp[(5) - (6)].typelist), (yyvsp[(6) - (6)].mbrlist)); }
    break;

  case 125:
#line 531 "xi-grammar.y"
    {  (yyval.chare) = new Array(lineno, (yyvsp[(3) - (6)].cattr), (yyvsp[(2) - (6)].ntype), (yyvsp[(4) - (6)].ntype), (yyvsp[(5) - (6)].typelist), (yyvsp[(6) - (6)].mbrlist)); }
    break;

  case 126:
#line 535 "xi-grammar.y"
    { (yyval.chare) = new Chare(lineno, (yyvsp[(2) - (5)].cattr)|Chare::CCHARE, new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist));}
    break;

  case 127:
#line 537 "xi-grammar.y"
    { (yyval.chare) = new MainChare(lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 128:
#line 541 "xi-grammar.y"
    { (yyval.chare) = new Group(lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 129:
#line 545 "xi-grammar.y"
    { (yyval.chare) = new NodeGroup( lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 130:
#line 549 "xi-grammar.y"
    { (yyval.chare) = new Array( lineno, 0, (yyvsp[(2) - (5)].ntype), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 131:
#line 553 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[(3) - (4)].strval))); }
    break;

  case 132:
#line 555 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[(3) - (7)].strval)), (yyvsp[(5) - (7)].mvlist)); }
    break;

  case 133:
#line 559 "xi-grammar.y"
    { (yyval.type) = 0; }
    break;

  case 134:
#line 561 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(2) - (2)].type); }
    break;

  case 135:
#line 565 "xi-grammar.y"
    { (yyval.strval) = 0; }
    break;

  case 136:
#line 567 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(2) - (2)].strval); }
    break;

  case 137:
#line 569 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(2) - (2)].strval); }
    break;

  case 138:
#line 573 "xi-grammar.y"
    { (yyval.tvar) = new TType(new NamedType((yyvsp[(2) - (3)].strval)), (yyvsp[(3) - (3)].type)); }
    break;

  case 139:
#line 575 "xi-grammar.y"
    { (yyval.tvar) = new TFunc((yyvsp[(1) - (2)].ftype), (yyvsp[(2) - (2)].strval)); }
    break;

  case 140:
#line 577 "xi-grammar.y"
    { (yyval.tvar) = new TName((yyvsp[(1) - (3)].type), (yyvsp[(2) - (3)].strval), (yyvsp[(3) - (3)].strval)); }
    break;

  case 141:
#line 581 "xi-grammar.y"
    { (yyval.tvarlist) = new TVarList((yyvsp[(1) - (1)].tvar)); }
    break;

  case 142:
#line 583 "xi-grammar.y"
    { (yyval.tvarlist) = new TVarList((yyvsp[(1) - (3)].tvar), (yyvsp[(3) - (3)].tvarlist)); }
    break;

  case 143:
#line 587 "xi-grammar.y"
    { (yyval.tvarlist) = (yyvsp[(3) - (4)].tvarlist); }
    break;

  case 144:
#line 591 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 145:
#line 593 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 146:
#line 595 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 147:
#line 597 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 148:
#line 599 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].message)); (yyvsp[(2) - (2)].message)->setTemplate((yyval.templat)); }
    break;

  case 149:
#line 603 "xi-grammar.y"
    { (yyval.mbrlist) = 0; }
    break;

  case 150:
#line 605 "xi-grammar.y"
    { (yyval.mbrlist) = (yyvsp[(2) - (4)].mbrlist); }
    break;

  case 151:
#line 609 "xi-grammar.y"
    { 
                  if (!connectEntries.empty()) {
                    (yyval.mbrlist) = new AstChildren<Member>(connectEntries);
		  } else {
		    (yyval.mbrlist) = 0; 
                  }
		}
    break;

  case 152:
#line 617 "xi-grammar.y"
    { (yyval.mbrlist) = new AstChildren<Member>(-1, (yyvsp[(1) - (2)].member), (yyvsp[(2) - (2)].mbrlist)); }
    break;

  case 153:
#line 621 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].readonly); }
    break;

  case 154:
#line 623 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].readonly); }
    break;

  case 156:
#line 626 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].member); }
    break;

  case 157:
#line 628 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(2) - (2)].pupable); }
    break;

  case 158:
#line 630 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(2) - (2)].includeFile); }
    break;

  case 159:
#line 632 "xi-grammar.y"
    { (yyval.member) = new ClassDeclaration(lineno,(yyvsp[(2) - (2)].strval)); }
    break;

  case 160:
#line 636 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 1); }
    break;

  case 161:
#line 638 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 1); }
    break;

  case 162:
#line 640 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[(3) - (9)].strval)) + '<' +
					    ((yyvsp[(5) - (9)].tparlist))->to_string() + '>').c_str()),
				    1);
		}
    break;

  case 163:
#line 646 "xi-grammar.y"
    { printf("Warning: deprecated use of initcall. Use initnode or initproc instead.\n"); 
		  (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 1); }
    break;

  case 164:
#line 649 "xi-grammar.y"
    { printf("Warning: deprecated use of initcall. Use initnode or initproc instead.\n");
		  (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 1); }
    break;

  case 165:
#line 654 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 0); }
    break;

  case 166:
#line 656 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 0); }
    break;

  case 167:
#line 658 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[(3) - (9)].strval)) + '<' +
					    ((yyvsp[(5) - (9)].tparlist))->to_string() + '>').c_str()),
				    0);
		}
    break;

  case 168:
#line 664 "xi-grammar.y"
    {
                  InitCall* rtn = new InitCall(lineno, (yyvsp[(6) - (9)].strval), 0);
                  rtn->setAccel();
                  (yyval.member) = rtn;
		}
    break;

  case 169:
#line 672 "xi-grammar.y"
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[(1) - (1)].ntype),0); }
    break;

  case 170:
#line 674 "xi-grammar.y"
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[(1) - (3)].ntype),(yyvsp[(3) - (3)].pupable)); }
    break;

  case 171:
#line 677 "xi-grammar.y"
    { (yyval.includeFile) = new IncludeFile(lineno,(yyvsp[(1) - (1)].strval)); }
    break;

  case 172:
#line 681 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (2)].member); }
    break;

  case 173:
#line 684 "xi-grammar.y"
    { yyerror("The preceding entry method declaration must be semicolon-terminated."); YYABORT; }
    break;

  case 174:
#line 688 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].entry); }
    break;

  case 175:
#line 690 "xi-grammar.y"
    {
                  (yyvsp[(2) - (2)].entry)->tspec = (yyvsp[(1) - (2)].tvarlist);
                  (yyval.member) = (yyvsp[(2) - (2)].entry);
                }
    break;

  case 176:
#line 695 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].member); }
    break;

  case 177:
#line 699 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 178:
#line 701 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 179:
#line 703 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 180:
#line 705 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 181:
#line 707 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 182:
#line 709 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 183:
#line 711 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 184:
#line 713 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 185:
#line 715 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 186:
#line 717 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 187:
#line 719 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 188:
#line 722 "xi-grammar.y"
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

  case 189:
#line 732 "xi-grammar.y"
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

  case 190:
#line 747 "xi-grammar.y"
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

  case 191:
#line 763 "xi-grammar.y"
    { (yyval.accelBlock) = new AccelBlock(lineno, new XStr((yyvsp[(3) - (5)].strval))); }
    break;

  case 192:
#line 765 "xi-grammar.y"
    { (yyval.accelBlock) = new AccelBlock(lineno, NULL); }
    break;

  case 193:
#line 769 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("void"); }
    break;

  case 194:
#line 771 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 195:
#line 775 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 196:
#line 777 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(2) - (3)].intval); }
    break;

  case 197:
#line 779 "xi-grammar.y"
    { printf("Invalid entry method attribute list\n"); YYABORT; }
    break;

  case 198:
#line 783 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (1)].intval); }
    break;

  case 199:
#line 785 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (3)].intval) | (yyvsp[(3) - (3)].intval); }
    break;

  case 200:
#line 789 "xi-grammar.y"
    { (yyval.intval) = STHREADED; }
    break;

  case 201:
#line 791 "xi-grammar.y"
    { (yyval.intval) = SSYNC; }
    break;

  case 202:
#line 793 "xi-grammar.y"
    { (yyval.intval) = SIGET; }
    break;

  case 203:
#line 795 "xi-grammar.y"
    { (yyval.intval) = SLOCKED; }
    break;

  case 204:
#line 797 "xi-grammar.y"
    { (yyval.intval) = SCREATEHERE; }
    break;

  case 205:
#line 799 "xi-grammar.y"
    { (yyval.intval) = SCREATEHOME; }
    break;

  case 206:
#line 801 "xi-grammar.y"
    { (yyval.intval) = SNOKEEP; }
    break;

  case 207:
#line 803 "xi-grammar.y"
    { (yyval.intval) = SNOTRACE; }
    break;

  case 208:
#line 805 "xi-grammar.y"
    { (yyval.intval) = SIMMEDIATE; }
    break;

  case 209:
#line 807 "xi-grammar.y"
    { (yyval.intval) = SSKIPSCHED; }
    break;

  case 210:
#line 809 "xi-grammar.y"
    { (yyval.intval) = SINLINE; }
    break;

  case 211:
#line 811 "xi-grammar.y"
    { (yyval.intval) = SLOCAL; }
    break;

  case 212:
#line 813 "xi-grammar.y"
    { (yyval.intval) = SPYTHON; }
    break;

  case 213:
#line 815 "xi-grammar.y"
    { (yyval.intval) = SMEM; }
    break;

  case 214:
#line 817 "xi-grammar.y"
    { (yyval.intval) = SREDUCE; }
    break;

  case 215:
#line 819 "xi-grammar.y"
    { printf("Invalid entry method attribute: %s\n", yylval); YYABORT; }
    break;

  case 216:
#line 823 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 217:
#line 825 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 218:
#line 827 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 219:
#line 831 "xi-grammar.y"
    { (yyval.strval) = ""; }
    break;

  case 220:
#line 833 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 221:
#line 835 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[(1) - (3)].strval))+strlen((yyvsp[(3) - (3)].strval))+3];
			sprintf(tmp,"%s, %s", (yyvsp[(1) - (3)].strval), (yyvsp[(3) - (3)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 222:
#line 843 "xi-grammar.y"
    { (yyval.strval) = ""; }
    break;

  case 223:
#line 845 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 224:
#line 847 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s[%s]%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 225:
#line 853 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s{%s}%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 226:
#line 859 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s(%s)%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 227:
#line 865 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(2) - (4)].strval))+strlen((yyvsp[(4) - (4)].strval))+3];
			sprintf(tmp,"(%s)%s", (yyvsp[(2) - (4)].strval), (yyvsp[(4) - (4)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 228:
#line 873 "xi-grammar.y"
    {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			(yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].type),(yyvsp[(2) - (3)].strval));
		}
    break;

  case 229:
#line 880 "xi-grammar.y"
    { 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			(yyval.intval) = 0;
		}
    break;

  case 230:
#line 888 "xi-grammar.y"
    { 
			in_braces=0;
			(yyval.intval) = 0;
		}
    break;

  case 231:
#line 895 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (1)].type));}
    break;

  case 232:
#line 897 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].type),(yyvsp[(2) - (3)].strval)); (yyval.pname)->setConditional((yyvsp[(3) - (3)].intval)); }
    break;

  case 233:
#line 899 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (4)].type),(yyvsp[(2) - (4)].strval),0,(yyvsp[(4) - (4)].val));}
    break;

  case 234:
#line 901 "xi-grammar.y"
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].pname)->getType(), (yyvsp[(1) - (3)].pname)->getName() ,(yyvsp[(2) - (3)].strval));
		}
    break;

  case 235:
#line 907 "xi-grammar.y"
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READONLY; }
    break;

  case 236:
#line 908 "xi-grammar.y"
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READWRITE; }
    break;

  case 237:
#line 909 "xi-grammar.y"
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_WRITEONLY; }
    break;

  case 238:
#line 912 "xi-grammar.y"
    { (yyval.xstrptr) = new XStr((yyvsp[(1) - (1)].strval)); }
    break;

  case 239:
#line 913 "xi-grammar.y"
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "->" << (yyvsp[(4) - (4)].strval); }
    break;

  case 240:
#line 914 "xi-grammar.y"
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[(1) - (3)].xstrptr)) << "." << (yyvsp[(3) - (3)].strval); }
    break;

  case 241:
#line 916 "xi-grammar.y"
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "[" << *((yyvsp[(3) - (4)].xstrptr)) << "]";
                  delete (yyvsp[(1) - (4)].xstrptr);
                  delete (yyvsp[(3) - (4)].xstrptr);
                }
    break;

  case 242:
#line 923 "xi-grammar.y"
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "[" << (yyvsp[(3) - (4)].strval) << "]";
                  delete (yyvsp[(1) - (4)].xstrptr);
                }
    break;

  case 243:
#line 929 "xi-grammar.y"
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "(" << *((yyvsp[(3) - (4)].xstrptr)) << ")";
                  delete (yyvsp[(1) - (4)].xstrptr);
                  delete (yyvsp[(3) - (4)].xstrptr);
                }
    break;

  case 244:
#line 938 "xi-grammar.y"
    {
                  in_bracket = 0;
                  (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].pname)->getType(), (yyvsp[(1) - (3)].pname)->getName(), (yyvsp[(2) - (3)].strval));
                }
    break;

  case 245:
#line 945 "xi-grammar.y"
    {
                  (yyval.pname) = new Parameter(lineno, (yyvsp[(3) - (7)].type), (yyvsp[(4) - (7)].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[(6) - (7)].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[(1) - (7)].intval));
                }
    break;

  case 246:
#line 951 "xi-grammar.y"
    {
		  (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (5)].type), (yyvsp[(2) - (5)].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[(4) - (5)].xstrptr));
                  (yyval.pname)->setAccelBufferType(Parameter::ACCEL_BUFFER_TYPE_READWRITE);
		}
    break;

  case 247:
#line 957 "xi-grammar.y"
    {
                  (yyval.pname) = (yyvsp[(3) - (6)].pname);
                  (yyval.pname)->setAccelInstName((yyvsp[(5) - (6)].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[(1) - (6)].intval));
		}
    break;

  case 248:
#line 965 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (1)].pname)); }
    break;

  case 249:
#line 967 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (3)].pname),(yyvsp[(3) - (3)].plist)); }
    break;

  case 250:
#line 971 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (1)].pname)); }
    break;

  case 251:
#line 973 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (3)].pname),(yyvsp[(3) - (3)].plist)); }
    break;

  case 252:
#line 977 "xi-grammar.y"
    { (yyval.plist) = (yyvsp[(2) - (3)].plist); }
    break;

  case 253:
#line 979 "xi-grammar.y"
    { (yyval.plist) = new ParamList(new Parameter(0, new BuiltinType("void"))); }
    break;

  case 254:
#line 983 "xi-grammar.y"
    { (yyval.plist) = (yyvsp[(2) - (3)].plist); }
    break;

  case 255:
#line 985 "xi-grammar.y"
    { (yyval.plist) = 0; }
    break;

  case 256:
#line 989 "xi-grammar.y"
    { (yyval.val) = 0; }
    break;

  case 257:
#line 991 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(3) - (3)].strval)); }
    break;

  case 258:
#line 995 "xi-grammar.y"
    { (yyval.sc) = 0; }
    break;

  case 259:
#line 997 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSDAGENTRY, (yyvsp[(1) - (1)].sc)); }
    break;

  case 260:
#line 999 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSDAGENTRY, (yyvsp[(2) - (3)].sc)); }
    break;

  case 261:
#line 1003 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSLIST, (yyvsp[(1) - (1)].sc)); }
    break;

  case 262:
#line 1005 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSLIST, (yyvsp[(1) - (2)].sc), (yyvsp[(2) - (2)].sc));  }
    break;

  case 263:
#line 1009 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SOLIST, (yyvsp[(1) - (1)].sc)); }
    break;

  case 264:
#line 1011 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SOLIST, (yyvsp[(1) - (2)].sc), (yyvsp[(2) - (2)].sc)); }
    break;

  case 265:
#line 1015 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SCASELIST, (yyvsp[(1) - (1)].when)); }
    break;

  case 266:
#line 1017 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SCASELIST, (yyvsp[(1) - (2)].when), (yyvsp[(2) - (2)].sc)); }
    break;

  case 267:
#line 1019 "xi-grammar.y"
    { yyerror("Case blocks in SDAG can only contain when clauses."); YYABORT; }
    break;

  case 268:
#line 1023 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 269:
#line 1025 "xi-grammar.y"
    { (yyval.strval) = 0; }
    break;

  case 270:
#line 1029 "xi-grammar.y"
    { (yyval.when) = new WhenConstruct((yyvsp[(2) - (4)].entrylist), 0); }
    break;

  case 271:
#line 1031 "xi-grammar.y"
    { (yyval.when) = new WhenConstruct((yyvsp[(2) - (3)].entrylist), (yyvsp[(3) - (3)].sc)); }
    break;

  case 272:
#line 1033 "xi-grammar.y"
    { (yyval.when) = new WhenConstruct((yyvsp[(2) - (5)].entrylist), (yyvsp[(4) - (5)].sc)); }
    break;

  case 273:
#line 1037 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 274:
#line 1039 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 275:
#line 1041 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 276:
#line 1043 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 277:
#line 1045 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 278:
#line 1047 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 279:
#line 1051 "xi-grammar.y"
    { (yyval.sc) = new AtomicConstruct((yyvsp[(4) - (5)].strval), (yyvsp[(2) - (5)].strval)); }
    break;

  case 280:
#line 1053 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SOVERLAP,0, 0,0,0,0,(yyvsp[(3) - (4)].sc), 0); }
    break;

  case 281:
#line 1055 "xi-grammar.y"
    { (yyval.sc) = (yyvsp[(1) - (1)].when); }
    break;

  case 282:
#line 1057 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SCASE, 0, 0, 0, 0, 0, (yyvsp[(3) - (4)].sc), 0); }
    break;

  case 283:
#line 1059 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFOR, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (11)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(5) - (11)].strval)),
		             new SdagConstruct(SINT_EXPR, (yyvsp[(7) - (11)].strval)), 0, (yyvsp[(10) - (11)].sc), 0); }
    break;

  case 284:
#line 1062 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFOR, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (9)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(5) - (9)].strval)), 
		         new SdagConstruct(SINT_EXPR, (yyvsp[(7) - (9)].strval)), 0, (yyvsp[(9) - (9)].sc), 0); }
    break;

  case 285:
#line 1065 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFORALL, 0, new SdagConstruct(SIDENT, (yyvsp[(3) - (12)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(6) - (12)].strval)), 
		             new SdagConstruct(SINT_EXPR, (yyvsp[(8) - (12)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(10) - (12)].strval)), (yyvsp[(12) - (12)].sc), 0); }
    break;

  case 286:
#line 1068 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFORALL, 0, new SdagConstruct(SIDENT, (yyvsp[(3) - (14)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(6) - (14)].strval)), 
		                 new SdagConstruct(SINT_EXPR, (yyvsp[(8) - (14)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(10) - (14)].strval)), (yyvsp[(13) - (14)].sc), 0); }
    break;

  case 287:
#line 1071 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (6)].strval)), (yyvsp[(6) - (6)].sc),0,0,(yyvsp[(5) - (6)].sc),0); }
    break;

  case 288:
#line 1073 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (8)].strval)), (yyvsp[(8) - (8)].sc),0,0,(yyvsp[(6) - (8)].sc),0); }
    break;

  case 289:
#line 1075 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SWHILE, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (5)].strval)), 0,0,0,(yyvsp[(5) - (5)].sc),0); }
    break;

  case 290:
#line 1077 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SWHILE, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (7)].strval)), 0,0,0,(yyvsp[(6) - (7)].sc),0); }
    break;

  case 291:
#line 1079 "xi-grammar.y"
    { (yyval.sc) = new AtomicConstruct((yyvsp[(2) - (3)].strval), NULL); }
    break;

  case 292:
#line 1081 "xi-grammar.y"
    { printf("Unknown SDAG construct or malformed entry method definition.\n"
                         "You may have forgotten to terminate an entry method definition with a"
                         " semicolon or forgotten to mark a block of sequential SDAG code as 'atomic'\n"); YYABORT; }
    break;

  case 293:
#line 1087 "xi-grammar.y"
    { (yyval.sc) = 0; }
    break;

  case 294:
#line 1089 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SELSE, 0,0,0,0,0, (yyvsp[(2) - (2)].sc),0); }
    break;

  case 295:
#line 1091 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SELSE, 0,0,0,0,0, (yyvsp[(3) - (4)].sc),0); }
    break;

  case 296:
#line 1095 "xi-grammar.y"
    { in_int_expr = 0; (yyval.intval) = 0; }
    break;

  case 297:
#line 1099 "xi-grammar.y"
    { in_int_expr = 1; (yyval.intval) = 0; }
    break;

  case 298:
#line 1103 "xi-grammar.y"
    { (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[(1) - (2)].strval), (yyvsp[(2) - (2)].plist), 0, 0, 0); }
    break;

  case 299:
#line 1105 "xi-grammar.y"
    { (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[(1) - (5)].strval), (yyvsp[(5) - (5)].plist), 0, 0, (yyvsp[(3) - (5)].strval)); }
    break;

  case 300:
#line 1109 "xi-grammar.y"
    { (yyval.entrylist) = new EntryList((yyvsp[(1) - (1)].entry)); }
    break;

  case 301:
#line 1111 "xi-grammar.y"
    { (yyval.entrylist) = new EntryList((yyvsp[(1) - (3)].entry),(yyvsp[(3) - (3)].entrylist)); }
    break;

  case 302:
#line 1115 "xi-grammar.y"
    { in_bracket=1; }
    break;

  case 303:
#line 1118 "xi-grammar.y"
    { in_bracket=0; }
    break;

  case 304:
#line 1122 "xi-grammar.y"
    { if (!macroDefined((yyvsp[(2) - (2)].strval), 1)) in_comment = 1; }
    break;

  case 305:
#line 1126 "xi-grammar.y"
    { if (!macroDefined((yyvsp[(2) - (2)].strval), 0)) in_comment = 1; }
    break;


/* Line 1267 of yacc.c.  */
#line 3803 "y.tab.c"
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
      /* If just tried and failed to reuse look-ahead token after an
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

  /* Else will try to reuse look-ahead token after shifting the error
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

  if (yyn == YYFINAL)
    YYACCEPT;

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

#ifndef yyoverflow
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEOF && yychar != YYEMPTY)
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


#line 1129 "xi-grammar.y"

void yyerror(const char *mesg)
{
    std::cerr << cur_file<<":"<<lineno<<": Charmxi syntax error> "
	      << mesg << std::endl;
}

