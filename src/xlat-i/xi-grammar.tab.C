
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
AstChildren<Module> *modlist;
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

/* Line 214 of yacc.c  */
#line 23 "xi-grammar.y"

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



/* Line 214 of yacc.c  */
#line 316 "y.tab.c"
} YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
#endif


/* Copy the second part of user declarations.  */


/* Line 264 of yacc.c  */
#line 328 "y.tab.c"

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
#define YYLAST   883

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  89
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  113
/* YYNRULES -- Number of rules.  */
#define YYNRULES  306
/* YYNRULES -- Number of states.  */
#define YYNSTATES  599

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
     671,   673,   675,   677,   680,   682,   684,   686,   688,   690,
     692,   693,   695,   699,   700,   702,   708,   714,   720,   725,
     729,   731,   733,   735,   739,   744,   748,   750,   752,   754,
     756,   761,   765,   770,   775,   780,   784,   792,   798,   805,
     807,   811,   813,   817,   821,   824,   828,   831,   832,   836,
     837,   839,   843,   845,   848,   850,   853,   855,   858,   860,
     862,   863,   868,   872,   878,   880,   882,   884,   886,   888,
     890,   896,   901,   903,   908,   920,   930,   943,   958,   965,
     974,   980,   988,   992,   994,   995,   998,  1003,  1005,  1007,
    1010,  1016,  1018,  1022,  1024,  1026,  1029
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
      90,     0,    -1,    91,    -1,    -1,    96,    91,    -1,    -1,
       5,    -1,    -1,    73,    -1,    53,    -1,    53,    -1,    95,
      74,    74,    53,    -1,     3,    94,    97,    -1,     4,    94,
      97,    -1,    73,    -1,    75,    98,    76,    93,    -1,    -1,
     100,    98,    -1,    52,    51,    95,    -1,    52,    95,    -1,
      92,   155,    -1,    92,   134,    -1,     5,    40,   165,   107,
      94,   104,   182,    -1,    92,    75,    98,    76,    93,    -1,
      51,    94,    75,    98,    76,    -1,    99,    73,    -1,    99,
     162,    -1,    92,    96,    -1,    92,   137,    -1,    92,   138,
      -1,    92,   139,    -1,    92,   141,    -1,    92,   152,    -1,
     200,    -1,   201,    -1,   164,    -1,     1,    -1,   113,    -1,
      54,    -1,    55,    -1,   101,    -1,   101,    77,   102,    -1,
      -1,   102,    -1,    -1,    78,   103,    79,    -1,    59,    -1,
      60,    -1,    61,    -1,    62,    -1,    65,    59,    -1,    65,
      60,    -1,    65,    60,    59,    -1,    65,    60,    60,    -1,
      65,    61,    -1,    65,    62,    -1,    60,    60,    -1,    63,
      -1,    64,    -1,    60,    64,    -1,    36,    -1,    94,   104,
      -1,    95,   104,    -1,   105,    -1,   107,    -1,   108,    80,
      -1,   109,    80,    -1,   110,    80,    -1,   112,    81,    80,
      94,    82,    81,   180,    82,    -1,   108,    -1,   109,    -1,
     110,    -1,   111,    -1,    37,   112,    -1,   112,    37,    -1,
     112,    83,    -1,   112,    -1,    54,    -1,    95,    -1,    84,
     114,    85,    -1,    -1,   115,   116,    -1,     6,   113,    95,
     116,    -1,     6,    16,   108,    80,    94,    -1,    -1,    36,
      -1,    -1,    84,   121,    85,    -1,   122,    -1,   122,    77,
     121,    -1,    38,    -1,    39,    -1,    -1,    84,   124,    85,
      -1,   129,    -1,   129,    77,   124,    -1,    -1,    55,    -1,
      49,    -1,    -1,    84,   128,    85,    -1,   126,    -1,   126,
      77,   128,    -1,    30,    -1,    49,    -1,    -1,    17,    -1,
      -1,    84,    85,    -1,   130,   113,    94,   131,    73,    -1,
     132,    -1,   132,   133,    -1,    16,   120,   106,    -1,    16,
     120,   106,    75,   133,    76,    -1,    -1,    74,   136,    -1,
     107,    -1,   107,    77,   136,    -1,    11,   123,   106,   135,
     153,    -1,    12,   123,   106,   135,   153,    -1,    13,   123,
     106,   135,   153,    -1,    14,   123,   106,   135,   153,    -1,
      84,    54,    94,    85,    -1,    84,    94,    85,    -1,    15,
     127,   140,   106,   135,   153,    -1,    15,   140,   127,   106,
     135,   153,    -1,    11,   123,    94,   135,   153,    -1,    12,
     123,    94,   135,   153,    -1,    13,   123,    94,   135,   153,
      -1,    14,   123,    94,   135,   153,    -1,    15,   140,    94,
     135,   153,    -1,    16,   120,    94,    73,    -1,    16,   120,
      94,    75,   133,    76,    73,    -1,    -1,    86,   113,    -1,
      -1,    86,    54,    -1,    86,    55,    -1,    18,    94,   147,
      -1,   111,   148,    -1,   113,    94,   148,    -1,   149,    -1,
     149,    77,   150,    -1,    22,    78,   150,    79,    -1,   151,
     142,    -1,   151,   143,    -1,   151,   144,    -1,   151,   145,
      -1,   151,   146,    -1,    73,    -1,    75,   154,    76,    93,
      -1,    -1,   160,   154,    -1,   117,    -1,   118,    -1,   157,
      -1,   156,    -1,    10,   158,    -1,    19,   159,    -1,    18,
      94,    -1,     8,   119,    95,    -1,     8,   119,    95,    81,
     119,    82,    -1,     8,   119,    95,    78,   102,    79,    81,
     119,    82,    -1,     7,   119,    95,    -1,     7,   119,    95,
      81,   119,    82,    -1,     9,   119,    95,    -1,     9,   119,
      95,    81,   119,    82,    -1,     9,   119,    95,    78,   102,
      79,    81,   119,    82,    -1,     9,    84,    66,    85,   119,
      95,    81,   119,    82,    -1,   107,    -1,   107,    77,   158,
      -1,    55,    -1,   161,    73,    -1,   161,   162,    -1,   163,
      -1,   151,   163,    -1,   155,    -1,    40,    -1,    76,    -1,
       7,    -1,     8,    -1,     9,    -1,    11,    -1,    12,    -1,
      15,    -1,    13,    -1,    14,    -1,     6,    -1,    40,   166,
     165,    94,   182,   184,   185,    -1,    40,   166,    94,   182,
     185,    -1,    40,    84,    66,    85,    36,    94,   182,   183,
     173,   171,   174,    94,    -1,    69,   173,   171,   174,    73,
      -1,    69,    73,    -1,    36,    -1,   109,    -1,    -1,    84,
     167,    85,    -1,     1,    -1,   168,    -1,   168,    77,   167,
      -1,    21,    -1,    23,    -1,    24,    -1,    25,    -1,    31,
      -1,    32,    -1,    33,    -1,    34,    -1,    35,    -1,    26,
      -1,    27,    -1,    28,    -1,    50,    -1,    49,   125,    -1,
      70,    -1,    71,    -1,     1,    -1,    55,    -1,    54,    -1,
      95,    -1,    -1,    56,    -1,    56,    77,   170,    -1,    -1,
      56,    -1,    56,    84,   171,    85,   171,    -1,    56,    75,
     171,    76,   171,    -1,    56,    81,   170,    82,   171,    -1,
      81,   171,    82,   171,    -1,   113,    94,    84,    -1,    75,
      -1,    76,    -1,   113,    -1,   113,    94,   130,    -1,   113,
      94,    86,   169,    -1,   172,   171,    85,    -1,     6,    -1,
      67,    -1,    68,    -1,    94,    -1,   177,    87,    79,    94,
      -1,   177,    88,    94,    -1,   177,    84,   177,    85,    -1,
     177,    84,    54,    85,    -1,   177,    81,   177,    82,    -1,
     172,   171,    85,    -1,   176,    74,   113,    94,    78,   177,
      79,    -1,   113,    94,    78,   177,    79,    -1,   176,    74,
     178,    78,   177,    79,    -1,   175,    -1,   175,    77,   180,
      -1,   179,    -1,   179,    77,   181,    -1,    81,   180,    82,
      -1,    81,    82,    -1,    84,   181,    85,    -1,    84,    85,
      -1,    -1,    20,    86,    54,    -1,    -1,   192,    -1,    75,
     186,    76,    -1,   192,    -1,   192,   186,    -1,   192,    -1,
     192,   186,    -1,   190,    -1,   190,   188,    -1,   191,    -1,
      55,    -1,    -1,    44,   197,    75,    76,    -1,    44,   197,
     192,    -1,    44,   197,    75,   186,    76,    -1,    46,    -1,
      45,    -1,    41,    -1,    42,    -1,    47,    -1,    43,    -1,
      46,   189,   173,   171,   174,    -1,    45,    75,   187,    76,
      -1,   190,    -1,    72,    75,   188,    76,    -1,    41,   195,
     171,    73,   171,    73,   171,   194,    75,   186,    76,    -1,
      41,   195,   171,    73,   171,    73,   171,   194,   192,    -1,
      42,    84,    53,    85,   195,   171,    74,   171,    77,   171,
     194,   192,    -1,    42,    84,    53,    85,   195,   171,    74,
     171,    77,   171,   194,    75,   186,    76,    -1,    47,   195,
     171,   194,   192,   193,    -1,    47,   195,   171,   194,    75,
     186,    76,   193,    -1,    43,   195,   171,   194,   192,    -1,
      43,   195,   171,   194,    75,   186,    76,    -1,   173,   171,
     174,    -1,     1,    -1,    -1,    48,   192,    -1,    48,    75,
     186,    76,    -1,    82,    -1,    81,    -1,    53,   182,    -1,
      53,   198,   171,   199,   182,    -1,   196,    -1,   196,    77,
     197,    -1,    84,    -1,    85,    -1,    57,    94,    -1,    58,
      94,    -1
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
     808,   810,   812,   814,   816,   818,   820,   824,   826,   828,
     833,   834,   836,   845,   846,   848,   854,   860,   866,   874,
     881,   889,   896,   898,   900,   902,   909,   910,   911,   914,
     915,   916,   917,   924,   930,   939,   946,   952,   958,   966,
     968,   972,   974,   978,   980,   984,   986,   991,   992,   997,
     998,  1000,  1004,  1006,  1010,  1012,  1016,  1018,  1020,  1024,
    1027,  1030,  1032,  1034,  1038,  1040,  1042,  1044,  1046,  1048,
    1052,  1054,  1056,  1058,  1060,  1063,  1066,  1069,  1072,  1074,
    1076,  1078,  1080,  1082,  1089,  1090,  1092,  1096,  1100,  1104,
    1106,  1110,  1112,  1116,  1119,  1123,  1127
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
      95,    95,    96,    96,    97,    97,    98,    98,    99,    99,
      99,    99,    99,   100,   100,   100,   100,   100,   100,   100,
     100,   100,   100,   100,   100,   100,   100,   101,   101,   101,
     102,   102,   103,   103,   104,   104,   105,   105,   105,   105,
     105,   105,   105,   105,   105,   105,   105,   105,   105,   105,
     105,   106,   107,   108,   108,   109,   110,   110,   111,   112,
     112,   112,   112,   112,   112,   113,   113,   114,   114,   115,
     116,   116,   117,   118,   119,   119,   120,   120,   121,   121,
     122,   122,   123,   123,   124,   124,   125,   125,   126,   127,
     127,   128,   128,   129,   129,   130,   130,   131,   131,   132,
     133,   133,   134,   134,   135,   135,   136,   136,   137,   137,
     138,   139,   140,   140,   141,   141,   142,   142,   143,   144,
     145,   146,   146,   147,   147,   148,   148,   148,   149,   149,
     149,   150,   150,   151,   152,   152,   152,   152,   152,   153,
     153,   154,   154,   155,   155,   155,   155,   155,   155,   155,
     156,   156,   156,   156,   156,   157,   157,   157,   157,   158,
     158,   159,   160,   160,   161,   161,   161,   162,   162,   162,
     162,   162,   162,   162,   162,   162,   162,   162,   163,   163,
     163,   164,   164,   165,   165,   166,   166,   166,   167,   167,
     168,   168,   168,   168,   168,   168,   168,   168,   168,   168,
     168,   168,   168,   168,   168,   168,   168,   169,   169,   169,
     170,   170,   170,   171,   171,   171,   171,   171,   171,   172,
     173,   174,   175,   175,   175,   175,   176,   176,   176,   177,
     177,   177,   177,   177,   177,   178,   179,   179,   179,   180,
     180,   181,   181,   182,   182,   183,   183,   184,   184,   185,
     185,   185,   186,   186,   187,   187,   188,   188,   188,   189,
     189,   190,   190,   190,   191,   191,   191,   191,   191,   191,
     192,   192,   192,   192,   192,   192,   192,   192,   192,   192,
     192,   192,   192,   192,   193,   193,   193,   194,   195,   196,
     196,   197,   197,   198,   199,   200,   201
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
       1,     1,     1,     2,     1,     1,     1,     1,     1,     1,
       0,     1,     3,     0,     1,     5,     5,     5,     4,     3,
       1,     1,     1,     3,     4,     3,     1,     1,     1,     1,
       4,     3,     4,     4,     4,     3,     7,     5,     6,     1,
       3,     1,     3,     3,     2,     3,     2,     0,     3,     0,
       1,     3,     1,     2,     1,     2,     1,     2,     1,     1,
       0,     4,     3,     5,     1,     1,     1,     1,     1,     1,
       5,     4,     1,     4,    11,     9,    12,    14,     6,     8,
       5,     7,     3,     1,     0,     2,     4,     1,     1,     2,
       5,     1,     3,     1,     1,     2,     2
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       3,     0,     0,     0,     2,     3,     9,     0,     0,     1,
       4,    14,     0,    12,    13,    36,     6,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    35,    33,    34,     0,
       0,     0,    10,    19,   305,   306,   192,   230,   223,     0,
      84,    84,    84,     0,    92,    92,    92,    92,     0,    86,
       0,     0,     0,     0,    27,   153,   154,    21,    28,    29,
      30,    31,     0,    32,    20,   156,   155,     7,   187,   179,
     180,   181,   182,   183,   185,   186,   184,   177,    25,   178,
      26,    17,    60,    46,    47,    48,    49,    57,    58,     0,
      44,    63,    64,     0,   194,     0,     0,    18,     0,   224,
     223,     0,     0,    60,     0,    69,    70,    71,    72,    76,
       0,    85,     0,     0,     0,     0,   169,   157,     0,     0,
       0,     0,     0,     0,     0,    99,     0,     0,   159,   171,
     158,     0,     0,    92,    92,    92,    92,     0,    86,   144,
     145,   146,   147,   148,     8,    15,    56,    59,    50,    51,
      54,    55,    42,    62,    65,     0,     0,     0,   223,   220,
     223,     0,   231,     0,     0,    73,    66,    67,    74,     0,
      75,    80,   163,   160,     0,   165,     0,   103,   104,     0,
      94,    44,   114,   114,   114,   114,    98,     0,     0,   101,
       0,     0,     0,     0,     0,    90,    91,     0,    88,   112,
       0,    72,     0,   141,     0,     7,     0,     0,     0,     0,
       0,     0,    52,    53,    38,    39,    40,    43,     0,    37,
      44,    24,    11,     0,   221,     0,     0,   223,   191,     0,
       0,     0,    80,    82,    84,     0,    84,    84,     0,    84,
     170,    93,     0,    61,     0,     0,     0,     0,     0,     0,
     123,     0,   100,   114,   114,    87,     0,   105,   133,     0,
     139,   135,     0,   143,    23,   114,   114,   114,   114,   114,
       0,     0,    45,     0,   223,   220,   223,   223,   228,    83,
       0,    77,    78,     0,    81,     0,     0,     0,     0,     0,
       0,    95,   116,   115,   149,   151,   118,   119,   120,   121,
     122,   102,     0,     0,    89,   106,     0,   105,     0,     0,
     138,   136,   137,   140,   142,     0,     0,     0,     0,     0,
     131,   105,    41,     0,    22,   226,   222,   227,   225,     0,
      79,   164,     0,   161,     0,     0,   166,     0,     0,     0,
       0,   176,   151,     0,   174,   124,   125,     0,   111,   113,
     134,   126,   127,   128,   129,   130,     0,   254,   232,   223,
     249,     0,     0,    84,    84,    84,   117,   197,     0,     0,
     175,     7,   152,   172,   173,   107,     0,   105,     0,     0,
     253,     0,     0,     0,     0,   216,   200,   201,   202,   203,
     209,   210,   211,   204,   205,   206,   207,   208,    96,   212,
       0,   214,   215,     0,   198,    10,     0,     0,   150,     0,
       0,   132,   229,     0,   233,   235,   250,    68,   162,   168,
     167,    97,   213,     0,   196,     0,     0,     0,   108,   109,
     218,   217,   219,   234,     0,   199,   293,     0,     0,     0,
       0,     0,   270,     0,     0,     0,   223,   189,   282,   260,
     257,     0,   298,   223,     0,   223,     0,   301,     0,     0,
     269,     0,   223,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   303,   299,   223,     0,     0,   272,     0,
       0,   223,     0,   276,   277,   279,   275,   274,   278,     0,
     266,   268,   261,   263,   292,     0,   188,     0,     0,   223,
       0,   297,     0,     0,   302,   271,     0,   281,   265,     0,
       0,   283,   267,   258,   236,   237,   238,   256,     0,     0,
     251,     0,   223,     0,   223,     0,   290,   304,     0,   273,
     280,     0,   294,     0,     0,     0,   255,     0,   223,     0,
       0,   300,     0,     0,   288,     0,     0,   223,     0,   252,
       0,     0,   223,   291,   294,     0,   295,   239,     0,     0,
       0,     0,   190,     0,     0,   289,     0,   247,     0,     0,
       0,     0,     0,   245,     0,     0,   285,   223,   296,     0,
       0,     0,     0,   241,     0,   248,     0,     0,   244,   243,
     242,   240,   246,   284,     0,     0,   286,     0,   287
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     3,     4,    22,   145,   181,    90,     5,    13,    23,
      24,    25,   216,   217,   218,   153,    91,   182,    92,   105,
     106,   107,   108,   109,   219,   283,   232,   233,    55,    56,
     112,   127,   197,   198,   119,   179,   422,   189,   124,   190,
     180,   306,   410,   307,   308,    57,   245,   293,    58,    59,
      60,   125,    61,   139,   140,   141,   142,   143,   310,   260,
     203,   204,   339,    63,   296,   340,   341,    65,    66,   117,
     130,   342,   343,    80,   344,    26,    95,   369,   403,   404,
     433,   225,   101,   359,   446,   163,   360,   519,   558,   548,
     520,   361,   521,   324,   498,   468,   447,   464,   479,   489,
     461,   448,   491,   465,   544,   502,   453,   457,   458,   475,
     528,    27,    28
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -471
static const yytype_int16 yypact[] =
{
     200,    11,    11,    29,  -471,   200,  -471,    59,    59,  -471,
    -471,  -471,   500,  -471,  -471,  -471,    43,    11,   122,    11,
      11,   127,   617,    52,   304,   500,  -471,  -471,  -471,   787,
      33,    85,  -471,    70,  -471,  -471,  -471,  -471,   -12,   704,
     119,   119,   -11,    85,    66,    66,    66,    66,    74,    77,
      11,   128,   115,   500,  -471,  -471,  -471,  -471,  -471,  -471,
    -471,  -471,   339,  -471,  -471,  -471,  -471,   139,  -471,  -471,
    -471,  -471,  -471,  -471,  -471,  -471,  -471,  -471,  -471,  -471,
    -471,  -471,   170,  -471,    20,  -471,  -471,  -471,  -471,   220,
      19,  -471,  -471,   146,  -471,    85,   500,    70,   169,    34,
     -12,   185,   800,  -471,   769,   146,   183,   186,  -471,    -9,
      85,  -471,    85,    85,   207,    85,   206,  -471,     2,    11,
      11,    11,    11,   211,   193,   201,   208,    11,  -471,  -471,
    -471,   724,   233,    66,    66,    66,    66,   193,    77,  -471,
    -471,  -471,  -471,  -471,  -471,  -471,  -471,  -471,  -471,   230,
    -471,  -471,   756,  -471,  -471,    11,   246,   243,   -12,   258,
     -12,   242,  -471,   256,   251,    -6,  -471,  -471,  -471,   261,
    -471,   -34,    55,   114,   277,   153,    85,  -471,  -471,   279,
     302,   265,   307,   307,   307,   307,  -471,    11,   300,   309,
     315,   241,    11,   338,    11,  -471,  -471,   320,   330,   333,
      11,    56,    11,   340,   337,   139,    11,    11,    11,    11,
      11,    11,  -471,  -471,  -471,  -471,   341,  -471,   344,  -471,
     265,  -471,  -471,   348,   343,   353,   345,   -12,  -471,    11,
      11,   254,   342,  -471,   119,   756,   119,   119,   756,   119,
    -471,  -471,     2,  -471,    85,   149,   149,   149,   149,   351,
    -471,   338,  -471,   307,   307,  -471,   208,   415,   365,   266,
    -471,   366,   724,  -471,  -471,   307,   307,   307,   307,   307,
     177,   756,  -471,   358,   -12,   258,   -12,   -12,  -471,  -471,
     371,  -471,    70,   362,  -471,   372,   377,   375,    85,   379,
     393,  -471,   382,  -471,  -471,   772,  -471,  -471,  -471,  -471,
    -471,  -471,   149,   149,  -471,  -471,   769,     3,   400,   769,
    -471,  -471,  -471,  -471,  -471,   149,   149,   149,   149,   149,
    -471,   415,  -471,   616,  -471,  -471,  -471,  -471,  -471,   396,
    -471,  -471,   397,  -471,    79,   398,  -471,    85,   103,   442,
     407,  -471,   772,   526,  -471,  -471,  -471,    11,  -471,  -471,
    -471,  -471,  -471,  -471,  -471,  -471,   408,  -471,    11,   -12,
     409,   405,   769,   119,   119,   119,  -471,  -471,   633,   818,
    -471,   139,  -471,  -471,  -471,   404,   416,     5,   406,   769,
    -471,   411,   412,   414,   418,  -471,  -471,  -471,  -471,  -471,
    -471,  -471,  -471,  -471,  -471,  -471,  -471,  -471,   435,  -471,
     436,  -471,  -471,   438,   443,   455,   358,    11,  -471,   439,
     469,  -471,  -471,   116,  -471,  -471,  -471,  -471,  -471,  -471,
    -471,  -471,  -471,   507,  -471,   684,   571,   358,  -471,  -471,
    -471,  -471,    70,  -471,    11,  -471,  -471,   465,   463,   465,
     497,   478,   499,   465,   480,   195,   -12,  -471,  -471,  -471,
     541,   358,  -471,   -12,   509,   -12,    68,   486,   329,   347,
    -471,   489,   -12,   562,   498,   173,   185,   479,   571,   503,
     522,   488,   514,  -471,  -471,   -12,   497,   212,  -471,   521,
     291,   -12,   514,  -471,  -471,  -471,  -471,  -471,  -471,   524,
     562,  -471,  -471,  -471,  -471,   544,  -471,   238,   489,   -12,
     465,  -471,   368,   516,  -471,  -471,   534,  -471,  -471,   185,
     420,  -471,  -471,  -471,  -471,  -471,  -471,  -471,    11,   563,
     561,   555,   -12,   568,   -12,   195,  -471,  -471,   358,  -471,
    -471,   195,   594,   567,   769,   685,  -471,   185,   -12,   574,
     573,  -471,   575,   427,  -471,    11,    11,   -12,   572,  -471,
      11,   514,   -12,  -471,   594,   195,  -471,  -471,    97,    -2,
     570,    11,  -471,   484,   585,  -471,   587,  -471,    11,   286,
     591,    11,    11,  -471,   110,   195,  -471,   -12,  -471,    32,
     586,   361,    11,  -471,   350,  -471,   596,   514,  -471,  -471,
    -471,  -471,  -471,  -471,   547,   195,  -471,   597,  -471
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -471,  -471,   669,  -471,  -199,    -1,   -10,   662,   678,    -8,
    -471,  -471,  -471,  -205,  -471,  -168,  -471,   -35,   -33,   -24,
     -22,  -471,  -122,   583,   -37,  -471,  -471,   456,  -471,  -471,
      -4,   551,   434,  -471,    12,   451,  -471,  -471,   569,   444,
    -471,   319,  -471,  -471,  -273,  -471,  -142,   360,  -471,  -471,
    -471,   -69,  -471,  -471,  -471,  -471,  -471,  -471,  -471,   440,
    -471,   452,   680,  -471,    80,   364,   691,  -471,  -471,   548,
    -471,  -471,  -471,   357,   384,  -471,   356,  -471,   301,  -471,
    -471,   453,   -97,   196,   -17,  -442,  -471,  -471,  -462,  -471,
    -471,  -309,   192,  -391,  -471,  -471,   263,  -454,  -471,   239,
    -471,  -436,  -471,  -412,   178,  -470,  -404,  -471,   259,  -471,
    -471,  -471,  -471
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -265
static const yytype_int16 yytable[] =
{
       7,     8,   110,   161,    38,    93,   264,    94,    33,   201,
     116,   493,   510,   243,   449,   426,    30,    81,    34,    35,
     305,    97,   305,   506,   494,   111,   508,   490,   168,     9,
     286,   168,   177,   289,   348,   455,   450,   113,   115,   462,
      98,   246,   247,   248,    99,   132,   478,   480,   356,   128,
     231,   178,   273,   381,   490,   192,   449,   120,   121,   122,
     469,   223,   155,   226,     6,   474,   322,   530,   210,   100,
     416,   540,   169,   114,   170,   169,   572,   542,   164,  -110,
     146,   563,   412,    29,   147,   183,   184,   185,   156,   412,
     526,   413,   199,    98,   202,   550,   524,   152,   532,   574,
     171,   566,   172,   173,   367,   175,   579,   581,    96,   158,
     584,   302,   303,   568,   588,   159,   569,   594,   160,   570,
     571,   586,   188,   315,   316,   317,   318,   319,    67,    98,
     278,   556,    11,  -135,    12,  -135,   234,   541,    32,  -195,
     201,   597,   259,   116,    98,   206,   207,   208,   209,   323,
     118,   576,   473,    98,   220,   111,  -195,   253,   123,   254,
     364,   126,  -195,  -195,  -195,  -195,  -195,  -195,  -195,    32,
     430,   431,   408,    31,   436,    32,   567,   325,   568,   327,
     328,   569,   596,   129,   570,   571,   249,   368,    98,   585,
     188,   568,   235,   131,   569,   236,   436,   570,   571,   258,
      36,   261,    37,     1,     2,   265,   266,   267,   268,   269,
     270,   292,   144,   436,   437,   438,   439,   440,   441,   442,
     443,   282,   294,  -193,   295,   202,   154,    98,   279,   280,
     285,   238,   287,   288,   239,   290,   437,   438,   439,   440,
     441,   442,   443,   157,   514,   444,   195,   196,    37,  -262,
     320,  -230,   321,   437,   438,   439,   440,   441,   442,   443,
     186,   162,   378,   166,     6,   187,   167,   444,  -230,   347,
      37,  -230,   350,   174,   103,   104,  -230,   191,   334,   148,
     149,   150,   151,   176,   444,   193,   358,    37,   505,   212,
     213,    32,   436,  -230,     6,   187,   222,    83,    84,    85,
      86,    87,    88,    89,   292,   515,   516,    32,   281,   205,
      68,    69,    70,    71,   224,    72,    73,    74,    75,    76,
     311,   312,   221,   517,   227,   358,   297,   298,   299,   228,
     436,   229,   437,   438,   439,   440,   441,   442,   443,     6,
     580,   230,   358,   152,    77,    93,   375,    94,   436,   466,
     133,   134,   135,   136,   137,   138,   470,   377,   472,   382,
     383,   384,   237,   444,   241,   482,    37,  -264,   406,   436,
     437,   438,   439,   440,   441,   442,   443,    78,   503,   242,
      79,   244,   345,   346,   509,   250,   251,   186,   437,   438,
     439,   440,   441,   442,   443,   351,   352,   353,   354,   355,
     252,   444,   523,   432,   477,   255,   427,   256,   257,   437,
     438,   439,   440,   441,   442,   443,   263,   262,   271,   444,
     275,   436,    37,   272,   274,   537,   231,   539,   436,   592,
     277,   568,   305,   451,   569,   276,   300,   570,   571,   323,
     444,   551,   568,   525,   481,   569,   590,   330,   570,   571,
     560,   309,   259,   329,   331,   564,   332,   333,   335,   337,
     518,   437,   438,   439,   440,   441,   442,   443,   437,   438,
     439,   440,   441,   442,   443,   336,   349,   362,   363,   365,
     587,   522,   338,   371,   376,   436,   379,   380,   409,   411,
     421,   415,   444,   417,   418,   531,   419,   546,   518,   444,
     420,    15,   555,    -5,    -5,    16,    -5,    -5,    -5,    -5,
      -5,    -5,    -5,    -5,    -5,    -5,    -5,   533,    -5,    -5,
     425,   423,    -5,   424,   428,   437,   438,   439,   440,   441,
     442,   443,    68,    69,    70,    71,    -9,    72,    73,    74,
      75,    76,   429,   434,   557,   559,   452,   454,   436,   562,
     456,    17,    18,   459,   460,   463,   444,    19,    20,   575,
     557,   467,   471,   476,    37,   495,    77,   557,   557,    21,
     583,   557,   436,   500,   492,    -5,   -16,  -259,  -259,  -259,
    -259,   591,  -259,  -259,  -259,  -259,  -259,   497,   437,   438,
     439,   440,   441,   442,   443,   499,   501,   507,   513,   373,
     511,   527,    79,   483,   484,   485,   440,   486,   487,   488,
     529,  -259,   437,   438,   439,   440,   441,   442,   443,   444,
       1,     2,   595,    39,    40,    41,    42,    43,    44,    45,
      46,    47,    48,    49,   385,    50,    51,   534,   535,    52,
     536,   538,   543,   444,  -259,   545,   445,  -259,   552,   553,
     561,   554,   103,   104,   386,   573,   387,   388,   389,   390,
     391,   392,   577,   578,   393,   394,   395,   396,   397,    32,
     582,   589,   593,   598,    10,    83,    84,    85,    86,    87,
      88,    89,   398,   399,    54,   385,    14,   165,   284,   211,
     304,   514,    53,   291,   194,   301,   414,   366,   357,   400,
     374,   313,    62,   401,   402,   386,   372,   387,   388,   389,
     390,   391,   392,    64,   314,   393,   394,   395,   396,   397,
     102,   103,   104,   370,   240,   407,   435,   549,   326,   512,
     547,   496,   565,   398,   399,   504,     0,     0,    32,     0,
     103,   104,   200,     0,    83,    84,    85,    86,    87,    88,
      89,     0,   515,   516,   401,   402,     0,    32,     0,     0,
     103,   104,     0,    83,    84,    85,    86,    87,    88,    89,
       0,     0,     0,     0,     0,     0,     0,    32,    39,    40,
      41,    42,    43,    83,    84,    85,    86,    87,    88,    89,
      50,    51,   103,   104,    52,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   103,   104,     0,     0,    32,
     214,   215,   338,     0,     0,    83,    84,    85,    86,    87,
      88,    89,    32,    82,     0,     0,     0,     0,    83,    84,
      85,    86,    87,    88,    89,     0,   103,     0,     0,     0,
      32,     0,     0,     0,     0,     0,    83,    84,    85,    86,
      87,    88,    89,    32,    82,     0,     0,     0,     0,    83,
      84,    85,    86,    87,    88,    89,     0,     0,     0,     0,
       0,   405,     0,     0,     0,     0,     0,    83,    84,    85,
      86,    87,    88,    89
};

static const yytype_int16 yycheck[] =
{
       1,     2,    39,   100,    21,    29,   205,    29,    18,   131,
      43,   465,   482,   181,   426,   406,    17,    25,    19,    20,
      17,    31,    17,   477,   466,    36,   480,   463,    37,     0,
     235,    37,    30,   238,   307,   439,   427,    41,    42,   443,
      74,   183,   184,   185,    56,    53,   458,   459,   321,    50,
      84,    49,   220,   362,   490,   124,   468,    45,    46,    47,
     451,   158,    95,   160,    53,   456,   271,   509,   137,    81,
     379,   525,    81,    84,    83,    81,    78,   531,   102,    76,
      60,   551,    84,    40,    64,   120,   121,   122,    96,    84,
     502,    86,   127,    74,   131,   537,   500,    78,   510,   561,
     110,   555,   112,   113,     1,   115,   568,   569,    75,    75,
     572,   253,   254,    81,    82,    81,    84,   587,    84,    87,
      88,   575,   123,   265,   266,   267,   268,   269,    76,    74,
     227,   543,    73,    77,    75,    79,    81,   528,    53,    36,
     262,   595,    86,   176,    74,   133,   134,   135,   136,    81,
      84,   563,    84,    74,   155,    36,    53,   192,    84,   194,
      81,    84,    59,    60,    61,    62,    63,    64,    65,    53,
      54,    55,   371,    51,     1,    53,    79,   274,    81,   276,
     277,    84,   594,    55,    87,    88,   187,    84,    74,    79,
     191,    81,    78,    78,    84,    81,     1,    87,    88,   200,
      73,   202,    75,     3,     4,   206,   207,   208,   209,   210,
     211,   244,    73,     1,    41,    42,    43,    44,    45,    46,
      47,   231,    73,    53,    75,   262,    80,    74,   229,   230,
     234,    78,   236,   237,    81,   239,    41,    42,    43,    44,
      45,    46,    47,    74,     6,    72,    38,    39,    75,    76,
      73,    56,    75,    41,    42,    43,    44,    45,    46,    47,
      49,    76,   359,    80,    53,    54,    80,    72,    56,   306,
      75,    76,   309,    66,    36,    37,    81,    84,   288,    59,
      60,    61,    62,    77,    72,    84,   323,    75,    76,    59,
      60,    53,     1,    81,    53,    54,    53,    59,    60,    61,
      62,    63,    64,    65,   337,    67,    68,    53,    54,    76,
       6,     7,     8,     9,    56,    11,    12,    13,    14,    15,
      54,    55,    76,    85,    82,   362,   246,   247,   248,    73,
       1,    80,    41,    42,    43,    44,    45,    46,    47,    53,
      54,    80,   379,    78,    40,   369,   347,   369,     1,   446,
      11,    12,    13,    14,    15,    16,   453,   358,   455,   363,
     364,   365,    85,    72,    85,   462,    75,    76,   369,     1,
      41,    42,    43,    44,    45,    46,    47,    73,   475,    77,
      76,    74,   302,   303,   481,    85,    77,    49,    41,    42,
      43,    44,    45,    46,    47,   315,   316,   317,   318,   319,
      85,    72,   499,   413,    75,    85,   407,    77,    75,    41,
      42,    43,    44,    45,    46,    47,    79,    77,    77,    72,
      77,     1,    75,    79,    76,   522,    84,   524,     1,    79,
      85,    81,    17,   434,    84,    82,    85,    87,    88,    81,
      72,   538,    81,    75,   461,    84,    85,    85,    87,    88,
     547,    86,    86,    82,    82,   552,    79,    82,    79,    77,
     497,    41,    42,    43,    44,    45,    46,    47,    41,    42,
      43,    44,    45,    46,    47,    82,    76,    81,    81,    81,
     577,   498,    40,    76,    76,     1,    77,    82,    84,    73,
      55,    85,    72,    82,    82,    75,    82,   534,   535,    72,
      82,     1,    75,     3,     4,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    14,    15,    16,   518,    18,    19,
      77,    85,    22,    85,    85,    41,    42,    43,    44,    45,
      46,    47,     6,     7,     8,     9,    81,    11,    12,    13,
      14,    15,    73,    36,   545,   546,    81,    84,     1,   550,
      53,    51,    52,    75,    55,    75,    72,    57,    58,    75,
     561,    20,    53,    77,    75,    86,    40,   568,   569,    69,
     571,   572,     1,    85,    76,    75,    76,     6,     7,     8,
       9,   582,    11,    12,    13,    14,    15,    84,    41,    42,
      43,    44,    45,    46,    47,    73,    82,    76,    54,    73,
      76,    85,    76,    41,    42,    43,    44,    45,    46,    47,
      76,    40,    41,    42,    43,    44,    45,    46,    47,    72,
       3,     4,    75,     6,     7,     8,     9,    10,    11,    12,
      13,    14,    15,    16,     1,    18,    19,    74,    77,    22,
      85,    73,    48,    72,    73,    78,    75,    76,    74,    76,
      78,    76,    36,    37,    21,    85,    23,    24,    25,    26,
      27,    28,    77,    76,    31,    32,    33,    34,    35,    53,
      79,    85,    76,    76,     5,    59,    60,    61,    62,    63,
      64,    65,    49,    50,    22,     1,     8,   104,   232,   138,
     256,     6,    75,   242,   125,   251,   377,   337,    82,    66,
     343,   261,    22,    70,    71,    21,   342,    23,    24,    25,
      26,    27,    28,    22,   262,    31,    32,    33,    34,    35,
      16,    36,    37,   339,   176,   369,   425,   535,   275,   490,
     534,   468,   554,    49,    50,   476,    -1,    -1,    53,    -1,
      36,    37,    18,    -1,    59,    60,    61,    62,    63,    64,
      65,    -1,    67,    68,    70,    71,    -1,    53,    -1,    -1,
      36,    37,    -1,    59,    60,    61,    62,    63,    64,    65,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    53,     6,     7,
       8,     9,    10,    59,    60,    61,    62,    63,    64,    65,
      18,    19,    36,    37,    22,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    36,    37,    -1,    -1,    53,
      54,    55,    40,    -1,    -1,    59,    60,    61,    62,    63,
      64,    65,    53,    36,    -1,    -1,    -1,    -1,    59,    60,
      61,    62,    63,    64,    65,    -1,    36,    -1,    -1,    -1,
      53,    -1,    -1,    -1,    -1,    -1,    59,    60,    61,    62,
      63,    64,    65,    53,    36,    -1,    -1,    -1,    -1,    59,
      60,    61,    62,    63,    64,    65,    -1,    -1,    -1,    -1,
      -1,    53,    -1,    -1,    -1,    -1,    -1,    59,    60,    61,
      62,    63,    64,    65
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     3,     4,    90,    91,    96,    53,    94,    94,     0,
      91,    73,    75,    97,    97,     1,     5,    51,    52,    57,
      58,    69,    92,    98,    99,   100,   164,   200,   201,    40,
      94,    51,    53,    95,    94,    94,    73,    75,   173,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      18,    19,    22,    75,    96,   117,   118,   134,   137,   138,
     139,   141,   151,   152,   155,   156,   157,    76,     6,     7,
       8,     9,    11,    12,    13,    14,    15,    40,    73,    76,
     162,    98,    36,    59,    60,    61,    62,    63,    64,    65,
      95,   105,   107,   108,   109,   165,    75,    95,    74,    56,
      81,   171,    16,    36,    37,   108,   109,   110,   111,   112,
     113,    36,   119,   119,    84,   119,   107,   158,    84,   123,
     123,   123,   123,    84,   127,   140,    84,   120,    94,    55,
     159,    78,    98,    11,    12,    13,    14,    15,    16,   142,
     143,   144,   145,   146,    73,    93,    60,    64,    59,    60,
      61,    62,    78,   104,    80,   107,    98,    74,    75,    81,
      84,   171,    76,   174,   108,   112,    80,    80,    37,    81,
      83,    95,    95,    95,    66,    95,    77,    30,    49,   124,
     129,    94,   106,   106,   106,   106,    49,    54,    94,   126,
     128,    84,   140,    84,   127,    38,    39,   121,   122,   106,
      18,   111,   113,   149,   150,    76,   123,   123,   123,   123,
     140,   120,    59,    60,    54,    55,   101,   102,   103,   113,
      94,    76,    53,   171,    56,   170,   171,    82,    73,    80,
      80,    84,   115,   116,    81,    78,    81,    85,    78,    81,
     158,    85,    77,   104,    74,   135,   135,   135,   135,    94,
      85,    77,    85,   106,   106,    85,    77,    75,    94,    86,
     148,    94,    77,    79,    93,    94,    94,    94,    94,    94,
      94,    77,    79,   104,    76,    77,    82,    85,   171,    94,
      94,    54,    95,   114,   116,   119,   102,   119,   119,   102,
     119,   124,   107,   136,    73,    75,   153,   153,   153,   153,
      85,   128,   135,   135,   121,    17,   130,   132,   133,    86,
     147,    54,    55,   148,   150,   135,   135,   135,   135,   135,
      73,    75,   102,    81,   182,   171,   170,   171,   171,    82,
      85,    82,    79,    82,    95,    79,    82,    77,    40,   151,
     154,   155,   160,   161,   163,   153,   153,   113,   133,    76,
     113,   153,   153,   153,   153,   153,   133,    82,   113,   172,
     175,   180,    81,    81,    81,    81,   136,     1,    84,   166,
     163,    76,   154,    73,   162,    94,    76,    94,   171,    77,
      82,   180,   119,   119,   119,     1,    21,    23,    24,    25,
      26,    27,    28,    31,    32,    33,    34,    35,    49,    50,
      66,    70,    71,   167,   168,    53,    94,   165,    93,    84,
     131,    73,    84,    86,   130,    85,   180,    82,    82,    82,
      82,    55,   125,    85,    85,    77,   182,    94,    85,    73,
      54,    55,    95,   169,    36,   167,     1,    41,    42,    43,
      44,    45,    46,    47,    72,    75,   173,   185,   190,   192,
     182,    94,    81,   195,    84,   195,    53,   196,   197,    75,
      55,   189,   195,    75,   186,   192,   171,    20,   184,   182,
     171,    53,   171,    84,   182,   198,    77,    75,   192,   187,
     192,   173,   171,    41,    42,    43,    45,    46,    47,   188,
     190,   191,    76,   186,   174,    86,   185,    84,   183,    73,
      85,    82,   194,   171,   197,    76,   186,    76,   186,   171,
     194,    76,   188,    54,     6,    67,    68,    85,   113,   176,
     179,   181,   173,   171,   195,    75,   192,    85,   199,    76,
     174,    75,   192,    94,    74,    77,    85,   171,    73,   171,
     186,   182,   186,    48,   193,    78,   113,   172,   178,   181,
     174,   171,    74,    76,    76,    75,   192,    94,   177,    94,
     171,    78,    94,   194,   171,   193,   186,    79,    81,    84,
      87,    88,    78,    85,   177,    75,   192,    77,    76,   177,
      54,   177,    79,    94,   177,    79,   186,   171,    82,    85,
      85,    94,    79,    76,   194,    75,   192,   186,    76
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
    { (yyval.modlist) = new AstChildren<Module>(lineno, (yyvsp[(1) - (2)].module), (yyvsp[(2) - (2)].modlist)); }
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

/* Line 1455 of yacc.c  */
#line 236 "xi-grammar.y"
    { if((yyvsp[(3) - (5)].conslist)) (yyvsp[(3) - (5)].conslist)->recurse<int&>((yyvsp[(1) - (5)].intval), &Construct::setExtern); (yyval.construct) = (yyvsp[(3) - (5)].conslist); }
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
    { (yyval.type) = new ConstType((yyvsp[(2) - (2)].type)); }
    break;

  case 74:

/* Line 1455 of yacc.c  */
#line 363 "xi-grammar.y"
    { (yyval.type) = new ConstType((yyvsp[(1) - (2)].type)); }
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
                    (yyval.mbrlist) = new AstChildren<Member>(connectEntries);
		  } else {
		    (yyval.mbrlist) = 0; 
                  }
		}
    break;

  case 152:

/* Line 1455 of yacc.c  */
#line 617 "xi-grammar.y"
    { (yyval.mbrlist) = new AstChildren<Member>(-1, (yyvsp[(1) - (2)].member), (yyvsp[(2) - (2)].mbrlist)); }
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

/* Line 1455 of yacc.c  */
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

/* Line 1455 of yacc.c  */
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

/* Line 1455 of yacc.c  */
#line 763 "xi-grammar.y"
    { (yyval.accelBlock) = new AccelBlock(lineno, new XStr((yyvsp[(3) - (5)].strval))); }
    break;

  case 192:

/* Line 1455 of yacc.c  */
#line 765 "xi-grammar.y"
    { (yyval.accelBlock) = new AccelBlock(lineno, NULL); }
    break;

  case 193:

/* Line 1455 of yacc.c  */
#line 769 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("void"); }
    break;

  case 194:

/* Line 1455 of yacc.c  */
#line 771 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 195:

/* Line 1455 of yacc.c  */
#line 775 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 196:

/* Line 1455 of yacc.c  */
#line 777 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(2) - (3)].intval); }
    break;

  case 197:

/* Line 1455 of yacc.c  */
#line 779 "xi-grammar.y"
    { printf("Invalid entry method attribute list\n"); YYABORT; }
    break;

  case 198:

/* Line 1455 of yacc.c  */
#line 783 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (1)].intval); }
    break;

  case 199:

/* Line 1455 of yacc.c  */
#line 785 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (3)].intval) | (yyvsp[(3) - (3)].intval); }
    break;

  case 200:

/* Line 1455 of yacc.c  */
#line 789 "xi-grammar.y"
    { (yyval.intval) = STHREADED; }
    break;

  case 201:

/* Line 1455 of yacc.c  */
#line 791 "xi-grammar.y"
    { (yyval.intval) = SSYNC; }
    break;

  case 202:

/* Line 1455 of yacc.c  */
#line 793 "xi-grammar.y"
    { (yyval.intval) = SIGET; }
    break;

  case 203:

/* Line 1455 of yacc.c  */
#line 795 "xi-grammar.y"
    { (yyval.intval) = SLOCKED; }
    break;

  case 204:

/* Line 1455 of yacc.c  */
#line 797 "xi-grammar.y"
    { (yyval.intval) = SCREATEHERE; }
    break;

  case 205:

/* Line 1455 of yacc.c  */
#line 799 "xi-grammar.y"
    { (yyval.intval) = SCREATEHOME; }
    break;

  case 206:

/* Line 1455 of yacc.c  */
#line 801 "xi-grammar.y"
    { (yyval.intval) = SNOKEEP; }
    break;

  case 207:

/* Line 1455 of yacc.c  */
#line 803 "xi-grammar.y"
    { (yyval.intval) = SNOTRACE; }
    break;

  case 208:

/* Line 1455 of yacc.c  */
#line 805 "xi-grammar.y"
    { (yyval.intval) = SAPPWORK; }
    break;

  case 209:

/* Line 1455 of yacc.c  */
#line 807 "xi-grammar.y"
    { (yyval.intval) = SIMMEDIATE; }
    break;

  case 210:

/* Line 1455 of yacc.c  */
#line 809 "xi-grammar.y"
    { (yyval.intval) = SSKIPSCHED; }
    break;

  case 211:

/* Line 1455 of yacc.c  */
#line 811 "xi-grammar.y"
    { (yyval.intval) = SINLINE; }
    break;

  case 212:

/* Line 1455 of yacc.c  */
#line 813 "xi-grammar.y"
    { (yyval.intval) = SLOCAL; }
    break;

  case 213:

/* Line 1455 of yacc.c  */
#line 815 "xi-grammar.y"
    { (yyval.intval) = SPYTHON; }
    break;

  case 214:

/* Line 1455 of yacc.c  */
#line 817 "xi-grammar.y"
    { (yyval.intval) = SMEM; }
    break;

  case 215:

/* Line 1455 of yacc.c  */
#line 819 "xi-grammar.y"
    { (yyval.intval) = SREDUCE; }
    break;

  case 216:

/* Line 1455 of yacc.c  */
#line 821 "xi-grammar.y"
    { printf("Invalid entry method attribute: %s\n", yylval); YYABORT; }
    break;

  case 217:

/* Line 1455 of yacc.c  */
#line 825 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 218:

/* Line 1455 of yacc.c  */
#line 827 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 219:

/* Line 1455 of yacc.c  */
#line 829 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 220:

/* Line 1455 of yacc.c  */
#line 833 "xi-grammar.y"
    { (yyval.strval) = ""; }
    break;

  case 221:

/* Line 1455 of yacc.c  */
#line 835 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 222:

/* Line 1455 of yacc.c  */
#line 837 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[(1) - (3)].strval))+strlen((yyvsp[(3) - (3)].strval))+3];
			sprintf(tmp,"%s, %s", (yyvsp[(1) - (3)].strval), (yyvsp[(3) - (3)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 223:

/* Line 1455 of yacc.c  */
#line 845 "xi-grammar.y"
    { (yyval.strval) = ""; }
    break;

  case 224:

/* Line 1455 of yacc.c  */
#line 847 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 225:

/* Line 1455 of yacc.c  */
#line 849 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s[%s]%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 226:

/* Line 1455 of yacc.c  */
#line 855 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s{%s}%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 227:

/* Line 1455 of yacc.c  */
#line 861 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s(%s)%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 228:

/* Line 1455 of yacc.c  */
#line 867 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(2) - (4)].strval))+strlen((yyvsp[(4) - (4)].strval))+3];
			sprintf(tmp,"(%s)%s", (yyvsp[(2) - (4)].strval), (yyvsp[(4) - (4)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 229:

/* Line 1455 of yacc.c  */
#line 875 "xi-grammar.y"
    {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			(yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].type),(yyvsp[(2) - (3)].strval));
		}
    break;

  case 230:

/* Line 1455 of yacc.c  */
#line 882 "xi-grammar.y"
    { 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			(yyval.intval) = 0;
		}
    break;

  case 231:

/* Line 1455 of yacc.c  */
#line 890 "xi-grammar.y"
    { 
			in_braces=0;
			(yyval.intval) = 0;
		}
    break;

  case 232:

/* Line 1455 of yacc.c  */
#line 897 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (1)].type));}
    break;

  case 233:

/* Line 1455 of yacc.c  */
#line 899 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].type),(yyvsp[(2) - (3)].strval)); (yyval.pname)->setConditional((yyvsp[(3) - (3)].intval)); }
    break;

  case 234:

/* Line 1455 of yacc.c  */
#line 901 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (4)].type),(yyvsp[(2) - (4)].strval),0,(yyvsp[(4) - (4)].val));}
    break;

  case 235:

/* Line 1455 of yacc.c  */
#line 903 "xi-grammar.y"
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].pname)->getType(), (yyvsp[(1) - (3)].pname)->getName() ,(yyvsp[(2) - (3)].strval));
		}
    break;

  case 236:

/* Line 1455 of yacc.c  */
#line 909 "xi-grammar.y"
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READONLY; }
    break;

  case 237:

/* Line 1455 of yacc.c  */
#line 910 "xi-grammar.y"
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READWRITE; }
    break;

  case 238:

/* Line 1455 of yacc.c  */
#line 911 "xi-grammar.y"
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_WRITEONLY; }
    break;

  case 239:

/* Line 1455 of yacc.c  */
#line 914 "xi-grammar.y"
    { (yyval.xstrptr) = new XStr((yyvsp[(1) - (1)].strval)); }
    break;

  case 240:

/* Line 1455 of yacc.c  */
#line 915 "xi-grammar.y"
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "->" << (yyvsp[(4) - (4)].strval); }
    break;

  case 241:

/* Line 1455 of yacc.c  */
#line 916 "xi-grammar.y"
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[(1) - (3)].xstrptr)) << "." << (yyvsp[(3) - (3)].strval); }
    break;

  case 242:

/* Line 1455 of yacc.c  */
#line 918 "xi-grammar.y"
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "[" << *((yyvsp[(3) - (4)].xstrptr)) << "]";
                  delete (yyvsp[(1) - (4)].xstrptr);
                  delete (yyvsp[(3) - (4)].xstrptr);
                }
    break;

  case 243:

/* Line 1455 of yacc.c  */
#line 925 "xi-grammar.y"
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "[" << (yyvsp[(3) - (4)].strval) << "]";
                  delete (yyvsp[(1) - (4)].xstrptr);
                }
    break;

  case 244:

/* Line 1455 of yacc.c  */
#line 931 "xi-grammar.y"
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "(" << *((yyvsp[(3) - (4)].xstrptr)) << ")";
                  delete (yyvsp[(1) - (4)].xstrptr);
                  delete (yyvsp[(3) - (4)].xstrptr);
                }
    break;

  case 245:

/* Line 1455 of yacc.c  */
#line 940 "xi-grammar.y"
    {
                  in_bracket = 0;
                  (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].pname)->getType(), (yyvsp[(1) - (3)].pname)->getName(), (yyvsp[(2) - (3)].strval));
                }
    break;

  case 246:

/* Line 1455 of yacc.c  */
#line 947 "xi-grammar.y"
    {
                  (yyval.pname) = new Parameter(lineno, (yyvsp[(3) - (7)].type), (yyvsp[(4) - (7)].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[(6) - (7)].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[(1) - (7)].intval));
                }
    break;

  case 247:

/* Line 1455 of yacc.c  */
#line 953 "xi-grammar.y"
    {
		  (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (5)].type), (yyvsp[(2) - (5)].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[(4) - (5)].xstrptr));
                  (yyval.pname)->setAccelBufferType(Parameter::ACCEL_BUFFER_TYPE_READWRITE);
		}
    break;

  case 248:

/* Line 1455 of yacc.c  */
#line 959 "xi-grammar.y"
    {
                  (yyval.pname) = (yyvsp[(3) - (6)].pname);
                  (yyval.pname)->setAccelInstName((yyvsp[(5) - (6)].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[(1) - (6)].intval));
		}
    break;

  case 249:

/* Line 1455 of yacc.c  */
#line 967 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (1)].pname)); }
    break;

  case 250:

/* Line 1455 of yacc.c  */
#line 969 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (3)].pname),(yyvsp[(3) - (3)].plist)); }
    break;

  case 251:

/* Line 1455 of yacc.c  */
#line 973 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (1)].pname)); }
    break;

  case 252:

/* Line 1455 of yacc.c  */
#line 975 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (3)].pname),(yyvsp[(3) - (3)].plist)); }
    break;

  case 253:

/* Line 1455 of yacc.c  */
#line 979 "xi-grammar.y"
    { (yyval.plist) = (yyvsp[(2) - (3)].plist); }
    break;

  case 254:

/* Line 1455 of yacc.c  */
#line 981 "xi-grammar.y"
    { (yyval.plist) = new ParamList(new Parameter(0, new BuiltinType("void"))); }
    break;

  case 255:

/* Line 1455 of yacc.c  */
#line 985 "xi-grammar.y"
    { (yyval.plist) = (yyvsp[(2) - (3)].plist); }
    break;

  case 256:

/* Line 1455 of yacc.c  */
#line 987 "xi-grammar.y"
    { (yyval.plist) = 0; }
    break;

  case 257:

/* Line 1455 of yacc.c  */
#line 991 "xi-grammar.y"
    { (yyval.val) = 0; }
    break;

  case 258:

/* Line 1455 of yacc.c  */
#line 993 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(3) - (3)].strval)); }
    break;

  case 259:

/* Line 1455 of yacc.c  */
#line 997 "xi-grammar.y"
    { (yyval.sc) = 0; }
    break;

  case 260:

/* Line 1455 of yacc.c  */
#line 999 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSDAGENTRY, (yyvsp[(1) - (1)].sc)); }
    break;

  case 261:

/* Line 1455 of yacc.c  */
#line 1001 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSDAGENTRY, (yyvsp[(2) - (3)].sc)); }
    break;

  case 262:

/* Line 1455 of yacc.c  */
#line 1005 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSLIST, (yyvsp[(1) - (1)].sc)); }
    break;

  case 263:

/* Line 1455 of yacc.c  */
#line 1007 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSLIST, (yyvsp[(1) - (2)].sc), (yyvsp[(2) - (2)].sc));  }
    break;

  case 264:

/* Line 1455 of yacc.c  */
#line 1011 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SOLIST, (yyvsp[(1) - (1)].sc)); }
    break;

  case 265:

/* Line 1455 of yacc.c  */
#line 1013 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SOLIST, (yyvsp[(1) - (2)].sc), (yyvsp[(2) - (2)].sc)); }
    break;

  case 266:

/* Line 1455 of yacc.c  */
#line 1017 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SCASELIST, (yyvsp[(1) - (1)].when)); }
    break;

  case 267:

/* Line 1455 of yacc.c  */
#line 1019 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SCASELIST, (yyvsp[(1) - (2)].when), (yyvsp[(2) - (2)].sc)); }
    break;

  case 268:

/* Line 1455 of yacc.c  */
#line 1021 "xi-grammar.y"
    { yyerror("Case blocks in SDAG can only contain when clauses."); YYABORT; }
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
    { (yyval.when) = new WhenConstruct((yyvsp[(2) - (4)].entrylist), 0); }
    break;

  case 272:

/* Line 1455 of yacc.c  */
#line 1033 "xi-grammar.y"
    { (yyval.when) = new WhenConstruct((yyvsp[(2) - (3)].entrylist), (yyvsp[(3) - (3)].sc)); }
    break;

  case 273:

/* Line 1455 of yacc.c  */
#line 1035 "xi-grammar.y"
    { (yyval.when) = new WhenConstruct((yyvsp[(2) - (5)].entrylist), (yyvsp[(4) - (5)].sc)); }
    break;

  case 274:

/* Line 1455 of yacc.c  */
#line 1039 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 275:

/* Line 1455 of yacc.c  */
#line 1041 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 276:

/* Line 1455 of yacc.c  */
#line 1043 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 277:

/* Line 1455 of yacc.c  */
#line 1045 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 278:

/* Line 1455 of yacc.c  */
#line 1047 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 279:

/* Line 1455 of yacc.c  */
#line 1049 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 280:

/* Line 1455 of yacc.c  */
#line 1053 "xi-grammar.y"
    { (yyval.sc) = new AtomicConstruct((yyvsp[(4) - (5)].strval), (yyvsp[(2) - (5)].strval)); }
    break;

  case 281:

/* Line 1455 of yacc.c  */
#line 1055 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SOVERLAP,0, 0,0,0,0,(yyvsp[(3) - (4)].sc), 0); }
    break;

  case 282:

/* Line 1455 of yacc.c  */
#line 1057 "xi-grammar.y"
    { (yyval.sc) = (yyvsp[(1) - (1)].when); }
    break;

  case 283:

/* Line 1455 of yacc.c  */
#line 1059 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SCASE, 0, 0, 0, 0, 0, (yyvsp[(3) - (4)].sc), 0); }
    break;

  case 284:

/* Line 1455 of yacc.c  */
#line 1061 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFOR, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (11)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(5) - (11)].strval)),
		             new SdagConstruct(SINT_EXPR, (yyvsp[(7) - (11)].strval)), 0, (yyvsp[(10) - (11)].sc), 0); }
    break;

  case 285:

/* Line 1455 of yacc.c  */
#line 1064 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFOR, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (9)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(5) - (9)].strval)), 
		         new SdagConstruct(SINT_EXPR, (yyvsp[(7) - (9)].strval)), 0, (yyvsp[(9) - (9)].sc), 0); }
    break;

  case 286:

/* Line 1455 of yacc.c  */
#line 1067 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFORALL, 0, new SdagConstruct(SIDENT, (yyvsp[(3) - (12)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(6) - (12)].strval)), 
		             new SdagConstruct(SINT_EXPR, (yyvsp[(8) - (12)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(10) - (12)].strval)), (yyvsp[(12) - (12)].sc), 0); }
    break;

  case 287:

/* Line 1455 of yacc.c  */
#line 1070 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFORALL, 0, new SdagConstruct(SIDENT, (yyvsp[(3) - (14)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(6) - (14)].strval)), 
		                 new SdagConstruct(SINT_EXPR, (yyvsp[(8) - (14)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(10) - (14)].strval)), (yyvsp[(13) - (14)].sc), 0); }
    break;

  case 288:

/* Line 1455 of yacc.c  */
#line 1073 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (6)].strval)), (yyvsp[(6) - (6)].sc),0,0,(yyvsp[(5) - (6)].sc),0); }
    break;

  case 289:

/* Line 1455 of yacc.c  */
#line 1075 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (8)].strval)), (yyvsp[(8) - (8)].sc),0,0,(yyvsp[(6) - (8)].sc),0); }
    break;

  case 290:

/* Line 1455 of yacc.c  */
#line 1077 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SWHILE, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (5)].strval)), 0,0,0,(yyvsp[(5) - (5)].sc),0); }
    break;

  case 291:

/* Line 1455 of yacc.c  */
#line 1079 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SWHILE, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (7)].strval)), 0,0,0,(yyvsp[(6) - (7)].sc),0); }
    break;

  case 292:

/* Line 1455 of yacc.c  */
#line 1081 "xi-grammar.y"
    { (yyval.sc) = new AtomicConstruct((yyvsp[(2) - (3)].strval), NULL); }
    break;

  case 293:

/* Line 1455 of yacc.c  */
#line 1083 "xi-grammar.y"
    { printf("Unknown SDAG construct or malformed entry method definition.\n"
                         "You may have forgotten to terminate an entry method definition with a"
                         " semicolon or forgotten to mark a block of sequential SDAG code as 'atomic'\n"); YYABORT; }
    break;

  case 294:

/* Line 1455 of yacc.c  */
#line 1089 "xi-grammar.y"
    { (yyval.sc) = 0; }
    break;

  case 295:

/* Line 1455 of yacc.c  */
#line 1091 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SELSE, 0,0,0,0,0, (yyvsp[(2) - (2)].sc),0); }
    break;

  case 296:

/* Line 1455 of yacc.c  */
#line 1093 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SELSE, 0,0,0,0,0, (yyvsp[(3) - (4)].sc),0); }
    break;

  case 297:

/* Line 1455 of yacc.c  */
#line 1097 "xi-grammar.y"
    { in_int_expr = 0; (yyval.intval) = 0; }
    break;

  case 298:

/* Line 1455 of yacc.c  */
#line 1101 "xi-grammar.y"
    { in_int_expr = 1; (yyval.intval) = 0; }
    break;

  case 299:

/* Line 1455 of yacc.c  */
#line 1105 "xi-grammar.y"
    { (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[(1) - (2)].strval), (yyvsp[(2) - (2)].plist), 0, 0, 0); }
    break;

  case 300:

/* Line 1455 of yacc.c  */
#line 1107 "xi-grammar.y"
    { (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[(1) - (5)].strval), (yyvsp[(5) - (5)].plist), 0, 0, (yyvsp[(3) - (5)].strval)); }
    break;

  case 301:

/* Line 1455 of yacc.c  */
#line 1111 "xi-grammar.y"
    { (yyval.entrylist) = new EntryList((yyvsp[(1) - (1)].entry)); }
    break;

  case 302:

/* Line 1455 of yacc.c  */
#line 1113 "xi-grammar.y"
    { (yyval.entrylist) = new EntryList((yyvsp[(1) - (3)].entry),(yyvsp[(3) - (3)].entrylist)); }
    break;

  case 303:

/* Line 1455 of yacc.c  */
#line 1117 "xi-grammar.y"
    { in_bracket=1; }
    break;

  case 304:

/* Line 1455 of yacc.c  */
#line 1120 "xi-grammar.y"
    { in_bracket=0; }
    break;

  case 305:

/* Line 1455 of yacc.c  */
#line 1124 "xi-grammar.y"
    { if (!macroDefined((yyvsp[(2) - (2)].strval), 1)) in_comment = 1; }
    break;

  case 306:

/* Line 1455 of yacc.c  */
#line 1128 "xi-grammar.y"
    { if (!macroDefined((yyvsp[(2) - (2)].strval), 0)) in_comment = 1; }
    break;



/* Line 1455 of yacc.c  */
#line 4438 "y.tab.c"
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
#line 1131 "xi-grammar.y"

void yyerror(const char *mesg)
{
    std::cerr << cur_file<<":"<<lineno<<": Charmxi syntax error> "
	      << mesg << std::endl;
}

