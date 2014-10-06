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
void ReservedWord(int token);
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
#line 24 "xi-grammar.y"
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
#line 301 "y.tab.c"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 314 "y.tab.c"

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
#define YYFINAL  55
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   1293

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  89
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  113
/* YYNRULES -- Number of rules.  */
#define YYNRULES  351
/* YYNRULES -- Number of states.  */
#define YYNSTATES  644

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
     299,   302,   305,   307,   309,   313,   314,   317,   322,   328,
     329,   331,   332,   336,   338,   342,   344,   346,   347,   351,
     353,   357,   358,   360,   362,   363,   367,   369,   373,   375,
     377,   378,   380,   381,   384,   390,   392,   395,   399,   406,
     407,   410,   412,   416,   422,   428,   434,   440,   445,   449,
     456,   463,   469,   475,   481,   487,   493,   498,   506,   507,
     510,   511,   514,   517,   521,   524,   528,   530,   534,   539,
     542,   545,   548,   551,   554,   556,   561,   562,   565,   567,
     569,   571,   573,   576,   579,   582,   586,   593,   603,   607,
     614,   618,   625,   635,   645,   647,   651,   653,   656,   659,
     661,   664,   666,   668,   670,   672,   674,   676,   678,   680,
     682,   684,   686,   688,   696,   702,   715,   721,   724,   726,
     728,   729,   733,   735,   737,   741,   743,   745,   747,   749,
     751,   753,   755,   757,   759,   761,   763,   765,   767,   770,
     772,   774,   776,   778,   780,   782,   783,   785,   789,   790,
     792,   798,   804,   810,   815,   819,   821,   823,   825,   829,
     834,   838,   840,   842,   844,   846,   851,   855,   860,   865,
     870,   874,   882,   888,   895,   897,   901,   903,   907,   911,
     914,   918,   921,   922,   926,   927,   929,   933,   935,   938,
     940,   943,   945,   948,   950,   952,   953,   958,   962,   968,
     970,   972,   974,   976,   978,   980,   986,   991,   993,   998,
    1010,  1020,  1033,  1048,  1055,  1064,  1070,  1078,  1082,  1084,
    1085,  1088,  1093,  1095,  1097,  1100,  1106,  1108,  1112,  1114,
    1116,  1119
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
      37,    -1,   112,    83,    -1,   112,    -1,   171,    -1,   198,
     114,   199,    -1,    -1,   115,   116,    -1,     6,   113,    95,
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
     421,   425,   427,   431,   435,   440,   441,   445,   449,   454,
     455,   460,   461,   471,   473,   477,   479,   484,   485,   489,
     491,   496,   497,   501,   506,   507,   511,   513,   517,   519,
     524,   525,   529,   530,   533,   537,   539,   543,   545,   550,
     551,   555,   557,   561,   563,   567,   571,   575,   581,   585,
     587,   591,   593,   597,   601,   605,   609,   611,   616,   617,
     622,   623,   625,   629,   631,   633,   637,   639,   643,   647,
     649,   651,   653,   655,   659,   661,   666,   673,   677,   679,
     681,   682,   684,   686,   688,   692,   694,   696,   702,   705,
     710,   712,   714,   720,   728,   730,   733,   737,   740,   744,
     746,   751,   755,   757,   759,   761,   763,   765,   767,   769,
     771,   773,   775,   778,   788,   803,   819,   821,   825,   827,
     832,   833,   835,   839,   841,   845,   847,   849,   851,   853,
     855,   857,   859,   861,   863,   865,   867,   869,   871,   873,
     875,   877,   881,   883,   885,   890,   891,   893,   902,   903,
     905,   911,   917,   923,   931,   938,   946,   953,   955,   957,
     959,   966,   967,   968,   971,   972,   973,   974,   981,   987,
     996,  1003,  1009,  1015,  1023,  1025,  1029,  1031,  1035,  1037,
    1041,  1043,  1048,  1049,  1054,  1055,  1057,  1061,  1063,  1067,
    1069,  1073,  1075,  1077,  1081,  1084,  1087,  1089,  1091,  1095,
    1097,  1099,  1101,  1103,  1105,  1109,  1111,  1113,  1115,  1117,
    1120,  1123,  1126,  1129,  1131,  1133,  1135,  1137,  1139,  1146,
    1147,  1149,  1153,  1157,  1161,  1163,  1167,  1169,  1173,  1176,
    1180,  1184
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

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
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

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -514
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

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -310
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
#line 156 "xi-grammar.y"
    { (yyval.modlist) = (yyvsp[(1) - (1)].modlist); modlist = (yyvsp[(1) - (1)].modlist); }
    break;

  case 3:
#line 160 "xi-grammar.y"
    { 
		  (yyval.modlist) = 0; 
		}
    break;

  case 4:
#line 164 "xi-grammar.y"
    { (yyval.modlist) = new AstChildren<Module>(lineno, (yyvsp[(1) - (2)].module), (yyvsp[(2) - (2)].modlist)); }
    break;

  case 5:
#line 168 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 6:
#line 170 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 7:
#line 174 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 8:
#line 176 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 9:
#line 181 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 10:
#line 182 "xi-grammar.y"
    { ReservedWord(MODULE); YYABORT; }
    break;

  case 11:
#line 183 "xi-grammar.y"
    { ReservedWord(MAINMODULE); YYABORT; }
    break;

  case 12:
#line 184 "xi-grammar.y"
    { ReservedWord(EXTERN); YYABORT; }
    break;

  case 13:
#line 186 "xi-grammar.y"
    { ReservedWord(INITCALL); YYABORT; }
    break;

  case 14:
#line 187 "xi-grammar.y"
    { ReservedWord(INITNODE); YYABORT; }
    break;

  case 15:
#line 188 "xi-grammar.y"
    { ReservedWord(INITPROC); YYABORT; }
    break;

  case 16:
#line 190 "xi-grammar.y"
    { ReservedWord(CHARE); }
    break;

  case 17:
#line 191 "xi-grammar.y"
    { ReservedWord(MAINCHARE); }
    break;

  case 18:
#line 192 "xi-grammar.y"
    { ReservedWord(GROUP); }
    break;

  case 19:
#line 193 "xi-grammar.y"
    { ReservedWord(NODEGROUP); }
    break;

  case 20:
#line 194 "xi-grammar.y"
    { ReservedWord(ARRAY); }
    break;

  case 21:
#line 198 "xi-grammar.y"
    { ReservedWord(INCLUDE); YYABORT; }
    break;

  case 22:
#line 199 "xi-grammar.y"
    { ReservedWord(STACKSIZE); YYABORT; }
    break;

  case 23:
#line 200 "xi-grammar.y"
    { ReservedWord(THREADED); YYABORT; }
    break;

  case 24:
#line 201 "xi-grammar.y"
    { ReservedWord(TEMPLATE); YYABORT; }
    break;

  case 25:
#line 202 "xi-grammar.y"
    { ReservedWord(SYNC); YYABORT; }
    break;

  case 26:
#line 203 "xi-grammar.y"
    { ReservedWord(IGET); YYABORT; }
    break;

  case 27:
#line 204 "xi-grammar.y"
    { ReservedWord(EXCLUSIVE); YYABORT; }
    break;

  case 28:
#line 205 "xi-grammar.y"
    { ReservedWord(IMMEDIATE); YYABORT; }
    break;

  case 29:
#line 206 "xi-grammar.y"
    { ReservedWord(SKIPSCHED); YYABORT; }
    break;

  case 30:
#line 207 "xi-grammar.y"
    { ReservedWord(INLINE); YYABORT; }
    break;

  case 31:
#line 208 "xi-grammar.y"
    { ReservedWord(VIRTUAL); YYABORT; }
    break;

  case 32:
#line 209 "xi-grammar.y"
    { ReservedWord(MIGRATABLE); YYABORT; }
    break;

  case 33:
#line 210 "xi-grammar.y"
    { ReservedWord(CREATEHERE); YYABORT; }
    break;

  case 34:
#line 211 "xi-grammar.y"
    { ReservedWord(CREATEHOME); YYABORT; }
    break;

  case 35:
#line 212 "xi-grammar.y"
    { ReservedWord(NOKEEP); YYABORT; }
    break;

  case 36:
#line 213 "xi-grammar.y"
    { ReservedWord(NOTRACE); YYABORT; }
    break;

  case 37:
#line 214 "xi-grammar.y"
    { ReservedWord(APPWORK); YYABORT; }
    break;

  case 38:
#line 217 "xi-grammar.y"
    { ReservedWord(PACKED); YYABORT; }
    break;

  case 39:
#line 218 "xi-grammar.y"
    { ReservedWord(VARSIZE); YYABORT; }
    break;

  case 40:
#line 219 "xi-grammar.y"
    { ReservedWord(ENTRY); YYABORT; }
    break;

  case 41:
#line 220 "xi-grammar.y"
    { ReservedWord(FOR); YYABORT; }
    break;

  case 42:
#line 221 "xi-grammar.y"
    { ReservedWord(FORALL); YYABORT; }
    break;

  case 43:
#line 222 "xi-grammar.y"
    { ReservedWord(WHILE); YYABORT; }
    break;

  case 44:
#line 223 "xi-grammar.y"
    { ReservedWord(WHEN); YYABORT; }
    break;

  case 45:
#line 224 "xi-grammar.y"
    { ReservedWord(OVERLAP); YYABORT; }
    break;

  case 46:
#line 225 "xi-grammar.y"
    { ReservedWord(ATOMIC); YYABORT; }
    break;

  case 47:
#line 226 "xi-grammar.y"
    { ReservedWord(IF); YYABORT; }
    break;

  case 48:
#line 227 "xi-grammar.y"
    { ReservedWord(ELSE); YYABORT; }
    break;

  case 49:
#line 229 "xi-grammar.y"
    { ReservedWord(LOCAL); YYABORT; }
    break;

  case 50:
#line 231 "xi-grammar.y"
    { ReservedWord(USING); YYABORT; }
    break;

  case 51:
#line 232 "xi-grammar.y"
    { ReservedWord(ACCEL); YYABORT; }
    break;

  case 52:
#line 235 "xi-grammar.y"
    { ReservedWord(ACCELBLOCK); YYABORT; }
    break;

  case 53:
#line 236 "xi-grammar.y"
    { ReservedWord(MEMCRITICAL); YYABORT; }
    break;

  case 54:
#line 237 "xi-grammar.y"
    { ReservedWord(REDUCTIONTARGET); YYABORT; }
    break;

  case 55:
#line 238 "xi-grammar.y"
    { ReservedWord(CASE); YYABORT; }
    break;

  case 56:
#line 242 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 57:
#line 244 "xi-grammar.y"
    {
		  char *tmp = new char[strlen((yyvsp[(1) - (4)].strval))+strlen((yyvsp[(4) - (4)].strval))+3];
		  sprintf(tmp,"%s::%s", (yyvsp[(1) - (4)].strval), (yyvsp[(4) - (4)].strval));
		  (yyval.strval) = tmp;
		}
    break;

  case 58:
#line 252 "xi-grammar.y"
    { 
		    (yyval.module) = new Module(lineno, (yyvsp[(2) - (3)].strval), (yyvsp[(3) - (3)].conslist)); 
		}
    break;

  case 59:
#line 256 "xi-grammar.y"
    {  
		    (yyval.module) = new Module(lineno, (yyvsp[(2) - (3)].strval), (yyvsp[(3) - (3)].conslist)); 
		    (yyval.module)->setMain();
		}
    break;

  case 60:
#line 263 "xi-grammar.y"
    { (yyval.conslist) = 0; }
    break;

  case 61:
#line 265 "xi-grammar.y"
    { (yyval.conslist) = (yyvsp[(2) - (4)].conslist); }
    break;

  case 62:
#line 269 "xi-grammar.y"
    { (yyval.conslist) = 0; }
    break;

  case 63:
#line 271 "xi-grammar.y"
    { (yyval.conslist) = new ConstructList(lineno, (yyvsp[(1) - (2)].construct), (yyvsp[(2) - (2)].conslist)); }
    break;

  case 64:
#line 275 "xi-grammar.y"
    { (yyval.construct) = new UsingScope((yyvsp[(3) - (3)].strval), false); }
    break;

  case 65:
#line 277 "xi-grammar.y"
    { (yyval.construct) = new UsingScope((yyvsp[(2) - (2)].strval), true); }
    break;

  case 66:
#line 279 "xi-grammar.y"
    { (yyvsp[(2) - (2)].member)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].member); }
    break;

  case 67:
#line 281 "xi-grammar.y"
    { (yyvsp[(2) - (2)].message)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].message); }
    break;

  case 68:
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
#line 295 "xi-grammar.y"
    { if((yyvsp[(3) - (5)].conslist)) (yyvsp[(3) - (5)].conslist)->recurse<int&>((yyvsp[(1) - (5)].intval), &Construct::setExtern); (yyval.construct) = (yyvsp[(3) - (5)].conslist); }
    break;

  case 70:
#line 297 "xi-grammar.y"
    { (yyval.construct) = new Scope((yyvsp[(2) - (5)].strval), (yyvsp[(4) - (5)].conslist)); }
    break;

  case 71:
#line 299 "xi-grammar.y"
    { (yyval.construct) = (yyvsp[(1) - (2)].construct); }
    break;

  case 72:
#line 301 "xi-grammar.y"
    { yyerror("The preceding construct must be semicolon terminated"); YYABORT; }
    break;

  case 73:
#line 303 "xi-grammar.y"
    { (yyvsp[(2) - (2)].module)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].module); }
    break;

  case 74:
#line 305 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 75:
#line 307 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 76:
#line 309 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 77:
#line 311 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 78:
#line 313 "xi-grammar.y"
    { (yyvsp[(2) - (2)].templat)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].templat); }
    break;

  case 79:
#line 315 "xi-grammar.y"
    { (yyval.construct) = NULL; }
    break;

  case 80:
#line 317 "xi-grammar.y"
    { (yyval.construct) = NULL; }
    break;

  case 81:
#line 319 "xi-grammar.y"
    { (yyval.construct) = (yyvsp[(1) - (1)].accelBlock); }
    break;

  case 82:
#line 321 "xi-grammar.y"
    { printf("Invalid construct\n"); YYABORT; }
    break;

  case 83:
#line 325 "xi-grammar.y"
    { (yyval.tparam) = new TParamType((yyvsp[(1) - (1)].type)); }
    break;

  case 84:
#line 327 "xi-grammar.y"
    { (yyval.tparam) = new TParamVal((yyvsp[(1) - (1)].strval)); }
    break;

  case 85:
#line 329 "xi-grammar.y"
    { (yyval.tparam) = new TParamVal((yyvsp[(1) - (1)].strval)); }
    break;

  case 86:
#line 333 "xi-grammar.y"
    { (yyval.tparlist) = new TParamList((yyvsp[(1) - (1)].tparam)); }
    break;

  case 87:
#line 335 "xi-grammar.y"
    { (yyval.tparlist) = new TParamList((yyvsp[(1) - (3)].tparam), (yyvsp[(3) - (3)].tparlist)); }
    break;

  case 88:
#line 339 "xi-grammar.y"
    { (yyval.tparlist) = 0; }
    break;

  case 89:
#line 341 "xi-grammar.y"
    { (yyval.tparlist) = (yyvsp[(1) - (1)].tparlist); }
    break;

  case 90:
#line 345 "xi-grammar.y"
    { (yyval.tparlist) = 0; }
    break;

  case 91:
#line 347 "xi-grammar.y"
    { (yyval.tparlist) = (yyvsp[(2) - (3)].tparlist); }
    break;

  case 92:
#line 351 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("int"); }
    break;

  case 93:
#line 353 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long"); }
    break;

  case 94:
#line 355 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("short"); }
    break;

  case 95:
#line 357 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("char"); }
    break;

  case 96:
#line 359 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned int"); }
    break;

  case 97:
#line 361 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long"); }
    break;

  case 98:
#line 363 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long"); }
    break;

  case 99:
#line 365 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long long"); }
    break;

  case 100:
#line 367 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned short"); }
    break;

  case 101:
#line 369 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned char"); }
    break;

  case 102:
#line 371 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long long"); }
    break;

  case 103:
#line 373 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("float"); }
    break;

  case 104:
#line 375 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("double"); }
    break;

  case 105:
#line 377 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long double"); }
    break;

  case 106:
#line 379 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("void"); }
    break;

  case 107:
#line 382 "xi-grammar.y"
    { (yyval.ntype) = new NamedType((yyvsp[(1) - (2)].strval),(yyvsp[(2) - (2)].tparlist)); }
    break;

  case 108:
#line 383 "xi-grammar.y"
    { 
                    const char* basename, *scope;
                    splitScopedName((yyvsp[(1) - (2)].strval), &scope, &basename);
                    (yyval.ntype) = new NamedType(basename, (yyvsp[(2) - (2)].tparlist), scope);
                }
    break;

  case 109:
#line 391 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 110:
#line 393 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ntype); }
    break;

  case 111:
#line 397 "xi-grammar.y"
    { (yyval.ptype) = new PtrType((yyvsp[(1) - (2)].type)); }
    break;

  case 112:
#line 401 "xi-grammar.y"
    { (yyvsp[(1) - (2)].ptype)->indirect(); (yyval.ptype) = (yyvsp[(1) - (2)].ptype); }
    break;

  case 113:
#line 403 "xi-grammar.y"
    { (yyvsp[(1) - (2)].ptype)->indirect(); (yyval.ptype) = (yyvsp[(1) - (2)].ptype); }
    break;

  case 114:
#line 407 "xi-grammar.y"
    { (yyval.ftype) = new FuncType((yyvsp[(1) - (8)].type), (yyvsp[(4) - (8)].strval), (yyvsp[(7) - (8)].plist)); }
    break;

  case 115:
#line 411 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 116:
#line 413 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 117:
#line 415 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 118:
#line 417 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ftype); }
    break;

  case 119:
#line 420 "xi-grammar.y"
    { (yyval.type) = new ConstType((yyvsp[(2) - (2)].type)); }
    break;

  case 120:
#line 422 "xi-grammar.y"
    { (yyval.type) = new ConstType((yyvsp[(1) - (2)].type)); }
    break;

  case 121:
#line 426 "xi-grammar.y"
    { (yyval.type) = new ReferenceType((yyvsp[(1) - (2)].type)); }
    break;

  case 122:
#line 428 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 123:
#line 432 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 124:
#line 436 "xi-grammar.y"
    { (yyval.val) = (yyvsp[(2) - (3)].val); }
    break;

  case 125:
#line 440 "xi-grammar.y"
    { (yyval.vallist) = 0; }
    break;

  case 126:
#line 442 "xi-grammar.y"
    { (yyval.vallist) = new ValueList((yyvsp[(1) - (2)].val), (yyvsp[(2) - (2)].vallist)); }
    break;

  case 127:
#line 446 "xi-grammar.y"
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[(2) - (4)].type), (yyvsp[(3) - (4)].strval), (yyvsp[(4) - (4)].vallist)); }
    break;

  case 128:
#line 450 "xi-grammar.y"
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[(3) - (5)].type), (yyvsp[(5) - (5)].strval), 0, 1); }
    break;

  case 129:
#line 454 "xi-grammar.y"
    { (yyval.intval) = 0;}
    break;

  case 130:
#line 456 "xi-grammar.y"
    { (yyval.intval) = 0;}
    break;

  case 131:
#line 460 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 132:
#line 462 "xi-grammar.y"
    { 
		  /*
		  printf("Warning: Message attributes are being phased out.\n");
		  printf("Warning: Please remove them from interface files.\n");
		  */
		  (yyval.intval) = (yyvsp[(2) - (3)].intval); 
		}
    break;

  case 133:
#line 472 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (1)].intval); }
    break;

  case 134:
#line 474 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (3)].intval) | (yyvsp[(3) - (3)].intval); }
    break;

  case 135:
#line 478 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 136:
#line 480 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 137:
#line 484 "xi-grammar.y"
    { (yyval.cattr) = 0; }
    break;

  case 138:
#line 486 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(2) - (3)].cattr); }
    break;

  case 139:
#line 490 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (1)].cattr); }
    break;

  case 140:
#line 492 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (3)].cattr) | (yyvsp[(3) - (3)].cattr); }
    break;

  case 141:
#line 496 "xi-grammar.y"
    { python_doc = NULL; (yyval.intval) = 0; }
    break;

  case 142:
#line 498 "xi-grammar.y"
    { python_doc = (yyvsp[(1) - (1)].strval); (yyval.intval) = 0; }
    break;

  case 143:
#line 502 "xi-grammar.y"
    { (yyval.cattr) = Chare::CPYTHON; }
    break;

  case 144:
#line 506 "xi-grammar.y"
    { (yyval.cattr) = 0; }
    break;

  case 145:
#line 508 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(2) - (3)].cattr); }
    break;

  case 146:
#line 512 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (1)].cattr); }
    break;

  case 147:
#line 514 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (3)].cattr) | (yyvsp[(3) - (3)].cattr); }
    break;

  case 148:
#line 518 "xi-grammar.y"
    { (yyval.cattr) = Chare::CMIGRATABLE; }
    break;

  case 149:
#line 520 "xi-grammar.y"
    { (yyval.cattr) = Chare::CPYTHON; }
    break;

  case 150:
#line 524 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 151:
#line 526 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 152:
#line 529 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 153:
#line 531 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 154:
#line 534 "xi-grammar.y"
    { (yyval.mv) = new MsgVar((yyvsp[(2) - (5)].type), (yyvsp[(3) - (5)].strval), (yyvsp[(1) - (5)].intval), (yyvsp[(4) - (5)].intval)); }
    break;

  case 155:
#line 538 "xi-grammar.y"
    { (yyval.mvlist) = new MsgVarList((yyvsp[(1) - (1)].mv)); }
    break;

  case 156:
#line 540 "xi-grammar.y"
    { (yyval.mvlist) = new MsgVarList((yyvsp[(1) - (2)].mv), (yyvsp[(2) - (2)].mvlist)); }
    break;

  case 157:
#line 544 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, (yyvsp[(3) - (3)].ntype)); }
    break;

  case 158:
#line 546 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, (yyvsp[(3) - (6)].ntype), (yyvsp[(5) - (6)].mvlist)); }
    break;

  case 159:
#line 550 "xi-grammar.y"
    { (yyval.typelist) = 0; }
    break;

  case 160:
#line 552 "xi-grammar.y"
    { (yyval.typelist) = (yyvsp[(2) - (2)].typelist); }
    break;

  case 161:
#line 556 "xi-grammar.y"
    { (yyval.typelist) = new TypeList((yyvsp[(1) - (1)].ntype)); }
    break;

  case 162:
#line 558 "xi-grammar.y"
    { (yyval.typelist) = new TypeList((yyvsp[(1) - (3)].ntype), (yyvsp[(3) - (3)].typelist)); }
    break;

  case 163:
#line 562 "xi-grammar.y"
    { (yyval.chare) = new Chare(lineno, (yyvsp[(2) - (5)].cattr)|Chare::CCHARE, (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 164:
#line 564 "xi-grammar.y"
    { (yyval.chare) = new MainChare(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 165:
#line 568 "xi-grammar.y"
    { (yyval.chare) = new Group(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 166:
#line 572 "xi-grammar.y"
    { (yyval.chare) = new NodeGroup(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 167:
#line 576 "xi-grammar.y"
    {/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",(yyvsp[(2) - (4)].strval));
			(yyval.ntype) = new NamedType(buf); 
		}
    break;

  case 168:
#line 582 "xi-grammar.y"
    { (yyval.ntype) = new NamedType((yyvsp[(2) - (3)].strval)); }
    break;

  case 169:
#line 586 "xi-grammar.y"
    {  (yyval.chare) = new Array(lineno, (yyvsp[(2) - (6)].cattr), (yyvsp[(3) - (6)].ntype), (yyvsp[(4) - (6)].ntype), (yyvsp[(5) - (6)].typelist), (yyvsp[(6) - (6)].mbrlist)); }
    break;

  case 170:
#line 588 "xi-grammar.y"
    {  (yyval.chare) = new Array(lineno, (yyvsp[(3) - (6)].cattr), (yyvsp[(2) - (6)].ntype), (yyvsp[(4) - (6)].ntype), (yyvsp[(5) - (6)].typelist), (yyvsp[(6) - (6)].mbrlist)); }
    break;

  case 171:
#line 592 "xi-grammar.y"
    { (yyval.chare) = new Chare(lineno, (yyvsp[(2) - (5)].cattr)|Chare::CCHARE, new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist));}
    break;

  case 172:
#line 594 "xi-grammar.y"
    { (yyval.chare) = new MainChare(lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 173:
#line 598 "xi-grammar.y"
    { (yyval.chare) = new Group(lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 174:
#line 602 "xi-grammar.y"
    { (yyval.chare) = new NodeGroup( lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 175:
#line 606 "xi-grammar.y"
    { (yyval.chare) = new Array( lineno, 0, (yyvsp[(2) - (5)].ntype), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 176:
#line 610 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[(3) - (4)].strval))); }
    break;

  case 177:
#line 612 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[(3) - (7)].strval)), (yyvsp[(5) - (7)].mvlist)); }
    break;

  case 178:
#line 616 "xi-grammar.y"
    { (yyval.type) = 0; }
    break;

  case 179:
#line 618 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(2) - (2)].type); }
    break;

  case 180:
#line 622 "xi-grammar.y"
    { (yyval.strval) = 0; }
    break;

  case 181:
#line 624 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(2) - (2)].strval); }
    break;

  case 182:
#line 626 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(2) - (2)].strval); }
    break;

  case 183:
#line 630 "xi-grammar.y"
    { (yyval.tvar) = new TType(new NamedType((yyvsp[(2) - (3)].strval)), (yyvsp[(3) - (3)].type)); }
    break;

  case 184:
#line 632 "xi-grammar.y"
    { (yyval.tvar) = new TFunc((yyvsp[(1) - (2)].ftype), (yyvsp[(2) - (2)].strval)); }
    break;

  case 185:
#line 634 "xi-grammar.y"
    { (yyval.tvar) = new TName((yyvsp[(1) - (3)].type), (yyvsp[(2) - (3)].strval), (yyvsp[(3) - (3)].strval)); }
    break;

  case 186:
#line 638 "xi-grammar.y"
    { (yyval.tvarlist) = new TVarList((yyvsp[(1) - (1)].tvar)); }
    break;

  case 187:
#line 640 "xi-grammar.y"
    { (yyval.tvarlist) = new TVarList((yyvsp[(1) - (3)].tvar), (yyvsp[(3) - (3)].tvarlist)); }
    break;

  case 188:
#line 644 "xi-grammar.y"
    { (yyval.tvarlist) = (yyvsp[(3) - (4)].tvarlist); }
    break;

  case 189:
#line 648 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 190:
#line 650 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 191:
#line 652 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 192:
#line 654 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 193:
#line 656 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].message)); (yyvsp[(2) - (2)].message)->setTemplate((yyval.templat)); }
    break;

  case 194:
#line 660 "xi-grammar.y"
    { (yyval.mbrlist) = 0; }
    break;

  case 195:
#line 662 "xi-grammar.y"
    { (yyval.mbrlist) = (yyvsp[(2) - (4)].mbrlist); }
    break;

  case 196:
#line 666 "xi-grammar.y"
    { 
                  if (!connectEntries.empty()) {
                    (yyval.mbrlist) = new AstChildren<Member>(connectEntries);
		  } else {
		    (yyval.mbrlist) = 0; 
                  }
		}
    break;

  case 197:
#line 674 "xi-grammar.y"
    { (yyval.mbrlist) = new AstChildren<Member>(-1, (yyvsp[(1) - (2)].member), (yyvsp[(2) - (2)].mbrlist)); }
    break;

  case 198:
#line 678 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].readonly); }
    break;

  case 199:
#line 680 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].readonly); }
    break;

  case 201:
#line 683 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].member); }
    break;

  case 202:
#line 685 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(2) - (2)].pupable); }
    break;

  case 203:
#line 687 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(2) - (2)].includeFile); }
    break;

  case 204:
#line 689 "xi-grammar.y"
    { (yyval.member) = new ClassDeclaration(lineno,(yyvsp[(2) - (2)].strval)); }
    break;

  case 205:
#line 693 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 1); }
    break;

  case 206:
#line 695 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 1); }
    break;

  case 207:
#line 697 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[(3) - (9)].strval)) + '<' +
					    ((yyvsp[(5) - (9)].tparlist))->to_string() + '>').c_str()),
				    1);
		}
    break;

  case 208:
#line 703 "xi-grammar.y"
    { printf("Warning: deprecated use of initcall. Use initnode or initproc instead.\n"); 
		  (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 1); }
    break;

  case 209:
#line 706 "xi-grammar.y"
    { printf("Warning: deprecated use of initcall. Use initnode or initproc instead.\n");
		  (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 1); }
    break;

  case 210:
#line 711 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 0); }
    break;

  case 211:
#line 713 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 0); }
    break;

  case 212:
#line 715 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno,
				    strdup((std::string((yyvsp[(3) - (9)].strval)) + '<' +
					    ((yyvsp[(5) - (9)].tparlist))->to_string() + '>').c_str()),
				    0);
		}
    break;

  case 213:
#line 721 "xi-grammar.y"
    {
                  InitCall* rtn = new InitCall(lineno, (yyvsp[(6) - (9)].strval), 0);
                  rtn->setAccel();
                  (yyval.member) = rtn;
		}
    break;

  case 214:
#line 729 "xi-grammar.y"
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[(1) - (1)].ntype),0); }
    break;

  case 215:
#line 731 "xi-grammar.y"
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[(1) - (3)].ntype),(yyvsp[(3) - (3)].pupable)); }
    break;

  case 216:
#line 734 "xi-grammar.y"
    { (yyval.includeFile) = new IncludeFile(lineno,(yyvsp[(1) - (1)].strval)); }
    break;

  case 217:
#line 738 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (2)].member); }
    break;

  case 218:
#line 741 "xi-grammar.y"
    { yyerror("The preceding entry method declaration must be semicolon-terminated."); YYABORT; }
    break;

  case 219:
#line 745 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].entry); }
    break;

  case 220:
#line 747 "xi-grammar.y"
    {
                  (yyvsp[(2) - (2)].entry)->tspec = (yyvsp[(1) - (2)].tvarlist);
                  (yyval.member) = (yyvsp[(2) - (2)].entry);
                }
    break;

  case 221:
#line 752 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].member); }
    break;

  case 222:
#line 756 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 223:
#line 758 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 224:
#line 760 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 225:
#line 762 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 226:
#line 764 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 227:
#line 766 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 228:
#line 768 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 229:
#line 770 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 230:
#line 772 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 231:
#line 774 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 232:
#line 776 "xi-grammar.y"
    { (yyval.member) = 0; }
    break;

  case 233:
#line 779 "xi-grammar.y"
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

  case 234:
#line 789 "xi-grammar.y"
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

  case 235:
#line 804 "xi-grammar.y"
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

  case 236:
#line 820 "xi-grammar.y"
    { (yyval.accelBlock) = new AccelBlock(lineno, new XStr((yyvsp[(3) - (5)].strval))); }
    break;

  case 237:
#line 822 "xi-grammar.y"
    { (yyval.accelBlock) = new AccelBlock(lineno, NULL); }
    break;

  case 238:
#line 826 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("void"); }
    break;

  case 239:
#line 828 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 240:
#line 832 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 241:
#line 834 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(2) - (3)].intval); }
    break;

  case 242:
#line 836 "xi-grammar.y"
    { printf("Invalid entry method attribute list\n"); YYABORT; }
    break;

  case 243:
#line 840 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (1)].intval); }
    break;

  case 244:
#line 842 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (3)].intval) | (yyvsp[(3) - (3)].intval); }
    break;

  case 245:
#line 846 "xi-grammar.y"
    { (yyval.intval) = STHREADED; }
    break;

  case 246:
#line 848 "xi-grammar.y"
    { (yyval.intval) = SSYNC; }
    break;

  case 247:
#line 850 "xi-grammar.y"
    { (yyval.intval) = SIGET; }
    break;

  case 248:
#line 852 "xi-grammar.y"
    { (yyval.intval) = SLOCKED; }
    break;

  case 249:
#line 854 "xi-grammar.y"
    { (yyval.intval) = SCREATEHERE; }
    break;

  case 250:
#line 856 "xi-grammar.y"
    { (yyval.intval) = SCREATEHOME; }
    break;

  case 251:
#line 858 "xi-grammar.y"
    { (yyval.intval) = SNOKEEP; }
    break;

  case 252:
#line 860 "xi-grammar.y"
    { (yyval.intval) = SNOTRACE; }
    break;

  case 253:
#line 862 "xi-grammar.y"
    { (yyval.intval) = SAPPWORK; }
    break;

  case 254:
#line 864 "xi-grammar.y"
    { (yyval.intval) = SIMMEDIATE; }
    break;

  case 255:
#line 866 "xi-grammar.y"
    { (yyval.intval) = SSKIPSCHED; }
    break;

  case 256:
#line 868 "xi-grammar.y"
    { (yyval.intval) = SINLINE; }
    break;

  case 257:
#line 870 "xi-grammar.y"
    { (yyval.intval) = SLOCAL; }
    break;

  case 258:
#line 872 "xi-grammar.y"
    { (yyval.intval) = SPYTHON; }
    break;

  case 259:
#line 874 "xi-grammar.y"
    { (yyval.intval) = SMEM; }
    break;

  case 260:
#line 876 "xi-grammar.y"
    { (yyval.intval) = SREDUCE; }
    break;

  case 261:
#line 878 "xi-grammar.y"
    { printf("Invalid entry method attribute: %s\n", yylval); YYABORT; }
    break;

  case 262:
#line 882 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 263:
#line 884 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 264:
#line 886 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 265:
#line 890 "xi-grammar.y"
    { (yyval.strval) = ""; }
    break;

  case 266:
#line 892 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 267:
#line 894 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[(1) - (3)].strval))+strlen((yyvsp[(3) - (3)].strval))+3];
			sprintf(tmp,"%s, %s", (yyvsp[(1) - (3)].strval), (yyvsp[(3) - (3)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 268:
#line 902 "xi-grammar.y"
    { (yyval.strval) = ""; }
    break;

  case 269:
#line 904 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 270:
#line 906 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s[%s]%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 271:
#line 912 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s{%s}%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 272:
#line 918 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s(%s)%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 273:
#line 924 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(2) - (4)].strval))+strlen((yyvsp[(4) - (4)].strval))+3];
			sprintf(tmp,"(%s)%s", (yyvsp[(2) - (4)].strval), (yyvsp[(4) - (4)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 274:
#line 932 "xi-grammar.y"
    {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			(yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].type),(yyvsp[(2) - (3)].strval));
		}
    break;

  case 275:
#line 939 "xi-grammar.y"
    { 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			(yyval.intval) = 0;
		}
    break;

  case 276:
#line 947 "xi-grammar.y"
    { 
			in_braces=0;
			(yyval.intval) = 0;
		}
    break;

  case 277:
#line 954 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (1)].type));}
    break;

  case 278:
#line 956 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].type),(yyvsp[(2) - (3)].strval)); (yyval.pname)->setConditional((yyvsp[(3) - (3)].intval)); }
    break;

  case 279:
#line 958 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (4)].type),(yyvsp[(2) - (4)].strval),0,(yyvsp[(4) - (4)].val));}
    break;

  case 280:
#line 960 "xi-grammar.y"
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].pname)->getType(), (yyvsp[(1) - (3)].pname)->getName() ,(yyvsp[(2) - (3)].strval));
		}
    break;

  case 281:
#line 966 "xi-grammar.y"
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READONLY; }
    break;

  case 282:
#line 967 "xi-grammar.y"
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READWRITE; }
    break;

  case 283:
#line 968 "xi-grammar.y"
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_WRITEONLY; }
    break;

  case 284:
#line 971 "xi-grammar.y"
    { (yyval.xstrptr) = new XStr((yyvsp[(1) - (1)].strval)); }
    break;

  case 285:
#line 972 "xi-grammar.y"
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "->" << (yyvsp[(4) - (4)].strval); }
    break;

  case 286:
#line 973 "xi-grammar.y"
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[(1) - (3)].xstrptr)) << "." << (yyvsp[(3) - (3)].strval); }
    break;

  case 287:
#line 975 "xi-grammar.y"
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "[" << *((yyvsp[(3) - (4)].xstrptr)) << "]";
                  delete (yyvsp[(1) - (4)].xstrptr);
                  delete (yyvsp[(3) - (4)].xstrptr);
                }
    break;

  case 288:
#line 982 "xi-grammar.y"
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "[" << (yyvsp[(3) - (4)].strval) << "]";
                  delete (yyvsp[(1) - (4)].xstrptr);
                }
    break;

  case 289:
#line 988 "xi-grammar.y"
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "(" << *((yyvsp[(3) - (4)].xstrptr)) << ")";
                  delete (yyvsp[(1) - (4)].xstrptr);
                  delete (yyvsp[(3) - (4)].xstrptr);
                }
    break;

  case 290:
#line 997 "xi-grammar.y"
    {
                  in_bracket = 0;
                  (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].pname)->getType(), (yyvsp[(1) - (3)].pname)->getName(), (yyvsp[(2) - (3)].strval));
                }
    break;

  case 291:
#line 1004 "xi-grammar.y"
    {
                  (yyval.pname) = new Parameter(lineno, (yyvsp[(3) - (7)].type), (yyvsp[(4) - (7)].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[(6) - (7)].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[(1) - (7)].intval));
                }
    break;

  case 292:
#line 1010 "xi-grammar.y"
    {
		  (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (5)].type), (yyvsp[(2) - (5)].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[(4) - (5)].xstrptr));
                  (yyval.pname)->setAccelBufferType(Parameter::ACCEL_BUFFER_TYPE_READWRITE);
		}
    break;

  case 293:
#line 1016 "xi-grammar.y"
    {
                  (yyval.pname) = (yyvsp[(3) - (6)].pname);
                  (yyval.pname)->setAccelInstName((yyvsp[(5) - (6)].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[(1) - (6)].intval));
		}
    break;

  case 294:
#line 1024 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (1)].pname)); }
    break;

  case 295:
#line 1026 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (3)].pname),(yyvsp[(3) - (3)].plist)); }
    break;

  case 296:
#line 1030 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (1)].pname)); }
    break;

  case 297:
#line 1032 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (3)].pname),(yyvsp[(3) - (3)].plist)); }
    break;

  case 298:
#line 1036 "xi-grammar.y"
    { (yyval.plist) = (yyvsp[(2) - (3)].plist); }
    break;

  case 299:
#line 1038 "xi-grammar.y"
    { (yyval.plist) = new ParamList(new Parameter(0, new BuiltinType("void"))); }
    break;

  case 300:
#line 1042 "xi-grammar.y"
    { (yyval.plist) = (yyvsp[(2) - (3)].plist); }
    break;

  case 301:
#line 1044 "xi-grammar.y"
    { (yyval.plist) = 0; }
    break;

  case 302:
#line 1048 "xi-grammar.y"
    { (yyval.val) = 0; }
    break;

  case 303:
#line 1050 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(3) - (3)].strval)); }
    break;

  case 304:
#line 1054 "xi-grammar.y"
    { (yyval.sc) = 0; }
    break;

  case 305:
#line 1056 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSDAGENTRY, (yyvsp[(1) - (1)].sc)); }
    break;

  case 306:
#line 1058 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSDAGENTRY, (yyvsp[(2) - (3)].sc)); }
    break;

  case 307:
#line 1062 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSLIST, (yyvsp[(1) - (1)].sc)); }
    break;

  case 308:
#line 1064 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSLIST, (yyvsp[(1) - (2)].sc), (yyvsp[(2) - (2)].sc));  }
    break;

  case 309:
#line 1068 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SOLIST, (yyvsp[(1) - (1)].sc)); }
    break;

  case 310:
#line 1070 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SOLIST, (yyvsp[(1) - (2)].sc), (yyvsp[(2) - (2)].sc)); }
    break;

  case 311:
#line 1074 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SCASELIST, (yyvsp[(1) - (1)].when)); }
    break;

  case 312:
#line 1076 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SCASELIST, (yyvsp[(1) - (2)].when), (yyvsp[(2) - (2)].sc)); }
    break;

  case 313:
#line 1078 "xi-grammar.y"
    { yyerror("Case blocks in SDAG can only contain when clauses."); YYABORT; }
    break;

  case 314:
#line 1082 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 315:
#line 1084 "xi-grammar.y"
    { (yyval.strval) = 0; }
    break;

  case 316:
#line 1088 "xi-grammar.y"
    { (yyval.when) = new WhenConstruct((yyvsp[(2) - (4)].entrylist), 0); }
    break;

  case 317:
#line 1090 "xi-grammar.y"
    { (yyval.when) = new WhenConstruct((yyvsp[(2) - (3)].entrylist), (yyvsp[(3) - (3)].sc)); }
    break;

  case 318:
#line 1092 "xi-grammar.y"
    { (yyval.when) = new WhenConstruct((yyvsp[(2) - (5)].entrylist), (yyvsp[(4) - (5)].sc)); }
    break;

  case 319:
#line 1096 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 320:
#line 1098 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 321:
#line 1100 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 322:
#line 1102 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 323:
#line 1104 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 324:
#line 1106 "xi-grammar.y"
    { (yyval.when) = 0; }
    break;

  case 325:
#line 1110 "xi-grammar.y"
    { (yyval.sc) = new AtomicConstruct((yyvsp[(4) - (5)].strval), (yyvsp[(2) - (5)].strval)); }
    break;

  case 326:
#line 1112 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SOVERLAP,0, 0,0,0,0,(yyvsp[(3) - (4)].sc), 0); }
    break;

  case 327:
#line 1114 "xi-grammar.y"
    { (yyval.sc) = (yyvsp[(1) - (1)].when); }
    break;

  case 328:
#line 1116 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SCASE, 0, 0, 0, 0, 0, (yyvsp[(3) - (4)].sc), 0); }
    break;

  case 329:
#line 1118 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFOR, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (11)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(5) - (11)].strval)),
		             new SdagConstruct(SINT_EXPR, (yyvsp[(7) - (11)].strval)), 0, (yyvsp[(10) - (11)].sc), 0); }
    break;

  case 330:
#line 1121 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFOR, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (9)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(5) - (9)].strval)), 
		         new SdagConstruct(SINT_EXPR, (yyvsp[(7) - (9)].strval)), 0, (yyvsp[(9) - (9)].sc), 0); }
    break;

  case 331:
#line 1124 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFORALL, 0, new SdagConstruct(SIDENT, (yyvsp[(3) - (12)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(6) - (12)].strval)), 
		             new SdagConstruct(SINT_EXPR, (yyvsp[(8) - (12)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(10) - (12)].strval)), (yyvsp[(12) - (12)].sc), 0); }
    break;

  case 332:
#line 1127 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFORALL, 0, new SdagConstruct(SIDENT, (yyvsp[(3) - (14)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(6) - (14)].strval)), 
		                 new SdagConstruct(SINT_EXPR, (yyvsp[(8) - (14)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(10) - (14)].strval)), (yyvsp[(13) - (14)].sc), 0); }
    break;

  case 333:
#line 1130 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (6)].strval)), (yyvsp[(6) - (6)].sc),0,0,(yyvsp[(5) - (6)].sc),0); }
    break;

  case 334:
#line 1132 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (8)].strval)), (yyvsp[(8) - (8)].sc),0,0,(yyvsp[(6) - (8)].sc),0); }
    break;

  case 335:
#line 1134 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SWHILE, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (5)].strval)), 0,0,0,(yyvsp[(5) - (5)].sc),0); }
    break;

  case 336:
#line 1136 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SWHILE, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (7)].strval)), 0,0,0,(yyvsp[(6) - (7)].sc),0); }
    break;

  case 337:
#line 1138 "xi-grammar.y"
    { (yyval.sc) = new AtomicConstruct((yyvsp[(2) - (3)].strval), NULL); }
    break;

  case 338:
#line 1140 "xi-grammar.y"
    { printf("Unknown SDAG construct or malformed entry method definition.\n"
                         "You may have forgotten to terminate an entry method definition with a"
                         " semicolon or forgotten to mark a block of sequential SDAG code as 'atomic'\n"); YYABORT; }
    break;

  case 339:
#line 1146 "xi-grammar.y"
    { (yyval.sc) = 0; }
    break;

  case 340:
#line 1148 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SELSE, 0,0,0,0,0, (yyvsp[(2) - (2)].sc),0); }
    break;

  case 341:
#line 1150 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SELSE, 0,0,0,0,0, (yyvsp[(3) - (4)].sc),0); }
    break;

  case 342:
#line 1154 "xi-grammar.y"
    { in_int_expr = 0; (yyval.intval) = 0; }
    break;

  case 343:
#line 1158 "xi-grammar.y"
    { in_int_expr = 1; (yyval.intval) = 0; }
    break;

  case 344:
#line 1162 "xi-grammar.y"
    { (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[(1) - (2)].strval), (yyvsp[(2) - (2)].plist), 0, 0, 0); }
    break;

  case 345:
#line 1164 "xi-grammar.y"
    { (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[(1) - (5)].strval), (yyvsp[(5) - (5)].plist), 0, 0, (yyvsp[(3) - (5)].strval)); }
    break;

  case 346:
#line 1168 "xi-grammar.y"
    { (yyval.entrylist) = new EntryList((yyvsp[(1) - (1)].entry)); }
    break;

  case 347:
#line 1170 "xi-grammar.y"
    { (yyval.entrylist) = new EntryList((yyvsp[(1) - (3)].entry),(yyvsp[(3) - (3)].entrylist)); }
    break;

  case 348:
#line 1174 "xi-grammar.y"
    { in_bracket=1; }
    break;

  case 349:
#line 1177 "xi-grammar.y"
    { in_bracket=0; }
    break;

  case 350:
#line 1181 "xi-grammar.y"
    { if (!macroDefined((yyvsp[(2) - (2)].strval), 1)) in_comment = 1; }
    break;

  case 351:
#line 1185 "xi-grammar.y"
    { if (!macroDefined((yyvsp[(2) - (2)].strval), 0)) in_comment = 1; }
    break;


/* Line 1267 of yacc.c.  */
#line 4169 "y.tab.c"
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


#line 1188 "xi-grammar.y"

void yyerror(const char *mesg)
{
    std::cerr << cur_file<<":"<<lineno<<": Charmxi syntax error> "
	      << mesg << std::endl;
}

