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
     ACCELBLOCK = 326
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




/* Copy the first part of user declarations.  */
#line 3 "xi-grammar.y"

#include "xi-symbol.h"
#include "EToken.h"
extern int yylex (void) ;
extern unsigned char in_comment;
void yyerror(const char *);
extern unsigned int lineno;
extern int in_bracket,in_braces,in_int_expr;
extern TList<Entry *> *connectEntries;
ModuleList *modlist;
extern int macroDefined(char *str, int istrue);
extern char *python_doc;
void splitScopedName(char* name, char** scope, char** basename);


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
#line 18 "xi-grammar.y"
{
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
  char *strval;
  int intval;
  Chare::attrib_t cattr;
  SdagConstruct *sc;
  XStr* xstrptr;
  AccelBlock* accelBlock;
}
/* Line 187 of yacc.c.  */
#line 291 "y.tab.c"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 304 "y.tab.c"

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
#define YYLAST   701

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  88
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  110
/* YYNRULES -- Number of rules.  */
#define YYNRULES  277
/* YYNRULES -- Number of states.  */
#define YYNSTATES  574

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
      17,    19,    24,    28,    32,    34,    39,    40,    43,    49,
      55,    60,    64,    67,    70,    74,    77,    80,    83,    86,
      89,    91,    93,    95,    97,    99,   101,   103,   107,   108,
     110,   111,   115,   117,   119,   121,   123,   126,   129,   133,
     137,   140,   143,   146,   148,   150,   153,   155,   158,   161,
     163,   165,   168,   171,   174,   183,   185,   187,   189,   191,
     194,   197,   200,   202,   204,   206,   210,   211,   214,   219,
     225,   226,   228,   229,   233,   235,   239,   241,   243,   244,
     248,   250,   254,   255,   257,   259,   260,   264,   266,   270,
     272,   274,   275,   277,   278,   281,   287,   289,   292,   296,
     303,   304,   307,   309,   313,   319,   325,   331,   337,   342,
     346,   353,   360,   366,   372,   378,   384,   390,   395,   403,
     404,   407,   408,   411,   414,   418,   421,   425,   427,   431,
     436,   439,   442,   445,   448,   451,   453,   458,   459,   462,
     465,   468,   471,   474,   478,   482,   486,   490,   497,   501,
     508,   512,   519,   529,   531,   535,   537,   540,   542,   550,
     556,   569,   575,   578,   580,   582,   583,   587,   589,   593,
     595,   597,   599,   601,   603,   605,   607,   609,   611,   613,
     615,   617,   620,   622,   624,   626,   627,   629,   633,   634,
     636,   642,   648,   654,   659,   663,   665,   667,   669,   673,
     678,   682,   684,   686,   688,   690,   695,   699,   704,   709,
     714,   718,   726,   732,   739,   741,   745,   747,   751,   755,
     758,   762,   765,   766,   770,   771,   773,   777,   779,   782,
     784,   787,   788,   793,   795,   799,   801,   802,   809,   818,
     823,   827,   833,   838,   850,   860,   873,   888,   895,   904,
     910,   918,   922,   926,   927,   930,   935,   937,   941,   943,
     945,   948,   954,   956,   960,   962,   964,   967
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
      89,     0,    -1,    90,    -1,    -1,    95,    90,    -1,    -1,
       5,    -1,    -1,    72,    -1,    55,    -1,    55,    -1,    94,
      73,    73,    55,    -1,     3,    93,    96,    -1,     4,    93,
      96,    -1,    72,    -1,    74,    97,    75,    92,    -1,    -1,
      98,    97,    -1,    91,    74,    97,    75,    92,    -1,    53,
      93,    74,    97,    75,    -1,    54,    53,    94,    72,    -1,
      54,    94,    72,    -1,    91,    95,    -1,    91,   153,    -1,
      91,   132,    72,    -1,    91,   135,    -1,    91,   136,    -1,
      91,   137,    -1,    91,   139,    -1,    91,   150,    -1,   196,
      -1,   197,    -1,   160,    -1,   111,    -1,    56,    -1,    57,
      -1,    99,    -1,    99,    76,   100,    -1,    -1,   100,    -1,
      -1,    77,   101,    78,    -1,    61,    -1,    62,    -1,    63,
      -1,    64,    -1,    67,    61,    -1,    67,    62,    -1,    67,
      62,    61,    -1,    67,    62,    62,    -1,    67,    63,    -1,
      67,    64,    -1,    62,    62,    -1,    65,    -1,    66,    -1,
      62,    66,    -1,    35,    -1,    93,   102,    -1,    94,   102,
      -1,   103,    -1,   105,    -1,   106,    79,    -1,   107,    79,
      -1,   108,    79,    -1,   110,    80,    79,    93,    81,    80,
     176,    81,    -1,   106,    -1,   107,    -1,   108,    -1,   109,
      -1,    36,   110,    -1,   110,    36,    -1,   110,    82,    -1,
     110,    -1,    56,    -1,    94,    -1,    83,   112,    84,    -1,
      -1,   113,   114,    -1,     6,   111,    94,   114,    -1,     6,
      16,   106,    79,    93,    -1,    -1,    35,    -1,    -1,    83,
     119,    84,    -1,   120,    -1,   120,    76,   119,    -1,    37,
      -1,    38,    -1,    -1,    83,   122,    84,    -1,   127,    -1,
     127,    76,   122,    -1,    -1,    57,    -1,    51,    -1,    -1,
      83,   126,    84,    -1,   124,    -1,   124,    76,   126,    -1,
      30,    -1,    51,    -1,    -1,    17,    -1,    -1,    83,    84,
      -1,   128,   111,    93,   129,    72,    -1,   130,    -1,   130,
     131,    -1,    16,   118,   104,    -1,    16,   118,   104,    74,
     131,    75,    -1,    -1,    73,   134,    -1,   105,    -1,   105,
      76,   134,    -1,    11,   121,   104,   133,   151,    -1,    12,
     121,   104,   133,   151,    -1,    13,   121,   104,   133,   151,
      -1,    14,   121,   104,   133,   151,    -1,    83,    56,    93,
      84,    -1,    83,    93,    84,    -1,    15,   125,   138,   104,
     133,   151,    -1,    15,   138,   125,   104,   133,   151,    -1,
      11,   121,    93,   133,   151,    -1,    12,   121,    93,   133,
     151,    -1,    13,   121,    93,   133,   151,    -1,    14,   121,
      93,   133,   151,    -1,    15,   138,    93,   133,   151,    -1,
      16,   118,    93,    72,    -1,    16,   118,    93,    74,   131,
      75,    72,    -1,    -1,    85,   111,    -1,    -1,    85,    56,
      -1,    85,    57,    -1,    18,    93,   145,    -1,   109,   146,
      -1,   111,    93,   146,    -1,   147,    -1,   147,    76,   148,
      -1,    22,    77,   148,    78,    -1,   149,   140,    -1,   149,
     141,    -1,   149,   142,    -1,   149,   143,    -1,   149,   144,
      -1,    72,    -1,    74,   152,    75,    92,    -1,    -1,   158,
     152,    -1,   115,    72,    -1,   116,    72,    -1,   155,    72,
      -1,   154,    72,    -1,    10,   156,    72,    -1,    19,   157,
      72,    -1,    18,    93,    72,    -1,     8,   117,    94,    -1,
       8,   117,    94,    80,   117,    81,    -1,     7,   117,    94,
      -1,     7,   117,    94,    80,   117,    81,    -1,     9,   117,
      94,    -1,     9,   117,    94,    80,   117,    81,    -1,     9,
      83,    68,    84,   117,    94,    80,   117,    81,    -1,   105,
      -1,   105,    76,   156,    -1,    57,    -1,   159,    72,    -1,
     153,    -1,    39,   162,   161,    93,   178,   180,   181,    -1,
      39,   162,    93,   178,   181,    -1,    39,    83,    68,    84,
      35,    93,   178,   179,   169,   167,   170,    93,    -1,    71,
     169,   167,   170,    72,    -1,    71,    72,    -1,    35,    -1,
     107,    -1,    -1,    83,   163,    84,    -1,   164,    -1,   164,
      76,   163,    -1,    21,    -1,    23,    -1,    24,    -1,    25,
      -1,    31,    -1,    32,    -1,    33,    -1,    34,    -1,    26,
      -1,    27,    -1,    28,    -1,    52,    -1,    51,   123,    -1,
      57,    -1,    56,    -1,    94,    -1,    -1,    58,    -1,    58,
      76,   166,    -1,    -1,    58,    -1,    58,    83,   167,    84,
     167,    -1,    58,    74,   167,    75,   167,    -1,    58,    80,
     166,    81,   167,    -1,    80,   167,    81,   167,    -1,   111,
      93,    83,    -1,    74,    -1,    75,    -1,   111,    -1,   111,
      93,   128,    -1,   111,    93,    85,   165,    -1,   168,   167,
      84,    -1,     6,    -1,    69,    -1,    70,    -1,    93,    -1,
     173,    86,    78,    93,    -1,   173,    87,    93,    -1,   173,
      83,   173,    84,    -1,   173,    83,    56,    84,    -1,   173,
      80,   173,    81,    -1,   168,   167,    84,    -1,   172,    73,
     111,    93,    77,   173,    78,    -1,   111,    93,    77,   173,
      78,    -1,   172,    73,   174,    77,   173,    78,    -1,   171,
      -1,   171,    76,   176,    -1,   175,    -1,   175,    76,   177,
      -1,    80,   176,    81,    -1,    80,    81,    -1,    83,   177,
      84,    -1,    83,    84,    -1,    -1,    20,    85,    56,    -1,
      -1,   187,    -1,    74,   182,    75,    -1,   187,    -1,   187,
     182,    -1,   187,    -1,   187,   182,    -1,    -1,    50,    80,
     185,    81,    -1,    55,    -1,    55,    76,   185,    -1,    57,
      -1,    -1,    45,   186,   169,   167,   170,   184,    -1,    49,
      80,    55,   178,    81,   169,   167,    75,    -1,    43,   193,
      74,    75,    -1,    43,   193,   187,    -1,    43,   193,    74,
     182,    75,    -1,    44,    74,   183,    75,    -1,    40,   191,
     167,    72,   167,    72,   167,   190,    74,   182,    75,    -1,
      40,   191,   167,    72,   167,    72,   167,   190,   187,    -1,
      41,    83,    55,    84,   191,   167,    73,   167,    76,   167,
     190,   187,    -1,    41,    83,    55,    84,   191,   167,    73,
     167,    76,   167,   190,    74,   182,    75,    -1,    47,   191,
     167,   190,   187,   188,    -1,    47,   191,   167,   190,    74,
     182,    75,   188,    -1,    42,   191,   167,   190,   187,    -1,
      42,   191,   167,   190,    74,   182,    75,    -1,    46,   189,
      72,    -1,   169,   167,   170,    -1,    -1,    48,   187,    -1,
      48,    74,   182,    75,    -1,    55,    -1,    55,    76,   189,
      -1,    81,    -1,    80,    -1,    55,   178,    -1,    55,   194,
     167,   195,   178,    -1,   192,    -1,   192,    76,   193,    -1,
      83,    -1,    84,    -1,    59,    93,    -1,    60,    93,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   147,   147,   152,   155,   160,   161,   166,   167,   171,
     175,   177,   185,   189,   196,   198,   203,   204,   208,   210,
     212,   214,   216,   218,   220,   222,   224,   226,   228,   230,
     232,   234,   236,   240,   242,   244,   248,   250,   255,   256,
     261,   262,   266,   268,   270,   272,   274,   276,   278,   280,
     282,   284,   286,   288,   290,   292,   294,   298,   299,   306,
     308,   312,   316,   318,   322,   326,   328,   330,   332,   335,
     337,   341,   343,   347,   349,   353,   358,   359,   363,   367,
     372,   373,   378,   379,   389,   391,   395,   397,   402,   403,
     407,   409,   414,   415,   419,   424,   425,   429,   431,   435,
     437,   442,   443,   447,   448,   451,   455,   457,   461,   463,
     468,   469,   473,   475,   479,   481,   485,   489,   493,   499,
     503,   505,   509,   511,   515,   519,   523,   527,   529,   534,
     535,   540,   541,   543,   547,   549,   551,   555,   557,   561,
     565,   567,   569,   571,   573,   577,   579,   584,   602,   606,
     608,   610,   611,   613,   615,   617,   621,   623,   625,   628,
     633,   635,   637,   645,   647,   650,   654,   656,   660,   671,
     682,   700,   702,   706,   708,   713,   714,   718,   720,   724,
     726,   728,   730,   732,   734,   736,   738,   740,   742,   744,
     746,   748,   752,   754,   756,   761,   762,   764,   773,   774,
     776,   782,   788,   794,   802,   809,   817,   824,   826,   828,
     830,   837,   838,   839,   842,   843,   844,   845,   852,   858,
     867,   874,   880,   886,   894,   896,   900,   902,   906,   908,
     912,   914,   919,   920,   925,   926,   928,   932,   934,   938,
     940,   945,   946,   950,   952,   956,   959,   962,   967,   981,
     983,   985,   987,   989,   992,   995,   998,  1001,  1003,  1005,
    1007,  1009,  1011,  1018,  1019,  1021,  1024,  1026,  1030,  1034,
    1038,  1046,  1054,  1056,  1060,  1063,  1067,  1071
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
  "ACCELBLOCK", "';'", "':'", "'{'", "'}'", "','", "'<'", "'>'", "'*'",
  "'('", "')'", "'&'", "'['", "']'", "'='", "'-'", "'.'", "$accept",
  "File", "ModuleEList", "OptExtern", "OptSemiColon", "Name", "QualName",
  "Module", "ConstructEList", "ConstructList", "Construct", "TParam",
  "TParamList", "TParamEList", "OptTParams", "BuiltinType", "NamedType",
  "QualNamedType", "SimpleType", "OnePtrType", "PtrType", "FuncType",
  "BaseType", "Type", "ArrayDim", "Dim", "DimList", "Readonly",
  "ReadonlyMsg", "OptVoid", "MAttribs", "MAttribList", "MAttrib",
  "CAttribs", "CAttribList", "PythonOptions", "ArrayAttrib",
  "ArrayAttribs", "ArrayAttribList", "CAttrib", "OptConditional",
  "MsgArray", "Var", "VarList", "Message", "OptBaseList", "BaseList",
  "Chare", "Group", "NodeGroup", "ArrayIndexType", "Array", "TChare",
  "TGroup", "TNodeGroup", "TArray", "TMessage", "OptTypeInit",
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
     325,   326,    59,    58,   123,   125,    44,    60,    62,    42,
      40,    41,    38,    91,    93,    61,    45,    46
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    88,    89,    90,    90,    91,    91,    92,    92,    93,
      94,    94,    95,    95,    96,    96,    97,    97,    98,    98,
      98,    98,    98,    98,    98,    98,    98,    98,    98,    98,
      98,    98,    98,    99,    99,    99,   100,   100,   101,   101,
     102,   102,   103,   103,   103,   103,   103,   103,   103,   103,
     103,   103,   103,   103,   103,   103,   103,   104,   105,   106,
     106,   107,   108,   108,   109,   110,   110,   110,   110,   110,
     110,   111,   111,   112,   112,   113,   114,   114,   115,   116,
     117,   117,   118,   118,   119,   119,   120,   120,   121,   121,
     122,   122,   123,   123,   124,   125,   125,   126,   126,   127,
     127,   128,   128,   129,   129,   130,   131,   131,   132,   132,
     133,   133,   134,   134,   135,   135,   136,   137,   138,   138,
     139,   139,   140,   140,   141,   142,   143,   144,   144,   145,
     145,   146,   146,   146,   147,   147,   147,   148,   148,   149,
     150,   150,   150,   150,   150,   151,   151,   152,   152,   153,
     153,   153,   153,   153,   153,   153,   154,   154,   154,   154,
     155,   155,   155,   156,   156,   157,   158,   158,   159,   159,
     159,   160,   160,   161,   161,   162,   162,   163,   163,   164,
     164,   164,   164,   164,   164,   164,   164,   164,   164,   164,
     164,   164,   165,   165,   165,   166,   166,   166,   167,   167,
     167,   167,   167,   167,   168,   169,   170,   171,   171,   171,
     171,   172,   172,   172,   173,   173,   173,   173,   173,   173,
     174,   175,   175,   175,   176,   176,   177,   177,   178,   178,
     179,   179,   180,   180,   181,   181,   181,   182,   182,   183,
     183,   184,   184,   185,   185,   186,   186,   187,   187,   187,
     187,   187,   187,   187,   187,   187,   187,   187,   187,   187,
     187,   187,   187,   188,   188,   188,   189,   189,   190,   191,
     192,   192,   193,   193,   194,   195,   196,   197
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     0,     2,     0,     1,     0,     1,     1,
       1,     4,     3,     3,     1,     4,     0,     2,     5,     5,
       4,     3,     2,     2,     3,     2,     2,     2,     2,     2,
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
       2,     2,     2,     2,     2,     1,     4,     0,     2,     2,
       2,     2,     2,     3,     3,     3,     3,     6,     3,     6,
       3,     6,     9,     1,     3,     1,     2,     1,     7,     5,
      12,     5,     2,     1,     1,     0,     3,     1,     3,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     2,     1,     1,     1,     0,     1,     3,     0,     1,
       5,     5,     5,     4,     3,     1,     1,     1,     3,     4,
       3,     1,     1,     1,     1,     4,     3,     4,     4,     4,
       3,     7,     5,     6,     1,     3,     1,     3,     3,     2,
       3,     2,     0,     3,     0,     1,     3,     1,     2,     1,
       2,     0,     4,     1,     3,     1,     0,     6,     8,     4,
       3,     5,     4,    11,     9,    12,    14,     6,     8,     5,
       7,     3,     3,     0,     2,     4,     1,     3,     1,     1,
       2,     5,     1,     3,     1,     1,     2,     2
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       3,     0,     0,     0,     2,     3,     9,     0,     0,     1,
       4,    14,     5,    12,    13,     6,     0,     0,     0,     0,
       0,     0,     0,     5,    32,    30,    31,     0,     0,    10,
       0,   276,   277,   172,   205,   198,     0,    80,    80,    80,
       0,    88,    88,    88,    88,     0,    82,     0,     0,     0,
       5,    22,     0,     0,     0,    25,    26,    27,    28,     0,
      29,    23,     0,     0,     7,    17,     5,     0,    21,     0,
     199,   198,     0,     0,    56,     0,    42,    43,    44,    45,
      53,    54,     0,    40,    59,    60,    65,    66,    67,    68,
      72,     0,    81,     0,     0,     0,     0,   163,     0,     0,
       0,     0,     0,     0,     0,     0,    95,     0,     0,     0,
     165,     0,     0,     0,   149,   150,    24,    88,    88,    88,
      88,     0,    82,   140,   141,   142,   143,   144,   152,   151,
       8,    15,     0,    20,     0,   198,   195,   198,     0,   206,
       0,     0,    69,    52,    55,    46,    47,    50,    51,    38,
      58,    61,    62,    63,    70,     0,    71,    76,   158,   156,
       0,   160,     0,   153,    99,   100,     0,    90,    40,   110,
     110,   110,   110,    94,     0,     0,    97,     0,     0,     0,
       0,     0,    86,    87,     0,    84,   108,   155,   154,     0,
      68,     0,   137,     0,     7,     0,     0,     0,     0,     0,
       0,    19,    11,     0,   196,     0,     0,   198,   171,     0,
      48,    49,    34,    35,    36,    39,     0,    33,     0,     0,
      76,    78,    80,    80,    80,    80,   164,    89,     0,    57,
       0,     0,     0,     0,     0,     0,   119,     0,    96,   110,
     110,    83,     0,   101,   129,     0,   135,   131,     0,   139,
      18,   110,   110,   110,   110,   110,     0,   198,   195,   198,
     198,   203,    79,     0,    41,     0,    73,    74,     0,    77,
       0,     0,     0,     0,    91,   112,   111,   145,   147,   114,
     115,   116,   117,   118,    98,     0,     0,    85,   102,     0,
     101,     0,     0,   134,   132,   133,   136,   138,     0,     0,
       0,     0,     0,   127,   101,   201,   197,   202,   200,    37,
       0,    75,   159,   157,     0,   161,     0,   175,     0,   167,
     147,     0,   120,   121,     0,   107,   109,   130,   122,   123,
     124,   125,   126,     0,     0,    80,   113,     0,     0,     7,
     148,   166,   103,     0,   207,   198,   224,     0,     0,   179,
     180,   181,   182,   187,   188,   189,   183,   184,   185,   186,
      92,   190,     0,     0,   177,    56,    10,     0,     0,   174,
       0,   146,     0,     0,   128,   101,     0,     0,    64,   162,
      93,   191,     0,   176,     0,     0,   234,     0,   104,   105,
     204,     0,   208,   210,   225,     0,   178,   229,     0,     0,
       0,     0,     0,     0,   246,     0,     0,     0,   205,   198,
     169,   235,   232,   193,   192,   194,   209,     0,   228,   269,
     198,     0,   198,     0,   272,     0,     0,   245,     0,   266,
       0,   198,     0,     0,   237,     0,     0,   234,     0,     0,
       0,     0,   274,   270,   198,     0,   205,   250,     0,   239,
     198,     0,   261,     0,     0,   236,   238,   262,     0,   168,
       0,     0,   198,     0,   268,     0,     0,   273,   249,     0,
     252,   240,     0,   267,     0,     0,   233,   211,   212,   213,
     231,     0,     0,   226,     0,   198,     0,   198,   205,   259,
     275,     0,   251,   241,   205,   263,     0,     0,     0,     0,
     230,     0,   198,     0,     0,   271,     0,   247,     0,     0,
     257,   198,     0,     0,   198,     0,   227,     0,     0,   198,
     260,     0,   263,   205,   264,     0,   214,     0,     0,     0,
       0,   170,     0,     0,   243,     0,   258,     0,   248,   222,
       0,     0,     0,     0,     0,   220,     0,   205,   254,   198,
       0,   242,   265,     0,     0,     0,     0,   216,     0,   223,
       0,     0,   244,   219,   218,   217,   215,   221,   253,     0,
     205,   255,     0,   256
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     3,     4,    21,   131,   168,    83,     5,    13,    22,
      23,   214,   215,   216,   150,    84,   169,    85,    86,    87,
      88,    89,    90,   344,   268,   220,   221,    52,    53,    93,
     108,   184,   185,   100,   166,   381,   176,   105,   177,   167,
     289,   373,   290,   291,    54,   231,   276,    55,    56,    57,
     106,    58,   123,   124,   125,   126,   127,   293,   246,   192,
     193,    59,    60,   279,   318,   319,    62,    63,    98,   111,
     320,   321,    24,   370,   338,   363,   364,   416,   205,    72,
     345,   409,   140,   346,   482,   527,   515,   483,   347,   484,
     386,   461,   437,   410,   433,   448,   507,   535,   428,   434,
     510,   430,   465,   420,   424,   425,   444,   491,    25,    26
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -447
static const yytype_int16 yypact[] =
{
     145,   -31,   -31,    55,  -447,   145,  -447,     1,     1,  -447,
    -447,  -447,   211,  -447,  -447,  -447,   -31,   125,   -31,   -31,
     112,   225,   -32,   211,  -447,  -447,  -447,     7,     8,  -447,
     154,  -447,  -447,  -447,  -447,   -28,   294,    56,    56,    -6,
       8,    27,    27,    27,    27,    74,    92,   -31,    69,    62,
     211,  -447,   109,   132,   135,  -447,  -447,  -447,  -447,   533,
    -447,  -447,   137,   170,   174,  -447,   211,   181,  -447,   182,
      15,   -28,   187,   564,  -447,   470,  -447,    38,  -447,  -447,
    -447,  -447,    97,    35,  -447,  -447,   189,   200,   206,  -447,
     -24,     8,  -447,     8,     8,   208,     8,   231,   247,    11,
     -31,   -31,   -31,   -31,    79,   244,   249,   220,   -31,   253,
    -447,   264,   335,   258,  -447,  -447,  -447,    27,    27,    27,
      27,   244,    92,  -447,  -447,  -447,  -447,  -447,  -447,  -447,
    -447,  -447,   263,  -447,   285,   -28,   284,   -28,   269,  -447,
     279,   275,   -15,  -447,  -447,  -447,   205,  -447,  -447,   551,
    -447,  -447,  -447,  -447,  -447,   283,  -447,   -22,   -26,    49,
     280,    58,     8,  -447,  -447,  -447,   282,   291,   295,   308,
     308,   308,   308,  -447,   -31,   304,   292,   305,   217,   -31,
     340,   -31,  -447,  -447,   311,   316,   334,  -447,  -447,   -31,
      -2,   -31,   333,   344,   174,   -31,   -31,   -31,   -31,   -31,
     -31,  -447,  -447,   336,   347,   343,   341,   -28,  -447,   -31,
    -447,  -447,  -447,  -447,   353,  -447,   361,  -447,   -31,   235,
     348,  -447,    56,    56,    56,    56,  -447,  -447,    11,  -447,
       8,   115,   115,   115,   115,   356,  -447,   340,  -447,   308,
     308,  -447,   220,   424,   357,   241,  -447,   359,   335,  -447,
    -447,   308,   308,   308,   308,   308,   119,   -28,   284,   -28,
     -28,  -447,  -447,   551,  -447,   366,  -447,   375,   365,  -447,
     369,   370,     8,   374,  -447,   377,  -447,  -447,   242,  -447,
    -447,  -447,  -447,  -447,  -447,   115,   115,  -447,  -447,   470,
      -4,   381,   470,  -447,  -447,  -447,  -447,  -447,   115,   115,
     115,   115,   115,  -447,   424,  -447,  -447,  -447,  -447,  -447,
     378,  -447,  -447,  -447,    67,  -447,     8,   376,   385,  -447,
     242,   389,  -447,  -447,   -31,  -447,  -447,  -447,  -447,  -447,
    -447,  -447,  -447,   388,   470,    56,  -447,   352,   585,   174,
    -447,  -447,   392,   405,   -31,   -28,   403,   383,   408,  -447,
    -447,  -447,  -447,  -447,  -447,  -447,  -447,  -447,  -447,  -447,
     434,  -447,   419,   420,   431,   454,   428,   430,   189,  -447,
     -31,  -447,   429,   442,  -447,     9,   444,   470,  -447,  -447,
    -447,  -447,   495,  -447,   570,   371,   372,   430,  -447,  -447,
    -447,   168,  -447,  -447,  -447,   -31,  -447,  -447,   457,   461,
     467,   461,   497,   479,   507,   499,   461,   494,   427,   -28,
    -447,  -447,   568,  -447,  -447,   375,  -447,   430,  -447,  -447,
     -28,   529,   -28,    89,   513,   441,   427,  -447,   518,   535,
     537,   -28,   550,   548,   427,   187,   539,   372,   549,   561,
     552,   553,  -447,  -447,   -28,   497,   271,  -447,   560,   427,
     -28,   499,  -447,   553,   430,  -447,  -447,  -447,   581,  -447,
     101,   518,   -28,   461,  -447,   453,   554,  -447,  -447,   566,
    -447,  -447,   187,  -447,   477,   558,  -447,  -447,  -447,  -447,
    -447,   -31,   569,   567,   571,   -28,   572,   -28,   427,  -447,
    -447,   430,  -447,   595,   427,   605,   518,   577,   470,   239,
    -447,   187,   -28,   583,   582,  -447,   578,  -447,   584,   516,
    -447,   -28,   -31,   -31,   -28,   586,  -447,   -31,   553,   -28,
    -447,   606,   605,   427,  -447,   587,  -447,    96,    19,   576,
     -31,  -447,   526,   588,   589,   590,  -447,   591,  -447,  -447,
     -31,   266,   592,   -31,   -31,  -447,   248,   427,  -447,   -28,
     606,  -447,  -447,   197,   593,   209,   -31,  -447,   261,  -447,
     594,   553,  -447,  -447,  -447,  -447,  -447,  -447,  -447,   536,
     427,  -447,   597,  -447
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -447,  -447,   662,  -447,  -188,    -1,    -9,   647,   665,    43,
    -447,  -447,   411,  -447,   508,  -447,   -68,   -29,   -69,   337,
    -447,  -107,   603,   -33,  -447,  -447,   459,  -447,  -447,   -11,
     559,   438,  -447,    25,   455,  -447,  -447,   579,   445,  -447,
     309,  -447,  -447,  -259,  -447,  -134,   373,  -447,  -447,  -447,
     -89,  -447,  -447,  -447,  -447,  -447,  -447,  -447,   439,  -447,
     440,  -447,  -447,   -80,   367,   669,  -447,  -447,   530,  -447,
    -447,  -447,  -447,  -447,  -447,   307,  -447,  -447,   435,   -57,
     196,   -18,  -415,  -447,  -447,  -416,  -447,  -447,  -335,   198,
    -364,  -447,  -447,   259,  -424,  -447,  -447,   148,  -447,  -377,
     173,   250,  -446,  -362,  -447,   254,  -447,  -447,  -447,  -447
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -174
static const yytype_int16 yytable[] =
{
       7,     8,    35,    91,   141,   190,   250,   474,    30,   411,
     456,    97,   154,   288,   138,    27,   179,    31,    32,    67,
     457,   154,   469,   412,     6,   471,   288,    94,    96,    92,
      70,   325,   199,   170,   171,   172,   232,   233,   234,   422,
     186,   164,   394,    64,   431,   333,   109,    69,   447,   449,
     398,    69,    71,   438,   222,     9,   155,   493,   156,   443,
     411,   219,   165,    29,   504,   155,    65,   101,   102,   103,
     508,  -106,   532,    11,  -131,    12,  -131,    95,   203,   191,
     206,    66,   157,   245,   158,   159,   517,   161,   489,   135,
     475,    92,   390,   113,   391,   136,   544,   495,   137,   537,
     143,   487,   390,   175,   144,   285,   286,   477,    69,   132,
      99,   239,   149,   240,   546,   569,   217,   298,   299,   300,
     301,   302,    69,   560,   553,   555,   110,   505,   558,   223,
     173,    69,   524,    97,     6,   174,    74,    75,   225,   112,
      69,   190,   195,   196,   197,   198,   572,   335,     1,     2,
     261,   371,   280,   281,   282,   548,    29,   104,   145,   146,
     147,   148,    76,    77,    78,    79,    80,    81,    82,   385,
     478,   479,   442,   235,   539,   107,   540,   175,    28,   541,
      29,   114,   542,   543,    33,   480,    34,   277,   244,   278,
     247,   303,   571,   304,   251,   252,   253,   254,   255,   256,
     305,   275,   307,   308,   115,   322,   323,   116,   262,   128,
     267,   270,   271,   272,   273,   191,    15,   265,   328,   329,
     330,   331,   332,    29,   413,   414,    68,    69,     1,     2,
     217,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,   129,    47,    48,   477,   130,    49,    36,    37,
      38,    39,    40,   133,    69,   134,   324,   182,   183,   327,
      47,    48,   139,   314,    16,    17,   210,   211,   151,   368,
      18,    19,     6,   174,    74,    75,   160,   540,   563,   152,
     541,   317,    20,   542,   543,   153,   -16,   275,   376,   540,
      29,   266,   541,   565,    29,   542,   543,   294,   295,    50,
      76,    77,    78,    79,    80,    81,    82,   162,   478,   479,
      73,   399,   400,   401,   402,   403,   404,   405,   406,   163,
     407,     6,   554,   342,   348,   187,   559,   178,   540,    74,
      75,   541,   180,   194,   542,   543,   188,   367,   201,   567,
     202,   540,   204,   375,   541,    34,   468,   542,   543,    29,
     207,   208,   435,   189,   209,    76,    77,    78,    79,    80,
      81,    82,   218,   439,   224,   441,   227,   228,   237,   387,
      74,    75,   149,   349,   453,   350,   351,   352,   353,   354,
     355,   230,   415,   356,   357,   358,   359,   466,   236,   238,
      29,   173,   242,   472,   417,   241,    76,    77,    78,    79,
      80,    81,    82,   360,   361,   486,    74,    75,   243,   248,
     450,   257,   399,   400,   401,   402,   403,   404,   405,   406,
     362,   407,   249,   258,   259,   260,    29,   481,   501,   263,
     503,   219,    76,    77,    78,    79,    80,    81,    82,   264,
     283,   288,   292,   485,   245,   518,   408,   310,    69,   311,
     312,   313,   397,   316,   525,   315,   326,   529,   334,   337,
     339,   341,   533,   343,   378,   513,   481,   399,   400,   401,
     402,   403,   404,   405,   406,   372,   407,   374,   511,   377,
     497,   399,   400,   401,   402,   403,   404,   405,   406,   379,
     407,   380,   561,   399,   400,   401,   402,   403,   404,   405,
     406,    34,   407,   382,   383,    74,    75,   384,    -9,  -173,
     385,   526,   528,   388,   389,   446,   531,   399,   400,   401,
     402,   403,   404,   405,   406,    29,   407,   488,   393,   526,
     395,    76,    77,    78,    79,    80,    81,    82,   418,   526,
     526,   419,   557,   526,   117,   118,   119,   120,   121,   122,
     421,   494,   423,   426,   429,   566,   399,   400,   401,   402,
     403,   404,   405,   406,   427,   407,   399,   400,   401,   402,
     403,   404,   405,   406,   432,   407,   399,   400,   401,   402,
     403,   404,   405,   406,   440,   407,    74,    75,   436,   445,
     523,   349,    34,   350,   351,   352,   353,   354,   355,    74,
     547,   356,   357,   358,   359,   454,    29,   212,   213,   452,
     570,   451,    76,    77,    78,    79,    80,    81,    82,    29,
     365,   360,   361,   455,   458,    76,    77,    78,    79,    80,
      81,    82,   460,   462,   464,   470,   463,   476,   490,   496,
     366,   492,   498,   499,   502,   506,    76,    77,    78,    79,
      80,    81,    82,   509,   512,   500,   519,   520,   521,   522,
     545,   534,   538,   530,   549,   550,   552,    10,    51,   568,
     556,   551,   573,    14,   309,   369,   229,   564,   142,   269,
     287,   200,   284,   274,   392,   181,   296,   340,   297,   336,
      61,   396,   226,   306,   514,   536,   459,   516,   562,   467,
       0,   473
};

static const yytype_int16 yycheck[] =
{
       1,     2,    20,    36,    73,   112,   194,   453,    17,   386,
     434,    40,    36,    17,    71,    16,   105,    18,    19,    28,
     435,    36,   446,   387,    55,   449,    17,    38,    39,    35,
      58,   290,   121,   101,   102,   103,   170,   171,   172,   401,
     108,    30,   377,    75,   406,   304,    47,    73,   425,   426,
     385,    73,    80,   417,    80,     0,    80,   472,    82,   423,
     437,    83,    51,    55,   488,    80,    23,    42,    43,    44,
     494,    75,   518,    72,    76,    74,    78,    83,   135,   112,
     137,    74,    91,    85,    93,    94,   501,    96,   465,    74,
     454,    35,    83,    50,    85,    80,    77,   474,    83,   523,
      62,   463,    83,   104,    66,   239,   240,     6,    73,    66,
      83,   179,    77,   181,   530,   561,   149,   251,   252,   253,
     254,   255,    73,   547,   540,   541,    57,   491,   544,    80,
      51,    73,   509,   162,    55,    56,    35,    36,    80,    77,
      73,   248,   117,   118,   119,   120,   570,    80,     3,     4,
     207,   339,   232,   233,   234,   532,    55,    83,    61,    62,
      63,    64,    61,    62,    63,    64,    65,    66,    67,    80,
      69,    70,    83,   174,    78,    83,    80,   178,    53,    83,
      55,    72,    86,    87,    72,    84,    74,    72,   189,    74,
     191,    72,   569,    74,   195,   196,   197,   198,   199,   200,
     257,   230,   259,   260,    72,   285,   286,    72,   209,    72,
     219,   222,   223,   224,   225,   248,     5,   218,   298,   299,
     300,   301,   302,    55,    56,    57,    72,    73,     3,     4,
     263,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    72,    18,    19,     6,    72,    22,     6,     7,
       8,     9,    10,    72,    73,    73,   289,    37,    38,   292,
      18,    19,    75,   272,    53,    54,    61,    62,    79,   338,
      59,    60,    55,    56,    35,    36,    68,    80,    81,    79,
      83,    39,    71,    86,    87,    79,    75,   316,   345,    80,
      55,    56,    83,    84,    55,    86,    87,    56,    57,    74,
      61,    62,    63,    64,    65,    66,    67,    76,    69,    70,
      16,    40,    41,    42,    43,    44,    45,    46,    47,    72,
      49,    55,    56,   324,   335,    72,    78,    83,    80,    35,
      36,    83,    83,    75,    86,    87,    72,   338,    75,    78,
      55,    80,    58,   344,    83,    74,    75,    86,    87,    55,
      81,    72,   409,    18,    79,    61,    62,    63,    64,    65,
      66,    67,    79,   420,    84,   422,    84,    76,    76,   370,
      35,    36,    77,    21,   431,    23,    24,    25,    26,    27,
      28,    73,   391,    31,    32,    33,    34,   444,    84,    84,
      55,    51,    76,   450,   395,    84,    61,    62,    63,    64,
      65,    66,    67,    51,    52,   462,    35,    36,    74,    76,
     428,    75,    40,    41,    42,    43,    44,    45,    46,    47,
      68,    49,    78,    76,    81,    84,    55,   460,   485,    76,
     487,    83,    61,    62,    63,    64,    65,    66,    67,    78,
      84,    17,    85,   461,    85,   502,    74,    81,    73,    84,
      81,    81,    81,    76,   511,    81,    75,   514,    80,    83,
      75,    72,   519,    75,    81,   498,   499,    40,    41,    42,
      43,    44,    45,    46,    47,    83,    49,    72,   496,    76,
     481,    40,    41,    42,    43,    44,    45,    46,    47,    81,
      49,    57,   549,    40,    41,    42,    43,    44,    45,    46,
      47,    74,    49,    84,    84,    35,    36,    76,    80,    55,
      80,   512,   513,    84,    72,    74,   517,    40,    41,    42,
      43,    44,    45,    46,    47,    55,    49,    74,    84,   530,
      35,    61,    62,    63,    64,    65,    66,    67,    81,   540,
     541,    80,   543,   544,    11,    12,    13,    14,    15,    16,
      83,    74,    55,    74,    55,   556,    40,    41,    42,    43,
      44,    45,    46,    47,    57,    49,    40,    41,    42,    43,
      44,    45,    46,    47,    80,    49,    40,    41,    42,    43,
      44,    45,    46,    47,    55,    49,    35,    36,    20,    76,
      74,    21,    74,    23,    24,    25,    26,    27,    28,    35,
      74,    31,    32,    33,    34,    55,    55,    56,    57,    72,
      74,    76,    61,    62,    63,    64,    65,    66,    67,    55,
      35,    51,    52,    75,    85,    61,    62,    63,    64,    65,
      66,    67,    83,    72,    81,    75,    84,    56,    84,    81,
      55,    75,    73,    76,    72,    50,    61,    62,    63,    64,
      65,    66,    67,    48,    77,    84,    73,    75,    80,    75,
      84,    55,    75,    77,    76,    76,    75,     5,    21,    75,
      78,    81,    75,     8,   263,   338,   168,    84,    75,   220,
     242,   122,   237,   228,   375,   106,   247,   320,   248,   316,
      21,   384,   162,   258,   498,   522,   437,   499,   550,   445,
      -1,   451
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     3,     4,    89,    90,    95,    55,    93,    93,     0,
      90,    72,    74,    96,    96,     5,    53,    54,    59,    60,
      71,    91,    97,    98,   160,   196,   197,    93,    53,    55,
      94,    93,    93,    72,    74,   169,     6,     7,     8,     9,
      10,    11,    12,    13,    14,    15,    16,    18,    19,    22,
      74,    95,   115,   116,   132,   135,   136,   137,   139,   149,
     150,   153,   154,   155,    75,    97,    74,    94,    72,    73,
      58,    80,   167,    16,    35,    36,    61,    62,    63,    64,
      65,    66,    67,    94,   103,   105,   106,   107,   108,   109,
     110,   111,    35,   117,   117,    83,   117,   105,   156,    83,
     121,   121,   121,   121,    83,   125,   138,    83,   118,    93,
      57,   157,    77,    97,    72,    72,    72,    11,    12,    13,
      14,    15,    16,   140,   141,   142,   143,   144,    72,    72,
      72,    92,    97,    72,    73,    74,    80,    83,   167,    75,
     170,   106,   110,    62,    66,    61,    62,    63,    64,    77,
     102,    79,    79,    79,    36,    80,    82,    94,    94,    94,
      68,    94,    76,    72,    30,    51,   122,   127,    93,   104,
     104,   104,   104,    51,    56,    93,   124,   126,    83,   138,
      83,   125,    37,    38,   119,   120,   104,    72,    72,    18,
     109,   111,   147,   148,    75,   121,   121,   121,   121,   138,
     118,    75,    55,   167,    58,   166,   167,    81,    72,    79,
      61,    62,    56,    57,    99,   100,   101,   111,    79,    83,
     113,   114,    80,    80,    84,    80,   156,    84,    76,   102,
      73,   133,   133,   133,   133,    93,    84,    76,    84,   104,
     104,    84,    76,    74,    93,    85,   146,    93,    76,    78,
      92,    93,    93,    93,    93,    93,    93,    75,    76,    81,
      84,   167,    93,    76,    78,    93,    56,    94,   112,   114,
     117,   117,   117,   117,   122,   105,   134,    72,    74,   151,
     151,   151,   151,    84,   126,   133,   133,   119,    17,   128,
     130,   131,    85,   145,    56,    57,   146,   148,   133,   133,
     133,   133,   133,    72,    74,   167,   166,   167,   167,   100,
      81,    84,    81,    81,    94,    81,    76,    39,   152,   153,
     158,   159,   151,   151,   111,   131,    75,   111,   151,   151,
     151,   151,   151,   131,    80,    80,   134,    83,   162,    75,
     152,    72,    93,    75,   111,   168,   171,   176,   117,    21,
      23,    24,    25,    26,    27,    28,    31,    32,    33,    34,
      51,    52,    68,   163,   164,    35,    55,    93,   106,   107,
     161,    92,    83,   129,    72,    93,   167,    76,    81,    81,
      57,   123,    84,    84,    76,    80,   178,    93,    84,    72,
      83,    85,   128,    84,   176,    35,   163,    81,   176,    40,
      41,    42,    43,    44,    45,    46,    47,    49,    74,   169,
     181,   187,   178,    56,    57,    94,   165,    93,    81,    80,
     191,    83,   191,    55,   192,   193,    74,    57,   186,    55,
     189,   191,    80,   182,   187,   167,    20,   180,   178,   167,
      55,   167,    83,   178,   194,    76,    74,   187,   183,   187,
     169,    76,    72,   167,    55,    75,   182,   170,    85,   181,
      83,   179,    72,    84,    81,   190,   167,   193,    75,   182,
      75,   182,   167,   189,   190,   178,    56,     6,    69,    70,
      84,   111,   172,   175,   177,   169,   167,   191,    74,   187,
      84,   195,    75,   170,    74,   187,    81,    93,    73,    76,
      84,   167,    72,   167,   182,   178,    50,   184,   182,    48,
     188,   169,    77,   111,   168,   174,   177,   170,   167,    73,
      75,    80,    75,    74,   187,   167,    93,   173,    93,   167,
      77,    93,   190,   167,    55,   185,   188,   182,    75,    78,
      80,    83,    86,    87,    77,    84,   173,    74,   187,    76,
      76,    81,    75,   173,    56,   173,    78,    93,   173,    78,
     182,   167,   185,    81,    84,    84,    93,    78,    75,   190,
      74,   187,   182,    75
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
#line 148 "xi-grammar.y"
    { (yyval.modlist) = (yyvsp[(1) - (1)].modlist); modlist = (yyvsp[(1) - (1)].modlist); }
    break;

  case 3:
#line 152 "xi-grammar.y"
    { 
		  (yyval.modlist) = 0; 
		}
    break;

  case 4:
#line 156 "xi-grammar.y"
    { (yyval.modlist) = new ModuleList(lineno, (yyvsp[(1) - (2)].module), (yyvsp[(2) - (2)].modlist)); }
    break;

  case 5:
#line 160 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 6:
#line 162 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 7:
#line 166 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 8:
#line 168 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 9:
#line 172 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 10:
#line 176 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 11:
#line 178 "xi-grammar.y"
    {
		  char *tmp = new char[strlen((yyvsp[(1) - (4)].strval))+strlen((yyvsp[(4) - (4)].strval))+3];
		  sprintf(tmp,"%s::%s", (yyvsp[(1) - (4)].strval), (yyvsp[(4) - (4)].strval));
		  (yyval.strval) = tmp;
		}
    break;

  case 12:
#line 186 "xi-grammar.y"
    { 
		    (yyval.module) = new Module(lineno, (yyvsp[(2) - (3)].strval), (yyvsp[(3) - (3)].conslist)); 
		}
    break;

  case 13:
#line 190 "xi-grammar.y"
    {  
		    (yyval.module) = new Module(lineno, (yyvsp[(2) - (3)].strval), (yyvsp[(3) - (3)].conslist)); 
		    (yyval.module)->setMain();
		}
    break;

  case 14:
#line 197 "xi-grammar.y"
    { (yyval.conslist) = 0; }
    break;

  case 15:
#line 199 "xi-grammar.y"
    { (yyval.conslist) = (yyvsp[(2) - (4)].conslist); }
    break;

  case 16:
#line 203 "xi-grammar.y"
    { (yyval.conslist) = 0; }
    break;

  case 17:
#line 205 "xi-grammar.y"
    { (yyval.conslist) = new ConstructList(lineno, (yyvsp[(1) - (2)].construct), (yyvsp[(2) - (2)].conslist)); }
    break;

  case 18:
#line 209 "xi-grammar.y"
    { if((yyvsp[(3) - (5)].conslist)) (yyvsp[(3) - (5)].conslist)->setExtern((yyvsp[(1) - (5)].intval)); (yyval.construct) = (yyvsp[(3) - (5)].conslist); }
    break;

  case 19:
#line 211 "xi-grammar.y"
    { (yyval.construct) = new Scope((yyvsp[(2) - (5)].strval), (yyvsp[(4) - (5)].conslist)); }
    break;

  case 20:
#line 213 "xi-grammar.y"
    { (yyval.construct) = new UsingScope((yyvsp[(3) - (4)].strval), false); }
    break;

  case 21:
#line 215 "xi-grammar.y"
    { (yyval.construct) = new UsingScope((yyvsp[(2) - (3)].strval), true); }
    break;

  case 22:
#line 217 "xi-grammar.y"
    { (yyvsp[(2) - (2)].module)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].module); }
    break;

  case 23:
#line 219 "xi-grammar.y"
    { (yyvsp[(2) - (2)].member)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].member); }
    break;

  case 24:
#line 221 "xi-grammar.y"
    { (yyvsp[(2) - (3)].message)->setExtern((yyvsp[(1) - (3)].intval)); (yyval.construct) = (yyvsp[(2) - (3)].message); }
    break;

  case 25:
#line 223 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 26:
#line 225 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 27:
#line 227 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 28:
#line 229 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 29:
#line 231 "xi-grammar.y"
    { (yyvsp[(2) - (2)].templat)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].templat); }
    break;

  case 30:
#line 233 "xi-grammar.y"
    { (yyval.construct) = NULL; }
    break;

  case 31:
#line 235 "xi-grammar.y"
    { (yyval.construct) = NULL; }
    break;

  case 32:
#line 237 "xi-grammar.y"
    { (yyval.construct) = (yyvsp[(1) - (1)].accelBlock); }
    break;

  case 33:
#line 241 "xi-grammar.y"
    { (yyval.tparam) = new TParamType((yyvsp[(1) - (1)].type)); }
    break;

  case 34:
#line 243 "xi-grammar.y"
    { (yyval.tparam) = new TParamVal((yyvsp[(1) - (1)].strval)); }
    break;

  case 35:
#line 245 "xi-grammar.y"
    { (yyval.tparam) = new TParamVal((yyvsp[(1) - (1)].strval)); }
    break;

  case 36:
#line 249 "xi-grammar.y"
    { (yyval.tparlist) = new TParamList((yyvsp[(1) - (1)].tparam)); }
    break;

  case 37:
#line 251 "xi-grammar.y"
    { (yyval.tparlist) = new TParamList((yyvsp[(1) - (3)].tparam), (yyvsp[(3) - (3)].tparlist)); }
    break;

  case 38:
#line 255 "xi-grammar.y"
    { (yyval.tparlist) = 0; }
    break;

  case 39:
#line 257 "xi-grammar.y"
    { (yyval.tparlist) = (yyvsp[(1) - (1)].tparlist); }
    break;

  case 40:
#line 261 "xi-grammar.y"
    { (yyval.tparlist) = 0; }
    break;

  case 41:
#line 263 "xi-grammar.y"
    { (yyval.tparlist) = (yyvsp[(2) - (3)].tparlist); }
    break;

  case 42:
#line 267 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("int"); }
    break;

  case 43:
#line 269 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long"); }
    break;

  case 44:
#line 271 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("short"); }
    break;

  case 45:
#line 273 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("char"); }
    break;

  case 46:
#line 275 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned int"); }
    break;

  case 47:
#line 277 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long"); }
    break;

  case 48:
#line 279 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long"); }
    break;

  case 49:
#line 281 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long long"); }
    break;

  case 50:
#line 283 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned short"); }
    break;

  case 51:
#line 285 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned char"); }
    break;

  case 52:
#line 287 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long long"); }
    break;

  case 53:
#line 289 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("float"); }
    break;

  case 54:
#line 291 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("double"); }
    break;

  case 55:
#line 293 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long double"); }
    break;

  case 56:
#line 295 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("void"); }
    break;

  case 57:
#line 298 "xi-grammar.y"
    { (yyval.ntype) = new NamedType((yyvsp[(1) - (2)].strval),(yyvsp[(2) - (2)].tparlist)); }
    break;

  case 58:
#line 299 "xi-grammar.y"
    { 
                    char* basename, *scope;
                    splitScopedName((yyvsp[(1) - (2)].strval), &scope, &basename);
                    (yyval.ntype) = new NamedType(basename, (yyvsp[(2) - (2)].tparlist), scope);
                }
    break;

  case 59:
#line 307 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 60:
#line 309 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ntype); }
    break;

  case 61:
#line 313 "xi-grammar.y"
    { (yyval.ptype) = new PtrType((yyvsp[(1) - (2)].type)); }
    break;

  case 62:
#line 317 "xi-grammar.y"
    { (yyvsp[(1) - (2)].ptype)->indirect(); (yyval.ptype) = (yyvsp[(1) - (2)].ptype); }
    break;

  case 63:
#line 319 "xi-grammar.y"
    { (yyvsp[(1) - (2)].ptype)->indirect(); (yyval.ptype) = (yyvsp[(1) - (2)].ptype); }
    break;

  case 64:
#line 323 "xi-grammar.y"
    { (yyval.ftype) = new FuncType((yyvsp[(1) - (8)].type), (yyvsp[(4) - (8)].strval), (yyvsp[(7) - (8)].plist)); }
    break;

  case 65:
#line 327 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 66:
#line 329 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 67:
#line 331 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 68:
#line 333 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ftype); }
    break;

  case 69:
#line 336 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(2) - (2)].type); }
    break;

  case 70:
#line 338 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (2)].type); }
    break;

  case 71:
#line 342 "xi-grammar.y"
    { (yyval.type) = new ReferenceType((yyvsp[(1) - (2)].type)); }
    break;

  case 72:
#line 344 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 73:
#line 348 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 74:
#line 350 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 75:
#line 354 "xi-grammar.y"
    { (yyval.val) = (yyvsp[(2) - (3)].val); }
    break;

  case 76:
#line 358 "xi-grammar.y"
    { (yyval.vallist) = 0; }
    break;

  case 77:
#line 360 "xi-grammar.y"
    { (yyval.vallist) = new ValueList((yyvsp[(1) - (2)].val), (yyvsp[(2) - (2)].vallist)); }
    break;

  case 78:
#line 364 "xi-grammar.y"
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[(2) - (4)].type), (yyvsp[(3) - (4)].strval), (yyvsp[(4) - (4)].vallist)); }
    break;

  case 79:
#line 368 "xi-grammar.y"
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[(3) - (5)].type), (yyvsp[(5) - (5)].strval), 0, 1); }
    break;

  case 80:
#line 372 "xi-grammar.y"
    { (yyval.intval) = 0;}
    break;

  case 81:
#line 374 "xi-grammar.y"
    { (yyval.intval) = 0;}
    break;

  case 82:
#line 378 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 83:
#line 380 "xi-grammar.y"
    { 
		  /*
		  printf("Warning: Message attributes are being phased out.\n");
		  printf("Warning: Please remove them from interface files.\n");
		  */
		  (yyval.intval) = (yyvsp[(2) - (3)].intval); 
		}
    break;

  case 84:
#line 390 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (1)].intval); }
    break;

  case 85:
#line 392 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (3)].intval) | (yyvsp[(3) - (3)].intval); }
    break;

  case 86:
#line 396 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 87:
#line 398 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 88:
#line 402 "xi-grammar.y"
    { (yyval.cattr) = 0; }
    break;

  case 89:
#line 404 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(2) - (3)].cattr); }
    break;

  case 90:
#line 408 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (1)].cattr); }
    break;

  case 91:
#line 410 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (3)].cattr) | (yyvsp[(3) - (3)].cattr); }
    break;

  case 92:
#line 414 "xi-grammar.y"
    { python_doc = NULL; (yyval.intval) = 0; }
    break;

  case 93:
#line 416 "xi-grammar.y"
    { python_doc = (yyvsp[(1) - (1)].strval); (yyval.intval) = 0; }
    break;

  case 94:
#line 420 "xi-grammar.y"
    { (yyval.cattr) = Chare::CPYTHON; }
    break;

  case 95:
#line 424 "xi-grammar.y"
    { (yyval.cattr) = 0; }
    break;

  case 96:
#line 426 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(2) - (3)].cattr); }
    break;

  case 97:
#line 430 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (1)].cattr); }
    break;

  case 98:
#line 432 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (3)].cattr) | (yyvsp[(3) - (3)].cattr); }
    break;

  case 99:
#line 436 "xi-grammar.y"
    { (yyval.cattr) = Chare::CMIGRATABLE; }
    break;

  case 100:
#line 438 "xi-grammar.y"
    { (yyval.cattr) = Chare::CPYTHON; }
    break;

  case 101:
#line 442 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 102:
#line 444 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 103:
#line 447 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 104:
#line 449 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 105:
#line 452 "xi-grammar.y"
    { (yyval.mv) = new MsgVar((yyvsp[(2) - (5)].type), (yyvsp[(3) - (5)].strval), (yyvsp[(1) - (5)].intval), (yyvsp[(4) - (5)].intval)); }
    break;

  case 106:
#line 456 "xi-grammar.y"
    { (yyval.mvlist) = new MsgVarList((yyvsp[(1) - (1)].mv)); }
    break;

  case 107:
#line 458 "xi-grammar.y"
    { (yyval.mvlist) = new MsgVarList((yyvsp[(1) - (2)].mv), (yyvsp[(2) - (2)].mvlist)); }
    break;

  case 108:
#line 462 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, (yyvsp[(3) - (3)].ntype)); }
    break;

  case 109:
#line 464 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, (yyvsp[(3) - (6)].ntype), (yyvsp[(5) - (6)].mvlist)); }
    break;

  case 110:
#line 468 "xi-grammar.y"
    { (yyval.typelist) = 0; }
    break;

  case 111:
#line 470 "xi-grammar.y"
    { (yyval.typelist) = (yyvsp[(2) - (2)].typelist); }
    break;

  case 112:
#line 474 "xi-grammar.y"
    { (yyval.typelist) = new TypeList((yyvsp[(1) - (1)].ntype)); }
    break;

  case 113:
#line 476 "xi-grammar.y"
    { (yyval.typelist) = new TypeList((yyvsp[(1) - (3)].ntype), (yyvsp[(3) - (3)].typelist)); }
    break;

  case 114:
#line 480 "xi-grammar.y"
    { (yyval.chare) = new Chare(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 115:
#line 482 "xi-grammar.y"
    { (yyval.chare) = new MainChare(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 116:
#line 486 "xi-grammar.y"
    { (yyval.chare) = new Group(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 117:
#line 490 "xi-grammar.y"
    { (yyval.chare) = new NodeGroup(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 118:
#line 494 "xi-grammar.y"
    {/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",(yyvsp[(2) - (4)].strval));
			(yyval.ntype) = new NamedType(buf); 
		}
    break;

  case 119:
#line 500 "xi-grammar.y"
    { (yyval.ntype) = new NamedType((yyvsp[(2) - (3)].strval)); }
    break;

  case 120:
#line 504 "xi-grammar.y"
    {  (yyval.chare) = new Array(lineno, (yyvsp[(2) - (6)].cattr), (yyvsp[(3) - (6)].ntype), (yyvsp[(4) - (6)].ntype), (yyvsp[(5) - (6)].typelist), (yyvsp[(6) - (6)].mbrlist)); }
    break;

  case 121:
#line 506 "xi-grammar.y"
    {  (yyval.chare) = new Array(lineno, (yyvsp[(3) - (6)].cattr), (yyvsp[(2) - (6)].ntype), (yyvsp[(4) - (6)].ntype), (yyvsp[(5) - (6)].typelist), (yyvsp[(6) - (6)].mbrlist)); }
    break;

  case 122:
#line 510 "xi-grammar.y"
    { (yyval.chare) = new Chare(lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist));}
    break;

  case 123:
#line 512 "xi-grammar.y"
    { (yyval.chare) = new MainChare(lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 124:
#line 516 "xi-grammar.y"
    { (yyval.chare) = new Group(lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 125:
#line 520 "xi-grammar.y"
    { (yyval.chare) = new NodeGroup( lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 126:
#line 524 "xi-grammar.y"
    { (yyval.chare) = new Array( lineno, 0, (yyvsp[(2) - (5)].ntype), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 127:
#line 528 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[(3) - (4)].strval))); }
    break;

  case 128:
#line 530 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[(3) - (7)].strval)), (yyvsp[(5) - (7)].mvlist)); }
    break;

  case 129:
#line 534 "xi-grammar.y"
    { (yyval.type) = 0; }
    break;

  case 130:
#line 536 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(2) - (2)].type); }
    break;

  case 131:
#line 540 "xi-grammar.y"
    { (yyval.strval) = 0; }
    break;

  case 132:
#line 542 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(2) - (2)].strval); }
    break;

  case 133:
#line 544 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(2) - (2)].strval); }
    break;

  case 134:
#line 548 "xi-grammar.y"
    { (yyval.tvar) = new TType(new NamedType((yyvsp[(2) - (3)].strval)), (yyvsp[(3) - (3)].type)); }
    break;

  case 135:
#line 550 "xi-grammar.y"
    { (yyval.tvar) = new TFunc((yyvsp[(1) - (2)].ftype), (yyvsp[(2) - (2)].strval)); }
    break;

  case 136:
#line 552 "xi-grammar.y"
    { (yyval.tvar) = new TName((yyvsp[(1) - (3)].type), (yyvsp[(2) - (3)].strval), (yyvsp[(3) - (3)].strval)); }
    break;

  case 137:
#line 556 "xi-grammar.y"
    { (yyval.tvarlist) = new TVarList((yyvsp[(1) - (1)].tvar)); }
    break;

  case 138:
#line 558 "xi-grammar.y"
    { (yyval.tvarlist) = new TVarList((yyvsp[(1) - (3)].tvar), (yyvsp[(3) - (3)].tvarlist)); }
    break;

  case 139:
#line 562 "xi-grammar.y"
    { (yyval.tvarlist) = (yyvsp[(3) - (4)].tvarlist); }
    break;

  case 140:
#line 566 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 141:
#line 568 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 142:
#line 570 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 143:
#line 572 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 144:
#line 574 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].message)); (yyvsp[(2) - (2)].message)->setTemplate((yyval.templat)); }
    break;

  case 145:
#line 578 "xi-grammar.y"
    { (yyval.mbrlist) = 0; }
    break;

  case 146:
#line 580 "xi-grammar.y"
    { (yyval.mbrlist) = (yyvsp[(2) - (4)].mbrlist); }
    break;

  case 147:
#line 584 "xi-grammar.y"
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

  case 148:
#line 603 "xi-grammar.y"
    { (yyval.mbrlist) = new MemberList((yyvsp[(1) - (2)].member), (yyvsp[(2) - (2)].mbrlist)); }
    break;

  case 149:
#line 607 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (2)].readonly); }
    break;

  case 150:
#line 609 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (2)].readonly); }
    break;

  case 152:
#line 612 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (2)].member); }
    break;

  case 153:
#line 614 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(2) - (3)].pupable); }
    break;

  case 154:
#line 616 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(2) - (3)].includeFile); }
    break;

  case 155:
#line 618 "xi-grammar.y"
    { (yyval.member) = new ClassDeclaration(lineno,(yyvsp[(2) - (3)].strval)); }
    break;

  case 156:
#line 622 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 1); }
    break;

  case 157:
#line 624 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 1); }
    break;

  case 158:
#line 626 "xi-grammar.y"
    { printf("Warning: deprecated use of initcall. Use initnode or initproc instead.\n"); 
		  (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 1); }
    break;

  case 159:
#line 629 "xi-grammar.y"
    { printf("Warning: deprecated use of initcall. Use initnode or initproc instead.\n");
		  (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 1); }
    break;

  case 160:
#line 634 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 0); }
    break;

  case 161:
#line 636 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 0); }
    break;

  case 162:
#line 638 "xi-grammar.y"
    {
                  InitCall* rtn = new InitCall(lineno, (yyvsp[(6) - (9)].strval), 0);
                  rtn->setAccel();
                  (yyval.member) = rtn;
		}
    break;

  case 163:
#line 646 "xi-grammar.y"
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[(1) - (1)].ntype),0); }
    break;

  case 164:
#line 648 "xi-grammar.y"
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[(1) - (3)].ntype),(yyvsp[(3) - (3)].pupable)); }
    break;

  case 165:
#line 651 "xi-grammar.y"
    { (yyval.includeFile) = new IncludeFile(lineno,(yyvsp[(1) - (1)].strval)); }
    break;

  case 166:
#line 655 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (2)].entry); }
    break;

  case 167:
#line 657 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].member); }
    break;

  case 168:
#line 661 "xi-grammar.y"
    { 
		  if ((yyvsp[(7) - (7)].sc) != 0) { 
		    (yyvsp[(7) - (7)].sc)->con1 = new SdagConstruct(SIDENT, (yyvsp[(4) - (7)].strval));
  		    if ((yyvsp[(5) - (7)].plist) != 0)
                      (yyvsp[(7) - (7)].sc)->param = new ParamList((yyvsp[(5) - (7)].plist));
 		    else 
 	 	      (yyvsp[(7) - (7)].sc)->param = new ParamList(new Parameter(0, new BuiltinType("void")));
                  }
		  (yyval.entry) = new Entry(lineno, (yyvsp[(2) - (7)].intval), (yyvsp[(3) - (7)].type), (yyvsp[(4) - (7)].strval), (yyvsp[(5) - (7)].plist), (yyvsp[(6) - (7)].val), (yyvsp[(7) - (7)].sc), 0, 0); 
		}
    break;

  case 169:
#line 672 "xi-grammar.y"
    { 
		  if ((yyvsp[(5) - (5)].sc) != 0) {
		    (yyvsp[(5) - (5)].sc)->con1 = new SdagConstruct(SIDENT, (yyvsp[(3) - (5)].strval));
		    if ((yyvsp[(4) - (5)].plist) != 0)
                      (yyvsp[(5) - (5)].sc)->param = new ParamList((yyvsp[(4) - (5)].plist));
		    else
                      (yyvsp[(5) - (5)].sc)->param = new ParamList(new Parameter(0, new BuiltinType("void")));
                  }
		  (yyval.entry) = new Entry(lineno, (yyvsp[(2) - (5)].intval),     0, (yyvsp[(3) - (5)].strval), (yyvsp[(4) - (5)].plist),  0, (yyvsp[(5) - (5)].sc), 0, 0); 
		}
    break;

  case 170:
#line 683 "xi-grammar.y"
    {
                  int attribs = SACCEL;
                  char* name = (yyvsp[(6) - (12)].strval);
                  ParamList* paramList = (yyvsp[(7) - (12)].plist);
                  ParamList* accelParamList = (yyvsp[(8) - (12)].plist);
		  XStr* codeBody = new XStr((yyvsp[(10) - (12)].strval));
                  char* callbackName = (yyvsp[(12) - (12)].strval);

                  (yyval.entry) = new Entry(lineno, attribs, new BuiltinType("void"), name, paramList,
                                 0, 0, 0, 0, 0
                                );
                  (yyval.entry)->setAccelParam(accelParamList);
                  (yyval.entry)->setAccelCodeBody(codeBody);
                  (yyval.entry)->setAccelCallbackName(new XStr(callbackName));
                }
    break;

  case 171:
#line 701 "xi-grammar.y"
    { (yyval.accelBlock) = new AccelBlock(lineno, new XStr((yyvsp[(3) - (5)].strval))); }
    break;

  case 172:
#line 703 "xi-grammar.y"
    { (yyval.accelBlock) = new AccelBlock(lineno, NULL); }
    break;

  case 173:
#line 707 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("void"); }
    break;

  case 174:
#line 709 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 175:
#line 713 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 176:
#line 715 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(2) - (3)].intval); }
    break;

  case 177:
#line 719 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (1)].intval); }
    break;

  case 178:
#line 721 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (3)].intval) | (yyvsp[(3) - (3)].intval); }
    break;

  case 179:
#line 725 "xi-grammar.y"
    { (yyval.intval) = STHREADED; }
    break;

  case 180:
#line 727 "xi-grammar.y"
    { (yyval.intval) = SSYNC; }
    break;

  case 181:
#line 729 "xi-grammar.y"
    { (yyval.intval) = SIGET; }
    break;

  case 182:
#line 731 "xi-grammar.y"
    { (yyval.intval) = SLOCKED; }
    break;

  case 183:
#line 733 "xi-grammar.y"
    { (yyval.intval) = SCREATEHERE; }
    break;

  case 184:
#line 735 "xi-grammar.y"
    { (yyval.intval) = SCREATEHOME; }
    break;

  case 185:
#line 737 "xi-grammar.y"
    { (yyval.intval) = SNOKEEP; }
    break;

  case 186:
#line 739 "xi-grammar.y"
    { (yyval.intval) = SNOTRACE; }
    break;

  case 187:
#line 741 "xi-grammar.y"
    { (yyval.intval) = SIMMEDIATE; }
    break;

  case 188:
#line 743 "xi-grammar.y"
    { (yyval.intval) = SSKIPSCHED; }
    break;

  case 189:
#line 745 "xi-grammar.y"
    { (yyval.intval) = SINLINE; }
    break;

  case 190:
#line 747 "xi-grammar.y"
    { (yyval.intval) = SLOCAL; }
    break;

  case 191:
#line 749 "xi-grammar.y"
    { (yyval.intval) = SPYTHON; }
    break;

  case 192:
#line 753 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 193:
#line 755 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 194:
#line 757 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 195:
#line 761 "xi-grammar.y"
    { (yyval.strval) = ""; }
    break;

  case 196:
#line 763 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 197:
#line 765 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[(1) - (3)].strval))+strlen((yyvsp[(3) - (3)].strval))+3];
			sprintf(tmp,"%s, %s", (yyvsp[(1) - (3)].strval), (yyvsp[(3) - (3)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 198:
#line 773 "xi-grammar.y"
    { (yyval.strval) = ""; }
    break;

  case 199:
#line 775 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 200:
#line 777 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s[%s]%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 201:
#line 783 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s{%s}%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 202:
#line 789 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s(%s)%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 203:
#line 795 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(2) - (4)].strval))+strlen((yyvsp[(4) - (4)].strval))+3];
			sprintf(tmp,"(%s)%s", (yyvsp[(2) - (4)].strval), (yyvsp[(4) - (4)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 204:
#line 803 "xi-grammar.y"
    {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			(yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].type),(yyvsp[(2) - (3)].strval));
		}
    break;

  case 205:
#line 810 "xi-grammar.y"
    { 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			(yyval.intval) = 0;
		}
    break;

  case 206:
#line 818 "xi-grammar.y"
    { 
			in_braces=0;
			(yyval.intval) = 0;
		}
    break;

  case 207:
#line 825 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (1)].type));}
    break;

  case 208:
#line 827 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].type),(yyvsp[(2) - (3)].strval)); (yyval.pname)->setConditional((yyvsp[(3) - (3)].intval)); }
    break;

  case 209:
#line 829 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (4)].type),(yyvsp[(2) - (4)].strval),0,(yyvsp[(4) - (4)].val));}
    break;

  case 210:
#line 831 "xi-grammar.y"
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].pname)->getType(), (yyvsp[(1) - (3)].pname)->getName() ,(yyvsp[(2) - (3)].strval));
		}
    break;

  case 211:
#line 837 "xi-grammar.y"
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READONLY; }
    break;

  case 212:
#line 838 "xi-grammar.y"
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READWRITE; }
    break;

  case 213:
#line 839 "xi-grammar.y"
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_WRITEONLY; }
    break;

  case 214:
#line 842 "xi-grammar.y"
    { (yyval.xstrptr) = new XStr((yyvsp[(1) - (1)].strval)); }
    break;

  case 215:
#line 843 "xi-grammar.y"
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "->" << (yyvsp[(4) - (4)].strval); }
    break;

  case 216:
#line 844 "xi-grammar.y"
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[(1) - (3)].xstrptr)) << "." << (yyvsp[(3) - (3)].strval); }
    break;

  case 217:
#line 846 "xi-grammar.y"
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "[" << *((yyvsp[(3) - (4)].xstrptr)) << "]";
                  delete (yyvsp[(1) - (4)].xstrptr);
                  delete (yyvsp[(3) - (4)].xstrptr);
                }
    break;

  case 218:
#line 853 "xi-grammar.y"
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "[" << (yyvsp[(3) - (4)].strval) << "]";
                  delete (yyvsp[(1) - (4)].xstrptr);
                }
    break;

  case 219:
#line 859 "xi-grammar.y"
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "(" << *((yyvsp[(3) - (4)].xstrptr)) << ")";
                  delete (yyvsp[(1) - (4)].xstrptr);
                  delete (yyvsp[(3) - (4)].xstrptr);
                }
    break;

  case 220:
#line 868 "xi-grammar.y"
    {
                  in_bracket = 0;
                  (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].pname)->getType(), (yyvsp[(1) - (3)].pname)->getName(), (yyvsp[(2) - (3)].strval));
                }
    break;

  case 221:
#line 875 "xi-grammar.y"
    {
                  (yyval.pname) = new Parameter(lineno, (yyvsp[(3) - (7)].type), (yyvsp[(4) - (7)].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[(6) - (7)].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[(1) - (7)].intval));
                }
    break;

  case 222:
#line 881 "xi-grammar.y"
    {
		  (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (5)].type), (yyvsp[(2) - (5)].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[(4) - (5)].xstrptr));
                  (yyval.pname)->setAccelBufferType(Parameter::ACCEL_BUFFER_TYPE_READWRITE);
		}
    break;

  case 223:
#line 887 "xi-grammar.y"
    {
                  (yyval.pname) = (yyvsp[(3) - (6)].pname);
                  (yyval.pname)->setAccelInstName((yyvsp[(5) - (6)].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[(1) - (6)].intval));
		}
    break;

  case 224:
#line 895 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (1)].pname)); }
    break;

  case 225:
#line 897 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (3)].pname),(yyvsp[(3) - (3)].plist)); }
    break;

  case 226:
#line 901 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (1)].pname)); }
    break;

  case 227:
#line 903 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (3)].pname),(yyvsp[(3) - (3)].plist)); }
    break;

  case 228:
#line 907 "xi-grammar.y"
    { (yyval.plist) = (yyvsp[(2) - (3)].plist); }
    break;

  case 229:
#line 909 "xi-grammar.y"
    { (yyval.plist) = 0; }
    break;

  case 230:
#line 913 "xi-grammar.y"
    { (yyval.plist) = (yyvsp[(2) - (3)].plist); }
    break;

  case 231:
#line 915 "xi-grammar.y"
    { (yyval.plist) = 0; }
    break;

  case 232:
#line 919 "xi-grammar.y"
    { (yyval.val) = 0; }
    break;

  case 233:
#line 921 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(3) - (3)].strval)); }
    break;

  case 234:
#line 925 "xi-grammar.y"
    { (yyval.sc) = 0; }
    break;

  case 235:
#line 927 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSDAGENTRY, (yyvsp[(1) - (1)].sc)); }
    break;

  case 236:
#line 929 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSDAGENTRY, (yyvsp[(2) - (3)].sc)); }
    break;

  case 237:
#line 933 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSLIST, (yyvsp[(1) - (1)].sc)); }
    break;

  case 238:
#line 935 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSLIST, (yyvsp[(1) - (2)].sc), (yyvsp[(2) - (2)].sc));  }
    break;

  case 239:
#line 939 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SOLIST, (yyvsp[(1) - (1)].sc)); }
    break;

  case 240:
#line 941 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SOLIST, (yyvsp[(1) - (2)].sc), (yyvsp[(2) - (2)].sc)); }
    break;

  case 241:
#line 945 "xi-grammar.y"
    { (yyval.sc) = 0; }
    break;

  case 242:
#line 947 "xi-grammar.y"
    { (yyval.sc) = (yyvsp[(3) - (4)].sc); }
    break;

  case 243:
#line 951 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SPUBLISHES, new SdagConstruct(SIDENT, (yyvsp[(1) - (1)].strval))); }
    break;

  case 244:
#line 953 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SPUBLISHES, new SdagConstruct(SIDENT, (yyvsp[(1) - (3)].strval)), (yyvsp[(3) - (3)].sc));  }
    break;

  case 245:
#line 957 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 246:
#line 959 "xi-grammar.y"
    { (yyval.strval) = 0; }
    break;

  case 247:
#line 963 "xi-grammar.y"
    { RemoveSdagComments((yyvsp[(4) - (6)].strval));
		   (yyval.sc) = new SdagConstruct(SATOMIC, new XStr((yyvsp[(4) - (6)].strval)), (yyvsp[(6) - (6)].sc), 0,0,0,0, 0 ); 
		   if ((yyvsp[(2) - (6)].strval)) { (yyvsp[(2) - (6)].strval)[strlen((yyvsp[(2) - (6)].strval))-1]=0; (yyval.sc)->traceName = new XStr((yyvsp[(2) - (6)].strval)+1); }
		 }
    break;

  case 248:
#line 968 "xi-grammar.y"
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

  case 249:
#line 982 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SWHEN, 0, 0,0,0,0,0,(yyvsp[(2) - (4)].entrylist)); }
    break;

  case 250:
#line 984 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SWHEN, 0, 0, 0,0,0, (yyvsp[(3) - (3)].sc), (yyvsp[(2) - (3)].entrylist)); }
    break;

  case 251:
#line 986 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SWHEN, 0, 0, 0,0,0, (yyvsp[(4) - (5)].sc), (yyvsp[(2) - (5)].entrylist)); }
    break;

  case 252:
#line 988 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SOVERLAP,0, 0,0,0,0,(yyvsp[(3) - (4)].sc), 0); }
    break;

  case 253:
#line 990 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFOR, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (11)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(5) - (11)].strval)),
		             new SdagConstruct(SINT_EXPR, (yyvsp[(7) - (11)].strval)), 0, (yyvsp[(10) - (11)].sc), 0); }
    break;

  case 254:
#line 993 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFOR, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (9)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(5) - (9)].strval)), 
		         new SdagConstruct(SINT_EXPR, (yyvsp[(7) - (9)].strval)), 0, (yyvsp[(9) - (9)].sc), 0); }
    break;

  case 255:
#line 996 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFORALL, 0, new SdagConstruct(SIDENT, (yyvsp[(3) - (12)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(6) - (12)].strval)), 
		             new SdagConstruct(SINT_EXPR, (yyvsp[(8) - (12)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(10) - (12)].strval)), (yyvsp[(12) - (12)].sc), 0); }
    break;

  case 256:
#line 999 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFORALL, 0, new SdagConstruct(SIDENT, (yyvsp[(3) - (14)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(6) - (14)].strval)), 
		                 new SdagConstruct(SINT_EXPR, (yyvsp[(8) - (14)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(10) - (14)].strval)), (yyvsp[(13) - (14)].sc), 0); }
    break;

  case 257:
#line 1002 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (6)].strval)), (yyvsp[(6) - (6)].sc),0,0,(yyvsp[(5) - (6)].sc),0); }
    break;

  case 258:
#line 1004 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (8)].strval)), (yyvsp[(8) - (8)].sc),0,0,(yyvsp[(6) - (8)].sc),0); }
    break;

  case 259:
#line 1006 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (5)].strval)), 0,0,0,(yyvsp[(5) - (5)].sc),0); }
    break;

  case 260:
#line 1008 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SWHILE, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (7)].strval)), 0,0,0,(yyvsp[(6) - (7)].sc),0); }
    break;

  case 261:
#line 1010 "xi-grammar.y"
    { (yyval.sc) = (yyvsp[(2) - (3)].sc); }
    break;

  case 262:
#line 1012 "xi-grammar.y"
    { RemoveSdagComments((yyvsp[(2) - (3)].strval));
                   (yyval.sc) = new SdagConstruct(SATOMIC, new XStr((yyvsp[(2) - (3)].strval)), NULL, 0,0,0,0, 0 );
                 }
    break;

  case 263:
#line 1018 "xi-grammar.y"
    { (yyval.sc) = 0; }
    break;

  case 264:
#line 1020 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SELSE, 0,0,0,0,0, (yyvsp[(2) - (2)].sc),0); }
    break;

  case 265:
#line 1022 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SELSE, 0,0,0,0,0, (yyvsp[(3) - (4)].sc),0); }
    break;

  case 266:
#line 1025 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFORWARD, new SdagConstruct(SIDENT, (yyvsp[(1) - (1)].strval))); }
    break;

  case 267:
#line 1027 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFORWARD, new SdagConstruct(SIDENT, (yyvsp[(1) - (3)].strval)), (yyvsp[(3) - (3)].sc));  }
    break;

  case 268:
#line 1031 "xi-grammar.y"
    { in_int_expr = 0; (yyval.intval) = 0; }
    break;

  case 269:
#line 1035 "xi-grammar.y"
    { in_int_expr = 1; (yyval.intval) = 0; }
    break;

  case 270:
#line 1039 "xi-grammar.y"
    { 
		  if ((yyvsp[(2) - (2)].plist) != 0)
		     (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[(1) - (2)].strval), (yyvsp[(2) - (2)].plist), 0, 0, 0, 0); 
		  else
		     (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[(1) - (2)].strval), 
				new ParamList(new Parameter(0, new BuiltinType("void"))), 0, 0, 0, 0); 
		}
    break;

  case 271:
#line 1047 "xi-grammar.y"
    { if ((yyvsp[(5) - (5)].plist) != 0)
		    (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[(1) - (5)].strval), (yyvsp[(5) - (5)].plist), 0, 0, (yyvsp[(3) - (5)].strval), 0); 
		  else
		    (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[(1) - (5)].strval), new ParamList(new Parameter(0, new BuiltinType("void"))), 0, 0, (yyvsp[(3) - (5)].strval), 0); 
		}
    break;

  case 272:
#line 1055 "xi-grammar.y"
    { (yyval.entrylist) = new EntryList((yyvsp[(1) - (1)].entry)); }
    break;

  case 273:
#line 1057 "xi-grammar.y"
    { (yyval.entrylist) = new EntryList((yyvsp[(1) - (3)].entry),(yyvsp[(3) - (3)].entrylist)); }
    break;

  case 274:
#line 1061 "xi-grammar.y"
    { in_bracket=1; }
    break;

  case 275:
#line 1064 "xi-grammar.y"
    { in_bracket=0; }
    break;

  case 276:
#line 1068 "xi-grammar.y"
    { if (!macroDefined((yyvsp[(2) - (2)].strval), 1)) in_comment = 1; }
    break;

  case 277:
#line 1072 "xi-grammar.y"
    { if (!macroDefined((yyvsp[(2) - (2)].strval), 0)) in_comment = 1; }
    break;


/* Line 1267 of yacc.c.  */
#line 3615 "y.tab.c"
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


#line 1075 "xi-grammar.y"

void yyerror(const char *mesg)
{
  cout << cur_file<<":"<<lineno<<": Charmxi syntax error> " << mesg << endl;
  // return 0;
}

