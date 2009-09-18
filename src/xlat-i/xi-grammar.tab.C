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
     ACCELBLOCK = 326,
     MEMCRITICAL = 327
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




/* Copy the first part of user declarations.  */
#line 2 "xi-grammar.y"

#include <iostream>
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
#line 294 "y.tab.c"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 307 "y.tab.c"

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
#define YYLAST   705

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  89
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  110
/* YYNRULES -- Number of rules.  */
#define YYNRULES  278
/* YYNRULES -- Number of states.  */
#define YYNSTATES  575

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
     615,   617,   620,   622,   624,   626,   628,   629,   631,   635,
     636,   638,   644,   650,   656,   661,   665,   667,   669,   671,
     675,   680,   684,   686,   688,   690,   692,   697,   701,   706,
     711,   716,   720,   728,   734,   741,   743,   747,   749,   753,
     757,   760,   764,   767,   768,   772,   773,   775,   779,   781,
     784,   786,   789,   790,   795,   797,   801,   803,   804,   811,
     820,   825,   829,   835,   840,   852,   862,   875,   890,   897,
     906,   912,   920,   924,   928,   929,   932,   937,   939,   943,
     945,   947,   950,   956,   958,   962,   964,   966,   969
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
      90,     0,    -1,    91,    -1,    -1,    96,    91,    -1,    -1,
       5,    -1,    -1,    73,    -1,    55,    -1,    55,    -1,    95,
      74,    74,    55,    -1,     3,    94,    97,    -1,     4,    94,
      97,    -1,    73,    -1,    75,    98,    76,    93,    -1,    -1,
      99,    98,    -1,    92,    75,    98,    76,    93,    -1,    53,
      94,    75,    98,    76,    -1,    54,    53,    95,    73,    -1,
      54,    95,    73,    -1,    92,    96,    -1,    92,   154,    -1,
      92,   133,    73,    -1,    92,   136,    -1,    92,   137,    -1,
      92,   138,    -1,    92,   140,    -1,    92,   151,    -1,   197,
      -1,   198,    -1,   161,    -1,   112,    -1,    56,    -1,    57,
      -1,   100,    -1,   100,    77,   101,    -1,    -1,   101,    -1,
      -1,    78,   102,    79,    -1,    61,    -1,    62,    -1,    63,
      -1,    64,    -1,    67,    61,    -1,    67,    62,    -1,    67,
      62,    61,    -1,    67,    62,    62,    -1,    67,    63,    -1,
      67,    64,    -1,    62,    62,    -1,    65,    -1,    66,    -1,
      62,    66,    -1,    35,    -1,    94,   103,    -1,    95,   103,
      -1,   104,    -1,   106,    -1,   107,    80,    -1,   108,    80,
      -1,   109,    80,    -1,   111,    81,    80,    94,    82,    81,
     177,    82,    -1,   107,    -1,   108,    -1,   109,    -1,   110,
      -1,    36,   111,    -1,   111,    36,    -1,   111,    83,    -1,
     111,    -1,    56,    -1,    95,    -1,    84,   113,    85,    -1,
      -1,   114,   115,    -1,     6,   112,    95,   115,    -1,     6,
      16,   107,    80,    94,    -1,    -1,    35,    -1,    -1,    84,
     120,    85,    -1,   121,    -1,   121,    77,   120,    -1,    37,
      -1,    38,    -1,    -1,    84,   123,    85,    -1,   128,    -1,
     128,    77,   123,    -1,    -1,    57,    -1,    51,    -1,    -1,
      84,   127,    85,    -1,   125,    -1,   125,    77,   127,    -1,
      30,    -1,    51,    -1,    -1,    17,    -1,    -1,    84,    85,
      -1,   129,   112,    94,   130,    73,    -1,   131,    -1,   131,
     132,    -1,    16,   119,   105,    -1,    16,   119,   105,    75,
     132,    76,    -1,    -1,    74,   135,    -1,   106,    -1,   106,
      77,   135,    -1,    11,   122,   105,   134,   152,    -1,    12,
     122,   105,   134,   152,    -1,    13,   122,   105,   134,   152,
      -1,    14,   122,   105,   134,   152,    -1,    84,    56,    94,
      85,    -1,    84,    94,    85,    -1,    15,   126,   139,   105,
     134,   152,    -1,    15,   139,   126,   105,   134,   152,    -1,
      11,   122,    94,   134,   152,    -1,    12,   122,    94,   134,
     152,    -1,    13,   122,    94,   134,   152,    -1,    14,   122,
      94,   134,   152,    -1,    15,   139,    94,   134,   152,    -1,
      16,   119,    94,    73,    -1,    16,   119,    94,    75,   132,
      76,    73,    -1,    -1,    86,   112,    -1,    -1,    86,    56,
      -1,    86,    57,    -1,    18,    94,   146,    -1,   110,   147,
      -1,   112,    94,   147,    -1,   148,    -1,   148,    77,   149,
      -1,    22,    78,   149,    79,    -1,   150,   141,    -1,   150,
     142,    -1,   150,   143,    -1,   150,   144,    -1,   150,   145,
      -1,    73,    -1,    75,   153,    76,    93,    -1,    -1,   159,
     153,    -1,   116,    73,    -1,   117,    73,    -1,   156,    73,
      -1,   155,    73,    -1,    10,   157,    73,    -1,    19,   158,
      73,    -1,    18,    94,    73,    -1,     8,   118,    95,    -1,
       8,   118,    95,    81,   118,    82,    -1,     7,   118,    95,
      -1,     7,   118,    95,    81,   118,    82,    -1,     9,   118,
      95,    -1,     9,   118,    95,    81,   118,    82,    -1,     9,
      84,    68,    85,   118,    95,    81,   118,    82,    -1,   106,
      -1,   106,    77,   157,    -1,    57,    -1,   160,    73,    -1,
     154,    -1,    39,   163,   162,    94,   179,   181,   182,    -1,
      39,   163,    94,   179,   182,    -1,    39,    84,    68,    85,
      35,    94,   179,   180,   170,   168,   171,    94,    -1,    71,
     170,   168,   171,    73,    -1,    71,    73,    -1,    35,    -1,
     108,    -1,    -1,    84,   164,    85,    -1,   165,    -1,   165,
      77,   164,    -1,    21,    -1,    23,    -1,    24,    -1,    25,
      -1,    31,    -1,    32,    -1,    33,    -1,    34,    -1,    26,
      -1,    27,    -1,    28,    -1,    52,    -1,    51,   124,    -1,
      72,    -1,    57,    -1,    56,    -1,    95,    -1,    -1,    58,
      -1,    58,    77,   167,    -1,    -1,    58,    -1,    58,    84,
     168,    85,   168,    -1,    58,    75,   168,    76,   168,    -1,
      58,    81,   167,    82,   168,    -1,    81,   168,    82,   168,
      -1,   112,    94,    84,    -1,    75,    -1,    76,    -1,   112,
      -1,   112,    94,   129,    -1,   112,    94,    86,   166,    -1,
     169,   168,    85,    -1,     6,    -1,    69,    -1,    70,    -1,
      94,    -1,   174,    87,    79,    94,    -1,   174,    88,    94,
      -1,   174,    84,   174,    85,    -1,   174,    84,    56,    85,
      -1,   174,    81,   174,    82,    -1,   169,   168,    85,    -1,
     173,    74,   112,    94,    78,   174,    79,    -1,   112,    94,
      78,   174,    79,    -1,   173,    74,   175,    78,   174,    79,
      -1,   172,    -1,   172,    77,   177,    -1,   176,    -1,   176,
      77,   178,    -1,    81,   177,    82,    -1,    81,    82,    -1,
      84,   178,    85,    -1,    84,    85,    -1,    -1,    20,    86,
      56,    -1,    -1,   188,    -1,    75,   183,    76,    -1,   188,
      -1,   188,   183,    -1,   188,    -1,   188,   183,    -1,    -1,
      50,    81,   186,    82,    -1,    55,    -1,    55,    77,   186,
      -1,    57,    -1,    -1,    45,   187,   170,   168,   171,   185,
      -1,    49,    81,    55,   179,    82,   170,   168,    76,    -1,
      43,   194,    75,    76,    -1,    43,   194,   188,    -1,    43,
     194,    75,   183,    76,    -1,    44,    75,   184,    76,    -1,
      40,   192,   168,    73,   168,    73,   168,   191,    75,   183,
      76,    -1,    40,   192,   168,    73,   168,    73,   168,   191,
     188,    -1,    41,    84,    55,    85,   192,   168,    74,   168,
      77,   168,   191,   188,    -1,    41,    84,    55,    85,   192,
     168,    74,   168,    77,   168,   191,    75,   183,    76,    -1,
      47,   192,   168,   191,   188,   189,    -1,    47,   192,   168,
     191,    75,   183,    76,   189,    -1,    42,   192,   168,   191,
     188,    -1,    42,   192,   168,   191,    75,   183,    76,    -1,
      46,   190,    73,    -1,   170,   168,   171,    -1,    -1,    48,
     188,    -1,    48,    75,   183,    76,    -1,    55,    -1,    55,
      77,   190,    -1,    82,    -1,    81,    -1,    55,   179,    -1,
      55,   195,   168,   196,   179,    -1,   193,    -1,   193,    77,
     194,    -1,    84,    -1,    85,    -1,    59,    94,    -1,    60,
      94,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   148,   148,   153,   156,   161,   162,   167,   168,   172,
     176,   178,   186,   190,   197,   199,   204,   205,   209,   211,
     213,   215,   217,   219,   221,   223,   225,   227,   229,   231,
     233,   235,   237,   241,   243,   245,   249,   251,   256,   257,
     262,   263,   267,   269,   271,   273,   275,   277,   279,   281,
     283,   285,   287,   289,   291,   293,   295,   299,   300,   307,
     309,   313,   317,   319,   323,   327,   329,   331,   333,   336,
     338,   342,   344,   348,   350,   354,   359,   360,   364,   368,
     373,   374,   379,   380,   390,   392,   396,   398,   403,   404,
     408,   410,   415,   416,   420,   425,   426,   430,   432,   436,
     438,   443,   444,   448,   449,   452,   456,   458,   462,   464,
     469,   470,   474,   476,   480,   482,   486,   490,   494,   500,
     504,   506,   510,   512,   516,   520,   524,   528,   530,   535,
     536,   541,   542,   544,   548,   550,   552,   556,   558,   562,
     566,   568,   570,   572,   574,   578,   580,   585,   603,   607,
     609,   611,   612,   614,   616,   618,   622,   624,   626,   629,
     634,   636,   638,   646,   648,   651,   655,   657,   661,   672,
     683,   701,   703,   707,   709,   714,   715,   719,   721,   725,
     727,   729,   731,   733,   735,   737,   739,   741,   743,   745,
     747,   749,   751,   755,   757,   759,   764,   765,   767,   776,
     777,   779,   785,   791,   797,   805,   812,   820,   827,   829,
     831,   833,   840,   841,   842,   845,   846,   847,   848,   855,
     861,   870,   877,   883,   889,   897,   899,   903,   905,   909,
     911,   915,   917,   922,   923,   928,   929,   931,   935,   937,
     941,   943,   948,   949,   953,   955,   959,   962,   965,   970,
     984,   986,   988,   990,   992,   995,   998,  1001,  1004,  1006,
    1008,  1010,  1012,  1014,  1021,  1022,  1024,  1027,  1029,  1033,
    1037,  1041,  1049,  1057,  1059,  1063,  1066,  1070,  1074
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
  "ACCELBLOCK", "MEMCRITICAL", "';'", "':'", "'{'", "'}'", "','", "'<'",
  "'>'", "'*'", "'('", "')'", "'&'", "'['", "']'", "'='", "'-'", "'.'",
  "$accept", "File", "ModuleEList", "OptExtern", "OptSemiColon", "Name",
  "QualName", "Module", "ConstructEList", "ConstructList", "Construct",
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
     325,   326,   327,    59,    58,   123,   125,    44,    60,    62,
      42,    40,    41,    38,    91,    93,    61,    45,    46
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    89,    90,    91,    91,    92,    92,    93,    93,    94,
      95,    95,    96,    96,    97,    97,    98,    98,    99,    99,
      99,    99,    99,    99,    99,    99,    99,    99,    99,    99,
      99,    99,    99,   100,   100,   100,   101,   101,   102,   102,
     103,   103,   104,   104,   104,   104,   104,   104,   104,   104,
     104,   104,   104,   104,   104,   104,   104,   105,   106,   107,
     107,   108,   109,   109,   110,   111,   111,   111,   111,   111,
     111,   112,   112,   113,   113,   114,   115,   115,   116,   117,
     118,   118,   119,   119,   120,   120,   121,   121,   122,   122,
     123,   123,   124,   124,   125,   126,   126,   127,   127,   128,
     128,   129,   129,   130,   130,   131,   132,   132,   133,   133,
     134,   134,   135,   135,   136,   136,   137,   138,   139,   139,
     140,   140,   141,   141,   142,   143,   144,   145,   145,   146,
     146,   147,   147,   147,   148,   148,   148,   149,   149,   150,
     151,   151,   151,   151,   151,   152,   152,   153,   153,   154,
     154,   154,   154,   154,   154,   154,   155,   155,   155,   155,
     156,   156,   156,   157,   157,   158,   159,   159,   160,   160,
     160,   161,   161,   162,   162,   163,   163,   164,   164,   165,
     165,   165,   165,   165,   165,   165,   165,   165,   165,   165,
     165,   165,   165,   166,   166,   166,   167,   167,   167,   168,
     168,   168,   168,   168,   168,   169,   170,   171,   172,   172,
     172,   172,   173,   173,   173,   174,   174,   174,   174,   174,
     174,   175,   176,   176,   176,   177,   177,   178,   178,   179,
     179,   180,   180,   181,   181,   182,   182,   182,   183,   183,
     184,   184,   185,   185,   186,   186,   187,   187,   188,   188,
     188,   188,   188,   188,   188,   188,   188,   188,   188,   188,
     188,   188,   188,   188,   189,   189,   189,   190,   190,   191,
     192,   193,   193,   194,   194,   195,   196,   197,   198
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
       1,     2,     1,     1,     1,     1,     0,     1,     3,     0,
       1,     5,     5,     5,     4,     3,     1,     1,     1,     3,
       4,     3,     1,     1,     1,     1,     4,     3,     4,     4,
       4,     3,     7,     5,     6,     1,     3,     1,     3,     3,
       2,     3,     2,     0,     3,     0,     1,     3,     1,     2,
       1,     2,     0,     4,     1,     3,     1,     0,     6,     8,
       4,     3,     5,     4,    11,     9,    12,    14,     6,     8,
       5,     7,     3,     3,     0,     2,     4,     1,     3,     1,
       1,     2,     5,     1,     3,     1,     1,     2,     2
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       3,     0,     0,     0,     2,     3,     9,     0,     0,     1,
       4,    14,     5,    12,    13,     6,     0,     0,     0,     0,
       0,     0,     0,     5,    32,    30,    31,     0,     0,    10,
       0,   277,   278,   172,   206,   199,     0,    80,    80,    80,
       0,    88,    88,    88,    88,     0,    82,     0,     0,     0,
       5,    22,     0,     0,     0,    25,    26,    27,    28,     0,
      29,    23,     0,     0,     7,    17,     5,     0,    21,     0,
     200,   199,     0,     0,    56,     0,    42,    43,    44,    45,
      53,    54,     0,    40,    59,    60,    65,    66,    67,    68,
      72,     0,    81,     0,     0,     0,     0,   163,     0,     0,
       0,     0,     0,     0,     0,     0,    95,     0,     0,     0,
     165,     0,     0,     0,   149,   150,    24,    88,    88,    88,
      88,     0,    82,   140,   141,   142,   143,   144,   152,   151,
       8,    15,     0,    20,     0,   199,   196,   199,     0,   207,
       0,     0,    69,    52,    55,    46,    47,    50,    51,    38,
      58,    61,    62,    63,    70,     0,    71,    76,   158,   156,
       0,   160,     0,   153,    99,   100,     0,    90,    40,   110,
     110,   110,   110,    94,     0,     0,    97,     0,     0,     0,
       0,     0,    86,    87,     0,    84,   108,   155,   154,     0,
      68,     0,   137,     0,     7,     0,     0,     0,     0,     0,
       0,    19,    11,     0,   197,     0,     0,   199,   171,     0,
      48,    49,    34,    35,    36,    39,     0,    33,     0,     0,
      76,    78,    80,    80,    80,    80,   164,    89,     0,    57,
       0,     0,     0,     0,     0,     0,   119,     0,    96,   110,
     110,    83,     0,   101,   129,     0,   135,   131,     0,   139,
      18,   110,   110,   110,   110,   110,     0,   199,   196,   199,
     199,   204,    79,     0,    41,     0,    73,    74,     0,    77,
       0,     0,     0,     0,    91,   112,   111,   145,   147,   114,
     115,   116,   117,   118,    98,     0,     0,    85,   102,     0,
     101,     0,     0,   134,   132,   133,   136,   138,     0,     0,
       0,     0,     0,   127,   101,   202,   198,   203,   201,    37,
       0,    75,   159,   157,     0,   161,     0,   175,     0,   167,
     147,     0,   120,   121,     0,   107,   109,   130,   122,   123,
     124,   125,   126,     0,     0,    80,   113,     0,     0,     7,
     148,   166,   103,     0,   208,   199,   225,     0,     0,   179,
     180,   181,   182,   187,   188,   189,   183,   184,   185,   186,
      92,   190,     0,   192,     0,   177,    56,    10,     0,     0,
     174,     0,   146,     0,     0,   128,   101,     0,     0,    64,
     162,    93,   191,     0,   176,     0,     0,   235,     0,   104,
     105,   205,     0,   209,   211,   226,     0,   178,   230,     0,
       0,     0,     0,     0,     0,   247,     0,     0,     0,   206,
     199,   169,   236,   233,   194,   193,   195,   210,     0,   229,
     270,   199,     0,   199,     0,   273,     0,     0,   246,     0,
     267,     0,   199,     0,     0,   238,     0,     0,   235,     0,
       0,     0,     0,   275,   271,   199,     0,   206,   251,     0,
     240,   199,     0,   262,     0,     0,   237,   239,   263,     0,
     168,     0,     0,   199,     0,   269,     0,     0,   274,   250,
       0,   253,   241,     0,   268,     0,     0,   234,   212,   213,
     214,   232,     0,     0,   227,     0,   199,     0,   199,   206,
     260,   276,     0,   252,   242,   206,   264,     0,     0,     0,
       0,   231,     0,   199,     0,     0,   272,     0,   248,     0,
       0,   258,   199,     0,     0,   199,     0,   228,     0,     0,
     199,   261,     0,   264,   206,   265,     0,   215,     0,     0,
       0,     0,   170,     0,     0,   244,     0,   259,     0,   249,
     223,     0,     0,     0,     0,     0,   221,     0,   206,   255,
     199,     0,   243,   266,     0,     0,     0,     0,   217,     0,
     224,     0,     0,   245,   220,   219,   218,   216,   222,   254,
       0,   206,   256,     0,   257
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     3,     4,    21,   131,   168,    83,     5,    13,    22,
      23,   214,   215,   216,   150,    84,   169,    85,    86,    87,
      88,    89,    90,   344,   268,   220,   221,    52,    53,    93,
     108,   184,   185,   100,   166,   382,   176,   105,   177,   167,
     289,   374,   290,   291,    54,   231,   276,    55,    56,    57,
     106,    58,   123,   124,   125,   126,   127,   293,   246,   192,
     193,    59,    60,   279,   318,   319,    62,    63,    98,   111,
     320,   321,    24,   371,   338,   364,   365,   417,   205,    72,
     345,   410,   140,   346,   483,   528,   516,   484,   347,   485,
     387,   462,   438,   411,   434,   449,   508,   536,   429,   435,
     511,   431,   466,   421,   425,   426,   445,   492,    25,    26
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -448
static const yytype_int16 yypact[] =
{
     145,   -16,   -16,    81,  -448,   145,  -448,   -21,   -21,  -448,
    -448,  -448,   201,  -448,  -448,  -448,   -16,   132,   -16,   -16,
     116,   225,    13,   201,  -448,  -448,  -448,    39,    75,  -448,
     184,  -448,  -448,  -448,  -448,   -26,   283,   134,   134,   -22,
      75,   109,   109,   109,   109,   121,   125,   -16,   159,   146,
     201,  -448,   169,   172,   177,  -448,  -448,  -448,  -448,   207,
    -448,  -448,   180,   195,   212,  -448,   201,   192,  -448,   206,
     -34,   -26,   222,   570,  -448,   372,  -448,    11,  -448,  -448,
    -448,  -448,    96,    22,  -448,  -448,   226,   234,   237,  -448,
     -20,    75,  -448,    75,    75,   254,    75,   249,   252,    14,
     -16,   -16,   -16,   -16,    83,   243,   244,   236,   -16,   257,
    -448,   278,   117,   276,  -448,  -448,  -448,   109,   109,   109,
     109,   243,   125,  -448,  -448,  -448,  -448,  -448,  -448,  -448,
    -448,  -448,   279,  -448,   299,   -26,   300,   -26,   274,  -448,
     284,   281,    -6,  -448,  -448,  -448,   214,  -448,  -448,   556,
    -448,  -448,  -448,  -448,  -448,   282,  -448,    -8,    21,    50,
     280,    66,    75,  -448,  -448,  -448,   288,   286,   289,   294,
     294,   294,   294,  -448,   -16,   293,   292,   296,   241,   -16,
     323,   -16,  -448,  -448,   302,   307,   314,  -448,  -448,   -16,
      15,   -16,   313,   317,   212,   -16,   -16,   -16,   -16,   -16,
     -16,  -448,  -448,   316,   327,   311,   320,   -26,  -448,   -16,
    -448,  -448,  -448,  -448,   332,  -448,   340,  -448,   -16,   247,
     341,  -448,   134,   134,   134,   134,  -448,  -448,    14,  -448,
      75,   173,   173,   173,   173,   339,  -448,   323,  -448,   294,
     294,  -448,   236,   409,   344,   248,  -448,   346,   117,  -448,
    -448,   294,   294,   294,   294,   294,   176,   -26,   300,   -26,
     -26,  -448,  -448,   556,  -448,   360,  -448,   369,   374,  -448,
     363,   375,    75,   378,  -448,   385,  -448,  -448,   500,  -448,
    -448,  -448,  -448,  -448,  -448,   173,   173,  -448,  -448,   372,
      -5,   388,   372,  -448,  -448,  -448,  -448,  -448,   173,   173,
     173,   173,   173,  -448,   409,  -448,  -448,  -448,  -448,  -448,
     384,  -448,  -448,  -448,    80,  -448,    75,   392,   402,  -448,
     500,   411,  -448,  -448,   -16,  -448,  -448,  -448,  -448,  -448,
    -448,  -448,  -448,   404,   372,   134,  -448,   308,   589,   212,
    -448,  -448,   413,   422,   -16,   -26,   423,   420,   429,  -448,
    -448,  -448,  -448,  -448,  -448,  -448,  -448,  -448,  -448,  -448,
     442,  -448,   430,  -448,   431,   437,   466,   451,   452,   226,
    -448,   -16,  -448,   449,   462,  -448,     7,   453,   372,  -448,
    -448,  -448,  -448,   501,  -448,   389,   336,   428,   452,  -448,
    -448,  -448,   119,  -448,  -448,  -448,   -16,  -448,  -448,   455,
     461,   469,   461,   511,   480,   519,   523,   461,   498,   445,
     -26,  -448,  -448,   569,  -448,  -448,   369,  -448,   452,  -448,
    -448,   -26,   539,   -26,   123,   529,   482,   445,  -448,   532,
     531,   536,   -26,   559,   534,   445,   222,   540,   428,   543,
     542,   544,   546,  -448,  -448,   -26,   511,   407,  -448,   562,
     445,   -26,   523,  -448,   546,   452,  -448,  -448,  -448,   583,
    -448,   101,   532,   -26,   461,  -448,   505,   557,  -448,  -448,
     564,  -448,  -448,   222,  -448,   518,   561,  -448,  -448,  -448,
    -448,  -448,   -16,   567,   568,   563,   -26,   573,   -26,   445,
    -448,  -448,   452,  -448,   597,   445,   601,   532,   579,   372,
     246,  -448,   222,   -26,   584,   585,  -448,   578,  -448,   586,
     528,  -448,   -26,   -16,   -16,   -26,   582,  -448,   -16,   546,
     -26,  -448,   608,   601,   445,  -448,   588,  -448,   183,    26,
     580,   -16,  -448,   541,   590,   591,   587,  -448,   594,  -448,
    -448,   -16,   265,   592,   -16,   -16,  -448,   298,   445,  -448,
     -26,   608,  -448,  -448,   202,   581,    41,   -16,  -448,   417,
    -448,   596,   546,  -448,  -448,  -448,  -448,  -448,  -448,  -448,
     555,   445,  -448,   598,  -448
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -448,  -448,   668,  -448,  -188,    -1,    -9,   654,   669,     8,
    -448,  -448,   415,  -448,   508,  -448,   -68,   -29,   -69,   342,
    -448,  -107,   604,   -33,  -448,  -448,   463,  -448,  -448,   -11,
     560,   439,  -448,    25,   456,  -448,  -448,   593,   448,  -448,
     310,  -448,  -448,  -248,  -448,  -134,   371,  -448,  -448,  -448,
     -76,  -448,  -448,  -448,  -448,  -448,  -448,  -448,   441,  -448,
     443,  -448,  -448,    -7,   370,   671,  -448,  -448,   527,  -448,
    -448,  -448,  -448,  -448,  -448,   309,  -448,  -448,   435,   -57,
     196,   -18,  -416,  -448,  -448,  -433,  -448,  -448,  -335,   197,
    -365,  -448,  -448,   258,  -425,  -448,  -448,   147,  -448,  -378,
     178,   250,  -447,  -381,  -448,   259,  -448,  -448,  -448,  -448
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -174
static const yytype_int16 yytable[] =
{
       7,     8,    35,    91,   141,   190,   250,   475,    30,   412,
     457,    97,   288,    92,   138,    27,   154,    31,    32,    67,
     458,   423,   470,   413,   288,   472,   432,    94,    96,   179,
     154,    65,    70,   170,   171,   172,   232,   233,   234,     6,
     186,   135,   325,   395,   164,   199,   109,   136,   448,   450,
     137,   399,    11,   439,    12,    71,   333,   494,   113,   444,
     412,   155,    95,   156,   505,   165,    69,   101,   102,   103,
     509,  -106,   533,   143,   132,   155,   219,   144,   203,   191,
     206,     9,   157,   488,   158,   159,   518,   161,   490,    64,
     476,   391,  -131,   392,  -131,    69,    69,   496,   547,   538,
     149,   245,   222,   175,   545,   285,   286,   478,   554,   556,
     391,   239,   559,   240,    66,   570,   217,   298,   299,   300,
     301,   302,   541,   561,    69,   542,   566,   506,   543,   544,
      29,   223,   525,    97,   173,   189,    74,    75,     6,   174,
      69,   190,   195,   196,   197,   198,   573,   225,     1,     2,
     261,   372,    74,    75,    69,   549,    29,   145,   146,   147,
     148,   335,    76,    77,    78,    79,    80,    81,    82,    92,
     479,   480,    29,   235,    29,   414,   415,   175,    76,    77,
      78,    79,    80,    81,    82,    28,   481,    29,   244,    33,
     247,    34,   572,    99,   251,   252,   253,   254,   255,   256,
     305,   275,   307,   308,   386,   104,    15,   443,   262,   107,
     267,   270,   271,   272,   273,   191,   110,   265,   117,   118,
     119,   120,   121,   122,   112,   280,   281,   282,     1,     2,
     217,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,   114,    47,    48,   115,   277,    49,   278,   303,
     116,   304,   478,   128,    16,    17,   324,    68,    69,   327,
      18,    19,   540,   314,   541,   133,    69,   542,   129,   369,
     543,   544,    20,   182,   183,   210,   211,   -16,   322,   323,
     134,    74,    75,   541,   564,   130,   542,   275,   377,   543,
     544,   328,   329,   330,   331,   332,     6,   174,   139,    73,
      50,    29,    29,   266,   294,   295,   151,    76,    77,    78,
      79,    80,    81,    82,   152,   479,   480,   153,    74,    75,
       6,   555,   160,   342,   348,   163,   162,   178,   180,   349,
     187,   350,   351,   352,   353,   354,   355,   368,    29,   356,
     357,   358,   359,   376,    76,    77,    78,    79,    80,    81,
      82,   188,   194,   436,   202,   201,   207,   208,   204,   360,
     361,   209,   218,   228,   440,   224,   442,   149,   230,   237,
     388,    74,    75,   227,   173,   454,   362,   560,   236,   541,
     363,   238,   542,   416,   242,   543,   544,   241,   467,   243,
     248,    29,   257,   259,   473,   418,   249,    76,    77,    78,
      79,    80,    81,    82,   258,   260,   487,    74,    75,   263,
     349,   451,   350,   351,   352,   353,   354,   355,   398,   264,
     356,   357,   358,   359,   283,   219,   288,    29,   482,   502,
     292,   504,   245,    76,    77,    78,    79,    80,    81,    82,
     360,   361,   310,    69,   486,   312,   519,   400,   401,   402,
     403,   404,   405,   406,   407,   526,   408,   313,   530,   311,
     315,   363,   316,   534,   326,   334,   514,   482,   400,   401,
     402,   403,   404,   405,   406,   407,   337,   408,   339,   512,
     343,   498,    34,   469,   341,   400,   401,   402,   403,   404,
     405,   406,   407,   562,   408,   375,   568,   373,   541,   381,
     378,   542,   379,   409,   543,   544,    36,    37,    38,    39,
      40,   380,   527,   529,   385,   383,   384,   532,    47,    48,
      34,  -173,   400,   401,   402,   403,   404,   405,   406,   407,
     527,   408,    -9,   386,   389,   390,   396,   419,   394,   317,
     527,   527,   420,   558,   527,   400,   401,   402,   403,   404,
     405,   406,   407,   422,   408,   427,   567,   447,   400,   401,
     402,   403,   404,   405,   406,   407,   424,   408,   400,   401,
     402,   403,   404,   405,   406,   407,   428,   408,   430,   433,
     489,   400,   401,   402,   403,   404,   405,   406,   407,   437,
     408,    74,    75,   495,   441,   400,   401,   402,   403,   404,
     405,   406,   407,   524,   408,    74,   446,    34,   452,   453,
     456,    29,   212,   213,   455,   463,   548,    76,    77,    78,
      79,    80,    81,    82,   366,    29,   459,   461,   465,   464,
     571,    76,    77,    78,    79,    80,    81,    82,   471,   477,
     493,   499,   491,   497,   367,   500,   503,   507,   501,   510,
      76,    77,    78,    79,    80,    81,    82,   513,   520,   522,
     531,   521,   523,   535,   539,   546,   565,   550,   551,   552,
     553,   557,   569,    10,   574,    51,   229,    14,   309,   142,
     370,   287,   200,   269,   274,   284,   393,   336,   296,   226,
     340,   297,    61,   306,   397,   515,   460,   517,   563,   181,
       0,   537,   474,     0,     0,   468
};

static const yytype_int16 yycheck[] =
{
       1,     2,    20,    36,    73,   112,   194,   454,    17,   387,
     435,    40,    17,    35,    71,    16,    36,    18,    19,    28,
     436,   402,   447,   388,    17,   450,   407,    38,    39,   105,
      36,    23,    58,   101,   102,   103,   170,   171,   172,    55,
     108,    75,   290,   378,    30,   121,    47,    81,   426,   427,
      84,   386,    73,   418,    75,    81,   304,   473,    50,   424,
     438,    81,    84,    83,   489,    51,    74,    42,    43,    44,
     495,    76,   519,    62,    66,    81,    84,    66,   135,   112,
     137,     0,    91,   464,    93,    94,   502,    96,   466,    76,
     455,    84,    77,    86,    79,    74,    74,   475,   531,   524,
      78,    86,    81,   104,    78,   239,   240,     6,   541,   542,
      84,   179,   545,   181,    75,   562,   149,   251,   252,   253,
     254,   255,    81,   548,    74,    84,    85,   492,    87,    88,
      55,    81,   510,   162,    51,    18,    35,    36,    55,    56,
      74,   248,   117,   118,   119,   120,   571,    81,     3,     4,
     207,   339,    35,    36,    74,   533,    55,    61,    62,    63,
      64,    81,    61,    62,    63,    64,    65,    66,    67,    35,
      69,    70,    55,   174,    55,    56,    57,   178,    61,    62,
      63,    64,    65,    66,    67,    53,    85,    55,   189,    73,
     191,    75,   570,    84,   195,   196,   197,   198,   199,   200,
     257,   230,   259,   260,    81,    84,     5,    84,   209,    84,
     219,   222,   223,   224,   225,   248,    57,   218,    11,    12,
      13,    14,    15,    16,    78,   232,   233,   234,     3,     4,
     263,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    73,    18,    19,    73,    73,    22,    75,    73,
      73,    75,     6,    73,    53,    54,   289,    73,    74,   292,
      59,    60,    79,   272,    81,    73,    74,    84,    73,   338,
      87,    88,    71,    37,    38,    61,    62,    76,   285,   286,
      74,    35,    36,    81,    82,    73,    84,   316,   345,    87,
      88,   298,   299,   300,   301,   302,    55,    56,    76,    16,
      75,    55,    55,    56,    56,    57,    80,    61,    62,    63,
      64,    65,    66,    67,    80,    69,    70,    80,    35,    36,
      55,    56,    68,   324,   335,    73,    77,    84,    84,    21,
      73,    23,    24,    25,    26,    27,    28,   338,    55,    31,
      32,    33,    34,   344,    61,    62,    63,    64,    65,    66,
      67,    73,    76,   410,    55,    76,    82,    73,    58,    51,
      52,    80,    80,    77,   421,    85,   423,    78,    74,    77,
     371,    35,    36,    85,    51,   432,    68,    79,    85,    81,
      72,    85,    84,   392,    77,    87,    88,    85,   445,    75,
      77,    55,    76,    82,   451,   396,    79,    61,    62,    63,
      64,    65,    66,    67,    77,    85,   463,    35,    36,    77,
      21,   429,    23,    24,    25,    26,    27,    28,    82,    79,
      31,    32,    33,    34,    85,    84,    17,    55,   461,   486,
      86,   488,    86,    61,    62,    63,    64,    65,    66,    67,
      51,    52,    82,    74,   462,    82,   503,    40,    41,    42,
      43,    44,    45,    46,    47,   512,    49,    82,   515,    85,
      82,    72,    77,   520,    76,    81,   499,   500,    40,    41,
      42,    43,    44,    45,    46,    47,    84,    49,    76,   497,
      76,   482,    75,    76,    73,    40,    41,    42,    43,    44,
      45,    46,    47,   550,    49,    73,    79,    84,    81,    57,
      77,    84,    82,    75,    87,    88,     6,     7,     8,     9,
      10,    82,   513,   514,    77,    85,    85,   518,    18,    19,
      75,    55,    40,    41,    42,    43,    44,    45,    46,    47,
     531,    49,    81,    81,    85,    73,    35,    82,    85,    39,
     541,   542,    81,   544,   545,    40,    41,    42,    43,    44,
      45,    46,    47,    84,    49,    75,   557,    75,    40,    41,
      42,    43,    44,    45,    46,    47,    55,    49,    40,    41,
      42,    43,    44,    45,    46,    47,    57,    49,    55,    81,
      75,    40,    41,    42,    43,    44,    45,    46,    47,    20,
      49,    35,    36,    75,    55,    40,    41,    42,    43,    44,
      45,    46,    47,    75,    49,    35,    77,    75,    77,    73,
      76,    55,    56,    57,    55,    73,    75,    61,    62,    63,
      64,    65,    66,    67,    35,    55,    86,    84,    82,    85,
      75,    61,    62,    63,    64,    65,    66,    67,    76,    56,
      76,    74,    85,    82,    55,    77,    73,    50,    85,    48,
      61,    62,    63,    64,    65,    66,    67,    78,    74,    81,
      78,    76,    76,    55,    76,    85,    85,    77,    77,    82,
      76,    79,    76,     5,    76,    21,   168,     8,   263,    75,
     338,   242,   122,   220,   228,   237,   376,   316,   247,   162,
     320,   248,    21,   258,   385,   499,   438,   500,   551,   106,
      -1,   523,   452,    -1,    -1,   446
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     3,     4,    90,    91,    96,    55,    94,    94,     0,
      91,    73,    75,    97,    97,     5,    53,    54,    59,    60,
      71,    92,    98,    99,   161,   197,   198,    94,    53,    55,
      95,    94,    94,    73,    75,   170,     6,     7,     8,     9,
      10,    11,    12,    13,    14,    15,    16,    18,    19,    22,
      75,    96,   116,   117,   133,   136,   137,   138,   140,   150,
     151,   154,   155,   156,    76,    98,    75,    95,    73,    74,
      58,    81,   168,    16,    35,    36,    61,    62,    63,    64,
      65,    66,    67,    95,   104,   106,   107,   108,   109,   110,
     111,   112,    35,   118,   118,    84,   118,   106,   157,    84,
     122,   122,   122,   122,    84,   126,   139,    84,   119,    94,
      57,   158,    78,    98,    73,    73,    73,    11,    12,    13,
      14,    15,    16,   141,   142,   143,   144,   145,    73,    73,
      73,    93,    98,    73,    74,    75,    81,    84,   168,    76,
     171,   107,   111,    62,    66,    61,    62,    63,    64,    78,
     103,    80,    80,    80,    36,    81,    83,    95,    95,    95,
      68,    95,    77,    73,    30,    51,   123,   128,    94,   105,
     105,   105,   105,    51,    56,    94,   125,   127,    84,   139,
      84,   126,    37,    38,   120,   121,   105,    73,    73,    18,
     110,   112,   148,   149,    76,   122,   122,   122,   122,   139,
     119,    76,    55,   168,    58,   167,   168,    82,    73,    80,
      61,    62,    56,    57,   100,   101,   102,   112,    80,    84,
     114,   115,    81,    81,    85,    81,   157,    85,    77,   103,
      74,   134,   134,   134,   134,    94,    85,    77,    85,   105,
     105,    85,    77,    75,    94,    86,   147,    94,    77,    79,
      93,    94,    94,    94,    94,    94,    94,    76,    77,    82,
      85,   168,    94,    77,    79,    94,    56,    95,   113,   115,
     118,   118,   118,   118,   123,   106,   135,    73,    75,   152,
     152,   152,   152,    85,   127,   134,   134,   120,    17,   129,
     131,   132,    86,   146,    56,    57,   147,   149,   134,   134,
     134,   134,   134,    73,    75,   168,   167,   168,   168,   101,
      82,    85,    82,    82,    95,    82,    77,    39,   153,   154,
     159,   160,   152,   152,   112,   132,    76,   112,   152,   152,
     152,   152,   152,   132,    81,    81,   135,    84,   163,    76,
     153,    73,    94,    76,   112,   169,   172,   177,   118,    21,
      23,    24,    25,    26,    27,    28,    31,    32,    33,    34,
      51,    52,    68,    72,   164,   165,    35,    55,    94,   107,
     108,   162,    93,    84,   130,    73,    94,   168,    77,    82,
      82,    57,   124,    85,    85,    77,    81,   179,    94,    85,
      73,    84,    86,   129,    85,   177,    35,   164,    82,   177,
      40,    41,    42,    43,    44,    45,    46,    47,    49,    75,
     170,   182,   188,   179,    56,    57,    95,   166,    94,    82,
      81,   192,    84,   192,    55,   193,   194,    75,    57,   187,
      55,   190,   192,    81,   183,   188,   168,    20,   181,   179,
     168,    55,   168,    84,   179,   195,    77,    75,   188,   184,
     188,   170,    77,    73,   168,    55,    76,   183,   171,    86,
     182,    84,   180,    73,    85,    82,   191,   168,   194,    76,
     183,    76,   183,   168,   190,   191,   179,    56,     6,    69,
      70,    85,   112,   173,   176,   178,   170,   168,   192,    75,
     188,    85,   196,    76,   171,    75,   188,    82,    94,    74,
      77,    85,   168,    73,   168,   183,   179,    50,   185,   183,
      48,   189,   170,    78,   112,   169,   175,   178,   171,   168,
      74,    76,    81,    76,    75,   188,   168,    94,   174,    94,
     168,    78,    94,   191,   168,    55,   186,   189,   183,    76,
      79,    81,    84,    87,    88,    78,    85,   174,    75,   188,
      77,    77,    82,    76,   174,    56,   174,    79,    94,   174,
      79,   183,   168,   186,    82,    85,    85,    94,    79,    76,
     191,    75,   188,   183,    76
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
#line 149 "xi-grammar.y"
    { (yyval.modlist) = (yyvsp[(1) - (1)].modlist); modlist = (yyvsp[(1) - (1)].modlist); }
    break;

  case 3:
#line 153 "xi-grammar.y"
    { 
		  (yyval.modlist) = 0; 
		}
    break;

  case 4:
#line 157 "xi-grammar.y"
    { (yyval.modlist) = new ModuleList(lineno, (yyvsp[(1) - (2)].module), (yyvsp[(2) - (2)].modlist)); }
    break;

  case 5:
#line 161 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 6:
#line 163 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 7:
#line 167 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 8:
#line 169 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 9:
#line 173 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 10:
#line 177 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 11:
#line 179 "xi-grammar.y"
    {
		  char *tmp = new char[strlen((yyvsp[(1) - (4)].strval))+strlen((yyvsp[(4) - (4)].strval))+3];
		  sprintf(tmp,"%s::%s", (yyvsp[(1) - (4)].strval), (yyvsp[(4) - (4)].strval));
		  (yyval.strval) = tmp;
		}
    break;

  case 12:
#line 187 "xi-grammar.y"
    { 
		    (yyval.module) = new Module(lineno, (yyvsp[(2) - (3)].strval), (yyvsp[(3) - (3)].conslist)); 
		}
    break;

  case 13:
#line 191 "xi-grammar.y"
    {  
		    (yyval.module) = new Module(lineno, (yyvsp[(2) - (3)].strval), (yyvsp[(3) - (3)].conslist)); 
		    (yyval.module)->setMain();
		}
    break;

  case 14:
#line 198 "xi-grammar.y"
    { (yyval.conslist) = 0; }
    break;

  case 15:
#line 200 "xi-grammar.y"
    { (yyval.conslist) = (yyvsp[(2) - (4)].conslist); }
    break;

  case 16:
#line 204 "xi-grammar.y"
    { (yyval.conslist) = 0; }
    break;

  case 17:
#line 206 "xi-grammar.y"
    { (yyval.conslist) = new ConstructList(lineno, (yyvsp[(1) - (2)].construct), (yyvsp[(2) - (2)].conslist)); }
    break;

  case 18:
#line 210 "xi-grammar.y"
    { if((yyvsp[(3) - (5)].conslist)) (yyvsp[(3) - (5)].conslist)->setExtern((yyvsp[(1) - (5)].intval)); (yyval.construct) = (yyvsp[(3) - (5)].conslist); }
    break;

  case 19:
#line 212 "xi-grammar.y"
    { (yyval.construct) = new Scope((yyvsp[(2) - (5)].strval), (yyvsp[(4) - (5)].conslist)); }
    break;

  case 20:
#line 214 "xi-grammar.y"
    { (yyval.construct) = new UsingScope((yyvsp[(3) - (4)].strval), false); }
    break;

  case 21:
#line 216 "xi-grammar.y"
    { (yyval.construct) = new UsingScope((yyvsp[(2) - (3)].strval), true); }
    break;

  case 22:
#line 218 "xi-grammar.y"
    { (yyvsp[(2) - (2)].module)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].module); }
    break;

  case 23:
#line 220 "xi-grammar.y"
    { (yyvsp[(2) - (2)].member)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].member); }
    break;

  case 24:
#line 222 "xi-grammar.y"
    { (yyvsp[(2) - (3)].message)->setExtern((yyvsp[(1) - (3)].intval)); (yyval.construct) = (yyvsp[(2) - (3)].message); }
    break;

  case 25:
#line 224 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 26:
#line 226 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 27:
#line 228 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 28:
#line 230 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 29:
#line 232 "xi-grammar.y"
    { (yyvsp[(2) - (2)].templat)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].templat); }
    break;

  case 30:
#line 234 "xi-grammar.y"
    { (yyval.construct) = NULL; }
    break;

  case 31:
#line 236 "xi-grammar.y"
    { (yyval.construct) = NULL; }
    break;

  case 32:
#line 238 "xi-grammar.y"
    { (yyval.construct) = (yyvsp[(1) - (1)].accelBlock); }
    break;

  case 33:
#line 242 "xi-grammar.y"
    { (yyval.tparam) = new TParamType((yyvsp[(1) - (1)].type)); }
    break;

  case 34:
#line 244 "xi-grammar.y"
    { (yyval.tparam) = new TParamVal((yyvsp[(1) - (1)].strval)); }
    break;

  case 35:
#line 246 "xi-grammar.y"
    { (yyval.tparam) = new TParamVal((yyvsp[(1) - (1)].strval)); }
    break;

  case 36:
#line 250 "xi-grammar.y"
    { (yyval.tparlist) = new TParamList((yyvsp[(1) - (1)].tparam)); }
    break;

  case 37:
#line 252 "xi-grammar.y"
    { (yyval.tparlist) = new TParamList((yyvsp[(1) - (3)].tparam), (yyvsp[(3) - (3)].tparlist)); }
    break;

  case 38:
#line 256 "xi-grammar.y"
    { (yyval.tparlist) = 0; }
    break;

  case 39:
#line 258 "xi-grammar.y"
    { (yyval.tparlist) = (yyvsp[(1) - (1)].tparlist); }
    break;

  case 40:
#line 262 "xi-grammar.y"
    { (yyval.tparlist) = 0; }
    break;

  case 41:
#line 264 "xi-grammar.y"
    { (yyval.tparlist) = (yyvsp[(2) - (3)].tparlist); }
    break;

  case 42:
#line 268 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("int"); }
    break;

  case 43:
#line 270 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long"); }
    break;

  case 44:
#line 272 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("short"); }
    break;

  case 45:
#line 274 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("char"); }
    break;

  case 46:
#line 276 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned int"); }
    break;

  case 47:
#line 278 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long"); }
    break;

  case 48:
#line 280 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long"); }
    break;

  case 49:
#line 282 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long long"); }
    break;

  case 50:
#line 284 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned short"); }
    break;

  case 51:
#line 286 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned char"); }
    break;

  case 52:
#line 288 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long long"); }
    break;

  case 53:
#line 290 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("float"); }
    break;

  case 54:
#line 292 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("double"); }
    break;

  case 55:
#line 294 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long double"); }
    break;

  case 56:
#line 296 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("void"); }
    break;

  case 57:
#line 299 "xi-grammar.y"
    { (yyval.ntype) = new NamedType((yyvsp[(1) - (2)].strval),(yyvsp[(2) - (2)].tparlist)); }
    break;

  case 58:
#line 300 "xi-grammar.y"
    { 
                    char* basename, *scope;
                    splitScopedName((yyvsp[(1) - (2)].strval), &scope, &basename);
                    (yyval.ntype) = new NamedType(basename, (yyvsp[(2) - (2)].tparlist), scope);
                }
    break;

  case 59:
#line 308 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 60:
#line 310 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ntype); }
    break;

  case 61:
#line 314 "xi-grammar.y"
    { (yyval.ptype) = new PtrType((yyvsp[(1) - (2)].type)); }
    break;

  case 62:
#line 318 "xi-grammar.y"
    { (yyvsp[(1) - (2)].ptype)->indirect(); (yyval.ptype) = (yyvsp[(1) - (2)].ptype); }
    break;

  case 63:
#line 320 "xi-grammar.y"
    { (yyvsp[(1) - (2)].ptype)->indirect(); (yyval.ptype) = (yyvsp[(1) - (2)].ptype); }
    break;

  case 64:
#line 324 "xi-grammar.y"
    { (yyval.ftype) = new FuncType((yyvsp[(1) - (8)].type), (yyvsp[(4) - (8)].strval), (yyvsp[(7) - (8)].plist)); }
    break;

  case 65:
#line 328 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 66:
#line 330 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 67:
#line 332 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 68:
#line 334 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ftype); }
    break;

  case 69:
#line 337 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(2) - (2)].type); }
    break;

  case 70:
#line 339 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (2)].type); }
    break;

  case 71:
#line 343 "xi-grammar.y"
    { (yyval.type) = new ReferenceType((yyvsp[(1) - (2)].type)); }
    break;

  case 72:
#line 345 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 73:
#line 349 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 74:
#line 351 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 75:
#line 355 "xi-grammar.y"
    { (yyval.val) = (yyvsp[(2) - (3)].val); }
    break;

  case 76:
#line 359 "xi-grammar.y"
    { (yyval.vallist) = 0; }
    break;

  case 77:
#line 361 "xi-grammar.y"
    { (yyval.vallist) = new ValueList((yyvsp[(1) - (2)].val), (yyvsp[(2) - (2)].vallist)); }
    break;

  case 78:
#line 365 "xi-grammar.y"
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[(2) - (4)].type), (yyvsp[(3) - (4)].strval), (yyvsp[(4) - (4)].vallist)); }
    break;

  case 79:
#line 369 "xi-grammar.y"
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[(3) - (5)].type), (yyvsp[(5) - (5)].strval), 0, 1); }
    break;

  case 80:
#line 373 "xi-grammar.y"
    { (yyval.intval) = 0;}
    break;

  case 81:
#line 375 "xi-grammar.y"
    { (yyval.intval) = 0;}
    break;

  case 82:
#line 379 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 83:
#line 381 "xi-grammar.y"
    { 
		  /*
		  printf("Warning: Message attributes are being phased out.\n");
		  printf("Warning: Please remove them from interface files.\n");
		  */
		  (yyval.intval) = (yyvsp[(2) - (3)].intval); 
		}
    break;

  case 84:
#line 391 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (1)].intval); }
    break;

  case 85:
#line 393 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (3)].intval) | (yyvsp[(3) - (3)].intval); }
    break;

  case 86:
#line 397 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 87:
#line 399 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 88:
#line 403 "xi-grammar.y"
    { (yyval.cattr) = 0; }
    break;

  case 89:
#line 405 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(2) - (3)].cattr); }
    break;

  case 90:
#line 409 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (1)].cattr); }
    break;

  case 91:
#line 411 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (3)].cattr) | (yyvsp[(3) - (3)].cattr); }
    break;

  case 92:
#line 415 "xi-grammar.y"
    { python_doc = NULL; (yyval.intval) = 0; }
    break;

  case 93:
#line 417 "xi-grammar.y"
    { python_doc = (yyvsp[(1) - (1)].strval); (yyval.intval) = 0; }
    break;

  case 94:
#line 421 "xi-grammar.y"
    { (yyval.cattr) = Chare::CPYTHON; }
    break;

  case 95:
#line 425 "xi-grammar.y"
    { (yyval.cattr) = 0; }
    break;

  case 96:
#line 427 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(2) - (3)].cattr); }
    break;

  case 97:
#line 431 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (1)].cattr); }
    break;

  case 98:
#line 433 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (3)].cattr) | (yyvsp[(3) - (3)].cattr); }
    break;

  case 99:
#line 437 "xi-grammar.y"
    { (yyval.cattr) = Chare::CMIGRATABLE; }
    break;

  case 100:
#line 439 "xi-grammar.y"
    { (yyval.cattr) = Chare::CPYTHON; }
    break;

  case 101:
#line 443 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 102:
#line 445 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 103:
#line 448 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 104:
#line 450 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 105:
#line 453 "xi-grammar.y"
    { (yyval.mv) = new MsgVar((yyvsp[(2) - (5)].type), (yyvsp[(3) - (5)].strval), (yyvsp[(1) - (5)].intval), (yyvsp[(4) - (5)].intval)); }
    break;

  case 106:
#line 457 "xi-grammar.y"
    { (yyval.mvlist) = new MsgVarList((yyvsp[(1) - (1)].mv)); }
    break;

  case 107:
#line 459 "xi-grammar.y"
    { (yyval.mvlist) = new MsgVarList((yyvsp[(1) - (2)].mv), (yyvsp[(2) - (2)].mvlist)); }
    break;

  case 108:
#line 463 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, (yyvsp[(3) - (3)].ntype)); }
    break;

  case 109:
#line 465 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, (yyvsp[(3) - (6)].ntype), (yyvsp[(5) - (6)].mvlist)); }
    break;

  case 110:
#line 469 "xi-grammar.y"
    { (yyval.typelist) = 0; }
    break;

  case 111:
#line 471 "xi-grammar.y"
    { (yyval.typelist) = (yyvsp[(2) - (2)].typelist); }
    break;

  case 112:
#line 475 "xi-grammar.y"
    { (yyval.typelist) = new TypeList((yyvsp[(1) - (1)].ntype)); }
    break;

  case 113:
#line 477 "xi-grammar.y"
    { (yyval.typelist) = new TypeList((yyvsp[(1) - (3)].ntype), (yyvsp[(3) - (3)].typelist)); }
    break;

  case 114:
#line 481 "xi-grammar.y"
    { (yyval.chare) = new Chare(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 115:
#line 483 "xi-grammar.y"
    { (yyval.chare) = new MainChare(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 116:
#line 487 "xi-grammar.y"
    { (yyval.chare) = new Group(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 117:
#line 491 "xi-grammar.y"
    { (yyval.chare) = new NodeGroup(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 118:
#line 495 "xi-grammar.y"
    {/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",(yyvsp[(2) - (4)].strval));
			(yyval.ntype) = new NamedType(buf); 
		}
    break;

  case 119:
#line 501 "xi-grammar.y"
    { (yyval.ntype) = new NamedType((yyvsp[(2) - (3)].strval)); }
    break;

  case 120:
#line 505 "xi-grammar.y"
    {  (yyval.chare) = new Array(lineno, (yyvsp[(2) - (6)].cattr), (yyvsp[(3) - (6)].ntype), (yyvsp[(4) - (6)].ntype), (yyvsp[(5) - (6)].typelist), (yyvsp[(6) - (6)].mbrlist)); }
    break;

  case 121:
#line 507 "xi-grammar.y"
    {  (yyval.chare) = new Array(lineno, (yyvsp[(3) - (6)].cattr), (yyvsp[(2) - (6)].ntype), (yyvsp[(4) - (6)].ntype), (yyvsp[(5) - (6)].typelist), (yyvsp[(6) - (6)].mbrlist)); }
    break;

  case 122:
#line 511 "xi-grammar.y"
    { (yyval.chare) = new Chare(lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist));}
    break;

  case 123:
#line 513 "xi-grammar.y"
    { (yyval.chare) = new MainChare(lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 124:
#line 517 "xi-grammar.y"
    { (yyval.chare) = new Group(lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 125:
#line 521 "xi-grammar.y"
    { (yyval.chare) = new NodeGroup( lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 126:
#line 525 "xi-grammar.y"
    { (yyval.chare) = new Array( lineno, 0, (yyvsp[(2) - (5)].ntype), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 127:
#line 529 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[(3) - (4)].strval))); }
    break;

  case 128:
#line 531 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[(3) - (7)].strval)), (yyvsp[(5) - (7)].mvlist)); }
    break;

  case 129:
#line 535 "xi-grammar.y"
    { (yyval.type) = 0; }
    break;

  case 130:
#line 537 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(2) - (2)].type); }
    break;

  case 131:
#line 541 "xi-grammar.y"
    { (yyval.strval) = 0; }
    break;

  case 132:
#line 543 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(2) - (2)].strval); }
    break;

  case 133:
#line 545 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(2) - (2)].strval); }
    break;

  case 134:
#line 549 "xi-grammar.y"
    { (yyval.tvar) = new TType(new NamedType((yyvsp[(2) - (3)].strval)), (yyvsp[(3) - (3)].type)); }
    break;

  case 135:
#line 551 "xi-grammar.y"
    { (yyval.tvar) = new TFunc((yyvsp[(1) - (2)].ftype), (yyvsp[(2) - (2)].strval)); }
    break;

  case 136:
#line 553 "xi-grammar.y"
    { (yyval.tvar) = new TName((yyvsp[(1) - (3)].type), (yyvsp[(2) - (3)].strval), (yyvsp[(3) - (3)].strval)); }
    break;

  case 137:
#line 557 "xi-grammar.y"
    { (yyval.tvarlist) = new TVarList((yyvsp[(1) - (1)].tvar)); }
    break;

  case 138:
#line 559 "xi-grammar.y"
    { (yyval.tvarlist) = new TVarList((yyvsp[(1) - (3)].tvar), (yyvsp[(3) - (3)].tvarlist)); }
    break;

  case 139:
#line 563 "xi-grammar.y"
    { (yyval.tvarlist) = (yyvsp[(3) - (4)].tvarlist); }
    break;

  case 140:
#line 567 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 141:
#line 569 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 142:
#line 571 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 143:
#line 573 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 144:
#line 575 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].message)); (yyvsp[(2) - (2)].message)->setTemplate((yyval.templat)); }
    break;

  case 145:
#line 579 "xi-grammar.y"
    { (yyval.mbrlist) = 0; }
    break;

  case 146:
#line 581 "xi-grammar.y"
    { (yyval.mbrlist) = (yyvsp[(2) - (4)].mbrlist); }
    break;

  case 147:
#line 585 "xi-grammar.y"
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
#line 604 "xi-grammar.y"
    { (yyval.mbrlist) = new MemberList((yyvsp[(1) - (2)].member), (yyvsp[(2) - (2)].mbrlist)); }
    break;

  case 149:
#line 608 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (2)].readonly); }
    break;

  case 150:
#line 610 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (2)].readonly); }
    break;

  case 152:
#line 613 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (2)].member); }
    break;

  case 153:
#line 615 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(2) - (3)].pupable); }
    break;

  case 154:
#line 617 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(2) - (3)].includeFile); }
    break;

  case 155:
#line 619 "xi-grammar.y"
    { (yyval.member) = new ClassDeclaration(lineno,(yyvsp[(2) - (3)].strval)); }
    break;

  case 156:
#line 623 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 1); }
    break;

  case 157:
#line 625 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 1); }
    break;

  case 158:
#line 627 "xi-grammar.y"
    { printf("Warning: deprecated use of initcall. Use initnode or initproc instead.\n"); 
		  (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 1); }
    break;

  case 159:
#line 630 "xi-grammar.y"
    { printf("Warning: deprecated use of initcall. Use initnode or initproc instead.\n");
		  (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 1); }
    break;

  case 160:
#line 635 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 0); }
    break;

  case 161:
#line 637 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 0); }
    break;

  case 162:
#line 639 "xi-grammar.y"
    {
                  InitCall* rtn = new InitCall(lineno, (yyvsp[(6) - (9)].strval), 0);
                  rtn->setAccel();
                  (yyval.member) = rtn;
		}
    break;

  case 163:
#line 647 "xi-grammar.y"
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[(1) - (1)].ntype),0); }
    break;

  case 164:
#line 649 "xi-grammar.y"
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[(1) - (3)].ntype),(yyvsp[(3) - (3)].pupable)); }
    break;

  case 165:
#line 652 "xi-grammar.y"
    { (yyval.includeFile) = new IncludeFile(lineno,(yyvsp[(1) - (1)].strval)); }
    break;

  case 166:
#line 656 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (2)].entry); }
    break;

  case 167:
#line 658 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].member); }
    break;

  case 168:
#line 662 "xi-grammar.y"
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
#line 673 "xi-grammar.y"
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
#line 684 "xi-grammar.y"
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
#line 702 "xi-grammar.y"
    { (yyval.accelBlock) = new AccelBlock(lineno, new XStr((yyvsp[(3) - (5)].strval))); }
    break;

  case 172:
#line 704 "xi-grammar.y"
    { (yyval.accelBlock) = new AccelBlock(lineno, NULL); }
    break;

  case 173:
#line 708 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("void"); }
    break;

  case 174:
#line 710 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 175:
#line 714 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 176:
#line 716 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(2) - (3)].intval); }
    break;

  case 177:
#line 720 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (1)].intval); }
    break;

  case 178:
#line 722 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (3)].intval) | (yyvsp[(3) - (3)].intval); }
    break;

  case 179:
#line 726 "xi-grammar.y"
    { (yyval.intval) = STHREADED; }
    break;

  case 180:
#line 728 "xi-grammar.y"
    { (yyval.intval) = SSYNC; }
    break;

  case 181:
#line 730 "xi-grammar.y"
    { (yyval.intval) = SIGET; }
    break;

  case 182:
#line 732 "xi-grammar.y"
    { (yyval.intval) = SLOCKED; }
    break;

  case 183:
#line 734 "xi-grammar.y"
    { (yyval.intval) = SCREATEHERE; }
    break;

  case 184:
#line 736 "xi-grammar.y"
    { (yyval.intval) = SCREATEHOME; }
    break;

  case 185:
#line 738 "xi-grammar.y"
    { (yyval.intval) = SNOKEEP; }
    break;

  case 186:
#line 740 "xi-grammar.y"
    { (yyval.intval) = SNOTRACE; }
    break;

  case 187:
#line 742 "xi-grammar.y"
    { (yyval.intval) = SIMMEDIATE; }
    break;

  case 188:
#line 744 "xi-grammar.y"
    { (yyval.intval) = SSKIPSCHED; }
    break;

  case 189:
#line 746 "xi-grammar.y"
    { (yyval.intval) = SINLINE; }
    break;

  case 190:
#line 748 "xi-grammar.y"
    { (yyval.intval) = SLOCAL; }
    break;

  case 191:
#line 750 "xi-grammar.y"
    { (yyval.intval) = SPYTHON; }
    break;

  case 192:
#line 752 "xi-grammar.y"
    { (yyval.intval) = SMEM; }
    break;

  case 193:
#line 756 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 194:
#line 758 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 195:
#line 760 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 196:
#line 764 "xi-grammar.y"
    { (yyval.strval) = ""; }
    break;

  case 197:
#line 766 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 198:
#line 768 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[(1) - (3)].strval))+strlen((yyvsp[(3) - (3)].strval))+3];
			sprintf(tmp,"%s, %s", (yyvsp[(1) - (3)].strval), (yyvsp[(3) - (3)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 199:
#line 776 "xi-grammar.y"
    { (yyval.strval) = ""; }
    break;

  case 200:
#line 778 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 201:
#line 780 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s[%s]%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 202:
#line 786 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s{%s}%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 203:
#line 792 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s(%s)%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 204:
#line 798 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(2) - (4)].strval))+strlen((yyvsp[(4) - (4)].strval))+3];
			sprintf(tmp,"(%s)%s", (yyvsp[(2) - (4)].strval), (yyvsp[(4) - (4)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 205:
#line 806 "xi-grammar.y"
    {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			(yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].type),(yyvsp[(2) - (3)].strval));
		}
    break;

  case 206:
#line 813 "xi-grammar.y"
    { 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			(yyval.intval) = 0;
		}
    break;

  case 207:
#line 821 "xi-grammar.y"
    { 
			in_braces=0;
			(yyval.intval) = 0;
		}
    break;

  case 208:
#line 828 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (1)].type));}
    break;

  case 209:
#line 830 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].type),(yyvsp[(2) - (3)].strval)); (yyval.pname)->setConditional((yyvsp[(3) - (3)].intval)); }
    break;

  case 210:
#line 832 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (4)].type),(yyvsp[(2) - (4)].strval),0,(yyvsp[(4) - (4)].val));}
    break;

  case 211:
#line 834 "xi-grammar.y"
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].pname)->getType(), (yyvsp[(1) - (3)].pname)->getName() ,(yyvsp[(2) - (3)].strval));
		}
    break;

  case 212:
#line 840 "xi-grammar.y"
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READONLY; }
    break;

  case 213:
#line 841 "xi-grammar.y"
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_READWRITE; }
    break;

  case 214:
#line 842 "xi-grammar.y"
    { (yyval.intval) = Parameter::ACCEL_BUFFER_TYPE_WRITEONLY; }
    break;

  case 215:
#line 845 "xi-grammar.y"
    { (yyval.xstrptr) = new XStr((yyvsp[(1) - (1)].strval)); }
    break;

  case 216:
#line 846 "xi-grammar.y"
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "->" << (yyvsp[(4) - (4)].strval); }
    break;

  case 217:
#line 847 "xi-grammar.y"
    { (yyval.xstrptr) = new XStr(""); *((yyval.xstrptr)) << *((yyvsp[(1) - (3)].xstrptr)) << "." << (yyvsp[(3) - (3)].strval); }
    break;

  case 218:
#line 849 "xi-grammar.y"
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "[" << *((yyvsp[(3) - (4)].xstrptr)) << "]";
                  delete (yyvsp[(1) - (4)].xstrptr);
                  delete (yyvsp[(3) - (4)].xstrptr);
                }
    break;

  case 219:
#line 856 "xi-grammar.y"
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "[" << (yyvsp[(3) - (4)].strval) << "]";
                  delete (yyvsp[(1) - (4)].xstrptr);
                }
    break;

  case 220:
#line 862 "xi-grammar.y"
    {
                  (yyval.xstrptr) = new XStr("");
                  *((yyval.xstrptr)) << *((yyvsp[(1) - (4)].xstrptr)) << "(" << *((yyvsp[(3) - (4)].xstrptr)) << ")";
                  delete (yyvsp[(1) - (4)].xstrptr);
                  delete (yyvsp[(3) - (4)].xstrptr);
                }
    break;

  case 221:
#line 871 "xi-grammar.y"
    {
                  in_bracket = 0;
                  (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].pname)->getType(), (yyvsp[(1) - (3)].pname)->getName(), (yyvsp[(2) - (3)].strval));
                }
    break;

  case 222:
#line 878 "xi-grammar.y"
    {
                  (yyval.pname) = new Parameter(lineno, (yyvsp[(3) - (7)].type), (yyvsp[(4) - (7)].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[(6) - (7)].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[(1) - (7)].intval));
                }
    break;

  case 223:
#line 884 "xi-grammar.y"
    {
		  (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (5)].type), (yyvsp[(2) - (5)].strval));
                  (yyval.pname)->setAccelInstName((yyvsp[(4) - (5)].xstrptr));
                  (yyval.pname)->setAccelBufferType(Parameter::ACCEL_BUFFER_TYPE_READWRITE);
		}
    break;

  case 224:
#line 890 "xi-grammar.y"
    {
                  (yyval.pname) = (yyvsp[(3) - (6)].pname);
                  (yyval.pname)->setAccelInstName((yyvsp[(5) - (6)].xstrptr));
                  (yyval.pname)->setAccelBufferType((yyvsp[(1) - (6)].intval));
		}
    break;

  case 225:
#line 898 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (1)].pname)); }
    break;

  case 226:
#line 900 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (3)].pname),(yyvsp[(3) - (3)].plist)); }
    break;

  case 227:
#line 904 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (1)].pname)); }
    break;

  case 228:
#line 906 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (3)].pname),(yyvsp[(3) - (3)].plist)); }
    break;

  case 229:
#line 910 "xi-grammar.y"
    { (yyval.plist) = (yyvsp[(2) - (3)].plist); }
    break;

  case 230:
#line 912 "xi-grammar.y"
    { (yyval.plist) = 0; }
    break;

  case 231:
#line 916 "xi-grammar.y"
    { (yyval.plist) = (yyvsp[(2) - (3)].plist); }
    break;

  case 232:
#line 918 "xi-grammar.y"
    { (yyval.plist) = 0; }
    break;

  case 233:
#line 922 "xi-grammar.y"
    { (yyval.val) = 0; }
    break;

  case 234:
#line 924 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(3) - (3)].strval)); }
    break;

  case 235:
#line 928 "xi-grammar.y"
    { (yyval.sc) = 0; }
    break;

  case 236:
#line 930 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSDAGENTRY, (yyvsp[(1) - (1)].sc)); }
    break;

  case 237:
#line 932 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSDAGENTRY, (yyvsp[(2) - (3)].sc)); }
    break;

  case 238:
#line 936 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSLIST, (yyvsp[(1) - (1)].sc)); }
    break;

  case 239:
#line 938 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSLIST, (yyvsp[(1) - (2)].sc), (yyvsp[(2) - (2)].sc));  }
    break;

  case 240:
#line 942 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SOLIST, (yyvsp[(1) - (1)].sc)); }
    break;

  case 241:
#line 944 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SOLIST, (yyvsp[(1) - (2)].sc), (yyvsp[(2) - (2)].sc)); }
    break;

  case 242:
#line 948 "xi-grammar.y"
    { (yyval.sc) = 0; }
    break;

  case 243:
#line 950 "xi-grammar.y"
    { (yyval.sc) = (yyvsp[(3) - (4)].sc); }
    break;

  case 244:
#line 954 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SPUBLISHES, new SdagConstruct(SIDENT, (yyvsp[(1) - (1)].strval))); }
    break;

  case 245:
#line 956 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SPUBLISHES, new SdagConstruct(SIDENT, (yyvsp[(1) - (3)].strval)), (yyvsp[(3) - (3)].sc));  }
    break;

  case 246:
#line 960 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 247:
#line 962 "xi-grammar.y"
    { (yyval.strval) = 0; }
    break;

  case 248:
#line 966 "xi-grammar.y"
    { RemoveSdagComments((yyvsp[(4) - (6)].strval));
		   (yyval.sc) = new SdagConstruct(SATOMIC, new XStr((yyvsp[(4) - (6)].strval)), (yyvsp[(6) - (6)].sc), 0,0,0,0, 0 ); 
		   if ((yyvsp[(2) - (6)].strval)) { (yyvsp[(2) - (6)].strval)[strlen((yyvsp[(2) - (6)].strval))-1]=0; (yyval.sc)->traceName = new XStr((yyvsp[(2) - (6)].strval)+1); }
		 }
    break;

  case 249:
#line 971 "xi-grammar.y"
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

  case 250:
#line 985 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SWHEN, 0, 0,0,0,0,0,(yyvsp[(2) - (4)].entrylist)); }
    break;

  case 251:
#line 987 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SWHEN, 0, 0, 0,0,0, (yyvsp[(3) - (3)].sc), (yyvsp[(2) - (3)].entrylist)); }
    break;

  case 252:
#line 989 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SWHEN, 0, 0, 0,0,0, (yyvsp[(4) - (5)].sc), (yyvsp[(2) - (5)].entrylist)); }
    break;

  case 253:
#line 991 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SOVERLAP,0, 0,0,0,0,(yyvsp[(3) - (4)].sc), 0); }
    break;

  case 254:
#line 993 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFOR, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (11)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(5) - (11)].strval)),
		             new SdagConstruct(SINT_EXPR, (yyvsp[(7) - (11)].strval)), 0, (yyvsp[(10) - (11)].sc), 0); }
    break;

  case 255:
#line 996 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFOR, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (9)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(5) - (9)].strval)), 
		         new SdagConstruct(SINT_EXPR, (yyvsp[(7) - (9)].strval)), 0, (yyvsp[(9) - (9)].sc), 0); }
    break;

  case 256:
#line 999 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFORALL, 0, new SdagConstruct(SIDENT, (yyvsp[(3) - (12)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(6) - (12)].strval)), 
		             new SdagConstruct(SINT_EXPR, (yyvsp[(8) - (12)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(10) - (12)].strval)), (yyvsp[(12) - (12)].sc), 0); }
    break;

  case 257:
#line 1002 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFORALL, 0, new SdagConstruct(SIDENT, (yyvsp[(3) - (14)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(6) - (14)].strval)), 
		                 new SdagConstruct(SINT_EXPR, (yyvsp[(8) - (14)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(10) - (14)].strval)), (yyvsp[(13) - (14)].sc), 0); }
    break;

  case 258:
#line 1005 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (6)].strval)), (yyvsp[(6) - (6)].sc),0,0,(yyvsp[(5) - (6)].sc),0); }
    break;

  case 259:
#line 1007 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (8)].strval)), (yyvsp[(8) - (8)].sc),0,0,(yyvsp[(6) - (8)].sc),0); }
    break;

  case 260:
#line 1009 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (5)].strval)), 0,0,0,(yyvsp[(5) - (5)].sc),0); }
    break;

  case 261:
#line 1011 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SWHILE, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (7)].strval)), 0,0,0,(yyvsp[(6) - (7)].sc),0); }
    break;

  case 262:
#line 1013 "xi-grammar.y"
    { (yyval.sc) = (yyvsp[(2) - (3)].sc); }
    break;

  case 263:
#line 1015 "xi-grammar.y"
    { RemoveSdagComments((yyvsp[(2) - (3)].strval));
                   (yyval.sc) = new SdagConstruct(SATOMIC, new XStr((yyvsp[(2) - (3)].strval)), NULL, 0,0,0,0, 0 );
                 }
    break;

  case 264:
#line 1021 "xi-grammar.y"
    { (yyval.sc) = 0; }
    break;

  case 265:
#line 1023 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SELSE, 0,0,0,0,0, (yyvsp[(2) - (2)].sc),0); }
    break;

  case 266:
#line 1025 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SELSE, 0,0,0,0,0, (yyvsp[(3) - (4)].sc),0); }
    break;

  case 267:
#line 1028 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFORWARD, new SdagConstruct(SIDENT, (yyvsp[(1) - (1)].strval))); }
    break;

  case 268:
#line 1030 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFORWARD, new SdagConstruct(SIDENT, (yyvsp[(1) - (3)].strval)), (yyvsp[(3) - (3)].sc));  }
    break;

  case 269:
#line 1034 "xi-grammar.y"
    { in_int_expr = 0; (yyval.intval) = 0; }
    break;

  case 270:
#line 1038 "xi-grammar.y"
    { in_int_expr = 1; (yyval.intval) = 0; }
    break;

  case 271:
#line 1042 "xi-grammar.y"
    { 
		  if ((yyvsp[(2) - (2)].plist) != 0)
		     (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[(1) - (2)].strval), (yyvsp[(2) - (2)].plist), 0, 0, 0, 0); 
		  else
		     (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[(1) - (2)].strval), 
				new ParamList(new Parameter(0, new BuiltinType("void"))), 0, 0, 0, 0); 
		}
    break;

  case 272:
#line 1050 "xi-grammar.y"
    { if ((yyvsp[(5) - (5)].plist) != 0)
		    (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[(1) - (5)].strval), (yyvsp[(5) - (5)].plist), 0, 0, (yyvsp[(3) - (5)].strval), 0); 
		  else
		    (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[(1) - (5)].strval), new ParamList(new Parameter(0, new BuiltinType("void"))), 0, 0, (yyvsp[(3) - (5)].strval), 0); 
		}
    break;

  case 273:
#line 1058 "xi-grammar.y"
    { (yyval.entrylist) = new EntryList((yyvsp[(1) - (1)].entry)); }
    break;

  case 274:
#line 1060 "xi-grammar.y"
    { (yyval.entrylist) = new EntryList((yyvsp[(1) - (3)].entry),(yyvsp[(3) - (3)].entrylist)); }
    break;

  case 275:
#line 1064 "xi-grammar.y"
    { in_bracket=1; }
    break;

  case 276:
#line 1067 "xi-grammar.y"
    { in_bracket=0; }
    break;

  case 277:
#line 1071 "xi-grammar.y"
    { if (!macroDefined((yyvsp[(2) - (2)].strval), 1)) in_comment = 1; }
    break;

  case 278:
#line 1075 "xi-grammar.y"
    { if (!macroDefined((yyvsp[(2) - (2)].strval), 0)) in_comment = 1; }
    break;


/* Line 1267 of yacc.c.  */
#line 3624 "y.tab.c"
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


#line 1078 "xi-grammar.y"

void yyerror(const char *mesg)
{
    std::cout << cur_file<<":"<<lineno<<": Charmxi syntax error> "
	      << mesg << std::endl;
}

