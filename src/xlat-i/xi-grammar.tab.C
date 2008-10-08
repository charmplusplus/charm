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
     CLASS = 272,
     INCLUDE = 273,
     STACKSIZE = 274,
     THREADED = 275,
     TEMPLATE = 276,
     SYNC = 277,
     IGET = 278,
     EXCLUSIVE = 279,
     IMMEDIATE = 280,
     SKIPSCHED = 281,
     INLINE = 282,
     VIRTUAL = 283,
     MIGRATABLE = 284,
     CREATEHERE = 285,
     CREATEHOME = 286,
     NOKEEP = 287,
     NOTRACE = 288,
     VOID = 289,
     CONST = 290,
     PACKED = 291,
     VARSIZE = 292,
     ENTRY = 293,
     FOR = 294,
     FORALL = 295,
     WHILE = 296,
     WHEN = 297,
     OVERLAP = 298,
     ATOMIC = 299,
     FORWARD = 300,
     IF = 301,
     ELSE = 302,
     CONNECT = 303,
     PUBLISHES = 304,
     PYTHON = 305,
     LOCAL = 306,
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
     UNSIGNED = 319
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
#define CLASS 272
#define INCLUDE 273
#define STACKSIZE 274
#define THREADED 275
#define TEMPLATE 276
#define SYNC 277
#define IGET 278
#define EXCLUSIVE 279
#define IMMEDIATE 280
#define SKIPSCHED 281
#define INLINE 282
#define VIRTUAL 283
#define MIGRATABLE 284
#define CREATEHERE 285
#define CREATEHOME 286
#define NOKEEP 287
#define NOTRACE 288
#define VOID 289
#define CONST 290
#define PACKED 291
#define VARSIZE 292
#define ENTRY 293
#define FOR 294
#define FORALL 295
#define WHILE 296
#define WHEN 297
#define OVERLAP 298
#define ATOMIC 299
#define FORWARD 300
#define IF 301
#define ELSE 302
#define CONNECT 303
#define PUBLISHES 304
#define PYTHON 305
#define LOCAL 306
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




/* Copy the first part of user declarations.  */
#line 2 "xi-grammar.y"

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
#line 17 "xi-grammar.y"
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
}
/* Line 193 of yacc.c.  */
#line 275 "y.tab.c"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 288 "y.tab.c"

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
#define YYLAST   559

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  79
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  101
/* YYNRULES -- Number of rules.  */
#define YYNRULES  248
/* YYNRULES -- Number of states.  */
#define YYNSTATES  491

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   319

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    75,     2,
      73,    74,    72,     2,    69,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    66,    65,
      70,    78,    71,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    76,     2,    77,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    67,     2,    68,     2,     2,     2,     2,
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
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     5,     6,     9,    10,    12,    13,    15,
      17,    19,    24,    28,    32,    34,    39,    40,    43,    49,
      52,    55,    59,    62,    65,    68,    71,    74,    76,    78,
      80,    82,    84,    86,    90,    91,    93,    94,    98,   100,
     102,   104,   106,   109,   112,   116,   120,   123,   126,   129,
     131,   133,   136,   138,   141,   144,   146,   148,   151,   154,
     157,   166,   168,   170,   172,   174,   177,   180,   183,   185,
     187,   189,   193,   194,   197,   202,   208,   209,   211,   212,
     216,   218,   222,   224,   226,   227,   231,   233,   237,   238,
     240,   242,   243,   247,   249,   253,   255,   257,   263,   265,
     268,   272,   279,   280,   283,   285,   289,   295,   301,   307,
     313,   318,   322,   329,   336,   342,   348,   354,   360,   366,
     371,   379,   380,   383,   384,   387,   390,   394,   397,   401,
     403,   407,   412,   415,   418,   421,   424,   427,   429,   434,
     435,   438,   441,   444,   447,   450,   454,   458,   462,   466,
     473,   477,   484,   488,   495,   497,   501,   503,   506,   508,
     516,   522,   524,   526,   527,   531,   533,   537,   539,   541,
     543,   545,   547,   549,   551,   553,   555,   557,   559,   561,
     564,   566,   568,   570,   571,   573,   577,   578,   580,   586,
     592,   598,   603,   607,   609,   611,   613,   616,   621,   625,
     627,   631,   635,   638,   639,   643,   644,   646,   650,   652,
     655,   657,   660,   661,   666,   668,   672,   674,   675,   682,
     691,   696,   700,   706,   711,   723,   733,   746,   761,   768,
     777,   783,   791,   795,   799,   800,   803,   808,   810,   814,
     816,   818,   821,   827,   829,   833,   835,   837,   840
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
      80,     0,    -1,    81,    -1,    -1,    86,    81,    -1,    -1,
       5,    -1,    -1,    65,    -1,    52,    -1,    52,    -1,    85,
      66,    66,    52,    -1,     3,    84,    87,    -1,     4,    84,
      87,    -1,    65,    -1,    67,    88,    68,    83,    -1,    -1,
      89,    88,    -1,    82,    67,    88,    68,    83,    -1,    82,
      86,    -1,    82,   142,    -1,    82,   121,    65,    -1,    82,
     124,    -1,    82,   125,    -1,    82,   126,    -1,    82,   128,
      -1,    82,   139,    -1,   178,    -1,   179,    -1,   102,    -1,
      53,    -1,    54,    -1,    90,    -1,    90,    69,    91,    -1,
      -1,    91,    -1,    -1,    70,    92,    71,    -1,    58,    -1,
      59,    -1,    60,    -1,    61,    -1,    64,    58,    -1,    64,
      59,    -1,    64,    59,    58,    -1,    64,    59,    59,    -1,
      64,    60,    -1,    64,    61,    -1,    59,    59,    -1,    62,
      -1,    63,    -1,    59,    63,    -1,    34,    -1,    84,    93,
      -1,    85,    93,    -1,    94,    -1,    96,    -1,    97,    72,
      -1,    98,    72,    -1,    99,    72,    -1,   101,    73,    72,
      84,    74,    73,   160,    74,    -1,    97,    -1,    98,    -1,
      99,    -1,   100,    -1,    35,   101,    -1,   101,    35,    -1,
     101,    75,    -1,   101,    -1,    53,    -1,    85,    -1,    76,
     103,    77,    -1,    -1,   104,   105,    -1,     6,   102,    85,
     105,    -1,     6,    16,    97,    72,    84,    -1,    -1,    34,
      -1,    -1,    76,   110,    77,    -1,   111,    -1,   111,    69,
     110,    -1,    36,    -1,    37,    -1,    -1,    76,   113,    77,
      -1,   118,    -1,   118,    69,   113,    -1,    -1,    54,    -1,
      50,    -1,    -1,    76,   117,    77,    -1,   115,    -1,   115,
      69,   117,    -1,    29,    -1,    50,    -1,   102,    84,    76,
      77,    65,    -1,   119,    -1,   119,   120,    -1,    16,   109,
      95,    -1,    16,   109,    95,    67,   120,    68,    -1,    -1,
      66,   123,    -1,    95,    -1,    95,    69,   123,    -1,    11,
     112,    95,   122,   140,    -1,    12,   112,    95,   122,   140,
      -1,    13,   112,    95,   122,   140,    -1,    14,   112,    95,
     122,   140,    -1,    76,    53,    84,    77,    -1,    76,    84,
      77,    -1,    15,   116,   127,    95,   122,   140,    -1,    15,
     127,   116,    95,   122,   140,    -1,    11,   112,    84,   122,
     140,    -1,    12,   112,    84,   122,   140,    -1,    13,   112,
      84,   122,   140,    -1,    14,   112,    84,   122,   140,    -1,
      15,   127,    84,   122,   140,    -1,    16,   109,    84,    65,
      -1,    16,   109,    84,    67,   120,    68,    65,    -1,    -1,
      78,   102,    -1,    -1,    78,    53,    -1,    78,    54,    -1,
      17,    84,   134,    -1,   100,   135,    -1,   102,    84,   135,
      -1,   136,    -1,   136,    69,   137,    -1,    21,    70,   137,
      71,    -1,   138,   129,    -1,   138,   130,    -1,   138,   131,
      -1,   138,   132,    -1,   138,   133,    -1,    65,    -1,    67,
     141,    68,    83,    -1,    -1,   147,   141,    -1,   106,    65,
      -1,   107,    65,    -1,   144,    65,    -1,   143,    65,    -1,
      10,   145,    65,    -1,    18,   146,    65,    -1,    17,    84,
      65,    -1,     8,   108,    85,    -1,     8,   108,    85,    73,
     108,    74,    -1,     7,   108,    85,    -1,     7,   108,    85,
      73,   108,    74,    -1,     9,   108,    85,    -1,     9,   108,
      85,    73,   108,    74,    -1,    96,    -1,    96,    69,   145,
      -1,    54,    -1,   148,    65,    -1,   142,    -1,    38,   150,
     149,    84,   161,   162,   163,    -1,    38,   150,    84,   161,
     163,    -1,    34,    -1,    98,    -1,    -1,    76,   151,    77,
      -1,   152,    -1,   152,    69,   151,    -1,    20,    -1,    22,
      -1,    23,    -1,    24,    -1,    30,    -1,    31,    -1,    32,
      -1,    33,    -1,    25,    -1,    26,    -1,    27,    -1,    51,
      -1,    50,   114,    -1,    54,    -1,    53,    -1,    85,    -1,
      -1,    55,    -1,    55,    69,   154,    -1,    -1,    55,    -1,
      55,    76,   155,    77,   155,    -1,    55,    67,   155,    68,
     155,    -1,    55,    73,   154,    74,   155,    -1,    73,   155,
      74,   155,    -1,   102,    84,    76,    -1,    67,    -1,    68,
      -1,   102,    -1,   102,    84,    -1,   102,    84,    78,   153,
      -1,   156,   155,    77,    -1,   159,    -1,   159,    69,   160,
      -1,    73,   160,    74,    -1,    73,    74,    -1,    -1,    19,
      78,    53,    -1,    -1,   169,    -1,    67,   164,    68,    -1,
     169,    -1,   169,   164,    -1,   169,    -1,   169,   164,    -1,
      -1,    49,    73,   167,    74,    -1,    52,    -1,    52,    69,
     167,    -1,    54,    -1,    -1,    44,   168,   157,   155,   158,
     166,    -1,    48,    73,    52,   161,    74,   157,   155,    68,
      -1,    42,   175,    67,    68,    -1,    42,   175,   169,    -1,
      42,   175,    67,   164,    68,    -1,    43,    67,   165,    68,
      -1,    39,   173,   155,    65,   155,    65,   155,   172,    67,
     164,    68,    -1,    39,   173,   155,    65,   155,    65,   155,
     172,   169,    -1,    40,    76,    52,    77,   173,   155,    66,
     155,    69,   155,   172,   169,    -1,    40,    76,    52,    77,
     173,   155,    66,   155,    69,   155,   172,    67,   164,    68,
      -1,    46,   173,   155,   172,   169,   170,    -1,    46,   173,
     155,   172,    67,   164,    68,   170,    -1,    41,   173,   155,
     172,   169,    -1,    41,   173,   155,   172,    67,   164,    68,
      -1,    45,   171,    65,    -1,   157,   155,   158,    -1,    -1,
      47,   169,    -1,    47,    67,   164,    68,    -1,    52,    -1,
      52,    69,   171,    -1,    74,    -1,    73,    -1,    52,   161,
      -1,    52,   176,   155,   177,   161,    -1,   174,    -1,   174,
      69,   175,    -1,    76,    -1,    77,    -1,    56,    84,    -1,
      57,    84,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   133,   133,   138,   141,   146,   147,   152,   153,   157,
     161,   163,   171,   175,   182,   184,   189,   190,   194,   196,
     198,   200,   202,   204,   206,   208,   210,   212,   214,   218,
     220,   222,   226,   228,   233,   234,   239,   240,   244,   246,
     248,   250,   252,   254,   256,   258,   260,   262,   264,   266,
     268,   270,   272,   276,   277,   279,   281,   285,   289,   291,
     295,   299,   301,   303,   305,   308,   310,   314,   316,   320,
     322,   326,   331,   332,   336,   340,   345,   346,   351,   352,
     362,   364,   368,   370,   375,   376,   380,   382,   387,   388,
     392,   397,   398,   402,   404,   408,   410,   414,   418,   420,
     424,   426,   431,   432,   436,   438,   442,   444,   448,   452,
     456,   462,   466,   468,   472,   474,   478,   482,   486,   490,
     492,   497,   498,   503,   504,   506,   510,   512,   514,   518,
     520,   524,   528,   530,   532,   534,   536,   540,   542,   547,
     565,   569,   571,   573,   574,   576,   578,   580,   584,   586,
     588,   591,   596,   598,   602,   604,   607,   611,   613,   617,
     628,   641,   643,   648,   649,   653,   655,   659,   661,   663,
     665,   667,   669,   671,   673,   675,   677,   679,   681,   683,
     687,   689,   691,   696,   697,   699,   708,   709,   711,   717,
     723,   729,   737,   744,   752,   759,   761,   763,   765,   772,
     774,   778,   780,   785,   786,   791,   792,   794,   798,   800,
     804,   806,   811,   812,   816,   818,   822,   825,   828,   833,
     847,   849,   851,   853,   855,   858,   861,   864,   867,   869,
     871,   873,   875,   877,   884,   885,   887,   890,   892,   896,
     900,   904,   912,   920,   922,   926,   929,   933,   937
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "MODULE", "MAINMODULE", "EXTERN",
  "READONLY", "INITCALL", "INITNODE", "INITPROC", "PUPABLE", "CHARE",
  "MAINCHARE", "GROUP", "NODEGROUP", "ARRAY", "MESSAGE", "CLASS",
  "INCLUDE", "STACKSIZE", "THREADED", "TEMPLATE", "SYNC", "IGET",
  "EXCLUSIVE", "IMMEDIATE", "SKIPSCHED", "INLINE", "VIRTUAL", "MIGRATABLE",
  "CREATEHERE", "CREATEHOME", "NOKEEP", "NOTRACE", "VOID", "CONST",
  "PACKED", "VARSIZE", "ENTRY", "FOR", "FORALL", "WHILE", "WHEN",
  "OVERLAP", "ATOMIC", "FORWARD", "IF", "ELSE", "CONNECT", "PUBLISHES",
  "PYTHON", "LOCAL", "IDENT", "NUMBER", "LITERAL", "CPROGRAM", "HASHIF",
  "HASHIFDEF", "INT", "LONG", "SHORT", "CHAR", "FLOAT", "DOUBLE",
  "UNSIGNED", "';'", "':'", "'{'", "'}'", "','", "'<'", "'>'", "'*'",
  "'('", "')'", "'&'", "'['", "']'", "'='", "$accept", "File",
  "ModuleEList", "OptExtern", "OptSemiColon", "Name", "QualName", "Module",
  "ConstructEList", "ConstructList", "Construct", "TParam", "TParamList",
  "TParamEList", "OptTParams", "BuiltinType", "NamedType", "QualNamedType",
  "SimpleType", "OnePtrType", "PtrType", "FuncType", "BaseType", "Type",
  "ArrayDim", "Dim", "DimList", "Readonly", "ReadonlyMsg", "OptVoid",
  "MAttribs", "MAttribList", "MAttrib", "CAttribs", "CAttribList",
  "PythonOptions", "ArrayAttrib", "ArrayAttribs", "ArrayAttribList",
  "CAttrib", "Var", "VarList", "Message", "OptBaseList", "BaseList",
  "Chare", "Group", "NodeGroup", "ArrayIndexType", "Array", "TChare",
  "TGroup", "TNodeGroup", "TArray", "TMessage", "OptTypeInit",
  "OptNameInit", "TVar", "TVarList", "TemplateSpec", "Template",
  "MemberEList", "MemberList", "NonEntryMember", "InitNode", "InitProc",
  "PUPableClass", "IncludeFile", "Member", "Entry", "EReturn", "EAttribs",
  "EAttribList", "EAttrib", "DefaultParameter", "CPROGRAM_List", "CCode",
  "ParamBracketStart", "ParamBraceStart", "ParamBraceEnd", "Parameter",
  "ParamList", "EParameters", "OptStackSize", "OptSdagCode", "Slist",
  "Olist", "OptPubList", "PublishesList", "OptTraceName",
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
     315,   316,   317,   318,   319,    59,    58,   123,   125,    44,
      60,    62,    42,    40,    41,    38,    91,    93,    61
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    79,    80,    81,    81,    82,    82,    83,    83,    84,
      85,    85,    86,    86,    87,    87,    88,    88,    89,    89,
      89,    89,    89,    89,    89,    89,    89,    89,    89,    90,
      90,    90,    91,    91,    92,    92,    93,    93,    94,    94,
      94,    94,    94,    94,    94,    94,    94,    94,    94,    94,
      94,    94,    94,    95,    96,    97,    97,    98,    99,    99,
     100,   101,   101,   101,   101,   101,   101,   102,   102,   103,
     103,   104,   105,   105,   106,   107,   108,   108,   109,   109,
     110,   110,   111,   111,   112,   112,   113,   113,   114,   114,
     115,   116,   116,   117,   117,   118,   118,   119,   120,   120,
     121,   121,   122,   122,   123,   123,   124,   124,   125,   126,
     127,   127,   128,   128,   129,   129,   130,   131,   132,   133,
     133,   134,   134,   135,   135,   135,   136,   136,   136,   137,
     137,   138,   139,   139,   139,   139,   139,   140,   140,   141,
     141,   142,   142,   142,   142,   142,   142,   142,   143,   143,
     143,   143,   144,   144,   145,   145,   146,   147,   147,   148,
     148,   149,   149,   150,   150,   151,   151,   152,   152,   152,
     152,   152,   152,   152,   152,   152,   152,   152,   152,   152,
     153,   153,   153,   154,   154,   154,   155,   155,   155,   155,
     155,   155,   156,   157,   158,   159,   159,   159,   159,   160,
     160,   161,   161,   162,   162,   163,   163,   163,   164,   164,
     165,   165,   166,   166,   167,   167,   168,   168,   169,   169,
     169,   169,   169,   169,   169,   169,   169,   169,   169,   169,
     169,   169,   169,   169,   170,   170,   170,   171,   171,   172,
     173,   174,   174,   175,   175,   176,   177,   178,   179
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     0,     2,     0,     1,     0,     1,     1,
       1,     4,     3,     3,     1,     4,     0,     2,     5,     2,
       2,     3,     2,     2,     2,     2,     2,     1,     1,     1,
       1,     1,     1,     3,     0,     1,     0,     3,     1,     1,
       1,     1,     2,     2,     3,     3,     2,     2,     2,     1,
       1,     2,     1,     2,     2,     1,     1,     2,     2,     2,
       8,     1,     1,     1,     1,     2,     2,     2,     1,     1,
       1,     3,     0,     2,     4,     5,     0,     1,     0,     3,
       1,     3,     1,     1,     0,     3,     1,     3,     0,     1,
       1,     0,     3,     1,     3,     1,     1,     5,     1,     2,
       3,     6,     0,     2,     1,     3,     5,     5,     5,     5,
       4,     3,     6,     6,     5,     5,     5,     5,     5,     4,
       7,     0,     2,     0,     2,     2,     3,     2,     3,     1,
       3,     4,     2,     2,     2,     2,     2,     1,     4,     0,
       2,     2,     2,     2,     2,     3,     3,     3,     3,     6,
       3,     6,     3,     6,     1,     3,     1,     2,     1,     7,
       5,     1,     1,     0,     3,     1,     3,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     2,
       1,     1,     1,     0,     1,     3,     0,     1,     5,     5,
       5,     4,     3,     1,     1,     1,     2,     4,     3,     1,
       3,     3,     2,     0,     3,     0,     1,     3,     1,     2,
       1,     2,     0,     4,     1,     3,     1,     0,     6,     8,
       4,     3,     5,     4,    11,     9,    12,    14,     6,     8,
       5,     7,     3,     3,     0,     2,     4,     1,     3,     1,
       1,     2,     5,     1,     3,     1,     1,     2,     2
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       3,     0,     0,     0,     2,     3,     9,     0,     0,     1,
       4,    14,     5,    12,    13,     6,     0,     0,     0,     0,
       5,    27,    28,   247,   248,     0,    76,    76,    76,     0,
      84,    84,    84,    84,     0,    78,     0,     0,     0,     5,
      19,     0,     0,     0,    22,    23,    24,    25,     0,    26,
      20,     0,     0,     7,    17,     0,    52,     0,    10,    38,
      39,    40,    41,    49,    50,     0,    36,    55,    56,    61,
      62,    63,    64,    68,     0,    77,     0,     0,     0,   154,
       0,     0,     0,     0,     0,     0,     0,     0,    91,     0,
       0,     0,   156,     0,     0,     0,   141,   142,    21,    84,
      84,    84,    84,     0,    78,   132,   133,   134,   135,   136,
     144,   143,     8,    15,     0,    65,    48,    51,    42,    43,
      46,    47,     0,    34,    54,    57,    58,    59,    66,     0,
      67,    72,   150,   148,   152,     0,   145,    95,    96,     0,
      86,    36,   102,   102,   102,   102,    90,     0,     0,    93,
       0,     0,     0,     0,     0,    82,    83,     0,    80,   100,
     147,   146,     0,    64,     0,   129,     0,     7,     0,     0,
       0,     0,     0,     0,     0,    44,    45,     0,    30,    31,
      32,    35,     0,    29,     0,     0,    72,    74,    76,    76,
      76,   155,    85,     0,    53,     0,     0,     0,     0,     0,
       0,   111,     0,    92,   102,   102,    79,     0,     0,   121,
       0,   127,   123,     0,   131,    18,   102,   102,   102,   102,
     102,     0,    75,    11,     0,    37,     0,    69,    70,     0,
      73,     0,     0,     0,    87,   104,   103,   137,   139,   106,
     107,   108,   109,   110,    94,     0,     0,    81,     0,    98,
       0,     0,   126,   124,   125,   128,   130,     0,     0,     0,
       0,     0,   119,     0,    33,     0,    71,   151,   149,   153,
       0,   163,     0,   158,   139,     0,   112,   113,     0,    99,
     101,   122,   114,   115,   116,   117,   118,     0,     0,   105,
       0,     0,     7,   140,   157,     0,     0,   195,   186,   199,
       0,   167,   168,   169,   170,   175,   176,   177,   171,   172,
     173,   174,    88,   178,     0,   165,    52,    10,     0,     0,
     162,     0,   138,     0,   120,   196,   187,   186,     0,     0,
      60,    89,   179,   164,     0,     0,   205,     0,    97,   192,
       0,   186,   183,   186,     0,   198,   200,   166,   202,     0,
       0,     0,     0,     0,     0,   217,     0,     0,     0,   193,
     186,   160,   206,   203,   181,   180,   182,   197,     0,   184,
       0,     0,   186,   201,   240,   186,     0,   186,     0,   243,
       0,     0,   216,     0,   237,     0,   186,     0,   193,     0,
     208,     0,     0,   205,   186,   183,   186,   186,   191,     0,
       0,     0,   245,   241,   186,     0,   193,   221,     0,   210,
     186,     0,   232,     0,     0,   207,   209,   194,   233,     0,
     159,   189,   185,   190,   188,   186,     0,   239,     0,     0,
     244,   220,     0,   223,   211,     0,   238,     0,     0,   204,
       0,   186,   193,   230,   246,     0,   222,   212,   193,   234,
       0,   186,     0,     0,   242,     0,   218,     0,     0,   228,
     186,     0,   186,   231,     0,   234,   193,   235,     0,     0,
       0,   214,     0,   229,     0,   219,   193,   225,   186,     0,
     213,   236,     0,     0,   215,   224,     0,   193,   226,     0,
     227
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     3,     4,    18,   113,   141,    66,     5,    13,    19,
      20,   180,   181,   182,   124,    67,   235,    68,    69,    70,
      71,    72,    73,   248,   229,   186,   187,    41,    42,    76,
      90,   157,   158,    82,   139,   332,   149,    87,   150,   140,
     249,   250,    43,   196,   236,    44,    45,    46,    88,    47,
     105,   106,   107,   108,   109,   252,   211,   165,   166,    48,
      49,   239,   272,   273,    51,    52,    80,    93,   274,   275,
     321,   291,   314,   315,   367,   370,   328,   298,   360,   418,
     299,   300,   336,   393,   361,   389,   408,   456,   472,   383,
     390,   459,   385,   428,   375,   379,   380,   404,   445,    21,
      22
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -387
static const yytype_int16 yypact[] =
{
      80,   -20,   -20,    47,  -387,    80,  -387,   117,   117,  -387,
    -387,  -387,    13,  -387,  -387,  -387,   -20,   -20,   196,   -16,
      13,  -387,  -387,  -387,  -387,   214,    38,    38,    38,    86,
      67,    67,    67,    67,   105,   150,   -20,   108,   123,    13,
    -387,   133,   164,   166,  -387,  -387,  -387,  -387,   270,  -387,
    -387,   177,   180,   187,  -387,   354,  -387,   335,  -387,  -387,
      36,  -387,  -387,  -387,  -387,    52,    56,  -387,  -387,   181,
     182,   190,  -387,    -9,    86,  -387,    86,    86,    86,   195,
     202,   -17,   -20,   -20,   -20,   -20,    99,   192,   194,   111,
     -20,   206,  -387,   222,   263,   211,  -387,  -387,  -387,    67,
      67,    67,    67,   192,   150,  -387,  -387,  -387,  -387,  -387,
    -387,  -387,  -387,  -387,   216,     4,  -387,  -387,  -387,   100,
    -387,  -387,   223,   322,  -387,  -387,  -387,  -387,  -387,   219,
    -387,   -21,    43,    66,    68,    86,  -387,  -387,  -387,   215,
     224,   225,   228,   228,   228,   228,  -387,   -20,   226,   230,
     227,   139,   -20,   250,   -20,  -387,  -387,   231,   232,   235,
    -387,  -387,   -20,   -35,   -20,   236,   238,   187,   -20,   -20,
     -20,   -20,   -20,   -20,   -20,  -387,  -387,   255,  -387,  -387,
     244,  -387,   243,  -387,   -20,   163,   240,  -387,    38,    38,
      38,  -387,  -387,   -17,  -387,   -20,   121,   121,   121,   121,
     241,  -387,   250,  -387,   228,   228,  -387,   111,   335,   239,
     197,  -387,   251,   263,  -387,  -387,   228,   228,   228,   228,
     228,   122,  -387,  -387,   322,  -387,   245,  -387,   264,   254,
    -387,   258,   259,   268,  -387,   280,  -387,  -387,   147,  -387,
    -387,  -387,  -387,  -387,  -387,   121,   121,  -387,   -20,   335,
     287,   335,  -387,  -387,  -387,  -387,  -387,   121,   121,   121,
     121,   121,  -387,   335,  -387,   285,  -387,  -387,  -387,  -387,
     -20,   283,   292,  -387,   147,   296,  -387,  -387,   286,  -387,
    -387,  -387,  -387,  -387,  -387,  -387,  -387,   295,   335,  -387,
     321,   367,   187,  -387,  -387,   288,   299,   -20,   -27,   297,
     293,  -387,  -387,  -387,  -387,  -387,  -387,  -387,  -387,  -387,
    -387,  -387,   314,  -387,   300,   304,   326,   306,   316,   181,
    -387,   -20,  -387,   325,  -387,   -13,    48,   -27,   315,   335,
    -387,  -387,  -387,  -387,   321,   276,   179,   316,  -387,  -387,
     126,   -27,   336,   -27,   328,  -387,  -387,  -387,  -387,   329,
     327,   331,   327,   352,   338,   355,   356,   327,   337,   393,
     -27,  -387,  -387,   392,  -387,  -387,   264,  -387,   353,   351,
     348,   346,   -27,  -387,  -387,   -27,   372,   -27,    64,   371,
     403,   393,  -387,   383,   434,   437,   -27,   452,  -387,   438,
     393,   439,   427,   179,   -27,   336,   -27,   -27,  -387,   444,
     431,   440,  -387,  -387,   -27,   352,   193,  -387,   445,   393,
     -27,   356,  -387,   440,   316,  -387,  -387,  -387,  -387,   458,
    -387,  -387,  -387,  -387,  -387,   -27,   327,  -387,   413,   435,
    -387,  -387,   447,  -387,  -387,   439,  -387,   423,   442,  -387,
     454,   -27,   393,  -387,  -387,   316,  -387,   468,   393,   471,
     383,   -27,   455,   456,  -387,   449,  -387,   457,   433,  -387,
     -27,   440,   -27,  -387,   474,   471,   393,  -387,   459,   443,
     460,   461,   462,  -387,   463,  -387,   393,  -387,   -27,   474,
    -387,  -387,   464,   440,  -387,  -387,   453,   393,  -387,   465,
    -387
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -387,  -387,   518,  -387,  -157,    -1,   -71,   510,   526,     1,
    -387,  -387,   311,  -387,   396,  -387,   -34,   -18,   -51,   247,
    -387,   -86,   482,   -23,  -387,  -387,   357,  -387,  -387,   -14,
     436,   334,  -387,    29,   349,  -387,  -387,   466,   342,  -387,
    -387,  -167,  -387,  -114,   275,  -387,  -387,  -387,   -50,  -387,
    -387,  -387,  -387,  -387,  -387,  -387,   339,  -387,   333,  -387,
    -387,    -2,   273,   530,  -387,  -387,   414,  -387,  -387,  -387,
    -387,  -387,   218,  -387,  -387,   155,  -318,  -387,  -363,   118,
    -387,  -262,  -320,  -387,   162,  -368,  -387,  -387,    77,  -387,
    -292,    92,   148,  -386,  -333,  -387,   153,  -387,  -387,  -387,
    -387
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -162
static const yytype_int16 yytable[] =
{
       7,     8,    74,   131,   114,   132,   133,   134,   163,   344,
     215,    79,   137,    77,    78,    23,    24,   363,    15,   377,
     410,    54,   416,   368,   386,   371,   128,   437,   326,   197,
     198,   199,     6,   138,  -123,    91,  -123,   152,   432,   128,
      95,   434,   391,   210,   362,   122,   327,     9,   142,   143,
     144,   145,    53,   172,   398,   185,   159,   399,   403,   401,
      83,    84,    85,   339,   129,   340,   130,   346,   413,    16,
      17,   164,    75,   349,   453,   469,   421,   129,   423,   424,
     457,   -16,   279,     1,     2,   148,   429,   460,   407,   409,
     245,   246,   435,   441,   438,   116,   287,   486,   474,   117,
     183,   362,   257,   258,   259,   260,   261,   440,   482,   122,
     118,   119,   120,   121,   228,   341,   188,    79,   204,   489,
     205,   342,   122,   452,   343,   454,   123,   163,   168,   169,
     170,   171,   122,   461,   122,   322,   443,   335,    58,   189,
     402,   190,   468,    81,   470,   449,   200,   155,   156,   146,
     148,     6,   147,    25,    26,    27,    28,    29,   175,   176,
     483,   209,    92,   212,    36,    37,   467,   216,   217,   218,
     219,   220,   221,   222,   231,   232,   233,   477,    58,   364,
     365,    86,    11,   226,    12,   271,   237,   262,   238,   263,
     164,     6,   147,    94,   488,   240,   241,   242,    96,     1,
       2,   183,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    37,    58,   227,    38,   350,   351,
     352,   353,   354,   355,   356,   357,    89,   358,   281,    97,
      55,    98,   350,   351,   352,   353,   354,   355,   356,   357,
     319,   358,   110,   276,   277,   111,   359,   278,    56,    57,
     253,   254,   112,   125,   126,   282,   283,   284,   285,   286,
     388,   431,   127,    39,   135,   297,    58,   136,   151,   366,
     153,   160,    59,    60,    61,    62,    63,    64,    65,   167,
     162,    99,   100,   101,   102,   103,   104,   161,   174,   177,
     318,   184,   192,   193,   195,   123,   325,    56,    57,   202,
     146,   207,   208,   201,   203,   213,   297,   223,   206,   214,
      56,    57,   297,   224,   225,    58,   185,   251,   243,   265,
     337,    59,    60,    61,    62,    63,    64,    65,    58,   210,
     122,   266,   267,   268,    59,    60,    61,    62,    63,    64,
      65,   301,   269,   302,   303,   304,   305,   306,   307,   270,
     348,   308,   309,   310,   311,   280,    56,    57,   288,   290,
     292,   294,   295,   296,   324,   323,   329,   330,   331,    56,
      57,   312,   313,   334,    58,   178,   179,   333,  -161,    -9,
      59,    60,    61,    62,    63,    64,    65,    58,    56,   335,
     338,   369,   345,    59,    60,    61,    62,    63,    64,    65,
     374,   316,   372,   373,   378,   381,    58,   376,   384,   382,
     387,   392,    59,    60,    61,    62,    63,    64,    65,   317,
     395,   394,   396,   397,   400,    59,    60,    61,    62,    63,
      64,    65,   350,   351,   352,   353,   354,   355,   356,   357,
     405,   358,   350,   351,   352,   353,   354,   355,   356,   357,
     388,   358,   350,   351,   352,   353,   354,   355,   356,   357,
     388,   358,   350,   351,   352,   353,   354,   355,   356,   357,
     406,   358,   350,   351,   352,   353,   354,   355,   356,   357,
     442,   358,   350,   351,   352,   353,   354,   355,   356,   357,
     448,   358,   350,   351,   352,   353,   354,   355,   356,   357,
     466,   358,   412,   411,   414,   419,   415,   417,   426,   425,
     476,   439,   444,   433,   427,   446,   450,   455,   458,   451,
     487,   462,   464,    10,   463,   465,   471,   475,    40,   478,
     479,   481,   485,   490,    14,   264,   480,   194,   320,   115,
     173,   247,   234,   230,   244,   289,   256,   293,    50,   191,
     422,   255,   347,   447,   154,   420,   484,   473,   430,   436
};

static const yytype_uint16 yycheck[] =
{
       1,     2,    25,    74,    55,    76,    77,    78,    94,   327,
     167,    29,    29,    27,    28,    16,    17,   337,     5,   352,
     383,    20,   390,   341,   357,   343,    35,   413,    55,   143,
     144,   145,    52,    50,    69,    36,    71,    87,   406,    35,
      39,   409,   360,    78,   336,    66,    73,     0,    82,    83,
      84,    85,    68,   103,   372,    76,    90,   375,   378,   377,
      31,    32,    33,    76,    73,    78,    75,   329,   386,    56,
      57,    94,    34,   335,   442,   461,   394,    73,   396,   397,
     448,    68,   249,     3,     4,    86,   404,   450,   380,   381,
     204,   205,   410,   426,   414,    59,   263,   483,   466,    63,
     123,   393,   216,   217,   218,   219,   220,   425,   476,    66,
      58,    59,    60,    61,   185,    67,    73,   135,   152,   487,
     154,    73,    66,   441,    76,   445,    70,   213,    99,   100,
     101,   102,    66,   451,    66,   292,   428,    73,    52,    73,
      76,    73,   460,    76,   462,   437,   147,    36,    37,    50,
     151,    52,    53,     6,     7,     8,     9,    10,    58,    59,
     478,   162,    54,   164,    17,    18,   458,   168,   169,   170,
     171,   172,   173,   174,   188,   189,   190,   469,    52,    53,
      54,    76,    65,   184,    67,    38,    65,    65,    67,    67,
     213,    52,    53,    70,   486,   197,   198,   199,    65,     3,
       4,   224,     6,     7,     8,     9,    10,    11,    12,    13,
      14,    15,    16,    17,    18,    52,    53,    21,    39,    40,
      41,    42,    43,    44,    45,    46,    76,    48,   251,    65,
      16,    65,    39,    40,    41,    42,    43,    44,    45,    46,
     291,    48,    65,   245,   246,    65,    67,   248,    34,    35,
      53,    54,    65,    72,    72,   257,   258,   259,   260,   261,
      67,    68,    72,    67,    69,   288,    52,    65,    76,   340,
      76,    65,    58,    59,    60,    61,    62,    63,    64,    68,
      17,    11,    12,    13,    14,    15,    16,    65,    72,    66,
     291,    72,    77,    69,    66,    70,   297,    34,    35,    69,
      50,    69,    67,    77,    77,    69,   329,    52,    77,    71,
      34,    35,   335,    69,    71,    52,    76,    78,    77,    74,
     321,    58,    59,    60,    61,    62,    63,    64,    52,    78,
      66,    77,    74,    74,    58,    59,    60,    61,    62,    63,
      64,    20,    74,    22,    23,    24,    25,    26,    27,    69,
      74,    30,    31,    32,    33,    68,    34,    35,    73,    76,
      68,    65,    76,    68,    65,    77,    69,    74,    54,    34,
      35,    50,    51,    69,    52,    53,    54,    77,    52,    73,
      58,    59,    60,    61,    62,    63,    64,    52,    34,    73,
      65,    55,    77,    58,    59,    60,    61,    62,    63,    64,
      73,    34,    74,    74,    52,    67,    52,    76,    52,    54,
      73,    19,    58,    59,    60,    61,    62,    63,    64,    52,
      69,    68,    74,    77,    52,    58,    59,    60,    61,    62,
      63,    64,    39,    40,    41,    42,    43,    44,    45,    46,
      69,    48,    39,    40,    41,    42,    43,    44,    45,    46,
      67,    48,    39,    40,    41,    42,    43,    44,    45,    46,
      67,    48,    39,    40,    41,    42,    43,    44,    45,    46,
      67,    48,    39,    40,    41,    42,    43,    44,    45,    46,
      67,    48,    39,    40,    41,    42,    43,    44,    45,    46,
      67,    48,    39,    40,    41,    42,    43,    44,    45,    46,
      67,    48,    65,    69,    52,    78,    68,    68,    77,    65,
      67,    53,    77,    68,    74,    68,    74,    49,    47,    65,
      67,    66,    73,     5,    68,    68,    52,    68,    18,    69,
      69,    68,    68,    68,     8,   224,    74,   141,   291,    57,
     104,   207,   193,   186,   202,   270,   213,   274,    18,   135,
     395,   212,   334,   435,    88,   393,   479,   465,   405,   411
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     3,     4,    80,    81,    86,    52,    84,    84,     0,
      81,    65,    67,    87,    87,     5,    56,    57,    82,    88,
      89,   178,   179,    84,    84,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    17,    18,    21,    67,
      86,   106,   107,   121,   124,   125,   126,   128,   138,   139,
     142,   143,   144,    68,    88,    16,    34,    35,    52,    58,
      59,    60,    61,    62,    63,    64,    85,    94,    96,    97,
      98,    99,   100,   101,   102,    34,   108,   108,   108,    96,
     145,    76,   112,   112,   112,   112,    76,   116,   127,    76,
     109,    84,    54,   146,    70,    88,    65,    65,    65,    11,
      12,    13,    14,    15,    16,   129,   130,   131,   132,   133,
      65,    65,    65,    83,    97,   101,    59,    63,    58,    59,
      60,    61,    66,    70,    93,    72,    72,    72,    35,    73,
      75,    85,    85,    85,    85,    69,    65,    29,    50,   113,
     118,    84,    95,    95,    95,    95,    50,    53,    84,   115,
     117,    76,   127,    76,   116,    36,    37,   110,   111,    95,
      65,    65,    17,   100,   102,   136,   137,    68,   112,   112,
     112,   112,   127,   109,    72,    58,    59,    66,    53,    54,
      90,    91,    92,   102,    72,    76,   104,   105,    73,    73,
      73,   145,    77,    69,    93,    66,   122,   122,   122,   122,
      84,    77,    69,    77,    95,    95,    77,    69,    67,    84,
      78,   135,    84,    69,    71,    83,    84,    84,    84,    84,
      84,    84,    84,    52,    69,    71,    84,    53,    85,   103,
     105,   108,   108,   108,   113,    95,   123,    65,    67,   140,
     140,   140,   140,    77,   117,   122,   122,   110,   102,   119,
     120,    78,   134,    53,    54,   135,   137,   122,   122,   122,
     122,   122,    65,    67,    91,    74,    77,    74,    74,    74,
      69,    38,   141,   142,   147,   148,   140,   140,    84,   120,
      68,   102,   140,   140,   140,   140,   140,   120,    73,   123,
      76,   150,    68,   141,    65,    76,    68,   102,   156,   159,
     160,    20,    22,    23,    24,    25,    26,    27,    30,    31,
      32,    33,    50,    51,   151,   152,    34,    52,    84,    97,
      98,   149,    83,    77,    65,    84,    55,    73,   155,    69,
      74,    54,   114,    77,    69,    73,   161,    84,    65,    76,
      78,    67,    73,    76,   155,    77,   160,   151,    74,   160,
      39,    40,    41,    42,    43,    44,    45,    46,    48,    67,
     157,   163,   169,   161,    53,    54,    85,   153,   155,    55,
     154,   155,    74,    74,    73,   173,    76,   173,    52,   174,
     175,    67,    54,   168,    52,   171,   173,    73,    67,   164,
     169,   155,    19,   162,    68,    69,    74,    77,   155,   155,
      52,   155,    76,   161,   176,    69,    67,   169,   165,   169,
     157,    69,    65,   155,    52,    68,   164,    68,   158,    78,
     163,   155,   154,   155,   155,    65,    77,    74,   172,   155,
     175,    68,   164,    68,   164,   155,   171,   172,   161,    53,
     155,   173,    67,   169,    77,   177,    68,   158,    67,   169,
      74,    65,   155,   164,   161,    49,   166,   164,    47,   170,
     157,   155,    66,    68,    73,    68,    67,   169,   155,   172,
     155,    52,   167,   170,   164,    68,    67,   169,    69,    69,
      74,    68,   164,   155,   167,    68,   172,    67,   169,   164,
      68
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
#line 134 "xi-grammar.y"
    { (yyval.modlist) = (yyvsp[(1) - (1)].modlist); modlist = (yyvsp[(1) - (1)].modlist); }
    break;

  case 3:
#line 138 "xi-grammar.y"
    { 
		  (yyval.modlist) = 0; 
		}
    break;

  case 4:
#line 142 "xi-grammar.y"
    { (yyval.modlist) = new ModuleList(lineno, (yyvsp[(1) - (2)].module), (yyvsp[(2) - (2)].modlist)); }
    break;

  case 5:
#line 146 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 6:
#line 148 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 7:
#line 152 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 8:
#line 154 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 9:
#line 158 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 10:
#line 162 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 11:
#line 164 "xi-grammar.y"
    {
		  char *tmp = new char[strlen((yyvsp[(1) - (4)].strval))+strlen((yyvsp[(4) - (4)].strval))+3];
		  sprintf(tmp,"%s::%s", (yyvsp[(1) - (4)].strval), (yyvsp[(4) - (4)].strval));
		  (yyval.strval) = tmp;
		}
    break;

  case 12:
#line 172 "xi-grammar.y"
    { 
		    (yyval.module) = new Module(lineno, (yyvsp[(2) - (3)].strval), (yyvsp[(3) - (3)].conslist)); 
		}
    break;

  case 13:
#line 176 "xi-grammar.y"
    {  
		    (yyval.module) = new Module(lineno, (yyvsp[(2) - (3)].strval), (yyvsp[(3) - (3)].conslist)); 
		    (yyval.module)->setMain();
		}
    break;

  case 14:
#line 183 "xi-grammar.y"
    { (yyval.conslist) = 0; }
    break;

  case 15:
#line 185 "xi-grammar.y"
    { (yyval.conslist) = (yyvsp[(2) - (4)].conslist); }
    break;

  case 16:
#line 189 "xi-grammar.y"
    { (yyval.conslist) = 0; }
    break;

  case 17:
#line 191 "xi-grammar.y"
    { (yyval.conslist) = new ConstructList(lineno, (yyvsp[(1) - (2)].construct), (yyvsp[(2) - (2)].conslist)); }
    break;

  case 18:
#line 195 "xi-grammar.y"
    { if((yyvsp[(3) - (5)].conslist)) (yyvsp[(3) - (5)].conslist)->setExtern((yyvsp[(1) - (5)].intval)); (yyval.construct) = (yyvsp[(3) - (5)].conslist); }
    break;

  case 19:
#line 197 "xi-grammar.y"
    { (yyvsp[(2) - (2)].module)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].module); }
    break;

  case 20:
#line 199 "xi-grammar.y"
    { (yyvsp[(2) - (2)].member)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].member); }
    break;

  case 21:
#line 201 "xi-grammar.y"
    { (yyvsp[(2) - (3)].message)->setExtern((yyvsp[(1) - (3)].intval)); (yyval.construct) = (yyvsp[(2) - (3)].message); }
    break;

  case 22:
#line 203 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 23:
#line 205 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 24:
#line 207 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 25:
#line 209 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 26:
#line 211 "xi-grammar.y"
    { (yyvsp[(2) - (2)].templat)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].templat); }
    break;

  case 27:
#line 213 "xi-grammar.y"
    { (yyval.construct) = NULL; }
    break;

  case 28:
#line 215 "xi-grammar.y"
    { (yyval.construct) = NULL; }
    break;

  case 29:
#line 219 "xi-grammar.y"
    { (yyval.tparam) = new TParamType((yyvsp[(1) - (1)].type)); }
    break;

  case 30:
#line 221 "xi-grammar.y"
    { (yyval.tparam) = new TParamVal((yyvsp[(1) - (1)].strval)); }
    break;

  case 31:
#line 223 "xi-grammar.y"
    { (yyval.tparam) = new TParamVal((yyvsp[(1) - (1)].strval)); }
    break;

  case 32:
#line 227 "xi-grammar.y"
    { (yyval.tparlist) = new TParamList((yyvsp[(1) - (1)].tparam)); }
    break;

  case 33:
#line 229 "xi-grammar.y"
    { (yyval.tparlist) = new TParamList((yyvsp[(1) - (3)].tparam), (yyvsp[(3) - (3)].tparlist)); }
    break;

  case 34:
#line 233 "xi-grammar.y"
    { (yyval.tparlist) = 0; }
    break;

  case 35:
#line 235 "xi-grammar.y"
    { (yyval.tparlist) = (yyvsp[(1) - (1)].tparlist); }
    break;

  case 36:
#line 239 "xi-grammar.y"
    { (yyval.tparlist) = 0; }
    break;

  case 37:
#line 241 "xi-grammar.y"
    { (yyval.tparlist) = (yyvsp[(2) - (3)].tparlist); }
    break;

  case 38:
#line 245 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("int"); }
    break;

  case 39:
#line 247 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long"); }
    break;

  case 40:
#line 249 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("short"); }
    break;

  case 41:
#line 251 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("char"); }
    break;

  case 42:
#line 253 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned int"); }
    break;

  case 43:
#line 255 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long"); }
    break;

  case 44:
#line 257 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long"); }
    break;

  case 45:
#line 259 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long long"); }
    break;

  case 46:
#line 261 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned short"); }
    break;

  case 47:
#line 263 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned char"); }
    break;

  case 48:
#line 265 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long long"); }
    break;

  case 49:
#line 267 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("float"); }
    break;

  case 50:
#line 269 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("double"); }
    break;

  case 51:
#line 271 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long double"); }
    break;

  case 52:
#line 273 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("void"); }
    break;

  case 53:
#line 276 "xi-grammar.y"
    { (yyval.ntype) = new NamedType((yyvsp[(1) - (2)].strval),(yyvsp[(2) - (2)].tparlist)); }
    break;

  case 54:
#line 277 "xi-grammar.y"
    { (yyval.ntype) = new NamedType((yyvsp[(1) - (2)].strval),(yyvsp[(2) - (2)].tparlist)); }
    break;

  case 55:
#line 280 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 56:
#line 282 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ntype); }
    break;

  case 57:
#line 286 "xi-grammar.y"
    { (yyval.ptype) = new PtrType((yyvsp[(1) - (2)].type)); }
    break;

  case 58:
#line 290 "xi-grammar.y"
    { (yyvsp[(1) - (2)].ptype)->indirect(); (yyval.ptype) = (yyvsp[(1) - (2)].ptype); }
    break;

  case 59:
#line 292 "xi-grammar.y"
    { (yyvsp[(1) - (2)].ptype)->indirect(); (yyval.ptype) = (yyvsp[(1) - (2)].ptype); }
    break;

  case 60:
#line 296 "xi-grammar.y"
    { (yyval.ftype) = new FuncType((yyvsp[(1) - (8)].type), (yyvsp[(4) - (8)].strval), (yyvsp[(7) - (8)].plist)); }
    break;

  case 61:
#line 300 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 62:
#line 302 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 63:
#line 304 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 64:
#line 306 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ftype); }
    break;

  case 65:
#line 309 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(2) - (2)].type); }
    break;

  case 66:
#line 311 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (2)].type); }
    break;

  case 67:
#line 315 "xi-grammar.y"
    { (yyval.type) = new ReferenceType((yyvsp[(1) - (2)].type)); }
    break;

  case 68:
#line 317 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 69:
#line 321 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 70:
#line 323 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 71:
#line 327 "xi-grammar.y"
    { (yyval.val) = (yyvsp[(2) - (3)].val); }
    break;

  case 72:
#line 331 "xi-grammar.y"
    { (yyval.vallist) = 0; }
    break;

  case 73:
#line 333 "xi-grammar.y"
    { (yyval.vallist) = new ValueList((yyvsp[(1) - (2)].val), (yyvsp[(2) - (2)].vallist)); }
    break;

  case 74:
#line 337 "xi-grammar.y"
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[(2) - (4)].type), (yyvsp[(3) - (4)].strval), (yyvsp[(4) - (4)].vallist)); }
    break;

  case 75:
#line 341 "xi-grammar.y"
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[(3) - (5)].type), (yyvsp[(5) - (5)].strval), 0, 1); }
    break;

  case 76:
#line 345 "xi-grammar.y"
    { (yyval.intval) = 0;}
    break;

  case 77:
#line 347 "xi-grammar.y"
    { (yyval.intval) = 0;}
    break;

  case 78:
#line 351 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 79:
#line 353 "xi-grammar.y"
    { 
		  /*
		  printf("Warning: Message attributes are being phased out.\n");
		  printf("Warning: Please remove them from interface files.\n");
		  */
		  (yyval.intval) = (yyvsp[(2) - (3)].intval); 
		}
    break;

  case 80:
#line 363 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (1)].intval); }
    break;

  case 81:
#line 365 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (3)].intval) | (yyvsp[(3) - (3)].intval); }
    break;

  case 82:
#line 369 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 83:
#line 371 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 84:
#line 375 "xi-grammar.y"
    { (yyval.cattr) = 0; }
    break;

  case 85:
#line 377 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(2) - (3)].cattr); }
    break;

  case 86:
#line 381 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (1)].cattr); }
    break;

  case 87:
#line 383 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (3)].cattr) | (yyvsp[(3) - (3)].cattr); }
    break;

  case 88:
#line 387 "xi-grammar.y"
    { python_doc = NULL; (yyval.intval) = 0; }
    break;

  case 89:
#line 389 "xi-grammar.y"
    { python_doc = (yyvsp[(1) - (1)].strval); (yyval.intval) = 0; }
    break;

  case 90:
#line 393 "xi-grammar.y"
    { (yyval.cattr) = Chare::CPYTHON; }
    break;

  case 91:
#line 397 "xi-grammar.y"
    { (yyval.cattr) = 0; }
    break;

  case 92:
#line 399 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(2) - (3)].cattr); }
    break;

  case 93:
#line 403 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (1)].cattr); }
    break;

  case 94:
#line 405 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (3)].cattr) | (yyvsp[(3) - (3)].cattr); }
    break;

  case 95:
#line 409 "xi-grammar.y"
    { (yyval.cattr) = Chare::CMIGRATABLE; }
    break;

  case 96:
#line 411 "xi-grammar.y"
    { (yyval.cattr) = Chare::CPYTHON; }
    break;

  case 97:
#line 415 "xi-grammar.y"
    { (yyval.mv) = new MsgVar((yyvsp[(1) - (5)].type), (yyvsp[(2) - (5)].strval)); }
    break;

  case 98:
#line 419 "xi-grammar.y"
    { (yyval.mvlist) = new MsgVarList((yyvsp[(1) - (1)].mv)); }
    break;

  case 99:
#line 421 "xi-grammar.y"
    { (yyval.mvlist) = new MsgVarList((yyvsp[(1) - (2)].mv), (yyvsp[(2) - (2)].mvlist)); }
    break;

  case 100:
#line 425 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, (yyvsp[(3) - (3)].ntype)); }
    break;

  case 101:
#line 427 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, (yyvsp[(3) - (6)].ntype), (yyvsp[(5) - (6)].mvlist)); }
    break;

  case 102:
#line 431 "xi-grammar.y"
    { (yyval.typelist) = 0; }
    break;

  case 103:
#line 433 "xi-grammar.y"
    { (yyval.typelist) = (yyvsp[(2) - (2)].typelist); }
    break;

  case 104:
#line 437 "xi-grammar.y"
    { (yyval.typelist) = new TypeList((yyvsp[(1) - (1)].ntype)); }
    break;

  case 105:
#line 439 "xi-grammar.y"
    { (yyval.typelist) = new TypeList((yyvsp[(1) - (3)].ntype), (yyvsp[(3) - (3)].typelist)); }
    break;

  case 106:
#line 443 "xi-grammar.y"
    { (yyval.chare) = new Chare(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 107:
#line 445 "xi-grammar.y"
    { (yyval.chare) = new MainChare(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 108:
#line 449 "xi-grammar.y"
    { (yyval.chare) = new Group(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 109:
#line 453 "xi-grammar.y"
    { (yyval.chare) = new NodeGroup(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 110:
#line 457 "xi-grammar.y"
    {/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",(yyvsp[(2) - (4)].strval));
			(yyval.ntype) = new NamedType(buf); 
		}
    break;

  case 111:
#line 463 "xi-grammar.y"
    { (yyval.ntype) = new NamedType((yyvsp[(2) - (3)].strval)); }
    break;

  case 112:
#line 467 "xi-grammar.y"
    {  (yyval.chare) = new Array(lineno, (yyvsp[(2) - (6)].cattr), (yyvsp[(3) - (6)].ntype), (yyvsp[(4) - (6)].ntype), (yyvsp[(5) - (6)].typelist), (yyvsp[(6) - (6)].mbrlist)); }
    break;

  case 113:
#line 469 "xi-grammar.y"
    {  (yyval.chare) = new Array(lineno, (yyvsp[(3) - (6)].cattr), (yyvsp[(2) - (6)].ntype), (yyvsp[(4) - (6)].ntype), (yyvsp[(5) - (6)].typelist), (yyvsp[(6) - (6)].mbrlist)); }
    break;

  case 114:
#line 473 "xi-grammar.y"
    { (yyval.chare) = new Chare(lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist));}
    break;

  case 115:
#line 475 "xi-grammar.y"
    { (yyval.chare) = new MainChare(lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 116:
#line 479 "xi-grammar.y"
    { (yyval.chare) = new Group(lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 117:
#line 483 "xi-grammar.y"
    { (yyval.chare) = new NodeGroup( lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 118:
#line 487 "xi-grammar.y"
    { (yyval.chare) = new Array( lineno, 0, (yyvsp[(2) - (5)].ntype), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 119:
#line 491 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[(3) - (4)].strval))); }
    break;

  case 120:
#line 493 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[(3) - (7)].strval)), (yyvsp[(5) - (7)].mvlist)); }
    break;

  case 121:
#line 497 "xi-grammar.y"
    { (yyval.type) = 0; }
    break;

  case 122:
#line 499 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(2) - (2)].type); }
    break;

  case 123:
#line 503 "xi-grammar.y"
    { (yyval.strval) = 0; }
    break;

  case 124:
#line 505 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(2) - (2)].strval); }
    break;

  case 125:
#line 507 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(2) - (2)].strval); }
    break;

  case 126:
#line 511 "xi-grammar.y"
    { (yyval.tvar) = new TType(new NamedType((yyvsp[(2) - (3)].strval)), (yyvsp[(3) - (3)].type)); }
    break;

  case 127:
#line 513 "xi-grammar.y"
    { (yyval.tvar) = new TFunc((yyvsp[(1) - (2)].ftype), (yyvsp[(2) - (2)].strval)); }
    break;

  case 128:
#line 515 "xi-grammar.y"
    { (yyval.tvar) = new TName((yyvsp[(1) - (3)].type), (yyvsp[(2) - (3)].strval), (yyvsp[(3) - (3)].strval)); }
    break;

  case 129:
#line 519 "xi-grammar.y"
    { (yyval.tvarlist) = new TVarList((yyvsp[(1) - (1)].tvar)); }
    break;

  case 130:
#line 521 "xi-grammar.y"
    { (yyval.tvarlist) = new TVarList((yyvsp[(1) - (3)].tvar), (yyvsp[(3) - (3)].tvarlist)); }
    break;

  case 131:
#line 525 "xi-grammar.y"
    { (yyval.tvarlist) = (yyvsp[(3) - (4)].tvarlist); }
    break;

  case 132:
#line 529 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 133:
#line 531 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 134:
#line 533 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 135:
#line 535 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 136:
#line 537 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].message)); (yyvsp[(2) - (2)].message)->setTemplate((yyval.templat)); }
    break;

  case 137:
#line 541 "xi-grammar.y"
    { (yyval.mbrlist) = 0; }
    break;

  case 138:
#line 543 "xi-grammar.y"
    { (yyval.mbrlist) = (yyvsp[(2) - (4)].mbrlist); }
    break;

  case 139:
#line 547 "xi-grammar.y"
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

  case 140:
#line 566 "xi-grammar.y"
    { (yyval.mbrlist) = new MemberList((yyvsp[(1) - (2)].member), (yyvsp[(2) - (2)].mbrlist)); }
    break;

  case 141:
#line 570 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (2)].readonly); }
    break;

  case 142:
#line 572 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (2)].readonly); }
    break;

  case 144:
#line 575 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (2)].member); }
    break;

  case 145:
#line 577 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(2) - (3)].pupable); }
    break;

  case 146:
#line 579 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(2) - (3)].includeFile); }
    break;

  case 147:
#line 581 "xi-grammar.y"
    { (yyval.member) = new ClassDeclaration(lineno,(yyvsp[(2) - (3)].strval)); }
    break;

  case 148:
#line 585 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 1); }
    break;

  case 149:
#line 587 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 1); }
    break;

  case 150:
#line 589 "xi-grammar.y"
    { printf("Warning: deprecated use of initcall. Use initnode or initproc instead.\n"); 
		  (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 1); }
    break;

  case 151:
#line 592 "xi-grammar.y"
    { printf("Warning: deprecated use of initcall. Use initnode or initproc instead.\n");
		  (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 1); }
    break;

  case 152:
#line 597 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 0); }
    break;

  case 153:
#line 599 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 0); }
    break;

  case 154:
#line 603 "xi-grammar.y"
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[(1) - (1)].ntype),0); }
    break;

  case 155:
#line 605 "xi-grammar.y"
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[(1) - (3)].ntype),(yyvsp[(3) - (3)].pupable)); }
    break;

  case 156:
#line 608 "xi-grammar.y"
    { (yyval.includeFile) = new IncludeFile(lineno,(yyvsp[(1) - (1)].strval)); }
    break;

  case 157:
#line 612 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (2)].entry); }
    break;

  case 158:
#line 614 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].member); }
    break;

  case 159:
#line 618 "xi-grammar.y"
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

  case 160:
#line 629 "xi-grammar.y"
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

  case 161:
#line 642 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("void"); }
    break;

  case 162:
#line 644 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 163:
#line 648 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 164:
#line 650 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(2) - (3)].intval); }
    break;

  case 165:
#line 654 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (1)].intval); }
    break;

  case 166:
#line 656 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (3)].intval) | (yyvsp[(3) - (3)].intval); }
    break;

  case 167:
#line 660 "xi-grammar.y"
    { (yyval.intval) = STHREADED; }
    break;

  case 168:
#line 662 "xi-grammar.y"
    { (yyval.intval) = SSYNC; }
    break;

  case 169:
#line 664 "xi-grammar.y"
    { (yyval.intval) = SIGET; }
    break;

  case 170:
#line 666 "xi-grammar.y"
    { (yyval.intval) = SLOCKED; }
    break;

  case 171:
#line 668 "xi-grammar.y"
    { (yyval.intval) = SCREATEHERE; }
    break;

  case 172:
#line 670 "xi-grammar.y"
    { (yyval.intval) = SCREATEHOME; }
    break;

  case 173:
#line 672 "xi-grammar.y"
    { (yyval.intval) = SNOKEEP; }
    break;

  case 174:
#line 674 "xi-grammar.y"
    { (yyval.intval) = SNOTRACE; }
    break;

  case 175:
#line 676 "xi-grammar.y"
    { (yyval.intval) = SIMMEDIATE; }
    break;

  case 176:
#line 678 "xi-grammar.y"
    { (yyval.intval) = SSKIPSCHED; }
    break;

  case 177:
#line 680 "xi-grammar.y"
    { (yyval.intval) = SINLINE; }
    break;

  case 178:
#line 682 "xi-grammar.y"
    { (yyval.intval) = SLOCAL; }
    break;

  case 179:
#line 684 "xi-grammar.y"
    { (yyval.intval) = SPYTHON; }
    break;

  case 180:
#line 688 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 181:
#line 690 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 182:
#line 692 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 183:
#line 696 "xi-grammar.y"
    { (yyval.strval) = ""; }
    break;

  case 184:
#line 698 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 185:
#line 700 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[(1) - (3)].strval))+strlen((yyvsp[(3) - (3)].strval))+3];
			sprintf(tmp,"%s, %s", (yyvsp[(1) - (3)].strval), (yyvsp[(3) - (3)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 186:
#line 708 "xi-grammar.y"
    { (yyval.strval) = ""; }
    break;

  case 187:
#line 710 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 188:
#line 712 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s[%s]%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 189:
#line 718 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s{%s}%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 190:
#line 724 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s(%s)%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 191:
#line 730 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(2) - (4)].strval))+strlen((yyvsp[(4) - (4)].strval))+3];
			sprintf(tmp,"(%s)%s", (yyvsp[(2) - (4)].strval), (yyvsp[(4) - (4)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 192:
#line 738 "xi-grammar.y"
    {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			(yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].type),(yyvsp[(2) - (3)].strval));
		}
    break;

  case 193:
#line 745 "xi-grammar.y"
    { 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			(yyval.intval) = 0;
		}
    break;

  case 194:
#line 753 "xi-grammar.y"
    { 
			in_braces=0;
			(yyval.intval) = 0;
		}
    break;

  case 195:
#line 760 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (1)].type));}
    break;

  case 196:
#line 762 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (2)].type),(yyvsp[(2) - (2)].strval));}
    break;

  case 197:
#line 764 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (4)].type),(yyvsp[(2) - (4)].strval),0,(yyvsp[(4) - (4)].val));}
    break;

  case 198:
#line 766 "xi-grammar.y"
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].pname)->getType(), (yyvsp[(1) - (3)].pname)->getName() ,(yyvsp[(2) - (3)].strval));
		}
    break;

  case 199:
#line 773 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (1)].pname)); }
    break;

  case 200:
#line 775 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (3)].pname),(yyvsp[(3) - (3)].plist)); }
    break;

  case 201:
#line 779 "xi-grammar.y"
    { (yyval.plist) = (yyvsp[(2) - (3)].plist); }
    break;

  case 202:
#line 781 "xi-grammar.y"
    { (yyval.plist) = 0; }
    break;

  case 203:
#line 785 "xi-grammar.y"
    { (yyval.val) = 0; }
    break;

  case 204:
#line 787 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(3) - (3)].strval)); }
    break;

  case 205:
#line 791 "xi-grammar.y"
    { (yyval.sc) = 0; }
    break;

  case 206:
#line 793 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSDAGENTRY, (yyvsp[(1) - (1)].sc)); }
    break;

  case 207:
#line 795 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSDAGENTRY, (yyvsp[(2) - (3)].sc)); }
    break;

  case 208:
#line 799 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSLIST, (yyvsp[(1) - (1)].sc)); }
    break;

  case 209:
#line 801 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSLIST, (yyvsp[(1) - (2)].sc), (yyvsp[(2) - (2)].sc));  }
    break;

  case 210:
#line 805 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SOLIST, (yyvsp[(1) - (1)].sc)); }
    break;

  case 211:
#line 807 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SOLIST, (yyvsp[(1) - (2)].sc), (yyvsp[(2) - (2)].sc)); }
    break;

  case 212:
#line 811 "xi-grammar.y"
    { (yyval.sc) = 0; }
    break;

  case 213:
#line 813 "xi-grammar.y"
    { (yyval.sc) = (yyvsp[(3) - (4)].sc); }
    break;

  case 214:
#line 817 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SPUBLISHES, new SdagConstruct(SIDENT, (yyvsp[(1) - (1)].strval))); }
    break;

  case 215:
#line 819 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SPUBLISHES, new SdagConstruct(SIDENT, (yyvsp[(1) - (3)].strval)), (yyvsp[(3) - (3)].sc));  }
    break;

  case 216:
#line 823 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 217:
#line 825 "xi-grammar.y"
    { (yyval.strval) = 0; }
    break;

  case 218:
#line 829 "xi-grammar.y"
    { RemoveSdagComments((yyvsp[(4) - (6)].strval));
		   (yyval.sc) = new SdagConstruct(SATOMIC, new XStr((yyvsp[(4) - (6)].strval)), (yyvsp[(6) - (6)].sc), 0,0,0,0, 0 ); 
		   if ((yyvsp[(2) - (6)].strval)) { (yyvsp[(2) - (6)].strval)[strlen((yyvsp[(2) - (6)].strval))-1]=0; (yyval.sc)->traceName = new XStr((yyvsp[(2) - (6)].strval)+1); }
		 }
    break;

  case 219:
#line 834 "xi-grammar.y"
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

  case 220:
#line 848 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SWHEN, 0, 0,0,0,0,0,(yyvsp[(2) - (4)].entrylist)); }
    break;

  case 221:
#line 850 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SWHEN, 0, 0, 0,0,0, (yyvsp[(3) - (3)].sc), (yyvsp[(2) - (3)].entrylist)); }
    break;

  case 222:
#line 852 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SWHEN, 0, 0, 0,0,0, (yyvsp[(4) - (5)].sc), (yyvsp[(2) - (5)].entrylist)); }
    break;

  case 223:
#line 854 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SOVERLAP,0, 0,0,0,0,(yyvsp[(3) - (4)].sc), 0); }
    break;

  case 224:
#line 856 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFOR, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (11)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(5) - (11)].strval)),
		             new SdagConstruct(SINT_EXPR, (yyvsp[(7) - (11)].strval)), 0, (yyvsp[(10) - (11)].sc), 0); }
    break;

  case 225:
#line 859 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFOR, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (9)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(5) - (9)].strval)), 
		         new SdagConstruct(SINT_EXPR, (yyvsp[(7) - (9)].strval)), 0, (yyvsp[(9) - (9)].sc), 0); }
    break;

  case 226:
#line 862 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFORALL, 0, new SdagConstruct(SIDENT, (yyvsp[(3) - (12)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(6) - (12)].strval)), 
		             new SdagConstruct(SINT_EXPR, (yyvsp[(8) - (12)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(10) - (12)].strval)), (yyvsp[(12) - (12)].sc), 0); }
    break;

  case 227:
#line 865 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFORALL, 0, new SdagConstruct(SIDENT, (yyvsp[(3) - (14)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(6) - (14)].strval)), 
		                 new SdagConstruct(SINT_EXPR, (yyvsp[(8) - (14)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(10) - (14)].strval)), (yyvsp[(13) - (14)].sc), 0); }
    break;

  case 228:
#line 868 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (6)].strval)), (yyvsp[(6) - (6)].sc),0,0,(yyvsp[(5) - (6)].sc),0); }
    break;

  case 229:
#line 870 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (8)].strval)), (yyvsp[(8) - (8)].sc),0,0,(yyvsp[(6) - (8)].sc),0); }
    break;

  case 230:
#line 872 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (5)].strval)), 0,0,0,(yyvsp[(5) - (5)].sc),0); }
    break;

  case 231:
#line 874 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SWHILE, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (7)].strval)), 0,0,0,(yyvsp[(6) - (7)].sc),0); }
    break;

  case 232:
#line 876 "xi-grammar.y"
    { (yyval.sc) = (yyvsp[(2) - (3)].sc); }
    break;

  case 233:
#line 878 "xi-grammar.y"
    { RemoveSdagComments((yyvsp[(2) - (3)].strval));
                   (yyval.sc) = new SdagConstruct(SATOMIC, new XStr((yyvsp[(2) - (3)].strval)), NULL, 0,0,0,0, 0 );
                 }
    break;

  case 234:
#line 884 "xi-grammar.y"
    { (yyval.sc) = 0; }
    break;

  case 235:
#line 886 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SELSE, 0,0,0,0,0, (yyvsp[(2) - (2)].sc),0); }
    break;

  case 236:
#line 888 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SELSE, 0,0,0,0,0, (yyvsp[(3) - (4)].sc),0); }
    break;

  case 237:
#line 891 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFORWARD, new SdagConstruct(SIDENT, (yyvsp[(1) - (1)].strval))); }
    break;

  case 238:
#line 893 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFORWARD, new SdagConstruct(SIDENT, (yyvsp[(1) - (3)].strval)), (yyvsp[(3) - (3)].sc));  }
    break;

  case 239:
#line 897 "xi-grammar.y"
    { in_int_expr = 0; (yyval.intval) = 0; }
    break;

  case 240:
#line 901 "xi-grammar.y"
    { in_int_expr = 1; (yyval.intval) = 0; }
    break;

  case 241:
#line 905 "xi-grammar.y"
    { 
		  if ((yyvsp[(2) - (2)].plist) != 0)
		     (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[(1) - (2)].strval), (yyvsp[(2) - (2)].plist), 0, 0, 0, 0); 
		  else
		     (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[(1) - (2)].strval), 
				new ParamList(new Parameter(0, new BuiltinType("void"))), 0, 0, 0, 0); 
		}
    break;

  case 242:
#line 913 "xi-grammar.y"
    { if ((yyvsp[(5) - (5)].plist) != 0)
		    (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[(1) - (5)].strval), (yyvsp[(5) - (5)].plist), 0, 0, (yyvsp[(3) - (5)].strval), 0); 
		  else
		    (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[(1) - (5)].strval), new ParamList(new Parameter(0, new BuiltinType("void"))), 0, 0, (yyvsp[(3) - (5)].strval), 0); 
		}
    break;

  case 243:
#line 921 "xi-grammar.y"
    { (yyval.entrylist) = new EntryList((yyvsp[(1) - (1)].entry)); }
    break;

  case 244:
#line 923 "xi-grammar.y"
    { (yyval.entrylist) = new EntryList((yyvsp[(1) - (3)].entry),(yyvsp[(3) - (3)].entrylist)); }
    break;

  case 245:
#line 927 "xi-grammar.y"
    { in_bracket=1; }
    break;

  case 246:
#line 930 "xi-grammar.y"
    { in_bracket=0; }
    break;

  case 247:
#line 934 "xi-grammar.y"
    { if (!macroDefined((yyvsp[(2) - (2)].strval), 1)) in_comment = 1; }
    break;

  case 248:
#line 938 "xi-grammar.y"
    { if (!macroDefined((yyvsp[(2) - (2)].strval), 0)) in_comment = 1; }
    break;


/* Line 1267 of yacc.c.  */
#line 3319 "y.tab.c"
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


#line 941 "xi-grammar.y"

void yyerror(const char *mesg)
{
  cout << cur_file<<":"<<lineno<<": Charmxi syntax error> " << mesg << endl;
  // return 0;
}

