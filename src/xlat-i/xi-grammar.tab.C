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
     UNSIGNED = 322
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
/* Line 187 of yacc.c.  */
#line 281 "y.tab.c"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 294 "y.tab.c"

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
#define YYLAST   577

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  82
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  103
/* YYNRULES -- Number of rules.  */
#define YYNRULES  255
/* YYNRULES -- Number of states.  */
#define YYNSTATES  506

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   322

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    78,     2,
      76,    77,    75,     2,    72,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    69,    68,
      73,    81,    74,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    79,     2,    80,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    70,     2,    71,     2,     2,     2,     2,
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
      65,    66,    67
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     5,     6,     9,    10,    12,    13,    15,
      17,    19,    24,    28,    32,    34,    39,    40,    43,    49,
      55,    60,    64,    67,    70,    74,    77,    80,    83,    86,
      89,    91,    93,    95,    97,    99,   101,   105,   106,   108,
     109,   113,   115,   117,   119,   121,   124,   127,   131,   135,
     138,   141,   144,   146,   148,   151,   153,   156,   159,   161,
     163,   166,   169,   172,   181,   183,   185,   187,   189,   192,
     195,   198,   200,   202,   204,   208,   209,   212,   217,   223,
     224,   226,   227,   231,   233,   237,   239,   241,   242,   246,
     248,   252,   253,   255,   257,   258,   262,   264,   268,   270,
     272,   273,   275,   276,   279,   285,   287,   290,   294,   301,
     302,   305,   307,   311,   317,   323,   329,   335,   340,   344,
     351,   358,   364,   370,   376,   382,   388,   393,   401,   402,
     405,   406,   409,   412,   416,   419,   423,   425,   429,   434,
     437,   440,   443,   446,   449,   451,   456,   457,   460,   463,
     466,   469,   472,   476,   480,   484,   488,   495,   499,   506,
     510,   517,   519,   523,   525,   528,   530,   538,   544,   546,
     548,   549,   553,   555,   559,   561,   563,   565,   567,   569,
     571,   573,   575,   577,   579,   581,   583,   586,   588,   590,
     592,   593,   595,   599,   600,   602,   608,   614,   620,   625,
     629,   631,   633,   635,   639,   644,   648,   650,   654,   658,
     661,   662,   666,   667,   669,   673,   675,   678,   680,   683,
     684,   689,   691,   695,   697,   698,   705,   714,   719,   723,
     729,   734,   746,   756,   769,   784,   791,   800,   806,   814,
     818,   822,   823,   826,   831,   833,   837,   839,   841,   844,
     850,   852,   856,   858,   860,   863
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
      83,     0,    -1,    84,    -1,    -1,    89,    84,    -1,    -1,
       5,    -1,    -1,    68,    -1,    55,    -1,    55,    -1,    88,
      69,    69,    55,    -1,     3,    87,    90,    -1,     4,    87,
      90,    -1,    68,    -1,    70,    91,    71,    86,    -1,    -1,
      92,    91,    -1,    85,    70,    91,    71,    86,    -1,    53,
      87,    70,    91,    71,    -1,    54,    53,    88,    68,    -1,
      54,    88,    68,    -1,    85,    89,    -1,    85,   147,    -1,
      85,   126,    68,    -1,    85,   129,    -1,    85,   130,    -1,
      85,   131,    -1,    85,   133,    -1,    85,   144,    -1,   183,
      -1,   184,    -1,   105,    -1,    56,    -1,    57,    -1,    93,
      -1,    93,    72,    94,    -1,    -1,    94,    -1,    -1,    73,
      95,    74,    -1,    61,    -1,    62,    -1,    63,    -1,    64,
      -1,    67,    61,    -1,    67,    62,    -1,    67,    62,    61,
      -1,    67,    62,    62,    -1,    67,    63,    -1,    67,    64,
      -1,    62,    62,    -1,    65,    -1,    66,    -1,    62,    66,
      -1,    35,    -1,    87,    96,    -1,    88,    96,    -1,    97,
      -1,    99,    -1,   100,    75,    -1,   101,    75,    -1,   102,
      75,    -1,   104,    76,    75,    87,    77,    76,   165,    77,
      -1,   100,    -1,   101,    -1,   102,    -1,   103,    -1,    36,
     104,    -1,   104,    36,    -1,   104,    78,    -1,   104,    -1,
      56,    -1,    88,    -1,    79,   106,    80,    -1,    -1,   107,
     108,    -1,     6,   105,    88,   108,    -1,     6,    16,   100,
      75,    87,    -1,    -1,    35,    -1,    -1,    79,   113,    80,
      -1,   114,    -1,   114,    72,   113,    -1,    37,    -1,    38,
      -1,    -1,    79,   116,    80,    -1,   121,    -1,   121,    72,
     116,    -1,    -1,    57,    -1,    51,    -1,    -1,    79,   120,
      80,    -1,   118,    -1,   118,    72,   120,    -1,    30,    -1,
      51,    -1,    -1,    17,    -1,    -1,    79,    80,    -1,   122,
     105,    87,   123,    68,    -1,   124,    -1,   124,   125,    -1,
      16,   112,    98,    -1,    16,   112,    98,    70,   125,    71,
      -1,    -1,    69,   128,    -1,    98,    -1,    98,    72,   128,
      -1,    11,   115,    98,   127,   145,    -1,    12,   115,    98,
     127,   145,    -1,    13,   115,    98,   127,   145,    -1,    14,
     115,    98,   127,   145,    -1,    79,    56,    87,    80,    -1,
      79,    87,    80,    -1,    15,   119,   132,    98,   127,   145,
      -1,    15,   132,   119,    98,   127,   145,    -1,    11,   115,
      87,   127,   145,    -1,    12,   115,    87,   127,   145,    -1,
      13,   115,    87,   127,   145,    -1,    14,   115,    87,   127,
     145,    -1,    15,   132,    87,   127,   145,    -1,    16,   112,
      87,    68,    -1,    16,   112,    87,    70,   125,    71,    68,
      -1,    -1,    81,   105,    -1,    -1,    81,    56,    -1,    81,
      57,    -1,    18,    87,   139,    -1,   103,   140,    -1,   105,
      87,   140,    -1,   141,    -1,   141,    72,   142,    -1,    22,
      73,   142,    74,    -1,   143,   134,    -1,   143,   135,    -1,
     143,   136,    -1,   143,   137,    -1,   143,   138,    -1,    68,
      -1,    70,   146,    71,    86,    -1,    -1,   152,   146,    -1,
     109,    68,    -1,   110,    68,    -1,   149,    68,    -1,   148,
      68,    -1,    10,   150,    68,    -1,    19,   151,    68,    -1,
      18,    87,    68,    -1,     8,   111,    88,    -1,     8,   111,
      88,    76,   111,    77,    -1,     7,   111,    88,    -1,     7,
     111,    88,    76,   111,    77,    -1,     9,   111,    88,    -1,
       9,   111,    88,    76,   111,    77,    -1,    99,    -1,    99,
      72,   150,    -1,    57,    -1,   153,    68,    -1,   147,    -1,
      39,   155,   154,    87,   166,   167,   168,    -1,    39,   155,
      87,   166,   168,    -1,    35,    -1,   101,    -1,    -1,    79,
     156,    80,    -1,   157,    -1,   157,    72,   156,    -1,    21,
      -1,    23,    -1,    24,    -1,    25,    -1,    31,    -1,    32,
      -1,    33,    -1,    34,    -1,    26,    -1,    27,    -1,    28,
      -1,    52,    -1,    51,   117,    -1,    57,    -1,    56,    -1,
      88,    -1,    -1,    58,    -1,    58,    72,   159,    -1,    -1,
      58,    -1,    58,    79,   160,    80,   160,    -1,    58,    70,
     160,    71,   160,    -1,    58,    76,   159,    77,   160,    -1,
      76,   160,    77,   160,    -1,   105,    87,    79,    -1,    70,
      -1,    71,    -1,   105,    -1,   105,    87,   122,    -1,   105,
      87,    81,   158,    -1,   161,   160,    80,    -1,   164,    -1,
     164,    72,   165,    -1,    76,   165,    77,    -1,    76,    77,
      -1,    -1,    20,    81,    56,    -1,    -1,   174,    -1,    70,
     169,    71,    -1,   174,    -1,   174,   169,    -1,   174,    -1,
     174,   169,    -1,    -1,    50,    76,   172,    77,    -1,    55,
      -1,    55,    72,   172,    -1,    57,    -1,    -1,    45,   173,
     162,   160,   163,   171,    -1,    49,    76,    55,   166,    77,
     162,   160,    71,    -1,    43,   180,    70,    71,    -1,    43,
     180,   174,    -1,    43,   180,    70,   169,    71,    -1,    44,
      70,   170,    71,    -1,    40,   178,   160,    68,   160,    68,
     160,   177,    70,   169,    71,    -1,    40,   178,   160,    68,
     160,    68,   160,   177,   174,    -1,    41,    79,    55,    80,
     178,   160,    69,   160,    72,   160,   177,   174,    -1,    41,
      79,    55,    80,   178,   160,    69,   160,    72,   160,   177,
      70,   169,    71,    -1,    47,   178,   160,   177,   174,   175,
      -1,    47,   178,   160,   177,    70,   169,    71,   175,    -1,
      42,   178,   160,   177,   174,    -1,    42,   178,   160,   177,
      70,   169,    71,    -1,    46,   176,    68,    -1,   162,   160,
     163,    -1,    -1,    48,   174,    -1,    48,    70,   169,    71,
      -1,    55,    -1,    55,    72,   176,    -1,    77,    -1,    76,
      -1,    55,   166,    -1,    55,   181,   160,   182,   166,    -1,
     179,    -1,   179,    72,   180,    -1,    79,    -1,    80,    -1,
      59,    87,    -1,    60,    87,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   137,   137,   142,   145,   150,   151,   156,   157,   161,
     165,   167,   175,   179,   186,   188,   193,   194,   198,   200,
     202,   204,   206,   208,   210,   212,   214,   216,   218,   220,
     222,   224,   228,   230,   232,   236,   238,   243,   244,   249,
     250,   254,   256,   258,   260,   262,   264,   266,   268,   270,
     272,   274,   276,   278,   280,   282,   286,   287,   289,   291,
     295,   299,   301,   305,   309,   311,   313,   315,   318,   320,
     324,   326,   330,   332,   336,   341,   342,   346,   350,   355,
     356,   361,   362,   372,   374,   378,   380,   385,   386,   390,
     392,   397,   398,   402,   407,   408,   412,   414,   418,   420,
     425,   426,   430,   431,   434,   438,   440,   444,   446,   451,
     452,   456,   458,   462,   464,   468,   472,   476,   482,   486,
     488,   492,   494,   498,   502,   506,   510,   512,   517,   518,
     523,   524,   526,   530,   532,   534,   538,   540,   544,   548,
     550,   552,   554,   556,   560,   562,   567,   585,   589,   591,
     593,   594,   596,   598,   600,   604,   606,   608,   611,   616,
     618,   622,   624,   627,   631,   633,   637,   648,   661,   663,
     668,   669,   673,   675,   679,   681,   683,   685,   687,   689,
     691,   693,   695,   697,   699,   701,   703,   707,   709,   711,
     716,   717,   719,   728,   729,   731,   737,   743,   749,   757,
     764,   772,   779,   781,   783,   785,   792,   794,   798,   800,
     805,   806,   811,   812,   814,   818,   820,   824,   826,   831,
     832,   836,   838,   842,   845,   848,   853,   867,   869,   871,
     873,   875,   878,   881,   884,   887,   889,   891,   893,   895,
     897,   904,   905,   907,   910,   912,   916,   920,   924,   932,
     940,   942,   946,   949,   953,   957
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
  "FLOAT", "DOUBLE", "UNSIGNED", "';'", "':'", "'{'", "'}'", "','", "'<'",
  "'>'", "'*'", "'('", "')'", "'&'", "'['", "']'", "'='", "$accept",
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
     315,   316,   317,   318,   319,   320,   321,   322,    59,    58,
     123,   125,    44,    60,    62,    42,    40,    41,    38,    91,
      93,    61
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    82,    83,    84,    84,    85,    85,    86,    86,    87,
      88,    88,    89,    89,    90,    90,    91,    91,    92,    92,
      92,    92,    92,    92,    92,    92,    92,    92,    92,    92,
      92,    92,    93,    93,    93,    94,    94,    95,    95,    96,
      96,    97,    97,    97,    97,    97,    97,    97,    97,    97,
      97,    97,    97,    97,    97,    97,    98,    99,   100,   100,
     101,   102,   102,   103,   104,   104,   104,   104,   104,   104,
     105,   105,   106,   106,   107,   108,   108,   109,   110,   111,
     111,   112,   112,   113,   113,   114,   114,   115,   115,   116,
     116,   117,   117,   118,   119,   119,   120,   120,   121,   121,
     122,   122,   123,   123,   124,   125,   125,   126,   126,   127,
     127,   128,   128,   129,   129,   130,   131,   132,   132,   133,
     133,   134,   134,   135,   136,   137,   138,   138,   139,   139,
     140,   140,   140,   141,   141,   141,   142,   142,   143,   144,
     144,   144,   144,   144,   145,   145,   146,   146,   147,   147,
     147,   147,   147,   147,   147,   148,   148,   148,   148,   149,
     149,   150,   150,   151,   152,   152,   153,   153,   154,   154,
     155,   155,   156,   156,   157,   157,   157,   157,   157,   157,
     157,   157,   157,   157,   157,   157,   157,   158,   158,   158,
     159,   159,   159,   160,   160,   160,   160,   160,   160,   161,
     162,   163,   164,   164,   164,   164,   165,   165,   166,   166,
     167,   167,   168,   168,   168,   169,   169,   170,   170,   171,
     171,   172,   172,   173,   173,   174,   174,   174,   174,   174,
     174,   174,   174,   174,   174,   174,   174,   174,   174,   174,
     174,   175,   175,   175,   176,   176,   177,   178,   179,   179,
     180,   180,   181,   182,   183,   184
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     0,     2,     0,     1,     0,     1,     1,
       1,     4,     3,     3,     1,     4,     0,     2,     5,     5,
       4,     3,     2,     2,     3,     2,     2,     2,     2,     2,
       1,     1,     1,     1,     1,     1,     3,     0,     1,     0,
       3,     1,     1,     1,     1,     2,     2,     3,     3,     2,
       2,     2,     1,     1,     2,     1,     2,     2,     1,     1,
       2,     2,     2,     8,     1,     1,     1,     1,     2,     2,
       2,     1,     1,     1,     3,     0,     2,     4,     5,     0,
       1,     0,     3,     1,     3,     1,     1,     0,     3,     1,
       3,     0,     1,     1,     0,     3,     1,     3,     1,     1,
       0,     1,     0,     2,     5,     1,     2,     3,     6,     0,
       2,     1,     3,     5,     5,     5,     5,     4,     3,     6,
       6,     5,     5,     5,     5,     5,     4,     7,     0,     2,
       0,     2,     2,     3,     2,     3,     1,     3,     4,     2,
       2,     2,     2,     2,     1,     4,     0,     2,     2,     2,
       2,     2,     3,     3,     3,     3,     6,     3,     6,     3,
       6,     1,     3,     1,     2,     1,     7,     5,     1,     1,
       0,     3,     1,     3,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     2,     1,     1,     1,
       0,     1,     3,     0,     1,     5,     5,     5,     4,     3,
       1,     1,     1,     3,     4,     3,     1,     3,     3,     2,
       0,     3,     0,     1,     3,     1,     2,     1,     2,     0,
       4,     1,     3,     1,     0,     6,     8,     4,     3,     5,
       4,    11,     9,    12,    14,     6,     8,     5,     7,     3,
       3,     0,     2,     4,     1,     3,     1,     1,     2,     5,
       1,     3,     1,     1,     2,     2
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       3,     0,     0,     0,     2,     3,     9,     0,     0,     1,
       4,    14,     5,    12,    13,     6,     0,     0,     0,     0,
       0,     0,     5,    30,    31,     0,     0,    10,     0,   254,
     255,     0,    79,    79,    79,     0,    87,    87,    87,    87,
       0,    81,     0,     0,     0,     5,    22,     0,     0,     0,
      25,    26,    27,    28,     0,    29,    23,     0,     0,     7,
      17,     5,     0,    21,     0,     0,    55,     0,    41,    42,
      43,    44,    52,    53,     0,    39,    58,    59,    64,    65,
      66,    67,    71,     0,    80,     0,     0,     0,   161,     0,
       0,     0,     0,     0,     0,     0,     0,    94,     0,     0,
       0,   163,     0,     0,     0,   148,   149,    24,    87,    87,
      87,    87,     0,    81,   139,   140,   141,   142,   143,   151,
     150,     8,    15,     0,    20,     0,     0,    68,    51,    54,
      45,    46,    49,    50,    37,    57,    60,    61,    62,    69,
       0,    70,    75,   157,   155,   159,     0,   152,    98,    99,
       0,    89,    39,   109,   109,   109,   109,    93,     0,     0,
      96,     0,     0,     0,     0,     0,    85,    86,     0,    83,
     107,   154,   153,     0,    67,     0,   136,     0,     7,     0,
       0,     0,     0,     0,     0,    19,    11,     0,    47,    48,
      33,    34,    35,    38,     0,    32,     0,     0,    75,    77,
      79,    79,    79,   162,    88,     0,    56,     0,     0,     0,
       0,     0,     0,   118,     0,    95,   109,   109,    82,     0,
     100,   128,     0,   134,   130,     0,   138,    18,   109,   109,
     109,   109,   109,     0,    78,     0,    40,     0,    72,    73,
       0,    76,     0,     0,     0,    90,   111,   110,   144,   146,
     113,   114,   115,   116,   117,    97,     0,     0,    84,   101,
       0,   100,     0,     0,   133,   131,   132,   135,   137,     0,
       0,     0,     0,     0,   126,   100,    36,     0,    74,   158,
     156,   160,     0,   170,     0,   165,   146,     0,   119,   120,
       0,   106,   108,   129,   121,   122,   123,   124,   125,     0,
       0,   112,     0,     0,     7,   147,   164,   102,     0,   202,
     193,   206,     0,   174,   175,   176,   177,   182,   183,   184,
     178,   179,   180,   181,    91,   185,     0,   172,    55,    10,
       0,     0,   169,     0,   145,     0,     0,   127,   100,   194,
     193,     0,     0,    63,    92,   186,   171,     0,     0,   212,
       0,   103,   104,   199,     0,   203,   193,   190,   193,     0,
     205,   207,   173,   209,     0,     0,     0,     0,     0,     0,
     224,     0,     0,     0,   200,   193,   167,   213,   210,   188,
     187,   189,   204,     0,   191,     0,     0,   193,   208,   247,
     193,     0,   193,     0,   250,     0,     0,   223,     0,   244,
       0,   193,     0,   200,     0,   215,     0,     0,   212,   193,
     190,   193,   193,   198,     0,     0,     0,   252,   248,   193,
       0,   200,   228,     0,   217,   193,     0,   239,     0,     0,
     214,   216,   201,   240,     0,   166,   196,   192,   197,   195,
     193,     0,   246,     0,     0,   251,   227,     0,   230,   218,
       0,   245,     0,     0,   211,     0,   193,   200,   237,   253,
       0,   229,   219,   200,   241,     0,   193,     0,     0,   249,
       0,   225,     0,     0,   235,   193,     0,   193,   238,     0,
     241,   200,   242,     0,     0,     0,   221,     0,   236,     0,
     226,   200,   232,   193,     0,   220,   243,     0,     0,   222,
     231,     0,   200,   233,     0,   234
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     3,     4,    20,   122,   152,    75,     5,    13,    21,
      22,   192,   193,   194,   135,    76,   246,    77,    78,    79,
      80,    81,    82,   309,   240,   198,   199,    47,    48,    85,
      99,   168,   169,    91,   150,   345,   160,    96,   161,   151,
     260,   336,   261,   262,    49,   208,   247,    50,    51,    52,
      97,    53,   114,   115,   116,   117,   118,   264,   223,   176,
     177,    54,    55,   250,   284,   285,    57,    58,    89,   102,
     286,   287,   333,   303,   326,   327,   382,   385,   341,   310,
     375,   433,   311,   312,   349,   408,   376,   404,   423,   471,
     487,   398,   405,   474,   400,   443,   390,   394,   395,   419,
     460,    23,    24
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -400
static const yytype_int16 yypact[] =
{
     218,    77,    77,    27,  -400,   218,  -400,    91,    91,  -400,
    -400,  -400,    75,  -400,  -400,  -400,    77,   115,    77,    77,
     201,    80,    75,  -400,  -400,    70,   107,  -400,   156,  -400,
    -400,   211,   130,   130,   130,   107,    92,    92,    92,    92,
     100,   109,    77,   137,   124,    75,  -400,   135,   150,   158,
    -400,  -400,  -400,  -400,   285,  -400,  -400,   162,   177,   187,
    -400,    75,   160,  -400,   189,   310,  -400,   292,  -400,    59,
    -400,  -400,  -400,  -400,   138,    69,  -400,  -400,   181,   182,
     186,  -400,    22,   107,  -400,   107,   107,   107,   190,   196,
     -18,    77,    77,    77,    77,    93,   184,   191,   195,    77,
     197,  -400,   199,   224,   198,  -400,  -400,  -400,    92,    92,
      92,    92,   184,   109,  -400,  -400,  -400,  -400,  -400,  -400,
    -400,  -400,  -400,   209,  -400,   213,   206,    11,  -400,  -400,
    -400,   174,  -400,  -400,   277,  -400,  -400,  -400,  -400,  -400,
     207,  -400,   -20,    -4,     9,    55,   107,  -400,  -400,  -400,
     203,   212,   220,   233,   233,   233,   233,  -400,    77,   223,
     232,   226,   185,    77,   256,    77,  -400,  -400,   228,   237,
     240,  -400,  -400,    77,   -26,    77,   243,   242,   187,    77,
      77,    77,    77,    77,    77,  -400,  -400,    77,  -400,  -400,
    -400,  -400,   245,  -400,   244,  -400,    77,   188,   250,  -400,
     130,   130,   130,  -400,  -400,   -18,  -400,    77,   108,   108,
     108,   108,   239,  -400,   256,  -400,   233,   233,  -400,   195,
     313,   265,   192,  -400,   267,   224,  -400,  -400,   233,   233,
     233,   233,   233,   125,  -400,   277,  -400,   254,  -400,   280,
     270,  -400,   274,   275,   284,  -400,   290,  -400,  -400,   148,
    -400,  -400,  -400,  -400,  -400,  -400,   108,   108,  -400,  -400,
     292,    15,   293,   292,  -400,  -400,  -400,  -400,  -400,   108,
     108,   108,   108,   108,  -400,   313,  -400,   287,  -400,  -400,
    -400,  -400,    77,   288,   295,  -400,   148,   300,  -400,  -400,
      77,  -400,  -400,  -400,  -400,  -400,  -400,  -400,  -400,   298,
     292,  -400,   372,   325,   187,  -400,  -400,   291,   311,    77,
     -15,   306,   304,  -400,  -400,  -400,  -400,  -400,  -400,  -400,
    -400,  -400,  -400,  -400,   326,  -400,   302,   312,   330,   318,
     339,   181,  -400,    77,  -400,   321,   334,  -400,    58,    14,
     -15,   337,   292,  -400,  -400,  -400,  -400,   372,   259,   385,
     339,  -400,  -400,  -400,    -5,  -400,   -15,   360,   -15,   342,
    -400,  -400,  -400,  -400,   343,   345,   354,   345,   380,   352,
     379,   392,   345,   373,   399,   -15,  -400,  -400,   431,  -400,
    -400,   280,  -400,   381,   378,   376,   374,   -15,  -400,  -400,
     -15,   409,   -15,    74,   394,   416,   399,  -400,   397,   396,
     410,   -15,   425,  -400,   411,   399,   412,   400,   385,   -15,
     360,   -15,   -15,  -400,   417,   404,   418,  -400,  -400,   -15,
     380,   367,  -400,   426,   399,   -15,   392,  -400,   418,   339,
    -400,  -400,  -400,  -400,   442,  -400,  -400,  -400,  -400,  -400,
     -15,   345,  -400,   430,   419,  -400,  -400,   438,  -400,  -400,
     412,  -400,   447,   434,  -400,   444,   -15,   399,  -400,  -400,
     339,  -400,   463,   399,   466,   397,   -15,   446,   445,  -400,
     450,  -400,   457,   461,  -400,   -15,   418,   -15,  -400,   474,
     466,   399,  -400,   459,   478,   468,   470,   467,  -400,   472,
    -400,   399,  -400,   -15,   474,  -400,  -400,   475,   418,  -400,
    -400,   492,   399,  -400,   476,  -400
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -400,  -400,   540,  -400,  -171,     2,   -17,   529,   542,    82,
    -400,  -400,   316,  -400,   401,  -400,   -69,   -24,   -64,   249,
    -400,   -95,   487,   -29,  -400,  -400,   357,  -400,  -400,   -28,
     443,   338,  -400,    -2,   353,  -400,  -400,   462,   346,  -400,
     225,  -400,  -400,  -222,  -400,  -114,   279,  -400,  -400,  -400,
     -58,  -400,  -400,  -400,  -400,  -400,  -400,  -400,   340,  -400,
     341,  -400,  -400,   -19,   281,   545,  -400,  -400,   422,  -400,
    -400,  -400,  -400,  -400,   222,  -400,  -400,   161,  -330,  -400,
    -382,   120,  -400,  -229,  -337,  -400,   164,  -390,  -400,  -400,
      79,  -400,  -332,    94,   149,  -399,  -353,  -400,   157,  -400,
    -400,  -400,  -400
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -169
static const yytype_int16 yytable[] =
{
      28,   126,    83,     7,     8,    86,    87,   227,   174,    62,
     359,    88,   148,   378,   392,   431,   425,   377,    25,   401,
      29,    30,   153,   154,   155,   156,   383,     9,   386,   452,
     170,   447,   259,   149,   449,    92,    93,    94,   163,   291,
     209,   210,   211,   339,   100,   406,  -130,   139,  -130,    64,
      27,   379,   380,   299,   183,   222,   418,   413,   139,   197,
     414,   340,   416,   422,   424,    64,   142,   468,   143,   144,
     145,   428,   200,   472,   175,   259,   377,   484,    64,   436,
      15,   438,   439,   475,   356,   201,  -105,   140,   456,   444,
     357,   489,   453,   358,   216,   450,   217,   159,   140,   501,
     141,   497,   256,   257,    60,   195,   179,   180,   181,   182,
     455,   458,   504,   361,   269,   270,   271,   272,   273,   364,
     464,   128,    88,   469,    64,   129,   467,   104,    16,    17,
     174,   202,     6,   334,    18,    19,   476,   353,    64,   354,
      61,   482,   134,   123,   157,   483,   -16,   485,     6,   158,
     348,    59,   492,   417,    31,    32,    33,    34,    35,    11,
     212,    12,    27,   498,   159,    84,    42,    43,    26,   503,
      27,    90,   242,   243,   244,   221,   248,   224,   249,    95,
     239,   228,   229,   230,   231,   232,   233,   283,    98,   234,
     251,   252,   253,   274,   101,   275,   175,   103,   237,   130,
     131,   132,   133,   105,     1,     2,   195,    31,    32,    33,
      34,    35,    36,    37,    38,    39,    40,    41,   106,    42,
      43,     1,     2,    44,    63,    64,   107,    65,   124,    64,
     119,   290,   166,   167,   293,   188,   189,   288,   289,   331,
       6,   158,   173,    27,   238,   120,    66,    67,   265,   266,
     294,   295,   296,   297,   298,   121,   136,   137,   125,    66,
      67,   138,   146,   162,   147,   171,    27,   172,   186,   178,
     164,    45,    68,    69,    70,    71,    72,    73,    74,    27,
     185,   187,   196,   204,   205,    68,    69,    70,    71,    72,
      73,    74,   307,   134,    66,    67,   108,   109,   110,   111,
     112,   113,   207,   213,   214,   330,   215,   157,   218,   219,
     220,   338,    66,    67,    27,   225,   226,   235,   236,   254,
      68,    69,    70,    71,    72,    73,    74,    66,    67,   197,
     259,   277,    27,   190,   191,   350,   363,   381,    68,    69,
      70,    71,    72,    73,    74,    66,   263,    27,   222,    64,
     278,   279,   280,    68,    69,    70,    71,    72,    73,    74,
     328,   281,   282,   300,   292,    27,   304,   302,   306,   308,
     335,    68,    69,    70,    71,    72,    73,    74,   342,   337,
     329,   343,   346,   344,   347,  -168,    68,    69,    70,    71,
      72,    73,    74,   313,    -9,   314,   315,   316,   317,   318,
     319,   351,   352,   320,   321,   322,   323,   365,   366,   367,
     368,   369,   370,   371,   372,   348,   373,   360,   384,   387,
     388,   389,   396,   324,   325,   365,   366,   367,   368,   369,
     370,   371,   372,   391,   373,   393,   397,   403,   446,   365,
     366,   367,   368,   369,   370,   371,   372,   399,   373,   402,
     410,   407,   409,   411,   412,   374,   365,   366,   367,   368,
     369,   370,   371,   372,   415,   373,   420,   403,   426,   403,
     365,   366,   367,   368,   369,   370,   371,   372,   427,   373,
     429,   434,   430,   432,   441,   440,   421,   365,   366,   367,
     368,   369,   370,   371,   372,   442,   373,   448,   454,   459,
     457,   365,   366,   367,   368,   369,   370,   371,   372,   461,
     373,   465,   466,   470,   473,   477,   478,   463,   365,   366,
     367,   368,   369,   370,   371,   372,   479,   373,   480,   486,
     490,   481,   365,   366,   367,   368,   369,   370,   371,   372,
     493,   373,   494,   496,   495,    10,   500,   505,   491,    46,
      14,   276,   332,   206,   127,   241,   184,   258,   245,   165,
     255,   301,   502,   355,   267,    56,   268,   305,   203,   362,
     462,   437,   435,   499,   488,   451,     0,   445
};

static const yytype_int16 yycheck[] =
{
      17,    65,    31,     1,     2,    33,    34,   178,   103,    26,
     340,    35,    30,   350,   367,   405,   398,   349,    16,   372,
      18,    19,    91,    92,    93,    94,   356,     0,   358,   428,
      99,   421,    17,    51,   424,    37,    38,    39,    96,   261,
     154,   155,   156,    58,    42,   375,    72,    36,    74,    69,
      55,    56,    57,   275,   112,    81,   393,   387,    36,    79,
     390,    76,   392,   395,   396,    69,    83,   457,    85,    86,
      87,   401,    76,   463,   103,    17,   408,   476,    69,   409,
       5,   411,   412,   465,    70,    76,    71,    76,   441,   419,
      76,   481,   429,    79,   163,   425,   165,    95,    76,   498,
      78,   491,   216,   217,    22,   134,   108,   109,   110,   111,
     440,   443,   502,   342,   228,   229,   230,   231,   232,   348,
     452,    62,   146,   460,    69,    66,   456,    45,    53,    54,
     225,    76,    55,   304,    59,    60,   466,    79,    69,    81,
      70,   473,    73,    61,    51,   475,    71,   477,    55,    56,
      76,    71,   484,    79,     6,     7,     8,     9,    10,    68,
     158,    70,    55,   493,   162,    35,    18,    19,    53,   501,
      55,    79,   200,   201,   202,   173,    68,   175,    70,    79,
     197,   179,   180,   181,   182,   183,   184,    39,    79,   187,
     209,   210,   211,    68,    57,    70,   225,    73,   196,    61,
      62,    63,    64,    68,     3,     4,   235,     6,     7,     8,
       9,    10,    11,    12,    13,    14,    15,    16,    68,    18,
      19,     3,     4,    22,    68,    69,    68,    16,    68,    69,
      68,   260,    37,    38,   263,    61,    62,   256,   257,   303,
      55,    56,    18,    55,    56,    68,    35,    36,    56,    57,
     269,   270,   271,   272,   273,    68,    75,    75,    69,    35,
      36,    75,    72,    79,    68,    68,    55,    68,    55,    71,
      79,    70,    61,    62,    63,    64,    65,    66,    67,    55,
      71,    75,    75,    80,    72,    61,    62,    63,    64,    65,
      66,    67,   290,    73,    35,    36,    11,    12,    13,    14,
      15,    16,    69,    80,    72,   303,    80,    51,    80,    72,
      70,   309,    35,    36,    55,    72,    74,    72,    74,    80,
      61,    62,    63,    64,    65,    66,    67,    35,    36,    79,
      17,    77,    55,    56,    57,   333,    77,   354,    61,    62,
      63,    64,    65,    66,    67,    35,    81,    55,    81,    69,
      80,    77,    77,    61,    62,    63,    64,    65,    66,    67,
      35,    77,    72,    76,    71,    55,    71,    79,    68,    71,
      79,    61,    62,    63,    64,    65,    66,    67,    72,    68,
      55,    77,    80,    57,    72,    55,    61,    62,    63,    64,
      65,    66,    67,    21,    76,    23,    24,    25,    26,    27,
      28,    80,    68,    31,    32,    33,    34,    40,    41,    42,
      43,    44,    45,    46,    47,    76,    49,    80,    58,    77,
      77,    76,    70,    51,    52,    40,    41,    42,    43,    44,
      45,    46,    47,    79,    49,    55,    57,    70,    71,    40,
      41,    42,    43,    44,    45,    46,    47,    55,    49,    76,
      72,    20,    71,    77,    80,    70,    40,    41,    42,    43,
      44,    45,    46,    47,    55,    49,    72,    70,    72,    70,
      40,    41,    42,    43,    44,    45,    46,    47,    68,    49,
      55,    81,    71,    71,    80,    68,    70,    40,    41,    42,
      43,    44,    45,    46,    47,    77,    49,    71,    56,    80,
      70,    40,    41,    42,    43,    44,    45,    46,    47,    71,
      49,    77,    68,    50,    48,    69,    71,    70,    40,    41,
      42,    43,    44,    45,    46,    47,    76,    49,    71,    55,
      71,    70,    40,    41,    42,    43,    44,    45,    46,    47,
      72,    49,    72,    71,    77,     5,    71,    71,    70,    20,
       8,   235,   303,   152,    67,   198,   113,   219,   205,    97,
     214,   282,    70,   338,   224,    20,   225,   286,   146,   347,
     450,   410,   408,   494,   480,   426,    -1,   420
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     3,     4,    83,    84,    89,    55,    87,    87,     0,
      84,    68,    70,    90,    90,     5,    53,    54,    59,    60,
      85,    91,    92,   183,   184,    87,    53,    55,    88,    87,
      87,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    18,    19,    22,    70,    89,   109,   110,   126,
     129,   130,   131,   133,   143,   144,   147,   148,   149,    71,
      91,    70,    88,    68,    69,    16,    35,    36,    61,    62,
      63,    64,    65,    66,    67,    88,    97,    99,   100,   101,
     102,   103,   104,   105,    35,   111,   111,   111,    99,   150,
      79,   115,   115,   115,   115,    79,   119,   132,    79,   112,
      87,    57,   151,    73,    91,    68,    68,    68,    11,    12,
      13,    14,    15,    16,   134,   135,   136,   137,   138,    68,
      68,    68,    86,    91,    68,    69,   100,   104,    62,    66,
      61,    62,    63,    64,    73,    96,    75,    75,    75,    36,
      76,    78,    88,    88,    88,    88,    72,    68,    30,    51,
     116,   121,    87,    98,    98,    98,    98,    51,    56,    87,
     118,   120,    79,   132,    79,   119,    37,    38,   113,   114,
      98,    68,    68,    18,   103,   105,   141,   142,    71,   115,
     115,   115,   115,   132,   112,    71,    55,    75,    61,    62,
      56,    57,    93,    94,    95,   105,    75,    79,   107,   108,
      76,    76,    76,   150,    80,    72,    96,    69,   127,   127,
     127,   127,    87,    80,    72,    80,    98,    98,    80,    72,
      70,    87,    81,   140,    87,    72,    74,    86,    87,    87,
      87,    87,    87,    87,    87,    72,    74,    87,    56,    88,
     106,   108,   111,   111,   111,   116,    98,   128,    68,    70,
     145,   145,   145,   145,    80,   120,   127,   127,   113,    17,
     122,   124,   125,    81,   139,    56,    57,   140,   142,   127,
     127,   127,   127,   127,    68,    70,    94,    77,    80,    77,
      77,    77,    72,    39,   146,   147,   152,   153,   145,   145,
     105,   125,    71,   105,   145,   145,   145,   145,   145,   125,
      76,   128,    79,   155,    71,   146,    68,    87,    71,   105,
     161,   164,   165,    21,    23,    24,    25,    26,    27,    28,
      31,    32,    33,    34,    51,    52,   156,   157,    35,    55,
      87,   100,   101,   154,    86,    79,   123,    68,    87,    58,
      76,   160,    72,    77,    57,   117,    80,    72,    76,   166,
      87,    80,    68,    79,    81,   122,    70,    76,    79,   160,
      80,   165,   156,    77,   165,    40,    41,    42,    43,    44,
      45,    46,    47,    49,    70,   162,   168,   174,   166,    56,
      57,    88,   158,   160,    58,   159,   160,    77,    77,    76,
     178,    79,   178,    55,   179,   180,    70,    57,   173,    55,
     176,   178,    76,    70,   169,   174,   160,    20,   167,    71,
      72,    77,    80,   160,   160,    55,   160,    79,   166,   181,
      72,    70,   174,   170,   174,   162,    72,    68,   160,    55,
      71,   169,    71,   163,    81,   168,   160,   159,   160,   160,
      68,    80,    77,   177,   160,   180,    71,   169,    71,   169,
     160,   176,   177,   166,    56,   160,   178,    70,   174,    80,
     182,    71,   163,    70,   174,    77,    68,   160,   169,   166,
      50,   171,   169,    48,   175,   162,   160,    69,    71,    76,
      71,    70,   174,   160,   177,   160,    55,   172,   175,   169,
      71,    70,   174,    72,    72,    77,    71,   169,   160,   172,
      71,   177,    70,   174,   169,    71
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
#line 138 "xi-grammar.y"
    { (yyval.modlist) = (yyvsp[(1) - (1)].modlist); modlist = (yyvsp[(1) - (1)].modlist); }
    break;

  case 3:
#line 142 "xi-grammar.y"
    { 
		  (yyval.modlist) = 0; 
		}
    break;

  case 4:
#line 146 "xi-grammar.y"
    { (yyval.modlist) = new ModuleList(lineno, (yyvsp[(1) - (2)].module), (yyvsp[(2) - (2)].modlist)); }
    break;

  case 5:
#line 150 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 6:
#line 152 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 7:
#line 156 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 8:
#line 158 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 9:
#line 162 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 10:
#line 166 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 11:
#line 168 "xi-grammar.y"
    {
		  char *tmp = new char[strlen((yyvsp[(1) - (4)].strval))+strlen((yyvsp[(4) - (4)].strval))+3];
		  sprintf(tmp,"%s::%s", (yyvsp[(1) - (4)].strval), (yyvsp[(4) - (4)].strval));
		  (yyval.strval) = tmp;
		}
    break;

  case 12:
#line 176 "xi-grammar.y"
    { 
		    (yyval.module) = new Module(lineno, (yyvsp[(2) - (3)].strval), (yyvsp[(3) - (3)].conslist)); 
		}
    break;

  case 13:
#line 180 "xi-grammar.y"
    {  
		    (yyval.module) = new Module(lineno, (yyvsp[(2) - (3)].strval), (yyvsp[(3) - (3)].conslist)); 
		    (yyval.module)->setMain();
		}
    break;

  case 14:
#line 187 "xi-grammar.y"
    { (yyval.conslist) = 0; }
    break;

  case 15:
#line 189 "xi-grammar.y"
    { (yyval.conslist) = (yyvsp[(2) - (4)].conslist); }
    break;

  case 16:
#line 193 "xi-grammar.y"
    { (yyval.conslist) = 0; }
    break;

  case 17:
#line 195 "xi-grammar.y"
    { (yyval.conslist) = new ConstructList(lineno, (yyvsp[(1) - (2)].construct), (yyvsp[(2) - (2)].conslist)); }
    break;

  case 18:
#line 199 "xi-grammar.y"
    { if((yyvsp[(3) - (5)].conslist)) (yyvsp[(3) - (5)].conslist)->setExtern((yyvsp[(1) - (5)].intval)); (yyval.construct) = (yyvsp[(3) - (5)].conslist); }
    break;

  case 19:
#line 201 "xi-grammar.y"
    { (yyval.construct) = new Scope((yyvsp[(2) - (5)].strval), (yyvsp[(4) - (5)].conslist)); }
    break;

  case 20:
#line 203 "xi-grammar.y"
    { (yyval.construct) = new UsingScope((yyvsp[(3) - (4)].strval)); }
    break;

  case 21:
#line 205 "xi-grammar.y"
    { (yyval.construct) = NULL; }
    break;

  case 22:
#line 207 "xi-grammar.y"
    { (yyvsp[(2) - (2)].module)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].module); }
    break;

  case 23:
#line 209 "xi-grammar.y"
    { (yyvsp[(2) - (2)].member)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].member); }
    break;

  case 24:
#line 211 "xi-grammar.y"
    { (yyvsp[(2) - (3)].message)->setExtern((yyvsp[(1) - (3)].intval)); (yyval.construct) = (yyvsp[(2) - (3)].message); }
    break;

  case 25:
#line 213 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 26:
#line 215 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 27:
#line 217 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 28:
#line 219 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 29:
#line 221 "xi-grammar.y"
    { (yyvsp[(2) - (2)].templat)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].templat); }
    break;

  case 30:
#line 223 "xi-grammar.y"
    { (yyval.construct) = NULL; }
    break;

  case 31:
#line 225 "xi-grammar.y"
    { (yyval.construct) = NULL; }
    break;

  case 32:
#line 229 "xi-grammar.y"
    { (yyval.tparam) = new TParamType((yyvsp[(1) - (1)].type)); }
    break;

  case 33:
#line 231 "xi-grammar.y"
    { (yyval.tparam) = new TParamVal((yyvsp[(1) - (1)].strval)); }
    break;

  case 34:
#line 233 "xi-grammar.y"
    { (yyval.tparam) = new TParamVal((yyvsp[(1) - (1)].strval)); }
    break;

  case 35:
#line 237 "xi-grammar.y"
    { (yyval.tparlist) = new TParamList((yyvsp[(1) - (1)].tparam)); }
    break;

  case 36:
#line 239 "xi-grammar.y"
    { (yyval.tparlist) = new TParamList((yyvsp[(1) - (3)].tparam), (yyvsp[(3) - (3)].tparlist)); }
    break;

  case 37:
#line 243 "xi-grammar.y"
    { (yyval.tparlist) = 0; }
    break;

  case 38:
#line 245 "xi-grammar.y"
    { (yyval.tparlist) = (yyvsp[(1) - (1)].tparlist); }
    break;

  case 39:
#line 249 "xi-grammar.y"
    { (yyval.tparlist) = 0; }
    break;

  case 40:
#line 251 "xi-grammar.y"
    { (yyval.tparlist) = (yyvsp[(2) - (3)].tparlist); }
    break;

  case 41:
#line 255 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("int"); }
    break;

  case 42:
#line 257 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long"); }
    break;

  case 43:
#line 259 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("short"); }
    break;

  case 44:
#line 261 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("char"); }
    break;

  case 45:
#line 263 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned int"); }
    break;

  case 46:
#line 265 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long"); }
    break;

  case 47:
#line 267 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long"); }
    break;

  case 48:
#line 269 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long long"); }
    break;

  case 49:
#line 271 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned short"); }
    break;

  case 50:
#line 273 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned char"); }
    break;

  case 51:
#line 275 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long long"); }
    break;

  case 52:
#line 277 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("float"); }
    break;

  case 53:
#line 279 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("double"); }
    break;

  case 54:
#line 281 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long double"); }
    break;

  case 55:
#line 283 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("void"); }
    break;

  case 56:
#line 286 "xi-grammar.y"
    { (yyval.ntype) = new NamedType((yyvsp[(1) - (2)].strval),(yyvsp[(2) - (2)].tparlist)); }
    break;

  case 57:
#line 287 "xi-grammar.y"
    { (yyval.ntype) = new NamedType((yyvsp[(1) - (2)].strval),(yyvsp[(2) - (2)].tparlist)); }
    break;

  case 58:
#line 290 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 59:
#line 292 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ntype); }
    break;

  case 60:
#line 296 "xi-grammar.y"
    { (yyval.ptype) = new PtrType((yyvsp[(1) - (2)].type)); }
    break;

  case 61:
#line 300 "xi-grammar.y"
    { (yyvsp[(1) - (2)].ptype)->indirect(); (yyval.ptype) = (yyvsp[(1) - (2)].ptype); }
    break;

  case 62:
#line 302 "xi-grammar.y"
    { (yyvsp[(1) - (2)].ptype)->indirect(); (yyval.ptype) = (yyvsp[(1) - (2)].ptype); }
    break;

  case 63:
#line 306 "xi-grammar.y"
    { (yyval.ftype) = new FuncType((yyvsp[(1) - (8)].type), (yyvsp[(4) - (8)].strval), (yyvsp[(7) - (8)].plist)); }
    break;

  case 64:
#line 310 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 65:
#line 312 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 66:
#line 314 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 67:
#line 316 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ftype); }
    break;

  case 68:
#line 319 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(2) - (2)].type); }
    break;

  case 69:
#line 321 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (2)].type); }
    break;

  case 70:
#line 325 "xi-grammar.y"
    { (yyval.type) = new ReferenceType((yyvsp[(1) - (2)].type)); }
    break;

  case 71:
#line 327 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 72:
#line 331 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 73:
#line 333 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 74:
#line 337 "xi-grammar.y"
    { (yyval.val) = (yyvsp[(2) - (3)].val); }
    break;

  case 75:
#line 341 "xi-grammar.y"
    { (yyval.vallist) = 0; }
    break;

  case 76:
#line 343 "xi-grammar.y"
    { (yyval.vallist) = new ValueList((yyvsp[(1) - (2)].val), (yyvsp[(2) - (2)].vallist)); }
    break;

  case 77:
#line 347 "xi-grammar.y"
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[(2) - (4)].type), (yyvsp[(3) - (4)].strval), (yyvsp[(4) - (4)].vallist)); }
    break;

  case 78:
#line 351 "xi-grammar.y"
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[(3) - (5)].type), (yyvsp[(5) - (5)].strval), 0, 1); }
    break;

  case 79:
#line 355 "xi-grammar.y"
    { (yyval.intval) = 0;}
    break;

  case 80:
#line 357 "xi-grammar.y"
    { (yyval.intval) = 0;}
    break;

  case 81:
#line 361 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 82:
#line 363 "xi-grammar.y"
    { 
		  /*
		  printf("Warning: Message attributes are being phased out.\n");
		  printf("Warning: Please remove them from interface files.\n");
		  */
		  (yyval.intval) = (yyvsp[(2) - (3)].intval); 
		}
    break;

  case 83:
#line 373 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (1)].intval); }
    break;

  case 84:
#line 375 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (3)].intval) | (yyvsp[(3) - (3)].intval); }
    break;

  case 85:
#line 379 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 86:
#line 381 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 87:
#line 385 "xi-grammar.y"
    { (yyval.cattr) = 0; }
    break;

  case 88:
#line 387 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(2) - (3)].cattr); }
    break;

  case 89:
#line 391 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (1)].cattr); }
    break;

  case 90:
#line 393 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (3)].cattr) | (yyvsp[(3) - (3)].cattr); }
    break;

  case 91:
#line 397 "xi-grammar.y"
    { python_doc = NULL; (yyval.intval) = 0; }
    break;

  case 92:
#line 399 "xi-grammar.y"
    { python_doc = (yyvsp[(1) - (1)].strval); (yyval.intval) = 0; }
    break;

  case 93:
#line 403 "xi-grammar.y"
    { (yyval.cattr) = Chare::CPYTHON; }
    break;

  case 94:
#line 407 "xi-grammar.y"
    { (yyval.cattr) = 0; }
    break;

  case 95:
#line 409 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(2) - (3)].cattr); }
    break;

  case 96:
#line 413 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (1)].cattr); }
    break;

  case 97:
#line 415 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (3)].cattr) | (yyvsp[(3) - (3)].cattr); }
    break;

  case 98:
#line 419 "xi-grammar.y"
    { (yyval.cattr) = Chare::CMIGRATABLE; }
    break;

  case 99:
#line 421 "xi-grammar.y"
    { (yyval.cattr) = Chare::CPYTHON; }
    break;

  case 100:
#line 425 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 101:
#line 427 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 102:
#line 430 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 103:
#line 432 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 104:
#line 435 "xi-grammar.y"
    { (yyval.mv) = new MsgVar((yyvsp[(2) - (5)].type), (yyvsp[(3) - (5)].strval), (yyvsp[(1) - (5)].intval), (yyvsp[(4) - (5)].intval)); }
    break;

  case 105:
#line 439 "xi-grammar.y"
    { (yyval.mvlist) = new MsgVarList((yyvsp[(1) - (1)].mv)); }
    break;

  case 106:
#line 441 "xi-grammar.y"
    { (yyval.mvlist) = new MsgVarList((yyvsp[(1) - (2)].mv), (yyvsp[(2) - (2)].mvlist)); }
    break;

  case 107:
#line 445 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, (yyvsp[(3) - (3)].ntype)); }
    break;

  case 108:
#line 447 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, (yyvsp[(3) - (6)].ntype), (yyvsp[(5) - (6)].mvlist)); }
    break;

  case 109:
#line 451 "xi-grammar.y"
    { (yyval.typelist) = 0; }
    break;

  case 110:
#line 453 "xi-grammar.y"
    { (yyval.typelist) = (yyvsp[(2) - (2)].typelist); }
    break;

  case 111:
#line 457 "xi-grammar.y"
    { (yyval.typelist) = new TypeList((yyvsp[(1) - (1)].ntype)); }
    break;

  case 112:
#line 459 "xi-grammar.y"
    { (yyval.typelist) = new TypeList((yyvsp[(1) - (3)].ntype), (yyvsp[(3) - (3)].typelist)); }
    break;

  case 113:
#line 463 "xi-grammar.y"
    { (yyval.chare) = new Chare(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 114:
#line 465 "xi-grammar.y"
    { (yyval.chare) = new MainChare(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 115:
#line 469 "xi-grammar.y"
    { (yyval.chare) = new Group(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 116:
#line 473 "xi-grammar.y"
    { (yyval.chare) = new NodeGroup(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 117:
#line 477 "xi-grammar.y"
    {/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",(yyvsp[(2) - (4)].strval));
			(yyval.ntype) = new NamedType(buf); 
		}
    break;

  case 118:
#line 483 "xi-grammar.y"
    { (yyval.ntype) = new NamedType((yyvsp[(2) - (3)].strval)); }
    break;

  case 119:
#line 487 "xi-grammar.y"
    {  (yyval.chare) = new Array(lineno, (yyvsp[(2) - (6)].cattr), (yyvsp[(3) - (6)].ntype), (yyvsp[(4) - (6)].ntype), (yyvsp[(5) - (6)].typelist), (yyvsp[(6) - (6)].mbrlist)); }
    break;

  case 120:
#line 489 "xi-grammar.y"
    {  (yyval.chare) = new Array(lineno, (yyvsp[(3) - (6)].cattr), (yyvsp[(2) - (6)].ntype), (yyvsp[(4) - (6)].ntype), (yyvsp[(5) - (6)].typelist), (yyvsp[(6) - (6)].mbrlist)); }
    break;

  case 121:
#line 493 "xi-grammar.y"
    { (yyval.chare) = new Chare(lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist));}
    break;

  case 122:
#line 495 "xi-grammar.y"
    { (yyval.chare) = new MainChare(lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 123:
#line 499 "xi-grammar.y"
    { (yyval.chare) = new Group(lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 124:
#line 503 "xi-grammar.y"
    { (yyval.chare) = new NodeGroup( lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 125:
#line 507 "xi-grammar.y"
    { (yyval.chare) = new Array( lineno, 0, (yyvsp[(2) - (5)].ntype), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 126:
#line 511 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[(3) - (4)].strval))); }
    break;

  case 127:
#line 513 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[(3) - (7)].strval)), (yyvsp[(5) - (7)].mvlist)); }
    break;

  case 128:
#line 517 "xi-grammar.y"
    { (yyval.type) = 0; }
    break;

  case 129:
#line 519 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(2) - (2)].type); }
    break;

  case 130:
#line 523 "xi-grammar.y"
    { (yyval.strval) = 0; }
    break;

  case 131:
#line 525 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(2) - (2)].strval); }
    break;

  case 132:
#line 527 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(2) - (2)].strval); }
    break;

  case 133:
#line 531 "xi-grammar.y"
    { (yyval.tvar) = new TType(new NamedType((yyvsp[(2) - (3)].strval)), (yyvsp[(3) - (3)].type)); }
    break;

  case 134:
#line 533 "xi-grammar.y"
    { (yyval.tvar) = new TFunc((yyvsp[(1) - (2)].ftype), (yyvsp[(2) - (2)].strval)); }
    break;

  case 135:
#line 535 "xi-grammar.y"
    { (yyval.tvar) = new TName((yyvsp[(1) - (3)].type), (yyvsp[(2) - (3)].strval), (yyvsp[(3) - (3)].strval)); }
    break;

  case 136:
#line 539 "xi-grammar.y"
    { (yyval.tvarlist) = new TVarList((yyvsp[(1) - (1)].tvar)); }
    break;

  case 137:
#line 541 "xi-grammar.y"
    { (yyval.tvarlist) = new TVarList((yyvsp[(1) - (3)].tvar), (yyvsp[(3) - (3)].tvarlist)); }
    break;

  case 138:
#line 545 "xi-grammar.y"
    { (yyval.tvarlist) = (yyvsp[(3) - (4)].tvarlist); }
    break;

  case 139:
#line 549 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 140:
#line 551 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 141:
#line 553 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 142:
#line 555 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 143:
#line 557 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].message)); (yyvsp[(2) - (2)].message)->setTemplate((yyval.templat)); }
    break;

  case 144:
#line 561 "xi-grammar.y"
    { (yyval.mbrlist) = 0; }
    break;

  case 145:
#line 563 "xi-grammar.y"
    { (yyval.mbrlist) = (yyvsp[(2) - (4)].mbrlist); }
    break;

  case 146:
#line 567 "xi-grammar.y"
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

  case 147:
#line 586 "xi-grammar.y"
    { (yyval.mbrlist) = new MemberList((yyvsp[(1) - (2)].member), (yyvsp[(2) - (2)].mbrlist)); }
    break;

  case 148:
#line 590 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (2)].readonly); }
    break;

  case 149:
#line 592 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (2)].readonly); }
    break;

  case 151:
#line 595 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (2)].member); }
    break;

  case 152:
#line 597 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(2) - (3)].pupable); }
    break;

  case 153:
#line 599 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(2) - (3)].includeFile); }
    break;

  case 154:
#line 601 "xi-grammar.y"
    { (yyval.member) = new ClassDeclaration(lineno,(yyvsp[(2) - (3)].strval)); }
    break;

  case 155:
#line 605 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 1); }
    break;

  case 156:
#line 607 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 1); }
    break;

  case 157:
#line 609 "xi-grammar.y"
    { printf("Warning: deprecated use of initcall. Use initnode or initproc instead.\n"); 
		  (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 1); }
    break;

  case 158:
#line 612 "xi-grammar.y"
    { printf("Warning: deprecated use of initcall. Use initnode or initproc instead.\n");
		  (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 1); }
    break;

  case 159:
#line 617 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 0); }
    break;

  case 160:
#line 619 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 0); }
    break;

  case 161:
#line 623 "xi-grammar.y"
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[(1) - (1)].ntype),0); }
    break;

  case 162:
#line 625 "xi-grammar.y"
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[(1) - (3)].ntype),(yyvsp[(3) - (3)].pupable)); }
    break;

  case 163:
#line 628 "xi-grammar.y"
    { (yyval.includeFile) = new IncludeFile(lineno,(yyvsp[(1) - (1)].strval)); }
    break;

  case 164:
#line 632 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (2)].entry); }
    break;

  case 165:
#line 634 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].member); }
    break;

  case 166:
#line 638 "xi-grammar.y"
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

  case 167:
#line 649 "xi-grammar.y"
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

  case 168:
#line 662 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("void"); }
    break;

  case 169:
#line 664 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 170:
#line 668 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 171:
#line 670 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(2) - (3)].intval); }
    break;

  case 172:
#line 674 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (1)].intval); }
    break;

  case 173:
#line 676 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (3)].intval) | (yyvsp[(3) - (3)].intval); }
    break;

  case 174:
#line 680 "xi-grammar.y"
    { (yyval.intval) = STHREADED; }
    break;

  case 175:
#line 682 "xi-grammar.y"
    { (yyval.intval) = SSYNC; }
    break;

  case 176:
#line 684 "xi-grammar.y"
    { (yyval.intval) = SIGET; }
    break;

  case 177:
#line 686 "xi-grammar.y"
    { (yyval.intval) = SLOCKED; }
    break;

  case 178:
#line 688 "xi-grammar.y"
    { (yyval.intval) = SCREATEHERE; }
    break;

  case 179:
#line 690 "xi-grammar.y"
    { (yyval.intval) = SCREATEHOME; }
    break;

  case 180:
#line 692 "xi-grammar.y"
    { (yyval.intval) = SNOKEEP; }
    break;

  case 181:
#line 694 "xi-grammar.y"
    { (yyval.intval) = SNOTRACE; }
    break;

  case 182:
#line 696 "xi-grammar.y"
    { (yyval.intval) = SIMMEDIATE; }
    break;

  case 183:
#line 698 "xi-grammar.y"
    { (yyval.intval) = SSKIPSCHED; }
    break;

  case 184:
#line 700 "xi-grammar.y"
    { (yyval.intval) = SINLINE; }
    break;

  case 185:
#line 702 "xi-grammar.y"
    { (yyval.intval) = SLOCAL; }
    break;

  case 186:
#line 704 "xi-grammar.y"
    { (yyval.intval) = SPYTHON; }
    break;

  case 187:
#line 708 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 188:
#line 710 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 189:
#line 712 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 190:
#line 716 "xi-grammar.y"
    { (yyval.strval) = ""; }
    break;

  case 191:
#line 718 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 192:
#line 720 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[(1) - (3)].strval))+strlen((yyvsp[(3) - (3)].strval))+3];
			sprintf(tmp,"%s, %s", (yyvsp[(1) - (3)].strval), (yyvsp[(3) - (3)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 193:
#line 728 "xi-grammar.y"
    { (yyval.strval) = ""; }
    break;

  case 194:
#line 730 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 195:
#line 732 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s[%s]%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 196:
#line 738 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s{%s}%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 197:
#line 744 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s(%s)%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 198:
#line 750 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(2) - (4)].strval))+strlen((yyvsp[(4) - (4)].strval))+3];
			sprintf(tmp,"(%s)%s", (yyvsp[(2) - (4)].strval), (yyvsp[(4) - (4)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 199:
#line 758 "xi-grammar.y"
    {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			(yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].type),(yyvsp[(2) - (3)].strval));
		}
    break;

  case 200:
#line 765 "xi-grammar.y"
    { 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			(yyval.intval) = 0;
		}
    break;

  case 201:
#line 773 "xi-grammar.y"
    { 
			in_braces=0;
			(yyval.intval) = 0;
		}
    break;

  case 202:
#line 780 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (1)].type));}
    break;

  case 203:
#line 782 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].type),(yyvsp[(2) - (3)].strval)); (yyval.pname)->setConditional((yyvsp[(3) - (3)].intval)); }
    break;

  case 204:
#line 784 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (4)].type),(yyvsp[(2) - (4)].strval),0,(yyvsp[(4) - (4)].val));}
    break;

  case 205:
#line 786 "xi-grammar.y"
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].pname)->getType(), (yyvsp[(1) - (3)].pname)->getName() ,(yyvsp[(2) - (3)].strval));
		}
    break;

  case 206:
#line 793 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (1)].pname)); }
    break;

  case 207:
#line 795 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (3)].pname),(yyvsp[(3) - (3)].plist)); }
    break;

  case 208:
#line 799 "xi-grammar.y"
    { (yyval.plist) = (yyvsp[(2) - (3)].plist); }
    break;

  case 209:
#line 801 "xi-grammar.y"
    { (yyval.plist) = 0; }
    break;

  case 210:
#line 805 "xi-grammar.y"
    { (yyval.val) = 0; }
    break;

  case 211:
#line 807 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(3) - (3)].strval)); }
    break;

  case 212:
#line 811 "xi-grammar.y"
    { (yyval.sc) = 0; }
    break;

  case 213:
#line 813 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSDAGENTRY, (yyvsp[(1) - (1)].sc)); }
    break;

  case 214:
#line 815 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSDAGENTRY, (yyvsp[(2) - (3)].sc)); }
    break;

  case 215:
#line 819 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSLIST, (yyvsp[(1) - (1)].sc)); }
    break;

  case 216:
#line 821 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSLIST, (yyvsp[(1) - (2)].sc), (yyvsp[(2) - (2)].sc));  }
    break;

  case 217:
#line 825 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SOLIST, (yyvsp[(1) - (1)].sc)); }
    break;

  case 218:
#line 827 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SOLIST, (yyvsp[(1) - (2)].sc), (yyvsp[(2) - (2)].sc)); }
    break;

  case 219:
#line 831 "xi-grammar.y"
    { (yyval.sc) = 0; }
    break;

  case 220:
#line 833 "xi-grammar.y"
    { (yyval.sc) = (yyvsp[(3) - (4)].sc); }
    break;

  case 221:
#line 837 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SPUBLISHES, new SdagConstruct(SIDENT, (yyvsp[(1) - (1)].strval))); }
    break;

  case 222:
#line 839 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SPUBLISHES, new SdagConstruct(SIDENT, (yyvsp[(1) - (3)].strval)), (yyvsp[(3) - (3)].sc));  }
    break;

  case 223:
#line 843 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 224:
#line 845 "xi-grammar.y"
    { (yyval.strval) = 0; }
    break;

  case 225:
#line 849 "xi-grammar.y"
    { RemoveSdagComments((yyvsp[(4) - (6)].strval));
		   (yyval.sc) = new SdagConstruct(SATOMIC, new XStr((yyvsp[(4) - (6)].strval)), (yyvsp[(6) - (6)].sc), 0,0,0,0, 0 ); 
		   if ((yyvsp[(2) - (6)].strval)) { (yyvsp[(2) - (6)].strval)[strlen((yyvsp[(2) - (6)].strval))-1]=0; (yyval.sc)->traceName = new XStr((yyvsp[(2) - (6)].strval)+1); }
		 }
    break;

  case 226:
#line 854 "xi-grammar.y"
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

  case 227:
#line 868 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SWHEN, 0, 0,0,0,0,0,(yyvsp[(2) - (4)].entrylist)); }
    break;

  case 228:
#line 870 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SWHEN, 0, 0, 0,0,0, (yyvsp[(3) - (3)].sc), (yyvsp[(2) - (3)].entrylist)); }
    break;

  case 229:
#line 872 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SWHEN, 0, 0, 0,0,0, (yyvsp[(4) - (5)].sc), (yyvsp[(2) - (5)].entrylist)); }
    break;

  case 230:
#line 874 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SOVERLAP,0, 0,0,0,0,(yyvsp[(3) - (4)].sc), 0); }
    break;

  case 231:
#line 876 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFOR, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (11)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(5) - (11)].strval)),
		             new SdagConstruct(SINT_EXPR, (yyvsp[(7) - (11)].strval)), 0, (yyvsp[(10) - (11)].sc), 0); }
    break;

  case 232:
#line 879 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFOR, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (9)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(5) - (9)].strval)), 
		         new SdagConstruct(SINT_EXPR, (yyvsp[(7) - (9)].strval)), 0, (yyvsp[(9) - (9)].sc), 0); }
    break;

  case 233:
#line 882 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFORALL, 0, new SdagConstruct(SIDENT, (yyvsp[(3) - (12)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(6) - (12)].strval)), 
		             new SdagConstruct(SINT_EXPR, (yyvsp[(8) - (12)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(10) - (12)].strval)), (yyvsp[(12) - (12)].sc), 0); }
    break;

  case 234:
#line 885 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFORALL, 0, new SdagConstruct(SIDENT, (yyvsp[(3) - (14)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(6) - (14)].strval)), 
		                 new SdagConstruct(SINT_EXPR, (yyvsp[(8) - (14)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(10) - (14)].strval)), (yyvsp[(13) - (14)].sc), 0); }
    break;

  case 235:
#line 888 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (6)].strval)), (yyvsp[(6) - (6)].sc),0,0,(yyvsp[(5) - (6)].sc),0); }
    break;

  case 236:
#line 890 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (8)].strval)), (yyvsp[(8) - (8)].sc),0,0,(yyvsp[(6) - (8)].sc),0); }
    break;

  case 237:
#line 892 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (5)].strval)), 0,0,0,(yyvsp[(5) - (5)].sc),0); }
    break;

  case 238:
#line 894 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SWHILE, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (7)].strval)), 0,0,0,(yyvsp[(6) - (7)].sc),0); }
    break;

  case 239:
#line 896 "xi-grammar.y"
    { (yyval.sc) = (yyvsp[(2) - (3)].sc); }
    break;

  case 240:
#line 898 "xi-grammar.y"
    { RemoveSdagComments((yyvsp[(2) - (3)].strval));
                   (yyval.sc) = new SdagConstruct(SATOMIC, new XStr((yyvsp[(2) - (3)].strval)), NULL, 0,0,0,0, 0 );
                 }
    break;

  case 241:
#line 904 "xi-grammar.y"
    { (yyval.sc) = 0; }
    break;

  case 242:
#line 906 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SELSE, 0,0,0,0,0, (yyvsp[(2) - (2)].sc),0); }
    break;

  case 243:
#line 908 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SELSE, 0,0,0,0,0, (yyvsp[(3) - (4)].sc),0); }
    break;

  case 244:
#line 911 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFORWARD, new SdagConstruct(SIDENT, (yyvsp[(1) - (1)].strval))); }
    break;

  case 245:
#line 913 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFORWARD, new SdagConstruct(SIDENT, (yyvsp[(1) - (3)].strval)), (yyvsp[(3) - (3)].sc));  }
    break;

  case 246:
#line 917 "xi-grammar.y"
    { in_int_expr = 0; (yyval.intval) = 0; }
    break;

  case 247:
#line 921 "xi-grammar.y"
    { in_int_expr = 1; (yyval.intval) = 0; }
    break;

  case 248:
#line 925 "xi-grammar.y"
    { 
		  if ((yyvsp[(2) - (2)].plist) != 0)
		     (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[(1) - (2)].strval), (yyvsp[(2) - (2)].plist), 0, 0, 0, 0); 
		  else
		     (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[(1) - (2)].strval), 
				new ParamList(new Parameter(0, new BuiltinType("void"))), 0, 0, 0, 0); 
		}
    break;

  case 249:
#line 933 "xi-grammar.y"
    { if ((yyvsp[(5) - (5)].plist) != 0)
		    (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[(1) - (5)].strval), (yyvsp[(5) - (5)].plist), 0, 0, (yyvsp[(3) - (5)].strval), 0); 
		  else
		    (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[(1) - (5)].strval), new ParamList(new Parameter(0, new BuiltinType("void"))), 0, 0, (yyvsp[(3) - (5)].strval), 0); 
		}
    break;

  case 250:
#line 941 "xi-grammar.y"
    { (yyval.entrylist) = new EntryList((yyvsp[(1) - (1)].entry)); }
    break;

  case 251:
#line 943 "xi-grammar.y"
    { (yyval.entrylist) = new EntryList((yyvsp[(1) - (3)].entry),(yyvsp[(3) - (3)].entrylist)); }
    break;

  case 252:
#line 947 "xi-grammar.y"
    { in_bracket=1; }
    break;

  case 253:
#line 950 "xi-grammar.y"
    { in_bracket=0; }
    break;

  case 254:
#line 954 "xi-grammar.y"
    { if (!macroDefined((yyvsp[(2) - (2)].strval), 1)) in_comment = 1; }
    break;

  case 255:
#line 958 "xi-grammar.y"
    { if (!macroDefined((yyvsp[(2) - (2)].strval), 0)) in_comment = 1; }
    break;


/* Line 1267 of yacc.c.  */
#line 3376 "y.tab.c"
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


#line 961 "xi-grammar.y"

void yyerror(const char *mesg)
{
  cout << cur_file<<":"<<lineno<<": Charmxi syntax error> " << mesg << endl;
  // return 0;
}

