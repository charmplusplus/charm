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
     IDENT = 308,
     NUMBER = 309,
     LITERAL = 310,
     CPROGRAM = 311,
     HASHIF = 312,
     HASHIFDEF = 313,
     SCOPE = 314,
     INT = 315,
     LONG = 316,
     SHORT = 317,
     CHAR = 318,
     FLOAT = 319,
     DOUBLE = 320,
     UNSIGNED = 321
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
#define IDENT 308
#define NUMBER 309
#define LITERAL 310
#define CPROGRAM 311
#define HASHIF 312
#define HASHIFDEF 313
#define SCOPE 314
#define INT 315
#define LONG 316
#define SHORT 317
#define CHAR 318
#define FLOAT 319
#define DOUBLE 320
#define UNSIGNED 321




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
#line 279 "y.tab.c"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */
#line 292 "y.tab.c"

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
#define YYLAST   562

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  81
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  103
/* YYNRULES -- Number of rules.  */
#define YYNRULES  252
/* YYNRULES -- Number of states.  */
#define YYNSTATES  494

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   321

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    76,     2,
      74,    75,    73,     2,    70,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    79,    67,
      71,    80,    72,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    77,     2,    78,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    68,     2,    69,     2,     2,     2,     2,
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
      65,    66
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     5,     6,     9,    10,    12,    13,    15,
      17,    19,    23,    27,    31,    33,    38,    39,    42,    48,
      51,    54,    58,    61,    64,    67,    70,    73,    75,    77,
      79,    81,    83,    85,    89,    90,    92,    93,    97,    99,
     101,   103,   105,   108,   111,   115,   119,   122,   125,   128,
     130,   132,   135,   137,   140,   143,   145,   147,   150,   153,
     156,   165,   167,   169,   171,   173,   176,   179,   182,   184,
     186,   188,   192,   193,   196,   201,   207,   208,   210,   211,
     215,   217,   221,   223,   225,   226,   230,   232,   236,   237,
     239,   241,   242,   246,   248,   252,   254,   256,   257,   259,
     260,   263,   269,   271,   274,   278,   285,   286,   289,   291,
     295,   301,   307,   313,   319,   324,   328,   335,   342,   348,
     354,   360,   366,   372,   377,   385,   386,   389,   390,   393,
     396,   400,   403,   407,   409,   413,   418,   421,   424,   427,
     430,   433,   435,   440,   441,   444,   447,   450,   453,   456,
     460,   464,   468,   472,   479,   483,   490,   494,   501,   503,
     507,   509,   512,   514,   522,   528,   530,   532,   533,   537,
     539,   543,   545,   547,   549,   551,   553,   555,   557,   559,
     561,   563,   565,   567,   570,   572,   574,   576,   577,   579,
     583,   584,   586,   592,   598,   604,   609,   613,   615,   617,
     619,   623,   628,   632,   634,   638,   642,   645,   646,   650,
     651,   653,   657,   659,   662,   664,   667,   668,   673,   675,
     679,   681,   682,   689,   698,   703,   707,   713,   718,   730,
     740,   753,   768,   775,   784,   790,   798,   802,   806,   807,
     810,   815,   817,   821,   823,   825,   828,   834,   836,   840,
     842,   844,   847
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
      82,     0,    -1,    83,    -1,    -1,    88,    83,    -1,    -1,
       5,    -1,    -1,    67,    -1,    53,    -1,    53,    -1,    87,
      59,    53,    -1,     3,    86,    89,    -1,     4,    86,    89,
      -1,    67,    -1,    68,    90,    69,    85,    -1,    -1,    91,
      90,    -1,    84,    68,    90,    69,    85,    -1,    84,    88,
      -1,    84,   146,    -1,    84,   125,    67,    -1,    84,   128,
      -1,    84,   129,    -1,    84,   130,    -1,    84,   132,    -1,
      84,   143,    -1,   182,    -1,   183,    -1,   104,    -1,    54,
      -1,    55,    -1,    92,    -1,    92,    70,    93,    -1,    -1,
      93,    -1,    -1,    71,    94,    72,    -1,    60,    -1,    61,
      -1,    62,    -1,    63,    -1,    66,    60,    -1,    66,    61,
      -1,    66,    61,    60,    -1,    66,    61,    61,    -1,    66,
      62,    -1,    66,    63,    -1,    61,    61,    -1,    64,    -1,
      65,    -1,    61,    65,    -1,    35,    -1,    86,    95,    -1,
      87,    95,    -1,    96,    -1,    98,    -1,    99,    73,    -1,
     100,    73,    -1,   101,    73,    -1,   103,    74,    73,    86,
      75,    74,   164,    75,    -1,    99,    -1,   100,    -1,   101,
      -1,   102,    -1,    36,   103,    -1,   103,    36,    -1,   103,
      76,    -1,   103,    -1,    54,    -1,    87,    -1,    77,   105,
      78,    -1,    -1,   106,   107,    -1,     6,   104,    87,   107,
      -1,     6,    16,    99,    73,    86,    -1,    -1,    35,    -1,
      -1,    77,   112,    78,    -1,   113,    -1,   113,    70,   112,
      -1,    37,    -1,    38,    -1,    -1,    77,   115,    78,    -1,
     120,    -1,   120,    70,   115,    -1,    -1,    55,    -1,    51,
      -1,    -1,    77,   119,    78,    -1,   117,    -1,   117,    70,
     119,    -1,    30,    -1,    51,    -1,    -1,    17,    -1,    -1,
      77,    78,    -1,   121,   104,    86,   122,    67,    -1,   123,
      -1,   123,   124,    -1,    16,   111,    97,    -1,    16,   111,
      97,    68,   124,    69,    -1,    -1,    79,   127,    -1,    98,
      -1,    98,    70,   127,    -1,    11,   114,    98,   126,   144,
      -1,    12,   114,    97,   126,   144,    -1,    13,   114,    97,
     126,   144,    -1,    14,   114,    97,   126,   144,    -1,    77,
      54,    86,    78,    -1,    77,    86,    78,    -1,    15,   118,
     131,    98,   126,   144,    -1,    15,   131,   118,    98,   126,
     144,    -1,    11,   114,    86,   126,   144,    -1,    12,   114,
      86,   126,   144,    -1,    13,   114,    86,   126,   144,    -1,
      14,   114,    86,   126,   144,    -1,    15,   131,    86,   126,
     144,    -1,    16,   111,    86,    67,    -1,    16,   111,    86,
      68,   124,    69,    67,    -1,    -1,    80,   104,    -1,    -1,
      80,    54,    -1,    80,    55,    -1,    18,    86,   138,    -1,
     102,   139,    -1,   104,    86,   139,    -1,   140,    -1,   140,
      70,   141,    -1,    22,    71,   141,    72,    -1,   142,   133,
      -1,   142,   134,    -1,   142,   135,    -1,   142,   136,    -1,
     142,   137,    -1,    67,    -1,    68,   145,    69,    85,    -1,
      -1,   151,   145,    -1,   108,    67,    -1,   109,    67,    -1,
     148,    67,    -1,   147,    67,    -1,    10,   149,    67,    -1,
      19,   150,    67,    -1,    18,    86,    67,    -1,     8,   110,
      87,    -1,     8,   110,    87,    74,   110,    75,    -1,     7,
     110,    87,    -1,     7,   110,    87,    74,   110,    75,    -1,
       9,   110,    87,    -1,     9,   110,    87,    74,   110,    75,
      -1,    98,    -1,    98,    70,   149,    -1,    55,    -1,   152,
      67,    -1,   146,    -1,    39,   154,   153,    86,   165,   166,
     167,    -1,    39,   154,    86,   165,   167,    -1,    35,    -1,
     100,    -1,    -1,    77,   155,    78,    -1,   156,    -1,   156,
      70,   155,    -1,    21,    -1,    23,    -1,    24,    -1,    25,
      -1,    31,    -1,    32,    -1,    33,    -1,    34,    -1,    26,
      -1,    27,    -1,    28,    -1,    52,    -1,    51,   116,    -1,
      55,    -1,    54,    -1,    87,    -1,    -1,    56,    -1,    56,
      70,   158,    -1,    -1,    56,    -1,    56,    77,   159,    78,
     159,    -1,    56,    68,   159,    69,   159,    -1,    56,    74,
     158,    75,   159,    -1,    74,   159,    75,   159,    -1,   104,
      86,    77,    -1,    68,    -1,    69,    -1,   104,    -1,   104,
      86,   121,    -1,   104,    86,    80,   157,    -1,   160,   159,
      78,    -1,   163,    -1,   163,    70,   164,    -1,    74,   164,
      75,    -1,    74,    75,    -1,    -1,    20,    80,    54,    -1,
      -1,   173,    -1,    68,   168,    69,    -1,   173,    -1,   173,
     168,    -1,   173,    -1,   173,   168,    -1,    -1,    50,    74,
     171,    75,    -1,    53,    -1,    53,    70,   171,    -1,    55,
      -1,    -1,    45,   172,   161,   159,   162,   170,    -1,    49,
      74,    53,   165,    75,   161,   159,    69,    -1,    43,   179,
      68,    69,    -1,    43,   179,   173,    -1,    43,   179,    68,
     168,    69,    -1,    44,    68,   169,    69,    -1,    40,   177,
     159,    67,   159,    67,   159,   176,    68,   168,    69,    -1,
      40,   177,   159,    67,   159,    67,   159,   176,   173,    -1,
      41,    77,    53,    78,   177,   159,    79,   159,    70,   159,
     176,   173,    -1,    41,    77,    53,    78,   177,   159,    79,
     159,    70,   159,   176,    68,   168,    69,    -1,    47,   177,
     159,   176,   173,   174,    -1,    47,   177,   159,   176,    68,
     168,    69,   174,    -1,    42,   177,   159,   176,   173,    -1,
      42,   177,   159,   176,    68,   168,    69,    -1,    46,   175,
      67,    -1,   161,   159,   162,    -1,    -1,    48,   173,    -1,
      48,    68,   168,    69,    -1,    53,    -1,    53,    70,   175,
      -1,    75,    -1,    74,    -1,    53,   165,    -1,    53,   180,
     159,   181,   165,    -1,   178,    -1,   178,    70,   179,    -1,
      77,    -1,    78,    -1,    57,    86,    -1,    58,    86,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   135,   135,   140,   143,   148,   149,   154,   155,   159,
     163,   165,   173,   177,   184,   186,   191,   192,   196,   198,
     200,   202,   204,   206,   208,   210,   212,   214,   216,   220,
     222,   224,   228,   230,   235,   236,   241,   242,   246,   248,
     250,   252,   254,   256,   258,   260,   262,   264,   266,   268,
     270,   272,   274,   278,   279,   281,   283,   287,   291,   293,
     297,   301,   303,   305,   307,   310,   312,   316,   318,   322,
     324,   328,   333,   334,   338,   342,   347,   348,   353,   354,
     364,   366,   370,   372,   377,   378,   382,   384,   389,   390,
     394,   399,   400,   404,   406,   410,   412,   417,   418,   422,
     423,   426,   430,   432,   436,   438,   443,   444,   448,   450,
     454,   456,   460,   464,   468,   474,   478,   480,   484,   486,
     490,   494,   498,   502,   504,   509,   510,   515,   516,   518,
     522,   524,   526,   530,   532,   536,   540,   542,   544,   546,
     548,   552,   554,   559,   577,   581,   583,   585,   586,   588,
     590,   592,   596,   598,   600,   603,   608,   610,   614,   616,
     619,   623,   625,   629,   640,   653,   655,   660,   661,   665,
     667,   671,   673,   675,   677,   679,   681,   683,   685,   687,
     689,   691,   693,   695,   699,   701,   703,   708,   709,   711,
     720,   721,   723,   729,   735,   741,   749,   756,   764,   771,
     773,   775,   777,   784,   786,   790,   792,   797,   798,   803,
     804,   806,   810,   812,   816,   818,   823,   824,   828,   830,
     834,   837,   840,   845,   859,   861,   863,   865,   867,   870,
     873,   876,   879,   881,   883,   885,   887,   889,   896,   897,
     899,   902,   904,   908,   912,   916,   924,   932,   934,   938,
     941,   945,   949
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
  "PYTHON", "LOCAL", "IDENT", "NUMBER", "LITERAL", "CPROGRAM", "HASHIF",
  "HASHIFDEF", "SCOPE", "INT", "LONG", "SHORT", "CHAR", "FLOAT", "DOUBLE",
  "UNSIGNED", "';'", "'{'", "'}'", "','", "'<'", "'>'", "'*'", "'('",
  "')'", "'&'", "'['", "']'", "':'", "'='", "$accept", "File",
  "ModuleEList", "OptExtern", "OptSemiColon", "Name", "QualName", "Module",
  "ConstructEList", "ConstructList", "Construct", "TParam", "TParamList",
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
  "Member", "Entry", "EReturn", "EAttribs", "EAttribList", "EAttrib",
  "DefaultParameter", "CPROGRAM_List", "CCode", "ParamBracketStart",
  "ParamBraceStart", "ParamBraceEnd", "Parameter", "ParamList",
  "EParameters", "OptStackSize", "OptSdagCode", "Slist", "Olist",
  "OptPubList", "PublishesList", "OptTraceName", "SingleConstruct",
  "HasElse", "ForwardList", "EndIntExpr", "StartIntExpr", "SEntry",
  "SEntryList", "SParamBracketStart", "SParamBracketEnd", "HashIFComment",
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
     315,   316,   317,   318,   319,   320,   321,    59,   123,   125,
      44,    60,    62,    42,    40,    41,    38,    91,    93,    58,
      61
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    81,    82,    83,    83,    84,    84,    85,    85,    86,
      87,    87,    88,    88,    89,    89,    90,    90,    91,    91,
      91,    91,    91,    91,    91,    91,    91,    91,    91,    92,
      92,    92,    93,    93,    94,    94,    95,    95,    96,    96,
      96,    96,    96,    96,    96,    96,    96,    96,    96,    96,
      96,    96,    96,    97,    98,    99,    99,   100,   101,   101,
     102,   103,   103,   103,   103,   103,   103,   104,   104,   105,
     105,   106,   107,   107,   108,   109,   110,   110,   111,   111,
     112,   112,   113,   113,   114,   114,   115,   115,   116,   116,
     117,   118,   118,   119,   119,   120,   120,   121,   121,   122,
     122,   123,   124,   124,   125,   125,   126,   126,   127,   127,
     128,   128,   129,   130,   131,   131,   132,   132,   133,   133,
     134,   135,   136,   137,   137,   138,   138,   139,   139,   139,
     140,   140,   140,   141,   141,   142,   143,   143,   143,   143,
     143,   144,   144,   145,   145,   146,   146,   146,   146,   146,
     146,   146,   147,   147,   147,   147,   148,   148,   149,   149,
     150,   151,   151,   152,   152,   153,   153,   154,   154,   155,
     155,   156,   156,   156,   156,   156,   156,   156,   156,   156,
     156,   156,   156,   156,   157,   157,   157,   158,   158,   158,
     159,   159,   159,   159,   159,   159,   160,   161,   162,   163,
     163,   163,   163,   164,   164,   165,   165,   166,   166,   167,
     167,   167,   168,   168,   169,   169,   170,   170,   171,   171,
     172,   172,   173,   173,   173,   173,   173,   173,   173,   173,
     173,   173,   173,   173,   173,   173,   173,   173,   174,   174,
     174,   175,   175,   176,   177,   178,   178,   179,   179,   180,
     181,   182,   183
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     0,     2,     0,     1,     0,     1,     1,
       1,     3,     3,     3,     1,     4,     0,     2,     5,     2,
       2,     3,     2,     2,     2,     2,     2,     1,     1,     1,
       1,     1,     1,     3,     0,     1,     0,     3,     1,     1,
       1,     1,     2,     2,     3,     3,     2,     2,     2,     1,
       1,     2,     1,     2,     2,     1,     1,     2,     2,     2,
       8,     1,     1,     1,     1,     2,     2,     2,     1,     1,
       1,     3,     0,     2,     4,     5,     0,     1,     0,     3,
       1,     3,     1,     1,     0,     3,     1,     3,     0,     1,
       1,     0,     3,     1,     3,     1,     1,     0,     1,     0,
       2,     5,     1,     2,     3,     6,     0,     2,     1,     3,
       5,     5,     5,     5,     4,     3,     6,     6,     5,     5,
       5,     5,     5,     4,     7,     0,     2,     0,     2,     2,
       3,     2,     3,     1,     3,     4,     2,     2,     2,     2,
       2,     1,     4,     0,     2,     2,     2,     2,     2,     3,
       3,     3,     3,     6,     3,     6,     3,     6,     1,     3,
       1,     2,     1,     7,     5,     1,     1,     0,     3,     1,
       3,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     2,     1,     1,     1,     0,     1,     3,
       0,     1,     5,     5,     5,     4,     3,     1,     1,     1,
       3,     4,     3,     1,     3,     3,     2,     0,     3,     0,
       1,     3,     1,     2,     1,     2,     0,     4,     1,     3,
       1,     0,     6,     8,     4,     3,     5,     4,    11,     9,
      12,    14,     6,     8,     5,     7,     3,     3,     0,     2,
       4,     1,     3,     1,     1,     2,     5,     1,     3,     1,
       1,     2,     2
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       3,     0,     0,     0,     2,     3,     9,     0,     0,     1,
       4,    14,     5,    12,    13,     6,     0,     0,     0,     0,
       5,    27,    28,   251,   252,     0,    76,    76,    76,     0,
      84,    84,    84,    84,     0,    78,     0,     0,     0,     5,
      19,     0,     0,     0,    22,    23,    24,    25,     0,    26,
      20,     0,     0,     7,    17,     0,    52,     0,    10,    38,
      39,    40,    41,    49,    50,     0,    36,    55,    56,    61,
      62,    63,    64,    68,     0,    77,     0,     0,     0,   158,
       0,     0,     0,     0,     0,     0,     0,     0,    91,     0,
       0,     0,   160,     0,     0,     0,   145,   146,    21,    84,
      84,    84,    84,     0,    78,   136,   137,   138,   139,   140,
     148,   147,     8,    15,     0,    65,    48,    51,    42,    43,
      46,    47,     0,    34,    54,    57,    58,    59,    66,     0,
      67,    72,   154,   152,   156,     0,   149,    95,    96,     0,
      86,   106,    36,   106,   106,   106,    90,     0,     0,    93,
       0,     0,     0,     0,     0,    82,    83,     0,    80,   104,
     151,   150,     0,    64,     0,   133,     0,     7,     0,     0,
       0,     0,     0,     0,     0,    44,    45,    11,    30,    31,
      32,    35,     0,    29,     0,     0,    72,    74,    76,    76,
      76,   159,    85,     0,     0,     0,    53,     0,     0,     0,
       0,   115,     0,    92,   106,   106,    79,     0,    97,   125,
       0,   131,   127,     0,   135,    18,   106,   106,   106,   106,
     106,     0,    75,     0,    37,     0,    69,    70,     0,    73,
       0,     0,     0,    87,   108,   107,   141,   143,   110,   111,
     112,   113,   114,    94,     0,     0,    81,    98,     0,    97,
       0,     0,   130,   128,   129,   132,   134,     0,     0,     0,
       0,     0,   123,    97,    33,     0,    71,   155,   153,   157,
       0,   167,     0,   162,   143,     0,   116,   117,     0,   103,
     105,   126,   118,   119,   120,   121,   122,     0,     0,   109,
       0,     0,     7,   144,   161,    99,     0,   199,   190,   203,
       0,   171,   172,   173,   174,   179,   180,   181,   175,   176,
     177,   178,    88,   182,     0,   169,    52,    10,     0,     0,
     166,     0,   142,     0,     0,   124,    97,   191,   190,     0,
       0,    60,    89,   183,   168,     0,     0,   209,     0,   100,
     101,   196,     0,   200,   190,   187,   190,     0,   202,   204,
     170,   206,     0,     0,     0,     0,     0,     0,   221,     0,
       0,     0,   197,   190,   164,   210,   207,   185,   184,   186,
     201,     0,   188,     0,     0,   190,   205,   244,   190,     0,
     190,     0,   247,     0,     0,   220,     0,   241,     0,   190,
       0,   197,     0,   212,     0,     0,   209,   190,   187,   190,
     190,   195,     0,     0,     0,   249,   245,   190,     0,   197,
     225,     0,   214,   190,     0,   236,     0,     0,   211,   213,
     198,   237,     0,   163,   193,   189,   194,   192,   190,     0,
     243,     0,     0,   248,   224,     0,   227,   215,     0,   242,
       0,     0,   208,     0,   190,   197,   234,   250,     0,   226,
     216,   197,   238,     0,   190,     0,     0,   246,     0,   222,
       0,     0,   232,   190,     0,   190,   235,     0,   238,   197,
     239,     0,     0,     0,   218,     0,   233,     0,   223,   197,
     229,   190,     0,   217,   240,     0,     0,   219,   228,     0,
     197,   230,     0,   231
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     3,     4,    18,   113,   142,    66,     5,    13,    19,
      20,   180,   181,   182,   124,    67,   143,    68,    69,    70,
      71,    72,    73,   297,   228,   186,   187,    41,    42,    76,
      90,   157,   158,    82,   139,   333,   149,    87,   150,   140,
     248,   324,   249,   250,    43,   195,   235,    44,    45,    46,
      88,    47,   105,   106,   107,   108,   109,   252,   211,   165,
     166,    48,    49,   238,   272,   273,    51,    52,    80,    93,
     274,   275,   321,   291,   314,   315,   370,   373,   329,   298,
     363,   421,   299,   300,   337,   396,   364,   392,   411,   459,
     475,   386,   393,   462,   388,   431,   378,   382,   383,   407,
     448,    21,    22
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -405
static const yytype_int16 yypact[] =
{
     116,    11,    11,    33,  -405,   116,  -405,    73,    73,  -405,
    -405,  -405,    39,  -405,  -405,  -405,    11,    11,   200,    -2,
      39,  -405,  -405,  -405,  -405,   218,    24,    24,    24,    45,
       9,     9,     9,     9,    15,    25,    11,    22,     8,    39,
    -405,    37,    42,    47,  -405,  -405,  -405,  -405,   212,  -405,
    -405,    76,    82,    89,  -405,   334,  -405,   320,  -405,  -405,
      19,  -405,  -405,  -405,  -405,   175,     2,  -405,  -405,    91,
      93,   104,  -405,    -8,    45,  -405,    45,    45,    45,    88,
      95,    -6,    45,    11,    11,    11,    77,   117,   119,   144,
      11,   135,  -405,   150,   238,   151,  -405,  -405,  -405,     9,
       9,     9,     9,   117,    25,  -405,  -405,  -405,  -405,  -405,
    -405,  -405,  -405,  -405,   156,   -16,  -405,  -405,  -405,   171,
    -405,  -405,   202,   299,  -405,  -405,  -405,  -405,  -405,   190,
    -405,   -40,   -25,    -4,     4,    45,  -405,  -405,  -405,   186,
     192,   188,   195,   188,   188,   188,  -405,    11,   191,   205,
     198,   187,    45,   214,    45,  -405,  -405,   207,   216,   204,
    -405,  -405,    11,    67,    11,   219,   220,    89,    11,    11,
      11,    11,    11,    11,    11,  -405,  -405,  -405,  -405,  -405,
     223,  -405,   222,  -405,    11,   189,   229,  -405,    24,    24,
      24,  -405,  -405,    -6,    45,   179,  -405,   179,   179,   179,
     217,  -405,   214,  -405,   188,   188,  -405,   144,   280,   227,
     194,  -405,   228,   238,  -405,  -405,   188,   188,   188,   188,
     188,   184,  -405,   299,  -405,   234,  -405,   251,   233,  -405,
     253,   258,   261,  -405,   267,  -405,  -405,   182,  -405,  -405,
    -405,  -405,  -405,  -405,   179,   179,  -405,  -405,   320,     6,
     269,   320,  -405,  -405,  -405,  -405,  -405,   179,   179,   179,
     179,   179,  -405,   280,  -405,   265,  -405,  -405,  -405,  -405,
      45,   263,   272,  -405,   182,   275,  -405,  -405,    11,  -405,
    -405,  -405,  -405,  -405,  -405,  -405,  -405,   274,   320,  -405,
     298,   353,    89,  -405,  -405,   268,   277,    11,   -26,   276,
     273,  -405,  -405,  -405,  -405,  -405,  -405,  -405,  -405,  -405,
    -405,  -405,   292,  -405,   279,   281,   305,   293,   294,    91,
    -405,    11,  -405,   288,   303,  -405,    26,    68,   -26,   296,
     320,  -405,  -405,  -405,  -405,   298,   252,   390,   294,  -405,
    -405,  -405,    98,  -405,   -26,   315,   -26,   297,  -405,  -405,
    -405,  -405,   300,   302,   301,   302,   324,   311,   335,   336,
     302,   317,   410,   -26,  -405,  -405,   372,  -405,  -405,   251,
    -405,   332,   323,   327,   325,   -26,  -405,  -405,   -26,   351,
     -26,   -21,   337,   420,   410,  -405,   340,   339,   338,   -26,
     357,  -405,   342,   410,   343,   348,   390,   -26,   315,   -26,
     -26,  -405,   371,   362,   366,  -405,  -405,   -26,   324,   380,
    -405,   373,   410,   -26,   336,  -405,   366,   294,  -405,  -405,
    -405,  -405,   389,  -405,  -405,  -405,  -405,  -405,   -26,   302,
    -405,   430,   367,  -405,  -405,   375,  -405,  -405,   343,  -405,
     440,   393,  -405,   379,   -26,   410,  -405,  -405,   294,  -405,
     397,   410,   472,   340,   -26,   442,   453,  -405,   449,  -405,
     455,   450,  -405,   -26,   366,   -26,  -405,   473,   472,   410,
    -405,   456,   460,   457,   459,   458,  -405,   461,  -405,   410,
    -405,   -26,   473,  -405,  -405,   462,   366,  -405,  -405,   470,
     410,  -405,   463,  -405
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -405,  -405,   529,  -405,  -159,    -1,   -72,   517,   528,     7,
    -405,  -405,   314,  -405,   398,  -405,    70,   -20,   -52,   248,
    -405,   -84,   484,   -18,  -405,  -405,   356,  -405,  -405,   -10,
     439,   341,  -405,    85,   352,  -405,  -405,   464,   344,  -405,
     221,  -405,  -405,  -224,  -405,   -93,   283,  -405,  -405,  -405,
     -56,  -405,  -405,  -405,  -405,  -405,  -405,  -405,   345,  -405,
     331,  -405,  -405,     0,   282,   531,  -405,  -405,   415,  -405,
    -405,  -405,  -405,  -405,   224,  -405,  -405,   153,  -306,  -405,
    -372,   120,  -405,  -294,  -327,  -405,   158,  -380,  -405,  -405,
      78,  -405,  -296,    87,   147,  -404,  -334,  -405,   154,  -405,
    -405,  -405,  -405
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -166
static const yytype_int16 yytable[] =
{
       7,     8,   131,   114,   132,   133,   134,    74,   215,    79,
     163,   366,   440,   419,   413,    23,    24,    77,    78,   122,
     128,   380,   347,   247,   137,   279,   389,    54,   128,   435,
     327,   152,   437,     9,   122,    91,   349,   185,   371,   287,
     374,   365,   352,   247,    15,   138,    95,   172,   328,   188,
     197,   198,   199,   336,   406,   122,   405,   394,   129,    75,
     472,   122,   141,   122,     6,   456,   129,    53,   130,   401,
     189,   460,   402,   123,   404,  -102,   164,    92,   190,    94,
     116,   463,   489,   416,   117,   148,    81,   410,   412,   477,
     441,   424,    86,   426,   427,   444,    16,    17,    58,   485,
     365,   432,    89,   341,    96,   183,   342,   438,   -16,    97,
     492,   244,   245,   227,    98,    79,    83,    84,    85,     1,
       2,   457,   443,   257,   258,   259,   260,   261,   146,   163,
       6,   147,   204,   322,   205,   446,   344,  -127,   455,  -127,
      11,    12,   345,   110,   452,   346,   200,   210,   464,   111,
     148,    58,   367,   368,   144,   145,   112,   471,   135,   473,
     159,   209,   136,   212,   125,   470,   126,   216,   217,   218,
     219,   220,   221,   222,   234,   486,   480,   127,   230,   231,
     232,   155,   156,   225,   168,   169,   170,   171,    25,    26,
      27,    28,    29,   491,   151,   164,   153,   239,   240,   241,
      36,    37,   160,     1,     2,   183,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    34,    35,   161,    36,    37,
     167,   271,    38,    99,   100,   101,   102,   103,   104,   174,
     278,   175,   176,   281,    55,   118,   119,   120,   121,   319,
       6,   147,    58,   226,   276,   277,   236,   237,   253,   254,
     234,   262,   263,    56,    57,   177,   162,   282,   283,   284,
     285,   286,   193,   184,   192,   146,   123,   194,    39,   201,
     369,    58,   208,    56,    57,   202,   203,   295,    59,    60,
      61,    62,    63,    64,    65,   206,   207,    56,    57,   213,
     318,    58,   214,   223,   224,   242,   326,   247,    59,    60,
      61,    62,    63,    64,    65,    58,   185,   251,   210,   265,
     122,   266,    59,    60,    61,    62,    63,    64,    65,   301,
     338,   302,   303,   304,   305,   306,   307,   351,   267,   308,
     309,   310,   311,   268,    56,    57,   269,   270,   280,   288,
     290,   292,   294,   296,   325,   323,   330,   332,   331,   312,
     313,   335,    58,   178,   179,    56,    57,   334,  -165,    59,
      60,    61,    62,    63,    64,    65,   339,    -9,   336,    56,
     340,   372,   375,    58,   348,   376,   377,   381,   379,   384,
      59,    60,    61,    62,    63,    64,    65,    58,   316,   387,
     385,   390,   395,   398,    59,    60,    61,    62,    63,    64,
      65,   397,   399,   400,   403,   415,   317,   408,   391,   414,
     417,   418,   420,    59,    60,    61,    62,    63,    64,    65,
     353,   354,   355,   356,   357,   358,   359,   360,   422,   361,
     353,   354,   355,   356,   357,   358,   359,   360,   428,   361,
     429,   430,   436,   442,   449,   447,   454,   458,   391,   434,
     353,   354,   355,   356,   357,   358,   359,   360,   362,   361,
     353,   354,   355,   356,   357,   358,   359,   360,   453,   361,
     353,   354,   355,   356,   357,   358,   359,   360,   391,   361,
     353,   354,   355,   356,   357,   358,   359,   360,   409,   361,
     353,   354,   355,   356,   357,   358,   359,   360,   445,   361,
     353,   354,   355,   356,   357,   358,   359,   360,   451,   361,
     353,   354,   355,   356,   357,   358,   359,   360,   469,   361,
     461,   465,   466,   467,   468,   478,   474,   481,   479,   482,
     484,   488,   493,   483,    10,    40,    14,   264,   490,   320,
     196,   115,   229,   173,   256,   233,   243,   343,   246,    50,
     191,   425,   154,   289,   423,   476,   293,   255,   450,   350,
     487,   439,   433
};

static const yytype_uint16 yycheck[] =
{
       1,     2,    74,    55,    76,    77,    78,    25,   167,    29,
      94,   338,   416,   393,   386,    16,    17,    27,    28,    59,
      36,   355,   328,    17,    30,   249,   360,    20,    36,   409,
      56,    87,   412,     0,    59,    36,   330,    77,   344,   263,
     346,   337,   336,    17,     5,    51,    39,   103,    74,    74,
     143,   144,   145,    74,   381,    59,    77,   363,    74,    35,
     464,    59,    82,    59,    53,   445,    74,    69,    76,   375,
      74,   451,   378,    71,   380,    69,    94,    55,    74,    71,
      61,   453,   486,   389,    65,    86,    77,   383,   384,   469,
     417,   397,    77,   399,   400,   429,    57,    58,    53,   479,
     396,   407,    77,    77,    67,   123,    80,   413,    69,    67,
     490,   204,   205,   185,    67,   135,    31,    32,    33,     3,
       4,   448,   428,   216,   217,   218,   219,   220,    51,   213,
      53,    54,   152,   292,   154,   431,    68,    70,   444,    72,
      67,    68,    74,    67,   440,    77,   147,    80,   454,    67,
     151,    53,    54,    55,    84,    85,    67,   463,    70,   465,
      90,   162,    67,   164,    73,   461,    73,   168,   169,   170,
     171,   172,   173,   174,   194,   481,   472,    73,   188,   189,
     190,    37,    38,   184,    99,   100,   101,   102,     6,     7,
       8,     9,    10,   489,    77,   213,    77,   197,   198,   199,
      18,    19,    67,     3,     4,   223,     6,     7,     8,     9,
      10,    11,    12,    13,    14,    15,    16,    67,    18,    19,
      69,    39,    22,    11,    12,    13,    14,    15,    16,    73,
     248,    60,    61,   251,    16,    60,    61,    62,    63,   291,
      53,    54,    53,    54,   244,   245,    67,    68,    54,    55,
     270,    67,    68,    35,    36,    53,    18,   257,   258,   259,
     260,   261,    70,    73,    78,    51,    71,    79,    68,    78,
     342,    53,    68,    35,    36,    70,    78,   278,    60,    61,
      62,    63,    64,    65,    66,    78,    70,    35,    36,    70,
     291,    53,    72,    70,    72,    78,   297,    17,    60,    61,
      62,    63,    64,    65,    66,    53,    77,    80,    80,    75,
      59,    78,    60,    61,    62,    63,    64,    65,    66,    21,
     321,    23,    24,    25,    26,    27,    28,    75,    75,    31,
      32,    33,    34,    75,    35,    36,    75,    70,    69,    74,
      77,    69,    67,    69,    67,    77,    70,    55,    75,    51,
      52,    70,    53,    54,    55,    35,    36,    78,    53,    60,
      61,    62,    63,    64,    65,    66,    78,    74,    74,    35,
      67,    56,    75,    53,    78,    75,    74,    53,    77,    68,
      60,    61,    62,    63,    64,    65,    66,    53,    35,    53,
      55,    74,    20,    70,    60,    61,    62,    63,    64,    65,
      66,    69,    75,    78,    53,    67,    53,    70,    68,    70,
      53,    69,    69,    60,    61,    62,    63,    64,    65,    66,
      40,    41,    42,    43,    44,    45,    46,    47,    80,    49,
      40,    41,    42,    43,    44,    45,    46,    47,    67,    49,
      78,    75,    69,    54,    69,    78,    67,    50,    68,    69,
      40,    41,    42,    43,    44,    45,    46,    47,    68,    49,
      40,    41,    42,    43,    44,    45,    46,    47,    75,    49,
      40,    41,    42,    43,    44,    45,    46,    47,    68,    49,
      40,    41,    42,    43,    44,    45,    46,    47,    68,    49,
      40,    41,    42,    43,    44,    45,    46,    47,    68,    49,
      40,    41,    42,    43,    44,    45,    46,    47,    68,    49,
      40,    41,    42,    43,    44,    45,    46,    47,    68,    49,
      48,    79,    69,    74,    69,    69,    53,    70,    68,    70,
      69,    69,    69,    75,     5,    18,     8,   223,    68,   291,
     142,    57,   186,   104,   213,   193,   202,   326,   207,    18,
     135,   398,    88,   270,   396,   468,   274,   212,   438,   335,
     482,   414,   408
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     3,     4,    82,    83,    88,    53,    86,    86,     0,
      83,    67,    68,    89,    89,     5,    57,    58,    84,    90,
      91,   182,   183,    86,    86,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    18,    19,    22,    68,
      88,   108,   109,   125,   128,   129,   130,   132,   142,   143,
     146,   147,   148,    69,    90,    16,    35,    36,    53,    60,
      61,    62,    63,    64,    65,    66,    87,    96,    98,    99,
     100,   101,   102,   103,   104,    35,   110,   110,   110,    98,
     149,    77,   114,   114,   114,   114,    77,   118,   131,    77,
     111,    86,    55,   150,    71,    90,    67,    67,    67,    11,
      12,    13,    14,    15,    16,   133,   134,   135,   136,   137,
      67,    67,    67,    85,    99,   103,    61,    65,    60,    61,
      62,    63,    59,    71,    95,    73,    73,    73,    36,    74,
      76,    87,    87,    87,    87,    70,    67,    30,    51,   115,
     120,    98,    86,    97,    97,    97,    51,    54,    86,   117,
     119,    77,   131,    77,   118,    37,    38,   112,   113,    97,
      67,    67,    18,   102,   104,   140,   141,    69,   114,   114,
     114,   114,   131,   111,    73,    60,    61,    53,    54,    55,
      92,    93,    94,   104,    73,    77,   106,   107,    74,    74,
      74,   149,    78,    70,    79,   126,    95,   126,   126,   126,
      86,    78,    70,    78,    98,    98,    78,    70,    68,    86,
      80,   139,    86,    70,    72,    85,    86,    86,    86,    86,
      86,    86,    86,    70,    72,    86,    54,    87,   105,   107,
     110,   110,   110,   115,    98,   127,    67,    68,   144,   144,
     144,   144,    78,   119,   126,   126,   112,    17,   121,   123,
     124,    80,   138,    54,    55,   139,   141,   126,   126,   126,
     126,   126,    67,    68,    93,    75,    78,    75,    75,    75,
      70,    39,   145,   146,   151,   152,   144,   144,   104,   124,
      69,   104,   144,   144,   144,   144,   144,   124,    74,   127,
      77,   154,    69,   145,    67,    86,    69,   104,   160,   163,
     164,    21,    23,    24,    25,    26,    27,    28,    31,    32,
      33,    34,    51,    52,   155,   156,    35,    53,    86,    99,
     100,   153,    85,    77,   122,    67,    86,    56,    74,   159,
      70,    75,    55,   116,    78,    70,    74,   165,    86,    78,
      67,    77,    80,   121,    68,    74,    77,   159,    78,   164,
     155,    75,   164,    40,    41,    42,    43,    44,    45,    46,
      47,    49,    68,   161,   167,   173,   165,    54,    55,    87,
     157,   159,    56,   158,   159,    75,    75,    74,   177,    77,
     177,    53,   178,   179,    68,    55,   172,    53,   175,   177,
      74,    68,   168,   173,   159,    20,   166,    69,    70,    75,
      78,   159,   159,    53,   159,    77,   165,   180,    70,    68,
     173,   169,   173,   161,    70,    67,   159,    53,    69,   168,
      69,   162,    80,   167,   159,   158,   159,   159,    67,    78,
      75,   176,   159,   179,    69,   168,    69,   168,   159,   175,
     176,   165,    54,   159,   177,    68,   173,    78,   181,    69,
     162,    68,   173,    75,    67,   159,   168,   165,    50,   170,
     168,    48,   174,   161,   159,    79,    69,    74,    69,    68,
     173,   159,   176,   159,    53,   171,   174,   168,    69,    68,
     173,    70,    70,    75,    69,   168,   159,   171,    69,   176,
      68,   173,   168,    69
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
#line 136 "xi-grammar.y"
    { (yyval.modlist) = (yyvsp[(1) - (1)].modlist); modlist = (yyvsp[(1) - (1)].modlist); }
    break;

  case 3:
#line 140 "xi-grammar.y"
    { 
		  (yyval.modlist) = 0; 
		}
    break;

  case 4:
#line 144 "xi-grammar.y"
    { (yyval.modlist) = new ModuleList(lineno, (yyvsp[(1) - (2)].module), (yyvsp[(2) - (2)].modlist)); }
    break;

  case 5:
#line 148 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 6:
#line 150 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 7:
#line 154 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 8:
#line 156 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 9:
#line 160 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 10:
#line 164 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 11:
#line 166 "xi-grammar.y"
    {
		  char *tmp = new char[strlen((yyvsp[(1) - (3)].strval))+strlen((yyvsp[(3) - (3)].strval))+3];
		  sprintf(tmp,"%s::%s", (yyvsp[(1) - (3)].strval), (yyvsp[(3) - (3)].strval));
		  (yyval.strval) = tmp;
		}
    break;

  case 12:
#line 174 "xi-grammar.y"
    { 
		    (yyval.module) = new Module(lineno, (yyvsp[(2) - (3)].strval), (yyvsp[(3) - (3)].conslist)); 
		}
    break;

  case 13:
#line 178 "xi-grammar.y"
    {  
		    (yyval.module) = new Module(lineno, (yyvsp[(2) - (3)].strval), (yyvsp[(3) - (3)].conslist)); 
		    (yyval.module)->setMain();
		}
    break;

  case 14:
#line 185 "xi-grammar.y"
    { (yyval.conslist) = 0; }
    break;

  case 15:
#line 187 "xi-grammar.y"
    { (yyval.conslist) = (yyvsp[(2) - (4)].conslist); }
    break;

  case 16:
#line 191 "xi-grammar.y"
    { (yyval.conslist) = 0; }
    break;

  case 17:
#line 193 "xi-grammar.y"
    { (yyval.conslist) = new ConstructList(lineno, (yyvsp[(1) - (2)].construct), (yyvsp[(2) - (2)].conslist)); }
    break;

  case 18:
#line 197 "xi-grammar.y"
    { if((yyvsp[(3) - (5)].conslist)) (yyvsp[(3) - (5)].conslist)->setExtern((yyvsp[(1) - (5)].intval)); (yyval.construct) = (yyvsp[(3) - (5)].conslist); }
    break;

  case 19:
#line 199 "xi-grammar.y"
    { (yyvsp[(2) - (2)].module)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].module); }
    break;

  case 20:
#line 201 "xi-grammar.y"
    { (yyvsp[(2) - (2)].member)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].member); }
    break;

  case 21:
#line 203 "xi-grammar.y"
    { (yyvsp[(2) - (3)].message)->setExtern((yyvsp[(1) - (3)].intval)); (yyval.construct) = (yyvsp[(2) - (3)].message); }
    break;

  case 22:
#line 205 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 23:
#line 207 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 24:
#line 209 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 25:
#line 211 "xi-grammar.y"
    { (yyvsp[(2) - (2)].chare)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].chare); }
    break;

  case 26:
#line 213 "xi-grammar.y"
    { (yyvsp[(2) - (2)].templat)->setExtern((yyvsp[(1) - (2)].intval)); (yyval.construct) = (yyvsp[(2) - (2)].templat); }
    break;

  case 27:
#line 215 "xi-grammar.y"
    { (yyval.construct) = NULL; }
    break;

  case 28:
#line 217 "xi-grammar.y"
    { (yyval.construct) = NULL; }
    break;

  case 29:
#line 221 "xi-grammar.y"
    { (yyval.tparam) = new TParamType((yyvsp[(1) - (1)].type)); }
    break;

  case 30:
#line 223 "xi-grammar.y"
    { (yyval.tparam) = new TParamVal((yyvsp[(1) - (1)].strval)); }
    break;

  case 31:
#line 225 "xi-grammar.y"
    { (yyval.tparam) = new TParamVal((yyvsp[(1) - (1)].strval)); }
    break;

  case 32:
#line 229 "xi-grammar.y"
    { (yyval.tparlist) = new TParamList((yyvsp[(1) - (1)].tparam)); }
    break;

  case 33:
#line 231 "xi-grammar.y"
    { (yyval.tparlist) = new TParamList((yyvsp[(1) - (3)].tparam), (yyvsp[(3) - (3)].tparlist)); }
    break;

  case 34:
#line 235 "xi-grammar.y"
    { (yyval.tparlist) = 0; }
    break;

  case 35:
#line 237 "xi-grammar.y"
    { (yyval.tparlist) = (yyvsp[(1) - (1)].tparlist); }
    break;

  case 36:
#line 241 "xi-grammar.y"
    { (yyval.tparlist) = 0; }
    break;

  case 37:
#line 243 "xi-grammar.y"
    { (yyval.tparlist) = (yyvsp[(2) - (3)].tparlist); }
    break;

  case 38:
#line 247 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("int"); }
    break;

  case 39:
#line 249 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long"); }
    break;

  case 40:
#line 251 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("short"); }
    break;

  case 41:
#line 253 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("char"); }
    break;

  case 42:
#line 255 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned int"); }
    break;

  case 43:
#line 257 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long"); }
    break;

  case 44:
#line 259 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long"); }
    break;

  case 45:
#line 261 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned long long"); }
    break;

  case 46:
#line 263 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned short"); }
    break;

  case 47:
#line 265 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("unsigned char"); }
    break;

  case 48:
#line 267 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long long"); }
    break;

  case 49:
#line 269 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("float"); }
    break;

  case 50:
#line 271 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("double"); }
    break;

  case 51:
#line 273 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("long double"); }
    break;

  case 52:
#line 275 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("void"); }
    break;

  case 53:
#line 278 "xi-grammar.y"
    { (yyval.ntype) = new NamedType((yyvsp[(1) - (2)].strval),(yyvsp[(2) - (2)].tparlist)); }
    break;

  case 54:
#line 279 "xi-grammar.y"
    { (yyval.ntype) = new NamedType((yyvsp[(1) - (2)].strval),(yyvsp[(2) - (2)].tparlist)); }
    break;

  case 55:
#line 282 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 56:
#line 284 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ntype); }
    break;

  case 57:
#line 288 "xi-grammar.y"
    { (yyval.ptype) = new PtrType((yyvsp[(1) - (2)].type)); }
    break;

  case 58:
#line 292 "xi-grammar.y"
    { (yyvsp[(1) - (2)].ptype)->indirect(); (yyval.ptype) = (yyvsp[(1) - (2)].ptype); }
    break;

  case 59:
#line 294 "xi-grammar.y"
    { (yyvsp[(1) - (2)].ptype)->indirect(); (yyval.ptype) = (yyvsp[(1) - (2)].ptype); }
    break;

  case 60:
#line 298 "xi-grammar.y"
    { (yyval.ftype) = new FuncType((yyvsp[(1) - (8)].type), (yyvsp[(4) - (8)].strval), (yyvsp[(7) - (8)].plist)); }
    break;

  case 61:
#line 302 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 62:
#line 304 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 63:
#line 306 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 64:
#line 308 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ftype); }
    break;

  case 65:
#line 311 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(2) - (2)].type); }
    break;

  case 66:
#line 313 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (2)].type); }
    break;

  case 67:
#line 317 "xi-grammar.y"
    { (yyval.type) = new ReferenceType((yyvsp[(1) - (2)].type)); }
    break;

  case 68:
#line 319 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].type); }
    break;

  case 69:
#line 323 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 70:
#line 325 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 71:
#line 329 "xi-grammar.y"
    { (yyval.val) = (yyvsp[(2) - (3)].val); }
    break;

  case 72:
#line 333 "xi-grammar.y"
    { (yyval.vallist) = 0; }
    break;

  case 73:
#line 335 "xi-grammar.y"
    { (yyval.vallist) = new ValueList((yyvsp[(1) - (2)].val), (yyvsp[(2) - (2)].vallist)); }
    break;

  case 74:
#line 339 "xi-grammar.y"
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[(2) - (4)].type), (yyvsp[(3) - (4)].strval), (yyvsp[(4) - (4)].vallist)); }
    break;

  case 75:
#line 343 "xi-grammar.y"
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[(3) - (5)].type), (yyvsp[(5) - (5)].strval), 0, 1); }
    break;

  case 76:
#line 347 "xi-grammar.y"
    { (yyval.intval) = 0;}
    break;

  case 77:
#line 349 "xi-grammar.y"
    { (yyval.intval) = 0;}
    break;

  case 78:
#line 353 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 79:
#line 355 "xi-grammar.y"
    { 
		  /*
		  printf("Warning: Message attributes are being phased out.\n");
		  printf("Warning: Please remove them from interface files.\n");
		  */
		  (yyval.intval) = (yyvsp[(2) - (3)].intval); 
		}
    break;

  case 80:
#line 365 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (1)].intval); }
    break;

  case 81:
#line 367 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (3)].intval) | (yyvsp[(3) - (3)].intval); }
    break;

  case 82:
#line 371 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 83:
#line 373 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 84:
#line 377 "xi-grammar.y"
    { (yyval.cattr) = 0; }
    break;

  case 85:
#line 379 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(2) - (3)].cattr); }
    break;

  case 86:
#line 383 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (1)].cattr); }
    break;

  case 87:
#line 385 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (3)].cattr) | (yyvsp[(3) - (3)].cattr); }
    break;

  case 88:
#line 389 "xi-grammar.y"
    { python_doc = NULL; (yyval.intval) = 0; }
    break;

  case 89:
#line 391 "xi-grammar.y"
    { python_doc = (yyvsp[(1) - (1)].strval); (yyval.intval) = 0; }
    break;

  case 90:
#line 395 "xi-grammar.y"
    { (yyval.cattr) = Chare::CPYTHON; }
    break;

  case 91:
#line 399 "xi-grammar.y"
    { (yyval.cattr) = 0; }
    break;

  case 92:
#line 401 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(2) - (3)].cattr); }
    break;

  case 93:
#line 405 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (1)].cattr); }
    break;

  case 94:
#line 407 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[(1) - (3)].cattr) | (yyvsp[(3) - (3)].cattr); }
    break;

  case 95:
#line 411 "xi-grammar.y"
    { (yyval.cattr) = Chare::CMIGRATABLE; }
    break;

  case 96:
#line 413 "xi-grammar.y"
    { (yyval.cattr) = Chare::CPYTHON; }
    break;

  case 97:
#line 417 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 98:
#line 419 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 99:
#line 422 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 100:
#line 424 "xi-grammar.y"
    { (yyval.intval) = 1; }
    break;

  case 101:
#line 427 "xi-grammar.y"
    { (yyval.mv) = new MsgVar((yyvsp[(2) - (5)].type), (yyvsp[(3) - (5)].strval), (yyvsp[(1) - (5)].intval), (yyvsp[(4) - (5)].intval)); }
    break;

  case 102:
#line 431 "xi-grammar.y"
    { (yyval.mvlist) = new MsgVarList((yyvsp[(1) - (1)].mv)); }
    break;

  case 103:
#line 433 "xi-grammar.y"
    { (yyval.mvlist) = new MsgVarList((yyvsp[(1) - (2)].mv), (yyvsp[(2) - (2)].mvlist)); }
    break;

  case 104:
#line 437 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, (yyvsp[(3) - (3)].ntype)); }
    break;

  case 105:
#line 439 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, (yyvsp[(3) - (6)].ntype), (yyvsp[(5) - (6)].mvlist)); }
    break;

  case 106:
#line 443 "xi-grammar.y"
    { (yyval.typelist) = 0; }
    break;

  case 107:
#line 445 "xi-grammar.y"
    { (yyval.typelist) = (yyvsp[(2) - (2)].typelist); }
    break;

  case 108:
#line 449 "xi-grammar.y"
    { (yyval.typelist) = new TypeList((yyvsp[(1) - (1)].ntype)); }
    break;

  case 109:
#line 451 "xi-grammar.y"
    { (yyval.typelist) = new TypeList((yyvsp[(1) - (3)].ntype), (yyvsp[(3) - (3)].typelist)); }
    break;

  case 110:
#line 455 "xi-grammar.y"
    { (yyval.chare) = new Chare(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 111:
#line 457 "xi-grammar.y"
    { (yyval.chare) = new MainChare(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 112:
#line 461 "xi-grammar.y"
    { (yyval.chare) = new Group(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 113:
#line 465 "xi-grammar.y"
    { (yyval.chare) = new NodeGroup(lineno, (yyvsp[(2) - (5)].cattr), (yyvsp[(3) - (5)].ntype), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 114:
#line 469 "xi-grammar.y"
    {/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",(yyvsp[(2) - (4)].strval));
			(yyval.ntype) = new NamedType(buf); 
		}
    break;

  case 115:
#line 475 "xi-grammar.y"
    { (yyval.ntype) = new NamedType((yyvsp[(2) - (3)].strval)); }
    break;

  case 116:
#line 479 "xi-grammar.y"
    {  (yyval.chare) = new Array(lineno, (yyvsp[(2) - (6)].cattr), (yyvsp[(3) - (6)].ntype), (yyvsp[(4) - (6)].ntype), (yyvsp[(5) - (6)].typelist), (yyvsp[(6) - (6)].mbrlist)); }
    break;

  case 117:
#line 481 "xi-grammar.y"
    {  (yyval.chare) = new Array(lineno, (yyvsp[(3) - (6)].cattr), (yyvsp[(2) - (6)].ntype), (yyvsp[(4) - (6)].ntype), (yyvsp[(5) - (6)].typelist), (yyvsp[(6) - (6)].mbrlist)); }
    break;

  case 118:
#line 485 "xi-grammar.y"
    { (yyval.chare) = new Chare(lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist));}
    break;

  case 119:
#line 487 "xi-grammar.y"
    { (yyval.chare) = new MainChare(lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 120:
#line 491 "xi-grammar.y"
    { (yyval.chare) = new Group(lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 121:
#line 495 "xi-grammar.y"
    { (yyval.chare) = new NodeGroup( lineno, (yyvsp[(2) - (5)].cattr), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 122:
#line 499 "xi-grammar.y"
    { (yyval.chare) = new Array( lineno, 0, (yyvsp[(2) - (5)].ntype), new NamedType((yyvsp[(3) - (5)].strval)), (yyvsp[(4) - (5)].typelist), (yyvsp[(5) - (5)].mbrlist)); }
    break;

  case 123:
#line 503 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[(3) - (4)].strval))); }
    break;

  case 124:
#line 505 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[(3) - (7)].strval)), (yyvsp[(5) - (7)].mvlist)); }
    break;

  case 125:
#line 509 "xi-grammar.y"
    { (yyval.type) = 0; }
    break;

  case 126:
#line 511 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(2) - (2)].type); }
    break;

  case 127:
#line 515 "xi-grammar.y"
    { (yyval.strval) = 0; }
    break;

  case 128:
#line 517 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(2) - (2)].strval); }
    break;

  case 129:
#line 519 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(2) - (2)].strval); }
    break;

  case 130:
#line 523 "xi-grammar.y"
    { (yyval.tvar) = new TType(new NamedType((yyvsp[(2) - (3)].strval)), (yyvsp[(3) - (3)].type)); }
    break;

  case 131:
#line 525 "xi-grammar.y"
    { (yyval.tvar) = new TFunc((yyvsp[(1) - (2)].ftype), (yyvsp[(2) - (2)].strval)); }
    break;

  case 132:
#line 527 "xi-grammar.y"
    { (yyval.tvar) = new TName((yyvsp[(1) - (3)].type), (yyvsp[(2) - (3)].strval), (yyvsp[(3) - (3)].strval)); }
    break;

  case 133:
#line 531 "xi-grammar.y"
    { (yyval.tvarlist) = new TVarList((yyvsp[(1) - (1)].tvar)); }
    break;

  case 134:
#line 533 "xi-grammar.y"
    { (yyval.tvarlist) = new TVarList((yyvsp[(1) - (3)].tvar), (yyvsp[(3) - (3)].tvarlist)); }
    break;

  case 135:
#line 537 "xi-grammar.y"
    { (yyval.tvarlist) = (yyvsp[(3) - (4)].tvarlist); }
    break;

  case 136:
#line 541 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 137:
#line 543 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 138:
#line 545 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 139:
#line 547 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].chare)); (yyvsp[(2) - (2)].chare)->setTemplate((yyval.templat)); }
    break;

  case 140:
#line 549 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[(1) - (2)].tvarlist), (yyvsp[(2) - (2)].message)); (yyvsp[(2) - (2)].message)->setTemplate((yyval.templat)); }
    break;

  case 141:
#line 553 "xi-grammar.y"
    { (yyval.mbrlist) = 0; }
    break;

  case 142:
#line 555 "xi-grammar.y"
    { (yyval.mbrlist) = (yyvsp[(2) - (4)].mbrlist); }
    break;

  case 143:
#line 559 "xi-grammar.y"
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

  case 144:
#line 578 "xi-grammar.y"
    { (yyval.mbrlist) = new MemberList((yyvsp[(1) - (2)].member), (yyvsp[(2) - (2)].mbrlist)); }
    break;

  case 145:
#line 582 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (2)].readonly); }
    break;

  case 146:
#line 584 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (2)].readonly); }
    break;

  case 148:
#line 587 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (2)].member); }
    break;

  case 149:
#line 589 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(2) - (3)].pupable); }
    break;

  case 150:
#line 591 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(2) - (3)].includeFile); }
    break;

  case 151:
#line 593 "xi-grammar.y"
    { (yyval.member) = new ClassDeclaration(lineno,(yyvsp[(2) - (3)].strval)); }
    break;

  case 152:
#line 597 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 1); }
    break;

  case 153:
#line 599 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 1); }
    break;

  case 154:
#line 601 "xi-grammar.y"
    { printf("Warning: deprecated use of initcall. Use initnode or initproc instead.\n"); 
		  (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 1); }
    break;

  case 155:
#line 604 "xi-grammar.y"
    { printf("Warning: deprecated use of initcall. Use initnode or initproc instead.\n");
		  (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 1); }
    break;

  case 156:
#line 609 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (3)].strval), 0); }
    break;

  case 157:
#line 611 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[(3) - (6)].strval), 0); }
    break;

  case 158:
#line 615 "xi-grammar.y"
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[(1) - (1)].ntype),0); }
    break;

  case 159:
#line 617 "xi-grammar.y"
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[(1) - (3)].ntype),(yyvsp[(3) - (3)].pupable)); }
    break;

  case 160:
#line 620 "xi-grammar.y"
    { (yyval.includeFile) = new IncludeFile(lineno,(yyvsp[(1) - (1)].strval)); }
    break;

  case 161:
#line 624 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (2)].entry); }
    break;

  case 162:
#line 626 "xi-grammar.y"
    { (yyval.member) = (yyvsp[(1) - (1)].member); }
    break;

  case 163:
#line 630 "xi-grammar.y"
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

  case 164:
#line 641 "xi-grammar.y"
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

  case 165:
#line 654 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("void"); }
    break;

  case 166:
#line 656 "xi-grammar.y"
    { (yyval.type) = (yyvsp[(1) - (1)].ptype); }
    break;

  case 167:
#line 660 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 168:
#line 662 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(2) - (3)].intval); }
    break;

  case 169:
#line 666 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (1)].intval); }
    break;

  case 170:
#line 668 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[(1) - (3)].intval) | (yyvsp[(3) - (3)].intval); }
    break;

  case 171:
#line 672 "xi-grammar.y"
    { (yyval.intval) = STHREADED; }
    break;

  case 172:
#line 674 "xi-grammar.y"
    { (yyval.intval) = SSYNC; }
    break;

  case 173:
#line 676 "xi-grammar.y"
    { (yyval.intval) = SIGET; }
    break;

  case 174:
#line 678 "xi-grammar.y"
    { (yyval.intval) = SLOCKED; }
    break;

  case 175:
#line 680 "xi-grammar.y"
    { (yyval.intval) = SCREATEHERE; }
    break;

  case 176:
#line 682 "xi-grammar.y"
    { (yyval.intval) = SCREATEHOME; }
    break;

  case 177:
#line 684 "xi-grammar.y"
    { (yyval.intval) = SNOKEEP; }
    break;

  case 178:
#line 686 "xi-grammar.y"
    { (yyval.intval) = SNOTRACE; }
    break;

  case 179:
#line 688 "xi-grammar.y"
    { (yyval.intval) = SIMMEDIATE; }
    break;

  case 180:
#line 690 "xi-grammar.y"
    { (yyval.intval) = SSKIPSCHED; }
    break;

  case 181:
#line 692 "xi-grammar.y"
    { (yyval.intval) = SINLINE; }
    break;

  case 182:
#line 694 "xi-grammar.y"
    { (yyval.intval) = SLOCAL; }
    break;

  case 183:
#line 696 "xi-grammar.y"
    { (yyval.intval) = SPYTHON; }
    break;

  case 184:
#line 700 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 185:
#line 702 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 186:
#line 704 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(1) - (1)].strval)); }
    break;

  case 187:
#line 708 "xi-grammar.y"
    { (yyval.strval) = ""; }
    break;

  case 188:
#line 710 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 189:
#line 712 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[(1) - (3)].strval))+strlen((yyvsp[(3) - (3)].strval))+3];
			sprintf(tmp,"%s, %s", (yyvsp[(1) - (3)].strval), (yyvsp[(3) - (3)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 190:
#line 720 "xi-grammar.y"
    { (yyval.strval) = ""; }
    break;

  case 191:
#line 722 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 192:
#line 724 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s[%s]%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 193:
#line 730 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s{%s}%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 194:
#line 736 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(1) - (5)].strval))+strlen((yyvsp[(3) - (5)].strval))+strlen((yyvsp[(5) - (5)].strval))+3];
			sprintf(tmp,"%s(%s)%s", (yyvsp[(1) - (5)].strval), (yyvsp[(3) - (5)].strval), (yyvsp[(5) - (5)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 195:
#line 742 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[(2) - (4)].strval))+strlen((yyvsp[(4) - (4)].strval))+3];
			sprintf(tmp,"(%s)%s", (yyvsp[(2) - (4)].strval), (yyvsp[(4) - (4)].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 196:
#line 750 "xi-grammar.y"
    {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			(yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].type),(yyvsp[(2) - (3)].strval));
		}
    break;

  case 197:
#line 757 "xi-grammar.y"
    { 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			(yyval.intval) = 0;
		}
    break;

  case 198:
#line 765 "xi-grammar.y"
    { 
			in_braces=0;
			(yyval.intval) = 0;
		}
    break;

  case 199:
#line 772 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (1)].type));}
    break;

  case 200:
#line 774 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].type),(yyvsp[(2) - (3)].strval)); (yyval.pname)->setConditional((yyvsp[(3) - (3)].intval)); }
    break;

  case 201:
#line 776 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (4)].type),(yyvsp[(2) - (4)].strval),0,(yyvsp[(4) - (4)].val));}
    break;

  case 202:
#line 778 "xi-grammar.y"
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[(1) - (3)].pname)->getType(), (yyvsp[(1) - (3)].pname)->getName() ,(yyvsp[(2) - (3)].strval));
		}
    break;

  case 203:
#line 785 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (1)].pname)); }
    break;

  case 204:
#line 787 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[(1) - (3)].pname),(yyvsp[(3) - (3)].plist)); }
    break;

  case 205:
#line 791 "xi-grammar.y"
    { (yyval.plist) = (yyvsp[(2) - (3)].plist); }
    break;

  case 206:
#line 793 "xi-grammar.y"
    { (yyval.plist) = 0; }
    break;

  case 207:
#line 797 "xi-grammar.y"
    { (yyval.val) = 0; }
    break;

  case 208:
#line 799 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[(3) - (3)].strval)); }
    break;

  case 209:
#line 803 "xi-grammar.y"
    { (yyval.sc) = 0; }
    break;

  case 210:
#line 805 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSDAGENTRY, (yyvsp[(1) - (1)].sc)); }
    break;

  case 211:
#line 807 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSDAGENTRY, (yyvsp[(2) - (3)].sc)); }
    break;

  case 212:
#line 811 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSLIST, (yyvsp[(1) - (1)].sc)); }
    break;

  case 213:
#line 813 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSLIST, (yyvsp[(1) - (2)].sc), (yyvsp[(2) - (2)].sc));  }
    break;

  case 214:
#line 817 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SOLIST, (yyvsp[(1) - (1)].sc)); }
    break;

  case 215:
#line 819 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SOLIST, (yyvsp[(1) - (2)].sc), (yyvsp[(2) - (2)].sc)); }
    break;

  case 216:
#line 823 "xi-grammar.y"
    { (yyval.sc) = 0; }
    break;

  case 217:
#line 825 "xi-grammar.y"
    { (yyval.sc) = (yyvsp[(3) - (4)].sc); }
    break;

  case 218:
#line 829 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SPUBLISHES, new SdagConstruct(SIDENT, (yyvsp[(1) - (1)].strval))); }
    break;

  case 219:
#line 831 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SPUBLISHES, new SdagConstruct(SIDENT, (yyvsp[(1) - (3)].strval)), (yyvsp[(3) - (3)].sc));  }
    break;

  case 220:
#line 835 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[(1) - (1)].strval); }
    break;

  case 221:
#line 837 "xi-grammar.y"
    { (yyval.strval) = 0; }
    break;

  case 222:
#line 841 "xi-grammar.y"
    { RemoveSdagComments((yyvsp[(4) - (6)].strval));
		   (yyval.sc) = new SdagConstruct(SATOMIC, new XStr((yyvsp[(4) - (6)].strval)), (yyvsp[(6) - (6)].sc), 0,0,0,0, 0 ); 
		   if ((yyvsp[(2) - (6)].strval)) { (yyvsp[(2) - (6)].strval)[strlen((yyvsp[(2) - (6)].strval))-1]=0; (yyval.sc)->traceName = new XStr((yyvsp[(2) - (6)].strval)+1); }
		 }
    break;

  case 223:
#line 846 "xi-grammar.y"
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

  case 224:
#line 860 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SWHEN, 0, 0,0,0,0,0,(yyvsp[(2) - (4)].entrylist)); }
    break;

  case 225:
#line 862 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SWHEN, 0, 0, 0,0,0, (yyvsp[(3) - (3)].sc), (yyvsp[(2) - (3)].entrylist)); }
    break;

  case 226:
#line 864 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SWHEN, 0, 0, 0,0,0, (yyvsp[(4) - (5)].sc), (yyvsp[(2) - (5)].entrylist)); }
    break;

  case 227:
#line 866 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SOVERLAP,0, 0,0,0,0,(yyvsp[(3) - (4)].sc), 0); }
    break;

  case 228:
#line 868 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFOR, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (11)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(5) - (11)].strval)),
		             new SdagConstruct(SINT_EXPR, (yyvsp[(7) - (11)].strval)), 0, (yyvsp[(10) - (11)].sc), 0); }
    break;

  case 229:
#line 871 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFOR, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (9)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(5) - (9)].strval)), 
		         new SdagConstruct(SINT_EXPR, (yyvsp[(7) - (9)].strval)), 0, (yyvsp[(9) - (9)].sc), 0); }
    break;

  case 230:
#line 874 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFORALL, 0, new SdagConstruct(SIDENT, (yyvsp[(3) - (12)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(6) - (12)].strval)), 
		             new SdagConstruct(SINT_EXPR, (yyvsp[(8) - (12)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(10) - (12)].strval)), (yyvsp[(12) - (12)].sc), 0); }
    break;

  case 231:
#line 877 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFORALL, 0, new SdagConstruct(SIDENT, (yyvsp[(3) - (14)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(6) - (14)].strval)), 
		                 new SdagConstruct(SINT_EXPR, (yyvsp[(8) - (14)].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[(10) - (14)].strval)), (yyvsp[(13) - (14)].sc), 0); }
    break;

  case 232:
#line 880 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (6)].strval)), (yyvsp[(6) - (6)].sc),0,0,(yyvsp[(5) - (6)].sc),0); }
    break;

  case 233:
#line 882 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (8)].strval)), (yyvsp[(8) - (8)].sc),0,0,(yyvsp[(6) - (8)].sc),0); }
    break;

  case 234:
#line 884 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (5)].strval)), 0,0,0,(yyvsp[(5) - (5)].sc),0); }
    break;

  case 235:
#line 886 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SWHILE, 0, new SdagConstruct(SINT_EXPR, (yyvsp[(3) - (7)].strval)), 0,0,0,(yyvsp[(6) - (7)].sc),0); }
    break;

  case 236:
#line 888 "xi-grammar.y"
    { (yyval.sc) = (yyvsp[(2) - (3)].sc); }
    break;

  case 237:
#line 890 "xi-grammar.y"
    { RemoveSdagComments((yyvsp[(2) - (3)].strval));
                   (yyval.sc) = new SdagConstruct(SATOMIC, new XStr((yyvsp[(2) - (3)].strval)), NULL, 0,0,0,0, 0 );
                 }
    break;

  case 238:
#line 896 "xi-grammar.y"
    { (yyval.sc) = 0; }
    break;

  case 239:
#line 898 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SELSE, 0,0,0,0,0, (yyvsp[(2) - (2)].sc),0); }
    break;

  case 240:
#line 900 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SELSE, 0,0,0,0,0, (yyvsp[(3) - (4)].sc),0); }
    break;

  case 241:
#line 903 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFORWARD, new SdagConstruct(SIDENT, (yyvsp[(1) - (1)].strval))); }
    break;

  case 242:
#line 905 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFORWARD, new SdagConstruct(SIDENT, (yyvsp[(1) - (3)].strval)), (yyvsp[(3) - (3)].sc));  }
    break;

  case 243:
#line 909 "xi-grammar.y"
    { in_int_expr = 0; (yyval.intval) = 0; }
    break;

  case 244:
#line 913 "xi-grammar.y"
    { in_int_expr = 1; (yyval.intval) = 0; }
    break;

  case 245:
#line 917 "xi-grammar.y"
    { 
		  if ((yyvsp[(2) - (2)].plist) != 0)
		     (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[(1) - (2)].strval), (yyvsp[(2) - (2)].plist), 0, 0, 0, 0); 
		  else
		     (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[(1) - (2)].strval), 
				new ParamList(new Parameter(0, new BuiltinType("void"))), 0, 0, 0, 0); 
		}
    break;

  case 246:
#line 925 "xi-grammar.y"
    { if ((yyvsp[(5) - (5)].plist) != 0)
		    (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[(1) - (5)].strval), (yyvsp[(5) - (5)].plist), 0, 0, (yyvsp[(3) - (5)].strval), 0); 
		  else
		    (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[(1) - (5)].strval), new ParamList(new Parameter(0, new BuiltinType("void"))), 0, 0, (yyvsp[(3) - (5)].strval), 0); 
		}
    break;

  case 247:
#line 933 "xi-grammar.y"
    { (yyval.entrylist) = new EntryList((yyvsp[(1) - (1)].entry)); }
    break;

  case 248:
#line 935 "xi-grammar.y"
    { (yyval.entrylist) = new EntryList((yyvsp[(1) - (3)].entry),(yyvsp[(3) - (3)].entrylist)); }
    break;

  case 249:
#line 939 "xi-grammar.y"
    { in_bracket=1; }
    break;

  case 250:
#line 942 "xi-grammar.y"
    { in_bracket=0; }
    break;

  case 251:
#line 946 "xi-grammar.y"
    { if (!macroDefined((yyvsp[(2) - (2)].strval), 1)) in_comment = 1; }
    break;

  case 252:
#line 950 "xi-grammar.y"
    { if (!macroDefined((yyvsp[(2) - (2)].strval), 0)) in_comment = 1; }
    break;


/* Line 1267 of yacc.c.  */
#line 3352 "y.tab.c"
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


#line 953 "xi-grammar.y"

void yyerror(const char *mesg)
{
  cout << cur_file<<":"<<lineno<<": Charmxi syntax error> " << mesg << endl;
  // return 0;
}

