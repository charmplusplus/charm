/* A Bison parser, made by GNU Bison 2.1.  */

/* Skeleton parser for Yacc-like parsing with Bison,
   Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003, 2004, 2005 Free Software Foundation, Inc.

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

/* As a special exception, when this file is copied by Bison into a
   Bison output file, you may use that output file without restriction.
   This special exception was added by the Free Software Foundation
   in version 1.24 of Bison.  */

/* Written by Richard Stallman by simplifying the original so called
   ``semantic'' parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "2.1"

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

#if ! defined (YYSTYPE) && ! defined (YYSTYPE_IS_DECLARED)
#line 17 "xi-grammar.y"
typedef union YYSTYPE {
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
} YYSTYPE;
/* Line 196 of yacc.c.  */
#line 264 "y.tab.c"
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 219 of yacc.c.  */
#line 276 "y.tab.c"

#if ! defined (YYSIZE_T) && defined (__SIZE_TYPE__)
# define YYSIZE_T __SIZE_TYPE__
#endif
#if ! defined (YYSIZE_T) && defined (size_t)
# define YYSIZE_T size_t
#endif
#if ! defined (YYSIZE_T) && (defined (__STDC__) || defined (__cplusplus))
# include <stddef.h> /* INFRINGES ON USER NAME SPACE */
# define YYSIZE_T size_t
#endif
#if ! defined (YYSIZE_T)
# define YYSIZE_T unsigned int
#endif

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

#if ! defined (yyoverflow) || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if defined (__STDC__) || defined (__cplusplus)
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#     define YYINCLUDED_STDLIB_H
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's `empty if-body' warning. */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2005 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM ((YYSIZE_T) -1)
#  endif
#  ifdef __cplusplus
extern "C" {
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if (! defined (malloc) && ! defined (YYINCLUDED_STDLIB_H) \
	&& (defined (__STDC__) || defined (__cplusplus)))
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if (! defined (free) && ! defined (YYINCLUDED_STDLIB_H) \
	&& (defined (__STDC__) || defined (__cplusplus)))
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifdef __cplusplus
}
#  endif
# endif
#endif /* ! defined (yyoverflow) || YYERROR_VERBOSE */


#if (! defined (yyoverflow) \
     && (! defined (__cplusplus) \
	 || (defined (YYSTYPE_IS_TRIVIAL) && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  short int yyss;
  YYSTYPE yyvs;
  };

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (short int) + sizeof (YYSTYPE))			\
      + YYSTACK_GAP_MAXIMUM)

/* Copy COUNT objects from FROM to TO.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined (__GNUC__) && 1 < __GNUC__
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
      while (0)
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
    while (0)

#endif

#if defined (__STDC__) || defined (__cplusplus)
   typedef signed char yysigned_char;
#else
   typedef short int yysigned_char;
#endif

/* YYFINAL -- State number of the termination state. */
#define YYFINAL  9
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   555

/* YYNTOKENS -- Number of terminals. */
#define YYNTOKENS  79
/* YYNNTS -- Number of nonterminals. */
#define YYNNTS  101
/* YYNRULES -- Number of rules. */
#define YYNRULES  247
/* YYNRULES -- Number of states. */
#define YYNSTATES  488

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   319

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const unsigned char yytranslate[] =
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
static const unsigned short int yyprhs[] =
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
     777,   783,   791,   795,   796,   799,   804,   806,   810,   812,
     814,   817,   823,   825,   829,   831,   833,   836
};

/* YYRHS -- A `-1'-separated list of the rules' RHS. */
static const short int yyrhs[] =
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
      85,    73,   108,    74,    -1,    85,    -1,    85,    69,   145,
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
      -1,    45,   171,    65,    -1,    -1,    47,   169,    -1,    47,
      67,   164,    68,    -1,    52,    -1,    52,    69,   171,    -1,
      74,    -1,    73,    -1,    52,   161,    -1,    52,   176,   155,
     177,   161,    -1,   174,    -1,   174,    69,   175,    -1,    76,
      -1,    77,    -1,    56,    84,    -1,    57,    84,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const unsigned short int yyrline[] =
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
     588,   591,   596,   598,   602,   604,   608,   612,   614,   618,
     629,   642,   644,   649,   650,   654,   656,   660,   662,   664,
     666,   668,   670,   672,   674,   676,   678,   680,   682,   684,
     688,   690,   692,   697,   698,   700,   709,   710,   712,   718,
     724,   730,   738,   745,   753,   760,   762,   764,   766,   773,
     775,   779,   781,   786,   787,   792,   793,   795,   799,   801,
     805,   807,   812,   813,   817,   819,   823,   826,   829,   834,
     848,   850,   852,   854,   856,   859,   862,   865,   868,   870,
     872,   874,   876,   881,   882,   884,   887,   889,   893,   897,
     901,   909,   917,   919,   923,   926,   930,   934
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals. */
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
static const unsigned short int yytoknum[] =
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
static const unsigned char yyr1[] =
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
     169,   169,   169,   170,   170,   170,   171,   171,   172,   173,
     174,   174,   175,   175,   176,   177,   178,   179
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const unsigned char yyr2[] =
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
       5,     7,     3,     0,     2,     4,     1,     3,     1,     1,
       2,     5,     1,     3,     1,     1,     2,     2
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const unsigned char yydefact[] =
{
       3,     0,     0,     0,     2,     3,     9,     0,     0,     1,
       4,    14,     5,    12,    13,     6,     0,     0,     0,     0,
       5,    27,    28,   246,   247,     0,    76,    76,    76,     0,
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
       0,     0,     0,     0,     0,   217,     0,     0,     0,     0,
     160,   206,   203,   181,   180,   182,   197,     0,   184,     0,
       0,   186,   201,   239,   186,     0,   186,     0,   242,     0,
       0,   216,     0,   236,     0,   186,     0,     0,   208,     0,
     205,   186,   183,   186,   186,   191,     0,     0,     0,   244,
     240,   186,     0,     0,   221,     0,   210,   193,   186,     0,
     232,     0,     0,   207,   209,     0,   159,   189,   185,   190,
     188,   186,     0,   238,     0,     0,   243,   220,     0,   223,
     211,     0,   237,     0,     0,   204,     0,   186,     0,   230,
     245,     0,   222,   194,   212,     0,   233,     0,   186,     0,
       0,   241,     0,   218,     0,     0,   228,   186,     0,   186,
     231,     0,   233,     0,   234,     0,     0,     0,   214,     0,
     229,     0,   219,     0,   225,   186,     0,   213,   235,     0,
       0,   215,   224,     0,     0,   226,     0,   227
};

/* YYDEFGOTO[NTERM-NUM]. */
static const short int yydefgoto[] =
{
      -1,     3,     4,    18,   113,   141,    66,     5,    13,    19,
      20,   180,   181,   182,   124,    67,   235,    68,    69,    70,
      71,    72,    73,   248,   229,   186,   187,    41,    42,    76,
      90,   157,   158,    82,   139,   332,   149,    87,   150,   140,
     249,   250,    43,   196,   236,    44,    45,    46,    88,    47,
     105,   106,   107,   108,   109,   252,   211,   165,   166,    48,
      49,   239,   272,   273,    51,    52,    80,    93,   274,   275,
     321,   291,   314,   315,   366,   369,   328,   298,   408,   444,
     299,   300,   336,   390,   360,   387,   405,   453,   469,   382,
     388,   456,   384,   424,   374,   378,   379,   401,   441,    21,
      22
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -401
static const short int yypact[] =
{
      68,    13,    13,   101,  -401,    68,  -401,    54,    54,  -401,
    -401,  -401,    47,  -401,  -401,  -401,    13,    13,   210,    37,
      47,  -401,  -401,  -401,  -401,   220,   104,   104,   104,    93,
      83,    83,    83,    83,    86,    90,    13,   134,    79,    47,
    -401,   129,   132,   133,  -401,  -401,  -401,  -401,   140,  -401,
    -401,   135,   136,   137,  -401,   348,  -401,   329,  -401,  -401,
      31,  -401,  -401,  -401,  -401,   119,    30,  -401,  -401,   121,
     127,   143,  -401,    -6,    93,  -401,    93,    93,    93,    44,
     139,   -20,    13,    13,    13,    13,   -16,   130,   153,    87,
      13,   147,  -401,   167,   239,   165,  -401,  -401,  -401,    83,
      83,    83,    83,   130,    90,  -401,  -401,  -401,  -401,  -401,
    -401,  -401,  -401,  -401,   162,   -28,  -401,  -401,  -401,    73,
    -401,  -401,   169,   316,  -401,  -401,  -401,  -401,  -401,   168,
    -401,   -33,   -35,   -11,    11,    93,  -401,  -401,  -401,   164,
     173,   174,   177,   177,   177,   177,  -401,    13,   171,   176,
     185,    95,    13,   155,    13,  -401,  -401,   186,   195,   179,
    -401,  -401,    13,    -3,    13,   196,   199,   137,    13,    13,
      13,    13,    13,    13,    13,  -401,  -401,   214,  -401,  -401,
     202,  -401,   204,  -401,    13,   112,   200,  -401,   104,   104,
     104,  -401,  -401,   -20,  -401,    13,    72,    72,    72,    72,
     208,  -401,   155,  -401,   177,   177,  -401,    87,   329,   209,
     128,  -401,   215,   239,  -401,  -401,   177,   177,   177,   177,
     177,    76,  -401,  -401,   316,  -401,   212,  -401,   222,   217,
    -401,   218,   221,   232,  -401,   238,  -401,  -401,   251,  -401,
    -401,  -401,  -401,  -401,  -401,    72,    72,  -401,    13,   329,
     241,   329,  -401,  -401,  -401,  -401,  -401,    72,    72,    72,
      72,    72,  -401,   329,  -401,   237,  -401,  -401,  -401,  -401,
      13,   235,   244,  -401,   251,   250,  -401,  -401,   240,  -401,
    -401,  -401,  -401,  -401,  -401,  -401,  -401,   249,   329,  -401,
     315,   361,   137,  -401,  -401,   242,   253,    13,   -32,   252,
     262,  -401,  -401,  -401,  -401,  -401,  -401,  -401,  -401,  -401,
    -401,  -401,   269,  -401,   247,   256,   274,   254,   276,   121,
    -401,    13,  -401,   278,  -401,    19,    15,   -32,   283,   329,
    -401,  -401,  -401,  -401,   315,   270,   397,   276,  -401,  -401,
      74,   -32,   307,   -32,   293,  -401,  -401,  -401,  -401,   297,
     299,   308,   299,   321,   318,   332,   331,   299,   323,   313,
    -401,  -401,   375,  -401,  -401,   222,  -401,   330,   328,   325,
     324,   -32,  -401,  -401,   -32,   350,   -32,    38,   334,   408,
     313,  -401,   337,   336,   349,   -32,   363,   366,   313,   338,
     397,   -32,   307,   -32,   -32,  -401,   352,   341,   370,  -401,
    -401,   -32,   321,   387,  -401,   378,   313,  -401,   -32,   331,
    -401,   370,   276,  -401,  -401,   404,  -401,  -401,  -401,  -401,
    -401,   -32,   299,  -401,   426,   381,  -401,  -401,   391,  -401,
    -401,   392,  -401,   437,   388,  -401,   396,   -32,   313,  -401,
    -401,   276,  -401,  -401,   414,   313,   439,   337,   -32,   407,
     416,  -401,   415,  -401,   419,   455,  -401,   -32,   370,   -32,
    -401,   438,   439,   313,  -401,   421,   466,   422,   423,   428,
    -401,   445,  -401,   313,  -401,   -32,   438,  -401,  -401,   447,
     370,  -401,  -401,   484,   313,  -401,   448,  -401
};

/* YYPGOTO[NTERM-NUM].  */
static const short int yypgoto[] =
{
    -401,  -401,   512,  -401,  -162,    -1,   -27,   500,   511,     1,
    -401,  -401,   296,  -401,   380,  -401,   -65,  -401,   -52,   243,
    -401,   -88,   474,   -21,  -401,  -401,   351,  -401,  -401,   -14,
     431,   333,  -401,    85,   343,  -401,  -401,   450,   339,  -401,
    -401,  -217,  -401,    -9,   272,  -401,  -401,  -401,   -43,  -401,
    -401,  -401,  -401,  -401,  -401,  -401,   327,  -401,   335,  -401,
    -401,    -8,   271,   525,  -401,  -401,   409,  -401,  -401,  -401,
    -401,  -401,   213,  -401,  -401,   154,  -315,  -401,   102,  -401,
    -401,  -272,  -329,  -401,   160,  -364,  -401,  -401,    77,  -401,
    -326,    92,   146,  -400,  -330,  -401,   150,  -401,  -401,  -401,
    -401
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -162
static const short int yytable[] =
{
       7,     8,    79,   114,    74,   215,   163,   128,   362,   137,
     361,   433,   344,    77,    78,    23,    24,   142,   143,   144,
     145,    54,   376,   326,   414,   159,   367,   385,   370,   128,
     138,   122,   279,   122,   146,    91,     6,   147,   188,   428,
      95,   327,   430,   185,   152,   129,   287,   131,   400,   132,
     133,   134,    15,   404,   406,   122,   395,   346,   466,   396,
     172,   398,   189,   349,   361,     6,  -123,   129,  -123,   130,
     411,     1,     2,   164,   450,   210,   417,   122,   419,   420,
     483,   454,   341,   434,   190,   148,   425,   204,   342,   205,
     116,   343,   437,   431,   117,   339,   122,   340,   439,   471,
     123,     9,   183,    16,    17,    53,   436,   446,    79,   479,
     122,   335,   451,   135,   399,   -16,    83,    84,    85,    11,
     486,    12,   449,   155,   156,   163,    58,   363,   364,   464,
     322,   175,   176,   458,   197,   198,   199,   237,    75,   238,
     474,   262,   465,   263,   467,    58,   200,     6,   147,    94,
     148,    99,   100,   101,   102,   103,   104,   485,   228,    81,
     480,   209,    86,   212,    58,   227,    89,   216,   217,   218,
     219,   220,   221,   222,   231,   232,   233,   118,   119,   120,
     121,   253,   254,   226,   168,   169,   170,   171,    92,   240,
     241,   242,   164,   125,    96,   245,   246,    97,    98,   126,
     110,   111,   112,   183,   136,   146,   151,   257,   258,   259,
     260,   261,   160,     1,     2,   127,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    34,    35,    36,    37,   153,
     281,    38,   161,   167,   174,   177,    55,   276,   277,   319,
     184,   192,   193,   195,   123,   202,   208,   278,   201,   282,
     283,   284,   285,   286,    56,    57,   162,    25,    26,    27,
      28,    29,   203,   206,   207,   213,   223,   297,    36,    37,
     214,   224,    58,    56,    57,   225,   185,    39,    59,    60,
      61,    62,    63,    64,    65,   243,   265,   251,   122,   271,
     318,    58,   267,   210,   266,   268,   325,    59,    60,    61,
      62,    63,    64,    65,    56,    57,   269,   270,   297,   280,
     288,   290,   292,   365,   297,   294,   295,   296,   324,   323,
     337,   329,    58,   331,   333,   334,  -161,    -9,    59,    60,
      61,    62,    63,    64,    65,   301,   330,   302,   303,   304,
     305,   306,   307,   338,   348,   308,   309,   310,   311,   335,
      56,    57,   350,   351,   352,   353,   354,   355,   356,   357,
     345,   358,   368,    56,    57,   312,   313,   371,    58,   178,
     179,   372,   373,   377,    59,    60,    61,    62,    63,    64,
      65,    58,    56,   383,   375,   380,   381,    59,    60,    61,
      62,    63,    64,    65,   389,   316,   386,   392,   391,   393,
      58,   394,   397,   402,   407,   409,    59,    60,    61,    62,
      63,    64,    65,   317,   410,   412,   415,   421,   422,    59,
      60,    61,    62,    63,    64,    65,   350,   351,   352,   353,
     354,   355,   356,   357,   413,   358,   350,   351,   352,   353,
     354,   355,   356,   357,   423,   358,   429,   350,   351,   352,
     353,   354,   355,   356,   357,   427,   358,   435,   440,   442,
     443,   448,   447,   452,   359,   350,   351,   352,   353,   354,
     355,   356,   357,   459,   358,   403,   350,   351,   352,   353,
     354,   355,   356,   357,   460,   358,   455,   462,   461,   472,
     468,   475,   476,   438,   350,   351,   352,   353,   354,   355,
     356,   357,   477,   358,   445,   350,   351,   352,   353,   354,
     355,   356,   357,   478,   358,   482,   487,    10,    40,    14,
     264,   194,   463,   350,   351,   352,   353,   354,   355,   356,
     357,   115,   358,   473,   320,   173,   234,   230,   154,   255,
     247,   244,   289,    50,   191,   293,   418,   347,   256,   457,
     416,   484,   426,   481,   470,   432
};

static const unsigned short int yycheck[] =
{
       1,     2,    29,    55,    25,   167,    94,    35,   337,    29,
     336,   411,   327,    27,    28,    16,    17,    82,    83,    84,
      85,    20,   352,    55,   388,    90,   341,   357,   343,    35,
      50,    66,   249,    66,    50,    36,    52,    53,    73,   403,
      39,    73,   406,    76,    87,    73,   263,    74,   377,    76,
      77,    78,     5,   379,   380,    66,   371,   329,   458,   374,
     103,   376,    73,   335,   390,    52,    69,    73,    71,    75,
     385,     3,     4,    94,   438,    78,   391,    66,   393,   394,
     480,   445,    67,   412,    73,    86,   401,   152,    73,   154,
      59,    76,   422,   408,    63,    76,    66,    78,   424,   463,
      70,     0,   123,    56,    57,    68,   421,   433,   135,   473,
      66,    73,   441,    69,    76,    68,    31,    32,    33,    65,
     484,    67,   437,    36,    37,   213,    52,    53,    54,   455,
     292,    58,    59,   448,   143,   144,   145,    65,    34,    67,
     466,    65,   457,    67,   459,    52,   147,    52,    53,    70,
     151,    11,    12,    13,    14,    15,    16,   483,   185,    76,
     475,   162,    76,   164,    52,    53,    76,   168,   169,   170,
     171,   172,   173,   174,   188,   189,   190,    58,    59,    60,
      61,    53,    54,   184,    99,   100,   101,   102,    54,   197,
     198,   199,   213,    72,    65,   204,   205,    65,    65,    72,
      65,    65,    65,   224,    65,    50,    76,   216,   217,   218,
     219,   220,    65,     3,     4,    72,     6,     7,     8,     9,
      10,    11,    12,    13,    14,    15,    16,    17,    18,    76,
     251,    21,    65,    68,    72,    66,    16,   245,   246,   291,
      72,    77,    69,    66,    70,    69,    67,   248,    77,   257,
     258,   259,   260,   261,    34,    35,    17,     6,     7,     8,
       9,    10,    77,    77,    69,    69,    52,   288,    17,    18,
      71,    69,    52,    34,    35,    71,    76,    67,    58,    59,
      60,    61,    62,    63,    64,    77,    74,    78,    66,    38,
     291,    52,    74,    78,    77,    74,   297,    58,    59,    60,
      61,    62,    63,    64,    34,    35,    74,    69,   329,    68,
      73,    76,    68,   340,   335,    65,    76,    68,    65,    77,
     321,    69,    52,    54,    77,    69,    52,    73,    58,    59,
      60,    61,    62,    63,    64,    20,    74,    22,    23,    24,
      25,    26,    27,    65,    74,    30,    31,    32,    33,    73,
      34,    35,    39,    40,    41,    42,    43,    44,    45,    46,
      77,    48,    55,    34,    35,    50,    51,    74,    52,    53,
      54,    74,    73,    52,    58,    59,    60,    61,    62,    63,
      64,    52,    34,    52,    76,    67,    54,    58,    59,    60,
      61,    62,    63,    64,    19,    34,    73,    69,    68,    74,
      52,    77,    52,    69,    67,    69,    58,    59,    60,    61,
      62,    63,    64,    52,    65,    52,    78,    65,    77,    58,
      59,    60,    61,    62,    63,    64,    39,    40,    41,    42,
      43,    44,    45,    46,    68,    48,    39,    40,    41,    42,
      43,    44,    45,    46,    74,    48,    68,    39,    40,    41,
      42,    43,    44,    45,    46,    68,    48,    53,    77,    68,
      68,    65,    74,    49,    67,    39,    40,    41,    42,    43,
      44,    45,    46,    66,    48,    67,    39,    40,    41,    42,
      43,    44,    45,    46,    68,    48,    47,    68,    73,    68,
      52,    69,    69,    67,    39,    40,    41,    42,    43,    44,
      45,    46,    74,    48,    67,    39,    40,    41,    42,    43,
      44,    45,    46,    68,    48,    68,    68,     5,    18,     8,
     224,   141,    67,    39,    40,    41,    42,    43,    44,    45,
      46,    57,    48,    67,   291,   104,   193,   186,    88,   212,
     207,   202,   270,    18,   135,   274,   392,   334,   213,   447,
     390,    67,   402,   476,   462,   409
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const unsigned char yystos[] =
{
       0,     3,     4,    80,    81,    86,    52,    84,    84,     0,
      81,    65,    67,    87,    87,     5,    56,    57,    82,    88,
      89,   178,   179,    84,    84,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    17,    18,    21,    67,
      86,   106,   107,   121,   124,   125,   126,   128,   138,   139,
     142,   143,   144,    68,    88,    16,    34,    35,    52,    58,
      59,    60,    61,    62,    63,    64,    85,    94,    96,    97,
      98,    99,   100,   101,   102,    34,   108,   108,   108,    85,
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
     163,   169,   161,    53,    54,    85,   153,   155,    55,   154,
     155,    74,    74,    73,   173,    76,   173,    52,   174,   175,
      67,    54,   168,    52,   171,   173,    73,   164,   169,    19,
     162,    68,    69,    74,    77,   155,   155,    52,   155,    76,
     161,   176,    69,    67,   169,   165,   169,    67,   157,    69,
      65,   155,    52,    68,   164,    78,   163,   155,   154,   155,
     155,    65,    77,    74,   172,   155,   175,    68,   164,    68,
     164,   155,   171,   172,   161,    53,   155,   173,    67,   169,
      77,   177,    68,    68,   158,    67,   169,    74,    65,   155,
     164,   161,    49,   166,   164,    47,   170,   157,   155,    66,
      68,    73,    68,    67,   169,   155,   172,   155,    52,   167,
     170,   164,    68,    67,   169,    69,    69,    74,    68,   164,
     155,   167,    68,   172,    67,   169,   164,    68
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
      YYPOPSTACK;						\
      goto yybackup;						\
    }								\
  else								\
    {								\
      yyerror (YY_("syntax error: cannot back up")); \
      YYERROR;							\
    }								\
while (0)


#define YYTERROR	1
#define YYERRCODE	256


/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#define YYRHSLOC(Rhs, K) ((Rhs)[K])
#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)				\
    do									\
      if (N)								\
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
    while (0)
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
} while (0)

# define YY_SYMBOL_PRINT(Title, Type, Value, Location)		\
do {								\
  if (yydebug)							\
    {								\
      YYFPRINTF (stderr, "%s ", Title);				\
      yysymprint (stderr,					\
                  Type, Value);	\
      YYFPRINTF (stderr, "\n");					\
    }								\
} while (0)

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

#if defined (__STDC__) || defined (__cplusplus)
static void
yy_stack_print (short int *bottom, short int *top)
#else
static void
yy_stack_print (bottom, top)
    short int *bottom;
    short int *top;
#endif
{
  YYFPRINTF (stderr, "Stack now");
  for (/* Nothing. */; bottom <= top; ++bottom)
    YYFPRINTF (stderr, " %d", *bottom);
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)				\
do {								\
  if (yydebug)							\
    yy_stack_print ((Bottom), (Top));				\
} while (0)


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

#if defined (__STDC__) || defined (__cplusplus)
static void
yy_reduce_print (int yyrule)
#else
static void
yy_reduce_print (yyrule)
    int yyrule;
#endif
{
  int yyi;
  unsigned long int yylno = yyrline[yyrule];
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu), ",
             yyrule - 1, yylno);
  /* Print the symbols being reduced, and their result.  */
  for (yyi = yyprhs[yyrule]; 0 <= yyrhs[yyi]; yyi++)
    YYFPRINTF (stderr, "%s ", yytname[yyrhs[yyi]]);
  YYFPRINTF (stderr, "-> %s\n", yytname[yyr1[yyrule]]);
}

# define YY_REDUCE_PRINT(Rule)		\
do {					\
  if (yydebug)				\
    yy_reduce_print (Rule);		\
} while (0)

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
#  if defined (__GLIBC__) && defined (_STRING_H)
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
static YYSIZE_T
#   if defined (__STDC__) || defined (__cplusplus)
yystrlen (const char *yystr)
#   else
yystrlen (yystr)
     const char *yystr;
#   endif
{
  const char *yys = yystr;

  while (*yys++ != '\0')
    continue;

  return yys - yystr - 1;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined (__GLIBC__) && defined (_STRING_H) && defined (_GNU_SOURCE)
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
static char *
#   if defined (__STDC__) || defined (__cplusplus)
yystpcpy (char *yydest, const char *yysrc)
#   else
yystpcpy (yydest, yysrc)
     char *yydest;
     const char *yysrc;
#   endif
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
      size_t yyn = 0;
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

#endif /* YYERROR_VERBOSE */



#if YYDEBUG
/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

#if defined (__STDC__) || defined (__cplusplus)
static void
yysymprint (FILE *yyoutput, int yytype, YYSTYPE *yyvaluep)
#else
static void
yysymprint (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE *yyvaluep;
#endif
{
  /* Pacify ``unused variable'' warnings.  */
  (void) yyvaluep;

  if (yytype < YYNTOKENS)
    YYFPRINTF (yyoutput, "token %s (", yytname[yytype]);
  else
    YYFPRINTF (yyoutput, "nterm %s (", yytname[yytype]);


# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# endif
  switch (yytype)
    {
      default:
        break;
    }
  YYFPRINTF (yyoutput, ")");
}

#endif /* ! YYDEBUG */
/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

#if defined (__STDC__) || defined (__cplusplus)
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
  /* Pacify ``unused variable'' warnings.  */
  (void) yyvaluep;

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
# if defined (__STDC__) || defined (__cplusplus)
int yyparse (void *YYPARSE_PARAM);
# else
int yyparse ();
# endif
#else /* ! YYPARSE_PARAM */
#if defined (__STDC__) || defined (__cplusplus)
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
# if defined (__STDC__) || defined (__cplusplus)
int yyparse (void *YYPARSE_PARAM)
# else
int yyparse (YYPARSE_PARAM)
  void *YYPARSE_PARAM;
# endif
#else /* ! YYPARSE_PARAM */
#if defined (__STDC__) || defined (__cplusplus)
int
yyparse (void)
#else
int
yyparse ()
    ;
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

  /* Three stacks and their tools:
     `yyss': related to states,
     `yyvs': related to semantic values,
     `yyls': related to locations.

     Refer to the stacks thru separate pointers, to allow yyoverflow
     to reallocate them elsewhere.  */

  /* The state stack.  */
  short int yyssa[YYINITDEPTH];
  short int *yyss = yyssa;
  short int *yyssp;

  /* The semantic value stack.  */
  YYSTYPE yyvsa[YYINITDEPTH];
  YYSTYPE *yyvs = yyvsa;
  YYSTYPE *yyvsp;



#define YYPOPSTACK   (yyvsp--, yyssp--)

  YYSIZE_T yystacksize = YYINITDEPTH;

  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;


  /* When reducing, the number of symbols on the RHS of the reduced
     rule.  */
  int yylen;

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
     have just been pushed. so pushing a state here evens the stacks.
     */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
	/* Give user a chance to reallocate the stack. Use copies of
	   these so that the &'s don't force the real ones into
	   memory.  */
	YYSTYPE *yyvs1 = yyvs;
	short int *yyss1 = yyss;


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
	short int *yyss1 = yyss;
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

/* Do appropriate processing given the current state.  */
/* Read a look-ahead token if we need one and don't already have one.  */
/* yyresume: */

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

  /* Shift the look-ahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the token being shifted unless it is eof.  */
  if (yychar != YYEOF)
    yychar = YYEMPTY;

  *++yyvsp = yylval;


  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  yystate = yyn;
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
    { (yyval.modlist) = (yyvsp[0].modlist); modlist = (yyvsp[0].modlist); }
    break;

  case 3:
#line 138 "xi-grammar.y"
    { 
		  (yyval.modlist) = 0; 
		}
    break;

  case 4:
#line 142 "xi-grammar.y"
    { (yyval.modlist) = new ModuleList(lineno, (yyvsp[-1].module), (yyvsp[0].modlist)); }
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
    { (yyval.strval) = (yyvsp[0].strval); }
    break;

  case 10:
#line 162 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[0].strval); }
    break;

  case 11:
#line 164 "xi-grammar.y"
    {
		  char *tmp = new char[strlen((yyvsp[-3].strval))+strlen((yyvsp[0].strval))+3];
		  sprintf(tmp,"%s::%s", (yyvsp[-3].strval), (yyvsp[0].strval));
		  (yyval.strval) = tmp;
		}
    break;

  case 12:
#line 172 "xi-grammar.y"
    { 
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist)); 
		}
    break;

  case 13:
#line 176 "xi-grammar.y"
    {  
		    (yyval.module) = new Module(lineno, (yyvsp[-1].strval), (yyvsp[0].conslist)); 
		    (yyval.module)->setMain();
		}
    break;

  case 14:
#line 183 "xi-grammar.y"
    { (yyval.conslist) = 0; }
    break;

  case 15:
#line 185 "xi-grammar.y"
    { (yyval.conslist) = (yyvsp[-2].conslist); }
    break;

  case 16:
#line 189 "xi-grammar.y"
    { (yyval.conslist) = 0; }
    break;

  case 17:
#line 191 "xi-grammar.y"
    { (yyval.conslist) = new ConstructList(lineno, (yyvsp[-1].construct), (yyvsp[0].conslist)); }
    break;

  case 18:
#line 195 "xi-grammar.y"
    { if((yyvsp[-2].conslist)) (yyvsp[-2].conslist)->setExtern((yyvsp[-4].intval)); (yyval.construct) = (yyvsp[-2].conslist); }
    break;

  case 19:
#line 197 "xi-grammar.y"
    { (yyvsp[0].module)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].module); }
    break;

  case 20:
#line 199 "xi-grammar.y"
    { (yyvsp[0].member)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].member); }
    break;

  case 21:
#line 201 "xi-grammar.y"
    { (yyvsp[-1].message)->setExtern((yyvsp[-2].intval)); (yyval.construct) = (yyvsp[-1].message); }
    break;

  case 22:
#line 203 "xi-grammar.y"
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
    break;

  case 23:
#line 205 "xi-grammar.y"
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
    break;

  case 24:
#line 207 "xi-grammar.y"
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
    break;

  case 25:
#line 209 "xi-grammar.y"
    { (yyvsp[0].chare)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].chare); }
    break;

  case 26:
#line 211 "xi-grammar.y"
    { (yyvsp[0].templat)->setExtern((yyvsp[-1].intval)); (yyval.construct) = (yyvsp[0].templat); }
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
    { (yyval.tparam) = new TParamType((yyvsp[0].type)); }
    break;

  case 30:
#line 221 "xi-grammar.y"
    { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
    break;

  case 31:
#line 223 "xi-grammar.y"
    { (yyval.tparam) = new TParamVal((yyvsp[0].strval)); }
    break;

  case 32:
#line 227 "xi-grammar.y"
    { (yyval.tparlist) = new TParamList((yyvsp[0].tparam)); }
    break;

  case 33:
#line 229 "xi-grammar.y"
    { (yyval.tparlist) = new TParamList((yyvsp[-2].tparam), (yyvsp[0].tparlist)); }
    break;

  case 34:
#line 233 "xi-grammar.y"
    { (yyval.tparlist) = 0; }
    break;

  case 35:
#line 235 "xi-grammar.y"
    { (yyval.tparlist) = (yyvsp[0].tparlist); }
    break;

  case 36:
#line 239 "xi-grammar.y"
    { (yyval.tparlist) = 0; }
    break;

  case 37:
#line 241 "xi-grammar.y"
    { (yyval.tparlist) = (yyvsp[-1].tparlist); }
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
    { (yyval.ntype) = new NamedType((yyvsp[-1].strval),(yyvsp[0].tparlist)); }
    break;

  case 54:
#line 277 "xi-grammar.y"
    { (yyval.ntype) = new NamedType((yyvsp[-1].strval),(yyvsp[0].tparlist)); }
    break;

  case 55:
#line 280 "xi-grammar.y"
    { (yyval.type) = (yyvsp[0].type); }
    break;

  case 56:
#line 282 "xi-grammar.y"
    { (yyval.type) = (yyvsp[0].ntype); }
    break;

  case 57:
#line 286 "xi-grammar.y"
    { (yyval.ptype) = new PtrType((yyvsp[-1].type)); }
    break;

  case 58:
#line 290 "xi-grammar.y"
    { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
    break;

  case 59:
#line 292 "xi-grammar.y"
    { (yyvsp[-1].ptype)->indirect(); (yyval.ptype) = (yyvsp[-1].ptype); }
    break;

  case 60:
#line 296 "xi-grammar.y"
    { (yyval.ftype) = new FuncType((yyvsp[-7].type), (yyvsp[-4].strval), (yyvsp[-1].plist)); }
    break;

  case 61:
#line 300 "xi-grammar.y"
    { (yyval.type) = (yyvsp[0].type); }
    break;

  case 62:
#line 302 "xi-grammar.y"
    { (yyval.type) = (yyvsp[0].ptype); }
    break;

  case 63:
#line 304 "xi-grammar.y"
    { (yyval.type) = (yyvsp[0].ptype); }
    break;

  case 64:
#line 306 "xi-grammar.y"
    { (yyval.type) = (yyvsp[0].ftype); }
    break;

  case 65:
#line 309 "xi-grammar.y"
    { (yyval.type) = (yyvsp[0].type); }
    break;

  case 66:
#line 311 "xi-grammar.y"
    { (yyval.type) = (yyvsp[-1].type); }
    break;

  case 67:
#line 315 "xi-grammar.y"
    { (yyval.type) = new ReferenceType((yyvsp[-1].type)); }
    break;

  case 68:
#line 317 "xi-grammar.y"
    { (yyval.type) = (yyvsp[0].type); }
    break;

  case 69:
#line 321 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[0].strval)); }
    break;

  case 70:
#line 323 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[0].strval)); }
    break;

  case 71:
#line 327 "xi-grammar.y"
    { (yyval.val) = (yyvsp[-1].val); }
    break;

  case 72:
#line 331 "xi-grammar.y"
    { (yyval.vallist) = 0; }
    break;

  case 73:
#line 333 "xi-grammar.y"
    { (yyval.vallist) = new ValueList((yyvsp[-1].val), (yyvsp[0].vallist)); }
    break;

  case 74:
#line 337 "xi-grammar.y"
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].vallist)); }
    break;

  case 75:
#line 341 "xi-grammar.y"
    { (yyval.readonly) = new Readonly(lineno, (yyvsp[-2].type), (yyvsp[0].strval), 0, 1); }
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
		  (yyval.intval) = (yyvsp[-1].intval); 
		}
    break;

  case 80:
#line 363 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[0].intval); }
    break;

  case 81:
#line 365 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[-2].intval) | (yyvsp[0].intval); }
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
    { (yyval.cattr) = (yyvsp[-1].cattr); }
    break;

  case 86:
#line 381 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[0].cattr); }
    break;

  case 87:
#line 383 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
    break;

  case 88:
#line 387 "xi-grammar.y"
    { python_doc = NULL; (yyval.intval) = 0; }
    break;

  case 89:
#line 389 "xi-grammar.y"
    { python_doc = (yyvsp[0].strval); (yyval.intval) = 0; }
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
    { (yyval.cattr) = (yyvsp[-1].cattr); }
    break;

  case 93:
#line 403 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[0].cattr); }
    break;

  case 94:
#line 405 "xi-grammar.y"
    { (yyval.cattr) = (yyvsp[-2].cattr) | (yyvsp[0].cattr); }
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
    { (yyval.mv) = new MsgVar((yyvsp[-4].type), (yyvsp[-3].strval)); }
    break;

  case 98:
#line 419 "xi-grammar.y"
    { (yyval.mvlist) = new MsgVarList((yyvsp[0].mv)); }
    break;

  case 99:
#line 421 "xi-grammar.y"
    { (yyval.mvlist) = new MsgVarList((yyvsp[-1].mv), (yyvsp[0].mvlist)); }
    break;

  case 100:
#line 425 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, (yyvsp[0].ntype)); }
    break;

  case 101:
#line 427 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, (yyvsp[-3].ntype), (yyvsp[-1].mvlist)); }
    break;

  case 102:
#line 431 "xi-grammar.y"
    { (yyval.typelist) = 0; }
    break;

  case 103:
#line 433 "xi-grammar.y"
    { (yyval.typelist) = (yyvsp[0].typelist); }
    break;

  case 104:
#line 437 "xi-grammar.y"
    { (yyval.typelist) = new TypeList((yyvsp[0].ntype)); }
    break;

  case 105:
#line 439 "xi-grammar.y"
    { (yyval.typelist) = new TypeList((yyvsp[-2].ntype), (yyvsp[0].typelist)); }
    break;

  case 106:
#line 443 "xi-grammar.y"
    { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
    break;

  case 107:
#line 445 "xi-grammar.y"
    { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
    break;

  case 108:
#line 449 "xi-grammar.y"
    { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
    break;

  case 109:
#line 453 "xi-grammar.y"
    { (yyval.chare) = new NodeGroup(lineno, (yyvsp[-3].cattr), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
    break;

  case 110:
#line 457 "xi-grammar.y"
    {/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",(yyvsp[-2].strval));
			(yyval.ntype) = new NamedType(buf); 
		}
    break;

  case 111:
#line 463 "xi-grammar.y"
    { (yyval.ntype) = new NamedType((yyvsp[-1].strval)); }
    break;

  case 112:
#line 467 "xi-grammar.y"
    {  (yyval.chare) = new Array(lineno, (yyvsp[-4].cattr), (yyvsp[-3].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
    break;

  case 113:
#line 469 "xi-grammar.y"
    {  (yyval.chare) = new Array(lineno, (yyvsp[-3].cattr), (yyvsp[-4].ntype), (yyvsp[-2].ntype), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
    break;

  case 114:
#line 473 "xi-grammar.y"
    { (yyval.chare) = new Chare(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist));}
    break;

  case 115:
#line 475 "xi-grammar.y"
    { (yyval.chare) = new MainChare(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
    break;

  case 116:
#line 479 "xi-grammar.y"
    { (yyval.chare) = new Group(lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
    break;

  case 117:
#line 483 "xi-grammar.y"
    { (yyval.chare) = new NodeGroup( lineno, (yyvsp[-3].cattr), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
    break;

  case 118:
#line 487 "xi-grammar.y"
    { (yyval.chare) = new Array( lineno, 0, (yyvsp[-3].ntype), new NamedType((yyvsp[-2].strval)), (yyvsp[-1].typelist), (yyvsp[0].mbrlist)); }
    break;

  case 119:
#line 491 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-1].strval))); }
    break;

  case 120:
#line 493 "xi-grammar.y"
    { (yyval.message) = new Message(lineno, new NamedType((yyvsp[-4].strval)), (yyvsp[-2].mvlist)); }
    break;

  case 121:
#line 497 "xi-grammar.y"
    { (yyval.type) = 0; }
    break;

  case 122:
#line 499 "xi-grammar.y"
    { (yyval.type) = (yyvsp[0].type); }
    break;

  case 123:
#line 503 "xi-grammar.y"
    { (yyval.strval) = 0; }
    break;

  case 124:
#line 505 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[0].strval); }
    break;

  case 125:
#line 507 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[0].strval); }
    break;

  case 126:
#line 511 "xi-grammar.y"
    { (yyval.tvar) = new TType(new NamedType((yyvsp[-1].strval)), (yyvsp[0].type)); }
    break;

  case 127:
#line 513 "xi-grammar.y"
    { (yyval.tvar) = new TFunc((yyvsp[-1].ftype), (yyvsp[0].strval)); }
    break;

  case 128:
#line 515 "xi-grammar.y"
    { (yyval.tvar) = new TName((yyvsp[-2].type), (yyvsp[-1].strval), (yyvsp[0].strval)); }
    break;

  case 129:
#line 519 "xi-grammar.y"
    { (yyval.tvarlist) = new TVarList((yyvsp[0].tvar)); }
    break;

  case 130:
#line 521 "xi-grammar.y"
    { (yyval.tvarlist) = new TVarList((yyvsp[-2].tvar), (yyvsp[0].tvarlist)); }
    break;

  case 131:
#line 525 "xi-grammar.y"
    { (yyval.tvarlist) = (yyvsp[-1].tvarlist); }
    break;

  case 132:
#line 529 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
    break;

  case 133:
#line 531 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
    break;

  case 134:
#line 533 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
    break;

  case 135:
#line 535 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].chare)); (yyvsp[0].chare)->setTemplate((yyval.templat)); }
    break;

  case 136:
#line 537 "xi-grammar.y"
    { (yyval.templat) = new Template((yyvsp[-1].tvarlist), (yyvsp[0].message)); (yyvsp[0].message)->setTemplate((yyval.templat)); }
    break;

  case 137:
#line 541 "xi-grammar.y"
    { (yyval.mbrlist) = 0; }
    break;

  case 138:
#line 543 "xi-grammar.y"
    { (yyval.mbrlist) = (yyvsp[-2].mbrlist); }
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
    { (yyval.mbrlist) = new MemberList((yyvsp[-1].member), (yyvsp[0].mbrlist)); }
    break;

  case 141:
#line 570 "xi-grammar.y"
    { (yyval.member) = (yyvsp[-1].readonly); }
    break;

  case 142:
#line 572 "xi-grammar.y"
    { (yyval.member) = (yyvsp[-1].readonly); }
    break;

  case 144:
#line 575 "xi-grammar.y"
    { (yyval.member) = (yyvsp[-1].member); }
    break;

  case 145:
#line 577 "xi-grammar.y"
    { (yyval.member) = (yyvsp[-1].pupable); }
    break;

  case 146:
#line 579 "xi-grammar.y"
    { (yyval.member) = (yyvsp[-1].includeFile); }
    break;

  case 147:
#line 581 "xi-grammar.y"
    { (yyval.member) = new ClassDeclaration(lineno,(yyvsp[-1].strval)); }
    break;

  case 148:
#line 585 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1); }
    break;

  case 149:
#line 587 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1); }
    break;

  case 150:
#line 589 "xi-grammar.y"
    { printf("Warning: deprecated use of initcall. Use initnode or initproc instead.\n"); 
		  (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 1); }
    break;

  case 151:
#line 592 "xi-grammar.y"
    { printf("Warning: deprecated use of initcall. Use initnode or initproc instead.\n");
		  (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 1); }
    break;

  case 152:
#line 597 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[0].strval), 0); }
    break;

  case 153:
#line 599 "xi-grammar.y"
    { (yyval.member) = new InitCall(lineno, (yyvsp[-3].strval), 0); }
    break;

  case 154:
#line 603 "xi-grammar.y"
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[0].strval),0); }
    break;

  case 155:
#line 605 "xi-grammar.y"
    { (yyval.pupable) = new PUPableClass(lineno,(yyvsp[-2].strval),(yyvsp[0].pupable)); }
    break;

  case 156:
#line 609 "xi-grammar.y"
    { (yyval.includeFile) = new IncludeFile(lineno,(yyvsp[0].strval)); }
    break;

  case 157:
#line 613 "xi-grammar.y"
    { (yyval.member) = (yyvsp[-1].entry); }
    break;

  case 158:
#line 615 "xi-grammar.y"
    { (yyval.member) = (yyvsp[0].member); }
    break;

  case 159:
#line 619 "xi-grammar.y"
    { 
		  if ((yyvsp[0].sc) != 0) { 
		    (yyvsp[0].sc)->con1 = new SdagConstruct(SIDENT, (yyvsp[-3].strval));
  		    if ((yyvsp[-2].plist) != 0)
                      (yyvsp[0].sc)->param = new ParamList((yyvsp[-2].plist));
 		    else 
 	 	      (yyvsp[0].sc)->param = new ParamList(new Parameter(0, new BuiltinType("void")));
                  }
		  (yyval.entry) = new Entry(lineno, (yyvsp[-5].intval), (yyvsp[-4].type), (yyvsp[-3].strval), (yyvsp[-2].plist), (yyvsp[-1].val), (yyvsp[0].sc), 0, 0); 
		}
    break;

  case 160:
#line 630 "xi-grammar.y"
    { 
		  if ((yyvsp[0].sc) != 0) {
		    (yyvsp[0].sc)->con1 = new SdagConstruct(SIDENT, (yyvsp[-2].strval));
		    if ((yyvsp[-1].plist) != 0)
                      (yyvsp[0].sc)->param = new ParamList((yyvsp[-1].plist));
		    else
                      (yyvsp[0].sc)->param = new ParamList(new Parameter(0, new BuiltinType("void")));
                  }
		  (yyval.entry) = new Entry(lineno, (yyvsp[-3].intval),     0, (yyvsp[-2].strval), (yyvsp[-1].plist),  0, (yyvsp[0].sc), 0, 0); 
		}
    break;

  case 161:
#line 643 "xi-grammar.y"
    { (yyval.type) = new BuiltinType("void"); }
    break;

  case 162:
#line 645 "xi-grammar.y"
    { (yyval.type) = (yyvsp[0].ptype); }
    break;

  case 163:
#line 649 "xi-grammar.y"
    { (yyval.intval) = 0; }
    break;

  case 164:
#line 651 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[-1].intval); }
    break;

  case 165:
#line 655 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[0].intval); }
    break;

  case 166:
#line 657 "xi-grammar.y"
    { (yyval.intval) = (yyvsp[-2].intval) | (yyvsp[0].intval); }
    break;

  case 167:
#line 661 "xi-grammar.y"
    { (yyval.intval) = STHREADED; }
    break;

  case 168:
#line 663 "xi-grammar.y"
    { (yyval.intval) = SSYNC; }
    break;

  case 169:
#line 665 "xi-grammar.y"
    { (yyval.intval) = SIGET; }
    break;

  case 170:
#line 667 "xi-grammar.y"
    { (yyval.intval) = SLOCKED; }
    break;

  case 171:
#line 669 "xi-grammar.y"
    { (yyval.intval) = SCREATEHERE; }
    break;

  case 172:
#line 671 "xi-grammar.y"
    { (yyval.intval) = SCREATEHOME; }
    break;

  case 173:
#line 673 "xi-grammar.y"
    { (yyval.intval) = SNOKEEP; }
    break;

  case 174:
#line 675 "xi-grammar.y"
    { (yyval.intval) = SNOTRACE; }
    break;

  case 175:
#line 677 "xi-grammar.y"
    { (yyval.intval) = SIMMEDIATE; }
    break;

  case 176:
#line 679 "xi-grammar.y"
    { (yyval.intval) = SSKIPSCHED; }
    break;

  case 177:
#line 681 "xi-grammar.y"
    { (yyval.intval) = SINLINE; }
    break;

  case 178:
#line 683 "xi-grammar.y"
    { (yyval.intval) = SLOCAL; }
    break;

  case 179:
#line 685 "xi-grammar.y"
    { (yyval.intval) = SPYTHON; }
    break;

  case 180:
#line 689 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[0].strval)); }
    break;

  case 181:
#line 691 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[0].strval)); }
    break;

  case 182:
#line 693 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[0].strval)); }
    break;

  case 183:
#line 697 "xi-grammar.y"
    { (yyval.strval) = ""; }
    break;

  case 184:
#line 699 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[0].strval); }
    break;

  case 185:
#line 701 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s, %s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 186:
#line 709 "xi-grammar.y"
    { (yyval.strval) = ""; }
    break;

  case 187:
#line 711 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[0].strval); }
    break;

  case 188:
#line 713 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s[%s]%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 189:
#line 719 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s{%s}%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 190:
#line 725 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-4].strval))+strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"%s(%s)%s", (yyvsp[-4].strval), (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 191:
#line 731 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen((yyvsp[-2].strval))+strlen((yyvsp[0].strval))+3];
			sprintf(tmp,"(%s)%s", (yyvsp[-2].strval), (yyvsp[0].strval));
			(yyval.strval) = tmp;
		}
    break;

  case 192:
#line 739 "xi-grammar.y"
    {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].type),(yyvsp[-1].strval));
		}
    break;

  case 193:
#line 746 "xi-grammar.y"
    { 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			(yyval.intval) = 0;
		}
    break;

  case 194:
#line 754 "xi-grammar.y"
    { 
			in_braces=0;
			(yyval.intval) = 0;
		}
    break;

  case 195:
#line 761 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[0].type));}
    break;

  case 196:
#line 763 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[-1].type),(yyvsp[0].strval));}
    break;

  case 197:
#line 765 "xi-grammar.y"
    { (yyval.pname) = new Parameter(lineno, (yyvsp[-3].type),(yyvsp[-2].strval),0,(yyvsp[0].val));}
    break;

  case 198:
#line 767 "xi-grammar.y"
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			(yyval.pname) = new Parameter(lineno, (yyvsp[-2].pname)->getType(), (yyvsp[-2].pname)->getName() ,(yyvsp[-1].strval));
		}
    break;

  case 199:
#line 774 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[0].pname)); }
    break;

  case 200:
#line 776 "xi-grammar.y"
    { (yyval.plist) = new ParamList((yyvsp[-2].pname),(yyvsp[0].plist)); }
    break;

  case 201:
#line 780 "xi-grammar.y"
    { (yyval.plist) = (yyvsp[-1].plist); }
    break;

  case 202:
#line 782 "xi-grammar.y"
    { (yyval.plist) = 0; }
    break;

  case 203:
#line 786 "xi-grammar.y"
    { (yyval.val) = 0; }
    break;

  case 204:
#line 788 "xi-grammar.y"
    { (yyval.val) = new Value((yyvsp[0].strval)); }
    break;

  case 205:
#line 792 "xi-grammar.y"
    { (yyval.sc) = 0; }
    break;

  case 206:
#line 794 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSDAGENTRY, (yyvsp[0].sc)); }
    break;

  case 207:
#line 796 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSDAGENTRY, (yyvsp[-1].sc)); }
    break;

  case 208:
#line 800 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSLIST, (yyvsp[0].sc)); }
    break;

  case 209:
#line 802 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SSLIST, (yyvsp[-1].sc), (yyvsp[0].sc));  }
    break;

  case 210:
#line 806 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SOLIST, (yyvsp[0].sc)); }
    break;

  case 211:
#line 808 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SOLIST, (yyvsp[-1].sc), (yyvsp[0].sc)); }
    break;

  case 212:
#line 812 "xi-grammar.y"
    { (yyval.sc) = 0; }
    break;

  case 213:
#line 814 "xi-grammar.y"
    { (yyval.sc) = (yyvsp[-1].sc); }
    break;

  case 214:
#line 818 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SPUBLISHES, new SdagConstruct(SIDENT, (yyvsp[0].strval))); }
    break;

  case 215:
#line 820 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SPUBLISHES, new SdagConstruct(SIDENT, (yyvsp[-2].strval)), (yyvsp[0].sc));  }
    break;

  case 216:
#line 824 "xi-grammar.y"
    { (yyval.strval) = (yyvsp[0].strval); }
    break;

  case 217:
#line 826 "xi-grammar.y"
    { (yyval.strval) = 0; }
    break;

  case 218:
#line 830 "xi-grammar.y"
    { RemoveSdagComments((yyvsp[-2].strval));
		   (yyval.sc) = new SdagConstruct(SATOMIC, new XStr((yyvsp[-2].strval)), (yyvsp[0].sc), 0,0,0,0, 0 ); 
		   if ((yyvsp[-4].strval)) { (yyvsp[-4].strval)[strlen((yyvsp[-4].strval))-1]=0; (yyval.sc)->traceName = new XStr((yyvsp[-4].strval)+1); }
		 }
    break;

  case 219:
#line 835 "xi-grammar.y"
    {  
		   in_braces = 0;
		   if (((yyvsp[-4].plist)->isVoid() == 0) && ((yyvsp[-4].plist)->isMessage() == 0))
                   {
		      connectEntries->append(new Entry(0, 0, new BuiltinType("void"), (yyvsp[-5].strval), 
	 	 			new ParamList(new Parameter(lineno, new PtrType( 
                                        new NamedType("CkMarshallMsg")), "_msg")), 0, 0, 0, 1, (yyvsp[-4].plist)));
		   }
		   else  {
		      connectEntries->append(new Entry(0, 0, new BuiltinType("void"), (yyvsp[-5].strval), (yyvsp[-4].plist), 0, 0, 0, 1, (yyvsp[-4].plist)));
                   }
                   (yyval.sc) = new SdagConstruct(SCONNECT, (yyvsp[-5].strval), (yyvsp[-1].strval), (yyvsp[-4].plist));
		}
    break;

  case 220:
#line 849 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SWHEN, 0, 0,0,0,0,0,(yyvsp[-2].entrylist)); }
    break;

  case 221:
#line 851 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SWHEN, 0, 0, 0,0,0, (yyvsp[0].sc), (yyvsp[-1].entrylist)); }
    break;

  case 222:
#line 853 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SWHEN, 0, 0, 0,0,0, (yyvsp[-1].sc), (yyvsp[-3].entrylist)); }
    break;

  case 223:
#line 855 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SOVERLAP,0, 0,0,0,0,(yyvsp[-1].sc), 0); }
    break;

  case 224:
#line 857 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFOR, 0, new SdagConstruct(SINT_EXPR, (yyvsp[-8].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[-6].strval)),
		             new SdagConstruct(SINT_EXPR, (yyvsp[-4].strval)), 0, (yyvsp[-1].sc), 0); }
    break;

  case 225:
#line 860 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFOR, 0, new SdagConstruct(SINT_EXPR, (yyvsp[-6].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[-4].strval)), 
		         new SdagConstruct(SINT_EXPR, (yyvsp[-2].strval)), 0, (yyvsp[0].sc), 0); }
    break;

  case 226:
#line 863 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFORALL, 0, new SdagConstruct(SIDENT, (yyvsp[-9].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[-6].strval)), 
		             new SdagConstruct(SINT_EXPR, (yyvsp[-4].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[-2].strval)), (yyvsp[0].sc), 0); }
    break;

  case 227:
#line 866 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFORALL, 0, new SdagConstruct(SIDENT, (yyvsp[-11].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[-8].strval)), 
		                 new SdagConstruct(SINT_EXPR, (yyvsp[-6].strval)), new SdagConstruct(SINT_EXPR, (yyvsp[-4].strval)), (yyvsp[-1].sc), 0); }
    break;

  case 228:
#line 869 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, (yyvsp[-3].strval)), (yyvsp[0].sc),0,0,(yyvsp[-1].sc),0); }
    break;

  case 229:
#line 871 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, (yyvsp[-5].strval)), (yyvsp[0].sc),0,0,(yyvsp[-2].sc),0); }
    break;

  case 230:
#line 873 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, (yyvsp[-2].strval)), 0,0,0,(yyvsp[0].sc),0); }
    break;

  case 231:
#line 875 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SWHILE, 0, new SdagConstruct(SINT_EXPR, (yyvsp[-4].strval)), 0,0,0,(yyvsp[-1].sc),0); }
    break;

  case 232:
#line 877 "xi-grammar.y"
    { (yyval.sc) = (yyvsp[-1].sc); }
    break;

  case 233:
#line 881 "xi-grammar.y"
    { (yyval.sc) = 0; }
    break;

  case 234:
#line 883 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SELSE, 0,0,0,0,0, (yyvsp[0].sc),0); }
    break;

  case 235:
#line 885 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SELSE, 0,0,0,0,0, (yyvsp[-1].sc),0); }
    break;

  case 236:
#line 888 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFORWARD, new SdagConstruct(SIDENT, (yyvsp[0].strval))); }
    break;

  case 237:
#line 890 "xi-grammar.y"
    { (yyval.sc) = new SdagConstruct(SFORWARD, new SdagConstruct(SIDENT, (yyvsp[-2].strval)), (yyvsp[0].sc));  }
    break;

  case 238:
#line 894 "xi-grammar.y"
    { in_int_expr = 0; (yyval.intval) = 0; }
    break;

  case 239:
#line 898 "xi-grammar.y"
    { in_int_expr = 1; (yyval.intval) = 0; }
    break;

  case 240:
#line 902 "xi-grammar.y"
    { 
		  if ((yyvsp[0].plist) != 0)
		     (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[-1].strval), (yyvsp[0].plist), 0, 0, 0, 0); 
		  else
		     (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[-1].strval), 
				new ParamList(new Parameter(0, new BuiltinType("void"))), 0, 0, 0, 0); 
		}
    break;

  case 241:
#line 910 "xi-grammar.y"
    { if ((yyvsp[0].plist) != 0)
		    (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[-4].strval), (yyvsp[0].plist), 0, 0, (yyvsp[-2].strval), 0); 
		  else
		    (yyval.entry) = new Entry(lineno, 0, 0, (yyvsp[-4].strval), new ParamList(new Parameter(0, new BuiltinType("void"))), 0, 0, (yyvsp[-2].strval), 0); 
		}
    break;

  case 242:
#line 918 "xi-grammar.y"
    { (yyval.entrylist) = new EntryList((yyvsp[0].entry)); }
    break;

  case 243:
#line 920 "xi-grammar.y"
    { (yyval.entrylist) = new EntryList((yyvsp[-2].entry),(yyvsp[0].entrylist)); }
    break;

  case 244:
#line 924 "xi-grammar.y"
    { in_bracket=1; }
    break;

  case 245:
#line 927 "xi-grammar.y"
    { in_bracket=0; }
    break;

  case 246:
#line 931 "xi-grammar.y"
    { if (!macroDefined((yyvsp[0].strval), 1)) in_comment = 1; }
    break;

  case 247:
#line 935 "xi-grammar.y"
    { if (!macroDefined((yyvsp[0].strval), 0)) in_comment = 1; }
    break;


      default: break;
    }

/* Line 1126 of yacc.c.  */
#line 3090 "y.tab.c"

  yyvsp -= yylen;
  yyssp -= yylen;


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
#if YYERROR_VERBOSE
      yyn = yypact[yystate];

      if (YYPACT_NINF < yyn && yyn < YYLAST)
	{
	  int yytype = YYTRANSLATE (yychar);
	  YYSIZE_T yysize0 = yytnamerr (0, yytname[yytype]);
	  YYSIZE_T yysize = yysize0;
	  YYSIZE_T yysize1;
	  int yysize_overflow = 0;
	  char *yymsg = 0;
#	  define YYERROR_VERBOSE_ARGS_MAXIMUM 5
	  char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
	  int yyx;

#if 0
	  /* This is so xgettext sees the translatable formats that are
	     constructed on the fly.  */
	  YY_("syntax error, unexpected %s");
	  YY_("syntax error, unexpected %s, expecting %s");
	  YY_("syntax error, unexpected %s, expecting %s or %s");
	  YY_("syntax error, unexpected %s, expecting %s or %s or %s");
	  YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s");
#endif
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
	  int yychecklim = YYLAST - yyn;
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
		yysize_overflow |= yysize1 < yysize;
		yysize = yysize1;
		yyfmt = yystpcpy (yyfmt, yyprefix);
		yyprefix = yyor;
	      }

	  yyf = YY_(yyformat);
	  yysize1 = yysize + yystrlen (yyf);
	  yysize_overflow |= yysize1 < yysize;
	  yysize = yysize1;

	  if (!yysize_overflow && yysize <= YYSTACK_ALLOC_MAXIMUM)
	    yymsg = (char *) YYSTACK_ALLOC (yysize);
	  if (yymsg)
	    {
	      /* Avoid sprintf, as that infringes on the user's name space.
		 Don't have undefined behavior even if the translation
		 produced a string with the wrong number of "%s"s.  */
	      char *yyp = yymsg;
	      int yyi = 0;
	      while ((*yyp = *yyf))
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
	      yyerror (yymsg);
	      YYSTACK_FREE (yymsg);
	    }
	  else
	    {
	      yyerror (YY_("syntax error"));
	      goto yyexhaustedlab;
	    }
	}
      else
#endif /* YYERROR_VERBOSE */
	yyerror (YY_("syntax error"));
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
	  yydestruct ("Error: discarding", yytoken, &yylval);
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
  if (0)
     goto yyerrorlab;

yyvsp -= yylen;
  yyssp -= yylen;
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


      yydestruct ("Error: popping", yystos[yystate], yyvsp);
      YYPOPSTACK;
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  if (yyn == YYFINAL)
    YYACCEPT;

  *++yyvsp = yylval;


  /* Shift the error token. */
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
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
		  yystos[*yyssp], yyvsp);
      YYPOPSTACK;
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
  return yyresult;
}


#line 938 "xi-grammar.y"

void yyerror(const char *mesg)
{
  cout << cur_file<<":"<<lineno<<": Charmxi syntax error> " << mesg << endl;
  // return 0;
}

