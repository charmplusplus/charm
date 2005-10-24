/* A Bison parser, made by GNU Bison 1.875.  */

/* Skeleton parser for Yacc-like parsing with Bison,
   Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002 Free Software Foundation, Inc.

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
   Foundation, Inc., 59 Temple Place - Suite 330,
   Boston, MA 02111-1307, USA.  */

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
     EXCLUSIVE = 278,
     IMMEDIATE = 279,
     SKIPSCHED = 280,
     INLINE = 281,
     VIRTUAL = 282,
     MIGRATABLE = 283,
     CREATEHERE = 284,
     CREATEHOME = 285,
     NOKEEP = 286,
     NOTRACE = 287,
     VOID = 288,
     CONST = 289,
     PACKED = 290,
     VARSIZE = 291,
     ENTRY = 292,
     FOR = 293,
     FORALL = 294,
     WHILE = 295,
     WHEN = 296,
     OVERLAP = 297,
     ATOMIC = 298,
     FORWARD = 299,
     IF = 300,
     ELSE = 301,
     CONNECT = 302,
     PUBLISHES = 303,
     PYTHON = 304,
     IDENT = 305,
     NUMBER = 306,
     LITERAL = 307,
     CPROGRAM = 308,
     HASHIF = 309,
     HASHIFDEF = 310,
     INT = 311,
     LONG = 312,
     SHORT = 313,
     CHAR = 314,
     FLOAT = 315,
     DOUBLE = 316,
     UNSIGNED = 317
   };
#endif
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
#define EXCLUSIVE 278
#define IMMEDIATE 279
#define SKIPSCHED 280
#define INLINE 281
#define VIRTUAL 282
#define MIGRATABLE 283
#define CREATEHERE 284
#define CREATEHOME 285
#define NOKEEP 286
#define NOTRACE 287
#define VOID 288
#define CONST 289
#define PACKED 290
#define VARSIZE 291
#define ENTRY 292
#define FOR 293
#define FORALL 294
#define WHILE 295
#define WHEN 296
#define OVERLAP 297
#define ATOMIC 298
#define FORWARD 299
#define IF 300
#define ELSE 301
#define CONNECT 302
#define PUBLISHES 303
#define PYTHON 304
#define IDENT 305
#define NUMBER 306
#define LITERAL 307
#define CPROGRAM 308
#define HASHIF 309
#define HASHIFDEF 310
#define INT 311
#define LONG 312
#define SHORT 313
#define CHAR 314
#define FLOAT 315
#define DOUBLE 316
#define UNSIGNED 317




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
/* Line 191 of yacc.c.  */
#line 250 "y.tab.c"
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 214 of yacc.c.  */
#line 262 "y.tab.c"

#if ! defined (yyoverflow) || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# if YYSTACK_USE_ALLOCA
#  define YYSTACK_ALLOC alloca
# else
#  ifndef YYSTACK_USE_ALLOCA
#   if defined (alloca) || defined (_ALLOCA_H)
#    define YYSTACK_ALLOC alloca
#   else
#    ifdef __GNUC__
#     define YYSTACK_ALLOC __builtin_alloca
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's `empty if-body' warning. */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
# else
#  if defined (__STDC__) || defined (__cplusplus)
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   define YYSIZE_T size_t
#  endif
#  define YYSTACK_ALLOC malloc
#  define YYSTACK_FREE free
# endif
#endif /* ! defined (yyoverflow) || YYERROR_VERBOSE */


#if (! defined (yyoverflow) \
     && (! defined (__cplusplus) \
	 || (YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  short yyss;
  YYSTYPE yyvs;
  };

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (short) + sizeof (YYSTYPE))				\
      + YYSTACK_GAP_MAXIMUM)

/* Copy COUNT objects from FROM to TO.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if 1 < __GNUC__
#   define YYCOPY(To, From, Count) \
      __builtin_memcpy (To, From, (Count) * sizeof (*(From)))
#  else
#   define YYCOPY(To, From, Count)		\
      do					\
	{					\
	  register YYSIZE_T yyi;		\
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
   typedef short yysigned_char;
#endif

/* YYFINAL -- State number of the termination state. */
#define YYFINAL  9
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   553

/* YYNTOKENS -- Number of terminals. */
#define YYNTOKENS  77
/* YYNNTS -- Number of nonterminals. */
#define YYNNTS  101
/* YYNRULES -- Number of rules. */
#define YYNRULES  245
/* YYNRULES -- Number of states. */
#define YYNSTATES  486

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   317

#define YYTRANSLATE(YYX) 						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const unsigned char yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    73,     2,
      71,    72,    70,     2,    67,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    64,    63,
      68,    76,    69,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    74,     2,    75,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    65,     2,    66,     2,     2,     2,     2,
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
      55,    56,    57,    58,    59,    60,    61,    62
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const unsigned short yyprhs[] =
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
     543,   545,   547,   549,   551,   553,   555,   557,   560,   562,
     564,   566,   567,   569,   573,   574,   576,   582,   588,   594,
     599,   603,   605,   607,   609,   612,   617,   621,   623,   627,
     631,   634,   635,   639,   640,   642,   646,   648,   651,   653,
     656,   657,   662,   664,   668,   670,   671,   678,   687,   692,
     696,   702,   707,   719,   729,   742,   757,   764,   773,   779,
     787,   791,   792,   795,   800,   802,   806,   808,   810,   813,
     819,   821,   825,   827,   829,   832
};

/* YYRHS -- A `-1'-separated list of the rules' RHS. */
static const short yyrhs[] =
{
      78,     0,    -1,    79,    -1,    -1,    84,    79,    -1,    -1,
       5,    -1,    -1,    63,    -1,    50,    -1,    50,    -1,    83,
      64,    64,    50,    -1,     3,    82,    85,    -1,     4,    82,
      85,    -1,    63,    -1,    65,    86,    66,    81,    -1,    -1,
      87,    86,    -1,    80,    65,    86,    66,    81,    -1,    80,
      84,    -1,    80,   140,    -1,    80,   119,    63,    -1,    80,
     122,    -1,    80,   123,    -1,    80,   124,    -1,    80,   126,
      -1,    80,   137,    -1,   176,    -1,   177,    -1,   100,    -1,
      51,    -1,    52,    -1,    88,    -1,    88,    67,    89,    -1,
      -1,    89,    -1,    -1,    68,    90,    69,    -1,    56,    -1,
      57,    -1,    58,    -1,    59,    -1,    62,    56,    -1,    62,
      57,    -1,    62,    57,    56,    -1,    62,    57,    57,    -1,
      62,    58,    -1,    62,    59,    -1,    57,    57,    -1,    60,
      -1,    61,    -1,    57,    61,    -1,    33,    -1,    82,    91,
      -1,    83,    91,    -1,    92,    -1,    94,    -1,    95,    70,
      -1,    96,    70,    -1,    97,    70,    -1,    99,    71,    70,
      82,    72,    71,   158,    72,    -1,    95,    -1,    96,    -1,
      97,    -1,    98,    -1,    34,    99,    -1,    99,    34,    -1,
      99,    73,    -1,    99,    -1,    51,    -1,    83,    -1,    74,
     101,    75,    -1,    -1,   102,   103,    -1,     6,   100,    83,
     103,    -1,     6,    16,    95,    70,    82,    -1,    -1,    33,
      -1,    -1,    74,   108,    75,    -1,   109,    -1,   109,    67,
     108,    -1,    35,    -1,    36,    -1,    -1,    74,   111,    75,
      -1,   116,    -1,   116,    67,   111,    -1,    -1,    52,    -1,
      49,    -1,    -1,    74,   115,    75,    -1,   113,    -1,   113,
      67,   115,    -1,    28,    -1,    49,    -1,   100,    82,    74,
      75,    63,    -1,   117,    -1,   117,   118,    -1,    16,   107,
      93,    -1,    16,   107,    93,    65,   118,    66,    -1,    -1,
      64,   121,    -1,    93,    -1,    93,    67,   121,    -1,    11,
     110,    93,   120,   138,    -1,    12,   110,    93,   120,   138,
      -1,    13,   110,    93,   120,   138,    -1,    14,   110,    93,
     120,   138,    -1,    74,    51,    82,    75,    -1,    74,    82,
      75,    -1,    15,   114,   125,    93,   120,   138,    -1,    15,
     125,   114,    93,   120,   138,    -1,    11,   110,    82,   120,
     138,    -1,    12,   110,    82,   120,   138,    -1,    13,   110,
      82,   120,   138,    -1,    14,   110,    82,   120,   138,    -1,
      15,   125,    82,   120,   138,    -1,    16,   107,    82,    63,
      -1,    16,   107,    82,    65,   118,    66,    63,    -1,    -1,
      76,   100,    -1,    -1,    76,    51,    -1,    76,    52,    -1,
      17,    82,   132,    -1,    98,   133,    -1,   100,    82,   133,
      -1,   134,    -1,   134,    67,   135,    -1,    21,    68,   135,
      69,    -1,   136,   127,    -1,   136,   128,    -1,   136,   129,
      -1,   136,   130,    -1,   136,   131,    -1,    63,    -1,    65,
     139,    66,    81,    -1,    -1,   145,   139,    -1,   104,    63,
      -1,   105,    63,    -1,   142,    63,    -1,   141,    63,    -1,
      10,   143,    63,    -1,    18,   144,    63,    -1,    17,    82,
      63,    -1,     8,   106,    83,    -1,     8,   106,    83,    71,
     106,    72,    -1,     7,   106,    83,    -1,     7,   106,    83,
      71,   106,    72,    -1,     9,   106,    83,    -1,     9,   106,
      83,    71,   106,    72,    -1,    83,    -1,    83,    67,   143,
      -1,    52,    -1,   146,    63,    -1,   140,    -1,    37,   148,
     147,    82,   159,   160,   161,    -1,    37,   148,    82,   159,
     161,    -1,    33,    -1,    96,    -1,    -1,    74,   149,    75,
      -1,   150,    -1,   150,    67,   149,    -1,    20,    -1,    22,
      -1,    23,    -1,    29,    -1,    30,    -1,    31,    -1,    32,
      -1,    24,    -1,    25,    -1,    26,    -1,    49,   112,    -1,
      52,    -1,    51,    -1,    83,    -1,    -1,    53,    -1,    53,
      67,   152,    -1,    -1,    53,    -1,    53,    74,   153,    75,
     153,    -1,    53,    65,   153,    66,   153,    -1,    53,    71,
     152,    72,   153,    -1,    71,   153,    72,   153,    -1,   100,
      82,    74,    -1,    65,    -1,    66,    -1,   100,    -1,   100,
      82,    -1,   100,    82,    76,   151,    -1,   154,   153,    75,
      -1,   157,    -1,   157,    67,   158,    -1,    71,   158,    72,
      -1,    71,    72,    -1,    -1,    19,    76,    51,    -1,    -1,
     167,    -1,    65,   162,    66,    -1,   167,    -1,   167,   162,
      -1,   167,    -1,   167,   162,    -1,    -1,    48,    71,   165,
      72,    -1,    50,    -1,    50,    67,   165,    -1,    52,    -1,
      -1,    43,   166,   155,   153,   156,   164,    -1,    47,    71,
      50,   159,    72,   155,   153,    66,    -1,    41,   173,    65,
      66,    -1,    41,   173,   167,    -1,    41,   173,    65,   162,
      66,    -1,    42,    65,   163,    66,    -1,    38,   171,   153,
      63,   153,    63,   153,   170,    65,   162,    66,    -1,    38,
     171,   153,    63,   153,    63,   153,   170,   167,    -1,    39,
      74,    50,    75,   171,   153,    64,   153,    67,   153,   170,
     167,    -1,    39,    74,    50,    75,   171,   153,    64,   153,
      67,   153,   170,    65,   162,    66,    -1,    45,   171,   153,
     170,   167,   168,    -1,    45,   171,   153,   170,    65,   162,
      66,   168,    -1,    40,   171,   153,   170,   167,    -1,    40,
     171,   153,   170,    65,   162,    66,    -1,    44,   169,    63,
      -1,    -1,    46,   167,    -1,    46,    65,   162,    66,    -1,
      50,    -1,    50,    67,   169,    -1,    72,    -1,    71,    -1,
      50,   159,    -1,    50,   174,   153,   175,   159,    -1,   172,
      -1,   172,    67,   173,    -1,    74,    -1,    75,    -1,    54,
      82,    -1,    55,    82,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const unsigned short yyrline[] =
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
     666,   668,   670,   672,   674,   676,   678,   680,   684,   686,
     688,   693,   694,   696,   705,   706,   708,   714,   720,   726,
     734,   741,   749,   756,   758,   760,   762,   769,   771,   775,
     777,   782,   783,   788,   789,   791,   795,   797,   801,   803,
     808,   809,   813,   815,   819,   822,   825,   830,   844,   846,
     848,   850,   852,   855,   858,   861,   864,   866,   868,   870,
     872,   877,   878,   880,   883,   885,   889,   893,   897,   905,
     913,   915,   919,   922,   926,   930
};
#endif

#if YYDEBUG || YYERROR_VERBOSE
/* YYTNME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals. */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "MODULE", "MAINMODULE", "EXTERN", 
  "READONLY", "INITCALL", "INITNODE", "INITPROC", "PUPABLE", "CHARE", 
  "MAINCHARE", "GROUP", "NODEGROUP", "ARRAY", "MESSAGE", "CLASS", 
  "INCLUDE", "STACKSIZE", "THREADED", "TEMPLATE", "SYNC", "EXCLUSIVE", 
  "IMMEDIATE", "SKIPSCHED", "INLINE", "VIRTUAL", "MIGRATABLE", 
  "CREATEHERE", "CREATEHOME", "NOKEEP", "NOTRACE", "VOID", "CONST", 
  "PACKED", "VARSIZE", "ENTRY", "FOR", "FORALL", "WHILE", "WHEN", 
  "OVERLAP", "ATOMIC", "FORWARD", "IF", "ELSE", "CONNECT", "PUBLISHES", 
  "PYTHON", "IDENT", "NUMBER", "LITERAL", "CPROGRAM", "HASHIF", 
  "HASHIFDEF", "INT", "LONG", "SHORT", "CHAR", "FLOAT", "DOUBLE", 
  "UNSIGNED", "';'", "':'", "'{'", "'}'", "','", "'<'", "'>'", "'*'", 
  "'('", "')'", "'&'", "'['", "']'", "'='", "$accept", "File", 
  "ModuleEList", "OptExtern", "OptSemiColon", "Name", "QualName", 
  "Module", "ConstructEList", "ConstructList", "Construct", "TParam", 
  "TParamList", "TParamEList", "OptTParams", "BuiltinType", "NamedType", 
  "QualNamedType", "SimpleType", "OnePtrType", "PtrType", "FuncType", 
  "BaseType", "Type", "ArrayDim", "Dim", "DimList", "Readonly", 
  "ReadonlyMsg", "OptVoid", "MAttribs", "MAttribList", "MAttrib", 
  "CAttribs", "CAttribList", "PythonOptions", "ArrayAttrib", 
  "ArrayAttribs", "ArrayAttribList", "CAttrib", "Var", "VarList", 
  "Message", "OptBaseList", "BaseList", "Chare", "Group", "NodeGroup", 
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
static const unsigned short yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
     315,   316,   317,    59,    58,   123,   125,    44,    60,    62,
      42,    40,    41,    38,    91,    93,    61
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const unsigned char yyr1[] =
{
       0,    77,    78,    79,    79,    80,    80,    81,    81,    82,
      83,    83,    84,    84,    85,    85,    86,    86,    87,    87,
      87,    87,    87,    87,    87,    87,    87,    87,    87,    88,
      88,    88,    89,    89,    90,    90,    91,    91,    92,    92,
      92,    92,    92,    92,    92,    92,    92,    92,    92,    92,
      92,    92,    92,    93,    94,    95,    95,    96,    97,    97,
      98,    99,    99,    99,    99,    99,    99,   100,   100,   101,
     101,   102,   103,   103,   104,   105,   106,   106,   107,   107,
     108,   108,   109,   109,   110,   110,   111,   111,   112,   112,
     113,   114,   114,   115,   115,   116,   116,   117,   118,   118,
     119,   119,   120,   120,   121,   121,   122,   122,   123,   124,
     125,   125,   126,   126,   127,   127,   128,   129,   130,   131,
     131,   132,   132,   133,   133,   133,   134,   134,   134,   135,
     135,   136,   137,   137,   137,   137,   137,   138,   138,   139,
     139,   140,   140,   140,   140,   140,   140,   140,   141,   141,
     141,   141,   142,   142,   143,   143,   144,   145,   145,   146,
     146,   147,   147,   148,   148,   149,   149,   150,   150,   150,
     150,   150,   150,   150,   150,   150,   150,   150,   151,   151,
     151,   152,   152,   152,   153,   153,   153,   153,   153,   153,
     154,   155,   156,   157,   157,   157,   157,   158,   158,   159,
     159,   160,   160,   161,   161,   161,   162,   162,   163,   163,
     164,   164,   165,   165,   166,   166,   167,   167,   167,   167,
     167,   167,   167,   167,   167,   167,   167,   167,   167,   167,
     167,   168,   168,   168,   169,   169,   170,   171,   172,   172,
     173,   173,   174,   175,   176,   177
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
       1,     1,     1,     1,     1,     1,     1,     2,     1,     1,
       1,     0,     1,     3,     0,     1,     5,     5,     5,     4,
       3,     1,     1,     1,     2,     4,     3,     1,     3,     3,
       2,     0,     3,     0,     1,     3,     1,     2,     1,     2,
       0,     4,     1,     3,     1,     0,     6,     8,     4,     3,
       5,     4,    11,     9,    12,    14,     6,     8,     5,     7,
       3,     0,     2,     4,     1,     3,     1,     1,     2,     5,
       1,     3,     1,     1,     2,     2
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const unsigned char yydefact[] =
{
       3,     0,     0,     0,     2,     3,     9,     0,     0,     1,
       4,    14,     5,    12,    13,     6,     0,     0,     0,     0,
       5,    27,    28,   244,   245,     0,    76,    76,    76,     0,
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
       0,     0,     7,   140,   157,     0,     0,   193,   184,   197,
       0,   167,   168,   169,   174,   175,   176,   170,   171,   172,
     173,    88,     0,   165,    52,    10,     0,     0,   162,     0,
     138,     0,   120,   194,   185,   184,     0,     0,    60,    89,
     177,   164,     0,     0,   203,     0,    97,   190,     0,   184,
     181,   184,     0,   196,   198,   166,   200,     0,     0,     0,
       0,     0,     0,   215,     0,     0,     0,     0,   160,   204,
     201,   179,   178,   180,   195,     0,   182,     0,     0,   184,
     199,   237,   184,     0,   184,     0,   240,     0,     0,   214,
       0,   234,     0,   184,     0,     0,   206,     0,   203,   184,
     181,   184,   184,   189,     0,     0,     0,   242,   238,   184,
       0,     0,   219,     0,   208,   191,   184,     0,   230,     0,
       0,   205,   207,     0,   159,   187,   183,   188,   186,   184,
       0,   236,     0,     0,   241,   218,     0,   221,   209,     0,
     235,     0,     0,   202,     0,   184,     0,   228,   243,     0,
     220,   192,   210,     0,   231,     0,   184,     0,     0,   239,
       0,   216,     0,     0,   226,   184,     0,   184,   229,     0,
     231,     0,   232,     0,     0,     0,   212,     0,   227,     0,
     217,     0,   223,   184,     0,   211,   233,     0,     0,   213,
     222,     0,     0,   224,     0,   225
};

/* YYDEFGOTO[NTERM-NUM]. */
static const short yydefgoto[] =
{
      -1,     3,     4,    18,   113,   141,    66,     5,    13,    19,
      20,   180,   181,   182,   124,    67,   235,    68,    69,    70,
      71,    72,    73,   248,   229,   186,   187,    41,    42,    76,
      90,   157,   158,    82,   139,   330,   149,    87,   150,   140,
     249,   250,    43,   196,   236,    44,    45,    46,    88,    47,
     105,   106,   107,   108,   109,   252,   211,   165,   166,    48,
      49,   239,   272,   273,    51,    52,    80,    93,   274,   275,
     319,   291,   312,   313,   364,   367,   326,   298,   406,   442,
     299,   300,   334,   388,   358,   385,   403,   451,   467,   380,
     386,   454,   382,   422,   372,   376,   377,   399,   439,    21,
      22
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -399
static const short yypact[] =
{
     120,   -43,   -43,    82,  -399,   120,  -399,    48,    48,  -399,
    -399,  -399,    60,  -399,  -399,  -399,   -43,   -43,   242,    34,
      60,  -399,  -399,  -399,  -399,   264,    77,    77,    77,    95,
      88,    88,    88,    88,   119,   123,   -43,    99,    81,    60,
    -399,    96,   103,   131,  -399,  -399,  -399,  -399,   166,  -399,
    -399,   135,   136,   139,  -399,   344,  -399,   326,  -399,  -399,
      10,  -399,  -399,  -399,  -399,   132,    33,  -399,  -399,   134,
     155,   157,  -399,   -11,    95,  -399,    95,    95,    95,     5,
     142,     3,   -43,   -43,   -43,   -43,    54,   154,   162,    92,
     -43,   143,  -399,   174,   282,   163,  -399,  -399,  -399,    88,
      88,    88,    88,   154,   123,  -399,  -399,  -399,  -399,  -399,
    -399,  -399,  -399,  -399,   168,   -25,  -399,  -399,  -399,    75,
    -399,  -399,   176,   313,  -399,  -399,  -399,  -399,  -399,   171,
    -399,   -31,     4,    13,    24,    95,  -399,  -399,  -399,   186,
     197,   205,   201,   201,   201,   201,  -399,   -43,   199,   212,
     206,    97,   -43,   243,   -43,  -399,  -399,   218,   227,   235,
    -399,  -399,   -43,   -35,   -43,   234,   233,   139,   -43,   -43,
     -43,   -43,   -43,   -43,   -43,  -399,  -399,   253,  -399,  -399,
     237,  -399,   236,  -399,   -43,   105,   239,  -399,    77,    77,
      77,  -399,  -399,     3,  -399,   -43,    56,    56,    56,    56,
     252,  -399,   243,  -399,   201,   201,  -399,    92,   326,   232,
     113,  -399,   241,   282,  -399,  -399,   201,   201,   201,   201,
     201,    78,  -399,  -399,   313,  -399,   247,  -399,   245,   255,
    -399,   259,   261,   262,  -399,   268,  -399,  -399,   225,  -399,
    -399,  -399,  -399,  -399,  -399,    56,    56,  -399,   -43,   326,
     270,   326,  -399,  -399,  -399,  -399,  -399,    56,    56,    56,
      56,    56,  -399,   326,  -399,   266,  -399,  -399,  -399,  -399,
     -43,   274,   283,  -399,   225,   287,  -399,  -399,   284,  -399,
    -399,  -399,  -399,  -399,  -399,  -399,  -399,   296,   326,  -399,
     246,   357,   139,  -399,  -399,   286,   303,   -43,   -16,   301,
     306,  -399,  -399,  -399,  -399,  -399,  -399,  -399,  -399,  -399,
    -399,   327,   305,   314,   339,   320,   321,   134,  -399,   -43,
    -399,   330,  -399,   -38,    -8,   -16,   322,   326,  -399,  -399,
    -399,  -399,   246,   295,   179,   321,  -399,  -399,    87,   -16,
     342,   -16,   324,  -399,  -399,  -399,  -399,   336,   328,   335,
     328,   348,   345,   359,   362,   328,   367,   469,  -399,  -399,
     409,  -399,  -399,   245,  -399,   374,   375,   369,   368,   -16,
    -399,  -399,   -16,   394,   -16,    20,   378,   382,   469,  -399,
     381,   400,   385,   -16,   419,   404,   469,   395,   179,   -16,
     342,   -16,   -16,  -399,   410,   397,   402,  -399,  -399,   -16,
     348,   244,  -399,   430,   469,  -399,   -16,   362,  -399,   402,
     321,  -399,  -399,   424,  -399,  -399,  -399,  -399,  -399,   -16,
     328,  -399,   392,   423,  -399,  -399,   433,  -399,  -399,   434,
    -399,   411,   405,  -399,   438,   -16,   469,  -399,  -399,   321,
    -399,  -399,   454,   469,   457,   381,   -16,   442,   451,  -399,
     447,  -399,   453,   421,  -399,   -16,   402,   -16,  -399,   470,
     457,   469,  -399,   455,   440,   437,   456,   452,  -399,   459,
    -399,   469,  -399,   -16,   470,  -399,  -399,   460,   402,  -399,
    -399,   450,   469,  -399,   461,  -399
};

/* YYPGOTO[NTERM-NUM].  */
static const short yypgoto[] =
{
    -399,  -399,   517,  -399,  -162,    -1,   -27,   510,   521,     1,
    -399,  -399,   307,  -399,   389,  -399,   -65,  -399,   -52,   248,
    -399,   -88,   475,   -21,  -399,  -399,   347,  -399,  -399,   -14,
     431,   329,  -399,    85,   341,  -399,  -399,   449,   338,  -399,
    -399,  -219,  -399,    -9,   271,  -399,  -399,  -399,   -58,  -399,
    -399,  -399,  -399,  -399,  -399,  -399,   331,  -399,   325,  -399,
    -399,   -45,   272,   524,  -399,  -399,   412,  -399,  -399,  -399,
    -399,  -399,   213,  -399,  -399,   158,  -313,  -399,   104,  -399,
    -399,  -237,  -327,  -399,   156,  -362,  -399,  -399,    76,  -399,
    -324,    91,   145,  -398,  -328,  -399,   153,  -399,  -399,  -399,
    -399
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -162
static const short yytable[] =
{
       7,     8,    79,   114,    74,   215,   163,     6,   360,   128,
     359,   431,   342,    77,    78,    23,    24,   142,   143,   144,
     145,    54,   374,   128,   412,   159,   365,   383,   368,   152,
     279,   137,  -123,   122,  -123,    91,   337,   324,   338,   426,
      95,   210,   428,   185,   287,   172,   129,   131,   398,   132,
     133,   134,   138,   402,   404,   325,   393,   339,   464,   394,
     129,   396,   130,   340,   359,    15,   341,   116,   122,   122,
     409,   117,   135,   164,   448,   188,   415,   122,   417,   418,
     481,   452,     9,   432,   189,   148,   423,   204,   122,   205,
     344,   333,   435,   429,   397,   190,   347,   122,   437,   469,
      53,   123,   183,   146,     6,   147,   434,   444,    79,   477,
      75,    11,   449,    12,    16,    17,    83,    84,    85,   237,
     484,   238,   447,     1,     2,   163,   -16,   155,   156,   462,
     320,   175,   176,   456,   197,   198,   199,    58,   361,   362,
     472,   262,   463,   263,   465,    58,   200,     6,   147,    94,
     148,    92,   240,   241,   242,    58,   227,   483,   228,    96,
     478,   209,    81,   212,   253,   254,    97,   216,   217,   218,
     219,   220,   221,   222,   231,   232,   233,    99,   100,   101,
     102,   103,   104,   226,   168,   169,   170,   171,   118,   119,
     120,   121,   164,    86,    98,   245,   246,    89,   110,   111,
     276,   277,   112,   183,   125,   136,   160,   257,   258,   259,
     260,   261,   282,   283,   284,   285,   286,   348,   349,   350,
     351,   352,   353,   354,   355,   126,   356,   127,   151,   167,
     281,    25,    26,    27,    28,    29,   153,   161,   174,   317,
     177,   184,    36,    37,   357,     1,     2,   278,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    36,
      37,   192,   271,    38,   193,   195,   301,   297,   302,   303,
     304,   305,   306,   123,   201,   307,   308,   309,   310,   202,
      55,   203,   348,   349,   350,   351,   352,   353,   354,   355,
     316,   356,   146,   206,   207,   311,   323,    56,    57,   162,
     208,   213,   214,   223,   224,   225,   297,    39,   251,   122,
     425,   363,   297,   185,    58,    56,    57,   210,   335,   265,
      59,    60,    61,    62,    63,    64,    65,   243,    56,    57,
     266,   267,    58,   268,   269,   270,   280,   288,    59,    60,
      61,    62,    63,    64,    65,    58,    56,    57,   290,   292,
     294,    59,    60,    61,    62,    63,    64,    65,   295,    56,
      57,   321,   296,    58,   178,   179,   322,   346,   327,    59,
      60,    61,    62,    63,    64,    65,    58,    56,   328,   329,
     331,   332,    59,    60,    61,    62,    63,    64,    65,  -161,
     314,    -9,   333,   336,    58,   366,   369,   343,   375,   371,
      59,    60,    61,    62,    63,    64,    65,   315,   370,   373,
     378,   379,   381,    59,    60,    61,    62,    63,    64,    65,
     348,   349,   350,   351,   352,   353,   354,   355,   387,   356,
     348,   349,   350,   351,   352,   353,   354,   355,   384,   356,
     389,   391,   390,   392,   395,   400,   405,   401,   408,   348,
     349,   350,   351,   352,   353,   354,   355,   436,   356,   348,
     349,   350,   351,   352,   353,   354,   355,   407,   356,   410,
     411,   413,   420,   419,   421,   433,   443,   445,   348,   349,
     350,   351,   352,   353,   354,   355,   461,   356,   348,   349,
     350,   351,   352,   353,   354,   355,   427,   356,   438,   440,
     441,   446,   450,   453,   473,   471,   457,   348,   349,   350,
     351,   352,   353,   354,   355,   482,   356,   458,   459,   460,
     466,   470,    10,   474,   475,   476,   480,   485,    40,    14,
     194,   264,   115,   230,   234,   173,   247,   154,   256,   318,
     244,   289,    50,   255,   414,   345,   293,   191,   416,   455,
     479,   468,   430,   424
};

static const unsigned short yycheck[] =
{
       1,     2,    29,    55,    25,   167,    94,    50,   335,    34,
     334,   409,   325,    27,    28,    16,    17,    82,    83,    84,
      85,    20,   350,    34,   386,    90,   339,   355,   341,    87,
     249,    28,    67,    64,    69,    36,    74,    53,    76,   401,
      39,    76,   404,    74,   263,   103,    71,    74,   375,    76,
      77,    78,    49,   377,   378,    71,   369,    65,   456,   372,
      71,   374,    73,    71,   388,     5,    74,    57,    64,    64,
     383,    61,    67,    94,   436,    71,   389,    64,   391,   392,
     478,   443,     0,   410,    71,    86,   399,   152,    64,   154,
     327,    71,   420,   406,    74,    71,   333,    64,   422,   461,
      66,    68,   123,    49,    50,    51,   419,   431,   135,   471,
      33,    63,   439,    65,    54,    55,    31,    32,    33,    63,
     482,    65,   435,     3,     4,   213,    66,    35,    36,   453,
     292,    56,    57,   446,   143,   144,   145,    50,    51,    52,
     464,    63,   455,    65,   457,    50,   147,    50,    51,    68,
     151,    52,   197,   198,   199,    50,    51,   481,   185,    63,
     473,   162,    74,   164,    51,    52,    63,   168,   169,   170,
     171,   172,   173,   174,   188,   189,   190,    11,    12,    13,
      14,    15,    16,   184,    99,   100,   101,   102,    56,    57,
      58,    59,   213,    74,    63,   204,   205,    74,    63,    63,
     245,   246,    63,   224,    70,    63,    63,   216,   217,   218,
     219,   220,   257,   258,   259,   260,   261,    38,    39,    40,
      41,    42,    43,    44,    45,    70,    47,    70,    74,    66,
     251,     6,     7,     8,     9,    10,    74,    63,    70,   291,
      64,    70,    17,    18,    65,     3,     4,   248,     6,     7,
       8,     9,    10,    11,    12,    13,    14,    15,    16,    17,
      18,    75,    37,    21,    67,    64,    20,   288,    22,    23,
      24,    25,    26,    68,    75,    29,    30,    31,    32,    67,
      16,    75,    38,    39,    40,    41,    42,    43,    44,    45,
     291,    47,    49,    75,    67,    49,   297,    33,    34,    17,
      65,    67,    69,    50,    67,    69,   327,    65,    76,    64,
      66,   338,   333,    74,    50,    33,    34,    76,   319,    72,
      56,    57,    58,    59,    60,    61,    62,    75,    33,    34,
      75,    72,    50,    72,    72,    67,    66,    71,    56,    57,
      58,    59,    60,    61,    62,    50,    33,    34,    74,    66,
      63,    56,    57,    58,    59,    60,    61,    62,    74,    33,
      34,    75,    66,    50,    51,    52,    63,    72,    67,    56,
      57,    58,    59,    60,    61,    62,    50,    33,    72,    52,
      75,    67,    56,    57,    58,    59,    60,    61,    62,    50,
      33,    71,    71,    63,    50,    53,    72,    75,    50,    71,
      56,    57,    58,    59,    60,    61,    62,    50,    72,    74,
      65,    52,    50,    56,    57,    58,    59,    60,    61,    62,
      38,    39,    40,    41,    42,    43,    44,    45,    19,    47,
      38,    39,    40,    41,    42,    43,    44,    45,    71,    47,
      66,    72,    67,    75,    50,    67,    65,    65,    63,    38,
      39,    40,    41,    42,    43,    44,    45,    65,    47,    38,
      39,    40,    41,    42,    43,    44,    45,    67,    47,    50,
      66,    76,    75,    63,    72,    51,    65,    72,    38,    39,
      40,    41,    42,    43,    44,    45,    65,    47,    38,    39,
      40,    41,    42,    43,    44,    45,    66,    47,    75,    66,
      66,    63,    48,    46,    67,    65,    64,    38,    39,    40,
      41,    42,    43,    44,    45,    65,    47,    66,    71,    66,
      50,    66,     5,    67,    72,    66,    66,    66,    18,     8,
     141,   224,    57,   186,   193,   104,   207,    88,   213,   291,
     202,   270,    18,   212,   388,   332,   274,   135,   390,   445,
     474,   460,   407,   400
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const unsigned char yystos[] =
{
       0,     3,     4,    78,    79,    84,    50,    82,    82,     0,
      79,    63,    65,    85,    85,     5,    54,    55,    80,    86,
      87,   176,   177,    82,    82,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    17,    18,    21,    65,
      84,   104,   105,   119,   122,   123,   124,   126,   136,   137,
     140,   141,   142,    66,    86,    16,    33,    34,    50,    56,
      57,    58,    59,    60,    61,    62,    83,    92,    94,    95,
      96,    97,    98,    99,   100,    33,   106,   106,   106,    83,
     143,    74,   110,   110,   110,   110,    74,   114,   125,    74,
     107,    82,    52,   144,    68,    86,    63,    63,    63,    11,
      12,    13,    14,    15,    16,   127,   128,   129,   130,   131,
      63,    63,    63,    81,    95,    99,    57,    61,    56,    57,
      58,    59,    64,    68,    91,    70,    70,    70,    34,    71,
      73,    83,    83,    83,    83,    67,    63,    28,    49,   111,
     116,    82,    93,    93,    93,    93,    49,    51,    82,   113,
     115,    74,   125,    74,   114,    35,    36,   108,   109,    93,
      63,    63,    17,    98,   100,   134,   135,    66,   110,   110,
     110,   110,   125,   107,    70,    56,    57,    64,    51,    52,
      88,    89,    90,   100,    70,    74,   102,   103,    71,    71,
      71,   143,    75,    67,    91,    64,   120,   120,   120,   120,
      82,    75,    67,    75,    93,    93,    75,    67,    65,    82,
      76,   133,    82,    67,    69,    81,    82,    82,    82,    82,
      82,    82,    82,    50,    67,    69,    82,    51,    83,   101,
     103,   106,   106,   106,   111,    93,   121,    63,    65,   138,
     138,   138,   138,    75,   115,   120,   120,   108,   100,   117,
     118,    76,   132,    51,    52,   133,   135,   120,   120,   120,
     120,   120,    63,    65,    89,    72,    75,    72,    72,    72,
      67,    37,   139,   140,   145,   146,   138,   138,    82,   118,
      66,   100,   138,   138,   138,   138,   138,   118,    71,   121,
      74,   148,    66,   139,    63,    74,    66,   100,   154,   157,
     158,    20,    22,    23,    24,    25,    26,    29,    30,    31,
      32,    49,   149,   150,    33,    50,    82,    95,    96,   147,
      81,    75,    63,    82,    53,    71,   153,    67,    72,    52,
     112,    75,    67,    71,   159,    82,    63,    74,    76,    65,
      71,    74,   153,    75,   158,   149,    72,   158,    38,    39,
      40,    41,    42,    43,    44,    45,    47,    65,   161,   167,
     159,    51,    52,    83,   151,   153,    53,   152,   153,    72,
      72,    71,   171,    74,   171,    50,   172,   173,    65,    52,
     166,    50,   169,   171,    71,   162,   167,    19,   160,    66,
      67,    72,    75,   153,   153,    50,   153,    74,   159,   174,
      67,    65,   167,   163,   167,    65,   155,    67,    63,   153,
      50,    66,   162,    76,   161,   153,   152,   153,   153,    63,
      75,    72,   170,   153,   173,    66,   162,    66,   162,   153,
     169,   170,   159,    51,   153,   171,    65,   167,    75,   175,
      66,    66,   156,    65,   167,    72,    63,   153,   162,   159,
      48,   164,   162,    46,   168,   155,   153,    64,    66,    71,
      66,    65,   167,   153,   170,   153,    50,   165,   168,   162,
      66,    65,   167,    67,    67,    72,    66,   162,   153,   165,
      66,   170,    65,   167,   162,    66
};

#if ! defined (YYSIZE_T) && defined (__SIZE_TYPE__)
# define YYSIZE_T __SIZE_TYPE__
#endif
#if ! defined (YYSIZE_T) && defined (size_t)
# define YYSIZE_T size_t
#endif
#if ! defined (YYSIZE_T)
# if defined (__STDC__) || defined (__cplusplus)
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# endif
#endif
#if ! defined (YYSIZE_T)
# define YYSIZE_T unsigned int
#endif

#define yyerrok		(yyerrstatus = 0)
#define yyclearin	(yychar = YYEMPTY)
#define YYEMPTY		(-2)
#define YYEOF		0

#define YYACCEPT	goto yyacceptlab
#define YYABORT		goto yyabortlab
#define YYERROR		goto yyerrlab1

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
    { 								\
      yyerror ("syntax error: cannot back up");\
      YYERROR;							\
    }								\
while (0)

#define YYTERROR	1
#define YYERRCODE	256

/* YYLLOC_DEFAULT -- Compute the default location (before the actions
   are run).  */

#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)         \
  Current.first_line   = Rhs[1].first_line;      \
  Current.first_column = Rhs[1].first_column;    \
  Current.last_line    = Rhs[N].last_line;       \
  Current.last_column  = Rhs[N].last_column;
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

# define YYDSYMPRINT(Args)			\
do {						\
  if (yydebug)					\
    yysymprint Args;				\
} while (0)

# define YYDSYMPRINTF(Title, Token, Value, Location)		\
do {								\
  if (yydebug)							\
    {								\
      YYFPRINTF (stderr, "%s ", Title);				\
      yysymprint (stderr, 					\
                  Token, Value);	\
      YYFPRINTF (stderr, "\n");					\
    }								\
} while (0)

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (cinluded).                                                   |
`------------------------------------------------------------------*/

#if defined (__STDC__) || defined (__cplusplus)
static void
yy_stack_print (short *bottom, short *top)
#else
static void
yy_stack_print (bottom, top)
    short *bottom;
    short *top;
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
  unsigned int yylineno = yyrline[yyrule];
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %u), ",
             yyrule - 1, yylineno);
  /* Print the symbols being reduced, and their result.  */
  for (yyi = yyprhs[yyrule]; 0 <= yyrhs[yyi]; yyi++)
    YYFPRINTF (stderr, "%s ", yytname [yyrhs[yyi]]);
  YYFPRINTF (stderr, "-> %s\n", yytname [yyr1[yyrule]]);
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
# define YYDSYMPRINT(Args)
# define YYDSYMPRINTF(Title, Token, Value, Location)
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
   SIZE_MAX < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#if YYMAXDEPTH == 0
# undef YYMAXDEPTH
#endif

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
  register const char *yys = yystr;

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
  register char *yyd = yydest;
  register const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

#endif /* !YYERROR_VERBOSE */



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
    {
      YYFPRINTF (yyoutput, "token %s (", yytname[yytype]);
# ifdef YYPRINT
      YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# endif
    }
  else
    YYFPRINTF (yyoutput, "nterm %s (", yytname[yytype]);

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
yydestruct (int yytype, YYSTYPE *yyvaluep)
#else
static void
yydestruct (yytype, yyvaluep)
    int yytype;
    YYSTYPE *yyvaluep;
#endif
{
  /* Pacify ``unused variable'' warnings.  */
  (void) yyvaluep;

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



/* The lookahead symbol.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
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

#endif
#endif
{
  
  register int yystate;
  register int yyn;
  int yyresult;
  /* Number of tokens to shift before error messages enabled.  */
  int yyerrstatus;
  /* Lookahead token as an internal (translated) token number.  */
  int yytoken = 0;

  /* Three stacks and their tools:
     `yyss': related to states,
     `yyvs': related to semantic values,
     `yyls': related to locations.

     Refer to the stacks thru separate pointers, to allow yyoverflow
     to reallocate them elsewhere.  */

  /* The state stack.  */
  short	yyssa[YYINITDEPTH];
  short *yyss = yyssa;
  register short *yyssp;

  /* The semantic value stack.  */
  YYSTYPE yyvsa[YYINITDEPTH];
  YYSTYPE *yyvs = yyvsa;
  register YYSTYPE *yyvsp;



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
	short *yyss1 = yyss;


	/* Each stack pointer address is followed by the size of the
	   data in use in that stack, in bytes.  This used to be a
	   conditional around just the two extra args, but that might
	   be undefined if yyoverflow is a macro.  */
	yyoverflow ("parser stack overflow",
		    &yyss1, yysize * sizeof (*yyssp),
		    &yyvs1, yysize * sizeof (*yyvsp),

		    &yystacksize);

	yyss = yyss1;
	yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyoverflowlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
	goto yyoverflowlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
	yystacksize = YYMAXDEPTH;

      {
	short *yyss1 = yyss;
	union yyalloc *yyptr =
	  (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
	if (! yyptr)
	  goto yyoverflowlab;
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
/* Read a lookahead token if we need one and don't already have one.  */
/* yyresume: */

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
      YYDSYMPRINTF ("Next token is", yytoken, &yylval, &yylloc);
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

  /* Shift the lookahead token.  */
  YYDPRINTF ((stderr, "Shifting token %s, ", yytname[yytoken]));

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
    { yyval.modlist = yyvsp[0].modlist; modlist = yyvsp[0].modlist; }
    break;

  case 3:
#line 138 "xi-grammar.y"
    { 
		  yyval.modlist = 0; 
		}
    break;

  case 4:
#line 142 "xi-grammar.y"
    { yyval.modlist = new ModuleList(lineno, yyvsp[-1].module, yyvsp[0].modlist); }
    break;

  case 5:
#line 146 "xi-grammar.y"
    { yyval.intval = 0; }
    break;

  case 6:
#line 148 "xi-grammar.y"
    { yyval.intval = 1; }
    break;

  case 7:
#line 152 "xi-grammar.y"
    { yyval.intval = 0; }
    break;

  case 8:
#line 154 "xi-grammar.y"
    { yyval.intval = 1; }
    break;

  case 9:
#line 158 "xi-grammar.y"
    { yyval.strval = yyvsp[0].strval; }
    break;

  case 10:
#line 162 "xi-grammar.y"
    { yyval.strval = yyvsp[0].strval; }
    break;

  case 11:
#line 164 "xi-grammar.y"
    {
		  char *tmp = new char[strlen(yyvsp[-3].strval)+strlen(yyvsp[0].strval)+3];
		  sprintf(tmp,"%s::%s", yyvsp[-3].strval, yyvsp[0].strval);
		  yyval.strval = tmp;
		}
    break;

  case 12:
#line 172 "xi-grammar.y"
    { 
		    yyval.module = new Module(lineno, yyvsp[-1].strval, yyvsp[0].conslist); 
		}
    break;

  case 13:
#line 176 "xi-grammar.y"
    {  
		    yyval.module = new Module(lineno, yyvsp[-1].strval, yyvsp[0].conslist); 
		    yyval.module->setMain();
		}
    break;

  case 14:
#line 183 "xi-grammar.y"
    { yyval.conslist = 0; }
    break;

  case 15:
#line 185 "xi-grammar.y"
    { yyval.conslist = yyvsp[-2].conslist; }
    break;

  case 16:
#line 189 "xi-grammar.y"
    { yyval.conslist = 0; }
    break;

  case 17:
#line 191 "xi-grammar.y"
    { yyval.conslist = new ConstructList(lineno, yyvsp[-1].construct, yyvsp[0].conslist); }
    break;

  case 18:
#line 195 "xi-grammar.y"
    { if(yyvsp[-2].conslist) yyvsp[-2].conslist->setExtern(yyvsp[-4].intval); yyval.construct = yyvsp[-2].conslist; }
    break;

  case 19:
#line 197 "xi-grammar.y"
    { yyvsp[0].module->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].module; }
    break;

  case 20:
#line 199 "xi-grammar.y"
    { yyvsp[0].member->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].member; }
    break;

  case 21:
#line 201 "xi-grammar.y"
    { yyvsp[-1].message->setExtern(yyvsp[-2].intval); yyval.construct = yyvsp[-1].message; }
    break;

  case 22:
#line 203 "xi-grammar.y"
    { yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; }
    break;

  case 23:
#line 205 "xi-grammar.y"
    { yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; }
    break;

  case 24:
#line 207 "xi-grammar.y"
    { yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; }
    break;

  case 25:
#line 209 "xi-grammar.y"
    { yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; }
    break;

  case 26:
#line 211 "xi-grammar.y"
    { yyvsp[0].templat->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].templat; }
    break;

  case 27:
#line 213 "xi-grammar.y"
    { yyval.construct = NULL; }
    break;

  case 28:
#line 215 "xi-grammar.y"
    { yyval.construct = NULL; }
    break;

  case 29:
#line 219 "xi-grammar.y"
    { yyval.tparam = new TParamType(yyvsp[0].type); }
    break;

  case 30:
#line 221 "xi-grammar.y"
    { yyval.tparam = new TParamVal(yyvsp[0].strval); }
    break;

  case 31:
#line 223 "xi-grammar.y"
    { yyval.tparam = new TParamVal(yyvsp[0].strval); }
    break;

  case 32:
#line 227 "xi-grammar.y"
    { yyval.tparlist = new TParamList(yyvsp[0].tparam); }
    break;

  case 33:
#line 229 "xi-grammar.y"
    { yyval.tparlist = new TParamList(yyvsp[-2].tparam, yyvsp[0].tparlist); }
    break;

  case 34:
#line 233 "xi-grammar.y"
    { yyval.tparlist = 0; }
    break;

  case 35:
#line 235 "xi-grammar.y"
    { yyval.tparlist = yyvsp[0].tparlist; }
    break;

  case 36:
#line 239 "xi-grammar.y"
    { yyval.tparlist = 0; }
    break;

  case 37:
#line 241 "xi-grammar.y"
    { yyval.tparlist = yyvsp[-1].tparlist; }
    break;

  case 38:
#line 245 "xi-grammar.y"
    { yyval.type = new BuiltinType("int"); }
    break;

  case 39:
#line 247 "xi-grammar.y"
    { yyval.type = new BuiltinType("long"); }
    break;

  case 40:
#line 249 "xi-grammar.y"
    { yyval.type = new BuiltinType("short"); }
    break;

  case 41:
#line 251 "xi-grammar.y"
    { yyval.type = new BuiltinType("char"); }
    break;

  case 42:
#line 253 "xi-grammar.y"
    { yyval.type = new BuiltinType("unsigned int"); }
    break;

  case 43:
#line 255 "xi-grammar.y"
    { yyval.type = new BuiltinType("unsigned long"); }
    break;

  case 44:
#line 257 "xi-grammar.y"
    { yyval.type = new BuiltinType("unsigned long"); }
    break;

  case 45:
#line 259 "xi-grammar.y"
    { yyval.type = new BuiltinType("unsigned long long"); }
    break;

  case 46:
#line 261 "xi-grammar.y"
    { yyval.type = new BuiltinType("unsigned short"); }
    break;

  case 47:
#line 263 "xi-grammar.y"
    { yyval.type = new BuiltinType("unsigned char"); }
    break;

  case 48:
#line 265 "xi-grammar.y"
    { yyval.type = new BuiltinType("long long"); }
    break;

  case 49:
#line 267 "xi-grammar.y"
    { yyval.type = new BuiltinType("float"); }
    break;

  case 50:
#line 269 "xi-grammar.y"
    { yyval.type = new BuiltinType("double"); }
    break;

  case 51:
#line 271 "xi-grammar.y"
    { yyval.type = new BuiltinType("long double"); }
    break;

  case 52:
#line 273 "xi-grammar.y"
    { yyval.type = new BuiltinType("void"); }
    break;

  case 53:
#line 276 "xi-grammar.y"
    { yyval.ntype = new NamedType(yyvsp[-1].strval,yyvsp[0].tparlist); }
    break;

  case 54:
#line 277 "xi-grammar.y"
    { yyval.ntype = new NamedType(yyvsp[-1].strval,yyvsp[0].tparlist); }
    break;

  case 55:
#line 280 "xi-grammar.y"
    { yyval.type = yyvsp[0].type; }
    break;

  case 56:
#line 282 "xi-grammar.y"
    { yyval.type = yyvsp[0].ntype; }
    break;

  case 57:
#line 286 "xi-grammar.y"
    { yyval.ptype = new PtrType(yyvsp[-1].type); }
    break;

  case 58:
#line 290 "xi-grammar.y"
    { yyvsp[-1].ptype->indirect(); yyval.ptype = yyvsp[-1].ptype; }
    break;

  case 59:
#line 292 "xi-grammar.y"
    { yyvsp[-1].ptype->indirect(); yyval.ptype = yyvsp[-1].ptype; }
    break;

  case 60:
#line 296 "xi-grammar.y"
    { yyval.ftype = new FuncType(yyvsp[-7].type, yyvsp[-4].strval, yyvsp[-1].plist); }
    break;

  case 61:
#line 300 "xi-grammar.y"
    { yyval.type = yyvsp[0].type; }
    break;

  case 62:
#line 302 "xi-grammar.y"
    { yyval.type = yyvsp[0].ptype; }
    break;

  case 63:
#line 304 "xi-grammar.y"
    { yyval.type = yyvsp[0].ptype; }
    break;

  case 64:
#line 306 "xi-grammar.y"
    { yyval.type = yyvsp[0].ftype; }
    break;

  case 65:
#line 309 "xi-grammar.y"
    { yyval.type = yyvsp[0].type; }
    break;

  case 66:
#line 311 "xi-grammar.y"
    { yyval.type = yyvsp[-1].type; }
    break;

  case 67:
#line 315 "xi-grammar.y"
    { yyval.type = new ReferenceType(yyvsp[-1].type); }
    break;

  case 68:
#line 317 "xi-grammar.y"
    { yyval.type = yyvsp[0].type; }
    break;

  case 69:
#line 321 "xi-grammar.y"
    { yyval.val = new Value(yyvsp[0].strval); }
    break;

  case 70:
#line 323 "xi-grammar.y"
    { yyval.val = new Value(yyvsp[0].strval); }
    break;

  case 71:
#line 327 "xi-grammar.y"
    { yyval.val = yyvsp[-1].val; }
    break;

  case 72:
#line 331 "xi-grammar.y"
    { yyval.vallist = 0; }
    break;

  case 73:
#line 333 "xi-grammar.y"
    { yyval.vallist = new ValueList(yyvsp[-1].val, yyvsp[0].vallist); }
    break;

  case 74:
#line 337 "xi-grammar.y"
    { yyval.readonly = new Readonly(lineno, yyvsp[-2].type, yyvsp[-1].strval, yyvsp[0].vallist); }
    break;

  case 75:
#line 341 "xi-grammar.y"
    { yyval.readonly = new Readonly(lineno, yyvsp[-2].type, yyvsp[0].strval, 0, 1); }
    break;

  case 76:
#line 345 "xi-grammar.y"
    { yyval.intval = 0;}
    break;

  case 77:
#line 347 "xi-grammar.y"
    { yyval.intval = 0;}
    break;

  case 78:
#line 351 "xi-grammar.y"
    { yyval.intval = 0; }
    break;

  case 79:
#line 353 "xi-grammar.y"
    { 
		  /*
		  printf("Warning: Message attributes are being phased out.\n");
		  printf("Warning: Please remove them from interface files.\n");
		  */
		  yyval.intval = yyvsp[-1].intval; 
		}
    break;

  case 80:
#line 363 "xi-grammar.y"
    { yyval.intval = yyvsp[0].intval; }
    break;

  case 81:
#line 365 "xi-grammar.y"
    { yyval.intval = yyvsp[-2].intval | yyvsp[0].intval; }
    break;

  case 82:
#line 369 "xi-grammar.y"
    { yyval.intval = 0; }
    break;

  case 83:
#line 371 "xi-grammar.y"
    { yyval.intval = 0; }
    break;

  case 84:
#line 375 "xi-grammar.y"
    { yyval.cattr = 0; }
    break;

  case 85:
#line 377 "xi-grammar.y"
    { yyval.cattr = yyvsp[-1].cattr; }
    break;

  case 86:
#line 381 "xi-grammar.y"
    { yyval.cattr = yyvsp[0].cattr; }
    break;

  case 87:
#line 383 "xi-grammar.y"
    { yyval.cattr = yyvsp[-2].cattr | yyvsp[0].cattr; }
    break;

  case 88:
#line 387 "xi-grammar.y"
    { python_doc = NULL; yyval.intval = 0; }
    break;

  case 89:
#line 389 "xi-grammar.y"
    { python_doc = yyvsp[0].strval; yyval.intval = 0; }
    break;

  case 90:
#line 393 "xi-grammar.y"
    { yyval.cattr = Chare::CPYTHON; }
    break;

  case 91:
#line 397 "xi-grammar.y"
    { yyval.cattr = 0; }
    break;

  case 92:
#line 399 "xi-grammar.y"
    { yyval.cattr = yyvsp[-1].cattr; }
    break;

  case 93:
#line 403 "xi-grammar.y"
    { yyval.cattr = yyvsp[0].cattr; }
    break;

  case 94:
#line 405 "xi-grammar.y"
    { yyval.cattr = yyvsp[-2].cattr | yyvsp[0].cattr; }
    break;

  case 95:
#line 409 "xi-grammar.y"
    { yyval.cattr = Chare::CMIGRATABLE; }
    break;

  case 96:
#line 411 "xi-grammar.y"
    { yyval.cattr = Chare::CPYTHON; }
    break;

  case 97:
#line 415 "xi-grammar.y"
    { yyval.mv = new MsgVar(yyvsp[-4].type, yyvsp[-3].strval); }
    break;

  case 98:
#line 419 "xi-grammar.y"
    { yyval.mvlist = new MsgVarList(yyvsp[0].mv); }
    break;

  case 99:
#line 421 "xi-grammar.y"
    { yyval.mvlist = new MsgVarList(yyvsp[-1].mv, yyvsp[0].mvlist); }
    break;

  case 100:
#line 425 "xi-grammar.y"
    { yyval.message = new Message(lineno, yyvsp[0].ntype); }
    break;

  case 101:
#line 427 "xi-grammar.y"
    { yyval.message = new Message(lineno, yyvsp[-3].ntype, yyvsp[-1].mvlist); }
    break;

  case 102:
#line 431 "xi-grammar.y"
    { yyval.typelist = 0; }
    break;

  case 103:
#line 433 "xi-grammar.y"
    { yyval.typelist = yyvsp[0].typelist; }
    break;

  case 104:
#line 437 "xi-grammar.y"
    { yyval.typelist = new TypeList(yyvsp[0].ntype); }
    break;

  case 105:
#line 439 "xi-grammar.y"
    { yyval.typelist = new TypeList(yyvsp[-2].ntype, yyvsp[0].typelist); }
    break;

  case 106:
#line 443 "xi-grammar.y"
    { yyval.chare = new Chare(lineno, yyvsp[-3].cattr, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 107:
#line 445 "xi-grammar.y"
    { yyval.chare = new MainChare(lineno, yyvsp[-3].cattr, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 108:
#line 449 "xi-grammar.y"
    { yyval.chare = new Group(lineno, yyvsp[-3].cattr, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 109:
#line 453 "xi-grammar.y"
    { yyval.chare = new NodeGroup(lineno, yyvsp[-3].cattr, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 110:
#line 457 "xi-grammar.y"
    {/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",yyvsp[-2].strval);
			yyval.ntype = new NamedType(buf); 
		}
    break;

  case 111:
#line 463 "xi-grammar.y"
    { yyval.ntype = new NamedType(yyvsp[-1].strval); }
    break;

  case 112:
#line 467 "xi-grammar.y"
    {  yyval.chare = new Array(lineno, yyvsp[-4].cattr, yyvsp[-3].ntype, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 113:
#line 469 "xi-grammar.y"
    {  yyval.chare = new Array(lineno, yyvsp[-3].cattr, yyvsp[-4].ntype, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 114:
#line 473 "xi-grammar.y"
    { yyval.chare = new Chare(lineno, yyvsp[-3].cattr, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist);}
    break;

  case 115:
#line 475 "xi-grammar.y"
    { yyval.chare = new MainChare(lineno, yyvsp[-3].cattr, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 116:
#line 479 "xi-grammar.y"
    { yyval.chare = new Group(lineno, yyvsp[-3].cattr, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 117:
#line 483 "xi-grammar.y"
    { yyval.chare = new NodeGroup( lineno, yyvsp[-3].cattr, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 118:
#line 487 "xi-grammar.y"
    { yyval.chare = new Array( lineno, 0, yyvsp[-3].ntype, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 119:
#line 491 "xi-grammar.y"
    { yyval.message = new Message(lineno, new NamedType(yyvsp[-1].strval)); }
    break;

  case 120:
#line 493 "xi-grammar.y"
    { yyval.message = new Message(lineno, new NamedType(yyvsp[-4].strval), yyvsp[-2].mvlist); }
    break;

  case 121:
#line 497 "xi-grammar.y"
    { yyval.type = 0; }
    break;

  case 122:
#line 499 "xi-grammar.y"
    { yyval.type = yyvsp[0].type; }
    break;

  case 123:
#line 503 "xi-grammar.y"
    { yyval.strval = 0; }
    break;

  case 124:
#line 505 "xi-grammar.y"
    { yyval.strval = yyvsp[0].strval; }
    break;

  case 125:
#line 507 "xi-grammar.y"
    { yyval.strval = yyvsp[0].strval; }
    break;

  case 126:
#line 511 "xi-grammar.y"
    { yyval.tvar = new TType(new NamedType(yyvsp[-1].strval), yyvsp[0].type); }
    break;

  case 127:
#line 513 "xi-grammar.y"
    { yyval.tvar = new TFunc(yyvsp[-1].ftype, yyvsp[0].strval); }
    break;

  case 128:
#line 515 "xi-grammar.y"
    { yyval.tvar = new TName(yyvsp[-2].type, yyvsp[-1].strval, yyvsp[0].strval); }
    break;

  case 129:
#line 519 "xi-grammar.y"
    { yyval.tvarlist = new TVarList(yyvsp[0].tvar); }
    break;

  case 130:
#line 521 "xi-grammar.y"
    { yyval.tvarlist = new TVarList(yyvsp[-2].tvar, yyvsp[0].tvarlist); }
    break;

  case 131:
#line 525 "xi-grammar.y"
    { yyval.tvarlist = yyvsp[-1].tvarlist; }
    break;

  case 132:
#line 529 "xi-grammar.y"
    { yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); }
    break;

  case 133:
#line 531 "xi-grammar.y"
    { yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); }
    break;

  case 134:
#line 533 "xi-grammar.y"
    { yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); }
    break;

  case 135:
#line 535 "xi-grammar.y"
    { yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); }
    break;

  case 136:
#line 537 "xi-grammar.y"
    { yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].message); yyvsp[0].message->setTemplate(yyval.templat); }
    break;

  case 137:
#line 541 "xi-grammar.y"
    { yyval.mbrlist = 0; }
    break;

  case 138:
#line 543 "xi-grammar.y"
    { yyval.mbrlist = yyvsp[-2].mbrlist; }
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
                    yyval.mbrlist = ml; 
		  }
		  else {
		    yyval.mbrlist = 0; 
                  }
		}
    break;

  case 140:
#line 566 "xi-grammar.y"
    { yyval.mbrlist = new MemberList(yyvsp[-1].member, yyvsp[0].mbrlist); }
    break;

  case 141:
#line 570 "xi-grammar.y"
    { yyval.member = yyvsp[-1].readonly; }
    break;

  case 142:
#line 572 "xi-grammar.y"
    { yyval.member = yyvsp[-1].readonly; }
    break;

  case 144:
#line 575 "xi-grammar.y"
    { yyval.member = yyvsp[-1].member; }
    break;

  case 145:
#line 577 "xi-grammar.y"
    { yyval.member = yyvsp[-1].pupable; }
    break;

  case 146:
#line 579 "xi-grammar.y"
    { yyval.member = yyvsp[-1].includeFile; }
    break;

  case 147:
#line 581 "xi-grammar.y"
    { yyval.member = new ClassDeclaration(lineno,yyvsp[-1].strval); }
    break;

  case 148:
#line 585 "xi-grammar.y"
    { yyval.member = new InitCall(lineno, yyvsp[0].strval, 1); }
    break;

  case 149:
#line 587 "xi-grammar.y"
    { yyval.member = new InitCall(lineno, yyvsp[-3].strval, 1); }
    break;

  case 150:
#line 589 "xi-grammar.y"
    { printf("Warning: deprecated use of initcall. Use initnode or initproc instead.\n"); 
		  yyval.member = new InitCall(lineno, yyvsp[0].strval, 1); }
    break;

  case 151:
#line 592 "xi-grammar.y"
    { printf("Warning: deprecated use of initcall. Use initnode or initproc instead.\n");
		  yyval.member = new InitCall(lineno, yyvsp[-3].strval, 1); }
    break;

  case 152:
#line 597 "xi-grammar.y"
    { yyval.member = new InitCall(lineno, yyvsp[0].strval, 0); }
    break;

  case 153:
#line 599 "xi-grammar.y"
    { yyval.member = new InitCall(lineno, yyvsp[-3].strval, 0); }
    break;

  case 154:
#line 603 "xi-grammar.y"
    { yyval.pupable = new PUPableClass(lineno,yyvsp[0].strval,0); }
    break;

  case 155:
#line 605 "xi-grammar.y"
    { yyval.pupable = new PUPableClass(lineno,yyvsp[-2].strval,yyvsp[0].pupable); }
    break;

  case 156:
#line 609 "xi-grammar.y"
    { yyval.includeFile = new IncludeFile(lineno,yyvsp[0].strval); }
    break;

  case 157:
#line 613 "xi-grammar.y"
    { yyval.member = yyvsp[-1].entry; }
    break;

  case 158:
#line 615 "xi-grammar.y"
    { yyval.member = yyvsp[0].member; }
    break;

  case 159:
#line 619 "xi-grammar.y"
    { 
		  if (yyvsp[0].sc != 0) { 
		    yyvsp[0].sc->con1 = new SdagConstruct(SIDENT, yyvsp[-3].strval);
  		    if (yyvsp[-2].plist != 0)
                      yyvsp[0].sc->param = new ParamList(yyvsp[-2].plist);
 		    else 
 	 	      yyvsp[0].sc->param = new ParamList(new Parameter(0, new BuiltinType("void")));
                  }
		  yyval.entry = new Entry(lineno, yyvsp[-5].intval, yyvsp[-4].type, yyvsp[-3].strval, yyvsp[-2].plist, yyvsp[-1].val, yyvsp[0].sc, 0, 0); 
		}
    break;

  case 160:
#line 630 "xi-grammar.y"
    { 
		  if (yyvsp[0].sc != 0) {
		    yyvsp[0].sc->con1 = new SdagConstruct(SIDENT, yyvsp[-2].strval);
		    if (yyvsp[-1].plist != 0)
                      yyvsp[0].sc->param = new ParamList(yyvsp[-1].plist);
		    else
                      yyvsp[0].sc->param = new ParamList(new Parameter(0, new BuiltinType("void")));
                  }
		  yyval.entry = new Entry(lineno, yyvsp[-3].intval,     0, yyvsp[-2].strval, yyvsp[-1].plist,  0, yyvsp[0].sc, 0, 0); 
		}
    break;

  case 161:
#line 643 "xi-grammar.y"
    { yyval.type = new BuiltinType("void"); }
    break;

  case 162:
#line 645 "xi-grammar.y"
    { yyval.type = yyvsp[0].ptype; }
    break;

  case 163:
#line 649 "xi-grammar.y"
    { yyval.intval = 0; }
    break;

  case 164:
#line 651 "xi-grammar.y"
    { yyval.intval = yyvsp[-1].intval; }
    break;

  case 165:
#line 655 "xi-grammar.y"
    { yyval.intval = yyvsp[0].intval; }
    break;

  case 166:
#line 657 "xi-grammar.y"
    { yyval.intval = yyvsp[-2].intval | yyvsp[0].intval; }
    break;

  case 167:
#line 661 "xi-grammar.y"
    { yyval.intval = STHREADED; }
    break;

  case 168:
#line 663 "xi-grammar.y"
    { yyval.intval = SSYNC; }
    break;

  case 169:
#line 665 "xi-grammar.y"
    { yyval.intval = SLOCKED; }
    break;

  case 170:
#line 667 "xi-grammar.y"
    { yyval.intval = SCREATEHERE; }
    break;

  case 171:
#line 669 "xi-grammar.y"
    { yyval.intval = SCREATEHOME; }
    break;

  case 172:
#line 671 "xi-grammar.y"
    { yyval.intval = SNOKEEP; }
    break;

  case 173:
#line 673 "xi-grammar.y"
    { yyval.intval = SNOTRACE; }
    break;

  case 174:
#line 675 "xi-grammar.y"
    { yyval.intval = SIMMEDIATE; }
    break;

  case 175:
#line 677 "xi-grammar.y"
    { yyval.intval = SSKIPSCHED; }
    break;

  case 176:
#line 679 "xi-grammar.y"
    { yyval.intval = SINLINE; }
    break;

  case 177:
#line 681 "xi-grammar.y"
    { yyval.intval = SPYTHON; }
    break;

  case 178:
#line 685 "xi-grammar.y"
    { yyval.val = new Value(yyvsp[0].strval); }
    break;

  case 179:
#line 687 "xi-grammar.y"
    { yyval.val = new Value(yyvsp[0].strval); }
    break;

  case 180:
#line 689 "xi-grammar.y"
    { yyval.val = new Value(yyvsp[0].strval); }
    break;

  case 181:
#line 693 "xi-grammar.y"
    { yyval.strval = ""; }
    break;

  case 182:
#line 695 "xi-grammar.y"
    { yyval.strval = yyvsp[0].strval; }
    break;

  case 183:
#line 697 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen(yyvsp[-2].strval)+strlen(yyvsp[0].strval)+3];
			sprintf(tmp,"%s, %s", yyvsp[-2].strval, yyvsp[0].strval);
			yyval.strval = tmp;
		}
    break;

  case 184:
#line 705 "xi-grammar.y"
    { yyval.strval = ""; }
    break;

  case 185:
#line 707 "xi-grammar.y"
    { yyval.strval = yyvsp[0].strval; }
    break;

  case 186:
#line 709 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen(yyvsp[-4].strval)+strlen(yyvsp[-2].strval)+strlen(yyvsp[0].strval)+3];
			sprintf(tmp,"%s[%s]%s", yyvsp[-4].strval, yyvsp[-2].strval, yyvsp[0].strval);
			yyval.strval = tmp;
		}
    break;

  case 187:
#line 715 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen(yyvsp[-4].strval)+strlen(yyvsp[-2].strval)+strlen(yyvsp[0].strval)+3];
			sprintf(tmp,"%s{%s}%s", yyvsp[-4].strval, yyvsp[-2].strval, yyvsp[0].strval);
			yyval.strval = tmp;
		}
    break;

  case 188:
#line 721 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen(yyvsp[-4].strval)+strlen(yyvsp[-2].strval)+strlen(yyvsp[0].strval)+3];
			sprintf(tmp,"%s(%s)%s", yyvsp[-4].strval, yyvsp[-2].strval, yyvsp[0].strval);
			yyval.strval = tmp;
		}
    break;

  case 189:
#line 727 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen(yyvsp[-2].strval)+strlen(yyvsp[0].strval)+3];
			sprintf(tmp,"(%s)%s", yyvsp[-2].strval, yyvsp[0].strval);
			yyval.strval = tmp;
		}
    break;

  case 190:
#line 735 "xi-grammar.y"
    {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			yyval.pname = new Parameter(lineno, yyvsp[-2].type,yyvsp[-1].strval);
		}
    break;

  case 191:
#line 742 "xi-grammar.y"
    { 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			yyval.intval = 0;
		}
    break;

  case 192:
#line 750 "xi-grammar.y"
    { 
			in_braces=0;
			yyval.intval = 0;
		}
    break;

  case 193:
#line 757 "xi-grammar.y"
    { yyval.pname = new Parameter(lineno, yyvsp[0].type);}
    break;

  case 194:
#line 759 "xi-grammar.y"
    { yyval.pname = new Parameter(lineno, yyvsp[-1].type,yyvsp[0].strval);}
    break;

  case 195:
#line 761 "xi-grammar.y"
    { yyval.pname = new Parameter(lineno, yyvsp[-3].type,yyvsp[-2].strval,0,yyvsp[0].val);}
    break;

  case 196:
#line 763 "xi-grammar.y"
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			yyval.pname = new Parameter(lineno, yyvsp[-2].pname->getType(), yyvsp[-2].pname->getName() ,yyvsp[-1].strval);
		}
    break;

  case 197:
#line 770 "xi-grammar.y"
    { yyval.plist = new ParamList(yyvsp[0].pname); }
    break;

  case 198:
#line 772 "xi-grammar.y"
    { yyval.plist = new ParamList(yyvsp[-2].pname,yyvsp[0].plist); }
    break;

  case 199:
#line 776 "xi-grammar.y"
    { yyval.plist = yyvsp[-1].plist; }
    break;

  case 200:
#line 778 "xi-grammar.y"
    { yyval.plist = 0; }
    break;

  case 201:
#line 782 "xi-grammar.y"
    { yyval.val = 0; }
    break;

  case 202:
#line 784 "xi-grammar.y"
    { yyval.val = new Value(yyvsp[0].strval); }
    break;

  case 203:
#line 788 "xi-grammar.y"
    { yyval.sc = 0; }
    break;

  case 204:
#line 790 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SSDAGENTRY, yyvsp[0].sc); }
    break;

  case 205:
#line 792 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SSDAGENTRY, yyvsp[-1].sc); }
    break;

  case 206:
#line 796 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SSLIST, yyvsp[0].sc); }
    break;

  case 207:
#line 798 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SSLIST, yyvsp[-1].sc, yyvsp[0].sc);  }
    break;

  case 208:
#line 802 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SOLIST, yyvsp[0].sc); }
    break;

  case 209:
#line 804 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SOLIST, yyvsp[-1].sc, yyvsp[0].sc); }
    break;

  case 210:
#line 808 "xi-grammar.y"
    { yyval.sc = 0; }
    break;

  case 211:
#line 810 "xi-grammar.y"
    { yyval.sc = yyvsp[-1].sc; }
    break;

  case 212:
#line 814 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SPUBLISHES, new SdagConstruct(SIDENT, yyvsp[0].strval)); }
    break;

  case 213:
#line 816 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SPUBLISHES, new SdagConstruct(SIDENT, yyvsp[-2].strval), yyvsp[0].sc);  }
    break;

  case 214:
#line 820 "xi-grammar.y"
    { yyval.strval = yyvsp[0].strval; }
    break;

  case 215:
#line 822 "xi-grammar.y"
    { yyval.strval = 0; }
    break;

  case 216:
#line 826 "xi-grammar.y"
    { RemoveSdagComments(yyvsp[-2].strval);
		   yyval.sc = new SdagConstruct(SATOMIC, new XStr(yyvsp[-2].strval), yyvsp[0].sc, 0,0,0,0, 0 ); 
		   if (yyvsp[-4].strval) { yyvsp[-4].strval[strlen(yyvsp[-4].strval)-1]=0; yyval.sc->traceName = new XStr(yyvsp[-4].strval+1); }
		 }
    break;

  case 217:
#line 831 "xi-grammar.y"
    {  
		   in_braces = 0;
		   if ((yyvsp[-4].plist->isVoid() == 0) && (yyvsp[-4].plist->isMessage() == 0))
                   {
		      connectEntries->append(new Entry(0, 0, new BuiltinType("void"), yyvsp[-5].strval, 
	 	 			new ParamList(new Parameter(lineno, new PtrType( 
                                        new NamedType("CkMarshallMsg")), "_msg")), 0, 0, 0, 1, yyvsp[-4].plist));
		   }
		   else  {
		      connectEntries->append(new Entry(0, 0, new BuiltinType("void"), yyvsp[-5].strval, yyvsp[-4].plist, 0, 0, 0, 1, yyvsp[-4].plist));
                   }
                   yyval.sc = new SdagConstruct(SCONNECT, yyvsp[-5].strval, yyvsp[-1].strval, yyvsp[-4].plist);
		}
    break;

  case 218:
#line 845 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SWHEN, 0, 0,0,0,0,0,yyvsp[-2].entrylist); }
    break;

  case 219:
#line 847 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SWHEN, 0, 0, 0,0,0, yyvsp[0].sc, yyvsp[-1].entrylist); }
    break;

  case 220:
#line 849 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SWHEN, 0, 0, 0,0,0, yyvsp[-1].sc, yyvsp[-3].entrylist); }
    break;

  case 221:
#line 851 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SOVERLAP,0, 0,0,0,0,yyvsp[-1].sc, 0); }
    break;

  case 222:
#line 853 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SFOR, 0, new SdagConstruct(SINT_EXPR, yyvsp[-8].strval), new SdagConstruct(SINT_EXPR, yyvsp[-6].strval),
		             new SdagConstruct(SINT_EXPR, yyvsp[-4].strval), 0, yyvsp[-1].sc, 0); }
    break;

  case 223:
#line 856 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SFOR, 0, new SdagConstruct(SINT_EXPR, yyvsp[-6].strval), new SdagConstruct(SINT_EXPR, yyvsp[-4].strval), 
		         new SdagConstruct(SINT_EXPR, yyvsp[-2].strval), 0, yyvsp[0].sc, 0); }
    break;

  case 224:
#line 859 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SFORALL, 0, new SdagConstruct(SIDENT, yyvsp[-9].strval), new SdagConstruct(SINT_EXPR, yyvsp[-6].strval), 
		             new SdagConstruct(SINT_EXPR, yyvsp[-4].strval), new SdagConstruct(SINT_EXPR, yyvsp[-2].strval), yyvsp[0].sc, 0); }
    break;

  case 225:
#line 862 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SFORALL, 0, new SdagConstruct(SIDENT, yyvsp[-11].strval), new SdagConstruct(SINT_EXPR, yyvsp[-8].strval), 
		                 new SdagConstruct(SINT_EXPR, yyvsp[-6].strval), new SdagConstruct(SINT_EXPR, yyvsp[-4].strval), yyvsp[-1].sc, 0); }
    break;

  case 226:
#line 865 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, yyvsp[-3].strval), yyvsp[0].sc,0,0,yyvsp[-1].sc,0); }
    break;

  case 227:
#line 867 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, yyvsp[-5].strval), yyvsp[0].sc,0,0,yyvsp[-2].sc,0); }
    break;

  case 228:
#line 869 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, yyvsp[-2].strval), 0,0,0,yyvsp[0].sc,0); }
    break;

  case 229:
#line 871 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SWHILE, 0, new SdagConstruct(SINT_EXPR, yyvsp[-4].strval), 0,0,0,yyvsp[-1].sc,0); }
    break;

  case 230:
#line 873 "xi-grammar.y"
    { yyval.sc = yyvsp[-1].sc; }
    break;

  case 231:
#line 877 "xi-grammar.y"
    { yyval.sc = 0; }
    break;

  case 232:
#line 879 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SELSE, 0,0,0,0,0, yyvsp[0].sc,0); }
    break;

  case 233:
#line 881 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SELSE, 0,0,0,0,0, yyvsp[-1].sc,0); }
    break;

  case 234:
#line 884 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SFORWARD, new SdagConstruct(SIDENT, yyvsp[0].strval)); }
    break;

  case 235:
#line 886 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SFORWARD, new SdagConstruct(SIDENT, yyvsp[-2].strval), yyvsp[0].sc);  }
    break;

  case 236:
#line 890 "xi-grammar.y"
    { in_int_expr = 0; yyval.intval = 0; }
    break;

  case 237:
#line 894 "xi-grammar.y"
    { in_int_expr = 1; yyval.intval = 0; }
    break;

  case 238:
#line 898 "xi-grammar.y"
    { 
		  if (yyvsp[0].plist != 0)
		     yyval.entry = new Entry(lineno, 0, 0, yyvsp[-1].strval, yyvsp[0].plist, 0, 0, 0, 0); 
		  else
		     yyval.entry = new Entry(lineno, 0, 0, yyvsp[-1].strval, 
				new ParamList(new Parameter(0, new BuiltinType("void"))), 0, 0, 0, 0); 
		}
    break;

  case 239:
#line 906 "xi-grammar.y"
    { if (yyvsp[0].plist != 0)
		    yyval.entry = new Entry(lineno, 0, 0, yyvsp[-4].strval, yyvsp[0].plist, 0, 0, yyvsp[-2].strval, 0); 
		  else
		    yyval.entry = new Entry(lineno, 0, 0, yyvsp[-4].strval, new ParamList(new Parameter(0, new BuiltinType("void"))), 0, 0, yyvsp[-2].strval, 0); 
		}
    break;

  case 240:
#line 914 "xi-grammar.y"
    { yyval.entrylist = new EntryList(yyvsp[0].entry); }
    break;

  case 241:
#line 916 "xi-grammar.y"
    { yyval.entrylist = new EntryList(yyvsp[-2].entry,yyvsp[0].entrylist); }
    break;

  case 242:
#line 920 "xi-grammar.y"
    { in_bracket=1; }
    break;

  case 243:
#line 923 "xi-grammar.y"
    { in_bracket=0; }
    break;

  case 244:
#line 927 "xi-grammar.y"
    { if (!macroDefined(yyvsp[0].strval, 1)) in_comment = 1; }
    break;

  case 245:
#line 931 "xi-grammar.y"
    { if (!macroDefined(yyvsp[0].strval, 0)) in_comment = 1; }
    break;


    }

/* Line 991 of yacc.c.  */
#line 2954 "y.tab.c"

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
	  YYSIZE_T yysize = 0;
	  int yytype = YYTRANSLATE (yychar);
	  char *yymsg;
	  int yyx, yycount;

	  yycount = 0;
	  /* Start YYX at -YYN if negative to avoid negative indexes in
	     YYCHECK.  */
	  for (yyx = yyn < 0 ? -yyn : 0;
	       yyx < (int) (sizeof (yytname) / sizeof (char *)); yyx++)
	    if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR)
	      yysize += yystrlen (yytname[yyx]) + 15, yycount++;
	  yysize += yystrlen ("syntax error, unexpected ") + 1;
	  yysize += yystrlen (yytname[yytype]);
	  yymsg = (char *) YYSTACK_ALLOC (yysize);
	  if (yymsg != 0)
	    {
	      char *yyp = yystpcpy (yymsg, "syntax error, unexpected ");
	      yyp = yystpcpy (yyp, yytname[yytype]);

	      if (yycount < 5)
		{
		  yycount = 0;
		  for (yyx = yyn < 0 ? -yyn : 0;
		       yyx < (int) (sizeof (yytname) / sizeof (char *));
		       yyx++)
		    if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR)
		      {
			const char *yyq = ! yycount ? ", expecting " : " or ";
			yyp = yystpcpy (yyp, yyq);
			yyp = yystpcpy (yyp, yytname[yyx]);
			yycount++;
		      }
		}
	      yyerror (yymsg);
	      YYSTACK_FREE (yymsg);
	    }
	  else
	    yyerror ("syntax error; also virtual memory exhausted");
	}
      else
#endif /* YYERROR_VERBOSE */
	yyerror ("syntax error");
    }



  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
	 error, discard it.  */

      /* Return failure if at end of input.  */
      if (yychar == YYEOF)
        {
	  /* Pop the error token.  */
          YYPOPSTACK;
	  /* Pop the rest of the stack.  */
	  while (yyss < yyssp)
	    {
	      YYDSYMPRINTF ("Error: popping", yystos[*yyssp], yyvsp, yylsp);
	      yydestruct (yystos[*yyssp], yyvsp);
	      YYPOPSTACK;
	    }
	  YYABORT;
        }

      YYDSYMPRINTF ("Error: discarding", yytoken, &yylval, &yylloc);
      yydestruct (yytoken, &yylval);
      yychar = YYEMPTY;

    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab2;


/*----------------------------------------------------.
| yyerrlab1 -- error raised explicitly by an action.  |
`----------------------------------------------------*/
yyerrlab1:

  /* Suppress GCC warning that yyerrlab1 is unused when no action
     invokes YYERROR.  */
#if defined (__GNUC_MINOR__) && 2093 <= (__GNUC__ * 1000 + __GNUC_MINOR__) \
    && !defined __cplusplus
  __attribute__ ((__unused__))
#endif


  goto yyerrlab2;


/*---------------------------------------------------------------.
| yyerrlab2 -- pop states until the error token can be shifted.  |
`---------------------------------------------------------------*/
yyerrlab2:
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

      YYDSYMPRINTF ("Error: popping", yystos[*yyssp], yyvsp, yylsp);
      yydestruct (yystos[yystate], yyvsp);
      yyvsp--;
      yystate = *--yyssp;

      YY_STACK_PRINT (yyss, yyssp);
    }

  if (yyn == YYFINAL)
    YYACCEPT;

  YYDPRINTF ((stderr, "Shifting error token, "));

  *++yyvsp = yylval;


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
/*----------------------------------------------.
| yyoverflowlab -- parser overflow comes here.  |
`----------------------------------------------*/
yyoverflowlab:
  yyerror ("parser stack overflow");
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
  return yyresult;
}


#line 934 "xi-grammar.y"

void yyerror(const char *mesg)
{
  cout << cur_file<<":"<<lineno<<": Charmxi syntax error> " << mesg << endl;
  // return 0;
}

