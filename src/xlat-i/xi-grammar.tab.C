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
#line 16 "xi-grammar.y"
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
#line 249 "y.tab.c"
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 214 of yacc.c.  */
#line 261 "y.tab.c"

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
#define YYLAST   548

/* YYNTOKENS -- Number of terminals. */
#define YYNTOKENS  77
/* YYNNTS -- Number of nonterminals. */
#define YYNNTS  100
/* YYNRULES -- Number of rules. */
#define YYNRULES  242
/* YYNRULES -- Number of states. */
#define YYNSTATES  481

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
     216,   218,   222,   224,   226,   227,   231,   233,   237,   239,
     240,   244,   246,   250,   252,   254,   260,   262,   265,   269,
     276,   277,   280,   282,   286,   292,   298,   304,   310,   315,
     319,   326,   333,   339,   345,   351,   357,   363,   368,   376,
     377,   380,   381,   384,   387,   391,   394,   398,   400,   404,
     409,   412,   415,   418,   421,   424,   426,   431,   432,   435,
     438,   441,   444,   447,   451,   455,   459,   466,   470,   477,
     481,   488,   490,   494,   496,   499,   501,   509,   515,   517,
     519,   520,   524,   526,   530,   532,   534,   536,   538,   540,
     542,   544,   546,   548,   550,   552,   554,   556,   558,   559,
     561,   565,   566,   568,   574,   580,   586,   591,   595,   597,
     599,   601,   604,   609,   613,   615,   619,   623,   626,   627,
     631,   632,   634,   638,   640,   643,   645,   648,   649,   654,
     656,   660,   662,   663,   670,   679,   684,   688,   694,   699,
     711,   721,   734,   749,   756,   765,   771,   779,   783,   784,
     787,   792,   794,   798,   800,   802,   805,   811,   813,   817,
     819,   821,   824
};

/* YYRHS -- A `-1'-separated list of the rules' RHS. */
static const short yyrhs[] =
{
      78,     0,    -1,    79,    -1,    -1,    84,    79,    -1,    -1,
       5,    -1,    -1,    63,    -1,    50,    -1,    50,    -1,    83,
      64,    64,    50,    -1,     3,    82,    85,    -1,     4,    82,
      85,    -1,    63,    -1,    65,    86,    66,    81,    -1,    -1,
      87,    86,    -1,    80,    65,    86,    66,    81,    -1,    80,
      84,    -1,    80,   139,    -1,    80,   118,    63,    -1,    80,
     121,    -1,    80,   122,    -1,    80,   123,    -1,    80,   125,
      -1,    80,   136,    -1,   175,    -1,   176,    -1,   100,    -1,
      51,    -1,    52,    -1,    88,    -1,    88,    67,    89,    -1,
      -1,    89,    -1,    -1,    68,    90,    69,    -1,    56,    -1,
      57,    -1,    58,    -1,    59,    -1,    62,    56,    -1,    62,
      57,    -1,    62,    57,    56,    -1,    62,    57,    57,    -1,
      62,    58,    -1,    62,    59,    -1,    57,    57,    -1,    60,
      -1,    61,    -1,    57,    61,    -1,    33,    -1,    82,    91,
      -1,    83,    91,    -1,    92,    -1,    94,    -1,    95,    70,
      -1,    96,    70,    -1,    97,    70,    -1,    99,    71,    70,
      82,    72,    71,   157,    72,    -1,    95,    -1,    96,    -1,
      97,    -1,    98,    -1,    34,    99,    -1,    99,    34,    -1,
      99,    73,    -1,    99,    -1,    51,    -1,    83,    -1,    74,
     101,    75,    -1,    -1,   102,   103,    -1,     6,   100,    83,
     103,    -1,     6,    16,    95,    70,    82,    -1,    -1,    33,
      -1,    -1,    74,   108,    75,    -1,   109,    -1,   109,    67,
     108,    -1,    35,    -1,    36,    -1,    -1,    74,   111,    75,
      -1,   115,    -1,   115,    67,   111,    -1,    49,    -1,    -1,
      74,   114,    75,    -1,   112,    -1,   112,    67,   114,    -1,
      28,    -1,    49,    -1,   100,    82,    74,    75,    63,    -1,
     116,    -1,   116,   117,    -1,    16,   107,    93,    -1,    16,
     107,    93,    65,   117,    66,    -1,    -1,    64,   120,    -1,
      93,    -1,    93,    67,   120,    -1,    11,   110,    93,   119,
     137,    -1,    12,   110,    93,   119,   137,    -1,    13,   110,
      93,   119,   137,    -1,    14,   110,    93,   119,   137,    -1,
      74,    51,    82,    75,    -1,    74,    82,    75,    -1,    15,
     113,   124,    93,   119,   137,    -1,    15,   124,   113,    93,
     119,   137,    -1,    11,   110,    82,   119,   137,    -1,    12,
     110,    82,   119,   137,    -1,    13,   110,    82,   119,   137,
      -1,    14,   110,    82,   119,   137,    -1,    15,   124,    82,
     119,   137,    -1,    16,   107,    82,    63,    -1,    16,   107,
      82,    65,   117,    66,    63,    -1,    -1,    76,   100,    -1,
      -1,    76,    51,    -1,    76,    52,    -1,    17,    82,   131,
      -1,    98,   132,    -1,   100,    82,   132,    -1,   133,    -1,
     133,    67,   134,    -1,    21,    68,   134,    69,    -1,   135,
     126,    -1,   135,   127,    -1,   135,   128,    -1,   135,   129,
      -1,   135,   130,    -1,    63,    -1,    65,   138,    66,    81,
      -1,    -1,   144,   138,    -1,   104,    63,    -1,   105,    63,
      -1,   141,    63,    -1,   140,    63,    -1,    10,   142,    63,
      -1,    18,   143,    63,    -1,     8,   106,    83,    -1,     8,
     106,    83,    71,   106,    72,    -1,     7,   106,    83,    -1,
       7,   106,    83,    71,   106,    72,    -1,     9,   106,    83,
      -1,     9,   106,    83,    71,   106,    72,    -1,    83,    -1,
      83,    67,   142,    -1,    52,    -1,   145,    63,    -1,   139,
      -1,    37,   147,   146,    82,   158,   159,   160,    -1,    37,
     147,    82,   158,   160,    -1,    33,    -1,    96,    -1,    -1,
      74,   148,    75,    -1,   149,    -1,   149,    67,   148,    -1,
      20,    -1,    22,    -1,    23,    -1,    29,    -1,    30,    -1,
      31,    -1,    32,    -1,    24,    -1,    25,    -1,    26,    -1,
      49,    -1,    52,    -1,    51,    -1,    83,    -1,    -1,    53,
      -1,    53,    67,   151,    -1,    -1,    53,    -1,    53,    74,
     152,    75,   152,    -1,    53,    65,   152,    66,   152,    -1,
      53,    71,   151,    72,   152,    -1,    71,   152,    72,   152,
      -1,   100,    82,    74,    -1,    65,    -1,    66,    -1,   100,
      -1,   100,    82,    -1,   100,    82,    76,   150,    -1,   153,
     152,    75,    -1,   156,    -1,   156,    67,   157,    -1,    71,
     157,    72,    -1,    71,    72,    -1,    -1,    19,    76,    51,
      -1,    -1,   166,    -1,    65,   161,    66,    -1,   166,    -1,
     166,   161,    -1,   166,    -1,   166,   161,    -1,    -1,    48,
      71,   164,    72,    -1,    50,    -1,    50,    67,   164,    -1,
      52,    -1,    -1,    43,   165,   154,   152,   155,   163,    -1,
      47,    71,    50,   158,    72,   154,   152,    66,    -1,    41,
     172,    65,    66,    -1,    41,   172,   166,    -1,    41,   172,
      65,   161,    66,    -1,    42,    65,   162,    66,    -1,    38,
     170,   152,    63,   152,    63,   152,   169,    65,   161,    66,
      -1,    38,   170,   152,    63,   152,    63,   152,   169,   166,
      -1,    39,    74,    50,    75,   170,   152,    64,   152,    67,
     152,   169,   166,    -1,    39,    74,    50,    75,   170,   152,
      64,   152,    67,   152,   169,    65,   161,    66,    -1,    45,
     170,   152,   169,   166,   167,    -1,    45,   170,   152,   169,
      65,   161,    66,   167,    -1,    40,   170,   152,   169,   166,
      -1,    40,   170,   152,   169,    65,   161,    66,    -1,    44,
     168,    63,    -1,    -1,    46,   166,    -1,    46,    65,   161,
      66,    -1,    50,    -1,    50,    67,   168,    -1,    72,    -1,
      71,    -1,    50,   158,    -1,    50,   173,   152,   174,   158,
      -1,   171,    -1,   171,    67,   172,    -1,    74,    -1,    75,
      -1,    54,    82,    -1,    55,    82,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const unsigned short yyrline[] =
{
       0,   132,   132,   137,   140,   145,   146,   151,   152,   156,
     160,   162,   170,   174,   181,   183,   188,   189,   193,   195,
     197,   199,   201,   203,   205,   207,   209,   211,   213,   217,
     219,   221,   225,   227,   232,   233,   238,   239,   243,   245,
     247,   249,   251,   253,   255,   257,   259,   261,   263,   265,
     267,   269,   271,   275,   276,   278,   280,   284,   288,   290,
     294,   298,   300,   302,   304,   307,   309,   313,   315,   319,
     321,   325,   330,   331,   335,   339,   344,   345,   350,   351,
     361,   363,   367,   369,   374,   375,   379,   381,   385,   390,
     391,   395,   397,   401,   403,   407,   411,   413,   417,   419,
     424,   425,   429,   431,   435,   437,   441,   445,   449,   455,
     459,   461,   465,   467,   471,   475,   479,   483,   485,   490,
     491,   496,   497,   499,   503,   505,   507,   511,   513,   517,
     521,   523,   525,   527,   529,   533,   535,   540,   558,   562,
     564,   566,   567,   569,   571,   575,   577,   579,   582,   587,
     589,   593,   595,   599,   603,   605,   609,   620,   633,   635,
     640,   641,   645,   647,   651,   653,   655,   657,   659,   661,
     663,   665,   667,   669,   671,   675,   677,   679,   684,   685,
     687,   696,   697,   699,   705,   711,   717,   725,   732,   740,
     747,   749,   751,   753,   760,   762,   766,   768,   773,   774,
     779,   780,   782,   786,   788,   792,   794,   799,   800,   804,
     806,   810,   813,   816,   821,   835,   837,   839,   841,   843,
     846,   849,   852,   855,   857,   859,   861,   863,   868,   869,
     871,   874,   876,   880,   884,   888,   896,   904,   906,   910,
     913,   917,   921
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
  "CAttribs", "CAttribList", "ArrayAttrib", "ArrayAttribs", 
  "ArrayAttribList", "CAttrib", "Var", "VarList", "Message", 
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
     108,   108,   109,   109,   110,   110,   111,   111,   112,   113,
     113,   114,   114,   115,   115,   116,   117,   117,   118,   118,
     119,   119,   120,   120,   121,   121,   122,   123,   124,   124,
     125,   125,   126,   126,   127,   128,   129,   130,   130,   131,
     131,   132,   132,   132,   133,   133,   133,   134,   134,   135,
     136,   136,   136,   136,   136,   137,   137,   138,   138,   139,
     139,   139,   139,   139,   139,   140,   140,   140,   140,   141,
     141,   142,   142,   143,   144,   144,   145,   145,   146,   146,
     147,   147,   148,   148,   149,   149,   149,   149,   149,   149,
     149,   149,   149,   149,   149,   150,   150,   150,   151,   151,
     151,   152,   152,   152,   152,   152,   152,   153,   154,   155,
     156,   156,   156,   156,   157,   157,   158,   158,   159,   159,
     160,   160,   160,   161,   161,   162,   162,   163,   163,   164,
     164,   165,   165,   166,   166,   166,   166,   166,   166,   166,
     166,   166,   166,   166,   166,   166,   166,   166,   167,   167,
     167,   168,   168,   169,   170,   171,   171,   172,   172,   173,
     174,   175,   176
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
       1,     3,     1,     1,     0,     3,     1,     3,     1,     0,
       3,     1,     3,     1,     1,     5,     1,     2,     3,     6,
       0,     2,     1,     3,     5,     5,     5,     5,     4,     3,
       6,     6,     5,     5,     5,     5,     5,     4,     7,     0,
       2,     0,     2,     2,     3,     2,     3,     1,     3,     4,
       2,     2,     2,     2,     2,     1,     4,     0,     2,     2,
       2,     2,     2,     3,     3,     3,     6,     3,     6,     3,
       6,     1,     3,     1,     2,     1,     7,     5,     1,     1,
       0,     3,     1,     3,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     0,     1,
       3,     0,     1,     5,     5,     5,     4,     3,     1,     1,
       1,     2,     4,     3,     1,     3,     3,     2,     0,     3,
       0,     1,     3,     1,     2,     1,     2,     0,     4,     1,
       3,     1,     0,     6,     8,     4,     3,     5,     4,    11,
       9,    12,    14,     6,     8,     5,     7,     3,     0,     2,
       4,     1,     3,     1,     1,     2,     5,     1,     3,     1,
       1,     2,     2
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const unsigned char yydefact[] =
{
       3,     0,     0,     0,     2,     3,     9,     0,     0,     1,
       4,    14,     5,    12,    13,     6,     0,     0,     0,     0,
       5,    27,    28,   241,   242,     0,    76,    76,    76,     0,
      84,    84,    84,    84,     0,    78,     0,     0,     5,    19,
       0,     0,     0,    22,    23,    24,    25,     0,    26,    20,
       0,     0,     7,    17,     0,    52,     0,    10,    38,    39,
      40,    41,    49,    50,     0,    36,    55,    56,    61,    62,
      63,    64,    68,     0,    77,     0,     0,     0,   151,     0,
       0,     0,     0,     0,     0,     0,     0,    89,     0,     0,
     153,     0,     0,     0,   139,   140,    21,    84,    84,    84,
      84,     0,    78,   130,   131,   132,   133,   134,   142,   141,
       8,    15,     0,    65,    48,    51,    42,    43,    46,    47,
       0,    34,    54,    57,    58,    59,    66,     0,    67,    72,
     147,   145,   149,     0,   143,    93,    94,     0,    86,    36,
     100,   100,   100,   100,    88,     0,     0,    91,     0,     0,
       0,     0,     0,    82,    83,     0,    80,    98,   144,     0,
      64,     0,   127,     0,     7,     0,     0,     0,     0,     0,
       0,     0,    44,    45,     0,    30,    31,    32,    35,     0,
      29,     0,     0,    72,    74,    76,    76,    76,   152,    85,
       0,    53,     0,     0,     0,     0,     0,     0,   109,     0,
      90,   100,   100,    79,     0,     0,   119,     0,   125,   121,
       0,   129,    18,   100,   100,   100,   100,   100,     0,    75,
      11,     0,    37,     0,    69,    70,     0,    73,     0,     0,
       0,    87,   102,   101,   135,   137,   104,   105,   106,   107,
     108,    92,     0,     0,    81,     0,    96,     0,     0,   124,
     122,   123,   126,   128,     0,     0,     0,     0,     0,   117,
       0,    33,     0,    71,   148,   146,   150,     0,   160,     0,
     155,   137,     0,   110,   111,     0,    97,    99,   120,   112,
     113,   114,   115,   116,     0,     0,   103,     0,     0,     7,
     138,   154,     0,     0,   190,   181,   194,     0,   164,   165,
     166,   171,   172,   173,   167,   168,   169,   170,   174,     0,
     162,    52,    10,     0,     0,   159,     0,   136,     0,   118,
     191,   182,   181,     0,     0,    60,   161,     0,     0,   200,
       0,    95,   187,     0,   181,   178,   181,     0,   193,   195,
     163,   197,     0,     0,     0,     0,     0,     0,   212,     0,
       0,     0,     0,   157,   201,   198,   176,   175,   177,   192,
       0,   179,     0,     0,   181,   196,   234,   181,     0,   181,
       0,   237,     0,     0,   211,     0,   231,     0,   181,     0,
       0,   203,     0,   200,   181,   178,   181,   181,   186,     0,
       0,     0,   239,   235,   181,     0,     0,   216,     0,   205,
     188,   181,     0,   227,     0,     0,   202,   204,     0,   156,
     184,   180,   185,   183,   181,     0,   233,     0,     0,   238,
     215,     0,   218,   206,     0,   232,     0,     0,   199,     0,
     181,     0,   225,   240,     0,   217,   189,   207,     0,   228,
       0,   181,     0,     0,   236,     0,   213,     0,     0,   223,
     181,     0,   181,   226,     0,   228,     0,   229,     0,     0,
       0,   209,     0,   224,     0,   214,     0,   220,   181,     0,
     208,   230,     0,     0,   210,   219,     0,     0,   221,     0,
     222
};

/* YYDEFGOTO[NTERM-NUM]. */
static const short yydefgoto[] =
{
      -1,     3,     4,    18,   111,   139,    65,     5,    13,    19,
      20,   177,   178,   179,   122,    66,   232,    67,    68,    69,
      70,    71,    72,   245,   226,   183,   184,    40,    41,    75,
      89,   155,   156,    81,   137,   147,    86,   148,   138,   246,
     247,    42,   193,   233,    43,    44,    45,    87,    46,   103,
     104,   105,   106,   107,   249,   208,   162,   163,    47,    48,
     236,   269,   270,    50,    51,    79,    91,   271,   272,   316,
     288,   309,   310,   359,   362,   323,   295,   401,   437,   296,
     297,   329,   383,   353,   380,   398,   446,   462,   375,   381,
     449,   377,   417,   367,   371,   372,   394,   434,    21,    22
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -397
static const short yypact[] =
{
     133,   -11,   -11,    34,  -397,   133,  -397,   -54,   -54,  -397,
    -397,  -397,    25,  -397,  -397,  -397,   -11,   -11,   207,    23,
      25,  -397,  -397,  -397,  -397,   196,    76,    76,    76,     8,
      44,    44,    44,    44,    48,    75,   123,   117,    25,  -397,
     124,   129,   134,  -397,  -397,  -397,  -397,   220,  -397,  -397,
     135,   136,   138,  -397,   296,  -397,   261,  -397,  -397,   -17,
    -397,  -397,  -397,  -397,    69,   -12,  -397,  -397,   132,   154,
     168,  -397,    -8,     8,  -397,     8,     8,     8,    19,   141,
     -16,   -11,   -11,   -11,   -11,   102,   165,   166,   107,   -11,
    -397,   178,   209,   137,  -397,  -397,  -397,    44,    44,    44,
      44,   165,    75,  -397,  -397,  -397,  -397,  -397,  -397,  -397,
    -397,  -397,   175,    -9,  -397,  -397,  -397,   106,  -397,  -397,
     183,   276,  -397,  -397,  -397,  -397,  -397,   179,  -397,   -36,
      33,    41,    50,     8,  -397,  -397,  -397,   173,   184,   182,
     197,   197,   197,   197,  -397,   -11,   185,   195,   188,   128,
     -11,   226,   -11,  -397,  -397,   201,   215,   218,  -397,   -11,
      32,   -11,   217,   216,   138,   -11,   -11,   -11,   -11,   -11,
     -11,   -11,  -397,  -397,   236,  -397,  -397,   221,  -397,   222,
    -397,   -11,   140,   230,  -397,    76,    76,    76,  -397,  -397,
     -16,  -397,   -11,    91,    91,    91,    91,   238,  -397,   226,
    -397,   197,   197,  -397,   107,   261,   229,   144,  -397,   248,
     209,  -397,  -397,   197,   197,   197,   197,   197,   111,  -397,
    -397,   276,  -397,   242,  -397,   228,   241,  -397,   253,   258,
     259,  -397,   273,  -397,  -397,   271,  -397,  -397,  -397,  -397,
    -397,  -397,    91,    91,  -397,   -11,   261,   293,   261,  -397,
    -397,  -397,  -397,  -397,    91,    91,    91,    91,    91,  -397,
     261,  -397,   289,  -397,  -397,  -397,  -397,   -11,   287,   297,
    -397,   271,   299,  -397,  -397,   291,  -397,  -397,  -397,  -397,
    -397,  -397,  -397,  -397,   300,   261,  -397,   319,   314,   138,
    -397,  -397,   292,   306,   -11,   -35,   318,   323,  -397,  -397,
    -397,  -397,  -397,  -397,  -397,  -397,  -397,  -397,  -397,   332,
     341,   360,   338,   340,   132,  -397,   -11,  -397,   350,  -397,
     112,     1,   -35,   337,   261,  -397,  -397,   319,   240,   349,
     340,  -397,  -397,    18,   -35,   362,   -35,   364,  -397,  -397,
    -397,  -397,   365,   363,   366,   363,   389,   373,   390,   391,
     363,   392,   446,  -397,  -397,   425,  -397,  -397,   228,  -397,
     399,   400,   394,   393,   -35,  -397,  -397,   -35,   419,   -35,
      67,   403,   359,   446,  -397,   406,   427,   410,   -35,   442,
     429,   446,   420,   349,   -35,   362,   -35,   -35,  -397,   434,
     423,   428,  -397,  -397,   -35,   389,   339,  -397,   433,   446,
    -397,   -35,   391,  -397,   428,   340,  -397,  -397,   451,  -397,
    -397,  -397,  -397,  -397,   -35,   363,  -397,   378,   430,  -397,
    -397,   437,  -397,  -397,   438,  -397,   388,   435,  -397,   443,
     -35,   446,  -397,  -397,   340,  -397,  -397,   460,   446,   463,
     406,   -35,   447,   444,  -397,   441,  -397,   448,   407,  -397,
     -35,   428,   -35,  -397,   465,   463,   446,  -397,   450,   417,
     452,   453,   445,  -397,   455,  -397,   446,  -397,   -35,   465,
    -397,  -397,   456,   428,  -397,  -397,   436,   446,  -397,   457,
    -397
};

/* YYPGOTO[NTERM-NUM].  */
static const short yypgoto[] =
{
    -397,  -397,   508,  -397,  -159,    -1,   -27,   500,   516,     3,
    -397,  -397,   304,  -397,   387,  -397,   -62,  -397,   -51,   239,
    -397,   -86,   472,   -21,  -397,  -397,   346,  -397,  -397,   -14,
     431,   326,  -397,    84,   342,  -397,   449,   335,  -397,  -397,
    -209,  -397,   -82,   264,  -397,  -397,  -397,   -44,  -397,  -397,
    -397,  -397,  -397,  -397,  -397,   328,  -397,   325,  -397,  -397,
     -49,   267,   521,  -397,  -397,   408,  -397,  -397,  -397,  -397,
    -397,   213,  -397,  -397,   157,  -291,  -397,   103,  -397,  -397,
    -243,  -323,  -397,   161,  -364,  -397,  -397,    77,  -397,  -319,
      90,   145,  -396,  -321,  -397,   153,  -397,  -397,  -397,  -397
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -159
static const short yytable[] =
{
       7,     8,    78,   112,    73,   212,   160,   355,   426,    11,
     354,    12,   135,    76,    77,    23,    24,   407,   321,   140,
     141,   142,   143,    53,   369,   126,   126,   157,   120,   378,
      15,   337,   421,   136,     9,   423,   322,   276,   182,     6,
     114,    93,   150,   360,   115,   363,   129,   393,   130,   131,
     132,   284,   120,   397,   399,   459,   121,   169,    57,   194,
     195,   196,   127,   127,   354,   128,   334,   443,    57,   356,
     357,   161,   335,   388,   447,   336,   389,   476,   391,    16,
      17,   339,   427,   120,   146,   342,   133,   404,   201,    52,
     202,   -16,   464,   410,   430,   412,   413,   120,   432,  -121,
     180,  -121,   472,   418,   185,   120,    78,   439,   207,    74,
     424,   444,   186,   479,   120,    82,    83,    84,    80,   242,
     243,   187,    85,   429,   160,   116,   117,   118,   119,   457,
     317,   254,   255,   256,   257,   258,     1,     2,   328,   442,
     467,   392,   153,   154,   197,   237,   238,   239,   146,    88,
     451,   144,     6,   145,   234,   225,   235,   478,   206,   458,
     209,   460,   172,   173,   213,   214,   215,   216,   217,   218,
     219,   228,   229,   230,   259,    90,   260,   473,     6,   145,
     223,   165,   166,   167,   168,    92,   332,    94,   333,   161,
      57,   224,    95,   273,   274,   250,   251,    96,   108,   109,
     180,   110,   123,   164,   134,   279,   280,   281,   282,   283,
       1,     2,    54,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    34,    35,   124,    36,   159,   278,    37,    55,
      56,    97,    98,    99,   100,   101,   102,   314,   125,   149,
     151,   158,    55,    56,   275,   171,    57,   174,   189,   181,
     121,   190,    58,    59,    60,    61,    62,    63,    64,    57,
     198,   192,   199,   200,   294,    58,    59,    60,    61,    62,
      63,    64,    38,    55,    56,   144,   203,    25,    26,    27,
      28,    29,   204,   205,   210,   211,   220,   313,   221,    36,
      57,   222,   120,   320,    55,    56,    58,    59,    60,    61,
      62,    63,    64,   294,   182,   248,   358,   294,   268,    55,
      56,    57,   341,   240,   262,   330,   263,    58,    59,    60,
      61,    62,    63,    64,   207,   264,    57,   175,   176,    55,
     265,   266,    58,    59,    60,    61,    62,    63,    64,   298,
     267,   299,   300,   301,   302,   303,    57,   311,   304,   305,
     306,   307,    58,    59,    60,    61,    62,    63,    64,   277,
     285,   287,   291,   289,   312,   292,   293,   318,   308,   319,
      58,    59,    60,    61,    62,    63,    64,   343,   344,   345,
     346,   347,   348,   349,   350,   324,   351,   343,   344,   345,
     346,   347,   348,   349,   350,   325,   351,   343,   344,   345,
     346,   347,   348,   349,   350,   420,   351,   326,   327,    -9,
    -158,   328,   338,   331,   352,   361,   343,   344,   345,   346,
     347,   348,   349,   350,   396,   351,   343,   344,   345,   346,
     347,   348,   349,   350,   366,   351,   364,   365,   373,   370,
     368,   376,   374,   431,   382,   343,   344,   345,   346,   347,
     348,   349,   350,   438,   351,   343,   344,   345,   346,   347,
     348,   349,   350,   379,   351,   384,   386,   385,   387,   390,
     395,   400,   456,   403,   343,   344,   345,   346,   347,   348,
     349,   350,   466,   351,   343,   344,   345,   346,   347,   348,
     349,   350,   405,   351,   402,   406,   408,   414,   415,   422,
     416,   477,   428,   435,   436,   433,   441,   440,   445,   448,
     453,   452,   454,    10,   455,   461,   465,   470,    39,   468,
     469,   471,   475,   480,    14,   261,   191,   315,   113,   227,
     244,   286,   231,   170,   241,   253,   152,   252,   290,    49,
     340,   188,   411,   450,   409,   463,   474,   425,   419
};

static const unsigned short yycheck[] =
{
       1,     2,    29,    54,    25,   164,    92,   330,   404,    63,
     329,    65,    28,    27,    28,    16,    17,   381,    53,    81,
      82,    83,    84,    20,   345,    34,    34,    89,    64,   350,
       5,   322,   396,    49,     0,   399,    71,   246,    74,    50,
      57,    38,    86,   334,    61,   336,    73,   370,    75,    76,
      77,   260,    64,   372,   373,   451,    68,   101,    50,   141,
     142,   143,    71,    71,   383,    73,    65,   431,    50,    51,
      52,    92,    71,   364,   438,    74,   367,   473,   369,    54,
      55,   324,   405,    64,    85,   328,    67,   378,   150,    66,
     152,    66,   456,   384,   415,   386,   387,    64,   417,    67,
     121,    69,   466,   394,    71,    64,   133,   426,    76,    33,
     401,   434,    71,   477,    64,    31,    32,    33,    74,   201,
     202,    71,    74,   414,   210,    56,    57,    58,    59,   448,
     289,   213,   214,   215,   216,   217,     3,     4,    71,   430,
     459,    74,    35,    36,   145,   194,   195,   196,   149,    74,
     441,    49,    50,    51,    63,   182,    65,   476,   159,   450,
     161,   452,    56,    57,   165,   166,   167,   168,   169,   170,
     171,   185,   186,   187,    63,    52,    65,   468,    50,    51,
     181,    97,    98,    99,   100,    68,    74,    63,    76,   210,
      50,    51,    63,   242,   243,    51,    52,    63,    63,    63,
     221,    63,    70,    66,    63,   254,   255,   256,   257,   258,
       3,     4,    16,     6,     7,     8,     9,    10,    11,    12,
      13,    14,    15,    16,    70,    18,    17,   248,    21,    33,
      34,    11,    12,    13,    14,    15,    16,   288,    70,    74,
      74,    63,    33,    34,   245,    70,    50,    64,    75,    70,
      68,    67,    56,    57,    58,    59,    60,    61,    62,    50,
      75,    64,    67,    75,   285,    56,    57,    58,    59,    60,
      61,    62,    65,    33,    34,    49,    75,     6,     7,     8,
       9,    10,    67,    65,    67,    69,    50,   288,    67,    18,
      50,    69,    64,   294,    33,    34,    56,    57,    58,    59,
      60,    61,    62,   324,    74,    76,   333,   328,    37,    33,
      34,    50,    72,    75,    72,   316,    75,    56,    57,    58,
      59,    60,    61,    62,    76,    72,    50,    51,    52,    33,
      72,    72,    56,    57,    58,    59,    60,    61,    62,    20,
      67,    22,    23,    24,    25,    26,    50,    33,    29,    30,
      31,    32,    56,    57,    58,    59,    60,    61,    62,    66,
      71,    74,    63,    66,    50,    74,    66,    75,    49,    63,
      56,    57,    58,    59,    60,    61,    62,    38,    39,    40,
      41,    42,    43,    44,    45,    67,    47,    38,    39,    40,
      41,    42,    43,    44,    45,    72,    47,    38,    39,    40,
      41,    42,    43,    44,    45,    66,    47,    75,    67,    71,
      50,    71,    75,    63,    65,    53,    38,    39,    40,    41,
      42,    43,    44,    45,    65,    47,    38,    39,    40,    41,
      42,    43,    44,    45,    71,    47,    72,    72,    65,    50,
      74,    50,    52,    65,    19,    38,    39,    40,    41,    42,
      43,    44,    45,    65,    47,    38,    39,    40,    41,    42,
      43,    44,    45,    71,    47,    66,    72,    67,    75,    50,
      67,    65,    65,    63,    38,    39,    40,    41,    42,    43,
      44,    45,    65,    47,    38,    39,    40,    41,    42,    43,
      44,    45,    50,    47,    67,    66,    76,    63,    75,    66,
      72,    65,    51,    66,    66,    75,    63,    72,    48,    46,
      66,    64,    71,     5,    66,    50,    66,    72,    18,    67,
      67,    66,    66,    66,     8,   221,   139,   288,    56,   183,
     204,   267,   190,   102,   199,   210,    87,   209,   271,    18,
     327,   133,   385,   440,   383,   455,   469,   402,   395
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const unsigned char yystos[] =
{
       0,     3,     4,    78,    79,    84,    50,    82,    82,     0,
      79,    63,    65,    85,    85,     5,    54,    55,    80,    86,
      87,   175,   176,    82,    82,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    18,    21,    65,    84,
     104,   105,   118,   121,   122,   123,   125,   135,   136,   139,
     140,   141,    66,    86,    16,    33,    34,    50,    56,    57,
      58,    59,    60,    61,    62,    83,    92,    94,    95,    96,
      97,    98,    99,   100,    33,   106,   106,   106,    83,   142,
      74,   110,   110,   110,   110,    74,   113,   124,    74,   107,
      52,   143,    68,    86,    63,    63,    63,    11,    12,    13,
      14,    15,    16,   126,   127,   128,   129,   130,    63,    63,
      63,    81,    95,    99,    57,    61,    56,    57,    58,    59,
      64,    68,    91,    70,    70,    70,    34,    71,    73,    83,
      83,    83,    83,    67,    63,    28,    49,   111,   115,    82,
      93,    93,    93,    93,    49,    51,    82,   112,   114,    74,
     124,    74,   113,    35,    36,   108,   109,    93,    63,    17,
      98,   100,   133,   134,    66,   110,   110,   110,   110,   124,
     107,    70,    56,    57,    64,    51,    52,    88,    89,    90,
     100,    70,    74,   102,   103,    71,    71,    71,   142,    75,
      67,    91,    64,   119,   119,   119,   119,    82,    75,    67,
      75,    93,    93,    75,    67,    65,    82,    76,   132,    82,
      67,    69,    81,    82,    82,    82,    82,    82,    82,    82,
      50,    67,    69,    82,    51,    83,   101,   103,   106,   106,
     106,   111,    93,   120,    63,    65,   137,   137,   137,   137,
      75,   114,   119,   119,   108,   100,   116,   117,    76,   131,
      51,    52,   132,   134,   119,   119,   119,   119,   119,    63,
      65,    89,    72,    75,    72,    72,    72,    67,    37,   138,
     139,   144,   145,   137,   137,    82,   117,    66,   100,   137,
     137,   137,   137,   137,   117,    71,   120,    74,   147,    66,
     138,    63,    74,    66,   100,   153,   156,   157,    20,    22,
      23,    24,    25,    26,    29,    30,    31,    32,    49,   148,
     149,    33,    50,    82,    95,    96,   146,    81,    75,    63,
      82,    53,    71,   152,    67,    72,    75,    67,    71,   158,
      82,    63,    74,    76,    65,    71,    74,   152,    75,   157,
     148,    72,   157,    38,    39,    40,    41,    42,    43,    44,
      45,    47,    65,   160,   166,   158,    51,    52,    83,   150,
     152,    53,   151,   152,    72,    72,    71,   170,    74,   170,
      50,   171,   172,    65,    52,   165,    50,   168,   170,    71,
     161,   166,    19,   159,    66,    67,    72,    75,   152,   152,
      50,   152,    74,   158,   173,    67,    65,   166,   162,   166,
      65,   154,    67,    63,   152,    50,    66,   161,    76,   160,
     152,   151,   152,   152,    63,    75,    72,   169,   152,   172,
      66,   161,    66,   161,   152,   168,   169,   158,    51,   152,
     170,    65,   166,    75,   174,    66,    66,   155,    65,   166,
      72,    63,   152,   161,   158,    48,   163,   161,    46,   167,
     154,   152,    64,    66,    71,    66,    65,   166,   152,   169,
     152,    50,   164,   167,   161,    66,    65,   166,    67,    67,
      72,    66,   161,   152,   164,    66,   169,    65,   166,   161,
      66
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
#line 133 "xi-grammar.y"
    { yyval.modlist = yyvsp[0].modlist; modlist = yyvsp[0].modlist; }
    break;

  case 3:
#line 137 "xi-grammar.y"
    { 
		  yyval.modlist = 0; 
		}
    break;

  case 4:
#line 141 "xi-grammar.y"
    { yyval.modlist = new ModuleList(lineno, yyvsp[-1].module, yyvsp[0].modlist); }
    break;

  case 5:
#line 145 "xi-grammar.y"
    { yyval.intval = 0; }
    break;

  case 6:
#line 147 "xi-grammar.y"
    { yyval.intval = 1; }
    break;

  case 7:
#line 151 "xi-grammar.y"
    { yyval.intval = 0; }
    break;

  case 8:
#line 153 "xi-grammar.y"
    { yyval.intval = 1; }
    break;

  case 9:
#line 157 "xi-grammar.y"
    { yyval.strval = yyvsp[0].strval; }
    break;

  case 10:
#line 161 "xi-grammar.y"
    { yyval.strval = yyvsp[0].strval; }
    break;

  case 11:
#line 163 "xi-grammar.y"
    {
		  char *tmp = new char[strlen(yyvsp[-3].strval)+strlen(yyvsp[0].strval)+3];
		  sprintf(tmp,"%s::%s", yyvsp[-3].strval, yyvsp[0].strval);
		  yyval.strval = tmp;
		}
    break;

  case 12:
#line 171 "xi-grammar.y"
    { 
		    yyval.module = new Module(lineno, yyvsp[-1].strval, yyvsp[0].conslist); 
		}
    break;

  case 13:
#line 175 "xi-grammar.y"
    {  
		    yyval.module = new Module(lineno, yyvsp[-1].strval, yyvsp[0].conslist); 
		    yyval.module->setMain();
		}
    break;

  case 14:
#line 182 "xi-grammar.y"
    { yyval.conslist = 0; }
    break;

  case 15:
#line 184 "xi-grammar.y"
    { yyval.conslist = yyvsp[-2].conslist; }
    break;

  case 16:
#line 188 "xi-grammar.y"
    { yyval.conslist = 0; }
    break;

  case 17:
#line 190 "xi-grammar.y"
    { yyval.conslist = new ConstructList(lineno, yyvsp[-1].construct, yyvsp[0].conslist); }
    break;

  case 18:
#line 194 "xi-grammar.y"
    { if(yyvsp[-2].conslist) yyvsp[-2].conslist->setExtern(yyvsp[-4].intval); yyval.construct = yyvsp[-2].conslist; }
    break;

  case 19:
#line 196 "xi-grammar.y"
    { yyvsp[0].module->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].module; }
    break;

  case 20:
#line 198 "xi-grammar.y"
    { yyvsp[0].member->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].member; }
    break;

  case 21:
#line 200 "xi-grammar.y"
    { yyvsp[-1].message->setExtern(yyvsp[-2].intval); yyval.construct = yyvsp[-1].message; }
    break;

  case 22:
#line 202 "xi-grammar.y"
    { yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; }
    break;

  case 23:
#line 204 "xi-grammar.y"
    { yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; }
    break;

  case 24:
#line 206 "xi-grammar.y"
    { yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; }
    break;

  case 25:
#line 208 "xi-grammar.y"
    { yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; }
    break;

  case 26:
#line 210 "xi-grammar.y"
    { yyvsp[0].templat->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].templat; }
    break;

  case 27:
#line 212 "xi-grammar.y"
    { yyval.construct = NULL; }
    break;

  case 28:
#line 214 "xi-grammar.y"
    { yyval.construct = NULL; }
    break;

  case 29:
#line 218 "xi-grammar.y"
    { yyval.tparam = new TParamType(yyvsp[0].type); }
    break;

  case 30:
#line 220 "xi-grammar.y"
    { yyval.tparam = new TParamVal(yyvsp[0].strval); }
    break;

  case 31:
#line 222 "xi-grammar.y"
    { yyval.tparam = new TParamVal(yyvsp[0].strval); }
    break;

  case 32:
#line 226 "xi-grammar.y"
    { yyval.tparlist = new TParamList(yyvsp[0].tparam); }
    break;

  case 33:
#line 228 "xi-grammar.y"
    { yyval.tparlist = new TParamList(yyvsp[-2].tparam, yyvsp[0].tparlist); }
    break;

  case 34:
#line 232 "xi-grammar.y"
    { yyval.tparlist = 0; }
    break;

  case 35:
#line 234 "xi-grammar.y"
    { yyval.tparlist = yyvsp[0].tparlist; }
    break;

  case 36:
#line 238 "xi-grammar.y"
    { yyval.tparlist = 0; }
    break;

  case 37:
#line 240 "xi-grammar.y"
    { yyval.tparlist = yyvsp[-1].tparlist; }
    break;

  case 38:
#line 244 "xi-grammar.y"
    { yyval.type = new BuiltinType("int"); }
    break;

  case 39:
#line 246 "xi-grammar.y"
    { yyval.type = new BuiltinType("long"); }
    break;

  case 40:
#line 248 "xi-grammar.y"
    { yyval.type = new BuiltinType("short"); }
    break;

  case 41:
#line 250 "xi-grammar.y"
    { yyval.type = new BuiltinType("char"); }
    break;

  case 42:
#line 252 "xi-grammar.y"
    { yyval.type = new BuiltinType("unsigned int"); }
    break;

  case 43:
#line 254 "xi-grammar.y"
    { yyval.type = new BuiltinType("unsigned long"); }
    break;

  case 44:
#line 256 "xi-grammar.y"
    { yyval.type = new BuiltinType("unsigned long"); }
    break;

  case 45:
#line 258 "xi-grammar.y"
    { yyval.type = new BuiltinType("unsigned long long"); }
    break;

  case 46:
#line 260 "xi-grammar.y"
    { yyval.type = new BuiltinType("unsigned short"); }
    break;

  case 47:
#line 262 "xi-grammar.y"
    { yyval.type = new BuiltinType("unsigned char"); }
    break;

  case 48:
#line 264 "xi-grammar.y"
    { yyval.type = new BuiltinType("long long"); }
    break;

  case 49:
#line 266 "xi-grammar.y"
    { yyval.type = new BuiltinType("float"); }
    break;

  case 50:
#line 268 "xi-grammar.y"
    { yyval.type = new BuiltinType("double"); }
    break;

  case 51:
#line 270 "xi-grammar.y"
    { yyval.type = new BuiltinType("long double"); }
    break;

  case 52:
#line 272 "xi-grammar.y"
    { yyval.type = new BuiltinType("void"); }
    break;

  case 53:
#line 275 "xi-grammar.y"
    { yyval.ntype = new NamedType(yyvsp[-1].strval,yyvsp[0].tparlist); }
    break;

  case 54:
#line 276 "xi-grammar.y"
    { yyval.ntype = new NamedType(yyvsp[-1].strval,yyvsp[0].tparlist); }
    break;

  case 55:
#line 279 "xi-grammar.y"
    { yyval.type = yyvsp[0].type; }
    break;

  case 56:
#line 281 "xi-grammar.y"
    { yyval.type = yyvsp[0].ntype; }
    break;

  case 57:
#line 285 "xi-grammar.y"
    { yyval.ptype = new PtrType(yyvsp[-1].type); }
    break;

  case 58:
#line 289 "xi-grammar.y"
    { yyvsp[-1].ptype->indirect(); yyval.ptype = yyvsp[-1].ptype; }
    break;

  case 59:
#line 291 "xi-grammar.y"
    { yyvsp[-1].ptype->indirect(); yyval.ptype = yyvsp[-1].ptype; }
    break;

  case 60:
#line 295 "xi-grammar.y"
    { yyval.ftype = new FuncType(yyvsp[-7].type, yyvsp[-4].strval, yyvsp[-1].plist); }
    break;

  case 61:
#line 299 "xi-grammar.y"
    { yyval.type = yyvsp[0].type; }
    break;

  case 62:
#line 301 "xi-grammar.y"
    { yyval.type = yyvsp[0].ptype; }
    break;

  case 63:
#line 303 "xi-grammar.y"
    { yyval.type = yyvsp[0].ptype; }
    break;

  case 64:
#line 305 "xi-grammar.y"
    { yyval.type = yyvsp[0].ftype; }
    break;

  case 65:
#line 308 "xi-grammar.y"
    { yyval.type = yyvsp[0].type; }
    break;

  case 66:
#line 310 "xi-grammar.y"
    { yyval.type = yyvsp[-1].type; }
    break;

  case 67:
#line 314 "xi-grammar.y"
    { yyval.type = new ReferenceType(yyvsp[-1].type); }
    break;

  case 68:
#line 316 "xi-grammar.y"
    { yyval.type = yyvsp[0].type; }
    break;

  case 69:
#line 320 "xi-grammar.y"
    { yyval.val = new Value(yyvsp[0].strval); }
    break;

  case 70:
#line 322 "xi-grammar.y"
    { yyval.val = new Value(yyvsp[0].strval); }
    break;

  case 71:
#line 326 "xi-grammar.y"
    { yyval.val = yyvsp[-1].val; }
    break;

  case 72:
#line 330 "xi-grammar.y"
    { yyval.vallist = 0; }
    break;

  case 73:
#line 332 "xi-grammar.y"
    { yyval.vallist = new ValueList(yyvsp[-1].val, yyvsp[0].vallist); }
    break;

  case 74:
#line 336 "xi-grammar.y"
    { yyval.readonly = new Readonly(lineno, yyvsp[-2].type, yyvsp[-1].strval, yyvsp[0].vallist); }
    break;

  case 75:
#line 340 "xi-grammar.y"
    { yyval.readonly = new Readonly(lineno, yyvsp[-2].type, yyvsp[0].strval, 0, 1); }
    break;

  case 76:
#line 344 "xi-grammar.y"
    { yyval.intval = 0;}
    break;

  case 77:
#line 346 "xi-grammar.y"
    { yyval.intval = 0;}
    break;

  case 78:
#line 350 "xi-grammar.y"
    { yyval.intval = 0; }
    break;

  case 79:
#line 352 "xi-grammar.y"
    { 
		  /*
		  printf("Warning: Message attributes are being phased out.\n");
		  printf("Warning: Please remove them from interface files.\n");
		  */
		  yyval.intval = yyvsp[-1].intval; 
		}
    break;

  case 80:
#line 362 "xi-grammar.y"
    { yyval.intval = yyvsp[0].intval; }
    break;

  case 81:
#line 364 "xi-grammar.y"
    { yyval.intval = yyvsp[-2].intval | yyvsp[0].intval; }
    break;

  case 82:
#line 368 "xi-grammar.y"
    { yyval.intval = 0; }
    break;

  case 83:
#line 370 "xi-grammar.y"
    { yyval.intval = 0; }
    break;

  case 84:
#line 374 "xi-grammar.y"
    { yyval.cattr = 0; }
    break;

  case 85:
#line 376 "xi-grammar.y"
    { yyval.cattr = yyvsp[-1].cattr; }
    break;

  case 86:
#line 380 "xi-grammar.y"
    { yyval.cattr = yyvsp[0].cattr; }
    break;

  case 87:
#line 382 "xi-grammar.y"
    { yyval.cattr = yyvsp[-2].cattr | yyvsp[0].cattr; }
    break;

  case 88:
#line 386 "xi-grammar.y"
    { yyval.cattr = Chare::CPYTHON; }
    break;

  case 89:
#line 390 "xi-grammar.y"
    { yyval.cattr = 0; }
    break;

  case 90:
#line 392 "xi-grammar.y"
    { yyval.cattr = yyvsp[-1].cattr; }
    break;

  case 91:
#line 396 "xi-grammar.y"
    { yyval.cattr = yyvsp[0].cattr; }
    break;

  case 92:
#line 398 "xi-grammar.y"
    { yyval.cattr = yyvsp[-2].cattr | yyvsp[0].cattr; }
    break;

  case 93:
#line 402 "xi-grammar.y"
    { yyval.cattr = Chare::CMIGRATABLE; }
    break;

  case 94:
#line 404 "xi-grammar.y"
    { yyval.cattr = Chare::CPYTHON; }
    break;

  case 95:
#line 408 "xi-grammar.y"
    { yyval.mv = new MsgVar(yyvsp[-4].type, yyvsp[-3].strval); }
    break;

  case 96:
#line 412 "xi-grammar.y"
    { yyval.mvlist = new MsgVarList(yyvsp[0].mv); }
    break;

  case 97:
#line 414 "xi-grammar.y"
    { yyval.mvlist = new MsgVarList(yyvsp[-1].mv, yyvsp[0].mvlist); }
    break;

  case 98:
#line 418 "xi-grammar.y"
    { yyval.message = new Message(lineno, yyvsp[0].ntype); }
    break;

  case 99:
#line 420 "xi-grammar.y"
    { yyval.message = new Message(lineno, yyvsp[-3].ntype, yyvsp[-1].mvlist); }
    break;

  case 100:
#line 424 "xi-grammar.y"
    { yyval.typelist = 0; }
    break;

  case 101:
#line 426 "xi-grammar.y"
    { yyval.typelist = yyvsp[0].typelist; }
    break;

  case 102:
#line 430 "xi-grammar.y"
    { yyval.typelist = new TypeList(yyvsp[0].ntype); }
    break;

  case 103:
#line 432 "xi-grammar.y"
    { yyval.typelist = new TypeList(yyvsp[-2].ntype, yyvsp[0].typelist); }
    break;

  case 104:
#line 436 "xi-grammar.y"
    { yyval.chare = new Chare(lineno, yyvsp[-3].cattr, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 105:
#line 438 "xi-grammar.y"
    { yyval.chare = new MainChare(lineno, yyvsp[-3].cattr, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 106:
#line 442 "xi-grammar.y"
    { yyval.chare = new Group(lineno, yyvsp[-3].cattr, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 107:
#line 446 "xi-grammar.y"
    { yyval.chare = new NodeGroup(lineno, yyvsp[-3].cattr, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 108:
#line 450 "xi-grammar.y"
    {/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",yyvsp[-2].strval);
			yyval.ntype = new NamedType(buf); 
		}
    break;

  case 109:
#line 456 "xi-grammar.y"
    { yyval.ntype = new NamedType(yyvsp[-1].strval); }
    break;

  case 110:
#line 460 "xi-grammar.y"
    {  yyval.chare = new Array(lineno, yyvsp[-4].cattr, yyvsp[-3].ntype, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 111:
#line 462 "xi-grammar.y"
    {  yyval.chare = new Array(lineno, yyvsp[-3].cattr, yyvsp[-4].ntype, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 112:
#line 466 "xi-grammar.y"
    { yyval.chare = new Chare(lineno, yyvsp[-3].cattr, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist);}
    break;

  case 113:
#line 468 "xi-grammar.y"
    { yyval.chare = new MainChare(lineno, yyvsp[-3].cattr, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 114:
#line 472 "xi-grammar.y"
    { yyval.chare = new Group(lineno, yyvsp[-3].cattr, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 115:
#line 476 "xi-grammar.y"
    { yyval.chare = new NodeGroup( lineno, yyvsp[-3].cattr, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 116:
#line 480 "xi-grammar.y"
    { yyval.chare = new Array( lineno, 0, yyvsp[-3].ntype, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 117:
#line 484 "xi-grammar.y"
    { yyval.message = new Message(lineno, new NamedType(yyvsp[-1].strval)); }
    break;

  case 118:
#line 486 "xi-grammar.y"
    { yyval.message = new Message(lineno, new NamedType(yyvsp[-4].strval), yyvsp[-2].mvlist); }
    break;

  case 119:
#line 490 "xi-grammar.y"
    { yyval.type = 0; }
    break;

  case 120:
#line 492 "xi-grammar.y"
    { yyval.type = yyvsp[0].type; }
    break;

  case 121:
#line 496 "xi-grammar.y"
    { yyval.strval = 0; }
    break;

  case 122:
#line 498 "xi-grammar.y"
    { yyval.strval = yyvsp[0].strval; }
    break;

  case 123:
#line 500 "xi-grammar.y"
    { yyval.strval = yyvsp[0].strval; }
    break;

  case 124:
#line 504 "xi-grammar.y"
    { yyval.tvar = new TType(new NamedType(yyvsp[-1].strval), yyvsp[0].type); }
    break;

  case 125:
#line 506 "xi-grammar.y"
    { yyval.tvar = new TFunc(yyvsp[-1].ftype, yyvsp[0].strval); }
    break;

  case 126:
#line 508 "xi-grammar.y"
    { yyval.tvar = new TName(yyvsp[-2].type, yyvsp[-1].strval, yyvsp[0].strval); }
    break;

  case 127:
#line 512 "xi-grammar.y"
    { yyval.tvarlist = new TVarList(yyvsp[0].tvar); }
    break;

  case 128:
#line 514 "xi-grammar.y"
    { yyval.tvarlist = new TVarList(yyvsp[-2].tvar, yyvsp[0].tvarlist); }
    break;

  case 129:
#line 518 "xi-grammar.y"
    { yyval.tvarlist = yyvsp[-1].tvarlist; }
    break;

  case 130:
#line 522 "xi-grammar.y"
    { yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); }
    break;

  case 131:
#line 524 "xi-grammar.y"
    { yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); }
    break;

  case 132:
#line 526 "xi-grammar.y"
    { yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); }
    break;

  case 133:
#line 528 "xi-grammar.y"
    { yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); }
    break;

  case 134:
#line 530 "xi-grammar.y"
    { yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].message); yyvsp[0].message->setTemplate(yyval.templat); }
    break;

  case 135:
#line 534 "xi-grammar.y"
    { yyval.mbrlist = 0; }
    break;

  case 136:
#line 536 "xi-grammar.y"
    { yyval.mbrlist = yyvsp[-2].mbrlist; }
    break;

  case 137:
#line 540 "xi-grammar.y"
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

  case 138:
#line 559 "xi-grammar.y"
    { yyval.mbrlist = new MemberList(yyvsp[-1].member, yyvsp[0].mbrlist); }
    break;

  case 139:
#line 563 "xi-grammar.y"
    { yyval.member = yyvsp[-1].readonly; }
    break;

  case 140:
#line 565 "xi-grammar.y"
    { yyval.member = yyvsp[-1].readonly; }
    break;

  case 142:
#line 568 "xi-grammar.y"
    { yyval.member = yyvsp[-1].member; }
    break;

  case 143:
#line 570 "xi-grammar.y"
    { yyval.member = yyvsp[-1].pupable; }
    break;

  case 144:
#line 572 "xi-grammar.y"
    { yyval.member = yyvsp[-1].includeFile; }
    break;

  case 145:
#line 576 "xi-grammar.y"
    { yyval.member = new InitCall(lineno, yyvsp[0].strval, 1); }
    break;

  case 146:
#line 578 "xi-grammar.y"
    { yyval.member = new InitCall(lineno, yyvsp[-3].strval, 1); }
    break;

  case 147:
#line 580 "xi-grammar.y"
    { printf("Warning: deprecated use of initcall. Use initnode or initproc instead.\n"); 
		  yyval.member = new InitCall(lineno, yyvsp[0].strval, 1); }
    break;

  case 148:
#line 583 "xi-grammar.y"
    { printf("Warning: deprecated use of initcall. Use initnode or initproc instead.\n");
		  yyval.member = new InitCall(lineno, yyvsp[-3].strval, 1); }
    break;

  case 149:
#line 588 "xi-grammar.y"
    { yyval.member = new InitCall(lineno, yyvsp[0].strval, 0); }
    break;

  case 150:
#line 590 "xi-grammar.y"
    { yyval.member = new InitCall(lineno, yyvsp[-3].strval, 0); }
    break;

  case 151:
#line 594 "xi-grammar.y"
    { yyval.pupable = new PUPableClass(lineno,yyvsp[0].strval,0); }
    break;

  case 152:
#line 596 "xi-grammar.y"
    { yyval.pupable = new PUPableClass(lineno,yyvsp[-2].strval,yyvsp[0].pupable); }
    break;

  case 153:
#line 600 "xi-grammar.y"
    { yyval.includeFile = new IncludeFile(lineno,yyvsp[0].strval,0); }
    break;

  case 154:
#line 604 "xi-grammar.y"
    { yyval.member = yyvsp[-1].entry; }
    break;

  case 155:
#line 606 "xi-grammar.y"
    { yyval.member = yyvsp[0].member; }
    break;

  case 156:
#line 610 "xi-grammar.y"
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

  case 157:
#line 621 "xi-grammar.y"
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

  case 158:
#line 634 "xi-grammar.y"
    { yyval.type = new BuiltinType("void"); }
    break;

  case 159:
#line 636 "xi-grammar.y"
    { yyval.type = yyvsp[0].ptype; }
    break;

  case 160:
#line 640 "xi-grammar.y"
    { yyval.intval = 0; }
    break;

  case 161:
#line 642 "xi-grammar.y"
    { yyval.intval = yyvsp[-1].intval; }
    break;

  case 162:
#line 646 "xi-grammar.y"
    { yyval.intval = yyvsp[0].intval; }
    break;

  case 163:
#line 648 "xi-grammar.y"
    { yyval.intval = yyvsp[-2].intval | yyvsp[0].intval; }
    break;

  case 164:
#line 652 "xi-grammar.y"
    { yyval.intval = STHREADED; }
    break;

  case 165:
#line 654 "xi-grammar.y"
    { yyval.intval = SSYNC; }
    break;

  case 166:
#line 656 "xi-grammar.y"
    { yyval.intval = SLOCKED; }
    break;

  case 167:
#line 658 "xi-grammar.y"
    { yyval.intval = SCREATEHERE; }
    break;

  case 168:
#line 660 "xi-grammar.y"
    { yyval.intval = SCREATEHOME; }
    break;

  case 169:
#line 662 "xi-grammar.y"
    { yyval.intval = SNOKEEP; }
    break;

  case 170:
#line 664 "xi-grammar.y"
    { yyval.intval = SNOTRACE; }
    break;

  case 171:
#line 666 "xi-grammar.y"
    { yyval.intval = SIMMEDIATE; }
    break;

  case 172:
#line 668 "xi-grammar.y"
    { yyval.intval = SSKIPSCHED; }
    break;

  case 173:
#line 670 "xi-grammar.y"
    { yyval.intval = SINLINE; }
    break;

  case 174:
#line 672 "xi-grammar.y"
    { yyval.intval = SPYTHON; }
    break;

  case 175:
#line 676 "xi-grammar.y"
    { yyval.val = new Value(yyvsp[0].strval); }
    break;

  case 176:
#line 678 "xi-grammar.y"
    { yyval.val = new Value(yyvsp[0].strval); }
    break;

  case 177:
#line 680 "xi-grammar.y"
    { yyval.val = new Value(yyvsp[0].strval); }
    break;

  case 178:
#line 684 "xi-grammar.y"
    { yyval.strval = ""; }
    break;

  case 179:
#line 686 "xi-grammar.y"
    { yyval.strval = yyvsp[0].strval; }
    break;

  case 180:
#line 688 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen(yyvsp[-2].strval)+strlen(yyvsp[0].strval)+3];
			sprintf(tmp,"%s, %s", yyvsp[-2].strval, yyvsp[0].strval);
			yyval.strval = tmp;
		}
    break;

  case 181:
#line 696 "xi-grammar.y"
    { yyval.strval = ""; }
    break;

  case 182:
#line 698 "xi-grammar.y"
    { yyval.strval = yyvsp[0].strval; }
    break;

  case 183:
#line 700 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen(yyvsp[-4].strval)+strlen(yyvsp[-2].strval)+strlen(yyvsp[0].strval)+3];
			sprintf(tmp,"%s[%s]%s", yyvsp[-4].strval, yyvsp[-2].strval, yyvsp[0].strval);
			yyval.strval = tmp;
		}
    break;

  case 184:
#line 706 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen(yyvsp[-4].strval)+strlen(yyvsp[-2].strval)+strlen(yyvsp[0].strval)+3];
			sprintf(tmp,"%s{%s}%s", yyvsp[-4].strval, yyvsp[-2].strval, yyvsp[0].strval);
			yyval.strval = tmp;
		}
    break;

  case 185:
#line 712 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen(yyvsp[-4].strval)+strlen(yyvsp[-2].strval)+strlen(yyvsp[0].strval)+3];
			sprintf(tmp,"%s(%s)%s", yyvsp[-4].strval, yyvsp[-2].strval, yyvsp[0].strval);
			yyval.strval = tmp;
		}
    break;

  case 186:
#line 718 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen(yyvsp[-2].strval)+strlen(yyvsp[0].strval)+3];
			sprintf(tmp,"(%s)%s", yyvsp[-2].strval, yyvsp[0].strval);
			yyval.strval = tmp;
		}
    break;

  case 187:
#line 726 "xi-grammar.y"
    {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			yyval.pname = new Parameter(lineno, yyvsp[-2].type,yyvsp[-1].strval);
		}
    break;

  case 188:
#line 733 "xi-grammar.y"
    { 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			yyval.intval = 0;
		}
    break;

  case 189:
#line 741 "xi-grammar.y"
    { 
			in_braces=0;
			yyval.intval = 0;
		}
    break;

  case 190:
#line 748 "xi-grammar.y"
    { yyval.pname = new Parameter(lineno, yyvsp[0].type);}
    break;

  case 191:
#line 750 "xi-grammar.y"
    { yyval.pname = new Parameter(lineno, yyvsp[-1].type,yyvsp[0].strval);}
    break;

  case 192:
#line 752 "xi-grammar.y"
    { yyval.pname = new Parameter(lineno, yyvsp[-3].type,yyvsp[-2].strval,0,yyvsp[0].val);}
    break;

  case 193:
#line 754 "xi-grammar.y"
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			yyval.pname = new Parameter(lineno, yyvsp[-2].pname->getType(), yyvsp[-2].pname->getName() ,yyvsp[-1].strval);
		}
    break;

  case 194:
#line 761 "xi-grammar.y"
    { yyval.plist = new ParamList(yyvsp[0].pname); }
    break;

  case 195:
#line 763 "xi-grammar.y"
    { yyval.plist = new ParamList(yyvsp[-2].pname,yyvsp[0].plist); }
    break;

  case 196:
#line 767 "xi-grammar.y"
    { yyval.plist = yyvsp[-1].plist; }
    break;

  case 197:
#line 769 "xi-grammar.y"
    { yyval.plist = 0; }
    break;

  case 198:
#line 773 "xi-grammar.y"
    { yyval.val = 0; }
    break;

  case 199:
#line 775 "xi-grammar.y"
    { yyval.val = new Value(yyvsp[0].strval); }
    break;

  case 200:
#line 779 "xi-grammar.y"
    { yyval.sc = 0; }
    break;

  case 201:
#line 781 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SSDAGENTRY, yyvsp[0].sc); }
    break;

  case 202:
#line 783 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SSDAGENTRY, yyvsp[-1].sc); }
    break;

  case 203:
#line 787 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SSLIST, yyvsp[0].sc); }
    break;

  case 204:
#line 789 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SSLIST, yyvsp[-1].sc, yyvsp[0].sc);  }
    break;

  case 205:
#line 793 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SOLIST, yyvsp[0].sc); }
    break;

  case 206:
#line 795 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SOLIST, yyvsp[-1].sc, yyvsp[0].sc); }
    break;

  case 207:
#line 799 "xi-grammar.y"
    { yyval.sc = 0; }
    break;

  case 208:
#line 801 "xi-grammar.y"
    { yyval.sc = yyvsp[-1].sc; }
    break;

  case 209:
#line 805 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SPUBLISHES, new SdagConstruct(SIDENT, yyvsp[0].strval)); }
    break;

  case 210:
#line 807 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SPUBLISHES, new SdagConstruct(SIDENT, yyvsp[-2].strval), yyvsp[0].sc);  }
    break;

  case 211:
#line 811 "xi-grammar.y"
    { yyval.strval = yyvsp[0].strval; }
    break;

  case 212:
#line 813 "xi-grammar.y"
    { yyval.strval = 0; }
    break;

  case 213:
#line 817 "xi-grammar.y"
    { RemoveSdagComments(yyvsp[-2].strval);
		   yyval.sc = new SdagConstruct(SATOMIC, new XStr(yyvsp[-2].strval), yyvsp[0].sc, 0,0,0,0, 0 ); 
		   if (yyvsp[-4].strval) { yyvsp[-4].strval[strlen(yyvsp[-4].strval)-1]=0; yyval.sc->traceName = new XStr(yyvsp[-4].strval+1); }
		 }
    break;

  case 214:
#line 822 "xi-grammar.y"
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

  case 215:
#line 836 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SWHEN, 0, 0,0,0,0,0,yyvsp[-2].entrylist); }
    break;

  case 216:
#line 838 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SWHEN, 0, 0, 0,0,0, yyvsp[0].sc, yyvsp[-1].entrylist); }
    break;

  case 217:
#line 840 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SWHEN, 0, 0, 0,0,0, yyvsp[-1].sc, yyvsp[-3].entrylist); }
    break;

  case 218:
#line 842 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SOVERLAP,0, 0,0,0,0,yyvsp[-1].sc, 0); }
    break;

  case 219:
#line 844 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SFOR, 0, new SdagConstruct(SINT_EXPR, yyvsp[-8].strval), new SdagConstruct(SINT_EXPR, yyvsp[-6].strval),
		             new SdagConstruct(SINT_EXPR, yyvsp[-4].strval), 0, yyvsp[-1].sc, 0); }
    break;

  case 220:
#line 847 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SFOR, 0, new SdagConstruct(SINT_EXPR, yyvsp[-6].strval), new SdagConstruct(SINT_EXPR, yyvsp[-4].strval), 
		         new SdagConstruct(SINT_EXPR, yyvsp[-2].strval), 0, yyvsp[0].sc, 0); }
    break;

  case 221:
#line 850 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SFORALL, 0, new SdagConstruct(SIDENT, yyvsp[-9].strval), new SdagConstruct(SINT_EXPR, yyvsp[-6].strval), 
		             new SdagConstruct(SINT_EXPR, yyvsp[-4].strval), new SdagConstruct(SINT_EXPR, yyvsp[-2].strval), yyvsp[0].sc, 0); }
    break;

  case 222:
#line 853 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SFORALL, 0, new SdagConstruct(SIDENT, yyvsp[-11].strval), new SdagConstruct(SINT_EXPR, yyvsp[-8].strval), 
		                 new SdagConstruct(SINT_EXPR, yyvsp[-6].strval), new SdagConstruct(SINT_EXPR, yyvsp[-4].strval), yyvsp[-1].sc, 0); }
    break;

  case 223:
#line 856 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, yyvsp[-3].strval), yyvsp[0].sc,0,0,yyvsp[-1].sc,0); }
    break;

  case 224:
#line 858 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, yyvsp[-5].strval), yyvsp[0].sc,0,0,yyvsp[-2].sc,0); }
    break;

  case 225:
#line 860 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, yyvsp[-2].strval), 0,0,0,yyvsp[0].sc,0); }
    break;

  case 226:
#line 862 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SWHILE, 0, new SdagConstruct(SINT_EXPR, yyvsp[-4].strval), 0,0,0,yyvsp[-1].sc,0); }
    break;

  case 227:
#line 864 "xi-grammar.y"
    { yyval.sc = yyvsp[-1].sc; }
    break;

  case 228:
#line 868 "xi-grammar.y"
    { yyval.sc = 0; }
    break;

  case 229:
#line 870 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SELSE, 0,0,0,0,0, yyvsp[0].sc,0); }
    break;

  case 230:
#line 872 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SELSE, 0,0,0,0,0, yyvsp[-1].sc,0); }
    break;

  case 231:
#line 875 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SFORWARD, new SdagConstruct(SIDENT, yyvsp[0].strval)); }
    break;

  case 232:
#line 877 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SFORWARD, new SdagConstruct(SIDENT, yyvsp[-2].strval), yyvsp[0].sc);  }
    break;

  case 233:
#line 881 "xi-grammar.y"
    { in_int_expr = 0; yyval.intval = 0; }
    break;

  case 234:
#line 885 "xi-grammar.y"
    { in_int_expr = 1; yyval.intval = 0; }
    break;

  case 235:
#line 889 "xi-grammar.y"
    { 
		  if (yyvsp[0].plist != 0)
		     yyval.entry = new Entry(lineno, 0, 0, yyvsp[-1].strval, yyvsp[0].plist, 0, 0, 0, 0); 
		  else
		     yyval.entry = new Entry(lineno, 0, 0, yyvsp[-1].strval, 
				new ParamList(new Parameter(0, new BuiltinType("void"))), 0, 0, 0, 0); 
		}
    break;

  case 236:
#line 897 "xi-grammar.y"
    { if (yyvsp[0].plist != 0)
		    yyval.entry = new Entry(lineno, 0, 0, yyvsp[-4].strval, yyvsp[0].plist, 0, 0, yyvsp[-2].strval, 0); 
		  else
		    yyval.entry = new Entry(lineno, 0, 0, yyvsp[-4].strval, new ParamList(new Parameter(0, new BuiltinType("void"))), 0, 0, yyvsp[-2].strval, 0); 
		}
    break;

  case 237:
#line 905 "xi-grammar.y"
    { yyval.entrylist = new EntryList(yyvsp[0].entry); }
    break;

  case 238:
#line 907 "xi-grammar.y"
    { yyval.entrylist = new EntryList(yyvsp[-2].entry,yyvsp[0].entrylist); }
    break;

  case 239:
#line 911 "xi-grammar.y"
    { in_bracket=1; }
    break;

  case 240:
#line 914 "xi-grammar.y"
    { in_bracket=0; }
    break;

  case 241:
#line 918 "xi-grammar.y"
    { if (!macroDefined(yyvsp[0].strval, 1)) in_comment = 1; }
    break;

  case 242:
#line 922 "xi-grammar.y"
    { if (!macroDefined(yyvsp[0].strval, 0)) in_comment = 1; }
    break;


    }

/* Line 991 of yacc.c.  */
#line 2933 "y.tab.c"

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


#line 925 "xi-grammar.y"

void yyerror(const char *mesg)
{
  cout << cur_file<<":"<<lineno<<": Charmxi syntax error> " << mesg << endl;
  // return 0;
}

