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
     VIRTUAL = 281,
     MIGRATABLE = 282,
     CREATEHERE = 283,
     CREATEHOME = 284,
     NOKEEP = 285,
     NOTRACE = 286,
     VOID = 287,
     CONST = 288,
     PACKED = 289,
     VARSIZE = 290,
     ENTRY = 291,
     FOR = 292,
     FORALL = 293,
     WHILE = 294,
     WHEN = 295,
     OVERLAP = 296,
     ATOMIC = 297,
     FORWARD = 298,
     IF = 299,
     ELSE = 300,
     CONNECT = 301,
     PUBLISHES = 302,
     IDENT = 303,
     NUMBER = 304,
     LITERAL = 305,
     CPROGRAM = 306,
     HASHIF = 307,
     HASHIFDEF = 308,
     INT = 309,
     LONG = 310,
     SHORT = 311,
     CHAR = 312,
     FLOAT = 313,
     DOUBLE = 314,
     UNSIGNED = 315
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
#define VIRTUAL 281
#define MIGRATABLE 282
#define CREATEHERE 283
#define CREATEHOME 284
#define NOKEEP 285
#define NOTRACE 286
#define VOID 287
#define CONST 288
#define PACKED 289
#define VARSIZE 290
#define ENTRY 291
#define FOR 292
#define FORALL 293
#define WHILE 294
#define WHEN 295
#define OVERLAP 296
#define ATOMIC 297
#define FORWARD 298
#define IF 299
#define ELSE 300
#define CONNECT 301
#define PUBLISHES 302
#define IDENT 303
#define NUMBER 304
#define LITERAL 305
#define CPROGRAM 306
#define HASHIF 307
#define HASHIFDEF 308
#define INT 309
#define LONG 310
#define SHORT 311
#define CHAR 312
#define FLOAT 313
#define DOUBLE 314
#define UNSIGNED 315




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
#line 245 "y.tab.c"
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 214 of yacc.c.  */
#line 257 "y.tab.c"

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
#define YYLAST   528

/* YYNTOKENS -- Number of terminals. */
#define YYNTOKENS  75
/* YYNNTS -- Number of nonterminals. */
#define YYNNTS  97
/* YYNRULES -- Number of rules. */
#define YYNRULES  231
/* YYNRULES -- Number of states. */
#define YYNSTATES  462

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   315

#define YYTRANSLATE(YYX) 						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const unsigned char yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    71,     2,
      69,    70,    68,     2,    65,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    62,    61,
      66,    74,    67,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    72,     2,    73,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    63,     2,    64,     2,     2,     2,     2,
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
      55,    56,    57,    58,    59,    60
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
     102,   104,   106,   109,   112,   115,   118,   121,   123,   125,
     128,   130,   133,   136,   138,   140,   143,   146,   149,   158,
     160,   162,   164,   166,   169,   172,   175,   177,   179,   181,
     185,   186,   189,   194,   200,   201,   203,   204,   208,   210,
     214,   216,   218,   219,   223,   225,   229,   231,   237,   239,
     242,   246,   253,   254,   257,   259,   263,   269,   275,   281,
     287,   292,   296,   302,   308,   314,   320,   326,   332,   337,
     345,   346,   349,   350,   353,   356,   360,   363,   367,   369,
     373,   378,   381,   384,   387,   390,   393,   395,   400,   401,
     404,   407,   410,   413,   416,   420,   424,   428,   435,   439,
     446,   450,   457,   459,   463,   465,   468,   470,   478,   484,
     486,   488,   489,   493,   495,   499,   501,   503,   505,   507,
     509,   511,   513,   515,   517,   519,   521,   523,   524,   526,
     530,   531,   533,   539,   545,   551,   556,   560,   562,   564,
     566,   569,   574,   578,   580,   584,   588,   591,   592,   596,
     597,   599,   603,   605,   608,   610,   613,   614,   619,   621,
     625,   627,   628,   635,   644,   649,   653,   659,   664,   676,
     686,   699,   714,   721,   730,   736,   744,   748,   749,   752,
     757,   759,   763,   765,   767,   770,   776,   778,   782,   784,
     786,   789
};

/* YYRHS -- A `-1'-separated list of the rules' RHS. */
static const short yyrhs[] =
{
      76,     0,    -1,    77,    -1,    -1,    82,    77,    -1,    -1,
       5,    -1,    -1,    61,    -1,    48,    -1,    48,    -1,    81,
      62,    62,    48,    -1,     3,    80,    83,    -1,     4,    80,
      83,    -1,    61,    -1,    63,    84,    64,    79,    -1,    -1,
      85,    84,    -1,    78,    63,    84,    64,    79,    -1,    78,
      82,    -1,    78,   134,    -1,    78,   113,    61,    -1,    78,
     116,    -1,    78,   117,    -1,    78,   118,    -1,    78,   120,
      -1,    78,   131,    -1,   170,    -1,   171,    -1,    98,    -1,
      49,    -1,    50,    -1,    86,    -1,    86,    65,    87,    -1,
      -1,    87,    -1,    -1,    66,    88,    67,    -1,    54,    -1,
      55,    -1,    56,    -1,    57,    -1,    60,    54,    -1,    60,
      55,    -1,    60,    56,    -1,    60,    57,    -1,    55,    55,
      -1,    58,    -1,    59,    -1,    55,    59,    -1,    32,    -1,
      80,    89,    -1,    81,    89,    -1,    90,    -1,    92,    -1,
      93,    68,    -1,    94,    68,    -1,    95,    68,    -1,    97,
      69,    68,    80,    70,    69,   152,    70,    -1,    93,    -1,
      94,    -1,    95,    -1,    96,    -1,    33,    97,    -1,    97,
      33,    -1,    97,    71,    -1,    97,    -1,    49,    -1,    81,
      -1,    72,    99,    73,    -1,    -1,   100,   101,    -1,     6,
      98,    81,   101,    -1,     6,    16,    93,    68,    80,    -1,
      -1,    32,    -1,    -1,    72,   106,    73,    -1,   107,    -1,
     107,    65,   106,    -1,    34,    -1,    35,    -1,    -1,    72,
     109,    73,    -1,   110,    -1,   110,    65,   109,    -1,    27,
      -1,    98,    80,    72,    73,    61,    -1,   111,    -1,   111,
     112,    -1,    16,   105,    91,    -1,    16,   105,    91,    63,
     112,    64,    -1,    -1,    62,   115,    -1,    91,    -1,    91,
      65,   115,    -1,    11,   108,    91,   114,   132,    -1,    12,
     108,    91,   114,   132,    -1,    13,   108,    91,   114,   132,
      -1,    14,   108,    91,   114,   132,    -1,    72,    49,    80,
      73,    -1,    72,    80,    73,    -1,    15,   119,    91,   114,
     132,    -1,    11,   108,    80,   114,   132,    -1,    12,   108,
      80,   114,   132,    -1,    13,   108,    80,   114,   132,    -1,
      14,   108,    80,   114,   132,    -1,    15,   119,    80,   114,
     132,    -1,    16,   105,    80,    61,    -1,    16,   105,    80,
      63,   112,    64,    61,    -1,    -1,    74,    98,    -1,    -1,
      74,    49,    -1,    74,    50,    -1,    17,    80,   126,    -1,
      96,   127,    -1,    98,    80,   127,    -1,   128,    -1,   128,
      65,   129,    -1,    21,    66,   129,    67,    -1,   130,   121,
      -1,   130,   122,    -1,   130,   123,    -1,   130,   124,    -1,
     130,   125,    -1,    61,    -1,    63,   133,    64,    79,    -1,
      -1,   139,   133,    -1,   102,    61,    -1,   103,    61,    -1,
     136,    61,    -1,   135,    61,    -1,    10,   137,    61,    -1,
      18,   138,    61,    -1,     8,   104,    81,    -1,     8,   104,
      81,    69,   104,    70,    -1,     7,   104,    81,    -1,     7,
     104,    81,    69,   104,    70,    -1,     9,   104,    81,    -1,
       9,   104,    81,    69,   104,    70,    -1,    81,    -1,    81,
      65,   137,    -1,    50,    -1,   140,    61,    -1,   134,    -1,
      36,   142,   141,    80,   153,   154,   155,    -1,    36,   142,
      80,   153,   155,    -1,    32,    -1,    94,    -1,    -1,    72,
     143,    73,    -1,   144,    -1,   144,    65,   143,    -1,    20,
      -1,    22,    -1,    23,    -1,    28,    -1,    29,    -1,    30,
      -1,    31,    -1,    24,    -1,    25,    -1,    50,    -1,    49,
      -1,    81,    -1,    -1,    51,    -1,    51,    65,   146,    -1,
      -1,    51,    -1,    51,    72,   147,    73,   147,    -1,    51,
      63,   147,    64,   147,    -1,    51,    69,   146,    70,   147,
      -1,    69,   147,    70,   147,    -1,    98,    80,    72,    -1,
      63,    -1,    64,    -1,    98,    -1,    98,    80,    -1,    98,
      80,    74,   145,    -1,   148,   147,    73,    -1,   151,    -1,
     151,    65,   152,    -1,    69,   152,    70,    -1,    69,    70,
      -1,    -1,    19,    74,    49,    -1,    -1,   161,    -1,    63,
     156,    64,    -1,   161,    -1,   161,   156,    -1,   161,    -1,
     161,   156,    -1,    -1,    47,    69,   159,    70,    -1,    48,
      -1,    48,    65,   159,    -1,    50,    -1,    -1,    42,   160,
     149,   147,   150,   158,    -1,    46,    69,    48,   153,    70,
     149,   147,    64,    -1,    40,   167,    63,    64,    -1,    40,
     167,   161,    -1,    40,   167,    63,   156,    64,    -1,    41,
      63,   157,    64,    -1,    37,   165,   147,    61,   147,    61,
     147,   164,    63,   156,    64,    -1,    37,   165,   147,    61,
     147,    61,   147,   164,   161,    -1,    38,    72,    48,    73,
     165,   147,    62,   147,    65,   147,   164,   161,    -1,    38,
      72,    48,    73,   165,   147,    62,   147,    65,   147,   164,
      63,   156,    64,    -1,    44,   165,   147,   164,   161,   162,
      -1,    44,   165,   147,   164,    63,   156,    64,   162,    -1,
      39,   165,   147,   164,   161,    -1,    39,   165,   147,   164,
      63,   156,    64,    -1,    43,   163,    61,    -1,    -1,    45,
     161,    -1,    45,    63,   156,    64,    -1,    48,    -1,    48,
      65,   163,    -1,    70,    -1,    69,    -1,    48,   153,    -1,
      48,   168,   147,   169,   153,    -1,   166,    -1,   166,    65,
     167,    -1,    72,    -1,    73,    -1,    52,    80,    -1,    53,
      80,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const unsigned short yyrline[] =
{
       0,   130,   130,   135,   138,   143,   144,   149,   150,   154,
     158,   160,   168,   172,   179,   181,   186,   187,   191,   193,
     195,   197,   199,   201,   203,   205,   207,   209,   211,   215,
     217,   219,   223,   225,   230,   231,   236,   237,   241,   243,
     245,   247,   249,   251,   253,   255,   257,   259,   261,   263,
     265,   269,   270,   272,   274,   278,   282,   284,   288,   292,
     294,   296,   298,   301,   303,   307,   309,   313,   315,   319,
     324,   325,   329,   333,   338,   339,   344,   345,   355,   357,
     361,   363,   368,   369,   373,   375,   379,   383,   387,   389,
     393,   395,   400,   401,   405,   407,   411,   413,   417,   421,
     425,   431,   435,   439,   441,   445,   449,   453,   457,   459,
     464,   465,   470,   471,   473,   477,   479,   481,   485,   487,
     491,   495,   497,   499,   501,   503,   507,   509,   514,   532,
     536,   538,   540,   541,   543,   545,   549,   551,   553,   556,
     561,   563,   567,   569,   573,   577,   579,   583,   594,   607,
     609,   614,   615,   619,   621,   625,   627,   629,   631,   633,
     635,   637,   639,   641,   645,   647,   649,   654,   655,   657,
     666,   667,   669,   675,   681,   687,   695,   702,   710,   717,
     719,   721,   723,   730,   732,   736,   738,   743,   744,   749,
     750,   752,   756,   758,   762,   764,   769,   770,   774,   776,
     780,   783,   786,   791,   805,   807,   809,   811,   813,   816,
     819,   822,   825,   827,   829,   831,   833,   838,   839,   841,
     844,   846,   850,   854,   858,   866,   874,   876,   880,   883,
     887,   891
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
  "IMMEDIATE", "SKIPSCHED", "VIRTUAL", "MIGRATABLE", "CREATEHERE", 
  "CREATEHOME", "NOKEEP", "NOTRACE", "VOID", "CONST", "PACKED", "VARSIZE", 
  "ENTRY", "FOR", "FORALL", "WHILE", "WHEN", "OVERLAP", "ATOMIC", 
  "FORWARD", "IF", "ELSE", "CONNECT", "PUBLISHES", "IDENT", "NUMBER", 
  "LITERAL", "CPROGRAM", "HASHIF", "HASHIFDEF", "INT", "LONG", "SHORT", 
  "CHAR", "FLOAT", "DOUBLE", "UNSIGNED", "';'", "':'", "'{'", "'}'", 
  "','", "'<'", "'>'", "'*'", "'('", "')'", "'&'", "'['", "']'", "'='", 
  "$accept", "File", "ModuleEList", "OptExtern", "OptSemiColon", "Name", 
  "QualName", "Module", "ConstructEList", "ConstructList", "Construct", 
  "TParam", "TParamList", "TParamEList", "OptTParams", "BuiltinType", 
  "NamedType", "QualNamedType", "SimpleType", "OnePtrType", "PtrType", 
  "FuncType", "BaseType", "Type", "ArrayDim", "Dim", "DimList", 
  "Readonly", "ReadonlyMsg", "OptVoid", "MAttribs", "MAttribList", 
  "MAttrib", "CAttribs", "CAttribList", "CAttrib", "Var", "VarList", 
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
     315,    59,    58,   123,   125,    44,    60,    62,    42,    40,
      41,    38,    91,    93,    61
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const unsigned char yyr1[] =
{
       0,    75,    76,    77,    77,    78,    78,    79,    79,    80,
      81,    81,    82,    82,    83,    83,    84,    84,    85,    85,
      85,    85,    85,    85,    85,    85,    85,    85,    85,    86,
      86,    86,    87,    87,    88,    88,    89,    89,    90,    90,
      90,    90,    90,    90,    90,    90,    90,    90,    90,    90,
      90,    91,    92,    93,    93,    94,    95,    95,    96,    97,
      97,    97,    97,    97,    97,    98,    98,    99,    99,   100,
     101,   101,   102,   103,   104,   104,   105,   105,   106,   106,
     107,   107,   108,   108,   109,   109,   110,   111,   112,   112,
     113,   113,   114,   114,   115,   115,   116,   116,   117,   118,
     119,   119,   120,   121,   121,   122,   123,   124,   125,   125,
     126,   126,   127,   127,   127,   128,   128,   128,   129,   129,
     130,   131,   131,   131,   131,   131,   132,   132,   133,   133,
     134,   134,   134,   134,   134,   134,   135,   135,   135,   135,
     136,   136,   137,   137,   138,   139,   139,   140,   140,   141,
     141,   142,   142,   143,   143,   144,   144,   144,   144,   144,
     144,   144,   144,   144,   145,   145,   145,   146,   146,   146,
     147,   147,   147,   147,   147,   147,   148,   149,   150,   151,
     151,   151,   151,   152,   152,   153,   153,   154,   154,   155,
     155,   155,   156,   156,   157,   157,   158,   158,   159,   159,
     160,   160,   161,   161,   161,   161,   161,   161,   161,   161,
     161,   161,   161,   161,   161,   161,   161,   162,   162,   162,
     163,   163,   164,   165,   166,   166,   167,   167,   168,   169,
     170,   171
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const unsigned char yyr2[] =
{
       0,     2,     1,     0,     2,     0,     1,     0,     1,     1,
       1,     4,     3,     3,     1,     4,     0,     2,     5,     2,
       2,     3,     2,     2,     2,     2,     2,     1,     1,     1,
       1,     1,     1,     3,     0,     1,     0,     3,     1,     1,
       1,     1,     2,     2,     2,     2,     2,     1,     1,     2,
       1,     2,     2,     1,     1,     2,     2,     2,     8,     1,
       1,     1,     1,     2,     2,     2,     1,     1,     1,     3,
       0,     2,     4,     5,     0,     1,     0,     3,     1,     3,
       1,     1,     0,     3,     1,     3,     1,     5,     1,     2,
       3,     6,     0,     2,     1,     3,     5,     5,     5,     5,
       4,     3,     5,     5,     5,     5,     5,     5,     4,     7,
       0,     2,     0,     2,     2,     3,     2,     3,     1,     3,
       4,     2,     2,     2,     2,     2,     1,     4,     0,     2,
       2,     2,     2,     2,     3,     3,     3,     6,     3,     6,
       3,     6,     1,     3,     1,     2,     1,     7,     5,     1,
       1,     0,     3,     1,     3,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     0,     1,     3,
       0,     1,     5,     5,     5,     4,     3,     1,     1,     1,
       2,     4,     3,     1,     3,     3,     2,     0,     3,     0,
       1,     3,     1,     2,     1,     2,     0,     4,     1,     3,
       1,     0,     6,     8,     4,     3,     5,     4,    11,     9,
      12,    14,     6,     8,     5,     7,     3,     0,     2,     4,
       1,     3,     1,     1,     2,     5,     1,     3,     1,     1,
       2,     2
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const unsigned char yydefact[] =
{
       3,     0,     0,     0,     2,     3,     9,     0,     0,     1,
       4,    14,     5,    12,    13,     6,     0,     0,     0,     0,
       5,    27,    28,   230,   231,     0,    74,    74,    74,     0,
      82,    82,    82,    82,     0,    76,     0,     0,     5,    19,
       0,     0,     0,    22,    23,    24,    25,     0,    26,    20,
       0,     0,     7,    17,     0,    50,     0,    10,    38,    39,
      40,    41,    47,    48,     0,    36,    53,    54,    59,    60,
      61,    62,    66,     0,    75,     0,     0,     0,   142,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   144,
       0,     0,     0,   130,   131,    21,    82,    82,    82,    82,
       0,    76,   121,   122,   123,   124,   125,   133,   132,     8,
      15,     0,    63,    46,    49,    42,    43,    44,    45,     0,
      34,    52,    55,    56,    57,    64,     0,    65,    70,   138,
     136,   140,     0,   134,    86,     0,    84,    36,    92,    92,
      92,    92,     0,     0,    92,    80,    81,     0,    78,    90,
     135,     0,    62,     0,   118,     0,     7,     0,     0,     0,
       0,     0,     0,     0,     0,    30,    31,    32,    35,     0,
      29,     0,     0,    70,    72,    74,    74,    74,   143,    83,
       0,    51,     0,     0,     0,     0,     0,     0,   101,     0,
      77,     0,     0,   110,     0,   116,   112,     0,   120,    18,
      92,    92,    92,    92,    92,     0,    73,    11,     0,    37,
       0,    67,    68,     0,    71,     0,     0,     0,    85,    94,
      93,   126,   128,    96,    97,    98,    99,   100,   102,    79,
       0,    88,     0,     0,   115,   113,   114,   117,   119,     0,
       0,     0,     0,     0,   108,     0,    33,     0,    69,   139,
     137,   141,     0,   151,     0,   146,   128,     0,     0,    89,
      91,   111,   103,   104,   105,   106,   107,     0,     0,    95,
       0,     0,     7,   129,   145,     0,     0,   179,   170,   183,
       0,   155,   156,   157,   162,   163,   158,   159,   160,   161,
       0,   153,    50,    10,     0,     0,   150,     0,   127,     0,
     109,   180,   171,   170,     0,     0,    58,   152,     0,     0,
     189,     0,    87,   176,     0,   170,   167,   170,     0,   182,
     184,   154,   186,     0,     0,     0,     0,     0,     0,   201,
       0,     0,     0,     0,   148,   190,   187,   165,   164,   166,
     181,     0,   168,     0,     0,   170,   185,   223,   170,     0,
     170,     0,   226,     0,     0,   200,     0,   220,     0,   170,
       0,     0,   192,     0,   189,   170,   167,   170,   170,   175,
       0,     0,     0,   228,   224,   170,     0,     0,   205,     0,
     194,   177,   170,     0,   216,     0,     0,   191,   193,     0,
     147,   173,   169,   174,   172,   170,     0,   222,     0,     0,
     227,   204,     0,   207,   195,     0,   221,     0,     0,   188,
       0,   170,     0,   214,   229,     0,   206,   178,   196,     0,
     217,     0,   170,     0,     0,   225,     0,   202,     0,     0,
     212,   170,     0,   170,   215,     0,   217,     0,   218,     0,
       0,     0,   198,     0,   213,     0,   203,     0,   209,   170,
       0,   197,   219,     0,     0,   199,   208,     0,     0,   210,
       0,   211
};

/* YYDEFGOTO[NTERM-NUM]. */
static const short yydefgoto[] =
{
      -1,     3,     4,    18,   110,   137,    65,     5,    13,    19,
      20,   167,   168,   169,   121,    66,   219,    67,    68,    69,
      70,    71,    72,   230,   213,   173,   174,    40,    41,    75,
      88,   147,   148,    81,   135,   136,   231,   232,    42,   183,
     220,    43,    44,    45,    86,    46,   102,   103,   104,   105,
     106,   234,   195,   154,   155,    47,    48,   223,   254,   255,
      50,    51,    79,    90,   256,   257,   297,   271,   290,   291,
     340,   343,   304,   278,   382,   418,   279,   280,   310,   364,
     334,   361,   379,   427,   443,   356,   362,   430,   358,   398,
     348,   352,   353,   375,   415,    21,    22
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -373
static const short yypact[] =
{
      27,   -29,   -29,    59,  -373,    27,  -373,     4,     4,  -373,
    -373,  -373,     5,  -373,  -373,  -373,   -29,   -29,   185,     0,
       5,  -373,  -373,  -373,  -373,   186,    56,    56,    56,    78,
      63,    63,    63,    63,    67,    71,   113,   102,     5,  -373,
     114,   125,   129,  -373,  -373,  -373,  -373,   241,  -373,  -373,
     146,   150,   153,  -373,   305,  -373,   288,  -373,  -373,    42,
    -373,  -373,  -373,  -373,    55,    41,  -373,  -373,   147,   148,
     158,  -373,   -16,    78,  -373,    78,    78,    78,    65,   166,
     190,   -29,   -29,   -29,   -29,    69,   -29,   112,   -29,  -373,
     167,   245,   168,  -373,  -373,  -373,    63,    63,    63,    63,
      67,    71,  -373,  -373,  -373,  -373,  -373,  -373,  -373,  -373,
    -373,   162,   -13,  -373,  -373,  -373,  -373,  -373,  -373,   169,
     275,  -373,  -373,  -373,  -373,  -373,   170,  -373,   -35,   -30,
       9,    12,    78,  -373,  -373,   163,   172,   173,   187,   187,
     187,   187,   -29,   177,   187,  -373,  -373,   188,   193,   196,
    -373,   -29,   -31,   -29,   195,   201,   153,   -29,   -29,   -29,
     -29,   -29,   -29,   -29,   221,  -373,  -373,   206,  -373,   205,
    -373,   -29,   156,   202,  -373,    56,    56,    56,  -373,  -373,
     190,  -373,   -29,    81,    81,    81,    81,   200,  -373,    81,
    -373,   112,   288,   161,    26,  -373,   207,   245,  -373,  -373,
     187,   187,   187,   187,   187,   103,  -373,  -373,   275,  -373,
     209,  -373,   213,   210,  -373,   212,   216,   219,  -373,   220,
    -373,  -373,   215,  -373,  -373,  -373,  -373,  -373,  -373,  -373,
     -29,   288,   228,   288,  -373,  -373,  -373,  -373,  -373,    81,
      81,    81,    81,    81,  -373,   288,  -373,   211,  -373,  -373,
    -373,  -373,   -29,   222,   231,  -373,   215,   236,   226,  -373,
    -373,  -373,  -373,  -373,  -373,  -373,  -373,   246,   288,  -373,
     149,   318,   153,  -373,  -373,   238,   248,   -29,   -40,   254,
     252,  -373,  -373,  -373,  -373,  -373,  -373,  -373,  -373,  -373,
     253,   262,   290,   270,   271,   147,  -373,   -29,  -373,   280,
    -373,    30,   -18,   -40,   276,   288,  -373,  -373,   149,   258,
     352,   271,  -373,  -373,    46,   -40,   300,   -40,   282,  -373,
    -373,  -373,  -373,   284,   286,   285,   286,   308,   295,   317,
     320,   286,   301,   454,  -373,  -373,   350,  -373,  -373,   213,
    -373,   307,   322,   327,   326,   -40,  -373,  -373,   -40,   353,
     -40,   -25,   335,   370,   454,  -373,   339,   338,   343,   -40,
     357,   361,   454,   354,   352,   -40,   300,   -40,   -40,  -373,
     366,   356,   360,  -373,  -373,   -40,   308,   342,  -373,   367,
     454,  -373,   -40,   320,  -373,   360,   271,  -373,  -373,   383,
    -373,  -373,  -373,  -373,  -373,   -40,   286,  -373,   380,   382,
    -373,  -373,   389,  -373,  -373,   392,  -373,   398,   364,  -373,
     396,   -40,   454,  -373,  -373,   271,  -373,  -373,   411,   454,
     414,   339,   -40,   400,   417,  -373,   391,  -373,   419,   408,
    -373,   -40,   360,   -40,  -373,   437,   414,   454,  -373,   420,
     426,   421,   422,   418,  -373,   438,  -373,   454,  -373,   -40,
     437,  -373,  -373,   439,   360,  -373,  -373,   436,   454,  -373,
     440,  -373
};

/* YYPGOTO[NTERM-NUM].  */
static const short yypgoto[] =
{
    -373,  -373,   485,  -373,  -149,    -1,   -27,   483,   497,     2,
    -373,  -373,   298,  -373,   371,  -373,    50,  -373,   -51,   239,
    -373,   -83,   451,   -21,  -373,  -373,   336,  -373,  -373,   -22,
     410,   321,  -373,    -7,   333,  -373,  -373,  -203,  -373,   -19,
     263,  -373,  -373,  -373,   416,  -373,  -373,  -373,  -373,  -373,
    -373,  -373,   323,  -373,   324,  -373,  -373,    24,   261,   496,
    -373,  -373,   386,  -373,  -373,  -373,  -373,  -373,   214,  -373,
    -373,   154,  -282,  -373,   104,  -373,  -373,  -181,  -299,  -373,
     159,  -339,  -373,  -373,    74,  -373,  -292,    90,   144,  -372,
    -317,  -373,   152,  -373,  -373,  -373,  -373
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -150
static const short yytable[] =
{
       7,     8,    78,   111,    73,    76,    77,   199,   152,   350,
      15,   302,   336,   407,   359,    23,    24,   125,   335,     6,
     125,   318,    53,   388,    82,    83,    84,   119,   259,   303,
       1,     2,   119,   341,  -112,   344,  -112,   172,   402,   175,
      92,   404,   267,   194,   309,   315,   128,   373,   129,   130,
     131,   316,   374,   126,   317,   127,   126,    16,    17,     9,
     440,   378,   380,   369,    52,    11,   370,    12,   372,   -16,
     153,   119,   335,   424,   119,   235,   236,   385,   176,   411,
     428,   177,   457,   391,   143,   393,   394,   408,    74,   157,
     158,   159,   160,   399,    57,   337,   338,   113,   445,   170,
     405,   114,   313,   119,   314,    78,   413,   120,   453,   115,
     116,   117,   118,   410,   152,   420,   425,     6,   142,   460,
     184,   185,   186,   298,   320,   189,    57,   119,   323,   423,
     132,   138,   139,   140,   141,    80,   144,   438,   149,    85,
     432,   187,   221,    87,   222,   212,   145,   146,   448,   439,
     193,   441,   196,   215,   216,   217,   200,   201,   202,   203,
     204,   205,   206,    89,   244,   459,   245,   454,    91,   281,
     210,   282,   283,   284,   285,    93,   153,   286,   287,   288,
     289,   239,   240,   241,   242,   243,    94,   170,     1,     2,
      95,    25,    26,    27,    28,    29,    30,    31,    32,    33,
      34,    35,    54,    36,    57,   211,    37,   107,   224,   225,
     226,   108,   261,   228,   109,   122,   123,   134,    55,    56,
     295,    25,    26,    27,    28,    29,   124,   133,   150,   258,
     163,   164,   156,    36,    57,   233,   179,   180,   171,   120,
      58,    59,    60,    61,    62,    63,    64,   277,    38,   182,
     188,   253,    96,    97,    98,    99,   100,   101,   191,   192,
     197,   190,   151,   262,   263,   264,   265,   266,   198,   207,
     294,   208,   209,   227,   172,   119,   301,    55,    56,   247,
     268,   194,   249,   248,   277,   252,   250,   339,   277,   251,
      55,    56,   260,    57,   270,   272,   311,   274,   275,    58,
      59,    60,    61,    62,    63,    64,    57,    55,    56,   300,
     276,   299,    58,    59,    60,    61,    62,    63,    64,   305,
      55,    56,   306,    57,   165,   166,   307,   308,   322,    58,
      59,    60,    61,    62,    63,    64,    57,    55,  -149,    -9,
     309,   312,    58,    59,    60,    61,    62,    63,    64,   319,
     292,   342,   345,    57,   346,   347,   351,   349,   354,    58,
      59,    60,    61,    62,    63,    64,   293,   355,   357,   363,
     360,   365,    58,    59,    60,    61,    62,    63,    64,   324,
     325,   326,   327,   328,   329,   330,   331,   366,   332,   324,
     325,   326,   327,   328,   329,   330,   331,   367,   332,   368,
     376,   371,   381,   383,   384,   386,   401,   324,   325,   326,
     327,   328,   329,   330,   331,   333,   332,   324,   325,   326,
     327,   328,   329,   330,   331,   387,   332,   395,   389,   396,
     397,   403,   409,   377,   421,   324,   325,   326,   327,   328,
     329,   330,   331,   412,   332,   324,   325,   326,   327,   328,
     329,   330,   331,   416,   332,   414,   417,   422,   426,   429,
     435,   419,   433,   324,   325,   326,   327,   328,   329,   330,
     331,   437,   332,   324,   325,   326,   327,   328,   329,   330,
     331,   434,   332,   436,   446,   442,   449,   450,   451,   447,
      10,   324,   325,   326,   327,   328,   329,   330,   331,   458,
     332,    39,   452,   456,   461,    14,   246,   112,   181,   214,
     296,   162,   229,   218,    49,   269,   161,   273,   178,   237,
     392,   238,   321,   390,   455,   431,   444,   406,   400
};

static const unsigned short yycheck[] =
{
       1,     2,    29,    54,    25,    27,    28,   156,    91,   326,
       5,    51,   311,   385,   331,    16,    17,    33,   310,    48,
      33,   303,    20,   362,    31,    32,    33,    62,   231,    69,
       3,     4,    62,   315,    65,   317,    67,    72,   377,    69,
      38,   380,   245,    74,    69,    63,    73,    72,    75,    76,
      77,    69,   351,    69,    72,    71,    69,    52,    53,     0,
     432,   353,   354,   345,    64,    61,   348,    63,   350,    64,
      91,    62,   364,   412,    62,    49,    50,   359,    69,   396,
     419,    69,   454,   365,    85,   367,   368,   386,    32,    96,
      97,    98,    99,   375,    48,    49,    50,    55,   437,   120,
     382,    59,    72,    62,    74,   132,   398,    66,   447,    54,
      55,    56,    57,   395,   197,   407,   415,    48,    49,   458,
     139,   140,   141,   272,   305,   144,    48,    62,   309,   411,
      65,    81,    82,    83,    84,    72,    86,   429,    88,    72,
     422,   142,    61,    72,    63,   172,    34,    35,   440,   431,
     151,   433,   153,   175,   176,   177,   157,   158,   159,   160,
     161,   162,   163,    50,    61,   457,    63,   449,    66,    20,
     171,    22,    23,    24,    25,    61,   197,    28,    29,    30,
      31,   200,   201,   202,   203,   204,    61,   208,     3,     4,
      61,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    16,    18,    48,    49,    21,    61,   184,   185,
     186,    61,   233,   189,    61,    68,    68,    27,    32,    33,
     271,     6,     7,     8,     9,    10,    68,    61,    61,   230,
      68,    62,    64,    18,    48,    74,    73,    65,    68,    66,
      54,    55,    56,    57,    58,    59,    60,   268,    63,    62,
      73,    36,    11,    12,    13,    14,    15,    16,    65,    63,
      65,    73,    17,   239,   240,   241,   242,   243,    67,    48,
     271,    65,    67,    73,    72,    62,   277,    32,    33,    70,
      69,    74,    70,    73,   305,    65,    70,   314,   309,    70,
      32,    33,    64,    48,    72,    64,   297,    61,    72,    54,
      55,    56,    57,    58,    59,    60,    48,    32,    33,    61,
      64,    73,    54,    55,    56,    57,    58,    59,    60,    65,
      32,    33,    70,    48,    49,    50,    73,    65,    70,    54,
      55,    56,    57,    58,    59,    60,    48,    32,    48,    69,
      69,    61,    54,    55,    56,    57,    58,    59,    60,    73,
      32,    51,    70,    48,    70,    69,    48,    72,    63,    54,
      55,    56,    57,    58,    59,    60,    48,    50,    48,    19,
      69,    64,    54,    55,    56,    57,    58,    59,    60,    37,
      38,    39,    40,    41,    42,    43,    44,    65,    46,    37,
      38,    39,    40,    41,    42,    43,    44,    70,    46,    73,
      65,    48,    63,    65,    61,    48,    64,    37,    38,    39,
      40,    41,    42,    43,    44,    63,    46,    37,    38,    39,
      40,    41,    42,    43,    44,    64,    46,    61,    74,    73,
      70,    64,    49,    63,    70,    37,    38,    39,    40,    41,
      42,    43,    44,    63,    46,    37,    38,    39,    40,    41,
      42,    43,    44,    64,    46,    73,    64,    61,    47,    45,
      69,    63,    62,    37,    38,    39,    40,    41,    42,    43,
      44,    63,    46,    37,    38,    39,    40,    41,    42,    43,
      44,    64,    46,    64,    64,    48,    65,    65,    70,    63,
       5,    37,    38,    39,    40,    41,    42,    43,    44,    63,
      46,    18,    64,    64,    64,     8,   208,    56,   137,   173,
     271,   101,   191,   180,    18,   252,   100,   256,   132,   196,
     366,   197,   308,   364,   450,   421,   436,   383,   376
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const unsigned char yystos[] =
{
       0,     3,     4,    76,    77,    82,    48,    80,    80,     0,
      77,    61,    63,    83,    83,     5,    52,    53,    78,    84,
      85,   170,   171,    80,    80,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    18,    21,    63,    82,
     102,   103,   113,   116,   117,   118,   120,   130,   131,   134,
     135,   136,    64,    84,    16,    32,    33,    48,    54,    55,
      56,    57,    58,    59,    60,    81,    90,    92,    93,    94,
      95,    96,    97,    98,    32,   104,   104,   104,    81,   137,
      72,   108,   108,   108,   108,    72,   119,    72,   105,    50,
     138,    66,    84,    61,    61,    61,    11,    12,    13,    14,
      15,    16,   121,   122,   123,   124,   125,    61,    61,    61,
      79,    93,    97,    55,    59,    54,    55,    56,    57,    62,
      66,    89,    68,    68,    68,    33,    69,    71,    81,    81,
      81,    81,    65,    61,    27,   109,   110,    80,    91,    91,
      91,    91,    49,    80,    91,    34,    35,   106,   107,    91,
      61,    17,    96,    98,   128,   129,    64,   108,   108,   108,
     108,   119,   105,    68,    62,    49,    50,    86,    87,    88,
      98,    68,    72,   100,   101,    69,    69,    69,   137,    73,
      65,    89,    62,   114,   114,   114,   114,    80,    73,   114,
      73,    65,    63,    80,    74,   127,    80,    65,    67,    79,
      80,    80,    80,    80,    80,    80,    80,    48,    65,    67,
      80,    49,    81,    99,   101,   104,   104,   104,   109,    91,
     115,    61,    63,   132,   132,   132,   132,    73,   132,   106,
      98,   111,   112,    74,   126,    49,    50,   127,   129,   114,
     114,   114,   114,   114,    61,    63,    87,    70,    73,    70,
      70,    70,    65,    36,   133,   134,   139,   140,    80,   112,
      64,    98,   132,   132,   132,   132,   132,   112,    69,   115,
      72,   142,    64,   133,    61,    72,    64,    98,   148,   151,
     152,    20,    22,    23,    24,    25,    28,    29,    30,    31,
     143,   144,    32,    48,    80,    93,    94,   141,    79,    73,
      61,    80,    51,    69,   147,    65,    70,    73,    65,    69,
     153,    80,    61,    72,    74,    63,    69,    72,   147,    73,
     152,   143,    70,   152,    37,    38,    39,    40,    41,    42,
      43,    44,    46,    63,   155,   161,   153,    49,    50,    81,
     145,   147,    51,   146,   147,    70,    70,    69,   165,    72,
     165,    48,   166,   167,    63,    50,   160,    48,   163,   165,
      69,   156,   161,    19,   154,    64,    65,    70,    73,   147,
     147,    48,   147,    72,   153,   168,    65,    63,   161,   157,
     161,    63,   149,    65,    61,   147,    48,    64,   156,    74,
     155,   147,   146,   147,   147,    61,    73,    70,   164,   147,
     167,    64,   156,    64,   156,   147,   163,   164,   153,    49,
     147,   165,    63,   161,    73,   169,    64,    64,   150,    63,
     161,    70,    61,   147,   156,   153,    47,   158,   156,    45,
     162,   149,   147,    62,    64,    69,    64,    63,   161,   147,
     164,   147,    48,   159,   162,   156,    64,    63,   161,    65,
      65,    70,    64,   156,   147,   159,    64,   164,    63,   161,
     156,    64
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
#line 131 "xi-grammar.y"
    { yyval.modlist = yyvsp[0].modlist; modlist = yyvsp[0].modlist; }
    break;

  case 3:
#line 135 "xi-grammar.y"
    { 
		  yyval.modlist = 0; 
		}
    break;

  case 4:
#line 139 "xi-grammar.y"
    { yyval.modlist = new ModuleList(lineno, yyvsp[-1].module, yyvsp[0].modlist); }
    break;

  case 5:
#line 143 "xi-grammar.y"
    { yyval.intval = 0; }
    break;

  case 6:
#line 145 "xi-grammar.y"
    { yyval.intval = 1; }
    break;

  case 7:
#line 149 "xi-grammar.y"
    { yyval.intval = 0; }
    break;

  case 8:
#line 151 "xi-grammar.y"
    { yyval.intval = 1; }
    break;

  case 9:
#line 155 "xi-grammar.y"
    { yyval.strval = yyvsp[0].strval; }
    break;

  case 10:
#line 159 "xi-grammar.y"
    { yyval.strval = yyvsp[0].strval; }
    break;

  case 11:
#line 161 "xi-grammar.y"
    {
		  char *tmp = new char[strlen(yyvsp[-3].strval)+strlen(yyvsp[0].strval)+3];
		  sprintf(tmp,"%s::%s", yyvsp[-3].strval, yyvsp[0].strval);
		  yyval.strval = tmp;
		}
    break;

  case 12:
#line 169 "xi-grammar.y"
    { 
		    yyval.module = new Module(lineno, yyvsp[-1].strval, yyvsp[0].conslist); 
		}
    break;

  case 13:
#line 173 "xi-grammar.y"
    {  
		    yyval.module = new Module(lineno, yyvsp[-1].strval, yyvsp[0].conslist); 
		    yyval.module->setMain();
		}
    break;

  case 14:
#line 180 "xi-grammar.y"
    { yyval.conslist = 0; }
    break;

  case 15:
#line 182 "xi-grammar.y"
    { yyval.conslist = yyvsp[-2].conslist; }
    break;

  case 16:
#line 186 "xi-grammar.y"
    { yyval.conslist = 0; }
    break;

  case 17:
#line 188 "xi-grammar.y"
    { yyval.conslist = new ConstructList(lineno, yyvsp[-1].construct, yyvsp[0].conslist); }
    break;

  case 18:
#line 192 "xi-grammar.y"
    { if(yyvsp[-2].conslist) yyvsp[-2].conslist->setExtern(yyvsp[-4].intval); yyval.construct = yyvsp[-2].conslist; }
    break;

  case 19:
#line 194 "xi-grammar.y"
    { yyvsp[0].module->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].module; }
    break;

  case 20:
#line 196 "xi-grammar.y"
    { yyvsp[0].member->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].member; }
    break;

  case 21:
#line 198 "xi-grammar.y"
    { yyvsp[-1].message->setExtern(yyvsp[-2].intval); yyval.construct = yyvsp[-1].message; }
    break;

  case 22:
#line 200 "xi-grammar.y"
    { yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; }
    break;

  case 23:
#line 202 "xi-grammar.y"
    { yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; }
    break;

  case 24:
#line 204 "xi-grammar.y"
    { yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; }
    break;

  case 25:
#line 206 "xi-grammar.y"
    { yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; }
    break;

  case 26:
#line 208 "xi-grammar.y"
    { yyvsp[0].templat->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].templat; }
    break;

  case 27:
#line 210 "xi-grammar.y"
    { yyval.construct = NULL; }
    break;

  case 28:
#line 212 "xi-grammar.y"
    { yyval.construct = NULL; }
    break;

  case 29:
#line 216 "xi-grammar.y"
    { yyval.tparam = new TParamType(yyvsp[0].type); }
    break;

  case 30:
#line 218 "xi-grammar.y"
    { yyval.tparam = new TParamVal(yyvsp[0].strval); }
    break;

  case 31:
#line 220 "xi-grammar.y"
    { yyval.tparam = new TParamVal(yyvsp[0].strval); }
    break;

  case 32:
#line 224 "xi-grammar.y"
    { yyval.tparlist = new TParamList(yyvsp[0].tparam); }
    break;

  case 33:
#line 226 "xi-grammar.y"
    { yyval.tparlist = new TParamList(yyvsp[-2].tparam, yyvsp[0].tparlist); }
    break;

  case 34:
#line 230 "xi-grammar.y"
    { yyval.tparlist = 0; }
    break;

  case 35:
#line 232 "xi-grammar.y"
    { yyval.tparlist = yyvsp[0].tparlist; }
    break;

  case 36:
#line 236 "xi-grammar.y"
    { yyval.tparlist = 0; }
    break;

  case 37:
#line 238 "xi-grammar.y"
    { yyval.tparlist = yyvsp[-1].tparlist; }
    break;

  case 38:
#line 242 "xi-grammar.y"
    { yyval.type = new BuiltinType("int"); }
    break;

  case 39:
#line 244 "xi-grammar.y"
    { yyval.type = new BuiltinType("long"); }
    break;

  case 40:
#line 246 "xi-grammar.y"
    { yyval.type = new BuiltinType("short"); }
    break;

  case 41:
#line 248 "xi-grammar.y"
    { yyval.type = new BuiltinType("char"); }
    break;

  case 42:
#line 250 "xi-grammar.y"
    { yyval.type = new BuiltinType("unsigned int"); }
    break;

  case 43:
#line 252 "xi-grammar.y"
    { yyval.type = new BuiltinType("unsigned long"); }
    break;

  case 44:
#line 254 "xi-grammar.y"
    { yyval.type = new BuiltinType("unsigned short"); }
    break;

  case 45:
#line 256 "xi-grammar.y"
    { yyval.type = new BuiltinType("unsigned char"); }
    break;

  case 46:
#line 258 "xi-grammar.y"
    { yyval.type = new BuiltinType("long long"); }
    break;

  case 47:
#line 260 "xi-grammar.y"
    { yyval.type = new BuiltinType("float"); }
    break;

  case 48:
#line 262 "xi-grammar.y"
    { yyval.type = new BuiltinType("double"); }
    break;

  case 49:
#line 264 "xi-grammar.y"
    { yyval.type = new BuiltinType("long double"); }
    break;

  case 50:
#line 266 "xi-grammar.y"
    { yyval.type = new BuiltinType("void"); }
    break;

  case 51:
#line 269 "xi-grammar.y"
    { yyval.ntype = new NamedType(yyvsp[-1].strval,yyvsp[0].tparlist); }
    break;

  case 52:
#line 270 "xi-grammar.y"
    { yyval.ntype = new NamedType(yyvsp[-1].strval,yyvsp[0].tparlist); }
    break;

  case 53:
#line 273 "xi-grammar.y"
    { yyval.type = yyvsp[0].type; }
    break;

  case 54:
#line 275 "xi-grammar.y"
    { yyval.type = yyvsp[0].ntype; }
    break;

  case 55:
#line 279 "xi-grammar.y"
    { yyval.ptype = new PtrType(yyvsp[-1].type); }
    break;

  case 56:
#line 283 "xi-grammar.y"
    { yyvsp[-1].ptype->indirect(); yyval.ptype = yyvsp[-1].ptype; }
    break;

  case 57:
#line 285 "xi-grammar.y"
    { yyvsp[-1].ptype->indirect(); yyval.ptype = yyvsp[-1].ptype; }
    break;

  case 58:
#line 289 "xi-grammar.y"
    { yyval.ftype = new FuncType(yyvsp[-7].type, yyvsp[-4].strval, yyvsp[-1].plist); }
    break;

  case 59:
#line 293 "xi-grammar.y"
    { yyval.type = yyvsp[0].type; }
    break;

  case 60:
#line 295 "xi-grammar.y"
    { yyval.type = yyvsp[0].ptype; }
    break;

  case 61:
#line 297 "xi-grammar.y"
    { yyval.type = yyvsp[0].ptype; }
    break;

  case 62:
#line 299 "xi-grammar.y"
    { yyval.type = yyvsp[0].ftype; }
    break;

  case 63:
#line 302 "xi-grammar.y"
    { yyval.type = yyvsp[0].type; }
    break;

  case 64:
#line 304 "xi-grammar.y"
    { yyval.type = yyvsp[-1].type; }
    break;

  case 65:
#line 308 "xi-grammar.y"
    { yyval.type = new ReferenceType(yyvsp[-1].type); }
    break;

  case 66:
#line 310 "xi-grammar.y"
    { yyval.type = yyvsp[0].type; }
    break;

  case 67:
#line 314 "xi-grammar.y"
    { yyval.val = new Value(yyvsp[0].strval); }
    break;

  case 68:
#line 316 "xi-grammar.y"
    { yyval.val = new Value(yyvsp[0].strval); }
    break;

  case 69:
#line 320 "xi-grammar.y"
    { yyval.val = yyvsp[-1].val; }
    break;

  case 70:
#line 324 "xi-grammar.y"
    { yyval.vallist = 0; }
    break;

  case 71:
#line 326 "xi-grammar.y"
    { yyval.vallist = new ValueList(yyvsp[-1].val, yyvsp[0].vallist); }
    break;

  case 72:
#line 330 "xi-grammar.y"
    { yyval.readonly = new Readonly(lineno, yyvsp[-2].type, yyvsp[-1].strval, yyvsp[0].vallist); }
    break;

  case 73:
#line 334 "xi-grammar.y"
    { yyval.readonly = new Readonly(lineno, yyvsp[-2].type, yyvsp[0].strval, 0, 1); }
    break;

  case 74:
#line 338 "xi-grammar.y"
    { yyval.intval = 0;}
    break;

  case 75:
#line 340 "xi-grammar.y"
    { yyval.intval = 0;}
    break;

  case 76:
#line 344 "xi-grammar.y"
    { yyval.intval = 0; }
    break;

  case 77:
#line 346 "xi-grammar.y"
    { 
		  /*
		  printf("Warning: Message attributes are being phased out.\n");
		  printf("Warning: Please remove them from interface files.\n");
		  */
		  yyval.intval = yyvsp[-1].intval; 
		}
    break;

  case 78:
#line 356 "xi-grammar.y"
    { yyval.intval = yyvsp[0].intval; }
    break;

  case 79:
#line 358 "xi-grammar.y"
    { yyval.intval = yyvsp[-2].intval | yyvsp[0].intval; }
    break;

  case 80:
#line 362 "xi-grammar.y"
    { yyval.intval = 0; }
    break;

  case 81:
#line 364 "xi-grammar.y"
    { yyval.intval = 0; }
    break;

  case 82:
#line 368 "xi-grammar.y"
    { yyval.cattr = 0; }
    break;

  case 83:
#line 370 "xi-grammar.y"
    { yyval.cattr = yyvsp[-1].cattr; }
    break;

  case 84:
#line 374 "xi-grammar.y"
    { yyval.cattr = yyvsp[0].cattr; }
    break;

  case 85:
#line 376 "xi-grammar.y"
    { yyval.cattr = yyvsp[-2].cattr | yyvsp[0].cattr; }
    break;

  case 86:
#line 380 "xi-grammar.y"
    { yyval.cattr = Chare::CMIGRATABLE; }
    break;

  case 87:
#line 384 "xi-grammar.y"
    { yyval.mv = new MsgVar(yyvsp[-4].type, yyvsp[-3].strval); }
    break;

  case 88:
#line 388 "xi-grammar.y"
    { yyval.mvlist = new MsgVarList(yyvsp[0].mv); }
    break;

  case 89:
#line 390 "xi-grammar.y"
    { yyval.mvlist = new MsgVarList(yyvsp[-1].mv, yyvsp[0].mvlist); }
    break;

  case 90:
#line 394 "xi-grammar.y"
    { yyval.message = new Message(lineno, yyvsp[0].ntype); }
    break;

  case 91:
#line 396 "xi-grammar.y"
    { yyval.message = new Message(lineno, yyvsp[-3].ntype, yyvsp[-1].mvlist); }
    break;

  case 92:
#line 400 "xi-grammar.y"
    { yyval.typelist = 0; }
    break;

  case 93:
#line 402 "xi-grammar.y"
    { yyval.typelist = yyvsp[0].typelist; }
    break;

  case 94:
#line 406 "xi-grammar.y"
    { yyval.typelist = new TypeList(yyvsp[0].ntype); }
    break;

  case 95:
#line 408 "xi-grammar.y"
    { yyval.typelist = new TypeList(yyvsp[-2].ntype, yyvsp[0].typelist); }
    break;

  case 96:
#line 412 "xi-grammar.y"
    { yyval.chare = new Chare(lineno, yyvsp[-3].cattr, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 97:
#line 414 "xi-grammar.y"
    { yyval.chare = new MainChare(lineno, yyvsp[-3].cattr, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 98:
#line 418 "xi-grammar.y"
    { yyval.chare = new Group(lineno, yyvsp[-3].cattr, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 99:
#line 422 "xi-grammar.y"
    { yyval.chare = new NodeGroup(lineno, yyvsp[-3].cattr, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 100:
#line 426 "xi-grammar.y"
    {/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",yyvsp[-2].strval);
			yyval.ntype = new NamedType(buf); 
		}
    break;

  case 101:
#line 432 "xi-grammar.y"
    { yyval.ntype = new NamedType(yyvsp[-1].strval); }
    break;

  case 102:
#line 436 "xi-grammar.y"
    {  yyval.chare = new Array(lineno, 0, yyvsp[-3].ntype, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 103:
#line 440 "xi-grammar.y"
    { yyval.chare = new Chare(lineno, yyvsp[-3].cattr, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist);}
    break;

  case 104:
#line 442 "xi-grammar.y"
    { yyval.chare = new MainChare(lineno, yyvsp[-3].cattr, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 105:
#line 446 "xi-grammar.y"
    { yyval.chare = new Group(lineno, yyvsp[-3].cattr, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 106:
#line 450 "xi-grammar.y"
    { yyval.chare = new NodeGroup( lineno, yyvsp[-3].cattr, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 107:
#line 454 "xi-grammar.y"
    { yyval.chare = new Array( lineno, 0, yyvsp[-3].ntype, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 108:
#line 458 "xi-grammar.y"
    { yyval.message = new Message(lineno, new NamedType(yyvsp[-1].strval)); }
    break;

  case 109:
#line 460 "xi-grammar.y"
    { yyval.message = new Message(lineno, new NamedType(yyvsp[-4].strval), yyvsp[-2].mvlist); }
    break;

  case 110:
#line 464 "xi-grammar.y"
    { yyval.type = 0; }
    break;

  case 111:
#line 466 "xi-grammar.y"
    { yyval.type = yyvsp[0].type; }
    break;

  case 112:
#line 470 "xi-grammar.y"
    { yyval.strval = 0; }
    break;

  case 113:
#line 472 "xi-grammar.y"
    { yyval.strval = yyvsp[0].strval; }
    break;

  case 114:
#line 474 "xi-grammar.y"
    { yyval.strval = yyvsp[0].strval; }
    break;

  case 115:
#line 478 "xi-grammar.y"
    { yyval.tvar = new TType(new NamedType(yyvsp[-1].strval), yyvsp[0].type); }
    break;

  case 116:
#line 480 "xi-grammar.y"
    { yyval.tvar = new TFunc(yyvsp[-1].ftype, yyvsp[0].strval); }
    break;

  case 117:
#line 482 "xi-grammar.y"
    { yyval.tvar = new TName(yyvsp[-2].type, yyvsp[-1].strval, yyvsp[0].strval); }
    break;

  case 118:
#line 486 "xi-grammar.y"
    { yyval.tvarlist = new TVarList(yyvsp[0].tvar); }
    break;

  case 119:
#line 488 "xi-grammar.y"
    { yyval.tvarlist = new TVarList(yyvsp[-2].tvar, yyvsp[0].tvarlist); }
    break;

  case 120:
#line 492 "xi-grammar.y"
    { yyval.tvarlist = yyvsp[-1].tvarlist; }
    break;

  case 121:
#line 496 "xi-grammar.y"
    { yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); }
    break;

  case 122:
#line 498 "xi-grammar.y"
    { yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); }
    break;

  case 123:
#line 500 "xi-grammar.y"
    { yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); }
    break;

  case 124:
#line 502 "xi-grammar.y"
    { yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); }
    break;

  case 125:
#line 504 "xi-grammar.y"
    { yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].message); yyvsp[0].message->setTemplate(yyval.templat); }
    break;

  case 126:
#line 508 "xi-grammar.y"
    { yyval.mbrlist = 0; }
    break;

  case 127:
#line 510 "xi-grammar.y"
    { yyval.mbrlist = yyvsp[-2].mbrlist; }
    break;

  case 128:
#line 514 "xi-grammar.y"
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

  case 129:
#line 533 "xi-grammar.y"
    { yyval.mbrlist = new MemberList(yyvsp[-1].member, yyvsp[0].mbrlist); }
    break;

  case 130:
#line 537 "xi-grammar.y"
    { yyval.member = yyvsp[-1].readonly; }
    break;

  case 131:
#line 539 "xi-grammar.y"
    { yyval.member = yyvsp[-1].readonly; }
    break;

  case 133:
#line 542 "xi-grammar.y"
    { yyval.member = yyvsp[-1].member; }
    break;

  case 134:
#line 544 "xi-grammar.y"
    { yyval.member = yyvsp[-1].pupable; }
    break;

  case 135:
#line 546 "xi-grammar.y"
    { yyval.member = yyvsp[-1].includeFile; }
    break;

  case 136:
#line 550 "xi-grammar.y"
    { yyval.member = new InitCall(lineno, yyvsp[0].strval, 1); }
    break;

  case 137:
#line 552 "xi-grammar.y"
    { yyval.member = new InitCall(lineno, yyvsp[-3].strval, 1); }
    break;

  case 138:
#line 554 "xi-grammar.y"
    { printf("Warning: deprecated use of initcall. Use initnode or initproc instead.\n"); 
		  yyval.member = new InitCall(lineno, yyvsp[0].strval, 1); }
    break;

  case 139:
#line 557 "xi-grammar.y"
    { printf("Warning: deprecated use of initcall. Use initnode or initproc instead.\n");
		  yyval.member = new InitCall(lineno, yyvsp[-3].strval, 1); }
    break;

  case 140:
#line 562 "xi-grammar.y"
    { yyval.member = new InitCall(lineno, yyvsp[0].strval, 0); }
    break;

  case 141:
#line 564 "xi-grammar.y"
    { yyval.member = new InitCall(lineno, yyvsp[-3].strval, 0); }
    break;

  case 142:
#line 568 "xi-grammar.y"
    { yyval.pupable = new PUPableClass(lineno,yyvsp[0].strval,0); }
    break;

  case 143:
#line 570 "xi-grammar.y"
    { yyval.pupable = new PUPableClass(lineno,yyvsp[-2].strval,yyvsp[0].pupable); }
    break;

  case 144:
#line 574 "xi-grammar.y"
    { yyval.includeFile = new IncludeFile(lineno,yyvsp[0].strval,0); }
    break;

  case 145:
#line 578 "xi-grammar.y"
    { yyval.member = yyvsp[-1].entry; }
    break;

  case 146:
#line 580 "xi-grammar.y"
    { yyval.member = yyvsp[0].member; }
    break;

  case 147:
#line 584 "xi-grammar.y"
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

  case 148:
#line 595 "xi-grammar.y"
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

  case 149:
#line 608 "xi-grammar.y"
    { yyval.type = new BuiltinType("void"); }
    break;

  case 150:
#line 610 "xi-grammar.y"
    { yyval.type = yyvsp[0].ptype; }
    break;

  case 151:
#line 614 "xi-grammar.y"
    { yyval.intval = 0; }
    break;

  case 152:
#line 616 "xi-grammar.y"
    { yyval.intval = yyvsp[-1].intval; }
    break;

  case 153:
#line 620 "xi-grammar.y"
    { yyval.intval = yyvsp[0].intval; }
    break;

  case 154:
#line 622 "xi-grammar.y"
    { yyval.intval = yyvsp[-2].intval | yyvsp[0].intval; }
    break;

  case 155:
#line 626 "xi-grammar.y"
    { yyval.intval = STHREADED; }
    break;

  case 156:
#line 628 "xi-grammar.y"
    { yyval.intval = SSYNC; }
    break;

  case 157:
#line 630 "xi-grammar.y"
    { yyval.intval = SLOCKED; }
    break;

  case 158:
#line 632 "xi-grammar.y"
    { yyval.intval = SCREATEHERE; }
    break;

  case 159:
#line 634 "xi-grammar.y"
    { yyval.intval = SCREATEHOME; }
    break;

  case 160:
#line 636 "xi-grammar.y"
    { yyval.intval = SNOKEEP; }
    break;

  case 161:
#line 638 "xi-grammar.y"
    { yyval.intval = SNOTRACE; }
    break;

  case 162:
#line 640 "xi-grammar.y"
    { yyval.intval = SIMMEDIATE; }
    break;

  case 163:
#line 642 "xi-grammar.y"
    { yyval.intval = SSKIPSCHED; }
    break;

  case 164:
#line 646 "xi-grammar.y"
    { yyval.val = new Value(yyvsp[0].strval); }
    break;

  case 165:
#line 648 "xi-grammar.y"
    { yyval.val = new Value(yyvsp[0].strval); }
    break;

  case 166:
#line 650 "xi-grammar.y"
    { yyval.val = new Value(yyvsp[0].strval); }
    break;

  case 167:
#line 654 "xi-grammar.y"
    { yyval.strval = ""; }
    break;

  case 168:
#line 656 "xi-grammar.y"
    { yyval.strval = yyvsp[0].strval; }
    break;

  case 169:
#line 658 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen(yyvsp[-2].strval)+strlen(yyvsp[0].strval)+3];
			sprintf(tmp,"%s, %s", yyvsp[-2].strval, yyvsp[0].strval);
			yyval.strval = tmp;
		}
    break;

  case 170:
#line 666 "xi-grammar.y"
    { yyval.strval = ""; }
    break;

  case 171:
#line 668 "xi-grammar.y"
    { yyval.strval = yyvsp[0].strval; }
    break;

  case 172:
#line 670 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen(yyvsp[-4].strval)+strlen(yyvsp[-2].strval)+strlen(yyvsp[0].strval)+3];
			sprintf(tmp,"%s[%s]%s", yyvsp[-4].strval, yyvsp[-2].strval, yyvsp[0].strval);
			yyval.strval = tmp;
		}
    break;

  case 173:
#line 676 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen(yyvsp[-4].strval)+strlen(yyvsp[-2].strval)+strlen(yyvsp[0].strval)+3];
			sprintf(tmp,"%s{%s}%s", yyvsp[-4].strval, yyvsp[-2].strval, yyvsp[0].strval);
			yyval.strval = tmp;
		}
    break;

  case 174:
#line 682 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen(yyvsp[-4].strval)+strlen(yyvsp[-2].strval)+strlen(yyvsp[0].strval)+3];
			sprintf(tmp,"%s(%s)%s", yyvsp[-4].strval, yyvsp[-2].strval, yyvsp[0].strval);
			yyval.strval = tmp;
		}
    break;

  case 175:
#line 688 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen(yyvsp[-2].strval)+strlen(yyvsp[0].strval)+3];
			sprintf(tmp,"(%s)%s", yyvsp[-2].strval, yyvsp[0].strval);
			yyval.strval = tmp;
		}
    break;

  case 176:
#line 696 "xi-grammar.y"
    {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			yyval.pname = new Parameter(lineno, yyvsp[-2].type,yyvsp[-1].strval);
		}
    break;

  case 177:
#line 703 "xi-grammar.y"
    { 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			yyval.intval = 0;
		}
    break;

  case 178:
#line 711 "xi-grammar.y"
    { 
			in_braces=0;
			yyval.intval = 0;
		}
    break;

  case 179:
#line 718 "xi-grammar.y"
    { yyval.pname = new Parameter(lineno, yyvsp[0].type);}
    break;

  case 180:
#line 720 "xi-grammar.y"
    { yyval.pname = new Parameter(lineno, yyvsp[-1].type,yyvsp[0].strval);}
    break;

  case 181:
#line 722 "xi-grammar.y"
    { yyval.pname = new Parameter(lineno, yyvsp[-3].type,yyvsp[-2].strval,0,yyvsp[0].val);}
    break;

  case 182:
#line 724 "xi-grammar.y"
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			yyval.pname = new Parameter(lineno, yyvsp[-2].pname->getType(), yyvsp[-2].pname->getName() ,yyvsp[-1].strval);
		}
    break;

  case 183:
#line 731 "xi-grammar.y"
    { yyval.plist = new ParamList(yyvsp[0].pname); }
    break;

  case 184:
#line 733 "xi-grammar.y"
    { yyval.plist = new ParamList(yyvsp[-2].pname,yyvsp[0].plist); }
    break;

  case 185:
#line 737 "xi-grammar.y"
    { yyval.plist = yyvsp[-1].plist; }
    break;

  case 186:
#line 739 "xi-grammar.y"
    { yyval.plist = 0; }
    break;

  case 187:
#line 743 "xi-grammar.y"
    { yyval.val = 0; }
    break;

  case 188:
#line 745 "xi-grammar.y"
    { yyval.val = new Value(yyvsp[0].strval); }
    break;

  case 189:
#line 749 "xi-grammar.y"
    { yyval.sc = 0; }
    break;

  case 190:
#line 751 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SSDAGENTRY, yyvsp[0].sc); }
    break;

  case 191:
#line 753 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SSDAGENTRY, yyvsp[-1].sc); }
    break;

  case 192:
#line 757 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SSLIST, yyvsp[0].sc); }
    break;

  case 193:
#line 759 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SSLIST, yyvsp[-1].sc, yyvsp[0].sc);  }
    break;

  case 194:
#line 763 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SOLIST, yyvsp[0].sc); }
    break;

  case 195:
#line 765 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SOLIST, yyvsp[-1].sc, yyvsp[0].sc); }
    break;

  case 196:
#line 769 "xi-grammar.y"
    { yyval.sc = 0; }
    break;

  case 197:
#line 771 "xi-grammar.y"
    { yyval.sc = yyvsp[-1].sc; }
    break;

  case 198:
#line 775 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SPUBLISHES, new SdagConstruct(SIDENT, yyvsp[0].strval)); }
    break;

  case 199:
#line 777 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SPUBLISHES, new SdagConstruct(SIDENT, yyvsp[-2].strval), yyvsp[0].sc);  }
    break;

  case 200:
#line 781 "xi-grammar.y"
    { yyval.strval = yyvsp[0].strval; }
    break;

  case 201:
#line 783 "xi-grammar.y"
    { yyval.strval = 0; }
    break;

  case 202:
#line 787 "xi-grammar.y"
    { RemoveSdagComments(yyvsp[-2].strval);
		   yyval.sc = new SdagConstruct(SATOMIC, new XStr(yyvsp[-2].strval), yyvsp[0].sc, 0,0,0,0, 0 ); 
		   if (yyvsp[-4].strval) { yyvsp[-4].strval[strlen(yyvsp[-4].strval)-1]=0; yyval.sc->traceName = new XStr(yyvsp[-4].strval+1); }
		 }
    break;

  case 203:
#line 792 "xi-grammar.y"
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

  case 204:
#line 806 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SWHEN, 0, 0,0,0,0,0,yyvsp[-2].entrylist); }
    break;

  case 205:
#line 808 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SWHEN, 0, 0, 0,0,0, yyvsp[0].sc, yyvsp[-1].entrylist); }
    break;

  case 206:
#line 810 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SWHEN, 0, 0, 0,0,0, yyvsp[-1].sc, yyvsp[-3].entrylist); }
    break;

  case 207:
#line 812 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SOVERLAP,0, 0,0,0,0,yyvsp[-1].sc, 0); }
    break;

  case 208:
#line 814 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SFOR, 0, new SdagConstruct(SINT_EXPR, yyvsp[-8].strval), new SdagConstruct(SINT_EXPR, yyvsp[-6].strval),
		             new SdagConstruct(SINT_EXPR, yyvsp[-4].strval), 0, yyvsp[-1].sc, 0); }
    break;

  case 209:
#line 817 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SFOR, 0, new SdagConstruct(SINT_EXPR, yyvsp[-6].strval), new SdagConstruct(SINT_EXPR, yyvsp[-4].strval), 
		         new SdagConstruct(SINT_EXPR, yyvsp[-2].strval), 0, yyvsp[0].sc, 0); }
    break;

  case 210:
#line 820 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SFORALL, 0, new SdagConstruct(SIDENT, yyvsp[-9].strval), new SdagConstruct(SINT_EXPR, yyvsp[-6].strval), 
		             new SdagConstruct(SINT_EXPR, yyvsp[-4].strval), new SdagConstruct(SINT_EXPR, yyvsp[-2].strval), yyvsp[0].sc, 0); }
    break;

  case 211:
#line 823 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SFORALL, 0, new SdagConstruct(SIDENT, yyvsp[-11].strval), new SdagConstruct(SINT_EXPR, yyvsp[-8].strval), 
		                 new SdagConstruct(SINT_EXPR, yyvsp[-6].strval), new SdagConstruct(SINT_EXPR, yyvsp[-4].strval), yyvsp[-1].sc, 0); }
    break;

  case 212:
#line 826 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, yyvsp[-3].strval), yyvsp[0].sc,0,0,yyvsp[-1].sc,0); }
    break;

  case 213:
#line 828 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, yyvsp[-5].strval), yyvsp[0].sc,0,0,yyvsp[-2].sc,0); }
    break;

  case 214:
#line 830 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, yyvsp[-2].strval), 0,0,0,yyvsp[0].sc,0); }
    break;

  case 215:
#line 832 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SWHILE, 0, new SdagConstruct(SINT_EXPR, yyvsp[-4].strval), 0,0,0,yyvsp[-1].sc,0); }
    break;

  case 216:
#line 834 "xi-grammar.y"
    { yyval.sc = yyvsp[-1].sc; }
    break;

  case 217:
#line 838 "xi-grammar.y"
    { yyval.sc = 0; }
    break;

  case 218:
#line 840 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SELSE, 0,0,0,0,0, yyvsp[0].sc,0); }
    break;

  case 219:
#line 842 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SELSE, 0,0,0,0,0, yyvsp[-1].sc,0); }
    break;

  case 220:
#line 845 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SFORWARD, new SdagConstruct(SIDENT, yyvsp[0].strval)); }
    break;

  case 221:
#line 847 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SFORWARD, new SdagConstruct(SIDENT, yyvsp[-2].strval), yyvsp[0].sc);  }
    break;

  case 222:
#line 851 "xi-grammar.y"
    { in_int_expr = 0; yyval.intval = 0; }
    break;

  case 223:
#line 855 "xi-grammar.y"
    { in_int_expr = 1; yyval.intval = 0; }
    break;

  case 224:
#line 859 "xi-grammar.y"
    { 
		  if (yyvsp[0].plist != 0)
		     yyval.entry = new Entry(lineno, 0, 0, yyvsp[-1].strval, yyvsp[0].plist, 0, 0, 0, 0); 
		  else
		     yyval.entry = new Entry(lineno, 0, 0, yyvsp[-1].strval, 
				new ParamList(new Parameter(0, new BuiltinType("void"))), 0, 0, 0, 0); 
		}
    break;

  case 225:
#line 867 "xi-grammar.y"
    { if (yyvsp[0].plist != 0)
		    yyval.entry = new Entry(lineno, 0, 0, yyvsp[-4].strval, yyvsp[0].plist, 0, 0, yyvsp[-2].strval, 0); 
		  else
		    yyval.entry = new Entry(lineno, 0, 0, yyvsp[-4].strval, new ParamList(new Parameter(0, new BuiltinType("void"))), 0, 0, yyvsp[-2].strval, 0); 
		}
    break;

  case 226:
#line 875 "xi-grammar.y"
    { yyval.entrylist = new EntryList(yyvsp[0].entry); }
    break;

  case 227:
#line 877 "xi-grammar.y"
    { yyval.entrylist = new EntryList(yyvsp[-2].entry,yyvsp[0].entrylist); }
    break;

  case 228:
#line 881 "xi-grammar.y"
    { in_bracket=1; }
    break;

  case 229:
#line 884 "xi-grammar.y"
    { in_bracket=0; }
    break;

  case 230:
#line 888 "xi-grammar.y"
    { if (!macroDefined(yyvsp[0].strval, 1)) in_comment = 1; }
    break;

  case 231:
#line 892 "xi-grammar.y"
    { if (!macroDefined(yyvsp[0].strval, 0)) in_comment = 1; }
    break;


    }

/* Line 991 of yacc.c.  */
#line 2855 "y.tab.c"

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


#line 895 "xi-grammar.y"

void yyerror(const char *mesg)
{
  cout << cur_file<<":"<<lineno<<": Charmxi syntax error> " << mesg << endl;
  // return 0;
}

