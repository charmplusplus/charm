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
     VOID = 286,
     CONST = 287,
     PACKED = 288,
     VARSIZE = 289,
     ENTRY = 290,
     FOR = 291,
     FORALL = 292,
     WHILE = 293,
     WHEN = 294,
     OVERLAP = 295,
     ATOMIC = 296,
     FORWARD = 297,
     IF = 298,
     ELSE = 299,
     CONNECT = 300,
     PUBLISHES = 301,
     IDENT = 302,
     NUMBER = 303,
     LITERAL = 304,
     CPROGRAM = 305,
     HASHIF = 306,
     HASHIFDEF = 307,
     INT = 308,
     LONG = 309,
     SHORT = 310,
     CHAR = 311,
     FLOAT = 312,
     DOUBLE = 313,
     UNSIGNED = 314
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
#define VOID 286
#define CONST 287
#define PACKED 288
#define VARSIZE 289
#define ENTRY 290
#define FOR 291
#define FORALL 292
#define WHILE 293
#define WHEN 294
#define OVERLAP 295
#define ATOMIC 296
#define FORWARD 297
#define IF 298
#define ELSE 299
#define CONNECT 300
#define PUBLISHES 301
#define IDENT 302
#define NUMBER 303
#define LITERAL 304
#define CPROGRAM 305
#define HASHIF 306
#define HASHIFDEF 307
#define INT 308
#define LONG 309
#define SHORT 310
#define CHAR 311
#define FLOAT 312
#define DOUBLE 313
#define UNSIGNED 314




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
#line 243 "y.tab.c"
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 214 of yacc.c.  */
#line 255 "y.tab.c"

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
#define YYLAST   527

/* YYNTOKENS -- Number of terminals. */
#define YYNTOKENS  74
/* YYNNTS -- Number of nonterminals. */
#define YYNNTS  97
/* YYNRULES -- Number of rules. */
#define YYNRULES  230
/* YYNRULES -- Number of states. */
#define YYNSTATES  461

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   314

#define YYTRANSLATE(YYX) 						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const unsigned char yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    70,     2,
      68,    69,    67,     2,    64,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    61,    60,
      65,    73,    66,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    71,     2,    72,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    62,     2,    63,     2,     2,     2,     2,
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
      55,    56,    57,    58,    59
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
     509,   511,   513,   515,   517,   519,   521,   522,   524,   528,
     529,   531,   537,   543,   549,   554,   558,   560,   562,   564,
     567,   572,   576,   578,   582,   586,   589,   590,   594,   595,
     597,   601,   603,   606,   608,   611,   612,   617,   619,   623,
     625,   626,   633,   642,   647,   651,   657,   662,   674,   684,
     697,   712,   719,   728,   734,   742,   746,   747,   750,   755,
     757,   761,   763,   765,   768,   774,   776,   780,   782,   784,
     787
};

/* YYRHS -- A `-1'-separated list of the rules' RHS. */
static const short yyrhs[] =
{
      75,     0,    -1,    76,    -1,    -1,    81,    76,    -1,    -1,
       5,    -1,    -1,    60,    -1,    47,    -1,    47,    -1,    80,
      61,    61,    47,    -1,     3,    79,    82,    -1,     4,    79,
      82,    -1,    60,    -1,    62,    83,    63,    78,    -1,    -1,
      84,    83,    -1,    77,    62,    83,    63,    78,    -1,    77,
      81,    -1,    77,   133,    -1,    77,   112,    60,    -1,    77,
     115,    -1,    77,   116,    -1,    77,   117,    -1,    77,   119,
      -1,    77,   130,    -1,   169,    -1,   170,    -1,    97,    -1,
      48,    -1,    49,    -1,    85,    -1,    85,    64,    86,    -1,
      -1,    86,    -1,    -1,    65,    87,    66,    -1,    53,    -1,
      54,    -1,    55,    -1,    56,    -1,    59,    53,    -1,    59,
      54,    -1,    59,    55,    -1,    59,    56,    -1,    54,    54,
      -1,    57,    -1,    58,    -1,    54,    58,    -1,    31,    -1,
      79,    88,    -1,    80,    88,    -1,    89,    -1,    91,    -1,
      92,    67,    -1,    93,    67,    -1,    94,    67,    -1,    96,
      68,    67,    79,    69,    68,   151,    69,    -1,    92,    -1,
      93,    -1,    94,    -1,    95,    -1,    32,    96,    -1,    96,
      32,    -1,    96,    70,    -1,    96,    -1,    48,    -1,    80,
      -1,    71,    98,    72,    -1,    -1,    99,   100,    -1,     6,
      97,    80,   100,    -1,     6,    16,    92,    67,    79,    -1,
      -1,    31,    -1,    -1,    71,   105,    72,    -1,   106,    -1,
     106,    64,   105,    -1,    33,    -1,    34,    -1,    -1,    71,
     108,    72,    -1,   109,    -1,   109,    64,   108,    -1,    27,
      -1,    97,    79,    71,    72,    60,    -1,   110,    -1,   110,
     111,    -1,    16,   104,    90,    -1,    16,   104,    90,    62,
     111,    63,    -1,    -1,    61,   114,    -1,    90,    -1,    90,
      64,   114,    -1,    11,   107,    90,   113,   131,    -1,    12,
     107,    90,   113,   131,    -1,    13,   107,    90,   113,   131,
      -1,    14,   107,    90,   113,   131,    -1,    71,    48,    79,
      72,    -1,    71,    79,    72,    -1,    15,   118,    90,   113,
     131,    -1,    11,   107,    79,   113,   131,    -1,    12,   107,
      79,   113,   131,    -1,    13,   107,    79,   113,   131,    -1,
      14,   107,    79,   113,   131,    -1,    15,   118,    79,   113,
     131,    -1,    16,   104,    79,    60,    -1,    16,   104,    79,
      62,   111,    63,    60,    -1,    -1,    73,    97,    -1,    -1,
      73,    48,    -1,    73,    49,    -1,    17,    79,   125,    -1,
      95,   126,    -1,    97,    79,   126,    -1,   127,    -1,   127,
      64,   128,    -1,    21,    65,   128,    66,    -1,   129,   120,
      -1,   129,   121,    -1,   129,   122,    -1,   129,   123,    -1,
     129,   124,    -1,    60,    -1,    62,   132,    63,    78,    -1,
      -1,   138,   132,    -1,   101,    60,    -1,   102,    60,    -1,
     135,    60,    -1,   134,    60,    -1,    10,   136,    60,    -1,
      18,   137,    60,    -1,     8,   103,    80,    -1,     8,   103,
      80,    68,   103,    69,    -1,     7,   103,    80,    -1,     7,
     103,    80,    68,   103,    69,    -1,     9,   103,    80,    -1,
       9,   103,    80,    68,   103,    69,    -1,    80,    -1,    80,
      64,   136,    -1,    49,    -1,   139,    60,    -1,   133,    -1,
      35,   141,   140,    79,   152,   153,   154,    -1,    35,   141,
      79,   152,   154,    -1,    31,    -1,    93,    -1,    -1,    71,
     142,    72,    -1,   143,    -1,   143,    64,   142,    -1,    20,
      -1,    22,    -1,    23,    -1,    28,    -1,    29,    -1,    30,
      -1,    24,    -1,    25,    -1,    49,    -1,    48,    -1,    80,
      -1,    -1,    50,    -1,    50,    64,   145,    -1,    -1,    50,
      -1,    50,    71,   146,    72,   146,    -1,    50,    62,   146,
      63,   146,    -1,    50,    68,   145,    69,   146,    -1,    68,
     146,    69,   146,    -1,    97,    79,    71,    -1,    62,    -1,
      63,    -1,    97,    -1,    97,    79,    -1,    97,    79,    73,
     144,    -1,   147,   146,    72,    -1,   150,    -1,   150,    64,
     151,    -1,    68,   151,    69,    -1,    68,    69,    -1,    -1,
      19,    73,    48,    -1,    -1,   160,    -1,    62,   155,    63,
      -1,   160,    -1,   160,   155,    -1,   160,    -1,   160,   155,
      -1,    -1,    46,    68,   158,    69,    -1,    47,    -1,    47,
      64,   158,    -1,    49,    -1,    -1,    41,   159,   148,   146,
     149,   157,    -1,    45,    68,    47,   152,    69,   148,   146,
      63,    -1,    39,   166,    62,    63,    -1,    39,   166,   160,
      -1,    39,   166,    62,   155,    63,    -1,    40,    62,   156,
      63,    -1,    36,   164,   146,    60,   146,    60,   146,   163,
      62,   155,    63,    -1,    36,   164,   146,    60,   146,    60,
     146,   163,   160,    -1,    37,    71,    47,    72,   164,   146,
      61,   146,    64,   146,   163,   160,    -1,    37,    71,    47,
      72,   164,   146,    61,   146,    64,   146,   163,    62,   155,
      63,    -1,    43,   164,   146,   163,   160,   161,    -1,    43,
     164,   146,   163,    62,   155,    63,   161,    -1,    38,   164,
     146,   163,   160,    -1,    38,   164,   146,   163,    62,   155,
      63,    -1,    42,   162,    60,    -1,    -1,    44,   160,    -1,
      44,    62,   155,    63,    -1,    47,    -1,    47,    64,   162,
      -1,    69,    -1,    68,    -1,    47,   152,    -1,    47,   167,
     146,   168,   152,    -1,   165,    -1,   165,    64,   166,    -1,
      71,    -1,    72,    -1,    51,    79,    -1,    52,    79,    -1
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
     635,   637,   639,   643,   645,   647,   652,   653,   655,   664,
     665,   667,   673,   679,   685,   693,   700,   708,   715,   717,
     719,   721,   728,   730,   734,   736,   741,   742,   747,   748,
     750,   754,   756,   760,   762,   767,   768,   772,   774,   778,
     781,   784,   789,   803,   805,   807,   809,   811,   814,   817,
     820,   823,   825,   827,   829,   831,   836,   837,   839,   842,
     844,   848,   852,   856,   864,   872,   874,   878,   881,   885,
     889
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
  "CREATEHOME", "NOKEEP", "VOID", "CONST", "PACKED", "VARSIZE", "ENTRY", 
  "FOR", "FORALL", "WHILE", "WHEN", "OVERLAP", "ATOMIC", "FORWARD", "IF", 
  "ELSE", "CONNECT", "PUBLISHES", "IDENT", "NUMBER", "LITERAL", 
  "CPROGRAM", "HASHIF", "HASHIFDEF", "INT", "LONG", "SHORT", "CHAR", 
  "FLOAT", "DOUBLE", "UNSIGNED", "';'", "':'", "'{'", "'}'", "','", "'<'", 
  "'>'", "'*'", "'('", "')'", "'&'", "'['", "']'", "'='", "$accept", 
  "File", "ModuleEList", "OptExtern", "OptSemiColon", "Name", "QualName", 
  "Module", "ConstructEList", "ConstructList", "Construct", "TParam", 
  "TParamList", "TParamEList", "OptTParams", "BuiltinType", "NamedType", 
  "QualNamedType", "SimpleType", "OnePtrType", "PtrType", "FuncType", 
  "BaseType", "Type", "ArrayDim", "Dim", "DimList", "Readonly", 
  "ReadonlyMsg", "OptVoid", "MAttribs", "MAttribList", "MAttrib", 
  "CAttribs", "CAttribList", "CAttrib", "Var", "VarList", "Message", 
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
      59,    58,   123,   125,    44,    60,    62,    42,    40,    41,
      38,    91,    93,    61
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const unsigned char yyr1[] =
{
       0,    74,    75,    76,    76,    77,    77,    78,    78,    79,
      80,    80,    81,    81,    82,    82,    83,    83,    84,    84,
      84,    84,    84,    84,    84,    84,    84,    84,    84,    85,
      85,    85,    86,    86,    87,    87,    88,    88,    89,    89,
      89,    89,    89,    89,    89,    89,    89,    89,    89,    89,
      89,    90,    91,    92,    92,    93,    94,    94,    95,    96,
      96,    96,    96,    96,    96,    97,    97,    98,    98,    99,
     100,   100,   101,   102,   103,   103,   104,   104,   105,   105,
     106,   106,   107,   107,   108,   108,   109,   110,   111,   111,
     112,   112,   113,   113,   114,   114,   115,   115,   116,   117,
     118,   118,   119,   120,   120,   121,   122,   123,   124,   124,
     125,   125,   126,   126,   126,   127,   127,   127,   128,   128,
     129,   130,   130,   130,   130,   130,   131,   131,   132,   132,
     133,   133,   133,   133,   133,   133,   134,   134,   134,   134,
     135,   135,   136,   136,   137,   138,   138,   139,   139,   140,
     140,   141,   141,   142,   142,   143,   143,   143,   143,   143,
     143,   143,   143,   144,   144,   144,   145,   145,   145,   146,
     146,   146,   146,   146,   146,   147,   148,   149,   150,   150,
     150,   150,   151,   151,   152,   152,   153,   153,   154,   154,
     154,   155,   155,   156,   156,   157,   157,   158,   158,   159,
     159,   160,   160,   160,   160,   160,   160,   160,   160,   160,
     160,   160,   160,   160,   160,   160,   161,   161,   161,   162,
     162,   163,   164,   165,   165,   166,   166,   167,   168,   169,
     170
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
       1,     1,     1,     1,     1,     1,     0,     1,     3,     0,
       1,     5,     5,     5,     4,     3,     1,     1,     1,     2,
       4,     3,     1,     3,     3,     2,     0,     3,     0,     1,
       3,     1,     2,     1,     2,     0,     4,     1,     3,     1,
       0,     6,     8,     4,     3,     5,     4,    11,     9,    12,
      14,     6,     8,     5,     7,     3,     0,     2,     4,     1,
       3,     1,     1,     2,     5,     1,     3,     1,     1,     2,
       2
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const unsigned char yydefact[] =
{
       3,     0,     0,     0,     2,     3,     9,     0,     0,     1,
       4,    14,     5,    12,    13,     6,     0,     0,     0,     0,
       5,    27,    28,   229,   230,     0,    74,    74,    74,     0,
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
       0,     0,     7,   129,   145,     0,     0,   178,   169,   182,
       0,   155,   156,   157,   161,   162,   158,   159,   160,     0,
     153,    50,    10,     0,     0,   150,     0,   127,     0,   109,
     179,   170,   169,     0,     0,    58,   152,     0,     0,   188,
       0,    87,   175,     0,   169,   166,   169,     0,   181,   183,
     154,   185,     0,     0,     0,     0,     0,     0,   200,     0,
       0,     0,     0,   148,   189,   186,   164,   163,   165,   180,
       0,   167,     0,     0,   169,   184,   222,   169,     0,   169,
       0,   225,     0,     0,   199,     0,   219,     0,   169,     0,
       0,   191,     0,   188,   169,   166,   169,   169,   174,     0,
       0,     0,   227,   223,   169,     0,     0,   204,     0,   193,
     176,   169,     0,   215,     0,     0,   190,   192,     0,   147,
     172,   168,   173,   171,   169,     0,   221,     0,     0,   226,
     203,     0,   206,   194,     0,   220,     0,     0,   187,     0,
     169,     0,   213,   228,     0,   205,   177,   195,     0,   216,
       0,   169,     0,     0,   224,     0,   201,     0,     0,   211,
     169,     0,   169,   214,     0,   216,     0,   217,     0,     0,
       0,   197,     0,   212,     0,   202,     0,   208,   169,     0,
     196,   218,     0,     0,   198,   207,     0,     0,   209,     0,
     210
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
      50,    51,    79,    90,   256,   257,   296,   271,   289,   290,
     339,   342,   303,   278,   381,   417,   279,   280,   309,   363,
     333,   360,   378,   426,   442,   355,   361,   429,   357,   397,
     347,   351,   352,   374,   414,    21,    22
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -372
static const short yypact[] =
{
     114,   -28,   -28,    32,  -372,   114,  -372,    16,    16,  -372,
    -372,  -372,     6,  -372,  -372,  -372,   -28,   -28,   186,    -4,
       6,  -372,  -372,  -372,  -372,   187,    44,    44,    44,    34,
      17,    17,    17,    17,    59,    64,    90,    78,     6,  -372,
     103,   108,   115,  -372,  -372,  -372,  -372,   386,  -372,  -372,
     120,   126,   128,  -372,   304,  -372,   287,  -372,  -372,    43,
    -372,  -372,  -372,  -372,    56,    42,  -372,  -372,   124,   144,
     150,  -372,   -15,    34,  -372,    34,    34,    34,   -30,   156,
     199,   -28,   -28,   -28,   -28,    79,   -28,   113,   -28,  -372,
     167,   243,   165,  -372,  -372,  -372,    17,    17,    17,    17,
      59,    64,  -372,  -372,  -372,  -372,  -372,  -372,  -372,  -372,
    -372,   163,   -12,  -372,  -372,  -372,  -372,  -372,  -372,   170,
     274,  -372,  -372,  -372,  -372,  -372,   168,  -372,   -34,   -17,
     -14,     3,    34,  -372,  -372,   160,   172,   173,   176,   176,
     176,   176,   -28,   177,   176,  -372,  -372,   179,   175,   200,
    -372,   -28,     1,   -28,   204,   203,   128,   -28,   -28,   -28,
     -28,   -28,   -28,   -28,   224,  -372,  -372,   208,  -372,   207,
    -372,   -28,   158,   206,  -372,    44,    44,    44,  -372,  -372,
     199,  -372,   -28,    82,    82,    82,    82,   209,  -372,    82,
    -372,   113,   287,   205,   166,  -372,   211,   243,  -372,  -372,
     176,   176,   176,   176,   176,   104,  -372,  -372,   274,  -372,
     213,  -372,   219,   220,  -372,   222,   225,   234,  -372,   221,
    -372,  -372,   215,  -372,  -372,  -372,  -372,  -372,  -372,  -372,
     -28,   287,   230,   287,  -372,  -372,  -372,  -372,  -372,    82,
      82,    82,    82,    82,  -372,   287,  -372,   239,  -372,  -372,
    -372,  -372,   -28,   237,   246,  -372,   215,   260,   253,  -372,
    -372,  -372,  -372,  -372,  -372,  -372,  -372,   254,   287,  -372,
     149,   317,   128,  -372,  -372,   264,   265,   -28,   -40,   273,
     269,  -372,  -372,  -372,  -372,  -372,  -372,  -372,  -372,   267,
     283,   302,   282,   284,   124,  -372,   -28,  -372,   293,  -372,
      31,   -32,   -40,   294,   287,  -372,  -372,   149,   257,   341,
     284,  -372,  -372,    47,   -40,   305,   -40,   285,  -372,  -372,
    -372,  -372,   296,   288,   297,   288,   320,   307,   336,   348,
     288,   355,   435,  -372,  -372,   385,  -372,  -372,   219,  -372,
     362,   363,   357,   356,   -40,  -372,  -372,   -40,   382,   -40,
     -26,   366,   351,   435,  -372,   370,   387,   393,   -40,   408,
     391,   435,   383,   341,   -40,   305,   -40,   -40,  -372,   398,
     388,   410,  -372,  -372,   -40,   320,   216,  -372,   394,   435,
    -372,   -40,   348,  -372,   410,   284,  -372,  -372,   433,  -372,
    -372,  -372,  -372,  -372,   -40,   288,  -372,   369,   411,  -372,
    -372,   419,  -372,  -372,   421,  -372,   379,   416,  -372,   426,
     -40,   435,  -372,  -372,   284,  -372,  -372,   442,   435,   445,
     370,   -40,   429,   428,  -372,   424,  -372,   430,   397,  -372,
     -40,   410,   -40,  -372,   447,   445,   435,  -372,   432,   407,
     434,   436,   427,  -372,   438,  -372,   435,  -372,   -40,   447,
    -372,  -372,   439,   410,  -372,  -372,   425,   435,  -372,   440,
    -372
};

/* YYPGOTO[NTERM-NUM].  */
static const short yypgoto[] =
{
    -372,  -372,   492,  -372,  -149,    -1,   -27,   481,   496,     2,
    -372,  -372,   298,  -372,   368,  -372,    50,  -372,   -51,   236,
    -372,   -83,   452,   -21,  -372,  -372,   337,  -372,  -372,   -22,
     412,   318,  -372,    -7,   331,  -372,  -372,  -202,  -372,   -19,
     262,  -372,  -372,  -372,   415,  -372,  -372,  -372,  -372,  -372,
    -372,  -372,   316,  -372,   319,  -372,  -372,    24,   261,   500,
    -372,  -372,   389,  -372,  -372,  -372,  -372,  -372,   212,  -372,
    -372,   155,  -281,  -372,   102,  -372,  -372,  -180,  -298,  -372,
     161,  -338,  -372,  -372,    74,  -372,  -291,    91,   143,  -371,
    -316,  -372,   152,  -372,  -372,  -372,  -372
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -150
static const short yytable[] =
{
       7,     8,    78,   111,    73,    76,    77,   199,   152,   349,
     301,    15,   335,   406,   358,    23,    24,   125,   334,     6,
     125,   317,    53,   387,    82,    83,    84,   119,   302,   259,
     314,   119,     9,   340,   132,   343,   315,   172,   401,   316,
      92,   403,   308,   267,   119,   372,   128,   119,   129,   130,
     131,   175,   373,   126,   176,   127,   126,    16,    17,    52,
     439,   377,   379,   368,   119,  -112,   369,  -112,   371,   -16,
     153,   177,   334,   423,   194,    74,    11,   384,    12,   410,
     427,    57,   456,   390,   143,   392,   393,   407,    80,   157,
     158,   159,   160,   398,    57,   336,   337,   113,   444,   170,
     404,   114,   312,   119,   313,    78,   412,   120,   452,   115,
     116,   117,   118,   409,   152,   419,   424,     1,     2,   459,
     184,   185,   186,   297,   319,   189,     6,   142,   322,   422,
      85,   138,   139,   140,   141,    87,   144,   437,   149,    89,
     431,   187,   221,    91,   222,   212,   145,   146,   447,   438,
     193,   440,   196,   215,   216,   217,   200,   201,   202,   203,
     204,   205,   206,    93,   244,   458,   245,   453,    94,   281,
     210,   282,   283,   284,   285,    95,   153,   286,   287,   288,
     107,   239,   240,   241,   242,   243,   108,   170,   109,     1,
       2,   122,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    54,    36,    57,   211,    37,   224,   225,
     226,   123,   261,   228,   235,   236,   133,   124,    55,    56,
     294,    25,    26,    27,    28,    29,   134,   150,   156,   258,
     163,   164,   179,    36,    57,   171,   180,   182,   120,   191,
      58,    59,    60,    61,    62,    63,    64,   277,    38,   188,
     253,   190,   323,   324,   325,   326,   327,   328,   329,   330,
     151,   331,   192,   262,   263,   264,   265,   266,   197,   198,
     293,   207,   208,   209,    55,    56,   300,   172,   233,   400,
     119,   227,   247,   277,   194,   252,   338,   277,    55,    56,
      57,   249,   248,   260,   250,   310,    58,    59,    60,    61,
      62,    63,    64,   251,    57,    55,    56,   268,   270,   272,
      58,    59,    60,    61,    62,    63,    64,   276,    55,    56,
     274,    57,   165,   166,   275,   299,   321,    58,    59,    60,
      61,    62,    63,    64,    57,    55,   298,   304,   305,   306,
      58,    59,    60,    61,    62,    63,    64,   307,   291,  -149,
      -9,    57,   308,   311,   344,   341,   346,    58,    59,    60,
      61,    62,    63,    64,   292,   345,   318,   350,   348,   353,
      58,    59,    60,    61,    62,    63,    64,   323,   324,   325,
     326,   327,   328,   329,   330,   354,   331,   323,   324,   325,
     326,   327,   328,   329,   330,   356,   331,    96,    97,    98,
      99,   100,   101,   332,   362,   323,   324,   325,   326,   327,
     328,   329,   330,   376,   331,   323,   324,   325,   326,   327,
     328,   329,   330,   359,   331,   364,   366,   365,   367,   370,
     375,   411,   380,   323,   324,   325,   326,   327,   328,   329,
     330,   418,   331,   323,   324,   325,   326,   327,   328,   329,
     330,   382,   331,   383,   386,   385,   388,   402,   394,   436,
     395,   323,   324,   325,   326,   327,   328,   329,   330,   446,
     331,   323,   324,   325,   326,   327,   328,   329,   330,   396,
     331,   408,   415,   413,   416,   420,   421,   457,   425,   428,
     432,   433,   434,   435,   441,   445,   450,    10,   448,    39,
     449,   451,   455,   460,    14,   181,   246,   295,   112,   229,
     214,   218,   237,   162,   269,   161,   238,   273,    49,   320,
     391,   178,   430,   454,   389,   405,   443,   399
};

static const unsigned short yycheck[] =
{
       1,     2,    29,    54,    25,    27,    28,   156,    91,   325,
      50,     5,   310,   384,   330,    16,    17,    32,   309,    47,
      32,   302,    20,   361,    31,    32,    33,    61,    68,   231,
      62,    61,     0,   314,    64,   316,    68,    71,   376,    71,
      38,   379,    68,   245,    61,    71,    73,    61,    75,    76,
      77,    68,   350,    68,    68,    70,    68,    51,    52,    63,
     431,   352,   353,   344,    61,    64,   347,    66,   349,    63,
      91,    68,   363,   411,    73,    31,    60,   358,    62,   395,
     418,    47,   453,   364,    85,   366,   367,   385,    71,    96,
      97,    98,    99,   374,    47,    48,    49,    54,   436,   120,
     381,    58,    71,    61,    73,   132,   397,    65,   446,    53,
      54,    55,    56,   394,   197,   406,   414,     3,     4,   457,
     139,   140,   141,   272,   304,   144,    47,    48,   308,   410,
      71,    81,    82,    83,    84,    71,    86,   428,    88,    49,
     421,   142,    60,    65,    62,   172,    33,    34,   439,   430,
     151,   432,   153,   175,   176,   177,   157,   158,   159,   160,
     161,   162,   163,    60,    60,   456,    62,   448,    60,    20,
     171,    22,    23,    24,    25,    60,   197,    28,    29,    30,
      60,   200,   201,   202,   203,   204,    60,   208,    60,     3,
       4,    67,     6,     7,     8,     9,    10,    11,    12,    13,
      14,    15,    16,    16,    18,    47,    48,    21,   184,   185,
     186,    67,   233,   189,    48,    49,    60,    67,    31,    32,
     271,     6,     7,     8,     9,    10,    27,    60,    63,   230,
      67,    61,    72,    18,    47,    67,    64,    61,    65,    64,
      53,    54,    55,    56,    57,    58,    59,   268,    62,    72,
      35,    72,    36,    37,    38,    39,    40,    41,    42,    43,
      17,    45,    62,   239,   240,   241,   242,   243,    64,    66,
     271,    47,    64,    66,    31,    32,   277,    71,    73,    63,
      61,    72,    69,   304,    73,    64,   313,   308,    31,    32,
      47,    69,    72,    63,    69,   296,    53,    54,    55,    56,
      57,    58,    59,    69,    47,    31,    32,    68,    71,    63,
      53,    54,    55,    56,    57,    58,    59,    63,    31,    32,
      60,    47,    48,    49,    71,    60,    69,    53,    54,    55,
      56,    57,    58,    59,    47,    31,    72,    64,    69,    72,
      53,    54,    55,    56,    57,    58,    59,    64,    31,    47,
      68,    47,    68,    60,    69,    50,    68,    53,    54,    55,
      56,    57,    58,    59,    47,    69,    72,    47,    71,    62,
      53,    54,    55,    56,    57,    58,    59,    36,    37,    38,
      39,    40,    41,    42,    43,    49,    45,    36,    37,    38,
      39,    40,    41,    42,    43,    47,    45,    11,    12,    13,
      14,    15,    16,    62,    19,    36,    37,    38,    39,    40,
      41,    42,    43,    62,    45,    36,    37,    38,    39,    40,
      41,    42,    43,    68,    45,    63,    69,    64,    72,    47,
      64,    62,    62,    36,    37,    38,    39,    40,    41,    42,
      43,    62,    45,    36,    37,    38,    39,    40,    41,    42,
      43,    64,    45,    60,    63,    47,    73,    63,    60,    62,
      72,    36,    37,    38,    39,    40,    41,    42,    43,    62,
      45,    36,    37,    38,    39,    40,    41,    42,    43,    69,
      45,    48,    63,    72,    63,    69,    60,    62,    46,    44,
      61,    63,    68,    63,    47,    63,    69,     5,    64,    18,
      64,    63,    63,    63,     8,   137,   208,   271,    56,   191,
     173,   180,   196,   101,   252,   100,   197,   256,    18,   307,
     365,   132,   420,   449,   363,   382,   435,   375
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const unsigned char yystos[] =
{
       0,     3,     4,    75,    76,    81,    47,    79,    79,     0,
      76,    60,    62,    82,    82,     5,    51,    52,    77,    83,
      84,   169,   170,    79,    79,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    18,    21,    62,    81,
     101,   102,   112,   115,   116,   117,   119,   129,   130,   133,
     134,   135,    63,    83,    16,    31,    32,    47,    53,    54,
      55,    56,    57,    58,    59,    80,    89,    91,    92,    93,
      94,    95,    96,    97,    31,   103,   103,   103,    80,   136,
      71,   107,   107,   107,   107,    71,   118,    71,   104,    49,
     137,    65,    83,    60,    60,    60,    11,    12,    13,    14,
      15,    16,   120,   121,   122,   123,   124,    60,    60,    60,
      78,    92,    96,    54,    58,    53,    54,    55,    56,    61,
      65,    88,    67,    67,    67,    32,    68,    70,    80,    80,
      80,    80,    64,    60,    27,   108,   109,    79,    90,    90,
      90,    90,    48,    79,    90,    33,    34,   105,   106,    90,
      60,    17,    95,    97,   127,   128,    63,   107,   107,   107,
     107,   118,   104,    67,    61,    48,    49,    85,    86,    87,
      97,    67,    71,    99,   100,    68,    68,    68,   136,    72,
      64,    88,    61,   113,   113,   113,   113,    79,    72,   113,
      72,    64,    62,    79,    73,   126,    79,    64,    66,    78,
      79,    79,    79,    79,    79,    79,    79,    47,    64,    66,
      79,    48,    80,    98,   100,   103,   103,   103,   108,    90,
     114,    60,    62,   131,   131,   131,   131,    72,   131,   105,
      97,   110,   111,    73,   125,    48,    49,   126,   128,   113,
     113,   113,   113,   113,    60,    62,    86,    69,    72,    69,
      69,    69,    64,    35,   132,   133,   138,   139,    79,   111,
      63,    97,   131,   131,   131,   131,   131,   111,    68,   114,
      71,   141,    63,   132,    60,    71,    63,    97,   147,   150,
     151,    20,    22,    23,    24,    25,    28,    29,    30,   142,
     143,    31,    47,    79,    92,    93,   140,    78,    72,    60,
      79,    50,    68,   146,    64,    69,    72,    64,    68,   152,
      79,    60,    71,    73,    62,    68,    71,   146,    72,   151,
     142,    69,   151,    36,    37,    38,    39,    40,    41,    42,
      43,    45,    62,   154,   160,   152,    48,    49,    80,   144,
     146,    50,   145,   146,    69,    69,    68,   164,    71,   164,
      47,   165,   166,    62,    49,   159,    47,   162,   164,    68,
     155,   160,    19,   153,    63,    64,    69,    72,   146,   146,
      47,   146,    71,   152,   167,    64,    62,   160,   156,   160,
      62,   148,    64,    60,   146,    47,    63,   155,    73,   154,
     146,   145,   146,   146,    60,    72,    69,   163,   146,   166,
      63,   155,    63,   155,   146,   162,   163,   152,    48,   146,
     164,    62,   160,    72,   168,    63,    63,   149,    62,   160,
      69,    60,   146,   155,   152,    46,   157,   155,    44,   161,
     148,   146,    61,    63,    68,    63,    62,   160,   146,   163,
     146,    47,   158,   161,   155,    63,    62,   160,    64,    64,
      69,    63,   155,   146,   158,    63,   163,    62,   160,   155,
      63
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
    { yyval.intval = SIMMEDIATE; }
    break;

  case 162:
#line 640 "xi-grammar.y"
    { yyval.intval = SSKIPSCHED; }
    break;

  case 163:
#line 644 "xi-grammar.y"
    { yyval.val = new Value(yyvsp[0].strval); }
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
#line 652 "xi-grammar.y"
    { yyval.strval = ""; }
    break;

  case 167:
#line 654 "xi-grammar.y"
    { yyval.strval = yyvsp[0].strval; }
    break;

  case 168:
#line 656 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen(yyvsp[-2].strval)+strlen(yyvsp[0].strval)+3];
			sprintf(tmp,"%s, %s", yyvsp[-2].strval, yyvsp[0].strval);
			yyval.strval = tmp;
		}
    break;

  case 169:
#line 664 "xi-grammar.y"
    { yyval.strval = ""; }
    break;

  case 170:
#line 666 "xi-grammar.y"
    { yyval.strval = yyvsp[0].strval; }
    break;

  case 171:
#line 668 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen(yyvsp[-4].strval)+strlen(yyvsp[-2].strval)+strlen(yyvsp[0].strval)+3];
			sprintf(tmp,"%s[%s]%s", yyvsp[-4].strval, yyvsp[-2].strval, yyvsp[0].strval);
			yyval.strval = tmp;
		}
    break;

  case 172:
#line 674 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen(yyvsp[-4].strval)+strlen(yyvsp[-2].strval)+strlen(yyvsp[0].strval)+3];
			sprintf(tmp,"%s{%s}%s", yyvsp[-4].strval, yyvsp[-2].strval, yyvsp[0].strval);
			yyval.strval = tmp;
		}
    break;

  case 173:
#line 680 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen(yyvsp[-4].strval)+strlen(yyvsp[-2].strval)+strlen(yyvsp[0].strval)+3];
			sprintf(tmp,"%s(%s)%s", yyvsp[-4].strval, yyvsp[-2].strval, yyvsp[0].strval);
			yyval.strval = tmp;
		}
    break;

  case 174:
#line 686 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen(yyvsp[-2].strval)+strlen(yyvsp[0].strval)+3];
			sprintf(tmp,"(%s)%s", yyvsp[-2].strval, yyvsp[0].strval);
			yyval.strval = tmp;
		}
    break;

  case 175:
#line 694 "xi-grammar.y"
    {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			yyval.pname = new Parameter(lineno, yyvsp[-2].type,yyvsp[-1].strval);
		}
    break;

  case 176:
#line 701 "xi-grammar.y"
    { 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			yyval.intval = 0;
		}
    break;

  case 177:
#line 709 "xi-grammar.y"
    { 
			in_braces=0;
			yyval.intval = 0;
		}
    break;

  case 178:
#line 716 "xi-grammar.y"
    { yyval.pname = new Parameter(lineno, yyvsp[0].type);}
    break;

  case 179:
#line 718 "xi-grammar.y"
    { yyval.pname = new Parameter(lineno, yyvsp[-1].type,yyvsp[0].strval);}
    break;

  case 180:
#line 720 "xi-grammar.y"
    { yyval.pname = new Parameter(lineno, yyvsp[-3].type,yyvsp[-2].strval,0,yyvsp[0].val);}
    break;

  case 181:
#line 722 "xi-grammar.y"
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			yyval.pname = new Parameter(lineno, yyvsp[-2].pname->getType(), yyvsp[-2].pname->getName() ,yyvsp[-1].strval);
		}
    break;

  case 182:
#line 729 "xi-grammar.y"
    { yyval.plist = new ParamList(yyvsp[0].pname); }
    break;

  case 183:
#line 731 "xi-grammar.y"
    { yyval.plist = new ParamList(yyvsp[-2].pname,yyvsp[0].plist); }
    break;

  case 184:
#line 735 "xi-grammar.y"
    { yyval.plist = yyvsp[-1].plist; }
    break;

  case 185:
#line 737 "xi-grammar.y"
    { yyval.plist = 0; }
    break;

  case 186:
#line 741 "xi-grammar.y"
    { yyval.val = 0; }
    break;

  case 187:
#line 743 "xi-grammar.y"
    { yyval.val = new Value(yyvsp[0].strval); }
    break;

  case 188:
#line 747 "xi-grammar.y"
    { yyval.sc = 0; }
    break;

  case 189:
#line 749 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SSDAGENTRY, yyvsp[0].sc); }
    break;

  case 190:
#line 751 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SSDAGENTRY, yyvsp[-1].sc); }
    break;

  case 191:
#line 755 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SSLIST, yyvsp[0].sc); }
    break;

  case 192:
#line 757 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SSLIST, yyvsp[-1].sc, yyvsp[0].sc);  }
    break;

  case 193:
#line 761 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SOLIST, yyvsp[0].sc); }
    break;

  case 194:
#line 763 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SOLIST, yyvsp[-1].sc, yyvsp[0].sc); }
    break;

  case 195:
#line 767 "xi-grammar.y"
    { yyval.sc = 0; }
    break;

  case 196:
#line 769 "xi-grammar.y"
    { yyval.sc = yyvsp[-1].sc; }
    break;

  case 197:
#line 773 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SPUBLISHES, new SdagConstruct(SIDENT, yyvsp[0].strval)); }
    break;

  case 198:
#line 775 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SPUBLISHES, new SdagConstruct(SIDENT, yyvsp[-2].strval), yyvsp[0].sc);  }
    break;

  case 199:
#line 779 "xi-grammar.y"
    { yyval.strval = yyvsp[0].strval; }
    break;

  case 200:
#line 781 "xi-grammar.y"
    { yyval.strval = 0; }
    break;

  case 201:
#line 785 "xi-grammar.y"
    { RemoveSdagComments(yyvsp[-2].strval);
		   yyval.sc = new SdagConstruct(SATOMIC, new XStr(yyvsp[-2].strval), yyvsp[0].sc, 0,0,0,0, 0 ); 
		   if (yyvsp[-4].strval) { yyvsp[-4].strval[strlen(yyvsp[-4].strval)-1]=0; yyval.sc->traceName = new XStr(yyvsp[-4].strval+1); }
		 }
    break;

  case 202:
#line 790 "xi-grammar.y"
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

  case 203:
#line 804 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SWHEN, 0, 0,0,0,0,0,yyvsp[-2].entrylist); }
    break;

  case 204:
#line 806 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SWHEN, 0, 0, 0,0,0, yyvsp[0].sc, yyvsp[-1].entrylist); }
    break;

  case 205:
#line 808 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SWHEN, 0, 0, 0,0,0, yyvsp[-1].sc, yyvsp[-3].entrylist); }
    break;

  case 206:
#line 810 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SOVERLAP,0, 0,0,0,0,yyvsp[-1].sc, 0); }
    break;

  case 207:
#line 812 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SFOR, 0, new SdagConstruct(SINT_EXPR, yyvsp[-8].strval), new SdagConstruct(SINT_EXPR, yyvsp[-6].strval),
		             new SdagConstruct(SINT_EXPR, yyvsp[-4].strval), 0, yyvsp[-1].sc, 0); }
    break;

  case 208:
#line 815 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SFOR, 0, new SdagConstruct(SINT_EXPR, yyvsp[-6].strval), new SdagConstruct(SINT_EXPR, yyvsp[-4].strval), 
		         new SdagConstruct(SINT_EXPR, yyvsp[-2].strval), 0, yyvsp[0].sc, 0); }
    break;

  case 209:
#line 818 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SFORALL, 0, new SdagConstruct(SIDENT, yyvsp[-9].strval), new SdagConstruct(SINT_EXPR, yyvsp[-6].strval), 
		             new SdagConstruct(SINT_EXPR, yyvsp[-4].strval), new SdagConstruct(SINT_EXPR, yyvsp[-2].strval), yyvsp[0].sc, 0); }
    break;

  case 210:
#line 821 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SFORALL, 0, new SdagConstruct(SIDENT, yyvsp[-11].strval), new SdagConstruct(SINT_EXPR, yyvsp[-8].strval), 
		                 new SdagConstruct(SINT_EXPR, yyvsp[-6].strval), new SdagConstruct(SINT_EXPR, yyvsp[-4].strval), yyvsp[-1].sc, 0); }
    break;

  case 211:
#line 824 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, yyvsp[-3].strval), yyvsp[0].sc,0,0,yyvsp[-1].sc,0); }
    break;

  case 212:
#line 826 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, yyvsp[-5].strval), yyvsp[0].sc,0,0,yyvsp[-2].sc,0); }
    break;

  case 213:
#line 828 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, yyvsp[-2].strval), 0,0,0,yyvsp[0].sc,0); }
    break;

  case 214:
#line 830 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SWHILE, 0, new SdagConstruct(SINT_EXPR, yyvsp[-4].strval), 0,0,0,yyvsp[-1].sc,0); }
    break;

  case 215:
#line 832 "xi-grammar.y"
    { yyval.sc = yyvsp[-1].sc; }
    break;

  case 216:
#line 836 "xi-grammar.y"
    { yyval.sc = 0; }
    break;

  case 217:
#line 838 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SELSE, 0,0,0,0,0, yyvsp[0].sc,0); }
    break;

  case 218:
#line 840 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SELSE, 0,0,0,0,0, yyvsp[-1].sc,0); }
    break;

  case 219:
#line 843 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SFORWARD, new SdagConstruct(SIDENT, yyvsp[0].strval)); }
    break;

  case 220:
#line 845 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SFORWARD, new SdagConstruct(SIDENT, yyvsp[-2].strval), yyvsp[0].sc);  }
    break;

  case 221:
#line 849 "xi-grammar.y"
    { in_int_expr = 0; yyval.intval = 0; }
    break;

  case 222:
#line 853 "xi-grammar.y"
    { in_int_expr = 1; yyval.intval = 0; }
    break;

  case 223:
#line 857 "xi-grammar.y"
    { 
		  if (yyvsp[0].plist != 0)
		     yyval.entry = new Entry(lineno, 0, 0, yyvsp[-1].strval, yyvsp[0].plist, 0, 0, 0, 0); 
		  else
		     yyval.entry = new Entry(lineno, 0, 0, yyvsp[-1].strval, 
				new ParamList(new Parameter(0, new BuiltinType("void"))), 0, 0, 0, 0); 
		}
    break;

  case 224:
#line 865 "xi-grammar.y"
    { if (yyvsp[0].plist != 0)
		    yyval.entry = new Entry(lineno, 0, 0, yyvsp[-4].strval, yyvsp[0].plist, 0, 0, yyvsp[-2].strval, 0); 
		  else
		    yyval.entry = new Entry(lineno, 0, 0, yyvsp[-4].strval, new ParamList(new Parameter(0, new BuiltinType("void"))), 0, 0, yyvsp[-2].strval, 0); 
		}
    break;

  case 225:
#line 873 "xi-grammar.y"
    { yyval.entrylist = new EntryList(yyvsp[0].entry); }
    break;

  case 226:
#line 875 "xi-grammar.y"
    { yyval.entrylist = new EntryList(yyvsp[-2].entry,yyvsp[0].entrylist); }
    break;

  case 227:
#line 879 "xi-grammar.y"
    { in_bracket=1; }
    break;

  case 228:
#line 882 "xi-grammar.y"
    { in_bracket=0; }
    break;

  case 229:
#line 886 "xi-grammar.y"
    { if (!macroDefined(yyvsp[0].strval, 1)) in_comment = 1; }
    break;

  case 230:
#line 890 "xi-grammar.y"
    { if (!macroDefined(yyvsp[0].strval, 0)) in_comment = 1; }
    break;


    }

/* Line 991 of yacc.c.  */
#line 2847 "y.tab.c"

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


#line 893 "xi-grammar.y"

void yyerror(const char *mesg)
{
  cout << cur_file<<":"<<lineno<<": Charmxi syntax error> " << mesg << endl;
  // return 0;
}

