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
     VIRTUAL = 280,
     MIGRATABLE = 281,
     CREATEHERE = 282,
     CREATEHOME = 283,
     NOKEEP = 284,
     VOID = 285,
     CONST = 286,
     PACKED = 287,
     VARSIZE = 288,
     ENTRY = 289,
     FOR = 290,
     FORALL = 291,
     WHILE = 292,
     WHEN = 293,
     OVERLAP = 294,
     ATOMIC = 295,
     FORWARD = 296,
     IF = 297,
     ELSE = 298,
     CONNECT = 299,
     PUBLISHES = 300,
     IDENT = 301,
     NUMBER = 302,
     LITERAL = 303,
     CPROGRAM = 304,
     HASHIF = 305,
     HASHIFDEF = 306,
     INT = 307,
     LONG = 308,
     SHORT = 309,
     CHAR = 310,
     FLOAT = 311,
     DOUBLE = 312,
     UNSIGNED = 313
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
#define VIRTUAL 280
#define MIGRATABLE 281
#define CREATEHERE 282
#define CREATEHOME 283
#define NOKEEP 284
#define VOID 285
#define CONST 286
#define PACKED 287
#define VARSIZE 288
#define ENTRY 289
#define FOR 290
#define FORALL 291
#define WHILE 292
#define WHEN 293
#define OVERLAP 294
#define ATOMIC 295
#define FORWARD 296
#define IF 297
#define ELSE 298
#define CONNECT 299
#define PUBLISHES 300
#define IDENT 301
#define NUMBER 302
#define LITERAL 303
#define CPROGRAM 304
#define HASHIF 305
#define HASHIFDEF 306
#define INT 307
#define LONG 308
#define SHORT 309
#define CHAR 310
#define FLOAT 311
#define DOUBLE 312
#define UNSIGNED 313




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
#line 241 "y.tab.c"
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 214 of yacc.c.  */
#line 253 "y.tab.c"

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
#define YYLAST   526

/* YYNTOKENS -- Number of terminals. */
#define YYNTOKENS  73
/* YYNNTS -- Number of nonterminals. */
#define YYNNTS  97
/* YYNRULES -- Number of rules. */
#define YYNRULES  229
/* YYNRULES -- Number of states. */
#define YYNSTATES  460

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   313

#define YYTRANSLATE(YYX) 						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const unsigned char yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    69,     2,
      67,    68,    66,     2,    63,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    60,    59,
      64,    72,    65,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    70,     2,    71,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    61,     2,    62,     2,     2,     2,     2,
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
      55,    56,    57,    58
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
     509,   511,   513,   515,   517,   519,   520,   522,   526,   527,
     529,   535,   541,   547,   552,   556,   558,   560,   562,   565,
     570,   574,   576,   580,   584,   587,   588,   592,   593,   595,
     599,   601,   604,   606,   609,   610,   615,   617,   621,   623,
     624,   631,   640,   645,   649,   655,   660,   672,   682,   695,
     710,   717,   726,   732,   740,   744,   745,   748,   753,   755,
     759,   761,   763,   766,   772,   774,   778,   780,   782,   785
};

/* YYRHS -- A `-1'-separated list of the rules' RHS. */
static const short yyrhs[] =
{
      74,     0,    -1,    75,    -1,    -1,    80,    75,    -1,    -1,
       5,    -1,    -1,    59,    -1,    46,    -1,    46,    -1,    79,
      60,    60,    46,    -1,     3,    78,    81,    -1,     4,    78,
      81,    -1,    59,    -1,    61,    82,    62,    77,    -1,    -1,
      83,    82,    -1,    76,    61,    82,    62,    77,    -1,    76,
      80,    -1,    76,   132,    -1,    76,   111,    59,    -1,    76,
     114,    -1,    76,   115,    -1,    76,   116,    -1,    76,   118,
      -1,    76,   129,    -1,   168,    -1,   169,    -1,    96,    -1,
      47,    -1,    48,    -1,    84,    -1,    84,    63,    85,    -1,
      -1,    85,    -1,    -1,    64,    86,    65,    -1,    52,    -1,
      53,    -1,    54,    -1,    55,    -1,    58,    52,    -1,    58,
      53,    -1,    58,    54,    -1,    58,    55,    -1,    53,    53,
      -1,    56,    -1,    57,    -1,    53,    57,    -1,    30,    -1,
      78,    87,    -1,    79,    87,    -1,    88,    -1,    90,    -1,
      91,    66,    -1,    92,    66,    -1,    93,    66,    -1,    95,
      67,    66,    78,    68,    67,   150,    68,    -1,    91,    -1,
      92,    -1,    93,    -1,    94,    -1,    31,    95,    -1,    95,
      31,    -1,    95,    69,    -1,    95,    -1,    47,    -1,    79,
      -1,    70,    97,    71,    -1,    -1,    98,    99,    -1,     6,
      96,    79,    99,    -1,     6,    16,    91,    66,    78,    -1,
      -1,    30,    -1,    -1,    70,   104,    71,    -1,   105,    -1,
     105,    63,   104,    -1,    32,    -1,    33,    -1,    -1,    70,
     107,    71,    -1,   108,    -1,   108,    63,   107,    -1,    26,
      -1,    96,    78,    70,    71,    59,    -1,   109,    -1,   109,
     110,    -1,    16,   103,    89,    -1,    16,   103,    89,    61,
     110,    62,    -1,    -1,    60,   113,    -1,    89,    -1,    89,
      63,   113,    -1,    11,   106,    89,   112,   130,    -1,    12,
     106,    89,   112,   130,    -1,    13,   106,    89,   112,   130,
      -1,    14,   106,    89,   112,   130,    -1,    70,    47,    78,
      71,    -1,    70,    78,    71,    -1,    15,   117,    89,   112,
     130,    -1,    11,   106,    78,   112,   130,    -1,    12,   106,
      78,   112,   130,    -1,    13,   106,    78,   112,   130,    -1,
      14,   106,    78,   112,   130,    -1,    15,   117,    78,   112,
     130,    -1,    16,   103,    78,    59,    -1,    16,   103,    78,
      61,   110,    62,    59,    -1,    -1,    72,    96,    -1,    -1,
      72,    47,    -1,    72,    48,    -1,    17,    78,   124,    -1,
      94,   125,    -1,    96,    78,   125,    -1,   126,    -1,   126,
      63,   127,    -1,    21,    64,   127,    65,    -1,   128,   119,
      -1,   128,   120,    -1,   128,   121,    -1,   128,   122,    -1,
     128,   123,    -1,    59,    -1,    61,   131,    62,    77,    -1,
      -1,   137,   131,    -1,   100,    59,    -1,   101,    59,    -1,
     134,    59,    -1,   133,    59,    -1,    10,   135,    59,    -1,
      18,   136,    59,    -1,     8,   102,    79,    -1,     8,   102,
      79,    67,   102,    68,    -1,     7,   102,    79,    -1,     7,
     102,    79,    67,   102,    68,    -1,     9,   102,    79,    -1,
       9,   102,    79,    67,   102,    68,    -1,    79,    -1,    79,
      63,   135,    -1,    48,    -1,   138,    59,    -1,   132,    -1,
      34,   140,   139,    78,   151,   152,   153,    -1,    34,   140,
      78,   151,   153,    -1,    30,    -1,    92,    -1,    -1,    70,
     141,    71,    -1,   142,    -1,   142,    63,   141,    -1,    20,
      -1,    22,    -1,    23,    -1,    27,    -1,    28,    -1,    29,
      -1,    24,    -1,    48,    -1,    47,    -1,    79,    -1,    -1,
      49,    -1,    49,    63,   144,    -1,    -1,    49,    -1,    49,
      70,   145,    71,   145,    -1,    49,    61,   145,    62,   145,
      -1,    49,    67,   144,    68,   145,    -1,    67,   145,    68,
     145,    -1,    96,    78,    70,    -1,    61,    -1,    62,    -1,
      96,    -1,    96,    78,    -1,    96,    78,    72,   143,    -1,
     146,   145,    71,    -1,   149,    -1,   149,    63,   150,    -1,
      67,   150,    68,    -1,    67,    68,    -1,    -1,    19,    72,
      47,    -1,    -1,   159,    -1,    61,   154,    62,    -1,   159,
      -1,   159,   154,    -1,   159,    -1,   159,   154,    -1,    -1,
      45,    67,   157,    68,    -1,    46,    -1,    46,    63,   157,
      -1,    48,    -1,    -1,    40,   158,   147,   145,   148,   156,
      -1,    44,    67,    46,   151,    68,   147,   145,    62,    -1,
      38,   165,    61,    62,    -1,    38,   165,   159,    -1,    38,
     165,    61,   154,    62,    -1,    39,    61,   155,    62,    -1,
      35,   163,   145,    59,   145,    59,   145,   162,    61,   154,
      62,    -1,    35,   163,   145,    59,   145,    59,   145,   162,
     159,    -1,    36,    70,    46,    71,   163,   145,    60,   145,
      63,   145,   162,   159,    -1,    36,    70,    46,    71,   163,
     145,    60,   145,    63,   145,   162,    61,   154,    62,    -1,
      42,   163,   145,   162,   159,   160,    -1,    42,   163,   145,
     162,    61,   154,    62,   160,    -1,    37,   163,   145,   162,
     159,    -1,    37,   163,   145,   162,    61,   154,    62,    -1,
      41,   161,    59,    -1,    -1,    43,   159,    -1,    43,    61,
     154,    62,    -1,    46,    -1,    46,    63,   161,    -1,    68,
      -1,    67,    -1,    46,   151,    -1,    46,   166,   145,   167,
     151,    -1,   164,    -1,   164,    63,   165,    -1,    70,    -1,
      71,    -1,    50,    78,    -1,    51,    78,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const unsigned short yyrline[] =
{
       0,   129,   129,   134,   137,   142,   143,   148,   149,   153,
     157,   159,   167,   171,   178,   180,   185,   186,   190,   192,
     194,   196,   198,   200,   202,   204,   206,   208,   210,   214,
     216,   218,   222,   224,   229,   230,   235,   236,   240,   242,
     244,   246,   248,   250,   252,   254,   256,   258,   260,   262,
     264,   268,   269,   271,   273,   277,   281,   283,   287,   291,
     293,   295,   297,   300,   302,   306,   308,   312,   314,   318,
     323,   324,   328,   332,   337,   338,   343,   344,   354,   356,
     360,   362,   367,   368,   372,   374,   378,   382,   386,   388,
     392,   394,   399,   400,   404,   406,   410,   412,   416,   420,
     424,   430,   434,   438,   440,   444,   448,   452,   456,   458,
     463,   464,   469,   470,   472,   476,   478,   480,   484,   486,
     490,   494,   496,   498,   500,   502,   506,   508,   513,   531,
     535,   537,   539,   540,   542,   544,   548,   550,   552,   555,
     560,   562,   566,   568,   572,   576,   578,   582,   593,   606,
     608,   613,   614,   618,   620,   624,   626,   628,   630,   632,
     634,   636,   640,   642,   644,   649,   650,   652,   661,   662,
     664,   670,   676,   682,   690,   697,   705,   712,   714,   716,
     718,   725,   727,   731,   733,   738,   739,   744,   745,   747,
     751,   753,   757,   759,   764,   765,   769,   771,   775,   778,
     781,   786,   800,   802,   804,   806,   808,   811,   814,   817,
     820,   822,   824,   826,   828,   833,   834,   836,   839,   841,
     845,   849,   853,   861,   869,   871,   875,   878,   882,   886
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
  "IMMEDIATE", "VIRTUAL", "MIGRATABLE", "CREATEHERE", "CREATEHOME", 
  "NOKEEP", "VOID", "CONST", "PACKED", "VARSIZE", "ENTRY", "FOR", 
  "FORALL", "WHILE", "WHEN", "OVERLAP", "ATOMIC", "FORWARD", "IF", "ELSE", 
  "CONNECT", "PUBLISHES", "IDENT", "NUMBER", "LITERAL", "CPROGRAM", 
  "HASHIF", "HASHIFDEF", "INT", "LONG", "SHORT", "CHAR", "FLOAT", 
  "DOUBLE", "UNSIGNED", "';'", "':'", "'{'", "'}'", "','", "'<'", "'>'", 
  "'*'", "'('", "')'", "'&'", "'['", "']'", "'='", "$accept", "File", 
  "ModuleEList", "OptExtern", "OptSemiColon", "Name", "QualName", 
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
     305,   306,   307,   308,   309,   310,   311,   312,   313,    59,
      58,   123,   125,    44,    60,    62,    42,    40,    41,    38,
      91,    93,    61
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const unsigned char yyr1[] =
{
       0,    73,    74,    75,    75,    76,    76,    77,    77,    78,
      79,    79,    80,    80,    81,    81,    82,    82,    83,    83,
      83,    83,    83,    83,    83,    83,    83,    83,    83,    84,
      84,    84,    85,    85,    86,    86,    87,    87,    88,    88,
      88,    88,    88,    88,    88,    88,    88,    88,    88,    88,
      88,    89,    90,    91,    91,    92,    93,    93,    94,    95,
      95,    95,    95,    95,    95,    96,    96,    97,    97,    98,
      99,    99,   100,   101,   102,   102,   103,   103,   104,   104,
     105,   105,   106,   106,   107,   107,   108,   109,   110,   110,
     111,   111,   112,   112,   113,   113,   114,   114,   115,   116,
     117,   117,   118,   119,   119,   120,   121,   122,   123,   123,
     124,   124,   125,   125,   125,   126,   126,   126,   127,   127,
     128,   129,   129,   129,   129,   129,   130,   130,   131,   131,
     132,   132,   132,   132,   132,   132,   133,   133,   133,   133,
     134,   134,   135,   135,   136,   137,   137,   138,   138,   139,
     139,   140,   140,   141,   141,   142,   142,   142,   142,   142,
     142,   142,   143,   143,   143,   144,   144,   144,   145,   145,
     145,   145,   145,   145,   146,   147,   148,   149,   149,   149,
     149,   150,   150,   151,   151,   152,   152,   153,   153,   153,
     154,   154,   155,   155,   156,   156,   157,   157,   158,   158,
     159,   159,   159,   159,   159,   159,   159,   159,   159,   159,
     159,   159,   159,   159,   159,   160,   160,   160,   161,   161,
     162,   163,   164,   164,   165,   165,   166,   167,   168,   169
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
       1,     1,     1,     1,     1,     0,     1,     3,     0,     1,
       5,     5,     5,     4,     3,     1,     1,     1,     2,     4,
       3,     1,     3,     3,     2,     0,     3,     0,     1,     3,
       1,     2,     1,     2,     0,     4,     1,     3,     1,     0,
       6,     8,     4,     3,     5,     4,    11,     9,    12,    14,
       6,     8,     5,     7,     3,     0,     2,     4,     1,     3,
       1,     1,     2,     5,     1,     3,     1,     1,     2,     2
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const unsigned char yydefact[] =
{
       3,     0,     0,     0,     2,     3,     9,     0,     0,     1,
       4,    14,     5,    12,    13,     6,     0,     0,     0,     0,
       5,    27,    28,   228,   229,     0,    74,    74,    74,     0,
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
       0,     0,     7,   129,   145,     0,     0,   177,   168,   181,
       0,   155,   156,   157,   161,   158,   159,   160,     0,   153,
      50,    10,     0,     0,   150,     0,   127,     0,   109,   178,
     169,   168,     0,     0,    58,   152,     0,     0,   187,     0,
      87,   174,     0,   168,   165,   168,     0,   180,   182,   154,
     184,     0,     0,     0,     0,     0,     0,   199,     0,     0,
       0,     0,   148,   188,   185,   163,   162,   164,   179,     0,
     166,     0,     0,   168,   183,   221,   168,     0,   168,     0,
     224,     0,     0,   198,     0,   218,     0,   168,     0,     0,
     190,     0,   187,   168,   165,   168,   168,   173,     0,     0,
       0,   226,   222,   168,     0,     0,   203,     0,   192,   175,
     168,     0,   214,     0,     0,   189,   191,     0,   147,   171,
     167,   172,   170,   168,     0,   220,     0,     0,   225,   202,
       0,   205,   193,     0,   219,     0,     0,   186,     0,   168,
       0,   212,   227,     0,   204,   176,   194,     0,   215,     0,
     168,     0,     0,   223,     0,   200,     0,     0,   210,   168,
       0,   168,   213,     0,   215,     0,   216,     0,     0,     0,
     196,     0,   211,     0,   201,     0,   207,   168,     0,   195,
     217,     0,     0,   197,   206,     0,     0,   208,     0,   209
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
      50,    51,    79,    90,   256,   257,   295,   271,   288,   289,
     338,   341,   302,   278,   380,   416,   279,   280,   308,   362,
     332,   359,   377,   425,   441,   354,   360,   428,   356,   396,
     346,   350,   351,   373,   413,    21,    22
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -371
static const short yypact[] =
{
      99,   -18,   -18,    42,  -371,    99,  -371,    83,    83,  -371,
    -371,  -371,     5,  -371,  -371,  -371,   -18,   -18,   192,    -3,
       5,  -371,  -371,  -371,  -371,   202,    34,    34,    34,    23,
      11,    11,    11,    11,    18,    60,    27,    74,     5,  -371,
      88,   109,   127,  -371,  -371,  -371,  -371,   364,  -371,  -371,
     129,   135,   138,  -371,   279,  -371,   184,  -371,  -371,     0,
    -371,  -371,  -371,  -371,    57,    -6,  -371,  -371,    97,   143,
     145,  -371,     9,    23,  -371,    23,    23,    23,    44,   160,
     205,   -18,   -18,   -18,   -18,    71,   -18,   195,   -18,  -371,
     162,   235,   173,  -371,  -371,  -371,    11,    11,    11,    11,
      18,    60,  -371,  -371,  -371,  -371,  -371,  -371,  -371,  -371,
    -371,   179,   -20,  -371,  -371,  -371,  -371,  -371,  -371,   186,
     266,  -371,  -371,  -371,  -371,  -371,   183,  -371,   -40,   -38,
     -28,   -16,    23,  -371,  -371,   180,   198,   199,   204,   204,
     204,   204,   -18,   191,   204,  -371,  -371,   197,   206,   210,
    -371,   -18,   -29,   -18,   209,   208,   138,   -18,   -18,   -18,
     -18,   -18,   -18,   -18,   221,  -371,  -371,   211,  -371,   212,
    -371,   -18,   170,   213,  -371,    34,    34,    34,  -371,  -371,
     205,  -371,   -18,   105,   105,   105,   105,   207,  -371,   105,
    -371,   195,   184,   203,   196,  -371,   226,   235,  -371,  -371,
     204,   204,   204,   204,   204,   110,  -371,  -371,   266,  -371,
     231,  -371,   224,   229,  -371,   240,   242,   243,  -371,   252,
    -371,  -371,   216,  -371,  -371,  -371,  -371,  -371,  -371,  -371,
     -18,   184,   254,   184,  -371,  -371,  -371,  -371,  -371,   105,
     105,   105,   105,   105,  -371,   184,  -371,   260,  -371,  -371,
    -371,  -371,   -18,   258,   267,  -371,   216,   271,   268,  -371,
    -371,  -371,  -371,  -371,  -371,  -371,  -371,   277,   184,  -371,
     104,   296,   138,  -371,  -371,   269,   282,   -18,   -30,   280,
     276,  -371,  -371,  -371,  -371,  -371,  -371,  -371,   274,   283,
     301,   306,   314,    97,  -371,   -18,  -371,   304,  -371,   108,
       4,   -30,   332,   184,  -371,  -371,   104,   249,   330,   314,
    -371,  -371,    48,   -30,   352,   -30,   336,  -371,  -371,  -371,
    -371,   337,   339,   338,   339,   361,   349,   381,   385,   339,
     365,   432,  -371,  -371,   415,  -371,  -371,   224,  -371,   371,
     372,   368,   367,   -30,  -371,  -371,   -30,   411,   -30,    76,
     396,   348,   432,  -371,   399,   398,   403,   -30,   417,   402,
     432,   394,   330,   -30,   352,   -30,   -30,  -371,   418,   407,
     412,  -371,  -371,   -30,   361,   320,  -371,   419,   432,  -371,
     -30,   385,  -371,   412,   314,  -371,  -371,   435,  -371,  -371,
    -371,  -371,  -371,   -30,   339,  -371,   358,   408,  -371,  -371,
     421,  -371,  -371,   422,  -371,   376,   420,  -371,   426,   -30,
     432,  -371,  -371,   314,  -371,  -371,   441,   432,   444,   399,
     -30,   429,   428,  -371,   424,  -371,   430,   386,  -371,   -30,
     412,   -30,  -371,   447,   444,   432,  -371,   433,   404,   431,
     434,   436,  -371,   437,  -371,   432,  -371,   -30,   447,  -371,
    -371,   438,   412,  -371,  -371,   414,   432,  -371,   439,  -371
};

/* YYPGOTO[NTERM-NUM].  */
static const short yypgoto[] =
{
    -371,  -371,   491,  -371,  -149,    -1,   -27,   480,   494,     7,
    -371,  -371,   295,  -371,   369,  -371,    91,  -371,   -51,   234,
    -371,   -83,   451,   -21,  -371,  -371,   335,  -371,  -371,   -22,
     409,   318,  -371,    -7,   331,  -371,  -371,  -214,  -371,   -19,
     261,  -371,  -371,  -371,   416,  -371,  -371,  -371,  -371,  -371,
    -371,  -371,   316,  -371,   317,  -371,  -371,   -50,   259,   499,
    -371,  -371,   387,  -371,  -371,  -371,  -371,  -371,   214,  -371,
    -371,   154,  -280,  -371,   102,  -371,  -371,  -206,  -297,  -371,
     161,  -337,  -371,  -371,    77,  -371,  -290,    90,   141,  -370,
    -315,  -371,   152,  -371,  -371,  -371,  -371
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -150
static const short yytable[] =
{
       7,     8,    78,   111,    73,    76,    77,   199,   152,   348,
      15,   125,   334,   405,   357,    23,    24,   259,   333,   300,
     119,   316,   119,   386,    82,    83,    84,    53,     6,   175,
     172,   267,   119,   339,  -112,   342,  -112,   301,   400,   176,
     125,   402,     9,   194,   119,    92,   128,   126,   129,   130,
     131,   177,   372,   113,   119,    16,    17,   114,   120,    52,
     438,   376,   378,   367,    74,   313,   368,   -16,   370,    57,
     153,   314,   333,   422,   315,    89,   126,   383,   127,   409,
     426,    80,   455,   389,   143,   391,   392,   406,    85,   157,
     158,   159,   160,   397,    57,   335,   336,   318,   443,   170,
     403,   321,     1,     2,   119,    78,   411,   132,   451,   115,
     116,   117,   118,   408,   152,   418,   423,     6,   142,   458,
     184,   185,   186,   296,   281,   189,   282,   283,   284,   421,
      87,   285,   286,   287,   224,   225,   226,   436,    91,   228,
     430,   187,    11,   307,    12,   212,   371,    93,   446,   437,
     193,   439,   196,   215,   216,   217,   200,   201,   202,   203,
     204,   205,   206,   122,   221,   457,   222,   452,    94,   244,
     210,   245,   138,   139,   140,   141,   153,   144,   311,   149,
     312,   239,   240,   241,   242,   243,    95,   170,   107,   262,
     263,   264,   265,   266,   108,     1,     2,   109,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,   123,
      36,   124,   261,    37,    55,    56,    57,   211,    54,   133,
     293,   150,    25,    26,    27,    28,    29,   145,   146,   258,
      57,   134,    55,    56,    36,   156,    58,    59,    60,    61,
      62,    63,    64,   235,   236,   163,   164,   277,    57,   171,
     253,   179,   151,    38,    58,    59,    60,    61,    62,    63,
      64,   180,   188,   120,   182,    55,    56,   207,   190,   191,
     292,   192,   197,   198,   208,   233,   299,   209,   227,    55,
      56,    57,   277,   172,   119,   337,   277,    58,    59,    60,
      61,    62,    63,    64,   309,    57,    55,    56,   194,   247,
     248,    58,    59,    60,    61,    62,    63,    64,   249,    55,
     250,   251,    57,   165,   166,   252,   260,   320,    58,    59,
      60,    61,    62,    63,    64,    57,   290,   268,   270,   272,
     274,    58,    59,    60,    61,    62,    63,    64,   275,   276,
     297,   298,   291,   303,   304,   305,   306,  -149,    58,    59,
      60,    61,    62,    63,    64,   322,   323,   324,   325,   326,
     327,   328,   329,   310,   330,   322,   323,   324,   325,   326,
     327,   328,   329,    -9,   330,    96,    97,    98,    99,   100,
     101,   307,   399,   322,   323,   324,   325,   326,   327,   328,
     329,   331,   330,   322,   323,   324,   325,   326,   327,   328,
     329,   340,   330,   317,   343,   344,   345,   349,   347,   375,
     352,   322,   323,   324,   325,   326,   327,   328,   329,   410,
     330,   322,   323,   324,   325,   326,   327,   328,   329,   353,
     330,   355,   358,   363,   361,   364,   365,   417,   366,   322,
     323,   324,   325,   326,   327,   328,   329,   435,   330,   322,
     323,   324,   325,   326,   327,   328,   329,   369,   330,   374,
     379,   381,   382,   384,   385,   445,   387,   322,   323,   324,
     325,   326,   327,   328,   329,   456,   330,   393,   394,   412,
     395,   401,   407,   414,   415,   420,   424,   427,   419,   431,
     432,   433,   434,   440,   447,   444,    10,   448,    39,   450,
     454,   459,    14,   246,   449,   294,   181,   112,   214,   229,
     162,   218,   237,   269,   238,   273,   161,    49,   390,   178,
     319,   429,   404,   388,   442,   453,   398
};

static const unsigned short yycheck[] =
{
       1,     2,    29,    54,    25,    27,    28,   156,    91,   324,
       5,    31,   309,   383,   329,    16,    17,   231,   308,    49,
      60,   301,    60,   360,    31,    32,    33,    20,    46,    67,
      70,   245,    60,   313,    63,   315,    65,    67,   375,    67,
      31,   378,     0,    72,    60,    38,    73,    67,    75,    76,
      77,    67,   349,    53,    60,    50,    51,    57,    64,    62,
     430,   351,   352,   343,    30,    61,   346,    62,   348,    46,
      91,    67,   362,   410,    70,    48,    67,   357,    69,   394,
     417,    70,   452,   363,    85,   365,   366,   384,    70,    96,
      97,    98,    99,   373,    46,    47,    48,   303,   435,   120,
     380,   307,     3,     4,    60,   132,   396,    63,   445,    52,
      53,    54,    55,   393,   197,   405,   413,    46,    47,   456,
     139,   140,   141,   272,    20,   144,    22,    23,    24,   409,
      70,    27,    28,    29,   184,   185,   186,   427,    64,   189,
     420,   142,    59,    67,    61,   172,    70,    59,   438,   429,
     151,   431,   153,   175,   176,   177,   157,   158,   159,   160,
     161,   162,   163,    66,    59,   455,    61,   447,    59,    59,
     171,    61,    81,    82,    83,    84,   197,    86,    70,    88,
      72,   200,   201,   202,   203,   204,    59,   208,    59,   239,
     240,   241,   242,   243,    59,     3,     4,    59,     6,     7,
       8,     9,    10,    11,    12,    13,    14,    15,    16,    66,
      18,    66,   233,    21,    30,    31,    46,    47,    16,    59,
     271,    59,     6,     7,     8,     9,    10,    32,    33,   230,
      46,    26,    30,    31,    18,    62,    52,    53,    54,    55,
      56,    57,    58,    47,    48,    66,    60,   268,    46,    66,
      34,    71,    17,    61,    52,    53,    54,    55,    56,    57,
      58,    63,    71,    64,    60,    30,    31,    46,    71,    63,
     271,    61,    63,    65,    63,    72,   277,    65,    71,    30,
      31,    46,   303,    70,    60,   312,   307,    52,    53,    54,
      55,    56,    57,    58,   295,    46,    30,    31,    72,    68,
      71,    52,    53,    54,    55,    56,    57,    58,    68,    30,
      68,    68,    46,    47,    48,    63,    62,    68,    52,    53,
      54,    55,    56,    57,    58,    46,    30,    67,    70,    62,
      59,    52,    53,    54,    55,    56,    57,    58,    70,    62,
      71,    59,    46,    63,    68,    71,    63,    46,    52,    53,
      54,    55,    56,    57,    58,    35,    36,    37,    38,    39,
      40,    41,    42,    59,    44,    35,    36,    37,    38,    39,
      40,    41,    42,    67,    44,    11,    12,    13,    14,    15,
      16,    67,    62,    35,    36,    37,    38,    39,    40,    41,
      42,    61,    44,    35,    36,    37,    38,    39,    40,    41,
      42,    49,    44,    71,    68,    68,    67,    46,    70,    61,
      61,    35,    36,    37,    38,    39,    40,    41,    42,    61,
      44,    35,    36,    37,    38,    39,    40,    41,    42,    48,
      44,    46,    67,    62,    19,    63,    68,    61,    71,    35,
      36,    37,    38,    39,    40,    41,    42,    61,    44,    35,
      36,    37,    38,    39,    40,    41,    42,    46,    44,    63,
      61,    63,    59,    46,    62,    61,    72,    35,    36,    37,
      38,    39,    40,    41,    42,    61,    44,    59,    71,    71,
      68,    62,    47,    62,    62,    59,    45,    43,    68,    60,
      62,    67,    62,    46,    63,    62,     5,    63,    18,    62,
      62,    62,     8,   208,    68,   271,   137,    56,   173,   191,
     101,   180,   196,   252,   197,   256,   100,    18,   364,   132,
     306,   419,   381,   362,   434,   448,   374
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const unsigned char yystos[] =
{
       0,     3,     4,    74,    75,    80,    46,    78,    78,     0,
      75,    59,    61,    81,    81,     5,    50,    51,    76,    82,
      83,   168,   169,    78,    78,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    18,    21,    61,    80,
     100,   101,   111,   114,   115,   116,   118,   128,   129,   132,
     133,   134,    62,    82,    16,    30,    31,    46,    52,    53,
      54,    55,    56,    57,    58,    79,    88,    90,    91,    92,
      93,    94,    95,    96,    30,   102,   102,   102,    79,   135,
      70,   106,   106,   106,   106,    70,   117,    70,   103,    48,
     136,    64,    82,    59,    59,    59,    11,    12,    13,    14,
      15,    16,   119,   120,   121,   122,   123,    59,    59,    59,
      77,    91,    95,    53,    57,    52,    53,    54,    55,    60,
      64,    87,    66,    66,    66,    31,    67,    69,    79,    79,
      79,    79,    63,    59,    26,   107,   108,    78,    89,    89,
      89,    89,    47,    78,    89,    32,    33,   104,   105,    89,
      59,    17,    94,    96,   126,   127,    62,   106,   106,   106,
     106,   117,   103,    66,    60,    47,    48,    84,    85,    86,
      96,    66,    70,    98,    99,    67,    67,    67,   135,    71,
      63,    87,    60,   112,   112,   112,   112,    78,    71,   112,
      71,    63,    61,    78,    72,   125,    78,    63,    65,    77,
      78,    78,    78,    78,    78,    78,    78,    46,    63,    65,
      78,    47,    79,    97,    99,   102,   102,   102,   107,    89,
     113,    59,    61,   130,   130,   130,   130,    71,   130,   104,
      96,   109,   110,    72,   124,    47,    48,   125,   127,   112,
     112,   112,   112,   112,    59,    61,    85,    68,    71,    68,
      68,    68,    63,    34,   131,   132,   137,   138,    78,   110,
      62,    96,   130,   130,   130,   130,   130,   110,    67,   113,
      70,   140,    62,   131,    59,    70,    62,    96,   146,   149,
     150,    20,    22,    23,    24,    27,    28,    29,   141,   142,
      30,    46,    78,    91,    92,   139,    77,    71,    59,    78,
      49,    67,   145,    63,    68,    71,    63,    67,   151,    78,
      59,    70,    72,    61,    67,    70,   145,    71,   150,   141,
      68,   150,    35,    36,    37,    38,    39,    40,    41,    42,
      44,    61,   153,   159,   151,    47,    48,    79,   143,   145,
      49,   144,   145,    68,    68,    67,   163,    70,   163,    46,
     164,   165,    61,    48,   158,    46,   161,   163,    67,   154,
     159,    19,   152,    62,    63,    68,    71,   145,   145,    46,
     145,    70,   151,   166,    63,    61,   159,   155,   159,    61,
     147,    63,    59,   145,    46,    62,   154,    72,   153,   145,
     144,   145,   145,    59,    71,    68,   162,   145,   165,    62,
     154,    62,   154,   145,   161,   162,   151,    47,   145,   163,
      61,   159,    71,   167,    62,    62,   148,    61,   159,    68,
      59,   145,   154,   151,    45,   156,   154,    43,   160,   147,
     145,    60,    62,    67,    62,    61,   159,   145,   162,   145,
      46,   157,   160,   154,    62,    61,   159,    63,    63,    68,
      62,   154,   145,   157,    62,   162,    61,   159,   154,    62
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
#line 130 "xi-grammar.y"
    { yyval.modlist = yyvsp[0].modlist; modlist = yyvsp[0].modlist; }
    break;

  case 3:
#line 134 "xi-grammar.y"
    { 
		  yyval.modlist = 0; 
		}
    break;

  case 4:
#line 138 "xi-grammar.y"
    { yyval.modlist = new ModuleList(lineno, yyvsp[-1].module, yyvsp[0].modlist); }
    break;

  case 5:
#line 142 "xi-grammar.y"
    { yyval.intval = 0; }
    break;

  case 6:
#line 144 "xi-grammar.y"
    { yyval.intval = 1; }
    break;

  case 7:
#line 148 "xi-grammar.y"
    { yyval.intval = 0; }
    break;

  case 8:
#line 150 "xi-grammar.y"
    { yyval.intval = 1; }
    break;

  case 9:
#line 154 "xi-grammar.y"
    { yyval.strval = yyvsp[0].strval; }
    break;

  case 10:
#line 158 "xi-grammar.y"
    { yyval.strval = yyvsp[0].strval; }
    break;

  case 11:
#line 160 "xi-grammar.y"
    {
		  char *tmp = new char[strlen(yyvsp[-3].strval)+strlen(yyvsp[0].strval)+3];
		  sprintf(tmp,"%s::%s", yyvsp[-3].strval, yyvsp[0].strval);
		  yyval.strval = tmp;
		}
    break;

  case 12:
#line 168 "xi-grammar.y"
    { 
		    yyval.module = new Module(lineno, yyvsp[-1].strval, yyvsp[0].conslist); 
		}
    break;

  case 13:
#line 172 "xi-grammar.y"
    {  
		    yyval.module = new Module(lineno, yyvsp[-1].strval, yyvsp[0].conslist); 
		    yyval.module->setMain();
		}
    break;

  case 14:
#line 179 "xi-grammar.y"
    { yyval.conslist = 0; }
    break;

  case 15:
#line 181 "xi-grammar.y"
    { yyval.conslist = yyvsp[-2].conslist; }
    break;

  case 16:
#line 185 "xi-grammar.y"
    { yyval.conslist = 0; }
    break;

  case 17:
#line 187 "xi-grammar.y"
    { yyval.conslist = new ConstructList(lineno, yyvsp[-1].construct, yyvsp[0].conslist); }
    break;

  case 18:
#line 191 "xi-grammar.y"
    { if(yyvsp[-2].conslist) yyvsp[-2].conslist->setExtern(yyvsp[-4].intval); yyval.construct = yyvsp[-2].conslist; }
    break;

  case 19:
#line 193 "xi-grammar.y"
    { yyvsp[0].module->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].module; }
    break;

  case 20:
#line 195 "xi-grammar.y"
    { yyvsp[0].member->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].member; }
    break;

  case 21:
#line 197 "xi-grammar.y"
    { yyvsp[-1].message->setExtern(yyvsp[-2].intval); yyval.construct = yyvsp[-1].message; }
    break;

  case 22:
#line 199 "xi-grammar.y"
    { yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; }
    break;

  case 23:
#line 201 "xi-grammar.y"
    { yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; }
    break;

  case 24:
#line 203 "xi-grammar.y"
    { yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; }
    break;

  case 25:
#line 205 "xi-grammar.y"
    { yyvsp[0].chare->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].chare; }
    break;

  case 26:
#line 207 "xi-grammar.y"
    { yyvsp[0].templat->setExtern(yyvsp[-1].intval); yyval.construct = yyvsp[0].templat; }
    break;

  case 27:
#line 209 "xi-grammar.y"
    { yyval.construct = NULL; }
    break;

  case 28:
#line 211 "xi-grammar.y"
    { yyval.construct = NULL; }
    break;

  case 29:
#line 215 "xi-grammar.y"
    { yyval.tparam = new TParamType(yyvsp[0].type); }
    break;

  case 30:
#line 217 "xi-grammar.y"
    { yyval.tparam = new TParamVal(yyvsp[0].strval); }
    break;

  case 31:
#line 219 "xi-grammar.y"
    { yyval.tparam = new TParamVal(yyvsp[0].strval); }
    break;

  case 32:
#line 223 "xi-grammar.y"
    { yyval.tparlist = new TParamList(yyvsp[0].tparam); }
    break;

  case 33:
#line 225 "xi-grammar.y"
    { yyval.tparlist = new TParamList(yyvsp[-2].tparam, yyvsp[0].tparlist); }
    break;

  case 34:
#line 229 "xi-grammar.y"
    { yyval.tparlist = 0; }
    break;

  case 35:
#line 231 "xi-grammar.y"
    { yyval.tparlist = yyvsp[0].tparlist; }
    break;

  case 36:
#line 235 "xi-grammar.y"
    { yyval.tparlist = 0; }
    break;

  case 37:
#line 237 "xi-grammar.y"
    { yyval.tparlist = yyvsp[-1].tparlist; }
    break;

  case 38:
#line 241 "xi-grammar.y"
    { yyval.type = new BuiltinType("int"); }
    break;

  case 39:
#line 243 "xi-grammar.y"
    { yyval.type = new BuiltinType("long"); }
    break;

  case 40:
#line 245 "xi-grammar.y"
    { yyval.type = new BuiltinType("short"); }
    break;

  case 41:
#line 247 "xi-grammar.y"
    { yyval.type = new BuiltinType("char"); }
    break;

  case 42:
#line 249 "xi-grammar.y"
    { yyval.type = new BuiltinType("unsigned int"); }
    break;

  case 43:
#line 251 "xi-grammar.y"
    { yyval.type = new BuiltinType("unsigned long"); }
    break;

  case 44:
#line 253 "xi-grammar.y"
    { yyval.type = new BuiltinType("unsigned short"); }
    break;

  case 45:
#line 255 "xi-grammar.y"
    { yyval.type = new BuiltinType("unsigned char"); }
    break;

  case 46:
#line 257 "xi-grammar.y"
    { yyval.type = new BuiltinType("long long"); }
    break;

  case 47:
#line 259 "xi-grammar.y"
    { yyval.type = new BuiltinType("float"); }
    break;

  case 48:
#line 261 "xi-grammar.y"
    { yyval.type = new BuiltinType("double"); }
    break;

  case 49:
#line 263 "xi-grammar.y"
    { yyval.type = new BuiltinType("long double"); }
    break;

  case 50:
#line 265 "xi-grammar.y"
    { yyval.type = new BuiltinType("void"); }
    break;

  case 51:
#line 268 "xi-grammar.y"
    { yyval.ntype = new NamedType(yyvsp[-1].strval,yyvsp[0].tparlist); }
    break;

  case 52:
#line 269 "xi-grammar.y"
    { yyval.ntype = new NamedType(yyvsp[-1].strval,yyvsp[0].tparlist); }
    break;

  case 53:
#line 272 "xi-grammar.y"
    { yyval.type = yyvsp[0].type; }
    break;

  case 54:
#line 274 "xi-grammar.y"
    { yyval.type = yyvsp[0].ntype; }
    break;

  case 55:
#line 278 "xi-grammar.y"
    { yyval.ptype = new PtrType(yyvsp[-1].type); }
    break;

  case 56:
#line 282 "xi-grammar.y"
    { yyvsp[-1].ptype->indirect(); yyval.ptype = yyvsp[-1].ptype; }
    break;

  case 57:
#line 284 "xi-grammar.y"
    { yyvsp[-1].ptype->indirect(); yyval.ptype = yyvsp[-1].ptype; }
    break;

  case 58:
#line 288 "xi-grammar.y"
    { yyval.ftype = new FuncType(yyvsp[-7].type, yyvsp[-4].strval, yyvsp[-1].plist); }
    break;

  case 59:
#line 292 "xi-grammar.y"
    { yyval.type = yyvsp[0].type; }
    break;

  case 60:
#line 294 "xi-grammar.y"
    { yyval.type = yyvsp[0].ptype; }
    break;

  case 61:
#line 296 "xi-grammar.y"
    { yyval.type = yyvsp[0].ptype; }
    break;

  case 62:
#line 298 "xi-grammar.y"
    { yyval.type = yyvsp[0].ftype; }
    break;

  case 63:
#line 301 "xi-grammar.y"
    { yyval.type = yyvsp[0].type; }
    break;

  case 64:
#line 303 "xi-grammar.y"
    { yyval.type = yyvsp[-1].type; }
    break;

  case 65:
#line 307 "xi-grammar.y"
    { yyval.type = new ReferenceType(yyvsp[-1].type); }
    break;

  case 66:
#line 309 "xi-grammar.y"
    { yyval.type = yyvsp[0].type; }
    break;

  case 67:
#line 313 "xi-grammar.y"
    { yyval.val = new Value(yyvsp[0].strval); }
    break;

  case 68:
#line 315 "xi-grammar.y"
    { yyval.val = new Value(yyvsp[0].strval); }
    break;

  case 69:
#line 319 "xi-grammar.y"
    { yyval.val = yyvsp[-1].val; }
    break;

  case 70:
#line 323 "xi-grammar.y"
    { yyval.vallist = 0; }
    break;

  case 71:
#line 325 "xi-grammar.y"
    { yyval.vallist = new ValueList(yyvsp[-1].val, yyvsp[0].vallist); }
    break;

  case 72:
#line 329 "xi-grammar.y"
    { yyval.readonly = new Readonly(lineno, yyvsp[-2].type, yyvsp[-1].strval, yyvsp[0].vallist); }
    break;

  case 73:
#line 333 "xi-grammar.y"
    { yyval.readonly = new Readonly(lineno, yyvsp[-2].type, yyvsp[0].strval, 0, 1); }
    break;

  case 74:
#line 337 "xi-grammar.y"
    { yyval.intval = 0;}
    break;

  case 75:
#line 339 "xi-grammar.y"
    { yyval.intval = 0;}
    break;

  case 76:
#line 343 "xi-grammar.y"
    { yyval.intval = 0; }
    break;

  case 77:
#line 345 "xi-grammar.y"
    { 
		  /*
		  printf("Warning: Message attributes are being phased out.\n");
		  printf("Warning: Please remove them from interface files.\n");
		  */
		  yyval.intval = yyvsp[-1].intval; 
		}
    break;

  case 78:
#line 355 "xi-grammar.y"
    { yyval.intval = yyvsp[0].intval; }
    break;

  case 79:
#line 357 "xi-grammar.y"
    { yyval.intval = yyvsp[-2].intval | yyvsp[0].intval; }
    break;

  case 80:
#line 361 "xi-grammar.y"
    { yyval.intval = 0; }
    break;

  case 81:
#line 363 "xi-grammar.y"
    { yyval.intval = 0; }
    break;

  case 82:
#line 367 "xi-grammar.y"
    { yyval.cattr = 0; }
    break;

  case 83:
#line 369 "xi-grammar.y"
    { yyval.cattr = yyvsp[-1].cattr; }
    break;

  case 84:
#line 373 "xi-grammar.y"
    { yyval.cattr = yyvsp[0].cattr; }
    break;

  case 85:
#line 375 "xi-grammar.y"
    { yyval.cattr = yyvsp[-2].cattr | yyvsp[0].cattr; }
    break;

  case 86:
#line 379 "xi-grammar.y"
    { yyval.cattr = Chare::CMIGRATABLE; }
    break;

  case 87:
#line 383 "xi-grammar.y"
    { yyval.mv = new MsgVar(yyvsp[-4].type, yyvsp[-3].strval); }
    break;

  case 88:
#line 387 "xi-grammar.y"
    { yyval.mvlist = new MsgVarList(yyvsp[0].mv); }
    break;

  case 89:
#line 389 "xi-grammar.y"
    { yyval.mvlist = new MsgVarList(yyvsp[-1].mv, yyvsp[0].mvlist); }
    break;

  case 90:
#line 393 "xi-grammar.y"
    { yyval.message = new Message(lineno, yyvsp[0].ntype); }
    break;

  case 91:
#line 395 "xi-grammar.y"
    { yyval.message = new Message(lineno, yyvsp[-3].ntype, yyvsp[-1].mvlist); }
    break;

  case 92:
#line 399 "xi-grammar.y"
    { yyval.typelist = 0; }
    break;

  case 93:
#line 401 "xi-grammar.y"
    { yyval.typelist = yyvsp[0].typelist; }
    break;

  case 94:
#line 405 "xi-grammar.y"
    { yyval.typelist = new TypeList(yyvsp[0].ntype); }
    break;

  case 95:
#line 407 "xi-grammar.y"
    { yyval.typelist = new TypeList(yyvsp[-2].ntype, yyvsp[0].typelist); }
    break;

  case 96:
#line 411 "xi-grammar.y"
    { yyval.chare = new Chare(lineno, yyvsp[-3].cattr, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 97:
#line 413 "xi-grammar.y"
    { yyval.chare = new MainChare(lineno, yyvsp[-3].cattr, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 98:
#line 417 "xi-grammar.y"
    { yyval.chare = new Group(lineno, yyvsp[-3].cattr, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 99:
#line 421 "xi-grammar.y"
    { yyval.chare = new NodeGroup(lineno, yyvsp[-3].cattr, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 100:
#line 425 "xi-grammar.y"
    {/*Stupid special case for [1D] indices*/
			char *buf=new char[40];
			sprintf(buf,"%sD",yyvsp[-2].strval);
			yyval.ntype = new NamedType(buf); 
		}
    break;

  case 101:
#line 431 "xi-grammar.y"
    { yyval.ntype = new NamedType(yyvsp[-1].strval); }
    break;

  case 102:
#line 435 "xi-grammar.y"
    {  yyval.chare = new Array(lineno, 0, yyvsp[-3].ntype, yyvsp[-2].ntype, yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 103:
#line 439 "xi-grammar.y"
    { yyval.chare = new Chare(lineno, yyvsp[-3].cattr, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist);}
    break;

  case 104:
#line 441 "xi-grammar.y"
    { yyval.chare = new MainChare(lineno, yyvsp[-3].cattr, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 105:
#line 445 "xi-grammar.y"
    { yyval.chare = new Group(lineno, yyvsp[-3].cattr, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 106:
#line 449 "xi-grammar.y"
    { yyval.chare = new NodeGroup( lineno, yyvsp[-3].cattr, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 107:
#line 453 "xi-grammar.y"
    { yyval.chare = new Array( lineno, 0, yyvsp[-3].ntype, new NamedType(yyvsp[-2].strval), yyvsp[-1].typelist, yyvsp[0].mbrlist); }
    break;

  case 108:
#line 457 "xi-grammar.y"
    { yyval.message = new Message(lineno, new NamedType(yyvsp[-1].strval)); }
    break;

  case 109:
#line 459 "xi-grammar.y"
    { yyval.message = new Message(lineno, new NamedType(yyvsp[-4].strval), yyvsp[-2].mvlist); }
    break;

  case 110:
#line 463 "xi-grammar.y"
    { yyval.type = 0; }
    break;

  case 111:
#line 465 "xi-grammar.y"
    { yyval.type = yyvsp[0].type; }
    break;

  case 112:
#line 469 "xi-grammar.y"
    { yyval.strval = 0; }
    break;

  case 113:
#line 471 "xi-grammar.y"
    { yyval.strval = yyvsp[0].strval; }
    break;

  case 114:
#line 473 "xi-grammar.y"
    { yyval.strval = yyvsp[0].strval; }
    break;

  case 115:
#line 477 "xi-grammar.y"
    { yyval.tvar = new TType(new NamedType(yyvsp[-1].strval), yyvsp[0].type); }
    break;

  case 116:
#line 479 "xi-grammar.y"
    { yyval.tvar = new TFunc(yyvsp[-1].ftype, yyvsp[0].strval); }
    break;

  case 117:
#line 481 "xi-grammar.y"
    { yyval.tvar = new TName(yyvsp[-2].type, yyvsp[-1].strval, yyvsp[0].strval); }
    break;

  case 118:
#line 485 "xi-grammar.y"
    { yyval.tvarlist = new TVarList(yyvsp[0].tvar); }
    break;

  case 119:
#line 487 "xi-grammar.y"
    { yyval.tvarlist = new TVarList(yyvsp[-2].tvar, yyvsp[0].tvarlist); }
    break;

  case 120:
#line 491 "xi-grammar.y"
    { yyval.tvarlist = yyvsp[-1].tvarlist; }
    break;

  case 121:
#line 495 "xi-grammar.y"
    { yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); }
    break;

  case 122:
#line 497 "xi-grammar.y"
    { yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); }
    break;

  case 123:
#line 499 "xi-grammar.y"
    { yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); }
    break;

  case 124:
#line 501 "xi-grammar.y"
    { yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].chare); yyvsp[0].chare->setTemplate(yyval.templat); }
    break;

  case 125:
#line 503 "xi-grammar.y"
    { yyval.templat = new Template(yyvsp[-1].tvarlist, yyvsp[0].message); yyvsp[0].message->setTemplate(yyval.templat); }
    break;

  case 126:
#line 507 "xi-grammar.y"
    { yyval.mbrlist = 0; }
    break;

  case 127:
#line 509 "xi-grammar.y"
    { yyval.mbrlist = yyvsp[-2].mbrlist; }
    break;

  case 128:
#line 513 "xi-grammar.y"
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
#line 532 "xi-grammar.y"
    { yyval.mbrlist = new MemberList(yyvsp[-1].member, yyvsp[0].mbrlist); }
    break;

  case 130:
#line 536 "xi-grammar.y"
    { yyval.member = yyvsp[-1].readonly; }
    break;

  case 131:
#line 538 "xi-grammar.y"
    { yyval.member = yyvsp[-1].readonly; }
    break;

  case 133:
#line 541 "xi-grammar.y"
    { yyval.member = yyvsp[-1].member; }
    break;

  case 134:
#line 543 "xi-grammar.y"
    { yyval.member = yyvsp[-1].pupable; }
    break;

  case 135:
#line 545 "xi-grammar.y"
    { yyval.member = yyvsp[-1].includeFile; }
    break;

  case 136:
#line 549 "xi-grammar.y"
    { yyval.member = new InitCall(lineno, yyvsp[0].strval, 1); }
    break;

  case 137:
#line 551 "xi-grammar.y"
    { yyval.member = new InitCall(lineno, yyvsp[-3].strval, 1); }
    break;

  case 138:
#line 553 "xi-grammar.y"
    { printf("Warning: deprecated use of initcall. Use initnode or initproc instead.\n"); 
		  yyval.member = new InitCall(lineno, yyvsp[0].strval, 1); }
    break;

  case 139:
#line 556 "xi-grammar.y"
    { printf("Warning: deprecated use of initcall. Use initnode or initproc instead.\n");
		  yyval.member = new InitCall(lineno, yyvsp[-3].strval, 1); }
    break;

  case 140:
#line 561 "xi-grammar.y"
    { yyval.member = new InitCall(lineno, yyvsp[0].strval, 0); }
    break;

  case 141:
#line 563 "xi-grammar.y"
    { yyval.member = new InitCall(lineno, yyvsp[-3].strval, 0); }
    break;

  case 142:
#line 567 "xi-grammar.y"
    { yyval.pupable = new PUPableClass(lineno,yyvsp[0].strval,0); }
    break;

  case 143:
#line 569 "xi-grammar.y"
    { yyval.pupable = new PUPableClass(lineno,yyvsp[-2].strval,yyvsp[0].pupable); }
    break;

  case 144:
#line 573 "xi-grammar.y"
    { yyval.includeFile = new IncludeFile(lineno,yyvsp[0].strval,0); }
    break;

  case 145:
#line 577 "xi-grammar.y"
    { yyval.member = yyvsp[-1].entry; }
    break;

  case 146:
#line 579 "xi-grammar.y"
    { yyval.member = yyvsp[0].member; }
    break;

  case 147:
#line 583 "xi-grammar.y"
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
#line 594 "xi-grammar.y"
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
#line 607 "xi-grammar.y"
    { yyval.type = new BuiltinType("void"); }
    break;

  case 150:
#line 609 "xi-grammar.y"
    { yyval.type = yyvsp[0].ptype; }
    break;

  case 151:
#line 613 "xi-grammar.y"
    { yyval.intval = 0; }
    break;

  case 152:
#line 615 "xi-grammar.y"
    { yyval.intval = yyvsp[-1].intval; }
    break;

  case 153:
#line 619 "xi-grammar.y"
    { yyval.intval = yyvsp[0].intval; }
    break;

  case 154:
#line 621 "xi-grammar.y"
    { yyval.intval = yyvsp[-2].intval | yyvsp[0].intval; }
    break;

  case 155:
#line 625 "xi-grammar.y"
    { yyval.intval = STHREADED; }
    break;

  case 156:
#line 627 "xi-grammar.y"
    { yyval.intval = SSYNC; }
    break;

  case 157:
#line 629 "xi-grammar.y"
    { yyval.intval = SLOCKED; }
    break;

  case 158:
#line 631 "xi-grammar.y"
    { yyval.intval = SCREATEHERE; }
    break;

  case 159:
#line 633 "xi-grammar.y"
    { yyval.intval = SCREATEHOME; }
    break;

  case 160:
#line 635 "xi-grammar.y"
    { yyval.intval = SNOKEEP; }
    break;

  case 161:
#line 637 "xi-grammar.y"
    { yyval.intval = SIMMEDIATE; }
    break;

  case 162:
#line 641 "xi-grammar.y"
    { yyval.val = new Value(yyvsp[0].strval); }
    break;

  case 163:
#line 643 "xi-grammar.y"
    { yyval.val = new Value(yyvsp[0].strval); }
    break;

  case 164:
#line 645 "xi-grammar.y"
    { yyval.val = new Value(yyvsp[0].strval); }
    break;

  case 165:
#line 649 "xi-grammar.y"
    { yyval.strval = ""; }
    break;

  case 166:
#line 651 "xi-grammar.y"
    { yyval.strval = yyvsp[0].strval; }
    break;

  case 167:
#line 653 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen(yyvsp[-2].strval)+strlen(yyvsp[0].strval)+3];
			sprintf(tmp,"%s, %s", yyvsp[-2].strval, yyvsp[0].strval);
			yyval.strval = tmp;
		}
    break;

  case 168:
#line 661 "xi-grammar.y"
    { yyval.strval = ""; }
    break;

  case 169:
#line 663 "xi-grammar.y"
    { yyval.strval = yyvsp[0].strval; }
    break;

  case 170:
#line 665 "xi-grammar.y"
    {  /*Returned only when in_bracket*/
			char *tmp = new char[strlen(yyvsp[-4].strval)+strlen(yyvsp[-2].strval)+strlen(yyvsp[0].strval)+3];
			sprintf(tmp,"%s[%s]%s", yyvsp[-4].strval, yyvsp[-2].strval, yyvsp[0].strval);
			yyval.strval = tmp;
		}
    break;

  case 171:
#line 671 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen(yyvsp[-4].strval)+strlen(yyvsp[-2].strval)+strlen(yyvsp[0].strval)+3];
			sprintf(tmp,"%s{%s}%s", yyvsp[-4].strval, yyvsp[-2].strval, yyvsp[0].strval);
			yyval.strval = tmp;
		}
    break;

  case 172:
#line 677 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen(yyvsp[-4].strval)+strlen(yyvsp[-2].strval)+strlen(yyvsp[0].strval)+3];
			sprintf(tmp,"%s(%s)%s", yyvsp[-4].strval, yyvsp[-2].strval, yyvsp[0].strval);
			yyval.strval = tmp;
		}
    break;

  case 173:
#line 683 "xi-grammar.y"
    { /*Returned only when in_braces*/
			char *tmp = new char[strlen(yyvsp[-2].strval)+strlen(yyvsp[0].strval)+3];
			sprintf(tmp,"(%s)%s", yyvsp[-2].strval, yyvsp[0].strval);
			yyval.strval = tmp;
		}
    break;

  case 174:
#line 691 "xi-grammar.y"
    {  /*Start grabbing CPROGRAM segments*/
			in_bracket=1;
			yyval.pname = new Parameter(lineno, yyvsp[-2].type,yyvsp[-1].strval);
		}
    break;

  case 175:
#line 698 "xi-grammar.y"
    { 
                   /*Start grabbing CPROGRAM segments*/
			in_braces=1;
			yyval.intval = 0;
		}
    break;

  case 176:
#line 706 "xi-grammar.y"
    { 
			in_braces=0;
			yyval.intval = 0;
		}
    break;

  case 177:
#line 713 "xi-grammar.y"
    { yyval.pname = new Parameter(lineno, yyvsp[0].type);}
    break;

  case 178:
#line 715 "xi-grammar.y"
    { yyval.pname = new Parameter(lineno, yyvsp[-1].type,yyvsp[0].strval);}
    break;

  case 179:
#line 717 "xi-grammar.y"
    { yyval.pname = new Parameter(lineno, yyvsp[-3].type,yyvsp[-2].strval,0,yyvsp[0].val);}
    break;

  case 180:
#line 719 "xi-grammar.y"
    { /*Stop grabbing CPROGRAM segments*/
			in_bracket=0;
			yyval.pname = new Parameter(lineno, yyvsp[-2].pname->getType(), yyvsp[-2].pname->getName() ,yyvsp[-1].strval);
		}
    break;

  case 181:
#line 726 "xi-grammar.y"
    { yyval.plist = new ParamList(yyvsp[0].pname); }
    break;

  case 182:
#line 728 "xi-grammar.y"
    { yyval.plist = new ParamList(yyvsp[-2].pname,yyvsp[0].plist); }
    break;

  case 183:
#line 732 "xi-grammar.y"
    { yyval.plist = yyvsp[-1].plist; }
    break;

  case 184:
#line 734 "xi-grammar.y"
    { yyval.plist = 0; }
    break;

  case 185:
#line 738 "xi-grammar.y"
    { yyval.val = 0; }
    break;

  case 186:
#line 740 "xi-grammar.y"
    { yyval.val = new Value(yyvsp[0].strval); }
    break;

  case 187:
#line 744 "xi-grammar.y"
    { yyval.sc = 0; }
    break;

  case 188:
#line 746 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SSDAGENTRY, yyvsp[0].sc); }
    break;

  case 189:
#line 748 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SSDAGENTRY, yyvsp[-1].sc); }
    break;

  case 190:
#line 752 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SSLIST, yyvsp[0].sc); }
    break;

  case 191:
#line 754 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SSLIST, yyvsp[-1].sc, yyvsp[0].sc);  }
    break;

  case 192:
#line 758 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SOLIST, yyvsp[0].sc); }
    break;

  case 193:
#line 760 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SOLIST, yyvsp[-1].sc, yyvsp[0].sc); }
    break;

  case 194:
#line 764 "xi-grammar.y"
    { yyval.sc = 0; }
    break;

  case 195:
#line 766 "xi-grammar.y"
    { yyval.sc = yyvsp[-1].sc; }
    break;

  case 196:
#line 770 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SPUBLISHES, new SdagConstruct(SIDENT, yyvsp[0].strval)); }
    break;

  case 197:
#line 772 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SPUBLISHES, new SdagConstruct(SIDENT, yyvsp[-2].strval), yyvsp[0].sc);  }
    break;

  case 198:
#line 776 "xi-grammar.y"
    { yyval.strval = yyvsp[0].strval; }
    break;

  case 199:
#line 778 "xi-grammar.y"
    { yyval.strval = 0; }
    break;

  case 200:
#line 782 "xi-grammar.y"
    { RemoveSdagComments(yyvsp[-2].strval);
		   yyval.sc = new SdagConstruct(SATOMIC, new XStr(yyvsp[-2].strval), yyvsp[0].sc, 0,0,0,0, 0 ); 
		   if (yyvsp[-4].strval) { yyvsp[-4].strval[strlen(yyvsp[-4].strval)-1]=0; yyval.sc->traceName = new XStr(yyvsp[-4].strval+1); }
		 }
    break;

  case 201:
#line 787 "xi-grammar.y"
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

  case 202:
#line 801 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SWHEN, 0, 0,0,0,0,0,yyvsp[-2].entrylist); }
    break;

  case 203:
#line 803 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SWHEN, 0, 0, 0,0,0, yyvsp[0].sc, yyvsp[-1].entrylist); }
    break;

  case 204:
#line 805 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SWHEN, 0, 0, 0,0,0, yyvsp[-1].sc, yyvsp[-3].entrylist); }
    break;

  case 205:
#line 807 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SOVERLAP,0, 0,0,0,0,yyvsp[-1].sc, 0); }
    break;

  case 206:
#line 809 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SFOR, 0, new SdagConstruct(SINT_EXPR, yyvsp[-8].strval), new SdagConstruct(SINT_EXPR, yyvsp[-6].strval),
		             new SdagConstruct(SINT_EXPR, yyvsp[-4].strval), 0, yyvsp[-1].sc, 0); }
    break;

  case 207:
#line 812 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SFOR, 0, new SdagConstruct(SINT_EXPR, yyvsp[-6].strval), new SdagConstruct(SINT_EXPR, yyvsp[-4].strval), 
		         new SdagConstruct(SINT_EXPR, yyvsp[-2].strval), 0, yyvsp[0].sc, 0); }
    break;

  case 208:
#line 815 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SFORALL, 0, new SdagConstruct(SIDENT, yyvsp[-9].strval), new SdagConstruct(SINT_EXPR, yyvsp[-6].strval), 
		             new SdagConstruct(SINT_EXPR, yyvsp[-4].strval), new SdagConstruct(SINT_EXPR, yyvsp[-2].strval), yyvsp[0].sc, 0); }
    break;

  case 209:
#line 818 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SFORALL, 0, new SdagConstruct(SIDENT, yyvsp[-11].strval), new SdagConstruct(SINT_EXPR, yyvsp[-8].strval), 
		                 new SdagConstruct(SINT_EXPR, yyvsp[-6].strval), new SdagConstruct(SINT_EXPR, yyvsp[-4].strval), yyvsp[-1].sc, 0); }
    break;

  case 210:
#line 821 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, yyvsp[-3].strval), yyvsp[0].sc,0,0,yyvsp[-1].sc,0); }
    break;

  case 211:
#line 823 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, yyvsp[-5].strval), yyvsp[0].sc,0,0,yyvsp[-2].sc,0); }
    break;

  case 212:
#line 825 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SIF, 0, new SdagConstruct(SINT_EXPR, yyvsp[-2].strval), 0,0,0,yyvsp[0].sc,0); }
    break;

  case 213:
#line 827 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SWHILE, 0, new SdagConstruct(SINT_EXPR, yyvsp[-4].strval), 0,0,0,yyvsp[-1].sc,0); }
    break;

  case 214:
#line 829 "xi-grammar.y"
    { yyval.sc = yyvsp[-1].sc; }
    break;

  case 215:
#line 833 "xi-grammar.y"
    { yyval.sc = 0; }
    break;

  case 216:
#line 835 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SELSE, 0,0,0,0,0, yyvsp[0].sc,0); }
    break;

  case 217:
#line 837 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SELSE, 0,0,0,0,0, yyvsp[-1].sc,0); }
    break;

  case 218:
#line 840 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SFORWARD, new SdagConstruct(SIDENT, yyvsp[0].strval)); }
    break;

  case 219:
#line 842 "xi-grammar.y"
    { yyval.sc = new SdagConstruct(SFORWARD, new SdagConstruct(SIDENT, yyvsp[-2].strval), yyvsp[0].sc);  }
    break;

  case 220:
#line 846 "xi-grammar.y"
    { in_int_expr = 0; yyval.intval = 0; }
    break;

  case 221:
#line 850 "xi-grammar.y"
    { in_int_expr = 1; yyval.intval = 0; }
    break;

  case 222:
#line 854 "xi-grammar.y"
    { 
		  if (yyvsp[0].plist != 0)
		     yyval.entry = new Entry(lineno, 0, 0, yyvsp[-1].strval, yyvsp[0].plist, 0, 0, 0, 0); 
		  else
		     yyval.entry = new Entry(lineno, 0, 0, yyvsp[-1].strval, 
				new ParamList(new Parameter(0, new BuiltinType("void"))), 0, 0, 0, 0); 
		}
    break;

  case 223:
#line 862 "xi-grammar.y"
    { if (yyvsp[0].plist != 0)
		    yyval.entry = new Entry(lineno, 0, 0, yyvsp[-4].strval, yyvsp[0].plist, 0, 0, yyvsp[-2].strval, 0); 
		  else
		    yyval.entry = new Entry(lineno, 0, 0, yyvsp[-4].strval, new ParamList(new Parameter(0, new BuiltinType("void"))), 0, 0, yyvsp[-2].strval, 0); 
		}
    break;

  case 224:
#line 870 "xi-grammar.y"
    { yyval.entrylist = new EntryList(yyvsp[0].entry); }
    break;

  case 225:
#line 872 "xi-grammar.y"
    { yyval.entrylist = new EntryList(yyvsp[-2].entry,yyvsp[0].entrylist); }
    break;

  case 226:
#line 876 "xi-grammar.y"
    { in_bracket=1; }
    break;

  case 227:
#line 879 "xi-grammar.y"
    { in_bracket=0; }
    break;

  case 228:
#line 883 "xi-grammar.y"
    { if (!macroDefined(yyvsp[0].strval, 1)) in_comment = 1; }
    break;

  case 229:
#line 887 "xi-grammar.y"
    { if (!macroDefined(yyvsp[0].strval, 0)) in_comment = 1; }
    break;


    }

/* Line 991 of yacc.c.  */
#line 2833 "y.tab.c"

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


#line 890 "xi-grammar.y"

void yyerror(const char *mesg)
{
  cout << cur_file<<":"<<lineno<<": Charmxi syntax error> " << mesg << endl;
  // return 0;
}

