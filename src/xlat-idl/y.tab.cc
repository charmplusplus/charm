
# line 72 "idl.yy"
#include <idl.hh>
#include <idl_extern.hh>

#include <fe_private.hh>

#include <stdio.h>

int yywrap();


# line 89 "idl.yy"
typedef union
#ifdef __cplusplus
	YYSTYPE
#endif
 {
  AST_Decl		*dcval;		/* Decl value		*/
  UTL_StrList		*slval;		/* String list		*/
  UTL_NameList		*nlval;		/* Name list		*/
  UTL_ExprList		*elval;		/* Expression list	*/
  UTL_LabelList		*llval;		/* Label list		*/
  UTL_DeclList		*dlval;		/* Declaration list	*/
  FE_InterfaceHeader	*ihval;		/* Interface header	*/
  AST_Expression	*exval;		/* Expression value	*/
  AST_UnionLabel	*ulval;		/* Union label		*/
  AST_Field		*ffval;		/* Field value		*/
  AST_Expression::ExprType etval;	/* Expression type	*/
  AST_Argument::Direction dival;	/* Argument direction	*/
  AST_Operation::Flags	ofval;		/* Operation flags	*/
  FE_Declarator		*deval;		/* Declarator value	*/
  idl_bool		bval;		/* Boolean value	*/
  long			ival;		/* Long value		*/
  double		dval;		/* Double value		*/
  float			fval;		/* Float value		*/
  char			cval;		/* Char value		*/
  
  String		*sval;		/* String value		*/
  char			*strval;	/* char * value		*/
  Identifier		*idval;		/* Identifier		*/
  UTL_IdList		*idlist;	/* Identifier list	*/
} YYSTYPE;
# define IDENTIFIER 257
# define CONST 258
# define MODULE 259
# define INTERFACE 260
# define TYPEDEF 261
# define LONG 262
# define SHORT 263
# define UNSIGNED 264
# define DOUBLE 265
# define FLOAT 266
# define CHAR 267
# define WCHAR 268
# define OCTET 269
# define BOOLEAN 270
# define ANY 271
# define STRUCT 272
# define UNION 273
# define SWITCH 274
# define ENUM 275
# define SEQUENCE 276
# define STRING 277
# define WSTRING 278
# define EXCEPTION 279
# define CASE 280
# define DEFAULT 281
# define READONLY 282
# define ATTRIBUTE 283
# define THREADED 284
# define ONEWAY 285
# define IDEMPOTENT 286
# define VOID 287
# define IN 288
# define OUT 289
# define INOUT 290
# define RAISES 291
# define CONTEXT 292
# define INTEGER_LITERAL 293
# define STRING_LITERAL 294
# define CHARACTER_LITERAL 295
# define FLOATING_PT_LITERAL 296
# define TRUETOK 297
# define FALSETOK 298
# define SCOPE_DELIMITOR 299
# define LEFT_SHIFT 300
# define RIGHT_SHIFT 301

#ifdef __STDC__
#include <stdlib.h>
#include <string.h>
#else
#include <malloc.h>
#include <memory.h>
#endif

#include <values.h>

#ifdef __cplusplus

#ifndef yyerror
	void yyerror(const char *);
#endif

#ifndef yylex
#ifdef __EXTERN_C__
	extern "C" { int yylex(void); }
#else
	int yylex(void);
#endif
#endif
	int yyparse(void);

#endif
#define yyclearin yychar = -1
#define yyerrok yyerrflag = 0
extern int yychar;
extern int yyerrflag;
YYSTYPE yylval;
YYSTYPE yyval;
typedef int yytabelem;
#ifndef YYMAXDEPTH
#define YYMAXDEPTH 150
#endif
#if YYMAXDEPTH > 0
int yy_yys[YYMAXDEPTH], *yys = yy_yys;
YYSTYPE yy_yyv[YYMAXDEPTH], *yyv = yy_yyv;
#else	/* user does initial allocation */
int *yys;
YYSTYPE *yyv;
#endif
static int yymaxdepth = YYMAXDEPTH;
# define YYERRCODE 256

# line 2185 "idl.yy"

/* programs */

/*
 * ???
 */
int
yywrap()
{
  return 1;
}

/*
 * Report an error situation discovered in a production
 *
 * This does not do anything since we report all error situations through
 * idl_global->err() operations
 */
void
yyerror(char *)
{
}
yytabelem yyexca[] ={
-1, 0,
	0, 3,
	-2, 0,
-1, 1,
	0, -1,
	-2, 0,
-1, 3,
	0, 3,
	125, 3,
	-2, 0,
-1, 23,
	123, 32,
	-2, 57,
-1, 131,
	91, 224,
	-2, 133,
-1, 146,
	257, 251,
	262, 251,
	263, 251,
	264, 251,
	265, 251,
	266, 251,
	267, 251,
	268, 251,
	269, 251,
	270, 251,
	271, 251,
	276, 251,
	277, 251,
	278, 251,
	283, 236,
	287, 251,
	299, 251,
	125, 25,
	-2, 0,
-1, 178,
	125, 3,
	-2, 0,
-1, 221,
	125, 240,
	-2, 0,
-1, 266,
	125, 156,
	-2, 0,
-1, 315,
	41, 254,
	-2, 256,
-1, 349,
	125, 179,
	-2, 0,
	};
# define YYNPROD 280
# define YYLAST 506
yytabelem yyact[]={

    68,   128,   152,   325,   337,   125,    51,   164,   127,   107,
   340,   226,    83,   229,   225,   206,   207,    74,    84,    85,
    72,    73,    75,    76,    78,    77,    79,    19,    20,   385,
    21,    86,    87,    88,    83,    98,   166,   100,   101,   102,
    55,    97,   380,   105,   344,   322,   166,    50,   339,   161,
    83,   162,   195,   196,    69,    74,    84,    85,    72,    73,
    75,    76,    78,    77,    79,    19,    20,   238,    21,    86,
    87,    88,   342,   341,    83,   109,    69,   342,   341,    74,
    84,    85,    72,    73,    75,    76,    78,    77,    79,   327,
   328,   329,    69,    86,    87,    88,   144,   109,   138,    63,
   110,    58,    13,    83,   241,    13,   230,   131,   302,   134,
   115,   116,   126,    99,   221,    62,    69,    83,    61,   140,
    59,   151,    74,    84,    85,    72,    73,    75,    76,    78,
    77,    79,   163,    92,   150,    83,    86,    87,    88,   159,
    74,    84,    85,    72,    73,    75,    76,    78,    77,    93,
   200,   155,    91,   194,    90,    87,    88,   158,   262,    69,
    83,   160,     6,     5,   156,   298,    84,    85,   157,   216,
    75,    76,    78,    77,   154,     4,     2,    69,    21,   359,
    25,   299,   219,    55,   189,   288,    14,    55,   287,    10,
   286,   231,     9,   203,    14,    18,    24,    10,   240,    55,
    19,    20,    69,    21,   320,   145,   143,    15,    19,    20,
   192,    21,   193,   195,   196,    15,   142,   141,   103,   204,
   309,   201,   111,   260,   259,   247,   220,   139,   114,    55,
    57,    12,    56,    11,    12,   113,    11,   205,   376,   363,
   112,   310,   290,   277,   276,   131,   281,   275,    13,   274,
   273,   280,   272,    83,    48,    47,    46,    45,   173,   174,
    44,    43,   378,    83,   285,   365,    41,   208,   131,   209,
   383,   346,   300,   212,    55,   198,   289,   297,   210,   197,
    13,   381,   375,   211,   333,   301,    55,   323,   311,   167,
   168,   169,   170,   171,   172,    69,   258,   367,   345,   167,
   168,   169,   170,   171,   172,    69,   315,   227,   131,   186,
   185,    60,    67,   384,    66,   374,   313,   355,   356,   332,
   312,   358,   184,   213,   214,   215,   335,   357,   334,   324,
   319,   318,   331,   317,   314,   308,   279,   263,   224,   266,
   177,   119,   347,    55,    34,    94,    96,   242,    95,   307,
   255,   256,   257,   278,   360,   222,   249,   366,    55,   131,
   370,   369,   368,   364,   126,   362,   253,   254,   283,   294,
   250,   296,   303,   131,   377,   251,   252,    12,   248,    11,
   246,   149,   218,   137,    82,   295,   217,   136,   293,    81,
   292,   175,   117,   202,   135,   306,   282,   271,   270,   228,
   181,   123,    38,   372,   373,   354,   353,   351,   371,    12,
   350,    11,   349,   348,   336,   330,   316,   305,   269,   180,
   122,    37,   268,   304,   267,   265,   223,   179,   121,    36,
   244,   106,    49,    32,   261,   176,   118,    33,   133,   108,
   243,   237,   236,   188,   235,   187,   234,   233,   232,   183,
   104,    42,   182,   146,   124,    39,    17,    16,   264,   178,
   120,    35,    31,    30,     8,    29,     7,    28,    27,    26,
     3,     1,    23,   190,   130,   129,   191,   326,    53,    71,
    70,    64,    89,   361,   165,   153,   284,    22,   148,   352,
   338,   245,   199,   321,    40,   147,   382,   379,   343,    80,
   239,   132,   291,    65,    54,    52 };
yytabelem yypact[]={

   -64,-10000000,-10000000,   -64,-10000000,-10000000,-10000000,-10000000,-10000000,-10000000,
-10000000,-10000000,-10000000,-10000000,-10000000,-10000000,-10000000,-10000000,-10000000,-10000000,
-10000000,-10000000,-10000000,   208,-10000000,-10000000,   202,   201,   198,   197,
   196,   195,  -207,  -122,  -154,  -144,  -154,  -154,  -154,    95,
-10000000,-10000000,  -154,-10000000,-10000000,-10000000,-10000000,-10000000,-10000000,-10000000,
-10000000,-10000000,-10000000,-10000000,-10000000,  -290,-10000000,-10000000,-10000000,-10000000,
-10000000,-10000000,-10000000,-10000000,-10000000,-10000000,-10000000,-10000000,-10000000,-10000000,
-10000000,-10000000,-10000000,-10000000,  -165,-10000000,-10000000,-10000000,-10000000,-10000000,
   178,   175,   168,-10000000,-10000000,  -152,-10000000,-10000000,-10000000,-10000000,
-10000000,-10000000,-10000000,-10000000,-10000000,-10000000,-10000000,  -290,-10000000,-10000000,
-10000000,-10000000,-10000000,-10000000,  -223,-10000000,  -154,-10000000,  -154,-10000000,
-10000000,-10000000,-10000000,-10000000,-10000000,  -164,-10000000,   167,  -154,    94,
    93,    83,  -178,    82,-10000000,-10000000,  -290,-10000000,-10000000,-10000000,
-10000000,-10000000,-10000000,  -154,-10000000,     6,     6,     6,-10000000,-10000000,
-10000000,-10000000,-10000000,-10000000,-10000000,-10000000,   -72,   235,   231,   130,
-10000000,-10000000,-10000000,    69,   125,   199,  -285,   224,   236,-10000000,
-10000000,    -4,    -4,    -4,  -290,-10000000,     6,-10000000,-10000000,-10000000,
-10000000,-10000000,-10000000,-10000000,-10000000,  -140,   165,-10000000,   -64,  -245,
   267,  -151,    66,-10000000,-10000000,-10000000,-10000000,-10000000,-10000000,-10000000,
  -216,  -183,-10000000,  -233,-10000000,-10000000,-10000000,-10000000,-10000000,-10000000,
-10000000,-10000000,   163,     6,     6,     6,     6,     6,     6,     6,
     6,     6,     6,-10000000,-10000000,-10000000,   255,   162,   161,-10000000,
-10000000,  -245,-10000000,-10000000,-10000000,-10000000,-10000000,-10000000,-10000000,-10000000,
-10000000,-10000000,   193,   191,   190,   188,   185,   184,-10000000,-10000000,
-10000000,-10000000,-10000000,  -223,  -154,   130,     6,-10000000,   125,   199,
  -285,   224,   224,   236,   236,-10000000,-10000000,-10000000,-10000000,-10000000,
-10000000,     6,-10000000,    65,    63,    60,  -245,  -154,   183,   -97,
    56,   228,-10000000,-10000000,-10000000,-10000000,-10000000,-10000000,  -140,  -149,
  -290,-10000000,-10000000,-10000000,-10000000,-10000000,-10000000,-10000000,-10000000,-10000000,
-10000000,-10000000,-10000000,-10000000,-10000000,-10000000,-10000000,  -290,  -187,-10000000,
-10000000,-10000000,-10000000,   127,   182,   247,  -151,  -154,   266,-10000000,
-10000000,-10000000,-10000000,-10000000,-10000000,-10000000,    81,  -246,   246,  -199,
-10000000,-10000000,-10000000,-10000000,   243,-10000000,-10000000,-10000000,-10000000,-10000000,
  -208,  -248,   258,-10000000,   227,  -140,-10000000,-10000000,-10000000,-10000000,
-10000000,-10000000,-10000000,-10000000,-10000000,-10000000,-10000000,-10000000,    54,  -208,
  -207,   180,  -203,   207,     6,   257,  -223,  -199,  -154,-10000000,
-10000000,-10000000,-10000000,-10000000,-10000000,-10000000,-10000000,-10000000,   241,-10000000,
-10000000,   179,  -154,   204,  -252,-10000000,-10000000,-10000000,-10000000,   240,
-10000000,-10000000,   226,-10000000,  -265,-10000000 };
yytabelem yypgo[]={

     0,    14,     6,   505,   504,   503,   314,   232,   101,   502,
   230,   501,   500,   499,   312,     7,   498,   497,   496,     5,
   495,   494,   493,   492,   491,   490,   489,     8,   488,   487,
   486,     2,   485,   174,   151,   164,   168,   157,   139,   161,
   484,   121,   150,    10,   483,   482,   120,   118,   115,   311,
   481,   480,   479,   478,    99,   477,   153,   476,     1,   475,
   474,   473,   472,     0,   471,   176,   470,   175,   469,   163,
   468,   162,   467,   466,   465,   464,   463,   462,   461,   460,
   459,   458,   457,   456,   455,   454,   453,   452,   451,   450,
   449,   448,   447,   446,   445,   444,   443,   442,   441,   440,
   439,   438,   437,   436,   435,   434,   433,   432,   431,   430,
   429,   428,   427,   426,   425,   158,   114,   424,   423,   422,
   421,   420,   419,   418,   417,   416,   415,   414,   413,     4,
   412,   410,   408,   407,   406,   405,   404,   403,   402,   401,
   400,   399,   398,    13,   397,   395,   394,   393,   392,   391,
   389,   387,   386,   384,   383,   382,   381,   380,   372,   353,
   349,   344,   341,   340,   337,   336,   335,   334,   333,   332,
   331,   330,   329,     3,   328,   327,   326,   321,   319,   318,
   317,   315,   313 };
yytabelem yyr1[]={

     0,    64,    65,    65,    68,    66,    70,    66,    72,    66,
    74,    66,    76,    66,    77,    66,    78,    79,    80,    81,
    75,    73,    73,    84,    85,    87,    82,    88,    62,    29,
    89,    21,    21,    86,    86,    91,    90,    92,    90,    93,
    90,    95,    90,    97,    90,    98,    90,    19,    99,    20,
    20,    15,   100,    15,   101,    15,    63,    83,   102,   103,
   104,   105,    69,    45,    45,    45,    45,    45,    45,    45,
    45,    30,    31,    32,    32,    33,    33,    34,    34,    35,
    35,    35,    36,    36,    36,    37,    37,    37,    37,    38,
    38,    38,    38,    39,    39,    39,    40,    40,    40,    40,
    40,    40,    41,   106,    67,    67,    67,    67,   108,   107,
     1,     1,     2,     2,     2,    53,    53,    53,    53,    53,
    53,     4,     4,     4,     3,     3,     3,    27,   109,    28,
    28,    58,    58,    59,    60,    46,    46,    51,    51,    51,
    52,    52,    52,    49,    49,    49,    47,    47,    54,    48,
    50,   110,   111,   112,   114,     7,   113,   116,   116,   117,
   118,   115,   119,   115,   120,   121,   122,   123,   124,   125,
   126,   128,    10,     9,     9,     9,     9,     9,     9,   127,
   130,   130,   131,   132,   129,   133,   129,    25,    26,    26,
   134,    43,   135,   136,    43,   137,    44,   138,   139,   140,
   142,     8,   141,   145,   144,   144,   143,   146,   147,     5,
     5,   148,   149,    13,   151,   152,     6,     6,   150,   154,
   155,    14,    14,   153,   156,    11,    23,    24,    24,   157,
   158,    42,   159,   160,    94,    61,    61,   161,   162,   163,
   164,    71,   165,   166,   168,   169,    96,    57,    57,    56,
    56,    56,    12,    12,   170,   167,   171,   167,   172,   175,
   174,   174,   176,   177,   173,    55,    55,    55,   178,   179,
    22,    22,   180,   181,    16,    16,    17,   182,    18,    18 };
yytabelem yyr2[]={

     0,     2,     4,     0,     1,     7,     1,     7,     1,     7,
     1,     7,     1,     7,     1,     7,     1,     1,     1,     1,
    19,     2,     2,     1,     1,     1,    15,     1,     7,     5,
     1,     7,     1,     4,     0,     1,     7,     1,     7,     1,
     7,     1,     7,     1,     7,     1,     7,     5,     1,     9,
     1,     3,     1,     7,     1,     9,     3,     3,     1,     1,
     1,     1,    19,     2,     2,     2,     2,     2,     3,     3,
     3,     2,     2,     2,     7,     2,     7,     2,     7,     2,
     7,     7,     2,     7,     7,     2,     7,     7,     7,     2,
     5,     5,     5,     3,     2,     7,     3,     3,     3,     3,
     3,     3,     3,     1,     6,     2,     2,     2,     1,     7,
     2,     2,     3,     2,     3,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     5,     1,     9,
     1,     2,     2,     3,     3,     2,     2,     3,     5,     3,
     5,     7,     5,     3,     3,     5,     3,     3,     3,     3,
     3,     1,     1,     1,     1,    19,     4,     4,     0,     1,
     1,    11,     1,     7,     1,     1,     1,     1,     1,     1,
     1,     1,    35,     3,     3,     3,     3,     2,     3,     4,
     4,     0,     1,     1,    11,     1,     7,     5,     5,     1,
     1,     7,     1,     1,    11,     1,     7,     1,     1,     1,
     1,    19,     4,     1,     8,     0,     3,     1,     1,    13,
     5,     1,     1,    11,     1,     1,    13,     3,     3,     1,
     1,    13,     3,     3,     1,     7,     5,     5,     1,     1,
     1,    11,     1,     1,    13,     3,     1,     1,     1,     1,
     1,    19,     1,     1,     1,     1,    21,     5,     3,     3,
     3,     1,     2,     3,     1,     7,     1,     9,     4,     1,
     8,     0,     1,     1,    11,     3,     3,     3,     1,     1,
    13,     1,     1,     1,    13,     1,     5,     1,     9,     1 };
yytabelem yychk[]={

-10000000,   -64,   -65,   -66,   -67,   -69,   -71,   -73,   -75,   256,
   261,    -7,   -10,    -8,   258,   279,   -82,   -83,   259,   272,
   273,   275,   -29,   -62,   260,   -65,   -68,   -70,   -72,   -74,
   -76,   -77,  -106,  -102,  -161,   -78,  -110,  -120,  -138,   -84,
   -21,    58,   -88,    59,    59,    59,    59,    59,    59,  -107,
    -1,    -2,    -3,   -53,    -4,   -15,    -7,   -10,    -8,   -46,
   -49,   -47,   -48,   -54,   -50,    -5,    -6,   -14,   -63,   299,
   -51,   -52,   265,   266,   262,   267,   268,   270,   269,   271,
   -13,  -150,  -153,   257,   263,   264,   276,   277,   278,   -45,
   -46,   -47,   -54,   -48,   -49,    -6,   -14,   -15,   -63,   257,
   -63,   -63,   -63,   123,   -89,   -63,  -108,   299,  -100,   262,
   265,    44,    62,    60,    60,   262,   263,  -148,  -103,  -162,
   -79,  -111,  -121,  -139,   -85,   -19,   -15,   -27,   -58,   -59,
   -60,   -63,   -11,  -101,   -63,  -146,  -151,  -154,   262,    60,
   -63,   123,   123,   123,   274,   123,   -86,   -20,   -28,  -156,
   -63,   -41,   -31,   -32,   -33,   -34,   -35,   -36,   -37,   -38,
   -39,    43,    45,   126,   -15,   -40,    40,   293,   294,   295,
   296,   297,   298,   -41,   -41,  -149,  -104,  -163,   -80,  -112,
  -122,  -140,   -87,   -90,   -67,   -69,   -71,   -94,   -96,   256,
   -61,   -57,   282,   284,   -56,   285,   286,    44,    44,   -23,
   -42,    91,  -147,   124,    94,    38,   300,   301,    43,    45,
    42,    47,    37,   -39,   -39,   -39,   -31,  -152,  -155,    -2,
    61,  -116,   -65,  -113,  -115,    -1,   256,    40,  -141,  -143,
   257,   125,   -91,   -92,   -93,   -95,   -97,   -98,   283,   -12,
    -2,   287,   -56,   -99,  -109,   -24,  -157,    62,   -33,   -34,
   -35,   -36,   -36,   -37,   -37,   -38,   -38,   -38,    41,    62,
    62,  -105,  -115,  -164,   -81,  -114,  -116,  -117,  -119,  -123,
  -142,  -144,    59,    59,    59,    59,    59,    59,  -159,  -165,
   -15,   -58,   -42,   -41,   -30,   -31,   125,   125,   125,   -27,
    59,    -9,   -46,   -47,   -54,   -48,    -8,   -15,   262,   125,
    44,    -2,   257,  -158,  -118,  -124,  -145,  -160,  -166,    93,
    59,    41,  -143,   -27,  -167,    40,  -125,  -168,  -170,  -171,
   123,   -22,   291,    41,  -172,  -173,   -55,   288,   289,   290,
  -126,  -169,  -178,    41,  -174,  -176,  -127,  -129,   -25,   256,
   -43,   281,   280,   -16,   292,    40,    44,    -2,  -128,  -130,
  -131,  -133,   -26,  -134,  -135,  -180,  -179,  -175,  -177,   125,
  -129,   -44,    -1,    59,   -43,    58,   -31,    40,   -19,  -173,
   -58,  -132,  -137,  -136,  -181,    41,    59,   -58,    58,   -17,
   294,    41,   -18,    44,  -182,   294 };
yytabelem yydef[]={

    -2,    -2,     1,    -2,     4,     6,     8,    10,    12,    14,
   103,   105,   106,   107,    58,   237,    21,    22,    16,   151,
   164,   197,    23,    -2,    27,     2,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    29,    30,     0,     5,     7,     9,    11,    13,    15,   104,
   108,   110,   111,   112,   113,   114,   124,   125,   126,   115,
   116,   117,   118,   119,   120,   121,   122,   123,    51,    52,
   135,   136,   143,   144,   137,   146,   147,   149,   148,   150,
     0,   217,   222,    56,   139,     0,   211,   218,   223,    59,
    63,    64,    65,    66,    67,    68,    69,    70,   238,    17,
   152,   165,   198,    24,     0,    28,     0,    54,     0,   138,
   145,   207,   210,   214,   219,   140,   142,     0,     0,     0,
     0,     0,     0,     0,    34,    31,    50,   109,   130,   131,
   132,    -2,   134,     0,    53,     0,     0,     0,   141,   212,
    60,   239,    18,   153,   166,   199,    -2,    47,   127,     0,
    55,   208,   102,    72,    73,    75,    77,    79,    82,    85,
    89,     0,     0,     0,    93,    94,     0,    96,    97,    98,
    99,   100,   101,   215,   220,     0,     0,   158,    -2,     0,
     0,     0,     0,    33,    35,    37,    39,    41,    43,    45,
     0,     0,   235,   251,   248,   249,   250,    48,   128,   225,
   228,   229,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,    90,    91,    92,     0,     0,     0,   213,
    61,    -2,    19,   154,   158,   159,   162,   167,   200,   205,
   206,    26,     0,     0,     0,     0,     0,     0,   232,   242,
   252,   253,   247,     0,     0,   226,     0,   209,    74,    76,
    78,    80,    81,    83,    84,    86,    87,    88,    95,   216,
   221,     0,   157,     0,     0,     0,    -2,     0,     0,     0,
     0,   202,    36,    38,    40,    42,    44,    46,     0,     0,
    49,   129,   227,   230,    62,    71,   241,    20,   155,   160,
   163,   168,   173,   174,   175,   176,   177,   178,   137,   201,
   203,   233,   243,     0,     0,     0,     0,     0,     0,   231,
   161,   169,   204,   234,   244,    -2,     0,   271,     0,     0,
   170,   245,   268,   255,     0,   261,   262,   265,   266,   267,
     0,   275,     0,   257,   258,     0,   171,   181,   182,   185,
   189,   190,   192,   246,   272,   269,   259,   263,     0,    -2,
     0,     0,   187,     0,     0,     0,     0,     0,     0,   172,
   180,   183,   195,   186,   188,   191,   193,   273,     0,   260,
   264,     0,     0,     0,     0,   270,   184,   196,   194,     0,
   279,   274,   276,   277,     0,   278 };
typedef struct
#ifdef __cplusplus
	yytoktype
#endif
{ char *t_name; int t_val; } yytoktype;
#ifndef YYDEBUG
#	define YYDEBUG	1	/* allow debugging */
#endif

#if YYDEBUG

yytoktype yytoks[] =
{
	"IDENTIFIER",	257,
	"CONST",	258,
	"MODULE",	259,
	"INTERFACE",	260,
	"TYPEDEF",	261,
	"LONG",	262,
	"SHORT",	263,
	"UNSIGNED",	264,
	"DOUBLE",	265,
	"FLOAT",	266,
	"CHAR",	267,
	"WCHAR",	268,
	"OCTET",	269,
	"BOOLEAN",	270,
	"ANY",	271,
	"STRUCT",	272,
	"UNION",	273,
	"SWITCH",	274,
	"ENUM",	275,
	"SEQUENCE",	276,
	"STRING",	277,
	"WSTRING",	278,
	"EXCEPTION",	279,
	"CASE",	280,
	"DEFAULT",	281,
	"READONLY",	282,
	"ATTRIBUTE",	283,
	"THREADED",	284,
	"ONEWAY",	285,
	"IDEMPOTENT",	286,
	"VOID",	287,
	"IN",	288,
	"OUT",	289,
	"INOUT",	290,
	"RAISES",	291,
	"CONTEXT",	292,
	"INTEGER_LITERAL",	293,
	"STRING_LITERAL",	294,
	"CHARACTER_LITERAL",	295,
	"FLOATING_PT_LITERAL",	296,
	"TRUETOK",	297,
	"FALSETOK",	298,
	"SCOPE_DELIMITOR",	299,
	"LEFT_SHIFT",	300,
	"RIGHT_SHIFT",	301,
	"-unknown-",	-1	/* ends search */
};

char * yyreds[] =
{
	"-no such reduction-",
	"start : definitions",
	"definitions : definition definitions",
	"definitions : /* empty */",
	"definition : type_dcl",
	"definition : type_dcl ';'",
	"definition : const_dcl",
	"definition : const_dcl ';'",
	"definition : exception",
	"definition : exception ';'",
	"definition : interface_def",
	"definition : interface_def ';'",
	"definition : module",
	"definition : module ';'",
	"definition : error",
	"definition : error ';'",
	"module : MODULE",
	"module : MODULE IDENTIFIER",
	"module : MODULE IDENTIFIER '{'",
	"module : MODULE IDENTIFIER '{' definitions",
	"module : MODULE IDENTIFIER '{' definitions '}'",
	"interface_def : interface",
	"interface_def : forward",
	"interface : interface_header",
	"interface : interface_header '{'",
	"interface : interface_header '{' exports",
	"interface : interface_header '{' exports '}'",
	"interface_decl : INTERFACE",
	"interface_decl : INTERFACE id",
	"interface_header : interface_decl inheritance_spec",
	"inheritance_spec : ':'",
	"inheritance_spec : ':' at_least_one_scoped_name",
	"inheritance_spec : /* empty */",
	"exports : exports export",
	"exports : /* empty */",
	"export : type_dcl",
	"export : type_dcl ';'",
	"export : const_dcl",
	"export : const_dcl ';'",
	"export : exception",
	"export : exception ';'",
	"export : attribute",
	"export : attribute ';'",
	"export : operation",
	"export : operation ';'",
	"export : error",
	"export : error ';'",
	"at_least_one_scoped_name : scoped_name scoped_names",
	"scoped_names : scoped_names ','",
	"scoped_names : scoped_names ',' scoped_name",
	"scoped_names : /* empty */",
	"scoped_name : id",
	"scoped_name : SCOPE_DELIMITOR",
	"scoped_name : SCOPE_DELIMITOR id",
	"scoped_name : scoped_name SCOPE_DELIMITOR",
	"scoped_name : scoped_name SCOPE_DELIMITOR id",
	"id : IDENTIFIER",
	"forward : interface_decl",
	"const_dcl : CONST",
	"const_dcl : CONST const_type",
	"const_dcl : CONST const_type id",
	"const_dcl : CONST const_type id '='",
	"const_dcl : CONST const_type id '=' expression",
	"const_type : integer_type",
	"const_type : char_type",
	"const_type : octet_type",
	"const_type : boolean_type",
	"const_type : floating_pt_type",
	"const_type : string_type_spec",
	"const_type : wstring_type_spec",
	"const_type : scoped_name",
	"expression : const_expr",
	"const_expr : or_expr",
	"or_expr : xor_expr",
	"or_expr : or_expr '|' xor_expr",
	"xor_expr : and_expr",
	"xor_expr : xor_expr '^' and_expr",
	"and_expr : shift_expr",
	"and_expr : and_expr '&' shift_expr",
	"shift_expr : add_expr",
	"shift_expr : shift_expr LEFT_SHIFT add_expr",
	"shift_expr : shift_expr RIGHT_SHIFT add_expr",
	"add_expr : mult_expr",
	"add_expr : add_expr '+' mult_expr",
	"add_expr : add_expr '-' mult_expr",
	"mult_expr : unary_expr",
	"mult_expr : mult_expr '*' unary_expr",
	"mult_expr : mult_expr '/' unary_expr",
	"mult_expr : mult_expr '%' unary_expr",
	"unary_expr : primary_expr",
	"unary_expr : '+' primary_expr",
	"unary_expr : '-' primary_expr",
	"unary_expr : '~' primary_expr",
	"primary_expr : scoped_name",
	"primary_expr : literal",
	"primary_expr : '(' const_expr ')'",
	"literal : INTEGER_LITERAL",
	"literal : STRING_LITERAL",
	"literal : CHARACTER_LITERAL",
	"literal : FLOATING_PT_LITERAL",
	"literal : TRUETOK",
	"literal : FALSETOK",
	"positive_int_expr : const_expr",
	"type_dcl : TYPEDEF",
	"type_dcl : TYPEDEF type_declarator",
	"type_dcl : struct_type",
	"type_dcl : union_type",
	"type_dcl : enum_type",
	"type_declarator : type_spec",
	"type_declarator : type_spec at_least_one_declarator",
	"type_spec : simple_type_spec",
	"type_spec : constructed_type_spec",
	"simple_type_spec : base_type_spec",
	"simple_type_spec : template_type_spec",
	"simple_type_spec : scoped_name",
	"base_type_spec : integer_type",
	"base_type_spec : floating_pt_type",
	"base_type_spec : char_type",
	"base_type_spec : boolean_type",
	"base_type_spec : octet_type",
	"base_type_spec : any_type",
	"template_type_spec : sequence_type_spec",
	"template_type_spec : string_type_spec",
	"template_type_spec : wstring_type_spec",
	"constructed_type_spec : struct_type",
	"constructed_type_spec : union_type",
	"constructed_type_spec : enum_type",
	"at_least_one_declarator : declarator declarators",
	"declarators : declarators ','",
	"declarators : declarators ',' declarator",
	"declarators : /* empty */",
	"declarator : simple_declarator",
	"declarator : complex_declarator",
	"simple_declarator : id",
	"complex_declarator : array_declarator",
	"integer_type : signed_int",
	"integer_type : unsigned_int",
	"signed_int : LONG",
	"signed_int : LONG LONG",
	"signed_int : SHORT",
	"unsigned_int : UNSIGNED LONG",
	"unsigned_int : UNSIGNED LONG LONG",
	"unsigned_int : UNSIGNED SHORT",
	"floating_pt_type : DOUBLE",
	"floating_pt_type : FLOAT",
	"floating_pt_type : LONG DOUBLE",
	"char_type : CHAR",
	"char_type : WCHAR",
	"octet_type : OCTET",
	"boolean_type : BOOLEAN",
	"any_type : ANY",
	"struct_type : STRUCT",
	"struct_type : STRUCT id",
	"struct_type : STRUCT id '{'",
	"struct_type : STRUCT id '{' at_least_one_member",
	"struct_type : STRUCT id '{' at_least_one_member '}'",
	"at_least_one_member : member members",
	"members : members member",
	"members : /* empty */",
	"member : type_spec",
	"member : type_spec at_least_one_declarator",
	"member : type_spec at_least_one_declarator ';'",
	"member : error",
	"member : error ';'",
	"union_type : UNION",
	"union_type : UNION id",
	"union_type : UNION id SWITCH",
	"union_type : UNION id SWITCH '('",
	"union_type : UNION id SWITCH '(' switch_type_spec",
	"union_type : UNION id SWITCH '(' switch_type_spec ')'",
	"union_type : UNION id SWITCH '(' switch_type_spec ')' '{'",
	"union_type : UNION id SWITCH '(' switch_type_spec ')' '{' at_least_one_case_branch",
	"union_type : UNION id SWITCH '(' switch_type_spec ')' '{' at_least_one_case_branch '}'",
	"switch_type_spec : integer_type",
	"switch_type_spec : char_type",
	"switch_type_spec : octet_type",
	"switch_type_spec : boolean_type",
	"switch_type_spec : enum_type",
	"switch_type_spec : scoped_name",
	"at_least_one_case_branch : case_branch case_branches",
	"case_branches : case_branches case_branch",
	"case_branches : /* empty */",
	"case_branch : at_least_one_case_label",
	"case_branch : at_least_one_case_label element_spec",
	"case_branch : at_least_one_case_label element_spec ';'",
	"case_branch : error",
	"case_branch : error ';'",
	"at_least_one_case_label : case_label case_labels",
	"case_labels : case_labels case_label",
	"case_labels : /* empty */",
	"case_label : DEFAULT",
	"case_label : DEFAULT ':'",
	"case_label : CASE",
	"case_label : CASE const_expr",
	"case_label : CASE const_expr ':'",
	"element_spec : type_spec",
	"element_spec : type_spec declarator",
	"enum_type : ENUM",
	"enum_type : ENUM id",
	"enum_type : ENUM id '{'",
	"enum_type : ENUM id '{' at_least_one_enumerator",
	"enum_type : ENUM id '{' at_least_one_enumerator '}'",
	"at_least_one_enumerator : enumerator enumerators",
	"enumerators : enumerators ','",
	"enumerators : enumerators ',' enumerator",
	"enumerators : /* empty */",
	"enumerator : IDENTIFIER",
	"sequence_type_spec : seq_head ','",
	"sequence_type_spec : seq_head ',' positive_int_expr",
	"sequence_type_spec : seq_head ',' positive_int_expr '>'",
	"sequence_type_spec : seq_head '>'",
	"seq_head : SEQUENCE",
	"seq_head : SEQUENCE '<'",
	"seq_head : SEQUENCE '<' simple_type_spec",
	"string_type_spec : string_head '<'",
	"string_type_spec : string_head '<' positive_int_expr",
	"string_type_spec : string_head '<' positive_int_expr '>'",
	"string_type_spec : string_head",
	"string_head : STRING",
	"wstring_type_spec : wstring_head '<'",
	"wstring_type_spec : wstring_head '<' positive_int_expr",
	"wstring_type_spec : wstring_head '<' positive_int_expr '>'",
	"wstring_type_spec : wstring_head",
	"wstring_head : WSTRING",
	"array_declarator : id",
	"array_declarator : id at_least_one_array_dim",
	"at_least_one_array_dim : array_dim array_dims",
	"array_dims : array_dims array_dim",
	"array_dims : /* empty */",
	"array_dim : '['",
	"array_dim : '[' positive_int_expr",
	"array_dim : '[' positive_int_expr ']'",
	"attribute : opt_readonly ATTRIBUTE",
	"attribute : opt_readonly ATTRIBUTE simple_type_spec",
	"attribute : opt_readonly ATTRIBUTE simple_type_spec at_least_one_declarator",
	"opt_readonly : READONLY",
	"opt_readonly : /* empty */",
	"exception : EXCEPTION",
	"exception : EXCEPTION id",
	"exception : EXCEPTION id '{'",
	"exception : EXCEPTION id '{' members",
	"exception : EXCEPTION id '{' members '}'",
	"operation : opt_op_attribute_t op_type_spec",
	"operation : opt_op_attribute_t op_type_spec IDENTIFIER",
	"operation : opt_op_attribute_t op_type_spec IDENTIFIER parameter_list",
	"operation : opt_op_attribute_t op_type_spec IDENTIFIER parameter_list opt_raises",
	"operation : opt_op_attribute_t op_type_spec IDENTIFIER parameter_list opt_raises opt_context",
	"opt_op_attribute_t : THREADED opt_op_attribute_2",
	"opt_op_attribute_t : opt_op_attribute_2",
	"opt_op_attribute_2 : ONEWAY",
	"opt_op_attribute_2 : IDEMPOTENT",
	"opt_op_attribute_2 : /* empty */",
	"op_type_spec : simple_type_spec",
	"op_type_spec : VOID",
	"parameter_list : '('",
	"parameter_list : '(' ')'",
	"parameter_list : '('",
	"parameter_list : '(' at_least_one_parameter ')'",
	"at_least_one_parameter : parameter parameters",
	"parameters : parameters ','",
	"parameters : parameters ',' parameter",
	"parameters : /* empty */",
	"parameter : direction",
	"parameter : direction simple_type_spec",
	"parameter : direction simple_type_spec declarator",
	"direction : IN",
	"direction : OUT",
	"direction : INOUT",
	"opt_raises : RAISES",
	"opt_raises : RAISES '('",
	"opt_raises : RAISES '(' at_least_one_scoped_name ')'",
	"opt_raises : /* empty */",
	"opt_context : CONTEXT",
	"opt_context : CONTEXT '('",
	"opt_context : CONTEXT '(' at_least_one_string_literal ')'",
	"opt_context : /* empty */",
	"at_least_one_string_literal : STRING_LITERAL string_literals",
	"string_literals : string_literals ','",
	"string_literals : string_literals ',' STRING_LITERAL",
	"string_literals : /* empty */",
};
#endif /* YYDEBUG */
# line	1 "/usr/ccs/bin/yaccpar"
/*
 * Copyright (c) 1993 by Sun Microsystems, Inc.
 */

#pragma ident	"@(#)yaccpar	6.12	93/06/07 SMI"

/*
** Skeleton parser driver for yacc output
*/

/*
** yacc user known macros and defines
*/
#define YYERROR		goto yyerrlab
#define YYACCEPT	return(0)
#define YYABORT		return(1)
#define YYBACKUP( newtoken, newvalue )\
{\
	if ( yychar >= 0 || ( yyr2[ yytmp ] >> 1 ) != 1 )\
	{\
		yyerror( "syntax error - cannot backup" );\
		goto yyerrlab;\
	}\
	yychar = newtoken;\
	yystate = *yyps;\
	yylval = newvalue;\
	goto yynewstate;\
}
#define YYRECOVERING()	(!!yyerrflag)
#define YYNEW(type)	malloc(sizeof(type) * yynewmax)
#define YYCOPY(to, from, type) \
	(type *) memcpy(to, (char *) from, yynewmax * sizeof(type))
#define YYENLARGE( from, type) \
	(type *) realloc((char *) from, yynewmax * sizeof(type))
#ifndef YYDEBUG
#	define YYDEBUG	1	/* make debugging available */
#endif

/*
** user known globals
*/
int yydebug;			/* set to 1 to get debugging */

/*
** driver internal defines
*/
#define YYFLAG		(-10000000)

/*
** global variables used by the parser
*/
YYSTYPE *yypv;			/* top of value stack */
int *yyps;			/* top of state stack */

int yystate;			/* current state */
int yytmp;			/* extra var (lasts between blocks) */

int yynerrs;			/* number of errors */
int yyerrflag;			/* error recovery flag */
int yychar;			/* current input token number */



#ifdef YYNMBCHARS
#define YYLEX()		yycvtok(yylex())
/*
** yycvtok - return a token if i is a wchar_t value that exceeds 255.
**	If i<255, i itself is the token.  If i>255 but the neither 
**	of the 30th or 31st bit is on, i is already a token.
*/
#if defined(__STDC__) || defined(__cplusplus)
int yycvtok(int i)
#else
int yycvtok(i) int i;
#endif
{
	int first = 0;
	int last = YYNMBCHARS - 1;
	int mid;
	wchar_t j;

	if(i&0x60000000){/*Must convert to a token. */
		if( yymbchars[last].character < i ){
			return i;/*Giving up*/
		}
		while ((last>=first)&&(first>=0)) {/*Binary search loop*/
			mid = (first+last)/2;
			j = yymbchars[mid].character;
			if( j==i ){/*Found*/ 
				return yymbchars[mid].tvalue;
			}else if( j<i ){
				first = mid + 1;
			}else{
				last = mid -1;
			}
		}
		/*No entry in the table.*/
		return i;/* Giving up.*/
	}else{/* i is already a token. */
		return i;
	}
}
#else/*!YYNMBCHARS*/
#define YYLEX()		yylex()
#endif/*!YYNMBCHARS*/

/*
** yyparse - return 0 if worked, 1 if syntax error not recovered from
*/
#if defined(__STDC__) || defined(__cplusplus)
int yyparse(void)
#else
int yyparse()
#endif
{
	register YYSTYPE *yypvt;	/* top of value stack for $vars */

#if defined(__cplusplus) || defined(lint)
/*
	hacks to please C++ and lint - goto's inside switch should never be
	executed; yypvt is set to 0 to avoid "used before set" warning.
*/
	static int __yaccpar_lint_hack__ = 0;
	switch (__yaccpar_lint_hack__)
	{
		case 1: goto yyerrlab;
		case 2: goto yynewstate;
	}
	yypvt = 0;
#endif

	/*
	** Initialize externals - yyparse may be called more than once
	*/
	yypv = &yyv[-1];
	yyps = &yys[-1];
	yystate = 0;
	yytmp = 0;
	yynerrs = 0;
	yyerrflag = 0;
	yychar = -1;

#if YYMAXDEPTH <= 0
	if (yymaxdepth <= 0)
	{
		if ((yymaxdepth = YYEXPAND(0)) <= 0)
		{
			yyerror("yacc initialization error");
			YYABORT;
		}
	}
#endif

	{
		register YYSTYPE *yy_pv;	/* top of value stack */
		register int *yy_ps;		/* top of state stack */
		register int yy_state;		/* current state */
		register int  yy_n;		/* internal state number info */
	goto yystack;	/* moved from 6 lines above to here to please C++ */

		/*
		** get globals into registers.
		** branch to here only if YYBACKUP was called.
		*/
	yynewstate:
		yy_pv = yypv;
		yy_ps = yyps;
		yy_state = yystate;
		goto yy_newstate;

		/*
		** get globals into registers.
		** either we just started, or we just finished a reduction
		*/
	yystack:
		yy_pv = yypv;
		yy_ps = yyps;
		yy_state = yystate;

		/*
		** top of for (;;) loop while no reductions done
		*/
	yy_stack:
		/*
		** put a state and value onto the stacks
		*/
#if YYDEBUG
		/*
		** if debugging, look up token value in list of value vs.
		** name pairs.  0 and negative (-1) are special values.
		** Note: linear search is used since time is not a real
		** consideration while debugging.
		*/
		if ( yydebug )
		{
			register int yy_i;

			printf( "State %d, token ", yy_state );
			if ( yychar == 0 )
				printf( "end-of-file\n" );
			else if ( yychar < 0 )
				printf( "-none-\n" );
			else
			{
				for ( yy_i = 0; yytoks[yy_i].t_val >= 0;
					yy_i++ )
				{
					if ( yytoks[yy_i].t_val == yychar )
						break;
				}
				printf( "%s\n", yytoks[yy_i].t_name );
			}
		}
#endif /* YYDEBUG */
		if ( ++yy_ps >= &yys[ yymaxdepth ] )	/* room on stack? */
		{
			/*
			** reallocate and recover.  Note that pointers
			** have to be reset, or bad things will happen
			*/
			int yyps_index = (yy_ps - yys);
			int yypv_index = (yy_pv - yyv);
			int yypvt_index = (yypvt - yyv);
			int yynewmax;
#ifdef YYEXPAND
			yynewmax = YYEXPAND(yymaxdepth);
#else
			yynewmax = 2 * yymaxdepth;	/* double table size */
			if (yymaxdepth == YYMAXDEPTH)	/* first time growth */
			{
				char *newyys = (char *)YYNEW(int);
				char *newyyv = (char *)YYNEW(YYSTYPE);
				if (newyys != 0 && newyyv != 0)
				{
					yys = YYCOPY(newyys, yys, int);
					yyv = YYCOPY(newyyv, yyv, YYSTYPE);
				}
				else
					yynewmax = 0;	/* failed */
			}
			else				/* not first time */
			{
				yys = YYENLARGE(yys, int);
				yyv = YYENLARGE(yyv, YYSTYPE);
				if (yys == 0 || yyv == 0)
					yynewmax = 0;	/* failed */
			}
#endif
			if (yynewmax <= yymaxdepth)	/* tables not expanded */
			{
				yyerror( "yacc stack overflow" );
				YYABORT;
			}
			yymaxdepth = yynewmax;

			yy_ps = yys + yyps_index;
			yy_pv = yyv + yypv_index;
			yypvt = yyv + yypvt_index;
		}
		*yy_ps = yy_state;
		*++yy_pv = yyval;

		/*
		** we have a new state - find out what to do
		*/
	yy_newstate:
		if ( ( yy_n = yypact[ yy_state ] ) <= YYFLAG )
			goto yydefault;		/* simple state */
#if YYDEBUG
		/*
		** if debugging, need to mark whether new token grabbed
		*/
		yytmp = yychar < 0;
#endif
		if ( ( yychar < 0 ) && ( ( yychar = YYLEX() ) < 0 ) )
			yychar = 0;		/* reached EOF */
#if YYDEBUG
		if ( yydebug && yytmp )
		{
			register int yy_i;

			printf( "Received token " );
			if ( yychar == 0 )
				printf( "end-of-file\n" );
			else if ( yychar < 0 )
				printf( "-none-\n" );
			else
			{
				for ( yy_i = 0; yytoks[yy_i].t_val >= 0;
					yy_i++ )
				{
					if ( yytoks[yy_i].t_val == yychar )
						break;
				}
				printf( "%s\n", yytoks[yy_i].t_name );
			}
		}
#endif /* YYDEBUG */
		if ( ( ( yy_n += yychar ) < 0 ) || ( yy_n >= YYLAST ) )
			goto yydefault;
		if ( yychk[ yy_n = yyact[ yy_n ] ] == yychar )	/*valid shift*/
		{
			yychar = -1;
			yyval = yylval;
			yy_state = yy_n;
			if ( yyerrflag > 0 )
				yyerrflag--;
			goto yy_stack;
		}

	yydefault:
		if ( ( yy_n = yydef[ yy_state ] ) == -2 )
		{
#if YYDEBUG
			yytmp = yychar < 0;
#endif
			if ( ( yychar < 0 ) && ( ( yychar = YYLEX() ) < 0 ) )
				yychar = 0;		/* reached EOF */
#if YYDEBUG
			if ( yydebug && yytmp )
			{
				register int yy_i;

				printf( "Received token " );
				if ( yychar == 0 )
					printf( "end-of-file\n" );
				else if ( yychar < 0 )
					printf( "-none-\n" );
				else
				{
					for ( yy_i = 0;
						yytoks[yy_i].t_val >= 0;
						yy_i++ )
					{
						if ( yytoks[yy_i].t_val
							== yychar )
						{
							break;
						}
					}
					printf( "%s\n", yytoks[yy_i].t_name );
				}
			}
#endif /* YYDEBUG */
			/*
			** look through exception table
			*/
			{
				register int *yyxi = yyexca;

				while ( ( *yyxi != -1 ) ||
					( yyxi[1] != yy_state ) )
				{
					yyxi += 2;
				}
				while ( ( *(yyxi += 2) >= 0 ) &&
					( *yyxi != yychar ) )
					;
				if ( ( yy_n = yyxi[1] ) < 0 )
					YYACCEPT;
			}
		}

		/*
		** check for syntax error
		*/
		if ( yy_n == 0 )	/* have an error */
		{
			/* no worry about speed here! */
			switch ( yyerrflag )
			{
			case 0:		/* new error */
				yyerror( "syntax error" );
				goto skip_init;
			yyerrlab:
				/*
				** get globals into registers.
				** we have a user generated syntax type error
				*/
				yy_pv = yypv;
				yy_ps = yyps;
				yy_state = yystate;
			skip_init:
				yynerrs++;
				/* FALLTHRU */
			case 1:
			case 2:		/* incompletely recovered error */
					/* try again... */
				yyerrflag = 3;
				/*
				** find state where "error" is a legal
				** shift action
				*/
				while ( yy_ps >= yys )
				{
					yy_n = yypact[ *yy_ps ] + YYERRCODE;
					if ( yy_n >= 0 && yy_n < YYLAST &&
						yychk[yyact[yy_n]] == YYERRCODE)					{
						/*
						** simulate shift of "error"
						*/
						yy_state = yyact[ yy_n ];
						goto yy_stack;
					}
					/*
					** current state has no shift on
					** "error", pop stack
					*/
#if YYDEBUG
#	define _POP_ "Error recovery pops state %d, uncovers state %d\n"
					if ( yydebug )
						printf( _POP_, *yy_ps,
							yy_ps[-1] );
#	undef _POP_
#endif
					yy_ps--;
					yy_pv--;
				}
				/*
				** there is no state on stack with "error" as
				** a valid shift.  give up.
				*/
				YYABORT;
			case 3:		/* no shift yet; eat a token */
#if YYDEBUG
				/*
				** if debugging, look up token in list of
				** pairs.  0 and negative shouldn't occur,
				** but since timing doesn't matter when
				** debugging, it doesn't hurt to leave the
				** tests here.
				*/
				if ( yydebug )
				{
					register int yy_i;

					printf( "Error recovery discards " );
					if ( yychar == 0 )
						printf( "token end-of-file\n" );
					else if ( yychar < 0 )
						printf( "token -none-\n" );
					else
					{
						for ( yy_i = 0;
							yytoks[yy_i].t_val >= 0;
							yy_i++ )
						{
							if ( yytoks[yy_i].t_val
								== yychar )
							{
								break;
							}
						}
						printf( "token %s\n",
							yytoks[yy_i].t_name );
					}
				}
#endif /* YYDEBUG */
				if ( yychar == 0 )	/* reached EOF. quit */
					YYABORT;
				yychar = -1;
				goto yy_newstate;
			}
		}/* end if ( yy_n == 0 ) */
		/*
		** reduction by production yy_n
		** put stack tops, etc. so things right after switch
		*/
#if YYDEBUG
		/*
		** if debugging, print the string that is the user's
		** specification of the reduction which is just about
		** to be done.
		*/
		if ( yydebug )
			printf( "Reduce by (%d) \"%s\"\n",
				yy_n, yyreds[ yy_n ] );
#endif
		yytmp = yy_n;			/* value to switch over */
		yypvt = yy_pv;			/* $vars top of value stack */
		/*
		** Look in goto table for next state
		** Sorry about using yy_state here as temporary
		** register variable, but why not, if it works...
		** If yyr2[ yy_n ] doesn't have the low order bit
		** set, then there is no action to be done for
		** this reduction.  So, no saving & unsaving of
		** registers done.  The only difference between the
		** code just after the if and the body of the if is
		** the goto yy_stack in the body.  This way the test
		** can be made before the choice of what to do is needed.
		*/
		{
			/* length of production doubled with extra bit */
			register int yy_len = yyr2[ yy_n ];

			if ( !( yy_len & 01 ) )
			{
				yy_len >>= 1;
				yyval = ( yy_pv -= yy_len )[1];	/* $$ = $1 */
				yy_state = yypgo[ yy_n = yyr1[ yy_n ] ] +
					*( yy_ps -= yy_len ) + 1;
				if ( yy_state >= YYLAST ||
					yychk[ yy_state =
					yyact[ yy_state ] ] != -yy_n )
				{
					yy_state = yyact[ yypgo[ yy_n ] ];
				}
				goto yy_stack;
			}
			yy_len >>= 1;
			yyval = ( yy_pv -= yy_len )[1];	/* $$ = $1 */
			yy_state = yypgo[ yy_n = yyr1[ yy_n ] ] +
				*( yy_ps -= yy_len ) + 1;
			if ( yy_state >= YYLAST ||
				yychk[ yy_state = yyact[ yy_state ] ] != -yy_n )
			{
				yy_state = yyact[ yypgo[ yy_n ] ];
			}
		}
					/* save until reenter driver code */
		yystate = yy_state;
		yyps = yy_ps;
		yypv = yy_pv;
	}
	/*
	** code supplied by user is placed in this switch
	*/
	switch( yytmp )
	{
		
case 4:
# line 230 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_TypeDeclSeen);
        } break;
case 5:
# line 234 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_NoState);
        } break;
case 6:
# line 238 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_ConstDeclSeen);
        } break;
case 7:
# line 242 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_NoState);
        } break;
case 8:
# line 246 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_ExceptDeclSeen);
        } break;
case 9:
# line 250 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_NoState);
        } break;
case 10:
# line 254 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_InterfaceDeclSeen);
        } break;
case 11:
# line 258 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_NoState);
        } break;
case 12:
# line 262 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_ModuleDeclSeen);
        } break;
case 13:
# line 266 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_NoState);
        } break;
case 14:
# line 270 "idl.yy"
{
	  idl_global->err()->syntax_error(idl_global->parse_state());
	} break;
case 15:
# line 274 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_NoState);
	  yyerrok;
	} break;
case 16:
# line 281 "idl.yy"
{
	    idl_global->set_parse_state(IDL_GlobalData::PS_ModuleSeen);
	  } break;
case 17:
# line 285 "idl.yy"
{
	    UTL_ScopedName 	*n =
		new UTL_ScopedName(new Identifier(yypvt[-0].strval, 1, 0, I_FALSE), NULL);
	    AST_Module		*m = NULL;
	    UTL_Scope		*s = idl_global->scopes()->top_non_null();
	    UTL_StrList		*p = idl_global->pragmas();	

	    idl_global->set_parse_state(IDL_GlobalData::PS_ModuleIDSeen);
	    /*
	     * Make a new module and add it to the enclosing scope
	     */
	    if (s != NULL) {
	      m = idl_global->gen()->create_module(n, p);
	      (void) s->fe_add_module(m);
	    }
	    /*
	     * Push it on the stack
	     */
	    idl_global->scopes()->push(m);
	  } break;
case 18:
# line 306 "idl.yy"
{
	    idl_global->set_parse_state(IDL_GlobalData::PS_ModuleSqSeen);
	  } break;
case 19:
# line 310 "idl.yy"
{
	    idl_global->set_parse_state(IDL_GlobalData::PS_ModuleBodySeen);
	  } break;
case 20:
# line 314 "idl.yy"
{
	    idl_global->set_parse_state(IDL_GlobalData::PS_ModuleQsSeen);
	    /*
	     * Finished with this module - pop it from the scope stack
	     */
	    idl_global->scopes()->pop();
	  } break;
case 23:
# line 330 "idl.yy"
{
	  UTL_Scope     *s = idl_global->scopes()->top_non_null();
	  AST_Interface *i = NULL;
	  AST_Decl	*v = NULL;
	  UTL_StrList   *p = idl_global->pragmas();
	  AST_Decl	*d = NULL;
	  AST_Interface *fd = NULL;

	  /*
	   * Make a new interface node and add it to its enclosing scope
	   */
	  if (s != NULL && yypvt[-0].ihval != NULL) {
	    i = idl_global->gen()->create_interface(yypvt[-0].ihval->interface_name(),
						    yypvt[-0].ihval->inherits(),
						    yypvt[-0].ihval->n_inherits(),
						    p);
	    if (i != NULL &&
		(d = s->lookup_by_name(i->name(), I_FALSE)) != NULL) {
	      /*
	       * See if we're defining a forward declared interface.
	       */
	      if (d->node_type() == AST_Decl::NT_interface) {
		/*
		 * Narrow to an interface
		 */
		fd = AST_Interface::narrow_from_decl(d);
		/*
		 * Successful?
		 */
		if (fd == NULL) {
		  /*
		   * Should we give an error here?
		   */
		}
		/*
		 * If it is a forward declared interface..
		 */
		else if (!fd->is_defined()) {
		  /*
		   * Check if redefining in same scope
		   */
		  if (fd->defined_in() != s) {
		    idl_global->err()
		       ->error3(UTL_Error::EIDL_SCOPE_CONFLICT,
				i,
				fd,
				ScopeAsDecl(s));
		  }
		  /*
		   * All OK, do the redefinition
		   */
		  else {
		    fd->set_inherits(yypvt[-0].ihval->inherits());
		    fd->set_n_inherits(yypvt[-0].ihval->n_inherits());
		    /*
		     * Update place of definition
		     */
		    fd->set_imported(idl_global->imported());
		    fd->set_in_main_file(idl_global->in_main_file());
		    fd->set_line(idl_global->lineno());
		    fd->set_file_name(idl_global->filename());
		    fd->add_pragmas(p);
		    /*
		     * Use full definition node
		     */
		    delete i;
		    i = fd;
		  }
	        }
	      }
	    }
	    /*
	     * Add the interface to its definition scope
	     */
	    (void) s->fe_add_interface(i);
	  }
	  /*
	   * Push it on the scope stack
	   */
	  idl_global->scopes()->push(i);
        } break;
case 24:
# line 412 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_InterfaceSqSeen);
	} break;
case 25:
# line 416 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_InterfaceBodySeen);
	} break;
case 26:
# line 420 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_InterfaceQsSeen);
	  /*
	   * Done with this interface - pop it off the scopes stack
	   */
	  idl_global->scopes()->pop();
	} break;
case 27:
# line 431 "idl.yy"
{
	   idl_global->set_parse_state(IDL_GlobalData::PS_InterfaceSeen);
	 } break;
case 28:
# line 435 "idl.yy"
{
	   idl_global->set_parse_state(IDL_GlobalData::PS_InterfaceIDSeen);
	   yyval.idval = yypvt[-0].idval;
	 } break;
case 29:
# line 443 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_InheritSpecSeen);
	  /*
	   * Create an AST representation of the information in the header
	   * part of an interface - this representation contains a computed
	   * list of all interfaces which this interface inherits from,
	   * recursively
	   */
	  yyval.ihval = new FE_InterfaceHeader(new UTL_ScopedName(yypvt[-1].idval, NULL), yypvt[-0].nlval);
	} break;
case 30:
# line 457 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_InheritColonSeen);
        } break;
case 31:
# line 461 "idl.yy"
{
	  yyval.nlval = yypvt[-0].nlval;
	} break;
case 32:
# line 465 "idl.yy"
{
	  yyval.nlval = NULL;
	} break;
case 35:
# line 477 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_TypeDeclSeen);
        } break;
case 36:
# line 481 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_NoState);
        } break;
case 37:
# line 485 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_ConstDeclSeen);
        } break;
case 38:
# line 489 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_NoState);
        } break;
case 39:
# line 493 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_ExceptDeclSeen);
        } break;
case 40:
# line 497 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_NoState);
        } break;
case 41:
# line 501 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_AttrDeclSeen);
        } break;
case 42:
# line 505 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_NoState);
        } break;
case 43:
# line 509 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_OpDeclSeen);
        } break;
case 44:
# line 513 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_NoState);
        } break;
case 45:
# line 517 "idl.yy"
{
	  idl_global->err()->syntax_error(idl_global->parse_state());
	} break;
case 46:
# line 521 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_NoState);
	  yyerrok;
	} break;
case 47:
# line 529 "idl.yy"
{
	  yyval.nlval = new UTL_NameList(yypvt[-1].idlist, yypvt[-0].nlval);
	} break;
case 48:
# line 537 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_SNListCommaSeen);
        } break;
case 49:
# line 541 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_ScopedNameSeen);

	  if (yypvt[-3].nlval == NULL)
	    yyval.nlval = new UTL_NameList(yypvt[-0].idlist, NULL);
	  else {
	    yypvt[-3].nlval->nconc(new UTL_NameList(yypvt[-0].idlist, NULL));
	    yyval.nlval = yypvt[-3].nlval;
	  }
	} break;
case 50:
# line 552 "idl.yy"
{
	  yyval.nlval = NULL;
	} break;
case 51:
# line 559 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_SN_IDSeen);

	  yyval.idlist = new UTL_IdList(yypvt[-0].idval, NULL);
	} break;
case 52:
# line 565 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_ScopeDelimSeen);
        } break;
case 53:
# line 569 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_SN_IDSeen);

	  yyval.idlist = new UTL_IdList(new Identifier(yypvt[-2].strval, 1, 0, I_FALSE),
			      new UTL_IdList(yypvt[-0].idval, NULL));
	} break;
case 54:
# line 577 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_ScopeDelimSeen);
        } break;
case 55:
# line 581 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_SN_IDSeen);

	  yypvt[-3].idlist->nconc(new UTL_IdList(yypvt[-0].idval, NULL));
	  yyval.idlist = yypvt[-3].idlist;
	} break;
case 56:
# line 590 "idl.yy"
{
            yyval.idval = new Identifier(yypvt[-0].strval, 1, 0, I_FALSE);
        } break;
case 57:
# line 597 "idl.yy"
{
	  UTL_Scope		*s = idl_global->scopes()->top_non_null();
	  UTL_ScopedName	*n = new UTL_ScopedName(yypvt[-0].idval, NULL);
	  AST_InterfaceFwd	*f = NULL;
	  UTL_StrList		*p = idl_global->pragmas();

	  idl_global->set_parse_state(IDL_GlobalData::PS_ForwardDeclSeen);
	  /*
	   * Create a node representing a forward declaration of an
	   * interface. Store it in the enclosing scope
	   */
	  if (s != NULL) {
	    f = idl_global->gen()->create_interface_fwd(n, p);
	    (void) s->fe_add_interface_fwd(f);
	  }
	} break;
case 58:
# line 617 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_ConstSeen);
        } break;
case 59:
# line 621 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_ConstTypeSeen);
        } break;
case 60:
# line 625 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_ConstIDSeen);
        } break;
case 61:
# line 629 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_ConstAssignSeen);
        } break;
case 62:
# line 633 "idl.yy"
{
	  UTL_ScopedName	*n = new UTL_ScopedName(yypvt[-4].idval, NULL);
	  UTL_Scope		*s = idl_global->scopes()->top_non_null();
	  AST_Constant		*c = NULL;
	  UTL_StrList		*p = idl_global->pragmas();
	  AST_Decl		*v = NULL;

	  idl_global->set_parse_state(IDL_GlobalData::PS_ConstExprSeen);
	  /*
	   * Create a node representing a constant declaration. Store
	   * it in the enclosing scope
	   */
	  if (yypvt[-0].exval != NULL && s != NULL) {
	    if (yypvt[-0].exval->coerce(yypvt[-6].etval) == NULL)
	      idl_global->err()->coercion_error(yypvt[-0].exval, yypvt[-6].etval);
	    else {
	      c =
		idl_global->gen()->create_constant(yypvt[-6].etval, yypvt[-0].exval, n, p);
	      (void) s->fe_add_constant(c);
	    }
	  }
	} break;
case 68:
# line 664 "idl.yy"
{
	  yyval.etval = AST_Expression::EV_string;
	} break;
case 69:
# line 668 "idl.yy"
{
	  yyval.etval = AST_Expression::EV_wstring;
        } break;
case 70:
# line 672 "idl.yy"
{
	  UTL_Scope		*s = idl_global->scopes()->top_non_null();
	  AST_Decl		*d = NULL;
	  AST_PredefinedType	*c = NULL;
	  AST_Typedef		*t = NULL;

	  /*
	   * If the constant's type is a scoped name, it must resolve
	   * to a scalar constant type
	   */
	  if (s != NULL && (d = s->lookup_by_name(yypvt[-0].idlist, I_TRUE)) != NULL) {
	    /*
	     * Look through typedefs
	     */
	    while (d->node_type() == AST_Decl::NT_typedef) {
	      t = AST_Typedef::narrow_from_decl(d);
	      if (t == NULL)
	        break;
	      d = t->base_type();
	    }
	    if (d == NULL)
	      yyval.etval = AST_Expression::EV_any;
	    else if (d->node_type() == AST_Decl::NT_pre_defined) {
	      c = AST_PredefinedType::narrow_from_decl(d);
	      if (c != NULL) {
	         yyval.etval = idl_global->PredefinedTypeToExprType(c->pt());
	      } else {
	         yyval.etval = AST_Expression::EV_any;
	      }
	    } else
	      yyval.etval = AST_Expression::EV_any;
	  } else
	    yyval.etval = AST_Expression::EV_any;
	} break;
case 74:
# line 714 "idl.yy"
{
	  yyval.exval = idl_global->gen()->create_expr(AST_Expression::EC_or, yypvt[-2].exval, yypvt[-0].exval);
	} break;
case 76:
# line 722 "idl.yy"
{
	  yyval.exval = idl_global->gen()->create_expr(AST_Expression::EC_xor, yypvt[-2].exval, yypvt[-0].exval);
	} break;
case 78:
# line 730 "idl.yy"
{
	  yyval.exval = idl_global->gen()->create_expr(AST_Expression::EC_and, yypvt[-2].exval, yypvt[-0].exval);
	} break;
case 80:
# line 738 "idl.yy"
{
	  yyval.exval = idl_global->gen()->create_expr(AST_Expression::EC_right,yypvt[-2].exval,yypvt[-0].exval);
	} break;
case 81:
# line 742 "idl.yy"
{
	  yyval.exval = idl_global->gen()->create_expr(AST_Expression::EC_left,yypvt[-2].exval,yypvt[-0].exval);
	} break;
case 83:
# line 750 "idl.yy"
{
	  yyval.exval = idl_global->gen()->create_expr(AST_Expression::EC_add, yypvt[-2].exval, yypvt[-0].exval);
	} break;
case 84:
# line 754 "idl.yy"
{
	  yyval.exval = idl_global->gen()->create_expr(AST_Expression::EC_minus,yypvt[-2].exval,yypvt[-0].exval);
	} break;
case 86:
# line 762 "idl.yy"
{
	  yyval.exval = idl_global->gen()->create_expr(AST_Expression::EC_mul, yypvt[-2].exval, yypvt[-0].exval);
	} break;
case 87:
# line 766 "idl.yy"
{
	  yyval.exval = idl_global->gen()->create_expr(AST_Expression::EC_div, yypvt[-2].exval, yypvt[-0].exval);
	} break;
case 88:
# line 770 "idl.yy"
{
	  yyval.exval = idl_global->gen()->create_expr(AST_Expression::EC_mod, yypvt[-2].exval, yypvt[-0].exval);
	} break;
case 90:
# line 778 "idl.yy"
{
	  yyval.exval = idl_global->gen()->create_expr(AST_Expression::EC_u_plus,
					      yypvt[-0].exval,
					      NULL);
	} break;
case 91:
# line 784 "idl.yy"
{
	  yyval.exval = idl_global->gen()->create_expr(AST_Expression::EC_u_minus,
					      yypvt[-0].exval,
					      NULL);
	} break;
case 92:
# line 790 "idl.yy"
{
	  yyval.exval = idl_global->gen()->create_expr(AST_Expression::EC_bit_neg,
					      yypvt[-0].exval,
					      NULL);
	} break;
case 93:
# line 799 "idl.yy"
{
	  /*
	   * An expression which is a scoped name is not resolved now,
	   * but only when it is evaluated (such as when it is assigned
	   * as a constant value)
	   */
	  yyval.exval = idl_global->gen()->create_expr(yypvt[-0].idlist);
	} break;
case 95:
# line 809 "idl.yy"
{
	  yyval.exval = yypvt[-1].exval;
	} break;
case 96:
# line 816 "idl.yy"
{
	  yyval.exval = idl_global->gen()->create_expr(yypvt[-0].ival);
	} break;
case 97:
# line 820 "idl.yy"
{
	  yyval.exval = idl_global->gen()->create_expr(yypvt[-0].sval);
	} break;
case 98:
# line 824 "idl.yy"
{
	  yyval.exval = idl_global->gen()->create_expr(yypvt[-0].cval);
	} break;
case 99:
# line 828 "idl.yy"
{
	  yyval.exval = idl_global->gen()->create_expr(yypvt[-0].dval);
	} break;
case 100:
# line 832 "idl.yy"
{
	  yyval.exval = idl_global->gen()->create_expr((idl_bool) I_TRUE,
					    AST_Expression::EV_bool);
	} break;
case 101:
# line 837 "idl.yy"
{
	  yyval.exval = idl_global->gen()->create_expr((idl_bool) I_FALSE,
					    AST_Expression::EV_bool);
	} break;
case 102:
# line 845 "idl.yy"
{
	    yypvt[-0].exval->evaluate(AST_Expression::EK_const);
	    yyval.exval = idl_global->gen()->create_expr(yypvt[-0].exval, AST_Expression::EV_ulong);
	} break;
case 103:
# line 853 "idl.yy"
{
	    idl_global->set_parse_state(IDL_GlobalData::PS_TypedefSeen);
	  } break;
case 108:
# line 864 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_TypeSpecSeen);
        } break;
case 109:
# line 868 "idl.yy"
{
	  UTL_Scope		*s = idl_global->scopes()->top_non_null();
	  UTL_DecllistActiveIterator *l;
	  FE_Declarator		*d = NULL;
	  AST_Typedef		*t = NULL;
	  AST_Decl		*v = NULL;
	  UTL_StrList		*p = idl_global->pragmas();

	  idl_global->set_parse_state(IDL_GlobalData::PS_DeclaratorsSeen);
	  /*
	   * Create a list of type renamings. Add them to the
	   * enclosing scope
	   */
	  if (s != NULL && yypvt[-2].dcval != NULL && yypvt[-0].dlval != NULL) {
	    l = new UTL_DecllistActiveIterator(yypvt[-0].dlval);
	    for (;!(l->is_done()); l->next()) {
	      d = l->item();
	      if (d == NULL) 
		continue;
              AST_Type * tp = d->compose(yypvt[-2].dcval);
              if (tp == NULL) 
		continue;
	      t = idl_global->gen()->create_typedef(tp, d->name(), p);
	      (void) s->fe_add_typedef(t);
	    }
	    delete l;
	  }
	} break;
case 112:
# line 905 "idl.yy"
{
	  yyval.dcval = idl_global->scopes()->bottom()->lookup_primitive_type(yypvt[-0].etval);
	} break;
case 114:
# line 910 "idl.yy"
{
	  UTL_Scope	*s = idl_global->scopes()->top_non_null();
	  AST_Decl	*d = NULL;

	  if (s != NULL)
	    d = s->lookup_by_name(yypvt[-0].idlist, I_TRUE);
	  if (d == NULL)
	    idl_global->err()->lookup_error(yypvt[-0].idlist);
	  yyval.dcval = d;
	} break;
case 127:
# line 945 "idl.yy"
{
	  yyval.dlval = new UTL_DeclList(yypvt[-1].deval, yypvt[-0].dlval);
	} break;
case 128:
# line 952 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_DeclsCommaSeen);
        } break;
case 129:
# line 956 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_DeclsDeclSeen);

	  if (yypvt[-3].dlval == NULL)
	    yyval.dlval = new UTL_DeclList(yypvt[-0].deval, NULL);
	  else {
	    yypvt[-3].dlval->nconc(new UTL_DeclList(yypvt[-0].deval, NULL));
	    yyval.dlval = yypvt[-3].dlval;
	  }
	} break;
case 130:
# line 967 "idl.yy"
{
	  yyval.dlval = NULL;
	} break;
case 133:
# line 979 "idl.yy"
{
	  yyval.deval = new FE_Declarator(new UTL_ScopedName(yypvt[-0].idval, NULL),
				 FE_Declarator::FD_simple, NULL);
	} break;
case 134:
# line 987 "idl.yy"
{
	  yyval.deval = new FE_Declarator(new UTL_ScopedName(yypvt[-0].dcval->local_name(), NULL),
				 FE_Declarator::FD_complex,
				 yypvt[-0].dcval);
	} break;
case 137:
# line 1001 "idl.yy"
{
	  yyval.etval = AST_Expression::EV_long;
	} break;
case 138:
# line 1005 "idl.yy"
{
	  yyval.etval = AST_Expression::EV_longlong;
        } break;
case 139:
# line 1009 "idl.yy"
{
	  yyval.etval = AST_Expression::EV_short;
	} break;
case 140:
# line 1016 "idl.yy"
{
	  yyval.etval = AST_Expression::EV_ulong;
	} break;
case 141:
# line 1020 "idl.yy"
{
	  yyval.etval = AST_Expression::EV_ulonglong;
        } break;
case 142:
# line 1024 "idl.yy"
{
	  yyval.etval = AST_Expression::EV_ushort;
	} break;
case 143:
# line 1031 "idl.yy"
{
	  yyval.etval = AST_Expression::EV_double;
	} break;
case 144:
# line 1035 "idl.yy"
{
	  yyval.etval = AST_Expression::EV_float;
	} break;
case 145:
# line 1039 "idl.yy"
{
	  yyval.etval = AST_Expression::EV_longdouble;
        } break;
case 146:
# line 1046 "idl.yy"
{
	  yyval.etval = AST_Expression::EV_char;
	} break;
case 147:
# line 1050 "idl.yy"
{
	  yyval.etval = AST_Expression::EV_wchar;
        } break;
case 148:
# line 1057 "idl.yy"
{ 
          yyval.etval = AST_Expression::EV_octet;
	} break;
case 149:
# line 1064 "idl.yy"
{ 
	  yyval.etval = AST_Expression::EV_bool;
        } break;
case 150:
# line 1071 "idl.yy"
{
	  yyval.etval = AST_Expression::EV_any;
	} break;
case 151:
# line 1078 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_StructSeen);
        } break;
case 152:
# line 1082 "idl.yy"
{
	  UTL_Scope		*s = idl_global->scopes()->top_non_null();
	  UTL_ScopedName	*n = new UTL_ScopedName(yypvt[-0].idval, NULL);
	  AST_Structure		*d = NULL;
	  UTL_StrList		*p = idl_global->pragmas();
	  AST_Decl		*v = NULL;

	  idl_global->set_parse_state(IDL_GlobalData::PS_StructIDSeen);
	  /*
	   * Create a node representing a struct declaration. Add it
	   * to the enclosing scope
	   */
	  if (s != NULL) {
	    d = idl_global->gen()->create_structure(n, p);
	    (void) s->fe_add_structure(d);
	  }
	  /*
	   * Push the scope of the struct on the scopes stack
	   */
	  idl_global->scopes()->push(d);
	} break;
case 153:
# line 1104 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_StructSqSeen);
        } break;
case 154:
# line 1108 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_StructBodySeen);
        } break;
case 155:
# line 1112 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_StructQsSeen);
	  /*
	   * Done with this struct. Pop its scope off the scopes stack
	   */
	  if (idl_global->scopes()->top() == NULL)
	    yyval.dcval = NULL;
	  else {
	    yyval.dcval =
	      AST_Structure::narrow_from_scope(
				   idl_global->scopes()->top_non_null());
	    idl_global->scopes()->pop();
	  }
	} break;
case 159:
# line 1137 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_MemberTypeSeen);
        } break;
case 160:
# line 1141 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_MemberDeclsSeen);
        } break;
case 161:
# line 1145 "idl.yy"
{
	  UTL_Scope		*s = idl_global->scopes()->top_non_null();
	  UTL_DecllistActiveIterator *l = NULL;
	  FE_Declarator		*d = NULL;
	  AST_Field		*f = NULL;
	  UTL_StrList		*p = idl_global->pragmas();

	  idl_global->set_parse_state(IDL_GlobalData::PS_MemberDeclsCompleted);
	  /*
	   * Check for illegal recursive use of type
	   */
	  if (yypvt[-4].dcval != NULL && AST_illegal_recursive_type(yypvt[-4].dcval))
	    idl_global->err()->error1(UTL_Error::EIDL_RECURSIVE_TYPE, yypvt[-4].dcval);
	  /*
	   * Create a node representing a struct or exception member
	   * Add it to the enclosing scope
	   */
	  else if (s != NULL && yypvt[-4].dcval != NULL && yypvt[-2].dlval != NULL) {
	    l = new UTL_DecllistActiveIterator(yypvt[-2].dlval);
	    for (;!(l->is_done()); l->next()) {
	      d = l->item();
	      if (d == NULL) 
		continue;
 	      AST_Type *tp = d->compose(yypvt[-4].dcval);
	      if (tp == NULL) 
		continue;
	      f = idl_global->gen()->create_field(tp, d->name(), p);
	      (void) s->fe_add_field(f);
	    }
	    delete l;
	  }
	} break;
case 162:
# line 1178 "idl.yy"
{
	  idl_global->err()->syntax_error(idl_global->parse_state());
	} break;
case 163:
# line 1182 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_NoState);
	  yyerrok;
	} break;
case 164:
# line 1190 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_UnionSeen);
        } break;
case 165:
# line 1194 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_UnionIDSeen);
        } break;
case 166:
# line 1198 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_SwitchSeen);
        } break;
case 167:
# line 1202 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_SwitchOpenParSeen);
        } break;
case 168:
# line 1206 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_SwitchTypeSeen);
        } break;
case 169:
# line 1210 "idl.yy"
{
	  UTL_Scope		*s = idl_global->scopes()->top_non_null();
	  UTL_ScopedName	*n = new UTL_ScopedName(yypvt[-8].idval, NULL);
	  AST_Union		*u = NULL;
	  AST_Decl		*v = NULL;
	  UTL_StrList		*p = idl_global->pragmas();

	  idl_global->set_parse_state(IDL_GlobalData::PS_SwitchCloseParSeen);
	  /*
	   * Create a node representing a union. Add it to its enclosing
	   * scope
	   */
	  if (yypvt[-2].dcval != NULL && s != NULL) {
 	    AST_ConcreteType    *tp = AST_ConcreteType::narrow_from_decl(yypvt[-2].dcval);
            if (tp == NULL) {
              idl_global->err()->not_a_type(yypvt[-2].dcval);
            } else {
	      u = idl_global->gen()->create_union(tp, n, p);
	      (void) s->fe_add_union(u);
 	    }
	  }
	  /*
	   * Push the scope of the union on the scopes stack
	   */
	  idl_global->scopes()->push(u);
	} break;
case 170:
# line 1237 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_UnionSqSeen);
        } break;
case 171:
# line 1241 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_UnionBodySeen);
        } break;
case 172:
# line 1245 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_UnionQsSeen);
	  /*
	   * Done with this union. Pop its scope from the scopes stack
	   */
	  if (idl_global->scopes()->top() == NULL)
	    yyval.dcval = NULL;
	  else {
	    yyval.dcval =
	      AST_Union::narrow_from_scope(
				idl_global->scopes()->top_non_null());
	    idl_global->scopes()->pop();
	  }
	} break;
case 173:
# line 1263 "idl.yy"
{
	  yyval.dcval = idl_global->scopes()->bottom()->lookup_primitive_type(yypvt[-0].etval);
	} break;
case 174:
# line 1267 "idl.yy"
{
	  yyval.dcval = idl_global->scopes()->bottom()->lookup_primitive_type(yypvt[-0].etval);
	} break;
case 175:
# line 1271 "idl.yy"
{
	  yyval.dcval = idl_global->scopes()->bottom()->lookup_primitive_type(yypvt[-0].etval);
	} break;
case 176:
# line 1275 "idl.yy"
{
	  yyval.dcval = idl_global->scopes()->bottom()->lookup_primitive_type(yypvt[-0].etval);
	} break;
case 178:
# line 1280 "idl.yy"
{
	  UTL_Scope		*s = idl_global->scopes()->top_non_null();
	  AST_Decl		*d = NULL;
	  AST_PredefinedType	*p = NULL;
	  AST_Typedef		*t = NULL;
	  long			found = I_FALSE;

	  /*
	   * The discriminator is a scoped name. Try to resolve to
	   * one of the scalar types or to an enum. Thread through
	   * typedef's to arrive at the base type at the end of the
	   * chain
	   */
	  if (s != NULL && (d = s->lookup_by_name(yypvt[-0].idlist, I_TRUE)) != NULL) {
	    while (!found) {
	      switch (d->node_type()) {
	      case AST_Decl::NT_enum:
		yyval.dcval = d;
		found = I_TRUE;
		break;
	      case AST_Decl::NT_pre_defined:
		p = AST_PredefinedType::narrow_from_decl(d);
		if (p != NULL) {
		  switch (p->pt()) {
		  case AST_PredefinedType::PT_long:
		  case AST_PredefinedType::PT_ulong:
		  case AST_PredefinedType::PT_longlong:
		  case AST_PredefinedType::PT_ulonglong:
		  case AST_PredefinedType::PT_short:
		  case AST_PredefinedType::PT_char:
		  case AST_PredefinedType::PT_wchar:
		  case AST_PredefinedType::PT_octet:
		  case AST_PredefinedType::PT_boolean:
		    yyval.dcval = p;
		    found = I_TRUE;
		    break;
		  default:
		    yyval.dcval = NULL;
		    found = I_TRUE;
		    break;
		  }
		}
		break;
	      case AST_Decl::NT_typedef:
		t = AST_Typedef::narrow_from_decl(d);
		if (t != NULL) d = t->base_type();
		break;
	      default:
		yyval.dcval = NULL;
		found = I_TRUE;
		break;
	      }
	    }
	  } else
	    yyval.dcval = NULL;

	  if (yyval.dcval == NULL)
	    idl_global->err()->lookup_error(yypvt[-0].idlist);
	} break;
case 182:
# line 1350 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_UnionLabelSeen);
        } break;
case 183:
# line 1354 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_UnionElemSeen);
        } break;
case 184:
# line 1358 "idl.yy"
{
	  UTL_Scope		*s = idl_global->scopes()->top_non_null();
	  UTL_LabellistActiveIterator *l = NULL;
	  AST_UnionLabel	*d = NULL;
	  AST_UnionBranch	*b = NULL;
	  AST_Field		*f = yypvt[-2].ffval;

	  idl_global->set_parse_state(IDL_GlobalData::PS_UnionElemCompleted);
	  /*
	   * Create several nodes representing branches of a union.
	   * Add them to the enclosing scope (the union scope)
	   */
	  if (s != NULL && yypvt[-4].llval != NULL && yypvt[-2].ffval != NULL) {
	    l = new UTL_LabellistActiveIterator(yypvt[-4].llval);
	    for (;!(l->is_done()); l->next()) {
	      d = l->item();
	      if (d == NULL)
		continue;
	      b = idl_global->gen()->create_union_branch(d,
						      f->field_type(),
						      f->name(),
						      f->pragmas());
	      (void) s->fe_add_union_branch(b);
	    }
	    delete l;
	  }
	} break;
case 185:
# line 1386 "idl.yy"
{
	  idl_global->err()->syntax_error(idl_global->parse_state());
	} break;
case 186:
# line 1390 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_NoState);
	  yyerrok;
	} break;
case 187:
# line 1398 "idl.yy"
{
	  yyval.llval = new UTL_LabelList(yypvt[-1].ulval, yypvt[-0].llval);
	} break;
case 188:
# line 1405 "idl.yy"
{
	  if (yypvt[-1].llval == NULL)
	    yyval.llval = new UTL_LabelList(yypvt[-0].ulval, NULL);
	  else {
	    yypvt[-1].llval->nconc(new UTL_LabelList(yypvt[-0].ulval, NULL));
	    yyval.llval = yypvt[-1].llval;
	  }
	} break;
case 189:
# line 1414 "idl.yy"
{
	  yyval.llval = NULL;
	} break;
case 190:
# line 1421 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_DefaultSeen);
        } break;
case 191:
# line 1425 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_LabelColonSeen);

	  yyval.ulval = idl_global->gen()->
	            create_union_label(AST_UnionLabel::UL_default,
				       NULL);
	} break;
case 192:
# line 1433 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_CaseSeen);
        } break;
case 193:
# line 1437 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_LabelExprSeen);
        } break;
case 194:
# line 1441 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_LabelColonSeen);

	  yyval.ulval = idl_global->gen()->create_union_label(AST_UnionLabel::UL_label,
						     yypvt[-2].exval);
	} break;
case 195:
# line 1451 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_UnionElemTypeSeen);
        } break;
case 196:
# line 1455 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_UnionElemDeclSeen);
	  /*
	   * Check for illegal recursive use of type
	   */
	  if (yypvt[-2].dcval != NULL && AST_illegal_recursive_type(yypvt[-2].dcval))
	    idl_global->err()->error1(UTL_Error::EIDL_RECURSIVE_TYPE, yypvt[-2].dcval);
	  /*
	   * Create a field in a union branch
	   */
	  else if (yypvt[-2].dcval == NULL || yypvt[-0].deval == NULL)
	    yyval.ffval = NULL;
	  else {
	    AST_Type *tp = yypvt[-0].deval->compose(yypvt[-2].dcval);
	    if (tp == NULL)
	      yyval.ffval = NULL;
 	    else
	      yyval.ffval = idl_global->gen()->create_field(tp,
			        		   yypvt[-0].deval->name(),
			        		   idl_global->pragmas());
	  }
	} break;
case 197:
# line 1481 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_EnumSeen);
        } break;
case 198:
# line 1485 "idl.yy"
{
	  UTL_Scope		*s = idl_global->scopes()->top_non_null();
	  UTL_ScopedName	*n = new UTL_ScopedName(yypvt[-0].idval, NULL);
	  AST_Enum		*e = NULL;
	  AST_Decl		*v = NULL;
	  UTL_StrList		*p = idl_global->pragmas();

	  idl_global->set_parse_state(IDL_GlobalData::PS_EnumIDSeen);
	  /*
	   * Create a node representing an enum and add it to its
	   * enclosing scope
	   */
	  if (s != NULL) {
	    e = idl_global->gen()->create_enum(n, p);
	    /*
	     * Add it to its defining scope
	     */
	    (void) s->fe_add_enum(e);
	  }
	  /*
	   * Push the enum scope on the scopes stack
	   */
	  idl_global->scopes()->push(e);
	} break;
case 199:
# line 1510 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_EnumSqSeen);
        } break;
case 200:
# line 1514 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_EnumBodySeen);
        } break;
case 201:
# line 1518 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_EnumQsSeen);
	  /*
	   * Done with this enum. Pop its scope from the scopes stack
	   */
	  if (idl_global->scopes()->top() == NULL)
	    yyval.dcval = NULL;
	  else {
	    yyval.dcval = AST_Enum::narrow_from_scope(idl_global->scopes()->top_non_null());
	    idl_global->scopes()->pop();
	  }
	} break;
case 203:
# line 1537 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_EnumCommaSeen);
        } break;
case 206:
# line 1546 "idl.yy"
{
	  UTL_Scope		*s = idl_global->scopes()->top_non_null();
	  UTL_ScopedName	*n =
		new UTL_ScopedName(new Identifier(yypvt[-0].strval, 1, 0, I_FALSE), NULL);
	  AST_EnumVal		*e = NULL;
	  AST_Enum		*c = NULL;
	  UTL_StrList		*p = idl_global->pragmas();

	  /*
	   * Create a node representing one enumerator in an enum
	   * Add it to the enclosing scope (the enum scope)
	   */
	  if (s != NULL && s->scope_node_type() == AST_Decl::NT_enum) {
	    c = AST_Enum::narrow_from_scope(s);
	    if (c != NULL) 
	      e = idl_global->gen()->create_enum_val(c->next_enum_val(), n, p);
	    (void) s->fe_add_enum_val(e);
	  }
	} break;
case 207:
# line 1570 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_SequenceCommaSeen);
        } break;
case 208:
# line 1574 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_SequenceExprSeen);
        } break;
case 209:
# line 1578 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_SequenceQsSeen);
	  /*
	   * Remove sequence marker from scopes stack
	   */
	  if (idl_global->scopes()->top() == NULL)
	    idl_global->scopes()->pop();
	  /*
	   * Create a node representing a sequence
	   */
	  if (yypvt[-2].exval == NULL || yypvt[-2].exval->coerce(AST_Expression::EV_ulong) == NULL) {
	    idl_global->err()->coercion_error(yypvt[-2].exval, AST_Expression::EV_ulong);
	    yyval.dcval = NULL;
	  } else if (yypvt[-5].dcval == NULL) {
	    yyval.dcval = NULL;
	  } else {
 	    AST_Type *tp = AST_Type::narrow_from_decl(yypvt[-5].dcval); 
	    if (tp == NULL) 
	      yyval.dcval = NULL;
	    else {
	      yyval.dcval = idl_global->gen()->create_sequence(yypvt[-2].exval, tp);
	      /*
	       * Add this AST_Sequence to the types defined in the global scope
	       */
	      (void) idl_global->root()
		        ->fe_add_sequence(AST_Sequence::narrow_from_decl(yyval.dcval));
 	    }
	  }
	} break;
case 210:
# line 1609 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_SequenceQsSeen);
	  /*
	   * Remove sequence marker from scopes stack
	   */
	  if (idl_global->scopes()->top() == NULL)
	    idl_global->scopes()->pop();
	  /*
	   * Create a node representing a sequence
	   */
	  if (yypvt[-1].dcval == NULL)
	    yyval.dcval = NULL;
	  else {
	    AST_Type *tp = AST_Type::narrow_from_decl(yypvt[-1].dcval);
	    if (tp == NULL)
	      yyval.dcval = NULL;
            else {
	      yyval.dcval =
	        idl_global->gen()->create_sequence(
		  	     idl_global->gen()->create_expr((unsigned long) 0),
			     tp);
	      /*
	       * Add this AST_Sequence to the types defined in the global scope
	       */
	      (void) idl_global->root()
		        ->fe_add_sequence(AST_Sequence::narrow_from_decl(yyval.dcval));
	    }
	  }
	} break;
case 211:
# line 1642 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_SequenceSeen);
	  /*
	   * Push a sequence marker on scopes stack
	   */
	  idl_global->scopes()->push(NULL);
	} break;
case 212:
# line 1650 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_SequenceSqSeen);
        } break;
case 213:
# line 1654 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_SequenceTypeSeen);
	  yyval.dcval = yypvt[-0].dcval;
        } break;
case 214:
# line 1663 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_StringSqSeen);
        } break;
case 215:
# line 1667 "idl.yy"
{
	   idl_global->set_parse_state(IDL_GlobalData::PS_StringExprSeen);
        } break;
case 216:
# line 1671 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_StringQsSeen);
	  /*
	   * Create a node representing a string
	   */
	  if (yypvt[-2].exval == NULL || yypvt[-2].exval->coerce(AST_Expression::EV_ulong) == NULL) {
	    idl_global->err()->coercion_error(yypvt[-2].exval, AST_Expression::EV_ulong);
	    yyval.dcval = NULL;
	  } else {
	    yyval.dcval = idl_global->gen()->create_string(yypvt[-2].exval);
	    /*
	     * Add this AST_String to the types defined in the global scope
	     */
	    (void) idl_global->root()
		      ->fe_add_string(AST_String::narrow_from_decl(yyval.dcval));
	  }
	} break;
case 217:
# line 1689 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_StringCompleted);
	  /*
	   * Create a node representing a string
	   */
	  yyval.dcval =
	    idl_global->gen()->create_string(
			 idl_global->gen()->create_expr((unsigned long) 0));
	  /*
	   * Add this AST_String to the types defined in the global scope
	   */
	  (void) idl_global->root()
                    ->fe_add_string(AST_String::narrow_from_decl(yyval.dcval));
	} break;
case 218:
# line 1707 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_StringSeen);
        } break;
case 219:
# line 1715 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_StringSqSeen);
        } break;
case 220:
# line 1719 "idl.yy"
{
	   idl_global->set_parse_state(IDL_GlobalData::PS_StringExprSeen);
        } break;
case 221:
# line 1723 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_StringQsSeen);
	  /*
	   * Create a node representing a string
	   */
	  if (yypvt[-2].exval == NULL || yypvt[-2].exval->coerce(AST_Expression::EV_ulong) == NULL) {
	    idl_global->err()->coercion_error(yypvt[-2].exval, AST_Expression::EV_ulong);
	    yyval.dcval = NULL;
	  } else {
	    yyval.dcval = idl_global->gen()->create_wstring(yypvt[-2].exval);
	    /*
	     * Add this AST_String to the types defined in the global scope
	     */
	    (void) idl_global->root()
		      ->fe_add_string(AST_String::narrow_from_decl(yyval.dcval));
	  }
	} break;
case 222:
# line 1741 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_StringCompleted);
	  /*
	   * Create a node representing a string
	   */
	  yyval.dcval =
	    idl_global->gen()->create_wstring(
			 idl_global->gen()->create_expr((unsigned long) 0));
	  /*
	   * Add this AST_String to the types defined in the global scope
	   */
	  (void) idl_global->root()
                    ->fe_add_string(AST_String::narrow_from_decl(yyval.dcval));
	} break;
case 223:
# line 1759 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_StringSeen);
        } break;
case 224:
# line 1766 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_ArrayIDSeen);
        } break;
case 225:
# line 1770 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_ArrayCompleted);
	  /*
	   * Create a node representing an array
	   */
	  if (yypvt[-0].elval != NULL) {
	     yyval.dcval = idl_global->gen()->create_array(new UTL_ScopedName(yypvt[-2].idval, NULL),
						  yypvt[-0].elval->length(), yypvt[-0].elval);
	  }
	} break;
case 226:
# line 1784 "idl.yy"
{
	  yyval.elval = new UTL_ExprList(yypvt[-1].exval, yypvt[-0].elval);
	} break;
case 227:
# line 1791 "idl.yy"
{
	  if (yypvt[-1].elval == NULL)
	    yyval.elval = new UTL_ExprList(yypvt[-0].exval, NULL);
	  else {
	    yypvt[-1].elval->nconc(new UTL_ExprList(yypvt[-0].exval, NULL));
	    yyval.elval = yypvt[-1].elval;
	  }
	} break;
case 228:
# line 1800 "idl.yy"
{
	  yyval.elval = NULL;
	} break;
case 229:
# line 1807 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_DimSqSeen);
        } break;
case 230:
# line 1811 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_DimExprSeen);
        } break;
case 231:
# line 1815 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_DimQsSeen);
	  /*
	   * Array dimensions are expressions which must be coerced to
	   * positive integers
	   */
	  if (yypvt[-2].exval == NULL || yypvt[-2].exval->coerce(AST_Expression::EV_ulong) == NULL) {
	    idl_global->err()->coercion_error(yypvt[-2].exval, AST_Expression::EV_ulong);
	    yyval.exval = NULL;
	  } else
	    yyval.exval = yypvt[-2].exval;
	} break;
case 232:
# line 1832 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_AttrSeen);
        } break;
case 233:
# line 1836 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_AttrTypeSeen);
        } break;
case 234:
# line 1840 "idl.yy"
{
	  UTL_Scope		*s = idl_global->scopes()->top_non_null();
	  UTL_DecllistActiveIterator *l = NULL;
	  AST_Attribute		*a = NULL;
	  FE_Declarator		*d = NULL;
	  UTL_StrList		*p = idl_global->pragmas();

	  idl_global->set_parse_state(IDL_GlobalData::PS_AttrCompleted);
	  /*
	   * Create nodes representing attributes and add them to the
	   * enclosing scope
	   */
	  if (s != NULL && yypvt[-2].dcval != NULL && yypvt[-0].dlval != NULL) {
	    l = new UTL_DecllistActiveIterator(yypvt[-0].dlval);
	    for (;!(l->is_done()); l->next()) {
	      d = l->item();
	      if (d == NULL)
		continue;
	      AST_Type *tp = d->compose(yypvt[-2].dcval);
 	      if (tp == NULL)
	 	continue;
	      a = idl_global->gen()->create_attribute(yypvt[-5].bval, tp, d->name(), p);
	      /*
	       * Add one attribute to the enclosing scope
	       */
	      (void) s->fe_add_attribute(a);
	    }
	    delete l;
	  }
	} break;
case 235:
# line 1874 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_AttrROSeen);
	  yyval.bval = I_TRUE;
	} break;
case 236:
# line 1879 "idl.yy"
{
	  yyval.bval = I_FALSE;
	} break;
case 237:
# line 1886 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_ExceptSeen);
	} break;
case 238:
# line 1890 "idl.yy"
{
	  UTL_Scope		*s = idl_global->scopes()->top_non_null();
	  UTL_ScopedName	*n = new UTL_ScopedName(yypvt[-0].idval, NULL);
	  AST_Exception		*e = NULL;
	  UTL_StrList		*p = idl_global->pragmas();
	  AST_Decl		*v = NULL;

	  idl_global->set_parse_state(IDL_GlobalData::PS_ExceptIDSeen);
	  /*
	   * Create a node representing an exception and add it to
	   * the enclosing scope
	   */
	  if (s != NULL) {
	    e = idl_global->gen()->create_exception(n, p);
	    (void) s->fe_add_exception(e);
	  }
	  /*
	   * Push the exception scope on the scope stack
	   */
	  idl_global->scopes()->push(e);
	} break;
case 239:
# line 1912 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_ExceptSqSeen);
        } break;
case 240:
# line 1916 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_ExceptBodySeen);
        } break;
case 241:
# line 1920 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_ExceptQsSeen);
	  /*
	   * Done with this exception. Pop its scope from the scope stack
	   */
	  idl_global->scopes()->pop();
	} break;
case 242:
# line 1932 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_OpTypeSeen);
        } break;
case 243:
# line 1936 "idl.yy"
{
	  UTL_Scope		*s = idl_global->scopes()->top_non_null();
	  UTL_ScopedName	*n =
		new UTL_ScopedName(new Identifier(yypvt[-0].strval, 1, 0, I_FALSE), NULL);
	  AST_Operation		*o = NULL;
	  UTL_StrList		*p = idl_global->pragmas();
	  int is_threaded = (yypvt[-3].ival >= 16);
	  if (is_threaded) yypvt[-3].ival = yypvt[-3].ival - 16;

	  idl_global->set_parse_state(IDL_GlobalData::PS_OpIDSeen);
	  /*
	   * Create a node representing an operation on an interface
	   * and add it to its enclosing scope
	   */
	  if (s != NULL && yypvt[-2].dcval != NULL) {
	    AST_Type *tp = AST_Type::narrow_from_decl(yypvt[-2].dcval);
            if (tp == NULL) {
              idl_global->err()->not_a_type(yypvt[-2].dcval);
            } else if (tp->node_type() == AST_Decl::NT_except) {
              idl_global->err()->not_a_type(yypvt[-2].dcval);
            } else {
	      o = idl_global->gen()->create_operation(tp, (AST_Operation::Flags)yypvt[-3].ival, n, p);
	      o->set_threaded(is_threaded);
	      (void) s->fe_add_operation(o);
	    }
	  }
	  /*
	   * Push the operation scope onto the scopes stack
	   */
	  idl_global->scopes()->push(o);
	} break;
case 244:
# line 1968 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_OpParsCompleted);
        } break;
case 245:
# line 1972 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_OpRaiseCompleted);
        } break;
case 246:
# line 1976 "idl.yy"
{
	  UTL_Scope		*s = idl_global->scopes()->top_non_null();
	  AST_Operation		*o = NULL;

	  idl_global->set_parse_state(IDL_GlobalData::PS_OpCompleted);
	  /*
	   * Add exceptions and context to the operation
	   */
	  if (s != NULL && s->scope_node_type() == AST_Decl::NT_op) {
	    o = AST_Operation::narrow_from_scope(s);

	    if (yypvt[-2].nlval != NULL && o != NULL)
	      (void) o->fe_add_exceptions(yypvt[-2].nlval);
	    if (yypvt[-0].slval != NULL)
	      (void) o->fe_add_context(yypvt[-0].slval);
	  }
	  /*
	   * Done with this operation. Pop its scope from the scopes stack
	   */
	  idl_global->scopes()->pop();
	} break;
case 247:
# line 2002 "idl.yy"
{
	  yyval.ival = 16 + yypvt[-0].ofval;
	} break;
case 248:
# line 2006 "idl.yy"
{
	  yyval.ival = yypvt[-0].ofval;
	} break;
case 249:
# line 2013 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_OpAttrSeen);
	  yyval.ofval = AST_Operation::OP_oneway;
	} break;
case 250:
# line 2018 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_OpAttrSeen);
	  yyval.ofval = AST_Operation::OP_idempotent;
	} break;
case 251:
# line 2023 "idl.yy"
{
	  yyval.ofval = AST_Operation::OP_noflags;
	} break;
case 253:
# line 2031 "idl.yy"
{
	  yyval.dcval =
	    idl_global->scopes()->bottom()
	       ->lookup_primitive_type(AST_Expression::EV_void);
	} break;
case 254:
# line 2040 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_OpSqSeen);
        } break;
case 255:
# line 2044 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_OpQsSeen);
        } break;
case 256:
# line 2048 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_OpSqSeen);
        } break;
case 257:
# line 2053 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_OpQsSeen);
        } break;
case 259:
# line 2063 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_OpParCommaSeen);
        } break;
case 262:
# line 2072 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_OpParDirSeen);
        } break;
case 263:
# line 2076 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_OpParTypeSeen);
        } break;
case 264:
# line 2080 "idl.yy"
{
	  UTL_Scope		*s = idl_global->scopes()->top_non_null();
	  AST_Argument		*a = NULL;
	  UTL_StrList		*p = idl_global->pragmas();

	  idl_global->set_parse_state(IDL_GlobalData::PS_OpParDeclSeen);
	  /*
	   * Create a node representing an argument to an operation
	   * Add it to the enclosing scope (the operation scope)
	   */
	  if (yypvt[-2].dcval != NULL && yypvt[-0].deval != NULL && s != NULL) {
	    AST_Type *tp = yypvt[-0].deval->compose(yypvt[-2].dcval);
	    if (tp != NULL) {
	      a = idl_global->gen()->create_argument(yypvt[-4].dival, tp, yypvt[-0].deval->name(), p);
	      (void) s->fe_add_argument(a);
	    }
	  }
	} break;
case 265:
# line 2102 "idl.yy"
{
	  yyval.dival = AST_Argument::dir_IN;
	} break;
case 266:
# line 2106 "idl.yy"
{
	  yyval.dival = AST_Argument::dir_OUT;
	} break;
case 267:
# line 2110 "idl.yy"
{
	  yyval.dival = AST_Argument::dir_INOUT;
	} break;
case 268:
# line 2117 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_OpRaiseSeen);
        } break;
case 269:
# line 2121 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_OpRaiseSqSeen);
        } break;
case 270:
# line 2126 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_OpRaiseQsSeen);
	  yyval.nlval = yypvt[-1].nlval;
	} break;
case 271:
# line 2131 "idl.yy"
{
	  yyval.nlval = NULL;
	} break;
case 272:
# line 2138 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_OpContextSeen);
        } break;
case 273:
# line 2142 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_OpContextSqSeen);
        } break;
case 274:
# line 2147 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_OpContextQsSeen);
	  yyval.slval = yypvt[-1].slval;
	} break;
case 275:
# line 2152 "idl.yy"
{
	  yyval.slval = NULL;
	} break;
case 276:
# line 2159 "idl.yy"
{
	  yyval.slval = new UTL_StrList(yypvt[-1].sval, yypvt[-0].slval);
	} break;
case 277:
# line 2167 "idl.yy"
{
	  idl_global->set_parse_state(IDL_GlobalData::PS_OpContextCommaSeen);
        } break;
case 278:
# line 2171 "idl.yy"
{
	  if (yypvt[-3].slval == NULL)
	    yyval.slval = new UTL_StrList(yypvt[-0].sval, NULL);
	  else {
	    yypvt[-3].slval->nconc(new UTL_StrList(yypvt[-0].sval, NULL));
	    yyval.slval = yypvt[-3].slval;
	  }
	} break;
case 279:
# line 2180 "idl.yy"
{
	  yyval.slval = NULL;
	} break;
# line	532 "/usr/ccs/bin/yaccpar"
	}
	goto yystack;		/* reset registers in driver code */
}

