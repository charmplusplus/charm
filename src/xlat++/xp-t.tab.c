
/*  A Bison parser, made from xp-t.bison
 by  GNU Bison version 1.25
  */

#define YYBISON 1  /* Identify Bison output.  */

#define	CHARE	258
#define	BRANCHED	259
#define	MESSAGE	260
#define	HANDLE	261
#define	GROUP	262
#define	ENTRY	263
#define	DOUBLEARROW	264
#define	ALL_NODES	265
#define	LOCAL	266
#define	ACCUMULATOR	267
#define	MONOTONIC	268
#define	READONLY	269
#define	WRITEONCE	270
#define	NEWCHARE	271
#define	NEWGROUP	272
#define	AUTO	273
#define	DOUBLE	274
#define	INT	275
#define	STRUCT	276
#define	BREAK	277
#define	ELSE	278
#define	LONG	279
#define	SWITCH	280
#define	CASE	281
#define	ENUM	282
#define	REGISTER	283
#define	TYPEDEF	284
#define	CHAR	285
#define	EXTERN	286
#define	RETURN	287
#define	UNION	288
#define	CONST	289
#define	FLOAT	290
#define	SHORT	291
#define	UNSIGNED	292
#define	WCHAR_TOKEN	293
#define	__WCHAR_TOKEN	294
#define	PTRDIFF_TOKEN	295
#define	CONTINUE	296
#define	FOR	297
#define	SIGNED	298
#define	VOID	299
#define	DEFAULT	300
#define	GOTO	301
#define	SIZEOF	302
#define	VOLATILE	303
#define	DO	304
#define	IF	305
#define	STATIC	306
#define	WHILE	307
#define	NEW	308
#define	DELETE	309
#define	THIS	310
#define	OPERATOR	311
#define	CLASS	312
#define	PUBLIC	313
#define	PROTECTED	314
#define	PRIVATE	315
#define	VIRTUAL	316
#define	FRIEND	317
#define	INLINE	318
#define	UNDERSCORE_INLINE	319
#define	OVERLOAD	320
#define	IDENTIFIER	321
#define	STRINGliteral	322
#define	FLOATINGconstant	323
#define	INTEGERconstant	324
#define	CHARACTERconstant	325
#define	OCTALconstant	326
#define	HEXconstant	327
#define	TYPEDEFname	328
#define	ARROW	329
#define	ICR	330
#define	DECR	331
#define	LSHIFT	332
#define	RSHIFT	333
#define	LE	334
#define	GE	335
#define	EQ	336
#define	NE	337
#define	ANDAND	338
#define	OROR	339
#define	ELLIPSIS	340
#define	CLCL	341
#define	DOTstar	342
#define	ARROWstar	343
#define	MULTassign	344
#define	DIVassign	345
#define	MODassign	346
#define	PLUSassign	347
#define	MINUSassign	348
#define	LSassign	349
#define	RSassign	350
#define	ANDassign	351
#define	ERassign	352
#define	ORassign	353

#line 1 "xp-t.bison"


/*  Copyright (C) 1989-1991 James A. Roskind, All rights reserved. 
    This grammar was developed  and  written  by  James  A.  Roskind. 
    Copying  of  this  grammar  description, as a whole, is permitted 
    providing this notice is intact and applicable  in  all  complete 
    copies.   Translations as a whole to other parser generator input 
    languages  (or  grammar  description  languages)   is   permitted 
    provided  that  this  notice is intact and applicable in all such 
    copies,  along  with  a  disclaimer  that  the  contents  are   a 
    translation.   The reproduction of derived text, such as modified 
    versions of this grammar, or the output of parser generators,  is 
    permitted,  provided  the  resulting  work includes the copyright 
    notice "Portions Copyright (c)  1989,  1990  James  A.  Roskind". 
    Derived products, such as compilers, translators, browsers, etc., 
    that  use  this  grammar,  must also provide the notice "Portions 
    Copyright  (c)  1989,  1990  James  A.  Roskind"  in   a   manner 
    appropriate  to  the  utility,  and in keeping with copyright law 
    (e.g.: EITHER displayed when first invoked/executed; OR displayed 
    continuously on display terminal; OR via placement in the  object 
    code  in  form  readable in a printout, with or near the title of 
    the work, or at the end of the file).  No royalties, licenses  or 
    commissions  of  any  kind are required to copy this grammar, its 
    translations, or derivative products, when the copies are made in 
    compliance with this notice. Persons or corporations that do make 
    copies in compliance with this notice may charge  whatever  price 
    is  agreeable  to  a  buyer, for such copies or derivative works. 
    THIS GRAMMAR IS PROVIDED ``AS IS'' AND  WITHOUT  ANY  EXPRESS  OR 
    IMPLIED  WARRANTIES,  INCLUDING,  WITHOUT LIMITATION, THE IMPLIED 
    WARRANTIES  OF  MERCHANTABILITY  AND  FITNESS  FOR  A  PARTICULAR 
    PURPOSE.

    James A. Roskind
    Independent Consultant
    516 Latania Palm Drive
    Indialantic FL, 32903
    (407)729-4348
    jar@hq.ileaf.com


    ---end of copyright notice---
*/

/*

1) template support: Not  done:  pending  syntax  specification  from 
ANSI.  (This looks like a major effort, as ANSI has decided to extend 
the  "TYPEDEFname"-feedback-to-the-lexer-hack  to  support   template 
names as a new kind of terminal token.)

2)  exception  handling:  Not done: pending syntax specification from 
ANSI (but it doesn't look hard)

done: 3) Support nested types, including identifier::name,  where  we 
realize  that  identifier was a hidden type.  Force the lexer to keep 
pace in this situation.   This  will  require  an  extension  of  the 
yacc-lex feedback loop.

done: 4) Support nested types even when derivations are used in class 
definitions.

done: 6) Allow declaration specifiers to be left out of  declarations 
at file and structure scope so that operator conversion functions can 
be  declared and/or defined.  Note that checking to see that it was a 
function type that does not require declaration_specifiers is  now  a 
constraint  check,  and  not  a  syntax  issue.  Within function body 
scopes, declaration specifiers are required, and this is critical  to 
distinguishing expressions.

*/

#line 151 "xp-t.bison"

/*************** Includes and Defines *****************************/
#define YYDEBUG_LEXER_TEXT (yylval) /* our lexer loads this up each time.
				     We are telling the graphical debugger
				     where to find the spelling of the 
				     tokens.*/
#define YYDEBUG 1        /* get the pretty debugging code to compile*/
#define YYSTYPE  char *  /* interface with flex: should be in header file */



#include "xp-lexer.h" 

StackStruct *StackTop=NULL ;
StackStruct *GlobalStack=NULL ;

AggState *PermanentAggTable[MAXAGGS] ;  
int PermanentAggTableSize = 0 ;
/* this table is only added to, never deleted from. It stores the objects
   defined in an aggregate type. 					  */
					

/* These tables just hold lists of Chares,BOCs,Accs ... */
ChareInfo * ChareTable[MAXCHARES] ;
ChareInfo * BOCTable[MAXCHARES] ;
MsgStruct MessageTable[MAXMSGS] ;
AccStruct * AccTable[MAXACCS] ;
AccStruct * MonoTable[MAXACCS] ;
char * DTableTable[MAXDTABLES] ;
char * ReadTable[MAXREADS] ;
char * ReadMsgTable[MAXREADS] ;
FunctionStruct FunctionTable[MAX_FUNCTIONS] ;
int charecount = -1 ;
int boccount = -1 ;
int TotalEntries = 0 ;
int TotalMsgs = 0 ;
int TotalAccs = 0 ;
int TotalMonos = 0 ;
int TotalDTables = 0 ;
int TotalReadMsgs = 0 ;
int TotalReads = 0 ;
int TotalFns = 0 ;

/* This table is used to distinguish between typedefs and idents */
SymEntry SymTable[MAXSYMBOLS] ;
int TotalSyms = 0 ;

/* the following three tables store all handle identifiers and their types */
HandleEntry ChareHandleTable[MAXIDENTS] ;
HandleEntry BOCHandleTable[MAXIDENTS] ;
HandleEntry AccHandleTable[MAXIDENTS] ;
HandleEntry MonoHandleTable[MAXIDENTS] ;
HandleEntry WrOnHandleTable[MAXIDENTS] ;
int ChareHandleTableSize = 0 ;
int BOCHandleTableSize = 0 ;
int AccHandleTableSize = 0 ;
int MonoHandleTableSize = 0 ;
int WrOnHandleTableSize = 0 ;

/* char modname[MAX_NAME_LENGTH] ;  */

char OutBuf[MAX_OUTBUF_SIZE] ;

int CurrentLine=1 ;
int CurrentScope = 0 ; 	/* 1 means file scope, > 1 means inside a block */
char CurrentFileName[MAX_NAME_LENGTH] = {'\0'} ;
int CurrentAccess = -1, CurrentAggType = -1, CurrentStorage = -1 ;
int CurrentCharmType = -1 ;
int CurrentCharmNameIndex = -1 ;
char CurrentTypedef[MAX_NAME_LENGTH] = {'\0'} ;
char CurrentDeclType[MAX_NAME_LENGTH] = {'\0'} ;
char CurrentAggName[MAX_NAME_LENGTH] = {'\0'} ;
char CurrentChare[MAX_NAME_LENGTH] = {'\0'} ; 
char CurrentEP[MAX_NAME_LENGTH] = {'\0'} ;
char CurrentFn[MAX_NAME_LENGTH] = {'\0'} ;
char CurrentMsgParm[MAX_NAME_LENGTH] = {'\0'} ;
char CurrentSharedHandle[MAX_NAME_LENGTH] = {'_','C','K','_','N','O','T','A','C','C','H','A','N','D','L','E'} ;
AccStruct *CurrentAcc ;
ChareInfo *CurrentCharePtr ;
char CurrentAssOp[5] ;
char CurrentAsterisk[MAX_NAME_LENGTH] ;
char *EpMsg=NULL ;
char SendEP[MAX_NAME_LENGTH] ;
char SendChare[MAX_NAME_LENGTH] ;
char SendPe[MAX_NAME_LENGTH] ;
char LastChare[MAX_NAME_LENGTH] ;
char LastEP[MAX_NAME_LENGTH] ;
char LastArg[MAX_NAME_LENGTH] ;
char *ParentArray[MAX_PARENTS] ;
int SendType = -1 ;
int main_argc_argv = FALSE ;
int foundargs = FALSE ;
int numparents = 0 ;
int SendMsgBranchPoss = FALSE ;
int FoundHandle = -1 ;
int FilledAccMsg = FALSE ;
int FoundConstructorBody = FALSE ;
int IsMonoCall = FALSE ;
int IsAccCall = FALSE ;
int FoundParms = FALSE ;
int FoundLocalBranch = FALSE ;
int AddedScope = 0 ;
int FoundGlobalScope = 0 ;
int FoundTable = FALSE ;
int FoundVarSize = FALSE ;
int FoundReadOnly = FALSE ;
int StructScope = FALSE ;
int AccFnScope = -1 ;
int FoundAccFnDef = FALSE ;
int MakeGraph = FALSE ;
int InsideChareCode = FALSE ;
int NewOpType = -1 ;
char *NewType ;
int FoundDeclarator=FALSE ; 
int CurrentFnIsInline=FALSE ;
int FoundChareEPPair=0 ;

int ErrVal = FALSE ;

char CoreName[MAX_NAME_LENGTH] ;
int shouldprint=1 ;
char prevtoken[MAX_TOKEN_SIZE] ;
FILE *outfile, *headerfile, *graphfile ;

extern int FoundLBrace, FoundRBrace ;

extern char *CheckSendError() ;
extern char *Mystrstr() ;
extern EP *SearchEPList() ;
extern char *Concat2() ;
extern char *Concat3() ;

#ifndef YYSTYPE
#define YYSTYPE int
#endif
#include <stdio.h>

#ifndef __cplusplus
#ifndef __STDC__
#define const
#endif
#endif



#define	YYFINAL		1333
#define	YYFLAG		-32768
#define	YYNTBASE	123

#define YYTRANSLATE(x) ((unsigned)(x) <= 353 ? yytranslate[x] : 340)

static const char yytranslate[] = {     0,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,   110,     2,     2,     2,   105,   107,     2,    99,
   100,   103,   101,   116,   102,   113,   104,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,   118,   120,   111,
   119,   112,   117,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
   114,     2,   115,   106,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,   121,   108,   122,   109,     2,     2,     2,     2,
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
     2,     2,     2,     2,     2,     1,     2,     3,     4,     5,
     6,     7,     8,     9,    10,    11,    12,    13,    14,    15,
    16,    17,    18,    19,    20,    21,    22,    23,    24,    25,
    26,    27,    28,    29,    30,    31,    32,    33,    34,    35,
    36,    37,    38,    39,    40,    41,    42,    43,    44,    45,
    46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
    56,    57,    58,    59,    60,    61,    62,    63,    64,    65,
    66,    67,    68,    69,    70,    71,    72,    73,    74,    75,
    76,    77,    78,    79,    80,    81,    82,    83,    84,    85,
    86,    87,    88,    89,    90,    91,    92,    93,    94,    95,
    96,    97,    98
};

#if YYDEBUG != 0
static const short yyprhs[] = {     0,
     0,     2,     4,     6,     8,    10,    12,    15,    17,    19,
    23,    25,    27,    29,    31,    33,    37,    39,    41,    43,
    45,    47,    49,    52,    56,    60,    61,    64,    67,    69,
    71,    73,    75,    77,    79,    81,    83,    85,    87,    89,
    91,    93,    95,    97,    99,   101,   103,   105,   107,   109,
   111,   113,   115,   117,   119,   121,   124,   127,   129,   131,
   133,   134,   136,   138,   143,   147,   152,   153,   158,   159,
   160,   166,   169,   172,   173,   181,   182,   187,   188,   196,
   200,   204,   209,   214,   219,   221,   223,   228,   233,   238,
   240,   244,   246,   249,   252,   255,   258,   261,   264,   267,
   270,   275,   277,   283,   292,   295,   301,   303,   305,   307,
   310,   314,   318,   319,   321,   324,   327,   330,   334,   339,
   340,   343,   347,   349,   354,   356,   359,   365,   370,   372,
   375,   377,   381,   385,   387,   391,   395,   399,   401,   405,
   409,   411,   415,   419,   421,   425,   429,   433,   437,   439,
   443,   447,   449,   453,   455,   459,   461,   465,   467,   471,
   473,   477,   479,   485,   487,   491,   493,   495,   497,   499,
   501,   503,   505,   507,   509,   511,   513,   515,   519,   521,
   522,   524,   527,   530,   533,   536,   539,   542,   543,   548,
   549,   554,   555,   561,   564,   567,   571,   572,   577,   578,
   583,   584,   589,   590,   595,   596,   601,   602,   608,   611,
   614,   617,   620,   623,   627,   629,   631,   636,   642,   644,
   647,   650,   657,   665,   673,   681,   686,   692,   699,   707,
   709,   712,   715,   720,   726,   733,   741,   743,   745,   747,
   749,   751,   753,   755,   757,   760,   763,   765,   768,   770,
   772,   774,   776,   778,   781,   784,   787,   790,   793,   796,
   799,   802,   805,   808,   811,   814,   817,   820,   823,   825,
   828,   831,   833,   836,   839,   842,   845,   848,   851,   854,
   857,   860,   863,   866,   869,   872,   874,   876,   878,   880,
   882,   884,   886,   888,   890,   892,   894,   896,   898,   900,
   902,   904,   906,   908,   910,   912,   914,   916,   918,   920,
   922,   924,   925,   926,   927,   936,   942,   945,   950,   954,
   958,   959,   962,   965,   967,   971,   973,   977,   981,   982,
   984,   985,   987,   989,   991,   993,   995,   997,   999,  1001,
  1004,  1007,  1010,  1013,  1014,  1017,  1020,  1023,  1026,  1028,
  1030,  1033,  1036,  1039,  1042,  1045,  1049,  1053,  1058,  1061,
  1064,  1068,  1072,  1073,  1078,  1082,  1084,  1089,  1092,  1095,
  1098,  1101,  1104,  1108,  1112,  1116,  1120,  1124,  1128,  1132,
  1134,  1141,  1148,  1155,  1162,  1167,  1172,  1179,  1186,  1193,
  1200,  1205,  1210,  1212,  1218,  1225,  1232,  1239,  1245,  1252,
  1259,  1266,  1267,  1270,  1272,  1273,  1278,  1281,  1282,  1287,
  1292,  1297,  1300,  1302,  1305,  1307,  1310,  1313,  1318,  1320,
  1322,  1323,  1326,  1330,  1335,  1341,  1346,  1349,  1353,  1358,
  1362,  1364,  1367,  1370,  1374,  1376,  1378,  1381,  1383,  1386,
  1390,  1395,  1399,  1401,  1404,  1406,  1409,  1411,  1414,  1417,
  1420,  1422,  1425,  1428,  1431,  1434,  1437,  1440,  1443,  1446,
  1449,  1452,  1455,  1457,  1459,  1461,  1463,  1465,  1468,  1471,
  1474,  1477,  1480,  1481,  1483,  1486,  1490,  1495,  1497,  1499,
  1503,  1505,  1507,  1509,  1511,  1513,  1515,  1517,  1521,  1526,
  1530,  1531,  1532,  1538,  1540,  1543,  1544,  1547,  1550,  1554,
  1560,  1564,  1568,  1572,  1573,  1580,  1581,  1588,  1596,  1597,
  1606,  1607,  1618,  1627,  1628,  1635,  1639,  1642,  1645,  1649,
  1651,  1653,  1654,  1657,  1659,  1661,  1663,  1666,  1669,  1672,
  1673,  1674,  1681,  1684,  1687,  1690,  1692,  1694,  1696,  1697,
  1701,  1702,  1707,  1708,  1713,  1714,  1719,  1720,  1725,  1726,
  1731,  1732,  1737,  1738,  1743,  1744,  1749,  1750,  1755,  1756,
  1760,  1761,  1766,  1767,  1772,  1773,  1778,  1779,  1784,  1785,
  1790,  1791,  1796,  1797,  1802,  1805,  1807,  1808,  1814,  1819,
  1822,  1825,  1828,  1831,  1836,  1843,  1849,  1850,  1857,  1858,
  1867,  1868,  1876,  1878,  1884,  1890,  1896,  1902,  1908,  1915,
  1922,  1929,  1936,  1943,  1950,  1957,  1964,  1971,  1979,  1987,
  1995,  2003,  2005,  2012,  2020,  2027,  2035,  2042,  2050,  2051,
  2053,  2056,  2060,  2065,  2069,  2074,  2078,  2083,  2087,  2091,
  2094,  2096,  2098,  2100,  2102,  2104,  2106,  2109,  2111,  2113,
  2116,  2119,  2123,  2128,  2130,  2135,  2140,  2145,  2150,  2153,
  2156,  2160,  2165,  2170,  2175,  2179,  2183,  2185,  2187,  2189,
  2192,  2195,  2198,  2202,  2207,  2209,  2212,  2215,  2220,  2224,
  2229,  2231,  2233,  2235,  2237,  2239,  2241,  2243,  2246,  2250,
  2255,  2257,  2259,  2262,  2265,  2269,  2273,  2277,  2282,  2284,
  2286,  2288,  2290,  2294,  2297,  2299,  2302,  2305,  2309,  2311,
  2313,  2314,  2317,  2319,  2321,  2324,  2326,  2329,  2331,  2334,
  2337,  2339,  2342,  2344,  2347,  2349,  2352,  2354,  2357,  2360,
  2362
};

static const short yyrhs[] = {    69,
     0,    68,     0,    71,     0,    72,     0,    70,     0,    67,
     0,   124,    67,     0,   332,     0,   333,     0,    99,   125,
   100,     0,   335,     0,   336,     0,    55,     0,   123,     0,
   124,     0,    99,   165,   100,     0,   195,     0,   192,     0,
   197,     0,   199,     0,    73,     0,   338,     0,    56,   130,
     0,    56,   188,   129,     0,    56,   127,   129,     0,     0,
   325,   129,     0,   324,   129,     0,   101,     0,   102,     0,
   103,     0,   104,     0,   105,     0,   106,     0,   107,     0,
   108,     0,   109,     0,   110,     0,   111,     0,   112,     0,
    77,     0,    78,     0,    83,     0,    84,     0,    74,     0,
    88,     0,   113,     0,    87,     0,    75,     0,    76,     0,
    79,     0,    80,     0,    81,     0,    82,     0,   164,     0,
    99,   100,     0,   114,   115,     0,    53,     0,    54,     0,
   116,     0,     0,   188,     0,   126,     0,   132,   114,   165,
   115,     0,   132,    99,   100,     0,   132,    99,   140,   100,
     0,     0,   132,   133,   113,   139,     0,     0,     0,   132,
   134,    74,   135,   139,     0,   132,    75,     0,   132,    76,
     0,     0,   132,   114,    11,   115,   136,    74,   139,     0,
     0,   132,     9,   137,   139,     0,     0,   132,   114,    10,
   115,     9,   138,   139,     0,    73,    99,   100,     0,   338,
    99,   100,     0,    73,    99,   140,   100,     0,   338,    99,
   140,   100,     0,   199,    99,   163,   100,     0,   332,     0,
   333,     0,   199,    86,   109,   199,     0,   187,    86,   109,
   187,     0,   188,    86,   109,   188,     0,   163,     0,   140,
   116,   163,     0,   132,     0,    75,   141,     0,    76,   141,
     0,   324,   148,     0,   102,   148,     0,   101,   148,     0,
   109,   148,     0,   110,   148,     0,    47,   141,     0,    47,
    99,   241,   100,     0,   142,     0,   143,    99,   241,   100,
   147,     0,   143,    99,   140,   100,    99,   241,   100,   147,
     0,   143,   144,     0,   143,    99,   140,   100,   144,     0,
    53,     0,    16,     0,    17,     0,   331,    53,     0,   188,
   145,   147,     0,   127,   145,   147,     0,     0,   146,     0,
   324,   145,     0,   325,   145,     0,   114,   115,     0,   114,
   165,   115,     0,   146,   114,   165,   115,     0,     0,    99,
   100,     0,    99,   140,   100,     0,   141,     0,    99,   241,
   100,   148,     0,   148,     0,   150,   149,     0,   150,   114,
   165,   115,   149,     0,   150,   114,   115,   149,     0,    54,
     0,   331,    54,     0,   149,     0,   151,    87,   149,     0,
   151,    88,   149,     0,   151,     0,   152,   103,   151,     0,
   152,   104,   151,     0,   152,   105,   151,     0,   152,     0,
   153,   101,   152,     0,   153,   102,   152,     0,   153,     0,
   154,    77,   153,     0,   154,    78,   153,     0,   154,     0,
   155,   111,   154,     0,   155,   112,   154,     0,   155,    79,
   154,     0,   155,    80,   154,     0,   155,     0,   156,    81,
   155,     0,   156,    82,   155,     0,   156,     0,   157,   107,
   156,     0,   157,     0,   158,   106,   157,     0,   158,     0,
   159,   108,   158,     0,   159,     0,   160,    83,   159,     0,
   160,     0,   161,    84,   160,     0,   161,     0,   161,   117,
   165,   118,   162,     0,   162,     0,   141,   164,   163,     0,
   119,     0,    89,     0,    90,     0,    91,     0,    92,     0,
    93,     0,    94,     0,    95,     0,    96,     0,    97,     0,
    98,     0,   163,     0,   165,   116,   163,     0,   162,     0,
     0,   165,     0,   173,   120,     0,   169,   120,     0,   193,
   120,     0,   195,   120,     0,   194,   120,     0,     1,   120,
     0,     0,   187,   313,   170,   242,     0,     0,   188,   313,
   171,   242,     0,     0,   169,   116,   313,   172,   242,     0,
   187,   183,     0,   188,   183,     0,   169,   116,   183,     0,
     0,   185,   305,   174,   242,     0,     0,   186,   305,   175,
   242,     0,     0,   199,   305,   176,   242,     0,     0,    73,
   305,   177,   242,     0,     0,   338,   305,   178,   242,     0,
     0,   173,   116,   305,   179,   242,     0,   185,   180,     0,
   186,   180,     0,   199,   180,     0,    73,   180,     0,   338,
   180,     0,   173,   116,   180,     0,   184,     0,   181,     0,
   312,    99,   140,   100,     0,   312,   320,    99,   140,   100,
     0,   182,     0,   324,   180,     0,   325,   180,     0,    99,
   310,   100,    99,   140,   100,     0,    99,   310,   100,   320,
    99,   140,   100,     0,    99,   312,   320,   100,    99,   140,
   100,     0,    99,    73,   320,   100,    99,   140,   100,     0,
    73,    99,   140,   100,     0,    73,   320,    99,   140,   100,
     0,    99,   308,   100,    99,   140,   100,     0,    99,   308,
   100,   320,    99,   140,   100,     0,   184,     0,   324,   183,
     0,   325,   183,     0,   125,    99,   140,   100,     0,   125,
   320,    99,   140,   100,     0,    99,   314,   100,    99,   140,
   100,     0,    99,   314,   100,   320,    99,   140,   100,     0,
   191,     0,   193,     0,   196,     0,   192,     0,   195,     0,
   194,     0,   197,     0,   198,     0,   188,   198,     0,   187,
   189,     0,   190,     0,   188,   190,     0,   198,     0,   190,
     0,    34,     0,    48,     0,    14,     0,   187,   199,     0,
   192,   198,     0,   199,   198,     0,   191,   189,     0,   191,
   199,     0,   188,   199,     0,   199,   199,     0,   199,   190,
     0,   192,   190,     0,   192,   199,     0,   187,   201,     0,
   187,   200,     0,   195,   198,     0,   194,   198,     0,   193,
   189,     0,   200,     0,   188,   200,     0,   194,   190,     0,
   201,     0,   188,   201,     0,   195,   190,     0,   187,    73,
     0,   187,   338,     0,   197,   198,     0,    73,   198,     0,
   338,   198,     0,   196,   189,     0,   188,    73,     0,   188,
   338,     0,    73,   190,     0,   338,   190,     0,   197,   190,
     0,    31,     0,    29,     0,    51,     0,    18,     0,    28,
     0,    62,     0,    65,     0,    63,     0,    64,     0,    61,
     0,    20,     0,    30,     0,    36,     0,    24,     0,    35,
     0,    40,     0,    38,     0,    39,     0,    19,     0,    43,
     0,    37,     0,    44,     0,   202,     0,   227,     0,   206,
     0,   228,     0,     0,     0,     0,   206,   207,   203,   121,
   204,   214,   205,   122,     0,   213,   207,   121,   214,   122,
     0,   213,   328,     0,   329,   327,   213,   328,     0,   329,
   213,   328,     0,   327,   213,   328,     0,     0,   118,   208,
     0,   118,     1,     0,   209,     0,   208,   116,   209,     0,
   339,     0,    61,   211,   339,     0,   212,   210,   339,     0,
     0,    61,     0,     0,   212,     0,    58,     0,    60,     0,
    59,     0,     8,     0,    21,     0,    33,     0,    57,     0,
     3,    57,     0,     5,    57,     0,    12,    57,     0,    13,
    57,     0,     0,   214,   215,     0,   217,   120,     0,   216,
   120,     0,   212,   118,     0,   271,     0,   295,     0,   195,
   120,     0,   194,   120,     0,   313,   120,     0,   196,   120,
     0,   193,   120,     0,   188,   313,   222,     0,   187,   313,
   222,     0,   216,   116,   313,   222,     0,   188,   225,     0,
   187,   225,     0,   216,   116,   225,     0,   186,   305,   222,
     0,     0,   199,   305,   218,   222,     0,   338,   305,   222,
     0,   219,     0,   217,   116,   305,   222,     0,   186,   223,
     0,   199,   223,     0,    73,   223,     0,   338,   223,     0,
   185,   223,     0,   217,   116,   223,     0,    73,   313,   222,
     0,    73,   307,   222,     0,    73,   312,   222,     0,   185,
   313,   222,     0,   185,   307,   222,     0,   185,   312,   222,
     0,   220,     0,    73,   324,    99,   312,   100,   222,     0,
    73,   325,    99,   312,   100,   222,     0,    73,   324,    99,
    73,   100,   222,     0,    73,   325,    99,    73,   100,   222,
     0,    73,   324,   310,   222,     0,    73,   325,   310,   222,
     0,   185,   324,    99,   312,   100,   222,     0,   185,   325,
    99,   312,   100,   222,     0,   185,   324,    99,    73,   100,
   222,     0,   185,   325,    99,    73,   100,   222,     0,   185,
   324,   310,   222,     0,   185,   325,   310,   222,     0,   221,
     0,    73,    99,   310,   100,   222,     0,    73,    99,   312,
   320,   100,   222,     0,    73,    99,    73,   320,   100,   222,
     0,    73,    99,   310,   100,   320,   222,     0,   185,    99,
   310,   100,   222,     0,   185,    99,   312,   320,   100,   222,
     0,   185,    99,    73,   320,   100,   222,     0,   185,    99,
   310,   100,   320,   222,     0,     0,   119,    71,     0,   225,
     0,     0,    73,   224,   118,   166,     0,   118,   166,     0,
     0,   313,   226,   118,   166,     0,   229,   121,   230,   122,
     0,   228,   121,   230,   122,     0,   229,   328,     0,    27,
     0,   331,    27,     0,   231,     0,   231,   116,     0,   232,
   233,     0,   231,   116,   232,   233,     0,    66,     0,    73,
     0,     0,   119,   166,     0,    99,   100,   131,     0,    99,
   241,   100,   131,     0,    99,   241,   243,   100,   131,     0,
    99,   236,   100,   131,     0,    99,   100,     0,    99,   241,
   100,     0,    99,   241,   243,   100,     0,    99,   236,   100,
     0,   238,     0,   238,   237,     0,   241,   237,     0,   241,
   243,   237,     0,    85,     0,    85,     0,   116,    85,     0,
   240,     0,   240,   243,     0,   241,   116,   239,     0,   241,
   243,   116,   239,     0,   238,   116,   239,     0,   241,     0,
   241,   243,     0,   240,     0,   240,   243,     0,   185,     0,
   185,   319,     0,   185,   313,     0,   185,   307,     0,   187,
     0,   187,   319,     0,   187,   313,     0,   186,   313,     0,
   186,   307,     0,   199,   313,     0,   199,   307,     0,    73,
   313,     0,    73,   307,     0,   338,   313,     0,   338,   307,
     0,   188,   313,     0,   186,     0,   199,     0,    73,     0,
   338,     0,   188,     0,   186,   319,     0,   199,   319,     0,
    73,   319,     0,   338,   319,     0,   188,   319,     0,     0,
   243,     0,   119,   244,     0,   121,   245,   122,     0,   121,
   245,   116,   122,     0,   163,     0,   244,     0,   245,   116,
   244,     0,   247,     0,   248,     0,   253,     0,   254,     0,
   256,     0,   262,     0,   168,     0,   263,   118,   246,     0,
    26,   166,   118,   246,     0,    45,   118,   246,     0,     0,
     0,   121,   249,   252,   250,   122,     0,   168,     0,   251,
   168,     0,     0,   252,   246,     0,   167,   120,     0,    50,
   255,   246,     0,    50,   255,   246,    23,   246,     0,    25,
   255,   246,     0,    99,   165,   100,     0,    99,     1,   100,
     0,     0,    52,    99,   167,   100,   257,   246,     0,     0,
    52,    99,     1,   100,   258,   246,     0,    49,   246,    52,
    99,   165,   100,   120,     0,     0,    49,   246,    52,    99,
     1,   100,   259,   120,     0,     0,    42,    99,   167,   120,
   167,   120,   167,   100,   260,   246,     0,    42,    99,   168,
   167,   120,   167,   100,   246,     0,     0,    42,    99,     1,
   100,   261,   246,     0,    46,   263,   120,     0,    41,   120,
     0,    22,   120,     0,    32,   167,   120,     0,    66,     0,
    73,     0,     0,   264,   265,     0,   269,     0,   270,     0,
   168,     0,   268,   269,     0,   268,   270,     0,   268,   168,
     0,     0,     0,   268,   121,   266,   264,   267,   122,     0,
    31,    67,     0,   313,   120,     0,   294,   120,     0,   271,
     0,   282,     0,   292,     0,     0,   313,   272,   248,     0,
     0,   185,   305,   273,   248,     0,     0,   185,     1,   274,
   248,     0,     0,   186,   305,   275,   248,     0,     0,   199,
   305,   276,   248,     0,     0,   199,     1,   277,   248,     0,
     0,    73,   305,   278,   248,     0,     0,   338,   305,   279,
   248,     0,     0,   187,   313,   280,   248,     0,     0,   188,
   313,   281,   248,     0,     0,   316,   283,   291,     0,     0,
   185,   316,   284,   291,     0,     0,   186,   316,   285,   291,
     0,     0,   199,   316,   286,   291,     0,     0,    73,   316,
   287,   291,     0,     0,   338,   316,   288,   291,     0,     0,
   187,   316,   289,   291,     0,     0,   188,   316,   290,   291,
     0,   251,   248,     0,   248,     0,     0,   338,   234,   302,
   293,   248,     0,   185,   234,   302,   248,     0,   338,   234,
     0,   185,   234,     0,   185,   296,     0,    73,   296,     0,
    99,   100,   131,   120,     0,    99,   241,   243,   100,   131,
   120,     0,    99,   236,   100,   131,   120,     0,     0,    99,
   100,   131,   302,   297,   248,     0,     0,    99,   241,   243,
   100,   131,   302,   298,   248,     0,     0,    99,   236,   100,
   131,   302,   299,   248,     0,   300,     0,    99,   186,   100,
   131,   120,     0,    99,   199,   100,   131,   120,     0,    99,
    73,   100,   131,   120,     0,    99,   338,   100,   131,   120,
     0,    99,   188,   100,   131,   120,     0,    99,   186,   319,
   100,   131,   120,     0,    99,   199,   319,   100,   131,   120,
     0,    99,   338,   319,   100,   131,   120,     0,    99,   188,
   319,   100,   131,   120,     0,    99,   186,   100,   131,   302,
   248,     0,    99,   199,   100,   131,   302,   248,     0,    99,
    73,   100,   131,   302,   248,     0,    99,   338,   100,   131,
   302,   248,     0,    99,   188,   100,   131,   302,   248,     0,
    99,   186,   319,   100,   131,   302,   248,     0,    99,   199,
   319,   100,   131,   302,   248,     0,    99,   338,   319,   100,
   131,   302,   248,     0,    99,   188,   319,   100,   131,   302,
   248,     0,   301,     0,    99,    73,   322,   100,   131,   120,
     0,    99,    73,   322,   100,   131,   302,   248,     0,    99,
    73,   323,   100,   131,   120,     0,    99,    73,   323,   100,
   131,   302,   248,     0,    99,    73,   320,   100,   131,   120,
     0,    99,    73,   320,   100,   131,   302,   248,     0,     0,
   303,     0,   118,   304,     0,   303,   116,   304,     0,    66,
    99,   140,   100,     0,    66,    99,   100,     0,    73,    99,
   140,   100,     0,    73,    99,   100,     0,   338,    99,   140,
   100,     0,   338,    99,   100,     0,    99,   140,   100,     0,
    99,   100,     0,   313,     0,   306,     0,   310,     0,   312,
     0,   307,     0,    73,     0,    73,   320,     0,   308,     0,
   309,     0,   324,   307,     0,   325,   307,     0,    99,   308,
   100,     0,    99,   308,   100,   320,     0,   311,     0,   324,
    99,   312,   100,     0,   325,    99,   312,   100,     0,   324,
    99,    73,   100,     0,   325,    99,    73,   100,     0,   324,
   310,     0,   325,   310,     0,    99,   310,   100,     0,    99,
   312,   320,   100,     0,    99,    73,   320,   100,     0,    99,
   310,   100,   320,     0,    99,    73,   100,     0,    99,   312,
   100,     0,   314,     0,   125,     0,   315,     0,   324,   313,
     0,   325,   313,     0,   125,   320,     0,    99,   314,   100,
     0,    99,   314,   100,   320,     0,   317,     0,   324,   316,
     0,   325,   316,     0,   125,    99,   140,   100,     0,    99,
   316,   100,     0,    99,   316,   100,   318,     0,   321,     0,
   235,     0,   322,     0,   323,     0,   320,     0,   321,     0,
   234,     0,   114,   115,     0,   114,   166,   115,     0,   321,
   114,   166,   115,     0,   324,     0,   325,     0,   324,   319,
     0,   325,   319,     0,    99,   322,   100,     0,    99,   323,
   100,     0,    99,   320,   100,     0,    99,   322,   100,   320,
     0,   103,     0,   107,     0,     6,     0,     7,     0,   327,
   103,   131,     0,   324,   188,     0,   328,     0,   213,   328,
     0,   326,    86,     0,   327,   326,    86,     0,    66,     0,
    73,     0,     0,   330,    86,     0,   329,     0,   327,     0,
   329,   327,     0,    66,     0,   327,    66,     0,   334,     0,
   327,   334,     0,   109,    73,     0,   128,     0,   329,   332,
     0,   332,     0,   329,   333,     0,   333,     0,   327,    73,
     0,   337,     0,   329,   337,     0,   329,    73,     0,    73,
     0,   338,     0
};

#endif

#if YYDEBUG != 0
static const short yyrline[] = { 0,
   355,   357,   361,   362,   363,   366,   368,   407,   409,   410,
   436,   438,   439,   442,   443,   444,   475,   477,   478,   480,
   481,   482,   490,   495,   496,   510,   512,   513,   518,   520,
   521,   522,   523,   524,   525,   526,   527,   528,   529,   530,
   531,   532,   533,   534,   535,   536,   537,   538,   539,   540,
   541,   542,   543,   544,   545,   546,   547,   548,   549,   550,
   563,   565,   584,   586,   594,   600,   632,   632,   634,   635,
   667,   685,   687,   692,   713,   718,   729,   744,   749,   771,
   773,   775,   780,   785,   807,   820,   821,   823,   824,   827,
   829,   835,   841,   842,   843,   850,   851,   852,   853,   854,
   855,   858,   879,   884,   888,   891,   897,   899,   900,   901,
   904,   908,   921,   923,   925,   927,   931,   933,   934,   937,
   939,   940,   943,   945,   953,   955,   956,   957,   962,   964,
   969,   971,   973,   977,   979,   981,   983,   987,   989,   991,
   995,   997,   999,  1003,  1005,  1007,  1009,  1011,  1015,  1017,
  1019,  1023,  1025,  1029,  1031,  1035,  1037,  1041,  1043,  1047,
  1049,  1053,  1056,  1063,  1065,  1069,  1071,  1072,  1073,  1074,
  1075,  1076,  1077,  1078,  1079,  1080,  1083,  1085,  1089,  1095,
  1097,  1114,  1121,  1126,  1128,  1129,  1130,  1153,  1155,  1155,
  1156,  1156,  1158,  1158,  1159,  1160,  1168,  1187,  1187,  1198,
  1198,  1200,  1200,  1208,  1208,  1210,  1210,  1225,  1225,  1226,
  1227,  1228,  1229,  1230,  1256,  1258,  1259,  1261,  1264,  1265,
  1266,  1269,  1273,  1276,  1279,  1284,  1287,  1290,  1293,  1298,
  1300,  1301,  1310,  1313,  1316,  1319,  1324,  1326,  1330,  1333,
  1335,  1336,  1337,  1340,  1342,  1343,  1346,  1348,  1351,  1353,
  1356,  1358,  1359,  1367,  1369,  1370,  1371,  1372,  1375,  1378,
  1379,  1380,  1381,  1384,  1386,  1387,  1388,  1389,  1392,  1394,
  1395,  1398,  1400,  1401,  1404,  1406,  1407,  1408,  1409,  1410,
  1413,  1415,  1417,  1418,  1420,  1448,  1450,  1451,  1452,  1453,
  1454,  1455,  1456,  1459,  1460,  1463,  1466,  1468,  1470,  1472,
  1474,  1476,  1478,  1480,  1482,  1484,  1486,  1490,  1492,  1495,
  1497,  1513,  1544,  1613,  1615,  1694,  1711,  1717,  1726,  1731,
  1742,  1744,  1745,  1748,  1754,  1761,  1763,  1764,  1767,  1769,
  1772,  1774,  1777,  1779,  1780,  1781,  1784,  1786,  1787,  1788,
  1790,  1791,  1793,  1803,  1805,  1809,  1811,  1813,  1815,  1816,
  1818,  1819,  1820,  1825,  1826,  1829,  1833,  1836,  1839,  1840,
  1841,  1852,  1860,  1893,  1893,  1900,  1901,  1903,  1904,  1905,
  1906,  1907,  1908,  1919,  1945,  1947,  1950,  1968,  1970,  1973,
  1984,  1987,  1989,  1991,  1993,  1995,  1998,  2000,  2002,  2004,
  2006,  2008,  2011,  2022,  2025,  2027,  2029,  2032,  2034,  2036,
  2038,  2045,  2047,  2061,  2063,  2063,  2074,  2076,  2076,  2079,
  2081,  2095,  2099,  2101,  2104,  2106,  2119,  2121,  2124,  2126,
  2129,  2131,  2139,  2141,  2144,  2145,  2170,  2172,  2173,  2174,
  2177,  2179,  2180,  2181,  2182,  2185,  2187,  2190,  2192,  2193,
  2194,  2195,  2207,  2209,  2210,  2211,  2261,  2263,  2264,  2265,
  2267,  2268,  2269,  2271,  2272,  2274,  2275,  2277,  2281,  2286,
  2287,  2289,  2292,  2294,  2295,  2296,  2297,  2299,  2300,  2301,
  2302,  2303,  2306,  2308,  2311,  2315,  2317,  2318,  2321,  2323,
  2329,  2331,  2332,  2333,  2334,  2335,  2336,  2339,  2341,  2342,
  2350,  2365,  2375,  2383,  2385,  2388,  2390,  2393,  2397,  2399,
  2400,  2403,  2405,  2409,  2411,  2411,  2415,  2415,  2416,  2420,
  2420,  2423,  2423,  2425,  2428,  2430,  2432,  2433,  2434,  2441,
  2443,  2449,  2451,  2455,  2457,  2458,  2459,  2460,  2461,  2462,
  2468,  2473,  2475,  2485,  2493,  2496,  2504,  2505,  2526,  2530,
  2536,  2549,  2553,  2557,  2557,  2560,  2564,  2593,  2598,  2602,
  2602,  2604,  2607,  2609,  2612,  2614,  2617,  2619,  2631,  2633,
  2633,  2634,  2634,  2635,  2635,  2636,  2636,  2637,  2637,  2638,
  2638,  2639,  2639,  2640,  2642,  2644,  2667,  2691,  2695,  2703,
  2706,  2729,  2732,  2761,  2765,  2768,  2773,  2778,  2780,  2785,
  2787,  2792,  2795,  2816,  2819,  2822,  2825,  2828,  2832,  2834,
  2839,  2841,  2845,  2848,  2851,  2854,  2857,  2861,  2864,  2869,
  2872,  2875,  2889,  2893,  2896,  2899,  2903,  2906,  2911,  2913,
  2916,  2918,  2921,  2923,  2925,  2926,  2927,  2928,  2930,  2931,
  2934,  2936,  2939,  2941,  2942,  2945,  2947,  2948,  2959,  2961,
  2962,  2965,  2967,  2976,  2978,  2979,  2980,  2981,  2982,  2983,
  2986,  2988,  2989,  2990,  2998,  3000,  3003,  3005,  3015,  3017,
  3037,  3040,  3043,  3044,  3048,  3050,  3051,  3074,  3076,  3077,
  3080,  3082,  3085,  3087,  3088,  3091,  3093,  3096,  3098,  3099,
  3102,  3104,  3105,  3106,  3109,  3111,  3112,  3113,  3116,  3118,
  3119,  3123,  3129,  3135,  3160,  3165,  3172,  3178,  3196,  3198,
  3201,  3203,  3205,  3207,  3212,  3226,  3231,  3292,  3294,  3301,
  3303,  3319,  3321,  3324,  3326,  3336,  3349,  3351,  3356,  3363,
  3365
};
#endif


#if YYDEBUG != 0 || defined (YYERROR_VERBOSE)

static const char * const yytname[] = {   "$","error","$undefined.","CHARE",
"BRANCHED","MESSAGE","HANDLE","GROUP","ENTRY","DOUBLEARROW","ALL_NODES","LOCAL",
"ACCUMULATOR","MONOTONIC","READONLY","WRITEONCE","NEWCHARE","NEWGROUP","AUTO",
"DOUBLE","INT","STRUCT","BREAK","ELSE","LONG","SWITCH","CASE","ENUM","REGISTER",
"TYPEDEF","CHAR","EXTERN","RETURN","UNION","CONST","FLOAT","SHORT","UNSIGNED",
"WCHAR_TOKEN","__WCHAR_TOKEN","PTRDIFF_TOKEN","CONTINUE","FOR","SIGNED","VOID",
"DEFAULT","GOTO","SIZEOF","VOLATILE","DO","IF","STATIC","WHILE","NEW","DELETE",
"THIS","OPERATOR","CLASS","PUBLIC","PROTECTED","PRIVATE","VIRTUAL","FRIEND",
"INLINE","UNDERSCORE_INLINE","OVERLOAD","IDENTIFIER","STRINGliteral","FLOATINGconstant",
"INTEGERconstant","CHARACTERconstant","OCTALconstant","HEXconstant","TYPEDEFname",
"ARROW","ICR","DECR","LSHIFT","RSHIFT","LE","GE","EQ","NE","ANDAND","OROR","ELLIPSIS",
"CLCL","DOTstar","ARROWstar","MULTassign","DIVassign","MODassign","PLUSassign",
"MINUSassign","LSassign","RSassign","ANDassign","ERassign","ORassign","'('",
"')'","'+'","'-'","'*'","'/'","'%'","'^'","'&'","'|'","'~'","'!'","'<'","'>'",
"'.'","'['","']'","','","'?'","':'","'='","';'","'{'","'}'","constant","string_literal_list",
"paren_identifier_declarator","primary_expression","non_elaborating_type_specifier",
"operator_function_name","operator_function_ptr_opt","any_operator","type_qualifier_list_opt",
"postfix_expression","@1","@2","@3","@4","@5","@6","member_name","argument_expression_list",
"unary_expression","allocation_expression","global_opt_scope_opt_operator_new",
"operator_new_type","operator_new_declarator_opt","operator_new_array_declarator",
"operator_new_initializer_opt","cast_expression","deallocation_expression","global_opt_scope_opt_delete",
"point_member_expression","multiplicative_expression","additive_expression",
"shift_expression","relational_expression","equality_expression","AND_expression",
"exclusive_OR_expression","inclusive_OR_expression","logical_AND_expression",
"logical_OR_expression","conditional_expression","assignment_expression","assignment_operator",
"comma_expression","constant_expression","comma_expression_opt","declaration",
"default_declaring_list","@7","@8","@9","declaring_list","@10","@11","@12","@13",
"@14","@15","constructed_declarator","constructed_paren_typedef_declarator",
"constructed_parameter_typedef_declarator","constructed_identifier_declarator",
"nonunary_constructed_identifier_declarator","declaration_specifier","type_specifier",
"declaration_qualifier_list","type_qualifier_list","declaration_qualifier","type_qualifier",
"basic_declaration_specifier","basic_type_specifier","sue_declaration_specifier",
"sue_type_specifier_elaboration","sue_type_specifier","typedef_declaration_specifier",
"typedef_type_specifier","storage_class","basic_type_name","elaborated_type_name_elaboration",
"elaborated_type_name","aggregate_name_elaboration","@16","@17","@18","aggregate_name",
"derivation_opt","derivation_list","parent_class","virtual_opt","access_specifier_opt",
"access_specifier","aggregate_key","member_declaration_list_opt","member_declaration",
"member_default_declaring_list","member_declaring_list","@19","member_conflict_declaring_item",
"member_conflict_paren_declaring_item","member_conflict_paren_postfix_declaring_item",
"member_pure_opt","bit_field_declarator","@20","bit_field_identifier_declarator",
"@21","enum_name_elaboration","enum_name","global_opt_scope_opt_enum_key","enumerator_list",
"enumerator_list_no_trailing_comma","enumerator_name","enumerator_value_opt",
"parameter_type_list","old_parameter_type_list","named_parameter_type_list",
"comma_opt_ellipsis","parameter_list","parameter_declaration","non_casting_parameter_declaration",
"type_name","initializer_opt","initializer","initializer_group","initializer_list",
"statement","labeled_statement","compound_statement","@22","@23","declaration_list",
"statement_list_opt","expression_statement","selection_statement","if_cond",
"iteration_statement","@24","@25","@26","@27","@28","jump_statement","label",
"translation_unit","external_definition","@29","@30","linkage_specifier","function_declaration",
"function_definition","new_function_definition","@31","@32","@33","@34","@35",
"@36","@37","@38","@39","@40","old_function_definition","@41","@42","@43","@44",
"@45","@46","@47","@48","old_function_body","constructor_function_definition",
"@49","constructor_function_declaration","constructor_function_in_class","constructor_parameter_list_and_body",
"@50","@51","@52","constructor_conflicting_parameter_list_and_body","constructor_conflicting_typedef_declarator",
"constructor_init_list_opt","constructor_init_list","constructor_init","declarator",
"typedef_declarator","parameter_typedef_declarator","clean_typedef_declarator",
"clean_postfix_typedef_declarator","paren_typedef_declarator","postfix_paren_typedef_declarator",
"simple_paren_typedef_declarator","identifier_declarator","unary_identifier_declarator",
"postfix_identifier_declarator","old_function_declarator","postfix_old_function_declarator",
"old_postfixing_abstract_declarator","abstract_declarator","postfixing_abstract_declarator",
"array_abstract_declarator","unary_abstract_declarator","postfix_abstract_declarator",
"asterisk_or_ampersand","unary_modifier","scoping_name","scope","tag_name","global_scope",
"@53","global_or_scope","scope_opt_identifier","scope_opt_complex_name","complex_name",
"global_opt_scope_opt_identifier","global_opt_scope_opt_complex_name","scoped_typedefname",
"global_or_scoped_typedefname","global_opt_scope_opt_typedefname", NULL
};
#endif

static const short yyr1[] = {     0,
   123,   123,   123,   123,   123,   124,   124,   125,   125,   125,
   126,   126,   126,   126,   126,   126,   127,   127,   127,   127,
   127,   127,   128,   128,   128,   129,   129,   129,   130,   130,
   130,   130,   130,   130,   130,   130,   130,   130,   130,   130,
   130,   130,   130,   130,   130,   130,   130,   130,   130,   130,
   130,   130,   130,   130,   130,   130,   130,   130,   130,   130,
   131,   131,   132,   132,   132,   132,   133,   132,   134,   135,
   132,   132,   132,   136,   132,   137,   132,   138,   132,   132,
   132,   132,   132,   132,   139,   139,   139,   139,   139,   140,
   140,   141,   141,   141,   141,   141,   141,   141,   141,   141,
   141,   141,   142,   142,   142,   142,   143,   143,   143,   143,
   144,   144,   145,   145,   145,   145,   146,   146,   146,   147,
   147,   147,   148,   148,   149,   149,   149,   149,   150,   150,
   151,   151,   151,   152,   152,   152,   152,   153,   153,   153,
   154,   154,   154,   155,   155,   155,   155,   155,   156,   156,
   156,   157,   157,   158,   158,   159,   159,   160,   160,   161,
   161,   162,   162,   163,   163,   164,   164,   164,   164,   164,
   164,   164,   164,   164,   164,   164,   165,   165,   166,   167,
   167,   168,   168,   168,   168,   168,   168,   170,   169,   171,
   169,   172,   169,   169,   169,   169,   174,   173,   175,   173,
   176,   173,   177,   173,   178,   173,   179,   173,   173,   173,
   173,   173,   173,   173,   180,   180,   180,   180,   180,   180,
   180,   181,   181,   181,   181,   182,   182,   182,   182,   183,
   183,   183,   184,   184,   184,   184,   185,   185,   185,   186,
   186,   186,   186,   187,   187,   187,   188,   188,   189,   189,
   190,   190,   190,   191,   191,   191,   191,   191,   192,   192,
   192,   192,   192,   193,   193,   193,   193,   193,   194,   194,
   194,   195,   195,   195,   196,   196,   196,   196,   196,   196,
   197,   197,   197,   197,   197,   198,   198,   198,   198,   198,
   198,   198,   198,   198,   198,   199,   199,   199,   199,   199,
   199,   199,   199,   199,   199,   199,   199,   200,   200,   201,
   201,   203,   204,   205,   202,   202,   206,   206,   206,   206,
   207,   207,   207,   208,   208,   209,   209,   209,   210,   210,
   211,   211,   212,   212,   212,   212,   213,   213,   213,   213,
   213,   213,   213,   214,   214,   215,   215,   215,   215,   215,
   215,   215,   215,   215,   215,   216,   216,   216,   216,   216,
   216,   217,   218,   217,   217,   217,   217,   217,   217,   217,
   217,   217,   217,   219,   219,   219,   219,   219,   219,   219,
   220,   220,   220,   220,   220,   220,   220,   220,   220,   220,
   220,   220,   220,   221,   221,   221,   221,   221,   221,   221,
   221,   222,   222,   223,   224,   223,   225,   226,   225,   227,
   227,   228,   229,   229,   230,   230,   231,   231,   232,   232,
   233,   233,   234,   234,   234,   234,   235,   235,   235,   235,
   236,   236,   236,   236,   236,   237,   237,   238,   238,   238,
   238,   238,   239,   239,   239,   239,   240,   240,   240,   240,
   240,   240,   240,   240,   240,   240,   240,   240,   240,   240,
   240,   240,   241,   241,   241,   241,   241,   241,   241,   241,
   241,   241,   242,   242,   243,   244,   244,   244,   245,   245,
   246,   246,   246,   246,   246,   246,   246,   247,   247,   247,
   249,   250,   248,   251,   251,   252,   252,   253,   254,   254,
   254,   255,   255,   257,   256,   258,   256,   256,   259,   256,
   260,   256,   256,   261,   256,   262,   262,   262,   262,   263,
   263,   264,   264,   265,   265,   265,   265,   265,   265,   266,
   267,   265,   268,   269,   269,   270,   270,   270,   272,   271,
   273,   271,   274,   271,   275,   271,   276,   271,   277,   271,
   278,   271,   279,   271,   280,   271,   281,   271,   283,   282,
   284,   282,   285,   282,   286,   282,   287,   282,   288,   282,
   289,   282,   290,   282,   291,   291,   293,   292,   292,   294,
   294,   295,   295,   296,   296,   296,   297,   296,   298,   296,
   299,   296,   296,   300,   300,   300,   300,   300,   300,   300,
   300,   300,   300,   300,   300,   300,   300,   300,   300,   300,
   300,   300,   301,   301,   301,   301,   301,   301,   302,   302,
   303,   303,   304,   304,   304,   304,   304,   304,   304,   304,
   305,   305,   306,   306,   306,   307,   307,   307,   308,   308,
   308,   309,   309,   310,   310,   310,   310,   310,   310,   310,
   311,   311,   311,   311,   312,   312,   313,   313,   314,   314,
   314,   315,   315,   315,   316,   316,   316,   317,   317,   317,
   318,   318,   319,   319,   319,   320,   320,   321,   321,   321,
   322,   322,   322,   322,   323,   323,   323,   323,   324,   324,
   324,   324,   325,   325,   326,   326,   327,   327,   328,   328,
   330,   329,   331,   331,   331,   332,   332,   333,   333,   334,
   334,   335,   335,   336,   336,   337,   338,   338,   338,   339,
   339
};

static const short yyr2[] = {     0,
     1,     1,     1,     1,     1,     1,     2,     1,     1,     3,
     1,     1,     1,     1,     1,     3,     1,     1,     1,     1,
     1,     1,     2,     3,     3,     0,     2,     2,     1,     1,
     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
     1,     1,     1,     1,     1,     2,     2,     1,     1,     1,
     0,     1,     1,     4,     3,     4,     0,     4,     0,     0,
     5,     2,     2,     0,     7,     0,     4,     0,     7,     3,
     3,     4,     4,     4,     1,     1,     4,     4,     4,     1,
     3,     1,     2,     2,     2,     2,     2,     2,     2,     2,
     4,     1,     5,     8,     2,     5,     1,     1,     1,     2,
     3,     3,     0,     1,     2,     2,     2,     3,     4,     0,
     2,     3,     1,     4,     1,     2,     5,     4,     1,     2,
     1,     3,     3,     1,     3,     3,     3,     1,     3,     3,
     1,     3,     3,     1,     3,     3,     3,     3,     1,     3,
     3,     1,     3,     1,     3,     1,     3,     1,     3,     1,
     3,     1,     5,     1,     3,     1,     1,     1,     1,     1,
     1,     1,     1,     1,     1,     1,     1,     3,     1,     0,
     1,     2,     2,     2,     2,     2,     2,     0,     4,     0,
     4,     0,     5,     2,     2,     3,     0,     4,     0,     4,
     0,     4,     0,     4,     0,     4,     0,     5,     2,     2,
     2,     2,     2,     3,     1,     1,     4,     5,     1,     2,
     2,     6,     7,     7,     7,     4,     5,     6,     7,     1,
     2,     2,     4,     5,     6,     7,     1,     1,     1,     1,
     1,     1,     1,     1,     2,     2,     1,     2,     1,     1,
     1,     1,     1,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     2,     2,     2,     1,     2,
     2,     1,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     2,     2,     2,     1,     1,     1,     1,     1,
     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
     1,     0,     0,     0,     8,     5,     2,     4,     3,     3,
     0,     2,     2,     1,     3,     1,     3,     3,     0,     1,
     0,     1,     1,     1,     1,     1,     1,     1,     1,     2,
     2,     2,     2,     0,     2,     2,     2,     2,     1,     1,
     2,     2,     2,     2,     2,     3,     3,     4,     2,     2,
     3,     3,     0,     4,     3,     1,     4,     2,     2,     2,
     2,     2,     3,     3,     3,     3,     3,     3,     3,     1,
     6,     6,     6,     6,     4,     4,     6,     6,     6,     6,
     4,     4,     1,     5,     6,     6,     6,     5,     6,     6,
     6,     0,     2,     1,     0,     4,     2,     0,     4,     4,
     4,     2,     1,     2,     1,     2,     2,     4,     1,     1,
     0,     2,     3,     4,     5,     4,     2,     3,     4,     3,
     1,     2,     2,     3,     1,     1,     2,     1,     2,     3,
     4,     3,     1,     2,     1,     2,     1,     2,     2,     2,
     1,     2,     2,     2,     2,     2,     2,     2,     2,     2,
     2,     2,     1,     1,     1,     1,     1,     2,     2,     2,
     2,     2,     0,     1,     2,     3,     4,     1,     1,     3,
     1,     1,     1,     1,     1,     1,     1,     3,     4,     3,
     0,     0,     5,     1,     2,     0,     2,     2,     3,     5,
     3,     3,     3,     0,     6,     0,     6,     7,     0,     8,
     0,    10,     8,     0,     6,     3,     2,     2,     3,     1,
     1,     0,     2,     1,     1,     1,     2,     2,     2,     0,
     0,     6,     2,     2,     2,     1,     1,     1,     0,     3,
     0,     4,     0,     4,     0,     4,     0,     4,     0,     4,
     0,     4,     0,     4,     0,     4,     0,     4,     0,     3,
     0,     4,     0,     4,     0,     4,     0,     4,     0,     4,
     0,     4,     0,     4,     2,     1,     0,     5,     4,     2,
     2,     2,     2,     4,     6,     5,     0,     6,     0,     8,
     0,     7,     1,     5,     5,     5,     5,     5,     6,     6,
     6,     6,     6,     6,     6,     6,     6,     7,     7,     7,
     7,     1,     6,     7,     6,     7,     6,     7,     0,     1,
     2,     3,     4,     3,     4,     3,     4,     3,     3,     2,
     1,     1,     1,     1,     1,     1,     2,     1,     1,     2,
     2,     3,     4,     1,     4,     4,     4,     4,     2,     2,
     3,     4,     4,     4,     3,     3,     1,     1,     1,     2,
     2,     2,     3,     4,     1,     2,     2,     4,     3,     4,
     1,     1,     1,     1,     1,     1,     1,     2,     3,     4,
     1,     1,     2,     2,     3,     3,     3,     4,     1,     1,
     1,     1,     3,     2,     1,     2,     2,     3,     1,     1,
     0,     2,     1,     1,     2,     1,     2,     1,     2,     2,
     1,     2,     1,     2,     1,     2,     1,     2,     2,     1,
     1
};

static const short yydefact[] = {   522,
     0,     0,     0,     0,   691,   692,     0,     0,   253,   289,
   304,   296,   337,   299,   413,   290,   287,   297,   286,   338,
   251,   300,   298,   306,   302,   303,   301,   305,   307,   252,
   288,   701,   339,   295,   291,   293,   294,   292,   706,   700,
     0,   689,   690,     0,   658,   711,   526,     0,     0,     0,
     0,   701,   701,   247,   237,   240,   238,   242,   241,   239,
   243,   244,     0,   269,   272,   308,   310,   321,   309,   311,
     0,   523,     0,   524,   525,   536,   537,   538,     0,   539,
   657,   659,   559,   665,     0,     0,     0,   704,   695,   703,
     0,     0,     8,     9,   708,   717,     0,   187,   340,   341,
   342,   343,   533,    58,    59,   699,    21,    45,    49,    50,
    41,    42,    51,    52,    53,    54,    43,    44,    48,    46,
   167,   168,   169,   170,   171,   172,   173,   174,   175,   176,
     0,    29,    30,    31,    32,    33,    34,    35,    36,    37,
    38,    39,    40,    47,     0,    60,   166,    26,    23,    55,
    26,    18,    17,    19,    20,   310,     0,   311,     0,   704,
    22,   286,   636,     0,   658,   212,   216,   219,   215,   283,
   278,     0,   203,   632,   635,   638,   639,   633,   644,   634,
   631,   567,     0,     0,     0,   700,     0,     0,     0,   710,
   701,   701,   677,   662,   676,     0,   183,     0,   182,   543,
   701,   209,   581,   197,   561,   210,   199,   563,   275,     0,
   194,   230,   246,   250,   249,   254,   265,   264,   188,   571,
     0,     0,   276,   281,   195,   248,   245,   259,   270,   273,
   190,   573,   282,   257,   258,   262,   255,   263,   184,   268,
   186,   271,   267,   185,   274,   266,   280,   285,   277,   549,
   211,   261,   256,   260,   201,   565,     0,   312,     0,   317,
     0,     0,   412,   530,   529,   527,   528,   535,   534,     0,
     0,   694,   660,   666,   661,   667,   697,   707,   716,    61,
     0,     0,   709,   719,     0,   705,   718,   702,   414,   213,
   284,   279,   580,   205,   569,    56,    57,    25,    26,    26,
     0,    24,   704,   701,   637,   700,     0,     0,     0,     0,
     0,     0,     0,   701,   662,   696,   473,     0,   701,     0,
     0,     0,   220,   640,   649,     0,     0,   221,   641,   650,
    10,   663,   669,   108,   109,   701,   107,   129,    13,     6,
     2,     1,     5,     3,     4,   465,   701,   701,   435,   701,
    61,   701,   701,   701,   701,    14,    15,    63,    92,     0,
   123,   102,   701,   125,   131,   701,   134,   138,   141,   144,
   149,   152,   154,   156,   158,   160,   162,   164,    90,   447,
   463,   451,   467,   238,   242,   241,   464,     0,   431,   438,
     0,   701,   704,   703,     0,   713,   715,    11,    12,   466,
   700,   678,   123,   179,     0,     0,   704,   703,     0,     0,
   701,     0,   658,   196,   192,     0,     0,     0,   214,   207,
     0,     0,     0,   465,   464,   466,   701,     0,   620,   473,
     0,     0,   473,     0,     0,   473,     0,     0,   231,   232,
   473,     0,     0,     0,   473,     0,     0,   323,   336,   333,
   335,   334,   331,   720,   322,   324,   329,     0,     0,   721,
   326,     0,   344,   419,   420,     0,   415,   421,     0,   522,
   491,   540,   700,   494,     0,     0,   701,   701,     0,   576,
     0,   560,     0,   693,    62,   320,   698,   319,     0,   577,
   473,     0,     0,    28,    27,     0,   701,   701,   655,     0,
   700,     0,     0,     0,   642,   651,   656,     0,   663,   636,
     0,     0,     0,   701,   701,   204,   474,   552,     0,   701,
   568,   700,     0,   700,     0,   664,   701,   672,   670,   671,
   701,   100,     0,   701,   658,   459,   458,   470,   675,   673,
   674,   681,   682,   701,    93,    94,   465,   177,     0,   463,
   467,   240,   242,   241,   243,   464,     0,   466,   423,    97,
    96,   710,    98,    99,     7,    76,    72,    73,   701,   701,
     0,     0,   668,   701,   701,   701,   113,   105,   113,   701,
   126,   701,   701,   701,   701,   701,   701,   701,   701,   701,
   701,   701,   701,   701,   701,   701,   701,   701,   701,   701,
   701,   701,   701,   450,   449,   448,   455,   454,   468,   701,
   453,   452,   681,   682,   462,   472,   701,   457,   456,   469,
    61,   436,   701,   432,   439,    61,   701,   433,     0,    95,
   705,   712,   714,   110,   130,   701,   461,   460,   471,   701,
   679,   701,   705,   701,     0,     0,     0,     0,     0,   701,
   473,     0,     0,     0,   473,     0,     0,   544,   675,   699,
   700,   701,   621,     0,   579,   701,   198,   542,   562,   200,
   546,   564,   189,   556,   572,   191,   558,   574,   550,   202,
   548,   566,   701,   332,   701,   330,   701,   313,   701,   411,
   416,   701,   417,   410,     0,   496,   203,   197,   199,   188,
   190,   201,   495,   575,   205,   318,     0,   206,   554,   570,
   226,     0,   465,   653,     0,   642,   651,     0,   701,   643,
   701,   654,   652,   701,   664,   637,   700,     0,   700,     0,
   668,     0,   701,   478,   475,   217,     0,   647,   645,   648,
   646,   427,     0,     0,     0,   701,    80,     0,     0,     0,
     0,   681,   713,   715,   683,   684,   701,   681,   682,    16,
   701,   701,   701,   701,   701,     0,    65,     0,     0,     0,
     0,     0,    70,    91,   165,     0,     0,   701,   120,   114,
   113,   113,   120,   701,     0,   132,   133,   135,   136,   137,
   139,   140,   142,   143,   147,   148,   145,   146,   150,   151,
   153,   155,   157,   159,   161,     0,     0,   426,   437,   442,
   445,   443,   424,   440,    61,   701,   434,    81,     0,    80,
    81,   680,     0,   193,     0,     0,   208,   701,   701,   630,
     0,   701,   622,   327,   325,   328,   344,   700,   316,     0,
     0,   701,   701,   238,   242,   241,   239,     0,     0,   345,
     0,     0,   366,   380,   393,   349,   350,   539,     0,   421,
   422,     0,     0,   578,   227,   701,   653,   643,   654,   652,
     0,   701,     0,   701,   701,     0,   701,   647,   645,   648,
   646,   234,   479,     0,   218,   430,   428,     0,   101,     0,
    82,   687,   685,   686,   636,   704,   701,   681,   178,   124,
    77,     0,     0,     0,     0,    85,    86,    66,     0,    74,
    64,    68,     0,   701,   120,   117,     0,   701,   112,   701,
   115,   116,   111,   128,   701,   701,    84,   446,   444,   425,
   441,    83,   233,   624,     0,   626,     0,   629,   628,     0,
   314,   636,   701,   701,   370,   404,   583,   593,   612,   551,
   402,   402,   402,     0,     0,   701,   372,   582,   541,   402,
   402,   402,     0,     0,   368,   402,   634,   631,   360,   402,
   359,   402,   355,   352,   351,   354,   369,   363,   348,     0,
   347,     0,   346,   353,   371,   402,   418,   532,     0,     0,
   701,   180,     0,     0,     0,     0,     0,     0,     0,   706,
   521,   181,     0,   487,     0,   497,   481,   482,     0,   483,
   484,   485,   486,     0,     0,     0,   228,     0,   222,     0,
     0,   235,     0,   701,   476,   429,    61,   688,   701,     0,
     0,     0,    78,     0,    71,   701,   106,   103,   118,   121,
     0,     0,   127,   163,   623,   625,   627,     0,     0,   465,
    61,   463,   467,   464,     0,     0,     0,     0,   466,   407,
     0,   375,   376,   374,     0,     0,   402,     0,   402,   465,
     0,     0,   378,   379,   377,     0,   402,     0,   402,   362,
   357,   356,   402,   361,   402,   373,   402,   365,   518,     0,
     0,     0,     0,   517,     0,     0,   520,   521,     0,     0,
     0,     0,   701,   498,   701,   493,     0,   701,   225,   229,
   223,   224,   236,   477,   480,     0,     0,     0,     0,     0,
   465,   464,     0,   466,   122,   119,   315,   701,    61,   675,
   673,   674,   619,    61,   468,    61,   472,    61,   469,    61,
     0,   402,     0,    61,   471,   403,   701,   700,     0,   385,
   700,     0,   386,   675,   402,     0,   700,     0,   391,   700,
     0,   392,   364,   358,   367,     0,     0,   501,     0,   519,
     0,   700,     0,   180,   490,   516,     0,   499,     0,     0,
   700,   701,   701,   488,    88,     0,    89,    87,    79,    75,
   120,   406,   619,    61,    61,    61,   584,   587,   619,    61,
   619,    61,   619,    61,   619,    61,   394,   402,   402,   619,
    61,   409,   402,   402,   402,   402,    61,   398,   402,   402,
   402,   402,   402,   402,   503,   502,   489,   514,   180,     0,
     0,     0,   506,   504,   700,   701,   104,   596,     0,   619,
   396,   619,   619,     0,   594,     0,   619,   598,     0,   619,
   595,     0,   619,   586,   591,   619,   397,   395,   597,     0,
   619,   383,   381,   384,   382,   400,   401,   399,   389,   387,
   390,   388,     0,     0,   180,     0,     0,   500,     0,     0,
     0,   700,   605,   617,     0,   613,     0,   615,     0,   588,
   603,   599,     0,   607,   602,     0,   604,   600,     0,     0,
   585,   589,   606,   601,     0,   515,   180,     0,   509,     0,
   507,   505,   618,   614,   616,   608,   611,   609,   592,     0,
   610,     0,     0,     0,   508,   590,   511,   513,   510,     0,
   512,     0,     0
};

static const short yydefgoto[] = {   356,
   357,   535,   358,   577,    46,   298,   149,   559,   359,   571,
   572,   913,  1034,   766,  1119,   901,   748,   361,   362,   363,
   578,   779,   780,   919,   364,   365,   366,   367,   368,   369,
   370,   371,   372,   373,   374,   375,   376,   377,   378,   548,
   150,  1002,   405,  1003,  1004,    48,   436,   441,   651,    49,
   430,   433,   445,   317,   491,   655,   166,   167,   168,   211,
   169,   380,   381,   382,   383,   213,    54,    55,    56,   384,
   385,   386,    60,    61,    62,   406,    64,    65,    66,   462,
   837,  1048,    67,   258,   455,   456,   687,   683,   457,   172,
   689,   850,   851,   852,  1083,   853,   854,   855,  1062,   945,
  1049,   946,  1065,    69,    70,    71,   466,   467,   468,   693,
   193,   528,   388,   628,   389,   810,   390,   391,   516,   517,
   735,   884,  1006,  1007,  1008,   696,  1009,   481,   863,  1010,
  1011,  1091,  1012,  1280,  1279,  1324,  1330,  1273,  1013,  1014,
     1,    72,   470,   862,    73,    74,    75,    76,   270,   431,
   423,   434,   446,   444,   318,   492,   437,   442,    77,   271,
   432,   435,   447,   321,   493,   438,   443,   482,    78,   707,
    79,   857,   947,  1244,  1320,  1300,   948,   949,   428,   429,
   663,   697,   174,   175,   176,   177,   178,   179,   180,   181,
    81,    82,   189,    84,   529,   538,   539,   195,   540,   541,
   392,   654,    87,   185,    89,   408,    91,   409,   396,   397,
    95,   398,   399,    96,   410,   461
};

static const short yypact[] = {-32768,
 12729,   -66,    25,    65,-32768,-32768,   136,   179,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,   157,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,  9452,-32768,-32768,-32768,-32768,-32768,-32768,   228, 16674,
  3313,-32768,-32768,   180,   159,-32768,-32768,   -35,   186,  1391,
  5099, 16023, 16122,-32768,  8650,  8650,  1696,  2455,  2648,  3651,
  3651,-32768, 12828,-32768,-32768,-32768,    84,   -17,-32768,   198,
   -16,-32768,  6722,-32768,-32768,-32768,-32768,-32768,   237,   258,
-32768,-32768,-32768,-32768,  5893,  3313,   300,    83,-32768,  1801,
   327,   317,-32768,-32768,-32768,-32768, 16738,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,    31,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
   332,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,   338,-32768,-32768,  2624,-32768,-32768,
 17173,  3602,    66,    66,  3602,-32768,    68,-32768,    68,  1996,
    66,-32768,   153,  5516,   183,-32768,-32768,-32768,-32768,-32768,
-32768,    68,   339,-32768,-32768,-32768,-32768,-32768,-32768,   193,
-32768,-32768,  5976,  6158,  1318,-32768,   306,   366,   371,-32768,
 12927,  9781,-32768,-32768,   433,  6312,-32768,  6528,-32768,-32768,
 15726,-32768,   170,   364,-32768,-32768,   377,-32768,   417,  3313,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,   440,-32768,
  6089,  7885,-32768,   417,-32768,-32768,-32768,-32768,-32768,-32768,
   483,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,   487,-32768,  4799,-32768,   490,   539,
   137,   137,-32768,-32768,-32768,-32768,-32768,-32768,-32768,   499,
  6411,    66,-32768,-32768,-32768,-32768,-32768,   228,   417,    66,
    68,   543,-32768,   417,    68,  1996,-32768,-32768,-32768,-32768,
-32768,-32768,   170,   513,-32768,-32768,-32768,-32768,  4663,  2624,
   709,-32768,  1012, 12927,   551,   437,  7914,   556,   563,   463,
   566,  7500, 10969, 12927,   570,-32768,   554,   499, 12927,   577,
  6411, 12483,-32768,-32768,-32768,   193, 12593,-32768,-32768,-32768,
-32768,   350,   425,-32768,-32768, 15195,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,-32768, 11958, 15303, 15303,-32768, 13143,
    66, 15411, 15411, 15519, 15411,-32768,   618,-32768,   531,   -73,
  2049,-32768, 17334,-32768,-32768, 10994,    74,   632,   199,   322,
   167,   343,   606,   591,   620,   636,    29,-32768,-32768,  2346,
  2346,  7739, 11401,  3651,  3651,  3651, 11698,   644,   -43,   554,
   -32, 15411,  1304,  1515,   144,-32768,-32768,-32768,-32768, 12150,
   660,-32768,-32768,-32768,   664,   685,  1304,  1515,   488,   689,
 14979, 14090,   445,-32768,-32768,  7613,  6312, 14198,-32768,-32768,
  8042, 14306,   499,  7107, 11797, 12214,  1167,   499,   679,   554,
   499,  6411,   554,   499,  6411,   554,   499,  6411,-32768,-32768,
   554,   499,  6411,   499,   554,   499,  6411,-32768,-32768,-32768,
-32768,-32768,    52,   417,   682,-32768,   744,  1996,  1801,-32768,
-32768,   687,-32768,-32768,-32768,   696,   713,   706,   737,-32768,
-32768,-32768, 16802,-32768,  6528,  6528, 16221, 16320, 16419,-32768,
  6411,-32768, 16802,-32768,    66,   539,-32768,   539,    68,-32768,
   554,   499,  6411,-32768,-32768,   -33, 14979,  8859,-32768,   772,
   437,   775,   785,   463,   532,   584,-32768,   791,   588,   191,
 14414, 14522,     2, 14979,  9250,-32768,-32768,-32768,    21, 14979,
-32768,   466,   518,   528,   537,-32768, 17265,-32768,-32768,   433,
 13143,-32768,   841, 10114,   350,-32768,-32768,-32768,-32768,-32768,
-32768,  2547,  2346, 14979,-32768,-32768,  2408,-32768,    48,  2263,
 11896,  3602,    66,    66,    66,  5050,   799,  4875,-32768,-32768,
-32768,    45,-32768,-32768,-32768,-32768,-32768,-32768, 13791, 13575,
   796,   849,-32768, 14979, 14979, 13143,  1129,-32768,  7300,  9892,
-32768, 14979, 14979, 14979, 14979, 14979, 14979, 14979, 14979, 14979,
 14979, 14979, 14979, 14979, 14979, 14979, 14979, 14979, 14979, 14979,
 14979, 14979, 11104,-32768,-32768,-32768,-32768,-32768,-32768, 11203,
-32768,-32768,  3234,  3750,-32768,-32768, 10224,-32768,-32768,-32768,
    66,-32768, 17432,-32768,-32768,    66, 17432,-32768,    51,-32768,
  1304,-32768,-32768,-32768,-32768, 10334,-32768,-32768,-32768, 13899,
-32768, 14979,  1304, 14007,   838, 14090,   548,  8218, 14090, 12927,
   554, 14630, 13766, 14846,   554, 14954, 15062,-32768,   772,   867,
   870, 14115,-32768,   873,-32768,  1167,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,  2736,-32768,  3165,-32768,  2736,-32768,  9169,-32768,
   137, 14979,-32768,-32768,  8336,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,   539,   499,-32768,-32768,-32768,
-32768,   117, 12022,   875,   826,   350,   350,   876, 12927,   878,
 12927,   880,   881, 12927,   883,-32768,   578,   581,   590,   617,
   307,   149,  9250,-32768,-32768,-32768,   150,   594,   611,   594,
   611,-32768,   886,     7,   887, 10224,   624,   168,   888,   892,
   893, 10774,   627,   634,-32768,-32768, 10444,  5436,  2263,-32768,
 14979, 11500, 10554, 15411, 10664, 16610,-32768,   240,   868,   879,
   363, 16610,-32768,-32768,-32768,   248,   897, 10003,   899,   885,
  2198,  1129,   899, 14979,   544,-32768,-32768,    74,    74,    74,
   632,   632,   199,   199,   322,   322,   322,   322,   167,   167,
   343,   606,   591,   620,   636,   -54,   900,-32768,-32768,-32768,
   554,   554,-32768,-32768,    66, 17432,-32768,   624,   292,-32768,
-32768,-32768,   295,-32768, 15170, 15278,-32768, 14223, 14331,-32768,
   335, 14439,-32768,-32768,-32768,-32768,-32768,  5775,-32768,  1142,
  1352,  9567,  9674,  2829,  2929,  3867,  4436,  9360,   884,-32768,
   418,   444,-32768,-32768,-32768,-32768,-32768,   889,  6984,   706,
-32768,   891,  8748,-32768,-32768, 14979,-32768,-32768,-32768,-32768,
   345, 14979,   347, 14979, 14979,   352, 14979,-32768,-32768,-32768,
-32768,-32768,-32768,    57,-32768,-32768,-32768,    58,-32768,     8,
-32768,-32768,   350,-32768,   259,    83, 10554, 10884,-32768,-32768,
-32768,  4258,  4586,   915,  1530,-32768,-32768,-32768,   996,-32768,
-32768,-32768, 16610, 17382,   899,-32768,   652, 14547,-32768, 14979,
-32768,-32768,-32768,-32768, 14979, 14979,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,   370,-32768,   372,-32768,-32768,   374,
 15627,   269, 15825, 14979,-32768,-32768,-32768,-32768,-32768,-32768,
    94,   206,   628, 13874, 15386, 15924,-32768,-32768,-32768,    94,
   206,   628, 13982, 15494,-32768,   451,-32768,   890,-32768,   714,
-32768,   745,-32768,-32768,-32768,-32768,-32768,   487,-32768,  1776,
-32768,  1352,-32768,-32768,-32768,   534,-32768,-32768,   896,   904,
 14979, 14655,   901,   912,   902,   265,  8936,   904,   919,   -47,
  8147,   903,   916,-32768, 16518,-32768,-32768,-32768,   917,-32768,
-32768,-32768,-32768,   920, 16866,   383,-32768,   384,-32768,   389,
   404,-32768,   411,  8545,-32768,-32768, 13467,-32768, 13035,   913,
   932,   933,-32768,   969,-32768, 17499,-32768,-32768,-32768,-32768,
   415,   698,-32768,-32768,-32768,-32768,-32768,   925,   931,  8400,
    66,  2723, 11302, 11599,   950,    13,   951,   463, 12086,-32768,
   982,-32768,-32768,-32768,   936, 16994,   657, 17005,   718,  8400,
   955,   463,-32768,-32768,-32768, 17063,   657, 17074,   718,-32768,
-32768,-32768,   938,-32768,   587,-32768,   938,-32768,-32768, 12508,
  8936,   940,   939,-32768,  9060,  8936,-32768,-32768,   941,  1008,
  8936, 12398, 14763,-32768, 15087,-32768,  8936, 14871,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,  3651,    66,  3007, 16610, 16610,
  3090, 12313,   963,  5436,-32768,-32768,-32768, 14979,    85,   966,
   968,   970,   390,    66,   971,    66,   974,    66,   976,    66,
    96,   298,   979,    66,   980,-32768, 14979,   639,   655,-32768,
   673,   686,-32768,   986,   298,   991,   690,   692,-32768,   702,
   722,-32768,-32768,-32768,-32768,   998,   416,-32768,  8936,-32768,
   -59, 16930,   962, 14655,-32768,-32768,  1000,  1082,  1009,  1010,
   724, 13251, 13683,-32768,  3651,  3651,    66,-32768,-32768,-32768,
   899,-32768,   691,   282,    66,    66,-32768,-32768,   740,    66,
   749,    66,   750,    66,   760,    66,-32768,   763,   779,   761,
    66,-32768,   801,   811,   815,   816,   282,-32768,   763,   779,
   801,   811,   815,   816,-32768,-32768,-32768,-32768, 14655,   992,
 12618,  8936,-32768,-32768,  4029, 13359,-32768,-32768,   499,   821,
-32768,   827,   828,   499,-32768,   499,   832,-32768,   499,   839,
-32768,   499,   840,-32768,-32768,   844,-32768,-32768,-32768,   499,
   845,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,  8936,   993, 14655,  1011,   429,-32768,  8936,  8936,
   826,  4335,-32768,-32768,   499,-32768,   499,-32768,   499,-32768,
-32768,-32768,   499,-32768,-32768,   499,-32768,-32768,   499,   499,
-32768,-32768,-32768,-32768,   499,-32768, 14655,  1014,-32768,   997,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,   499,
-32768,  1016,  8936,  1002,-32768,-32768,-32768,-32768,-32768,  8936,
-32768,  1125,-32768
};

static const short yypgoto[] = {-32768,
-32768,  7527,-32768,  1096,-32768,  -123,-32768,   898,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,  -735,  2938,  6334,-32768,-32768,
   215,  -553,-32768,  -770,  -277,  -322,-32768,   268,   257,   313,
   256,   320,   533,   540,   538,   552,   530,-32768,  -174,  3654,
   783,  -287,  -321,  -857,     1,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,   192,-32768,-32768,   -13,
   -37,   408,  1218,  1491,  1804,   -48,  5218,-32768,   -24,  1505,
   994,  1191,  -670,   -22,  7486,   855,   -49,   -23,-32768,-32768,
-32768,-32768,   -31,  1083,-32768,   468,-32768,-32768,  -448,  2582,
   319,-32768,-32768,-32768,-32768,-32768,-32768,-32768,  2516,  -793,
-32768,  -810,-32768,-32768,   -26,   -21,   895,-32768,   470,   304,
   -27,-32768,  -513,  -367,-32768,  -610,  -589,    88,  -182,  -355,
  -713,-32768,  -140,-32768,  -212,-32768,-32768,-32768,-32768,-32768,
-32768,   169,-32768,-32768,-32768,-32768,-32768,-32768,-32768,   175,
   695,-32768,-32768,-32768,-32768,  1093,  1104,  -665,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,-32768,
-32768,-32768,-32768,-32768,-32768,-32768,-32768,  -275,-32768,-32768,
-32768,-32768,   344,-32768,-32768,-32768,-32768,-32768,  -272,-32768,
   519,   224,-32768,  3792,  2484,-32768,  3595,-32768,  5004,  7881,
  2838,-32768,   102,-32768,-32768,  1639,  5411,   854,  -396,  -240,
  4181,  4917,     9,    -1,    23,  2930,-32768,  1294,  6196,  7019,
   -19,-32768,-32768,   -65,  3517,  -116
};


#define	YYLAST		17572


static const short yytable[] = {    88,
   156,    47,   217,   229,   684,   158,   234,   152,   240,   154,
   159,   247,   923,   743,   212,   212,   814,   404,   847,   883,
   490,   624,   203,   856,   287,   783,   573,   302,   218,   230,
   160,   969,   971,   811,   625,   629,   912,   811,  -699,   225,
  1228,   622,   574,   581,     9,   521,   957,   965,   106,   106,
    88,    88,   622,    98,   977,   186,   186,   472,   480,   449,
    98,   761,   549,   926,    21,   985,   711,   626,   283,   293,
  -520,    88,   623,   265,   560,   561,   563,   564,    30,     9,
   196,    99,   574,   627,   197,     3,   515,     4,   286,   645,
   260,   622,   622,   263,     7,     8,   282,   622,     9,    21,
   257,   731,    83,    13,   262,   518,   887,  1027,   480,   450,
   451,   452,   601,    30,   630,    20,  -700,   574,    21,   156,
   736,   100,   627,   627,   158,   515,   515,   230,   627,   159,
  -700,   515,    30,   106,  1093,   622,   574,   750,    32,    33,
   186,   182,   622,   640,  1038,   602,   301,   760,   278,   303,
   815,   205,   208,   220,   232,   279,   669,  1026,   212,   672,
   582,   583,   675,   761,   256,   283,   816,   678,   282,  1084,
   289,   682,  1024,   816,    83,   494,   495,  1035,  1025,   260,
   622,   263,   414,   212,   212,   280,   274,   276,  1086,   393,
   407,    44,   101,   282,   316,  1206,   634,   635,   295,    88,
  -655,   257,   464,  -655,  -321,   931,   750,   439,   440,   465,
   658,   816,  1061,   750,  -635,   665,   865,   710,   668,   480,
   750,   671,   480,   103,   674,   480,   811,   921,   922,   677,
   480,   679,   574,   681,   480,   102,   404,  1173,  -700,   750,
   849,   202,   206,   549,  1180,   591,   592,   667,   882,   885,
   670,   304,   190,   673,   251,   458,   549,   191,   676,   786,
   787,   817,   680,   173,   574,   574,   192,   891,   704,   160,
   847,   474,   192,   204,   207,   856,  -700,   593,   594,   709,
   480,   314,   771,   574,   274,   276,   255,   427,   290,   498,
  -619,   319,   785,   751,   282,     9,   192,   301,   301,   587,
   588,   198,   393,   486,   192,   199,   192,   488,   708,   282,
  1115,   282,   393,  -699,   806,    21,  1230,   393,   261,   160,
   294,   474,   274,   276,  1061,   552,  -634,   555,   287,    30,
  1097,   156,   217,   229,   407,   240,   158,  1098,   152,   908,
   154,   159,   287,   289,  -700,   407,   407,   914,   393,   750,
   407,   407,   407,   407,  -700,   574,   268,  1029,   218,   230,
   750,   160,   751,   574,   407,   750,   750,   498,   750,   751,
   861,  1274,   192,   283,   323,   328,   751,   269,   212,   212,
    88,    88,   192,  1189,  1190,   277,  -405,   283,   888,   419,
   407,   932,   631,   287,   933,   751,   498,  -402,   589,   590,
  1061,   282,   439,   440,   191,   331,   643,   574,    50,   407,
   574,   192,   288,   274,   276,   282,  1061,  1308,  -651,   192,
  1237,   420,  -233,   595,   596,   458,  -233,   217,   229,  1055,
   160,   296,   474,   160,   938,   474,   160,   557,   474,   212,
   212,   160,  1055,   474,  1017,   160,  1019,   474,   498,  1322,
   574,  1022,   297,   218,   230,   928,   929,   458,   549,  -551,
   574,   924,   574,   192,   225,   332,   282,   574,   824,  1045,
   333,  1046,   827,  1047,   630,    88,    88,   911,   761,   160,
    50,   703,  1109,  1110,  -541,   574,   900,   574,  1111,   574,
   917,   160,   849,   474,   864,   407,   160,  -545,   574,   574,
   750,   229,  -700,  1112,   574,   751,   552,   427,   555,  1197,
  1113,   706,   407,   407,  1125,  1226,   751,   404,   407,   574,
   817,   751,   751,   527,   751,   160,   574,   230,  1310,   393,
   574,   761,    88,   980,   629,   498,   499,   981,   192,   566,
   634,   635,   407,   650,   761,   301,   411,   156,   301,   303,
   192,   552,   158,   555,   301,   230,   301,   159,   192,   982,
  -555,   498,   507,   983,   498,   738,   834,   407,   407,  1061,
   836,  -545,   407,   407,   393,   301,   192,   303,   407,   192,
   407,   407,   407,   407,   407,   407,   407,   407,   407,   407,
   407,   407,   407,   407,   407,   407,   407,   407,   407,   407,
   407,    88,  1043,  -557,   -69,   567,   568,  -547,    88,   549,
   463,   283,   323,   328,   744,    88,   498,   739,   745,   471,
   630,   160,  1060,   283,  -696,   160,   498,   740,   487,   569,
   719,   192,  1042,  -553,    88,   498,   741,     9,   407,   282,
   407,   192,   407,   -67,   570,   192,   498,   331,   393,   497,
   192,   282,  1061,  1131,  -553,   505,   751,    21,   925,   761,
   407,   192,   506,   777,   458,   509,   202,   206,   514,  1092,
   251,    30,   515,  1131,   290,   520,   498,   878,   475,   498,
   879,   458,   721,   458,   565,   458,   724,    88,   498,   880,
   407,   192,  -655,    88,   192,    47,   598,   192,   698,   699,
  1141,   192,   702,   192,  -408,  1061,   705,  -655,   -61,  -656,
   812,     3,   597,     4,   812,   498,   881,   393,   600,   393,
     7,     8,   393,   -61,  -656,    -8,    -8,   599,   475,    13,
   192,   407,    -9,    -9,   584,   585,   586,   498,  1213,   -61,
    -8,    20,   -61,   621,    88,  -408,  1061,    -9,  -631,   900,
   896,  1044,   192,   498,  1214,    88,   301,   301,   640,   407,
   303,    88,   407,    88,   905,    33,  1039,   761,   192,   404,
   905,   498,  1215,   817,   106,  1061,   407,  -649,   641,   301,
   301,   186,   407,   642,   498,  1216,   192,   644,   498,  1221,
   498,  1222,   217,   229,   666,   240,    83,   685,   247,   192,
   498,  1223,  1167,   192,   686,   192,  1192,   688,   427,  1132,
  1238,   280,  1126,   761,   160,   192,   404,   690,   218,   230,
   498,  1224,  1029,   499,   692,  1212,   407,   407,   691,  1132,
   407,  -408,  1061,   890,  -555,   192,  1061,   192,  -650,   475,
    88,    88,   475,   791,   792,   475,   795,   796,   797,   798,
   475,   788,   789,   790,   475,    63,  1100,   427,   694,  1245,
  1198,   393,  -408,  1061,   407,  -557,   427,   427,  1248,  1251,
   407,   714,   407,   407,   716,   407,   283,   427,   427,  1254,
  1259,  1061,   156,  -654,   717,   283,   155,   158,   475,   152,
   723,   154,   159,   634,   549,    88,   896,  1061,   764,  -652,
   475,   793,   794,   812,   282,   630,   216,   228,   772,   235,
   238,   905,   160,   282,   799,   800,   407,   254,   407,  1061,
  1239,  -647,   773,   407,   407,   867,  1246,    63,  1249,  1061,
  1252,  -645,  1255,  1061,  1061,  -648,  -646,  1260,   427,    88,
  1284,    88,   407,  1277,   427,   427,  1286,  1288,   549,   427,
  1168,  1292,   822,   404,    88,  1175,   427,   427,  1295,  1298,
  1178,   427,   427,  1301,  1304,   828,  1184,  1285,   829,  1287,
  1289,   832,   404,   866,  1293,   870,   872,  1296,   874,   875,
  1299,   877,   909,  1302,   890,   886,   889,   892,  1305,   407,
   407,   893,   894,   910,    58,   393,   915,   918,   920,   927,
  1032,   979,  1090,   229,  1033,   228,   238,  -408,   984,   254,
  1095,   552,   988,   555,     3,  1089,     4,  1102,   761,  1096,
  1094,  1116,   407,     7,     8,   407,  1283,   393,  1227,   230,
  1056,  1290,    13,  1291,   160,  1104,  1294,  1107,  1106,  1297,
  1117,  1118,  1120,  1056,    20,   387,  1127,  1303,  1128,  1140,
  1142,    88,  1146,  1147,  1155,   425,  1061,  1169,  1170,  1177,
  1176,   950,  1191,   959,   966,  1194,    58,  1195,    33,  1196,
  1200,   978,  1313,  1202,  1314,  1204,  1315,   106,  1209,  1211,
  1316,  1229,   986,  1317,   279,  1217,  1318,  1319,   407,   393,
  1220,  1278,  1321,   393,   393,  1174,   840,  1225,  1231,   393,
   407,   896,    50,   896,  1232,   393,   896,  1326,  1233,  1234,
  1309,  1275,  1307,  1323,   280,  1327,  1325,   905,   905,   301,
   301,  1329,   301,  1123,  1333,   479,   407,   148,  1037,   801,
   805,     3,  1306,     4,     5,     6,   803,   802,  1311,  1312,
     7,     8,   200,   575,     3,   407,     4,     5,     6,    13,
   259,   804,   835,     7,     8,   941,   469,   552,   387,   555,
   860,    20,    13,   987,   695,   266,  1101,   393,   387,     3,
  1099,     4,   407,   387,    20,   479,   267,   484,     7,     8,
    88,   896,  1328,   958,   833,    33,   530,    13,     0,  1331,
     0,    59,     0,     0,   106,     0,   251,    32,    33,    20,
     0,   186,     0,     0,   556,  1087,   290,    39,     0,     0,
     0,   552,     0,   555,   942,     0,     0,   155,    51,     0,
     0,     0,   153,    33,     0,     0,     0,   407,   702,   407,
   393,    42,   660,   301,    88,    43,   216,   228,   705,   661,
   956,   254,   778,     0,    42,     0,     0,     0,    43,     0,
    44,     0,     0,     0,     0,     0,     0,     0,     0,   944,
     0,     0,     0,    59,    58,   662,     0,     0,     0,   557,
   475,   393,     0,   407,     0,     0,     0,   393,   393,   254,
   301,     0,     0,     0,     0,     0,   479,     0,     0,   479,
    51,     0,   479,     0,    92,     0,     0,   479,     0,     0,
     0,   479,     0,     0,     0,   407,     3,     0,     4,     0,
     0,     0,     0,     0,    58,     7,     8,     0,     0,     0,
     3,   393,     4,   557,    13,    92,     0,     0,   393,     7,
     8,   216,   228,   254,     0,   479,    20,     0,    13,     0,
     0,     0,     0,   553,     0,    92,    92,   479,   840,     0,
    20,     0,   425,     0,     3,     0,     4,     5,     6,    32,
    33,     0,     0,     7,     8,     0,    92,     0,     0,   278,
     0,     0,    13,    32,    33,     0,   279,     0,     0,     0,
     0,   425,     0,   278,    20,   556,     0,     0,   387,     0,
   186,   200,     0,     3,     0,     4,     5,     6,     0,     0,
     0,     0,     7,     8,   475,   228,   238,    32,    33,     0,
   254,    13,    44,     0,     0,     0,     0,    39,     0,     0,
   280,     0,     0,    20,   942,    58,    44,     0,    58,     0,
   556,    58,     0,   228,     0,     0,    58,     0,     0,     0,
    58,     0,     0,     0,    92,     0,    32,    33,     0,     0,
   652,     0,     0,     0,    42,     0,    39,   425,    43,     0,
    44,    59,     0,   163,   425,     0,     0,     0,     0,   944,
     0,   387,     0,     0,    58,     0,     0,   425,     0,     0,
     0,   425,     0,     0,   395,     0,    58,     0,   476,   201,
   387,    52,     0,    42,    92,     0,     0,    43,   475,    44,
     0,     0,   475,   475,   387,    57,     0,     0,   475,     0,
     0,    59,     0,     0,   475,     0,     0,     3,   808,     4,
     0,     0,     0,   813,   553,     0,     7,     8,     0,     0,
     0,     0,     3,     0,     4,    13,     0,     0,   476,     0,
   554,     7,     8,   848,     0,     0,     0,    20,     0,    63,
    13,     0,     0,   153,     0,     0,     0,     0,     0,     0,
     0,     0,    20,    52,    92,     0,     0,   550,     0,   553,
    32,    33,     0,   387,     0,   387,   475,    57,   387,     0,
    39,     0,     0,     0,     0,    32,    33,   284,     0,     0,
     0,     0,     0,     0,     0,   278,     0,   395,     0,     0,
   387,     0,   186,     0,     0,     0,     0,   395,     0,     0,
     0,   387,   395,     0,    92,     0,   425,   387,     0,   387,
   904,     0,    59,    44,     0,    59,   904,     0,    59,   533,
     0,     0,     0,    59,     0,     0,     0,    59,    44,   475,
   533,   533,     0,   395,     0,   533,   533,   533,   533,   476,
     0,     0,   476,     0,     0,   476,    92,     0,     0,     0,
   476,     0,     0,     0,   476,     0,     0,     0,     0,     0,
   425,    59,     0,     0,     0,    92,    92,     0,     0,     0,
   475,     0,   845,    59,     0,   533,   475,   475,    58,     0,
     0,     0,     0,     0,     0,     0,   216,   228,   476,     0,
     0,     0,   254,     0,     0,     0,     0,     0,     0,     9,
   476,     0,   930,    10,     0,     0,     0,  1005,     0,     0,
     0,   554,     0,    16,    17,    92,   162,     0,    92,    21,
   475,    92,     0,     0,     0,     0,    92,   475,     0,     0,
    92,     0,     0,    30,     0,     0,    31,     0,   550,     0,
     0,   387,     0,     0,     0,     0,    34,    35,    36,    37,
    38,   477,     0,     0,     0,     0,   554,   904,   155,     0,
    92,    92,     0,     0,    92,    57,     0,     0,     3,     0,
     4,     5,     6,     0,     0,     0,    92,     7,     8,     0,
     0,    92,     0,   550,     0,   848,    13,  1054,     0,     0,
     0,     0,     0,     3,    53,     4,     0,     0,    20,     0,
  1054,   477,     7,     8,     0,   239,     0,     0,     0,     0,
    92,    13,     0,     0,   395,    57,     0,   395,     0,     0,
     0,    32,    33,    20,     0,   151,     0,     0,     0,     0,
     0,    39,     0,     0,    92,     0,     0,     0,   186,     0,
     0,  1005,     0,     0,     0,     0,    58,    33,     0,   254,
     0,     0,     0,     0,     0,     0,   106,     0,     0,   395,
     0,     0,    92,   284,   646,     0,    53,     0,    42,   846,
     0,     0,    43,   387,    44,    59,     0,     0,   272,     0,
  1122,     0,     0,   944,     0,     0,    92,     0,     0,     0,
     0,     0,     0,    92,     0,     0,   841,   228,   254,     0,
   395,     0,    51,     0,     0,     0,    92,     0,     0,     0,
    92,     0,   477,     0,   813,   477,     0,     0,   477,   395,
     0,     0,     0,   477,   845,     0,    57,   477,     0,    57,
     0,     0,    57,   395,     0,  1005,     0,    57,  1133,  1005,
  1005,    57,     0,     0,     0,  1005,     0,     0,     0,     0,
     0,  1005,     0,     0,     0,     0,     0,     0,     0,     0,
     0,   477,  1188,   904,   904,     0,   254,     0,     0,     0,
     0,     0,    92,   477,     0,    57,   272,     0,    92,     0,
    58,     0,     0,     0,     0,     0,     0,    57,     3,     0,
     4,     0,     0,     0,     0,     0,     0,     7,     8,     0,
     0,     0,   395,     0,   395,     0,    13,   395,   606,   609,
   612,   616,     0,  1005,   272,   620,  1193,     0,    20,   553,
     0,  1199,     0,  1201,     0,  1203,   556,  1205,   639,   395,
     0,  1210,     0,     0,     0,   533,     0,     0,     0,     0,
   395,     0,    33,    59,     0,    92,   395,   533,   395,     0,
     0,   106,     0,   620,   639,     0,     0,     0,   279,     0,
     0,     0,     0,     0,   478,     0,     0,     0,     0,     0,
   476,     0,     0,   485,    58,     0,  1005,     0,    58,    58,
   556,  1240,  1242,  1243,    58,     0,     0,  1247,     0,  1250,
    58,  1253,   272,  1256,   153,     0,     0,     0,  1261,    92,
     0,     0,     0,     0,  1240,   272,     0,     0,     0,     0,
     0,     0,     0,     0,   478,     0,     0,  1005,     0,     0,
     0,   846,     0,  1005,  1005,    92,    92,   121,   122,   123,
   124,   125,   126,   127,   128,   129,   130,     0,     0,     0,
     0,     0,     0,   551,   485,     0,   395,     0,   841,     0,
  1052,     0,    58,     0,     0,     0,   579,   147,     0,     0,
     0,     0,     0,  1052,     0,   553,     0,  1005,     0,   842,
   755,   756,     0,     0,  1005,    52,     0,    59,   609,   616,
   395,   533,     0,   844,   620,     0,   639,     0,     0,    57,
     3,     0,     4,     5,     6,     0,     0,    92,     0,     7,
     8,     9,     0,     0,   476,     0,     0,     0,    13,   272,
     0,     0,     0,     0,   272,    58,   554,     0,     0,   553,
    20,    21,     0,     0,    92,   478,    92,     0,   478,     0,
     0,   478,     0,     0,     0,    30,   478,     0,     0,    92,
   478,   755,   756,   550,    33,     0,   902,     0,     0,     0,
     0,     0,   902,   106,     0,     3,    58,     4,     5,     6,
   186,     0,    58,    58,     7,     8,     0,     0,     0,     0,
     0,    59,     0,    13,   478,    59,    59,     0,     0,     0,
   395,    59,     0,     0,     0,    20,   478,    59,     0,     0,
    42,     0,     0,     0,    43,     0,     0,     0,   476,     0,
     0,   778,   476,   476,     0,     0,    58,     0,   476,    33,
   533,     0,   395,    58,   476,     0,     0,     0,   106,    92,
     0,     0,     0,     0,   551,   186,     0,     0,     0,     0,
     0,     0,     0,     0,     0,   272,    92,     0,     3,     0,
     4,     5,     6,   477,     0,     0,     0,     7,     8,    59,
     0,   762,     0,     0,     0,    42,    13,    57,     0,    43,
     0,     0,   554,     0,     0,     0,   192,     0,    20,   551,
     0,     0,     0,     0,   395,     0,   476,     0,   395,   395,
   755,     0,     0,     0,   395,     0,   755,   756,     0,   550,
   395,    32,    33,   902,     0,     0,     0,     0,     0,     0,
     3,    39,     4,     5,     6,     0,   272,     0,   510,     7,
     8,     9,    59,     0,   485,     0,   554,     0,    13,   485,
     0,   842,     0,     0,     0,     0,     0,     0,     0,     0,
    20,    21,     0,     0,   603,   844,     0,     0,    42,   476,
     0,   272,    43,   550,    44,    30,   272,     0,     0,   192,
     0,     0,   395,    59,    33,     0,     0,     0,     9,    59,
    59,     0,    10,   106,     0,   395,   533,     0,     0,     0,
   186,     0,    16,    17,     0,   162,     0,   477,    21,     0,
   476,     0,   843,  -700,     0,     0,   476,   476,    53,     0,
     0,    57,    30,     0,     0,    31,   757,     0,     0,     0,
    42,     0,     0,    59,    43,    34,    35,    36,    37,    38,
    59,   192,     0,     0,     0,   395,     0,     0,     0,   395,
     0,     0,     0,     0,     0,     0,   755,     0,     0,     0,
   476,     0,     0,     0,     0,     0,     0,   476,     0,     3,
   485,     4,     5,     6,     0,   272,     0,     0,     7,     8,
     9,   272,     0,     0,     0,     0,   395,    13,     0,   903,
     0,     0,   395,   395,   241,   903,     0,     0,     0,    20,
    21,   477,    68,     0,   272,   477,   477,     0,     0,     0,
     0,   477,     0,     0,    30,    57,     0,   477,     0,    57,
    57,     0,    32,    33,     0,    57,  1185,     0,     0,   902,
   902,    57,    39,   157,     0,     0,   395,     0,   485,   510,
     0,   485,     0,   395,     0,     0,     3,     0,     4,     5,
     6,     0,     0,    68,    68,     7,     8,     0,     0,     0,
     0,     0,     0,     0,    13,   603,     0,   308,     0,    42,
     0,     0,     0,    43,    68,    44,    20,     0,     0,   477,
   192,     9,     0,     0,     0,    10,   478,     0,     0,   281,
     0,   285,     0,    57,     0,    16,    17,     0,   162,     0,
    33,    21,     0,     0,   308,     0,     0,     0,     0,   106,
  1135,  1137,  1139,     0,     0,    30,   186,  1145,    31,     0,
     0,   272,     0,     0,     0,     0,     0,     0,    34,    35,
    36,    37,    38,     0,     0,     0,   903,   579,     0,     0,
     0,     0,   477,     0,     0,     3,    42,     4,     5,     6,
    43,     0,   157,     0,     7,     8,    57,     0,     3,     0,
     4,   281,     0,    13,   843,     0,  1053,     7,     8,     0,
     0,     0,     0,     0,     0,    20,    13,   272,     0,  1053,
   620,     0,   639,   477,     0,     0,   272,   244,    20,   477,
   477,     0,    68,     0,     0,     0,     0,    57,    32,    33,
     0,     0,    68,    57,    57,     0,     0,     0,    39,     0,
   502,     0,    33,     0,     0,   510,     0,     0,     0,     0,
   478,   106,     0,     0,     0,   308,     0,     0,   454,     0,
   308,     0,     0,   477,     0,     0,     0,     0,     0,     0,
   477,   603,  1134,     0,     0,    42,     0,    57,     0,    43,
   485,    44,     0,     0,    57,     0,   192,     0,     0,   551,
     0,     0,     9,     0,     0,     0,    10,     0,     0,     0,
     0,     0,    68,     0,   485,     0,    16,    17,     0,   162,
     0,     0,    21,     0,     0,     0,     0,   489,     0,     0,
     0,     0,     0,     0,     0,     0,    30,     0,   188,    31,
     0,     0,     0,     0,   281,    68,     0,     0,     0,    34,
    35,    36,    37,    38,   478,    68,     0,     0,   478,   478,
    68,   308,    68,     0,   478,     0,     0,     0,     0,     0,
   478,     0,     0,     0,     0,     0,     0,     0,     0,  1186,
  1187,     0,   903,   903,     0,     0,     0,     0,     0,     0,
    90,    68,   485,     0,     0,     0,     0,   485,     0,   485,
     0,   485,     9,   485,   157,     0,    10,   485,   973,     0,
     0,     0,     0,     0,     0,     0,    16,    17,     0,   162,
     0,    90,    21,    68,    68,     0,     0,     0,     0,     0,
     0,     0,   478,     0,   281,   285,    30,     0,     0,    31,
     0,    90,    90,     0,     0,   551,   272,     0,     0,    34,
    35,    36,    37,    38,   502,   502,     0,   485,   485,   485,
     0,   311,    90,   485,     0,   485,     0,   485,     0,   485,
     0,     0,     0,    68,   485,     0,    68,   502,     0,    68,
   485,     0,     0,     0,    68,    11,    12,     0,    68,     0,
    14,     0,     0,     0,     0,   478,    18,     0,   311,   551,
     0,    22,    23,    24,    25,    26,    27,   311,   974,    28,
    29,     0,     0,     0,     0,     0,     0,     0,    68,    68,
     0,     0,    68,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,    68,     0,   478,     0,     0,    68,
    90,     0,   478,   478,     0,     0,   502,     0,     0,     0,
     0,     0,     3,     0,     4,     5,     6,     0,     0,     0,
   502,     7,     8,     9,     0,     0,     0,     0,    68,     0,
    13,     0,    68,     0,     0,    68,     0,     0,     0,   502,
   394,     0,    20,    21,     0,     0,   478,     0,   360,     0,
    90,     0,    68,   478,     0,   502,     0,    30,     0,   308,
   308,     0,     0,     0,   188,     0,    33,     0,     0,     0,
     0,     0,     0,     0,     0,   106,     0,    68,     0,   311,
   157,     0,   186,     0,   311,     0,     0,     3,     0,     4,
     0,     0,   449,     0,     0,  -700,     7,     8,     0,     0,
     0,     0,     0,     0,    68,    13,   459,     0,   762,     0,
     0,    68,    42,     0,     0,     0,    43,    20,    68,     0,
    90,     0,     0,   192,    68,     0,     0,     0,    68,     0,
     0,     0,   489,     0,     0,     0,     0,    68,     0,     0,
     0,    33,   450,   451,   452,   453,     0,     0,     0,   502,
   106,    68,     0,   394,     0,     0,     3,   454,     4,     5,
     6,   496,     0,   394,     0,     7,     8,     9,   394,   311,
    90,   513,     0,     0,    13,   311,   519,     0,     0,     0,
     0,     0,     0,     0,     0,     0,    20,    21,     0,     0,
    68,     0,     0,     0,     0,     0,    68,     0,     0,   394,
     0,    30,     0,     0,     0,     0,     0,     0,     0,    32,
    33,     0,    90,     0,     0,     0,     0,     0,     0,    39,
    68,     0,    68,     0,     0,    68,   186,     0,   502,   502,
     0,    90,    90,     0,     0,     3,     0,     4,     5,     6,
     0,     0,     0,     0,     7,     8,     0,    68,     0,     0,
     0,     0,   610,    13,     0,     0,    42,     0,    68,     0,
    43,     0,    44,    68,    68,    20,    68,   192,   188,   188,
     0,     0,     0,     0,     0,     0,   459,     0,     0,     0,
     0,    90,     0,     0,    90,     0,     0,    90,    32,    33,
     0,   188,    90,     0,     0,     0,    90,     0,    39,     0,
     0,     0,     0,     0,     0,   186,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,    68,     0,     0,
     0,     0,     0,     0,     0,     0,    90,    90,     0,     0,
    90,    41,     0,     0,     0,    42,     0,     0,     0,    43,
     0,    44,    90,    68,    68,     0,   502,    90,     0,     0,
     0,     0,     0,     0,   712,     0,     0,     0,     0,   502,
   188,     0,     0,     0,    68,     0,     0,   188,     0,     0,
     0,   732,     0,     0,   188,     0,    90,   737,     0,     0,
   394,     0,     0,   394,     0,     0,     0,  1063,  1064,     0,
     0,     0,     0,   188,     0,  1073,  1074,  1075,    68,     0,
    90,  1080,     0,   188,     0,  1081,     0,  1082,     0,   188,
     0,     0,     0,   311,   311,   157,     0,     0,     0,     0,
     0,  1088,     0,     0,     0,   394,   768,     0,    90,     0,
     0,     0,     0,   776,     0,     0,     0,    97,     0,     0,
     0,     0,    68,     0,    68,     0,     0,     0,     0,     0,
     0,     0,    90,     0,     0,     0,     0,    68,     0,    90,
     0,     0,     0,     0,     0,     0,   394,     0,   161,   502,
     0,   502,    90,     0,     0,     0,    90,     0,     0,   502,
     0,   502,     0,     0,     0,   394,     0,     0,   223,   233,
     0,     0,     0,   819,     0,     0,     0,     0,    68,   394,
     0,   819,  1150,   188,  1153,     0,   308,   823,   308,    97,
     0,   308,  1159,     0,  1162,   459,     0,     0,  1163,   831,
  1164,     0,  1165,     0,     0,     0,     0,     0,     0,     0,
    68,     0,   459,     0,   459,     9,   459,    68,    90,     0,
    11,    12,     0,     0,    90,    14,     0,     0,     0,     0,
     0,    18,     0,     0,    68,    21,    22,    23,    24,    25,
    26,    27,     0,     0,    28,    29,     0,     0,   394,    30,
   394,     0,     0,   394,     0,     0,   871,  1207,   873,     0,
     0,   876,   188,   188,     9,   502,     0,   233,    10,     0,
  1218,     0,    68,     0,     0,   394,    68,    68,    16,    17,
     0,   162,    68,     0,    21,     0,   394,     0,    68,     0,
     0,    90,   394,     0,   394,     0,     0,     0,    30,     0,
     0,    31,   819,     0,     0,     0,     0,   400,     0,  1241,
     0,    34,    35,    36,    37,    38,     0,   426,     0,   502,
     0,     0,     0,  1257,  1258,     0,     0,     0,  1262,  1263,
  1264,  1265,  1266,     0,  1267,  1268,  1269,  1270,  1271,  1272,
     0,     0,     0,     0,     0,    90,     0,     0,     0,     0,
    68,     0,     3,     0,     4,     5,     6,     0,   309,     0,
     0,     7,     8,    68,     0,   935,   937,     0,     0,   940,
    13,    90,    90,   460,     0,     0,     0,   325,   330,     0,
   188,     0,    20,     0,     0,     0,     0,   483,     0,     0,
     0,     0,   394,   188,     0,   309,     0,     0,     0,     0,
     0,     0,     0,  1016,     0,    32,    33,     0,     0,  1018,
     0,  1020,  1021,    68,  1023,    39,     0,    68,     0,     0,
   400,     0,   186,     0,     0,     0,   394,     0,     0,     0,
   400,     0,     0,     0,     0,   400,     0,   483,     0,     0,
     0,     0,     0,    90,   379,     0,     0,     0,   610,     0,
     0,     0,    42,     0,    68,  1041,    43,     0,    44,     0,
    68,    68,     0,   192,     0,     0,   558,     0,     0,     0,
    90,     0,    90,     0,     0,     0,     0,     0,     0,   161,
     9,     0,     0,     0,    10,    90,     0,     0,     0,     0,
     0,     0,     0,     0,    16,    17,     0,   162,   223,   233,
    21,   503,     0,   188,    68,   188,   325,   330,     0,     0,
     0,    68,     0,   188,    30,   188,   309,    31,     0,     0,
     0,   309,     0,     0,     0,     0,   394,    34,    35,    36,
    37,    38,     0,     0,     0,     0,     0,     0,     0,     0,
   311,     0,   311,   664,     0,   311,     0,     0,   483,     0,
     0,   483,     0,     0,   483,     0,     0,   379,   394,   483,
     0,     0,     0,   483,     0,    90,     0,   379,     0,     0,
     0,     0,   379,     0,   324,   329,     0,     0,     0,     0,
     0,     0,    90,     0,     0,     0,   975,     0,     0,     0,
     0,     0,     0,   223,   233,     0,     0,   483,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,   483,
     0,     0,   309,     0,   426,   325,   330,     0,     0,   188,
   394,     0,     0,     0,   394,   394,     0,     0,     0,     0,
   394,     3,     0,     4,     5,     6,   394,     0,     0,     0,
     7,     8,     9,   426,     0,   819,     0,   558,     0,    13,
   400,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,    20,    21,     0,     0,     0,     0,   233,     0,     0,
     0,     0,     0,   188,     0,     0,    30,     0,     0,     0,
     0,     0,     0,     0,     0,    33,     0,     0,     0,     0,
     0,     0,   558,     0,   106,   233,     0,     0,   394,     0,
     0,   186,     0,   324,   329,   503,   503,     0,     0,     0,
     0,   394,     0,     0,     0,     0,     0,     0,     0,   426,
     0,     0,     0,     0,     0,     0,   426,   757,   499,     0,
     0,    42,     0,   400,     0,    43,     0,   536,     0,   426,
     0,     0,   192,   426,     0,     0,     0,     0,     0,     0,
   379,     0,   400,     0,     0,     0,     0,     0,     0,     0,
     0,   394,     0,     0,     0,   394,   400,   379,   734,     0,
     0,   604,   607,   379,     0,     0,     0,     0,   618,     0,
     0,    85,   664,     0,     0,     0,     0,   379,     0,     0,
     0,   637,     0,     0,     0,     0,     0,     0,     0,   460,
     0,   460,   394,   460,     0,   859,     0,     0,   394,   394,
     0,    97,   324,   329,     0,   536,   618,   637,     0,     0,
   183,    85,   379,     0,     0,     0,     0,   774,   775,   379,
   183,   183,   221,   221,     0,   400,     0,   400,     0,     0,
   400,     0,     0,   183,     0,     0,   503,   325,   330,     0,
   309,   309,   394,    85,     0,     0,     0,     0,     0,   394,
     0,     0,   400,     0,     0,    85,    85,     0,     0,     0,
   807,     9,     0,   400,     0,    10,     0,   183,   426,   400,
     0,   400,     0,     0,     0,    16,    17,     0,   162,   379,
     0,    21,     0,   379,     0,   807,     0,   379,     0,     0,
     0,     0,     0,   379,     0,    30,     0,     0,    31,     0,
     0,     0,     0,     0,     0,   379,     0,     0,    34,    35,
    36,    37,    38,     0,     0,     0,     0,     0,   299,     0,
     0,   299,   426,   324,   329,     0,     0,     3,     0,     4,
     5,     6,     0,  1030,   312,     0,     7,     8,     9,     0,
     0,     0,     0,     0,     0,    13,     0,     0,   223,   233,
     0,     0,     0,   183,   183,     0,     0,    20,    21,     0,
     0,     0,   379,     0,   379,     0,   416,   379,   421,  1015,
     0,   312,    30,     0,     0,     0,   734,     0,     0,     0,
    85,    33,     0,     0,     0,     0,     0,     0,     0,     0,
   106,   221,   221,     0,     0,     0,     0,   186,     0,     0,
   379,     0,     0,   400,   899,     0,   807,     0,   379,   503,
   503,     0,     0,     0,     0,     0,     0,     0,     0,     0,
   161,     0,     0,   757,   878,     0,     0,    42,     0,     0,
     0,    43,     0,     0,   324,   329,     0,     0,   192,     9,
     0,     0,     0,    10,     0,     0,     0,   859,     0,  1059,
     0,     0,     0,    16,    17,     0,   162,     0,     0,    21,
     0,     0,  1059,     0,     0,     0,     0,     0,     0,   299,
   299,   379,   379,    30,     0,   379,    31,   312,     0,     0,
     0,     0,   312,   312,     0,     0,    34,    35,    36,    37,
    38,     0,   312,     0,   536,     0,     0,   312,     0,     0,
     0,     0,     0,  1015,     0,     0,     0,     0,     0,   379,
     0,     0,     0,     0,     0,   379,   542,   379,   379,     0,
   379,     0,     0,     0,     0,     0,     0,  1057,     0,     0,
     0,     0,     0,   324,     0,   400,     0,     0,  1067,  1069,
  1071,     0,  1124,     0,     0,   976,     0,  1077,  1079,     0,
   542,   542,   613,   613,     0,     0,     0,   542,     0,   233,
     0,   379,     0,     0,     0,     0,     0,     0,     0,     0,
   542,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,   648,     0,     0,     0,   416,   416,   653,     9,
     0,   421,   421,    10,   542,   542,   542,  1015,     0,     0,
     0,  1015,  1015,    16,    17,     0,   162,  1015,     0,    21,
     0,     0,     0,  1015,     0,     0,     0,     0,     0,   951,
     0,   960,     0,    30,     0,     0,    31,     0,     0,     0,
     0,     0,     0,     0,     0,     0,    34,    35,    36,    37,
    38,     0,     0,   421,     0,   421,   421,   416,   416,   421,
   503,     0,   503,   421,     0,     3,     0,     4,     5,     6,
   503,  1031,   503,     0,     7,     8,     9,   734,     0,     0,
     0,     0,   379,    13,     0,  1015,     0,     0,     0,     0,
     0,   312,   312,     0,     0,    20,    21,   309,   558,   309,
     0,     0,   309,     0,     0,     0,     0,     0,     0,     0,
    30,     0,     0,     0,   752,     0,     0,     0,     0,    33,
     0,     0,   542,   542,     0,     0,     0,   758,   106,     0,
   758,   758,     0,     0,     0,   186,   758,     0,   758,     0,
     0,     0,     0,     0,     0,   324,   329,     0,  1015,     0,
     0,     0,   558,     0,   324,   329,   379,   781,   807,   781,
     0,   379,     0,     0,     0,    42,     0,     0,     0,    43,
     0,     0,     0,     0,     0,     0,   503,   325,     0,     0,
     0,     0,     0,   542,     0,     0,     0,     0,     0,  1015,
   613,     0,     0,   613,   613,  1015,  1015,   752,     0,   448,
     0,     3,     0,     4,     0,     0,   449,     0,     0,     0,
     7,     8,     0,     0,     0,     0,   752,     0,     0,    13,
     0,     0,     0,     0,     0,     0,   648,     0,   648,   648,
   503,    20,   653,   653,   653,     0,   653,   653,     0,  1015,
     0,   536,     0,   607,     0,   618,  1015,     0,     0,     0,
   637,     0,     0,     0,     0,    33,   450,   451,   452,   453,
     0,   536,     0,     0,   106,     0,     0,     0,     0,   648,
     0,   454,     0,     0,     0,    85,     0,     3,     0,     4,
     5,     6,     0,     0,  -701,     0,     7,     8,     9,     0,
     0,     0,     0,   542,     0,    13,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,    20,    21,     0,
     0,     0,     0,     0,     0,     0,     0,    86,     0,     0,
     0,     0,    30,     0,     0,     0,   752,     0,     0,     0,
     0,    33,   752,     0,     0,     0,     0,   898,   758,   758,
   106,     0,   758,   898,     0,   898,     0,   186,     0,     0,
     0,     0,     0,     0,     0,     0,   184,    86,     0,     0,
     0,   781,   781,     0,     0,     0,   184,   184,   222,   222,
     0,     0,     0,   765,   324,     0,     0,    42,     0,   184,
     0,    43,     0,     0,     0,     0,     0,     0,   192,    86,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,    86,    86,     0,     0,   653,   653,     0,     0,     0,
     0,     0,     0,   184,     0,     0,     0,     0,   954,     0,
   963,   653,   648,   648,     0,     0,     0,     0,   653,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,   653,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     3,     0,     4,     5,     6,     0,     0,     0,
     0,     7,     8,     9,   300,     0,     0,   300,    11,    12,
    13,     0,     0,    14,     0,     0,     0,   898,   898,    18,
   313,     0,    20,    21,    22,    23,    24,    25,    26,    27,
     0,     0,    28,    29,     0,     0,     0,    30,     0,   184,
   184,     3,     0,     4,     5,     6,    33,     0,     0,     0,
     7,     8,   417,     0,   422,   106,     0,   313,     0,    13,
     0,   648,   186,   653,     0,     0,    86,     0,     0,     0,
     0,    20,     0,     0,   653,   653,   653,   222,   222,     0,
     0,     0,     0,   653,   653,     0,     0,     0,   763,     0,
     0,     0,    42,     0,    32,    33,    43,     0,     0,     0,
   648,     0,   653,   192,    39,     0,     0,   310,     0,     0,
     0,   163,     0,     0,     0,     0,     0,     0,     0,     0,
     0,   421,     0,     0,     0,   421,   326,   326,     0,     0,
     0,     0,     0,     0,     0,   421,     0,   164,     0,     0,
     0,    42,     0,     0,   310,    43,     0,    44,     0,     0,
     0,     0,     0,     0,     0,   300,   300,     0,     0,     0,
     0,     0,     0,   313,     0,     0,     0,     0,   313,   313,
   542,     0,   542,   613,   542,     0,     0,     0,   313,   542,
     0,     0,     0,   313,     0,     0,   653,     0,   653,     0,
   542,     0,     0,     0,     0,     0,   653,   170,   653,     0,
     0,     0,   543,     0,     0,     0,     0,     0,     0,   214,
   226,     0,   214,   236,   214,   242,   245,   214,   248,     0,
   252,     0,     0,  1183,     0,  1183,     0,     0,  1183,     0,
     0,     0,     0,     0,     0,     0,   543,   543,   614,   614,
     0,   758,   758,   543,   758,     0,     0,     0,     0,     0,
   504,     0,     0,     0,   291,     0,   543,     0,     0,     0,
     0,     0,     0,     0,   170,   523,     0,     0,   649,     0,
   525,     0,   417,   417,     0,     0,     0,   422,   422,     0,
   543,   543,   543,     0,     0,     0,     0,     0,     0,     0,
     0,     0,   421,     0,     0,     0,     0,     0,     0,     0,
     0,     0,  1183,  1183,     0,     0,     0,     0,   226,   236,
   245,   248,   252,     0,     0,     0,     0,     0,   291,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,   422,
     0,   422,   422,   417,   417,   422,     0,     0,     0,   422,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,   758,  1183,     0,     0,     0,
     0,   310,     0,     0,   326,   326,     0,   313,   313,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     3,     0,
     4,     5,     6,     0,     0,     0,     0,     7,     8,     9,
   543,     0,     0,     0,     0,   194,    13,     0,   543,   543,
     0,     0,   758,   759,     0,     0,   759,   759,    20,    21,
     0,     0,   759,     0,   759,     0,     0,     0,     0,     0,
     0,     0,     0,    30,     0,     0,     0,     0,     0,   226,
     0,     0,    33,   782,     0,   782,     0,     0,     0,     0,
     0,   106,     0,     0,     0,     0,     0,     0,   186,     0,
     0,     0,     0,     0,   728,   730,     0,     0,     3,   543,
     4,     5,     6,     0,     0,     0,   614,     7,     8,   614,
   614,     0,     0,   543,   762,     0,    13,     0,    42,     0,
     0,     0,    43,     0,     0,     0,     0,     0,    20,   192,
     0,     0,   543,     0,     0,     0,     0,     0,     0,     0,
     0,     0,   649,   170,   649,   649,     0,     0,     0,     0,
     0,    32,    33,   305,     0,   315,     0,     0,     0,     0,
     0,    39,     0,     0,     0,     0,     0,     0,   306,     0,
   320,     0,     0,     0,     0,     0,     0,   194,     0,   214,
   226,   214,   242,   245,   252,   649,     0,     0,     0,     0,
     0,    86,     0,     0,   307,     0,     0,   291,    42,     0,
     0,     0,    43,     0,    44,     0,     0,     0,     0,   543,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,   170,   252,   291,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,   504,     0,     0,     0,   523,
   525,     0,   543,     0,     0,     0,     0,     0,   543,     0,
     0,     0,     0,   759,   759,   759,     0,     0,   759,   759,
     0,   759,     0,     0,     0,     0,     0,     0,     0,     0,
   170,     0,     0,     0,   214,   226,   252,   782,   782,     0,
   291,     0,   226,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,   500,     0,     0,     0,
   508,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,   320,     0,     0,     0,
     0,     0,   526,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,   955,     0,   964,     0,   649,   649,
     0,     0,     0,     0,   170,     0,     0,     0,   226,   236,
   242,   245,   248,   252,     0,   291,     0,     3,     0,     4,
     5,     6,     0,     0,     0,     0,     7,     8,     9,     0,
     0,     0,    10,     0,     0,    13,   226,     0,     0,     0,
     0,     0,    16,    17,     0,   162,     0,    20,    21,     0,
     0,     0,     0,   759,   759,     0,     0,     0,     0,     0,
     0,     0,    30,   315,     0,    31,     0,     0,   728,   730,
    32,    33,     0,     0,   659,    34,    35,    36,    37,    38,
    39,   952,     0,   961,   967,     0,     0,   942,     0,     0,
     0,   967,     0,     0,     0,     0,     0,   649,     0,     0,
     0,     0,   967,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,   943,     0,     0,     0,    42,     0,     0,
     0,    43,     0,    44,     0,     0,     0,     0,     0,     0,
     0,     0,   944,     0,     0,     3,   649,     4,     5,     6,
     0,     0,     0,     0,     7,     8,     9,     0,     0,     0,
     0,   715,     0,    13,   718,   720,   722,   422,     0,   725,
   726,   422,     0,     0,     0,    20,    21,     0,     0,     0,
   170,   422,   500,   508,   500,   508,     0,     0,     0,     0,
    30,     0,     0,     0,   749,   194,  1058,     0,    32,    33,
     0,     0,     0,     0,     0,     0,     0,     0,    39,  1072,
     0,     0,     0,     0,     0,   186,   543,     0,   543,   614,
   543,     0,     0,     0,     0,   543,     0,     0,     3,     0,
     4,     5,     6,     0,     0,   967,   543,     7,     8,     9,
     0,    41,     0,     0,     0,    42,    13,     0,     0,    43,
     0,    44,     0,     0,     0,     0,     0,     0,    20,    21,
     0,     0,     0,   749,     0,     0,     0,     0,     0,     0,
   749,     0,     0,    30,     0,     0,     0,   749,     0,     0,
     0,    32,    33,     0,     0,     0,     0,   759,   759,     0,
   759,    39,     0,     0,     0,     0,   749,     0,   163,     0,
     0,     0,     0,     0,     0,   170,     0,   194,     0,   214,
   226,   214,   242,   245,   214,   252,     0,     0,     0,  1149,
     0,  1152,     0,     0,   322,     0,   291,     0,    42,  1158,
     0,  1161,    43,     0,    44,     0,     0,     0,   422,     0,
     0,     3,     0,     4,     5,     6,     0,     0,     0,     0,
     7,     8,     9,     0,     0,     0,   310,     0,   310,    13,
     0,   310,     0,     0,     0,     0,     0,     0,     0,   214,
   226,    20,    21,     0,     0,     0,   868,   869,     0,     0,
     0,     0,     0,     0,     0,     0,    30,   715,   718,   715,
   718,     0,     0,     0,    32,    33,     0,     0,     0,     0,
     0,   759,     0,     0,    39,     0,   749,     0,     0,     0,
     3,   186,     4,     5,     6,     0,     0,   749,     0,     7,
     8,     0,   749,   749,     0,   749,     0,     0,    13,     0,
     0,     0,     0,     0,     0,   504,     0,   210,     0,     0,
    20,    42,     0,     0,     0,    43,    93,    44,   759,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,    32,    33,     0,     0,     0,   170,     0,
     0,     0,   252,    39,     0,     0,     0,     0,     0,     0,
   163,     0,   291,     0,     0,    93,    93,     0,     0,   728,
     0,     0,     0,     0,     0,    93,    93,    93,    93,     0,
     0,     0,     0,     0,     0,     0,   327,     0,    93,     0,
    42,     0,     0,     0,    43,     0,    44,   170,    93,     0,
   226,   252,     0,     0,     0,     0,   291,     0,     0,     0,
    93,    93,     0,     0,     0,     0,     0,   170,     0,     0,
     0,     0,    93,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,  1028,     0,   726,     0,   749,     0,     0,
     0,     0,     0,     0,     3,     0,     4,     5,     6,     0,
     0,     0,     0,     7,     8,     0,     0,     0,     0,     0,
     0,     0,    13,     0,     0,     0,     0,     0,   170,   252,
     0,   291,     0,     0,    20,     0,     0,     0,     0,     0,
     0,     0,   726,     0,     0,     0,     0,     0,     0,    93,
     0,     0,     0,     0,     0,     0,     0,    32,    33,     0,
     0,     0,     0,     0,     0,     0,     0,    39,    93,    93,
     0,     0,     0,     0,   186,     0,     0,     0,     0,   170,
     0,    93,     0,    93,     0,     0,    93,     0,     0,     0,
     0,     0,   214,   226,   226,    93,     0,     0,     0,     0,
   412,     2,     0,     3,    42,     4,    93,    93,    43,     0,
    44,     0,     7,     8,     9,     0,     0,     0,    10,    11,
    12,    13,     0,     0,    14,     0,     0,    15,    16,    17,
    18,   162,     0,    20,    21,    22,    23,    24,    25,    26,
    27,     0,   170,    28,    29,     0,     0,     0,    30,     0,
  1130,    31,     0,     0,     0,     0,     0,    33,  1143,     0,
     0,    34,    35,    36,    37,    38,   106,     0,     0,     0,
  1154,     0,  1156,   473,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,  -701,     0,     0,   170,
     0,     0,    93,     0,     0,     0,     0,    93,    93,     0,
     0,     0,     0,     0,     0,     0,     0,    93,     0,     0,
     0,     0,    93,     0,     0,   403,     0,     0,     0,     0,
     3,   471,     4,     5,     6,     0,     0,     0,     0,     7,
     8,    93,     0,     0,     0,     0,     0,     0,    13,     0,
     0,     0,  1208,     0,     0,     0,     0,     0,   715,   718,
    20,   715,   718,     0,     0,  1219,     0,   715,   718,     0,
   715,   718,     0,     0,     0,    93,    93,    93,    93,     0,
     0,     0,    93,    32,    33,     0,     0,     0,     0,   632,
     0,   500,     0,    39,     0,    93,     0,     0,     0,     0,
   163,     0,     0,   632,     0,     0,     0,    93,     0,     0,
     0,    93,    93,    93,     0,     0,    93,    93,     0,    93,
    93,    93,     0,     0,     0,     0,   418,     0,     0,     0,
    42,     0,     0,     0,    43,     0,    44,     0,     0,     0,
     0,     0,     0,     0,     0,  1281,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,    93,   532,
    93,    93,    93,    93,    93,     0,     0,     0,    93,     0,
   545,   546,     0,     0,     0,   403,   403,   403,   403,     0,
     0,     0,  1281,     0,     0,     0,     0,     0,     0,   403,
     0,     0,     0,     0,     0,     0,    93,    93,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     2,     0,     3,   403,     4,     5,     6,   753,
     0,     0,     0,     7,     8,     9,     0,    93,    93,    10,
    11,    12,    13,     0,   403,    14,     0,     0,    15,    16,
    17,    18,   162,     0,    20,    21,    22,    23,    24,    25,
    26,    27,     0,     0,    28,    29,     0,     0,     0,    30,
     0,     0,    31,     0,     0,     0,     0,    32,    33,     0,
     0,     0,    34,    35,    36,    37,    38,    39,     0,     0,
     0,     0,     0,     0,    40,     0,     0,     0,    93,     0,
     0,     0,     0,     0,     0,    93,     0,  -701,    93,    93,
     0,     0,   753,     0,     0,     0,     0,     0,     0,     0,
    41,     0,     0,     0,    42,     0,     0,     0,    43,     0,
    44,   753,     0,     0,     0,     0,     0,     0,     0,     0,
     0,    93,   264,    93,    93,     0,     0,    93,    93,    93,
     0,    93,    93,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,    93,     0,     0,     0,     0,     0,
    93,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,    93,     0,
     0,     0,     0,     0,     0,   403,   403,   403,   403,   403,
   403,   403,   403,   403,   403,   403,   403,   403,   403,   403,
   403,   403,   403,   403,   403,     0,     0,     0,     0,     0,
     0,   753,     0,     0,     0,     0,     0,   753,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,   906,     0,     0,     0,     0,     0,   906,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     3,     0,     4,     5,
     6,     0,     0,     0,     0,     7,     8,     9,     0,     0,
     0,    10,     0,     0,    13,     0,     0,     0,     0,     0,
     0,    16,    17,     0,   162,     0,    20,    21,     0,    94,
    93,    93,     0,     0,     0,   403,     0,     0,     0,     0,
     0,    30,     0,    93,    31,    93,    93,    93,    93,    32,
    33,     0,     0,    93,    34,    35,    36,    37,    38,    39,
     0,     0,     0,     0,    93,     0,   942,     0,    94,    94,
     0,     0,     0,     0,     0,     0,     0,     0,    94,    94,
    94,    94,     0,     0,     0,     0,     0,     0,     0,     0,
     0,    94,   652,     0,     0,   403,    42,     0,     0,     0,
    43,    94,    44,     0,     0,     0,     0,   403,     0,     0,
     0,   944,     0,    94,    94,     0,     0,     0,   906,     3,
     0,     4,     5,     6,     0,    94,     0,   403,     7,     8,
     9,     0,     0,     0,    10,     0,     0,    13,     0,     0,
     0,     0,     0,     0,    16,    17,    93,   162,    93,    20,
    21,     0,     0,     0,     0,     0,     0,     0,     0,    93,
    93,    93,     0,     0,    30,     0,     0,    31,    93,    93,
     0,     0,    32,    33,     0,     0,     0,    34,    35,    36,
    37,    38,    39,     0,     0,    93,     0,    93,     0,   510,
     0,     0,    94,     0,     0,     0,     0,     0,     0,     0,
     0,     0,  -700,     0,     0,     0,    93,     0,     0,     0,
    93,    94,    94,     0,     0,   603,   499,     0,     0,    42,
    93,     0,     0,    43,    94,    44,    94,     0,     0,    94,
   192,     0,     0,     0,     0,     0,     0,     0,    94,     0,
     0,   403,     0,     0,     0,     0,     0,     0,     0,    94,
    94,     0,     0,     0,     0,    93,     0,    93,    93,    93,
     0,     0,     0,     0,    93,     0,     0,     0,   403,   403,
     0,    93,     0,    93,     0,    93,     0,     0,     0,     0,
     0,    93,     0,    93,     0,     0,     0,   403,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,   753,     0,
   753,     0,     3,   753,     4,     5,     6,     0,     0,     0,
     0,     7,     8,     9,   906,   906,     0,     0,    11,    12,
    13,     0,     0,    14,   403,    94,    15,     0,     0,    18,
    94,    94,    20,    21,    22,    23,    24,    25,    26,    27,
    94,     0,    28,    29,     0,    94,     0,    30,     0,     0,
     0,     0,     0,     0,     0,     0,    33,     0,     0,     0,
   403,     0,     0,     0,    94,   106,     0,    93,     0,     0,
     0,     0,   224,     0,     0,     0,     0,   753,   753,     0,
     0,     0,     0,     0,     0,  -701,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,    94,    94,
    94,    94,    42,     0,     0,    94,    43,     0,     0,     0,
     0,     0,   633,   778,     0,     0,     0,     0,    94,     0,
     0,     0,     0,     0,     0,     0,   633,     0,     0,     0,
    94,   753,     0,     0,    94,    94,    94,     0,     0,    94,
    94,     0,    94,    94,    94,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,   403,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
   403,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,    94,     0,    94,    94,    94,    94,    94,     0,     0,
     0,    94,     3,     0,     4,     5,     6,     0,     0,     0,
     0,     7,     8,     9,     0,     0,   403,     0,     0,     0,
    13,     0,     0,     0,     0,   171,     0,    45,     0,    94,
    94,     0,    20,    21,     0,     0,     0,   215,   227,     0,
   215,   237,   215,   243,   246,   215,   249,    30,   253,     0,
     0,     0,   754,     0,     0,    32,    33,     0,     0,     0,
    94,    94,     0,     0,     0,    39,   165,   187,     0,     0,
     0,     0,   510,     0,     0,     0,   165,   165,   165,   165,
     0,     0,   292,     0,     0,     0,     0,     0,     0,   165,
     0,     0,     0,     0,     0,     0,     0,     0,   511,    45,
     0,     0,    42,     0,     0,     0,    43,     0,    44,     0,
     0,    45,    45,     0,     0,     3,     0,     4,     5,     6,
     0,    94,     0,   165,     7,     8,     9,     0,    94,     0,
     0,    94,    94,    13,     0,   754,     0,     0,     0,     0,
     0,     0,     0,     0,     0,    20,    21,     0,     0,     0,
     0,     0,     0,     0,   754,     0,     0,     0,     0,     0,
    30,     0,     0,     0,    94,     0,    94,    94,    32,    33,
    94,    94,    94,     0,    94,    94,     0,     0,    39,     0,
     0,     0,     0,     0,     0,   186,     0,     0,     0,     0,
   187,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,    94,     0,   165,
   165,   412,     0,    94,     0,    42,     0,     0,     0,    43,
     0,    44,   413,     0,   413,     0,     0,   187,     0,     0,
     0,    94,     0,     0,     0,     0,   187,     0,     0,     0,
     0,     3,     0,     4,     5,     6,     0,   165,   165,     0,
     7,     8,     9,     0,     0,     0,    10,    11,    12,    13,
     0,     0,    14,     0,   754,    15,    16,    17,    18,   162,
   754,    20,    21,    22,    23,    24,    25,    26,    27,     0,
     0,    28,    29,     0,   907,     0,    30,     0,     0,    31,
   907,     0,     0,     0,    32,    33,     0,     0,     0,    34,
    35,    36,    37,    38,    39,     0,     0,     0,     0,     0,
     0,   209,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,  -701,     0,     0,     0,     0,     0,
     0,   171,     0,   187,     0,     0,     0,   610,    45,    45,
     0,    42,     0,    94,    94,    43,     0,    44,   187,     0,
     0,     0,   192,   187,     0,     0,    94,     0,    94,    94,
    94,    94,     0,     0,     0,     0,    94,   215,   227,   215,
   243,   246,   253,     0,     0,     0,     0,    94,     0,     0,
     0,    80,     0,     0,     0,   292,     0,     3,     0,     4,
     5,     6,     0,     0,     0,     0,     7,     8,     0,     0,
     0,     0,     0,     0,     0,    13,     0,     0,     0,   171,
   253,   292,     0,     0,     0,     0,     3,    20,     4,     5,
     6,     0,     0,     0,     0,     7,     8,     0,     0,     0,
     0,   907,   219,   231,    13,     0,     0,     0,   647,     0,
    32,    33,   413,   413,   647,     0,    20,   413,   413,     0,
    39,     0,     0,    80,     0,     0,     0,   186,   171,    94,
     0,    94,   215,   227,   253,   273,   275,     0,   292,    32,
    33,     0,    94,    94,    94,     0,     0,     0,     0,    39,
     0,    94,    94,   210,     0,     0,   501,    42,     0,     0,
     0,    43,     0,    44,     0,     0,     0,     0,    94,   413,
    94,   413,   413,   413,   413,   413,     0,     0,     0,   413,
     0,     0,   307,     0,     0,     0,    42,     0,     0,    94,
    43,     0,    44,    94,     0,     0,     0,     0,     0,     0,
     0,     0,     0,    94,     0,     0,     0,   187,   187,     0,
     0,     0,     0,     0,     3,     0,     4,     5,     6,     0,
     0,     0,     0,     7,     8,     9,     0,     0,     0,     0,
   647,     0,    13,   273,   275,     0,     0,     0,    94,     0,
    94,    94,    94,     0,    20,    21,   415,    94,     0,     0,
     0,     0,     0,     0,    94,     0,    94,     0,    94,    30,
     0,     0,     0,     0,    94,     0,    94,    32,    33,     0,
     0,   273,   275,     0,     0,     0,     0,    39,     0,     0,
     0,     0,     0,     0,   163,     0,     0,     0,     0,     0,
     0,   754,     0,   754,     0,     0,   754,     0,     0,   647,
     0,     0,     0,     0,     0,     0,   647,   907,   907,     0,
   656,     0,     0,   647,    42,     0,     0,     0,    43,     3,
    44,     4,     5,     6,     0,     0,     0,     0,     7,     8,
     9,     0,   647,     0,    10,     0,     0,    13,     0,     0,
     0,     0,   647,     0,    16,    17,     0,   162,   647,    20,
    21,     0,   647,   647,     0,     0,     0,     0,     0,     0,
    94,     0,   273,   275,    30,     0,     0,    31,   171,     0,
   754,   754,    32,    33,     0,     0,     0,    34,    35,    36,
    37,    38,    39,     0,     0,     0,     0,     0,     0,   163,
     3,    45,     4,     5,     6,     0,   537,     0,     0,     7,
     8,     9,  -700,     0,     0,     0,     0,     0,    13,     0,
     0,     0,     0,     0,     0,  1103,     0,     0,     0,    42,
    20,    21,     0,    43,   754,    44,     0,     0,     0,     0,
   605,   608,   611,   615,     0,    30,     0,   619,     0,     0,
     0,     0,   647,    32,    33,     0,     0,     0,     0,     0,
   638,     0,     0,    39,     0,     0,     0,     0,     0,     0,
   186,     0,     0,     0,     0,     0,   273,   275,     0,     0,
     0,   273,   275,     0,   537,   619,   638,     0,     0,     0,
     0,     0,     0,     0,     0,     0,   646,     0,     0,     0,
    42,     0,     0,   171,    43,     0,    44,   215,   227,   215,
   243,   246,   215,   253,     0,     0,     2,     0,     3,     0,
     4,     5,     6,     0,   292,     0,     0,     7,     8,     9,
     0,   647,   647,    10,    11,    12,    13,   700,   701,    14,
     0,     0,    15,    16,    17,    18,    19,     0,    20,    21,
    22,    23,    24,    25,    26,    27,     0,     0,    28,    29,
     0,     0,     0,    30,     0,     0,    31,   215,   227,     0,
     0,    32,    33,     0,     0,     0,    34,    35,    36,    37,
    38,    39,     3,     0,     4,     5,     6,     0,    40,     0,
     0,     7,     8,     9,     0,     0,     0,    10,     0,     0,
    13,  -701,   273,   275,     0,     0,     0,    16,    17,     0,
   162,     0,    20,    21,    41,     0,     0,     0,    42,     0,
     0,     0,    43,     0,    44,     0,     0,    30,     0,     0,
    31,     0,     0,     0,     0,    32,    33,  -531,     0,     0,
    34,    35,    36,    37,    38,    39,     0,     0,     0,   647,
     0,     0,   510,     0,     0,     0,     0,     0,     0,     0,
     0,     0,   647,     0,     0,  -700,   171,     0,     0,     0,
   253,     0,     0,   273,   275,     0,     0,     0,   603,  1129,
   292,     0,    42,     0,     0,     0,    43,     0,    44,     0,
     0,     0,     0,   192,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,   413,   273,   275,
     0,   413,     0,   273,   275,   171,     0,     0,   227,   253,
     0,   413,     0,     0,   292,     0,     0,     3,     0,     4,
     5,     6,     0,     0,     0,   171,     7,     8,     0,     0,
   334,   335,     0,    11,    12,    13,     0,     0,    14,   858,
     0,     0,     0,     0,    18,    80,     0,    20,     0,    22,
    23,    24,    25,    26,    27,     0,     0,    28,    29,     0,
     0,   336,   647,   537,   647,     0,     0,   337,   338,   339,
    32,    33,   647,     0,   647,     0,     0,     0,     0,     0,
    39,   340,   341,   342,   343,   344,   345,   401,     0,   347,
   348,     0,     0,     0,     0,     0,     0,     0,     0,   647,
     0,   647,   273,     0,   647,     0,     0,     0,     0,     0,
     0,     0,     0,   350,     0,   352,   353,    42,     0,     0,
     0,    43,     0,   354,   355,     0,     0,   171,     0,     0,
     0,     0,     0,     9,     0,   733,  1114,    10,    11,    12,
   215,   227,     0,    14,     0,     0,     0,    16,    17,    18,
   162,     0,     0,    21,    22,    23,    24,    25,    26,    27,
     0,     0,    28,    29,     0,     0,     0,    30,   413,     0,
    31,     0,     0,     0,     0,     0,     0,     0,   647,     0,
    34,    35,    36,    37,    38,     0,     0,     0,   953,     0,
   962,   968,   970,   972,     0,     0,     0,     0,   968,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,   968,
     0,     0,     0,     0,     0,     0,     0,     0,     2,     0,
     3,     0,     4,     5,     6,     0,     0,     0,     0,     7,
     8,     9,   647,   334,   335,    10,    11,    12,    13,   989,
     0,    14,   990,   991,    15,    16,    17,    18,   162,   992,
    20,    21,    22,    23,    24,    25,    26,    27,   993,   994,
    28,    29,   995,   996,   336,    30,   997,   998,    31,   999,
   337,   338,   339,    32,    33,     0,     0,     0,    34,    35,
    36,    37,    38,  1000,   340,   341,   342,   343,   344,   345,
  1001,   858,   347,   348,     0,     0,     0,     0,     0,     0,
     0,     0,     0,  -701,   273,   275,     0,     0,     0,     0,
     0,     0,     0,   273,   275,     0,   350,     0,   352,   353,
    42,     0,     0,     0,    43,     0,   354,   355,     0,     0,
  1085,     3,   968,     4,     0,     0,     0,  -180,   471,  -492,
     7,     8,     9,     0,     0,     0,    10,    11,    12,    13,
     0,     0,    14,     0,     0,    15,    16,    17,    18,   162,
     0,    20,    21,    22,    23,    24,    25,    26,    27,     0,
     0,    28,    29,     0,     0,     0,    30,     0,     0,    31,
     0,     0,     0,     0,     0,    33,     0,     0,     0,    34,
    35,    36,    37,    38,   106,     0,     0,     0,     0,     0,
   537,   713,   608,   615,   619,     0,     2,     0,     3,   638,
     4,     5,     6,   349,     0,     0,     0,     7,     8,     9,
   537,   334,   335,    10,    11,    12,    13,   989,   351,    14,
   990,   991,    15,    16,    17,    18,   162,   992,    20,    21,
    22,    23,    24,    25,    26,    27,   993,   994,    28,    29,
   995,   996,   336,    30,   997,   998,    31,   999,   337,   338,
   339,    32,    33,     0,     0,     0,    34,    35,    36,    37,
    38,  1000,   340,   341,   342,   343,   344,   345,  1001,     0,
   347,   348,     0,     0,     0,     0,     0,     0,     0,     0,
     0,  -701,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,   350,     0,   352,   353,    42,     0,
     0,     0,    43,     0,   354,   355,     0,     0,     0,     0,
     0,     0,     0,     0,     0,  -180,   471,     0,     0,     0,
  1171,     0,     3,   273,     4,     5,     6,     0,     0,     0,
     0,     7,     8,     9,     0,   334,   335,    10,    11,    12,
    13,     0,     0,    14,     0,     0,    15,    16,    17,    18,
   162,     0,    20,    21,    22,    23,    24,    25,    26,    27,
     0,     0,    28,    29,     0,     0,   336,    30,     0,     0,
    31,     0,   337,   338,   339,    32,    33,     0,     0,     0,
    34,    35,    36,    37,    38,    39,   340,   341,   342,   343,
   344,   345,  1172,     0,   347,   348,     0,     0,     0,     0,
     0,     0,     0,     0,     0,  -701,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,   350,     0,
   352,   353,    42,     0,     0,     0,    43,     0,   354,   355,
     0,     3,     0,     4,     5,     6,   449,     0,     0,  -180,
     7,     8,     9,     0,     0,     0,    10,    11,    12,    13,
     0,     0,    14,     0,     0,    15,    16,    17,    18,   162,
     0,    20,    21,    22,    23,    24,    25,    26,    27,     0,
     0,    28,    29,     0,     0,     0,    30,     0,     0,    31,
     0,     0,     0,     0,    32,    33,   450,   451,   452,    34,
    35,    36,    37,    38,    39,     0,     0,     0,     0,     0,
     0,   838,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     3,     0,     4,     5,     6,     0,     0,     0,
     0,     7,     8,     0,     0,   334,   335,   646,    11,    12,
    13,    42,     0,    14,     0,    43,     0,    44,     0,    18,
     0,     0,    20,     0,    22,    23,    24,    25,    26,    27,
   839,     0,    28,    29,     0,     0,   336,     0,     0,     0,
     0,     0,   337,   338,   339,    32,    33,     0,     0,     0,
     0,     0,     0,     0,     0,    39,   340,   341,   342,   343,
   344,   345,   401,     0,   347,   348,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,   350,     0,
   352,   353,    42,     0,     0,     0,    43,     0,   354,   355,
   250,     0,     3,     0,     4,     5,     6,     0,     0,     0,
   733,     7,     8,     9,     0,     0,     0,    10,    11,    12,
    13,     0,     0,    14,     0,     0,     0,    16,    17,    18,
   162,     0,    20,    21,    22,    23,    24,    25,    26,    27,
     0,     0,    28,    29,     0,     0,     0,    30,     0,     0,
    31,     0,     0,     0,     0,    32,    33,     0,     0,     0,
    34,    35,    36,    37,    38,    39,     0,     0,     0,     0,
     0,     0,   942,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     3,     0,     4,     0,   652,     0,
     0,     0,    42,     7,     8,     9,    43,     0,    44,     0,
    11,    12,    13,     0,     0,    14,     0,   944,    15,     0,
     0,    18,     0,     0,    20,    21,    22,    23,    24,    25,
    26,    27,     0,     0,    28,    29,     0,     0,     0,    30,
     0,     0,     0,     0,   104,   105,     0,     0,    33,     0,
     0,     0,     0,     0,     0,     0,     0,   106,     0,     0,
     0,     0,     0,     0,   107,   108,   109,   110,   111,   112,
   113,   114,   115,   116,   117,   118,     0,     0,   119,   120,
   121,   122,   123,   124,   125,   126,   127,   128,   129,   130,
   131,     0,   132,   133,   134,   135,   136,   137,   138,   139,
   140,   141,   142,   143,   144,   145,     0,   146,     0,     3,
   147,     4,     5,     6,     0,     0,     0,     0,     7,     8,
     9,     0,     0,     0,    10,    11,    12,    13,     0,     0,
    14,     0,     0,    15,    16,    17,    18,   162,     0,    20,
    21,    22,    23,    24,    25,    26,    27,     0,     0,    28,
    29,     0,     0,     0,    30,     0,     0,    31,     0,     0,
     0,     0,    32,    33,     0,     0,     0,    34,    35,    36,
    37,    38,    39,     0,     0,     0,     0,     0,     0,   209,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,   646,     0,     0,     0,    42,
     0,     0,     0,    43,     0,    44,     3,     0,     4,     5,
     6,     0,     0,     0,   944,     7,     8,     9,     0,     0,
     0,    10,    11,    12,    13,     0,     0,    14,     0,     0,
    15,    16,    17,    18,   162,     0,    20,    21,    22,    23,
    24,    25,    26,    27,     0,     0,    28,    29,     0,     0,
     0,    30,     0,     0,    31,     0,     0,     0,     0,    32,
    33,     0,     0,     0,    34,    35,    36,    37,    38,    39,
     0,     0,     0,     0,     0,     0,   224,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,   646,     0,     0,     0,    42,     0,     0,     0,
    43,     0,    44,     3,     0,     4,     5,     6,     0,     0,
     0,   944,     7,     8,     0,     0,   334,   335,     0,    11,
    12,    13,     0,     0,    14,     0,     0,     0,     0,     0,
    18,     0,     0,    20,     0,    22,    23,    24,    25,    26,
    27,     0,     0,    28,    29,     0,     0,   336,     0,     0,
     0,     0,     0,   337,   338,   339,    32,    33,     0,     0,
     0,     0,     0,     0,     0,     0,    39,   340,   341,   342,
   343,   344,   345,   401,     0,   347,   348,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,   350,
     0,   352,   353,    42,     0,     0,     0,    43,     0,   354,
   355,     0,     0,     0,     3,   402,     4,     5,     6,     0,
     0,     0,     0,     7,     8,     0,     0,   334,   335,     0,
    11,    12,    13,     0,     0,    14,     0,     0,     0,     0,
     0,    18,     0,     0,    20,     0,    22,    23,    24,    25,
    26,    27,     0,     0,    28,    29,     0,     0,   336,     0,
     0,     0,     0,     0,   337,   338,   339,    32,    33,     0,
     0,     0,     0,     0,     0,     0,     0,    39,   340,   341,
   342,   343,   344,   345,   401,     0,   347,   348,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
   350,     0,   352,   353,    42,     0,     0,     0,    43,     0,
   354,   355,     0,     0,     0,     3,   784,     4,     5,     6,
     0,     0,     0,     0,     7,     8,     0,     0,   334,   335,
     0,    11,    12,    13,     0,     0,    14,     0,     0,     0,
     0,     0,    18,     0,     0,    20,     0,    22,    23,    24,
    25,    26,    27,     0,     0,    28,    29,     0,     0,   336,
     0,     0,     0,     0,     0,   337,   338,   339,    32,    33,
     0,     0,     0,     0,     0,     0,     0,     0,    39,   340,
   341,   342,   343,   344,   345,   401,     0,   347,   348,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,   350,     0,   352,   353,    42,     0,     0,     0,    43,
     0,   354,   355,     0,     0,     0,     3,   916,     4,     5,
     6,     0,     0,     0,     0,     7,     8,     9,     0,   334,
   335,    10,    11,    12,    13,     0,     0,    14,     0,     0,
    15,    16,    17,    18,   162,     0,    20,    21,    22,    23,
    24,    25,    26,    27,     0,     0,    28,    29,     0,     0,
   336,    30,     0,     0,    31,     0,   337,   338,   339,    32,
    33,     0,     0,     0,    34,    35,    36,    37,    38,    39,
   340,   341,   342,   343,   344,   345,   346,     0,   347,   348,
     0,     0,     0,     0,     0,     0,     0,     0,   349,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,   746,   747,   352,   353,    42,     0,     0,     0,
    43,     0,   354,   355,     0,     0,     3,   192,     4,     5,
     6,     0,     0,     0,     0,     7,     8,     9,     0,   334,
   335,    10,    11,    12,    13,     0,     0,    14,     0,     0,
    15,    16,    17,    18,   162,     0,    20,    21,    22,    23,
    24,    25,    26,    27,     0,     0,    28,    29,     0,     0,
   336,    30,     0,     0,    31,     0,   337,   338,   339,    32,
    33,     0,     0,     0,    34,    35,    36,    37,    38,    39,
   340,   341,   342,   343,   344,   345,   346,     0,   347,   348,
     0,     0,     0,     0,     0,     0,     0,     0,   349,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,   746,   351,   352,   353,    42,     0,     0,     0,
    43,     0,   354,   355,     0,     0,     3,   192,     4,     5,
     6,     0,     0,     0,     0,     7,     8,     9,     0,   334,
   335,    10,    11,    12,    13,     0,     0,    14,     0,     0,
    15,    16,    17,    18,   162,     0,    20,    21,    22,    23,
    24,    25,    26,    27,     0,     0,    28,    29,     0,     0,
   336,    30,     0,     0,    31,     0,   337,   338,   339,    32,
    33,     0,     0,     0,    34,    35,    36,    37,    38,    39,
   340,   341,   342,   343,   344,   345,   346,     0,   347,   348,
     0,     0,     0,     0,     0,     0,     0,     0,   349,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,   746,   818,   352,   353,    42,     0,     0,     0,
    43,     0,   354,   355,     0,     0,     3,   192,     4,     5,
     6,     0,     0,     0,     0,     7,     8,     9,     0,   334,
   335,    10,    11,    12,    13,     0,     0,    14,     0,     0,
    15,    16,    17,    18,   162,     0,    20,    21,    22,    23,
    24,    25,    26,    27,     0,     0,    28,    29,     0,     0,
   336,    30,     0,     0,    31,     0,   337,   338,   339,    32,
    33,     0,     0,     0,    34,    35,    36,    37,    38,    39,
   340,   341,   342,   343,   344,   345,   346,     0,   347,   348,
     0,     0,     0,     0,     0,     0,     0,     0,   349,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,   897,   747,   352,   353,    42,     0,     0,     0,
    43,     0,   354,   355,     0,     0,     3,   192,     4,     5,
     6,     0,     0,     0,     0,     7,     8,     9,     0,   334,
   335,    10,    11,    12,    13,     0,     0,    14,     0,     0,
    15,    16,    17,    18,   162,     0,    20,    21,    22,    23,
    24,    25,    26,    27,     0,     0,    28,    29,     0,     0,
   336,    30,     0,     0,    31,     0,   337,   338,   339,    32,
    33,     0,     0,     0,    34,    35,    36,    37,    38,    39,
   340,   341,   342,   343,   344,   345,   346,     0,   347,   348,
     0,     0,     0,     0,     0,     0,     0,     0,   349,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,   897,   351,   352,   353,    42,     0,     0,     0,
    43,     0,   354,   355,     0,     0,     3,   192,     4,     5,
     6,     0,     0,     0,     0,     7,     8,     9,     0,   334,
   335,    10,    11,    12,    13,     0,     0,    14,     0,     0,
    15,    16,    17,    18,   162,     0,    20,    21,    22,    23,
    24,    25,    26,    27,     0,     0,    28,    29,     0,     0,
   336,    30,     0,     0,    31,     0,   337,   338,   339,    32,
    33,     0,     0,     0,    34,    35,    36,    37,    38,    39,
   340,   341,   342,   343,   344,   345,   346,     0,   347,   348,
     0,     0,     0,     0,     0,     0,     0,     0,   349,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,   897,   818,   352,   353,    42,     0,     0,     0,
    43,     0,   354,   355,     0,     0,     3,   192,     4,     5,
     6,     0,     0,     0,     0,     7,     8,     9,     0,   334,
   335,     0,    11,    12,    13,     0,     0,    14,     0,     0,
     0,     0,     0,    18,     0,     0,    20,    21,    22,    23,
    24,    25,    26,    27,     0,     0,    28,    29,     0,     0,
   336,    30,     0,     0,     0,     0,   337,     0,   339,    32,
    33,     0,     0,     0,     0,     0,     0,     0,     0,    39,
   340,   341,   342,   343,   344,   345,   895,     0,   347,   348,
     0,     0,     0,     0,     0,     0,     0,     0,     0,  -701,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,   746,     0,   352,   353,    42,     0,     0,     0,
    43,     0,   354,   355,     0,     0,     3,   192,     4,     5,
     6,     0,     0,     0,     0,     7,     8,     9,     0,   334,
   335,     0,    11,    12,    13,     0,     0,    14,     0,     0,
     0,     0,     0,    18,     0,     0,    20,    21,    22,    23,
    24,    25,    26,    27,     0,     0,    28,    29,     0,     0,
   336,    30,     0,     0,     0,     0,   337,     0,   339,    32,
    33,     0,     0,     0,     0,     0,     0,     0,     0,    39,
   340,   341,   342,   343,   344,   345,   401,     0,   347,   348,
     0,     0,     0,     0,     0,     0,     0,     0,     0,  -701,
     0,     3,     0,     4,     5,     6,     0,     0,     0,     0,
     7,     8,   897,     0,   352,   353,    42,     0,     0,    13,
    43,     0,   354,   355,     0,     0,     3,   192,     4,     5,
     6,    20,     0,     0,     0,     7,     8,     0,     0,   334,
   335,     0,    11,    12,    13,     0,     0,    14,     0,     0,
     0,     0,     0,    18,    32,    33,    20,     0,    22,    23,
    24,    25,    26,    27,    39,     0,    28,    29,     0,     0,
   336,   510,     0,     0,     0,     0,   337,   338,   339,    32,
    33,     0,     0,     0,     0,     0,     0,     0,     0,    39,
   340,   341,   342,   343,   344,   345,   401,   512,   347,   348,
     0,    42,     0,     0,     0,    43,     0,    44,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,   350,     0,   352,   353,    42,     0,     0,     0,
    43,     0,   354,   355,     0,     0,     3,   580,     4,     5,
     6,     0,     0,     0,     0,     7,     8,     9,     0,     0,
     0,    10,    11,    12,    13,     0,     0,    14,     0,     0,
    15,    16,    17,    18,   162,     0,    20,    21,    22,    23,
    24,    25,    26,    27,     0,     0,    28,    29,     0,     0,
     0,    30,     0,     0,    31,     0,     0,     0,     0,    32,
    33,     0,     0,     0,    34,    35,    36,    37,    38,    39,
     0,     0,     0,     0,     0,     0,   713,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,   349,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,   603,   351,     0,     3,    42,     4,     5,     6,
    43,     0,    44,     0,     7,     8,     9,   192,     0,     0,
    10,    11,    12,    13,     0,     0,    14,     0,     0,    15,
    16,    17,    18,   162,     0,    20,    21,    22,    23,    24,
    25,    26,    27,     0,     0,    28,    29,     0,     0,     0,
    30,     0,     0,    31,     0,     0,     0,     0,    32,    33,
     0,     0,     0,    34,    35,    36,    37,    38,    39,     0,
     0,     0,     0,     0,     0,   713,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,   349,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,   610,   351,     0,     3,    42,     4,     5,     6,    43,
     0,    44,     0,     7,     8,     9,   192,     0,     0,    10,
    11,    12,    13,     0,     0,    14,     0,     0,    15,    16,
    17,    18,   162,     0,    20,    21,    22,    23,    24,    25,
    26,    27,     0,     0,    28,    29,     0,     0,     0,    30,
     0,     0,    31,     0,     0,     0,     0,    32,    33,     0,
     0,     0,    34,    35,    36,    37,    38,    39,     0,     0,
     0,     0,     0,     0,   224,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,  -701,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
   610,  1136,     0,     3,    42,     4,     5,     6,    43,     0,
    44,     0,     7,     8,     9,   192,     0,     0,    10,    11,
    12,    13,     0,     0,    14,     0,     0,    15,    16,    17,
    18,   162,     0,    20,    21,    22,    23,    24,    25,    26,
    27,     0,     0,    28,    29,     0,     0,     0,    30,     0,
     0,    31,     0,     0,     0,     0,    32,    33,     0,     0,
     0,    34,    35,    36,    37,    38,    39,     0,     0,     0,
     0,     0,     0,   224,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,  -701,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,   610,
     0,     0,     3,    42,     4,     5,     6,    43,     0,    44,
     0,     7,     8,     9,   192,     0,     0,    10,    11,    12,
    13,     0,     0,    14,     0,     0,    15,    16,    17,    18,
   162,     0,    20,    21,    22,    23,    24,    25,    26,    27,
     0,     0,    28,    29,     0,     0,     0,    30,     0,     0,
    31,     0,     0,     0,     0,     0,    33,     0,     0,     0,
    34,    35,    36,    37,    38,   106,     0,     0,     0,     0,
     0,     0,   713,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,   349,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,   762,   351,
     0,     3,    42,     4,     5,     6,    43,     0,     0,     0,
     7,     8,     9,   192,     0,     0,    10,    11,    12,    13,
     0,     0,    14,     0,     0,     0,    16,    17,    18,   162,
     0,    20,    21,    22,    23,    24,    25,    26,    27,     0,
     0,    28,    29,     0,     0,     0,    30,     0,     0,    31,
     0,     0,     0,     0,    32,    33,     0,     0,     0,    34,
    35,    36,    37,    38,    39,     0,     0,     0,     0,     0,
     0,   510,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,   603,  1138,     0,
     3,    42,     4,     5,     6,    43,     0,    44,     0,     7,
     8,     9,   192,     0,     0,    10,    11,    12,    13,     0,
     0,    14,     0,     0,     0,    16,    17,    18,   162,     0,
    20,    21,    22,    23,    24,    25,    26,    27,     0,     0,
    28,    29,     0,     0,     0,    30,     0,     0,    31,     0,
     0,     0,     0,    32,    33,     0,     0,     0,    34,    35,
    36,    37,    38,    39,     0,     0,     0,     0,     0,     0,
   510,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,   617,     0,     0,     3,
    42,     4,     5,     6,    43,     0,    44,     0,     7,     8,
     9,   192,     0,     0,    10,    11,    12,    13,     0,     0,
    14,     0,     0,     0,    16,    17,    18,   162,     0,    20,
    21,    22,    23,    24,    25,    26,    27,     0,     0,    28,
    29,     0,     0,     0,    30,     0,     0,    31,     0,     0,
     0,     0,    32,    33,     0,     0,     0,    34,    35,    36,
    37,    38,    39,     0,     0,     0,     0,     0,     0,   510,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,   603,     0,     0,     3,    42,
     4,     5,     6,    43,     0,    44,     0,     7,     8,     9,
   192,     0,     0,     0,    11,    12,    13,     0,     0,    14,
     0,     0,    15,     0,     0,    18,     0,     0,    20,    21,
    22,    23,    24,    25,    26,    27,     0,     0,    28,    29,
     0,     0,     0,    30,     0,     0,     0,     0,     0,     0,
     0,     0,    33,     0,     0,     0,     0,     0,     0,     0,
     3,   106,     4,     5,     6,     0,     0,     0,   224,     7,
     8,     9,     0,     0,     0,    10,     0,     0,    13,     0,
     0,  -701,     0,     0,     0,    16,    17,     0,   162,     0,
    20,    21,     0,     0,   762,     0,     0,     0,    42,     0,
     0,     0,    43,     0,     0,    30,     0,     0,    31,   192,
     0,     0,     0,    32,    33,     0,     0,     0,    34,    35,
    36,    37,    38,    39,     3,     0,     4,     5,     6,     0,
   510,     0,     0,     7,     8,     9,     0,     0,     0,    10,
     0,     0,    13,  -700,     0,     0,     0,     0,     0,    16,
    17,     0,   162,     0,    20,    21,   534,     0,     0,     0,
    42,     0,     0,     0,    43,     0,    44,     0,     0,    30,
     0,   192,    31,     0,     0,     0,     0,    32,    33,     0,
     0,     0,    34,    35,    36,    37,    38,    39,     3,     0,
     4,     5,     6,     0,   510,     0,     0,     7,     8,     9,
     0,     0,     0,    10,     0,     0,    13,  -700,     0,     0,
     0,     0,     0,    16,    17,     0,   162,     0,    20,    21,
   603,     0,     0,     0,    42,     0,     0,     0,    43,     0,
    44,     0,     0,    30,     0,   192,    31,     0,     0,     0,
     0,    32,    33,     0,     0,     0,    34,    35,    36,    37,
    38,    39,     3,     0,     4,     5,     6,     0,   510,     0,
     0,     7,     8,     9,     0,     0,     0,    10,     0,     0,
    13,     0,     0,     0,     0,     0,     0,    16,    17,     0,
   162,     0,    20,    21,   603,  1144,     0,     0,    42,     0,
     0,     0,    43,     0,    44,     0,     0,    30,     0,   192,
    31,     0,     0,     0,     0,    32,    33,     0,     0,     0,
    34,    35,    36,    37,    38,    39,     3,     0,     4,     5,
     6,     0,   510,     0,     0,     7,     8,     9,     0,     0,
     0,    10,     0,     0,    13,     0,     0,     0,     0,     0,
     0,    16,    17,     0,   162,     0,    20,    21,   636,     0,
     0,     0,    42,     0,     0,     0,    43,     0,    44,     0,
     0,    30,     0,   192,    31,     0,     0,     0,     0,    32,
    33,     0,     0,     0,    34,    35,    36,    37,    38,    39,
     0,     0,     0,     0,     0,     0,   510,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,   603,     0,     0,     3,    42,     4,     5,     6,
    43,     0,    44,     0,     7,     8,     9,   192,     0,     0,
     0,    11,    12,    13,     0,     0,    14,     0,     0,     0,
     0,     0,    18,     0,     0,    20,    21,    22,    23,    24,
    25,    26,    27,     0,     0,    28,    29,     0,     0,     0,
    30,     0,     0,     0,     0,     0,     0,     0,     0,    33,
     0,     0,     0,     0,     0,     0,     0,     0,   106,     0,
     0,     0,     0,     0,     0,   186,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,  1179,     0,
     3,     0,     4,     5,     6,     0,     0,     0,     0,     7,
     8,   762,     0,   334,   335,    42,    11,    12,    13,    43,
     0,    14,     0,     0,     0,     0,   192,    18,     0,     0,
    20,     0,    22,    23,    24,    25,    26,    27,     0,     0,
    28,    29,     0,     0,   336,     0,     0,     0,     0,     0,
   337,   338,   339,    32,    33,     0,     0,     0,     0,     0,
     0,     0,     0,    39,   340,   341,   342,   343,   344,   345,
   401,     0,   347,   348,     0,     0,     0,     0,     0,     0,
     0,     0,     0,  -701,     0,     3,     0,     4,     5,     6,
     0,     0,     0,     0,     7,     8,   350,  -180,   352,   353,
    42,     0,     0,    13,    43,     0,   354,   355,  1166,     0,
     3,     0,     4,     5,     6,    20,     0,     0,     0,     7,
     8,     0,     0,   334,   335,     0,    11,    12,    13,     0,
     0,    14,     0,     0,     0,     0,     0,    18,    32,    33,
    20,     0,    22,    23,    24,    25,    26,    27,    39,     0,
    28,    29,     0,     0,   336,   522,     0,     0,     0,     0,
   337,   338,   339,    32,    33,     0,     0,     0,     0,     0,
     0,     0,     0,    39,   340,   341,   342,   343,   344,   345,
   401,   307,   347,   348,     0,    42,     0,     0,     0,    43,
     0,    44,     0,  -701,     0,     3,     0,     4,     5,     6,
     0,     0,     0,     0,     7,     8,   350,     0,   352,   353,
    42,     0,     0,    13,    43,     0,   354,   355,  1276,     0,
     3,     0,     4,     5,     6,    20,     0,     0,     0,     7,
     8,     0,     0,   334,   335,     0,    11,    12,    13,     0,
     0,    14,     0,     0,     0,     0,     0,    18,    32,    33,
    20,     0,    22,    23,    24,    25,    26,    27,    39,     0,
    28,    29,     0,     0,   336,   524,     0,     0,     0,     0,
   337,   338,   339,    32,    33,     0,     0,     0,     0,     0,
     0,     0,     0,    39,   340,   341,   342,   343,   344,   345,
   401,   307,   347,   348,     0,    42,     0,     0,     0,    43,
     0,    44,     0,  -701,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,   350,     0,   352,   353,
    42,     0,     0,     0,    43,     0,   354,   355,  1332,     2,
     0,     3,     0,     4,     5,     6,     0,     0,     0,     0,
     7,     8,     9,     0,     0,     0,    10,    11,    12,    13,
     0,     0,    14,     0,     0,    15,    16,    17,    18,    19,
     0,    20,    21,    22,    23,    24,    25,    26,    27,     0,
     0,    28,    29,     0,     0,     0,    30,     0,     0,    31,
     0,     0,     0,     0,    32,    33,     0,     0,     0,    34,
    35,    36,    37,    38,    39,     0,     0,     0,     0,     0,
     0,    40,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,  -701,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,    41,   250,     0,
     3,    42,     4,     5,     6,    43,     0,    44,     0,     7,
     8,     9,     0,     0,     0,    10,    11,    12,    13,     0,
     0,    14,     0,     0,     0,    16,    17,    18,   162,     0,
    20,    21,    22,    23,    24,    25,    26,    27,     0,     0,
    28,    29,     0,     0,     0,    30,     0,     0,    31,     0,
     0,     0,     0,    32,    33,     0,     0,     0,    34,    35,
    36,    37,    38,    39,     0,     0,     0,     0,     0,     0,
   163,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,   164,     0,     0,     3,
    42,     4,     5,     6,    43,     0,    44,     0,     7,     8,
     9,     0,   334,   335,    10,    11,    12,    13,     0,     0,
    14,     0,     0,    15,    16,    17,    18,   162,     0,    20,
    21,    22,    23,    24,    25,    26,    27,     0,     0,    28,
    29,     0,     0,   336,    30,     0,     0,    31,     0,   337,
   338,   339,    32,    33,     0,     0,     0,    34,    35,    36,
    37,    38,    39,   340,   341,   342,   343,   344,   345,   346,
     0,   347,   348,     0,     0,     0,     0,     0,     0,     0,
     0,   349,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,   350,   351,   352,   353,    42,
     0,     0,     0,    43,     0,   354,   355,     3,     0,     4,
     5,     6,     0,     0,     0,     0,     7,     8,     9,     0,
   334,   335,    10,    11,    12,    13,     0,     0,    14,     0,
     0,    15,    16,    17,    18,   162,     0,    20,    21,    22,
    23,    24,    25,    26,    27,     0,     0,    28,    29,     0,
     0,   336,    30,     0,     0,    31,     0,   337,   338,   339,
    32,    33,     0,     0,     0,    34,    35,    36,    37,    38,
    39,   340,   341,   342,   343,   344,   345,   346,     0,   347,
   348,     0,     0,     0,     0,     0,     0,     0,     0,   349,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,   350,   747,   352,   353,    42,     0,     0,
     0,    43,     0,   354,   355,     3,     0,     4,     5,     6,
     0,     0,     0,     0,     7,     8,     9,     0,   334,   335,
     0,    11,    12,    13,     0,     0,    14,     0,     0,    15,
     0,     0,    18,     0,     0,    20,    21,    22,    23,    24,
    25,    26,    27,     0,     0,    28,    29,     0,     0,   336,
    30,     0,     0,     0,     0,   337,   338,   339,    32,    33,
     0,     0,     0,     0,     0,     0,     0,     0,    39,   340,
   341,   342,   343,   344,   345,   547,     0,   347,   348,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,   350,     0,   352,   353,    42,     0,     0,     0,    43,
     0,   354,   355,     3,     0,     4,     5,     6,     0,     0,
     0,     0,     7,     8,     9,     0,   334,   335,     0,    11,
    12,    13,     0,     0,    14,     0,     0,    15,     0,     0,
    18,     0,     0,    20,    21,    22,    23,    24,    25,    26,
    27,     0,     0,    28,    29,     0,     0,   336,    30,     0,
     0,     0,     0,   337,   338,   339,    32,    33,     0,     0,
     0,     0,     0,     0,     0,     0,    39,   340,   341,   342,
   343,   344,   345,  1235,     0,   347,   348,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,  1182,
     0,   352,   353,    42,     0,     0,     0,    43,     0,   354,
   355,     3,     0,     4,     5,     6,     0,     0,     0,     0,
     7,     8,     9,     0,   334,   335,     0,    11,    12,    13,
     0,     0,    14,     0,     0,    15,     0,     0,    18,     0,
     0,    20,    21,    22,    23,    24,    25,    26,    27,     0,
     0,    28,    29,     0,     0,   336,    30,     0,     0,     0,
     0,   337,   338,   339,    32,    33,     0,     0,     0,     0,
     0,     0,     0,     0,    39,   340,   341,   342,   343,   344,
   345,  1282,     0,   347,   348,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,  1182,     0,   352,
   353,    42,     0,     0,     0,    43,     0,   354,   355,     3,
     0,     4,     5,     6,     0,     0,     0,     0,     7,     8,
     9,     0,   334,   335,     0,    11,    12,    13,     0,     0,
    14,     0,     0,     0,     0,     0,    18,     0,     0,    20,
    21,    22,    23,    24,    25,    26,    27,     0,     0,    28,
    29,     0,     0,   336,    30,     0,     0,     0,     0,   337,
     0,   339,    32,    33,     0,     0,     0,     0,     0,     0,
     0,     0,    39,   340,   341,   342,   343,   344,   345,   401,
     0,   347,   348,     0,     0,     0,     0,     0,     0,     0,
     0,     0,  -701,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,   350,     0,   352,   353,    42,
     0,     0,     0,    43,     0,   354,   355,     3,     0,     4,
     5,     6,     0,     0,   769,   770,     7,     8,     0,     0,
   334,   335,     0,    11,    12,    13,     0,     0,    14,     0,
     0,     0,     0,     0,    18,     0,     0,    20,     0,    22,
    23,    24,    25,    26,    27,     0,     0,    28,    29,     0,
     0,   336,     0,     0,     0,     0,     0,   337,   338,   339,
    32,    33,     0,     0,     0,     0,     0,     0,     0,     0,
    39,   340,   341,   342,   343,   344,   345,   401,     0,   347,
   348,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,   350,     0,   352,   353,    42,     0,     0,
     0,    43,     0,   354,   355,     3,     0,     4,     5,     6,
     0,     0,     0,     0,     7,     8,     9,     0,   334,   335,
     0,    11,    12,    13,     0,     0,    14,     0,     0,     0,
     0,     0,    18,     0,     0,    20,    21,    22,    23,    24,
    25,    26,    27,     0,     0,    28,    29,     0,     0,   336,
    30,     0,     0,     0,     0,   337,     0,   339,    32,    33,
     0,     0,     0,     0,     0,     0,     0,     0,    39,   340,
   341,   342,   343,   344,   345,   895,     0,   347,   348,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     3,     0,
     4,     5,     6,     0,     0,     0,     0,     7,     8,     9,
     0,  1236,     0,   352,   353,    42,    13,     0,     0,    43,
     0,   354,   355,     3,     0,     4,     5,     6,    20,    21,
     0,     0,     7,     8,     0,     0,   334,   335,     0,    11,
    12,    13,     0,    30,    14,     0,     0,     0,     0,     0,
    18,    32,    33,    20,     0,    22,    23,    24,    25,    26,
    27,    39,     0,    28,    29,     0,     0,   336,   510,     0,
     0,     0,     0,   337,   338,   339,    32,    33,     0,     0,
     0,     0,     0,     0,     0,     0,    39,   340,   341,   342,
   343,   344,   345,   401,   825,   347,   348,     0,    42,     0,
     0,     0,    43,     0,    44,     0,     3,     0,     4,     5,
     6,     0,     0,     0,     0,     7,     8,     9,     0,   350,
   767,   352,   353,    42,    13,     0,     0,    43,     0,   354,
   355,     3,     0,     4,     5,     6,    20,    21,     0,     0,
     7,     8,     0,     0,   334,   335,     0,    11,    12,    13,
     0,    30,    14,     0,     0,     0,     0,     0,    18,    32,
    33,    20,     0,    22,    23,    24,    25,    26,    27,    39,
     0,    28,    29,     0,     0,   336,   510,     0,     0,     0,
     0,   337,   338,   339,    32,    33,     0,     0,     0,     0,
     0,     0,     0,     0,    39,   340,   341,   342,   343,   344,
   345,   401,  1066,   347,   348,     0,    42,     0,     0,     0,
    43,     0,    44,     0,     3,     0,     4,     5,     6,     0,
     0,     0,     0,     7,     8,     9,     0,   350,   820,   352,
   353,    42,    13,     0,     0,    43,     0,   354,   355,     3,
     0,     4,     5,     6,    20,    21,     0,     0,     7,     8,
     0,     0,   334,   335,     0,    11,    12,    13,     0,    30,
    14,     0,     0,     0,     0,     0,    18,    32,    33,    20,
     0,    22,    23,    24,    25,    26,    27,    39,     0,    28,
    29,     0,     0,   336,   510,     0,     0,     0,     0,   337,
   338,   339,    32,    33,     0,     0,     0,     0,     0,     0,
     0,     0,    39,   340,   341,   342,   343,   344,   345,   401,
  1076,   347,   348,     0,    42,     0,     0,     0,    43,     0,
    44,     0,     3,     0,     4,     5,     6,     0,     0,     0,
     0,     7,     8,     0,     0,   350,   821,   352,   353,    42,
    13,     0,     0,    43,     0,   354,   355,     3,     0,     4,
     5,     6,    20,     0,     0,     0,     7,     8,     0,     0,
   334,   335,     0,    11,    12,    13,     0,     0,    14,     0,
     0,     0,     0,     0,    18,    32,    33,    20,     0,    22,
    23,    24,    25,    26,    27,    39,     0,    28,    29,     0,
     0,   336,   186,     0,     0,     0,     0,   337,   338,   339,
    32,    33,     0,     0,     0,     0,     0,     0,     0,     0,
    39,   340,   341,   342,   343,   344,   345,   401,   646,   347,
   348,     0,    42,     0,     0,     0,    43,     0,    44,     0,
     3,     0,     4,     5,     6,     0,     0,     0,     0,     7,
     8,     0,     0,   350,   830,   352,   353,    42,    13,     0,
     0,    43,     0,   354,   355,     3,     0,     4,     5,     6,
    20,     0,     0,     0,     7,     8,     0,     0,   334,   335,
     0,    11,    12,    13,     0,     0,    14,     0,     0,     0,
     0,     0,    18,    32,    33,    20,     0,    22,    23,    24,
    25,    26,    27,    39,     0,    28,    29,     0,     0,   336,
   306,     0,     0,     0,     0,   337,   338,   339,    32,    33,
     0,     0,     0,     0,     0,     0,     0,     0,    39,   340,
   341,   342,   343,   344,   345,   401,   652,   347,   348,     0,
    42,     0,     0,     0,    43,     0,    44,     0,     3,     0,
     4,     5,     6,     0,     0,     0,     0,     7,     8,     0,
     0,   350,   934,   352,   353,    42,    13,     0,     0,    43,
     0,   354,   355,     3,     0,     4,     5,     6,    20,     0,
     0,     0,     7,     8,     0,     0,   334,   335,     0,    11,
    12,    13,     0,     0,    14,     0,     0,     0,     0,     0,
    18,    32,    33,    20,     0,    22,    23,    24,    25,    26,
    27,    39,     0,    28,    29,     0,     0,   336,   163,     0,
     0,     0,     0,   337,   338,   339,    32,    33,     0,     0,
     0,     0,     0,     0,     0,     0,    39,   340,   341,   342,
   343,   344,   345,   401,   657,   347,   348,     0,    42,     0,
     0,     0,    43,     0,    44,     0,     3,     0,     4,     5,
     6,     0,     0,     0,     0,     7,     8,     0,     0,   350,
   936,   352,   353,    42,    13,     0,     0,    43,     0,   354,
   355,     3,     0,     4,     5,     6,    20,     0,     0,     0,
     7,     8,     0,     0,   334,   335,     0,    11,    12,    13,
     0,     0,    14,     0,     0,     0,     0,     0,    18,    32,
    33,    20,     0,    22,    23,    24,    25,    26,    27,    39,
     0,    28,    29,     0,     0,   336,   727,     0,     0,     0,
     0,   337,   338,   339,    32,    33,     0,     0,     0,     0,
     0,     0,     0,     0,    39,   340,   341,   342,   343,   344,
   345,   401,   307,   347,   348,     0,    42,     0,     0,     0,
    43,     0,    44,     0,     3,     0,     4,     5,     6,     0,
     0,     0,     0,     7,     8,     0,     0,   350,   939,   352,
   353,    42,    13,     0,     0,    43,     0,   354,   355,     3,
     0,     4,     5,     6,    20,     0,     0,     0,     7,     8,
     0,     0,   334,   335,     0,    11,    12,    13,     0,     0,
    14,     0,     0,     0,     0,     0,    18,    32,    33,    20,
     0,    22,    23,    24,    25,    26,    27,    39,     0,    28,
    29,     0,     0,   336,   729,     0,     0,     0,     0,   337,
   338,   339,    32,    33,     0,     0,     0,     0,     0,     0,
     0,     0,    39,   340,   341,   342,   343,   344,   345,   401,
   307,   347,   348,     0,    42,     0,     0,     0,    43,     0,
    44,     0,     3,     0,     4,     5,     6,     0,     0,     0,
     0,     7,     8,     0,     0,   350,  1040,   352,   353,    42,
    13,     0,     0,    43,     0,   354,   355,     3,     0,     4,
     5,     6,    20,     0,     0,     0,     7,     8,     0,     0,
   334,   335,     0,    11,    12,    13,     0,     0,    14,     0,
     0,     0,     0,     0,    18,    32,    33,    20,     0,    22,
    23,    24,    25,    26,    27,    39,     0,    28,    29,     0,
     0,   336,   501,     0,     0,     0,     0,   337,   338,   339,
    32,    33,     0,     0,     0,     0,     0,     0,     0,     0,
    39,   340,   341,   342,   343,   344,   345,   401,   652,   347,
   348,     0,    42,     0,     0,     0,    43,     0,    44,     0,
  -701,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,   350,     0,   352,   353,    42,     0,     0,
     0,    43,     0,   354,   355,     3,     0,     4,     5,     6,
     0,     0,     0,     0,     7,     8,     0,     0,   334,   335,
     0,    11,    12,    13,     0,     0,    14,     0,     0,     0,
     0,     0,    18,     0,     0,    20,     0,    22,    23,    24,
    25,    26,    27,     0,     0,    28,    29,     0,     0,   336,
     0,     0,     0,     0,     0,   337,   338,   339,    32,    33,
     0,     0,     0,     0,     0,     0,     0,     0,    39,   340,
   341,   342,   343,   344,   345,  1181,     0,   347,   348,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     3,     0,
     4,     5,     6,     0,     0,     0,     0,     7,     8,     0,
     0,  1182,   820,   352,   353,    42,    13,     0,     0,    43,
     0,   354,   355,     3,     0,     4,     5,     6,    20,     0,
     0,     0,     7,     8,     0,     0,   334,   335,     0,    11,
    12,    13,     0,     0,    14,     0,     0,     0,     0,     0,
    18,    32,    33,    20,     0,    22,    23,    24,    25,    26,
    27,    39,     0,    28,    29,     0,     0,   336,   510,     0,
     0,     0,     0,   337,   338,   339,    32,    33,     0,     0,
     0,     0,     0,     0,     0,     0,    39,   340,   341,   342,
   343,   344,   345,  1181,   826,   347,   348,     0,    42,     0,
     0,     0,    43,     0,    44,     0,     3,     0,     4,     5,
     6,     0,     0,     0,     0,     7,     8,     0,     0,  1182,
   821,   352,   353,    42,    13,     0,     0,    43,     0,   354,
   355,     3,     0,     4,     5,     6,    20,     0,     0,     0,
     7,     8,     0,     0,   334,   335,     0,    11,    12,    13,
     0,     0,    14,     0,     0,     0,     0,     0,    18,    32,
    33,    20,     0,    22,    23,    24,    25,    26,    27,    39,
     0,    28,    29,     0,     0,   336,   522,     0,     0,     0,
     0,   337,   338,   339,    32,    33,     0,     0,     0,     0,
     0,     0,     0,     0,    39,   340,   341,   342,   343,   344,
   345,   401,   652,   347,   348,     0,    42,     0,     0,     0,
    43,     0,    44,     0,     3,     0,     4,     5,     6,     0,
     0,     0,     0,     7,     8,     0,     0,   350,     0,   352,
   353,    42,    13,     0,     0,    43,     0,   354,   355,     3,
     0,     4,     5,     6,    20,     0,     0,     0,     7,     8,
     0,     0,   334,   335,     0,    11,    12,    13,     0,     0,
    14,     0,     0,     0,     0,     0,    18,    32,    33,    20,
     0,    22,    23,    24,    25,    26,    27,    39,     0,    28,
    29,     0,     0,   336,   524,     0,     0,     0,     0,   337,
   338,   339,    32,    33,     0,     0,     0,     0,     0,     0,
     0,     0,    39,   340,   341,   342,   343,   344,   345,  1181,
   652,   347,   348,     0,    42,     0,     0,     0,    43,     0,
    44,     0,     3,     0,     4,     5,     6,     0,     0,     0,
     0,     7,     8,     0,     0,  1182,     0,   352,   353,    42,
    13,     0,     0,    43,     0,   354,   355,     3,     0,     4,
     5,     6,    20,     0,     0,     0,     7,     8,     0,     0,
   334,   335,     0,    11,    12,    13,     0,     0,    14,     0,
     0,     0,     0,     0,    18,    32,    33,    20,     0,    22,
    23,    24,    25,    26,    27,    39,     0,    28,    29,     0,
     0,   336,   727,     0,     0,     0,     0,   337,     0,   339,
    32,    33,     0,     0,     0,     0,     0,     0,     0,     0,
    39,   340,   341,   342,   343,   344,   345,   401,   652,   347,
   348,     0,    42,     0,     0,     0,    43,     0,    44,     0,
     3,     0,     4,     5,     6,     0,     0,     0,     0,     7,
     8,     0,     0,   531,     0,   352,   353,    42,    13,     0,
     0,    43,     0,   354,   355,     3,     0,     4,     5,     6,
    20,     0,     0,     0,     7,     8,     0,     0,   334,   335,
     0,    11,    12,    13,     0,     0,    14,     0,     0,     0,
     0,     0,    18,    32,    33,    20,     0,    22,    23,    24,
    25,    26,    27,    39,     0,    28,    29,     0,     0,   336,
   729,     0,     0,     0,     0,   337,     0,   339,    32,    33,
     0,     0,     0,     0,     0,     0,     0,     0,    39,   340,
   341,   342,   343,   344,   345,   401,   652,   347,   348,     0,
    42,     0,     0,     0,    43,     0,    44,     0,     3,     0,
     4,     5,     6,     0,     0,     0,     0,     7,     8,     0,
     0,   544,     0,   352,   353,    42,    13,     0,     0,    43,
     0,   354,   355,     3,     0,     4,     5,     6,    20,     0,
     0,     0,     7,     8,     0,     0,   334,   335,     0,    11,
    12,    13,     0,     0,    14,     0,     0,     0,     0,     0,
    18,    32,    33,    20,     0,    22,    23,    24,    25,    26,
    27,    39,     0,    28,    29,     0,     0,   336,   510,     0,
     0,     0,     0,   337,     0,   339,    32,    33,     0,     0,
     0,     0,     0,     0,     0,     0,    39,   340,   341,   342,
   343,   344,   345,   401,  1068,   347,   348,     0,    42,     0,
     0,     0,    43,     0,    44,     0,     3,     0,     4,     5,
     6,     0,     0,     0,     0,     7,     8,     0,     0,   350,
     0,   352,   353,    42,    13,     0,     0,    43,     0,   354,
   355,     3,     0,     4,     5,     6,    20,     0,     0,     0,
     7,     8,     0,     0,   334,   335,     0,    11,    12,    13,
     0,     0,    14,     0,     0,     0,     0,     0,    18,    32,
    33,    20,     0,    22,    23,    24,    25,    26,    27,    39,
     0,    28,    29,     0,     0,   336,   510,     0,     0,     0,
     0,   337,     0,   339,    32,    33,     0,     0,     0,     0,
     0,     0,     0,     0,    39,   340,   341,   342,   343,   344,
   345,   562,  1078,   347,   348,     0,    42,     0,     0,     0,
    43,     0,    44,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,   350,     0,   352,
   353,    42,     0,     0,     0,    43,     0,   354,   355,     3,
     0,     4,     5,     6,   449,     0,     0,     0,     7,     8,
     9,     0,     0,     0,    10,    11,    12,    13,     0,     0,
    14,     0,     0,    15,    16,    17,    18,   162,     0,    20,
    21,    22,    23,    24,    25,    26,    27,     0,     0,    28,
    29,     0,     0,     0,    30,     0,     0,    31,     0,     0,
     0,     0,    32,    33,   450,   451,   452,    34,    35,    36,
    37,    38,    39,     0,     0,     0,     0,     0,     0,   838,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,  -701,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,   646,     0,     0,     3,    42,
     4,     5,     6,    43,     0,    44,     0,     7,     8,     9,
     0,     0,     0,    10,    11,    12,    13,     0,     0,    14,
     0,     0,    15,    16,    17,    18,   162,     0,    20,    21,
    22,    23,    24,    25,    26,    27,     0,     0,    28,    29,
     0,     0,     0,    30,     0,     0,    31,     0,     0,     0,
     0,    32,    33,     0,     0,     0,    34,    35,    36,    37,
    38,    39,     0,     0,     0,     0,     0,     0,   424,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
   349,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,   307,   351,     0,     3,    42,     4,
     5,     6,    43,     0,    44,     0,     7,     8,     9,     0,
     0,     0,    10,    11,    12,    13,     0,     0,    14,     0,
     0,    15,    16,    17,    18,   162,     0,    20,    21,    22,
    23,    24,    25,    26,    27,     0,     0,    28,    29,     0,
     0,     0,    30,     0,     0,    31,     0,     0,     0,     0,
    32,    33,     0,     0,     0,    34,    35,    36,    37,    38,
    39,     0,     0,     0,     0,     0,     0,  1050,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,   349,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,   652,  1051,     0,     3,    42,     4,     5,
     6,    43,     0,    44,     0,     7,     8,     9,     0,     0,
     0,    10,    11,    12,    13,     0,     0,    14,     0,     0,
    15,    16,    17,    18,   162,     0,    20,    21,    22,    23,
    24,    25,    26,    27,     0,     0,    28,    29,     0,     0,
     0,    30,     0,     0,    31,     0,     0,     0,     0,    32,
    33,     0,     0,     0,    34,    35,    36,    37,    38,    39,
     0,     0,     0,     0,     0,     0,  1070,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,   349,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,   652,  1051,     0,     3,    42,     4,     5,     6,
    43,     0,    44,     0,     7,     8,     9,     0,     0,     0,
    10,    11,    12,    13,     0,     0,    14,     0,     0,    15,
    16,    17,    18,   162,     0,    20,    21,    22,    23,    24,
    25,    26,    27,     0,     0,    28,    29,     0,     0,     0,
    30,     0,     0,    31,     0,     0,     0,     0,    32,    33,
     0,     0,     0,    34,    35,    36,    37,    38,    39,     0,
     0,     0,     0,     0,     0,   209,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,   210,     0,     0,     3,    42,     4,     5,     6,    43,
     0,    44,     0,     7,     8,     9,     0,     0,     0,    10,
    11,    12,    13,     0,     0,    14,     0,     0,    15,    16,
    17,    18,   162,     0,    20,    21,    22,    23,    24,    25,
    26,    27,     0,     0,    28,    29,     0,     0,     0,    30,
     0,     0,    31,     0,     0,     0,     0,    32,    33,     0,
     0,     0,    34,    35,    36,    37,    38,    39,     0,     0,
     0,     0,     0,     0,   224,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
   210,     0,     0,     3,    42,     4,     5,     6,    43,     0,
    44,     0,     7,     8,     9,     0,     0,     0,    10,    11,
    12,    13,     0,     0,    14,     0,     0,    15,    16,    17,
    18,   162,     0,    20,    21,    22,    23,    24,    25,    26,
    27,     0,     0,    28,    29,     0,     0,     0,    30,     0,
     0,    31,     0,     0,     0,     0,    32,    33,     0,     0,
     0,    34,    35,    36,    37,    38,    39,     0,     0,     0,
     0,     0,     0,   209,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,   412,
     0,     0,     3,    42,     4,     5,     6,    43,     0,    44,
     0,     7,     8,     9,     0,     0,     0,    10,    11,    12,
    13,     0,     0,    14,     0,     0,    15,    16,    17,    18,
   162,     0,    20,    21,    22,    23,    24,    25,    26,    27,
     0,     0,    28,    29,     0,     0,     0,    30,     0,     0,
    31,     0,     0,     0,     0,    32,    33,     0,     0,     0,
    34,    35,    36,    37,    38,    39,     0,     0,     0,     0,
     0,     0,   224,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,   412,     0,
     0,     3,    42,     4,     5,     6,    43,     0,    44,     0,
     7,     8,     9,     0,     0,     0,    10,    11,    12,    13,
     0,     0,    14,     0,     0,     0,    16,    17,    18,   162,
     0,    20,    21,    22,    23,    24,    25,    26,    27,     0,
     0,    28,    29,     0,     0,     0,    30,     0,     0,    31,
     0,     0,     0,     0,    32,    33,     0,     0,     0,    34,
    35,    36,    37,    38,    39,     0,     0,     0,     0,     0,
     0,   163,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,   418,     0,     0,
     3,    42,     4,     5,     6,    43,     0,    44,     0,     7,
     8,     9,     0,     0,     0,    10,    11,    12,    13,     0,
     0,    14,     0,     0,     0,    16,    17,    18,   162,     0,
    20,    21,    22,    23,    24,    25,    26,    27,     0,     0,
    28,    29,     0,     0,     0,    30,     0,     0,    31,     0,
     0,     0,     0,    32,    33,     0,     0,     0,    34,    35,
    36,    37,    38,    39,     0,     0,     0,     0,     0,     0,
   163,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     0,     3,     0,     4,     0,  1105,     0,     0,     0,
    42,     7,     8,     9,    43,     0,    44,    10,    11,    12,
    13,     0,     0,    14,     0,     0,     0,    16,    17,    18,
   162,     0,    20,    21,    22,    23,    24,    25,    26,    27,
     0,     0,    28,    29,     0,     0,     0,    30,     0,     0,
    31,     0,     0,     0,     0,    32,    33,     0,     0,     0,
    34,    35,    36,    37,    38,    39,     3,     0,     4,     5,
     6,     0,   186,     0,     0,     7,     8,     9,     0,     0,
     0,    10,     0,     0,    13,     0,     0,     0,     0,     0,
     0,    16,    17,     0,   162,     0,    20,    21,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,    44,     0,
     0,    30,     0,     0,    31,     0,     0,     0,     0,    32,
    33,     0,     0,     0,    34,    35,    36,    37,    38,    39,
     3,     0,     4,     5,     6,     0,   163,     0,     0,     7,
     8,     9,     0,     0,     0,    10,     0,     0,    13,     0,
     0,     0,     0,     0,     0,    16,    17,     0,   162,     0,
    20,    21,   164,     0,     0,     0,    42,     0,     0,     0,
    43,     0,    44,     0,     0,    30,     0,     0,    31,     0,
     0,     0,     0,    32,    33,     0,     0,     0,    34,    35,
    36,    37,    38,    39,     3,     0,     4,     5,     6,     0,
   163,     0,     0,     7,     8,     9,     0,     0,     0,    10,
     0,     0,    13,     0,     0,     0,     0,     0,     0,    16,
    17,     0,   162,     0,    20,    21,   201,     0,     0,     0,
    42,     0,     0,     0,    43,     0,    44,     0,     0,    30,
     0,     0,    31,     0,     0,     0,     0,    32,    33,     0,
     0,     0,    34,    35,    36,    37,    38,    39,     3,     0,
     4,     5,     6,     0,   163,     0,     0,     7,     8,     9,
     0,     0,     0,    10,     0,     0,    13,     0,     0,     0,
     0,     0,     0,    16,    17,     0,   162,     0,    20,    21,
   418,     0,     0,     0,    42,     0,     0,     0,    43,     0,
    44,     0,     0,    30,     0,     0,    31,     0,     0,     0,
     0,    32,    33,     0,     0,     0,    34,    35,    36,    37,
    38,    39,     3,     0,     4,     5,     6,     0,   163,     0,
     0,     7,     8,     9,     0,     0,     0,    10,     0,     0,
    13,     0,     0,     0,     0,     0,     0,    16,    17,     0,
   162,     0,    20,    21,  1108,     0,     0,     0,    42,     0,
     0,     0,    43,     0,    44,     0,     0,    30,     0,     0,
    31,     0,     0,     0,     0,    32,    33,     0,     0,     0,
    34,    35,    36,    37,    38,    39,     3,     0,     4,     5,
     6,     0,   163,     0,     0,     7,     8,     3,     0,     4,
     5,     6,     0,     0,    13,     0,     7,     8,     0,     0,
     0,     0,     0,     0,     0,    13,    20,     0,  1103,     0,
     0,     0,    42,     0,     0,     0,    43,    20,    44,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,    32,
    33,     0,     0,     0,     0,     0,     0,     0,     0,    39,
    32,    33,     0,     0,     0,     3,  1148,     4,     5,     6,
    39,     0,     0,     0,     7,     8,     3,  1151,     4,     5,
     6,     0,     0,    13,     0,     7,     8,     0,     0,     0,
     0,     0,   652,     0,    13,    20,    42,     0,     0,     0,
    43,     0,    44,   652,     0,     0,    20,    42,     0,     0,
     0,    43,     0,    44,     0,     0,     0,     0,    32,    33,
     0,     0,     0,     0,     0,     0,     0,     0,    39,    32,
    33,     0,     0,     0,     0,  1157,     0,     0,     0,    39,
     0,     0,     0,     0,     0,     0,  1160,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,   652,     0,     0,     0,    42,     0,     0,     0,    43,
     0,    44,   652,     0,     0,     3,    42,     4,     5,     6,
    43,     0,    44,     0,     7,     8,     9,     0,     0,     0,
     0,    11,    12,    13,     0,     0,    14,     0,     0,    15,
     0,     0,    18,     0,     0,    20,    21,    22,    23,    24,
    25,    26,    27,     0,     0,    28,    29,     0,     0,     0,
    30,     0,     0,     0,     0,     0,     0,     0,     0,    33,
     0,     0,     0,     0,     0,     0,     0,     0,   106,     0,
     0,     0,     0,     0,     0,   224,     0,     0,     0,     0,
     0,     0,     0,     0,     0,     0,     0,     0,  -701,     0,
     0,     0,     0,     0,     0,     0,     0,     3,     0,     4,
     0,     0,     0,     0,     0,    42,     7,     8,     9,    43,
     0,     0,    10,    11,    12,    13,     0,     0,    14,     0,
     0,    15,    16,    17,    18,   162,     0,    20,    21,    22,
    23,    24,    25,    26,    27,     0,     0,    28,    29,     0,
     0,     0,    30,     0,     0,    31,     0,     0,     0,     0,
     0,    33,     0,     0,     0,    34,    35,    36,    37,    38,
   106,     0,     0,     0,     0,     0,     3,   713,     4,     0,
     0,     0,     0,     0,     0,     7,     8,     9,     0,   349,
     0,     0,    11,    12,    13,     0,     0,    14,     0,     0,
    15,     0,     0,    18,   742,     0,    20,    21,    22,    23,
    24,    25,    26,    27,     0,     0,    28,    29,     0,     0,
     0,    30,     0,     0,     3,     0,     4,     0,     0,     0,
    33,     0,     0,     7,     8,     9,     0,     0,     0,   106,
    11,    12,    13,     0,     0,    14,   107,     0,    15,     0,
     0,    18,     0,     0,    20,    21,    22,    23,    24,    25,
    26,    27,     0,     0,    28,    29,     0,     0,     0,    30,
     0,     0,   576,     0,     3,     0,     4,     0,    33,     0,
     0,     0,     0,     7,     8,     9,     0,   106,     0,    10,
    11,    12,    13,     0,   107,    14,     0,     0,    15,    16,
    17,    18,   162,     0,    20,    21,    22,    23,    24,    25,
    26,    27,     0,     0,    28,    29,     0,     0,     0,    30,
  1036,     0,    31,     0,     0,     0,     0,     0,    33,     0,
     0,     0,    34,    35,    36,    37,    38,   106,     0,     0,
     0,     3,     0,     4,   713,     0,     0,     0,     0,     0,
     7,     8,     9,     0,     0,     0,   809,    11,    12,    13,
     0,     0,    14,     0,     0,    15,     0,     0,    18,     0,
     0,    20,    21,    22,    23,    24,    25,    26,    27,     0,
     0,    28,    29,     0,     0,     0,    30,     0,     0,     0,
     0,     0,     0,     0,     0,    33,     0,     0,     0,     0,
     0,     0,     0,     0,   106,     0,     0,     0,     0,     0,
     0,  1121
};

static const short yycheck[] = {     1,
    32,     1,    52,    53,   453,    32,    55,    32,    57,    32,
    32,    60,   783,   527,    52,    53,   627,   192,   689,   733,
   293,   389,    50,   689,    90,   579,   100,   151,    52,    53,
    32,   842,   843,   623,   390,   391,   772,   627,    86,    53,
   100,    85,   116,   366,    14,   321,   840,   841,    66,    66,
    52,    53,    85,   120,   848,    73,    73,   270,   271,     8,
   120,   116,   350,   118,    34,   859,   100,   100,    88,    97,
   118,    73,   116,    73,   352,   353,   354,   355,    48,    14,
   116,    57,   116,   116,   120,     3,   119,     5,    90,   411,
    68,    85,    85,    71,    12,    13,    88,    85,    14,    34,
   118,   100,     1,    21,   121,   318,   100,   100,   321,    58,
    59,    60,    84,    48,   392,    33,    86,   116,    34,   151,
   100,    57,   116,   116,   151,   119,   119,   151,   116,   151,
    86,   119,    48,    66,   992,    85,   116,   534,    56,    57,
    73,    40,    85,    99,   915,   117,   148,   100,    66,   151,
   100,    50,    51,    52,    53,    73,   432,   100,   196,   435,
    87,    88,   438,   116,    63,   185,   116,   443,   160,   980,
    27,   447,   116,   116,    73,   299,   300,   913,   122,   157,
    85,   159,   196,   221,   222,   103,    85,    86,   982,   191,
   192,   109,    57,   185,   172,   100,    53,    54,    97,   201,
   116,   118,    66,   119,   121,   816,   603,   221,   222,    73,
   423,   116,   119,   610,   121,   428,   100,   493,   431,   432,
   617,   434,   435,    67,   437,   438,   816,   781,   782,   442,
   443,   444,   116,   446,   447,    57,   411,  1095,    86,   636,
   689,    50,    51,   531,  1102,    79,    80,   430,   100,   100,
   433,    99,    73,   436,    63,   257,   544,    99,   441,   582,
   583,   629,   445,    40,   116,   116,   114,   100,   481,   271,
   941,   271,   114,    50,    51,   941,    86,   111,   112,   492,
   493,    99,   570,   116,   183,   184,    63,   118,    97,    99,
   121,    99,   580,   534,   286,    14,   114,   299,   300,   101,
   102,   116,   304,   281,   114,   120,   114,   285,   491,   301,
  1024,   303,   314,    86,   602,    34,  1174,   319,   121,   321,
    97,   321,   221,   222,   119,   350,   121,   350,   394,    48,
    66,   363,   382,   383,   336,   384,   363,    73,   363,   100,
   363,   363,   408,    27,    86,   347,   348,   100,   350,   746,
   352,   353,   354,   355,    86,   116,   120,    99,   382,   383,
   757,   363,   603,   116,   366,   762,   763,    99,   765,   610,
   692,  1229,   114,   393,   183,   184,   617,   120,   416,   417,
   382,   383,   114,  1119,  1120,    86,   118,   407,   744,   198,
   392,   100,   394,   459,   100,   636,    99,   116,    77,    78,
   119,   393,   416,   417,    99,   100,   408,   116,     1,   411,
   116,   114,    86,   312,   313,   407,   119,  1275,   121,   114,
  1191,   198,   116,    81,    82,   427,   120,   477,   478,   943,
   432,   100,   432,   435,   100,   435,   438,   350,   438,   477,
   478,   443,   956,   443,   100,   447,   100,   447,    99,  1307,
   116,   100,   115,   477,   478,   811,   812,   459,   746,   121,
   116,   784,   116,   114,   478,   100,   458,   116,   651,   100,
   100,   100,   655,   100,   752,   477,   478,   115,   116,   481,
    73,   481,   100,   100,   121,   116,   764,   116,   100,   116,
   778,   493,   941,   493,   707,   497,   498,   121,   116,   116,
   897,   551,    86,   100,   116,   746,   531,   118,   531,   120,
   100,   489,   514,   515,   100,   100,   757,   692,   520,   116,
   888,   762,   763,    99,   765,   527,   116,   551,   100,   531,
   116,   116,   534,   116,   890,    99,   100,   120,   114,     9,
    53,    54,   544,    99,   116,   547,   114,   579,   550,   551,
   114,   576,   579,   576,   556,   579,   558,   579,   114,   116,
   121,    99,   100,   120,    99,   100,   683,   569,   570,   119,
   687,   121,   574,   575,   576,   577,   114,   579,   580,   114,
   582,   583,   584,   585,   586,   587,   588,   589,   590,   591,
   592,   593,   594,   595,   596,   597,   598,   599,   600,   601,
   602,   603,   925,   121,    74,    75,    76,   121,   610,   897,
   121,   631,   421,   422,   527,   617,    99,   100,   531,   121,
   898,   623,   944,   643,    86,   627,    99,   100,    86,    99,
    99,   114,   920,   121,   636,    99,   100,    14,   640,   631,
   642,   114,   644,   113,   114,   114,    99,   100,   650,    99,
   114,   643,   119,  1050,   121,   100,   897,    34,   115,   116,
   662,   114,   100,   576,   666,   100,   475,   476,    99,   991,
   479,    48,   119,  1070,   483,    99,    99,   100,   271,    99,
   100,   683,    99,   685,    67,   687,    99,   689,    99,   100,
   692,   114,    99,   695,   114,   695,   106,   114,   475,   476,
  1056,   114,   479,   114,   118,   119,   483,   114,    85,    99,
   623,     3,   107,     5,   627,    99,   100,   719,    83,   721,
    12,    13,   724,   100,   114,    99,   100,   108,   321,    21,
   114,   733,    99,   100,   103,   104,   105,    99,   100,   116,
   114,    33,   119,   100,   746,   118,   119,   114,   121,  1027,
   752,   926,   114,    99,   100,   757,   758,   759,    99,   761,
   762,   763,   764,   765,   766,    57,   115,   116,   114,   944,
   772,    99,   100,  1141,    66,   119,   778,   121,   115,   781,
   782,    73,   784,    99,    99,   100,   114,    99,    99,   100,
    99,   100,   842,   843,   116,   844,   695,   116,   847,   114,
    99,   100,  1090,   114,    61,   114,  1128,   121,   118,  1050,
   120,   103,   115,   116,   816,   114,   991,   122,   842,   843,
    99,   100,    99,   100,   119,  1147,   828,   829,   116,  1070,
   832,   118,   119,   746,   121,   114,   119,   114,   121,   432,
   842,   843,   435,   587,   588,   438,   591,   592,   593,   594,
   443,   584,   585,   586,   447,     1,   997,   118,   122,   120,
  1133,   863,   118,   119,   866,   121,   118,   118,   120,   120,
   872,   100,   874,   875,   100,   877,   896,   118,   118,   120,
   120,   119,   914,   121,   100,   905,    32,   914,   481,   914,
   100,   914,   914,    53,  1182,   897,   898,   119,   100,   121,
   493,   589,   590,   816,   896,  1183,    52,    53,   113,    55,
    56,   913,   914,   905,   595,   596,   918,    63,   920,   119,
  1193,   121,    74,   925,   926,   100,  1199,    73,  1201,   119,
  1203,   121,  1205,   119,   119,   121,   121,  1210,   118,   941,
   120,   943,   944,  1231,   118,   118,   120,   120,  1236,   118,
  1091,   120,   115,  1128,   956,  1096,   118,   118,   120,   120,
  1101,   118,   118,   120,   120,    99,  1107,  1240,    99,  1242,
  1243,    99,  1147,    99,  1247,   100,    99,  1250,    99,    99,
  1253,    99,   115,  1256,   897,   100,   100,   100,  1261,   991,
   992,   100,   100,   115,     1,   997,   100,    99,   114,   100,
    86,   118,    99,  1053,     9,   151,   152,   118,   120,   155,
    99,  1036,   122,  1036,     3,   120,     5,    99,   116,   118,
   120,   109,  1024,    12,    13,  1027,  1239,  1029,  1169,  1053,
   943,  1244,    21,  1246,  1036,   120,  1249,   118,   122,  1252,
   109,   109,    74,   956,    33,   191,   122,  1260,   118,   100,
   100,  1053,    71,   118,   100,   201,   119,   118,   120,    52,
   120,   838,   100,   840,   841,   100,    73,   100,    57,   100,
   100,   848,  1285,   100,  1287,   100,  1289,    66,   100,   100,
  1293,   120,   859,  1296,    73,   100,  1299,  1300,  1090,  1091,
   100,  1232,  1305,  1095,  1096,  1095,   689,   100,    99,  1101,
  1102,  1103,   695,  1105,    23,  1107,  1108,  1320,   100,   100,
   100,   120,   120,   100,   103,   100,   120,  1119,  1120,  1121,
  1122,   120,  1124,  1036,     0,   271,  1128,    32,   914,   597,
   601,     3,  1273,     5,     6,     7,   599,   598,  1279,  1280,
    12,    13,     1,   361,     3,  1147,     5,     6,     7,    21,
    68,   600,   685,    12,    13,   837,   262,  1182,   304,  1182,
   691,    33,    21,   860,   470,    73,   998,  1169,   314,     3,
   996,     5,  1174,   319,    33,   321,    73,   280,    12,    13,
  1182,  1183,  1323,   840,   666,    57,   333,    21,    -1,  1330,
    -1,     1,    -1,    -1,    66,    -1,  1005,    56,    57,    33,
    -1,    73,    -1,    -1,   350,   982,  1015,    66,    -1,    -1,
    -1,  1236,    -1,  1236,    73,    -1,    -1,   363,     1,    -1,
    -1,    -1,    32,    57,    -1,    -1,    -1,  1229,  1005,  1231,
  1232,   103,    66,  1235,  1236,   107,   382,   383,  1015,    73,
    99,   387,   114,    -1,   103,    -1,    -1,    -1,   107,    -1,
   109,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   118,
    -1,    -1,    -1,    73,   271,    99,    -1,    -1,    -1,  1182,
   863,  1273,    -1,  1275,    -1,    -1,    -1,  1279,  1280,   425,
  1282,    -1,    -1,    -1,    -1,    -1,   432,    -1,    -1,   435,
    73,    -1,   438,    -1,     1,    -1,    -1,   443,    -1,    -1,
    -1,   447,    -1,    -1,    -1,  1307,     3,    -1,     5,    -1,
    -1,    -1,    -1,    -1,   321,    12,    13,    -1,    -1,    -1,
     3,  1323,     5,  1236,    21,    32,    -1,    -1,  1330,    12,
    13,   477,   478,   479,    -1,   481,    33,    -1,    21,    -1,
    -1,    -1,    -1,   350,    -1,    52,    53,   493,   941,    -1,
    33,    -1,   498,    -1,     3,    -1,     5,     6,     7,    56,
    57,    -1,    -1,    12,    13,    -1,    73,    -1,    -1,    66,
    -1,    -1,    21,    56,    57,    -1,    73,    -1,    -1,    -1,
    -1,   527,    -1,    66,    33,   531,    -1,    -1,   534,    -1,
    73,     1,    -1,     3,    -1,     5,     6,     7,    -1,    -1,
    -1,    -1,    12,    13,   997,   551,   552,    56,    57,    -1,
   556,    21,   109,    -1,    -1,    -1,    -1,    66,    -1,    -1,
   103,    -1,    -1,    33,    73,   432,   109,    -1,   435,    -1,
   576,   438,    -1,   579,    -1,    -1,   443,    -1,    -1,    -1,
   447,    -1,    -1,    -1,   151,    -1,    56,    57,    -1,    -1,
    99,    -1,    -1,    -1,   103,    -1,    66,   603,   107,    -1,
   109,   271,    -1,    73,   610,    -1,    -1,    -1,    -1,   118,
    -1,   617,    -1,    -1,   481,    -1,    -1,   623,    -1,    -1,
    -1,   627,    -1,    -1,   191,    -1,   493,    -1,   271,    99,
   636,     1,    -1,   103,   201,    -1,    -1,   107,  1091,   109,
    -1,    -1,  1095,  1096,   650,     1,    -1,    -1,  1101,    -1,
    -1,   321,    -1,    -1,  1107,    -1,    -1,     3,   621,     5,
    -1,    -1,    -1,   626,   531,    -1,    12,    13,    -1,    -1,
    -1,    -1,     3,    -1,     5,    21,    -1,    -1,   321,    -1,
   350,    12,    13,   689,    -1,    -1,    -1,    33,    -1,   695,
    21,    -1,    -1,   363,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    33,    73,   271,    -1,    -1,   350,    -1,   576,
    56,    57,    -1,   719,    -1,   721,  1169,    73,   724,    -1,
    66,    -1,    -1,    -1,    -1,    56,    57,    73,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    66,    -1,   304,    -1,    -1,
   746,    -1,    73,    -1,    -1,    -1,    -1,   314,    -1,    -1,
    -1,   757,   319,    -1,   321,    -1,   762,   763,    -1,   765,
   766,    -1,   432,   109,    -1,   435,   772,    -1,   438,   336,
    -1,    -1,    -1,   443,    -1,    -1,    -1,   447,   109,  1232,
   347,   348,    -1,   350,    -1,   352,   353,   354,   355,   432,
    -1,    -1,   435,    -1,    -1,   438,   363,    -1,    -1,    -1,
   443,    -1,    -1,    -1,   447,    -1,    -1,    -1,    -1,    -1,
   816,   481,    -1,    -1,    -1,   382,   383,    -1,    -1,    -1,
  1273,    -1,   689,   493,    -1,   392,  1279,  1280,   695,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,   842,   843,   481,    -1,
    -1,    -1,   848,    -1,    -1,    -1,    -1,    -1,    -1,    14,
   493,    -1,   815,    18,    -1,    -1,    -1,   863,    -1,    -1,
    -1,   531,    -1,    28,    29,   432,    31,    -1,   435,    34,
  1323,   438,    -1,    -1,    -1,    -1,   443,  1330,    -1,    -1,
   447,    -1,    -1,    48,    -1,    -1,    51,    -1,   531,    -1,
    -1,   897,    -1,    -1,    -1,    -1,    61,    62,    63,    64,
    65,   271,    -1,    -1,    -1,    -1,   576,   913,   914,    -1,
   477,   478,    -1,    -1,   481,   271,    -1,    -1,     3,    -1,
     5,     6,     7,    -1,    -1,    -1,   493,    12,    13,    -1,
    -1,   498,    -1,   576,    -1,   941,    21,   943,    -1,    -1,
    -1,    -1,    -1,     3,     1,     5,    -1,    -1,    33,    -1,
   956,   321,    12,    13,    -1,   120,    -1,    -1,    -1,    -1,
   527,    21,    -1,    -1,   531,   321,    -1,   534,    -1,    -1,
    -1,    56,    57,    33,    -1,    32,    -1,    -1,    -1,    -1,
    -1,    66,    -1,    -1,   551,    -1,    -1,    -1,    73,    -1,
    -1,   997,    -1,    -1,    -1,    -1,   863,    57,    -1,  1005,
    -1,    -1,    -1,    -1,    -1,    -1,    66,    -1,    -1,   576,
    -1,    -1,   579,    73,    99,    -1,    73,    -1,   103,   689,
    -1,    -1,   107,  1029,   109,   695,    -1,    -1,    85,    -1,
  1036,    -1,    -1,   118,    -1,    -1,   603,    -1,    -1,    -1,
    -1,    -1,    -1,   610,    -1,    -1,   689,  1053,  1054,    -1,
   617,    -1,   695,    -1,    -1,    -1,   623,    -1,    -1,    -1,
   627,    -1,   432,    -1,  1027,   435,    -1,    -1,   438,   636,
    -1,    -1,    -1,   443,   941,    -1,   432,   447,    -1,   435,
    -1,    -1,   438,   650,    -1,  1091,    -1,   443,  1051,  1095,
  1096,   447,    -1,    -1,    -1,  1101,    -1,    -1,    -1,    -1,
    -1,  1107,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,   481,  1118,  1119,  1120,    -1,  1122,    -1,    -1,    -1,
    -1,    -1,   689,   493,    -1,   481,   183,    -1,   695,    -1,
   997,    -1,    -1,    -1,    -1,    -1,    -1,   493,     3,    -1,
     5,    -1,    -1,    -1,    -1,    -1,    -1,    12,    13,    -1,
    -1,    -1,   719,    -1,   721,    -1,    21,   724,   380,   381,
   382,   383,    -1,  1169,   221,   387,  1129,    -1,    33,  1036,
    -1,  1134,    -1,  1136,    -1,  1138,  1182,  1140,   400,   746,
    -1,  1144,    -1,    -1,    -1,   752,    -1,    -1,    -1,    -1,
   757,    -1,    57,   863,    -1,   762,   763,   764,   765,    -1,
    -1,    66,    -1,   425,   426,    -1,    -1,    -1,    73,    -1,
    -1,    -1,    -1,    -1,   271,    -1,    -1,    -1,    -1,    -1,
   863,    -1,    -1,   280,  1091,    -1,  1232,    -1,  1095,  1096,
  1236,  1194,  1195,  1196,  1101,    -1,    -1,  1200,    -1,  1202,
  1107,  1204,   299,  1206,   914,    -1,    -1,    -1,  1211,   816,
    -1,    -1,    -1,    -1,  1217,   312,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,   321,    -1,    -1,  1273,    -1,    -1,
    -1,   941,    -1,  1279,  1280,   842,   843,    89,    90,    91,
    92,    93,    94,    95,    96,    97,    98,    -1,    -1,    -1,
    -1,    -1,    -1,   350,   351,    -1,   863,    -1,   941,    -1,
   943,    -1,  1169,    -1,    -1,    -1,   363,   119,    -1,    -1,
    -1,    -1,    -1,   956,    -1,  1182,    -1,  1323,    -1,   689,
   542,   543,    -1,    -1,  1330,   695,    -1,   997,   550,   551,
   897,   898,    -1,   689,   556,    -1,   558,    -1,    -1,   695,
     3,    -1,     5,     6,     7,    -1,    -1,   914,    -1,    12,
    13,    14,    -1,    -1,   997,    -1,    -1,    -1,    21,   416,
    -1,    -1,    -1,    -1,   421,  1232,  1036,    -1,    -1,  1236,
    33,    34,    -1,    -1,   941,   432,   943,    -1,   435,    -1,
    -1,   438,    -1,    -1,    -1,    48,   443,    -1,    -1,   956,
   447,   613,   614,  1036,    57,    -1,   766,    -1,    -1,    -1,
    -1,    -1,   772,    66,    -1,     3,  1273,     5,     6,     7,
    73,    -1,  1279,  1280,    12,    13,    -1,    -1,    -1,    -1,
    -1,  1091,    -1,    21,   481,  1095,  1096,    -1,    -1,    -1,
   997,  1101,    -1,    -1,    -1,    33,   493,  1107,    -1,    -1,
   103,    -1,    -1,    -1,   107,    -1,    -1,    -1,  1091,    -1,
    -1,   114,  1095,  1096,    -1,    -1,  1323,    -1,  1101,    57,
  1027,    -1,  1029,  1330,  1107,    -1,    -1,    -1,    66,  1036,
    -1,    -1,    -1,    -1,   531,    73,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,   542,  1053,    -1,     3,    -1,
     5,     6,     7,   863,    -1,    -1,    -1,    12,    13,  1169,
    -1,    99,    -1,    -1,    -1,   103,    21,   863,    -1,   107,
    -1,    -1,  1182,    -1,    -1,    -1,   114,    -1,    33,   576,
    -1,    -1,    -1,    -1,  1091,    -1,  1169,    -1,  1095,  1096,
   752,    -1,    -1,    -1,  1101,    -1,   758,   759,    -1,  1182,
  1107,    56,    57,   913,    -1,    -1,    -1,    -1,    -1,    -1,
     3,    66,     5,     6,     7,    -1,   613,    -1,    73,    12,
    13,    14,  1232,    -1,   621,    -1,  1236,    -1,    21,   626,
    -1,   941,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    33,    34,    -1,    -1,    99,   941,    -1,    -1,   103,  1232,
    -1,   648,   107,  1236,   109,    48,   653,    -1,    -1,   114,
    -1,    -1,  1169,  1273,    57,    -1,    -1,    -1,    14,  1279,
  1280,    -1,    18,    66,    -1,  1182,  1183,    -1,    -1,    -1,
    73,    -1,    28,    29,    -1,    31,    -1,   997,    34,    -1,
  1273,    -1,   689,    86,    -1,    -1,  1279,  1280,   695,    -1,
    -1,   997,    48,    -1,    -1,    51,    99,    -1,    -1,    -1,
   103,    -1,    -1,  1323,   107,    61,    62,    63,    64,    65,
  1330,   114,    -1,    -1,    -1,  1232,    -1,    -1,    -1,  1236,
    -1,    -1,    -1,    -1,    -1,    -1,   898,    -1,    -1,    -1,
  1323,    -1,    -1,    -1,    -1,    -1,    -1,  1330,    -1,     3,
   747,     5,     6,     7,    -1,   752,    -1,    -1,    12,    13,
    14,   758,    -1,    -1,    -1,    -1,  1273,    21,    -1,   766,
    -1,    -1,  1279,  1280,   120,   772,    -1,    -1,    -1,    33,
    34,  1091,     1,    -1,   781,  1095,  1096,    -1,    -1,    -1,
    -1,  1101,    -1,    -1,    48,  1091,    -1,  1107,    -1,  1095,
  1096,    -1,    56,    57,    -1,  1101,  1116,    -1,    -1,  1119,
  1120,  1107,    66,    32,    -1,    -1,  1323,    -1,   815,    73,
    -1,   818,    -1,  1330,    -1,    -1,     3,    -1,     5,     6,
     7,    -1,    -1,    52,    53,    12,    13,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    21,    99,    -1,   164,    -1,   103,
    -1,    -1,    -1,   107,    73,   109,    33,    -1,    -1,  1169,
   114,    14,    -1,    -1,    -1,    18,   863,    -1,    -1,    88,
    -1,    90,    -1,  1169,    -1,    28,    29,    -1,    31,    -1,
    57,    34,    -1,    -1,   201,    -1,    -1,    -1,    -1,    66,
  1052,  1053,  1054,    -1,    -1,    48,    73,  1059,    51,    -1,
    -1,   898,    -1,    -1,    -1,    -1,    -1,    -1,    61,    62,
    63,    64,    65,    -1,    -1,    -1,   913,   914,    -1,    -1,
    -1,    -1,  1232,    -1,    -1,     3,   103,     5,     6,     7,
   107,    -1,   151,    -1,    12,    13,  1232,    -1,     3,    -1,
     5,   160,    -1,    21,   941,    -1,   943,    12,    13,    -1,
    -1,    -1,    -1,    -1,    -1,    33,    21,   954,    -1,   956,
  1122,    -1,  1124,  1273,    -1,    -1,   963,   120,    33,  1279,
  1280,    -1,   191,    -1,    -1,    -1,    -1,  1273,    56,    57,
    -1,    -1,   201,  1279,  1280,    -1,    -1,    -1,    66,    -1,
   307,    -1,    57,    -1,    -1,    73,    -1,    -1,    -1,    -1,
   997,    66,    -1,    -1,    -1,   322,    -1,    -1,    73,    -1,
   327,    -1,    -1,  1323,    -1,    -1,    -1,    -1,    -1,    -1,
  1330,    99,   100,    -1,    -1,   103,    -1,  1323,    -1,   107,
  1027,   109,    -1,    -1,  1330,    -1,   114,    -1,    -1,  1036,
    -1,    -1,    14,    -1,    -1,    -1,    18,    -1,    -1,    -1,
    -1,    -1,   271,    -1,  1051,    -1,    28,    29,    -1,    31,
    -1,    -1,    34,    -1,    -1,    -1,    -1,   286,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    48,    -1,    41,    51,
    -1,    -1,    -1,    -1,   303,   304,    -1,    -1,    -1,    61,
    62,    63,    64,    65,  1091,   314,    -1,    -1,  1095,  1096,
   319,   418,   321,    -1,  1101,    -1,    -1,    -1,    -1,    -1,
  1107,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,  1116,
  1117,    -1,  1119,  1120,    -1,    -1,    -1,    -1,    -1,    -1,
     1,   350,  1129,    -1,    -1,    -1,    -1,  1134,    -1,  1136,
    -1,  1138,    14,  1140,   363,    -1,    18,  1144,   120,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    28,    29,    -1,    31,
    -1,    32,    34,   382,   383,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,  1169,    -1,   393,   394,    48,    -1,    -1,    51,
    -1,    52,    53,    -1,    -1,  1182,  1183,    -1,    -1,    61,
    62,    63,    64,    65,   511,   512,    -1,  1194,  1195,  1196,
    -1,   164,    73,  1200,    -1,  1202,    -1,  1204,    -1,  1206,
    -1,    -1,    -1,   432,  1211,    -1,   435,   534,    -1,   438,
  1217,    -1,    -1,    -1,   443,    19,    20,    -1,   447,    -1,
    24,    -1,    -1,    -1,    -1,  1232,    30,    -1,   201,  1236,
    -1,    35,    36,    37,    38,    39,    40,   210,   120,    43,
    44,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   477,   478,
    -1,    -1,   481,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,   493,    -1,  1273,    -1,    -1,   498,
   151,    -1,  1279,  1280,    -1,    -1,   603,    -1,    -1,    -1,
    -1,    -1,     3,    -1,     5,     6,     7,    -1,    -1,    -1,
   617,    12,    13,    14,    -1,    -1,    -1,    -1,   527,    -1,
    21,    -1,   531,    -1,    -1,   534,    -1,    -1,    -1,   636,
   191,    -1,    33,    34,    -1,    -1,  1323,    -1,   191,    -1,
   201,    -1,   551,  1330,    -1,   652,    -1,    48,    -1,   656,
   657,    -1,    -1,    -1,   307,    -1,    57,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    66,    -1,   576,    -1,   322,
   579,    -1,    73,    -1,   327,    -1,    -1,     3,    -1,     5,
    -1,    -1,     8,    -1,    -1,    86,    12,    13,    -1,    -1,
    -1,    -1,    -1,    -1,   603,    21,   257,    -1,    99,    -1,
    -1,   610,   103,    -1,    -1,    -1,   107,    33,   617,    -1,
   271,    -1,    -1,   114,   623,    -1,    -1,    -1,   627,    -1,
    -1,    -1,   631,    -1,    -1,    -1,    -1,   636,    -1,    -1,
    -1,    57,    58,    59,    60,    61,    -1,    -1,    -1,   746,
    66,   650,    -1,   304,    -1,    -1,     3,    73,     5,     6,
     7,   304,    -1,   314,    -1,    12,    13,    14,   319,   412,
   321,   314,    -1,    -1,    21,   418,   319,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    33,    34,    -1,    -1,
   689,    -1,    -1,    -1,    -1,    -1,   695,    -1,    -1,   350,
    -1,    48,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    56,
    57,    -1,   363,    -1,    -1,    -1,    -1,    -1,    -1,    66,
   719,    -1,   721,    -1,    -1,   724,    73,    -1,   825,   826,
    -1,   382,   383,    -1,    -1,     3,    -1,     5,     6,     7,
    -1,    -1,    -1,    -1,    12,    13,    -1,   746,    -1,    -1,
    -1,    -1,    99,    21,    -1,    -1,   103,    -1,   757,    -1,
   107,    -1,   109,   762,   763,    33,   765,   114,   511,   512,
    -1,    -1,    -1,    -1,    -1,    -1,   427,    -1,    -1,    -1,
    -1,   432,    -1,    -1,   435,    -1,    -1,   438,    56,    57,
    -1,   534,   443,    -1,    -1,    -1,   447,    -1,    66,    -1,
    -1,    -1,    -1,    -1,    -1,    73,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,   816,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,   477,   478,    -1,    -1,
   481,    99,    -1,    -1,    -1,   103,    -1,    -1,    -1,   107,
    -1,   109,   493,   842,   843,    -1,   943,   498,    -1,    -1,
    -1,    -1,    -1,    -1,   497,    -1,    -1,    -1,    -1,   956,
   603,    -1,    -1,    -1,   863,    -1,    -1,   610,    -1,    -1,
    -1,   514,    -1,    -1,   617,    -1,   527,   520,    -1,    -1,
   531,    -1,    -1,   534,    -1,    -1,    -1,   952,   953,    -1,
    -1,    -1,    -1,   636,    -1,   960,   961,   962,   897,    -1,
   551,   966,    -1,   646,    -1,   970,    -1,   972,    -1,   652,
    -1,    -1,    -1,   656,   657,   914,    -1,    -1,    -1,    -1,
    -1,   986,    -1,    -1,    -1,   576,   569,    -1,   579,    -1,
    -1,    -1,    -1,   576,    -1,    -1,    -1,     1,    -1,    -1,
    -1,    -1,   941,    -1,   943,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,   603,    -1,    -1,    -1,    -1,   956,    -1,   610,
    -1,    -1,    -1,    -1,    -1,    -1,   617,    -1,    32,  1066,
    -1,  1068,   623,    -1,    -1,    -1,   627,    -1,    -1,  1076,
    -1,  1078,    -1,    -1,    -1,   636,    -1,    -1,    52,    53,
    -1,    -1,    -1,   636,    -1,    -1,    -1,    -1,   997,   650,
    -1,   644,  1067,   746,  1069,    -1,  1103,   650,  1105,    73,
    -1,  1108,  1077,    -1,  1079,   666,    -1,    -1,  1083,   662,
  1085,    -1,  1087,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
  1029,    -1,   683,    -1,   685,    14,   687,  1036,   689,    -1,
    19,    20,    -1,    -1,   695,    24,    -1,    -1,    -1,    -1,
    -1,    30,    -1,    -1,  1053,    34,    35,    36,    37,    38,
    39,    40,    -1,    -1,    43,    44,    -1,    -1,   719,    48,
   721,    -1,    -1,   724,    -1,    -1,   719,  1142,   721,    -1,
    -1,   724,   825,   826,    14,  1182,    -1,   151,    18,    -1,
  1155,    -1,  1091,    -1,    -1,   746,  1095,  1096,    28,    29,
    -1,    31,  1101,    -1,    34,    -1,   757,    -1,  1107,    -1,
    -1,   762,   763,    -1,   765,    -1,    -1,    -1,    48,    -1,
    -1,    51,   765,    -1,    -1,    -1,    -1,   191,    -1,  1194,
    -1,    61,    62,    63,    64,    65,    -1,   201,    -1,  1236,
    -1,    -1,    -1,  1208,  1209,    -1,    -1,    -1,  1213,  1214,
  1215,  1216,  1217,    -1,  1219,  1220,  1221,  1222,  1223,  1224,
    -1,    -1,    -1,    -1,    -1,   816,    -1,    -1,    -1,    -1,
  1169,    -1,     3,    -1,     5,     6,     7,    -1,   164,    -1,
    -1,    12,    13,  1182,    -1,   828,   829,    -1,    -1,   832,
    21,   842,   843,   257,    -1,    -1,    -1,   183,   184,    -1,
   943,    -1,    33,    -1,    -1,    -1,    -1,   271,    -1,    -1,
    -1,    -1,   863,   956,    -1,   201,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,   866,    -1,    56,    57,    -1,    -1,   872,
    -1,   874,   875,  1232,   877,    66,    -1,  1236,    -1,    -1,
   304,    -1,    73,    -1,    -1,    -1,   897,    -1,    -1,    -1,
   314,    -1,    -1,    -1,    -1,   319,    -1,   321,    -1,    -1,
    -1,    -1,    -1,   914,   191,    -1,    -1,    -1,    99,    -1,
    -1,    -1,   103,    -1,  1273,   918,   107,    -1,   109,    -1,
  1279,  1280,    -1,   114,    -1,    -1,   350,    -1,    -1,    -1,
   941,    -1,   943,    -1,    -1,    -1,    -1,    -1,    -1,   363,
    14,    -1,    -1,    -1,    18,   956,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    28,    29,    -1,    31,   382,   383,
    34,   307,    -1,  1066,  1323,  1068,   312,   313,    -1,    -1,
    -1,  1330,    -1,  1076,    48,  1078,   322,    51,    -1,    -1,
    -1,   327,    -1,    -1,    -1,    -1,   997,    61,    62,    63,
    64,    65,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
  1103,    -1,  1105,   427,    -1,  1108,    -1,    -1,   432,    -1,
    -1,   435,    -1,    -1,   438,    -1,    -1,   304,  1029,   443,
    -1,    -1,    -1,   447,    -1,  1036,    -1,   314,    -1,    -1,
    -1,    -1,   319,    -1,   183,   184,    -1,    -1,    -1,    -1,
    -1,    -1,  1053,    -1,    -1,    -1,   120,    -1,    -1,    -1,
    -1,    -1,    -1,   477,   478,    -1,    -1,   481,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   493,
    -1,    -1,   418,    -1,   498,   421,   422,    -1,    -1,  1182,
  1091,    -1,    -1,    -1,  1095,  1096,    -1,    -1,    -1,    -1,
  1101,     3,    -1,     5,     6,     7,  1107,    -1,    -1,    -1,
    12,    13,    14,   527,    -1,  1108,    -1,   531,    -1,    21,
   534,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    33,    34,    -1,    -1,    -1,    -1,   551,    -1,    -1,
    -1,    -1,    -1,  1236,    -1,    -1,    48,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    57,    -1,    -1,    -1,    -1,
    -1,    -1,   576,    -1,    66,   579,    -1,    -1,  1169,    -1,
    -1,    73,    -1,   312,   313,   511,   512,    -1,    -1,    -1,
    -1,  1182,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   603,
    -1,    -1,    -1,    -1,    -1,    -1,   610,    99,   100,    -1,
    -1,   103,    -1,   617,    -1,   107,    -1,   346,    -1,   623,
    -1,    -1,   114,   627,    -1,    -1,    -1,    -1,    -1,    -1,
   497,    -1,   636,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,  1232,    -1,    -1,    -1,  1236,   650,   514,   515,    -1,
    -1,   380,   381,   520,    -1,    -1,    -1,    -1,   387,    -1,
    -1,     1,   666,    -1,    -1,    -1,    -1,   534,    -1,    -1,
    -1,   400,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   683,
    -1,   685,  1273,   687,    -1,   689,    -1,    -1,  1279,  1280,
    -1,   695,   421,   422,    -1,   424,   425,   426,    -1,    -1,
    40,    41,   569,    -1,    -1,    -1,    -1,   574,   575,   576,
    50,    51,    52,    53,    -1,   719,    -1,   721,    -1,    -1,
   724,    -1,    -1,    63,    -1,    -1,   652,   653,   654,    -1,
   656,   657,  1323,    73,    -1,    -1,    -1,    -1,    -1,  1330,
    -1,    -1,   746,    -1,    -1,    85,    86,    -1,    -1,    -1,
   617,    14,    -1,   757,    -1,    18,    -1,    97,   762,   763,
    -1,   765,    -1,    -1,    -1,    28,    29,    -1,    31,   636,
    -1,    34,    -1,   640,    -1,   642,    -1,   644,    -1,    -1,
    -1,    -1,    -1,   650,    -1,    48,    -1,    -1,    51,    -1,
    -1,    -1,    -1,    -1,    -1,   662,    -1,    -1,    61,    62,
    63,    64,    65,    -1,    -1,    -1,    -1,    -1,   148,    -1,
    -1,   151,   816,   542,   543,    -1,    -1,     3,    -1,     5,
     6,     7,    -1,    86,   164,    -1,    12,    13,    14,    -1,
    -1,    -1,    -1,    -1,    -1,    21,    -1,    -1,   842,   843,
    -1,    -1,    -1,   183,   184,    -1,    -1,    33,    34,    -1,
    -1,    -1,   719,    -1,   721,    -1,   196,   724,   198,   863,
    -1,   201,    48,    -1,    -1,    -1,   733,    -1,    -1,    -1,
   210,    57,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    66,   221,   222,    -1,    -1,    -1,    -1,    73,    -1,    -1,
   757,    -1,    -1,   897,   761,    -1,   763,    -1,   765,   825,
   826,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
   914,    -1,    -1,    99,   100,    -1,    -1,   103,    -1,    -1,
    -1,   107,    -1,    -1,   653,   654,    -1,    -1,   114,    14,
    -1,    -1,    -1,    18,    -1,    -1,    -1,   941,    -1,   943,
    -1,    -1,    -1,    28,    29,    -1,    31,    -1,    -1,    34,
    -1,    -1,   956,    -1,    -1,    -1,    -1,    -1,    -1,   299,
   300,   828,   829,    48,    -1,   832,    51,   307,    -1,    -1,
    -1,    -1,   312,   313,    -1,    -1,    61,    62,    63,    64,
    65,    -1,   322,    -1,   713,    -1,    -1,   327,    -1,    -1,
    -1,    -1,    -1,   997,    -1,    -1,    -1,    -1,    -1,   866,
    -1,    -1,    -1,    -1,    -1,   872,   346,   874,   875,    -1,
   877,    -1,    -1,    -1,    -1,    -1,    -1,   943,    -1,    -1,
    -1,    -1,    -1,   752,    -1,  1029,    -1,    -1,   954,   955,
   956,    -1,  1036,    -1,    -1,   120,    -1,   963,   964,    -1,
   380,   381,   382,   383,    -1,    -1,    -1,   387,    -1,  1053,
    -1,   918,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
   400,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,   412,    -1,    -1,    -1,   416,   417,   418,    14,
    -1,   421,   422,    18,   424,   425,   426,  1091,    -1,    -1,
    -1,  1095,  1096,    28,    29,    -1,    31,  1101,    -1,    34,
    -1,    -1,    -1,  1107,    -1,    -1,    -1,    -1,    -1,   838,
    -1,   840,    -1,    48,    -1,    -1,    51,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    61,    62,    63,    64,
    65,    -1,    -1,   473,    -1,   475,   476,   477,   478,   479,
  1066,    -1,  1068,   483,    -1,     3,    -1,     5,     6,     7,
  1076,    86,  1078,    -1,    12,    13,    14,  1024,    -1,    -1,
    -1,    -1,  1029,    21,    -1,  1169,    -1,    -1,    -1,    -1,
    -1,   511,   512,    -1,    -1,    33,    34,  1103,  1182,  1105,
    -1,    -1,  1108,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    48,    -1,    -1,    -1,   534,    -1,    -1,    -1,    -1,    57,
    -1,    -1,   542,   543,    -1,    -1,    -1,   547,    66,    -1,
   550,   551,    -1,    -1,    -1,    73,   556,    -1,   558,    -1,
    -1,    -1,    -1,    -1,    -1,   954,   955,    -1,  1232,    -1,
    -1,    -1,  1236,    -1,   963,   964,  1103,   577,  1105,   579,
    -1,  1108,    -1,    -1,    -1,   103,    -1,    -1,    -1,   107,
    -1,    -1,    -1,    -1,    -1,    -1,  1182,  1183,    -1,    -1,
    -1,    -1,    -1,   603,    -1,    -1,    -1,    -1,    -1,  1273,
   610,    -1,    -1,   613,   614,  1279,  1280,   617,    -1,     1,
    -1,     3,    -1,     5,    -1,    -1,     8,    -1,    -1,    -1,
    12,    13,    -1,    -1,    -1,    -1,   636,    -1,    -1,    21,
    -1,    -1,    -1,    -1,    -1,    -1,   646,    -1,   648,   649,
  1236,    33,   652,   653,   654,    -1,   656,   657,    -1,  1323,
    -1,  1050,    -1,  1052,    -1,  1054,  1330,    -1,    -1,    -1,
  1059,    -1,    -1,    -1,    -1,    57,    58,    59,    60,    61,
    -1,  1070,    -1,    -1,    66,    -1,    -1,    -1,    -1,   689,
    -1,    73,    -1,    -1,    -1,   695,    -1,     3,    -1,     5,
     6,     7,    -1,    -1,    86,    -1,    12,    13,    14,    -1,
    -1,    -1,    -1,   713,    -1,    21,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    33,    34,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,    -1,
    -1,    -1,    48,    -1,    -1,    -1,   746,    -1,    -1,    -1,
    -1,    57,   752,    -1,    -1,    -1,    -1,   757,   758,   759,
    66,    -1,   762,   763,    -1,   765,    -1,    73,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    40,    41,    -1,    -1,
    -1,   781,   782,    -1,    -1,    -1,    50,    51,    52,    53,
    -1,    -1,    -1,    99,  1183,    -1,    -1,   103,    -1,    63,
    -1,   107,    -1,    -1,    -1,    -1,    -1,    -1,   114,    73,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    85,    86,    -1,    -1,   825,   826,    -1,    -1,    -1,
    -1,    -1,    -1,    97,    -1,    -1,    -1,    -1,   838,    -1,
   840,   841,   842,   843,    -1,    -1,    -1,    -1,   848,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   859,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,     3,    -1,     5,     6,     7,    -1,    -1,    -1,
    -1,    12,    13,    14,   148,    -1,    -1,   151,    19,    20,
    21,    -1,    -1,    24,    -1,    -1,    -1,   897,   898,    30,
   164,    -1,    33,    34,    35,    36,    37,    38,    39,    40,
    -1,    -1,    43,    44,    -1,    -1,    -1,    48,    -1,   183,
   184,     3,    -1,     5,     6,     7,    57,    -1,    -1,    -1,
    12,    13,   196,    -1,   198,    66,    -1,   201,    -1,    21,
    -1,   941,    73,   943,    -1,    -1,   210,    -1,    -1,    -1,
    -1,    33,    -1,    -1,   954,   955,   956,   221,   222,    -1,
    -1,    -1,    -1,   963,   964,    -1,    -1,    -1,    99,    -1,
    -1,    -1,   103,    -1,    56,    57,   107,    -1,    -1,    -1,
   980,    -1,   982,   114,    66,    -1,    -1,   164,    -1,    -1,
    -1,    73,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,  1001,    -1,    -1,    -1,  1005,   183,   184,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,  1015,    -1,    99,    -1,    -1,
    -1,   103,    -1,    -1,   201,   107,    -1,   109,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,   299,   300,    -1,    -1,    -1,
    -1,    -1,    -1,   307,    -1,    -1,    -1,    -1,   312,   313,
  1050,    -1,  1052,  1053,  1054,    -1,    -1,    -1,   322,  1059,
    -1,    -1,    -1,   327,    -1,    -1,  1066,    -1,  1068,    -1,
  1070,    -1,    -1,    -1,    -1,    -1,  1076,    40,  1078,    -1,
    -1,    -1,   346,    -1,    -1,    -1,    -1,    -1,    -1,    52,
    53,    -1,    55,    56,    57,    58,    59,    60,    61,    -1,
    63,    -1,    -1,  1103,    -1,  1105,    -1,    -1,  1108,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,   380,   381,   382,   383,
    -1,  1121,  1122,   387,  1124,    -1,    -1,    -1,    -1,    -1,
   307,    -1,    -1,    -1,    97,    -1,   400,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,   107,   322,    -1,    -1,   412,    -1,
   327,    -1,   416,   417,    -1,    -1,    -1,   421,   422,    -1,
   424,   425,   426,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,  1172,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,  1182,  1183,    -1,    -1,    -1,    -1,   151,   152,
   153,   154,   155,    -1,    -1,    -1,    -1,    -1,   161,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   473,
    -1,   475,   476,   477,   478,   479,    -1,    -1,    -1,   483,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,  1235,  1236,    -1,    -1,    -1,
    -1,   418,    -1,    -1,   421,   422,    -1,   511,   512,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     3,    -1,
     5,     6,     7,    -1,    -1,    -1,    -1,    12,    13,    14,
   534,    -1,    -1,    -1,    -1,    45,    21,    -1,   542,   543,
    -1,    -1,  1282,   547,    -1,    -1,   550,   551,    33,    34,
    -1,    -1,   556,    -1,   558,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    48,    -1,    -1,    -1,    -1,    -1,   272,
    -1,    -1,    57,   577,    -1,   579,    -1,    -1,    -1,    -1,
    -1,    66,    -1,    -1,    -1,    -1,    -1,    -1,    73,    -1,
    -1,    -1,    -1,    -1,   511,   512,    -1,    -1,     3,   603,
     5,     6,     7,    -1,    -1,    -1,   610,    12,    13,   613,
   614,    -1,    -1,   617,    99,    -1,    21,    -1,   103,    -1,
    -1,    -1,   107,    -1,    -1,    -1,    -1,    -1,    33,   114,
    -1,    -1,   636,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,   646,   346,   648,   649,    -1,    -1,    -1,    -1,
    -1,    56,    57,   163,    -1,   165,    -1,    -1,    -1,    -1,
    -1,    66,    -1,    -1,    -1,    -1,    -1,    -1,    73,    -1,
   180,    -1,    -1,    -1,    -1,    -1,    -1,   187,    -1,   382,
   383,   384,   385,   386,   387,   689,    -1,    -1,    -1,    -1,
    -1,   695,    -1,    -1,    99,    -1,    -1,   400,   103,    -1,
    -1,    -1,   107,    -1,   109,    -1,    -1,    -1,    -1,   713,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,   424,   425,   426,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,   652,    -1,    -1,    -1,   656,
   657,    -1,   746,    -1,    -1,    -1,    -1,    -1,   752,    -1,
    -1,    -1,    -1,   757,   758,   759,    -1,    -1,   762,   763,
    -1,   765,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
   473,    -1,    -1,    -1,   477,   478,   479,   781,   782,    -1,
   483,    -1,   485,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,   306,    -1,    -1,    -1,
   310,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,   326,    -1,    -1,    -1,
    -1,    -1,   332,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,   838,    -1,   840,    -1,   842,   843,
    -1,    -1,    -1,    -1,   547,    -1,    -1,    -1,   551,   552,
   553,   554,   555,   556,    -1,   558,    -1,     3,    -1,     5,
     6,     7,    -1,    -1,    -1,    -1,    12,    13,    14,    -1,
    -1,    -1,    18,    -1,    -1,    21,   579,    -1,    -1,    -1,
    -1,    -1,    28,    29,    -1,    31,    -1,    33,    34,    -1,
    -1,    -1,    -1,   897,   898,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    48,   413,    -1,    51,    -1,    -1,   825,   826,
    56,    57,    -1,    -1,   424,    61,    62,    63,    64,    65,
    66,   838,    -1,   840,   841,    -1,    -1,    73,    -1,    -1,
    -1,   848,    -1,    -1,    -1,    -1,    -1,   941,    -1,    -1,
    -1,    -1,   859,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    99,    -1,    -1,    -1,   103,    -1,    -1,
    -1,   107,    -1,   109,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,   118,    -1,    -1,     3,   980,     5,     6,     7,
    -1,    -1,    -1,    -1,    12,    13,    14,    -1,    -1,    -1,
    -1,   501,    -1,    21,   504,   505,   506,  1001,    -1,   509,
   510,  1005,    -1,    -1,    -1,    33,    34,    -1,    -1,    -1,
   713,  1015,   522,   523,   524,   525,    -1,    -1,    -1,    -1,
    48,    -1,    -1,    -1,   534,   535,   943,    -1,    56,    57,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    66,   956,
    -1,    -1,    -1,    -1,    -1,    73,  1050,    -1,  1052,  1053,
  1054,    -1,    -1,    -1,    -1,  1059,    -1,    -1,     3,    -1,
     5,     6,     7,    -1,    -1,   982,  1070,    12,    13,    14,
    -1,    99,    -1,    -1,    -1,   103,    21,    -1,    -1,   107,
    -1,   109,    -1,    -1,    -1,    -1,    -1,    -1,    33,    34,
    -1,    -1,    -1,   603,    -1,    -1,    -1,    -1,    -1,    -1,
   610,    -1,    -1,    48,    -1,    -1,    -1,   617,    -1,    -1,
    -1,    56,    57,    -1,    -1,    -1,    -1,  1121,  1122,    -1,
  1124,    66,    -1,    -1,    -1,    -1,   636,    -1,    73,    -1,
    -1,    -1,    -1,    -1,    -1,   838,    -1,   647,    -1,   842,
   843,   844,   845,   846,   847,   848,    -1,    -1,    -1,  1066,
    -1,  1068,    -1,    -1,    99,    -1,   859,    -1,   103,  1076,
    -1,  1078,   107,    -1,   109,    -1,    -1,    -1,  1172,    -1,
    -1,     3,    -1,     5,     6,     7,    -1,    -1,    -1,    -1,
    12,    13,    14,    -1,    -1,    -1,  1103,    -1,  1105,    21,
    -1,  1108,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   902,
   903,    33,    34,    -1,    -1,    -1,   716,   717,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    48,   727,   728,   729,
   730,    -1,    -1,    -1,    56,    57,    -1,    -1,    -1,    -1,
    -1,  1235,    -1,    -1,    66,    -1,   746,    -1,    -1,    -1,
     3,    73,     5,     6,     7,    -1,    -1,   757,    -1,    12,
    13,    -1,   762,   763,    -1,   765,    -1,    -1,    21,    -1,
    -1,    -1,    -1,    -1,    -1,  1182,    -1,    99,    -1,    -1,
    33,   103,    -1,    -1,    -1,   107,     1,   109,  1282,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    56,    57,    -1,    -1,    -1,  1001,    -1,
    -1,    -1,  1005,    66,    -1,    -1,    -1,    -1,    -1,    -1,
    73,    -1,  1015,    -1,    -1,    40,    41,    -1,    -1,  1236,
    -1,    -1,    -1,    -1,    -1,    50,    51,    52,    53,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    99,    -1,    63,    -1,
   103,    -1,    -1,    -1,   107,    -1,   109,  1050,    73,    -1,
  1053,  1054,    -1,    -1,    -1,    -1,  1059,    -1,    -1,    -1,
    85,    86,    -1,    -1,    -1,    -1,    -1,  1070,    -1,    -1,
    -1,    -1,    97,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,   893,    -1,   895,    -1,   897,    -1,    -1,
    -1,    -1,    -1,    -1,     3,    -1,     5,     6,     7,    -1,
    -1,    -1,    -1,    12,    13,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    21,    -1,    -1,    -1,    -1,    -1,  1121,  1122,
    -1,  1124,    -1,    -1,    33,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,   942,    -1,    -1,    -1,    -1,    -1,    -1,   164,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    56,    57,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    66,   183,   184,
    -1,    -1,    -1,    -1,    73,    -1,    -1,    -1,    -1,  1172,
    -1,   196,    -1,   198,    -1,    -1,   201,    -1,    -1,    -1,
    -1,    -1,  1185,  1186,  1187,   210,    -1,    -1,    -1,    -1,
    99,     1,    -1,     3,   103,     5,   221,   222,   107,    -1,
   109,    -1,    12,    13,    14,    -1,    -1,    -1,    18,    19,
    20,    21,    -1,    -1,    24,    -1,    -1,    27,    28,    29,
    30,    31,    -1,    33,    34,    35,    36,    37,    38,    39,
    40,    -1,  1235,    43,    44,    -1,    -1,    -1,    48,    -1,
  1050,    51,    -1,    -1,    -1,    -1,    -1,    57,  1058,    -1,
    -1,    61,    62,    63,    64,    65,    66,    -1,    -1,    -1,
  1070,    -1,  1072,    73,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    86,    -1,    -1,  1282,
    -1,    -1,   307,    -1,    -1,    -1,    -1,   312,   313,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,   322,    -1,    -1,
    -1,    -1,   327,    -1,    -1,   192,    -1,    -1,    -1,    -1,
     3,   121,     5,     6,     7,    -1,    -1,    -1,    -1,    12,
    13,   346,    -1,    -1,    -1,    -1,    -1,    -1,    21,    -1,
    -1,    -1,  1142,    -1,    -1,    -1,    -1,    -1,  1148,  1149,
    33,  1151,  1152,    -1,    -1,  1155,    -1,  1157,  1158,    -1,
  1160,  1161,    -1,    -1,    -1,   380,   381,   382,   383,    -1,
    -1,    -1,   387,    56,    57,    -1,    -1,    -1,    -1,   394,
    -1,  1181,    -1,    66,    -1,   400,    -1,    -1,    -1,    -1,
    73,    -1,    -1,   408,    -1,    -1,    -1,   412,    -1,    -1,
    -1,   416,   417,   418,    -1,    -1,   421,   422,    -1,   424,
   425,   426,    -1,    -1,    -1,    -1,    99,    -1,    -1,    -1,
   103,    -1,    -1,    -1,   107,    -1,   109,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,  1235,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   473,   336,
   475,   476,   477,   478,   479,    -1,    -1,    -1,   483,    -1,
   347,   348,    -1,    -1,    -1,   352,   353,   354,   355,    -1,
    -1,    -1,  1282,    -1,    -1,    -1,    -1,    -1,    -1,   366,
    -1,    -1,    -1,    -1,    -1,    -1,   511,   512,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,     1,    -1,     3,   392,     5,     6,     7,   534,
    -1,    -1,    -1,    12,    13,    14,    -1,   542,   543,    18,
    19,    20,    21,    -1,   411,    24,    -1,    -1,    27,    28,
    29,    30,    31,    -1,    33,    34,    35,    36,    37,    38,
    39,    40,    -1,    -1,    43,    44,    -1,    -1,    -1,    48,
    -1,    -1,    51,    -1,    -1,    -1,    -1,    56,    57,    -1,
    -1,    -1,    61,    62,    63,    64,    65,    66,    -1,    -1,
    -1,    -1,    -1,    -1,    73,    -1,    -1,    -1,   603,    -1,
    -1,    -1,    -1,    -1,    -1,   610,    -1,    86,   613,   614,
    -1,    -1,   617,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    99,    -1,    -1,    -1,   103,    -1,    -1,    -1,   107,    -1,
   109,   636,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,   646,   121,   648,   649,    -1,    -1,   652,   653,   654,
    -1,   656,   657,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,   689,    -1,    -1,    -1,    -1,    -1,
   695,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   713,    -1,
    -1,    -1,    -1,    -1,    -1,   582,   583,   584,   585,   586,
   587,   588,   589,   590,   591,   592,   593,   594,   595,   596,
   597,   598,   599,   600,   601,    -1,    -1,    -1,    -1,    -1,
    -1,   746,    -1,    -1,    -1,    -1,    -1,   752,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,   766,    -1,    -1,    -1,    -1,    -1,   772,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,     3,    -1,     5,     6,
     7,    -1,    -1,    -1,    -1,    12,    13,    14,    -1,    -1,
    -1,    18,    -1,    -1,    21,    -1,    -1,    -1,    -1,    -1,
    -1,    28,    29,    -1,    31,    -1,    33,    34,    -1,     1,
   825,   826,    -1,    -1,    -1,   692,    -1,    -1,    -1,    -1,
    -1,    48,    -1,   838,    51,   840,   841,   842,   843,    56,
    57,    -1,    -1,   848,    61,    62,    63,    64,    65,    66,
    -1,    -1,    -1,    -1,   859,    -1,    73,    -1,    40,    41,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    50,    51,
    52,    53,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    63,    99,    -1,    -1,   752,   103,    -1,    -1,    -1,
   107,    73,   109,    -1,    -1,    -1,    -1,   764,    -1,    -1,
    -1,   118,    -1,    85,    86,    -1,    -1,    -1,   913,     3,
    -1,     5,     6,     7,    -1,    97,    -1,   784,    12,    13,
    14,    -1,    -1,    -1,    18,    -1,    -1,    21,    -1,    -1,
    -1,    -1,    -1,    -1,    28,    29,   941,    31,   943,    33,
    34,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   954,
   955,   956,    -1,    -1,    48,    -1,    -1,    51,   963,   964,
    -1,    -1,    56,    57,    -1,    -1,    -1,    61,    62,    63,
    64,    65,    66,    -1,    -1,   980,    -1,   982,    -1,    73,
    -1,    -1,   164,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    86,    -1,    -1,    -1,  1001,    -1,    -1,    -1,
  1005,   183,   184,    -1,    -1,    99,   100,    -1,    -1,   103,
  1015,    -1,    -1,   107,   196,   109,   198,    -1,    -1,   201,
   114,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   210,    -1,
    -1,   898,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   221,
   222,    -1,    -1,    -1,    -1,  1050,    -1,  1052,  1053,  1054,
    -1,    -1,    -1,    -1,  1059,    -1,    -1,    -1,   925,   926,
    -1,  1066,    -1,  1068,    -1,  1070,    -1,    -1,    -1,    -1,
    -1,  1076,    -1,  1078,    -1,    -1,    -1,   944,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,  1103,    -1,
  1105,    -1,     3,  1108,     5,     6,     7,    -1,    -1,    -1,
    -1,    12,    13,    14,  1119,  1120,    -1,    -1,    19,    20,
    21,    -1,    -1,    24,   991,   307,    27,    -1,    -1,    30,
   312,   313,    33,    34,    35,    36,    37,    38,    39,    40,
   322,    -1,    43,    44,    -1,   327,    -1,    48,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    57,    -1,    -1,    -1,
  1027,    -1,    -1,    -1,   346,    66,    -1,  1172,    -1,    -1,
    -1,    -1,    73,    -1,    -1,    -1,    -1,  1182,  1183,    -1,
    -1,    -1,    -1,    -1,    -1,    86,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   380,   381,
   382,   383,   103,    -1,    -1,   387,   107,    -1,    -1,    -1,
    -1,    -1,   394,   114,    -1,    -1,    -1,    -1,   400,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,   408,    -1,    -1,    -1,
   412,  1236,    -1,    -1,   416,   417,   418,    -1,    -1,   421,
   422,    -1,   424,   425,   426,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,  1128,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
  1147,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,   473,    -1,   475,   476,   477,   478,   479,    -1,    -1,
    -1,   483,     3,    -1,     5,     6,     7,    -1,    -1,    -1,
    -1,    12,    13,    14,    -1,    -1,  1183,    -1,    -1,    -1,
    21,    -1,    -1,    -1,    -1,    40,    -1,     1,    -1,   511,
   512,    -1,    33,    34,    -1,    -1,    -1,    52,    53,    -1,
    55,    56,    57,    58,    59,    60,    61,    48,    63,    -1,
    -1,    -1,   534,    -1,    -1,    56,    57,    -1,    -1,    -1,
   542,   543,    -1,    -1,    -1,    66,    40,    41,    -1,    -1,
    -1,    -1,    73,    -1,    -1,    -1,    50,    51,    52,    53,
    -1,    -1,    97,    -1,    -1,    -1,    -1,    -1,    -1,    63,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    99,    73,
    -1,    -1,   103,    -1,    -1,    -1,   107,    -1,   109,    -1,
    -1,    85,    86,    -1,    -1,     3,    -1,     5,     6,     7,
    -1,   603,    -1,    97,    12,    13,    14,    -1,   610,    -1,
    -1,   613,   614,    21,    -1,   617,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    33,    34,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,   636,    -1,    -1,    -1,    -1,    -1,
    48,    -1,    -1,    -1,   646,    -1,   648,   649,    56,    57,
   652,   653,   654,    -1,   656,   657,    -1,    -1,    66,    -1,
    -1,    -1,    -1,    -1,    -1,    73,    -1,    -1,    -1,    -1,
   164,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,   689,    -1,   183,
   184,    99,    -1,   695,    -1,   103,    -1,    -1,    -1,   107,
    -1,   109,   196,    -1,   198,    -1,    -1,   201,    -1,    -1,
    -1,   713,    -1,    -1,    -1,    -1,   210,    -1,    -1,    -1,
    -1,     3,    -1,     5,     6,     7,    -1,   221,   222,    -1,
    12,    13,    14,    -1,    -1,    -1,    18,    19,    20,    21,
    -1,    -1,    24,    -1,   746,    27,    28,    29,    30,    31,
   752,    33,    34,    35,    36,    37,    38,    39,    40,    -1,
    -1,    43,    44,    -1,   766,    -1,    48,    -1,    -1,    51,
   772,    -1,    -1,    -1,    56,    57,    -1,    -1,    -1,    61,
    62,    63,    64,    65,    66,    -1,    -1,    -1,    -1,    -1,
    -1,    73,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    86,    -1,    -1,    -1,    -1,    -1,
    -1,   346,    -1,   307,    -1,    -1,    -1,    99,   312,   313,
    -1,   103,    -1,   825,   826,   107,    -1,   109,   322,    -1,
    -1,    -1,   114,   327,    -1,    -1,   838,    -1,   840,   841,
   842,   843,    -1,    -1,    -1,    -1,   848,   382,   383,   384,
   385,   386,   387,    -1,    -1,    -1,    -1,   859,    -1,    -1,
    -1,     1,    -1,    -1,    -1,   400,    -1,     3,    -1,     5,
     6,     7,    -1,    -1,    -1,    -1,    12,    13,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    21,    -1,    -1,    -1,   424,
   425,   426,    -1,    -1,    -1,    -1,     3,    33,     5,     6,
     7,    -1,    -1,    -1,    -1,    12,    13,    -1,    -1,    -1,
    -1,   913,    52,    53,    21,    -1,    -1,    -1,   412,    -1,
    56,    57,   416,   417,   418,    -1,    33,   421,   422,    -1,
    66,    -1,    -1,    73,    -1,    -1,    -1,    73,   473,   941,
    -1,   943,   477,   478,   479,    85,    86,    -1,   483,    56,
    57,    -1,   954,   955,   956,    -1,    -1,    -1,    -1,    66,
    -1,   963,   964,    99,    -1,    -1,    73,   103,    -1,    -1,
    -1,   107,    -1,   109,    -1,    -1,    -1,    -1,   980,   473,
   982,   475,   476,   477,   478,   479,    -1,    -1,    -1,   483,
    -1,    -1,    99,    -1,    -1,    -1,   103,    -1,    -1,  1001,
   107,    -1,   109,  1005,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,  1015,    -1,    -1,    -1,   511,   512,    -1,
    -1,    -1,    -1,    -1,     3,    -1,     5,     6,     7,    -1,
    -1,    -1,    -1,    12,    13,    14,    -1,    -1,    -1,    -1,
   534,    -1,    21,   183,   184,    -1,    -1,    -1,  1050,    -1,
  1052,  1053,  1054,    -1,    33,    34,   196,  1059,    -1,    -1,
    -1,    -1,    -1,    -1,  1066,    -1,  1068,    -1,  1070,    48,
    -1,    -1,    -1,    -1,  1076,    -1,  1078,    56,    57,    -1,
    -1,   221,   222,    -1,    -1,    -1,    -1,    66,    -1,    -1,
    -1,    -1,    -1,    -1,    73,    -1,    -1,    -1,    -1,    -1,
    -1,  1103,    -1,  1105,    -1,    -1,  1108,    -1,    -1,   603,
    -1,    -1,    -1,    -1,    -1,    -1,   610,  1119,  1120,    -1,
    99,    -1,    -1,   617,   103,    -1,    -1,    -1,   107,     3,
   109,     5,     6,     7,    -1,    -1,    -1,    -1,    12,    13,
    14,    -1,   636,    -1,    18,    -1,    -1,    21,    -1,    -1,
    -1,    -1,   646,    -1,    28,    29,    -1,    31,   652,    33,
    34,    -1,   656,   657,    -1,    -1,    -1,    -1,    -1,    -1,
  1172,    -1,   312,   313,    48,    -1,    -1,    51,   713,    -1,
  1182,  1183,    56,    57,    -1,    -1,    -1,    61,    62,    63,
    64,    65,    66,    -1,    -1,    -1,    -1,    -1,    -1,    73,
     3,   695,     5,     6,     7,    -1,   346,    -1,    -1,    12,
    13,    14,    86,    -1,    -1,    -1,    -1,    -1,    21,    -1,
    -1,    -1,    -1,    -1,    -1,    99,    -1,    -1,    -1,   103,
    33,    34,    -1,   107,  1236,   109,    -1,    -1,    -1,    -1,
   380,   381,   382,   383,    -1,    48,    -1,   387,    -1,    -1,
    -1,    -1,   746,    56,    57,    -1,    -1,    -1,    -1,    -1,
   400,    -1,    -1,    66,    -1,    -1,    -1,    -1,    -1,    -1,
    73,    -1,    -1,    -1,    -1,    -1,   416,   417,    -1,    -1,
    -1,   421,   422,    -1,   424,   425,   426,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    99,    -1,    -1,    -1,
   103,    -1,    -1,   838,   107,    -1,   109,   842,   843,   844,
   845,   846,   847,   848,    -1,    -1,     1,    -1,     3,    -1,
     5,     6,     7,    -1,   859,    -1,    -1,    12,    13,    14,
    -1,   825,   826,    18,    19,    20,    21,   477,   478,    24,
    -1,    -1,    27,    28,    29,    30,    31,    -1,    33,    34,
    35,    36,    37,    38,    39,    40,    -1,    -1,    43,    44,
    -1,    -1,    -1,    48,    -1,    -1,    51,   902,   903,    -1,
    -1,    56,    57,    -1,    -1,    -1,    61,    62,    63,    64,
    65,    66,     3,    -1,     5,     6,     7,    -1,    73,    -1,
    -1,    12,    13,    14,    -1,    -1,    -1,    18,    -1,    -1,
    21,    86,   542,   543,    -1,    -1,    -1,    28,    29,    -1,
    31,    -1,    33,    34,    99,    -1,    -1,    -1,   103,    -1,
    -1,    -1,   107,    -1,   109,    -1,    -1,    48,    -1,    -1,
    51,    -1,    -1,    -1,    -1,    56,    57,   122,    -1,    -1,
    61,    62,    63,    64,    65,    66,    -1,    -1,    -1,   943,
    -1,    -1,    73,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,   956,    -1,    -1,    86,  1001,    -1,    -1,    -1,
  1005,    -1,    -1,   613,   614,    -1,    -1,    -1,    99,   100,
  1015,    -1,   103,    -1,    -1,    -1,   107,    -1,   109,    -1,
    -1,    -1,    -1,   114,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,  1001,   648,   649,
    -1,  1005,    -1,   653,   654,  1050,    -1,    -1,  1053,  1054,
    -1,  1015,    -1,    -1,  1059,    -1,    -1,     3,    -1,     5,
     6,     7,    -1,    -1,    -1,  1070,    12,    13,    -1,    -1,
    16,    17,    -1,    19,    20,    21,    -1,    -1,    24,   689,
    -1,    -1,    -1,    -1,    30,   695,    -1,    33,    -1,    35,
    36,    37,    38,    39,    40,    -1,    -1,    43,    44,    -1,
    -1,    47,  1066,   713,  1068,    -1,    -1,    53,    54,    55,
    56,    57,  1076,    -1,  1078,    -1,    -1,    -1,    -1,    -1,
    66,    67,    68,    69,    70,    71,    72,    73,    -1,    75,
    76,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,  1103,
    -1,  1105,   752,    -1,  1108,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    99,    -1,   101,   102,   103,    -1,    -1,
    -1,   107,    -1,   109,   110,    -1,    -1,  1172,    -1,    -1,
    -1,    -1,    -1,    14,    -1,   121,   122,    18,    19,    20,
  1185,  1186,    -1,    24,    -1,    -1,    -1,    28,    29,    30,
    31,    -1,    -1,    34,    35,    36,    37,    38,    39,    40,
    -1,    -1,    43,    44,    -1,    -1,    -1,    48,  1172,    -1,
    51,    -1,    -1,    -1,    -1,    -1,    -1,    -1,  1182,    -1,
    61,    62,    63,    64,    65,    -1,    -1,    -1,   838,    -1,
   840,   841,   842,   843,    -1,    -1,    -1,    -1,   848,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   859,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,
     3,    -1,     5,     6,     7,    -1,    -1,    -1,    -1,    12,
    13,    14,  1236,    16,    17,    18,    19,    20,    21,    22,
    -1,    24,    25,    26,    27,    28,    29,    30,    31,    32,
    33,    34,    35,    36,    37,    38,    39,    40,    41,    42,
    43,    44,    45,    46,    47,    48,    49,    50,    51,    52,
    53,    54,    55,    56,    57,    -1,    -1,    -1,    61,    62,
    63,    64,    65,    66,    67,    68,    69,    70,    71,    72,
    73,   941,    75,    76,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    86,   954,   955,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,   963,   964,    -1,    99,    -1,   101,   102,
   103,    -1,    -1,    -1,   107,    -1,   109,   110,    -1,    -1,
   980,     3,   982,     5,    -1,    -1,    -1,   120,   121,   122,
    12,    13,    14,    -1,    -1,    -1,    18,    19,    20,    21,
    -1,    -1,    24,    -1,    -1,    27,    28,    29,    30,    31,
    -1,    33,    34,    35,    36,    37,    38,    39,    40,    -1,
    -1,    43,    44,    -1,    -1,    -1,    48,    -1,    -1,    51,
    -1,    -1,    -1,    -1,    -1,    57,    -1,    -1,    -1,    61,
    62,    63,    64,    65,    66,    -1,    -1,    -1,    -1,    -1,
  1050,    73,  1052,  1053,  1054,    -1,     1,    -1,     3,  1059,
     5,     6,     7,    85,    -1,    -1,    -1,    12,    13,    14,
  1070,    16,    17,    18,    19,    20,    21,    22,   100,    24,
    25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
    35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
    45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
    55,    56,    57,    -1,    -1,    -1,    61,    62,    63,    64,
    65,    66,    67,    68,    69,    70,    71,    72,    73,    -1,
    75,    76,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    86,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    99,    -1,   101,   102,   103,    -1,
    -1,    -1,   107,    -1,   109,   110,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,   120,   121,    -1,    -1,    -1,
     1,    -1,     3,  1183,     5,     6,     7,    -1,    -1,    -1,
    -1,    12,    13,    14,    -1,    16,    17,    18,    19,    20,
    21,    -1,    -1,    24,    -1,    -1,    27,    28,    29,    30,
    31,    -1,    33,    34,    35,    36,    37,    38,    39,    40,
    -1,    -1,    43,    44,    -1,    -1,    47,    48,    -1,    -1,
    51,    -1,    53,    54,    55,    56,    57,    -1,    -1,    -1,
    61,    62,    63,    64,    65,    66,    67,    68,    69,    70,
    71,    72,    73,    -1,    75,    76,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    86,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    99,    -1,
   101,   102,   103,    -1,    -1,    -1,   107,    -1,   109,   110,
    -1,     3,    -1,     5,     6,     7,     8,    -1,    -1,   120,
    12,    13,    14,    -1,    -1,    -1,    18,    19,    20,    21,
    -1,    -1,    24,    -1,    -1,    27,    28,    29,    30,    31,
    -1,    33,    34,    35,    36,    37,    38,    39,    40,    -1,
    -1,    43,    44,    -1,    -1,    -1,    48,    -1,    -1,    51,
    -1,    -1,    -1,    -1,    56,    57,    58,    59,    60,    61,
    62,    63,    64,    65,    66,    -1,    -1,    -1,    -1,    -1,
    -1,    73,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,     3,    -1,     5,     6,     7,    -1,    -1,    -1,
    -1,    12,    13,    -1,    -1,    16,    17,    99,    19,    20,
    21,   103,    -1,    24,    -1,   107,    -1,   109,    -1,    30,
    -1,    -1,    33,    -1,    35,    36,    37,    38,    39,    40,
   122,    -1,    43,    44,    -1,    -1,    47,    -1,    -1,    -1,
    -1,    -1,    53,    54,    55,    56,    57,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    66,    67,    68,    69,    70,
    71,    72,    73,    -1,    75,    76,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    99,    -1,
   101,   102,   103,    -1,    -1,    -1,   107,    -1,   109,   110,
     1,    -1,     3,    -1,     5,     6,     7,    -1,    -1,    -1,
   121,    12,    13,    14,    -1,    -1,    -1,    18,    19,    20,
    21,    -1,    -1,    24,    -1,    -1,    -1,    28,    29,    30,
    31,    -1,    33,    34,    35,    36,    37,    38,    39,    40,
    -1,    -1,    43,    44,    -1,    -1,    -1,    48,    -1,    -1,
    51,    -1,    -1,    -1,    -1,    56,    57,    -1,    -1,    -1,
    61,    62,    63,    64,    65,    66,    -1,    -1,    -1,    -1,
    -1,    -1,    73,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,     3,    -1,     5,    -1,    99,    -1,
    -1,    -1,   103,    12,    13,    14,   107,    -1,   109,    -1,
    19,    20,    21,    -1,    -1,    24,    -1,   118,    27,    -1,
    -1,    30,    -1,    -1,    33,    34,    35,    36,    37,    38,
    39,    40,    -1,    -1,    43,    44,    -1,    -1,    -1,    48,
    -1,    -1,    -1,    -1,    53,    54,    -1,    -1,    57,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    66,    -1,    -1,
    -1,    -1,    -1,    -1,    73,    74,    75,    76,    77,    78,
    79,    80,    81,    82,    83,    84,    -1,    -1,    87,    88,
    89,    90,    91,    92,    93,    94,    95,    96,    97,    98,
    99,    -1,   101,   102,   103,   104,   105,   106,   107,   108,
   109,   110,   111,   112,   113,   114,    -1,   116,    -1,     3,
   119,     5,     6,     7,    -1,    -1,    -1,    -1,    12,    13,
    14,    -1,    -1,    -1,    18,    19,    20,    21,    -1,    -1,
    24,    -1,    -1,    27,    28,    29,    30,    31,    -1,    33,
    34,    35,    36,    37,    38,    39,    40,    -1,    -1,    43,
    44,    -1,    -1,    -1,    48,    -1,    -1,    51,    -1,    -1,
    -1,    -1,    56,    57,    -1,    -1,    -1,    61,    62,    63,
    64,    65,    66,    -1,    -1,    -1,    -1,    -1,    -1,    73,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    99,    -1,    -1,    -1,   103,
    -1,    -1,    -1,   107,    -1,   109,     3,    -1,     5,     6,
     7,    -1,    -1,    -1,   118,    12,    13,    14,    -1,    -1,
    -1,    18,    19,    20,    21,    -1,    -1,    24,    -1,    -1,
    27,    28,    29,    30,    31,    -1,    33,    34,    35,    36,
    37,    38,    39,    40,    -1,    -1,    43,    44,    -1,    -1,
    -1,    48,    -1,    -1,    51,    -1,    -1,    -1,    -1,    56,
    57,    -1,    -1,    -1,    61,    62,    63,    64,    65,    66,
    -1,    -1,    -1,    -1,    -1,    -1,    73,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    99,    -1,    -1,    -1,   103,    -1,    -1,    -1,
   107,    -1,   109,     3,    -1,     5,     6,     7,    -1,    -1,
    -1,   118,    12,    13,    -1,    -1,    16,    17,    -1,    19,
    20,    21,    -1,    -1,    24,    -1,    -1,    -1,    -1,    -1,
    30,    -1,    -1,    33,    -1,    35,    36,    37,    38,    39,
    40,    -1,    -1,    43,    44,    -1,    -1,    47,    -1,    -1,
    -1,    -1,    -1,    53,    54,    55,    56,    57,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    66,    67,    68,    69,
    70,    71,    72,    73,    -1,    75,    76,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    99,
    -1,   101,   102,   103,    -1,    -1,    -1,   107,    -1,   109,
   110,    -1,    -1,    -1,     3,   115,     5,     6,     7,    -1,
    -1,    -1,    -1,    12,    13,    -1,    -1,    16,    17,    -1,
    19,    20,    21,    -1,    -1,    24,    -1,    -1,    -1,    -1,
    -1,    30,    -1,    -1,    33,    -1,    35,    36,    37,    38,
    39,    40,    -1,    -1,    43,    44,    -1,    -1,    47,    -1,
    -1,    -1,    -1,    -1,    53,    54,    55,    56,    57,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    66,    67,    68,
    69,    70,    71,    72,    73,    -1,    75,    76,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    99,    -1,   101,   102,   103,    -1,    -1,    -1,   107,    -1,
   109,   110,    -1,    -1,    -1,     3,   115,     5,     6,     7,
    -1,    -1,    -1,    -1,    12,    13,    -1,    -1,    16,    17,
    -1,    19,    20,    21,    -1,    -1,    24,    -1,    -1,    -1,
    -1,    -1,    30,    -1,    -1,    33,    -1,    35,    36,    37,
    38,    39,    40,    -1,    -1,    43,    44,    -1,    -1,    47,
    -1,    -1,    -1,    -1,    -1,    53,    54,    55,    56,    57,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    66,    67,
    68,    69,    70,    71,    72,    73,    -1,    75,    76,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    99,    -1,   101,   102,   103,    -1,    -1,    -1,   107,
    -1,   109,   110,    -1,    -1,    -1,     3,   115,     5,     6,
     7,    -1,    -1,    -1,    -1,    12,    13,    14,    -1,    16,
    17,    18,    19,    20,    21,    -1,    -1,    24,    -1,    -1,
    27,    28,    29,    30,    31,    -1,    33,    34,    35,    36,
    37,    38,    39,    40,    -1,    -1,    43,    44,    -1,    -1,
    47,    48,    -1,    -1,    51,    -1,    53,    54,    55,    56,
    57,    -1,    -1,    -1,    61,    62,    63,    64,    65,    66,
    67,    68,    69,    70,    71,    72,    73,    -1,    75,    76,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    85,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    99,   100,   101,   102,   103,    -1,    -1,    -1,
   107,    -1,   109,   110,    -1,    -1,     3,   114,     5,     6,
     7,    -1,    -1,    -1,    -1,    12,    13,    14,    -1,    16,
    17,    18,    19,    20,    21,    -1,    -1,    24,    -1,    -1,
    27,    28,    29,    30,    31,    -1,    33,    34,    35,    36,
    37,    38,    39,    40,    -1,    -1,    43,    44,    -1,    -1,
    47,    48,    -1,    -1,    51,    -1,    53,    54,    55,    56,
    57,    -1,    -1,    -1,    61,    62,    63,    64,    65,    66,
    67,    68,    69,    70,    71,    72,    73,    -1,    75,    76,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    85,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    99,   100,   101,   102,   103,    -1,    -1,    -1,
   107,    -1,   109,   110,    -1,    -1,     3,   114,     5,     6,
     7,    -1,    -1,    -1,    -1,    12,    13,    14,    -1,    16,
    17,    18,    19,    20,    21,    -1,    -1,    24,    -1,    -1,
    27,    28,    29,    30,    31,    -1,    33,    34,    35,    36,
    37,    38,    39,    40,    -1,    -1,    43,    44,    -1,    -1,
    47,    48,    -1,    -1,    51,    -1,    53,    54,    55,    56,
    57,    -1,    -1,    -1,    61,    62,    63,    64,    65,    66,
    67,    68,    69,    70,    71,    72,    73,    -1,    75,    76,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    85,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    99,   100,   101,   102,   103,    -1,    -1,    -1,
   107,    -1,   109,   110,    -1,    -1,     3,   114,     5,     6,
     7,    -1,    -1,    -1,    -1,    12,    13,    14,    -1,    16,
    17,    18,    19,    20,    21,    -1,    -1,    24,    -1,    -1,
    27,    28,    29,    30,    31,    -1,    33,    34,    35,    36,
    37,    38,    39,    40,    -1,    -1,    43,    44,    -1,    -1,
    47,    48,    -1,    -1,    51,    -1,    53,    54,    55,    56,
    57,    -1,    -1,    -1,    61,    62,    63,    64,    65,    66,
    67,    68,    69,    70,    71,    72,    73,    -1,    75,    76,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    85,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    99,   100,   101,   102,   103,    -1,    -1,    -1,
   107,    -1,   109,   110,    -1,    -1,     3,   114,     5,     6,
     7,    -1,    -1,    -1,    -1,    12,    13,    14,    -1,    16,
    17,    18,    19,    20,    21,    -1,    -1,    24,    -1,    -1,
    27,    28,    29,    30,    31,    -1,    33,    34,    35,    36,
    37,    38,    39,    40,    -1,    -1,    43,    44,    -1,    -1,
    47,    48,    -1,    -1,    51,    -1,    53,    54,    55,    56,
    57,    -1,    -1,    -1,    61,    62,    63,    64,    65,    66,
    67,    68,    69,    70,    71,    72,    73,    -1,    75,    76,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    85,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    99,   100,   101,   102,   103,    -1,    -1,    -1,
   107,    -1,   109,   110,    -1,    -1,     3,   114,     5,     6,
     7,    -1,    -1,    -1,    -1,    12,    13,    14,    -1,    16,
    17,    18,    19,    20,    21,    -1,    -1,    24,    -1,    -1,
    27,    28,    29,    30,    31,    -1,    33,    34,    35,    36,
    37,    38,    39,    40,    -1,    -1,    43,    44,    -1,    -1,
    47,    48,    -1,    -1,    51,    -1,    53,    54,    55,    56,
    57,    -1,    -1,    -1,    61,    62,    63,    64,    65,    66,
    67,    68,    69,    70,    71,    72,    73,    -1,    75,    76,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    85,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    99,   100,   101,   102,   103,    -1,    -1,    -1,
   107,    -1,   109,   110,    -1,    -1,     3,   114,     5,     6,
     7,    -1,    -1,    -1,    -1,    12,    13,    14,    -1,    16,
    17,    -1,    19,    20,    21,    -1,    -1,    24,    -1,    -1,
    -1,    -1,    -1,    30,    -1,    -1,    33,    34,    35,    36,
    37,    38,    39,    40,    -1,    -1,    43,    44,    -1,    -1,
    47,    48,    -1,    -1,    -1,    -1,    53,    -1,    55,    56,
    57,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    66,
    67,    68,    69,    70,    71,    72,    73,    -1,    75,    76,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    86,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    99,    -1,   101,   102,   103,    -1,    -1,    -1,
   107,    -1,   109,   110,    -1,    -1,     3,   114,     5,     6,
     7,    -1,    -1,    -1,    -1,    12,    13,    14,    -1,    16,
    17,    -1,    19,    20,    21,    -1,    -1,    24,    -1,    -1,
    -1,    -1,    -1,    30,    -1,    -1,    33,    34,    35,    36,
    37,    38,    39,    40,    -1,    -1,    43,    44,    -1,    -1,
    47,    48,    -1,    -1,    -1,    -1,    53,    -1,    55,    56,
    57,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    66,
    67,    68,    69,    70,    71,    72,    73,    -1,    75,    76,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    86,
    -1,     3,    -1,     5,     6,     7,    -1,    -1,    -1,    -1,
    12,    13,    99,    -1,   101,   102,   103,    -1,    -1,    21,
   107,    -1,   109,   110,    -1,    -1,     3,   114,     5,     6,
     7,    33,    -1,    -1,    -1,    12,    13,    -1,    -1,    16,
    17,    -1,    19,    20,    21,    -1,    -1,    24,    -1,    -1,
    -1,    -1,    -1,    30,    56,    57,    33,    -1,    35,    36,
    37,    38,    39,    40,    66,    -1,    43,    44,    -1,    -1,
    47,    73,    -1,    -1,    -1,    -1,    53,    54,    55,    56,
    57,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    66,
    67,    68,    69,    70,    71,    72,    73,    99,    75,    76,
    -1,   103,    -1,    -1,    -1,   107,    -1,   109,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    99,    -1,   101,   102,   103,    -1,    -1,    -1,
   107,    -1,   109,   110,    -1,    -1,     3,   114,     5,     6,
     7,    -1,    -1,    -1,    -1,    12,    13,    14,    -1,    -1,
    -1,    18,    19,    20,    21,    -1,    -1,    24,    -1,    -1,
    27,    28,    29,    30,    31,    -1,    33,    34,    35,    36,
    37,    38,    39,    40,    -1,    -1,    43,    44,    -1,    -1,
    -1,    48,    -1,    -1,    51,    -1,    -1,    -1,    -1,    56,
    57,    -1,    -1,    -1,    61,    62,    63,    64,    65,    66,
    -1,    -1,    -1,    -1,    -1,    -1,    73,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    85,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    99,   100,    -1,     3,   103,     5,     6,     7,
   107,    -1,   109,    -1,    12,    13,    14,   114,    -1,    -1,
    18,    19,    20,    21,    -1,    -1,    24,    -1,    -1,    27,
    28,    29,    30,    31,    -1,    33,    34,    35,    36,    37,
    38,    39,    40,    -1,    -1,    43,    44,    -1,    -1,    -1,
    48,    -1,    -1,    51,    -1,    -1,    -1,    -1,    56,    57,
    -1,    -1,    -1,    61,    62,    63,    64,    65,    66,    -1,
    -1,    -1,    -1,    -1,    -1,    73,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    85,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    99,   100,    -1,     3,   103,     5,     6,     7,   107,
    -1,   109,    -1,    12,    13,    14,   114,    -1,    -1,    18,
    19,    20,    21,    -1,    -1,    24,    -1,    -1,    27,    28,
    29,    30,    31,    -1,    33,    34,    35,    36,    37,    38,
    39,    40,    -1,    -1,    43,    44,    -1,    -1,    -1,    48,
    -1,    -1,    51,    -1,    -1,    -1,    -1,    56,    57,    -1,
    -1,    -1,    61,    62,    63,    64,    65,    66,    -1,    -1,
    -1,    -1,    -1,    -1,    73,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    86,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    99,   100,    -1,     3,   103,     5,     6,     7,   107,    -1,
   109,    -1,    12,    13,    14,   114,    -1,    -1,    18,    19,
    20,    21,    -1,    -1,    24,    -1,    -1,    27,    28,    29,
    30,    31,    -1,    33,    34,    35,    36,    37,    38,    39,
    40,    -1,    -1,    43,    44,    -1,    -1,    -1,    48,    -1,
    -1,    51,    -1,    -1,    -1,    -1,    56,    57,    -1,    -1,
    -1,    61,    62,    63,    64,    65,    66,    -1,    -1,    -1,
    -1,    -1,    -1,    73,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    86,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    99,
    -1,    -1,     3,   103,     5,     6,     7,   107,    -1,   109,
    -1,    12,    13,    14,   114,    -1,    -1,    18,    19,    20,
    21,    -1,    -1,    24,    -1,    -1,    27,    28,    29,    30,
    31,    -1,    33,    34,    35,    36,    37,    38,    39,    40,
    -1,    -1,    43,    44,    -1,    -1,    -1,    48,    -1,    -1,
    51,    -1,    -1,    -1,    -1,    -1,    57,    -1,    -1,    -1,
    61,    62,    63,    64,    65,    66,    -1,    -1,    -1,    -1,
    -1,    -1,    73,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    85,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    99,   100,
    -1,     3,   103,     5,     6,     7,   107,    -1,    -1,    -1,
    12,    13,    14,   114,    -1,    -1,    18,    19,    20,    21,
    -1,    -1,    24,    -1,    -1,    -1,    28,    29,    30,    31,
    -1,    33,    34,    35,    36,    37,    38,    39,    40,    -1,
    -1,    43,    44,    -1,    -1,    -1,    48,    -1,    -1,    51,
    -1,    -1,    -1,    -1,    56,    57,    -1,    -1,    -1,    61,
    62,    63,    64,    65,    66,    -1,    -1,    -1,    -1,    -1,
    -1,    73,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    99,   100,    -1,
     3,   103,     5,     6,     7,   107,    -1,   109,    -1,    12,
    13,    14,   114,    -1,    -1,    18,    19,    20,    21,    -1,
    -1,    24,    -1,    -1,    -1,    28,    29,    30,    31,    -1,
    33,    34,    35,    36,    37,    38,    39,    40,    -1,    -1,
    43,    44,    -1,    -1,    -1,    48,    -1,    -1,    51,    -1,
    -1,    -1,    -1,    56,    57,    -1,    -1,    -1,    61,    62,
    63,    64,    65,    66,    -1,    -1,    -1,    -1,    -1,    -1,
    73,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    99,    -1,    -1,     3,
   103,     5,     6,     7,   107,    -1,   109,    -1,    12,    13,
    14,   114,    -1,    -1,    18,    19,    20,    21,    -1,    -1,
    24,    -1,    -1,    -1,    28,    29,    30,    31,    -1,    33,
    34,    35,    36,    37,    38,    39,    40,    -1,    -1,    43,
    44,    -1,    -1,    -1,    48,    -1,    -1,    51,    -1,    -1,
    -1,    -1,    56,    57,    -1,    -1,    -1,    61,    62,    63,
    64,    65,    66,    -1,    -1,    -1,    -1,    -1,    -1,    73,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    99,    -1,    -1,     3,   103,
     5,     6,     7,   107,    -1,   109,    -1,    12,    13,    14,
   114,    -1,    -1,    -1,    19,    20,    21,    -1,    -1,    24,
    -1,    -1,    27,    -1,    -1,    30,    -1,    -1,    33,    34,
    35,    36,    37,    38,    39,    40,    -1,    -1,    43,    44,
    -1,    -1,    -1,    48,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    57,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     3,    66,     5,     6,     7,    -1,    -1,    -1,    73,    12,
    13,    14,    -1,    -1,    -1,    18,    -1,    -1,    21,    -1,
    -1,    86,    -1,    -1,    -1,    28,    29,    -1,    31,    -1,
    33,    34,    -1,    -1,    99,    -1,    -1,    -1,   103,    -1,
    -1,    -1,   107,    -1,    -1,    48,    -1,    -1,    51,   114,
    -1,    -1,    -1,    56,    57,    -1,    -1,    -1,    61,    62,
    63,    64,    65,    66,     3,    -1,     5,     6,     7,    -1,
    73,    -1,    -1,    12,    13,    14,    -1,    -1,    -1,    18,
    -1,    -1,    21,    86,    -1,    -1,    -1,    -1,    -1,    28,
    29,    -1,    31,    -1,    33,    34,    99,    -1,    -1,    -1,
   103,    -1,    -1,    -1,   107,    -1,   109,    -1,    -1,    48,
    -1,   114,    51,    -1,    -1,    -1,    -1,    56,    57,    -1,
    -1,    -1,    61,    62,    63,    64,    65,    66,     3,    -1,
     5,     6,     7,    -1,    73,    -1,    -1,    12,    13,    14,
    -1,    -1,    -1,    18,    -1,    -1,    21,    86,    -1,    -1,
    -1,    -1,    -1,    28,    29,    -1,    31,    -1,    33,    34,
    99,    -1,    -1,    -1,   103,    -1,    -1,    -1,   107,    -1,
   109,    -1,    -1,    48,    -1,   114,    51,    -1,    -1,    -1,
    -1,    56,    57,    -1,    -1,    -1,    61,    62,    63,    64,
    65,    66,     3,    -1,     5,     6,     7,    -1,    73,    -1,
    -1,    12,    13,    14,    -1,    -1,    -1,    18,    -1,    -1,
    21,    -1,    -1,    -1,    -1,    -1,    -1,    28,    29,    -1,
    31,    -1,    33,    34,    99,   100,    -1,    -1,   103,    -1,
    -1,    -1,   107,    -1,   109,    -1,    -1,    48,    -1,   114,
    51,    -1,    -1,    -1,    -1,    56,    57,    -1,    -1,    -1,
    61,    62,    63,    64,    65,    66,     3,    -1,     5,     6,
     7,    -1,    73,    -1,    -1,    12,    13,    14,    -1,    -1,
    -1,    18,    -1,    -1,    21,    -1,    -1,    -1,    -1,    -1,
    -1,    28,    29,    -1,    31,    -1,    33,    34,    99,    -1,
    -1,    -1,   103,    -1,    -1,    -1,   107,    -1,   109,    -1,
    -1,    48,    -1,   114,    51,    -1,    -1,    -1,    -1,    56,
    57,    -1,    -1,    -1,    61,    62,    63,    64,    65,    66,
    -1,    -1,    -1,    -1,    -1,    -1,    73,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    99,    -1,    -1,     3,   103,     5,     6,     7,
   107,    -1,   109,    -1,    12,    13,    14,   114,    -1,    -1,
    -1,    19,    20,    21,    -1,    -1,    24,    -1,    -1,    -1,
    -1,    -1,    30,    -1,    -1,    33,    34,    35,    36,    37,
    38,    39,    40,    -1,    -1,    43,    44,    -1,    -1,    -1,
    48,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    57,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    66,    -1,
    -1,    -1,    -1,    -1,    -1,    73,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     1,    -1,
     3,    -1,     5,     6,     7,    -1,    -1,    -1,    -1,    12,
    13,    99,    -1,    16,    17,   103,    19,    20,    21,   107,
    -1,    24,    -1,    -1,    -1,    -1,   114,    30,    -1,    -1,
    33,    -1,    35,    36,    37,    38,    39,    40,    -1,    -1,
    43,    44,    -1,    -1,    47,    -1,    -1,    -1,    -1,    -1,
    53,    54,    55,    56,    57,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    66,    67,    68,    69,    70,    71,    72,
    73,    -1,    75,    76,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    86,    -1,     3,    -1,     5,     6,     7,
    -1,    -1,    -1,    -1,    12,    13,    99,   100,   101,   102,
   103,    -1,    -1,    21,   107,    -1,   109,   110,     1,    -1,
     3,    -1,     5,     6,     7,    33,    -1,    -1,    -1,    12,
    13,    -1,    -1,    16,    17,    -1,    19,    20,    21,    -1,
    -1,    24,    -1,    -1,    -1,    -1,    -1,    30,    56,    57,
    33,    -1,    35,    36,    37,    38,    39,    40,    66,    -1,
    43,    44,    -1,    -1,    47,    73,    -1,    -1,    -1,    -1,
    53,    54,    55,    56,    57,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    66,    67,    68,    69,    70,    71,    72,
    73,    99,    75,    76,    -1,   103,    -1,    -1,    -1,   107,
    -1,   109,    -1,    86,    -1,     3,    -1,     5,     6,     7,
    -1,    -1,    -1,    -1,    12,    13,    99,    -1,   101,   102,
   103,    -1,    -1,    21,   107,    -1,   109,   110,     1,    -1,
     3,    -1,     5,     6,     7,    33,    -1,    -1,    -1,    12,
    13,    -1,    -1,    16,    17,    -1,    19,    20,    21,    -1,
    -1,    24,    -1,    -1,    -1,    -1,    -1,    30,    56,    57,
    33,    -1,    35,    36,    37,    38,    39,    40,    66,    -1,
    43,    44,    -1,    -1,    47,    73,    -1,    -1,    -1,    -1,
    53,    54,    55,    56,    57,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    66,    67,    68,    69,    70,    71,    72,
    73,    99,    75,    76,    -1,   103,    -1,    -1,    -1,   107,
    -1,   109,    -1,    86,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    99,    -1,   101,   102,
   103,    -1,    -1,    -1,   107,    -1,   109,   110,     0,     1,
    -1,     3,    -1,     5,     6,     7,    -1,    -1,    -1,    -1,
    12,    13,    14,    -1,    -1,    -1,    18,    19,    20,    21,
    -1,    -1,    24,    -1,    -1,    27,    28,    29,    30,    31,
    -1,    33,    34,    35,    36,    37,    38,    39,    40,    -1,
    -1,    43,    44,    -1,    -1,    -1,    48,    -1,    -1,    51,
    -1,    -1,    -1,    -1,    56,    57,    -1,    -1,    -1,    61,
    62,    63,    64,    65,    66,    -1,    -1,    -1,    -1,    -1,
    -1,    73,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    86,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    99,     1,    -1,
     3,   103,     5,     6,     7,   107,    -1,   109,    -1,    12,
    13,    14,    -1,    -1,    -1,    18,    19,    20,    21,    -1,
    -1,    24,    -1,    -1,    -1,    28,    29,    30,    31,    -1,
    33,    34,    35,    36,    37,    38,    39,    40,    -1,    -1,
    43,    44,    -1,    -1,    -1,    48,    -1,    -1,    51,    -1,
    -1,    -1,    -1,    56,    57,    -1,    -1,    -1,    61,    62,
    63,    64,    65,    66,    -1,    -1,    -1,    -1,    -1,    -1,
    73,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    99,    -1,    -1,     3,
   103,     5,     6,     7,   107,    -1,   109,    -1,    12,    13,
    14,    -1,    16,    17,    18,    19,    20,    21,    -1,    -1,
    24,    -1,    -1,    27,    28,    29,    30,    31,    -1,    33,
    34,    35,    36,    37,    38,    39,    40,    -1,    -1,    43,
    44,    -1,    -1,    47,    48,    -1,    -1,    51,    -1,    53,
    54,    55,    56,    57,    -1,    -1,    -1,    61,    62,    63,
    64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
    -1,    75,    76,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    85,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    99,   100,   101,   102,   103,
    -1,    -1,    -1,   107,    -1,   109,   110,     3,    -1,     5,
     6,     7,    -1,    -1,    -1,    -1,    12,    13,    14,    -1,
    16,    17,    18,    19,    20,    21,    -1,    -1,    24,    -1,
    -1,    27,    28,    29,    30,    31,    -1,    33,    34,    35,
    36,    37,    38,    39,    40,    -1,    -1,    43,    44,    -1,
    -1,    47,    48,    -1,    -1,    51,    -1,    53,    54,    55,
    56,    57,    -1,    -1,    -1,    61,    62,    63,    64,    65,
    66,    67,    68,    69,    70,    71,    72,    73,    -1,    75,
    76,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    85,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    99,   100,   101,   102,   103,    -1,    -1,
    -1,   107,    -1,   109,   110,     3,    -1,     5,     6,     7,
    -1,    -1,    -1,    -1,    12,    13,    14,    -1,    16,    17,
    -1,    19,    20,    21,    -1,    -1,    24,    -1,    -1,    27,
    -1,    -1,    30,    -1,    -1,    33,    34,    35,    36,    37,
    38,    39,    40,    -1,    -1,    43,    44,    -1,    -1,    47,
    48,    -1,    -1,    -1,    -1,    53,    54,    55,    56,    57,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    66,    67,
    68,    69,    70,    71,    72,    73,    -1,    75,    76,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    99,    -1,   101,   102,   103,    -1,    -1,    -1,   107,
    -1,   109,   110,     3,    -1,     5,     6,     7,    -1,    -1,
    -1,    -1,    12,    13,    14,    -1,    16,    17,    -1,    19,
    20,    21,    -1,    -1,    24,    -1,    -1,    27,    -1,    -1,
    30,    -1,    -1,    33,    34,    35,    36,    37,    38,    39,
    40,    -1,    -1,    43,    44,    -1,    -1,    47,    48,    -1,
    -1,    -1,    -1,    53,    54,    55,    56,    57,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    66,    67,    68,    69,
    70,    71,    72,    73,    -1,    75,    76,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    99,
    -1,   101,   102,   103,    -1,    -1,    -1,   107,    -1,   109,
   110,     3,    -1,     5,     6,     7,    -1,    -1,    -1,    -1,
    12,    13,    14,    -1,    16,    17,    -1,    19,    20,    21,
    -1,    -1,    24,    -1,    -1,    27,    -1,    -1,    30,    -1,
    -1,    33,    34,    35,    36,    37,    38,    39,    40,    -1,
    -1,    43,    44,    -1,    -1,    47,    48,    -1,    -1,    -1,
    -1,    53,    54,    55,    56,    57,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    66,    67,    68,    69,    70,    71,
    72,    73,    -1,    75,    76,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    99,    -1,   101,
   102,   103,    -1,    -1,    -1,   107,    -1,   109,   110,     3,
    -1,     5,     6,     7,    -1,    -1,    -1,    -1,    12,    13,
    14,    -1,    16,    17,    -1,    19,    20,    21,    -1,    -1,
    24,    -1,    -1,    -1,    -1,    -1,    30,    -1,    -1,    33,
    34,    35,    36,    37,    38,    39,    40,    -1,    -1,    43,
    44,    -1,    -1,    47,    48,    -1,    -1,    -1,    -1,    53,
    -1,    55,    56,    57,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    66,    67,    68,    69,    70,    71,    72,    73,
    -1,    75,    76,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    86,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    99,    -1,   101,   102,   103,
    -1,    -1,    -1,   107,    -1,   109,   110,     3,    -1,     5,
     6,     7,    -1,    -1,    10,    11,    12,    13,    -1,    -1,
    16,    17,    -1,    19,    20,    21,    -1,    -1,    24,    -1,
    -1,    -1,    -1,    -1,    30,    -1,    -1,    33,    -1,    35,
    36,    37,    38,    39,    40,    -1,    -1,    43,    44,    -1,
    -1,    47,    -1,    -1,    -1,    -1,    -1,    53,    54,    55,
    56,    57,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    66,    67,    68,    69,    70,    71,    72,    73,    -1,    75,
    76,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    99,    -1,   101,   102,   103,    -1,    -1,
    -1,   107,    -1,   109,   110,     3,    -1,     5,     6,     7,
    -1,    -1,    -1,    -1,    12,    13,    14,    -1,    16,    17,
    -1,    19,    20,    21,    -1,    -1,    24,    -1,    -1,    -1,
    -1,    -1,    30,    -1,    -1,    33,    34,    35,    36,    37,
    38,    39,    40,    -1,    -1,    43,    44,    -1,    -1,    47,
    48,    -1,    -1,    -1,    -1,    53,    -1,    55,    56,    57,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    66,    67,
    68,    69,    70,    71,    72,    73,    -1,    75,    76,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     3,    -1,
     5,     6,     7,    -1,    -1,    -1,    -1,    12,    13,    14,
    -1,    99,    -1,   101,   102,   103,    21,    -1,    -1,   107,
    -1,   109,   110,     3,    -1,     5,     6,     7,    33,    34,
    -1,    -1,    12,    13,    -1,    -1,    16,    17,    -1,    19,
    20,    21,    -1,    48,    24,    -1,    -1,    -1,    -1,    -1,
    30,    56,    57,    33,    -1,    35,    36,    37,    38,    39,
    40,    66,    -1,    43,    44,    -1,    -1,    47,    73,    -1,
    -1,    -1,    -1,    53,    54,    55,    56,    57,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    66,    67,    68,    69,
    70,    71,    72,    73,    99,    75,    76,    -1,   103,    -1,
    -1,    -1,   107,    -1,   109,    -1,     3,    -1,     5,     6,
     7,    -1,    -1,    -1,    -1,    12,    13,    14,    -1,    99,
   100,   101,   102,   103,    21,    -1,    -1,   107,    -1,   109,
   110,     3,    -1,     5,     6,     7,    33,    34,    -1,    -1,
    12,    13,    -1,    -1,    16,    17,    -1,    19,    20,    21,
    -1,    48,    24,    -1,    -1,    -1,    -1,    -1,    30,    56,
    57,    33,    -1,    35,    36,    37,    38,    39,    40,    66,
    -1,    43,    44,    -1,    -1,    47,    73,    -1,    -1,    -1,
    -1,    53,    54,    55,    56,    57,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    66,    67,    68,    69,    70,    71,
    72,    73,    99,    75,    76,    -1,   103,    -1,    -1,    -1,
   107,    -1,   109,    -1,     3,    -1,     5,     6,     7,    -1,
    -1,    -1,    -1,    12,    13,    14,    -1,    99,   100,   101,
   102,   103,    21,    -1,    -1,   107,    -1,   109,   110,     3,
    -1,     5,     6,     7,    33,    34,    -1,    -1,    12,    13,
    -1,    -1,    16,    17,    -1,    19,    20,    21,    -1,    48,
    24,    -1,    -1,    -1,    -1,    -1,    30,    56,    57,    33,
    -1,    35,    36,    37,    38,    39,    40,    66,    -1,    43,
    44,    -1,    -1,    47,    73,    -1,    -1,    -1,    -1,    53,
    54,    55,    56,    57,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    66,    67,    68,    69,    70,    71,    72,    73,
    99,    75,    76,    -1,   103,    -1,    -1,    -1,   107,    -1,
   109,    -1,     3,    -1,     5,     6,     7,    -1,    -1,    -1,
    -1,    12,    13,    -1,    -1,    99,   100,   101,   102,   103,
    21,    -1,    -1,   107,    -1,   109,   110,     3,    -1,     5,
     6,     7,    33,    -1,    -1,    -1,    12,    13,    -1,    -1,
    16,    17,    -1,    19,    20,    21,    -1,    -1,    24,    -1,
    -1,    -1,    -1,    -1,    30,    56,    57,    33,    -1,    35,
    36,    37,    38,    39,    40,    66,    -1,    43,    44,    -1,
    -1,    47,    73,    -1,    -1,    -1,    -1,    53,    54,    55,
    56,    57,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    66,    67,    68,    69,    70,    71,    72,    73,    99,    75,
    76,    -1,   103,    -1,    -1,    -1,   107,    -1,   109,    -1,
     3,    -1,     5,     6,     7,    -1,    -1,    -1,    -1,    12,
    13,    -1,    -1,    99,   100,   101,   102,   103,    21,    -1,
    -1,   107,    -1,   109,   110,     3,    -1,     5,     6,     7,
    33,    -1,    -1,    -1,    12,    13,    -1,    -1,    16,    17,
    -1,    19,    20,    21,    -1,    -1,    24,    -1,    -1,    -1,
    -1,    -1,    30,    56,    57,    33,    -1,    35,    36,    37,
    38,    39,    40,    66,    -1,    43,    44,    -1,    -1,    47,
    73,    -1,    -1,    -1,    -1,    53,    54,    55,    56,    57,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    66,    67,
    68,    69,    70,    71,    72,    73,    99,    75,    76,    -1,
   103,    -1,    -1,    -1,   107,    -1,   109,    -1,     3,    -1,
     5,     6,     7,    -1,    -1,    -1,    -1,    12,    13,    -1,
    -1,    99,   100,   101,   102,   103,    21,    -1,    -1,   107,
    -1,   109,   110,     3,    -1,     5,     6,     7,    33,    -1,
    -1,    -1,    12,    13,    -1,    -1,    16,    17,    -1,    19,
    20,    21,    -1,    -1,    24,    -1,    -1,    -1,    -1,    -1,
    30,    56,    57,    33,    -1,    35,    36,    37,    38,    39,
    40,    66,    -1,    43,    44,    -1,    -1,    47,    73,    -1,
    -1,    -1,    -1,    53,    54,    55,    56,    57,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    66,    67,    68,    69,
    70,    71,    72,    73,    99,    75,    76,    -1,   103,    -1,
    -1,    -1,   107,    -1,   109,    -1,     3,    -1,     5,     6,
     7,    -1,    -1,    -1,    -1,    12,    13,    -1,    -1,    99,
   100,   101,   102,   103,    21,    -1,    -1,   107,    -1,   109,
   110,     3,    -1,     5,     6,     7,    33,    -1,    -1,    -1,
    12,    13,    -1,    -1,    16,    17,    -1,    19,    20,    21,
    -1,    -1,    24,    -1,    -1,    -1,    -1,    -1,    30,    56,
    57,    33,    -1,    35,    36,    37,    38,    39,    40,    66,
    -1,    43,    44,    -1,    -1,    47,    73,    -1,    -1,    -1,
    -1,    53,    54,    55,    56,    57,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    66,    67,    68,    69,    70,    71,
    72,    73,    99,    75,    76,    -1,   103,    -1,    -1,    -1,
   107,    -1,   109,    -1,     3,    -1,     5,     6,     7,    -1,
    -1,    -1,    -1,    12,    13,    -1,    -1,    99,   100,   101,
   102,   103,    21,    -1,    -1,   107,    -1,   109,   110,     3,
    -1,     5,     6,     7,    33,    -1,    -1,    -1,    12,    13,
    -1,    -1,    16,    17,    -1,    19,    20,    21,    -1,    -1,
    24,    -1,    -1,    -1,    -1,    -1,    30,    56,    57,    33,
    -1,    35,    36,    37,    38,    39,    40,    66,    -1,    43,
    44,    -1,    -1,    47,    73,    -1,    -1,    -1,    -1,    53,
    54,    55,    56,    57,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    66,    67,    68,    69,    70,    71,    72,    73,
    99,    75,    76,    -1,   103,    -1,    -1,    -1,   107,    -1,
   109,    -1,     3,    -1,     5,     6,     7,    -1,    -1,    -1,
    -1,    12,    13,    -1,    -1,    99,   100,   101,   102,   103,
    21,    -1,    -1,   107,    -1,   109,   110,     3,    -1,     5,
     6,     7,    33,    -1,    -1,    -1,    12,    13,    -1,    -1,
    16,    17,    -1,    19,    20,    21,    -1,    -1,    24,    -1,
    -1,    -1,    -1,    -1,    30,    56,    57,    33,    -1,    35,
    36,    37,    38,    39,    40,    66,    -1,    43,    44,    -1,
    -1,    47,    73,    -1,    -1,    -1,    -1,    53,    54,    55,
    56,    57,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    66,    67,    68,    69,    70,    71,    72,    73,    99,    75,
    76,    -1,   103,    -1,    -1,    -1,   107,    -1,   109,    -1,
    86,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    99,    -1,   101,   102,   103,    -1,    -1,
    -1,   107,    -1,   109,   110,     3,    -1,     5,     6,     7,
    -1,    -1,    -1,    -1,    12,    13,    -1,    -1,    16,    17,
    -1,    19,    20,    21,    -1,    -1,    24,    -1,    -1,    -1,
    -1,    -1,    30,    -1,    -1,    33,    -1,    35,    36,    37,
    38,    39,    40,    -1,    -1,    43,    44,    -1,    -1,    47,
    -1,    -1,    -1,    -1,    -1,    53,    54,    55,    56,    57,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    66,    67,
    68,    69,    70,    71,    72,    73,    -1,    75,    76,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,     3,    -1,
     5,     6,     7,    -1,    -1,    -1,    -1,    12,    13,    -1,
    -1,    99,   100,   101,   102,   103,    21,    -1,    -1,   107,
    -1,   109,   110,     3,    -1,     5,     6,     7,    33,    -1,
    -1,    -1,    12,    13,    -1,    -1,    16,    17,    -1,    19,
    20,    21,    -1,    -1,    24,    -1,    -1,    -1,    -1,    -1,
    30,    56,    57,    33,    -1,    35,    36,    37,    38,    39,
    40,    66,    -1,    43,    44,    -1,    -1,    47,    73,    -1,
    -1,    -1,    -1,    53,    54,    55,    56,    57,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    66,    67,    68,    69,
    70,    71,    72,    73,    99,    75,    76,    -1,   103,    -1,
    -1,    -1,   107,    -1,   109,    -1,     3,    -1,     5,     6,
     7,    -1,    -1,    -1,    -1,    12,    13,    -1,    -1,    99,
   100,   101,   102,   103,    21,    -1,    -1,   107,    -1,   109,
   110,     3,    -1,     5,     6,     7,    33,    -1,    -1,    -1,
    12,    13,    -1,    -1,    16,    17,    -1,    19,    20,    21,
    -1,    -1,    24,    -1,    -1,    -1,    -1,    -1,    30,    56,
    57,    33,    -1,    35,    36,    37,    38,    39,    40,    66,
    -1,    43,    44,    -1,    -1,    47,    73,    -1,    -1,    -1,
    -1,    53,    54,    55,    56,    57,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    66,    67,    68,    69,    70,    71,
    72,    73,    99,    75,    76,    -1,   103,    -1,    -1,    -1,
   107,    -1,   109,    -1,     3,    -1,     5,     6,     7,    -1,
    -1,    -1,    -1,    12,    13,    -1,    -1,    99,    -1,   101,
   102,   103,    21,    -1,    -1,   107,    -1,   109,   110,     3,
    -1,     5,     6,     7,    33,    -1,    -1,    -1,    12,    13,
    -1,    -1,    16,    17,    -1,    19,    20,    21,    -1,    -1,
    24,    -1,    -1,    -1,    -1,    -1,    30,    56,    57,    33,
    -1,    35,    36,    37,    38,    39,    40,    66,    -1,    43,
    44,    -1,    -1,    47,    73,    -1,    -1,    -1,    -1,    53,
    54,    55,    56,    57,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    66,    67,    68,    69,    70,    71,    72,    73,
    99,    75,    76,    -1,   103,    -1,    -1,    -1,   107,    -1,
   109,    -1,     3,    -1,     5,     6,     7,    -1,    -1,    -1,
    -1,    12,    13,    -1,    -1,    99,    -1,   101,   102,   103,
    21,    -1,    -1,   107,    -1,   109,   110,     3,    -1,     5,
     6,     7,    33,    -1,    -1,    -1,    12,    13,    -1,    -1,
    16,    17,    -1,    19,    20,    21,    -1,    -1,    24,    -1,
    -1,    -1,    -1,    -1,    30,    56,    57,    33,    -1,    35,
    36,    37,    38,    39,    40,    66,    -1,    43,    44,    -1,
    -1,    47,    73,    -1,    -1,    -1,    -1,    53,    -1,    55,
    56,    57,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    66,    67,    68,    69,    70,    71,    72,    73,    99,    75,
    76,    -1,   103,    -1,    -1,    -1,   107,    -1,   109,    -1,
     3,    -1,     5,     6,     7,    -1,    -1,    -1,    -1,    12,
    13,    -1,    -1,    99,    -1,   101,   102,   103,    21,    -1,
    -1,   107,    -1,   109,   110,     3,    -1,     5,     6,     7,
    33,    -1,    -1,    -1,    12,    13,    -1,    -1,    16,    17,
    -1,    19,    20,    21,    -1,    -1,    24,    -1,    -1,    -1,
    -1,    -1,    30,    56,    57,    33,    -1,    35,    36,    37,
    38,    39,    40,    66,    -1,    43,    44,    -1,    -1,    47,
    73,    -1,    -1,    -1,    -1,    53,    -1,    55,    56,    57,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    66,    67,
    68,    69,    70,    71,    72,    73,    99,    75,    76,    -1,
   103,    -1,    -1,    -1,   107,    -1,   109,    -1,     3,    -1,
     5,     6,     7,    -1,    -1,    -1,    -1,    12,    13,    -1,
    -1,    99,    -1,   101,   102,   103,    21,    -1,    -1,   107,
    -1,   109,   110,     3,    -1,     5,     6,     7,    33,    -1,
    -1,    -1,    12,    13,    -1,    -1,    16,    17,    -1,    19,
    20,    21,    -1,    -1,    24,    -1,    -1,    -1,    -1,    -1,
    30,    56,    57,    33,    -1,    35,    36,    37,    38,    39,
    40,    66,    -1,    43,    44,    -1,    -1,    47,    73,    -1,
    -1,    -1,    -1,    53,    -1,    55,    56,    57,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    66,    67,    68,    69,
    70,    71,    72,    73,    99,    75,    76,    -1,   103,    -1,
    -1,    -1,   107,    -1,   109,    -1,     3,    -1,     5,     6,
     7,    -1,    -1,    -1,    -1,    12,    13,    -1,    -1,    99,
    -1,   101,   102,   103,    21,    -1,    -1,   107,    -1,   109,
   110,     3,    -1,     5,     6,     7,    33,    -1,    -1,    -1,
    12,    13,    -1,    -1,    16,    17,    -1,    19,    20,    21,
    -1,    -1,    24,    -1,    -1,    -1,    -1,    -1,    30,    56,
    57,    33,    -1,    35,    36,    37,    38,    39,    40,    66,
    -1,    43,    44,    -1,    -1,    47,    73,    -1,    -1,    -1,
    -1,    53,    -1,    55,    56,    57,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    66,    67,    68,    69,    70,    71,
    72,    73,    99,    75,    76,    -1,   103,    -1,    -1,    -1,
   107,    -1,   109,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    99,    -1,   101,
   102,   103,    -1,    -1,    -1,   107,    -1,   109,   110,     3,
    -1,     5,     6,     7,     8,    -1,    -1,    -1,    12,    13,
    14,    -1,    -1,    -1,    18,    19,    20,    21,    -1,    -1,
    24,    -1,    -1,    27,    28,    29,    30,    31,    -1,    33,
    34,    35,    36,    37,    38,    39,    40,    -1,    -1,    43,
    44,    -1,    -1,    -1,    48,    -1,    -1,    51,    -1,    -1,
    -1,    -1,    56,    57,    58,    59,    60,    61,    62,    63,
    64,    65,    66,    -1,    -1,    -1,    -1,    -1,    -1,    73,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    86,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    99,    -1,    -1,     3,   103,
     5,     6,     7,   107,    -1,   109,    -1,    12,    13,    14,
    -1,    -1,    -1,    18,    19,    20,    21,    -1,    -1,    24,
    -1,    -1,    27,    28,    29,    30,    31,    -1,    33,    34,
    35,    36,    37,    38,    39,    40,    -1,    -1,    43,    44,
    -1,    -1,    -1,    48,    -1,    -1,    51,    -1,    -1,    -1,
    -1,    56,    57,    -1,    -1,    -1,    61,    62,    63,    64,
    65,    66,    -1,    -1,    -1,    -1,    -1,    -1,    73,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    85,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    99,   100,    -1,     3,   103,     5,
     6,     7,   107,    -1,   109,    -1,    12,    13,    14,    -1,
    -1,    -1,    18,    19,    20,    21,    -1,    -1,    24,    -1,
    -1,    27,    28,    29,    30,    31,    -1,    33,    34,    35,
    36,    37,    38,    39,    40,    -1,    -1,    43,    44,    -1,
    -1,    -1,    48,    -1,    -1,    51,    -1,    -1,    -1,    -1,
    56,    57,    -1,    -1,    -1,    61,    62,    63,    64,    65,
    66,    -1,    -1,    -1,    -1,    -1,    -1,    73,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    85,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    99,   100,    -1,     3,   103,     5,     6,
     7,   107,    -1,   109,    -1,    12,    13,    14,    -1,    -1,
    -1,    18,    19,    20,    21,    -1,    -1,    24,    -1,    -1,
    27,    28,    29,    30,    31,    -1,    33,    34,    35,    36,
    37,    38,    39,    40,    -1,    -1,    43,    44,    -1,    -1,
    -1,    48,    -1,    -1,    51,    -1,    -1,    -1,    -1,    56,
    57,    -1,    -1,    -1,    61,    62,    63,    64,    65,    66,
    -1,    -1,    -1,    -1,    -1,    -1,    73,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    85,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    99,   100,    -1,     3,   103,     5,     6,     7,
   107,    -1,   109,    -1,    12,    13,    14,    -1,    -1,    -1,
    18,    19,    20,    21,    -1,    -1,    24,    -1,    -1,    27,
    28,    29,    30,    31,    -1,    33,    34,    35,    36,    37,
    38,    39,    40,    -1,    -1,    43,    44,    -1,    -1,    -1,
    48,    -1,    -1,    51,    -1,    -1,    -1,    -1,    56,    57,
    -1,    -1,    -1,    61,    62,    63,    64,    65,    66,    -1,
    -1,    -1,    -1,    -1,    -1,    73,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    99,    -1,    -1,     3,   103,     5,     6,     7,   107,
    -1,   109,    -1,    12,    13,    14,    -1,    -1,    -1,    18,
    19,    20,    21,    -1,    -1,    24,    -1,    -1,    27,    28,
    29,    30,    31,    -1,    33,    34,    35,    36,    37,    38,
    39,    40,    -1,    -1,    43,    44,    -1,    -1,    -1,    48,
    -1,    -1,    51,    -1,    -1,    -1,    -1,    56,    57,    -1,
    -1,    -1,    61,    62,    63,    64,    65,    66,    -1,    -1,
    -1,    -1,    -1,    -1,    73,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    99,    -1,    -1,     3,   103,     5,     6,     7,   107,    -1,
   109,    -1,    12,    13,    14,    -1,    -1,    -1,    18,    19,
    20,    21,    -1,    -1,    24,    -1,    -1,    27,    28,    29,
    30,    31,    -1,    33,    34,    35,    36,    37,    38,    39,
    40,    -1,    -1,    43,    44,    -1,    -1,    -1,    48,    -1,
    -1,    51,    -1,    -1,    -1,    -1,    56,    57,    -1,    -1,
    -1,    61,    62,    63,    64,    65,    66,    -1,    -1,    -1,
    -1,    -1,    -1,    73,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    99,
    -1,    -1,     3,   103,     5,     6,     7,   107,    -1,   109,
    -1,    12,    13,    14,    -1,    -1,    -1,    18,    19,    20,
    21,    -1,    -1,    24,    -1,    -1,    27,    28,    29,    30,
    31,    -1,    33,    34,    35,    36,    37,    38,    39,    40,
    -1,    -1,    43,    44,    -1,    -1,    -1,    48,    -1,    -1,
    51,    -1,    -1,    -1,    -1,    56,    57,    -1,    -1,    -1,
    61,    62,    63,    64,    65,    66,    -1,    -1,    -1,    -1,
    -1,    -1,    73,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    99,    -1,
    -1,     3,   103,     5,     6,     7,   107,    -1,   109,    -1,
    12,    13,    14,    -1,    -1,    -1,    18,    19,    20,    21,
    -1,    -1,    24,    -1,    -1,    -1,    28,    29,    30,    31,
    -1,    33,    34,    35,    36,    37,    38,    39,    40,    -1,
    -1,    43,    44,    -1,    -1,    -1,    48,    -1,    -1,    51,
    -1,    -1,    -1,    -1,    56,    57,    -1,    -1,    -1,    61,
    62,    63,    64,    65,    66,    -1,    -1,    -1,    -1,    -1,
    -1,    73,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    99,    -1,    -1,
     3,   103,     5,     6,     7,   107,    -1,   109,    -1,    12,
    13,    14,    -1,    -1,    -1,    18,    19,    20,    21,    -1,
    -1,    24,    -1,    -1,    -1,    28,    29,    30,    31,    -1,
    33,    34,    35,    36,    37,    38,    39,    40,    -1,    -1,
    43,    44,    -1,    -1,    -1,    48,    -1,    -1,    51,    -1,
    -1,    -1,    -1,    56,    57,    -1,    -1,    -1,    61,    62,
    63,    64,    65,    66,    -1,    -1,    -1,    -1,    -1,    -1,
    73,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    -1,     3,    -1,     5,    -1,    99,    -1,    -1,    -1,
   103,    12,    13,    14,   107,    -1,   109,    18,    19,    20,
    21,    -1,    -1,    24,    -1,    -1,    -1,    28,    29,    30,
    31,    -1,    33,    34,    35,    36,    37,    38,    39,    40,
    -1,    -1,    43,    44,    -1,    -1,    -1,    48,    -1,    -1,
    51,    -1,    -1,    -1,    -1,    56,    57,    -1,    -1,    -1,
    61,    62,    63,    64,    65,    66,     3,    -1,     5,     6,
     7,    -1,    73,    -1,    -1,    12,    13,    14,    -1,    -1,
    -1,    18,    -1,    -1,    21,    -1,    -1,    -1,    -1,    -1,
    -1,    28,    29,    -1,    31,    -1,    33,    34,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   109,    -1,
    -1,    48,    -1,    -1,    51,    -1,    -1,    -1,    -1,    56,
    57,    -1,    -1,    -1,    61,    62,    63,    64,    65,    66,
     3,    -1,     5,     6,     7,    -1,    73,    -1,    -1,    12,
    13,    14,    -1,    -1,    -1,    18,    -1,    -1,    21,    -1,
    -1,    -1,    -1,    -1,    -1,    28,    29,    -1,    31,    -1,
    33,    34,    99,    -1,    -1,    -1,   103,    -1,    -1,    -1,
   107,    -1,   109,    -1,    -1,    48,    -1,    -1,    51,    -1,
    -1,    -1,    -1,    56,    57,    -1,    -1,    -1,    61,    62,
    63,    64,    65,    66,     3,    -1,     5,     6,     7,    -1,
    73,    -1,    -1,    12,    13,    14,    -1,    -1,    -1,    18,
    -1,    -1,    21,    -1,    -1,    -1,    -1,    -1,    -1,    28,
    29,    -1,    31,    -1,    33,    34,    99,    -1,    -1,    -1,
   103,    -1,    -1,    -1,   107,    -1,   109,    -1,    -1,    48,
    -1,    -1,    51,    -1,    -1,    -1,    -1,    56,    57,    -1,
    -1,    -1,    61,    62,    63,    64,    65,    66,     3,    -1,
     5,     6,     7,    -1,    73,    -1,    -1,    12,    13,    14,
    -1,    -1,    -1,    18,    -1,    -1,    21,    -1,    -1,    -1,
    -1,    -1,    -1,    28,    29,    -1,    31,    -1,    33,    34,
    99,    -1,    -1,    -1,   103,    -1,    -1,    -1,   107,    -1,
   109,    -1,    -1,    48,    -1,    -1,    51,    -1,    -1,    -1,
    -1,    56,    57,    -1,    -1,    -1,    61,    62,    63,    64,
    65,    66,     3,    -1,     5,     6,     7,    -1,    73,    -1,
    -1,    12,    13,    14,    -1,    -1,    -1,    18,    -1,    -1,
    21,    -1,    -1,    -1,    -1,    -1,    -1,    28,    29,    -1,
    31,    -1,    33,    34,    99,    -1,    -1,    -1,   103,    -1,
    -1,    -1,   107,    -1,   109,    -1,    -1,    48,    -1,    -1,
    51,    -1,    -1,    -1,    -1,    56,    57,    -1,    -1,    -1,
    61,    62,    63,    64,    65,    66,     3,    -1,     5,     6,
     7,    -1,    73,    -1,    -1,    12,    13,     3,    -1,     5,
     6,     7,    -1,    -1,    21,    -1,    12,    13,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    21,    33,    -1,    99,    -1,
    -1,    -1,   103,    -1,    -1,    -1,   107,    33,   109,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    56,
    57,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    66,
    56,    57,    -1,    -1,    -1,     3,    73,     5,     6,     7,
    66,    -1,    -1,    -1,    12,    13,     3,    73,     5,     6,
     7,    -1,    -1,    21,    -1,    12,    13,    -1,    -1,    -1,
    -1,    -1,    99,    -1,    21,    33,   103,    -1,    -1,    -1,
   107,    -1,   109,    99,    -1,    -1,    33,   103,    -1,    -1,
    -1,   107,    -1,   109,    -1,    -1,    -1,    -1,    56,    57,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    66,    56,
    57,    -1,    -1,    -1,    -1,    73,    -1,    -1,    -1,    66,
    -1,    -1,    -1,    -1,    -1,    -1,    73,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
    -1,    99,    -1,    -1,    -1,   103,    -1,    -1,    -1,   107,
    -1,   109,    99,    -1,    -1,     3,   103,     5,     6,     7,
   107,    -1,   109,    -1,    12,    13,    14,    -1,    -1,    -1,
    -1,    19,    20,    21,    -1,    -1,    24,    -1,    -1,    27,
    -1,    -1,    30,    -1,    -1,    33,    34,    35,    36,    37,
    38,    39,    40,    -1,    -1,    43,    44,    -1,    -1,    -1,
    48,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    57,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    66,    -1,
    -1,    -1,    -1,    -1,    -1,    73,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    86,    -1,
    -1,    -1,    -1,    -1,    -1,    -1,    -1,     3,    -1,     5,
    -1,    -1,    -1,    -1,    -1,   103,    12,    13,    14,   107,
    -1,    -1,    18,    19,    20,    21,    -1,    -1,    24,    -1,
    -1,    27,    28,    29,    30,    31,    -1,    33,    34,    35,
    36,    37,    38,    39,    40,    -1,    -1,    43,    44,    -1,
    -1,    -1,    48,    -1,    -1,    51,    -1,    -1,    -1,    -1,
    -1,    57,    -1,    -1,    -1,    61,    62,    63,    64,    65,
    66,    -1,    -1,    -1,    -1,    -1,     3,    73,     5,    -1,
    -1,    -1,    -1,    -1,    -1,    12,    13,    14,    -1,    85,
    -1,    -1,    19,    20,    21,    -1,    -1,    24,    -1,    -1,
    27,    -1,    -1,    30,   100,    -1,    33,    34,    35,    36,
    37,    38,    39,    40,    -1,    -1,    43,    44,    -1,    -1,
    -1,    48,    -1,    -1,     3,    -1,     5,    -1,    -1,    -1,
    57,    -1,    -1,    12,    13,    14,    -1,    -1,    -1,    66,
    19,    20,    21,    -1,    -1,    24,    73,    -1,    27,    -1,
    -1,    30,    -1,    -1,    33,    34,    35,    36,    37,    38,
    39,    40,    -1,    -1,    43,    44,    -1,    -1,    -1,    48,
    -1,    -1,    99,    -1,     3,    -1,     5,    -1,    57,    -1,
    -1,    -1,    -1,    12,    13,    14,    -1,    66,    -1,    18,
    19,    20,    21,    -1,    73,    24,    -1,    -1,    27,    28,
    29,    30,    31,    -1,    33,    34,    35,    36,    37,    38,
    39,    40,    -1,    -1,    43,    44,    -1,    -1,    -1,    48,
    99,    -1,    51,    -1,    -1,    -1,    -1,    -1,    57,    -1,
    -1,    -1,    61,    62,    63,    64,    65,    66,    -1,    -1,
    -1,     3,    -1,     5,    73,    -1,    -1,    -1,    -1,    -1,
    12,    13,    14,    -1,    -1,    -1,    85,    19,    20,    21,
    -1,    -1,    24,    -1,    -1,    27,    -1,    -1,    30,    -1,
    -1,    33,    34,    35,    36,    37,    38,    39,    40,    -1,
    -1,    43,    44,    -1,    -1,    -1,    48,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    -1,    57,    -1,    -1,    -1,    -1,
    -1,    -1,    -1,    -1,    66,    -1,    -1,    -1,    -1,    -1,
    -1,    73
};
/* -*-C-*-  Note some compilers choke on comments on `#line' lines.  */
#line 3 "/local/encap/bison-1.25/share/bison.simple"

/* Skeleton output parser for bison,
   Copyright (C) 1984, 1989, 1990 Free Software Foundation, Inc.

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
   Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.  */

/* As a special exception, when this file is copied by Bison into a
   Bison output file, you may use that output file without restriction.
   This special exception was added by the Free Software Foundation
   in version 1.24 of Bison.  */

#ifndef alloca
#ifdef __GNUC__
#define alloca __builtin_alloca
#else /* not GNU C.  */
#if (!defined (__STDC__) && defined (sparc)) || defined (__sparc__) || defined (__sparc) || defined (__sgi)
#include <alloca.h>
#else /* not sparc */
#if defined (MSDOS) && !defined (__TURBOC__)
#include <malloc.h>
#else /* not MSDOS, or __TURBOC__ */
#if defined(_AIX)
#include <malloc.h>
 #pragma alloca
#else /* not MSDOS, __TURBOC__, or _AIX */
#ifdef __hpux
#ifdef __cplusplus
extern "C" {
void *alloca (unsigned int);
};
#else /* not __cplusplus */
void *alloca ();
#endif /* not __cplusplus */
#endif /* __hpux */
#endif /* not _AIX */
#endif /* not MSDOS, or __TURBOC__ */
#endif /* not sparc.  */
#endif /* not GNU C.  */
#endif /* alloca not defined.  */

/* This is the parser code that is written into each bison parser
  when the %semantic_parser declaration is not specified in the grammar.
  It was written by Richard Stallman by simplifying the hairy parser
  used when %semantic_parser is specified.  */

/* Note: there must be only one dollar sign in this file.
   It is replaced by the list of actions, each action
   as one case of the switch.  */

#define yyerrok		(yyerrstatus = 0)
#define yyclearin	(yychar = YYEMPTY)
#define YYEMPTY		-2
#define YYEOF		0
#define YYACCEPT	return(0)
#define YYABORT 	return(1)
#define YYERROR		goto yyerrlab1
/* Like YYERROR except do call yyerror.
   This remains here temporarily to ease the
   transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  */
#define YYFAIL		goto yyerrlab
#define YYRECOVERING()  (!!yyerrstatus)
#define YYBACKUP(token, value) \
do								\
  if (yychar == YYEMPTY && yylen == 1)				\
    { yychar = (token), yylval = (value);			\
      yychar1 = YYTRANSLATE (yychar);				\
      YYPOPSTACK;						\
      goto yybackup;						\
    }								\
  else								\
    { yyerror ("syntax error: cannot back up"); YYERROR; }	\
while (0)

#define YYTERROR	1
#define YYERRCODE	256

#ifndef YYPURE
#define YYLEX		yylex()
#endif

#ifdef YYPURE
#ifdef YYLSP_NEEDED
#ifdef YYLEX_PARAM
#define YYLEX		yylex(&yylval, &yylloc, YYLEX_PARAM)
#else
#define YYLEX		yylex(&yylval, &yylloc)
#endif
#else /* not YYLSP_NEEDED */
#ifdef YYLEX_PARAM
#define YYLEX		yylex(&yylval, YYLEX_PARAM)
#else
#define YYLEX		yylex(&yylval)
#endif
#endif /* not YYLSP_NEEDED */
#endif

/* If nonreentrant, generate the variables here */

#ifndef YYPURE

int	yychar;			/*  the lookahead symbol		*/
YYSTYPE	yylval;			/*  the semantic value of the		*/
				/*  lookahead symbol			*/

#ifdef YYLSP_NEEDED
YYLTYPE yylloc;			/*  location data for the lookahead	*/
				/*  symbol				*/
#endif

int yynerrs;			/*  number of parse errors so far       */
#endif  /* not YYPURE */

#if YYDEBUG != 0
int yydebug;			/*  nonzero means print parse trace	*/
/* Since this is uninitialized, it does not stop multiple parsers
   from coexisting.  */
#endif

/*  YYINITDEPTH indicates the initial size of the parser's stacks	*/

#ifndef	YYINITDEPTH
#define YYINITDEPTH 200
#endif

/*  YYMAXDEPTH is the maximum size the stacks can grow to
    (effective only if the built-in stack extension method is used).  */

#if YYMAXDEPTH == 0
#undef YYMAXDEPTH
#endif

#ifndef YYMAXDEPTH
#define YYMAXDEPTH 10000
#endif

/* Prevent warning if -Wstrict-prototypes.  */
#ifdef __GNUC__
int yyparse (void);
#endif

#if __GNUC__ > 1		/* GNU C and GNU C++ define this.  */
#define __yy_memcpy(TO,FROM,COUNT)	__builtin_memcpy(TO,FROM,COUNT)
#else				/* not GNU C or C++ */
#ifndef __cplusplus

/* This is the most reliable way to avoid incompatibilities
   in available built-in functions on various systems.  */
static void
__yy_memcpy (to, from, count)
     char *to;
     char *from;
     int count;
{
  register char *f = from;
  register char *t = to;
  register int i = count;

  while (i-- > 0)
    *t++ = *f++;
}

#else /* __cplusplus */

/* This is the most reliable way to avoid incompatibilities
   in available built-in functions on various systems.  */
static void
__yy_memcpy (char *to, char *from, int count)
{
  register char *f = from;
  register char *t = to;
  register int i = count;

  while (i-- > 0)
    *t++ = *f++;
}

#endif
#endif

#line 196 "/local/encap/bison-1.25/share/bison.simple"

/* The user can define YYPARSE_PARAM as the name of an argument to be passed
   into yyparse.  The argument should have type void *.
   It should actually point to an object.
   Grammar actions can access the variable by casting it
   to the proper pointer type.  */

#ifdef YYPARSE_PARAM
#ifdef __cplusplus
#define YYPARSE_PARAM_ARG void *YYPARSE_PARAM
#define YYPARSE_PARAM_DECL
#else /* not __cplusplus */
#define YYPARSE_PARAM_ARG YYPARSE_PARAM
#define YYPARSE_PARAM_DECL void *YYPARSE_PARAM;
#endif /* not __cplusplus */
#else /* not YYPARSE_PARAM */
#define YYPARSE_PARAM_ARG
#define YYPARSE_PARAM_DECL
#endif /* not YYPARSE_PARAM */

int
yyparse(YYPARSE_PARAM_ARG)
     YYPARSE_PARAM_DECL
{
  register int yystate;
  register int yyn;
  register short *yyssp;
  register YYSTYPE *yyvsp;
  int yyerrstatus;	/*  number of tokens to shift before error messages enabled */
  int yychar1 = 0;		/*  lookahead token as an internal (translated) token number */

  short	yyssa[YYINITDEPTH];	/*  the state stack			*/
  YYSTYPE yyvsa[YYINITDEPTH];	/*  the semantic value stack		*/

  short *yyss = yyssa;		/*  refer to the stacks thru separate pointers */
  YYSTYPE *yyvs = yyvsa;	/*  to allow yyoverflow to reallocate them elsewhere */

#ifdef YYLSP_NEEDED
  YYLTYPE yylsa[YYINITDEPTH];	/*  the location stack			*/
  YYLTYPE *yyls = yylsa;
  YYLTYPE *yylsp;

#define YYPOPSTACK   (yyvsp--, yyssp--, yylsp--)
#else
#define YYPOPSTACK   (yyvsp--, yyssp--)
#endif

  int yystacksize = YYINITDEPTH;

#ifdef YYPURE
  int yychar;
  YYSTYPE yylval;
  int yynerrs;
#ifdef YYLSP_NEEDED
  YYLTYPE yylloc;
#endif
#endif

  YYSTYPE yyval;		/*  the variable used to return		*/
				/*  semantic values from the action	*/
				/*  routines				*/

  int yylen;

#if YYDEBUG != 0
  if (yydebug)
    fprintf(stderr, "Starting parse\n");
#endif

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY;		/* Cause a token to be read.  */

  /* Initialize stack pointers.
     Waste one element of value and location stack
     so that they stay on the same level as the state stack.
     The wasted elements are never initialized.  */

  yyssp = yyss - 1;
  yyvsp = yyvs;
#ifdef YYLSP_NEEDED
  yylsp = yyls;
#endif

/* Push a new state, which is found in  yystate  .  */
/* In all cases, when you get here, the value and location stacks
   have just been pushed. so pushing a state here evens the stacks.  */
yynewstate:

  *++yyssp = yystate;

  if (yyssp >= yyss + yystacksize - 1)
    {
      /* Give user a chance to reallocate the stack */
      /* Use copies of these so that the &'s don't force the real ones into memory. */
      YYSTYPE *yyvs1 = yyvs;
      short *yyss1 = yyss;
#ifdef YYLSP_NEEDED
      YYLTYPE *yyls1 = yyls;
#endif

      /* Get the current used size of the three stacks, in elements.  */
      int size = yyssp - yyss + 1;

#ifdef yyoverflow
      /* Each stack pointer address is followed by the size of
	 the data in use in that stack, in bytes.  */
#ifdef YYLSP_NEEDED
      /* This used to be a conditional around just the two extra args,
	 but that might be undefined if yyoverflow is a macro.  */
      yyoverflow("parser stack overflow",
		 &yyss1, size * sizeof (*yyssp),
		 &yyvs1, size * sizeof (*yyvsp),
		 &yyls1, size * sizeof (*yylsp),
		 &yystacksize);
#else
      yyoverflow("parser stack overflow",
		 &yyss1, size * sizeof (*yyssp),
		 &yyvs1, size * sizeof (*yyvsp),
		 &yystacksize);
#endif

      yyss = yyss1; yyvs = yyvs1;
#ifdef YYLSP_NEEDED
      yyls = yyls1;
#endif
#else /* no yyoverflow */
      /* Extend the stack our own way.  */
      if (yystacksize >= YYMAXDEPTH)
	{
	  yyerror("parser stack overflow");
	  return 2;
	}
      yystacksize *= 2;
      if (yystacksize > YYMAXDEPTH)
	yystacksize = YYMAXDEPTH;
      yyss = (short *) alloca (yystacksize * sizeof (*yyssp));
      __yy_memcpy ((char *)yyss, (char *)yyss1, size * sizeof (*yyssp));
      yyvs = (YYSTYPE *) alloca (yystacksize * sizeof (*yyvsp));
      __yy_memcpy ((char *)yyvs, (char *)yyvs1, size * sizeof (*yyvsp));
#ifdef YYLSP_NEEDED
      yyls = (YYLTYPE *) alloca (yystacksize * sizeof (*yylsp));
      __yy_memcpy ((char *)yyls, (char *)yyls1, size * sizeof (*yylsp));
#endif
#endif /* no yyoverflow */

      yyssp = yyss + size - 1;
      yyvsp = yyvs + size - 1;
#ifdef YYLSP_NEEDED
      yylsp = yyls + size - 1;
#endif

#if YYDEBUG != 0
      if (yydebug)
	fprintf(stderr, "Stack size increased to %d\n", yystacksize);
#endif

      if (yyssp >= yyss + yystacksize - 1)
	YYABORT;
    }

#if YYDEBUG != 0
  if (yydebug)
    fprintf(stderr, "Entering state %d\n", yystate);
#endif

  goto yybackup;
 yybackup:

/* Do appropriate processing given the current state.  */
/* Read a lookahead token if we need one and don't already have one.  */
/* yyresume: */

  /* First try to decide what to do without reference to lookahead token.  */

  yyn = yypact[yystate];
  if (yyn == YYFLAG)
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* yychar is either YYEMPTY or YYEOF
     or a valid token in external form.  */

  if (yychar == YYEMPTY)
    {
#if YYDEBUG != 0
      if (yydebug)
	fprintf(stderr, "Reading a token: ");
#endif
      yychar = YYLEX;
    }

  /* Convert token to internal form (in yychar1) for indexing tables with */

  if (yychar <= 0)		/* This means end of input. */
    {
      yychar1 = 0;
      yychar = YYEOF;		/* Don't call YYLEX any more */

#if YYDEBUG != 0
      if (yydebug)
	fprintf(stderr, "Now at end of input.\n");
#endif
    }
  else
    {
      yychar1 = YYTRANSLATE(yychar);

#if YYDEBUG != 0
      if (yydebug)
	{
	  fprintf (stderr, "Next token is %d (%s", yychar, yytname[yychar1]);
	  /* Give the individual parser a way to print the precise meaning
	     of a token, for further debugging info.  */
#ifdef YYPRINT
	  YYPRINT (stderr, yychar, yylval);
#endif
	  fprintf (stderr, ")\n");
	}
#endif
    }

  yyn += yychar1;
  if (yyn < 0 || yyn > YYLAST || yycheck[yyn] != yychar1)
    goto yydefault;

  yyn = yytable[yyn];

  /* yyn is what to do for this token type in this state.
     Negative => reduce, -yyn is rule number.
     Positive => shift, yyn is new state.
       New state is final state => don't bother to shift,
       just return success.
     0, or most negative number => error.  */

  if (yyn < 0)
    {
      if (yyn == YYFLAG)
	goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }
  else if (yyn == 0)
    goto yyerrlab;

  if (yyn == YYFINAL)
    YYACCEPT;

  /* Shift the lookahead token.  */

#if YYDEBUG != 0
  if (yydebug)
    fprintf(stderr, "Shifting token %d (%s), ", yychar, yytname[yychar1]);
#endif

  /* Discard the token being shifted unless it is eof.  */
  if (yychar != YYEOF)
    yychar = YYEMPTY;

  *++yyvsp = yylval;
#ifdef YYLSP_NEEDED
  *++yylsp = yylloc;
#endif

  /* count tokens shifted since error; after three, turn off error status.  */
  if (yyerrstatus) yyerrstatus--;

  yystate = yyn;
  goto yynewstate;

/* Do the default action for the current state.  */
yydefault:

  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;

/* Do a reduction.  yyn is the number of a rule to reduce with.  */
yyreduce:
  yylen = yyr2[yyn];
  if (yylen > 0)
    yyval = yyvsp[1-yylen]; /* implement default value of the action */

#if YYDEBUG != 0
  if (yydebug)
    {
      int i;

      fprintf (stderr, "Reducing via rule %d (line %d), ",
	       yyn, yyrline[yyn]);

      /* Print the symbols being reduced, and their result.  */
      for (i = yyprhs[yyn]; yyrhs[i] > 0; i++)
	fprintf (stderr, "%s ", yytname[yyrhs[i]]);
      fprintf (stderr, " -> %s\n", yytname[yyr1[yyn]]);
    }
#endif


  switch (yyn) {

case 1:
#line 356 "xp-t.bison"
{ yyval = yyvsp[0] ; ;
    break;}
case 2:
#line 357 "xp-t.bison"
{ yyval = yyvsp[0] ; ;
    break;}
case 3:
#line 361 "xp-t.bison"
{ yyval = yyvsp[0] ; ;
    break;}
case 4:
#line 362 "xp-t.bison"
{ yyval = yyvsp[0] ; ;
    break;}
case 5:
#line 363 "xp-t.bison"
{ yyval = yyvsp[0] ; ;
    break;}
case 8:
#line 408 "xp-t.bison"
{  yyval = yyvsp[0] ; ;
    break;}
case 9:
#line 409 "xp-t.bison"
{ yyval = yyvsp[0] ; ;
    break;}
case 11:
#line 437 "xp-t.bison"
{ yyval = yyvsp[0] ; ;
    break;}
case 13:
#line 439 "xp-t.bison"
{ yyval = (char *)malloc(sizeof(char)*5);
					  strcpy(yyval,"this") ;
					;
    break;}
case 14:
#line 442 "xp-t.bison"
{ yyval = yyvsp[0] ; ;
    break;}
case 16:
#line 444 "xp-t.bison"
{ yyval = Concat3("(",yyvsp[-1],")") ; ;
    break;}
case 23:
#line 492 "xp-t.bison"
{ 	yyval = (char *)malloc(sizeof(char)*(9+strlen(yyvsp[0]))) ;
		sprintf(yyval,"operator %s",yyvsp[0]) ;	
	;
    break;}
case 63:
#line 585 "xp-t.bison"
{ yyval = yyvsp[0] ; ;
    break;}
case 64:
#line 587 "xp-t.bison"
{	char *temp ;	
		strcpy(SendPe,yyvsp[-1]);
		strcpy(SendChare,yyvsp[-3]) ;
		SendMsgBranchPoss = TRUE ;
		temp = Concat3(yyvsp[-3],"[",yyvsp[-1]) ;
		yyval = Concat2(temp,"]") ;
	  ;
    break;}
case 65:
#line 595 "xp-t.bison"
{ yyval = Concat3(yyvsp[-2],"(",")") ; 
	    if ( MakeGraph && ( InsideChareCode || CurrentAggType==CHARE 
				|| CurrentAggType==BRANCHED) )
		Graph_OutputPrivateCall(yyvsp[-2]) ; 
	  ;
    break;}
case 66:
#line 601 "xp-t.bison"
{ char *charename, *scopestr, *temp ; 

	    /***
	    if ( MakeGraph && ( strcmp($1,"_CK_CreateBoc") == 0 
			        || strcmp($1,"_CK_CreateChare") == 0)  ) 
		Graph_OutputCreate($1,LastArg,LastChare,LastEP) ;
	    ***/
	    if ( SendType != -1 ) {
		char *sptr = Mystrstr(OutBuf,yyvsp[-1]) ;
		if ( sptr != NULL ) 
			*sptr = '\0' ;
		else 
			fprintf(stderr,"TRANSLATOR ERROR : %s, line %d : couldnt discard => etc.\n",CurrentFileName,CurrentLine) ;
		FLUSHBUF() ;
	
	    /* Now output the Send functions */
		scopestr = CheckSendError(SendChare,SendEP,yyvsp[-1],SendType,
								&charename) ;
		OutputSend(SendChare,SendEP,yyvsp[-1],SendType,charename,scopestr,
								SendPe) ;

		SendType = -1 ;
	    }
	    else if ( MakeGraph && ( InsideChareCode || CurrentAggType==CHARE 
				     || CurrentAggType==BRANCHED) )
		Graph_OutputPrivateCall(yyvsp[-3]) ; 

	    temp = Concat3(yyvsp[-3],"(",yyvsp[-1]) ;
	    yyval = Concat2(temp,")") ;
	  ;
    break;}
case 67:
#line 632 "xp-t.bison"
{StructScope=1;;
    break;}
case 68:
#line 633 "xp-t.bison"
{ yyval = Concat3(yyvsp[-3],".",yyvsp[0]) ;StructScope=0; ;
    break;}
case 69:
#line 634 "xp-t.bison"
{StructScope=1;;
    break;}
case 70:
#line 635 "xp-t.bison"
{ if ( strcmp(CurrentSharedHandle,"_CK_NOTACCHANDLE") != 0 ) {
	        int handleindex ;
	        char *sptr = Mystrstr(OutBuf,CurrentSharedHandle) ;

	        if ( sptr != NULL ) 
	            *sptr = '\0' ;
	        else 
	            fprintf(stderr,"TRANSLATOR ERROR in shared object handle use: %s, line %d: \n",CurrentFileName,CurrentLine) ;
	        FLUSHBUF() ;
	        handleindex = SearchHandleTable(AccHandleTable,
				AccHandleTableSize,CurrentSharedHandle) ;
	        if ( handleindex != -1 ) {
	            fprintf(outfile,"((%s *)(_CK_9GetAccDataPtr(GetBocDataPtr(%s))))",AccHandleTable[handleindex].typestr,CurrentSharedHandle) ;
		    IsAccCall = TRUE ;
	        }
	        else if ( (handleindex = SearchHandleTable(MonoHandleTable,
			      MonoHandleTableSize,CurrentSharedHandle)) != -1 )
	        {   fprintf(outfile,"((%s *)(_CK_9GetMonoDataPtr(GetBocDataPtr(%s))))",MonoHandleTable[handleindex].typestr,CurrentSharedHandle) ;
		    IsMonoCall = TRUE ;
	        }
	        else if ( SearchHandleTable(WrOnHandleTable,
				WrOnHandleTableSize,CurrentSharedHandle) != -1)
	       		;  /* action taken at end of this rule */ 
	        else 
	            fprintf(stderr,"TRANSLATOR ERROR: %s, line %d, couldnt find acc/mono handle in table.\n",CurrentFileName,CurrentLine) ;


	        strcat(OutBuf,prevtoken) ; /* prevtoken now is "->" */
	        strcpy(prevtoken,"") ;  
	    }
	  ;
    break;}
case 71:
#line 667 "xp-t.bison"
{	char *wovid ;
		StructScope = 0 ;
	        if ( SearchHandleTable(WrOnHandleTable,
				WrOnHandleTableSize,CurrentSharedHandle) != -1 )
		{	if ( strcmp(yyvsp[0],"DerefWriteOnce") != 0 )
				CharmError("writeonce variables have only the DerefWriteOnce method") ;
			else {	
				wovid = Mystrstr(OutBuf,CurrentSharedHandle) ;
				*wovid = '\0' ;
				strcat(OutBuf,"DerefWriteOnce(") ;
				strcat(OutBuf,CurrentSharedHandle) ;
				strcat(OutBuf,")") ;
			/*	strcpy(prevtoken,"") ;	*/
			}
		}
	        strcpy(CurrentSharedHandle,"_CK_NOTACCHANDLE") ;	
	  	yyval = Concat3(yyvsp[-4],"->",yyvsp[0]) ;
	  ;
    break;}
case 72:
#line 686 "xp-t.bison"
{ yyval = Concat2(yyvsp[-1],"++") ; ;
    break;}
case 73:
#line 688 "xp-t.bison"
{ yyval = Concat2(yyvsp[-1],"--") ; ;
    break;}
case 74:
#line 693 "xp-t.bison"
{	char *sptr ; int i ; char str[64] ;
		sptr = Mystrstr(OutBuf,yyvsp[-3]) ;
		if ( sptr != NULL ) 
			*sptr = '\0' ;
		else 
			fprintf(stderr,"TRANSLATOR ERROR : %s, line %d : couldnt discard [LOCAL] etc.\n",CurrentFileName,CurrentLine) ;
	        i = SearchHandleTable(BOCHandleTable,BOCHandleTableSize,yyvsp[-3]) ;
		if ( i == -1 ) {
			fprintf(stderr,"ERROR : %s, line %d : %s is not a branched chare group id.\n",CurrentFileName,CurrentLine,yyvsp[-3]) ;
		}
		sprintf(str,"((%s *)GetBocDataPtr(%s))",BOCHandleTable[i].typestr,yyvsp[-3]);
		strcat(OutBuf,str) ;
		strcpy(prevtoken,"") ;   /* prevtoken is ']' */
		FLUSHBUF() ;

		if ( MakeGraph ) {
			fprintf(graphfile,"CALLBOC %s %s : %s", CurrentChare, 
					CurrentEP, BOCHandleTable[i].typestr) ;
		}
	  ;
    break;}
case 75:
#line 714 "xp-t.bison"
{
		if ( MakeGraph ) 
			fprintf(graphfile," %s\n",yyvsp[0]) ;
	  ;
    break;}
case 76:
#line 719 "xp-t.bison"
{	if ( !SendMsgBranchPoss ) {
			SendType = SIMPLE ;
			strcpy(SendChare,yyvsp[-1]) ;
		}
		else {
			SendType = BRANCH ;
			SendMsgBranchPoss = FALSE ;
		}
	  ;
    break;}
case 77:
#line 729 "xp-t.bison"
{ 	char *sptr ;	
		strcpy(SendEP,yyvsp[0]) ; 
	    	/* discard all the CHARM++ `=>' stuff */
		sptr = Mystrstr(OutBuf,yyvsp[0]) ;
		if ( sptr != NULL ) 
			*sptr = '\0' ;
		sptr = Mystrstr(OutBuf,SendChare) ;
		if ( sptr != NULL ) 
			*sptr = '\0' ;
		else 
			fprintf(stderr,"TRANSLATOR ERROR : %s, line %d : couldnt discard => etc.\n",CurrentFileName,CurrentLine) ;
		strcpy(prevtoken,"") ;
		FLUSHBUF() ;
	  ;
    break;}
case 78:
#line 745 "xp-t.bison"
{	strcpy(SendChare,yyvsp[-4]) ;
		SendType = BROADCAST ;
	  ;
    break;}
case 79:
#line 749 "xp-t.bison"
{ 	char *sptr ;
		strcpy(SendEP,yyvsp[0]) ; 
	    	/* discard all the CHARM++ `=>' stuff */
		sptr = Mystrstr(OutBuf,yyvsp[0]) ;
		if ( sptr != NULL ) 
			*sptr = '\0' ;
		sptr = Mystrstr(OutBuf,SendChare) ;
		if ( sptr != NULL ) 
			*sptr = '\0' ;
		else 
			fprintf(stderr,"TRANSLATOR ERROR : %s, line %d : couldnt discard => etc.\n",CurrentFileName,CurrentLine) ;
		strcpy(prevtoken,"") ;
		FLUSHBUF() ;
	  ;
    break;}
case 80:
#line 772 "xp-t.bison"
{ yyval = Concat3(yyvsp[-2],"(",")") ; ;
    break;}
case 81:
#line 774 "xp-t.bison"
{ yyval = Concat3(yyvsp[-2],"(",")") ; ;
    break;}
case 82:
#line 776 "xp-t.bison"
{ char *temp ;
	    temp = Concat3(yyvsp[-3],"(",yyvsp[-1]) ;
	    yyval = Concat2(temp,")") ;
	  ;
    break;}
case 83:
#line 781 "xp-t.bison"
{ char *temp ;
	    temp = Concat3(yyvsp[-3],"(",yyvsp[-1]) ;
	    yyval = Concat2(temp,")") ;
	  ;
    break;}
case 84:
#line 786 "xp-t.bison"
{ char *temp ;
	    temp = Concat3(yyvsp[-3],"(",yyvsp[-1]) ;
	    yyval = Concat2(temp,")") ;
	  ;
    break;}
case 85:
#line 809 "xp-t.bison"
{	char *str ;	
		if ( IsMonoCall && strcmp(yyvsp[0],"Update")==0 ) {
		        str=Mystrstr(OutBuf,"Update") ;
			*str = '\0' ;
			strcat(OutBuf,"_CK_Update") ;
		}
		else 
			yyval = yyvsp[0];
		IsMonoCall = FALSE ;
		IsAccCall = FALSE ;
	;
    break;}
case 90:
#line 828 "xp-t.bison"
{ yyval = yyvsp[0] ; /* FLUSHBUF() ;*/ ;
    break;}
case 91:
#line 830 "xp-t.bison"
{ yyval = Concat3(yyvsp[-2],",",yyvsp[0]) ; 
	    strcpy(LastArg,yyvsp[0]) ;
	  ;
    break;}
case 92:
#line 837 "xp-t.bison"
{ 	yyval = yyvsp[0] ; 
		strcpy(CurrentSharedHandle,"_CK_NOTACCHANDLE") ; 
		SendMsgBranchPoss = FALSE ;
	;
    break;}
case 93:
#line 841 "xp-t.bison"
{ yyval = Concat2("++",yyvsp[0]) ; ;
    break;}
case 94:
#line 842 "xp-t.bison"
{ yyval = Concat2("--",yyvsp[0]) ; ;
    break;}
case 95:
#line 844 "xp-t.bison"
{  if ( ! FoundChareEPPair ) 
		yyval = Concat2(CurrentAsterisk,yyvsp[0]) ; 
             else
		yyval = yyvsp[0] ;
	     FoundChareEPPair = 0 ;
	  ;
    break;}
case 96:
#line 850 "xp-t.bison"
{ yyval = Concat2("-",yyvsp[0]) ; ;
    break;}
case 97:
#line 851 "xp-t.bison"
{ yyval = Concat2("+",yyvsp[0]) ; ;
    break;}
case 98:
#line 852 "xp-t.bison"
{ yyval = Concat2("~",yyvsp[0]) ; ;
    break;}
case 99:
#line 853 "xp-t.bison"
{ yyval = Concat2("!",yyvsp[0]) ; ;
    break;}
case 100:
#line 854 "xp-t.bison"
{ yyval = Concat2("sizeof",yyvsp[0]) ; ;
    break;}
case 101:
#line 855 "xp-t.bison"
{ char *temp = Concat3("sizeof","(",yyvsp[-1]) ;
					  yyval = Concat2(temp,")") ;
					;
    break;}
case 102:
#line 858 "xp-t.bison"
{ yyval = yyvsp[0] ; ;
    break;}
case 103:
#line 882 "xp-t.bison"
{ 	OutputNewChareMsg(yyvsp[-2], yyvsp[0], NULL) ; ;
    break;}
case 104:
#line 885 "xp-t.bison"
{ 	OutputNewChareMsg(yyvsp[-2], yyvsp[0], yyvsp[-5]) ; ;
    break;}
case 105:
#line 889 "xp-t.bison"
{ 	OutputNewChareMsg(NewType, yyvsp[0], NULL) ;;
    break;}
case 106:
#line 892 "xp-t.bison"
{ 	OutputNewChareMsg(NewType, yyvsp[0], yyvsp[-2]) ; ;
    break;}
case 107:
#line 898 "xp-t.bison"
{ NewOpType = NEW ; ;
    break;}
case 108:
#line 899 "xp-t.bison"
{ NewOpType = NEWCHARE ; ;
    break;}
case 109:
#line 900 "xp-t.bison"
{ NewOpType = NEWGROUP ; ;
    break;}
case 110:
#line 901 "xp-t.bison"
{ NewOpType = NEW ; ;
    break;}
case 112:
#line 910 "xp-t.bison"
{ 	NewType = (char *)malloc(strlen(yyvsp[-2])+1) ;
		strcpy(NewType, yyvsp[-2]) ;
		yyval = yyvsp[0] ;
	  ;
    break;}
case 113:
#line 922 "xp-t.bison"
{ FoundDeclarator = FALSE; ;
    break;}
case 114:
#line 924 "xp-t.bison"
{ FoundDeclarator = TRUE; ;
    break;}
case 115:
#line 926 "xp-t.bison"
{ FoundDeclarator = TRUE; ;
    break;}
case 116:
#line 928 "xp-t.bison"
{ FoundDeclarator = TRUE; ;
    break;}
case 120:
#line 938 "xp-t.bison"
{ yyval = NULL ; ;
    break;}
case 121:
#line 939 "xp-t.bison"
{ yyval = NULL ; ;
    break;}
case 122:
#line 940 "xp-t.bison"
{ yyval = yyvsp[-1] ; ;
    break;}
case 123:
#line 944 "xp-t.bison"
{ yyval=yyvsp[0] ; ;
    break;}
case 124:
#line 946 "xp-t.bison"
{ char *temp = Concat3("(",yyvsp[-2],")") ; 
	    yyval = Concat2(temp,yyvsp[0]) ;
	  ;
    break;}
case 125:
#line 954 "xp-t.bison"
{ yyval=yyvsp[0] ; ;
    break;}
case 131:
#line 970 "xp-t.bison"
{ yyval=yyvsp[0] ; ;
    break;}
case 132:
#line 972 "xp-t.bison"
{ yyval = Concat3(yyvsp[-2],".*",yyvsp[0]) ; ;
    break;}
case 133:
#line 974 "xp-t.bison"
{ yyval = Concat3(yyvsp[-2],"->*",yyvsp[0]) ; ;
    break;}
case 134:
#line 978 "xp-t.bison"
{ yyval=yyvsp[0] ; ;
    break;}
case 135:
#line 980 "xp-t.bison"
{ yyval = Concat3(yyvsp[-2],"*",yyvsp[0]) ; ;
    break;}
case 136:
#line 982 "xp-t.bison"
{ yyval = Concat3(yyvsp[-2],"/",yyvsp[0]) ; ;
    break;}
case 137:
#line 984 "xp-t.bison"
{ yyval = Concat3(yyvsp[-2],"%",yyvsp[0]) ; ;
    break;}
case 138:
#line 988 "xp-t.bison"
{ yyval=yyvsp[0] ; ;
    break;}
case 139:
#line 990 "xp-t.bison"
{ yyval = Concat3(yyvsp[-2],"+",yyvsp[0]) ; ;
    break;}
case 140:
#line 992 "xp-t.bison"
{ yyval = Concat3(yyvsp[-2],"-",yyvsp[0]) ; ;
    break;}
case 141:
#line 996 "xp-t.bison"
{ yyval=yyvsp[0] ; ;
    break;}
case 142:
#line 998 "xp-t.bison"
{ yyval = Concat3(yyvsp[-2],"<<",yyvsp[0]) ; ;
    break;}
case 143:
#line 1000 "xp-t.bison"
{ yyval = Concat3(yyvsp[-2],">>",yyvsp[0]) ; ;
    break;}
case 144:
#line 1004 "xp-t.bison"
{ yyval=yyvsp[0] ; ;
    break;}
case 145:
#line 1006 "xp-t.bison"
{ yyval = Concat3(yyvsp[-2],"<",yyvsp[0]) ; ;
    break;}
case 146:
#line 1008 "xp-t.bison"
{ yyval = Concat3(yyvsp[-2],">",yyvsp[0]) ; ;
    break;}
case 147:
#line 1010 "xp-t.bison"
{ yyval = Concat3(yyvsp[-2],"<=",yyvsp[0]) ; ;
    break;}
case 148:
#line 1012 "xp-t.bison"
{ yyval = Concat3(yyvsp[-2],">=",yyvsp[0]) ; ;
    break;}
case 149:
#line 1016 "xp-t.bison"
{ yyval=yyvsp[0] ; ;
    break;}
case 150:
#line 1018 "xp-t.bison"
{ yyval = Concat3(yyvsp[-2],"==",yyvsp[0]) ; ;
    break;}
case 151:
#line 1020 "xp-t.bison"
{ yyval = Concat3(yyvsp[-2],"!=",yyvsp[0]) ; ;
    break;}
case 152:
#line 1024 "xp-t.bison"
{ yyval=yyvsp[0] ; ;
    break;}
case 153:
#line 1026 "xp-t.bison"
{ yyval = Concat3(yyvsp[-2],"&",yyvsp[0]) ; ;
    break;}
case 154:
#line 1030 "xp-t.bison"
{ yyval=yyvsp[0] ; ;
    break;}
case 155:
#line 1032 "xp-t.bison"
{ yyval = Concat3(yyvsp[-2],"^",yyvsp[0]) ; ;
    break;}
case 156:
#line 1036 "xp-t.bison"
{ yyval=yyvsp[0] ; ;
    break;}
case 157:
#line 1038 "xp-t.bison"
{ yyval = Concat3(yyvsp[-2],"|",yyvsp[0]) ; ;
    break;}
case 158:
#line 1042 "xp-t.bison"
{ yyval=yyvsp[0] ; ;
    break;}
case 159:
#line 1044 "xp-t.bison"
{ yyval = Concat3(yyvsp[-2],"&&",yyvsp[0]) ; ;
    break;}
case 160:
#line 1048 "xp-t.bison"
{ yyval=yyvsp[0] ; ;
    break;}
case 161:
#line 1050 "xp-t.bison"
{ yyval = Concat3(yyvsp[-2],"||",yyvsp[0]) ; ;
    break;}
case 162:
#line 1054 "xp-t.bison"
{ yyval=yyvsp[0] ; ;
    break;}
case 163:
#line 1058 "xp-t.bison"
{ char *temp = Concat3(yyvsp[-4],"?",yyvsp[-2]) ;
	    yyval = Concat3(temp,":",yyvsp[0]) ;
	  ;
    break;}
case 164:
#line 1064 "xp-t.bison"
{ yyval=yyvsp[0] ; ;
    break;}
case 165:
#line 1066 "xp-t.bison"
{ yyval = Concat3(yyvsp[-2],CurrentAssOp,yyvsp[0]) ; ;
    break;}
case 166:
#line 1070 "xp-t.bison"
{ yyval = yyvsp[0] ; strcpy(CurrentAssOp,"=") ;;
    break;}
case 167:
#line 1071 "xp-t.bison"
{ yyval = yyvsp[0] ; strcpy(CurrentAssOp,"*=") ;;
    break;}
case 168:
#line 1072 "xp-t.bison"
{ yyval = yyvsp[0] ; strcpy(CurrentAssOp,"/=") ;;
    break;}
case 169:
#line 1073 "xp-t.bison"
{ yyval = yyvsp[0] ; strcpy(CurrentAssOp,"%=") ;;
    break;}
case 170:
#line 1074 "xp-t.bison"
{ yyval = yyvsp[0] ; strcpy(CurrentAssOp,"+=") ;;
    break;}
case 171:
#line 1075 "xp-t.bison"
{ yyval = yyvsp[0] ; strcpy(CurrentAssOp,"-=") ;;
    break;}
case 172:
#line 1076 "xp-t.bison"
{ yyval = yyvsp[0] ; strcpy(CurrentAssOp,"<<=") ;;
    break;}
case 173:
#line 1077 "xp-t.bison"
{ yyval = yyvsp[0] ; strcpy(CurrentAssOp,">>=") ;;
    break;}
case 174:
#line 1078 "xp-t.bison"
{ yyval = yyvsp[0] ; strcpy(CurrentAssOp,"&=") ;;
    break;}
case 175:
#line 1079 "xp-t.bison"
{ yyval = yyvsp[0] ; strcpy(CurrentAssOp,"^=") ;;
    break;}
case 176:
#line 1080 "xp-t.bison"
{ yyval = yyvsp[0] ; strcpy(CurrentAssOp,"|=") ;;
    break;}
case 177:
#line 1084 "xp-t.bison"
{ yyval=yyvsp[0] ; ;
    break;}
case 178:
#line 1086 "xp-t.bison"
{ yyval = Concat3(yyvsp[-2],",",yyvsp[0]) ; ;
    break;}
case 182:
#line 1116 "xp-t.bison"
{ 	
		strcpy(CurrentDeclType,"") ;
		FoundReadOnly = FALSE ;
	  ;
    break;}
case 183:
#line 1122 "xp-t.bison"
{ 	
		strcpy(CurrentDeclType,"") ;
		FoundReadOnly = FALSE ;
	  ;
    break;}
case 184:
#line 1126 "xp-t.bison"
{ /* this is constraint error, as it
                                        includes a storage class!?!*/ ;
    break;}
case 187:
#line 1130 "xp-t.bison"
{ FLUSHBUF() ; ;
    break;}
case 188:
#line 1154 "xp-t.bison"
{;
    break;}
case 190:
#line 1155 "xp-t.bison"
{;
    break;}
case 192:
#line 1156 "xp-t.bison"
{;
    break;}
case 197:
#line 1170 "xp-t.bison"
{	strcpy(CurrentDeclType,yyvsp[-1]) ;
		if ( CurrentStorage == TYPEDEF ) {
			InsertSymTable(yyvsp[0]) ;
		}
		else if ( CurrentStorage == EXTERN && CurrentScope == 0 
			  && FoundParms )
			InsertFunctionTable(yyvsp[0],FALSE) ;
		else if ( FoundReadOnly && CurrentStorage != EXTERN ) {
			if ( FoundInMsgTable(yyvsp[-1]) != -1 )
                        	CurrentAggType = READMSG ;
                	else 
                        	CurrentAggType = READONLY ;
                	InsertObjTable(yyvsp[0]) ;
		}
		CurrentStorage = -1 ;
	;
    break;}
case 199:
#line 1188 "xp-t.bison"
{	strcpy(CurrentDeclType,yyvsp[-1]) ; 
		if ( FoundReadOnly && CurrentStorage != EXTERN ) {
			if ( FoundInMsgTable(yyvsp[-1]) != -1 )
                        	CurrentAggType = READMSG ;
                	else
                        	CurrentAggType = READONLY ;
                	InsertObjTable(yyvsp[0]) ;
		}
		CurrentStorage = -1 ;
	  ;
    break;}
case 201:
#line 1199 "xp-t.bison"
{strcpy(CurrentDeclType,yyvsp[-1]) ;;
    break;}
case 203:
#line 1201 "xp-t.bison"
{	strcpy(CurrentDeclType,yyvsp[-1]) ;
	  	if ( strcmp(yyvsp[-1],"table")==0 ) {
			CurrentAggType = DTABLE ;
                	InsertObjTable(yyvsp[0]) ;
		}
	  ;
    break;}
case 205:
#line 1209 "xp-t.bison"
{strcpy(CurrentDeclType,yyvsp[-1]) ;;
    break;}
case 207:
#line 1211 "xp-t.bison"
{	if ( FoundReadOnly && CurrentStorage != EXTERN ) {
			if ( FoundInMsgTable(CurrentDeclType) != -1 )
                        	CurrentAggType = READMSG ;
                	else
                        	CurrentAggType = READONLY ;
                	InsertObjTable(yyvsp[0]) ;
		}
	  	else if ( strcmp(CurrentDeclType,"table")==0 ) {
                        CurrentAggType = DTABLE ;
                	InsertObjTable(yyvsp[0]) ;
		}
		CurrentStorage = -1 ;
	  ;
    break;}
case 237:
#line 1325 "xp-t.bison"
{yyval=yyvsp[0];;
    break;}
case 238:
#line 1327 "xp-t.bison"
{ yyval = (char *)malloc(sizeof(char)*2) ;
	    strcpy(yyval,"") ; 
          ;
    break;}
case 239:
#line 1330 "xp-t.bison"
{yyval=yyvsp[0];;
    break;}
case 240:
#line 1334 "xp-t.bison"
{yyval = yyvsp[0];;
    break;}
case 243:
#line 1337 "xp-t.bison"
{yyval = yyvsp[0];;
    break;}
case 253:
#line 1360 "xp-t.bison"
{ if ( CurrentScope > 0 ) 
		CharmError("readonly variables allowed only at file scope") ;
	  else 
		FoundReadOnly = TRUE ;
	;
    break;}
case 254:
#line 1368 "xp-t.bison"
{ yyval = yyvsp[0] ; ;
    break;}
case 255:
#line 1369 "xp-t.bison"
{ yyval = yyvsp[-1] ; ;
    break;}
case 256:
#line 1370 "xp-t.bison"
{ yyval = yyvsp[-1] ; ;
    break;}
case 257:
#line 1371 "xp-t.bison"
{ yyval = yyvsp[-1] ; ;
    break;}
case 258:
#line 1372 "xp-t.bison"
{ yyval = yyvsp[-1] ; ;
    break;}
case 259:
#line 1377 "xp-t.bison"
{ yyval = yyvsp[-1] ; ;
    break;}
case 260:
#line 1378 "xp-t.bison"
{ yyval = yyvsp[-1] ; ;
    break;}
case 261:
#line 1379 "xp-t.bison"
{ yyval = yyvsp[-1] ; ;
    break;}
case 262:
#line 1380 "xp-t.bison"
{ yyval = yyvsp[-1] ; ;
    break;}
case 263:
#line 1381 "xp-t.bison"
{ yyval = yyvsp[-1] ; ;
    break;}
case 275:
#line 1405 "xp-t.bison"
{ yyval = yyvsp[0]; ;
    break;}
case 276:
#line 1406 "xp-t.bison"
{ yyval = yyvsp[0]; ;
    break;}
case 277:
#line 1407 "xp-t.bison"
{ yyval = yyvsp[-1]; ;
    break;}
case 278:
#line 1408 "xp-t.bison"
{ yyval = yyvsp[-1]; ;
    break;}
case 279:
#line 1409 "xp-t.bison"
{ yyval = yyvsp[-1]; ;
    break;}
case 280:
#line 1410 "xp-t.bison"
{ yyval = yyvsp[-1]; ;
    break;}
case 281:
#line 1414 "xp-t.bison"
{ yyval = yyvsp[0] ; ;
    break;}
case 282:
#line 1415 "xp-t.bison"
{ yyval = yyvsp[0] ; ;
    break;}
case 283:
#line 1417 "xp-t.bison"
{ yyval = yyvsp[-1] ; ;
    break;}
case 284:
#line 1418 "xp-t.bison"
{ yyval = yyvsp[-1] ; ;
    break;}
case 285:
#line 1420 "xp-t.bison"
{ yyval = yyvsp[-1] ; ;
    break;}
case 286:
#line 1449 "xp-t.bison"
{ CurrentStorage = EXTERN ; ;
    break;}
case 287:
#line 1450 "xp-t.bison"
{ CurrentStorage = TYPEDEF ; ;
    break;}
case 288:
#line 1451 "xp-t.bison"
{ CurrentStorage = STATIC ; ;
    break;}
case 289:
#line 1452 "xp-t.bison"
{ CurrentStorage = AUTO ; ;
    break;}
case 290:
#line 1453 "xp-t.bison"
{ CurrentStorage = REGISTER ; ;
    break;}
case 291:
#line 1454 "xp-t.bison"
{ CurrentStorage = FRIEND ; ;
    break;}
case 292:
#line 1455 "xp-t.bison"
{ CurrentStorage = OVERLOAD ; ;
    break;}
case 293:
#line 1456 "xp-t.bison"
{ CurrentStorage = INLINE ; 
						  CurrentFnIsInline = TRUE ;
						;
    break;}
case 294:
#line 1459 "xp-t.bison"
{ CurrentStorage = INLINE ; ;
    break;}
case 295:
#line 1460 "xp-t.bison"
{ CurrentStorage = VIRTUAL ; ;
    break;}
case 296:
#line 1464 "xp-t.bison"
{ yyval = (char *)malloc(4) ; 
			  strcpy(yyval,"int") ;	   ;
    break;}
case 297:
#line 1466 "xp-t.bison"
{ yyval = (char *)malloc(5) ; 
			  strcpy(yyval,"char") ;	   ;
    break;}
case 298:
#line 1468 "xp-t.bison"
{ yyval = (char *)malloc(6) ; 
			  strcpy(yyval,"short") ;	   ;
    break;}
case 299:
#line 1470 "xp-t.bison"
{ yyval = (char *)malloc(5) ; 
			  strcpy(yyval,"long") ;	   ;
    break;}
case 300:
#line 1472 "xp-t.bison"
{ yyval = (char *)malloc(6) ; 
			  strcpy(yyval,"float") ;	   ;
    break;}
case 301:
#line 1474 "xp-t.bison"
{ yyval = (char *)malloc(10) ; 
			  strcpy(yyval,"ptrdiff_t") ; ;
    break;}
case 302:
#line 1476 "xp-t.bison"
{ yyval = (char *)malloc(10) ; 
			  strcpy(yyval,"wchar_t") ; ;
    break;}
case 303:
#line 1478 "xp-t.bison"
{ yyval = (char *)malloc(10) ; 
			  strcpy(yyval,"__wchar_t") ; ;
    break;}
case 304:
#line 1480 "xp-t.bison"
{ yyval = (char *)malloc(7) ; 
			  strcpy(yyval,"double") ;	   ;
    break;}
case 305:
#line 1482 "xp-t.bison"
{ yyval = (char *)malloc(7) ; 
			  strcpy(yyval,"signed") ;	   ;
    break;}
case 306:
#line 1484 "xp-t.bison"
{ yyval = (char *)malloc(9) ; 
			  strcpy(yyval,"unsigned") ;  ;
    break;}
case 307:
#line 1486 "xp-t.bison"
{ yyval = (char *)malloc(5) ; 
			  strcpy(yyval,"void") ;	   ;
    break;}
case 312:
#line 1515 "xp-t.bison"
{  	if ( CurrentAggType == CHARE ) {
			FLUSHBUF() ;	
			if ( numparents == 0  )
                        	fprintf(outfile," : public _CK_Object ") ;
		}
		/*
	 	else if ( CurrentAggType == BRANCHED ) {
			FLUSHBUF() ;	
			if ( numparents == 0  )
                        	fprintf(outfile," : public _CK_BOC ") ;
		}
		*/
	 	else if ( CurrentAggType == ACCUMULATOR ) {
			FLUSHBUF() ;	
			if ( numparents == 0  )
                        	fprintf(outfile," : public _CK_Accumulator ") ;
		}
	 	else if ( CurrentAggType == MONOTONIC ) {
			FLUSHBUF() ;	
			if ( numparents == 0  )
                        	fprintf(outfile," : public _CK_Monotonic ") ;
		}
	 	else if ( CurrentAggType == MESSAGE ) {
			FLUSHBUF() ;	
			if ( numparents == 0  )
                        	fprintf(outfile," : public comm_object ") ;
		}
	;
    break;}
case 313:
#line 1544 "xp-t.bison"
{	int num, i ;	
		ChareInfo *chare ;
		char *mymsg ;
		char *myacc ;

		FLUSHBUF() ;

		
		if ( CurrentAggType != CHARE )
			InsertObjTable(yyvsp[-3]) ;
		else {
			/* First find out if this is a normal chare or a BOC */
			if ( numparents > 0 ) {
			    /* Find if any parent is "groupmember" or a BOC */
			    for ( i=0; i<numparents; i++ ) {
				if (strcmp(ParentArray[i], "groupmember")==0) {
				    CurrentAggType = BRANCHED ;
				    break ;
				}
				else if ( FoundInChareTable(BOCTable,boccount+1,ParentArray[i]) != -1 ) {
				    CurrentAggType = BRANCHED ;
				    break ;
				}
			    }
			}

			InsertObjTable(yyvsp[-3]) ;


			if ( shouldprint ) fprintf(outfile,"%s",prevtoken) ;
			strcpy(prevtoken,"") ;

/***** Sanjeev 10/10/95
			if ( numparents == 0 ) { * got to be a chare *
			    fprintf(outfile,"\n\tpublic: %s(CHARE_BLOCK *c) : _CK_Object(c) {}\n",CurrentChare) ;   
			}
			else {
			    fprintf(outfile,"\n\tpublic: %s(CHARE_BLOCK *c) : ",CurrentChare) ;   
			    for ( i=0; i<numparents-1; i++ ) 
				fprintf(outfile,"%s(c), ",ParentArray[i]) ;
			    fprintf(outfile,"%s(c) {}\n",
						   ParentArray[numparents-1]) ;
			}
			if ( strcmp(CurrentChare,"main") != 0 )
			    fprintf(outfile,"\n\t        %s() {}\n",CurrentChare) ;   
			fprintf(outfile,"private:\n") ;
			fprintf(outfile,"#line %d \"%s\"\n",CurrentLine,
							CurrentFileName) ;
******/

			 
			if ( CurrentAggType == CHARE )
		        	chare = ChareTable[FoundInChareTable(ChareTable,charecount+1,CurrentChare)] ;
			else
		        	chare = BOCTable[FoundInChareTable(BOCTable,boccount+1,CurrentChare)] ;

			for ( i=0; i<numparents; i++ ) {
			    if ( CurrentAggType == CHARE )
				InsertParent(chare,ParentArray[i],ChareTable,charecount+1) ;
			    else
				InsertParent(chare,ParentArray[i],BOCTable,boccount+1) ;
			}
			AddInheritedEps(chare) ;
		}
		numparents = 0 ; /* so that sequential class derivations dont
				    affect us */

		CurrentAccess = PRIVATE ;
	;
    break;}
case 314:
#line 1613 "xp-t.bison"
{ FillPermanentAggTable(yyvsp[-5]) ; ;
    break;}
case 315:
#line 1615 "xp-t.bison"
{   FLUSHBUF() ;	
	    if ( CurrentAggType == ACCUMULATOR || CurrentAggType == MONOTONIC )
	    {   if ( !FilledAccMsg ) 
	    	    CharmError("Unable to detect message in accumulator/monotonic class") ;
	        else {
	    	    fprintf(outfile,"\npublic:void * _CK_GetMsgPtr() {\n");
	    	    fprintf(outfile,"\t\treturn((void *)%s) ;\n\t}\n",
							     CurrentAcc->msg) ;
	    	    if ( CurrentAggType == ACCUMULATOR ) {
	    	    	fprintf(outfile,"\n       void _CK_Combine(void *msg) {\n");
	    	    	fprintf(outfile,"\t\tCombine((%s *)msg) ;\n\t}\n",
							 CurrentAcc->msgtype) ;
	    	    }
	    	    else { /*  MONOTONIC  */
	    	    	fprintf(outfile,"\n       void _CK_SysUpdate(void *msg) {\n");
	    	    	fprintf(outfile,"\t\tUpdate((%s *)msg) ;\n\t}\n",
							 CurrentAcc->msgtype) ;
	    	    	fprintf(outfile,"\n       %s *MonoValue() {\n",
							 CurrentAcc->msgtype) ;
	    	    	fprintf(outfile,"\t\treturn( (%s *)(::MonoValue(_CK_MyId)) ) ;\n\t}\n",CurrentAcc->msgtype) ;


	    	    	fprintf(outfile,"\n      int _CK_Update(%s *msg) {\n",
							 CurrentAcc->msgtype) ;
	    	    	fprintf(outfile,"\t\tint v ;\n") ;
			/* Locking not needed anymore 
			fprintf(outfile,"\t\t_CK_9LockMonoDataArea(GetBocDataPtr(_CK_MyId)) ;\n") ;
			*/
			fprintf(outfile,"\t\tv = Update(msg) ;\n") ;
			/* Locking not needed anymore 
			fprintf(outfile,"\t\t_CK_9UnlockMonoDataArea(GetBocDataPtr(_CK_MyId)) ;\n") ;
			*/
			fprintf(outfile,"\t\tif ( v )\n") ;
	    	    	fprintf(outfile,"\t\t\t_CK_BroadcastMono((void *)%s,_CK_MyId) ;\n",CurrentAcc->msg) ;
	    	    	fprintf(outfile,"\t\treturn v ;\n\t}\n") ;

	    	    }
	    	    fprintf(outfile,"\n#line %d \"%s\"\n",CurrentLine,CurrentFileName) ;
	        }
	    }
	    else if ( CurrentAggType == MESSAGE && 
				     MessageTable[TotalMsgs-1].numvarsize!=0 ) 
	    { /* Output the pack and unpack and alloc functions for varSize */
		int i ;

		MsgStruct *thismsg = &(MessageTable[TotalMsgs-1]) ;
		VarSizeStruct *vs = thismsg->varsizearray ;

		fprintf(outfile,"\nvoid *pack(int *length) \n\t{\n") ;

	      /*fprintf(outfile,"\n\t*length = _CK_3GetSizeOfMsg(this) ;\n");
		the length (size) parm is no longer used in the PACK macro
		in common/headers/communication.h : Sanjeev, 1 Sep 94 */
		fprintf(outfile,"\n\t*length = 0 ;\n") ;

		for ( i=0; i<thismsg->numvarsize; i++ ) 
			fprintf(outfile,"\n\t%s = (%s *) ((char *)%s - ((char *)&(%s)));\n",vs[i].name,vs[i].type,vs[i].name,vs[i].name) ;
		fprintf(outfile,"\treturn((void *)this) ;\n}\n") ;

		fprintf(outfile,"\nvoid unpack() \n\t{\n") ;
		for ( i=0; i<thismsg->numvarsize; i++ ) 
			fprintf(outfile,"\n\t%s = (%s *) ((char *)(&%s) + (size_t)(%s));\n",vs[i].name,vs[i].type,vs[i].name,vs[i].name) ;
		fprintf(outfile,"}\n") ;

		/* Output the operator delete */
		fprintf(outfile,"\nvoid operator delete(void *msg) {\n") ;
		fprintf(outfile,"\tCkFreeMsg(msg) ;\n}\n") ;

	    	fprintf(outfile,"#line %d \"%s\"\n",CurrentLine,CurrentFileName) ;
	    }
		
		
	    CurrentAggType = -1 ; 
	    CurrentAccess = -1 ; 
	    strcpy(CurrentAggName,"") ;
	    strcpy(CurrentChare,"_CK_NOTACHARE") ;
	    CurrentCharePtr = NULL ;
	    InsideChareCode = 0 ;
	;
    break;}
case 317:
#line 1713 "xp-t.bison"
{	strcpy(CurrentAggName,yyvsp[0]);
	   	InsertSymTable(yyvsp[0]) ;
	   	yyval = yyvsp[0] ;
	  ;
    break;}
case 318:
#line 1718 "xp-t.bison"
{	if ( AddedScope > 0 ) {
			PopStack() ;	
			AddedScope = 0 ;
		}
		strcpy(CurrentAggName,yyvsp[0]);
	     	InsertSymTable(yyvsp[0]) ;
	     	yyval = yyvsp[0] ;
	  ;
    break;}
case 319:
#line 1727 "xp-t.bison"
{	strcpy(CurrentAggName,yyvsp[0]);
	    	InsertSymTable(yyvsp[0]) ;
	     	yyval = yyvsp[0] ;
	  ;
    break;}
case 320:
#line 1732 "xp-t.bison"
{	if ( AddedScope > 0 ) {
			PopStack() ;	
			AddedScope = 0 ;
		}
	  	strcpy(CurrentAggName,yyvsp[0]);
	     	InsertSymTable(yyvsp[0]) ;
	     	yyval = yyvsp[0] ;
	  ;
    break;}
case 323:
#line 1745 "xp-t.bison"
{FLUSHBUF(); SyntaxError("class defn header");;
    break;}
case 324:
#line 1750 "xp-t.bison"
{	ParentArray[numparents] = (char *)malloc(strlen(yyvsp[0])+1) ;
		strcpy(ParentArray[numparents],yyvsp[0]) ;
		numparents++ ;
	;
    break;}
case 325:
#line 1755 "xp-t.bison"
{	ParentArray[numparents] = (char *)malloc(strlen(yyvsp[0])+1) ;
		strcpy(ParentArray[numparents],yyvsp[0]) ;
		numparents++ ;
	;
    break;}
case 326:
#line 1762 "xp-t.bison"
{yyval=yyvsp[0];;
    break;}
case 327:
#line 1763 "xp-t.bison"
{yyval=yyvsp[0];;
    break;}
case 328:
#line 1764 "xp-t.bison"
{yyval=yyvsp[0];;
    break;}
case 333:
#line 1778 "xp-t.bison"
{ CurrentAccess = PUBLIC ; ;
    break;}
case 334:
#line 1779 "xp-t.bison"
{ CurrentAccess = PRIVATE ; ;
    break;}
case 335:
#line 1780 "xp-t.bison"
{ CurrentAccess = PROTECTED ; ;
    break;}
case 336:
#line 1781 "xp-t.bison"
{ CurrentAccess = ENTRY ; ;
    break;}
case 337:
#line 1785 "xp-t.bison"
{ CurrentAggType = STRUCT ; ;
    break;}
case 338:
#line 1786 "xp-t.bison"
{ CurrentAggType = UNION ; ;
    break;}
case 339:
#line 1787 "xp-t.bison"
{ CurrentAggType = CLASS ; ;
    break;}
case 340:
#line 1788 "xp-t.bison"
{ CurrentAggType = CHARE ; ;
    break;}
case 341:
#line 1790 "xp-t.bison"
{ CurrentAggType = MESSAGE ; ;
    break;}
case 342:
#line 1791 "xp-t.bison"
{ CurrentAggType = ACCUMULATOR ; 
				  FilledAccMsg=FALSE; ;
    break;}
case 343:
#line 1793 "xp-t.bison"
{ CurrentAggType = MONOTONIC ; 
				  FilledAccMsg=FALSE; ;
    break;}
case 345:
#line 1806 "xp-t.bison"
{ FLUSHBUF() ;;
    break;}
case 346:
#line 1810 "xp-t.bison"
{ strcpy(CurrentDeclType,"") ;;
    break;}
case 347:
#line 1811 "xp-t.bison"
{ strcpy(CurrentDeclType,"") ;;
    break;}
case 362:
#line 1854 "xp-t.bison"
{strcpy(CurrentDeclType,yyvsp[-2]) ;	
	   if (FoundParms&&(CurrentAggType==CHARE||CurrentAggType==BRANCHED)) {
		if ( CurrentAccess == PRIVATE || CurrentAccess == PUBLIC )
		    ProcessFn(yyvsp[-1]) ;
	   }
	  ;
    break;}
case 363:
#line 1861 "xp-t.bison"
{int ind ;
	   if ( FoundParms ) {
	     if (CurrentAggType==CHARE || CurrentAggType==BRANCHED) {
		if ( CurrentAccess == ENTRY )
		    ProcessEP(yyvsp[0],FALSE);
		else if ( CurrentAccess == PRIVATE||CurrentAccess == PUBLIC ) 
	   	    ProcessFn(yyvsp[0]) ;
	     }
	     else if ( CurrentAggType == MESSAGE ) {
		if ( strcmp(yyvsp[0],"pack")==0 || strcmp(yyvsp[0],"unpack")==0 ) {
		    if ( (ind=FoundInMsgTable(CurrentAggName)) != -1 )
			MessageTable[ind].pack = TRUE ;	
		    else 
		    	CharmError("TRANSLATOR : did not find message type in message table") ;
		}			
		else 
			CharmError("Messages are allowed to have only pack or unpack functions") ;
	     }
	     FoundParms = FALSE ;
	   }
	   else if ( CurrentAggType == MESSAGE && FoundVarSize ) {
		char *varname = Mystrstr(OutBuf,yyvsp[0]) ;
		*varname = '*' ;
		*(varname+1) = '\0' ;
		strcat(OutBuf,yyvsp[0]) ;
	  	
		InsertVarSize(yyvsp[-1],yyvsp[0]) ;
		FoundVarSize = FALSE ;
	   } 
	  ;
    break;}
case 365:
#line 1894 "xp-t.bison"
{strcpy(CurrentDeclType,yyvsp[-2]) ;	
	   if (FoundParms&&(CurrentAggType==CHARE||CurrentAggType==BRANCHED)) {
		if ( CurrentAccess == PRIVATE || CurrentAccess == PUBLIC )
		    ProcessFn(yyvsp[-1]) ;
	   }
	  ;
    break;}
case 374:
#line 1921 "xp-t.bison"
{ if ( (CurrentAggType==ACCUMULATOR || CurrentAggType==MONOTONIC) 
		&& !FilledAccMsg ) 
	  {	CurrentAcc->msgtype = (char *)malloc((strlen(yyvsp[-2])+1)*sizeof(char));
        	strcpy(CurrentAcc->msgtype,yyvsp[-2]) ;	

		CurrentAcc->msg = (char *)malloc((strlen(yyvsp[-1])+1)*sizeof(char)) ;
        	strcpy(CurrentAcc->msg,yyvsp[-1]) ;	
		FilledAccMsg = TRUE ;
	  }
	  else if ( CurrentAggType == MESSAGE && FoundVarSize ) {
		char *varname = Mystrstr(OutBuf,yyvsp[-1]) ;
		*varname = '*' ;
		*(varname+1) = '\0' ;
		strcat(OutBuf,yyvsp[-1]) ;
	  	
		if ( SearchHandleTable(ChareHandleTable,ChareHandleTableSize,yyvsp[-1]) != -1 ) 
			InsertVarSize("ChareIDType",yyvsp[-1]) ;
		else
			InsertVarSize(yyvsp[-2],yyvsp[-1]) ;
		FoundVarSize = FALSE ;
	  } 
	  strcpy(CurrentDeclType,yyvsp[-2]) ;
	;
    break;}
case 375:
#line 1946 "xp-t.bison"
{	strcpy(CurrentDeclType,yyvsp[-2]) ;  	;
    break;}
case 376:
#line 1948 "xp-t.bison"
{     strcpy(CurrentDeclType,yyvsp[-2]) ;    ;
    break;}
case 377:
#line 1951 "xp-t.bison"
{     strcpy(CurrentDeclType,yyvsp[-2]) ;    
		if ( CurrentStorage == TYPEDEF ) {
			InsertSymTable(yyvsp[-1]) ;
			CurrentStorage = -1 ;
		}
	   	else if ( FoundParms ) {
			if (CurrentAggType==CHARE || CurrentAggType==BRANCHED){
				if ( CurrentAccess == ENTRY ) {
		    			ProcessEP(yyvsp[-1],FALSE);
				}
				else if ( CurrentAccess == PRIVATE || 
				  	  CurrentAccess == PUBLIC )
		    			ProcessFn(yyvsp[-1]) ;
			}	
	     		FoundParms = FALSE ;
	   	}
	  ;
    break;}
case 378:
#line 1969 "xp-t.bison"
{     strcpy(CurrentDeclType,yyvsp[-2]) ;    ;
    break;}
case 379:
#line 1971 "xp-t.bison"
{     strcpy(CurrentDeclType,yyvsp[-2]) ;    ;
    break;}
case 405:
#line 2063 "xp-t.bison"
{;
    break;}
case 408:
#line 2076 "xp-t.bison"
{;
    break;}
case 412:
#line 2096 "xp-t.bison"
{ InsertSymTable(yyvsp[0]) ; ;
    break;}
case 424:
#line 2142 "xp-t.bison"
{ 	EpMsg=yyvsp[-2]; 
	  ;
    break;}
case 458:
#line 2278 "xp-t.bison"
{ 	EpMsg=yyvsp[-1]; 
		strcpy(CurrentMsgParm,yyvsp[-1]) ;
	  ;
    break;}
case 459:
#line 2282 "xp-t.bison"
{ 	EpMsg=yyvsp[-1]; 
		strcpy(CurrentMsgParm,yyvsp[-1]) ;
	  ;
    break;}
case 465:
#line 2295 "xp-t.bison"
{ yyval = yyvsp[0] ; ;
    break;}
case 491:
#line 2352 "xp-t.bison"
{	if ( FoundAccFnDef ) {
			strcat(OutBuf,prevtoken) ;
			strcpy(prevtoken,"") ;
			FLUSHBUF() ;
			/* No locking needed anymore 
			fprintf(outfile,"\n_CK_9LockAccDataArea(GetBocDataPtr(_CK_MyId)) ;\n") ;
			*/
			fprintf(outfile,"\n#line %d \"%s\"\n",CurrentLine,CurrentFileName) ;
			AccFnScope = CurrentScope ;
			FoundAccFnDef = FALSE ;
		}
	;
    break;}
case 492:
#line 2365 "xp-t.bison"
{ 	if ( AccFnScope == CurrentScope+1 ) {
		/* +1 because the '}' has already been lexed at this point */
			FLUSHBUF() ;
			/* No locking needed anymore 
			fprintf(outfile,"\n_CK_9UnlockAccDataArea(GetBocDataPtr(_CK_MyId)) ;\n") ;
			*/
			fprintf(outfile,"\n#line %d \"%s\"\n",CurrentLine,CurrentFileName) ;
			AccFnScope = -1 ;
		}
	;
    break;}
case 493:
#line 2375 "xp-t.bison"
{ FLUSHBUF() ; ;
    break;}
case 494:
#line 2384 "xp-t.bison"
{ FLUSHBUF() ; ;
    break;}
case 495:
#line 2385 "xp-t.bison"
{ FLUSHBUF() ; ;
    break;}
case 497:
#line 2390 "xp-t.bison"
{ FLUSHBUF() ; ;
    break;}
case 502:
#line 2404 "xp-t.bison"
{ FLUSHBUF(); ;
    break;}
case 503:
#line 2405 "xp-t.bison"
{ FLUSHBUF(); 
			  SyntaxError("if/switch condition") ;;
    break;}
case 504:
#line 2410 "xp-t.bison"
{FLUSHBUF();;
    break;}
case 506:
#line 2411 "xp-t.bison"
{ FLUSHBUF() ; 
				  SyntaxError("while loop condition");;
    break;}
case 509:
#line 2416 "xp-t.bison"
{ FLUSHBUF() ; 
				    SyntaxError("do loop condition"); ;
    break;}
case 511:
#line 2421 "xp-t.bison"
{FLUSHBUF();;
    break;}
case 514:
#line 2425 "xp-t.bison"
{ FLUSHBUF() ; 
				  SyntaxError("for loop header") ; ;
    break;}
case 523:
#line 2451 "xp-t.bison"
{ FLUSHBUF() ; 
						  strcpy(CurrentAggName,"") ;
						;
    break;}
case 530:
#line 2463 "xp-t.bison"
{	
	    	if ( FoundLBrace ) 
			FoundLBrace = 0 ;
	  ;
    break;}
case 531:
#line 2468 "xp-t.bison"
{
	    	if ( FoundRBrace ) 
			FoundRBrace = 0 ;
	  ;
    break;}
case 536:
#line 2498 "xp-t.bison"
{ 	
		if ( !CurrentFnIsInline )
			InsertFunctionTable(yyvsp[0],TRUE) ; 
		else
			CurrentFnIsInline = FALSE ;
 	;
    break;}
case 539:
#line 2528 "xp-t.bison"
{	AddScope(yyvsp[0]) ;	;
    break;}
case 540:
#line 2530 "xp-t.bison"
{ RemoveScope(yyvsp[-2]) ; yyval = yyvsp[-2] ;
	    if (CurrentAggType==CHARE || CurrentAggType==BRANCHED) {
		if ( CurrentAccess == PRIVATE || CurrentAccess == PUBLIC )
		    ProcessFn(yyvsp[-2]) ;
	    }
	  ;
    break;}
case 541:
#line 2537 "xp-t.bison"
{if ( CurrentAggType==CHARE || CurrentAggType==BRANCHED ) {
                if ( CurrentAccess == ENTRY )
                        ProcessEP(yyvsp[0],TRUE);
		else if ( CurrentAccess == PRIVATE || CurrentAccess == PUBLIC )
			ProcessFn(yyvsp[0]) ;
		InsideChareCode = 1 ;
	   }
	   else 
		SetDefinedIfEp(yyvsp[0]) ;
	   AddScope(yyvsp[0]) ;		
	  ;
    break;}
case 542:
#line 2549 "xp-t.bison"
{ RemoveScope(yyvsp[-2]) ; 
	    yyval = yyvsp[-2] ;
	    InsideChareCode = 0 ;
	  ;
    break;}
case 543:
#line 2553 "xp-t.bison"
{ FLUSHBUF() ; 
			  	  	  SyntaxError("function header") ; ;
    break;}
case 545:
#line 2558 "xp-t.bison"
{       AddScope(yyvsp[0]) ;  ;
    break;}
case 546:
#line 2560 "xp-t.bison"
{ RemoveScope(yyvsp[-2]) ; 
	    yyval = yyvsp[-2] ;
	  ;
    break;}
case 547:
#line 2565 "xp-t.bison"
{int ind ;
	   if ( CurrentAggType==CHARE || CurrentAggType==BRANCHED ) {
		if ( CurrentAccess == ENTRY )
			ProcessEP(yyvsp[0],TRUE);
		else if ( CurrentAccess == PRIVATE || CurrentAccess == PUBLIC )
			ProcessFn(yyvsp[0]) ;
		InsideChareCode = 1 ;
	   }
	   else if ( CurrentAggType == MESSAGE ) {
		if ( strcmp(yyvsp[0],"pack")==0 || strcmp(yyvsp[0],"unpack")==0 ) {
			if ( (ind=FoundInMsgTable(CurrentAggName)) != -1 )
				MessageTable[ind].pack = TRUE ;	
			else 
				CharmError("TRANSLATOR : did not find message type in message table") ;
		}			
		else 
			CharmError("Messages are allowed to have only pack or unpack functions") ;
	   }
	   else if ( CurrentAggType == ACCUMULATOR ) {
		if ( strcmp(yyvsp[0],"Accumulate")==0 && strcmp(yyvsp[-1],"void")==0 ) 
			FoundAccFnDef = TRUE ;
	   }
	   else 
		SetDefinedIfEp(yyvsp[0]) ;
	   FLUSHBUF() ;
	   AddScope(yyvsp[0]) ;		
	  ;
    break;}
case 548:
#line 2593 "xp-t.bison"
{ RemoveScope(yyvsp[-2]) ; 
	    yyval = yyvsp[-2] ;
	    InsideChareCode = 0 ;
	  ;
    break;}
case 549:
#line 2598 "xp-t.bison"
{ FLUSHBUF() ; 
			  	  SyntaxError("function header") ; ;
    break;}
case 551:
#line 2603 "xp-t.bison"
{AddScope(yyvsp[0]) ;;
    break;}
case 552:
#line 2604 "xp-t.bison"
{ RemoveScope(yyvsp[-2]) ; 
	    yyval = yyvsp[-2] ;
	  ;
    break;}
case 553:
#line 2608 "xp-t.bison"
{AddScope(yyvsp[0]) ;;
    break;}
case 554:
#line 2609 "xp-t.bison"
{ RemoveScope(yyvsp[-2]) ; 
	    yyval = yyvsp[-2] ;
	  ;
    break;}
case 555:
#line 2613 "xp-t.bison"
{AddScope(yyvsp[0]) ;;
    break;}
case 556:
#line 2614 "xp-t.bison"
{ RemoveScope(yyvsp[-2]) ; 
	    yyval = yyvsp[-2] ;
	  ;
    break;}
case 557:
#line 2618 "xp-t.bison"
{AddScope(yyvsp[0]) ;;
    break;}
case 558:
#line 2619 "xp-t.bison"
{ RemoveScope(yyvsp[-2]) ; 
	    yyval = yyvsp[-2] ;
	  ;
    break;}
case 559:
#line 2632 "xp-t.bison"
{;
    break;}
case 561:
#line 2633 "xp-t.bison"
{;
    break;}
case 563:
#line 2634 "xp-t.bison"
{;
    break;}
case 565:
#line 2635 "xp-t.bison"
{;
    break;}
case 567:
#line 2636 "xp-t.bison"
{;
    break;}
case 569:
#line 2637 "xp-t.bison"
{;
    break;}
case 571:
#line 2638 "xp-t.bison"
{;
    break;}
case 573:
#line 2639 "xp-t.bison"
{;
    break;}
case 577:
#line 2670 "xp-t.bison"
{	int ind ;	

		SetDefinedIfEp(yyvsp[-2]) ;
	   	CheckConstructorEP(yyvsp[-2],TRUE) ; 

		if ( (ind=FoundInAccTable(AccTable,TotalAccs,yyvsp[-2])) != -1 ) {
		/* This is an Accumulator constructor */
			AccTable[ind]->initmsgtype = (char *)malloc((strlen(CurrentMsgParm)+1)*sizeof(char)) ;
			strcpy(AccTable[ind]->initmsgtype,CurrentMsgParm) ;
			AccTable[ind]->defined = 1 ;
		}
		else if ( (ind=FoundInAccTable(MonoTable,TotalMonos,yyvsp[-2])) != -1 ) {
		/* This is a Monotonic constructor */
			MonoTable[ind]->initmsgtype = (char *)malloc((strlen(CurrentMsgParm)+1)*sizeof(char)) ;
			strcpy(MonoTable[ind]->initmsgtype,CurrentMsgParm) ;
			MonoTable[ind]->defined = 1 ;
		}

	        AddScope(yyvsp[-2]) ;		
	;
    break;}
case 578:
#line 2691 "xp-t.bison"
{	RemoveScope(yyvsp[-4]) ;
		InsideChareCode = 0 ;
	;
    break;}
case 582:
#line 2731 "xp-t.bison"
{ 	foundargs = FALSE ; ;
    break;}
case 583:
#line 2733 "xp-t.bison"
{	int ind ;	

		if ( (ind=FoundInAccTable(AccTable,TotalAccs,yyvsp[-1])) != -1 ) {
		/* This is an Accumulator constructor */
			AccTable[ind]->initmsgtype = (char *)malloc((strlen(CurrentMsgParm)+1)*sizeof(char)) ;
			strcpy(AccTable[ind]->initmsgtype,CurrentMsgParm) ;
			AccTable[ind]->defined = FoundConstructorBody ;
		}
		else if ( (ind=FoundInAccTable(MonoTable,TotalMonos,yyvsp[-1])) != -1 ) {
		/* This is a Monotonic constructor */
			MonoTable[ind]->initmsgtype = (char *)malloc((strlen(CurrentMsgParm)+1)*sizeof(char)) ;
			strcpy(MonoTable[ind]->initmsgtype,CurrentMsgParm) ;
			MonoTable[ind]->defined = FoundConstructorBody ;
		}

		foundargs = FALSE ;
	  ;
    break;}
case 584:
#line 2763 "xp-t.bison"
{ CheckConstructorEP(yyvsp[-4],FALSE) ; 
	     FoundConstructorBody = FALSE ; ;
    break;}
case 585:
#line 2766 "xp-t.bison"
{ CheckConstructorEP(yyvsp[-6],FALSE) ; 
	     FoundConstructorBody = FALSE ; ;
    break;}
case 586:
#line 2769 "xp-t.bison"
{ foundargs = TRUE ; 
	     CheckConstructorEP(yyvsp[-5],FALSE) ; 
	     FoundConstructorBody = FALSE ; ;
    break;}
case 587:
#line 2775 "xp-t.bison"
{    CheckConstructorEP(yyvsp[-4],TRUE) ; 
	   ;
    break;}
case 588:
#line 2778 "xp-t.bison"
{ 	FoundConstructorBody = TRUE ; 
	   ;
    break;}
case 589:
#line 2782 "xp-t.bison"
{    CheckConstructorEP(yyvsp[-6],TRUE) ; 
	   ;
    break;}
case 590:
#line 2785 "xp-t.bison"
{ 	FoundConstructorBody = TRUE ; 
	   ;
    break;}
case 591:
#line 2789 "xp-t.bison"
{    foundargs = TRUE ; CheckConstructorEP(yyvsp[-5],TRUE) ; 
	   ;
    break;}
case 592:
#line 2792 "xp-t.bison"
{ 	FoundConstructorBody = TRUE ; 
	   ;
    break;}
case 593:
#line 2796 "xp-t.bison"
{   	EpMsg = CurrentTypedef ; 
		CheckConstructorEP(yyvsp[-1],FALSE) ; 	;
    break;}
case 631:
#line 2935 "xp-t.bison"
{ yyval = yyvsp[0]; ;
    break;}
case 632:
#line 2936 "xp-t.bison"
{ yyval = yyvsp[0]; ;
    break;}
case 635:
#line 2942 "xp-t.bison"
{yyval=yyvsp[0];;
    break;}
case 636:
#line 2946 "xp-t.bison"
{ yyval = yyvsp[0]; ;
    break;}
case 637:
#line 2947 "xp-t.bison"
{ yyval = yyvsp[-1]; ;
    break;}
case 657:
#line 3004 "xp-t.bison"
{ yyval = yyvsp[0]; ;
    break;}
case 658:
#line 3005 "xp-t.bison"
{ yyval = yyvsp[0]; FoundParms = FALSE ;;
    break;}
case 659:
#line 3016 "xp-t.bison"
{ yyval = yyvsp[0]; ;
    break;}
case 660:
#line 3018 "xp-t.bison"
{ if ( FoundHandle > 0 ) {
		if ( CurrentCharmType == ACCUMULATOR )
			InsertHandleTable(AccHandleTable,&AccHandleTableSize,yyvsp[0]) ;
		else if ( CurrentCharmType == MONOTONIC )
			InsertHandleTable(MonoHandleTable,&MonoHandleTableSize,yyvsp[0]) ;
		else if ( CurrentCharmType == WRITEONCE )
			InsertHandleTable(WrOnHandleTable,&WrOnHandleTableSize,yyvsp[0]) ;
		else if ( FoundHandle == GROUP )
			InsertHandleTable(BOCHandleTable,&BOCHandleTableSize,yyvsp[0]) ;
		else if (CurrentCharmType==CHARE || CurrentCharmType==BRANCHED)
			InsertHandleTable(ChareHandleTable,&ChareHandleTableSize,yyvsp[0]);
		else 
			fprintf(stderr,"ERROR : %s, line %d : %s is not a proper type for a handle.\n",CurrentFileName,CurrentLine,CurrentAsterisk) ;
			
		FoundHandle = -1 ;
	    }
	    yyval = yyvsp[0] ;
	  ;
    break;}
case 662:
#line 3042 "xp-t.bison"
{ yyval = yyvsp[-1]; ;
    break;}
case 664:
#line 3045 "xp-t.bison"
{ yyval = yyvsp[-2] ; ;
    break;}
case 677:
#line 3093 "xp-t.bison"
{ FoundParms = TRUE ; ;
    break;}
case 689:
#line 3117 "xp-t.bison"
{ strcpy(CurrentAsterisk,"*") ; ;
    break;}
case 690:
#line 3118 "xp-t.bison"
{ strcpy(CurrentAsterisk,"&") ; ;
    break;}
case 691:
#line 3119 "xp-t.bison"
{	/* CheckCharmName() ;	Done in t.l now */
			FoundHandle = HANDLE ;
			strcpy(CurrentAsterisk, CurrentTypedef) ;
		   ;
    break;}
case 692:
#line 3123 "xp-t.bison"
{	/* CheckCharmName() ;	Done in t.l now */
			FoundHandle = GROUP ;
			strcpy(CurrentAsterisk, CurrentTypedef) ;
		   ;
    break;}
case 693:
#line 3130 "xp-t.bison"
{ if ( AddedScope > 0 ) {
							AddedScope = 0 ;
							PopStack() ;
						  }
						;
    break;}
case 695:
#line 3162 "xp-t.bison"
{ 	yyval = yyvsp[0] ; 
		AddOneScope(yyvsp[0]) ;
	  ;
    break;}
case 696:
#line 3166 "xp-t.bison"
{ 	yyval = yyvsp[0] ; 
		AddOneScope(yyvsp[0]) ;
	  ;
    break;}
case 697:
#line 3174 "xp-t.bison"
{ 	yyval = (char *)malloc(sizeof(char)*(strlen(yyvsp[-1])+3)) ;
		strcpy(yyval,yyvsp[-1]) ;
		strcat(yyval,yyvsp[0]) ; 
	  ;
    break;}
case 698:
#line 3179 "xp-t.bison"
{ 	yyval = (char *)malloc(sizeof(char)*(strlen(yyvsp[-2])+strlen(yyvsp[-1])+3)) ;
		strcpy(yyval,yyvsp[-2]) ;
		strcat(yyval,yyvsp[-1]) ; 
		strcat(yyval,yyvsp[0]) ; 
	  ;
    break;}
case 699:
#line 3197 "xp-t.bison"
{ yyval = yyvsp[0] ; ;
    break;}
case 700:
#line 3198 "xp-t.bison"
{ yyval = yyvsp[0] ; ;
    break;}
case 701:
#line 3202 "xp-t.bison"
{ /*scan for upcoming name in file scope */ FoundGlobalScope=1 ; ;
    break;}
case 704:
#line 3207 "xp-t.bison"
{ if ( AddedScope > 0 ) { 
				PopStack() ;		
				AddedScope = 0 ;
			  }
			;
    break;}
case 705:
#line 3212 "xp-t.bison"
{ if ( AddedScope > 0 ) { 
					PopStack() ;		
					AddedScope = 0 ;
			  	  }
				;
    break;}
case 706:
#line 3228 "xp-t.bison"
{ 	yyval = yyvsp[0] ; 
		CheckSharedHandle(yyvsp[0]);
	  ;
    break;}
case 707:
#line 3232 "xp-t.bison"
{ 	char *sptr, *sptr2, *lastcoln ;
		int ch, bo ;	
		EP *ep ;
		char epstr[128] ;

                if ( AddedScope > 0 ) { 
			PopStack() ;		
			AddedScope = 0 ;
		}

                /* Note : $$ must fit Chare::EP and _CK_ep_Chare_EP */
                yyval = (char *)malloc(sizeof(char)*(strlen(yyvsp[-1])+strlen(yyvsp[0])+10)) ;
		strcpy(yyval,yyvsp[-1]) ;
		strcat(yyval,yyvsp[0]) ; 
		FoundChareEPPair = 0 ;

		CheckSharedHandle(yyvsp[0]);
		if ( SendType == -1 ) {
		    lastcoln = Mystrstr(yyvsp[-1],"::") ;
		    *lastcoln = '\0' ;
	
		    ch = FoundInChareTable(ChareTable,charecount+1,yyvsp[-1]) ;
		    bo = FoundInChareTable(BOCTable,boccount+1,yyvsp[-1]) ;
		    if ( ch != -1 || bo != -1 )  {
		        
			/* Now we have a Chare::EP pair */
			if ( ch != -1 )
				ep = SearchEPList(ChareTable[ch]->eps,yyvsp[0]) ;
			else if ( bo != -1 )
				ep = SearchEPList(BOCTable[bo]->eps,yyvsp[0]) ;
			if ( ep != NULL ) {
			  sptr = Mystrstr(OutBuf,"&") ; 
			  if ( sptr != NULL ) {
				*sptr = ' ' ; /* remove ampersand */
			  	sptr2 = Mystrstr(sptr,yyvsp[-1]) ;
			  	if ( sptr2 != NULL ) 
				  	*sptr2 = '\0' ;
			  	else 
				  	fprintf(stderr,"TRANSLATOR ERROR in ChareType::EntryFn usage, %s, line %d\n",CurrentFileName,CurrentLine) ;

				/* dont flush because OutBuf may have stuff we
			  	   FLUSHBUF() ; want later */ 

			  	sprintf(epstr,"_CK_ep_%s_%s",yyvsp[-1],yyvsp[0]) ;
				strcat(OutBuf,epstr) ;

				FoundChareEPPair = 1 ;
				strcpy(yyval,epstr) ; 
				/* so that higher-level rules get _CK_ep_...
				   instead of &(Chare::EP) */
				
				strcpy(LastChare,yyvsp[-1]) ;
				strcpy(LastEP,yyvsp[0]) ;
			  }
			}
		    }
		}
	  ;
    break;}
case 708:
#line 3293 "xp-t.bison"
{ 	yyval = yyvsp[0] ; ;
    break;}
case 709:
#line 3294 "xp-t.bison"
{ 	if ( AddedScope > 0 ) { 
						PopStack() ;		
						AddedScope = 0 ;
					}
				;
    break;}
case 711:
#line 3303 "xp-t.bison"
{ yyval = yyvsp[0] ; ;
    break;}
case 712:
#line 3320 "xp-t.bison"
{ yyval = yyvsp[0] ; ;
    break;}
case 713:
#line 3321 "xp-t.bison"
{ yyval = yyvsp[0] ; ;
    break;}
case 716:
#line 3338 "xp-t.bison"
{ yyval = (char *)malloc(sizeof(char)*(strlen(yyvsp[-1])+strlen(yyvsp[0])+1)) ;
	  strcpy(yyval,yyvsp[-1]) ;
	  strcat(yyval,yyvsp[0]) ; 
	  strcpy(CurrentTypedef,yyval) ;
          if ( AddedScope > 0 ) { 
		  PopStack() ;		
		  AddedScope = 0 ;
	  }
	;
    break;}
case 717:
#line 3350 "xp-t.bison"
{ yyval = yyvsp[0] ; ;
    break;}
case 718:
#line 3352 "xp-t.bison"
{ 	yyval = yyvsp[0] ; 
		strcpy(CurrentTypedef,yyvsp[-1]) ;
		strcat(CurrentTypedef,yyvsp[0]) ;
	  ;
    break;}
case 719:
#line 3357 "xp-t.bison"
{ 	yyval = yyvsp[0] ; 
		strcpy(CurrentTypedef,yyvsp[-1]) ;
		strcat(CurrentTypedef,yyvsp[0]) ;
	  ;
    break;}
case 720:
#line 3364 "xp-t.bison"
{ yyval = yyvsp[0] ; ;
    break;}
case 721:
#line 3365 "xp-t.bison"
{ yyval = yyvsp[0] ; ;
    break;}
}
   /* the action file gets copied in in place of this dollarsign */
#line 498 "/local/encap/bison-1.25/share/bison.simple"

  yyvsp -= yylen;
  yyssp -= yylen;
#ifdef YYLSP_NEEDED
  yylsp -= yylen;
#endif

#if YYDEBUG != 0
  if (yydebug)
    {
      short *ssp1 = yyss - 1;
      fprintf (stderr, "state stack now");
      while (ssp1 != yyssp)
	fprintf (stderr, " %d", *++ssp1);
      fprintf (stderr, "\n");
    }
#endif

  *++yyvsp = yyval;

#ifdef YYLSP_NEEDED
  yylsp++;
  if (yylen == 0)
    {
      yylsp->first_line = yylloc.first_line;
      yylsp->first_column = yylloc.first_column;
      yylsp->last_line = (yylsp-1)->last_line;
      yylsp->last_column = (yylsp-1)->last_column;
      yylsp->text = 0;
    }
  else
    {
      yylsp->last_line = (yylsp+yylen-1)->last_line;
      yylsp->last_column = (yylsp+yylen-1)->last_column;
    }
#endif

  /* Now "shift" the result of the reduction.
     Determine what state that goes to,
     based on the state we popped back to
     and the rule number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTBASE] + *yyssp;
  if (yystate >= 0 && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTBASE];

  goto yynewstate;

yyerrlab:   /* here on detecting error */

  if (! yyerrstatus)
    /* If not already recovering from an error, report this error.  */
    {
      ++yynerrs;

#ifdef YYERROR_VERBOSE
      yyn = yypact[yystate];

      if (yyn > YYFLAG && yyn < YYLAST)
	{
	  int size = 0;
	  char *msg;
	  int x, count;

	  count = 0;
	  /* Start X at -yyn if nec to avoid negative indexes in yycheck.  */
	  for (x = (yyn < 0 ? -yyn : 0);
	       x < (sizeof(yytname) / sizeof(char *)); x++)
	    if (yycheck[x + yyn] == x)
	      size += strlen(yytname[x]) + 15, count++;
	  msg = (char *) malloc(size + 15);
	  if (msg != 0)
	    {
	      strcpy(msg, "parse error");

	      if (count < 5)
		{
		  count = 0;
		  for (x = (yyn < 0 ? -yyn : 0);
		       x < (sizeof(yytname) / sizeof(char *)); x++)
		    if (yycheck[x + yyn] == x)
		      {
			strcat(msg, count == 0 ? ", expecting `" : " or `");
			strcat(msg, yytname[x]);
			strcat(msg, "'");
			count++;
		      }
		}
	      yyerror(msg);
	      free(msg);
	    }
	  else
	    yyerror ("parse error; also virtual memory exceeded");
	}
      else
#endif /* YYERROR_VERBOSE */
	yyerror("parse error");
    }

  goto yyerrlab1;
yyerrlab1:   /* here on error raised explicitly by an action */

  if (yyerrstatus == 3)
    {
      /* if just tried and failed to reuse lookahead token after an error, discard it.  */

      /* return failure if at end of input */
      if (yychar == YYEOF)
	YYABORT;

#if YYDEBUG != 0
      if (yydebug)
	fprintf(stderr, "Discarding token %d (%s).\n", yychar, yytname[yychar1]);
#endif

      yychar = YYEMPTY;
    }

  /* Else will try to reuse lookahead token
     after shifting the error token.  */

  yyerrstatus = 3;		/* Each real token shifted decrements this */

  goto yyerrhandle;

yyerrdefault:  /* current state does not do anything special for the error token. */

#if 0
  /* This is wrong; only states that explicitly want error tokens
     should shift them.  */
  yyn = yydefact[yystate];  /* If its default is to accept any token, ok.  Otherwise pop it.*/
  if (yyn) goto yydefault;
#endif

yyerrpop:   /* pop the current state because it cannot handle the error token */

  if (yyssp == yyss) YYABORT;
  yyvsp--;
  yystate = *--yyssp;
#ifdef YYLSP_NEEDED
  yylsp--;
#endif

#if YYDEBUG != 0
  if (yydebug)
    {
      short *ssp1 = yyss - 1;
      fprintf (stderr, "Error: state stack now");
      while (ssp1 != yyssp)
	fprintf (stderr, " %d", *++ssp1);
      fprintf (stderr, "\n");
    }
#endif

yyerrhandle:

  yyn = yypact[yystate];
  if (yyn == YYFLAG)
    goto yyerrdefault;

  yyn += YYTERROR;
  if (yyn < 0 || yyn > YYLAST || yycheck[yyn] != YYTERROR)
    goto yyerrdefault;

  yyn = yytable[yyn];
  if (yyn < 0)
    {
      if (yyn == YYFLAG)
	goto yyerrpop;
      yyn = -yyn;
      goto yyreduce;
    }
  else if (yyn == 0)
    goto yyerrpop;

  if (yyn == YYFINAL)
    YYACCEPT;

#if YYDEBUG != 0
  if (yydebug)
    fprintf(stderr, "Shifting error token, ");
#endif

  *++yyvsp = yylval;
#ifdef YYLSP_NEEDED
  *++yylsp = yylloc;
#endif

  yystate = yyn;
  goto yynewstate;
}
#line 3368 "xp-t.bison"

yyerror(string)
char*string;
{
	fprintf(stderr,"ERROR : %s, line %d : %s near \"%s\".\n", CurrentFileName,CurrentLine,string,prevtoken);
	ErrVal = TRUE ;
}


char *Concat2(s1,s2)
char *s1, *s2 ;
{
	int len = strlen(s1) + strlen(s2) + 1 ;
	char *s = (char *)malloc(sizeof(char)*len) ;
	sprintf(s,"%s%s",s1,s2) ;
/* 	free(s1) ; free(s2) ;  */
	
	return(s) ;
}

char *Concat3(s1,s2,s3)
char *s1, *s2, *s3 ;
{
	int len = strlen(s1) + strlen(s2) + strlen(s3) + 1 ;
	char *s = (char *)malloc(sizeof(char)*len) ;
	sprintf(s,"%s%s%s",s1,s2,s3) ;
/* 	free(s1) ; free(s2) ; free(s3)  */
	
	return(s) ;
}
