#ifndef lint
static char yysccsid[] = "@(#)yaccpar	1.8 (Berkeley) 01/20/90";
#endif
#define YYBYACC 1
#line 3 "xi-parse.bison"

#include "xi-symbol.h"

extern int lineno;
extern int yylex (void) ;
extern Module *thismodule ;

/* Local data and functions*/
int yyerror(char *);

#line 19 "xi-parse.bison"
typedef union {
	char *strval;
	int intval;
} YYSTYPE;
#line 22 "y.tab.c"
#define BOC 257
#define CHARE 258
#define ENTRY 259
#define MESSAGE 260
#define PACKMESSAGE 261
#define READONLY 262
#define TABLE 263
#define THREADED 264
#define EXTERN 265
#define IDENTIFIER 266
#define YYERRCODE 256
short yylhs[] = {                                        -1,
    0,   12,   12,   13,   13,   14,   14,   14,   14,   14,
   14,   11,   11,   22,   15,   23,   16,    2,    1,   21,
   21,    9,    9,   10,   10,   24,    3,   17,   20,    4,
   18,   18,    7,    8,    5,   19,    6,
};
short yylen[] = {                                         2,
    1,    0,    1,    1,    2,    1,    1,    1,    1,    1,
    1,    0,    1,    0,    8,    0,    8,    1,    1,    1,
    2,    0,    2,    0,    1,    8,    1,    4,    4,    1,
    4,    4,    1,    2,    1,    3,    1,
};
short yydefred[] = {                                      0,
    0,    0,   13,    0,    0,    1,    0,    4,    6,    7,
    8,    9,   10,   11,   19,    0,    0,    0,   37,    0,
    0,    0,    0,    0,    5,   34,   35,    0,    0,   36,
   18,    0,    0,   30,    0,    0,   31,   32,   14,   16,
   28,   29,    0,    0,   25,    0,    0,   20,    0,    0,
    0,    0,   21,    0,   23,    0,   15,   17,   27,    0,
    0,    0,    0,   26,
};
short yydgoto[] = {                                       4,
   34,   32,   60,   50,   28,   20,   17,   18,   51,   46,
    5,    6,    7,    8,    9,   10,   11,   12,   13,   14,
   47,   43,   44,   48,
};
short yysindex[] = {                                   -249,
 -257, -257,    0,    0, -250,    0, -249,    0,    0,    0,
    0,    0,    0,    0,    0,  -24, -257, -257,    0,  -34,
 -257, -257, -257, -257,    0,    0,    0,  -33,  -31,    0,
    0,  -96,  -94,    0,  -29,  -28,    0,    0,    0,    0,
    0,    0, -232, -232,    0, -257, -125,    0, -123,   -9,
 -225,  -23,    0,  -22,    0, -257,    0,    0,    0,   -5,
 -257,   -3,  -20,    0,
};
short yyrindex[] = {                                      1,
    0,    0,    0,    0,    0,    0,    6,    0,    0,    0,
    0,    0,    0,    0,    0, -226,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0, -254, -254,    0, -218, -254,    0, -254,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    3,    0,    0,    0,
};
short yygindex[] = {                                      0,
    2,   20,    0,   -2,   25,    0,    0,    0,  -16,    0,
    0,    0,    0,   39,    0,    0,    0,    0,    0,    0,
    4,    0,    0,  -32,
};
#define YYTABLESIZE 267
short yytable[] = {                                      52,
    2,   54,   16,   19,   24,    3,   21,   22,   15,   23,
   24,   24,    1,    2,   53,    3,   53,   26,   27,   27,
   35,   36,   31,   31,   30,   37,   39,   38,   40,   41,
   42,   45,   55,   56,   61,   57,   58,   63,   64,   33,
   22,   33,   29,   22,   62,   25,    0,   49,    0,    0,
    0,    0,    0,    0,    0,    0,    0,   59,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,   45,    0,
   45,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,   12,   12,    0,
   12,   12,   12,   12,    0,   12,   12,
};
short yycheck[] = {                                     125,
    0,  125,    1,    2,  259,    0,  257,  258,  266,  260,
  261,  266,  262,  263,   47,  265,   49,   42,   17,   18,
   23,   24,   21,   22,   59,   59,  123,   59,  123,   59,
   59,  264,   42,  259,   40,   59,   59,   41,   59,  266,
  259,   22,   18,   41,   61,    7,   -1,   44,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   56,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,  264,   -1,
  264,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,  257,  258,   -1,
  260,  261,  257,  258,   -1,  260,  261,
};
#define YYFINAL 4
#ifndef YYDEBUG
#define YYDEBUG 0
#endif
#define YYMAXTOKEN 266
#if YYDEBUG
char *yyname[] = {
"end-of-file",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,"'('","')'","'*'",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"';'",0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"'{'",0,"'}'",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"BOC",
"CHARE","ENTRY","MESSAGE","PACKMESSAGE","READONLY","TABLE","THREADED","EXTERN",
"IDENTIFIER",
};
char *yyrule[] = {
"$accept : File",
"File : ItemElist",
"ItemElist :",
"ItemElist : ItemList",
"ItemList : Item",
"ItemList : ItemList Item",
"Item : Boc",
"Item : Chare",
"Item : Message",
"Item : ReadOnly",
"Item : Table",
"Item : PackMessage",
"OptionalExtern :",
"OptionalExtern : EXTERN",
"$$1 :",
"Boc : OptionalExtern BOC ChareName '{' $$1 EntryList '}' ';'",
"$$2 :",
"Chare : OptionalExtern CHARE ChareName '{' $$2 EntryList '}' ';'",
"ChareName : Id",
"Id : IDENTIFIER",
"EntryList : Entry",
"EntryList : EntryList Entry",
"OptionalMessagePtr :",
"OptionalMessagePtr : MessageName '*'",
"OptionalThreaded :",
"OptionalThreaded : THREADED",
"Entry : OptionalThreaded OptionalMessagePtr ENTRY EntryName '(' OptionalMessagePtr ')' ';'",
"EntryName : Id",
"Message : OptionalExtern MESSAGE MessageName ';'",
"PackMessage : OptionalExtern PACKMESSAGE MessageName ';'",
"MessageName : Id",
"ReadOnly : READONLY SimpleType ReadOnlyName ';'",
"ReadOnly : READONLY PtrType ReadOnlyName ';'",
"SimpleType : Id",
"PtrType : Id '*'",
"ReadOnlyName : Id",
"Table : TABLE TableName ';'",
"TableName : Id",
};
#endif
#define yyclearin (yychar=(-1))
#define yyerrok (yyerrflag=0)
#ifdef YYSTACKSIZE
#ifndef YYMAXDEPTH
#define YYMAXDEPTH YYSTACKSIZE
#endif
#else
#ifdef YYMAXDEPTH
#define YYSTACKSIZE YYMAXDEPTH
#else
#define YYSTACKSIZE 500
#define YYMAXDEPTH 500
#endif
#endif
int yydebug;
int yynerrs;
int yyerrflag;
int yychar;
short *yyssp;
YYSTYPE *yyvsp;
YYSTYPE yyval;
YYSTYPE yylval;
short yyss[YYSTACKSIZE];
YYSTYPE yyvs[YYSTACKSIZE];
#define yystacksize YYSTACKSIZE
#line 174 "xi-parse.bison"

int yyerror(char *mesg)
{
	cout << "Syntax error at line " << lineno << ": " << mesg << endl;
	return 0;
}

#line 233 "y.tab.c"
#define YYABORT goto yyabort
#define YYACCEPT goto yyaccept
#define YYERROR goto yyerrlab
int
yyparse()
{
    register int yym, yyn, yystate;
#if YYDEBUG
    register char *yys;
    extern char *getenv();

    if (yys = getenv("YYDEBUG"))
    {
        yyn = *yys;
        if (yyn >= '0' && yyn <= '9')
            yydebug = yyn - '0';
    }
#endif

    yynerrs = 0;
    yyerrflag = 0;
    yychar = (-1);

    yyssp = yyss;
    yyvsp = yyvs;
    *yyssp = yystate = 0;

yyloop:
    if (yyn = yydefred[yystate]) goto yyreduce;
    if (yychar < 0)
    {
        if ((yychar = yylex()) < 0) yychar = 0;
#if YYDEBUG
        if (yydebug)
        {
            yys = 0;
            if (yychar <= YYMAXTOKEN) yys = yyname[yychar];
            if (!yys) yys = "illegal-symbol";
            printf("yydebug: state %d, reading %d (%s)\n", yystate,
                    yychar, yys);
        }
#endif
    }
    if ((yyn = yysindex[yystate]) && (yyn += yychar) >= 0 &&
            yyn <= YYTABLESIZE && yycheck[yyn] == yychar)
    {
#if YYDEBUG
        if (yydebug)
            printf("yydebug: state %d, shifting to state %d\n",
                    yystate, yytable[yyn]);
#endif
        if (yyssp >= yyss + yystacksize - 1)
        {
            goto yyoverflow;
        }
        *++yyssp = yystate = yytable[yyn];
        *++yyvsp = yylval;
        yychar = (-1);
        if (yyerrflag > 0)  --yyerrflag;
        goto yyloop;
    }
    if ((yyn = yyrindex[yystate]) && (yyn += yychar) >= 0 &&
            yyn <= YYTABLESIZE && yycheck[yyn] == yychar)
    {
        yyn = yytable[yyn];
        goto yyreduce;
    }
    if (yyerrflag) goto yyinrecovery;
#ifdef lint
    goto yynewerror;
#endif
yynewerror:
    yyerror("syntax error");
#ifdef lint
    goto yyerrlab;
#endif
yyerrlab:
    ++yynerrs;
yyinrecovery:
    if (yyerrflag < 3)
    {
        yyerrflag = 3;
        for (;;)
        {
            if ((yyn = yysindex[*yyssp]) && (yyn += YYERRCODE) >= 0 &&
                    yyn <= YYTABLESIZE && yycheck[yyn] == YYERRCODE)
            {
#if YYDEBUG
                if (yydebug)
                    printf("yydebug: state %d, error recovery shifting\
 to state %d\n", *yyssp, yytable[yyn]);
#endif
                if (yyssp >= yyss + yystacksize - 1)
                {
                    goto yyoverflow;
                }
                *++yyssp = yystate = yytable[yyn];
                *++yyvsp = yylval;
                goto yyloop;
            }
            else
            {
#if YYDEBUG
                if (yydebug)
                    printf("yydebug: error recovery discarding state %d\n",
                            *yyssp);
#endif
                if (yyssp <= yyss) goto yyabort;
                --yyssp;
                --yyvsp;
            }
        }
    }
    else
    {
        if (yychar == 0) goto yyabort;
#if YYDEBUG
        if (yydebug)
        {
            yys = 0;
            if (yychar <= YYMAXTOKEN) yys = yyname[yychar];
            if (!yys) yys = "illegal-symbol";
            printf("yydebug: state %d, error recovery discards token %d (%s)\n",
                    yystate, yychar, yys);
        }
#endif
        yychar = (-1);
        goto yyloop;
    }
yyreduce:
#if YYDEBUG
    if (yydebug)
        printf("yydebug: state %d, reducing by rule %d (%s)\n",
                yystate, yyn, yyrule[yyn]);
#endif
    yym = yylen[yyn];
    yyval = yyvsp[1-yym];
    switch (yyn)
    {
case 12:
#line 58 "xi-parse.bison"
{ yyval.intval = FALSE; }
break;
case 13:
#line 60 "xi-parse.bison"
{ yyval.intval = TRUE; }
break;
case 14:
#line 64 "xi-parse.bison"
{
			Chare *c = new Chare(yyvsp[-1].strval, BOC, yyvsp[-3].intval) ;
			delete yyvsp[-1].strval;
			thismodule->AddChare(c) ;
		}
break;
case 16:
#line 73 "xi-parse.bison"
{
			Chare *c = new Chare(yyvsp[-1].strval, CHARE, yyvsp[-3].intval) ;
			delete yyvsp[-1].strval;
			thismodule->AddChare(c) ;
		}
break;
case 22:
#line 93 "xi-parse.bison"
{
			yyval.strval = NULL;
		}
break;
case 24:
#line 101 "xi-parse.bison"
{ yyval.intval = FALSE; }
break;
case 25:
#line 103 "xi-parse.bison"
{ yyval.intval = TRUE; }
break;
case 26:
#line 107 "xi-parse.bison"
{
			thismodule->chares->AddEntry(yyvsp[-4].strval, yyvsp[-2].strval, yyvsp[-7].intval, yyvsp[-6].strval) ;
			delete yyvsp[-4].strval; delete yyvsp[-2].strval;
		}
break;
case 28:
#line 117 "xi-parse.bison"
{
			Message *m = new Message(yyvsp[-1].strval, 0, yyvsp[-3].intval) ;
			delete yyvsp[-1].strval;
			thismodule->AddMessage(m) ;
		}
break;
case 29:
#line 125 "xi-parse.bison"
{
			Message *m = new Message(yyvsp[-1].strval, 1, yyvsp[-3].intval) ;
			delete yyvsp[-1].strval;
			thismodule->AddMessage(m) ;
		}
break;
case 31:
#line 136 "xi-parse.bison"
{
			ReadOnly *r = new ReadOnly(yyvsp[-1].strval, yyvsp[-2].strval, 0) ;
			delete yyvsp[-2].strval;
			thismodule->AddReadOnly(r) ;
		}
break;
case 32:
#line 142 "xi-parse.bison"
{
			ReadOnly *r = new ReadOnly(yyvsp[-1].strval, yyvsp[-2].strval, 1) ;
			delete yyvsp[-2].strval;
			thismodule->AddReadOnly(r) ;
		}
break;
case 34:
#line 153 "xi-parse.bison"
{
			yyval.strval = strcat(strcpy(new char[2+strlen(yyvsp[-1].strval)], yyvsp[-1].strval), "*");
			delete yyvsp[-1].strval;
		}
break;
case 36:
#line 163 "xi-parse.bison"
{
			Table *t = new Table(yyvsp[-1].strval) ;
			delete yyvsp[-1].strval;
			thismodule->AddTable(t) ;
		}
break;
#line 465 "y.tab.c"
    }
    yyssp -= yym;
    yystate = *yyssp;
    yyvsp -= yym;
    yym = yylhs[yyn];
    if (yystate == 0 && yym == 0)
    {
#if YYDEBUG
        if (yydebug)
            printf("yydebug: after reduction, shifting from state 0 to\
 state %d\n", YYFINAL);
#endif
        yystate = YYFINAL;
        *++yyssp = YYFINAL;
        *++yyvsp = yyval;
        if (yychar < 0)
        {
            if ((yychar = yylex()) < 0) yychar = 0;
#if YYDEBUG
            if (yydebug)
            {
                yys = 0;
                if (yychar <= YYMAXTOKEN) yys = yyname[yychar];
                if (!yys) yys = "illegal-symbol";
                printf("yydebug: state %d, reading %d (%s)\n",
                        YYFINAL, yychar, yys);
            }
#endif
        }
        if (yychar == 0) goto yyaccept;
        goto yyloop;
    }
    if ((yyn = yygindex[yym]) && (yyn += yystate) >= 0 &&
            yyn <= YYTABLESIZE && yycheck[yyn] == yystate)
        yystate = yytable[yyn];
    else
        yystate = yydgoto[yym];
#if YYDEBUG
    if (yydebug)
        printf("yydebug: after reduction, shifting from state %d \
to state %d\n", *yyssp, yystate);
#endif
    if (yyssp >= yyss + yystacksize - 1)
    {
        goto yyoverflow;
    }
    *++yyssp = yystate;
    *++yyvsp = yyval;
    goto yyloop;
yyoverflow:
    yyerror("yacc stack overflow");
yyabort:
    return (1);
yyaccept:
    return (0);
}
