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
#define STACKSIZE 263
#define TABLE 264
#define THREADED 265
#define VARSIZE 266
#define EXTERN 267
#define IDENTIFIER 268
#define NUMBER 269
#define YYERRCODE 256
short yylhs[] = {                                        -1,
    0,   13,   13,   14,   14,   15,   15,   15,   15,   15,
   15,   15,   11,   11,   23,   23,   24,   24,   25,   16,
   27,   17,    2,    1,   26,   26,    9,    9,   10,   10,
   12,   12,   28,    3,   18,   21,   22,    4,   19,   19,
    7,    8,    5,   20,    6,
};
short yylen[] = {                                         2,
    1,    0,    1,    1,    2,    1,    1,    1,    1,    1,
    1,    1,    0,    1,    0,    2,    1,    3,    0,    9,
    0,    9,    1,    1,    1,    2,    0,    2,    0,    1,
    0,    2,    9,    1,    4,    4,    4,    1,    5,    5,
    1,    2,    1,    4,    1,
};
short yydefred[] = {                                      0,
   14,    0,    0,    1,    0,    4,    6,    7,    8,    9,
   10,   11,   12,    0,    0,    0,    0,    0,    0,    0,
    5,   24,   23,   19,   21,   38,    0,    0,    0,    0,
    0,   45,    0,    0,    0,    0,   35,   36,   42,   43,
    0,    0,   44,   37,    0,    0,    0,   39,   40,    0,
   16,    0,    0,    0,   30,    0,    0,   25,    0,   18,
    0,    0,    0,   26,    0,   28,    0,   20,   22,   34,
    0,    0,    0,    0,    0,    0,   32,   33,
};
short yydgoto[] = {                                       2,
   26,   50,   71,   61,   41,   33,   30,   31,   62,   56,
    3,   76,    4,    5,    6,    7,    8,    9,   10,   11,
   12,   13,   46,   51,   35,   57,   36,   58,
};
short yysindex[] = {                                   -262,
    0,    0, -248,    0, -262,    0,    0,    0,    0,    0,
    0,    0,    0, -251, -251, -251, -251, -251, -251, -251,
    0,    0,    0,    0,    0,    0,  -36,  -30,  -12, -251,
 -251,    0,  -28,  -27,  -25,  -25,    0,    0,    0,    0,
  -24,  -23,    0,    0, -251,  -86,  -85,    0,    0,   -5,
    0, -225, -225, -251,    0, -251, -125,    0, -123,    0,
   -1, -217,  -15,    0,  -14,    0, -251,    0,    0,    0,
    6, -251,    7, -216, -220,   -9,    0,    0,
};
short yyrindex[] = {                                      1,
    0,    0,    0,    0,   11,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0, -215,    0,
    0,    0,    0,    0,  -72,  -72,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,  -71,
    0, -253, -253,    0,    0, -205, -253,    0, -253,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,   14,    0,   -2,    0,    0,    0,    0,
};
short yygindex[] = {                                      0,
  -11,   13,    0,    5,   27,    0,    0,    0,  -13,    0,
    0,    0,    0,    0,   55,    0,    0,    0,    0,    0,
    0,    0,   25,    8,    0,   10,    0,  -33,
};
#define YYTABLESIZE 277
short yytable[] = {                                      63,
    2,   65,   23,   23,    1,   29,   29,   32,   14,   15,
    3,   16,   17,   18,   29,   19,   22,   20,   40,   40,
   27,   28,   37,   64,   34,   64,   24,   25,   38,   39,
   43,   44,   45,   23,   48,   49,   52,   53,   54,   55,
   66,   67,   23,   68,   69,   72,   75,   74,   77,   78,
   15,   17,   41,   27,   27,   70,   31,   42,   73,   21,
   47,   60,   59,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,   55,
    0,   55,    0,    0,    0,    0,    0,    0,    0,    0,
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
    0,    0,    0,    0,    0,    0,    0,   13,   13,    0,
   13,   13,   13,    0,   13,    0,   13,   13,   13,    0,
   13,   13,   13,    0,   13,    0,   13,
};
short yycheck[] = {                                     125,
    0,  125,   14,   15,  267,  259,   18,   19,  257,  258,
    0,  260,  261,  262,  268,  264,  268,  266,   30,   31,
   16,   17,   59,   57,   20,   59,   14,   15,   59,   42,
   59,   59,   58,   45,   59,   59,  123,  123,   44,  265,
   42,  259,   54,   59,   59,   40,  263,   41,  269,   59,
  123,  123,  268,  259,   41,   67,   59,   31,   72,    5,
   36,   54,   53,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,  265,
   -1,  265,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
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
  260,  261,  262,   -1,  264,   -1,  266,  257,  258,   -1,
  260,  261,  262,   -1,  264,   -1,  266,
};
#define YYFINAL 2
#ifndef YYDEBUG
#define YYDEBUG 0
#endif
#define YYMAXTOKEN 269
#if YYDEBUG
char *yyname[] = {
"end-of-file",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,"'('","')'","'*'",0,"','",0,0,0,0,0,0,0,0,0,0,0,0,0,"':'","';'",0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"'{'",0,"'}'",0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
"BOC","CHARE","ENTRY","MESSAGE","PACKMESSAGE","READONLY","STACKSIZE","TABLE",
"THREADED","VARSIZE","EXTERN","IDENTIFIER","NUMBER",
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
"Item : VarsizeMessage",
"OptionalExtern :",
"OptionalExtern : EXTERN",
"OptionalBaseList :",
"OptionalBaseList : ':' BaseList",
"BaseList : ChareName",
"BaseList : ChareName ',' BaseList",
"$$1 :",
"Boc : OptionalExtern BOC ChareName $$1 OptionalBaseList '{' EntryList '}' ';'",
"$$2 :",
"Chare : OptionalExtern CHARE ChareName $$2 OptionalBaseList '{' EntryList '}' ';'",
"ChareName : Id",
"Id : IDENTIFIER",
"EntryList : Entry",
"EntryList : EntryList Entry",
"OptionalMessagePtr :",
"OptionalMessagePtr : MessageName '*'",
"OptionalThreaded :",
"OptionalThreaded : THREADED",
"OptionalStackSize :",
"OptionalStackSize : STACKSIZE NUMBER",
"Entry : OptionalThreaded OptionalMessagePtr ENTRY EntryName '(' OptionalMessagePtr ')' OptionalStackSize ';'",
"EntryName : Id",
"Message : OptionalExtern MESSAGE MessageName ';'",
"PackMessage : OptionalExtern PACKMESSAGE MessageName ';'",
"VarsizeMessage : OptionalExtern VARSIZE MessageName ';'",
"MessageName : Id",
"ReadOnly : OptionalExtern READONLY SimpleType ReadOnlyName ';'",
"ReadOnly : OptionalExtern READONLY PtrType ReadOnlyName ';'",
"SimpleType : Id",
"PtrType : Id '*'",
"ReadOnlyName : Id",
"Table : OptionalExtern TABLE TableName ';'",
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
#line 205 "xi-parse.bison"

int yyerror(char *mesg)
{
	cout << "Syntax error at line " << lineno << ": " << mesg << endl;
	return 0;
}

#line 251 "y.tab.c"
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
case 13:
#line 60 "xi-parse.bison"
{ yyval.intval = 0; }
break;
case 14:
#line 62 "xi-parse.bison"
{ yyval.intval = 1; }
break;
case 17:
#line 72 "xi-parse.bison"
{ thismodule->curChare->AddBase(yyvsp[0].strval); }
break;
case 18:
#line 74 "xi-parse.bison"
{ thismodule->curChare->AddBase(yyvsp[-2].strval); }
break;
case 19:
#line 78 "xi-parse.bison"
{
			Chare *c = new Chare(yyvsp[0].strval, BOC, yyvsp[-2].intval) ;
			delete yyvsp[0].strval;
			thismodule->AddChare(c) ;
		}
break;
case 21:
#line 88 "xi-parse.bison"
{
			Chare *c = new Chare(yyvsp[0].strval, CHARE, yyvsp[-2].intval) ;
			delete yyvsp[0].strval;
			thismodule->AddChare(c) ;
		}
break;
case 27:
#line 109 "xi-parse.bison"
{
			yyval.strval = NULL;
		}
break;
case 29:
#line 117 "xi-parse.bison"
{ yyval.intval = 0; }
break;
case 30:
#line 119 "xi-parse.bison"
{ yyval.intval = 1; }
break;
case 31:
#line 124 "xi-parse.bison"
{ yyval.intval = 0; }
break;
case 32:
#line 126 "xi-parse.bison"
{ yyval.intval = yyvsp[0].intval; }
break;
case 33:
#line 130 "xi-parse.bison"
{
			thismodule->curChare->AddEntry(yyvsp[-5].strval, yyvsp[-3].strval, yyvsp[-8].intval, yyvsp[-7].strval, yyvsp[-1].intval) ;
			delete yyvsp[-5].strval; delete yyvsp[-3].strval;
		}
break;
case 35:
#line 140 "xi-parse.bison"
{
			Message *m = new Message(yyvsp[-1].strval, 0, 0, yyvsp[-3].intval) ;
			delete yyvsp[-1].strval;
			thismodule->AddMessage(m) ;
		}
break;
case 36:
#line 148 "xi-parse.bison"
{
			Message *m = new Message(yyvsp[-1].strval, 1, 0, yyvsp[-3].intval) ;
			delete yyvsp[-1].strval;
			thismodule->AddMessage(m) ;
		}
break;
case 37:
#line 156 "xi-parse.bison"
{
			Message *m = new Message(yyvsp[-1].strval, 1, 1, yyvsp[-3].intval) ;
			delete yyvsp[-1].strval;
			thismodule->AddMessage(m) ;
		}
break;
case 39:
#line 167 "xi-parse.bison"
{
			ReadOnly *r = new ReadOnly(yyvsp[-1].strval, yyvsp[-2].strval, 0, yyvsp[-4].intval) ;
			delete yyvsp[-2].strval;
			thismodule->AddReadOnly(r) ;
		}
break;
case 40:
#line 173 "xi-parse.bison"
{
			ReadOnly *r = new ReadOnly(yyvsp[-1].strval, yyvsp[-2].strval, 1, yyvsp[-4].intval) ;
			delete yyvsp[-2].strval;
			thismodule->AddReadOnly(r) ;
		}
break;
case 42:
#line 184 "xi-parse.bison"
{
			yyval.strval = strcat(strcpy(new char[2+strlen(yyvsp[-1].strval)], yyvsp[-1].strval), "*");
			delete yyvsp[-1].strval;
		}
break;
case 44:
#line 194 "xi-parse.bison"
{
			Table *t = new Table(yyvsp[-1].strval, yyvsp[-3].intval) ;
			delete yyvsp[-1].strval;
			thismodule->AddTable(t) ;
		}
break;
#line 507 "y.tab.c"
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
