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
   15,   15,   11,   11,   24,   16,   25,   17,    2,    1,
   23,   23,    9,    9,   10,   10,   12,   12,   26,    3,
   18,   21,   22,    4,    4,   19,   19,    7,    8,    5,
   20,    6,
};
short yylen[] = {                                         2,
    1,    0,    1,    1,    2,    1,    1,    1,    1,    1,
    1,    1,    0,    1,    0,    8,    0,    8,    1,    1,
    1,    2,    0,    2,    0,    1,    0,    2,    9,    1,
    4,    4,    4,    1,    1,    4,    4,    1,    2,    1,
    3,    1,
};
short yydefred[] = {                                      0,
    0,    0,   14,    0,    0,    1,    0,    4,    6,    7,
    8,    9,   10,   11,   12,   20,    0,    0,    0,   42,
    0,    0,    0,    0,    0,    0,    5,   39,   40,    0,
    0,   41,   19,    0,    0,   34,    0,    0,    0,   36,
   37,   15,   17,   31,   32,   33,    0,    0,   26,    0,
    0,   21,    0,    0,    0,    0,   22,    0,   24,    0,
   16,   18,   30,    0,    0,    0,    0,    0,    0,   28,
   29,
};
short yydgoto[] = {                                       4,
   36,   34,   64,   54,   30,   21,   18,   19,   55,   50,
    5,   69,    6,    7,    8,    9,   10,   11,   12,   13,
   14,   15,   51,   47,   48,   52,
};
short yysindex[] = {                                   -245,
 -242, -242,    0,    0, -251,    0, -245,    0,    0,    0,
    0,    0,    0,    0,    0,    0,  -19, -242, -242,    0,
  -32, -242, -242, -242, -242, -242,    0,    0,    0,  -31,
  -30,    0,    0,  -93,  -92,    0,  -27,  -26,  -25,    0,
    0,    0,    0,    0,    0,    0, -230, -230,    0, -242,
 -125,    0, -123,   -6, -222,  -21,    0,  -20,    0, -242,
    0,    0,    0,    3, -242,   -1, -221, -228,  -15,    0,
    0,
};
short yyrindex[] = {                                      1,
    0,    0,    0,    0,    0,    0,    8,    0,    0,    0,
    0,    0,    0,    0,    0,    0, -223,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0, -254, -254,    0, -213,
 -254,    0, -254,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    6,    0,  -11,    0,    0,    0,
    0,
};
short yygindex[] = {                                      0,
    2,   26,    0,  -13,   31,    0,    0,    0,  -14,    0,
    0,    0,    0,    0,   45,    0,    0,    0,    0,    0,
    0,    0,    5,    0,    0,  -35,
};
#define YYTABLESIZE 274
short yytable[] = {                                      56,
    2,   58,   17,   20,   25,   22,   23,    3,   24,   25,
   37,   38,   39,   25,   26,   57,    1,   57,    2,   29,
   29,    3,   28,   33,   33,   16,   32,   40,   41,   42,
   43,   44,   45,   46,   49,   59,   60,   61,   62,   67,
   70,   68,   65,   71,   38,   23,   23,   27,   35,   31,
   66,   27,   53,    0,    0,    0,    0,    0,    0,    0,
    0,   63,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0,    0,   49,
    0,   49,    0,    0,    0,    0,    0,    0,    0,    0,
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
   13,   13,    0,    0,   13,   13,   13,   13,   13,    0,
    0,    0,    0,   13,
};
short yycheck[] = {                                     125,
    0,  125,    1,    2,  259,  257,  258,    0,  260,  261,
   24,   25,   26,  268,  266,   51,  262,   53,  264,   18,
   19,  267,   42,   22,   23,  268,   59,   59,   59,  123,
  123,   59,   59,   59,  265,   42,  259,   59,   59,   41,
  269,  263,   40,   59,  268,  259,   41,   59,   23,   19,
   65,    7,   48,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
   -1,   60,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
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
  260,  261,   -1,   -1,  257,  258,  266,  260,  261,   -1,
   -1,   -1,   -1,  266,
};
#define YYFINAL 4
#ifndef YYDEBUG
#define YYDEBUG 0
#endif
#define YYMAXTOKEN 269
#if YYDEBUG
char *yyname[] = {
"end-of-file",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,"'('","')'","'*'",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"';'",0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"'{'",0,"'}'",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"BOC",
"CHARE","ENTRY","MESSAGE","PACKMESSAGE","READONLY","STACKSIZE","TABLE",
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
"OptionalStackSize :",
"OptionalStackSize : STACKSIZE NUMBER",
"Entry : OptionalThreaded OptionalMessagePtr ENTRY EntryName '(' OptionalMessagePtr ')' OptionalStackSize ';'",
"EntryName : Id",
"Message : OptionalExtern MESSAGE MessageName ';'",
"PackMessage : OptionalExtern PACKMESSAGE MessageName ';'",
"VarsizeMessage : OptionalExtern VARSIZE MessageName ';'",
"MessageName : Id",
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
#line 192 "xi-parse.bison"

int yyerror(char *mesg)
{
	cout << "Syntax error at line " << lineno << ": " << mesg << endl;
	return 0;
}

#line 248 "y.tab.c"
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
{ yyval.intval = FALSE; }
break;
case 14:
#line 62 "xi-parse.bison"
{ yyval.intval = TRUE; }
break;
case 15:
#line 66 "xi-parse.bison"
{
			Chare *c = new Chare(yyvsp[-1].strval, BOC, yyvsp[-3].intval) ;
			delete yyvsp[-1].strval;
			thismodule->AddChare(c) ;
		}
break;
case 17:
#line 75 "xi-parse.bison"
{
			Chare *c = new Chare(yyvsp[-1].strval, CHARE, yyvsp[-3].intval) ;
			delete yyvsp[-1].strval;
			thismodule->AddChare(c) ;
		}
break;
case 23:
#line 95 "xi-parse.bison"
{
			yyval.strval = NULL;
		}
break;
case 25:
#line 103 "xi-parse.bison"
{ yyval.intval = FALSE; }
break;
case 26:
#line 105 "xi-parse.bison"
{ yyval.intval = TRUE; }
break;
case 27:
#line 110 "xi-parse.bison"
{ yyval.intval = 0; }
break;
case 28:
#line 112 "xi-parse.bison"
{ yyval.intval = yyvsp[0].intval; }
break;
case 29:
#line 116 "xi-parse.bison"
{
			thismodule->chares->AddEntry(yyvsp[-5].strval, yyvsp[-3].strval, yyvsp[-8].intval, yyvsp[-7].strval, yyvsp[-1].intval) ;
			delete yyvsp[-5].strval; delete yyvsp[-3].strval;
		}
break;
case 31:
#line 126 "xi-parse.bison"
{
			Message *m = new Message(yyvsp[-1].strval, 0, 0, yyvsp[-3].intval) ;
			delete yyvsp[-1].strval;
			thismodule->AddMessage(m) ;
		}
break;
case 32:
#line 134 "xi-parse.bison"
{
			Message *m = new Message(yyvsp[-1].strval, 1, 0, yyvsp[-3].intval) ;
			delete yyvsp[-1].strval;
			thismodule->AddMessage(m) ;
		}
break;
case 33:
#line 142 "xi-parse.bison"
{
			Message *m = new Message(yyvsp[-1].strval, 1, 1, yyvsp[-3].intval) ;
			delete yyvsp[-1].strval;
			thismodule->AddMessage(m) ;
		}
break;
case 36:
#line 154 "xi-parse.bison"
{
			ReadOnly *r = new ReadOnly(yyvsp[-1].strval, yyvsp[-2].strval, 0) ;
			delete yyvsp[-2].strval;
			thismodule->AddReadOnly(r) ;
		}
break;
case 37:
#line 160 "xi-parse.bison"
{
			ReadOnly *r = new ReadOnly(yyvsp[-1].strval, yyvsp[-2].strval, 1) ;
			delete yyvsp[-2].strval;
			thismodule->AddReadOnly(r) ;
		}
break;
case 39:
#line 171 "xi-parse.bison"
{
			yyval.strval = strcat(strcpy(new char[2+strlen(yyvsp[-1].strval)], yyvsp[-1].strval), "*");
			delete yyvsp[-1].strval;
		}
break;
case 41:
#line 181 "xi-parse.bison"
{
			Table *t = new Table(yyvsp[-1].strval) ;
			delete yyvsp[-1].strval;
			thismodule->AddTable(t) ;
		}
break;
#line 496 "y.tab.c"
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
