
%{

#include "xi-symbol.h"
#include "xi-parse.tab.h"
#include <ctype.h>

/* Global Variables and Functions - used in parse.y */
unsigned int lineno = 1;
// int yylex()

/* Local to file */
static unsigned char in_comment=0;
int binsearch(char *s, int lb, int ub);
static int check_name(char *);

/* We return Tokens only when not in a comment. */
#define Return if (!in_comment) return
#define	Token(x) x

#ifdef yywrap
#undef yywrap
#endif
%}

ws	[ \t]+
nl	[\n]
alpha	[A-Za-z]
digit	[0-9]

name	({alpha})({alpha}|{digit}|[_])*

string1	\'[^\n']*\'
string2	\'[^\n']*(\'\'[^\n']*)+\'
string3	\"[^\n"]*\"
string	{string1}|{string2}

int	[-+]?{digit}+

expo	([eE][-+]?{digit}+)?
real1	{int}\.?{expo}
real2	[-+]?{digit}*\.{digit}+{expo}
real	{real1}|{real2}

bool	1|0

%%
"//".*		{ /* ignore single line comments */ }
"/*"		{ in_comment = 1; /* Single line C-style comments */ }
"*/"		{ in_comment = 0; }
{ws}		{ /* ignore white space */ }
{nl}		{ lineno++; /* Return Token(NL); */ }
{int}		{ yylval.intval = (atoi(yytext)); Return Token(NUMBER); }
{name}		{ Return Token(check_name(yytext)); }
.		{ Return Token(yytext[0]); }
%%

/* {nl}/{nl}	{ lineno++; } */
struct rwtable {
	char *s;	int tok;
};

/* Reserved word table */
struct rwtable rwtable[] = {
	"",		12,
/* MUST BE IN SORTED ORDER */
	"boc",		BOC,
	"chare",	CHARE,
	"entry",	ENTRY,
	"extern",	EXTERN,
	"group",	BOC,
	"message",	MESSAGE,
	"packmessage",	PACKMESSAGE,
	"readonly",	READONLY,
	"stacksize",	STACKSIZE,
	"table",	TABLE,
	"threaded",	THREADED,
        "varsize",      VARSIZE,
/* MAKE SURE TO UPDATE THE NUMBER OF ENTRIES ABOVE */
	"",		0,
};

int binsearch(char *s, int lb, int ub)
{
	int mid = (lb+ub)/2;
	int result = 0;

	if (lb>ub) return 0;	/* not found */
	else if ((result = strcmp(s, rwtable[mid].s))==0)
		return mid; /* found */
	else if (result<0) return binsearch(s, lb, mid-1);	/* lower half */
	else return binsearch(s, mid+1, ub);	/* upper half */
}

/* Distinguish between reserved words and identifiers. */
static int check_name(char *ss)
{
	int i;
	char *s = new char[1+strlen(ss)];

	/* for case insensitivity, we convert to lower case */
	for(i=0; i<= strlen(ss); i++) s[i] = tolower(ss[i]);

	/* Is "s" a reserved word ? */
	if ( (i=binsearch(s, 1, rwtable[0].tok)) )
		{	delete s;
			return rwtable[i].tok;
		}

	/* ... otherwise it must be an identifier. */
	yylval.strval = strcpy(new char[yyleng+1], yytext);
/*
	yylval = createASTnode( AIDENTIFIER, lookup(s),
			NULL, NULL, NULL, NULL);
*/

	delete s;

	return IDENTIFIER;
}

int yywrap() { return 1; }
