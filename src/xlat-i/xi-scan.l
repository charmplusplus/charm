%{
#include <string.h>
#include "xi-symbol.h"
#include "xi-grammar.tab.h"
#include <ctype.h>

/* Global Variables and Functions - used in grammar.y */
unsigned int lineno = 1;

/* Local to file */
static unsigned char in_comment=0;
int search(char *s);

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

name	({alpha}|[_])({alpha}|{digit}|[_])*

string1	\'[^\n']*\'
string2	\'[^\n']*(\'({ws}|{nl})*\'[^\n']*)+\'
string3	\"[^\n"]*\"
string4	\"[^\n"]*(\"({ws}|{nl})*\"[^\n"]*)+\"
string	{string1}|{string2}|{string3}|{string4}

int	[-+]?{digit}+

number	{int}[KkMm]?
expo	([eE][-+]?{digit}+)?
real1	{int}\.?{expo}
real2	[-+]?{digit}*\.{digit}+{expo}
real	{real1}|{real2}

literal	{number}|{real}|{string}

%%
"//".*		{ /* ignore single line comments */ }
"/*"		{ in_comment = 1; /* Single line C-style comments */ }
"*/"		{ in_comment = 0; }
{ws}		{ /* ignore white space */ }
{nl}		{ lineno++;}
{number}	{ yylval.strval = strcpy(new char[yyleng+1], yytext); 
		  Return Token(NUMBER); }
{literal}	{ yylval.strval = strcpy(new char[yyleng+1], yytext); 
		  Return Token(LITERAL); }
{name}		{ Return Token(search(yytext)); }
.		{ Return Token(yytext[0]); }
%%

struct rwtable {
  const char *res;	int tok;
};

/* Reserved word table */
struct rwtable rwtable[] = {
  "module",	MODULE,
  "mainmodule",	MAINMODULE,
  "chare",	CHARE,
  "group",	GROUP,
  "nodegroup",  NODEGROUP,
  "array",	ARRAY,
  "message",	MESSAGE,
  "extern",	EXTERN,
  "readonly",	READONLY,
  "stacksize",	STACKSIZE,
  "threaded",	THREADED,
  "migratable",	MIGRATABLE,
  "template",	TEMPLATE,
  "class",	CLASS,
  "sync",	SYNC,
  "exclusive",	EXCLUSIVE,
  "virtual",    VIRTUAL,
  "void",	VOID,
  "mainchare",	MAINCHARE,
  "packed",     PACKED,
  "varsize",    VARSIZE,
  "entry",      ENTRY,
  "int",        INT,
  "short",      SHORT,
  "long",       LONG,
  "char",       CHAR,
  "float",      FLOAT,
  "double",     DOUBLE,
  "unsigned",   UNSIGNED,
  "",		0,
};

int search(char *s)
{
  int i;
  for(i=0; rwtable[i].tok != 0; i++) {
    if(strcmp(rwtable[i].res, s) == 0)
      return(rwtable[i].tok);
  }
  yylval.strval = strcpy(new char[yyleng+1], s);
  return IDENT;
}

int yywrap() { return 1; }
