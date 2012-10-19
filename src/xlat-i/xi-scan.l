%{
#include <string.h>
#include "xi-symbol.h"
#include <ctype.h>
using namespace xi;
#include "xi-grammar.tab.h"
#include <list>

/* Global Variables and Functions - used in grammar.y */
unsigned int lineno = 1;
int in_bracket=0; /*Horrific hack to get "array length" code snippets (between []'s)*/
int in_braces=0; /*Horrific hack to get SDAG entry code snippets (between {}'s)*/
int in_int_expr=0;
std::list<Entry *> connectEntries;
/* Local to file */
unsigned char in_comment=0;
int search(char *s);
int count_newlines(char *s);

/* We return Tokens only when not in a comment. */
#define Return if (!in_comment) return
#define	Token(x) x
#define Yval yylval.strval = strcpy(new char[yyleng+1], yytext)

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

nonBracket [^][]*
nonBraces [^{}]*
nonIntExpr [^();,:]*

%%
{nonBracket}	{ if (in_bracket) {Yval;return Token(CPROGRAM);} else REJECT;}
{nonBraces}	{ if (in_braces) {Yval; lineno += count_newlines(yytext); return Token(CPROGRAM);} else REJECT;}
{nonIntExpr}	{ if (in_int_expr) {Yval;return Token(CPROGRAM);} else REJECT;}
"//".*		{ /* ignore single line comments */ }
"\r"		{ /* ignore ^M characters for dos-unix compat */ }
"/*"		{ in_comment = 1; /* Single line C-style comments */ }
"*/"		{ in_comment = 0; }
"#if" 		{ Return Token(HASHIF); }
"#ifdef" 	{ Return Token(HASHIFDEF); }
"#endif" 	{ in_comment = 0; /* comments */ }
{ws}		{ /* ignore white space */ }
{nl}		{ lineno++;}
{number}	{ Yval; Return Token(NUMBER); }
{literal}	{ Yval; Return Token(LITERAL); }
{name}		{ Return Token(search(yytext)); }
.		{ Return Token(yytext[0]); }
%%

struct rwtable {
  const char *res;	int tok;
};

/* Reserved word table */
struct rwtable rwtable[] = {
{  "module",	MODULE },
{  "mainmodule",	MAINMODULE },
{  "chare",	CHARE },
{  "group",	GROUP },
{  "nodegroup",  NODEGROUP },
{  "array",	ARRAY },
{  "message",	MESSAGE },
{  "conditional",CONDITIONAL },
{  "extern",	EXTERN },
{  "initcall",	INITCALL },
{  "initnode",	INITNODE },
{  "initproc",	INITPROC },
{  "readonly",	READONLY },
{  "stacksize",	STACKSIZE },
{  "threaded",	THREADED },
{  "migratable",	MIGRATABLE },
{  "PUPable",	PUPABLE },
{  "pupable",	PUPABLE },
{  "createhere",	CREATEHERE },
{  "createhome",	CREATEHOME },
{  "nokeep",	NOKEEP },
{  "notrace",	NOTRACE },
{  "template",	TEMPLATE },
{  "class",	CLASS },
{  "include",	INCLUDE },
{  "sync",	SYNC },
{  "iget",       IGET },
{  "exclusive",	EXCLUSIVE },
{  "immediate",  IMMEDIATE },
{  "expedited",  SKIPSCHED },
{  "inline",  	INLINE },
{  "local",      LOCAL },
{  "virtual",    VIRTUAL },
{  "mainchare",	MAINCHARE },
{  "packed",     PACKED },
{  "varsize",    VARSIZE },
{  "entry",      ENTRY },
{  "int",        INT },
{  "short",      SHORT },
{  "long",       LONG },
{  "char",       CHAR },
{  "float",      FLOAT },
{  "double",     DOUBLE },
{  "unsigned",   UNSIGNED },
{  "void",	VOID },
{  "const",	CONST },
{  "atomic", 	ATOMIC },
{  "serial",    ATOMIC },
{  "forward", 	FORWARD },
{  "when", 	WHEN },
{  "while", 	WHILE },
{  "for",	FOR },
{  "forall", 	FORALL },
{  "if", 	IF },
{  "else", 	ELSE },
{  "overlap", 	OVERLAP },
{  "connect",    CONNECT },
{  "publishes",  PUBLISHES },
{  "python",     PYTHON },
{  "namespace",  NAMESPACE },
{  "using",      USING },
{  "accel",      ACCEL },
{  "readwrite",  READWRITE },
{  "writeonly",  WRITEONLY },
{  "accelblock", ACCELBLOCK },
{  "memcritical", MEMCRITICAL },
{  "reductiontarget", REDUCTIONTARGET },
{  "case",       CASE },
{  "",		0 }
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

// Oh my $DEITY...
int count_newlines(char *s) {
    int count = 0;

    while (*s != '\0') {
        if (*s == '\n') count++;
        s++;
    }

    return count;
}

int yywrap() { return 1; }
