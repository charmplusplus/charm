%{
#include <string.h>
#include "sdag/constructs/Constructs.h"
#include "xi-symbol.h"
#include <ctype.h>
using namespace xi;
#include "xi-grammar.tab.h"
#include <list>

/* Global Variables and Functions - used in grammar.y */
unsigned int yycolumn = 1;
unsigned int lineno = 1;
int in_bracket=0; /*Horrific hack to get "array length" code snippets (between []'s)*/
int in_braces=0; /*Horrific hack to get SDAG entry code snippets (between {}'s)*/
int in_int_expr=0;
std::list<Entry *> connectEntries;
/* Local to file */
unsigned char in_comment=0;
int search(char *s);
void update_position();
int count_newlines(const char *s);

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
{nonBracket}	{ if (in_bracket)  {Yval; update_position(); return Token(CPROGRAM);} else REJECT;}
{nonBraces}		{ if (in_braces)   {Yval; lineno += count_newlines(yytext); update_position(); return Token(CPROGRAM);} else REJECT;}
{nonIntExpr}	{ if (in_int_expr) {Yval; update_position(); return Token(CPROGRAM);} else REJECT;}
"//".*		{ update_position(); /* ignore single line comments */ }
"\r"		{ update_position(); /* ignore ^M characters for dos-unix compat */ }
"/*"		{ update_position(); in_comment = 1; /* Single line C-style comments */ }
"*/"		{ update_position(); in_comment = 0; }
"#if" 		{ update_position(); Return Token(HASHIF); }
"#ifdef" 	{ update_position(); Return Token(HASHIFDEF); }
"#endif" 	{ update_position(); in_comment = 0; /* comments */ }
{ws}		{ update_position(); /* ignore white space */ }
{nl}		{ yycolumn = 1; lineno++;}
{number}	{ update_position(); Yval; Return Token(NUMBER); }
{literal}	{ update_position(); Yval; Return Token(LITERAL); }
{name}		{ update_position(); Return Token(search(yytext)); }
.			{ update_position(); Return Token(yytext[0]); }
%%

/* Reserved word table */
struct rwentry rwtable[] = {
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
{  "appwork",	APPWORK},
{  "template",	TEMPLATE },
{  "class",	CLASS },
{  "include",	INCLUDE },
{  "sync",	SYNC },
{  "iget",       IGET },
{  "exclusive",	EXCLUSIVE },
{  "immediate",  IMMEDIATE },
{  "expedited",  SKIPSCHED },
{  "aggregate", AGGREGATE },
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
{  "when", 	WHEN },
{  "while", 	WHILE },
{  "for",	FOR },
{  "forall", 	FORALL },
{  "if", 	IF },
{  "else", 	ELSE },
{  "overlap", 	OVERLAP },
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

void update_position()
{
  yylloc.first_line = yylloc.last_line = lineno;
  yylloc.first_column = yycolumn;
  yylloc.last_column = yycolumn + yyleng-1;
  yycolumn += yyleng;
}

int count_newlines(const char *s)
{
  int count = 0;

  while (*s != '\0') {
    if (*s == '\n')
      count++;
    s++;
  }

  return count;
}

void scan_string(const char *str)
{
  yy_scan_string(str);
}

int yywrap() { return 1; }
