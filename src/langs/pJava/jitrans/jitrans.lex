
%{

#include <ctype.h>

int lineno = 1;

/* Local to file */
static int check_name(char *);

#ifdef yywrap
#undef yywrap
#endif
%}

ws	[ \t]+
nl	[\n]
alpha	[A-Za-z]
digit	[0-9]
comment "/*".*"*/"

name	({alpha})({alpha}|{digit}|[_])*

%%
"//".*		{ /* ignore single line comments */ }
{ws}		{ /* ignore white space */ }
{nl}		{ lineno++; }
{name}		{ return (check_name(yytext)); }
{comment}       { /* ignore comments */ }
"{"             { return LBRAC; }
"}"             { return RBRAC; }
"("             { return LPAR; }
")"             { return RPAR; }
";"             { return SEMI; }
.               { return (yytext[0]); }
%%

/* Distinguish between reserved words and identifiers. */
static int check_name(char *ss)
{
	int i;
	char *s;

	s = (char *) calloc(1+strlen(ss), sizeof(char));

	/* for case insensitivity, we convert to lower case */
	for(i=0; i<= strlen(ss); i++) s[i] = tolower(ss[i]);

	/* Is "s" a reserved word ? */
	if (strcmp(s, "class")==0)
	  return CLASS;
	else if (strcmp(s, "entry")==0)
	  return ENTRY;
	else if (strcmp(s, "group")==0)
	  return GROUP;
	/* ... otherwise it must be an identifier. */
	s = (char *) calloc(strlen(yytext), sizeof(char));
	yylval.strval = strcpy(s, yytext);
	return IDENTIFIER;
}

int yywrap() { return 1; }
