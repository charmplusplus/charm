%{
#include "EToken.h"

/* Assume that the C-style comments are not present in the source */
char *mysltext;

#define Return(x) { mysltext = (char *) malloc(strlen(sltext)+1); \
                    strcpy(mysltext, sltext); return (x); }

%}

comment "//".*
identifier [a-zA-Z_][0-9a-zA-Z_]*
simple_escape [abfnrtv'"?\\]
octal_escape  [0-7]{1,3}
hex_escape "x"[0-9a-fA-F]+
escape_sequence [\\]({simple_escape}|{octal_escape}|{hex_escape})
c_char [^'\\\n]|{escape_sequence}
s_char [^"\\\n]|{escape_sequence}

ht [\011]
ff [\014]
vt [\013]
cret [\015]
hwhite [ ]|{ht}

exponent_part [eE][-+]?[0-9]+
fractional_constant ([0-9]*"."[0-9]+)|([0-9]+".")
floating_constant (({fractional_constant}{exponent_part}?)|([0-9]+{exponent_part}))[FfLl]?

integer_suffix_opt ([uU]?[lL]?)|([lL][uU])
decimal_constant [1-9][0-9]*{integer_suffix_opt}
octal_constant "0"[0-7]*{integer_suffix_opt}
hex_constant "0"[xX][0-9a-fA-F]+{integer_suffix_opt}
%%

{comment}		{}

{hwhite}+		{Return(WSPACE);}
({vt}|{cret}|{ff})+	{Return(WSPACE);}
({hwhite}|{vt}|{cret}|{ff})*"\n"	{Return(NEW_LINE);}


class			{Return(CLASS);}
sdagentry		{Return(SDAGENTRY);}
overlap			{Return(OVERLAP);}
when			{Return(WHEN);}
if			{Return(IF);}
else			{Return(ELSE);}
while			{Return(WHILE);}
forall			{Return(FORALL);}
for			{Return(FOR);}
atomic			{Return(ATOMIC);}

{identifier}		{Return(IDENT);}

"L"?[']{c_char}+[']	{Return(CHAR);}
"L"?["]{s_char}*["]	{Return(STRING);}

"("			{Return(LP);}
")"			{Return(RP);}
"["			{Return(LB);}
"]"			{Return(RB);}
"{"			{Return(LBRACE);}
"}"			{Return(RBRACE);}
"*"			{Return(STAR);}
":"			{Return(COLON);}
","			{Return(COMMA);}
";"			{Return(SEMICOLON);}

"#"			{Return(DEFAULT);}
"."			{Return(DEFAULT);}
"&"			{Return(DEFAULT);}
"+"			{Return(DEFAULT);}
"-"			{Return(DEFAULT);}
"~"			{Return(DEFAULT);}
"!"			{Return(DEFAULT);}
"/"			{Return(DEFAULT);}
"%"			{Return(DEFAULT);}
"<"			{Return(DEFAULT);}
">"			{Return(DEFAULT);}
"^"			{Return(DEFAULT);}
"|"			{Return(DEFAULT);}
"?"			{Return(DEFAULT);}
"="			{Return(DEFAULT);}

{decimal_constant}  	{Return(DEFAULT);}
{octal_constant}    	{Return(DEFAULT);}
{hex_constant}      	{Return(DEFAULT);}
{floating_constant} 	{Return(DEFAULT);}

%%

slwrap()
{
  return 1;
}

Unput(c)
int c;
{
  unput(c);
}
