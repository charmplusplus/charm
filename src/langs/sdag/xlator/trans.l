%{
#include "EToken.h"

/* Assume that the C-style comments are not present in the source */

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

{comment}		{return WSPACE;}

{hwhite}+		{return WSPACE;}
({vt}|{cret}|{ff})+	{return WSPACE;}
({hwhite}|{vt}|{cret}|{ff})*"\n"	{return(NEW_LINE);}


class			{return(CLASS);}
sdagentry		{return(SDAGENTRY);}
overlap			{return(OVERLAP);}
when			{return(WHEN);}
if			{return(IF);}
else			{return(ELSE);}
while			{return(WHILE);}
forall			{return(FORALL);}
for			{return(FOR);}
atomic			{return(ATOMIC);}

{identifier}		{return(IDENT);}

"L"?[']{c_char}+[']	{return(CHAR);}
"L"?["]{s_char}*["]	{return(STRING);}

"("			{return(LP);}
")"			{return(RP);}
"["			{return(LB);}
"]"			{return(RB);}
"{"			{return(LBRACE);}
"}"			{return(RBRACE);}
"*"			{return(STAR);}
":"			{return(COLON);}
","			{return(COMMA);}
";"			{return(SEMICOLON);}

"#"			{return(DEFAULT);}
"."			{return(DEFAULT);}
"&"			{return(DEFAULT);}
"+"			{return(DEFAULT);}
"-"			{return(DEFAULT);}
"~"			{return(DEFAULT);}
"!"			{return(DEFAULT);}
"/"			{return(DEFAULT);}
"%"			{return(DEFAULT);}
"<"			{return(DEFAULT);}
">"			{return(DEFAULT);}
"^"			{return(DEFAULT);}
"|"			{return(DEFAULT);}
"?"			{return(DEFAULT);}
"="			{return(DEFAULT);}

{decimal_constant}  	{return(DEFAULT);}
{octal_constant}    	{return(DEFAULT);}
{hex_constant}      	{return(DEFAULT);}
{floating_constant} 	{return(DEFAULT);}

%%

yywrap()
{
  return 1;
}

Unput(c)
int c;
{
  unput(c);
}
