%{

/*  Copyright (C) 1989-1991 James A. Roskind, All rights reserved.
    This lexer description was written by James A.  Roskind.  Copying
    of  this  file, as a whole, is permitted providing this notice is
    intact  and  applicable   in   all   complete   copies.    Direct
    translations  as a whole to other lexer generator input languages
    (or lexical description languages)  is  permitted  provided  that
    this  notice  is  intact and applicable in all such copies, along
    with a disclaimer that  the  contents  are  a  translation.   The
    reproduction  of derived files or text, such as modified versions
    of this file, or the output of scanner generators, is  permitted,
    provided   the  resulting  work  includes  the  copyright  notice
    "Portions Copyright (c) 1989, 1990 James  A.   Roskind".  Derived
    products  must  also  provide  the notice "Portions Copyright (c)
    1989, 1990 James A.  Roskind" in  a  manner  appropriate  to  the
    utility,   and  in  keeping  with  copyright  law  (e.g.:  EITHER
    displayed when first invoked/executed; OR displayed  continuously
    on  display terminal; OR via placement in the object code in form
    readable in a printout, with or near the title of the work, or at
    the end of the file).  No royalties, licenses or  commissions  of
    any  kind  are  required  to copy this file, its translations, or
    derivative products, when the copies are made in compliance  with
    this  notice.  Persons  or  corporations  that  do make copies in
    compliance  with  this  notice  may  charge  whatever  price   is
    agreeable  to  a buyer, for such copies or derivative works. THIS
    FILE IS PROVIDED ``AS IS'' AND WITHOUT  ANY  EXPRESS  OR  IMPLIED
    WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTIES
    OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

    James A. Roskind
    Independent Consultant
    516 Latania Palm Drive
    Indialantic FL, 32903
    (407)729-4348
    jar@hq.ileaf.com
    or ...!uunet!leafusa!jar

    ---end of copyright notice---

*/

/*

Comment  removal  must  be done during the lexing, as context (such as
enclosure in string literals) must be  observed.   For  this  cut-down
lexer,  we  will  assume that comments have been removed (don't assume
this if you are writing a compiler or browser!).


For each IDENTIFIER like string that is found,  there  are  several
distinct interpretations that can be applied:

1)  The  preprocessor  may  interpret  the  string as a "keyword" in a
directive (eg: "pragma" or "include", "defined").

2) The parser may interpret the string as a keyword. (eg: "int").

3) Both parser and preprocessor may interpret the string as a  keyword
(eg: "if").

Since  this  file  is based on source that actually lexically analyses
text for both preprocessing and parsing, macro definitions  were  used
throughout.   The macro definitions supplied here have been customized
to a C++ parse only, and  all  preprocessor  keywords  are  passed  as
IDENTIFIER  or  TYPEDEFname.   Also, since there is no symbol table to
interrogate to decide whether a string  is  a  TYPEDEFname,  I  simply
assume  that  any  identifier beginning with an upper case letter is a
TYPEDEFname.  This hack  should  allow  you  to  check  out  how  code
segments  are  parsed  using my grammar.  Unfortunately, if you really
want to parse major league code, you have to write a symbol table, and
maintain appropriate scoping information.

*/


/* Included code before lex code */
/*************** Includes and Defines *****************************/


char *CurrentIdent, *CurrentConst ;
int FoundLBrace=0, FoundRBrace=0 ;

#define YYSTYPE char *
#include "xp-t.tab.h" /* YACC generated definitions based on C++ grammar */
#include "xp-lexer.h"
#include "xp-extn.h"
extern YYSTYPE yylval;

#define WHITE_RETURN(x) if ( shouldprint ) strcat(OutBuf,prevtoken) ; \
			strcpy(prevtoken,x);

/*
#define NEW_LINE_RETURN() CurrentLine++; \
			  if ( shouldprint ) strcat(OutBuf,prevtoken); \
			  strcpy(prevtoken,yytext);
*/
			  

#define PA_KEYWORD_RETURN(x)   RETURN_VAL(x)  /* standard C PArser Keyword */
#define CPP_KEYWORD_RETURN(x)  PA_KEYWORD_RETURN(x)  /* C++ keyword */
#define PPPA_KEYWORD_RETURN(x) RETURN_VAL(x)  /* both PreProcessor and PArser keyword */
#define PP_KEYWORD_RETURN(x)   IDENTIFIER_RETURN()

/*
#define IDENTIFIER_RETURN() RETURN_VAL(isaTYPE(yytext)?TYPEDEFname:IDENTIFIER)
*/

#define IDENTIFIER_RETURN() CurrentIdent = (char *)malloc((strlen(yytext)+1)*sizeof(char)); \
		      	    strcpy(CurrentIdent,yytext);  \
			    yylval = CurrentIdent ; \
			    if (shouldprint) strcat(OutBuf,prevtoken); \
			    strcpy(prevtoken,yytext) ; \
			    if ( FoundLBrace ) {PushStack() ;FoundLBrace=0;}\
			    else if ( FoundRBrace ){PopStack();FoundRBrace=0;}\
			    return(isaTYPE(yytext)?TYPEDEFname:IDENTIFIER);

#define PPOP_RETURN(x)       RETURN_VAL((int)*yytext) /* PreProcess and Parser operator */
#define NAMED_PPOP_RETURN(x) /* error: PreProcessor ONLY operator;  Do nothing */
#define ASCIIOP_RETURN(x)    RETURN_VAL((int)*yytext) /* a single character operator */
#define LBRACE_RETURN(x)     CurrentScope++ ; \
			     RETURN_VAL((int)*yytext) ;
#define RBRACE_RETURN(x)     CurrentScope-- ; \
			     RETURN_VAL((int)*yytext) ;

#define NAMEDOP_RETURN(x)    RETURN_VAL(x)            /* a multichar operator, with a name */

/* #define NUMERICAL_RETURN(x) RETURN_VAL(x)         * some sort of constant */

#define NUMERICAL_RETURN(x) CurrentConst = (char *)malloc((strlen(yytext)+1)*sizeof(char)); \
		      	    strcpy(CurrentConst,yytext);  \
			    yylval = CurrentConst ; \
			    if (shouldprint) strcat(OutBuf,prevtoken); \
			    strcpy(prevtoken,yytext) ; \
			    if ( FoundLBrace ) {PushStack() ;FoundLBrace=0;}\
			    else if ( FoundRBrace ){PopStack();FoundRBrace=0;}\
			    return(x);
			    

#define LITERAL_RETURN(x)   RETURN_VAL(x)            /* a string literal */

#define RETURN_VAL(x) 	yylval = yytext; \
			if ( shouldprint ) strcat(OutBuf,prevtoken); \
			strcpy(prevtoken,yytext) ; \
			if ( FoundLBrace ) {PushStack() ;FoundLBrace=0;}\
			else if ( FoundRBrace ){PopStack();FoundRBrace=0;}\
		        if ( prevtoken[0] == '{' ) FoundLBrace=1; \
		        else if ( prevtoken[0] == '}' ) FoundRBrace=1; \
			return(x);

#define RETURN_VAL_NOPRINT(x) 	yylval = yytext; return(x);

#define CHARM_KEYWORD_RETURN(x)	RETURN_VAL_NOPRINT(x)

#ifdef yywrap
#undef yywrap
#endif

%}

%p 4000
%e 1500

comment "//".*
TabSpace [ \t]*
ASTRNG ([^"\\\n]|\\(['"?\\abfnrtv\n]|[0-7]{1,3}|[xX][0-9a-fA-F]{1,3}))*

identifier [a-zA-Z_][0-9a-zA-Z_]*

exponent_part [eE][-+]?[0-9]+
fractional_constant ([0-9]*"."[0-9]+)|([0-9]+".")
floating_constant (({fractional_constant}{exponent_part}?)|([0-9]+{exponent_part}))[FfLl]?

integer_suffix_opt ([uU]?[lL]?)|([lL][uU])
decimal_constant [1-9][0-9]*{integer_suffix_opt}
octal_constant "0"[0-7]*{integer_suffix_opt}
hex_constant "0"[xX][0-9a-fA-F]+{integer_suffix_opt}

simple_escape [abfnrtv'"?\\]
octal_escape  [0-7]{1,3}
hex_escape "x"[0-9a-fA-F]+

escape_sequence [\\]({simple_escape}|{octal_escape}|{hex_escape})
c_char [^'\\\n]|{escape_sequence}
s_char [^"\\\n]|{escape_sequence}


h_tab [\011]
form_feed [\014]
v_tab [\013]
c_return [\015]

horizontal_white [ ]|{h_tab}



%%

{comment}		{ }

{horizontal_white}+     {
			WHITE_RETURN(" ");
			}

({v_tab}|{c_return}|{form_feed})+   {
			WHITE_RETURN(" ");
			}


({horizontal_white}|{v_tab}|{c_return}|{form_feed})*"\n"   {
			NEW_LINE_RETURN();
			}

chare		    {CHARM_KEYWORD_RETURN(CHARE);}
accumulator	    {CHARM_KEYWORD_RETURN(ACCUMULATOR);}
monotonic	    {CHARM_KEYWORD_RETURN(MONOTONIC);}
readonly	    {CHARM_KEYWORD_RETURN(READONLY);}
writeonce	    {CHARM_KEYWORD_RETURN(WRITEONCE);}

message		    {CHARM_KEYWORD_RETURN(MESSAGE);}
handle		    {	if ( CheckCharmName() ) {
				CHARM_KEYWORD_RETURN(HANDLE);
			}
			else {
				IDENTIFIER_RETURN() ;
			}
		    }
group		    {	if ( CheckCharmName() ) {
				CHARM_KEYWORD_RETURN(GROUP);
				/* handle and group are processed same way */
			}
			else {
				IDENTIFIER_RETURN() ;
			}
		    }
entry		    {if ( shouldprint ) strcat(OutBuf,prevtoken); 
		     strcpy(prevtoken,"public") ;	
		     CHARM_KEYWORD_RETURN(ENTRY);}
"=>"		    {CHARM_KEYWORD_RETURN(DOUBLEARROW);}
"="{TabSpace}">"    {CHARM_KEYWORD_RETURN(DOUBLEARROW);}
"ALL"	    	    {CHARM_KEYWORD_RETURN(ALL_NODES);}
"LOCAL"	    	    {CHARM_KEYWORD_RETURN(LOCAL);}
VARSIZE	    	    { FoundVarSize = TRUE ; }
newchare	    {CPP_KEYWORD_RETURN(NEWCHARE);}
newgroup	    {CPP_KEYWORD_RETURN(NEWGROUP);}
newaccumulator	    {CPP_KEYWORD_RETURN(NEW);}
newmonotonic	    {CPP_KEYWORD_RETURN(NEW);}


auto                {PA_KEYWORD_RETURN(AUTO);}
break               {PA_KEYWORD_RETURN(BREAK);}
case                {PA_KEYWORD_RETURN(CASE);}
char                {PA_KEYWORD_RETURN(CHAR);}
const               {PA_KEYWORD_RETURN(CONST);}
continue            {PA_KEYWORD_RETURN(CONTINUE);}
default             {PA_KEYWORD_RETURN(DEFAULT);}
define              {PP_KEYWORD_RETURN(DEFINE);}
defined             {PP_KEYWORD_RETURN(OPDEFINED);}
do                  {PA_KEYWORD_RETURN(DO);}
double              {PA_KEYWORD_RETURN(DOUBLE);}
elif                {PP_KEYWORD_RETURN(ELIF);}
else                {PPPA_KEYWORD_RETURN(ELSE);}
endif               {PP_KEYWORD_RETURN(ENDIF);}
enum                {PA_KEYWORD_RETURN(ENUM);}
error               {PP_KEYWORD_RETURN(ERROR);}
extern              {PA_KEYWORD_RETURN(EXTERN);}
float               {PA_KEYWORD_RETURN(FLOAT);}
for                 {PA_KEYWORD_RETURN(FOR);}
goto                {PA_KEYWORD_RETURN(GOTO);}
if                  {PPPA_KEYWORD_RETURN(IF);}
ifdef               {PP_KEYWORD_RETURN(IFDEF);}
ifndef              {PP_KEYWORD_RETURN(IFNDEF);}
include             {PP_KEYWORD_RETURN(INCLUDE); }
int                 {PA_KEYWORD_RETURN(INT);}
line                {PP_KEYWORD_RETURN(LINE);}
long                {PA_KEYWORD_RETURN(LONG);}
pragma              {PP_KEYWORD_RETURN(PRAGMA);}
ptrdiff_t           {if (ptrdiff_is_predefined)
                       { PA_KEYWORD_RETURN(PTRDIFF_TOKEN); }
                     else { IDENTIFIER_RETURN(); } }
register            {PA_KEYWORD_RETURN(REGISTER);}
return              {PA_KEYWORD_RETURN(RETURN);}
short               {PA_KEYWORD_RETURN(SHORT);}
signed              {PA_KEYWORD_RETURN(SIGNED);}
sizeof              {PA_KEYWORD_RETURN(SIZEOF);}
static              {PA_KEYWORD_RETURN(STATIC);}
struct              {PA_KEYWORD_RETURN(STRUCT);}
switch              {PA_KEYWORD_RETURN(SWITCH);}
typedef             {PA_KEYWORD_RETURN(TYPEDEF);}
undef               {PP_KEYWORD_RETURN(UNDEF);}
union               {PA_KEYWORD_RETURN(UNION);}
unsigned            {PA_KEYWORD_RETURN(UNSIGNED);}
void                {PA_KEYWORD_RETURN(VOID);}
volatile            {PA_KEYWORD_RETURN(VOLATILE);}
while               {PA_KEYWORD_RETURN(WHILE);}
wchar_t             { if (wchar_is_predefined) {
                         PA_KEYWORD_RETURN(WCHAR_TOKEN);
                      } else {
                         IDENTIFIER_RETURN();
                    }}
__wchar_t           { PA_KEYWORD_RETURN(__WCHAR_TOKEN); }

class               {CPP_KEYWORD_RETURN(CLASS);}
delete              {CPP_KEYWORD_RETURN(DELETE);}
friend              {CPP_KEYWORD_RETURN(FRIEND);}
inline              {CPP_KEYWORD_RETURN(INLINE);}
__inline__          {CPP_KEYWORD_RETURN(UNDERSCORE_INLINE);}
new                 {CPP_KEYWORD_RETURN(NEW);}
operator            {CPP_KEYWORD_RETURN(OPERATOR);}
overload            {CPP_KEYWORD_RETURN(OVERLOAD);}
protected           {CPP_KEYWORD_RETURN(PROTECTED);}
private             {CPP_KEYWORD_RETURN(PRIVATE);}
public              {CPP_KEYWORD_RETURN(PUBLIC);}
this                {CPP_KEYWORD_RETURN(THIS);}
virtual             {CPP_KEYWORD_RETURN(VIRTUAL);}



"CFunctionNameToRef"{TabSpace}?"("{TabSpace}?{identifier}{TabSpace}?")"  {
			/* Find the identifier */
			char str[128] ;
			char *ident = str ;
			char *ptr1 = strchr(yytext,'(') ;
			ptr1++ ;
			while ( *ptr1==' ' || *ptr1=='\t' )
				ptr1++ ;			
			while ( *ptr1!=' ' && *ptr1!='\t' && *ptr1!=')' )
				*(ident++) = *(ptr1++) ;
			*ident = '\0' ;

			strcpy(yytext,"_CK_func_") ;
			strcat(yytext,str) ;

                     	IDENTIFIER_RETURN();
			}



{identifier}        { IDENTIFIER_RETURN(); }

{decimal_constant}  {NUMERICAL_RETURN(INTEGERconstant);}
{octal_constant}    {NUMERICAL_RETURN(OCTALconstant);}
{hex_constant}      {NUMERICAL_RETURN(HEXconstant);}
{floating_constant} {NUMERICAL_RETURN(FLOATINGconstant);}


"L"?[']{c_char}+[']     {
			NUMERICAL_RETURN(CHARACTERconstant);
			}


"L"?["]{s_char}*["]     {
			LITERAL_RETURN(STRINGliteral);}




"("                  {PPOP_RETURN(LP);}
")"                  {PPOP_RETURN(RP);}
","                  {PPOP_RETURN(COMMA);}
"#"                  {NAMED_PPOP_RETURN('#') ;}
"##"                 {NAMED_PPOP_RETURN(POUNDPOUND);}

"{"                  { LBRACE_RETURN(LC); }
"}"                  { RBRACE_RETURN(RC); }

"["                  {ASCIIOP_RETURN(LB);}
"]"                  {ASCIIOP_RETURN(RB);}
"."                  {ASCIIOP_RETURN(DOT);}
"&"                  {ASCIIOP_RETURN(AND);}
"*"                  {ASCIIOP_RETURN(STAR);}
"+"                  {ASCIIOP_RETURN(PLUS);}
"-"                  {ASCIIOP_RETURN(MINUS);}
"~"                  {ASCIIOP_RETURN(NEGATE);}
"!"                  {ASCIIOP_RETURN(NOT);}
"/"                  {ASCIIOP_RETURN(DIV);}
"%"                  {ASCIIOP_RETURN(MOD);}
"<"                  {ASCIIOP_RETURN(LT);}
">"                  {ASCIIOP_RETURN(GT);}
"^"                  {ASCIIOP_RETURN(XOR);}
"|"                  {ASCIIOP_RETURN(PIPE);}
"?"                  {ASCIIOP_RETURN(QUESTION);}
":"                  {ASCIIOP_RETURN(COLON);}
";"                  {ASCIIOP_RETURN(SEMICOLON);}
"="                  {ASCIIOP_RETURN(ASSIGN);}

".*"                 {NAMEDOP_RETURN(DOTstar);}
"::"                 {NAMEDOP_RETURN(CLCL);}
"->"                 {NAMEDOP_RETURN(ARROW);}
"->*"                {NAMEDOP_RETURN(ARROWstar);}
"++"                 {NAMEDOP_RETURN(ICR);}
"--"                 {NAMEDOP_RETURN(DECR);}
"<<"                 {NAMEDOP_RETURN(LSHIFT);}
">>"                 {NAMEDOP_RETURN(RSHIFT);}
"<="                 {NAMEDOP_RETURN(LE);}
">="                 {NAMEDOP_RETURN(GE);}
"=="                 {NAMEDOP_RETURN(EQ);}
"!="                 {NAMEDOP_RETURN(NE);}
"&&"                 {NAMEDOP_RETURN(ANDAND);}
"||"                 {NAMEDOP_RETURN(OROR);}
"*="                 {NAMEDOP_RETURN(MULTassign);}
"/="                 {NAMEDOP_RETURN(DIVassign);}
"%="                 {NAMEDOP_RETURN(MODassign);}
"+="                 {NAMEDOP_RETURN(PLUSassign);}
"-="                 {NAMEDOP_RETURN(MINUSassign);}
"<<="                {NAMEDOP_RETURN(LSassign);}
">>="                {NAMEDOP_RETURN(RSassign);}
"&="                 {NAMEDOP_RETURN(ANDassign);}
"^="                 {NAMEDOP_RETURN(ERassign);}
"|="                 {NAMEDOP_RETURN(ORassign);}
"..."                {NAMEDOP_RETURN(ELLIPSIS);}

"#"{TabSpace}("line")?{TabSpace}[0-9]+{TabSpace}\"{ASTRNG}\"({TabSpace}[0-9]+)*                                                { int i=0,j=0;
                                                  char temp[MAX_NAME_LENGTH];

						  strcat(OutBuf,prevtoken) ;
						  strcpy(prevtoken,"") ;
						  strcat(OutBuf,yytext) ;
						  FLUSHBUF() ;

                                                  while ((yytext[i]<'0')||
                                                         (yytext[i]>'9'))
                                                        i++;
                                                  while ((yytext[i]>='0') &&
                                                         (yytext[i]<='9'))
                                                        temp[j++]=yytext[i++];
                                                  temp[j]='\0';
                                                  CurrentLine = atoi(temp)-1;
                                                  while (yytext[i]!='\"')
                                                        i++;

					  	  /* remove double quote */
						  j = yyleng-1 ;
						  while ( yytext[j] != '"' )
							j-- ;
                                                  yytext[j]='\0';
                                                  strcpy(CurrentFileName,
                                                                yytext+i+1);
                                                }

"#"{TabSpace}?("line")?{TabSpace}[0-9]+         { /* #line used by SP */
						
						int i=0,j=0;     
                                                char temp[MAX_NAME_LENGTH];
						
						strcat(OutBuf,prevtoken) ;
						strcpy(prevtoken,"") ;
						strcat(OutBuf,yytext) ;
						FLUSHBUF() ;

						/* Skip to the line number */
						while ((yytext[i]<'0')||
						       (yytext[i]>'9'))
						     i++;
						
						/* Copy the line number */
						while ((yytext[i]>='0') &&
						       (yytext[i]<='9'))
						     temp[j++]=yytext[i++];
						temp[j]='\0';
						
						/* Now place it in 
						   'CurrentLine */
						CurrentLine = atoi(temp)-1;
					   }


"#"{TabSpace}?("pragma"){TabSpace}("when").*	{ /* #pragma when ... */

		/* This pragma is used to specify dependences between EPs */
		    char * rest ;
		    if ( MakeGraph ) {
			rest = strchr(yytext,'n') ;
			rest++ ;
			fprintf(graphfile,"WHEN %s %s : %s\n",CurrentChare,
						CurrentEP, rest) ;
		    }
		}


"#"{TabSpace}?("pragma").*	{ /* #pragma used by G++: copy to output */
					strcat(OutBuf,prevtoken) ;
					strcpy(prevtoken,"") ;
					strcat(OutBuf,yytext) ;
					FLUSHBUF() ;
				}

"#"{TabSpace}?("ident").*	{ /* #ident used by G++: copy to output */
					strcat(OutBuf,prevtoken) ;
					strcpy(prevtoken,"") ;
					strcat(OutBuf,yytext) ;
					FLUSHBUF() ;
				}

"#"{TabSpace}?("file").*	{ /* #file stuff used in nCUBE CC */
					strcat(OutBuf,prevtoken) ;
					strcpy(prevtoken,"") ;
					strcat(OutBuf,yytext) ;
					FLUSHBUF() ;
				}

%%

yywrap() { return(1); }

/* I won't bother to provide any error recovery. I won't  even  handle
unknown characters */

/*******************************************************************
int isaTYPE(string)
char * string;
{
    *  We  should  really  be  maintaining  a  symbol  table,  and be
    carefully keeping track of what the current scope is  (or  in  the
    case  of  "rescoped"  stuff,  what  scope  to  look in). Since the
    grammar is not annotated with  actions  to  track  transitions  to
    various  scopes,  and  there  is no symbol table, we will supply a
    hack to allow folks to test  the  grammar  out.   THIS  IS  NOT  A
    COMPLETE IMPLEMENTATION!!!! *

    if ( strncmp(string,"NULL",4) == 0 )
	return 0 ;

    return ('A' <= string[0] && 'Z' >= string[0]);
}
********************************************************************/

int isaTYPE(string)
char *string ;
{
	int i ;

	if ( StructScope )
		return FALSE ;

	if ( !FoundGlobalScope || GlobalStack == NULL ) {
		for ( i=TotalSyms-1; i>=0; i-- ) {  /* from inner to outer */
			if ( strcmp(string,SymTable[i].name) == 0 ) {
				strcpy(CurrentTypedef,string) ;
				return TRUE ;
			}
		}
	}
	else {
		for ( i=0; i<GlobalStack->TotalSyms; i++ ) {  
			/* search global state */
			if ( strcmp(string,SymTable[i].name) == 0 ) {
				strcpy(CurrentTypedef,string) ;
				return TRUE ;
			}
		}
                FoundGlobalScope = 0 ;
	}
	strcpy(CurrentTypedef,"") ;
	return FALSE ;
}

NEW_LINE_RETURN() 
{
	CurrentLine++; 
	if ( shouldprint ) strcat(OutBuf,prevtoken); 
	strcpy(prevtoken,yytext);
}


