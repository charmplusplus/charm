 
%{
#define YYSTYPE YSNPTR
#include "xl-lex.h"
#include "xl-yacc.tab.h"
#include "xl-sym.h"
#include <string.h>

extern YSNPTR yylval;
extern int MsgToStructFlag;
int CurrentInputLineNo=1;
int SavedLineNo;
int CurrentOutputLineNo=0;
char CurrentFileName[2*FILENAMELENGTH];
char SavedFileName[2*FILENAMELENGTH];
int OutputLineLength=0;
extern FILE *outfile;

extern SYMTABPTR GlobalModuleSearch();
extern char token[];
int PrevToken=0;

#define CREATE 1
#define NOCREATE 0

#ifdef yywrap
#undef yywrap
#endif
%}


%e 1500
%p 3900

EXP    ([eE][+-]?[0-9]+)
FS     [flFL]
IS     ([uU][lL]?|[lL][uU]?)
ASTRNG ([^"\\\n]|\\(['"?\\abfnrtv\n]|[0-7]{1,3}|[xX][0-9a-fA-F]{1,3}))*
WhiteSpace	[ \t]*
ID     [_a-zA-Z][_a-zA-Z0-9]*
WS     [ \t\n]*

%%
"@"		{ strcpy(token,yytext);return(PrevToken=AT); }
"?"		{ mywriteoutput("?",NOCREATE);strcpy(token,yytext);return(PrevToken=QUESTION); }
":"		{ strcpy(token,yytext);return(PrevToken=COLON); }
"|"		{ mywriteoutput("|",NOCREATE);strcpy(token,yytext);return(PrevToken=OR); }
"&"		{ mywriteoutput("&",NOCREATE);strcpy(token,yytext);return(PrevToken=AND); }
"^"		{ mywriteoutput("^",NOCREATE);strcpy(token,yytext);return(PrevToken=HAT); }
"+"		{ mywriteoutput("+",NOCREATE);strcpy(token,yytext);return(PrevToken=PLUS); }
"-"		{ mywriteoutput("-",NOCREATE);strcpy(token,yytext);return(PrevToken=MINUS); }
"*"		{ mywriteoutput("*",NOCREATE);strcpy(token,yytext);return(PrevToken=MULT); }
"/"		{ mywriteoutput("/",NOCREATE);strcpy(token,yytext);return(PrevToken=DIV); }
"%"		{ mywriteoutput("%",NOCREATE);strcpy(token,yytext);return(PrevToken=MOD); }
"("		{ mywriteoutput("(",NOCREATE);strcpy(token,yytext);return(PrevToken=L_PAREN); }
")"		{ mywriteoutput(")",NOCREATE);strcpy(token,yytext);return(PrevToken=R_PAREN); }
"["		{ mywriteoutput("[",NOCREATE);strcpy(token,yytext);return(PrevToken=L_SQUARE); }
"]"		{ mywriteoutput("]",NOCREATE);strcpy(token,yytext);return(PrevToken=R_SQUARE); }
"."		{ mywriteoutput(".",NOCREATE);strcpy(token,yytext);return(PrevToken=DOT); }
"!"		{ mywriteoutput("!",NOCREATE);strcpy(token,yytext);return(PrevToken=EXCLAIM); }
"~"		{ mywriteoutput("~",NOCREATE);strcpy(token,yytext);return(PrevToken=TILDE); }
"->"		{ mywriteoutput("->",NOCREATE);strcpy(token,yytext);return(PrevToken=POINTERREF); }
"++"           { mywriteoutput("++",CREATE);strcpy(token,yytext);return(PrevToken=INCDEC); }
"--"           { mywriteoutput("--",CREATE);strcpy(token,yytext);return(PrevToken=INCDEC); }
"<<"           { mywriteoutput("<<",CREATE);strcpy(token,yytext);return(PrevToken=SHIFT); }
">>"           { mywriteoutput(">>",CREATE);strcpy(token,yytext);return(PrevToken=SHIFT); }
"<"            { mywriteoutput("<",CREATE);strcpy(token,yytext);return(PrevToken=COMPARE); }
">"            { mywriteoutput(">",CREATE);strcpy(token,yytext);return(PrevToken=COMPARE); }
"<="           { mywriteoutput("<=",CREATE);strcpy(token,yytext);return(PrevToken=COMPARE); }
">="           { mywriteoutput(">=",CREATE);strcpy(token,yytext);return(PrevToken=COMPARE); }
"=="           { mywriteoutput("==",NOCREATE);strcpy(token,yytext);return(PrevToken=EQUALEQUAL); }
"!="           { mywriteoutput("!=",NOCREATE);strcpy(token,yytext);return(PrevToken=NOTEQUAL); }
"&&"           { mywriteoutput("&&",NOCREATE);strcpy(token,yytext);return(PrevToken=ANDAND); }
"||"           { mywriteoutput("||",NOCREATE);strcpy(token,yytext);return(PrevToken=OROR); }
"="            { mywriteoutput("=",NOCREATE);strcpy(token,yytext);return(PrevToken=EQUAL); }
"+="           { mywriteoutput("+=",CREATE);strcpy(token,yytext);return(PrevToken=ASGNOP); }
"-="           { mywriteoutput("-=",CREATE);strcpy(token,yytext);return(PrevToken=ASGNOP); }
"*="           { mywriteoutput("*=",CREATE);strcpy(token,yytext);return(PrevToken=ASGNOP); }
"/="           { mywriteoutput("/=",CREATE);strcpy(token,yytext);return(PrevToken=ASGNOP); }
"%="           { mywriteoutput("%=",CREATE);strcpy(token,yytext);return(PrevToken=ASGNOP); }
">>="          { mywriteoutput(">>=",CREATE);strcpy(token,yytext);return(PrevToken=ASGNOP); }
"<<="          { mywriteoutput("<<=",CREATE);strcpy(token,yytext);return(PrevToken=ASGNOP); }
"&="           { mywriteoutput("&=",CREATE);strcpy(token,yytext);return(PrevToken=ASGNOP); }
"|="           { mywriteoutput("|=",CREATE);strcpy(token,yytext);return(PrevToken=ASGNOP); }
"^="           { mywriteoutput("^=",CREATE);strcpy(token,yytext);return(PrevToken=ASGNOP); }
","            { mywriteoutput(",",NOCREATE);strcpy(token,yytext);return(PrevToken=COMMA); }
"{"            { strcpy(token,yytext);return(PrevToken=L_BRACE); }
"}"            { strcpy(token,yytext);return(PrevToken=R_BRACE); }
";"            { mywriteoutput(";",NOCREATE);strcpy(token,yytext);return(PrevToken=SEMICOLON); }
","[ \t\n]*"{"	{ int i,count=0;
		
		  strcpy(token,yytext);
		  for (i=0;i<yyleng;i++)
			if (yytext[i]=='\n') count++;
		  CurrentInputLineNo += count;
		  mywriteoutput(",{ ",NOCREATE);return(PrevToken=COMMA_L_BRACE); 
		}
","[ \t\n]*"}"	{ int i,count=0;
		
		  for (i=0;i<yyleng;i++)
			if (yytext[i]=='\n') count++;
		  CurrentInputLineNo += count;
		  mywriteoutput(",} ",NOCREATE);return(PrevToken=COMMA_R_BRACE); }


"#"{WhiteSpace}("line")?{WhiteSpace}[0-9]+{WhiteSpace}\"{ASTRNG}\" { int i=0,j=0;
						  char temp[FILENAMELENGTH];

						  while ((yytext[i]<'0')||
						         (yytext[i]>'9')) 
							i++;
						  while ((yytext[i]>='0') &&
							 (yytext[i]<='9'))
							temp[j++]=yytext[i++];
						  temp[j]='\0';
						  CurrentInputLineNo = 
							atoi(temp);
						  while (yytext[i]!='\"')
							i++;
						  yytext[yyleng-1]='\0';
						  strcpy(CurrentFileName,
								yytext+i+1);
						}

"#"{WhiteSpace}("line")?{WhiteSpace}[0-9]+ { int i=0,j=0;
						  char temp[FILENAMELENGTH];

						  while ((yytext[i]<'0')||
						         (yytext[i]>'9')) 
							i++;
						  while ((yytext[i]>='0') &&
							 (yytext[i]<='9'))
							temp[j++]=yytext[i++];
						  temp[j]='\0';
						  CurrentInputLineNo = 
							atoi(temp);
						}

{ID}		           { int retvalue;
		
			      strcpy(token,yytext);
			      yylval=GetYSN();
			      yylval->string=MakeString(yytext);
			      if ((retvalue=SearchKey(yytext)) == -1)
				  return(PrevToken=TypeToToken(IDType(yytext)));
			      else { if (IsKey(yytext))
					writeoutput(yytext,NOFREE);
				     if (retvalue==0) return(IDENTIFIER);
				     if (MsgToStructFlag && (retvalue==MESSAGE))
					{ retvalue=STRUCT;
					  writeoutput("struct ",NOFREE);
					}
				     if (retvalue==MAIN)
					if (PrevToken!=CHARE)
						return(PrevToken=IDENTIFIER);
			 	     return(PrevToken=retvalue);
				   }
			    }

{ID}{WS}"::"{WS}{ID}	{ SYMTABPTR worksymtab;
			  char *modname,*name,*string;
			  int i;char ch;

			  strcpy(token,yytext);
			  yylval=GetYSN();
			  i=0;ch=yytext[0];
			  string=GetMem(strlen(yytext));
			  while ((ch!=' ')&&(ch!='\t')&&(ch!='\n')&&(ch!=':'))
				{ string[i]=ch;ch=yytext[++i]; }
			  string[i]='\0';
			  modname=yylval->modstring=MakeString(string);
			  while ((ch==' ')||(ch=='\t')||(ch=='\n')||(ch==':'))
				{ if (ch=='\n') CurrentInputLineNo++;
				  ch=yytext[++i]; 
				}
			  name=yylval->string=MakeString(yytext+i);

			  worksymtab=GlobalModuleSearch(name,modname);
			  if (worksymtab==NULL)
				{ error("Bad Module Reference",EXIT);
				  return(ID_DCOLON_ID);
				}
			  switch (TypeToToken(worksymtab->idtype))
			  { case IDENTIFIER : return(ID_DCOLON_ID);
			    case TYPE_IDENTIFIER : return(TYPE_IDENTIFIER);
			  }
			}

[0-9]+"."[0-9]*{EXP}?{FS}?   |
"."[0-9]+{EXP}?{FS}?                 |
[0-9]+{EXP}{FS}?                     |
[1-9][0-9]*{IS}?                     |
0[0-7]*{IS}?                         |
0[xX][0-9a-fA-F]+{IS}?          { mywriteoutput(yytext,CREATE);
				  strcpy(token,yytext);return(PrevToken=NUMBER);                
				}

\'([^'\\\n]|\\(['"?\\abfnrtv]|[0-7]{1,3}|[xX][0-9a-fA-F]{1,3}))+\'   {
				mywriteoutput(yytext,CREATE);
		                  strcpy(token,yytext);return(PrevToken=CHAR_CONST);              }

\"{ASTRNG}\"          		{ mywriteoutput(yytext,CREATE);
                                  strcpy(token,yytext);return(PrevToken=STRING);
                                }

[ \t\f]+      { writeoutput(yytext,NOFREE); }        /* Skip whitespace...*/
[\n]          { CurrentInputLineNo++; WriteReturn(); 
		/*printf("%d Line\n",yylineno);*/
		}

"#"{WhiteSpace}("pragma"){WhiteSpace}[^\n]*	{writeoutput(yytext,NOFREE);
				 	CurrentInputLineNo++;
					WriteReturn();}

"#"{WhiteSpace}("ident"){WhiteSpace}[^\n]*	{writeoutput(yytext,NOFREE);
				 	CurrentInputLineNo++;
					WriteReturn();}

.   		{ error("Lexical Oddity",EXIT); }

%%

yywrap(){ PrevToken=0; return(1); }

mywriteoutput(string,flag)
char *string;
int flag;
{ if (!flag) return;
  yylval=GetYSN();
  yylval->string=MakeString(string);
}

TypeToToken(type)
int type;
{ switch (type)
  { case TYPENAME :
    case MESSAGENAME : return(TYPE_IDENTIFIER);
    case MODULENAME :
    case CHARENAME  :
    case BOCNAME    :
    case ENTRYNAME  :
    default    : return(IDENTIFIER);
  }
}
