%{
#include <string.h>
int currentline=1;

#ifdef yywrap
#undef yywrap
#endif
%}

WS [ \t\n]*
WSN [ \t]*

%%
CreateChare{WS}"("	{ CheckReturns(yytext);fprintf(yyout,"CreateChare("); }
CreateBoc{WS}"("	{ CheckReturns(yytext);fprintf(yyout,"CreateBoc("); }
MyBocNum{WS}"("		{ CheckReturns(yytext);fprintf(yyout,"MyBocNum("); }
MyBranchID{WS}"("	{ CheckReturns(yytext);fprintf(yyout,"MyBranchID("); }
SendMsgBranch{WS}"("	{ CheckReturns(yytext);fprintf(yyout,"SendMsgBranch("); }
ImmSendMsgBranch{WS}"("	{ CheckReturns(yytext);fprintf(yyout,"ImmSendMsgBranch("); }
BroadcastMsgBranch{WS}"("  { CheckReturns(yytext);fprintf(yyout,"BroadcastMsgBranch("); }
"#"{WSN}[line]?{WSN}[0-9]+ { currentline=GetLine(yytext);fprintf(yyout,"%s",yytext); }
"#"{WSN}[line]?{WSN}[0-9]+{WSN}\"[^\n]*\"{WSN}[0-9]+ { currentline = GetLine(yytext);
			   output_proper_line(yytext);}
.			{ CountReturns(yytext); fprintf(yyout,"%s",yytext); }
%%

yywrap(){ return(1); }

main() { writem4(); writeundef(); yylex(); }

GetLine(string)
char string[];
{ int i=0,j;
  char dummy[10];

  while ((string[i]<'0')||(string[i]>'9')) i++;
  j=0;
  while ((string[i]>='0')&&(string[i]<='9')) dummy[j++] = string[i++];
  dummy[j]='\0';
  return(atoi(dummy));
}

output_proper_line(string)
char string[];
{
   int length;

   length=strlen(string)-1;
   while (string[length-1]!='"') length--;
   string[length]='\0';
   fprintf(yyout,"%s",string);
}

CountReturns(string)
char *string;
{
  while (*string) {
    if (*string=='\n') currentline++;
    string++;
  }
}

CheckReturns(string)
char *string;
{
  int anyret=0;
  while (*string) {
    if (*string=='\n') { currentline++; anyret=1; }
    string++;
  }
  if (anyret)
    fprintf(yyout,"# line %d\n",currentline);
}


char *createchare="define(CreateChare,`_CK_CreateChare($1,$2,$3,ifelse($4,,NULL_VID,$4),ifelse($5,,NULL_PE,$5))')";

char *createboc="define(CreateBoc,`_CK_CreateBoc($1,$2,$3,ifelse($4,,-1`,'NULL,$4`,'$5))')";

char *mybocnum="define(MyBocNum,`_CK_MyBocNum(_CK_4mydata)')";
char *mybranchid="define(MyBranchID,`_CK_MyBranchID($1,_CK_4mydata)')";
char *sendmsgbranch="define(SendMsgBranch,`_CK_SendMsgBranch($1,$2,ifelse($4,,_CK_MyBocNum(_CK_4mydata),$4),$3)')";
char *immsendmsgbranch="define(ImmSendMsgBranch,`_CK_ImmSendMsgBranch($1,$2,ifelse($4,,_CK_MyBocNum(_CK_4mydata),$4),$3)')";
char *broadcastmsgbranch="define(BroadcastMsgBranch,`_CK_BroadcastMsgBranch($1,$2,ifelse($3,,_CK_MyBocNum(_CK_4mydata),$3))')";
char *createacc="define(CreateAcc,`_CK_CreateAcc($1,$2,ifelse($3,,-1`,'NULL,$3`,'$4))')";
char *createmono="define(CreateMono,`_CK_CreateMono($1,$2,ifelse($3,,-1`,'NULL,$3`,'$4))')";

writem4()
{ printf("%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n",createchare,createboc,mybocnum,mybranchid,
		sendmsgbranch,immsendmsgbranch,broadcastmsgbranch,createacc,createmono);
}

writeundef()
{ printf("undefine(`changequote')\n");
  printf("undefine(`divert')\n");
  printf("undefine(`divnum')\n");
  printf("undefine(`dnl')\n");
  printf("undefine(`dumpdef')\n");
  printf("undefine(`errprint')\n");
  printf("undefine(`eval')\n");
  printf("undefine(`ifdef')\n");
  printf("undefine(`include')\n");
  printf("undefine(`incr')\n");
  printf("undefine(`index')\n");
  printf("undefine(`len')\n");
  printf("undefine(`maketemp')\n");
  printf("undefine(`sinclude')\n");
  printf("undefine(`substr')\n");
  printf("undefine(`syscmd')\n");
  printf("undefine(`translit')\n");
  printf("undefine(`undivert')\n");
  printf("undefine(`define')\n");
  printf("undefine(`undefine')\n");
}
