%{
#include <string.h>

char CoreName[256] ;
int currentline=1;
#ifdef yywrap
#undef yywrap
#endif
%}

WS [ \t\n]*
WSN [ \t]*

MODULE module
STR [ \t\n]*
NAME [a-z0-9A-Z_]+
NA [^a-zA-Z0-9_]

%%

FunctionNameToRef{WS}"("  { CheckReturns(yytext); fprintf(yyout,"FunctionNameToRef("); }
FunctionRefToName{WS}"("  { CheckReturns(yytext); fprintf(yyout,"FunctionRefToName("); }
new_message{WS}"("	  { CheckReturns(yytext); fprintf(yyout,"new_message("); }
new_packbuffer{WS}"("	  { CheckReturns(yytext); fprintf(yyout,"new_packbuffer("); }
delete_message{WS}"("	  { CheckReturns(yytext); fprintf(yyout,"delete_message("); }
new_prio_message{WS}"("	  { CheckReturns(yytext); fprintf(yyout,"new_prio_message("); }
SendMsg{WS}"("		  { CheckReturns(yytext); fprintf(yyout,"SendMsg("); }
Insert{WS}"("		  { CheckReturns(yytext); fprintf(yyout,"Insert("); }
Find{WS}"("		  { CheckReturns(yytext); fprintf(yyout,"Find("); }
Delete{WS}"("		  { CheckReturns(yytext); fprintf(yyout,"Delete("); }
  
new_chare{WS}"("	  { CheckReturns(yytext); fprintf(yyout,"new_chare("); }
new_branched_chare{WS}"(" { CheckReturns(yytext); fprintf(yyout,"new_branched_chare("); }
new_accumulator{WS}"("	  { CheckReturns(yytext); fprintf(yyout,"new_accumulator("); }
CollectValue{WS}"("	  { CheckReturns(yytext); fprintf(yyout,"CollectValue(");}
new_monotonic{WS}"("	  { CheckReturns(yytext); fprintf(yyout,"new_monotonic("); }
CharmExit{WS}"("  	  { CheckReturns(yytext); fprintf(yyout,"CharmExit("); }
CPriorityPtr{WS}"("  	  { CheckReturns(yytext); fprintf(yyout,"CPriorityPtr("); }
CTimer{WS}"("  	          { CheckReturns(yytext); fprintf(yyout,"CTimer("); }
CUTimer{WS}"("  	  { CheckReturns(yytext); fprintf(yyout,"CUTimer("); }
CHTimer{WS}"("  	  { CheckReturns(yytext); fprintf(yyout,"CHTimer("); }

"#"{WSN}[line]{WSN}[0-9]+                               { currentline=GetLine(yytext); fprintf(yyout,"%s",yytext); }
"#"{WSN}[line]?{WSN}[0-9]+{WSN}\"[^\n]*\"{WSN}[0-9]+    { currentline=GetLine(yytext); fprintf(yyout,"%s",yytext); }
"#"{WSN}[line]?{WSN}[0-9]+{WSN}\"[^\n]*\"{WSN}[0-9]+{WSN}[0-9]+ { currentline = GetLine(yytext); fprintf(yyout,"%s",yytext);}
"#"{WSN}[line]?{WSN}[0-9]+{WSN}\"[^\n]*\"{WSN}[0-9]+{WSN}[0-9]+{WSN}[0-9]+ { currentline = GetLine(yytext); fprintf(yyout,"%s",yytext);}

"#"{WSN}[0-9]+{WSN}\n     { CountReturns(yytext); fprintf(yyout,"\n") ; }
.			  { CountReturns(yytext); fprintf(yyout,"%s",yytext); }

%%

yywrap(){ return(1); }


main(argc,argv) 
int argc ;
char *argv[] ;
{ 
	char pgm[256] ;
        int len; char *bgn, *end;

	if ( argc != 2 ) {
		printf("Spacefilter invoked improperly ! Aborting.\n") ;
		exit(1) ;	
	}
/* Find the file name */
	strcpy(pgm,argv[1]);

        bgn = strrchr(argv[1], '/');
        if (bgn==0) bgn=argv[1];
        end = bgn;
        while (1)
            {
            char c = *end;
            if (!(((c>='a')&&(c<='z'))||
                  ((c>='A')&&(c<='Z'))||
                  ((c>='0')&&(c<='9'))||
                  (c=='_'))) break;
            end++;
            }
	strncpy(CoreName, bgn, end-bgn);
	CoreName[(end-bgn)] = '\0' ;

	yyin = fopen(argv[1],"r") ;

/* Do the rest of the stuff.. */
	writem4(); 
	writeundef(); 
	yylex(); 
}

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




/* SANJEEV */
char *ckfreemsg="define(delete_message,`CkFreeMsg((void *)$1)')";

char *charmexit="define(CharmExit,`CkExit()')";
char *cpriorityptr="define(CPriorityPtr,`CkPriorityPtr($1)')";
char *ctimer="define(CTimer,`CkTimer()')";
char *cutimer="define(CUTimer,`CkUTimer()')";
char *chtimer="define(CHTimer,`CkHTimer()')";


char * createchare="define(new_chare,`_CK_CreateChare(_CK_chare_$1,$2,$3,ifelse($4,,NULL_VID,$4),ifelse($5,,CK_PE_ANY,$5))')" ;

char * createboc="define(new_branched_chare,`_CK_CreateBoc(_CK_chare_$1,$2,$3,ifelse($4,,-1`,'NULL,$4`,'$5))')" ;

char * createacc="define(new_accumulator,`_CK_CreateAcc(_CK_acc_$1,$2,ifelse($3,,-1`,'NULL,$3`,'$4))')" ;
char * createmono="define(new_monotonic,`_CK_CreateMono(_CK_mono_$1,$2,ifelse($3,,-1`,'NULL,$3`,'$4))')" ;

char * functionreftoname="define(FunctionRefToName,`CsvAccess(_CK_9_GlobalFunctionTable)[$1]')" ;
char * functionnametoref="define(FunctionNameToRef,`_CK_func_$1')" ;

char * ckallocpackbuffer="define(new_packbuffer,`CkAllocPackBuffer($1,$2)')" ;



writem4()
{ 
  char ckallocmsg[256] ;
  char ckallocpriomsg[256] ;

/* SANJEEV */
  printf("%s\n",ckfreemsg) ;
  printf("%s\n%s\n%s\n%s\n%s\n",
	 charmexit,cpriorityptr,ctimer,cutimer,chtimer) ;
  printf("%s\n%s\n%s\n%s\n",createacc,createmono,createchare, createboc);

sprintf(ckallocmsg,"define(new_message,`ifelse($2,,GenericCkAlloc(_CK_%s._CK_msg_$1`,'sizeof($1)`,'0),((ALLOCFNPTR)(CsvAccess(MsgToStructTable)[_CK_%s._CK_msg_$1].alloc))(_CK_%s._CK_msg_$1`,'sizeof($1)`,'$2`,'0))')", CoreName, CoreName, CoreName );

sprintf(ckallocpriomsg,"define(new_prio_message,`ifelse($3,,GenericCkAlloc(_CK_%s._CK_msg_$1`,'sizeof($1)`,'$2),((ALLOCFNPTR)(CsvAccess(MsgToStructTable)[_CK_%s._CK_msg_$1].alloc))(_CK_%s._CK_msg_$1`,'sizeof($1)`,'$3`,'$2))')", CoreName, CoreName, CoreName ) ;

  printf("%s\n%s\n%s\n%s\n%s\n\n",ckallocmsg, ckallocpriomsg, 
		functionreftoname, functionnametoref, ckallocpackbuffer );
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

