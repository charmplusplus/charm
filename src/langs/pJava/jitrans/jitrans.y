
%union {
	char *strval;
	int intval;
}

%token CLASS ENTRY LBRAC RBRAC LPAR RPAR SEMI GROUP
%token <strval>	IDENTIFIER

%type <strval> ClassName EntryName MessageName 

%%

File	:	ClassDecl
	|	File ClassDecl
	;
ClassDecl:	CLASS ClassName 
		{ CurrentClassName = (char *)calloc(strlen($2), sizeof(char));
		  Entries = (StringPtr *)calloc(lineno+1, sizeof(StringPtr));
		  Messages = (StringPtr *)calloc(lineno+1, sizeof(StringPtr));
		  strcpy(CurrentClassName, $2);	}
		LBRAC Entries RBRAC
		{ WriteClass(CurrentClassName,Entries, Messages,EntryCount); }
         | GROUP ClassName
		{ CurrentClassName = (char *)calloc(strlen($2), sizeof(char));
		  Entries = (StringPtr *)calloc(lineno+1, sizeof(StringPtr));
		  Messages = (StringPtr *)calloc(lineno+1, sizeof(StringPtr));
		  strcpy(CurrentClassName, $2);	}
		LBRAC Entries RBRAC
		{ WriteGroup(CurrentClassName,Entries, Messages,EntryCount); }
	;
ClassName:	IDENTIFIER
	;
Entries:	Entry
	|	Entries Entry
	;
Entry:	ENTRY EntryName LPAR MessageName RPAR SEMI
		{ Entries[EntryCount] = 
			(StringPtr)calloc(strlen($2)+1, sizeof(char));
		  Messages[EntryCount] = 
			(StringPtr)calloc(strlen($4)+1, sizeof(char));
		  strcpy(Entries[EntryCount], $2);
		  strcpy(Messages[EntryCount], $4);
		  EntryCount++;	}
	;
MessageName:	IDENTIFIER
	;
EntryName:	IDENTIFIER
	;

%%

#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include "lex.yy.c"

extern int lineno;
extern int yylex (void) ;

typedef char *StringPtr;

int yyerror(char *);
void UsageError();
void WriteClass(char *classname, StringPtr *entries, StringPtr *messages, 
		int num);
void WriteGroup(char *classname, StringPtr *entries, StringPtr *messages, 
		int num);
void WriteRegister(StringPtr *files, StringPtr *msgs, int nf, int nm);

char *CurrentClassName;
StringPtr *Entries;
StringPtr *Messages;
int EntryCount = 0;
FILE *yyout;

static void P(int indent, const char *format, ...) {
  va_list args;
  int i;
  for(i=0;i<indent;i++)
    fprintf(yyout, "  ");
  va_start(args, format);
  vfprintf(yyout, format, args);
}

int main(int argc, char *argv[])
{
  int mode = 0, filecount = 0, msgcount = 0, i;
  StringPtr *files, *msgs;
  char *outname="", *inname="";

  if (argc >= 2)
    if (strcmp(argv[1], "-register") != 0) {
	  files = (StringPtr *)calloc(argc-1, sizeof(StringPtr));
	  for (i=1; i<argc; i++) {
	    if (argv[i][0] == '-')
	      UsageError();
	    files[i-1] = (StringPtr)calloc(strlen(argv[i]), sizeof(char));
	    strcpy(files[i-1], argv[i]);
	    filecount++;
	  }
    } else {
	  mode = 1;
	  if ((argc<=3)||(strcmp(argv[2],"-classes")!=0)||(argv[3][0]=='-'))
	    UsageError();
	  files = (StringPtr *)calloc(argc-1, sizeof(StringPtr));
	  for (i=3; i<argc && (argv[i][0] != '-'); i++) {
	    files[i-3] = (StringPtr)calloc(strlen(argv[i]), sizeof(char));
	    strcpy(files[i-3], argv[i]);
	    filecount++;
	  }
	  if ((argc <= i+1) || (strcmp(argv[i], "-messages") != 0))
	    UsageError();
	  msgs = (StringPtr *)calloc(argc-1, sizeof(StringPtr));
	  for (i=i+1; i<argc; i++) {
	    if (argv[i][0] == '-')
	      UsageError();
	    msgs[i-filecount-4] = (StringPtr)calloc(strlen(argv[i]), sizeof(char));
	    strcpy(msgs[i-filecount-4], argv[i]);
	    msgcount++;
	  }
    }
  else
    UsageError();

  if (mode == 0)
    for (i=0; i<filecount; i++) {
	  inname = (char *)calloc(strlen(files[i])+4, sizeof(char));
	  sprintf(inname, "%s.ji", files[i]);
	  if ((yyin = fopen(inname, "r")) == NULL)
	    fprintf(stderr, "Input file %s.ji not found: skipping.\n", files[i]);
	  else { 
	    outname = (char *)calloc(strlen(files[i])+12, sizeof(char));
	    sprintf(outname, "Proxy_%s.java", files[i]);
	    yyout = fopen(outname, "w+");
	    yyparse();
	    fclose(yyin);
	    fclose(yyout);
	  }
    }
  else {
    yyout = fopen("RegisterAll.java", "w+");
    WriteRegister(files, msgs, filecount, msgcount);
    fclose(yyout);
  }
  exit(0);
}

void UsageError()
{
  fprintf(stderr,"\nUsage: jitrans [ClassName]+\n");
  fprintf(stderr,"Or:jitrans -register -classes [ClassName]+ -messages [MessageName]+\n\n");
  exit(1);
}

int yyerror(char *mesg)
{
	printf("Syntax error at line %d: %s\n", lineno, mesg);
	return 0;
}

StringPtr *MakeUnique(StringPtr *messages, int num) {
  StringPtr *umsgs = (StringPtr *) malloc(num*sizeof(StringPtr));
  int nunique = 0;
  int i,j,found;

  for(i=0;i<num;i++) umsgs[i] = 0;
  for(i=0;i<num;i++) {
    found = 0;
    for(j=0;j<nunique;j++) {
      if(strcmp(messages[i], umsgs[j])==0) {
        found = 1;
        break;
      }
    }
    if(!found) {
      umsgs[nunique++] = messages[i];
    }
  }
  return umsgs;
}

void WriteClass(char *classname, StringPtr *entries, StringPtr *messages, 
		int num)
{
  int i;
  StringPtr *umsgs = MakeUnique(messages,num);
  
  P(0,"import parallel.PRuntime;\n");
  P(0,"import parallel.RemoteObjectHandle;\n\n");
  P(0,"public class Proxy_%s {\n", classname);
  P(1,"public static int classID;\n\n");
  
  for (i=0; i<num; i++)
    P(1,"private static int %s_%s;\n", entries[i], messages[i]);
  P(0,"\n");
  for (i=0; i<num; i++)
    if(umsgs[i]!=0) P(1,"private static int %s_ID;\n", umsgs[i]);
  P(0,"\n");
  P(1,"public RemoteObjectHandle thishandle;\n\n");
    
  P(1,"public Proxy_%s(RemoteObjectHandle handle) {\n", classname);
  P(2,"thishandle = (RemoteObjectHandle) handle.clone();\n");
  P(1,"}\n\n");
 
  for (i=0; i<num; i++) {
    if(strcmp(entries[i],classname)==0) { /* Constructor */
      P(1,"public Proxy_%s(int pe, %s m) {\n", classname, messages[i]);
      P(2,"thishandle = PRuntime.CreateRemoteObject(pe, classID,\n");
      P(3,"%s_%s, m);\n", classname, messages[i]);
      P(1,"}\n\n");
    } else { /* Normal Entry */
      P(1,"public void %s(%s m) {\n", entries[i], messages[i]);
      P(2,"PRuntime.InvokeMethod(thishandle,%s_%s,m);\n",entries[i],
           messages[i]);
      P(1,"}\n\n");
    }
  }
  
  P(1,"static {\n");
  P(2,"classID = PRuntime.RegisterClass(\"%s\");\n", classname);
  for (i=0; i<num; i++)
    if(umsgs[i]!=0) P(2,"%s_ID = PRuntime.GetMessageID(\"%s\");\n",
                        umsgs[i],umsgs[i]);
  P(0,"\n");
  for(i=0; i<num; i++) {
    if(strcmp(entries[i],classname)==0) {
      P(2,"%s_%s = PRuntime.RegisterConstructor(classID, %s_ID);\n", 
	      entries[i], messages[i], messages[i]);
    } else {
      P(2,"%s_%s = PRuntime.RegisterEntry(\"%s\", classID, %s_ID);\n", 
	      entries[i], messages[i], entries[i], messages[i]);
    }
  }
  P(1,"}\n");
  P(0,"}\n");
}

void WriteGroup(char *classname, StringPtr *entries, StringPtr *messages, 
		int num)
{
  int i;
  StringPtr *umsgs = MakeUnique(messages, num);
  
  P(0,"import parallel.PRuntime;\n");
  P(0,"import parallel.RemoteObjectHandle;\n\n");
  P(0,"public class Proxy_%s {\n", classname);
  P(1,"public static int classID;\n\n");
  
  for (i=0; i<num; i++)
    P(1,"private static int %s_%s;\n", entries[i], messages[i]);
  P(0,"\n");
  for(i=0; i<num; i++)
    if(umsgs[i]!=0) P(1,"private static int %s_ID;\n\n", umsgs[i]);
  P(1,"public RemoteObjectHandle thishandle[];\n\n");
    
  P(1,"public Proxy_%s(RemoteObjectHandle handle[]) {\n",classname);
  P(2,"for(int i=0;i<handle.length;i++)\n");
  P(3,"thishandle[i] = (RemoteObjectHandle) handle[i].clone();\n");
  P(1,"}\n\n");

  for (i=0; i<num; i++) {
    if(strcmp(entries[i], classname)==0) { /*Constructor*/
      P(1,"public Proxy_%s(%s m) {\n", classname,messages[i]);
      P(2,"thishandle = new RemoteObjectHandle[PRuntime.NumPes()];\n");
      P(2,"for(int i=0;i<PRuntime.NumPes();i++)\n");
      P(3,"thishandle[i] = PRuntime.CreateRemoteObject(i, classID,\n");
      P(4,"%s_%s, m);\n", classname, messages[i]);
      P(1,"}\n\n");
    } else { /*Normal Entry*/
      P(1,"public void %s(%s m) {\n", entries[i], messages[i]);
      P(2,"for(int i=0;i<thishandle.length;i++)\n");
      P(3,"PRuntime.InvokeMethod(thishandle[i],%s_%s,m);\n",entries[i],
           messages[i]);
      P(1,"}\n\n");
      P(1,"public void %s(int pe, %s m) {\n", entries[i], messages[i]);
      P(2,"PRuntime.InvokeMethod(thishandle[pe],%s_%s,m);\n",entries[i],
           messages[i]);
      P(1,"}\n\n");
    }
  }
  
  P(1,"static {\n");
  P(2,"classID = PRuntime.RegisterClass(\"%s\");\n", classname);
  for (i=0; i<num; i++)
    if(umsgs[i]!=0) P(2,"%s_ID = PRuntime.GetMessageID(\"%s\");\n", 
                         umsgs[i], umsgs[i]);
  P(0,"\n");
  for(i=0; i<num; i++) {
    if(strcmp(entries[i],classname)==0) {
      P(2,"%s_%s = PRuntime.RegisterConstructor(classID, %s_ID);\n", 
	      entries[i], messages[i], messages[i]);
    } else {
      P(2,"%s_%s = PRuntime.RegisterEntry(\"%s\", classID, %s_ID);\n", 
	      entries[i], messages[i], entries[i], messages[i]);
    }
  }
  P(1,"}\n");
  P(0,"}\n");
}

void WriteRegister(StringPtr *files, StringPtr *msgs, int nf, int nm)
{
  int i;
  
  P(0, "import parallel.PRuntime;\n");
  P(0, "import java.lang.Class;\n\n");
  
  P(0, "public class RegisterAll {\n\n");
  P(1, "static void registerAll() {\n");
  for (i=0; i<nm; i++)
    P(2,"PRuntime.RegisterMessage(\"%s\");\n", msgs[i]);
  P(2, "try {\n");
  for (i=0; i<nf; i++)
    P(3,"Class.forName(\"Proxy_%s\");\n", files[i]);
  P(2, "} catch (ClassNotFoundException e) {\n");
  P(3, "PRuntime.out.println(\"Cannot find Class!!\");\n");
  P(3, "PRuntime.exit(1);\n");
  P(2, "}\n");
  P(1,"}\n");
  P(0,"}\n");
}

