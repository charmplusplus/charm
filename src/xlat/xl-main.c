/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile$
 *	$Author$	$Locker$		$State$
 *	$Revision$	$Date$
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 ***************************************************************************
 * REVISION HISTORY:
 *
 * $Log$
 * Revision 2.3  1997-10-29 23:53:11  milind
 * Fixed CthInitialize bug on uth machines.
 *
 * Revision 2.2  1996/08/01 21:03:16  jyelon
 * Updated everything to bison and flex.
 *
 * Revision 2.1  1995/06/15 20:27:11  jyelon
 * got rid of myfree.
 *
 * Revision 2.0  1995/06/05  18:52:05  brunner
 * Reorganized file structure
 *
 * Revision 1.1  1994/11/03  17:41:41  brunner
 * Initial revision
 *
 ***************************************************************************/
static char ident[] = "@(#)$Header$";

#include <stdio.h>
#include <string.h>
#include "xl-sym.h"
#include "xl-lex.h"

extern BUFFERTYPE buffer;

#define EXIT 1
#define NAMELIMIT 100
char ModuleName[NAMELIMIT];

extern yyparse();
extern void ReadTokens();
extern void ReadKeys();

extern int CurrentInputLineNo;
extern char CurrentFileName[],*MakeString();
extern FILE *yyin,*outfile,*outh1,*outh2,*outh0;
extern int SavedLineNo;
extern char SavedFileName[];

extern void GenerateOuth();
extern void InitMapHead();

char outhfilename[FILENAMELENGTH];
int InPass1=0;

void CreateTempFile(void);
void CopyFile(void);
void SkipInterface(void);
void RealDummy();

void ParseCommandLine(argc,argv)
int argc;
char *argv[];
{ if (argc!=3) { fprintf(stderr,"usage: translate <InFile> <OutFile>. Stop\n"); 
	  	 exit(1); }
  strcpy(CurrentFileName,argv[1]);
  yyin = fopen(CurrentFileName,"r");
  outfile = fopen(argv[2],"w");
  strcpy(outhfilename,argv[2]);strcat(outhfilename,".0.h");
  outh0 = fopen(outhfilename,"w");
  strcpy(outhfilename,argv[2]);strcat(outhfilename,".1.h");
  outh1 = fopen(outhfilename,"w");
  strcpy(outhfilename,argv[2]);strcat(outhfilename,".2.h");
  outh2 = fopen(outhfilename,"w");
  if ((yyin==NULL)||(outfile==NULL)||(outh1==NULL)||(outh2==NULL)||(outh0==NULL))
	{ fprintf(stderr,"Cannot open file. Stop\n"); exit(1); }
  strcpy(outhfilename,argv[2]);
}

void InitOutputFile()
{ fprintf(outfile,"#include \"ckdefs.h\"\n"); 
  fprintf(outfile,"#include \"trans_externs.h\"\n"); 
  fprintf(outfile,"#include \"%s.0.h\"\n",outhfilename);
  fprintf(outfile,"#include \"%s.1.h\"\n",outhfilename);
  OUT0=GetOutStruct(outh0);
  OUT1=GetOutStruct(outh1);
  OUT=GetOutStruct(outfile);
  CurrentOut=NULL;
}

void ReInitializeParameters()
{ 
  RealDummy(SavedLineNo);
  CurrentInputLineNo=SavedLineNo-1;
  strcpy(CurrentFileName,SavedFileName);
}

main(argc,argv)
int argc;
char *argv[];
{ 
  ParseCommandLine(argc,argv);
  InitMapHead();
  ReadTokens();ReadKeys();
  InitSymTable(); 
  CreateTempFile();	/* temp file = outh2 */
  fclose(yyin);fclose(outh2);
  yyin=fopen(CurrentFileName,"r");		/* read temp file */
  if (yyin==NULL)  error("Can't open input file for first pass",EXIT);
  InitOutputFile();
  InPass1=0;
  buffer.count=0;
  yyparse();
  fclose(yyin);
  strcat(outhfilename,".2.h");
  yyin=fopen(outhfilename,"r");
  strcpy(outhfilename,argv[2]);
  if (yyin==NULL)
	  error("Can't open input for second pass",EXIT);
  ReInitializeParameters();
  yyparse();
  fclose(yyin);
  fprintf(outfile,"#include \"%s.2.h\"\n",outhfilename);
  fclose(outfile);
  strcat(outhfilename,".2.h");
  outh2=fopen(outhfilename,"w");
  if (outh2==NULL)
	error("Cannot Open File For Generating Translator Functions",EXIT);
  GenerateOuth();
  fclose(outh0);fclose(outh1);fclose(outh2);

  exit(0);
}

#define IsModule 1
#define IsInterface 2

void CreateTempFile(void)
{ char ch;
  int filechar;

  ch=filechar=getc(yyin);
  while (filechar!=EOF)
  { if ((ch!='i')&&(ch!='m'))
	ch=filechar=getc(yyin);
    else switch (NextToken(ch))
	 { case IsModule : CopyFile(); return;
	   case IsInterface : SkipInterface();
	   default	: ch=filechar=getc(yyin);
	}
  }
  error("Unexpected End of File",EXIT);
}

IsToken(token)
char *token;
{ int i,length;
  char ch;

  length=strlen(token);
  for (i=0;i<length;i++)
	{ ch=getc(yyin);
	  if (ch!= *token) return(0); else token++; 
	}
  ch=getc(yyin);
  if ((ch==' ')||(ch=='\t')||(ch=='\n')) return(1);
  return(0);	 
}

NextToken(ch)
char ch;
{ if (ch=='m')
	{ if (IsToken("odule")) return(IsModule); else return(0); }
  else 	{ if (IsToken("nterface")) return(IsInterface); else return(0); }
}

void CopyFile(void)
{ char ch;
  int index=0;
  int filechar;

  fprintf(outh2,"module ");
  ch=filechar=getc(yyin);
  while ((ch==' ')||(ch=='\t')||(ch=='\n'))
	{ putc(ch,outh2); ch=filechar=getc(yyin); }

  while (!((ch==' ')||(ch=='\t')||(ch=='\n')))
	{ putc(ch,outh2); ModuleName[index++]=ch; 
          if (index==NAMELIMIT) index--;
          if (filechar==EOF)
		{ ModuleName[index]='\0'; return; }
          ch=filechar=getc(yyin);
	}
  ModuleName[index]='\0';
  while (filechar!=EOF)
	{ putc(ch,outh2); ch=filechar=getc(yyin); }
}

void SkipInterface(void)
{ int count;
  char ch;
  int filechar;
  
  ch=filechar=getc(yyin);
  while ((filechar!=EOF)&&(ch!='{')) ch=filechar=getc(yyin);
  count=1;
  while ((count)&&(filechar!=EOF))
	{ ch=filechar=getc(yyin);
	  if (ch=='{') count++;
	  if (ch=='}') count--;
	}
  if (count) error("Unexpected End of File",EXIT);
}

#undef free

void RealDummy(a)
{}

