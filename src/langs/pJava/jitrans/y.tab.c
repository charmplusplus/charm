/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/


# line 2 "jitrans.y"
typedef union
#ifdef __cplusplus
	YYSTYPE
#endif
 {
	char *strval;
	int intval;
} YYSTYPE;
# define CLASS 257
# define ENTRY 258
# define LBRAC 259
# define RBRAC 260
# define LPAR 261
# define RPAR 262
# define SEMI 263
# define GROUP 264
# define IDENTIFIER 265

#ifdef __STDC__
#include <stdlib.h>
#include <string.h>
#else
#include <malloc.h>
#include <memory.h>
#endif

#include <values.h>

#ifdef __cplusplus

#ifndef yyerror
	void yyerror(const char *);
#endif

#ifndef yylex
#ifdef __EXTERN_C__
	extern "C" { int yylex(void); }
#else
	int yylex(void);
#endif
#endif
	int yyparse(void);

#endif
#define yyclearin yychar = -1
#define yyerrok yyerrflag = 0
extern int yychar;
extern int yyerrflag;
YYSTYPE yylval;
YYSTYPE yyval;
typedef int yytabelem;
#ifndef YYMAXDEPTH
#define YYMAXDEPTH 150
#endif
#if YYMAXDEPTH > 0
int yy_yys[YYMAXDEPTH], *yys = yy_yys;
YYSTYPE yy_yyv[YYMAXDEPTH], *yyv = yy_yyv;
#else	/* user does initial allocation */
int *yys;
YYSTYPE *yyv;
#endif
static int yymaxdepth = YYMAXDEPTH;
# define YYERRCODE 256

# line 51 "jitrans.y"


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

yytabelem yyexca[] ={
-1, 1,
	0, -1,
	-2, 0,
	};
# define YYNPROD 13
# define YYLAST 31
yytabelem yyact[]={

    24,    20,     7,     3,    26,    25,    22,    15,    15,    21,
     4,    15,    14,    17,    12,    11,     6,    13,     2,    10,
     5,     8,     9,     1,    23,    19,    18,     0,     0,    18,
    16 };
yytabelem yypact[]={

  -254,  -254,-10000000,  -263,  -263,-10000000,-10000000,-10000000,-10000000,  -244,
  -245,  -250,  -250,  -247,-10000000,  -264,  -251,-10000000,-10000000,  -255,
-10000000,-10000000,  -265,  -257,-10000000,  -259,-10000000 };
yytabelem yypgo[]={

     0,    16,    25,    24,    23,    18,    22,    17,    19,    12 };
yytabelem yyr1[]={

     0,     4,     4,     6,     5,     8,     5,     1,     7,     7,
     9,     3,     2 };
yytabelem yyr2[]={

     0,     2,     4,     1,    13,     1,    13,     2,     2,     4,
    13,     2,     2 };
yytabelem yychk[]={

-10000000,    -4,    -5,   257,   264,    -5,    -1,   265,    -1,    -6,
    -8,   259,   259,    -7,    -9,   258,    -7,   260,    -9,    -2,
   265,   260,   261,    -3,   265,   262,   263 };
yytabelem yydef[]={

     0,    -2,     1,     0,     0,     2,     3,     7,     5,     0,
     0,     0,     0,     0,     8,     0,     0,     4,     9,     0,
    12,     6,     0,     0,    11,     0,    10 };
typedef struct
#ifdef __cplusplus
	yytoktype
#endif
{ char *t_name; int t_val; } yytoktype;
#ifndef YYDEBUG
#	define YYDEBUG	0	/* don't allow debugging */
#endif

#if YYDEBUG

yytoktype yytoks[] =
{
	"CLASS",	257,
	"ENTRY",	258,
	"LBRAC",	259,
	"RBRAC",	260,
	"LPAR",	261,
	"RPAR",	262,
	"SEMI",	263,
	"GROUP",	264,
	"IDENTIFIER",	265,
	"-unknown-",	-1	/* ends search */
};

char * yyreds[] =
{
	"-no such reduction-",
	"File : ClassDecl",
	"File : File ClassDecl",
	"ClassDecl : CLASS ClassName",
	"ClassDecl : CLASS ClassName LBRAC Entries RBRAC",
	"ClassDecl : GROUP ClassName",
	"ClassDecl : GROUP ClassName LBRAC Entries RBRAC",
	"ClassName : IDENTIFIER",
	"Entries : Entry",
	"Entries : Entries Entry",
	"Entry : ENTRY EntryName LPAR MessageName RPAR SEMI",
	"MessageName : IDENTIFIER",
	"EntryName : IDENTIFIER",
};
#endif /* YYDEBUG */
/*
 * Copyright (c) 1993 by Sun Microsystems, Inc.
 */

#pragma ident	"@(#)yaccpar	6.12	93/06/07 SMI"

/*
** Skeleton parser driver for yacc output
*/

/*
** yacc user known macros and defines
*/
#define YYERROR		goto yyerrlab
#define YYACCEPT	return(0)
#define YYABORT		return(1)
#define YYBACKUP( newtoken, newvalue )\
{\
	if ( yychar >= 0 || ( yyr2[ yytmp ] >> 1 ) != 1 )\
	{\
		yyerror( "syntax error - cannot backup" );\
		goto yyerrlab;\
	}\
	yychar = newtoken;\
	yystate = *yyps;\
	yylval = newvalue;\
	goto yynewstate;\
}
#define YYRECOVERING()	(!!yyerrflag)
#define YYNEW(type)	malloc(sizeof(type) * yynewmax)
#define YYCOPY(to, from, type) \
	(type *) memcpy(to, (char *) from, yynewmax * sizeof(type))
#define YYENLARGE( from, type) \
	(type *) realloc((char *) from, yynewmax * sizeof(type))
#ifndef YYDEBUG
#	define YYDEBUG	1	/* make debugging available */
#endif

/*
** user known globals
*/
int yydebug;			/* set to 1 to get debugging */

/*
** driver internal defines
*/
#define YYFLAG		(-10000000)

/*
** global variables used by the parser
*/
YYSTYPE *yypv;			/* top of value stack */
int *yyps;			/* top of state stack */

int yystate;			/* current state */
int yytmp;			/* extra var (lasts between blocks) */

int yynerrs;			/* number of errors */
int yyerrflag;			/* error recovery flag */
int yychar;			/* current input token number */



#ifdef YYNMBCHARS
#define YYLEX()		yycvtok(yylex())
/*
** yycvtok - return a token if i is a wchar_t value that exceeds 255.
**	If i<255, i itself is the token.  If i>255 but the neither 
**	of the 30th or 31st bit is on, i is already a token.
*/
#if defined(__STDC__) || defined(__cplusplus)
int yycvtok(int i)
#else
int yycvtok(i) int i;
#endif
{
	int first = 0;
	int last = YYNMBCHARS - 1;
	int mid;
	wchar_t j;

	if(i&0x60000000){/*Must convert to a token. */
		if( yymbchars[last].character < i ){
			return i;/*Giving up*/
		}
		while ((last>=first)&&(first>=0)) {/*Binary search loop*/
			mid = (first+last)/2;
			j = yymbchars[mid].character;
			if( j==i ){/*Found*/ 
				return yymbchars[mid].tvalue;
			}else if( j<i ){
				first = mid + 1;
			}else{
				last = mid -1;
			}
		}
		/*No entry in the table.*/
		return i;/* Giving up.*/
	}else{/* i is already a token. */
		return i;
	}
}
#else/*!YYNMBCHARS*/
#define YYLEX()		yylex()
#endif/*!YYNMBCHARS*/

/*
** yyparse - return 0 if worked, 1 if syntax error not recovered from
*/
#if defined(__STDC__) || defined(__cplusplus)
int yyparse(void)
#else
int yyparse()
#endif
{
	register YYSTYPE *yypvt;	/* top of value stack for $vars */

#if defined(__cplusplus) || defined(lint)
/*
	hacks to please C++ and lint - goto's inside switch should never be
	executed; yypvt is set to 0 to avoid "used before set" warning.
*/
	static int __yaccpar_lint_hack__ = 0;
	switch (__yaccpar_lint_hack__)
	{
		case 1: goto yyerrlab;
		case 2: goto yynewstate;
	}
	yypvt = 0;
#endif

	/*
	** Initialize externals - yyparse may be called more than once
	*/
	yypv = &yyv[-1];
	yyps = &yys[-1];
	yystate = 0;
	yytmp = 0;
	yynerrs = 0;
	yyerrflag = 0;
	yychar = -1;

#if YYMAXDEPTH <= 0
	if (yymaxdepth <= 0)
	{
		if ((yymaxdepth = YYEXPAND(0)) <= 0)
		{
			yyerror("yacc initialization error");
			YYABORT;
		}
	}
#endif

	{
		register YYSTYPE *yy_pv;	/* top of value stack */
		register int *yy_ps;		/* top of state stack */
		register int yy_state;		/* current state */
		register int  yy_n;		/* internal state number info */
	goto yystack;	/* moved from 6 lines above to here to please C++ */

		/*
		** get globals into registers.
		** branch to here only if YYBACKUP was called.
		*/
	yynewstate:
		yy_pv = yypv;
		yy_ps = yyps;
		yy_state = yystate;
		goto yy_newstate;

		/*
		** get globals into registers.
		** either we just started, or we just finished a reduction
		*/
	yystack:
		yy_pv = yypv;
		yy_ps = yyps;
		yy_state = yystate;

		/*
		** top of for (;;) loop while no reductions done
		*/
	yy_stack:
		/*
		** put a state and value onto the stacks
		*/
#if YYDEBUG
		/*
		** if debugging, look up token value in list of value vs.
		** name pairs.  0 and negative (-1) are special values.
		** Note: linear search is used since time is not a real
		** consideration while debugging.
		*/
		if ( yydebug )
		{
			register int yy_i;

			printf( "State %d, token ", yy_state );
			if ( yychar == 0 )
				printf( "end-of-file\n" );
			else if ( yychar < 0 )
				printf( "-none-\n" );
			else
			{
				for ( yy_i = 0; yytoks[yy_i].t_val >= 0;
					yy_i++ )
				{
					if ( yytoks[yy_i].t_val == yychar )
						break;
				}
				printf( "%s\n", yytoks[yy_i].t_name );
			}
		}
#endif /* YYDEBUG */
		if ( ++yy_ps >= &yys[ yymaxdepth ] )	/* room on stack? */
		{
			/*
			** reallocate and recover.  Note that pointers
			** have to be reset, or bad things will happen
			*/
			int yyps_index = (yy_ps - yys);
			int yypv_index = (yy_pv - yyv);
			int yypvt_index = (yypvt - yyv);
			int yynewmax;
#ifdef YYEXPAND
			yynewmax = YYEXPAND(yymaxdepth);
#else
			yynewmax = 2 * yymaxdepth;	/* double table size */
			if (yymaxdepth == YYMAXDEPTH)	/* first time growth */
			{
				char *newyys = (char *)YYNEW(int);
				char *newyyv = (char *)YYNEW(YYSTYPE);
				if (newyys != 0 && newyyv != 0)
				{
					yys = YYCOPY(newyys, yys, int);
					yyv = YYCOPY(newyyv, yyv, YYSTYPE);
				}
				else
					yynewmax = 0;	/* failed */
			}
			else				/* not first time */
			{
				yys = YYENLARGE(yys, int);
				yyv = YYENLARGE(yyv, YYSTYPE);
				if (yys == 0 || yyv == 0)
					yynewmax = 0;	/* failed */
			}
#endif
			if (yynewmax <= yymaxdepth)	/* tables not expanded */
			{
				yyerror( "yacc stack overflow" );
				YYABORT;
			}
			yymaxdepth = yynewmax;

			yy_ps = yys + yyps_index;
			yy_pv = yyv + yypv_index;
			yypvt = yyv + yypvt_index;
		}
		*yy_ps = yy_state;
		*++yy_pv = yyval;

		/*
		** we have a new state - find out what to do
		*/
	yy_newstate:
		if ( ( yy_n = yypact[ yy_state ] ) <= YYFLAG )
			goto yydefault;		/* simple state */
#if YYDEBUG
		/*
		** if debugging, need to mark whether new token grabbed
		*/
		yytmp = yychar < 0;
#endif
		if ( ( yychar < 0 ) && ( ( yychar = YYLEX() ) < 0 ) )
			yychar = 0;		/* reached EOF */
#if YYDEBUG
		if ( yydebug && yytmp )
		{
			register int yy_i;

			printf( "Received token " );
			if ( yychar == 0 )
				printf( "end-of-file\n" );
			else if ( yychar < 0 )
				printf( "-none-\n" );
			else
			{
				for ( yy_i = 0; yytoks[yy_i].t_val >= 0;
					yy_i++ )
				{
					if ( yytoks[yy_i].t_val == yychar )
						break;
				}
				printf( "%s\n", yytoks[yy_i].t_name );
			}
		}
#endif /* YYDEBUG */
		if ( ( ( yy_n += yychar ) < 0 ) || ( yy_n >= YYLAST ) )
			goto yydefault;
		if ( yychk[ yy_n = yyact[ yy_n ] ] == yychar )	/*valid shift*/
		{
			yychar = -1;
			yyval = yylval;
			yy_state = yy_n;
			if ( yyerrflag > 0 )
				yyerrflag--;
			goto yy_stack;
		}

	yydefault:
		if ( ( yy_n = yydef[ yy_state ] ) == -2 )
		{
#if YYDEBUG
			yytmp = yychar < 0;
#endif
			if ( ( yychar < 0 ) && ( ( yychar = YYLEX() ) < 0 ) )
				yychar = 0;		/* reached EOF */
#if YYDEBUG
			if ( yydebug && yytmp )
			{
				register int yy_i;

				printf( "Received token " );
				if ( yychar == 0 )
					printf( "end-of-file\n" );
				else if ( yychar < 0 )
					printf( "-none-\n" );
				else
				{
					for ( yy_i = 0;
						yytoks[yy_i].t_val >= 0;
						yy_i++ )
					{
						if ( yytoks[yy_i].t_val
							== yychar )
						{
							break;
						}
					}
					printf( "%s\n", yytoks[yy_i].t_name );
				}
			}
#endif /* YYDEBUG */
			/*
			** look through exception table
			*/
			{
				register int *yyxi = yyexca;

				while ( ( *yyxi != -1 ) ||
					( yyxi[1] != yy_state ) )
				{
					yyxi += 2;
				}
				while ( ( *(yyxi += 2) >= 0 ) &&
					( *yyxi != yychar ) )
					;
				if ( ( yy_n = yyxi[1] ) < 0 )
					YYACCEPT;
			}
		}

		/*
		** check for syntax error
		*/
		if ( yy_n == 0 )	/* have an error */
		{
			/* no worry about speed here! */
			switch ( yyerrflag )
			{
			case 0:		/* new error */
				yyerror( "syntax error" );
				goto skip_init;
			yyerrlab:
				/*
				** get globals into registers.
				** we have a user generated syntax type error
				*/
				yy_pv = yypv;
				yy_ps = yyps;
				yy_state = yystate;
			skip_init:
				yynerrs++;
				/* FALLTHRU */
			case 1:
			case 2:		/* incompletely recovered error */
					/* try again... */
				yyerrflag = 3;
				/*
				** find state where "error" is a legal
				** shift action
				*/
				while ( yy_ps >= yys )
				{
					yy_n = yypact[ *yy_ps ] + YYERRCODE;
					if ( yy_n >= 0 && yy_n < YYLAST &&
						yychk[yyact[yy_n]] == YYERRCODE)					{
						/*
						** simulate shift of "error"
						*/
						yy_state = yyact[ yy_n ];
						goto yy_stack;
					}
					/*
					** current state has no shift on
					** "error", pop stack
					*/
#if YYDEBUG
#	define _POP_ "Error recovery pops state %d, uncovers state %d\n"
					if ( yydebug )
						printf( _POP_, *yy_ps,
							yy_ps[-1] );
#	undef _POP_
#endif
					yy_ps--;
					yy_pv--;
				}
				/*
				** there is no state on stack with "error" as
				** a valid shift.  give up.
				*/
				YYABORT;
			case 3:		/* no shift yet; eat a token */
#if YYDEBUG
				/*
				** if debugging, look up token in list of
				** pairs.  0 and negative shouldn't occur,
				** but since timing doesn't matter when
				** debugging, it doesn't hurt to leave the
				** tests here.
				*/
				if ( yydebug )
				{
					register int yy_i;

					printf( "Error recovery discards " );
					if ( yychar == 0 )
						printf( "token end-of-file\n" );
					else if ( yychar < 0 )
						printf( "token -none-\n" );
					else
					{
						for ( yy_i = 0;
							yytoks[yy_i].t_val >= 0;
							yy_i++ )
						{
							if ( yytoks[yy_i].t_val
								== yychar )
							{
								break;
							}
						}
						printf( "token %s\n",
							yytoks[yy_i].t_name );
					}
				}
#endif /* YYDEBUG */
				if ( yychar == 0 )	/* reached EOF. quit */
					YYABORT;
				yychar = -1;
				goto yy_newstate;
			}
		}/* end if ( yy_n == 0 ) */
		/*
		** reduction by production yy_n
		** put stack tops, etc. so things right after switch
		*/
#if YYDEBUG
		/*
		** if debugging, print the string that is the user's
		** specification of the reduction which is just about
		** to be done.
		*/
		if ( yydebug )
			printf( "Reduce by (%d) \"%s\"\n",
				yy_n, yyreds[ yy_n ] );
#endif
		yytmp = yy_n;			/* value to switch over */
		yypvt = yy_pv;			/* $vars top of value stack */
		/*
		** Look in goto table for next state
		** Sorry about using yy_state here as temporary
		** register variable, but why not, if it works...
		** If yyr2[ yy_n ] doesn't have the low order bit
		** set, then there is no action to be done for
		** this reduction.  So, no saving & unsaving of
		** registers done.  The only difference between the
		** code just after the if and the body of the if is
		** the goto yy_stack in the body.  This way the test
		** can be made before the choice of what to do is needed.
		*/
		{
			/* length of production doubled with extra bit */
			register int yy_len = yyr2[ yy_n ];

			if ( !( yy_len & 01 ) )
			{
				yy_len >>= 1;
				yyval = ( yy_pv -= yy_len )[1];	/* $$ = $1 */
				yy_state = yypgo[ yy_n = yyr1[ yy_n ] ] +
					*( yy_ps -= yy_len ) + 1;
				if ( yy_state >= YYLAST ||
					yychk[ yy_state =
					yyact[ yy_state ] ] != -yy_n )
				{
					yy_state = yyact[ yypgo[ yy_n ] ];
				}
				goto yy_stack;
			}
			yy_len >>= 1;
			yyval = ( yy_pv -= yy_len )[1];	/* $$ = $1 */
			yy_state = yypgo[ yy_n = yyr1[ yy_n ] ] +
				*( yy_ps -= yy_len ) + 1;
			if ( yy_state >= YYLAST ||
				yychk[ yy_state = yyact[ yy_state ] ] != -yy_n )
			{
				yy_state = yyact[ yypgo[ yy_n ] ];
			}
		}
					/* save until reenter driver code */
		yystate = yy_state;
		yyps = yy_ps;
		yypv = yy_pv;
	}
	/*
	** code supplied by user is placed in this switch
	*/
	switch( yytmp )
	{
		
case 3:
# line 18 "jitrans.y"
{ CurrentClassName = (char *)calloc(strlen(yypvt[-0].strval), sizeof(char));
		  Entries = (StringPtr *)calloc(lineno+1, sizeof(StringPtr));
		  Messages = (StringPtr *)calloc(lineno+1, sizeof(StringPtr));
		  strcpy(CurrentClassName, yypvt[-0].strval);	} break;
case 4:
# line 23 "jitrans.y"
{ WriteClass(CurrentClassName,Entries, Messages,EntryCount); } break;
case 5:
# line 25 "jitrans.y"
{ CurrentClassName = (char *)calloc(strlen(yypvt[-0].strval), sizeof(char));
		  Entries = (StringPtr *)calloc(lineno+1, sizeof(StringPtr));
		  Messages = (StringPtr *)calloc(lineno+1, sizeof(StringPtr));
		  strcpy(CurrentClassName, yypvt[-0].strval);	} break;
case 6:
# line 30 "jitrans.y"
{ WriteGroup(CurrentClassName,Entries, Messages,EntryCount); } break;
case 10:
# line 38 "jitrans.y"
{ Entries[EntryCount] = 
			(StringPtr)calloc(strlen(yypvt[-4].strval)+1, sizeof(char));
		  Messages[EntryCount] = 
			(StringPtr)calloc(strlen(yypvt[-2].strval)+1, sizeof(char));
		  strcpy(Entries[EntryCount], yypvt[-4].strval);
		  strcpy(Messages[EntryCount], yypvt[-2].strval);
		  EntryCount++;	} break;
	}
	goto yystack;		/* reset registers in driver code */
}

