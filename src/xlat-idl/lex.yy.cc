#include <stdio.h>
# define U(x) x
# define NLSTATE yyprevious=YYNEWLINE
# define BEGIN yybgin = yysvec + 1 +
# define INITIAL 0
# define YYLERR yysvec
# define YYSTATE (yyestate-yysvec-1)
# define YYOPTIM 1
# define YYLMAX BUFSIZ
#ifndef __cplusplus
# define output(c) (void)putc(c,yyout)
#else
# define lex_output(c) (void)putc(c,yyout)
#endif

#if defined(__cplusplus) || defined(__STDC__)

#if defined(__cplusplus) && defined(__EXTERN_C__)
extern "C" {
#endif
	int yyback(int *, int);
	int yyinput(void);
	int yylook(void);
	void yyoutput(int);
	int yyracc(int);
	int yyreject(void);
	void yyunput(int);
	int yylex(void);
#ifdef YYLEX_E
	void yywoutput(wchar_t);
	wchar_t yywinput(void);
#endif
#ifndef yyless
	int yyless(int);
#endif
#ifndef yywrap
	int yywrap(void);
#endif
#ifdef LEXDEBUG
	void allprint(char);
	void sprint(char *);
#endif
#if defined(__cplusplus) && defined(__EXTERN_C__)
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
	void exit(int);
#ifdef __cplusplus
}
#endif

#endif
# define unput(c) {yytchar= (c);if(yytchar=='\n')yylineno--;*yysptr++=yytchar;}
# define yymore() (yymorfg=1)
#ifndef __cplusplus
# define input() (((yytchar=yysptr>yysbuf?U(*--yysptr):getc(yyin))==10?(yylineno++,yytchar):yytchar)==EOF?0:yytchar)
#else
# define lex_input() (((yytchar=yysptr>yysbuf?U(*--yysptr):getc(yyin))==10?(yylineno++,yytchar):yytchar)==EOF?0:yytchar)
#endif
#define ECHO fprintf(yyout, "%s",yytext)
# define REJECT { nstr = yyreject(); goto yyfussy;}
int yyleng;
char yytext[YYLMAX];
int yymorfg;
extern char *yysptr, yysbuf[];
int yytchar;
FILE *yyin = {stdin}, *yyout = {stdout};
extern int yylineno;
struct yysvf { 
	struct yywork *yystoff;
	struct yysvf *yyother;
	int *yystops;};
struct yysvf *yyestate;
extern struct yysvf yysvec[], *yybgin;

# line 4 "idl.ll"
/*

COPYRIGHT

Copyright 1992, 1993, 1994 Sun Microsystems, Inc.  Printed in the United
States of America.  All Rights Reserved.

This product is protected by copyright and distributed under the following
license restricting its use.

The Interface Definition Language Compiler Front End (CFE) is made
available for your use provided that you include this license and copyright
notice on all media and documentation and the software program in which
this product is incorporated in whole or part. You may copy and extend
functionality (but may not remove functionality) of the Interface
Definition Language CFE without charge, but you are not authorized to
license or distribute it to anyone else except as part of a product or
program developed by you or with the express written consent of Sun
Microsystems, Inc. ("Sun").

The names of Sun Microsystems, Inc. and any of its subsidiaries or
affiliates may not be used in advertising or publicity pertaining to
distribution of Interface Definition Language CFE as permitted herein.

This license is effective until terminated by Sun for failure to comply
with this license.  Upon termination, you shall destroy or return all code
and documentation for the Interface Definition Language CFE.

INTERFACE DEFINITION LANGUAGE CFE IS PROVIDED AS IS WITH NO WARRANTIES OF
ANY KIND INCLUDING THE WARRANTIES OF DESIGN, MERCHANTIBILITY AND FITNESS
FOR A PARTICULAR PURPOSE, NONINFRINGEMENT, OR ARISING FROM A COURSE OF
DEALING, USAGE OR TRADE PRACTICE.

INTERFACE DEFINITION LANGUAGE CFE IS PROVIDED WITH NO SUPPORT AND WITHOUT
ANY OBLIGATION ON THE PART OF Sun OR ANY OF ITS SUBSIDIARIES OR AFFILIATES
TO ASSIST IN ITS USE, CORRECTION, MODIFICATION OR ENHANCEMENT.

SUN OR ANY OF ITS SUBSIDIARIES OR AFFILIATES SHALL HAVE NO LIABILITY WITH
RESPECT TO THE INFRINGEMENT OF COPYRIGHTS, TRADE SECRETS OR ANY PATENTS BY
INTERFACE DEFINITION LANGUAGE CFE OR ANY PART THEREOF.

IN NO EVENT WILL SUN OR ANY OF ITS SUBSIDIARIES OR AFFILIATES BE LIABLE FOR
ANY LOST REVENUE OR PROFITS OR OTHER SPECIAL, INDIRECT AND CONSEQUENTIAL
DAMAGES, EVEN IF SUN HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

Use, duplication, or disclosure by the government is subject to
restrictions as set forth in subparagraph (c)(1)(ii) of the Rights in
Technical Data and Computer Software clause at DFARS 252.227-7013 and FAR
52.227-19.

Sun, Sun Microsystems and the Sun logo are trademarks or registered
trademarks of Sun Microsystems, Inc.

SunSoft, Inc.  
2550 Garcia Avenue 
Mountain View, California  94043

NOTE:

SunOS, SunSoft, Sun, Solaris, Sun Microsystems or the Sun logo are
trademarks or registered trademarks of Sun Microsystems, Inc.

 */


# line 68 "idl.ll"
/*
 * idl.ll - Lexical scanner for IDL 1.1
 */

#include <idl.hh>
#include <idl_extern.hh>

#include <fe_private.hh>

#include <y.tab.hh>

#include <string.h>

static char	idl_escape_reader(char *);
static double	idl_atof(char *);
static long	idl_atoi(char *, long);
static void	idl_parse_line_and_file(char *);
static void	idl_store_pragma(char *);

// HPUX has yytext typed to unsigned char *. We make sure here that
// we'll always use char *
static char	*__yytext = (char *) yytext;

# define YYNEWLINE 10
yylex(){
int nstr; extern int yyprevious;
#ifdef __cplusplus
/* to avoid CC and lint complaining yyfussy not being used ...*/
static int __lex_hack = 0;
if (__lex_hack) goto yyfussy;
#endif
while((nstr = yylook()) >= 0)
yyfussy: switch(nstr){
case 0:
if(yywrap()) return(0); break;
case 1:

# line 94 "idl.ll"
	return MODULE;
break;
case 2:

# line 95 "idl.ll"
	return RAISES;
break;
case 3:

# line 96 "idl.ll"
return READONLY;
break;
case 4:

# line 97 "idl.ll"
return ATTRIBUTE;
break;
case 5:

# line 98 "idl.ll"
return EXCEPTION;
break;
case 6:

# line 99 "idl.ll"
	return CONTEXT;
break;
case 7:

# line 100 "idl.ll"
return INTERFACE;
break;
case 8:

# line 101 "idl.ll"
	return CONST;
break;
case 9:

# line 102 "idl.ll"
	return TYPEDEF;
break;
case 10:

# line 103 "idl.ll"
	return STRUCT;
break;
case 11:

# line 104 "idl.ll"
	return ENUM;
break;
case 12:

# line 105 "idl.ll"
	return STRING;
break;
case 13:

# line 106 "idl.ll"
return WSTRING;
break;
case 14:

# line 107 "idl.ll"
return SEQUENCE;
break;
case 15:

# line 108 "idl.ll"
	return UNION;
break;
case 16:

# line 109 "idl.ll"
	return SWITCH;
break;
case 17:

# line 110 "idl.ll"
	return CASE;
break;
case 18:

# line 111 "idl.ll"
	return DEFAULT;
break;
case 19:

# line 112 "idl.ll"
	return FLOAT;
break;
case 20:

# line 113 "idl.ll"
	return DOUBLE;
break;
case 21:

# line 114 "idl.ll"
	return LONG;
break;
case 22:

# line 115 "idl.ll"
	return SHORT;
break;
case 23:

# line 116 "idl.ll"
return UNSIGNED;
break;
case 24:

# line 117 "idl.ll"
	return CHAR;
break;
case 25:

# line 118 "idl.ll"
	return WCHAR;
break;
case 26:

# line 119 "idl.ll"
	return BOOLEAN;
break;
case 27:

# line 120 "idl.ll"
	return OCTET;
break;
case 28:

# line 121 "idl.ll"
	return VOID;
break;
case 29:

# line 123 "idl.ll"
	return TRUETOK;
break;
case 30:

# line 124 "idl.ll"
	return FALSETOK;
break;
case 31:

# line 126 "idl.ll"
	return INOUT;
break;
case 32:

# line 127 "idl.ll"
	return IN;
break;
case 33:

# line 128 "idl.ll"
	return OUT;
break;
case 34:

# line 129 "idl.ll"
	return ONEWAY;
break;
case 35:

# line 130 "idl.ll"
return THREADED;
break;
case 36:

# line 132 "idl.ll"
	return LEFT_SHIFT;
break;
case 37:

# line 133 "idl.ll"
	return RIGHT_SHIFT;
break;
case 38:

# line 134 "idl.ll"
	{
		  yylval.strval = "::";    
		  return SCOPE_DELIMITOR;
		}
break;
case 39:

# line 139 "idl.ll"
{
    char *z = (char *) malloc(strlen(__yytext) + 1);
    strcpy(z, __yytext);
    yylval.strval = z;
    return IDENTIFIER;
}
break;
case 40:

# line 146 "idl.ll"
     {
                  yylval.dval = idl_atof(__yytext);
                  return FLOATING_PT_LITERAL;
                }
break;
case 41:

# line 150 "idl.ll"
 {
                  yylval.dval = idl_atof(__yytext);
                  return FLOATING_PT_LITERAL;
                }
break;
case 42:

# line 155 "idl.ll"
{
		  yylval.ival = idl_atoi(__yytext, 10);
		  return INTEGER_LITERAL;
	        }
break;
case 43:

# line 159 "idl.ll"
{
		  yylval.ival = idl_atoi(__yytext, 16);
		  return INTEGER_LITERAL;
	        }
break;
case 44:

# line 163 "idl.ll"
{
		  yylval.ival = idl_atoi(__yytext, 8);
		  return INTEGER_LITERAL;
	      	}
break;
case 45:

# line 168 "idl.ll"
{
		  __yytext[strlen(__yytext)-1] = '\0';
		  yylval.sval = new String(__yytext + 1);
		  return STRING_LITERAL;
	      	}
break;
case 46:

# line 173 "idl.ll"
	{
		  yylval.cval = __yytext[1];
		  return CHARACTER_LITERAL;
	      	}
break;
case 47:

# line 177 "idl.ll"
{
		  // octal character constant
		  yylval.cval = idl_escape_reader(__yytext + 1);
		  return CHARACTER_LITERAL;
		}
break;
case 48:

# line 182 "idl.ll"
{
		  yylval.cval = idl_escape_reader(__yytext + 1);
		  return CHARACTER_LITERAL;
		}
break;
case 49:

# line 186 "idl.ll"
{/* remember pragma */
  		  idl_global->set_lineno(idl_global->lineno() + 1);
		  idl_store_pragma(__yytext);
		}
break;
case 50:

# line 190 "idl.ll"
	{
		  idl_parse_line_and_file(__yytext);
		}
break;
case 51:

# line 193 "idl.ll"
		{
		  idl_parse_line_and_file(__yytext);
		}
break;
case 52:

# line 196 "idl.ll"
{
		  idl_parse_line_and_file(__yytext);
	        }
break;
case 53:

# line 199 "idl.ll"
{
		  /* ignore cpp ident */
  		  idl_global->set_lineno(idl_global->lineno() + 1);
		}
break;
case 54:

# line 203 "idl.ll"
{
		  /* ignore comments */
  		  idl_global->set_lineno(idl_global->lineno() + 1);
		}
break;
case 55:

# line 207 "idl.ll"
	{
		  for(;;) {
		    char c = yyinput();
		    if (c == '*') {
		      char next = yyinput();
		      if (next == '/')
			break;
		      else
			yyunput(c);
	              if (c == '\n') 
		        idl_global->set_lineno(idl_global->lineno() + 1);
		    }
	          }
	        }
break;
case 56:

# line 221 "idl.ll"
	;
break;
case 57:

# line 222 "idl.ll"
	{
  		  idl_global->set_lineno(idl_global->lineno() + 1);
		}
break;
case 58:

# line 225 "idl.ll"
	return __yytext[0];
break;
case -1:
break;
default:
(void)fprintf(yyout,"bad switch yylook %d",nstr);
} return(0); }
/* end of yylex */
	/* subroutines */

/*
 * Strip down a name to the last component, i.e. everything after the last
 * '/' character
 */
static char *
stripped_name(UTL_String *fn)
{
    char	*n = fn->get_string();
    long	l;

    if (n == NULL)
	return NULL;
    l = strlen(n);
    for (n += l; l > 0 && *n != '/'; l--, n--);
    if (*n == '/') n++;
    return n;
}

/*
 * Parse a #line statement generated by the C preprocessor
 */
static void
idl_parse_line_and_file(char *buf)
{
  char		*r = buf;
  char 		*h;
  UTL_String	*nm;

  /* Skip initial '#' */
  if (*r != '#') {
    return;
  }

  /* Find line number */
  for (r++; *r == ' ' || *r == '\t'; r++);
  h = r;
  for (; *r != '\0' && *r != ' ' && *r != '\t'; r++);
  *r++ = 0;
  idl_global->set_lineno(idl_atoi(h, 10));

  /* Find file name, if present */
  for (; *r != '"'; r++) {
    if (*r == '\n' || *r == '\0')
      return;
  }
  h = ++r;
  for (; *r != '"'; r++);
  *r = 0;
  if (*h == '\0')
    idl_global->set_filename(new String("standard input"));
  else
    idl_global->set_filename(new String(h));

  idl_global->set_in_main_file(
    (idl_global->filename()->compare(idl_global->real_filename())) ?
    I_TRUE :
    I_FALSE
  );
  /*
   * If it's an import file store the stripped name for the BE to use
   */
  if (!(idl_global->in_main_file()) && idl_global->import()) {
    nm = new UTL_String(stripped_name(idl_global->filename()));
    idl_global->store_include_file_name(nm);
  }
}

/*
 * Store a #pragma line into the list of pragmas
 */
static void
idl_store_pragma(char *buf)
{
  char *cp = buf + 1;
  while(*cp != 'p')
    cp++;
  while(*cp != ' ' && *cp != '\t')
    cp++;
  while(*cp == ' ' || *cp == '\t')
    cp++;
  char pragma[80];
  char *pp = pragma;
  while(*cp != '\n') {
    *pp++ = *cp++;
  }
  *pp = 0;
  if (strcmp(pragma, "import") == 0) {
    idl_global->set_import(I_TRUE);
    return;
  }
  if (strcmp(pragma, "include") == 0) {
    idl_global->set_import(I_FALSE);
    return;
  }
  UTL_StrList *p = idl_global->pragmas();
  if (p == NULL)
    idl_global->set_pragmas(new UTL_StrList(new String(buf), NULL));
  else {
    p->nconc(new UTL_StrList(new String(buf), NULL));
    idl_global->set_pragmas(p);
  }
}

/*
 * idl_atoi - Convert a string of digits into an integer according to base b
 */
static long
idl_atoi(char *s, long b)
{
	long	r = 0;
	long	negative = 0;

	if (*s == '-') {
	  negative = 1;
	  s++;
	}
	if (b == 8 && *s == '0')
	  s++;
	else if (b == 16 && *s == '0' && (*(s + 1) == 'x' || *(s + 1) == 'X'))
	  s += 2;

	for (; *s; s++)
	  if (*s <= '9' && *s >= '0')
	    r = (r * b) + (*s - '0');
	  else if (b > 10 && *s <= 'f' && *s >= 'a')
	    r = (r * b) + (*s - 'a' + 10);
	  else if (b > 10 && *s <= 'F' && *s >= 'A')
	    r = (r * b) + (*s - 'A' + 10);
	  else
	    break;

	if (negative)
	  r *= -1;

	return r;
}

/*
 * Convert a string to a float; atof doesn't seem to work, always.
 */
static double
idl_atof(char *s)
{
	char    *h = s;
	double	d = 0.0;
	double	f = 0.0;
	double	e, k;
	long	neg = 0, negexp = 0;

	if (*s == '-') {
	  neg = 1;
	  s++;
	}
	while (*s >= '0' && *s <= '9') {
		d = (d * 10) + *s - '0';
		s++;
	}
	if (*s == '.') {
		s++;
		e = 10;
		while (*s >= '0' && *s <= '9') {
			d += (*s - '0') / (e * 1.0);
			e *= 10;
			s++;
		}
	}
	if (*s == 'e' || *s == 'E') {
		s++;
		if (*s == '-') {
			negexp = 1;
			s++;
		} else if (*s == '+')
			s++;
		e = 0;
		while (*s >= '0' && *s <= '9') {
			e = (e * 10) + *s - '0';
			s++;
		}
		if (e > 0) {
			for (k = 1; e > 0; k *= 10, e--);
			if (negexp)
				d /= k;
			else
				d *= k;
		}
	}

	if (neg) d *= -1.0;

	return d;
}

/*
 * Convert (some) escaped characters into their ascii values
 */
static char
idl_escape_reader(
    char *str
)
{
    if (str[0] != '\\') {
	return str[0];
    }

    switch (str[1]) {
      case 'n':
	return '\n';
      case 't':
	return '\t';
      case 'v':
	return '\v';
      case 'b':
	return '\b';
      case 'r':
	return '\r';
      case 'f':
	return '\f';
      case 'a':
	return '\a';
      case '\\':
	return '\\';
      case '\?':
	return '?';
      case '\'':
	return '\'';
      case '"':
	return '"';
      case 'x':
	{
	    // hex value
	    for (int i = 2; str[i] != '\0' && isxdigit(str[i]); i++) {
		continue;
	    }
	    char save = str[i];
	    str[i] = '\0';
	    char out = (char)idl_atoi(&str[2], 16);
	    str[i] = save;
	    return out;
	}
	break;
      default:
	// check for octal value
	if (str[1] >= '0' && str[1] <= '7') {
	    for (int i = 1; str[i] >= '0' && str[i] <= '7'; i++) {
		continue;
	    }
	    char save = str[i];
	    str[i] = '\0';
	    char out = (char)idl_atoi(&str[1], 8);
	    str[i] = save;
	    return out;
	} else {
	  return str[1] - 'a';
	}
	break;
    }
}
int yyvstop[] = {
0,

56,
0, 

56,
0, 

58,
0, 

56,
58,
0, 

57,
0, 

58,
0, 

58,
0, 

58,
0, 

58,
0, 

44,
58,
0, 

42,
58,
0, 

58,
0, 

58,
0, 

58,
0, 

39,
58,
0, 

39,
58,
0, 

39,
58,
0, 

39,
58,
0, 

39,
58,
0, 

39,
58,
0, 

39,
58,
0, 

39,
58,
0, 

39,
58,
0, 

39,
58,
0, 

39,
58,
0, 

39,
58,
0, 

39,
58,
0, 

39,
58,
0, 

39,
58,
0, 

39,
58,
0, 

39,
58,
0, 

39,
58,
0, 

39,
58,
0, 

58,
0, 

56,
0, 

45,
0, 

44,
0, 

42,
0, 

55,
0, 

40,
0, 

44,
0, 

38,
0, 

36,
0, 

37,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

32,
39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

52,
0, 

46,
0, 

46,
0, 

54,
0, 

40,
0, 

41,
0, 

43,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

33,
39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

48,
0, 

47,
48,
0, 

40,
0, 

41,
0, 

39,
0, 

29,
39,
0, 

39,
0, 

39,
0, 

17,
39,
0, 

24,
39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

11,
39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

21,
39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

28,
39,
0, 

39,
0, 

39,
0, 

47,
0, 

30,
39,
0, 

39,
0, 

39,
0, 

8,
39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

19,
39,
0, 

31,
39,
0, 

39,
0, 

39,
0, 

27,
39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

22,
39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

15,
39,
0, 

39,
0, 

39,
0, 

39,
0, 

51,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

20,
39,
0, 

39,
0, 

39,
0, 

1,
39,
0, 

34,
39,
0, 

2,
39,
0, 

39,
0, 

39,
0, 

12,
39,
0, 

10,
39,
0, 

16,
39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

50,
0, 

39,
0, 

26,
39,
0, 

6,
39,
0, 

18,
39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

39,
0, 

9,
39,
0, 

39,
0, 

25,
39,
0, 

39,
0, 

53,
0, 

39,
0, 

39,
0, 

39,
0, 

3,
39,
0, 

14,
39,
0, 

35,
39,
0, 

23,
39,
0, 

39,
0, 

4,
39,
0, 

5,
39,
0, 

7,
39,
0, 

13,
39,
0, 

49,
0, 
0};
# define YYTYPE int
struct yywork { YYTYPE verify, advance; } yycrank[] = {
0,0,	0,0,	1,3,	0,0,	
0,0,	0,0,	0,0,	0,0,	
0,0,	0,0,	1,4,	1,5,	
0,0,	4,35,	0,0,	0,0,	
0,0,	0,0,	0,0,	0,0,	
0,0,	0,0,	0,0,	0,0,	
0,0,	0,0,	0,0,	0,0,	
0,0,	0,0,	0,0,	0,0,	
0,0,	0,0,	0,0,	1,6,	
4,35,	0,0,	0,0,	0,0,	
1,7,	0,0,	0,0,	0,0,	
1,3,	0,0,	1,8,	85,131,	
1,9,	1,10,	1,11,	38,89,	
90,135,	0,0,	0,0,	9,42,	
0,0,	1,11,	0,0,	1,12,	
9,43,	1,13,	0,0,	1,14,	
0,0,	0,0,	1,15,	11,44,	
0,0,	11,41,	1,15,	1,16,	
1,15,	12,49,	13,50,	14,51,	
0,0,	1,15,	16,53,	0,0,	
0,0,	0,0,	0,0,	100,142,	
0,0,	1,17,	0,0,	0,0,	
53,99,	1,15,	11,47,	0,0,	
0,0,	132,131,	17,54,	0,0,	
1,3,	54,100,	1,18,	1,19,	
1,20,	1,21,	1,22,	1,23,	
99,141,	6,36,	1,24,	0,0,	
0,0,	1,25,	1,26,	58,104,	
1,27,	6,36,	6,36,	1,28,	
1,29,	1,30,	1,31,	1,32,	
1,33,	2,34,	11,47,	19,56,	
23,64,	2,7,	21,60,	20,57,	
18,55,	22,62,	24,65,	2,8,	
25,66,	2,9,	20,58,	26,67,	
21,61,	30,77,	6,37,	22,63,	
27,68,	20,59,	31,79,	32,80,	
2,12,	28,71,	2,13,	6,36,	
2,14,	28,72,	33,81,	27,69,	
6,36,	6,36,	30,78,	41,44,	
55,101,	41,41,	27,70,	56,102,	
6,36,	29,73,	57,103,	59,105,	
29,74,	60,106,	33,82,	61,107,	
62,108,	6,36,	2,17,	63,109,	
64,110,	6,36,	6,36,	6,36,	
29,75,	65,111,	41,47,	29,76,	
6,36,	7,38,	65,112,	2,18,	
2,19,	2,20,	2,21,	2,22,	
2,23,	7,38,	7,0,	2,24,	
6,36,	66,113,	2,25,	2,26,	
67,114,	2,27,	68,115,	6,36,	
2,28,	2,29,	2,30,	2,31,	
2,32,	2,33,	69,116,	70,117,	
71,118,	72,119,	41,47,	73,120,	
74,121,	75,122,	7,38,	46,46,	
46,46,	46,46,	46,46,	46,46,	
46,46,	46,46,	46,46,	7,38,	
76,123,	77,124,	78,125,	79,126,	
7,38,	7,38,	80,128,	81,129,	
82,130,	87,133,	88,134,	97,140,	
7,38,	79,127,	101,143,	102,144,	
103,145,	97,140,	104,146,	105,147,	
105,148,	7,38,	106,149,	107,150,	
108,151,	7,38,	7,38,	7,38,	
109,152,	110,153,	111,154,	112,155,	
7,38,	8,40,	8,41,	8,41,	
8,41,	8,41,	8,41,	8,41,	
8,41,	8,41,	8,41,	97,140,	
7,38,	113,156,	114,157,	115,158,	
7,39,	97,140,	10,44,	7,38,	
10,45,	10,45,	10,45,	10,45,	
10,45,	10,45,	10,45,	10,45,	
10,46,	10,46,	116,159,	118,160,	
119,161,	120,162,	121,163,	123,166,	
124,167,	47,96,	125,168,	47,96,	
126,169,	10,47,	47,97,	47,97,	
47,97,	47,97,	47,97,	47,97,	
47,97,	47,97,	47,97,	47,97,	
92,136,	127,170,	128,171,	129,172,	
130,173,	122,164,	133,175,	134,176,	
10,48,	92,137,	92,137,	92,137,	
92,137,	92,137,	92,137,	92,137,	
92,137,	122,165,	141,179,	143,180,	
144,181,	10,47,	147,182,	148,183,	
15,52,	15,52,	15,52,	15,52,	
15,52,	15,52,	15,52,	15,52,	
15,52,	15,52,	149,184,	150,185,	
152,186,	153,187,	154,188,	155,189,	
10,48,	15,52,	15,52,	15,52,	
15,52,	15,52,	15,52,	15,52,	
15,52,	15,52,	15,52,	15,52,	
15,52,	15,52,	15,52,	15,52,	
15,52,	15,52,	15,52,	15,52,	
15,52,	15,52,	15,52,	15,52,	
15,52,	15,52,	15,52,	157,190,	
158,191,	159,192,	160,193,	15,52,	
161,194,	15,52,	15,52,	15,52,	
15,52,	15,52,	15,52,	15,52,	
15,52,	15,52,	15,52,	15,52,	
15,52,	15,52,	15,52,	15,52,	
15,52,	15,52,	15,52,	15,52,	
15,52,	15,52,	15,52,	15,52,	
15,52,	15,52,	15,52,	34,83,	
34,84,	94,138,	162,195,	94,138,	
163,196,	164,197,	94,139,	94,139,	
94,139,	94,139,	94,139,	94,139,	
94,139,	94,139,	94,139,	94,139,	
39,90,	165,198,	166,199,	43,43,	
137,177,	167,200,	34,85,	168,201,	
39,90,	39,0,	169,202,	43,43,	
43,93,	137,178,	137,178,	137,178,	
137,178,	137,178,	137,178,	137,178,	
137,178,	170,203,	34,86,	34,86,	
34,86,	34,86,	34,86,	34,86,	
34,86,	34,86,	34,86,	34,86,	
172,204,	39,90,	173,205,	174,206,	
43,43,	175,208,	39,91,	176,209,	
178,177,	180,210,	39,90,	181,211,	
183,212,	43,43,	184,213,	39,92,	
39,92,	185,214,	43,43,	43,43,	
186,215,	139,95,	189,216,	39,90,	
190,217,	174,207,	43,43,	139,95,	
192,218,	193,219,	194,220,	195,221,	
39,90,	197,222,	198,223,	43,43,	
39,90,	39,90,	39,90,	43,43,	
43,43,	43,43,	199,224,	39,90,	
200,225,	201,226,	43,43,	34,87,	
203,227,	204,228,	205,229,	208,231,	
209,232,	139,95,	34,88,	39,90,	
210,233,	211,234,	43,43,	139,95,	
212,235,	213,236,	39,90,	215,237,	
216,238,	43,43,	44,44,	44,44,	
44,44,	44,44,	44,44,	44,44,	
44,44,	44,44,	44,44,	44,44,	
96,97,	96,97,	96,97,	96,97,	
96,97,	96,97,	96,97,	96,97,	
96,97,	96,97,	220,239,	44,94,	
44,95,	221,240,	225,241,	226,242,	
227,243,	45,44,	44,95,	45,45,	
45,45,	45,45,	45,45,	45,45,	
45,45,	45,45,	45,45,	45,46,	
45,46,	138,139,	138,139,	138,139,	
138,139,	138,139,	138,139,	138,139,	
138,139,	138,139,	138,139,	228,244,	
45,47,	229,245,	232,247,	44,94,	
44,95,	86,84,	233,248,	237,249,	
238,250,	239,251,	44,95,	240,252,	
241,253,	243,254,	245,255,	207,230,	
48,98,	48,98,	48,98,	48,98,	
48,98,	48,98,	48,98,	48,98,	
48,98,	48,98,	248,257,	86,132,	
249,258,	250,259,	255,260,	131,131,	
45,47,	48,98,	48,98,	48,98,	
48,98,	48,98,	48,98,	131,131,	
131,131,	0,0,	0,0,	86,86,	
86,86,	86,86,	86,86,	86,86,	
86,86,	86,86,	86,86,	86,86,	
86,86,	207,207,	207,207,	207,207,	
207,207,	207,207,	207,207,	207,207,	
207,207,	207,207,	207,207,	0,0,	
131,174,	48,98,	48,98,	48,98,	
48,98,	48,98,	48,98,	247,256,	
231,231,	131,131,	0,0,	256,256,	
0,0,	0,0,	131,131,	131,131,	
231,231,	231,246,	0,0,	256,256,	
256,261,	0,0,	131,131,	0,0,	
0,0,	0,0,	0,0,	0,0,	
0,0,	0,0,	247,256,	131,131,	
0,0,	0,0,	0,0,	131,131,	
131,131,	131,131,	0,0,	0,0,	
0,0,	231,231,	131,131,	0,0,	
256,256,	0,0,	0,0,	0,0,	
0,0,	0,0,	231,231,	0,0,	
0,0,	256,256,	131,131,	231,231,	
231,231,	0,0,	256,256,	256,256,	
0,0,	131,131,	0,0,	231,231,	
0,0,	0,0,	256,256,	0,0,	
0,0,	0,0,	0,0,	0,0,	
231,231,	0,0,	0,0,	256,256,	
231,231,	231,231,	231,231,	256,256,	
256,256,	256,256,	0,0,	231,231,	
0,0,	0,0,	256,256,	0,0,	
0,0,	0,0,	0,0,	0,0,	
0,0,	0,0,	0,0,	231,231,	
0,0,	0,0,	256,256,	0,0,	
0,0,	0,0,	231,231,	0,0,	
0,0,	256,256,	0,0,	0,0,	
0,0};
struct yysvf yysvec[] = {
0,	0,	0,
yycrank+-1,	0,		yyvstop+1,
yycrank+-86,	yysvec+1,	yyvstop+3,
yycrank+0,	0,		yyvstop+5,
yycrank+4,	0,		yyvstop+7,
yycrank+0,	0,		yyvstop+10,
yycrank+-104,	0,		yyvstop+12,
yycrank+-180,	0,		yyvstop+14,
yycrank+209,	0,		yyvstop+16,
yycrank+13,	0,		yyvstop+18,
yycrank+228,	0,		yyvstop+20,
yycrank+21,	yysvec+8,	yyvstop+23,
yycrank+15,	0,		yyvstop+26,
yycrank+14,	0,		yyvstop+28,
yycrank+13,	0,		yyvstop+30,
yycrank+284,	0,		yyvstop+32,
yycrank+13,	yysvec+15,	yyvstop+35,
yycrank+12,	yysvec+15,	yyvstop+38,
yycrank+12,	yysvec+15,	yyvstop+41,
yycrank+12,	yysvec+15,	yyvstop+44,
yycrank+30,	yysvec+15,	yyvstop+47,
yycrank+25,	yysvec+15,	yyvstop+50,
yycrank+19,	yysvec+15,	yyvstop+53,
yycrank+16,	yysvec+15,	yyvstop+56,
yycrank+20,	yysvec+15,	yyvstop+59,
yycrank+21,	yysvec+15,	yyvstop+62,
yycrank+24,	yysvec+15,	yyvstop+65,
yycrank+41,	yysvec+15,	yyvstop+68,
yycrank+48,	yysvec+15,	yyvstop+71,
yycrank+60,	yysvec+15,	yyvstop+74,
yycrank+33,	yysvec+15,	yyvstop+77,
yycrank+32,	yysvec+15,	yyvstop+80,
yycrank+32,	yysvec+15,	yyvstop+83,
yycrank+51,	yysvec+15,	yyvstop+86,
yycrank+398,	0,		yyvstop+89,
yycrank+0,	yysvec+4,	yyvstop+91,
yycrank+0,	yysvec+6,	0,	
yycrank+0,	0,		yyvstop+93,
yycrank+12,	0,		0,	
yycrank+-423,	0,		0,	
yycrank+0,	yysvec+10,	yyvstop+95,
yycrank+109,	yysvec+8,	yyvstop+97,
yycrank+0,	0,		yyvstop+99,
yycrank+-426,	0,		0,	
yycrank+474,	0,		yyvstop+101,
yycrank+503,	0,		yyvstop+103,
yycrank+167,	yysvec+45,	0,	
yycrank+250,	0,		0,	
yycrank+540,	0,		0,	
yycrank+0,	0,		yyvstop+105,
yycrank+0,	0,		yyvstop+107,
yycrank+0,	0,		yyvstop+109,
yycrank+0,	yysvec+15,	yyvstop+111,
yycrank+12,	yysvec+15,	yyvstop+113,
yycrank+12,	yysvec+15,	yyvstop+115,
yycrank+40,	yysvec+15,	yyvstop+117,
yycrank+48,	yysvec+15,	yyvstop+119,
yycrank+47,	yysvec+15,	yyvstop+121,
yycrank+14,	yysvec+15,	yyvstop+123,
yycrank+53,	yysvec+15,	yyvstop+125,
yycrank+63,	yysvec+15,	yyvstop+127,
yycrank+50,	yysvec+15,	yyvstop+129,
yycrank+51,	yysvec+15,	yyvstop+131,
yycrank+72,	yysvec+15,	yyvstop+133,
yycrank+61,	yysvec+15,	yyvstop+135,
yycrank+66,	yysvec+15,	yyvstop+137,
yycrank+83,	yysvec+15,	yyvstop+140,
yycrank+96,	yysvec+15,	yyvstop+142,
yycrank+82,	yysvec+15,	yyvstop+144,
yycrank+105,	yysvec+15,	yyvstop+146,
yycrank+91,	yysvec+15,	yyvstop+148,
yycrank+103,	yysvec+15,	yyvstop+150,
yycrank+112,	yysvec+15,	yyvstop+152,
yycrank+98,	yysvec+15,	yyvstop+154,
yycrank+101,	yysvec+15,	yyvstop+156,
yycrank+99,	yysvec+15,	yyvstop+158,
yycrank+119,	yysvec+15,	yyvstop+160,
yycrank+111,	yysvec+15,	yyvstop+162,
yycrank+114,	yysvec+15,	yyvstop+164,
yycrank+122,	yysvec+15,	yyvstop+166,
yycrank+125,	yysvec+15,	yyvstop+168,
yycrank+127,	yysvec+15,	yyvstop+170,
yycrank+116,	yysvec+15,	yyvstop+172,
yycrank+0,	yysvec+34,	0,	
yycrank+0,	0,		yyvstop+174,
yycrank+13,	yysvec+34,	0,	
yycrank+567,	0,		0,	
yycrank+133,	0,		0,	
yycrank+120,	0,		0,	
yycrank+0,	0,		yyvstop+176,
yycrank+13,	0,		0,	
yycrank+0,	yysvec+90,	yyvstop+178,
yycrank+269,	0,		0,	
yycrank+0,	0,		yyvstop+180,
yycrank+366,	0,		0,	
yycrank+0,	0,		yyvstop+182,
yycrank+484,	0,		0,	
yycrank+165,	yysvec+96,	yyvstop+184,
yycrank+0,	yysvec+48,	yyvstop+186,
yycrank+21,	yysvec+15,	yyvstop+188,
yycrank+14,	yysvec+15,	yyvstop+190,
yycrank+124,	yysvec+15,	yyvstop+192,
yycrank+131,	yysvec+15,	yyvstop+194,
yycrank+139,	yysvec+15,	yyvstop+196,
yycrank+128,	yysvec+15,	yyvstop+198,
yycrank+128,	yysvec+15,	yyvstop+200,
yycrank+149,	yysvec+15,	yyvstop+202,
yycrank+149,	yysvec+15,	yyvstop+204,
yycrank+139,	yysvec+15,	yyvstop+206,
yycrank+151,	yysvec+15,	yyvstop+208,
yycrank+156,	yysvec+15,	yyvstop+210,
yycrank+137,	yysvec+15,	yyvstop+212,
yycrank+154,	yysvec+15,	yyvstop+214,
yycrank+166,	yysvec+15,	yyvstop+216,
yycrank+153,	yysvec+15,	yyvstop+218,
yycrank+170,	yysvec+15,	yyvstop+220,
yycrank+167,	yysvec+15,	yyvstop+222,
yycrank+0,	yysvec+15,	yyvstop+224,
yycrank+172,	yysvec+15,	yyvstop+227,
yycrank+188,	yysvec+15,	yyvstop+229,
yycrank+172,	yysvec+15,	yyvstop+231,
yycrank+176,	yysvec+15,	yyvstop+233,
yycrank+208,	yysvec+15,	yyvstop+235,
yycrank+175,	yysvec+15,	yyvstop+237,
yycrank+191,	yysvec+15,	yyvstop+239,
yycrank+193,	yysvec+15,	yyvstop+241,
yycrank+185,	yysvec+15,	yyvstop+243,
yycrank+204,	yysvec+15,	yyvstop+245,
yycrank+210,	yysvec+15,	yyvstop+247,
yycrank+214,	yysvec+15,	yyvstop+249,
yycrank+198,	yysvec+15,	yyvstop+251,
yycrank+-602,	0,		0,	
yycrank+59,	0,		0,	
yycrank+213,	0,		0,	
yycrank+218,	0,		0,	
yycrank+0,	0,		yyvstop+253,
yycrank+0,	0,		yyvstop+255,
yycrank+389,	0,		0,	
yycrank+513,	0,		0,	
yycrank+407,	yysvec+138,	yyvstop+258,
yycrank+0,	0,		yyvstop+260,
yycrank+257,	yysvec+15,	yyvstop+262,
yycrank+0,	yysvec+15,	yyvstop+264,
yycrank+222,	yysvec+15,	yyvstop+267,
yycrank+227,	yysvec+15,	yyvstop+269,
yycrank+0,	yysvec+15,	yyvstop+271,
yycrank+0,	yysvec+15,	yyvstop+274,
yycrank+214,	yysvec+15,	yyvstop+277,
yycrank+230,	yysvec+15,	yyvstop+279,
yycrank+225,	yysvec+15,	yyvstop+281,
yycrank+235,	yysvec+15,	yyvstop+283,
yycrank+0,	yysvec+15,	yyvstop+285,
yycrank+232,	yysvec+15,	yyvstop+288,
yycrank+229,	yysvec+15,	yyvstop+290,
yycrank+230,	yysvec+15,	yyvstop+292,
yycrank+233,	yysvec+15,	yyvstop+294,
yycrank+0,	yysvec+15,	yyvstop+296,
yycrank+267,	yysvec+15,	yyvstop+299,
yycrank+260,	yysvec+15,	yyvstop+301,
yycrank+280,	yysvec+15,	yyvstop+303,
yycrank+277,	yysvec+15,	yyvstop+305,
yycrank+269,	yysvec+15,	yyvstop+307,
yycrank+309,	yysvec+15,	yyvstop+309,
yycrank+296,	yysvec+15,	yyvstop+311,
yycrank+303,	yysvec+15,	yyvstop+313,
yycrank+326,	yysvec+15,	yyvstop+315,
yycrank+327,	yysvec+15,	yyvstop+317,
yycrank+332,	yysvec+15,	yyvstop+319,
yycrank+331,	yysvec+15,	yyvstop+321,
yycrank+324,	yysvec+15,	yyvstop+323,
yycrank+342,	yysvec+15,	yyvstop+325,
yycrank+0,	yysvec+15,	yyvstop+327,
yycrank+342,	yysvec+15,	yyvstop+330,
yycrank+353,	yysvec+15,	yyvstop+332,
yycrank+449,	0,		0,	
yycrank+351,	0,		0,	
yycrank+360,	0,		0,	
yycrank+0,	0,		yyvstop+334,
yycrank+425,	0,		0,	
yycrank+0,	yysvec+15,	yyvstop+336,
yycrank+367,	yysvec+15,	yyvstop+339,
yycrank+370,	yysvec+15,	yyvstop+341,
yycrank+0,	yysvec+15,	yyvstop+343,
yycrank+348,	yysvec+15,	yyvstop+346,
yycrank+362,	yysvec+15,	yyvstop+348,
yycrank+372,	yysvec+15,	yyvstop+350,
yycrank+360,	yysvec+15,	yyvstop+352,
yycrank+0,	yysvec+15,	yyvstop+354,
yycrank+0,	yysvec+15,	yyvstop+357,
yycrank+376,	yysvec+15,	yyvstop+360,
yycrank+379,	yysvec+15,	yyvstop+362,
yycrank+0,	yysvec+15,	yyvstop+364,
yycrank+363,	yysvec+15,	yyvstop+367,
yycrank+370,	yysvec+15,	yyvstop+369,
yycrank+376,	yysvec+15,	yyvstop+371,
yycrank+377,	yysvec+15,	yyvstop+373,
yycrank+0,	yysvec+15,	yyvstop+375,
yycrank+386,	yysvec+15,	yyvstop+378,
yycrank+374,	yysvec+15,	yyvstop+380,
yycrank+394,	yysvec+15,	yyvstop+382,
yycrank+400,	yysvec+15,	yyvstop+384,
yycrank+400,	yysvec+15,	yyvstop+386,
yycrank+0,	yysvec+15,	yyvstop+388,
yycrank+394,	yysvec+15,	yyvstop+391,
yycrank+410,	yysvec+15,	yyvstop+393,
yycrank+396,	yysvec+15,	yyvstop+395,
yycrank+0,	0,		yyvstop+397,
yycrank+577,	0,		0,	
yycrank+391,	0,		0,	
yycrank+399,	0,		0,	
yycrank+395,	yysvec+15,	yyvstop+399,
yycrank+403,	yysvec+15,	yyvstop+401,
yycrank+400,	yysvec+15,	yyvstop+403,
yycrank+401,	yysvec+15,	yyvstop+405,
yycrank+0,	yysvec+15,	yyvstop+407,
yycrank+414,	yysvec+15,	yyvstop+410,
yycrank+423,	yysvec+15,	yyvstop+412,
yycrank+0,	yysvec+15,	yyvstop+414,
yycrank+0,	yysvec+15,	yyvstop+417,
yycrank+0,	yysvec+15,	yyvstop+420,
yycrank+434,	yysvec+15,	yyvstop+423,
yycrank+446,	yysvec+15,	yyvstop+425,
yycrank+0,	yysvec+15,	yyvstop+427,
yycrank+0,	yysvec+15,	yyvstop+430,
yycrank+0,	yysvec+15,	yyvstop+433,
yycrank+445,	yysvec+15,	yyvstop+436,
yycrank+445,	yysvec+15,	yyvstop+438,
yycrank+447,	yysvec+15,	yyvstop+440,
yycrank+455,	yysvec+15,	yyvstop+442,
yycrank+470,	yysvec+15,	yyvstop+444,
yycrank+0,	0,		yyvstop+446,
yycrank+-643,	0,		0,	
yycrank+477,	0,		0,	
yycrank+462,	yysvec+15,	yyvstop+448,
yycrank+0,	yysvec+15,	yyvstop+450,
yycrank+0,	yysvec+15,	yyvstop+453,
yycrank+0,	yysvec+15,	yyvstop+456,
yycrank+468,	yysvec+15,	yyvstop+459,
yycrank+481,	yysvec+15,	yyvstop+461,
yycrank+460,	yysvec+15,	yyvstop+463,
yycrank+482,	yysvec+15,	yyvstop+465,
yycrank+484,	yysvec+15,	yyvstop+467,
yycrank+0,	yysvec+15,	yyvstop+469,
yycrank+485,	yysvec+15,	yyvstop+472,
yycrank+0,	yysvec+15,	yyvstop+474,
yycrank+491,	yysvec+15,	yyvstop+477,
yycrank+0,	0,		yyvstop+479,
yycrank+634,	0,		0,	
yycrank+497,	yysvec+15,	yyvstop+481,
yycrank+490,	yysvec+15,	yyvstop+483,
yycrank+500,	yysvec+15,	yyvstop+485,
yycrank+0,	yysvec+15,	yyvstop+487,
yycrank+0,	yysvec+15,	yyvstop+490,
yycrank+0,	yysvec+15,	yyvstop+493,
yycrank+0,	yysvec+15,	yyvstop+496,
yycrank+486,	yysvec+15,	yyvstop+499,
yycrank+-646,	0,		0,	
yycrank+0,	yysvec+15,	yyvstop+501,
yycrank+0,	yysvec+15,	yyvstop+504,
yycrank+0,	yysvec+15,	yyvstop+507,
yycrank+0,	yysvec+15,	yyvstop+510,
yycrank+0,	0,		yyvstop+513,
0,	0,	0};
struct yywork *yytop = yycrank+741;
struct yysvf *yybgin = yysvec+1;
char yymatch[] = {
  0,   1,   1,   1,   1,   1,   1,   1, 
  1,   9,  10,   1,   1,   1,   1,   1, 
  1,   1,   1,   1,   1,   1,   1,   1, 
  1,   1,   1,   1,   1,   1,   1,   1, 
  9,   1,  34,   1,   1,   1,   1,   1, 
  1,   1,   1,  43,   1,  43,   1,   1, 
 48,  49,  49,  49,  49,  49,  49,  49, 
 56,  56,   1,   1,   1,   1,   1,   1, 
  1,  65,  65,  65,  65,  69,  70,  71, 
 71,  71,  71,  71,  76,  71,  71,  71, 
 71,  71,  71,  71,  71,  71,  71,  71, 
 88,  71,  71,   1,   1,   1,   1,  95, 
  1,  65,  65,  65,  65,  69,  70,  71, 
 71,  71,  71,  71,  76,  71,  71,  71, 
 71,  71,  71,  71,  71,  71,  71,  71, 
 88,  71,  71,   1,   1,   1,   1,   1, 
  1,   1,   1,   1,   1,   1,   1,   1, 
  1,   1,   1,   1,   1,   1,   1,   1, 
  1,   1,   1,   1,   1,   1,   1,   1, 
  1,   1,   1,   1,   1,   1,   1,   1, 
  1,   1,   1,   1,   1,   1,   1,   1, 
  1,   1,   1,   1,   1,   1,   1,   1, 
  1,   1,   1,   1,   1,   1,   1,   1, 
  1,   1,   1,   1,   1,   1,   1,   1, 
  1,   1,   1,   1,   1,   1,   1,   1, 
  1,   1,   1,   1,   1,   1,   1,   1, 
  1,   1,   1,   1,   1,   1,   1,   1, 
  1,   1,   1,   1,   1,   1,   1,   1, 
  1,   1,   1,   1,   1,   1,   1,   1, 
  1,   1,   1,   1,   1,   1,   1,   1, 
  1,   1,   1,   1,   1,   1,   1,   1, 
  1,   1,   1,   1,   1,   1,   1,   1, 
0};
char yyextra[] = {
0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,
0};
/*	Copyright (c) 1989 AT&T	*/
/*	  All Rights Reserved  	*/

/*	THIS IS UNPUBLISHED PROPRIETARY SOURCE CODE OF AT&T	*/
/*	The copyright notice above does not evidence any   	*/
/*	actual or intended publication of such source code.	*/

#pragma ident	"@(#)ncform	6.8	95/02/11 SMI"

int yylineno =1;
# define YYU(x) x
# define NLSTATE yyprevious=YYNEWLINE
struct yysvf *yylstate [YYLMAX], **yylsp, **yyolsp;
char yysbuf[YYLMAX];
char *yysptr = yysbuf;
int *yyfnd;
extern struct yysvf *yyestate;
int yyprevious = YYNEWLINE;
#if defined(__cplusplus) || defined(__STDC__)
int yylook(void)
#else
yylook()
#endif
{
	register struct yysvf *yystate, **lsp;
	register struct yywork *yyt;
	struct yysvf *yyz;
	int yych, yyfirst;
	struct yywork *yyr;
# ifdef LEXDEBUG
	int debug;
# endif
	char *yylastch;
	/* start off machines */
# ifdef LEXDEBUG
	debug = 0;
# endif
	yyfirst=1;
	if (!yymorfg)
		yylastch = yytext;
	else {
		yymorfg=0;
		yylastch = yytext+yyleng;
		}
	for(;;){
		lsp = yylstate;
		yyestate = yystate = yybgin;
		if (yyprevious==YYNEWLINE) yystate++;
		for (;;){
# ifdef LEXDEBUG
			if(debug)fprintf(yyout,"state %d\n",yystate-yysvec-1);
# endif
			yyt = yystate->yystoff;
			if(yyt == yycrank && !yyfirst){  /* may not be any transitions */
				yyz = yystate->yyother;
				if(yyz == 0)break;
				if(yyz->yystoff == yycrank)break;
				}
#ifndef __cplusplus
			*yylastch++ = yych = input();
#else
			*yylastch++ = yych = lex_input();
#endif
			if(yylastch > &yytext[YYLMAX]) {
				fprintf(yyout,"Input string too long, limit %d\n",YYLMAX);
				exit(1);
			}
			yyfirst=0;
		tryagain:
# ifdef LEXDEBUG
			if(debug){
				fprintf(yyout,"char ");
				allprint(yych);
				putchar('\n');
				}
# endif
			yyr = yyt;
			if ( (int)yyt > (int)yycrank){
				yyt = yyr + yych;
				if (yyt <= yytop && yyt->verify+yysvec == yystate){
					if(yyt->advance+yysvec == YYLERR)	/* error transitions */
						{unput(*--yylastch);break;}
					*lsp++ = yystate = yyt->advance+yysvec;
					if(lsp > &yylstate[YYLMAX]) {
						fprintf(yyout,"Input string too long, limit %d\n",YYLMAX);
						exit(1);
					}
					goto contin;
					}
				}
# ifdef YYOPTIM
			else if((int)yyt < (int)yycrank) {		/* r < yycrank */
				yyt = yyr = yycrank+(yycrank-yyt);
# ifdef LEXDEBUG
				if(debug)fprintf(yyout,"compressed state\n");
# endif
				yyt = yyt + yych;
				if(yyt <= yytop && yyt->verify+yysvec == yystate){
					if(yyt->advance+yysvec == YYLERR)	/* error transitions */
						{unput(*--yylastch);break;}
					*lsp++ = yystate = yyt->advance+yysvec;
					if(lsp > &yylstate[YYLMAX]) {
						fprintf(yyout,"Input string too long, limit %d\n",YYLMAX);
						exit(1);
					}
					goto contin;
					}
				yyt = yyr + YYU(yymatch[yych]);
# ifdef LEXDEBUG
				if(debug){
					fprintf(yyout,"try fall back character ");
					allprint(YYU(yymatch[yych]));
					putchar('\n');
					}
# endif
				if(yyt <= yytop && yyt->verify+yysvec == yystate){
					if(yyt->advance+yysvec == YYLERR)	/* error transition */
						{unput(*--yylastch);break;}
					*lsp++ = yystate = yyt->advance+yysvec;
					if(lsp > &yylstate[YYLMAX]) {
						fprintf(yyout,"Input string too long, limit %d\n",YYLMAX);
						exit(1);
					}
					goto contin;
					}
				}
			if ((yystate = yystate->yyother) && (yyt= yystate->yystoff) != yycrank){
# ifdef LEXDEBUG
				if(debug)fprintf(yyout,"fall back to state %d\n",yystate-yysvec-1);
# endif
				goto tryagain;
				}
# endif
			else
				{unput(*--yylastch);break;}
		contin:
# ifdef LEXDEBUG
			if(debug){
				fprintf(yyout,"state %d char ",yystate-yysvec-1);
				allprint(yych);
				putchar('\n');
				}
# endif
			;
			}
# ifdef LEXDEBUG
		if(debug){
			fprintf(yyout,"stopped at %d with ",*(lsp-1)-yysvec-1);
			allprint(yych);
			putchar('\n');
			}
# endif
		while (lsp-- > yylstate){
			*yylastch-- = 0;
			if (*lsp != 0 && (yyfnd= (*lsp)->yystops) && *yyfnd > 0){
				yyolsp = lsp;
				if(yyextra[*yyfnd]){		/* must backup */
					while(yyback((*lsp)->yystops,-*yyfnd) != 1 && lsp > yylstate){
						lsp--;
						unput(*yylastch--);
						}
					}
				yyprevious = YYU(*yylastch);
				yylsp = lsp;
				yyleng = yylastch-yytext+1;
				yytext[yyleng] = 0;
# ifdef LEXDEBUG
				if(debug){
					fprintf(yyout,"\nmatch ");
					sprint(yytext);
					fprintf(yyout," action %d\n",*yyfnd);
					}
# endif
				return(*yyfnd++);
				}
			unput(*yylastch);
			}
		if (yytext[0] == 0  /* && feof(yyin) */)
			{
			yysptr=yysbuf;
			return(0);
			}
#ifndef __cplusplus
		yyprevious = yytext[0] = input();
		if (yyprevious>0)
			output(yyprevious);
#else
		yyprevious = yytext[0] = lex_input();
		if (yyprevious>0)
			lex_output(yyprevious);
#endif
		yylastch=yytext;
# ifdef LEXDEBUG
		if(debug)putchar('\n');
# endif
		}
	}
#if defined(__cplusplus) || defined(__STDC__)
int yyback(int *p, int m)
#else
yyback(p, m)
	int *p;
#endif
{
	if (p==0) return(0);
	while (*p) {
		if (*p++ == m)
			return(1);
	}
	return(0);
}
	/* the following are only used in the lex library */
#if defined(__cplusplus) || defined(__STDC__)
int yyinput(void)
#else
yyinput()
#endif
{
#ifndef __cplusplus
	return(input());
#else
	return(lex_input());
#endif
	}
#if defined(__cplusplus) || defined(__STDC__)
void yyoutput(int c)
#else
yyoutput(c)
  int c; 
#endif
{
#ifndef __cplusplus
	output(c);
#else
	lex_output(c);
#endif
	}
#if defined(__cplusplus) || defined(__STDC__)
void yyunput(int c)
#else
yyunput(c)
   int c; 
#endif
{
	unput(c);
	}
