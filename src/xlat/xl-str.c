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
 * Revision 2.1  1995-06-15 20:57:00  jyelon
 * *** empty log message ***
 *
 * Revision 2.0  1995/06/05  18:52:05  brunner
 * Reorganized file structure
 *
 * Revision 1.1  1994/11/03  17:41:58  brunner
 * Initial revision
 *
 ***************************************************************************/
static char ident[] = "@(#)$Header$";

#include <stdio.h>
#include <string.h>
#include "xl-lex.h"
#include "xl-sym.h"

extern SYMTABPTR SavedFnNode;

extern char *calloc();

char *MakeString(string)
char *string;
{ char *dummy;

  dummy=calloc((1+strlen(string)),sizeof(char));
  if (dummy==NULL) memory_error("Out of Memory in MakeString()",EXIT);
  if (!strcmp("sizesLEFT",string))
  	{ printf ("0x%x dummy\n",dummy); }
  strcpy(dummy,string);
  return(dummy);
}

char *MakeString_n(a,n)
char **a;
int n;
{ char *dummy;
  int i=0;
  int j,k;
  
  for (j=0;j<n;j++)	i += strlen(*(a+j));
  dummy=calloc(i+1,sizeof(char));
  if (dummy==NULL) memory_error("Out of Memory in MakeString()",EXIT);
  k=0;
  for (j=0;j<n;j++)
	{ strcpy(dummy+k,*(a+j)); k+=strlen(*(a+j)); }
  return(dummy);
}

DestroyString(string)
char *string;
{ dontfree(string); }
	
