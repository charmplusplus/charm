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
	
