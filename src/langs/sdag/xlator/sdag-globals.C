/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include <stdarg.h>
#include <stdio.h>
#include "sdag-globals.h"

void Indent(int indent)
{
  for(int i=0;i<indent;i++)
    printf("  ");
}

int numSdagEntries=0;
int numSlists=0;
int numOverlaps=0;
int numWhens=0;
int numFors=0;
int numIfs=0;
int numElses=0;
int numEntries=0;
int numOlists=0;
int numWhiles=0;
int numAtomics=0;
int numForalls=0;

void resetNumbers(void)
{
  numSdagEntries=0;
  numSlists=0;
  numOverlaps=0;
  numWhens=0;
  numFors=0;
  numIfs=0;
  numElses=0;
  numEntries=0;
  numOlists=0;
  numWhiles=0;
  numAtomics=0;
  numForalls=0;
}

