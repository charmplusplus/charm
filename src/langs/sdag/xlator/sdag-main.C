/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "CLexer.h"
#include "CToken.h"
#include "CParsedFile.h"
#include "CParser.h"
#include "sdag-globals.h"
#include "xi-util.h"

void Usage(void)
{
  fprintf(stderr, "Usage: sdagx <filename>\n");
}

int main(int argc, char *argv[])
{
  if(argc<2) {
    Usage();
    exit(1);
  }
  for(int i=1; i<argc; i++) {
    resetNumbers();
    XStr input, output;
    // printf("%s:\n", argv[i]);
    FILE *fp = fopen(argv[i], "r");
    if(fp == 0) {
      fprintf(stderr, "sdagx: cannot open %s for reading !!\n", argv[i]);
      exit(1);
    }
    char str[1024], *s;
    while(0 != (s=fgets(str,1024,fp)))
      input << str;
    fclose(fp);
    CParser *cParser = new CParser(input);
    CParsedFile *parsedFile = cParser->doParse();
    parsedFile->doProcess(output);
    sprintf(str, "%s.h", argv[i]);
    fp = fopen(str, "w");
    if(fp ==0) {
      fprintf(stderr, "sdagx: cannot open %s for writing !!\n", str);
      exit(1);
    }
    fprintf(fp, "%s", output.charstar());
    fclose(fp);
    // parsedFile->print(0);
    delete parsedFile;
    delete cParser;
  }
  exit(0);
}
