#include <stdio.h>
#include <stdlib.h>
#include "CLexer.h"
#include "CToken.h"
#include "CParsedFile.h"
#include "CParser.h"
#include "sdag-globals.h"

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
    // printf("%s:\n", argv[i]);
    CParser *cParser = new CParser(argv[i]);
    CParsedFile *parsedFile = cParser->doParse();
    parsedFile->doProcess();
    // parsedFile->print(0);
    delete parsedFile;
    delete cParser;
  }
  exit(0);
}
