/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef _CParser_H_
#define _CParser_H_

#include <stdio.h>
#include <stdlib.h>
#include "CLexer.h"
#include "CParsedFile.h"
#include "CToken.h"
#include "sdag-globals.h"

class CParser {
  private:
    CLexer *cLexer;
    char *sourceFile;

  public:
    CParser(char *filename) {
      sourceFile = filename;
      cLexer = new CLexer();
      if(cLexer->sourceFile(filename) == 0) {
        fprintf(stderr, "sdagx: Cannot open file %s for reading !\n", filename);
        exit(1);
      }
    }

    ~CParser(void) {;}

    CParsedFile *doParse(void);
    CToken *lookForToken(EToken t);
    CToken *lookForToken2(EToken t1, EToken t2);
    CToken *lookForStatement(void);
};

#endif
