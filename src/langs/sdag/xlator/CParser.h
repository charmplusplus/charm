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
#include "xi-util.h"

class CParser {
  private:
    CLexer *cLexer;

  public:
    CParser(XStr& input) {
      cLexer = new CLexer(input.charstar());
    }

    ~CParser(void) {;}

    CParsedFile *doParse(void);
    CToken *lookForToken(EToken t);
    CToken *lookForToken2(EToken t1, EToken t2);
    CToken *lookForStatement(void);
};

#endif
