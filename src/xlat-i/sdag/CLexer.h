/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef _CLexer_H_
#define _CLexer_H_

#include <stdio.h>
#include "CToken.h"
#include "sdag-globals.h"
#include "string.h"

extern "C" EToken sllex();
extern "C" char *mysltext;
extern "C" void Unput(char);

class CLexer {
  private:
    unsigned int wsSignificant;
  public:
    unsigned int lineNum;
    unsigned int charNum;
    CLexer(char *);
    ~CLexer(void);
    CToken *lookAhead(void);
    CToken *getNextToken(void);
    CToken *getMatchedCode(const char *, EToken, EToken);
    CToken *getIntExpr(EToken term);
};

#endif /* _CLexer_H_ */
