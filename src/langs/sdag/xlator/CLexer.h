#ifndef _CLexer_H_
#define _CLexer_H_

#include <stdio.h>
#include "CToken.h"
#include "sdag-globals.h"
#include "string.h"

extern "C" EToken yylex();
extern char *myyytext;
extern "C" void Unput(char);

class CLexer {
  private:
    unsigned int wsSignificant;
  public:
    unsigned int lineNum;
    unsigned int charNum;
    CLexer(void);
    ~CLexer(void);
    int sourceFile(char *filename);
    CToken *lookAhead(void);
    CToken *getNextToken(void);
    CToken *getBracedCode(void);
    CToken *getParenCode(void);
    CToken *getIntExpr(EToken term);
};

#endif /* _CLexer_H_ */
