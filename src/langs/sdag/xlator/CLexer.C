#include "CLexer.h"
#include <stdlib.h>

CLexer::CLexer(void)
{
  lineNum = 1;
  charNum = 1;
  wsSignificant = 0;
}

CLexer::~CLexer(void)
{
}

int CLexer::sourceFile(char *filename)
{
  FILE *fp;
  if((fp=fopen(filename, "r"))==(FILE *)0)
    return 0;
  fclose(fp);
  freopen(filename, "r", stdin);
  return 1;
}

CToken *CLexer::lookAhead(void)
{
  while(1) {
    CToken *cToken = getNextToken();
    if(cToken==0)
      return cToken;
    if(cToken->type == NEW_LINE) {
      delete cToken;
      continue;
    } else {
      char *yycopy = strdup(yytext);
      for(int i=strlen(yycopy)-1; i>=0; i-- ) {
        charNum--;
        Unput(yycopy[i]) ;
      }
      free(yycopy);
      return cToken;
    }
  }
  return (CToken *) 0;
}

CToken *CLexer::getNextToken(void)
{
  EToken type;
  CToken *cToken;

  while(1) {
    type = yylex();
    if ((int)type == 0)
      return (CToken *) 0;
    charNum += strlen(yytext);
    if(type == NEW_LINE) {
      lineNum++;
      charNum = 1;
      if (wsSignificant)
        return new CToken(type, yytext);
      else
        continue;
    }
    if((type != WSPACE) || wsSignificant) {
      cToken = new CToken(type, yytext);
      // cToken->print(0);
      return cToken;
    }
  }
}

CToken *CLexer::getBracedCode(void)
{
  CToken *code = new CToken(BRACE_MATCHED_CPP_CODE, "{ ");
  int currentScope = 1;
  wsSignificant = 1;
  // Code to eat C++ code
  while(currentScope != 0) {
    CToken *cToken = getNextToken();
    if(cToken==0)
      return cToken;
    if(cToken->type == LBRACE) {
      currentScope++;
    }
    if(cToken->type == RBRACE) {
      currentScope--;
    }
    code->text->append(cToken->text);
    delete cToken;
  }
  wsSignificant = 0;
  return code;
}

CToken *CLexer::getParenCode(void)
{
  CToken *code = new CToken(BRACE_MATCHED_CPP_CODE, "( ");
  int currentScope = 1;
  wsSignificant = 1;
  // Code to eat C++ code
  while(currentScope != 0) {
    CToken *cToken = getNextToken();
    if(cToken==0)
      return cToken;
    if(cToken->type == LP) {
      currentScope++;
    }
    if(cToken->type == RP) {
      currentScope--;
    }
    code->text->append(cToken->text);
    delete cToken;
  }
  wsSignificant = 0;
  return code;
}

CToken *CLexer::getIntExpr(EToken term)
{
  CToken *expr = new CToken(INT_EXPR, "");
  unsigned int endExpr = 0;
  unsigned int numBraces=(term==RBRACE)?1:0;
  unsigned int numParens=(term==RP)?1:0;
  unsigned int numBrackets=(term==RB)?1:0;
  wsSignificant = 1;
  while(!endExpr) {
    CToken *cToken = getNextToken();
    if(cToken==0)
      return cToken;
    switch(cToken->type) {
      case LP:
        numParens++;
        break;
      case RP:
        numParens--;
        break;
      case LB:
        numBrackets++;
        break;
      case RB:
        numBrackets--;
        break;
      case LBRACE:
        numBraces++;
        break;
      case RBRACE:
        numBraces--;
        break;
    }
    if(cToken->type == term && !numBraces && !numParens && !numBrackets) {
      for(int i=strlen(yytext)-1; i>=0; i-- ) {
        charNum--;
        Unput(yytext[i]) ;
      }
      endExpr = 1;
    } else {
      expr->text->append(cToken->text);
    }
    delete cToken;
  }
  wsSignificant = 0;
  return expr;
}
