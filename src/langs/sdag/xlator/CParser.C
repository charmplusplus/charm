#include "CParser.h"

CToken *CParser::lookForToken(EToken t)
{
  CToken *tok;
  while((tok=cLexer->getNextToken()) !=0) {
    if(tok->type == NEW_LINE) {
      delete tok;
      continue;
    }
    if(tok->type == t) {
      return tok;
    } else {
      fprintf(stderr, "Syntax Error near line %d, char %d\n",
                      cLexer->lineNum, cLexer->charNum);
      exit(1);
    }
  }
  return (CToken *) 0; // End of File
}

CToken *CParser::lookForToken2(EToken t1, EToken t2)
{
  CToken *tok;
  while((tok=cLexer->getNextToken()) !=0) {
    if(tok->type == NEW_LINE) {
      delete tok;
      continue;
    }
    if(tok->type == t1 || tok->type == t2) {
      return tok;
    } else
      break;
  }
  fprintf(stderr, "Syntax Error near line %d, char %d\n",
                  cLexer->lineNum, cLexer->charNum);
  exit(1);
  return (CToken *) 0; // Just to satisfy the compiler
}

CToken *CParser::lookForStatement(void)
{
  CToken *tok;
  while((tok=cLexer->getNextToken()) !=0) {
    if(tok->type == NEW_LINE) {
      delete tok;
      continue;
    }
    if(tok->type == OVERLAP ||
       tok->type == WHEN ||
       tok->type == FOR ||
       tok->type == IF ||
       tok->type == WHILE ||
       tok->type == ATOMIC ||
       tok->type == FORALL) {
      return tok;
    } else
      break;
  }
  fprintf(stderr, "Syntax Error near line %d, char %d\n",
                  cLexer->lineNum, cLexer->charNum);
  exit(1);
  return (CToken *) 0; // Just to satisfy the compiler
}

CParsedFile *CParser::doParse(void)
{
  CParsedFile *cParsedFile = new CParsedFile(sourceFile);
  CToken *tok;

  tok = lookForToken(CLASS); delete tok;
  tok = lookForToken(IDENT);
  cParsedFile->className = tok->text;
  while(1) {
    tok = lookForToken(SDAGENTRY);
    if(tok == 0)
      break;
    cParsedFile->nodeList->append(new CParseNode(SDAGENTRY, cLexer, this));
  }
  return cParsedFile;
}

