/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include "CParseNode.h"
#include "CParser.h"

void printc(TList *cons, int indent, char *sep)
{
  for(CParseNode *tmp=(CParseNode *)(cons->begin()); !cons->end();) {
    tmp->print(indent);
    tmp = (CParseNode *)(cons->next());
    if(!cons->end())
      printf("%s", sep);
  }
}

void CParseNode::print(int indent)
{
  Indent(indent);
  switch(type) {
    case SDAGENTRY:
      printf("sdagentry "); con1->print(0); printf("(");
      con2->print(0); printf(" *"); con3->print(0);
      printf(")\n"); ::printc(constructs,indent, "\n"); printf("\n");
      break;
    case OVERLAP:
      printf("overlap\n"); printc(constructs,indent, "\n"); printf("\n");
      break;
    case WHEN:
      printf("when "); con1->print(0); printf("\n");
      printc(constructs,indent, "\n"); printf("\n");
      break;
    case FOR:
      printf("for ("); con1->print(0); 
      printf(";"); con2->print(0);
      printf(";"); con3->print(0);
      printf(")\n"); printc(constructs,indent, "\n");
      printf("\n");
      break;
    case WHILE:
      printf("while "); con1->print(0); printf("\n"); 
      printc(constructs,indent,"\n");
      printf("\n");
      break;
    case IF:
      printf("if "); con1->print(0); 
      printc(constructs,indent, "\n");
      if(con2!=0) {
        con2->print(indent);
        printf("\n");
      }
      break;
    case ELSE:
      printf("else ");
      printc(constructs,indent, "\n");
      break;
    case FORALL:
      printf("forall ["); con1->print(0); printf("]");
      printf(" ("); con2->print(0); printf(":"); con3->print(0); 
      printf(","); con4->print(0); printf(")\n");
      printc(constructs,indent, "\n");
      printf("\n");
      break;
    case ELIST:
      printc(constructs,0, ",");
      break;
    case SLIST:
    case OLIST:
      printf("{\n");
      printc(constructs,indent+1, "\n");
      printf("\n");
      Indent(indent);
      printf("}\n");
      break;
    case INT_EXPR:
      text->print(0);
      break;
    case IDENT:
      text->print(0);
      break;
    case ENTRY:
      con1->print(0);
      if(con2) { printf("["); con2->print(0); printf("]"); }
      printf("("); con3->print(0); printf(" *");
      con4->print(0); printf(")");
      break;
    case ATOMIC:
      printf("atomic\n");
      text->print(indent+1);
      break;
    default:
      printf("ERROR!!!\n");
      break;
  }
}

static void syntaxError(CLexer *cLexer)
{
  fprintf(stderr, "Syntax error near line %d, char %d\n",
                  cLexer->lineNum, cLexer->charNum);
  exit(1);
}

CParseNode::CParseNode(EToken t, CLexer *cLexer, CParser *cParser)
{
  CToken *tok;

  type = t; text = 0; constructs = new TList();
  con1 = con2 = con3 = con4 = 0;
  switch (t) {
    case SDAGENTRY:
      tok = cParser->lookForToken(IDENT);
      con1 = new CParseNode(IDENT, tok->text);
      tok = cParser->lookForToken(LP); delete tok;
      tok = cParser->lookForToken(IDENT);
      con2 = new CParseNode(IDENT, tok->text);
      tok = cParser->lookForToken(STAR); delete tok;
      tok = cParser->lookForToken(IDENT);
      con3 = new CParseNode(IDENT, tok->text);
      tok = cParser->lookForToken(RP); delete tok;
      tok = cLexer->getNextToken();
      if(tok->type == LBRACE) {
        delete tok;
        constructs->append(new CParseNode(SLIST, cLexer, cParser));
      } else
        constructs->append(new CParseNode(tok->type, cLexer, cParser));
      break;
    case OVERLAP:
      tok = cLexer->getNextToken();
      if(tok->type == LBRACE) {
        delete tok;
        constructs->append(new CParseNode(OLIST, cLexer, cParser));
      } else {
        constructs->append(new CParseNode(tok->type, cLexer, cParser));
      }
      break;
    case WHEN:
      con1 = new CParseNode(ELIST, cLexer, cParser);
      tok = cLexer->getNextToken();
      if(tok->type == LBRACE) {
        delete tok;
        constructs->append(new CParseNode(SLIST, cLexer, cParser));
      } else {
        constructs->append(new CParseNode(tok->type, cLexer, cParser));
      }
      break;
    case IF:
      tok = cParser->lookForToken(LP); delete tok;
      tok = cLexer->getParenCode();
      con1 = new CParseNode(INT_EXPR, tok->text);
      tok = cLexer->getNextToken();
      if(tok->type == LBRACE) {
        delete tok;
        constructs->append(new CParseNode(SLIST, cLexer, cParser));
      } else {
        constructs->append(new CParseNode(tok->type, cLexer, cParser));
      }
      tok = cLexer->lookAhead();
      if (tok->type != ELSE) {
        break;
      }
      delete tok;
      tok = cLexer->getNextToken(); delete tok;
      con2 = new CParseNode(ELSE, cLexer, cParser);
      break;
    case ELSE:
      tok = cLexer->getNextToken();
      if(tok->type == LBRACE) {
        delete tok;
        constructs->append(new CParseNode(SLIST, cLexer, cParser));
      } else {
        constructs->append(new CParseNode(tok->type, cLexer, cParser));
      }
      break;
    case FOR:
      tok = cParser->lookForToken(LP); delete tok;
      tok = cLexer->getIntExpr(SEMICOLON);
      con1 = new CParseNode(INT_EXPR, tok->text);
      tok = cParser->lookForToken(SEMICOLON); delete tok;
      tok = cLexer->getIntExpr(SEMICOLON);
      con2 = new CParseNode(INT_EXPR, tok->text);
      tok = cParser->lookForToken(SEMICOLON); delete tok;
      tok = cLexer->getIntExpr(RP);
      con3 = new CParseNode(INT_EXPR, tok->text);
      tok = cParser->lookForToken(RP); delete tok;
      tok = cLexer->getNextToken();
      if(tok->type == LBRACE) {
        delete tok;
        constructs->append(new CParseNode(SLIST, cLexer, cParser));
      } else {
        constructs->append(new CParseNode(tok->type, cLexer, cParser));
      }
      break;
    case WHILE:
      tok = cParser->lookForToken(LP); delete tok;
      tok = cLexer->getParenCode();
      con1 = new CParseNode(INT_EXPR, tok->text);
      tok = cLexer->getNextToken();
      if(tok->type == LBRACE) {
        delete tok;
        constructs->append(new CParseNode(SLIST, cLexer, cParser));
      } else {
        constructs->append(new CParseNode(tok->type, cLexer, cParser));
      }
      break;
    case ATOMIC:
      tok = cParser->lookForToken(LBRACE); delete tok;
      tok = cLexer->getBracedCode();
      text = tok->text;
      break;
    case OLIST:
    case SLIST:
      tok = cLexer->getNextToken();
      while(tok->type != RBRACE) {
        if (tok->type == LBRACE) {
          delete tok;
          constructs->append(new CParseNode(SLIST, cLexer, cParser));
        } else {
          constructs->append(new CParseNode(tok->type, cLexer, cParser));
        }
        tok = cLexer->getNextToken();
      }
      break;
    case ELIST:
      tok = cLexer->lookAhead();
      while (tok->type == IDENT) {
        constructs->append(new CParseNode(ENTRY, cLexer, cParser));
        tok = cLexer->lookAhead();
      }
      break;
    case ENTRY:
      tok = cParser->lookForToken(IDENT);
      con1 = new CParseNode(IDENT, tok->text);
      tok = cParser->lookForToken2(LB, LP);
      if(tok->type == LB) {
        delete tok;
        tok = cLexer->getIntExpr(RB);
        con2 = new CParseNode(INT_EXPR, tok->text);
        tok = cParser->lookForToken(RB); delete tok;
        tok = cParser->lookForToken(LP); delete tok;
      }
      tok = cParser->lookForToken(IDENT);
      con3 = new CParseNode(IDENT, tok->text);
      tok = cParser->lookForToken(STAR); delete tok;
      tok = cParser->lookForToken(IDENT);
      con4 = new CParseNode(IDENT, tok->text);
      tok = cParser->lookForToken(RP); delete tok;
      tok = cLexer->lookAhead();
      if(tok->type == COMMA) {
        delete tok;
        tok = cLexer->getNextToken();
      }
      break;
    case FORALL:
      tok = cParser->lookForToken(LB); delete tok;
      tok = cParser->lookForToken(IDENT);
      con1 = new CParseNode(IDENT, tok->text);
      tok = cParser->lookForToken(RB); delete tok;
      tok = cParser->lookForToken(LP); delete tok;
      tok = cLexer->getIntExpr(COLON);
      con2 = new CParseNode(INT_EXPR, tok->text);
      tok = cParser->lookForToken(COLON); delete tok;
      tok = cLexer->getIntExpr(COMMA);
      con3 = new CParseNode(INT_EXPR, tok->text);
      tok = cParser->lookForToken(COMMA); delete tok;
      tok = cLexer->getIntExpr(RP);
      con4 = new CParseNode(INT_EXPR, tok->text);
      tok = cParser->lookForToken(RP); delete tok;
      tok = cLexer->getNextToken();
      if(tok->type == LBRACE) {
        delete tok;
        constructs->append(new CParseNode(SLIST, cLexer, cParser));
      } else {
        constructs->append(new CParseNode(tok->type, cLexer, cParser));
      }
      break;
    default:
      syntaxError(cLexer);
      break;
  }
}

