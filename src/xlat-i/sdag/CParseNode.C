/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include "CParseNode.h"
#include "CParser.h"

void printc(TList<CParseNode*> *cons, int indent, char *sep)
{
  for(CParseNode *tmp=cons->begin(); !cons->end();) {
    tmp->print(indent);
    tmp = cons->next();
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

CParseNode::CParseNode(EToken t, CLexer *cLexer, CParser *cParser, int overlaps)
{
  CToken *tok;
  CToken *tok1;
  int numberOfPointers;
  int count;
  isOverlaped = 0;

  type = t; text = 0; constructs = new TList<CParseNode*>();
  con1 = con2 = con3 = con4 = 0;
  switch (t) {
    case SDAGENTRY:
      tok = cParser->lookForToken(IDENT);
      con1 = new CParseNode(IDENT, tok->text);
      tok = cParser->lookForToken(LP); delete tok;
      con2 = new CParseNode(PARAMLIST, cLexer, cParser, 0); 
      tok = cLexer->getNextToken();
      if(tok->type == LBRACE) {
        delete tok;
        constructs->append(new CParseNode(SLIST, cLexer, cParser, 0));
      } else
        constructs->append(new CParseNode(tok->type, cLexer, cParser, 0));
      break;
    case OVERLAP:
      isOverlaped = 1;
      tok = cLexer->getNextToken();
      if(tok->type == LBRACE) {
        delete tok;
        constructs->append(new CParseNode(OLIST, cLexer, cParser, 1 ));
      } else {
        constructs->append(new CParseNode(tok->type, cLexer, cParser, 1));
      }
      break;
    case WHEN:
      isOverlaped = overlaps; 
      con1 = new CParseNode(ELIST, cLexer, cParser, overlaps);
      tok = cLexer->getNextToken();
      if(tok->type == LBRACE) {
        delete tok;
        tok = cLexer->lookAhead();
        if (tok->type == RBRACE) {
           tok = cLexer->getNextToken(); 
           delete tok;
        }
        else
           constructs->append(new CParseNode(SLIST, cLexer, cParser, overlaps));
      } else {
        constructs->append(new CParseNode(tok->type, cLexer, cParser, overlaps));
      }
      break;
    case IF:
      isOverlaped = overlaps;
      tok = cParser->lookForToken(LP); delete tok;
      tok = cLexer->getMatchedCode("( ", LP, RP);
      con1 = new CParseNode(INT_EXPR, tok->text);
      tok = cLexer->getNextToken();
      if(tok->type == LBRACE) {
        delete tok;
        constructs->append(new CParseNode(SLIST, cLexer, cParser, overlaps));
      } else {
        constructs->append(new CParseNode(tok->type, cLexer, cParser, overlaps));
      }
      tok = cLexer->lookAhead();
      if (tok->type != ELSE) {
        break;
      }
      delete tok;
      tok = cLexer->getNextToken(); delete tok;
      con2 = new CParseNode(ELSE, cLexer, cParser, overlaps);
      break;
    case ELSE:
      isOverlaped = overlaps;
      tok = cLexer->getNextToken();
      if(tok->type == LBRACE) {
        delete tok;
        constructs->append(new CParseNode(SLIST, cLexer, cParser, overlaps));
      } else {
        constructs->append(new CParseNode(tok->type, cLexer, cParser, overlaps));
      }
      break;
    case FOR:
      isOverlaped = overlaps;
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
        constructs->append(new CParseNode(SLIST, cLexer, cParser, overlaps));
      } else {
        constructs->append(new CParseNode(tok->type, cLexer, cParser, overlaps));
      }
      break;
    case WHILE:
      isOverlaped = overlaps;
      tok = cParser->lookForToken(LP); delete tok;
      tok = cLexer->getMatchedCode("( ", LP, RP);
      con1 = new CParseNode(INT_EXPR, tok->text);
      tok = cLexer->getNextToken();
      if(tok->type == LBRACE) {
        delete tok;
        constructs->append(new CParseNode(SLIST, cLexer, cParser, overlaps));
      } else {
        constructs->append(new CParseNode(tok->type, cLexer, cParser, overlaps));
      }
      break;
    case ATOMIC:
      isOverlaped = overlaps;
      tok = cParser->lookForToken(LBRACE); delete tok;
      tok = cLexer->getMatchedCode("{ ", LBRACE, RBRACE);
      text = tok->text;
      break;
    case OLIST:
    case SLIST:
      isOverlaped = overlaps;
      tok = cLexer->getNextToken();
      while(tok->type != RBRACE) {
        if (tok->type == LBRACE) {
          delete tok;
          constructs->append(new CParseNode(SLIST, cLexer, cParser, overlaps));
        } else {
          constructs->append(new CParseNode(tok->type, cLexer, cParser, overlaps));
        }
        tok = cLexer->getNextToken();
      }
      break;
    case ELIST:
      isOverlaped = overlaps;
      tok = cLexer->lookAhead();
      while (tok->type == IDENT) {
        constructs->append(new CParseNode(ENTRY, cLexer, cParser, overlaps));
        tok = cLexer->lookAhead();
      }
      break;
    case ENTRY:
      isOverlaped = overlaps;
      tok = cParser->lookForToken(IDENT);
      con1 = new CParseNode(IDENT, tok->text);
      tok = cParser->lookForToken2(LB, LP);
      con2 = 0;
      if(tok->type == LB) {
        delete tok;
        tok = cLexer->getIntExpr(RB);
        con2 = new CParseNode(INT_EXPR, tok->text);
        tok = cParser->lookForToken(RB); delete tok;
        tok = cParser->lookForToken(LP); delete tok;
      }
      con3 = new CParseNode(PARAMLIST, cLexer, cParser, overlaps);
      isVoid = con3->isVoid;
      tok = cLexer->lookAhead();
      if(tok->type == COMMA) {
        delete tok;
        tok = cLexer->getNextToken();
      }
      break;
    case FORALL:
      isOverlaped = overlaps;
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
        constructs->append(new CParseNode(SLIST, cLexer, cParser, overlaps));
      } else {
        constructs->append(new CParseNode(tok->type, cLexer, cParser, overlaps));
      }
      break;
   case PARAMLIST:
      isOverlaped = overlaps;
      tok = cLexer->lookAhead();
      if (tok->type == RP)  {
            isVoid = 1;
      }
      else 
         isVoid = 0;
     
      while (tok->type != RP) {
        CParseNode *parameter1 = new CParseNode(PARAMETER, cLexer, cParser, overlaps);
	if (parameter1->isVoid == 1) {
	   isVoid = 1;
        }
      	constructs->append(parameter1);
        tok = cLexer->lookAhead();
        if (tok->type != RP) {
 	   tok = cParser->lookForToken(COMMA); delete tok;
	   tok = cLexer->lookAhead();
 	}
      }
      tok = cParser->lookForToken(RP); delete tok;
      break;
   case PARAMETER:
      isOverlaped = overlaps;
      tok = cLexer->lookAhead();
      con1 = new CParseNode(VARTYPE, cLexer, cParser, overlaps);  // con1 holds the type
      isVoid = con1->isVoid;
      con2 = 0; con3 = 0; con4 = 0;
      tok = cLexer->lookAhead();
      if (tok->type == IDENT) { // Case where it is "Type Name" 
         tok = cParser->lookForToken(IDENT);
         con2 = new CParseNode(IDENT, tok->text);  // con2 holds the Name
        tok = cLexer->lookAhead();
         if (tok->type == LB) {  // Case where it is "Type Name [ArrayLengthExpression]"
	    tok = cParser->lookForToken(LB);  delete tok;
            tok = cLexer->getNextToken();
	    int q = 0;
	    XStr *arrayExp = new XStr("");;
	    while (tok->type != RB) {
	      q++;
	      arrayExp->append(*(tok->text));
	      tok = cLexer->getNextToken();
	    }
	    if (q == 0)
	      printf("ERROR: Need to have the length of the array specified\n");
	    delete tok;
            con3 = new CParseNode(IDENT, arrayExp); // con3 holds the expression for the array length
         }
         else if (tok->type == EQUAL) {  // Case where it is "Type Name = DEFAULTPARAMETER"
	    tok = cLexer->getNextToken(); delete tok;
	    tok = cLexer->getNextToken();
	    if ((tok->type == LITERAL) || (tok->type == NUMBER))
	      con4 = new CParseNode(tok->type, tok->text); //con4 holds the DEFAULTPARAMETER 
            else {
              delete tok;
              count = 0;
              tok = cLexer->lookAhead();
	      while (((tok->type != RP) && (tok->type != COMMA)) || (count != 0)) {
                 if (tok->type == LP)
                    count++;
                 else if (tok->type == RP)
                    count--;
                 tok = cLexer->getNextToken(); delete tok;
	         tok = cLexer->lookAhead();
	      }
	    }
 	 }
      }
      break;
   case VARTYPE:
      isOverlaped = overlaps;
      isVoid = 0;
      tok = cLexer->getNextToken();
      con1 = 0; con2 = 0; con3 = 0; con4 = 0;
      if (tok->type == CONST) {
         con1 = new CParseNode(CONST, tok->text);
	 con2 = new CParseNode(VARTYPE, cLexer, cParser, overlaps);
      }
      else {
        if ((tok->type == INT) || (tok->type == LONG) || (tok->type == SHORT) || (tok->type == DOUBLE) 
	      || (tok->type == FLOAT) || (tok->type == UNSIGNED) || (tok->type == VOID)
	      || (tok->type == CHAR) || (tok->type == IDENT)) {
           tok1 = cLexer->lookAhead();
	   if ((tok->type == VOID) && (tok1->type == RP)) {
	      isVoid = 1;
              delete tok;
	   }
	   else {
	     if (tok->type == UNSIGNED) {
      	       tok1 = cLexer->getNextToken(); 
      	     }
             else if (tok->type == LONG) {
     	       tok1 = cLexer->lookAhead();
               if ((tok1->type == LONG) || (tok1->type == DOUBLE))
                 tok1 = cLexer->getNextToken();
	       else
	         tok1 = 0;
       	     }
      	     else {
	       tok1 = 0;
             }
	     CToken *temptok;
             temptok = cLexer->lookAhead();
             numberOfPointers = 0;
             while (temptok->type == STAR) {
 	        temptok = cParser->lookForToken(STAR); delete temptok;
	        numberOfPointers = numberOfPointers + 1;
	        temptok = cLexer->lookAhead();
             }
             if (numberOfPointers  == 0) {
	        con3 = new CParseNode(SIMPLETYPE, cLexer, cParser, tok, tok1, 0);
             }
	     else if (numberOfPointers >= 1) {
                con3 = new CParseNode(PTRTYPE, cLexer, cParser, tok, tok1, numberOfPointers);
	     } 
           }
	}
      }  
      tok = cLexer->lookAhead();
      if (tok->type == AMPERESIGN) {
	tok = cLexer->getNextToken();
        con4 = new CParseNode(AMPERESIGN, tok->text);
      }
      else if (tok->type == LP) {
         tok = cLexer->getNextToken(); delete tok;
         con4 = new CParseNode(FUNCTYPE, cLexer, cParser, overlaps);
      }
      break;
    case FUNCTYPE:
      isOverlaped =overlaps;
      tok = cParser->lookForToken(STAR); delete tok;     
      tok = cParser->lookForToken(IDENT);
      con1 = new CParseNode(IDENT, tok->text);
      tok = cParser->lookForToken(RP); delete tok;
      tok = cParser->lookForToken(LP); delete tok;
      con2 = new CParseNode(PARAMLIST, cLexer, cParser, overlaps);
      break;
    default:
      syntaxError(cLexer);
      break;
  }
}

CParseNode::CParseNode(EToken t, CLexer *cLexer, CParser *cParser, CToken *tokA, CToken *tokB, int pointers)
{
  CToken *tok;

  type = t; text = 0; constructs = new TList<CParseNode*>();
  con1 = con2 = con3 = con4 = 0;
  switch (t) {
    case SIMPLETYPE:
      if ((tokA->type == INT) || (tokA->type == LONG) || (tokA->type == CHAR) || (tokA->type == SHORT)
	|| (tokA->type == UNSIGNED) || (tokA->type == FLOAT) || (tokA->type == DOUBLE) || (tokA->type == VOID) || (tokA->type == IDENT))
	   con1 = new CParseNode(BUILTINTYPE, cLexer, cParser, tokA, tokB, pointers);
      else
          printf("ERROR: The parser doesn't handle this type \n");
      break;
    case BUILTINTYPE:
      con2 = 0;
      con1 = new CParseNode(tokA->type, tokA->text);
      if (tokB != 0)
         con2 = new CParseNode(tokB->type, tokB->text);
      break;
    case PTRTYPE:
      numPtrs = pointers;
      con1 = new CParseNode(SIMPLETYPE, cLexer, cParser, tokA, tokB, 0);
      break;
    default:
      syntaxError(cLexer);
      break;
    }
}


