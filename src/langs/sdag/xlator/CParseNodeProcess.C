/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include <stdlib.h>
#include "sdag-globals.h"
#include "CParseNode.h"

void CParseNode::numberNodes(void)
{
  switch(type) {
    case SDAGENTRY: nodeNum = numSdagEntries++; break;
    case OVERLAP: nodeNum = numOverlaps++; break;
    case WHEN: nodeNum = numWhens++; break;
    case FOR: nodeNum = numFors++; break;
    case WHILE: nodeNum = numWhiles++; break;
    case IF: nodeNum = numIfs++; if(con2!=0) con2->numberNodes(); break;
    case ELSE: nodeNum = numElses++; break;
    case FORALL: nodeNum = numForalls++; break;
    case SLIST: nodeNum = numSlists++; break;
    case OLIST: nodeNum = numOlists++; break;
    case ATOMIC: nodeNum = numAtomics++; break;
    case ELIST:
    case INT_EXPR:
    case IDENT:
    case ENTRY:
    default:
      break;
  }
  CParseNode *cn;
  for(cn=(CParseNode *)(constructs->begin()); !constructs->end(); cn=(CParseNode *)(constructs->next())) {
    cn->numberNodes();
  }
}

void CParseNode::labelNodes(XStr *cname)
{
  char text[128];

  className = cname;
  switch(type) {
    case ENTRY:
    case SDAGENTRY:
      sprintf(text, "%s", con1->text->charstar());
      label = new XStr(text);
      break;
    case OVERLAP: 
      sprintf(text, "_overlap_%d", nodeNum); 
      label = new XStr(text);
      break;
    case WHEN: 
      sprintf(text, "_when_%d", nodeNum); 
      label = new XStr(text);
      break;
    case FOR: 
      sprintf(text, "_for_%d", nodeNum); 
      label = new XStr(text);
      break;
    case WHILE: 
      sprintf(text, "_while_%d", nodeNum); 
      label = new XStr(text);
      break;
    case IF: 
      sprintf(text, "_if_%d", nodeNum); 
      label = new XStr(text);
      if(con2!=0) con2->labelNodes(cname);
      break;
    case ELSE: 
      sprintf(text, "_else_%d", nodeNum); 
      label = new XStr(text);
      break;
    case FORALL: 
      sprintf(text, "_forall_%d", nodeNum); 
      label = new XStr(text);
      break;
    case SLIST: 
      sprintf(text, "_slist_%d", nodeNum); 
      label = new XStr(text);
      break;
    case OLIST: 
      sprintf(text, "_olist_%d", nodeNum); 
      label = new XStr(text);
      break;
    case ATOMIC: 
      sprintf(text, "_atomic_%d", nodeNum); 
      label = new XStr(text);
      break;
    case ELIST:
    case INT_EXPR:
    case IDENT:
    default:
      break;
  }
  CParseNode *cn;
  for(cn=(CParseNode *)(constructs->begin()); !constructs->end(); cn=(CParseNode *)(constructs->next())) {
    cn->labelNodes(cname);
  }
}

void CParseNode::generateEntryList(TList *list, CParseNode *thisWhen)
{
  switch(type) {
    case WHEN:
      con1->generateEntryList(list, this);
      break;
    case IF:
      if(con2!=0) con2->generateEntryList(list,thisWhen);
      break;
    case ENTRY:
      CEntry *entry;
      int found=0;
      for(entry=(CEntry *)(list->begin()); !list->end(); 
          entry=(CEntry *)(list->next())) {
        if(*(entry->entry) == *(con1->text) &&
           *(entry->msgType) == *(con3->text)) {
           found = 1;
           // check to see if thisWhen is already in entry's whenList
           int whenFound = 0;
           TList *tmpList = entry->whenList;
           CParseNode *tmpNode;
           for(tmpNode = (CParseNode *) tmpList->begin(); !tmpList->end();
               tmpNode = (CParseNode *) tmpList->next()) {
             if(tmpNode->nodeNum == thisWhen->nodeNum)
               whenFound = 1;
           }
           if(!whenFound)
             entry->whenList->append(thisWhen);
           entryPtr = entry;
	   if(con2)
	     entry->refNumNeeded = 1;
           break;
        }
      }
      if(!found) {
        CEntry *newEntry;
        newEntry = new CEntry(new XStr(*(con1->text)), new XStr(*(con3->text)));
        list->append(newEntry);
        entryPtr = newEntry;
        newEntry->whenList->append(thisWhen);
	if(con2)
	  newEntry->refNumNeeded = 1;
      }
      break;
  }
  CParseNode *cn;
  for(cn=(CParseNode *)(constructs->begin()); !constructs->end(); cn=(CParseNode *)(constructs->next())) {
    cn->generateEntryList(list,thisWhen);
  }
}

void CParseNode::propogateState(TList *list)
{
  CStateVar *sv;
  stateVars = new TList();
  switch(type) {
    case SDAGENTRY:
      {
        XStr *vType = new XStr(*(con2->text));
        vType->append(" *");
        sv = new CStateVar(vType, new XStr(*(con3->text)));
      }
      stateVars->append(sv);
      stateVarsChildren = stateVars;
      break;
    case FORALL:
      stateVarsChildren = new TList();
      for(sv=(CStateVar *)(list->begin()); !list->end(); sv=(CStateVar *)(list->next())) {
        stateVars->append(sv);
        stateVarsChildren->append(sv);
      }
      sv = new CStateVar(new XStr("int"), new XStr(*(con1->text)));
      stateVarsChildren->append(sv);
      {
        char txt[128];
        sprintf(txt, "_cf%d", nodeNum);
        counter = new XStr(txt);
        sv = new CStateVar(new XStr("CCounter *"), counter);
        stateVarsChildren->append(sv);
      }
      break;
    case WHEN:
      stateVarsChildren = new TList();
      for(sv=(CStateVar *)(list->begin()); !list->end(); sv=(CStateVar *)(list->next())) {
        stateVars->append(sv);
        stateVarsChildren->append(sv);
      }
      {
        TList *elist = con1->constructs;
        CParseNode *en;
        for(en=(CParseNode *)(elist->begin()); !elist->end(); en=(CParseNode *)(elist->next())) {
          XStr *vType = new XStr(*(en->con3->text));
          vType->append(" *");
          sv = new CStateVar(vType, new XStr(*(en->con4->text)));
          stateVarsChildren->append(sv);
        }
      }
      break;
    case IF:
      for(sv=(CStateVar *)(list->begin()); !list->end(); sv=(CStateVar *)(list->next())) {
        stateVars->append(sv);
      }
      stateVarsChildren = stateVars;
      if(con2 != 0) con2->propogateState(list);
      break;
    case OLIST:
      stateVarsChildren = new TList();
      for(sv=(CStateVar *)(list->begin()); !list->end(); sv=(CStateVar *)(list->next())) {
        stateVars->append(sv);
        stateVarsChildren->append(sv);
      }
      {
        char txt[128];
        sprintf(txt, "_co%d", nodeNum);
        counter = new XStr(txt);
        sv = new CStateVar(new XStr("CCounter *"), counter);
        stateVarsChildren->append(sv);
      }
      break;
    case FOR:
    case WHILE:
    case ELSE:
    case SLIST:
    case OVERLAP:
    case ATOMIC:
      for(sv=(CStateVar *)(list->begin()); !list->end(); sv=(CStateVar *)(list->next())) {
        stateVars->append(sv);
      }
      stateVarsChildren = stateVars;
      break;
    case INT_EXPR:
    case IDENT:
    case ENTRY:
    case ELIST:
    default:
      break;
  }
  CParseNode *cn;
  for(cn=(CParseNode *)(constructs->begin()); !constructs->end(); cn=(CParseNode *)(constructs->next())) {
    cn->propogateState(stateVarsChildren);
  }
}

void CParseNode::generateCode(XStr& op)
{
  switch(type) {
    case SDAGENTRY:
      generateSdagEntry(op);
      break;
    case SLIST:
      generateSlist(op);
      break;
    case OLIST:
      generateOlist(op);
      break;
    case FORALL:
      generateForall(op);
      break;
    case ATOMIC:
      generateAtomic(op);
      break;
    case IF:
      generateIf(op);
      if(con2 != 0)
        con2->generateCode(op);
      break;
    case ELSE:
      generateElse(op);
      break;
    case WHILE:
      generateWhile(op);
      break;
    case FOR:
      generateFor(op);
      break;
    case OVERLAP:
      generateOverlap(op);
      break;
    case WHEN:
      generateWhen(op);
      break;
    default:
      break;
  }
  CParseNode *cn;
  for(cn=(CParseNode *)(constructs->begin()); !constructs->end(); cn=(CParseNode *)(constructs->next())) {
    cn->generateCode(op);
  }
}

void CParseNode::generateWhen(XStr& op)
{
  op << "  int " << label->charstar() << "(";
  generatePrototype(op, stateVars);
  op << ") {\n";
  TList *elist = con1->constructs;
  CParseNode *el;
  for(el=(CParseNode *)(elist->begin()); !elist->end(); el=(CParseNode *)(elist->next())) {
    op << "    CMsgBuffer *"<<el->con4->text->charstar()<<"_buf;\n";
    op << "    " << el->con3->text->charstar() << " *" <<
                    el->con4->text->charstar() << ";\n";
  }
  op << "\n";
  for(el=(CParseNode *)(elist->begin()); !elist->end(); el=(CParseNode *)(elist->next())) {
    if(el->con2 == 0)
      op << "    " << el->con4->text->charstar() << 
            "_buf = __cDep->getMessage(" << el->entryPtr->entryNum << ");\n";
    else
      op << "    " << el->con4->text->charstar() << 
            "_buf = __cDep->getMessage(" << el->entryPtr->entryNum <<
            ", " << el->con2->text->charstar() << ");\n";
  }
  op << "\n";
  op << "    if (";
  for(el=(CParseNode *)(elist->begin()); !elist->end();) {
    op << "(" << el->con4->text->charstar() << "_buf != 0)";
    el = (CParseNode *)(elist->next());
    if(el != 0)
      op << "&&";
  }
  op << ") {\n";
  for(el=(CParseNode *)(elist->begin()); !elist->end(); el=(CParseNode *)(elist->next())) {
    op << "      " << el->con4->text->charstar() << " = (" << 
          el->con3->text->charstar() << " *) " <<
          el->con4->text->charstar() << "_buf->msg;\n";
    op << "      __cDep->removeMessage(" << el->con4->text->charstar() <<
          "_buf);\n";
  }
  op << "      " << ((CParseNode *)constructs->front())->label->charstar() << 
        "(";
  generateCall(op, stateVarsChildren);
  op << ");\n";
  op << "      return 1;\n";
  op << "    } else {\n";

  int nRefs=0, nAny=0;
  for(el=(CParseNode *)(elist->begin()); !elist->end(); el=(CParseNode *)(elist->next())) {
    if(el->con2 == 0)
      nAny++;
    else
      nRefs++;
  }

// keep these consts consistent with sdag.h in runtime

#define MAXARG 8
#define MAXANY 8
#define MAXREF 8

  if(stateVars->length() > MAXARG) {
    fprintf(stderr, "numStateVars more that %d, contact developers.\n",
		     MAXARG);
    exit(1);
  }
  if(nRefs > MAXREF) {
    fprintf(stderr, "numDepends more that %d, contact developers.\n",
		     MAXREF);
    exit(1);
  }
  if(nAny > MAXANY) {
    fprintf(stderr, "numDepends more that %d, contact developers.\n",
		     MAXANY);
    exit(1);
  }

  op << "      CWhenTrigger *tr;\n";
  op << "      tr = new CWhenTrigger(" << nodeNum << ", " <<
        stateVars->length() << ", " << nRefs << ", " << nAny << ");\n";
  CStateVar *sv;
  int iArgs=0;
  for(sv=(CStateVar *)(stateVars->begin());!stateVars->end();sv=(CStateVar *)(stateVars->next())) {
    op << "      tr->args[" << iArgs++ << "] = (size_t) " <<
          sv->name->charstar() << ";\n";
  }
  int iRef=0, iAny=0;
  for(el=(CParseNode *)(elist->begin()); !elist->end(); el=(CParseNode *)(elist->next())) {
    if(el->con2 == 0) {
      op << "      tr->anyEntries[" << iAny++ << "] = " <<
            el->entryPtr->entryNum << ";\n";
    } else {
      op << "      tr->entries[" << iRef << "] = " << 
            el->entryPtr->entryNum << ";\n";
      op << "      tr->refnums[" << iRef++ << "] = " <<
            el->con2->text->charstar() << ";\n";
    }
  }
  op << "      __cDep->Register(tr);\n";
  op << "      return 0;\n";
  op << "    }\n";
  // end actual code
  op << "  }\n\n";
  // end function
  op << "  void " << label->charstar() << "_end(";
  generatePrototype(op, stateVarsChildren);
  op << ") {\n";
  // actual code here 
  for(el=(CParseNode *)(elist->begin()); !elist->end(); el=(CParseNode *)(elist->next())) {
    // op << "    delete " <<  el->con4->text->charstar() << ";\n";
  }
  if(nextBeginOrEnd == 1)
   op << "    " << next->label->charstar() << "(";
  else
   op << "    " << next->label->charstar() << "_end(";
  generateCall(op, stateVars);
  op << ");\n";
  // end actual code
  op << "  }\n\n";
}

void CParseNode::generateWhile(XStr& op)
{
  // inlined start function
  op << "  void " << label->charstar() << "(";
  generatePrototype(op, stateVars);
  op << ") {\n";
  // actual code here 
  op << "    if (" << con1->text->charstar() << ") {\n";
  op << "      " << ((CParseNode *)constructs->front())->label->charstar() << 
        "(";
  generateCall(op, stateVarsChildren);
  op << ");\n";
  op << "    } else {\n";
  if(nextBeginOrEnd == 1)
   op << "      " << next->label->charstar() << "(";
  else
   op << "      " << next->label->charstar() << "_end(";
  generateCall(op, stateVars);
  op << ");\n";
  op << "    }\n";
  // end actual code
  op << "  }\n\n";
  // inlined end function
  op << "  void " << label->charstar() << "_end(";
  generatePrototype(op, stateVarsChildren);
  op << ") {\n";
  // actual code here 
  op << "    if (" << con1->text->charstar() << ") {\n";
  op << "      " << ((CParseNode *)constructs->front())->label->charstar() <<
        "(";
  generateCall(op, stateVarsChildren);
  op << ");\n";
  op << "    } else {\n";
  if(nextBeginOrEnd == 1)
   op << "      " <<  next->label->charstar() << "(";
  else
   op << "      " << next->label->charstar() << "_end(";
  generateCall(op, stateVars);
  op << ");\n";
  op << "    }\n";
  // end actual code
  op << "  }\n\n";
}

void CParseNode::generateFor(XStr& op)
{
  // inlined start function
  op << "  void " << label->charstar() << "(";
  generatePrototype(op, stateVars);
  op << ") {\n";
  // actual code here 
  op << "    " << con1->text->charstar() << ";\n";
  op << "    if (" << con2->text->charstar() << ") {\n";
  op << "      " << ((CParseNode *)constructs->front())->label->charstar() <<
        "(";
  generateCall(op, stateVarsChildren);
  op << ");\n";
  op << "    } else {\n";
  if(nextBeginOrEnd == 1)
   op << "      " << next->label->charstar() << "(";
  else
   op << "      " << next->label->charstar() << "_end(";
  generateCall(op, stateVars);
  op << ");\n";
  op << "    }\n";
  // end actual code
  op << "  }\n";
  // inlined end function
  op << "  void " << label->charstar() << "_end(";
  generatePrototype(op, stateVarsChildren);
  op << ") {\n";
  // actual code here 
  op << con3->text->charstar() << ";\n";
  op << "    if (" << con2->text->charstar() << ") {\n";
  op << "      " << ((CParseNode *)constructs->front())->label->charstar() <<
        "(";
  generateCall(op, stateVarsChildren);
  op << ");\n";
  op << "    } else {\n";
  if(nextBeginOrEnd == 1)
   op << "      " << next->label->charstar() << "(";
  else
   op << "      " << next->label->charstar() << "_end(";
  generateCall(op, stateVars);
  op << ");\n";
  op << "    }\n";
  // end actual code
  op << "  }\n";
}

void CParseNode::generateIf(XStr& op)
{
  // inlined start function
  op << "  void " << label->charstar() << "(";
  generatePrototype(op, stateVars);
  op << ") {\n";
  // actual code here 
  op << "    if (" << con1->text->charstar() << ") {\n";
  op << "      " << ((CParseNode *)constructs->front())->label->charstar() <<
        "(";
  generateCall(op, stateVarsChildren);
  op << ");\n";
  op << "    } else {\n";
  if (con2 != 0) {
    op << "      " << con2->label->charstar() << "(";
    generateCall(op, stateVarsChildren);
    op << ");\n";
  } else {
    op << "      " << label->charstar() << "_end(";
    generateCall(op, stateVarsChildren);
    op << ");\n";
  }
  op << "    }\n";
  // end actual code
  op << "  }\n\n";
  // inlined end function
  op << "  void " << label->charstar() << "_end(";
  generatePrototype(op, stateVarsChildren);
  op << ") {\n";
  // actual code here 
  if(nextBeginOrEnd == 1)
   op << "      " << next->label->charstar() << "(";
  else
   op << "      " << next->label->charstar() << "_end(";
  generateCall(op, stateVars);
  op << ");\n";
  // end actual code
  op << "  }\n\n";
}

void CParseNode::generateElse(XStr& op)
{
  // inlined start function
  op << "  void " << label->charstar() << "(";
  generatePrototype(op, stateVars);
  op << ") {\n";
  // actual code here 
  op << "    " << ((CParseNode *)constructs->front())->label->charstar() << 
        "(";
  generateCall(op, stateVarsChildren);
  op << ");\n";
  // end actual code
  op << "  }\n\n";
  // inlined end function
  op << "  void " << label->charstar() << "_end(";
  generatePrototype(op, stateVarsChildren);
  op << ") {\n";
  // actual code here 
  if(nextBeginOrEnd == 1)
   op << "      " << next->label->charstar() << "(";
  else
   op << "      " << next->label->charstar() << "_end(";
  generateCall(op, stateVars);
  op << ");\n";
  // end actual code
  op << "  }\n\n";
}

void CParseNode::generateForall(XStr& op)
{
  // inlined start function
  op << "  void " << label->charstar() << "(";
  generatePrototype(op, stateVars);
  op << ") {\n";
  // actual code here 
  op << "    int __first = (" << con2->text->charstar() <<
        "), __last = (" << con3->text->charstar() << 
        "), __stride = (" << con4->text->charstar() << ");\n";
  op << "    if (__first > __last) {\n";
  op << "      int __tmp=__first; __first=__last; __last=__tmp;\n";
  op << "      __stride = -__stride;\n";
  op << "    }\n";
  op << "    CCounter *" << counter->charstar() << 
        " = new CCounter(__first,__last,__stride);\n"; 
  op << "    for(int " << con1->text->charstar() << 
        "=__first;" << con1->text->charstar() <<
        "<=__last;" << con1->text->charstar() << "+=__stride) {\n";
  op << "      " << ((CParseNode *)constructs->front())->label->charstar() <<
        "(";
  generateCall(op, stateVarsChildren);
  op << ");\n";
  op << "    }\n";
  // end actual code
  op << "  }\n\n";
  // inlined end function
  op << "  void " << label->charstar() << "_end(";
  generatePrototype(op, stateVarsChildren);
  op << ") {\n";
  // actual code here 
  op << "    " << counter->charstar() << "->decrement();\n";
  op << "    if (" << counter->charstar() << "->isDone()) {\n";
  op << "      delete " << counter->charstar() << ";\n";
  if(nextBeginOrEnd == 1)
   op << "      " << next->label->charstar() << "(";
  else
   op << "      " << next->label->charstar() << "_end(";
  generateCall(op, stateVars);
  op << ");\n";
  // end actual code
  op << "    }\n  }\n\n";
}

void CParseNode::generateOlist(XStr& op)
{
  // inlined start function
  op << "  void " << label->charstar() << "(";
  generatePrototype(op, stateVars);
  op << ") {\n";
  // actual code here 
  op << "    CCounter *" << counter->charstar() << "= new CCounter(" <<
        constructs->length() << ");\n";
  for(CParseNode *cn=(CParseNode *)(constructs->begin()); 
                     !constructs->end(); cn=(CParseNode *)(constructs->next())) {
    op << "    " << cn->label->charstar() << "(";
    generateCall(op, stateVarsChildren);
    op << ");\n";
  }
  // end actual code
  op << "  }\n";
  // inlined end function
  op << "  void " << label->charstar() << "_end(";
  generatePrototype(op, stateVarsChildren);
  op << ") {\n";
  // actual code here 
  op << "    " << counter->charstar() << "->decrement();\n";
  op << "    if (" << counter->charstar() << "->isDone()) {\n";
  op << "      delete " << counter->charstar() << ";\n";
  if(nextBeginOrEnd == 1)
   op << "      " << next->label->charstar() << "(";
  else
   op << "      " << next->label->charstar() << "_end(";
  generateCall(op, stateVars);
  op << ");\n";
  // end actual code
  op << "    }\n";
  op << "  }\n";
}

void CParseNode::generateOverlap(XStr& op)
{
  // inlined start function
  op << "  void " << label->charstar() << "(";
  generatePrototype(op, stateVars);
  op << ") {\n";
  // actual code here 
  op << "    " << ((CParseNode *)constructs->front())->label->charstar() <<
        "(";
  generateCall(op, stateVarsChildren);
  op << ");\n";
  // end actual code
  op << "  }\n";
  // inlined end function
  op << "  void " << label->charstar() << "_end(";
  generatePrototype(op, stateVarsChildren);
  op << ") {\n";
  // actual code here 
  if(nextBeginOrEnd == 1)
   op << "    " << next->label->charstar() << "(";
  else
   op << "    " << next->label->charstar() << "_end(";
  generateCall(op, stateVars);
  op << ");\n";
  // end actual code
  op << "  }\n";
}

void CParseNode::generateSlist(XStr& op)
{
  // inlined start function
  op << "  void " << label->charstar() << "(";
  generatePrototype(op, stateVars);
  op << ") {\n";
  // actual code here 
  op << "    " << ((CParseNode *)constructs->front())->label->charstar() <<
        "(";
  generateCall(op, stateVarsChildren);
  op << ");\n";
  // end actual code
  op << "  }\n";
  // inlined end function
  op << "  void " << label->charstar() << "_end(";
  generatePrototype(op, stateVarsChildren);
  op << ") {\n";
  // actual code here 
  if(nextBeginOrEnd == 1)
   op << "    " << next->label->charstar() << "(";
  else
   op << "    " << next->label->charstar() << "_end(";
  generateCall(op, stateVars);
  op << ");\n";
  // end actual code
  op << "  }\n";
}

void CParseNode::generateSdagEntry(XStr& op)
{
  // header file
  op << "public:\n";
  op << "  void " << con1->text->charstar() << "(";
  generatePrototype(op, stateVars);
  op << ") {\n";
  // actual code here 
  op << "    " << ((CParseNode *)constructs->front())->label->charstar() <<
        "(";
  generateCall(op, stateVarsChildren);
  op << ");\n";
  // end actual code
  op << "  }\n\n";
  op << "private:\n";
  op << "  void " << con1->text->charstar() << "_end(";
  generatePrototype(op, stateVars);
  op << ") {\n";
  // actual code here 
  // op << "    delete " << con3->text->charstar() << ";\n";
  // end actual code
  op << "  }\n\n";
}

void CParseNode::generateAtomic(XStr& op)
{
  op << "  void " << label->charstar() << "(";
  generatePrototype(op, stateVars);
  op << ") {\n";
  op << "    " << text->charstar() << "\n";
  if(nextBeginOrEnd == 1)
    op << "    " << next->label->charstar() << "(";
  else
    op << "    " << next->label->charstar() << "_end(";
  generateCall(op, stateVars);
  op << ");\n";
  op << "  }\n\n";
}

void CParseNode::generatePrototype(XStr& op, TList *list)
{
  CStateVar *sv;
  for(sv=(CStateVar *)(list->begin()); !list->end(); ) {
    op << sv->type->charstar() << " " << sv->name->charstar();
    sv = (CStateVar *)list->next();
    if (sv != 0)
      op << ", ";
  }
}

void CParseNode::generateCall(XStr& op, TList *list) {
  CStateVar *sv;
  for(sv=(CStateVar *)list->begin(); !list->end(); ) {
    op << sv->name->charstar();
    sv = (CStateVar *)list->next();
    if (sv != 0)
      op << ", ";
  }
}

// boe = 1, if the next call is to begin construct
// boe = 0, if the next call is to end a contruct
void CParseNode::setNext(CParseNode *n, int boe)
{
  switch(type) {
    case SLIST:
      next = n;
      nextBeginOrEnd = boe;
      {
        CParseNode *cn=(CParseNode *)constructs->begin();
        if (cn==0) // empty slist
          return;
        CParseNode *nextNode=(CParseNode *)constructs->next();
        for(; nextNode != 0;) {
          cn->setNext(nextNode, 1);
          cn = nextNode;
          nextNode = (CParseNode *)constructs->next();
        }
        cn->setNext(this, 0);
      }
      return;
    case SDAGENTRY:
    case OVERLAP:
    case OLIST:
    case FORALL:
    case WHEN:
    case FOR:
    case WHILE:
    case ATOMIC:
    case ELSE:
      next = n;
      nextBeginOrEnd = boe;
      n = this; boe = 0; break;
    case IF:
      next = n;
      nextBeginOrEnd = boe;
      if(con2 != 0)
        con2->setNext(n, boe);
      n = this;
      boe = 0;
      break;
    default:
      break;
  }
  CParseNode *cn;
  for(cn=(CParseNode *)constructs->begin(); !constructs->end(); cn=(CParseNode *)constructs->next()) {
    cn->setNext(n, boe);
  }
}

