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

void CParseNode::labelNodes(CString *cname)
{
  char text[128];

  className = cname;
  switch(type) {
    case ENTRY:
    case SDAGENTRY:
      sprintf(text, "%s", con1->text->charstar());
      label = new CString(text);
      break;
    case OVERLAP: 
      sprintf(text, "_overlap_%d", nodeNum); 
      label = new CString(text);
      break;
    case WHEN: 
      sprintf(text, "_when_%d", nodeNum); 
      label = new CString(text);
      break;
    case FOR: 
      sprintf(text, "_for_%d", nodeNum); 
      label = new CString(text);
      break;
    case WHILE: 
      sprintf(text, "_while_%d", nodeNum); 
      label = new CString(text);
      break;
    case IF: 
      sprintf(text, "_if_%d", nodeNum); 
      label = new CString(text);
      if(con2!=0) con2->labelNodes(cname);
      break;
    case ELSE: 
      sprintf(text, "_else_%d", nodeNum); 
      label = new CString(text);
      break;
    case FORALL: 
      sprintf(text, "_forall_%d", nodeNum); 
      label = new CString(text);
      break;
    case SLIST: 
      sprintf(text, "_slist_%d", nodeNum); 
      label = new CString(text);
      break;
    case OLIST: 
      sprintf(text, "_olist_%d", nodeNum); 
      label = new CString(text);
      break;
    case ATOMIC: 
      sprintf(text, "_atomic_%d", nodeNum); 
      label = new CString(text);
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
      for(entry=(CEntry *)(list->begin()); !list->end(); entry=(CEntry *)(list->next())) {
        if(entry->entry->equal(con1->text) &&
           entry->msgType->equal(con3->text)) {
           found = 1;
           entry->whenList->append(thisWhen);
           entryPtr = entry;
	   if(con2)
	     entry->refNumNeeded = 1;
           break;
        }
      }
      if(!found) {
        CEntry *newEntry;
        newEntry = new CEntry(new CString(con1->text), new CString(con3->text));
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
        CString *vType = new CString(con2->text);
        vType->append(" *");
        sv = new CStateVar(vType, new CString(con3->text));
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
      sv = new CStateVar(new CString("int"), new CString(con1->text));
      stateVarsChildren->append(sv);
      {
        char txt[128];
        sprintf(txt, "_cf%d", nodeNum);
        counter = new CString(txt);
        sv = new CStateVar(new CString("CCounter *"), counter);
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
          CString *vType = new CString(en->con3->text);
          vType->append(" *");
          sv = new CStateVar(vType, new CString(en->con4->text));
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
        counter = new CString(txt);
        sv = new CStateVar(new CString("CCounter *"), counter);
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

void CParseNode::generateCode(void)
{
  switch(type) {
    case SDAGENTRY:
      generateSdagEntry();
      break;
    case SLIST:
      generateSlist();
      break;
    case OLIST:
      generateOlist();
      break;
    case FORALL:
      generateForall();
      break;
    case ATOMIC:
      generateAtomic();
      break;
    case IF:
      generateIf();
      if(con2 != 0)
        con2->generateCode();
      break;
    case ELSE:
      generateElse();
      break;
    case WHILE:
      generateWhile();
      break;
    case FOR:
      generateFor();
      break;
    case OVERLAP:
      generateOverlap();
      break;
    case WHEN:
      generateWhen();
      break;
    default:
      break;
  }
  CParseNode *cn;
  for(cn=(CParseNode *)(constructs->begin()); !constructs->end(); cn=(CParseNode *)(constructs->next())) {
    cn->generateCode();
  }
}

void CParseNode::generateWhen(void)
{
  // header file
  pH(1,"int %s(", label->charstar());
  generatePrototype(fh, stateVars);
  pH(0,");\n");
  pH(1,"void %s_end(", label->charstar());
  generatePrototype(fh, stateVarsChildren);
  pH(0,");\n");

  // C++ file: start function
  pC(0, "int %s::%s(", className->charstar(), label->charstar());
  generatePrototype(fC, stateVars);
  pC(0, ") {\n");
  // actual code here 
  TList *elist = con1->constructs;
  CParseNode *el;
  for(el=(CParseNode *)(elist->begin()); !elist->end(); el=(CParseNode *)(elist->next())) {
    pC(1, "CMsgBuffer *%s_buf;\n", el->con4->text->charstar());
    pC(1, "%s *%s;\n", el->con3->text->charstar(),
                el->con4->text->charstar());
  }
  pC(0, "\n");
  for(el=(CParseNode *)(elist->begin()); !elist->end(); el=(CParseNode *)(elist->next())) {
    if(el->con2 == 0)
      pC(1, "%s_buf = __cDep->getMessage(%d);\n",
                  el->con4->text->charstar(),
                  el->entryPtr->entryNum);
    else
      pC(1, "%s_buf = __cDep->getMessage(%d, %s);\n",
                  el->con4->text->charstar(),
                  el->entryPtr->entryNum,
                  el->con2->text->charstar());
  }
  pC(0, "\n");
  pC(1, "if (");
  for(el=(CParseNode *)(elist->begin()); !elist->end();) {
    pC(0, "(%s_buf != 0)", el->con4->text->charstar());
    el = (CParseNode *)(elist->next());
    if(el != 0)
      pC(0, "&&");
  }
  pC(0, ") {\n");
  for(el=(CParseNode *)(elist->begin()); !elist->end(); el=(CParseNode *)(elist->next())) {
    pC(2, "%s = (%s *) %s_buf->msg;\n",
                  el->con4->text->charstar(),
                  el->con3->text->charstar(),
                  el->con4->text->charstar());
    pC(2, "__cDep->removeMessage(%s_buf);\n",
                el->con4->text->charstar());
  }
  pC(2, "%s(", ((CParseNode *)constructs->front())->label->charstar());
  generateCall(fC, stateVarsChildren);
  pC(0, ");\n");
  pC(2, "return 1;\n");
  pC(1, "} else {\n");

  int nRefs=0, nAny=0;
  for(el=(CParseNode *)(elist->begin()); !elist->end(); el=(CParseNode *)(elist->next())) {
    if(el->con2 == 0)
      nAny++;
    else
      nRefs++;
  }

// keep these consts consistent with CWhenTrigger.h in runtime

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

  pC(2, "CWhenTrigger *tr;\n");
  pC(2, "tr = new CWhenTrigger(%d, %d, %d, %d);\n",
                   nodeNum, 
                   stateVars->length(), nRefs, nAny);
  CStateVar *sv;
  int iArgs=0;
  for(sv=(CStateVar *)(stateVars->begin());!stateVars->end();sv=(CStateVar *)(stateVars->next())) {
    pC(2, "tr->args[%d] = (size_t) %s;\n", iArgs++,
                sv->name->charstar());
  }
  int iRef=0, iAny=0;
  for(el=(CParseNode *)(elist->begin()); !elist->end(); el=(CParseNode *)(elist->next())) {
    if(el->con2 == 0) {
      pC(2, "tr->anyEntries[%d] = %d;\n", iAny++,
                  el->entryPtr->entryNum);
    } else {
      pC(2, "tr->entries[%d] = %d;\n", iRef,
                  el->entryPtr->entryNum);
      pC(2, "tr->refnums[%d] = %s;\n", iRef++,
                  el->con2->text->charstar());
    }
  }
  pC(2, "__cDep->Register(tr);\n");
  pC(2, "return 0;\n");
  pC(1, "}\n");
  // end actual code
  pC(0, "}\n\n");
  // C++ file: end function
  pC(0, "void %s::%s_end(", className->charstar(), label->charstar());
  generatePrototype(fC, stateVarsChildren);
  pC(0, ") {\n");
  // actual code here 
  for(el=(CParseNode *)(elist->begin()); !elist->end(); el=(CParseNode *)(elist->next())) {
    // pC(1, "delete %s;\n", el->con4->text->charstar());
  }
  if(nextBeginOrEnd == 1)
   pC(1, "%s(", next->label->charstar());
  else
   pC(1, "%s_end(", next->label->charstar());
  generateCall(fC, stateVars);
  pC(0, ");\n");
  // end actual code
  pC(0, "}\n\n");
}

void CParseNode::generateWhile(void)
{
  // header file
  pH(1,"void %s(", label->charstar());
  generatePrototype(fh, stateVars);
  pH(0,");\n");
  pH(1,"void %s_end(", label->charstar());
  generatePrototype(fh, stateVarsChildren);
  pH(0,");\n");

  // C++ file: start function
  pC(0, "void %s::%s(", className->charstar(), label->charstar());
  generatePrototype(fC, stateVars);
  pC(0, ") {\n");
  // actual code here 
  pC(1, "if (%s) {\n", con1->text->charstar());
  pC(2, "%s(", ((CParseNode *)constructs->front())->label->charstar());
  generateCall(fC, stateVarsChildren);
  pC(0, ");\n");
  pC(1, "} else {\n");
  if(nextBeginOrEnd == 1)
   pC(2, "%s(", next->label->charstar());
  else
   pC(2, "%s_end(", next->label->charstar());
  generateCall(fC, stateVars);
  pC(0, ");\n");
  pC(1, "}\n");
  // end actual code
  pC(0, "}\n\n");
  // C++ file: end function
  pC(0, "void %s::%s_end(", className->charstar(), label->charstar());
  generatePrototype(fC, stateVarsChildren);
  pC(0, ") {\n");
  // actual code here 
  pC(1, "if (%s) {\n", con1->text->charstar());
  pC(2, "%s(", ((CParseNode *)constructs->front())->label->charstar());
  generateCall(fC, stateVarsChildren);
  pC(0, ");\n");
  pC(1, "} else {\n");
  if(nextBeginOrEnd == 1)
   pC(2, "%s(", next->label->charstar());
  else
   pC(2, "%s_end(", next->label->charstar());
  generateCall(fC, stateVars);
  pC(0, ");\n");
  pC(1, "}\n");
  // end actual code
  pC(0, "}\n\n");
}

void CParseNode::generateFor(void)
{
#if 0
  // header file
  pH(1,"void %s(", label->charstar());
  generatePrototype(fh, stateVars);
  pH(0,");\n");
  pH(1,"void %s_end(", label->charstar());
  generatePrototype(fh, stateVarsChildren);
  pH(0,");\n");
#endif

  // header file: inlined start function
  pH(1, "void %s(", label->charstar());
  generatePrototype(fh, stateVars);
  pH(0, ") {\n");
  // actual code here 
  pH(2, "%s;\n", con1->text->charstar());
  pH(2, "if (%s) {\n", con2->text->charstar());
  pH(3, "%s(", ((CParseNode *)constructs->front())->label->charstar());
  generateCall(fh, stateVarsChildren);
  pH(0, ");\n");
  pH(2, "} else {\n");
  if(nextBeginOrEnd == 1)
   pH(3, "%s(", next->label->charstar());
  else
   pH(3, "%s_end(", next->label->charstar());
  generateCall(fh, stateVars);
  pH(0, ");\n");
  pH(2, "}\n");
  // end actual code
  pH(1, "}\n");
  // header file: inlined end function
  pH(1, "void %s_end(",  label->charstar());
  generatePrototype(fh, stateVarsChildren);
  pH(0, ") {\n");
  // actual code here 
  pH(2, "%s;\n", con3->text->charstar());
  pH(2, "if (%s) {\n", con2->text->charstar());
  pH(3, "%s(", ((CParseNode *)constructs->front())->label->charstar());
  generateCall(fh, stateVarsChildren);
  pH(0, ");\n");
  pH(2, "} else {\n");
  if(nextBeginOrEnd == 1)
   pH(3, "%s(", next->label->charstar());
  else
   pH(3, "%s_end(", next->label->charstar());
  generateCall(fh, stateVars);
  pH(0, ");\n");
  pH(2, "}\n");
  // end actual code
  pH(1, "}\n");
}

void CParseNode::generateIf(void)
{
  // header file
  pH(1,"void %s(", label->charstar());
  generatePrototype(fh, stateVars);
  pH(0,");\n");
  pH(1,"void %s_end(", label->charstar());
  generatePrototype(fh, stateVarsChildren);
  pH(0,");\n");

  // C++ file: start function
  pC(0, "void %s::%s(", className->charstar(), label->charstar());
  generatePrototype(fC, stateVars);
  pC(0, ") {\n");
  // actual code here 
  pC(1, "if (%s) {\n", con1->text->charstar());
  pC(2, "%s(", ((CParseNode *)constructs->front())->label->charstar());
  generateCall(fC, stateVarsChildren);
  pC(0, ");\n");
  pC(1, "} else {\n");
  if (con2 != 0) {
    pC(2, "%s(", con2->label->charstar());
    generateCall(fC, stateVarsChildren);
    pC(0, ");\n");
  } else {
    pC(2, "%s_end(", label->charstar());
    generateCall(fC, stateVarsChildren);
    pC(0, ");\n");
  }
  pC(1, "}\n");
  // end actual code
  pC(0, "}\n\n");
  // C++ file: end function
  pC(0, "void %s::%s_end(", className->charstar(), label->charstar());
  generatePrototype(fC, stateVarsChildren);
  pC(0, ") {\n");
  // actual code here 
  if(nextBeginOrEnd == 1)
   pC(2, "%s(", next->label->charstar());
  else
   pC(2, "%s_end(", next->label->charstar());
  generateCall(fC, stateVars);
  pC(0, ");\n");
  // end actual code
  pC(0, "}\n\n");
}

void CParseNode::generateElse(void)
{
  // header file
  pH(1,"void %s(", label->charstar());
  generatePrototype(fh, stateVars);
  pH(0,");\n");
  pH(1,"void %s_end(", label->charstar());
  generatePrototype(fh, stateVarsChildren);
  pH(0,");\n");

  // C++ file: start function
  pC(0, "void %s::%s(", className->charstar(), label->charstar());
  generatePrototype(fC, stateVars);
  pC(0, ") {\n");
  // actual code here 
  pC(1, "%s(", ((CParseNode *)constructs->front())->label->charstar());
  generateCall(fC, stateVarsChildren);
  pC(0, ");\n");
  // end actual code
  pC(0, "}\n\n");
  // C++ file: end function
  pC(0, "void %s::%s_end(", className->charstar(), label->charstar());
  generatePrototype(fC, stateVarsChildren);
  pC(0, ") {\n");
  // actual code here 
  if(nextBeginOrEnd == 1)
   pC(2, "%s(", next->label->charstar());
  else
   pC(2, "%s_end(", next->label->charstar());
  generateCall(fC, stateVars);
  pC(0, ");\n");
  // end actual code
  pC(0, "}\n\n");
}

void CParseNode::generateForall(void)
{
  // header file
  pH(1,"void %s(", label->charstar());
  generatePrototype(fh, stateVars);
  pH(0,");\n");
  pH(1,"void %s_end(", label->charstar());
  generatePrototype(fh, stateVarsChildren);
  pH(0,");\n");

  // C++ file: start function
  pC(0, "void %s::%s(", className->charstar(), label->charstar());
  generatePrototype(fC, stateVars);
  pC(0, ") {\n");
  // actual code here 
  pC(1, "int __first = (%s), __last = (%s), __stride = (%s);\n",
              con2->text->charstar(), con3->text->charstar(),
              con4->text->charstar());
  pC(1, "if (__first > __last) {\n");
  pC(2, "__swap(&__first, &__last);\n");
  pC(2, "__stride = -__stride;\n");
  pC(1, "}\n");
  pC(1, "CCounter *%s = new CCounter(__first,__last,__stride);\n", 
              counter->charstar());
  pC(1, "for(int %s=__first;%s<=__last;%s+=__stride) {\n",
              con1->text->charstar(), con1->text->charstar(),
              con1->text->charstar());
  pC(2, "%s(", ((CParseNode *)constructs->front())->label->charstar());
  generateCall(fC, stateVarsChildren);
  pC(0, ");\n");
  pC(1, "}\n");
  // end actual code
  pC(0, "}\n\n");
  // C++ file: end function
  pC(0, "void %s::%s_end(", className->charstar(), label->charstar());
  generatePrototype(fC, stateVarsChildren);
  pC(0, ") {\n");
  // actual code here 
  pC(1, "%s->decrement();\n", counter->charstar());
  pC(1, "if (%s->isDone()) {\n", counter->charstar());
  pC(2, "delete %s;\n", counter->charstar());
  if(nextBeginOrEnd == 1)
   pC(2, "%s(", next->label->charstar());
  else
   pC(2, "%s_end(", next->label->charstar());
  generateCall(fC, stateVars);
  pC(0, ");\n");
  // end actual code
  pC(1, "}\n}\n\n");
}

void CParseNode::generateOlist(void)
{
#if 0
  // header file
  pH(1,"void %s(", label->charstar());
  generatePrototype(fh, stateVars);
  pH(0,");\n");
  pH(1,"void %s_end(", label->charstar());
  generatePrototype(fh, stateVarsChildren);
  pH(0,");\n");
#endif

  // header file: inlined start function
  pH(1, "void %s(",  label->charstar());
  generatePrototype(fh, stateVars);
  pH(0, ") {\n");
  // actual code here 
  pH(2, "CCounter *%s = new CCounter(%d);\n", counter->charstar(),
                                                      (CParseNode *)constructs->length());
  for(CParseNode *cn=(CParseNode *)(constructs->begin()); 
                     !constructs->end(); cn=(CParseNode *)(constructs->next())) {
    pH(2, "%s(", cn->label->charstar());
    generateCall(fh, stateVarsChildren);
    pH(0, ");\n");
  }
  // end actual code
  pH(1, "}\n");
  // header file: inlined end function
  pH(1, "void %s_end(", label->charstar());
  generatePrototype(fh, stateVarsChildren);
  pH(0, ") {\n");
  // actual code here 
  pH(2, "%s->decrement();\n", counter->charstar());
  pH(2, "if (%s->isDone()) {\n", counter->charstar());
  pH(3, "delete %s;\n", counter->charstar());
  if(nextBeginOrEnd == 1)
   pH(3, "%s(", next->label->charstar());
  else
   pH(3, "%s_end(", next->label->charstar());
  generateCall(fh, stateVars);
  pH(0, ");\n");
  // end actual code
  pH(2, "}\n");
  pH(1, "}\n");
}

void CParseNode::generateOverlap(void)
{
#if 0
  // header file
  pH(1,"void %s(", label->charstar());
  generatePrototype(fh, stateVars);
  pH(0,");\n");
  pH(1,"void %s_end(", label->charstar());
  generatePrototype(fh, stateVarsChildren);
  pH(0,");\n");
#endif

  // header file: inlined start function
  pH(1, "void %s(", label->charstar());
  generatePrototype(fh, stateVars);
  pH(0, ") {\n");
  // actual code here 
  pH(2, "%s(", ((CParseNode *)constructs->front())->label->charstar());
  generateCall(fh, stateVarsChildren);
  pH(0, ");\n");
  // end actual code
  pH(1, "}\n");
  // header file: inlined end function
  pH(1, "void %s_end(", label->charstar());
  generatePrototype(fh, stateVarsChildren);
  pH(0, ") {\n");
  // actual code here 
  if(nextBeginOrEnd == 1)
   pH(2, "%s(", next->label->charstar());
  else
   pH(2, "%s_end(", next->label->charstar());
  generateCall(fh, stateVars);
  pH(0, ");\n");
  // end actual code
  pH(1, "}\n");
}

void CParseNode::generateSlist(void)
{
#if 0
  // header file
  pH(1,"void %s(", label->charstar());
  generatePrototype(fh, stateVars);
  pH(0,");\n");
  pH(1,"void %s_end(", label->charstar());
  generatePrototype(fh, stateVarsChildren);
  pH(0,");\n");
#endif

  // header file: inlined start function
  pH(1, "void %s(", label->charstar());
  generatePrototype(fh, stateVars);
  pH(0, ") {\n");
  // actual code here 
  pH(2, "%s(", ((CParseNode *)constructs->front())->label->charstar());
  generateCall(fh, stateVarsChildren);
  pH(0, ");\n");
  // end actual code
  pH(1, "}\n");
  // header file: inlined end function
  pH(1, "void %s_end(", label->charstar());
  generatePrototype(fh, stateVarsChildren);
  pH(0, ") {\n");
  // actual code here 
  if(nextBeginOrEnd == 1)
   pH(2, "%s(", next->label->charstar());
  else
   pH(2, "%s_end(", next->label->charstar());
  generateCall(fh, stateVars);
  pH(0, ");\n");
  // end actual code
  pH(1, "}\n");
}

void CParseNode::generateSdagEntry(void)
{
  // header file
  pH(0,"public:\n");
  pH(1,"void %s(", con1->text->charstar());
  generatePrototype(fh, stateVars);
  pH(0,");\n");
  pH(0,"private:\n");
  pH(1,"void %s_end(", con1->text->charstar());
  generatePrototype(fh, stateVarsChildren);
  pH(0,");\n");

  // C++ file
  pC(0, "void %s::%s(", className->charstar(), con1->text->charstar());
  generatePrototype(fC, stateVars);
  pC(0, ") {\n");
  // actual code here 
  pC(1, "%s(", ((CParseNode *)constructs->front())->label->charstar());
  generateCall(fC, stateVarsChildren);
  pC(0, ");\n");
  // end actual code
  pC(0, "}\n\n");
  pC(0, "void %s::%s_end(", className->charstar(), con1->text->charstar());
  generatePrototype(fC, stateVars);
  pC(0, ") {\n");
  // actual code here 
  // pC(1, "delete %s;\n", con3->text->charstar());
  // end actual code
  pC(0, "}\n\n");
}

void CParseNode::generateAtomic(void)
{
  pH(1,"void %s(", label->charstar());
  generatePrototype(fh, stateVars);
  pH(0,");\n");
  pC(0, "void %s::%s(", className->charstar(), label->charstar());
  generatePrototype(fC, stateVars);
  pC(0, ") {\n");
  pC(0, "%s\n", text->charstar());
  if(nextBeginOrEnd == 1)
    pC(1, "%s(", next->label->charstar());
  else
    pC(1, "%s_end(", next->label->charstar());
  generateCall(fC, stateVars);
  pC(0, ");\n");
  pC(0, "}\n\n");
}

void CParseNode::generatePrototype(FILE *f, TList *list)
{
  CStateVar *sv;
  for(sv=(CStateVar *)(list->begin()); !list->end(); ) {
    fprintf(f, "%s %s", sv->type->charstar(), sv->name->charstar());
    sv = (CStateVar *)list->next();
    if (sv != 0)
      fprintf(f, ", ");
  }
}

void CParseNode::generateCall(FILE *f, TList *list) {
  CStateVar *sv;
  for(sv=(CStateVar *)list->begin(); !list->end(); ) {
    fprintf(f, "%s", sv->name->charstar());
    sv = (CStateVar *)list->next();
    if (sv != 0)
      fprintf(f, ", ");
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

