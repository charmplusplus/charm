#include "CEntry.h"
#include "CParseNode.h"

void CEntry::generateDeps(void)
{
  CParseNode *cn;
  for(cn=(CParseNode *)whenList->begin(); !whenList->end(); cn=(CParseNode *)whenList->next()) {
    pH(2, "__cDep->addDepends(%d, %d);\n", cn->nodeNum, entryNum);
  }
}

void CEntry::generateCode(CString *className)
{
  // header file
  pH(1,"void %s(%s *);\n", entry->charstar(), msgType->charstar());

  CParseNode *cn;
  // C++ file
  pC(0, "void %s::%s(%s *msg) {\n",
              className->charstar(),
              entry->charstar(),
              msgType->charstar());
  // actual code begins
  pC(1,"CWhenTrigger *tr;\n");
  if(refNumNeeded) {
    pC(1,"int refnum = CkGetRefNum(msg);\n");
    pC(1,"__cDep->bufferMessage(%d, (void *) msg, refnum);\n", entryNum);
    pC(1,"tr = __cDep->getTrigger(%d, refnum);\n", entryNum);
  } else {
    pC(1,"__cDep->bufferMessage(%d, (void *) msg, 0);\n", entryNum);
    pC(1,"tr = __cDep->getTrigger(%d, 0);\n", entryNum);
  }
  pC(1,"if (tr == 0)\n");
  pC(2,"return;\n");
  if(whenList->length() == 1) {
    cn = (CParseNode *)whenList->begin();
    pC(1,"%s(", cn->label->charstar());
    CStateVar *sv = (CStateVar *)cn->stateVars->begin();
    int i = 0;
    for(; i<(cn->stateVars->length());i++, sv=(CStateVar *)cn->stateVars->next()) {
      if(i!=0)
        pC(0,", ");
      pC(0,"(%s) tr->args[%d]", sv->type->charstar(), i);
    }
    pC(0,");\n");
    pC(1,"return;\n");
  } else {
    pC(1,"switch(tr->whenID) {\n");
    for(cn=(CParseNode *)whenList->begin(); !whenList->end(); cn=(CParseNode *)whenList->next()) {
      pC(2,"case %d:\n", cn->nodeNum);
      pC(3,"%s(", cn->label->charstar());
      CStateVar *sv = (CStateVar *)cn->stateVars->begin();
      int i = 0;
      for(; i<(cn->stateVars->length());i++, sv=(CStateVar *)cn->stateVars->next()) {
        if(i!=0)
          pC(0,", ");
        pC(0,"(%s) tr->args[%d]", sv->type->charstar(), i);
      }
      pC(0,");\n");
      pC(3,"return;\n");
    }
    pC(1,"}\n");
  }
  // actual code ends
  pC(0,"}\n\n");
}

