/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

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
  CParseNode *cn;
  // C++ file
  pH(1, "void %s(%s *msg) {\n",
              entry->charstar(),
              msgType->charstar());
  // actual code begins
  pH(2,"CWhenTrigger *tr;\n");
  if(refNumNeeded) {
    pH(2,"int refnum = CkGetRefNum(msg);\n");
    pH(2,"__cDep->bufferMessage(%d, (void *) msg, refnum);\n", entryNum);
    pH(2,"tr = __cDep->getTrigger(%d, refnum);\n", entryNum);
  } else {
    pH(2,"__cDep->bufferMessage(%d, (void *) msg, 0);\n", entryNum);
    pH(2,"tr = __cDep->getTrigger(%d, 0);\n", entryNum);
  }
  pH(2,"if (tr == 0)\n");
  pH(3,"return;\n");
  if(whenList->length() == 1) {
    cn = (CParseNode *)whenList->begin();
    pH(2,"%s(", cn->label->charstar());
    CStateVar *sv = (CStateVar *)cn->stateVars->begin();
    int i = 0;
    for(; i<(cn->stateVars->length());i++, sv=(CStateVar *)cn->stateVars->next()) {
      if(i!=0)
        pH(1,", ");
      pH(1,"(%s) tr->args[%d]", sv->type->charstar(), i);
    }
    pH(1,");\n");
    pH(2,"return;\n");
  } else {
    pH(2,"switch(tr->whenID) {\n");
    for(cn=(CParseNode *)whenList->begin(); !whenList->end(); cn=(CParseNode *)whenList->next()) {
      pH(3,"case %d:\n", cn->nodeNum);
      pH(4,"%s(", cn->label->charstar());
      CStateVar *sv = (CStateVar *)cn->stateVars->begin();
      int i = 0;
      for(; i<(cn->stateVars->length());i++, sv=(CStateVar *)cn->stateVars->next()) {
        if(i!=0)
          pH(1,", ");
        pH(1,"(%s) tr->args[%d]", sv->type->charstar(), i);
      }
      pH(1,");\n");
      pH(4,"return;\n");
    }
    pH(2,"}\n");
  }
  // actual code ends
  pH(1,"}\n\n");
}

