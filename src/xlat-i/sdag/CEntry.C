/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include "CEntry.h"
#include "CParseNode.h"

void CEntry::generateDeps(XStr& op)
{
  CParseNode *cn;
  for(cn=whenList.begin(); !whenList.end(); cn=whenList.next()) {
    op << "    __cDep->addDepends("<<cn->nodeNum<<","<<entryNum<<");\n";
  }
}

void CEntry::generateCode(XStr& op)
{
  CParseNode *cn;
  op << "  void "<< *entry << "(" << *msgType << " *msg) {\n";
  op << "    CWhenTrigger *tr;\n";
  if(refNumNeeded) {
    op << "    int refnum = CkGetRefNum(msg);\n";
    op << "    __cDep->bufferMessage("<<entryNum<<",(void *) msg,refnum);\n";
    op << "    tr = __cDep->getTrigger("<<entryNum<<", refnum);\n";
  } else {
    op << "    __cDep->bufferMessage("<<entryNum<<", (void *) msg, 0);\n";
    op << "    tr = __cDep->getTrigger("<<entryNum<<", 0);\n";
  }
  op << "    if (tr == 0)\n";
  op << "      return;\n";
  if(whenList.length() == 1) {
    cn = whenList.begin();
    op << "    " << cn->label->charstar() << "(";
    CStateVar *sv = (CStateVar *)cn->stateVars->begin();
    int i = 0;
    for(; i<(cn->stateVars->length());i++, sv=(CStateVar *)cn->stateVars->next()) {
      if(i!=0)
        op << ", ";
      op << "(" << sv->type->charstar() << ") tr->args[" << i << "]";
    }
    op << ");\n";
// gzheng
    op << "    delete tr;\n";
    op << "    return;\n";
  } else {
    op << "    switch(tr->whenID) {\n";
    for(cn=whenList.begin(); !whenList.end(); cn=whenList.next()) {
      op << "      case " << cn->nodeNum << ":\n";
      op << cn->label->charstar() << "(";
      CStateVar *sv = (CStateVar *)cn->stateVars->begin();
      int i = 0;
      for(; i<(cn->stateVars->length());i++, sv=(CStateVar *)cn->stateVars->next()) {
        if(i!=0)
          op << ", ";
        op << "(" << sv->type->charstar() << ") tr->args[" << i << "]";
      }
      op << ");\n";
// gzheng
      op << "      delete tr;\n";
      op << "      return;\n";
    }
    op << "    }\n";
  }
  // actual code ends
  op << "  }\n\n";
}

