/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include <stdio.h>
#include "CParsedFile.h"

void CParsedFile::print(int indent)
{
  for(CEntry *ce=(CEntry *)(entryList->begin()); !entryList->end(); ce=(CEntry *)(entryList->next()))
  {
    ce->print(indent);
    printf("\n");
  }
  for(CParseNode *cn=(CParseNode *)(nodeList->begin()); !nodeList->end(); cn=(CParseNode *)(nodeList->next()))
  {
    cn->print(indent);
    printf("\n");
  }
}

void CParsedFile::numberNodes(void)
{
  for(CParseNode *cn=(CParseNode *)(nodeList->begin()); !nodeList->end(); cn=(CParseNode *)(nodeList->next())) {
    cn->numberNodes();
  }
}

void CParsedFile::labelNodes(void)
{
  for(CParseNode *cn=(CParseNode *)(nodeList->begin()); !nodeList->end(); cn=(CParseNode *)(nodeList->next())) {
    cn->labelNodes();
  }
}

void CParsedFile::propogateState(void)
{
  for(CParseNode *cn=(CParseNode *)(nodeList->begin()); !nodeList->end(); cn=(CParseNode *)(nodeList->next())) {
    cn->propogateState(0);
  }
}

void CParsedFile::generateEntryList(void)
{
  for(CParseNode *cn=(CParseNode *)(nodeList->begin()); !nodeList->end(); cn=(CParseNode *)(nodeList->next())) {
    cn->generateEntryList(entryList, 0);
  }
}

void CParsedFile::generateCode(XStr& op)
{
  for(CParseNode *cn=(CParseNode *)(nodeList->begin()); !nodeList->end(); cn=(CParseNode *)(nodeList->next())) {
    cn->setNext(0,0);
    cn->generateCode(op);
  }
}

void CParsedFile::generateEntries(XStr& op)
{
  CEntry *en;
  op << "public:\n";
  for(en=(CEntry *)(entryList->begin()); !entryList->end(); en=(CEntry *)(entryList->next())) {
    en->generateCode(op);
  }
}

void CParsedFile::generateInitFunction(XStr& op)
{
  op << "private:\n";
  op << "  CDep *__cDep;\n";
  op << "  void __sdag_init(void) {\n";
  op << "    __cDep = new CDep("<<numEntries<<","<<numWhens<<");\n";
  CEntry *en;
  for(en=(CEntry *)(entryList->begin()); !entryList->end(); en=(CEntry *)(entryList->next())) {
    en->generateDeps(op);
  }
  op << "  }\n";
}
