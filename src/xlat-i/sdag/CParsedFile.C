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
  for(CEntry *ce=entryList.begin(); !entryList.end(); ce=entryList.next())
  {
    ce->print(indent);
    printf("\n");
  }
  for(CParseNode *cn=nodeList.begin(); !nodeList.end(); cn=nodeList.next())
  {
    cn->print(indent);
    printf("\n");
  }
}

void CParsedFile::numberNodes(void)
{
  for(CParseNode *cn=nodeList.begin(); !nodeList.end(); cn=nodeList.next()) {
    cn->numberNodes();
  }
}

void CParsedFile::labelNodes(void)
{
  for(CParseNode *cn=nodeList.begin(); !nodeList.end(); cn=nodeList.next()) {
    cn->labelNodes();
  }
}

void CParsedFile::propagateState(void)
{
  for(CParseNode *cn=nodeList.begin(); !nodeList.end(); cn=nodeList.next()) {
    cn->propagateState();
  }
}

void CParsedFile::generateEntryList(void)
{
  for(CParseNode *cn=nodeList.begin(); !nodeList.end(); cn=nodeList.next()) {
    cn->generateEntryList(entryList, 0);
  }
}

void CParsedFile::generateCode(XStr& op)
{
  for(CParseNode *cn=nodeList.begin(); !nodeList.end(); cn=nodeList.next()) {
    cn->setNext(0,0);
    cn->generateCode(op);
  }
}

void CParsedFile::generateEntries(XStr& op)
{
  CEntry *en;
  op << "public:\n";
  for(en=entryList.begin(); !entryList.end(); en=entryList.next()) {
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
  for(en=entryList.begin(); !entryList.end(); en=entryList.next()) {
    en->generateDeps(op);
  }
  op << "  }\n";
}
