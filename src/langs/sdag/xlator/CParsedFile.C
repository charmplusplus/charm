#include <stdio.h>
#include "CParsedFile.h"

void CParsedFile::print(int indent)
{
  sourceFile->print(indent);
  printf(":\nclass ");
  className->print(indent);
  printf("\n");
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
    cn->labelNodes(className);
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

void CParsedFile::generateCode(void)
{
  for(CParseNode *cn=(CParseNode *)(nodeList->begin()); !nodeList->end(); cn=(CParseNode *)(nodeList->next())) {
    cn->setNext(0,0);
    cn->generateCode();
  }
}

void CParsedFile::generateEntries(void)
{
  CEntry *en;
  pH(0,"public:\n");
  for(en=(CEntry *)(entryList->begin()); !entryList->end(); en=(CEntry *)(entryList->next())) {
    en->generateCode(className);
  }
}

void CParsedFile::generateInitFunction(void)
{
  pH(0,"private:\n");
  pH(1,"CDep *__cDep;\n");
  // pH(1,"void __sdag_init(void);\n");

  pH(1,"void __sdag_init(void) {\n");
  pH(2,"__cDep = new CDep(%d, %d);\n", numEntries, numWhens);
  CEntry *en;
  for(en=(CEntry *)(entryList->begin()); !entryList->end(); en=(CEntry *)(entryList->next())) {
    en->generateDeps();
  }
  pH(1,"}\n");
}
