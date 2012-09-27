#include "CParsedFile.h"

namespace xi {

///////////////////////////// CPARSEDFILE //////////////////////
/*void CParsedFile::print(int indent)
{
  for(CEntry *ce=entryList.begin(); !entryList.end(); ce=entryList.next())
  {
    ce->print(indent);
    printf("\n");
  }
  for(SdagConstruct *cn=nodeList.begin(); !nodeList.end(); cn=nodeList.next())
  {
    cn->print(indent);
    printf("\n");
  }
}
*/
XStr *CParsedFile::className = NULL;

void CParsedFile::numberNodes(void)
{
  for(std::list<Entry*>::iterator cn = nodeList.begin(); cn != nodeList.end(); ++cn) {
    if ((*cn)->sdagCon != 0) {
      (*cn)->sdagCon->numberNodes();
    }
  }
}

void CParsedFile::labelNodes(void)
{
  for(std::list<Entry*>::iterator cn = nodeList.begin(); cn != nodeList.end(); ++cn) {
    if ((*cn)->sdagCon != 0) {
      (*cn)->sdagCon->labelNodes();
    }
  }
}

void CParsedFile::propagateState(void)
{
  for(std::list<Entry*>::iterator cn = nodeList.begin(); cn != nodeList.end(); ++cn) {
    (*cn)->sdagCon->propagateState(0);
  }
}

void CParsedFile::mapCEntry(void)
{
  CEntry *en;
  for(en=entryList.begin(); !entryList.end(); en=entryList.next()) {
    container->lookforCEntry(en);
  }
}

void CParsedFile::generateEntryList(void)
{
  for(std::list<Entry*>::iterator cn = nodeList.begin(); cn != nodeList.end(); ++cn) {
    (*cn)->sdagCon->generateEntryList(entryList, (SdagConstruct*)0);
  }
}

void CParsedFile::generateConnectEntryList(void)
{
  for(std::list<Entry*>::iterator cn = nodeList.begin(); cn != nodeList.end(); ++cn) {
    (*cn)->sdagCon->generateConnectEntryList(connectEntryList);
  }
}

void CParsedFile::generateCode(XStr& decls, XStr& defs)
{
  for(std::list<Entry*>::iterator cn = nodeList.begin(); cn != nodeList.end(); ++cn) {
    (*cn)->sdagCon->setNext(0, 0);
    (*cn)->sdagCon->generateCode(decls, defs, *cn);
  }
}

void CParsedFile::generateEntries(XStr& decls, XStr& defs)
{
  CEntry *en;
  SdagConstruct *sc;
  decls << "public:\n";
  for(sc=connectEntryList.begin(); !connectEntryList.end(); sc=connectEntryList.next())
     sc->generateConnectEntries(decls);
  for(en=entryList.begin(); !entryList.end(); en=entryList.next()) {
    en->generateCode(decls, defs);
  }
}

void CParsedFile::generateInitFunction(XStr& decls, XStr& defs)
{
  decls << "private:\n";
  decls << "  CDep *__cDep;\n";

  XStr name = "__sdag_init";
  generateSignature(decls, defs, container, false, "void", &name, false, NULL);
  defs << "    __cDep = new CDep(" << numEntries << "," << numWhens << ");\n";
  CEntry *en;
  for(en=entryList.begin(); !entryList.end(); en=entryList.next()) {
    en->generateDeps(defs);
  }
  endMethod(defs);
}


/**
    Create a merging point for each of the places where multiple
    dependencies lead into some future task.

    Used by Isaac's critical path detection
*/
void CParsedFile::generateDependencyMergePoints(XStr& decls)
{
  decls << " \n";

  // Each when statement will have a set of message dependencies, and
  // also the dependencies from completion of previous task
  for(int i=0;i<numWhens;i++){
    decls << "  MergeablePathHistory _when_" << i << "_PathMergePoint; /* For Critical Path Detection */ \n";
  }

  // The end of each overlap block will have multiple paths that merge
  // before the subsequent task is executed
  for(int i=0;i<numOlists;i++){
    decls << "  MergeablePathHistory olist__co" << i << "_PathMergePoint; /* For Critical Path Detection */ \n";
  }
}

void CParsedFile::generatePupFunction(XStr& decls)
{
  decls << "public:\n";
  decls << "  void __sdag_pup(PUP::er& p) {\n";
  decls << "    if (__cDep) { __cDep->pup(p); }\n";
  decls << "  }\n";
}

void CParsedFile::generateTrace()
{
  for(std::list<Entry*>::iterator cn = nodeList.begin(); cn != nodeList.end(); ++cn) {
    if ((*cn)->sdagCon != 0) {
      (*cn)->sdagCon->generateTrace();
    }
  }
}

void CParsedFile::generateRegisterEp(XStr& decls, XStr& defs)
{
  XStr name = "__sdag_register";
  generateSignature(decls, defs, container, true, "void", &name, false, NULL);

  for(std::list<Entry*>::iterator cn = nodeList.begin(); cn != nodeList.end(); ++cn) {
    if ((*cn)->sdagCon != 0) {
      (*cn)->sdagCon->generateRegisterEp(defs);
    }
  }
  endMethod(defs);
}

void CParsedFile::generateTraceEp(XStr& decls, XStr& defs)
{
  for(std::list<Entry*>::iterator cn = nodeList.begin(); cn != nodeList.end(); ++cn) {
    if ((*cn)->sdagCon != 0) {
      (*cn)->sdagCon->generateTraceEp(decls, defs, container);
    }
  }
}

}
