#include <algorithm>
#include "CParsedFile.h"

using std::for_each;

namespace xi {

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

  template <typename T>
  struct SdagConCall {
    void (SdagConstruct::*fn)(T);
    T arg;

    SdagConCall(void (SdagConstruct::*fn_)(T), const T& arg_) : fn(fn_), arg(arg_) { }
    void operator()(Entry *e) {
      if (e->sdagCon) {
        (e->sdagCon->*fn)(arg);
      }
    }
  };

  template <>
  struct SdagConCall<void> {
    void (SdagConstruct::*fn)();
    SdagConCall(void (SdagConstruct::*fn_)()) : fn(fn_) { }
    void operator()(Entry *e) {
      if (e->sdagCon) {
        (e->sdagCon->*fn)();
      }
    }
  };

void CParsedFile::doProcess(XStr& classname, XStr& decls, XStr& defs) {
  className = &classname;
  decls << "#define " << classname << "_SDAG_CODE \n";

  for_each(nodeList.begin(), nodeList.end(), SdagConCall<void>(&SdagConstruct::numberNodes));
  for_each(nodeList.begin(), nodeList.end(), SdagConCall<void>(&SdagConstruct::labelNodes));
  for_each(nodeList.begin(), nodeList.end(), SdagConCall<int>(&SdagConstruct::propagateState, 0));
  generateConnectEntryList();
  for_each(nodeList.begin(), nodeList.end(), SdagConCall<void>(&SdagConstruct::generateTrace));
  generateEntryList();
  mapCEntry();

  generateCode(decls, defs);
  generateEntries(decls, defs);
  generateInitFunction(decls, defs);
  generatePupFunction(decls);
  generateRegisterEp(decls, defs);
  generateTraceEp(decls, defs);

#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
  generateDependencyMergePoints(decls); // for Isaac's Critical Path Detection
#endif

  decls.line_append_padding('\\');
  decls << "\n";
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
  decls << "public:\n";
  decls << "  CDep *__cDep;\n";

  XStr name = "_sdag_init";
  generateSignature(decls, defs, container, false, "void", &name, false, NULL);
  defs << "    __cDep = new CDep(" << numEntries << "," << numWhens << ");\n";
  CEntry *en;
  for(en=entryList.begin(); !entryList.end(); en=entryList.next()) {
    en->generateDeps(defs);
  }
  endMethod(defs);

  // Backwards compatibility
  XStr oldname = "__sdag_init";
  generateSignature(decls, defs, container, false, "void", &oldname, false, NULL);
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
