#include "SdagEntry.h"
#include "CStateVar.h"
#include "xi-Entry.h"
#include "xi-Parameter.h"
#include "xi-Chare.h"
#include <list>

namespace xi {

SdagEntryConstruct::SdagEntryConstruct(SdagConstruct *body)
: SdagConstruct(SSDAGENTRY, body)
{ }

SdagEntryConstruct::SdagEntryConstruct(SListConstruct *body)
: SdagConstruct(SSDAGENTRY, body)
{ }

void SdagEntryConstruct::generateCode(XStr& decls, XStr& defs, Entry *entry) {
  buildTypes(encapState);
  buildTypes(encapStateChild);

  if (entry->isConstructor()) {
    std::cerr << cur_file << ":" << entry->getLine()
              << ": Chare constructor cannot be defined with SDAG code" << std::endl;
    exit(1);
  }

  decls << "public:\n";

  XStr signature;

  signature << con1->text;
  signature << "(";
  if (stateVars) {
    int count = 0;
    for (std::list<CStateVar*>::iterator iter = stateVars->begin(); iter != stateVars->end(); ++iter) {
      CStateVar& var = **iter;
      if (var.isVoid != 1) {
        if (count != 0) signature << ", ";
        if (var.byConst) signature << "const ";
        if (var.type != 0) signature << var.type << " ";
        if (var.arrayLength != NULL) signature << "* ";
        if (var.declaredRef) signature << "& ";
        if (var.name != 0) signature << var.name;
        count++;
      }
    }
  }
  signature << ")";

  if (!entry->param->isVoid())
    decls << "  void " <<  signature << ";\n";

  // generate wrapper for local calls to the function
  if (entry->paramIsMarshalled() && !entry->param->isVoid())
    generateLocalWrapper(decls, defs, entry->param->isVoid(), signature, entry, stateVars, con1->text);

  generateClosureSignature(decls, defs, entry, false, "void", con1->text, false, encapState);

#if CMK_BIGSIM_CHARM
  generateEndSeq(defs);
#endif
  if (!entry->getContainer()->isGroup() || !entry->isConstructor())
    generateTraceEndCall(defs, 1);

  defs << "  if (!__dep.get()) _sdag_init();\n";

  // is a message sdag entry, in this case since this is a SDAG entry, there
  // will only be one parameter which is the message (called 'gen0')
  if (!entry->paramIsMarshalled() && !entry->param->isVoid()) {
    // increase reference count by one for the state parameter
    defs << "  CmiReference(UsrToEnv(gen0));\n";
  }

  defs << "  ";
  generateCall(defs, encapStateChild, encapStateChild, constructs->front()->label);

#if CMK_BIGSIM_CHARM
  generateTlineEndCall(defs);
  generateBeginExec(defs, "spaceholder");
#endif
  if (!entry->getContainer()->isGroup() || !entry->isConstructor())
    generateDummyBeginExecute(defs, 1);

  endMethod(defs);

  decls << "private:\n";
  generateClosureSignature(decls, defs, entry, false, "void", con1->text, true,
#if CMK_BIGSIM_CHARM
                       encapStateChild
#else
                       encapState
#endif
                       );

  if (!entry->paramIsMarshalled() && !entry->param->isVoid()) {
    // decrease reference count by one for the message
    defs << "  CmiFree(UsrToEnv(gen0));\n";
  }

  endMethod(defs);

  generateChildrenCode(decls, defs, entry);
}

void SdagEntryConstruct::numberNodes() {
  nodeNum = numSdagEntries++;
  SdagConstruct::numberNodes();
}

void SdagEntryConstruct::labelNodes() {
  label = createLabel(con1->text->charstar(), -1);
  SdagConstruct::labelNodes();
}

}   // namespace xi
