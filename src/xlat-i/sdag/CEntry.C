#include "CEntry.h"
#include "CStateVar.h"
#include "constructs/Constructs.h"
#include "xi-Chare.h"
#include "xi-symbol.h"

using std::list;

namespace xi {
void CEntry::generateDeps(XStr& op) {
  for (list<WhenConstruct*>::iterator cn = whenList.begin(); cn != whenList.end(); ++cn)
    op << "  __dep->addDepends(" << (*cn)->nodeNum << "," << entryNum << ");\n";
}

void generateLocalWrapper(XStr& decls, XStr& defs, int isVoid, XStr& signature,
                          Entry* entry, std::list<CStateVar*>* params, XStr* next, bool isDummy) {
  int numRdmaParams = 0;
  for (std::list<CStateVar*>::iterator it = params->begin(); it != params->end(); ++it) {
    CStateVar& var = **it;
    if (var.isRdma) numRdmaParams++;
  }
  // generate wrapper for local calls to the function
  templateGuardBegin(false, defs);
  defs << entry->getContainer()->tspec() << "void " << entry->getContainer()->baseName()
       << "::" << signature << "{\n";

  if(isDummy) {
    // isDummy being true indicates that the function is a dummy function generated solely
    // for throwing an error if the sdag function is called directly, without a proxy
    defs << "  " << "CkPrintf(\"Error> Direct call to SDAG entry method \\\'%s::%s\\\'!\\n\", \""
         << entry->getContainer()->baseName() << "\", \"" << signature << "\"); \n";
    defs << "  " << "CkAbort(\"Direct SDAG call is not allowed for SDAG entry methods having when constructs. Call such SDAG methods using a proxy\"); \n";
  } else {
    defs << "  " << *entry->genClosureTypeNameProxyTemp << "*"
         << " genClosure = new " << *entry->genClosureTypeNameProxyTemp << "()"
         << ";\n";
    if (params) {
      int i = 0;
      for (std::list<CStateVar*>::iterator it = params->begin(); it != params->end();
           ++it, ++i) {
        CStateVar& var = **it;
        if (var.name) {
          if (var.isRdma) {
            defs << "#if CMK_ONESIDED_IMPL\n";
            if (var.isFirstRdma) {
              defs << "  genClosure->getP" << i++ << "() = " << numRdmaParams << ";\n";
              // Root node is used for ZC Bcast and since this is a direct sdag call
              // CkMyNode() is the root source node for the broadcast
              defs << "  genClosure->getP" << i++ << "() = " << "CkMyNode()" << ";\n";
            }
            defs << "  genClosure->getP" << i << "() = "
                 << "ncpyBuffer_" << var.name << ";\n";
            defs << "#else\n";
            defs << "  genClosure->getP" << i << "() = "
                 << "(" << var.type << "*)"
                 << "ncpyBuffer_" << var.name << ".ptr"
                 << ";\n";
            defs << "#endif\n";
          } else
            defs << "  genClosure->getP" << i << "() = " << var.name << ";\n";
        }
      }
    }
    defs << "  " << ( entry->containsWhenConstruct ? "_sdag_fnc_" : "") << *next << "(genClosure);\n";
    defs << "  genClosure->deref();\n";
  }
  defs << "}\n\n";
  templateGuardEnd(defs);
}

void CEntry::generateCode(XStr& decls, XStr& defs) {
  CStateVar* sv;
  int i = 0;
  int isVoid = 1;
  int lastWasVoid;

  XStr signature;
  signature << *entry << "(";
  for (list<CStateVar*>::iterator it = myParameters.begin(); it != myParameters.end();
       ++it, ++i) {
    sv = *it;
    isVoid = sv->isVoid;
    if ((sv->isMsg != 1) && (sv->isVoid != 1)) {
      if (i > 0) signature << ", ";
      if (sv->byConst) signature << "const ";

      if (sv->isRdma)
        signature << "CkNcpyBuffer ";
      else
        signature << sv->type << " ";
      if (sv->arrayLength != 0)
        signature << "*";
      else if (sv->declaredRef) {
        signature << "&";
      }
      if (sv->numPtrs != 0) {
        for (int k = 0; k < sv->numPtrs; k++) signature << "*";
      }
      if (sv->name != 0) {
        if (sv->isRdma) {
          signature << "ncpyBuffer_" << sv->name;
        } else {
          signature << sv->name;
        }
      }
    } else if (sv->isVoid != 1) {
      if (i < 1)
        signature << sv->type << "* " << sv->name << "_msg";
      else
        printf("ERROR: A message must be the only parameter in an entry function\n");
    } else
      signature << "void";
  }

  signature << ")";

  XStr newSig;

  if (isVoid || needsParamMarshalling) {
    newSig << *entry << "(" << *decl_entry->genClosureTypeNameProxyTemp
           << "* genClosure)";
    decls << "  void " << newSig << ";\n";
    // generate local wrapper decls
    decls << "  void " << signature << ";\n";
    generateLocalWrapper(decls, defs, isVoid, signature, decl_entry, &myParameters,
                         entry);
  } else {  // a message
    newSig << signature << "";
    decls << "  void " << newSig << ";\n";
  }

  templateGuardBegin(false, defs);
  defs << decl_entry->getContainer()->tspec() << "void "
       << decl_entry->getContainer()->baseName() << "::" << newSig << "{\n";
  defs << "  if (!__dep.get()) _sdag_init();\n";

#if CMK_BIGSIM_CHARM
  defs << "  void* _bgParentLog = NULL;\n";
  defs << "  CkElapse(0.01e-6);\n";
  SdagConstruct::generateTlineEndCall(defs);
#endif

  if (needsParamMarshalling || isVoid) {
    // If the genClosure doesn't have a refnum yet, then assign the first
    // parameter as its refnum
    if (refNumNeeded && !isVoid)
      defs << "  if (!genClosure->hasRefnum) "
              "genClosure->setRefnum(genClosure->getP0());\n";

// add the incoming message to a buffer
#if CMK_BIGSIM_CHARM
    defs << "  SDAG::Buffer* cmsgbuf = ";
#endif

    // note that there will always be a closure even when the method has no
    // parameters for consistency
    defs << "  __dep->pushBuffer(" << entryNum << ", genClosure);\n";
  } else {
    // possible memory pressure problem: this message will be kept as long as
    // it is a state parameter! there are ways to remediate this, but it
    // involves either live variable analysis (which is not feasible) or
    // keeping a meta-structure for every message passed in

    // increase reference count by one for the state parameter
    defs << "  CmiReference(UsrToEnv(" << sv->name << "_msg));\n";

#if CMK_BIGSIM_CHARM
    defs << "  SDAG::Buffer* cmsgbuf = ";
#endif
    // refnum automatically extracted from msg by MsgClosure::MsgClosure(...)
    defs << "  __dep->pushBuffer(" << entryNum << ", new SDAG::MsgClosure(" << sv->name
         << "_msg"
         << "));\n";
  }
  // @todo write the code to fetch the message with the ref num

  // search for a continuation to restart execution
  defs << "  SDAG::Continuation* c = __dep->tryFindContinuation(" << entryNum << ");\n";

  // found a continuation
  defs << "  if (c) {\n";

#if USE_CRITICAL_PATH_HEADER_ARRAY
  defs << "    MergeablePathHistory *currentSaved = c->getPath();\n";
  defs << "    mergePathHistory(currentSaved);\n";
#endif
  SdagConstruct::generateTraceEndCall(defs, 2);
#if CMK_BIGSIM_CHARM
  SdagConstruct::generateEndExec(defs);
#endif

  if (whenList.size() == 1) {
    (*whenList.begin())->generateWhenCode(defs, 2);
  } else {
    // switch on the possible entry points for the continuation
    // each continuation entry knows how to generate its own code
    defs << "    switch(c->whenID) {\n";
    for (list<WhenConstruct*>::iterator cn = whenList.begin(); cn != whenList.end();
         ++cn) {
      defs << "    case " << (*cn)->nodeNum << ":\n";
      (*cn)->generateWhenCode(defs, 3);
      defs << "    break;\n";
    }
    defs << "    }\n";
  }

  SdagConstruct::generateDummyBeginExecute(defs, 2, decl_entry);

  // delete the continuation now that we are finished with it
  defs << "    delete c;\n";
  defs << "  }\n";

#if USE_CRITICAL_PATH_HEADER_ARRAY
  defs << "else {\n";
  defs << "    MergeablePathHistory *currentSaved = saveCurrentPath();\n";
  defs << "    buff0->setPath(currentSaved);\n";
  defs << "}\n";
#endif
  defs << "}\n\n";
  templateGuardEnd(defs);
}

list<Entry*> CEntry::getCandidates() { return candidateEntries_; }

void CEntry::addCandidate(Entry* e) { candidateEntries_.push_front(e); }

void CEntry::check() {
  if (decl_entry == NULL) {
    XStr str;
    paramlist->printTypes(str);
    std::string msg = "no matching declaration for entry method \'" +
                      std::string(entry->get_string_const()) + "(" +
                      std::string(str.get_string_const()) + ")\'";
    XLAT_ERROR_NOCOL(msg, first_line_);

    std::list<Entry*> clist = getCandidates();
    if (!clist.empty())
      for (std::list<Entry*>::iterator it = clist.begin(); it != clist.end(); ++it)
        XLAT_NOTE("candidate method not viable: type signatures must match exactly",
                  (*it)->first_line_);
  }
}

}  // namespace xi
