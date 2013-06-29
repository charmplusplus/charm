#include "CEntry.h"
#include "xi-symbol.h"
#include "CStateVar.h"

using std::list;

namespace xi {
  void CEntry::generateDeps(XStr& op) {
    for (list<WhenConstruct*>::iterator cn = whenList.begin(); cn != whenList.end(); ++cn)
      op << "    __dep->addDepends(" << (*cn)->nodeNum << "," << entryNum << ");\n";
  }

  void CEntry::generateLocalWrapper(XStr& decls, XStr& defs, int isVoid, XStr& signature) {
    // generate wrapper for local calls to the function
    if (needsParamMarshalling || isVoid) {
      templateGuardBegin(false, defs);
      defs << decl_entry->getContainer()->tspec() << " void " << decl_entry->getContainer()->baseName() << "::" << signature << "{\n";
      defs << "  " << *decl_entry->genClosureTypeNameProxyTemp << "*" <<
        " genClosure = new " << *decl_entry->genClosureTypeNameProxyTemp << "()" << ";\n";
      {
        int i = 0;
        for (list<CStateVar*>::iterator it = myParameters.begin(); it != myParameters.end(); ++it, ++i) {
          CStateVar& var = **it;
          if (i == 0 && *var.type == "int")
            defs << "  genClosure->__refnum" << " = " << var.name << ";\n";
          else if (i == 0)
            defs << "  genClosure->__refnum" << " = 0;\n";
          defs << "  genClosure->getP" << i << "() = " << var.name << ";\n";
        }
      }

      defs << "  " << *entry << "(genClosure);\n";
      defs << "  genClosure->deref();\n";
      defs << "}\n\n";
      templateGuardEnd(defs);
    }
  }

  void CEntry::generateCode(XStr& decls, XStr& defs) {
    CStateVar *sv;
    int i = 0;
    int isVoid = 1;
    int lastWasVoid;

    XStr signature;
    signature <<  *entry << "(";
    for(list<CStateVar*>::iterator it = myParameters.begin();
        it != myParameters.end(); ++it, ++i) {
      sv = *it;
      isVoid = sv->isVoid;
      if ((sv->isMsg != 1) && (sv->isVoid != 1)) {
        if (i >0)
          signature <<", ";
        if (sv->byConst)
          signature << "const ";
        signature << sv->type << " ";
        if (sv->arrayLength != 0)
          signature << "*";
        else if (sv->declaredRef) {
          signature <<"&";
        }
        if (sv->numPtrs != 0) {
          for(int k = 0; k< sv->numPtrs; k++)
	    signature << "*";
        }
        if (sv->name != 0)
          signature << sv->name;
      }
      else if (sv->isVoid != 1){
        if (i < 1) 
          signature << sv->type << "* " << sv->name << "_msg";
        else
          printf("ERROR: A message must be the only parameter in an entry function\n");
      }
      else
        signature <<"void";
    }

    signature << ")";

    XStr newSig;

    if (needsParamMarshalling || isVoid) {
      newSig << *entry << "(" << *decl_entry->genClosureTypeNameProxyTemp << "* genClosure)";
      decls << "  void " <<  newSig << ";\n";
      // generate local wrapper decls
      decls << "  void " <<  signature << ";\n";
    } else { // a message
      newSig << signature << "";
      decls << "  void " <<  newSig << ";\n";
    }

    generateLocalWrapper(decls, defs, isVoid, signature);

    templateGuardBegin(false, defs);
    defs << decl_entry->getContainer()->tspec() << " void " << decl_entry->getContainer()->baseName() << "::" << newSig << "{\n";
    defs << "  if (!__dep.get()) _sdag_init();\n";

    if (needsParamMarshalling || isVoid) {
      // add the incoming message to a buffer

      // note that there will always be a closure even when the method has no
      // parameters for consistency
      defs << "  __dep->pushBuffer(" << entryNum << ", genClosure, genClosure->__refnum);\n";
    } else {
      defs << "  int refnum = ";
      if (refNumNeeded)
        defs << "CkGetRefNum(" << sv->name << "_msg);\n";
      else
        defs << "0;\n";

      // possible memory pressure problem: this message will be kept as long as
      // it is a state parameter! there are ways to remediate this, but it
      // involves either live variable analysis (which is not feasible) or
      // keeping a meta-structure for every message passed in

      //increase reference count by one for the state parameter
      defs << "  CmiReference(UsrToEnv(" << sv->name << "_msg));\n";

      defs << "  __dep->pushBuffer(" << entryNum << ", new SDAG::MsgClosure(" << sv->name << "_msg" << "), refnum);\n";
    }
    // @todo write the code to fetch the message with the ref num

    // search for a continuation to restart execution
    defs << "  SDAG::Continuation* c = __dep->tryFindContinuation(" << entryNum << ");\n";

    // found a continuation
    defs << "  if (c) {\n";
    if (whenList.size() == 1) {
      //defs << "    {\n";
      (*whenList.begin())->generateWhenCodeNew(defs);
      //defs << "    }\n";
    } else {
      // switch on the possible entry points for the continuation
      // each continuation entry knows how to generate its own code
      defs << "    switch(c->whenID) {\n";
      for(list<WhenConstruct*>::iterator cn = whenList.begin(); cn != whenList.end(); ++cn) {
        defs << "    case " << (*cn)->nodeNum << ":\n";
        (*cn)->generateWhenCodeNew(defs);
        defs << "    break;\n";
      }
      defs << "    }\n";
    }

    // delete the continuation now that we are finished with it
    defs << "    delete c;\n";
    defs << "  } else {\n";

    SdagConstruct::generateTraceEndCall(defs);

    defs << "  }\n";

    defs << "}\n\n";
    templateGuardEnd(defs);
  }
}
