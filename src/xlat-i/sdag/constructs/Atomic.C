#include "Atomic.h"

namespace xi {

extern void RemoveSdagComments(char *str);

AtomicConstruct::AtomicConstruct(const char *code, const char *trace_name)
: BlockConstruct(SATOMIC, NULL, 0, 0, 0, 0, 0, 0)
{
  char *tmp = strdup(code);
  RemoveSdagComments(tmp);
  text = new XStr(tmp);
  free(tmp);

  if (trace_name)
  {
    tmp = strdup(trace_name);
    tmp[strlen(tmp)-1]=0;
    traceName = new XStr(tmp+1);
    free(tmp);
  }

  label_str = "atomic";
}

void AtomicConstruct::propagateStateToChildren(std::list<EncapState*> encap,
                                               std::list<CStateVar*>& stateVarsChildren,
                                               std::list<CStateVar*>& wlist,
                                               int uniqueVarNum)
{}

void AtomicConstruct::generateCode(XStr& decls, XStr& defs, Entry* entry) {
  generateClosureSignature(decls, defs, entry, false, "void", label, false, encapState);

#if CMK_BIGSIM_CHARM
  sprintf(nameStr,"%s%s", CParsedFile::className->charstar(),label->charstar());
  generateBeginExec(defs, nameStr);
#endif

  generateTraceBeginCall(defs, 1);

  char* str = text->get_string();
  bool hasCode = false;

  while (*str != '\0') {
    if (*str != '\n' && *str != ' ' && *str != '\t') {
      hasCode = true;
      break;
    }
    str++;
  }

  if (hasCode) {
    int indent = unravelClosuresBegin(defs);

    indentBy(defs, indent);
    defs << "{ // begin serial block\n";
    defs << text << "\n";
    indentBy(defs, indent);
    defs << "} // end serial block\n";

    unravelClosuresEnd(defs);
  }

  generateTraceEndCall(defs, 1);

#if CMK_BIGSIM_CHARM
  generateEndExec(defs);
#endif

  indentBy(defs, 1);
  generateCall(defs, encapState, encapState, next->label, nextBeginOrEnd ? 0 : "_end");
  endMethod(defs);
}

void AtomicConstruct::generateTrace() {
  char traceText[1024];
  if (traceName) {
    sprintf(traceText, "%s_%s", CParsedFile::className->charstar(), traceName->charstar());
    // remove blanks
    for (unsigned int i=0; i<strlen(traceText); i++)
      if (traceText[i]==' '||traceText[i]=='\t') traceText[i]='_';
  }
  else {
    sprintf(traceText, "%s%s", CParsedFile::className->charstar(), label->charstar());
  }
  traceName = new XStr(traceText);

  if (con1) con1->generateTrace();
}

void AtomicConstruct::numberNodes(void) {
  nodeNum = numAtomics++;
  SdagConstruct::numberNodes();
}

}   // namespace xi
