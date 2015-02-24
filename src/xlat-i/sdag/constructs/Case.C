#include "Case.h"

namespace xi {

CaseConstruct::CaseConstruct(SdagConstruct *body)
: BlockConstruct(SCASE, 0, 0, 0, 0, 0, body, 0)
{
  label_str = "case";
}

  void CaseConstruct::generateCode(XStr& decls, XStr& defs, Entry* entry) {
    sprintf(nameStr,"%s%s", CParsedFile::className->charstar(),label->charstar());
    generateClosureSignature(decls, defs, entry, false, "void", label, false, encapState);
#if CMK_BIGSIM_CHARM
    generateBeginTime(defs);
    generateEventBracket(defs, SOVERLAP);
#endif
    defs << "  ";
    generateCall(defs, encapStateChild, encapStateChild, constructs->front()->label);
    endMethod(defs);

    // trace
    sprintf(nameStr,"%s%s", CParsedFile::className->charstar(),label->charstar());
    strcat(nameStr,"_end");
    generateClosureSignature(decls, defs, entry, false, "void", label, true, encapStateChild);
#if CMK_BIGSIM_CHARM
    generateBeginTime(defs);
    generateEventBracket(defs, SOVERLAP_END);
#endif
    defs << "  ";
    generateCall(defs, encapState, encapState, next->label, nextBeginOrEnd ? 0 : "_end");
    endMethod(defs);

    generateChildrenCode(decls, defs, entry);
  }

void CaseConstruct::numberNodes(void) {
  nodeNum = numCases++;
  SdagConstruct::numberNodes();
}

}   // namespace xi
