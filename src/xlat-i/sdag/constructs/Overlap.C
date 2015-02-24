#include "Overlap.h"

namespace xi {

OverlapConstruct::OverlapConstruct(SdagConstruct *olist)
: BlockConstruct(SOVERLAP, 0, 0, 0, 0, 0, olist, 0)
{
  label_str = "overlap";
}

void OverlapConstruct::generateCode(XStr& decls, XStr& defs, Entry* entry) {
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

void OverlapConstruct::numberNodes(void) {
  nodeNum = numOverlaps++;
  SdagConstruct::numberNodes();
}

}   // namespace xi
