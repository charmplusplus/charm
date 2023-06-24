#include "Overlap.h"

namespace xi {

OverlapConstruct::OverlapConstruct(SdagConstruct* olist)
    : BlockConstruct(SOVERLAP, 0, 0, 0, 0, 0, olist, 0) {
  label_str = "overlap";
}

void OverlapConstruct::generateCode(XStr& decls, XStr& defs, Entry* entry) {
  snprintf(nameStr, sizeof(nameStr), "%s%s", CParsedFile::className->charstar(), label->charstar());
  generateClosureSignature(decls, defs, entry, false, "void", label, false, encapState);
  defs << "  ";
  generateCall(defs, encapStateChild, encapStateChild, constructs->front()->label);
  endMethod(defs);

  // trace
  snprintf(nameStr, sizeof(nameStr), "%s%s", CParsedFile::className->charstar(), label->charstar());
  strcat(nameStr, "_end");
  generateClosureSignature(decls, defs, entry, false, "void", label, true,
                           encapStateChild);
  defs << "  ";
  generateCall(defs, encapState, encapState, next->label, nextBeginOrEnd ? 0 : "_end");
  endMethod(defs);

  generateChildrenCode(decls, defs, entry);
}

void OverlapConstruct::numberNodes(void) {
  nodeNum = numOverlaps++;
  SdagConstruct::numberNodes();
}

}  // namespace xi
