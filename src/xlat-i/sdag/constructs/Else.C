#include "Else.h"
#include "CParsedFile.h"

namespace xi {

ElseConstruct::ElseConstruct(SdagConstruct *body)
: BlockConstruct(SELSE, 0, 0, 0, 0, 0, body, 0)
{
  label_str = "else";
}

void ElseConstruct::generateCode(XStr& decls, XStr& defs, Entry* entry) {
  strcpy(nameStr,label->charstar());
  generateClosureSignature(decls, defs, entry, false, "void", label, false, encapState);
#if CMK_BIGSIM_CHARM
  // trace
  generateBeginTime(defs);
  generateEventBracket(defs, SELSE);
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
  generateEventBracket(defs,SELSE_END);
#endif
  defs << "  ";
  generateCall(defs, encapState, encapState, next->label, nextBeginOrEnd ? 0 : "_end");
  endMethod(defs);

  generateChildrenCode(decls, defs, entry);
}

void ElseConstruct::numberNodes(void) {
  nodeNum = numElses++;
  SdagConstruct::numberNodes();
}

}   // namespace xi
