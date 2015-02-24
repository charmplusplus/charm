#include "If.h"

using std::list;

namespace xi {

IfConstruct::IfConstruct(SdagConstruct *pred, SdagConstruct *then_body, SdagConstruct *else_body)
: BlockConstruct(SIF, 0, pred, else_body, 0, 0, then_body, 0)
{
  label_str = "if";
}

  void IfConstruct::generateCode(XStr& decls, XStr& defs, Entry* entry) {
    strcpy(nameStr,label->charstar());
    generateClosureSignature(decls, defs, entry, false, "void", label, false, encapState);

#if CMK_BIGSIM_CHARM
    generateBeginTime(defs);
    generateEventBracket(defs, SIF);
#endif

    int indent = unravelClosuresBegin(defs);

    indentBy(defs, indent);
    defs << "if (" << con1->text << ") {\n";
    indentBy(defs, indent + 1);
    generateCall(defs, encapStateChild, encapStateChild, constructs->front()->label);
    indentBy(defs, indent);
    defs << "} else {\n";
    indentBy(defs, indent + 1);
    if (con2 != 0)
      generateCall(defs, encapStateChild, encapStateChild, con2->label);
    else
      generateCall(defs, encapStateChild, encapStateChild, label, "_end");
    indentBy(defs, indent);
    defs << "}\n";

    unravelClosuresEnd(defs);

    endMethod(defs);

    strcpy(nameStr,label->charstar());
    strcat(nameStr,"_end");
    generateClosureSignature(decls, defs, entry, false, "void", label, true, encapStateChild);
#if CMK_BIGSIM_CHARM
    generateBeginTime(defs);
    generateEventBracket(defs,SIF_END);
#endif
    indentBy(defs, 1);
    generateCall(defs, encapState, encapState, next->label, nextBeginOrEnd ? 0 : "_end");
    endMethod(defs);

    if (con2 != 0) con2->generateCode(decls, defs, entry);

    generateChildrenCode(decls, defs, entry);
  }

  void IfConstruct::propagateState(list<EncapState*> encap, list<CStateVar*>& plist, list<CStateVar*>& wlist,  int uniqueVarNum) {
    BlockConstruct::propagateState(encap, plist, wlist, uniqueVarNum);
    if(con2 != 0) con2->propagateState(encap, plist, wlist, uniqueVarNum);
  }

void IfConstruct::numberNodes(void) {
  nodeNum = numIfs++;
  if (con2 != 0) con2->numberNodes();
  SdagConstruct::numberNodes();
}

void IfConstruct::labelNodes() {
  SdagConstruct::labelNodes();
  if (con2 != 0) con2->labelNodes();
}

}   // namespace xi
