#include "While.h"

namespace xi {

WhileConstruct::WhileConstruct(SdagConstruct *pred, SdagConstruct *body)
: BlockConstruct(SWHILE, 0, pred, 0, 0, 0, body, 0)
{
  label_str = "while";
}

  void WhileConstruct::generateCode(XStr& decls, XStr& defs, Entry* entry) {
    generateClosureSignature(decls, defs, entry, false, "void", label, false, encapState);

    int indent = unravelClosuresBegin(defs);
    indentBy(defs, indent);
    defs << "if (" << con1->text << ") {\n";
    indentBy(defs, indent + 1);
    generateCall(defs, encapStateChild, encapStateChild, constructs->front()->label);
    indentBy(defs, indent);
    defs << "} else {\n";
    indentBy(defs, indent + 1);
    generateCall(defs, encapState, encapState, next->label, nextBeginOrEnd ? 0 : "_end");
    indentBy(defs, indent);
    defs << "}\n";
    unravelClosuresEnd(defs);
    endMethod(defs);

    generateClosureSignature(decls, defs, entry, false, "void", label, true, encapStateChild);
    indent = unravelClosuresBegin(defs);
    indentBy(defs, indent);
    defs << "if (" << con1->text << ") {\n";
    indentBy(defs, indent + 1);
    generateCall(defs, encapStateChild, encapStateChild, constructs->front()->label);
    indentBy(defs, indent);
    defs << "} else {\n";
    indentBy(defs, indent + 1);
    generateCall(defs, encapState, encapState, next->label, nextBeginOrEnd ? 0 : "_end");
    indentBy(defs, indent);
    defs << "}\n";
    unravelClosuresEnd(defs);
    endMethod(defs);

    generateChildrenCode(decls, defs, entry);
  }

void WhileConstruct::numberNodes(void) {
  nodeNum = numWhiles++;
  SdagConstruct::numberNodes();
}

}   // namespace xi
