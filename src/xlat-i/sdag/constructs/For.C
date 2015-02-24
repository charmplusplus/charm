#include "For.h"
#include "IntExpr.h"

namespace xi {

ForConstruct::ForConstruct(IntExprConstruct *decl,
                           IntExprConstruct *pred,
                           IntExprConstruct *advance,
                           SdagConstruct *body)
: BlockConstruct(SFOR, 0, decl, pred, advance, 0, body, 0)
{
  label_str = "for";
}

  void ForConstruct::generateCode(XStr& decls, XStr& defs, Entry* entry) {
    sprintf(nameStr,"%s%s", CParsedFile::className->charstar(),label->charstar());

    generateClosureSignature(decls, defs, entry, false, "void", label, false, encapState);
#if CMK_BIGSIM_CHARM
    generateBeginTime(defs);
#endif

    int indent = unravelClosuresBegin(defs);

    indentBy(defs, indent);
    defs << con1->text << ";\n";
    //Record only the beginning for FOR
#if CMK_BIGSIM_CHARM
    generateEventBracket(defs, SFOR);
#endif
    indentBy(defs, indent);
    defs << "if (" << con2->text << ") {\n";
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

    // trace
    sprintf(nameStr,"%s%s", CParsedFile::className->charstar(),label->charstar());
    strcat(nameStr,"_end");

    generateClosureSignature(decls, defs, entry, false, "void", label, true, encapStateChild);
#if CMK_BIGSIM_CHARM
    generateBeginTime(defs);
#endif
    indent = unravelClosuresBegin(defs);

    indentBy(defs, indent);
    defs << con3->text << ";\n";
    indentBy(defs, indent);
    defs << "if (" << con2->text << ") {\n";
    indentBy(defs, indent + 1);
    generateCall(defs, encapStateChild, encapStateChild, constructs->front()->label);
    indentBy(defs, indent);
    defs << "} else {\n";
#if CMK_BIGSIM_CHARM
    generateEventBracket(defs, SFOR_END);
#endif
    indentBy(defs, indent + 1);
    generateCall(defs, encapState, encapState, next->label, nextBeginOrEnd ? 0 : "_end");
    indentBy(defs, indent);
    defs << "}\n";

    unravelClosuresEnd(defs);

    endMethod(defs);

    generateChildrenCode(decls, defs, entry);
  }

void ForConstruct::numberNodes() {
  nodeNum = numFors++;
  SdagConstruct::numberNodes();
}

}   // namespace xi
