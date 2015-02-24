#include "Forall.h"
#include "IntExpr.h"

namespace xi {

ForallConstruct::ForallConstruct(SdagConstruct *tag,
                                 IntExprConstruct *begin,
                                 IntExprConstruct *end,
                                 IntExprConstruct *step,
                                 SdagConstruct *body)
: BlockConstruct(SFORALL, 0, tag, begin, end, step, body, 0)
{
  label_str = "forall";
}

void ForallConstruct::generateCode(XStr& decls, XStr& defs, Entry* entry) {
  generateClosureSignature(decls, defs, entry, false, "void", label, false, encapState);

  unravelClosuresBegin(defs);

  defs << "  int __first = (" << con2->text << "), __last = (" << con3->text
       << "), __stride = (" << con4->text << ");\n";
  defs << "  if (__first > __last) {\n";
  defs << "    int __tmp=__first; __first=__last; __last=__tmp;\n";
  defs << "    __stride = -__stride;\n";
  defs << "  }\n";
  defs << "  SDAG::CCounter *" << counter << " = new SDAG::CCounter(__first, __last, __stride);\n";
  defs << "  for(int " << con1->text << "=__first;" << con1->text << "<=__last;"
       << con1->text << "+=__stride) {\n";
  defs << "    SDAG::ForallClosure* " << con1->text << "_cl = new SDAG::ForallClosure(" << con1->text << ");\n";
  defs << "    ";
  generateCall(defs, encapStateChild, encapStateChild, constructs->front()->label);

  defs << "  }\n";

  unravelClosuresEnd(defs);

  endMethod(defs);

  generateClosureSignature(decls, defs, entry, false, "void", label, true, encapStateChild);
  defs << "  " << counter << "->decrement(); /* DECREMENT 1 */ \n";
  defs << "  " << con1->text << "_cl->deref();\n";
  defs << "  if (" << counter << "->isDone()) {\n";
  defs << "    " << counter << "->deref();\n";
  defs << "    ";
  generateCall(defs, encapState, encapState, next->label, nextBeginOrEnd ? 0 : "_end");
  defs << "  }\n";
  endMethod(defs);

  generateChildrenCode(decls, defs, entry);
}

void ForallConstruct::propagateState(std::list<EncapState*> encap,
                                     std::list<CStateVar*>& plist,
                                     std::list<CStateVar*>& wlist,
                                     int uniqueVarNum) {
  encapState = encap;

  stateVars = new std::list<CStateVar*>();
  stateVars->insert(stateVars->end(), plist.begin(), plist.end());
  stateVarsChildren = new std::list<CStateVar*>(plist);

  CStateVar *sv = new CStateVar(0,"int", 0, con1->text->charstar(), 0,NULL, 0);
  stateVarsChildren->push_back(sv);

  {
    std::list<CStateVar*> lst;
    lst.push_back(sv);
    EncapState *state = new EncapState(NULL, lst);
    state->isForall = true;
    state->type = new XStr("SDAG::ForallClosure");
    XStr* name = new XStr();
    *name << con1->text << "_cl";
    state->name = name;
    encap.push_back(state);
  }

  {
    char txt[128];
    sprintf(txt, "_cf%d", nodeNum);
    counter = new XStr(txt);
    sv = new CStateVar(0, "SDAG::CCounter *", 0, txt, 0, NULL, 1);
    sv->isCounter = true;
    stateVarsChildren->push_back(sv);

    std::list<CStateVar*> lst;
    lst.push_back(sv);
    EncapState *state = new EncapState(NULL, lst);
    state->type = new XStr("SDAG::CCounter");
    state->name = new XStr(txt);
    encap.push_back(state);
  }

  encapStateChild = encap;

  propagateStateToChildren(encap, *stateVarsChildren, wlist, uniqueVarNum);
}

void ForallConstruct::numberNodes(void) {
  nodeNum = numForalls++;
  SdagConstruct::numberNodes();
}

}   // namespace xi
