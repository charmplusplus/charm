#include "SList.h"

namespace xi {

SListConstruct::SListConstruct(SdagConstruct *single_construct)
: SdagConstruct(SSLIST, single_construct)
{
  label_str = "slist";
}

SListConstruct::SListConstruct(SdagConstruct *single_construct, SListConstruct *tail)
: SdagConstruct(SSLIST, single_construct, tail)
{
  label_str = "slist";
}

void SListConstruct::generateCode(XStr& decls, XStr& defs, Entry* entry) {
  buildTypes(encapState);
  buildTypes(encapStateChild);

  generateClosureSignature(decls, defs, entry, false, "void", label, false, encapState);
  defs << "  ";
  generateCall(defs, encapState, encapState, constructs->front()->label);
  endMethod(defs);

  generateClosureSignature(decls, defs, entry, false, "void", label, true, encapStateChild);
  defs << "  ";
  generateCall(defs, encapState, encapStateChild, next->label, nextBeginOrEnd ? 0 : "_end");
  endMethod(defs);

  generateChildrenCode(decls, defs, entry);
}

void SListConstruct::numberNodes() {
  nodeNum = numSlists++;
  SdagConstruct::numberNodes();
}

void SListConstruct::propagateState(std::list<EncapState*> encap,
                                    std::list<CStateVar*>& plist,
                                    std::list<CStateVar*>& wlist,
                                    int uniqueVarNum) {
  stateVars = new std::list<CStateVar*>();
  stateVars->insert(stateVars->end(), plist.begin(), plist.end());
  stateVarsChildren = stateVars;

  encapState = encap;
  encapStateChild = encap;

  propagateStateToChildren(encap, *stateVarsChildren, wlist, uniqueVarNum);
}

}   // namespace xi
