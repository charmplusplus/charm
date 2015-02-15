#include "xi-BlockConstruct.h"

namespace xi {

// TODO(Ralf): Find out if this is a good way to propagate the constructor elements.
BlockConstruct::BlockConstruct(EToken t, XStr *txt,
                               SdagConstruct *c1, SdagConstruct *c2,
                               SdagConstruct *c3, SdagConstruct *c4,
                               SdagConstruct *constructAppend, EntryList *el)
: SdagConstruct(t, txt, c1, c2, c3, c4, constructAppend, el)
{}

void BlockConstruct::propagateState(std::list<EncapState*> encap,
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
