#include "xi-BlockConstruct.h"

namespace xi {

extern void RemoveSdagComments(char *str);

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

/***************** AtomicConstruct **************/
AtomicConstruct::AtomicConstruct(const char *code, const char *trace_name)
: BlockConstruct(SATOMIC, NULL, 0, 0, 0, 0, 0, 0)
{
  char *tmp = strdup(code);
  RemoveSdagComments(tmp);
  text = new XStr(tmp);
  free(tmp);

  if (trace_name)
  {
    tmp = strdup(trace_name);
    tmp[strlen(tmp)-1]=0;
    traceName = new XStr(tmp+1);
    free(tmp);
  }
}

void AtomicConstruct::propagateStateToChildren(std::list<EncapState*> encap,
                                               std::list<CStateVar*>& stateVarsChildren,
                                               std::list<CStateVar*>& wlist,
                                               int uniqueVarNum)
{}

/***************** WhenConstruct **************/
WhenConstruct::WhenConstruct(EntryList *el, SdagConstruct *body)
: BlockConstruct(SWHEN, 0, 0, 0, 0, 0, body, el)
, speculativeState(0)
{ }

/***************** WhileConstruct **************/
WhileConstruct::WhileConstruct(SdagConstruct *pred, SdagConstruct *body)
: BlockConstruct(SWHILE, 0, pred, 0, 0, 0, body, 0)
{ }

/***************** IfConstruct **************/
IfConstruct::IfConstruct(SdagConstruct *pred, SdagConstruct *then_body, SdagConstruct *else_body)
: BlockConstruct(SIF, 0, pred, else_body, 0, 0, then_body, 0)
{ }

/***************** ForConstruct **************/
ForConstruct::ForConstruct(SdagConstruct *decl, SdagConstruct *pred, SdagConstruct *advance, SdagConstruct *body)
: BlockConstruct(SFOR, 0, decl, pred, advance, 0, body, 0)
{ }

/***************** ForallConstruct **************/
ForallConstruct::ForallConstruct(SdagConstruct *tag, SdagConstruct *begin, SdagConstruct *end, SdagConstruct *step, SdagConstruct *body)
: BlockConstruct(SFORALL, 0, tag, begin, end, step, body, 0)
{ }

/***************** CaseConstruct **************/
CaseConstruct::CaseConstruct(SdagConstruct *body)
: BlockConstruct(SCASE, 0, 0, 0, 0, 0, body, 0)
{ }

/***************** OverlapConstruct **************/
OverlapConstruct::OverlapConstruct(SdagConstruct *olist)
: BlockConstruct(SOVERLAP, 0, 0, 0, 0, 0, olist, 0)
{ }


}   // namespace xi
