#ifndef _XI_BLOCKCONSTRUCT_H
#define _XI_BLOCKCONSTRUCT_H

#include "xi-SdagConstruct.h"
#include <list>

namespace xi {

class BlockConstruct : public SdagConstruct {
 public:
  BlockConstruct(EToken t, XStr *txt, SdagConstruct *c1, SdagConstruct *c2, SdagConstruct *c3,
                 SdagConstruct *c4, SdagConstruct *constructAppend, EntryList *el);
  void propagateState(std::list<EncapState*> encap,
                      std::list<CStateVar*>& plist,
                      std::list<CStateVar*>& wlist,
                      int uniqueVarNum);
};

/***************** AtomicConstruct **************/
class AtomicConstruct : public BlockConstruct {
 public:
  AtomicConstruct(const char *code, const char *trace_name);
  void propagateStateToChildren(std::list<EncapState*> encap,
                                std::list<CStateVar*>& stateVarsChildren,
                                std::list<CStateVar*>& wlist,
                                int uniqueVarNum);
  void generateCode(XStr&, XStr&, Entry *);
  void generateTrace();
};

/***************** WhenConstruct **************/
class WhenConstruct : public BlockConstruct {
 public:
  // TODO(Ralf): Make this be private?
  CStateVar* speculativeState;
  WhenConstruct(EntryList *el, SdagConstruct *body);
  void generateCode(XStr& decls, XStr& defs, Entry *entry);
  void generateEntryList(std::list<CEntry*>& CEntrylist, WhenConstruct *thisWhen);
  void propagateState(std::list<EncapState*>, std::list<CStateVar*>&, std::list<CStateVar*>&, int);
  void generateEntryName(XStr& defs, Entry* e, int curEntry);
  void generateWhenCode(XStr& op, int indent);
};


/***************** WhileConstruct **************/
class WhileConstruct : public BlockConstruct {
 public:
  WhileConstruct(SdagConstruct *pred, SdagConstruct *body);
  void generateCode(XStr&, XStr&, Entry *);
};

/***************** IfConstruct **************/
class IfConstruct : public BlockConstruct {
 public:
  IfConstruct(SdagConstruct *pred, SdagConstruct *then_body, SdagConstruct *else_body);
  void propagateState(std::list<EncapState*>, std::list<CStateVar*>&, std::list<CStateVar*>&, int);
  void generateCode(XStr&, XStr&, Entry *);
};

/***************** ForConstruct **************/
class ForConstruct : public BlockConstruct {
 public:
  ForConstruct(SdagConstruct *decl, SdagConstruct *pred, SdagConstruct *advance, SdagConstruct *body);
  void generateCode(XStr&, XStr&, Entry *);
};

/***************** ForallConstruct **************/
class ForallConstruct : public BlockConstruct {
 public:
  ForallConstruct(SdagConstruct *tag, SdagConstruct *begin, SdagConstruct *end, SdagConstruct *step, SdagConstruct *body);
  void propagateState(std::list<EncapState*>, std::list<CStateVar*>&, std::list<CStateVar*>&, int);
  void generateCode(XStr&, XStr&, Entry *);
};

/***************** CaseConstruct **************/
class CaseConstruct : public BlockConstruct {
 public:
  CaseConstruct(SdagConstruct *body);
  void generateCode(XStr&, XStr&, Entry *);
};

/***************** OverlapConstruct **************/
class OverlapConstruct : public BlockConstruct {
 public:
  OverlapConstruct(SdagConstruct *olist);
  void generateCode(XStr&, XStr&, Entry *);
};


}   // namespace xi

#endif  // ifndef _XI_BLOCKCONSTRUCT_H
