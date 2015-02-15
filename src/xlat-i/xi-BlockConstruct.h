#ifndef _XI_BLOCKCONSTRUCT_H
#define _XI_BLOCKCONSTRUCT_H

#include "xi-SdagConstruct.h"
#include "CStateVar.h"
#include <list>

namespace xi {

// TODO(Ralf): Find a good place to put these functions,
//             which are used by all constructs.
extern void generateClosureSignature(XStr& decls, XStr& defs,
                                     const Chare* chare, bool declareStatic,
                                     const char* returnType, const XStr* name,
                                     bool isEnd, std::list<EncapState*> params,
                                     int numRefs = 0);
extern void generateClosureSignature(XStr& decls, XStr& defs,
                                     const Entry* entry, bool declareStatic,
                                     const char* returnType, const XStr* name,
                                     bool isEnd, std::list<EncapState*> params,
                                     int numRefs = 0);
extern void endMethod(XStr& op);

class BlockConstruct : public SdagConstruct {
 public:
  BlockConstruct(EToken t, XStr *txt, SdagConstruct *c1, SdagConstruct *c2, SdagConstruct *c3,
                 SdagConstruct *c4, SdagConstruct *constructAppend, EntryList *el);
  void propagateState(std::list<EncapState*> encap,
                      std::list<CStateVar*>& plist,
                      std::list<CStateVar*>& wlist,
                      int uniqueVarNum);
};

}   // namespace xi

#endif  // ifndef _XI_BLOCKCONSTRUCT_H
