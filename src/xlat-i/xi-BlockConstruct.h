#ifndef _XI_BLOCKCONSTRUCT_H
#define _XI_BLOCKCONSTRUCT_H

#include "xi-SdagConstruct.h"
#include "CStateVar.h"
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

}   // namespace xi

#endif  // ifndef _XI_BLOCKCONSTRUCT_H
