#ifndef _FORALL_H
#define _FORALL_H

#include "xi-BlockConstruct.h"

namespace xi {

class ForallConstruct : public BlockConstruct {
 public:
  ForallConstruct(SdagConstruct *tag, SdagConstruct *begin, SdagConstruct *end, SdagConstruct *step, SdagConstruct *body);
  void propagateState(std::list<EncapState*>, std::list<CStateVar*>&, std::list<CStateVar*>&, int);
  void generateCode(XStr&, XStr&, Entry *);
  void numberNodes();
};

}   // namespace xi

#endif  // ifndef _FORALL_H
