#ifndef _SLIST_H
#define _SLIST_H

#include "xi-SdagConstruct.h"

namespace xi {

class SListConstruct : public SdagConstruct {
 public:
  SListConstruct(SdagConstruct *);
  SListConstruct(SdagConstruct *, SListConstruct *);
  void generateCode(XStr&, XStr&, Entry *);
  void propagateState(std::list<EncapState*>, std::list<CStateVar*>&, std::list<CStateVar*>&, int);
  void numberNodes();
};

}   // namespace xi

#endif  // ifndef _SLIST_H
