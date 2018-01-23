#ifndef _OLIST_H
#define _OLIST_H

#include "CStateVar.h"
#include "SList.h"
#include "xi-SdagConstruct.h"

namespace xi {

class OListConstruct : public SdagConstruct {
 public:
  OListConstruct(SdagConstruct*);
  OListConstruct(SdagConstruct*, SListConstruct*);
  void generateCode(XStr&, XStr&, Entry*);
  void propagateState(std::list<EncapState*>, std::list<CStateVar*>&,
                      std::list<CStateVar*>&, int);
  void numberNodes();
};

}  // namespace xi

#endif  // ifndef _OLIST_H
