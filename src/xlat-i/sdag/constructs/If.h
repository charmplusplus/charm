#ifndef _IF_H
#define _IF_H

#include "xi-BlockConstruct.h"

namespace xi {

class IfConstruct : public BlockConstruct {
 public:
  IfConstruct(SdagConstruct *pred, SdagConstruct *then_body, SdagConstruct *else_body);
  void propagateState(std::list<EncapState*>, std::list<CStateVar*>&, std::list<CStateVar*>&, int);
  void generateCode(XStr&, XStr&, Entry *);
  void numberNodes();
  void labelNodes();
};

}   // namespace xi

#endif  // ifndef _IF_H
