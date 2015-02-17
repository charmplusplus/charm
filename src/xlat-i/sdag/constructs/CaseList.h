#ifndef _CASELIST_H
#define _CASELIST_H

#include "xi-BlockConstruct.h"

namespace xi {

class CaseListConstruct : public SdagConstruct {
 public:
  CaseListConstruct(WhenConstruct *);
  CaseListConstruct(WhenConstruct *, CaseListConstruct *);
  void generateCode(XStr&, XStr&, Entry *);
  void propagateState(std::list<EncapState*>, std::list<CStateVar*>&, std::list<CStateVar*>&, int);
  void numberNodes();
};

}   // namespace xi

#endif  // ifndef _CASELIST_H
