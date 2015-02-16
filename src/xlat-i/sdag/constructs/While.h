#ifndef _WHILE_H
#define _WHILE_H

#include "xi-BlockConstruct.h"

namespace xi {

class WhileConstruct : public BlockConstruct {
 public:
  WhileConstruct(SdagConstruct *pred, SdagConstruct *body);
  void generateCode(XStr&, XStr&, Entry *);
  void numberNodes();
};

}   // namespace xi

#endif  // ifndef _WHILE_H
