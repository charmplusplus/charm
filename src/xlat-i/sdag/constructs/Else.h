#ifndef _ELSE_H
#define _ELSE_H

#include "xi-BlockConstruct.h"

namespace xi {

class ElseConstruct : public BlockConstruct {
 public:
  ElseConstruct(SdagConstruct *body);
  void generateCode(XStr&, XStr&, Entry *);
  void numberNodes();
};

}   // namespace xi

#endif  // ifndef _ELSE_H
