#ifndef _INTEXPR_H
#define _INTEXPR_H

#include "xi-SdagConstruct.h"

namespace xi {

class IntExprConstruct : public SdagConstruct {
 public:
  explicit IntExprConstruct(const char *ccode);
};

}   // namespace xi

#endif  // ifndef _INTEXPR_H
