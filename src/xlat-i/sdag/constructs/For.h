#ifndef _FOR_H
#define _FOR_H

#include "xi-BlockConstruct.h"
#include "CParsedFile.h"

namespace xi {

class ForConstruct : public BlockConstruct {
 public:
  ForConstruct(SdagConstruct *decl, SdagConstruct *pred, SdagConstruct *advance, SdagConstruct *body);
  void generateCode(XStr&, XStr&, Entry *);
};

}   // namespace xi

#endif  // ifndef _FOR_H
