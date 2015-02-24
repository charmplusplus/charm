#ifndef _FOR_H
#define _FOR_H

#include "xi-BlockConstruct.h"
#include "CParsedFile.h"

namespace xi {

class IntExprConstruct;

class ForConstruct : public BlockConstruct {
 public:
  ForConstruct(IntExprConstruct *decl,
               IntExprConstruct *pred,
               IntExprConstruct *advance,
               SdagConstruct *body);
  void generateCode(XStr&, XStr&, Entry *);
  void numberNodes();
};

}   // namespace xi

#endif  // ifndef _FOR_H
