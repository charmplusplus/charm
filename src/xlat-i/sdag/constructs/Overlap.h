#ifndef _OVERLAP_H
#define _OVERLAP_H

#include "CParsedFile.h"
#include "xi-BlockConstruct.h"

namespace xi {

class OverlapConstruct : public BlockConstruct {
 public:
  OverlapConstruct(SdagConstruct* olist);
  void generateCode(XStr&, XStr&, Entry*);
  void numberNodes();
};

}  // namespace xi

#endif  // ifndef _OVERLAP_H
