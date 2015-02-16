#ifndef _OVERLAP_H
#define _OVERLAP_H

#include "xi-BlockConstruct.h"
#include "CParsedFile.h"

namespace xi {

class OverlapConstruct : public BlockConstruct {
 public:
  OverlapConstruct(SdagConstruct *olist);
  void generateCode(XStr&, XStr&, Entry *);
  void numberNodes();
};

}   // namespace xi

#endif  // ifndef _OVERLAP_H
