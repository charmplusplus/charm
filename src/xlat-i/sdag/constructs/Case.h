#ifndef _CASE_H
#define _CASE_H

#include "CParsedFile.h"
#include "xi-BlockConstruct.h"

namespace xi {

class CaseConstruct : public BlockConstruct {
 public:
  CaseConstruct(SdagConstruct* body);
  void generateCode(XStr&, XStr&, Entry*);
  void numberNodes();
};

}  // namespace xi

#endif  // ifndef _CASE_H
