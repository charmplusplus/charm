#ifndef _SDAGENTRY_H
#define _SDAGENTRY_H

#include "xi-SdagConstruct.h"
#include "SList.h"

namespace xi {

class SdagEntryConstruct : public SdagConstruct {
 public:
  SdagEntryConstruct(SdagConstruct *);
  SdagEntryConstruct(SListConstruct *);
  void generateCode(XStr&, XStr&, Entry *);
  void numberNodes();
  void labelNodes();
};

}   // namespace xi

#endif  // ifndef _SDAGENTRY_H
