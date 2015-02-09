#ifndef _SDAG_COLLECTION_H
#define _SDAG_COLLECTION_H

#include "CParsedFile.h"

namespace xi {

struct SdagCollection
{
  CParsedFile *pf;
  bool sdagPresent;
  SdagCollection(CParsedFile *p);
  void addNode(Entry *e);
};

}   // namespace xi

#endif  // ifndef _SDAG_COLLECTION_H
