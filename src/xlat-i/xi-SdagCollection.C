#include "xi-SdagCollection.h"

namespace xi {

SdagCollection::SdagCollection(CParsedFile *p) : pf(p), sdagPresent(false) {}

void SdagCollection::addNode(Entry *e) {
  sdagPresent = true;
  pf->addNode(e);
}

}   // namespace xi
