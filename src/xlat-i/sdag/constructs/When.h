#ifndef _WHEN_H
#define _WHEN_H

#include "xi-BlockConstruct.h"
#include "CParsedFile.h"

namespace xi {

class WhenConstruct : public BlockConstruct {
 public:
  // TODO(Ralf): Make this be private?
  CStateVar* speculativeState;
  WhenConstruct(EntryList *el, SdagConstruct *body);
  void generateCode(XStr& decls, XStr& defs, Entry *entry);
  void generateEntryList(std::list<CEntry*>& CEntrylist, WhenConstruct *thisWhen);
  void propagateState(std::list<EncapState*>, std::list<CStateVar*>&, std::list<CStateVar*>&, int);
  void generateEntryName(XStr& defs, Entry* e, int curEntry);
  void generateWhenCode(XStr& op, int indent);
  void numberNodes();
};

}   // namespace xi

#endif  // ifndef _WHEN_H
