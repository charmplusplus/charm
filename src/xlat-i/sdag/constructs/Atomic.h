#ifndef _ATOMIC_H
#define _ATOMIC_H

#include "xi-BlockConstruct.h"
#include "CParsedFile.h"

namespace xi {

class AtomicConstruct : public BlockConstruct {
 public:
  AtomicConstruct(const char *code, const char *trace_name);
  void propagateStateToChildren(std::list<EncapState*> encap,
                                std::list<CStateVar*>& stateVarsChildren,
                                std::list<CStateVar*>& wlist,
                                int uniqueVarNum);
  void generateCode(XStr&, XStr&, Entry *);
  void generateTrace();
  void numberNodes();
};

}   // namespace xi

#endif  // ifndef _ATOMIC_H
