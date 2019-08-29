#ifndef _SERIAL_H
#define _SERIAL_H

#include "CParsedFile.h"
#include "xi-BlockConstruct.h"

namespace xi {

class SerialConstruct : public BlockConstruct {
 public:
  SerialConstruct(const char* code, const char* trace_name, int line_no);
  void propagateStateToChildren(std::list<EncapState*> encap,
                                std::list<CStateVar*>& stateVarsChildren,
                                std::list<CStateVar*>& wlist, int uniqueVarNum);
  void generateCode(XStr&, XStr&, Entry*);
  void generateTrace();
  void numberNodes();

 private:
  int line_no_;
};

}  // namespace xi

#endif  // ifndef _SERIAL_H
