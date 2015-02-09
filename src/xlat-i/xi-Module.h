#ifndef _MODULE_H
#define _MODULE_H

#include "xi-Construct.h"

namespace xi {

class Module : public Construct {
  int _isMain;
  const char *name;
  ConstructList *clist;

 public:
  Module(int l, const char *n, ConstructList *c);
  void print(XStr& str);
  void printChareNames() { if (clist) clist->printChareNames(); }
  void check();
  void generate();
  void setModule();
  void prependConstruct(Construct *c) { clist = new ConstructList(-1, c, clist); }
  void preprocess();
  void genDepend(const char *cifile);
  void genDecls(XStr& str);
  void genDefs(XStr& str);
  void genReg(XStr& str);
  void setMain(void) { _isMain = 1; }
  int isMain(void) { return _isMain; }

  // DMK - Accel Support
  int genAccels_spe_c_funcBodies(XStr& str);
  void genAccels_spe_c_regFuncs(XStr& str);
  void genAccels_spe_c_callInits(XStr& str);
  void genAccels_spe_h_includes(XStr& str);
  void genAccels_spe_h_fiCountDefs(XStr& str);
  void genAccels_ppe_c_regFuncs(XStr& str);
};

} // namespace xi

#endif  // ifndef _MODULE_H
