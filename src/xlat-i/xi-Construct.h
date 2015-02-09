#ifndef _CONSTRUCT_H
#define _CONSTRUCT_H

#include "xi-AstNode.h"

namespace xi { 

class Module;

class Construct : public virtual AstNode {
 protected:
  int external;

 public:
  // FIXME?: isn't this circular since Module < Construct?
  Module *containerModule;
  explicit Construct();
  void setExtern(int& e);
  void setModule(Module *m);
};

// FIXME?: shouldn't the "public virtual" be here instead of in the Construct baseclass?
class ConstructList : public AstChildren<Construct>, public Construct {
 public:
  ConstructList(int l, Construct *c, ConstructList *n=0);
};

/******************** AccelBlock : Block of code for accelerator **********************/
class AccelBlock : public Construct {
 protected:
  XStr* code;

 private:
  void outputCode(XStr& str);

 public:
  /// Constructor(s)/Destructor ///
  AccelBlock(int l, XStr* c);
  ~AccelBlock();

  /// Printable Methods ///
  void print(XStr& str);

  /// Construct Methods ///
  void genDefs(XStr& str);

  /// Construct Accel Support Methods ///
  int genAccels_spe_c_funcBodies(XStr& str);
};

} // namespace xi

#endif // ifndef _CONSTRUCT_H
