#ifndef _AST_NODE_H
#define _AST_NODE_H

#include "xi-util.h"
#include <list>

namespace xi {

class AstNode : public Printable {
 protected:
  int line;

 public:
  explicit AstNode(int line_ = -1) : line(line_) { }
  virtual void outputClosuresDecl(XStr& str) { (void)str; }
  virtual void outputClosuresDef(XStr& str) { (void)str; }
  virtual void genDecls(XStr& str) { (void)str; }
  virtual void genDefs(XStr& str) { (void)str; }
  virtual void genClosureEntryDecls(XStr& str) { }
  virtual void genClosureEntryDefs(XStr& str) { }
  virtual void genReg(XStr& str) { (void)str; }
  virtual void genGlobalCode(XStr scope, XStr &decls, XStr &defs)
  { (void)scope; (void)decls; (void)defs; }
  virtual void preprocess() { }
  virtual void check() { }
  virtual void printChareNames() { }

  // DMK - Accel Support
  virtual int genAccels_spe_c_funcBodies(XStr& str) { (void)str; return 0; }
  virtual void genAccels_spe_c_regFuncs(XStr& str) { (void)str; }
  virtual void genAccels_spe_c_callInits(XStr& str) { (void)str; }
  virtual void genAccels_spe_h_includes(XStr& str) { (void)str; }
  virtual void genAccels_spe_h_fiCountDefs(XStr& str) { (void)str; }
  virtual void genAccels_ppe_c_regFuncs(XStr& str) { (void)str; }
};

template <typename Child>
class AstChildren : public virtual AstNode {
 protected:
  std::list<Child*> children;

 public:
  AstChildren(int line_, Child *c, AstChildren *cs)
    : AstNode(line_)
  {
    children.push_back(c);
    if (cs)
      children.insert(children.end(), cs->children.begin(), cs->children.end());
  }

  template <typename T>
  explicit AstChildren(std::list<T*> &l)
  {
    children.insert(children.begin(), l.begin(), l.end());
    l.clear();
  }
  void push_back(Child *c);

  void preprocess();
  void check();
  void print(XStr& str);

  void printChareNames();

  void outputClosuresDecl(XStr& str);
  void outputClosuresDef(XStr& str);

  void genClosureEntryDecls(XStr& str);
  void genClosureEntryDefs(XStr& str);
  void genDecls(XStr& str);
  void genDefs(XStr& str);
  void genReg(XStr& str);
  void genGlobalCode(XStr scope, XStr &decls, XStr &defs);

  // Accelerated Entry Method support
  int genAccels_spe_c_funcBodies(XStr& str);
  void genAccels_spe_c_regFuncs(XStr& str);
  void genAccels_spe_c_callInits(XStr& str);
  void genAccels_spe_h_includes(XStr& str);
  void genAccels_spe_h_fiCountDefs(XStr& str);
  void genAccels_ppe_c_regFuncs(XStr& str);

  template <typename T>
  void recurse(T arg, void (Child::*fn)(T));
  void recursev(void (Child::*fn)());
};

} // namespace xi

#endif // ifndef _AST_NODE_H
