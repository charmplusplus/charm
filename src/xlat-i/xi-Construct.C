#include "xi-Construct.h"

namespace xi {

Construct::Construct() : external(0) {}
void Construct::setExtern(int& e) { external = e; }
void Construct::setModule(Module *m) { containerModule = m; }

ConstructList::ConstructList(int l, Construct *c, ConstructList *n)
: AstChildren<Construct>(l, c, n) { }

void AccelBlock::outputCode(XStr& str) {
  if (code != NULL) {
    str << "\n";
    templateGuardBegin(false, str);
    str << "/***** Accel_Block Start *****/\n"
        << (*(code))
        << "\n/***** Accel_Block End *****/\n";
    templateGuardEnd(str);
    str << "\n";
  }
}

AccelBlock::AccelBlock(int l, XStr* c) { line = l; code = c; }
AccelBlock::~AccelBlock() { delete code; }

/// Printable Methods ///
void AccelBlock::print(XStr& str) { (void)str; }

/// Construct Methods ///
void AccelBlock::genDefs(XStr& str) { outputCode(str); }

/// Construct Accel Support Methods ///
int AccelBlock::genAccels_spe_c_funcBodies(XStr& str) { outputCode(str); return 0; }

}   // namespace xi
