#include "xi-Module.h"

#include <fstream>
#include <iostream>

using std::cout;
using std::cerr;
using std::endl;

namespace xi {

extern int fortranMode;

Module::Module(int l, const char* n, ConstructList* c) : name(n), clist(c) {
  line = l;
  _isMain = 0;
  if (clist) clist->recurse(this, &Construct::setModule);
}

void Module::print(XStr& str) {
  if (external) str << "extern ";
  str << "module " << name;
  if (clist) {
    str << " {\n";
    clist->print(str);
    str << "}\n";
  } else {
    str << ";\n";
  }
}

void Module::check() {
  if (clist) clist->check();
}

void Module::generate() {
  using std::ofstream;
  XStr declstr, defstr;
  XStr pubDeclStr, pubDefStr, pubDefConstr;

  declstr << "#ifndef _DECL_" << name
          << "_H_\n"
             "#define _DECL_"
          << name
          << "_H_\n"
             "#include \"charm++.h\"\n"
             "#include \"envelope.h\"\n"
             "#include <memory>\n"
             "#include \"sdag.h\"\n";

  if (isTramTarget()) declstr << "#include \"NDMeshStreamer.h\"\n";
  if (fortranMode) declstr << "#include \"charm-api.h\"\n";
  if (clist) clist->genDecls(declstr);
  if (clist) clist->outputClosuresDecl(declstr);
  if (clist) clist->outputClosuresDef(defstr);
  declstr << "extern void _register" << name << "(void);\n";
  if (isMain()) {
    declstr << "extern \"C\" void CkRegisterMainModule(void);\n";
  }

  // defstr << "#ifndef _DEFS_"<<name<<"_H_"<<endx;
  // defstr << "#define _DEFS_"<<name<<"_H_"<<endx;
  genDefs(defstr);
  templateGuardBegin(false, defstr);
  defstr << "void _register" << name
         << "(void)\n"
            "{\n"
            "  static int _done = 0; if(_done) return; _done = 1;\n";
  if (isTramTarget()) defstr << "  _registerNDMeshStreamer();\n";
  if (clist) clist->genReg(defstr);
  defstr << "}\n";
  if (isMain()) {
    if (fortranMode) defstr << "extern void _registerf90main(void);\n";
    defstr << "extern \"C\" void CkRegisterMainModule(void) {\n";
    if (fortranMode) {  // For Fortran90
      defstr << "  // FORTRAN\n";
      defstr << "  _registerf90main();\n";
    }
    defstr << "  _register" << name
           << "();\n"
              "}\n";
  }
  templateGuardEnd(defstr);
  // defstr << "#endif"<<endx;

  if (clist) clist->genGlobalCode("", declstr, defstr);
  declstr << "#endif" << endx;

  XStr topname, botname;
  topname << name << ".decl.h";
  botname << name << ".def.h";
  ofstream decl(topname.get_string()), def(botname.get_string());
  if (!decl || !def) {
    cerr << "Cannot open " << topname.get_string() << "or " << botname.get_string()
         << " for writing!!\n";
    die("cannot create output files (check directory permissions)\n");
  }

  std::string defstr_with_line_numbers =
      addLineNumbers(defstr.get_string(), botname.get_string());
  std::string sanitizedDefs(defstr_with_line_numbers.c_str());
  desanitizeCode(sanitizedDefs);
  decl << declstr.get_string();
  def << sanitizedDefs.c_str();
}

void Module::preprocess() {
  if (clist != NULL) clist->preprocess();
}

void Module::genDepend(const char* cifile) {
  cout << name << ".decl.h " << name << ".def.h: " << cifile << ".stamp" << endl;
}

void Module::genDecls(XStr& str) {
  if (external) {
    str << "#include \"" << name << ".decl.h\"\n";
  } else {
    if (clist) clist->genDecls(str);
  }
}

void Module::genDefs(XStr& str) {
  if (!external)
    if (clist) clist->genDefs(str);
}

void Module::genReg(XStr& str) {
  if (external) {
    str << "  _register" << name << "();" << endx;
  } else {
    if (clist) clist->genDefs(str);
  }
}

bool Module::isTramTarget() {
  if (clist)
    return clist->isTramTarget();
  else
    return false;
}

}  // namespace xi
