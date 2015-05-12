#include "xi-Module.h"

#include <iostream>
#include <fstream>

using std::cout;
using std::cerr;
using std::endl;

namespace xi {

extern int fortranMode;

Module::Module(int l, const char *n, ConstructList *c)
  : name(n)
  , clist(c)
{
  line = l;
  _isMain=0;
  if (clist)
    clist->recurse(this, &Construct::setModule);
}

void Module::print(XStr& str)
{
  if (external)
    str << "extern ";
  str << "module "<<name;
  if (clist) {
    str << " {\n";
    clist->print(str);
    str << "}\n";
  } else {
    str << ";\n";
  }
}

void Module::check() {
  if (clist)
    clist->check();
}

void Module::generate()
{
  using std::ofstream;
  XStr declstr, defstr;
  XStr pubDeclStr, pubDefStr, pubDefConstr;

  // DMK - Accel Support
  #if CMK_CELL != 0
  XStr accelstr_spe_c, accelstr_spe_h;
  #endif

  declstr <<
  "#ifndef _DECL_"<<name<<"_H_\n"
  "#define _DECL_"<<name<<"_H_\n"
  "#include \"charm++.h\"\n"
  "#include \"envelope.h\"\n"
  "#include <memory>\n"
  "#include \"sdag.h\"\n";
  if (fortranMode) declstr << "#include \"charm-api.h\"\n";
  if (clist) clist->genDecls(declstr);
  if (clist) clist->outputClosuresDecl(declstr);
  if (clist) clist->outputClosuresDef(defstr);
  declstr << "extern void _register"<<name<<"(void);\n";
  if(isMain()) {
    declstr << "extern \"C\" void CkRegisterMainModule(void);\n";
  }

  // defstr << "#ifndef _DEFS_"<<name<<"_H_"<<endx;
  // defstr << "#define _DEFS_"<<name<<"_H_"<<endx;
  genDefs(defstr);
  templateGuardBegin(false, defstr);
  defstr <<
  "void _register"<<name<<"(void)\n"
  "{\n"
  "  static int _done = 0; if(_done) return; _done = 1;\n";
  if (clist) clist->genReg(defstr);
  defstr << "}\n";
  if(isMain()) {
    if (fortranMode) defstr << "extern void _registerf90main(void);\n";
    defstr << "extern \"C\" void CkRegisterMainModule(void) {\n";
    if (fortranMode) { // For Fortran90
      defstr << "  // FORTRAN\n";
      defstr << "  _registerf90main();\n";
    }
    defstr <<
    "  _register"<<name<<"();\n"
    "}\n";
  }
  templateGuardEnd(defstr);
  // defstr << "#endif"<<endx;

  if (clist) clist->genGlobalCode("", declstr, defstr);
  declstr << "#endif" << endx;

  // DMK - Accel Support
  #if CMK_CELL != 0

  /// Generate the SPE code file contents ///
  accelstr_spe_c << "#ifndef __ACCEL_" << name << "_C__\n"
                 << "#define __ACCEL_" << name << "_C__\n\n\n";
  int numAccelEntries = genAccels_spe_c_funcBodies(accelstr_spe_c);
  accelstr_spe_c << "\n\n#endif //__ACCEL_" << name << "_C__\n";

  /// Generate the SPE header file contents ///
  accelstr_spe_h << "#ifndef __ACCEL_" << name << "_H__\n"
                 << "#define __ACCEL_" << name << "_H__\n\n\n";
  genAccels_spe_h_includes(accelstr_spe_h);
  accelstr_spe_h << "\n\n";
  accelstr_spe_h << "#define MODULE_" << name << "_FUNC_INDEX_COUNT (" << numAccelEntries;
  genAccels_spe_h_fiCountDefs(accelstr_spe_h);
  accelstr_spe_h << ")\n\n\n";
  accelstr_spe_h << "#endif //__ACCEL_" << name << "_H__\n";

  #endif


  XStr topname, botname;
  topname<<name<<".decl.h";
  botname<<name<<".def.h";
  ofstream decl(topname.get_string()), def(botname.get_string());
  if(!decl || !def) {
    cerr<<"Cannot open "<<topname.get_string()<<"or "
	<<botname.get_string()<<" for writing!!\n";
    die("cannot create output files (check directory permissions)\n");
  }

  std::string defstr_with_line_numbers = addLineNumbers(defstr.get_string(), botname.get_string());
  std::string sanitizedDefs(defstr_with_line_numbers.c_str());
  desanitizeCode(sanitizedDefs);
  decl<<declstr.get_string();
  def<<sanitizedDefs.c_str();

  // DMK - Accel Support
  #if CMK_CELL != 0

  /// Generate this module's code (actually create the files) ///
  XStr accelname_c, accelname_h;
  accelname_c << name << ".genSPECode.c";
  accelname_h << name << ".genSPECode.h";
  ofstream accel_c(accelname_c.get_string()), accel_h(accelname_h.get_string());
  if (!accel_c) {
    cerr << "Cannot open " << accelname_c.get_string() << " for writing!!\n";
    die("Cannot create output files (check directory permissions)\n");
  }
  if (!accel_h) {
    cerr << "Cannot open " << accelname_h.get_string() << " for writing!!\n";
    die("Cannot create output files (check directory permissions)\n");
  }
  accel_c << accelstr_spe_c.get_string();
  accel_h << accelstr_spe_h.get_string();
  
  // If this is the main module, generate the general C file and include this modules accel.h file
  if (isMain()) {

    XStr mainAccelStr_c;
    mainAccelStr_c << "#include \"main__funcLookup__.genSPECode.h" << "\"\n"
                   << "#include \"" << name << ".genSPECode.c" << "\"\n";
    ofstream mainAccel_c("main__funcLookup__.genSPECode.c");
    if (!mainAccel_c) {
      cerr << "Cannot open main__funcLookup__.genSPECode.c for writing!!\n";
      die("Cannot create output files (check directory permissions)");
    }
    mainAccel_c << mainAccelStr_c.get_string();

    XStr mainAccelStr_h;
    mainAccelStr_h << "#ifndef __MAIN_FUNCLOOKUP_H__\n"
                   << "#define __MAIN_FUNCLOOKUP_H__\n\n"
		   << "#include <spu_intrinsics.h>\n"
		   << "#include <stdlib.h>\n"
		   << "#include <stdio.h>\n"
		   << "#include \"spert.h\"\n\n"
		   << "#include \"simd.h\"\n"
                   << "#include \"" << name << ".genSPECode.h" << "\"\n\n"
                   << "#endif //__MAIN_FUNCLOOKUP_H__\n";
    ofstream mainAccel_h("main__funcLookup__.genSPECode.h");
    if (!mainAccel_h) {
      cerr << "Cannot open main__funcLookup__.genSPECode.h for writing!!\n";
      die("Cannot create output files (check directory permissions)");
    }
    mainAccel_h << mainAccelStr_h.get_string();

  }

  #endif
}

void Module::preprocess()
{
  if (clist!=NULL) clist->preprocess();
}

void Module::genDepend(const char *cifile)
{
  cout << name << ".decl.h " << name << ".def.h: "
       << cifile << ".stamp" << endl;
}


void Module::genDecls(XStr& str)
{
  if(external) {
    str << "#include \""<<name<<".decl.h\"\n";
  } else {
    if (clist) clist->genDecls(str);
  }

  #if CMK_CELL != 0
    str << "extern int register_accel_spe_funcs__module_" << name << "(int curIndex);\n";
    if (isMain()) {
      str << "extern \"C\" void register_accel_spe_funcs(void);\n";
    }
  #endif
}

void Module::genDefs(XStr& str)
{
  if(!external)
    if (clist)
      clist->genDefs(str);

  // DMK - Accel Support
  #if CMK_CELL != 0

    if (!external) {
      templateGuardBegin(false, str);

      // Create the registration function
      // NOTE: Add a check so modules won't register more than once.  It is possible that to modules
      //   could have 'extern' references to each other, creating infinite loops in the registration
      //   process.  Avoid this problem.
      str << "int register_accel_spe_funcs__module_" << name << "(int curIndex) {\n"
          << "  static int hasAlreadyRegisteredFlag = 0;\n"
          << "  if (hasAlreadyRegisteredFlag) { return curIndex; };\n"
          << "  hasAlreadyRegisteredFlag = 1;\n";
      genAccels_ppe_c_regFuncs(str);
      str << "  return curIndex;\n"
          << "}\n";

      // Check to see if this is the main module (create top register function if so)
      if (isMain()) {
        str << "#include\"spert.h\"\n"  // NOTE: Make sure SPE_FUNC_INDEX_USER is defined
            << "extern \"C\" void register_accel_spe_funcs(void) {\n"
	    << "  register_accel_spe_funcs__module_" << name << "(SPE_FUNC_INDEX_USER);\n"
	    << "}\n";
      }

      templateGuardEnd(str);
    }

  #endif
}

void Module::genReg(XStr& str)
{
  if(external) {
    str << "  _register"<<name<<"();"<<endx;
  } else {
    if (clist) clist->genDefs(str);
  }
}


int Module::genAccels_spe_c_funcBodies(XStr& str) {

  // If this is an external module decloration, just place an include
  if (external) {
    str << "#include \"" << name << ".genSPECode.c\"\n";
    return 0;
  }

  // If this is the main module, generate the function lookup table
  if (isMain()) {
    str << "typedef void(*AccelFuncPtr)(DMAListEntry*);\n\n"
        << "typedef struct __func_lookup_table_entry {\n"
        << "  int funcIndex;\n"
        << "  AccelFuncPtr funcPtr;\n"
        << "} FuncLookupTableEntry;\n\n"
        << "FuncLookupTableEntry funcLookupTable[MODULE_" << name << "_FUNC_INDEX_COUNT];\n\n\n";
  }

  // Process each of the sub-constructs
  int rtn = 0;
  if (clist) { rtn += clist->genAccels_spe_c_funcBodies(str); }

  // Create the accelerated function registration function for accelerated entries local to this module
  // NOTE: Add a check so modules won't register more than once.  It is possible that to modules
  //   could have 'extern' references to each other, creating infinite loops in the registration
  //   process.  Avoid this problem.
  str << "int register_accel_funcs_" << name << "(int curIndex) {\n"
      << "  static int hasAlreadyRegisteredFlag = 0;\n"
      << "  if (hasAlreadyRegisteredFlag) { return curIndex; };\n"
      << "  hasAlreadyRegisteredFlag = 1;\n";
  genAccels_spe_c_regFuncs(str);
  str << "  return curIndex;\n"
      << "}\n\n\n";

  // If this is the main module, generate the funcLookup function
  if (isMain()) {

    str << "\n\n";
    str << "#ifdef __cplusplus\n"
        << "extern \"C\"\n"
        << "#endif\n"
        << "void funcLookup(int funcIndex,\n"
        << "                void* readWritePtr, int readWriteLen,\n"
        << "                void* readOnlyPtr, int readOnlyLen,\n"
        << "                void* writeOnlyPtr, int writeOnlyLen,\n"
        << "                DMAListEntry* dmaList\n"
        << "               ) {\n\n";

    str << "  if ((funcIndex >= SPE_FUNC_INDEX_USER) && (funcIndex < (SPE_FUNC_INDEX_USER + MODULE_" << name << "_FUNC_INDEX_COUNT))) {\n"

        //<< "    // DMK - DEBUG\n"
        //<< "    printf(\"[DEBUG-ACCEL] :: [SPE_%d] - Calling funcIndex %d...\\n\", (int)getSPEID(), funcIndex);\n"

        << "    (funcLookupTable[funcIndex - SPE_FUNC_INDEX_USER].funcPtr)(dmaList);\n"
        << "  } else if (funcIndex == SPE_FUNC_INDEX_INIT) {\n"
        << "    if (register_accel_funcs_" << name << "(0) != MODULE_" << name << "_FUNC_INDEX_COUNT) {\n"
        << "      printf(\"ERROR : register_accel_funcs_" << name << "() returned an invalid value.\\n\");\n"
	<< "    };\n";
    genAccels_spe_c_callInits(str);
    str << "  } else if (funcIndex == SPE_FUNC_INDEX_CLOSE) {\n"
        << "    // NOTE : Do nothing on close, but handle case...\n"
        << "  } else {\n"
	<< "    printf(\"ERROR : Unknown funcIndex (%d) passed to funcLookup()... ignoring.\\n\", funcIndex);\n"
	<< "  }\n";

    str << "}\n";
  }

  return rtn;
}

void Module::genAccels_spe_c_regFuncs(XStr& str) {
  if (external) {
    str << "  curIndex = register_accel_funcs_" << name << "(curIndex);\n";
  } else {
    if (clist) { clist->genAccels_spe_c_regFuncs(str); }
  }
}

void Module::genAccels_spe_c_callInits(XStr& str) {
  if (clist) { clist->genAccels_spe_c_callInits(str); }
}

void Module::genAccels_spe_h_includes(XStr& str) {
  if (external) {
    str << "#include \"" << name << ".genSPECode.h\"\n";
  }
  if (clist) { clist->genAccels_spe_h_includes(str); }
}

void Module::genAccels_spe_h_fiCountDefs(XStr& str) {
  if (external) {
    str << " + MODULE_" << name << "_FUNC_INDEX_COUNT";
  }
  if (clist) { clist->genAccels_spe_h_fiCountDefs(str); }
}

void Module::genAccels_ppe_c_regFuncs(XStr& str) {
  if (external) {
    str << "  curIndex = register_accel_spe_funcs__module_" << name << "(curIndex);\n";
  } else {
    if (clist) { clist->genAccels_ppe_c_regFuncs(str); }
  }
}

}   // namespace xi
