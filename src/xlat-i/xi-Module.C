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
  #if CMK_CUDA != 0
    XStr accelstr_cuda_c, accelstr_cuda_h;
  #endif

  declstr <<
  "#ifndef _DECL_"<<name<<"_H_\n"
  "#define _DECL_"<<name<<"_H_\n"
  "#include \"charm++.h\"\n"
  "#include \"envelope.h\"\n"
  "#include <memory>\n"
  "#include \"sdag.h\"\n";
   // DMK - Forward declare external references for the hybrid api's memory pooling functions
  #if CMK_CUDA != 0
    declstr << "extern void hapi_poolFree(void*);\n"
            << "extern void* hapi_poolMalloc(int);\n";
  #endif

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
   // DMK - Accel Support - GPGPU-specific file/code generation
  #if CMK_CUDA != 0

    /// Generate the CUDA code file contents ///
    accelstr_cuda_c << "#ifndef __ACCEL_" << name << "_CU__\n";
    accelstr_cuda_c << "#define __ACCEL_" << name << "_CU__\n\n\n";
    int numAccelEntries = genAccels_cuda_c_funcBodies(accelstr_cuda_c);
    accelstr_cuda_c << "\n\n#endif //__ACCEL_" << name << "_CU__\n";

    /// Generate the CUDA header file contents ///
    accelstr_cuda_h << "#ifndef __ACCEL_" << name << "_H__\n";
    accelstr_cuda_h << "#define __ACCEL_" << name << "_H__\n\n\n";
    genAccels_cuda_h_includes(accelstr_cuda_h);
    accelstr_cuda_h << "\n\n";
    //accelstr_cuda_h << "#define MODULE_" << name << "_FUNC_INDEX_COUNT (" << numAccelEntries;
    //genAccels_cuda_h_fiCountDefs(accelstr_cuda_h);
    //accelstr_cuda_h << ")\n\n\n";
    accelstr_cuda_h << "#endif //__ACCEL_" << name << "_H__\n";

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
  #if CMK_CUDA != 0

    // Generate this module's code (actually create the files) ///
    XStr accelname_cuda_c, accelname_cuda_h;
    accelname_cuda_c << name << ".genCUDACode.cu";
    accelname_cuda_h << name << ".genCUDACode.h";
    ofstream accel_cuda_c(accelname_cuda_c.get_string()), accel_cuda_h(accelname_cuda_h.get_string());
    if (!accel_cuda_c) {
      cerr << "ERROR: Unable to open " << accelname_cuda_c.get_string() << " for writing!\n";
      die("Cannot create output files (check directory permissions)\n");
    }
    if (!accel_cuda_h) {
      cerr << "ERROR: Unable to open " << accelname_cuda_h.get_string() << " for writing!\n";
      die("Cannot create output files (check directory permissions)\n");
    }
    accel_cuda_c << accelstr_cuda_c.get_string();
    accel_cuda_h << accelstr_cuda_h.get_string();

    // Create the main-module-only code
    if (isMain()) {

      // Create the general "__triggered_main.genCUDACode.cu" file
      ofstream accel_cuda_c("__triggered_main.genCUDACode.cu");
      if (!accel_cuda_c) {
        cerr << "ERROR: Unable to open \"__triggered_main.genCUDACode.cu\" for writting!\n";
        die("Unable to open \"__triggered_main.genCUDACode.cu\" for writting!\n");
      }

      accel_cuda_c << "// DMK - NOTE | TODO | FIXME - This code only supports a single mainmodule, since\n"
                   << "//   the charmxi tool can only process a single ci file at a time.  As a result, each\n"
                   << "//   mainmodule declaration will overwrite the previous mainmodule's includes as the\n"
                   << "//   main generated source code file is recreated.  Fix this in the future if needed\n"
                   << "//   (after checking to see if multiple mainmodules is even allowed in Charm++).\n\n";

      accel_cuda_c << "// DMK - NOTE | TODO | FIXME - There are currently two registration passes/functions\n"
                   << "//   for each module, one for the host/Charm++ code and one for the generated CUDA code.\n"
                   << "//   This is a result of the generated CUDA code not being able to include Charm++\n"
                   << "//   header files, but requiring access to some data structures.  So, the shared data\n"
                   << "//   structures (e.g. funcLookupTable) are in one pass, while the Charm++ calls (e.g.\n"
                   << "//   user event registration) are in the other pass.\n\n";

      accel_cuda_c << "// DMK - NOTE | TODO | FIXME - There was some issue with having includes/library paths\n"
                   << "//   specific for device code, but as I add this note, I don't recall what it issue was\n"
                   << "//   (perhaps it has been fixed by all the updates I've done since setting this up in the\n"
                   << "//   first place).  At the moment, the generate source codes for each module are included\n"
                   << "//   directly into one another, with a main generated code file being the individual file\n"
                   << "//   that is passed to the CUDA compiler.  This could get fixed in the future if\n"
                   << "//   accelerator support within charmc is cleaned up and a scheme for paths in created.\n";

      accel_cuda_c << "// Include standard C/C++ files\n"
                   << "#include <stdlib.h>\n"
                   << "#include <stdio.h>\n\n";

      #if CMK_CUDA != 0
        accel_cuda_c << "// Define the GPU_MEMPOOL macro so memory pooling is used by the Hybrid API\n"
                     << "#define GPU_MEMPOOL (1)\n\n";
      #endif

      accel_cuda_c << "// Include Charm++ specific header files\n"
                   << "#include \"wr.h\"  // Hybrid API\n"
                   << "#include \"ckaccel_common.h\" // Accelerator Manager common file\n\n";

      accel_cuda_c << "\n///// The types and variables required for function lookup (registration code) /////\n\n"
                   << "typedef void(*AccelFuncPtr)(workRequest*);\n\n"
                   << "typedef struct __func_lookup_table_entry {\n"
                   << "  int funcIndex;\n"
                   << "  AccelFuncPtr funcPtr;\n"
                   << "} FuncLookupTableEntry;\n\n"
                   << "FuncLookupTableEntry *funcLookupTable = NULL;\n"
                   << "int funcLookupTable_len = 0;\n"
                   << "int funcLookupTable_maxLen = 0;\n\n";

      accel_cuda_c << "\n///// Forward declare functions required by module-specific code /////\n\n"
                   << "extern void traceKernelIssueTime();\n"
                   << "void kernelSetup(int funcIndex, void* data, int dataLen, void* callback, void* userData, int numThreads, void** wrPtr, void** diPtr);  // See declaration below for patameter descriptions\n\n";

      accel_cuda_c << "\n///// Include the main module's generated code files (see note above) /////\n\n"
                   << "#include \"" << name << ".genCUDACode.h\"\n"
                   << "#include \"" << name << ".genCUDACode.cu\"\n\n";

      accel_cuda_c << "\n///// The top-level registration functions /////\n\n"
                   << "extern \"C\" int register_accel_cuda_funcs__module_" << name << "(int);\n\n"
                   << "extern \"C\" void register_accel_cuda_funcs(void) {\n"
                   << "  // Register the AEM functions\n"
                   << "  int funcCount0 = register_accel_cuda_funcs__module_" << name << "(0);\n"
                   << "  int funcCount1 = register_accel_funcs_" << name << "(0);\n"
                   << "  if (funcCount0 != funcCount1) { // NOTE: Verify that the counts match as a safety check (in case one registration pass is modified, but the other is not)\n"
                   << "    printf(\"[ACCEL-ERROR] :: Registration counts mismatch (%d vs %d)\\n\", funcCount0, funcCount1);\n"
                   << "  }\n"
                   << "  funcLookupTable_len = funcCount0;\n"
                   << "}\n\n"
                   << "extern \"C\" void cleanup_registration(void) {\n"
                   << "  if (funcLookupTable != NULL) { delete [] funcLookupTable; }\n"
                   << "  funcLookupTable = NULL;\n"
                   << "  funcLookupTable_maxLen = 0;\n"
                   << "}\n\n";

      accel_cuda_c << "\n///// Kernel processing /////\n\n"
                   << "// Declare the kernelSetup function, which is used to issue kernels to the Hybrid API\n"
                   << "void kernelSetup(int funcIndex,     // Function index\n"
                   << "                 void* data,        // Pointer to kernel data structure (batched set of elements)\n"
                   << "                 int dataLen,       // Length (in bytes) of buffer pointed to by data\n"
                   << "                 void* callback,    // Pointer to the callback for this work request\n"
                   << "                 void* userData,    // User data for the callback function (pointer to callback data structure)\n"
                   << "                 int numThreads,    // Number of threads required by the kernel\n"
                   << "                 void** wrPtr,      // Pointer to work request data structure (will be filled in)\n"
                   << "                 void** diPtr       // Pointer to data info for work request (will be filled in)\n"
                   << "                ) {\n"
                   << "\n"
                   << "  // Write in the header info\n"
                   << "  int *data_int = (int*)data;\n"
                   << "  data_int[ACCEL_CUDA_KERNEL_LEN_INDEX          ] = dataLen;  // buffer length : written by host\n"
                   << "  data_int[ACCEL_CUDA_KERNEL_BIT_PATTERN_0_INDEX] = 0;        // bit pattern 0 : written by device\n"
                   << "  data_int[ACCEL_CUDA_KERNEL_BIT_PATTERN_1_INDEX] = 0;        // bit pattern 1 : written by device\n"
                   << "  data_int[ACCEL_CUDA_KERNEL_ERROR_INDEX        ] = 0;        //    error code : written by device\n"
                   << "  // NOTE: ACCEL_CUDA_KERNEL_NUM_SPLITS and ACCEL_CUDA_KERNEL_SET_SIZE should be set by caller !!!\n"
                   << "\n"
                   << "  // Create the workRequest\n"
                   << "  workRequest *wr = new workRequest;\n"
                   << "  if (wr == NULL) { printf(\"[ERROR] :: Unable to allocate memory for workRequest...\\n\"); }\n"
                   << "  *wrPtr = wr;  // Store this pointer within the userData (cbStruct) so it can be deleted later\n"
                   << "\n"
                   << "  // Create the dataInfo\n"
                   << "  dataInfo *di = (dataInfo*)(malloc(sizeof(dataInfo)));\n"
                   << "  if (di == NULL) { printf(\"[ERROR] :: Unable to allocate memory for dataInfo...\\n\"); }\n"
                   << "  *diPtr = di;  // Store this pointer within the userData (cbStruct) so it can be deleted later\n"
                   << "  di->bufferID = -1; // Let the Hybrid API assign the bufferID\n"
                   << "  di->transferToDevice = 1;\n"
                   << "  di->transferFromDevice = 1;\n"
                   << "  di->freeBuffer = 1;\n"
                   << "  di->hostBuffer = data;\n"
                   << "  di->size = dataLen;\n"
                   << "\n"
                   << "  // Fill in the dataInfo and the workRequest data structures\n"
                   << "  wr->dimGrid.x = 1;\n"
                   << "  wr->dimGrid.y = 1;\n"
                   << "  wr->dimGrid.z = 1;\n"
                   << "  wr->dimBlock.x = numThreads;\n"
                   << "  wr->dimBlock.y = 1;\n"
                   << "  wr->dimBlock.z = 1;\n"
                   << "  wr->smemSize = 0;\n"
                   << "  wr->nBuffers = 1;\n"
                   << "  wr->bufferInfo = di;\n"
                   << "  wr->callbackFn = callback;\n"
                   << "  wr->id = funcIndex;\n"
                   << "  wr->state = 0;\n"
                   << "  wr->userData = data;\n"
                   << "\n"
                   << "  // Enqueue the workRequest\n"
                   << "  enqueue(wrQueue, wr);\n"
                    << "  markKernelStart();\n"
                   << "}\n\n";

      accel_cuda_c << "void kernelCleanup(void *wr, void *di) {\n"
                   << "  delete ((workRequest*)wr);\n"
                   << "  // delete ((dataInfo*)di); <-- Not forgotten... it seems the Hybrid API deletes this structure for us, so do not delete again\n"
                   << "}\n\n";

      accel_cuda_c << "void kernelSelect(workRequest *wr) {\n"
                   << "  int funcIndex = wr->id;\n"
                   << "  if ((funcIndex >= 0) && (funcIndex < funcLookupTable_len)) {\n"
                   << "    (funcLookupTable[funcIndex].funcPtr)(wr);\n"
                   << "  } else {\n"
                   << "    printf(\"ERROR : Unknown funcIndex (%d) passed to kernelSelect()... ignoring.\\n\", funcIndex);\n"
                   << "  }\n"
                   << "}\n\n";

      accel_cuda_c << "\n///// Device buffer functions /////\n\n"
                   << "void* newDeviceBuffer(size_t size) {\n"
                   << "  if (size <= 0) { return NULL; }\n"
                   << "  void *devicePtr = NULL;\n"
                   << "  if (cudaSuccess != cudaMalloc(&devicePtr, size)) { return NULL; }\n"
                   << "  return devicePtr;\n"
                   << "}\n\n";

      accel_cuda_c << "// Returns -1 on failure, 0 on success\n"
                   << "int deleteDeviceBuffer(void *devicePtr) {\n"
                   << "  if (devicePtr == NULL) { return -1; }\n"
                   << "  if (cudaSuccess != cudaFree(devicePtr)) { return -1; }\n"
                   << "  return 0;\n"
                   << "}\n\n";

      accel_cuda_c << "// Returns -1 on failure, 0 on success\n"
                   << "int pushToDevice(void *hostPtr, void *devicePtr, size_t size) {\n"
                   << "  if (hostPtr == NULL || devicePtr == NULL || size <= 0) { return -1; }\n"
                   << "  if (cudaSuccess != cudaMemcpy(devicePtr, hostPtr, size, cudaMemcpyHostToDevice)) { return -1; }\n"
                   << "  return 0;\n"
                   << "}\n\n";

      accel_cuda_c << "int pullFromDevice(void *hostPtr, void *devicePtr, size_t size) {\n"
                   << "  if (hostPtr == NULL || devicePtr == NULL || size <= 0) { return -1; }\n"
                   << "  if (cudaSuccess != cudaMemcpy(hostPtr, devicePtr, size, cudaMemcpyDeviceToHost)) { return -1; }\n"
                   << "  return 0;\n"
                   << "}\n\n";
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
  #if CMK_CUDA != 0
    str << "extern \"C\" int register_accel_cuda_funcs__module_" << name << "(int curIndex);\n";
    if (isMain()) {
      str << "extern \"C\" void register_accel_cuda_funcs(void);\n";
    }
  #endif
}

void Module::genDefs(XStr& str)
{
  str << "#include \"ckaccel_common.h\"\n"
      << "#include \"ckaccel.h\"\n";
  #if CMK_CUDA != 0
    //str << "#include \"__triggered_main.genCUDACode.h\"\n";
    str << "#ifndef __ACCEL_CUDA_KERNEL_FUNCS__\n"
        << "#define __ACCEL_CUDA_KERNEL_FUNCS__\n"
        << "  extern void kernelSetup(int, void*, int, void*, void*, int, /*int,*/ void**, void**);\n"
        << "  extern void kernelCleanup(void*, void*);\n"
        << "#endif //__ACCEL_CUDA_KERNEL_FUNCS__\n\n\n";
  #endif
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
  #if CMK_CUDA != 0

    if (!external) {

      // Protect this function with CK_TEMPLATES_ONLY check
      str << "#ifndef CK_TEMPLATES_ONLY\n";

      // Create the registration function
      str << "extern \"C\" int register_accel_cuda_funcs__module_" << name << "(int curIndex) {\n"
          << "  static int hasAlreadyRegisteredFlag = 0;\n"
          << "  if (hasAlreadyRegisteredFlag) { return curIndex; };\n"
          << "  hasAlreadyRegisteredFlag = 1;\n";
      genAccels_cuda_host_c_regFuncs(str);
      str << "  return curIndex;\n"
          << "}\n";
      str << "#endif /*CK_TEMPLATES_ONLY*/\n";
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

int Module::genAccels_cuda_c_funcBodies(XStr& str) {

  // If this is an external module decloration, just place an include for the specified module's source code file
  if (external) {
    str << "#include \"" << name << ".genCUDACode.cu\"\n";
    return 0;
  }

  // If this is the main module, include the header files
  if (isMain()) {
    str << "#include <stdlib.h>\n"
        << "#include <stdio.h>\n"
        << "#include \"simd.h\"\n";
  }
  str << "#include \"" << name << ".genCUDACode.h\"\n\n\n";

  //// DMK - DEBUG - Moved to main code file
  //if (isMain()) {
  //  str << "extern void traceKernelIssueTime();\n\n";
  //}

  //// If this is the main module, generate the code for work request lookup
  //if (isMain()) {
  //  str << "typedef void(*AccelFuncPtr)(workRequest*);\n\n"
  //      << "typedef struct __func_lookup_table_entry {\n"
  //      << "  int funcIndex;\n"
  //      << "  AccelFuncPtr funcPtr;\n"
  //      << "} FuncLookupTableEntry;\n\n"
  //      << "FuncLookupTableEntry funcLookupTable[MODULE_" << name << "_FUNC_INDEX_COUNT];\n\n\n";
  //}

  str << "// Generate includes for external modules (see NOTE at top of the main generated code file)\n";
  int rtn = 0;
  if (clist) {
    rtn += clist->genAccels_cuda_c_funcBodies(str);
  }

  // Create the accelerated function registration function for accelerated entries local to this module
  // NOTE: Add a check so modules won't register more than once.  It is possible that to modules
  //   could have 'extern' references to each other, creating infinite loops in the registration
  //   process.  Avoid this problem.
  str << "// CUDA code registration function for this module\n"
      << "// NOTE: This function takes an index as input, which in increments by the number of AEMs within\n"
      << "//   this module.  The return value is the passed in value of curIndex plus the number of AEMs.\n"
      << "int register_accel_funcs_" << name << "(int curIndex) {\n\n"
      << "  // Create a check to ensure that this function is only executed once\n"
      << "  static int hasAlreadyRegisteredFlag = 0;\n"
      << "  if (hasAlreadyRegisteredFlag) { return curIndex; };\n"
      << "  hasAlreadyRegisteredFlag = 1;\n\n";
  genAccels_cuda_c_regFuncs(str);
  str << "  return curIndex;\n"
      << "}\n";

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
void Module::genAccels_cuda_c_regFuncs(XStr& str) {
  if (external) {
    str << "  curIndex = register_accel_funcs_" << name << "(curIndex);\n";
  } else {
    if (clist) { clist->genAccels_cuda_c_regFuncs(str); }
  }
}

void Module::genAccels_cuda_host_c_regFuncs(XStr& str) {
  if (external) {
    str << "  curIndex = register_accel_cuda_funcs__module_" << name << "(curIndex);\n";
  } else {
    if (clist) { clist->genAccels_cuda_host_c_regFuncs(str); }
  }
}

void Module::genAccels_cuda_h_includes(XStr& str) {
  if (external) {
    str << "#include \"" << name << ".genCUDACode.h\"\n";
  }
  if (clist) { clist->genAccels_cuda_h_includes(str); }
}

void Module::genAccels_cuda_h_fiCountDefs(XStr& str) {
  //if (external) {
  //  str << " + MODULE_" << name << "_FUNC_INDEX_COUNT";
  //}
  if (clist) { clist->genAccels_cuda_h_fiCountDefs(str); }
}

}   // namespace xi
