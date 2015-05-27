#if !defined(CHARMXI_MAIN)
#define CHARMXI_MAIN

#include "sdag/constructs/Constructs.h"
#include "xi-symbol.h"
#include "xi-util.h"

int processAst(AstChildren<Module> *m, const bool chareNames,
               const bool dependsMode, const int fortranMode_,
               const int internalMode_, char* fname_, char* origFile_);

#endif /* CHARMXI_MAIN */
