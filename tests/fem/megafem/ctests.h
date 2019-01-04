/*
FEM Test routines: external interface
*/
#include "charm-api.h"

CLINKAGE void RUN_Test(void);
FLINKAGE void FTN_NAME(RUN_TEST,run_test)(void);

CLINKAGE void RUN_Abort(int v);
FLINKAGE void FTN_NAME(RUN_ABORT,run_abort)(int *v);

