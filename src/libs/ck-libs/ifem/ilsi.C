/**
Iterative Linear Solver Interface

Orion Sky Lawlor, olawlor@acm.org, 1/16/2003
*/
#include "ilsi.h"

ILSI_Comm::~ILSI_Comm() {}


CDECL void ILSI_Param_new(ILSI_Param *param)
{
	param->maxResidual=1.0e-6;
	param->maxIterations=0;
	int i;
	for (i=0;i<8;i++) param->solverIn[i]=0;
	param->residual=-1;
	param->iterations=-1;
	for (i=0;i<8;i++) param->solverOut[i]=0;
}
FORTRAN_AS_C(ILSI_PARAM_NEW,ILSI_Param_new,ilsi_param_new,
	(int *param),((ILSI_Param *)param))


