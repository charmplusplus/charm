#include <stdio.h>
#include <stdlib.h>
#include "converse.h"
#include "machineEvents.h"
#include "traceCoreCommon.h"
#include "machineProjections.h"


extern "C" void machine_procIdle()
{
	int iData[1];
	iData[0] = CmiMyPe();
	LogEvent1(_MACHINE_LANG_ID, _E_PROC_IDLE, 1, iData);
}

extern "C" void machine_procBusy()
{
	int iData[1];
	iData[0] = CmiMyPe();
	LogEvent1(_MACHINE_LANG_ID, _E_PROC_BUSY, 1, iData);
}

