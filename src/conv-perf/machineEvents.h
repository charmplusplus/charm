/**
	Defines events for the basic machine language .....
	proc idle and busy type of events

**/

#ifndef __MACHINE_EVENTS_H__
#define __MACHINE_EVENTS_H__

#include "machineProjections.h"
#include "traceCoreAPI.h"



#define _MACHINE_LANG_ID	3	// language ID for machine


#define _E_PROC_IDLE		6	// Processor goes idle
#define _E_PROC_BUSY		7  	// Processor goes busy


#define REGISTER_MACHINE \
	{ RegisterLanguage(_MACHINE_LANG_ID, "machine\0"); \
  		RegisterEvent(_MACHINE_LANG_ID, _E_PROC_IDLE    ); \
	  	RegisterEvent(_MACHINE_LANG_ID, _E_PROC_BUSY    ); \
		\
	}

#define _LOG_E_PROC_IDLE() \
	{ LOGCONDITIONAL(machine_procIdle()); }
#define _LOG_E_PROC_BUSY() \
	{ LOGCONDITIONAL(machine_procBusy()); }

#endif

