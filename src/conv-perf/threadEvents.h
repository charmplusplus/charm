
#ifndef __THREAD_EVENTS_H__
#define __THREAD_EVENTS_H__

#include "traceCoreCommon.h"

/* Language ID */
#define _THREAD_LANG_ID		3	// language ID for threads 

/* Event IDs */
#define _E_THREAD_CREATION	0
#define _E_THREAD_AWAKEN	1
#define _E_THREAD_RESUME	2
#define _E_THREAD_SUSPEND	3

/* Trace Macros */
// TODO Currently there is no EventDataPrototype for the purpose of testing
#define REGISTER_THREAD \
	{ RegisterLanguage(_THREAD_LANG_ID, "thread"); \
	  RegisterEvent(_THREAD_LANG_ID, _E_THREAD_CREATION); \
	  RegisterEvent(_THREAD_LANG_ID, _E_THREAD_AWAKEN  ); \
	  RegisterEvent(_THREAD_LANG_ID, _E_THREAD_RESUME  ); \
	  RegisterEvent(_THREAD_LANG_ID, _E_THREAD_SUSPEND ); \
	}
#define _LOG_E_THREAD_CREATION() \
	{ LogEvent(_THREAD_LANG_ID, _E_THREAD_CREATION); }
#define _LOG_E_THREAD_AWAKEN() \
	{ LogEvent(_THREAD_LANG_ID, _E_THREAD_AWAKEN); }
#define _LOG_E_THREAD_RESUME() \
	{ LogEvent(_THREAD_LANG_ID, _E_THREAD_RESUME); }
#define _LOG_E_THREAD_SUSPEND() \
	{ LogEvent(_THREAD_LANG_ID, _E_THREAD_SUSPEND); }

#endif
