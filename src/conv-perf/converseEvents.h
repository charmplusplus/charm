
#ifndef __CONVERSE_EVENTS_H__
#define __CONVERSE_EVENTS_H__

#include "traceCoreCommon.h"

/* Language ID */
#define _CONVERSE_LANG_ID	0	// language ID for converse

/* Event IDs */
#define _E_MSG_SENT			0
#define _E_MSG_QUEUED		1 	// DOUBT: Queued where ?
#define _E_MSG_RECV_MC		2	// Message received in machine layer
#define _E_MSG_RECV_SC		3	// Message received in scheduler
#define _E_HANDLER_BEGIN 	4 
#define _E_HANDLER_END		5 
#define _E_PROC_IDLE		6	// Processor goes idle 
#define _E_PROC_BUSY		7  	// Processor goes busy 

/* Trace Macros */
// TODO Currently there is no EventDataPrototype for the purpose of testing
#define REGISTER_CONVERSE \
	{ RegisterLanguage(_CONVERSE_LANG_ID); \
	  RegisterEvent(_CONVERSE_LANG_ID, _E_MSG_SENT     ); \
	  RegisterEvent(_CONVERSE_LANG_ID, _E_MSG_QUEUED   ); \
	  RegisterEvent(_CONVERSE_LANG_ID, _E_MSG_RECV_MC  ); \
	  RegisterEvent(_CONVERSE_LANG_ID, _E_MSG_RECV_SC  ); \
	  RegisterEvent(_CONVERSE_LANG_ID, _E_HANDLER_BEGIN); \
	  RegisterEvent(_CONVERSE_LANG_ID, _E_HANDLER_END  ); \
	  RegisterEvent(_CONVERSE_LANG_ID, _E_PROC_IDLE    ); \
	  RegisterEvent(_CONVERSE_LANG_ID, _E_PROC_BUSY    ); \
	}
#define _LOG_E_MSG_SENT() \
	{ LogEvent(_CONVERSE_LANG_ID, _E_MSG_SENT); }
#define _LOG_E_MSG_QUEUED() \
	{ LogEvent(_CONVERSE_LANG_ID, _E_MSG_QUEUED); }
#define _LOG_E_MSG_RECV_MC() \
	{ LogEvent(_CONVERSE_LANG_ID, _E_MSG_RECV_MC); }
#define _LOG_E_MSG_RECV_SC() \
	{ LogEvent(_CONVERSE_LANG_ID, _E_MSG_RECV_SC); }
#define _LOG_E_HANDLER_BEGIN() \
	{ LogEvent(_CONVERSE_LANG_ID, _E_HANDLER_BEGIN); }
#define _LOG_E_HANDLER_END() \
	{ LogEvent(_CONVERSE_LANG_ID, _E_HANDLER_END); }
#define _LOG_E_PROC_IDLE() \
	{ LogEvent(_CONVERSE_LANG_ID, _E_PROC_IDLE); }
#define _LOG_E_PROC_BUSY() \
	{ LogEvent(_CONVERSE_LANG_ID, _E_PROC_BUSY); }

#endif
