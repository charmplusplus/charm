
#ifndef __CHARM_EVENTS_H__
#define __CHARM_EVENTS_H__

#include "traceCoreCommon.h"

/* Language ID */
#define _CHARM_LANG_ID		2	// language ID for charm 

/* Event IDs */
#define  _E_CREATION           	1
#define  _E_BEGIN_PROCESSING   	2
#define  _E_END_PROCESSING     	3
#define  _E_ENQUEUE            	4
#define  _E_DEQUEUE            	5
#define  _E_BEGIN_COMPUTATION  	6
#define  _E_END_COMPUTATION    	7
#define  _E_BEGIN_INTERRUPT    	8
#define  _E_END_INTERRUPT      	9
#define  _E_MSG_RECV_CHARM     	10
#define  _E_USER_EVENT_CHARM	13
#define  _E_BEGIN_PACK          16
#define  _E_END_PACK            17
#define  _E_BEGIN_UNPACK        18
#define  _E_END_UNPACK          19

/* Trace Macros */
// TODO Currently there is no EventDataPrototype for the purpose of testing
#define REGISTER_CHARM \
	{ RegisterLanguage(_CHARM_LANG_ID); \
	  RegisterEvent(_CHARM_LANG_ID, _E_CREATION         ); \
	  RegisterEvent(_CHARM_LANG_ID, _E_BEGIN_PROCESSING ); \
	  RegisterEvent(_CHARM_LANG_ID, _E_END_PROCESSING   ); \
	  RegisterEvent(_CHARM_LANG_ID, _E_ENQUEUE          ); \
	  RegisterEvent(_CHARM_LANG_ID, _E_DEQUEUE          ); \
	  RegisterEvent(_CHARM_LANG_ID, _E_BEGIN_COMPUTATION); \
	  RegisterEvent(_CHARM_LANG_ID, _E_END_COMPUTATION  ); \
	  RegisterEvent(_CHARM_LANG_ID, _E_BEGIN_INTERRUPT  ); \
	  RegisterEvent(_CHARM_LANG_ID, _E_END_INTERRUPT    ); \
	  RegisterEvent(_CHARM_LANG_ID, _E_MSG_RECV_CHARM   ); \
	  RegisterEvent(_CHARM_LANG_ID, _E_BEGIN_PACK       ); \
	  RegisterEvent(_CHARM_LANG_ID, _E_END_PACK         ); \
	  RegisterEvent(_CHARM_LANG_ID, _E_BEGIN_UNPACK     ); \
	  RegisterEvent(_CHARM_LANG_ID, _E_END_UNPACK       ); \
	}
#define _LOG_E_CREATION() \
	{ LogEvent(_CHARM_LANG_ID, _E_CREATION); }
#define _LOG_E_BEGIN_PROCESSING() \
	{ LogEvent(_CHARM_LANG_ID, _E_BEGIN_PROCESSING); }
#define _LOG_E_END_PROCESSING() \
	{ LogEvent(_CHARM_LANG_ID, _E_END_PROCESSING); }
#define _LOG_E_ENQUEUE() \
	{ LogEvent(_CHARM_LANG_ID, _E_ENQUEUE); }
#define _LOG_E_DEQUEUE() \
	{ LogEvent(_CHARM_LANG_ID, _E_DEQUEUE); }
#define _LOG_E_BEGIN_COMPUTATION() \
	{ LogEvent(_CHARM_LANG_ID, _E_BEGIN_COMPUTATION); }
#define _LOG_E_END_COMPUTATION() \
	{ LogEvent(_CHARM_LANG_ID, _E_END_COMPUTATION); }
#define _LOG_E_BEGIN_INTERRUPT() \
	{ LogEvent(_CHARM_LANG_ID, _E_BEGIN_INTERRUPT); }
#define _LOG_E_END_INTERRUPT() \
	{ LogEvent(_CHARM_LANG_ID, _E_END_INTERRUPT); }
#define _LOG_E_MSG_RECV_CHARM() \
	{ LogEvent(_CHARM_LANG_ID, _E_MSG_RECV_CHARM); }
#define _LOG_E_BEGIN_PACK() \
	{ LogEvent(_CHARM_LANG_ID, _E_BEGIN_PACK); }
#define _LOG_E_END_PACK() \
	{ LogEvent(_CHARM_LANG_ID, _E_END_PACK); }
#define _LOG_E_BEGIN_UNPACK() \
	{ LogEvent(_CHARM_LANG_ID, _E_BEGIN_UNPACK); }
#define _LOG_E_END_UNPACK() \
	{ LogEvent(_CHARM_LANG_ID, _E_END_UNPACK); }

#endif
