
#ifndef __CHARM_EVENTS_H__
#define __CHARM_EVENTS_H__

#include "charmProjections.h"
#include "traceCoreAPI.h"

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
#define REGISTER_CHARM \
	{ RegisterLanguage(_CHARM_LANG_ID, "charm\0"); \
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
	  RegisterEvent(_CHARM_LANG_ID, _E_USER_EVENT_CHARM ); \
	  RegisterEvent(_CHARM_LANG_ID, _E_BEGIN_PACK       ); \
	  RegisterEvent(_CHARM_LANG_ID, _E_END_PACK         ); \
	  RegisterEvent(_CHARM_LANG_ID, _E_BEGIN_UNPACK     ); \
	  RegisterEvent(_CHARM_LANG_ID, _E_END_UNPACK       ); \
	  \
	}
#define _LOG_E_CREATION_1(env) 		{ charm_creation(env); }
#define _LOG_E_CREATION_N(env, n) 	{ charm_creation(env, n); }
#define _LOG_E_BEGIN_EXECUTE(env) 	{ charm_beginExecute(env); }
#define _LOG_E_BEGIN_EXECUTE_DETAILED(event, msgType, ep, srcPe, ml) \
	{ charm_beginExecuteDetailed(event, msgType, ep, srcPe, ml); }
#define _LOG_E_END_EXECUTE()	 	{ charm_endExecute(); }
//what is EXECUTE ? isn't it the same as processing --- schak
//TODO#define _LOG_E_BEGIN_PROCESSING()
//TODO#define _LOG_E_END_PROCESSING()
#define _LOG_E_ENQUEUE(env) 		{ charm_enqueueMsg(env); }
#define _LOG_E_DEQUEUE(env) 		{ charm_dequeueMsg(env); }
#define _LOG_E_BEGIN_COMPUTATION() 	{ charm_beginComputation(); }
#define _LOG_E_END_COMPUTATION() 	{ charm_endComputation(); }
//even projection actually doesn't use these  two
#define _LOG_E_BEGIN_INTERRUPT()
#define _LOG_E_END_INTERRUPT()


#define _LOG_E_MSG_RECV_CHARM(env, pe) 	{ charm_messageRecv(env, pe); }
#define _LOG_E_USER_EVENT_CHARM(x) 		{ charm_userEvent(x); }
#define _LOG_E_BEGIN_PACK() 			{ charm_beginPack(); }
#define _LOG_E_END_PACK() 				{ charm_endPack(); }
#define _LOG_E_BEGIN_UNPACK() 			{ charm_beginUnpack(); }
#define _LOG_E_END_UNPACK() 			{ charm_endUnpack(); }

#endif
