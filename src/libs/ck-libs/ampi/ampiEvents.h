#ifndef __AMPI_EVENTS_H__
#define __AMPI_EVENTS_H__

#include "../../../traceCoreAPI.h"
#include "ampiProjections.h"

#define _AMPI_LANG_ID	4 /*language id for ampi*/

/* event IDs */
#define _E_BEGIN_AMPI_PROCESSING 25
#define _E_END_AMPI_PROCESSING	 26
#define _E_AMPI_MSG_SEND	 27
#define _E_AMPI_BEGIN_FUNC 28
#define _E_AMPI_END_FUNC 29

/* Registering Macro */
#define REGISTER_AMPI LOGCONDITIONAL(\
	{ RegisterLanguage(_AMPI_LANG_ID, "ampi\0"); \
	  RegisterEvent(_AMPI_LANG_ID, _E_BEGIN_AMPI_PROCESSING); \
	  RegisterEvent(_AMPI_LANG_ID, _E_END_AMPI_PROCESSING); \
	  RegisterEvent(_AMPI_LANG_ID,_E_AMPI_MSG_SEND); \
	  \
	})
#define _LOG_E_BEGIN_AMPI_PROCESSING(rank,src,count) { LOGCONDITIONAL(ampi_beginProcessing(rank,src,count));}
#define _LOG_E_END_AMPI_PROCESSING(rank)		    { LOGCONDITIONAL(ampi_endProcessing(rank));}
#define _LOG_E_AMPI_MSG_SEND(tag,dest,count,size)   { LOGCONDITIONAL(ampi_msgSend(tag,dest,count,size));}

#define _AMPI_REGISTER_FUNC(name) { LOGCONDITIONAL(ampi_registerFunc(name))}
#define _LOG_E_AMPI_BEGIN_FUNC(funcName,comm) {LOGCONDITIONAL(ampi_beginFunc(funcName,comm))}
#define _LOG_E_AMPI_END_FUNC(funcName,comm) {LOGCONDITIONAL(ampi_endFunc(funcName,comm))}

#endif
