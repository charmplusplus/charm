#ifndef __AMPI_EVENTS_H__
#define __AMPI_EVENTS_H__

#include "traceCoreAPI.h"
#include "ampiProjections.h"

#define _AMPI_LANG_ID	4 /*language id for ampi*/

/* event IDs */
#define _E_BEGIN_AMPI_PROCESSING 25
#define _E_END_AMPI_PROCESSING	 26
#define _E_AMPI_MSG_SEND	 27

/* Registering Macro */
#define REGISTER_AMPI \
	{ RegisterLanguage(_AMPI_LANG_ID, "ampi\0"); \
	  RegisterEvent(_AMPI_LANG_ID, _E_BEGIN_AMPI_PROCESSING); \
	  RegisterEvent(_AMPI_LANG_ID, _E_END_AMPI_PROCESSING); \
	  RegisterEvent(_AMPI_LANG_ID,_E_AMPI_MSG_SEND); \
	  \
	}
#define _LOG_E_BEGIN_AMPI_PROCESSING(tag,src,count) { ampi_beginProcessing(tag,src,count);}
#define _LOG_E_END_AMPI_PROCESSING()		    { ampi_endProcessing();}
#define _LOG_E_AMPI_MSG_SEND(tag,dest,count,size)   { ampi_msgSend(tag,dest,count,size);}
#endif
