
#ifndef __CONVERSE_EVENTS_H__
#define __CONVERSE_EVENTS_H__

/* Language ID */
#define _CONVERSE_LANG_ID	0	// language ID for converse

/* Event IDs */
#define _E_MSG_SENT			0
#define _E_MSG_QUEUED		1 	// DOUBT Queued where
#define _E_MSG_RECV_MC		2	// Message received in machine layer
#define _E_MSG_RECV_SC		3	// Message received in scheduler
#define _E_HANDLER_BEGIN 	4 
#define _E_HANDLER_END		5 
#define _E_PROC_IDLE		6	// Processor goes idle 
#define _E_PROC_BUSY		7  	// Processor goes busy 

/* Trace Macros */
//TODO
#define REGISTER_CONVERSE
#define LOG_CONVERSE_EVENT

#endif
