SENDMSGAFTER

NAME
	SendMsgAfter

SYNOPSIS
	#include "condsend.h"
	#include "chare.h"

	void SendMsgAfter(deltaT, entry, msgToSend, size, pChareID)
	UNSIGNED_INT32  deltaT;
	int             entry;
	void            *msgToSend;
	int             size;
	ChareIDType     *pChareID;

DESCRIPTION

	This function allows the user to send a message at some point in the
 	future. Time is measured relative to the beginning of the SendMsgAfter
    call (ie the SendMsgAfter routine gets the current time, adds deltaT
    to it and saves this value). Periodically, the system will compare the
    current time to the saved value. If the current time is greater (ie
    it is past the time requested by the SendMsgAfter call), then the message
    will be sent, just like a regular SendMsg call.

EXAMPLES

SEE ALSO
	CallBocAfter

RESTRICTION

