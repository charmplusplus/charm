CALLBOCAFTER

NAME
	CallBocAfter

SYNOPSIS
	#include "condsend.h"
	#include "chare.h"

	void CallBocAfter(deltaT,fn_ptr,bocNum)
	UNSIGNED_INT32 deltaT;
	FUNCTION_PTR   fn_ptr;
	int            bocNum;

DESCRIPTION
	This routine will call a boc access function after a specified time
 	time interval. The time will be relative to the current time when
   	CallBocAfter is entered (ie CallBocAfter gets the current time, adds
    deltaT to it and saves the information. Periodically, the system
	will compare the current time to this saved value. If it is higher (ie
    the scheduled time has passed),  then the boc access function will
	be called.
	
EXAMPLES

SEE ALSO
	SendMsgAfter

RESTRICTIONS

