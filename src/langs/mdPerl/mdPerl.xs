#ifdef __cplusplus
extern "C" {
#endif
#include "EXTERN.h"
#include "perl.h"
#include "XSUB.h"
#include "converse.h"
#ifdef __cplusplus
}
#endif

static int handle;

void perlHandler(char *msg)
{
	int datasize, subnamesize;
	char *data, *subname;

	subname = msg+CmiMsgHeaderSizeBytes+4;
	subnamesize = strlen(subname)+1;
	memcpy(&datasize,msg+CmiMsgHeaderSizeBytes,4);
	if(datasize==0) {
		perl_call_pv(subname, G_DISCARD|G_NOARGS);
	} else {
		dSP;

		data=msg+CmiMsgHeaderSizeBytes+4+subnamesize;
		PUSHMARK(sp);
		XPUSHs(sv_2mortal(newSVpv(data, datasize)));
		PUTBACK ;

		perl_call_pv(subname, G_DISCARD);
	}
	return;
}

void user_main(int argc, char *argv[])
{
	/* to satisfy the linker */
}

MODULE = mdPerl		PACKAGE = mdPerl
PROTOTYPES: Enabled

void
mdInit()
	CODE:
    {
		char *args[1];
		args[0] = (char *) 0;
		ConverseInit(args);
		handle = CmiRegisterHandler(perlHandler);
	}

void
mdExit()
	CODE:
	{
		ConverseExit();
	}

void 
mdPrintf(str)
	char *str
	CODE:
	{
		CmiPrintf(str);
	}

void 
mdError(str)
	char *str
	CODE:
	{
		CmiError(str);
	}

int 
mdMyPe()
	CODE:
	RETVAL = CmiMyPe();
	OUTPUT:
	RETVAL

int 
mdNumPes()
	CODE:
	RETVAL = CmiNumPes();
	OUTPUT:
	RETVAL

void
mdCall(pe, subname, ...)
	int pe
	char *subname
	CODE:
	{
		char *msg, *data=0;
		STRLEN datasize = 0;
		int subnamesize=strlen(subname)+1;
		int msglen;

		if(items >2) {
			data = (char *) SvPV(ST(2), datasize);
		}
		msglen = CmiMsgHeaderSizeBytes+4+subnamesize+datasize;
		msg = (char *) CmiAlloc(msglen);
		CmiSetHandler(msg, handle);
		memcpy(msg+CmiMsgHeaderSizeBytes, &datasize, 4);
		strcpy(msg+CmiMsgHeaderSizeBytes+4, subname);
		if(data) {
			memcpy(msg+CmiMsgHeaderSizeBytes+4+subnamesize, data,datasize);
		}
		CmiSyncSendAndFree(pe, msglen, msg);
	}

void
mdScheduler(nmsg)
	int nmsg
	CODE:
	{
		CsdScheduler(nmsg);
	}

void
mdExitScheduler()
	CODE:
	{
		CsdExitScheduler();
	}

double
mdTimer()
	CODE:
	RETVAL = CmiTimer();
	OUTPUT:
	RETVAL
