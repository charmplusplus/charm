/*
Stand-in for ccs-builtins.C in builds that have no Converse debugger.

The built-in CCS handlers (memory statistics, the CWeb performance feed, and
the parallel debugger's object browser) are written against Converse internals
that Reconverse does not have: the CpdList accessor machinery in debug-conv.h
and the Converse scheduler queue in queueing.h, whose Queue type conflicts with
Reconverse's own.

None of that is needed to use CCS as a control channel, which is what the
shrink/expand path does: it registers handlers and replies to requests, all of
which lives in conv-ccs.C. So this file supplies the one symbol conv-ccs.C
needs from the builtins, plus PUP_fmt, which is unrelated to CCS but happens to
live in ccs-builtins.C and is used by pup_c.C. Building the real
ccs-builtins.C against Reconverse is a separate job, and only worth doing if
someone wants the debugger there.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "converse.h"
#include "pup.h"
#include "conv-ccs.h"
#include "ccs-builtins.h"

#if CMK_CCS_AVAILABLE
void ccs_getinfo(char *msg);
#endif

void CcsBuiltinsInit(char **argv)
{
  (void)argv;
#if CMK_CCS_AVAILABLE
  /* Every CCS client asks for this first: CcsConnect uses it to learn the node
     and PE layout, and gets no reply at all if it is missing. The rest of the
     builtins (kill-port, kill-PE, CWeb, the debugger's object lists) are not
     registered here; see the note above. */
  CcsRegisterHandler("ccs_getinfo", (CmiHandler)ccs_getinfo);
#endif
}

void PUP_fmt::fieldHeader(typeCode_t typeCode,int nItems) {
    // Compute and write intro byte:
    lengthLen_t ll;
    if (nItems==1) ll=lengthLen_single;
    else if (nItems<256) ll=lengthLen_byte;
    else ll=lengthLen_int;
    // CmiPrintf("Intro byte: l=%d t=%d\n",(int)ll,(int)typeCode);
    byte intro=(((int)ll)<<4)+(int)typeCode;
    p(intro);
    // Compute and write length:
    switch(ll) {
    case lengthLen_single: break; // Single item
    case lengthLen_byte: {
        byte l=nItems;
        p(l);
        } break;
    case lengthLen_int: {
        p(nItems); 
        } break;
    case lengthLen_long: CmiAbort("Should not have reached here!"); break;
    };
}

void PUP_fmt::comment(const char *message) {
	size_t nItems=strlen(message);
	fieldHeader(typeCode_comment,nItems);
	p((char *)message,nItems);
}
void PUP_fmt::synchronize(unsigned int m) {
	fieldHeader(typeCode_sync,1);
	p(m);
}

void PUP_fmt::pup_buffer(void *&ptr,size_t n,size_t itemSize,PUP::dataType t) {
  bytes(ptr, n, itemSize, t);
}

void PUP_fmt::pup_buffer(void *&ptr,size_t n, size_t itemSize, PUP::dataType t, std::function<void *(size_t)> allocate, std::function<void (void *)> deallocate){
  bytes(ptr, n, itemSize, t);
}

void PUP_fmt::bytes(void *ptr,size_t n,size_t itemSize,PUP::dataType t) {
	if(itemSize > INT_MAX || n > INT_MAX || itemSize*n > INT_MAX)
		CmiAbort("Ccs does not support messages greater than INT_MAX...\n");
	switch(t) {
	case PUP::Tchar:
	case PUP::Tuchar:
	case PUP::Tbyte:
		fieldHeader(typeCode_byte,n);
		p.bytes(ptr,n,itemSize,t);
		break;
	case PUP::Tshort: case PUP::Tint:
	case PUP::Tushort: case PUP::Tuint:
	case PUP::Tbool:
		fieldHeader(typeCode_int,n);
		p.bytes(ptr,n,itemSize,t);
		break;
	// treat "long" and "pointer" as 8-bytes, in conformity with pup_toNetwork.C
	case PUP::Tlong: case PUP::Tlonglong:
	case PUP::Tulong: case PUP::Tulonglong:
		fieldHeader(typeCode_long,n);
		p.bytes(ptr,n,itemSize,t);
		break;
	case PUP::Tfloat:
		fieldHeader(typeCode_float,n);
		p.bytes(ptr,n,itemSize,t);
		break;
	case PUP::Tdouble: case PUP::Tlongdouble:
		fieldHeader(typeCode_double,n);
		p.bytes(ptr,n,itemSize,t);
		break;
    case PUP::Tpointer:
        fieldHeader(typeCode_pointer,n);
        p.bytes(ptr,n,itemSize,t);
        break;
	default: CmiAbort("Unrecognized type code in PUP_fmt::bytes");
	};
}
