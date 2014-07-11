#include <ck.h>
#include "envelope.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "queueing.h"

#if 0
/* Queue message extraction calls: */
  CqsEnumerateQueue((Queue)CpvAccess(CsdSchedQueue), &schedQueue);
  FIFOQueue = CdsFifo_Enumerate((CdsFifo)CpvAccess(CmiLocalQueue));
  DQueue = CdsFifo_Enumerate(CpvAccess(debugQueue));
#endif

/************ Charm++ Message PUP *****************/
/* 
 pack_mode = 
                  0:  pup each field separately, allows for debugging;
                  1:  pup the message as a single array of bytes;
                  2:  try to handle the cross platform compatibility where
		      Converse header sizes are different. Works same as 0, 
		      but ignore the converse header. Cannot handle the case 
		      in heterogeneous platforms where data type length is 
		      different.
*/
void CkPupMessage(PUP::er &p,void **atMsg,int pack_mode) {
	UChar type;
	int size,prioBits,envSize;

	/* pup this simple flag so that we can handle the NULL msg */
	int isNull = (*atMsg == NULL);   // be overwritten when unpacking
	p(isNull);
	if (isNull) { *atMsg = NULL; return; }

	envelope *env=UsrToEnv(*atMsg);
	unsigned char wasPacked=0;
	p.comment("Begin Charm++ Message {");
	if (!p.isUnpacking()) {
		wasPacked=env->isPacked();
		if (0==wasPacked) //If it's not already packed...
		  CkPackMessage(&env); //Pack it
		type=env->getMsgtype();
		size=env->getTotalsize();
		prioBits=env->getPriobits();
		envSize=sizeof(envelope);
	}
	p(type);
	p(wasPacked);
	p(size);
	p(prioBits);
	p(envSize);
	int userSize=size-envSize-sizeof(int)*CkPriobitsToInts(prioBits);
	if (p.isUnpacking())
		env=_allocEnv(type,userSize,prioBits);
	if (pack_mode == 1) {
	  /*Pup entire header and message as raw bytes.*/
	  p((char *)env,size);
	} 
 	else if (pack_mode == 2) {
	    /*Pup header in detail and message separately.*/
	    /* note that it can be that sizeof(envelope) != envSize */
	    env->pup(p);
	    p((char*)env+sizeof(envelope),size-envSize);
 	}
	else 
	{ /*Pup each field separately, which allows debugging*/
	  p.comment("Message Envelope:");
	  env->pup(p);
	  p.comment("Message User Data:");
#if 0 /* Messages *should* be packed according to entry point: */
	  int ep=env->getEpIdx();
	  if (ep>0 && ep<_numEntries)
	    _entryTable[ep]->pupFn(p,*atMsg);
	  else
#endif
	    ((CkMessage *)*atMsg)->pup(p);
	  p.comment("} End Charm++ Message");
	}
	if (0==wasPacked) //Restore the packed-ness to previous state-- unpacked
	  CkUnpackMessage(&env);
	*atMsg=EnvToUsr(env);
}

void envelope::pup(PUP::er &p) {
	//message type, totalsize, and priobits are already pup'd (above)
	int convHeaderSize;
	if (!p.isUnpacking()) convHeaderSize = CmiReservedHeaderSize;
	p(convHeaderSize);
	//puping converse hdr hopefully not go beyond boundry
	p((char *)core,convHeaderSize);
	p(ref);
	//p((char *)&attribs,sizeof(attribs));
	p(attribs.msgIdx);
        p(attribs.mtype);
        int d;
        if (!p.isUnpacking()) d = attribs.queueing;
	//cppcheck-suppress uninitvar
        p|d;
        if (p.isUnpacking()) attribs.queueing = d;
        if (!p.isUnpacking()) d = attribs.isPacked;
        p|d;
        if (p.isUnpacking()) attribs.isPacked = d;
        if (!p.isUnpacking()) d = attribs.isUsed;
        p|d;
        if (p.isUnpacking()) attribs.isUsed = d; 
	p(epIdx);
	p(pe);
#if CMK_REPLAYSYSTEM || CMK_TRACE_ENABLED
	p(event);
#endif
	p((char*)getPrioPtr(),getPrioBytes());
	switch(getMsgtype()) {
	case NewChareMsg: case NewVChareMsg: 
	case ForChareMsg: case ForVidMsg: case FillVidMsg:
		p((char *)&(type.chare.ptr),sizeof(void *));
		p(type.chare.forAnyPe);
		break;
	case NodeBocInitMsg: case BocInitMsg: case ForNodeBocMsg: case ForBocMsg:
		p|type.group.g;
		p|type.group.rednMgr;
		p|type.group.dep;
		p|type.group.epoch;
		p|type.group.arrayEp;
		break;
	case ArrayEltInitMsg: case ForArrayEltMsg:
		p|type.array.arr;
                p|type.array.id;
		p(type.array.hopCount);
		p(type.array.ifNotThere);
		break;
	case RODataMsg:
		p(type.roData.count);
		break;
	case ROMsgMsg:
		p(type.roMsg.roIdx);
		break;
	default: /*No type-dependent fields to pack*/
		break;
	}
}

void CkMessage::pup(PUP::er &p) {
	//Default message pup: just copy user portion as bytes
	envelope *env=UsrToEnv((void *)this);
	int userSize=env->getTotalsize()-sizeof(envelope)-env->getPrioBytes();
	p((char *)this,userSize);
}
