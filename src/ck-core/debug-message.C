/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

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
 fast_and_dirty = 0:  pup directly according to the size;
                  1:  pup each field separately, allows for debugging;
                  2:  try to handle the cross platform compatibility where
		      Converse header sizes are different. Works same as 0, 
		      but ignore the converse header. Cannot handle the case 
		      in heterogeneous platforms where data type length is 
		      different.
*/
void CkPupMessage(PUP::er &p,void **atMsg,int fast_and_dirty) {
	UChar type;
	int size,prioBits,envSize;
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
	int userSize=size-envSize-sizeof(int)*PW(prioBits);
	if (p.isUnpacking())
		env=_allocEnv(type,userSize,prioBits);
	if (fast_and_dirty == 1) {
	  /*Pup entire header and message as raw bytes.*/
	  p((void *)env,size);
	} 
 	else if (fast_and_dirty == 2) {
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
	p((void *)core,convHeaderSize);
	p(ref);
	p((void *)&attribs,sizeof(attribs));
	p(epIdx);
	p(pe);
	p(event);
	p(getPrioPtr(),getPrioBytes());
	switch(getMsgtype()) {
	case NewChareMsg: case NewVChareMsg: 
	case ForChareMsg: case ForVidMsg: case FillVidMsg:
		p((void *)&(type.chare.ptr),sizeof(void *));
		p(type.chare.forAnyPe);
		break;
	case BocInitMsg: case ForNodeBocMsg:
		p|type.group.g;
		break;
	case ForBocMsg:
		p|type.array.loc;
		p|type.array.arr;
		p(type.array.hopCount);
		p(type.array.epIdx);
		p(type.array.listenerData,CK_ARRAYLISTENER_MAXLEN);
		p(type.array.srcPe);
		p(type.array.index.nInts);
		p(type.array.index.index,CK_ARRAYINDEX_MAXLEN);
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
	p((void *)this,userSize);
}

void CkMessage::setImmediate(CmiBool i) {
	envelope *env=UsrToEnv((void *)this);
	env->setImmediate(i);
}
