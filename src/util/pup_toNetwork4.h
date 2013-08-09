/*
Pack/unpack to 4-byte network-byte-order integer/float.

Orion Sky Lawlor, olawlor@acm.org, 11/1/2001
 */
#ifndef __OSL_PUP_TONETWORK4_H
#define __OSL_PUP_TONETWORK4_H

#include "pup.h"

class PUP_toNetwork4_sizer : public PUP::er {
	size_t nBytes;
	virtual void bytes(void *p,int n,size_t itemSize,PUP::dataType t);
 public:
	PUP_toNetwork4_sizer(void) :PUP::er(IS_SIZING) {nBytes=0;}
	size_t size(void) const {return nBytes;}

};

class PUP_toNetwork4_pack : public PUP::er {
	unsigned char *buf,*start;
    // NOTE: Need to use a union to force gcc compiler to consider pointer aliasing
	inline void w(float f) {
		//Write floating-point number to stream.
		// Assumes "int4" and "float" type have the
		// same size and endianness (true on every current machine).
	    union { float f; CMK_TYPEDEF_INT4 i; } uaw;
	    uaw.f=f;
	    w(uaw.i); //Take out as integer and write out
	}
	inline void w(int i) {
		//Write big-endian integer to output stream
		*buf++=(unsigned char)(i>>24); //High end first
		*buf++=(unsigned char)(i>>16);
		*buf++=(unsigned char)(i>>8);
		*buf++=(unsigned char)(i>>0);  //Low end last
	}

	virtual void bytes(void *p,int n,size_t itemSize,PUP::dataType t);
 public:
	PUP_toNetwork4_pack(void *dest) :PUP::er(IS_PACKING) {
		start=buf=(unsigned char *)dest;
	}
  	PUP_toNetwork4_pack(const PUP_toNetwork4_pack &p); //don't copy
  	void operator=(const PUP_toNetwork4_pack &p);	      // don't copy
	inline size_t size(void) const {return buf-start;}
};

class PUP_toNetwork4_unpack : public PUP::er {
	const unsigned char *buf,*start;
    // NOTE: Need to use a union to force gcc compiler to consider pointer aliasing
	inline float read_float(void) {
		//Read floating-point number from stream.
		// Assumes "int4" and "float" type have the
		// same size and endianness (true on every current machine).
	    union { float f; CMK_TYPEDEF_INT4 i; } uaw;
		uaw.i=read_int();
		return uaw.f;
	}
	inline int read_int(void) {
		//Read big-endian integer to output stream
		int ret=(buf[0]<<24)|(buf[1]<<16)|(buf[2]<<8)|(buf[3]);
		buf+=4;
		return ret;
	}

	virtual void bytes(void *p,int n,size_t itemSize,PUP::dataType t);
 public:
	PUP_toNetwork4_unpack(const void *src) :PUP::er(IS_UNPACKING) {
		start=buf=(const unsigned char *)src;
	}
  	PUP_toNetwork4_unpack(const PUP_toNetwork4_unpack &p); //don't copy
  	void operator=(const PUP_toNetwork4_unpack &p);	      // don't copy
	inline size_t size(void) const {return buf-start;}
};

#endif

