/*
Pack/unpack to 4-byte network-byte-order integer/float.

Orion Sky Lawlor, olawlor@acm.org, 11/1/2001
 */
#ifndef __OSL_PUP_TONETWORK4_H
#define __OSL_PUP_TONETWORK4_H

#include "pup.h"

class PUP_toNetwork4_sizer : public PUP::er {
	int nBytes;
	virtual void bytes(void *p,int n,size_t itemSize,PUP::dataType t)
	{
		switch (t) {
		case PUP::Tchar: //Strings and bytes get copied as-is
		case PUP::Tbyte:
			nBytes+=n;
			break;
		default: //Everything else goes as a network int
			nBytes+=n*4;
		}
	}
 public:
	PUP_toNetwork4_sizer(void) :PUP::er(IS_SIZING) {nBytes=0;}
	int size(void) const {return nBytes;}

};

class PUP_toNetwork4_pack : public PUP::er {
	unsigned char *buf;
	inline void w(float f) {
		//Write floating-point number to stream.
		// Assumes "int4" and "float" type have the
		// same size and endianness (true on every current machine).
		union {
			float f;
			CMK_TYPEDEF_INT4 i;
		} mixer;
		mixer.f=f; //Put in as float
		w(mixer.i); //Take out as integer and write out
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
		buf=(unsigned char *)dest;
	}
};

class PUP_toNetwork4_unpack : public PUP::er {
	const unsigned char *buf;
	inline float read_float(void) {
		//Read floating-point number from stream.
		// Assumes "int4" and "float" type have the
		// same size and endianness (true on every current machine).
		union {
			float f;
			CMK_TYPEDEF_INT4 i;
		} mixer;
		mixer.i=read_int();
		return mixer.f;
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
		buf=(const unsigned char *)src;
	}
};


void PUP_toNetwork4_pack::bytes(void *p,int n,size_t itemSize,PUP::dataType t)
{
	int i;
	switch (t) {
	case PUP::Tchar: //Strings and bytes get copied as-is
	case PUP::Tbyte:
		memcpy(buf,p,n);
		buf+=n;
		break;
#define casePUP_toNetwork4_type(enumName,typeName,writeAs) \
	case PUP::enumName: \
	        for (i=0;i<n;i++) \
		  w((writeAs)( ((typeName *)p)[i] ));\
	        break
			
	casePUP_toNetwork4_type(Tfloat,float,float);
	casePUP_toNetwork4_type(Tdouble,double,float);
	casePUP_toNetwork4_type(Tuchar,unsigned char,int);
	case PUP::Tushort: //Fallthrough (no special treatment)
	casePUP_toNetwork4_type(Tshort,short,int);
	case PUP::Tuint: 
	casePUP_toNetwork4_type(Tint,int,int);
       	case PUP::Tulong: 
       	casePUP_toNetwork4_type(Tlong,long,int);
       	casePUP_toNetwork4_type(Tbool,bool,int);
       	case PUP::Tsync:
       		break; //Ignore
       	default: 
       		CkAbort("Unrecognized type passed to PUP_toNetwork4_pack!\n");
       	}
#undef casePUP_toNetwork4_type
}

void PUP_toNetwork4_unpack::bytes(void *p,int n,size_t itemSize,PUP::dataType t)
{
	int i;
	switch (t) {
	case PUP::Tchar: //Strings and bytes get copied as-is
	case PUP::Tbyte:
		memcpy(p,buf,n);
		buf+=n;
		break;
#define casePUP_toNetwork4_type(enumName,typeName,readAs) \
	case PUP::enumName: \
	        for (i=0;i<n;i++) \
		  ((typeName *)p)[i]=(typeName)read_##readAs();\
	        break
		
	casePUP_toNetwork4_type(Tfloat,float,float);
	casePUP_toNetwork4_type(Tdouble,double,float);
	casePUP_toNetwork4_type(Tuchar,unsigned char,int);
	case PUP::Tushort: //Fallthrough (no special treatment)
	casePUP_toNetwork4_type(Tshort,short,int);
	case PUP::Tuint: 
	casePUP_toNetwork4_type(Tint,int,int);
       	case PUP::Tulong: 
       	casePUP_toNetwork4_type(Tlong,long,int);
       	casePUP_toNetwork4_type(Tbool,bool,int);
       	case PUP::Tsync:
       		break; //Ignore
       	default: 
       		CkAbort("Unrecognized type passed to PUP_toNetwork4_unpack!\n");
       	}
}



#endif

