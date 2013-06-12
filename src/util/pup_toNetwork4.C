/*
Implementation of pup_toNetwork4.h

Orion Sky Lawlor, olawlor@acm.org, 2004/3/18
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "converse.h"
#include "pup.h"
#include "pup_toNetwork4.h"

/****************** toNetwork4 ********************
Pack/unpack to 4-byte network-byte-order integer/float.
This makes it easy to use platform-neutral on-the-wire types.
*/

void PUP_toNetwork4_sizer::bytes(void *p,int n,size_t itemSize,PUP::dataType t)
{
	switch (t) {
	case PUP::Tchar: //Strings and bytes get copied as-is
	case PUP::Tuchar:
	case PUP::Tbyte:
		nBytes+=n;
		break;
	default: //Everything else goes as a network int
		nBytes+=n*4;
	}
}

void PUP_toNetwork4_pack::bytes(void *p,int n,size_t itemSize,PUP::dataType t)
{
	int i;
	switch (t) {
	case PUP::Tchar: //Strings and bytes get copied as-is
	case PUP::Tuchar:
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
       		CmiAbort("Unrecognized type passed to PUP_toNetwork4_pack!\n");
       	}
#undef casePUP_toNetwork4_type
}

void PUP_toNetwork4_unpack::bytes(void *p,int n,size_t itemSize,PUP::dataType t)
{
	int i;
	switch (t) {
	case PUP::Tchar: //Strings and bytes get copied as-is
	case PUP::Tuchar:
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
       		CmiAbort("Unrecognized type passed to PUP_toNetwork4_unpack!\n");
       	}
}



