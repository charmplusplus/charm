#ifndef _CONV_CPM_H
#define _CONV_CPM_H

#ifdef __cplusplus
extern "C" {
#endif

/* For CmiMsgHeaderSizeBytes */
#include "converse.h"

/************************************************************************
 *
 * CpmDestination
 *
 * A CpmDestination structure enables the user of the Cpm module to tell
 * the parameter-marshalling system what kind of envelope to put int the
 * message, and what to do with it after it has been filled.
 *
 ***********************************************************************/

typedef struct CpmDestinationStruct *CpmDestination;

typedef void *(*CpmSender)(CpmDestination, int, void *);

struct CpmDestinationStruct
{
  CpmSender sendfn;
  int envsize;
};

#define CpmPE(n) n
#define CpmALL (-1)
#define CpmOTHERS (-2)

CpmDestination CpmSend(int pe);
CpmDestination CpmMakeThread(int pe);
CpmDestination CpmMakeThreadSize(int pe, int size);
CpmDestination CpmEnqueueFIFO(int pe);
CpmDestination CpmEnqueueLIFO(int pe);
CpmDestination CpmEnqueueIFIFO(int pe, int prio);
CpmDestination CpmEnqueueILIFO(int pe, int prio);
CpmDestination CpmEnqueueBFIFO(int pe, int priobits, unsigned int *prioptr);
CpmDestination CpmEnqueueBLIFO(int pe, int priobits, unsigned int *prioptr);
CpmDestination CpmEnqueueLFIFO(int pe, int priobits, unsigned int *prioptr);
CpmDestination CpmEnqueueLLIFO(int pe, int priobits, unsigned int *prioptr);
CpmDestination CpmEnqueue(int pe,int qs,int priobits,unsigned int *prioptr);

/***********************************************************************
 *
 * CPM macros
 *
 *      CpmInvokable
 *      CpmDeclareSimple(x)
 *      CpmDeclarePointer(x)
 *
 * These macros expand into CPM ``declarations''.  The CPM ``declarations''
 * are actually C code that has no effect, but when the CPM scanner sees
 * them, it recognizes them and understands them as declarations.
 *
 **********************************************************************/

typedef void CpmInvokable;
typedef int CpmDeclareSimple1;
typedef int CpmDeclarePointer1;
#define CpmDeclareSimple(c) typedef CpmDeclareSimple1 CpmType_##c
#define CpmDeclarePointer(c) typedef CpmDeclarePointer1 CpmType_##c

/***********************************************************************
 *
 * Accessing a CPM message:
 *
 ***********************************************************************/

struct CpmHeader
{
  char convcore[CmiMsgHeaderSizeBytes];
  int envpos;
};
#define CpmEnv(msg) (((char *)msg)+(((struct CpmHeader *)msg)->envpos))
#define CpmAlign(val, type) ((val+sizeof(type)-1)&(~(sizeof(type)-1)))

/***********************************************************************
 *
 * Built-in CPM types
 *
 **********************************************************************/

CpmDeclareSimple(char);
#define CpmPack_char(v) do{}while(0)
#define CpmUnpack_char(v) do{}while(0)

CpmDeclareSimple(short);
#define CpmPack_short(v) do{}while(0)
#define CpmUnpack_short(v) do{}while(0)

CpmDeclareSimple(int);
#define CpmPack_int(v) do{}while(0)
#define CpmUnpack_int(v) do{}while(0)

CpmDeclareSimple(long);
#define CpmPack_long(v) do{}while(0)
#define CpmUnpack_long(v) do{}while(0)

CpmDeclareSimple(float);
#define CpmPack_float(v) do{}while(0)
#define CpmUnpack_float(v) do{}while(0)

CpmDeclareSimple(double);
#define CpmPack_double(v) do{}while(0)
#define CpmUnpack_double(v) do{}while(0)

typedef int CpmDim;
CpmDeclareSimple(CpmDim);
#define CpmPack_CpmDim(v) do{}while(0)
#define CpmUnpack_CpmDim(v) do{}while(0)

CpmDeclareSimple(Cfuture);
#define CpmPack_Cfuture(v) do{}while(0)
#define CpmUnpack_Cfuture(v) do{}while(0)

typedef char *CpmStr;
CpmDeclarePointer(CpmStr);
#define CpmPtrSize_CpmStr(v) (strlen(v)+1)
#define CpmPtrPack_CpmStr(p, v) (strcpy(p, v))
#define CpmPtrUnpack_CpmStr(v) do{}while(0)
#define CpmPtrFree_CpmStr(v) do{}while(0)

#ifdef __cplusplus
}
#endif

#endif
