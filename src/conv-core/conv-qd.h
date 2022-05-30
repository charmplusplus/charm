#include "conv-config.h"

#if CMI_QD

#ifndef _CONV_QD_H
#define _CONV_QD_H

#ifdef __cplusplus
extern "C" {
#endif

struct ConvQdMsg;
struct ConvQdState;
typedef struct ConvQdMsg    *CQdMsg;
typedef struct ConvQdState  *CQdState;
typedef CcdCondFn CQdCondFn;

CpvExtern(CQdState, cQdState);

void CQdInit(void);
void CQdCpvInit(void);
void CQdCreate(CQdState, CmiInt8);
void CQdProcess(CQdState, CmiInt8);
CmiInt8 CQdGetCreated(CQdState);
CmiInt8 CQdGetProcessed(CQdState);
void CQdRegisterCallback(CQdCondFn, void *);
void CmiStartQD(CQdCondFn, void *);

#ifdef __cplusplus
}
#endif
#endif

#endif
