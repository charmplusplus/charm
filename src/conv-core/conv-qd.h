#ifndef _CONV_QD_H
#define _CONV_QD_H

#ifdef __cplusplus
extern "C" {
#endif

struct ConvQdMsg;
struct ConvQdState;
typedef struct ConvQdMsg    *CQdMsg;
typedef struct ConvQdState  *CQdState;
typedef CcdVoidFn CQdVoidFn;

CpvExtern(CQdState, cQdState);

void CQdInit(void);
void CQdCpvInit(void);
void CQdCreate(CQdState, int);
void CQdProcess(CQdState, int);
int  CQdGetCreated(CQdState);
int  CQdGetProcessed(CQdState);
void CQdRegisterCallback(CQdVoidFn, void *);
void CmiStartQD(CQdVoidFn, void *);

#ifdef __cplusplus
}
#endif
#endif
