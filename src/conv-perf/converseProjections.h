
#ifndef __CONVERSE_PROJECTIONS_H__
#define __CONVERSE_PROJECTIONS_H__

#ifdef __cplusplus
extern "C" {
#endif

void converse_msgSent(int destPE, int size);
void converse_msgQueued(void);		/* TODO */
void converse_msgRecvMC(void);		/* TODO */
void converse_msgRecvSC(void);		/* TODO */
void converse_handlerBegin(int handlerIdx);
void converse_handlerEnd  (int handlerIdx);


#ifdef __cplusplus
}
#endif

#endif
