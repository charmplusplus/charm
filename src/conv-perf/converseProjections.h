
#ifndef __CONVERSE_PROJECTIONS_H__
#define __CONVERSE_PROJECTIONS_H__

#ifdef __cplusplus
extern "C" {
#endif

void msgSent(int destPE, int size);
void msgQueued();		//TODO
void msgRecvMC();		//TODO
void msgRecvSC();		//TODO
void handlerBegin(int handlerIdx);
void handlerEnd  (int handlerIdx);
void procIdle();
void procBusy();

#ifdef __cplusplus
}
#endif

#endif
