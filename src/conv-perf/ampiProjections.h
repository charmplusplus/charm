#ifndef __AMPI_PROJECTIONS_H__
#define __AMPI_PROJECTIONS_H__

#ifdef __cplusplus
extern "C" {
#endif
void initAmpiProjections();
void closeAmpiProjections();
void ampi_beginProcessing(int tag,int src,int count);
void ampi_endProcessing();
void ampi_msgSend(int tag,int dest,int count,int size);
#ifdef __cplusplus
}
#endif

#endif
