#ifndef __AMPI_PROJECTIONS_H__
#define __AMPI_PROJECTIONS_H__
#include "ampi.h"
#ifdef __cplusplus
extern "C" {
#endif
void initAmpiProjections(void);
void closeAmpiProjections(void);
void ampi_beginProcessing(int rank, int src, int count);
void ampi_endProcessing(int rank);
void ampi_msgSend(int tag,int dest,int count,int size);
int ampi_registerFunc(char *funcName);
void ampi_beginFunc(int funcNo,MPI_Comm comm);
void ampi_endFunc(int funcNo,MPI_Comm comm);
#if CMK_BIGSIM_CHARM
void BgPrintf(const char *str);
#endif
#ifdef __cplusplus
}
#endif

#endif
