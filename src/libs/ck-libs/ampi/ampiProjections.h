#ifndef __AMPI_PROJECTIONS_H__
#define __AMPI_PROJECTIONS_H__
#include "ampi.h"
#ifdef __cplusplus
extern "C" {
#endif
void initAmpiProjections();
void closeAmpiProjections();
void ampi_beginProcessing(int tag,int src,int count);
void ampi_endProcessing(int tag);
void ampi_msgSend(int tag,int dest,int count,int size);
int ampi_registerFunc(char *funcName);
void ampi_beginFunc(int funcNo,MPI_Comm comm);
void ampi_endFunc(int funcNo,MPI_Comm comm);
#ifdef __cplusplus
}
#endif

#endif
