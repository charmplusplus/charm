#ifndef  __COMPRESS_H_
#define __COMPRESS_H_

#ifdef __cplusplus
extern "C" {
#endif 
extern void compressFloatingPoint(void *src, void *dst, int s, int *compressSize, void *bData);
extern void decompressFloatingPoint(void *cData, void *dData, int s, int compressSize, void *bData);
#ifdef __cplusplus
}
#endif
#endif
