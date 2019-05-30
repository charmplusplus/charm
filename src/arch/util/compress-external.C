#include "zlib-compress.C"
//#include "quicklz-compress.C"
#include "lz4.h"
#define     CMODE_NOCOMPRESS 0
#define     CMODE_ZLIB  1
//#define     CMODE_QUICKLZ 2
#define     CMODE_LZ4     3
#define OUT_PLACE 1

//quicklz crashes 
#include <sys/time.h>

/*
struct timeval tv;
double get_clock()
{
       struct timeval tv; int ok;
       ok = gettimeofday(&tv, NULL);
       if (ok<0) { CmiPrintf("gettimeofday error");  }
       return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}
*/
static int compress_mode = CMODE_ZLIB;
//static int compress_mode = CMODE_LZ4;
static int initDone=0;
void initCompress(void)
{
    switch(compress_mode)
    {
        case CMODE_ZLIB:    zlib_init();    break;
        //case CMODE_QUICKLZ: quicklz_init(); break;
        case CMODE_LZ4: break;
    }

}
//#define DEBUG  1 

void compressZlib(void *src, void *dst, int size, int *compressSize, void *bData)
{
#if DEBUG
    double t1 = get_clock();
#endif
    char *src_ptr = (char*)src;
    char *bdata_ptr = (char*)bData;
    if(initDone==0)
    {
        initCompress();
        initDone = 1;
    }
    auto my_xor = (char *)malloc(size);
    int i;
    for(i=0; i<size; i++)
        my_xor[i] = (src_ptr[i])^(bdata_ptr[i]);
    zlib_compress(my_xor, dst, size, compressSize);
#if DEBUG
    double t = get_clock()-t1;
    printf("+%d external compression done compressing(%d===>%d) (reduction:%d) ration=%f time=%d us\n", compress_mode, (int)(size*sizeof(char)), *compressSize, (int)(size*sizeof(char)-*compressSize), (1-(float)*compressSize/(size*sizeof(char)))*100, (int)(t*1000000));
#endif
}

void decompressZlib(void *cData, void *dData, int size, int compressSize, void *bData) {

#if DEBUG
    double t1 = get_clock();
#endif
    char *my_xor=(char*)malloc(size);
    zlib_decompress(cData, my_xor, compressSize, size);
    int i;
    char *dptr = (char*)dData;
    char *bptr = (char*)bData; 
    for(i=0; i<size; i++)
        dptr[i] = (bptr[i])^(my_xor[i]);
    free(my_xor);
#if DEBUG
    double t = get_clock()-t1;
    printf("------done decompressing.....  orig size:%d time:%d us \n", (int)size, (int)(t*1000000)) ;
#endif
}

void compressLz4(void *src, void *dst, int size, int *compressSize, void *bData)
{
#if DEBUG
    double t1 = get_clock();
#endif
    char *src_ptr = (char*)src;
    char *bdata_ptr = (char*)bData;
    if(initDone==0)
    {
        initCompress();
        initDone = 1;
    }
    auto my_xor = (char *)malloc(size);
    int i;
    for(i=0; i<size; i++)
        my_xor[i] = (src_ptr[i])^(bdata_ptr[i]);

    *compressSize = LZ4_compress_default(my_xor, (char *)dst, size, LZ4_compressBound(size));

#if DEBUG
    double t = get_clock()-t1;
    printf("+%d external compression done compressing(%d===>%d) (reduction:%d) ration=%f time=%d us\n", compress_mode, (int)(size*sizeof(char)), *compressSize, (int)(size*sizeof(char)-*compressSize), (1-(float)*compressSize/(size*sizeof(char)))*100, (int)(t*1000000));
#endif
}

void decompressLz4(void *cData, void *dData, int size, int compressSize, void *bData) {

#if DEBUG
    double t1 = get_clock();
#endif
    char *my_xor=(char*)malloc(size);

    if (LZ4_decompress_safe((char *)cData, my_xor, compressSize, size) < 0)
        CmiAbort("decode fails\n");

    int i;
    char *dptr = (char*)dData;
    char *bptr = (char*)bData; 
    for(i=0; i<size; i++)
        dptr[i] = (bptr[i])^(my_xor[i]);
    free(my_xor);
#if DEBUG
    double t = get_clock()-t1;
    printf("------done decompressing.....  orig size:%d time:%d us \n", (int)size, (int)(t*1000000)) ;
#endif
}
