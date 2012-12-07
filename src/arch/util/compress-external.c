#include "zlib-compress.c"
#include "quicklz-compress.c"
#include "lz4.h"
static int compress_mode;
#define     CMODE_NOCOMPRESS 0
#define     CMODE_ZLIB  1
#define     CMODE_QUICKLZ 2
#define     CMODE_LZ4     3
#define OUT_PLACE 1

//quicklz crashes 
struct timeval tv;
#include <sys/time.h>

double get_clock()
{
       struct timeval tv; int ok;
       ok = gettimeofday(&tv, NULL);
       if (ok<0) { CmiPrintf("gettimeofday error");  }
       return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

static int compress_mode = CMODE_LZ4;
static int initDone=0;
void initCompress()
{
    switch(compress_mode)
    {
        case CMODE_ZLIB:    zlib_init();    break;
        case CMODE_QUICKLZ: quicklz_init(); break;
        case CMODE_LZ4:     lz4_init();     break;
    }

}
#define DEBUG  1 

void compressChar(void *src, void *dst, int size, int *compressSize, void *bData)
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
    void  (*compressFn) (void*, void*, int, int*);
    switch(compress_mode)
    {
     case CMODE_ZLIB:    compressFn = zlib_compress; break;
     case CMODE_QUICKLZ: compressFn = quicklz_compress; break;
     case CMODE_LZ4:     compressFn = lz4_wrapper_compress; break;
    }
    char *xor=malloc(size);
    int i;
    for(i=0; i<size; i++)
        xor[i] = (src_ptr[i])^(bdata_ptr[i]);       
    compressFn(xor, dst, size, compressSize);
#if DEBUG
    double t = get_clock()-t1;
    printf(" external compression done compressing(%d===>%d) (reduction:%d) ration=%f time=%d us\n", (int)(size*sizeof(char)), *compressSize, (int)(size*sizeof(char)-*compressSize), (1-(float)*compressSize/(size*sizeof(char)))*100, (int)(t*1000000));
#endif
}

void decompressChar(void *cData, void *dData, int size, int compressSize, void *bData) {

#if DEBUG
    double t1 = get_clock();
#endif
    int (*decompressFn)(void *, void *, int, int);
    switch(compress_mode)
    {
    case CMODE_ZLIB: decompressFn = zlib_decompress; break;
    case CMODE_QUICKLZ: decompressFn = quicklz_decompress; break;
    case CMODE_LZ4: decompressFn = lz4_wrapper_decompress; break;
    }
    char *xor=(char*)malloc(size);
    decompressFn(cData, xor, compressSize, size);
    int i;
    char *dptr = (char*)dData;
    char *bptr = (char*)bData; 
    for(i=0; i<size; i++)
        dptr[i] = (bptr[i])^(xor[i]); 
    free(xor);
#if DEBUG
    double t = get_clock()-t1;
    printf("------done decompressing.....  orig size:%d time:%d us \n", (int)size, (int)(t*1000000)) ;
#endif
}
