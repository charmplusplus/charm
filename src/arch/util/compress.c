/*
 * =====================================================================================
 *
 *       Filename:  Compress.C
 *
 *    Description: Floating point compression/Decompression algorithm 
 *
 *        Version:  1.0
 *        Created:  09/02/2012 02:53:08 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
struct timeval tv;
#include <sys/time.h>

double get_clock()
{
       struct timeval tv; int ok;
       ok = gettimeofday(&tv, NULL);
       if (ok<0) { printf("gettimeofday error");  }
       return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

#define     COMPRESS 1 
#define     DEBUG  1
#define     CHAR_BIT 8
#define     FLOAT_BIT CHAR_BIT*sizeof(float)
#define     FLOAT_BYTE sizeof(float)

#define  COMPRESS_EXP 1


#if  COMPRESS_EXP

#define     SETBIT(dest, i)  (dest[i>>3]) |= (1 << (i&7) )
#define     TESTBIT(dest, i) ((dest[i>>3]) >>  (i&7)) & 1 

void compressFloatingPoint(void *src, void *dst, int s, int *compressSize, void *bData)
{
    int size = s/FLOAT_BYTE;
    float *source = (float*)src;
    float *dest = (float*)dst;
    float *baseData = (float*)bData;
    register unsigned int *bptr = (unsigned int*) baseData;
    register unsigned int  *uptr = (unsigned int *) source;
    register char *uchar;
    register int i;
#if DEBUG
    double t1 = get_clock();
#endif

#if !COMPRESS 
    memcpy(dest, source, size*sizeof(float)); 
    *compressSize = s;
#else
    // Is this the first time we're sending stuff to this node?
    if (baseData == NULL) {
        baseData = (float*)malloc(size*sizeof(float));
        memcpy(baseData, source, size*sizeof(float));
        memcpy(dest, source, size*sizeof(float)); 
        *compressSize = s;
    } else {
        // Create message to receive the compressed buffer.
        register unsigned char *cdst = (unsigned char*)dest; 
        register int _dataIndex = (size+7)/8;
        memset(cdst, 0, (size+7)/8 );
        for (i = 0; i < size; ++i) {
            // Bitmask everything but the exponents, then check if they match.
            unsigned int prevExp = bptr[i] & 0x7f800000;
            unsigned int currExp = uptr[i] & 0x7f800000;
            if (currExp != prevExp) {
                // If not, mark this exponent as "different" and store it to send with the message.
                SETBIT(cdst, i);
                memcpy(cdst+_dataIndex, &(uptr[i]), 4);
                _dataIndex += 4;
            }else
            {
                unsigned int ui = uptr[i];
                ui = (ui << 1) | (ui >> 31);
                memcpy(cdst+_dataIndex, &ui, 3);
                _dataIndex += 3;
            }

        }
        *compressSize = _dataIndex;
    }
#endif
#if DEBUG
    double t = get_clock()-t1;
    //printf("[%d] ===>done compressingcompressed size:(%d===>%d) (reduction:%d) ration=%f Timer:%f ms\n\n", CmiMyPe(), size*sizeof(float), *compressSize, (size*sizeof(float)-*compressSize), (1-(float)*compressSize/(size*sizeof(float)))*100, (CmiWallTimer()-startTimer)*1000);
    printf(" ===>done compressingcompressed size:(%d===>%d) (reduction:%d) ration=%f time=%d us\n", (int)(size*sizeof(float)), *compressSize, (int)(size*sizeof(float)-*compressSize), (1-(float)*compressSize/(size*sizeof(float)))*100, (int)(t*1000000000));
#endif
}

void decompressFloatingPoint(void *cData, void *dData, int s, int compressSize, void *bData) {
    int size = s/FLOAT_BYTE;
#if DEBUG
    double t1 = get_clock();
#endif
#if !COMPRESS
    memcpy(dData, cData, size*sizeof(float));
#else
    float *compressData = (float*)cData;
    float *baseData = (float*)bData;
    register unsigned int *decompressData =(unsigned int*)dData;
    register int _sdataIndex = (size+7)/8;
    register char *src = (char*)compressData;
    register int exponent;
    register unsigned int mantissa;
    register unsigned int *bptr = (unsigned int*)baseData;
    register int i;
    for(i=0; i<size; ++i)
    {
       if(TESTBIT(src, i)) // different
       {

           decompressData[i] = *((unsigned int*)(src+_sdataIndex));
           _sdataIndex += 4;
       }else        //same exponet
       {
           exponent = bptr[i]  & 0x7f800000;
           mantissa = *((unsigned int*)(src+_sdataIndex)) & 0x00FFFFFF;
           mantissa = (mantissa >> 1) | (mantissa << 31) ;
           mantissa |= exponent;
           decompressData[i] = mantissa;   
           _sdataIndex += 3;
       }
    }
#endif
#if DEBUG
    double t = get_clock()-t1;
    printf("done decompressing.....  orig size:%d\n time:%d us", (int)size, (int)(t*1000000000)) ;
#endif

}

#else


#define TESTBIT(data, b) (data>>(b)) & 1
#define SETBIT(data, index, bit) (data |= ((bit)<<(index)))

void printbitssimple(int n) {
    unsigned int i;
    i = 1<<(sizeof(n) * 8 - 1);
    
    while (i > 0) {
        if (n & i)
            printf("1");
        else
            printf("0");
        i >>= 1;
    }
}
void compressFloatingPoint(void *src, void *dst, int s, int *compressSize, void *bData)
{
    register unsigned int *dest = (unsigned int*)dst;
    register unsigned int *bptr = (unsigned int*) bData;
    register unsigned int  *uptr = (unsigned int *) src;
    int size = s/sizeof(float);
#if DEBUG
    double t1 = get_clock();
#endif
    
#if !COMPRESS 
    memcpy(dest, src, size*sizeof(float));
    *compressSize = s;
#else
    register unsigned int comp_data = 0;
    register int f_index = 0;
    register int i;
    register int j;
    register int b;
    register int zers;
    register unsigned int xor_data;
    bzero(dest, s);
    for (i = 0; i < size; ++i) {
        xor_data = (uptr[i])^(bptr[i]);
        zers = 0;
        //int value = xor_data;
        //printbitssimple(value);
        //printf("\n\n");
        b=FLOAT_BIT-1; 
        while(!TESTBIT(xor_data, b) && zers<15){
            zers++;
            b--;
        }
        //set the LZC 4 bits
        for(j=0; j<4; j++)
        {
            SETBIT(dest[(int)(f_index>>5)], (f_index&0x1f), TESTBIT(zers, j));
            f_index++;
        }
        while(b>=0)
        {
            SETBIT(dest[(f_index>>5)], f_index&0x1f, TESTBIT(xor_data, b));
            f_index++;
            b--;
        } 
    }
 /*   for (int k=0; k<f_index; k++) {
        printf("%e ",dest[k]);
    }
   */
    *compressSize = f_index/8;
    float compressRatio = (1-(float)(*compressSize)/s)*100;
    
#if DEBUG
    double t = get_clock()-t1;
    printf("===>done compressing compressed size:(%d===>%d) (reduction:%d) ration=%f Timer:%d us\n\n", (int)(size*sizeof(float)), *compressSize, (int)((size*sizeof(float)-*compressSize)), (1-(float)*compressSize/(size*sizeof(float)))*100, (int)(t*1000000000));
#endif

#endif
}

void decompressFloatingPoint(void *cData, void *dData, int s, int compressSize, void *bData) {
    int size = s/sizeof(float);
#if DEBUG
    double t1 = get_clock();
#endif
#if !COMPRESS
    memcpy(dData, cData, size*sizeof(float));
#else
    register unsigned int *compressData = (unsigned int*)cData;
    register unsigned int *decompressData = (unsigned int*)dData;
    register unsigned int *baseData = (unsigned int*)bData;
    bzero(decompressData, s);
    register int index;
    register unsigned int xor_data;
    register int data = 0;
    register int d_index=0;
    register int compp = 0;
    register int i;
    register int j;
    register int f;
    for (i=0; i<size; i++) {
        index = FLOAT_BIT-1;
        data = 0;
        //read 4 bits and puts index acccordingly
        for (f=0; f<4; f++,compp++) {
            if(TESTBIT(compressData[(int)(compp>>5)], (compp&0x1f))){
                for (j=0; j < (1<<f); j++) {
                    SETBIT(data, index, 0);
                    index--;
                }
            }
        }
        while(index>=0){
            SETBIT(data, index, TESTBIT(compressData[(int)(compp>>5)], (compp&0x1f)));
            index--; compp++;
        }
        xor_data = data^(baseData[i]);
        decompressData[i] = xor_data;
    }

#if DEBUG
    double t = get_clock()-t1;
    printf("done decompressing.....  orig size:%d time:%d us \n", size, (int)(t*1000000000));
#endif

#endif
}

#endif
