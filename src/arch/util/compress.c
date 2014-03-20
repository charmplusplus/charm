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
 *         Author:  Yanhua Sun 
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
#include <assert.h>

//#define USE_SSE 1 

#if USE_SSE
#include <smmintrin.h>
#endif

double get_clock()
{
       struct timeval tv; int ok;
       ok = gettimeofday(&tv, NULL);
       if (ok<0) { CmiPrintf("gettimeofday error");  }
       return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

#define     COMPRESS 1 
//#define     DEBUG  1
#define     CHAR_BIT 8
#define     FLOAT_BIT CHAR_BIT*sizeof(float)
#define     FLOAT_BYTE sizeof(float)

#define  COMPRESS_EXP 1

#if  COMPRESS_EXP
#define     SETBIT(dest, i)  (dest[i>>3]) |= (1 << (i&7) )
#define     TESTBIT(dest, i) ((dest[i>>3]) >>  (i&7)) & 1 
#define     SETBIT11(dest, i)  (dest[(i)>>3]) |= (3 << ((i)&7) )
#define     TESTBIT11(dest, i) ((dest[(i)>>3]) >>  ((i)&7)) & 0x3l 

#else

#define TESTBIT(data, b) (data>>(b)) & 1
#define SETBIT(data, index, bit) (data |= ((bit)<<(index)))
#endif

/** compress char is the general algorithm   for any data*/
void compressChar(void *src, void *dst, int size, int *compressSize, void *bData)
{
    register char *source = (char*)src;
    register char *dest = (char*)dst;
    register char *baseData = (char*)bData;
    register int i;
#if DEBUG
    double t1 = get_clock();
#endif

#if !COMPRESS 
    memcpy(dest, source, size*sizeof(char)); 
    *compressSize = size;
#else
    register int _dataIndex = (size+7)/8;
    memset(dest, 0, (size+7)/8 );
    for (i = 0; i < size&&_dataIndex<size; ++i) {
        // Bitmask everything but the exponents, then check if they match.
         char xor_d =  source[i] ^ baseData[i];
         short different= xor_d  & 0xff;
         if (different) {
             // If not, mark this exponent as "different" and store it to send with the message.
             dest[_dataIndex] = source[i];
             _dataIndex += 1;
         }else
         {
             SETBIT(dest, i);
         }

    }
    *compressSize = _dataIndex;
#endif
#if DEBUG
    double t = get_clock()-t1;
    printf(" +++++CHAR done compressing(%d===>%d) (reduction:%d) ration=%f time=%d us\n", (int)(size*sizeof(char)), *compressSize, (int)(size*sizeof(char)-*compressSize), (1-(float)*compressSize/(size*sizeof(char)))*100, (int)(t*1000000));
#endif
}

void decompressChar(void *cData, void *dData, int size, int compressSize, void *bData) {
#if DEBUG
    double t1 = get_clock();
#endif
#if !COMPRESS
    memcpy(dData, cData, size*sizeof(char));
#else
    char *compressData = (char*)cData;
    char *baseData = (char*)bData;
    register char *decompressData =(char*)dData;
    register int sdataIndex = (size+7)/8;
    register char *src = (char*)compressData;
    register int i;
    for(i=0; i<size; ++i)
    {
       if(TESTBIT(src, i)) // same 
       {
           decompressData[i] = baseData[i];
       }else        //different exponet
       {
           decompressData[i] = compressData[sdataIndex];   
           sdataIndex += 1;
       }
    }
#endif
#if DEBUG
    double t = get_clock()-t1;
    printf("------CHAR done decompressing.....  orig size:%d time:%d us \n", (int)size, (int)(t*1000000)) ;
#endif

}

#if  COMPRESS_EXP

#if USE_SSE
void compressFloatingPoint(void *src, void *dst, int s, int *compressSize, void *bData)
{
    int size = s/FLOAT_BYTE;
    float *source = (float*)src;
    float *dest = (float*)dst;
    float *baseData = (float*)bData;
    register unsigned int *bptr = (unsigned int*) baseData;
    register unsigned int  *uptr = (unsigned int *) source;
    register char *uchar;
    register int i, j;
#if DEBUG
    double t1 = get_clock();
#endif

#if !COMPRESS 
    memcpy(dest, source, size*sizeof(float)); 
    *compressSize = s;
#else
    assert(baseData != NULL);
    {
        // Create message to receive the compressed buffer.
        register unsigned char *cdst = (unsigned char*)dest; 
        register int _dataIndex = (size+7)/8;
        register unsigned int diff;
	memset(cdst, 0, (size+7)/8 );
        
        register const __m128i* b_ptr = (__m128i*)bptr;
        register const __m128i* u_ptr = (__m128i*)uptr;
        
        register __m128i xmm_f = _mm_set1_epi32(0xFF000000);
        for (i = 0; i < size; i+=4) {
            // Bitmask everything but the exponents, then check if they match.
            __m128i xmm_b = _mm_load_si128(b_ptr);
            __m128i xmm_u = _mm_load_si128(u_ptr);
            __m128i xmm_d = _mm_xor_si128(xmm_b, xmm_u);     //  XOR  4 32-bit words
            xmm_d = _mm_and_si128(xmm_d, xmm_f);
            
            if (_mm_extract_epi32(xmm_d, 0)) {
                SETBIT(cdst, i);
                memcpy(cdst+_dataIndex, &(uptr[i]), 4);
                _dataIndex += 4;
            }
            else{
                memcpy(cdst+_dataIndex, &(uptr[i]), 3);
                _dataIndex += 3;
            }
            if (_mm_extract_epi32(xmm_d, 1)) {
                SETBIT(cdst, i+1);
                memcpy(cdst+_dataIndex, &(uptr[i+1]), 4);
                _dataIndex += 4;
            }else{
                memcpy(cdst+_dataIndex, &(uptr[i+1]), 3);
                _dataIndex += 3;
            }
            if (_mm_extract_epi32(xmm_d, 2)) {
                SETBIT(cdst, i+2);
                memcpy(cdst+_dataIndex, &(uptr[i+2]), 4);
                _dataIndex += 4;
            }else{
                memcpy(cdst+_dataIndex, &(uptr[i+2]), 3);
                _dataIndex += 3;
            }
            if (_mm_extract_epi32(xmm_d, 3)) {
                SETBIT(cdst, i+3);
                memcpy(cdst+_dataIndex, &(uptr[i+3]), 4);
                _dataIndex += 4;
            }else{
                memcpy(cdst+_dataIndex, &(uptr[i+3]), 3);
                _dataIndex += 3;
            }
            ++b_ptr;
            ++u_ptr;
        }
        *compressSize = _dataIndex;
    }
#endif
#if DEBUG
    double t = get_clock()-t1;
    printf(" ===>floating compare done compressingcompressed size:(%d===>%d) (reduction:%d) ration=%f time=%d us \n", (int)(size*sizeof(float)), *compressSize, (int)(size*sizeof(float)-*compressSize), (1-(float)*compressSize/(size*sizeof(float)))*100, (int)(t*1000000));
#endif
}

#else

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
    assert(baseData != NULL);
    // Is this the first time we're sending stuff to this node?
    {
        // Create message to receive the compressed buffer.
        register unsigned char *cdst = (unsigned char*)dest; 
        register int _dataIndex = (size+7)/8;
        register unsigned int diff;
	memset(cdst, 0, (size+7)/8 );
        for (i = 0; i < size; ++i) {
            // Bitmask everything but the exponents, then check if they match.
        diff = (bptr[i] ^ uptr[i]) & 0xff000000 ;    
	if (diff) {
                // If not, mark this exponent as "different" and store it to send with the message.
                SETBIT(cdst, i);
                memcpy(cdst+_dataIndex, &(uptr[i]), 4);
                _dataIndex += 4;
            }else
            {
                memcpy(cdst+_dataIndex, &(uptr[i]), 3);
                _dataIndex += 3;
            }

        }
        *compressSize = _dataIndex;
    }
#endif
#if DEBUG
    double t = get_clock()-t1;
    CmiPrintf(" ===> FLOATING done compressingcompressed size:(%d===>%d) (reduction:%d) ration=%f time=%d us\n", (int)(size*sizeof(float)), *compressSize, (int)(size*sizeof(float)-*compressSize), (1-(float)*compressSize/(size*sizeof(float)))*100, (int)(t*1000000));
#endif
}

#endif

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
	   exponent = bptr[i]  & 0xff000000;
           mantissa = *((unsigned int*)(src+_sdataIndex)) & 0x00FFFFFF;
           mantissa |= exponent;
           decompressData[i] = mantissa;
           _sdataIndex += 3;
	}
    }
#endif
#if DEBUG
    double t = get_clock()-t1;
    //CmiPrintf("--- FLOATING done decompressing.....  orig size:%d\n time:%d us", (int)size, (int)(t*1000000)) ;
#endif

}

#else
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
    memset(dest, 0, s);
    for (i = 0; i < size; ++i) {
        xor_data = (uptr[i])^(bptr[i]);
        zers = 0;
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
    *compressSize = f_index/8;
    float compressRatio = (1-(float)(*compressSize)/s)*100;
    
#if DEBUG
    double t = get_clock()-t1;
    CmiPrintf("===>[floating point lzc]done compressing compressed size:(%d===>%d) (reduction:%d) ration=%f Timer:%d us\n\n", (int)(size*sizeof(float)), *compressSize, (int)((size*sizeof(float)-*compressSize)), (1-(float)*compressSize/(size*sizeof(float)))*100, (int)(t*1000000));
#endif

#endif
}

void decompressFloatingPoint(void *cData, void *dData, int s, int compressSize, void *bData) {
    int size = s/sizeof(float);
#if DEBUG
    double t1 = get_clock();
    if(CmiMyPe() == 5)
        CmiPrintf("[%d] starting decompressing \n", CmiMyPe());
#endif
#if !COMPRESS
    memcpy(dData, cData, size*sizeof(float));
#else
    register unsigned int *compressData = (unsigned int*)cData;
    register unsigned int *decompressData = (unsigned int*)dData;
    register unsigned int *baseData = (unsigned int*)bData;
    memset(decompressData, 0, s);
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
    if(CmiMyPe() == 5)
        CmiPrintf("[%d] done decompressing.....  orig size:%d time:%d us \n", CmiMyPe(), size, (int)(t*1000000));
#endif

#endif
}

#endif


/***************************
 *
 * algorithms to compress doubles
 * *****************/

#define DOUBLE_BYTE sizeof(double)
#define BITS_DOUBLE sizeof(double)*8

#if COMPRESS_EXP

void compressDouble(void *src, void *dst, int s, int *compressSize, void *bData)
{
    int size = s/DOUBLE_BYTE;
    double *source = (double*)src;
    double *dest = (double*)dst;
    double *baseData = (double*)bData;
    register unsigned long *bptr = (unsigned long*) baseData;
    register unsigned long *uptr = (unsigned long*) source;
    register char *uchar;
    register int i;
#if DEBUG
    double t1 = get_clock();
#endif

#if !COMPRESS 
    memcpy(dest, source, s); 
    *compressSize = s;
#else
    assert(baseData != NULL);
    // Is this the first time we're sending stuff to this node?
    {
        *compressSize = s;
        // Create message to receive the compressed buffer.
        register unsigned char *cdst = (unsigned char*)dest; 
        register int _dataIndex = (2*size+7)/8;
        memset(cdst, 0, (2*size+7)/8 );
        for (i = 0; i < size; ++i) {
            // Bitmask everything but the exponents, then check if they match.
            unsigned long xord = bptr[i] ^ uptr[i];
            unsigned long eight = xord &  0xff00000000000000;
            unsigned long sixteen = xord &  0xffff000000000000;
            if(sixteen == 0l)    //00
            {
                unsigned long ui = uptr[i];
                memcpy(cdst+_dataIndex, &ui, 6);
                _dataIndex += 6;
            }
            else if(eight == 0l)//01
            {
                SETBIT(cdst, i<<1);
                unsigned long ui = uptr[i];
                memcpy(cdst+_dataIndex, &ui, 7);
                _dataIndex += 7;
            }else   //11
            {
                SETBIT11(cdst, i<<1);
                unsigned long ui = uptr[i];
                memcpy(cdst+_dataIndex, &ui, 8);
                _dataIndex += 8;
            }
        }
        *compressSize = _dataIndex;
    }
#endif
#if DEBUG
    double t = get_clock()-t1;
    printf(" ===>[double lzc] done compressingcompressed size:(%d===>%d) (reduction:%d) ration=%f time=%d us\n", (int)(size*sizeof(double)), *compressSize, (int)(size*sizeof(double)-*compressSize), (1-(double)*compressSize/(size*sizeof(double)))*100, (int)(t*1000000));
#endif
}

void decompressDouble(void *cData, void *dData, int s, int compressSize, void *bData) {
    int size = s/DOUBLE_BYTE;
#if DEBUG
    double t1 = get_clock();
#endif
#if !COMPRESS
    memcpy(dData, cData, s);
#else
    double *compressData = (double*)cData;
    double *baseData = (double*)bData;
    register unsigned long *decompressData =(unsigned long*)dData;
    register int _sdataIndex = (2*size+7)/8;
    register char *src = (char*)compressData;
    register unsigned long exponent;
    register unsigned long mantissa;
    register unsigned long *bptr = (unsigned long*)baseData;
    register int i;
    for(i=0; i<size; ++i)
    {
        int bitss = TESTBIT(src, i<<1);
        if(bitss==3) // different
        {

            decompressData[i] = *((unsigned long*)(src+_sdataIndex));
            _sdataIndex += 8;
        }else if(bitss==1) 
        {
            exponent = bptr[i]  & 0xff00000000000000;
            mantissa = *((unsigned long*)(src+_sdataIndex)) & 0x00ffffffffffffff;
            mantissa |= exponent;
            decompressData[i] = mantissa;   
            _sdataIndex += 7;
        }else
        {
            exponent = bptr[i]  & 0xffff000000000000;
            mantissa = *((unsigned long*)(src+_sdataIndex)) & 0x0000ffffffffffff;
            mantissa |= exponent;
            decompressData[i] = mantissa;   
            _sdataIndex += 6;
        }
    }
#endif
#if DEBUG
    double t = get_clock()-t1;
    printf("done decompressing.....  orig size:%d\n time:%d us", (int)size, (int)(t*1000000)) ;
#endif

}


#else

void compressDouble(void *src, void *dst, int s, int *compressSize, void *bData)
{
    register unsigned long *dest = (unsigned long*)dst;
    register unsigned long *bptr = (unsigned long*) bData;
    register unsigned long  *uptr = (unsigned long*) src;
    int size = s/sizeof(double);
#if DEBUG
    double t1 = get_clock();
#endif
    
#if !COMPRESS
    memcpy(dest, src, size*sizeof(double));
    *compressSize = s;
#else
    register int f_index = 0;
    register int i;
    register int j;
    register int b;
    register int zers;
    register unsigned long xor_data;
    memset(dest, 0, s);
    for (i = 0; i < size; ++i) {
        xor_data = (uptr[i])^(bptr[i]);
        zers = 0;
        //int value = xor_data;
        //printbitssimple(value);
        //printf("\n\n");
        b=BITS_DOUBLE-1;
        while(!TESTBIT(xor_data, b) && zers<15){
            zers++;
            b--;
        }
        //cout<<"c: "<<zers<<endl;
        //set the LZC 4 bits
        for(j=0; j<4; j++)
        {
            SETBIT(dest[(int)(f_index>>6)], (f_index&0x3f), ((unsigned long)(TESTBIT(zers, j))));
            f_index++;
        }
        while(b>=0)
        {
            SETBIT(dest[(f_index>>6)], f_index&0x3f, TESTBIT(xor_data, b));
            f_index++;
            b--;
        }
    }
    /*for (int k=0; k<size; k++) {
     printf(" %f ",dest[k]);
     }*/
    
    *compressSize = f_index/8;
    double compressRatio = (1-(double)(*compressSize)/s)*100;
    
#if DEBUG
    double t = get_clock()-t1;
    printf("===>double lzc done compressing compressed size:(%d===>%d) (reduction:%d) ration=%f Timer:%d us\n\n", (int)(size*sizeof(double)), *compressSize, (int)((size*sizeof(double)-*compressSize)), (1-(double)*compressSize/(size*sizeof(double)))*100, (int)(t*1000000));
#endif
    
#endif
}

void decompressDouble(void *cData, void *dData, int s, int compressSize, void *bData) {
    int size = s/sizeof(double);
#if DEBUG
    double t1 = get_clock();
#endif
#if !COMPRESS
    memcpy(dData, cData, size*sizeof(double));
#else
    register unsigned long *compressData = (unsigned long*)cData;
    register unsigned long *decompressData = (unsigned long*)dData;
    register unsigned long *baseData = (unsigned long*)bData;
    /*for (int k=0; k<size; k++) {
        printf("d: %d ",compressData[k]);
    }*/
    
    memset(decompressData, 0, s);
    register int index;
    register unsigned long xor_data;
    register unsigned long data = 0;
    register int d_index=0;
    register int compp = 0;
    register int i;
    register int j;
    register int f;
    for (i=0; i<size; i++) {
        index = BITS_DOUBLE-1;
        data = 0; int zers=0;
        //read 4 bits and puts index acccordingly
        for (f=0; f<4; f++,compp++) {
            if(TESTBIT(compressData[(int)(compp>>6)], (compp&0x3f))){
                for (j=0; j < (1<<f); j++) {
                    index--; zers++;
                }
            }
        }
        //cout<<"d: "<<zers<<endl;
        //printbitssimple();
        while(index>=0){
            SETBIT(data, index, TESTBIT(compressData[(int)(compp>>6)], (compp&0x3f)));
            index--; compp++;
        }
        xor_data = data^(baseData[i]);
        decompressData[i] = xor_data;
    }
    
#if DEBUG
    double t = get_clock()-t1;
    printf("done decompressing.....  orig size:%d time:%d us \n", size, (int)(t*1000000));
#endif
    
#endif
}

#endif


