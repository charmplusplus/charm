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
#include "compress.h"
#include "charm++.h"

#define     SETBIT(dest, i)  (dest[i>>3]) |= (1 << i%8 )
#define     TESTBIT(dest, i) ((dest[i>>3]) >>  (i%8)) & 1 
//#define     COMPRESS 1 
#define     DEBUG  1
#define     CHAR_BIT 8
#define     FLOAT_BIT CHAR_BIT*sizeof(float)
#define     FLOAT_BYTE sizeof(float)

//#define  COMPRESS_EXP 1


#if  COMPRESS_EXP

void compressFloatingPoint(void *src, void *dst, int s, int *compressSize, void *bData) 
{
    int size = s/FLOAT_BYTE;
    float *source = (float*)src;
    float *dest = (float*)dst;
    float *baseData = (float*)bData;
    unsigned int *bptr = (unsigned int*) baseData;
    unsigned int  *uptr = (unsigned int *) source;
    char *uchar;
    int i;
#if DEBUG
    double startTimer = CmiWallTimer();
    printf("[%d]starting compressing.....  orig size:%d ", CmiMyPe(), size);
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
        unsigned char *cdst = (unsigned char*)dest; 
        int _dataIndex = (size+7)/8;
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
    printf("[%d] ===>done compressingcompressed size:(%d===>%d) (reduction:%d) ration=%f Timer:%f ms\n\n", CmiMyPe(), size*sizeof(float), *compressSize, (size*sizeof(float)-*compressSize), (1-(float)*compressSize/(size*sizeof(float)))*100, (CmiWallTimer()-startTimer)*1000);
    //printf(" ===>done compressingcompressed size:(%d===>%d) (reduction:%d) ration=%f \n", size*sizeof(float), *compressSize, (size*sizeof(float)-*compressSize), (1-(float)*compressSize/(size*sizeof(float)))*100);
#endif
}

void decompressFloatingPoint(void *cData, void *dData, int s, int compressSize, void *bData) {
    int size = s/FLOAT_BYTE;
#if DEBUG
    double startTimer  = CmiWallTimer();
    printf("starting decompressing..... ");
#endif
#if !COMPRESS
    memcpy(dData, cData, size*sizeof(float));
#else
    float *compressData = (float*)cData;
    float *decompressData =(float*)dData;
    float *baseData = (float*)bData;
    int _sdataIndex = (size+7)/8;
    char *src = (char*)compressData;
    int exponent;
    unsigned int mantissa;
    unsigned int *bptr = (unsigned int*)baseData;
    int i;
    for(i=0; i<size; ++i)
    {
       if(TESTBIT(src, i)) // different
       {

           decompressData[i] = *((float*)(src+_sdataIndex));
           _sdataIndex += 4;
       }else        //same exponet
       {
           exponent = bptr[i]  & 0x7f800000;
           mantissa = *((unsigned int*)(src+_sdataIndex)) & 0x00FFFFFF;
           mantissa = (mantissa >> 1) | (mantissa << 31) ;
           mantissa |= exponent;
           decompressData[i] = *((float*)&mantissa);   
           _sdataIndex += 3;
       }
    }
#endif
#if DEBUG
    printf("done decompressing.....  orig size:%d\n time:%f ms", size, (CmiWallTimer()-startTimer)*1000);
#endif

}


#else

#include <bitset>
#include <vector>
using namespace std;

void compressFloatingPoint(void *src, void *dst, int s, int *compressSize, void *bData)
{
    float *source = (float*)src;
    float *dest = (float*)dst;
    float *baseData = (float*)bData;
    int size = s/sizeof(float);
#if DEBUG
    double startTimer = CmiWallTimer();
    printf("[%d]starting compressing.....  orig size:%d ", CmiMyPe(), size);
#endif
    
#if !COMPRESS 
    memcpy(dest, source, size*sizeof(float)); 
    *compressSize = s;
#else
    std::bitset<FLOAT_BIT> comp_data;    
    int index = 0;
    int f_index = 0;
    unsigned long tmp;
    for (int i = 0; i < size; ++i) {
        std::bitset<FLOAT_BIT> data(*(unsigned long*)(&(source[i])));
        std::bitset<FLOAT_BIT> prev_data(*(unsigned long*)(&(baseData[i])));
        std::bitset<FLOAT_BIT> xor_data = data^prev_data;
        int zers = 0;
        bool flag = false;
        for(int b=xor_data.size()-1; b>=0; b--){
            if(!flag){
                if(!xor_data[b] && zers<15){
                    zers++;
                }
                else{
                    flag = true;
                    std:bitset<4> bs(zers);
                    for(int j=3;j>=0; j--)
                    {
                        comp_data[index++]=bs[j];
                        if(index == FLOAT_BIT){ tmp = comp_data.to_ulong(); index = 0;  dest[f_index++] = *(float*)(&tmp);}
                     }
                        comp_data[index++]=xor_data[b];
                        if(index == FLOAT_BIT){ tmp = comp_data.to_ulong(); index = 0;  dest[f_index++] = *(float*)(&tmp);}
                }
            }
            else{
                comp_data[index++]=xor_data[b];
                if(index == FLOAT_BIT){ tmp = comp_data.to_ulong(); index = 0;  dest[f_index++] = *(float*)(&tmp);}
            }
        }
    }
    if(index > 0)
    { tmp = comp_data.to_ulong(); dest[f_index++] = *(float*)(&tmp);}

    *compressSize = f_index*sizeof(float);
    float compressRatio = (1-(float)(*compressSize)/s)*100;
    
#if DEBUG
    printf("[%d] ===>done compressingcompressed size:(%d===>%d) (reduction:%d) ration=%f Timer:%f ms\n\n", CmiMyPe(), size*sizeof(float), *compressSize, (size*sizeof(float)-*compressSize), (1-(float)*compressSize/(size*sizeof(float)))*100, (CmiWallTimer()-startTimer)*1000);
#endif

#endif
}

void decompressFloatingPoint(void *cData, void *dData, int s, int compressSize, void *bData) {
    int size = s/sizeof(float);
#if DEBUG
    double startTimer  = CmiWallTimer();
    printf("starting decompressing..... ");
#endif
#if !COMPRESS
    memcpy(dData, cData, size*sizeof(float));
#else
    float*compData = (float*)cData;
    float *decompressData = (float*)dData;
    float *baseData = (float*)bData;
    int compp = 0;
    vector<bool>  compressData;
    std::bitset<FLOAT_BIT> xor_data;
    std::bitset<FLOAT_BIT> data(0ul);
    int index = 0;
    for(int i=0; i<compressSize/sizeof(float); i++)
    {
        std::bitset<FLOAT_BIT> cbits(*(unsigned long*) (&(compData[i])));
        for(int j=0; j<FLOAT_BIT; j++)
            compressData.push_back(cbits[j]);
    }

    for (int i=0; i<size; i++) {
        int index = FLOAT_BIT-1;
        std::bitset<FLOAT_BIT> prev_data(*(unsigned long*)(&(baseData[i])));
        //read 4 bits and puts index acccordingly
        for (int f=3; f>=0; f--,compp++) {
            if(compressData[compp] == 1){
                for (int ff=0; ff<pow(2,f); ff++) {
                    data[index] = 0; index--;
                }
            }
        }
        while(index>=0){
            data[index] = compressData[compp];
            index--; compp++;
        }
     
        xor_data = data^prev_data;
        unsigned long tmp = xor_data.to_ulong();
        decompressData[i] = *(reinterpret_cast<float*>(&tmp));
    }

#if DEBUG
    printf("done decompressing.....  orig size:%d\n time:%f ms", size, (CmiWallTimer()-startTimer)*1000);
#endif

#endif
}

#endif
