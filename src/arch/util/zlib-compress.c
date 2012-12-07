/*
 * =====================================================================================
 *
 *       Filename:  compress.c
 *
 *    Description:  Compress messages before sending and uncompress after receiving 
 *
 *        Created:  07/30/2012 03:11:54 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Yanhua Sun(), 
 *   Organization:  
 *
 * =====================================================================================
 */
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include "zlib.h"

#if defined(MSDOS) || defined(OS2) || defined(WIN32) || defined(__CYGWIN__)
#  include <fcntl.h>
#  include <io.h>
#  define SET_BINARY_MODE(file) setmode(fileno(file), O_BINARY)
#else
#  define SET_BINARY_MODE(file)
#endif

#define COMP_DEBUG 0

void zlib_compress(void *in_data, void *out_data, int in_data_size, int *out_data_size)
{
#if 1
    double start_timer=CmiWallTimer();
    z_stream strm;
    //FILE *fp = fopen("~/compressInfo.dat", "a+");
    strm.zalloc = 0;
    strm.zfree = 0;
    strm.next_in = (uint8_t *)(in_data);
    strm.avail_in = in_data_size;
    strm.next_out = (uint8_t *)(out_data);
    strm.avail_out = in_data_size;

    deflateInit(&strm, Z_BEST_COMPRESSION);

    if(strm.avail_in != 0)
    {
        int res = deflate(&strm, Z_NO_FLUSH);
        assert(res == Z_OK);
        
        int deflate_res = Z_OK;
        while (deflate_res == Z_OK)
        {
            deflate_res = deflate(&strm, Z_FINISH);
        }
    }
    deflateEnd(&strm);

    *out_data_size = in_data_size - strm.avail_out;
#if COMP_DEBUG
    CmiPrintf( "[%d]: %d \t\t %d \t\t %.3lf  cost:%.3f\n", CmiMyPe(), in_data_size, *out_data_size, *out_data_size/(float)in_data_size, (CmiWallTimer()-start_timer)*1000); 
#endif
    //fclose(fp);
   
#else
    memcpy(out_data, in_data, in_data_size);
    *out_data_size = in_data_size;
#endif
}

int zlib_decompress(const void *src, void *dst, int srcLen, int dstLen) {
#if 1
    double start_timer=CmiWallTimer();
    z_stream strm  = {0};
    strm.total_in  = strm.avail_in  = srcLen;
    strm.total_out = strm.avail_out = dstLen;
    strm.next_in   = (Bytef *) src;
    strm.next_out  = (Bytef *) dst;

    strm.zalloc = Z_NULL;
    strm.zfree  = Z_NULL;
    strm.opaque = Z_NULL;

    int err = -1;
    int ret = -1;

    err = inflateInit(&strm);//, (15 + 32)); //15 window bits, and the +32 tells zlib to to detect if using gzip or zlib
    if (err == Z_OK) {
        err = inflate(&strm, Z_NO_FLUSH);
        if (err == Z_STREAM_END) {
            ret = strm.total_out;
        }
        inflateEnd(&strm);
    }
    else {
        inflateEnd(&strm);
        ret = err;
    }
#if COMP_DEBUG
    CmiPrintf("Decompress time:%d   %.3f\n", dstLen, (CmiWallTimer()-start_timer));
#endif
    return ret;
#else
    memcpy(dst, src, srcLen);
    return 0;
#endif
}

void zlib_init() {}
