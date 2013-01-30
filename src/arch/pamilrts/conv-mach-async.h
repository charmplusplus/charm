#define CMK_ENABLE_ASYNC_PROGRESS                          1

//#define  DELTA_COMPRESS                                     1

#undef CMK_MSG_HEADER_EXT_ 
#if DELTA_COMPRESS
#define CMK_MSG_HEADER_EXT_    CmiUInt2 rank, hdl,xhdl,info, stratid; unsigned char cksum, magic; int root, size, dstnode; CmiUInt2 redID, padding; char work[6*sizeof(uintptr_t)]; CmiUInt4 compressStart; CmiUInt2 compress_flag,xxhdl; CmiUInt8 persistRecvHandler;
#else
#define CMK_MSG_HEADER_EXT_    CmiUInt2 rank, hdl,xhdl,info, stratid; unsigned char cksum, magic; int root, size, dstnode; CmiUInt2 redID, padding; char work[6*sizeof(uintptr_t)]; 
#endif


