#define CMK_ENABLE_ASYNC_PROGRESS                          1

//#define  DELTA_COMPRESS                                     1

#undef CMK_MSG_HEADER_EXT_ 
#if DELTA_COMPRESS
#define CMK_MSG_HEADER_EXT_    CmiUInt8 persistRecvHandler; int root, size, dstnode; CmiUInt4 compressStart; CmiUInt2 rank, hdl,xhdl,info, type, redID, padding, compress_flag,xxhdl; unsigned char cksum, magic; char work[6*sizeof(void *)]; CmiUInt1 cmaMsgType:2, nokeep:1;
#else
#define CMK_MSG_HEADER_EXT_    int root, size, dstnode; CmiUInt2 rank, hdl,xhdl,info, type, redID, padding; unsigned char cksum, magic; char work[6*sizeof(void *)]; CmiUInt1 cmaMsgType:2, nokeep:1;
#endif


