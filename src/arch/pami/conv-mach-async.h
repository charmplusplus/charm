#define CMK_ENABLE_ASYNC_PROGRESS                          1

#undef CMK_MSG_HEADER_EXT_ 

#define CMK_MSG_HEADER_EXT_    int root, size, dstnode; CmiUInt2 rank, hdl,xhdl,info, stratid; unsigned char cksum, magic, redID, padding; char work[8*sizeof(void *)]; CmiUInt1 cmaMsgType:2, nokeep:1;


