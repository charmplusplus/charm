
#undef CMK_MSG_HEADER_EXT_
//#undef CMK_MSG_HEADER_EXT
/* expand the header to store the restart phase counter(pn) */
#define CMK_MSG_HEADER_EXT_   CmiUInt2 rank, root, hdl,xhdl,info, type, pn,d7,redID; unsigned char cksum, magic, mpiMsgType; CmiUInt1 zcMsgType:4, cmaMsgType:2, nokeep:1;
//#define CMK_MSG_HEADER_EXT    { CMK_MSG_HEADER_EXT_ }

#define CmiGetRestartPhase(m)       ((((CmiMsgHeaderExt*)m)->pn))

#define __FAULT__					   1

#define CMK_MEM_CHECKPOINT				   1
