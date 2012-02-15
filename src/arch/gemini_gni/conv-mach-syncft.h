
#undef CMK_MSG_HEADER_EXT_
//#undef CMK_MSG_HEADER_EXT
//#undef CMK_MSG_HEADER_BIGSIM_
/* expand the header to store the restart phase counter(pn) */
#define CMK_MSG_HEADER_EXT_    CmiUInt4 size; CmiUInt4 seq; CmiUInt2 rank,hdl,xhdl,info,stratid,redID,pn,d9; CmiInt4 root; 
//#define CMK_MSG_HEADER_EXT    { CMK_MSG_HEADER_EXT_ }
//#define CMK_MSG_HEADER_BIGSIM_    { CmiUInt2 d0,d1,d2,d3,d4,d5,hdl,xhdl,pn,info; int nd, n; double rt; CmiInt2 tID; CmiUInt2 hID; char t; int msgID; int srcPe;}
//#define CMK_MSG_HEADER_BIGSIM_  { CMK_MSG_HEADER_EXT_ CMK_BIGSIM_FIELDS }

#define CmiGetRestartPhase(m)       ((((CmiMsgHeaderExt*)m)->pn))

#define __FAULT__					   1

#define CMK_MEM_CHECKPOINT				   1
