
#define CMK_MSG_HEADER_BASIC  CMK_MSG_HEADER_EXT
#define CMK_MSG_HEADER_EXT    { CmiUInt2 hdl,xhdl,info,d3; }
#define CMK_MSG_HEADER_BLUEGENE    { CmiUInt2 hdl,xhdl,info,d3; int nd, n; double rt; CmiInt2 tID; CmiUInt2 hID; char t; int msgID; int srcPe;}

#define CMK_LBDB_ON					   0
