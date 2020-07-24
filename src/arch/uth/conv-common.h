
#define CMK_UTH_VERSION                                    1

#define CMK_MSG_HEADER_FIELD  CmiUInt2 hdl,xhdl,info,stratid,root,redID,pad2,pad3; CmiUInt1 zcMsgType:4, cmaMsgType:2, nokeep:1, msgLayerType:1, commSender:1; CmiInt4 uniqMsgId; CmiUInt4 msgSrcPe, msgSrcNode;
#define CMK_MSG_HEADER_BASIC  CMK_MSG_HEADER_EXT
#define CMK_MSG_HEADER_EXT    { CMK_MSG_HEADER_FIELD }

#define CMK_LBDB_ON					   0
