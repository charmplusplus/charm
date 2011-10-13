
#define CMK_UTH_VERSION                                    1

#define CMK_MSG_HEADER_FIELD  CmiUInt2 hdl,xhdl,info,stratid,root,redID,pad2,pad3;
#define CMK_MSG_HEADER_BASIC  CMK_MSG_HEADER_EXT
#define CMK_MSG_HEADER_EXT    { CMK_MSG_HEADER_FIELD }
#define CMK_MSG_HEADER_BIGSIM_    { CMK_MSG_HEADER_FIELD CMK_BIGSIM_FIELDS }

#define CMK_LBDB_ON					   0
