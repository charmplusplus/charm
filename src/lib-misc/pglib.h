#define MAX_SPAN_CHILDREN 2
#define HASH_TABLE_SIZE 11     /*This should be some (smallish) prime number */
#define MAX_HASH_TABLE_SIZE 13 /*This should be the next larrgest prime num */
#define PG_COORD 0

message {} GENERIC_MESSAGE ;

message {} CREATE_ROOT ;

message {
    EntryPointType ep ;
    ChareNumType boc ;
} PG_CREATE_ROOT ;

message {
    ChareNumType rootBocNum ;
} ROOT_GROUP_CREATED ;

message {
    int newGid ;
} PARTITION_CREATED ;

message {
    int groupID ;
    int partNum ;
    int copyNum ;
    int requestor ;
    int refNum ;
    EntryPointType retEP ;
    ChareNumType retBoc ;
} PARTITION_AT_ROOT ;

message {
    int requestor ;
    int groupID ;
    int msgFrom ;
    int rootProc ;
    int copyNum ;
    int partNum ;
    int descendantCount ;
    int rank ;
    EntryPointType notifyEP ;
    ChareNumType notifyBoc ;
    int notifyRefNum ;
} NEW_MEMBER ;

message {
  int gid ;
  int newGid ;
  int descendantCount ;
  int totalGroupSize ;
} DONE_MSG ;

message {
    int gid ;
    int controlRefNum ;
    int userRefNum ;
    EntryPointType retEP ;
    ChareNumType retBoc ;
} MULTICAST_MESSAGE ;

message {
    int gid ;
    int controlRefNum ;
} SYNCHRONIZE_MESSAGE  ;

message {
    int requestor ;
} GETSIZE_MESSAGE ;

typedef struct ControlMsgType {
    int controlRefNum ;
    int userRef ;
    EntryPointType deliverEP ;
    ChareNumType deliverBoc ;
    ChareIDType deliverId ;
    GENERIC_MESSAGE *controlBufMsg ;
    int groupID ;
    int gotControl ;
    int gotMessage ;
    int count ;
    struct ControlMsgType *next ;
} ControlMsgType ;

typedef struct BufMsgList {
    GENERIC_MESSAGE *bufMsg ;
    struct BufMsgList *next ;
} BufMsgList ;

typedef struct BufferedMessages {
    int groupID ;
    BufMsgList *bufMsgHead ;
    struct BufferedMessages *next ;
} BufferedMessages ;

typedef struct {
    int exp ;
    int direction ;
    int range ;
    int val ;
} SpanTreeBuildDetails ;

typedef struct {
    int gid ;
    int rootProc ; 
} GidInfoType ;

typedef struct PartInfoType {
  int partNum ;
  int joinedThisPartition ;
  GidInfoType gidInfo ;
  struct PartInfoType *nextPart ;
} PartInfoType ;

typedef struct CopyInfoType {
  int copyNum ;
  int joinedThisCopy ;
  int needToJoinThisCopy ;
  PartInfoType *partInfo ;
  struct CopyInfoType *nextCopy ;
} CopyInfoType ;

typedef struct {
    int spanParent ;
    int spanNumChildren ;
    int spanChildren[MAX_SPAN_CHILDREN] ;
    SpanTreeBuildDetails spanDetails ;
    int descendants ;
    int totalGroupSize ;
    int groupRank ;
    EntryPointType notifyEP ;
    ChareNumType notifyBoc ;
    int notifyRefNum ;
    CopyInfoType *copyInfo ;
} GroupInfoType ;

typedef struct RootGidType {
    GidInfoType gidInfo ;
    GroupInfoType groupInfo ;
    struct RootGidType *nextGid ;   
} RootGidType ;
