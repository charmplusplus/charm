#define TBL_REPLY 1
#define TBL_NOREPLY 2

#define TBL_WAIT_AFTER_FIRST 1
#define TBL_NEVER_WAIT 2
#define TBL_ALWAYS_WAIT 3

message {
	int key;
	char *data;
} TBL_MSG;


#define TblInsert(x1, x2, x3, x4, x5, x6) \
		_CK_Insert(x1, -1, x2, x3, x4, x5, x6, -1)
#define TblDelete(x1, x2, x3, x4, x5)  _CK_Delete(x1, -1, x2, x3, x4, x5)
#define TblFind(x1, x2, x3, x4, x5) _CK_Find(x1, -1, x2, x3, x4, x5)

message {
	int msgs_processed;
} QUIESCENCE_MSG;
