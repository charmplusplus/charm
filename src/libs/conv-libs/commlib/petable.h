/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef PETABLE_H
#define PETABLE_H

#ifndef NULL
#define NULL 0
#endif

#define MSGQLEN 32

typedef struct ptinfo {
  int refCount;
  int magic;
  int offset;
  int freelistindex;
  int msgsize;
  void *msg;
  struct ptinfo * next;
} PTinfo;

typedef struct {
  int refCount;
  int flag;
  void * ptr;
} InNode;

class GList {
 private:
	InNode *InList;
	int InListIndex;
 public:
	GList();
	~GList();
	int AddWholeMsg(void *);
	void setRefcount(int, int);
	void DeleteWholeMsg(int);
	void DeleteWholeMsg(int, int);
	void GarbageCollect();
	void Add(void *);
	void Delete();
};

class PeTable {
  private:
	PTinfo ***PeList, **ptrlist;
	PTinfo *PTFreeList;
	//	char * CombBuffer;
	int *msgnum, *MaxSize;
	int NumPes;
	int magic;
	GList *FreeList;
	int TotalMsgSize(int, int *, int *, int *);
  public:
	PeTable(int n);
	~PeTable();
	void InsertMsgs(int npe, int *pelist, int nmsgs, void **msglist);
	void InsertMsgs(int npe, int *pelist, int size, void *msg);
	//int ExtractMsgs(int npe, int *pelist, int *nmsgs, void **msglist);
	void ExtractAndDeliverLocalMsgs(int pe);
	int UnpackAndInsert(void *in);
	char * ExtractAndPack(comID, int, int, int *pelist, int *length);
	//int ExtractAsVector(comID, int, int, int *, int **, char ***) 
        //	{ return(0);}
	void GarbageCollect();
	void Purge();
};

#endif
