#ifndef _PUP_PAGED_H_
#define _PUP_PAGED_H_
#define PUP_BLOCK 512
#include <stdio.h>
#include <string.h>
#include "pup.h"
typedef struct _list{
	int n;
	struct _list *next;
} pup_list;
// each pageentry is indexed by the pointer of the object
typedef struct _pageentry{
	void *ptr;
	pup_list *blklist;
	struct _pageentry *next;
} pup_pageentry;	

typedef struct {
	pup_list *freelist;
	pup_list *tailfreelist; //tail of freelist
	pup_pageentry *table; 
	pup_pageentry *tailtable; // tail of pagetable entries
	int maxblk; // the number of blocks that have been written out by now
	FILE *fp;
	char *fName;
} pup_pagetable;


pup_pagetable *getNewPagetable(char *fName);

class PUP_pagedDisk : public PUP::er {
	protected:
	pup_pagetable *_pagetable;
	void  *handle; // handle of the object to be restored
	PUP_pagedDisk(unsigned int type,void *objhandle,pup_pagetable *pgtable):PUP::er(type),handle(objhandle),_pagetable(pgtable){
	};

};

class PUP_toPagedDisk : public PUP_pagedDisk{
  protected:
  virtual void bytes(void *p,int n,size_t itemSize,PUP::dataType t);
	pup_pageentry *entry;
	long current_block;
	long bytes_left;
	FILE *fp;
	pup_list *tailblklist;
	public:
	PUP_toPagedDisk(void *objhandle,pup_pagetable *pgtable):PUP_pagedDisk(IS_PACKING,objhandle,pgtable){
		addpageentry();
		nextblock();
		fp = _pagetable->fp;
	}
	
	void addpageentry();
	void nextblock();

	
};

class PUP_fromPagedDisk : public PUP_pagedDisk{
	protected:
	virtual void bytes(void *p,int n,size_t itemSize,PUP::dataType );
	pup_pageentry *entry;
	long current_block;
	long bytes_unread;
	FILE *fp;
	public:
	PUP_fromPagedDisk(void *objhandle,pup_pagetable *pgtable):PUP_pagedDisk(IS_UNPACKING,objhandle,pgtable){
		findpageentry();
		current_block = -1;
		nextblock();
		fp = _pagetable->fp;
	}
	
	~PUP_fromPagedDisk(){
		nextblock();
		delete entry;
	}
	void findpageentry();
	void nextblock();
};

#endif
