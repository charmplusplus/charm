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
	int maxblk; // the number of blocks that have been written out by now
	FILE *fp;
	char fName[100];
} pup_pagetable;

CpvExtern(pup_pagetable *,_pagetable);
CpvExtern(int,_openPagetableFile);

class PUP_pagedDisk : public PUP::er {
	protected:
	void  *handle; // handle of the object to be restored
	PUP_pagedDisk(unsigned int type,void *objhandle):er(type),handle(objhandle){
		if(CpvAccess(_openPagetableFile) == 0){
			CpvAccess(_pagetable)->fp = fopen(CpvAccess(_pagetable)->fName,"wb");
			fclose(CpvAccess(_pagetable)->fp);
			CpvAccess(_pagetable)->fp = fopen(CpvAccess(_pagetable)->fName,"r+b");
			CpvAccess(_openPagetableFile) = 1;
		}
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
	PUP_toPagedDisk(void *objhandle):PUP_pagedDisk(IS_PACKING,objhandle){
		addpageentry();
		nextblock();
		fp = CpvAccess(_pagetable)->fp;
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
	PUP_fromPagedDisk(void *objhandle):PUP_pagedDisk(IS_UNPACKING,objhandle){
		findpageentry();
		current_block = -1;
		nextblock();
		fp = CpvAccess(_pagetable)->fp;
	}
	
	~PUP_fromPagedDisk(){
		nextblock();
	}
	void findpageentry();
	void nextblock();
};

#endif
