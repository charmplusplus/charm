#include "charm.h"
#include "pup_paged.h"

CpvDeclare(pup_pagetable *,_pagetable);
CpvDeclare(int,_openPagetableFile); /*checks if the data file has been openned. 
																			if not it is openned during a constructor call to pagedDisk.
																			this prevents the data file from being created if no
																			pagedDisk pupper is created.*/


void _pupModuleInit(){
	CpvInitialize(pup_pagetable *,_pagetable);
	CpvInitialize(int,_openPagetableFile);
	CpvAccess(_pagetable) = new pup_pagetable;
	CpvAccess(_pagetable)->freelist = NULL;
	CpvAccess(_pagetable)->table = NULL;
	CpvAccess(_pagetable)->maxblk=0;
	sprintf(CpvAccess(_pagetable)->fName,"_data%d.dat",CkMyPe());
	CpvAccess(_openPagetableFile)=0;
	
}

void PUP_toPagedDisk::addpageentry(){
	pup_pageentry *p, *q;
	p = CpvAccess(_pagetable)->table;
	q = NULL;
	while(p != NULL){
		q = p;
		p = p->next;
	}
	entry = new pup_pageentry;
	entry->next = NULL;
	entry->ptr = handle;
	entry->blklist = NULL;
	tailblklist = NULL;
	if(q == NULL){
		CpvAccess(_pagetable)->table = entry;
	}else{
		q->next = entry;
	}
}

void PUP_toPagedDisk::nextblock(){
	pup_list *f;
	f = CpvAccess(_pagetable)->freelist;
	if(f != NULL){
		current_block =  f->n;
		CpvAccess(_pagetable)->freelist=f->next;
		delete f;
	}else{
		current_block = CpvAccess(_pagetable)->maxblk;
		CpvAccess(_pagetable)->maxblk = current_block+1;
	}
	pup_list *newblk = new pup_list;
	newblk->n = current_block;
	newblk->next = NULL;
	if(tailblklist == NULL){
		entry->blklist = newblk;
	}else{
		tailblklist->next = newblk;
	}
	tailblklist = newblk;
	bytes_left = PUP_BLOCK;
}



void PUP_toPagedDisk::bytes(void *p,int n,size_t itemSize,PUP::dataType) {
	long size = itemSize*n;
	char *c = (char *)p;
	while(size > bytes_left){
		long next=current_block*PUP_BLOCK + PUP_BLOCK - bytes_left;
		fseek(fp,next,SEEK_SET);
		fwrite(c,1,bytes_left,fp);
		size -= bytes_left;
		c += bytes_left;
		bytes_left = 0;
		nextblock();
	}
	long next=current_block*PUP_BLOCK + PUP_BLOCK - bytes_left;
	fseek(fp,next,SEEK_SET);
	fwrite(c,1,size,fp);
	bytes_left -= size;
}



void PUP_fromPagedDisk::findpageentry(){
	entry = CpvAccess(_pagetable)->table;
	while(entry != NULL && entry->ptr != handle){
		entry = entry->next;
	}
}

void PUP_fromPagedDisk::nextblock(){
	if(current_block != -1){
	// add blocks to the free list;
		pup_list *freenode = new pup_list;
		freenode->n = current_block;
		freenode->next = NULL;
		if(CpvAccess(_pagetable)->freelist == NULL){
			CpvAccess(_pagetable)->freelist = freenode;
			CpvAccess(_pagetable)->tailfreelist = freenode;
		}else{
			CpvAccess(_pagetable)->tailfreelist->next = freenode;
			CpvAccess(_pagetable)->tailfreelist = freenode;
		}
	}
	if(entry->blklist != NULL){
		current_block = entry->blklist->n;
		entry->blklist = entry->blklist->next;
	}
	bytes_unread = PUP_BLOCK;
}

void PUP_fromPagedDisk::bytes(void *p,int n,size_t itemSize,PUP::dataType ){
	long size = n*itemSize;
	char *c = (char *)p;
	while(size > bytes_unread){
		long next = current_block*PUP_BLOCK + PUP_BLOCK - bytes_unread;
		fseek(fp,next,SEEK_SET);
		fread(c,1,bytes_unread,fp);
		size -= bytes_unread;
		c += bytes_unread;
		bytes_unread = 0;
		nextblock();
	}
	long next = current_block*PUP_BLOCK + PUP_BLOCK - bytes_unread;
	fseek(fp,next,SEEK_SET);
	fread(c,1,size,fp);
	bytes_unread -= size;
}

