#include "converse.h"
#include "pup_paged.h"



pup_pagetable *getNewPagetable(char *fName){
	pup_pagetable *_pagetable = new pup_pagetable;
	_pagetable->freelist = NULL;
	_pagetable->table = NULL;
	_pagetable->tailtable = NULL;
	_pagetable->maxblk=0;
	_pagetable->fName = new char[strlen(fName)+20];
	sprintf(_pagetable->fName,"%s_%d.dat",fName,CmiMyPe());
	_pagetable->fp = fopen(_pagetable->fName,"wb");
	fclose(_pagetable->fp);
	_pagetable->fp = fopen(_pagetable->fName,"r+b");
	
	return _pagetable;
}

void PUP_toPagedDisk::addpageentry(){
	entry = new pup_pageentry;
	entry->next = NULL;
	entry->ptr = handle;
	entry->blklist = NULL;
	tailblklist = NULL;
	if(_pagetable->tailtable == NULL){
		_pagetable->table = entry;
	}else{
		_pagetable->tailtable->next = entry;
	}
	_pagetable->tailtable = entry;
}

void PUP_toPagedDisk::nextblock(){
	pup_list *f;
	f = _pagetable->freelist;
	if(f != NULL){
		current_block =  f->n;
		_pagetable->freelist=f->next;
		delete f;
	}else{
		current_block = _pagetable->maxblk;
		_pagetable->maxblk = current_block+1;
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
	pup_pageentry *p;
	p = NULL;
	entry = _pagetable->table;
	while(entry != NULL && entry->ptr != handle){
		p = entry;
		entry = entry->next;
	}
	if( p == NULL){
		_pagetable->table = entry->next;
	}else{
		p->next = entry->next;
	}
	if(_pagetable->tailtable == entry){
		_pagetable->tailtable = p;
	}
}

void PUP_fromPagedDisk::nextblock(){
	if(current_block != -1){
	// add blocks to the free list;
		pup_list *freenode = new pup_list;
		freenode->n = current_block;
		freenode->next = NULL;
		if(_pagetable->freelist == NULL){
			_pagetable->freelist = freenode;
			_pagetable->tailfreelist = freenode;
		}else{
			_pagetable->tailfreelist->next = freenode;
			_pagetable->tailfreelist = freenode;
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

