#ifndef _PUP_PAGED_H_
#define _PUP_PAGED_H_
#define PUP_BLOCK 512
#include <stdio.h>
#include <string.h>
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
} pup_pagetable;
#endif
