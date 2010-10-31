/*priority queue, using a binary heap with static maximum size
This code is in the public domain, but I would be glad if you leave my name in:
Rene Wiermer, rwiermer@googlemail.com. Feedback is welcome.

The implemementation is based on Heapsort as described in
"Introduction to Algorithms" (Cormen, Leiserson, Rivest, 24. printing)
*/

/*needed for test only*/
#include <stdio.h>
/**/

#define MAX_SIZE 100000
typedef struct priormsg_s {
    unsigned int priority;
    int     pe;
    int     load;
    void* msg;
} priormsg;


typedef struct {
    priormsg* msgs[MAX_SIZE];
    unsigned int size;
} MsgHeap;


void heap_init(MsgHeap* h) {
    h->size=0;
}

int heap_size(MsgHeap* h)
{
    return h->size;
}
void heap_heapify(MsgHeap* h,int i) {
    int l,r,smallest;
    priormsg* tmp;
    l=2*i; /*left child*/
    r=2*i+1; /*right child*/

    if ((l < h->size)&&(h->msgs[l]->priority < h->msgs[i]->priority))
        smallest=l;
    else 
        smallest=i;
    if ((r < h->size)&&(h->msgs[r]->priority < h->msgs[smallest]->priority))
        smallest=r;
    if (smallest!=i) {
        /*exchange to maintain heap property*/
        tmp=h->msgs[smallest];
        h->msgs[smallest]=h->msgs[i];
        h->msgs[i]=tmp;
        heap_heapify(h,smallest);
    }
}

void heap_addItem(MsgHeap* h, priormsg* packet) {
    unsigned int i,parent;  
    h->size=h->size+1;
    i=h->size-1;
    parent=i/2;
    /*find the correct place to insert*/
    while ((i > 0)&&(h->msgs[parent]->priority > packet->priority)) {
        h->msgs[i]=h->msgs[parent];
        i=parent;
        parent=i/2;
    }
    h->msgs[i]=packet;
}

priormsg* heap_extractMin(MsgHeap* h) {
    priormsg* max;
    if (heap_isEmpty(h))
        return 0;
    max=h->msgs[0];
    h->msgs[0]=h->msgs[h->size-1];
    h->size=h->size-1;
    heap_heapify(h,0);
    return max;
}

int heap_isEmpty(MsgHeap *h) {
    return h->size==0;
}

int heap_isFull(MsgHeap *h) {
    return h->size>=MAX_SIZE;
}

