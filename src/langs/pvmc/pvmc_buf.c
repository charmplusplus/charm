#include <stddef.h>
#include <stdio.h>
#include <converse.h>
#include "pvmc.h"

#define MAX_BUFFERS 1000

typedef struct pvmc_item_s {
  int type;
  int size;
  int free_data;
  char *data;
  struct pvmc_item_s *nxt;
} pvmc_item;

typedef struct pvmc_buffer_s {
  int bufid;
  int bytes;
  int tag;
  int tid;
  int num_items;
  int refcount;
  pvmc_item *first_item;
  pvmc_item *cur_item;
  pvmc_item *last_item;
  struct pvmc_buffer_s *nxt_free;
  char *data_buf;
} pvmc_buffer;

CpvStaticDeclare(pvmc_buffer*,pvmc_bufarray);
CpvStaticDeclare(pvmc_buffer*,pvmc_freebufs);
CpvStaticDeclare(int,pvmc_sbufid);
CpvStaticDeclare(int,pvmc_rbufid);

void pvmc_init_bufs(void)
{
  int i;

#ifdef PVM_DEBUG
  PRINTF("Pe(%d) tid=%d:%s:%d pvmc_init_bufs() initializing buffer array\n",
	MYPE(),pvm_mytid(),__FILE__,__LINE__);
#endif

  CpvInitialize(pvmc_buffer*,pvmc_bufarray);
  CpvAccess(pvmc_bufarray)=MALLOC(sizeof(pvmc_buffer)*MAX_BUFFERS);
  if (CpvAccess(pvmc_bufarray)==NULL) {
    PRINTF("Pe(%d) tid=%d:%s:%d pvmc_init_bufs() can't alloc buffer array\n",
	   MYPE(),pvm_mytid(),__FILE__,__LINE__);
    exit(1);
  }
    
  CpvInitialize(pvmc_buffer*,pvmc_freebufs);
  CpvAccess(pvmc_freebufs)=&(CpvAccess(pvmc_bufarray)[1]);  /* throw away first bufid */
  
  for(i=0;i<MAX_BUFFERS;i++) {
    CpvAccess(pvmc_bufarray)[i].bufid=i;
    CpvAccess(pvmc_bufarray)[i].bytes=0;
    CpvAccess(pvmc_bufarray)[i].tag=0;
    CpvAccess(pvmc_bufarray)[i].tid=-1;
    CpvAccess(pvmc_bufarray)[i].num_items=-1;
    CpvAccess(pvmc_bufarray)[i].refcount=0;
    CpvAccess(pvmc_bufarray)[i].first_item=(pvmc_item *)NULL;
    CpvAccess(pvmc_bufarray)[i].cur_item=(pvmc_item *)NULL;
    CpvAccess(pvmc_bufarray)[i].last_item=(pvmc_item *)NULL;
    if (i==MAX_BUFFERS-1)
      CpvAccess(pvmc_bufarray)[i].nxt_free=(pvmc_buffer *)NULL;
    else
      CpvAccess(pvmc_bufarray)[i].nxt_free=&(CpvAccess(pvmc_bufarray)[i+1]);

    CpvAccess(pvmc_bufarray)[i].data_buf=(char *)NULL;
  }

  CpvInitialize(int,pvmc_sbufid);
  CpvAccess(pvmc_sbufid) = 0;

  CpvInitialize(int,pvmc_rbufid);
  CpvAccess(pvmc_rbufid) = 0;
}

int pvm_mkbuf(int encoding)
{
  pvmc_buffer *new_buf;

#ifdef PVM_DEBUG
  PRINTF("Pe(%d) tid=%d:pvm_mkbuf(%d)\n",
	MYPE(),pvm_mytid(),encoding);

  if (encoding != PvmDataRaw)
    PRINTF("Pe(%d) tid=%d:%s:%d Warning: only encoding=PvmDataRaw supported\n",
	   MYPE(),pvm_mytid(),__FILE__,__LINE__);
#endif

  new_buf = CpvAccess(pvmc_freebufs);
  if (new_buf == NULL) {
    PRINTF("Pe(%d) tid=%d:%s:%d pvm_mkbuf() no more buffers\n",
	   MYPE(),pvm_mytid(),__FILE__,__LINE__);
    return -1;
  }

  CpvAccess(pvmc_freebufs)=CpvAccess(pvmc_freebufs)->nxt_free;
  new_buf->bytes=0;
  new_buf->tag=0;
  new_buf->tid=pvm_mytid();
  new_buf->num_items = 0;
  if ((new_buf->first_item=
       (pvmc_item *)MALLOC(sizeof(pvmc_item))) == NULL) {
    PRINTF("Pe(%d) tid=%d:%s:%d pvm_mkbuf() MALLOC failed\n",
	   MYPE(),pvm_mytid(),__FILE__,__LINE__);
    return -1;
  }
  new_buf->first_item->type=0;
  new_buf->first_item->size=0;
  new_buf->first_item->free_data=FALSE;
  new_buf->first_item->data=(char *)NULL;
  new_buf->first_item->nxt=(pvmc_item *)NULL;

  new_buf->cur_item=new_buf->first_item;
  new_buf->last_item=new_buf->first_item;
  new_buf->refcount=1;
  return new_buf->bufid;
}
  
static void pvmc_emptybuf(pvmc_buffer *cur_buf)
{
  pvmc_item *nxt_item, *prv_item; 

  if (cur_buf->data_buf) {
    FREE(cur_buf->data_buf);
    cur_buf->data_buf=(char *)NULL;
  }

  nxt_item=cur_buf->first_item;
  while (nxt_item) {
    prv_item=nxt_item;
    nxt_item=nxt_item->nxt;
    if (prv_item->free_data)
      FREE(prv_item->data);
    FREE(prv_item);
  }
  cur_buf->bytes=0;
  cur_buf->tag=0;
  cur_buf->tid=-1;
  cur_buf->num_items=0;
  cur_buf->first_item=(pvmc_item *)NULL;
  cur_buf->cur_item=(pvmc_item *)NULL;
  cur_buf->last_item=(pvmc_item *)NULL;
}

int pvm_freebuf(int bufid)
{
  pvmc_buffer *cur_buf;
  int result=0;

#ifdef PVM_DEBUG
  PRINTF("Pe(%d) tid=%d:pvm_freebuf(%d)\n",MYPE(),pvm_mytid(),bufid);
#endif

  if ((bufid<=0) || (bufid>=MAX_BUFFERS)) {
    PRINTF("Pe(%d) tid=%d:%s:%d pvm_freebuf() attempted to free out of range bufid\n",
	   MYPE(),pvm_mytid(),__FILE__,__LINE__);
    return -1;
  }
    
  cur_buf = &(CpvAccess(pvmc_bufarray)[bufid]);

  if (cur_buf->refcount < 1) {
    PRINTF("Pe(%d) tid=%d:%s:%d pvm_freebuf(%d) refcount=%d, i'm confused\n",
	   MYPE(),pvm_mytid(),__FILE__,__LINE__,bufid,cur_buf->refcount);
    result=-2;
  }
  cur_buf->refcount--;

  if (cur_buf->refcount==0) {
    pvmc_emptybuf(cur_buf);
    cur_buf->nxt_free=CpvAccess(pvmc_freebufs);
    CpvAccess(pvmc_freebufs)=cur_buf;
  }

#ifdef PVM_DEBUG
  {
    int x;
    int Counter=0;
    int FreeCounter=0;
    struct pvmc_buffer_s *FreeList;
    /* find the number that we think are free */
    for(x=0; x<MAX_BUFFERS; x++)
    {
      if (CpvAccess(pvmc_bufarray)[x].refcount == 0) Counter++;
    }
    /* find the number that are linked as free */
    FreeList = CpvAccess(pvmc_freebufs);
    while(FreeList != NULL)
    {
      FreeCounter++;
      FreeList = FreeList->nxt_free;
    }
    /* show the results */
    PRINTF("Pe(%d) tid=%d:%s:%d unused=(%d) sizeof(freelist)=%d\n",
	   MYPE(),pvm_mytid(),__FILE__,__LINE__,Counter,FreeCounter);
  }
#endif

  return result;
}  

static int pvmc_getbuf(int bufid)
{
  pvmc_buffer *cur_buf;

  if ((bufid<=0) || (bufid>=MAX_BUFFERS)) {
    PRINTF("Pe(%d) tid=%d:%s:%d pvmc_getbuf(%d) attempted to get out of range bufid\n",
	   MYPE(),pvm_mytid(),__FILE__,__LINE__,bufid);
    return -1;
  }
    
  cur_buf = &(CpvAccess(pvmc_bufarray)[bufid]);

  if (cur_buf->refcount<1) {
    PRINTF("Pe(%d) tid=%d:%s:%d pvm_getbuf() trying with refcount=%d, i'm confused\n",
	   MYPE(),pvm_mytid(),__FILE__,__LINE__,cur_buf->refcount);
    return -1;
  }

  cur_buf->refcount++;
  return bufid;
}

int pvm_getsbuf(void)
{
#ifdef PVM_DEBUG
  PRINTF("Pe(%d) tid=%d:pvm_getsbuf()\n",MYPE(),pvm_mytid());
#endif
  return CpvAccess(pvmc_sbufid);
}

int pvm_setsbuf(int bufid)
{
  int prv_sbufid;

#ifdef PVM_DEBUG
  PRINTF("Pe(%d) tid=%d:pvm_setsbuf(%d)\n",MYPE(),pvm_mytid(),bufid);
#endif

  prv_sbufid=CpvAccess(pvmc_sbufid);

  /*
  if (prv_sbufid>0) {
    pvmc_getbuf(prv_sbufid);
    pvm_freebuf(prv_sbufid);
  }
  */

  CpvAccess(pvmc_sbufid)=bufid;

  return prv_sbufid;
}

int pvm_getrbuf(void)
{
#ifdef PVM_DEBUG
  PRINTF("Pe(%d) tid=%d:pvm_getrbuf()\n",MYPE(),pvm_mytid());
#endif
  return CpvAccess(pvmc_rbufid);
}

int pvm_setrbuf(int bufid)
{
  int prv_rbufid;

#ifdef PVM_DEBUG
  PRINTF("Pe(%d) tid=%d:pvm_setrbuf(%d)\n",MYPE(),pvm_mytid(),bufid);
#endif
  prv_rbufid=CpvAccess(pvmc_rbufid);

  /*
  if (prv_rbufid>0) {
    pvmc_getbuf(prv_rbufid);
    pvm_freebuf(prv_rbufid);
  }
  */
    

  CpvAccess(pvmc_rbufid)=bufid;

  return prv_rbufid;
}

int pvm_initsend(int encoding)
{
  int newbufid;

#ifdef PVM_DEBUG
  PRINTF("Pe(%d) tid=%d:pvm_initsend(%d)\n",MYPE(),pvm_mytid(),encoding);
#endif
  if (CpvAccess(pvmc_sbufid) > 0)
    pvm_freebuf(CpvAccess(pvmc_sbufid));

  newbufid=pvm_mkbuf(encoding);

  if (newbufid<=0) {
    PRINTF("Pe(%d) tid=%d:%s:%d pvm_initsend() couldn't alloc new buffer\n",
	   MYPE(),pvm_mytid(),__FILE__,__LINE__);
  }
  CpvAccess(pvmc_sbufid)=newbufid;
  return CpvAccess(pvmc_sbufid);
}

int pvm_bufinfo(int bufid, int *bytes, int *msgtag, int *tid)
{
#ifdef PVM_DEBUG
  PRINTF("Pe(%d) tid=%d:pvm_bufinfo(%d,0x%x,0x%x,0x%x)\n",
	 pvm_mytid(),bufid,bytes,msgtag,tid);
#endif

  if ((bufid<=0) || (bufid >= MAX_BUFFERS) ||
      (CpvAccess(pvmc_bufarray)[bufid].refcount <= 0)) {
    PRINTF("Pe(%d) tid=%d:%s:%d pvm_bufinfo(%d) info requested about unused buffer\n",
	   MYPE(),pvm_mytid(),__FILE__,__LINE__,bufid);
    return -1;
  }
  if (bytes)
    *bytes=CpvAccess(pvmc_bufarray)[bufid].bytes;
  if (msgtag)
    *msgtag=CpvAccess(pvmc_bufarray)[bufid].tag;
  if (tid)
    *tid=CpvAccess(pvmc_bufarray)[bufid].tid;
  return 0;
}

int pvmc_sendmsgsz(void)
{
  int msgsz;
  pvmc_buffer *cur_buf;
  pvmc_item *cur_item;

  if ((CpvAccess(pvmc_sbufid)<=0) || (CpvAccess(pvmc_sbufid) >= MAX_BUFFERS) ||
      (CpvAccess(pvmc_bufarray)[CpvAccess(pvmc_sbufid)].refcount <= 0)) {
    PRINTF("Pe(%d) tid=%d:%s:%d pvmc_sendmsgsz() size requested for unused send buffer\n",
	   MYPE(),pvm_mytid(),__FILE__,__LINE__);
    return -1;
  }

  cur_buf = &(CpvAccess(pvmc_bufarray)[CpvAccess(pvmc_sbufid)]);

  msgsz=sizeof(cur_buf->bytes)+sizeof(cur_buf->tag)+
    sizeof(cur_buf->tid)+sizeof(cur_buf->num_items);

  cur_item=cur_buf->first_item;
  while (cur_item != cur_buf->last_item) {
    msgsz += cur_item->size+sizeof(cur_item->type)+sizeof(cur_item->size);
    cur_item = cur_item->nxt;
  }

  return msgsz;
}

int pvmc_settidtag(int pvm_tid, int tag)
{
  pvmc_buffer *cur_buf;

  if ((CpvAccess(pvmc_sbufid)<=0) || (CpvAccess(pvmc_sbufid) >= MAX_BUFFERS) ||
      (CpvAccess(pvmc_bufarray)[CpvAccess(pvmc_sbufid)].refcount <= 0)) {
    PRINTF("Pe(%d) tid=%d:%s:%d pvmc_setidtag() unused send buffer\n",
	   MYPE(),pvm_mytid(),__FILE__,__LINE__);
    return -1;
  }
  cur_buf=&(CpvAccess(pvmc_bufarray)[CpvAccess(pvmc_sbufid)]);
  cur_buf->tag = tag;
  cur_buf->tid = pvm_tid;
}


int pvmc_packmsg(void *msgbuf)
{
  pvmc_buffer *cur_buf;
  pvmc_item *cur_item;
  int bytes_packed=0;

  if ((CpvAccess(pvmc_sbufid)<=0) || (CpvAccess(pvmc_sbufid) >= MAX_BUFFERS) ||
      (CpvAccess(pvmc_bufarray)[CpvAccess(pvmc_sbufid)].refcount <= 0)) {
    PRINTF("Pe(%d) tid=%d:%s:%d pvmc_packmsg() unused send buffer\n",
	   MYPE(),pvm_mytid(),__FILE__,__LINE__);
    return -1;
  }
  cur_buf=&(CpvAccess(pvmc_bufarray)[CpvAccess(pvmc_sbufid)]);
  *((int *)((char *)msgbuf+bytes_packed)) = cur_buf->bytes;
  bytes_packed+=sizeof(int);
  *((int *)((char *)msgbuf+bytes_packed)) = cur_buf->tag;
  bytes_packed+=sizeof(int);
  *((int *)((char *)msgbuf+bytes_packed)) = cur_buf->tid;
  bytes_packed+=sizeof(int);
  *((int *)((char *)msgbuf+bytes_packed)) = cur_buf->num_items;
  bytes_packed+=sizeof(int);

#ifdef PVM_DEBUG
  PRINTF("Pe(%d) pvmc_packmsg: %d items packed for tag %d\n",
	 MYPE(),cur_buf->num_items,cur_buf->tag);
#endif
  cur_item=cur_buf->first_item;
  while(cur_item!=cur_buf->last_item) {
    *((int *)((char *)msgbuf+bytes_packed)) = cur_item->type;
    bytes_packed+=sizeof(int);
    *((int *)((char *)msgbuf+bytes_packed)) = cur_item->size;
    bytes_packed+=sizeof(int);
    cur_item=cur_item->nxt;
  }
    
  cur_item=cur_buf->first_item;
  while(cur_item!=cur_buf->last_item) {
    if (cur_item->size > 0) {
      memcpy((void *)((char *)msgbuf+bytes_packed),cur_item->data,
	     cur_item->size);
      bytes_packed+=cur_item->size;
    }
    cur_item=cur_item->nxt;
  }
  return bytes_packed;
}

int pvmc_unpackmsg(void *msgbuf, void *start_of_msg)
{
  pvmc_buffer *cur_buf;
  pvmc_item *cur_item, *nxt_item;
  int bytes_unpacked=0;
  int i;

  if ((CpvAccess(pvmc_rbufid)<=0) || (CpvAccess(pvmc_rbufid) >= MAX_BUFFERS) ||
      (CpvAccess(pvmc_bufarray)[CpvAccess(pvmc_rbufid)].refcount <= 0)) {
    PRINTF("Pe(%d) tid=%d:%s:%d pvmc_unpackmsg() uninitialized recv buffer\n",
	   MYPE(),__FILE__,__LINE__);
    return -1;
  }
  cur_buf = &(CpvAccess(pvmc_bufarray)[CpvAccess(pvmc_rbufid)]);
  pvmc_emptybuf(cur_buf);

  cur_buf->bytes = *((int *)((char *)start_of_msg+bytes_unpacked));
  bytes_unpacked += sizeof(int);
  cur_buf->tag = *((int *)((char *)start_of_msg+bytes_unpacked));
  bytes_unpacked += sizeof(int);
  cur_buf->tid = *((int *)((char *)start_of_msg+bytes_unpacked));
  bytes_unpacked += sizeof(int);
  cur_buf->num_items = *((int *)((char *)start_of_msg+bytes_unpacked));
  bytes_unpacked += sizeof(int);

#ifdef PVM_DEBUG
  PRINTF("Pe(%d) pvmc_unpackmsg: %d items unpacked for tag %d\n",
	 MYPE(),cur_buf->num_items,cur_buf->tag);
#endif
  if (msgbuf)
    cur_buf->data_buf = msgbuf;
  else cur_buf->data_buf = (void *)NULL;

  cur_item=(pvmc_item *)MALLOC(sizeof(pvmc_item));
  cur_buf->first_item=cur_item;
  cur_buf->cur_item=cur_item;

  if (cur_item==(pvmc_item *)NULL) {
    PRINTF("Pe(%d) tid=%d:%s:%d pvmc_unpackmsg() can't allocate memory\n",
	   MYPE(),pvm_mytid(),__FILE__,__LINE__);
    return -1;
  }

#if PVM_DEBUG
  PRINTF("Pe(%d) tid=%d:%s:%d pvmc_unpackmsg() unpacking %d messages.\n",
	MYPE(),pvm_mytid(),__FILE__,__LINE__,cur_buf->num_items);
#endif

  for(i=0;i<cur_buf->num_items;i++) {
    cur_item->type = *((int *)((char *)start_of_msg+bytes_unpacked));
    bytes_unpacked+=sizeof(int);
    cur_item->size = *((int *)((char *)start_of_msg+bytes_unpacked));
    bytes_unpacked+=sizeof(int);
    
    nxt_item=(pvmc_item *)MALLOC(sizeof(pvmc_item));

    if (!nxt_item) {
      PRINTF("Pe(%d) tid=%d:%s:%d pvmc_unpackmsg() can't allocate memory\n",
	     MYPE(),pvm_mytid(),__FILE__,__LINE__);
      return -1;
    }
    cur_item->nxt = nxt_item;
    cur_item = nxt_item;
  }
  
  cur_item->type = 0;
  cur_item->size = 0;
  cur_item->free_data = FALSE;
  cur_item->data = (char *) NULL;
  cur_item->nxt = (pvmc_item *) NULL;

  cur_buf->last_item = cur_item;
    
  cur_item = cur_buf->first_item;
  while(cur_item!=cur_buf->last_item) {
    if (cur_item->size > 0) {
      cur_item->free_data=FALSE;
      cur_item->data = (void *)((char *)start_of_msg+bytes_unpacked);
      bytes_unpacked+=cur_item->size;
    }
    else cur_item->data=NULL;
    cur_item = cur_item->nxt;
  }

  return bytes_unpacked;
}

int pvmc_gettag(void *msgbuf)
{
  return *((int *)msgbuf+1);
}

void *pvmc_mkitem(int nbytes, int type)
{
  pvmc_buffer *buf;
  void *databuf;

  if ((CpvAccess(pvmc_sbufid)<=0) || (CpvAccess(pvmc_sbufid) >= MAX_BUFFERS) ||
      (CpvAccess(pvmc_bufarray)[CpvAccess(pvmc_sbufid)].refcount <= 0)) {
    PRINTF("Pe(%d) tid=%d:%s:%d pvmc_mkitem() unused send buffer\n",
	   MYPE(),pvm_mytid(),__FILE__,__LINE__);
    return NULL;
  }
  buf = &(CpvAccess(pvmc_bufarray)[CpvAccess(pvmc_sbufid)]);

  buf->last_item->type=type;
  buf->last_item->size=nbytes;
  databuf=MALLOC(nbytes);
  if (!databuf) {
    PRINTF("Pe(%d) tid=%d:%s:%d pvmc_mkitem() can't allocate data space\n",
	   MYPE(),pvm_mytid(),__FILE__,__LINE__);
    return databuf;
  }
  buf->last_item->free_data=TRUE;
  buf->last_item->data=databuf;
  buf->last_item->nxt=(pvmc_item *)MALLOC(sizeof(pvmc_item));
  if (buf->last_item->nxt==NULL) {
    PRINTF("Pe(%d) tid=%d:%s:%d pvmc_mkitem() can't allocate new item\n",
	   MYPE(),pvm_mytid(),__FILE__,__LINE__);
    return NULL;
  }

  buf->last_item=buf->last_item->nxt;
  buf->last_item->type=0;
  buf->last_item->size=0;
  buf->last_item->free_data=FALSE;
  buf->last_item->data=(char *)NULL;
  buf->last_item->nxt=(pvmc_item *)NULL;
  buf->num_items++;

  return databuf;
}

void *pvmc_getitem(int n_bytes, int type)
{
  pvmc_buffer *buf;
  pvmc_item *item;
  void *data;

  if ((CpvAccess(pvmc_rbufid)<=0) || (CpvAccess(pvmc_rbufid) >= MAX_BUFFERS) ||
      (CpvAccess(pvmc_bufarray)[CpvAccess(pvmc_rbufid)].refcount <= 0)) {
    PRINTF("Pe(%d) tid=%d:%s:%d pvmc_getitem() uninitialized recv buffer\n",
	   MYPE(),pvm_mytid(),__FILE__,__LINE__);
    return NULL;
  }
  buf = &(CpvAccess(pvmc_bufarray)[CpvAccess(pvmc_rbufid)]);

  item = buf->cur_item;

  if (item==buf->last_item) {
    PRINTF("Pe(%d) tid=%d:%s:%d pvmc_getitem() no more items\n",
	  MYPE(),pvm_mytid(), __FILE__,__LINE__);
    data=NULL;
  } else if (item->data==(void *)NULL) {
    PRINTF("Pe(%d) tid=%d:%s:%d pvmc_getitem() uninitialized data\n",
	  MYPE(),pvm_mytid(), __FILE__,__LINE__);
    data=NULL;
  } else if (item->size < n_bytes) {
    PRINTF("Pe(%d) tid=%d:%s:%d pvmc_getitem() data size mismatch\n",
	  MYPE(),pvm_mytid(), __FILE__,__LINE__);
    data=NULL;
  } else if (item->type != type) {
    PRINTF("Pe(%d) tid=%d:%s:%d pvmc_getitem() type mismatch\n",
	  MYPE(),pvm_mytid(), __FILE__,__LINE__);
    data=NULL;
  } else {
    data=item->data;
  }

  buf->cur_item = buf->cur_item->nxt;
  return data;
}

void *pvmc_getstritem(int *n_bytes)
{
  pvmc_buffer *buf;
  pvmc_item *item;
  void *data;

  if ((CpvAccess(pvmc_rbufid)<=0) || (CpvAccess(pvmc_rbufid) >= MAX_BUFFERS) ||
      (CpvAccess(pvmc_bufarray)[CpvAccess(pvmc_rbufid)].refcount <= 0)) {
    PRINTF("Pe(%d) tid=%d:%s:%d pvmc_getstritem() uninitialized recv buffer\n",
	   MYPE(),pvm_mytid(),__FILE__,__LINE__);
    return NULL;
  }
  buf = &(CpvAccess(pvmc_bufarray)[CpvAccess(pvmc_rbufid)]);

  item = buf->cur_item;
  *n_bytes = item->size;

  if (item==buf->last_item) {
    PRINTF("Pe(%d) tid=%d:%s:%d pvmc_getstritem() no more items\n",
	   MYPE(),pvm_mytid(),__FILE__,__LINE__);
    data=NULL;
  } else if (item->data==(void *)NULL) {
    PRINTF("Pe(%d) tid=%d:%s:%d pvmc_getstritem() uninitialized data\n",
	   MYPE(),pvm_mytid(),__FILE__,__LINE__);
    data=NULL;
  } else if (item->type != PVM_STR) {
    PRINTF("Pe(%d) tid=%d:%s:%d pvmc_getstritem() type mismatch\n",
	   MYPE(),pvm_mytid(),__FILE__,__LINE__);
    data=NULL;
  } else {
    data=item->data;
  }

  buf->cur_item = buf->cur_item->nxt;
  return data;
}

