#include <converse.h>
#include <stdarg.h>

/******************************************************************************
 *
 * Data definitions and globals
 *
 *****************************************************************************/

#define  CrcGenericAlign 8
#define  CrcGenericAlignInt(n) (((n)+CrcGenericAlign-1)&(~(CrcGenericAlign-1)))
#define  CrcGenericAlignPtr(p) ((void*)CrcGenericAlignInt((size_t)(p)))

typedef struct buffer buffer;
typedef struct eltset    *eltset;
typedef struct single    *single;
typedef struct reduction *reduction;

struct buffer
{
  unsigned char *data;
  int            fill;
  int            size;
};

struct eltset
{
  eltset next;
  CPath path;
  int eltno;
  int root;   /* who is the root of the set */
  int inset;  /* True if I am in the set */
  int parent; /* my parent in the set */
  int child1; /* my first child in the set */
  int child2; /* my second child in the set */
  reduction reductions; /* list of reductions in progress over this set */
  int nlocals;
  single locals[1];
};

struct single
{
  single next;
  CPath path;
  int eltno;
  int waiting;
  CmmTable mm;
  CthThread thread;
};

struct reduction
{
  eltset over;
  CPath dest; int desteltno;
  int reducer;
  int vecsize; int eltsize;
  char *msg_begin, *msg_data, *msg_end;
  int anydata, waitcount;
  struct reduction *next;
  int ntags;
  int tags[1];
};

CpvStaticDeclare(int *,  PElist);
CpvStaticDeclare(char *, PEflags);
CpvStaticDeclare(int, seqno);
CpvStaticDeclare(int, CPathSendIdx);
CpvStaticDeclare(int, CPathReduceIdx);
CpvStaticDeclare(eltset *, EltsetTable);
CpvStaticDeclare(int, EltsetTabsize);
CpvStaticDeclare(single *, SingleTable);
CpvStaticDeclare(int, SingleTabsize);
CpvStaticDeclare(reduction *, ReduceTable);
CpvStaticDeclare(int, ReduceTabsize);
CtvStaticDeclare(single, thisthread);

/******************************************************************************
 *
 * bufalloc
 *
 * Given a pointer to a buffer, allocates N aligned bytes at the end
 * of the buffer.  Resizes the buffer if necessary.
 *
 *****************************************************************************/

static void *bufalloc(buffer *b, int n)
{
  int fill = CrcGenericAlignInt(b->fill);
  int nfill = fill+n;
  int nsize; char *ndata;

  b->fill = nfill;
  if (nfill > b->size) {
    nsize = nfill*2;
    b->size = nsize;
    ndata = (char*)CmiAlloc(nsize);
    memcpy(ndata, b->data, fill);
    CmiFree(b->data);
    b->data = ndata;
  }
  return b->data + fill;
}

/******************************************************************************
 *
 * CPathIndicesToEltno
 * CPathEltnoToIndices
 *
 * Converts an index-set to a more compact representation and back again.
 *
 *****************************************************************************/

unsigned int CPathIndicesToEltno(int nsizes, int *sizes, int *indices)
{
  unsigned int size, eltno; int i, index;
  eltno=0;
  for (i=0; i<nsizes; i++) {
    size = sizes[i];
    index = indices[i];
    eltno *= (size + 1);
    if (index == CPATH_ALL) index = size;
    eltno += index;
  }
  return eltno;
}

void CPathEltnoToIndices(int nsizes, int *sizes,
			 unsigned int eltno, int *indices)
{
  unsigned int size; int i, index;
  for (i=nsizes-1; i>=0; i--) {
    size = sizes[i];
    index = eltno % (size + 1);
    eltno /= (size + 1);
    if (index == size) index = CPATH_ALL;
    indices[i] = index;
  }
}

/******************************************************************************
 *
 * CPathGetEltset
 *
 * Given an element set specified as a path and set of indices (possibly
 * with wildcards), returns the information about that set of elements.
 *
 * Note to reimplementors: this function has a lot of local variables.
 * It just goes down the list and fills them in one by one, top to bottom.
 *
 *****************************************************************************/

int CPathMap(CPath *path, int *indices)
{
  CPathMapFn mapper = (CPathMapFn)CmiHandlerToFunction(path->mapfn);
  return mapper(path, indices) % CmiNumPes();
}

int CPathRoot(CPath *path, int *indices)
{
  int nindices[13], i;
  for (i=0; i<path->nsizes; i++) {
    nindices[i] = indices[i];
    if (nindices[i] == CPATH_ALL)
      nindices[i] = path->sizes[i] - 1;
  }
  return CPathMap(path, nindices);
}

void CPathExecute(single t)
{
  unsigned int indices[13];
  CthVoidFn fn = CmiHandlerToFunction(t->path.startfn);
  CtvAccess(thisthread) = t;
  CPathEltnoToIndices(t->path.nsizes, t->path.sizes, t->eltno, indices);
  fn(&(t->path), indices);
  /* Insert garbage-collection code here */
}

single CPathGetSingle(CPath *path, int *indices)
{
  single *table = CpvAccess(SingleTable);
  unsigned int eltno = CPathIndicesToEltno(path->nsizes, path->sizes, indices);
  unsigned int hashval = path->seqno ^ path->creator ^ eltno;
  unsigned int bucket = hashval % CpvAccess(SingleTabsize);
  single result;

  for (result=table[bucket]; result; result=result->next) {
    if ((result->path.seqno == path->seqno)&&
	(result->path.creator == path->creator)&&
	(result->eltno == eltno))
      return result;
  }
  
  result = (single)malloc(sizeof(struct single));
  _MEMCHECK(result);
  result->path = *path;
  result->eltno = eltno;
  result->waiting = 0;
  result->mm = CmmNew();
  result->thread = CthCreate(CPathExecute, result, 0);
  result->next = table[bucket];
  table[bucket] = result;
  CthAwaken(result->thread);
  return result;
}

eltset CPathAllocEltset(eltset old, int nlocals)
{
  return (eltset)realloc(old, sizeof(struct eltset) + nlocals*sizeof(single));
}

void CPathDecrementOdometer(int *odometer, int *sizes, int nvary, int *vary)
{
  int i = 0;
  while (odometer[vary[i]]==0) {
    odometer[vary[i]] = sizes[vary[i]] - 1;
    i++;
  }
  odometer[vary[i]]--;
}

eltset CPathGetEltset(CPath *path, int *indices)
{
  int           nsizes;      /* extracted from path */
  int          *sizes;       /* extracted from path */
  unsigned int  eltno;       /* element number associated with indices */
  eltset       *table;       /* hash table containing eltsets */
  unsigned int  hashval;     /* hash value of this eltset */
  unsigned int  bucket;      /* hash table bucket holding this eltset */
  int           nelts;       /* total size of eltset (not just local) */
  int           nvary;       /* number of indices equal to CPATH_ALL */
  int           vary[13];    /* list of indices equal to CPATH_ALL */
  int           mype;        /* my processor number */
  int           root;        /* processor holding first elt of set */
  char         *peflags;     /* set of PEs containing elts */
  int          *pelist;      /* list of PEs containing elts */
  int           pecount;     /* count of PEs containing elts */
  int           peself;      /* My position in the PE list */
  eltset        result;      /* result buffer */
  int           resultfill;  /* number of locals filled in result buffer */
  int           resultsize;  /* number of locals available in result buffer */
  int tsize, tpe, teltno, i, j;  /* temporary vars */

  /* Compute nsizes, sizes */
  nsizes = path->nsizes;
  sizes  = path->sizes;
  
  /* Compute eltno, hashval and bucket, table */
  eltno = CPathIndicesToEltno(nsizes, sizes, indices);
  hashval = path->seqno ^ path->creator ^ eltno;
  bucket = hashval % CpvAccess(EltsetTabsize);
  table = CpvAccess(EltsetTable);
  
  /* Scan for eltset in hash table.  If present, return it. */
  for (result=table[bucket]; result; result=result->next) {
    if ((result->eltno == eltno) &&
	(result->path.seqno == path->seqno) &&
	(result->path.creator == path->creator)) {
      return result;
    }
  }
  
  /* Compute nelts, vary, nvary. Initialize indices to starting values. */
  nelts = 1; nvary = 0;
  for (i=0; i<nsizes; i++) {
    if (indices[i] == CPATH_ALL) {
      tsize = sizes[i];
      nelts *= tsize;
      indices[i] = tsize - 1;
      vary[nvary++] = i;
    }
  }
  
  /* Compute root, mype */
  root = CPathMap(path, indices);
  mype = CmiMyPe();
  
  /* put nelts==1 optimization here */
  
  /* Initialize the buffers */
  pelist  = CpvAccess(PElist);
  peflags = CpvAccess(PEflags);
  pecount = 0;
  peself  = -1;
  resultfill = 0;
  resultsize = (nelts > 10) ? 10 : nelts;
  result = CPathAllocEltset(0, resultsize);
  
  /* Gen all elts. Compute peflags, pelist, pecount, peself, locals */
  i = 0;
  while (1) {
    tpe = CPathMap(path, indices);
    if (peflags[tpe]==0) {
      if (tpe == mype) peself = pecount;
      peflags[tpe] = 1;
      pelist[pecount++] = tpe;
    }
    if (tpe == mype) {
      if (resultfill == resultsize) {
	resultsize <<= 1;
	result = CPathAllocEltset(result, resultsize);
      }
      result->locals[resultfill++] = CPathGetSingle(path, indices);
    }
    i++; if (i==nelts) break;
    CPathDecrementOdometer(indices, sizes, nvary, vary);
  }
  
  /* clear the PEflags for later use, reset indices */
  for (i=0; i<pecount; i++) peflags[pelist[i]] = 0;
  for (i=0; i<nvary; i++) indices[vary[i]] = CPATH_ALL;
  
  /* fill in fields of result structure */
  result = CPathAllocEltset(result, resultfill);
  result->next = table[bucket];
  table[bucket] = result;
  result->path = (*path);
  result->eltno = eltno;
  result->root = root;
  if (peself < 0) {
    result->inset = 0;
    result->parent = -1;
    result->child1 = -1;
    result->child2 = -1;
  } else {
    i = (peself+1)<<1;
    j = i-1;
    result->inset = 1;
    result->parent = peself ? pelist[(peself-1)>>1] : -1;
    result->child1 = (i<pecount) ? pelist[i] : -1;
    result->child2 = (j<pecount) ? pelist[j] : -1;
  }
  result->nlocals = resultfill;
  return result;
}

/******************************************************************************
 *
 * CPathSend
 *
 * Send a message to a thread or set of threads.  If you send to a set
 * of elements, finds the root of the spanning tree and sends to that
 * element.
 *
 *****************************************************************************/

void CPathSend(int key, ...)
{
  buffer buf; va_list p; CPath *path;
  char *src, *cpy; int root, i, vecsize, eltsize, size, idx, ntags, eltno;
  int indices[13], tags[256], *tagv;
  
  buf.data = (unsigned char*)CmiAlloc(128);
  buf.fill = CmiMsgHeaderSizeBytes;
  buf.size = 128;
  va_start(p, key);
  
  /* Insert path and indices into message */
  switch(key) {
  case CPATH_DEST:
    path = va_arg(p, CPath *);
    for (i=0; i<path->nsizes; i++)
      indices[i] = va_arg(p, int);
    break;
  case CPATH_DESTELT:
    path = va_arg(p, CPath *);
    eltno = va_arg(p, int);
    CPathEltnoToIndices(path->nsizes, path->sizes, eltno, indices);
    break;
  default: goto synerr;
  }
  cpy = bufalloc(&buf,sizeof(CPath));
  memcpy(cpy, path, sizeof(CPath));
  cpy = bufalloc(&buf,path->nsizes * sizeof(int));
  memcpy(cpy, indices, path->nsizes * sizeof(int));
  root = CPathRoot(path, indices);
  
  /* Insert tags into message */
  key = va_arg(p, int);
  switch (key) {
  case CPATH_TAG:
    ntags = 1;
    tags[0] = va_arg(p, int);
    tagv = tags;
    break;
  case CPATH_TAGS:
    ntags = va_arg(p, int);
    for (i=0; i<ntags; i++)
      tags[i] = va_arg(p, int);
    tagv = tags;
    break;
  case CPATH_TAGVEC:
    ntags = va_arg(p, int);
    tagv = va_arg(p, int*);
    break;
  }
  cpy = bufalloc(&buf,(ntags+1)*sizeof(int));
  ((int*)cpy)[0] = ntags;
  for (i=0; i<ntags; i++)
    ((int*)cpy)[i+1] = tagv[i];
  
  /* Insert data into message */
  key = va_arg(p, int);
  switch (key) {
  case CPATH_BYTES:
    size = va_arg(p, int);
    src = va_arg(p, char*);
    cpy = bufalloc(&buf,sizeof(int)+size);
    *((int*)cpy) = size;
    memcpy(cpy + sizeof(int), src, size);
    break;
  case CPATH_REDBYTES:
    vecsize = va_arg(p, int);
    eltsize = va_arg(p, int);
    size = vecsize * eltsize;
    src = va_arg(p, char *);
    cpy = bufalloc(&buf,2*sizeof(int)+(vecsize*eltsize));
    ((int*)cpy)[0] = size;
    ((int*)cpy)[1] = vecsize;
    memcpy(cpy + sizeof(int) + sizeof(int), src, size);
    break;
  default: goto synerr;
  }
  key = va_arg(p, int);
  if (key!=CPATH_END) goto synerr;

  /* send message to root of eltset */
  CmiSetHandler(buf.data, CpvAccess(CPathSendIdx));
  CmiSyncSendAndFree(root, buf.fill, buf.data);
  return;
  
synerr:
  CmiError("CPathSend: syntax error in argument list.\n");
  exit(1);
}

void CPathSendHandler(char *decoded)
{
  char *p; CPath *path;
  int *indices, *tags, ntags, mype, i, *tail, len;
  eltset set; single sing; char *mend;

  mype = CmiMyPe();
  path = (CPath*)CrcGenericAlignPtr(decoded + CmiMsgHeaderSizeBytes);
  indices = (int*)CrcGenericAlignPtr(path + 1);
  tags = (int*)CrcGenericAlignPtr(indices + path->nsizes);
  ntags = *tags++;
  tail = (int*)CrcGenericAlignPtr(tags + ntags);
  len = *tail++;
  mend = ((char*)tail)+len;
  /* put optimization here: check for a send which is just to me */
  set = CPathGetEltset(path, indices);
  if (set->child1 >= 0) CmiSyncSend(set->child1, mend-decoded, decoded);
  if (set->child2 >= 0) CmiSyncSend(set->child2, mend-decoded, decoded);
  /* this next statement is putting a refcount in the converse core */
  *((int*)decoded) = set->nlocals;
  for (i=0; i<set->nlocals; i++) {
    sing = set->locals[i];
    CmmPut(sing->mm, ntags, tags, decoded);
    if (sing->waiting) {
      sing->waiting = 0;
      CthAwaken(sing->thread);
    }
  }
}

void *CPathRecv(int key, ...)
{
  va_list p; int i, ntags, *tags; int tagbuf[256], rtags[256];
  int *hlen; char **buffer; void *msg;
  single t = CtvAccess(thisthread);
  
  va_start(p, key);
  switch(key) {
  case CPATH_TAG:
    ntags = 1;
    tagbuf[0] = va_arg(p, int);
    tags = tagbuf;
    break;
  case CPATH_TAGS:
    ntags = va_arg(p, int);
    if (ntags>256) goto sizerr;
    for (i=0; i<ntags; i++)
      tagbuf[i] = va_arg(p, int);
    tags = tagbuf;
    break;
  default: goto synerr;
  }
  
  key = va_arg(p, int);
  if (key != CPATH_END) goto synerr;
  /* do something about the return tags someday */
  while (1) {
    msg = CmmGet(t->mm, ntags, tags, rtags);
    if (msg) break;
    t->waiting = 1;
    CthSuspend();
  }
  return msg;
  
synerr:
  CmiError("CPathRecv: syntax error in argument list.\n");
  exit(1);
sizerr:
  CmiError("CPathRecv: too many tags.\n");
  exit(1);
}

void CPathMsgDecodeBytes(void *msg, int *len, void *bytes)
{
  CPath *path; int *indices; unsigned int *tags, ntags, *tail;
  path = (CPath*)CrcGenericAlignPtr(((char*)msg)+CmiMsgHeaderSizeBytes);
  indices = (int*)CrcGenericAlignPtr(path + 1);
  tags = CrcGenericAlignPtr(indices + path->nsizes);
  ntags = *tags++;
  tail = CrcGenericAlignPtr(tags + ntags);
  *len = *tail++;
  *((void**)bytes) = (void*)tail;
}

void CPathMsgDecodeReduction(void *msg,int *vecsize,int *eltsize,void *bytes)
{
  CPath *path; int *indices; unsigned int *tags, ntags, *tail, size;
  path = (CPath*)CrcGenericAlignPtr(((char*)msg)+CmiMsgHeaderSizeBytes);
  indices = (int*)CrcGenericAlignPtr(path + 1);
  tags = CrcGenericAlignPtr(indices + path->nsizes);
  ntags = *tags++;
  tail = CrcGenericAlignPtr(tags + ntags);
  size = *tail++;
  *vecsize = *tail++;
  *eltsize = size / (*vecsize);
  *((void**)bytes) = (void*)tail;
}

void CPathMsgFree(void *msg)
{
  int ref = ((int*)msg)[0] - 1;
  if (ref==0) {
    CmiFree(msg);
  } else {
    ((int*)msg)[0] = ref;
  }
}

void CPathMakeArray(CPath *path, int startfn, int mapfn, ...)
{
  int size, seq, pe, nsizes; va_list p;
  va_start(p, mapfn);
  nsizes = 0;
  while (1) {
    size = va_arg(p, int);
    if (size==0) break;
    if (nsizes==13) {
      CmiError("CPathMakeArray: Limit of 13 dimensions.\n");
      exit(1);
    }
    path->sizes[nsizes++] = size;
  }
  path->creator = CmiMyPe();
  path->seqno = CpvAccess(seqno)++;
  path->mapfn = mapfn;
  path->startfn = startfn;
  path->nsizes = nsizes;
}

/******************************************************************************
 *
 * CPathReduce(CPATH_OVER, array, i, j, k,
 *             CPATH_TAGS, ...
 *	       CPATH_REDUCER, fn,
 *             CPATH_BYTES, vecsize, eltsize, data,
 *             CPATH_DEST, array, i, j, k,
 *             CPATH_END);
 *
 *****************************************************************************/

void CPathMergeReduction(reduction red, void *data);

void CPathReduceMismatch()
{
  CmiError("CPathReduce: all members of reduction do not agree on reduction parameters.\n");
  exit(1);
}

void CPathCreateRedmsg(reduction red)
{
  /* create a reduction message, leaving the data area */
  /* uninitialized. format of reduction message is: */
  /* over&dest,overeltno&desteltno&reducer&vecsize&eltsize&ntags,tags,data*/
  
  int o_paths, o_params, o_tags, o_data, o_end;
  int i, *t; char *msg;
  
  o_paths = CrcGenericAlignInt(CmiMsgHeaderSizeBytes);
  o_params = CrcGenericAlignInt(o_paths + 2*sizeof(CPath));
  o_tags = CrcGenericAlignInt(o_params + 6*sizeof(int));
  o_data = CrcGenericAlignInt(o_tags + red->ntags*sizeof(int));
  o_end = o_data + (red->vecsize * red->eltsize);
  
  msg = (char*)CmiAlloc(o_end);
  CmiSetHandler(msg, CpvAccess(CPathReduceIdx));
  ((CPath*)(msg+o_paths))[0] = red->over->path;
  ((CPath*)(msg+o_paths))[1] = red->dest;
  ((int*)(msg+o_params))[0] = red->over->eltno;
  ((int*)(msg+o_params))[1] = red->desteltno;
  ((int*)(msg+o_params))[2] = red->reducer;
  ((int*)(msg+o_params))[3] = red->vecsize;
  ((int*)(msg+o_params))[4] = red->eltsize;
  ((int*)(msg+o_params))[5] = red->ntags;
  t = (int*)(msg+o_tags);
  for (i=0; i<red->ntags; i++) *t++ = red->tags[i];
  
  red->msg_begin = msg;
  red->msg_data = msg + o_data;
  red->msg_end = msg + o_end;
}

reduction CPathGetReduction(eltset set, int ntags, int *tags,
			    int vecsize, int eltsize,
			    int reducer, CPath *dest, int desteltno)
{
  reduction red; int i;
  
  for (red=set->reductions; red; red=red->next) {
    if (red->ntags != ntags) continue;
    for (i=0; i<ntags; i++)
      if (red->tags[i] != tags[i]) continue;
    
    if (red->vecsize != vecsize) CPathReduceMismatch();
    if (red->eltsize != eltsize) CPathReduceMismatch();
    if (red->reducer != reducer) CPathReduceMismatch();
    if (red->dest.creator != dest->creator) CPathReduceMismatch();
    if (red->dest.seqno   != dest->seqno) CPathReduceMismatch();
    if (red->desteltno    != desteltno) CPathReduceMismatch();
    
    return red;
  }
  
  red = (reduction)malloc(sizeof(struct reduction) + ntags*sizeof(int));
  _MEMCHECK(red);
  
  red->over = set;
  red->ntags = ntags;
  for (i=0; i<ntags; i++) red->tags[i] = tags[i];
  red->vecsize = vecsize;
  red->eltsize = eltsize;
  red->reducer = reducer;
  red->dest = (*dest);
  red->desteltno = desteltno;
  
  red->anydata = 0;
  red->waitcount = set->nlocals;
  if (set->child1 >= 0) red->waitcount++;
  if (set->child2 >= 0) red->waitcount++;

  CPathCreateRedmsg(red);

  red->next = set->reductions;
  set->reductions = red;

  return red;
}

void CPathReduceHandler(void *decoded)
{
  /* over&dest,overeltno&desteltno&reducer&vecsize&eltsize&ntags,tags,data*/
  CPath *paths, *over, *dest; int *params, *tags;
  int o_paths, o_params, o_tags, o_data, o_end;
  int overeltno, desteltno, reducer, vecsize, eltsize, ntags;
  eltset set; reduction red; int overidx[13]; void *data;
  
  paths = (CPath*)CrcGenericAlignPtr(decoded + CmiMsgHeaderSizeBytes);
  params = (int*)CrcGenericAlignPtr(paths + 2);
  over = paths+0;
  dest = paths+1;
  overeltno = params[0];
  desteltno = params[1];
  reducer   = params[2];
  vecsize   = params[3];
  eltsize   = params[4];
  ntags     = params[5];
  tags = (int*)CrcGenericAlignPtr(params+6);
  data = (void*)CrcGenericAlignPtr(tags + ntags);
  
  CPathEltnoToIndices(over->nsizes, over->sizes, overeltno, overidx);
  set = CPathGetEltset(over, overidx);
  red = CPathGetReduction
    (set, ntags, tags, vecsize, eltsize, reducer, dest, desteltno);
  CPathMergeReduction(red, data);
}

void CPathForwardReduction(reduction red)
{
  int pe; eltset set; reduction *hred;
  
  pe = red->over->parent;
  if (pe >= 0) {
    CmiSyncSendAndFree(pe, (red->msg_end) - (red->msg_begin), red->msg_begin);
  } else {
    CPathSend(CPATH_DESTELT, &(red->dest), red->desteltno,
	      CPATH_TAGVEC, red->ntags, red->tags,
	      CPATH_REDBYTES, red->vecsize, red->eltsize, red->msg_data,
	      CPATH_END);
    CmiFree(red->msg_begin);
  }
  /* free the reduction */
  set = red->over;
  hred = &(set->reductions);
  while (*hred) {
    if (*hred == red) *hred = red->next;
    else hred = &((*hred)->next);
  }
  free(red);
}

void CPathMergeReduction(reduction red, void *data)
{
  if (red->anydata) {
    CmiHandlerToFunction(red->reducer)(red->vecsize, red->msg_data, data);
  } else {
    memcpy(red->msg_data, data, red->vecsize * red->eltsize);
  }
  red->anydata = 1;
  red->waitcount--;
  if (red->waitcount==0) CPathForwardReduction(red);
}

void CPathReduce(int key, ...)
{
  CPath *over; int overidx[13]; 
  CPath *dest; int destidx[13];
  int desteltno;
  int ntags, tags[256];
  int reducer;
  int vecsize, eltsize; void *data;
  va_list p; int i;
  eltset set; reduction red;


  va_start(p, key);
  if (key != CPATH_OVER) goto synerr;
  over = va_arg(p, CPath *);
  for (i=0; i<over->nsizes; i++)
    overidx[i] = va_arg(p, int);
  
  key = va_arg(p, int);
  switch(key) {
  case CPATH_TAG:
    ntags = 1;
    tags[0] = va_arg(p, int);
    break;
  case CPATH_TAGS:
    ntags = va_arg(p, int);
    if (ntags > 256) goto synerr;
    for (i=0; i<ntags; i++)
      tags[i] = va_arg(p, int);
    break;
  default: goto synerr;
  }
  
  key = va_arg(p, int);
  if (key != CPATH_REDUCER) goto synerr;
  reducer = va_arg(p,int);
  
  key = va_arg(p, int);
  if (key != CPATH_DEST) goto synerr;
  dest = va_arg(p, CPath *);
  for (i=0; i<dest->nsizes; i++)
    destidx[i] = va_arg(p,int);
  desteltno = CPathIndicesToEltno(dest->nsizes, dest->sizes, destidx);
  
  key = va_arg(p, int);
  if (key != CPATH_BYTES) goto synerr;
  vecsize = va_arg(p, int);
  eltsize = va_arg(p, int);
  data = va_arg(p, void *);
  
  key = va_arg(p, int);
  if (key != CPATH_END) goto synerr;

  set = CPathGetEltset(over, overidx);
  red = CPathGetReduction
    (set, ntags, tags, vecsize, eltsize, reducer, dest, desteltno);
  CPathMergeReduction(red, data);
  return;
  
synerr:
  CmiError("CPathReduce: arglist must have these clauses: OVER, TAGS, REDUCER, DEST, BYTES, END (in that order).\n");
  exit(1);
}

void CPathModuleInit()
{
  CpvInitialize(int, seqno);
  CpvInitialize(int, CPathSendIdx);
  CpvInitialize(int, CPathReduceIdx);
  CpvInitialize(eltset *, EltsetTable);
  CpvInitialize(int, EltsetTabsize);
  CpvInitialize(single *, SingleTable);
  CpvInitialize(int, SingleTabsize);
  CpvInitialize(reduction *, ReduceTable);
  CpvInitialize(int, ReduceTabsize);
  CtvInitialize(single, thisthread);
  CpvInitialize(char *, PEflags);
  CpvInitialize(int *, PElist);
  
  CpvAccess(seqno) = 0;
  CpvAccess(CPathSendIdx) = CmiRegisterHandler(CPathSendHandler);
  CpvAccess(CPathReduceIdx) = CmiRegisterHandler(CPathReduceHandler);
  CpvAccess(EltsetTabsize) = 1091;
  CpvAccess(EltsetTable) = (eltset*)calloc(1091,sizeof(eltset));
  CpvAccess(SingleTabsize) = 1091;
  CpvAccess(SingleTable) = (single*)calloc(1091,sizeof(single));
  CpvAccess(ReduceTabsize) = 1091;
  CpvAccess(ReduceTable) = (reduction*)calloc(1091,sizeof(reduction));
  
  CpvAccess(PEflags) = (char*)calloc(1,CmiNumPes());
  CpvAccess(PElist) = (int*)malloc(CmiNumPes()*sizeof(int));
  _MEMCHECK(CpvAccess(PElist));
}
