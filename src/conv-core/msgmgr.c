/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include <converse.h>

typedef struct CmmEntryStruct *CmmEntry;

struct CmmEntryStruct
{
  CmmEntry next;
  void    *msg;
  int      ntags;
  int      tags[1];
};

struct CmmTableStruct
{
  CmmEntry  first;
  CmmEntry *lasth;
};


CmmTable CmmNew()
{
  CmmTable result = (CmmTable)CmiAlloc(sizeof(struct CmmTableStruct));
  result->first = 0;
  result->lasth = &(result->first);
  return result;
}

void CmmFree(t)
CmmTable t;
{
  CmmEntry n;
  CmmEntry e = t->first;
  while (e) {
    CmiFree(e->msg);
    n = e->next;
    CmiFree(e);
    e = n;
  }
  CmiFree(t);
}

void CmmPut(t, ntags, tags, msg)
CmmTable t;
int ntags;
int *tags;
void *msg;
{
  int i;
  CmmEntry e=(CmmEntry)CmiAlloc(sizeof(struct CmmEntryStruct)+(ntags*sizeof(int)));
  e->next = 0;
  e->msg = msg;
  e->ntags = ntags;
  for (i=0; i<ntags; i++) e->tags[i] = tags[i];
  *(t->lasth) = e;
  t->lasth = &(e->next);
}

static int CmmTagsMatch(ntags1, tags1, ntags2, tags2)
int ntags1; int *tags1; int ntags2; int *tags2;
{
  int ntags = ntags1;
  if (ntags1 != ntags2) return 0;
  while (1) {
    int tag1, tag2;
    if (ntags == 0) return 1;
    ntags--;
    tag1 = *tags1++;
    tag2 = *tags2++;
    if (tag1==tag2) continue;
    if (tag1==CmmWildCard) continue;
    if (tag2==CmmWildCard) continue;
    return 0;
  }
}

void *CmmFind(t, ntags, tags, rtags, del)
CmmTable t;
int ntags;
int *tags;
int *rtags;
int del;
{
  CmmEntry *enth; CmmEntry ent; void *msg; int i;
  enth = &(t->first);
  while (1) {
    ent = (*enth);
    if (ent==0) return 0;
    if (CmmTagsMatch(ntags, tags, ent->ntags, ent->tags)) {
      if (rtags) for (i=0; i<ntags; i++) rtags[i] = ent->tags[i];
      msg = ent->msg;
      if (del) {
	CmmEntry next = ent->next;
	(*enth) = next;
	if (next == 0) t->lasth = enth;
	CmiFree(ent);
      }
      return msg;
    }
    enth = &(ent->next);
  }
}

int CmmEntries(t)
CmmTable t;
{
  int n = 0;
  CmmEntry e = t->first;
  while (e) {
    e = e->next;
    n++;
  }
  return n;
}

#define MSGSIZE(msg) (((int*)((char*)(msg)-(2*sizeof(int))))[0])

int CmmPackBufSize(CmmTable t)
{
  int nentries = CmmEntries(t);
  CmmEntry e = t->first;
  int size = sizeof(int); /* store nentries */
  while(e) {
    /* messages are always allocated with CmiAlloc */
    int msize = MSGSIZE(e->msg);
    /* first two integers store ntags, and msize */
    size += 2*sizeof(int)+(e->ntags*sizeof(int))+msize;
  }
  return size;
}

void CmmPackTable(CmmTable t, void *buffer)
{
  char *buf = (char*) buffer;
  int nentries = CmmEntries(t);
  CmmEntry e = t->first;
  memcpy(buf, &nentries, sizeof(int)); buf += sizeof(int);
  while(e) {
    int msize = MSGSIZE(e->msg);
    memcpy(buf, &(e->ntags), sizeof(int)); buf += sizeof(int);
    memcpy(buf, &msize, sizeof(int)); buf += sizeof(int);
    memcpy(buf, e->tags, e->ntags*sizeof(int)); buf += e->ntags*sizeof(int);
    memcpy(buf, e->msg, msize); buf += msize;
  }
  CmmFree(t);
}

CmmTable CmmUnpackTable(void *buffer)
{
  char *buf = (char*) buffer;
  int i, nentries;
  CmmTable t = CmmNew();
  memcpy(&nentries, buf, sizeof(int)); buf += sizeof(int);
  for(i=0;i<nentries;i++)
  {
    int ntags, msize, *tags;
    void *msg;
    memcpy(&ntags, buf, sizeof(int)); buf += sizeof(int);
    memcpy(&msize, buf, sizeof(int)); buf += sizeof(int);
    tags = (int*) buf;
    buf += ntags*sizeof(int);
    msg = CmiAlloc(msize);
    memcpy(msg, buf, msize); buf += msize;
    CmmPut(t, ntags, tags, msg);
  }
  return t;
}

