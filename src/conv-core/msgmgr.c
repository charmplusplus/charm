/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include <stdlib.h>
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
  CmmEntry n,e;
  if (t==NULL) return;
  e = t->first;
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

CmmTable CmmPup(pup_er p, CmmTable t)
{
  int nentries;

  if(!pup_isUnpacking(p))
  {
    CmmEntry e = t->first;
    nentries = CmmEntries(t);
    pup_int(p, &nentries);
    while(e) {
      /* messages are always allocated with CmiAlloc */
      int msize = MSGSIZE(e->msg);
      pup_int(p, &(e->ntags));
      pup_int(p, &msize);
      pup_ints(p, e->tags, e->ntags);
      pup_bytes(p, e->msg, msize);
      e = e->next;
    }
    if(pup_isDeleting(p)) {
      CmmFree(t);
      return 0;
    } else
      return t;
  }
  if(pup_isUnpacking(p))
  {
    int i;
    t = CmmNew();
    pup_int(p, &nentries);
    for(i=0;i<nentries;i++)
    {
      int ntags, msize, *tags;
      void *msg;
      pup_int(p, &ntags);
      pup_int(p, &msize);
      tags = (int*) malloc(ntags*sizeof(int));
      pup_ints(p, tags, ntags);
      msg = CmiAlloc(msize);
      pup_bytes(p, msg, msize);
      CmmPut(t, ntags, tags, msg);
      free(tags);
    }
    return t;
  }
  return NULL;/*<- never executed*/
}

