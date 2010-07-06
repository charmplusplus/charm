
#include <stdlib.h>
#include <converse.h>

#define CmiAlloc  malloc
#define CmiFree   free

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
  if (t==NULL) return;
#if (!defined(_FAULT_MLOG_) && !defined(_FAULT_CAUSAL_))    
  if (t->first!=NULL) CmiAbort("Cannot free a non-empty message table!");
#endif
  CmiFree(t);
}

/* free all table entries but not the space pointed by "msg" */
void CmmFreeAll(CmmTable t){
    CmmEntry cur;
    if(t==NULL) return;
    cur = t->first;
    while(cur){
	CmmEntry toDel = cur;
	cur = cur->next;
	CmiFree(toDel);
    }
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
/* added by Chao Mei in case that t is already freed
  which happens in ~ampi() when doing out-of-core emulation for AMPI programs */
  if(t==NULL) return NULL;

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

/* match the first ntags tags and return the last tag */
int CmmGetLastTag(t,ntags,tags)
CmmTable t;
int ntags;
int* tags;
{
  CmmEntry *enth; CmmEntry ent;
  enth = &(t->first);
  while (1) {
    ent = (*enth);
    if (ent==0) return -1;
    if (CmmTagsMatch(ntags, tags, ntags, ent->tags)) {
      return (ent->tags[ent->ntags-1]);
    }
    enth = &(ent->next);
  }
  return -1;
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


CmmTable CmmPup(pup_er p, CmmTable t, CmmPupMessageFn msgpup)
{
  int nentries;

  if(!pup_isUnpacking(p))
  {
    CmmEntry e = t->first, doomed;
    nentries = CmmEntries(t);
    pup_int(p, &nentries);
    while(e) {
      pup_int(p, &(e->ntags));
      pup_ints(p, e->tags, e->ntags);
      msgpup(p,&e->msg);
      doomed=e;
      e = e->next;
      if (pup_isDeleting(p)) 
        CmiFree(doomed);
    }
    if(pup_isDeleting(p)) 
    { /* We've now deleted all the links */
      t->first=NULL;
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
      int ntags, *tags;
      void *msg;
      pup_int(p, &ntags);
      tags = (int*) malloc(ntags*sizeof(int));
      pup_ints(p, tags, ntags);
      msgpup(p,&msg);
      CmmPut(t, ntags, tags, msg);
      free(tags);
    }
    return t;
  }
  return NULL;/*<- never executed*/
}

