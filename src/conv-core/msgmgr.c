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


