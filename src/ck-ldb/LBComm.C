#include <converse.h>

#if CMK_LBDB_ON

#include <stdio.h>
#include <math.h>
#include "LBComm.h"

// Hash table mostly based on open hash table from Introduction to
// Algorithms by Cormen, Leiserson, and Rivest

static inline CmiBool ObjIDEqual(const LDObjid i1, const LDObjid i2)
{
  return static_cast<CmiBool>(i1.id[0] == i2.id[0] 
	 && i1.id[1] == i2.id[1] && i1.id[2] == i2.id[2] 
	 && i1.id[3] == i2.id[3]);
};

LBCommData* LBCommTable::HashInsert(const LBCommData data)
{
  if (in_use > cur_sz/2)
    Resize();
  int i = 0;
  int j;
  do {
    j = data.hash(i,cur_sz);
    //    CmiPrintf("Hashing to %d, %d %d\n",j,i,cur_sz);
    if (state[j] == nil) {
      state[j] = InUse;
      set[j] = data;
      in_use++;
      return &set[j];
    } else i++;
  } while (i != cur_sz);

  // No room for item, but I should never get here, because I would have
  // resized the list
  CmiPrintf("HashInsert Couldn't insert!\n");
  return 0;
}

LBCommData* LBCommTable::HashSearch(const LBCommData data)
{
  int i=0;
  int j;
  do {
    j = data.hash(i,cur_sz);
    if (state[j] != nil && set[j].equal(data)) {
      return &set[j];
    }
    i++;
  } while (state[j] != nil && i != cur_sz);
  return 0;
}

LBCommData* LBCommTable::HashInsertUnique(const LBCommData data)
{
  LBCommData* item = HashSearch(data);
  if (!item) {
    item = HashInsert(data);
  }
  return item;
}

void LBCommTable::Resize()
{
  LBCommData* old_set = set;
  TableState* old_state = state;
  int old_sz = cur_sz;

  NewTable(old_sz*2);
  for(int i=0; i < old_sz; i++) {
    if (old_state[i] == InUse)
      HashInsert(old_set[i]);
  }
  delete [] old_set;
  delete [] old_state;
}	

CmiBool LBCommData::equal(const LBCommData d2) const
{
  if (from_proc) {
    if (src_proc != d2.src_proc)
      return CmiFalse;
  } else {
    if (srcObj.omhandle.id.id != d2.srcObj.omhandle.id.id 
	|| !ObjIDEqual(srcObj.id,d2.srcObj.id) )
      return CmiFalse;
  }
  if (destOM.id != d2.destOM.id 
      || !ObjIDEqual(destObj,d2.destObj))
    return CmiFalse;
  else return CmiTrue;
}

int LBCommData::compute_key()
{
  int kstring[80];
  char* kptr = static_cast<char*>(static_cast<void*>(&(kstring[0])));
  int pcount;

  if (from_proc) {
    pcount = sprintf(kptr,"%d",src_proc);
    kptr += pcount;
  } else {
    pcount = sprintf(kptr,"%d%d%d%d%d",srcObj.omhandle.id.id,
		     srcObj.id.id[0],srcObj.id.id[1],
		     srcObj.id.id[2],srcObj.id.id[3]);
    kptr += pcount;
  }
  pcount += sprintf(kptr,"%d%d%d%d%d",destOM.id,
		    destObj.id[0],destObj.id[1],
		    destObj.id[2],destObj.id[3]);
  int k;
  for(int i=0; i < (pcount+3)/4; i++)
    k ^= kstring[i];

  return k;
}

int LBCommData::hash(const int i, const int m) const
{
  const double a = 0.6803398875;
  const int k = key();
  const double ka = k * a;

  int h1 = floor(m*(ka-floor(ka)));
  int h2 = 1;  // Should be odd, to guarantee that h2 and size of table
	       // are relatively prime.

  //  CmiPrintf("k=%d h1=%d h2=%d m=%d\n",k,h1,h2,m);
  return (h1 + i * h2) % m;
}

void LBCommTable::GetCommData(LDCommData* data)
{
  LDCommData* out=data;
  LBCommData* curtable=set;
  TableState* curstate=state;
  int i;

  for(i=0; i < cur_sz; i++, curtable++, curstate++) {
    if (*curstate == InUse) {
      out->to_proc = CmiFalse;
      if (curtable->from_proc) {
	out->from_proc = CmiTrue;
	out->src_proc = curtable->src_proc;
      } else {
	out->from_proc = CmiFalse;
	out->src_proc = -1;
	out->senderOM = curtable->srcObj.omhandle.id;
	out->sender = curtable->srcObj.id;
      }
      out->receiverOM = curtable->destOM;
      out->receiver = curtable->destObj;
      out->messages = curtable->n_messages;
      out->bytes = curtable->n_bytes;
      out++;
    }
  }
}

#endif // CMK_LBDB_ON
