/**
 * \addtogroup CkLdb
*/
/*@{*/

#include <converse.h>

#if CMK_LBDB_ON

#include <math.h>
#include "LBComm.h"
#include <set>

#include "TopoManager.h"

// Hash table mostly based on open hash table from Introduction to
// Algorithms by Cormen, Leiserson, and Rivest

// Moved comparison function to LDObjIDEqual

// static inline bool ObjIDEqual(const LDObjid i1, const LDObjid i2)
// {
//   return (bool)(i1.id[0] == i2.id[0] 
// 	 && i1.id[1] == i2.id[1] && i1.id[2] == i2.id[2] 
// 	 && i1.id[3] == i2.id[3]);
// };

LBCommData* LBCommTable::HashInsert(const LBCommData &data)
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

LBCommData* LBCommTable::HashSearch(const LBCommData &data)
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

LBCommData* LBCommTable::HashInsertUnique(const LBCommData &data)
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

bool LBCommData::equal(const LBCommData &d2) const
{
  if (from_proc()) {
    if (src_proc != d2.src_proc)
      return false;
  } else {
    if (srcObj.omID() != d2.srcObj.omID()
	|| srcObj.objID() != d2.srcObj.objID() )
      return false;
  }
  return (bool)(destObj == d2.destObj);
}

int LBCommData::compute_key()
{
  char kstring[320];
  char* kptr = &kstring[0];
  int* kintarr = (int*)((void*)(&kstring[0]));
  int pcount;

  if (from_proc()) {
    pcount = snprintf(kptr,sizeof(kstring),"%d",src_proc);
    kptr += pcount;
  } else {
    pcount = snprintf(kptr,sizeof(kstring),"%d%" PRIu64 "",srcObj.omID().id.idx,
		     srcObj.id);
    kptr += pcount;
  }

  //CmiAssert(destObj.get_type() == LD_OBJ_MSG);
  switch (destObj.get_type()) {
  case LD_PROC_MSG:
       pcount += snprintf(kptr, sizeof(kstring) - pcount, "%d", destObj.proc());
       break;
  case LD_OBJ_MSG: {
       LDObjKey &destKey = destObj.get_destObj();
       pcount += snprintf(kptr, sizeof(kstring) - pcount, "%d%" PRIu64 "XXXXXXXX",destKey.omID().id.idx,
		    destKey.objID());
       pcount -= 8;  /* The 'X's insure that the next few bytes are fixed */
       break;
       }
  case LD_OBJLIST_MSG: {
       int len;
       const LDObjKey *destKeys = destObj.get_destObjs(len);
       CmiAssert(len>0);
       pcount += snprintf(kptr, sizeof(kstring) - pcount, "%d%" PRIu64 "XXXXXXXX",destKeys[0].omID().id.idx,
		    destKeys[0].objID());
       pcount -= 8;  /* The 'X's insure that the next few bytes are fixed */
       break;
       }
  }

  int k=-1;
  for(int i=0; i < (pcount+sizeof(int)-1)/sizeof(int); i++)
    k ^= kintarr[i];

  // CmiPrintf("New key %d, %s\n",k,kstring);

  return k;
}

int LBCommData::hash(const int i, const int m) const
{
  const double a = 0.6803398875;
  const int k = key();
  const double ka = k * a;

  int h1 = (int) floor(m*(ka-floor(ka)));
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
      out->clearHash();
      if (curtable->from_proc()) {
	out->src_proc = curtable->src_proc;
      } else {
	out->src_proc = -1;
        out->sender.omID() = curtable->srcObj.omID();
        out->sender.objID() = curtable->srcObj.objID();
      }
      out->receiver = curtable->destObj;
      out->messages = curtable->n_messages;
      out->bytes = curtable->n_bytes;
      out++;
    }
  }
}

struct LDCommDescComp {
  bool operator() (const LDCommDesc& lhs, const LDCommDesc &rhs) const {
    return (lhs.get_destObj() < rhs.get_destObj());
  }
};

void LBCommTable::GetCommInfo(int& bytes, int& msgs, int& outsidepemsgs, int&
    outsidepebytes, int& num_nghbor, int& hops, int& hopbytes) {

  LBCommData* curtable=set;
  TableState* curstate=state;
  int i;
  bytes = 0;
  msgs = 0;
  outsidepemsgs = 0;
  outsidepebytes = 0;
  hops = 0;
  hopbytes = 0;
  std::set<LDCommDesc, LDCommDescComp> num_neighbors;

  int h;

  for(i=0; i < cur_sz; i++, curtable++, curstate++) {
    if (*curstate == InUse) {
      msgs += curtable->n_messages;
      bytes += curtable->n_bytes;
      if (curtable->destObj.get_type() == LD_OBJ_MSG) {
        num_neighbors.insert(curtable->destObj);
      }

      if (curtable->destObj.lastKnown() != CkMyPe()) {
        outsidepebytes += curtable->n_bytes;
        outsidepemsgs += curtable->n_messages;
        if(curtable->destObj.lastKnown()>=0 && curtable->destObj.lastKnown()<CkNumPes()){
          TopoManager_getHopsBetweenPeRanks(CkMyPe(), curtable->destObj.lastKnown(), &h);
          hops += curtable->n_messages * h;
          hopbytes += curtable->n_bytes * h;
        }
      }
    }
  }
  num_nghbor = num_neighbors.size();
}

#endif // CMK_LBDB_ON

/*@}*/
