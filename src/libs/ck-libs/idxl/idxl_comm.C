/**
IDXL--C++ structures used by idxl library.

Orion Sky Lawlor, olawlor@acm.org, 1/7/2003
*/
#include "charm++.h"
#include "idxl.h"

/*Rec: lists the chunks that share a single entity.
 */
IDXL_Rec::IDXL_Rec(int entity_) {
	entity=entity_; 
}
IDXL_Rec::~IDXL_Rec() {}
void IDXL_Rec::pup(PUP::er &p)
{
	p|entity; 
	p|shares;
}
void IDXL_Rec::add(int chk,int idx) 
{
	int n=shares.size();
#ifndef CMK_OPTIMIZE
	if (chk<0 || chk>1000000)
		CkAbort("FEM IDXL_Rec::add> Tried to add absurd chunk number!\n");
#endif
	shares.setSize(n+1); //Grow slowly, to save memory
	shares.push_back(IDXL_Share(chk,idx));
}

/*Map: map entity number to IDXL_Rec.  
 */
IDXL_Map::IDXL_Map() {}
IDXL_Map::~IDXL_Map() {
	//Delete the hashtable entries
	CkHashtableIterator *it=map.iterator();
	IDXL_Rec **rec;
	while (NULL!=(rec=(IDXL_Rec **)it->next()))
		delete *rec;
	delete it;
}

//Add a comm. entry for this entity
void IDXL_Map::add(int entity,int chk,int idx)
{
	IDXL_Rec *rec;
	if (NULL!=(rec=map.get(entity))) 
	{ //Already have a record in the table
		rec->add(chk,idx);
	}
	else
	{ //Make new record for this entity
		rec=new IDXL_Rec(entity);
		rec->add(chk,idx);
		map.put(entity)=rec;
	}
}

//Look up this entity's IDXL_Rec.  Returns NULL if entity is not shared.
const IDXL_Rec *IDXL_Map::get(int entity) const
{
	return map.get(entity);
}

IDXL_List::IDXL_List()
{
	chunk=-1;
}
IDXL_List::IDXL_List(int otherchunk)
{
	chunk=otherchunk;
}
IDXL_List::~IDXL_List() {}
void IDXL_List::pup(PUP::er &p)
{
	p(chunk);
	shared.pup(p);
}

/* IDXL_Side Itself: list the IDXL_Lists for all chunks
*/
IDXL_Side::IDXL_Side(void) :cached_map(NULL) 
{}
IDXL_Side::~IDXL_Side() {
	flushMap();
}
void IDXL_Side::pup(PUP::er &p)  //For migration
{
	comm.pup(p);
	// Cached_map need not be migrated
}

int IDXL_Side::total(void) const {
	int ret=0;
	for (int s=0;s<size();s++) ret+=comm[s]->size();
	return ret;
}

IDXL_Map &IDXL_Side::getMap(void) {
	if (cached_map) return *cached_map;
	//Build cached_map from comm data:
	cached_map=new IDXL_Map;
	IDXL_Map &map=*cached_map;
	for (int c=0;c<comm.size();c++) //Loop over chunks
	for (int idx=0;idx<comm[c]->size();idx++) //Loop over shared entities
		map.add((*comm[c])[idx],comm[c]->getDest(),idx);
	return map;
}
void IDXL_Side::flushMap(void) {
	if (cached_map) {
		delete cached_map;
		cached_map=NULL;
	}
}

const IDXL_Rec *IDXL_Side::getRec(int entity) const
{
	//Cast away constness--cached_map should be declared "mutable"
	IDXL_Side *writable=(IDXL_Side *)this; 
	return writable->getMap().get(entity);
}

void IDXL_Side::add(int myChunk,int myLocalNo,
	 int hisChunk,int hisLocalNo,IDXL_Side &his)
{
	IDXL_List *myList=getListN(hisChunk);
	if (myList==NULL) 
	{//These two chunks have never communicated before-- must add to both lists
		//Figure out our names in the other guy's table
		myList=new IDXL_List(hisChunk);
		IDXL_List *hisList=new IDXL_List(myChunk);
		comm.push_back(myList);
		his.comm.push_back(hisList);
	}
	IDXL_List *hisList=his.getListN(myChunk);
	
	//Add our local numbers to our lists
	myList->push_back(myLocalNo);
	hisList->push_back(hisLocalNo);
	flushMap(); his.flushMap();
}

class IDXL_Identity_Map : public IDXL_Print_Map {
	virtual void map(int srcIdx) const { 
		CkPrintf("%d  ",srcIdx);
	}
};

void IDXL_Side::print(const IDXL_Print_Map *idxmap) const
{
  IDXL_Identity_Map im;
  if (idxmap==NULL) idxmap=&im;
  CkPrintf("Communication list: %d chunks, %d total entries\n",size(),total());
  for (int p=0;p<size();p++) {
    const IDXL_List &l=getLocalList(p);
    CkPrintf("     With %d:",l.getDest(),l.size());
    for (int n=0;n<l.size();n++)
      idxmap->map(l[n]);
    CkPrintf("\n");
  }
}



