/**
IDXL--C++ structures used by idxl library.

Orion Sky Lawlor, olawlor@acm.org, 1/7/2003
*/
#include "idxl.h"

/*Rec: lists the chunks that share a single entity.
 */
IDXL_Rec::IDXL_Rec(int entity_) {
	entity=entity_; 
	oldlength=0;
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
#if CMK_ERROR_CHECKING
	if (chk<0 || chk>1000000)
		CkAbort("FEM IDXL_Rec::add> Tried to add absurd chunk number!\n");
#endif
	if(n >= oldlength) {
	  oldlength = 2*(n+1);
	  shares.reserve(oldlength); //Grow fast, to save extra copies
	}
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
	lock = false;
}
IDXL_List::IDXL_List(int otherchunk)
{
	chunk=otherchunk;
	lock = false;
}
IDXL_List::~IDXL_List() {}

bool IDXL_List::lockIdxl(){
  if(!lock) {
    lock = true;
    return true;
  }
  return false;
}

void IDXL_List::unlockIdxl(){
  lock = false;
}

bool IDXL_List::isLocked() {
  return lock;
}

int IDXL_List::push_back(int localIdx) {
  int ret=shared.size();
  shared.push_back(localIdx);
  return ret;
}

bool IDXL_List::set(int localIdx, int sharedIdx) {
  int size = shared.size();
  if(sharedIdx<0 || sharedIdx > size+50) return false;
  if(sharedIdx < size) {
    shared[sharedIdx] = localIdx;
  }
  else {
    for(int i=size; i<sharedIdx; i++) {
      shared.push_back(-1);
    }
    shared.push_back(localIdx);
  }
  return true;
}

bool IDXL_List::unset(int sharedIdx) {
  int size = shared.size();
  if(sharedIdx >= size || sharedIdx < 0) return false;
  shared[sharedIdx] = -1;
  return true;
}

int IDXL_List::exists(int localIdx) {
  for(int i=0; i<shared.size(); i++) {
    if(shared[i]==localIdx) return i;
  }
  return -1;
}

int IDXL_List::get(int sharedIdx) {
  int size = shared.size();
  if(sharedIdx >= size || sharedIdx < 0) return -1;
  return shared[sharedIdx];
}

void IDXL_List::pup(PUP::er &p)
{
	p|chunk;
	p|shared;
}

bool doubleeq(double a,double b){
		if(a-b < 1e-100 && a-b > -1e-100){
			return true;
		}
		return false;
}

void IDXL_List::sort2d(double *coord){
	double *dist = new double[shared.size()];
	int i;
	for(i=0;i<shared.size();i++){
		int idx = shared[i];
		dist[i] = coord[2*idx]*coord[2*idx]+coord[2*idx+1]*coord[2*idx+1];
	}
	for(i=0;i<shared.size();i++){
		for(int j=i;j<shared.size();j++){
			/**
				if the 2 points are equidistant from the origin,
				sort by x
			*/
			if(doubleeq(dist[i],dist[j])){
					int idxi = shared[i];
					int idxj = shared[j];
					if(coord[2*idxi] > coord[2*idxj]){
						int k = shared[j];
						double temp = dist[j];
						shared[j] = shared[i];
						dist[j] = dist[i];
						shared[i] = k;
						dist[i] = temp;
					}
			}
			if(dist[i] > dist[j]){
				int k = shared[j];
				double temp = dist[j];
				shared[j] = shared[i];
				dist[j] = dist[i];
				shared[i] = k;
				dist[i] = temp;
			}
		}
	}
	delete [] dist;
}

void IDXL_List::sort3d(double *coord){
	double *dist = new double[shared.size()];
	int i;
	for(i=0;i<shared.size();i++){
		int idx = shared[i];
		dist[i] = coord[3*idx]*coord[3*idx]+coord[3*idx+1]*coord[3*idx+1]+coord[3*idx+2]*coord[3*idx+2];
	}
	for(i=0;i<shared.size();i++){
		for(int j=i;j<shared.size();j++){
			/**
				if the 2 points are equidistant from the origin,
				sort by x and then by y
			*/
			if(doubleeq(dist[i],dist[j])){
					int idxi = shared[i];
					int idxj = shared[j];
					if(coord[3*idxi] > coord[3*idxj]){
						int k = shared[j];
						double temp = dist[j];
						shared[j] = shared[i];
						dist[j] = dist[i];
						shared[i] = k;
						dist[i] = temp;
					}
					else if(coord[3*idxi] == coord[3*idxj]) {
					  if(coord[3*idxi+1] > coord[3*idxj+1]){
						int k = shared[j];
						double temp = dist[j];
						shared[j] = shared[i];
						dist[j] = dist[i];
						shared[i] = k;
						dist[i] = temp;
					  }
					}
			}
			if(dist[i] > dist[j]){
				int k = shared[j];
				double temp = dist[j];
				shared[j] = shared[i];
				dist[j] = dist[i];
				shared[i] = k;
				dist[i] = temp;
			}
		}
	}
	delete [] dist;
}


/* IDXL_Side Itself: list the IDXL_Lists for all chunks
*/
IDXL_Side::IDXL_Side(void) :cached_map(NULL) 
{}
IDXL_Side::~IDXL_Side() {
	flushMap();
}

void IDXL_Side::clear(){
	flushMap();
	comm.free();
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

/// This local entity number is shared with the given local chunk
int IDXL_Side::addNode(int localNo,int sharedWithChk) {
  int shdIdx = addList(sharedWithChk).push_back(localNo);
  flushMap();
  return shdIdx;
}

int IDXL_Side::removeNode(int localNo, int sharedWithChk) {
  int local = findLocalList(sharedWithChk);
  int shdIdx = -1;
  for(int i=0; i<comm[local]->size(); i++) {
    if((*comm[local])[i] == localNo) {
      //amounts to removing this element from this list for this chunk
      (*comm[local])[i] = -1;
      shdIdx = i;
    }
  }
  flushMap();
  return shdIdx;
}





bool IDXL_Side::setNode(int localNo, int sharedWithChk, int sharedIdx) {
  bool done = addList(sharedWithChk).set(localNo, sharedIdx);
  flushMap();
  return done;
}

bool IDXL_Side::unsetNode(int sharedWithChk, int sharedIdx) {
  bool done = addList(sharedWithChk).unset(sharedIdx);
  flushMap();
  return done;
}

int IDXL_Side::existsNode(int localNo, int sharedWithChk) {
  return addList(sharedWithChk).exists(localNo);
}

int IDXL_Side::getNode(int sharedWithChk, int sharedIdx) {
  return addList(sharedWithChk).get(sharedIdx);
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

void IDXL_Side::sort2d(double *coord){
	for(int i=0;i<comm.size();i++){
		IDXL_List *l = comm[i];
		l->sort2d(coord);
	}
	flushMap();
}

void IDXL_Side::sort3d(double *coord){
	for(int i=0;i<comm.size();i++){
		IDXL_List *l = comm[i];
		l->sort3d(coord);
	}
	flushMap();
}


/*
	IDXL functions
*/

void IDXL::sort2d(double *coord){
	IDXL_Side &side = getSend();
	side.sort2d(coord);
	
	if(!isSingle()){
		IDXL_Side &recvSide = getRecv();
		recvSide.sort2d(coord);
	}
}

void IDXL::sort3d(double *coord){
	IDXL_Side &side = getSend();
	side.sort3d(coord);
	
	if(!isSingle()){
		IDXL_Side &recvSide = getRecv();
		recvSide.sort3d(coord);
	}
}
