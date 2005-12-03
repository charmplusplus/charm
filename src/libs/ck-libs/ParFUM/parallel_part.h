/*
	File containing the data structures, function declaration 
	and msa array declarations used during parallel partitioning
	of the mesh.
	Author Sayantan Chakravorty
	05/30/2004
*/

#ifndef __FEM_PARALLEL_PART_H__
#define __FEM_PARALLEL_PART_H__

//#define PARALLEL_DEBUG

#ifdef PARALLEL_DEBUG
#define DEBUG(x) x
#else
#define DEBUG(x)
#endif

#define MESH_CHUNK_TAG 3000

template <class T, bool PUP_EVERY_ELEMENT=true >
class DefaultListEntry {
public:
    inline void accumulate(T& a, const T& b) { a += b; }
    // identity for initializing at start of accumulate
    inline T getIdentity() { return T(); }
    inline bool pupEveryElement(){ return PUP_EVERY_ELEMENT; }
};

template <class T>
class ElemList{
public:
	CkVec<T> *vec;
	ElemList(){
		vec = new CkVec<T>();
	}
	~ElemList(){
		delete vec;
	}
	ElemList(const ElemList &rhs){
		*this=rhs;
	}
	ElemList& operator=(const ElemList& rhs){
		 vec = new CkVec<T>();
		*vec = *(rhs.vec);
	}
	ElemList& operator+=(const ElemList& rhs){
		/*
			add the new unique elements to the List
		*/
		int len = vec->size();
		for(int i=0;i<rhs.vec->length();i++){
			int flag=0;
			for(int j=0;j<len;j++){
				if((*vec)[j] == (*(rhs.vec))[i]){					
					flag = 1;
					break;
				}
			}
			if(!flag){
				vec->push_back((*(rhs.vec))[i]);
			}	
		}
	}
	ElemList(const T &val){
		vec =new CkVec<T>();
		vec->push_back(val);
	};
	virtual void pup(PUP::er &p){
		if(p.isUnpacking()){
			vec = new CkVec<T>();
		}
		pupCkVec(p,*vec);
	}
};

class NodeElem {
public:
	//global number of this node
	int global;
	/*no of chunks that share this node
		owned by 1 element - numShared 0
		owned by 2 elements - numShared 2
	*/
	int numShared;
	/*somewhat horrible semantics
		-if numShared == 0 shared is NULL
		- else stores all the chunks that share this node
	*/
	int *shared;
	NodeElem(){
		global = -1;
		numShared = 0;
		shared = NULL;
	}
	NodeElem(int g_,int num_){
		global = g_; numShared= num_;
		if(numShared != 0){
			shared = new int[numShared];
		}else{
			shared = NULL;
		}
	}
	NodeElem(int g_){
		global = g_;
		numShared = 0;
		shared = NULL;
	}
	NodeElem(const NodeElem &rhs){
		shared=NULL;
		*this = rhs;
	}
	NodeElem& operator=(const NodeElem &rhs){
		global = rhs.global;
		numShared = rhs.numShared;
		if(shared != NULL){
			delete [] shared;
		}
		shared = new int[numShared];
		memcpy(shared,rhs.shared,numShared*sizeof(int));
	}

	bool operator == (const NodeElem &rhs){
		if(global == rhs.global){
			return true;
		}else{
			return false;
		}
	}
	virtual void pup(PUP::er &p){
		p | global;
		p | numShared;
		if(p.isUnpacking()){
			if(numShared != 0){
				shared = new int[numShared];
			}else{
				shared = NULL;
			}
		}
		p(shared,numShared);
	}
	~NodeElem(){
		if(shared != NULL){
			delete [] shared;
		}
	}
};

/*
	This class is an MSA Entity. It is used for 2 purposes
	1 It is used for storing the mesh while creating the mesh
	for each chunk
	2	It is used for storing the ghost elements and nodes for
	a chunk.
*/
class MeshElem{
	public: 
	FEM_Mesh *m;
	CkVec<int> gedgechunk; // Chunk number of 
	MeshElem(){
		m = new FEM_Mesh;
	}
	MeshElem(int dummy){
		m = new FEM_Mesh;
	}
	~MeshElem(){
		delete m;
	}
	MeshElem(const MeshElem &rhs){
		m = NULL;
		*this = rhs;
	}
	MeshElem& operator=(const MeshElem &rhs){
		if(m != NULL){
			delete m;
		}	
		m = new FEM_Mesh;
		m->copyShape(*(rhs.m));
		(*this)+= rhs;
	}
	MeshElem& operator+=(const MeshElem &rhs){
		int oldel = m->nElems();
		m->copyShape(*(rhs.m));
		for(int i=0;i<rhs.m->node.size();i++){
			m->node.push_back((rhs.m)->node,i);
		}	
		if((rhs.m)->elem.size()>0){
			for(int t=0;t<(rhs.m)->elem.size();t++){
				if((rhs.m)->elem.has(t)){
					for(int e=0;e<(rhs.m)->elem.get(t).size();e++){
						m->elem[t].push_back((rhs.m)->elem.get(t),e);
					}	
				}	
			}
		}	
	}
	virtual void pup(PUP::er &p){
		if(p.isUnpacking()){
			m = new FEM_Mesh;
		}
		m->pup(p);
	}
};


class Hashnode{
public:
	class tupledata{
		public:
		enum {MAX_TUPLE = 8};
			int nodes[MAX_TUPLE];
			tupledata(int _nodes[MAX_TUPLE]){
				memcpy(nodes,_nodes,sizeof(int)*MAX_TUPLE);
			}
			tupledata(tupledata &rhs){
				memcpy(nodes,rhs.nodes,sizeof(int)*MAX_TUPLE);
			}
			tupledata(){};
			//dont store the returned string
			char *toString(int numnodes,char *str){
				str[0]='\0';
				for(int i=0;i<numnodes;i++){
					sprintf(&str[strlen(str)],"%d ",nodes[i]);
				}
				return str;
			}
			int &operator[](int i){
				return nodes[i];
			}
			const int &operator[](int i) const {
				return nodes[i];
			}
			virtual void pup(PUP::er &p){
				p(nodes,MAX_TUPLE);
			}
	};
	int numnodes; //number of nodes in this tuple
	//TODO: replace *nodes with the above tupledata class
	tupledata nodes;	//the nodes in the tuple
	int chunk;		//the chunk number to which this element belongs 
	int elementNo;		//local number of that element
	Hashnode(){
		numnodes=0;
	};
	Hashnode(int _num,int _chunk,int _elNo,int _nodes[tupledata::MAX_TUPLE]): nodes(_nodes){
		numnodes = _num;
		chunk = _chunk;
		elementNo = _elNo;
	}
	Hashnode(const Hashnode &rhs){
		*this = rhs;
	}
	Hashnode &operator=(const Hashnode &rhs){
		numnodes = rhs.numnodes;
		for(int i=0;i<numnodes;i++){
			nodes[i] = rhs.nodes[i];
		}
		chunk = rhs.chunk;
		elementNo = rhs.elementNo;
	}
	bool operator==(const Hashnode &rhs){
		if(numnodes != rhs.numnodes){
			return false;
		}
		for(int i=0;i<numnodes;i++){
			if(nodes[i] != rhs.nodes[i]){
				return false;
			}
		}
		if(chunk != rhs.chunk){
			return false;
		}
		if(elementNo != rhs.elementNo){
			return false;
		}
		return true;
	}
	bool equals(tupledata &tuple){
		for(int i=0;i<numnodes;i++){
			if(tuple.nodes[i] != nodes[i]){
				return false;
			}
		}
		return true;
	}
	virtual void pup(PUP::er &p){
		p | numnodes;
		p | nodes;
		p | chunk;
		p | elementNo;
	}
};

template <class T>
ostream& operator << (ostream& os, const ElemList<T> & s){
};

template <class T>
CkOutStream& operator << (CkOutStream& os, const ElemList<T>& s) {
};

typedef MSA2D<int, DefaultEntry<int>, MSA_DEFAULT_ENTRIES_PER_PAGE, MSA_ROW_MAJOR> MSA2DRM;

typedef MSA1D<int, DefaultEntry<int>, MSA_DEFAULT_ENTRIES_PER_PAGE> MSA1DINT;

typedef ElemList<int> IntList;
typedef MSA1D<IntList, DefaultListEntry<IntList,true>,MSA_DEFAULT_ENTRIES_PER_PAGE> MSA1DINTLIST;

typedef ElemList<NodeElem> NodeList;
typedef MSA1D<NodeList, DefaultListEntry<NodeList,true>,MSA_DEFAULT_ENTRIES_PER_PAGE> MSA1DNODELIST;

typedef MSA1D<MeshElem,DefaultEntry<MeshElem,true>,1> MSA1DFEMMESH;

typedef ElemList<Hashnode> Hashtuple;
typedef MSA1D<Hashtuple,DefaultListEntry<Hashtuple,true>,MSA_DEFAULT_ENTRIES_PER_PAGE> MSA1DHASH;



struct conndata{
	int nelem;
	int nnode;
	MSA1DINT arr1;
	MSA1DINT arr2;

	void pup(PUP::er &p){
		p|nelem;
		p|nnode;
		arr1.pup(p);
		arr2.pup(p);
	}
};

/*
	Structure to store connectivity data after the 
	global element partition has been returned by parmetis
*/
struct partconndata{
	int nelem;
	int startindex;
	int *eptr,*eind;
	int *part;
	~partconndata(){
		delete [] eptr;
		delete [] eind;
		delete [] part;
	};
};

/*
	structure for storing the ghost layers
*/
struct ghostdata{
	int numLayers;
	FEM_Ghost_Layer **layers;
	void pup(PUP::er &p){
		p | numLayers;
		if(p.isUnpacking()){
			layers = new FEM_Ghost_Layer *[numLayers];
			for(int i=0;i<numLayers;i++){
				layers[i] = new FEM_Ghost_Layer;
			}

		}
		for(int i=0;i<numLayers;i++){
			layers[i]->pup(p);
		}
	}
	~ghostdata(){
			printf("destructor on ghostdata called \n");
			for(int i=0;i<numLayers;i++){
					delete layers[i];
			}
			delete [] layers;
	};
};


class MsaHashtable{
public:
	int numSlots;
	MSA1DHASH table;
	MsaHashtable(int _numSlots,int numWorkers):numSlots(_numSlots),table(_numSlots,numWorkers){
	}
	MsaHashtable(){};

	virtual void pup(PUP::er &p){
		p | numSlots;
		p | table;
	}
	int addTuple(int *tuple,int nodesPerTuple,int chunk,int elementNo){
		//sort the tuples to get a canonical form
		// bubble sort should do just as well since the number
		// of nodes is less than 10.
		for(int i=0;i<nodesPerTuple-1;i++){
			for(int j=i+1;j<nodesPerTuple;j++){
				if(tuple[j] < tuple[i]){
					int t = tuple[j];
					tuple[j] = tuple[i];
					tuple[i] = t;
				}
			}
		}
		//find out the index
		long long sum = 0;
		for(int i=0;i<nodesPerTuple;i++){
			sum = sum *numSlots + tuple[i];
		}
		int index = (int )(sum %(long )numSlots);
		Hashnode entry(nodesPerTuple,chunk,elementNo,tuple);
	
		Hashtuple &list=table.accumulate(index);
		list.vec->push_back(entry);
		char str[100];
		DEBUG(printf("[%d] adding tuple %s element %d to index %d \n",chunk,entry.nodes.toString(nodesPerTuple,str),elementNo,index));
		return index;
	}

	void print(){
		char str[100];
		for(int i=0;i<numSlots;i++){
			const Hashtuple &t = table.get(i);
			for(int j=0;j<t.vec->size();j++){
				Hashnode &tuple = (*t.vec)[j];
				printf("ghost element chunk %d element %d index %d tuple < %s>\n",tuple.chunk,tuple.elementNo,i,tuple.nodes.toString(tuple.numnodes,str));
			}
		}
	}
	void sync(){
		table.sync();
	}
	const Hashtuple &get(int i){
		return table.get(i);
	}
	
};



int FEM_master_parallel_part(int ,int ,FEM_Comm_t);
int FEM_slave_parallel_part(int ,int ,FEM_Comm_t);
struct partconndata* FEM_call_parmetis(struct conndata &data,FEM_Comm_t comm_context);
void FEM_write_nodepart(MSA1DINTLIST	&nodepart,struct partconndata *data);
void FEM_write_part2node(MSA1DINTLIST	&nodepart,MSA1DNODELIST &part2node,struct partconndata *data,MPI_Comm comm_context);
void FEM_write_part2elem(MSA1DINTLIST &part2elem,struct partconndata *data,MPI_Comm comm_context);
FEM_Mesh * FEM_break_mesh(FEM_Mesh *m,int numElements,int numChunks);
void sendBrokenMeshes(FEM_Mesh *mesh_array,FEM_Comm_t comm_context);
void	FEM_write_part2mesh(MSA1DFEMMESH &part2mesh,struct partconndata *partdata,struct conndata *data,MSA1DINTLIST &nodepart,int numChunks,int myChunk,FEM_Mesh *mypiece);
void addIDXLists(FEM_Mesh *m,NodeList &lnodes,int myChunk);
struct ghostdata *gatherGhosts();
void makeGhosts(FEM_Mesh *m,MPI_Comm comm,int masterRank,int numLayers,FEM_Ghost_Layer **layers);
void makeGhost(FEM_Mesh *m,MPI_Comm comm,int masterRank,int totalShared,FEM_Ghost_Layer *layer,	CkHashtableT<CkHashtableAdaptorT<int>,char> &sharedNode,CkHashtableT<CkHashtableAdaptorT<int>,int> &global2local);
bool sharedWith(int lnode,int chunk,FEM_Mesh *m);

#endif
