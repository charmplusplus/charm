/*Charm++ Finite Element Framework:
C++ implementation file

This code implements fem_split and fem_assemble.
Fem_split takes a mesh and partitioning table (which maps elements
to chunks) and creates a sub-mesh for each chunk,
including the communication lists. Fem_assemble is the inverse.

The fem_split algorithm is O(n) in space and time (with n nodes,
e elements, p processors; and n>e>p^2).  The central data structure
is a bit unusual-- it's a table that maps nodes to lists of processors
(all the processors that share the node).  For any reasonable problem,
the vast majority of nodes are not shared between processors; 
this algorithm uses this to keep space and time costs low.

Memory usage for the large temporary arrays is n*sizeof(peList),
allocated contiguously.  Shared nodes will result in a few independently 
allocated peList entries, but shared nodes should be rare so this should 
not be expensive.

Originally written by Orion Sky Lawlor, olawlor@acm.org, 9/28/2000
*/
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "fem_impl.h"

class elemList;

static FEM_Symmetries_t noSymmetries=(FEM_Symmetries_t)0;

/*This object maps a single item to a list of the PEs
that have a copy of it.  For the vast majority
of items, the list will contain exactly one element.
*/
class peList : public CkNoncopyable {
public:
	int pe;//Processor number or -1 if the list is empty
	int localNo;//Local number of this item on this PE
	FEM_Symmetries_t sym; //Symmetries this item was reached via
	peList *next;
	peList() {pe=-1;next=NULL;}
	peList(int pe_,int localNo_,FEM_Symmetries_t sym_) {
		pe=pe_;
		localNo=localNo_;
		sym=sym_;
		next=NULL;
	}
	~peList() {delete next;}
	void set(int p,int l,FEM_Symmetries_t s) {
		pe=p; localNo=l; sym=0;
	}
	//Is this processor in the list?  If so, return false.
	// If not, add it and return true.
	bool addPE(int p,int l,FEM_Symmetries_t s) {
		//Add PE p to the list with local index l,
		// if it's not there already
		if (pe==p && sym==s) return false;//Already in the list
		if (pe==-1) {set(p,l,s);return true;}
		if (next==NULL) {next=new peList(p,l,s);return true;}
		else return next->addPE(p,l,s);
	}
	//Return this node's local number on PE p (or -1 if none)
	int localOnPE(int p,FEM_Symmetries_t s) const {
		const peList *l=onPE(p,s);
		if (l==NULL) return -1;
		else return l->localNo;
	}
	const peList *onPE(int p,FEM_Symmetries_t s) const {
		if (pe==p && sym==s) return this;
		else if (next==NULL) return NULL;
		else return next->onPE(p,s);
	}
	int isEmpty(void) const //Return 1 if this is an empty list 
		{return (pe==-1);}
	int isShared(void) const //Return 1 if this is a shared node
		{return next!=NULL;}
	int length(void) const {
		if (next==NULL) return isEmpty()?0:1;
		else return 1+next->length();
	}
	peList &operator[](int i) {
		if (i==0) return *this;
		else return (*next)[i-1];
	}
};

//Create a list of the chunks that all these lists belong to
static void intersectPeLists(peList **lists,int nLists,int *chunks,int *nChunks)
{
	*nChunks=0; //Initially, intersection is empty
	if (nLists==0) return; //No lists, so intersection is empty
	
	//Traverse the first list, testing each entry:
	for (const peList *cur=lists[0];cur!=NULL;cur=cur->next)
	{
		int testPE=cur->pe;
		for (int i=1;i<nLists;i++) 
			if (lists[i]->localOnPE(testPE,noSymmetries)==-1) 
			{ //List i does *not* contain testPE-- testPE is not in intersection
				testPE=-1;
				break;
			}
		if (testPE!=-1) 
		{ //TestPE lies in the intersection of all the lists-- return it
			chunks[*nChunks]=testPE;
			(*nChunks)++;
		}
	}	
}


static void checkArrayEntries(const int *arr,int nArr,int max,const char *what)
{
#ifndef CMK_OPTIMIZE
	//Check the array for out-of-bounds values
	for (int e=0;e<nArr;e++) {
		if ((arr[e]<0)||(arr[e]>=max)) {
			CkError("FEM Map Error> Entry %d of %s is %d (but should be below %d)\n",
				e,what,arr[e],max);
			CkAbort("FEM Array element out of bounds");
		} 
	}
#endif
}

//Check all user-settable fields in this object for validity
static void checkMesh(const FEM_Mesh *mesh) {
	for (int t=0;t<mesh->elem.size();t++) {
	//Check the user's connectivity array
		checkArrayEntries(mesh->elem[t].getConn().getData(),
		     mesh->elem[t].size()*mesh->elem[t].getNodesPer(),
		     mesh->node.size(), "element connectivity, from FEM_Set_Elem_Conn,");
	}
	for (int s=0;s<mesh->nSparse();s++) {
	//Check the sparse data
		const FEM_Sparse &src=mesh->getSparse(s);
		checkArrayEntries(src.getNodes(0),src.size()*src.getNodesPer(),
			mesh->node.size(), "sparse data nodes, from FEM_Set_Sparse,");
		if (src.getElem())
		  checkArrayEntries(src.getElem(),src.size()*2,
			mesh->nElems(),"sparse data elements, from FEM_Set_Sparse_Elem,");
	}
}
static void checkGhost(const FEM_Ghost &ghosts,const FEM_Mesh *mesh) {
	const int *canon=ghosts.getCanon();
	if (canon!=NULL)
	  checkArrayEntries(canon,mesh->node.size(),mesh->node.size(),
		"node canonicalization array, from FEM_Set_Symmetries");
}

/************************ Splitter *********************/

//A dynamic representation for an item (node or element)
class dynItem {
public:
	int gNo; //Global item number
	FEM_Symmetries_t sym; //Symmetries item belongs to
	dynItem(int g_,FEM_Symmetries_t s_) :gNo(g_), sym(s_) {}
#if CMK_EXPLICIT
	explicit 
#endif
	dynItem(int sillyConstructorNeededByCkVec) {gNo=-1;sym=noSymmetries;}
	dynItem() {}
};

typedef CkVec<dynItem> dynList;

//A dynamic (growing) representation of a chunk.  
// Lists the global numbers of the chunk.
class dynChunk {
public:
	class elemCount : public dynList {
	public:
		//For local elements with symmetries, this table gives
		//  the (heap-allocated) element local-node connectivity array.
		//  The table is indexed by local number.
		CkHashtableT<CkHashtableAdaptorT<int>,int *> symConn;
		
		void pup(PUP::er &p) {CkAbort("FEM> Can't call dynChunk::elemCount pup!");}
	};
	NumberedVec<elemCount> elem;
	dynList node;
	
	//Add an element to this list
	int addElement(int type,int globalNo,FEM_Symmetries_t sym) {
		elem[type].push_back(dynItem(globalNo,sym));
		return elem[type].size()-1;
	}
	int addNode(int globalNo,FEM_Symmetries_t sym) {
		node.push_back(dynItem(globalNo,sym));
		return node.size()-1;
	}
};

class splitter {
//Inputs:
	FEM_Mesh *mesh; //Unsplit source mesh
	const int *elem2chunk; //Maps global element number to destination chunk
	int nchunks; //Number of pieces to divide into
//Output:
	MeshChunk **msgs; //Output chunks [nchunks]

//Processing:
	peList *gNode; //Map global node to owning processors [mesh->node.size()]
	dynChunk *dyn; //Growing list of output elements [nchunks]

//------- Used by sparse data splitter ------------
	typedef peList* peListPtr;
	peList **srcPes;
	
	int getSparseChunks(const FEM_Sparse &src,int record,int *chunks)
	{
		const int *srcElem=src.getElem();
		if (srcElem!=NULL) 
		{ //We have an element list-- put this record on its element's chunk
			const int *elemRec=&srcElem[2*record];
			chunks[0]=elem2chunk[mesh->getGlobalElem(elemRec[0],elemRec[1])];
			return 1;
		}
		else
		{ //No element list-- split up based on nodes
			int nNodes=src.getNodesPer();
			for (int i=0;i<nNodes;i++) 
				srcPes[i]=&gNode[src.getNodes(record)[i]];
			int nDest=0;
			intersectPeLists(srcPes,nNodes,chunks,&nDest);
			return nDest;
		}
	}

//---- Used by ghost layer builder ---
	CkVec<peList *> gElem;//Map global element to owning processors (like gNode, but for elements)
	unsigned char *ghostNode; //Flag: this node borders ghost elements [mesh->node.size()]
	const int *canon; //Node canonicalization array (may be NULL)
	const FEM_Symmetries_t *sym; //Node symmetries array (may be NULL)
	int totGhostElem,totGhostNode; //For debugging printouts

	//Add an entire layer of ghost elements
	void addLayer(const ghostLayer &g,const FEM_Ghost &ghosts);
	
	//Return true if any of these nodes are ghost nodes
	bool hasGhostNodes(const int *conn,int nodesPer) 
	{
		for (int i=0;i<nodesPer;i++)
			if (ghostNode[conn[i]])
				return true;
		return false;
	}

	//Return an elemList entry if this tuple should be a ghost:
	bool addTuple(int *dest,FEM_Symmetries_t *destSym,
		const int *elem2tuple,int nodesPerTuple,const int *conn) const;

	//Check if src should be added as a ghost on dest's chunk
	void addGhostPair(const elemList *src,const elemList *dest,bool addNodes);
	void addSymmetryGhost(const elemList *a);

	//Add this global element as a ghost between src and dest, or return false
	bool addGhostElement(int t,int gNo,int srcPe,int destPe,FEM_Symmetries_t sym)
	{
		if (addGhost(gNo,srcPe,destPe,dyn[destPe].elem[t],gElem[t],
		     msgs[srcPe]->m.elem[t],msgs[destPe]->m.elem[t],sym)) 
		{
			totGhostElem++;
			return true;
		}
		return false;
	}
	//Add this global node as a ghost between src and dest, or return false
	bool addGhostNode(int gnNo,int srcPe,int destPe,FEM_Symmetries_t sym)
	{
		if (addGhost(gnNo,srcPe,destPe,dyn[destPe].node,gNode,
		     msgs[srcPe]->m.node,msgs[destPe]->m.node,sym))
		{
			totGhostNode++;
			return true;
		}
		return false;
	}

	//Add this global number as a new entry in the given dest.
	// Applies to both nodes and elements.
	// Returns false if this is not a good ghost pair.
	bool addGhost(int global,int srcPe,int destPe,
		      dynList &destChunk, peList *gDest,
		      FEM_Item &src,FEM_Item &dest,FEM_Symmetries_t sym)
	{
		if (gDest[global].onPE(destPe,sym))
			return false; //Dest already has a copy of this item
		int srcLocal=gDest[global].localOnPE(srcPe,noSymmetries);
		if (srcLocal==-1) 
			return false; //"src" is not actually local (no second-generation ghosts)
		if (src.isGhostIndex(srcLocal))
			return false; //Never add ghosts of ghosts!
		int destLocal=destChunk.size();
		destChunk.push_back(dynItem(global,sym));
		gDest[global].addPE(destPe,destLocal,sym);
		src.ghostSend.add(srcPe,srcLocal,destPe,destLocal,
				  dest.ghostRecv);
		return true;
	}

//--- Used by consistency check ---
	void bad(const char *why) {
		CkAbort(why);
	}
	void equal(int is,int should,const char *what) {
		if (is!=should) {
			CkPrintf("ERROR! Expected %s to be %d, got %d!\n",
				what,should,is);
			bad("Internal FEM data structure corruption! (unequal)\n");
		}
	}
	void range(int value,int lo,int hi,const char *what) {
		if (value<lo || value>=hi) {
			CkPrintf("ERROR! %s out of range (%d,%d)!\n",value,lo,hi);
			bad("Internal FEM data structure corruption! (out of range)\n");
		}
	}
	void nonnegative(int value,const char *what) {
		if (value<0) {
			CkPrintf("ERROR! Expected %s to be non-negative, got %d!\n",
				what,value);
		bad("Internal FEM data structure corruption! (negative)\n");
		}
	}

public:
	splitter(FEM_Mesh *mesh_,const int *elem2chunk_,int nchunks_);
	~splitter();

	//Fill the gNode[] array with the processors sharing each node
	void buildCommLists(void);
	
	//Add the layers of ghost elements
	void addGhosts(const FEM_Ghost &ghost);
	
	//Divide up the sparse data lists
	void separateSparse(void);
	
	//Divide off this portion of the serial data
	MeshChunk *createMessage(int c);
	
	void consistencyCheck(void);

};


splitter::splitter(FEM_Mesh *mesh_,const int *elem2chunk_,int nchunks_)
	:mesh(mesh_), elem2chunk(elem2chunk_),nchunks(nchunks_)
{
	msgs=new MeshChunk* [nchunks];
	dyn=new dynChunk[nchunks];
	int c;//chunk number (to receive message)
	for (c=0;c<nchunks;c++) {
		msgs[c]=new MeshChunk; //Ctor starts all node and element counts at zero
		msgs[c]->m.makeRoom(*mesh);
		dyn[c].elem.makeLonger(mesh->elem.size()-1);
	}
	
	gNode=new peList[mesh->node.size()];
	//Initialize ghost data:
	gElem.setSize(mesh->elem.size());
	for (int t=0;t<mesh->elem.size();t++) {
		gElem[t]=new peList[mesh->elem[t].size()];
	}
	ghostNode=NULL;
}

splitter::~splitter() {
	delete[] gNode;
	delete[] dyn;
	for (int t=0;t<mesh->elem.size();t++)
		delete[] gElem[t];
	for (int c=0;c<nchunks;c++)
		delete msgs[c];
	delete[] msgs;
}

void splitter::buildCommLists(void)
{
//First pass: add PEs to each node they touch
//  (also find the local elements and node counts)
	int t;//Element type
	int e;//Element number in type
	int n;//Node number around element
	for (t=0;t<mesh->elem.size();t++) {
		int typeStart=mesh->nElems(t);//Number of elements before type t
		const FEM_Elem &src=mesh->elem[t]; 
		for (e=0;e<src.size();e++) {
			int c=elem2chunk[typeStart+e];
			const int *srcConn=src.getConn().getRow(e);
			gElem[t][e].set(c,dyn[c].addElement(t,e,noSymmetries),noSymmetries);
			int lNo=dyn[c].node.size();
			for (n=0;n<src.getNodesPer();n++) {
				int gNo=srcConn[n];
				if (gNode[gNo].addPE(c,lNo,noSymmetries)) {
					//Found a new local node
					dyn[c].addNode(gNo,noSymmetries);
					lNo++;
				}
			}
		}
	}
	
//Second pass: add shared nodes to comm. lists
	for (n=0;n<mesh->node.size();n++) {
		if (gNode[n].isShared()) 
		{/*Node is referenced by more than one processor-- 
		   Add it to all the processor's comm. lists.*/
			int len=gNode[n].length();
			for (int bi=0;bi<len;bi++)
				for (int ai=0;ai<bi;ai++)
				{
					peList &a=gNode[n][ai];
		       			peList &b=gNode[n][bi];
					msgs[a.pe]->comm.add(a.pe,a.localNo,
							     b.pe,b.localNo,msgs[b.pe]->comm);
		       		}
		}
	}
}

//Decide which chunks each sparse data record belongs in.
// This is another ridiculously complex two-phase "loop-count; loop-copy" algorithm.
void splitter::separateSparse(void) 
{
	if (mesh->nSparse()==0) return; //No sparse data at all
	
	typedef FEM_Sparse* FEM_Sparse_Ptr;
	FEM_Sparse **dest=new FEM_Sparse_Ptr[nchunks]; //Destination sparse records
	int *destLen=new int[nchunks]; //Number of records per chunk
	int *destChunks=new int[nchunks]; //List of chunks a record belongs in
	int s,i;
	for (int t=0;t<mesh->nSparse();t++) 
	{ //For each kind of sparse data:
		//Prepare the messages to receive sparse records:
		const FEM_Sparse &src=mesh->getSparse(t);
		int c;//chunk number (to receive message)
		for (c=0;c<nchunks;c++) { //Build and add a structure to receive this chunk's data
			dest[c]=new FEM_Sparse(src.getNodesPer(),src.getDataPer());
			destLen[c]=0;
			msgs[c]->m.setSparse(t,dest[c]);
		}
		
		//Determine which chunks each sparse record will land in,
		// to count up the length of each chunk in destLen:
		int nSrc=src.size(),nNodes=src.getNodesPer();
		srcPes=new peListPtr[nNodes]; //Used in getSparseChunks
		for (s=0;s<nSrc;s++) {
			int nDest=getSparseChunks(src,s,destChunks);
			if (nDest==0) CkAbort("FEM Partitioning> Sparse record does not lie on any PE!");
			for (int destNo=0;destNo<nDest;destNo++) 
				destLen[destChunks[destNo]]++;
		}
		
		//Allocate room in each chunk for its sparse records
		for (c=0;c<nchunks;c++) {
			if (destLen[c]>0) {
				dest[c]->allocate(destLen[c]);
				destLen[c]=0; //Reset counter for use while inserting
			}
		}
		
		//Insert each sparse record into its chunks:
		for (s=0;s<nSrc;s++) {
			int nDest=getSparseChunks(src,s,destChunks);
			for (int destNo=0;destNo<nDest;destNo++) {
				c=destChunks[destNo];
				int d=destLen[c]++;
				//Copying sparse record s into slot d of chunk c.
				dest[c]->setData(d,src.getData(s));//Data is easy
				//Nodes have to be given local numbers:
				int *destNodes=dest[c]->getNodes(d);
				for (i=0;i<nNodes;i++)
					destNodes[i]=gNode[src.getNodes(s)[i]].localOnPE(c,noSymmetries);
			}
		}
		delete[] srcPes;
	}
	delete[] dest;
	delete[] destLen;
	delete[] destChunks;
}

//Now that we know which elements and nodes go where,
// actually copy the user and connectivity data there.
MeshChunk *splitter::createMessage(int c)
{
	int t;
	MeshChunk *msg=msgs[c];
	FEM_Mesh &dest=msg->m;

//Make room in the chunk for the incoming data
	dest.node.allocate(dyn[c].node.size(),mesh->node.getDataPer());
	for (t=0;t<dest.elem.size();t++) {
		dest.elem[t].allocate(dyn[c].elem[t].size(),
			mesh->elem[t].getDataPer(),
			mesh->elem[t].getNodesPer());
	}
	msg->allocate();

//Add each local node
	const dynList &srcIdx=dyn[c].node;
	int nNode=srcIdx.size();	
	for (int lNo=0;lNo<nNode;lNo++) {
		FEM_Symmetries_t sym=srcIdx[lNo].sym;
		int gNo=srcIdx[lNo].gNo;
		peList *head=&gNode[gNo];
		const peList *cur=head->onPE(c,sym);
		//A node is primary on its first processor
		msg->isPrimary[lNo]=(head==cur);
		msg->nodeNums[lNo]=gNo;
		//Copy the node userdata & symmetries
		dest.node.udataIs(lNo,mesh->node.udataFor(gNo));
		dest.node.setSymmetries(lNo,sym);
	}

//Add each local element
	int localStart=0; //Local element counter, across all element types
	for (t=0;t<dest.elem.size();t++) {
		const FEM_Elem &src=mesh->elem[t];
		const dynList &srcIdx=dyn[c].elem[t];
		int nEl=srcIdx.size();
		int globalStart=mesh->nElems(t);
		for (int lNo=0;lNo<nEl;lNo++) {
			FEM_Symmetries_t sym=srcIdx[lNo].sym;
			int gNo=srcIdx[lNo].gNo;
			msg->elemNums[localStart+lNo]=globalStart+gNo;
			//Copy the element userdata & symmetries
			dest.elem[t].udataIs(lNo,src.udataFor(gNo));
			dest.elem[t].setSymmetries(lNo,sym);
			if (sym==0) 
			{ //Translate the connectivity from global to local
				for (int n=0;n<src.getNodesPer();n++) {
					int gnNo=src.getConn()(gNo,n);
					int lnNo=gNode[gnNo].localOnPE(c,sym);
					dest.elem[t].setConn()(lNo,n)=lnNo;
				}
			} 
			else 
			{ //For symmetries, grab pretranslated connectivity from symConn:
				int *conn=dyn[c].elem[t].symConn.get(lNo);
				dest.elem[t].connIs(lNo,conn);
				delete[] conn;
			}
		}
		localStart+=nEl;
	}
	
	msgs[c]=NULL;
	return msg;
}

/*External entry point:
Create a sub-mesh for each chunk's elements,
including communication lists between chunks.
*/
void fem_split(FEM_Mesh *mesh,int nchunks,const int *elem2chunk,
	       const FEM_Ghost &ghosts,MeshChunkOutput *out)
{
	//Check the elem2chunk mapping:
	checkArrayEntries(elem2chunk,mesh->nElems(),nchunks,
		"elem2chunk, from FEM_Set_Partition or metis,");
	checkMesh(mesh);
	checkGhost(ghosts,mesh);
	
	mesh->setSymList(ghosts.getSymList());
	splitter s(mesh,elem2chunk,nchunks);

	s.buildCommLists();
	
	s.separateSparse();
	
	s.addGhosts(ghosts);
	
	//Split up and send out the mesh
	for (int c=0;c<nchunks;c++)
		out->accept(c,s.createMessage(c));
}


/***************************  Ghosts   *****************************
Ghost elements: read-only copies of elements on other processors.

We define the ghost region in "layers".  A layer of ghosts
is formed by adding all the elements that share 
exactly n nodes with an existing element to each chunk.

To do this, we need to map element -> adjacent elements.  One sensible
way to do this is to define a "tuple" as a group of n nodes,
then map each element to its adjacent tuples (using, e.g., a 
tiny table passed by the user); then map tuples to adjacent elements.

Mapping tuples to elements requires something like a hashtable, 
and done naively would require a large amount of space.
To make this efficient, we start off by restricting
the table to only "interesting" tuples-- tuples consisting
of nodes that could cross a processor boundary.  

The "interesting" nodes are marked in ghostNodes, and for
the first layer are equal to the shared nodes.  Each time
a ghost layer is added, the set of interesting nodes grows.
*/

//A linked list of elements surrounding a tuple.
//ElemLists are allocated one at a time, and hence a bit cleaner than pelists:
class elemList {
public:
	int pe;
	int localNo;//Local number of this element on this PE
	int type; //Kind of element
	FEM_Symmetries_t sym; //Symmetries this element was reached via
	elemList *next;

	elemList(int pe_,int localNo_,int type_,FEM_Symmetries_t sym_)
		:pe(pe_),localNo(localNo_),type(type_), sym(sym_) 
		{ next=NULL; }
	~elemList() {if (next) delete next;}
	void setNext(elemList *n) {next=n;}
	
	
};

static CkHashCode CkHashFunction_ints(const void *keyData,size_t keyLen)
{
	const int *d=(const int *)keyData;
	int l=keyLen/sizeof(int);
	CkHashCode ret=d[0];
	for (int i=1;i<l;i++)
		ret=ret^circleShift(d[i],i*23);
	return ret;
}
static int CkHashCompare_ints(const void *k1,const void *k2,size_t keyLen)
{
	const int *d1=(const int *)k1;
	const int *d2=(const int *)k2;
	int l=keyLen/sizeof(int);
	for (int i=0;i<l;i++) if (d1[i]!=d2[i]) return 0;
	return 1;
}

extern "C" int ck_fem_map_compare_int(const void *a, const void *b)
{
	return (*(const int *)a)-(*(const int *)b);
}

//Maps node tuples to element lists
class tupleTable : public CkHashtable {
  int tupleLen; //Nodes in a tuple
  CkHashtableIterator *it;
  static int roundUp(int val,int to) {
    return ((val+to-1)/to)*to;
  }
  static CkHashtableLayout makeLayout(int tupleLen) {
    int ks=tupleLen*sizeof(int);
    int oo=roundUp(ks+sizeof(char),sizeof(void *));
    int os=sizeof(elemList *);
    return CkHashtableLayout(ks,ks,oo,os,oo+os);
  }
  
	//Make a canonical version of this tuple, so different
	// orderings of the same nodes don't end up in different lists.
	//I canonicalize by sorting:
	void canonicalize(const int *tuple,int *can)
	{
		switch(tupleLen) {
		case 1: //Short lists are easy to sort:
			can[0]=tuple[0]; break;
		case 2:
			if (tuple[0]<tuple[1])
			  {can[0]=tuple[0]; can[1]=tuple[1];}
			else
			  {can[0]=tuple[1]; can[1]=tuple[0];}
			break;
		default: //Should use std::sort here:
			memcpy(can,tuple,tupleLen*sizeof(int));
			qsort(can,tupleLen,sizeof(int),ck_fem_map_compare_int);
		};
	}
public:
	enum {MAX_TUPLE=8};

	tupleTable(int tupleLen_)
		:CkHashtable(makeLayout(tupleLen_),
			     137,0.5,
			     CkHashFunction_ints,
			     CkHashCompare_ints)
	{
		tupleLen=tupleLen_;
		if (tupleLen>MAX_TUPLE) CkAbort("Cannot have that many shared nodes!\n");
		it=NULL;
	}
	~tupleTable() {
		beginLookup();
		elemList *doomed;
		while (NULL!=(doomed=(elemList *)lookupNext()))
			delete doomed;
	}
	//Lookup the elemList associated with this tuple, or return NULL
	elemList **lookupTuple(const int *tuple) {
		int can[MAX_TUPLE];
		canonicalize(tuple,can);
		return (elemList **)get(can);
	}	
	
	//Register this (new'd) element with this tuple
	void addTuple(const int *tuple,elemList *nu)
	{
		int can[MAX_TUPLE];
		canonicalize(tuple,can);
		//First try for an existing list:
		elemList **dest=(elemList **)get(can);
		if (dest!=NULL) 
		{ //A list already exists here-- link it into the new list
			nu->setNext(*dest);
		} else {//No pre-existing list-- initialize a new one.
			dest=(elemList **)put(can);
		}
		*dest=nu;
	}
	//Return all registered elemLists:
	void beginLookup(void) {
		it=iterator();
	}
	elemList *lookupNext(void) {
		void *ret=it->next();
		if (ret==NULL) {
			delete it; 
			return NULL;
		}
		return *(elemList **)ret;
	}
};

//Add all the layers of ghost elements
void splitter::addGhosts(const FEM_Ghost &ghosts)
{
	int c;
	
//Set up the ghostStart counts:
	for (c=0;c<nchunks;c++) {
		msgs[c]->m.node.startGhosts(dyn[c].node.size());
		for (int t=0;t<mesh->elem.size();t++) 
			msgs[c]->m.elem[t].startGhosts(dyn[c].elem[t].size());
	}
	
	int nLayers=ghosts.getLayers();
	if (nLayers==0) return; //No ghost layers-- nothing to do
	
//Build initial ghostNode table-- just the shared nodes
	ghostNode=new unsigned char[mesh->node.size()];
	int n,nNode=mesh->node.size();
	for (n=0;n<nNode;n++) {
		ghostNode[n]=(gNode[n].isShared());
	}

//Mark the symmetry nodes as being ghost-capable
	canon=ghosts.getCanon();
	sym=ghosts.getSymmetries();
	if (sym!=NULL)
	  for (n=0;n<nNode;n++)
	    if (sym[n]!=(FEM_Symmetries_t)0)
	      ghostNode[n]=1;
	
//Add each layer
	consistencyCheck();
	for (int i=0;i<nLayers;i++) {
		addLayer(ghosts.getLayer(i),ghosts);
		consistencyCheck();
	}

//Free up memory
	delete[] ghostNode; ghostNode=NULL;
	for (int t=0;t<mesh->elem.size();t++)
	  {delete[] gElem[t];gElem[t]=NULL;}
}

//Return an elemList entry if this tuple should be a ghost:
bool splitter::addTuple(int *dest,FEM_Symmetries_t *destSym,const int *elem2tuple,
	int nodesPerTuple,const int *conn) const
{
	FEM_Symmetries_t allSym=(FEM_Symmetries_t)(~0);
	for (int i=0;i<nodesPerTuple;i++) {
		int eidx=elem2tuple[i];
		if (eidx==-1) { //"not-there" node--
			dest[i]=-1; //Don't map via connectivity
		} else { //Ordinary node
			int n=conn[eidx];
			if (!ghostNode[n]) 
				return false; //This tuple doesn't lie on a ghost boundary
			if (sym!=NULL) {
				allSym=allSym & sym[n]; //Collect symmetries
				n=canon[n]; //Map node via canon array
			}
			dest[i]=n;
		}
	}
	//If we get here, it's a good tuple
	if (sym!=NULL) *destSym=allSym; else *destSym=0;
	return true;
}

/* Add a record for this ordered pair of adjacent elements:
Consider adding src as a ghost element on dest's processor.
*/
void splitter::addGhostPair(const elemList *src,const elemList *dest,bool addNodes)
{
	//assert((src->sym==0) == (dest->sym==0))
	int srcPe=src->pe;
	int destPe=dest->pe;
	int elemSym=dest->sym;
	bool isSymmetry=(dest->sym!=0);
	
	if ((!isSymmetry) && srcPe==destPe) 
		return; //Unless via symmetry, never add interchunk ghosts from same processor
	
	int t=src->type;
	int gNo=dyn[srcPe].elem[t][src->localNo].gNo;
	
	if (!addGhostElement(t,gNo,srcPe,destPe,elemSym)) 
		return; //Ghost is already there
	
	//If we get here, we just added src as a new ghost to dest--
	int lNo=gElem[t][gNo].localOnPE(destPe,dest->sym); //Number of the new element
	
	// add this element's nodes as ghost nodes
	const int *srcConn=mesh->elem[t].connFor(gNo);
	int srcConnN=mesh->elem[t].getNodesPer();
	int dt=dest->type;
	const int *destConn=NULL;
	int destConnN=mesh->elem[dt].getNodesPer();
	int *newConn=NULL;
	if (isSymmetry) { //Symmetry elements get a new connectivity array
		destConn=mesh->elem[dt].connFor(dyn[destPe].elem[dt][dest->localNo].gNo);
		newConn=new int[srcConnN];
	}
	
	for (int sn=0;sn<srcConnN;sn++)
	{
		FEM_Symmetries_t nodeSym=noSymmetries;
		int gnNo=srcConn[sn];
		ghostNode[gnNo]=1; //Mark this node as "interesting" for later ghost layers
		if (isSymmetry) 
		{ // Nodes of symmetry ghosts may need serious tweaking-- 
			//By default, new nodes take the symmetry of the tuple
			// FIXME: handle multiple reflections here
			nodeSym=dest->sym; 
			for (int dn=0;dn<destConnN;dn++) 
				if (canon[destConn[dn]]==canon[gnNo])
				{ // gNo is actually a symmetric copy of destConn[dn]
					gnNo=destConn[dn];
					//FIXME: nodeSym should take into account multiple reflections
					nodeSym=noSymmetries; 
					break;
				}
		}
		if (addNodes)
			addGhostNode(gnNo,srcPe,destPe,nodeSym);
		if (isSymmetry) {
			newConn[sn]=gNode[gnNo].localOnPE(destPe,nodeSym);
		}
	}
	
	if (isSymmetry) {
		dyn[destPe].elem[t].symConn.put(lNo)=newConn;
		//newConn will be delete'd by splitter::createMessage
	}
}

//Add a record for this single-element tuple list
void splitter::addSymmetryGhost(const elemList *a)
{
	if (a->sym==0) return; //Not a symmetry ghost
	CkAbort("FEM map> Mirror symmetry ghosts not yet supported");
}


//Add one layer of ghost elements
void splitter::addLayer(const ghostLayer &g,const FEM_Ghost &ghosts)
{
	tupleTable table(g.nodesPerTuple);
	
	totGhostElem=0,totGhostNode=0; //For debugging
	
	//Build table mapping node-tuples to lists of adjacent elements
	for (int c=0;c<nchunks;c++) {
	   for (int t=0;t<mesh->elem.size();t++) {
		if (!g.elem[t].add) continue;
		//For every element of every chunk:
		int nEl=dyn[c].elem[t].size();
		for (int e=0;e<nEl;e++) {
			int gNo=dyn[c].elem[t][e].gNo;
			const int *conn=mesh->elem[t].connFor(gNo);
			if (hasGhostNodes(conn,mesh->elem[t].getNodesPer()))
			{ //Loop over this element's tuples:
			  for (int u=0;u<g.elem[t].tuplesPerElem;u++) {
			  	int tuple[tupleTable::MAX_TUPLE];
				FEM_Symmetries_t allSym;
				if (addTuple(tuple,&allSym,
				    &g.elem[t].elem2tuple[u*g.nodesPerTuple],
				    g.nodesPerTuple,conn))
					table.addTuple(tuple,new elemList(c,e,t,allSym));
			  }
			}
	        }
	   }
	}
	
	//Loop over all the tuples, connecting adjacent elements
	table.beginLookup();
	elemList *l;
	while (NULL!=(l=table.lookupNext())) {
		//Consider adding ghosts for all element pairs on this tuple:
		for (const elemList *a=l;a!=NULL;a=a->next)
		for (const elemList *b=l;b!=NULL;b=b->next) 
			if (a!=b)
				addGhostPair(a,b,g.addNodes);
		if (l->next==NULL) //One-entry list: must be a symmetry
			addSymmetryGhost(l);
	}
	
	CkPrintf("FEM Ghost layer> %d new ghost elements, %d new ghost nodes\n",
	       totGhostElem,totGhostNode);
}


//Basic debugging tool: check interlinked ghost element data structures for consistency
void splitter::consistencyCheck(void)
{
	bool skipCheck=false;
	if (skipCheck) return; //Skip check in production code

	printf("FEM> Performing consistency check...\n");

	int t,c;
	//Make sure everything in dyn is also in gElem[] and gNode[]
	for (c=0;c<nchunks;c++) {
		for (t=0;t<mesh->elem.size();t++) {
			const dynList &srcIdx=dyn[c].elem[t];
			int nEl=srcIdx.size();
			for (int lNo=0;lNo<nEl;lNo++) {
				int gNo=srcIdx[lNo].gNo;
				FEM_Symmetries_t sym=srcIdx[lNo].sym;
				range(gNo,0,mesh->elem[t].size(),"global element number");
				equal(gElem[t][gNo].localOnPE(c,sym),lNo,"gElem[t] local number");
			}
		} 
		const dynList &srcIdx=dyn[c].node;
		int nNo=srcIdx.size();
		for (int lNo=0;lNo<nNo;lNo++) {
			int gNo=srcIdx[lNo].gNo;
			FEM_Symmetries_t sym=srcIdx[lNo].sym;
			range(gNo,0,mesh->node.size(),"global node number");
			equal(gNode[gNo].localOnPE(c,sym),lNo,"gNode[] local number");
		}
	}
	
	//Make sure everything in gElem and gNode is also in dyn
	for (t=0;t<mesh->elem.size();t++) {
		for (int gNo=0;gNo<mesh->elem[t].size();gNo++) {
			for (peList *l=&gElem[t][gNo];l!=NULL;l=l->next) {
				range(l->pe,0,nchunks,"gElem pe");
				equal(dyn[l->pe].elem[t][l->localNo].gNo,gNo,"chunk element");
			}
		}
	}
	for (int gNo=0;gNo<mesh->node.size();gNo++) {
		for (peList *l=&gNode[gNo];l!=NULL;l=l->next) {
			range(l->pe,0,nchunks,"gNode pe");
			equal(dyn[l->pe].node[l->localNo].gNo,gNo,"chunk node");
		}
	}

	//FIXME: Make sure all the communication lists exactly match
	
	printf("FEM> Consistency check passed\n");
}

/****************** Assembly ******************
The inverse of fem_map: reassemble split chunks into a 
single mesh.  If nodes and elements elements aren't added 
or removed, this is straightforward; but the desired semantics
under additions and deletions is unclear to me.

For now, deleted nodes and elements leave a hole in the
global-numbered node- and element- table; and added nodes
and elements are added at the end. 
*/
FEM_Mesh *fem_assemble(int nchunks,MeshChunk **msgs)
{
	FEM_Mesh *m=new FEM_Mesh;
	int t,c,e,n;

//Find the global total number of nodes and elements
	int minOld_n=1000000000,maxOld_n=0,new_n=0; //Pre-existing and newly-created nodes
	for(c=0; c<nchunks;c++) {
		const MeshChunk *msg=msgs[c];
		for (n=0;n<msg->m.node.size();n++)
			if (msg->isPrimary[n])
			{
				int g=msg->nodeNums[n];
				if (g==-1) new_n++; //Newly-created node
				else {//pre-existing node
					if (maxOld_n<=g) maxOld_n=g+1; 
					if (minOld_n>=g) minOld_n=g; 
				}
			}
	}
	if (minOld_n>maxOld_n) minOld_n=maxOld_n;
	m->node.allocate((maxOld_n-minOld_n)+new_n,msgs[0]->m.node.getDataPer());
	m->node.setUdata().set(0.0);
	m->makeRoom(msgs[0]->m);
	
	int nElemTypes=msgs[0]->m.elem.size();
	int *minOld_e=new int[nElemTypes];
	int *maxOld_e=new int[nElemTypes];	
	int new_e;
	for (t=0;t<nElemTypes;t++) {
		minOld_e[t]=1000000000;
		maxOld_e[t]=0;
		new_e=0;
		for(c=0; c<nchunks;c++) {
			const MeshChunk *msg=msgs[c];
			int startDex=msg->m.nElems(t);
			for (e=0;e<msg->m.elem[t].size();e++)
			{
				int g=msg->elemNums[startDex+e];
				if (g==-1) new_e++; //Newly-created element
				else {//pre-existing element
					if (maxOld_e[t]<=g) maxOld_e[t]=g+1; 
					if (minOld_e[t]>=g) minOld_e[t]=g; 
				}
			}
		}
		if (minOld_e[t]>maxOld_e[t]) minOld_e[t]=maxOld_e[t];
		m->elem[t].allocate((maxOld_e[t]-minOld_e[t])+new_e,
			msgs[0]->m.elem[t].getDataPer(),
			msgs[0]->m.elem[t].getNodesPer());
		m->elem[t].setUdata().set(0.0);
		m->elem[t].setConn().set(0);
	}
	
//Now copy over the local data and connectivity into the global mesh
	new_n=0;
	for(c=0; c<nchunks;c++) {
		MeshChunk *msg=msgs[c];
		for (n=0;n<msg->m.node.size();n++)
			if (msg->isPrimary[n])
			{
				int g=msg->nodeNums[n];
				if (g==-1) //Newly-created node-- assign a global number
					g=(maxOld_n-minOld_n)+new_n++;
				else //An existing node
					g-=minOld_n;
				
				//Copy over user data
				m->node.udataIs(g,msg->m.node.udataFor(n));
				msg->nodeNums[n]=g;
			}
	}
	for (t=0;t<m->elem.size();t++) {
		new_e=0;
		for(c=0; c<nchunks;c++) {
			const MeshChunk *msg=msgs[c];
			int startDex=msg->m.nElems(t);
			for (e=0;e<msg->m.elem[t].size();e++)
			{
				int g=msg->elemNums[startDex+e];
				if (g==-1)//Newly-created element
					g=(maxOld_e[t]-minOld_e[t])+new_e++;
				else //An existing element
					g-=minOld_e[t];
				
				//Copy over user data
				m->elem[t].udataIs(g,msg->m.elem[t].udataFor(e));
				
				//Copy over connectivity, translating from local to global
				const int *srcConn=msg->m.elem[t].connFor(e);
				int *dstConn=m->elem[t].connFor(g);
				for (n=0;n<msg->m.elem[t].getNodesPer();n++)
					dstConn[n]=msg->nodeNums[srcConn[n]];
			}
		}
	}

	delete[] minOld_e;
	delete[] maxOld_e;
	return m;
}
