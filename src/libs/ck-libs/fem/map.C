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

Memory usage for the large temporary arrays is n*sizeof(peList)
+p^2*sizeof(peList), all allocated contiguously.  Shared nodes
will result in a few independently allocated peList entries,
but shared nodes should be rare so this should not be expensive.

Originally written by Orion Sky Lawlor, olawlor@acm.org, 9/28/2000
*/
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "fem_impl.h"

/*This object maps a single node to a list of the PEs
that have a copy of it.  For the vast majority
of nodes, the list will contain exactly one element.
*/
class peList : public CkNoncopyable {
public:
	int pe;//Processor number or -1 if the list is empty
	int localNo;//Local number of this node on this PE
	peList *next;
	peList() {pe=-1;next=NULL;}
	peList(int Npe,int NlocalNo) {
		pe=Npe;
		localNo=NlocalNo;
		next=NULL;
	}
	~peList() {delete next;}
	void set(int p,int l) {
		pe=p; localNo=l; 
	}
	//Is this processor in the list?  If so, return false.
	// If not, add it and return true.
	bool addPE(int p,int l) {
		//Add PE p to the list with local index l,
		// if it's not there already
		if (pe==p) return false;//Already in the list
		if (pe==-1) {set(p,l);return true;}
		if (next==NULL) {next=new peList(p,l);return true;}
		else return next->addPE(p,l);
	}
	//Return this node's local number on PE p (or -1 if none)
	int localOnPE(int p) const {
		if (pe==p) return localNo;
		else if (next==NULL) return -1;
		else return next->localOnPE(p);
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
			if (lists[i]->localOnPE(testPE)==-1) 
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


//A dynamic (growing) representation of a chunk.  
// Lists the global numbers of the chunk.
class dynChunk {
public:
	class elemCount : public CkVec<int> {
	public:
		void pup(PUP::er &p) {CkAbort("FEM> Can't call dynChunk::elemCount pup!");}
	};
	NumberedVec<elemCount> elem;
	CkVec<int> node;
	
	//Add an element to this list
	int addElement(int type,int globalNo) {
		elem[type].push_back(globalNo);
		return elem[type].size()-1;
	}
	int addNode(int globalNo) {
		node.push_back(globalNo);
		return node.size()-1;
	}
};

class splitter {
//Inputs:
	const FEM_Mesh *mesh; //Unsplit source mesh
	int *elem2chunk; //Maps global element number to destination chunk
	int nchunks; //Number of pieces to divide into
//Output:
	MeshChunk **msgs; //Output chunks [nchunks]

//Processing:
	peList *gNode; //Map global node to owning processors [mesh->node.n]
	dynChunk *chunks; //Growing list of output elements [nchunks]

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
	CkVec<peList *> gElem;//Map global element to owning processors
	unsigned char *ghostNode; //Flag: this node borders ghost elements [mesh->node.n]

//Return true if any of these nodes are ghost nodes
bool hasGhostNodes(const int *conn,int nodesPer) 
{
	for (int i=0;i<nodesPer;i++)
		if (ghostNode[conn[i]])
			return true;
	return false;
}

//Add this global number as a new entry in the given dest.
// Applies to both nodes and elements.
// Returns false if this is not a good ghost pair.
bool addGhost(int global,int srcPe,int destPe,
	      CkVec<int> &destChunk, peList *gDest,
	      FEM_Item &src,FEM_Item &dest)
{
	if (srcPe==destPe) 
		return false; //Never add ghosts from same pe!
	if (-1!=gDest[global].localOnPE(destPe))
		return false; //Dest already has a copy of this item
	int srcLocal=gDest[global].localOnPE(srcPe);
	if (src.isGhostIndex(srcLocal))
		return false; //Never add ghosts of ghosts!
	int destLocal=destChunk.size();
	destChunk.push_back(global);
	gDest[global].addPE(destPe,destLocal);
	src.ghostSend.add(srcPe,srcLocal,destPe,destLocal,
			  dest.ghostRecv);
	return true;
}

//Add an entire layer of ghost elements
void add(const ghostLayer &g);



public:
splitter(const FEM_Mesh *mesh_,int *elem2chunk_,int nchunks_)
	:mesh(mesh_), elem2chunk(elem2chunk_),nchunks(nchunks_)
{
	msgs=new MeshChunk* [nchunks];
	chunks=new dynChunk[nchunks];
	int c;//chunk number (to receive message)
	for (c=0;c<nchunks;c++) {
		msgs[c]=new MeshChunk; //Ctor starts all node and element counts at zero
		msgs[c]->m.copyType(*mesh);
		chunks[c].elem.makeLonger(mesh->elem.size()-1);
	}
	
	gNode=new peList[mesh->node.n];
	//Initialize ghost data:
	gElem.setSize(mesh->elem.size());
	for (int t=0;t<mesh->elem.size();t++) {
		gElem[t]=new peList[mesh->elem[t].n];
	}
	ghostNode=NULL;
}

~splitter() {
	delete[] gNode;
	delete[] chunks;
	for (int t=0;t<mesh->elem.size();t++)
		delete[] gElem[t];
	for (int c=0;c<nchunks;c++)
		delete msgs[c];
	delete[] msgs;
}

	//Fill the gNode[] array with the processors sharing each node
	void buildCommLists(void);
	
	//Add the layers of ghost elements
	void addGhosts(int nLayers,const ghostLayer *g);
	
	//Divide up the sparse data lists
	void separateSparse(void);
	
	//Divide off this portion of the serial data
	MeshChunk *createMessage(int c);
	
	void consistencyCheck(void);
	
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

};

//Basic debugging tool: check interlinked ghost element data structures for consistency
void splitter::consistencyCheck(void)
{
	if (1) return; //Skip check in production code

	printf("FEM> Performing consistency check...\n");

	int t,c;
	//Make sure everything in chunks is also in gElem[] and gNode[]
	for (c=0;c<nchunks;c++) {
		for (t=0;t<mesh->elem.size();t++) {
			const CkVec<int> &srcIdx=chunks[c].elem[t];
			int nEl=srcIdx.size();
			for (int lNo=0;lNo<nEl;lNo++) {
				int gNo=srcIdx[lNo];
				range(gNo,0,mesh->elem[t].n,"global element number");
				equal(gElem[t][gNo].localOnPE(c),lNo,"gElem[t] local number");
			}
		} 
		const CkVec<int> &srcIdx=chunks[c].node;
		int nNo=srcIdx.size();
		for (int lNo=0;lNo<nNo;lNo++) {
			int gNo=srcIdx[lNo];
			range(gNo,0,mesh->node.n,"global node number");
			equal(gNode[gNo].localOnPE(c),lNo,"gNode[] local number");
		}
	}
	
	//Make sure everything in gElem and gNode is also in chunks
	for (t=0;t<mesh->elem.size();t++) {
		for (int gNo=0;gNo<mesh->elem[t].n;gNo++) {
			for (peList *l=&gElem[t][gNo];l!=NULL;l=l->next) {
				range(l->pe,0,nchunks,"gElem pe");
				equal(chunks[l->pe].elem[t][l->localNo],gNo,"chunk element");
			}
		}
	}
	for (int gNo=0;gNo<mesh->node.n;gNo++) {
		for (peList *l=&gNode[gNo];l!=NULL;l=l->next) {
			range(l->pe,0,nchunks,"gNode pe");
			equal(chunks[l->pe].node[l->localNo],gNo,"chunk node");
		}
	}

	//FIXME: Make sure all the communication lists exactly match
	
	printf("FEM> Consistency check passed\n");
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
		for (e=0;e<src.n;e++) {
			int c=elem2chunk[typeStart+e];
			const int *srcConn=&src.conn[e*src.nodesPer];
			gElem[t][e].set(c,chunks[c].addElement(t,e));
			int lNo=chunks[c].node.size();
			for (n=0;n<src.nodesPer;n++) {
				int gNo=srcConn[n];
				if (gNode[gNo].addPE(c,lNo)) {
					//Found a new local node
					chunks[c].addNode(gNo);
					lNo++;
				}
			}
		}
	}
	
//Second pass: add shared nodes to comm. lists
	for (n=0;n<mesh->node.n;n++) {
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
					destNodes[i]=gNode[src.getNodes(s)[i]].localOnPE(c);
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
	dest.node.n=chunks[c].node.size();
	dest.node.allocUdata();
	for (t=0;t<dest.elem.size();t++) {
		dest.elem[t].n=chunks[c].elem[t].size();
		dest.elem[t].allocUdata();
		dest.elem[t].allocConn();
	}	
	msg->allocate();

//Add each local node
	const CkVec<int> &srcIdx=chunks[c].node;
	int nNode=srcIdx.size();	
	for (int lNo=0;lNo<nNode;lNo++) {
		int gNo=srcIdx[lNo];
		peList *l=&gNode[gNo];
		//A node is primary on its first processor
		msg->isPrimary[lNo]=(l->pe==c);
		msg->nodeNums[lNo]=gNo;
		//Copy the node userdata
		dest.node.udataIs(lNo,mesh->node.udataFor(gNo));
	}

//Add each local element
	int localStart=0; //Local element counter, across all element types
	for (t=0;t<dest.elem.size();t++) {
		const FEM_Elem &src=mesh->elem[t];
		const CkVec<int> &srcIdx=chunks[c].elem[t];
		int nEl=srcIdx.size();
		int globalStart=mesh->nElems(t);
		for (int lNo=0;lNo<nEl;lNo++) {
			int gNo=srcIdx[lNo];
			msg->elemNums[localStart+lNo]=globalStart+gNo;
			//Copy the element userdata
			dest.elem[t].udataIs(lNo,src.udataFor(gNo));
			//Translate the connectivity from global to local
			for (int n=0;n<src.nodesPer;n++) {
				int gnNo=src.conn[gNo*src.nodesPer+n];
				int lnNo=gNode[gnNo].localOnPE(c);
				dest.elem[t].conn[lNo*src.nodesPer+n]=lnNo;
			}
		}
		localStart+=nEl;
	}
	
	msgs[c]=NULL;
	return msg;
}

static void checkArrayEntries(const int *arr,int nArr,int max,const char *what)
{
#ifndef CMK_OPTIMIZE
	//Check the array for out-of-bounds values
	for (int e=0;e<nArr;e++) {
		if ((arr[e]<0)||(arr[e]>=max)) {
			CkError("FEM Map Error> Entry %d of %s is %d--out of bounds!\n",
				e,what,arr[e]);
			CkAbort("FEM Array element out of bounds");
		} 
	}
#endif
}


/*External entry point:
Create a sub-mesh for each chunk's elements,
including communication lists between chunks.
*/
void fem_split(const FEM_Mesh *mesh,int nchunks,int *elem2chunk,
	       int nGhostLayers,const ghostLayer *g,MeshChunkOutput *out)
{
	//Check the elem2chunk mapping:
	checkArrayEntries(elem2chunk,mesh->nElems(),nchunks,
		"elem2chunk, from FEM_Set_Partition or metis,");
	for (int t=0;t<mesh->elem.size();t++) {
	//Check the user's connectivity array
		checkArrayEntries(mesh->elem[t].conn,mesh->elem[t].n,
		     mesh->node.n, "element connectivity, from FEM_Set_Elem_Conn,");
	}
	
	splitter s(mesh,elem2chunk,nchunks);

	s.buildCommLists();
	
	s.separateSparse();
	
	s.addGhosts(nGhostLayers,g);
	
	//Split up and send out the mesh
	for (int c=0;c<nchunks;c++)
		out->accept(c,s.createMessage(c));
}


/********************************************************
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

//Add one layer of ghost elements
void splitter::addGhosts(int nLayers,const ghostLayer *g)
{
	if (nLayers==0) return; //No ghost layers-- nothing to do
	
//Build initial ghostNode table-- just the shared nodes
	ghostNode=new unsigned char[mesh->node.n];
	int n,nNode=mesh->node.n;
	for (n=0;n<nNode;n++) {
		ghostNode[n]=(gNode[n].isShared());
	}
//Set up the ghostStart counts:
	for (int c=0;c<nchunks;c++) {
		msgs[c]->m.node.ghostStart=chunks[c].node.size();
		for (int t=0;t<mesh->elem.size();t++) 
			msgs[c]->m.elem[t].ghostStart=chunks[c].elem[t].size();
	}

//Add each layer
	consistencyCheck();
	for (int i=0;i<nLayers;i++) {
		add(g[i]);
		consistencyCheck();
	}

//Free up memory
	delete[] ghostNode; ghostNode=NULL;
	for (int t=0;t<mesh->elem.size();t++)
	  {delete[] gElem[t];gElem[t]=NULL;}
}


//A linked list of elements surrounding a tuple.
//ElemLists are allocated one at a time, and hence much simpler than pelists:
class elemList {
public:
	int pe;
	int localNo;//Local number of this element on this PE
	int type; //Kind of element
	elemList *next;

	elemList() {next=NULL;}
	elemList(int pe_,int localNo_,int type_,elemList *next_=NULL)
		:pe(pe_),localNo(localNo_),type(type_),next(next_) {}
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
    int os=sizeof(elemList);
    return CkHashtableLayout(ks,ks,oo,os,oo+os);
  }
public:
	enum {MAX_TUPLE=8};
	tupleTable(int tupleLen_)
		:CkHashtable(makeLayout(tupleLen_),
			     137,0.75,
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
			delete doomed->next;
	}
	void addTuple(const int *tuple,int pe,int localNo,int type)
	{
		//Must make a canonical version of this tuple so different
		// orderings of the same nodes don't end up in different lists.
		//I canonicalize by sorting:
		int can[MAX_TUPLE];
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
		
		//First try for an existing list:
		elemList *dest=(elemList *)get(can);
		if (dest!=NULL) 
		{ //A list already exists here-- add in the new record
			dest->next=new elemList(pe,localNo,type,dest->next);
		} else {//No pre-existing list-- initialize a new one.
			dest=(elemList *)put(can);
			*dest=elemList(pe,localNo,type);
		}
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
		return (elemList *)ret;
	}
};

//Add one layer of ghost elements
void splitter::add(const ghostLayer &g)
{
	int totTuples=0,totGhostElem=0,totGhostNode=0; //For debugging
	//Build table mapping node-tuples to adjacent elements
	tupleTable table(g.nodesPerTuple);
	for (int c=0;c<nchunks;c++) {
	   for (int t=0;t<mesh->elem.size();t++) {
		if (!g.elem[t].add) continue;
		const CkVec<int> &src=chunks[c].elem[t];
		const int *elem2tuple=g.elem[t].elem2tuple;
		//For every element of every chunk:
		for (int e=0;e<src.size();e++) {
			int i,gNo=src[e];
			const int *conn=mesh->elem[t].connFor(gNo);
			if (hasGhostNodes(conn,mesh->elem[t].nodesPer))
			{ //Loop over this element's tuples:
			  for (int u=0;u<g.elem[t].tuplesPerElem;u++) {
				int tuple[tupleTable::MAX_TUPLE];
				for (i=0;i<g.nodesPerTuple;i++) {
					int eidx=elem2tuple[i+u*g.nodesPerTuple];
					if (eidx==-1) { //"not-there" node--
					  tuple[i]=-1; //Don't map via connectivity
					} else { //Ordinary node
					  int n=conn[eidx];
					  if (!ghostNode[n]) 
						break; //This tuple doesn't lie on a ghost boundary
					  tuple[i]=n;
					}
				}
				if (i==g.nodesPerTuple) 
				{ //This was a good tuple-- add it
					table.addTuple(tuple,c,e,t);
					totTuples++;
				}
			  }
			}
	        }
	   }
	}
       
	//Loop over all the tuples, connecting adjacent elements
	table.beginLookup();
	elemList *l;
	while (NULL!=(l=table.lookupNext())) {
		//Find all referenced processors:
		int pes[20];
		int nPes=0;
		for (elemList *b=l;b!=NULL;b=b->next) {
			int p;
			for (p=0;p<nPes;p++)
				if (b->pe==pes[p])
					break;
			if (p==nPes) //Not yet in list
				pes[nPes++]=b->pe;
		}
		
		//Consider adding each element as a ghost on each processor
		for (const elemList *a=l;a!=NULL;a=a->next)
		{
		  int t=a->type;
		  int srcPe=a->pe;
		  int gNo=chunks[srcPe].elem[t][a->localNo];
		  for (int p=0;p<nPes;p++)
		  {
			int destPe=pes[p];
			if (addGhost(gNo,srcPe,destPe,chunks[destPe].elem[t],gElem[t],
				     msgs[srcPe]->m.elem[t],msgs[destPe]->m.elem[t]))
			{
				totGhostElem++;
				//Add this element's nodes as ghost nodes
				const int *conn=mesh->elem[t].connFor(gNo);
				for (int i=0;i<mesh->elem[t].nodesPer;i++)
				{
					int gnNo=conn[i];
					ghostNode[gnNo]=1;
					if (g.addNodes) {
						if (addGhost(gnNo,srcPe,destPe,chunks[destPe].node,gNode,
							     msgs[srcPe]->m.node,msgs[destPe]->m.node))
							totGhostNode++;
					}
				}
			}
		  }
		}
	}
	
	printf("FEM Ghost layer> %d tuples, %d new ghost elements, %d new ghost nodes\n",
	       totTuples,totGhostElem,totGhostNode);
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
	int i,t,c,e,n;

//Find the global total number of nodes and elements
	int minOld_n=1000000000,maxOld_n=0,new_n=0; //Pre-existing and newly-created nodes
	for(c=0; c<nchunks;c++) {
		const MeshChunk *msg=msgs[c];
		for (n=0;n<msg->m.node.n;n++)
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
	m->node.n=(maxOld_n-minOld_n)+new_n;
	m->node.dataPer=msgs[0]->m.node.dataPer;
	m->node.allocUdata();
	for (i=m->node.udataCount()-1;i>=0;i--)
		m->node.udata[i]=0.0;
	
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
			for (e=0;e<msg->m.elem[t].n;e++)
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
		m->elem[t].n=(maxOld_e[t]-minOld_e[t])+new_e;
		m->elem[t].dataPer=msgs[0]->m.elem[t].dataPer;
		m->elem[t].allocUdata();
		for (i=m->elem[t].udataCount()-1;i>=0;i--)
			m->elem[t].udata[i]=0.0;
		m->elem[t].nodesPer=msgs[0]->m.elem[t].nodesPer;
		m->elem[t].allocConn();
		for (i=m->elem[t].connCount()-1;i>=0;i--)
			m->elem[t].conn[i]=-1;
	}
	
//Now copy over the local data and connectivity into the global mesh
	new_n=0;
	for(c=0; c<nchunks;c++) {
		MeshChunk *msg=msgs[c];
		for (n=0;n<msg->m.node.n;n++)
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
			for (e=0;e<msg->m.elem[t].n;e++)
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
				for (n=0;n<msg->m.elem[t].nodesPer;n++)
					dstConn[n]=msg->nodeNums[srcConn[n]];
			}
		}
	}

	delete[] minOld_e;
	delete[] maxOld_e;
	return m;
}
