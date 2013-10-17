/**
 * \addtogroup ParFUM
*/
/*@{*/

/*Charm++ Finite Element Framework:
  C++ implementation file

  This code implements fem_split and fem_assemble.
  Fem_split takes a mesh and partitioning table (which maps elements
  to chunks) and creates a sub-mesh for each chunk,
  including the communication lists. Fem_assemble is the inverse.

  The fem_split algorithm is O(n) in space and time (with n nodes,
  e elements, c chunks; and n>e>c^2).  The central data structure
  is a bit unusual-- it's a table that maps nodes to lists of chunks
  (all the chunks that share the node).  For any reasonable problem,
  the vast majority of nodes are not shared between chunks; 
  this algorithm uses this to keep space and time costs low.

  Memory usage for the large temporary arrays is n*sizeof(chunkList),
  allocated contiguously.  Shared nodes will result in a few independently 
  allocated chunkList entries, but shared nodes should be rare so this should 
  not be expensive.

  Originally written by Orion Sky Lawlor, olawlor@acm.org, 9/28/2000
*/


#include "ParFUM.h"
#include "ParFUM_internals.h"



static void checkEquality(const char *what, 
			  int v1,const char *src1, 
			  int v2,const char *src2)
{
  if (v1!=v2) {
    CkPrintf("FEM> %s value %d, from %d, doesn't match %d, from %d!\n",
	     what, v1,src1, v2,src2);
    CkAbort("FEM Equality assertation failed");
  }
}
static void checkRange(const char *what,int v,int max)
{
  if ((v<0)||(v>=max)) {
    CkPrintf("FEM> %s value %d should be between 0 and %d!\n",
	     what,v,max);
    CkAbort("FEM Range assertation failed");
  }
}

static void checkArrayEntries(const int *arr,int nArr,int max,const char *what)
{
#if CMK_ERROR_CHECKING
  //Check the array for out-of-bounds values
  for (int e=0;e<nArr;e++) checkRange(what,arr[e],max);
#endif
}

//Check all user-settable fields in this object for validity
static void check(const FEM_Mesh *mesh) {
#if 0 //FIXME
  for (int t=0;t<mesh->elem.size();t++) 
    if (mesh->elem.has(t)) {
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
#endif
}
static void check(const FEM_Partition &partition,const FEM_Mesh *mesh) {
  const int *canon=partition.getCanon();
  if (canon!=NULL)
    checkArrayEntries(canon,mesh->node.size(),mesh->node.size(),
		      "node canonicalization array, from FEM_Set_Symmetries");
}

/// Make sure this stencil makes sense for this mesh.
void FEM_Ghost_Stencil::check(const FEM_Mesh &mesh) const {
  const FEM_Elem &elem=mesh.elem[mesh.chkET(elType)];
  checkEquality("number of elements",
		n,"FEM_Ghost_stencil",
		elem.size(),"mesh");
  int i,prevEnd=0;
  for (i=0;i<n;i++) {
    int nElts=ends[i]-prevEnd;
    checkRange("FEM_Ghost_stencil index array",nElts,n);
    prevEnd=ends[i];
  }
  int m=ends[n-1];
  for (i=0;i<m;i++) {
    int t=adj[2*i+0];
    mesh.chkET(t);
    checkRange("FEM_Ghost_stencil neighbor array",
	       adj[2*i+1],mesh.elem[t].size());
  }
}

/************************ Splitter *********************/

/**
 * A dynamic (growing) representation of a chunk.  
 * Lists the global numbers of the local real elements of each chunk.
 *
 * This saves some memory by allowing us to stream out the chunks,
 * doing the copy for the large number of local elements as a final step
 * before sending the chunk off.
 *
 * As such, this (rather nasty!) class is *not* used for sparse data,
 * ghosts, symmetries, etc.
 */
class dynChunk {
public:
  typedef CkVec<int> dynList;
  NumberedVec<dynList> elem; 
  dynList node;
	
  //Add an element to the current list
  int addRealElement(int type,int globalNo) {
    elem[type].push_back(globalNo);
    return elem[type].size()-1;
  }
  int addRealNode(int globalNo) {
    node.push_back(globalNo);
    return node.size()-1;
  }
};

/**
 * Splitter is a big utility class used to separate an FEM_Mesh into pieces.
 * It build communication lists, ghosts, etc. and copies mesh data.
 */
class splitter {
  //Inputs:
  FEM_Mesh *mesh; //Unsplit source mesh
  const int *elem2chunk; //Maps global element number to destination chunk
  int nChunks; //Number of pieces to divide into
  //Output:
  FEM_Mesh **chunks; //Output mesh chunks [nChunks]

  //Processing:
  chunkList *gNode; //Map global node to owning chunks [mesh->node.size()]
  CkVec<chunkList *> gElem;//Map global element to owning chunks [mesh->elem.size()]
  dynChunk *dyn; //Growing list of local output elements [nChunks]
	
  /**
   * Renumber the global node numbers in this row of this table
   * to be local to this chunk, with these symmetries.
   */
  void renumberNodesLocal(int row, BasicTable2d<int> &table, 
			  int chunk, FEM_Symmetries_t sym)
  {
    int *nodes=table.getRow(row);
    int nNodes=table.width();
    for (int i=0;i<nNodes;i++)
      nodes[i]=gNode[nodes[i]].localOnChunk(chunk,sym);
  }

  //------- Used by sparse data splitter ------------
  FEM_Sparse **sparseDest; //Place to put each sparse record
	
  /// Copy the global sparse record src[s] into this chunk with these symmetries:
  void copySparse(const FEM_Sparse &src,int s,int chunk,FEM_Symmetries_t sym) {
    FEM_Sparse *dest=sparseDest[chunk];
    int d=dest->push_back(src,s);
    renumberNodesLocal(d, dest->setConn(), chunk, sym);
    //Renumber elements to be local
    if (dest->hasElements()) {
      // int *elem=dest->setElem().getRow(d);
      // int elType=elem[0]; //Element type is unchanged
      //FIXME: change elem[1] to be a local element number
    }
  }
	
  /// Copy the global sparse record src[s] into all the chunks it belongs in.
  void copySparseChunks(const FEM_Sparse &src,int s,bool forGhost);
	
	
  //---- Used by ghost layer builder ---
  unsigned char *ghostNode; //Flag: this node borders ghost elements [mesh->node.size()]
  const int *canon; //Node canonicalization array (may be NULL)
  const FEM_Symmetries_t *sym; //Node symmetries array (may be NULL)
  int curGhostLayerNo; // For preventing duplicate ghost adds
  int totGhostElem,totGhostNode; //For debugging printouts

  ///Add a ghost stencil: an explicit list of needed ghosts
  void addStencil(const FEM_Ghost_Stencil &s,const FEM_Partition &partition);

  /// Add an entire layer of ghost elements
  void addLayer(const FEM_Ghost_Layer &g,const FEM_Partition &partition);
	
  /// Return true if any of these global nodes are ghost nodes
  bool hasGhostNodes(const int *conn,int nodesPer) 
  {
    for (int i=0;i<nodesPer;i++)
      if (ghostNode[conn[i]])
	return true;
    return false;
  }

  /// Return an elemList entry if this tuple should be a ghost:
  bool addTuple(int *dest,FEM_Symmetries_t *destSym,
		const int *elem2tuple,int nodesPerTuple,const int *conn) const;

  /// Add this ghost, which arrises because of a mirror symmetry condition.
  void addSymmetryGhost(const elemList &a);
  /// Add the real element (srcType,srcNum) as a ghost for use by (destType,destNum).
  void addGlobalGhost(int srcType,int srcNum,  int destType,int destNum, bool addNodes);
	
  /// Check if src should be added as a ghost on dest's chunk.
  /// Calls addGhostElement and addGhostNode.
  void addGhostPair(const elemList &src,const elemList &dest,bool addNodes);

  /// Add this global element as a ghost between src and dest, or return -1
  int addGhostElement(int t,int gNo,int srcChunk,int destChunk,FEM_Symmetries_t sym);
	
  /// Add this global node as a ghost between src and dest, or return -1
  int addGhostNode(int gnNo,int srcChunk,int destChunk,FEM_Symmetries_t sym);
	
  /// Utility used by addGhostNode and addGhostElement
  int addGhostInner(const FEM_Entity &gEnt,int gNo, chunkList &gDest,
		    int srcChunk,FEM_Entity &srcEnt, int destChunk,FEM_Entity &destEnt,
		    FEM_Symmetries_t sym, int isNode, int t);
	
  ///isaac's element to element adjacency creation
  void addElemElems(const FEM_Partition &partition);
  void buildElemElemData(const FEM_ElemAdj_Layer &g,const FEM_Partition &partition);
  


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
      CkPrintf("ERROR! %s: %d is out of range (%d,%d)!\n",what,value,lo,hi);
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
  splitter(FEM_Mesh *mesh_,const int *elem2chunk_,int nChunks_);
  ~splitter();

  //Fill the gNode[] array with the chunks sharing each node
  void buildCommLists(void);
	
  //Add the layers of ghost elements
  void addGhosts(const FEM_Partition &partition);
	
  //Divide up the sparse data lists
  void separateSparse(bool forGhost);
	
  //Free up everything except what's needed during createMesh
  void aboutToCreate(void);
	
  //Divide off this portion of the serial data (memory intensive)
  FEM_Mesh *createMesh(int c);
	
  void consistencyCheck(void);

};


splitter::splitter(FEM_Mesh *mesh_,const int *elem2chunk_,int nChunks_)
  :mesh(mesh_), elem2chunk(elem2chunk_),nChunks(nChunks_)
{
  chunks=new FEM_Mesh* [nChunks];
  dyn=new dynChunk[nChunks];
  int c;//chunk number (to receive message)
  for (c=0;c<nChunks;c++) {
    chunks[c]=new FEM_Mesh; //Ctor starts all node and element counts at zero
    chunks[c]->copyShape(*mesh);
    dyn[c].elem.makeLonger(mesh->elem.size()-1);
  }
	
  gNode=new chunkList[mesh->node.size()];
  gElem.resize(mesh->elem.size());
  for (int t=0;t<mesh->elem.size();t++) {
    if (mesh->elem.has(t))
      gElem[t]=new chunkList[mesh->elem[t].size()];
    else
      gElem[t]=NULL;
  }
	
  sparseDest=new FEM_Sparse* [nChunks];
  ghostNode=NULL;
  canon=NULL;
}

splitter::~splitter() {
  delete[] gNode;
  delete[] dyn;
  for (int t=0;t<mesh->elem.size();t++)
    delete[] gElem[t];
  for (int c=0;c<nChunks;c++)
    delete chunks[c];
  delete[] chunks;
  delete[] sparseDest;
}

void splitter::buildCommLists(void)
{
  //First pass: add chunks to the lists for each node they touch
  //  (also find the local elements and node counts)
  int t;//Element type
  int e;//Element number in type
  int n;//Node number around element
  for (t=0;t<mesh->elem.size();t++) 
    if (mesh->elem.has(t)) {
      int typeStart=mesh->nElems(t);//Number of elements before type t
      const FEM_Elem &src=mesh->elem[t]; 
      for (e=0;e<src.size();e++) {
	int c=elem2chunk[typeStart+e];
	const int *srcConn=src.getConn().getRow(e);
	gElem[t][e].set(c,dyn[c].addRealElement(t,e),noSymmetries,-1);
	int lNo=dyn[c].node.size();
	for (n=0;n<src.getNodesPer();n++) {
	  int gNo=srcConn[n];
	  if (gNode[gNo].addchunk(c,lNo,noSymmetries,-1)) {
	    //Found a new local node
	    dyn[c].addRealNode(gNo);
	    lNo++;
	  }
	}
      }
    }
	
  //Second pass: add shared nodes to comm. lists
  for (n=0;n<mesh->node.size();n++) {
    if (gNode[n].isShared()) 
      {/*Node is referenced by more than one chunk-- 
	 Add it to all the chunk's comm. lists.*/
	int len=gNode[n].length();
	for (int bi=0;bi<len;bi++)
	  for (int ai=0;ai<bi;ai++)
	    {
	      chunkList &a=gNode[n][ai];
	      chunkList &b=gNode[n][bi];
	      chunks[a.chunk]->node.shared.add(
					       a.chunk,a.localNo,
					       b.chunk,b.localNo,
					       chunks[b.chunk]->node.shared);
	    }
      }
  }
}


//Free up everything except what's needed during createMesh
void splitter::aboutToCreate() 
{
  for (int t=0;t<mesh->elem.size();t++)
    {delete[] gElem[t];gElem[t]=NULL;}
}

//Now that we know which elements and nodes go where,
// actually copy the local real node and element data.
// This is postponed to the very end, because the local data is big.
FEM_Mesh *splitter::createMesh(int c)
{
  int t;
  FEM_Mesh &dest=*chunks[c];

  dest.udata=mesh->udata;
	
  //Add each local real node
  FEM_Node *destNode=&dest.node;
  const dynChunk::dynList &srcIdx=dyn[c].node;
  int nNode=srcIdx.size();
  destNode->setLength(nNode);
  for (int lNo=0;lNo<nNode;lNo++) {
    int gNo=srcIdx[lNo];
    //Copy the node userdata & symmetries
    destNode->copyEntity(lNo, mesh->node, gNo);
    destNode->setSymmetries(lNo,noSymmetries);
    //A node is primary on its first chunk
    chunkList *head=&gNode[gNo];
    const chunkList *cur=head->onChunk(c,noSymmetries);
    destNode->setPrimary(lNo,head==cur);
  }

  //Add each local real element
  for (t=0;t<dest.elem.size();t++) 
    if (dest.elem.has(t)) 
      {
	FEM_Elem *destElem=&dest.elem[t];
	const dynChunk::dynList &srcIdx=dyn[c].elem[t];
	int nEl=srcIdx.size();
	destElem->setLength(nEl);
	for (int lNo=0;lNo<nEl;lNo++) {
	  int gNo=srcIdx[lNo];
	  //Copy the element userdata & symmetries
	  destElem->copyEntity(lNo, mesh->elem[t], gNo);
	  //Translate the connectivity from global to local normally
	  renumberNodesLocal(lNo, destElem->setConn(), c,noSymmetries);
	}
      }
	
  chunks[c]=NULL;
  dest.becomeGetting(); //Done modifying this mesh
  return &dest;
}

/*External entry point:
  Create a sub-mesh for each chunk's elements,
  including communication lists between chunks.
*/
void FEM_Mesh_split(FEM_Mesh *mesh,int nChunks,
		    const FEM_Partition &partition,FEM_Mesh_Output *out)
{
  const int *elem2chunk=partition.getPartition(mesh,nChunks);
  //Check the elem2chunk mapping:
  CkThresholdTimer time("FEM Split> Setting up",1.0);
  checkArrayEntries(elem2chunk,mesh->nElems(),nChunks,
		    "elem2chunk, from FEM_Set_Partition or metis,");
  check(mesh);
  check(partition,mesh);
	
  mesh->setSymList(partition.getSymList());
  splitter s(mesh,elem2chunk,nChunks);

  time.start("FEM Split> Finding comm lists");
  s.buildCommLists();
	
  time.start("FEM Split> Finding ghosts");
  s.separateSparse(false); //Copies real sparse elements	
  s.addGhosts(partition);
  s.separateSparse(true); //Copies ghost sparse elements
	
  time.start("FEM Split> Copying mesh data");
  //Split up and send out the mesh
  s.aboutToCreate(); //Free up memory
  for (int c=0;c<nChunks;c++)
    out->accept(c,s.createMesh(c));
}

/*************************** Sparse **************************/

/// Copy the global sparse record src[s] into all the chunks it belongs in.
void splitter::copySparseChunks(const FEM_Sparse &src,int s,bool forGhost)
{
  if (src.hasElements()) 
    { //We have an element list-- put this record on its element's chunk
      const int *elemRec=src.getElem().getRow(s);
      int t=elemRec[0];
      int e=elemRec[1];
      if (!forGhost) 
	{ //Real sparse goes where its real element lives:
	  copySparse(src,s,elem2chunk[mesh->getGlobalElem(t,e)],noSymmetries);
	} else { //Ghost sparse goes wherever its element has ghosts:
	  chunkList *cur=&gElem[t][e]; //List of places our element lives:
	  while (cur!=NULL) {
	    if (FEM_Is_ghost_index(cur->localNo)) //He's a ghost here:
	      copySparse(src,s,cur->chunk,cur->sym);
	    cur=cur->next;
	  }
	}
    }
  else
    { //No element list-- split up by intersecting the node lists:
      int nNodes=src.getConn().width();
      if (nNodes==0) CkAbort("Registered an FEM sparse without nodes or elements");
      FEM_Symmetries_t sym=noSymmetries; //FIXME: node-sparse with symmetries (like elems?)
      const int *nodes=src.getConn().getRow(s);
      int nCopied=0;
		
      //Traverse the first node's list, testing each entry:
      for (const chunkList *cur=&gNode[nodes[0]];cur!=NULL;cur=cur->next)
	{
	  int testchunk=cur->chunk;
	  bool hasGhost=false; //Some of our nodes are ghost
	  for (int i=0;i<nNodes;i++) {
	    const chunkList *li=gNode[nodes[i]].onChunk(testchunk,sym);
	    if (li==NULL) //List i does not contain testchunk
	      { testchunk=-1; break; /* jump out */ }
	    if (FEM_Is_ghost_index(li->localNo))
	      hasGhost=true;
	  }
	  if (testchunk==-1) continue; //Not even on this chunk
	  if (forGhost && !hasGhost) continue; //Already copied this one
	  copySparse(src,s,testchunk,sym);
	  nCopied++;
	}
      if (nCopied==0 && !forGhost) FEM_Abort("copySparseChunks",
					     "Sparse record %d does not lie on any chunk!",s);
    }
}

//Decide which chunks each sparse data record belongs in, and copy the 
// sparse record there.
void splitter::separateSparse(bool forGhost) 
{
  for (int t=0;t<mesh->sparse.size();t++) 
    if (mesh->sparse.has(t))
      { //For each kind of sparse data:
	//Prepare the chunks to receive sparse records:
	const FEM_Sparse &src=mesh->sparse.get(t);
	for (int c=0;c<nChunks;c++) 
	  { //Find the structure to receive this chunk's data
	    FEM_Sparse *d=&chunks[c]->sparse.set(t);
	    if (forGhost) d=(FEM_Sparse *)d->getGhost();
	    sparseDest[c]=d;
	  }
		
	for (int s=0;s<src.size();s++) copySparseChunks(src,s,forGhost);
      }
}


/***************************  Ghosts   *****************************
Ghost elements: read-only copies of elements on other chunks.

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
of nodes that could cross a chunk boundary.  

The "interesting" nodes are marked in ghostNodes, and for
the first layer are equal to the shared nodes.  Each time
a ghost layer is added, the set of interesting nodes grows.
*/



CkHashCode CkHashFunction_ints(const void *keyData,size_t keyLen)
{
  const int *d=(const int *)keyData;
  int l=keyLen/sizeof(int);
  CkHashCode ret=d[0];
  for (int i=1;i<l;i++)
    ret=ret^circleShift(d[i],i*23);
  return ret;
}

int CkHashCompare_ints(const void *k1,const void *k2,size_t keyLen)
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

//Add all the layers of ghost elements
void splitter::addGhosts(const FEM_Partition &partition)
{
  if (partition.getRegions()==0) return; //No ghost region-- nothing to do
	
  //Build initial ghostNode table-- just the shared nodes
  ghostNode=new unsigned char[mesh->node.size()];
  int n,nNode=mesh->node.size();
  for (n=0;n<nNode;n++) {
    ghostNode[n]=(gNode[n].isShared());
  }

  //Mark the symmetry nodes as being ghost-capable
  canon=partition.getCanon();
  sym=partition.getSymmetries();
  if (sym!=NULL)
    for (n=0;n<nNode;n++)
      if (sym[n]!=(FEM_Symmetries_t)0)
	ghostNode[n]=1;

  // Add all the ghost regions
  curGhostLayerNo=0;
  for (int regNo=0;regNo<partition.getRegions();regNo++)
    {
      const FEM_Ghost_Region &r=partition.getRegion(regNo);
      totGhostElem=0,totGhostNode=0; //For debugging
      int thisLayer=curGhostLayerNo;
		
      if (r.layer!=0) {
	addLayer(*r.layer,partition);
	curGhostLayerNo++;
      }
      else if (r.stencil!=0) 
	{ /* HACK: stencils don't update layer number until marker */
	  addStencil(*r.stencil,partition);
	}
      else {/* layer and stencil==0: a stencil layer marker */
	curGhostLayerNo++;
      }
		
      if (totGhostElem>0)
	CkPrintf("FEM Ghost %s %d> %d new ghost elements, %d new ghost nodes\n",
		 r.layer?"layer":"stencil",
		 1+thisLayer,
		 totGhostElem,totGhostNode);
    }
	
  delete[] ghostNode; ghostNode=NULL;
}

//Add a ghost stencil: an explicit list of needed ghosts
void splitter::addStencil(const FEM_Ghost_Stencil &s,const FEM_Partition &partition)
{
  int t=s.getType();
  s.check(*mesh);
  int i,j,n=mesh->elem[t].size();
  for (i=0;i<n;i++) {
    const int *nbor;
    j=0;
    while (NULL!= (nbor=s.getNeighbor(i,j++))) {
      /* Element i (of type t) depends on 
	 element nbor[1] (of type nbor[0]) */
      addGlobalGhost(nbor[0],nbor[1],t,i, s.wantNodes());
    }
  }
}

/// Add the real element (srcType,srcNum) as a ghost for use by (destType,destNum).
///  This routine is used in adding stencils.
void splitter::addGlobalGhost(
			      int srcType,int srcNum,  int destType,int destNum, bool doAddNodes)
{
  // Source is the real element
  chunkList &srcC=gElem[srcType][srcNum];
  // Loop over the list of real and ghost elements for dest:
  for (chunkList *destC=&gElem[destType][destNum];
       destC!=NULL;
       destC=destC->next)
    if (srcC.chunk!=destC->chunk // We're not on the same chunk
	&& destC->layerNo<curGhostLayerNo  //I wasn't just added
	)
      { // Add src as a ghost on the chunk of dest.
	elemList src(srcC.chunk, srcC.localNo, srcType, (FEM_Symmetries_t)0);
	elemList dest(destC->chunk, destC->localNo, destType, (FEM_Symmetries_t)0);
	addGhostPair(src,dest,doAddNodes);
      }
}

//Add one layer of ghost elements
void splitter::addLayer(const FEM_Ghost_Layer &g,const FEM_Partition &partition)
{
  tupleTable table(g.nodesPerTuple);
	
  //Build table mapping node-tuples to lists of adjacent elements
  for (int t=0;t<mesh->elem.size();t++) 
    if (mesh->elem.has(t)) {
      if (!g.elem[t].add) continue; //Don't add this kind of element to the layer
      //For every element of this type:
      int gElemCount=mesh->elem[t].size();
      for (int gNo=0;gNo<gElemCount;gNo++) 
	{
	  const int *conn=mesh->elem[t].connFor(gNo);
	  if (hasGhostNodes(conn,mesh->elem[t].getNodesPer()))
	    { //Loop over this element's ghosts and tuples:
	      for (chunkList *cur=&gElem[t][gNo];cur!=NULL;cur=cur->next)
		for (int u=0;u<g.elem[t].tuplesPerElem;u++)
		  {
		    int tuple[tupleTable::MAX_TUPLE];
		    FEM_Symmetries_t allSym;
		    if (addTuple(tuple,&allSym,
				 &g.elem[t].elem2tuple[u*g.nodesPerTuple],
				 g.nodesPerTuple,conn)) {
		      table.addTuple(tuple,new elemList(cur->chunk,cur->localNo,t,allSym));
		    }
		  }
	    }
	}
    }
	
  //Loop over all the tuples, connecting adjacent elements
  table.beginLookup();
  elemList *l;
  while (NULL!=(l=table.lookupNext())) {
    if (l->next==NULL) //One-entry list: must be a symmetry
      addSymmetryGhost(*l);
    else { /* Several elements in list: normal case */
      //Consider adding ghosts for all element pairs on this tuple:
      for (const elemList *a=l;a!=NULL;a=a->next)
	for (const elemList *b=l;b!=NULL;b=b->next) 
	  if (a!=b && a->localNo>=0) /* only add ghosts of real elements */
	    addGhostPair(*a,*b,g.addNodes);
    }
  }
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

//Add a record for this single-element tuple list
void splitter::addSymmetryGhost(const elemList &a)
{
  if (a.sym==0) return; //Not a symmetry ghost
  CkAbort("FEM map> Mirror symmetry ghosts not yet supported");
}


/**
 * Add a record for this ordered pair of adjacent elements:
 * Consider adding the real element src as a ghost element 
 * on dest's chunk.  Src and dest do *not* need to have the
 * same type, but they will be adjacent.
 * 
 * If either of src and dest have symmetries, both must have symmetries.
 */
void splitter::addGhostPair(const elemList &src,const elemList &dest,bool addNodes)
{
  //assert((src.sym==0) == (dest.sym==0))
  int srcChunk=src.chunk;
  int destChunk=dest.chunk;
  int elemSym=dest.sym;
  bool isSymmetry=(dest.sym!=0);
	
  if ((!isSymmetry) && srcChunk==destChunk) 
    return; //Unless via symmetry, never add interchunk ghosts from same chunk
	
  int t=src.type;
  if (src.localNo<0) FEM_Abort("addGhostPair","Cannot add a ghost of a ghost (src num=%d)",src.localNo);
  int gNo=dyn[srcChunk].elem[t][src.localNo];
	
  int newNo=addGhostElement(t,gNo,srcChunk,destChunk,elemSym);
  if (newNo==-1) return; //Ghost is already there
  
  //If we get here, we just added src as a new ghost ("new") to dest's chunk--
  // add src's nodes as ghost nodes:
  int srcConnN=mesh->elem[t].getNodesPer();
  const int *srcConn=mesh->elem[t].connFor(gNo);
  
  int dt=dest.type;
  int destConnN=mesh->elem[dt].getNodesPer();
  const int *destConn=NULL; //Connectivity for old target
  FEM_Elem *destElemGhosts=(FEM_Elem *)chunks[destChunk]->elem[t].getGhost();
  int *newConn=destElemGhosts->connFor(newNo); //Connectivity for new ghost of src
  if (isSymmetry) { //Symmetry elements get a new connectivity array
    destConn=mesh->elem[dt].connFor(dyn[destChunk].elem[dt][dest.localNo]); //FIXME: what if dest is itself a ghost (localNo<0)
  }
  
  //Consider adding a ghost of each of src's nodes, too:
  for (int sn=0;sn<srcConnN;sn++)
    {
      FEM_Symmetries_t nodeSym=noSymmetries;
      int gnNo=srcConn[sn];
      ghostNode[gnNo]=1; //Mark this node as "interesting" for later ghost layers
      if (isSymmetry) 
	{ // Nodes of symmetry ghosts may need serious tweaking:
	  // FIXME: handle multiple reflections here
	  //Unconnected new nodes take the symmetry of the tuple
	  nodeSym=dest.sym; 
	  for (int dn=0;dn<destConnN;dn++) 
	    if (canon[destConn[dn]]==canon[gnNo])
	      { // This node is actually a symmetric copy of destConn[dn]
		gnNo=destConn[dn];
		//FIXME: nodeSym should take into account multiple reflections
		nodeSym=noSymmetries; 
		break;
	      }
	}
      if (addNodes)
	addGhostNode(gnNo,srcChunk,destChunk,nodeSym);
		
      //Renumber the element connectivity
      newConn[sn]=gNode[gnNo].localOnChunk(destChunk,nodeSym);
    }
}


/// Add this global element as a ghost between src and dest, or return -1
int splitter::addGhostElement(int t,int gNo,int srcChunk,int destChunk,FEM_Symmetries_t sym)
{
  FEM_Elem &destEnt=*(FEM_Elem *)chunks[destChunk]->elem[t].getGhost();
  int destNo=addGhostInner(mesh->elem[t],gNo,gElem[t][gNo],
			   srcChunk,chunks[srcChunk]->elem[t], 
			   destChunk, destEnt,
			   sym, 0, t);
  if (destNo!=-1)
    {
      totGhostElem++;
      return destNo;
    }
  return -1;
}
/// Add this global node as a ghost between src and dest, or return -1
int splitter::addGhostNode(int gnNo,int srcChunk,int destChunk,FEM_Symmetries_t sym)
{
  int destNo=addGhostInner(mesh->node,gnNo,gNode[gnNo],
			   srcChunk,chunks[srcChunk]->node, 
			   destChunk,*chunks[destChunk]->node.getGhost(),
			   sym, 1, 0);
  if (destNo!=-1)
    {
      totGhostNode++;
      return destNo;
    }
  return -1;
}

/**
 * Add this global number (gEnt[gNo], with list gDest)
 * as a ghost sent from srcChunk's real entity srcEnt, 
 * as a ghost recv'd at destChunk's ghost entity destEnt,
 * under the symmetries sym.  Since this routine is so horrible,
 * always use the wrapper routines addGhostNode or addGhostElem.
 *
 * Returns the (non-ghostindex'd) index of this entity in
 * dstEnt, or -1 if the ghost was not added.
 */
int splitter::addGhostInner(const FEM_Entity &gEnt,int gNo, chunkList &gDest,
			    int srcChunk,FEM_Entity &srcEnt, int destChunk,FEM_Entity &destEnt,
			    FEM_Symmetries_t sym, int isNode, int t)
{
  if (gDest.onChunk(destChunk,sym))
    return -1; //Dest already has a copy of this entity
  int srcNo=gDest.localOnChunk(srcChunk,noSymmetries);
  if (srcNo<0)
    return -1; //Never add ghosts of ghosts!
  //Actually copy the element data (from global) to dest.
  // We can't copy from src yet, because srcNo is real and
  // hence hasn't been copied yet (it's sitting in the dyn array).
  int destNo=destEnt.push_back(gEnt,gNo);
  destEnt.setSymmetries(destNo,sym);
  gDest.addchunk(destChunk,FEM_To_ghost_index(destNo),sym,curGhostLayerNo);
  chunkList *l = &gDest;
  while(l->next != NULL) {
    if(l->chunk != srcChunk && l->chunk != destChunk && l->localNo>=0) {
      if(isNode) 
	chunks[l->chunk]->node.setGhostSend().add(l->chunk,l->localNo,destChunk,destNo,destEnt.setGhostRecv());
      else 
	chunks[l->chunk]->elem[t].setGhostSend().add(l->chunk,l->localNo,destChunk,destNo,destEnt.setGhostRecv());
    }
    l = l->next;
  }
  srcEnt.setGhostSend().add(srcChunk,srcNo,destChunk,destNo,
			    destEnt.setGhostRecv());
  return destNo;
}


//Basic debugging tool: check interlinked ghost element data structures for consistency
void splitter::consistencyCheck(void)
{
#if CMK_ERROR_CHECKING
  bool skipCheck=false; 
#else
  bool skipCheck=true; /* Skip the check in production code */
#endif
  if (skipCheck) return; //Skip check in production code

  CkThresholdTimer time("FEM Split> Performing consistency check",1.0);

  int t,c;
  FEM_Symmetries_t sym=noSymmetries;
  //Make sure all the local elements listed in dyn are also in gElem[] and gNode[]
  for (c=0;c<nChunks;c++) {
    for (t=0;t<mesh->elem.size();t++)
      if (mesh->elem.has(t)) {
	const dynChunk::dynList &srcIdx=dyn[c].elem[t];
	int nEl=srcIdx.size();
	for (int lNo=0;lNo<nEl;lNo++) {
	  int gNo=srcIdx[lNo];
	  range(gNo,0,mesh->elem[t].size(),"global element number");
	  equal(gElem[t][gNo].localOnChunk(c,sym),lNo,"gElem[t] local number");
	}
      } 
    const dynChunk::dynList &srcIdx=dyn[c].node;
    int nNo=srcIdx.size();
    for (int lNo=0;lNo<nNo;lNo++) {
      int gNo=srcIdx[lNo];
      range(gNo,0,mesh->node.size(),"global node number");
      equal(gNode[gNo].localOnChunk(c,sym),lNo,"gNode[] local number");
    }
  }
	
  //Make sure everything in gNode and gElem is either a ghost or in dyn
  for (int gNo=0;gNo<mesh->node.size();gNo++) {
    for (chunkList *l=&gNode[gNo];l!=NULL;l=l->next) {
      if (l->chunk<0) /* l->chunk==-1 at initialization */
	FEM_Abort("partitioning","Global node %d (0-based) is not connected to any elements!",gNo);
      range(l->chunk,0,nChunks,"gNode chunk");
      if (!FEM_Is_ghost_index(l->localNo))
	equal(dyn[l->chunk].node[l->localNo],gNo,"chunk local node");
    }
  }
	
  for (t=0;t<mesh->elem.size();t++) 
    if (mesh->elem.has(t)) {
      for (int gNo=0;gNo<mesh->elem[t].size();gNo++) {
	for (chunkList *l=&gElem[t][gNo];l!=NULL;l=l->next) {
	  range(l->chunk,0,nChunks,"gElem chunk");
	  if (!FEM_Is_ghost_index(l->localNo))
	    equal(dyn[l->chunk].elem[t][l->localNo],gNo,"chunk local element");
	}
      }
    }

  //FIXME: Make sure all the communication lists exactly match
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

/// Reassemble entities based on their global numbers.
class FEM_Entity_numberer {
  CkVec<unsigned char> occupied; //If 1, this global number is taken
  int occupiedBefore; //First global number that might *not* be occupied
	
  //Find next un-occupied global number:
  int nextUnoccupied(void) {
    while (occupiedBefore<occupied.size()) {
      if (occupied[occupiedBefore]==0) 
	return occupiedBefore;
      else occupiedBefore++;
    }
    /* occupiedBefore==occupied.size(), so add to end of list */
    occupied.push_back(1);
    return occupiedBefore;
  }
	
public:
  FEM_Entity_numberer() { occupiedBefore=0; }
	
  /// Mark this entity's global numbers as used:
  void mark(FEM_Entity &src) {
    int l,len=src.size();
    for (l=0;l<len;l++) {
      int g=src.getGlobalno(l);
      if (g!=-1) {
	while (occupied.size()<=g) { //Make room for this marker
	  occupied.push_back(0);
	}
	//FIXME: make sure element global numbers aren't repeated
	// (tough because *node* global numbers may be repeated)
	occupied[g]=1;
      }
    }
  }
	
  /// Add src's global numbers to g_dest
  void add(FEM_Entity &l_src,FEM_Entity &g_dest) {
    int l,len=l_src.size();
    for (l=0;l<len;l++) add(l_src,l,g_dest);
  }
	
  /// Add the single entity l from src to g_dest.  Returns global number used.
  int add(FEM_Entity &l_src,int l,FEM_Entity &g_dest) {
    int g=l_src.getGlobalno(l);
    if (g==-1) { //Find a new unused global number
      g=nextUnoccupied();
      occupied[g]=1;
      //Mark this local entity's new global number:
      l_src.setGlobalno(l,g);
    }
    if (g_dest.size()<=g) g_dest.setLength(g+1);
    g_dest.copyEntity(g,l_src,l);
    g_dest.setGlobalno(g,g);
    return g;
  }
};

// Renumber the node connectivity of src_t[l] to dest_e[g] via the nodes listed.
static void renumberConn(const FEM_Elem &src_e,int l,FEM_Elem &dest_e,int g,
			 const FEM_Mesh &mesh)
{
  const int *srcConn=src_e.connFor(l);
  int *dstConn=dest_e.setConn().getRow(g);
  for (int n=0;n<src_e.getNodesPer();n++)
    dstConn[n]=mesh.node.getGlobalno(srcConn[n]);
}

FEM_Mesh *FEM_Mesh_assemble(int nChunks,FEM_Mesh **chunks)
{
  int t,c;
  FEM_Mesh *m=new FEM_Mesh;
  for(c=0; c<nChunks;c++) //Union up all possible shapes
    m->copyShape(*chunks[c]);
	
  m->udata=chunks[0]->udata;
	
  // Copy over nodes:
  FEM_Entity_numberer nodeNum;
  for(c=0; c<nChunks;c++) nodeNum.mark(chunks[c]->node);
  for(c=0; c<nChunks;c++) nodeNum.add(chunks[c]->node,m->node);
	
  // Copy over elements
  int nElemTypes=m->elem.size();
  for (t=0;t<nElemTypes;t++) {
    FEM_Entity_numberer elemNum;
    for(c=0; c<nChunks;c++) 
      if (chunks[c]->elem.has(t))
	elemNum.mark(chunks[c]->elem[t]);
		
    for(c=0; c<nChunks;c++) 
      if (chunks[c]->elem.has(t)) {
	FEM_Elem &src_e=chunks[c]->elem[t];
	if (!m->elem.has(t)) m->elem.set(t).copyShape(src_e);
	FEM_Elem &dest_e=m->elem[t];
	for (int l=0;l<src_e.size();l++) {
	  int g=elemNum.add(src_e,l,dest_e);
	  renumberConn(src_e,l,dest_e,g,*chunks[c]);
	}
      }
  }

  // Copy over sparse data
  int nSparseTypes=m->sparse.size();
  for (t=0;t<nSparseTypes;t++) {
    FEM_Entity_numberer sparseNum;
    for(c=0; c<nChunks;c++) 
      if (chunks[c]->sparse.has(t))
	sparseNum.mark(chunks[c]->sparse[t]);
		
    for(c=0; c<nChunks;c++) 
      if (chunks[c]->sparse.has(t)) {
	FEM_Sparse &src_e=chunks[c]->sparse[t];
	if (!m->sparse.has(t)) m->sparse.set(t).copyShape(src_e);
	FEM_Sparse &dest_e=m->sparse[t];
	for (int l=0;l<src_e.size();l++) {
	  int g=sparseNum.add(src_e,l,dest_e);
	  renumberConn(src_e,l,dest_e,g,*chunks[c]);
	  // FIXME: renumber sparse elem, too!
	}
      }
  }
	
  m->becomeGetting(); //Done modifying this mesh (for now)
  return m;
}
/*@}*/
