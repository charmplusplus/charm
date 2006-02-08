/*
Finite Element Method (FEM) Framework for Charm++
Parallel Programming Lab, Univ. of Illinois 2002
Orion Sky Lawlor, olawlor@acm.org, 12/20/2002

Backward compatability file: implements non-mesh routines
in terms of the new mesh routines. We do this *without* 
using anything in impl.h; this file mostly relies on
public, documented interfaces.
*/
#include "ParFUM.h"
#include "ParFUM_internals.h"

#define S FEM_Mesh_default_write()
#define G FEM_Mesh_default_read()

/***************** Mesh Utility ******************/

// Renumber connectivity from bizarre new double-ended indexing to 
//  sensible old "first real, then ghost" connectivity.
static void renumberGhostConn(int nodeGhosts, //Number of real nodes
	int *conn,int per,int n,int idxbase) 
{
	for (int r=0;r<n;r++)
	for (int c=0;c<per;c++)
	{
		int i=conn[r*per+c];
		if (idxbase==0) { /* C version: */
			if (FEM_Is_ghost_index(i))
				conn[r*per+c]=nodeGhosts+FEM_From_ghost_index(i);
		} else { /* f90 version: */
			if (i<0)
				conn[r*per+c]=nodeGhosts+(-i);
		}
	}
}

// get/set this entity's connectivity, as both real and ghost:
static void mesh_conn(int fem_mesh,int entity,int *conn,int idxBase) {
	int n=FEM_Mesh_get_length(fem_mesh,entity);
	int per=FEM_Mesh_get_width(fem_mesh,entity,FEM_CONN);
	FEM_Mesh_data(fem_mesh,entity,FEM_CONN, conn, 0,n, FEM_INDEX_0+idxBase,per);
	
	if (FEM_Mesh_is_get(fem_mesh))
	{ /* Consider getting ghost data (never set ghost data) */
		int nGhosts=FEM_Mesh_get_length(fem_mesh,FEM_GHOST+entity);
		if (nGhosts>0) { /* Grab ghost data on the end of the regular data */ 
			int *ghostConn=&conn[n*per];
			FEM_Mesh_data(fem_mesh,FEM_GHOST+entity,FEM_CONN,
				ghostConn, 0,nGhosts, FEM_INDEX_0+idxBase,per);
			renumberGhostConn(FEM_Mesh_get_length(fem_mesh,FEM_NODE),ghostConn,per,nGhosts,idxBase);
		}
	}
}

// Little utility routine: get/set real and ghost data
static void mesh_data(int fem_mesh,int entity, int dtype,void *v_data) {
	int n=FEM_Mesh_get_length(fem_mesh,entity);
	int per=FEM_Mesh_get_width(fem_mesh,entity,FEM_DATA);
	FEM_Mesh_data(fem_mesh,entity,FEM_DATA,v_data, 0,n, dtype,per);
	
	if (FEM_Mesh_is_get(fem_mesh))
	{ /* Consider getting ghost data (never set ghost data) */
		int nGhosts=FEM_Mesh_get_length(fem_mesh,FEM_GHOST+entity);
		char *data=(char *)v_data;
		int size=IDXL_Layout::type_size(dtype);
		if (nGhosts>0) /* Grab ghost data on the end of the regular data */ 
			FEM_Mesh_data(fem_mesh,FEM_GHOST+entity,FEM_DATA,
				(void *)&data[n*per*size], 0,nGhosts, dtype,per);
	}
}

/***************** Mesh Set ******************/

// Lengths
CDECL void FEM_Set_node(int nNodes,int doublePerNode) {
	FEM_Mesh_set_length(S,FEM_NODE,nNodes);
	FEM_Mesh_set_width(S,FEM_NODE,FEM_DATA,doublePerNode);
}
FORTRAN_AS_C(FEM_SET_NODE,FEM_Set_node,fem_set_node,
	(int *n,int *d), (*n,*d)
)

CDECL void FEM_Set_elem(int elType,int n,int doublePerElem,int nodePerElem) {
	FEM_Mesh_set_length(S,FEM_ELEM+elType,n);
	FEM_Mesh_set_width(S,FEM_ELEM+elType,FEM_DATA,doublePerElem);
	FEM_Mesh_set_width(S,FEM_ELEM+elType,FEM_CONN,nodePerElem);
}
FORTRAN_AS_C(FEM_SET_ELEM,FEM_Set_elem,fem_set_elem,
	(int *t,int *n,int *d,int *c), (*t,*n,*d,*c)
)


// Data
  // FIXME: add FEM_[GS]et_[node|elem]_data_c
CDECL void FEM_Set_node_data(const double *data) {
	mesh_data(S,FEM_NODE,FEM_DOUBLE,(void *)data);
}
FORTRAN_AS_C(FEM_SET_NODE_DATA_R,FEM_Set_node_data,fem_set_node_data_r,
	(const double *data), (data)
)

CDECL void FEM_Set_elem_data(int elType,const double *data) {
	mesh_data(S,FEM_ELEM+elType,FEM_DOUBLE,(void *)data);
}
FORTRAN_AS_C(FEM_SET_ELEM_DATA_R,FEM_Set_elem_data,fem_set_elem_data_r,
	(int *t,const double *d), (*t,d)
)

// Connectivity
CDECL void FEM_Set_elem_conn(int elType,const int *conn) {
	mesh_conn(S,FEM_ELEM+elType, (int *)conn, 0);
}

FDECL void FTN_NAME(FEM_SET_ELEM_CONN_R,fem_set_elem_conn_r)
	(int *elType,const int *conn) 
{
	mesh_conn(S,FEM_ELEM+*elType, (int *)conn, 1);
}

// Sparse
CDECL void FEM_Set_sparse(int sid,int nRecords,
  	const int *nodes,int nodesPerRec,
  	const void *data,int dataPerRec,int dataType) 
{
	int entity=FEM_SPARSE+sid;
	FEM_Mesh_set_data(S,entity,FEM_CONN, (int *)nodes, 0,nRecords, FEM_INDEX_0,nodesPerRec);
	FEM_Mesh_set_data(S,entity,FEM_DATA, (void *)data, 0,nRecords, dataType,dataPerRec); 
}
FDECL void FTN_NAME(FEM_SET_SPARSE,fem_set_sparse)
	(int *sid,int *nRecords,
  	const int *nodes,int *nodesPerRec,
  	const void *data,int *dataPerRec,int *dataType) 
{
	int entity=FEM_SPARSE+*sid;
	int n=*nRecords;
	FEM_Mesh_set_data(S,entity,FEM_CONN, (int *)nodes, 0,n, FEM_INDEX_1,*nodesPerRec);
	FEM_Mesh_set_data(S,entity,FEM_DATA, (void *)data, 0,n, *dataType,*dataPerRec); 
}

CDECL void FEM_Set_sparse_elem(int sid,const int *rec2elem) 
{
	int entity=FEM_SPARSE+sid;
	int n=FEM_Mesh_get_length(S,FEM_SPARSE+sid);
	FEM_Mesh_set_data(S,entity,FEM_SPARSE_ELEM, (void *)rec2elem, 0,n, FEM_INDEX_0,2);
}
FDECL void FTN_NAME(FEM_SET_SPARSE_ELEM,fem_set_sparse_elem)
	(int *sid,int *rec2elem) 
{
	int entity=FEM_SPARSE+*sid;
	int i,n=FEM_Mesh_get_length(S,entity);
	// FEM_ELEM+rec2elem[2*i+0] is an element entity type--0 based
	// rec2elem[2*i+1] is an element number--1-based
	//  This means I can't naively use FEM_INDEX_0 *or* FEM_INDEX_1,
	//  so I have to do the index conversion right here.
	for (i=0;i<n;i++) rec2elem[2*i+1]--; //F to C indexing
	FEM_Mesh_set_data(S,entity,FEM_SPARSE_ELEM,
		(void *)rec2elem, 0,n, FEM_INDEX_0,2);
	for (i=0;i<n;i++) rec2elem[2*i+1]++; //Convert back	
}




/***************** Mesh Get ******************/
// Get total number of real and ghosts for this entity type
static int mesh_get_ghost_length(int mesh,int entity) {
	return FEM_Mesh_get_length(mesh,entity)+FEM_Mesh_get_length(mesh,FEM_GHOST+entity);
}

// Lengths
CDECL void FEM_Get_node(int *nNodes,int *perNode) {
	*nNodes=mesh_get_ghost_length(G,FEM_NODE);
	*perNode=FEM_Mesh_get_width(G,FEM_NODE,FEM_DATA);
}
FORTRAN_AS_C(FEM_GET_NODE,FEM_Get_node,fem_get_node,
	(int *n,int *per), (n,per))


CDECL void FEM_Get_elem(int elType,int *nElem,int *doublePerElem,int *nodePerElem) {
	*nElem=mesh_get_ghost_length(G,FEM_ELEM+elType);
	*doublePerElem=FEM_Mesh_get_width(G,FEM_ELEM+elType,FEM_DATA);
	*nodePerElem=FEM_Mesh_get_width(G,FEM_ELEM+elType,FEM_CONN);
}
FORTRAN_AS_C(FEM_GET_ELEM,FEM_Get_elem,fem_get_elem,
	(int *t,int *n,int *per,int *c), (*t,n,per,c))

// Data
CDECL void FEM_Get_node_data(double *data) {
	mesh_data(G,FEM_NODE,FEM_DOUBLE,data);
}
FORTRAN_AS_C(FEM_GET_NODE_DATA_R,FEM_Get_node_data,fem_get_node_data_r,
	(double *data), (data))

CDECL void FEM_Get_elem_data(int elType,double *data) {
	mesh_data(G,FEM_ELEM+elType,FEM_DOUBLE,data);
}
FORTRAN_AS_C(FEM_GET_ELEM_DATA_R,FEM_Get_elem_data,fem_get_elem_data_r,
	(int *elType,double *data), (*elType,data))


// Connectivity
CDECL void FEM_Get_elem_conn(int elType,int *conn) {
	mesh_conn(G,FEM_ELEM+elType,conn,0);
}

FDECL void FTN_NAME(FEM_GET_ELEM_CONN_R, fem_get_elem_conn_r)
	(int *elType,int *conn)
{
	mesh_conn(G,FEM_ELEM+*elType,conn,1);
}


// Sparse
CDECL int  FEM_Get_sparse_length(int sid) {
	return mesh_get_ghost_length(G,FEM_SPARSE+sid);
}
FORTRAN_AS_C_RETURN(int, FEM_GET_SPARSE_LENGTH,FEM_Get_sparse_length,fem_get_sparse_length,
	(int *sid), (*sid))

CDECL void FEM_Get_sparse(int sid,int *nodes,void *data) {
	int fem_mesh=G;
	int entity=FEM_SPARSE+sid;
	int dataType=FEM_Mesh_get_datatype(fem_mesh,entity,FEM_DATA);
	mesh_data(fem_mesh,entity,dataType,data);
	mesh_conn(fem_mesh,entity,nodes,0);
}
FDECL void FTN_NAME(FEM_GET_SPARSE,fem_get_sparse)(int *sid,int *nodes,void *data) {
	int fem_mesh=G;
	int entity=FEM_SPARSE+*sid;
	int dataType=FEM_Mesh_get_datatype(fem_mesh,entity,FEM_DATA);
	mesh_data(fem_mesh,entity,dataType,data);
	mesh_conn(fem_mesh,entity,nodes,1);
}

CDECL int FEM_Get_node_ghost(void) 
{ // Index of first ghost node==number of real nodes
	return FEM_Mesh_get_length(G,FEM_NODE);
}
FDECL int FTN_NAME(FEM_GET_NODE_GHOST,fem_get_node_ghost)(void) {
	return 1+FEM_Get_node_ghost();
}

CDECL int FEM_Get_elem_ghost(int elemType) 
{ // Index of first ghost element==number of real elements
	return FEM_Mesh_get_length(G,FEM_ELEM+elemType);
} 
FDECL int FTN_NAME(FEM_GET_ELEM_GHOST,fem_get_elem_ghost)(int *elType) {
	return 1+FEM_Get_elem_ghost(*elType);
}


/** Symmetries */

CDECL void FEM_Get_sym(int elTypeOrMinusOne,int *destSym)
{
	const char *callingRoutine="FEM_Get_sym";
	FEMAPI(callingRoutine);
	int mesh=G;
	int entity=FEM_ELEM+elTypeOrMinusOne;
	if (elTypeOrMinusOne==-1) entity=FEM_NODE;
	FEM_Entity &l=*FEM_Entity_lookup(mesh,entity,callingRoutine);
	int i,n=l.size();
	for (i=0;i<n;i++) destSym[i]=l.getSymmetries(i);
	int g=l.getGhost()->size();
	for (i=0;i<g;i++) destSym[n+i]=l.getGhost()->getSymmetries(i);
}
FDECL void FTN_NAME(FEM_GET_SYM,fem_get_sym)
	(int *elTypeOrZero,int *destSym)
{
	FEM_Get_sym(zeroToMinusOne(*elTypeOrZero),destSym);
}



/** Ancient compatability */

CDECL void FEM_Set_mesh(int nelem, int nnodes, int nodePerElem, int* conn) {
	FEM_Set_node(nnodes,0);
	FEM_Set_elem(0,nelem,0,nodePerElem);
	FEM_Set_elem_conn(0,conn);
}
FDECL void FTN_NAME(FEM_SET_MESH,fem_set_mesh)
        (int *nelem, int *nnodes, int *ctype, int *conn)
{
	int elType=1,zero=0;
	FTN_NAME(FEM_SET_NODE,fem_set_node) (nnodes,&zero);
	FTN_NAME(FEM_SET_ELEM,fem_set_elem) (&elType,nelem,&zero,ctype);
	FTN_NAME(FEM_SET_ELEM_CONN_R,fem_set_elem_conn_r) (&elType,conn);
}

/*************************************************
Mesh assembly/disassembly.  This really only makes
sense when you read it together with femmain.C.
*/

class updateState {
	int doWhat;
	MPI_Comm comm;
	int myRank, master;
	int or_mesh; /* old partitioned read mesh */
	int osr_mesh; /* old serial read mesh */
	int nsw_mesh; /* new serial write mesh */
public:
	updateState(int doWhat_) :doWhat(doWhat_) {
		comm=(MPI_Comm)FEM_chunk::get("FEM_Update_mesh")->defaultComm;
		MPI_Comm_rank(comm,&myRank);
		master=0;
		or_mesh=osr_mesh=nsw_mesh=-1;
	}
	~updateState() {
	}
	
	bool pre(void) 
	{
		/* Assemble serial read mesh from default write */
		or_mesh=FEM_Mesh_default_read(); /* stash the old read mesh */
		int ow_mesh=FEM_Mesh_default_write();
		FEM_Mesh_copy_globalno(or_mesh,ow_mesh);
		osr_mesh=FEM_Mesh_reduce(ow_mesh,master,(FEM_Comm_t)comm);
		FEM_Mesh_deallocate(ow_mesh);
		
		if (myRank==master && doWhat==FEM_MESH_UPDATE) 
			nsw_mesh=FEM_Mesh_allocate();
		
		FEM_Mesh_set_default_read(osr_mesh);
		FEM_Mesh_set_default_write(nsw_mesh);
		return (myRank==master);
	}
	void post(void) {
		if (doWhat==FEM_MESH_FINALIZE)
			MPI_Barrier(comm); /* other processors wait for main to finish */
		if (osr_mesh>0) FEM_Mesh_deallocate(osr_mesh);
		
		if (doWhat==FEM_MESH_UPDATE) 
		{ /* Partition the new serial mesh */
			FEM_Mesh_deallocate(or_mesh); /* get rid of the old read mesh */
			int nr_mesh=FEM_Mesh_broadcast(nsw_mesh,master,(FEM_Comm_t)comm);
			FEM_Mesh_set_default_read(nr_mesh);
		}
		else /* no update, switch back to old read mesh */
			FEM_Mesh_set_default_read(or_mesh);
		FEM_Mesh_set_default_write(FEM_Mesh_allocate());
		
		if (nsw_mesh>0) FEM_Mesh_deallocate(nsw_mesh);
	}
};

CDECL void FEM_Update_mesh(FEM_Update_mesh_fn callFn,int userValue,int doWhat) 
{
  updateState update(doWhat);
  if (update.pre() && 0 != userValue)
  	(callFn)(userValue);
  update.post();
}

FDECL void FTN_NAME(FEM_UPDATE_MESH,fem_update_mesh)
  (FEM_Update_mesh_fortran_fn callFn,int *userValue,int *doWhat) 
{ 
  updateState update(*doWhat);
  if (update.pre() && (0 != *userValue))
  	(callFn)(userValue);
  update.post();
}

/******************************************
FEM_Serial: these routines are only used by Rocflu's
mesh prep/post utilities.
*/
static int *splitMesh=NULL;
static int splitChunks=0;
CDECL void FEM_Serial_split(int nchunks) {
	splitChunks=nchunks;
	splitMesh=new int[splitChunks];
	FEM_Mesh_partition(FEM_Mesh_default_write(),splitChunks,splitMesh);
}
FORTRAN_AS_C(FEM_SERIAL_SPLIT,FEM_Serial_split,fem_serial_split, (int *n),(*n))

CDECL void FEM_Serial_begin(int chunkNo) {
	FEM_Mesh_write(splitMesh[chunkNo],"fem_mesh",chunkNo,splitChunks);
	FEM_Mesh_set_default_read(splitMesh[chunkNo]);
}
FORTRAN_AS_C(FEM_SERIAL_BEGIN,FEM_Serial_begin,fem_serial_begin, (int *c),(*c-1))


CDECL void FEM_Serial_read(int chunkNo,int nChunks) {
	if (splitMesh==NULL) {
		splitChunks=nChunks;
		splitMesh=new int[splitChunks];
		FEM_Mesh_deallocate(FEM_Mesh_default_write());
	}
	int readMesh=FEM_Mesh_read("fem_mesh",chunkNo,splitChunks);
	int writeMesh=FEM_Mesh_copy(readMesh);
	FEM_Mesh_become_set(writeMesh);
	splitMesh[chunkNo]=writeMesh;
	FEM_Mesh_deallocate(FEM_Mesh_default_read());
	FEM_Mesh_set_default_read(readMesh);
	FEM_Mesh_set_default_write(writeMesh);
}
FORTRAN_AS_C(FEM_SERIAL_READ,FEM_Serial_read,fem_serial_read, (int *c,int *n),(*c-1,*n))

CDECL void FEM_Serial_assemble(void) {
	int serialMesh=FEM_Mesh_assemble(splitChunks,splitMesh);
	for (int i=0;i<splitChunks;i++)
		FEM_Mesh_deallocate(splitMesh[i]);
	FEM_Mesh_set_default_read(serialMesh);
	FEM_Mesh_set_default_write(FEM_Mesh_allocate());
}
FORTRAN_AS_C(FEM_SERIAL_ASSEMBLE,FEM_Serial_assemble,fem_serial_assemble,(void),())


