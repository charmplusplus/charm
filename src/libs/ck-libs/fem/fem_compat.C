/*
Finite Element Method (FEM) Framework for Charm++
Parallel Programming Lab, Univ. of Illinois 2002
Orion Sky Lawlor, olawlor@acm.org, 12/20/2002

Backward compatability file: implements non-mesh routines
in terms of the new mesh routines. We do this *without* 
using anything in fem_impl.h; this file mostly relies on
public, documented interfaces.
*/
#include "fem.h"
#include "fem_impl.h"
#include "charm-api.h" /*for CDECL, FTN_NAME*/

#define S FEM_Mesh_default_write()
#define G FEM_Mesh_default_read()

/***************** Mesh Set ******************/

// Little utility routine: get/set real and ghost data, as doubles
static void mesh_double_data(int fem_mesh,int entity, const double *data) {
	int n=FEM_Mesh_get_length(fem_mesh,entity);
	int per=FEM_Mesh_get_width(fem_mesh,entity,FEM_DATA);
	FEM_Mesh_data(fem_mesh,entity,FEM_DATA,(void *)data, 0,n, FEM_DOUBLE,per);
	int nGhosts=FEM_Mesh_get_length(fem_mesh,FEM_GHOST+entity);
	if (nGhosts>0) /* Grab ghost data on the end of the regular data */ 
		FEM_Mesh_data(fem_mesh,FEM_GHOST+entity,FEM_DATA,
			(void *)&data[n*per], 0,nGhosts, FEM_DOUBLE,per);
}
static void mesh_fortran_conn(int mesh,int entity, int *conn, int first,int n, int per)
{
	FEM_Mesh_data(mesh,entity, FEM_CONN, conn, first,n, FEM_INDEX_1, per);
}

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
	mesh_double_data(S,FEM_NODE,data);
}
FORTRAN_AS_C(FEM_SET_NODE_DATA_R,FEM_Set_node_data,fem_set_node_data_r,
	(const double *data), (data)
)

CDECL void FEM_Set_elem_data(int elType,const double *data) {
	mesh_double_data(S,FEM_ELEM+elType,data);
}
FORTRAN_AS_C(FEM_SET_ELEM_DATA_R,FEM_Set_elem_data,fem_set_elem_data_r,
	(int *t,const double *d), (*t,d)
)

// Connectivity
CDECL void FEM_Set_elem_conn(int elType,const int *conn) {
	int entity=FEM_ELEM+elType;
	int n=FEM_Mesh_get_length(S,entity);
	int per=FEM_Mesh_get_width(S,entity,FEM_CONN);
	FEM_Mesh_set_data(S,entity, FEM_CONN, (int *)conn, 0,n, FEM_INDEX_0, per);
}

FDECL void FTN_NAME(FEM_SET_ELEM_CONN_R,fem_set_elem_conn_r)
	(int *elType,const int *conn) 
{
	int entity=FEM_ELEM+*elType;
	int n=FEM_Mesh_get_length(S,entity);
	int per=FEM_Mesh_get_width(S,entity,FEM_CONN);
	mesh_fortran_conn(S,entity, (int *)conn, 0,n, per);
}

// Sparse
CDECL void FEM_Set_sparse(int sid,int nRecords,
  	const int *nodes,int nodesPerRec,
  	const void *data,int dataPerRec,int dataType) 
{
	int entity=FEM_SPARSE+sid;
	FEM_Mesh_set_conn(S,entity,
		(int *)nodes, 0,nRecords, nodesPerRec);
	FEM_Mesh_set_data(S,entity,FEM_DATA,
		(void *)data, 0,nRecords, dataType,dataPerRec); 
}
FDECL void FTN_NAME(FEM_SET_SPARSE,fem_set_sparse)
	(int *sid,int *nRecords,
  	const int *nodes,int *nodesPerRec,
  	const void *data,int *dataPerRec,int *dataType) 
{
	int entity=FEM_SPARSE+*sid;
	int n=*nRecords;
	mesh_fortran_conn(S,entity, (int *)nodes, 0,n, *nodesPerRec);
	FEM_Mesh_set_data(S,entity,FEM_DATA, (void *)data, 0,n, *dataType,*dataPerRec); 
}

CDECL void FEM_Set_sparse_elem(int sid,const int *rec2elem) 
{
	int entity=FEM_SPARSE+sid;
	int n=FEM_Mesh_get_length(S,FEM_SPARSE+sid);
	FEM_Mesh_set_data(S,entity,FEM_SPARSE_ELEM,
		(void *)rec2elem, 0,n, FEM_INDEX_0,2);
}
FDECL void FTN_NAME(FEM_SET_SPARSE_ELEM,fem_set_sparse_elem)
	(int *sid,const int *rec2elem) 
{
	int entity=FEM_SPARSE+*sid;
	int n=FEM_Mesh_get_length(S,entity);
	FEM_Mesh_set_data(S,entity,FEM_SPARSE_ELEM,
		(void *)rec2elem, 0,n, FEM_INDEX_1,2);
}




/***************** Mesh Get ******************/
// Lengths

int mesh_get_ghost_length(int mesh,int entity) {
	return FEM_Mesh_get_length(mesh,entity)+FEM_Mesh_get_length(mesh,FEM_GHOST+entity);
}

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
	mesh_double_data(G,FEM_NODE,data);
}
FORTRAN_AS_C(FEM_GET_NODE_DATA_R,FEM_Get_node_data,fem_get_node_data_r,
	(double *data), (data))

CDECL void FEM_Get_elem_data(int elType,double *data) {
	mesh_double_data(G,FEM_ELEM+elType,data);
}
FORTRAN_AS_C(FEM_GET_ELEM_DATA_R,FEM_Get_elem_data,fem_get_elem_data_r,
	(int *elType,double *data), (*elType,data))


// Connectivity
CDECL void FEM_Get_elem_conn(int elType,int *conn) {
	int fem_mesh=G;
	int entity=FEM_ELEM+elType;
	int n=FEM_Mesh_get_length(fem_mesh,entity);
	int per=FEM_Mesh_get_width(fem_mesh,entity,FEM_CONN);
	FEM_Mesh_conn(fem_mesh,entity,conn, 0,n, per);
	int nGhosts=FEM_Mesh_get_length(fem_mesh,FEM_GHOST+entity);
	if (nGhosts>0) { /* Grab ghost data on the end of the regular data */ 
		int *ghostConn=&conn[n*per];
		FEM_Mesh_conn(fem_mesh,FEM_GHOST+entity,
			ghostConn, 0,nGhosts, per);
		// Renumber indices for ghost nodes (put them after regular nodes):
		int nodeGhostStart=FEM_Mesh_get_length(fem_mesh,FEM_NODE);
		for (int r=0;r<nGhosts;r++) {
			for (int c=0;c<per;c++) {
				int i=ghostConn[r*per+c];
				if (FEM_Is_ghost_index(i))
					ghostConn[r*per+c]=nodeGhostStart+FEM_From_ghost_index(i);
			}
		}
	}
}

FDECL void FTN_NAME(FEM_GET_ELEM_CONN_R, fem_get_elem_conn_r)
	(int *elType,int *conn)
{
	FEM_Get_elem_conn(*elType,conn); //Grab the zero-based version:
	int fem_mesh=G;
	int entity=FEM_ELEM+*elType;
	int n=mesh_get_ghost_length(fem_mesh,entity);
	int per=FEM_Mesh_get_width(fem_mesh,entity,FEM_CONN);
	//Swap from zero to one-based indexing
	for (int i=0;i<n;i++) 
	for (int j=0;j<per;j++)
		conn[i*per+j]++;
}


// Sparse
CDECL int  FEM_Get_sparse_length(int sid) {
	return FEM_Mesh_get_length(G,FEM_SPARSE+sid);
}
FORTRAN_AS_C_RETURN(int, FEM_GET_SPARSE_LENGTH,FEM_Get_sparse_length,fem_get_sparse_length,
	(int *sid), (*sid))

CDECL void FEM_Get_sparse(int sid,int *nodes,void *data) {
	int fem_mesh=G;
	int entity=FEM_SPARSE+sid;
	int nRecords=FEM_Mesh_get_length(fem_mesh,entity);
	int nodesPerRec=FEM_Mesh_get_width(fem_mesh,entity,FEM_CONN);
	int dataPerRec=FEM_Mesh_get_width(fem_mesh,entity,FEM_DATA);
	int dataType=FEM_Mesh_get_datatype(fem_mesh,entity,FEM_DATA);
	FEM_Mesh_conn(fem_mesh,entity,
		nodes, 0,nRecords, nodesPerRec);
	FEM_Mesh_data(fem_mesh,entity,FEM_DATA,
		data, 0,nRecords, dataType,dataPerRec);
}
FDECL void FTN_NAME(FEM_GET_SPARSE,fem_get_sparse)(int *sid,int *nodes,void *data) {
	int fem_mesh=G;
	int entity=FEM_SPARSE+*sid;
	int nRecords=FEM_Mesh_get_length(fem_mesh,entity);
	int nodesPerRec=FEM_Mesh_get_width(fem_mesh,entity,FEM_CONN);
	int dataPerRec=FEM_Mesh_get_width(fem_mesh,entity,FEM_DATA);
	int dataType=FEM_Mesh_get_datatype(fem_mesh,entity,FEM_DATA);
	mesh_fortran_conn(fem_mesh,entity,
		nodes, 0,nRecords, nodesPerRec);
	FEM_Mesh_get_data(fem_mesh,entity,FEM_DATA,
		data, 0,nRecords, dataType,dataPerRec);
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
	int n=l.size();
	for (int i=0;i<n;i++) destSym[i]=l.getSymmetries(i);
	int g=l.getGhost()->size();
	for (int i=0;i<g;i++) destSym[n+i]=l.getGhost()->getSymmetries(i);
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

/*
CDECL void FEM_Update_mesh(FEM_Update_mesh_fn callFn,int userValue,int doWhat) {
	FEM_Mesh_assemble(S, -1, -1, userValue?callFn:0, userValue, doWhat);
}

*/
