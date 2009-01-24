/**
 * Connect mesh to FEM framework for crack propagation code.
 */
#include <stddef.h>
#include "crack.h"


/// Send/recv this element's material type and element connectivity:
void sendRecvElement(int fem_mesh,int fem_entity,
	int matOffset,int connOffset,int connLen,
	int elSize,int nEl, void *elData)
{
  // Element connectivity:
  IDXL_Layout_t eConn=IDXL_Layout_offset(IDXL_INDEX_0,connLen, //connLen indices
	connOffset, elSize, 0);
  FEM_Mesh_data_layout(fem_mesh,fem_entity,FEM_CONN,elData,
  	0,nEl, eConn);
  IDXL_Layout_destroy(eConn);
  
  // Element material type:
  IDXL_Layout_t eMat=IDXL_Layout_offset(IDXL_INT,1, //1 int
	matOffset, elSize, 0);
  FEM_Mesh_data_layout(fem_mesh,fem_entity,FEM_DATA+0,elData,
  	0,nEl, eMat);
  IDXL_Layout_destroy(eMat);
}

/// Send/recv this node field
void sendRecvNode(MeshData *mesh,int fem_mesh,int attrib,
	int offset,int datatype,int dataLen)
{
  IDXL_Layout_t l=IDXL_Layout_offset(datatype,dataLen,
	offset, sizeof(Node), 0);
  FEM_Mesh_data_layout(fem_mesh,FEM_NODE,attrib,mesh->nodes,
  	0,mesh->nn, l);
  IDXL_Layout_destroy(l);
}

/// Send/recv this mesh's data to/from the FEM framework:
void sendRecvMesh(MeshData *mesh,int fem_mesh)
{
// Nodes:
  sendRecvNode(mesh,fem_mesh,FEM_DATA+10,
  	offsetof(Node,pos), FEM_DOUBLE, 2);
  
  // FIXME: should a list of boundary nodes, instead of
  // (normally zero!) boundary data for every node
  sendRecvNode(mesh,fem_mesh,FEM_DATA+20,
  	offsetof(Node,r), FEM_DOUBLE, 2);
  sendRecvNode(mesh,fem_mesh,FEM_DATA+30,
  	offsetof(Node,isbnd), FEM_BYTE, 3);
  
// Volumetric element:
  sendRecvElement(fem_mesh,FEM_ELEM+0,
  	offsetof(Vol,material),offsetof(Vol,conn),6,
	sizeof(Vol),mesh->ne,mesh->vols);
  
// Cohesive element:
  sendRecvElement(fem_mesh,FEM_ELEM+1,
  	offsetof(Coh,material),offsetof(Coh,conn),6,
	sizeof(Coh),mesh->nc,mesh->cohs);
  
}

/// Send this mesh out to the FEM framework:
void sendMesh(MeshData *mesh,int fem_mesh)
{
   sendRecvMesh(mesh,fem_mesh);
}

/// Recv this mesh from the FEM framework:
void recvMesh(MeshData *mesh,int fem_mesh)
{
  // Allocate storage for the mesh
  mesh->nn=FEM_Mesh_get_length(fem_mesh,FEM_NODE);
  mesh->ne=FEM_Mesh_get_length(fem_mesh,FEM_ELEM+0);
  mesh->nc=FEM_Mesh_get_length(fem_mesh,FEM_ELEM+1);
  mesh->nodes=new Node[mesh->nn];
  mesh->vols=new Vol[mesh->ne];
  mesh->cohs=new Coh[mesh->nc];
   
  // Grab the data from the FEM framework
  sendRecvMesh(mesh,fem_mesh);
  
  // Initialize the mesh, which fills out the fields we haven't copied
  setupMesh(mesh);
  
  // Sum up element masses (nodes[i].xM) from other processors:
  int massfield = IDXL_Layout_offset(FEM_DOUBLE, 2, 
  	offsetof(Node, xM), sizeof(Node), 0);
  int nodeComm = FEM_Comm_shared(fem_mesh,FEM_NODE);
  IDXL_Comm_sendsum(0,nodeComm,massfield, mesh->nodes);
  IDXL_Layout_destroy(massfield);
}

extern "C" void pupMesh(pup_er p,MeshData *mesh)
{
  pup_int(p,&mesh->nn);
  pup_int(p,&mesh->ne);
  pup_int(p,&mesh->nc);
  pup_int(p,&mesh->numBound);
  pup_int(p,&mesh->nImp);
  if(pup_isUnpacking(p)) {
    mesh->nodes=new Node[mesh->nn];
    mesh->vols=new Vol[mesh->ne];
    mesh->cohs=new Coh[mesh->nc];
  }
  //EVIL: instead of packing each node/vol/coh field separately,
  //  just pack them as flat runs of bytes.
  pup_bytes(p, (void*)mesh->nodes, mesh->nn*sizeof(Node));
  pup_bytes(p, (void*)mesh->vols, mesh->ne*sizeof(Vol));
  pup_bytes(p, (void*)mesh->cohs, mesh->nc*sizeof(Coh));
  
  if (pup_isDeleting(p)) {
    deleteMesh(mesh);
  }
}


