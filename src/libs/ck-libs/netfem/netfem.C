/*Charm++ Network FEM: C implementation file
  Orion Sky Lawlor, olawlor@acm.org, 10/28/2001
  Isaac Dooley, idooley2@uiuc.edu, 2/17/2005
*/
#include "charm++.h"
#include "netfem.h"
#include "netfem.decl.h"
#include "charm-api.h"
#include "pup_toNetwork4.h"
#include "conv-ccs.h"
#include <string.h>
#include <iostream>
#include <iomanip>
#include <sstream> 
#include <string>
#include <cassert>

#include "netfem_data.h"

#include "tcharm.h"
#define NETFEMAPI(routineName) TCHARM_API_TRACE(routineName,"netfem")


//Describes what to do with incoming data
class NetFEM_flavor {
public:	
  int flavor; //What to do with the data
  bool doCopy;
  bool doWritePUP;
  bool doWriteVTK;
  NetFEM_flavor(int flavor_) 
    :flavor(flavor_)
  {
    doCopy=(flavor>=NetFEM_COPY);
    doWritePUP=(flavor==NetFEM_WRITE);     // if doWritePUP is set, we will output old NetFEM files
    doWriteVTK=(flavor==NetFEM_VTK_WRITE); // if doWriteVTK is set, we will output new VTK files
  }
};



//As above, but including what to do with the data
class NetFEM_updatePackage : public NetFEM_update {
  NetFEM_flavor flavor; //What to do with the data
public:
  NetFEM_updatePackage(int src_,int ts_,int dim_,int flavor_)
    :NetFEM_update(src_,ts_,dim_),flavor(flavor_) {}
  
  const NetFEM_flavor &getFlavor(void) const {return flavor;}
  
  /// Return a malloc'd buffer containing our data in (*len) bytes.
  /// The buffer must be free'd when done.
  /// The format used is just the pup to network format for this object
  void *pupMallocBuf(int *len) {
    //Figure out how long our response will be
    int respLen;
    {PUP_toNetwork4_sizer p; pup(p); respLen=p.size();}
    //Allocate a buffer and pack our response into it
    void *respBuf=malloc(respLen);
    {PUP_toNetwork4_pack p(respBuf); pup(p); }
    *len=respLen;
    return respBuf;
  }
  
  
  /// Return a malloc'd buffer containing our data in (*len) bytes.
  /// The buffer must be free'd when done.
  /// The format used is a standard VTK 3.0 Legacy ASCII file for use 
  /// with Paraview or other VTK tools
  void *vtkFileBuf(int *len) {
    std::ostringstream resp;   // we'll build the buffer dynamically, instead of into a fixed length buffer    
    resp << "<?xml version=\"1.0\"?>\n";
    resp << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    resp << "<UnstructuredGrid>\n";

    const NetFEM_nodes &t = getNodes();
    int npoints=t.getItems();
    const NetFEM_elems &el = getElem(0);
    int ncells = el.getItems();
    int dimensions = getDim();

    resp << "<Piece NumberOfPoints=\"" << npoints << "\" NumberOfCells=\"" << ncells << "\">\n";

    // Print the Points

    resp << "<Points>\n";
    resp << "<DataArray type=\"Float64\" NumberOfComponents=\""<< 3 << "\" format=\"ascii\">\n";
    resp << setiosflags(std::ios::showpoint | std::ios::scientific) << std::setprecision(7);;
    for(int i=0; i<npoints; i++){
      const double *d = t.getField(0).getData(i);
      dimensions==1 && resp << d[0] << " 0.0 0.0" << "\n";
      dimensions==2 && resp << d[0] << " " << d[1] << " 0.0\n";
      dimensions==3 && resp << d[0] << " " << d[1] << " " << d[2] << "\n";
    }
    resp << "</DataArray>\n</Points>\n";

    // Print the Cells 
    // Currently we just extract the cells from the first registered set of Elements
	std::ostringstream connString, offsetString, typeString;
	int offset=0;

	for(int e=0;e<getElems();e++){
	  const NetFEM_elems &el = getElem(e);

	  if(el.getCellType() > 0){ // we have a valid VTK cell type, and are therefore cells
		int wid = el.getNodesPer();
		for(int i=0; i<ncells; i++){      
		  for(int j=0;j<wid;j++)
			connString << el.getConnData(i,j) <<  " ";
		}
		
		for(int i=0;i<ncells;i++){
		  offset += wid;
		  offsetString  << offset << " ";
		}
		
		for(int i=0;i<ncells;i++)
		  typeString<< el.getCellType() << " ";
	  }
	}
	
    resp << "<Cells>\n";
    resp << "<DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
	resp << connString.str() << "\n";
    resp << "</DataArray>\n";
    resp << "<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
	resp << offsetString.str() << "\n";
	resp << "</DataArray>\n";
    resp << "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
	resp << typeString.str() << "\n";
	resp << "</DataArray>\n";
    resp << "</Cells>\n";


    // Print Cell Attribute Fields
    resp << "<CellData Scalars=\"";
    int nf=el.getFields();
    for (int fn=0;fn<nf;fn++) {
      const NetFEM_doubleField &f=el.getField(fn);
      int wid=f.getDoublesPerItem();
      int n=f.getItems();
      resp << f.getName();
      if(nf>1 && fn<nf-1)
		resp << ",";
    }
    resp << "\">\n";
    for (int fn=0;fn<nf;fn++) {
      const NetFEM_doubleField &f=el.getField(fn);
      int wid=f.getDoublesPerItem();
      int n=f.getItems();
      resp << "<DataArray type=\"Float64\" Name=\"" << f.getName() << "\" format=\"ascii\" NumberOfComponents=\"" << wid << "\">\n";
      for(int i=0;i<n;i++){
		for(int j=0;j<wid;j++){
		  resp << f.getData(i)[j] << " ";
		}
		resp << "</DataArray>\n";
	  }
	  resp << "</CellData>\n";
	  
	  
	  // Print Point Attribute Fields
	  resp << "<PointData Scalars=\"";
	  nf=t.getFields();
	  for (int fn=1;fn<nf;fn++) {        // Start at 1 since the coordinates are in field 0
		const NetFEM_doubleField &f=t.getField(fn);
		int wid=f.getDoublesPerItem();
		int n=f.getItems();
		resp << f.getName();
		if(nf>1 && fn<nf-1)
		  resp << ",";
	  }
	  resp << "\">\n";
	  for (int fn=1;fn<nf;fn++) {        // Start at 1 since the coordinates are in field 0
		const NetFEM_doubleField &f=t.getField(fn);
		int wid=f.getDoublesPerItem();
		int n=f.getItems();
		resp << "<DataArray type=\"Float64\" Name=\"" << f.getName() << "\" format=\"ascii\" NumberOfComponents=\"" << wid << "\">\n";
		for(int i=0;i<n;i++)
		  for(int j=0;j<wid;j++)
			resp << f.getData(i)[j] << " ";
		resp << "</DataArray>\n";
	  }
	  resp << "</PointData>\n";
	  
	  resp << "</Piece>\n</UnstructuredGrid>\n</VTKFile>\n";
	  
	  // copy from STL string to a malloc'd c string
	  std::string s = resp.str();
	  const char *resp_cstr = s.c_str();
	  *len=strlen(resp_cstr);
	  char *respBuf=(char*)malloc( (*len+1) * sizeof(char));
	  assert(respBuf);
	  strcpy(respBuf,resp_cstr);
	  return respBuf;
	}
  }
  
  /// Return a malloc'd buffer containing our data in (*len) bytes.
  /// The buffer must be free'd when done.
  /// The format used is a standard VTK 3.0 Legacy ASCII file for use 
  /// with Paraview or other VTK tools
  void *vtkIndexFileBuf(int *len, int timestep) {
	std::ostringstream resp;   // we'll build the buffer dynamically, instead of into a fixed buffer    
	
	resp << "<?xml version=\"1.0\"?>\n";
	resp << "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
	resp << "<PUnstructuredGrid GhostLevel=\"0\">\n\n";
	
	const NetFEM_nodes &t = getNodes();
	int npoints=t.getItems();
	const NetFEM_elems &el = getElem(0);
	int ntriangles = el.getItems();
	int dimensions = getDim();
	
	resp << "<PPoints>\n";
	resp << "<PDataArray type=\"Float64\" NumberOfComponents=\""<< 3 << "\" format=\"ascii\"/>\n";
	resp << "</PPoints>\n\n";
	
	resp << "<PCells>\n";
	resp << "<PDataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\"/>\n";
	resp << "<PDataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\"/>\n";
	resp << "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\"/>\n";
	resp << "</PCells>\n\n";
		
	// Print Cell Attribute Fields
	resp << "<PCellData Scalars=\"";
	int nf=el.getFields();
	for (int fn=0;fn<nf;fn++) {
	  const NetFEM_doubleField &f=el.getField(fn);
	  int wid=f.getDoublesPerItem();
	  int n=f.getItems();
	  resp << f.getName();
	  if(nf>1 && fn<nf-1)
		resp << ",";
	}
	resp << "\">\n";
	for (int fn=0;fn<nf;fn++) {
	  const NetFEM_doubleField &f=el.getField(fn);
	  int wid=f.getDoublesPerItem();
	  int n=f.getItems();
	  resp << "<PDataArray type=\"Float64\" Name=\"" << f.getName() << "\" format=\"ascii\" NumberOfComponents=\"" << wid << "\"/>\n";
		  
	}
	resp << "</PCellData>\n\n";
		

	// Print Point Attribute Fields
	resp << "<PPointData Scalars=\"";
	nf=t.getFields();
	for (int fn=1;fn<nf;fn++) {        // Start at 1 since the coordinates are in field 0
	  const NetFEM_doubleField &f=t.getField(fn);
	  int wid=f.getDoublesPerItem();
	  int n=f.getItems();
	  resp << f.getName();
	  if(nf>1 && fn<nf-1)
		resp << ",";
	}
	resp << "\">\n";
	for (int fn=1;fn<nf;fn++) {        // Start at 1 since the coordinates are in field 0
	  const NetFEM_doubleField &f=t.getField(fn);
	  int wid=f.getDoublesPerItem();
	  int n=f.getItems();
	  resp << "<PDataArray type=\"Float64\" Name=\"" << f.getName() << "\" format=\"ascii\" NumberOfComponents=\"" << wid << "\"/>\n";
     
	}
	resp << "</PPointData>\n\n";

	int numchunks;

	for(int i=0;i<getNumChunks();i++)
	  resp << "<Piece Source=\"../chunk-" << i << "/time-" << timestep << ".vtu\" />\n";

	resp << "\n</PUnstructuredGrid>\n</VTKFile>\n";
    
	// copy from STL string to a malloc'd c string
	std::string s = resp.str();
	const char *resp_cstr = s.c_str();
	*len=strlen(resp_cstr);
	char *respBuf=(char*)malloc( (*len+1) * sizeof(char));
	assert(respBuf);
	strcpy(respBuf,resp_cstr);
	return respBuf;
  }
};


//Keeps all stored FEM data for a processor
class NetFEM_state {
  /*STUPID HACK:  Only keep one update around, ever.
	Need to:
	-Index updates by source ID
	-Make copies of updates if asked
	-Write updates to disk if asked
  */
  NetFEM_updatePackage *cur;
public:
  NetFEM_state() {cur=NULL;}
  ~NetFEM_state() {delete cur;}
  void add(NetFEM_updatePackage *u)
  {
	delete cur;
	cur=u;
  }
	
  void getCurrent(char *request,CcsDelayedReply reply)
  {
	//HACK: ignore requested data (should grab timestep, source, etc.)
	int respLen;
	void *respBuf=cur->pupMallocBuf(&respLen);
	//Deliver the response
	CcsSendDelayedReply(reply,respLen,respBuf);
	free(respBuf);
  }
  NetFEM_update *stateForStep(int stepNo) {
	/*HACK: need to look up based on element and step*/
	return cur;
  }
};

CpvDeclare(NetFEM_state *,netfem_state);

static NetFEM_state *getState(void) {
  NetFEM_state *ret=CpvAccess(netfem_state);
  if (ret==NULL) {
	ret=new NetFEM_state;
	CpvAccess(netfem_state)=ret;
  }
  return ret;
}

extern "C" void NetFEM_getCurrent(void *request) 
{
  CcsDelayedReply reply=CcsDelayReply();
  char *reqPtr=(char *)request;
  reqPtr+=CmiMsgHeaderSizeBytes;//Skip over converse header
  getState()->getCurrent(reqPtr,reply);
  CmiFree(request);
}

//Must be called exactly once on each node
void NetFEM_Init(void) 
{
  CpvInitialize(NetFEM_state *,netfem_state);
  CcsRegisterHandler("NetFEM_current",(CmiHandler)NetFEM_getCurrent);
}

#define FTN_STR_DECL const char *strData,int strLen
#define FTN_STR makeCString(strData,strLen)
static CkShortStr makeCString(const char *data,int len)
{
  if (len>32 || len<0)
	CkAbort("F90 string has suspicious length-- is NetFEM ignorant of how your f90 compiler passes strings?");
  return CkShortStr(data,len);
}
typedef int *NetFEMF;
#define N ((NetFEM_updatePackage *)n)
#define NF ((NetFEM_updatePackage *)*nf)

/*----------------------------------------------
  All NetFEM calls must be between a Begin and End pair:*/
CDECL NetFEM NetFEM_Begin(
						  int source,/*Integer ID for the source of this data (need not be sequential)*/
						  int timestep,/*Integer ID for this instant (need not be sequential)*/
						  int dim,/*Number of spatial dimensions (2 or 3)*/
						  int flavor /*What to do with data (point at, write, or copy)*/
						  ) 
{
  NETFEMAPI("NetFEM_Begin");
  //On one processor, this is our only chance to network!
  if (CkNumPes()==1) CthYield();
  //FIXME: actually use source
  return (NetFEM)(new NetFEM_updatePackage(source,timestep,dim,flavor));
}
FDECL NetFEMF FTN_NAME(NETFEM_BEGIN,netfem_begin)(int *s,int *t,int *d,int *f)
{
  return (NetFEMF)NetFEM_Begin(*s,*t,*d,*f);
}



CDECL void NetFEM_End(NetFEM n) { /*Publish these updates*/
  NETFEMAPI("NetFEM_End");
  if (N->getFlavor().doWritePUP)
	{ /* Write data to disk, in file named "NetFEM/<timestep>/<vp>.dat" */
	  char dirName[256], fName[256];
	  const char *baseDir="NetFEM";
	  CmiMkdir(baseDir);
	  sprintf(dirName,"%s/%d",baseDir,N->getTimestep());
	  CmiMkdir(dirName);
	  sprintf(fName,"%s/%d.dat",dirName,N->getSource());
	  FILE *f=fopen(fName,"wb");
	  if (f!=NULL) {
		int respLen;
		void *respBuf;
		respBuf=N->pupMallocBuf(&respLen);
		if (respLen!=(int)fwrite(respBuf,1,respLen,f))
		  CkAbort("Error writing NetFEM output file!");
		free(respBuf);
		fclose(f);
	  }
	  else /*f==NULL*/ {
		CkError("ERROR> Can't create NetFEM output file %s!\n",fName);
		CkAbort("Can't create NetFEM output file!");
	  }
	}

  if (N->getFlavor().doWriteVTK) 
	{ /* Write chunk data to disk, in file named "NetFEM/<timestep>/<vp>.vtp */
	  char dirName[256], fName[256];
	  const char *baseDir="NetFEM-VTK";
	  CmiMkdir(baseDir);
	  sprintf(dirName,"%s/chunk-%d",baseDir,N->getSource());
	  CmiMkdir(dirName);
	  sprintf(fName,"%s/time-%d.vtu",dirName,N->getTimestep());
	  FILE *f=fopen(fName,"w");
	  if (f!=NULL) {
		int respLen;
		void *respBuf;
		respBuf=N->vtkFileBuf(&respLen);
		if (respLen!=(int)fwrite(respBuf,1,respLen,f))
		  CkAbort("Error writing netFEM/VTK output file!");
		free(respBuf);
		fclose(f);
	  }
	  else /*f==NULL*/ {
		CkError("ERROR> Can't create NetFEM/VTK output file %s!\n",fName);
		CkAbort("Can't create NetFEM/VTK output file!");
	  }
   
	  /* One process will write the XML index file that references all the separate chunk data files */
	  if(N->getSource()==0){
		char fName[256], dirName[256];
		const char *baseDir="NetFEM-VTK";
		CmiMkdir(baseDir);
		sprintf(dirName,"%s/ParaView-Data",baseDir);
		CmiMkdir(dirName);
		sprintf(fName,"%s/timestep%d.pvtu",dirName,N->getTimestep());
		FILE *f=fopen(fName,"w");
		if (f!=NULL) {
		  int respLen;
		  void *respBuf;
		  respBuf=N->vtkIndexFileBuf(&respLen,N->getTimestep());
		  if (respLen!=(int)fwrite(respBuf,1,respLen,f))
			CkAbort("Error writing netFEM/VTK output file!");
		  free(respBuf);
		  fclose(f);
		}
		else /*f==NULL*/ {
		  CkError("ERROR> Can't create NetFEM/VTK output file %s!\n",fName);
		  CkAbort("Can't create NetFEM/VTK output file!");
		}
	  }
	}
  getState()->add(N);
}
FDECL void FTN_NAME(NETFEM_END,netfem_end)(NetFEMF nf) {
  NetFEM_End((NetFEM)NF);
}



/*---- Register the number of partitions, (required for VTK file output) ----*/

CDECL void NetFEM_Partitions(NetFEM n,int nPartitions)
{
  NETFEMAPI("NetFEM_Partitions");
  N->setPartitions(nPartitions);
}

FDECL void FTN_NAME(NETFEM_PARTITIONS,netfem_partitions)
  (NetFEMF nf,int *nPartitions,FTN_STR_DECL)
{
  CkShortStr s=FTN_STR;
  NetFEM_Partitions((NetFEM)NF,*nPartitions);
}

CDECL void NetFEM_Partitions_field(NetFEM n,int nPartitions)
{
  NETFEMAPI("NetFEM_Partitions_field");
  N->setPartitions(nPartitions);
}

FDECL void FTN_NAME(NETFEM_PARTITIONS_FIELD,netfem_partitions_field)
  (NetFEMF nf,int *nPartitions,FTN_STR_DECL)
{
  CkShortStr s=FTN_STR;
  NetFEM_Partitions_field((NetFEM)NF,*nPartitions);
}


/*---- Register the locations of the nodes.  (Exactly once, required)
  In 2D, node i has location (loc[2*i+0],loc[2*i+1])
  In 3D, node i has location (loc[3*i+0],loc[3*i+1],loc[3*i+2])
*/

CDECL void NetFEM_Nodes_field(NetFEM n,int nNodes,
							  int init_offset,int distance,
							  const void *loc,const char *name) 
{
  NETFEMAPI("NetFEM_Nodes");
  int d=N->getDim();
  N->addNodes(new NetFEM_nodes(nNodes,NetFEM_format(d,distance),
							   CkShiftPointer((double *)loc,init_offset),name));
}

FDECL void FTN_NAME(NETFEM_NODES_FIELD,netfem_nodes_field)
  (NetFEMF nf,int *nNodes,int *off,int *dist,const void *loc,FTN_STR_DECL)
{
  CkShortStr s=FTN_STR;
  NetFEM_Nodes_field((NetFEM)NF,*nNodes,*off,*dist,loc,s);
}

CDECL void NetFEM_Nodes(NetFEM n,int nNodes,const double *loc,const char *name) {
  NetFEM_Nodes_field(n,nNodes,0,N->getDim()*sizeof(double),loc,name);
}

FDECL void FTN_NAME(NETFEM_NODES,netfem_nodes)
  (NetFEMF nf,int *nNodes,const double *loc,FTN_STR_DECL)
{
  CkShortStr s=FTN_STR;
  NetFEM_Nodes((NetFEM)NF,*nNodes,loc,s);
}

/*----- Register the connectivity of the elements. 
  Element i is adjacent to nodes conn[nodePerEl*i+{0,1,...,nodePerEl-1}]
*/

CDECL void NetFEM_Elements_field(NetFEM n,int nEl,int nodePerEl,
								 int initOffset,int bytePerEl,int idxBase,
								 const void *conn,const char *name)
{
  NETFEMAPI("NetFEM_Elements_field");
  N->addElems(new NetFEM_elems(nEl,nodePerEl,bytePerEl,idxBase,
							   CkShiftPointer((int *)conn,initOffset),name,0)) ;
}

FDECL void FTN_NAME(NETFEM_ELEMENTS_FIELD,netfem_elements_field)
  (NetFEMF nf,int *nEl,int *nodePer,
   int *initOff,int *bytePer,int *idxBase,
   const void *conn,FTN_STR_DECL)
{
  CkShortStr s=FTN_STR;
  NetFEM_Elements_field((NetFEM)NF,*nEl,*nodePer,*initOff,*bytePer,*idxBase,conn,s);
}


CDECL void NetFEM_VTK_Elements_field(NetFEM n,int nEl,int nodePerEl,
									 int initOffset,int bytePerEl,int idxBase,
									 const void *conn,const char *name, int vtk_cell_type)
{
  NETFEMAPI("NetFEM_VTK_Elements_field");
  N->addElems(new NetFEM_elems(nEl,nodePerEl,bytePerEl,
							   idxBase,CkShiftPointer((int *)conn,initOffset),name,vtk_cell_type));
}

FDECL void FTN_NAME(NETFEM_VTK_ELEMENTS_FIELD,netfem_vtk_elements_field)
  (NetFEMF nf,int *nEl,int *nodePer,
   int *initOff,int *bytePer,int *idxBase,
   const void *conn,FTN_STR_DECL, int vtk_cell_type)
{
  CkShortStr s=FTN_STR;
  NetFEM_VTK_Elements_field((NetFEM)NF,*nEl,*nodePer,*initOff,*bytePer,*idxBase,conn,s,vtk_cell_type);
}


CDECL void NetFEM_Elements(NetFEM n,int nEl,int nodePerEl,const int *conn,const char *name)
{
  NetFEM_Elements_field(n,nEl,nodePerEl,0,sizeof(int)*nodePerEl,0,conn,name);
}

FDECL void FTN_NAME(NETFEM_ELEMENTS,netfem_elements)
  (NetFEMF nf,int *nEl,int *nodePerEl,const int *conn,FTN_STR_DECL)
{
  CkShortStr s=FTN_STR;
  NetFEM_Elements_field((NetFEM)NF,*nEl,*nodePerEl,0,sizeof(int)* *nodePerEl,1,conn,s);
}


CDECL void NetFEM_VTK_Elements(NetFEM n,int nEl,int nodePerEl,const int *conn,const char *name, int cell_type)
{
  NetFEM_VTK_Elements_field(n,nEl,nodePerEl,0,sizeof(int)*nodePerEl,0,conn,name,cell_type);
}

FDECL void FTN_NAME(NETFEM_VTK_ELEMENTS,netfem_vtk_elements)
  (NetFEMF nf,int *nEl,int *nodePerEl,const int *conn,FTN_STR_DECL,int cell_type)
{
  CkShortStr s=FTN_STR;
  NetFEM_VTK_Elements_field((NetFEM)NF,*nEl,*nodePerEl,0,sizeof(int)* *nodePerEl,1,conn,s,cell_type);
}



/*--------------------------------------------------
  Associate a spatial vector (e.g., displacement, velocity, accelleration)
  with each of the previous objects (nodes or elements).
*/
CDECL void NetFEM_Vector_field(NetFEM n,const void *start,
							   int init_offset,int distance,
							   const char *name)
{
  NETFEMAPI("NetFEM_Vector_field");
  NetFEM_format fmt(N->getDim(),distance);
  N->getItem()->add(CkShiftPointer((double *)start,init_offset),fmt,name,true);
}
FDECL void FTN_NAME(NETFEM_VECTOR_FIELD,netfem_vector_field)
  (NetFEMF nf,const double *start,int *init_offset,int *distance,FTN_STR_DECL)
{
  NETFEMAPI("NetFEM_vector_field");
  CkShortStr s=FTN_STR;
  NetFEM_Vector_field((NetFEM)NF,start,*init_offset,*distance,s);
}

/*Simpler version of the above if your data is packed as
  data[item*3+{0,1,2}].
*/
CDECL void NetFEM_Vector(NetFEM n,const double *data,const char *name)
{
  NetFEM_Vector_field(n,data,0,sizeof(double)*N->getDim(),name);
}
FDECL void FTN_NAME(NETFEM_VECTOR,netfem_vector)
  (NetFEMF nf,const double *data,FTN_STR_DECL)
{
  CkShortStr s=FTN_STR;
  NetFEM_Vector((NetFEM)NF,data,s);
}

/*--------------------------------------------------
  Associate a scalar (e.g., stress, temperature, pressure, damage)
  with each of the previous objects (nodes or elements).
*/
CDECL void NetFEM_Scalar_field(NetFEM n,const void *start,
							   int vec_len,int init_offset,int distance,
							   const char *name)
{
  NETFEMAPI("NetFEM_Scalar_field");
  NetFEM_format fmt(vec_len,distance);
  N->getItem()->add(CkShiftPointer((double *)start,init_offset),fmt,name,false);
}

FDECL void FTN_NAME(NETFEM_SCALAR_FIELD,netfem_scalar_field)
  (NetFEMF nf,const double *start,int *veclen,int *init_offset,
   int *distance,FTN_STR_DECL)
{
  NETFEMAPI("NetFEM_scalar_field");
  CkShortStr s=FTN_STR;
  NetFEM_Scalar_field((NetFEM)NF,start,*veclen,*init_offset,*distance,s);
}


/*Simpler version of above for contiguous double-precision data*/
CDECL void NetFEM_Scalar(NetFEM n,const double *start,int doublePer,
						 const char *name)
{
  NetFEM_Scalar_field(n,start,doublePer,0,sizeof(double)*doublePer,name);
}
FDECL void FTN_NAME(NETFEM_SCALAR,netfem_scalar)
  (NetFEMF nf,const double *start,int *veclen,FTN_STR_DECL)
{
  CkShortStr s=FTN_STR;
  NetFEM_Scalar((NetFEM)NF,start,*veclen,s);
}


#include "netfem.def.h"


