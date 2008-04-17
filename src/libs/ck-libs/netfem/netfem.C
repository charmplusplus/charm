/*Charm++ Network FEM: C implementation file

Orion Sky Lawlor, olawlor@acm.org, 10/28/2001
*/
#include "charm++.h"
#include "netfem.h"
#include "netfem.decl.h"
#include "charm-api.h"
#include "pup_toNetwork4.h"
#include "conv-ccs.h"
#include <string.h>

#include "netfem_data.h"

#include "pup_toNetwork4.h"

#include "tcharm.h"
#define NETFEMAPI(routineName) TCHARM_API_TRACE(routineName,"netfem")

//Describes what to do with incoming data
class NetFEM_flavor {
public:	
	int flavor; //What to do with the data
	bool doCopy;
	bool doWrite;
	NetFEM_flavor(int flavor_) 
		:flavor(flavor_)
	{
		doCopy=(flavor>=NetFEM_COPY);
		doWrite=(flavor==NetFEM_WRITE);		
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
	///  The buffer must be free'd when done.
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
	~NetFEM_state() {
		delete cur;
	}
	void add(NetFEM_updatePackage *u)
	{
		delete cur;
		cur=u;
	}
	
	void getCurrent(char *request,CcsDelayedReply reply)
	{
		//HACK: ignore requested data (should grab timestep, source, etc.)
		int respLen = 0;
		void *respBuf=cur?cur->pupMallocBuf(&respLen):NULL;
		//Deliver the response
		CcsSendDelayedReply(reply,respLen,respBuf);
		if (respBuf != NULL) free(respBuf);
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
	CpvAccess(netfem_state) = NULL;
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
	if (N->getFlavor().doWrite) 
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
			void *respBuf=N->pupMallocBuf(&respLen);
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
	getState()->add(N);
}
FDECL void FTN_NAME(NETFEM_END,netfem_end)(NetFEMF nf) {
	NetFEM_End((NetFEM)NF);
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
	N->addElems(new NetFEM_elems(nEl,nodePerEl,bytePerEl,
		idxBase,CkShiftPointer((int *)conn,initOffset),name));
}

FDECL void FTN_NAME(NETFEM_ELEMENTS_FIELD,netfem_elements_field)
	(NetFEMF nf,int *nEl,int *nodePer,
	int *initOff,int *bytePer,int *idxBase,
	const void *conn,FTN_STR_DECL)
{
	CkShortStr s=FTN_STR;
	NetFEM_Elements_field((NetFEM)NF,*nEl,*nodePer,*initOff,*bytePer,*idxBase,conn,s);
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

