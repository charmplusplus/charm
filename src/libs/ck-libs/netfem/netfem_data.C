/*Charm++ Network FEM: internal data format implementation

Orion Sky Lawlor, olawlor@acm.org, 11/2/2001
*/
#include <stdlib.h>
#include <string.h>
#include "charm++.h"
#include "pup.h"
#include "netfem_data.h"
#include "pup_toNetwork4.h"

void *pack(NetFEM_update &u,int *len) 
{
	//Figure out how long our response will be
	int respLen;
	{PUP_toNetwork4_sizer p; u.pup(p); respLen=p.size();}
	//Allocate a buffer and pack our response into it
	void *respBuf=malloc(respLen);
	{PUP_toNetwork4_pack p(respBuf); u.pup(p); }
	*len=respLen;
	return respBuf;
}

void unpack(NetFEM_update &u,const void *buf,int bufLen)
{
	PUP_toNetwork4_unpack p(buf);
	u.pup(p);
}

void CkShortStr::copyFrom(const char *s,int len)
{
	if (len>=maxLen) {
		CkPrintf("[%d] NetFEM passed bad %d-character name '%s'!\n",
			 CkMyPe(),strlen(s),s);
		CkAbort("Name passed to NetFEM is too long!");
	}
	strncpy(store,s,maxLen); //,min(len,maxLen));
	store[len]=0;
	terminate();
}

void NetFEM_doubleField::pup(PUP::er &p) {
	int version=1; p(version);		
	p(isSpatial);
	p(n);
	name.pup(p);
	p(fmt.vec_len);
	if (p.isUnpacking()) { //Unpack data into heap
		fmt=NetFEM_format(fmt.vec_len);
		allocate();
		p(start,n*fmt.vec_len);
	}
	else { //Pack data from wherever it is now
		if ((!isHeapAllocated) &&
		    (fmt.distance!=(int)(fmt.vec_len*sizeof(double))))
			copy(); //Make local copy of data in contiguous format
		p(fmt.forItem(start,0),n*fmt.vec_len);
	}
}
	
//Make a heap-allocated local copy of this data
void NetFEM_doubleField::copy(void) {
	if (isHeapAllocated) 
		return; //Already copied-- nothing to do
	int vl=fmt.vec_len;
	const double *src=start;
	//Allocate and fill heap buffer from user data
	allocate();
	for (int i=0;i<n;i++) {
		const double *isrc=fmt.forItem(src,i);
		for (int j=0;j<vl;j++)
			start[i*vl+j]=isrc[j];
	}
	//Data is now in contiguous form
	fmt=NetFEM_format(vl);
}

NetFEM_item::~NetFEM_item() {
	//Field destructors called automatically
}

void NetFEM_item::add(double *start,const NetFEM_format &fmt,
	const char *name, bool isSpatialVector)
{
	NetFEM_doubleField &f=fields[nFields++];
	if (nFields>maxFields) 
		CkAbort("NetFEM: Added too many scalar or vector fields!\n");
	f.n=n;
	f.start=start;
	f.fmt=fmt;
	f.name=name;
	f.isSpatial=isSpatialVector;
}
void NetFEM_item::copy(void) { //Make heap copies of all referenced data
	for (int i=0;i<nFields;i++)
		fields[i].copy();
}
void NetFEM_item::pup(PUP::er &p) {
	int version=1; p(version);		
	p(n);
	p(nFields);
	for (int i=0;i<nFields;i++)
		fields[i].pup(p);
}

void NetFEM_elems::localCopy(void) {
  //  CkPrintf("localCopy: nodesPer=%d, items=%d, bytesPer=%d\n",nodesPer,getItems(),bytesPer);
	if (isHeapAllocated) return;
	const int *src=conn;
	//Make a canonical heap copy of this data
	allocate();
	for (int i=0;i<getItems();i++)
		for (int j=0;j<nodesPer;j++)
			conn[i*nodesPer+j]=CkShiftPointer(src,i*bytesPer)[j]-idxBase;
	idxBase=0; //Array is now zero-based
}
void NetFEM_elems::copy(void) {
	super::copy();
	localCopy();
}
void NetFEM_elems::pup(PUP::er &p) {
	int version=1; p(version);		
	name.pup(p);
	super::pup(p);
	p(nodesPer); 
	if (p.isUnpacking()) { //Unpack data into heap
		allocate();
		bytesPer=nodesPer*sizeof(int);
	}
	else //Make canonical copy of data
		localCopy();
	p(conn,getItems()*nodesPer);
}

void NetFEM_update::addNodes(NetFEM_nodes *n) {
	if (nodes!=NULL)
		CmiAbort("Can only call NetFEM_Nodes once per begin/end!");
	nodes=n;
	cur=n;//Subsequent vectors/scalars go here
}

void NetFEM_update::addElems(NetFEM_elems *n) {
	if (nElems>=maxElems)
		CmiAbort("Called NetFEM_Elements too many times!");
	elems[nElems++]=n;
	cur=n;//Subsequent vectors/scalars go here
}

NetFEM_item *NetFEM_update::getItem(void) {
	if (cur==NULL) CmiAbort("You must call NetFEM_Nodes or NetFEM_Elements before NetFEM_Scalar or NetFEM_Vector!");
	return cur;
}

void NetFEM_update::pup(PUP::er &p) {
	int version=2; p(version);
	if (version>=2) p(source);
	p(timestep);
	p(dim);
	
	if (nodes==NULL) {
		if (!p.isUnpacking()) CmiAbort("You forgot to call NetFEM_Nodes!");
		else nodes=new NetFEM_nodes;
	}
	nodes->pup(p);
		
	p(nElems);
	for (int i=0;i<nElems;i++) {
		if (p.isUnpacking()) elems[i]=new NetFEM_elems;
		elems[i]->pup(p);
	}
}

