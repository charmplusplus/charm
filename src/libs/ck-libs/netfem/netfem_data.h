/*Charm++ Network FEM: internal data format

Orion Sky Lawlor, olawlor@acm.org, 11/2/2001
Isaac Dooley 3/15/05
*/
#ifndef __OSL_NETFEM_DATA_H
#define __OSL_NETFEM_DATA_H

#include "charm++.h"
#include "pup.h"

	class CkShortStr {
		enum {maxLen=32};
		char store[maxLen];
		void terminate(void) {
			store[maxLen-1]=0;//Bit of paranoia
		}
	public:
		CkShortStr() {} //store[0]=0;}
		CkShortStr(const char *s) {copyFrom(s);}
		CkShortStr(const char *s,int l) {copyFrom(s,l);}
		
		void operator=(const char *s) { copyFrom(s);}
		void copyFrom(const char *s) {copyFrom(s,strlen(s));}
		void copyFrom(const char *s,int len);
		operator const char *() const {return store;}
		
		void pup(PUP::er &p) {
			int len=strlen(store)+1;
			p(len);
			p(store,len);
			terminate();
		}
	};

template <class T>
inline T *CkShiftPointer(T *p,int bytesToShift) {
	return (T *)(bytesToShift+(char *)p);
}

//Describes layout of doubles in memory
class NetFEM_format {
public:
	int vec_len;//Doubles per source item
	int distance;//Bytes from first item to second item
	NetFEM_format() {}
	NetFEM_format(int len) 
		:vec_len(len), distance(len*sizeof(double))
		{}
	NetFEM_format(int len,int dist)
		:vec_len(len), distance(dist) 
		{}
	
	//Return the start of this item's data list
	double *forItem(double *start,int item) const {
		char *s=(char *)start;
		return (double *)(s+item*distance);
	}
	const double *forItem(const double *start,int item) const {
		return forItem((double *)start,item);
	}
		
	//Return the length of each item's data list
	int nDoubles(void) const {return vec_len;}
};

//An array of information describing an item
class NetFEM_doubleField {
	friend class NetFEM_item; //<- friend so he can set our fields directly
	int n; //Number of items
	double *start;//Beginning of data
	NetFEM_format fmt; //Layout of data
	bool isSpatial;//Should be interpreted as x,y,z
	CkShortStr name;
	bool isHeapAllocated;//Is data a heap copy (true) or user data (false)
		
	void allocate(void) {
		start=new double[n*fmt.vec_len];
		isHeapAllocated=true;
	}
public:
	NetFEM_doubleField() {
		isHeapAllocated=false;
	}
	~NetFEM_doubleField() {
		if (isHeapAllocated) {
			delete[] start;
		}
	}

	const char *getName(void) const {return name;}
	bool getSpatial(void) const {return isSpatial;}
	int getItems(void) const {return n;}
	int getDoublesPerItem(void) const {return fmt.vec_len;}
	int getStride(void) const {return fmt.vec_len;}
	const double *getData(int itemNo) const {return fmt.forItem(start,itemNo);}
	
	void pup(PUP::er &p);
	
	//Make a heap-allocated local copy of this data
	void copy(void);
};

//Any list of items we can associate user data with.
class NetFEM_item {
private:
	int n; //Number of source items
	enum {maxFields=8};
	NetFEM_doubleField fields[maxFields];
	int nFields;
public:
	NetFEM_item() {
		n=0;
		nFields=0;
	}
	NetFEM_item(int nItems) {
		n=nItems;
		nFields=0;
	}

	virtual ~NetFEM_item();

	int getItems(void) const {return n;}
	int getFields(void) const {return nFields;}
	const NetFEM_doubleField &getField(int fieldNo) const 
		{return fields[fieldNo];}

	void add(double *start,const NetFEM_format &fmt,
		const char *name, bool isSpatialVector);
	virtual void copy(void); 
	virtual void pup(PUP::er &p);
};

class NetFEM_nodes : public NetFEM_item {
public:
	NetFEM_nodes() {}

	const NetFEM_doubleField &getCoord(void) const {return getField(0);}

	NetFEM_nodes(int nNode,const NetFEM_format &fmt,double *coord,const char *name) 
		:NetFEM_item(nNode)
	{
		//HACK: The first field of the nodes are the node coordinates
		add(coord,fmt,name,true);
	}
};


class NetFEM_elems : public NetFEM_item {
	typedef NetFEM_item super;

//Name of element type (e.g. "Volumetric Tets"; "Surface Triangles")
	CkShortStr name; 
	int nodesPer,bytesPer,idxBase;
	int *conn;
	bool isHeapAllocated; //Conn array on heap(true) or in user area(false)

	void allocate(void) {
		isHeapAllocated=true;
		conn=new int[getItems()*nodesPer];
	}
	void localCopy(void);
public:
	NetFEM_elems() {
		nodesPer=0;
		bytesPer=0;
		idxBase=0;
		conn=NULL;
		isHeapAllocated=false;
	}

	NetFEM_elems(int nEl,int nodesPerEl,int bytesPerEl,
			int idxBase_,int *conn_,const char *name_) 
		:super(nEl),nodesPer(nodesPerEl),bytesPer(bytesPerEl),idxBase(idxBase_),
		 conn(conn_),isHeapAllocated(false)
	{
		name=name_;
	}
	~NetFEM_elems() {
		if (isHeapAllocated) delete[] conn;
	}
	int getNodesPer(void) const {return nodesPer;}
	const int *getNodes(int elementNo) const {return &conn[elementNo*nodesPer];}
	const char *getName(void) const {return name;}

	// To make accessing conn easier for Isaac's Paraview Converter:
    // get the element connectivity information. dimension is zero indexed
    // For a triangle, you will have dimensions 0,1,2 available
    int getConnData(int elementNo, int dimension) const {return conn[elementNo*bytesPer/sizeof(int)+dimension];}
    int getConnWidth() const {return bytesPer/sizeof(int);}

	virtual void copy(void);
	virtual void pup(PUP::er &p);
};


//The local data for one update of the FEM mesh
class NetFEM_update {
	int source,timestep,dim;

	NetFEM_nodes *nodes;
	enum {maxElems=20};
	NetFEM_elems *elems[maxElems];
	int nElems;
	NetFEM_item *cur;
public:
	NetFEM_update(int src_,int ts,int dim_)
		:source(src_),timestep(ts),dim(dim_)
	{
		nodes=NULL;
		for (int i=0;i<maxElems;i++) elems[i]=NULL;
		nElems=0;
		cur=NULL;
	}
	~NetFEM_update() {
		delete nodes;
		for (int i=0;i<nElems;i++)
			delete elems[i];
	}
	int getSource(void) const {return source;}
	int getTimestep(void) const {return timestep;}
	int getDim(void) const {return dim;}
	int getElems(void) const {return nElems;}
	const NetFEM_nodes &getNodes(void) const {return *nodes;}
	const NetFEM_elems &getElem(int i) const {return *elems[i];}

	void addNodes(NetFEM_nodes *n);
	void addElems(NetFEM_elems *n);

	NetFEM_item *getItem(void);
	void pup(PUP::er &p);
};

/// Return a malloc'd buffer containing our data in (*len) bytes.
///  The buffer must be free'd when done.
void *pack(NetFEM_update &u,int *len);
void unpack(NetFEM_update &u,const void *buf,int bufLen);

#endif

