/*Charm++ Network FEM: internal data format

Orion Sky Lawlor, olawlor@acm.org, 11/2/2001
*/
#ifndef __OSL_NETFEM_DATA_H
#define __OSL_NETFEM_DATA_H

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
		void copyFrom(const char *s,int len) {
			if (len>=maxLen) {
				CkPrintf("[%d] NetFEM passed bad %d-character name '%s'!\n",
					 CkMyPe(),strlen(s),s);
				CkAbort("Name passed to NetFEM is too long!");
			}
			strncpy(store,s,maxLen); //,min(len,maxLen));
			store[len]=0;
			terminate();
		}
		operator const char *() const {return store;}
		
		void pup(PUP::er &p) {
			int len=strlen(store)+1;
			p(len);
			p(store,len);
			terminate();
		}
	};



//Any list of items we can associate user data with.
class NetFEM_item {
public:
	//Describes layout of doubles in memory
	class format {
	public:
		int vec_len;//Doubles per source item
		int init_offset; //Bytes from start to first double of first item
		int distance;//Bytes from first item to second item
		format() {}
		format(int len) 
			:vec_len(len), init_offset(0), distance(len*sizeof(double))
			{}
		format(int len,int off,int dist)
			:vec_len(len), init_offset(off), distance(dist) 
			{}
		
		//Return the start of this item's data list
		double *forItem(double *start,int item) const {
			char *s=(char *)start;
			return (double *)(s+init_offset+item*distance);
		}
		const double *forItem(const double *start,int item) const {
			return forItem((double *)start,item);
		}
		
		//Return the length of each item's data list
		int nDoubles(void) const {return vec_len;}
	};

public:
	class doubleField {
		friend class NetFEM_item; //<- friend so he can set our fields directly
      		int n; //Number of items
		double *start;//Beginning of data
		format fmt; //Layout of data
		bool isSpatial;//Should be interpreted as x,y,z
		CkShortStr name;
		bool isHeapAllocated;//Is data a heap copy (true) or user data (false)
		
		void allocate(void) {
			start=new double[n*fmt.vec_len];
			isHeapAllocated=true;
		}
	public:
		doubleField() {
			isHeapAllocated=false;
		}
		~doubleField() {
			if (isHeapAllocated) {
				delete[] start;
			}
		}

		const char *getName(void) const {return name;}
		bool getSpatial(void) const {return isSpatial;}
		int getItems(void) const {return n;}
		int getDoublesPerItem(void) const {return fmt.vec_len;}
		const double *getData(int itemNo) const {return fmt.forItem(start,itemNo);}
		
		void pup(PUP::er &p) {
			int version=1; p(version);		
			p(isSpatial);
			p(n);
			name.pup(p);
			p(fmt.vec_len);
			if (p.isUnpacking()) { //Unpack data into heap
				fmt=format(fmt.vec_len);
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
		void copy(void) {
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
			fmt=format(vl);
		}
	};
private:
	int n; //Number of source items
	enum {maxFields=8};
	doubleField fields[maxFields];
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

	virtual ~NetFEM_item() {
		//Field destructors called automatically
	}

	int getItems(void) const {return n;}
	int getFields(void) const {return nFields;}
	const doubleField &getField(int fieldNo) const {return fields[fieldNo];}

	void add(double *start,const format &fmt,
		const CkShortStr &name, bool isSpatialVector)
	{
		doubleField &f=fields[nFields++];
		if (nFields>maxFields) 
			CkAbort("NetFEM: Added too many scalar or vector fields!\n");
		f.n=n;
		f.start=start;
		f.fmt=fmt;
		f.name=name;
		f.isSpatial=isSpatialVector;
	}
	virtual void copy(void) { //Make heap copies of all referenced data
		for (int i=0;i<nFields;i++)
			fields[i].copy();
	}
	virtual void pup(PUP::er &p) {
		int version=1; p(version);		
		p(n);
		p(nFields);
		for (int i=0;i<nFields;i++)
			fields[i].pup(p);
	}
};

class NetFEM_nodes : public NetFEM_item {
public:
	NetFEM_nodes() {}

	NetFEM_nodes(int nNode,int dim,double *coord,const CkShortStr &name) 
		:NetFEM_item(nNode)
	{
		//HACK: The first field of the nodes are the node coordinates
		add(coord,format(dim),name,true);
	}
};


class NetFEM_elems : public NetFEM_item {
	typedef NetFEM_item super;

	CkShortStr name; //Name of element type (e.g. "Volumetric Tets"; "Surface Triangles")
	int nodesPer,idxBase;
	int *conn;
	bool isHeapAllocated; //Conn array on heap(true) or in user area(false)

	void allocate(void) {
		isHeapAllocated=true;
		conn=new int[getItems()*nodesPer];
	}
	void localCopy(void) {
		if (isHeapAllocated) return;
		const int *src=conn;
		//Make a canonical heap copy of this data
		allocate();
		for (int i=0;i<getItems();i++)
			for (int j=0;j<nodesPer;j++)
				conn[i*nodesPer+j]=src[i*nodesPer+j]-idxBase;
		idxBase=0; //Array is now zero-based
	}
public:
	NetFEM_elems() {
		nodesPer=0;
		idxBase=0;
		conn=NULL;
		isHeapAllocated=false;
	}

	NetFEM_elems(int nEl,int nodesPerEl,int *conn_,int idxBase_,
		     const CkShortStr &name_) 
		:super(nEl),nodesPer(nodesPerEl),idxBase(idxBase_),
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

	virtual void copy(void) {
		super::copy();
		localCopy();
	}
	
	virtual void pup(PUP::er &p) {
		int version=1; p(version);		
		name.pup(p);
		super::pup(p);
		p(nodesPer); 
		if (p.isUnpacking()) //Unpack data into heap
			allocate();
		else //Make canonical copy of data
			localCopy();
		p(conn,getItems()*nodesPer);
	}
};


//The local data for one update of the FEM mesh
class NetFEM_update {
	int timestep,dim;

	NetFEM_nodes *nodes;
	enum {maxElems=20};
	NetFEM_elems *elems[maxElems];
	int nElems;
	NetFEM_item *cur;
public:
	NetFEM_update(int dim_,int ts)
		:timestep(ts),dim(dim_)
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
	int getChunk(void) const {return -1;}
	int getTimestep(void) const {return timestep;}
	int getDim(void) const {return dim;}
	int getElems(void) const {return nElems;}
	const NetFEM_nodes &getNodes(void) const {return *nodes;}
	const NetFEM_elems &getElem(int i) const {return *elems[i];}

	void addNodes(NetFEM_nodes *n) {
		if (nodes!=NULL)
			CmiAbort("Can only call NetFEM_Nodes once per begin/end!");
		nodes=n;
		cur=n;//Subsequent vectors/scalars go here
	}

	void addElems(NetFEM_elems *n) {
		if (nElems>=maxElems)
			CmiAbort("Called NetFEM_Elements too many times!");
		elems[nElems++]=n;
		cur=n;//Subsequent vectors/scalars go here
	}

	NetFEM_item *getItem(void) {
		if (cur==NULL) CmiAbort("You must call NetFEM_Nodes or NetFEM_Elements before NetFEM_Scalar or NetFEM_Vector!");
		return cur;
	}

	void pup(PUP::er &p) {
		int version=1; p(version);
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
};

#endif

